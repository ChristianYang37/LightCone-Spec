"""Request-slot adapter bank (spec 8.3).

Fixed-address device storage per request slot: canonical FP32 active/staging
masters plus a model-dtype forward bank.  Publishing quantizes through the
forward bank and dequantizes back into the canonical masters, so training and
serving bind the exact same parameter point without reallocating graph-visible
storage.  Slot reuse increments the epoch; a mismatch between a request's
epoch and its slot's epoch is an ABA violation and fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from lightcone_spec.exit_codes import ExactnessViolation


# Bump these whenever the corresponding deterministic sizing equations change.
# They are persisted in the memory-calibration identity so a warmup peak can
# never be reused across a different allocation contract.
ADAPTATION_MEMORY_ESTIMATOR_SCHEMA_VERSION = 2
DFLASH_SUPERVISION_FANOUT_SCHEMA_VERSION = 2


def resolve_adapter_row_capacity(
    *,
    max_running_requests: int,
    cuda_graph_max_bs_decode: int | None = None,
    cuda_graph_batch_sizes: tuple[int, ...] | list[int] | None = None,
) -> int:
    """Resolve the one row capacity shared by preflight and live workers.

    Graph replay can address any captured row, while eager execution can admit
    up to ``max_running_requests``.  The bank therefore has to cover both.  A
    caller that has not resolved SGLang's graph defaults yet should pass no
    graph values; LightCone then deliberately caps decode capture at the
    request limit instead of inheriting SGLang's hardware-wide default.
    """

    request_capacity = int(max_running_requests)
    if request_capacity <= 0:
        raise ValueError("max_running_requests must be positive")
    values = [request_capacity]
    if cuda_graph_max_bs_decode is not None:
        graph_max = int(cuda_graph_max_bs_decode)
        if graph_max <= 0:
            raise ValueError("cuda_graph_max_bs_decode must be positive")
        values.append(graph_max)
    for value in cuda_graph_batch_sizes or ():
        batch_size = int(value)
        if batch_size <= 0:
            raise ValueError("cuda graph batch sizes must be positive")
        values.append(batch_size)
    return max(values)


class AdaptationCapacityError(RuntimeError):
    """Bounded adaptation capacity is exhausted; base DSpark may continue."""


@dataclass(frozen=True)
class AdaptationMemoryLedger:
    """Deterministic HBM ledger used before SGLang sizes its KV pool.

    The named fields are deliberately aligned with actual allocation owners.
    Adaptation state is resident and non-evictable; only KV capacity/admission
    may shrink under pressure.  No field implies a CPU-offload fallback.
    """

    fixed_bytes: int
    transient_bytes: int
    calibrated_bytes: int
    reserve_bytes: int
    num_slots: int
    max_in_flight: int
    num_params: int
    adapter_row_capacity: int
    active_staging_bytes: int = 0
    forward_active_bytes: int = 0
    forward_candidate_scratch_bytes: int = 0
    fp32_master_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0
    candidate_scratch_bytes: int = 0
    graph_row_buffer_bytes: int = 0
    trace_bytes: int = 0
    supervision_fanout_bytes: int = 0
    activation_reserve_bytes: int = 0
    artifact_bytes: int = 0

    @property
    def reserve_mb(self) -> int:
        return math.ceil(self.reserve_bytes / (1 << 20))

    def category_bytes(self) -> dict[str, int]:
        """Return mutually exclusive bytes that sum to the KV-visible charge.

        Resident tensors are charged implicitly because they exist before KV
        sizing. ``runtime_headroom`` is the one explicit subtraction from the
        KV budget.  Trace retention is part of the transient peak used to size
        that headroom, so reporting it as another top-level category would
        double count it.
        """
        return {
            "active_staging": self.active_staging_bytes,
            "forward_active": self.forward_active_bytes,
            "forward_candidate_scratch": self.forward_candidate_scratch_bytes,
            "fp32_master": self.fp32_master_bytes,
            "gradient": self.gradient_bytes,
            "optimizer": self.optimizer_bytes,
            "candidate_scratch": self.candidate_scratch_bytes,
            "graph_row_buffers": self.graph_row_buffer_bytes,
            "artifacts": self.artifact_bytes,
            "runtime_headroom": self.reserve_bytes,
        }

    def headroom_breakdown(self) -> dict[str, int]:
        """Diagnostics used to derive ``runtime_headroom`` without recharging it.

        ``trace_within_transient`` is intentionally descriptive rather than an
        additive category: those bytes are already included in
        ``transient_peak``.  The final margin is always non-negative because
        the configured safety factor is at least one and calibration is a
        floor on the reserve.
        """
        return {
            "reserved_total": self.reserve_bytes,
            "transient_peak": self.transient_bytes,
            "trace_within_transient": self.trace_bytes,
            "supervision_fanout_within_transient": self.supervision_fanout_bytes,
            "calibration_floor": self.calibrated_bytes,
            "safety_or_calibration_margin": max(
                self.reserve_bytes - self.transient_bytes, 0
            ),
        }


def estimate_dflash_supervision_fanout_bytes(
    *,
    batch_capacity: int,
    active_capacity: int,
    vocab_size: int,
    hidden_size: int,
    draft_depth: int,
    forward_dtype_bytes: int,
    output_residual: bool,
    stochastic: bool,
    tensor_parallel_size: int = 1,
    trace_capture: bool = False,
) -> int:
    """Return DFlash vectorized-batch fan-out beyond the first candidate.

    Online candidates retain a compact source-bound snapshot: corrected and
    target scores stay in model dtype, raw scores are omitted, and the common
    batch backward materializes FP32 target/proposal probabilities plus one
    native-dtype STE gradient.  The generic workspace already charges the
    first row, so this function charges every additional active row.  Full raw
    and corrected proposal outputs remain live for the scheduler batch until
    acceptance completes and are accounted separately.  Instrumented trace
    runs additionally own a reconstructable raw/target/source signal per
    admitted request; performance runs leave that branch at zero.
    """

    values = {
        "batch_capacity": batch_capacity,
        "active_capacity": active_capacity,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "draft_depth": draft_depth,
        "forward_dtype_bytes": forward_dtype_bytes,
        "tensor_parallel_size": tensor_parallel_size,
    }
    if any(int(value) < 1 for value in values.values()):
        raise ValueError("DFlash supervision sizing inputs must be positive")
    if int(active_capacity) > int(batch_capacity):
        raise ValueError("DFlash active capacity cannot exceed batch capacity")
    if int(forward_dtype_bytes) not in (1, 2, 4, 8):
        raise ValueError("forward_dtype_bytes must describe a real tensor dtype")

    batch = int(batch_capacity)
    active = int(active_capacity)
    vocab = int(vocab_size)
    hidden = int(hidden_size)
    depth = int(draft_depth)
    forward = int(forward_dtype_bytes)
    fp32 = 4

    compact_snapshot_and_working = depth * (
        # corrected+target snapshots, four FP32 loss/probability workspaces,
        # and the native STE logit gradient
        vocab * (2 * forward + 4 * fp32 + forward)
        + 128 * forward
        + 1  # semantic bool mask
        + (fp32 if stochastic else 0)  # inverse temperature
        + (0 if output_residual else hidden * forward)
    )
    proposal_pairs = 2 if int(tensor_parallel_size) > 1 else 1
    retained_proposal = depth * vocab * (2 * forward) * proposal_pairs
    trace_signal = (
        active
        * depth
        * (
            vocab * (forward + 2 * fp32)
            + 128 * forward
            + 1
            + fp32
            + (fp32 if stochastic else 0)
            + (0 if output_residual else hidden * forward)
        )
        if trace_capture
        else 0
    )
    return (
        (active - 1) * compact_snapshot_and_working
        + (batch - 1) * retained_proposal
        + trace_signal
    )


def estimate_adaptation_memory(
    *,
    num_slots: int,
    max_in_flight: int,
    num_params: int,
    vocab_size: int,
    rank: int,
    markov_dim: int,
    hidden_size: int,
    draft_depth: int,
    adapter_row_capacity: int,
    with_optimizer: bool,
    with_fisher: bool,
    with_optimizer_preview: bool,
    retain_source_signal: bool,
    trace_capture: bool,
    safety_factor: float,
    calibrated_reserve_mb: int = 0,
    enabled: bool = True,
    weight_update_mode: str = "output_residual",
    forward_dtype_bytes: int = 4,
    supervision_fanout_bytes: int = 0,
) -> AdaptationMemoryLedger:
    """Return resident bytes and the unallocated runtime headroom requirement.

    Resident buffers are charged automatically because they are allocated before
    KV sizing. ``reserve_bytes`` covers tensors created later by autograd and
    proposal supervision, and is therefore subtracted explicitly from KV budget.
    """
    if forward_dtype_bytes not in (1, 2, 4, 8):
        raise ValueError("forward_dtype_bytes must describe a real tensor dtype")
    if int(supervision_fanout_bytes) < 0:
        raise ValueError("supervision_fanout_bytes must be non-negative")
    from lightcone_spec.adapters.adapter_params import canonical_tail_layout_mode

    mode = canonical_tail_layout_mode(weight_update_mode)
    fp32 = 4
    active_staging = 2 * num_slots * num_params * fp32 if enabled else 0
    # One graph-visible model-dtype row per request slot plus one process-wide
    # scratch row for the serialized side-stream candidate reconstruction.
    # Publish never borrows that scratch: staging -> forward_active is slot
    # local, so main-stream publication cannot race the side stream.
    forward_active = (
        num_slots * num_params * forward_dtype_bytes if enabled else 0
    )
    forward_candidate_scratch = (
        num_params * forward_dtype_bytes if enabled else 0
    )
    # One immutable request-start centre is shared by every slot.  Slot-local
    # active/staging rows above are themselves the FP32 masters.
    fp32_master = num_params * fp32 if enabled else 0
    gradient = num_slots * max_in_flight * num_params * fp32 if enabled else 0
    optimizer_vectors = (2 if with_optimizer else 0) + (
        2 if with_optimizer_preview else 0
    )
    optimizer = num_slots * optimizer_vectors * num_params * fp32 if enabled else 0
    # phi_source + candidate_delta lanes; candidate_grad is accounted above.
    scratch_vectors = 2 * max_in_flight
    if with_fisher:
        scratch_vectors += 1 + max_in_flight
    candidate_scratch = (
        num_slots * scratch_vectors * num_params * fp32 if enabled else 0
    )
    graph_rows = (
        adapter_row_capacity * num_params * forward_dtype_bytes
        + adapter_row_capacity * (8 + forward_dtype_bytes)
        if enabled
        else 0
    )
    # Output-residual owns a compact B/R_h projection.  Tail modes reuse the
    # already-accounted frozen LM head and must not charge or copy it again.
    artifact_values = (
        vocab_size * rank + hidden_size * 128
        if enabled and mode == "output_residual"
        else 0
    )
    artifact_bytes = artifact_values * forward_dtype_bytes
    if enabled and (with_fisher or trace_capture):
        # Frozen event-sketch bucket/sign tables.
        artifact_bytes += vocab_size * (8 + fp32)
    fixed = (
        active_staging
        + forward_active
        + forward_candidate_scratch
        + fp32_master
        + gradient
        + optimizer
        + candidate_scratch
        + graph_rows
        + artifact_bytes
    )

    signal_values = (
        3 * draft_depth * vocab_size
        + draft_depth * (hidden_size + 128 + markov_dim + 8)
    )
    # The common workspace charges the first row. Backend-specific batch
    # fan-out above adds every simultaneously vectorized row, while serialized
    # DSpark continues to pay only this one-row workspace.
    working_values = (
        8 * draft_depth * vocab_size
        + draft_depth * (hidden_size + 128 + markov_dim + 8)
    )
    per_candidate = fp32 * (working_values + 6 * num_params)
    transient = (
        per_candidate + int(supervision_fanout_bytes) if enabled else 0
    )
    if retain_source_signal:
        transient += (
            fp32 * num_slots * max_in_flight * signal_values
            if enabled
            else 0
        )
    trace_bytes = 0
    if enabled and trace_capture:
        # Runtime admission permits at most one live replay label per request.
        # The normal label owns five P-sized snapshots.  Poll-before-observe can
        # temporarily retain its two pending phi snapshots while constructing
        # that label, so seven P is the true no-sync peak.
        trace_bytes = fp32 * num_slots * (7 * num_params + 3 * 387)
        transient += trace_bytes
    calibrated = max(0, int(calibrated_reserve_mb)) * (1 << 20)
    reserve = max(math.ceil(transient * safety_factor), calibrated)
    return AdaptationMemoryLedger(
        fixed_bytes=int(fixed),
        transient_bytes=int(transient),
        calibrated_bytes=calibrated,
        reserve_bytes=int(reserve),
        num_slots=num_slots,
        max_in_flight=max_in_flight,
        num_params=num_params,
        adapter_row_capacity=adapter_row_capacity,
        active_staging_bytes=int(active_staging),
        forward_active_bytes=int(forward_active),
        forward_candidate_scratch_bytes=int(forward_candidate_scratch),
        fp32_master_bytes=int(fp32_master),
        gradient_bytes=int(gradient),
        optimizer_bytes=int(optimizer),
        candidate_scratch_bytes=int(candidate_scratch),
        graph_row_buffer_bytes=int(graph_rows),
        trace_bytes=int(trace_bytes),
        supervision_fanout_bytes=(
            int(supervision_fanout_bytes) if enabled else 0
        ),
        activation_reserve_bytes=int(reserve),
        artifact_bytes=int(artifact_bytes),
    )


@dataclass
class SlotState:
    slot_index: int
    request_epoch: int = 0
    tenant_id_hash: str = ""
    request_id: str = ""
    active_version: int = 0
    active_has_effect: bool = False
    in_use: bool = False


class AdapterBank:
    def __init__(
        self,
        num_slots: int,
        num_params: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        forward_dtype: torch.dtype | None = None,
        max_in_flight: int = 1,
        with_optimizer: bool = True,
        with_fisher: bool = False,
        with_optimizer_preview: bool = False,
    ):
        self.num_slots = num_slots
        self.num_params = num_params
        self.max_in_flight = max_in_flight
        if dtype is not torch.float32:
            raise ValueError(
                "AdapterBank canonical master dtype must be torch.float32"
            )
        forward_dtype = forward_dtype or dtype
        if not forward_dtype.is_floating_point:
            raise ValueError("AdapterBank forward dtype must be floating point")
        self.forward_dtype = forward_dtype
        # Fixed allocation; addresses never change after construction.
        self.active = torch.zeros(num_slots, num_params, dtype=dtype, device=device)
        self.staging = torch.zeros(num_slots, num_params, dtype=dtype, device=device)
        self.forward_active = torch.zeros(
            num_slots,
            num_params,
            dtype=forward_dtype,
            device=device,
        )
        # CommonCandidateGenerator runs on one serialized side stream, so one
        # process-lifetime row is sufficient.  It is deliberately not reused
        # by publish(), which runs at a main-stream graph boundary.
        self.canonical_forward_scratch = torch.zeros(
            num_params,
            dtype=forward_dtype,
            device=device,
        )
        self.exp_avg = (
            torch.zeros(num_slots, num_params, dtype=dtype, device=device)
            if with_optimizer
            else None
        )
        self.exp_avg_sq = (
            torch.zeros(num_slots, num_params, dtype=dtype, device=device)
            if with_optimizer
            else None
        )
        self.preview_exp_avg = (
            torch.zeros(num_slots, num_params, dtype=dtype, device=device)
            if with_optimizer_preview
            else None
        )
        self.preview_exp_avg_sq = (
            torch.zeros(num_slots, num_params, dtype=dtype, device=device)
            if with_optimizer_preview
            else None
        )
        scratch_shape = (num_slots, max_in_flight, num_params)
        self.phi_source = torch.zeros(scratch_shape, dtype=dtype, device=device)
        self.candidate_grad = torch.zeros(scratch_shape, dtype=dtype, device=device)
        self.candidate_delta = torch.zeros(scratch_shape, dtype=dtype, device=device)
        self.fisher = (
            torch.zeros(num_slots, num_params, dtype=dtype, device=device)
            if with_fisher
            else None
        )
        self.candidate_fisher = (
            torch.zeros(scratch_shape, dtype=dtype, device=device)
            if with_fisher
            else None
        )
        # One stable host byte per candidate lane carries the device-side
        # finite/health predicate to the control plane.  CUDA lanes are pinned
        # so the side stream can enqueue an asynchronous D2H copy; the
        # candidate completion event is recorded *after* that copy.  This is
        # deliberately host memory and is therefore not charged to the HBM
        # ledger above.
        self.candidate_health_host = torch.zeros(
            (num_slots, max_in_flight),
            dtype=torch.bool,
            device="cpu",
            pin_memory=str(device).startswith("cuda"),
        )
        self._candidate_health_generation = [
            [0 for _ in range(max_in_flight)] for _ in range(num_slots)
        ]
        self._candidate_health_epoch = [
            [0 for _ in range(max_in_flight)] for _ in range(num_slots)
        ]
        self.slots = [SlotState(slot_index=i) for i in range(num_slots)]

    # ---- lifecycle -----------------------------------------------------

    def allocate(self, request_id: str, tenant_id_hash: str) -> SlotState:
        for slot in self.slots:
            if not slot.in_use:
                slot.in_use = True
                slot.request_epoch += 1
                slot.request_id = request_id
                slot.tenant_id_hash = tenant_id_hash
                slot.active_version = 0
                slot.active_has_effect = False
                self.active[slot.slot_index].zero_()
                self.staging[slot.slot_index].zero_()
                self.forward_active[slot.slot_index].zero_()
                self.phi_source[slot.slot_index].zero_()
                self.candidate_grad[slot.slot_index].zero_()
                self.candidate_delta[slot.slot_index].zero_()
                if self.exp_avg is not None:
                    self.exp_avg[slot.slot_index].zero_()
                    self.exp_avg_sq[slot.slot_index].zero_()
                if self.preview_exp_avg is not None:
                    self.preview_exp_avg[slot.slot_index].zero_()
                    self.preview_exp_avg_sq[slot.slot_index].zero_()
                if self.fisher is not None:
                    self.fisher[slot.slot_index].zero_()
                    self.candidate_fisher[slot.slot_index].zero_()
                return slot
        raise AdaptationCapacityError("adapter bank exhausted: no free slot")

    def free(self, slot_index: int) -> None:
        slot = self.slots[slot_index]
        slot.in_use = False
        slot.request_id = ""
        # Parameters are cleared on the *next* allocate, so late kernels
        # can never write into a fresh request's storage unnoticed (the
        # epoch check catches them first).

    def initialize_slot(self, slot_index: int, params: torch.Tensor) -> None:
        """Install a request-start no-op without changing fixed addresses.

        Tail LoRA has a non-zero input factor and a zero output factor, so
        clearing every row to numeric zero would make its first gradient zero.
        This copy is done once at request admission, before the slot is exposed
        to proposal kernels.
        """
        if params.numel() != self.num_params:
            raise ValueError(
                f"initial parameter count {params.numel()} != {self.num_params}"
            )
        self.staging[slot_index].copy_(
            params.to(device=self.staging.device, dtype=self.staging.dtype)
        )
        self._canonicalize_slot(slot_index)

    # ---- access with ownership checks ------------------------------------

    def check_owner(
        self, slot_index: int, request_epoch: int, tenant_id_hash: str
    ) -> None:
        slot = self.slots[slot_index]
        if slot.request_epoch != request_epoch:
            raise ExactnessViolation(
                f"slot {slot_index}: epoch {slot.request_epoch} != request "
                f"epoch {request_epoch} (ABA reuse)"
            )
        if slot.tenant_id_hash != tenant_id_hash:
            raise ExactnessViolation(
                f"slot {slot_index}: tenant mismatch (cross-tenant read/write)"
            )

    def read_active(self, slot_index: int) -> torch.Tensor:
        """Return the canonical FP32 master (backward-compatible alias)."""
        return self.active[slot_index]

    def read_forward_active(self, slot_index: int) -> torch.Tensor:
        """Return the fixed-address model-dtype serving parameters."""
        return self.forward_active[slot_index]

    def candidate_forward_buffer(self) -> torch.Tensor:
        """Return the fixed process-wide model-dtype candidate scratch row.

        The runtime owns stream ordering for this shared row.  It is safe only
        because candidate generation is serialized on one side stream.
        """
        return self.canonical_forward_scratch

    def optimizer_state(self, slot_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.exp_avg is None or self.exp_avg_sq is None:
            raise RuntimeError("adapter bank has no optimizer state")
        return self.exp_avg[slot_index], self.exp_avg_sq[slot_index]

    def optimizer_preview_state(
        self, slot_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.preview_exp_avg is None or self.preview_exp_avg_sq is None:
            raise RuntimeError("adapter bank has no optimizer preview state")
        return (
            self.preview_exp_avg[slot_index],
            self.preview_exp_avg_sq[slot_index],
        )

    def candidate_buffers(
        self, slot_index: int, lane: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lane < 0 or lane >= self.max_in_flight:
            raise AdaptationCapacityError(
                f"candidate lane {lane} exceeds max_in_flight={self.max_in_flight}"
            )
        return (
            self.phi_source[slot_index, lane],
            self.candidate_grad[slot_index, lane],
            self.candidate_delta[slot_index, lane],
        )

    def candidate_fisher_buffer(
        self, slot_index: int, lane: int
    ) -> torch.Tensor:
        if self.candidate_fisher is None:
            raise RuntimeError("adapter bank has no Fisher candidate state")
        return self.candidate_fisher[slot_index, lane]

    def prepare_candidate_health(
        self, slot_index: int, request_epoch: int, lane: int
    ) -> tuple[torch.Tensor, int]:
        """Reserve a stable host health byte for one candidate generation.

        The request epoch protects slot reuse and the monotonically increasing
        generation protects lane reuse within an epoch.  Callers must retain
        both values and validate them before reading the byte.
        """

        if lane < 0 or lane >= self.max_in_flight:
            raise AdaptationCapacityError(
                f"candidate lane {lane} exceeds max_in_flight={self.max_in_flight}"
            )
        slot = self.slots[slot_index]
        if not slot.in_use or slot.request_epoch != request_epoch:
            raise ExactnessViolation(
                f"candidate health reservation for stale slot {slot_index}: "
                f"request epoch {request_epoch}, active epoch "
                f"{slot.request_epoch}, in_use={slot.in_use}"
            )
        generation = self._candidate_health_generation[slot_index][lane] + 1
        self._candidate_health_generation[slot_index][lane] = generation
        self._candidate_health_epoch[slot_index][lane] = request_epoch
        health = self.candidate_health_host[slot_index, lane]
        health.fill_(False)
        return health, generation

    def read_candidate_health(
        self,
        slot_index: int,
        request_epoch: int,
        lane: int,
        generation: int,
    ) -> bool:
        """Read a completed health lane, failing closed on ABA/lane reuse.

        Completion ordering is owned by the runtime's candidate CUDA event;
        this method intentionally performs no CUDA query or synchronization.
        """

        if lane < 0 or lane >= self.max_in_flight:
            raise ExactnessViolation(
                f"candidate health read uses invalid lane {lane}"
            )
        slot = self.slots[slot_index]
        actual_generation = self._candidate_health_generation[slot_index][lane]
        actual_epoch = self._candidate_health_epoch[slot_index][lane]
        if (
            not slot.in_use
            or slot.request_epoch != request_epoch
            or actual_epoch != request_epoch
            or actual_generation != generation
        ):
            raise ExactnessViolation(
                f"candidate health lane mismatch for slot {slot_index}, "
                f"lane {lane}: request epoch/generation "
                f"({request_epoch}, {generation}), active "
                f"({slot.request_epoch}, {actual_generation}), lane epoch "
                f"{actual_epoch}, in_use={slot.in_use}"
            )
        return bool(self.candidate_health_host[slot_index, lane])

    def write_staging(
        self, slot_index: int, request_epoch: int, params: torch.Tensor
    ) -> None:
        slot = self.slots[slot_index]
        if slot.request_epoch != request_epoch:
            raise ExactnessViolation(
                f"stale side-stream write to slot {slot_index}: epoch "
                f"{request_epoch} != {slot.request_epoch}"
            )
        self.staging[slot_index].copy_(
            params.to(device=self.staging.device, dtype=self.staging.dtype)
        )

    def publish(self, slot_index: int, request_epoch: int) -> int:
        """Publish one Q-DQ canonical row at a legal graph boundary.

        Program order is FP32 staging -> model-dtype forward_active -> FP32
        active/staging.  The host version advances only after all fixed-address
        copies have been enqueued on the caller's current stream.
        """
        slot = self.slots[slot_index]
        if slot.request_epoch != request_epoch:
            raise ExactnessViolation(
                f"stale publish to slot {slot_index}: epoch {request_epoch} "
                f"!= {slot.request_epoch}"
            )
        self._canonicalize_slot(slot_index)
        slot.active_version += 1
        # A publication may be driven by a device-resident predicate, so
        # checking whether the row is numerically zero here would force a host
        # synchronization.  Conservatively mark it effective after the first
        # legal publish; before that point A_d/A_m/A_c can be skipped exactly.
        slot.active_has_effect = True
        return slot.active_version

    def _canonicalize_slot(self, slot_index: int) -> None:
        """Quantize once and make both FP32 banks the exact dequantized point."""
        self.forward_active[slot_index].copy_(self.staging[slot_index])
        self.active[slot_index].copy_(self.forward_active[slot_index])
        self.staging[slot_index].copy_(self.active[slot_index])
