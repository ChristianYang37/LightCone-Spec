"""GPU-side adaptation runtime (spec 8.2-8.5).

Implements the seven fork hooks on top of the same primitives the CPU
reference engine uses: canvas version locks, teacher-signal capture,
the shared candidate generators, controller decisions and the adapter
bank's staging -> active publish protocol. Update math runs on a
dedicated low-priority side stream; completion is signalled by CUDA
events polled at hook 6.

This module requires CUDA + the pinned fork and is exercised by the
`gpu`/`integration` test markers on declared hardware; it is import-safe
on CPU hosts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterShapes,
    canonicalize_master_vector,
    initial_parameter_vector,
)
from lightcone_spec.config.schema import AdaptationConfig, CONTROLLER_METHODS
from lightcone_spec.exit_codes import ExactnessViolation
from lightcone_spec.methods.base import (
    ArrivalContext,
    CandidateUpdate,
    DecisionKind,
    MethodRuntime,
    PublishPolicy,
    SourceBoundCandidateBatch,
    TeacherSignal,
    apply_delta_with_trust_region,
)
from lightcone_spec.runtime.events import UpdateEventChain, monotonic_us
from lightcone_spec.sglang_bridge.bank import AdapterBank
from lightcone_spec.sglang_bridge.hooks import (
    AcceptanceDone,
    AdaptationHooks,
    DraftInputsReady,
    ProposalReady,
    RequestLifecycle,
    RoundCommitted,
    UpdatePollPoint,
    VerifyLogitsReady,
)
from lightcone_spec.sglang_bridge.telemetry import (
    CudaTelemetryLanePool,
    RoundTelemetry,
    TelemetrySink,
    UpdateTelemetry,
)
from lightcone_spec.trajectory.distance import DistanceWeights
from lightcone_spec.trajectory.state import TrajectoryState
from lightcone_spec.trajectory.zvector import ZVectorizer, default_zvectorizer


_UPDATE_CUDA_EVENT_NAMES = (
    "ready_event",
    "side_start",
    "preview_ready",
    "candidate_end",
    "backward_start",
    "backward_end",
    "optimizer_end",
    "controller_start",
    "controller_end",
    "publish_start",
    "publish_end",
    "telemetry_ready",
)


def _sequence_group_from_request_id(request_id: str) -> str:
    """Recover the prompt-only group encoded by the benchmark client."""
    parts = request_id.split("-", 3)
    if len(parts) >= 2 and parts[0] == "lightcone":
        token = parts[1]
        if len(parts) >= 3 and len(parts[2]) == 65 and parts[2][0] == "p":
            token = parts[2]
        if len(token) == 65 and token[0] in ("g", "p"):
            digest = token[1:]
            if all(char in "0123456789abcdef" for char in digest):
                return digest
    return request_id


_PAIRED_REQUEST_ID = re.compile(
    r"^lightcone-g(?P<checkpoint>[0-9a-f]{64})-"
    r"p(?P<prompt>[0-9a-f]{64})-(?P<run>.+)-"
    r"(?P<repetition>-?\d+)-(?P<index>\d+)$"
)
_SAMPLING_SEED_SUFFIX = re.compile(r"-s(?P<seed>[0-9a-f]{16})$")


def _evaluation_pair_from_request_id(request_id: str) -> str:
    """Stable prompt/checkpoint/seed unit shared across method runs.

    The run id contains the method and therefore cannot be used for pairing.
    The exact checkpoint hash, repetition and deterministic job index are
    method-invariant in the benchmark client and distinguish multiple seeds
    of the same base prompt.  Unknown request ids remain request-local, which
    fails closed instead of accidentally pairing unrelated traffic.
    """

    match = _PAIRED_REQUEST_ID.fullmatch(request_id)
    if match is None:
        return request_id
    seed_match = _SAMPLING_SEED_SUFFIX.search(match.group("run"))
    seed = seed_match.group("seed") if seed_match is not None else "unbound"
    return (
        f"g{match.group('checkpoint')}::r{match.group('repetition')}::"
        f"i{match.group('index')}::s{seed}"
    )


def _host_scalar_or_zero(value) -> float:
    """Serialize host scalars now and CUDA scalars in the deferred worker."""
    return 0.0 if isinstance(value, torch.Tensor) else float(value)


def _l2_delta_from_arrival_state(
    raw_gradient: torch.Tensor,
    arrival_state,
    lr: float,
    valid: bool | torch.Tensor = True,
    *,
    parameter: torch.Tensor | None = None,
    weight_decay: float = 0.0,
) -> torch.Tensor:
    """Pure paired-L2 preview from a pre-L3 arrival-state snapshot."""

    from lightcone_spec.methods.optim import adamw_delta

    return adamw_delta(
        raw_gradient,
        arrival_state,
        lr,
        valid=valid,
        parameter=parameter,
        weight_decay=weight_decay,
    )


@dataclass
class _RequestCtx:
    request_id: str
    slot_index: int
    request_epoch: int
    tenant_id_hash: str
    stream_id: Optional[str]
    round_id: int = -1
    prefix_len: int = 0
    canvas_version: int = -1
    states: list[Any] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    update_seq: int = 0
    phi_source_by_update: dict[str, torch.Tensor] = field(default_factory=dict)
    prefix_len_by_update: dict[str, int] = field(default_factory=dict)
    wall_by_round: dict[int, float] = field(default_factory=dict)
    pending_states: list[dict] = field(default_factory=list)
    signals_by_round: dict[int, TeacherSignal] = field(default_factory=dict)
    replay_labels: list[dict] = field(default_factory=list)
    pending_replay_starts: list[dict] = field(default_factory=list)
    # Sticky: once a backend reports an approximate host prefix, token-delay
    # features for all later updates in this request are no longer exact.
    prefix_feature_exact: bool = True
    # Staged replay capture measures output progress from the first proposal
    # prefix.  It is request-local so slot reuse cannot inherit a phase.
    trace_start_prefix_len: int | None = None


@dataclass(frozen=True)
class _SourceBinding:
    source_round: int
    source_version: int


@dataclass
class _GpuTrajectoryState:
    round_id: int
    topk_token_ids: torch.Tensor
    topk_probs: torch.Tensor
    other_mass: torch.Tensor
    hidden_proj: torch.Tensor
    event_sketch: torch.Tensor


class AdaptationRuntime(AdaptationHooks):
    """One runtime per engine process; request-slot isolation inside."""

    def __init__(
        self,
        config: AdaptationConfig,
        method_factory,
        shapes: AdapterShapes,
        basis: torch.Tensor,
        telemetry: TelemetrySink,
        num_slots: int = 16,
        device: str = "cuda",
        distance_weights: Optional[DistanceWeights] = None,
        zvectorizer: Optional[ZVectorizer] = None,
        constant_controller_delay: Optional[int] = None,
        parameter_layout_sha256: str | None = None,
        enable_replay_writer: bool = True,
        gradient_consensus_fn=None,
        forward_dtype: torch.dtype = torch.float32,
        update_telemetry_lane_count: int | None = None,
    ):
        self.config = config
        self.method_factory = method_factory
        self.shapes = shapes
        self.basis = basis
        self.telemetry = telemetry
        self.device = device
        self.forward_dtype = forward_dtype
        self.bank = AdapterBank(
            num_slots,
            shapes.num_params(),
            device=device,
            max_in_flight=config.async_.max_in_flight,
            with_optimizer=config.optimizer == "adamw",
            with_fisher=config.method == "lc_transport",
            with_optimizer_preview=config.method == "lc_transport",
            forward_dtype=forward_dtype,
        )
        self.methods: dict[int, MethodRuntime] = {}
        self.requests: dict[str, _RequestCtx] = {}
        self.weights = distance_weights or DistanceWeights(
            a_p=1 / 3, a_h=1 / 3, a_e=1 / 3
        )
        self.zvec = zvectorizer or default_zvectorizer()
        self.constant_controller_delay = constant_controller_delay
        self.parameter_layout_sha256 = parameter_layout_sha256
        self.enable_replay_writer = bool(enable_replay_writer)
        self.gradient_consensus_fn = gradient_consensus_fn
        self._needs_trajectory = bool(
            (
                config.trace.trace_capture_max_bytes > 0
                and self.enable_replay_writer
            )
            or (
                config.method in CONTROLLER_METHODS
                and constant_controller_delay is None
            )
        )
        self.phi0 = canonicalize_master_vector(
            initial_parameter_vector(shapes, device=device), forward_dtype
        )
        if device.startswith("cuda"):
            # Lower numeric values are higher priority. Torch clamps values to
            # the device-supported range, which is the only public API shared
            # by the supported Torch releases (the old range query is gone in
            # 2.11). The default 0 therefore remains the lowest-priority lane.
            self.side_stream = torch.cuda.Stream(
                priority=int(config.async_.stream_priority)
            )
            # Counterfactual labels are evidence-only.  They run on their own
            # default/low-priority lane and are ordered after a main-ready
            # event; the decode stream never waits for this lane.
            self.trace_stream = torch.cuda.Stream(priority=0)
            self.trace_clock = "instrumented_low_priority_stream_v1"
        else:
            self.side_stream = None
            self.trace_stream = None
            self.trace_clock = "synchronous_cpu_v1"
        self._trace_bytes_written = 0
        self._trace_records_by_request: dict[str, int] = {}
        self._trace_live_by_request: dict[str, int] = {}
        self._distance_tensor_cache: dict[str, torch.Tensor | None] = {}
        self._sketch_bucket = None
        self._sketch_sign = None
        self._z_mean = None
        self._z_std = None
        self.max_allocator_growth_bytes = 0
        self._adaptation_frozen_reason: str | None = None
        self._update_telemetry_pool: CudaTelemetryLanePool | None = None
        if device.startswith("cuda"):
            lane_count = (
                int(update_telemetry_lane_count)
                if update_telemetry_lane_count is not None
                else max(2, 2 * num_slots * config.async_.max_in_flight)
            )
            self._update_telemetry_pool = CudaTelemetryLanePool(
                device=device,
                max_batch_size=1,
                lane_count=lane_count,
                event_names=_UPDATE_CUDA_EVENT_NAMES,
                host_names=(),
                untimed_event_names=("preview_ready", "telemetry_ready"),
                device_scalars={
                    "candidate_delta_norm": torch.float32,
                    "optimizer_step": torch.int64,
                },
            )
        if device.startswith("cuda") and self._needs_trajectory:
            self._init_gpu_trajectory_artifacts()

    @property
    def needs_trajectory(self) -> bool:
        return self._needs_trajectory

    def wants_trace_signal(self, request_id: str) -> bool:
        """Whether a backend should retain its next heavy teacher window.

        A live counterfactual label must keep receiving signals until its
        horizon completes.  Otherwise, stop retaining full-vocabulary logits
        as soon as either the per-request record quota or the process-wide byte
        budget cannot admit another label.  This query is host-only and does
        not synchronize CUDA.
        """

        if (
            not self.enable_replay_writer
            or self.config.trace.trace_capture_max_bytes <= 0
        ):
            return False
        if request_id not in self.requests:
            return False
        if self._trace_live_by_request.get(request_id, 0) > 0:
            return True
        if (
            self._trace_records_by_request.get(request_id, 0)
            >= self.config.trace.trace_capture_max_records_per_request
        ):
            return False
        if not self._trace_stage_due(self.requests[request_id]):
            return False
        return (
            self._trace_bytes_written + self._default_trace_reservation_bytes()
            <= self.config.trace.trace_capture_max_bytes
        )

    def _default_trace_reservation_bytes(self) -> int:
        # One live replay label owns source/arrival/actual/counterfactual
        # parameter snapshots plus gradients.  Reserve the real upper bound
        # before cloning so a full-rank tail cannot silently exceed the trace
        # quota.  The normal inference/adaptation pools are unaffected.
        return (
            9 * self.shapes.num_params() * torch.float32.itemsize
            + 3 * 387 * torch.float32.itemsize
            + (64 << 10)
        )

    def _run_on_trace_stream(self, callback, *values) -> None:
        """Schedule trace-only tensor work without a trace->main dependency."""

        if self.trace_stream is None:
            callback()
            return
        current = torch.cuda.current_stream(device=self.device)
        main_ready = torch.cuda.Event()
        main_ready.record(current)
        self.trace_stream.wait_event(main_ready)
        for value in values:
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(self.trace_stream)
        with torch.cuda.stream(self.trace_stream):
            callback()

    def _init_gpu_trajectory_artifacts(self) -> None:
        """Copy frozen clock artifacts once; never inside a decode round."""

        def tensor(value):
            return (
                None
                if value is None
                else torch.as_tensor(value, device=self.device, dtype=torch.float32)
            )

        self._distance_tensor_cache = {
            "hidden_mean": tensor(self.weights.hidden_mean),
            "hidden_std": tensor(self.weights.hidden_std),
            "event_mean": tensor(self.weights.event_mean),
            "event_std": tensor(self.weights.event_std),
        }
        if self.config.method == "lc_transport" or (
            self.config.trace.trace_capture_max_bytes > 0
        ):
            buckets, signs = zip(
                *(self.zvec.sketch._hashes(token) for token in range(self.shapes.vocab_size))
            )
            self._sketch_bucket = torch.tensor(
                buckets, device=self.device, dtype=torch.int64
            )
            self._sketch_sign = torch.tensor(
                signs, device=self.device, dtype=torch.float32
            )
            self._z_mean = tensor(self.zvec.mean)
            self._z_std = tensor(self.zvec.std)

    # ---- hook 7: lifecycle ------------------------------------------------

    def on_request_lifecycle(self, ev: RequestLifecycle) -> None:
        if ev.event in ("begin", "stream_begin"):
            slot = self.bank.allocate(ev.request_id, ev.tenant_id_hash)
            self.bank.initialize_slot(slot.slot_index, self.phi0)
            ctx = _RequestCtx(
                request_id=ev.request_id,
                slot_index=slot.slot_index,
                request_epoch=slot.request_epoch,
                tenant_id_hash=ev.tenant_id_hash,
                stream_id=ev.stream_id,
            )
            self.requests[ev.request_id] = ctx
            # A stream is one long request across speculation rounds. Never
            # retain method/controller state across request IDs or tenants.
            self.methods[slot.slot_index] = self.method_factory()
            method = self.methods[slot.slot_index]
            exp_avg = exp_avg_sq = None
            if self.bank.exp_avg is not None:
                exp_avg, exp_avg_sq = self.bank.optimizer_state(slot.slot_index)
            fisher = (
                self.bank.fisher[slot.slot_index]
                if self.bank.fisher is not None
                else None
            )
            method.bind_slot_state(exp_avg, exp_avg_sq, fisher)
            method.bind_gradient_consensus(self.gradient_consensus_fn)
            if self.bank.preview_exp_avg is not None:
                method.bind_candidate_preview(
                    *self.bank.optimizer_preview_state(slot.slot_index)
                )
        elif ev.event in ("end", "stream_end"):
            ctx = self.requests.pop(ev.request_id, None)
            if ctx is not None:
                self._retire_request_work(ctx)
                self.bank.free(ctx.slot_index)
                self.methods.pop(ctx.slot_index, None)

    def cancel_pending(self, request_id: str) -> None:
        """Make pending request-local work unpublishable without host waits."""
        ctx = self.requests.get(request_id)
        if ctx is None:
            return
        self._retire_request_work(ctx)

    def _retire_request_work(self, ctx: _RequestCtx) -> None:
        for label in list(ctx.replay_labels):
            if label.get("paired_tts_barrier"):
                self._record_incomplete_tts_pair(
                    ctx,
                    label["update_id"],
                    "request_ended_before_paired_horizons_completed",
                    candidate_arrival_round=label.get(
                        "candidate_arrival_round"
                    ),
                    actual_arrival_round=label.get("actual_arrival_round"),
                )
        for pending in list(ctx.pending_replay_starts):
            if pending.get("paired_tts_barrier"):
                self._record_incomplete_tts_pair(
                    ctx,
                    pending["cand"].update_id,
                    "request_ended_before_candidate_arrival_signal",
                    candidate_arrival_round=pending.get("arrival_round"),
                    actual_arrival_round=pending.get(
                        "actual_arrival_round"
                    ),
                )
        if self.side_stream is not None:
            current = torch.cuda.current_stream(device=self.device)
            for item in ctx.pending:
                event = item.get("event")
                if event is not None:
                    # Queue lifetime ordering; never synchronize the host.
                    current.wait_event(event)
                cand = item.get("candidate")
                if cand is not None:
                    for value in (
                        cand.raw_gradient,
                        cand.candidate_delta,
                        cand.loss.total,
                        cand.loss.expected_accepted_prefix,
                        cand.fisher_snapshot,
                        *(vars(cand.signal).values() if cand.signal is not None else ()),
                    ):
                        if isinstance(value, torch.Tensor) and value.is_cuda:
                            value.record_stream(current)
        for item in ctx.pending:
            cand = item.get("candidate")
            if cand is not None:
                chain = item["chain"]
                record = UpdateTelemetry(
                    request_id=ctx.request_id,
                    update_id=cand.update_id,
                    source_round=cand.source_round,
                    source_version=cand.source_version,
                    snapshot_ts_us=chain.snapshot_ts_us or 0.0,
                    source_prefix_len=ctx.prefix_len_by_update[cand.update_id],
                    teacher_ts_us=chain.teacher_ts_us,
                    launch_ts_us=chain.launch_ts_us,
                    done_ts_us=chain.done_ts_us,
                    decision="discard",
                    failure_reason="request_ended",
                    effective_delay_rounds=max(ctx.round_id - cand.source_round, 0),
                    prefix_feature_exact=ctx.prefix_feature_exact,
                    barrier_wait_cpu_us=float(
                        item.get("barrier_wait_cpu_us", 0.0)
                    ),
                    optimizer_step=(
                        0
                        if isinstance(cand.optimizer_step, torch.Tensor)
                        else int(cand.optimizer_step)
                    ),
                )
                event = item.get("event")
                if event is not None:
                    pool = self._update_telemetry_pool
                    telemetry_lane = item.get("telemetry_lane")
                    if pool is None or telemetry_lane is None:
                        raise RuntimeError(
                            "CUDA candidate is missing its telemetry lane"
                        )
                    # The loop above has already enqueued the candidate-end
                    # wait and every record_stream ownership transfer on this
                    # main stream.  This fixed fence therefore guards both
                    # execution lifetime and deferred scalar materialization.
                    telemetry_ready = pool.event(
                        telemetry_lane, "telemetry_ready"
                    )
                    telemetry_ready.record(current)
                    self.telemetry.emit_update_deferred(
                        record,
                        {
                            "event": telemetry_ready,
                            "numerical_ok": cand.numerical_ok,
                            "grad_norm": cand.grad_norm,
                            "candidate_delta_norm": item[
                                "candidate_delta_norm"
                            ],
                            "source_training_loss": cand.loss.total,
                            "source_expected_accepted_prefix": (
                                cand.loss.expected_accepted_prefix
                            ),
                            "optimizer_step": cand.optimizer_step,
                            "ready_event": item.get("ready_event"),
                            "side_start": item.get("side_start"),
                            "candidate_end": event,
                            **(cand.cuda_timing_ref or {}),
                            "release": self._update_lane_release_callback(
                                item
                            ),
                        },
                    )
                else:
                    record.source_training_loss = float(cand.loss.total)
                    record.source_expected_accepted_prefix = float(
                        cand.loss.expected_accepted_prefix
                    )
                    record.grad_norm = float(cand.grad_norm)
                    record.candidate_delta_norm = float(
                        torch.linalg.vector_norm(cand.candidate_delta)
                    )
                    self.telemetry.emit("update", record)
                self._forget_update(ctx, cand.update_id)
        ctx.pending.clear()
        ctx.pending_states.clear()
        ctx.replay_labels.clear()
        ctx.pending_replay_starts.clear()
        ctx.signals_by_round.clear()
        self._trace_live_by_request.pop(ctx.request_id, None)

    def observe_signal(self, request_id: str, signal: TeacherSignal) -> None:
        """Feed a real teacher window to the bounded replay-label recorder.

        This path is disabled unless ``trace_capture_max_bytes`` is positive;
        performance runs therefore pay no counterfactual-loss overhead.
        """
        if (
            not self.enable_replay_writer
            or self.config.trace.trace_capture_max_bytes <= 0
            or request_id.startswith(("lightcone-warmup-", "lightcone-cancel-"))
        ):
            return
        with torch.profiler.record_function("lightcone::trace_oracle_replay"):
            tensors = tuple(
                value
                for value in vars(signal).values()
                if isinstance(value, torch.Tensor)
            )
            self._run_on_trace_stream(
                lambda: self._observe_trace_signal(request_id, signal),
                *tensors,
            )

    def _observe_trace_signal(
        self, request_id: str, signal: TeacherSignal
    ) -> None:
        """Advance instrumented counterfactual labels for one real window."""

        ctx = self.requests[request_id]
        ctx.signals_by_round[signal.source_round] = signal
        for old_round in tuple(ctx.signals_by_round):
            if old_round < signal.source_round:
                ctx.signals_by_round.pop(old_round, None)
        for pending in list(ctx.pending_replay_starts):
            if pending["arrival_round"] != signal.source_round:
                continue
            ctx.pending_replay_starts.remove(pending)
            self._start_replay_label(ctx=ctx, **pending)
        self._advance_replay_labels(ctx, signal)
        # The current signal remains live only through this round's legal poll
        # boundary.  The worker observes verify evidence before it polls, so
        # dropping it here would make every arrival label miss its only window.

    # ---- hook 1 ------------------------------------------------------------

    def on_draft_inputs_ready(self, ev: DraftInputsReady) -> None:
        ctx = self.requests[ev.request_id]
        self.bank.check_owner(ev.slot_index, ev.request_epoch, ctx.tenant_id_hash)
        ctx.round_id = ev.round_id
        if ctx.trace_start_prefix_len is None:
            ctx.trace_start_prefix_len = int(ev.prefix_len)
        ctx.prefix_len = ev.prefix_len
        ctx.canvas_version = ev.active_version

    # ---- hook 2 ------------------------------------------------------------

    def on_proposal_ready(self, ev: ProposalReady) -> None:
        ctx = self.requests[ev.request_id]
        if ev.proposal_version != ctx.canvas_version:
            raise ExactnessViolation(
                f"{ev.request_id} r{ev.round_id}: proposal bound to version "
                f"{ev.proposal_version} but canvas locked {ctx.canvas_version}"
            )

    # ---- hook 3 ------------------------------------------------------------

    def on_verify_logits_ready(self, ev: VerifyLogitsReady) -> None:
        ctx = self.requests[ev.request_id]
        if ev.proposal_version != ctx.canvas_version:
            raise ExactnessViolation(
                f"{ev.request_id} r{ev.round_id}: verify logits version race"
            )
        # Reduce the full-vocabulary distribution on device.  Only the top-64
        # state and three event scalars are materialized on the host later at
        # a legal poll point; verify never performs a DtoH copy.
        if (
            self._needs_trajectory
            and ev.target_logits_ref is not None
            and ev.target_hidden_ref is not None
        ):
            logits = ev.target_logits_ref[0].detach().float()
            log_norm = torch.logsumexp(logits, dim=-1)
            k = min(int(self.config.trajectory.topk), int(logits.shape[-1]))
            top_logits, top_ids = torch.topk(logits, k=k, dim=-1, sorted=True)
            top_probs = torch.exp(top_logits - log_norm)
            other_mass = (1.0 - top_probs.sum()).clamp_min(0.0)
            probs = torch.exp(logits - log_norm)
            entropy = -(probs * (logits - log_norm)).sum()
            margin = (
                top_probs[0] - top_probs[1]
                if k > 1
                else top_probs[0]
            )
            event_sketch = torch.stack((entropy, top_probs[0], margin))
            done = None
            if logits.is_cuda:
                done = torch.cuda.Event()
                done.record(torch.cuda.current_stream(logits.device))
            ctx.pending_states.append(
                {
                    "round_id": ev.round_id,
                    "top_ids": top_ids.to(torch.int32),
                    "top_probs": top_probs,
                    "other_mass": other_mass,
                    "hidden": ev.target_hidden_ref[0].detach().float(),
                    "events": event_sketch,
                    "event": done,
                }
            )
        # Trigger-round teacher capture happens in the fork by enqueuing a
        # capture dict; consumed by maybe_launch_update below.

    # ---- hook 4/5 ------------------------------------------------------------

    def on_acceptance_done(self, ev: AcceptanceDone) -> None:
        if not (
            ev.proposal_version == ev.denominator_version == ev.residual_version
        ):
            raise ExactnessViolation(
                f"{ev.request_id} r{ev.round_id}: version mismatch "
                f"p={ev.proposal_version} d={ev.denominator_version} "
                f"r={ev.residual_version}"
            )

    def on_round_committed(self, ev: RoundCommitted) -> None:
        ctx = self.requests[ev.request_id]
        prefix_before = ctx.prefix_len
        ctx.prefix_feature_exact = bool(
            ctx.prefix_feature_exact and ev.prefix_feature_exact
        )
        if ev.prefix_len_after is not None:
            ctx.prefix_len = ev.prefix_len_after
        ctx.wall_by_round[ev.round_id] = monotonic_us()
        record = RoundTelemetry(
                request_id=ev.request_id,
                round_id=ev.round_id,
                active_version=ev.active_version,
                proposal_version=ev.proposal_version,
                draft_tokens=ev.draft_tokens,
                accepted_drafts=ev.accepted_drafts,
                committed_per_verify=ev.committed_per_verify,
                target_calls=ev.target_calls,
                draft_cuda_us=ev.draft_cuda_us,
                verify_cuda_us=ev.verify_cuda_us,
                accept_cuda_us=ev.accept_cuda_us,
                draft_cpu_us=ev.draft_cpu_us,
                verify_cpu_us=ev.verify_cpu_us,
                rng_substream_id=ev.rng_substream_id,
                version_canary_ok=ev.proposal_version == ctx.canvas_version,
                prefix_pos_before=prefix_before,
                prefix_pos_after=(
                    ev.prefix_len_after
                    if ev.prefix_len_after is not None
                    else prefix_before
                ),
                prefix_len_before=prefix_before,
                verify_len=ev.verify_len,
                batch_size=ev.batch_size,
                offered_concurrency=ev.offered_concurrency,
                round_wall_us=ev.round_wall_us,
                prefix_feature_exact=ctx.prefix_feature_exact,
                algorithmic_censored=ev.algorithmic_censored,
            )
        self.telemetry.emit_round_deferred(record, ev.cuda_timing_ref)

    # ---- side-stream candidate launch ---------------------------------------

    def _reject_update_launch(
        self,
        ctx: _RequestCtx,
        signal: TeacherSignal,
        reason: str,
    ) -> None:
        """Record a pre-launch rejection without creating CUDA work."""

        update_id = f"{ctx.request_id}-u{ctx.update_seq}"
        ctx.update_seq += 1
        self.telemetry.emit(
            "update",
            UpdateTelemetry(
                request_id=ctx.request_id,
                update_id=update_id,
                source_round=signal.source_round,
                source_version=signal.source_version,
                snapshot_ts_us=monotonic_us(),
                active_version_at_arrival=self.bank.slots[
                    ctx.slot_index
                ].active_version,
                decision="discard",
                failure_reason=reason,
                effective_delay_rounds=0,
                prefix_feature_exact=ctx.prefix_feature_exact,
            ),
        )

    def _acquire_update_telemetry_lane(
        self, ctx: _RequestCtx, signal: TeacherSignal
    ) -> int | None:
        pool = self._update_telemetry_pool
        if pool is None:
            return None
        if self._adaptation_frozen_reason is not None:
            self._reject_update_launch(
                ctx,
                signal,
                f"adaptation_frozen:{self._adaptation_frozen_reason}",
            )
            return None
        try:
            return pool.acquire()
        except RuntimeError:
            # Continuing without evidence would silently change the benchmark
            # semantics.  Freeze adaptation process-wide while the original
            # speculative service remains available.
            self._adaptation_frozen_reason = "cuda_update_lane_exhausted"
            self._reject_update_launch(
                ctx, signal, self._adaptation_frozen_reason
            )
            return None

    def _defer_update_lane_release(self, lane: int, event=None) -> None:
        pool = self._update_telemetry_pool
        if pool is None:
            return
        self.telemetry.defer(lambda: pool.release(lane), event=event)

    def _update_lane_release_callback(self, item: dict):
        lane = item.get("telemetry_lane")
        if lane is None:
            return None
        if item.get("telemetry_release_enqueued"):
            raise RuntimeError("CUDA update telemetry lane release was enqueued twice")
        pool = self._update_telemetry_pool
        if pool is None:
            raise RuntimeError("CUDA update telemetry pool disappeared")
        item["telemetry_release_enqueued"] = True
        return lambda: pool.release(int(lane))

    def launch_update(
        self, request_id: str, signal: TeacherSignal
    ) -> Optional[str]:
        """Called by the fork at trigger rounds with the captured teacher
        signal. Runs the candidate math on the side stream."""
        ctx = self.requests[request_id]
        active_version = self.bank.slots[ctx.slot_index].active_version
        if signal.source_round != ctx.round_id:
            self._reject_update_launch(
                ctx,
                signal,
                "source_round_mismatch:"
                f"signal={signal.source_round}:canvas={ctx.round_id}",
            )
            return None
        if signal.source_version != ctx.canvas_version:
            self._reject_update_launch(
                ctx,
                signal,
                "source_canvas_version_mismatch:"
                f"signal={signal.source_version}:canvas={ctx.canvas_version}",
            )
            return None
        if ctx.canvas_version != active_version:
            self._reject_update_launch(
                ctx,
                signal,
                "canvas_active_version_mismatch:"
                f"canvas={ctx.canvas_version}:active={active_version}",
            )
            return None
        if len(ctx.pending) >= self.config.async_.max_in_flight:
            self._reject_update_launch(ctx, signal, "max_in_flight")
            return None
        telemetry_lane = self._acquire_update_telemetry_lane(ctx, signal)
        if self._update_telemetry_pool is not None and telemetry_lane is None:
            return None
        method = self.methods[ctx.slot_index]
        used_lanes = {int(item["lane"]) for item in ctx.pending}
        lane = next(
            i for i in range(self.config.async_.max_in_flight) if i not in used_lanes
        )
        phi_buf, grad_buf, delta_buf = self.bank.candidate_buffers(
            ctx.slot_index, lane
        )
        health_host = None
        health_generation = None
        if self.side_stream is not None:
            health_host, health_generation = self.bank.prepare_candidate_health(
                ctx.slot_index, ctx.request_epoch, lane
            )
        chain = UpdateEventChain(
            update_id=f"{request_id}-u{ctx.update_seq}",
            source_round=signal.source_round,
            source_version=signal.source_version,
        )
        ctx.update_seq += 1
        chain.mark("snapshot")
        chain.mark("teacher")
        ready_event = None
        side_start = None
        preview_event = None
        candidate_delta_norm = None
        if self.side_stream is not None:
            assert telemetry_lane is not None
            telemetry_pool = self._update_telemetry_pool
            assert telemetry_pool is not None
            allocator_before = torch.cuda.memory_reserved(device=self.device)
            current = torch.cuda.current_stream(device=self.device)
            ready_event = telemetry_pool.event(telemetry_lane, "ready_event")
            ready_event.record(current)
            self.side_stream.wait_event(ready_event)
            with torch.cuda.stream(self.side_stream):
                side_start = telemetry_pool.event(telemetry_lane, "side_start")
                side_start.record(self.side_stream)
                phi_buf.copy_(self.bank.read_active(ctx.slot_index))
                phi_source = phi_buf
                forward_phi_source = self.bank.candidate_forward_buffer()
                forward_phi_source.copy_(
                    self.bank.read_forward_active(ctx.slot_index)
                )
                method.prepare_candidate_preview()
                # This event is the side->main ownership fence for every
                # source-bound candidate.  It is deliberately recorded before
                # loss/gradient work: a later main-stream publication may
                # overlap that work, but must not overwrite active/forward
                # banks until their per-lane source copies (and L3 moment
                # preview) are complete.
                preview_event = telemetry_pool.event(
                    telemetry_lane, "preview_ready"
                )
                preview_event.record(self.side_stream)
                chain.mark("launch")
                cuda_timing_ref = {
                    "backward_start": telemetry_pool.event(
                        telemetry_lane, "backward_start"
                    ),
                    "backward_end": telemetry_pool.event(
                        telemetry_lane, "backward_end"
                    ),
                    "optimizer_end": telemetry_pool.event(
                        telemetry_lane, "optimizer_end"
                    ),
                    "optimizer_step_out": telemetry_pool.device_scalar(
                        telemetry_lane, "optimizer_step"
                    ),
                }
                try:
                    cand = method.make_candidate(
                        phi_source,
                        signal,
                        forward_phi_source=forward_phi_source,
                        cuda_timing_ref=cuda_timing_ref,
                    )
                    if cand is not None:
                        grad_buf.copy_(cand.raw_gradient)
                        delta_buf.copy_(cand.candidate_delta)
                        cand.raw_gradient = grad_buf
                        cand.candidate_delta = delta_buf
                        if cand.fisher_snapshot is not None:
                            fisher_buf = self.bank.candidate_fisher_buffer(
                                ctx.slot_index, lane
                            )
                            fisher_buf.copy_(cand.fisher_snapshot)
                            cand.fisher_snapshot = fisher_buf
                        candidate_delta_norm = telemetry_pool.device_scalar(
                            telemetry_lane, "candidate_delta_norm"
                        )
                        torch.linalg.vector_norm(
                            cand.candidate_delta, out=candidate_delta_norm
                        )
                        if isinstance(cand.numerical_ok, torch.Tensor):
                            if (
                                not cand.numerical_ok.is_cuda
                                or cand.numerical_ok.numel() != 1
                            ):
                                raise ExactnessViolation(
                                    "CUDA candidate health must be one device scalar"
                                )
                            # This is the only control D2H in candidate
                            # production: one byte on the low-priority side
                            # stream into a preallocated pinned lane.
                            health_host.copy_(
                                cand.numerical_ok.reshape(()), non_blocking=True
                            )
                        else:
                            health_host.fill_(bool(cand.numerical_ok))
                except Exception:
                    # The SGLang wrapper may fail open to the native backend on
                    # OOM/exactness errors.  Fence all already-enqueued work and
                    # return the fixed telemetry lane even though no pending
                    # item was installed; otherwise repeated request-local
                    # fallback would leak the bounded pool process-wide.
                    failed = telemetry_pool.event(
                        telemetry_lane, "candidate_end"
                    )
                    failed.record(self.side_stream)
                    for value in vars(signal).values():
                        if isinstance(value, torch.Tensor) and value.is_cuda:
                            value.record_stream(self.side_stream)
                    self._defer_update_lane_release(telemetry_lane, failed)
                    raise
            self.max_allocator_growth_bytes = max(
                self.max_allocator_growth_bytes,
                int(torch.cuda.memory_reserved(device=self.device))
                - int(allocator_before),
            )
            done_event = telemetry_pool.event(telemetry_lane, "candidate_end")
            done_event.record(self.side_stream)
            # Signal tensors were allocated/produced on the main stream and
            # remain referenced by CandidateUpdate.  Record side-stream use so
            # the caching allocator cannot recycle them early.
            for value in vars(signal).values():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    value.record_stream(self.side_stream)
        else:
            phi_buf.copy_(self.bank.read_active(ctx.slot_index))
            phi_source = phi_buf
            forward_phi_source = self.bank.candidate_forward_buffer()
            forward_phi_source.copy_(
                self.bank.read_forward_active(ctx.slot_index)
            )
            method.prepare_candidate_preview()
            chain.mark("launch")
            cand = method.make_candidate(
                phi_source,
                signal,
                forward_phi_source=forward_phi_source,
                cuda_timing_ref=None,
            )
            if cand is not None:
                grad_buf.copy_(cand.raw_gradient)
                delta_buf.copy_(cand.candidate_delta)
                cand.raw_gradient = grad_buf
                cand.candidate_delta = delta_buf
                if cand.fisher_snapshot is not None:
                    fisher_buf = self.bank.candidate_fisher_buffer(
                        ctx.slot_index, lane
                    )
                    fisher_buf.copy_(cand.fisher_snapshot)
                    cand.fisher_snapshot = fisher_buf
            done_event = None
            chain.mark("done")
        if cand is None:
            if telemetry_lane is not None:
                self._defer_update_lane_release(telemetry_lane, done_event)
            return None
        if not method.retain_source_signal:
            cand.signal = None
        cand.update_id = chain.update_id
        ctx.phi_source_by_update[cand.update_id] = phi_source
        ctx.prefix_len_by_update[cand.update_id] = ctx.prefix_len
        pending_item = {
            "candidate": cand,
            "chain": chain,
            "event": done_event,
            "ready_event": ready_event,
            "side_start": side_start,
            "preview_event": preview_event,
            "lane": lane,
            "telemetry_lane": telemetry_lane,
            "candidate_delta_norm": candidate_delta_norm,
            "candidate_ready_round": self._candidate_ready_round(
                signal.source_round
            ),
            "trace_early_started": False,
            # Earliest exposure is the next decode round.  The configured
            # value is *additional* logical delay beyond that pipeline
            # minimum, so d=0 has one-round physical latency everywhere.
            "ready_round": self._ready_round(method, signal.source_round),
            "health_host": health_host,
            "health_generation": health_generation,
            "health_ok_direct": (
                bool(cand.numerical_ok)
                if not isinstance(cand.numerical_ok, torch.Tensor)
                else None
            ),
            "barrier_wait_cpu_us": 0.0,
        }
        ctx.pending.append(pending_item)
        return cand.update_id

    def launch_update_batch(
        self,
        request_ids: Sequence[str],
        signal: SourceBoundCandidateBatch,
    ) -> list[str]:
        """Launch one source-bound candidate per request as a true batch.

        Request ownership, AdamW moments, version checks, candidate lanes and
        publication remain independent.  Only the full-vocabulary loss,
        analytic tail backward and AdamW tensor expressions are vectorized on
        the common side stream.
        """

        request_ids = tuple(str(value) for value in request_ids)
        if len(request_ids) != signal.batch_size:
            raise ValueError("request ids and source-bound signal batch differ")
        signal.validate(self.shapes)

        # Fail before leasing telemetry lanes if a method does not implement
        # the common AdamW source-point contract.
        methods: list[MethodRuntime] = []
        generators = []
        for request_id in request_ids:
            ctx = self.requests[request_id]
            method = self.methods[ctx.slot_index]
            generator = method.common_candidate_generator()
            if generator is None:
                raise RuntimeError(
                    f"method {method.key!r} does not support source-bound batching"
                )
            methods.append(method)
            generators.append(generator)

        prepared: list[dict] = []
        selected_rows: list[int] = []
        for row, (request_id, method) in enumerate(zip(request_ids, methods)):
            ctx = self.requests[request_id]
            binding = _SourceBinding(
                source_round=signal.source_rounds[row],
                source_version=signal.source_versions[row],
            )
            active_version = self.bank.slots[ctx.slot_index].active_version
            reason = None
            if binding.source_round != ctx.round_id:
                reason = (
                    "source_round_mismatch:"
                    f"signal={binding.source_round}:canvas={ctx.round_id}"
                )
            elif binding.source_version != ctx.canvas_version:
                reason = (
                    "source_canvas_version_mismatch:"
                    f"signal={binding.source_version}:canvas={ctx.canvas_version}"
                )
            elif ctx.canvas_version != active_version:
                reason = (
                    "canvas_active_version_mismatch:"
                    f"canvas={ctx.canvas_version}:active={active_version}"
                )
            elif len(ctx.pending) >= self.config.async_.max_in_flight:
                reason = "max_in_flight"
            if reason is not None:
                self._reject_update_launch(ctx, binding, reason)
                continue

            telemetry_lane = self._acquire_update_telemetry_lane(ctx, binding)
            if self._update_telemetry_pool is not None and telemetry_lane is None:
                continue
            used_lanes = {int(item["lane"]) for item in ctx.pending}
            lane = next(
                index
                for index in range(self.config.async_.max_in_flight)
                if index not in used_lanes
            )
            phi_buf, grad_buf, delta_buf = self.bank.candidate_buffers(
                ctx.slot_index, lane
            )
            health_host = health_generation = None
            if self.side_stream is not None:
                health_host, health_generation = self.bank.prepare_candidate_health(
                    ctx.slot_index, ctx.request_epoch, lane
                )
            chain = UpdateEventChain(
                update_id=f"{request_id}-u{ctx.update_seq}",
                source_round=binding.source_round,
                source_version=binding.source_version,
            )
            ctx.update_seq += 1
            chain.mark("snapshot")
            chain.mark("teacher")
            prepared.append(
                {
                    "row": row,
                    "request_id": request_id,
                    "ctx": ctx,
                    "method": method,
                    "generator": generators[row],
                    "binding": binding,
                    "lane": lane,
                    "telemetry_lane": telemetry_lane,
                    "phi_buf": phi_buf,
                    "grad_buf": grad_buf,
                    "delta_buf": delta_buf,
                    "health_host": health_host,
                    "health_generation": health_generation,
                    "chain": chain,
                }
            )
            selected_rows.append(row)
        if not prepared:
            return []

        selected_signal = signal.select(selected_rows)
        selected_methods = [item["method"] for item in prepared]
        selected_generators = [item["generator"] for item in prepared]
        defer_state_advance = bool(
            getattr(selected_methods[0], "defer_state_advance", False)
        )
        if any(
            bool(getattr(method, "defer_state_advance", False))
            != defer_state_advance
            for method in selected_methods
        ):
            raise RuntimeError("candidate batch mixes optimizer advance policies")

        timing_refs: list[dict[str, object]] = []
        allocator_before = 0
        if self.side_stream is not None:
            pool = self._update_telemetry_pool
            assert pool is not None
            allocator_before = int(torch.cuda.memory_reserved(device=self.device))
            current = torch.cuda.current_stream(device=self.device)
            for item in prepared:
                telemetry_lane = int(item["telemetry_lane"])
                ready = pool.event(telemetry_lane, "ready_event")
                ready.record(current)
                self.side_stream.wait_event(ready)
                item["ready_event"] = ready
            try:
                with torch.cuda.stream(self.side_stream):
                    for item in prepared:
                        telemetry_lane = int(item["telemetry_lane"])
                        side_start = pool.event(telemetry_lane, "side_start")
                        side_start.record(self.side_stream)
                        item["side_start"] = side_start
                        item["chain"].mark("launch")
                    slot_indices = torch.tensor(
                        [item["ctx"].slot_index for item in prepared],
                        dtype=torch.int64,
                        device=self.device,
                    )
                    lanes = sorted({int(item["lane"]) for item in prepared})
                    if len(lanes) == 1:
                        lane = lanes[0]
                        phi_sources = self.bank.active.index_select(
                            0, slot_indices
                        )
                        self.bank.phi_source[:, lane].index_copy_(
                            0, slot_indices, phi_sources
                        )
                        forward_phi_sources = self.bank.forward_active.index_select(
                            0, slot_indices
                        )
                    else:
                        for item in prepared:
                            item["phi_buf"].copy_(
                                self.bank.read_active(item["ctx"].slot_index)
                            )
                        phi_sources = torch.stack(
                            [item["phi_buf"] for item in prepared], dim=0
                        )
                        forward_phi_sources = torch.stack(
                            [
                                self.bank.read_forward_active(
                                    item["ctx"].slot_index
                                )
                                for item in prepared
                            ],
                            dim=0,
                        )
                    for item in prepared:
                        telemetry_lane = int(item["telemetry_lane"])
                        preview = pool.event(telemetry_lane, "preview_ready")
                        preview.record(self.side_stream)
                        item["preview_event"] = preview
                        timing = {
                            "backward_start": pool.event(
                                telemetry_lane, "backward_start"
                            ),
                            "backward_end": pool.event(
                                telemetry_lane, "backward_end"
                            ),
                            "optimizer_end": pool.event(
                                telemetry_lane, "optimizer_end"
                            ),
                            "optimizer_step_out": pool.device_scalar(
                                telemetry_lane, "optimizer_step"
                            ),
                        }
                        timing_refs.append(timing)

                    if self.bank.exp_avg is None or self.bank.exp_avg_sq is None:
                        raise RuntimeError("batched AdamW has no resident bank state")
                    exp_avg = self.bank.exp_avg.index_select(0, slot_indices)
                    exp_avg_sq = self.bank.exp_avg_sq.index_select(0, slot_indices)
                    steps = torch.stack(
                        [
                            torch.as_tensor(
                                generator.state.step,
                                dtype=torch.int64,
                                device=phi_sources.device,
                            ).reshape(())
                            for generator in selected_generators
                        ]
                    )
                    candidates = selected_generators[0].candidate_batch(
                        selected_generators,
                        phi_sources,
                        forward_phi_sources,
                        selected_signal,
                        exp_avg=exp_avg,
                        exp_avg_sq=exp_avg_sq,
                        steps=steps,
                        defer_state_advance=defer_state_advance,
                        cuda_timing_refs=timing_refs,
                    )
                    if not defer_state_advance:
                        self.bank.exp_avg.index_copy_(0, slot_indices, exp_avg)
                        self.bank.exp_avg_sq.index_copy_(
                            0, slot_indices, exp_avg_sq
                        )
                        for row, generator in enumerate(selected_generators):
                            if isinstance(generator.state.step, torch.Tensor):
                                generator.state.step.copy_(steps[row])
                            else:
                                raise RuntimeError(
                                    "CUDA batched AdamW step state is not device-resident"
                                )
                        # The gathered batched moments are not the authoritative
                        # request state until these fixed-row copies complete.
                        # Re-record the leased end events so optimizer telemetry
                        # includes that state commit.
                        for timing in timing_refs:
                            timing["optimizer_end"].record(self.side_stream)

                    raw_gradient_batch = torch.stack(
                        [candidate.raw_gradient for candidate in candidates]
                    )
                    candidate_delta_batch = torch.stack(
                        [candidate.candidate_delta for candidate in candidates]
                    )
                    # Commit batched outputs to their fixed request/lane banks
                    # with one indexed copy per lane, not two kernels per
                    # request.  Different in-flight lanes remain disjoint.
                    if len(lanes) == 1:
                        lane = lanes[0]
                        self.bank.candidate_grad[:, lane].index_copy_(
                            0, slot_indices, raw_gradient_batch
                        )
                        self.bank.candidate_delta[:, lane].index_copy_(
                            0, slot_indices, candidate_delta_batch
                        )
                    else:
                        for lane in lanes:
                            batch_rows = [
                                row
                                for row, item in enumerate(prepared)
                                if int(item["lane"]) == lane
                            ]
                            row_index = torch.tensor(
                                batch_rows,
                                dtype=torch.int64,
                                device=phi_sources.device,
                            )
                            bank_slots = slot_indices.index_select(0, row_index)
                            self.bank.candidate_grad[:, lane].index_copy_(
                                0,
                                bank_slots,
                                raw_gradient_batch.index_select(0, row_index),
                            )
                            self.bank.candidate_delta[:, lane].index_copy_(
                                0,
                                bank_slots,
                                candidate_delta_batch.index_select(0, row_index),
                            )
                    delta_norms = torch.linalg.vector_norm(
                        candidate_delta_batch, dim=1
                    )
                    for row, (item, candidate) in enumerate(
                        zip(prepared, candidates)
                    ):
                        item["method"].after_batched_candidate(candidate)
                        candidate.raw_gradient = item["grad_buf"]
                        candidate.candidate_delta = item["delta_buf"]
                        candidate.phi_source = item["phi_buf"]
                        if candidate.fisher_snapshot is not None:
                            fisher_buf = self.bank.candidate_fisher_buffer(
                                item["ctx"].slot_index, item["lane"]
                            )
                            fisher_buf.copy_(candidate.fisher_snapshot)
                            candidate.fisher_snapshot = fisher_buf
                        telemetry_lane = int(item["telemetry_lane"])
                        norm_out = pool.device_scalar(
                            telemetry_lane, "candidate_delta_norm"
                        )
                        norm_out.copy_(delta_norms[row])
                        item["candidate_delta_norm"] = norm_out
                        numerical_ok = candidate.numerical_ok
                        if (
                            not isinstance(numerical_ok, torch.Tensor)
                            or not numerical_ok.is_cuda
                            or numerical_ok.numel() != 1
                        ):
                            raise ExactnessViolation(
                                "CUDA batched candidate health must be one "
                                "device scalar per request"
                            )
                        item["health_host"].copy_(
                            numerical_ok.reshape(()), non_blocking=True
                        )
                        item["candidate"] = candidate
                    for item in prepared:
                        telemetry_lane = int(item["telemetry_lane"])
                        done = pool.event(telemetry_lane, "candidate_end")
                        done.record(self.side_stream)
                        item["event"] = done
                    for value in vars(selected_signal).values():
                        if isinstance(value, torch.Tensor) and value.is_cuda:
                            value.record_stream(self.side_stream)
            except Exception:
                for item in prepared:
                    telemetry_lane = item.get("telemetry_lane")
                    if telemetry_lane is None:
                        continue
                    failed = pool.event(int(telemetry_lane), "candidate_end")
                    failed.record(self.side_stream)
                    self._defer_update_lane_release(int(telemetry_lane), failed)
                raise
            self.max_allocator_growth_bytes = max(
                self.max_allocator_growth_bytes,
                int(torch.cuda.memory_reserved(device=self.device)) - allocator_before,
            )
        else:
            for item in prepared:
                item["phi_buf"].copy_(
                    self.bank.read_active(item["ctx"].slot_index)
                )
                item["chain"].mark("launch")
            phi_sources = torch.stack([item["phi_buf"] for item in prepared])
            forward_phi_sources = torch.stack(
                [
                    self.bank.read_forward_active(item["ctx"].slot_index)
                    for item in prepared
                ]
            )
            exp_avg = torch.stack(
                [generator.state.exp_avg for generator in selected_generators]
            )
            exp_avg_sq = torch.stack(
                [generator.state.exp_avg_sq for generator in selected_generators]
            )
            steps = torch.tensor(
                [int(generator.state.step) for generator in selected_generators],
                dtype=torch.int64,
                device=phi_sources.device,
            )
            candidates = selected_generators[0].candidate_batch(
                selected_generators,
                phi_sources,
                forward_phi_sources,
                selected_signal,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                steps=steps,
                defer_state_advance=defer_state_advance,
            )
            if not defer_state_advance:
                for row, generator in enumerate(selected_generators):
                    generator.state.exp_avg.copy_(exp_avg[row])
                    generator.state.exp_avg_sq.copy_(exp_avg_sq[row])
                    generator.state.step = int(steps[row])
            for item, candidate in zip(prepared, candidates):
                item["method"].after_batched_candidate(candidate)
                item["grad_buf"].copy_(candidate.raw_gradient)
                item["delta_buf"].copy_(candidate.candidate_delta)
                candidate.raw_gradient = item["grad_buf"]
                candidate.candidate_delta = item["delta_buf"]
                candidate.phi_source = item["phi_buf"]
                if candidate.fisher_snapshot is not None:
                    fisher_buf = self.bank.candidate_fisher_buffer(
                        item["ctx"].slot_index, item["lane"]
                    )
                    fisher_buf.copy_(candidate.fisher_snapshot)
                    candidate.fisher_snapshot = fisher_buf
                item["candidate"] = candidate
                item["event"] = None
                item["candidate_delta_norm"] = None
                item["chain"].mark("done")

        update_ids: list[str] = []
        for item in prepared:
            ctx = item["ctx"]
            candidate = item["candidate"]
            candidate.update_id = item["chain"].update_id
            ctx.phi_source_by_update[candidate.update_id] = item["phi_buf"]
            ctx.prefix_len_by_update[candidate.update_id] = ctx.prefix_len
            pending_item = {
                "candidate": candidate,
                "chain": item["chain"],
                "event": item["event"],
                "ready_event": item.get("ready_event"),
                "side_start": item.get("side_start"),
                "preview_event": item.get("preview_event"),
                "lane": item["lane"],
                "telemetry_lane": item["telemetry_lane"],
                "candidate_delta_norm": item["candidate_delta_norm"],
                "candidate_ready_round": self._candidate_ready_round(
                    candidate.source_round
                ),
                "trace_early_started": False,
                "ready_round": self._ready_round(
                    item["method"], candidate.source_round
                ),
                "health_host": item["health_host"],
                "health_generation": item["health_generation"],
                "health_ok_direct": (
                    bool(candidate.numerical_ok)
                    if not isinstance(candidate.numerical_ok, torch.Tensor)
                    else None
                ),
                "barrier_wait_cpu_us": 0.0,
            }
            ctx.pending.append(pending_item)
            update_ids.append(candidate.update_id)
        return update_ids

    def _ready_round(self, method: MethodRuntime, source_round: int) -> int:
        earliest = self._candidate_ready_round(source_round)
        if method.publish_policy == PublishPolicy.FIXED_STRIDE_BARRIER:
            stride = self.config.update_stride
            return ((earliest + stride - 1) // stride) * stride
        return earliest

    def _candidate_ready_round(self, source_round: int) -> int:
        return source_round + 1 + self.config.async_.logical_delay_rounds

    def _candidate_health_ok(self, ctx: _RequestCtx, item: dict) -> bool:
        """Read health only after the candidate completion event is ready."""

        health_host = item.get("health_host")
        if health_host is None:
            health = item.get("health_ok_direct")
            if health is None:
                raise ExactnessViolation(
                    f"{ctx.request_id}: candidate has no completed health signal"
                )
            return bool(health)
        generation = item.get("health_generation")
        if generation is None:
            raise ExactnessViolation(
                f"{ctx.request_id}: candidate health generation is missing"
            )
        return self.bank.read_candidate_health(
            ctx.slot_index,
            ctx.request_epoch,
            int(item["lane"]),
            int(generation),
        )

    @staticmethod
    def _owned_trace_candidate(cand: CandidateUpdate) -> CandidateUpdate:
        """Snapshot fixed candidate lanes before they can be overwritten."""

        def owned(value):
            return (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
            )

        loss = replace(
            cand.loss,
            total=owned(cand.loss.total),
            distillation=owned(cand.loss.distillation),
            confidence=owned(cand.loss.confidence),
            proximal=owned(cand.loss.proximal),
            expected_accepted_prefix=owned(
                cand.loss.expected_accepted_prefix
            ),
        )
        return replace(
            cand,
            raw_gradient=owned(cand.raw_gradient),
            candidate_delta=owned(cand.candidate_delta),
            grad_norm=owned(cand.grad_norm),
            grad_clip_scale=owned(cand.grad_clip_scale),
            loss=loss,
            signal=None,
            phi_source=None,
            fisher_snapshot=None,
        )

    def _start_tts_early_trace(
        self,
        ctx: _RequestCtx,
        item: dict,
        ev: UpdatePollPoint,
    ) -> None:
        """Label the first query-ready TTS boundary without publishing."""

        if (
            item.get("trace_early_started")
            or not self.enable_replay_writer
            or self.config.trace.trace_capture_max_bytes <= 0
        ):
            return
        reservation = self._reserve_trace_label(
            ctx,
            source_prefix_len=ctx.prefix_len_by_update.get(
                item["candidate"].update_id
            ),
        )
        if reservation is None:
            return
        trace_stage_index = self._trace_records_by_request[ctx.request_id] - 1
        cand = self._owned_trace_candidate(item["candidate"])
        active_version = self.bank.slots[ctx.slot_index].active_version
        arrival_ev = replace(ev, active_version=active_version)
        arrival = self._arrival_context(ctx, cand, arrival_ev)
        phi_before = self.bank.read_active(ctx.slot_index).float().detach().clone()
        item["trace_early_started"] = True

        def start() -> None:
            phi_after = apply_delta_with_trust_region(
                phi_before,
                cand.candidate_delta,
                self.phi0,
                self.config.trust_region_radius,
            )
            self._start_replay_label(
                ctx,
                cand,
                arrival,
                phi_before,
                phi_after,
                ev.round_id,
                trace_reservation=reservation,
                trace_stage_index=trace_stage_index,
                actual_arrival_round=None,
                actual_phi_before=None,
                actual_phi_after=None,
                paired_tts_barrier=True,
            )

        self._run_on_trace_stream(
            start,
            phi_before,
            cand.raw_gradient,
            cand.candidate_delta,
            cand.loss.total,
            cand.loss.expected_accepted_prefix,
            arrival.rho_path,
            arrival.endpoint_distance,
            arrival.parameter_displacement,
            arrival.delta_z,
            arrival.source_z_raw,
            arrival.arrival_z_raw,
        )

    def _start_published_replay_trace(
        self,
        ctx: _RequestCtx,
        method: MethodRuntime,
        cand: CandidateUpdate,
        decision,
        l2_arrival_delta: torch.Tensor | None,
        arrival: ArrivalContext,
        phi_before: torch.Tensor,
        phi_after: torch.Tensor,
        arrival_round: int,
    ) -> None:
        reservation = self._reserve_trace_label(
            ctx,
            source_prefix_len=ctx.prefix_len_by_update.get(cand.update_id),
        )
        if reservation is None:
            return
        trace_stage_index = self._trace_records_by_request[ctx.request_id] - 1
        owned_cand = self._owned_trace_candidate(cand)
        owned_before = phi_before.detach().clone()
        owned_after = phi_after.detach().clone()
        l3_evaluation = None
        if self.config.method == "lc_transport":
            if decision.kind is not DecisionKind.TRANSPORT:
                # Only the complete joint transport decision is admissible as
                # L3 utility evidence.  Negative-control variants still write
                # ordinary replay labels but can never open the gate.
                l3_evaluation = None
            elif getattr(method, "variant", None) == "joint":
                if l2_arrival_delta is None:
                    raise ExactnessViolation(
                        "joint L3 trace is missing its arrival-state L2 preview"
                    )
                kappa = decision.damping_factor
                if kappa is None:
                    raise ExactnessViolation(
                        "joint L3 trace is missing its applied damping factor"
                    )
                l2_after = apply_delta_with_trust_region(
                    owned_before,
                    l2_arrival_delta.detach().clone() * kappa,
                    self.phi0,
                    self.config.trust_region_radius,
                )
                artifact = getattr(method, "artifact", None)
                l3_evaluation = {
                    "l2_phi_after": l2_after.detach().clone(),
                    "transport_evaluation_contract": (
                        "joint_fisher_transport_adamw_damping_v1"
                    ),
                    "transport_variant": "joint",
                    "transport_map_sha256": (
                        artifact.extra.get("transport_map_sha256")
                        if artifact is not None
                        else None
                    ),
                }
        self._run_on_trace_stream(
            lambda: self._start_replay_label(
                ctx,
                owned_cand,
                arrival,
                owned_before,
                owned_after,
                arrival_round,
                trace_reservation=reservation,
                trace_stage_index=trace_stage_index,
                actual_arrival_round=arrival_round,
                actual_phi_before=owned_before,
                actual_phi_after=owned_after,
                paired_tts_barrier=False,
                l3_evaluation=l3_evaluation,
            ),
            owned_before,
            owned_after,
            (
                l3_evaluation["l2_phi_after"]
                if l3_evaluation is not None
                else None
            ),
            owned_cand.raw_gradient,
            owned_cand.candidate_delta,
            owned_cand.loss.total,
            owned_cand.loss.expected_accepted_prefix,
            arrival.rho_path,
            arrival.endpoint_distance,
            arrival.parameter_displacement,
            arrival.delta_z,
            arrival.source_z_raw,
            arrival.arrival_z_raw,
            l2_arrival_delta,
        )

    # ---- hook 6: poll + publish -----------------------------------------------

    def on_update_poll(self, ev: UpdatePollPoint) -> Optional[int]:
        ctx = self.requests[ev.request_id]
        if ev.in_replay:
            return None
        self.bank.check_owner(ev.slot_index, ev.request_epoch, ctx.tenant_id_hash)
        bank_version = self.bank.slots[ctx.slot_index].active_version
        if ev.active_version != bank_version:
            raise ExactnessViolation(
                f"{ev.request_id} r{ev.round_id}: poll reported active version "
                f"{ev.active_version}, bank holds {bank_version}"
            )
        self._materialize_states(ctx)
        method = self.methods[ctx.slot_index]
        new_version: Optional[int] = None
        for item in list(ctx.pending):
            event = item["event"]
            current = (
                torch.cuda.current_stream(device=self.device)
                if event is not None
                else None
            )
            is_tts_barrier = (
                method.publish_policy == PublishPolicy.FIXED_STRIDE_BARRIER
            )
            if ev.round_id < item["candidate_ready_round"]:
                continue
            candidate_done = event is None or event.query()
            if item["ready_round"] > ev.round_id:
                # Before the fixed TTS barrier, query-only polling records the
                # candidate's true first-ready boundary for paired replay.  It
                # never waits and never publishes.
                if is_tts_barrier and candidate_done:
                    if current is not None:
                        current.wait_event(event)
                    if not self._candidate_health_ok(ctx, item):
                        cand = item["candidate"]
                        chain = item["chain"]
                        cand.failure_reason = "non_finite_candidate"
                        self._emit_discard(
                            ctx,
                            cand,
                            chain,
                            ev,
                            cand.failure_reason,
                            item=item,
                        )
                        self._abandon_tts_pair(
                            ctx, cand.update_id, cand.failure_reason
                        )
                        ctx.pending.remove(item)
                        self._forget_update(ctx, cand.update_id)
                        continue
                    self._start_tts_early_trace(ctx, item, ev)
                continue
            if not candidate_done:
                if method.publish_policy in (
                    PublishPolicy.BLOCKING,
                    PublishPolicy.FIXED_STRIDE_BARRIER,
                ):
                    # Sync-Fresh and TTS are baselines with an explicit
                    # synchronization contract.  They wait at their original
                    # legal boundary; moving ready_round would silently turn
                    # TTS into a different delayed algorithm.
                    wait_start = monotonic_us()
                    event.synchronize()
                    item["barrier_wait_cpu_us"] += monotonic_us() - wait_start
                    current.wait_event(event)
                    candidate_done = True
                else:
                    if (
                        self.constant_controller_delay is not None
                        and ev.round_id >= item["ready_round"]
                    ):
                        cand = item["candidate"]
                        chain = item["chain"]
                        self._emit_discard(
                            ctx,
                            cand,
                            chain,
                            ev,
                            "constant_controller_deadline_miss",
                            item=item,
                        )
                        ctx.pending.remove(item)
                        self._forget_update(ctx, cand.update_id)
                    continue  # async methods never block the boundary
            elif event is not None:
                current.wait_event(event)
            if not self._candidate_health_ok(ctx, item):
                cand = item["candidate"]
                chain = item["chain"]
                cand.failure_reason = "non_finite_candidate"
                self._emit_discard(
                    ctx,
                    cand,
                    chain,
                    ev,
                    cand.failure_reason,
                    item=item,
                )
                self._abandon_tts_pair(
                    ctx, cand.update_id, cand.failure_reason
                )
                ctx.pending.remove(item)
                self._forget_update(ctx, cand.update_id)
                continue
            if (
                is_tts_barrier
                and ev.round_id >= item["candidate_ready_round"]
            ):
                # If the candidate became ready only at the barrier, early and
                # actual arrival intentionally share this legal boundary.
                self._start_tts_early_trace(ctx, item, ev)
            if current is not None:
                # Fixed active/forward banks (and L3 shared Adam moments) may
                # change on the main stream only after every in-flight
                # candidate has copied its source.  This short stream wait is
                # the required side->main publication fence, never a host
                # synchronize and never a wait for the candidate backward.
                for pending_item in ctx.pending:
                    snapshot = pending_item.get("preview_event")
                    if snapshot is not None:
                        current.wait_event(snapshot)
            cand: CandidateUpdate = item["candidate"]
            if current is not None:
                for value in (
                    cand.raw_gradient,
                    cand.candidate_delta,
                    cand.loss.total,
                    cand.loss.expected_accepted_prefix,
                    cand.fisher_snapshot,
                    *(vars(cand.signal).values() if cand.signal is not None else ()),
                ):
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        value.record_stream(current)
            chain: UpdateEventChain = item["chain"]
            if chain.done_ts_us is None:
                chain.mark("done")
            active_version = self.bank.slots[ctx.slot_index].active_version
            arrival_ev = replace(ev, active_version=active_version)
            # All non-transport methods fail closed instead of silently
            # rebasing a candidate computed from an older adapter canvas.
            if (
                cand.source_version != active_version
                and self.config.method != "lc_transport"
            ):
                self._emit_discard(
                    ctx, cand, chain, ev, "version_conflict", item=item
                )
                self._abandon_tts_pair(ctx, cand.update_id, "version_conflict")
                ctx.pending.remove(item)
                self._forget_update(ctx, cand.update_id)
                continue
            arrival = self._arrival_context(ctx, cand, arrival_ev)
            controller_start = controller_end = None
            controller_cpu_start = monotonic_us()
            l2_arrival_state = None
            if (
                self.config.trace.trace_capture_max_bytes > 0
                and self.config.method == "lc_transport"
                and getattr(method, "variant", None) == "joint"
            ):
                # L3 may have two candidates in flight.  Its source-time
                # candidate preview is then not the L2 step that would be
                # produced from the optimizer state at this arrival.  Snapshot
                # that state before transport advances it and evaluate the raw
                # gradient through the same AdamW transform for the paired L2
                # counterfactual.
                l2_arrival_state = method.generator.state.clone()
            if current is not None and self._update_telemetry_pool is not None:
                pool = self._update_telemetry_pool
                telemetry_lane = item.get("telemetry_lane")
                if pool is None or telemetry_lane is None:
                    raise RuntimeError(
                        "CUDA candidate is missing its telemetry lane"
                    )
                controller_start = pool.event(
                    telemetry_lane, "controller_start"
                )
                controller_start.record(current)
            decision = method.decide(cand, arrival)
            l2_arrival_delta = None
            if l2_arrival_state is not None and decision.kind is DecisionKind.TRANSPORT:
                l2_arrival_delta = _l2_delta_from_arrival_state(
                    cand.raw_gradient,
                    l2_arrival_state,
                    method.generator.cfg.lr,
                    valid=cand.numerical_ok,
                    parameter=arrival.phi_active,
                    weight_decay=method.generator.cfg.weight_decay,
                )
            if current is not None and self._update_telemetry_pool is not None:
                controller_end = pool.event(
                    telemetry_lane, "controller_end"
                )
                controller_end.record(current)
            controller_cpu_us = monotonic_us() - controller_cpu_start
            if decision.kind in (DecisionKind.DISCARD, DecisionKind.VERSION_CONFLICT):
                self._emit_discard(
                    ctx,
                    cand,
                    chain,
                    ev,
                    decision.kind.value,
                    item=item,
                    controller_cpu_us=controller_cpu_us,
                    controller_start=controller_start,
                    controller_end=controller_end,
                    arrival=arrival,
                    decision=decision,
                )
                self._abandon_tts_pair(ctx, cand.update_id, decision.kind.value)
                ctx.pending.remove(item)
                self._forget_update(ctx, cand.update_id)
                continue
            phi_active = self.bank.read_active(ctx.slot_index).float()
            publish_start = publish_end = None
            if current is not None and self._update_telemetry_pool is not None:
                publish_start = pool.event(
                    telemetry_lane, "publish_start"
                )
                publish_start.record(current)
            published_delta = decision.published_delta
            if isinstance(cand.numerical_ok, torch.Tensor):
                published_delta = torch.where(
                    cand.numerical_ok,
                    published_delta,
                    torch.zeros_like(published_delta),
                )
            # The active bank has a fixed address and publish() overwrites it.
            # Real replay therefore needs an owned pre-publication snapshot;
            # otherwise phi_before aliases the just-published parameters and
            # the counterfactual utility label silently collapses to zero.
            phi_before = (
                phi_active.detach().clone()
                if self.config.trace.trace_capture_max_bytes > 0
                else phi_active
            )
            new_params = apply_delta_with_trust_region(
                phi_active,
                published_delta,
                self.phi0,
                self.config.trust_region_radius,
            )
            self.bank.write_staging(ctx.slot_index, ev.request_epoch, new_params)
            new_version = self.bank.publish(ctx.slot_index, ev.request_epoch)
            published_params = self.bank.read_active(ctx.slot_index)
            if current is not None and self._update_telemetry_pool is not None:
                publish_end = pool.event(
                    telemetry_lane, "publish_end"
                )
                publish_end.record(current)
            if self.config.trace.trace_capture_max_bytes > 0:
                if is_tts_barrier and item.get("trace_early_started"):
                    actual_before = phi_before.detach().clone()
                    actual_after = published_params.detach().clone()
                    self._run_on_trace_stream(
                        lambda: self._attach_tts_actual_replay(
                            ctx,
                            cand.update_id,
                            actual_before,
                            actual_after,
                            ev.round_id,
                        ),
                        actual_before,
                        actual_after,
                    )
                elif not is_tts_barrier:
                    self._start_published_replay_trace(
                        ctx,
                        method,
                        cand,
                        decision,
                        l2_arrival_delta,
                        arrival,
                        phi_before,
                        published_params,
                        ev.round_id,
                    )
            chain.apply_round = ev.round_id
            chain.mark("commit")
            chain.exposure_round = ev.round_id
            chain.mark("exposure")
            update_record = UpdateTelemetry(
                request_id=ev.request_id,
                update_id=cand.update_id,
                source_round=cand.source_round,
                source_version=cand.source_version,
                snapshot_ts_us=chain.snapshot_ts_us or 0.0,
                source_prefix_len=ctx.prefix_len_by_update[cand.update_id],
                active_version_at_arrival=active_version,
                staging_version=new_version,
                teacher_ts_us=chain.teacher_ts_us,
                launch_ts_us=chain.launch_ts_us,
                done_ts_us=chain.done_ts_us,
                commit_ts_us=chain.commit_ts_us,
                exposure_ts_us=chain.exposure_ts_us,
                published_version=new_version,
                grad_norm=(
                    float(cand.grad_norm)
                    if not isinstance(cand.grad_norm, torch.Tensor)
                    else 0.0
                ),
                candidate_delta_norm=0.0,
                decision=decision.kind.value,
                effective_delay_rounds=ev.round_id - cand.source_round,
                delay_tokens=arrival.delay_tokens,
                delay_wall_us=arrival.delay_wall_us,
                delay_versions=arrival.delay_versions,
                rho_path=_host_scalar_or_zero(arrival.rho_path),
                endpoint_distance=_host_scalar_or_zero(arrival.endpoint_distance),
                parameter_displacement=_host_scalar_or_zero(
                    arrival.parameter_displacement
                ),
                predicted_utility=(
                    decision.predicted_utility
                    if isinstance(decision.predicted_utility, float)
                    else None
                ),
                predicted_mismatch=(
                    decision.predicted_mismatch
                    if isinstance(decision.predicted_mismatch, float)
                    else None
                ),
                predicted_harm_probability=(
                    decision.predicted_harm_probability
                    if isinstance(decision.predicted_harm_probability, float)
                    else None
                ),
                threshold=decision.threshold,
                damping_factor=(
                    decision.damping_factor
                    if isinstance(decision.damping_factor, float)
                    else None
                ),
                grad_clip_scale=_host_scalar_or_zero(cand.grad_clip_scale),
                optimizer_step=(
                    0
                    if isinstance(cand.optimizer_step, torch.Tensor)
                    else int(cand.optimizer_step)
                ),
                barrier_wait_cpu_us=float(
                    item.get("barrier_wait_cpu_us", 0.0)
                ),
                controller_cpu_us=controller_cpu_us,
                prefix_feature_exact=ctx.prefix_feature_exact,
            )
            if isinstance(cand.numerical_ok, torch.Tensor):
                pool = self._update_telemetry_pool
                telemetry_lane = item.get("telemetry_lane")
                if pool is None or telemetry_lane is None:
                    raise RuntimeError(
                        "CUDA candidate is missing its telemetry lane"
                    )
                telemetry_event = pool.event(
                    telemetry_lane, "telemetry_ready"
                )
                telemetry_event.record(current)
                self.telemetry.emit_update_deferred(
                    update_record,
                    {
                        "event": telemetry_event,
                        "numerical_ok": cand.numerical_ok,
                        "grad_norm": cand.grad_norm,
                        "candidate_delta_norm": item[
                            "candidate_delta_norm"
                        ],
                        "source_training_loss": cand.loss.total,
                        "source_expected_accepted_prefix": (
                            cand.loss.expected_accepted_prefix
                        ),
                        "grad_clip_scale": cand.grad_clip_scale,
                        "rho_path": arrival.rho_path,
                        "endpoint_distance": arrival.endpoint_distance,
                        "parameter_displacement": arrival.parameter_displacement,
                        "gate_applied": decision.gate_applied,
                        "predicted_utility": decision.predicted_utility,
                        "predicted_mismatch": decision.predicted_mismatch,
                        "predicted_harm_probability": (
                            decision.predicted_harm_probability
                        ),
                        "damping_factor": decision.damping_factor,
                        "optimizer_step": cand.optimizer_step,
                        "ready_event": item.get("ready_event"),
                        "side_start": item.get("side_start"),
                        "candidate_end": item.get("event"),
                        **(cand.cuda_timing_ref or {}),
                        "controller_start": controller_start,
                        "controller_end": controller_end,
                        "publish_start": publish_start,
                        "publish_end": publish_end,
                        "release": self._update_lane_release_callback(item),
                    },
                )
            else:
                update_record.source_training_loss = float(cand.loss.total)
                update_record.source_expected_accepted_prefix = float(
                    cand.loss.expected_accepted_prefix
                )
                update_record.candidate_delta_norm = float(
                    torch.linalg.vector_norm(cand.candidate_delta)
                )
                self.telemetry.emit("update", update_record)
            ctx.pending.remove(item)
            self._forget_update(ctx, cand.update_id)
        # Bound the retained full-vocabulary teacher window to one round.  If a
        # backend polls before observing its signal, _start_replay_label stores
        # a bounded pending start and observe_signal resolves it later.
        ctx.signals_by_round.pop(ev.round_id, None)
        return new_version

    @staticmethod
    def _forget_update(ctx: _RequestCtx, update_id: str) -> None:
        ctx.phi_source_by_update.pop(update_id, None)
        ctx.prefix_len_by_update.pop(update_id, None)

    def _emit_discard(
        self,
        ctx: _RequestCtx,
        cand: CandidateUpdate,
        chain: UpdateEventChain,
        ev: UpdatePollPoint,
        reason: str,
        *,
        item: Optional[dict] = None,
        controller_cpu_us: float = 0.0,
        controller_start=None,
        controller_end=None,
        arrival: Optional[ArrivalContext] = None,
        decision=None,
    ) -> None:
        rho_path = endpoint_distance = parameter_displacement = 0.0
        if arrival is not None:
            rho_path = _host_scalar_or_zero(arrival.rho_path)
            endpoint_distance = _host_scalar_or_zero(arrival.endpoint_distance)
            parameter_displacement = _host_scalar_or_zero(
                arrival.parameter_displacement
            )
        record = UpdateTelemetry(
                request_id=ctx.request_id,
                update_id=cand.update_id,
                source_round=cand.source_round,
                source_version=cand.source_version,
                snapshot_ts_us=chain.snapshot_ts_us or 0.0,
                source_prefix_len=ctx.prefix_len_by_update[cand.update_id],
                active_version_at_arrival=self.bank.slots[
                    ctx.slot_index
                ].active_version,
                teacher_ts_us=chain.teacher_ts_us,
                launch_ts_us=chain.launch_ts_us,
                done_ts_us=chain.done_ts_us,
                decision="discard",
                failure_reason=reason,
                effective_delay_rounds=ev.round_id - cand.source_round,
                delay_tokens=arrival.delay_tokens if arrival is not None else 0,
                delay_wall_us=arrival.delay_wall_us if arrival is not None else 0.0,
                delay_versions=arrival.delay_versions if arrival is not None else 0,
                rho_path=rho_path,
                endpoint_distance=endpoint_distance,
                parameter_displacement=parameter_displacement,
                predicted_utility=(
                    _host_scalar_or_zero(decision.predicted_utility)
                    if decision is not None
                    and decision.predicted_utility is not None
                    else None
                ),
                predicted_mismatch=(
                    _host_scalar_or_zero(decision.predicted_mismatch)
                    if decision is not None
                    and decision.predicted_mismatch is not None
                    else None
                ),
                predicted_harm_probability=(
                    _host_scalar_or_zero(decision.predicted_harm_probability)
                    if decision is not None
                    and decision.predicted_harm_probability is not None
                    else None
                ),
                threshold=decision.threshold if decision is not None else None,
                damping_factor=(
                    _host_scalar_or_zero(decision.damping_factor)
                    if decision is not None
                    else None
                ),
                grad_clip_scale=_host_scalar_or_zero(cand.grad_clip_scale),
                optimizer_step=(
                    0
                    if isinstance(cand.optimizer_step, torch.Tensor)
                    else int(cand.optimizer_step)
                ),
                barrier_wait_cpu_us=float(
                    item.get("barrier_wait_cpu_us", 0.0)
                    if item is not None
                    else 0.0
                ),
                controller_cpu_us=controller_cpu_us,
                prefix_feature_exact=ctx.prefix_feature_exact,
            )
        if (
            item is not None
            and item.get("event") is not None
            and cand.candidate_delta.is_cuda
        ):
            telemetry_ready = (
                controller_end
                if controller_end is not None
                else item["event"]
            )
            if controller_end is not None:
                pool = self._update_telemetry_pool
                telemetry_lane = item.get("telemetry_lane")
                if pool is None or telemetry_lane is None:
                    raise RuntimeError(
                        "CUDA candidate is missing its telemetry lane"
                    )
                telemetry_ready = pool.event(
                    telemetry_lane, "telemetry_ready"
                )
                telemetry_ready.record(
                    torch.cuda.current_stream(cand.candidate_delta.device)
                )
            self.telemetry.emit_update_deferred(
                record,
                {
                    # The norm and optimizer-step scalars were written on the
                    # candidate side stream.  Controller decisions use the
                    # fixed final fence; pre-controller discards can consume
                    # them directly after candidate_end.
                    "event": telemetry_ready,
                    "numerical_ok": cand.numerical_ok,
                    "grad_norm": cand.grad_norm,
                    "candidate_delta_norm": item[
                        "candidate_delta_norm"
                    ],
                    "source_training_loss": cand.loss.total,
                    "source_expected_accepted_prefix": (
                        cand.loss.expected_accepted_prefix
                    ),
                    "grad_clip_scale": cand.grad_clip_scale,
                    "rho_path": arrival.rho_path if arrival is not None else None,
                    "endpoint_distance": (
                        arrival.endpoint_distance if arrival is not None else None
                    ),
                    "parameter_displacement": (
                        arrival.parameter_displacement if arrival is not None else None
                    ),
                    "predicted_utility": (
                        decision.predicted_utility if decision is not None else None
                    ),
                    "predicted_mismatch": (
                        decision.predicted_mismatch if decision is not None else None
                    ),
                    "predicted_harm_probability": (
                        decision.predicted_harm_probability
                        if decision is not None
                        else None
                    ),
                    "damping_factor": (
                        decision.damping_factor if decision is not None else None
                    ),
                    "optimizer_step": cand.optimizer_step,
                    "ready_event": item.get("ready_event"),
                    "side_start": item.get("side_start"),
                    "candidate_end": item.get("event"),
                    **(cand.cuda_timing_ref or {}),
                    "controller_start": controller_start,
                    "controller_end": controller_end,
                    "release": self._update_lane_release_callback(item),
                },
            )
        else:
            record.source_training_loss = float(cand.loss.total)
            record.source_expected_accepted_prefix = float(
                cand.loss.expected_accepted_prefix
            )
            record.candidate_delta_norm = float(
                torch.linalg.vector_norm(cand.candidate_delta)
            )
            self.telemetry.emit("update", record)

    def _materialize_states(self, ctx: _RequestCtx) -> None:
        import numpy as np

        for item in list(ctx.pending_states):
            event = item["event"]
            if event is not None and not event.query():
                continue
            if item["top_ids"].is_cuda:
                ctx.states.append(
                    _GpuTrajectoryState(
                        round_id=item["round_id"],
                        topk_token_ids=item["top_ids"],
                        topk_probs=item["top_probs"],
                        other_mass=item["other_mass"],
                        hidden_proj=item["hidden"],
                        event_sketch=item["events"],
                    )
                )
            else:
                ctx.states.append(
                    TrajectoryState(
                        round_id=item["round_id"],
                        topk_token_ids=item["top_ids"].numpy().astype(np.int32),
                        topk_probs=item["top_probs"].numpy().astype(np.float32),
                        other_mass=float(item["other_mass"].item()),
                        hidden_proj=item["hidden"].numpy().astype(np.float32),
                        event_sketch=item["events"].numpy().astype(np.float32),
                    )
                )
            ctx.pending_states.remove(item)

    def _start_replay_label(
        self,
        ctx: _RequestCtx,
        cand: CandidateUpdate,
        arrival: ArrivalContext,
        phi_before: torch.Tensor,
        phi_after: torch.Tensor,
        arrival_round: int,
        trace_reservation: int | None = None,
        trace_stage_index: int | None = None,
        actual_arrival_round: int | None = None,
        actual_phi_before: torch.Tensor | None = None,
        actual_phi_after: torch.Tensor | None = None,
        paired_tts_barrier: bool = False,
        prefix_feature_exact: bool | None = None,
        l3_evaluation: dict | None = None,
    ) -> None:
        from lightcone_spec.adapters.adapter_params import clip_gradient_global_norm
        from lightcone_spec.methods.base import evaluate_loss_and_grad

        if trace_reservation is None:
            trace_reservation = self._reserve_trace_label(
                ctx,
                source_prefix_len=ctx.prefix_len_by_update.get(cand.update_id),
            )
            if trace_reservation is None:
                return
        if trace_stage_index is None:
            trace_stage_index = self._trace_records_by_request[ctx.request_id] - 1
        if trace_stage_index < 0:
            raise RuntimeError("replay trace stage was not reserved")
        arrival_signal = ctx.signals_by_round.get(arrival_round)
        if arrival_signal is None:
            ctx.pending_replay_starts.append(
                {
                    "cand": cand,
                    "arrival": arrival,
                    "phi_before": phi_before.detach().clone(),
                    "phi_after": phi_after.detach().clone(),
                    "arrival_round": arrival_round,
                    "trace_reservation": trace_reservation,
                    "trace_stage_index": trace_stage_index,
                    "actual_arrival_round": actual_arrival_round,
                    "actual_phi_before": actual_phi_before,
                    "actual_phi_after": actual_phi_after,
                    "paired_tts_barrier": paired_tts_barrier,
                    "prefix_feature_exact": (
                        ctx.prefix_feature_exact
                        if prefix_feature_exact is None
                        else prefix_feature_exact
                    ),
                    "l3_evaluation": l3_evaluation,
                }
            )
            return
        fresh_loss, fresh_grad = evaluate_loss_and_grad(
            phi_before,
            arrival_signal,
            self.shapes,
            self.basis,
            confidence_loss_weight=self.config.confidence_loss_weight,
        )
        if fresh_grad is None:
            self._release_trace_label(ctx)
            return
        fresh_gradient_ok = torch.isfinite(fresh_grad).all() & torch.isfinite(
            fresh_loss.total
        )
        # Real replay is intentionally rank-0-only.  Calling the serving TP
        # collective here would deadlock because peer ranks do not enter this
        # trace-only path.  This mismatch diagnostic is therefore explicitly
        # writer-rank-local; online candidate/fresh-arrival gradients still
        # use the bound consensus hook before optimizer state changes.
        fresh_grad = torch.where(
            fresh_gradient_ok, fresh_grad, torch.zeros_like(fresh_grad)
        )
        fresh_grad, _ = clip_gradient_global_norm(fresh_grad, self.config.grad_clip)
        if not fresh_grad.is_cuda and not fresh_gradient_ok:
            if paired_tts_barrier:
                self._record_incomplete_tts_pair(
                    ctx,
                    cand.update_id,
                    "non_finite_fresh_gradient",
                    candidate_arrival_round=arrival_round,
                    actual_arrival_round=actual_arrival_round,
                )
            self._release_trace_label(ctx)
            return
        if actual_arrival_round is not None:
            if actual_phi_before is None or actual_phi_after is None:
                raise ValueError("actual replay arrival requires both phi snapshots")
        label = {
            "update_id": cand.update_id,
            "source_round": cand.source_round,
            # ``arrival_round`` remains the controller-feature arrival for
            # compatibility.  In paired TTS traces this is candidate-ready,
            # not the later fixed-stride publication.
            "arrival_round": arrival_round,
            "candidate_arrival_round": arrival_round,
            "actual_arrival_round": actual_arrival_round,
            "paired_tts_barrier": bool(paired_tts_barrier),
            "trace_stage_index": trace_stage_index,
            "trace_stage_count": (
                self.config.trace.trace_capture_max_records_per_request
            ),
            "trace_capture_sampling": self.config.trace.trace_capture_sampling,
            "phi_before": phi_before.detach().clone(),
            "phi_after": phi_after.detach().clone(),
            "actual_phi_before": (
                actual_phi_before.detach().clone()
                if actual_phi_before is not None
                else None
            ),
            "actual_phi_after": (
                actual_phi_after.detach().clone()
                if actual_phi_after is not None
                else None
            ),
            "l2_phi_after": (
                l3_evaluation["l2_phi_after"].detach().clone()
                if l3_evaluation is not None
                else None
            ),
            "transport_evaluation_contract": (
                l3_evaluation["transport_evaluation_contract"]
                if l3_evaluation is not None
                else None
            ),
            "transport_variant": (
                l3_evaluation["transport_variant"]
                if l3_evaluation is not None
                else None
            ),
            "transport_map_sha256": (
                l3_evaluation["transport_map_sha256"]
                if l3_evaluation is not None
                else None
            ),
            "candidate_delta": cand.candidate_delta.detach().clone(),
            "g_stale": cand.raw_gradient.detach().clone(),
            "g_fresh": fresh_grad.detach().clone(),
            "delta_z": (
                torch.zeros(387, device=phi_before.device)
                if arrival.delta_z is None
                else arrival.delta_z.detach().clone()
            ),
            "source_z_raw": (
                torch.zeros(387, device=phi_before.device)
                if arrival.source_z_raw is None
                else arrival.source_z_raw.detach().clone()
            ),
            "arrival_z_raw": (
                torch.zeros(387, device=phi_before.device)
                if arrival.arrival_z_raw is None
                else arrival.arrival_z_raw.detach().clone()
            ),
            "round_delay": arrival.delay_rounds,
            "token_delay": arrival.delay_tokens,
            "wall_us": arrival.delay_wall_us,
            "endpoint_distance": arrival.endpoint_distance,
            "rho_path": arrival.rho_path,
            "parameter_displacement": arrival.parameter_displacement,
            # The exact source snapshot travels with the arrival context so a
            # replay label deferred past publication never consults current or
            # final request length after the per-update map is retired.
            "source_prefix_len": arrival.source_prefix_len,
            "source_acceptance": cand.loss.expected_accepted_prefix.detach(),
            "source_training_loss": cand.loss.total.detach(),
            "source_grad_norm": cand.grad_norm,
            "prefix_feature_exact": (
                ctx.prefix_feature_exact
                if prefix_feature_exact is None
                else prefix_feature_exact
            ),
            "fresh_gradient_ok": fresh_gradient_ok,
            "utility_terms": [],
            "oracle_utility_terms": {
                kappa: []
                for kappa in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
            },
            "training_loss_gain_terms": [],
            "l2_utility_terms": [],
            "oracle_rounds_seen": set(),
            "actual_rounds_seen": set(),
            "horizon": 8,
            "trace_reservation": trace_reservation,
        }
        ctx.replay_labels.append(label)
        # Include the arrival window itself in U_r(H).
        self._accumulate_replay_utility(label, arrival_signal)

    def _attach_tts_actual_replay(
        self,
        ctx: _RequestCtx,
        update_id: str,
        phi_before: torch.Tensor,
        phi_after: torch.Tensor,
        actual_arrival_round: int,
    ) -> None:
        """Attach the real barrier publication to its trace-only early label."""

        for pending in ctx.pending_replay_starts:
            if pending["cand"].update_id != update_id:
                continue
            pending["actual_arrival_round"] = actual_arrival_round
            pending["actual_phi_before"] = phi_before.detach().clone()
            pending["actual_phi_after"] = phi_after.detach().clone()
            pending["prefix_feature_exact"] = bool(
                pending.get("prefix_feature_exact", True)
                and ctx.prefix_feature_exact
            )
            return
        for label in ctx.replay_labels:
            if label["update_id"] != update_id:
                continue
            if label["actual_arrival_round"] is not None:
                raise RuntimeError(f"duplicate actual replay arrival for {update_id}")
            label["actual_arrival_round"] = int(actual_arrival_round)
            label["actual_phi_before"] = phi_before.detach().clone()
            label["actual_phi_after"] = phi_after.detach().clone()
            label["prefix_feature_exact"] = bool(
                label["prefix_feature_exact"] and ctx.prefix_feature_exact
            )
            signal = ctx.signals_by_round.get(actual_arrival_round)
            if signal is not None:
                self._accumulate_replay_utility(label, signal)
                if self._replay_label_complete(label):
                    self._write_replay_label(ctx, label)
                    ctx.replay_labels.remove(label)
            return
        # A reserved early label should always be either pending on its first
        # signal or active.  Missing it would silently turn paired evidence
        # into two trajectories, so leave explicit failure evidence.
        self._record_incomplete_tts_pair(
            ctx,
            update_id,
            "early_label_missing_at_barrier",
            candidate_arrival_round=None,
            actual_arrival_round=actual_arrival_round,
        )

    def _advance_replay_labels(
        self, ctx: _RequestCtx, signal: TeacherSignal
    ) -> None:
        for label in list(ctx.replay_labels):
            self._accumulate_replay_utility(label, signal)
            if self._replay_label_complete(label):
                self._write_replay_label(ctx, label)
                ctx.replay_labels.remove(label)

    @staticmethod
    def _replay_label_complete(label: dict) -> bool:
        horizon = int(label["horizon"])
        return bool(
            label.get("actual_arrival_round") is not None
            and len(label["utility_terms"]) >= horizon
            and all(
                len(terms) >= horizon
                for terms in label["oracle_utility_terms"].values()
            )
            and (
                label.get("l2_phi_after") is None
                or len(label.get("l2_utility_terms", ())) >= horizon
            )
        )

    def _accumulate_replay_utility(
        self, label: dict, signal: TeacherSignal
    ) -> None:
        from lightcone_spec.methods.base import (
            evaluate_loss_and_grad,
            survival_weighted_acceptance,
        )

        r = int(signal.source_round)
        candidate_arrival = int(label["candidate_arrival_round"])
        horizon = int(label["horizon"])
        if (
            candidate_arrival <= r < candidate_arrival + horizon
            and r not in label["oracle_rounds_seen"]
        ):
            candidate_before = survival_weighted_acceptance(
                label["phi_before"],
                signal,
                self.shapes,
                self.basis,
                greedy=self.config.sampling.temperature == 0.0,
            )
            # Trace-only oracle upper bounds start at the first query-ready
            # boundary and never alter the live TTS trajectory.
            for kappa, terms in label["oracle_utility_terms"].items():
                oracle_phi = apply_delta_with_trust_region(
                    label["phi_before"],
                    label["candidate_delta"] * kappa,
                    self.phi0,
                    self.config.trust_region_radius,
                )
                oracle_acceptance = survival_weighted_acceptance(
                    oracle_phi,
                    signal,
                    self.shapes,
                    self.basis,
                    greedy=self.config.sampling.temperature == 0.0,
                )
                terms.append(
                    (oracle_acceptance - candidate_before).detach()
                )
            if label.get("l2_phi_after") is not None:
                l2_acceptance = survival_weighted_acceptance(
                    label["l2_phi_after"],
                    signal,
                    self.shapes,
                    self.basis,
                    greedy=self.config.sampling.temperature == 0.0,
                )
                label["l2_utility_terms"].append(
                    (l2_acceptance - candidate_before).detach()
                )
            label["oracle_rounds_seen"].add(r)

        actual_arrival = label.get("actual_arrival_round")
        if (
            actual_arrival is not None
            and actual_arrival <= r < actual_arrival + horizon
            and r not in label["actual_rounds_seen"]
        ):
            actual_before_loss, _ = evaluate_loss_and_grad(
                label["actual_phi_before"],
                signal,
                self.shapes,
                self.basis,
                confidence_loss_weight=self.config.confidence_loss_weight,
                need_grad=False,
            )
            actual_after_loss, _ = evaluate_loss_and_grad(
                label["actual_phi_after"],
                signal,
                self.shapes,
                self.basis,
                confidence_loss_weight=self.config.confidence_loss_weight,
                need_grad=False,
            )
            actual_before = survival_weighted_acceptance(
                label["actual_phi_before"],
                signal,
                self.shapes,
                self.basis,
                greedy=self.config.sampling.temperature == 0.0,
            )
            actual_after = survival_weighted_acceptance(
                label["actual_phi_after"],
                signal,
                self.shapes,
                self.basis,
                greedy=self.config.sampling.temperature == 0.0,
            )
            label["utility_terms"].append(
                (actual_after - actual_before).detach()
            )
            label["training_loss_gain_terms"].append(
                (actual_before_loss.total - actual_after_loss.total).detach()
            )
            label["actual_rounds_seen"].add(r)

    def _abandon_tts_pair(
        self, ctx: _RequestCtx, update_id: str, reason: str
    ) -> None:
        for collection in (ctx.replay_labels, ctx.pending_replay_starts):
            for item in list(collection):
                item_update_id = (
                    item.get("update_id")
                    or item.get("cand").update_id
                )
                if (
                    item_update_id != update_id
                    or not item.get("paired_tts_barrier")
                ):
                    continue
                self._record_incomplete_tts_pair(
                    ctx,
                    update_id,
                    reason,
                    candidate_arrival_round=(
                        item.get("candidate_arrival_round")
                        or item.get("arrival_round")
                    ),
                    actual_arrival_round=item.get("actual_arrival_round"),
                )
                collection.remove(item)
                self._release_trace_label(ctx)
                return

    def _record_incomplete_tts_pair(
        self,
        ctx: _RequestCtx,
        update_id: str,
        reason: str,
        *,
        candidate_arrival_round: int | None,
        actual_arrival_round: int | None,
    ) -> None:
        if not self.enable_replay_writer:
            return
        payload = {
            "schema_version": 2,
            "sequence_id": _sequence_group_from_request_id(ctx.request_id),
            "update_id": update_id,
            "reason": reason,
            "candidate_arrival_round": candidate_arrival_round,
            "actual_arrival_round": actual_arrival_round,
            "paired_tts_barrier": True,
            "prefix_feature_exact": ctx.prefix_feature_exact,
            "trace_clock": self.trace_clock,
        }
        self.telemetry.defer(lambda: self._append_incomplete_tts_pair(payload))

    def _append_incomplete_tts_pair(self, payload: dict) -> None:
        import json
        import os
        from pathlib import Path

        root = Path(self.config.trace.artifact_root) / "real-replay"
        root.mkdir(parents=True, exist_ok=True)
        with open(
            root / f"incomplete-paired-tts-p{os.getpid()}.jsonl",
            "a",
            encoding="utf-8",
        ) as fh:
            fh.write(json.dumps(payload) + "\n")

    def _write_replay_label(self, ctx: _RequestCtx, label: dict) -> None:
        if not self.enable_replay_writer:
            self._release_trace_label(ctx)
            return
        actual_published_utility = torch.stack(label["utility_terms"]).sum()
        training_loss_gain = torch.stack(
            label.get("training_loss_gain_terms")
            or [torch.zeros_like(actual_published_utility)]
        ).sum()
        transport_contract = label.get("transport_evaluation_contract")
        paired_l2_utility = None
        if transport_contract is not None:
            l2_terms = label.get("l2_utility_terms") or []
            if not l2_terms:
                # Never emit a partially populated schema-v3 L3 record: its
                # absence is actionable fail-closed evidence, whereas a zero
                # placeholder could spuriously open the production gate.
                self._release_trace_label(ctx)
                return
            paired_l2_utility = torch.stack(l2_terms).sum()
        oracle_terms = label.get("oracle_utility_terms")
        oracle_payload: dict[str, object] = {}
        if oracle_terms and all(oracle_terms.values()):
            utility_by_kappa = {
                float(kappa): torch.stack(terms).sum()
                for kappa, terms in oracle_terms.items()
            }
            full_candidate_utility = utility_by_kappa[1.0]
            oracle_l1_utility = full_candidate_utility.clamp_min(0.0)
            oracle_values = torch.stack(list(utility_by_kappa.values()))
            oracle_l2_utility, oracle_l2_index = oracle_values.max(dim=0)
            oracle_kappas = tuple(utility_by_kappa)
            oracle_kappa_tensor = torch.as_tensor(
                oracle_kappas,
                dtype=torch.float32,
                device=oracle_values.device,
            )[oracle_l2_index]
            oracle_payload = {
                "full_candidate_utility": full_candidate_utility,
                "oracle_l1_utility": oracle_l1_utility,
                "oracle_l2_utility": oracle_l2_utility,
                "oracle_l2_kappa": oracle_kappa_tensor,
                "utility_by_kappa": utility_by_kappa,
            }
        if not oracle_payload:
            # A controller label without the same-arrival raw-candidate replay is
            # policy-dependent and must never enter a real controller artifact.
            self._release_trace_label(ctx)
            return
        g_stale = label["g_stale"]
        g_fresh = label["g_fresh"]
        denom = torch.linalg.vector_norm(g_fresh).clamp_min(1e-12)
        mismatch = torch.linalg.vector_norm(g_stale - g_fresh) / denom
        cosine = torch.nn.functional.cosine_similarity(
            g_stale.view(1, -1), g_fresh.view(1, -1)
        )[0]
        delta_g = (g_fresh - g_stale).detach().float()
        tensor_values = {
            "delta_g": delta_g,
            "delta_z": label["delta_z"].detach().float(),
            "source_z_raw": label["source_z_raw"].detach().float(),
            "arrival_z_raw": label["arrival_z_raw"].detach().float(),
        }
        tensor_bytes = sum(
            value.numel() * value.element_size() for value in tensor_values.values()
        )
        reserved_bytes = tensor_bytes + (64 << 10)
        if "trace_reservation" not in label:
            # Compatibility for direct unit callers. Normal runtime labels
            # reserve quota before cloning any P-sized tensor.
            reservation = self._reserve_trace_label(
                ctx,
                reserved_bytes=reserved_bytes,
                source_prefix_len=int(label.get("source_prefix_len", ctx.prefix_len)),
            )
            if reservation is None:
                return
        sequence_id = _sequence_group_from_request_id(ctx.request_id)
        evaluation_pair_id = _evaluation_pair_from_request_id(ctx.request_id)
        trace_stage_index = int(label.get("trace_stage_index", 0))
        trace_stage_count = int(
            label.get(
                "trace_stage_count",
                self.config.trace.trace_capture_max_records_per_request,
            )
        )
        trace_capture_sampling = str(
            label.get(
                "trace_capture_sampling",
                self.config.trace.trace_capture_sampling,
            )
        )
        if not 0 <= trace_stage_index < trace_stage_count:
            self._release_trace_label(ctx)
            raise RuntimeError("replay trace stage index is outside its stage count")
        update_id = label["update_id"]
        candidate_arrival_round = int(
            label.get("candidate_arrival_round", label["arrival_round"])
        )
        actual_arrival_round = int(
            label.get("actual_arrival_round", label["arrival_round"])
        )
        paired_tts_barrier = bool(label.get("paired_tts_barrier", False))
        prefix_feature_exact = bool(label.get("prefix_feature_exact", True))
        fresh_gradient_ok = label.get("fresh_gradient_ok", True)
        metadata = {
            name: label[name]
            for name in (
                "source_round",
                "arrival_round",
                "round_delay",
                "token_delay",
                "wall_us",
                "endpoint_distance",
                "rho_path",
                "parameter_displacement",
                "source_prefix_len",
                "source_acceptance",
                "source_training_loss",
                "source_grad_norm",
            )
        }
        ready = None
        if actual_published_utility.is_cuda:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(actual_published_utility.device))

        def write_trace() -> None:
            import json
            import os
            from pathlib import Path

            from lightcone_spec.locking.hashing import sha256_file

            if not bool(fresh_gradient_ok):
                if paired_tts_barrier:
                    self._append_incomplete_tts_pair(
                        {
                            "schema_version": 2,
                            "sequence_id": sequence_id,
                            "update_id": update_id,
                            "reason": "non_finite_fresh_gradient",
                            "candidate_arrival_round": candidate_arrival_round,
                            "actual_arrival_round": actual_arrival_round,
                            "paired_tts_barrier": True,
                            "prefix_feature_exact": prefix_feature_exact,
                            "trace_clock": self.trace_clock,
                        }
                    )
                return

            payload = {
                "schema_version": 3,
                "sequence_id": sequence_id,
                "evaluation_pair_id": evaluation_pair_id,
                "trace_stage_index": trace_stage_index,
                "trace_stage_count": trace_stage_count,
                "trace_capture_sampling": trace_capture_sampling,
                "evaluation_concurrency": self.config.runtime.concurrency,
                "update_id": update_id,
                "provenance_method": self.config.method,
                "utility_metric": "survival_weighted_accepted_prefix_v1",
                "controller_label_source": "full_candidate_utility",
                "training_loss_gain": float(training_loss_gain),
                **{name: float(value) for name, value in metadata.items()},
                "source_round": int(metadata["source_round"]),
                "arrival_round": int(metadata["arrival_round"]),
                "candidate_arrival_round": candidate_arrival_round,
                "actual_arrival_round": actual_arrival_round,
                "paired_tts_barrier": paired_tts_barrier,
                "prefix_feature_exact": prefix_feature_exact,
                "trace_clock": self.trace_clock,
                "fresh_gradient_scope": "writer_rank_local_v1",
                "actual_published_utility": float(actual_published_utility),
                "full_candidate_utility": float(
                    oracle_payload["full_candidate_utility"]
                ),
                "relative_gradient_mismatch": float(mismatch),
                "cosine": float(cosine),
                "harmful": int(
                    float(oracle_payload["full_candidate_utility"]) < 0.0
                ),
                **{name: value.cpu() for name, value in tensor_values.items()},
            }
            if transport_contract is not None:
                payload.update(
                    {
                        # The actual publication was the complete joint L3
                        # decision.  The paired L2 value is evaluated over the
                        # same future signals and same pre-publication state,
                        # changing only transport to the raw-gradient AdamW
                        # preview with the identical damping factor.
                        "transported_candidate_utility": float(
                            actual_published_utility
                        ),
                        "paired_l2_utility": float(paired_l2_utility),
                        "transport_evaluation_contract": transport_contract,
                        "transport_variant": label.get("transport_variant"),
                        "transport_map_sha256": label.get(
                            "transport_map_sha256"
                        ),
                    }
                )
            if oracle_payload:
                payload.update(
                    {
                        "oracle_l1_utility": float(
                            oracle_payload["oracle_l1_utility"]
                        ),
                        "oracle_l2_utility": float(
                            oracle_payload["oracle_l2_utility"]
                        ),
                        "oracle_l2_kappa": float(
                            oracle_payload["oracle_l2_kappa"]
                        ),
                        "utility_by_kappa": {
                            str(kappa): float(value)
                            for kappa, value in oracle_payload[
                                "utility_by_kappa"
                            ].items()
                        },
                    }
                )
            root = Path(self.config.trace.artifact_root) / "real-replay"
            root.mkdir(parents=True, exist_ok=True)
            safe_id = update_id.replace("/", "_")
            path = root / f"p{os.getpid()}-{safe_id}.pt"
            torch.save(payload, path)
            size = path.stat().st_size
            with open(
                root / f"index-p{os.getpid()}.jsonl", "a", encoding="utf-8"
            ) as fh:
                fh.write(
                    json.dumps(
                        {
                            "path": path.name,
                            "sha256": sha256_file(path),
                            "bytes": size,
                            "sequence_id": sequence_id,
                            "update_id": update_id,
                            "parameter_layout_sha256": (
                                self.parameter_layout_sha256
                            ),
                        }
                    )
                    + "\n"
                )

        self.telemetry.defer(write_trace, ready)
        self._release_trace_label(ctx)

    def _reserve_trace_label(
        self,
        ctx: _RequestCtx,
        *,
        reserved_bytes: int | None = None,
        source_prefix_len: int | None = None,
    ) -> int | None:
        """Reserve disk quota and one live-label slot before P-sized clones."""
        if not self.enable_replay_writer:
            return None
        request_records = self._trace_records_by_request.get(ctx.request_id, 0)
        if self._trace_live_by_request.get(ctx.request_id, 0) >= 1:
            return None
        if (
            request_records
            >= self.config.trace.trace_capture_max_records_per_request
        ):
            return None
        if not self._trace_stage_due(ctx, source_prefix_len=source_prefix_len):
            return None
        if reserved_bytes is None:
            reserved_bytes = self._default_trace_reservation_bytes()
        if (
            self._trace_bytes_written + reserved_bytes
            > self.config.trace.trace_capture_max_bytes
        ):
            return None
        self._trace_bytes_written += reserved_bytes
        self._trace_records_by_request[ctx.request_id] = request_records + 1
        self._trace_live_by_request[ctx.request_id] = 1
        return int(reserved_bytes)

    def _trace_stage_due(
        self,
        ctx: _RequestCtx,
        *,
        source_prefix_len: int | None = None,
    ) -> bool:
        """Return whether the next bounded trace phase has been reached.

        ``first`` preserves the schema-v1 recorder. ``staged`` spreads the
        request quota across early/middle/late output progress.  The final
        target is 80% so the fixed eight-round utility horizon can finish
        before a max-length request is retired.
        """

        if self.config.trace.trace_capture_sampling != "staged":
            return True
        total = int(self.config.trace.trace_capture_max_records_per_request)
        index = int(self._trace_records_by_request.get(ctx.request_id, 0))
        if index >= total:
            return False
        if total <= 1:
            fraction = 0.0
        elif total == 3:
            fraction = (0.0, 0.5, 0.8)[index]
        else:
            fraction = 0.8 * index / (total - 1)
        start = ctx.trace_start_prefix_len
        if start is None:
            # The first proposal establishes the origin and is the early phase.
            return index == 0
        prefix = ctx.prefix_len if source_prefix_len is None else source_prefix_len
        generated = max(int(prefix) - int(start), 0)
        target = fraction * int(self.config.sampling.max_new_tokens)
        return generated >= target

    def _release_trace_label(self, ctx: _RequestCtx) -> None:
        self._trace_live_by_request.pop(ctx.request_id, None)

    def _arrival_context(
        self, ctx: _RequestCtx, cand: CandidateUpdate, ev: UpdatePollPoint
    ) -> ArrivalContext:
        from lightcone_spec.trajectory.distance import d_z

        by_round = {s.round_id: s for s in ctx.states}
        src = cand.source_round
        end = max((r for r in by_round if r <= ev.round_id), default=src)
        gpu_states = bool(by_round) and isinstance(
            next(iter(by_round.values())), _GpuTrajectoryState
        )
        zero = torch.zeros((), device=self.device) if gpu_states else 0.0
        rho, endp, delta_z = zero, zero, None
        source_z_raw, arrival_z_raw = None, None
        if src in by_round and end in by_round and end > src:
            distance = self._gpu_distance if gpu_states else (
                lambda a, b: d_z(a, b, self.weights)
            )
            rho = sum(
                distance(by_round[j - 1], by_round[j])
                for j in range(src + 1, end + 1)
                if j - 1 in by_round and j in by_round
            )
            endp = distance(by_round[src], by_round[end])
            if gpu_states:
                source_z_raw = self._gpu_z_raw(by_round[src])
                arrival_z_raw = self._gpu_z_raw(by_round[end])
                source_z = self._gpu_z_normalize(source_z_raw)
                arrival_z = self._gpu_z_normalize(arrival_z_raw)
                delta_z = arrival_z - source_z
            else:
                source_z_raw = torch.from_numpy(self.zvec.raw(by_round[src])).float()
                arrival_z_raw = torch.from_numpy(self.zvec.raw(by_round[end])).float()
                delta_z = torch.from_numpy(
                    self.zvec.delta_z(by_round[src], by_round[end])
                ).float()
        phi_active = self.bank.read_active(ctx.slot_index).float()
        phi_src = ctx.phi_source_by_update.get(
            cand.update_id, torch.zeros_like(phi_active)
        )
        src_wall = ctx.wall_by_round.get(src, monotonic_us())
        return ArrivalContext(
            arrival_round=ev.round_id,
            active_version=ev.active_version,
            phi_active=phi_active,
            delay_rounds=ev.round_id - src,
            delay_tokens=ctx.prefix_len
            - ctx.prefix_len_by_update[cand.update_id],
            delay_wall_us=monotonic_us() - src_wall,
            delay_versions=ev.active_version - cand.source_version,
            rho_path=rho,
            endpoint_distance=endp,
            parameter_displacement=(
                torch.linalg.vector_norm(phi_active - phi_src)
                if phi_active.is_cuda
                else float(torch.linalg.vector_norm(phi_active - phi_src))
            ),
            delta_z=delta_z,
            source_z_raw=source_z_raw,
            arrival_z_raw=arrival_z_raw,
            source_prefix_len=ctx.prefix_len_by_update[cand.update_id],
            source_acceptance=cand.loss.expected_accepted_prefix,
            source_training_loss=cand.loss.total,
            source_grad_norm=cand.grad_norm,
        )

    def _gpu_distance(
        self, a: _GpuTrajectoryState, b: _GpuTrajectoryState
    ) -> torch.Tensor:
        """Exact top-k+other JSD and frozen clock norms on CUDA."""
        ids = torch.cat((a.topk_token_ids.long(), b.topk_token_ids.long()))
        counts = (ids[:, None] == ids[None, :]).sum(dim=1).float()
        pa = (
            (ids[:, None] == a.topk_token_ids.long()[None, :]).float()
            @ a.topk_probs.float()
        ) / counts
        pb = (
            (ids[:, None] == b.topk_token_ids.long()[None, :]).float()
            @ b.topk_probs.float()
        ) / counts
        pa = torch.cat((pa, a.other_mass.float().view(1))).clamp_min(1e-12)
        pb = torch.cat((pb, b.other_mass.float().view(1))).clamp_min(1e-12)
        pa = pa / pa.sum()
        pb = pb / pb.sum()
        midpoint = 0.5 * (pa + pb)
        js = 0.5 * (pa * torch.log(pa / midpoint)).sum() + 0.5 * (
            pb * torch.log(pb / midpoint)
        ).sum()

        def normalize(value, mean_key, std_key):
            mean = self._distance_tensor_cache[mean_key]
            std = self._distance_tensor_cache[std_key]
            if mean is None or std is None:
                return value.float()
            n = min(value.numel(), mean.numel())
            return (value[:n].float() - mean[:n]) / std[:n].clamp_min(1e-8)

        ha = normalize(a.hidden_proj, "hidden_mean", "hidden_std")
        hb = normalize(b.hidden_proj, "hidden_mean", "hidden_std")
        ea = normalize(a.event_sketch, "event_mean", "event_std")
        eb = normalize(b.event_sketch, "event_mean", "event_std")
        n_event = min(ea.numel(), eb.numel())
        h_term = (ha - hb).square().sum() / 128.0
        e_term = (ea[:n_event] - eb[:n_event]).square().sum()
        return torch.sqrt(
            float(self.weights.a_p) * js.clamp_min(0.0)
            + float(self.weights.a_h) * h_term
            + float(self.weights.a_e) * e_term
        )

    def _gpu_z_raw(self, state: _GpuTrajectoryState) -> torch.Tensor:
        if self._sketch_bucket is None or self._sketch_sign is None:
            return torch.zeros(387, device=self.device)
        sketch = torch.zeros(self.zvec.sketch.dim, device=self.device)
        ids = state.topk_token_ids.long()
        sketch.scatter_add_(
            0,
            self._sketch_bucket[ids],
            self._sketch_sign[ids] * state.topk_probs.float(),
        )
        events = torch.zeros(3, device=self.device)
        n = min(3, state.event_sketch.numel())
        events[:n] = state.event_sketch[:n]
        return torch.cat((sketch, state.hidden_proj.float(), events))

    def _gpu_z_normalize(self, raw: torch.Tensor) -> torch.Tensor:
        if self._z_mean is None or self._z_std is None:
            return raw
        return (raw - self._z_mean) / self._z_std.clamp_min(1e-8)
