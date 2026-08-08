"""Allocation-bounded round telemetry for native speculative baselines.

This observer is intentionally not an adaptation manager.  It owns no tail
parameters, proposal signals, adapter versions, optimizer state or correction
callback.  Worker glue supplies only request ids, exact prefix lengths and the
native scalar length tensors that already exist after verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Sequence

import torch

from lightcone_spec.sglang_bridge.telemetry import (
    CudaTelemetryLanePool,
    RoundTelemetry,
    TelemetrySink,
)
from lightcone_spec.sglang_bridge.hooks import rng_substream_identity


def evidence_only_static_bypass(config) -> tuple[bool, str | None]:
    """Return whether native speculation can replace candidate generation.

    This is deliberately narrower than "the current gate rejected an update".
    Candidate work can be skipped only when a frozen artifact proves that the
    gate discards every reachable arrival and no trace requires the candidate
    itself.  Ordinary L1 still performs backward/Adam before its per-update
    decision, exactly as the method definition requires.
    """

    if config.method == "static":
        if int(config.trace.trace_capture_max_bytes) != 0:
            from lightcone_spec.exit_codes import ConfigError

            raise ConfigError(
                "static speculative telemetry cannot capture candidate traces; "
                "set trace_capture_max_bytes=0 or use a candidate-producing "
                "trace method"
            )
        return True, "static"
    if config.method != "lc_gate":
        return False, None
    artifact_path = getattr(config.controller, "artifact_path", None)
    if not artifact_path:
        return False, None

    from lightcone_spec.controller.artifact import load_bound_controller_artifact
    from lightcone_spec.methods.registry import validate_controller_artifact

    # A constant-discard shortcut is still an L1 decision.  Loading only the
    # JSON/hash would allow an artifact from another model pair or update tier
    # to silently turn a live run into Static.  Reuse the normal controller
    # binding/validation path before trusting its evidence.
    # Do not use ``getattr(config, "tail_layout_mode",
    # config.weight_update_mode)`` here: Python evaluates the default argument
    # eagerly, so legacy/minimal config objects without ``weight_update_mode``
    # fail even when ``tail_layout_mode`` is present.  Resolve the migration
    # aliases in sequence and leave canonical validation to the artifact
    # loader below.
    layout_mode = getattr(config, "tail_layout_mode", None)
    if layout_mode is None:
        layout_mode = getattr(config, "weight_update_mode", None)
    if layout_mode is None:
        layout_mode = getattr(config, "trainable_scope", "output_residual")
    artifact = load_bound_controller_artifact(
        artifact_path,
        model_pair_id=config.model.pair_id,
        weight_update_mode=layout_mode,
    )
    validate_controller_artifact(config, artifact)
    if int(config.trace.trace_capture_max_bytes) != 0:
        return False, None
    if artifact.gate_discard_all:
        return True, "controller_gate_discard_all"
    # A per-delay constant profile is safe only *after* a candidate's actual
    # arrival delay is known.  CUDA contention can make that delay exceed the
    # earliest legal round, so replacing the whole run using its planned delay
    # would silently discard candidates that L1 may apply at a later bucket.
    return False, None


@dataclass
class _ObservedBatch:
    lane: int | None
    request_ids: tuple[str, ...]
    prefix_before: tuple[int, ...] | torch.Tensor
    draft_tokens: int
    batch_size: int
    wall_start_us: float
    sampling_seeds: tuple[int | None, ...]
    greedy_flags: tuple[bool, ...]
    cpu_stage_us: dict[str, float] = field(default_factory=dict)
    events: dict[str, object] = field(default_factory=dict)
    commit_lens_cpu: torch.Tensor | None = None
    verify_lens_cpu: torch.Tensor | None = None
    new_seq_lens_cpu: torch.Tensor | None = None
    algorithmic_censored_cpu: torch.Tensor | None = None


class StaticSpeculativeObserver:
    """Observe a native DSpark/DFlash/EAGLE round without changing it.

    CUDA mode uses a bounded ring of pinned host counters and reusable events.
    A lane is returned only after the telemetry worker has consumed the last
    row for that batch, so no hot-path allocation or buffer reuse race is
    hidden by an implicit synchronization.
    """

    _STAGES = ("draft_start", "draft_end", "verify_end", "accept_end")
    _HOST_DTYPES = (torch.int32, torch.int64)

    def __init__(
        self,
        *,
        telemetry: TelemetrySink,
        device: str,
        offered_concurrency: int,
        max_batch_size: int,
        lane_count: int = 64,
    ) -> None:
        if offered_concurrency <= 0 or max_batch_size <= 0 or lane_count <= 0:
            raise ValueError(
                "static observer concurrency, batch capacity and lane count "
                "must be positive"
            )
        self.telemetry = telemetry
        self.device = str(device)
        self.offered_concurrency = int(offered_concurrency)
        self.max_batch_size = int(max_batch_size)
        self.lane_count = int(lane_count)
        self._cuda = self.device.startswith("cuda")
        self._round_of: dict[str, int] = {}
        self._active: _ObservedBatch | None = None
        self._lane_pool: CudaTelemetryLanePool | None = None
        self.controller_static_fallback = False
        self.static_bypass_reason: str | None = None
        if self._cuda:
            self._lane_pool = CudaTelemetryLanePool(
                device=self.device,
                max_batch_size=self.max_batch_size,
                lane_count=self.lane_count,
                event_names=(*self._STAGES, "telemetry_ready"),
                host_names=(
                    "prefix",
                    "commit",
                    "verify",
                    "new_seq",
                    "algorithmic_censored",
                ),
            )

    def _record_stage(self, state: _ObservedBatch, name: str) -> None:
        if name not in self._STAGES:
            raise ValueError(f"unknown static telemetry stage {name!r}")
        state.cpu_stage_us[name] = time.monotonic() * 1e6
        if self._cuda:
            assert self._lane_pool is not None
            event = self._lane_pool.event(int(state.lane), name)
            event.record(torch.cuda.current_stream(self.device))
            state.events[name] = event

    def _capture_prefix_lens(
        self, prefix_lens, batch_size: int, lane: int | None
    ) -> tuple[int, ...] | torch.Tensor:
        if isinstance(prefix_lens, torch.Tensor):
            if prefix_lens.is_cuda:
                if not self._cuda or lane is None:
                    raise TypeError("CPU static observer received device prefix")
                if prefix_lens.ndim != 1 or int(prefix_lens.numel()) != batch_size:
                    raise ValueError(
                        f"prefix length count {prefix_lens.numel()} != batch size "
                        f"{batch_size}"
                    )
                if prefix_lens.dtype not in self._HOST_DTYPES:
                    raise TypeError(
                        "prefix lengths must use int32 or int64, got "
                        f"{prefix_lens.dtype}"
                    )
                assert self._lane_pool is not None
                host = self._lane_pool.host(
                    lane, "prefix", prefix_lens.dtype, batch_size
                )
                # The telemetry_ready event is recorded after verification and
                # also fences this start-of-round nonblocking copy.  No hot-path
                # D2H synchronization is needed when seq_lens_cpu is disabled.
                host.copy_(prefix_lens, non_blocking=True)
                return host
            values = prefix_lens.tolist()
        else:
            values = list(prefix_lens)
        if len(values) != batch_size:
            raise ValueError(
                f"prefix length count {len(values)} != batch size {batch_size}"
            )
        result = tuple(int(value) for value in values)
        if any(value < 0 for value in result):
            raise ValueError("prefix lengths must be non-negative")
        return result

    def begin_round(
        self,
        *,
        request_ids: Sequence[str],
        prefix_lens,
        draft_tokens: int,
        sampling_seeds: Sequence[int | None] | None = None,
        greedy_flags: Sequence[bool] | None = None,
    ) -> None:
        if self._active is not None:
            raise RuntimeError("static observer already has an active batch")
        request_ids = tuple(str(value) for value in request_ids)
        batch_size = len(request_ids)
        if batch_size <= 0 or batch_size > self.max_batch_size:
            raise RuntimeError(
                f"static telemetry batch {batch_size} exceeds capacity "
                f"{self.max_batch_size}"
            )
        if len(set(request_ids)) != batch_size:
            raise ValueError("static telemetry request ids must be unique")
        sampling_seeds = tuple(
            None if value is None else int(value)
            for value in (
                sampling_seeds
                if sampling_seeds is not None
                else (None,) * batch_size
            )
        )
        greedy_flags = tuple(
            bool(value)
            for value in (
                greedy_flags
                if greedy_flags is not None
                else (True,) * batch_size
            )
        )
        if len(sampling_seeds) != batch_size or len(greedy_flags) != batch_size:
            raise ValueError("static telemetry RNG metadata must match batch size")
        if draft_tokens <= 0:
            raise ValueError("draft_tokens must be positive")
        lane = None
        if self._cuda:
            assert self._lane_pool is not None
            lane = self._lane_pool.acquire()
        try:
            prefix_before = self._capture_prefix_lens(
                prefix_lens, batch_size, lane
            )
        except Exception:
            if lane is not None:
                assert self._lane_pool is not None
                self._lane_pool.release(int(lane))
            raise
        now = time.monotonic() * 1e6
        self._active = _ObservedBatch(
            lane=lane,
            request_ids=request_ids,
            prefix_before=prefix_before,
            draft_tokens=int(draft_tokens),
            batch_size=batch_size,
            wall_start_us=now,
            sampling_seeds=sampling_seeds,
            greedy_flags=greedy_flags,
        )
        for rid in request_ids:
            self._round_of.setdefault(rid, 0)
        self._record_stage(self._active, "draft_start")

    def record_stage(self, name: str) -> None:
        if self._active is None:
            raise RuntimeError("static observer has no active batch")
        self._record_stage(self._active, name)

    def set_draft_tokens(self, draft_tokens: int) -> None:
        """Bind a backend's proposal width once it becomes observable."""
        if self._active is None:
            raise RuntimeError("static observer has no active batch")
        if int(draft_tokens) <= 0:
            raise ValueError("draft_tokens must be positive")
        self._active.draft_tokens = int(draft_tokens)

    def _copy_length_tensor(
        self, value, name: str, *, boolean: bool = False
    ) -> torch.Tensor:
        state = self._active
        if state is None:
            raise RuntimeError("static observer has no active batch")
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional tensor")
        if int(value.numel()) != state.batch_size:
            raise ValueError(
                f"{name} count {value.numel()} != batch size {state.batch_size}"
            )
        allowed = (torch.bool,) if boolean else self._HOST_DTYPES
        if value.dtype not in allowed:
            expected = "bool" if boolean else "int32 or int64"
            raise TypeError(f"{name} must use {expected}, got {value.dtype}")
        if self._cuda:
            if not value.is_cuda:
                raise TypeError(f"CUDA static observer received host {name}")
            assert self._lane_pool is not None
            host = self._lane_pool.host(
                int(state.lane), name, value.dtype, state.batch_size
            )
            host.copy_(value, non_blocking=True)
            return host
        if value.is_cuda:
            raise TypeError(f"CPU static observer received device {name}")
        return value.detach().clone()

    def after_accept(
        self,
        *,
        commit_lens,
        verify_lens=None,
        algorithmic_censored=None,
    ) -> None:
        state = self._active
        if state is None:
            raise RuntimeError("static observer has no active batch")
        state.commit_lens_cpu = self._copy_length_tensor(commit_lens, "commit")
        if verify_lens is not None:
            state.verify_lens_cpu = self._copy_length_tensor(verify_lens, "verify")
        if algorithmic_censored is not None:
            state.algorithmic_censored_cpu = self._copy_length_tensor(
                algorithmic_censored,
                "algorithmic_censored",
                boolean=True,
            )
        if "accept_end" not in state.cpu_stage_us:
            self._record_stage(state, "accept_end")

    def commit_round(self, *, new_seq_lens) -> None:
        state = self._active
        if state is None:
            raise RuntimeError("static observer has no active batch")
        if state.commit_lens_cpu is None:
            raise RuntimeError("static observer commit precedes acceptance")
        state.new_seq_lens_cpu = self._copy_length_tensor(new_seq_lens, "new_seq")
        now = time.monotonic() * 1e6
        if self._cuda:
            assert self._lane_pool is not None
            ready = self._lane_pool.event(int(state.lane), "telemetry_ready")
            ready.record(torch.cuda.current_stream(self.device))
            state.events["telemetry_ready"] = ready

        def release(lane=state.lane) -> None:
            if lane is not None:
                assert self._lane_pool is not None
                self._lane_pool.release(int(lane))

        for index, rid in enumerate(state.request_ids):
            round_id = self._round_of[rid]
            if self._cuda:
                committed = accepted = prefix_before = prefix_after = 0
                verify_len = state.draft_tokens + 1
                algorithmic_censored = False
            else:
                committed = int(state.commit_lens_cpu[index])
                accepted = max(committed - 1, 0)
                prefix_before = int(state.prefix_before[index])
                prefix_after = int(state.new_seq_lens_cpu[index])
                if prefix_after - prefix_before != committed:
                    raise RuntimeError(
                        "static telemetry prefix/count invariant failed: "
                        f"{prefix_after} - {prefix_before} != {committed}"
                    )
                verify_len = (
                    int(state.verify_lens_cpu[index])
                    if state.verify_lens_cpu is not None
                    else state.draft_tokens + 1
                )
                algorithmic_censored = bool(
                    state.algorithmic_censored_cpu is not None
                    and state.algorithmic_censored_cpu[index]
                )
            record = RoundTelemetry(
                request_id=rid,
                round_id=round_id,
                active_version=0,
                proposal_version=0,
                draft_tokens=state.draft_tokens,
                accepted_drafts=accepted,
                committed_per_verify=committed,
                target_calls=1,
                draft_cuda_us=0.0,
                verify_cuda_us=0.0,
                accept_cuda_us=0.0,
                draft_cpu_us=max(
                    state.cpu_stage_us.get("draft_end", now)
                    - state.cpu_stage_us["draft_start"],
                    0.0,
                ),
                verify_cpu_us=max(
                    state.cpu_stage_us.get("verify_end", now)
                    - state.cpu_stage_us.get("draft_end", now),
                    0.0,
                ),
                rng_substream_id=(
                    ""
                    if self._cuda
                    else rng_substream_identity(
                        request_id=rid,
                        sampling_seed=state.sampling_seeds[index],
                        is_greedy=state.greedy_flags[index],
                        round_id=round_id,
                        prefix_len=prefix_before,
                    )
                ),
                version_canary_ok=True,
                prefix_pos_before=prefix_before,
                prefix_pos_after=prefix_after,
                prefix_len_before=prefix_before,
                verify_len=verify_len,
                batch_size=state.batch_size,
                offered_concurrency=self.offered_concurrency,
                round_wall_us=max(now - state.wall_start_us, 0.0),
                prefix_feature_exact=True,
                algorithmic_censored=algorithmic_censored,
            )
            timing = None
            if self._cuda:
                timing = {
                    **state.events,
                    "request_index": index,
                    "prefix_lens_cpu": state.prefix_before,
                    "commit_lens_cpu": state.commit_lens_cpu,
                    "new_seq_lens_cpu": state.new_seq_lens_cpu,
                    "rng_sampling_seed": state.sampling_seeds[index],
                    "rng_is_greedy": state.greedy_flags[index],
                    "rng_round_id": round_id,
                }
                if state.verify_lens_cpu is not None:
                    timing["verify_lens_cpu"] = state.verify_lens_cpu
                if state.algorithmic_censored_cpu is not None:
                    timing["algorithmic_censored_cpu"] = (
                        state.algorithmic_censored_cpu
                    )
                if index == state.batch_size - 1:
                    timing["release"] = release
            self.telemetry.emit_round_deferred(record, timing)
            self._round_of[rid] = round_id + 1
        self._active = None

    def finish_request(self, rid: str) -> None:
        self._round_of.pop(str(rid), None)

    def diagnostics(self) -> dict:
        telemetry_flushed = self.telemetry.flush()
        health = self.telemetry.health()
        return {
            "observer": "native_static_round_telemetry_v1",
            "static_observer": True,
            "candidate_generation_bypassed": True,
            "controller_static_fallback": self.controller_static_fallback,
            "static_bypass_reason": self.static_bypass_reason,
            "active_requests": len(self._round_of),
            "active_batch": self._active is not None,
            "telemetry_path": str(self.telemetry.path),
            "telemetry_flushed": telemetry_flushed,
            "telemetry_error_count": health["error_count"],
            "telemetry_last_error": health["last_error"],
            "memory": {
                "fixed_bytes": 0,
                "transient_bytes": 0,
                "reserve_bytes": 0,
                "reserve_mb": 0,
                "num_slots": 0,
                "num_params": 0,
                "adapter_row_capacity": 0,
                "allocator_growth_bytes": 0,
            },
        }

    def close(self) -> None:
        if self._active is not None:
            raise RuntimeError("cannot close static observer with an active batch")
        self.telemetry.close()

    def __del__(self):
        try:
            if self._active is None:
                self.telemetry.close()
        except Exception:
            pass


def build_static_speculative_observer(
    *, config, server_args, worker, algorithm: str, bypass_reason: str | None = None
) -> StaticSpeculativeObserver:
    """Construct the one shared native-baseline observer for any backend."""

    from lightcone_spec.config.schema import MODEL_PAIRS

    algorithm = str(algorithm).upper()
    if algorithm not in ("DSPARK", "DFLASH", "EAGLE", "EAGLE3"):
        raise ValueError(f"unsupported static speculative backend {algorithm!r}")
    pair = MODEL_PAIRS.get(config.model.pair_id)
    if pair is not None and pair["speculative_algorithm"] != algorithm:
        raise ValueError(
            "static telemetry model/backend mismatch: "
            f"pair {config.model.pair_id!r} declares "
            f"{pair['speculative_algorithm']}, worker is {algorithm}"
        )
    ps = getattr(worker, "ps", None)
    rank = int(
        getattr(ps, "tp_rank", os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    )
    telemetry_root = (
        Path(config.trace.telemetry_path).parent
        if config.trace.telemetry_path
        else Path(config.trace.artifact_root)
    )
    telemetry_path = telemetry_root / (
        f"adaptation-telemetry-p{os.getpid()}-r{rank}.jsonl"
    )
    graph_config = getattr(server_args, "cuda_graph_config", None)
    decode = getattr(graph_config, "decode", None)
    graph_bs = getattr(decode, "bs", ()) or ()
    max_batch_size = max(
        int(getattr(server_args, "max_running_requests", 0) or 0),
        int(config.runtime.concurrency),
        max((int(value) for value in graph_bs), default=0),
        1,
    )
    observer = StaticSpeculativeObserver(
        telemetry=TelemetrySink(telemetry_path),
        device=str(worker.device),
        offered_concurrency=int(config.runtime.concurrency),
        max_batch_size=max_batch_size,
    )
    observer.config = config
    observer.algorithm = algorithm
    observer.static_bypass_reason = bypass_reason
    observer.controller_static_fallback = config.method == "lc_gate"
    return observer


def maybe_build_static_speculative_observer(
    server_args, worker, *, algorithm: str
) -> StaticSpeculativeObserver | None:
    """Build the evidence-only observer for an explicit static manifest."""

    path = getattr(server_args, "speculative_adaptation_config", None)
    if not path:
        return None
    from lightcone_spec.config.loader import load_adaptation_config

    config = load_adaptation_config(path)
    bypass, reason = evidence_only_static_bypass(config)
    if not bypass:
        return None
    return build_static_speculative_observer(
        config=config,
        server_args=server_args,
        worker=worker,
        algorithm=algorithm,
        bypass_reason=reason,
    )
