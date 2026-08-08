"""Telemetry emitted by the fork (spec 8.5) and consumed by the bridge
to build the spec-11 artifact rows on the GPU path."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from lightcone_spec.runtime.events import monotonic_us
from lightcone_spec.sglang_bridge.hooks import rng_substream_identity


class CudaTelemetryLanePool:
    """Fixed-address CUDA events and pinned scalar rows for hot-path evidence.

    A lane remains leased until the deferred telemetry worker has consumed every
    record that references it.  Callers decide that release point and enqueue
    :meth:`release` through :meth:`TelemetrySink.defer`; exhaustion is explicit
    rather than falling back to an unbounded per-round allocation.
    """

    _HOST_DTYPES = (torch.bool, torch.int32, torch.int64)

    def __init__(
        self,
        *,
        device: str,
        max_batch_size: int,
        event_names: Sequence[str],
        host_names: Sequence[str],
        device_scalars: Mapping[str, torch.dtype] | None = None,
        untimed_event_names: Sequence[str] | None = None,
        lane_count: int = 64,
    ) -> None:
        if not str(device).startswith("cuda"):
            raise ValueError("CUDA telemetry lanes require a CUDA device")
        if max_batch_size <= 0 or lane_count <= 0:
            raise ValueError("telemetry lane capacity must be positive")
        event_names = tuple(str(name) for name in event_names)
        host_names = tuple(str(name) for name in host_names)
        if len(set(event_names)) != len(event_names):
            raise ValueError("telemetry event names must be unique")
        untimed_event_names = frozenset(
            str(name)
            for name in (
                untimed_event_names
                if untimed_event_names is not None
                else (("telemetry_ready",) if "telemetry_ready" in event_names else ())
            )
        )
        unknown_untimed = untimed_event_names.difference(event_names)
        if unknown_untimed:
            raise ValueError(
                "untimed telemetry events are not declared: "
                + ", ".join(sorted(unknown_untimed))
            )
        if len(set(host_names)) != len(host_names):
            raise ValueError("telemetry host-buffer names must be unique")
        device_scalars = {
            str(name): dtype for name, dtype in (device_scalars or {}).items()
        }
        for name, dtype in device_scalars.items():
            if not isinstance(dtype, torch.dtype):
                raise TypeError(
                    f"CUDA telemetry scalar {name!r} has invalid dtype {dtype!r}"
                )
        self.device = str(device)
        self.max_batch_size = int(max_batch_size)
        self.lane_count = int(lane_count)
        self._free: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._lease_lock = threading.Lock()
        self._leased = [False] * self.lane_count
        self._events = [
            {
                name: torch.cuda.Event(enable_timing=name not in untimed_event_names)
                for name in event_names
            }
            for _ in range(self.lane_count)
        ]
        self._host = {
            dtype: {
                name: torch.empty(
                    (self.lane_count, self.max_batch_size),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for name in host_names
            }
            for dtype in self._HOST_DTYPES
        }
        # Scalars used only for deferred evidence must not allocate a fresh
        # zero-dimensional tensor in every decode/update round.  One row per
        # lane also gives tests a stable-address contract analogous to graph
        # replay buffers.
        self._device_scalars = {
            name: torch.empty(
                (self.lane_count,), dtype=dtype, device=self.device
            )
            for name, dtype in device_scalars.items()
        }
        for lane in range(self.lane_count):
            self._free.put(lane)

    def acquire(self) -> int:
        try:
            lane = self._free.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(
                "CUDA telemetry lanes exhausted; refusing unbounded hot-path "
                "allocation and incomplete timing evidence"
            ) from exc
        with self._lease_lock:
            if self._leased[lane]:
                raise RuntimeError(f"CUDA telemetry lane {lane} was leased twice")
            self._leased[lane] = True
        return lane

    def release(self, lane: int) -> None:
        lane = int(lane)
        if not 0 <= lane < self.lane_count:
            raise IndexError(f"CUDA telemetry lane {lane} is out of range")
        with self._lease_lock:
            if not self._leased[lane]:
                raise RuntimeError(f"CUDA telemetry lane {lane} was released twice")
            self._leased[lane] = False
        self._free.put(lane)

    def event(self, lane: int, name: str):
        return self._events[int(lane)][str(name)]

    def host(self, lane: int, name: str, dtype, batch_size: int) -> torch.Tensor:
        if dtype not in self._HOST_DTYPES:
            raise TypeError(
                f"telemetry scalar rows require int32 or int64, got {dtype}"
            )
        batch_size = int(batch_size)
        if not 0 < batch_size <= self.max_batch_size:
            raise RuntimeError(
                f"telemetry batch {batch_size} exceeds lane capacity "
                f"{self.max_batch_size}"
            )
        try:
            storage = self._host[dtype][str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown telemetry host buffer {name!r}") from exc
        return storage[int(lane), :batch_size]

    def device_scalar(self, lane: int, name: str) -> torch.Tensor:
        """Return a fixed-address scalar view owned by ``lane``."""

        lane = int(lane)
        if not 0 <= lane < self.lane_count:
            raise IndexError(f"CUDA telemetry lane {lane} is out of range")
        try:
            storage = self._device_scalars[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown CUDA telemetry scalar {name!r}") from exc
        return storage[lane]

    @property
    def host_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for by_name in self._host.values()
            for tensor in by_name.values()
        )

    @property
    def device_scalar_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._device_scalars.values()
        )

    @property
    def leased_count(self) -> int:
        with self._lease_lock:
            return sum(self._leased)


@dataclass
class RoundTelemetry:
    request_id: str
    round_id: int
    active_version: int
    proposal_version: int
    draft_tokens: int
    accepted_drafts: int
    committed_per_verify: int
    target_calls: int
    draft_cuda_us: float
    verify_cuda_us: float
    accept_cuda_us: float
    draft_cpu_us: float
    verify_cpu_us: float
    rng_substream_id: str
    version_canary_ok: bool
    prefix_pos_before: int = 0
    prefix_pos_after: int = 0
    # P5 names the semantic quantity explicitly. ``prefix_pos_before`` stays
    # in the v1 parquet schema so completed P0--P4 artifacts remain readable.
    prefix_len_before: int = 0
    # Target positions launched, including the mandatory anchor/bonus token.
    # Thus max(verify_len - 1, 0) is the number of verified draft tokens.
    verify_len: int = 0
    batch_size: int = 1
    offered_concurrency: int = 1
    round_wall_us: float = 0.0
    prefix_feature_exact: bool = True
    # Physical counters/timings above describe work actually executed by the
    # backend.  A censored round crossed max_new_tokens or an exact terminal
    # token and must not contribute acceptance/committed-token algorithmic
    # gains.  P5 keeps its physical cost but excludes it from headline counts.
    algorithmic_censored: bool = False
    # Optional provenance for parameter-dependent draft caches.  Existing
    # cache-invariant tail modes intentionally leave these unset.
    cache_policy: Optional[str] = None
    proposal_weight_version: Optional[int] = None
    kv_version_min: Optional[int] = None
    kv_version_max: Optional[int] = None
    kv_append_version: Optional[int] = None
    cache_version_canary_ok: Optional[bool] = None
    # Main-stream work that prepares owned teacher tensors and enqueues online
    # candidates.  DFlash records one fixed-lane event pair per scheduler
    # batch; the deferred worker allocates the batch duration evenly across
    # its request rows so summing rows reconstructs the physical batch work.
    # ``None`` means uninstrumented/unknown, never a synthetic zero.
    signal_prep_cuda_us: Optional[float] = None


@dataclass
class UpdateTelemetry:
    request_id: str
    update_id: str
    source_round: int
    source_version: int
    snapshot_ts_us: float
    # Source-bound optimization evidence. CUDA producers leave the scalar
    # fields unset on the decode thread and materialize them only after the
    # candidate ready event in ``_deferred_worker``.
    source_training_loss: Optional[float] = None
    source_expected_accepted_prefix: Optional[float] = None
    source_prefix_len: Optional[int] = None
    active_version_at_arrival: Optional[int] = None
    staging_version: Optional[int] = None
    teacher_ts_us: Optional[float] = None
    launch_ts_us: Optional[float] = None
    done_ts_us: Optional[float] = None
    commit_ts_us: Optional[float] = None
    exposure_ts_us: Optional[float] = None
    published_version: Optional[int] = None
    grad_norm: float = 0.0
    candidate_delta_norm: float = 0.0
    decision: str = ""
    failure_reason: Optional[str] = None
    effective_delay_rounds: int = 0
    pipeline_min_delay_rounds: int = 1
    delay_tokens: int = 0
    delay_wall_us: float = 0.0
    delay_versions: int = 0
    rho_path: float = 0.0
    endpoint_distance: float = 0.0
    parameter_displacement: float = 0.0
    predicted_utility: Optional[float] = None
    predicted_mismatch: Optional[float] = None
    predicted_harm_probability: Optional[float] = None
    threshold: Optional[float] = None
    damping_factor: Optional[float] = None
    grad_clip_scale: float = 1.0
    optimizer_step: int = 0
    side_queue_cuda_us: float = 0.0
    candidate_cuda_us: float = 0.0
    backward_cuda_us: Optional[float] = None
    optimizer_cuda_us: Optional[float] = None
    # Source-bound DFlash/EAGLE candidates may share one batched backward and
    # AdamW launch.  Timings below are amortized per request so summing update
    # rows reconstructs physical side-stream work.
    candidate_batch_size: int = 1
    barrier_wait_cpu_us: float = 0.0
    controller_cpu_us: float = 0.0
    controller_cuda_us: float = 0.0
    publish_cuda_us: float = 0.0
    prefix_feature_exact: bool = True
    # Optional truncated-gradient provenance for future draft-backbone modes.
    gradient_weight_version: Optional[int] = None
    gradient_kv_version_min: Optional[int] = None
    gradient_kv_version_max: Optional[int] = None
    gradient_version_canary_ok: Optional[bool] = None


@dataclass
class SystemSampleTelemetry:
    timestamp_us: float
    gpu_index: int
    hbm_used_bytes: int
    gpu_utilization: float
    power_watts: float
    energy_joules_delta: float
    main_stream_active: Optional[bool]
    side_stream_active: Optional[bool]
    stream_contention_class: Optional[str]
    sync_us_delta: Optional[float]
    sm_occupancy: Optional[float] = None
    sample_source: Optional[str] = None
    activity_provenance: Optional[str] = None
    contention_provenance: Optional[str] = None
    sync_provenance: Optional[str] = None


class TelemetrySink:
    """Thread-safe JSONL sink; one file per run. The bridge converts the
    JSONL stream into the parquet tables after the run completes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")
        self._closed = False
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._deferred: queue.Queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._deferred_worker,
            name=f"lightcone-telemetry-{self.path.name}",
            daemon=True,
        )
        self._worker.start()

    def emit(self, kind: str, record) -> None:
        with self._lock:
            if self._closed:
                return
            self._fh.write(json.dumps({"kind": kind, **asdict(record)}) + "\n")
            # Never issue a filesystem flush from a serving or side-stream
            # control callback. ``flush`` and ``close`` are explicit run
            # boundaries and artifact validation rejects incomplete runs.

    def emit_round_deferred(self, record: RoundTelemetry, timing: dict | None) -> None:
        """Materialize CUDA event durations off the decode/control thread."""
        if not timing:
            self.emit("round", record)
            return
        self._deferred.put(("round", record, timing))

    def emit_update_deferred(self, record: UpdateTelemetry, values: dict) -> None:
        """Materialize update scalars only after the side-stream event."""
        self._deferred.put(("update", record, values))

    def defer(self, callback, event=None) -> None:
        """Run bounded trace serialization off the serving thread."""
        self._deferred.put(("callback", callback, {"event": event}))

    def flush(self, timeout_s: float = 120.0) -> bool:
        """Drain deferred CUDA materialization before evidence collection."""
        if self._closed:
            return True
        done = threading.Event()
        self.defer(done.set)
        completed = done.wait(timeout_s)
        with self._lock:
            self._fh.flush()
        return completed

    def health(self) -> dict:
        return {
            "error_count": self._error_count,
            "last_error": self._last_error,
        }

    def _deferred_worker(self) -> None:
        while True:
            item = self._deferred.get()
            if item is None:
                return
            kind, record, timing = item
            release = timing.get("release") if isinstance(timing, dict) else None
            try:
                if kind == "callback":
                    event = timing.get("event")
                    if event is not None:
                        event.synchronize()
                    record()
                    continue
                if kind == "round":
                    timing.get("telemetry_ready", timing["accept_end"]).synchronize()
                    record.draft_cuda_us = 1000.0 * timing[
                        "draft_start"
                    ].elapsed_time(timing["draft_end"])
                    record.verify_cuda_us = 1000.0 * timing[
                        "draft_end"
                    ].elapsed_time(timing["verify_end"])
                    record.accept_cuda_us = 1000.0 * timing[
                        "verify_end"
                    ].elapsed_time(timing["accept_end"])
                    signal_start = timing.get("signal_prep_start")
                    signal_end = timing.get("signal_prep_end")
                    if signal_start is not None and signal_end is not None:
                        record.signal_prep_cuda_us = (
                            1000.0
                            * signal_start.elapsed_time(signal_end)
                            / max(int(record.batch_size), 1)
                        )
                    index = timing.get("request_index")
                    if index is not None and "commit_lens_cpu" in timing:
                        committed = int(timing["commit_lens_cpu"][index])
                        expected = timing.get("expected_committed_per_verify")
                        if expected is not None and committed != int(expected):
                            raise RuntimeError(
                                "deferred committed count disagrees with exact "
                                f"host prefix delta: {committed} != {int(expected)}"
                            )
                        record.committed_per_verify = committed
                        record.accepted_drafts = max(committed - 1, 0)
                    if index is not None and "new_seq_lens_cpu" in timing:
                        prefix_after = int(timing["new_seq_lens_cpu"][index])
                        expected = timing.get("expected_prefix_pos_after")
                        if expected is not None and prefix_after != int(expected):
                            raise RuntimeError(
                                "deferred prefix length disagrees with exact host "
                                f"boundary: {prefix_after} != {int(expected)}"
                            )
                        record.prefix_pos_after = prefix_after
                    if index is not None and "prefix_lens_cpu" in timing:
                        prefix_before = int(timing["prefix_lens_cpu"][index])
                        if prefix_before < 0:
                            raise RuntimeError(
                                "deferred prefix length must be non-negative"
                            )
                        record.prefix_pos_before = prefix_before
                        record.prefix_len_before = prefix_before
                        if "rng_is_greedy" in timing:
                            record.rng_substream_id = rng_substream_identity(
                                request_id=record.request_id,
                                sampling_seed=timing.get("rng_sampling_seed"),
                                is_greedy=bool(timing["rng_is_greedy"]),
                                round_id=int(timing["rng_round_id"]),
                                prefix_len=prefix_before,
                            )
                        if (
                            "commit_lens_cpu" in timing
                            and "new_seq_lens_cpu" in timing
                        ):
                            committed = int(timing["commit_lens_cpu"][index])
                            prefix_after = int(timing["new_seq_lens_cpu"][index])
                            if prefix_after - prefix_before != committed:
                                raise RuntimeError(
                                    "deferred static prefix/count invariant failed: "
                                    f"{prefix_after} - {prefix_before} != {committed}"
                                )
                    if index is not None and "verify_lens_cpu" in timing:
                        record.verify_len = int(timing["verify_lens_cpu"][index])
                    if index is not None and "algorithmic_censored_cpu" in timing:
                        record.algorithmic_censored = bool(
                            timing["algorithmic_censored_cpu"][index]
                        )
                else:
                    event = timing.get("event")
                    if event is not None:
                        event.synchronize()
                    if record.done_ts_us is None:
                        record.done_ts_us = max(
                            monotonic_us(),
                            record.launch_ts_us or record.snapshot_ts_us,
                        )
                    finite = timing["numerical_ok"]
                    record.grad_norm = float(timing["grad_norm"])
                    record.candidate_delta_norm = float(timing["candidate_delta_norm"])
                    for name in (
                        "source_training_loss",
                        "source_expected_accepted_prefix",
                    ):
                        value = timing.get(name)
                        if value is None:
                            raise RuntimeError(
                                "deferred candidate telemetry is missing " + name
                            )
                        setattr(record, name, float(value))
                    if not bool(finite):
                        record.failure_reason = "non_finite_candidate"
                        if record.published_version is not None:
                            raise RuntimeError(
                                "non-finite candidate was reported as published"
                            )
                        record.decision = "discard"
                    gate_applied = timing.get("gate_applied")
                    if gate_applied is not None and not bool(gate_applied):
                        record.decision = "discard_noop_publish"
                    for name in (
                        "rho_path",
                        "endpoint_distance",
                        "parameter_displacement",
                        "predicted_utility",
                        "predicted_mismatch",
                        "predicted_harm_probability",
                        "damping_factor",
                        "grad_clip_scale",
                    ):
                        value = timing.get(name)
                        if value is not None:
                            setattr(record, name, float(value))
                    optimizer_step = timing.get("optimizer_step")
                    if optimizer_step is not None:
                        record.optimizer_step = int(optimizer_step)
                    candidate_batch_size = max(
                        int(timing.get("candidate_batch_size", 1)), 1
                    )
                    record.candidate_batch_size = candidate_batch_size
                    for field_name, start_name, end_name in (
                        ("side_queue_cuda_us", "ready_event", "side_start"),
                        ("candidate_cuda_us", "side_start", "candidate_end"),
                        ("backward_cuda_us", "backward_start", "backward_end"),
                        ("optimizer_cuda_us", "backward_end", "optimizer_end"),
                        ("controller_cuda_us", "controller_start", "controller_end"),
                        ("publish_cuda_us", "publish_start", "publish_end"),
                    ):
                        start = timing.get(start_name)
                        end = timing.get(end_name)
                        if start is not None and end is not None:
                            elapsed = 1000.0 * start.elapsed_time(end)
                            if field_name in (
                                "candidate_cuda_us",
                                "backward_cuda_us",
                                "optimizer_cuda_us",
                            ):
                                elapsed /= candidate_batch_size
                            setattr(record, field_name, elapsed)
            except Exception as exc:
                # Never crash serving from its evidence thread, but surface
                # the failure before the run can be labelled complete_valid
                # with default-zero component timings.
                self._error_count += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            if kind == "callback":
                continue
            self.emit(kind, record)
            if release is not None:
                try:
                    release()
                except Exception as exc:
                    self._error_count += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        if not self._closed:
            self._deferred.put(None)
            self._worker.join()
        with self._lock:
            if not self._closed:
                self._fh.flush()
                self._fh.close()
                self._closed = True

    def __enter__(self) -> "TelemetrySink":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
