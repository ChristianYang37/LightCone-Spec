"""Fail-closed authority for auditing a source-owned shared-session reset.

The pinned server does not yet produce this contract.  The code in this module
can validate an exact native lifecycle with CPU fakes, but it cannot authorize
formal process reuse.  Callers cannot replace native state with local counters:
every capability, state snapshot, reset, warm-up, clock, trace, and close row is
obtained from one source-owned runtime and content-bound to one process.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol, Self

from lightcone_spec import PINNED_SGLANG_TREE

SOURCE_OWNED_SESSION_HOOK = "sglang.schema_v3.source_owned_all_reset_session.v1"
OFFICIAL_ALL_RESET_PRODUCER_AVAILABLE = False
SESSION_REUSE_BLOCK_REASON = "official_source_owned_all_reset_producer_unavailable"
FRESH_PROCESS_FALLBACK_MODE = "fresh_process_per_trace"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")

RESET_STATE_FIELDS = (
    "request_kv",
    "prefix_state",
    "rng",
    "counters",
    "cuda_peaks",
    "scheduler_statistics",
    "telemetry",
    "inference_weights",
    "fp32_master",
    "optimizer_moments",
    "candidate_buffers",
    "adapter_version",
    "optimizer_generation",
    "cohort_state",
    "update_counters",
    "allocator_hbm",
    "completion_event",
)


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\n" in value:
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _exact_mapping(value: object, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} fields are incomplete or unexpected")
    return value


def _validate_bound_sha256(raw: Mapping[str, object], field: str) -> None:
    observed = _require_sha256(field, raw[field])
    unsigned = dict(raw)
    unsigned.pop(field)
    if observed != _canonical_sha256(unsigned):
        raise ValueError(f"{field} is not content-bound")


@dataclass(frozen=True)
class ConnectionAccounting:
    """Server-owned cumulative HTTP accounting for one process generation."""

    generation: int
    connections_created: int
    connections_closed: int
    submitted_requests: int
    reused_requests: int
    abort_requests: int

    @classmethod
    def parse(cls, value: object) -> Self:
        raw = _exact_mapping(
            value, set(cls.__dataclass_fields__), "connection accounting"
        )
        result = cls(
            **{
                name: _nonnegative_integer(f"connection.{name}", raw[name])
                for name in cls.__dataclass_fields__
            }
        )
        if result.connections_closed > result.connections_created:
            raise ValueError("closed connections exceed created connections")
        if result.reused_requests > result.submitted_requests:
            raise ValueError("reused requests exceed submitted requests")
        if result.abort_requests > result.submitted_requests:
            raise ValueError("abort requests exceed submitted requests")
        return result

    def require_continuation(self, prior: ConnectionAccounting) -> None:
        if self.generation != prior.generation:
            raise ValueError(
                "connection-accounting generation changed inside a session"
            )
        for name in (
            "connections_created",
            "connections_closed",
            "submitted_requests",
            "reused_requests",
            "abort_requests",
        ):
            if getattr(self, name) < getattr(prior, name):
                raise ValueError(f"connection accounting moved backwards: {name}")


@dataclass(frozen=True)
class SourceOwnedSessionCapability:
    schema_version: int
    hook: str
    producer: str
    patched_sglang_tree: str
    session_plan_sha256: str
    process_identity: str
    process_started_ns: int
    session_epoch: int
    adapted_method: bool
    reset_state_fields: tuple[str, ...]
    continuous_connection_accounting: bool
    fault_reset_supported: bool
    capability_sha256: str

    @classmethod
    def parse(cls, value: object, *, session_plan_sha256: str) -> Self:
        raw = _exact_mapping(value, set(cls.__dataclass_fields__), "session capability")
        _validate_bound_sha256(raw, "capability_sha256")
        reset_fields = raw["reset_state_fields"]
        if not isinstance(reset_fields, list):
            raise TypeError("reset_state_fields must be one ordered JSON list")
        result = cls(
            schema_version=raw["schema_version"],
            hook=raw["hook"],
            producer=raw["producer"],
            patched_sglang_tree=raw["patched_sglang_tree"],
            session_plan_sha256=raw["session_plan_sha256"],
            process_identity=raw["process_identity"],
            process_started_ns=raw["process_started_ns"],
            session_epoch=raw["session_epoch"],
            adapted_method=raw["adapted_method"],
            reset_state_fields=tuple(reset_fields),
            continuous_connection_accounting=raw["continuous_connection_accounting"],
            fault_reset_supported=raw["fault_reset_supported"],
            capability_sha256=raw["capability_sha256"],
        )
        if result.schema_version != 1 or result.hook != SOURCE_OWNED_SESSION_HOOK:
            raise ValueError("source-owned session capability schema/hook mismatch")
        if result.producer != "native_server":
            raise ValueError("session capability is not source-owned")
        if (
            not isinstance(result.patched_sglang_tree, str)
            or _GIT_OBJECT.fullmatch(result.patched_sglang_tree) is None
            or result.patched_sglang_tree != PINNED_SGLANG_TREE
        ):
            raise ValueError("session capability names the wrong patched SGLang tree")
        if result.session_plan_sha256 != _require_sha256(
            "session_plan_sha256", session_plan_sha256
        ):
            raise ValueError("session capability belongs to another session plan")
        _require_text("process_identity", result.process_identity)
        _nonnegative_integer("process_started_ns", result.process_started_ns)
        _nonnegative_integer("session_epoch", result.session_epoch)
        if type(result.adapted_method) is not bool:
            raise ValueError("adapted_method must be boolean")
        if result.reset_state_fields != RESET_STATE_FIELDS:
            raise ValueError(
                "session capability does not cover the exact all-reset state"
            )
        if result.continuous_connection_accounting is not True:
            raise ValueError(
                "session capability lacks continuous connection accounting"
            )
        if type(result.fault_reset_supported) is not bool:
            raise ValueError("fault_reset_supported must be boolean")
        return result


@dataclass(frozen=True)
class SourceOwnedResetState:
    process_identity: str
    session_epoch: int
    reset_generation: int
    active_requests: int
    queued_requests: int
    request_kv_entries: int
    prefix_entries: int
    prefix_policy: str
    registered_prefix_sha256: str | None
    rng_sha256: str
    counters_sha256: str
    cuda_peaks_sha256: str
    scheduler_statistics_sha256: str
    telemetry_sha256: str
    inference_weights_sha256: str
    fp32_master_sha256: str | None
    optimizer_moments_sha256: str | None
    candidate_buffers_sha256: str | None
    adapter_version: int
    optimizer_generation: int
    cohort_state_sha256: str | None
    update_counter: int
    allocator_allocated_bytes: int
    allocator_reserved_bytes: int
    hbm_state_sha256: str
    completion_event_generation: int
    completion_event_complete: bool
    completion_event_sha256: str
    connection_accounting: ConnectionAccounting

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        capability: SourceOwnedSessionCapability,
    ) -> Self:
        keys = set(cls.__dataclass_fields__)
        raw = _exact_mapping(value, keys, "source-owned reset state")
        result = cls(
            **{
                **{name: raw[name] for name in keys - {"connection_accounting"}},
                "connection_accounting": ConnectionAccounting.parse(
                    raw["connection_accounting"]
                ),
            }
        )
        if (
            result.process_identity != capability.process_identity
            or result.session_epoch != capability.session_epoch
        ):
            raise ValueError("reset state belongs to another process/session")
        for name in (
            "reset_generation",
            "active_requests",
            "queued_requests",
            "request_kv_entries",
            "prefix_entries",
            "adapter_version",
            "optimizer_generation",
            "update_counter",
            "allocator_allocated_bytes",
            "allocator_reserved_bytes",
            "completion_event_generation",
        ):
            _nonnegative_integer(name, getattr(result, name))
        for name in (
            "rng_sha256",
            "counters_sha256",
            "cuda_peaks_sha256",
            "scheduler_statistics_sha256",
            "telemetry_sha256",
            "inference_weights_sha256",
            "hbm_state_sha256",
            "completion_event_sha256",
        ):
            _require_sha256(name, getattr(result, name))
        for name in (
            "registered_prefix_sha256",
            "fp32_master_sha256",
            "optimizer_moments_sha256",
            "candidate_buffers_sha256",
            "cohort_state_sha256",
        ):
            item = getattr(result, name)
            if item is not None:
                _require_sha256(name, item)
        if result.prefix_policy not in {"clear", "registered"}:
            raise ValueError("prefix policy must be clear or registered")
        if result.prefix_policy == "clear" and (
            result.prefix_entries != 0 or result.registered_prefix_sha256 is not None
        ):
            raise ValueError("unregistered prefix state was retained")
        if result.prefix_policy == "registered" and (
            result.registered_prefix_sha256 is None
        ):
            raise ValueError("registered prefix reuse lacks its exact identity")
        if type(result.completion_event_complete) is not bool:
            raise ValueError("completion event status must be boolean")
        adaptation_rows = (
            result.fp32_master_sha256,
            result.optimizer_moments_sha256,
            result.candidate_buffers_sha256,
            result.cohort_state_sha256,
        )
        if capability.adapted_method:
            if any(item is None for item in adaptation_rows):
                raise ValueError(
                    "adapted reset state lacks train/optimizer/cohort state"
                )
        elif any(item is not None for item in adaptation_rows):
            raise ValueError("non-adapted reset state carries adaptation state")
        return result

    @property
    def clean_state_sha256(self) -> str:
        """Digest resettable state, excluding lineage/events/connection totals."""

        value = asdict(self)
        for name in (
            "reset_generation",
            "completion_event_generation",
            "completion_event_complete",
            "completion_event_sha256",
            "connection_accounting",
        ):
            value.pop(name)
        return _canonical_sha256(value)

    def require_clean(self, *, capability: SourceOwnedSessionCapability) -> None:
        if self.active_requests or self.queued_requests:
            raise ValueError("reset boundary did not drain all live and queued work")
        if self.request_kv_entries:
            raise ValueError("reset boundary retained request KV state")
        if not self.completion_event_complete:
            raise ValueError("reset completion event is not complete")
        if capability.adapted_method:
            if (
                self.adapter_version != 0
                or self.optimizer_generation != 0
                or self.update_counter != 0
            ):
                raise ValueError("adapted reset did not restore initial generations")
        elif (
            self.adapter_version != 0
            or self.optimizer_generation != 0
            or self.update_counter != 0
        ):
            raise ValueError("non-adapted reset carries adaptation generations")


@dataclass(frozen=True)
class SourceOwnedResetReceipt:
    schema_version: int
    hook: str
    capability_sha256: str
    session_plan_sha256: str
    process_identity: str
    session_epoch: int
    prior_execution_plan_sha256: str | None
    next_execution_plan_sha256: str
    before: SourceOwnedResetState
    after: SourceOwnedResetState
    reset_duration_ns: int
    reset_receipt_sha256: str

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        capability: SourceOwnedSessionCapability,
        prior_execution_plan_sha256: str | None,
        next_execution_plan_sha256: str,
        clean_state_sha256: str,
        prior_accounting: ConnectionAccounting,
    ) -> Self:
        keys = set(cls.__dataclass_fields__)
        raw = _exact_mapping(value, keys, "source-owned reset receipt")
        _validate_bound_sha256(raw, "reset_receipt_sha256")
        before = SourceOwnedResetState.parse(raw["before"], capability=capability)
        after = SourceOwnedResetState.parse(raw["after"], capability=capability)
        result = cls(
            **{
                **{name: raw[name] for name in keys - {"before", "after"}},
                "before": before,
                "after": after,
            }
        )
        if result.schema_version != 1 or result.hook != SOURCE_OWNED_SESSION_HOOK:
            raise ValueError("source-owned reset receipt schema/hook mismatch")
        if (
            result.capability_sha256 != capability.capability_sha256
            or result.session_plan_sha256 != capability.session_plan_sha256
            or result.process_identity != capability.process_identity
            or result.session_epoch != capability.session_epoch
        ):
            raise ValueError("reset receipt belongs to another capability/process")
        if (
            result.prior_execution_plan_sha256 != prior_execution_plan_sha256
            or result.next_execution_plan_sha256 != next_execution_plan_sha256
        ):
            raise ValueError("reset receipt breaks the ordered trace chain")
        if result.prior_execution_plan_sha256 is not None:
            _require_sha256("prior execution plan", result.prior_execution_plan_sha256)
        _require_sha256("next execution plan", result.next_execution_plan_sha256)
        _nonnegative_integer("reset_duration_ns", result.reset_duration_ns)
        if result.after.reset_generation != result.before.reset_generation + 1:
            raise ValueError("reset generation did not advance exactly once")
        result.after.require_clean(capability=capability)
        if result.after.clean_state_sha256 != clean_state_sha256:
            raise ValueError("reset did not restore the source-attested clean state")
        result.before.connection_accounting.require_continuation(prior_accounting)
        result.after.connection_accounting.require_continuation(
            result.before.connection_accounting
        )
        if (
            result.after.connection_accounting.connections_closed
            != result.before.connection_accounting.connections_closed
        ):
            raise ValueError("shared HTTP pool closed during a reset boundary")
        return result


@dataclass(frozen=True)
class SourceOwnedWarmupReceipt:
    schema_version: int
    execution_plan_sha256: str
    request_pool_sha256: str
    excluded: bool
    started_ns: int
    completed_ns: int
    connection_accounting: ConnectionAccounting
    warmup_receipt_sha256: str

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        execution_plan_sha256: str,
        prior_accounting: ConnectionAccounting,
    ) -> Self:
        raw = _exact_mapping(value, set(cls.__dataclass_fields__), "warm-up receipt")
        _validate_bound_sha256(raw, "warmup_receipt_sha256")
        accounting = ConnectionAccounting.parse(raw["connection_accounting"])
        result = cls(
            **{
                **{
                    name: raw[name]
                    for name in cls.__dataclass_fields__
                    if name != "connection_accounting"
                },
                "connection_accounting": accounting,
            }
        )
        if result.schema_version != 1 or result.excluded is not True:
            raise ValueError("trace warm-up is not explicitly excluded")
        if result.execution_plan_sha256 != execution_plan_sha256:
            raise ValueError("warm-up belongs to another logical trace")
        _require_sha256("request_pool_sha256", result.request_pool_sha256)
        started = _nonnegative_integer("warmup.started_ns", result.started_ns)
        completed = _nonnegative_integer("warmup.completed_ns", result.completed_ns)
        if completed < started:
            raise ValueError("warm-up monotonic clock moved backwards")
        accounting.require_continuation(prior_accounting)
        return result


@dataclass(frozen=True)
class SourceOwnedScoredClockReceipt:
    schema_version: int
    execution_plan_sha256: str
    warmup_receipt_sha256: str
    clock_generation: int
    scored_started_ns: int
    clock_receipt_sha256: str

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        execution_plan_sha256: str,
        warmup: SourceOwnedWarmupReceipt,
        prior_clock_generation: int,
    ) -> Self:
        raw = _exact_mapping(
            value, set(cls.__dataclass_fields__), "scored clock receipt"
        )
        _validate_bound_sha256(raw, "clock_receipt_sha256")
        result = cls(**raw)
        if result.schema_version != 1:
            raise ValueError("scored clock schema is unsupported")
        if (
            result.execution_plan_sha256 != execution_plan_sha256
            or result.warmup_receipt_sha256 != warmup.warmup_receipt_sha256
        ):
            raise ValueError("scored clock is not bound to this trace warm-up")
        if result.clock_generation != prior_clock_generation + 1:
            raise ValueError("logical traces do not have independent scored clocks")
        if (
            _nonnegative_integer("scored_started_ns", result.scored_started_ns)
            < warmup.completed_ns
        ):
            raise ValueError("scored clock started before excluded warm-up completed")
        return result


@dataclass(frozen=True)
class SourceOwnedTraceReceipt:
    schema_version: int
    execution_plan_sha256: str
    clock_receipt_sha256: str
    terminal_receipt_sha256: str
    aborted: bool
    connection_accounting: ConnectionAccounting
    trace_receipt_sha256: str

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        execution_plan_sha256: str,
        clock: SourceOwnedScoredClockReceipt,
        prior_accounting: ConnectionAccounting,
    ) -> Self:
        raw = _exact_mapping(
            value, set(cls.__dataclass_fields__), "source trace receipt"
        )
        _validate_bound_sha256(raw, "trace_receipt_sha256")
        accounting = ConnectionAccounting.parse(raw["connection_accounting"])
        result = cls(
            **{
                **{
                    name: raw[name]
                    for name in cls.__dataclass_fields__
                    if name != "connection_accounting"
                },
                "connection_accounting": accounting,
            }
        )
        if result.schema_version != 1:
            raise ValueError("source trace schema is unsupported")
        if (
            result.execution_plan_sha256 != execution_plan_sha256
            or result.clock_receipt_sha256 != clock.clock_receipt_sha256
        ):
            raise ValueError("source trace receipt belongs to another scored clock")
        _require_sha256("terminal_receipt_sha256", result.terminal_receipt_sha256)
        if type(result.aborted) is not bool:
            raise ValueError("trace aborted flag must be boolean")
        accounting.require_continuation(prior_accounting)
        return result


@dataclass(frozen=True)
class SourceOwnedCloseReceipt:
    schema_version: int
    process_identity: str
    closed: bool
    connection_accounting: ConnectionAccounting
    close_receipt_sha256: str

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        capability: SourceOwnedSessionCapability,
        prior_accounting: ConnectionAccounting,
    ) -> Self:
        raw = _exact_mapping(
            value, set(cls.__dataclass_fields__), "session close receipt"
        )
        _validate_bound_sha256(raw, "close_receipt_sha256")
        accounting = ConnectionAccounting.parse(raw["connection_accounting"])
        result = cls(
            schema_version=raw["schema_version"],
            process_identity=raw["process_identity"],
            closed=raw["closed"],
            connection_accounting=accounting,
            close_receipt_sha256=raw["close_receipt_sha256"],
        )
        if (
            result.schema_version != 1
            or result.process_identity != capability.process_identity
            or result.closed is not True
        ):
            raise ValueError("source did not close the exact shared process")
        accounting.require_continuation(prior_accounting)
        if accounting.connections_closed != accounting.connections_created:
            raise ValueError("source close left HTTP connections open")
        return result


class SourceOwnedSessionAuditRuntime(Protocol):
    async def capability(self, *, session_plan_sha256: str) -> object: ...

    async def initial_state(self, *, capability_sha256: str) -> object: ...

    async def reset_boundary(
        self,
        *,
        capability_sha256: str,
        prior_execution_plan_sha256: str | None,
        next_execution_plan_sha256: str,
    ) -> object: ...

    async def excluded_warmup(self, *, execution_plan_sha256: str) -> object: ...

    async def start_scored_clock(self, *, execution_plan_sha256: str) -> object: ...

    async def finish_trace(self, *, execution_plan_sha256: str) -> object: ...

    async def close(self, *, capability_sha256: str) -> object: ...


@dataclass(frozen=True)
class SessionReuseAuditResult:
    status: str
    reason: str
    session_plan_sha256: str
    capability_sha256: str | None
    reset_receipt_sha256s: tuple[str, ...]
    warmup_receipt_sha256s: tuple[str, ...]
    clock_receipt_sha256s: tuple[str, ...]
    trace_receipt_sha256s: tuple[str, ...]
    close_receipt_sha256: str | None
    reuse_authorized: bool
    fallback_mode: str

    @property
    def sha256(self) -> str:
        return _canonical_sha256({"schema_version": 1, **asdict(self)})


async def audit_source_owned_reuse_contract(
    *,
    session_plan_sha256: str,
    execution_plan_sha256s: Sequence[str],
    runtime: SourceOwnedSessionAuditRuntime,
    fault_injection: bool = False,
) -> SessionReuseAuditResult:
    """Exercise the exact CPU/native lifecycle without authorizing reuse.

    Any failure or aborted trace closes the source process and requires the
    existing fresh-process-per-trace fallback.  A successful audit is still
    BLOCKED until the pinned patch ships the official producer.
    """

    plan_sha = _require_sha256("session_plan_sha256", session_plan_sha256)
    trace_shas = tuple(execution_plan_sha256s)
    if not trace_shas or len(set(trace_shas)) != len(trace_shas):
        raise ValueError("reuse audit requires unique ordered logical traces")
    for trace_sha in trace_shas:
        _require_sha256("execution_plan_sha256", trace_sha)

    capability: SourceOwnedSessionCapability | None = None
    accounting: ConnectionAccounting | None = None
    reset_receipts: list[str] = []
    warmup_receipts: list[str] = []
    clock_receipts: list[str] = []
    trace_receipts: list[str] = []
    close_sha: str | None = None
    failure_reason = SESSION_REUSE_BLOCK_REASON

    try:
        capability = SourceOwnedSessionCapability.parse(
            await runtime.capability(session_plan_sha256=plan_sha),
            session_plan_sha256=plan_sha,
        )
        initial = SourceOwnedResetState.parse(
            await runtime.initial_state(capability_sha256=capability.capability_sha256),
            capability=capability,
        )
        initial.require_clean(capability=capability)
        accounting = initial.connection_accounting
        if fault_injection and not capability.fault_reset_supported:
            raise RuntimeError("fault_injection_requires_fresh_process")
        prior_trace: str | None = None
        clock_generation = 0
        for trace_sha in trace_shas:
            reset = SourceOwnedResetReceipt.parse(
                await runtime.reset_boundary(
                    capability_sha256=capability.capability_sha256,
                    prior_execution_plan_sha256=prior_trace,
                    next_execution_plan_sha256=trace_sha,
                ),
                capability=capability,
                prior_execution_plan_sha256=prior_trace,
                next_execution_plan_sha256=trace_sha,
                clean_state_sha256=initial.clean_state_sha256,
                prior_accounting=accounting,
            )
            accounting = reset.after.connection_accounting
            reset_receipts.append(reset.reset_receipt_sha256)
            warmup = SourceOwnedWarmupReceipt.parse(
                await runtime.excluded_warmup(execution_plan_sha256=trace_sha),
                execution_plan_sha256=trace_sha,
                prior_accounting=accounting,
            )
            accounting = warmup.connection_accounting
            warmup_receipts.append(warmup.warmup_receipt_sha256)
            clock = SourceOwnedScoredClockReceipt.parse(
                await runtime.start_scored_clock(execution_plan_sha256=trace_sha),
                execution_plan_sha256=trace_sha,
                warmup=warmup,
                prior_clock_generation=clock_generation,
            )
            clock_generation = clock.clock_generation
            clock_receipts.append(clock.clock_receipt_sha256)
            trace = SourceOwnedTraceReceipt.parse(
                await runtime.finish_trace(execution_plan_sha256=trace_sha),
                execution_plan_sha256=trace_sha,
                clock=clock,
                prior_accounting=accounting,
            )
            accounting = trace.connection_accounting
            trace_receipts.append(trace.trace_receipt_sha256)
            if trace.aborted:
                raise RuntimeError("source_trace_aborted")
            prior_trace = trace_sha
    except Exception as error:  # noqa: BLE001 - source failure must trigger close
        failure_reason = f"shared_session_audit_failed:{type(error).__name__}:{error}"
    finally:
        if capability is not None and accounting is not None:
            try:
                close = SourceOwnedCloseReceipt.parse(
                    await runtime.close(capability_sha256=capability.capability_sha256),
                    capability=capability,
                    prior_accounting=accounting,
                )
                close_sha = close.close_receipt_sha256
            except Exception as close_error:  # noqa: BLE001 - close is evidence
                close_failure = (
                    "shared_session_close_failed:"
                    f"{type(close_error).__name__}:{close_error}"
                )
                if failure_reason.startswith("shared_session_audit_failed:"):
                    failure_reason = f"{failure_reason};{close_failure}"
                else:
                    failure_reason = close_failure

    status = (
        "CPU_CONTRACT_ONLY"
        if len(trace_receipts) == len(trace_shas)
        and not failure_reason.startswith("shared_session_")
        else "FRESH_PROCESS_REQUIRED"
    )
    # The current pinned patch has no all-reset producer.  Even a complete CPU
    # audit therefore stays non-executable and names the safe fallback.
    if not OFFICIAL_ALL_RESET_PRODUCER_AVAILABLE and status == "CPU_CONTRACT_ONLY":
        failure_reason = SESSION_REUSE_BLOCK_REASON
    return SessionReuseAuditResult(
        status=status,
        reason=failure_reason,
        session_plan_sha256=plan_sha,
        capability_sha256=(
            None if capability is None else capability.capability_sha256
        ),
        reset_receipt_sha256s=tuple(reset_receipts),
        warmup_receipt_sha256s=tuple(warmup_receipts),
        clock_receipt_sha256s=tuple(clock_receipts),
        trace_receipt_sha256s=tuple(trace_receipts),
        close_receipt_sha256=close_sha,
        reuse_authorized=False,
        fallback_mode=FRESH_PROCESS_FALLBACK_MODE,
    )


__all__ = (
    "FRESH_PROCESS_FALLBACK_MODE",
    "OFFICIAL_ALL_RESET_PRODUCER_AVAILABLE",
    "RESET_STATE_FIELDS",
    "SESSION_REUSE_BLOCK_REASON",
    "SOURCE_OWNED_SESSION_HOOK",
    "ConnectionAccounting",
    "SessionReuseAuditResult",
    "SourceOwnedSessionAuditRuntime",
    "audit_source_owned_reuse_contract",
)
