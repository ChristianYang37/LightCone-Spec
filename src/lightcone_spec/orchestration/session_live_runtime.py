"""Live consumer for the pinned source-owned session lifecycle.

This module is deliberately a consumer, not another serving plane.  Normal
generate and abort traffic stays on :class:`PinnedBenchServingTransport`; the
three native terminal lifecycle requests are routed to the pinned atomic
session endpoints on that same transport instance and connection pool.

The source-owned reset audit and ``NativeTerminalProvider`` remain the receipt
authorities.  A complete CPU exercise therefore remains non-authorizing until
the registered GPU reset contracts pass on the pinned runtime.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

from lightcone_spec.experiments.serving import PinnedBenchServingTransport
from lightcone_spec.orchestration.native_terminal import (
    CAPABILITY_PATH,
    TERMINAL_EVIDENCE_PATH,
    NativeTerminalProvider,
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    ValidatedNativeTerminalEvidence,
    canonical_json_bytes,
    canonical_sha256,
)
from lightcone_spec.orchestration.session_reuse_authority import (
    SESSION_REUSE_BLOCK_REASON,
    SESSION_REUSE_GPU_VERIFIED_REASON,
    SOURCE_OWNED_SESSION_HOOK,
    SessionReuseAuditResult,
    SourceOwnedSessionAuditRuntime,
    audit_source_owned_reuse_contract,
)
from lightcone_spec.runtime.readiness import VerifiedNativeRuntimeGpuProof

_SESSION_CAPABILITY_PATH = "/v1/lightcone-spec/session-reset/capability"
_SESSION_INITIAL_STATE_PATH = "/v1/lightcone-spec/session-reset/initial-state"
_SESSION_RESET_PATH = "/v1/lightcone-spec/session-reset"
_SESSION_TRACE_PATHS = {
    "begin": "/v1/lightcone-spec/session-reset/trace/begin",
    "reset": "/v1/lightcone-spec/session-reset/trace/reset",
    "finalize": "/v1/lightcone-spec/session-reset/trace/finalize",
}
_SESSION_CLOSE_PATH = "/v1/lightcone-spec/session-reset/close-terminal"


@dataclass(frozen=True)
class SessionLiveStepBinding:
    """Canonical raw response retained in memory with its content digest."""

    step: str
    execution_plan_sha256: str | None
    raw_json: str
    content_sha256: str

    @classmethod
    def capture(
        cls,
        *,
        step: str,
        value: object,
        execution_plan_sha256: str | None = None,
    ) -> Self:
        if not isinstance(step, str) or not step or "\n" in step:
            raise ValueError("live session step must be non-empty single-line text")
        raw = canonical_json_bytes(value).decode("utf-8")
        return cls(
            step=step,
            execution_plan_sha256=execution_plan_sha256,
            raw_json=raw,
            content_sha256=canonical_sha256(value),
        )


class SessionLiveEvidenceSink(Protocol):
    """Internal sink for crash-durable raw lifecycle publication."""

    def record_step(self, step: SessionLiveStepBinding) -> None: ...

    def finalize(self, result: SessionLiveContractResult) -> None: ...

    def close_partial(self) -> None: ...


class SessionLiveTraceDriver(Protocol):
    """Execute one trace's warm-up and scored requests on the shared pool."""

    async def run_warmup(
        self,
        *,
        binding: NativeTerminalRunBinding,
        transport: SessionLivePinnedBenchTransport,
    ) -> Sequence[TerminalRequestExpectation]: ...

    async def run_scored(
        self,
        *,
        binding: NativeTerminalRunBinding,
        transport: SessionLivePinnedBenchTransport,
    ) -> Sequence[TerminalRequestExpectation]: ...


class SessionLiveTerminalObserver(Protocol):
    """Receive one already-validated terminal before the next reset begins.

    The observer is an evidence sink, not a serving callback: request execution
    remains owned by ``SessionLiveTraceDriver`` and the pinned transport.  A
    raised observer error therefore fails the source-owned session and selects
    the fresh-process fallback.
    """

    async def terminal_finalized(
        self,
        *,
        trace_index: int,
        binding: NativeTerminalRunBinding,
        terminal: ValidatedNativeTerminalEvidence,
    ) -> None: ...


@dataclass(frozen=True)
class SessionLiveTraceInput:
    binding: NativeTerminalRunBinding
    driver: SessionLiveTraceDriver

    def validate(self) -> None:
        if type(self.binding) is not NativeTerminalRunBinding:
            raise TypeError("live trace requires an exact native terminal binding")
        self.binding.validate()
        if not callable(getattr(self.driver, "run_warmup", None)) or not callable(
            getattr(self.driver, "run_scored", None)
        ):
            raise TypeError("live trace driver lacks warm-up or scored execution")


class SessionLiveProcessOwner(Protocol):
    """Owner of the already-launched pinned server process."""

    async def close(self) -> None: ...

    async def force_close(self) -> None: ...


@dataclass(frozen=True)
class SessionLiveContractResult:
    """Non-release result of exercising one live shared process."""

    audit: SessionReuseAuditResult
    execution_plan_sha256s: tuple[str, ...]
    steps: tuple[SessionLiveStepBinding, ...]
    native_terminals: tuple[ValidatedNativeTerminalEvidence, ...]
    transport_closed: bool
    process_closed: bool
    process_force_closed: bool
    verified_gpu_proof_sha256: str | None = None

    def validate(self) -> None:
        if self.audit.status not in {
            "CPU_CONTRACT_ONLY",
            "FRESH_PROCESS_REQUIRED",
            "GPU_VERIFIED",
        }:
            raise ValueError("live session audit returned an unsupported status")
        if self.audit.status in {"CPU_CONTRACT_ONLY", "GPU_VERIFIED"}:
            if (
                self.audit.reason
                != (
                    SESSION_REUSE_GPU_VERIFIED_REASON
                    if self.audit.status == "GPU_VERIFIED"
                    else SESSION_REUSE_BLOCK_REASON
                )
                or not self.transport_closed
                or not self.process_closed
                or self.process_force_closed
            ):
                raise ValueError(
                    "successful live audit did not close safely or stay blocked"
                )
        elif (
            not self.transport_closed
            or self.process_closed
            or not self.process_force_closed
        ):
            raise ValueError(
                "failed live audit did not close pool and force process exit"
            )
        if self.process_closed and self.process_force_closed:
            raise ValueError("process close disposition is ambiguous")
        if (
            self.audit.status in {"CPU_CONTRACT_ONLY", "GPU_VERIFIED"}
            and not self.steps
        ):
            raise ValueError("live session result requires raw response bindings")
        if (
            self.audit.status == "GPU_VERIFIED"
            and self.verified_gpu_proof_sha256 is None
        ) or self.audit.reuse_authorized != (self.audit.status == "GPU_VERIFIED"):
            raise ValueError("live session GPU proof/reuse status is inconsistent")
        if self.verified_gpu_proof_sha256 is not None and (
            len(self.verified_gpu_proof_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.verified_gpu_proof_sha256
            )
        ):
            raise ValueError("live session GPU proof digest is invalid")
        if not self.execution_plan_sha256s or len(
            set(self.execution_plan_sha256s)
        ) != len(self.execution_plan_sha256s):
            raise ValueError("live session result requires unique ordered traces")
        for execution_plan_sha256 in self.execution_plan_sha256s:
            canonical_sha256({"execution_plan_sha256": execution_plan_sha256})
        for step in self.steps:
            if type(step) is not SessionLiveStepBinding:
                raise TypeError("live session steps must be exact typed bindings")
            if (
                canonical_sha256(_strict_json_load(step.raw_json))
                != step.content_sha256
            ):
                raise ValueError("live session raw response digest changed")
        expected_steps: list[tuple[str, str | None]] = [
            ("session_capability", None),
            ("session_initial_state", None),
        ]
        for execution_plan_sha256 in self.execution_plan_sha256s:
            expected_steps.extend(
                (
                    ("session_reset_boundary", execution_plan_sha256),
                    ("native_terminal_capability", execution_plan_sha256),
                    ("atomic_trace_begin", execution_plan_sha256),
                    ("atomic_trace_reset", execution_plan_sha256),
                    ("atomic_trace_finalize", execution_plan_sha256),
                )
            )
        expected_steps.append(("session_close_terminal", None))
        observed_steps = [
            (step.step, step.execution_plan_sha256) for step in self.steps
        ]
        if self.audit.status in {"CPU_CONTRACT_ONLY", "GPU_VERIFIED"}:
            if observed_steps != expected_steps:
                raise ValueError("successful live session step chain is incomplete")
        else:
            without_close = observed_steps
            has_close = bool(without_close) and without_close[-1] == (
                "session_close_terminal",
                None,
            )
            if has_close:
                without_close = without_close[:-1]
            expected_without_close = expected_steps[:-1]
            if without_close != expected_without_close[: len(without_close)] or (
                has_close and len(without_close) > len(expected_without_close)
            ):
                raise ValueError("failed live session step chain is not a valid prefix")
        if len(self.native_terminals) != len(self.audit.trace_receipt_sha256s):
            raise ValueError("validated native terminal coverage is incomplete")
        if (
            tuple(
                terminal.binding.execution_plan_sha256
                for terminal in self.native_terminals
            )
            != self.execution_plan_sha256s[: len(self.native_terminals)]
        ):
            raise ValueError("validated native terminals changed registered order")
        if self.audit.status in {"CPU_CONTRACT_ONLY", "GPU_VERIFIED"}:
            self._validate_complete_receipt_chain()

    def _validate_complete_receipt_chain(self) -> None:
        """Bind retained raw responses to the already parsed audit authorities."""

        by_name = {
            step.step: step for step in self.steps if step.execution_plan_sha256 is None
        }
        capability = _strict_json_mapping(
            by_name["session_capability"].raw_json,
            "session capability",
        )
        initial = _strict_json_mapping(
            by_name["session_initial_state"].raw_json,
            "session initial state",
        )
        close = _strict_json_mapping(
            by_name["session_close_terminal"].raw_json,
            "session close terminal",
        )
        _require_bound_digest(
            capability,
            digest_field="capability_sha256",
            field="session capability",
        )
        _require_bound_digest(
            initial,
            digest_field="initial_state_receipt_sha256",
            field="session initial state",
        )
        _require_bound_digest(
            close,
            digest_field="close_receipt_sha256",
            field="session close terminal",
        )
        if (
            capability.get("capability_sha256") != self.audit.capability_sha256
            or initial.get("initial_state_receipt_sha256")
            != self.audit.initial_state_receipt_sha256
            or close.get("close_receipt_sha256") != self.audit.close_receipt_sha256
            or capability.get("session_plan_sha256") != self.audit.session_plan_sha256
            or tuple(capability.get("execution_plan_sha256s", ()))
            != self.execution_plan_sha256s
        ):
            raise ValueError("live session raw authority changed after validation")

        trace_steps: dict[str, dict[str, SessionLiveStepBinding]] = {
            plan: {} for plan in self.execution_plan_sha256s
        }
        for step in self.steps:
            if step.execution_plan_sha256 is not None:
                trace_steps[step.execution_plan_sha256][step.step] = step
        prior_run_id: str | None = None
        for index, (plan, terminal) in enumerate(
            zip(self.execution_plan_sha256s, self.native_terminals, strict=True)
        ):
            terminal.binding.validate()
            if (
                terminal.binding.session_id != self.audit.session_plan_sha256
                or terminal.binding.session_epoch != index + 1
                or terminal.binding.previous_run_id != prior_run_id
            ):
                raise ValueError("validated native terminal session lineage changed")
            rows = trace_steps[plan]
            reset = _strict_json_mapping(
                rows["session_reset_boundary"].raw_json,
                "session reset boundary",
            )
            _require_bound_digest(
                reset,
                digest_field="reset_receipt_sha256",
                field="session reset boundary",
            )
            begin = _combined_response(
                rows["atomic_trace_begin"].raw_json,
                action="begin",
            )
            ready = _combined_response(
                rows["atomic_trace_reset"].raw_json,
                action="reset",
            )
            finalized = _combined_response(
                rows["atomic_trace_finalize"].raw_json,
                action="finalize",
            )
            ready_source = ready["source_owned_session"]
            finalized_source = finalized["source_owned_session"]
            source_begin = begin["source_owned_session"]
            _require_bound_digest(
                source_begin,
                digest_field="session_begin_sha256",
                field="atomic source trace begin",
            )
            _require_bound_digest(
                ready_source,
                digest_field="trace_ready_sha256",
                field="atomic source trace ready",
            )
            _require_bound_digest(
                finalized_source,
                digest_field="trace_receipt_sha256",
                field="atomic source trace finalize",
            )
            warmup = _exact_dict(ready_source.get("warmup_receipt"), "warm-up")
            clock = _exact_dict(ready_source.get("clock_receipt"), "scored clock")
            _require_bound_digest(
                warmup,
                digest_field="warmup_receipt_sha256",
                field="source warm-up",
            )
            _require_bound_digest(
                clock,
                digest_field="clock_receipt_sha256",
                field="source scored clock",
            )
            if (
                reset.get("reset_receipt_sha256")
                != self.audit.reset_receipt_sha256s[index]
                or warmup.get("warmup_receipt_sha256")
                != self.audit.warmup_receipt_sha256s[index]
                or clock.get("clock_receipt_sha256")
                != self.audit.clock_receipt_sha256s[index]
                or finalized_source.get("trace_receipt_sha256")
                != self.audit.trace_receipt_sha256s[index]
                or finalized_source.get("terminal_receipt_sha256")
                != terminal.terminal_sha256
                or source_begin.get("capability_sha256") != self.audit.capability_sha256
                or source_begin.get("execution_plan_sha256") != plan
                or source_begin.get("begin_sha256")
                != terminal.begin_receipt.begin_sha256
                or ready_source.get("capability_sha256") != self.audit.capability_sha256
                or clock.get("native_reset_sha256")
                != terminal.reset_receipt.reset_sha256
                or canonical_json_bytes(begin["terminal"]).decode("utf-8")
                != terminal.begin_receipt.raw_json
                or canonical_json_bytes(ready["terminal"]).decode("utf-8")
                != terminal.reset_receipt.raw_json
                or canonical_json_bytes(finalized["terminal"]).decode("utf-8")
                != terminal.raw_json
            ):
                raise ValueError("live session retained receipt chain was replaced")
            prior_run_id = terminal.binding.run_id

    @property
    def reuse_authorized(self) -> bool:
        self.validate()
        return self.audit.reuse_authorized


def _strict_json_load(value: str) -> object:
    import json

    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"canonical JSON contains duplicate key: {key}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> object:
        raise ValueError(f"canonical JSON contains non-finite value: {constant}")

    result = json.loads(
        value,
        object_pairs_hook=exact_object,
        parse_constant=reject_constant,
    )
    if canonical_json_bytes(result).decode("utf-8") != value:
        raise ValueError("live session response is not exact canonical JSON")
    return result


def _strict_json_mapping(value: str, field: str) -> dict[str, object]:
    return _exact_dict(_strict_json_load(value), field)


def _exact_dict(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{field} response must be one exact JSON object")
    return value


def _content_bound_response(
    value: object,
    *,
    keys: set[str],
    digest_field: str,
    field: str,
) -> dict[str, object]:
    raw = _exact_dict(value, field)
    if set(raw) != keys:
        raise ValueError(f"{field} response fields are incomplete or unexpected")
    _require_bound_digest(raw, digest_field=digest_field, field=field)
    return raw


def _require_bound_digest(
    raw: Mapping[str, object],
    *,
    digest_field: str,
    field: str,
) -> None:
    if digest_field not in raw:
        raise ValueError(f"{field} response lacks its content digest")
    observed = raw[digest_field]
    unsigned = dict(raw)
    unsigned.pop(digest_field)
    if not isinstance(observed, str) or observed != canonical_sha256(unsigned):
        raise ValueError(f"{field} response is not content-bound")


def _combined_response(value: str, *, action: str) -> dict[str, object]:
    combined = _strict_json_mapping(value, f"atomic trace {action}")
    if set(combined) != {"schema_version", "terminal", "source_owned_session"}:
        raise ValueError(f"atomic trace {action} response is incomplete")
    if type(combined["schema_version"]) is not int or combined["schema_version"] != 1:
        raise ValueError(f"atomic trace {action} response schema is unsupported")
    _exact_dict(combined["terminal"], f"atomic trace {action} terminal")
    _exact_dict(combined["source_owned_session"], f"atomic trace {action} source")
    return combined


class SessionLivePinnedBenchTransport(PinnedBenchServingTransport):
    """Pinned transport with only atomic session-terminal request routing added."""

    def __init__(
        self,
        *,
        evidence_sink: SessionLiveEvidenceSink | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if evidence_sink is not None and (
            not callable(getattr(evidence_sink, "record_step", None))
            or not callable(getattr(evidence_sink, "finalize", None))
            or not callable(getattr(evidence_sink, "close_partial", None))
        ):
            raise TypeError("live session evidence sink is incomplete")
        self._evidence_sink = evidence_sink
        self._live_capability_sha256: str | None = None
        self._live_execution_plan_sha256: str | None = None
        self._live_source_responses: dict[str, object] = {}
        self._live_steps: list[SessionLiveStepBinding] = []

    @property
    def live_steps(self) -> tuple[SessionLiveStepBinding, ...]:
        return tuple(self._live_steps)

    def bind_live_capability(self, capability_sha256: str) -> None:
        if self._live_capability_sha256 is not None:
            raise RuntimeError("live session capability is already bound")
        canonical_sha256({"capability_sha256": capability_sha256})
        self._live_capability_sha256 = capability_sha256

    def bind_live_trace(self, execution_plan_sha256: str) -> None:
        if self._live_execution_plan_sha256 is not None:
            raise RuntimeError("another live trace is already active")
        canonical_sha256({"execution_plan_sha256": execution_plan_sha256})
        self._live_execution_plan_sha256 = execution_plan_sha256

    def pop_live_source_response(self, action: str) -> object:
        try:
            return self._live_source_responses.pop(action)
        except KeyError as error:
            raise RuntimeError(
                f"missing atomic source-owned {action} response"
            ) from error

    def finish_live_trace(self) -> None:
        if self._live_source_responses:
            raise RuntimeError("live trace retained an unconsumed source receipt")
        if self._live_execution_plan_sha256 is None:
            raise RuntimeError("no live trace is active")
        self._live_execution_plan_sha256 = None

    def capture_live_response(
        self,
        step: str,
        value: object,
        *,
        execution_plan_sha256: str | None = None,
    ) -> None:
        binding = SessionLiveStepBinding.capture(
            step=step,
            value=value,
            execution_plan_sha256=execution_plan_sha256,
        )
        self._live_steps.append(binding)
        if self._evidence_sink is not None:
            self._evidence_sink.record_step(binding)

    async def get_json(self, path: str, /) -> object:
        value = await super().get_json(path)
        if path == CAPABILITY_PATH:
            self.capture_live_response(
                "native_terminal_capability",
                value,
                execution_plan_sha256=self._live_execution_plan_sha256,
            )
        return value

    async def post_json(
        self,
        path: str,
        body: Mapping[str, object],
        /,
    ) -> object:
        if path != TERMINAL_EVIDENCE_PATH:
            return await super().post_json(path, body)
        if set(body) != {"action", "payload"}:
            raise ValueError("native terminal action envelope is incomplete")
        action = body["action"]
        payload = body["payload"]
        if action not in _SESSION_TRACE_PATHS or not isinstance(payload, dict):
            raise ValueError("native terminal action is unsupported for live session")
        if self._live_capability_sha256 is None:
            raise RuntimeError("live terminal action lacks source capability")
        if self._live_execution_plan_sha256 is None:
            raise RuntimeError("live terminal action lacks an active trace")
        combined = await super().post_json(
            _SESSION_TRACE_PATHS[action],
            {
                "capability_sha256": self._live_capability_sha256,
                "terminal_payload": payload,
            },
        )
        if not isinstance(combined, dict) or set(combined) != {
            "schema_version",
            "terminal",
            "source_owned_session",
        }:
            raise ValueError("atomic session-terminal response is incomplete")
        if (
            type(combined["schema_version"]) is not int
            or combined["schema_version"] != 1
        ):
            raise ValueError("atomic session-terminal response schema is unsupported")
        if action in self._live_source_responses:
            raise RuntimeError("atomic source-owned response was replayed")
        self.capture_live_response(
            f"atomic_trace_{action}",
            combined,
            execution_plan_sha256=self._live_execution_plan_sha256,
        )
        self._live_source_responses[action] = combined["source_owned_session"]
        return combined["terminal"]


class _LiveAuditRuntime(SourceOwnedSessionAuditRuntime):
    def __init__(
        self,
        *,
        traces: tuple[SessionLiveTraceInput, ...],
        transport: SessionLivePinnedBenchTransport,
        provider: NativeTerminalProvider,
        process_owner: SessionLiveProcessOwner,
        terminal_observer: SessionLiveTerminalObserver | None,
    ) -> None:
        self._traces = traces
        self._transport = transport
        self._provider = provider
        self._process_owner = process_owner
        self._terminal_observer = terminal_observer
        self._trace_index = 0
        self._scored: tuple[TerminalRequestExpectation, ...] | None = None
        self._pending_clock: object | None = None
        self.native_terminals: list[ValidatedNativeTerminalEvidence] = []
        self.force_closed = False

    @property
    def _trace(self) -> SessionLiveTraceInput:
        if self._trace_index >= len(self._traces):
            raise RuntimeError("live audit has no pending trace")
        return self._traces[self._trace_index]

    async def capability(
        self,
        *,
        session_plan_sha256: str,
        execution_plan_sha256s: Sequence[str],
    ) -> object:
        value = await self._transport.post_json(
            _SESSION_CAPABILITY_PATH,
            {
                "session_plan_sha256": session_plan_sha256,
                "execution_plan_sha256s": list(execution_plan_sha256s),
            },
        )
        self._transport.capture_live_response("session_capability", value)
        if not isinstance(value, dict) or not isinstance(
            value.get("capability_sha256"), str
        ):
            raise TypeError("source-owned capability lacks its digest")
        self._transport.bind_live_capability(value["capability_sha256"])
        return value

    async def initial_state(self, *, capability_sha256: str) -> object:
        value = await self._transport.post_json(
            _SESSION_INITIAL_STATE_PATH,
            {"capability_sha256": capability_sha256},
        )
        self._transport.capture_live_response("session_initial_state", value)
        return value

    async def reset_boundary(
        self,
        *,
        capability_sha256: str,
        prior_execution_plan_sha256: str | None,
        next_execution_plan_sha256: str,
    ) -> object:
        trace = self._trace
        if trace.binding.execution_plan_sha256 != next_execution_plan_sha256:
            raise ValueError("live trace binding breaks the registered plan order")
        self._transport.bind_live_trace(next_execution_plan_sha256)
        value = await self._transport.post_json(
            _SESSION_RESET_PATH,
            {
                "capability_sha256": capability_sha256,
                "prior_execution_plan_sha256": prior_execution_plan_sha256,
                "next_execution_plan_sha256": next_execution_plan_sha256,
            },
        )
        self._transport.capture_live_response(
            "session_reset_boundary",
            value,
            execution_plan_sha256=next_execution_plan_sha256,
        )
        return value

    async def excluded_warmup(self, *, execution_plan_sha256: str) -> object:
        trace = self._trace
        if trace.binding.execution_plan_sha256 != execution_plan_sha256:
            raise ValueError("warm-up names another live trace")
        begin = await self._provider.begin(trace.binding)
        source_begin = _content_bound_response(
            self._transport.pop_live_source_response("begin"),
            keys={
                "schema_version",
                "hook",
                "capability_sha256",
                "execution_plan_sha256",
                "begin_sha256",
                "session_begin_sha256",
            },
            digest_field="session_begin_sha256",
            field="atomic source trace begin",
        )
        if (
            type(source_begin["schema_version"]) is not int
            or source_begin["schema_version"] != 1
            or source_begin["hook"] != SOURCE_OWNED_SESSION_HOOK
            or source_begin["capability_sha256"]
            != self._transport._live_capability_sha256
            or source_begin["execution_plan_sha256"] != execution_plan_sha256
            or source_begin["begin_sha256"] != begin.begin_sha256
        ):
            raise ValueError("atomic source begin is not bound to the native trace")
        warmup = tuple(
            await trace.driver.run_warmup(
                binding=trace.binding,
                transport=self._transport,
            )
        )
        native_reset = await self._provider.reset(warmup_requests=warmup)
        source = _content_bound_response(
            self._transport.pop_live_source_response("reset"),
            keys={
                "schema_version",
                "hook",
                "capability_sha256",
                "warmup_receipt",
                "clock_receipt",
                "trace_ready_sha256",
            },
            digest_field="trace_ready_sha256",
            field="atomic source trace ready",
        )
        clock = _exact_dict(source["clock_receipt"], "source scored clock")
        if (
            type(source["schema_version"]) is not int
            or source["schema_version"] != 1
            or source["hook"] != SOURCE_OWNED_SESSION_HOOK
            or source["capability_sha256"] != self._transport._live_capability_sha256
            or clock.get("native_reset_sha256") != native_reset.reset_sha256
        ):
            raise ValueError("source scored clock is not bound to the native reset")
        self._pending_clock = source["clock_receipt"]
        return source["warmup_receipt"]

    async def start_scored_clock(self, *, execution_plan_sha256: str) -> object:
        trace = self._trace
        if (
            trace.binding.execution_plan_sha256 != execution_plan_sha256
            or self._pending_clock is None
        ):
            raise RuntimeError("scored clock is not ready for this trace")
        return self._pending_clock

    async def finish_trace(self, *, execution_plan_sha256: str) -> object:
        trace = self._trace
        if (
            trace.binding.execution_plan_sha256 != execution_plan_sha256
            or self._pending_clock is None
        ):
            raise RuntimeError("live trace is not ready for finalization")
        # ``audit_source_owned_reuse_contract`` parsed the returned clock before
        # invoking this method, so no scored request can precede host validation
        # of its independent clock receipt.
        self._scored = tuple(
            await trace.driver.run_scored(
                binding=trace.binding,
                transport=self._transport,
            )
        )
        terminal = await self._provider.finalize(requests=self._scored)
        source = self._transport.pop_live_source_response("finalize")
        if not isinstance(source, dict) or (
            source.get("terminal_receipt_sha256") != terminal.terminal_sha256
        ):
            raise ValueError("source trace is not bound to validated terminal evidence")
        if self._terminal_observer is not None:
            await self._terminal_observer.terminal_finalized(
                trace_index=self._trace_index,
                binding=trace.binding,
                terminal=terminal,
            )
        self._transport.finish_live_trace()
        self.native_terminals.append(terminal)
        self._scored = None
        self._pending_clock = None
        self._trace_index += 1
        return source

    async def close(self, *, capability_sha256: str) -> object:
        value = await self._transport.post_json(
            _SESSION_CLOSE_PATH,
            {"capability_sha256": capability_sha256},
        )
        self._transport.capture_live_response("session_close_terminal", value)
        return value

    async def force_close(self) -> None:
        if not self.force_closed:
            await self._process_owner.force_close()
            self.force_closed = True


async def run_session_live_contract(
    *,
    session_plan_sha256: str,
    traces: Sequence[SessionLiveTraceInput],
    base_url: str,
    request_timeout_s: float,
    abort_timeout_s: float,
    transport: SessionLivePinnedBenchTransport,
    provider: NativeTerminalProvider,
    process_owner: SessionLiveProcessOwner,
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    terminal_observer: SessionLiveTerminalObserver | None = None,
) -> SessionLiveContractResult:
    """Exercise the live native chain, close it, and retain a blocked audit.

    The process must already be launched and ready.  This function owns the
    transport lifecycle and the terminal disposition of that process.
    """

    trace_values = tuple(traces)
    if not trace_values:
        raise ValueError("live session requires at least one trace")
    for trace in trace_values:
        if type(trace) is not SessionLiveTraceInput:
            raise TypeError("live session requires exact trace inputs")
        trace.validate()
    if type(transport) is not SessionLivePinnedBenchTransport:
        raise TypeError("live session requires the pinned atomic transport")
    if terminal_observer is not None and not callable(
        getattr(terminal_observer, "terminal_finalized", None)
    ):
        raise TypeError("live session terminal observer is incomplete")
    if (
        type(provider) is not NativeTerminalProvider
        or provider._transport is not transport
    ):
        raise TypeError("native terminal provider must own the same pinned HTTP pool")
    metrics = transport.metrics()
    if (
        type(metrics) is not dict
        or set(metrics)
        != {"connections_created", "submitted_requests", "reused_requests"}
        or any(type(item) is not int or item != 0 for item in metrics.values())
        or transport._session is not None
        or transport._native_admin_base_url is not None
        or transport._live_capability_sha256 is not None
        or transport._live_execution_plan_sha256 is not None
        or transport._live_source_responses
        or transport.live_steps
    ):
        raise ValueError("live session rejects a stale or previously used HTTP pool")
    if (
        provider.phase != "IDLE"
        or provider._binding is not None
        or provider._begin is not None
        or provider._reset is not None
        or provider._process is not None
        or provider._reset_generation != 0
        or provider._next_session_epoch != 1
        or provider._last_finalized_run_id is not None
        or provider._session_id is not None
        or provider._seen_runs
        or provider._seen_attempts
    ):
        raise ValueError("live session rejects stale native terminal state")
    execution_plan_sha256s = tuple(
        trace.binding.execution_plan_sha256 for trace in trace_values
    )
    if len(set(execution_plan_sha256s)) != len(execution_plan_sha256s):
        raise ValueError("live session execution plans must be unique")
    for index, trace in enumerate(trace_values):
        expected_previous = (
            None if index == 0 else trace_values[index - 1].binding.run_id
        )
        if (
            trace.binding.session_id != session_plan_sha256
            or trace.binding.session_epoch != index + 1
            or trace.binding.previous_run_id != expected_previous
        ):
            raise ValueError("native terminal binding breaks live session lineage")

    runtime = _LiveAuditRuntime(
        traces=trace_values,
        transport=transport,
        provider=provider,
        process_owner=process_owner,
        terminal_observer=terminal_observer,
    )
    transport_closed = False
    process_closed = False
    opened = False
    try:
        await transport.open(
            request_timeout_s=request_timeout_s,
            abort_timeout_s=abort_timeout_s,
        )
        opened = True
        transport.bind_native_admin_base_url(base_url)
        audit = await audit_source_owned_reuse_contract(
            session_plan_sha256=session_plan_sha256,
            execution_plan_sha256s=execution_plan_sha256s,
            runtime=runtime,
            verified_gpu_proof=verified_gpu_proof,
        )
        if audit.status in {"CPU_CONTRACT_ONLY", "GPU_VERIFIED"}:
            await transport.close()
            transport_closed = True
            await process_owner.close()
            process_closed = True
        else:
            if opened and transport._session is not None:
                await transport.close()
                transport_closed = True
            await runtime.force_close()
    except BaseException as error:  # cancellation must clean owned resources
        cleanup_errors: list[BaseException] = []
        if opened and transport._session is not None:
            try:
                await asyncio.shield(transport.close())
                transport_closed = True
            except BaseException as close_error:  # noqa: BLE001 - preserve root failure
                cleanup_errors.append(close_error)
        try:
            await asyncio.shield(runtime.force_close())
        except BaseException as force_error:  # noqa: BLE001 - preserve root failure
            cleanup_errors.append(force_error)
        if transport._evidence_sink is not None:
            try:
                transport._evidence_sink.close_partial()
            except BaseException as sink_error:  # noqa: BLE001 - preserve root failure
                cleanup_errors.append(sink_error)
        for cleanup_error in cleanup_errors:
            error.add_note(
                "live session cleanup failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise

    value = SessionLiveContractResult(
        audit=audit,
        execution_plan_sha256s=execution_plan_sha256s,
        steps=transport.live_steps,
        native_terminals=tuple(runtime.native_terminals),
        transport_closed=transport_closed,
        process_closed=process_closed,
        process_force_closed=runtime.force_closed,
        verified_gpu_proof_sha256=(
            None if verified_gpu_proof is None else verified_gpu_proof.sha256
        ),
    )
    value.validate()
    if transport._evidence_sink is not None:
        try:
            transport._evidence_sink.finalize(value)
        except BaseException:
            transport._evidence_sink.close_partial()
            raise
    return value


__all__ = (
    "SessionLiveContractResult",
    "SessionLiveEvidenceSink",
    "SessionLivePinnedBenchTransport",
    "SessionLiveProcessOwner",
    "SessionLiveStepBinding",
    "SessionLiveTerminalObserver",
    "SessionLiveTraceDriver",
    "SessionLiveTraceInput",
    "run_session_live_contract",
)
