"""Source-owned E5 fault primitives and an all-rank CPU lifecycle harness.

This is the layer below :mod:`failure_authority`: it specifies what each of the
eleven registered failures means operationally.  It deliberately does not
unlock production.  The release capability map is empty, and the release entry
point blocks before invoking an actuator or creating an output file.

The executable function in this module is a deterministic CPU-test harness. It
accepts only the exact source specification for the plan scenario, validates
arm/trigger/proof/recover/terminal observations on every rank of both registered
topologies, and publishes a diagnostic receipt only after complete recovery.
That receipt explicitly carries no formal execution authority.
"""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NoReturn, Protocol

from lightcone_spec.experiments.failure_authority import (
    FailureInjectionAuthorityResult,
    ReleaseFailurePlan,
)
from lightcone_spec.experiments.registry import E5_FAILURES, content_sha256

FAILURE_ACTUATOR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e5_first_party_failure_actuator_protocol",
        "scenarios": list(E5_FAILURES),
        "phases": ["arm", "trigger", "proof", "recover", "terminal"],
        "coverage": "both_registered_topologies_and_every_rank",
        "publication": "atomic_after_recovery_only",
        "caller_operation_or_proof_forbidden": True,
        "formal_execution_available": False,
    }
)
FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON = (
    "failure_actuator_release_capability_unavailable"
)

# Source-owned.  A future release may add one exact implementation identity
# only together with GPU/device tests and the trusted terminal signer.
RELEASE_FAILURE_ACTUATOR_CAPABILITIES: tuple[tuple[str, str], ...] = ()

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
FailurePhase = Literal["arm", "trigger", "proof", "recover", "terminal"]


class FailureActuatorBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"E5 first-party fault actuation is BLOCKED: {reason}")
        self.reason = reason


def _require_sha256(label: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _require_safe(label: str, value: object) -> str:
    if type(value) is not str or _SAFE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _require_nonnegative(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class FailureScenarioSemantics:
    scenario: str
    arm_operation: str
    trigger_operation: str
    proof_event: str
    recovery_invariant: str
    terminal_counter: str
    parameters: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.scenario not in E5_FAILURES:
            raise ValueError("fault semantics names an unregistered scenario")
        for label, value in (
            ("arm operation", self.arm_operation),
            ("trigger operation", self.trigger_operation),
            ("proof event", self.proof_event),
            ("recovery invariant", self.recovery_invariant),
            ("terminal counter", self.terminal_counter),
        ):
            _require_safe(label, value)
        names = tuple(name for name, _ in self.parameters)
        if names != tuple(sorted(set(names))) or not self.parameters:
            raise ValueError("fault semantics parameters must be sorted and unique")
        for name, value in self.parameters:
            _require_safe("fault parameter name", name)
            _require_safe("fault parameter value", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "arm_operation": self.arm_operation,
            "trigger_operation": self.trigger_operation,
            "proof_event": self.proof_event,
            "recovery_invariant": self.recovery_invariant,
            "terminal_counter": self.terminal_counter,
            "parameters": [list(value) for value in self.parameters],
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _semantics(
    scenario: str,
    arm: str,
    trigger: str,
    proof: str,
    recover: str,
    counter: str,
    **parameters: str,
) -> FailureScenarioSemantics:
    return FailureScenarioSemantics(
        scenario,
        arm,
        trigger,
        proof,
        recover,
        counter,
        tuple(sorted(parameters.items())),
    )


# These are concrete actuator primitives, not prose summaries.  Changing any
# token changes the protocol digest below and requires review of every receipt.
FAILURE_SCENARIO_SEMANTICS: tuple[FailureScenarioSemantics, ...] = (
    _semantics(
        "queue_saturation",
        "admission_queue_hold_capacity_one",
        "submit_request_beyond_capacity",
        "admission_rejected_queue_full",
        "release_hold_and_queue_depth_zero",
        "admission_rejections",
        capacity="1",
        overflow_requests="1",
    ),
    _semantics(
        "cancellation",
        "hold_request_before_first_token",
        "abort_exact_request_id",
        "cancelled_terminal_without_post_abort_commit",
        "inflight_zero_and_abort_acknowledged",
        "cancellations",
        abort_grace="registered",
        target_state="pre_first_token",
    ),
    _semantics(
        "duplicate_retry",
        "record_idempotency_key_and_terminal_digest",
        "resubmit_identical_key_and_payload",
        "duplicate_suppressed_without_second_execution",
        "one_terminal_record_and_same_digest",
        "duplicate_retry_suppressions",
        duplicate_count="1",
        payload_relation="byte_identical",
    ),
    _semantics(
        "nonfinite_candidate",
        "freeze_finite_active_candidate_version",
        "inject_nan_before_candidate_publish",
        "nonfinite_rejected_and_version_unchanged",
        "clear_candidate_and_active_state_finite",
        "nonfinite_updates",
        injection_value="nan",
        publication_stage="pre_publish",
    ),
    _semantics(
        "oom_candidate",
        "arm_fail_once_candidate_allocator",
        "allocate_candidate_scratch",
        "oom_caught_and_active_version_unchanged",
        "release_scratch_and_allocator_probe_succeeds",
        "oom_events",
        failure_count="1",
        injection_site="candidate_scratch",
    ),
    _semantics(
        "evidence_backpressure",
        "pause_consumer_with_queue_capacity_one",
        "enqueue_evidence_beyond_capacity",
        "backpressure_observed_without_evidence_drop",
        "resume_consumer_flush_and_queue_zero",
        "evidence_backpressure_events",
        capacity="1",
        overflow_mode="block",
    ),
    _semantics(
        "disk_quota",
        "arm_fail_once_checkpoint_fsync_enospc",
        "publish_checkpoint_boundary",
        "enospc_observed_without_terminal_commit",
        "remove_fault_and_recovery_checkpoint_fsyncs",
        "disk_quota_events",
        errno="ENOSPC",
        failure_count="1",
    ),
    _semantics(
        "slow_rank",
        "arm_rank_one_collective_delay",
        "enter_publication_barrier",
        "slow_rank_detected_without_partial_apply",
        "remove_delay_and_barrier_versions_converge",
        "slow_rank_events",
        delayed_rank="1",
        delay="registered_timeout_plus_one",
    ),
    _semantics(
        "communicator_failure",
        "arm_fail_once_prepare_collective",
        "enter_all_rank_prepare_collective",
        "collective_aborted_on_all_ranks_without_apply",
        "fresh_communicator_generation_barrier_converges",
        "communicator_failures",
        collective="prepare",
        failure_count="1",
    ),
    _semantics(
        "replica_drain",
        "arm_replica_one_with_inflight_request",
        "mark_replica_one_draining",
        "new_admissions_fenced_and_inflight_terminal",
        "zero_inflight_then_route_set_restored",
        "replica_drain_events",
        replica="1",
        route_policy="fence_then_restore",
    ),
    _semantics(
        "replica_restart",
        "record_replica_one_process_generation",
        "terminate_and_restart_replica_one",
        "old_generation_fenced_and_new_process_observed",
        "new_generation_ready_then_route_restored",
        "replica_restart_events",
        generation_delta="1",
        replica="1",
    ),
)

if tuple(value.scenario for value in FAILURE_SCENARIO_SEMANTICS) != E5_FAILURES:
    raise AssertionError("E5 fault semantics no longer exactly match the registry")
if len({value.sha256 for value in FAILURE_SCENARIO_SEMANTICS}) != len(E5_FAILURES):
    raise AssertionError("E5 fault semantics identities are not unique")

FAILURE_SCENARIO_SEMANTICS_SHA256 = content_sha256(
    [value.to_dict() for value in FAILURE_SCENARIO_SEMANTICS]
)


def failure_semantics(scenario: str) -> FailureScenarioSemantics:
    for value in FAILURE_SCENARIO_SEMANTICS:
        if value.scenario == scenario:
            return value
    raise ValueError("fault scenario has no source-owned semantics")


@dataclass(frozen=True)
class FailureActuatorContext:
    plan_sha256: str
    scenario_semantics_sha256: str
    topology: str
    rank: int
    world_size: int
    process_id: int
    process_start_monotonic_ns: int
    session_epoch: int
    run_nonce_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("fault context plan", self.plan_sha256)
        _require_sha256("fault context semantics", self.scenario_semantics_sha256)
        _require_sha256("fault context nonce", self.run_nonce_sha256)
        if self.topology not in {"tp2_dp1", "two_replica_tp1_dp2"}:
            raise ValueError("fault context topology is unsupported")
        if self.world_size != 2 or self.rank not in range(self.world_size):
            raise ValueError("fault context rank coverage is invalid")
        if type(self.process_id) is not int or self.process_id < 1:
            raise ValueError("fault context process ID is invalid")
        if (
            type(self.process_start_monotonic_ns) is not int
            or self.process_start_monotonic_ns < 1
        ):
            raise ValueError("fault context process start is invalid")
        if self.session_epoch != 0:
            raise ValueError("fault actuation requires a fresh process epoch")


@dataclass(frozen=True)
class FailurePhaseObservation:
    phase: FailurePhase
    operation: str
    monotonic_ns: int
    event_count: int
    observation_sha256: str

    def __post_init__(self) -> None:
        if self.phase not in {"arm", "trigger", "proof", "recover", "terminal"}:
            raise ValueError("fault phase observation is unsupported")
        _require_safe("fault phase operation", self.operation)
        if type(self.monotonic_ns) is not int or self.monotonic_ns < 1:
            raise ValueError("fault phase time is invalid")
        _require_nonnegative("fault phase event count", self.event_count)
        _require_sha256("fault phase observation", self.observation_sha256)


@dataclass(frozen=True)
class FailureTerminalObservation:
    phase: Literal["terminal"]
    operation: str
    monotonic_ns: int
    event_count: int
    observation_sha256: str
    counters: tuple[tuple[str, int], ...]
    recovery_valid: bool

    def __post_init__(self) -> None:
        FailurePhaseObservation(
            self.phase,
            self.operation,
            self.monotonic_ns,
            self.event_count,
            self.observation_sha256,
        )
        names = tuple(name for name, _ in self.counters)
        if names != tuple(sorted(set(names))):
            raise ValueError("fault terminal counters must be sorted and unique")
        for name, value in self.counters:
            _require_safe("fault terminal counter", name)
            _require_nonnegative("fault terminal counter value", value)
        if self.recovery_valid is not True:
            raise ValueError("fault terminal observation did not prove recovery")


class FirstPartyFailureActuator(Protocol):
    actuator_id: str
    actuator_version_sha256: str

    def fresh_context(
        self,
        *,
        plan: ReleaseFailurePlan,
        semantics: FailureScenarioSemantics,
        topology: str,
        rank: int,
        run_nonce_sha256: str,
    ) -> FailureActuatorContext: ...

    def arm(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailurePhaseObservation: ...

    def trigger(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailurePhaseObservation: ...

    def prove(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailurePhaseObservation: ...

    def recover(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailurePhaseObservation: ...

    def terminal(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailureTerminalObservation: ...


def failure_phase_observation_sha256(
    context: FailureActuatorContext,
    semantics: FailureScenarioSemantics,
    *,
    phase: FailurePhase,
    operation: str,
    event_count: int,
) -> str:
    """Canonical identity for one source-owned rank/phase observation."""

    context.__post_init__()
    semantics.__post_init__()
    if context.scenario_semantics_sha256 != semantics.sha256:
        raise ValueError("fault observation context uses other scenario semantics")
    if phase not in {"arm", "trigger", "proof", "recover", "terminal"}:
        raise ValueError("fault observation phase is unsupported")
    _require_safe("fault observation operation", operation)
    _require_nonnegative("fault observation event count", event_count)
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "e5_first_party_failure_phase_observation",
            "protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
            "plan_sha256": context.plan_sha256,
            "scenario_semantics_sha256": semantics.sha256,
            "topology": context.topology,
            "rank": context.rank,
            "world_size": context.world_size,
            "process_id": context.process_id,
            "process_start_monotonic_ns": context.process_start_monotonic_ns,
            "session_epoch": context.session_epoch,
            "run_nonce_sha256": context.run_nonce_sha256,
            "phase": phase,
            "operation": operation,
            "event_count": event_count,
        }
    )


def require_release_failure_actuator(
    authority: FailureInjectionAuthorityResult | None = None,
) -> NoReturn:
    """Block before calling an actuator or creating an output in this release."""

    if authority is not None:
        if type(authority) is not FailureInjectionAuthorityResult:
            raise TypeError("fault actuator gate requires an exact authority result")
        authority.plan.__post_init__()
    if not RELEASE_FAILURE_ACTUATOR_CAPABILITIES:
        raise FailureActuatorBlocked(FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON)
    raise FailureActuatorBlocked("failure_actuator_trusted_signer_unavailable")


@dataclass(frozen=True)
class FailureActuatorRankReceipt:
    topology: str
    rank: int
    process_id: int
    process_start_monotonic_ns: int
    session_epoch: int
    phases: tuple[FailurePhaseObservation, ...]
    counters: tuple[tuple[str, int], ...]
    recovery_valid: bool

    def __post_init__(self) -> None:
        if self.topology not in {"tp2_dp1", "two_replica_tp1_dp2"}:
            raise ValueError("fault rank receipt topology is unsupported")
        if self.rank not in {0, 1}:
            raise ValueError("fault rank receipt rank is unsupported")
        if type(self.process_id) is not int or self.process_id < 1:
            raise ValueError("fault rank receipt process ID is invalid")
        if (
            type(self.process_start_monotonic_ns) is not int
            or self.process_start_monotonic_ns < 1
            or self.session_epoch != 0
        ):
            raise ValueError("fault rank receipt does not prove a fresh process")
        if tuple(value.phase for value in self.phases) != (
            "arm",
            "trigger",
            "proof",
            "recover",
            "terminal",
        ):
            raise ValueError("fault rank receipt phases are incomplete")
        for value in self.phases:
            value.__post_init__()
        names = tuple(name for name, _ in self.counters)
        if names != tuple(sorted(set(names))):
            raise ValueError("fault rank receipt counters are incomplete")
        if self.recovery_valid is not True:
            raise ValueError("fault rank receipt did not prove recovery")

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "rank": self.rank,
            "process_id": self.process_id,
            "process_start_monotonic_ns": self.process_start_monotonic_ns,
            "session_epoch": self.session_epoch,
            "phases": [asdict(value) for value in self.phases],
            "counters": {name: value for name, value in self.counters},
            "recovery_valid": self.recovery_valid,
        }


@dataclass(frozen=True)
class DiagnosticFailureActuationReceipt:
    schema_version: int
    kind: str
    protocol_sha256: str
    plan_sha256: str
    authority_sha256: str
    scenario: str
    scenario_semantics_sha256: str
    semantics_universe_sha256: str
    run_nonce_sha256: str
    actuator_id: str
    actuator_version_sha256: str
    rank_receipts: tuple[FailureActuatorRankReceipt, ...]
    recovered: bool
    committed: bool
    correctness_only: bool
    formal_execution_authorized: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "e5_diagnostic_fault_actuation":
            raise ValueError("diagnostic fault receipt schema is unsupported")
        for label, value in (
            ("fault receipt protocol", self.protocol_sha256),
            ("fault receipt plan", self.plan_sha256),
            ("fault receipt authority", self.authority_sha256),
            ("fault receipt semantics", self.scenario_semantics_sha256),
            ("fault receipt semantics universe", self.semantics_universe_sha256),
            ("fault receipt nonce", self.run_nonce_sha256),
            ("fault receipt actuator version", self.actuator_version_sha256),
        ):
            _require_sha256(label, value)
        if self.protocol_sha256 != FAILURE_ACTUATOR_PROTOCOL_SHA256:
            raise ValueError("diagnostic fault receipt uses another protocol")
        if self.semantics_universe_sha256 != FAILURE_SCENARIO_SEMANTICS_SHA256:
            raise ValueError("diagnostic fault receipt uses another semantics universe")
        semantics = failure_semantics(self.scenario)
        if semantics.sha256 != self.scenario_semantics_sha256:
            raise ValueError("diagnostic fault receipt semantics identity differs")
        _require_safe("fault receipt actuator", self.actuator_id)
        coverage = tuple((value.topology, value.rank) for value in self.rank_receipts)
        if coverage != (
            ("tp2_dp1", 0),
            ("tp2_dp1", 1),
            ("two_replica_tp1_dp2", 0),
            ("two_replica_tp1_dp2", 1),
        ):
            raise ValueError("diagnostic fault receipt lacks atomic all-rank coverage")
        for value in self.rank_receipts:
            value.__post_init__()
        if self.recovered is not True or self.committed is not True:
            raise ValueError("diagnostic fault receipt is not recovered and committed")
        if (
            self.correctness_only is not True
            or self.formal_execution_authorized is not False
        ):
            raise ValueError("diagnostic fault receipt cannot authorize formal claims")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "plan_sha256": self.plan_sha256,
            "authority_sha256": self.authority_sha256,
            "scenario": self.scenario,
            "scenario_semantics_sha256": self.scenario_semantics_sha256,
            "semantics_universe_sha256": self.semantics_universe_sha256,
            "run_nonce_sha256": self.run_nonce_sha256,
            "actuator_id": self.actuator_id,
            "actuator_version_sha256": self.actuator_version_sha256,
            "rank_receipts": [value.to_dict() for value in self.rank_receipts],
            "recovered": self.recovered,
            "committed": self.committed,
            "correctness_only": self.correctness_only,
            "formal_execution_authorized": self.formal_execution_authorized,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


_UNIVERSAL_ZERO_COUNTERS = {
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "partial_target_continuations",
    "retractions",
}


def _validate_rank_lifecycle(
    *,
    plan: ReleaseFailurePlan,
    semantics: FailureScenarioSemantics,
    context: FailureActuatorContext,
    phases: tuple[FailurePhaseObservation, ...],
    terminal: FailureTerminalObservation,
) -> None:
    expected_operations = (
        ("arm", semantics.arm_operation),
        ("trigger", semantics.trigger_operation),
        ("proof", semantics.proof_event),
        ("recover", semantics.recovery_invariant),
        ("terminal", semantics.terminal_counter),
    )
    observed = tuple((value.phase, value.operation) for value in (*phases, terminal))
    if observed != expected_operations:
        raise ValueError("fault lifecycle differs from source-owned scenario semantics")
    for value in (*phases, terminal):
        expected_sha256 = failure_phase_observation_sha256(
            context,
            semantics,
            phase=value.phase,
            operation=value.operation,
            event_count=value.event_count,
        )
        if value.observation_sha256 != expected_sha256:
            raise ValueError("fault phase observation identity is not canonical")
    times = tuple(value.monotonic_ns for value in (*phases, terminal))
    if (
        times != tuple(sorted(set(times)))
        or times[0] <= context.process_start_monotonic_ns
    ):
        raise ValueError("fault lifecycle times are incomplete or unordered")
    arm, trigger, _proof, recover, terminal_time = times
    windows = (
        (context.process_start_monotonic_ns, arm, plan.lifecycle.arm_timeout_ms),
        (arm, trigger, plan.lifecycle.trigger_timeout_ms),
        (trigger, recover, plan.lifecycle.recover_timeout_ms),
        (recover, terminal_time, plan.lifecycle.terminal_timeout_ms),
    )
    if any(right - left > timeout * 1_000_000 for left, right, timeout in windows):
        raise ValueError("fault lifecycle exceeds the registered timeout")
    if phases[1].event_count != 1 or phases[2].event_count != 1:
        raise ValueError("fault trigger and proof must each observe exactly one event")
    if phases[3].event_count != 1 or terminal.event_count != 1:
        raise ValueError("fault recovery and terminal proof must be singular")
    counters = dict(terminal.counters)
    expected_names = {value.name for value in plan.expected_counters}
    if set(counters) != expected_names:
        raise ValueError("fault terminal counter coverage is incomplete")
    if any(counters[name] != 0 for name in _UNIVERSAL_ZERO_COUNTERS):
        raise ValueError("fault lifecycle violates a universal safety counter")


def validate_failure_recovery_receipt(
    receipt: DiagnosticFailureActuationReceipt,
    authority: FailureInjectionAuthorityResult,
) -> None:
    """Recheck exact plan, semantics, all-rank recovery, and counter coverage."""

    if type(receipt) is not DiagnosticFailureActuationReceipt:
        raise TypeError("fault recovery validation requires an exact receipt")
    if type(authority) is not FailureInjectionAuthorityResult:
        raise TypeError("fault recovery validation requires exact authority")
    receipt.__post_init__()
    plan = authority.plan
    semantics = failure_semantics(plan.scenario)
    if (
        receipt.plan_sha256 != plan.sha256
        or receipt.authority_sha256 != authority.binding.sha256
        or receipt.scenario != plan.scenario
        or receipt.scenario_semantics_sha256 != semantics.sha256
    ):
        raise ValueError("fault recovery receipt belongs to another authority")
    expected_counters = {value.name: value for value in plan.expected_counters}
    for target in plan.topology_targets:
        ranks = tuple(
            value
            for value in receipt.rank_receipts
            if value.topology == target.topology
        )
        if tuple(value.rank for value in ranks) != target.target_ranks:
            raise ValueError("fault recovery receipt has partial rank coverage")
        trigger_total = 0
        for rank in ranks:
            counters = dict(rank.counters)
            if set(counters) != set(expected_counters):
                raise ValueError("fault recovery receipt counter coverage differs")
            for name, expectation in expected_counters.items():
                if (
                    expectation.comparison == "eq"
                    and counters[name] != expectation.value
                ):
                    raise ValueError(f"fault recovery safety counter {name} differs")
            trigger_total += counters[semantics.terminal_counter]
        if trigger_total < 1:
            raise ValueError(
                "fault recovery receipt lacks the topology trigger counter"
            )


def _publish_atomic(path: Path, value: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def execute_failure_actuator_for_cpu_test(
    authority: FailureInjectionAuthorityResult,
    actuator: FirstPartyFailureActuator,
    *,
    run_nonce_sha256: str,
    receipt_path: str | Path,
) -> DiagnosticFailureActuationReceipt:
    """Run the exact diagnostic lifecycle; publish only after all-rank recovery."""

    if type(authority) is not FailureInjectionAuthorityResult:
        raise TypeError("CPU fault lifecycle requires exact authority")
    plan = authority.plan
    plan.__post_init__()
    semantics = failure_semantics(plan.scenario)
    _require_sha256("fault CPU nonce", run_nonce_sha256)
    _require_safe("fault CPU actuator", actuator.actuator_id)
    _require_sha256("fault CPU actuator version", actuator.actuator_version_sha256)
    output = Path(receipt_path)
    if (
        not output.is_absolute()
        or output != output.resolve(strict=False)
        or output == Path(output.anchor)
        or output.exists()
        or output.parent.is_symlink()
        or not output.parent.is_dir()
    ):
        raise ValueError("fault CPU receipt path must be a new normalized file")

    rank_receipts: list[FailureActuatorRankReceipt] = []
    topology_trigger_totals: dict[str, int] = {}
    for target in plan.topology_targets:
        trigger_total = 0
        process_ids: set[int] = set()
        for rank in target.target_ranks:
            context = actuator.fresh_context(
                plan=plan,
                semantics=semantics,
                topology=target.topology,
                rank=rank,
                run_nonce_sha256=run_nonce_sha256,
            )
            context.__post_init__()
            if (
                context.plan_sha256 != plan.sha256
                or context.scenario_semantics_sha256 != semantics.sha256
                or context.topology != target.topology
                or context.rank != rank
                or context.world_size != target.world_size
                or context.run_nonce_sha256 != run_nonce_sha256
            ):
                raise ValueError("fault actuator returned a foreign execution context")
            if context.process_id in process_ids:
                raise ValueError(
                    "fault actuator reused a process identity across ranks"
                )
            process_ids.add(context.process_id)
            arm = actuator.arm(context, semantics)
            trigger = actuator.trigger(context, semantics)
            proof = actuator.prove(context, semantics)
            recover = actuator.recover(context, semantics)
            terminal = actuator.terminal(context, semantics)
            for value in (arm, trigger, proof, recover):
                value.__post_init__()
            terminal.__post_init__()
            _validate_rank_lifecycle(
                plan=plan,
                semantics=semantics,
                context=context,
                phases=(arm, trigger, proof, recover),
                terminal=terminal,
            )
            counters = dict(terminal.counters)
            trigger_total += counters[semantics.terminal_counter]
            rank_receipts.append(
                FailureActuatorRankReceipt(
                    topology=target.topology,
                    rank=rank,
                    process_id=context.process_id,
                    process_start_monotonic_ns=context.process_start_monotonic_ns,
                    session_epoch=context.session_epoch,
                    phases=(
                        arm,
                        trigger,
                        proof,
                        recover,
                        FailurePhaseObservation(
                            terminal.phase,
                            terminal.operation,
                            terminal.monotonic_ns,
                            terminal.event_count,
                            terminal.observation_sha256,
                        ),
                    ),
                    counters=terminal.counters,
                    recovery_valid=terminal.recovery_valid,
                )
            )
        topology_trigger_totals[target.topology] = trigger_total
    if any(value < 1 for value in topology_trigger_totals.values()):
        raise ValueError("fault trigger counter is missing from a registered topology")

    receipt = DiagnosticFailureActuationReceipt(
        schema_version=1,
        kind="e5_diagnostic_fault_actuation",
        protocol_sha256=FAILURE_ACTUATOR_PROTOCOL_SHA256,
        plan_sha256=plan.sha256,
        authority_sha256=authority.binding.sha256,
        scenario=plan.scenario,
        scenario_semantics_sha256=semantics.sha256,
        semantics_universe_sha256=FAILURE_SCENARIO_SEMANTICS_SHA256,
        run_nonce_sha256=run_nonce_sha256,
        actuator_id=actuator.actuator_id,
        actuator_version_sha256=actuator.actuator_version_sha256,
        rank_receipts=tuple(rank_receipts),
        recovered=True,
        committed=True,
        correctness_only=True,
        formal_execution_authorized=False,
    )
    receipt.__post_init__()
    validate_failure_recovery_receipt(receipt, authority)
    _publish_atomic(output, receipt.to_dict())
    metadata = output.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or output.read_bytes() != (
        json.dumps(
            receipt.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    ):
        raise ValueError("fault diagnostic receipt changed after publication")
    return receipt


__all__ = [
    "FAILURE_ACTUATOR_PROTOCOL_SHA256",
    "FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON",
    "FAILURE_SCENARIO_SEMANTICS",
    "FAILURE_SCENARIO_SEMANTICS_SHA256",
    "RELEASE_FAILURE_ACTUATOR_CAPABILITIES",
    "DiagnosticFailureActuationReceipt",
    "FailureActuatorBlocked",
    "FailureActuatorContext",
    "FailureActuatorRankReceipt",
    "FailurePhaseObservation",
    "FailureScenarioSemantics",
    "FailureTerminalObservation",
    "FirstPartyFailureActuator",
    "execute_failure_actuator_for_cpu_test",
    "failure_phase_observation_sha256",
    "failure_semantics",
    "require_release_failure_actuator",
    "validate_failure_recovery_receipt",
]
