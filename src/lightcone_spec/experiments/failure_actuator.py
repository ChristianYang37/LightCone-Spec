"""Source-owned E5 fault primitives and external-control evidence.

The remote GPU host receives no release private key.  It executes the eleven
registered child-process-scoped fault lifecycles and emits one canonical,
explicitly unsigned all-rank recovery terminal.  A local release coordinator can
then bind that exact terminal with a signed control envelope and a durable,
single-use replay reservation.  The CPU harness exercises the same lifecycle
without granting formal evidence authority.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NoReturn, Protocol
from urllib.parse import urlsplit

from lightcone_spec.experiments.failure_authority import (
    FailureExecutionAuthorityToken,
    FailureInjectionAuthorityResult,
    ReleaseFailurePlan,
    require_failure_execution_lifecycle,
)
from lightcone_spec.experiments.registry import (
    E5_FAILURES,
    ExperimentCell,
    content_sha256,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FAILURE_ACTUATOR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e5_first_party_failure_actuator_protocol",
        "scenarios": list(E5_FAILURES),
        "phases": ["arm", "trigger", "proof", "recover", "terminal"],
        "coverage": "all_three_registered_topologies_and_every_rank",
        "publication": "atomic_after_recovery_only",
        "caller_operation_or_proof_forbidden": True,
        "formal_execution": "dynamic_external_control_after_raw_recovery",
    }
)
FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON = (
    "failure_actuator_release_capability_unavailable"
)

# This is the sole release-owned failure-actuator registry.  A reviewed future
# release may add an entry only together with its concrete factory, GPU/device
# tests, executor lifecycle wiring, and trusted terminal signer.  Keeping the
# callable beside its identity prevents two independent allowlists from
# unlocking each other accidentally.
type FailureActuatorFactory = Callable[..., "FirstPartyFailureActuator"]


@dataclass(frozen=True)
class ReleaseFailureActuatorCapability:
    actuator_id: str
    actuator_version_sha256: str
    factory_module: str
    factory_qualname: str
    factory: FailureActuatorFactory

    def __post_init__(self) -> None:
        _require_safe("release failure actuator", self.actuator_id)
        if self.actuator_id.lower().startswith(("cpu", "test", "fixture")):
            raise ValueError("diagnostic actuator identities cannot enter release")
        _require_sha256(
            "release failure actuator version", self.actuator_version_sha256
        )
        _require_safe("release failure actuator factory module", self.factory_module)
        _require_safe(
            "release failure actuator factory qualname", self.factory_qualname
        )
        if not callable(self.factory):
            raise TypeError("release failure actuator factory must be callable")
        if not self.factory_module.startswith("lightcone_spec."):
            raise ValueError("release failure actuator factory is not source-owned")
        if (
            getattr(self.factory, "__module__", None) != self.factory_module
            or getattr(self.factory, "__qualname__", None) != self.factory_qualname
        ):
            raise ValueError("release failure actuator factory identity differs")

    @property
    def sha256(self) -> str:
        self.__post_init__()
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "release_failure_actuator_capability",
                "actuator_id": self.actuator_id,
                "actuator_version_sha256": self.actuator_version_sha256,
                "factory_module": self.factory_module,
                "factory_qualname": self.factory_qualname,
            }
        )


RELEASE_FAILURE_ACTUATOR_CAPABILITIES: tuple[ReleaseFailureActuatorCapability, ...]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
FailurePhase = Literal["arm", "trigger", "proof", "recover", "terminal"]


class FailureActuatorBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"E5 first-party fault actuation is BLOCKED: {reason}")
        self.reason = reason


def release_failure_actuator_capability() -> ReleaseFailureActuatorCapability:
    """Resolve the one source-owned identity and callable as one capability."""

    if (
        len(RELEASE_FAILURE_ACTUATOR_CAPABILITIES) != 1
        or type(RELEASE_FAILURE_ACTUATOR_CAPABILITIES[0])
        is not ReleaseFailureActuatorCapability
    ):
        raise FailureActuatorBlocked(FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON)
    capability = RELEASE_FAILURE_ACTUATOR_CAPABILITIES[0]
    try:
        capability.__post_init__()
    except (TypeError, ValueError) as error:
        raise FailureActuatorBlocked(
            FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON
        ) from error
    return capability


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
        if self.topology not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("fault context topology is unsupported")
        expected_world_size = 1 if self.topology == "tp1_dp1" else 2
        if self.world_size != expected_world_size or self.rank not in range(
            self.world_size
        ):
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
class FailureRankLaunchBinding:
    """Exact child-process scope for one rank; never a host-wide fault target."""

    topology: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    rank: int
    process_id: int
    process_group_id: int
    process_start_monotonic_ns: int
    gpu_uuid: str
    control_url: str
    temp_quota_root: str

    def __post_init__(self) -> None:
        if self.topology not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("fault launch topology is unsupported")
        expected_ranks = {0} if self.topology == "tp1_dp1" else {0, 1}
        if self.rank not in expected_ranks:
            raise ValueError("fault launch rank is unsupported")
        for label, value in (
            ("fault child process", self.process_id),
            ("fault child process group", self.process_group_id),
            ("fault child process start", self.process_start_monotonic_ns),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} is invalid")
        _require_safe("fault assigned GPU UUID", self.gpu_uuid)
        endpoint = urlsplit(self.control_url)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.port is None
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                "fault actuator control endpoint must be explicit localhost"
            )
        quota = Path(self.temp_quota_root)
        if (
            not quota.is_absolute()
            or quota.resolve(strict=False) != quota
            or not quota.is_dir()
            or quota.is_symlink()
        ):
            raise ValueError(
                "fault quota target must be one existing resolved directory"
            )
        metadata = quota.stat(follow_symlinks=False)
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError("fault quota target must be current-user private")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> FailureRankLaunchBinding:
        expected = {
            "topology",
            "rank",
            "process_id",
            "process_group_id",
            "process_start_monotonic_ns",
            "gpu_uuid",
            "control_url",
            "temp_quota_root",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("failure rank launch fields differ")
        return cls(**value)  # type: ignore[arg-type]

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class FailureActuatorLaunchBinding:
    """Source execution scope for one E5 scenario across both exact topologies."""

    schema_version: int
    kind: Literal["e5_failure_actuator_launch_binding"]
    assignment_sha256: str
    inventory_sha256: str
    registry_sha256: str
    plan_sha256: str
    run_nonce_sha256: str
    ranks: tuple[FailureRankLaunchBinding, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "e5_failure_actuator_launch_binding"
        ):
            raise ValueError("failure actuator launch schema is unsupported")
        for label, value in (
            ("failure assignment", self.assignment_sha256),
            ("failure inventory", self.inventory_sha256),
            ("failure registry", self.registry_sha256),
            ("failure plan", self.plan_sha256),
            ("failure run nonce", self.run_nonce_sha256),
        ):
            _require_sha256(label, value)
        if type(self.ranks) is not tuple or tuple(
            (row.topology, row.rank) for row in self.ranks
        ) != (
            ("tp1_dp1", 0),
            ("tp2_dp1", 0),
            ("tp2_dp1", 1),
            ("tp1_dp2", 0),
            ("tp1_dp2", 1),
        ):
            raise ValueError("failure launch lacks exact all-rank topology coverage")
        for row in self.ranks:
            if type(row) is not FailureRankLaunchBinding:
                raise TypeError("failure launch ranks require exact bindings")
            row.__post_init__()
        process_ids = tuple(row.process_id for row in self.ranks)
        quota_roots = tuple(row.temp_quota_root for row in self.ranks)
        if len(set(process_ids)) != 5 or len(set(quota_roots)) != 5:
            raise ValueError("failure launches require isolated process/quota scopes")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "assignment_sha256": self.assignment_sha256,
            "inventory_sha256": self.inventory_sha256,
            "registry_sha256": self.registry_sha256,
            "plan_sha256": self.plan_sha256,
            "run_nonce_sha256": self.run_nonce_sha256,
            "ranks": [row.to_dict() for row in self.ranks],
        }

    @classmethod
    def from_dict(cls, value: object) -> FailureActuatorLaunchBinding:
        expected = {
            "schema_version",
            "kind",
            "assignment_sha256",
            "inventory_sha256",
            "registry_sha256",
            "plan_sha256",
            "run_nonce_sha256",
            "ranks",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("failure actuator launch fields differ")
        raw_ranks = value["ranks"]
        if type(raw_ranks) is not list:
            raise TypeError("failure actuator launch ranks must be an array")
        row = dict(value)
        row["ranks"] = tuple(
            FailureRankLaunchBinding.from_dict(item) for item in raw_ranks
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def rank_binding(self, topology: str, rank: int) -> FailureRankLaunchBinding:
        matches = tuple(
            row for row in self.ranks if row.topology == topology and row.rank == rank
        )
        if len(matches) != 1:
            raise ValueError("failure launch rank binding is not exact")
        return matches[0]


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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> FailurePhaseObservation:
        expected = {
            "phase",
            "operation",
            "monotonic_ns",
            "event_count",
            "observation_sha256",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("fault phase observation fields differ")
        return cls(**value)  # type: ignore[arg-type]


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
        if type(self.recovery_valid) is not bool:
            raise TypeError("fault terminal recovery disposition must be bool")


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


class FailureNativeControlTransport(Protocol):
    """Host-local patched-runtime control channel used by the release actuator."""

    def invoke(
        self,
        rank: FailureRankLaunchBinding,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]: ...


_FAILURE_CONTROL_PATH = "/v1/lightcone-spec/failure-actuator"
_MAX_FAILURE_CONTROL_RESPONSE_BYTES = 1024 * 1024


def linux_process_start_monotonic_ns(process_id: int) -> int:
    """Return Linux ``/proc`` start time on the monotonic boot clock.

    Formal launch bindings use this value, rather than a wall-clock sample, so
    PID reuse is detected at every actuator phase.  Non-Linux hosts fail
    closed; release failure injection is a Linux GPU-host capability.
    """

    if type(process_id) is not int or process_id < 1:
        raise ValueError("failure process ID is invalid")
    stat_path = Path(f"/proc/{process_id}/stat")
    try:
        raw = stat_path.read_text(encoding="ascii")
        tail = raw[raw.rindex(")") + 2 :].split()
        start_ticks = int(tail[19])
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError) as error:
        raise FailureActuatorBlocked("failure_process_identity_unavailable") from error
    if start_ticks < 1 or ticks_per_second < 1:
        raise FailureActuatorBlocked("failure_process_identity_unavailable")
    return start_ticks * 1_000_000_000 // ticks_per_second


def _process_visible_gpu_uuids(process_id: int) -> tuple[str, ...]:
    try:
        raw = Path(f"/proc/{process_id}/environ").read_bytes()
    except OSError as error:
        raise FailureActuatorBlocked(
            "failure_process_environment_unavailable"
        ) from error
    environment: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", maxsplit=1)
        try:
            environment[key.decode("ascii")] = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise FailureActuatorBlocked(
                "failure_process_environment_noncanonical"
            ) from error
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        visible = environment.get("NVIDIA_VISIBLE_DEVICES")
    if visible is None:
        raise FailureActuatorBlocked("failure_process_gpu_scope_unavailable")
    values = tuple(value.strip() for value in visible.split(","))
    if not values or any(not value or not value.startswith("GPU-") for value in values):
        raise FailureActuatorBlocked("failure_process_gpu_scope_noncanonical")
    return values


def _revalidate_failure_rank_process(rank: FailureRankLaunchBinding) -> None:
    rank.__post_init__()
    try:
        os.kill(rank.process_id, 0)
        actual_group = os.getpgid(rank.process_id)
    except OSError as error:
        raise FailureActuatorBlocked("failure_child_process_unavailable") from error
    if actual_group != rank.process_group_id:
        raise FailureActuatorBlocked("failure_child_process_group_changed")
    if linux_process_start_monotonic_ns(rank.process_id) != (
        rank.process_start_monotonic_ns
    ):
        raise FailureActuatorBlocked("failure_child_process_generation_changed")
    if rank.gpu_uuid not in _process_visible_gpu_uuids(rank.process_id):
        raise FailureActuatorBlocked("failure_child_gpu_assignment_changed")
    quota = Path(rank.temp_quota_root)
    metadata = quota.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise FailureActuatorBlocked("failure_private_quota_scope_changed")


class SglangHttpFailureNativeControlTransport:
    """Concrete localhost transport for the patched rank-local SGLang hook.

    The transport performs no actuation itself.  It revalidates the exact
    child generation, process group, GPU UUID, and private quota root before
    and after every request, then invokes only the registered patched endpoint.
    A test callback or alternate URL cannot be injected into formal execution.
    """

    def invoke(
        self,
        rank: FailureRankLaunchBinding,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        if type(rank) is not FailureRankLaunchBinding:
            raise TypeError("failure transport requires an exact rank binding")
        if type(payload) is not dict:
            raise TypeError("failure transport payload must be one exact object")
        expected = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "launch_binding_sha256",
            "assignment_sha256",
            "inventory_sha256",
            "plan_sha256",
            "scenario",
            "scenario_semantics_sha256",
            "phase",
            "operation",
            "parameters",
            "topology",
            "rank",
            "process_id",
            "process_group_id",
            "process_start_monotonic_ns",
            "gpu_uuid",
            "temp_quota_root",
            "run_nonce_sha256",
        }
        if set(payload) != expected:
            raise ValueError("failure transport payload fields differ")
        if (
            payload["schema_version"] != 1
            or payload["kind"] != "lightcone_e5_failure_actuator_command"
            or payload["protocol_sha256"] != FAILURE_ACTUATOR_PROTOCOL_SHA256
            or payload["topology"] != rank.topology
            or payload["rank"] != rank.rank
            or payload["process_id"] != rank.process_id
            or payload["process_group_id"] != rank.process_group_id
            or payload["process_start_monotonic_ns"] != rank.process_start_monotonic_ns
            or payload["gpu_uuid"] != rank.gpu_uuid
            or payload["temp_quota_root"] != rank.temp_quota_root
        ):
            raise ValueError("failure transport payload differs from rank scope")
        endpoint = urlsplit(rank.control_url)
        if endpoint.path != _FAILURE_CONTROL_PATH:
            raise ValueError("failure transport endpoint path is not registered")
        _revalidate_failure_rank_process(rank)
        request = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        connection = http.client.HTTPConnection(
            endpoint.hostname,
            endpoint.port,
            timeout=30.0,
        )
        try:
            connection.request(
                "POST",
                _FAILURE_CONTROL_PATH,
                body=request,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(request)),
                },
            )
            response = connection.getresponse()
            body = response.read(_MAX_FAILURE_CONTROL_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException) as error:
            raise FailureActuatorBlocked("failure_rank_control_unavailable") from error
        finally:
            connection.close()
        _revalidate_failure_rank_process(rank)
        if response.status != 200:
            raise FailureActuatorBlocked(f"failure_rank_control_http_{response.status}")
        if len(body) > _MAX_FAILURE_CONTROL_RESPONSE_BYTES:
            raise FailureActuatorBlocked("failure_rank_control_response_too_large")
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FailureActuatorBlocked(
                "failure_rank_control_response_malformed"
            ) from error
        if type(value) is not dict:
            raise FailureActuatorBlocked("failure_rank_control_response_malformed")
        return value


FAILURE_ACTUATOR_SOURCE_VERSION_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
        "scenario_semantics_sha256": FAILURE_SCENARIO_SEMANTICS_SHA256,
        "transport": (
            "host_local_patched_runtime_control_exact_http_endpoint_with_"
            "pre_and_post_process_generation_gpu_quota_revalidation"
        ),
        "scope": "bound_child_process_gpu_uuid_and_private_temp_quota_root",
        "remote_trust": "unsigned_raw_only",
    }
)

FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e5_failure_actuation_external_control_protocol",
        "actuator_protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
        "source_version_sha256": FAILURE_ACTUATOR_SOURCE_VERSION_SHA256,
        "remote_terminal": "canonical_unsigned_all_rank_recovery",
        "local_trust_lift": "release_control_attestation",
        "replay": "atomic_control_deployment_and_run_nonce",
    }
)


class SourceOwnedSglangFailureActuator:
    """Concrete lifecycle adapter for the patched SGLang failure endpoint."""

    actuator_id = "release.sglang_child_fault_actuator.v1"
    actuator_version_sha256 = FAILURE_ACTUATOR_SOURCE_VERSION_SHA256

    def __init__(
        self,
        launch: FailureActuatorLaunchBinding,
        transport: FailureNativeControlTransport,
    ) -> None:
        if type(launch) is not FailureActuatorLaunchBinding:
            raise TypeError("release fault actuator requires exact launch binding")
        launch.__post_init__()
        if not callable(getattr(transport, "invoke", None)):
            raise TypeError("release fault actuator requires native control transport")
        self._launch = launch
        self._transport = transport
        self._contexts: dict[tuple[str, int], FailureActuatorContext] = {}

    def fresh_context(
        self,
        *,
        plan: ReleaseFailurePlan,
        semantics: FailureScenarioSemantics,
        topology: str,
        rank: int,
        run_nonce_sha256: str,
    ) -> FailureActuatorContext:
        if (
            plan.sha256 != self._launch.plan_sha256
            or plan.registry_sha256 != self._launch.registry_sha256
            or run_nonce_sha256 != self._launch.run_nonce_sha256
            or semantics.scenario != plan.scenario
        ):
            raise ValueError("release fault launch differs from plan/run identity")
        key = (topology, rank)
        if key in self._contexts:
            raise RuntimeError("release fault rank context was reused")
        bound = self._launch.rank_binding(topology, rank)
        context = FailureActuatorContext(
            plan_sha256=plan.sha256,
            scenario_semantics_sha256=semantics.sha256,
            topology=topology,
            rank=rank,
            world_size=1 if topology == "tp1_dp1" else 2,
            process_id=bound.process_id,
            process_start_monotonic_ns=bound.process_start_monotonic_ns,
            session_epoch=0,
            run_nonce_sha256=run_nonce_sha256,
        )
        self._contexts[key] = context
        return context

    def _invoke(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
        *,
        phase: FailurePhase,
        operation: str,
        terminal: bool,
    ) -> FailurePhaseObservation | FailureTerminalObservation:
        key = (context.topology, context.rank)
        if self._contexts.get(key) != context:
            raise ValueError("release fault context is foreign or stale")
        bound = self._launch.rank_binding(*key)
        response = self._transport.invoke(
            bound,
            {
                "schema_version": 1,
                "kind": "lightcone_e5_failure_actuator_command",
                "protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
                "launch_binding_sha256": self._launch.sha256,
                "assignment_sha256": self._launch.assignment_sha256,
                "inventory_sha256": self._launch.inventory_sha256,
                "plan_sha256": context.plan_sha256,
                "scenario": semantics.scenario,
                "scenario_semantics_sha256": semantics.sha256,
                "phase": phase,
                "operation": operation,
                "parameters": [list(value) for value in semantics.parameters],
                "topology": context.topology,
                "rank": context.rank,
                "process_id": bound.process_id,
                "process_group_id": bound.process_group_id,
                "process_start_monotonic_ns": bound.process_start_monotonic_ns,
                "gpu_uuid": bound.gpu_uuid,
                "temp_quota_root": bound.temp_quota_root,
                "run_nonce_sha256": context.run_nonce_sha256,
            },
        )
        expected = {
            "phase",
            "operation",
            "monotonic_ns",
            "event_count",
            "observation_sha256",
        }
        if terminal:
            expected |= {"counters", "recovery_valid"}
        if type(response) is not dict or set(response) != expected:
            raise ValueError("native fault response fields differ from protocol")
        if response["phase"] != phase or response["operation"] != operation:
            raise ValueError("native fault response changed phase/operation")
        common = {
            "phase": phase,
            "operation": operation,
            "monotonic_ns": response["monotonic_ns"],
            "event_count": response["event_count"],
            "observation_sha256": response["observation_sha256"],
        }
        if not terminal:
            return FailurePhaseObservation(**common)
        counters = response["counters"]
        if type(counters) is not dict:
            raise TypeError("native fault terminal counters must be an object")
        return FailureTerminalObservation(
            **common,
            counters=tuple(sorted(counters.items())),
            recovery_valid=response["recovery_valid"],
        )

    def arm(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailurePhaseObservation:
        result = self._invoke(
            context,
            semantics,
            phase="arm",
            operation=semantics.arm_operation,
            terminal=False,
        )
        assert type(result) is FailurePhaseObservation
        return result

    def trigger(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailurePhaseObservation:
        result = self._invoke(
            context,
            semantics,
            phase="trigger",
            operation=semantics.trigger_operation,
            terminal=False,
        )
        assert type(result) is FailurePhaseObservation
        return result

    def prove(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailurePhaseObservation:
        result = self._invoke(
            context,
            semantics,
            phase="proof",
            operation=semantics.proof_event,
            terminal=False,
        )
        assert type(result) is FailurePhaseObservation
        return result

    def recover(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailurePhaseObservation:
        result = self._invoke(
            context,
            semantics,
            phase="recover",
            operation=semantics.recovery_invariant,
            terminal=False,
        )
        assert type(result) is FailurePhaseObservation
        return result

    def terminal(
        self, context: FailureActuatorContext, semantics: FailureScenarioSemantics
    ) -> FailureTerminalObservation:
        result = self._invoke(
            context,
            semantics,
            phase="terminal",
            operation=semantics.terminal_counter,
            terminal=True,
        )
        assert type(result) is FailureTerminalObservation
        return result


def release_failure_actuator_factory(
    launch: FailureActuatorLaunchBinding,
    transport: FailureNativeControlTransport,
) -> SourceOwnedSglangFailureActuator:
    return SourceOwnedSglangFailureActuator(launch, transport)


RELEASE_FAILURE_ACTUATOR_CAPABILITIES = (
    ReleaseFailureActuatorCapability(
        actuator_id=SourceOwnedSglangFailureActuator.actuator_id,
        actuator_version_sha256=FAILURE_ACTUATOR_SOURCE_VERSION_SHA256,
        factory_module=release_failure_actuator_factory.__module__,
        factory_qualname=release_failure_actuator_factory.__qualname__,
        factory=release_failure_actuator_factory,
    ),
)


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
    release_failure_actuator_capability()
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
        if self.topology not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("fault rank receipt topology is unsupported")
        expected_ranks = {0} if self.topology == "tp1_dp1" else {0, 1}
        if self.rank not in expected_ranks:
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
        for name, value in self.counters:
            _require_safe("fault rank receipt counter", name)
            _require_nonnegative("fault rank receipt counter value", value)
        if type(self.recovery_valid) is not bool:
            raise TypeError("fault rank recovery disposition must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "rank": self.rank,
            "process_id": self.process_id,
            "process_start_monotonic_ns": self.process_start_monotonic_ns,
            "session_epoch": self.session_epoch,
            "phases": [value.to_dict() for value in self.phases],
            "counters": {name: value for name, value in self.counters},
            "recovery_valid": self.recovery_valid,
        }

    @classmethod
    def from_dict(cls, value: object) -> FailureActuatorRankReceipt:
        expected = {
            "topology",
            "rank",
            "process_id",
            "process_start_monotonic_ns",
            "session_epoch",
            "phases",
            "counters",
            "recovery_valid",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("fault rank receipt fields differ")
        phases = value["phases"]
        counters = value["counters"]
        if type(phases) is not list or type(counters) is not dict:
            raise TypeError("fault rank receipt phases/counters are malformed")
        if any(type(name) is not str for name in counters):
            raise TypeError("fault rank receipt counter names must be strings")
        row = dict(value)
        row["phases"] = tuple(
            FailurePhaseObservation.from_dict(item) for item in phases
        )
        row["counters"] = tuple(sorted(counters.items()))
        return cls(**row)  # type: ignore[arg-type]


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
            ("tp1_dp1", 0),
            ("tp2_dp1", 0),
            ("tp2_dp1", 1),
            ("tp1_dp2", 0),
            ("tp1_dp2", 1),
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

    @classmethod
    def from_dict(cls, value: object) -> DiagnosticFailureActuationReceipt:
        expected = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "plan_sha256",
            "authority_sha256",
            "scenario",
            "scenario_semantics_sha256",
            "semantics_universe_sha256",
            "run_nonce_sha256",
            "actuator_id",
            "actuator_version_sha256",
            "rank_receipts",
            "recovered",
            "committed",
            "correctness_only",
            "formal_execution_authorized",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("diagnostic fault receipt fields differ")
        rank_receipts = value["rank_receipts"]
        if type(rank_receipts) is not list:
            raise TypeError("diagnostic fault receipt ranks must be an array")
        row = dict(value)
        row["rank_receipts"] = tuple(
            FailureActuatorRankReceipt.from_dict(item) for item in rank_receipts
        )
        return cls(**row)  # type: ignore[arg-type]


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
            context = FailureActuatorContext(
                plan_sha256=plan.sha256,
                scenario_semantics_sha256=semantics.sha256,
                topology=rank.topology,
                rank=rank.rank,
                world_size=target.world_size,
                process_id=rank.process_id,
                process_start_monotonic_ns=rank.process_start_monotonic_ns,
                session_epoch=rank.session_epoch,
                run_nonce_sha256=receipt.run_nonce_sha256,
            )
            terminal_phase = rank.phases[-1]
            _validate_rank_lifecycle(
                plan=plan,
                semantics=semantics,
                context=context,
                phases=rank.phases[:4],
                terminal=FailureTerminalObservation(
                    phase="terminal",
                    operation=terminal_phase.operation,
                    monotonic_ns=terminal_phase.monotonic_ns,
                    event_count=terminal_phase.event_count,
                    observation_sha256=terminal_phase.observation_sha256,
                    counters=rank.counters,
                    recovery_valid=rank.recovery_valid,
                ),
            )
            trigger_total += counters[semantics.terminal_counter]
        if trigger_total < 1:
            raise ValueError(
                "fault recovery receipt lacks the topology trigger counter"
            )


def _validate_release_launch(
    authority: FailureInjectionAuthorityResult,
    launch: FailureActuatorLaunchBinding,
) -> None:
    if type(authority) is not FailureInjectionAuthorityResult:
        raise TypeError("release fault launch requires exact authority")
    if type(launch) is not FailureActuatorLaunchBinding:
        raise TypeError("release fault launch requires exact launch binding")
    launch.__post_init__()
    if (
        launch.plan_sha256 != authority.plan.sha256
        or launch.registry_sha256 != authority.plan.registry_sha256
    ):
        raise ValueError("release fault launch differs from plan authority")


def _parse_unsigned_failure_terminal(
    value: object,
    *,
    authority: FailureInjectionAuthorityResult,
    expected_launch: FailureActuatorLaunchBinding,
) -> tuple[FailureActuatorLaunchBinding, DiagnosticFailureActuationReceipt]:
    expected = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "source_capability_sha256",
        "launch_binding",
        "launch_binding_sha256",
        "recovery_receipt",
        "recovery_receipt_sha256",
        "formal_execution_authorized",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("unsigned fault terminal fields differ")
    if (
        value["schema_version"] != 1
        or value["kind"] != "e5_unsigned_failure_recovery_terminal"
        or value["protocol_sha256"] != FAILURE_ACTUATOR_PROTOCOL_SHA256
        or value["formal_execution_authorized"] is not False
    ):
        raise ValueError("unsigned fault terminal schema/authority is unsupported")
    capability = release_failure_actuator_capability()
    if value["source_capability_sha256"] != capability.sha256:
        raise ValueError("unsigned fault terminal uses another source capability")
    launch = FailureActuatorLaunchBinding.from_dict(value["launch_binding"])
    if launch.sha256 != value["launch_binding_sha256"] or launch != expected_launch:
        raise ValueError("unsigned fault terminal launch binding differs")
    _validate_release_launch(authority, launch)
    receipt = DiagnosticFailureActuationReceipt.from_dict(value["recovery_receipt"])
    if receipt.sha256 != value["recovery_receipt_sha256"]:
        raise ValueError("unsigned fault terminal recovery receipt digest differs")
    if (
        receipt.plan_sha256 != launch.plan_sha256
        or receipt.run_nonce_sha256 != launch.run_nonce_sha256
        or receipt.actuator_id != capability.actuator_id
        or receipt.actuator_version_sha256 != capability.actuator_version_sha256
    ):
        raise ValueError("unsigned fault terminal execution identity differs")
    validate_failure_recovery_receipt(receipt, authority)
    for rank_receipt, rank_launch in zip(
        receipt.rank_receipts, launch.ranks, strict=True
    ):
        if (
            rank_receipt.topology != rank_launch.topology
            or rank_receipt.rank != rank_launch.rank
            or rank_receipt.process_id != rank_launch.process_id
            or rank_receipt.process_start_monotonic_ns
            != rank_launch.process_start_monotonic_ns
        ):
            raise ValueError("unsigned fault terminal rank launch differs")
    return launch, receipt


@dataclass(frozen=True)
class FailureActuationExternalControlBinding:
    """Exact unsigned terminal identity presented to the local signer."""

    schema_version: int
    kind: Literal["e5_failure_actuation_external_control_binding"]
    canonical_raw_sha256: str
    semantic_artifact_sha256: str
    recovery_receipt_sha256: str
    launch_binding_sha256: str
    assignment_sha256: str
    inventory_sha256: str
    registry_sha256: str
    plan_sha256: str
    authority_sha256: str
    cell_id: str
    scenario: str
    run_nonce_sha256: str
    source_capability_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "e5_failure_actuation_external_control_binding"
        ):
            raise ValueError("failure external-control binding schema is unsupported")
        for label, value in (
            ("failure raw terminal", self.canonical_raw_sha256),
            ("failure semantic terminal", self.semantic_artifact_sha256),
            ("failure recovery receipt", self.recovery_receipt_sha256),
            ("failure launch", self.launch_binding_sha256),
            ("failure assignment", self.assignment_sha256),
            ("failure inventory", self.inventory_sha256),
            ("failure registry", self.registry_sha256),
            ("failure plan", self.plan_sha256),
            ("failure authority", self.authority_sha256),
            ("failure cell", self.cell_id),
            ("failure nonce", self.run_nonce_sha256),
            ("failure source capability", self.source_capability_sha256),
        ):
            _require_sha256(label, value)
        if self.scenario not in E5_FAILURES:
            raise ValueError("failure external-control scenario is unregistered")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def lineage_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "e5_failure_actuation_external_control_lineage",
                "binding_sha256": self.sha256,
                "assignment_sha256": self.assignment_sha256,
                "inventory_sha256": self.inventory_sha256,
                "registry_sha256": self.registry_sha256,
                "plan_sha256": self.plan_sha256,
                "authority_sha256": self.authority_sha256,
                "cell_id": self.cell_id,
                "scenario": self.scenario,
                "run_nonce_sha256": self.run_nonce_sha256,
                "launch_binding_sha256": self.launch_binding_sha256,
                "recovery_receipt_sha256": self.recovery_receipt_sha256,
            }
        )


_VERIFIED_FAILURE_ACTUATION_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedFailureActuationResult:
    """Verifier-constructed E5 completion; raw receipts cannot instantiate it."""

    raw_terminal_raw_sha256: str
    raw_terminal_semantic_sha256: str
    control_binding_sha256: str
    control_envelope_sha256: str
    replay_reservation_sha256: str
    source_capability_sha256: str
    assignment_sha256: str
    inventory_sha256: str
    registry_sha256: str
    plan_sha256: str
    authority_sha256: str
    cell_id: str
    scenario: str
    run_nonce_sha256: str
    launch_binding_sha256: str
    recovery_receipt: DiagnosticFailureActuationReceipt
    correctness_only: bool

    def __init__(
        self,
        *,
        raw_terminal_raw_sha256: str,
        raw_terminal_semantic_sha256: str,
        control_binding_sha256: str,
        control_envelope_sha256: str,
        replay_reservation_sha256: str,
        source_capability_sha256: str,
        assignment_sha256: str,
        inventory_sha256: str,
        registry_sha256: str,
        plan_sha256: str,
        authority_sha256: str,
        cell_id: str,
        scenario: str,
        run_nonce_sha256: str,
        launch_binding_sha256: str,
        recovery_receipt: DiagnosticFailureActuationReceipt,
        correctness_only: bool,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_FAILURE_ACTUATION_SENTINEL:
            raise TypeError("verified failure result requires external-control proof")
        for name, value in locals().copy().items():
            if name not in {"self", "_verification_tag"}:
                object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        for label, value in (
            ("failure raw", self.raw_terminal_raw_sha256),
            ("failure semantic raw", self.raw_terminal_semantic_sha256),
            ("failure control binding", self.control_binding_sha256),
            ("failure control envelope", self.control_envelope_sha256),
            ("failure replay reservation", self.replay_reservation_sha256),
            ("failure source capability", self.source_capability_sha256),
            ("failure assignment", self.assignment_sha256),
            ("failure inventory", self.inventory_sha256),
            ("failure registry", self.registry_sha256),
            ("failure plan", self.plan_sha256),
            ("failure authority", self.authority_sha256),
            ("failure cell", self.cell_id),
            ("failure nonce", self.run_nonce_sha256),
            ("failure launch", self.launch_binding_sha256),
        ):
            _require_sha256(label, value)
        if self.scenario not in E5_FAILURES:
            raise ValueError("verified failure scenario is unregistered")
        if type(self.recovery_receipt) is not DiagnosticFailureActuationReceipt:
            raise TypeError("verified failure result requires exact recovery receipt")
        self.recovery_receipt.__post_init__()
        if self.correctness_only is not True:
            raise ValueError("E5 failure result must remain correctness-only")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "verified_e5_failure_actuation_result",
            "raw_terminal_raw_sha256": self.raw_terminal_raw_sha256,
            "raw_terminal_semantic_sha256": self.raw_terminal_semantic_sha256,
            "control_binding_sha256": self.control_binding_sha256,
            "control_envelope_sha256": self.control_envelope_sha256,
            "replay_reservation_sha256": self.replay_reservation_sha256,
            "source_capability_sha256": self.source_capability_sha256,
            "assignment_sha256": self.assignment_sha256,
            "inventory_sha256": self.inventory_sha256,
            "registry_sha256": self.registry_sha256,
            "plan_sha256": self.plan_sha256,
            "authority_sha256": self.authority_sha256,
            "cell_id": self.cell_id,
            "scenario": self.scenario,
            "run_nonce_sha256": self.run_nonce_sha256,
            "launch_binding_sha256": self.launch_binding_sha256,
            "recovery_receipt": self.recovery_receipt.to_dict(),
            "correctness_only": self.correctness_only,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class FailureActuationProofArtifact:
    """Durable local trust lift for one unsigned all-rank E5 terminal."""

    schema_version: int
    kind: Literal["e5_failure_actuation_proof_artifact"]
    raw_terminal: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    expected_root_manifest_sha256: str
    result: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "e5_failure_actuation_proof_artifact"
        ):
            raise ValueError("failure proof artifact schema is unsupported")
        if type(self.raw_terminal) is not CanonicalJsonProofBinding:
            raise TypeError("failure proof artifact requires raw terminal binding")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("failure proof artifact requires exact control envelope")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("failure proof artifact requires replay reservation")
        self.raw_terminal.__post_init__()
        self.control_attestation.__post_init__()
        self.replay_reservation.__post_init__()
        _require_sha256(
            "failure proof release root", self.expected_root_manifest_sha256
        )
        if type(self.result) is not dict:
            raise TypeError("failure proof artifact result must be an object")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_terminal": self.raw_terminal.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
            "result": self.result,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> FailureActuationProofArtifact:
        expected = {
            "schema_version",
            "kind",
            "raw_terminal",
            "control_attestation",
            "replay_reservation",
            "expected_root_manifest_sha256",
            "result",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("failure proof artifact fields differ")
        row = dict(value)
        raw_terminal = CanonicalJsonProofBinding.from_dict(row.pop("raw_terminal"))
        control = ControlArtifactAttestation.from_dict(row.pop("control_attestation"))
        reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("replay_reservation")
        )
        return cls(
            **row,  # type: ignore[arg-type]
            raw_terminal=raw_terminal,
            control_attestation=control,
            replay_reservation=reservation,
        )


def _failure_control_binding(
    raw_terminal: CanonicalJsonProofBinding,
    *,
    authority: FailureInjectionAuthorityResult,
    launch: FailureActuatorLaunchBinding,
    receipt: DiagnosticFailureActuationReceipt,
) -> FailureActuationExternalControlBinding:
    return FailureActuationExternalControlBinding(
        schema_version=1,
        kind="e5_failure_actuation_external_control_binding",
        canonical_raw_sha256=raw_terminal.raw_sha256,
        semantic_artifact_sha256=raw_terminal.semantic_sha256,
        recovery_receipt_sha256=receipt.sha256,
        launch_binding_sha256=launch.sha256,
        assignment_sha256=launch.assignment_sha256,
        inventory_sha256=launch.inventory_sha256,
        registry_sha256=launch.registry_sha256,
        plan_sha256=launch.plan_sha256,
        authority_sha256=authority.binding.sha256,
        cell_id=authority.plan.cell_id,
        scenario=authority.plan.scenario,
        run_nonce_sha256=launch.run_nonce_sha256,
        source_capability_sha256=release_failure_actuator_capability().sha256,
    )


def _verified_failure_result(
    *,
    raw_terminal: CanonicalJsonProofBinding,
    binding: FailureActuationExternalControlBinding,
    control_attestation: ControlArtifactAttestation,
    replay_reservation_sha256: str,
    receipt: DiagnosticFailureActuationReceipt,
) -> VerifiedFailureActuationResult:
    return VerifiedFailureActuationResult(
        raw_terminal_raw_sha256=raw_terminal.raw_sha256,
        raw_terminal_semantic_sha256=raw_terminal.semantic_sha256,
        control_binding_sha256=binding.sha256,
        control_envelope_sha256=control_attestation.sha256,
        replay_reservation_sha256=replay_reservation_sha256,
        source_capability_sha256=binding.source_capability_sha256,
        assignment_sha256=binding.assignment_sha256,
        inventory_sha256=binding.inventory_sha256,
        registry_sha256=binding.registry_sha256,
        plan_sha256=binding.plan_sha256,
        authority_sha256=binding.authority_sha256,
        cell_id=binding.cell_id,
        scenario=binding.scenario,
        run_nonce_sha256=binding.run_nonce_sha256,
        launch_binding_sha256=binding.launch_binding_sha256,
        recovery_receipt=receipt,
        correctness_only=True,
        _verification_tag=_VERIFIED_FAILURE_ACTUATION_SENTINEL,
    )


def execute_release_failure_actuator_unsigned(
    token: FailureExecutionAuthorityToken,
    *,
    cell: ExperimentCell,
    expected_registry_sha256: str,
    launch: FailureActuatorLaunchBinding,
    transport: FailureNativeControlTransport,
    raw_terminal_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Execute the exact source actuator and publish no authority on the host."""

    output = _validate_new_receipt_path(
        raw_terminal_path, label="unsigned fault terminal"
    )
    authority = require_failure_execution_lifecycle(
        token,
        cell=cell,
        expected_registry_sha256=expected_registry_sha256,
    )
    _validate_release_launch(authority, launch)
    capability = release_failure_actuator_capability()
    if capability.factory is not release_failure_actuator_factory:
        raise FailureActuatorBlocked(FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON)
    actuator = capability.factory(launch, transport)
    if type(actuator) is not SourceOwnedSglangFailureActuator:
        raise FailureActuatorBlocked(FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON)
    receipt = _execute_failure_actuator_lifecycle(
        authority,
        actuator,
        run_nonce_sha256=launch.run_nonce_sha256,
    )
    value = {
        "schema_version": 1,
        "kind": "e5_unsigned_failure_recovery_terminal",
        "protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
        "source_capability_sha256": capability.sha256,
        "launch_binding": launch.to_dict(),
        "launch_binding_sha256": launch.sha256,
        "recovery_receipt": receipt.to_dict(),
        "recovery_receipt_sha256": receipt.sha256,
        "formal_execution_authorized": False,
    }
    publish_canonical_json_no_replace(output, value)
    raw = CanonicalJsonProofBinding.bind(output)
    _parse_unsigned_failure_terminal(
        raw.reopen(), authority=authority, expected_launch=launch
    )
    return raw


def build_failure_actuation_external_control_binding(
    raw_terminal_path: str,
    *,
    token: FailureExecutionAuthorityToken,
    cell: ExperimentCell,
    expected_registry_sha256: str,
    expected_assignment_sha256: str,
    expected_inventory_sha256: str,
) -> FailureActuationExternalControlBinding:
    """Deep-reopen an unsigned terminal and derive its local signing subject."""

    _require_sha256("failure expected assignment", expected_assignment_sha256)
    _require_sha256("failure expected inventory", expected_inventory_sha256)
    authority = require_failure_execution_lifecycle(
        token,
        cell=cell,
        expected_registry_sha256=expected_registry_sha256,
    )
    raw = CanonicalJsonProofBinding.bind(raw_terminal_path)
    value = raw.reopen()
    if type(value) is not dict or type(value.get("launch_binding")) is not dict:
        raise ValueError("unsigned fault terminal lacks launch binding")
    launch = FailureActuatorLaunchBinding.from_dict(value["launch_binding"])
    if (
        launch.assignment_sha256 != expected_assignment_sha256
        or launch.inventory_sha256 != expected_inventory_sha256
    ):
        raise ValueError("unsigned fault terminal assignment/inventory differs")
    _, receipt = _parse_unsigned_failure_terminal(
        value, authority=authority, expected_launch=launch
    )
    return _failure_control_binding(
        raw,
        authority=authority,
        launch=launch,
        receipt=receipt,
    )


def _revalidate_failure_actuation_proof(
    artifact: FailureActuationProofArtifact,
    *,
    token: FailureExecutionAuthorityToken,
    cell: ExperimentCell,
    expected_registry_sha256: str,
    expected_assignment_sha256: str,
    expected_inventory_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> VerifiedFailureActuationResult:
    artifact.__post_init__()
    if artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256:
        raise ValueError("failure proof uses another release root")
    authority = require_failure_execution_lifecycle(
        token,
        cell=cell,
        expected_registry_sha256=expected_registry_sha256,
    )
    value = artifact.raw_terminal.reopen()
    if type(value) is not dict or type(value.get("launch_binding")) is not dict:
        raise ValueError("failure proof raw terminal lacks launch binding")
    launch = FailureActuatorLaunchBinding.from_dict(value["launch_binding"])
    if (
        launch.assignment_sha256 != expected_assignment_sha256
        or launch.inventory_sha256 != expected_inventory_sha256
        or launch.registry_sha256 != expected_registry_sha256
    ):
        raise ValueError("failure proof formal execution identity differs")
    _, receipt = _parse_unsigned_failure_terminal(
        value, authority=authority, expected_launch=launch
    )
    binding = _failure_control_binding(
        artifact.raw_terminal,
        authority=authority,
        launch=launch,
        receipt=receipt,
    )
    subject = artifact.control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != binding.sha256
        or subject.protocol_sha256 != FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != binding.registry_sha256
        or subject.lineage_sha256 != binding.lineage_sha256
    ):
        raise ValueError("failure proof external-control subject differs")
    if (
        artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("failure proof control uses another release root")
    reserved = artifact.replay_reservation.revalidate()
    if type(now_ns) is not int or now_ns < artifact.replay_reservation.reserved_ns:
        raise ValueError("failure proof current time precedes reservation")
    verified = verify_release_control_artifact_attestation(
        artifact.control_attestation,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=artifact.replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    expected_challenges = tuple(
        sorted(
            {
                launch.run_nonce_sha256,
                verified.challenge_sha256,
                verified.deployment_policy_challenge_sha256,
            }
        )
    )
    expected_reservation_sha256 = control_challenge_reservation_sha256(
        (verified,),
        reserved_ns=artifact.replay_reservation.reserved_ns,
        additional_challenge_sha256s=(launch.run_nonce_sha256,),
    )
    if (
        reserved != expected_challenges
        or artifact.replay_reservation.reservation_sha256 != expected_reservation_sha256
    ):
        raise ValueError("failure proof replay reservation differs")
    result = _verified_failure_result(
        raw_terminal=artifact.raw_terminal,
        binding=binding,
        control_attestation=artifact.control_attestation,
        replay_reservation_sha256=expected_reservation_sha256,
        receipt=receipt,
    )
    if result.to_dict() != artifact.result:
        raise ValueError("failure proof derived result changed")
    return result


def publish_failure_actuation_proof_artifact(
    raw_terminal_path: str,
    *,
    token: FailureExecutionAuthorityToken,
    cell: ExperimentCell,
    expected_registry_sha256: str,
    expected_assignment_sha256: str,
    expected_inventory_sha256: str,
    expected_root_manifest_sha256: str,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    """Locally trust-lift one raw E5 terminal and publish durable proof."""

    _validate_new_receipt_path(proof_artifact_path, label="failure proof artifact")
    binding = build_failure_actuation_external_control_binding(
        raw_terminal_path,
        token=token,
        cell=cell,
        expected_registry_sha256=expected_registry_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        expected_inventory_sha256=expected_inventory_sha256,
    )
    subject = control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != binding.sha256
        or subject.protocol_sha256 != FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_registry_sha256
        or subject.lineage_sha256 != binding.lineage_sha256
    ):
        raise ValueError("failure external-control subject differs")
    if (
        control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("failure external control uses another release root")
    verified_controls = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=(binding.run_nonce_sha256,),
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified_controls,
        reserved_ns=now_ns,
        additional_challenge_sha256s=(binding.run_nonce_sha256,),
    )
    reservation = replay_store.bind_reservation(reservation_sha256)
    raw = CanonicalJsonProofBinding.bind(raw_terminal_path)
    authority = require_failure_execution_lifecycle(
        token,
        cell=cell,
        expected_registry_sha256=expected_registry_sha256,
    )
    value = raw.reopen()
    launch = FailureActuatorLaunchBinding.from_dict(value["launch_binding"])
    _, receipt = _parse_unsigned_failure_terminal(
        value, authority=authority, expected_launch=launch
    )
    result = _verified_failure_result(
        raw_terminal=raw,
        binding=binding,
        control_attestation=control_attestation,
        replay_reservation_sha256=reservation_sha256,
        receipt=receipt,
    )
    artifact = FailureActuationProofArtifact(
        schema_version=1,
        kind="e5_failure_actuation_proof_artifact",
        raw_terminal=raw,
        control_attestation=control_attestation,
        replay_reservation=reservation,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        result=result.to_dict(),
    )
    try:
        publish_canonical_json_no_replace(proof_artifact_path, artifact.to_dict())
    except Exception as error:
        raise RuntimeError(
            "failure proof publication failed after replay reservation; issue a new "
            "control challenge and retain the orphaned reservation"
        ) from error
    return CanonicalJsonProofBinding.bind(
        proof_artifact_path, semantic_sha256=artifact.sha256
    )


def validate_failure_actuation_proof_artifact(
    proof_artifact_path: str,
    *,
    token: FailureExecutionAuthorityToken,
    cell: ExperimentCell,
    expected_registry_sha256: str,
    expected_assignment_sha256: str,
    expected_inventory_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> VerifiedFailureActuationResult:
    """Deep-reopen formal E5 completion without consuming replay twice."""

    for label, value in (
        ("failure expected registry", expected_registry_sha256),
        ("failure expected assignment", expected_assignment_sha256),
        ("failure expected inventory", expected_inventory_sha256),
        ("failure expected release root", expected_root_manifest_sha256),
    ):
        _require_sha256(label, value)
    file_binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FailureActuationProofArtifact.from_dict(file_binding.reopen())
    if file_binding.semantic_sha256 != artifact.sha256:
        raise ValueError("failure proof artifact file identity differs")
    return _revalidate_failure_actuation_proof(
        artifact,
        token=token,
        cell=cell,
        expected_registry_sha256=expected_registry_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
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


def _execute_failure_actuator_lifecycle(
    authority: FailureInjectionAuthorityResult,
    actuator: FirstPartyFailureActuator,
    *,
    run_nonce_sha256: str,
) -> DiagnosticFailureActuationReceipt:
    """Run the exact all-rank lifecycle without granting evidence authority."""

    if type(authority) is not FailureInjectionAuthorityResult:
        raise TypeError("CPU fault lifecycle requires exact authority")
    plan = authority.plan
    plan.__post_init__()
    semantics = failure_semantics(plan.scenario)
    _require_sha256("fault CPU nonce", run_nonce_sha256)
    _require_safe("fault CPU actuator", actuator.actuator_id)
    _require_sha256("fault CPU actuator version", actuator.actuator_version_sha256)
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
            if not terminal.recovery_valid:
                raise ValueError("fault terminal observation did not prove recovery")
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
    return receipt


def _validate_new_receipt_path(receipt_path: str | Path, *, label: str) -> Path:
    output = Path(receipt_path)
    if (
        not output.is_absolute()
        or output != output.resolve(strict=False)
        or output == Path(output.anchor)
        or output.exists()
        or output.parent.is_symlink()
        or not output.parent.is_dir()
    ):
        raise ValueError(f"{label} path must be a new normalized file")
    return output


def execute_failure_actuator_for_cpu_test(
    authority: FailureInjectionAuthorityResult,
    actuator: FirstPartyFailureActuator,
    *,
    run_nonce_sha256: str,
    receipt_path: str | Path,
) -> DiagnosticFailureActuationReceipt:
    """Run the exact diagnostic lifecycle; publish only after all-rank recovery."""

    output = _validate_new_receipt_path(receipt_path, label="fault CPU receipt")
    receipt = _execute_failure_actuator_lifecycle(
        authority,
        actuator,
        run_nonce_sha256=run_nonce_sha256,
    )
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


def validate_formal_failure_actuation_proof_artifact(*args, **kwargs):
    """Deep-reopen a formally controlled E5 proof without an import cycle."""

    from lightcone_spec.experiments.formal_failure_actuator import (
        validate_formal_failure_actuation_proof_artifact as _validate,
    )

    return _validate(*args, **kwargs)


__all__ = [
    "FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256",
    "FAILURE_ACTUATOR_PROTOCOL_SHA256",
    "FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON",
    "FAILURE_SCENARIO_SEMANTICS",
    "FAILURE_SCENARIO_SEMANTICS_SHA256",
    "RELEASE_FAILURE_ACTUATOR_CAPABILITIES",
    "DiagnosticFailureActuationReceipt",
    "FailureActuationExternalControlBinding",
    "FailureActuationProofArtifact",
    "FailureActuatorBlocked",
    "FailureActuatorContext",
    "FailureActuatorLaunchBinding",
    "FailureActuatorRankReceipt",
    "FailureNativeControlTransport",
    "FailurePhaseObservation",
    "FailureRankLaunchBinding",
    "FailureScenarioSemantics",
    "FailureTerminalObservation",
    "FirstPartyFailureActuator",
    "ReleaseFailureActuatorCapability",
    "SglangHttpFailureNativeControlTransport",
    "SourceOwnedSglangFailureActuator",
    "VerifiedFailureActuationResult",
    "build_failure_actuation_external_control_binding",
    "execute_failure_actuator_for_cpu_test",
    "execute_release_failure_actuator_unsigned",
    "failure_phase_observation_sha256",
    "failure_semantics",
    "linux_process_start_monotonic_ns",
    "publish_failure_actuation_proof_artifact",
    "release_failure_actuator_capability",
    "release_failure_actuator_factory",
    "require_release_failure_actuator",
    "validate_failure_actuation_proof_artifact",
    "validate_failure_recovery_receipt",
    "validate_formal_failure_actuation_proof_artifact",
]
