"""Single-topology source-owned E5 execution and durable trust lift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

from lightcone_spec.experiments.failure_actuator import (
    FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256,
    FAILURE_ACTUATOR_PROTOCOL_SHA256,
    FailureActuatorContext,
    FailureActuatorRankReceipt,
    FailureNativeControlTransport,
    FailurePhase,
    FailurePhaseObservation,
    FailureRankLaunchBinding,
    FailureScenarioSemantics,
    FailureTerminalObservation,
    SglangHttpFailureNativeControlTransport,
    failure_phase_observation_sha256,
    failure_semantics,
    release_failure_actuator_capability,
)
from lightcone_spec.experiments.formal_failure_execution import (
    FormalSingleOperatorE5FailureExecutionDescriptor,
    VerifiedFormalFailureExecutionBinding,
    require_verified_formal_failure_execution_binding,
)
from lightcone_spec.experiments.registry import E5_FAILURES, content_sha256
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


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _new_output(path: str | Path) -> Path:
    value = Path(path)
    if (
        not value.is_absolute()
        or value != value.resolve(strict=False)
        or value.exists()
        or not value.parent.is_dir()
        or value.parent.is_symlink()
    ):
        raise ValueError("formal failure output must be a new normalized file")
    return value


@dataclass(frozen=True)
class FormalFailureActuatorLaunchBinding:
    schema_version: Literal[1]
    kind: Literal["formal_e5_failure_actuator_launch_binding"]
    formal_failure_execution_binding_sha256: str
    assignment_sha256: str
    inventory_sha256: str
    registry_sha256: str
    serving_execution_plan_sha256: str
    run_nonce_sha256: str
    topology: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    ranks: tuple[FailureRankLaunchBinding, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_e5_failure_actuator_launch_binding"
        ):
            raise ValueError("formal failure launch schema differs")
        for label, value in (
            ("execution binding", self.formal_failure_execution_binding_sha256),
            ("assignment", self.assignment_sha256),
            ("inventory", self.inventory_sha256),
            ("registry", self.registry_sha256),
            ("serving plan", self.serving_execution_plan_sha256),
            ("run nonce", self.run_nonce_sha256),
        ):
            _require_sha256(f"formal failure launch {label}", value)
        if self.topology not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("formal failure launch topology differs")
        expected_ranks = (0,) if self.topology == "tp1_dp1" else (0, 1)
        if (
            type(self.ranks) is not tuple
            or tuple(row.rank for row in self.ranks) != expected_ranks
            or any(
                type(row) is not FailureRankLaunchBinding
                or row.topology != self.topology
                for row in self.ranks
            )
        ):
            raise ValueError("formal failure launch rank coverage differs")
        for row in self.ranks:
            row.__post_init__()
        if len({row.process_id for row in self.ranks}) != len(self.ranks) or len(
            {row.temp_quota_root for row in self.ranks}
        ) != len(self.ranks):
            raise ValueError("formal failure process/quota scopes overlap")

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "ranks": [row.to_dict() for row in self.ranks]}

    @classmethod
    def from_dict(cls, value: object) -> FormalFailureActuatorLaunchBinding:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("formal failure launch fields differ")
        row = dict(value)
        ranks = row.pop("ranks")
        if type(ranks) is not list:
            raise TypeError("formal failure launch ranks must be an array")
        return cls(
            **row,
            ranks=tuple(FailureRankLaunchBinding.from_dict(item) for item in ranks),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class FormalFailureActuationReceipt:
    schema_version: Literal[1]
    kind: Literal["formal_e5_failure_actuation_receipt"]
    protocol_sha256: str
    formal_failure_execution_binding_sha256: str
    assignment_sha256: str
    inventory_sha256: str
    registry_sha256: str
    serving_execution_plan_sha256: str
    materialized_cell_id: str
    scenario: str
    scenario_semantics_sha256: str
    run_nonce_sha256: str
    topology: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    launch_binding_sha256: str
    rank_receipts: tuple[FailureActuatorRankReceipt, ...]
    recovered: bool
    correctness_only: Literal[True]
    formal_execution_authorized: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_e5_failure_actuation_receipt"
            or self.protocol_sha256 != FAILURE_ACTUATOR_PROTOCOL_SHA256
            or self.scenario not in E5_FAILURES
            or self.topology not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}
            or self.correctness_only is not True
            or self.formal_execution_authorized is not False
        ):
            raise ValueError("formal failure receipt schema/status differs")
        if type(self.recovered) is not bool:
            raise TypeError("formal failure recovery disposition must be bool")
        for label, value in (
            ("execution binding", self.formal_failure_execution_binding_sha256),
            ("assignment", self.assignment_sha256),
            ("inventory", self.inventory_sha256),
            ("registry", self.registry_sha256),
            ("serving plan", self.serving_execution_plan_sha256),
            ("cell", self.materialized_cell_id),
            ("semantics", self.scenario_semantics_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("launch", self.launch_binding_sha256),
        ):
            _require_sha256(f"formal failure receipt {label}", value)
        semantics = failure_semantics(self.scenario)
        if semantics.sha256 != self.scenario_semantics_sha256:
            raise ValueError("formal failure receipt semantics differ")
        expected_ranks = (0,) if self.topology == "tp1_dp1" else (0, 1)
        if tuple(row.rank for row in self.rank_receipts) != expected_ranks or any(
            row.topology != self.topology for row in self.rank_receipts
        ):
            raise ValueError("formal failure receipt rank coverage differs")
        for row in self.rank_receipts:
            row.__post_init__()
            if dict(row.counters).get(semantics.terminal_counter, 0) < 1:
                raise ValueError("formal failure receipt lacks scenario counter")
        if self.recovered != all(row.recovery_valid for row in self.rank_receipts):
            raise ValueError("formal failure recovery summary differs from ranks")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "rank_receipts": [row.to_dict() for row in self.rank_receipts],
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalFailureActuationReceipt:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("formal failure receipt fields differ")
        row = dict(value)
        ranks = row.pop("rank_receipts")
        if type(ranks) is not list:
            raise TypeError("formal failure receipt ranks must be an array")
        return cls(
            **row,
            rank_receipts=tuple(
                FailureActuatorRankReceipt.from_dict(item) for item in ranks
            ),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


FormalFailureActuatorExecutionBinding = (
    VerifiedFormalFailureExecutionBinding
    | FormalSingleOperatorE5FailureExecutionDescriptor
)


def _actuator_execution(
    value: FormalFailureActuatorExecutionBinding,
) -> FormalFailureActuatorExecutionBinding:
    if type(value) is FormalSingleOperatorE5FailureExecutionDescriptor:
        value.__post_init__()
        return value
    return require_verified_formal_failure_execution_binding(value)


def _actuator_binding_sha256(
    binding: FormalFailureActuatorExecutionBinding,
) -> str:
    if type(binding) is FormalSingleOperatorE5FailureExecutionDescriptor:
        return binding.expected_failure_execution_binding_sha256
    return binding.sha256


def _actuator_gpu_uuids(
    binding: FormalFailureActuatorExecutionBinding,
) -> tuple[str, ...]:
    if type(binding) is FormalSingleOperatorE5FailureExecutionDescriptor:
        return binding.gpu_uuids
    return binding.serving_execution.subject.gpu_uuids


def _validate_launch(
    binding: FormalFailureActuatorExecutionBinding,
    launch: FormalFailureActuatorLaunchBinding,
) -> None:
    subject = binding.subject
    launch.__post_init__()
    if (
        launch.formal_failure_execution_binding_sha256
        != _actuator_binding_sha256(binding)
        or launch.assignment_sha256 != subject.assignment_sha256
        or launch.inventory_sha256 != subject.inventory_sha256
        or launch.registry_sha256 != subject.registry_sha256
        or launch.serving_execution_plan_sha256 != subject.serving_execution_plan_sha256
        or launch.run_nonce_sha256 != subject.run_nonce_sha256
        or launch.topology != subject.topology
        or tuple(row.gpu_uuid for row in launch.ranks) != _actuator_gpu_uuids(binding)
    ):
        raise ValueError("formal failure launch differs from sealed execution")


def _invoke_phase(
    *,
    binding: FormalFailureActuatorExecutionBinding,
    launch: FormalFailureActuatorLaunchBinding,
    rank: FailureRankLaunchBinding,
    transport: FailureNativeControlTransport,
    context: FailureActuatorContext,
    semantics: FailureScenarioSemantics,
    phase: FailurePhase,
    operation: str,
    terminal: bool,
) -> FailurePhaseObservation | FailureTerminalObservation:
    response = transport.invoke(
        rank,
        {
            "schema_version": 1,
            "kind": "lightcone_e5_failure_actuator_command",
            "protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
            "launch_binding_sha256": launch.sha256,
            "assignment_sha256": binding.subject.assignment_sha256,
            "inventory_sha256": binding.subject.inventory_sha256,
            "plan_sha256": binding.subject.serving_execution_plan_sha256,
            "scenario": semantics.scenario,
            "scenario_semantics_sha256": semantics.sha256,
            "phase": phase,
            "operation": operation,
            "parameters": [list(value) for value in semantics.parameters],
            "topology": rank.topology,
            "rank": rank.rank,
            "process_id": rank.process_id,
            "process_group_id": rank.process_group_id,
            "process_start_monotonic_ns": rank.process_start_monotonic_ns,
            "gpu_uuid": rank.gpu_uuid,
            "temp_quota_root": rank.temp_quota_root,
            "run_nonce_sha256": binding.subject.run_nonce_sha256,
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
        raise ValueError("formal failure phase response fields differ")
    expected_digest = failure_phase_observation_sha256(
        context,
        semantics,
        phase=phase,
        operation=operation,
        event_count=response["event_count"],
    )
    if (
        response["phase"] != phase
        or response["operation"] != operation
        or response["observation_sha256"] != expected_digest
    ):
        raise ValueError("formal failure phase response identity differs")
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
    if type(counters) is not dict or any(type(key) is not str for key in counters):
        raise TypeError("formal failure terminal counters are malformed")
    return FailureTerminalObservation(
        **common,
        counters=tuple(sorted(counters.items())),
        recovery_valid=response["recovery_valid"],
    )


def execute_formal_failure_actuator_unsigned(
    binding: FormalFailureActuatorExecutionBinding,
    *,
    launch: FormalFailureActuatorLaunchBinding,
    transport: FailureNativeControlTransport,
    raw_terminal_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Execute one sealed E5 topology and publish unsigned recovery bytes."""

    verified = _actuator_execution(binding)
    _validate_launch(verified, launch)
    if type(transport) is not SglangHttpFailureNativeControlTransport:
        raise TypeError("formal failure execution requires source HTTP transport")
    output = _new_output(raw_terminal_path)
    subject = verified.subject
    semantics = failure_semantics(subject.scenario)
    operation_rows: tuple[tuple[FailurePhase, str, bool], ...] = (
        ("arm", semantics.arm_operation, False),
        ("trigger", semantics.trigger_operation, False),
        ("proof", semantics.proof_event, False),
        ("recover", semantics.recovery_invariant, False),
        ("terminal", semantics.terminal_counter, True),
    )
    receipts = []
    for rank in launch.ranks:
        context = FailureActuatorContext(
            plan_sha256=subject.serving_execution_plan_sha256,
            scenario_semantics_sha256=semantics.sha256,
            topology=subject.topology,
            rank=rank.rank,
            world_size=len(launch.ranks),
            process_id=rank.process_id,
            process_start_monotonic_ns=rank.process_start_monotonic_ns,
            session_epoch=0,
            run_nonce_sha256=subject.run_nonce_sha256,
        )
        observations = tuple(
            _invoke_phase(
                binding=verified,
                launch=launch,
                rank=rank,
                transport=transport,
                context=context,
                semantics=semantics,
                phase=phase,
                operation=operation,
                terminal=terminal,
            )
            for phase, operation, terminal in operation_rows
        )
        terminal_observation = observations[-1]
        assert type(terminal_observation) is FailureTerminalObservation
        receipts.append(
            FailureActuatorRankReceipt(
                topology=subject.topology,
                rank=rank.rank,
                process_id=rank.process_id,
                process_start_monotonic_ns=rank.process_start_monotonic_ns,
                session_epoch=0,
                phases=tuple(
                    FailurePhaseObservation(
                        phase=row.phase,
                        operation=row.operation,
                        monotonic_ns=row.monotonic_ns,
                        event_count=row.event_count,
                        observation_sha256=row.observation_sha256,
                    )
                    for row in observations
                ),
                counters=terminal_observation.counters,
                recovery_valid=terminal_observation.recovery_valid,
            )
        )
    receipt = FormalFailureActuationReceipt(
        schema_version=1,
        kind="formal_e5_failure_actuation_receipt",
        protocol_sha256=FAILURE_ACTUATOR_PROTOCOL_SHA256,
        formal_failure_execution_binding_sha256=(_actuator_binding_sha256(verified)),
        assignment_sha256=subject.assignment_sha256,
        inventory_sha256=subject.inventory_sha256,
        registry_sha256=subject.registry_sha256,
        serving_execution_plan_sha256=subject.serving_execution_plan_sha256,
        materialized_cell_id=subject.materialized_cell_id,
        scenario=subject.scenario,
        scenario_semantics_sha256=semantics.sha256,
        run_nonce_sha256=subject.run_nonce_sha256,
        topology=subject.topology,
        launch_binding_sha256=launch.sha256,
        rank_receipts=tuple(receipts),
        recovered=all(row.recovery_valid for row in receipts),
        correctness_only=True,
        formal_execution_authorized=False,
    )
    publish_canonical_json_no_replace(
        output,
        {
            "schema_version": 1,
            "kind": "e5_unsigned_formal_failure_recovery_terminal",
            "protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
            "source_capability_sha256": release_failure_actuator_capability().sha256,
            "formal_failure_execution_binding_sha256": (
                _actuator_binding_sha256(verified)
            ),
            "launch_binding": launch.to_dict(),
            "launch_binding_sha256": launch.sha256,
            "recovery_receipt": receipt.to_dict(),
            "recovery_receipt_sha256": receipt.sha256,
            "formal_execution_authorized": False,
        },
    )
    return CanonicalJsonProofBinding.bind(output)


@dataclass(frozen=True)
class FormalFailureActuationProofArtifact:
    schema_version: Literal[1]
    kind: Literal["formal_e5_failure_actuation_proof_artifact"]
    raw_terminal: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    expected_root_manifest_sha256: str
    result: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_e5_failure_actuation_proof_artifact"
        ):
            raise ValueError("formal failure proof schema differs")
        self.raw_terminal.__post_init__()
        self.control_attestation.__post_init__()
        self.replay_reservation.__post_init__()
        _require_sha256(
            "formal failure proof release root", self.expected_root_manifest_sha256
        )
        if type(self.result) is not dict:
            raise TypeError("formal failure proof result must be an object")

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

    @classmethod
    def from_dict(cls, value: object) -> FormalFailureActuationProofArtifact:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("formal failure proof fields differ")
        row = dict(value)
        raw = CanonicalJsonProofBinding.from_dict(row.pop("raw_terminal"))
        control = ControlArtifactAttestation.from_dict(row.pop("control_attestation"))
        reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("replay_reservation")
        )
        return cls(
            **row,
            raw_terminal=raw,
            control_attestation=control,
            replay_reservation=reservation,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


_VERIFIED_FORMAL_FAILURE_RESULT_SEAL = object()


@dataclass(frozen=True, init=False)
class VerifiedFormalFailureActuationResult:
    assignment_sha256: str
    inventory_sha256: str
    registry_sha256: str
    cell_id: str
    scenario: str
    serving_execution_plan_sha256: str
    formal_failure_execution_binding_sha256: str
    control_envelope_sha256: str
    replay_reservation_sha256: str
    recovery_receipt_sha256: str
    recovered: bool
    correctness_only: Literal[True]

    def __init__(self, *, _verification_tag: object, **values) -> None:
        if _verification_tag is not _VERIFIED_FORMAL_FAILURE_RESULT_SEAL:
            raise TypeError("verified formal failure result is verifier-owned")
        if set(values) != set(self.__dataclass_fields__):
            raise TypeError("verified formal failure result fields differ")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "assignment_sha256",
            "inventory_sha256",
            "registry_sha256",
            "cell_id",
            "serving_execution_plan_sha256",
            "formal_failure_execution_binding_sha256",
            "control_envelope_sha256",
            "replay_reservation_sha256",
            "recovery_receipt_sha256",
        ):
            _require_sha256(f"verified formal failure {name}", getattr(self, name))
        if (
            self.scenario not in E5_FAILURES
            or type(self.recovered) is not bool
            or self.correctness_only is not True
        ):
            raise ValueError("verified formal failure scenario/status differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "verified_formal_e5_failure_actuation_result",
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _reopen_raw(
    raw: CanonicalJsonProofBinding,
    *,
    binding: VerifiedFormalFailureExecutionBinding,
) -> tuple[FormalFailureActuatorLaunchBinding, FormalFailureActuationReceipt]:
    value = raw.reopen()
    expected = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "source_capability_sha256",
        "formal_failure_execution_binding_sha256",
        "launch_binding",
        "launch_binding_sha256",
        "recovery_receipt",
        "recovery_receipt_sha256",
        "formal_execution_authorized",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("formal failure raw terminal fields differ")
    launch = FormalFailureActuatorLaunchBinding.from_dict(value["launch_binding"])
    receipt = FormalFailureActuationReceipt.from_dict(value["recovery_receipt"])
    _validate_launch(binding, launch)
    if (
        value["schema_version"] != 1
        or value["kind"] != "e5_unsigned_formal_failure_recovery_terminal"
        or value["protocol_sha256"] != FAILURE_ACTUATOR_PROTOCOL_SHA256
        or value["source_capability_sha256"]
        != release_failure_actuator_capability().sha256
        or value["formal_failure_execution_binding_sha256"] != binding.sha256
        or value["launch_binding_sha256"] != launch.sha256
        or value["recovery_receipt_sha256"] != receipt.sha256
        or value["formal_execution_authorized"] is not False
        or receipt.formal_failure_execution_binding_sha256 != binding.sha256
    ):
        raise ValueError("formal failure raw terminal identity differs")
    return launch, receipt


def _control_identity(
    raw: CanonicalJsonProofBinding,
    *,
    binding: VerifiedFormalFailureExecutionBinding,
    receipt: FormalFailureActuationReceipt,
) -> tuple[str, str]:
    control = content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_e5_failure_external_control_binding",
            "raw_terminal_raw_sha256": raw.raw_sha256,
            "raw_terminal_semantic_sha256": raw.semantic_sha256,
            "formal_failure_execution_binding_sha256": binding.sha256,
            "assignment_sha256": binding.subject.assignment_sha256,
            "inventory_sha256": binding.subject.inventory_sha256,
            "registry_sha256": binding.subject.registry_sha256,
            "recovery_receipt_sha256": receipt.sha256,
        }
    )
    lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_e5_failure_external_control_lineage",
            "control_binding_sha256": control,
            "assignment_sha256": binding.subject.assignment_sha256,
            "serving_execution_plan_sha256": (
                binding.subject.serving_execution_plan_sha256
            ),
            "run_nonce_sha256": binding.subject.run_nonce_sha256,
        }
    )
    return control, lineage


def _result(
    *,
    binding: VerifiedFormalFailureExecutionBinding,
    receipt: FormalFailureActuationReceipt,
    control: ControlArtifactAttestation,
    reservation: ChallengeReplayReservationBinding,
) -> VerifiedFormalFailureActuationResult:
    return VerifiedFormalFailureActuationResult(
        assignment_sha256=binding.subject.assignment_sha256,
        inventory_sha256=binding.subject.inventory_sha256,
        registry_sha256=binding.subject.registry_sha256,
        cell_id=binding.subject.materialized_cell_id,
        scenario=binding.subject.scenario,
        serving_execution_plan_sha256=binding.subject.serving_execution_plan_sha256,
        formal_failure_execution_binding_sha256=binding.sha256,
        control_envelope_sha256=control.sha256,
        replay_reservation_sha256=reservation.reservation_sha256,
        recovery_receipt_sha256=receipt.sha256,
        recovered=receipt.recovered,
        correctness_only=True,
        _verification_tag=_VERIFIED_FORMAL_FAILURE_RESULT_SEAL,
    )


def publish_formal_failure_actuation_proof_artifact(
    raw_terminal_path: str,
    *,
    binding: VerifiedFormalFailureExecutionBinding,
    expected_root_manifest_sha256: str,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    verified = require_verified_formal_failure_execution_binding(binding)
    output = _new_output(proof_artifact_path)
    raw = CanonicalJsonProofBinding.bind(raw_terminal_path)
    _launch, receipt = _reopen_raw(raw, binding=verified)
    control_sha, lineage_sha = _control_identity(
        raw,
        binding=verified,
        receipt=receipt,
    )
    subject = control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != control_sha
        or subject.protocol_sha256 != FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != verified.subject.registry_sha256
        or subject.lineage_sha256 != lineage_sha
        or control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("formal failure external control differs")
    controls = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=verified.subject.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=(verified.subject.run_nonce_sha256,),
    )
    reservation_sha = control_challenge_reservation_sha256(
        controls,
        reserved_ns=now_ns,
        additional_challenge_sha256s=(verified.subject.run_nonce_sha256,),
    )
    reservation = replay_store.bind_reservation(reservation_sha)
    result = _result(
        binding=verified,
        receipt=receipt,
        control=control_attestation,
        reservation=reservation,
    )
    artifact = FormalFailureActuationProofArtifact(
        schema_version=1,
        kind="formal_e5_failure_actuation_proof_artifact",
        raw_terminal=raw,
        control_attestation=control_attestation,
        replay_reservation=reservation,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        result=result.to_dict(),
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)


def validate_formal_failure_actuation_proof_artifact(
    proof_artifact_path: str,
    *,
    binding: VerifiedFormalFailureExecutionBinding,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> VerifiedFormalFailureActuationResult:
    verified = require_verified_formal_failure_execution_binding(binding)
    proof = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalFailureActuationProofArtifact.from_dict(proof.reopen())
    if (
        artifact.sha256 != proof.semantic_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("formal failure proof identity/root differs")
    _launch, receipt = _reopen_raw(artifact.raw_terminal, binding=verified)
    control_sha, lineage_sha = _control_identity(
        artifact.raw_terminal,
        binding=verified,
        receipt=receipt,
    )
    subject = artifact.control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != control_sha
        or subject.protocol_sha256 != FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != verified.subject.registry_sha256
        or subject.lineage_sha256 != lineage_sha
        or artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("formal failure proof control differs")
    reserved = artifact.replay_reservation.revalidate()
    if type(now_ns) is not int or now_ns < artifact.replay_reservation.reserved_ns:
        raise ValueError("formal failure proof time precedes reservation")
    control = verify_release_control_artifact_attestation(
        artifact.control_attestation,
        expected_inventory_sha256=verified.subject.inventory_sha256,
        now_ns=artifact.replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    expected_challenges = tuple(
        sorted(
            {
                verified.subject.run_nonce_sha256,
                control.challenge_sha256,
                control.deployment_policy_challenge_sha256,
            }
        )
    )
    expected_reservation = control_challenge_reservation_sha256(
        (control,),
        reserved_ns=artifact.replay_reservation.reserved_ns,
        additional_challenge_sha256s=(verified.subject.run_nonce_sha256,),
    )
    if (
        reserved != expected_challenges
        or artifact.replay_reservation.reservation_sha256 != expected_reservation
    ):
        raise ValueError("formal failure proof replay reservation differs")
    result = _result(
        binding=verified,
        receipt=receipt,
        control=artifact.control_attestation,
        reservation=artifact.replay_reservation,
    )
    if result.to_dict() != artifact.result:
        raise ValueError("formal failure proof derived result changed")
    return result


__all__ = [
    "FormalFailureActuationProofArtifact",
    "FormalFailureActuationReceipt",
    "FormalFailureActuatorLaunchBinding",
    "VerifiedFormalFailureActuationResult",
    "execute_formal_failure_actuator_unsigned",
    "publish_formal_failure_actuation_proof_artifact",
    "validate_formal_failure_actuation_proof_artifact",
]
