"""Path-bound local trust lift for one callback-free formal serving run.

The physical host publishes only unsigned terminal, native-ITL, and lifecycle
artifacts.  This module derives every offline signing subject from the sealed
run plan, then publishes durable proofs after external control.  It never
accepts request text, token IDs, launch argv, ports, or a transport callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import load_run_config
from lightcone_spec.experiments.itl_authority import (
    StageItlExecutionIdentity,
    StageItlTimestampAuthority,
    StageItlTimestampProofArtifact,
    build_stage_itl_control_subject,
    publish_stage_itl_timestamp_proof_artifact,
    publish_stage_itl_timestamp_raw_receipt,
    validate_stage_itl_timestamp_proof_artifact,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FormalServingRunPlan,
)
from lightcone_spec.orchestration.formal_terminal_result import (
    FormalDistributedTerminalResultProjection,
    build_formal_terminal_control_subject,
    formal_scored_native_itl_pointers,
    publish_formal_terminal_result_proof_artifact,
    validate_formal_terminal_result_proof_artifact,
)
from lightcone_spec.orchestration.live_sglang import (
    build_pinned_sglang_lifecycle_timing_control_subject,
    publish_pinned_sglang_lifecycle_timing_proof_artifact,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
)
from lightcone_spec.runtime.distributed import (
    DistributedRuntimeGpuProofArtifact,
    DistributedRuntimeGpuProofReceipt,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NativeRuntimeGpuProofArtifact,
    NativeRuntimeGpuProofReceipt,
)


def _load_plan(plan_path: str | Path) -> FormalServingRunPlan:
    binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = FormalServingRunPlan.from_dict(binding.reopen())
    if plan.sha256 != binding.semantic_sha256:
        raise ValueError("formal serving lift run-plan identity changed")
    return plan


def formal_serving_stage_itl_execution_identity(
    plan_path: str | Path,
    *,
    expected_registry_sha256: str,
) -> StageItlExecutionIdentity:
    """Derive the stage-neutral ITL identity from the immutable run plan."""

    plan = _load_plan(plan_path)
    binding = plan.native_terminal_binding
    return StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id=plan.materialized_cell_id,
        inventory_sha256=plan.inventory_sha256,
        registry_sha256=expected_registry_sha256,
        execution_plan_sha256=binding.execution_plan_sha256,
        rank_config_sha256=binding.rank_config_sha256,
        run_id=binding.run_id,
        run_nonce_sha256=binding.run_nonce_sha256,
        attempt_id=binding.attempt_id,
        method=binding.method,  # type: ignore[arg-type]
        runtime_trust_mode=binding.runtime_trust_mode,
        formal_measurement=binding.formal_measurement,
    )


def formal_serving_stage_itl_gpu_proof_path(
    plan_path: str | Path,
) -> str:
    """Select the exact path-bound native or distributed ITL GPU proof."""

    plan = _load_plan(plan_path)
    matches: list[str] = []
    for binding in plan.runtime_gpu_proof_artifacts:
        value = binding.reopen()
        if type(value) is not dict:
            raise TypeError("formal serving runtime GPU proof must be an object")
        kind = value.get("kind")
        if kind == "lightcone_native_runtime_gpu_proof_artifact":
            artifact = NativeRuntimeGpuProofArtifact.from_dict(value)
            receipt = NativeRuntimeGpuProofReceipt.from_dict(artifact.receipt.reopen())
            if plan.topology_mode == "tp1_dp1" and (
                receipt.suite_id == "native_hot_path_tp1"
            ):
                matches.append(binding.absolute_path)
        elif kind == "lightcone_distributed_runtime_gpu_proof_artifact":
            artifact = DistributedRuntimeGpuProofArtifact.from_dict(value)
            receipt = DistributedRuntimeGpuProofReceipt.from_dict(
                artifact.receipt.reopen()
            )
            if receipt.topology_mode == plan.topology_mode:
                matches.append(binding.absolute_path)
        else:
            raise ValueError("formal serving runtime GPU proof kind is unsupported")
    if len(matches) != 1:
        raise ValueError("formal serving plan lacks one exact ITL GPU proof")
    return matches[0]


def build_formal_serving_terminal_control_subject(
    plan_path: str | Path,
    *,
    expected_registry_sha256: str,
) -> ControlArtifactSubject:
    plan = _load_plan(plan_path)
    return build_formal_terminal_control_subject(
        plan_path=str(Path(plan_path)),
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )


def publish_formal_serving_terminal_proof(
    plan_path: str | Path,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str | Path,
) -> CanonicalJsonProofBinding:
    plan = _load_plan(plan_path)
    return publish_formal_terminal_result_proof_artifact(
        plan_path=str(Path(plan_path)),
        control_attestation=control_attestation,
        replay_store=replay_store,
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
        proof_artifact_path=str(Path(proof_artifact_path)),
    )


def publish_formal_serving_itl_raw_receipt(
    plan_path: str | Path,
    *,
    native_result_proof_path: str | Path,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    raw_receipt_path: str | Path,
) -> CanonicalJsonProofBinding:
    identity = formal_serving_stage_itl_execution_identity(
        plan_path,
        expected_registry_sha256=expected_registry_sha256,
    )
    pointers = formal_scored_native_itl_pointers(
        plan_path=str(Path(plan_path)),
        expected_inventory_sha256=identity.inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    return publish_stage_itl_timestamp_raw_receipt(
        str(Path(raw_receipt_path)),
        native_result_proof_path=str(Path(native_result_proof_path)),
        native_gpu_proof_path=formal_serving_stage_itl_gpu_proof_path(plan_path),
        execution_identity=identity,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        native_result_pointers=pointers,
        now_ns=now_ns,
    )


def build_formal_serving_itl_control_subject(
    plan_path: str | Path,
    *,
    raw_receipt_path: str | Path,
    native_result_proof_path: str | Path,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> ControlArtifactSubject:
    identity = formal_serving_stage_itl_execution_identity(
        plan_path,
        expected_registry_sha256=expected_registry_sha256,
    )
    return build_stage_itl_control_subject(
        str(Path(raw_receipt_path)),
        native_result_proof_path=str(Path(native_result_proof_path)),
        native_gpu_proof_path=formal_serving_stage_itl_gpu_proof_path(plan_path),
        execution_identity=identity,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
    )


def publish_formal_serving_itl_proof(
    plan_path: str | Path,
    *,
    raw_receipt_path: str | Path,
    native_result_proof_path: str | Path,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str | Path,
) -> CanonicalJsonProofBinding:
    identity = formal_serving_stage_itl_execution_identity(
        plan_path,
        expected_registry_sha256=expected_registry_sha256,
    )
    output = Path(proof_artifact_path)
    inner_path = output.with_name(f"{output.name}.stage-itl.json")
    inner = publish_stage_itl_timestamp_proof_artifact(
        str(Path(raw_receipt_path)),
        native_result_proof_path=str(Path(native_result_proof_path)),
        native_gpu_proof_path=formal_serving_stage_itl_gpu_proof_path(plan_path),
        execution_identity=identity,
        control_attestation=control_attestation,
        replay_store=replay_store,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
        proof_artifact_path=str(inner_path),
    )
    authority = validate_stage_itl_timestamp_proof_artifact(
        inner.absolute_path,
        expected_inventory_sha256=identity.inventory_sha256,
        expected_registry_sha256=identity.registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_execution_plan_sha256=identity.execution_plan_sha256,
        expected_rank_config_sha256=identity.rank_config_sha256,
        expected_run_id=identity.run_id,
        expected_run_nonce_sha256=identity.run_nonce_sha256,
        expected_attempt_id=identity.attempt_id,
        expected_method=identity.method,
        now_ns=now_ns,
    )
    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = _load_plan(plan_path)
    terminal = CanonicalJsonProofBinding.bind(native_result_proof_path)
    launch_admission, launch_consumption, budget_consumption = (
        _formal_serving_launch_bindings(plan, now_ns=now_ns)
    )
    artifact = FormalServingItlProofArtifact(
        schema_version=1,
        kind="formal_serving_itl_proof_artifact",
        plan=plan_binding,
        stage_itl_proof=inner,
        terminal_result_proof=terminal,
        launch_admission=launch_admission,
        launch_consumption=launch_consumption,
        budget_consumption=budget_consumption,
        authority_sha256=authority.sha256,
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)


def _formal_serving_launch_bindings(
    plan: FormalServingRunPlan,
    *,
    now_ns: int,
) -> tuple[
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
]:
    from lightcone_spec.orchestration.formal_launch_admission import (
        validate_formal_stage_launch_evidence_lineage,
    )

    receipt_binding = CanonicalJsonProofBinding.bind(plan.live_run_receipt_output_path)
    receipt = receipt_binding.reopen()
    if type(receipt) is not dict:
        raise TypeError("formal serving run receipt must be an object")
    keys = (
        "formal_launch_admission",
        "formal_launch_consumption",
        "budget_consumption",
    )
    try:
        admission, consumption, budget = tuple(
            CanonicalJsonProofBinding.from_dict(receipt[key]) for key in keys
        )
    except KeyError as error:
        raise ValueError("formal serving run receipt lacks launch lineage") from error
    validate_formal_stage_launch_evidence_lineage(
        admission=admission,
        launch_consumption=consumption,
        budget_consumption=budget,
        run_plan_path=plan.private_output_root + "/formal-serving-run-plan.json",
        current_ns=now_ns,
    )
    return admission, consumption, budget


@dataclass(frozen=True)
class FormalServingItlProofArtifact:
    """Formal-only ITL wrapper joining timing control to launch admission."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_itl_proof_artifact"]
    plan: CanonicalJsonProofBinding
    stage_itl_proof: CanonicalJsonProofBinding
    terminal_result_proof: CanonicalJsonProofBinding
    launch_admission: CanonicalJsonProofBinding
    launch_consumption: CanonicalJsonProofBinding
    budget_consumption: CanonicalJsonProofBinding
    authority_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "plan": self.plan.to_dict(),
            "stage_itl_proof": self.stage_itl_proof.to_dict(),
            "terminal_result_proof": self.terminal_result_proof.to_dict(),
            "launch_admission": self.launch_admission.to_dict(),
            "launch_consumption": self.launch_consumption.to_dict(),
            "budget_consumption": self.budget_consumption.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("formal serving ITL proof fields differ")
        row = dict(value)
        for name in (
            "plan",
            "stage_itl_proof",
            "terminal_result_proof",
            "launch_admission",
            "launch_consumption",
            "budget_consumption",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.schema_version != 1 or artifact.kind != (
            "formal_serving_itl_proof_artifact"
        ):
            raise ValueError("formal serving ITL proof schema differs")
        if len(artifact.authority_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in artifact.authority_sha256
        ):
            raise ValueError("formal serving ITL authority digest differs")
        return artifact


def validate_formal_serving_itl_proof(
    proof_artifact_path: str | Path,
    *,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> StageItlTimestampAuthority:
    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalServingItlProofArtifact.from_dict(binding.reopen())
    plan = _load_plan(artifact.plan.absolute_path)
    if binding.semantic_sha256 != artifact.sha256:
        raise ValueError("formal serving ITL proof semantic identity changed")
    for member in (
        artifact.plan,
        artifact.stage_itl_proof,
        artifact.terminal_result_proof,
        artifact.launch_admission,
        artifact.launch_consumption,
        artifact.budget_consumption,
    ):
        if CanonicalJsonProofBinding.bind(member.absolute_path) != member:
            raise ValueError("formal serving ITL proof member identity changed")
    expected_launch = _formal_serving_launch_bindings(plan, now_ns=now_ns)
    if expected_launch != (
        artifact.launch_admission,
        artifact.launch_consumption,
        artifact.budget_consumption,
    ):
        raise ValueError("formal serving ITL launch lineage differs")
    stage_artifact = StageItlTimestampProofArtifact.from_dict(
        artifact.stage_itl_proof.reopen()
    )
    if stage_artifact.native_result_proof != artifact.terminal_result_proof:
        raise ValueError("formal serving ITL terminal join differs")
    identity = formal_serving_stage_itl_execution_identity(
        artifact.plan.absolute_path,
        expected_registry_sha256=expected_registry_sha256,
    )
    authority = validate_stage_itl_timestamp_proof_artifact(
        artifact.stage_itl_proof.absolute_path,
        expected_inventory_sha256=identity.inventory_sha256,
        expected_registry_sha256=identity.registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_execution_plan_sha256=identity.execution_plan_sha256,
        expected_rank_config_sha256=identity.rank_config_sha256,
        expected_run_id=identity.run_id,
        expected_run_nonce_sha256=identity.run_nonce_sha256,
        expected_attempt_id=identity.attempt_id,
        expected_method=identity.method,
        now_ns=now_ns,
    )
    if authority.sha256 != artifact.authority_sha256:
        raise ValueError("formal serving ITL authority projection changed")
    return authority


def _telemetry_detail(plan: FormalServingRunPlan) -> Literal["headline", "profile"]:
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    detail = load_run_config(launch.run_config_path).runtime.telemetry_detail
    if detail not in {"headline", "profile"}:
        raise ValueError("formal serving telemetry detail is unsupported")
    return detail


@dataclass(frozen=True)
class FormalDistributedLifecycleTimingProofArtifact:
    """Distributed lifecycle proof inherited from its controlled terminal DAG."""

    schema_version: Literal[1]
    kind: Literal["formal_distributed_lifecycle_timing_proof_artifact"]
    plan: CanonicalJsonProofBinding
    terminal_result_proof: CanonicalJsonProofBinding
    raw_lifecycle_timing: CanonicalJsonProofBinding
    launch_admission: CanonicalJsonProofBinding
    launch_consumption: CanonicalJsonProofBinding
    budget_consumption: CanonicalJsonProofBinding
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    topology_mode: Literal["tp2_dp1", "tp1_dp2"]
    terminal_projection_sha256: str

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "plan": self.plan.to_dict(),
            "terminal_result_proof": self.terminal_result_proof.to_dict(),
            "raw_lifecycle_timing": self.raw_lifecycle_timing.to_dict(),
            "launch_admission": self.launch_admission.to_dict(),
            "launch_consumption": self.launch_consumption.to_dict(),
            "budget_consumption": self.budget_consumption.to_dict(),
            "expected_inventory_sha256": self.expected_inventory_sha256,
            "expected_registry_sha256": self.expected_registry_sha256,
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
            "topology_mode": self.topology_mode,
            "terminal_projection_sha256": self.terminal_projection_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("formal distributed lifecycle proof fields differ")
        row = dict(value)
        plan = CanonicalJsonProofBinding.from_dict(row.pop("plan"))
        terminal = CanonicalJsonProofBinding.from_dict(row.pop("terminal_result_proof"))
        raw_timing = CanonicalJsonProofBinding.from_dict(
            row.pop("raw_lifecycle_timing")
        )
        launch_admission = CanonicalJsonProofBinding.from_dict(
            row.pop("launch_admission")
        )
        launch_consumption = CanonicalJsonProofBinding.from_dict(
            row.pop("launch_consumption")
        )
        budget_consumption = CanonicalJsonProofBinding.from_dict(
            row.pop("budget_consumption")
        )
        return cls(
            **row,  # type: ignore[arg-type]
            plan=plan,
            terminal_result_proof=terminal,
            raw_lifecycle_timing=raw_timing,
            launch_admission=launch_admission,
            launch_consumption=launch_consumption,
            budget_consumption=budget_consumption,
        )


def validate_formal_distributed_lifecycle_timing_proof_artifact(
    proof_artifact_path: str | Path,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> FormalDistributedLifecycleTimingProofArtifact:
    """Deep-reopen the distributed lifecycle projection and its terminal DAG."""

    proof_binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalDistributedLifecycleTimingProofArtifact.from_dict(
        proof_binding.reopen()
    )
    plan_binding = CanonicalJsonProofBinding.bind(artifact.plan.absolute_path)
    terminal_binding = CanonicalJsonProofBinding.bind(
        artifact.terminal_result_proof.absolute_path
    )
    raw_binding = CanonicalJsonProofBinding.bind(
        artifact.raw_lifecycle_timing.absolute_path
    )
    admission_binding = CanonicalJsonProofBinding.bind(
        artifact.launch_admission.absolute_path
    )
    launch_consumption_binding = CanonicalJsonProofBinding.bind(
        artifact.launch_consumption.absolute_path
    )
    budget_consumption_binding = CanonicalJsonProofBinding.bind(
        artifact.budget_consumption.absolute_path
    )
    if (
        proof_binding.semantic_sha256 != artifact.sha256
        or plan_binding != artifact.plan
        or terminal_binding != artifact.terminal_result_proof
        or raw_binding != artifact.raw_lifecycle_timing
        or admission_binding != artifact.launch_admission
        or launch_consumption_binding != artifact.launch_consumption
        or budget_consumption_binding != artifact.budget_consumption
    ):
        raise ValueError("formal distributed lifecycle proof path identity changed")
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if (
        plan.sha256 != plan_binding.semantic_sha256
        or plan.topology_mode not in {"tp2_dp1", "tp1_dp2"}
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
        or artifact.topology_mode != plan.topology_mode
        or raw_binding.absolute_path != plan.lifecycle_timing_output_path
    ):
        raise ValueError("formal distributed lifecycle proof authority differs")
    projection = validate_formal_terminal_result_proof_artifact(
        terminal_binding.absolute_path,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_execution_plan_sha256=(
            plan.native_terminal_binding.execution_plan_sha256
        ),
        expected_rank_config_sha256=plan.native_terminal_binding.rank_config_sha256,
        expected_run_id=plan.native_terminal_binding.run_id,
        expected_run_nonce_sha256=plan.native_terminal_binding.run_nonce_sha256,
        expected_attempt_id=plan.native_terminal_binding.attempt_id,
        expected_method=plan.native_terminal_binding.method,
        expected_stage=plan.stage,
        expected_topology=plan.topology_mode,
        now_ns=now_ns,
    )
    if (
        type(projection) is not FormalDistributedTerminalResultProjection
        or projection.sha256 != artifact.terminal_projection_sha256
        or projection.lifecycle_timing_sha256 != raw_binding.semantic_sha256
        or projection.launch_admission_sha256 != admission_binding.semantic_sha256
        or projection.launch_consumption_sha256
        != launch_consumption_binding.semantic_sha256
        or projection.budget_consumption_sha256
        != budget_consumption_binding.semantic_sha256
    ):
        raise ValueError("formal distributed lifecycle terminal DAG differs")
    return artifact


def build_formal_serving_lifecycle_control_subject(
    plan_path: str | Path,
    *,
    native_result_proof_path: str | Path,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> ControlArtifactSubject:
    """Build TP1 lifecycle control, or the shared distributed terminal control."""

    plan = _load_plan(plan_path)
    if plan.topology_mode != "tp1_dp1":
        return build_formal_serving_terminal_control_subject(
            plan_path,
            expected_registry_sha256=expected_registry_sha256,
        )
    # A formal TP1 lifecycle control is never allowed to bless a legacy raw
    # run.  Deep-open the persisted launch admission and both one-shot
    # consumptions before deriving the signing subject.
    _formal_serving_launch_bindings(plan, now_ns=now_ns)
    return build_pinned_sglang_lifecycle_timing_control_subject(
        plan.lifecycle_timing_output_path,
        live_run_receipt_path=plan.live_run_receipt_output_path,
        native_result_proof_artifact_path=str(Path(native_result_proof_path)),
        expected_binding=plan.native_terminal_binding,
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_telemetry_detail=_telemetry_detail(plan),
        now_ns=now_ns,
    )


def publish_formal_serving_lifecycle_proof(
    plan_path: str | Path,
    *,
    native_result_proof_path: str | Path,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str | Path,
    control_attestation: ControlArtifactAttestation | None = None,
    replay_store: ChallengeReplayStore | None = None,
) -> CanonicalJsonProofBinding:
    """Publish TP1 lifecycle control or a distributed controlled projection."""

    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = _load_plan(plan_path)
    if plan.topology_mode == "tp1_dp1":
        if (
            type(control_attestation) is not ControlArtifactAttestation
            or type(replay_store) is not ChallengeReplayStore
        ):
            raise TypeError("TP1 lifecycle proof requires control and replay store")
        _formal_serving_launch_bindings(plan, now_ns=now_ns)
        return publish_pinned_sglang_lifecycle_timing_proof_artifact(
            plan.lifecycle_timing_output_path,
            live_run_receipt_path=plan.live_run_receipt_output_path,
            native_result_proof_artifact_path=str(Path(native_result_proof_path)),
            control_attestation=control_attestation,
            replay_store=replay_store,
            expected_binding=plan.native_terminal_binding,
            expected_inventory_sha256=plan.inventory_sha256,
            expected_registry_sha256=expected_registry_sha256,
            expected_root_manifest_sha256=expected_root_manifest_sha256,
            expected_telemetry_detail=_telemetry_detail(plan),
            now_ns=now_ns,
            proof_artifact_path=str(Path(proof_artifact_path)),
        )
    if control_attestation is not None or replay_store is not None:
        raise ValueError("distributed lifecycle is covered by terminal control")
    terminal_binding = CanonicalJsonProofBinding.bind(native_result_proof_path)
    projection = validate_formal_terminal_result_proof_artifact(
        terminal_binding.absolute_path,
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_execution_plan_sha256=(
            plan.native_terminal_binding.execution_plan_sha256
        ),
        expected_rank_config_sha256=plan.native_terminal_binding.rank_config_sha256,
        expected_run_id=plan.native_terminal_binding.run_id,
        expected_run_nonce_sha256=plan.native_terminal_binding.run_nonce_sha256,
        expected_attempt_id=plan.native_terminal_binding.attempt_id,
        expected_method=plan.native_terminal_binding.method,
        expected_stage=plan.stage,
        expected_topology=plan.topology_mode,
        now_ns=now_ns,
    )
    if type(projection) is not FormalDistributedTerminalResultProjection:
        raise TypeError("distributed lifecycle requires distributed terminal proof")
    raw_timing = CanonicalJsonProofBinding.bind(plan.lifecycle_timing_output_path)
    if projection.lifecycle_timing_sha256 != raw_timing.semantic_sha256:
        raise ValueError("distributed lifecycle differs from terminal control DAG")
    run_receipt = CanonicalJsonProofBinding.bind(
        plan.live_run_receipt_output_path
    ).reopen()
    if type(run_receipt) is not dict:
        raise TypeError("distributed lifecycle run receipt must be an object")
    launch_admission = CanonicalJsonProofBinding.from_dict(
        run_receipt["formal_launch_admission"]
    )
    launch_consumption = CanonicalJsonProofBinding.from_dict(
        run_receipt["formal_launch_consumption"]
    )
    budget_consumption = CanonicalJsonProofBinding.from_dict(
        run_receipt["budget_consumption"]
    )
    if (
        projection.launch_admission_sha256 != launch_admission.semantic_sha256
        or projection.launch_consumption_sha256 != launch_consumption.semantic_sha256
        or projection.budget_consumption_sha256 != budget_consumption.semantic_sha256
    ):
        raise ValueError("distributed lifecycle launch-admission DAG differs")
    artifact = FormalDistributedLifecycleTimingProofArtifact(
        schema_version=1,
        kind="formal_distributed_lifecycle_timing_proof_artifact",
        plan=plan_binding,
        terminal_result_proof=terminal_binding,
        raw_lifecycle_timing=raw_timing,
        launch_admission=launch_admission,
        launch_consumption=launch_consumption,
        budget_consumption=budget_consumption,
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        topology_mode=plan.topology_mode,  # type: ignore[arg-type]
        terminal_projection_sha256=projection.sha256,
    )
    publish_canonical_json_no_replace(
        str(Path(proof_artifact_path)), artifact.to_dict()
    )
    return CanonicalJsonProofBinding.bind(
        proof_artifact_path,
        semantic_sha256=artifact.sha256,
    )


__all__ = [
    "FormalDistributedLifecycleTimingProofArtifact",
    "FormalServingItlProofArtifact",
    "build_formal_serving_itl_control_subject",
    "build_formal_serving_lifecycle_control_subject",
    "build_formal_serving_terminal_control_subject",
    "formal_serving_stage_itl_execution_identity",
    "formal_serving_stage_itl_gpu_proof_path",
    "publish_formal_serving_itl_proof",
    "publish_formal_serving_itl_raw_receipt",
    "publish_formal_serving_lifecycle_proof",
    "publish_formal_serving_terminal_proof",
    "validate_formal_distributed_lifecycle_timing_proof_artifact",
    "validate_formal_serving_itl_proof",
]
