"""Current-cell dispatch for the trusted single-operator workflow.

This module is deliberately a router, not a second experiment registry.  It
deep-opens the current execution source, finds exactly one materialized cell,
and selects one of the already registered physical runners from the cell's
stage/task identity.  No method, recipe, topology, argv, port, or workload
value is accepted from the caller.

Only E4 screen/local currently has a complete source-owned downstream launch
mapper.  The route table nevertheless covers every downstream cell so the
operator cannot accidentally send profiler, failure, compatibility, or
NEXTN-interface work through the ordinary serving validator.  A route being
known does not make its physical inputs available: callers must still provide
the durable source bundle required by that runner, and missing source-owned
authority remains a fail-closed error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_preflight_inputs import (
    FormalPreflightExecutionInputs,
)
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorExecutionSource,
    FormalSingleOperatorNode,
    formal_single_operator_node_spec,
    load_formal_single_operator_execution_source,
)
from lightcone_spec.experiments.stage_materialization import MaterializedCell
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FormalSingleOperatorPhysicalKind = Literal[
    "serving",
    "profiler",
    "e5_failure",
    "e6_interface_preflight",
    "e0_compatibility_decision",
]
FormalSingleOperatorAuxiliaryKind = Literal["e0_compatibility"]

_EARLY_NODES = frozenset({"e3a", "tts_cal", "e1", "e2_r0", "e2_r1", "e2_r2", "e2_r3"})
_E4_HEADLINE_TASKS = frozenset(
    {
        "mechanism_strength2_screen_headline",
        "winner_neighborhood_local_factorial_headline",
    }
)
_E6_SERVING_TASKS = frozenset({"LiveCodeBench", "MATH-500"})
_ONLINE_SPEC_ROLES = frozenset(
    {
        "OnlineSPEC-OGD",
        "OnlineSPEC-OPT",
        "OnlineSPEC-ENS",
        "OnlineSPEC-Optimistic-OGD",
        "OnlineSPEC-Hedge",
        "OnlineSPEC-OGD-candidate",
        "OnlineSPEC-OPT-candidate",
        "OnlineSPEC-ENS-candidate",
        "OnlineSPEC-Optimistic-OGD-candidate",
        "OnlineSPEC-Hedge-candidate",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class FormalSingleOperatorDispatchBlocked(RuntimeError):
    """A registered route is missing its source-owned physical input."""

    def __init__(self, reason_code: str) -> None:
        if (
            type(reason_code) is not str
            or not reason_code
            or reason_code.strip() != reason_code
        ):
            raise ValueError("single-operator dispatch reason must be canonical text")
        self.reason_code = reason_code
        super().__init__(f"single-operator dispatch is BLOCKED: {reason_code}")


@dataclass(frozen=True)
class FormalSingleOperatorCellDispatch:
    """Code-owned selection of the physical runner for one exact current cell."""

    node: FormalSingleOperatorNode
    stage: str
    phase: str
    materialized_cell_id: str
    task: str
    method_role: str
    physical_kind: FormalSingleOperatorPhysicalKind
    expected_terminal_kind: str

    def __post_init__(self) -> None:
        for value in (self.stage, self.phase, self.task, self.method_role):
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError("single-operator dispatch text is invalid")
        if (
            type(self.materialized_cell_id) is not str
            or len(self.materialized_cell_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.materialized_cell_id
            )
        ):
            raise ValueError("single-operator dispatch cell ID is invalid")
        expected_terminal = {
            "serving": "formal_single_operator_run_manifest",
            "profiler": "formal_single_operator_profiler_terminal",
            "e5_failure": "formal_single_operator_e5_physical_outcome",
            "e6_interface_preflight": "e6_nextn_model_authority_input",
            "e0_compatibility_decision": (
                "formal_single_operator_e0_compatibility_decision_terminal"
            ),
        }[self.physical_kind]
        if self.expected_terminal_kind != expected_terminal:
            raise ValueError("single-operator dispatch terminal kind differs")


def route_formal_single_operator_materialized_cell(
    *,
    node: FormalSingleOperatorNode,
    phase: str,
    cell: MaterializedCell,
) -> FormalSingleOperatorCellDispatch:
    """Select a runner strictly from the registered stage/task identity."""

    spec = formal_single_operator_node_spec(node)
    if cell.stage != spec.stage or phase != spec.phase:
        raise ValueError("single-operator dispatch node/stage/phase differs")
    if node in _EARLY_NODES:
        physical_kind: FormalSingleOperatorPhysicalKind = "serving"
    elif cell.stage == "E4":
        if cell.task in _E4_HEADLINE_TASKS and node in {"e4_screen", "e4_local"}:
            physical_kind = "serving"
        elif cell.task == "mechanism_profile_only" and node == "e4_profiler":
            physical_kind = "profiler"
        else:
            raise ValueError("single-operator E4 cell has no registered physical route")
    elif cell.stage == "E3b" or cell.stage == "E1a":
        physical_kind = "serving"
    elif cell.stage == "E5":
        if cell.task == "production_slo_power_prefix":
            physical_kind = "serving"
        elif cell.task == "deterministic_failure_injection":
            physical_kind = "e5_failure"
        else:
            raise ValueError("single-operator E5 cell has no registered physical route")
    elif cell.stage == "E6":
        if cell.task == "immutable_metadata_interface_and_fit_preflight":
            physical_kind = "e6_interface_preflight"
        elif cell.task in _E6_SERVING_TASKS:
            physical_kind = "serving"
        else:
            raise ValueError("single-operator E6 cell has no registered physical route")
    elif cell.stage == "E0":
        if (
            node == "e0_tuning"
            and cell.method_role == "Compatibility"
            and cell.task == "compatibility_decision"
        ):
            physical_kind = "e0_compatibility_decision"
        else:
            allowed_roles = {
                "Target-only",
                "Static",
                "TTS",
                "L0-naive",
                "LightCone",
                *_ONLINE_SPEC_ROLES,
            }
            if cell.method_role not in allowed_roles:
                raise ValueError(
                    "single-operator E0 role has no registered physical route"
                )
            physical_kind = "serving"
    else:
        raise ValueError("single-operator cell stage has no registered physical route")
    terminal = {
        "serving": "formal_single_operator_run_manifest",
        "profiler": "formal_single_operator_profiler_terminal",
        "e5_failure": "formal_single_operator_e5_physical_outcome",
        "e6_interface_preflight": "e6_nextn_model_authority_input",
        "e0_compatibility_decision": (
            "formal_single_operator_e0_compatibility_decision_terminal"
        ),
    }[physical_kind]
    return FormalSingleOperatorCellDispatch(
        node=node,
        stage=cell.stage,
        phase=phase,
        materialized_cell_id=cell.cell_id,
        task=cell.task,
        method_role=cell.method_role,
        physical_kind=physical_kind,
        expected_terminal_kind=terminal,
    )


def route_formal_single_operator_cell(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
) -> tuple[
    FormalSingleOperatorExecutionSource,
    MaterializedCell,
    FormalSingleOperatorCellDispatch,
]:
    """Deep-open the current source and route exactly one materialized cell."""

    source = load_formal_single_operator_execution_source(execution_source_path)
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="single-operator dispatch materialization"
        )
    )
    matches = tuple(
        cell for cell in materialization.cells if cell.cell_id == materialized_cell_id
    )
    if len(matches) != 1:
        raise ValueError("single-operator dispatch cell is outside materialization")
    cell = matches[0]
    if materialization.stage != source.stage or cell.stage != source.stage:
        raise ValueError("single-operator dispatch stage lineage differs")
    route = route_formal_single_operator_materialized_cell(
        node=source.node,
        phase=source.phase,
        cell=cell,
    )
    return source, cell, route


def route_formal_single_operator_auxiliary(
    auxiliary_kind: str,
) -> FormalSingleOperatorAuxiliaryKind:
    """Select the sole pre-materialization auxiliary physical workflow."""

    if auxiliary_kind != "e0_compatibility":
        raise ValueError("single-operator auxiliary route is unsupported")
    return "e0_compatibility"


@dataclass(frozen=True)
class FormalSingleOperatorE0CompatibilityDecisionDispatch:
    """Exact non-GPU decision row selected from the canonical 108-row source."""

    materialized_cell_id: str
    evidence: CanonicalJsonProofBinding
    compatibility_bundle_sha256: str
    compatibility_receipt_sha256: str
    decision_id: str
    model: str
    backend: str
    task: str
    disposition: Literal["VALID", "N/A"]
    reason_code: str
    interface_sha256: str
    task_native_workload_sha256: str

    def __post_init__(self) -> None:
        for digest in (
            self.materialized_cell_id,
            self.compatibility_bundle_sha256,
            self.compatibility_receipt_sha256,
            self.decision_id,
            self.interface_sha256,
            self.task_native_workload_sha256,
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("E0 compatibility dispatch digest differs")
        if type(self.evidence) is not CanonicalJsonProofBinding:
            raise TypeError("E0 compatibility evidence is not path-bound")
        if CanonicalJsonProofBinding.bind(self.evidence.absolute_path) != self.evidence:
            raise ValueError("E0 compatibility evidence changed")
        for value in (self.model, self.backend, self.task, self.reason_code):
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError("E0 compatibility dispatch text differs")
        if self.disposition not in {"VALID", "N/A"}:
            raise ValueError("E0 compatibility disposition differs")


def revalidate_formal_single_operator_e0_compatibility_decision(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
) -> FormalSingleOperatorE0CompatibilityDecisionDispatch:
    """Select one materialized Compatibility row from its exact 108-row source."""

    from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
    from lightcone_spec.experiments.formal_single_operator_downstream import (
        _e0_compatibility_from_auxiliary,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorE0CompatibilityActualValidator,
        rebuild_formal_single_operator_stage_completion,
    )

    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    if route.physical_kind != "e0_compatibility_decision":
        raise FormalSingleOperatorDispatchBlocked(
            "e0_compatibility_decision_route_required"
        )
    binding = source.auxiliary_source_binding("e0_compatibility")
    auxiliary = source.reopen_auxiliary_source("e0_compatibility")
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="E0 dispatch execution materialization"
        )
    )
    if source.predecessor_completion_source is None:
        raise ValueError("E0 compatibility execution source lacks E6 completion")
    predecessor = rebuild_formal_single_operator_stage_completion(
        source.predecessor_completion_source.absolute_path
    )
    protocol_lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(label="E0 dispatch execution ProtocolLock")
    )
    validator = FormalSingleOperatorE0CompatibilityActualValidator(
        protocol_lock=protocol_lock,
        predecessor=predecessor,
        compatibility_source=binding,
    )
    validation = validator.validate(
        path=Path(binding.absolute_path),
        node=formal_single_operator_node_spec(source.node),
        materialization=materialization,
        cell=cell,
    )
    if validation.status != "COMPLETE":
        raise ValueError("E0 compatibility decision did not validate COMPLETE")
    compatibility, _authority, bundle_sha256, _evidence_sha256 = (
        _e0_compatibility_from_auxiliary(
            predecessor,
            protocol_lock,
            auxiliary,
        )
    )
    dimensions = dict(cell.dimensions)
    decision_id = dimensions.get("compatibility_decision_id")
    matches = tuple(
        row for row in compatibility.decisions if row.decision_id == decision_id
    )
    if len(matches) != 1:
        raise ValueError("E0 compatibility evidence lacks the exact decision")
    decision = matches[0]
    if (
        materialization.source_decision_sha256 != bundle_sha256
        or dimensions.get("e0_compatibility_bundle_sha256") != bundle_sha256
        or dimensions.get("compatibility_receipt_sha256") != compatibility.sha256
        or (decision.model, decision.backend) != (cell.model, cell.backend)
        or dimensions.get("deployment_task") != decision.task
    ):
        raise ValueError("E0 compatibility evidence differs from current cell")
    return FormalSingleOperatorE0CompatibilityDecisionDispatch(
        materialized_cell_id=cell.cell_id,
        evidence=CanonicalJsonProofBinding.bind(binding.absolute_path),
        compatibility_bundle_sha256=bundle_sha256,
        compatibility_receipt_sha256=compatibility.sha256,
        decision_id=decision.decision_id,
        model=decision.model,
        backend=decision.backend,
        task=decision.task,
        disposition=decision.disposition,
        reason_code=decision.reason_code,
        interface_sha256=decision.interface_sha256,
        task_native_workload_sha256=decision.task_native_workload_sha256,
    )


@dataclass(frozen=True)
class FormalSingleOperatorDownstreamRunPlanInputs:
    """Ordinary path bundle for a source-owned downstream serving launch."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_downstream_run_plan_inputs"]
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    materialized_cell_id: str
    stage: Literal["E4", "E3b", "E1a", "E5", "E6", "E0"]
    materialization: CanonicalJsonProofBinding
    materialization_sha256: str
    preflight_inputs: CanonicalJsonProofBinding
    compile_launch_manifest: CanonicalJsonProofBinding
    private_output_root: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_single_operator_downstream_run_plan_inputs"
        ):
            raise ValueError("single-operator downstream plan input schema differs")
        for binding in (
            self.execution_source,
            self.materialization,
            self.preflight_inputs,
            self.compile_launch_manifest,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("single-operator downstream input is not path-bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError("single-operator downstream input changed")
        source, cell, route = route_formal_single_operator_cell(
            execution_source_path=self.execution_source.absolute_path,
            materialized_cell_id=self.materialized_cell_id,
        )
        materialization = stage_materialization_receipt_from_dict(
            self.materialization.reopen()
        )
        preflight = FormalPreflightExecutionInputs.from_dict(
            self.preflight_inputs.reopen()
        )
        root = Path(self.private_output_root)
        if (
            source.sha256 != self.execution_source_sha256
            or source.materialization_sha256 != self.materialization_sha256
            or materialization.sha256 != self.materialization_sha256
            or source.stage != self.stage
            or cell.stage != self.stage
            or route.physical_kind != "serving"
            or preflight.sha256 != self.preflight_inputs.semantic_sha256
            or not root.is_absolute()
            or root != root.resolve(strict=False)
            or not root.is_dir()
            or root.is_symlink()
        ):
            raise ValueError("single-operator downstream plan input lineage differs")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "execution_source": self.execution_source.to_dict(),
            "execution_source_sha256": self.execution_source_sha256,
            "materialized_cell_id": self.materialized_cell_id,
            "stage": self.stage,
            "materialization": self.materialization.to_dict(),
            "materialization_sha256": self.materialization_sha256,
            "preflight_inputs": self.preflight_inputs.to_dict(),
            "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
            "private_output_root": self.private_output_root,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("single-operator downstream plan input fields differ")
        row = dict(value)
        for name in (
            "execution_source",
            "materialization",
            "preflight_inputs",
            "compile_launch_manifest",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(**row)  # type: ignore[arg-type]


def materialize_formal_single_operator_e4_direct_run_plan_inputs(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    repository_root: str | Path,
    preflight_inputs_path: str | Path,
    private_output_root: str | Path,
) -> FormalSingleOperatorDownstreamRunPlanInputs:
    """Publish the current E4 headline launch and its direct-plan descriptor."""

    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    if source.node not in {"e4_screen", "e4_local"} or route.physical_kind != "serving":
        raise FormalSingleOperatorDispatchBlocked(
            "source_owned_downstream_launch_mapper_unavailable"
        )
    preflight_binding = CanonicalJsonProofBinding.bind(preflight_inputs_path)
    preflight = FormalPreflightExecutionInputs.from_dict(preflight_binding.reopen())
    if preflight.sha256 != preflight_binding.semantic_sha256:
        raise ValueError("single-operator preflight input digest differs")
    from lightcone_spec.experiments.formal_single_operator_e4_execution import (
        materialize_formal_single_operator_e4_compile_launch,
    )

    context = materialize_formal_single_operator_e4_compile_launch(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        repository_root=repository_root,
        inventory_path=preflight.inventory.absolute_path,
        private_output_root=private_output_root,
    )
    root = Path(private_output_root).resolve(strict=False)
    launch_binding = CanonicalJsonProofBinding.bind(
        root / "formal-single-operator-e4-compile-launch.json"
    )
    value = FormalSingleOperatorDownstreamRunPlanInputs(
        schema_version=1,
        kind="formal_single_operator_downstream_run_plan_inputs",
        execution_source=CanonicalJsonProofBinding.bind(execution_source_path),
        execution_source_sha256=source.sha256,
        materialized_cell_id=cell.cell_id,
        stage="E4",
        materialization=CanonicalJsonProofBinding.bind(
            source.materialization_source.absolute_path
        ),
        materialization_sha256=context.materialization.sha256,
        preflight_inputs=preflight_binding,
        compile_launch_manifest=launch_binding,
        private_output_root=str(root),
    )
    output = root / "formal-single-operator-downstream-run-plan-inputs.json"
    publish_canonical_json_no_replace(output, value.to_dict())
    rebound = FormalSingleOperatorDownstreamRunPlanInputs.from_dict(
        CanonicalJsonProofBinding.bind(output).reopen()
    )
    if rebound != value or rebound.sha256 != value.sha256:
        raise RuntimeError("single-operator downstream plan inputs changed")
    return value


@dataclass(frozen=True)
class FormalSingleOperatorPreparedDownstreamRunPlanInputs:
    """Exact post-materialization launch/schedule inputs for one serving cell."""

    schema_version: Literal[1, 2]
    kind: Literal["formal_single_operator_prepared_downstream_run_plan_inputs"]
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    prepared_launch_bundle: CanonicalJsonProofBinding
    prepared_launch_bundle_sha256: str
    prepared_launch_entry_sha256: str
    materialized_cell_id: str
    stage: Literal["E3b", "E1a", "E5", "E6", "E0"]
    materialization: CanonicalJsonProofBinding
    materialization_sha256: str
    inventory: CanonicalJsonProofBinding
    content_verification_receipt: CanonicalJsonProofBinding | None
    compile_launch_manifest: CanonicalJsonProofBinding
    request_schedule_receipt: CanonicalJsonProofBinding
    execution_binding_sha256: str
    subject_sha256: str
    private_output_root: str
    content_source_binding: FormalContentSourceBinding | None = None
    trusted_eagle3_execution_authority: CanonicalJsonProofBinding | None = None
    trusted_chronobelief_gpu_parity_proof: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2} or self.kind != (
            "formal_single_operator_prepared_downstream_run_plan_inputs"
        ):
            raise ValueError("prepared downstream plan input schema differs")
        if self.schema_version == 1:
            if (
                type(self.content_verification_receipt) is not CanonicalJsonProofBinding
                or self.content_source_binding is not None
            ):
                raise ValueError("legacy prepared downstream content differs")
        elif (
            self.content_verification_receipt is not None
            or type(self.content_source_binding) is not FormalContentSourceBinding
            or self.content_source_binding.mode != "trusted_single_operator"
        ):
            raise ValueError("trusted prepared downstream content differs")
        for binding in (
            self.execution_source,
            self.prepared_launch_bundle,
            self.materialization,
            self.inventory,
            self.compile_launch_manifest,
            self.request_schedule_receipt,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("prepared downstream input is not path-bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError("prepared downstream input changed")
        if self.content_verification_receipt is not None and (
            CanonicalJsonProofBinding.bind(
                self.content_verification_receipt.absolute_path
            )
            != self.content_verification_receipt
        ):
            raise ValueError("prepared downstream content receipt changed")
        if self.content_source_binding is not None:
            self.content_source_binding.reopen()
        if self.trusted_eagle3_execution_authority is not None and (
            CanonicalJsonProofBinding.bind(
                self.trusted_eagle3_execution_authority.absolute_path
            )
            != self.trusted_eagle3_execution_authority
        ):
            raise ValueError("prepared downstream EAGLE3 authority changed")
        if self.trusted_chronobelief_gpu_parity_proof is not None and (
            CanonicalJsonProofBinding.bind(
                self.trusted_chronobelief_gpu_parity_proof.absolute_path
            )
            != self.trusted_chronobelief_gpu_parity_proof
        ):
            raise ValueError("prepared downstream ChronoBelief proof changed")
        for digest in (
            self.execution_source_sha256,
            self.prepared_launch_bundle_sha256,
            self.prepared_launch_entry_sha256,
            self.materialized_cell_id,
            self.materialization_sha256,
            self.execution_binding_sha256,
            self.subject_sha256,
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("prepared downstream digest differs")
        root = Path(self.private_output_root)
        if (
            not root.is_absolute()
            or root != root.resolve(strict=False)
            or not root.is_dir()
            or root.is_symlink()
        ):
            raise ValueError("prepared downstream run root differs")
        from lightcone_spec.config import load_run_config
        from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

        launch = CompileLaunchManifest.load(self.compile_launch_manifest.absolute_path)
        config = load_run_config(launch.run_config_path)
        requires_eagle3 = (
            self.schema_version == 2
            and self.stage == "E0"
            and config.model.algorithm == "EAGLE3"
            and config.adaptation is not None
        )
        if requires_eagle3 != (self.trusted_eagle3_execution_authority is not None):
            raise ValueError("prepared downstream EAGLE3 authority coverage differs")
        adaptation = config.adaptation
        requires_chronobelief = (
            self.schema_version == 2
            and self.stage == "E1a"
            and adaptation is not None
            and adaptation.optimizer.name == "chronobelief"
        )
        if requires_chronobelief != (
            self.trusted_chronobelief_gpu_parity_proof is not None
        ):
            raise ValueError("prepared downstream ChronoBelief proof coverage differs")
        if self.trusted_eagle3_execution_authority is not None:
            from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
                load_trusted_single_operator_eagle3_execution_authority,
            )

            authority = load_trusted_single_operator_eagle3_execution_authority(
                self.trusted_eagle3_execution_authority.absolute_path
            )
            if (
                authority.execution_source != self.execution_source
                or authority.execution_source_sha256 != self.execution_source_sha256
                or authority.materialized_cell_id != self.materialized_cell_id
                or authority.compile_launch_manifest != self.compile_launch_manifest
            ):
                raise ValueError("prepared downstream EAGLE3 authority lineage differs")
        if self.trusted_chronobelief_gpu_parity_proof is not None:
            from lightcone_spec.experiments.formal_single_operator_chronobelief import (
                revalidate_trusted_single_operator_chronobelief_for_prepared_launch,
            )

            revalidate_trusted_single_operator_chronobelief_for_prepared_launch(
                proof_path=(self.trusted_chronobelief_gpu_parity_proof.absolute_path),
                execution_source_path=self.execution_source.absolute_path,
                prepared_launch_path=self.compile_launch_manifest.absolute_path,
            )

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "execution_source": self.execution_source.to_dict(),
            "execution_source_sha256": self.execution_source_sha256,
            "prepared_launch_bundle": self.prepared_launch_bundle.to_dict(),
            "prepared_launch_bundle_sha256": (self.prepared_launch_bundle_sha256),
            "prepared_launch_entry_sha256": self.prepared_launch_entry_sha256,
            "materialized_cell_id": self.materialized_cell_id,
            "stage": self.stage,
            "materialization": self.materialization.to_dict(),
            "materialization_sha256": self.materialization_sha256,
            "inventory": self.inventory.to_dict(),
            "content_verification_receipt": (
                None
                if self.content_verification_receipt is None
                else self.content_verification_receipt.to_dict()
            ),
            "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
            "request_schedule_receipt": self.request_schedule_receipt.to_dict(),
            "execution_binding_sha256": self.execution_binding_sha256,
            "subject_sha256": self.subject_sha256,
            "private_output_root": self.private_output_root,
        }
        if self.schema_version == 2:
            assert self.content_source_binding is not None
            value["content_source_binding"] = self.content_source_binding.to_dict()
            value["trusted_eagle3_execution_authority"] = (
                None
                if self.trusted_eagle3_execution_authority is None
                else self.trusted_eagle3_execution_authority.to_dict()
            )
            value["trusted_chronobelief_gpu_parity_proof"] = (
                None
                if self.trusted_chronobelief_gpu_parity_proof is None
                else self.trusted_chronobelief_gpu_parity_proof.to_dict()
            )
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise ValueError("prepared downstream plan input fields differ")
        schema_version = value.get("schema_version")
        expected = set(cls.__dataclass_fields__)
        if schema_version == 1:
            expected.remove("content_source_binding")
            expected.remove("trusted_eagle3_execution_authority")
            expected.remove("trusted_chronobelief_gpu_parity_proof")
        if set(value) != expected:
            raise ValueError("prepared downstream plan input fields differ")
        row = dict(value)
        for name in (
            "execution_source",
            "prepared_launch_bundle",
            "materialization",
            "inventory",
            "compile_launch_manifest",
            "request_schedule_receipt",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_receipt = row.pop("content_verification_receipt")
        row["content_verification_receipt"] = (
            None
            if raw_receipt is None
            else CanonicalJsonProofBinding.from_dict(raw_receipt)
        )
        raw_content_source = row.pop("content_source_binding", None)
        row["content_source_binding"] = (
            None
            if raw_content_source is None
            else FormalContentSourceBinding.from_dict(raw_content_source)
        )
        raw_eagle3 = row.pop("trusted_eagle3_execution_authority", None)
        row["trusted_eagle3_execution_authority"] = (
            None
            if raw_eagle3 is None
            else CanonicalJsonProofBinding.from_dict(raw_eagle3)
        )
        raw_chronobelief = row.pop("trusted_chronobelief_gpu_parity_proof", None)
        row["trusted_chronobelief_gpu_parity_proof"] = (
            None
            if raw_chronobelief is None
            else CanonicalJsonProofBinding.from_dict(raw_chronobelief)
        )
        return cls(**row)  # type: ignore[arg-type]


def materialize_formal_single_operator_prepared_downstream_run_plan_inputs(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    prepared_launch_bundle_path: str | Path,
    private_output_root: str | Path,
    current_ns: int,
) -> FormalSingleOperatorPreparedDownstreamRunPlanInputs:
    """Publish one ordinary downstream serving input from an exact bundle row."""

    from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
        formal_single_operator_prepared_execution_identities,
        revalidate_formal_single_operator_prepared_launch_bundle,
    )

    validated = revalidate_formal_single_operator_prepared_launch_bundle(
        execution_source_path=execution_source_path,
        prepared_launch_bundle_path=prepared_launch_bundle_path,
        materialized_cell_id=materialized_cell_id,
        current_ns=current_ns,
    )
    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    if (
        route.physical_kind != "serving"
        or source.node in _EARLY_NODES
        or source.node in {"e4_screen", "e4_local"}
        or source.stage not in {"E3b", "E1a", "E5", "E6", "E0"}
    ):
        raise FormalSingleOperatorDispatchBlocked(
            "prepared_downstream_serving_route_required"
        )
    entry = validated.entry(cell.cell_id)
    if entry.request_schedule_receipt is None:
        raise FormalSingleOperatorDispatchBlocked(
            "source_owned_request_schedule_missing"
        )
    execution_binding_sha256, subject_sha256 = (
        formal_single_operator_prepared_execution_identities(
            bundle=validated.bundle,
            entry=entry,
        )
    )
    root = Path(private_output_root).resolve(strict=False)
    trusted_content_source = validated.bundle.content_source_binding
    trusted_eagle3 = None
    if trusted_content_source is not None:
        from lightcone_spec.config import load_run_config
        from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
            publish_trusted_single_operator_eagle3_execution_authority,
        )
        from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

        launch = CompileLaunchManifest.load(entry.compile_launch_manifest.absolute_path)
        config = load_run_config(launch.run_config_path)
        if (
            source.stage == "E0"
            and config.model.algorithm == "EAGLE3"
            and config.adaptation is not None
        ):
            authority_path = root / "trusted-eagle3-execution-authority.json"
            publish_trusted_single_operator_eagle3_execution_authority(
                execution_source_path=execution_source_path,
                materialized_cell_id=cell.cell_id,
                compile_launch_manifest_path=(
                    entry.compile_launch_manifest.absolute_path
                ),
                output_path=authority_path,
            )
            trusted_eagle3 = CanonicalJsonProofBinding.bind(authority_path)
    value = FormalSingleOperatorPreparedDownstreamRunPlanInputs(
        schema_version=(1 if trusted_content_source is None else 2),
        kind="formal_single_operator_prepared_downstream_run_plan_inputs",
        execution_source=CanonicalJsonProofBinding.bind(execution_source_path),
        execution_source_sha256=source.sha256,
        prepared_launch_bundle=CanonicalJsonProofBinding.bind(
            prepared_launch_bundle_path
        ),
        prepared_launch_bundle_sha256=validated.bundle.sha256,
        prepared_launch_entry_sha256=entry.sha256,
        materialized_cell_id=cell.cell_id,
        stage=source.stage,  # type: ignore[arg-type]
        materialization=CanonicalJsonProofBinding.bind(
            source.materialization_source.absolute_path
        ),
        materialization_sha256=source.materialization_sha256,
        inventory=validated.bundle.inventory,
        content_verification_receipt=(validated.bundle.content_verification_receipt),
        compile_launch_manifest=entry.compile_launch_manifest,
        request_schedule_receipt=entry.request_schedule_receipt,
        execution_binding_sha256=execution_binding_sha256,
        subject_sha256=subject_sha256,
        private_output_root=str(root),
        content_source_binding=trusted_content_source,
        trusted_eagle3_execution_authority=trusted_eagle3,
        trusted_chronobelief_gpu_parity_proof=(
            entry.trusted_chronobelief_gpu_parity_proof
        ),
    )
    output = root / "formal-single-operator-prepared-downstream-inputs.json"
    publish_canonical_json_no_replace(output, value.to_dict())
    rebound = FormalSingleOperatorPreparedDownstreamRunPlanInputs.from_dict(
        CanonicalJsonProofBinding.bind(output).reopen()
    )
    if rebound != value or rebound.sha256 != value.sha256:
        raise RuntimeError("prepared downstream run inputs changed")
    return value


def revalidate_formal_single_operator_prepared_downstream_run_plan_inputs(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalSingleOperatorPreparedDownstreamRunPlanInputs:
    """Deep-open one prepared descriptor and rejoin its exact bundle entry.

    The descriptor is deliberately small, but it is not a bearer token.  A
    restart must reopen the current execution source, the whole prepared
    launch bundle, and the selected schedule row before the physical child is
    allowed to consume any path from it.
    """

    from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
        formal_single_operator_prepared_execution_identities,
        revalidate_formal_single_operator_prepared_launch_bundle,
    )

    binding = CanonicalJsonProofBinding.bind(path)
    value = FormalSingleOperatorPreparedDownstreamRunPlanInputs.from_dict(
        binding.reopen()
    )
    if value.sha256 != binding.semantic_sha256:
        raise ValueError("prepared downstream plan input digest differs")
    validated = revalidate_formal_single_operator_prepared_launch_bundle(
        execution_source_path=value.execution_source.absolute_path,
        prepared_launch_bundle_path=value.prepared_launch_bundle.absolute_path,
        materialized_cell_id=value.materialized_cell_id,
        current_ns=current_ns,
    )
    entry = validated.entry(value.materialized_cell_id)
    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=value.execution_source.absolute_path,
        materialized_cell_id=value.materialized_cell_id,
    )
    execution_binding_sha256, subject_sha256 = (
        formal_single_operator_prepared_execution_identities(
            bundle=validated.bundle,
            entry=entry,
        )
    )
    materialization = stage_materialization_receipt_from_dict(
        value.materialization.reopen()
    )
    trusted_eagle3_sha256 = None
    if value.trusted_eagle3_execution_authority is not None:
        from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
            load_trusted_single_operator_eagle3_execution_authority,
        )

        trusted_eagle3_sha256 = load_trusted_single_operator_eagle3_execution_authority(
            value.trusted_eagle3_execution_authority.absolute_path
        ).sha256
    trusted_chronobelief_binding = None
    if value.trusted_chronobelief_gpu_parity_proof is not None:
        from lightcone_spec.experiments.formal_single_operator_chronobelief import (
            load_trusted_single_operator_chronobelief_gpu_parity_proof,
        )

        load_trusted_single_operator_chronobelief_gpu_parity_proof(
            value.trusted_chronobelief_gpu_parity_proof.absolute_path
        )
        trusted_chronobelief_binding = CanonicalJsonProofBinding.bind(
            value.trusted_chronobelief_gpu_parity_proof.absolute_path
        )
    if (
        Path(binding.absolute_path)
        != Path(value.private_output_root)
        / "formal-single-operator-prepared-downstream-inputs.json"
        or source.sha256 != value.execution_source_sha256
        or source.stage != value.stage
        or cell.stage != value.stage
        or route.physical_kind != "serving"
        or validated.bundle.sha256 != value.prepared_launch_bundle_sha256
        or entry.sha256 != value.prepared_launch_entry_sha256
        or source.materialization_source != value.materialization
        or source.materialization_sha256 != value.materialization_sha256
        or materialization.sha256 != value.materialization_sha256
        or validated.bundle.inventory != value.inventory
        or validated.bundle.content_verification_receipt
        != value.content_verification_receipt
        or validated.bundle.content_source_binding != value.content_source_binding
        or (
            value.trusted_eagle3_execution_authority is not None
            and value.trusted_eagle3_execution_authority.semantic_sha256
            != trusted_eagle3_sha256
        )
        or (
            value.trusted_chronobelief_gpu_parity_proof is not None
            and value.trusted_chronobelief_gpu_parity_proof
            != trusted_chronobelief_binding
        )
        or entry.trusted_chronobelief_gpu_parity_proof
        != value.trusted_chronobelief_gpu_parity_proof
        or entry.compile_launch_manifest != value.compile_launch_manifest
        or entry.request_schedule_receipt != value.request_schedule_receipt
        or execution_binding_sha256 != value.execution_binding_sha256
        or subject_sha256 != value.subject_sha256
    ):
        raise ValueError("prepared downstream plan input lineage differs")
    return value


__all__ = [
    "FormalSingleOperatorCellDispatch",
    "FormalSingleOperatorDispatchBlocked",
    "FormalSingleOperatorDownstreamRunPlanInputs",
    "FormalSingleOperatorE0CompatibilityDecisionDispatch",
    "FormalSingleOperatorPreparedDownstreamRunPlanInputs",
    "materialize_formal_single_operator_e4_direct_run_plan_inputs",
    "materialize_formal_single_operator_prepared_downstream_run_plan_inputs",
    "revalidate_formal_single_operator_e0_compatibility_decision",
    "revalidate_formal_single_operator_prepared_downstream_run_plan_inputs",
    "route_formal_single_operator_auxiliary",
    "route_formal_single_operator_cell",
    "route_formal_single_operator_materialized_cell",
]
