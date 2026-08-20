"""Source-owned prerequisite launch production for downstream preparation.

``prepare_launch_draft`` deliberately accepts only complete schema-2 launch
manifests.  This module closes the preceding boundary: it discovers model,
backend, and topology identities from the current materialization and its
deeply replayed predecessor/auxiliary evidence, then publishes one canonical
launch per required runtime key.  The public producer accepts paths only; no
scientific scalar or caller-authored ``RunConfig`` crosses this API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import RunConfig, load_run_config
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    e0_compatibility_receipt_from_dict,
    protocol_lock_from_dict,
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_prepared_launch_producer import (
    PreparedLaunchRuntimeKey,
    _cell_backend,
    _cell_topology,
    prerequisite_runtime_key,
)
from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
    route_formal_single_operator_materialized_cell,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorExecutionSource,
    RebuiltFormalSingleOperatorStageCompletion,
    load_formal_single_operator_execution_source,
    rebuild_formal_single_operator_stage_completion,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.stage_materialization import (
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
    StageMaterializationReceipt,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration.runtime import derive_diagnostic_compile_cache_key
from lightcone_spec.runtime.compile_cache import CompileCacheLaunchPlan
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

PrerequisiteAuthorityKind = Literal[
    "upstream_actual_run",
    "preflight_native_qualification",
    "e6_interface_fit",
    "e0_valid_compatibility",
]

_SUPPORTED_STAGES = frozenset({"E4", "E3b", "E1a", "E5", "E6", "E0"})
_SHA256 = frozenset("0123456789abcdef")

FORMAL_SINGLE_OPERATOR_PREREQUISITE_LAUNCH_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_prerequisite_launch_producer",
        "input": (
            "path_only_current_execution_source_and_one_verified_base_"
            "environment_launch"
        ),
        "runtime_identity": (
            "BOUND_content_inventory_complete_PASS_doctor_patched_sglang_"
            "driver_cuda_cache_environment"
        ),
        "stage_authority": {
            "E4": "exact_selected_E4_local_profiler_subject_launch",
            "E3b": "E4_actual_selected_LightCone_DFlash_launch",
            "E1a": (
                "preflight_dspark_tp1_native_qualification_launch_plus_"
                "independent_chronobelief_parity_join_only_for_chrono_winner"
            ),
            "E5": (
                "E3b_DFlash_or_E1a_DSpark_actual_tp1_and_exact_preflight_"
                "backend_distributed_qualification"
            ),
            "E6": "exact_two_model_interface_fit_plan_and_terminal_launch",
            "E0": (
                "schema2_VALID_compatibility_interface_receipt_and_task_"
                "terminal_launch_only"
            ),
        },
        "selection": (
            "all_matching_actuals_must_share_exact_model_identity_then_"
            "lexicographically_first_materialized_cell_id"
        ),
        "output": "canonical_no_replace_schema2_launch_manifest_and_index",
        "forbidden": (
            "caller_RunConfig",
            "caller_scientific_knobs",
            "preflight_Qwen_DFlash_clone_for_foreign_backend_or_model",
            "N_A_E0_launch",
            "unproved_topology",
        ),
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _absolute_directory(label: str, value: str | Path) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or not path.is_dir()
        or path.is_symlink()
    ):
        raise ValueError(f"{label} must be an existing normalized directory")
    return path


def _strict(value: object, expected: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


@dataclass(frozen=True, order=True)
class PrerequisiteLaunchDemand:
    """Materialization-owned runtime demand without a caller model revision."""

    model: str
    backend: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]

    def __post_init__(self) -> None:
        if (
            type(self.model) is not str
            or not self.model
            or self.backend not in {"DFLASH", "DSPARK", "NEXTN", "EAGLE3"}
            or self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}
        ):
            raise ValueError("prerequisite launch demand differs")


@dataclass(frozen=True)
class FormalSingleOperatorPrerequisiteLaunchEntry:
    runtime_key: PreparedLaunchRuntimeKey
    launch_manifest: CanonicalJsonProofBinding
    authority_kind: PrerequisiteAuthorityKind
    authority_sources: tuple[CanonicalJsonProofBinding, ...]
    trusted_chronobelief_gpu_parity_proof: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if type(self.runtime_key) is not PreparedLaunchRuntimeKey:
            raise TypeError("prerequisite entry runtime key differs")
        if type(self.launch_manifest) is not CanonicalJsonProofBinding:
            raise TypeError("prerequisite entry launch is not path-bound")
        launch = CompileLaunchManifest.load(self.launch_manifest.absolute_path)
        if (
            launch.sha256 != self.launch_manifest.semantic_sha256
            or prerequisite_runtime_key(launch) != self.runtime_key
        ):
            raise ValueError("prerequisite entry launch/runtime key differs")
        if self.authority_kind not in {
            "upstream_actual_run",
            "preflight_native_qualification",
            "e6_interface_fit",
            "e0_valid_compatibility",
        }:
            raise ValueError("prerequisite entry authority kind differs")
        if (
            type(self.authority_sources) is not tuple
            or not self.authority_sources
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.authority_sources
            )
            or tuple(row.absolute_path for row in self.authority_sources)
            != tuple(sorted({row.absolute_path for row in self.authority_sources}))
        ):
            raise ValueError("prerequisite entry authority sources are not canonical")
        for source in self.authority_sources:
            if CanonicalJsonProofBinding.bind(source.absolute_path) != source:
                raise ValueError("prerequisite entry authority source changed")
        if self.trusted_chronobelief_gpu_parity_proof is not None and (
            type(self.trusted_chronobelief_gpu_parity_proof)
            is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(
                self.trusted_chronobelief_gpu_parity_proof.absolute_path
            )
            != self.trusted_chronobelief_gpu_parity_proof
        ):
            raise ValueError("prerequisite ChronoBelief proof changed")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_key": self.runtime_key.to_dict(),
            "launch_manifest": self.launch_manifest.to_dict(),
            "authority_kind": self.authority_kind,
            "authority_sources": [row.to_dict() for row in self.authority_sources],
            "trusted_chronobelief_gpu_parity_proof": (
                None
                if self.trusted_chronobelief_gpu_parity_proof is None
                else self.trusted_chronobelief_gpu_parity_proof.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(value, set(cls.__dataclass_fields__), label="prerequisite entry")
        runtime = _strict(
            row.pop("runtime_key"),
            {field.name for field in fields(PreparedLaunchRuntimeKey)},
            label="prerequisite runtime key",
        )
        raw_sources = row.pop("authority_sources")
        if type(raw_sources) is not list:
            raise TypeError("prerequisite authority sources must be an array")
        raw_launch = row.pop("launch_manifest")
        raw_chronobelief = row.pop("trusted_chronobelief_gpu_parity_proof")
        return cls(
            **row,  # type: ignore[arg-type]
            runtime_key=PreparedLaunchRuntimeKey(**runtime),  # type: ignore[arg-type]
            launch_manifest=CanonicalJsonProofBinding.from_dict(raw_launch),
            authority_sources=tuple(
                CanonicalJsonProofBinding.from_dict(item) for item in raw_sources
            ),
            trusted_chronobelief_gpu_parity_proof=(
                None
                if raw_chronobelief is None
                else CanonicalJsonProofBinding.from_dict(raw_chronobelief)
            ),
        )


@dataclass(frozen=True)
class FormalSingleOperatorPrerequisiteLaunchIndex:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_prerequisite_launch_index"]
    protocol_sha256: str
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    protocol_lock_sha256: str
    materialization_sha256: str
    content_source_binding_sha256: str
    inventory_sha256: str
    doctor_sha256: str
    base_environment_launch: CanonicalJsonProofBinding
    entries: tuple[FormalSingleOperatorPrerequisiteLaunchEntry, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_prerequisite_launch_index"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREREQUISITE_LAUNCH_PROTOCOL_SHA256
        ):
            raise ValueError("prerequisite launch index schema differs")
        for label, digest in (
            ("execution source", self.execution_source_sha256),
            ("ProtocolLock", self.protocol_lock_sha256),
            ("materialization", self.materialization_sha256),
            ("content source", self.content_source_binding_sha256),
            ("inventory", self.inventory_sha256),
            ("doctor", self.doctor_sha256),
        ):
            _require_sha256(f"prerequisite index {label}", digest)
        if (
            type(self.execution_source) is not CanonicalJsonProofBinding
            or type(self.base_environment_launch) is not CanonicalJsonProofBinding
        ):
            raise TypeError("prerequisite index root sources are not path-bound")
        source = load_formal_single_operator_execution_source(
            self.execution_source.absolute_path
        )
        if (
            self.execution_source.reopen().get("execution_source_sha256")
            != source.sha256
            or source.sha256 != self.execution_source_sha256
            or source.protocol_lock_sha256 != self.protocol_lock_sha256
            or source.materialization_sha256 != self.materialization_sha256
            or source.content_source_binding is None
            or source.content_source_binding.sha256
            != self.content_source_binding_sha256
        ):
            raise ValueError("prerequisite index execution lineage differs")
        base = CompileLaunchManifest.load(self.base_environment_launch.absolute_path)
        if base.sha256 != self.base_environment_launch.semantic_sha256:
            raise ValueError("prerequisite index base environment launch differs")
        if (
            type(self.entries) is not tuple
            or not self.entries
            or self.entries
            != tuple(sorted(self.entries, key=lambda row: row.runtime_key))
            or len({row.runtime_key for row in self.entries}) != len(self.entries)
        ):
            raise ValueError("prerequisite index entries are not canonical")
        for entry in self.entries:
            launch = CompileLaunchManifest.load(entry.launch_manifest.absolute_path)
            if (
                launch.content_source_binding != source.content_source_binding
                or launch.inventory_sha256 != self.inventory_sha256
                or entry.runtime_key.stage != source.stage
            ):
                raise ValueError("prerequisite entry belongs to another runtime")
        from lightcone_spec.experiments.formal_single_operator_prepared_launch_producer import (
            _trusted_chain_recipe_context,
        )

        requires_chronobelief = (
            source.stage == "E1a"
            and _trusted_chain_recipe_context(source).lightcone_recipe.optimizer
            == "chronobelief"
        )
        if any(
            (entry.trusted_chronobelief_gpu_parity_proof is not None)
            != requires_chronobelief
            for entry in self.entries
        ):
            raise ValueError(
                "prerequisite ChronoBelief proof coverage differs from E2 winner"
            )
        if requires_chronobelief:
            from lightcone_spec.experiments.formal_single_operator_chronobelief import (
                load_trusted_single_operator_chronobelief_gpu_parity_proof,
            )

            for entry in self.entries:
                proof_binding = entry.trusted_chronobelief_gpu_parity_proof
                assert proof_binding is not None
                proof = load_trusted_single_operator_chronobelief_gpu_parity_proof(
                    proof_binding.absolute_path
                )
                if (
                    proof.execution_source != self.execution_source
                    or proof.prerequisite_launch != entry.launch_manifest
                ):
                    raise ValueError("prerequisite ChronoBelief proof lineage differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    @property
    def launch_manifest_paths(self) -> tuple[str, ...]:
        """Exact path tuple accepted by :func:`prepare_launch_draft`."""

        return tuple(row.launch_manifest.absolute_path for row in self.entries)

    @property
    def chronobelief_gpu_parity_proof_paths(self) -> tuple[str, ...]:
        """Exact optional empirical proof paths accepted by launch preparation."""

        return tuple(
            proof.absolute_path
            for row in self.entries
            for proof in (row.trusted_chronobelief_gpu_parity_proof,)
            if proof is not None
        )

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "execution_source": self.execution_source.to_dict(),
            "execution_source_sha256": self.execution_source_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_sha256": self.materialization_sha256,
            "content_source_binding_sha256": self.content_source_binding_sha256,
            "inventory_sha256": self.inventory_sha256,
            "doctor_sha256": self.doctor_sha256,
            "base_environment_launch": self.base_environment_launch.to_dict(),
            "entries": [row.to_dict() for row in self.entries],
        }
        if include_sha256:
            value["index_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            set(cls.__dataclass_fields__) | {"index_sha256"},
            label="prerequisite launch index",
        )
        expected = _require_sha256("prerequisite launch index", row.pop("index_sha256"))
        raw_entries = row.pop("entries")
        if type(raw_entries) is not list:
            raise TypeError("prerequisite launch index entries must be an array")
        raw_execution_source = row.pop("execution_source")
        raw_base_environment = row.pop("base_environment_launch")
        result = cls(
            **row,  # type: ignore[arg-type]
            execution_source=CanonicalJsonProofBinding.from_dict(raw_execution_source),
            base_environment_launch=CanonicalJsonProofBinding.from_dict(
                raw_base_environment
            ),
            entries=tuple(
                FormalSingleOperatorPrerequisiteLaunchEntry.from_dict(item)
                for item in raw_entries
            ),
        )
        if result.sha256 != expected:
            raise ValueError("prerequisite launch index digest differs")
        return result


@dataclass(frozen=True)
class _LaunchAuthority:
    binding: CanonicalJsonProofBinding
    launch: CompileLaunchManifest
    authority_kind: PrerequisiteAuthorityKind
    authority_sources: tuple[CanonicalJsonProofBinding, ...]
    source_stage: str
    source_role: str | None
    source_cell_id: str

    @property
    def config(self) -> RunConfig:
        return load_run_config(self.launch.run_config_path)


def materialized_prerequisite_launch_demands(
    *,
    source: FormalSingleOperatorExecutionSource,
    materialization: StageMaterializationReceipt,
) -> tuple[PrerequisiteLaunchDemand, ...]:
    """Derive the exact current runtime demand set from materialized cells."""

    if source.stage not in _SUPPORTED_STAGES:
        raise ValueError("current stage has no downstream prerequisite producer")
    demands: set[PrerequisiteLaunchDemand] = set()
    for cell in materialization.cells:
        route = route_formal_single_operator_materialized_cell(
            node=source.node,
            phase=source.phase,
            cell=cell,
        )
        if route.physical_kind in {
            "e6_interface_preflight",
            "e0_compatibility_decision",
        }:
            continue
        demands.add(
            PrerequisiteLaunchDemand(
                model=cell.model,
                backend=_cell_backend(cell),
                topology_mode=_cell_topology(cell),
            )
        )
    if not demands:
        if _is_registered_e0_zero_launch_demand(
            source=source,
            materialization=materialization,
        ):
            return ()
        raise ValueError("current materialization has no launch demand")
    return tuple(sorted(demands))


def execution_source_prerequisite_launch_demands(
    source: FormalSingleOperatorExecutionSource,
) -> tuple[PrerequisiteLaunchDemand, ...]:
    """Deep-reopen one execution source and replay its exact launch demands."""

    if type(source) is not FormalSingleOperatorExecutionSource:
        raise TypeError("prerequisite demand replay requires an exact execution source")
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="prerequisite demand execution materialization"
        )
    )
    return materialized_prerequisite_launch_demands(
        source=source,
        materialization=materialization,
    )


def _all_na_compatibility_from_payload(payload: object) -> bool:
    if type(payload) is not dict:
        return False
    raw = payload.get("compatibility")
    try:
        compatibility = e0_compatibility_receipt_from_dict(raw)
    except (TypeError, ValueError):
        return False
    expected = {
        (model, backend, task)
        for model in E0_MODELS
        for backend in E0_BACKENDS
        for task in E0_TASKS
    }
    observed = {(row.model, row.backend, row.task) for row in compatibility.decisions}
    return (
        compatibility.valid_count == 0
        and len(compatibility.decisions) == 108
        and observed == expected
        and all(row.disposition == "N/A" for row in compatibility.decisions)
    )


def _is_exact_e0_na_decision_cell(
    cell: object,
    *,
    bundle_sha256: str,
) -> bool:
    from lightcone_spec.experiments.stage_materialization import MaterializedCell

    if type(cell) is not MaterializedCell:
        return False
    dimensions = dict(cell.dimensions)
    expected_fields = {
        "compatibility_decision_id",
        "deployment_task",
        "disposition",
        "reason_code",
        "interface_sha256",
        "task_native_workload_sha256",
        "compatibility_receipt_sha256",
        "compatibility_evidence_manifest_sha256",
        "e0_compatibility_bundle_sha256",
    }
    for name in (
        "compatibility_decision_id",
        "interface_sha256",
        "task_native_workload_sha256",
        "compatibility_receipt_sha256",
        "compatibility_evidence_manifest_sha256",
        "e0_compatibility_bundle_sha256",
    ):
        try:
            _require_sha256(f"E0 zero-demand {name}", dimensions.get(name))
        except ValueError:
            return False
    return (
        set(dimensions) == expected_fields
        and cell.method_role == "Compatibility"
        and cell.task == "compatibility_decision"
        and cell.publication_policy == "decision_only"
        and cell.recipe_sha256 is None
        and dimensions.get("deployment_task") in E0_TASKS
        and dimensions.get("disposition") == "N/A"
        and type(dimensions.get("reason_code")) is str
        and bool(dimensions["reason_code"])
        and dimensions.get("compatibility_decision_id")
        == content_sha256((cell.model, cell.backend, dimensions["deployment_task"]))
        and dimensions.get("e0_compatibility_bundle_sha256") == bundle_sha256
    )


def _is_registered_e0_zero_launch_demand(
    *,
    source: FormalSingleOperatorExecutionSource,
    materialization: StageMaterializationReceipt,
) -> bool:
    """Recognize only the registered E0 V=0 auxiliary/zero-cell chain.

    This is intentionally narrower than "no routed cells".  The source and
    materialization identities, exact E0 node/rule, and immediate ALL_NA
    decision are replayed before an empty launch set can become authoritative.
    """

    if (
        type(source) is not FormalSingleOperatorExecutionSource
        or type(materialization) is not StageMaterializationReceipt
        or source.node not in {"e0_tuning", "e0_pilot", "e0_final"}
    ):
        return False
    rebound = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="registered E0 zero-demand materialization"
        )
    )
    if (
        rebound != materialization
        or source.stage != "E0"
        or materialization.stage != source.stage
        or materialization.sha256 != source.materialization_sha256
        or materialization.source_decision_sha256
        != source.materialization_source_decision_sha256
        or materialization.upstream_receipt_sha256s
        != source.materialization_upstream_receipt_sha256s
    ):
        raise ValueError("registered E0 zero-demand materialization identity differs")

    if source.node == "e0_tuning":
        expected = {
            (model, backend, task)
            for model in E0_MODELS
            for backend in E0_BACKENDS
            for task in E0_TASKS
        }
        observed = {
            (
                cell.model,
                cell.backend,
                dict(cell.dimensions).get("deployment_task"),
            )
            for cell in materialization.cells
        }
        return (
            source.phase == "tuning"
            and materialization.materialization_rule
            == "108_compatibility_decisions_plus_239_rows_per_valid"
            and materialization.expected_cell_count == 108
            and observed == expected
            and all(
                _is_exact_e0_na_decision_cell(
                    cell,
                    bundle_sha256=materialization.source_decision_sha256,
                )
                and route_formal_single_operator_materialized_cell(
                    node=source.node,
                    phase=source.phase,
                    cell=cell,
                ).physical_kind
                == "e0_compatibility_decision"
                for cell in materialization.cells
            )
        )

    predecessor_binding = source.predecessor_completion_source
    if predecessor_binding is None:
        raise ValueError("registered E0 zero-demand source lacks predecessor")
    predecessor = rebuild_formal_single_operator_stage_completion(
        predecessor_binding.absolute_path
    )
    if (
        predecessor.artifact.sha256 != source.predecessor_completion_sha256
        or predecessor.decision.sha256 != source.predecessor_decision_sha256
        or predecessor.decision.next_materialization_source_decision_sha256
        != materialization.source_decision_sha256
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != materialization.upstream_receipt_sha256s
        or materialization.upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("registered E0 zero-demand predecessor identity differs")
    payload = predecessor.decision.payload
    if not _all_na_compatibility_from_payload(payload):
        return False
    if source.node == "e0_pilot":
        return (
            source.phase == "excluded_pilot"
            and predecessor.artifact.node == "e0_tuning"
            and predecessor.decision.decision_kind == "e0_tuning_actual_reduced"
            and payload.get("status") == "ALL_NA"
            and payload.get("valid_count") == 0
            and materialization.materialization_rule
            == "valid_x_8_roles_x_2_loads_x_4_excluded_pilots"
            and materialization.expected_cell_count == 0
            and not materialization.cells
        )
    return (
        source.phase == "final"
        and predecessor.artifact.node == "e0_pilot"
        and predecessor.decision.decision_kind == "e0_pilot_all_na"
        and payload.get("status") == "ALL_NA"
        and payload.get("selected_final_blocks") == 0
        and materialization.materialization_rule
        == "valid_x_8_roles_x_2_loads_x_powered_final_blocks"
        and materialization.expected_cell_count == 0
        and not materialization.cells
    )


def _completion_chain(
    source: FormalSingleOperatorExecutionSource,
) -> tuple[RebuiltFormalSingleOperatorStageCompletion, ...]:
    if source.predecessor_completion_source is None:
        return ()
    current = rebuild_formal_single_operator_stage_completion(
        source.predecessor_completion_source.absolute_path
    )
    rows = []
    while current is not None:
        rows.append(current)
        current = current.predecessor
    return tuple(rows)


def _actual_launch_authorities(
    chain: tuple[RebuiltFormalSingleOperatorStageCompletion, ...],
    *,
    repository_root: Path,
) -> tuple[_LaunchAuthority, ...]:
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRunPlan,
        _load_formal_single_operator_trusted_run_plan,
    )
    from lightcone_spec.runtime.formal_single_operator import (
        revalidate_formal_single_operator_run_manifest,
    )

    rows = []
    for completion in chain:
        for actual in completion.artifact.actual_results:
            if actual.validator_kind not in {
                "formal_single_operator_run_manifest_revalidator",
                "formal_single_operator_onlinespec_run_manifest_revalidator",
            }:
                continue
            manifest_binding = CanonicalJsonProofBinding.bind(
                actual.source.absolute_path
            )
            manifest = revalidate_formal_single_operator_run_manifest(
                repository_root=repository_root,
                manifest_path=manifest_binding.absolute_path,
            )
            artifacts = {row.name: row for row in manifest.artifacts}
            plan_artifact = artifacts.get("run_plan")
            if plan_artifact is None or plan_artifact.status != "PRESENT":
                raise ValueError("upstream actual lacks its run plan")
            plan_path = Path(manifest.run_directory) / plan_artifact.relative_path
            plan_binding = CanonicalJsonProofBinding.bind(plan_path)
            plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
            replayed, launch, _schedule = _load_formal_single_operator_trusted_run_plan(
                plan_binding.absolute_path
            )
            if (
                replayed != plan
                or plan.sha256 != plan_binding.semantic_sha256
                or launch.sha256 != plan.launch_manifest.semantic_sha256
                or manifest.launch_manifest_sha256 != launch.sha256
            ):
                raise ValueError("upstream actual launch replay differs")
            rows.append(
                _LaunchAuthority(
                    binding=plan.launch_manifest,
                    launch=launch,
                    authority_kind="upstream_actual_run",
                    authority_sources=tuple(
                        sorted(
                            (manifest_binding, plan_binding, plan.launch_manifest),
                            key=lambda row: row.absolute_path,
                        )
                    ),
                    source_stage=manifest.stage,
                    source_role=manifest.role,
                    source_cell_id=manifest.cell_id,
                )
            )
    return tuple(rows)


def _preflight_qualification_authorities(
    chain: tuple[RebuiltFormalSingleOperatorStageCompletion, ...],
) -> dict[str, _LaunchAuthority]:
    from lightcone_spec.experiments.formal_preflight_coverage import (
        FormalPreflightStageCoverageProofArtifact,
        revalidate_formal_preflight_stage_coverage_proof_artifact,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorPreflightActualReceipt,
    )
    from lightcone_spec.runtime.native_qualification_runner import (
        NativeRuntimeQualificationAssignment,
        NativeRuntimeQualificationResultPointer,
    )

    matches = tuple(row for row in chain if row.artifact.node == "preflight")
    if len(matches) != 1:
        raise ValueError("prerequisite producer lacks one exact preflight completion")
    actual_sources = {
        row.source.absolute_path: row.source
        for row in matches[0].artifact.actual_results
    }
    if len(actual_sources) != 1:
        raise ValueError("prerequisite preflight actual source is not unique")
    actual_source = next(iter(actual_sources.values()))
    raw_actual = actual_source.reopen(label="prerequisite preflight actual")
    if (
        type(raw_actual) is dict
        and raw_actual.get("kind")
        == "formal_single_operator_exact_ten_preflight_completion"
    ):
        from lightcone_spec.experiments.formal_preflight_inputs import (
            FormalPreflightExecutionInputs,
            FormalSingleOperatorPreflightCompletion,
            revalidate_formal_single_operator_preflight_completion,
        )
        from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
            TRUSTED_PREFLIGHT_QUALIFICATION_SUITES,
            load_formal_single_operator_preflight_qualification_plan,
            load_formal_single_operator_preflight_qualification_plan_index,
            revalidate_formal_single_operator_preflight_qualification_result,
        )

        serialized = FormalSingleOperatorPreflightCompletion.from_dict(raw_actual)
        completion = revalidate_formal_single_operator_preflight_completion(
            actual_source.absolute_path,
            current_ns=serialized.finished_ns,
        )
        inputs = FormalPreflightExecutionInputs.from_dict(
            completion.execution_inputs.reopen()
        )
        if inputs.schema_version != 4 or inputs.qualification_plan_index is None:
            raise ValueError("trusted preflight qualification plan index is absent")
        index = load_formal_single_operator_preflight_qualification_plan_index(
            inputs.qualification_plan_index.absolute_path
        )
        by_suite = {}
        for plan_binding in index.plans:
            plan = load_formal_single_operator_preflight_qualification_plan(
                plan_binding.absolute_path
            )
            result_pointer = CanonicalJsonProofBinding.bind(plan.result_path)
            result = revalidate_formal_single_operator_preflight_qualification_result(
                result_pointer.absolute_path
            )
            assignment = NativeRuntimeQualificationAssignment.load(
                result.assignment.absolute_path
            )
            launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
            if (
                result.status != "COMPLETE"
                or assignment.schema_version != 2
                or assignment.suite_id != plan.suite_id
                or assignment.launch_manifest != plan.launch_manifest
            ):
                raise ValueError("trusted preflight qualification result differs")
            by_suite[plan.suite_id] = _LaunchAuthority(
                binding=plan.launch_manifest,
                launch=launch,
                authority_kind="preflight_native_qualification",
                authority_sources=tuple(
                    sorted(
                        (
                            CanonicalJsonProofBinding.bind(actual_source.absolute_path),
                            completion.execution_inputs,
                            inputs.qualification_plan_index,
                            plan_binding,
                            plan.dispatch_authority,
                            result_pointer,
                            result.assignment,
                            result.empirical_proof,
                            plan.launch_manifest,
                        ),
                        key=lambda row: row.absolute_path,
                    )
                ),
                source_stage="preflight",
                source_role=None,
                source_cell_id=assignment.sha256,
            )
        required = {
            "dspark_tp1",
            "dspark_tp2",
            "dspark_dp2",
            "tp2_dp1",
            "tp1_dp2",
        }
        if set(by_suite) != set(
            TRUSTED_PREFLIGHT_QUALIFICATION_SUITES
        ) or not required.issubset(by_suite):
            raise ValueError("trusted preflight qualification coverage differs")
        return {suite_id: by_suite[suite_id] for suite_id in sorted(required)}
    actual = matches[0].artifact.actual_results[0]
    receipt = FormalSingleOperatorPreflightActualReceipt.from_dict(
        actual.source.reopen(label="prerequisite preflight actual")
    )
    coverage_binding = CanonicalJsonProofBinding.bind(
        receipt.final_evidence_source.absolute_path
    )
    coverage = FormalPreflightStageCoverageProofArtifact.from_dict(
        coverage_binding.reopen()
    )
    revalidate_formal_preflight_stage_coverage_proof_artifact(
        coverage_binding.absolute_path,
        now_ns=receipt.verified_ns,
    )
    result: dict[str, _LaunchAuthority] = {}
    for proof in coverage.qualification_proof_sources:
        if proof.suite_id not in {
            "dspark_tp1",
            "dspark_tp2",
            "dspark_dp2",
            "tp2_dp1",
            "tp1_dp2",
        }:
            continue
        pointer = NativeRuntimeQualificationResultPointer.load(
            proof.result_pointer.absolute_path
        )
        assignment = NativeRuntimeQualificationAssignment.load(
            pointer.assignment.absolute_path
        )
        launch = CompileLaunchManifest.load(assignment.launch_manifest.absolute_path)
        if assignment.suite_id != proof.suite_id:
            raise ValueError("preflight qualification suite/assignment differs")
        result[proof.suite_id] = _LaunchAuthority(
            binding=assignment.launch_manifest,
            launch=launch,
            authority_kind="preflight_native_qualification",
            authority_sources=tuple(
                sorted(
                    (
                        coverage_binding,
                        proof.result_pointer,
                        proof.proof_artifact,
                        pointer.assignment,
                        assignment.launch_manifest,
                    ),
                    key=lambda row: row.absolute_path,
                )
            ),
            source_stage="preflight",
            source_role=None,
            source_cell_id=assignment.sha256,
        )
    if set(result) != {
        "dspark_tp1",
        "dspark_tp2",
        "dspark_dp2",
        "tp2_dp1",
        "tp1_dp2",
    }:
        raise ValueError("preflight prerequisite qualification coverage differs")
    return result


def trusted_preflight_qualification_launch_paths_from_completion(
    completion_path: str | Path,
) -> dict[str, str]:
    """Recover the fresh exact-ten backend/topology launch authorities.

    This is the source-owned bootstrap seam: callers provide only a current
    stage-completion path.  The function walks its immutable predecessor DAG,
    deep-reopens the exact-ten qualification evidence, and returns the
    code-named launch paths without accepting a launch path or digest from the
    operator.
    """

    current = rebuild_formal_single_operator_stage_completion(completion_path)
    chain = []
    while current is not None:
        chain.append(current)
        current = current.predecessor
    authorities = _preflight_qualification_authorities(tuple(chain))
    expected = {
        "dspark_dp2",
        "dspark_tp1",
        "dspark_tp2",
        "tp1_dp2",
        "tp2_dp1",
    }
    if set(authorities) != expected:
        raise ValueError("trusted preflight bootstrap launch coverage differs")
    return {
        suite: authorities[suite].binding.absolute_path for suite in sorted(expected)
    }


def _e6_authorities(
    *,
    source: FormalSingleOperatorExecutionSource,
    protocol_lock: object,
) -> dict[str, _LaunchAuthority]:
    from lightcone_spec.experiments.formal_protocol import ProtocolLock
    from lightcone_spec.experiments.formal_single_operator_e6_interface import (
        E6_MODELS,
        revalidate_formal_single_operator_e6_interface_fit_bundle_value,
        revalidate_formal_single_operator_e6_interface_fit_plan,
        revalidate_formal_single_operator_e6_interface_fit_terminal,
        terminal_for_model,
    )

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E6 prerequisite producer requires exact ProtocolLock")
    auxiliary = source.auxiliary_source_binding("e6_interface_fit")
    bundle = revalidate_formal_single_operator_e6_interface_fit_bundle_value(
        auxiliary.reopen(label="prerequisite E6 interface bundle"),
        protocol_lock=protocol_lock,
    )
    result = {}
    for model in E6_MODELS:
        terminal_binding = terminal_for_model(bundle, model)
        terminal = revalidate_formal_single_operator_e6_interface_fit_terminal(
            terminal_binding.absolute_path
        )
        plan = revalidate_formal_single_operator_e6_interface_fit_plan(
            terminal.plan.absolute_path
        )
        launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
        result[model] = _LaunchAuthority(
            binding=plan.launch_manifest,
            launch=launch,
            authority_kind="e6_interface_fit",
            authority_sources=tuple(
                sorted(
                    (
                        CanonicalJsonProofBinding.bind(auxiliary.absolute_path),
                        terminal_binding,
                        terminal.plan,
                        plan.launch_manifest,
                    ),
                    key=lambda row: row.absolute_path,
                )
            ),
            source_stage="E6",
            source_role=None,
            source_cell_id=terminal.sha256,
        )
    return result


def _e0_authorities(
    *, source: FormalSingleOperatorExecutionSource
) -> dict[tuple[str, str], _LaunchAuthority]:
    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        load_e0_prepared_model_backend_interface_receipt,
        revalidate_trusted_e0_compatibility_bundle_value,
    )

    auxiliary = source.auxiliary_source_binding("e0_compatibility")
    publication = revalidate_trusted_e0_compatibility_bundle_value(
        auxiliary.reopen(label="prerequisite E0 compatibility bundle")
    )
    decisions = {
        (row.model, row.backend, row.task): row
        for row in publication.compatibility.decisions
    }
    result = {}
    for binding in publication.evidence_manifest.interface_receipts:
        receipt = load_e0_prepared_model_backend_interface_receipt(
            binding.absolute_path
        )
        valid = tuple(
            row
            for key, row in decisions.items()
            if key[:2] == (receipt.model, receipt.backend)
            and row.disposition == "VALID"
        )
        if not valid:
            # The interface receipt records the one generic model/backend
            # smoke launch even when every task-specific probe is N/A.  Such
            # a pair must not enter the serving prerequisite index, but its
            # physical interface evidence remains part of the immutable
            # 12/36/108 compatibility publication.
            continue
        if receipt.support_status != "READY" or receipt.compile_launch_manifest is None:
            raise ValueError("VALID E0 pair lacks its exact interface launch")
        launch = CompileLaunchManifest.load(
            receipt.compile_launch_manifest.absolute_path
        )
        terminal_bindings = tuple(
            terminal_binding
            for terminal_binding in publication.evidence_manifest.probe_terminals
            if (
                (terminal := terminal_binding.reopen()).get("model") == receipt.model
                and terminal.get("backend") == receipt.backend
                and decisions[
                    (receipt.model, receipt.backend, terminal["task"])
                ].disposition
                == "VALID"
            )
        )
        result[(receipt.model, receipt.backend)] = _LaunchAuthority(
            binding=receipt.compile_launch_manifest,
            launch=launch,
            authority_kind="e0_valid_compatibility",
            authority_sources=tuple(
                sorted(
                    (
                        CanonicalJsonProofBinding.bind(auxiliary.absolute_path),
                        binding,
                        receipt.compile_launch_manifest,
                        *terminal_bindings,
                    ),
                    key=lambda row: row.absolute_path,
                )
            ),
            source_stage="E0",
            source_role=None,
            source_cell_id=receipt.sha256,
        )
    return result


def _e4_profiler_authority(
    *,
    execution_source_path: str | Path,
    source: FormalSingleOperatorExecutionSource,
    repository_root: Path,
) -> _LaunchAuthority:
    """Derive the profiler prerequisite from its exact selected E4-local run."""

    from lightcone_spec.experiments.formal_single_operator_profiler_subject_producer import (
        derive_formal_single_operator_profiler_subject_requirement,
    )

    requirement = derive_formal_single_operator_profiler_subject_requirement(
        execution_source_path=execution_source_path,
        repository_root=repository_root,
    )
    if load_formal_single_operator_execution_source(execution_source_path) != source:
        raise RuntimeError("E4 profiler execution source changed")
    launch = CompileLaunchManifest.load(
        requirement.selected_compile_launch_manifest.absolute_path
    )
    sources = tuple(
        sorted(
            (
                requirement.selected_compile_launch_manifest,
                requirement.selected_full_run_config,
                requirement.code_owned_profiler_subject_workload,
                requirement.code_owned_request_schedule,
            ),
            key=lambda row: row.absolute_path,
        )
    )
    return _LaunchAuthority(
        binding=requirement.selected_compile_launch_manifest,
        launch=launch,
        authority_kind="upstream_actual_run",
        authority_sources=sources,
        source_stage="E4",
        source_role="LightCone",
        source_cell_id=requirement.source_headline_cell_id,
    )


def _actual_authority(
    actuals: tuple[_LaunchAuthority, ...],
    *,
    stage: str,
    role: str,
    demand: PrerequisiteLaunchDemand,
) -> _LaunchAuthority:
    matches = tuple(
        row
        for row in actuals
        if row.source_stage == stage
        and row.source_role == role
        and row.config.model.target == demand.model
        and row.config.model.algorithm == demand.backend
        and row.config.runtime.topology_mode == demand.topology_mode
    )
    identities = {
        (
            row.config.model.target,
            row.config.model.target_revision,
            row.config.model.drafter,
            row.config.model.drafter_revision,
            row.launch.tokenizer_model_id,
            row.launch.tokenizer_revision,
        )
        for row in matches
    }
    if not matches or len(identities) != 1:
        raise ValueError("upstream actual runtime identity is missing or mixed")
    return min(matches, key=lambda row: row.source_cell_id)


def _qualification_authority(
    qualifications: dict[str, _LaunchAuthority],
    *,
    suite: str,
    demand: PrerequisiteLaunchDemand,
) -> _LaunchAuthority:
    authority = qualifications[suite]
    config = authority.config
    if (
        config.model.target != demand.model
        or config.model.algorithm != demand.backend
        or config.runtime.topology_mode != demand.topology_mode
    ):
        raise ValueError("preflight qualification cannot be relabelled for demand")
    return authority


def _select_authority(
    *,
    source: FormalSingleOperatorExecutionSource,
    demand: PrerequisiteLaunchDemand,
    actuals: tuple[_LaunchAuthority, ...],
    qualifications: dict[str, _LaunchAuthority],
    e4: _LaunchAuthority | None,
    e6: dict[str, _LaunchAuthority],
    e0: dict[tuple[str, str], _LaunchAuthority],
) -> _LaunchAuthority:
    if source.stage == "E4":
        if e4 is None:
            raise ValueError("E4 profiler lacks its selected subject launch")
        authority = e4
    elif source.stage == "E3b":
        return _actual_authority(
            actuals,
            stage="E4",
            role="LightCone",
            demand=demand,
        )
    elif source.stage == "E1a":
        return _qualification_authority(
            qualifications,
            suite="dspark_tp1",
            demand=demand,
        )
    elif source.stage == "E5":
        if demand.topology_mode == "tp1_dp1":
            return _actual_authority(
                actuals,
                stage="E3b" if demand.backend == "DFLASH" else "E1a",
                role="LightCone",
                demand=demand,
            )
        suite = {
            ("DFLASH", "tp2_dp1"): "tp2_dp1",
            ("DFLASH", "tp1_dp2"): "tp1_dp2",
            ("DSPARK", "tp2_dp1"): "dspark_tp2",
            ("DSPARK", "tp1_dp2"): "dspark_dp2",
        }.get((demand.backend, demand.topology_mode))
        if suite is None:
            raise ValueError("E5 backend/topology lacks registered authority")
        return _qualification_authority(
            qualifications,
            suite=suite,
            demand=demand,
        )
    elif source.stage == "E6":
        try:
            authority = e6[demand.model]
        except KeyError as error:
            raise ValueError("E6 model lacks exact interface-fit launch") from error
    elif source.stage == "E0":
        try:
            authority = e0[(demand.model, demand.backend)]
        except KeyError as error:
            raise ValueError("E0 VALID pair lacks exact interface launch") from error
    else:
        raise AssertionError("unsupported prerequisite stage")
    config = authority.config
    if (
        config.model.target != demand.model
        or config.model.algorithm != demand.backend
        or config.runtime.topology_mode != demand.topology_mode
    ):
        raise ValueError("interface authority runtime differs from materialization")
    return authority


def _runtime_inputs(
    source: FormalSingleOperatorExecutionSource,
) -> tuple[
    GpuInventory,
    CanonicalJsonProofBinding,
    dict[str, object],
    CanonicalJsonProofBinding,
]:
    if source.content_source_binding is None:
        raise ValueError("prerequisite producer requires trusted content source")
    content = source.content_source_binding.reopen()
    observations = content.runtime_observations
    if content.runtime_binding_status != "BOUND" or observations is None:
        raise ValueError("prerequisite producer lacks BOUND runtime observations")
    inventory_binding = CanonicalJsonProofBinding.bind(
        observations.inventory.absolute_path
    )
    doctor_binding = CanonicalJsonProofBinding.bind(observations.doctor.absolute_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    doctor = doctor_binding.reopen()
    if type(doctor) is not dict:
        raise TypeError("prerequisite doctor report is not an object")
    return inventory, inventory_binding, doctor, doctor_binding


def _validate_base_environment(
    *,
    source: FormalSingleOperatorExecutionSource,
    base: CompileLaunchManifest,
    inventory: GpuInventory,
    doctor: dict[str, object],
) -> CompileCacheLaunchPlan:
    if (
        base.schema_version != 2
        or base.content_source_binding != source.content_source_binding
        or base.inventory_sha256 != inventory.sha256
    ):
        raise ValueError("base environment launch belongs to another runtime")
    config = load_run_config(base.run_config_path)
    lock = ModelLock(
        schema_version=2,
        models=tuple(
            sorted(
                (
                    LockedModel(config.model.target, config.model.target_revision),
                    LockedModel(config.model.drafter, config.model.drafter_revision),
                ),
                key=lambda row: row.model_id,
            )
        ),
    )
    expected_key = derive_diagnostic_compile_cache_key(
        doctor_report=doctor,
        model_lock=lock,
        config=config,
        gpu_uuid=base.gpu_uuids[0],
    )
    plan = CompileCacheLaunchPlan.load(base.compile_cache_plan_path)
    if expected_key != plan.key:
        raise ValueError("base environment launch differs from PASS doctor")
    return plan


def _validate_shared_environment(
    *,
    base: CompileLaunchManifest,
    base_plan: CompileCacheLaunchPlan,
    authority: _LaunchAuthority,
) -> None:
    launch = authority.launch
    plan = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
    base_key = base_plan.key
    key = plan.key
    environment_fields = (
        "patched_sglang_tree",
        "patch_manifest_sha256",
        "patch_sha256",
        "source_sha256",
        "python_version",
        "torch_version",
        "triton_version",
        "cuda_version",
        "driver_version",
        "sm_architecture",
        "gpu_model",
        "allocator",
        "build_flags",
    )
    if (
        any(
            getattr(key, field) != getattr(base_key, field)
            for field in environment_fields
        )
        or plan.cache_root != base_plan.cache_root
        or launch.patched_sglang_checkout != base.patched_sglang_checkout
        or launch.patched_sglang_commit != base.patched_sglang_commit
        or launch.patched_sglang_tree != base.patched_sglang_tree
        or launch.content_source_binding != base.content_source_binding
        or launch.inventory_sha256 != base.inventory_sha256
        or launch.path_entries != base.path_entries
        or launch.library_path_entries != base.library_path_entries
        or launch.cuda_home != base.cuda_home
    ):
        raise ValueError("prerequisite authority uses another environment/cache")


def _derive_stage_launch(
    *,
    source: FormalSingleOperatorExecutionSource,
    materialization: StageMaterializationReceipt,
    demand: PrerequisiteLaunchDemand,
    authority: _LaunchAuthority,
    output_directory: Path,
) -> CanonicalJsonProofBinding:
    launch = authority.launch
    derived = replace(
        launch,
        formal_stage=source.stage,
        physical_assignment_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "prerequisite_launch_physical_assignment",
                "execution_source_sha256": source.sha256,
                "demand": {
                    "model": demand.model,
                    "backend": demand.backend,
                    "topology_mode": demand.topology_mode,
                },
                "inventory_sha256": launch.inventory_sha256,
                "gpu_uuids": list(launch.gpu_uuids),
            }
        ),
        experiment_budget_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "prerequisite_launch_budget_subject",
                "materialization_sha256": materialization.sha256,
                "runtime_demand": {
                    "model": demand.model,
                    "backend": demand.backend,
                    "topology_mode": demand.topology_mode,
                },
            }
        ),
        budget_materialization_authority_sha256=source.sha256,
    )
    derived.validate(reopen_inputs=True)
    name = f"{demand.backend.lower()}-{demand.topology_mode}-{derived.sha256[:16]}.json"
    path = output_directory / name
    derived.write(path)
    return CanonicalJsonProofBinding.bind(path, semantic_sha256=derived.sha256)


def publish_formal_single_operator_prerequisite_launch_index(
    *,
    execution_source_path: str | Path,
    base_environment_launch_manifest_path: str | Path,
    repository_root: str | Path,
    private_output_root: str | Path,
) -> FormalSingleOperatorPrerequisiteLaunchIndex:
    """Publish the current stage's launch prerequisites from path-only inputs."""

    root = _absolute_directory("prerequisite private output root", private_output_root)
    repository = _absolute_directory("prerequisite repository root", repository_root)
    index_path = root / "prerequisite-launch-index.json"
    launch_root = root / "prerequisite-launches"
    if os.path.lexists(index_path) or os.path.lexists(launch_root):
        raise FileExistsError("prerequisite launch publication already exists")
    execution_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source = load_formal_single_operator_execution_source(
        execution_binding.absolute_path
    )
    if source.stage not in _SUPPORTED_STAGES:
        raise ValueError("current stage has no prerequisite launch producer")
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="prerequisite producer execution materialization"
        )
    )
    protocol_lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(
            label="prerequisite producer execution ProtocolLock"
        )
    )
    inventory, _inventory_binding, doctor, doctor_binding = _runtime_inputs(source)
    base = CompileLaunchManifest.load(base_environment_launch_manifest_path)
    base_binding = CanonicalJsonProofBinding.bind(
        base_environment_launch_manifest_path,
        semantic_sha256=base.sha256,
    )
    base_plan = _validate_base_environment(
        source=source,
        base=base,
        inventory=inventory,
        doctor=doctor,
    )
    chain = _completion_chain(source)
    actuals = _actual_launch_authorities(chain, repository_root=repository)
    qualifications = (
        _preflight_qualification_authorities(chain)
        if source.stage in {"E1a", "E5"}
        else {}
    )
    e4 = (
        _e4_profiler_authority(
            execution_source_path=execution_binding.absolute_path,
            source=source,
            repository_root=repository,
        )
        if source.stage == "E4"
        else None
    )
    e6 = (
        _e6_authorities(source=source, protocol_lock=protocol_lock)
        if source.stage == "E6"
        else {}
    )
    e0 = _e0_authorities(source=source) if source.stage == "E0" else {}
    demands = materialized_prerequisite_launch_demands(
        source=source,
        materialization=materialization,
    )
    from lightcone_spec.experiments.formal_single_operator_prepared_launch_producer import (
        _trusted_chain_recipe_context,
    )

    requires_chronobelief = (
        source.stage == "E1a"
        and _trusted_chain_recipe_context(source).lightcone_recipe.optimizer
        == "chronobelief"
    )
    launch_root.mkdir(mode=0o700)
    entries = []
    for demand in demands:
        authority = _select_authority(
            source=source,
            demand=demand,
            actuals=actuals,
            qualifications=qualifications,
            e4=e4,
            e6=e6,
            e0=e0,
        )
        _validate_shared_environment(
            base=base,
            base_plan=base_plan,
            authority=authority,
        )
        if source.stage in {"E4", "E6", "E0"}:
            launch_binding = authority.binding
        else:
            launch_binding = _derive_stage_launch(
                source=source,
                materialization=materialization,
                demand=demand,
                authority=authority,
                output_directory=launch_root,
            )
        launch = CompileLaunchManifest.load(launch_binding.absolute_path)
        chronobelief_binding = None
        authority_sources = authority.authority_sources
        if requires_chronobelief:
            from lightcone_spec.experiments.formal_single_operator_chronobelief import (
                publish_trusted_single_operator_chronobelief_gpu_parity_proof,
            )

            proof_path = launch_root / (
                f"chronobelief-parity-{launch.sha256[:16]}.json"
            )
            publish_trusted_single_operator_chronobelief_gpu_parity_proof(
                execution_source_path=execution_binding.absolute_path,
                prerequisite_launch_path=launch_binding.absolute_path,
                output_path=proof_path,
            )
            chronobelief_binding = CanonicalJsonProofBinding.bind(proof_path)
            authority_sources = tuple(
                sorted(
                    (*authority_sources, chronobelief_binding),
                    key=lambda row: row.absolute_path,
                )
            )
        entries.append(
            FormalSingleOperatorPrerequisiteLaunchEntry(
                runtime_key=prerequisite_runtime_key(launch),
                launch_manifest=launch_binding,
                authority_kind=authority.authority_kind,
                authority_sources=authority_sources,
                trusted_chronobelief_gpu_parity_proof=chronobelief_binding,
            )
        )
    index = FormalSingleOperatorPrerequisiteLaunchIndex(
        schema_version=1,
        kind="formal_single_operator_prerequisite_launch_index",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREREQUISITE_LAUNCH_PROTOCOL_SHA256,
        execution_source=execution_binding,
        execution_source_sha256=source.sha256,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_sha256=materialization.sha256,
        content_source_binding_sha256=source.content_source_binding.sha256,
        inventory_sha256=inventory.sha256,
        doctor_sha256=doctor_binding.semantic_sha256,
        base_environment_launch=base_binding,
        entries=tuple(sorted(entries, key=lambda row: row.runtime_key)),
    )
    expected_demands = {(row.model, row.backend, row.topology_mode) for row in demands}
    actual_demands = {
        (
            row.runtime_key.target_model_id,
            row.runtime_key.backend,
            row.runtime_key.topology_mode,
        )
        for row in index.entries
    }
    if actual_demands != expected_demands:
        raise ValueError("prerequisite launch index does not cover materialization")
    publish_canonical_json_no_replace(index_path, index.to_dict())
    rebound = load_formal_single_operator_prerequisite_launch_index(index_path)
    if rebound != index:
        raise RuntimeError("prerequisite launch index changed during publication")
    return rebound


def load_formal_single_operator_prerequisite_launch_index(
    path: str | Path,
) -> FormalSingleOperatorPrerequisiteLaunchIndex:
    binding = CanonicalJsonProofBinding.bind(path)
    index = FormalSingleOperatorPrerequisiteLaunchIndex.from_dict(binding.reopen())
    if binding.semantic_sha256 != content_sha256(index.to_dict()):
        raise ValueError("prerequisite launch index binding differs")
    source = load_formal_single_operator_execution_source(
        index.execution_source.absolute_path
    )
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="prerequisite index execution materialization"
        )
    )
    expected = {
        (row.model, row.backend, row.topology_mode)
        for row in materialized_prerequisite_launch_demands(
            source=source,
            materialization=materialization,
        )
    }
    actual = {
        (
            row.runtime_key.target_model_id,
            row.runtime_key.backend,
            row.runtime_key.topology_mode,
        )
        for row in index.entries
    }
    if expected != actual:
        raise ValueError("prerequisite launch index/materialization coverage differs")
    return index


__all__ = [
    "FORMAL_SINGLE_OPERATOR_PREREQUISITE_LAUNCH_PROTOCOL_SHA256",
    "FormalSingleOperatorPrerequisiteLaunchEntry",
    "FormalSingleOperatorPrerequisiteLaunchIndex",
    "PrerequisiteLaunchDemand",
    "execution_source_prerequisite_launch_demands",
    "load_formal_single_operator_prerequisite_launch_index",
    "materialized_prerequisite_launch_demands",
    "publish_formal_single_operator_prerequisite_launch_index",
    "trusted_preflight_qualification_launch_paths_from_completion",
]
