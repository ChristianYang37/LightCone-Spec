"""Fail-closed CLI for the Static/TTS/L0 speed study."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import pyarrow as pa
import pyarrow.parquet as pq

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)
from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.doctor import format_doctor
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.capacity_authority import bind_capacity_authority
from lightcone_spec.experiments.data import (
    DFLASH_MODEL_CONTEXT_LIMIT,
    LongContinuationAdapter,
    load_natural_prompts,
    sample_set_sha256,
)
from lightcone_spec.experiments.evidence import (
    GpuEvidenceAttestation,
    GreedyTargetReference,
    evidence_files_sha256,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuDispatchPlan,
    GpuDispatchPlanningContext,
    GpuInventory,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.industrial_analysis import (
    _DISABLED_SESSION_RUN_FIELDS,
    BoundArtifact,
    IndustrialBlockEvidence,
    IndustrialCellEvidence,
    RawEvidenceAliasManifest,
    _validate_allocation_free_performance,
    _validate_disabled_session_run_fields,
    raw_evidence_alias_manifest_from_dict,
    reduce_confirmation_family_power,
    reduce_e2_stage_from_raw,
    reduce_evidence_alias,
    reduce_industrial_schema_v3,
)
from lightcone_spec.experiments.inventory import (
    build_serial_interference_envelope,
    collect_gpu_inventory,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_METHODS,
    ONLINE_SPEC_STUDY_METHODS,
    ONLINE_SPEC_TUNING_STAGES,
    OnlineSpecCandidate,
    OnlineSpecGpuAttestation,
    OnlineSpecManifest,
    OnlineSpecSelection,
    OnlineSpecTuningMeasurement,
    compare_onlinespec,
    onlinespec_candidates,
    onlinespec_tuning_stage,
    reduce_onlinespec_tuning_stage,
    select_onlinespec,
    select_onlinespec_heldout_anchor,
    verify_onlinespec_source_checkout,
)
from lightcone_spec.experiments.planning import (
    BudgetDispositionStatus,
    BudgetObservationReceipt,
    BudgetPlan,
    ConfirmationFamilyPowerReductionArtifact,
    DispositionStatus,
    EvidenceDependenceMap,
    FamilyActivationArtifact,
    budget_inventory_identity_from_gpu_inventory,
    build_evidence_dependence_map,
    estimate_industrial_budget,
    materialize_confirmation_pilots,
    materialize_confirmation_prefix,
    materialize_e2_final_recipe,
    materialize_industrial_budgets,
    reduce_e1_activation,
    reduce_e2_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_load_binding_from_dict,
    budget_plan_from_dict,
    budget_plan_to_dict,
    budget_policy_from_dict,
    capacity_envelope_from_dict,
    confirmation_family_identity_from_dict,
    confirmation_family_power_reduction_artifact_to_dict,
    e1_pareto_artifact_from_dict,
    e2_final_recipe_artifact_from_dict,
    e2_stage_reduction_artifact_to_dict,
    evidence_alias_reduction_artifact_from_dict,
    evidence_alias_reduction_artifact_to_dict,
    evidence_dependence_map_from_dict,
    evidence_dependence_map_to_dict,
    experiment_budget_from_dict,
    family_activation_artifact_from_dict,
    family_activation_artifact_to_dict,
    industrial_budget_report_to_dict,
    reducer_activation_artifact_from_dict,
    reducer_activation_artifact_to_dict,
    sealed_e3a_selection_from_dict,
)
from lightcone_spec.experiments.protocol import (
    DFLASH_LOSS_POSITION_DECAY,
    TUNING_STAGES,
    assert_confirmation_slice_config,
    assert_matched_confirmation_configs,
    confirmation_blocks,
    onlinespec_blocks,
    select_static_load,
    tuning_candidates,
    tuning_stage,
)
from lightcone_spec.experiments.registry import (
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    build_industrial_registry,
)
from lightcone_spec.experiments.runner import (
    collect_confirmation_performance,
    collect_onlinespec_performance,
    measure_controlled_slice,
    run_confirmation_slice,
    run_greedy_target_reference,
    run_natural_replication_slice,
    run_onlinespec_confirmation_slice,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.selection import (
    CandidateMeasurement,
    SelectionArtifact,
    SliceMeasurement,
    reduce_tuning_stage,
    select_heldout_anchor,
    select_shared_config,
)
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
    RegistryStageDispositionStatus,
    materialize_registry_stage_activation,
    registry_stage_activation_from_dict,
    registry_stage_activation_to_dict,
    verify_registry_stage_activation,
)
from lightcone_spec.experiments.statistics import (
    HardwareEnvelope,
    evaluate_speed_gate,
)
from lightcone_spec.locking import ModelLock, prepare_models, resolve_model_lock
from lightcone_spec.orchestration import (
    SpeedStudyManifest,
    render_onlinespec_runtime_plan,
    render_onlinespec_tuning_runtime_plan,
    render_replication_runtime_plan,
    render_runtime_plan,
    render_static_load_runtime_plan,
    render_target_only_runtime_plan,
    render_tuning_runtime_plan,
)
from lightcone_spec.orchestration.execution_bundle import (
    ExecutionBundleBlockedError,
    execute_dispatch_wave_bundles,
)
from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment
from lightcone_spec.sglang_bridge import (
    SGLangHTTPClient,
    sglang_adaptation_sha256,
    verify_patched_checkout,
)
from lightcone_spec.telemetry import OUTPUT_HASH_FORMAT, load_completed_evidence


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        body = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            body.extend(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or not stat.S_ISREG(current.st_mode)
        ):
            raise ValueError(f"{label} changed while it was read: {path}")
        return bytes(body)
    finally:
        os.close(descriptor)


def _publish_immutable_bytes(path: Path, body: bytes, *, label: str) -> None:
    try:
        existing = _read_regular_bytes(path, label=label)
    except ValueError:
        if path.exists() or path.is_symlink():
            raise
    else:
        if existing != body:
            raise ValueError(f"refusing to overwrite immutable artifact {path}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if _read_regular_bytes(path, label=label) != body:
                raise ValueError(f"refusing to overwrite immutable artifact {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: str | Path, value: object) -> None:
    output = Path(os.path.abspath(os.fspath(path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or output.parent.resolve() != output.parent:
        raise ValueError("artifact parent must be an existing resolved directory")
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = _canonical_sha256(value)
    sidecar = Path(f"{output}.sha256")
    _publish_immutable_bytes(output, body, label="JSON artifact")
    _publish_immutable_bytes(
        sidecar,
        f"{digest}\n".encode("ascii"),
        label="JSON artifact sidecar",
    )
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(body).hexdigest()


def _strict_json_bytes(body: bytes, *, label: str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{label} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error

    def reject_nonfinite(item: object) -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError(f"{label} contains a non-finite JSON number")
        if type(item) is dict:
            for nested in item.values():
                reject_nonfinite(nested)
        elif type(item) is list:
            for nested in item:
                reject_nonfinite(nested)

    reject_nonfinite(value)
    return value


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_request_output_identity(row: dict[str, object]) -> None:
    if row.get("output_hash_format") != OUTPUT_HASH_FORMAT:
        raise ValueError("industrial request has a wrong output hash format")
    serialized = row.get("output_token_ids")
    if not isinstance(serialized, str):
        raise TypeError("industrial request lacks ordered output token IDs")
    try:
        token_ids = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("industrial request has malformed output token IDs") from exc
    if (
        not isinstance(token_ids, list)
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in token_ids
        )
        or json.dumps(token_ids, separators=(",", ":")) != serialized
    ):
        raise ValueError("industrial request has non-canonical output token IDs")
    output_tokens = row.get("output_tokens")
    if (
        not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or len(token_ids) != output_tokens
    ):
        raise ValueError("industrial request token IDs do not cover its output")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if (
        row.get("output_token_ids_sha256") != digest
        or row.get("output_sha256") != digest
    ):
        raise ValueError("industrial request token-ID digest is inconsistent")


def _file_sha256(path: str | Path) -> str:
    source = Path(os.path.abspath(os.fspath(path)))
    return hashlib.sha256(
        _read_regular_bytes(source, label="content-bound artifact")
    ).hexdigest()


def _load_bound_json(path: str | Path) -> object:
    source = Path(os.path.abspath(os.fspath(path)))
    value = _strict_json_bytes(
        _read_regular_bytes(source, label="JSON artifact"),
        label="JSON artifact",
    )
    sidecar = Path(f"{source}.sha256")
    try:
        sidecar_value = _read_regular_bytes(
            sidecar, label="JSON artifact sidecar"
        ).decode("ascii")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            f"JSON artifact sidecar is missing or invalid: {source}"
        ) from error
    if sidecar_value != f"{_canonical_sha256(value)}\n":
        raise ValueError(f"JSON artifact sidecar is missing or invalid: {source}")
    return value


def _artifact_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.suffix.lower() == ".json":
        return _canonical_sha256(_load_bound_json(source))
    return _file_sha256(source)


def _load_bound_run_config(path: str | Path) -> RunConfig:
    source = Path(path)
    config = load_run_config(source)
    sidecar = Path(f"{source}.sha256")
    if not sidecar.is_file() or sidecar.read_text(
        encoding="utf-8"
    ).strip() != run_config_sha256(config):
        raise ValueError(f"run-config sidecar is missing or invalid: {source}")
    return config


_INDUSTRIAL_REGISTRY_GENERATOR = (
    "lightcone_spec.experiments.registry.build_industrial_registry:v2"
)
# No hardware-rooted/provider signing identity is registered in this source
# release.  Self-authored JSON plus hashes proves content consistency, not GPU
# provenance, and therefore cannot mint a claim-bearing receipt.
_TRUSTED_HARDWARE_ATTESTER_ID: str | None = None


def _trusted_attester_unavailable(label: str) -> RuntimeError:
    return RuntimeError(f"{label} is BLOCKED: trusted_hardware_attester_unavailable")


def _industrial_registry_artifact(
    registry: ExperimentRegistry,
    *,
    base_port: int,
    cache_root: str,
    evidence_root: str,
    seed: int,
) -> dict:
    return {
        "schema_version": 2,
        "generator": _INDUSTRIAL_REGISTRY_GENERATOR,
        "parameters": {
            "logical_gpu_slots": list(registry.gpu_uuids),
            "base_port": base_port,
            "cache_root": cache_root,
            "evidence_root": evidence_root,
            "seed": seed,
        },
        "registry_sha256": registry.sha256,
        "registry": registry.to_dict(),
    }


def _load_industrial_registry(path: str | Path) -> ExperimentRegistry:
    value = _load_bound_json(path)
    if not isinstance(value, dict):
        raise TypeError("industrial registry artifact must be an object")
    if set(value) != {
        "schema_version",
        "generator",
        "parameters",
        "registry_sha256",
        "registry",
    }:
        raise ValueError("industrial registry artifact fields do not match schema")
    if (
        value.get("schema_version") != 2
        or value.get("generator") != _INDUSTRIAL_REGISTRY_GENERATOR
    ):
        raise ValueError("industrial registry generator identity mismatch")
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("industrial registry parameters are missing")
    expected_parameters = {
        "logical_gpu_slots",
        "base_port",
        "cache_root",
        "evidence_root",
        "seed",
    }
    if set(parameters) != expected_parameters:
        raise ValueError("industrial registry parameter fields do not match schema")
    logical_gpu_slots = parameters.get("logical_gpu_slots")
    if (
        not isinstance(logical_gpu_slots, list)
        or not logical_gpu_slots
        or len(set(logical_gpu_slots)) != len(logical_gpu_slots)
        or not all(
            isinstance(item, str)
            and item.strip()
            and "\n" not in item
            and "\r" not in item
            for item in logical_gpu_slots
        )
    ):
        raise ValueError("industrial registry requires unique logical rank slots")
    if (
        type(parameters["base_port"]) is not int
        or type(parameters["seed"]) is not int
        or type(parameters["cache_root"]) is not str
        or not parameters["cache_root"]
        or type(parameters["evidence_root"]) is not str
        or not parameters["evidence_root"]
    ):
        raise TypeError("industrial registry parameter types do not match schema")
    registry = build_industrial_registry(
        gpu_uuids=tuple(logical_gpu_slots),
        base_port=parameters["base_port"],
        cache_root=parameters["cache_root"],
        evidence_root=parameters["evidence_root"],
        seed=parameters["seed"],
    )
    if value.get("registry_sha256") != registry.sha256:
        raise ValueError("industrial registry content digest mismatch")
    if value.get("registry") != registry.to_dict():
        raise ValueError(
            "industrial registry declarations were edited after generation"
        )
    return registry


def _load_gpu_inventory(path: str | Path) -> GpuInventory:
    value = _load_bound_json(path)
    inventory = GpuInventory.from_dict(value)
    if inventory.sha256 != _canonical_sha256(inventory.to_dict()):
        raise ValueError("GPU inventory canonical identity mismatch")
    return inventory


def _load_interference_envelope(path: str | Path) -> InterferenceEnvelope:
    value = _load_bound_json(path)
    envelope = InterferenceEnvelope.from_dict(value)
    if envelope.sha256 != _canonical_sha256(envelope.to_dict()):
        raise ValueError("interference envelope canonical identity mismatch")
    return envelope


def _load_budget_plan(path: str | Path) -> BudgetPlan:
    return budget_plan_from_dict(_load_bound_json(path))


def _load_budget_materialization_inputs(
    *,
    policy_path: str | Path,
    load_binding_paths: list[str],
    capacity_envelope_path: str | Path,
    capacity_manifest_path: str | Path | None = None,
    capacity_verification_receipt_path: str | Path | None = None,
):
    policy = budget_policy_from_dict(_load_bound_json(policy_path))
    bindings = tuple(
        budget_load_binding_from_dict(_load_bound_json(path))
        for path in load_binding_paths
    )
    if len({binding.cell_id for binding in bindings}) != len(bindings):
        raise ValueError("duplicate budget load binding")
    capacity = capacity_envelope_from_dict(_load_bound_json(capacity_envelope_path))
    if (capacity_manifest_path is None) != (capacity_verification_receipt_path is None):
        raise ValueError(
            "capacity manifest and verification receipt must be supplied together"
        )
    capacity_authority = (
        None
        if capacity_manifest_path is None
        else bind_capacity_authority(
            capacity_manifest_path,
            capacity_verification_receipt_path,
        )
    )
    return policy, bindings, capacity, capacity_authority


def _load_budget_activation_bundle(
    *,
    activation_paths: list[str],
    family_activation_paths: list[str],
    family_power_plan_paths: list[str],
):
    activations = tuple(
        artifact
        for path in activation_paths
        if (artifact := _load_stage_activation_plan(path)) is not None
    )
    if len({artifact.sha256 for artifact in activations}) != len(activations):
        raise ValueError("duplicate reducer activation artifact")
    family_activations = _load_family_activations(family_activation_paths)
    family_power_reductions = _load_family_power_reductions(family_power_plan_paths)
    if not activations and not family_activations:
        raise RuntimeError(
            "industrial budget materialization is BLOCKED: "
            "reducer_owned_activation_manifest_missing; no bound reducer-owned "
            "activation manifest was supplied"
        )
    return activations, family_activations, family_power_reductions


def _rematerialize_budget_plan(
    *,
    registry: ExperimentRegistry,
    gpu_inventory: GpuInventory,
    declared_plan_path: str | Path,
    activation_paths: list[str],
    family_activation_paths: list[str],
    family_power_plan_paths: list[str],
    policy_path: str | Path,
    load_binding_paths: list[str],
    capacity_envelope_path: str | Path,
    capacity_manifest_path: str | Path | None = None,
    capacity_verification_receipt_path: str | Path | None = None,
    require_ready: bool = True,
) -> tuple[BudgetPlan, tuple, tuple, tuple]:
    """Rebuild one plan from raw authority and reject serialized-plan trust."""

    activations, family_activations, family_power_reductions = (
        _load_budget_activation_bundle(
            activation_paths=activation_paths,
            family_activation_paths=family_activation_paths,
            family_power_plan_paths=family_power_plan_paths,
        )
    )
    (
        policy,
        load_bindings,
        capacity_envelope,
        capacity_authority,
    ) = _load_budget_materialization_inputs(
        policy_path=policy_path,
        load_binding_paths=load_binding_paths,
        capacity_envelope_path=capacity_envelope_path,
        capacity_manifest_path=capacity_manifest_path,
        capacity_verification_receipt_path=(capacity_verification_receipt_path),
    )
    rematerialized = materialize_industrial_budgets(
        registry,
        activations=activations,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
        load_bindings=load_bindings,
        policy=policy,
        inventory=budget_inventory_identity_from_gpu_inventory(gpu_inventory),
        capacity_envelope=capacity_envelope,
        capacity_authority=capacity_authority,
    )
    declared = _load_budget_plan(declared_plan_path)
    if declared != rematerialized or declared.sha256 != rematerialized.sha256:
        raise ValueError(
            "serialized BudgetPlan differs from first-party rematerialization"
        )
    if require_ready:
        declared.require_ready()
    return (
        declared,
        activations,
        family_activations,
        family_power_reductions,
    )


def _industrial_receipt_from_value(value: object) -> ExperimentReceipt:
    if not isinstance(value, dict):
        raise TypeError("industrial dependency receipt must be an object")
    expected = {
        "experiment",
        "registry_sha256",
        "runtime_sha256",
        "split_sha256",
        "completed_cells_sha256",
        "dependency_receipts",
        "outputs",
        "selection_state",
    }
    if set(value) != expected:
        raise ValueError("industrial dependency receipt fields do not match schema")

    def locked_rows(name: str) -> tuple[LockedOutput, ...]:
        rows = value.get(name)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise TypeError(f"industrial receipt {name} must be a list")
        if any(set(row) != {"name", "content_sha256"} for row in rows):
            raise ValueError(f"industrial receipt {name} row fields differ")
        return tuple(
            LockedOutput(
                name=row["name"],
                content_sha256=row["content_sha256"],
            )
            for row in rows
        )

    return ExperimentReceipt(
        experiment=value["experiment"],
        registry_sha256=value["registry_sha256"],
        runtime_sha256=value["runtime_sha256"],
        split_sha256=value["split_sha256"],
        completed_cells_sha256=value["completed_cells_sha256"],
        dependency_receipts=locked_rows("dependency_receipts"),
        outputs=locked_rows("outputs"),
        selection_state=value["selection_state"],
    )


def _load_industrial_receipts(paths: list[str]) -> tuple[ExperimentReceipt, ...]:
    if paths and _TRUSTED_HARDWARE_ATTESTER_ID is None:
        raise _trusted_attester_unavailable("industrial dependency receipts")
    return tuple(
        _industrial_receipt_from_value(_load_bound_json(path)) for path in paths
    )


def _load_registry_stage_activation_manifest(
    path: str | Path,
) -> RegistryStageActivationArtifact:
    value = _load_bound_json(path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "experiment",
        "runtime_artifact",
        "split_artifact",
        "dependency_receipts",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("registry-stage activation manifest fields differ")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("kind") != ("industrial_registry_stage_activation_manifest")
    ):
        raise ValueError("registry-stage activation manifest identity mismatch")

    def artifact_path(name: str) -> str:
        candidate = value.get(name)
        if (
            not isinstance(candidate, str)
            or not candidate.strip()
            or "\n" in candidate
            or "\r" in candidate
        ):
            raise ValueError(f"registry-stage manifest {name} is invalid")
        return candidate

    experiment = value.get("experiment")
    if (
        not isinstance(experiment, str)
        or not experiment.strip()
        or "\n" in experiment
        or "\r" in experiment
    ):
        raise ValueError("registry-stage manifest experiment is invalid")
    dependency_paths = value.get("dependency_receipts")
    if (
        not isinstance(dependency_paths, list)
        or any(
            not isinstance(item, str)
            or not item.strip()
            or "\n" in item
            or "\r" in item
            for item in dependency_paths
        )
        or len(dependency_paths) != len(set(dependency_paths))
    ):
        raise ValueError(
            "registry-stage dependency receipt paths must be unique strings"
        )
    registry = _load_industrial_registry(artifact_path("registry_artifact"))
    artifact = materialize_registry_stage_activation(
        registry,
        experiment=experiment,
        dependency_receipts=_load_industrial_receipts(dependency_paths),
        runtime_sha256=_artifact_sha256(artifact_path("runtime_artifact")),
        split_sha256=_artifact_sha256(artifact_path("split_artifact")),
    )
    verify_registry_stage_activation(registry, artifact)
    return artifact


def _load_stage_activation_plan(path: str | None):
    if path is None:
        return None
    value = _load_bound_json(path)
    if (
        isinstance(value, dict)
        and value.get("kind") == "industrial_registry_stage_activation_manifest"
    ):
        return _load_registry_stage_activation_manifest(path)
    if (
        isinstance(value, dict)
        and value.get("artifact_kind") == "registry_stage_activation"
    ):
        raise ValueError(
            "registry-stage consumers require the bound raw activation manifest"
        )
    if (
        isinstance(value, dict)
        and value.get("kind") == "industrial_e2_activation_manifest"
    ):
        registry, receipt, pareto, stage_index, prior = _load_e2_activation_manifest(
            path
        )
        return reduce_e2_activation(
            registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=stage_index,
            prior_reduction=prior,
        )
    artifact = reducer_activation_artifact_from_dict(value)
    if artifact.plan.experiment == "E2":
        raise ValueError("E2 consumers require a bound raw activation manifest")
    return artifact


def _load_family_activations(paths: list[str]):
    artifacts = tuple(
        family_activation_artifact_from_dict(_load_bound_json(path)) for path in paths
    )
    if len({artifact.sha256 for artifact in artifacts}) != len(artifacts):
        raise ValueError("duplicate confirmation family activation artifact")
    return artifacts


def _load_family_power_reductions(paths: list[str]):
    reductions = tuple(
        reduce_confirmation_family_power(
            registry=registry,
            pilot_activation=pilot,
            blocks=blocks,
            hardware_envelope=envelope,
            inventory=inventory,
            confirmation_data_visible=False,
        )
        for registry, pilot, inventory, envelope, blocks in (
            _load_family_power_manifest(path) for path in paths
        )
    )
    if len({reduction.sha256 for reduction in reductions}) != len(reductions):
        raise ValueError("duplicate confirmation family power artifact")
    return reductions


def _validate_family_artifact_bundle(
    registry: ExperimentRegistry,
    *,
    activations,
    power_reductions,
) -> tuple[str, ...]:
    """Validate full pilot/final lineage and return the current selected round."""

    by_round = {}
    for artifact in activations:
        family = artifact.family
        if family.registry_sha256 != registry.sha256:
            raise ValueError("confirmation activation belongs to another registry")
        key = (family.sha256, artifact.activation_round)
        if key in by_round:
            raise ValueError("duplicate confirmation family activation round")
        by_round[key] = artifact
    by_family_power = {}
    for reduction in power_reductions:
        if reduction.family.registry_sha256 != registry.sha256:
            raise ValueError("confirmation power reduction belongs to another registry")
        if reduction.family.sha256 in by_family_power:
            raise ValueError("duplicate confirmation family power reduction")
        by_family_power[reduction.family.sha256] = reduction

    pilots = {
        family_sha256: artifact
        for (family_sha256, round_name), artifact in by_round.items()
        if round_name == "excluded_pilots"
    }
    finals = {
        family_sha256: artifact
        for (family_sha256, round_name), artifact in by_round.items()
        if round_name == "final_prefix"
    }
    if set(finals) - pilots.keys():
        raise ValueError("family final prefix lacks its pilot activation")
    if set(finals) != set(by_family_power):
        raise ValueError(
            "family power reductions must exactly cover final-prefix activations"
        )

    selected: list[str] = []
    for family_sha256, pilot in sorted(pilots.items()):
        expected_pilot = materialize_confirmation_pilots(registry, pilot.family)
        if pilot != expected_pilot:
            raise ValueError("family pilot activation is not reducer-generated")
        final = finals.get(family_sha256)
        if final is None:
            selected.extend(pilot.activated_cell_ids)
            continue
        reduction = by_family_power[family_sha256]
        expected_final = materialize_confirmation_prefix(
            registry,
            family=pilot.family,
            reduction=reduction,
            pilot_activation=pilot,
        )
        if final != expected_final:
            raise ValueError("family final activation is not reducer-generated")
        selected.extend(final.activated_cell_ids)
    if len(selected) != len(set(selected)):
        raise ValueError("confirmation families select overlapping cells")
    return tuple(sorted(selected))


_BUDGET_OBSERVATION_COMPONENTS = (
    "startup_model_load",
    "compile_jit_graph_prewarm",
    "excluded_warmup",
    "scored_arrival",
    "drain",
    "reset_finalization",
    "evidence_flush_shutdown",
    "soak",
    "failure_injection",
    "retry",
    "profiler",
    "download_compile_reservation",
)
_BUDGET_OBSERVATION_KIND = "industrial_budget_observation_receipt_v1"
_RESERVED_GANG_MEASUREMENT = "exclusive_reserved_gang_wall_ms_x_gpu_count"
_WHOLE_INSTANCE_BILLING = "whole_inventory_wall_clock_v1"


def _industrial_physical_assignment_from_dict(
    value: object,
) -> IndustrialPhysicalAssignment:
    expected = {
        "schema_version",
        "kind",
        "inventory_sha256",
        "inventory_source_receipt_sha256",
        "dispatch_plan_sha256",
        "experiment_budget_sha256",
        "budget_plan_sha256",
        "capacity_authority_sha256",
        "budget_materialization_authority_sha256",
        "assignment_sha256",
        "work_item_sha256",
        "gpu_uuids",
        "rank_groups",
        "ports",
        "gang_shape",
        "fixed_instance_gpu_count",
        "fixed_instance_billing_semantics",
        "host_id",
        "topology_group_ids",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("physical assignment receipt fields differ from schema")
    if value.get("schema_version") != 3 or value.get("kind") != (
        "industrial_physical_assignment"
    ):
        raise ValueError("physical assignment receipt identity mismatch")
    if value.get("fixed_instance_billing_semantics") != _WHOLE_INSTANCE_BILLING:
        raise ValueError("physical assignment billing semantics mismatch")
    gang = value.get("gang_shape")
    if not isinstance(gang, dict) or set(gang) != {
        "tensor_parallel_size",
        "data_parallel_size",
    }:
        raise ValueError("physical assignment gang shape is incomplete")
    gpu_uuids = value.get("gpu_uuids")
    rank_groups = value.get("rank_groups")
    ports = value.get("ports")
    topology_group_ids = value.get("topology_group_ids")
    if (
        not isinstance(gpu_uuids, list)
        or not isinstance(rank_groups, list)
        or not all(isinstance(group, list) for group in rank_groups)
        or not isinstance(ports, list)
        or not isinstance(topology_group_ids, list)
        or not all(isinstance(group, list) for group in topology_group_ids)
    ):
        raise TypeError("physical assignment receipt arrays are malformed")
    try:
        assignment = IndustrialPhysicalAssignment(
            inventory_sha256=value["inventory_sha256"],
            inventory_source_receipt_sha256=value["inventory_source_receipt_sha256"],
            dispatch_plan_sha256=value["dispatch_plan_sha256"],
            experiment_budget_sha256=value["experiment_budget_sha256"],
            budget_plan_sha256=value["budget_plan_sha256"],
            capacity_authority_sha256=value["capacity_authority_sha256"],
            budget_materialization_authority_sha256=value[
                "budget_materialization_authority_sha256"
            ],
            assignment_sha256=value["assignment_sha256"],
            work_item_sha256=value["work_item_sha256"],
            gpu_uuids=tuple(gpu_uuids),
            rank_groups=tuple(tuple(group) for group in rank_groups),
            ports=tuple(ports),
            tensor_parallel_size=gang["tensor_parallel_size"],
            data_parallel_size=gang["data_parallel_size"],
            fixed_instance_gpu_count=value["fixed_instance_gpu_count"],
            host_id=value["host_id"],
            topology_group_ids=tuple(tuple(group) for group in topology_group_ids),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("physical assignment receipt is invalid") from error
    if assignment.to_dict() != value:
        raise ValueError("physical assignment receipt is not canonical")
    return assignment


def _validate_assignment_inventory_authority(
    assignment: IndustrialPhysicalAssignment,
    inventory: GpuInventory,
) -> None:
    """Recompute assignment host, UUID, topology, and billing authority."""

    if len(inventory.host_ids) != 1:
        raise ValueError("formal completion requires one whole-instance GPU host")
    host_id = inventory.host_ids[0]
    if (
        assignment.inventory_sha256 != inventory.sha256
        or assignment.inventory_source_receipt_sha256 != inventory.source_receipt_sha256
        or assignment.fixed_instance_gpu_count != len(inventory.devices)
        or assignment.host_id != host_id
    ):
        raise ValueError("physical assignment differs from the bound GPU inventory")
    devices = {device.uuid: device for device in inventory.devices}
    if any(uuid not in devices for uuid in assignment.gpu_uuids) or any(
        devices[uuid].host_id != host_id for uuid in assignment.gpu_uuids
    ):
        raise ValueError("physical assignment names a foreign GPU or host")
    groups = {group.group_id: group for group in inventory.topology_groups}
    for rank_group, group_ids in zip(
        assignment.rank_groups,
        assignment.topology_group_ids,
        strict=True,
    ):
        if assignment.tensor_parallel_size == 1:
            if group_ids:
                raise ValueError("TP1 assignment cannot claim topology groups")
            continue
        rank_set = set(rank_group)
        if any(
            group_id not in groups
            or groups[group_id].host_id != host_id
            or not rank_set <= set(groups[group_id].gpu_uuids)
            or any(
                group_id not in devices[uuid].allowed_topology_groups
                for uuid in rank_group
            )
            for group_id in group_ids
        ):
            raise ValueError("physical TP assignment lacks bound topology authority")


def _industrial_completion_activation_contract(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    runtime_sha256: str,
    split_sha256: str,
    direct_dependency_receipt_sha256: str | None,
    activation_artifact,
    family_activations,
    family_power_reductions,
    require_stage_sealable: bool = False,
) -> tuple[tuple[str, ...], dict[str, dict[str, str]], dict[str, object]]:
    """Return the exact materialized set and immutable stage dispositions."""

    stage_cells = {cell.cell_id: cell for cell in registry.cells_for(experiment)}
    family_rows = tuple(family_activations)
    power_rows = tuple(family_power_reductions)
    dispositions: dict[str, dict[str, str]] = {}
    requires_dependency = bool(registry.definition(experiment).dependencies)
    if requires_dependency and not _is_lower_sha256(direct_dependency_receipt_sha256):
        raise ValueError("stage completion lacks its direct dependency receipt")
    if not requires_dependency and direct_dependency_receipt_sha256 is not None:
        raise ValueError("root stage cannot claim a dependency receipt")

    if experiment in {"E1", "E2"}:
        if activation_artifact is None:
            raise ValueError(f"{experiment} completion requires an activation artifact")
        if family_rows or power_rows:
            raise ValueError("family artifacts do not belong to this stage")
        plan = activation_artifact.plan
        if (
            plan.registry_sha256 != registry.sha256
            or plan.experiment != experiment
            or plan.runtime_sha256 != runtime_sha256
            or plan.split_sha256 != split_sha256
        ):
            raise ValueError("stage activation runtime/split identity mismatch")
        if plan.dependency_receipt_sha256 != direct_dependency_receipt_sha256:
            raise ValueError("stage activation dependency lineage mismatch")
        activation_round = plan.activation_round
        if experiment == "E1" and activation_round != "e3a_locked_reference":
            raise ValueError("E1 completion requires its locked reference activation")
        if experiment == "E2" and activation_round not in {
            "halving_0",
            "halving_1",
            "halving_2",
            "halving_3",
        }:
            raise ValueError("E2 completion has an unknown halving activation")
        if (
            require_stage_sealable
            and experiment == "E2"
            and activation_round != "halving_3"
        ):
            raise ValueError("E2 cannot seal before the final halving activation")
        disposition_rows = activation_artifact.dispositions
    elif experiment in {"E3b", "E5"}:
        if activation_artifact is not None:
            raise ValueError(
                "stage activation artifact does not belong to confirmation"
            )
        if not family_rows:
            raise ValueError(
                "confirmation completion requires family activation artifacts"
            )
        _validate_family_artifact_bundle(
            registry,
            activations=family_rows,
            power_reductions=power_rows,
        )
        by_round = {
            (artifact.family.sha256, artifact.activation_round): artifact
            for artifact in family_rows
        }
        family_ids = {artifact.family.sha256 for artifact in family_rows}
        final_family_ids = {
            family_sha256
            for family_sha256, round_name in by_round
            if round_name == "final_prefix"
        }
        if require_stage_sealable and final_family_ids != family_ids:
            raise ValueError(
                "confirmation stage cannot seal from pilot-only activations"
            )
        activation_round = (
            "final_prefix"
            if final_family_ids == family_ids
            else "excluded_pilots"
            if not final_family_ids
            else "family_incremental"
        )
        current = []
        for family_sha256 in sorted(family_ids):
            final = by_round.get((family_sha256, "final_prefix"))
            pilot = by_round.get((family_sha256, "excluded_pilots"))
            artifact = final if final is not None else pilot
            if artifact is None:  # pragma: no cover - family_ids construction
                raise RuntimeError("confirmation family lost its validated activation")
            if (
                artifact.family.experiment != experiment
                or artifact.family.runtime_sha256 != runtime_sha256
                or artifact.family.split_sha256 != split_sha256
            ):
                raise ValueError("family activation runtime/split identity mismatch")
            current.extend(artifact.dispositions)
        disposition_rows = tuple(current)
    else:
        if family_rows or power_rows:
            raise ValueError("family artifacts do not belong to this stage")
        if not isinstance(activation_artifact, RegistryStageActivationArtifact):
            raise ValueError(
                "generic stage completion requires reducer-owned registry activation"
            )
        verify_registry_stage_activation(registry, activation_artifact)
        if (
            activation_artifact.experiment != experiment
            or activation_artifact.runtime_sha256 != runtime_sha256
            or activation_artifact.split_sha256 != split_sha256
            or activation_artifact.direct_dependency_receipt_sha256
            != direct_dependency_receipt_sha256
        ):
            raise ValueError(
                "registry-stage activation runtime/split/dependency identity mismatch"
            )
        if require_stage_sealable and activation_artifact.status != "AVAILABLE":
            raise ValueError(
                "generic stage cannot seal without an AVAILABLE registry activation"
            )
        status_map = {
            RegistryStageDispositionStatus.ACTIVATED: DispositionStatus.ACTIVATED,
            RegistryStageDispositionStatus.BLOCKED: DispositionStatus.BLOCKED,
            RegistryStageDispositionStatus.NOT_APPLICABLE: (
                DispositionStatus.NOT_APPLICABLE
            ),
        }
        disposition_rows = tuple(
            {
                "cell_id": row.cell_id,
                "status": status_map[row.status],
                "reason_code": row.reason_code,
            }
            for row in activation_artifact.dispositions
        )
        activation_round = activation_artifact.activation_round

    for row in disposition_rows:
        if isinstance(row, dict):
            cell_id = row["cell_id"]
            status = row["status"]
            reason_code = row["reason_code"]
        else:
            cell_id = row.cell_id
            status = row.status
            reason_code = row.reason_code
        if cell_id not in stage_cells or cell_id in dispositions:
            raise ValueError("activation has an unknown or duplicate disposition")
        if not isinstance(status, DispositionStatus):
            raise TypeError("activation disposition status is not canonical")
        cell = stage_cells[cell_id]
        if status is DispositionStatus.ACTIVATED and not cell.runnable:
            raise ValueError("activation promotes a registry-blocked cell")
        if not cell.runnable:
            expected_status = DispositionStatus(cell.status.value)
            if status is not expected_status or reason_code != cell.reason_code:
                raise ValueError("activation changes an immutable registry disposition")
        dispositions[cell_id] = {
            "cell_id": cell_id,
            "status": status.value,
            "reason_code": reason_code,
        }
    if set(dispositions) != set(stage_cells):
        raise ValueError("activation must disposition every stage template exactly")
    activated = tuple(
        sorted(
            cell_id
            for cell_id, row in dispositions.items()
            if row["status"] == DispositionStatus.ACTIVATED.value
        )
    )
    encoded_dispositions = [dispositions[cell_id] for cell_id in sorted(dispositions)]
    binding = {
        "schema_version": 1,
        "kind": "industrial_stage_activation_binding",
        "stage_activation_sha256": (
            None if activation_artifact is None else activation_artifact.sha256
        ),
        "family_activation_sha256s": sorted(
            artifact.sha256 for artifact in family_rows
        ),
        "family_power_reduction_sha256s": sorted(
            reduction.sha256 for reduction in power_rows
        ),
        "direct_dependency_receipt_sha256": direct_dependency_receipt_sha256,
        "activation_round": activation_round,
        "dispositions_sha256": _canonical_sha256(encoded_dispositions),
    }
    return activated, dispositions, binding


def _validate_budget_observation_receipt(
    path: object,
    *,
    expected_sha256: object,
    experiment_budget_sha256: str,
    prepared_receipt_sha256: object,
    cell,
    evidence_root: Path,
    fixed_instance_gpu_count: int,
) -> None:
    if not _is_lower_sha256(prepared_receipt_sha256):
        raise ValueError("serving completion lacks its prepared receipt binding")
    if not isinstance(path, str) or not path:
        raise ValueError("serving completion lacks a budget observation path")
    source = Path(path)
    sidecar = Path(f"{source}.sha256")
    resolved_root = evidence_root.resolve()
    if (
        source.is_symlink()
        or not source.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
        or resolved_root not in source.resolve().parents
    ):
        raise ValueError("budget observation must be bound inside the evidence root")
    if (
        not _is_lower_sha256(expected_sha256)
        or sidecar.read_text(encoding="utf-8") != f"{expected_sha256}\n"
    ):
        raise ValueError("budget observation semantic sidecar mismatch")
    try:
        artifact = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("budget observation is not valid JSON") from error
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "experiment_budget_sha256",
        "budget_observation_sha256",
        "budget",
        "observed_component_ms",
        "measured_gpu_ms",
        "fixed_instance_billed_gpu_ms",
        "terminal_evidence_sha256",
        "observed_wall_ms",
        "registered_wall_delta_ms",
        "registered_gpu_delta_ms",
        "registered_billed_delta_ms",
        "gpu_measurement_semantics",
        "fixed_instance_billing_semantics",
    }
    if not isinstance(artifact, dict) or set(artifact) != expected_fields:
        raise ValueError("budget observation fields differ from schema")
    budget_value = artifact.get("budget")
    if not isinstance(budget_value, dict):
        raise TypeError("budget observation lacks an ExperimentBudget")
    try:
        budget = experiment_budget_from_dict(
            {
                "artifact_kind": "experiment_budget",
                "artifact_sha256": experiment_budget_sha256,
                **budget_value,
            }
        )
    except (TypeError, ValueError) as error:
        raise ValueError("budget observation contains a forged budget") from error
    if (
        budget.sha256 != experiment_budget_sha256
        or budget.cell_id != cell.cell_id
        or budget.experiment != cell.identity.experiment
        or budget.method != cell.identity.method
        or budget.workload_class is not cell.resources.workload_class
        or budget.gpu_count != cell.resources.gpu_count
        or budget.topology != cell.identity.topology
        or budget.measured_gpu_ms is not None
    ):
        raise ValueError("budget observation belongs to another registry cell")
    rows = artifact.get("observed_component_ms")
    if (
        not isinstance(rows, list)
        or tuple(row[0] for row in rows if isinstance(row, list) and len(row) == 2)
        != _BUDGET_OBSERVATION_COMPONENTS
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], int)
            or isinstance(row[1], bool)
            or row[1] < 0
            for row in rows
        )
    ):
        raise ValueError("budget observation component coverage is invalid")
    try:
        observation = BudgetObservationReceipt(
            schema_version=artifact["schema_version"],
            budget=budget,
            observed_component_ms=tuple((row[0], row[1]) for row in rows),
            measured_gpu_ms=artifact["measured_gpu_ms"],
            fixed_instance_billed_gpu_ms=artifact["fixed_instance_billed_gpu_ms"],
            terminal_evidence_sha256=artifact["terminal_evidence_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("budget observation violates its accounting schema") from error
    expected_artifact = {
        "schema_version": 1,
        "artifact_kind": _BUDGET_OBSERVATION_KIND,
        "experiment_budget_sha256": budget.sha256,
        "budget_observation_sha256": observation.sha256,
        "budget": asdict(budget),
        "observed_component_ms": [
            list(row) for row in observation.observed_component_ms
        ],
        "measured_gpu_ms": observation.measured_gpu_ms,
        "fixed_instance_billed_gpu_ms": observation.fixed_instance_billed_gpu_ms,
        "terminal_evidence_sha256": observation.terminal_evidence_sha256,
        "observed_wall_ms": observation.observed_wall_ms,
        "registered_wall_delta_ms": observation.registered_wall_delta_ms,
        "registered_gpu_delta_ms": observation.registered_gpu_delta_ms,
        "registered_billed_delta_ms": observation.registered_billed_delta_ms,
        "gpu_measurement_semantics": _RESERVED_GANG_MEASUREMENT,
        "fixed_instance_billing_semantics": _WHOLE_INSTANCE_BILLING,
    }
    if (
        artifact != expected_artifact
        or expected_sha256 != observation.sha256
        or observation.terminal_evidence_sha256 != prepared_receipt_sha256
        or observation.measured_gpu_ms
        != observation.observed_wall_ms * budget.gpu_count
        or observation.fixed_instance_billed_gpu_ms
        != observation.observed_wall_ms * fixed_instance_gpu_count
    ):
        raise ValueError("budget observation content binding is invalid")


def _analysis_manifest_path(
    manifest_path: Path,
    value: object,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _analysis_file_binding(
    manifest_path: Path,
    value: object,
    *,
    label: str,
) -> BoundArtifact:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} binding fields do not match schema")
    sha256 = value.get("sha256")
    if not _is_lower_sha256(sha256):
        raise ValueError(f"{label} digest must be lower-case SHA-256")
    return BoundArtifact(
        path=_analysis_manifest_path(manifest_path, value.get("path"), label=label),
        sha256=sha256,
    )


def _analysis_bound_json_path(
    manifest_path: Path,
    value: object,
    *,
    label: str,
) -> Path:
    binding = _analysis_file_binding(manifest_path, value, label=label)
    path = binding.path
    sidecar = Path(f"{path}.sha256")
    if (
        path.is_symlink()
        or not path.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
    ):
        raise ValueError(f"{label} must be a regular bound JSON artifact")
    payload = _load_bound_json(path)
    if _canonical_sha256(payload) != binding.sha256:
        raise ValueError(f"{label} manifest digest mismatch")
    return path


def _analysis_hardware_envelope(value: object) -> HardwareEnvelope:
    expected = {
        "gpu_clock_mhz_min",
        "gpu_clock_mhz_max",
        "memory_clock_mhz_min",
        "memory_clock_mhz_max",
        "temperature_c_max",
        "power_watts_min",
        "power_watts_max",
        "power_state",
        "allowed_throttling_reasons",
        "allowed_background_processes",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("hardware envelope fields do not match schema")
    throttling = value.get("allowed_throttling_reasons")
    processes = value.get("allowed_background_processes")
    if not isinstance(throttling, list) or not isinstance(processes, list):
        raise TypeError("hardware envelope allowlists must be JSON arrays")
    return HardwareEnvelope(
        gpu_clock_mhz_min=value["gpu_clock_mhz_min"],
        gpu_clock_mhz_max=value["gpu_clock_mhz_max"],
        memory_clock_mhz_min=value["memory_clock_mhz_min"],
        memory_clock_mhz_max=value["memory_clock_mhz_max"],
        temperature_c_max=value["temperature_c_max"],
        power_watts_min=value["power_watts_min"],
        power_watts_max=value["power_watts_max"],
        power_state=value["power_state"],
        allowed_throttling_reasons=tuple(throttling),
        allowed_background_processes=tuple(processes),
    )


def _analysis_blocks(
    manifest_path: Path,
    value: object,
) -> tuple[IndustrialBlockEvidence, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("industrial analysis manifest requires block evidence")
    blocks: list[IndustrialBlockEvidence] = []
    for raw_block in value:
        if not isinstance(raw_block, dict) or set(raw_block) != {
            "block",
            "qualification_lock",
            "cells",
        }:
            raise ValueError("industrial analysis block fields do not match schema")
        block = raw_block.get("block")
        raw_cells = raw_block.get("cells")
        if (
            not isinstance(block, int)
            or isinstance(block, bool)
            or not isinstance(raw_cells, list)
            or not raw_cells
        ):
            raise ValueError("industrial analysis block identity is invalid")
        cells: list[IndustrialCellEvidence] = []
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, dict) or set(raw_cell) != {
                "cell_id",
                "terminal_receipts",
                "hardware_receipt",
                "budget_observation",
            }:
                raise ValueError("industrial analysis cell fields do not match schema")
            terminal = raw_cell.get("terminal_receipts")
            if not isinstance(terminal, list) or not terminal:
                raise ValueError("industrial analysis cell requires rank receipts")
            cells.append(
                IndustrialCellEvidence(
                    cell_id=raw_cell.get("cell_id"),
                    terminal_receipts=tuple(
                        _analysis_file_binding(
                            manifest_path,
                            receipt,
                            label="terminal receipt",
                        )
                        for receipt in terminal
                    ),
                    hardware_receipt=_analysis_file_binding(
                        manifest_path,
                        raw_cell.get("hardware_receipt"),
                        label="hardware receipt",
                    ),
                    budget_observation=_analysis_file_binding(
                        manifest_path,
                        raw_cell.get("budget_observation"),
                        label="budget observation",
                    ),
                )
            )
        blocks.append(
            IndustrialBlockEvidence(
                block=block,
                cells=tuple(cells),
                qualification_lock=_analysis_file_binding(
                    manifest_path,
                    raw_block.get("qualification_lock"),
                    label="qualification lock",
                ),
            )
        )
    return tuple(blocks)


def _analysis_cells(
    manifest_path: Path,
    value: object,
) -> tuple[IndustrialCellEvidence, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("E2 raw manifest requires cell evidence")
    cells: list[IndustrialCellEvidence] = []
    for raw_cell in value:
        if not isinstance(raw_cell, dict) or set(raw_cell) != {
            "cell_id",
            "terminal_receipts",
            "hardware_receipt",
            "budget_observation",
        }:
            raise ValueError("E2 raw cell fields do not match schema")
        terminal = raw_cell.get("terminal_receipts")
        if not isinstance(terminal, list) or not terminal:
            raise ValueError("E2 raw cell requires terminal rank receipts")
        cells.append(
            IndustrialCellEvidence(
                cell_id=raw_cell.get("cell_id"),
                terminal_receipts=tuple(
                    _analysis_file_binding(
                        manifest_path,
                        receipt,
                        label="E2 terminal receipt",
                    )
                    for receipt in terminal
                ),
                hardware_receipt=_analysis_file_binding(
                    manifest_path,
                    raw_cell.get("hardware_receipt"),
                    label="E2 hardware receipt",
                ),
                budget_observation=_analysis_file_binding(
                    manifest_path,
                    raw_cell.get("budget_observation"),
                    label="E2 budget observation",
                ),
            )
        )
    return tuple(cells)


def _load_e2_stage_manifest(
    path: str | Path,
    *,
    _seen: frozenset[Path] = frozenset(),
):
    manifest_path = Path(path).resolve()
    if manifest_path in _seen:
        raise ValueError("E2 prior-stage manifests contain a cycle")
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "e1_receipt",
        "pareto",
        "stage_index",
        "prior_stage_manifest",
        "gpu_inventory",
        "hardware_envelope",
        "cells",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("E2 raw stage manifest fields do not match schema")
    stage_index = value.get("stage_index")
    if (
        value.get("schema_version") != 2
        or value.get("kind") != "industrial_e2_stage_reduction_manifest"
        or not isinstance(stage_index, int)
        or isinstance(stage_index, bool)
    ):
        raise ValueError("E2 raw stage manifest identity is invalid")
    registry_path = _analysis_bound_json_path(
        manifest_path,
        value.get("registry_artifact"),
        label="E2 registry artifact",
    )
    receipt_path = _analysis_bound_json_path(
        manifest_path,
        value.get("e1_receipt"),
        label="E1 receipt",
    )
    pareto_path = _analysis_bound_json_path(
        manifest_path,
        value.get("pareto"),
        label="E1 Pareto artifact",
    )
    registry = _load_industrial_registry(registry_path)
    inventory_path = _analysis_bound_json_path(
        manifest_path,
        value.get("gpu_inventory"),
        label="E2 GPU inventory",
    )
    inventory = _load_gpu_inventory(inventory_path)
    receipt = _single_industrial_receipt(receipt_path, experiment="E1")
    pareto = e1_pareto_artifact_from_dict(_load_bound_json(pareto_path))
    prior_value = value.get("prior_stage_manifest")
    if stage_index == 0:
        if prior_value is not None:
            raise ValueError("E2 stage zero cannot name prior raw evidence")
        prior = None
    else:
        prior_path = _analysis_bound_json_path(
            manifest_path,
            prior_value,
            label="prior E2 raw stage manifest",
        )
        prior = _load_e2_stage_manifest(
            prior_path,
            _seen=_seen | {manifest_path},
        )
        if prior.stage_index != stage_index - 1:
            raise ValueError("E2 prior raw stage is not the immediate predecessor")
    return reduce_e2_stage_from_raw(
        registry=registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=stage_index,
        cells=_analysis_cells(manifest_path, value.get("cells")),
        hardware_envelope=_analysis_hardware_envelope(value.get("hardware_envelope")),
        inventory=inventory,
        prior_stage_reduction=prior,
        confirmation_data_visible=False,
    )


def _load_e2_activation_manifest(path: str | Path):
    manifest_path = Path(path)
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "e1_receipt",
        "pareto",
        "stage_index",
        "prior_stage_manifest",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("E2 activation manifest fields do not match schema")
    stage_index = value.get("stage_index")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "industrial_e2_activation_manifest"
        or not isinstance(stage_index, int)
        or isinstance(stage_index, bool)
    ):
        raise ValueError("E2 activation manifest identity is invalid")
    registry_path = _analysis_bound_json_path(
        manifest_path,
        value.get("registry_artifact"),
        label="E2 registry artifact",
    )
    receipt_path = _analysis_bound_json_path(
        manifest_path,
        value.get("e1_receipt"),
        label="E1 receipt",
    )
    pareto_path = _analysis_bound_json_path(
        manifest_path,
        value.get("pareto"),
        label="E1 Pareto artifact",
    )
    registry = _load_industrial_registry(registry_path)
    receipt = _single_industrial_receipt(receipt_path, experiment="E1")
    pareto = e1_pareto_artifact_from_dict(_load_bound_json(pareto_path))
    prior_value = value.get("prior_stage_manifest")
    if stage_index == 0:
        if prior_value is not None:
            raise ValueError("E2 stage zero cannot name prior raw evidence")
        prior = None
    else:
        prior_path = _analysis_bound_json_path(
            manifest_path,
            prior_value,
            label="prior E2 raw stage manifest",
        )
        prior = _load_e2_stage_manifest(prior_path)
        if prior.stage_index != stage_index - 1:
            raise ValueError("E2 prior raw stage is not the immediate predecessor")
    return registry, receipt, pareto, stage_index, prior


def _load_family_power_manifest(
    path: str | Path,
) -> tuple[
    ExperimentRegistry,
    FamilyActivationArtifact,
    GpuInventory,
    HardwareEnvelope,
    tuple[IndustrialBlockEvidence, ...],
]:
    manifest_path = Path(path)
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "pilot_activation",
        "gpu_inventory",
        "hardware_envelope",
        "blocks",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("family power manifest fields do not match schema")
    if (
        value.get("schema_version") != 2
        or value.get("kind") != "industrial_family_power_manifest"
    ):
        raise ValueError("family power manifest identity mismatch")
    registry_path = _analysis_bound_json_path(
        manifest_path,
        value.get("registry_artifact"),
        label="registry artifact",
    )
    pilot_path = _analysis_bound_json_path(
        manifest_path,
        value.get("pilot_activation"),
        label="pilot activation",
    )
    registry = _load_industrial_registry(registry_path)
    pilot = family_activation_artifact_from_dict(_load_bound_json(pilot_path))
    inventory_path = _analysis_bound_json_path(
        manifest_path,
        value.get("gpu_inventory"),
        label="family GPU inventory",
    )
    inventory = _load_gpu_inventory(inventory_path)
    envelope = _analysis_hardware_envelope(value.get("hardware_envelope"))
    return (
        registry,
        pilot,
        inventory,
        envelope,
        _analysis_blocks(manifest_path, value["blocks"]),
    )


def _load_industrial_analysis_manifest(
    path: str | Path,
) -> tuple[
    ExperimentRegistry,
    FamilyActivationArtifact,
    FamilyActivationArtifact,
    ConfirmationFamilyPowerReductionArtifact,
    GpuInventory,
    HardwareEnvelope,
    tuple[IndustrialBlockEvidence, ...],
    tuple[RawEvidenceAliasManifest, ...],
    EvidenceDependenceMap | None,
    BoundArtifact | None,
    BoundArtifact | None,
    int,
    int,
]:
    manifest_path = Path(path)
    manifest_sidecar = Path(f"{manifest_path}.sha256")
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_sidecar.is_symlink()
        or not manifest_sidecar.is_file()
    ):
        raise ValueError("industrial analysis manifest must be a regular bound file")
    value = _load_bound_json(manifest_path)
    expected = {
        "schema_version",
        "kind",
        "registry_artifact",
        "pilot_activation",
        "final_activation",
        "confirmation_power_manifest",
        "gpu_inventory",
        "evidence_alias_manifests",
        "evidence_dependence_map",
        "gpu_attestation",
        "doctor_report",
        "hardware_envelope",
        "bootstrap",
        "blocks",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("industrial analysis manifest fields do not match schema")
    if (
        value.get("schema_version") != 3
        or value.get("kind") != "industrial_analysis_manifest"
    ):
        raise ValueError("industrial analysis manifest identity mismatch")

    registry_path = _analysis_bound_json_path(
        manifest_path,
        value.get("registry_artifact"),
        label="registry artifact",
    )
    pilot_path = _analysis_bound_json_path(
        manifest_path,
        value.get("pilot_activation"),
        label="pilot activation",
    )
    final_path = _analysis_bound_json_path(
        manifest_path,
        value.get("final_activation"),
        label="final activation",
    )
    power_manifest_path = _analysis_bound_json_path(
        manifest_path,
        value.get("confirmation_power_manifest"),
        label="confirmation family power manifest",
    )
    registry = _load_industrial_registry(registry_path)
    pilot = family_activation_artifact_from_dict(_load_bound_json(pilot_path))
    final = family_activation_artifact_from_dict(_load_bound_json(final_path))
    inventory_path = _analysis_bound_json_path(
        manifest_path,
        value.get("gpu_inventory"),
        label="analysis GPU inventory",
    )
    inventory = _load_gpu_inventory(inventory_path)
    raw_alias_bindings = value.get("evidence_alias_manifests")
    if type(raw_alias_bindings) is not list:
        raise ValueError("industrial analysis alias manifests must be a JSON array")
    alias_manifests = tuple(
        raw_evidence_alias_manifest_from_dict(
            _load_bound_json(
                _analysis_bound_json_path(
                    manifest_path,
                    binding,
                    label=f"raw evidence alias manifest {index}",
                )
            )
        )
        for index, binding in enumerate(raw_alias_bindings)
    )
    if len({manifest.sha256 for manifest in alias_manifests}) != len(alias_manifests):
        raise ValueError("duplicate raw evidence alias manifest")
    raw_dependence = value.get("evidence_dependence_map")
    dependence = None
    if raw_dependence is not None:
        dependence_path = _analysis_bound_json_path(
            manifest_path,
            raw_dependence,
            label="evidence dependence map",
        )
        dependence = evidence_dependence_map_from_dict(
            _load_bound_json(dependence_path)
        )
    raw_attestation = value.get("gpu_attestation")
    raw_doctor = value.get("doctor_report")
    if (raw_attestation is None) != (raw_doctor is None):
        raise ValueError(
            "industrial GPU attestation and doctor report must be supplied together"
        )
    gpu_attestation = (
        None
        if raw_attestation is None
        else _analysis_file_binding(
            manifest_path,
            raw_attestation,
            label="industrial GPU attestation",
        )
    )
    doctor_report = (
        None
        if raw_doctor is None
        else _analysis_file_binding(
            manifest_path,
            raw_doctor,
            label="doctor report",
        )
    )
    envelope = _analysis_hardware_envelope(value.get("hardware_envelope"))
    power_registry, power_pilot, power_inventory, power_envelope, power_blocks = (
        _load_family_power_manifest(power_manifest_path)
    )
    if (
        power_registry.sha256 != registry.sha256
        or power_pilot != pilot
        or power_inventory != inventory
        or power_envelope != envelope
    ):
        raise ValueError(
            "analysis family-power manifest differs from registry/pilot/hardware"
        )
    plan = reduce_confirmation_family_power(
        registry=power_registry,
        pilot_activation=power_pilot,
        blocks=power_blocks,
        hardware_envelope=power_envelope,
        inventory=power_inventory,
        confirmation_data_visible=False,
    )

    bootstrap = value.get("bootstrap")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {
        "repetitions",
        "seed",
    }:
        raise ValueError("industrial analysis bootstrap fields do not match schema")
    repetitions = bootstrap.get("repetitions")
    seed = bootstrap.get("seed")
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 100
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("industrial analysis bootstrap values are invalid")

    blocks = _analysis_blocks(manifest_path, value.get("blocks"))
    return (
        registry,
        pilot,
        final,
        plan,
        inventory,
        envelope,
        blocks,
        alias_manifests,
        dependence,
        gpu_attestation,
        doctor_report,
        repetitions,
        seed,
    )


def _parse_locked_outputs(values: list[str]) -> dict[str, str]:
    return {
        name: _artifact_sha256(path)
        for name, path in _parse_locked_output_paths(values).items()
    }


def _parse_locked_output_paths(values: list[str]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in outputs:
            raise ValueError("locked outputs must be unique NAME=PATH values")
        outputs[name] = path
    return outputs


def _static_load_rows(
    value: object,
    *,
    manifest: SpeedStudyManifest,
) -> list[dict]:
    if not isinstance(value, dict):
        raise TypeError("Static load screen must be a schema-v2 terminal artifact")
    expected = {
        "schema_version": 2,
        "phase": "static_load_screen",
        "manifest_sha256": manifest.sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "window_sha256": manifest.controlled_window_hashes["load"],
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("Static load-screen artifact identity mismatch")
    model_lock_sha256 = value.get("model_lock_sha256")
    if not _is_lower_sha256(model_lock_sha256):
        raise ValueError("Static load-screen model-lock identity is invalid")
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("Static load-screen artifact lacks measurement rows")
    return rows


def _formal_table_metadata(
    *,
    manifest: SpeedStudyManifest,
    selection: SelectionArtifact,
    model_lock: ModelLock,
    config_sha256: dict[str, str],
    source_evidence_sha256: str,
    target_reference_sha256: str,
) -> dict[bytes, bytes]:
    return {
        b"lightcone_schema_version": b"2",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": model_lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": (
            manifest.sampling_profile_sha256.encode()
        ),
        b"lightcone_execution_policy_sha256": (
            manifest.execution_policy_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_config_set_sha256": _canonical_sha256(config_sha256).encode(),
        b"lightcone_source_evidence_sha256": source_evidence_sha256.encode(),
        b"lightcone_target_reference_sha256": target_reference_sha256.encode(),
    }


def _load_formal_table(
    path: str | Path,
    *,
    manifest: SpeedStudyManifest,
    selection: SelectionArtifact,
    model_lock: ModelLock,
    target_reference: GreedyTargetReference,
) -> pa.Table:
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    expected = {
        b"lightcone_schema_version": b"2",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": model_lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": (
            manifest.sampling_profile_sha256.encode()
        ),
        b"lightcone_execution_policy_sha256": (
            manifest.execution_policy_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("formal speed table identity metadata mismatch")
    for key in (
        b"lightcone_config_set_sha256",
        b"lightcone_source_evidence_sha256",
    ):
        value = metadata.get(key, b"").decode(errors="ignore")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("formal speed table evidence metadata is invalid")
    return table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lightcone-spec")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor")
    doctor.add_argument(
        "--project-root",
        help="clean LightCone-Spec checkout containing the runtime manifest",
    )
    doctor.add_argument(
        "--sglang-root",
        help="separate clean checkout with the complete pinned SGLang patch set",
    )
    doctor.add_argument(
        "--path",
        help="legacy shorthand that uses one path for both roots (not ready)",
    )

    validate = commands.add_parser("validate-config")
    validate.add_argument("config")

    build = commands.add_parser("build-speed-study")
    build.add_argument("--output", required=True)

    build_industrial = commands.add_parser("build-industrial-registry")
    build_industrial.add_argument(
        "--logical-gpu-slot",
        nargs="+",
        default=("logical-rank-slot-0", "logical-rank-slot-1"),
        metavar="SLOT",
        help=(
            "one or more stable logical rank slots; physical GPU UUIDs come only from "
            "the inventory and frozen dispatch assignment"
        ),
    )
    build_industrial.add_argument("--base-port", type=int, default=24000)
    build_industrial.add_argument("--cache-root", default="runtime-cache/industrial")
    build_industrial.add_argument("--evidence-root", default="artifacts/industrial")
    build_industrial.add_argument("--seed", type=int, default=20260811)
    build_industrial.add_argument("--output", required=True)

    collect_inventory = commands.add_parser("collect-gpu-inventory")
    collect_inventory.add_argument("--challenge-nonce-sha256", required=True)
    collect_inventory.add_argument("--receipt-output", required=True)
    collect_inventory.add_argument("--output", required=True)

    build_interference = commands.add_parser("build-interference-envelope")
    build_interference.add_argument("--inventory", required=True)
    build_interference.add_argument("--receipt-output", required=True)
    build_interference.add_argument("--output", required=True)

    seal_industrial = commands.add_parser("seal-industrial-stage")
    seal_industrial.add_argument("--registry", required=True)
    seal_industrial.add_argument("--experiment", required=True)
    seal_industrial.add_argument("--runtime-artifact", required=True)
    seal_industrial.add_argument("--split-artifact", required=True)
    seal_industrial.add_argument("--completed-cells", required=True)
    seal_industrial.add_argument("--inventory", required=True)
    seal_industrial.add_argument("--e2-final-stage-manifest")
    seal_industrial.add_argument("--activation-plan")
    seal_industrial.add_argument("--family-activation", action="append", default=[])
    seal_industrial.add_argument("--family-power-plan", action="append", default=[])
    seal_industrial.add_argument("--dependency-receipt", action="append", default=[])
    seal_industrial.add_argument("--locked-output", action="append", required=True)
    seal_industrial.add_argument("--output", required=True)

    plan_industrial = commands.add_parser("plan-industrial-dispatch")
    plan_industrial.add_argument("--registry", required=True)
    plan_industrial.add_argument("--inventory", required=True)
    plan_industrial.add_argument("--interference-envelope", required=True)
    plan_industrial.add_argument("--budget-plan", required=True)
    plan_industrial.add_argument("--budget-policy", required=True)
    plan_industrial.add_argument("--budget-load-binding", action="append", default=[])
    plan_industrial.add_argument("--capacity-envelope", required=True)
    plan_industrial.add_argument("--capacity-manifest")
    plan_industrial.add_argument("--capacity-verification-receipt")
    plan_industrial.add_argument("--receipt", action="append", default=[])
    plan_industrial.add_argument("--completed-cells")
    plan_industrial.add_argument("--completed-e2-stage-manifest")
    plan_industrial.add_argument("--activation-plan", action="append", default=[])
    plan_industrial.add_argument("--family-activation", action="append", default=[])
    plan_industrial.add_argument("--family-power-plan", action="append", default=[])
    plan_industrial.add_argument("--output", required=True)

    execute_wave = commands.add_parser("execute-dispatch-wave", allow_abbrev=False)
    execute_wave.add_argument("--bundle", action="append", default=[])
    execute_wave.add_argument("--wave-index", type=int, required=True)
    execute_wave.add_argument("--resume-receipt")
    execute_wave.add_argument("--receipt-output", required=True)

    materialize_budget = commands.add_parser("materialize-industrial-budgets")
    materialize_budget.add_argument("--registry", required=True)
    materialize_budget.add_argument("--activation-plan", action="append", default=[])
    materialize_budget.add_argument("--family-activation", action="append", default=[])
    materialize_budget.add_argument("--family-power-plan", action="append", default=[])
    materialize_budget.add_argument("--budget-policy", required=True)
    materialize_budget.add_argument(
        "--budget-load-binding", action="append", default=[]
    )
    materialize_budget.add_argument("--capacity-envelope", required=True)
    materialize_budget.add_argument("--capacity-manifest")
    materialize_budget.add_argument("--capacity-verification-receipt")
    materialize_budget.add_argument("--inventory", required=True)
    materialize_budget.add_argument("--output", required=True)

    materialize_stage = commands.add_parser("materialize-stage-activation")
    materialize_stage.add_argument("--manifest", required=True)
    materialize_stage.add_argument("--output", required=True)

    estimate_budget = commands.add_parser("estimate-industrial-budget")
    estimate_budget.add_argument("--registry", required=True)
    estimate_budget.add_argument("--activation-plan", action="append", default=[])
    estimate_budget.add_argument("--family-activation", action="append", default=[])
    estimate_budget.add_argument("--family-power-plan", action="append", default=[])
    estimate_budget.add_argument("--inventory", required=True)
    estimate_budget.add_argument("--interference-envelope", required=True)
    estimate_budget.add_argument("--budget-plan", required=True)
    estimate_budget.add_argument("--budget-policy", required=True)
    estimate_budget.add_argument("--budget-load-binding", action="append", default=[])
    estimate_budget.add_argument("--capacity-envelope", required=True)
    estimate_budget.add_argument("--capacity-manifest")
    estimate_budget.add_argument("--capacity-verification-receipt")
    estimate_budget.add_argument("--output", required=True)

    reduce_e1 = commands.add_parser("reduce-e1-activation")
    reduce_e1.add_argument("--registry", required=True)
    reduce_e1.add_argument("--e3a-receipt", required=True)
    reduce_e1.add_argument("--selection", required=True)
    reduce_e1.add_argument("--output", required=True)

    reduce_e2 = commands.add_parser("reduce-e2-activation")
    reduce_e2.add_argument("--manifest", required=True)
    reduce_e2.add_argument("--output", required=True)

    halve_e2 = commands.add_parser("reduce-e2-successive-halving")
    halve_e2.add_argument("--manifest", required=True)
    halve_e2.add_argument("--output", required=True)

    family_pilots = commands.add_parser("materialize-confirmation-pilots")
    family_pilots.add_argument("--registry", required=True)
    family_pilots.add_argument("--family", required=True)
    family_pilots.add_argument("--output", required=True)

    family_power = commands.add_parser("reduce-confirmation-family-power")
    family_power.add_argument("--manifest", required=True)
    family_power.add_argument("--output", required=True)

    family_prefix = commands.add_parser("materialize-confirmation-prefix")
    family_prefix.add_argument("--power-manifest", required=True)
    family_prefix.add_argument("--output", required=True)

    validate_alias = commands.add_parser("validate-evidence-alias", allow_abbrev=False)
    validate_alias.add_argument("--manifest", required=True)
    validate_alias.add_argument("--registry", required=True)
    validate_alias.add_argument("--inventory", required=True)
    validate_alias.add_argument("--hardware-envelope", required=True)
    validate_alias.add_argument("--output", required=True)

    dependence = commands.add_parser(
        "build-evidence-dependence-map", allow_abbrev=False
    )
    dependence.add_argument("--direct-map", required=True)
    dependence.add_argument("--alias-reduction", action="append", default=[])
    dependence.add_argument("--output", required=True)

    analyze_industrial = commands.add_parser("analyze-industrial")
    analyze_industrial.add_argument("--manifest", required=True)
    analyze_industrial.add_argument("--output", required=True)

    build_online = commands.add_parser("build-onlinespec-study")
    build_online.add_argument("--output", required=True)

    verify_online_source = commands.add_parser("verify-onlinespec-source")
    verify_online_source.add_argument("--checkout", required=True)
    verify_online_source.add_argument("--audit", required=True)
    verify_online_source.add_argument("--output", required=True)

    list_online = commands.add_parser("list-onlinespec-candidates")
    list_online.add_argument("--output", required=True)

    select_online = commands.add_parser("select-onlinespec-config")
    select_online.add_argument("--measurements", required=True)
    select_online.add_argument("--manifest", required=True)
    select_online.add_argument("--model-lock", required=True)
    select_online.add_argument("--sampling-profile", required=True)
    select_online.add_argument("--core-selection", required=True)
    select_online.add_argument("--output", required=True)

    select_online_anchor = commands.add_parser("select-onlinespec-anchor-config")
    select_online_anchor.add_argument("--measurements", nargs=4, required=True)
    select_online_anchor.add_argument("--candidate-ids", nargs=3, required=True)
    select_online_anchor.add_argument("--manifest", required=True)
    select_online_anchor.add_argument("--model-lock", required=True)
    select_online_anchor.add_argument("--sampling-profile", required=True)
    select_online_anchor.add_argument("--core-selection", required=True)
    select_online_anchor.add_argument("--output", required=True)

    lock = commands.add_parser("lock-models")
    lock.add_argument("--output", required=True)
    lock.add_argument("models", nargs="+")

    prepare = commands.add_parser("prepare-models")
    prepare.add_argument("--lockfile", required=True)
    prepare.add_argument("--model-cache", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--offline", action="store_true")

    select = commands.add_parser("select-speed-config")
    select.add_argument("--measurements", required=True)
    select.add_argument("--static-load-screen", required=True)
    select.add_argument("--manifest", required=True)
    select.add_argument("--model-lock", required=True)
    select.add_argument("--sampling-profile", required=True)
    select.add_argument("--output", required=True)

    select_anchor = commands.add_parser("select-anchor-config")
    select_anchor.add_argument("--measurements", nargs=3, required=True)
    select_anchor.add_argument("--candidate-id", required=True)
    select_anchor.add_argument("--static-load-screen", required=True)
    select_anchor.add_argument("--manifest", required=True)
    select_anchor.add_argument("--model-lock", required=True)
    select_anchor.add_argument("--sampling-profile", required=True)
    select_anchor.add_argument("--output", required=True)

    render = commands.add_parser("render-runtime")
    render.add_argument("--selection", required=True)
    render.add_argument("--model-lock", required=True)
    render.add_argument("--model-roots", required=True)
    render.add_argument("--sglang-checkout", required=True)
    render.add_argument("--sampling-profile", required=True)
    render.add_argument("--adaptation-group-id", required=True)
    render.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render.add_argument("--mem-fraction-static", type=float, required=True)
    render.add_argument("--output-root", required=True)
    render.add_argument("--host", default="127.0.0.1")
    render.add_argument("--first-port", type=int, default=30000)

    render_online = commands.add_parser("render-onlinespec-runtime")
    render_online.add_argument("--selection", required=True)
    render_online.add_argument("--model-lock", required=True)
    render_online.add_argument("--model-roots", required=True)
    render_online.add_argument("--sglang-checkout", required=True)
    render_online.add_argument("--sampling-profile", required=True)
    render_online.add_argument("--adaptation-group-id", required=True)
    render_online.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_online.add_argument("--mem-fraction-static", type=float, required=True)
    render_online.add_argument("--output-root", required=True)
    render_online.add_argument("--host", default="127.0.0.1")
    render_online.add_argument("--first-port", type=int, default=30000)

    render_online_tune = commands.add_parser("render-onlinespec-tuning-runtime")
    render_online_tune.add_argument("--candidate-id", required=True)
    render_online_tune.add_argument("--concurrency", type=int, required=True)
    render_online_tune.add_argument("--model-lock", required=True)
    render_online_tune.add_argument("--model-roots", required=True)
    render_online_tune.add_argument("--sglang-checkout", required=True)
    render_online_tune.add_argument("--sampling-profile", required=True)
    render_online_tune.add_argument("--adaptation-group-id", required=True)
    render_online_tune.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_online_tune.add_argument("--mem-fraction-static", type=float, required=True)
    render_online_tune.add_argument("--output-root", required=True)
    render_online_tune.add_argument("--host", default="127.0.0.1")
    render_online_tune.add_argument("--first-port", type=int, default=30000)

    render_static = commands.add_parser("render-static-load-runtime")
    render_static.add_argument("--concurrency", type=int, required=True)
    render_static.add_argument("--model-lock", required=True)
    render_static.add_argument("--model-roots", required=True)
    render_static.add_argument("--sglang-checkout", required=True)
    render_static.add_argument("--sampling-profile", required=True)
    render_static.add_argument("--mem-fraction-static", type=float, required=True)
    render_static.add_argument("--output-root", required=True)
    render_static.add_argument("--host", default="127.0.0.1")
    render_static.add_argument("--first-port", type=int, default=30000)

    render_target = commands.add_parser("render-target-only-runtime")
    render_target.add_argument("--concurrency", type=int, required=True)
    render_target.add_argument("--model-lock", required=True)
    render_target.add_argument("--model-roots", required=True)
    render_target.add_argument("--sglang-checkout", required=True)
    render_target.add_argument("--sampling-profile", required=True)
    render_target.add_argument("--mem-fraction-static", type=float, required=True)
    render_target.add_argument("--output-root", required=True)
    render_target.add_argument("--host", default="127.0.0.1")
    render_target.add_argument("--first-port", type=int, default=30000)

    render_tuning = commands.add_parser("render-tuning-runtime")
    render_tuning.add_argument("--candidate-id", required=True)
    render_tuning.add_argument("--concurrency", type=int, required=True)
    render_tuning.add_argument("--model-lock", required=True)
    render_tuning.add_argument("--model-roots", required=True)
    render_tuning.add_argument("--sglang-checkout", required=True)
    render_tuning.add_argument("--sampling-profile", required=True)
    render_tuning.add_argument("--adaptation-group-id", required=True)
    render_tuning.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_tuning.add_argument("--mem-fraction-static", type=float, required=True)
    render_tuning.add_argument("--output-root", required=True)
    render_tuning.add_argument("--host", default="127.0.0.1")
    render_tuning.add_argument("--first-port", type=int, default=30000)

    replication = commands.add_parser("render-replication-runtime")
    replication.add_argument("--phase", choices=("natural", "profile"), required=True)
    replication.add_argument("--selection", required=True)
    replication.add_argument("--model-lock", required=True)
    replication.add_argument("--model-roots", required=True)
    replication.add_argument("--sglang-checkout", required=True)
    replication.add_argument("--sampling-profile", required=True)
    replication.add_argument("--adaptation-group-id", required=True)
    replication.add_argument("--adaptation-reserve-mb", type=int, required=True)
    replication.add_argument("--mem-fraction-static", type=float, required=True)
    replication.add_argument("--output-root", required=True)
    replication.add_argument("--host", default="127.0.0.1")
    replication.add_argument("--first-port", type=int, default=30000)

    candidates = commands.add_parser("list-tuning-candidates")
    candidates.add_argument("--output", required=True)

    controlled = commands.add_parser("run-controlled-slice")
    controlled.add_argument("--manifest", required=True)
    controlled.add_argument("--model-lock", required=True)
    controlled.add_argument("--sampling-profile", required=True)
    controlled.add_argument("--config", required=True)
    controlled.add_argument("--url", required=True)
    controlled.add_argument("--phase", choices=("static-load", "tune"), required=True)
    controlled.add_argument("--method", choices=("static", "tts", "l0"), required=True)
    controlled.add_argument("--candidate-id")
    controlled.add_argument("--stage", type=int, default=-1)
    controlled.add_argument("--concurrency", type=int, required=True)
    controlled.add_argument("--output", required=True)
    controlled.add_argument("--no-warmup", action="store_true")

    controlled_online = commands.add_parser("run-onlinespec-tuning-slice")
    controlled_online.add_argument("--manifest", required=True)
    controlled_online.add_argument("--model-lock", required=True)
    controlled_online.add_argument("--sampling-profile", required=True)
    controlled_online.add_argument("--config", required=True)
    controlled_online.add_argument("--url", required=True)
    controlled_online.add_argument(
        "--method", choices=ONLINE_SPEC_STUDY_METHODS, required=True
    )
    controlled_online.add_argument("--candidate-id")
    controlled_online.add_argument("--stage", type=int, required=True)
    controlled_online.add_argument("--concurrency", type=int, required=True)
    controlled_online.add_argument("--output", required=True)
    controlled_online.add_argument("--no-warmup", action="store_true")

    natural = commands.add_parser("run-natural-slice")
    natural.add_argument("--manifest", required=True)
    natural.add_argument("--selection", required=True)
    natural.add_argument("--model-lock", required=True)
    natural.add_argument("--sampling-profile", required=True)
    natural.add_argument("--config", required=True)
    natural.add_argument("--url", required=True)
    natural.add_argument("--method", choices=("static", "tts", "l0"), required=True)
    natural.add_argument(
        "--dataset", choices=("livecodebench", "math500"), required=True
    )
    natural.add_argument("--dataset-revision", required=True)
    natural.add_argument("--split", required=True)
    natural.add_argument("--output-root", required=True)
    natural.add_argument("--no-warmup", action="store_true")

    profiler = commands.add_parser("build-profiler-plan")
    profiler.add_argument("--launch-plan", required=True)
    profiler.add_argument("--method", choices=("static", "tts", "l0"), required=True)
    profiler.add_argument("--trace-root", required=True)
    profiler.add_argument("--output", required=True)
    profiler.add_argument("workload_argv", nargs=argparse.REMAINDER)

    load_collect = commands.add_parser("collect-static-load-screen")
    load_collect.add_argument("--manifest", required=True)
    load_collect.add_argument("--measurements", nargs="+", required=True)
    load_collect.add_argument("--output", required=True)

    advance = commands.add_parser("advance-tuning-stage")
    advance.add_argument("--manifest", required=True)
    advance.add_argument("--stage", type=int, required=True)
    advance.add_argument("--measurements", nargs="+", required=True)
    advance.add_argument("--active-set")
    advance.add_argument("--output", required=True)

    advance_online = commands.add_parser("advance-onlinespec-tuning-stage")
    advance_online.add_argument("--manifest", required=True)
    advance_online.add_argument("--stage", type=int, required=True)
    advance_online.add_argument("--measurements", nargs="+", required=True)
    advance_online.add_argument("--active-set")
    advance_online.add_argument("--output", required=True)

    run = commands.add_parser("run-confirmation")
    run.add_argument("--manifest", required=True)
    run.add_argument("--selection", required=True)
    run.add_argument("--model-lock", required=True)
    run.add_argument("--sampling-profile", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--url", required=True)
    run.add_argument("--method", choices=("static", "tts", "l0"), required=True)
    run.add_argument("--block", type=int, required=True)
    run.add_argument("--adaptation-group-id", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--no-warmup", action="store_true")

    run_online = commands.add_parser("run-onlinespec-confirmation")
    run_online.add_argument("--manifest", required=True)
    run_online.add_argument("--selection", required=True)
    run_online.add_argument("--model-lock", required=True)
    run_online.add_argument("--sampling-profile", required=True)
    run_online.add_argument("--config", required=True)
    run_online.add_argument("--url", required=True)
    run_online.add_argument(
        "--method", choices=ONLINE_SPEC_STUDY_METHODS, required=True
    )
    run_online.add_argument("--block", type=int, required=True)
    run_online.add_argument("--adaptation-group-id", required=True)
    run_online.add_argument("--output-root", required=True)
    run_online.add_argument("--no-warmup", action="store_true")

    target_reference = commands.add_parser("run-target-reference")
    target_reference.add_argument("--model-lock", required=True)
    target_reference.add_argument("--sampling-profile", required=True)
    target_reference.add_argument("--url", required=True)
    target_reference.add_argument("--concurrency", type=int, required=True)
    target_reference.add_argument("--doctor-json", required=True)
    target_reference.add_argument("--output", required=True)
    target_reference.add_argument("--no-warmup", action="store_true")

    collect = commands.add_parser("collect-speed-study")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--selection", required=True)
    collect.add_argument("--model-lock", required=True)
    collect.add_argument("--static-config", required=True)
    collect.add_argument("--tts-config", required=True)
    collect.add_argument("--l0-config", required=True)
    collect.add_argument("--evidence-root", required=True)
    collect.add_argument("--target-reference", required=True)
    collect.add_argument("--output", required=True)

    collect_online = commands.add_parser("collect-onlinespec-study")
    collect_online.add_argument("--manifest", required=True)
    collect_online.add_argument("--selection", required=True)
    collect_online.add_argument("--model-lock", required=True)
    collect_online.add_argument("--static-config", required=True)
    collect_online.add_argument("--ogd-config", required=True)
    collect_online.add_argument("--opt-config", required=True)
    collect_online.add_argument("--ens-config", required=True)
    collect_online.add_argument("--evidence-root", required=True)
    collect_online.add_argument("--target-reference", required=True)
    collect_online.add_argument("--output", required=True)

    queue = commands.add_parser("build-confirmation-queue")
    queue.add_argument("--manifest", required=True)
    queue.add_argument("--selection", required=True)
    queue.add_argument("--model-lock", required=True)
    queue.add_argument("--sampling-profile", required=True)
    queue.add_argument("--launch-plan", required=True)
    queue.add_argument("--evidence-root", required=True)
    queue.add_argument("--output", required=True)

    queue_online = commands.add_parser("build-onlinespec-queue")
    queue_online.add_argument("--manifest", required=True)
    queue_online.add_argument("--selection", required=True)
    queue_online.add_argument("--model-lock", required=True)
    queue_online.add_argument("--sampling-profile", required=True)
    queue_online.add_argument("--launch-plan", required=True)
    queue_online.add_argument("--evidence-root", required=True)
    queue_online.add_argument("--output", required=True)

    attest = commands.add_parser("attest-speed-study")
    attest.add_argument("--manifest", required=True)
    attest.add_argument("--selection", required=True)
    attest.add_argument("--model-lock", required=True)
    attest.add_argument("--performance", required=True)
    attest.add_argument("--target-reference", required=True)
    attest.add_argument("--doctor-json", required=True)
    attest.add_argument("--output", required=True)

    attest_online = commands.add_parser("attest-onlinespec-study")
    attest_online.add_argument("--manifest", required=True)
    attest_online.add_argument("--selection", required=True)
    attest_online.add_argument("--model-lock", required=True)
    attest_online.add_argument("--performance", required=True)
    attest_online.add_argument("--target-reference", required=True)
    attest_online.add_argument("--doctor-json", required=True)
    attest_online.add_argument("--output", required=True)

    analyze = commands.add_parser("analyze-speed-study")
    analyze.add_argument("--performance", required=True)
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--selection", required=True)
    analyze.add_argument("--model-lock", required=True)
    analyze.add_argument("--target-reference", required=True)
    analyze.add_argument("--attestation")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--bootstrap-seed", type=int, default=0)

    analyze_online = commands.add_parser("analyze-onlinespec-study")
    analyze_online.add_argument("--performance", required=True)
    analyze_online.add_argument("--manifest", required=True)
    analyze_online.add_argument("--selection", required=True)
    analyze_online.add_argument("--model-lock", required=True)
    analyze_online.add_argument("--target-reference", required=True)
    analyze_online.add_argument("--attestation")
    analyze_online.add_argument("--output", required=True)
    analyze_online.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def _select(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    tuning_artifact = _load_bound_json(args.measurements)
    load_artifact = _load_bound_json(args.static_load_screen)
    load_rows = _static_load_rows(load_artifact, manifest=manifest)
    measurements_value = tuning_artifact
    if isinstance(tuning_artifact, dict):
        if (
            tuning_artifact.get("schema_version") != 2
            or tuning_artifact.get("phase") != "shared_config_tuning"
            or tuning_artifact.get("manifest_sha256") != manifest.sha256
            or tuning_artifact.get("stage") != len(TUNING_STAGES) - 1
            or tuning_artifact.get("next_stage") is not None
            or not _is_lower_sha256(tuning_artifact.get("prior_stage_sha256"))
        ):
            raise ValueError("selection requires the terminal tuning-stage artifact")
        measurements_value = tuning_artifact.get("candidate_measurements")
    else:
        raise TypeError("selection requires a terminal tuning-stage artifact")
    if not isinstance(measurements_value, list):
        raise TypeError("terminal tuning measurements must be a JSON array")
    lock = ModelLock.load(args.model_lock)
    selected_concurrency = select_static_load(
        load_rows,
        required_context_limit=manifest.safe_context_limit,
    )
    if (
        tuning_artifact.get("model_lock_sha256") != lock.sha256
        or load_artifact.get("model_lock_sha256") != lock.sha256
    ):
        raise ValueError("selection inputs belong to a different model lock")
    if (
        tuning_artifact.get("sampling_profile_sha256") != sampling.sha256
        or tuning_artifact.get("execution_policy_sha256")
        != manifest.execution_policy_sha256
        or tuning_artifact.get("window_sha256")
        != manifest.controlled_window_hashes["tune"]
        or tuning_artifact.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
        or tuning_artifact.get("concurrency") != selected_concurrency
    ):
        raise ValueError(
            "terminal tuning artifact is not bound to this study and selected load"
        )
    measurements = [CandidateMeasurement(**row) for row in measurements_value]
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    artifact = select_shared_config(
        measurements,
        candidates=grid,
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest.sha256,
        sampling_profile_sha256=sampling.sha256,
        tuning_grid_sha256=manifest.tuning_grid_sha256,
        load_screen_sha256=_canonical_sha256(load_artifact),
        tuning_window_sha256=LongContinuationAdapter().window_sha256("tune"),
        model_lock_sha256=lock.sha256,
        tuning_evidence_sha256=_canonical_sha256(tuning_artifact),
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _select_anchor(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    lock = ModelLock.load(args.model_lock)
    load_artifact = _load_bound_json(args.static_load_screen)
    load_rows = _static_load_rows(load_artifact, manifest=manifest)
    selected_concurrency = select_static_load(
        load_rows,
        required_context_limit=manifest.safe_context_limit,
    )
    if load_artifact.get("model_lock_sha256") != lock.sha256:
        raise ValueError("Static load screen belongs to a different model lock")
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("anchor candidate is outside the registered tuning grid")
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    expected_count, expected_context = tuning_stage(len(TUNING_STAGES) - 1)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.model_lock_sha256 != lock.sha256
        or row.sampling_profile_sha256 != sampling.sha256
        or row.window_sha256 != expected_window
        or row.prompt_count != expected_count
        or row.context_limit != expected_context
        for row in measurements
    ):
        raise ValueError("anchor measurement identity is not terminal tuning evidence")
    evidence = _canonical_sha256(
        {
            "selection_protocol": "heldout_anchor",
            "candidate_id": candidate.candidate_id,
            "measurement_sha256": sorted(row.sha256 for row in measurements),
        }
    )
    artifact = select_heldout_anchor(
        measurements,
        candidate=candidate,
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest.sha256,
        sampling_profile_sha256=sampling.sha256,
        tuning_grid_sha256=manifest.tuning_grid_sha256,
        load_screen_sha256=_canonical_sha256(load_artifact),
        tuning_window_sha256=manifest.controlled_window_hashes["tune"],
        model_lock_sha256=lock.sha256,
        tuning_evidence_sha256=evidence,
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _select_onlinespec(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    core_selection = SelectionArtifact.load(args.core_selection)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC manifest uses another sampling profile")
    if (
        core_selection.manifest_sha256 != SpeedStudyManifest.default().sha256
        or core_selection.model_lock_sha256 != lock.sha256
        or core_selection.sampling_profile_sha256 != sampling.sha256
    ):
        raise ValueError(
            "OnlineSPEC selection requires the registered core Static load"
        )
    value = _load_bound_json(args.measurements)
    if (
        value.get("schema_version") != 2
        or value.get("phase") != "onlinespec_tuning"
        or value.get("manifest_sha256") != manifest.sha256
        or value.get("model_lock_sha256") != lock.sha256
        or value.get("sampling_profile_sha256") != sampling.sha256
        or value.get("execution_policy_sha256") != manifest.execution_policy_sha256
        or value.get("window_sha256") != manifest.tuning_window_sha256
        or value.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
        or value.get("stage") != len(ONLINE_SPEC_TUNING_STAGES) - 1
        or value.get("next_stage") is not None
        or not _is_lower_sha256(value.get("prior_stage_sha256"))
        or value.get("concurrency") != core_selection.selected_concurrency
    ):
        raise ValueError("OnlineSPEC tuning artifact identity mismatch")
    raw_rows = value.get("measurements")
    if not isinstance(raw_rows, list):
        raise TypeError("OnlineSPEC tuning artifact lacks measurements")
    candidates = {
        candidate.candidate_id: candidate for candidate in onlinespec_candidates()
    }
    selection = select_onlinespec(
        [OnlineSpecTuningMeasurement(**row) for row in raw_rows],
        candidates=candidates,
        selected_concurrency=core_selection.selected_concurrency,
        manifest_sha256=manifest.sha256,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        reference_core_selection_sha256=core_selection.sha256,
        tuning_evidence_sha256=_canonical_sha256(value),
    )
    selection.write(args.output)
    print(selection.sha256)
    return 0


def _select_onlinespec_anchor(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    core_selection = SelectionArtifact.load(args.core_selection)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC manifest uses another sampling profile")
    if (
        core_selection.manifest_sha256 != SpeedStudyManifest.default().sha256
        or core_selection.model_lock_sha256 != lock.sha256
        or core_selection.sampling_profile_sha256 != sampling.sha256
    ):
        raise ValueError(
            "OnlineSPEC selection requires the registered core Static load"
        )
    candidate_ids = tuple(args.candidate_ids)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("OnlineSPEC anchor candidate identities must be unique")
    registered = {
        candidate.candidate_id: candidate for candidate in onlinespec_candidates()
    }
    try:
        candidates = {
            candidate_id: registered[candidate_id] for candidate_id in candidate_ids
        }
    except KeyError as exc:
        raise ValueError(
            "OnlineSPEC anchor is outside the registered tuning grid"
        ) from exc
    if {candidate.method for candidate in candidates.values()} != set(
        ONLINE_SPEC_METHODS
    ):
        raise ValueError("OnlineSPEC anchor requires one candidate per learner")
    measurements = tuple(SliceMeasurement.load(path) for path in args.measurements)
    expected_count, expected_context = onlinespec_tuning_stage(
        len(ONLINE_SPEC_TUNING_STAGES) - 1
    )
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.model_lock_sha256 != lock.sha256
        or row.sampling_profile_sha256 != sampling.sha256
        or row.window_sha256 != expected_window
        or row.prompt_count != expected_count
        or row.context_limit != expected_context
        for row in measurements
    ):
        raise ValueError(
            "OnlineSPEC anchor measurement identity is not terminal tuning evidence"
        )
    evidence = _canonical_sha256(
        {
            "selection_protocol": "heldout_anchor",
            "candidate_ids": sorted(candidate_ids),
            "measurement_sha256": sorted(row.sha256 for row in measurements),
        }
    )
    selection = select_onlinespec_heldout_anchor(
        measurements,
        candidates=candidates,
        selected_concurrency=core_selection.selected_concurrency,
        manifest_sha256=manifest.sha256,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        reference_core_selection_sha256=core_selection.sha256,
        tuning_evidence_sha256=evidence,
    )
    selection.write(args.output)
    print(selection.sha256)
    return 0


def _assert_onlinespec_study(
    manifest: OnlineSpecManifest,
    selection: OnlineSpecSelection,
    lock: ModelLock,
    sampling: SamplingProfile | None = None,
) -> None:
    if (
        selection.manifest_sha256 != manifest.sha256
        or selection.model_lock_sha256 != lock.sha256
        or selection.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or (
            sampling is not None
            and selection.sampling_profile_sha256 != sampling.sha256
        )
    ):
        raise ValueError("OnlineSPEC study identities do not match")


def _study_inputs(
    args: argparse.Namespace,
) -> tuple[SpeedStudyManifest, ModelLock, SamplingProfile]:
    manifest = SpeedStudyManifest.load(args.manifest)
    model_lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    return manifest, model_lock, sampling


def _run_controlled_slice(args: argparse.Namespace) -> int:
    manifest, model_lock, sampling = _study_inputs(args)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling,
    )
    if config.method != args.method:
        raise ValueError("run config is bound to another method")
    if config.runtime.max_running_requests < args.concurrency:
        raise ValueError("slice concurrency exceeds the runtime admission limit")
    adapter = LongContinuationAdapter()
    if args.phase == "static-load":
        if args.method != "static" or args.candidate_id is not None or args.stage != -1:
            raise ValueError(
                "Static load screen accepts only an unadapted stage -1 slice"
            )
        if args.concurrency not in manifest.concurrency_grid:
            raise ValueError("Static load concurrency is outside the registered grid")
        samples = adapter.window("load")
        phase = "static_load_screen"
        context_limit = manifest.load_screen_context_limit
        candidate_id = None
    else:
        prompt_count, context_limit = tuning_stage(args.stage)
        samples = adapter.window("tune")[:prompt_count]
        phase = "shared_config_tuning"
        candidate_id = args.candidate_id
        if args.method == "static":
            if candidate_id is not None:
                raise ValueError("Static tuning baseline has no candidate ID")
        else:
            grid = {
                candidate.candidate_id: candidate for candidate in tuning_candidates()
            }
            candidate = grid.get(candidate_id or "")
            if candidate is None:
                raise ValueError("adapted tuning slice has an unknown candidate ID")
            assert_confirmation_slice_config(
                config,
                method=args.method,
                selected_candidate=candidate,
                selected_concurrency=args.concurrency,
            )
    if config.model.max_context_length < context_limit:
        raise ValueError("slice context exceeds the locked model configuration")
    group = (
        "static-preselection"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    measurement = measure_controlled_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        samples=samples,
        phase=phase,
        stage=args.stage,
        candidate_id=candidate_id,
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        model_lock_sha256=model_lock.sha256,
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        sampling_profile=sampling,
        context_limit=context_limit,
        concurrency=args.concurrency,
        adaptation_group_id=group,
        warmup=not args.no_warmup,
    )
    measurement.write(args.output)
    print(measurement.sha256)
    return 0


def _collect_static_load(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    expected_window = LongContinuationAdapter().window_sha256("load")
    if len(measurements) != len(manifest.concurrency_grid):
        raise ValueError("Static load screen has incomplete concurrency coverage")
    rows = []
    model_locks = {row.model_lock_sha256 for row in measurements}
    if len(model_locks) != 1:
        raise ValueError("Static load screen mixes model-lock identities")
    for row in measurements:
        if (
            row.phase != "static_load_screen"
            or row.stage != -1
            or row.method != "static"
            or row.manifest_sha256 != manifest.sha256
            or row.window_sha256 != expected_window
            or row.prompt_count != 8
            or row.context_limit != manifest.load_screen_context_limit
            or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        ):
            raise ValueError("Static load slice identity mismatch")
        rows.append(
            {
                "concurrency": row.concurrency,
                "decode_goodput_tps": row.decode_goodput_tps,
                "itl_p99_ms": row.itl_p99_ms,
                "oom_events": row.oom_events,
                "retractions": row.retractions,
                "kv_token_capacity": row.kv_token_capacity,
                "measurement_sha256": row.sha256,
            }
        )
    rows.sort(key=lambda value: int(value["concurrency"]))
    select_static_load(
        rows,
        required_context_limit=manifest.safe_context_limit,
    )
    artifact = {
        "schema_version": 2,
        "phase": "static_load_screen",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": next(iter(model_locks)),
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "window_sha256": expected_window,
        "rows": rows,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _advance_tuning(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    tuning_stage(args.stage)
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    prior = None
    if args.stage == 0:
        if args.active_set:
            raise ValueError("stage zero must start from the complete registered grid")
        active = tuple(sorted(grid))
    else:
        if not args.active_set:
            raise ValueError("later tuning stages require the prior survivor artifact")
        prior = _load_bound_json(args.active_set)
        if (
            not isinstance(prior, dict)
            or prior.get("schema_version") != 2
            or prior.get("phase") != "shared_config_tuning"
            or prior.get("manifest_sha256") != manifest.sha256
            or prior.get("next_stage") != args.stage
            or prior.get("stage") != args.stage - 1
            or not isinstance(prior.get("survivors"), list)
            or prior.get("sampling_profile_sha256") != manifest.sampling_profile_sha256
            or prior.get("execution_policy_sha256") != manifest.execution_policy_sha256
            or prior.get("window_sha256") != manifest.controlled_window_hashes["tune"]
            or prior.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
            or (args.stage == 1 and prior.get("prior_stage_sha256") is not None)
            or (
                args.stage > 1 and not _is_lower_sha256(prior.get("prior_stage_sha256"))
            )
        ):
            raise ValueError("prior tuning survivor artifact is invalid")
        active = tuple(str(value) for value in prior["survivors"])
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    model_locks = {row.model_lock_sha256 for row in measurements}
    if len(model_locks) != 1:
        raise ValueError("tuning stage mixes model-lock identities")
    model_lock_sha256 = next(iter(model_locks))
    if prior is not None and prior.get("model_lock_sha256") != model_lock_sha256:
        raise ValueError(
            "tuning stage uses a different model lock than its predecessor"
        )
    expected_count, _ = tuning_stage(args.stage)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    concurrencies = {row.concurrency for row in measurements}
    if len(concurrencies) != 1:
        raise ValueError("tuning stage mixes runtime concurrency")
    concurrency = next(iter(concurrencies))
    if prior is not None and prior.get("concurrency") != concurrency:
        raise ValueError("tuning stage changes the selected runtime load")
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.window_sha256 != expected_window
        or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        for row in measurements
    ):
        raise ValueError("tuning measurements use another manifest or prompt window")
    survivors, candidate_rows = reduce_tuning_stage(
        measurements,
        candidates=grid,
        active_candidate_ids=active,
        stage=args.stage,
    )
    next_stage = args.stage + 1 if args.stage + 1 < len(TUNING_STAGES) else None
    artifact = {
        "schema_version": 2,
        "phase": "shared_config_tuning",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": model_lock_sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "window_sha256": manifest.controlled_window_hashes["tune"],
        "tuning_grid_sha256": manifest.tuning_grid_sha256,
        "concurrency": concurrency,
        "stage": args.stage,
        "next_stage": next_stage,
        "prior_stage_sha256": (None if prior is None else _canonical_sha256(prior)),
        "active_candidates": list(active),
        "survivors": list(survivors),
        "measurement_sha256": sorted(row.sha256 for row in measurements),
        "candidate_measurements": [asdict(row) for row in candidate_rows],
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _list_tuning_candidates(args: argparse.Namespace) -> int:
    rows = [
        {**asdict(candidate), "candidate_id": candidate.candidate_id}
        for candidate in tuning_candidates()
    ]
    _write_json(args.output, rows)
    print(_canonical_sha256(rows))
    return 0


def _list_onlinespec_candidates(args: argparse.Namespace) -> int:
    rows = [
        {**asdict(candidate), "candidate_id": candidate.candidate_id}
        for candidate in onlinespec_candidates()
    ]
    _write_json(args.output, rows)
    print(_canonical_sha256(rows))
    return 0


def _confirmation_inputs(
    args: argparse.Namespace,
) -> tuple[SpeedStudyManifest, SelectionArtifact, ModelLock, SamplingProfile]:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    if selection.tuning_window_sha256 != manifest.controlled_window_hashes["tune"]:
        raise ValueError("selection artifact belongs to a different tuning window")
    sampling_profile = SamplingProfile.load(args.sampling_profile)
    if (
        sampling_profile.sha256 != manifest.sampling_profile_sha256
        or selection.sampling_profile_sha256 != sampling_profile.sha256
        or selection.tuning_grid_sha256 != manifest.tuning_grid_sha256
    ):
        raise ValueError("sampling profile belongs to a different manifest")
    return manifest, selection, model_lock, sampling_profile


def _assert_selection_study(
    selection: SelectionArtifact, manifest: SpeedStudyManifest
) -> None:
    if (
        selection.manifest_sha256 != manifest.sha256
        or selection.tuning_grid_sha256 != manifest.tuning_grid_sha256
        or selection.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or selection.tuning_window_sha256 != manifest.controlled_window_hashes["tune"]
    ):
        raise ValueError("selection artifact belongs to a different speed study")


def _assert_locked_config(
    config: RunConfig,
    *,
    model_lock: ModelLock,
    sampling_profile: SamplingProfile,
) -> None:
    if config.runtime.sampling_profile_sha256 != sampling_profile.sha256:
        raise ValueError("run config does not match the sampling profile")
    if config.runtime.execution_policy_sha256 != ControlledExecutionPolicy().sha256:
        raise ValueError("run config does not match the registered execution policy")
    locked = {model.model_id: model.revision for model in model_lock.models}
    pair = config.model
    if (
        locked.get(pair.target) != pair.target_revision
        or locked.get(pair.drafter) != pair.drafter_revision
    ):
        raise ValueError("run config does not match the immutable model lock")


def _load_target_reference(
    path: str | Path,
    *,
    model_lock: ModelLock,
    sampling_profile_sha256: str,
    concurrency: int,
) -> GreedyTargetReference:
    revisions = {model.model_id: model.revision for model in model_lock.models}
    target_revision = revisions.get("Qwen/Qwen3-8B")
    if target_revision is None:
        raise ValueError("model lock lacks the formal Qwen3-8B target")
    reference = GreedyTargetReference.load(path)
    reference.verify_study(
        model_lock_sha256=model_lock.sha256,
        target_revision=target_revision,
        sampling_profile_sha256=sampling_profile_sha256,
        execution_policy_sha256=ControlledExecutionPolicy().sha256,
        window_sha256=LongContinuationAdapter().window_sha256("confirm"),
        concurrency=concurrency,
    )
    return reference


def _load_patched_gpu_doctor(path: str | Path, *, purpose: str) -> dict:
    hardware = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(hardware, dict):
        raise TypeError(f"{purpose} doctor evidence is not an object")
    commands = hardware.get("commands")
    if not isinstance(commands, dict):
        raise TypeError(f"{purpose} doctor commands are malformed")
    nvidia = commands.get("nvidia_smi")
    source_tree = hardware.get("source_tree")
    if not isinstance(nvidia, str) or not nvidia.strip():
        raise ValueError(f"{purpose} requires a successful nvidia-smi report")
    if (
        not isinstance(source_tree, dict)
        or source_tree.get("is_git_checkout") is not True
        or source_tree.get("tree") != PINNED_SGLANG_TREE
        or source_tree.get("dirty") is not False
        or source_tree.get("pinned_ancestor") is not True
        or source_tree.get("patch_commits") != PINNED_SGLANG_PATCH_COUNT
    ):
        raise ValueError(f"{purpose} requires the exact clean patched checkout")
    return hardware


def _run_confirmation(args: argparse.Namespace) -> int:
    manifest, selection, model_lock, sampling_profile = _confirmation_inputs(args)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling_profile,
    )
    assert_confirmation_slice_config(
        config,
        method=args.method,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    if (
        config.adaptation is not None
        and config.adaptation.adaptation_group_id != args.adaptation_group_id
    ):
        raise ValueError("run argument and config adaptation groups differ")
    written = run_confirmation_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        block=args.block,
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=args.adaptation_group_id,
        schedule_seed=manifest.confirmation_schedule_seed,
        sampling_profile=sampling_profile,
        model_pair=manifest.model_pair,
        warmup=not args.no_warmup,
    )
    if not written:
        raise RuntimeError("confirmation slice produced no completed evidence")
    print(f"completed {args.block}/{args.method}: {len(written)} files")
    return 0


def _assert_onlinespec_candidate_config(
    config: RunConfig,
    *,
    method: str,
    candidate: OnlineSpecCandidate | None,
    concurrency: int,
) -> None:
    model = config.model
    runtime = config.runtime
    if (
        model.key != "qwen3_8b_dflash16"
        or model.target != "Qwen/Qwen3-8B"
        or model.drafter != "z-lab/Qwen3-8B-DFlash-b16"
        or model.algorithm != "DFLASH"
        or model.max_context_length != DFLASH_MODEL_CONTEXT_LIMIT
        or model.draft_depth != 15
        or runtime.speculative_num_draft_tokens != 16
        or runtime.telemetry_detail != "headline"
    ):
        raise ValueError("OnlineSPEC run config is outside the registered DFlash study")
    if config.method != method or runtime.max_running_requests != concurrency:
        raise ValueError("OnlineSPEC run config method or load mismatch")
    if method == "static":
        if candidate is not None:
            raise ValueError("OnlineSPEC Static reference has no candidate")
        if config.adaptation is not None or config.online_spec is not None:
            raise ValueError("OnlineSPEC Static reference allocated adaptation state")
        return
    adaptation = config.adaptation
    learner = config.online_spec
    if (
        candidate is None
        or candidate.method != method
        or adaptation is None
        or learner is None
    ):
        raise ValueError("OnlineSPEC run config is incomplete")
    actual = {
        "weight_update_mode": adaptation.weight_update_mode,
        "parameter_scope": adaptation.parameter_scope,
        "learning_rate": adaptation.optimizer.learning_rate,
        "rank": adaptation.rank,
        "stride": adaptation.stride,
        "projection_radius": learner.projection_radius,
        "additional_learning_rates": learner.additional_learning_rates,
        "hedge_learning_rate": learner.hedge_learning_rate,
        "grad_clip": adaptation.optimizer.grad_clip,
    }
    expected = {
        key: value for key, value in asdict(candidate).items() if key not in {"method"}
    }
    if (
        actual != expected
        or adaptation.optimizer.name != "sgd"
        or not math.isclose(
            adaptation.loss_position_decay,
            DFLASH_LOSS_POSITION_DECAY,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("OnlineSPEC run config does not match its tuning selection")


def _assert_onlinespec_config(
    config: RunConfig,
    *,
    method: str,
    selection: OnlineSpecSelection,
) -> None:
    candidate = next(
        (candidate for candidate in selection.selected if candidate.method == method),
        None,
    )
    _assert_onlinespec_candidate_config(
        config,
        method=method,
        candidate=candidate,
        concurrency=selection.selected_concurrency,
    )


def _run_onlinespec_tuning_slice(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC tuning uses another sampling profile")
    config = _load_bound_run_config(args.config)
    _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
    prompt_count, context_limit = onlinespec_tuning_stage(args.stage)
    samples = LongContinuationAdapter().window("tune")[:prompt_count]
    candidate = None
    if args.method == "static":
        if args.candidate_id is not None:
            raise ValueError("OnlineSPEC Static tuning has no candidate ID")
    else:
        grid = {row.candidate_id: row for row in onlinespec_candidates()}
        candidate = grid.get(args.candidate_id or "")
        if candidate is None:
            raise ValueError("OnlineSPEC tuning candidate is not registered")
    _assert_onlinespec_candidate_config(
        config,
        method=args.method,
        candidate=candidate,
        concurrency=args.concurrency,
    )
    if config.model.max_context_length < context_limit:
        raise ValueError("OnlineSPEC tuning exceeds the locked model context")
    group = (
        "onlinespec-static-tuning"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    measurement = measure_controlled_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        samples=samples,
        phase="onlinespec_tuning",
        stage=args.stage,
        candidate_id=(None if candidate is None else candidate.candidate_id),
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        model_lock_sha256=lock.sha256,
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        sampling_profile=sampling,
        context_limit=context_limit,
        concurrency=args.concurrency,
        adaptation_group_id=group,
        warmup=not args.no_warmup,
    )
    measurement.write(args.output)
    print(measurement.sha256)
    return 0


def _advance_onlinespec_tuning(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    onlinespec_tuning_stage(args.stage)
    grid = {candidate.candidate_id: candidate for candidate in onlinespec_candidates()}
    prior = None
    if args.stage == 0:
        if args.active_set:
            raise ValueError("stage zero starts from the complete OnlineSPEC grid")
        active = tuple(sorted(grid))
    else:
        if not args.active_set:
            raise ValueError("later OnlineSPEC stages require survivor evidence")
        prior = _load_bound_json(args.active_set)
        if (
            prior.get("schema_version") != 2
            or prior.get("phase") != "onlinespec_tuning"
            or prior.get("manifest_sha256") != manifest.sha256
            or prior.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
            or prior.get("window_sha256") != manifest.tuning_window_sha256
            or prior.get("sampling_profile_sha256") != manifest.sampling_profile_sha256
            or prior.get("execution_policy_sha256") != manifest.execution_policy_sha256
            or prior.get("stage") != args.stage - 1
            or prior.get("next_stage") != args.stage
            or not isinstance(prior.get("survivors"), list)
            or (args.stage == 1 and prior.get("prior_stage_sha256") is not None)
            or (
                args.stage > 1 and not _is_lower_sha256(prior.get("prior_stage_sha256"))
            )
        ):
            raise ValueError("prior OnlineSPEC tuning artifact is invalid")
        active = tuple(str(value) for value in prior["survivors"])
    measurements = tuple(SliceMeasurement.load(path) for path in args.measurements)
    model_locks = {row.model_lock_sha256 for row in measurements}
    concurrencies = {row.concurrency for row in measurements}
    if len(model_locks) != 1 or len(concurrencies) != 1:
        raise ValueError("OnlineSPEC stage mixes model locks or runtime loads")
    model_lock_sha256 = next(iter(model_locks))
    concurrency = next(iter(concurrencies))
    if prior is not None and (
        prior.get("model_lock_sha256") != model_lock_sha256
        or prior.get("concurrency") != concurrency
    ):
        raise ValueError("OnlineSPEC tuning changed its model lock or load")
    prompt_count, _ = onlinespec_tuning_stage(args.stage)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:prompt_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or row.window_sha256 != expected_window
        for row in measurements
    ):
        raise ValueError("OnlineSPEC measurements use another registered input")
    survivors, reduced = reduce_onlinespec_tuning_stage(
        measurements,
        candidates=grid,
        active_candidate_ids=active,
        stage=args.stage,
    )
    next_stage = (
        args.stage + 1 if args.stage + 1 < len(ONLINE_SPEC_TUNING_STAGES) else None
    )
    artifact = {
        "schema_version": 2,
        "phase": "onlinespec_tuning",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": model_lock_sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "tuning_grid_sha256": manifest.tuning_grid_sha256,
        "window_sha256": manifest.tuning_window_sha256,
        "measurement_window_sha256": expected_window,
        "concurrency": concurrency,
        "stage": args.stage,
        "next_stage": next_stage,
        "prior_stage_sha256": (None if prior is None else _canonical_sha256(prior)),
        "active_candidates": list(active),
        "survivors": list(survivors),
        "measurement_sha256": sorted(row.sha256 for row in measurements),
        "measurements": [asdict(row) for row in reduced],
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _run_onlinespec_confirmation(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    _assert_onlinespec_study(manifest, selection, lock, sampling)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
    _assert_onlinespec_config(config, method=args.method, selection=selection)
    if (
        config.adaptation is not None
        and config.adaptation.adaptation_group_id != args.adaptation_group_id
    ):
        raise ValueError("OnlineSPEC adaptation group mismatch")
    written = run_onlinespec_confirmation_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        block=args.block,
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=args.adaptation_group_id,
        schedule_seed=manifest.confirmation_schedule_seed,
        sampling_profile=sampling,
        warmup=not args.no_warmup,
    )
    if not written:
        raise RuntimeError("OnlineSPEC slice produced no completed evidence")
    print(f"completed OnlineSPEC {args.block}/{args.method}: {len(written)} files")
    return 0


def _run_target_reference(args: argparse.Namespace) -> int:
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    hardware = _load_patched_gpu_doctor(
        args.doctor_json,
        purpose="target reference",
    )
    revisions = {model.model_id: model.revision for model in lock.models}
    target_revision = revisions.get("Qwen/Qwen3-8B")
    if target_revision is None:
        raise ValueError("model lock lacks the formal Qwen3-8B target")
    artifact = run_greedy_target_reference(
        client=SGLangHTTPClient(args.url),
        model_lock_sha256=lock.sha256,
        target_revision=target_revision,
        hardware_sha256=_canonical_sha256(hardware),
        concurrency=args.concurrency,
        sampling_profile=sampling,
        warmup=not args.no_warmup,
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _confirmation_configs(args: argparse.Namespace) -> dict:
    return {
        "static": _load_bound_run_config(args.static_config),
        "tts": _load_bound_run_config(args.tts_config),
        "l0": _load_bound_run_config(args.l0_config),
    }


def _concat_evidence_tables(paths: tuple[Path, ...]) -> pa.Table:
    """Concatenate evidence while promoting only all-null inferred columns."""
    tables = [pq.read_table(path) for path in paths]
    if not tables:
        raise ValueError("evidence table set cannot be empty")
    column_names = tables[0].column_names
    if any(table.column_names != column_names for table in tables[1:]):
        raise ValueError("evidence tables have different columns")
    for index, name in enumerate(column_names):
        concrete_types: list[pa.DataType] = []
        for table in tables:
            data_type = table.schema.field(index).type
            if not pa.types.is_null(data_type) and data_type not in concrete_types:
                concrete_types.append(data_type)
        if len(concrete_types) > 1:
            rendered = ", ".join(str(value) for value in concrete_types)
            raise ValueError(
                f"evidence column {name!r} has incompatible types: {rendered}"
            )
    return pa.concat_tables(tables, promote_options="default")


def _collect_speed_study(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    configs = _confirmation_configs(args)
    for config in configs.values():
        if config.runtime.sampling_profile_sha256 != manifest.sampling_profile_sha256:
            raise ValueError("run config does not match the manifest sampling profile")
        if config.runtime.execution_policy_sha256 != manifest.execution_policy_sha256:
            raise ValueError("run config does not match the execution policy")
        locked = {model.model_id: model.revision for model in model_lock.models}
        if (
            locked.get(config.model.target) != config.model.target_revision
            or locked.get(config.model.drafter) != config.model.drafter_revision
        ):
            raise ValueError("run config does not match the immutable model lock")
    assert_matched_confirmation_configs(
        configs,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=model_lock,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    target_revision = next(
        model.revision
        for model in model_lock.models
        if model.model_id == "Qwen/Qwen3-8B"
    )
    performance, source_evidence_sha256 = collect_confirmation_performance(
        evidence_root=args.evidence_root,
        manifest_sha256=manifest.sha256,
        config_sha256={
            method: run_config_sha256(config) for method, config in configs.items()
        },
        concurrency=selection.selected_concurrency,
        target_reference=target_reference,
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        execution_policy_sha256=manifest.execution_policy_sha256,
        target_revision=target_revision,
    )
    table = _concat_evidence_tables(performance)
    table = table.replace_schema_metadata(
        _formal_table_metadata(
            manifest=manifest,
            selection=selection,
            model_lock=model_lock,
            config_sha256={
                method: run_config_sha256(config) for method, config in configs.items()
            },
            source_evidence_sha256=source_evidence_sha256,
            target_reference_sha256=target_reference.sha256,
        )
    )
    output = Path(args.output)
    if output.exists():
        existing = _load_formal_table(
            output,
            manifest=manifest,
            selection=selection,
            model_lock=model_lock,
            target_reference=target_reference,
        )
        if existing.schema.metadata == table.schema.metadata and existing.equals(table):
            print(output)
            return 0
        raise RuntimeError("existing speed-study table differs from evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, output)
    print(output)
    return 0


def _collect_onlinespec_study(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    _assert_onlinespec_study(manifest, selection, lock)
    configs = {
        "static": _load_bound_run_config(args.static_config),
        "onlinespec_ogd": _load_bound_run_config(args.ogd_config),
        "onlinespec_opt": _load_bound_run_config(args.opt_config),
        "onlinespec_ens": _load_bound_run_config(args.ens_config),
    }
    locked = {model.model_id: model.revision for model in lock.models}
    for method, config in configs.items():
        if config.runtime.sampling_profile_sha256 != selection.sampling_profile_sha256:
            raise ValueError("OnlineSPEC config sampling identity mismatch")
        if config.runtime.execution_policy_sha256 != manifest.execution_policy_sha256:
            raise ValueError("OnlineSPEC config execution-policy identity mismatch")
        if (
            locked.get(config.model.target) != config.model.target_revision
            or locked.get(config.model.drafter) != config.model.drafter_revision
        ):
            raise ValueError("OnlineSPEC config model-lock identity mismatch")
        _assert_onlinespec_config(config, method=method, selection=selection)
    config_hashes = {
        method: run_config_sha256(config) for method, config in configs.items()
    }
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=lock,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    target_revision = next(
        model.revision for model in lock.models if model.model_id == "Qwen/Qwen3-8B"
    )
    performance, evidence_sha256 = collect_onlinespec_performance(
        evidence_root=args.evidence_root,
        manifest_sha256=manifest.sha256,
        config_sha256=config_hashes,
        concurrency=selection.selected_concurrency,
        target_reference=target_reference,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        execution_policy_sha256=manifest.execution_policy_sha256,
        target_revision=target_revision,
    )
    table = _concat_evidence_tables(performance)
    metadata = {
        b"lightcone_schema_version": b"2",
        b"lightcone_study": b"onlinespec-clean-room-baseline",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": selection.sampling_profile_sha256.encode(),
        b"lightcone_execution_policy_sha256": (
            manifest.execution_policy_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_config_set_sha256": _canonical_sha256(config_hashes).encode(),
        b"lightcone_source_evidence_sha256": evidence_sha256.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    table = table.replace_schema_metadata(metadata)
    output = Path(args.output)
    if output.exists():
        existing = pq.read_table(output)
        if existing.schema.metadata == metadata and existing.equals(table):
            print(output)
            return 0
        raise RuntimeError("existing OnlineSPEC table differs from evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, output)
    print(output)
    return 0


def _render_runtime(args: argparse.Namespace) -> int:
    selection = SelectionArtifact.load(args.selection)
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    launches = render_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    print(Path(args.output_root).resolve() / "launch-plan.json")
    if len(launches) != 3:
        raise AssertionError("runtime renderer did not create three method slices")
    if len({launch.base_url for launch in launches}) != 1 or not all(
        launch.exclusive_device for launch in launches
    ):
        raise AssertionError("formal method slices must share one exclusive endpoint")
    return 0


def _render_onlinespec_runtime(args: argparse.Namespace) -> int:
    selection = OnlineSpecSelection.load(args.selection)
    launches = render_onlinespec_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=ModelLock.load(args.model_lock),
        model_roots=_load_bound_json(args.model_roots),
        sampling_profile=SamplingProfile.load(args.sampling_profile),
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != list(ONLINE_SPEC_STUDY_METHODS):
        raise AssertionError("OnlineSPEC runtime plan has incomplete method coverage")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_onlinespec_tuning_runtime(args: argparse.Namespace) -> int:
    grid = {candidate.candidate_id: candidate for candidate in onlinespec_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("OnlineSPEC candidate ID is outside the registered grid")
    launches = render_onlinespec_tuning_runtime_plan(
        output_root=args.output_root,
        candidate=candidate,
        concurrency=args.concurrency,
        model_lock=ModelLock.load(args.model_lock),
        model_roots=_load_bound_json(args.model_roots),
        sampling_profile=SamplingProfile.load(args.sampling_profile),
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != ["static", candidate.method]:
        raise AssertionError("OnlineSPEC tuning plan is not paired to Static")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_static_load_runtime(args: argparse.Namespace) -> int:
    if args.concurrency not in {1, 2, 4, 8, 16, 32, 48}:
        raise ValueError("Static concurrency is outside the registered load grid")
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.purpose != "controlled":
        raise ValueError("Static load screen requires the controlled sampling profile")
    launches = render_static_load_runtime_plan(
        output_root=args.output_root,
        concurrency=args.concurrency,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if (
        len(launches) != 1
        or launches[0].method != "static"
        or launches[0].adaptation_config is not None
        or launches[0].telemetry_path is not None
        or "--speculative-adaptation-config" in launches[0].argv
    ):
        raise AssertionError("Static load renderer allocated an adaptation path")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_target_only_runtime(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("Target-only concurrency must be positive")
    launches = render_target_only_runtime_plan(
        output_root=args.output_root,
        concurrency=args.concurrency,
        model_lock=ModelLock.load(args.model_lock),
        model_roots=_load_bound_json(args.model_roots),
        sampling_profile=SamplingProfile.load(args.sampling_profile),
        sglang_checkout=args.sglang_checkout,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    launch = launches[0]
    if (
        launch.method != "target_only"
        or launch.adaptation_config is not None
        or launch.telemetry_path is not None
        or any("speculative" in argument for argument in launch.argv)
        or any("draft" in argument for argument in launch.argv)
    ):
        raise AssertionError("Target-only renderer allocated speculative state")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_tuning_runtime(args: argparse.Namespace) -> int:
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("candidate ID is outside the registered tuning grid")
    if args.concurrency not in {1, 2, 4, 8, 16, 32, 48}:
        raise ValueError("tuning concurrency is outside the registered load grid")
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    launches = render_tuning_runtime_plan(
        output_root=args.output_root,
        candidate=candidate,
        concurrency=args.concurrency,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != ["tts", "l0"] or len(
        {launch.base_url for launch in launches}
    ) != 1:
        raise AssertionError("tuning runtime must contain two adapted slices")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_replication_runtime(args: argparse.Namespace) -> int:
    selection = SelectionArtifact.load(args.selection)
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    phase = (
        "natural_task_replication"
        if args.phase == "natural"
        else "independent_profiler"
    )
    if phase == "natural_task_replication" and sampling.purpose != "natural":
        raise ValueError("natural runtime requires an EOS-enabled sampling profile")
    if phase == "independent_profiler" and sampling.purpose != "controlled":
        raise ValueError("profiler runtime requires the controlled sampling profile")
    launches = render_replication_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        phase=phase,
        host=args.host,
        first_port=args.first_port,
    )
    if len(launches) != 3 or not all(launch.exclusive_device for launch in launches):
        raise AssertionError("replication runtime slices are not exclusive")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _run_natural_slice(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to another model lock")
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.purpose != "natural" or sampling.ignore_eos:
        raise ValueError("natural side table requires the EOS-enabled profile")
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling,
    )
    assert_confirmation_slice_config(
        config,
        method=args.method,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    samples = load_natural_prompts(
        args.dataset,
        revision=args.dataset_revision,
        split=args.split,
        limit=32,
    )
    adaptation_group_id = (
        "natural-static"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    paths = run_natural_replication_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        dataset_name=args.dataset,
        samples=samples,
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=adaptation_group_id,
        sampling_profile=sampling,
        model_pair=manifest.model_pair,
        warmup=not args.no_warmup,
    )
    print(f"completed natural {args.dataset}/{args.method}: {len(paths)} files")
    return 0


def _build_profiler_plan(args: argparse.Namespace) -> int:
    source = _load_bound_json(args.launch_plan)
    if (
        source.get("schema_version") != 2
        or source.get("phase") != "independent_profiler"
        or source.get("execution_mode") != "sequential_exclusive_device"
        or source.get("patched_sglang_tree") != PINNED_SGLANG_TREE
        or source.get("execution_policy_sha256") != ControlledExecutionPolicy().sha256
    ):
        raise ValueError("profiler requires an independent-profiler launch plan")
    verify_patched_checkout(str(source.get("sglang_checkout", "")))
    servers = source.get("servers")
    if not isinstance(servers, list):
        raise TypeError("profiler launch plan lacks server slices")
    matching = [row for row in servers if row.get("method") == args.method]
    if len(matching) != 1 or matching[0].get("exclusive_device") is not True:
        raise ValueError("profiler method slice is missing or not exclusive")
    workload = list(args.workload_argv)
    if workload and workload[0] == "--":
        workload = workload[1:]
    if not workload or not all(isinstance(value, str) and value for value in workload):
        raise ValueError("profiler plan requires an explicit workload argv after --")
    trace_root = Path(args.trace_root).resolve()
    server_argv = matching[0].get("argv")
    if not isinstance(server_argv, list) or not server_argv:
        raise ValueError("profiler server argv is missing")
    artifact = {
        "schema_version": 2,
        "phase": "independent_profiler",
        "method": args.method,
        "launch_plan_sha256": _canonical_sha256(source),
        "exclusive_device": True,
        "profile_launch_argv": [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--force-overwrite=false",
            "--output",
            str(trace_root / args.method),
            *server_argv,
        ],
        "workload_argv": workload,
        "device_monitor_argv": [
            "nvidia-smi",
            "dmon",
            "-s",
            "puctm",
            "-o",
            "DT",
        ],
        "headline_evidence_forbidden": True,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _build_confirmation_queue(args: argparse.Namespace) -> int:
    manifest, selection, model_lock, sampling = _confirmation_inputs(args)
    plan_path = Path(args.launch_plan).resolve()
    plan = _load_bound_json(plan_path)
    if (
        plan.get("schema_version") != 2
        or plan.get("execution_mode") != "sequential_exclusive_device"
        or plan.get("selection_sha256") != selection.sha256
        or plan.get("model_lock_sha256") != model_lock.sha256
        or plan.get("sampling_profile_sha256") != sampling.sha256
        or plan.get("execution_policy_sha256") != manifest.execution_policy_sha256
        or plan.get("patched_sglang_tree") != PINNED_SGLANG_TREE
    ):
        raise ValueError("launch plan identity does not match the speed study")
    verify_patched_checkout(str(plan.get("sglang_checkout", "")))
    rows = plan.get("servers")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("launch plan must contain exactly three method slices")
    servers: dict[str, dict] = {}
    configs: dict[str, RunConfig] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("launch plan server entry is not an object")
        method = row.get("method")
        if method not in {"static", "tts", "l0"} or method in servers:
            raise ValueError("launch plan has invalid or duplicate methods")
        if row.get("exclusive_device") is not True:
            raise ValueError("every formal slice must own the GPU exclusively")
        argv = row.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(value, str) and value for value in argv
        ):
            raise ValueError("launch argv must be a non-empty string vector")
        config_path = Path(str(row.get("run_config"))).resolve()
        config = _load_bound_run_config(config_path)
        _assert_locked_config(
            config,
            model_lock=model_lock,
            sampling_profile=sampling,
        )
        servers[method] = {**row, "run_config": str(config_path)}
        configs[method] = config
    if len({str(row["base_url"]) for row in servers.values()}) != 1:
        raise ValueError("sequential method slices must reuse one endpoint")
    assert_matched_confirmation_configs(
        configs,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    adaptation = configs["tts"].adaptation
    if adaptation is None:
        raise AssertionError("TTS launch lacks adaptation identity")
    common = [
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--selection",
        str(Path(args.selection).resolve()),
        "--model-lock",
        str(Path(args.model_lock).resolve()),
        "--sampling-profile",
        str(Path(args.sampling_profile).resolve()),
        "--adaptation-group-id",
        adaptation.adaptation_group_id,
        "--output-root",
        str(Path(args.evidence_root).resolve()),
    ]
    jobs: list[dict] = []
    ordinal = 0
    for block in confirmation_blocks(manifest.confirmation_schedule_seed):
        for method in block.method_order:
            server = servers[method]
            jobs.append(
                {
                    "ordinal": ordinal,
                    "block": block.block,
                    "method": method,
                    "launch_argv": server["argv"],
                    "run_argv": [
                        "lightcone-spec",
                        "run-confirmation",
                        *common,
                        "--config",
                        server["run_config"],
                        "--url",
                        str(server["base_url"]),
                        "--method",
                        method,
                        "--block",
                        str(block.block),
                    ],
                    "requires_clean_server_start": True,
                    "requires_server_exit_after": True,
                }
            )
            ordinal += 1
    artifact = {
        "schema_version": 2,
        "execution_mode": "sequential_exclusive_device",
        "manifest_sha256": manifest.sha256,
        "selection_sha256": selection.sha256,
        "model_lock_sha256": model_lock.sha256,
        "sampling_profile_sha256": sampling.sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "launch_plan_sha256": _canonical_sha256(plan),
        "schedule_seed": manifest.confirmation_schedule_seed,
        "jobs": jobs,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _build_onlinespec_queue(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    _assert_onlinespec_study(manifest, selection, lock, sampling)
    plan = _load_bound_json(args.launch_plan)
    if (
        plan.get("schema_version") != 2
        or plan.get("phase") != "onlinespec_paired_confirmation"
        or plan.get("selection_sha256") != selection.sha256
        or plan.get("model_lock_sha256") != lock.sha256
        or plan.get("sampling_profile_sha256") != sampling.sha256
        or plan.get("execution_policy_sha256") != manifest.execution_policy_sha256
        or plan.get("patched_sglang_tree") != PINNED_SGLANG_TREE
    ):
        raise ValueError("OnlineSPEC launch plan identity mismatch")
    verify_patched_checkout(str(plan.get("sglang_checkout", "")))
    raw_servers = plan.get("servers")
    if not isinstance(raw_servers, list) or len(raw_servers) != 4:
        raise ValueError("OnlineSPEC launch plan requires four method slices")
    servers = {}
    for row in raw_servers:
        if (
            not isinstance(row, dict)
            or row.get("method") not in ONLINE_SPEC_STUDY_METHODS
        ):
            raise ValueError("OnlineSPEC launch plan contains an invalid server")
        method = str(row["method"])
        if method in servers or row.get("exclusive_device") is not True:
            raise ValueError("OnlineSPEC servers must be unique and exclusive")
        config = _load_bound_run_config(row["run_config"])
        _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
        _assert_onlinespec_config(config, method=method, selection=selection)
        servers[method] = {**row, "config": config}
    if (
        set(servers) != set(ONLINE_SPEC_STUDY_METHODS)
        or len({str(row["base_url"]) for row in servers.values()}) != 1
    ):
        raise ValueError("OnlineSPEC servers do not share one exclusive endpoint")
    group_ids = {
        row["config"].adaptation.adaptation_group_id
        for method, row in servers.items()
        if method != "static"
    }
    if len(group_ids) != 1:
        raise ValueError("OnlineSPEC configs do not share one cohort group")
    common = [
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--selection",
        str(Path(args.selection).resolve()),
        "--model-lock",
        str(Path(args.model_lock).resolve()),
        "--sampling-profile",
        str(Path(args.sampling_profile).resolve()),
        "--adaptation-group-id",
        next(iter(group_ids)),
        "--output-root",
        str(Path(args.evidence_root).resolve()),
    ]
    jobs = []
    ordinal = 0
    for block in onlinespec_blocks(manifest.confirmation_schedule_seed):
        for method in block.method_order:
            server = servers[method]
            jobs.append(
                {
                    "ordinal": ordinal,
                    "block": block.block,
                    "method": method,
                    "launch_argv": server["argv"],
                    "run_argv": [
                        "lightcone-spec",
                        "run-onlinespec-confirmation",
                        *common,
                        "--config",
                        str(Path(server["run_config"]).resolve()),
                        "--url",
                        str(server["base_url"]),
                        "--method",
                        method,
                        "--block",
                        str(block.block),
                    ],
                    "requires_clean_server_start": True,
                    "requires_server_exit_after": True,
                }
            )
            ordinal += 1
    artifact = {
        "schema_version": 2,
        "execution_mode": "sequential_exclusive_device",
        "study": "onlinespec-clean-room-baseline",
        "manifest_sha256": manifest.sha256,
        "selection_sha256": selection.sha256,
        "model_lock_sha256": lock.sha256,
        "sampling_profile_sha256": sampling.sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "launch_plan_sha256": _canonical_sha256(plan),
        "schedule_seed": manifest.confirmation_schedule_seed,
        "jobs": jobs,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


_ATTESTATION_DOCTOR_CHECKS = frozenset(
    {
        "compatibility_manifest",
        "project_patch_binding",
        "project_sglang_roots_distinct",
        "project_source_tree",
        "python",
        "linux_host",
        "torch",
        "triton",
        "flashinfer",
        "flashinfer_cuda_flavor",
        "cuda_build",
        "cuda_toolkit",
        "torch_cuda_visibility",
        "driver",
        "gpu_count",
        "gpu_identity",
        "gpu_memory",
        "gpu_topology",
        "compiler",
        "disk",
        "network",
        "sglang_commit_lineage",
        "sglang_tree",
        "sglang_import",
    }
)


def _validate_attestation_doctor(hardware: object, *, label: str) -> dict:
    def reject(reason: str) -> NoReturn:
        raise ValueError(
            f"{label} attestation requires a complete PASS doctor report: {reason}"
        )

    if not isinstance(hardware, dict) or hardware.get("schema_version") != 1:
        reject("doctor schema-v1 object is required")
    readiness = hardware.get("readiness")
    compatibility = hardware.get("compatibility")
    if hardware.get("status") != "PASS":
        reject("top-level readiness is not PASS")
    if not isinstance(readiness, dict) or readiness.get("status") != "PASS":
        reject("readiness.status is not PASS")
    if not isinstance(compatibility, dict) or compatibility.get("status") != "PASS":
        reject("compatibility.status is not PASS")

    runtime_manifest = hardware.get("runtime_manifest")
    if (
        not isinstance(runtime_manifest, dict)
        or runtime_manifest.get("valid") is not True
    ):
        reject("runtime compatibility manifest is not valid")
    manifest_sha256 = runtime_manifest.get("sha256")
    if (
        not _is_lower_sha256(manifest_sha256)
        or runtime_manifest.get("sidecar_sha256") != manifest_sha256
        or runtime_manifest.get("error") is not None
        or compatibility.get("manifest_sha256") != manifest_sha256
    ):
        reject("runtime compatibility manifest digests are inconsistent")

    checks = hardware.get("checks")
    if not isinstance(checks, dict):
        reject("doctor checks are missing")
    missing_checks = sorted(_ATTESTATION_DOCTOR_CHECKS - checks.keys())
    if missing_checks:
        reject(f"required doctor checks are missing: {', '.join(missing_checks)}")
    if any(
        not isinstance(check, dict) or check.get("status") != "PASS"
        for check in checks.values()
    ):
        reject("every doctor check must be PASS")
    if (
        readiness.get("pass_count") != len(checks)
        or readiness.get("fail_count") != 0
        or readiness.get("unknown_count") != 0
    ):
        reject("doctor readiness counters do not match its checks")

    roots = hardware.get("roots")
    if not isinstance(roots, dict):
        reject("project and patched-SGLang roots are missing")
    project_root = roots.get("project")
    sglang_root = roots.get("patched_sglang")
    if (
        roots.get("distinct") is not True
        or not isinstance(project_root, str)
        or not project_root
        or not isinstance(sglang_root, str)
        or not sglang_root
        or project_root == sglang_root
    ):
        reject("project and patched-SGLang roots must be distinct")

    source_tree = hardware.get("source_tree")
    source_head = source_tree.get("head") if isinstance(source_tree, dict) else None
    if (
        not isinstance(source_tree, dict)
        or source_tree.get("path") != sglang_root
        or source_tree.get("is_git_checkout") is not True
        or source_tree.get("root_matches_toplevel") is not True
        or not isinstance(source_head, str)
        or len(source_head) != 40
        or any(char not in "0123456789abcdef" for char in source_head)
        or source_tree.get("tree") != PINNED_SGLANG_TREE
        or source_tree.get("dirty") is not False
        or source_tree.get("pinned_ancestor") is not True
        or source_tree.get("patch_commits") != PINNED_SGLANG_PATCH_COUNT
    ):
        reject("patched SGLang source-tree identity is not exact")
    if (
        compatibility.get("sglang_commit") != PINNED_SGLANG_COMMIT
        or compatibility.get("sglang_tree") != PINNED_SGLANG_TREE
        or compatibility.get("patch_count") != PINNED_SGLANG_PATCH_COUNT
        or compatibility.get("python_supported") is not True
        or compatibility.get("single_node_only") is not True
        or compatibility.get("multi_node_supported") is not False
    ):
        reject("compatibility identities do not match the release")

    commands = hardware.get("commands")
    nvidia = commands.get("nvidia_smi") if isinstance(commands, dict) else None
    gpu = hardware.get("gpu")
    if (
        not isinstance(nvidia, str)
        or not nvidia.strip()
        or not isinstance(gpu, dict)
        or gpu.get("gpu_pool_visible") is not True
        or not isinstance(gpu.get("visible_gpu_count"), int)
        or isinstance(gpu.get("visible_gpu_count"), bool)
        or gpu["visible_gpu_count"] < 1
    ):
        reject("successful same-host GPU-pool nvidia-smi evidence is required")
    return hardware


def _attest(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    revisions = {model.model_id: model.revision for model in model_lock.models}
    target_revision = revisions.get("Qwen/Qwen3-8B")
    drafter_revision = revisions.get("z-lab/Qwen3-8B-DFlash-b16")
    if target_revision is None or drafter_revision is None:
        raise ValueError("model lock lacks the formal Qwen3-8B/DFlash pair")
    hardware = _validate_attestation_doctor(
        json.loads(Path(args.doctor_json).read_text(encoding="utf-8")),
        label="GPU",
    )
    if _TRUSTED_HARDWARE_ATTESTER_ID is None:
        raise _trusted_attester_unavailable("legacy GPU attestation")
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=model_lock,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    _load_formal_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        model_lock=model_lock,
        target_reference=target_reference,
    )
    if target_reference.hardware_sha256 != _canonical_sha256(hardware):
        raise ValueError("target reference belongs to a different GPU report")
    attestation = GpuEvidenceAttestation(
        schema_version=2,
        status="MEASURED",
        manifest_sha256=manifest.sha256,
        selection_sha256=selection.sha256,
        model_lock_sha256=model_lock.sha256,
        performance_sha256=evidence_files_sha256((args.performance,)),
        target_reference_sha256=target_reference.sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        target_revision=target_revision,
        drafter_revision=drafter_revision,
        hardware_sha256=_canonical_sha256(hardware),
        methods=manifest.methods,
        repetitions=manifest.confirmation_repetitions,
        context_start=manifest.formal_context_start,
        context_limit=manifest.safe_context_limit,
    )
    attestation.write(args.output)
    print(attestation.sha256)
    return 0


def _onlinespec_table(
    path: str | Path,
    *,
    manifest: OnlineSpecManifest,
    selection: OnlineSpecSelection,
    lock: ModelLock,
    target_reference: GreedyTargetReference,
) -> pa.Table:
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    expected = {
        b"lightcone_schema_version": b"2",
        b"lightcone_study": b"onlinespec-clean-room-baseline",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": selection.sampling_profile_sha256.encode(),
        b"lightcone_execution_policy_sha256": (
            manifest.execution_policy_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("OnlineSPEC table identity metadata mismatch")
    return table


def _attest_onlinespec(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    _assert_onlinespec_study(manifest, selection, lock)
    hardware = _validate_attestation_doctor(
        json.loads(Path(args.doctor_json).read_text(encoding="utf-8")),
        label="OnlineSPEC GPU",
    )
    if _TRUSTED_HARDWARE_ATTESTER_ID is None:
        raise _trusted_attester_unavailable("legacy OnlineSPEC GPU attestation")
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=lock,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    _onlinespec_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        lock=lock,
        target_reference=target_reference,
    )
    if target_reference.hardware_sha256 != _canonical_sha256(hardware):
        raise ValueError("target reference belongs to a different GPU report")
    attestation = OnlineSpecGpuAttestation(
        schema_version=2,
        status="MEASURED",
        manifest_sha256=manifest.sha256,
        selection_sha256=selection.sha256,
        model_lock_sha256=lock.sha256,
        performance_sha256=evidence_files_sha256((args.performance,)),
        target_reference_sha256=target_reference.sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        hardware_sha256=_canonical_sha256(hardware),
        methods=manifest.methods,
        repetitions=manifest.confirmation_repetitions,
    )
    attestation.write(args.output)
    print(attestation.sha256)
    return 0


def _analyze(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    if args.attestation and _TRUSTED_HARDWARE_ATTESTER_ID is None:
        raise _trusted_attester_unavailable("legacy analysis attestation")
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=model_lock,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    table = _load_formal_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        model_lock=model_lock,
        target_reference=target_reference,
    )
    evidence_state = "UNMEASURED"
    evidence_sha256 = None
    if args.attestation:
        attestation = GpuEvidenceAttestation.load(args.attestation)
        attestation.verify_performance((args.performance,))
        attestation.verify_target_reference(target_reference)
        if attestation.manifest_sha256 != manifest.sha256:
            raise ValueError("attestation manifest identity mismatch")
        if attestation.selection_sha256 != selection.sha256:
            raise ValueError("attestation selection identity mismatch")
        if attestation.model_lock_sha256 != model_lock.sha256:
            raise ValueError("attestation model-lock identity mismatch")
        revisions = {model.model_id: model.revision for model in model_lock.models}
        if attestation.target_revision != revisions.get(
            "Qwen/Qwen3-8B"
        ) or attestation.drafter_revision != revisions.get("z-lab/Qwen3-8B-DFlash-b16"):
            raise ValueError("attestation model revisions mismatch")
        evidence_state = "MEASURED"
        evidence_sha256 = attestation.sha256
    gate = evaluate_speed_gate(
        table.to_pylist(),
        seed=args.bootstrap_seed,
        gpu_evidence=evidence_state,
        evidence_sha256=evidence_sha256,
    )
    _write_json(
        args.output,
        {
            **asdict(gate),
            "selection_protocol": selection.selection_protocol,
            "optimized_grid_claim": (
                selection.selection_protocol == "successive_halving"
            ),
        },
    )
    return 0 if gate.passed else 42


def _analyze_onlinespec(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    _assert_onlinespec_study(manifest, selection, lock)
    if args.attestation and _TRUSTED_HARDWARE_ATTESTER_ID is None:
        raise _trusted_attester_unavailable("legacy OnlineSPEC analysis attestation")
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=lock,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    table = _onlinespec_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        lock=lock,
        target_reference=target_reference,
    )
    evidence = "UNMEASURED"
    attestation_sha256 = None
    if args.attestation:
        attestation = OnlineSpecGpuAttestation.load(args.attestation)
        if (
            attestation.manifest_sha256 != manifest.sha256
            or attestation.selection_sha256 != selection.sha256
            or attestation.model_lock_sha256 != lock.sha256
            or attestation.performance_sha256
            != evidence_files_sha256((args.performance,))
            or attestation.target_reference_sha256 != target_reference.sha256
        ):
            raise ValueError("OnlineSPEC attestation does not bind this table")
        evidence = "MEASURED"
        attestation_sha256 = attestation.sha256
    comparisons = compare_onlinespec(table.to_pylist(), seed=args.bootstrap_seed)
    safety_pass = all(comparison.safety_pass for comparison in comparisons)
    acceleration_pass = any(comparison.acceleration_pass for comparison in comparisons)
    status = (
        "UNMEASURED"
        if evidence == "UNMEASURED"
        else "PASS"
        if safety_pass and acceleration_pass
        else "BLOCKED"
    )
    _write_json(
        args.output,
        {
            "schema_version": 2,
            "study": "onlinespec-clean-room-baseline",
            "gpu_evidence": evidence,
            "status": status,
            "attestation_sha256": attestation_sha256,
            "core_speed_gate_affected": False,
            "safety_pass": safety_pass,
            "at_least_one_acceleration_pass": acceleration_pass,
            "passing_methods": [
                comparison.method for comparison in comparisons if comparison.passed
            ],
            "selection_protocol": selection.selection_protocol,
            "optimized_grid_claim": (
                selection.selection_protocol == "successive_halving"
            ),
            "comparisons": [asdict(row) for row in comparisons],
        },
    )
    return 0 if status == "PASS" else 42


def _build_industrial_registry(args: argparse.Namespace) -> int:
    logical_slots = tuple(args.logical_gpu_slot)
    if (
        not logical_slots
        or len(set(logical_slots)) != len(logical_slots)
        or any(
            not isinstance(slot, str)
            or not slot.strip()
            or "\n" in slot
            or "\r" in slot
            for slot in logical_slots
        )
    ):
        raise ValueError("logical GPU rank slots must be one or more unique names")
    registry = build_industrial_registry(
        gpu_uuids=logical_slots,
        base_port=args.base_port,
        cache_root=args.cache_root,
        evidence_root=args.evidence_root,
        seed=args.seed,
    )
    artifact = _industrial_registry_artifact(
        registry,
        base_port=args.base_port,
        cache_root=args.cache_root,
        evidence_root=args.evidence_root,
        seed=args.seed,
    )
    _write_json(args.output, artifact)
    print(registry.sha256)
    return 0


def _validate_e2_final_seal_authority(
    *,
    registry: ExperimentRegistry,
    reduction,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
    direct_dependency_receipt_sha256: str | None,
    completed_cell_ids: tuple[str, ...],
    completed_cells_path: str,
    locked_output_paths: dict[str, str],
) -> None:
    """Require the exact raw halving_3 candidate artifact before E2 sealing."""

    evidence = reduction.stage_evidence
    completed_value = _load_bound_json(completed_cells_path)
    if not isinstance(completed_value, dict) or not isinstance(
        completed_value.get("rows"), list
    ):
        raise TypeError("E2 completed-cell authority is malformed")
    measured_rows = tuple(
        row
        for row in completed_value["rows"]
        if isinstance(row, dict) and row.get("status") == "MEASURED"
    )
    terminal_receipts = tuple(
        sorted(str(row.get("terminal_receipt_sha256")) for row in measured_rows)
    )
    budget_observations = tuple(
        sorted(
            {
                str(row["budget_observation_sha256"])
                for row in measured_rows
                if row.get("budget_observation_sha256") is not None
            }
        )
    )
    if (
        reduction.registry_sha256 != registry.sha256
        or reduction.stage_index != 3
        or reduction.runtime_sha256 != runtime_sha256
        or reduction.split_sha256 != split_sha256
        or evidence.inventory_sha256 != inventory.sha256
        or evidence.inventory_source_receipt_sha256 != inventory.source_receipt_sha256
        or evidence.fixed_instance_gpu_count != len(inventory.devices)
        or evidence.inventory_host_id
        != (inventory.host_ids[0] if len(inventory.host_ids) == 1 else None)
        or reduction.activation.plan.dependency_receipt_sha256
        != direct_dependency_receipt_sha256
        or tuple(sorted(completed_cell_ids))
        != reduction.survivor_receipt.completed_stage_cell_ids
        or terminal_receipts != evidence.terminal_receipt_sha256s
        or budget_observations != evidence.budget_observation_sha256s
    ):
        raise ValueError("E2 final-stage authority differs from the sealing inputs")
    if set(locked_output_paths) != {"dflash_recipe"}:
        raise ValueError("E2 seal requires exactly one dflash_recipe=PATH output")
    expected = materialize_e2_final_recipe(registry, reduction)
    actual = e2_final_recipe_artifact_from_dict(
        _load_bound_json(locked_output_paths["dflash_recipe"])
    )
    if actual != expected:
        raise ValueError("dflash_recipe is not the raw halving_3 final candidate")


def _seal_industrial_stage(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    registry.definition(args.experiment)
    if _TRUSTED_HARDWARE_ATTESTER_ID is None:
        artifact = {
            "schema_version": 1,
            "kind": "industrial_stage_seal_decision",
            "status": "BLOCKED",
            "gpu_evidence": "UNMEASURED",
            "reason_code": "trusted_hardware_attester_unavailable",
            "registry_sha256": registry.sha256,
            "experiment": args.experiment,
            "trusted_attester_id": None,
        }
        _write_json(args.output, artifact)
        print(_canonical_sha256(artifact))
        return 42
    dependencies = _load_industrial_receipts(args.dependency_receipt)
    inventory = _load_gpu_inventory(args.inventory)
    activation_artifact = _load_stage_activation_plan(args.activation_plan)
    family_activations = _load_family_activations(args.family_activation)
    family_power_reductions = _load_family_power_reductions(args.family_power_plan)
    dependency_by_experiment = registry.validate_receipts(dependencies)
    definition = registry.definition(args.experiment)
    direct_dependency_sha256 = (
        None
        if not definition.dependencies
        else dependency_by_experiment[definition.dependencies[-1]].sha256
    )
    runtime_sha256 = _artifact_sha256(args.runtime_artifact)
    split_contract = _load_bound_json(args.split_artifact)
    split_sha256 = _canonical_sha256(split_contract)
    completed_cell_ids, completed_sha256 = _completed_industrial_cells(
        args.completed_cells,
        registry,
        experiment=args.experiment,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        split_contract=split_contract,
        require_industrial_contract=True,
        direct_dependency_receipt_sha256=direct_dependency_sha256,
        activation_artifact=activation_artifact,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
        require_stage_sealable=True,
        inventory=inventory,
    )
    if completed_sha256 is None:
        raise ValueError("stage sealing requires content-bound completed-cell evidence")
    locked_output_paths = _parse_locked_output_paths(args.locked_output)
    if args.experiment == "E2":
        if args.e2_final_stage_manifest is None:
            raise ValueError("E2 seal requires --e2-final-stage-manifest")
        _validate_e2_final_seal_authority(
            registry=registry,
            reduction=_load_e2_stage_manifest(args.e2_final_stage_manifest),
            inventory=inventory,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            direct_dependency_receipt_sha256=direct_dependency_sha256,
            completed_cell_ids=completed_cell_ids,
            completed_cells_path=args.completed_cells,
            locked_output_paths=locked_output_paths,
        )
    elif args.e2_final_stage_manifest is not None:
        raise ValueError("E2 final-stage authority cannot seal another experiment")
    receipt = registry.make_receipt(
        args.experiment,
        {name: _artifact_sha256(path) for name, path in locked_output_paths.items()},
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        completed_cells_sha256=completed_sha256,
        dependencies=dependencies,
    )
    _write_json(args.output, receipt.to_dict())
    print(receipt.sha256)
    return 0


def _completed_industrial_cells(
    path: str | None,
    registry: ExperimentRegistry,
    *,
    experiment: str | None = None,
    runtime_sha256: str | None = None,
    split_sha256: str | None = None,
    split_contract: object | None = None,
    require_industrial_contract: bool = False,
    direct_dependency_receipt_sha256: str | None = None,
    activation_artifact=None,
    family_activations=(),
    family_power_reductions=(),
    require_stage_sealable: bool = False,
    inventory: GpuInventory | None = None,
) -> tuple[tuple[str, ...], str | None]:
    if path is None:
        return (), None
    value = _load_bound_json(path)
    if not isinstance(value, dict):
        raise TypeError("completed-cell artifact must be an object")
    schema_version = value.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in {1, 2, 3, 4}
        or value.get("kind") != "industrial_completed_cells"
        or value.get("registry_sha256") != registry.sha256
    ):
        raise ValueError("completed-cell artifact identity mismatch")
    strict = schema_version in {2, 3, 4}
    formal = schema_version == 4
    if require_industrial_contract and not formal:
        raise ValueError(
            "stage sealing requires a schema-version-4 inventory-bound contract"
        )
    if formal and inventory is None:
        raise ValueError("formal completion requires a bound GPU inventory")

    known = {cell.cell_id: cell for cell in registry.cells}
    contract_by_cell: dict[str, dict] = {}
    stage: str | None = None
    activated: tuple[str, ...] = ()
    dispositions: dict[str, dict[str, str]] = {}
    activation_binding: dict[str, object] | None = None
    if strict:
        stage = value.get("experiment")
        embedded_split = value.get("split_contract")
        if (
            not isinstance(stage, str)
            or not stage
            or value.get("runtime_sha256") is None
            or not _is_lower_sha256(value.get("runtime_sha256"))
            or not _is_lower_sha256(value.get("split_sha256"))
            or not isinstance(embedded_split, dict)
            or _canonical_sha256(embedded_split) != value.get("split_sha256")
        ):
            raise ValueError("industrial completion contract identity is incomplete")
        if experiment is not None and stage != experiment:
            raise ValueError("completed cells belong to another industrial stage")
        if runtime_sha256 is not None and value["runtime_sha256"] != runtime_sha256:
            raise ValueError("completed cells bind another runtime artifact")
        if split_sha256 is not None and value["split_sha256"] != split_sha256:
            raise ValueError("completed cells bind another locked split")
        if split_contract is not None and embedded_split != split_contract:
            raise ValueError("embedded locked split differs from the sealing artifact")
        if formal:
            expected_completion_fields = {
                "schema_version",
                "kind",
                "registry_sha256",
                "experiment",
                "runtime_sha256",
                "split_sha256",
                "split_contract",
                "activation_binding",
                "inventory_sha256",
                "inventory_source_receipt_sha256",
                "rows",
            }
            if set(value) != expected_completion_fields:
                raise ValueError("formal completion fields differ from schema")
            if inventory is None:  # pragma: no cover - guarded above
                raise RuntimeError("formal completion lost its GPU inventory")
            if (
                value.get("inventory_sha256") != inventory.sha256
                or value.get("inventory_source_receipt_sha256")
                != inventory.source_receipt_sha256
            ):
                raise ValueError("formal completion binds another GPU inventory")
            activated, dispositions, activation_binding = (
                _industrial_completion_activation_contract(
                    registry,
                    experiment=stage,
                    runtime_sha256=value["runtime_sha256"],
                    split_sha256=value["split_sha256"],
                    direct_dependency_receipt_sha256=(direct_dependency_receipt_sha256),
                    activation_artifact=activation_artifact,
                    family_activations=family_activations,
                    family_power_reductions=family_power_reductions,
                    require_stage_sealable=require_stage_sealable,
                )
            )
            if value.get("activation_binding") != activation_binding:
                raise ValueError("completion activation binding is missing or forged")
        if (
            not isinstance(embedded_split.get("schema_version"), int)
            or isinstance(embedded_split.get("schema_version"), bool)
            or embedded_split.get("schema_version") != 1
            or embedded_split.get("kind") != "industrial_locked_split"
            or embedded_split.get("registry_sha256") != registry.sha256
            or embedded_split.get("experiment") != stage
        ):
            raise ValueError("industrial locked-split identity mismatch")
        if formal and set(embedded_split) != {
            "schema_version",
            "kind",
            "registry_sha256",
            "experiment",
            "cells",
        }:
            raise ValueError("formal locked-split fields differ from schema")
        split_cells = embedded_split.get("cells")
        if not isinstance(split_cells, list) or not all(
            isinstance(row, dict) for row in split_cells
        ):
            raise TypeError("industrial locked split lacks cell contracts")
        runnable = {
            cell.cell_id: cell for cell in registry.cells_for(stage) if cell.runnable
        }
        materialized = (
            {cell_id: runnable[cell_id] for cell_id in activated}
            if formal
            else runnable
        )
        for contract in split_cells:
            cell_id = contract.get("cell_id")
            if cell_id not in materialized or cell_id in contract_by_cell:
                raise ValueError("locked split contains an unknown or duplicate cell")
            cell = materialized[str(cell_id)]
            if formal:
                expected_contract_fields = {
                    "cell_id",
                    "request_ids",
                    "expected_request_rows",
                    "expected_round_rows",
                    "expected_update_rows",
                    "expected_performance_rows",
                    "request_ids_sha256",
                    "corpus_sha256",
                    "arrival_trace_sha256",
                    "sampling_profile_sha256",
                    "model_lock_sha256",
                    "patched_sglang_tree",
                    "workload_contract",
                    "rank_config_sha256s",
                    "physical_assignment",
                    "physical_binding_sha256",
                    "topology_receipt_sha256",
                    "experiment_budget_sha256",
                    "experiment_budget",
                    "execution_plan_sha256",
                    "execution_split_sha256",
                }
                if set(contract) != expected_contract_fields:
                    raise ValueError("locked split cell fields differ from schema")
                assignment = _industrial_physical_assignment_from_dict(
                    contract.get("physical_assignment")
                )
                if inventory is None:  # pragma: no cover - guarded above
                    raise RuntimeError("formal completion lost its GPU inventory")
                _validate_assignment_inventory_authority(assignment, inventory)
                try:
                    experiment_budget = experiment_budget_from_dict(
                        contract.get("experiment_budget")
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "locked split contains a forged ExperimentBudget"
                    ) from error
                if (
                    contract.get("physical_binding_sha256") != assignment.sha256
                    or contract.get("experiment_budget_sha256")
                    != assignment.experiment_budget_sha256
                    or contract.get("experiment_budget_sha256")
                    != experiment_budget.sha256
                    or not _is_lower_sha256(contract.get("topology_receipt_sha256"))
                    or not _is_lower_sha256(contract.get("execution_plan_sha256"))
                    or not _is_lower_sha256(contract.get("execution_split_sha256"))
                ):
                    raise ValueError(
                        "locked split physical assignment/budget binding is invalid"
                    )
                if (
                    experiment_budget.cell_id != cell.cell_id
                    or experiment_budget.experiment != cell.identity.experiment
                    or experiment_budget.method != cell.identity.method
                    or experiment_budget.workload_class
                    is not cell.resources.workload_class
                    or experiment_budget.gpu_count != cell.resources.gpu_count
                    or experiment_budget.topology != cell.identity.topology
                    or experiment_budget.measured_gpu_ms is not None
                    or experiment_budget.fixed_instance_billed_gpu_ms
                    != experiment_budget.wall_time.scale(len(inventory.devices))
                ):
                    raise ValueError(
                        "locked ExperimentBudget differs from its registry cell"
                    )
                expected_topology = {
                    "tp1_dp1": (1, 1),
                    "tp2_dp1": (2, 1),
                    "two_replica_tp1_dp2": (1, 2),
                    "two_gpu_host": (1, 2),
                    "two_independent_tp1": (1, 2),
                }.get(cell.identity.topology)
                if expected_topology is None or expected_topology != (
                    assignment.tensor_parallel_size,
                    assignment.data_parallel_size,
                ):
                    raise ValueError(
                        "physical assignment disagrees with registry topology"
                    )
            request_ids = contract.get("request_ids")
            expected_request_rows = contract.get("expected_request_rows")
            if (
                not isinstance(request_ids, list)
                or not request_ids
                or not all(isinstance(item, str) and item for item in request_ids)
                or len(request_ids) != len(set(request_ids))
                or not isinstance(expected_request_rows, int)
                or isinstance(expected_request_rows, bool)
                or expected_request_rows != len(request_ids)
                or contract.get("request_ids_sha256") != _canonical_sha256(request_ids)
            ):
                raise ValueError("locked split request coverage is invalid")
            for name in (
                "corpus_sha256",
                "arrival_trace_sha256",
                "request_ids_sha256",
                "sampling_profile_sha256",
                "model_lock_sha256",
            ):
                if not _is_lower_sha256(contract.get(name)):
                    raise ValueError(f"locked split {name} is invalid")
            rank_config_sha256s = contract.get("rank_config_sha256s")
            if stage == "preflight":
                if rank_config_sha256s is not None:
                    raise ValueError(
                        "preflight split cannot claim serving RunConfig identities"
                    )
            elif (
                not isinstance(rank_config_sha256s, list)
                or len(rank_config_sha256s)
                != (
                    len(assignment.gpu_uuids)
                    if formal
                    else len(cell.resources.gpu_uuids)
                )
                or not all(_is_lower_sha256(item) for item in rank_config_sha256s)
            ):
                raise ValueError(
                    "serving split must bind one rank-config digest per rank"
                )
            if contract.get("patched_sglang_tree") != PINNED_SGLANG_TREE:
                raise ValueError("locked split uses another patched SGLang tree")
            expected_workload = (
                f"industrial_preflight_{cell.identity.method}"
                if stage == "preflight"
                else (
                    f"industrial_{cell.identity.method}"
                    if cell.identity.method in {"target_only", "static"}
                    else "industrial_adapted"
                )
            )
            if contract.get("workload_contract") != expected_workload:
                raise ValueError(
                    "locked split workload contract disagrees with its cell"
                )
            counts = {
                name: contract.get(name)
                for name in (
                    "expected_round_rows",
                    "expected_update_rows",
                    "expected_performance_rows",
                )
            }
            if (
                any(
                    not isinstance(count, int) or isinstance(count, bool) or count < 0
                    for count in counts.values()
                )
                or counts["expected_performance_rows"] < 1
            ):
                raise ValueError("locked split evidence row counts are invalid")
            adapted = cell.identity.method not in {"target_only", "static"}
            requires_rounds = adapted
            requires_updates = adapted
            if (counts["expected_round_rows"] > 0) is not requires_rounds or (
                counts["expected_update_rows"] > 0
            ) is not requires_updates:
                raise ValueError(
                    "locked split table coverage disagrees with the method"
                )
            contract_by_cell[str(cell_id)] = contract
        if set(contract_by_cell) != set(materialized):
            raise ValueError("locked split does not cover every activated stage cell")

    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("completed-cell rows are missing")
    if strict:
        if stage is None:
            raise RuntimeError("strict completion lost its stage identity")
        stage_ids = {cell.cell_id for cell in registry.cells_for(stage)}
        if any(row.get("cell_id") not in stage_ids for row in rows):
            raise ValueError("completed-cell rows cross an industrial stage boundary")
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("cell_id")), []).append(row)
        if set(grouped) != stage_ids:
            raise ValueError(
                "completion outcomes do not cover every declared stage cell"
            )
        for cell_id, cell_rows in grouped.items():
            cell = known[cell_id]
            if formal and cell_id not in set(activated):
                if cell_rows != [dispositions[cell_id]]:
                    raise ValueError(
                        "non-activated cells require one exact immutable disposition"
                    )
                continue
            if cell.runnable:
                expected_world_size = (
                    len(
                        _industrial_physical_assignment_from_dict(
                            contract_by_cell[cell_id]["physical_assignment"]
                        ).gpu_uuids
                    )
                    if formal
                    else len(cell.resources.gpu_uuids)
                )
                if len(cell_rows) != expected_world_size or any(
                    row.get("status") != "MEASURED" for row in cell_rows
                ):
                    raise ValueError(
                        "activated cells require one measured outcome per physical rank"
                    )
                continue
            expected_disposition = (
                dispositions[cell_id]
                if formal
                else {
                    "cell_id": cell_id,
                    "status": cell.status.value,
                    "reason_code": cell.reason_code,
                    "reason": cell.reason,
                }
            )
            if cell_rows != [expected_disposition]:
                raise ValueError(
                    "BLOCKED/N/A cells require one exact immutable outcome row"
                )

    completed: list[str] = []
    rank_coverage: dict[str, set[int]] = {}
    consensus: dict[str, dict[str, object]] = {}
    budget_consensus: dict[str, tuple[object, object, object]] = {}
    nonce_owner: dict[str, str] = {}
    run_owner: dict[str, str] = {}
    for row in rows:
        cell_id = row.get("cell_id")
        if (
            strict
            and cell_id in known
            and (
                not known[str(cell_id)].runnable
                or (formal and cell_id not in set(activated))
            )
        ):
            continue
        evidence_root = row.get("evidence_root")
        run_id = row.get("run_id")
        rank = row.get("rank")
        assignment = None
        row_cell = known.get(str(cell_id))
        physical_world_size = (
            0 if row_cell is None else len(row_cell.resources.gpu_uuids)
        )
        if formal and cell_id in contract_by_cell:
            formal_row_fields = {
                "cell_id",
                "evidence_root",
                "run_id",
                "rank",
                "evidence_sha256",
                "terminal_receipt_sha256",
                "physical_gpu_uuid",
                "physical_binding_sha256",
                "experiment_budget_sha256",
                "budget_observation_status",
                "budget_observation_reason_code",
                "budget_observation_path",
                "budget_observation_sha256",
                "preflight_attestation_path",
                "preflight_attestation_sha256",
                "status",
            }
            if set(row) != formal_row_fields:
                raise ValueError("formal completed-rank fields differ from schema")
            assignment = _industrial_physical_assignment_from_dict(
                contract_by_cell[str(cell_id)]["physical_assignment"]
            )
            physical_world_size = len(assignment.gpu_uuids)
        if (
            not _is_lower_sha256(cell_id)
            or cell_id not in known
            or not isinstance(evidence_root, str)
            or not evidence_root
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 0
            or rank >= physical_world_size
            or not _is_lower_sha256(row.get("evidence_sha256"))
            or not _is_lower_sha256(row.get("terminal_receipt_sha256"))
            or row.get("status") != "MEASURED"
        ):
            raise ValueError(
                "completed cells require known identities and durable measured evidence"
            )
        cell = known[str(cell_id)]
        if formal:
            if assignment is None:
                raise RuntimeError("formal completion lost its physical assignment")
            contract = contract_by_cell[str(cell_id)]
            if (
                row.get("physical_binding_sha256") != assignment.sha256
                or row.get("physical_gpu_uuid") != assignment.gpu_uuids[rank]
                or row.get("experiment_budget_sha256")
                != assignment.experiment_budget_sha256
                or row.get("experiment_budget_sha256")
                != contract["experiment_budget_sha256"]
            ):
                raise ValueError(
                    "completed rank differs from its physical assignment/budget"
                )
        owner = run_owner.setdefault(run_id, str(cell_id))
        if strict and owner != cell_id:
            raise ValueError("industrial run identity is reused across registry cells")
        root = Path(evidence_root)
        if strict and root.resolve() != Path(cell.resources.evidence_root).resolve():
            raise ValueError(
                "completed-cell evidence root differs from its resource claim"
            )
        evidence = load_completed_evidence(root, run_id=run_id, rank=rank)
        if evidence is None:
            raise ValueError("completed cell has no valid terminal evidence receipt")
        receipt_path = root / f"{run_id}.rank{rank}.complete.json"
        if _file_sha256(receipt_path) != row["terminal_receipt_sha256"]:
            raise ValueError("completed-cell terminal receipt digest mismatch")
        try:
            terminal_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("completed-cell terminal receipt is invalid") from error
        if not isinstance(terminal_receipt, dict):
            raise TypeError("completed-cell terminal receipt is malformed")
        if evidence_files_sha256(evidence.values()) != row["evidence_sha256"]:
            raise ValueError("completed-cell evidence digest mismatch")
        run_columns = ["manifest_sha256", "config_sha256", "status"]
        if strict:
            run_columns.extend(
                (
                    "method",
                    "model_pair",
                    "repetition_block",
                    "started_ns",
                    "industrial_cell_id",
                    "rank_config_sha256",
                    "runtime_sha256",
                    "split_sha256",
                    "corpus_sha256",
                    "arrival_trace_sha256",
                    "request_ids_sha256",
                    "sampling_profile_sha256",
                    "model_lock_sha256",
                    "patched_sglang_tree",
                    "run_nonce_sha256",
                    "topology_sha256",
                    "tensor_parallel_size",
                    "data_parallel_size",
                    "world_size",
                    "rank",
                    "expected_request_rows",
                    "expected_round_rows",
                    "expected_update_rows",
                    "expected_performance_rows",
                    "workload_contract",
                    "experiment_budget_sha256",
                    "preflight_attestation_sha256",
                    *_DISABLED_SESSION_RUN_FIELDS,
                )
            )
            run_schema_names = pq.ParquetFile(evidence["run"]).schema_arrow.names
            for field in run_schema_names:
                if field.startswith("session_") and field not in run_columns:
                    run_columns.append(field)
        run = pq.read_table(evidence["run"], columns=run_columns).to_pylist()
        if len(run) != 1:
            raise ValueError("completed evidence must contain exactly one run identity")
        run_row = run[0]
        if any(
            run_row.get(name) != expected
            for name, expected in (
                ("manifest_sha256", registry.sha256),
                ("config_sha256", cell_id),
                ("status", "complete"),
            )
        ):
            raise ValueError("completed evidence is not bound to its registry cell")
        contract: dict | None = None
        topology_sha256: str | None = None
        if strict:
            _validate_disabled_session_run_fields(run_row)
            contract = contract_by_cell[str(cell_id)]
            topology = cell.identity.topology
            if formal:
                if assignment is None:
                    raise RuntimeError("formal completion lost its physical assignment")
                tensor_parallel_size = assignment.tensor_parallel_size
                data_parallel_size = assignment.data_parallel_size
                world_size = len(assignment.gpu_uuids)
                topology_sha256 = _canonical_sha256(
                    {
                        "schema_version": 1,
                        "cell_id": cell_id,
                        "topology": topology,
                        "topology_receipt_sha256": contract["topology_receipt_sha256"],
                        "physical_assignment_sha256": (assignment.assignment_sha256),
                        "physical_binding_sha256": assignment.sha256,
                        "physical_host_id": assignment.host_id,
                        "physical_gpu_uuids": list(assignment.gpu_uuids),
                        "physical_rank_groups": [
                            list(group) for group in assignment.rank_groups
                        ],
                        "physical_ports": list(assignment.ports),
                        "topology_group_ids": [
                            list(group) for group in assignment.topology_group_ids
                        ],
                        "tensor_parallel_size": tensor_parallel_size,
                        "data_parallel_size": data_parallel_size,
                        "world_size": world_size,
                    }
                )
            else:
                tensor_parallel_size = 2 if topology == "tp2_dp1" else 1
                data_parallel_size = 2 if topology == "two_replica_tp1_dp2" else 1
                world_size = len(cell.resources.gpu_uuids)
                topology_sha256 = _canonical_sha256(
                    {
                        "schema_version": 1,
                        "cell_id": cell_id,
                        "topology": topology,
                        "gpu_uuids": list(cell.resources.gpu_uuids),
                        "tensor_parallel_size": tensor_parallel_size,
                        "data_parallel_size": data_parallel_size,
                        "world_size": world_size,
                    }
                )
            expected_run = {
                "method": cell.identity.method,
                "model_pair": cell.identity.model,
                "repetition_block": cell.identity.block,
                "industrial_cell_id": cell_id,
                "rank_config_sha256": (
                    None
                    if stage == "preflight"
                    else contract["rank_config_sha256s"][rank]
                ),
                "runtime_sha256": (
                    contract["execution_plan_sha256"]
                    if formal
                    else value["runtime_sha256"]
                ),
                "split_sha256": (
                    contract["execution_split_sha256"]
                    if formal
                    else value["split_sha256"]
                ),
                "corpus_sha256": contract["corpus_sha256"],
                "arrival_trace_sha256": contract["arrival_trace_sha256"],
                "request_ids_sha256": contract["request_ids_sha256"],
                "sampling_profile_sha256": contract["sampling_profile_sha256"],
                "model_lock_sha256": contract["model_lock_sha256"],
                "patched_sglang_tree": PINNED_SGLANG_TREE,
                "topology_sha256": topology_sha256,
                "tensor_parallel_size": tensor_parallel_size,
                "data_parallel_size": data_parallel_size,
                "world_size": world_size,
                "rank": rank,
                "expected_round_rows": contract["expected_round_rows"],
                "expected_update_rows": contract["expected_update_rows"],
                "expected_performance_rows": contract["expected_performance_rows"],
                "workload_contract": contract["workload_contract"],
                "experiment_budget_sha256": (
                    contract["experiment_budget_sha256"] if formal else None
                ),
            }
            closed_loop = str(cell.identity.arrival).startswith("closed_loop")
            if not closed_loop:
                expected_run["expected_request_rows"] = contract[
                    "expected_request_rows"
                ]
            if any(
                run_row.get(name) != expected for name, expected in expected_run.items()
            ):
                raise ValueError(
                    "industrial run identity differs from its locked contract"
                )
            if not _is_lower_sha256(run_row.get("run_nonce_sha256")):
                raise ValueError("industrial run lacks a content-bound nonce")
            nonce = str(run_row["run_nonce_sha256"])
            owner = nonce_owner.setdefault(nonce, str(cell_id))
            if owner != cell_id:
                raise ValueError("industrial run nonce is reused across registry cells")
            if cell.identity.experiment == "preflight":
                attestation_path = row.get("preflight_attestation_path")
                if not isinstance(attestation_path, str) or not attestation_path:
                    raise ValueError("preflight completion lacks an attestation path")
                attestation_source = Path(attestation_path)
                attestation_sidecar = Path(f"{attestation_source}.sha256")
                if (
                    not attestation_source.is_file()
                    or attestation_source.is_symlink()
                    or not attestation_sidecar.is_file()
                    or attestation_sidecar.is_symlink()
                    or attestation_source.resolve().parent != root.resolve()
                ):
                    raise ValueError(
                        "preflight attestation must live in its evidence root"
                    )
                attestation = _load_bound_json(attestation_source)
                attestation_sha256 = _canonical_sha256(attestation)
                if (
                    not isinstance(attestation, dict)
                    or row.get("preflight_attestation_sha256") != attestation_sha256
                    or run_row.get("preflight_attestation_sha256") != attestation_sha256
                ):
                    raise ValueError("preflight attestation digest mismatch")
                required_checks = {
                    "environment_and_patch_preflight": {
                        "identity",
                        "environment",
                        "patch_apply",
                        "compile",
                        "patch_tests",
                        "compatibility",
                    },
                    "exactness_memory_telemetry_preflight": {
                        "exactness",
                        "memory",
                        "telemetry",
                        "target_only_allocation",
                        "static_allocation",
                    },
                    "simultaneous_single_gpu_interference": {
                        "isolated",
                        "simultaneous",
                        "hardware",
                        "paired_blocks",
                    },
                }[cell.identity.task]
                checks = attestation.get("checks")
                source_files = attestation.get("source_files")
                if (
                    not isinstance(attestation.get("schema_version"), int)
                    or isinstance(attestation.get("schema_version"), bool)
                    or attestation.get("schema_version") != 1
                    or attestation.get("kind") != "industrial_preflight_attestation"
                    or attestation.get("status") != "PASS"
                    or attestation.get("registry_sha256") != registry.sha256
                    or attestation.get("cell_id") != cell_id
                    or attestation.get("runtime_sha256") != value["runtime_sha256"]
                    or attestation.get("split_sha256") != value["split_sha256"]
                    or attestation.get("run_nonce_sha256")
                    != run_row["run_nonce_sha256"]
                    or attestation.get("topology_sha256") != topology_sha256
                    or not isinstance(attestation.get("rank"), int)
                    or isinstance(attestation.get("rank"), bool)
                    or attestation.get("rank") != rank
                    or attestation.get("gpu_uuid")
                    != (
                        assignment.gpu_uuids[rank]
                        if formal and assignment is not None
                        else cell.resources.gpu_uuids[rank]
                    )
                    or not isinstance(checks, dict)
                    or set(checks) != required_checks
                    or any(result != "PASS" for result in checks.values())
                    or not isinstance(source_files, list)
                    or not source_files
                    or not all(isinstance(item, str) and item for item in source_files)
                    or len(source_files) != len(set(source_files))
                    or not _is_lower_sha256(attestation.get("source_evidence_sha256"))
                    or evidence_files_sha256(source_files)
                    != attestation.get("source_evidence_sha256")
                ):
                    raise ValueError("preflight attestation contract is incomplete")
                resolved_root = root.resolve()
                source_paths = tuple(Path(item) for item in source_files)
                if any(
                    not source.is_file()
                    or source.is_symlink()
                    or (
                        source.resolve() != resolved_root
                        and resolved_root not in source.resolve().parents
                    )
                    for source in source_paths
                ):
                    raise ValueError(
                        "preflight source evidence must be regular files in its root"
                    )
            elif (
                run_row.get("preflight_attestation_sha256") is not None
                or row.get("preflight_attestation_path") is not None
                or row.get("preflight_attestation_sha256") is not None
            ):
                raise ValueError(
                    "non-preflight completion carries a preflight attestation"
                )

            if formal:
                if assignment is None:
                    raise RuntimeError("formal completion lost its physical assignment")
                non_serving = stage == "preflight" or (
                    cell.resources.workload_class.value in {"compile", "download"}
                )
                budget_status = row.get("budget_observation_status")
                budget_reason = row.get("budget_observation_reason_code")
                budget_path = row.get("budget_observation_path")
                budget_sha256 = row.get("budget_observation_sha256")
                if non_serving:
                    if (
                        budget_status != "NOT_APPLICABLE"
                        or budget_reason != "preflight_or_non_serving_execution"
                        or budget_path is not None
                        or budget_sha256 is not None
                    ):
                        raise ValueError(
                            "non-serving budget observation disposition is not exact"
                        )
                else:
                    expected_budget_status = (
                        "OBSERVED" if rank == 0 else "BOUND_TO_RANK0"
                    )
                    expected_budget_reason = (
                        None if rank == 0 else "gang_observation_published_by_rank0"
                    )
                    if (
                        budget_status != expected_budget_status
                        or budget_reason != expected_budget_reason
                        or not isinstance(budget_path, str)
                        or not _is_lower_sha256(budget_sha256)
                    ):
                        raise ValueError(
                            "serving completion lacks its exact budget observation"
                        )
                    budget_identity = (
                        budget_path,
                        budget_sha256,
                        row["experiment_budget_sha256"],
                    )
                    previous_budget = budget_consensus.setdefault(
                        str(cell_id), budget_identity
                    )
                    if previous_budget != budget_identity:
                        raise ValueError(
                            "cell ranks disagree on one budget observation"
                        )
                    if rank == 0:
                        _validate_budget_observation_receipt(
                            budget_path,
                            expected_sha256=budget_sha256,
                            experiment_budget_sha256=row["experiment_budget_sha256"],
                            prepared_receipt_sha256=terminal_receipt.get(
                                "prepared_receipt_sha256"
                            ),
                            cell=cell,
                            evidence_root=root,
                            fixed_instance_gpu_count=(len(inventory.devices)),
                        )

        performance = pq.read_table(
            evidence["performance"],
            columns=[
                "region",
                "method",
                "optimizer_bytes",
                "adaptation_memory_ledger",
                "trainable_parameters",
                "training_cuda_ms",
                "optimizer_cuda_ms",
                "merge_cuda_ms",
                "publish_cuda_ms",
                "barrier_cuda_ms",
                "exposed_update_ms",
                "main_side_overlap_ratio",
                "updates_launched",
                "updates_published",
                "exactness_violations",
                "version_mismatches",
                "fallbacks",
                "nonfinite_updates",
                "oom_events",
                "retractions",
                "communicator_failures",
                "admission_rejections",
                "timeouts",
                "cancellations",
                "offered_requests",
                "admitted_requests",
                "completed_requests",
                "unfinished_requests",
                "evidence_backpressure_events",
                "evidence_dropped_rows",
            ],
        ).to_pylist()
        if (
            not performance
            or (strict and len(performance) != contract["expected_performance_rows"])
            or any(
                row["method"] != cell.identity.method
                or row["evidence_dropped_rows"] != 0
                for row in performance
            )
        ):
            raise ValueError(
                "industrial completion requires nonempty, lossless performance evidence"
            )
        if cell.identity.task != "failure_injection" and any(
            any(
                row[field] != 0
                for field in (
                    "exactness_violations",
                    "version_mismatches",
                    "fallbacks",
                    "nonfinite_updates",
                    "oom_events",
                    "retractions",
                    "communicator_failures",
                )
            )
            for row in performance
        ):
            raise ValueError("industrial completion contains a safety violation")
        if strict and any(
            not isinstance(row["evidence_backpressure_events"], int)
            or isinstance(row["evidence_backpressure_events"], bool)
            or row["evidence_backpressure_events"] < 0
            for row in performance
        ):
            raise ValueError("industrial completion lacks backpressure accounting")
        if cell.identity.method in {"target_only", "static"}:
            for performance_row in performance:
                _validate_allocation_free_performance(
                    performance_row,
                    method=cell.identity.method,
                )
        elif cell.identity.task != "failure_injection" and any(
            not isinstance(row["updates_launched"], int)
            or row["updates_launched"] < 1
            or not isinstance(row["updates_published"], int)
            or row["updates_published"] < 1
            for row in performance
        ):
            raise ValueError("adapted completion has no launched/published update")
        requests = pq.read_table(
            evidence["request"],
            columns=[
                "request_id",
                "method",
                "repetition_block",
                "finished",
                "outcome_status",
                "output_hash_format",
                "output_tokens",
                "output_sha256",
                "output_token_ids",
                "output_token_ids_sha256",
                "arrival_ns",
                "admitted_ns",
                "completed_ns",
            ],
        ).to_pylist()
        allowed_outcomes = {
            "completed",
            "rejected",
            "timed_out",
            "cancelled",
            "unfinished",
        }
        if not requests or any(
            row["method"] != cell.identity.method
            or row["repetition_block"] != cell.identity.block
            or row["outcome_status"] not in allowed_outcomes
            or row["finished"] is not (row["outcome_status"] == "completed")
            or (row["completed_ns"] is None and row["outcome_status"] != "unfinished")
            or (
                row["completed_ns"] is not None
                and row["outcome_status"] == "unfinished"
            )
            for row in requests
        ):
            raise ValueError(
                "industrial completion has incomplete terminal-outcome evidence"
            )
        for request_row in requests:
            _validate_request_output_identity(request_row)
        request_outputs = {
            request["request_id"]: request["output_sha256"] for request in requests
        }
        if len(request_outputs) != len(requests):
            raise ValueError(
                "industrial request evidence contains duplicate identities"
            )
        if strict:
            if contract is None:
                raise RuntimeError("strict completion lost its split contract")
            closed_loop = str(cell.identity.arrival).startswith("closed_loop")
            if closed_loop:
                concurrency = cell.identity.concurrency
                if concurrency is None and cell.identity.arrival == "closed_loop_c1":
                    concurrency = 1
                if (
                    not isinstance(concurrency, int)
                    or isinstance(concurrency, bool)
                    or concurrency < 1
                    or not requests
                    or run_row["expected_request_rows"] != len(requests)
                ):
                    raise ValueError(
                        "closed-loop request coverage lacks a realized population"
                    )
                actual_ids = set(request_outputs)
                if actual_ids - set(contract["request_ids"]):
                    raise ValueError("closed-loop evidence crosses its request pool")
                request_by_id = {str(row["request_id"]): row for row in requests}
                for lane in range(concurrency):
                    seen_gap = False
                    previous_completed_ns: int | None = None
                    lane_request_ids = contract["request_ids"][lane::concurrency]
                    if not lane_request_ids or lane_request_ids[0] not in actual_ids:
                        raise ValueError(
                            "closed-loop evidence omits a registered client lane"
                        )
                    for request_id in lane_request_ids:
                        present = request_id in actual_ids
                        if seen_gap and present:
                            raise ValueError(
                                "closed-loop evidence is not a per-client prefix"
                            )
                        if present and stage != "preflight":
                            request = request_by_id[request_id]
                            arrival_ns = request.get("arrival_ns")
                            completed_ns = request.get("completed_ns")
                            if (
                                not isinstance(arrival_ns, int)
                                or isinstance(arrival_ns, bool)
                                or not isinstance(completed_ns, int)
                                or isinstance(completed_ns, bool)
                                or (
                                    previous_completed_ns is None
                                    and arrival_ns != run_row["started_ns"]
                                )
                                or (
                                    previous_completed_ns is not None
                                    and arrival_ns != previous_completed_ns
                                )
                            ):
                                raise ValueError(
                                    "closed-loop evidence is not zero-think"
                                )
                            previous_completed_ns = completed_ns
                        seen_gap = seen_gap or not present
            elif len(requests) != contract["expected_request_rows"] or set(
                request_outputs
            ) != set(contract["request_ids"]):
                raise ValueError("request evidence does not match the locked split")
            if contract["expected_round_rows"]:
                rounds = pq.read_table(
                    evidence["round"], columns=["request_id"]
                ).to_pylist()
                if len(rounds) != contract["expected_round_rows"] or {
                    round_row["request_id"] for round_row in rounds
                } != set(contract["request_ids"]):
                    raise ValueError("round evidence does not match the locked split")
            if contract["expected_update_rows"]:
                updates = pq.read_table(
                    evidence["update"], columns=["request_ids"]
                ).to_pylist()
                if len(updates) != contract["expected_update_rows"]:
                    raise ValueError(
                        "update evidence count differs from the locked split"
                    )
                for update in updates:
                    try:
                        update_request_ids = json.loads(update["request_ids"])
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            "update evidence has invalid request identities"
                        ) from exc
                    if (
                        not isinstance(update_request_ids, list)
                        or not update_request_ids
                        or not set(update_request_ids) <= set(contract["request_ids"])
                    ):
                        raise ValueError("update evidence crosses the locked split")
            outcome_counts = {
                outcome: sum(
                    request["outcome_status"] == outcome for request in requests
                )
                for outcome in allowed_outcomes
            }
            if outcome_counts["unfinished"]:
                raise ValueError(
                    "unfinished requests cannot enter a completed claim artifact"
                )
            aggregate_rows = [
                performance_row
                for performance_row in performance
                if performance_row["offered_requests"] is not None
            ]
            if len(aggregate_rows) != 1:
                raise ValueError(
                    "industrial completion requires one exact load-accounting row"
                )
            aggregate = aggregate_rows[0]
            expected_accounting = {
                "offered_requests": len(requests),
                "admitted_requests": sum(
                    request["admitted_ns"] is not None for request in requests
                ),
                "completed_requests": outcome_counts["completed"],
                "admission_rejections": outcome_counts["rejected"],
                "timeouts": outcome_counts["timed_out"],
                "cancellations": outcome_counts["cancelled"],
                "unfinished_requests": outcome_counts["unfinished"],
            }
            if any(
                not isinstance(aggregate.get(name), int)
                or isinstance(aggregate.get(name), bool)
                or aggregate.get(name) != expected
                for name, expected in expected_accounting.items()
            ):
                raise ValueError(
                    "performance load accounting differs from request outcomes"
                )
        covered = rank_coverage.setdefault(str(cell_id), set())
        if rank in covered:
            raise ValueError("completed-cell artifact duplicates a rank receipt")
        covered.add(rank)
        if strict:
            current_consensus = {
                "run_id": run_id,
                "run_nonce_sha256": run_row["run_nonce_sha256"],
                "topology_sha256": topology_sha256,
                "physical_binding_sha256": (
                    None if not formal else row["physical_binding_sha256"]
                ),
                "experiment_budget_sha256": (
                    None if not formal else row["experiment_budget_sha256"]
                ),
                "runtime_sha256": run_row["runtime_sha256"],
                "split_sha256": run_row["split_sha256"],
                "request_ids_sha256": run_row["request_ids_sha256"],
                "request_outputs": request_outputs,
            }
            previous_consensus = consensus.setdefault(str(cell_id), current_consensus)
            if previous_consensus != current_consensus:
                raise ValueError("cell ranks do not agree on one run/output identity")
        completed.append(str(cell_id))
    for cell_id, ranks in rank_coverage.items():
        expected_ranks = set(
            range(
                len(
                    _industrial_physical_assignment_from_dict(
                        contract_by_cell[cell_id]["physical_assignment"]
                    ).gpu_uuids
                )
                if formal
                else len(known[cell_id].resources.gpu_uuids)
            )
        )
        if ranks != expected_ranks:
            raise ValueError("completed cell lacks exact per-rank evidence coverage")
    unique = tuple(dict.fromkeys(completed))
    return unique, _canonical_sha256(value)


def _single_industrial_receipt(
    path: str | Path, *, experiment: str
) -> ExperimentReceipt:
    (receipt,) = _load_industrial_receipts([str(path)])
    if receipt.experiment != experiment:
        raise ValueError(f"expected a sealed {experiment} receipt")
    return receipt


def _write_planning_artifact(
    output: str | Path,
    *,
    artifact,
    encode,
) -> int:
    value = encode(artifact)
    _write_json(output, value)
    print(artifact.sha256)
    return 0


def _reduce_e1_activation(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    receipt = _single_industrial_receipt(args.e3a_receipt, experiment="E3a")
    selection = sealed_e3a_selection_from_dict(_load_bound_json(args.selection))
    artifact = reduce_e1_activation(
        registry,
        e3a_receipt=receipt,
        selection=selection,
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=reducer_activation_artifact_to_dict,
    )


def _reduce_e2_activation(args: argparse.Namespace) -> int:
    registry, receipt, pareto, stage_index, prior = _load_e2_activation_manifest(
        args.manifest
    )
    artifact = reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=stage_index,
        prior_reduction=prior,
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=reducer_activation_artifact_to_dict,
    )


def _reduce_e2_halving(args: argparse.Namespace) -> int:
    artifact = _load_e2_stage_manifest(args.manifest)
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=e2_stage_reduction_artifact_to_dict,
    )


def _materialize_confirmation_pilots(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    family = confirmation_family_identity_from_dict(_load_bound_json(args.family))
    artifact = materialize_confirmation_pilots(registry, family)
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=family_activation_artifact_to_dict,
    )


def _reduce_confirmation_family_power(args: argparse.Namespace) -> int:
    registry, pilot, inventory, envelope, blocks = _load_family_power_manifest(
        args.manifest
    )
    plan = reduce_confirmation_family_power(
        registry=registry,
        pilot_activation=pilot,
        blocks=blocks,
        hardware_envelope=envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    return _write_planning_artifact(
        args.output,
        artifact=plan,
        encode=confirmation_family_power_reduction_artifact_to_dict,
    )


def _materialize_confirmation_final_prefix(args: argparse.Namespace) -> int:
    registry, pilot, inventory, envelope, blocks = _load_family_power_manifest(
        args.power_manifest
    )
    reduction = reduce_confirmation_family_power(
        registry=registry,
        pilot_activation=pilot,
        blocks=blocks,
        hardware_envelope=envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    artifact = materialize_confirmation_prefix(
        registry,
        family=pilot.family,
        reduction=reduction,
        pilot_activation=pilot,
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=family_activation_artifact_to_dict,
    )


def _validate_evidence_alias(args: argparse.Namespace) -> int:
    manifest = raw_evidence_alias_manifest_from_dict(_load_bound_json(args.manifest))
    artifact = reduce_evidence_alias(
        registry=_load_industrial_registry(args.registry),
        manifest=manifest,
        hardware_envelope=_analysis_hardware_envelope(
            _load_bound_json(args.hardware_envelope)
        ),
        inventory=_load_gpu_inventory(args.inventory),
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=evidence_alias_reduction_artifact_to_dict,
    )


def _build_evidence_dependence_map(args: argparse.Namespace) -> int:
    direct_map = evidence_dependence_map_from_dict(_load_bound_json(args.direct_map))
    direct_cell_ids: list[str] = []
    for unit in direct_map.units:
        if unit.member_cell_ids != (unit.source_cell_id,) or unit.unit_sha256 != (
            _canonical_sha256({"direct_observation_cell_id": unit.source_cell_id})
        ):
            raise ValueError(
                "direct dependence-map input must contain singleton observations"
            )
        direct_cell_ids.append(unit.source_cell_id)
    aliases = tuple(
        evidence_alias_reduction_artifact_from_dict(_load_bound_json(path))
        for path in args.alias_reduction
    )
    if len({alias.sha256 for alias in aliases}) != len(aliases):
        raise ValueError("duplicate evidence alias artifact")
    artifact = build_evidence_dependence_map(
        direct_observation_cell_ids=tuple(direct_cell_ids),
        aliases=aliases,
    )
    return _write_planning_artifact(
        args.output,
        artifact=artifact,
        encode=evidence_dependence_map_to_dict,
    )


def _collect_gpu_inventory(args: argparse.Namespace) -> int:
    inventory, receipt = collect_gpu_inventory(
        challenge_nonce_sha256=args.challenge_nonce_sha256
    )
    if inventory.source_receipt_sha256 != receipt.get("receipt_sha256"):
        raise RuntimeError("GPU inventory source receipt binding changed")
    _write_json(args.receipt_output, receipt)
    _write_json(args.output, inventory.to_dict())
    reloaded = _load_gpu_inventory(args.output)
    if reloaded != inventory:
        raise RuntimeError("written GPU inventory changed identity")
    print(inventory.sha256)
    return 0


def _build_interference_envelope(args: argparse.Namespace) -> int:
    inventory = _load_gpu_inventory(args.inventory)
    envelope, receipt = build_serial_interference_envelope(inventory)
    if envelope.source_receipt_sha256 != receipt.get("receipt_sha256"):
        raise RuntimeError("interference-envelope source receipt binding changed")
    _write_json(args.receipt_output, receipt)
    _write_json(args.output, envelope.to_dict())
    reloaded = _load_interference_envelope(args.output)
    if reloaded != envelope:
        raise RuntimeError("written interference envelope changed identity")
    print(envelope.sha256)
    return 0


def _materialize_industrial_budget_plan(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    gpu_inventory = _load_gpu_inventory(args.inventory)
    if not args.activation_plan and not args.family_activation:
        decision = {
            "schema_version": 1,
            "kind": "industrial_budget_materialization_decision",
            "status": "BLOCKED",
            "reason_code": "reducer_owned_activation_manifest_missing",
            "registry_sha256": registry.sha256,
            "gpu_inventory_sha256": gpu_inventory.sha256,
            "detail": ("no bound reducer-owned activation manifest was supplied"),
        }
        _write_json(args.output, decision)
        print(_canonical_sha256(decision))
        return 42
    activations, family_activations, family_power_reductions = (
        _load_budget_activation_bundle(
            activation_paths=args.activation_plan,
            family_activation_paths=args.family_activation,
            family_power_plan_paths=args.family_power_plan,
        )
    )
    (
        policy,
        load_bindings,
        capacity_envelope,
        capacity_authority,
    ) = _load_budget_materialization_inputs(
        policy_path=args.budget_policy,
        load_binding_paths=args.budget_load_binding,
        capacity_envelope_path=args.capacity_envelope,
        capacity_manifest_path=args.capacity_manifest,
        capacity_verification_receipt_path=args.capacity_verification_receipt,
    )
    plan = materialize_industrial_budgets(
        registry,
        activations=activations,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
        load_bindings=load_bindings,
        policy=policy,
        inventory=budget_inventory_identity_from_gpu_inventory(gpu_inventory),
        capacity_envelope=capacity_envelope,
        capacity_authority=capacity_authority,
    )
    _write_json(args.output, budget_plan_to_dict(plan))
    if _load_budget_plan(args.output) != plan:
        raise RuntimeError("written BudgetPlan changed identity")
    print(plan.sha256)
    return 0 if plan.status == "READY" else 42


def _materialize_stage_activation(args: argparse.Namespace) -> int:
    artifact = _load_registry_stage_activation_manifest(args.manifest)
    _write_json(args.output, registry_stage_activation_to_dict(artifact))
    reloaded = registry_stage_activation_from_dict(_load_bound_json(args.output))
    if reloaded != artifact:
        raise RuntimeError("written registry-stage activation changed identity")
    print(artifact.sha256)
    return 0 if artifact.status == "AVAILABLE" else 42


def _estimate_industrial_budget(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    gpu_inventory = _load_gpu_inventory(args.inventory)
    interference_envelope = _load_interference_envelope(args.interference_envelope)
    plan, _, _, _ = _rematerialize_budget_plan(
        registry=registry,
        gpu_inventory=gpu_inventory,
        declared_plan_path=args.budget_plan,
        activation_paths=args.activation_plan,
        family_activation_paths=args.family_activation,
        family_power_plan_paths=args.family_power_plan,
        policy_path=args.budget_policy,
        load_binding_paths=args.budget_load_binding,
        capacity_envelope_path=args.capacity_envelope,
        capacity_manifest_path=args.capacity_manifest,
        capacity_verification_receipt_path=args.capacity_verification_receipt,
        require_ready=False,
    )
    plan_assumptions = tuple(
        sorted(
            {
                row.reason_code
                for row in plan.dispositions
                if row.status is BudgetDispositionStatus.UNRESOLVED
            }
        )
    )
    report = estimate_industrial_budget(
        registry,
        activated_cell_ids=plan.activated_cell_ids,
        activation_sha256=plan.activation_sha256,
        budgets=plan.diagnostic_budgets,
        inventory=plan.inventory,
        gpu_inventory=gpu_inventory,
        interference_envelope=interference_envelope,
        unresolved_assumptions=plan_assumptions,
    )
    artifact = industrial_budget_report_to_dict(report)
    _write_json(args.output, artifact)
    print(report.sha256)
    return 0 if not report.unresolved_assumptions else 42


def _plan_industrial_dispatch(args: argparse.Namespace) -> int:
    registry = _load_industrial_registry(args.registry)
    inventory = _load_gpu_inventory(args.inventory)
    envelope = _load_interference_envelope(args.interference_envelope)
    budget_plan, activations, family_activations, family_power_reductions = (
        _rematerialize_budget_plan(
            registry=registry,
            gpu_inventory=inventory,
            declared_plan_path=args.budget_plan,
            activation_paths=args.activation_plan,
            family_activation_paths=args.family_activation,
            family_power_plan_paths=args.family_power_plan,
            policy_path=args.budget_policy,
            load_binding_paths=args.budget_load_binding,
            capacity_envelope_path=args.capacity_envelope,
            capacity_manifest_path=args.capacity_manifest,
            capacity_verification_receipt_path=(args.capacity_verification_receipt),
        )
    )
    if len(activations) > 1:
        raise ValueError("dispatch accepts at most one reducer activation artifact")
    receipts = _load_industrial_receipts(args.receipt)
    activation_artifact = None if not activations else activations[0]
    completed_activation_artifact = activation_artifact
    completed_family_activations = tuple(
        artifact
        for artifact in family_activations
        if artifact.activation_round == "excluded_pilots"
    )
    ready_experiment = registry.ready_experiment(receipts)
    completed_e2_requested = args.completed_e2_stage_manifest is not None
    if ready_experiment == "E2" and args.completed_cells is not None:
        if not completed_e2_requested:
            raise ValueError(
                "E2 completed-cell evidence requires its raw completed-stage manifest"
            )
        completed_reduction = _load_e2_stage_manifest(args.completed_e2_stage_manifest)
        if (
            completed_reduction.registry_sha256 != registry.sha256
            or activation_artifact is None
            or activation_artifact.plan.activation_round
            != f"halving_{completed_reduction.stage_index + 1}"
            or activation_artifact.plan.source_selection_sha256
            != completed_reduction.sha256
        ):
            raise ValueError(
                "E2 completed-stage authority is not the immediate predecessor of "
                "the next activation"
            )
        completed_activation_artifact = completed_reduction.activation
    elif completed_e2_requested:
        raise ValueError(
            "completed E2 stage authority requires E2 completed-cell evidence"
        )
    direct_dependency_sha256 = None
    if ready_experiment is not None:
        definition = registry.definition(ready_experiment)
        if definition.dependencies:
            dependency_name = definition.dependencies[-1]
            matching = tuple(
                receipt for receipt in receipts if receipt.experiment == dependency_name
            )
            if len(matching) != 1:
                raise ValueError(
                    "dispatch requires one exact direct dependency receipt"
                )
            direct_dependency_sha256 = matching[0].sha256
    completed, _ = _completed_industrial_cells(
        args.completed_cells,
        registry,
        experiment=ready_experiment,
        require_industrial_contract=args.completed_cells is not None,
        direct_dependency_receipt_sha256=direct_dependency_sha256,
        activation_artifact=completed_activation_artifact,
        family_activations=completed_family_activations,
        family_power_reductions=(),
        inventory=inventory,
    )
    planning_context = GpuDispatchPlanningContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        budgets=tuple(
            budget
            for budget in budget_plan.require_ready()
            if budget.cell_id not in completed
        ),
        receipts=receipts,
        completed_cell_ids=tuple(sorted(completed)),
        activation_artifact=activation_artifact,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
    )
    plan = planning_context.issue_plan()
    artifact = plan.to_dict()
    _write_json(args.output, artifact)
    if (
        GpuDispatchPlan.from_dict(
            artifact,
            planning_context=planning_context,
        )
        != plan
    ):
        raise RuntimeError("written GPU dispatch plan changed identity")
    print(plan.sha256)
    return 0


def _execute_dispatch_wave(args: argparse.Namespace) -> int:
    """Run one receipt-bounded wave, or report an immutable prelaunch block."""

    try:
        receipt = asyncio.run(
            execute_dispatch_wave_bundles(
                tuple(args.bundle),
                wave_index=args.wave_index,
                receipt_output=args.receipt_output,
                resume_receipt_path=args.resume_receipt,
            )
        )
    except ExecutionBundleBlockedError as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "industrial_dispatch_wave_execution_decision",
                    "status": "BLOCKED",
                    "reason_code": error.reason_code,
                    "wave_index": args.wave_index,
                    "bundle_count": len(args.bundle),
                },
                sort_keys=True,
            )
        )
        return 42
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "industrial_dispatch_wave_execution_decision",
                    "status": "BLOCKED",
                    "reason_code": "industrial_execution_bundle_or_resume_invalid",
                    "wave_index": args.wave_index,
                    "bundle_count": len(args.bundle),
                    "failure_sha256": _canonical_sha256(
                        {
                            "exception_type": type(error).__qualname__,
                            "message": str(error),
                        }
                    ),
                },
                sort_keys=True,
            )
        )
        return 42
    print(receipt.sha256)
    return 42 if receipt.phase.value == "FAILED" else 0


def _analyze_industrial(args: argparse.Namespace) -> int:
    (
        registry,
        pilot,
        final,
        plan,
        inventory,
        envelope,
        blocks,
        alias_manifests,
        dependence,
        gpu_attestation,
        doctor_report,
        repetitions,
        seed,
    ) = _load_industrial_analysis_manifest(args.manifest)
    reduction = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilot,
        final_activation=final,
        confirmation_reduction=plan,
        blocks=blocks,
        hardware_envelope=envelope,
        inventory=inventory,
        evidence_dependence_map=dependence,
        evidence_alias_manifests=alias_manifests,
        gpu_attestation=gpu_attestation,
        doctor_report=doctor_report,
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed,
    )
    artifact = reduction.artifact.to_dict()
    if _canonical_sha256(artifact) != reduction.artifact.sha256:
        raise RuntimeError("industrial reducer artifact digest is not canonical")
    _write_json(args.output, artifact)
    if _artifact_sha256(args.output) != reduction.artifact.sha256:
        raise RuntimeError("written industrial reducer artifact identity changed")
    print(reduction.artifact.sha256)
    # The current release has no hardware-rooted attester. Content-bound caller
    # artifacts remain diagnostic and can never produce a positive exit claim.
    return 42


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        project_root = args.project_root or args.path or "."
        sglang_root = args.sglang_root or args.path
        formatted = format_doctor(project_root, sglang_root)
        print(formatted)
        report = json.loads(formatted)
        return 0 if report.get("status") == "PASS" else 42
    if args.command == "validate-config":
        config = load_run_config(args.config)
        print(config.model_dump_json(indent=2))
        return 0
    if args.command == "build-speed-study":
        manifest = SpeedStudyManifest.default()
        manifest.write(args.output)
        print(manifest.sha256)
        return 0
    if args.command == "build-industrial-registry":
        return _build_industrial_registry(args)
    if args.command == "collect-gpu-inventory":
        return _collect_gpu_inventory(args)
    if args.command == "build-interference-envelope":
        return _build_interference_envelope(args)
    if args.command == "seal-industrial-stage":
        return _seal_industrial_stage(args)
    if args.command == "plan-industrial-dispatch":
        return _plan_industrial_dispatch(args)
    if args.command == "execute-dispatch-wave":
        return _execute_dispatch_wave(args)
    if args.command == "materialize-industrial-budgets":
        return _materialize_industrial_budget_plan(args)
    if args.command == "materialize-stage-activation":
        return _materialize_stage_activation(args)
    if args.command == "estimate-industrial-budget":
        return _estimate_industrial_budget(args)
    if args.command == "reduce-e1-activation":
        return _reduce_e1_activation(args)
    if args.command == "reduce-e2-activation":
        return _reduce_e2_activation(args)
    if args.command == "reduce-e2-successive-halving":
        return _reduce_e2_halving(args)
    if args.command == "materialize-confirmation-pilots":
        return _materialize_confirmation_pilots(args)
    if args.command == "reduce-confirmation-family-power":
        return _reduce_confirmation_family_power(args)
    if args.command == "materialize-confirmation-prefix":
        return _materialize_confirmation_final_prefix(args)
    if args.command == "validate-evidence-alias":
        return _validate_evidence_alias(args)
    if args.command == "build-evidence-dependence-map":
        return _build_evidence_dependence_map(args)
    if args.command == "analyze-industrial":
        return _analyze_industrial(args)
    if args.command == "build-onlinespec-study":
        manifest = OnlineSpecManifest.default()
        manifest.write(args.output)
        print(manifest.sha256)
        return 0
    if args.command == "verify-onlinespec-source":
        receipt = verify_onlinespec_source_checkout(args.checkout, args.audit)
        _write_json(args.output, receipt)
        print(_canonical_sha256(receipt))
        return 0
    if args.command == "list-onlinespec-candidates":
        return _list_onlinespec_candidates(args)
    if args.command == "lock-models":
        lock = resolve_model_lock(tuple(args.models), token=os.environ.get("HF_TOKEN"))
        lock.write(args.output)
        print(lock.sha256)
        return 0
    if args.command == "prepare-models":
        lock = ModelLock.load(args.lockfile)
        roots = prepare_models(
            lock,
            args.model_cache,
            token=os.environ.get("HF_TOKEN"),
            local_files_only=args.offline,
        )
        _write_json(
            args.output,
            {
                "schema_version": 2,
                "lock_sha256": lock.sha256,
                "roots": roots,
            },
        )
        return 0
    if args.command == "select-speed-config":
        return _select(args)
    if args.command == "select-anchor-config":
        return _select_anchor(args)
    if args.command == "select-onlinespec-config":
        return _select_onlinespec(args)
    if args.command == "select-onlinespec-anchor-config":
        return _select_onlinespec_anchor(args)
    if args.command == "render-runtime":
        return _render_runtime(args)
    if args.command == "render-onlinespec-runtime":
        return _render_onlinespec_runtime(args)
    if args.command == "render-onlinespec-tuning-runtime":
        return _render_onlinespec_tuning_runtime(args)
    if args.command == "render-static-load-runtime":
        return _render_static_load_runtime(args)
    if args.command == "render-target-only-runtime":
        return _render_target_only_runtime(args)
    if args.command == "render-tuning-runtime":
        return _render_tuning_runtime(args)
    if args.command == "render-replication-runtime":
        return _render_replication_runtime(args)
    if args.command == "list-tuning-candidates":
        return _list_tuning_candidates(args)
    if args.command == "run-controlled-slice":
        return _run_controlled_slice(args)
    if args.command == "run-onlinespec-tuning-slice":
        return _run_onlinespec_tuning_slice(args)
    if args.command == "run-natural-slice":
        return _run_natural_slice(args)
    if args.command == "build-profiler-plan":
        return _build_profiler_plan(args)
    if args.command == "collect-static-load-screen":
        return _collect_static_load(args)
    if args.command == "advance-tuning-stage":
        return _advance_tuning(args)
    if args.command == "advance-onlinespec-tuning-stage":
        return _advance_onlinespec_tuning(args)
    if args.command == "run-confirmation":
        return _run_confirmation(args)
    if args.command == "run-onlinespec-confirmation":
        return _run_onlinespec_confirmation(args)
    if args.command == "run-target-reference":
        return _run_target_reference(args)
    if args.command == "collect-speed-study":
        return _collect_speed_study(args)
    if args.command == "collect-onlinespec-study":
        return _collect_onlinespec_study(args)
    if args.command == "build-confirmation-queue":
        return _build_confirmation_queue(args)
    if args.command == "build-onlinespec-queue":
        return _build_onlinespec_queue(args)
    if args.command == "attest-speed-study":
        return _attest(args)
    if args.command == "attest-onlinespec-study":
        return _attest_onlinespec(args)
    if args.command == "analyze-speed-study":
        return _analyze(args)
    if args.command == "analyze-onlinespec-study":
        return _analyze_onlinespec(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
