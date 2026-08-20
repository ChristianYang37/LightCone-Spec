"""Path-only trainable-plan producers for trusted formal v03 cold start.

The caller supplies one runtime-BOUND trusted content bundle and one output
path.  Model identities, revisions, prepared roots, structural slots, recipes,
and every supporting raw-artifact path are selected and rebuilt by code.  The
two public entry points expose no digest, mode, scope, optimizer, cell, winner,
or selection input.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from lightcone_spec.adaptation.parameters import DFlashParameterPlan
from lightcone_spec.adaptation.plan_authority import (
    E1RecipeAnchorTrainablePlanSemantics,
    PreparedDrafterParameterInventory,
    PreparedParameterMetadata,
    TrainablePlanAuthorityBinding,
    TtsCalibrationTrainablePlanSemantics,
    bind_trainable_plan_authority,
    build_e1_recipe_anchor_trainable_plan_semantics,
    build_tts_calibration_trainable_plan_semantics,
    materialize_trainable_plan_authority_manifest,
    trainable_plan_authority_binding_from_dict,
    trainable_plan_authority_binding_to_dict,
    trainable_plan_cell_source_to_dict,
)
from lightcone_spec.config import run_config_sha256
from lightcone_spec.config.schema import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedModelSnapshotMember,
    TrustedSingleOperatorContentBundleBinding,
)
from lightcone_spec.experiments.formal_single_operator_model_registry import (
    require_formal_v03_bound_content_bundle,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.locking.prepared_models import (
    PreparedModelSnapshotContent,
    bind_prepared_model_content_authority,
    bind_prepared_models,
    materialize_prepared_model_content_manifest,
    revalidate_prepared_model_content_authority,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_TARGET_MODEL_ID = "Qwen/Qwen3-8B"
_DRAFTER_MODEL_ID = "z-lab/Qwen3-8B-DFlash-b16"


@dataclass(frozen=True)
class _MetadataTensorView:
    shape: tuple[int, ...]
    dtype: str

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_floating_point(self) -> bool:
        return self.dtype in {
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
            "torch.float64",
        }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_exclusive(path: Path, body: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bound_json(directory: Path, name: str, value: object) -> Path:
    path = directory / name
    body = _canonical_bytes(value)
    digest = hashlib.sha256(body).hexdigest()
    _write_exclusive(path, body)
    _write_exclusive(Path(f"{path}.sha256"), f"{digest}\n".encode("ascii"))
    return path


def _normalized_output(value: str | Path) -> Path:
    output = Path(value)
    if (
        not output.is_absolute()
        or Path(os.path.abspath(output)) != output
        or output.resolve(strict=False) != output
        or not output.name
    ):
        raise ValueError("trainable-plan output must be absolute and normalized")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("trainable-plan output parent must be a regular directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError("trainable-plan output already exists")
    return output


def _member(
    members: tuple[TrustedModelSnapshotMember, ...],
    *,
    model_id: str,
    role: str,
) -> TrustedModelSnapshotMember:
    matches = tuple(
        row for row in members if row.model_id == model_id and row.role == role
    )
    if len(matches) != 1:
        raise ValueError("trusted content lacks one canonical trainable-plan model")
    return matches[0]


def _supporting_directory(output: Path) -> Path:
    directory = output.with_name(f"{output.name}.sources")
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError as error:
        raise FileExistsError(
            "trainable-plan supporting artifact directory already exists"
        ) from error
    return directory


def _prevalidate_inventory(
    *,
    prepared_content: dict[str, object],
    model_lock: ModelLock,
    drafter: TrustedModelSnapshotMember,
    semantics: (
        TtsCalibrationTrainablePlanSemantics | E1RecipeAnchorTrainablePlanSemantics
    ),
) -> PreparedDrafterParameterInventory:
    raw_snapshots = prepared_content.get("snapshots")
    if type(raw_snapshots) is not list:
        raise TypeError("prepared model content snapshots are not an array")
    snapshots = tuple(
        PreparedModelSnapshotContent.from_dict(value) for value in raw_snapshots
    )
    matches = tuple(row for row in snapshots if row.model_id == drafter.model_id)
    if len(matches) != 1:
        raise ValueError("prepared model content lacks the canonical drafter")
    snapshot = matches[0]
    if snapshot.revision != drafter.revision or snapshot.root != (
        drafter.local_snapshot_path
    ):
        raise ValueError("prepared drafter differs from trusted content")
    inventory = PreparedDrafterParameterInventory(
        schema_version=1,
        kind="prepared_drafter_parameter_inventory",
        model_lock_sha256=model_lock.sha256,
        drafter_model_id=drafter.model_id,
        prepared_drafter_revision=drafter.revision,
        dspark_native_heads=None,
        parameters=tuple(
            PreparedParameterMetadata(
                name=tensor.name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                ownership="sharded",
            )
            for tensor in snapshot.tensors
        ),
    )
    plan = DFlashParameterPlan.build(
        (
            (
                row.name,
                _MetadataTensorView(shape=row.shape, dtype=row.dtype),
            )
            for row in inventory.parameters
        ),
        mode=semantics.expected_weight_update_mode,
        scope=semantics.expected_parameter_scope,
        rank=None,
    )
    plan.predict_memory(semantics.expected_optimizer)
    return inventory


def _publish_trusted_structural_trainable_plan(
    *,
    trusted_content_bundle_path: str | Path,
    output_path: str | Path,
    semantics: (
        TtsCalibrationTrainablePlanSemantics | E1RecipeAnchorTrainablePlanSemantics
    ),
) -> TrainablePlanAuthorityBinding:
    output = _normalized_output(output_path)
    content = TrustedSingleOperatorContentBundleBinding.bind(
        trusted_content_bundle_path
    )
    if content.runtime_binding_status != "BOUND":
        raise ValueError("trainable-plan producer requires runtime-BOUND content")
    bundle = content.reopen()
    require_formal_v03_bound_content_bundle(bundle)
    target = _member(bundle.model_members, model_id=_TARGET_MODEL_ID, role="target")
    drafter = _member(
        bundle.model_members,
        model_id=_DRAFTER_MODEL_ID,
        role="drafter",
    )
    if (
        semantics.expected_target_model_id != target.model_id
        or semantics.expected_drafter_model_id != drafter.model_id
    ):
        raise ValueError("structural plan models differ from trusted content")

    model_lock = ModelLock(
        schema_version=2,
        models=tuple(
            LockedModel(model_id=row.model_id, revision=row.revision)
            for row in sorted((target, drafter), key=lambda row: row.model_id)
        ),
    )
    model_lock.validate()
    prepared_models = bind_prepared_models(
        model_lock,
        {
            target.model_id: target.local_snapshot_path,
            drafter.model_id: drafter.local_snapshot_path,
        },
    )
    prepared_content = materialize_prepared_model_content_manifest(
        model_lock,
        prepared_models,
    )
    inventory = _prevalidate_inventory(
        prepared_content=prepared_content,
        model_lock=model_lock,
        drafter=drafter,
        semantics=semantics,
    )

    config = RunConfig(
        method=semantics.expected_method,
        model=ModelPair(
            target=target.model_id,
            drafter=drafter.model_id,
            target_revision=target.revision,
            drafter_revision=drafter.revision,
            algorithm=semantics.expected_backend,
            max_context_length=semantics.expected_model_max_context_length,
            draft_depth=semantics.expected_draft_depth,
        ),
        runtime=RuntimeConfig(
            context_length=semantics.expected_runtime_context_length,
            random_seed=semantics.expected_runtime_random_seed,
            sampling_profile_sha256=semantics.expected_sampling_profile_sha256,
            speculation_enabled=True,
            speculative_num_draft_tokens=semantics.expected_draft_width,
            max_running_requests=semantics.source_plan_concurrency,
        ),
        adaptation=semantics.adaptation_config,
    )
    semantics.validate_run_config(config)
    cell = semantics.cell_declaration
    cell_value = trainable_plan_cell_source_to_dict(cell)
    split = (
        {
            "schema_version": 1,
            "kind": "tts_calibration_trainable_plan_split",
            "cell_id": cell.cell_id,
            "run_config_sha256": run_config_sha256(config),
            "purpose": "parameter_inventory_only_not_calibration_selection",
        }
        if type(semantics) is TtsCalibrationTrainablePlanSemantics
        else {
            "schema_version": 1,
            "kind": "e1_recipe_anchor_trainable_plan_split",
            "cell_id": cell.cell_id,
            "run_config_sha256": run_config_sha256(config),
            "purpose": (
                "parameter_inventory_only_not_e1_activation_selection_or_winner"
            ),
        }
    )

    # Complete bundle/revision validation, model snapshot scanning, RunConfig
    # construction, cell serialization, and split derivation before reserving
    # the deterministic supporting directory.  A foreign or incomplete bundle
    # therefore fails without leaving a path that would force manual cleanup.
    directory = _supporting_directory(output)
    model_lock_path = _publish_bound_json(
        directory,
        "model-lock.json",
        model_lock.to_dict(),
    )
    prepared_content_path = _publish_bound_json(
        directory,
        "prepared-model-content.json",
        prepared_content,
    )
    prepared_content_authority = bind_prepared_model_content_authority(
        model_lock,
        prepared_models,
        prepared_content_path,
        expected_release_manifest_sha256=hashlib.sha256(
            _canonical_bytes(prepared_content)
        ).hexdigest(),
    )
    run_config_path = _publish_bound_json(
        directory,
        "run-config.json",
        config.model_dump(mode="json"),
    )
    cell_path = _publish_bound_json(directory, "cell.json", cell_value)
    split_path = _publish_bound_json(directory, "split.json", split)

    content_result = revalidate_prepared_model_content_authority(
        model_lock,
        prepared_content_authority,
        expected_release_manifest_sha256=(
            prepared_content_authority.release_manifest_sha256
        ),
    )
    snapshot = content_result.snapshot(drafter.model_id)
    rebound_inventory = PreparedDrafterParameterInventory(
        schema_version=1,
        kind="prepared_drafter_parameter_inventory",
        model_lock_sha256=model_lock.sha256,
        drafter_model_id=drafter.model_id,
        prepared_drafter_revision=drafter.revision,
        dspark_native_heads=None,
        parameters=tuple(
            PreparedParameterMetadata(
                name=tensor.name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                ownership="sharded",
            )
            for tensor in snapshot.tensors
        ),
    )
    if rebound_inventory != inventory:
        raise RuntimeError("prepared drafter inventory changed after publication")
    inventory_path = _publish_bound_json(
        directory,
        "prepared-drafter-inventory.json",
        inventory.to_dict(),
    )
    manifest = materialize_trainable_plan_authority_manifest(
        model_lock_artifact=model_lock_path,
        prepared_drafter_artifact=inventory_path,
        run_config_artifact=run_config_path,
        split_artifact=split_path,
        cell_artifact=cell_path,
        prepared_model_content_authority=prepared_content_authority,
        execution_semantics=semantics,
    )
    manifest_path = _publish_bound_json(
        directory,
        "trainable-plan-manifest.json",
        manifest,
    )
    binding = bind_trainable_plan_authority(
        manifest_path,
        prepared_model_content_authority=prepared_content_authority,
        expected_execution_semantics_sha256=semantics.sha256,
    )
    if {
        row.model_id: (row.revision, row.root)
        for row in binding.prepared_model_content_authority.prepared_model_set.snapshots
    } != {
        target.model_id: (target.revision, target.local_snapshot_path),
        drafter.model_id: (drafter.revision, drafter.local_snapshot_path),
    }:
        raise RuntimeError("published trainable plan detached from trusted content")

    publish_canonical_json_no_replace(
        output,
        trainable_plan_authority_binding_to_dict(binding),
    )
    proof = CanonicalJsonProofBinding.bind(output)
    reopened = trainable_plan_authority_binding_from_dict(proof.reopen())
    replayed = reopened.revalidate()
    if (
        reopened != binding
        or replayed.binding != binding
        or proof.semantic_sha256 != binding.sha256
    ):
        raise RuntimeError("published trainable-plan authority changed")
    return binding


def publish_trusted_tts_calibration_trainable_plan_authority(
    *,
    trusted_content_bundle_path: str | Path,
    output_path: str | Path,
) -> TrainablePlanAuthorityBinding:
    return _publish_trusted_structural_trainable_plan(
        trusted_content_bundle_path=trusted_content_bundle_path,
        output_path=output_path,
        semantics=build_tts_calibration_trainable_plan_semantics(),
    )


def publish_trusted_e1_recipe_anchor_trainable_plan_authority(
    *,
    trusted_content_bundle_path: str | Path,
    output_path: str | Path,
) -> TrainablePlanAuthorityBinding:
    return _publish_trusted_structural_trainable_plan(
        trusted_content_bundle_path=trusted_content_bundle_path,
        output_path=output_path,
        semantics=build_e1_recipe_anchor_trainable_plan_semantics(),
    )


__all__ = [
    "publish_trusted_e1_recipe_anchor_trainable_plan_authority",
    "publish_trusted_tts_calibration_trainable_plan_authority",
]
