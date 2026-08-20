"""Source-owned downstream prepared-launch production.

This module is the trusted single-operator counterpart of the legacy signed
launch mapper.  It deliberately accepts paths, never scientific values.  The
current execution source owns the cell universe and frozen recipes; a runtime
BOUND content bundle owns model/data paths; the doctor and inventory embedded
in that bundle own the machine; and completed prerequisite launch manifests
own backend/model/topology-specific runtime details.

Request tokenization is a durable second phase.  :func:`prepare_launch_draft`
publishes every GPU-allocation input except a request-schedule receipt.
``finalize_prepared_launch_bundle`` accepts only path-bound receipts and then
runs the canonical prepared-bundle revalidator.  This prevents a caller from
injecting prompts, loads, seeds, widths, recipes, or placements while allowing
the tokenizer worker to run out of process.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
    )

from lightcone_spec.config import (
    AdaptationConfig,
    OnlineSpecConfig,
    OptimizerConfig,
    RunConfig,
    load_run_config,
    run_config_sha256,
)
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_from_dict,
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
    FORMAL_SINGLE_OPERATOR_PREPARED_ENTRY_SHARD_ARTIFACT_KIND,
    TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256,
    TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256,
    TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256,
    FormalSingleOperatorPreparedLaunchBlocked,
    FormalSingleOperatorPreparedLaunchBundle,
    FormalSingleOperatorPreparedLaunchEntry,
    FormalSingleOperatorProfilerSubjectRequirement,
    _trusted_adaptation_group,
    _trusted_chain_recipe_context,
    _trusted_expected_load,
    _trusted_expected_width,
    _trusted_geometry,
    _validate_trusted_chain_run_config,
    formal_single_operator_launch_compatibility_key,
    formal_single_operator_prepared_entries_artifact_id,
    formal_single_operator_prepared_execution_identities,
    revalidate_formal_single_operator_prepared_launch_bundle,
    shard_formal_single_operator_prepared_launch_bundle,
)
from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
    route_formal_single_operator_materialized_cell,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorExecutionSource,
    load_formal_single_operator_execution_source,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.onlinespec import onlinespec_candidates
from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.stage_materialization import (
    E1A_FIXED_VERIFICATION_BUDGET,
    MaterializedCell,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration.runtime import (
    _render_server,
    derive_diagnostic_compile_cache_key,
)
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
)
from lightcone_spec.runtime.compile_runner import (
    TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileLaunchManifest,
    write_compile_prewarm_manifest,
)
from lightcone_spec.runtime.formal_sharded_artifact import (
    FormalCanonicalSequenceShardIndex,
    load_formal_canonical_sequence_shard_index,
    publish_formal_canonical_sequence_shards,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

PreparedProducerPhysicalKind = Literal["serving", "profiler", "e5_failure"]

_ADAPTATION_RESERVE_MB = 4096
_SHA256 = frozenset("0123456789abcdef")


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


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _absolute_existing_directory(label: str, value: str | Path) -> Path:
    requested = Path(value)
    resolved = requested.resolve(strict=False)
    if (
        not requested.is_absolute()
        or requested != resolved
        or not resolved.is_dir()
        or resolved.is_symlink()
    ):
        raise ValueError(f"{label} must be an existing normalized directory")
    return resolved


def _flag_value(argv: tuple[str, ...], flag: str) -> str:
    positions = tuple(index for index, value in enumerate(argv) if value == flag)
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(argv)
        or argv[positions[0] + 1].startswith("--")
    ):
        raise ValueError(f"prerequisite launch lacks exact {flag}")
    return argv[positions[0] + 1]


FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_PRODUCER_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_prepared_launch_producer",
        "source": "exact_current_execution_materialization_and_predecessor_chain",
        "content": "runtime_BOUND_tagged_trusted_single_operator_bundle",
        "runtime": "bundle_bound_inventory_and_complete_PASS_doctor",
        "prerequisite": (
            "exact_same_stage_model_backend_topology_schema2_compile_launch"
        ),
        "configuration": "frozen_chain_and_materialized_cell_only",
        "controlled_context": (
            "E3b_E6_prompt_plus_requested_generation_exact_budget_from_one_"
            "draft_owned_LCB_MATH_filler_authority_per_tokenizer_member"
        ),
        "placement": (
            "block_modulo_balanced_then_paired_identity_hash_over_sorted_"
            "ready_inventory"
        ),
        "publication": (
            "per_cell_run_config_sampling_model_lock_cache_prewarm_launch_"
            "then_durable_schedule_then_canonical_bundle"
        ),
        "forbidden": (
            "caller_scientific_scalars",
            "foreign_model_backend_topology_template",
            "unbound_content_or_runtime",
            "replacement",
        ),
    }
)
TRUSTED_SINGLE_OPERATOR_SHARDED_PREPARED_LAUNCH_DRAFT_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 2,
        "kind": "formal_single_operator_prepared_launch_draft_protocol",
        "header": "same_exact_source_content_runtime_chain_as_schema1",
        "entries": "sorted_materialized_cell_id_canonical_sequence_shards",
        "index": "small_path_bound_index_without_payload_rows",
        "lookup": "materialization_owned_cell_ordinal_opens_one_shard",
        "complete_audit": "all_shards_deep_replayed_in_exact_cell_order",
        "legacy": "schema1_serialization_unchanged",
    }
)

_PREPARED_LAUNCH_DRAFT_ENTRY_SHARD_ARTIFACT_KIND = (
    "formal_single_operator_prepared_launch_draft_entries"
)


@dataclass(frozen=True, order=True)
class PreparedLaunchRuntimeKey:
    """Compatibility key used only to select an exact prerequisite launch."""

    stage: str
    target_model_id: str
    target_revision: str
    drafter_model_id: str
    drafter_revision: str
    tokenizer_model_id: str
    tokenizer_revision: str
    backend: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]

    def __post_init__(self) -> None:
        for label, value in (
            ("stage", self.stage),
            ("target model", self.target_model_id),
            ("target revision", self.target_revision),
            ("drafter model", self.drafter_model_id),
            ("drafter revision", self.drafter_revision),
            ("tokenizer model", self.tokenizer_model_id),
            ("tokenizer revision", self.tokenizer_revision),
            ("backend", self.backend),
        ):
            _require_text(f"prepared runtime {label}", value)
        if self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("prepared runtime topology is unsupported")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def prerequisite_runtime_key(
    launch: CompileLaunchManifest,
) -> PreparedLaunchRuntimeKey:
    """Extract a key; never accept a caller-authored compatibility label."""

    if type(launch) is not CompileLaunchManifest:
        raise TypeError("prerequisite runtime key requires an exact launch")
    launch.validate(reopen_inputs=True)
    config = load_run_config(launch.run_config_path)
    if launch.schema_version not in {2, 3} or launch.formal_stage is None:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "compatible_trusted_prerequisite_launch_missing"
        )
    return PreparedLaunchRuntimeKey(
        stage=launch.formal_stage,
        target_model_id=launch.target_model_id,
        target_revision=launch.target_revision,
        drafter_model_id=config.model.drafter,
        drafter_revision=config.model.drafter_revision,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        backend=config.model.algorithm,
        topology_mode=config.runtime.topology_mode,
    )


def _cell_backend(cell: MaterializedCell) -> str:
    dimensions = dict(cell.dimensions)
    value = {
        "E3b": "DFLASH",
        "E1a": "DSPARK",
        "E5": dimensions.get("backend_authority", cell.backend),
        "E6": "NEXTN",
        "E0": cell.backend,
    }.get(cell.stage, cell.backend)
    if value == "NONE":
        value = {"E3b": "DFLASH", "E1a": "DSPARK", "E6": "NEXTN"}.get(cell.stage)
    if type(value) is not str or value not in {
        "DFLASH",
        "DSPARK",
        "EAGLE3",
        "NEXTN",
    }:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_backend_runtime_authority_missing"
        )
    return value


def _cell_topology(cell: MaterializedCell) -> Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]:
    raw = dict(cell.dimensions).get(
        "topology", "tp2_dp1" if cell.stage == "E6" else "tp1_dp1"
    )
    if raw not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
        raise ValueError("materialized cell topology is unsupported")
    return raw  # type: ignore[return-value]


def _pairing_identity(cell: MaterializedCell) -> str:
    dimensions = dict(cell.dimensions)
    axes = {
        key: value
        for key, value in dimensions.items()
        if key
        not in {
            "tts_l0_pair_id",
            "candidate_id",
            "onlinespec_method",
            "p99_anchor_id",
            "p99_minimum_completions",
            "p99_selection_receipt_sha256",
        }
    }
    return _sha256(
        {
            "stage": cell.stage,
            "model": cell.model,
            "backend": _cell_backend(cell),
            "task": cell.task,
            "topology": _cell_topology(cell),
            "dimensions": axes,
        }
    )


def deterministic_prepared_gpu_assignment(
    *, inventory: GpuInventory, cell: MaterializedCell
) -> tuple[str, ...]:
    """Place paired rows together while balancing independent pair groups."""

    if type(inventory) is not GpuInventory or type(cell) is not MaterializedCell:
        raise TypeError("prepared placement requires exact inventory/cell")
    ready = tuple(sorted(device.uuid for device in inventory.devices if device.ready))
    if not ready:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_ready_gpu_inventory_empty"
        )
    topology = _cell_topology(cell)
    if topology == "tp1_dp1":
        block = dict(cell.dimensions).get("block")
        lane = (
            block % len(ready)
            if type(block) is int and block >= 0
            else int(_pairing_identity(cell)[:16], 16) % len(ready)
        )
        return (ready[lane],)
    if len(ready) != 2:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_two_gpu_topology_unavailable"
        )
    groups = tuple(
        group
        for group in inventory.topology_groups
        if set(group.gpu_uuids) == set(ready)
    )
    if len(groups) != 1:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_two_gpu_topology_group_missing"
        )
    return tuple(groups[0].gpu_uuids)


def _chronobelief_qualified_gpu_assignment(
    *,
    inventory: GpuInventory,
    cell: MaterializedCell,
    qualified_gpu_uuids: tuple[str, ...],
) -> tuple[str]:
    """Select a TP1 placement only from GPUs with actual parity evidence."""

    if (
        type(inventory) is not GpuInventory
        or type(cell) is not MaterializedCell
        or type(qualified_gpu_uuids) is not tuple
        or not qualified_gpu_uuids
        or qualified_gpu_uuids != tuple(sorted(set(qualified_gpu_uuids)))
    ):
        raise ValueError("ChronoBelief qualified GPU placement input differs")
    ready = {device.uuid for device in inventory.devices if device.ready}
    if set(qualified_gpu_uuids) - ready:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_chronobelief_proved_gpu_assignment_missing"
        )
    lane = int(
        _sha256(
            {
                "kind": "qualified_chronobelief_gpu_placement",
                "cell_id": cell.cell_id,
                "qualified_gpu_uuids": qualified_gpu_uuids,
            }
        )[:16],
        16,
    ) % len(qualified_gpu_uuids)
    return (qualified_gpu_uuids[lane],)


def _method(cell: MaterializedCell) -> str:
    from lightcone_spec.experiments.formal_stage_execution import _method_for_cell

    return _method_for_cell(cell)


def _runtime_load(context: object, cell: MaterializedCell, base: RunConfig) -> int:
    if cell.stage == "E5":
        dimensions = dict(cell.dimensions)
        if cell.task == "deterministic_failure_injection":
            return int(context.common_load)
        family = dimensions.get("family")
        if family == "closed_loop":
            value = dimensions.get("concurrency")
            if type(value) is not int or value < 1:
                raise ValueError("E5 closed-loop cell lacks concurrency")
            return value
        if family in {"open_loop", "trace_or_soak", "topology_cohort"}:
            return int(context.common_load)
        raise ValueError("E5 cell lacks a registered load family")
    value = _trusted_expected_load(context, cell)
    return base.runtime.max_running_requests if value is None else value


def _eagle3_authority(
    source: FormalSingleOperatorExecutionSource,
    cell: MaterializedCell,
    base: RunConfig,
) -> dict[str, str]:
    """Deep-replay the exact task authority for adaptive E0 EAGLE3.

    The prerequisite E0 interface launch is intentionally a generic static
    smoke launch.  Adaptive claims therefore cannot be copied from that
    ``RunConfig``.  They are projected from the current execution source's
    schema-2 compatibility auxiliary, keyed by the materialized task.
    """

    if base.model.algorithm != "EAGLE3":
        return {}
    if source.stage != "E0" or cell.stage != "E0" or cell.backend != "EAGLE3":
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_eagle3_runtime_authority_outside_e0"
        )
    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        e0_eagle3_runtime_authority_for_task,
        e0_eagle3_runtime_proof_row_for_task,
        load_e0_compatibility_probe_terminal,
        load_e0_prepared_model_backend_interface_receipt,
        revalidate_trusted_e0_compatibility_bundle_value,
    )

    auxiliary = source.auxiliary_source_binding("e0_compatibility")
    publication = revalidate_trusted_e0_compatibility_bundle_value(
        auxiliary.reopen(label="prepared E0 compatibility auxiliary")
    )
    decisions = tuple(
        row
        for row in publication.compatibility.decisions
        if (row.model, row.backend, row.task) == (cell.model, "EAGLE3", cell.task)
    )
    if len(decisions) != 1 or decisions[0].disposition != "VALID":
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_eagle3_runtime_authority_missing"
        )
    receipt_bindings = tuple(
        binding
        for binding in publication.evidence_manifest.interface_receipts
        if (
            (raw := binding.reopen()).get("model") == cell.model
            and raw.get("backend") == "EAGLE3"
        )
    )
    if len(receipt_bindings) != 1:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_eagle3_interface_receipt_missing"
        )
    receipt = load_e0_prepared_model_backend_interface_receipt(
        receipt_bindings[0].absolute_path
    )
    if receipt.compile_launch_manifest is None:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_eagle3_interface_launch_missing"
        )
    interface_launch = CompileLaunchManifest.load(
        receipt.compile_launch_manifest.absolute_path
    )
    interface_config = load_run_config(interface_launch.run_config_path)
    if (
        interface_config != base
        or interface_launch.run_config_semantic_sha256 != run_config_sha256(base)
        or receipt.target_model_id != base.model.target
        or receipt.target_revision != base.model.target_revision
        or receipt.drafter_model_id != base.model.drafter
        or receipt.drafter_revision != base.model.drafter_revision
    ):
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_eagle3_interface_launch_differs"
        )
    terminal_bindings = tuple(
        binding
        for binding in publication.evidence_manifest.probe_terminals
        if (
            (raw := binding.reopen()).get("model") == cell.model
            and raw.get("backend") == "EAGLE3"
            and raw.get("task") == cell.task
        )
    )
    if len(terminal_bindings) != 1:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_eagle3_task_terminal_missing"
        )
    terminal = load_e0_compatibility_probe_terminal(terminal_bindings[0].absolute_path)
    proof_row = e0_eagle3_runtime_proof_row_for_task(
        receipt,
        task=cell.task,
        terminal=terminal,
    )
    if (
        terminal.disposition != "VALID"
        or terminal.interface_receipt_sha256 != receipt.sha256
        or terminal.eagle3_runtime_proof_row_sha256 != proof_row.sha256
    ):
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_eagle3_task_authority_differs"
        )
    return e0_eagle3_runtime_authority_for_task(
        receipt,
        task=cell.task,
        terminal=terminal,
    )


def derive_prepared_run_config(
    *,
    source: FormalSingleOperatorExecutionSource,
    cell: MaterializedCell,
    prerequisite: RunConfig,
    gpu_uuids: tuple[str, ...],
    trusted_chronobelief_gpu_parity_proof_sha256: str | None = None,
) -> RunConfig:
    """Derive every scientific field from the current cell/frozen chain."""

    if (
        type(source) is not FormalSingleOperatorExecutionSource
        or type(cell) is not MaterializedCell
        or type(prerequisite) is not RunConfig
    ):
        raise TypeError("prepared RunConfig derivation requires exact inputs")
    context = _trusted_chain_recipe_context(source)
    backend = _cell_backend(cell)
    topology = _cell_topology(cell)
    if (
        prerequisite.model.target != cell.model
        or prerequisite.model.algorithm != backend
        or prerequisite.runtime.topology_mode != topology
        or len(gpu_uuids) != (1 if topology == "tp1_dp1" else 2)
    ):
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "compatible_model_backend_topology_prerequisite_missing"
        )
    if cell.task == "mechanism_profile_only":
        if (
            cell.stage != "E4"
            or prerequisite.method != "l0"
            or prerequisite.adaptation is None
        ):
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_profiler_selected_launch_missing"
            )
        # The canonical profiler revalidator requires an exact clone of the
        # selected E4-local saturation/mixed headline configuration.  Only the
        # telemetry mode and per-cell adaptation namespace may change.
        return RunConfig.model_validate(
            prerequisite.model_copy(
                update={
                    "runtime": prerequisite.runtime.model_copy(
                        update={"telemetry_detail": "profile"}
                    ),
                    "adaptation": prerequisite.adaptation.model_copy(
                        update={
                            "adaptation_group_id": (
                                f"formal-single-e4-{cell.cell_id[:24]}"
                            )
                        }
                    ),
                }
            ).model_dump(mode="json")
        )
    dimensions = dict(cell.dimensions)
    if cell.stage == "E6":
        for name, actual in (
            ("target_model_id", prerequisite.model.target),
            ("target_revision", prerequisite.model.target_revision),
            ("drafter_model_id", prerequisite.model.drafter),
            ("drafter_revision", prerequisite.model.drafter_revision),
            ("nextn_mtp_mode", prerequisite.model.nextn_mtp_mode),
            (
                "target_snapshot_sha256",
                prerequisite.model.target_snapshot_sha256,
            ),
            ("mtp_component_sha256", prerequisite.model.mtp_component_sha256),
        ):
            if (
                name in dimensions
                or prerequisite.model.nextn_mtp_mode == "built_in_mtp"
            ) and dimensions.get(name) != actual:
                raise ValueError("E6 prerequisite differs from interface authority")
    width = _trusted_expected_width(context, cell, prerequisite)
    load = _runtime_load(context, cell, prerequisite)
    runtime_update: dict[str, object] = {
        "device_identity": ",".join(gpu_uuids),
        "max_running_requests": load,
        "speculation_enabled": _method(cell) != "target_only",
        "speculative_num_draft_tokens": width,
        "telemetry_detail": "profile"
        if cell.task == "mechanism_profile_only"
        else "headline",
        "rendezvous_identity": f"formal-single-{cell.cell_id[:24]}",
        "router_identity": (
            f"formal-single-sticky-{cell.cell_id[:24]}"
            if topology == "tp1_dp2"
            else "single-replica"
        ),
    }
    if cell.stage != "E4":
        runtime_update.update(
            {
                "adaptation_microbatch_size": 1,
                "adaptation_publication_coalescing": 1,
                "adaptation_stream_priority": "default",
            }
        )
    runtime = prerequisite.runtime.model_copy(update=runtime_update)
    model = prerequisite.model.model_copy(update={"draft_depth": width - 1})
    method = _method(cell)
    if method in {"target_only", "static"}:
        result = RunConfig(
            method=method,
            model=model,
            runtime=runtime,
            tenant_id=prerequisite.tenant_id,
        )
        _validate_trusted_chain_run_config(
            context=context, source=source, cell=cell, config=result
        )
        return result

    eagle3 = _eagle3_authority(source, cell, prerequisite)
    if cell.method_role in {"TTS", "L0-naive"}:
        adaptation = AdaptationConfig(
            weight_update_mode="full",
            parameter_scope="all",
            reset_scope="request",
            request_admission_policy="serialized_native_scheduler_v1",
            adaptation_group_id=_trusted_adaptation_group(cell, paired_tts_l0=True),
            optimizer=OptimizerConfig(
                name="adam",
                learning_rate=context.tts_learning_rate,
                weight_decay=0.0,
                grad_clip=None,
                schedule="constant",
            ),
            stride=context.tts_stride,
            canvas_tokens=width,
            loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
            **eagle3,
        )
        result = RunConfig(
            method=method,
            model=model,
            runtime=runtime,
            adaptation=adaptation,
            tenant_id=prerequisite.tenant_id,
        )
        _validate_trusted_chain_run_config(
            context=context, source=source, cell=cell, config=result
        )
        return result

    if cell.method_role.startswith("OnlineSPEC-"):
        candidates = {row.candidate_id: row for row in onlinespec_candidates()}
        candidate = candidates.get(cell.recipe_sha256 or "")
        if candidate is None:
            raise ValueError("OnlineSPEC cell lacks a registered recipe")
        adaptation = AdaptationConfig(
            weight_update_mode=candidate.weight_update_mode,
            parameter_scope=candidate.parameter_scope,
            reset_scope="cohort",
            request_admission_policy="cohort_batching_v1",
            adaptation_group_id=f"e0:{cell.cell_id}",
            optimizer=OptimizerConfig(
                name="sgd",
                learning_rate=candidate.learning_rate,
                weight_decay=0.0,
                grad_clip=candidate.grad_clip,
            ),
            rank=candidate.rank,
            lora_alpha=(
                candidate.rank if candidate.weight_update_mode == "lora" else None
            ),
            stride=candidate.stride,
            canvas_tokens=width,
            loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
            **eagle3,
        )
        result = RunConfig(
            method=method,
            model=model,
            runtime=runtime,
            adaptation=adaptation,
            online_spec=OnlineSpecConfig(
                projection_radius=candidate.projection_radius,
                additional_learning_rates=candidate.additional_learning_rates,
                hedge_learning_rate=candidate.hedge_learning_rate,
            ),
            tenant_id=prerequisite.tenant_id,
        )
        _validate_trusted_chain_run_config(
            context=context, source=source, cell=cell, config=result
        )
        return result

    if cell.method_role not in {"LightCone", "LightCone-candidate"}:
        raise ValueError("prepared adaptive method role is unsupported")
    recipe = context.lightcone_recipe
    chronobelief_proof = None
    if recipe.optimizer == "chronobelief":
        if cell.stage != "E1a" or trusted_chronobelief_gpu_parity_proof_sha256 is None:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_chronobelief_gpu_proof_missing"
            )
        chronobelief_proof = _require_sha256(
            "source-owned ChronoBelief GPU proof",
            trusted_chronobelief_gpu_parity_proof_sha256,
        )
    elif trusted_chronobelief_gpu_parity_proof_sha256 is not None:
        raise ValueError("non-ChronoBelief recipe cannot consume GPU parity proof")
    adaptation = default_e2_recipe_grid_authority().adaptation_config_for(
        recipe,
        canvas_tokens=width,
        adaptation_group_id=_trusted_adaptation_group(cell, paired_tts_l0=False),
        chronobelief_gpu_proof_sha256=chronobelief_proof,
    )
    configuration = None
    if cell.stage == "E1a":
        configuration = (
            ("parameterization", dimensions.get("parameterization")),
            ("rank", dimensions.get("rank")),
            ("scope", dimensions.get("scope")),
        )
    elif cell.stage == "E5" and backend == "DSPARK":
        configuration = context.dspark_selected_configuration
        if configuration is None or context.dspark_selected_recipe_sha256 is None:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "trusted_dspark_winner_missing"
            )
    if configuration is not None:
        parameterization, scope, rank, native_head_policy = _trusted_geometry(
            configuration
        )
        verification = dimensions.get("verification_mode")
        adaptation = AdaptationConfig(
            **{
                **adaptation.model_dump(),
                "weight_update_mode": parameterization,
                "parameter_scope": scope,
                "rank": rank,
                "lora_alpha": rank if parameterization == "lora" else None,
                "native_head_policy": native_head_policy,
                "verification_mode": (
                    "fixed_budget"
                    if verification == "fixed_verification_budget"
                    else "native_scheduler"
                ),
                "fixed_verification_budget": (
                    E1A_FIXED_VERIFICATION_BUDGET
                    if verification == "fixed_verification_budget"
                    else None
                ),
            }
        )
    if cell.stage == "E4" and cell.task != "mechanism_profile_only":
        adaptation = adaptation.model_copy(
            update={"stride": int(dimensions["update_stride"])}
        )
        runtime = runtime.model_copy(
            update={
                "adaptation_microbatch_size": int(dimensions["microbatch"]),
                "adaptation_publication_coalescing": int(dimensions["coalescing"]),
                "adaptation_stream_priority": dimensions["stream_priority"],
            }
        )
    if eagle3:
        adaptation = AdaptationConfig(**{**adaptation.model_dump(), **eagle3})
    result = RunConfig(
        method=method,
        model=model,
        runtime=runtime,
        adaptation=adaptation,
        tenant_id=prerequisite.tenant_id,
    )
    _validate_trusted_chain_run_config(
        context=context,
        source=source,
        cell=cell,
        config=result,
        trusted_chronobelief_gpu_parity_proof_sha256=chronobelief_proof,
    )
    return result


@dataclass(frozen=True)
class PreparedLaunchDraftEntry:
    materialized_cell_id: str
    physical_kind: PreparedProducerPhysicalKind
    prerequisite_launch: CanonicalJsonProofBinding
    run_config: CanonicalJsonProofBinding
    compile_cache_plan: CanonicalJsonProofBinding
    prewarm_manifest: CanonicalJsonProofBinding
    sampling_profile: CanonicalJsonProofBinding
    compile_launch_manifest: CanonicalJsonProofBinding
    launch_compatibility_key_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    gpu_uuids: tuple[str, ...]
    schedule_state: Literal["PENDING_DURABLE_TOKENIZATION"]
    trusted_chronobelief_gpu_parity_proof: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        _require_sha256("prepared draft cell", self.materialized_cell_id)
        _require_sha256(
            "prepared draft compatibility", self.launch_compatibility_key_sha256
        )
        if self.physical_kind not in {"serving", "profiler", "e5_failure"}:
            raise ValueError("prepared draft physical kind differs")
        for value in (
            self.prerequisite_launch,
            self.run_config,
            self.compile_cache_plan,
            self.prewarm_manifest,
            self.sampling_profile,
            self.compile_launch_manifest,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("prepared draft artifact is not path-bound")
            if CanonicalJsonProofBinding.bind(value.absolute_path) != value:
                raise ValueError("prepared draft artifact changed")
        if self.trusted_chronobelief_gpu_parity_proof is not None and (
            type(self.trusted_chronobelief_gpu_parity_proof)
            is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(
                self.trusted_chronobelief_gpu_parity_proof.absolute_path
            )
            != self.trusted_chronobelief_gpu_parity_proof
        ):
            raise ValueError("prepared draft ChronoBelief proof changed")
        expected = 1 if self.topology_mode == "tp1_dp1" else 2
        if len(self.gpu_uuids) != expected or len(set(self.gpu_uuids)) != expected:
            raise ValueError("prepared draft GPU assignment differs")
        if self.schedule_state != "PENDING_DURABLE_TOKENIZATION":
            raise ValueError("prepared draft schedule state differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "materialized_cell_id": self.materialized_cell_id,
            "physical_kind": self.physical_kind,
            "prerequisite_launch": self.prerequisite_launch.to_dict(),
            "run_config": self.run_config.to_dict(),
            "compile_cache_plan": self.compile_cache_plan.to_dict(),
            "prewarm_manifest": self.prewarm_manifest.to_dict(),
            "sampling_profile": self.sampling_profile.to_dict(),
            "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
            "launch_compatibility_key_sha256": (self.launch_compatibility_key_sha256),
            "topology_mode": self.topology_mode,
            "gpu_uuids": list(self.gpu_uuids),
            "schedule_state": self.schedule_state,
            "trusted_chronobelief_gpu_parity_proof": (
                None
                if self.trusted_chronobelief_gpu_parity_proof is None
                else self.trusted_chronobelief_gpu_parity_proof.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict("prepared draft entry", value, set(cls.__dataclass_fields__))
        for name in (
            "prerequisite_launch",
            "run_config",
            "compile_cache_plan",
            "prewarm_manifest",
            "sampling_profile",
            "compile_launch_manifest",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_gpus = row.pop("gpu_uuids")
        raw_chronobelief = row.pop("trusted_chronobelief_gpu_parity_proof")
        if type(raw_gpus) is not list:
            raise TypeError("prepared draft GPU assignment is not an array")
        return cls(
            **row,
            gpu_uuids=tuple(raw_gpus),
            trusted_chronobelief_gpu_parity_proof=(
                None
                if raw_chronobelief is None
                else CanonicalJsonProofBinding.from_dict(raw_chronobelief)
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreparedLaunchDraft:
    schema_version: Literal[1, 2]
    kind: Literal["formal_single_operator_prepared_launch_draft"]
    protocol_sha256: str
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    materialization: CanonicalJsonProofBinding
    materialization_sha256: str
    content_source_binding: FormalContentSourceBinding
    inventory: CanonicalJsonProofBinding
    inventory_sha256: str
    doctor: CanonicalJsonProofBinding
    entries: tuple[PreparedLaunchDraftEntry, ...]
    entries_shard_index: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2}
            or self.kind != "formal_single_operator_prepared_launch_draft"
            or self.protocol_sha256
            != (
                FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_PRODUCER_PROTOCOL_SHA256
                if self.schema_version == 1
                else (
                    TRUSTED_SINGLE_OPERATOR_SHARDED_PREPARED_LAUNCH_DRAFT_PROTOCOL_SHA256
                )
            )
        ):
            raise ValueError("prepared launch draft schema differs")
        for label, value in (
            ("execution source", self.execution_source_sha256),
            ("materialization", self.materialization_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"prepared draft {label}", value)
        if (
            type(self.content_source_binding) is not FormalContentSourceBinding
            or self.content_source_binding.mode != "trusted_single_operator"
        ):
            raise ValueError("prepared draft content source is not trusted/BOUND")
        for value in (
            self.execution_source,
            self.materialization,
            self.inventory,
            self.doctor,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("prepared draft source is not path-bound")
            if CanonicalJsonProofBinding.bind(value.absolute_path) != value:
                raise ValueError("prepared draft source changed")
        ids = tuple(row.materialized_cell_id for row in self.entries)
        if self.schema_version == 1:
            if self.entries_shard_index is not None or ids != tuple(sorted(set(ids))):
                raise ValueError("prepared draft entries are not canonical")
        elif (
            self.entries
            or type(self.entries_shard_index) is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(self.entries_shard_index.absolute_path)
            != self.entries_shard_index
        ):
            raise ValueError("sharded prepared draft must bind only its entry index")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "execution_source": self.execution_source.to_dict(),
            "execution_source_sha256": self.execution_source_sha256,
            "materialization": self.materialization.to_dict(),
            "materialization_sha256": self.materialization_sha256,
            "content_source_binding": self.content_source_binding.to_dict(),
            "inventory": self.inventory.to_dict(),
            "inventory_sha256": self.inventory_sha256,
            "doctor": self.doctor.to_dict(),
        }
        if self.schema_version == 1:
            value["entries"] = [row.to_dict() for row in self.entries]
        else:
            assert self.entries_shard_index is not None
            value["entries_shard_index"] = self.entries_shard_index.to_dict()
        if include_sha256:
            value["draft_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("prepared launch draft must be an object")
        schema_version = value.get("schema_version")
        expected = set(cls.__dataclass_fields__) | {"draft_sha256"}
        if schema_version == 1:
            expected.remove("entries_shard_index")
        elif schema_version == 2:
            expected.remove("entries")
        row = _strict(
            "prepared launch draft",
            value,
            expected,
        )
        declared = _require_sha256("prepared launch draft", row.pop("draft_sha256"))
        for name in ("execution_source", "materialization", "inventory", "doctor"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["content_source_binding"] = FormalContentSourceBinding.from_dict(
            row["content_source_binding"]
        )
        raw_entries = row.pop("entries", [])
        if type(raw_entries) is not list:
            raise TypeError("prepared launch draft entries are not an array")
        raw_index = row.pop("entries_shard_index", None)
        row["entries_shard_index"] = (
            None
            if raw_index is None
            else CanonicalJsonProofBinding.from_dict(raw_index)
        )
        result = cls(
            **row,
            entries=tuple(
                PreparedLaunchDraftEntry.from_dict(item) for item in raw_entries
            ),
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("prepared launch draft digest differs")
        return result


def prepared_launch_draft_entries_artifact_id(
    *,
    draft: FormalSingleOperatorPreparedLaunchDraft,
    materialized_cell_ids: tuple[str, ...],
) -> str:
    """Derive the non-circular identity for one draft's entry sequence."""

    if type(draft) is not FormalSingleOperatorPreparedLaunchDraft:
        raise TypeError("prepared draft entry artifact requires an exact draft")
    if (
        type(materialized_cell_ids) is not tuple
        or not materialized_cell_ids
        or materialized_cell_ids != tuple(sorted(set(materialized_cell_ids)))
    ):
        raise ValueError("prepared draft sharded cell IDs are not canonical")
    for cell_id in materialized_cell_ids:
        _require_sha256("prepared draft sharded cell", cell_id)
    return _sha256(
        {
            "schema_version": 1,
            "kind": _PREPARED_LAUNCH_DRAFT_ENTRY_SHARD_ARTIFACT_KIND,
            "execution_source_sha256": draft.execution_source_sha256,
            "materialization_sha256": draft.materialization_sha256,
            "content_source_binding_sha256": (draft.content_source_binding.sha256),
            "content_bundle_sha256": (draft.content_source_binding.content_sha256),
            "inventory_sha256": draft.inventory_sha256,
            "doctor_raw_sha256": draft.doctor.raw_sha256,
            "doctor_semantic_sha256": draft.doctor.semantic_sha256,
            "ordered_materialized_cell_ids_sha256": _sha256(materialized_cell_ids),
            "entry_count": len(materialized_cell_ids),
        }
    )


def _expected_draft_cell_ids(
    draft: FormalSingleOperatorPreparedLaunchDraft,
) -> tuple[str, ...]:
    source = load_formal_single_operator_execution_source(
        draft.execution_source.absolute_path
    )
    materialization = stage_materialization_receipt_from_dict(
        draft.materialization.reopen()
    )
    if (
        source.sha256 != draft.execution_source_sha256
        or materialization.sha256 != draft.materialization_sha256
    ):
        raise ValueError("prepared draft source/materialization identity changed")
    return tuple(
        sorted(
            cell.cell_id
            for cell in materialization.cells
            if route_formal_single_operator_materialized_cell(
                node=source.node,
                phase=source.phase,
                cell=cell,
            ).physical_kind
            not in {"e6_interface_preflight", "e0_compatibility_decision"}
        )
    )


def _load_prepared_draft_entry_shard_index(
    *,
    draft: FormalSingleOperatorPreparedLaunchDraft,
    materialized_cell_ids: tuple[str, ...],
    deep: bool,
) -> FormalCanonicalSequenceShardIndex:
    if draft.schema_version != 2 or draft.entries_shard_index is None:
        raise ValueError("prepared launch draft is not sharded schema 2")
    index = load_formal_canonical_sequence_shard_index(
        draft.entries_shard_index.absolute_path,
        deep=deep,
    )
    if (
        index.artifact_kind != _PREPARED_LAUNCH_DRAFT_ENTRY_SHARD_ARTIFACT_KIND
        or index.artifact_id
        != prepared_launch_draft_entries_artifact_id(
            draft=draft,
            materialized_cell_ids=materialized_cell_ids,
        )
        or index.total_rows != len(materialized_cell_ids)
        or draft.entries_shard_index.semantic_sha256 != _sha256(index.to_dict())
    ):
        raise ValueError("prepared draft entry shard index identity differs")
    return index


def _prepared_draft_entries(
    draft: FormalSingleOperatorPreparedLaunchDraft,
    *,
    materialized_cell_id: str | None = None,
    deep: bool = False,
) -> tuple[PreparedLaunchDraftEntry, ...]:
    required_cell_ids = _expected_draft_cell_ids(draft)
    if materialized_cell_id is not None:
        _require_sha256("prepared draft requested cell", materialized_cell_id)
        if materialized_cell_id not in required_cell_ids:
            raise ValueError("prepared draft requested cell is absent")
    if draft.schema_version == 1:
        if (
            tuple(row.materialized_cell_id for row in draft.entries)
            != required_cell_ids
        ):
            raise ValueError("prepared draft entry coverage differs")
        return (
            draft.entries
            if materialized_cell_id is None
            else tuple(
                row
                for row in draft.entries
                if row.materialized_cell_id == materialized_cell_id
            )
        )
    index = _load_prepared_draft_entry_shard_index(
        draft=draft,
        materialized_cell_ids=required_cell_ids,
        deep=deep,
    )
    if materialized_cell_id is None:
        entries = tuple(
            PreparedLaunchDraftEntry.from_dict(row) for row in index.iter_rows()
        )
        if tuple(row.materialized_cell_id for row in entries) != required_cell_ids:
            raise ValueError("prepared draft sharded entry coverage differs")
        return entries
    ordinal = required_cell_ids.index(materialized_cell_id)
    entry = PreparedLaunchDraftEntry.from_dict(index.row_at(ordinal))
    if entry.materialized_cell_id != materialized_cell_id:
        raise ValueError("prepared draft sharded entry ordinal differs")
    return (entry,)


def _runtime_inputs(
    content_source: FormalContentSourceBinding,
) -> tuple[
    object,
    CanonicalJsonProofBinding,
    GpuInventory,
    CanonicalJsonProofBinding,
    dict[str, object],
]:
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
    )

    content = content_source.reopen()
    if (
        type(content) is not TrustedSingleOperatorContentBundle
        or content.runtime_binding_status != "BOUND"
        or content.runtime_observations is None
    ):
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "trusted_content_runtime_binding_missing"
        )
    observations = content.runtime_observations
    inventory_binding = CanonicalJsonProofBinding.bind(
        observations.inventory.absolute_path
    )
    doctor_binding = CanonicalJsonProofBinding.bind(observations.doctor.absolute_path)
    if (
        inventory_binding.raw_sha256 != observations.inventory.raw_sha256
        or inventory_binding.semantic_sha256 != observations.inventory.semantic_sha256
        or doctor_binding.raw_sha256 != observations.doctor.raw_sha256
        or doctor_binding.semantic_sha256 != observations.doctor.semantic_sha256
    ):
        raise RuntimeError("trusted runtime observation changed")
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    if inventory.sha256 != inventory_binding.semantic_sha256:
        raise ValueError("trusted inventory canonical identity differs")
    doctor = doctor_binding.reopen()
    if type(doctor) is not dict:
        raise TypeError("trusted doctor report is not an object")
    return content, inventory_binding, inventory, doctor_binding, doctor


def _prerequisites(
    paths: tuple[str | Path, ...],
    *,
    content_source: FormalContentSourceBinding,
    inventory: GpuInventory,
) -> dict[
    PreparedLaunchRuntimeKey, tuple[CanonicalJsonProofBinding, CompileLaunchManifest]
]:
    rows: dict[
        PreparedLaunchRuntimeKey,
        tuple[CanonicalJsonProofBinding, CompileLaunchManifest],
    ] = {}
    for path in paths:
        launch = CompileLaunchManifest.load(path)
        binding = CanonicalJsonProofBinding.bind(path, semantic_sha256=launch.sha256)
        key = prerequisite_runtime_key(launch)
        if (
            launch.content_source_binding != content_source
            or launch.inventory_sha256 != inventory.sha256
        ):
            raise ValueError("prerequisite launch belongs to another content/runtime")
        if key in rows:
            raise ValueError("prerequisite launch runtime key is ambiguous")
        rows[key] = (binding, launch)
    return rows


def _matching_prerequisite(
    *,
    source: FormalSingleOperatorExecutionSource,
    cell: MaterializedCell,
    rows: dict[
        PreparedLaunchRuntimeKey,
        tuple[CanonicalJsonProofBinding, CompileLaunchManifest],
    ],
) -> tuple[CanonicalJsonProofBinding, CompileLaunchManifest, RunConfig]:
    dimensions = dict(cell.dimensions)
    matches = []
    for key, value in rows.items():
        if (
            key.stage == source.stage
            and key.target_model_id == cell.model
            and key.backend == _cell_backend(cell)
            and key.topology_mode == _cell_topology(cell)
            and (
                cell.stage != "E6"
                or (
                    key.target_revision == dimensions.get("target_revision")
                    and key.drafter_model_id == dimensions.get("drafter_model_id")
                    and key.drafter_revision == dimensions.get("drafter_revision")
                )
            )
        ):
            matches.append(value)
    if len(matches) != 1:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "compatible_model_backend_topology_prerequisite_missing"
        )
    binding, launch = matches[0]
    return binding, launch, load_run_config(launch.run_config_path)


def _chronobelief_proofs(
    paths: tuple[str | Path, ...],
) -> dict[str, tuple[CanonicalJsonProofBinding, object]]:
    from lightcone_spec.experiments.formal_single_operator_chronobelief import (
        load_trusted_single_operator_chronobelief_gpu_parity_proof,
    )

    rows: dict[str, tuple[CanonicalJsonProofBinding, object]] = {}
    for path in paths:
        binding = CanonicalJsonProofBinding.bind(path)
        proof = load_trusted_single_operator_chronobelief_gpu_parity_proof(
            binding.absolute_path
        )
        key = proof.prerequisite_launch.semantic_sha256
        if key in rows:
            raise ValueError("ChronoBelief prerequisite proof is ambiguous")
        rows[key] = (binding, proof)
    return rows


def _sampling_for_cell(cell: MaterializedCell) -> SamplingProfile:
    return (
        SamplingProfile(purpose="natural", ignore_eos=False)
        if cell.stage in {"E1a", "E6", "E0"}
        else SamplingProfile()
    )


def _write_model_lock(config: RunConfig, path: Path) -> ModelLock:
    identities = {
        LockedModel(config.model.target, config.model.target_revision),
        LockedModel(config.model.drafter, config.model.drafter_revision),
    }
    lock = ModelLock(
        schema_version=2,
        models=tuple(sorted(identities, key=lambda row: row.model_id)),
    )
    lock.write(path)
    return lock


def _port(source: FormalSingleOperatorExecutionSource, cell: MaterializedCell) -> int:
    return (
        20_000
        + int(
            _sha256(
                {
                    "protocol_sha256": (
                        FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_PRODUCER_PROTOCOL_SHA256
                    ),
                    "execution_source_sha256": source.sha256,
                    "cell_id": cell.cell_id,
                }
            )[:8],
            16,
        )
        % 40_000
    )


def _materialize_cell_launch(
    *,
    root: Path,
    source: FormalSingleOperatorExecutionSource,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    content_source: FormalContentSourceBinding,
    content: object,
    inventory: GpuInventory,
    doctor: dict[str, object],
    prerequisite_binding: CanonicalJsonProofBinding,
    prerequisite: CompileLaunchManifest,
    prerequisite_config: RunConfig,
    chronobelief_proof_binding: CanonicalJsonProofBinding | None,
    chronobelief_proof: object | None,
) -> PreparedLaunchDraftEntry:
    cell_root = root / "cells" / cell.cell_id
    if os.path.lexists(cell_root):
        raise FileExistsError("prepared launch producer refuses to replace a cell")
    cell_root.mkdir(parents=True, mode=0o700)
    context = _trusted_chain_recipe_context(source)
    requires_chronobelief = (
        cell.method_role in {"LightCone", "LightCone-candidate"}
        and context.lightcone_recipe.optimizer == "chronobelief"
    )
    if requires_chronobelief:
        from lightcone_spec.experiments.formal_single_operator_chronobelief import (
            TrustedSingleOperatorChronoBeliefGpuParityProof,
        )

        if (
            source.stage != "E1a"
            or type(chronobelief_proof)
            is not TrustedSingleOperatorChronoBeliefGpuParityProof
            or type(chronobelief_proof_binding) is not CanonicalJsonProofBinding
            or chronobelief_proof.execution_source_sha256 != source.sha256
            or chronobelief_proof.prerequisite_launch != prerequisite_binding
        ):
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_chronobelief_gpu_proof_missing"
            )
        gpu_uuids = _chronobelief_qualified_gpu_assignment(
            inventory=inventory,
            cell=cell,
            qualified_gpu_uuids=chronobelief_proof.qualified_gpu_uuids,
        )
    elif source.stage == "E0" and prerequisite_config.model.algorithm == "EAGLE3":
        # The task-keyed EAGLE3 execution authority contains one exact native
        # GPU receipt.  Moving the cell to another otherwise-ready GPU would
        # sever that proof join, so the source-owned prerequisite placement
        # takes precedence over the ordinary balanced TP1 placement.
        ready = {device.uuid for device in inventory.devices if device.ready}
        gpu_uuids = prerequisite.gpu_uuids
        if len(gpu_uuids) != 1 or set(gpu_uuids) - ready:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_eagle3_proved_gpu_assignment_missing"
            )
    elif cell.task == "mechanism_profile_only":
        ready = {device.uuid for device in inventory.devices if device.ready}
        gpu_uuids = prerequisite.gpu_uuids
        if (
            len(gpu_uuids) != (1 if _cell_topology(cell) == "tp1_dp1" else 2)
            or set(gpu_uuids) - ready
        ):
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_profiler_selected_gpu_assignment_missing"
            )
    else:
        gpu_uuids = deterministic_prepared_gpu_assignment(
            inventory=inventory, cell=cell
        )
    config = derive_prepared_run_config(
        source=source,
        cell=cell,
        prerequisite=prerequisite_config,
        gpu_uuids=gpu_uuids,
        trusted_chronobelief_gpu_parity_proof_sha256=(
            chronobelief_proof.sha256 if requires_chronobelief else None
        ),
    )
    built_in_mtp = prerequisite.schema_version == 3
    if built_in_mtp != (config.model.nextn_mtp_mode == "built_in_mtp"):
        raise ValueError("prepared NEXTN MTP mode differs from prerequisite")
    model_members = tuple(content.model_members)

    def member(role: str, model_id: str, revision: str) -> object:
        matches = tuple(
            candidate
            for candidate in model_members
            if candidate.role == role
            and candidate.model_id == model_id
            and candidate.revision == revision
            and source.stage in candidate.stages
        )
        if len(matches) != 1:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_model_member_missing"
            )
        return matches[0]

    target = member("target", config.model.target, config.model.target_revision)
    drafter = (
        target
        if built_in_mtp
        else member(
            "drafter",
            config.model.drafter,
            config.model.drafter_revision,
        )
    )
    tokenizer = member(
        "tokenizer", prerequisite.tokenizer_model_id, prerequisite.tokenizer_revision
    )
    if cell.task == "mechanism_profile_only":
        # Path identity, not merely semantic identity, is part of the selected
        # profiler subject contract.
        sampling_path = Path(prerequisite.sampling_profile_path)
        sampling = SamplingProfile.load(sampling_path)
    else:
        sampling = _sampling_for_cell(cell)
        sampling_path = cell_root / "sampling-profile.json"
        sampling.write(sampling_path)
    if config.runtime.sampling_profile_sha256 != sampling.sha256:
        config = RunConfig.model_validate(
            config.model_copy(
                update={
                    "runtime": config.runtime.model_copy(
                        update={"sampling_profile_sha256": sampling.sha256}
                    )
                }
            ).model_dump(mode="json")
        )
    lock = _write_model_lock(config, cell_root / "model-lock.json")
    key = derive_diagnostic_compile_cache_key(
        doctor_report=doctor,
        model_lock=lock,
        config=config,
        gpu_uuid=gpu_uuids[0],
    )
    prerequisite_plan = CompileCacheLaunchPlan.load(
        prerequisite.compile_cache_plan_path
    )
    cache_plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=prerequisite_plan.cache_root,
        cache_mode="build",
    )
    cache_path = cell_root / "compile-cache-plan.json"
    cache_plan.write(cache_path)
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        payloads=(
            CompileOnlyPrewarmPayload(
                request_id=f"prewarm-{cell.cell_id[:24]}",
                graph_bucket=1,
                input_token_ids=(1,),
                requested_output_tokens=1,
                sampling_seed=int(cell.cell_id[:16], 16),
            ),
        ),
    )
    prewarm_path = cell_root / "compile-prewarm.json"
    write_compile_prewarm_manifest(prewarm, prewarm_path)
    roots = {
        config.model.target: target.local_snapshot_path,
        config.model.drafter: drafter.local_snapshot_path,
    }
    server = _render_server(
        output=cell_root,
        method=config.method,
        config=config,
        verified_checkout=Path(prerequisite.patched_sglang_checkout),
        roots=roots,
        target_id=config.model.target,
        drafter_id=config.model.drafter,
        adaptation_reserve_mb=(
            0 if config.method in {"target_only", "static"} else _ADAPTATION_RESERVE_MB
        ),
        mem_fraction_static=float(
            _flag_value(prerequisite.server_argv, "--mem-fraction-static")
        ),
        host="127.0.0.1",
        port=_port(source, cell),
        compile_cache_plan_path=cache_path,
    )
    config_path = Path(server.run_config).resolve()
    config_binding = CanonicalJsonProofBinding.bind(config_path)
    cache_binding = CanonicalJsonProofBinding.bind(cache_path)
    prewarm_binding = CanonicalJsonProofBinding.bind(prewarm_path)
    sampling_binding = CanonicalJsonProofBinding.bind(sampling_path)
    trusted_binding = content_source.trusted_single_operator
    assert trusted_binding is not None
    launch = CompileLaunchManifest(
        schema_version=3 if built_in_mtp else 2,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
            if built_in_mtp
            else TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
        ),
        patched_sglang_checkout=prerequisite.patched_sglang_checkout,
        patched_sglang_commit=prerequisite.patched_sglang_commit,
        patched_sglang_tree=prerequisite.patched_sglang_tree,
        run_config_path=str(config_path),
        run_config_raw_sha256=config_binding.raw_sha256,
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=cache_binding.raw_sha256,
        compile_cache_plan_sha256=cache_plan.sha256,
        prewarm_manifest_path=str(prewarm_path),
        prewarm_manifest_raw_sha256=prewarm_binding.raw_sha256,
        prewarm_manifest_sha256=prewarm.sha256,
        sampling_profile_path=str(sampling_path),
        sampling_profile_raw_sha256=sampling_binding.raw_sha256,
        prepared_model_content_manifest_path=trusted_binding.absolute_path,
        prepared_model_content_manifest_raw_sha256=trusted_binding.raw_sha256,
        prepared_model_content_manifest_sha256=trusted_binding.semantic_sha256,
        prepared_model_content_manifest_size=trusted_binding.size,
        target_content_member_id=target.sha256,
        target_model_id=config.model.target,
        target_snapshot_path=target.local_snapshot_path,
        target_revision=config.model.target_revision,
        target_content_authority_sha256=None,
        drafter_content_member_id=(
            drafter.sha256
            if built_in_mtp
            else (None if config.method == "target_only" else drafter.sha256)
        ),
        drafter_model_id=(
            config.model.drafter
            if built_in_mtp
            else (None if config.method == "target_only" else config.model.drafter)
        ),
        drafter_snapshot_path=(
            drafter.local_snapshot_path
            if built_in_mtp
            else (
                None if config.method == "target_only" else drafter.local_snapshot_path
            )
        ),
        drafter_revision=(
            config.model.drafter_revision
            if built_in_mtp
            else (
                None
                if config.method == "target_only"
                else config.model.drafter_revision
            )
        ),
        drafter_content_authority_sha256=None,
        tokenizer_content_member_id=tokenizer.sha256,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_snapshot_path=tokenizer.local_snapshot_path,
        tokenizer_revision=tokenizer.revision,
        tokenizer_content_authority_sha256=None,
        server_argv=server.argv,
        server_argv_sha256=_sha256({"argv": list(server.argv)}),
        localhost_port=_port(source, cell),
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        physical_assignment_sha256=_sha256(
            {
                "kind": "formal_single_operator_prepared_assignment",
                "execution_source_sha256": source.sha256,
                "cell_id": cell.cell_id,
                "inventory_sha256": inventory.sha256,
                "gpu_uuids": gpu_uuids,
            }
        ),
        experiment_budget_sha256=_sha256(
            {
                "kind": "formal_single_operator_prepared_budget_subject",
                "materialization_sha256": materialization.sha256,
                "cell_id": cell.cell_id,
            }
        ),
        budget_materialization_authority_sha256=source.sha256,
        inventory_sha256=inventory.sha256,
        gpu_uuids=gpu_uuids,
        path_entries=prerequisite.path_entries,
        library_path_entries=prerequisite.library_path_entries,
        cuda_home=prerequisite.cuda_home,
        formal_stage=source.stage,
        content_source_binding=content_source,
        nextn_mtp_mode=("built_in_mtp" if built_in_mtp else None),
        target_snapshot_sha256=(
            prerequisite.target_snapshot_sha256 if built_in_mtp else None
        ),
        mtp_component_sha256=(
            prerequisite.mtp_component_sha256 if built_in_mtp else None
        ),
        mtp_component_binding=(
            prerequisite.mtp_component_binding if built_in_mtp else None
        ),
    )
    launch.validate(reopen_inputs=True)
    launch_path = cell_root / "compile-launch.json"
    launch.write(launch_path)
    if requires_chronobelief:
        from lightcone_spec.experiments.formal_single_operator_chronobelief import (
            revalidate_trusted_single_operator_chronobelief_for_prepared_launch,
        )

        assert chronobelief_proof_binding is not None
        revalidate_trusted_single_operator_chronobelief_for_prepared_launch(
            proof_path=chronobelief_proof_binding.absolute_path,
            execution_source_path=chronobelief_proof.execution_source.absolute_path,
            prepared_launch_path=launch_path,
        )
    route = route_formal_single_operator_materialized_cell(
        node=source.node, phase=source.phase, cell=cell
    )
    return PreparedLaunchDraftEntry(
        materialized_cell_id=cell.cell_id,
        physical_kind=route.physical_kind,  # type: ignore[arg-type]
        prerequisite_launch=prerequisite_binding,
        run_config=config_binding,
        compile_cache_plan=cache_binding,
        prewarm_manifest=prewarm_binding,
        sampling_profile=sampling_binding,
        compile_launch_manifest=CanonicalJsonProofBinding.bind(
            launch_path, semantic_sha256=launch.sha256
        ),
        launch_compatibility_key_sha256=(
            formal_single_operator_launch_compatibility_key(
                launch=launch, config=config
            )
        ),
        topology_mode=config.runtime.topology_mode,
        gpu_uuids=gpu_uuids,
        schedule_state="PENDING_DURABLE_TOKENIZATION",
        trusted_chronobelief_gpu_parity_proof=(
            chronobelief_proof_binding if requires_chronobelief else None
        ),
    )


def prepare_launch_draft(
    *,
    execution_source_path: str | Path,
    content_source_path: str | Path,
    prerequisite_launch_manifest_paths: tuple[str | Path, ...],
    chronobelief_gpu_parity_proof_paths: tuple[str | Path, ...] = (),
    private_output_root: str | Path,
) -> FormalSingleOperatorPreparedLaunchDraft:
    """Publish complete per-cell launch inputs before durable tokenization."""

    root = _absolute_existing_directory(
        "prepared launch private output root", private_output_root
    )
    draft_path = root / "prepared-launch-draft.json"
    cells_path = root / "cells"
    draft_entries_path = root / "prepared-launch-draft-entries"
    if (
        os.path.lexists(draft_path)
        or os.path.lexists(cells_path)
        or os.path.lexists(draft_entries_path)
    ):
        raise FileExistsError("prepared launch draft output already exists")
    execution_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source = load_formal_single_operator_execution_source(
        execution_binding.absolute_path
    )
    if execution_binding.reopen().get("execution_source_sha256") != source.sha256:
        raise ValueError("prepared producer execution source identity differs")
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="prepared producer execution materialization"
        )
    )
    protocol_lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(
            label="prepared producer execution ProtocolLock"
        )
    )
    content_source = FormalContentSourceBinding.bind_trusted_single_operator(
        str(content_source_path)
    )
    if (
        source.content_source_binding != content_source
        or protocol_lock.content_source_mode != "trusted_single_operator"
        or protocol_lock.trusted_single_operator_content_bundle_sha256
        != content_source.content_sha256
    ):
        raise ValueError("prepared producer content differs from ProtocolLock")
    content, inventory_binding, inventory, doctor_binding, doctor = _runtime_inputs(
        content_source
    )
    prerequisites = _prerequisites(
        prerequisite_launch_manifest_paths,
        content_source=content_source,
        inventory=inventory,
    )
    chronobelief_proofs = _chronobelief_proofs(chronobelief_gpu_parity_proof_paths)
    requires_chronobelief_proofs = (
        source.stage == "E1a"
        and _trusted_chain_recipe_context(source).lightcone_recipe.optimizer
        == "chronobelief"
    )
    if bool(chronobelief_proofs) != requires_chronobelief_proofs or (
        chronobelief_proofs
        and set(chronobelief_proofs)
        != {binding.semantic_sha256 for binding, _launch in prerequisites.values()}
    ):
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_chronobelief_gpu_proof_coverage_differs"
        )
    entries = []
    for cell in materialization.cells:
        route = route_formal_single_operator_materialized_cell(
            node=source.node, phase=source.phase, cell=cell
        )
        if route.physical_kind in {
            "e6_interface_preflight",
            "e0_compatibility_decision",
        }:
            continue
        prerequisite_binding, prerequisite, config = _matching_prerequisite(
            source=source, cell=cell, rows=prerequisites
        )
        chronobelief_row = chronobelief_proofs.get(prerequisite_binding.semantic_sha256)
        entries.append(
            _materialize_cell_launch(
                root=root,
                source=source,
                materialization=materialization,
                cell=cell,
                content_source=content_source,
                content=content,
                inventory=inventory,
                doctor=doctor,
                prerequisite_binding=prerequisite_binding,
                prerequisite=prerequisite,
                prerequisite_config=config,
                chronobelief_proof_binding=(
                    None if chronobelief_row is None else chronobelief_row[0]
                ),
                chronobelief_proof=(
                    None if chronobelief_row is None else chronobelief_row[1]
                ),
            )
        )
    canonical_entries = tuple(sorted(entries, key=lambda row: row.materialized_cell_id))
    if not canonical_entries:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_prepared_launch_entries_empty"
        )
    monolithic = FormalSingleOperatorPreparedLaunchDraft(
        schema_version=1,
        kind="formal_single_operator_prepared_launch_draft",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_PRODUCER_PROTOCOL_SHA256
        ),
        execution_source=execution_binding,
        execution_source_sha256=source.sha256,
        materialization=CanonicalJsonProofBinding.bind(
            source.materialization_source.absolute_path
        ),
        materialization_sha256=materialization.sha256,
        content_source_binding=content_source,
        inventory=inventory_binding,
        inventory_sha256=inventory.sha256,
        doctor=doctor_binding,
        entries=canonical_entries,
    )
    materialized_cell_ids = tuple(row.materialized_cell_id for row in canonical_entries)
    draft_entries_path.mkdir(mode=0o700)
    index_binding, _index = publish_formal_canonical_sequence_shards(
        artifact_kind=_PREPARED_LAUNCH_DRAFT_ENTRY_SHARD_ARTIFACT_KIND,
        artifact_id=prepared_launch_draft_entries_artifact_id(
            draft=monolithic,
            materialized_cell_ids=materialized_cell_ids,
        ),
        rows=tuple(row.to_dict() for row in canonical_entries),
        output_directory=draft_entries_path,
        maximum_shard_rows=64,
    )
    draft = FormalSingleOperatorPreparedLaunchDraft(
        schema_version=2,
        kind=monolithic.kind,
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SHARDED_PREPARED_LAUNCH_DRAFT_PROTOCOL_SHA256
        ),
        execution_source=monolithic.execution_source,
        execution_source_sha256=monolithic.execution_source_sha256,
        materialization=monolithic.materialization,
        materialization_sha256=monolithic.materialization_sha256,
        content_source_binding=monolithic.content_source_binding,
        inventory=monolithic.inventory,
        inventory_sha256=monolithic.inventory_sha256,
        doctor=monolithic.doctor,
        entries=(),
        entries_shard_index=index_binding,
    )
    _load_prepared_draft_entry_shard_index(
        draft=draft,
        materialized_cell_ids=materialized_cell_ids,
        deep=True,
    )
    publish_canonical_json_no_replace(draft_path, draft.to_dict())
    rebound = FormalSingleOperatorPreparedLaunchDraft.from_dict(
        CanonicalJsonProofBinding.bind(draft_path).reopen()
    )
    if rebound != draft:
        raise RuntimeError("prepared launch draft failed round-trip replay")
    return draft


def load_prepared_launch_draft(
    path: str | Path,
) -> FormalSingleOperatorPreparedLaunchDraft:
    binding = CanonicalJsonProofBinding.bind(path)
    draft = FormalSingleOperatorPreparedLaunchDraft.from_dict(binding.reopen())
    if draft.sha256 != binding.reopen().get("draft_sha256"):
        raise ValueError("prepared launch draft canonical identity differs")
    if draft.schema_version == 2:
        _load_prepared_draft_entry_shard_index(
            draft=draft,
            materialized_cell_ids=_expected_draft_cell_ids(draft),
            deep=False,
        )
    return draft


def _provisional_bundle_and_entry(
    *,
    source: FormalSingleOperatorExecutionSource,
    materialization: StageMaterializationReceipt,
    draft: FormalSingleOperatorPreparedLaunchDraft,
    row: PreparedLaunchDraftEntry,
    schedule: CanonicalJsonProofBinding,
    profiler: FormalSingleOperatorProfilerSubjectRequirement | None,
) -> tuple[
    FormalSingleOperatorPreparedLaunchBundle, FormalSingleOperatorPreparedLaunchEntry
]:
    launch = CompileLaunchManifest.load(row.compile_launch_manifest.absolute_path)
    built_in_mtp = launch.schema_version == 3
    entry = FormalSingleOperatorPreparedLaunchEntry(
        schema_version=3 if built_in_mtp else 2,
        kind="formal_single_operator_prepared_launch_entry",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256
            if built_in_mtp
            else TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256
        ),
        materialized_cell_id=row.materialized_cell_id,
        physical_kind=row.physical_kind,
        run_config=row.run_config,
        compile_launch_manifest=row.compile_launch_manifest,
        request_schedule_receipt=(
            None if row.physical_kind == "profiler" else schedule
        ),
        launch_compatibility_key_sha256=row.launch_compatibility_key_sha256,
        target_content_member_id=launch.target_content_member_id,
        drafter_content_member_id=launch.drafter_content_member_id,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        inventory_sha256=draft.inventory_sha256,
        topology_mode=row.topology_mode,
        gpu_uuids=row.gpu_uuids,
        server_argv_sha256=launch.server_argv_sha256,
        profiler_subject=profiler,
        trusted_chronobelief_gpu_parity_proof=(
            row.trusted_chronobelief_gpu_parity_proof
        ),
        nextn_mtp_mode=("built_in_mtp" if built_in_mtp else None),
        target_snapshot_sha256=launch.target_snapshot_sha256,
        mtp_component_sha256=launch.mtp_component_sha256,
        mtp_component=launch.mtp_component_binding,
    )
    kwargs: dict[str, object] = {
        "schema_version": 2,
        "kind": "formal_single_operator_prepared_launch_bundle",
        "protocol_sha256": (
            TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
        ),
        "node": source.node,
        "stage": source.stage,
        "phase": source.phase,
        "execution_source": draft.execution_source,
        "execution_source_sha256": source.sha256,
        "protocol_lock_sha256": source.protocol_lock_sha256,
        "materialization_sha256": materialization.sha256,
        "materialization_source_decision_sha256": (
            materialization.source_decision_sha256
        ),
        "inventory": draft.inventory,
        "entries": (entry,),
    }
    kwargs["content_source_binding"] = draft.content_source_binding
    kwargs["content_verification_receipt"] = None
    return FormalSingleOperatorPreparedLaunchBundle(**kwargs), entry  # type: ignore[arg-type]


def expected_schedule_identities(
    *,
    draft: FormalSingleOperatorPreparedLaunchDraft,
    materialized_cell_id: str,
) -> tuple[str, str]:
    """Expose the two non-circular identities for tokenizer phase two."""

    source = load_formal_single_operator_execution_source(
        draft.execution_source.absolute_path
    )
    materialization = stage_materialization_receipt_from_dict(
        draft.materialization.reopen()
    )
    matches = _prepared_draft_entries(
        draft,
        materialized_cell_id=materialized_cell_id,
    )
    if len(matches) != 1 or matches[0].physical_kind == "profiler":
        raise ValueError("prepared schedule identity cell differs")
    # A stable placeholder is sufficient because schedule content is excluded
    # from both identities.  It is never published or accepted as evidence.
    placeholder_path = Path(matches[0].run_config.absolute_path)
    bundle, entry = _provisional_bundle_and_entry(
        source=source,
        materialization=materialization,
        draft=draft,
        row=matches[0],
        schedule=CanonicalJsonProofBinding.bind(placeholder_path),
        profiler=None,
    )
    return formal_single_operator_prepared_execution_identities(
        bundle=bundle, entry=entry
    )


def _materialize_context_filler_for_cell(
    *,
    draft_path: str | Path,
    draft: FormalSingleOperatorPreparedLaunchDraft,
    row: PreparedLaunchDraftEntry,
    cell: MaterializedCell,
) -> CanonicalJsonProofBinding:
    """Publish or deep-reopen one draft-owned tokenizer filler authority."""

    from lightcone_spec.experiments.formal_single_operator_context_artifact import (
        load_trusted_context_filler_artifact,
        materialize_trusted_context_filler_artifact,
    )

    if cell.stage not in {"E3b", "E6"}:
        raise ValueError("context filler is restricted to controlled-context cells")
    launch = CompileLaunchManifest.load(row.compile_launch_manifest.absolute_path)
    member_id = _require_sha256(
        "prepared context filler tokenizer member",
        launch.tokenizer_content_member_id,
    )
    draft_file = Path(draft_path)
    if (
        not draft_file.is_absolute()
        or draft_file != draft_file.resolve(strict=False)
        or not draft_file.is_file()
        or draft_file.is_symlink()
    ):
        raise ValueError("prepared context filler draft path differs")
    parent = draft_file.parent
    filler_root = parent / "context-filler-artifacts"
    if not os.path.lexists(filler_root):
        filler_root.mkdir(mode=0o700)
    elif not filler_root.is_dir() or filler_root.is_symlink():
        raise ValueError("prepared context filler root differs")
    member_root = filler_root / member_id
    artifact_path = member_root / "context-filler-authority.json"
    if os.path.lexists(artifact_path):
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise ValueError("prepared context filler artifact path differs")
        binding = CanonicalJsonProofBinding.bind(artifact_path)
    else:
        if os.path.lexists(member_root):
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_context_filler_artifact_incomplete"
            )
        member_root.mkdir(mode=0o700)
        binding = materialize_trusted_context_filler_artifact(
            content_source_binding=draft.content_source_binding,
            compile_launch_manifest_path=(row.compile_launch_manifest.absolute_path),
            output_directory=member_root,
        )
        if binding.absolute_path != str(artifact_path):
            raise RuntimeError("prepared context filler publication path differs")
    load_trusted_context_filler_artifact(
        binding.absolute_path,
        content_source_binding=draft.content_source_binding,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
    )
    return binding


def materialize_prepared_context_filler_artifact(
    *,
    draft_path: str | Path,
    materialized_cell_id: str,
) -> CanonicalJsonProofBinding:
    """Materialize the shared exact-token filler for one E3b/E6 draft cell."""

    draft = load_prepared_launch_draft(draft_path)
    matches = _prepared_draft_entries(
        draft,
        materialized_cell_id=materialized_cell_id,
    )
    if len(matches) != 1 or matches[0].physical_kind == "profiler":
        raise ValueError("prepared context filler cell is absent or profiler-only")
    materialization = stage_materialization_receipt_from_dict(
        draft.materialization.reopen()
    )
    cells = tuple(
        cell for cell in materialization.cells if cell.cell_id == materialized_cell_id
    )
    if len(cells) != 1 or cells[0].stage not in {"E3b", "E6"}:
        raise ValueError("prepared context filler cell is not controlled-context")
    return _materialize_context_filler_for_cell(
        draft_path=draft_path,
        draft=draft,
        row=matches[0],
        cell=cells[0],
    )


def materialize_prepared_request_schedule(
    *,
    draft_path: str | Path,
    materialized_cell_id: str,
    private_output_root: str | Path,
    e5_arrival_plan_path: str | Path | None = None,
) -> FormalServingRequestScheduleReceipt:
    """Tokenize one draft cell through the tagged trusted schedule reducer.

    The caller selects only an already-materialized cell and, for an E5
    headline, supplies the separately published path-bound arrival-plan
    actual.  Workload paths and both non-circular prepared identities are
    derived here from the sealed draft and its BOUND content bundle.
    """

    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
    )
    from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
        E0TaskNativeSourceAuthority,
        load_e0_task_native_source_authority,
    )
    from lightcone_spec.experiments.workload_authority import (
        bind_formal_workload_authority,
        formal_workload_authority_cli_artifact,
    )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        _workload_id_for_cell,
        materialize_trusted_single_operator_request_schedule,
    )

    draft = load_prepared_launch_draft(draft_path)
    matches = _prepared_draft_entries(
        draft,
        materialized_cell_id=materialized_cell_id,
    )
    if len(matches) != 1 or matches[0].physical_kind == "profiler":
        raise ValueError("prepared schedule cell is absent or profiler-only")
    row = matches[0]
    materialization = stage_materialization_receipt_from_dict(
        draft.materialization.reopen()
    )
    cells = tuple(
        cell for cell in materialization.cells if cell.cell_id == materialized_cell_id
    )
    if len(cells) != 1:
        raise ValueError("prepared schedule cell differs from materialization")
    cell = cells[0]
    if cell.stage == "E5" and cell.task == "production_slo_power_prefix":
        if e5_arrival_plan_path is None:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_e5_arrival_plan_missing"
            )
    elif e5_arrival_plan_path is not None:
        raise ValueError("non-headline prepared schedule carries an E5 arrival plan")

    output = _absolute_existing_directory(
        "prepared schedule private output root", private_output_root
    )
    content = draft.content_source_binding.reopen()
    if type(content) is not TrustedSingleOperatorContentBundle:
        raise TypeError("prepared schedule trusted content bundle differs")
    workload_id = _workload_id_for_cell(cell)
    if workload_id in {"livecodebench_v6_hard", "math500_level5"}:
        members = tuple(
            member
            for member in content.locked_workloads
            if member.workload_id == workload_id
        )
        if len(members) != 1:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_locked_workload_missing"
            )
        member = members[0]
        authority = bind_formal_workload_authority(
            member.workload_id, member.raw_source_path
        )
        if authority.sha256 != member.authority_sha256:
            raise ValueError("prepared schedule locked workload identity differs")
        workload_path = output / "trusted-workload-source.json"
        publish_canonical_json_no_replace(
            workload_path, formal_workload_authority_cli_artifact(authority)
        )
    else:
        members = tuple(
            member
            for member in content.e0_task_native_descriptors
            if member.task == workload_id
        )
        if len(members) != 1:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_e0_workload_missing"
            )
        workload_path = Path(members[0].source.absolute_path)
        workload = load_e0_task_native_source_authority(workload_path)
        if (
            type(workload) is not E0TaskNativeSourceAuthority
            or workload.task != workload_id
            or workload.support_status != "READY"
        ):
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_e0_workload_not_serving_ready"
            )

    execution_binding_sha256, subject_sha256 = expected_schedule_identities(
        draft=draft,
        materialized_cell_id=materialized_cell_id,
    )
    context_filler_artifact_path = None
    if cell.stage in {"E3b", "E6"}:
        context_filler_artifact_path = _materialize_context_filler_for_cell(
            draft_path=draft_path,
            draft=draft,
            row=row,
            cell=cell,
        ).absolute_path
    receipt = materialize_trusted_single_operator_request_schedule(
        execution_source_path=draft.execution_source.absolute_path,
        materialized_cell_id=materialized_cell_id,
        compile_launch_manifest_path=row.compile_launch_manifest.absolute_path,
        workload_source_path=workload_path,
        execution_binding_sha256=execution_binding_sha256,
        subject_sha256=subject_sha256,
        private_output_root=output,
        e5_arrival_plan_path=e5_arrival_plan_path,
        context_filler_artifact_path=context_filler_artifact_path,
    )
    if (
        receipt.materialized_cell_id != materialized_cell_id
        or receipt.execution_binding_sha256 != execution_binding_sha256
        or receipt.subject_sha256 != subject_sha256
        or receipt.content_source_binding != draft.content_source_binding
    ):
        raise RuntimeError("prepared request schedule differs after tokenization")
    return receipt


def publish_sharded_prepared_launch_bundle(
    *,
    bundle: FormalSingleOperatorPreparedLaunchBundle,
    output_path: str | Path,
) -> FormalSingleOperatorPreparedLaunchBundle:
    """Publish bounded entry shards and the small canonical schema-3 bundle."""

    if (
        type(bundle) is not FormalSingleOperatorPreparedLaunchBundle
        or bundle.schema_version != 2
        or not bundle.entries
    ):
        raise ValueError("prepared shard publisher requires populated schema 2")
    destination = Path(output_path)
    if (
        not destination.is_absolute()
        or destination != destination.resolve(strict=False)
        or not destination.parent.is_dir()
        or destination.parent.is_symlink()
    ):
        raise ValueError("prepared bundle output path must be absolute/normalized")
    shard_root = destination.parent / f"{destination.name}.entries"
    if os.path.lexists(destination) or os.path.lexists(shard_root):
        raise FileExistsError("prepared sharded bundle output already exists")
    materialized_cell_ids = tuple(row.materialized_cell_id for row in bundle.entries)
    if materialized_cell_ids != tuple(sorted(set(materialized_cell_ids))):
        raise ValueError("prepared shard publisher entries are not canonical")
    shard_root.mkdir(mode=0o700)
    index_binding, _index = publish_formal_canonical_sequence_shards(
        artifact_kind=(FORMAL_SINGLE_OPERATOR_PREPARED_ENTRY_SHARD_ARTIFACT_KIND),
        artifact_id=formal_single_operator_prepared_entries_artifact_id(
            bundle=bundle,
            materialized_cell_ids=materialized_cell_ids,
        ),
        rows=tuple(row.to_dict() for row in bundle.entries),
        output_directory=shard_root,
        maximum_shard_rows=128,
    )
    sharded = shard_formal_single_operator_prepared_launch_bundle(
        bundle=bundle,
        entries_shard_index=index_binding,
    )
    publish_canonical_json_no_replace(destination, sharded.to_dict())
    rebound = FormalSingleOperatorPreparedLaunchBundle.from_dict(
        CanonicalJsonProofBinding.bind(destination).reopen()
    )
    if rebound != sharded:
        raise RuntimeError("prepared sharded bundle failed canonical replay")
    return sharded


def finalize_prepared_launch_bundle(
    *,
    draft_path: str | Path,
    request_schedule_receipt_paths: tuple[str | Path, ...],
    output_path: str | Path,
    profiler_subject_requirement_path: str | Path | None = None,
    current_ns: int,
) -> FormalSingleOperatorPreparedLaunchBundle:
    """Seal phase-two receipts, publish once, and deep-revalidate everything."""

    draft = load_prepared_launch_draft(draft_path)
    source = load_formal_single_operator_execution_source(
        draft.execution_source.absolute_path
    )
    materialization = stage_materialization_receipt_from_dict(
        draft.materialization.reopen()
    )
    schedules: dict[str, CanonicalJsonProofBinding] = {}
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
    )

    for path in request_schedule_receipt_paths:
        binding = CanonicalJsonProofBinding.bind(path)
        receipt = FormalServingRequestScheduleReceipt.from_dict(binding.reopen())
        if receipt.sha256 != binding.semantic_sha256:
            raise ValueError("prepared schedule receipt identity differs")
        if receipt.materialized_cell_id in schedules:
            raise ValueError("prepared schedule receipt cell is duplicated")
        schedules[receipt.materialized_cell_id] = binding
    profiler = None
    if profiler_subject_requirement_path is not None:
        profiler_binding = CanonicalJsonProofBinding.bind(
            profiler_subject_requirement_path
        )
        profiler = FormalSingleOperatorProfilerSubjectRequirement.from_dict(
            profiler_binding.reopen()
        )
    draft_entries = _prepared_draft_entries(draft)
    entries = []
    for row in draft_entries:
        schedule = schedules.get(row.materialized_cell_id)
        if row.physical_kind == "profiler":
            if profiler is None:
                raise FormalSingleOperatorPreparedLaunchBlocked(
                    "source_owned_profiler_subject_missing"
                )
            # Placeholder is not retained for profiler entries.
            schedule = row.run_config
        elif schedule is None:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_request_schedule_missing"
            )
        _bundle, entry = _provisional_bundle_and_entry(
            source=source,
            materialization=materialization,
            draft=draft,
            row=row,
            schedule=schedule,
            profiler=profiler if row.physical_kind == "profiler" else None,
        )
        entries.append(entry)
    expected_schedule_ids = {
        row.materialized_cell_id
        for row in draft_entries
        if row.physical_kind != "profiler"
    }
    if set(schedules) != expected_schedule_ids:
        raise ValueError("prepared schedule receipt coverage differs")
    monolithic = FormalSingleOperatorPreparedLaunchBundle(
        schema_version=2,
        kind="formal_single_operator_prepared_launch_bundle",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
        ),
        node=source.node,
        stage=source.stage,  # type: ignore[arg-type]
        phase=source.phase,
        execution_source=draft.execution_source,
        execution_source_sha256=source.sha256,
        protocol_lock_sha256=source.protocol_lock_sha256,
        materialization_sha256=materialization.sha256,
        materialization_source_decision_sha256=(materialization.source_decision_sha256),
        inventory=draft.inventory,
        content_verification_receipt=None,
        content_source_binding=draft.content_source_binding,
        entries=tuple(sorted(entries, key=lambda row: row.materialized_cell_id)),
    )
    destination = Path(output_path)
    bundle = publish_sharded_prepared_launch_bundle(
        bundle=monolithic,
        output_path=destination,
    )
    rebound = revalidate_formal_single_operator_prepared_launch_bundle(
        execution_source_path=draft.execution_source.absolute_path,
        prepared_launch_bundle_path=destination,
        current_ns=current_ns,
    )
    if rebound.bundle != bundle:
        raise RuntimeError("prepared launch bundle failed deep replay")
    return bundle


__all__ = [
    "FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_PRODUCER_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_SHARDED_PREPARED_LAUNCH_DRAFT_PROTOCOL_SHA256",
    "FormalSingleOperatorPreparedLaunchDraft",
    "PreparedLaunchDraftEntry",
    "PreparedLaunchRuntimeKey",
    "derive_prepared_run_config",
    "deterministic_prepared_gpu_assignment",
    "expected_schedule_identities",
    "finalize_prepared_launch_bundle",
    "load_prepared_launch_draft",
    "materialize_prepared_context_filler_artifact",
    "materialize_prepared_request_schedule",
    "prepare_launch_draft",
    "prepared_launch_draft_entries_artifact_id",
    "prerequisite_runtime_key",
    "publish_sharded_prepared_launch_bundle",
]
