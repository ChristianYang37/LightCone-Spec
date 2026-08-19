"""Strict prepared-launch envelope for downstream single-operator cells.

The downstream stage materializers intentionally do not accept a caller-authored
``RunConfig`` or launch command.  This module defines the durable bundle which a
source-owned mapper must produce before ``prepare-run`` can consider a downstream
cell.  Revalidation joins every launch to the exact current materialization,
prepared-model content receipt, GPU inventory, topology, and compatibility key.

This bundle is deliberately not a legacy
``VerifiedFormalStageMaterializationSource`` bearer token.  Trusted current
execution instead reopens the public single-operator predecessor chain, joins
the exact current materialization, and consumes only the per-cell launch and
schedule paths sealed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_from_dict,
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorExecutionSource,
    FormalSingleOperatorNode,
    load_formal_single_operator_execution_source,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.stage_materialization import (
    E1A_FIXED_VERIFICATION_BUDGET,
    E1A_NATIVE_VERIFICATION_BUDGET,
)
from lightcone_spec.locking.prepared_models import PreparedModelSnapshotContent
from lightcone_spec.runtime.compile_cache import CompileCacheLaunchPlan
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.content_authorization import (
    AuthorizedPreparedModel,
    ContentVerificationReceipt,
    VerifiedPreparedModelContentRelease,
)
from lightcone_spec.runtime.formal_sharded_artifact import (
    FormalCanonicalSequenceShardIndex,
    load_formal_canonical_sequence_shard_index,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

PreparedLaunchPhysicalKind = Literal["serving", "profiler", "e5_failure"]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _strict(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _stable_binding(
    binding: CanonicalJsonProofBinding,
    *,
    label: str,
) -> CanonicalJsonProofBinding:
    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} is not path-bound")
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError(f"{label} changed")
    return binding


FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_prepared_launch_bundle_protocol",
        "scope": "one_exact_current_materialization",
        "entries": (
            "every_serving_profiler_or_e5_failure_cell_exactly_once_sorted_by_cell_id"
        ),
        "run_config": "canonical_path_raw_and_semantic_bound_by_compile_launch",
        "content": (
            "exact_stage_target_drafter_tokenizer_members_reopened_from_receipt"
        ),
        "assignment": "inventory_gpu_uuid_order_and_registered_topology",
        "launch_compatibility_key": (
            "model_members_revisions_backend_topology_context_graph_tp_dp_argv_cache"
        ),
        "trusted_chain_run_config": (
            "deep_rebuilt_tts_lr_stride_e2_final_recipe_e1a_dspark_winner_"
            "e1a_fixed_verification_budget_8_e0_onlinespec_candidate_or_"
            "selected_winner_and_exact_role_policy"
        ),
        "tts_l0_identity": (
            "same_frozen_candidate_numeric_adaptation_and_pair_state_namespace_"
            "publication_changes_only_from_tts_to_l0"
        ),
        "profiler_subject": (
            "e4_local_winner_configuration_at_saturation_mixed_prefill_decode_"
            "unique_headline_cell_schedule_and_launch_reused_by_all_three_variants_"
            "subject_command_derived_only_after_run_plan_publication"
        ),
        "forbidden": (
            "preflight_qwen_dflash_template_substitution",
            "caller_authored_load_or_traffic_for_profiler",
            "prepared_bundle_as_private_stage_source_token",
        ),
    }
)
TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_single_operator_prepared_launch_bundle_protocol",
        "content_source": (
            "exact_one_runtime_BOUND_trusted_bundle_propagated_without_"
            "offline_authorization_claims"
        ),
        "launches": "schema2_compile_launch_exact_stage_role_model_revision",
        "legacy_signed_schema": "schema1_unchanged",
    }
)
TRUSTED_SINGLE_OPERATOR_SHARDED_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256 = (
    _content_sha256(
        {
            "schema_version": 3,
            "kind": "formal_single_operator_prepared_launch_bundle_protocol",
            "content_source": "trusted_single_operator_BOUND_bundle",
            "entries": (
                "deterministic_contiguous_canonical_sequence_shards_"
                "sorted_by_materialized_cell_id"
            ),
            "index": "bundle_binds_only_the_small_path_bound_shard_index",
            "execution": ("one_requested_cell_reopens_only_its_expected_ordinal_shard"),
            "complete_audit": "all_shards_deep_replayed_in_exact_cell_order",
            "legacy": "schema1_and_schema2_serialization_unchanged",
        }
    )
)
FORMAL_SINGLE_OPERATOR_PREPARED_ENTRY_SHARD_ARTIFACT_KIND = (
    "formal_single_operator_prepared_launch_entries"
)
TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_single_operator_prepared_launch_entry",
        "base": FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256,
        "content_mode": "trusted_single_operator",
        "optional_runtime_authority": (
            "path_bound_chronobelief_gpu_parity_only_when_frozen_optimizer_"
            "is_chronobelief"
        ),
        "legacy_schema1_serialization": "unchanged",
    }
)
TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256 = (
    _content_sha256(
        {
            "schema_version": 3,
            "kind": "formal_single_operator_prepared_launch_entry",
            "base": TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256,
            "nextn_mode": "built_in_mtp_same_frozen_target_snapshot",
            "component": "path_bound_deep_revalidated_mtp_component",
            "external_drafter": "forbidden",
            "legacy_schema1_and_schema2_serialization": "unchanged",
        }
    )
)


class FormalSingleOperatorPreparedLaunchBlocked(RuntimeError):
    """The current cell has no complete source-owned prepared launch."""

    def __init__(self, reason_code: str) -> None:
        _require_text("prepared launch BLOCKED reason", reason_code)
        self.reason_code = reason_code
        super().__init__(f"prepared launch is BLOCKED: {reason_code}")


@dataclass(frozen=True)
class FormalSingleOperatorProfilerSubjectRequirement:
    """Immutable inputs still required before an E4 profile can execute.

    No load or traffic scalar appears in this schema.  The complete request
    workload must already be reduced into the code-owned subject artifact and
    request schedule; the operator cannot pick a convenient profiler stratum.
    """

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_profiler_subject_requirement"]
    protocol_sha256: str
    source_headline_cell_id: str
    selected_configuration_sha256: str
    selected_full_run_config: CanonicalJsonProofBinding
    selected_compile_launch_manifest: CanonicalJsonProofBinding
    code_owned_profiler_subject_workload: CanonicalJsonProofBinding
    code_owned_request_schedule: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_profiler_subject_requirement"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
        ):
            raise ValueError("profiler subject requirement schema differs")
        for label, value in (
            ("source headline cell", self.source_headline_cell_id),
            ("selected configuration", self.selected_configuration_sha256),
        ):
            _require_sha256(f"profiler subject {label}", value)
        for label, binding in (
            ("selected full RunConfig", self.selected_full_run_config),
            (
                "selected compile launch",
                self.selected_compile_launch_manifest,
            ),
            (
                "code-owned workload subject",
                self.code_owned_profiler_subject_workload,
            ),
            ("code-owned request schedule", self.code_owned_request_schedule),
        ):
            _stable_binding(binding, label=f"profiler subject {label}")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "source_headline_cell_id": self.source_headline_cell_id,
            "selected_configuration_sha256": (self.selected_configuration_sha256),
            "selected_full_run_config": self.selected_full_run_config.to_dict(),
            "selected_compile_launch_manifest": (
                self.selected_compile_launch_manifest.to_dict()
            ),
            "code_owned_profiler_subject_workload": (
                self.code_owned_profiler_subject_workload.to_dict()
            ),
            "code_owned_request_schedule": (self.code_owned_request_schedule.to_dict()),
        }
        if include_sha256:
            value["requirement_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "profiler subject requirement",
            value,
            {*cls.__dataclass_fields__, "requirement_sha256"},
        )
        declared = _require_sha256(
            "profiler subject requirement",
            row.pop("requirement_sha256"),
        )
        for name in (
            "selected_full_run_config",
            "selected_compile_launch_manifest",
            "code_owned_profiler_subject_workload",
            "code_owned_request_schedule",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        result = cls(**row)  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("profiler subject requirement digest differs")
        return result


@dataclass(frozen=True)
class FormalSingleOperatorPreparedLaunchEntry:
    schema_version: Literal[1, 2, 3]
    kind: Literal["formal_single_operator_prepared_launch_entry"]
    protocol_sha256: str
    materialized_cell_id: str
    physical_kind: PreparedLaunchPhysicalKind
    run_config: CanonicalJsonProofBinding
    compile_launch_manifest: CanonicalJsonProofBinding
    request_schedule_receipt: CanonicalJsonProofBinding | None
    launch_compatibility_key_sha256: str
    target_content_member_id: str
    drafter_content_member_id: str | None
    tokenizer_content_member_id: str
    inventory_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    gpu_uuids: tuple[str, ...]
    server_argv_sha256: str
    profiler_subject: FormalSingleOperatorProfilerSubjectRequirement | None
    trusted_chronobelief_gpu_parity_proof: CanonicalJsonProofBinding | None = None
    nextn_mtp_mode: Literal["built_in_mtp"] | None = None
    target_snapshot_sha256: str | None = None
    mtp_component_sha256: str | None = None
    mtp_component: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2, 3}
            or self.kind != "formal_single_operator_prepared_launch_entry"
            or self.protocol_sha256
            != {
                1: FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256,
                2: TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256,
                3: (
                    TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256
                ),
            }[self.schema_version]
            or self.physical_kind not in {"serving", "profiler", "e5_failure"}
        ):
            raise ValueError("prepared launch entry schema differs")
        for label, value in (
            ("cell", self.materialized_cell_id),
            ("compatibility key", self.launch_compatibility_key_sha256),
            ("inventory", self.inventory_sha256),
            ("server argv", self.server_argv_sha256),
        ):
            _require_sha256(f"prepared launch {label}", value)
        _require_text("prepared launch target member", self.target_content_member_id)
        _require_text(
            "prepared launch tokenizer member", self.tokenizer_content_member_id
        )
        if self.drafter_content_member_id is not None:
            _require_text(
                "prepared launch drafter member", self.drafter_content_member_id
            )
        _stable_binding(self.run_config, label="prepared launch RunConfig")
        _stable_binding(
            self.compile_launch_manifest,
            label="prepared launch compile manifest",
        )
        if self.physical_kind in {"serving", "e5_failure"}:
            if self.request_schedule_receipt is None:
                raise FormalSingleOperatorPreparedLaunchBlocked(
                    "source_owned_request_schedule_missing"
                )
            _stable_binding(
                self.request_schedule_receipt,
                label="prepared launch request schedule",
            )
        elif self.request_schedule_receipt is not None:
            raise ValueError(
                "profiler launch uses its dedicated code-owned request schedule"
            )
        expected_gpus = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpus
            or len(set(self.gpu_uuids)) != len(self.gpu_uuids)
            or any(type(value) is not str or not value for value in self.gpu_uuids)
        ):
            raise ValueError("prepared launch GPU order differs from topology")
        if (self.physical_kind == "profiler") != (self.profiler_subject is not None):
            raise ValueError(
                "only profiler launches require a profiler subject requirement"
            )
        if self.profiler_subject is not None and type(self.profiler_subject) is not (
            FormalSingleOperatorProfilerSubjectRequirement
        ):
            raise TypeError("prepared launch profiler subject is not exact")
        if self.schema_version == 1:
            if self.trusted_chronobelief_gpu_parity_proof is not None:
                raise ValueError("legacy prepared launch carries trusted proof")
        elif self.trusted_chronobelief_gpu_parity_proof is not None:
            _stable_binding(
                self.trusted_chronobelief_gpu_parity_proof,
                label="prepared launch trusted ChronoBelief proof",
            )
        if self.schema_version == 3:
            if (
                self.nextn_mtp_mode != "built_in_mtp"
                or self.target_snapshot_sha256 is None
                or self.mtp_component_sha256 is None
                or self.target_snapshot_sha256 == self.mtp_component_sha256
                or type(self.mtp_component) is not CanonicalJsonProofBinding
                or self.mtp_component.semantic_sha256 != self.mtp_component_sha256
                or self.target_content_member_id != self.drafter_content_member_id
            ):
                raise ValueError("prepared built-in MTP identity differs")
            _require_sha256(
                "prepared built-in MTP target snapshot",
                self.target_snapshot_sha256,
            )
            _require_sha256(
                "prepared built-in MTP component",
                self.mtp_component_sha256,
            )
            _stable_binding(
                self.mtp_component,
                label="prepared built-in MTP component",
            )
        elif any(
            value is not None
            for value in (
                self.nextn_mtp_mode,
                self.target_snapshot_sha256,
                self.mtp_component_sha256,
                self.mtp_component,
            )
        ):
            raise ValueError("legacy/external prepared launch carries built-in MTP")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "materialized_cell_id": self.materialized_cell_id,
            "physical_kind": self.physical_kind,
            "run_config": self.run_config.to_dict(),
            "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
            "request_schedule_receipt": (
                None
                if self.request_schedule_receipt is None
                else self.request_schedule_receipt.to_dict()
            ),
            "launch_compatibility_key_sha256": (self.launch_compatibility_key_sha256),
            "target_content_member_id": self.target_content_member_id,
            "drafter_content_member_id": self.drafter_content_member_id,
            "tokenizer_content_member_id": self.tokenizer_content_member_id,
            "inventory_sha256": self.inventory_sha256,
            "topology_mode": self.topology_mode,
            "gpu_uuids": list(self.gpu_uuids),
            "server_argv_sha256": self.server_argv_sha256,
            "profiler_subject": (
                None
                if self.profiler_subject is None
                else self.profiler_subject.to_dict()
            ),
        }
        if self.schema_version == 2:
            value["trusted_chronobelief_gpu_parity_proof"] = (
                None
                if self.trusted_chronobelief_gpu_parity_proof is None
                else self.trusted_chronobelief_gpu_parity_proof.to_dict()
            )
        elif self.schema_version == 3:
            value["trusted_chronobelief_gpu_parity_proof"] = (
                None
                if self.trusted_chronobelief_gpu_parity_proof is None
                else self.trusted_chronobelief_gpu_parity_proof.to_dict()
            )
            assert self.mtp_component is not None
            value["nextn_mtp_mode"] = self.nextn_mtp_mode
            value["target_snapshot_sha256"] = self.target_snapshot_sha256
            value["mtp_component_sha256"] = self.mtp_component_sha256
            value["mtp_component"] = self.mtp_component.to_dict()
        if include_sha256:
            value["entry_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("prepared launch entry must be an object")
        schema_version = value.get("schema_version")
        expected = {*cls.__dataclass_fields__, "entry_sha256"}
        if schema_version == 1:
            expected.remove("trusted_chronobelief_gpu_parity_proof")
        if schema_version in {1, 2}:
            expected -= {
                "nextn_mtp_mode",
                "target_snapshot_sha256",
                "mtp_component_sha256",
                "mtp_component",
            }
        row = _strict(
            "prepared launch entry",
            value,
            expected,
        )
        declared = _require_sha256("prepared launch entry", row.pop("entry_sha256"))
        row["run_config"] = CanonicalJsonProofBinding.from_dict(row["run_config"])
        row["compile_launch_manifest"] = CanonicalJsonProofBinding.from_dict(
            row["compile_launch_manifest"]
        )
        raw_schedule = row.pop("request_schedule_receipt")
        row["request_schedule_receipt"] = (
            None
            if raw_schedule is None
            else CanonicalJsonProofBinding.from_dict(raw_schedule)
        )
        raw_gpus = row.pop("gpu_uuids")
        if type(raw_gpus) is not list:
            raise TypeError("prepared launch GPU UUIDs must be an array")
        raw_profiler = row.pop("profiler_subject")
        raw_chronobelief = row.pop("trusted_chronobelief_gpu_parity_proof", None)
        raw_mtp_component = row.pop("mtp_component", None)
        result = cls(
            **row,
            gpu_uuids=tuple(raw_gpus),
            profiler_subject=(
                None
                if raw_profiler is None
                else FormalSingleOperatorProfilerSubjectRequirement.from_dict(
                    raw_profiler
                )
            ),
            trusted_chronobelief_gpu_parity_proof=(
                None
                if raw_chronobelief is None
                else CanonicalJsonProofBinding.from_dict(raw_chronobelief)
            ),
            mtp_component=(
                None
                if raw_mtp_component is None
                else CanonicalJsonProofBinding.from_dict(raw_mtp_component)
            ),
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("prepared launch entry digest differs")
        return result


@dataclass(frozen=True)
class FormalSingleOperatorPreparedLaunchBundle:
    """Canonical concrete launch set for one exact current materialization."""

    schema_version: Literal[1, 2, 3]
    kind: Literal["formal_single_operator_prepared_launch_bundle"]
    protocol_sha256: str
    node: FormalSingleOperatorNode
    stage: Literal["E4", "E3b", "E1a", "E5", "E6", "E0"]
    phase: str
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    protocol_lock_sha256: str
    materialization_sha256: str
    materialization_source_decision_sha256: str
    inventory: CanonicalJsonProofBinding
    content_verification_receipt: CanonicalJsonProofBinding | None
    entries: tuple[FormalSingleOperatorPreparedLaunchEntry, ...]
    content_source_binding: FormalContentSourceBinding | None = None
    entries_shard_index: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2, 3}
            or self.kind != "formal_single_operator_prepared_launch_bundle"
        ):
            raise ValueError("prepared launch bundle schema differs")
        if self.schema_version == 1:
            if (
                self.protocol_sha256
                != FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
                or type(self.content_verification_receipt)
                is not CanonicalJsonProofBinding
                or self.content_source_binding is not None
            ):
                raise ValueError("legacy prepared launch content source differs")
        else:
            expected_protocol = (
                TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
                if self.schema_version == 2
                else (
                    TRUSTED_SINGLE_OPERATOR_SHARDED_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
                )
            )
            if (
                self.protocol_sha256 != expected_protocol
                or self.content_verification_receipt is not None
                or type(self.content_source_binding) is not FormalContentSourceBinding
                or self.content_source_binding.mode != "trusted_single_operator"
            ):
                raise ValueError("trusted prepared launch content source differs")
        _require_text("prepared launch bundle node", self.node)
        _require_text("prepared launch bundle phase", self.phase)
        for label, value in (
            ("execution source", self.execution_source_sha256),
            ("ProtocolLock", self.protocol_lock_sha256),
            ("materialization", self.materialization_sha256),
            ("source decision", self.materialization_source_decision_sha256),
        ):
            _require_sha256(f"prepared launch bundle {label}", value)
        for label, binding in (
            ("execution source", self.execution_source),
            ("inventory", self.inventory),
        ):
            _stable_binding(binding, label=f"prepared launch bundle {label}")
        if self.content_verification_receipt is not None:
            _stable_binding(
                self.content_verification_receipt,
                label="prepared launch bundle content receipt",
            )
        if self.content_source_binding is not None:
            self.content_source_binding.reopen()
        if type(self.entries) is not tuple or any(
            type(row) is not FormalSingleOperatorPreparedLaunchEntry
            for row in self.entries
        ):
            raise ValueError("prepared launch bundle entries are not canonical")
        if self.schema_version in {1, 2}:
            if (
                self.entries_shard_index is not None
                or tuple(row.materialized_cell_id for row in self.entries)
                != tuple(sorted({row.materialized_cell_id for row in self.entries}))
                or len({row.sha256 for row in self.entries}) != len(self.entries)
            ):
                raise ValueError("prepared launch bundle entries are not canonical")
        elif (
            self.entries
            or type(self.entries_shard_index) is not CanonicalJsonProofBinding
        ):
            raise ValueError("sharded prepared launch must bind only its entry index")
        if self.entries_shard_index is not None:
            _stable_binding(
                self.entries_shard_index,
                label="prepared launch bundle entry shard index",
            )

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "node": self.node,
            "stage": self.stage,
            "phase": self.phase,
            "execution_source": self.execution_source.to_dict(),
            "execution_source_sha256": self.execution_source_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_sha256": self.materialization_sha256,
            "materialization_source_decision_sha256": (
                self.materialization_source_decision_sha256
            ),
            "inventory": self.inventory.to_dict(),
            "content_verification_receipt": (
                None
                if self.content_verification_receipt is None
                else self.content_verification_receipt.to_dict()
            ),
        }
        if self.schema_version in {1, 2}:
            value["entries"] = [row.to_dict() for row in self.entries]
        else:
            assert self.entries_shard_index is not None
            value["entries_shard_index"] = self.entries_shard_index.to_dict()
        if self.schema_version in {2, 3}:
            assert self.content_source_binding is not None
            value["content_source_binding"] = self.content_source_binding.to_dict()
        if include_sha256:
            value["bundle_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("prepared launch bundle must be an object")
        schema_version = value.get("schema_version")
        expected = {*cls.__dataclass_fields__, "bundle_sha256"}
        if schema_version == 1:
            expected.remove("content_source_binding")
        if schema_version in {1, 2}:
            expected.remove("entries_shard_index")
        elif schema_version == 3:
            expected.remove("entries")
        row = _strict(
            "prepared launch bundle",
            value,
            expected,
        )
        declared = _require_sha256("prepared launch bundle", row.pop("bundle_sha256"))
        for name in ("execution_source", "inventory"):
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
        raw_entries = row.pop("entries", [])
        if type(raw_entries) is not list:
            raise TypeError("prepared launch bundle entries must be an array")
        raw_shard_index = row.pop("entries_shard_index", None)
        row["entries_shard_index"] = (
            None
            if raw_shard_index is None
            else CanonicalJsonProofBinding.from_dict(raw_shard_index)
        )
        result = cls(
            **row,
            entries=tuple(
                FormalSingleOperatorPreparedLaunchEntry.from_dict(item)
                for item in raw_entries
            ),
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("prepared launch bundle digest differs")
        return result


def formal_single_operator_prepared_entries_artifact_id(
    *,
    bundle: FormalSingleOperatorPreparedLaunchBundle,
    materialized_cell_ids: tuple[str, ...],
) -> str:
    """Return the non-circular identity used by the schema-3 entry shards."""

    if type(bundle) is not FormalSingleOperatorPreparedLaunchBundle:
        raise TypeError("prepared launch entry artifact requires an exact bundle")
    if (
        type(materialized_cell_ids) is not tuple
        or not materialized_cell_ids
        or materialized_cell_ids != tuple(sorted(set(materialized_cell_ids)))
    ):
        raise ValueError("prepared launch sharded cell IDs are not canonical")
    for cell_id in materialized_cell_ids:
        _require_sha256("prepared launch sharded cell", cell_id)
    if bundle.content_source_binding is None:
        content_source_identity: dict[str, object] = {
            "mode": "offline_root_signed",
            "content_verification_receipt_sha256": (
                None
                if bundle.content_verification_receipt is None
                else bundle.content_verification_receipt.semantic_sha256
            ),
        }
    else:
        content_source_identity = {
            "mode": bundle.content_source_binding.mode,
            "content_source_binding_sha256": bundle.content_source_binding.sha256,
            "content_bundle_sha256": bundle.content_source_binding.content_sha256,
        }
    return _content_sha256(
        {
            "schema_version": 1,
            "kind": FORMAL_SINGLE_OPERATOR_PREPARED_ENTRY_SHARD_ARTIFACT_KIND,
            "execution_source_sha256": bundle.execution_source_sha256,
            "protocol_lock_sha256": bundle.protocol_lock_sha256,
            "materialization_sha256": bundle.materialization_sha256,
            "materialization_source_decision_sha256": (
                bundle.materialization_source_decision_sha256
            ),
            "node": bundle.node,
            "stage": bundle.stage,
            "phase": bundle.phase,
            "inventory_sha256": bundle.inventory.semantic_sha256,
            "content_source": content_source_identity,
            "ordered_materialized_cell_ids_sha256": _content_sha256(
                materialized_cell_ids
            ),
            "entry_count": len(materialized_cell_ids),
        }
    )


def _load_prepared_entry_shard_index(
    *,
    bundle: FormalSingleOperatorPreparedLaunchBundle,
    materialized_cell_ids: tuple[str, ...],
    deep: bool,
) -> FormalCanonicalSequenceShardIndex:
    if bundle.schema_version != 3 or bundle.entries_shard_index is None:
        raise ValueError("prepared launch bundle is not sharded schema 3")
    index = load_formal_canonical_sequence_shard_index(
        bundle.entries_shard_index.absolute_path,
        deep=deep,
    )
    if (
        index.artifact_kind != FORMAL_SINGLE_OPERATOR_PREPARED_ENTRY_SHARD_ARTIFACT_KIND
        or index.artifact_id
        != formal_single_operator_prepared_entries_artifact_id(
            bundle=bundle,
            materialized_cell_ids=materialized_cell_ids,
        )
        or index.total_rows != len(materialized_cell_ids)
        or bundle.entries_shard_index.semantic_sha256
        != _content_sha256(index.to_dict())
    ):
        raise ValueError("prepared launch entry shard index identity differs")
    return index


def shard_formal_single_operator_prepared_launch_bundle(
    *,
    bundle: FormalSingleOperatorPreparedLaunchBundle,
    entries_shard_index: CanonicalJsonProofBinding,
) -> FormalSingleOperatorPreparedLaunchBundle:
    """Lift an in-memory trusted schema-2 bundle into bounded schema 3.

    The caller publishes ``entry.to_dict()`` rows using the generic formal
    sequence publisher and the artifact identity returned by
    :func:`formal_single_operator_prepared_entries_artifact_id` first.
    """

    if (
        type(bundle) is not FormalSingleOperatorPreparedLaunchBundle
        or bundle.schema_version != 2
        or bundle.content_source_binding is None
        or not bundle.entries
        or type(entries_shard_index) is not CanonicalJsonProofBinding
    ):
        raise ValueError("only a populated trusted schema-2 bundle can be sharded")
    cell_ids = tuple(row.materialized_cell_id for row in bundle.entries)
    result = FormalSingleOperatorPreparedLaunchBundle(
        schema_version=3,
        kind=bundle.kind,
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SHARDED_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
        ),
        node=bundle.node,
        stage=bundle.stage,
        phase=bundle.phase,
        execution_source=bundle.execution_source,
        execution_source_sha256=bundle.execution_source_sha256,
        protocol_lock_sha256=bundle.protocol_lock_sha256,
        materialization_sha256=bundle.materialization_sha256,
        materialization_source_decision_sha256=(
            bundle.materialization_source_decision_sha256
        ),
        inventory=bundle.inventory,
        content_verification_receipt=None,
        entries=(),
        content_source_binding=bundle.content_source_binding,
        entries_shard_index=entries_shard_index,
    )
    _load_prepared_entry_shard_index(
        bundle=result,
        materialized_cell_ids=cell_ids,
        deep=True,
    )
    return result


@dataclass(frozen=True)
class RevalidatedFormalSingleOperatorPreparedLaunchBundle:
    """Deep-checked projection of one trusted current prepared bundle."""

    bundle: FormalSingleOperatorPreparedLaunchBundle
    execution_source: FormalSingleOperatorExecutionSource
    inventory: GpuInventory
    required_cell_ids: tuple[str, ...]
    validated_entries: tuple[FormalSingleOperatorPreparedLaunchEntry, ...]
    entry_run_configs: tuple[tuple[str, RunConfig], ...]

    def __post_init__(self) -> None:
        if (
            type(self.bundle) is not FormalSingleOperatorPreparedLaunchBundle
            or type(self.execution_source) is not FormalSingleOperatorExecutionSource
            or type(self.inventory) is not GpuInventory
            or self.required_cell_ids != tuple(sorted(set(self.required_cell_ids)))
            or tuple(row.materialized_cell_id for row in self.validated_entries)
            != tuple(row[0] for row in self.entry_run_configs)
            or tuple(row[0] for row in self.entry_run_configs)
            != tuple(sorted({row[0] for row in self.entry_run_configs}))
            or {row[0] for row in self.entry_run_configs} - set(self.required_cell_ids)
        ):
            raise ValueError("revalidated prepared launch projection differs")

    def entry(
        self, materialized_cell_id: str
    ) -> FormalSingleOperatorPreparedLaunchEntry:
        matches = tuple(
            row
            for row in self.validated_entries
            if row.materialized_cell_id == materialized_cell_id
        )
        if len(matches) != 1:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_prepared_launch_entry_missing"
            )
        return matches[0]


def formal_single_operator_prepared_execution_identities(
    *,
    bundle: FormalSingleOperatorPreparedLaunchBundle,
    entry: FormalSingleOperatorPreparedLaunchEntry,
) -> tuple[str, str]:
    """Return the non-circular binding/subject identities used by a schedule."""

    if (
        type(bundle) is not FormalSingleOperatorPreparedLaunchBundle
        or type(entry) is not FormalSingleOperatorPreparedLaunchEntry
    ):
        raise TypeError("prepared execution identity requires exact bundle/entry")
    binding_payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_single_operator_prepared_execution_binding",
        "protocol_sha256": (
            FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
        ),
        "execution_source_sha256": bundle.execution_source_sha256,
        "materialization_sha256": bundle.materialization_sha256,
        "materialized_cell_id": entry.materialized_cell_id,
        "run_config_sha256": entry.run_config.semantic_sha256,
        "compile_launch_manifest_sha256": (
            entry.compile_launch_manifest.semantic_sha256
        ),
        "inventory_sha256": entry.inventory_sha256,
        "topology_mode": entry.topology_mode,
        "gpu_uuids": entry.gpu_uuids,
    }
    if bundle.schema_version == 1:
        assert bundle.content_verification_receipt is not None
        binding_payload["content_verification_receipt_sha256"] = (
            bundle.content_verification_receipt.semantic_sha256
        )
    else:
        assert bundle.content_source_binding is not None
        binding_payload["content_source_binding_sha256"] = (
            bundle.content_source_binding.sha256
        )
    execution_binding_sha256 = _content_sha256(binding_payload)
    subject_sha256 = _content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_prepared_serving_subject",
            "execution_binding_sha256": execution_binding_sha256,
            "launch_compatibility_key_sha256": (entry.launch_compatibility_key_sha256),
        }
    )
    return execution_binding_sha256, subject_sha256


def _validate_prepared_request_schedule(
    *,
    bundle: FormalSingleOperatorPreparedLaunchBundle,
    entry: FormalSingleOperatorPreparedLaunchEntry,
    source: FormalSingleOperatorExecutionSource,
    cell: object,
    config: RunConfig,
) -> None:
    schedule_binding = entry.request_schedule_receipt
    if schedule_binding is None:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_request_schedule_missing"
        )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
        FormalServingRequestScheduleSource,
        formal_serving_request_schedule_rows,
        formal_serving_request_schedule_source_rows,
    )

    schedule = FormalServingRequestScheduleReceipt.from_dict(schedule_binding.reopen())
    schedule_source = FormalServingRequestScheduleSource.from_dict(
        schedule.schedule_source.load()
    )
    execution_binding_sha256, subject_sha256 = (
        formal_single_operator_prepared_execution_identities(
            bundle=bundle,
            entry=entry,
        )
    )
    expected_workload_authority = (
        source.formal_workload_e0_authorization_sha256
        if bundle.stage == "E0"
        else source.formal_workload_e3a_authorization_sha256
    )
    current_materialization = CanonicalJsonProofBinding.bind(
        source.materialization_source.absolute_path
    )
    if bundle.schema_version == 1:
        assert bundle.content_verification_receipt is not None
        content_lineage_matches = (
            schedule.schema_version in {3, 4}
            and schedule.content_verification_receipt_sha256
            == bundle.content_verification_receipt.semantic_sha256
            and schedule.content_verification_receipt
            == bundle.content_verification_receipt
            and schedule.content_source_binding is None
        )
    else:
        assert bundle.content_source_binding is not None
        content_lineage_matches = (
            schedule.schema_version in {5, 6, 7}
            and schedule.content_verification_receipt_sha256 is None
            and schedule.content_verification_receipt is None
            and schedule.content_source_binding == bundle.content_source_binding
            and schedule_source.schema_version == schedule.schema_version
            and schedule_source.content_source_binding_sha256
            == bundle.content_source_binding.sha256
            and schedule.trusted_workload_member_sha256
            == schedule_source.trusted_workload_member_sha256
        )
    if (
        schedule.sha256 != schedule_binding.semantic_sha256
        or schedule.execution_binding_sha256 != execution_binding_sha256
        or schedule.subject_sha256 != subject_sha256
        or schedule.materialized_cell_id != entry.materialized_cell_id
        or schedule.workload_authority_sha256 != expected_workload_authority
        or not content_lineage_matches
        or schedule.topology_mode != entry.topology_mode
        or schedule.materialization != current_materialization
        or schedule.compile_launch_manifest != entry.compile_launch_manifest
        or schedule.tokenizer_model_id
        != CompileLaunchManifest.load(
            entry.compile_launch_manifest.absolute_path
        ).tokenizer_model_id
        or schedule_source.sha256 != schedule.schedule_source.semantic_sha256
        or schedule_source.subject_sha256 != subject_sha256
        or schedule_source.materialization_receipt_sha256
        != bundle.materialization_sha256
        or schedule_source.materialized_cell_id != entry.materialized_cell_id
        or schedule_source.topology_mode != entry.topology_mode
        or schedule_source.max_running_requests != config.runtime.max_running_requests
        or (
            schedule_source.request_count
            if schedule_source.schema_version in {6, 7}
            else len(schedule_source.requests)
        )
        != (
            schedule.request_count
            if schedule.schema_version in {6, 7}
            else len(schedule.requests)
        )
    ):
        raise ValueError("prepared launch request schedule lineage differs")
    source_rows = formal_serving_request_schedule_source_rows(schedule_source)
    receipt_rows = formal_serving_request_schedule_rows(schedule)
    for source_row, receipt_row in zip(
        source_rows,
        receipt_rows,
        strict=True,
    ):
        if (
            source_row.source_member_sha256 != receipt_row.source_member_sha256
            or source_row.source_sample_id != receipt_row.source_sample_id
            or source_row.prompt_sha256 != receipt_row.prompt_sha256
            or source_row.phase != receipt_row.phase
            or source_row.routed_dp_rank != receipt_row.routed_dp_rank
            or source_row.arrival_us != receipt_row.request.arrival_us
            or source_row.cohort_id != receipt_row.request.cohort_id
            or source_row.requested_output_tokens
            != receipt_row.request.requested_output_tokens
            or source_row.sampling != receipt_row.request.sampling.items
        ):
            raise ValueError("prepared launch request schedule rows differ")
    dimensions = dict(cell.dimensions)
    if cell.stage == "E3b" and (
        schedule_source.context_tokens != dimensions.get("context")
        or schedule_source.regime != dimensions.get("regime")
    ):
        raise ValueError("prepared E3b request schedule scientific axes differ")
    if cell.stage == "E1a" and schedule_source.workload_id != ("livecodebench_v6_hard"):
        raise ValueError("prepared E1a request schedule workload differs")
    if cell.stage == "E6":
        expected_workload = {
            "LiveCodeBench": "livecodebench_v6_hard",
            "MATH-500": "math500_level5",
        }.get(cell.task)
        if (
            expected_workload is None
            or schedule_source.workload_id != expected_workload
            or schedule_source.context_tokens != dimensions.get("context")
        ):
            raise ValueError("prepared E6 request schedule scientific axes differ")
    if cell.stage == "E0":
        if schedule_source.schema_version in {5, 6, 7}:
            if schedule_source.trusted_task_native_workload_sha256 != dimensions.get(
                "task_native_workload_sha256"
            ):
                raise ValueError("prepared trusted E0 workload differs")
        elif schedule_source.workload_source_descriptor_sha256 != dimensions.get(
            "task_native_workload_sha256"
        ):
            raise ValueError("prepared E0 task-native workload differs")
    # Reopen every path-bound derivation input without invoking the legacy
    # E3a-only workload reducer (E0 task-native schedules use another reducer).
    schedule.workload_source.load()
    schedule.sampling_profile.reopen()
    if schedule.schema_version in {6, 7}:
        schedule.reopen()
    else:
        token_input = schedule.tokenization_input.reopen()
        token_output = schedule.tokenization_output.reopen()
        if (
            token_input.get("schedule_source_sha256") != schedule_source.sha256
            or token_output.get("schedule_source_sha256") != schedule_source.sha256
            or token_output.get("requests") is None
        ):
            raise ValueError("prepared launch tokenization lineage differs")
    _validate_prepared_e5_arrival_plan(
        cell=cell,
        schedule=schedule,
        schedule_source=schedule_source,
    )


def _expected_e5_paired_trace_sha256(cell: object) -> str:
    from lightcone_spec.experiments.formal_single_operator_loads import (
        FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256,
    )

    dimensions = dict(cell.dimensions)
    family_axes = {
        "backend_authority",
        "family_id",
        "topology",
        "concurrency",
        "load_factor",
        "arrival",
        "cohort_count",
        "cohort_distribution",
    }
    scientific = {
        name: dimensions[name] for name in sorted(family_axes & set(dimensions))
    }
    return _content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_e5_paired_trace_identity",
            "protocol_sha256": FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256,
            "family": dimensions.get("family"),
            "block": dimensions.get("block"),
            "scientific_axes": scientific,
        }
    )


def _validate_prepared_e5_arrival_plan(
    *,
    cell: object,
    schedule: object,
    schedule_source: object,
) -> object | None:
    """Deep-check the signed-schema4 or trusted-schema5 E5 arrival plan."""

    source_plan_binding = schedule_source.e5_arrival_plan
    receipt_plan_binding = schedule.e5_arrival_plan
    if cell.stage != "E5":
        if source_plan_binding is not None or receipt_plan_binding is not None:
            raise ValueError("non-E5 prepared schedule carries an arrival plan")
        return None
    if cell.task == "deterministic_failure_injection":
        if source_plan_binding is not None or receipt_plan_binding is not None:
            raise ValueError("E5 failure schedule carries a headline arrival plan")
        return None
    if cell.task != "production_slo_power_prefix":
        raise ValueError("E5 prepared schedule task is unsupported")
    from lightcone_spec.experiments.formal_single_operator_loads import E5ArrivalPlan

    if (
        schedule.schema_version not in {4, 5, 6}
        or schedule_source.schema_version != schedule.schema_version
        or type(source_plan_binding) is not CanonicalJsonProofBinding
        or source_plan_binding != receipt_plan_binding
    ):
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_e5_arrival_plan_missing"
        )
    plan = E5ArrivalPlan.from_dict(source_plan_binding.reopen())
    dimensions = dict(cell.dimensions)
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        formal_serving_request_schedule_rows,
        formal_serving_request_schedule_source_rows,
    )

    source_scored_arrivals = tuple(
        row.arrival_us
        for row in formal_serving_request_schedule_source_rows(schedule_source)
        if row.phase == "scored"
    )
    receipt_scored_arrivals = tuple(
        row.request.arrival_us
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    if (
        plan.sha256 != source_plan_binding.semantic_sha256
        or plan.cell_id != cell.cell_id
        or plan.block != dimensions.get("block")
        or plan.family != dimensions.get("family")
        or plan.paired_trace_sha256 != _expected_e5_paired_trace_sha256(cell)
        or schedule_source.load_protocol_sha256 != plan.sha256
        or schedule_source.arrival_policy != plan.arrival_policy
        or schedule_source.max_running_requests != plan.concurrency
        or source_scored_arrivals != plan.arrivals_us
        or receipt_scored_arrivals != plan.arrivals_us
        or (dimensions.get("p99_extension_anchor_id") is not None)
        != (plan.p99_extension_minimum_completed is not None)
        or (
            plan.p99_extension_minimum_completed
            != dimensions.get("p99_extension_minimum_completions")
        )
        or (
            plan.p99_extension_offered_requests
            != dimensions.get("p99_extension_offered_requests")
        )
    ):
        raise ValueError("prepared E5 arrival plan differs from current cell")
    return plan


def _validate_prepared_e5_arrival_pairing(
    *,
    bundle: FormalSingleOperatorPreparedLaunchBundle,
    entries: tuple[FormalSingleOperatorPreparedLaunchEntry, ...],
    cells: dict[str, object],
) -> None:
    """Require byte-identical arrivals across all five methods in each pair."""

    groups: dict[str, list[tuple[object, tuple[int, ...]]]] = {}
    for entry in entries:
        cell = cells[entry.materialized_cell_id]
        if cell.stage != "E5" or cell.task != "production_slo_power_prefix":
            continue
        if entry.request_schedule_receipt is None:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_request_schedule_missing"
            )
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            FormalServingRequestScheduleReceipt,
            FormalServingRequestScheduleSource,
        )

        schedule = FormalServingRequestScheduleReceipt.from_dict(
            entry.request_schedule_receipt.reopen()
        )
        schedule_source = FormalServingRequestScheduleSource.from_dict(
            schedule.schedule_source.load()
        )
        plan = _validate_prepared_e5_arrival_plan(
            cell=cell,
            schedule=schedule,
            schedule_source=schedule_source,
        )
        assert plan is not None
        groups.setdefault(plan.paired_trace_sha256, []).append((cell, plan.arrivals_us))
    expected_roles = {"Target-only", "Static", "TTS", "L0-naive", "LightCone"}
    if any(
        len(rows) != 5
        or {cell.method_role for cell, _arrivals in rows} != expected_roles
        or len({_content_sha256(arrivals) for _cell, arrivals in rows}) != 1
        for rows in groups.values()
    ):
        raise ValueError("E5 paired methods use different arrival bytes")


def _prepared_release(
    receipt: ContentVerificationReceipt,
    *,
    current_ns: int,
) -> VerifiedPreparedModelContentRelease:
    verified = receipt.revalidate_formal_scope(current_ns=current_ns)
    prepared = tuple(
        row for row in verified if type(row) is VerifiedPreparedModelContentRelease
    )
    if len(prepared) != 1:
        raise ValueError("prepared launch bundle lacks one model-content authority")
    return prepared[0]


def _snapshot_for_member(
    *,
    receipt: ContentVerificationReceipt,
    member: AuthorizedPreparedModel,
) -> PreparedModelSnapshotContent:
    matches = tuple(
        artifact
        for artifact in receipt.content_artifacts
        if artifact.raw_sha256 == member.snapshot_manifest_raw_sha256
        and artifact.semantic_sha256 == member.snapshot_manifest_semantic_sha256
    )
    identities = {
        (row.path, row.raw_sha256, row.semantic_sha256, row.size) for row in matches
    }
    if len(identities) != 1:
        raise ValueError("prepared launch member snapshot is not exact")
    snapshot = PreparedModelSnapshotContent.from_dict(matches[0].load())
    if (snapshot.model_id, snapshot.revision) != (
        member.model_id,
        member.revision,
    ):
        raise ValueError("prepared launch member snapshot identity differs")
    return snapshot


def _validate_launch_content(
    *,
    stage: str,
    launch: CompileLaunchManifest,
    config: RunConfig,
    receipt: ContentVerificationReceipt | None,
    prepared: VerifiedPreparedModelContentRelease | None,
    content_source_binding: FormalContentSourceBinding | None,
) -> None:
    if launch.schema_version in {2, 3}:
        if (
            type(content_source_binding) is not FormalContentSourceBinding
            or content_source_binding.mode != "trusted_single_operator"
            or launch.content_source_binding != content_source_binding
            or launch.formal_stage != stage
            or receipt is not None
            or prepared is not None
        ):
            raise ValueError("trusted prepared launch content lineage differs")
        # CompileLaunchManifest.load has already deep-reopened the BOUND bundle
        # and checked exact stage/role/model/revision/path/tree membership.
        content_source_binding.reopen()
        return
    if (
        type(receipt) is not ContentVerificationReceipt
        or type(prepared) is not VerifiedPreparedModelContentRelease
        or content_source_binding is not None
    ):
        raise ValueError("signed prepared launch content lineage differs")
    authorization = prepared.authorization
    if (
        launch.model_lock_sha256 != authorization.model_lock_sha256
        or launch.prepared_model_content_manifest_raw_sha256
        != authorization.content_manifest_raw_sha256
        or launch.prepared_model_content_manifest_sha256
        != authorization.content_manifest_semantic_sha256
        or launch.prepared_model_content_manifest_size
        != authorization.content_manifest_size
    ):
        raise ValueError("prepared launch manifest differs from content authority")
    stage_members = {row.member_id: row for row in prepared.require_stage(stage)}

    def require_member(
        *,
        member_id: str,
        role: str,
        model_id: str,
        revision: str,
        snapshot_path: str,
        content_authority_sha256: str,
    ) -> None:
        member = stage_members.get(member_id)
        if (
            type(member) is not AuthorizedPreparedModel
            or member.role != role
            or member.backend != config.model.algorithm
            or member.model_id != model_id
            or member.revision != revision
            or content_authority_sha256 != prepared.authorization_sha256
        ):
            raise ValueError("prepared launch content member differs from cell")
        snapshot = _snapshot_for_member(receipt=receipt, member=member)
        if snapshot.root != snapshot_path:
            raise ValueError("prepared launch snapshot root differs from authority")

    require_member(
        member_id=launch.target_content_member_id,
        role="target",
        model_id=launch.target_model_id,
        revision=launch.target_revision,
        snapshot_path=launch.target_snapshot_path,
        content_authority_sha256=launch.target_content_authority_sha256,
    )
    if launch.drafter_content_member_id is not None:
        assert launch.drafter_model_id is not None
        assert launch.drafter_revision is not None
        assert launch.drafter_snapshot_path is not None
        assert launch.drafter_content_authority_sha256 is not None
        require_member(
            member_id=launch.drafter_content_member_id,
            role="drafter",
            model_id=launch.drafter_model_id,
            revision=launch.drafter_revision,
            snapshot_path=launch.drafter_snapshot_path,
            content_authority_sha256=(launch.drafter_content_authority_sha256),
        )
    require_member(
        member_id=launch.tokenizer_content_member_id,
        role="tokenizer",
        model_id=launch.tokenizer_model_id,
        revision=launch.tokenizer_revision,
        snapshot_path=launch.tokenizer_snapshot_path,
        content_authority_sha256=launch.tokenizer_content_authority_sha256,
    )


def formal_single_operator_launch_compatibility_key(
    *,
    launch: CompileLaunchManifest,
    config: RunConfig,
) -> str:
    """Derive the exact code-owned launch grouping key from concrete inputs."""

    if type(launch) is not CompileLaunchManifest or type(config) is not RunConfig:
        raise TypeError("launch compatibility requires exact launch and RunConfig")
    cache_plan = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_single_operator_launch_compatibility_key",
        "target": (
            launch.target_content_member_id,
            launch.target_model_id,
            launch.target_revision,
            launch.target_content_authority_sha256,
        ),
        "drafter": (
            launch.drafter_content_member_id,
            launch.drafter_model_id,
            launch.drafter_revision,
            launch.drafter_content_authority_sha256,
        ),
        "tokenizer": (
            launch.tokenizer_content_member_id,
            launch.tokenizer_model_id,
            launch.tokenizer_revision,
            launch.tokenizer_content_authority_sha256,
        ),
        "backend": config.model.algorithm,
        "topology": config.runtime.topology_mode,
        "server_context_limit": config.runtime.context_length,
        "graph_mode": config.runtime.cuda_graph_mode,
        "graph_buckets": config.runtime.cuda_graph_batch_sizes,
        "tensor_parallel_size": config.runtime.tensor_parallel_size,
        "data_parallel_size": config.runtime.data_parallel_size,
        "patched_sglang_tree": launch.patched_sglang_tree,
        "server_argv_sha256": launch.server_argv_sha256,
        "cache_policy": {
            "disable_radix_cache": config.runtime.disable_radix_cache,
            "cache_mode": cache_plan.cache_mode,
            "compile_cache_key_sha256": cache_plan.key.sha256,
        },
    }
    if launch.schema_version == 3:
        value.update(
            {
                "nextn_mtp_mode": launch.nextn_mtp_mode,
                "target_snapshot_sha256": launch.target_snapshot_sha256,
                "mtp_component_sha256": launch.mtp_component_sha256,
            }
        )
    return _content_sha256(value)


def _validate_profiler_requirement(
    *,
    bundle: FormalSingleOperatorPreparedLaunchBundle,
    entry: FormalSingleOperatorPreparedLaunchEntry,
    cell: object,
    source: FormalSingleOperatorExecutionSource,
) -> None:
    requirement = entry.profiler_subject
    if requirement is None:
        raise ValueError("profiler prepared launch lacks its subject requirement")
    dimensions = dict(cell.dimensions)
    if requirement.selected_configuration_sha256 != dimensions.get(
        "selected_configuration_sha256"
    ):
        raise ValueError("profiler subject names another selected configuration")
    selected_launch = CompileLaunchManifest.load(
        requirement.selected_compile_launch_manifest.absolute_path
    )
    profile_launch = CompileLaunchManifest.load(
        entry.compile_launch_manifest.absolute_path
    )
    selected_config = load_run_config(
        requirement.selected_full_run_config.absolute_path
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        rebuild_formal_single_operator_stage_completion,
    )
    from lightcone_spec.experiments.stage_materialization import (
        E4_SCREEN_FACTOR_LEVELS,
    )

    if source.node != "e4_profiler" or source.predecessor_completion_source is None:
        raise ValueError("profiler subject requires the completed E4-local source")
    predecessor = rebuild_formal_single_operator_stage_completion(
        source.predecessor_completion_source.absolute_path
    )
    raw_winner = predecessor.decision.payload.get("winner_configuration")
    if type(raw_winner) is not list:
        raise ValueError("profiler subject lacks the E4-local winner configuration")
    winner_rows: list[tuple[str, str | int]] = []
    for row in raw_winner:
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) not in {str, int}
        ):
            raise ValueError("profiler E4-local winner configuration differs")
        winner_rows.append((row[0], row[1]))
    winner = tuple(winner_rows)
    if predecessor.artifact.node != "e4_local" or tuple(
        name for name, _levels in E4_SCREEN_FACTOR_LEVELS
    ) != tuple(name for name, _value in winner):
        raise ValueError("profiler predecessor is not the exact E4-local winner")
    headline_matches = tuple(
        row
        for row in predecessor.materialization.cells
        if dict(row.dimensions).get("load") == "saturation"
        and dict(row.dimensions).get("traffic") == "mixed_prefill_decode"
        and tuple(
            (name, dict(row.dimensions).get(name))
            for name, _levels in E4_SCREEN_FACTOR_LEVELS
        )
        == winner
    )
    if len(headline_matches) != 1:
        raise ValueError("profiler subject lacks one deterministic headline stratum")
    headline_cell = headline_matches[0]
    profile_config = load_run_config(entry.run_config.absolute_path)
    if selected_config.adaptation is None or profile_config.adaptation is None:
        raise ValueError("profiler selected full configuration lacks adaptation")
    factor_values = {
        "update_stride": selected_config.adaptation.stride,
        "microbatch": selected_config.runtime.adaptation_microbatch_size,
        "coalescing": (selected_config.runtime.adaptation_publication_coalescing),
        "stream_priority": selected_config.runtime.adaptation_stream_priority,
    }
    selected_configuration_sha256 = _content_sha256(
        tuple((name, factor_values[name]) for name, _levels in E4_SCREEN_FACTOR_LEVELS)
    )
    expected_profile_config = selected_config.model_copy(
        update={
            "runtime": selected_config.runtime.model_copy(
                update={"telemetry_detail": "profile"}
            ),
            "adaptation": selected_config.adaptation.model_copy(
                update={
                    "adaptation_group_id": (f"formal-single-e4-{cell.cell_id[:24]}")
                }
            ),
        }
    )
    expected_profile_config = RunConfig.model_validate(
        expected_profile_config.model_dump(mode="json")
    )
    if (
        requirement.selected_compile_launch_manifest.semantic_sha256
        != selected_launch.sha256
        or requirement.selected_full_run_config.semantic_sha256
        != run_config_sha256(selected_config)
        or selected_launch.run_config_path
        != requirement.selected_full_run_config.absolute_path
        or selected_launch.run_config_semantic_sha256
        != run_config_sha256(selected_config)
        or selected_launch.inventory_sha256 != profile_launch.inventory_sha256
        or selected_launch.gpu_uuids != profile_launch.gpu_uuids
        or selected_launch.target_content_member_id
        != profile_launch.target_content_member_id
        or selected_launch.drafter_content_member_id
        != profile_launch.drafter_content_member_id
        or selected_launch.tokenizer_content_member_id
        != profile_launch.tokenizer_content_member_id
        or selected_launch.sampling_profile_sha256
        != profile_launch.sampling_profile_sha256
        or selected_launch.sampling_profile_path != profile_launch.sampling_profile_path
        or selected_config.runtime.telemetry_detail != "headline"
        or profile_config.runtime.telemetry_detail != "profile"
        or profile_config != expected_profile_config
        or requirement.selected_configuration_sha256 != selected_configuration_sha256
        or selected_configuration_sha256 != _content_sha256(winner)
        or requirement.source_headline_cell_id != headline_cell.cell_id
        or {
            "update_stride": profile_config.adaptation.stride,
            "microbatch": profile_config.runtime.adaptation_microbatch_size,
            "coalescing": (profile_config.runtime.adaptation_publication_coalescing),
            "stream_priority": profile_config.runtime.adaptation_stream_priority,
        }
        != factor_values
    ):
        raise ValueError("profiler selected full configuration differs")
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
    )

    schedule = FormalServingRequestScheduleReceipt.from_dict(
        requirement.code_owned_request_schedule.reopen()
    )
    schedule.reopen()
    workload_binding = CanonicalJsonProofBinding.bind(schedule.schedule_source.path)
    selected_materialization = CanonicalJsonProofBinding.bind(
        predecessor.node_materialization.materialization_source.absolute_path
    )
    if bundle.schema_version == 1:
        assert bundle.content_verification_receipt is not None
        content_lineage_matches = (
            schedule.content_verification_receipt == bundle.content_verification_receipt
            and schedule.content_verification_receipt_sha256
            == bundle.content_verification_receipt.semantic_sha256
        )
    else:
        assert bundle.content_source_binding is not None
        content_lineage_matches = (
            schedule.schema_version in {5, 6, 7}
            and schedule.content_verification_receipt is None
            and schedule.content_verification_receipt_sha256 is None
            and schedule.content_source_binding == bundle.content_source_binding
        )
    if (
        schedule.sha256 != requirement.code_owned_request_schedule.semantic_sha256
        or schedule.materialized_cell_id != requirement.source_headline_cell_id
        or schedule.materialization != selected_materialization
        or not content_lineage_matches
        or schedule.compile_launch_manifest
        != requirement.selected_compile_launch_manifest
        or workload_binding != requirement.code_owned_profiler_subject_workload
    ):
        raise ValueError("profiler code-owned workload subject differs")


@dataclass(frozen=True)
class _TrustedChainRecipeContext:
    """Numeric recipe state reconstructed only from the completed current DAG."""

    protocol_lock: object
    matched_width: int
    common_load: int
    frozen_tts_recipe_sha256: str
    tts_learning_rate: float
    tts_stride: int
    lightcone_recipe: object
    dspark_selected_configuration: tuple[tuple[str, str | int], ...] | None
    dspark_selected_recipe_sha256: str | None
    e0_selected_recipes: tuple[tuple[str, str, str], ...]


def _trusted_completion_chain(
    source: FormalSingleOperatorExecutionSource,
) -> dict[str, object]:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        rebuild_formal_single_operator_stage_completion,
    )

    if source.predecessor_completion_source is None:
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "trusted_predecessor_chain_missing"
        )
    current = rebuild_formal_single_operator_stage_completion(
        source.predecessor_completion_source.absolute_path
    )
    result: dict[str, object] = {}
    while current is not None:
        node = current.artifact.node
        if node in result:
            raise ValueError("trusted predecessor chain repeats a node")
        result[node] = current
        current = current.predecessor
    return result


def _trusted_chain_recipe_context(
    source: FormalSingleOperatorExecutionSource,
) -> _TrustedChainRecipeContext:
    """Recover frozen recipes from deeply rebuilt decisions, never defaults."""

    from lightcone_spec.experiments.formal_protocol import content_sha256
    from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
    from lightcone_spec.experiments.formal_single_operator_downstream import (
        _e0_selected_recipes,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        _e2_recipe_from_payload,
    )
    from lightcone_spec.experiments.stage_materialization import (
        default_e2_recipe_grid_authority,
    )

    chain = _trusted_completion_chain(source)
    required = {"e3a", "tts_cal", "e2_r3"}
    if not required <= set(chain):
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "trusted_frozen_recipe_chain_incomplete"
        )
    protocol_lock = protocol_lock_from_dict(source.protocol_lock_source.reopen())
    e3a = chain["e3a"].decision.payload
    tts = chain["tts_cal"].decision.payload
    e2 = chain["e2_r3"].decision.payload
    matched_width = e3a.get("matched_width")
    common_load = e3a.get("common_load")
    learning_rate = tts.get("learning_rate")
    stride = tts.get("stride")
    candidate_id = tts.get("candidate_id")
    if (
        type(matched_width) is not int
        or matched_width not in {4, 8, 16}
        or type(common_load) is not int
        or common_load < 1
        or type(learning_rate) is not float
        or learning_rate <= 0
        or type(stride) is not int
        or stride < 1
        or candidate_id
        != content_sha256(
            {
                "authority_sha256": (protocol_lock.tts_calibration_authority_sha256),
                "learning_rate": learning_rate,
                "stride": stride,
            }
        )
    ):
        raise ValueError("trusted frozen TTS decision differs")
    lightcone = _e2_recipe_from_payload(e2.get("final_recipe"))
    grid = default_e2_recipe_grid_authority()
    if (
        lightcone.sha256 != e2.get("final_recipe", {}).get("recipe_sha256")
        or grid.sha256 != protocol_lock.e2_recipe_grid_authority_sha256
    ):
        raise ValueError("trusted sealed LightCone recipe differs")

    dspark_configuration: tuple[tuple[str, str | int], ...] | None = None
    dspark_recipe: str | None = None
    if "e1a" in chain:
        payload = chain["e1a"].decision.payload
        raw_configuration = payload.get("selected_configuration")
        if type(raw_configuration) is not list:
            raise ValueError("trusted E1a decision lacks selected configuration")
        parsed: list[tuple[str, str | int]] = []
        for row in raw_configuration:
            if (
                type(row) is not list
                or len(row) != 2
                or type(row[0]) is not str
                or type(row[1]) not in {str, int}
            ):
                raise ValueError("trusted E1a selected configuration differs")
            parsed.append((row[0], row[1]))
        dspark_configuration = tuple(parsed)
        if tuple(name for name, _value in dspark_configuration) != (
            "parameterization",
            "rank",
            "scope",
        ):
            raise ValueError("trusted E1a configuration keys differ")
        dspark_recipe = content_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_e1a_selected_dspark_recipe",
                "source_lightcone_recipe_sha256": lightcone.sha256,
                "configuration": raw_configuration,
                "rule": (
                    "max_minimum_confidence_lower_ratio_then_min_hbm_p99_exposed_digest"
                ),
            }
        )
        if (
            payload.get("source_lightcone_recipe_sha256") != lightcone.sha256
            or payload.get("selected_dspark_recipe_sha256") != dspark_recipe
        ):
            raise ValueError("trusted E1a selected DSpark recipe differs")

    selected: tuple[tuple[str, str, str], ...] = ()
    if source.node in {"e0_pilot", "e0_final"}:
        tuning = chain.get("e0_tuning")
        if tuning is None:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "trusted_e0_tuning_winner_chain_missing"
            )
        selected = tuple(
            sorted(
                (
                    decision_id,
                    role,
                    recipe,
                )
                for (decision_id, role), recipe in _e0_selected_recipes(
                    tuning.decision.payload
                ).items()
            )
        )
    return _TrustedChainRecipeContext(
        protocol_lock=protocol_lock,
        matched_width=matched_width,
        common_load=common_load,
        frozen_tts_recipe_sha256=candidate_id,
        tts_learning_rate=learning_rate,
        tts_stride=stride,
        lightcone_recipe=lightcone,
        dspark_selected_configuration=dspark_configuration,
        dspark_selected_recipe_sha256=dspark_recipe,
        e0_selected_recipes=selected,
    )


def _expected_publication_policy(cell: object) -> str:
    if cell.task in {"mechanism_profile_only", "deterministic_failure_injection"}:
        return "diagnostic_only"
    if cell.method_role == "TTS":
        return "fixed_barrier"
    if cell.method_role in {"L0-naive", "LightCone", "LightCone-candidate"}:
        return "first_ready"
    if cell.method_role.startswith("OnlineSPEC-"):
        return (
            "tuning_only"
            if cell.method_role.endswith("-candidate")
            else "independent_online"
        )
    return "none"


def _trusted_adaptation_group(cell: object, *, paired_tts_l0: bool) -> str:
    if paired_tts_l0:
        dimensions = dict(cell.dimensions)
        pair = dimensions.get("tts_l0_pair_id")
        if pair is None:
            pair = _content_sha256(
                {
                    "stage": cell.stage,
                    "model": cell.model,
                    "backend": cell.backend,
                    "task": cell.task,
                    "dimensions": dimensions,
                }
            )
        _require_sha256("trusted TTS/L0 pair", pair)
        return f"formal-single-tts-l0-{pair[:24]}"
    return f"formal-single-{cell.stage.lower()}-{cell.cell_id[:24]}"


def _trusted_expected_width(
    context: _TrustedChainRecipeContext,
    cell: object,
    config: RunConfig,
) -> int:
    dimensions = dict(cell.dimensions)
    if cell.stage == "E1a":
        verification = dimensions.get("verification_mode")
        fixed_budget = dimensions.get("fixed_verification_budget")
        if (
            verification == "fixed_verification_budget"
            and fixed_budget != E1A_FIXED_VERIFICATION_BUDGET
        ) or (
            verification == "native_scheduler"
            and fixed_budget != E1A_NATIVE_VERIFICATION_BUDGET
        ):
            raise ValueError("trusted E1a fixed verification budget differs")
        if verification not in {
            "fixed_verification_budget",
            "native_scheduler",
        }:
            raise ValueError("trusted E1a verification mode differs")
    if cell.stage == "E3b":
        panel = dimensions.get("width_panel")
        if panel == "matched":
            return context.matched_width
        if panel == "deployment_optimal":
            # DFlash's registered deployment panel is its full width 16.  The
            # panel is fixed before E3b and therefore never selected from E3b.
            return 16
        raise ValueError("trusted E3b width panel differs")
    if cell.stage == "E1a":
        return context.matched_width
    if cell.stage == "E5":
        return 16
    if cell.stage in {"E6", "E0"}:
        return config.model.draft_depth + 1
    return config.runtime.speculative_num_draft_tokens


def _trusted_expected_load(
    context: _TrustedChainRecipeContext,
    cell: object,
) -> int | None:
    dimensions = dict(cell.dimensions)
    if cell.stage == "E3b":
        if dimensions.get("load") == "concurrency_one":
            return 1
        if dimensions.get("load") == "common_load":
            return context.common_load
        raise ValueError("trusted E3b load panel differs")
    if cell.stage == "E1a":
        return context.common_load
    if cell.stage in {"E6", "E0"}:
        if dimensions.get("load") == "concurrency_one":
            return 1
        if dimensions.get("load") == "common_slo_load":
            return context.common_load
        if cell.stage == "E6" or cell.task != "independent_onlinespec_tuning":
            raise ValueError("trusted common-SLO load panel differs")
        return context.common_load
    return None


def _trusted_geometry(
    configuration: tuple[tuple[str, str | int], ...],
) -> tuple[str, str, int | None, str]:
    values = dict(configuration)
    parameterization = values.get("parameterization")
    rank_value = values.get("rank")
    scope = values.get("scope")
    if parameterization not in {"full", "lora"} or type(scope) is not str or not scope:
        raise ValueError("trusted DSpark geometry differs")
    rank = None if rank_value == "none" else int(rank_value)
    native_head_policy = "full" if scope.endswith("_native_heads") else "frozen"
    return str(parameterization), scope, rank, native_head_policy


def _trusted_e0_eagle3_task_authority(
    *,
    source: FormalSingleOperatorExecutionSource,
    cell: object,
    config: RunConfig,
) -> dict[str, str]:
    """Reopen one exact task proof row from the current E0 auxiliary."""

    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        e0_eagle3_runtime_authority_for_task,
        e0_eagle3_runtime_proof_row_for_task,
        load_e0_compatibility_probe_terminal,
        load_e0_prepared_model_backend_interface_receipt,
        revalidate_trusted_e0_compatibility_bundle_value,
    )

    if (
        source.stage != "E0"
        or cell.stage != "E0"
        or cell.backend != "EAGLE3"
        or config.model.algorithm != "EAGLE3"
        or config.adaptation is None
    ):
        raise ValueError("trusted E0 EAGLE3 task authority scope differs")
    auxiliary = source.auxiliary_source_binding("e0_compatibility")
    publication = revalidate_trusted_e0_compatibility_bundle_value(
        auxiliary.reopen(label="prepared bundle E0 compatibility auxiliary")
    )
    decisions = tuple(
        row
        for row in publication.compatibility.decisions
        if (row.model, row.backend, row.task) == (cell.model, "EAGLE3", cell.task)
    )
    interface_bindings = tuple(
        binding
        for binding in publication.evidence_manifest.interface_receipts
        if (
            (raw := binding.reopen()).get("model") == cell.model
            and raw.get("backend") == "EAGLE3"
        )
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
    if (
        len(decisions) != 1
        or decisions[0].disposition != "VALID"
        or len(interface_bindings) != 1
        or len(terminal_bindings) != 1
    ):
        raise FormalSingleOperatorPreparedLaunchBlocked(
            "source_owned_eagle3_runtime_authority_missing"
        )
    interface = load_e0_prepared_model_backend_interface_receipt(
        interface_bindings[0].absolute_path
    )
    terminal = load_e0_compatibility_probe_terminal(terminal_bindings[0].absolute_path)
    proof_row = e0_eagle3_runtime_proof_row_for_task(
        interface,
        task=cell.task,
        terminal=terminal,
    )
    if (
        interface.schema_version not in {2, 3}
        or interface.support_status != "READY"
        or interface.target_model_id != config.model.target
        or interface.target_revision != config.model.target_revision
        or interface.drafter_model_id != config.model.drafter
        or interface.drafter_revision != config.model.drafter_revision
        or terminal.schema_version != interface.schema_version
        or terminal.disposition != "VALID"
        or terminal.interface_receipt_sha256 != interface.sha256
        or terminal.eagle3_runtime_proof_row_sha256 != proof_row.sha256
    ):
        raise ValueError("trusted E0 EAGLE3 task proof lineage differs")
    execution = proof_row.execution_authority.reopen()
    if (
        type(execution) is not dict
        or execution.get("task") != cell.task
        or execution.get("target_revision") != config.model.target_revision
        or execution.get("drafter_revision") != config.model.drafter_revision
        or execution.get("interface_sha256") != decisions[0].interface_sha256
        or execution.get("inventory_sha256")
        != proof_row.native_gpu_proof.reopen().get("inventory_sha256")
    ):
        raise ValueError("trusted E0 EAGLE3 execution proof scope differs")
    return e0_eagle3_runtime_authority_for_task(
        interface,
        task=cell.task,
        terminal=terminal,
    )


def _validate_trusted_chain_run_config(
    *,
    context: _TrustedChainRecipeContext,
    source: FormalSingleOperatorExecutionSource,
    cell: object,
    config: RunConfig,
    trusted_chronobelief_gpu_parity_proof_sha256: str | None = None,
) -> None:
    """Bind every reconstructible downstream RunConfig field to current DAG state."""

    from lightcone_spec.config import (
        AdaptationConfig,
        OnlineSpecConfig,
        OptimizerConfig,
    )
    from lightcone_spec.experiments.onlinespec import onlinespec_candidates
    from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY
    from lightcone_spec.experiments.stage_materialization import (
        default_e2_recipe_grid_authority,
    )

    if cell.publication_policy != _expected_publication_policy(cell):
        raise ValueError("trusted cell publication policy differs from role")
    dimensions = dict(cell.dimensions)
    expected_algorithm = {
        "E3b": "DFLASH",
        "E1a": "DSPARK",
        "E5": dimensions.get("backend_authority", cell.backend),
        "E6": "NEXTN",
        "E0": cell.backend,
    }.get(cell.stage, cell.backend)
    if expected_algorithm == "NONE":
        expected_algorithm = {
            "E3b": "DFLASH",
            "E1a": "DSPARK",
            "E6": "NEXTN",
        }.get(cell.stage)
    if expected_algorithm is not None and config.model.algorithm != expected_algorithm:
        raise ValueError("trusted RunConfig backend differs from materialized cell")
    if cell.stage == "E6":
        if config.runtime.topology_mode != "tp2_dp1":
            raise ValueError("trusted E6 RunConfig is not TP2/DP1")
        for name, actual in (
            ("target_model_id", config.model.target),
            ("target_revision", config.model.target_revision),
            ("drafter_model_id", config.model.drafter),
            ("drafter_revision", config.model.drafter_revision),
            ("nextn_mtp_mode", config.model.nextn_mtp_mode),
            ("target_snapshot_sha256", config.model.target_snapshot_sha256),
            ("mtp_component_sha256", config.model.mtp_component_sha256),
        ):
            if (
                name in dimensions or config.model.nextn_mtp_mode == "built_in_mtp"
            ) and dimensions.get(name) != actual:
                raise ValueError(
                    "trusted E6 model pair differs from interface authority"
                )
    expected_load = _trusted_expected_load(context, cell)
    if (
        expected_load is not None
        and config.runtime.max_running_requests != expected_load
    ):
        raise ValueError("trusted RunConfig load differs from frozen selection")
    width = _trusted_expected_width(context, cell, config)
    if cell.method_role != "Target-only" and (
        config.runtime.speculative_num_draft_tokens != width
    ):
        raise ValueError("trusted RunConfig width differs from frozen panel")

    if cell.method_role in {"Target-only", "Static"}:
        if (
            cell.recipe_sha256 is not None
            or config.adaptation is not None
            or config.online_spec is not None
        ):
            raise ValueError("trusted non-adaptive role allocated recipe state")
        return
    if config.adaptation is None:
        raise ValueError("trusted adaptive role lacks adaptation state")
    eagle3 = (
        _trusted_e0_eagle3_task_authority(
            source=source,
            cell=cell,
            config=config,
        )
        if config.model.algorithm == "EAGLE3"
        else {}
    )

    if cell.method_role in {"TTS", "L0-naive"}:
        if (
            cell.recipe_sha256 != context.frozen_tts_recipe_sha256
            or config.online_spec is not None
        ):
            raise ValueError("trusted TTS/L0 recipe identity differs")
        expected = AdaptationConfig(
            weight_update_mode="full",
            parameter_scope="all",
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
        if config.adaptation != expected:
            raise ValueError("trusted TTS/L0 numeric RunConfig differs")
        return

    if cell.method_role.startswith("OnlineSPEC-"):
        candidates = {row.candidate_id: row for row in onlinespec_candidates()}
        candidate = candidates.get(cell.recipe_sha256 or "")
        if candidate is None:
            raise ValueError("trusted OnlineSPEC recipe is not registered")
        if source.node in {"e0_pilot", "e0_final"}:
            selected = {
                (decision_id, role): recipe
                for decision_id, role, recipe in context.e0_selected_recipes
            }
            if (
                selected.get(
                    (str(dimensions.get("compatibility_decision_id")), cell.method_role)
                )
                != candidate.candidate_id
            ):
                raise ValueError("trusted OnlineSPEC serving row is not its winner")
        elif (
            source.node != "e0_tuning"
            or not cell.method_role.endswith("-candidate")
            or dimensions.get("candidate_id") != candidate.candidate_id
            or dimensions.get("onlinespec_method") != candidate.method
        ):
            raise ValueError("trusted OnlineSPEC tuning row differs")
        expected_adaptation = AdaptationConfig(
            weight_update_mode=candidate.weight_update_mode,
            parameter_scope=candidate.parameter_scope,
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
        expected_online = OnlineSpecConfig(
            projection_radius=candidate.projection_radius,
            additional_learning_rates=candidate.additional_learning_rates,
            hedge_learning_rate=candidate.hedge_learning_rate,
        )
        if (
            config.method != candidate.method
            or config.adaptation != expected_adaptation
            or config.online_spec != expected_online
        ):
            raise ValueError("trusted OnlineSPEC numeric RunConfig differs")
        return

    if cell.method_role not in {"LightCone", "LightCone-candidate"}:
        raise ValueError("trusted adaptive method role is unsupported")
    if config.online_spec is not None:
        raise ValueError("trusted LightCone allocated OnlineSPEC state")
    grid = default_e2_recipe_grid_authority()
    recipe = context.lightcone_recipe
    expected_recipe = recipe.sha256
    configuration: tuple[tuple[str, str | int], ...] | None = None
    if cell.stage == "E1a":
        configuration = (
            ("parameterization", dimensions.get("parameterization")),
            ("rank", dimensions.get("rank")),
            ("scope", dimensions.get("scope")),
        )
    elif cell.stage == "E5" and config.model.algorithm == "DSPARK":
        configuration = context.dspark_selected_configuration
        expected_recipe = context.dspark_selected_recipe_sha256
        if configuration is None or expected_recipe is None:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "trusted_dspark_winner_missing"
            )
    if cell.recipe_sha256 != expected_recipe:
        raise ValueError("trusted LightCone recipe differs from frozen winner")
    chronobelief_proof = None
    if recipe.optimizer == "chronobelief":
        if cell.stage != "E1a" or trusted_chronobelief_gpu_parity_proof_sha256 is None:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_chronobelief_gpu_proof_missing"
            )
        chronobelief_proof = _require_sha256(
            "trusted ChronoBelief GPU parity proof",
            trusted_chronobelief_gpu_parity_proof_sha256,
        )
    elif trusted_chronobelief_gpu_parity_proof_sha256 is not None:
        raise ValueError("non-ChronoBelief RunConfig carries GPU parity proof")
    expected = grid.adaptation_config_for(
        recipe,
        canvas_tokens=width,
        adaptation_group_id=_trusted_adaptation_group(cell, paired_tts_l0=False),
        chronobelief_gpu_proof_sha256=chronobelief_proof,
    )
    if eagle3:
        expected = AdaptationConfig(**{**expected.model_dump(), **eagle3})
    if configuration is not None:
        parameterization, scope, rank, native_head_policy = _trusted_geometry(
            configuration
        )
        verification = dimensions.get("verification_mode")
        expected = AdaptationConfig(
            **{
                **expected.model_dump(),
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
    if config.adaptation != expected:
        raise ValueError("trusted LightCone numeric RunConfig differs")
    if cell.stage == "E4":
        profile = cell.task == "mechanism_profile_only"
        if config.runtime.telemetry_detail != ("profile" if profile else "headline"):
            raise ValueError("trusted E4 telemetry mode differs")
        if not profile and (
            config.adaptation.stride != int(dimensions["update_stride"])
            or config.runtime.adaptation_microbatch_size
            != int(dimensions["microbatch"])
            or config.runtime.adaptation_publication_coalescing
            != int(dimensions["coalescing"])
            or config.runtime.adaptation_stream_priority
            != dimensions["stream_priority"]
        ):
            raise ValueError("trusted E4 factor RunConfig differs")


def load_formal_single_operator_prepared_launch_bundle(
    path: str | Path,
) -> FormalSingleOperatorPreparedLaunchBundle:
    binding = CanonicalJsonProofBinding.bind(path)
    value = FormalSingleOperatorPreparedLaunchBundle.from_dict(binding.reopen())
    if value.sha256 != binding.semantic_sha256:
        raise ValueError("prepared launch bundle canonical identity differs")
    return value


def revalidate_formal_single_operator_prepared_launch_bundle(
    *,
    execution_source_path: str | Path,
    prepared_launch_bundle_path: str | Path,
    materialized_cell_id: str | None = None,
    current_ns: int,
) -> RevalidatedFormalSingleOperatorPreparedLaunchBundle:
    """Deep-check the current source's prepared launch envelope.

    A physical mapper must still bind the selected row into a schema-2 plan;
    the bundle alone cannot name a different cell or supply caller values.
    """

    if type(current_ns) is not int or current_ns < 1:
        raise ValueError("prepared launch revalidation time is invalid")
    source_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source = load_formal_single_operator_execution_source(source_binding.absolute_path)
    bundle_source = CanonicalJsonProofBinding.bind(prepared_launch_bundle_path)
    bundle = FormalSingleOperatorPreparedLaunchBundle.from_dict(
        bundle_source.reopen(
            label="single-operator prepared launch/stage-source bundle"
        )
    )
    if bundle.sha256 != bundle_source.semantic_sha256:
        raise ValueError("prepared launch bundle semantic identity differs")
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="prepared launch current materialization"
        )
    )
    if (
        bundle.node != source.node
        or bundle.stage != source.stage
        or bundle.phase != source.phase
        or bundle.execution_source != source_binding
        or bundle.execution_source_sha256 != source.sha256
        or bundle.protocol_lock_sha256 != source.protocol_lock_sha256
        or bundle.materialization_sha256 != materialization.sha256
        or bundle.materialization_source_decision_sha256
        != materialization.source_decision_sha256
    ):
        raise ValueError("prepared launch bundle belongs to another current source")
    inventory = GpuInventory.from_dict(bundle.inventory.reopen())
    if inventory.sha256 != bundle.inventory.semantic_sha256:
        raise ValueError("prepared launch inventory identity differs")
    receipt: ContentVerificationReceipt | None
    prepared: VerifiedPreparedModelContentRelease | None
    if bundle.schema_version == 1:
        assert bundle.content_verification_receipt is not None
        receipt = ContentVerificationReceipt.from_dict(
            bundle.content_verification_receipt.reopen()
        )
        if receipt.sha256 != bundle.content_verification_receipt.semantic_sha256:
            raise ValueError("prepared launch content receipt identity differs")
        prepared = _prepared_release(receipt, current_ns=current_ns)
        if (
            prepared.authorization_sha256
            != source.prepared_model_content_authorization_sha256
        ):
            raise ValueError(
                "prepared launch model authority differs from current source"
            )
    else:
        assert bundle.content_source_binding is not None
        trusted_bundle = bundle.content_source_binding.reopen()
        protocol_lock = protocol_lock_from_dict(
            source.protocol_lock_source.reopen(
                label="prepared launch trusted ProtocolLock"
            )
        )
        if (
            protocol_lock.schema_version != 5
            or protocol_lock.content_source_mode != "trusted_single_operator"
            or protocol_lock.trusted_single_operator_content_bundle_sha256
            != bundle.content_source_binding.content_sha256
            or trusted_bundle.runtime_binding_status != "BOUND"
        ):
            raise ValueError("prepared launch trusted bundle differs from ProtocolLock")
        receipt = None
        prepared = None

    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        route_formal_single_operator_materialized_cell,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        _method_for_cell,
        _validate_base_run_config,
    )

    required: dict[str, PreparedLaunchPhysicalKind] = {}
    cells = {cell.cell_id: cell for cell in materialization.cells}
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
        required[cell.cell_id] = route.physical_kind  # type: ignore[assignment]
    required_cell_ids = tuple(sorted(required))
    if materialized_cell_id is not None:
        _require_sha256("prepared launch requested cell", materialized_cell_id)
        if materialized_cell_id not in required:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_prepared_launch_entry_missing"
            )
    if bundle.schema_version in {1, 2}:
        all_entries = bundle.entries
        if tuple(row.materialized_cell_id for row in all_entries) != required_cell_ids:
            raise FormalSingleOperatorPreparedLaunchBlocked(
                "source_owned_prepared_launch_entry_missing_or_extra"
            )
        validated_entries = (
            all_entries
            if materialized_cell_id is None
            else tuple(
                row
                for row in all_entries
                if row.materialized_cell_id == materialized_cell_id
            )
        )
    else:
        entry_index = _load_prepared_entry_shard_index(
            bundle=bundle,
            materialized_cell_ids=required_cell_ids,
            deep=False,
        )
        if materialized_cell_id is None:
            all_entries = tuple(
                FormalSingleOperatorPreparedLaunchEntry.from_dict(row)
                for row in entry_index.iter_rows()
            )
            if (
                tuple(row.materialized_cell_id for row in all_entries)
                != required_cell_ids
            ):
                raise FormalSingleOperatorPreparedLaunchBlocked(
                    "source_owned_prepared_launch_entry_missing_or_extra"
                )
            validated_entries = all_entries
        else:
            entry_ordinal = required_cell_ids.index(materialized_cell_id)
            selected_entry = FormalSingleOperatorPreparedLaunchEntry.from_dict(
                entry_index.row_at(entry_ordinal)
            )
            if selected_entry.materialized_cell_id != materialized_cell_id:
                raise FormalSingleOperatorPreparedLaunchBlocked(
                    "source_owned_prepared_launch_entry_missing"
                )
            all_entries = ()
            validated_entries = (selected_entry,)
    profiler_entries = tuple(
        row for row in all_entries if row.physical_kind == "profiler"
    )
    if materialized_cell_id is None and profiler_entries:
        subject_keys = {
            (
                requirement.source_headline_cell_id,
                requirement.selected_configuration_sha256,
                requirement.selected_full_run_config,
                requirement.selected_compile_launch_manifest,
                requirement.code_owned_profiler_subject_workload,
                requirement.code_owned_request_schedule,
            )
            for row in profiler_entries
            for requirement in (row.profiler_subject,)
            if requirement is not None
        }
        if len(profiler_entries) != 3 or len(subject_keys) != 1:
            raise ValueError(
                "three profiler variants must reuse one deterministic subject"
            )
    if materialized_cell_id is None and bundle.stage == "E5":
        _validate_prepared_e5_arrival_pairing(
            bundle=bundle,
            entries=all_entries,
            cells=cells,
        )
    if materialized_cell_id is None:
        selected_cell_ids = required_cell_ids
    else:
        selected_cell_ids = (materialized_cell_id,)
    entries = {row.materialized_cell_id: row for row in validated_entries}
    trusted_context = _trusted_chain_recipe_context(source)
    configurations: list[tuple[str, RunConfig]] = []
    for cell_id in selected_cell_ids:
        entry = entries[cell_id]
        cell = cells[cell_id]
        if entry.physical_kind != required[cell_id]:
            raise ValueError("prepared launch physical route differs from cell")
        launch = CompileLaunchManifest.load(entry.compile_launch_manifest.absolute_path)
        config = load_run_config(entry.run_config.absolute_path)
        if (bundle.schema_version == 1) != (entry.schema_version == 1):
            raise ValueError("prepared launch entry/content mode differs")
        chronobelief_proof_sha256 = None
        if entry.trusted_chronobelief_gpu_parity_proof is not None:
            from lightcone_spec.experiments.formal_single_operator_chronobelief import (
                revalidate_trusted_single_operator_chronobelief_for_prepared_launch,
            )

            proof = revalidate_trusted_single_operator_chronobelief_for_prepared_launch(
                proof_path=(entry.trusted_chronobelief_gpu_parity_proof.absolute_path),
                execution_source_path=source_binding.absolute_path,
                prepared_launch_path=entry.compile_launch_manifest.absolute_path,
            )
            chronobelief_proof_sha256 = proof.sha256
        if (
            entry.compile_launch_manifest.semantic_sha256 != launch.sha256
            or entry.run_config.semantic_sha256 != run_config_sha256(config)
            or launch.run_config_path != entry.run_config.absolute_path
            or launch.run_config_semantic_sha256 != run_config_sha256(config)
            or launch.inventory_sha256 != inventory.sha256
            or entry.inventory_sha256 != inventory.sha256
            or entry.gpu_uuids != launch.gpu_uuids
            or entry.topology_mode != config.runtime.topology_mode
            or entry.server_argv_sha256 != launch.server_argv_sha256
            or entry.target_content_member_id != launch.target_content_member_id
            or entry.drafter_content_member_id != launch.drafter_content_member_id
            or entry.tokenizer_content_member_id != launch.tokenizer_content_member_id
            or entry.launch_compatibility_key_sha256
            != formal_single_operator_launch_compatibility_key(
                launch=launch,
                config=config,
            )
            or launch.target_model_id != config.model.target
            or launch.target_revision != config.model.target_revision
            or entry.schema_version != (3 if launch.schema_version == 3 else 2)
            or entry.nextn_mtp_mode
            != ("built_in_mtp" if launch.schema_version == 3 else None)
            or entry.target_snapshot_sha256 != launch.target_snapshot_sha256
            or entry.mtp_component_sha256 != launch.mtp_component_sha256
            or entry.mtp_component != launch.mtp_component_binding
            or (
                config.method != "target_only"
                and (
                    launch.drafter_model_id != config.model.drafter
                    or launch.drafter_revision != config.model.drafter_revision
                )
            )
        ):
            raise ValueError("prepared launch entry concrete identity differs")
        for gpu_uuid in entry.gpu_uuids:
            inventory.device(gpu_uuid)
        _validate_base_run_config(
            cell,
            config,
            expected_method=_method_for_cell(cell),
            topology_mode=entry.topology_mode,
            gpu_uuids=entry.gpu_uuids,
        )
        _validate_trusted_chain_run_config(
            context=trusted_context,
            source=source,
            cell=cell,
            config=config,
            trusted_chronobelief_gpu_parity_proof_sha256=(chronobelief_proof_sha256),
        )
        _validate_launch_content(
            stage=bundle.stage,
            launch=launch,
            config=config,
            receipt=receipt,
            prepared=prepared,
            content_source_binding=bundle.content_source_binding,
        )
        if entry.physical_kind in {"serving", "e5_failure"}:
            _validate_prepared_request_schedule(
                bundle=bundle,
                entry=entry,
                source=source,
                cell=cell,
                config=config,
            )
        if entry.physical_kind == "profiler":
            _validate_profiler_requirement(
                bundle=bundle,
                entry=entry,
                cell=cell,
                source=source,
            )
        configurations.append((cell_id, config))
    return RevalidatedFormalSingleOperatorPreparedLaunchBundle(
        bundle=bundle,
        execution_source=source,
        inventory=inventory,
        required_cell_ids=required_cell_ids,
        validated_entries=validated_entries,
        entry_run_configs=tuple(configurations),
    )


__all__ = [
    "FORMAL_SINGLE_OPERATOR_PREPARED_ENTRY_SHARD_ARTIFACT_KIND",
    "FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_ENTRY_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_SHARDED_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256",
    "FormalSingleOperatorPreparedLaunchBlocked",
    "FormalSingleOperatorPreparedLaunchBundle",
    "FormalSingleOperatorPreparedLaunchEntry",
    "FormalSingleOperatorProfilerSubjectRequirement",
    "RevalidatedFormalSingleOperatorPreparedLaunchBundle",
    "formal_single_operator_launch_compatibility_key",
    "formal_single_operator_prepared_entries_artifact_id",
    "formal_single_operator_prepared_execution_identities",
    "load_formal_single_operator_prepared_launch_bundle",
    "revalidate_formal_single_operator_prepared_launch_bundle",
    "shard_formal_single_operator_prepared_launch_bundle",
]
