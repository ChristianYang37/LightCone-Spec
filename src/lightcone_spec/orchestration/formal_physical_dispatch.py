"""Sealed formal request materialization and callback-free physical dispatch.

The formal stage mapper proves *what* one materialized cell means.  This module
is the narrow bridge that turns that private-sealed binding into physical
requests.  Request text and schedule values come only from a content receipt;
the caller supplies neither prompts, token IDs, ports, argv, nor a live
transport callback.

The GPU process emits unsigned raw evidence.  A later external-control step is
still required before any result is formal authority.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import unicodedata
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields, replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self
from xml.sax.saxutils import quoteattr

from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_single_operator_context_compiler import (
    CompiledContextRequest,
    ContextFillerAuthority,
    TokenizedContextSourceRow,
    compile_context_requests,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FormalServingExecutionBinding,
    FormalSingleOperatorExecutionBinding,
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
    verify_formal_single_operator_execution_binding,
)
from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ImmutableRequest,
    RequestTemplate,
    cohort_assignments,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.serving import (
    BoundServingRequest,
    PinnedBenchServingTransport,
)
from lightcone_spec.experiments.stage_materialization import (
    MaterializedCell,
    StageMaterializationReceipt,
)
from lightcone_spec.experiments.workload_authority import (
    FormalWorkloadAuthority,
    FormalWorkloadId,
    FormalWorkloadSample,
    formal_workload_authority_artifact_id,
    formal_workload_authority_cli_artifact,
    formal_workload_authority_from_cli_artifact,
    revalidate_authorized_formal_workload_authority,
)
from lightcone_spec.orchestration.formal_terminal_shards import (
    publish_scalable_client_request_lifecycle,
    publish_scalable_formal_gang_itl_bundle,
    publish_scalable_formal_gang_request_terminal,
    publish_scalable_formal_gang_terminal,
)
from lightcone_spec.orchestration.live_sglang import (
    _ABORT_TIMEOUT_SECONDS,
    PinnedNvidiaSmiTool,
    ValidatedUnsignedPinnedSglangServingRun,
    _capture_gpu_process_snapshot,
    _execute_source_owned_phase,
    _observe_live_server_execution_policy,
    _publish_gpu_snapshot_error,
    _require_port_unused,
    _terminate_process_group,
    _wait_server_ready,
    execute_unsigned_native_serving_run,
)
from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding
from lightcone_spec.runtime.backend import (
    VerifiedEagle3E0ExecutionAuthority,
    VerifiedNextNTp2Authority,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.content_authorization import (
    ContentJsonArtifactBinding,
    ContentVerificationReceipt,
    TtsCalibrationTuningWindow,
    VerifiedReleaseWorkloadSources,
)
from lightcone_spec.runtime.formal_sharded_artifact import (
    load_formal_canonical_sequence_shard_index,
    publish_formal_canonical_sequence_shards,
)
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import VerifiedNativeRuntimeGpuProof

if TYPE_CHECKING:
    from lightcone_spec.experiments.formal_single_operator_lcb_tokenizer import (
        LiveCodeBenchTokenizedPrompt,
    )
    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 2,
        "kind": "formal_serving_physical_dispatch_protocol",
        "authority": (
            "private_sealed_formal_binding_plus_signed_content_receipt_plus_"
            "deep_reopened_compile_launch_manifest"
        ),
        "tokenization": (
            "first_party_subprocess_local_prepared_snapshot_no_remote_code_"
            "ordered_exact_token_ids"
        ),
        "request_source": "signed_schedule_source_not_caller_values",
        "paired_request_pool": (
            "method_independent_scientific_axes_and_complete_registered_pool"
        ),
        "execution_policy": (
            "source_owned_arrival_deadline_drain_concurrency_and_exact_pool_mode"
        ),
        "process_hard_timeout": (
            "schedule_wave_count_plus_deadline_abort_reconciliation_and_"
            "startup_cleanup_reserves"
        ),
        "outputs": "private_root_atomic_no_replace_unsigned_raw_evidence",
        "live_injection": False,
    }
)
TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 3,
        "kind": "trusted_single_operator_serving_physical_dispatch_protocol",
        "authority": (
            "current_execution_materialization_plus_BOUND_tagged_content_bundle_"
            "plus_deep_reopened_schema2_compile_launch"
        ),
        "tokenization": (
            "first_party_subprocess_local_trusted_snapshot_no_remote_code_"
            "ordered_exact_token_ids"
        ),
        "livecodebench_task_native": (
            "E1_E2_exact_hard80_tokenizer_member_bound_statistics_and_"
            "prompt_plus_requested_output_le_40928_before_GPU_allocation"
        ),
        "request_source": (
            "exact_trusted_LCB_MATH_or_E0_member_plus_code_owned_load_reducer"
        ),
        "paired_request_pool": (
            "method_independent_block_task_load_context_axes_or_E5_paired_trace"
        ),
        "execution_policy": (
            "source_owned_arrival_deadline_drain_concurrency_and_exact_pool_mode"
        ),
        "process_hard_timeout": (
            "schedule_wave_count_plus_deadline_abort_reconciliation_and_"
            "startup_cleanup_reserves"
        ),
        "e5": "exact_deep_reopened_and_rederived_E5ArrivalPlan",
        "chronobelief": (
            "E1a_DSpark_only_deep_replayed_empirical_parity_proof_and_"
            "qualified_GPU_UUID_membership_rechecked_before_child_launch"
        ),
        "claim": "trusted_single_operator_empirical_not_root_authorized",
        "outputs": "private_root_atomic_no_replace_unsigned_raw_evidence",
        "live_injection": False,
    }
)
FORMAL_GANG_SERVING_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 2,
        "kind": "sglang_source_owned_formal_gang_serving_protocol",
        "topologies": ["tp2_dp1", "tp1_dp2"],
        "phases": [
            "begin",
            "warmup_terminal",
            "reset",
            "scored_terminal",
            "finalize",
        ],
        "tp2": "all_rank_prepare_then_atomic_commit_or_zero_rank_abort",
        "dp2": "sticky_cohort_disjoint_request_partition_no_gradient_collective",
        "publication": "unsigned_rank_terminals_then_one_all_rank_aggregate",
        "client_terminal_partition": (
            "scored_lifecycle_digest_exact_schedule_and_native_submission_reconciliation"
        ),
        "warmup": "all_rows_offered_submitted_and_native_completed",
        "scored_non_submission": (
            "only_exact_digest_bound_client_lifecycle_ids_may_omit_native_rank_rows"
        ),
        "caller_callback": False,
    }
)

FORMAL_SERVING_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 3,
        "kind": "formal_serving_request_schedule_derivation_protocol",
        "workload": (
            "root_verified_raw_source_full_row_replay_and_materialized_cell_"
            "deterministic_reduction"
        ),
        "selection": "all_rows_once_warmup_prefix_content_disjoint_from_scored",
        "tts_calibration": (
            "typed_disjoint_window_entries_replayed_to_unique_root_authorized_prompts"
        ),
        "arrival": "closed_loop_zero_think_or_sealed_closed_loop_common_load",
        "output": {
            "short_input_long_generation": "min_2048_or_half_context",
            "long_input_short_generation": "min_256_or_quarter_context",
        },
        "sampling": "path_bound_sampling_profile_plus_source_sample_seed",
        "request_identity": (
            "method_independent_stage_task_block_load_context_pool_namespace_"
            "with_all_replay_fields_content_addressed"
        ),
        "prompt_text": (
            "nonempty_unicode_nfc_exact_bytes_preserved_including_LF_or_CRLF_"
            "multiline_and_terminal_newline_no_nul"
        ),
        "cancellation": "none_unless_a_future_protocol_revision_registers_it",
        "cohort": "source_seeded_code_owned_uniform_assignment",
        "dp_route": "sticky_sha256_cohort_mod_two_with_both_replicas_required",
        "e5_arrival": (
            "schema4_path_bound_E5ArrivalPlan_exact_scored_arrival_offsets_"
            "paired_trace_identity_across_methods"
        ),
        "unsupported": "BLOCKED_before_tokenization",
    }
)
TRUSTED_SINGLE_OPERATOR_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 3,
        "kind": "trusted_single_operator_request_schedule_derivation_protocol",
        "content": ("exact_member_of_BOUND_tagged_bundle_with_deep_file_tree_reopen"),
        "workloads": (
            "locked_LiveCodeBench_v6_hard_or_MATH500_level5_or_exact_READY_"
            "single_turn_E0_task_native_authority"
        ),
        "selection": "all_source_rows_deterministic_warmup_and_scored_reduction",
        "e5": (
            "code_owned_E5ArrivalPlan_exact_rederivation_then_one_scored_row_per_"
            "registered_arrival_offset"
        ),
        "request_identity": (
            "E5_paired_trace_or_method_independent_stage_task_block_load_context_"
            "pool_namespace_with_all_replay_fields_content_addressed"
        ),
        "tokenizer": "exact_stage_role_model_revision_member_of_same_bundle",
        "livecodebench_task_native": (
            "E1_E2_only_complete_hard80_runtime_tokenizer_authority_never_"
            "controlled_E3a_E3b_context_axis"
        ),
        "forbidden": (
            "signed_authorization_claims_or_caller_prompt_arrival_seed_or_load"
        ),
    }
)
TRUSTED_SINGLE_OPERATOR_SHARDED_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 2,
        "kind": "trusted_single_operator_request_schedule_derivation_protocol",
        "rows": (
            "source_and_tokenized_requests_in_deterministic_contiguous_"
            "canonical_sequence_shards"
        ),
        "tokenization": (
            "bounded_first_party_worker_batches_each_path_raw_semantic_bound"
        ),
        "receipt": "small_header_binds_row_and_tokenization_indexes",
        "legacy": "schemas3_4_5_unchanged",
    }
)
TRUSTED_SINGLE_OPERATOR_SHARDED_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 2,
        "kind": "trusted_single_operator_serving_physical_dispatch_protocol",
        "content": "runtime_BOUND_tagged_bundle",
        "schedule": "schema6_sharded_source_and_receipt",
        "claim": "trusted_single_operator_empirical_not_root_authorized",
    }
)
TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 3,
        "kind": "trusted_single_operator_request_schedule_derivation_protocol",
        "scope": ["E3b", "E6"],
        "context_axis": "exact_prompt_plus_requested_generation_tokens",
        "core": "complete_locked_task_native_core_tokens_never_truncated",
        "filler": (
            "one_reusable_path_bound_LCB_hard_plus_MATH_level5_tokenizer_"
            "authority_then_deterministic_compiler"
        ),
        "rows": "bounded_contiguous_source_compiled_and_receipt_shards",
        "claim": "trusted_single_operator_empirical_not_root_authorized",
    }
)
TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256 = (
    _sha256(
        {
            "schema_version": 3,
            "kind": "trusted_single_operator_serving_physical_dispatch_protocol",
            "content": "runtime_BOUND_tagged_bundle",
            "schedule": "schema7_controlled_context_source_and_receipt",
            "context": "compiled_token_ids_deep_replayed_from_locked_core_and_filler",
            "claim": "trusted_single_operator_empirical_not_root_authorized",
        }
    )
)
FORMAL_SINGLE_OPERATOR_EXECUTION_REBUILD_SOURCE_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_execution_rebuild_source_protocol",
        "authority": (
            "path_bound_current_execution_source_runtime_content_inventory_"
            "compile_and_runtime_gpu_proofs"
        ),
        "optional_stage_sources": (
            "tts_calibration_e1_anchor_registry_receipt_and_clean_repository"
        ),
        "rebuild": (
            "rerun_public_single_operator_execution_verifier_then_exact_plan_join"
        ),
        "caller_private_binding": "forbidden",
    }
)

_SCHEDULE_ARTIFACT_PREFIX = "derived_formal_serving_request_schedule:"
FORMAL_TRUSTED_SCHEDULE_SOURCE_ROW_SHARD_ARTIFACT_KIND = (
    "formal_trusted_schedule_source_rows"
)
FORMAL_TRUSTED_SCHEDULE_RECEIPT_ROW_SHARD_ARTIFACT_KIND = (
    "formal_trusted_schedule_receipt_rows"
)
FORMAL_TRUSTED_CONTROLLED_CONTEXT_COMPILED_ROW_SHARD_ARTIFACT_KIND = (
    "formal_trusted_controlled_context_compiled_rows"
)
FORMAL_TRUSTED_CONTROLLED_CONTEXT_SOURCE_ROW_SHARD_ARTIFACT_KIND = (
    "formal_trusted_controlled_context_source_rows"
)
FORMAL_TRUSTED_CONTROLLED_CONTEXT_RECEIPT_ROW_SHARD_ARTIFACT_KIND = (
    "formal_trusted_controlled_context_receipt_rows"
)
_SPLITS = frozenset({"warmup", "tuning", "pilot", "confirmation", "broad_replication"})
_PHASES = frozenset({"warmup", "scored"})
_TOPOLOGIES = frozenset({"tp1_dp1", "tp2_dp1", "tp1_dp2"})
_TRUSTED_E0_WORKLOAD_IDS = frozenset(
    {
        "GSM8K",
        "AIME-2025",
        "MBPP",
        "HumanEval",
        "Alpaca",
        "Arena-Hard",
    }
)
_MAX_REQUEST_ROWS = 100_000
_SINGLE_OPERATOR_PROCESS_TIMEOUT_SECONDS = 3_600.0
_CURRENT_NON_E5_WARMUP_DURATION_US = 600_000_000
_CURRENT_NON_E5_ARRIVAL_DURATION_US = 1
_CURRENT_NON_E5_REQUEST_DEADLINE_US = 600_000_000
_CURRENT_NON_E5_DRAIN_DURATION_US = 120_000_000
_CURRENT_PROCESS_STARTUP_RESERVE_US = 600_000_000
_CURRENT_PROCESS_CLEANUP_RESERVE_US = 120_000_000
_MAX_CURRENT_PROCESS_TIMEOUT_NS = 60 * 24 * 60 * 60 * 1_000_000_000
# The serving task's source-owned hard cap ends before its resident worker has
# deep-validated, hashed, and atomically published the evidence.  The outer
# scheduler reserves this bounded, GPU-free publication interval separately.
FORMAL_SERVING_TERMINAL_PUBLICATION_GRACE_NS = 15 * 60 * 1_000_000_000


class FormalPhysicalDispatchError(RuntimeError):
    def __init__(self, reason_code: str, fatal_pointer: CanonicalJsonProofBinding):
        self.reason_code = reason_code
        self.fatal_pointer = fatal_pointer
        super().__init__(f"formal physical dispatch failed: {reason_code}")


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise ValueError(f"{label} must be canonical text")
    return value


def _require_prompt(value: object) -> str:
    """Preserve NFC prompt bytes exactly, including LF/CRLF and final newline."""

    if (
        type(value) is not str
        or not value
        or not value.strip()
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError("formal request prompt must be nonempty NFC text")
    return value


def _strict_object(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _private_output_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError("formal dispatch output root must be absolute and resolved")
    if not path.is_dir() or path.is_symlink():
        raise ValueError("formal dispatch output root must exist as a directory")
    status = path.stat(follow_symlinks=False)
    if status.st_uid != os.geteuid() or stat.S_IMODE(status.st_mode) & 0o077:
        raise ValueError(
            "formal dispatch output root must be private and current-user-owned"
        )
    return path


def _bound_request_to_dict(value: BoundServingRequest) -> dict[str, object]:
    value.validate()
    return {
        "request_id": value.request_id,
        "namespace": value.namespace,
        "split": value.split,
        "ordinal": value.ordinal,
        "input_token_ids": list(value.input_token_ids),
        "requested_output_tokens": value.requested_output_tokens,
        "arrival_us": value.arrival_us,
        "cancellation_offset_us": value.cancellation_offset_us,
        "cohort_id": value.cohort_id,
        "cohort_sha256": value.cohort_sha256,
        "route_id": value.route_id,
        "sampling": dict(value.sampling.items),
    }


def _bound_request_from_dict(value: object) -> BoundServingRequest:
    row = _strict_object(
        "formal serving bound request",
        value,
        {
            "request_id",
            "namespace",
            "split",
            "ordinal",
            "input_token_ids",
            "requested_output_tokens",
            "arrival_us",
            "cancellation_offset_us",
            "cohort_id",
            "cohort_sha256",
            "route_id",
            "sampling",
        },
    )
    input_token_ids = row.pop("input_token_ids")
    sampling = row.pop("sampling")
    if type(input_token_ids) is not list or type(sampling) is not dict:
        raise TypeError("formal serving token/sampling rows must be arrays/objects")
    result = BoundServingRequest(
        **row,
        input_token_ids=tuple(input_token_ids),
        sampling=FrozenSamplingParameters.from_mapping(sampling),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class FormalServingRequestScheduleSourceRow:
    """One root-authorized request before first-party tokenization."""

    source_member_sha256: str
    source_raw_file_sha256: str
    source_selected_rows_sha256: str
    source_sample_id: str
    prompt: str
    prompt_sha256: str
    phase: Literal["warmup", "scored"]
    namespace: str
    split: str
    ordinal: int
    requested_output_tokens: int
    arrival_us: int
    cancellation_offset_us: int | None
    cohort_id: str
    routed_dp_rank: int | None
    sampling: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        _require_sha256("formal request source member", self.source_member_sha256)
        _require_sha256("formal request raw source", self.source_raw_file_sha256)
        _require_sha256(
            "formal request selected source", self.source_selected_rows_sha256
        )
        _require_text("formal request source sample", self.source_sample_id)
        _require_prompt(self.prompt)
        if self.prompt_sha256 != _sha256(self.prompt):
            raise ValueError("formal request prompt digest differs")
        if self.phase not in _PHASES or self.split not in _SPLITS:
            raise ValueError("formal request phase/split is unsupported")
        if (self.phase == "warmup") != (self.split == "warmup"):
            raise ValueError("formal request warmup phase/split differs")
        for label, value in (
            ("namespace", self.namespace),
            ("cohort", self.cohort_id),
        ):
            _require_text(f"formal request {label}", value)
        if (
            type(self.ordinal) is not int
            or isinstance(self.ordinal, bool)
            or self.ordinal < 0
            or type(self.requested_output_tokens) is not int
            or isinstance(self.requested_output_tokens, bool)
            or self.requested_output_tokens < 1
            or type(self.arrival_us) is not int
            or isinstance(self.arrival_us, bool)
            or self.arrival_us < 0
        ):
            raise ValueError("formal request ordinal/length/arrival is invalid")
        if self.cancellation_offset_us is not None and (
            type(self.cancellation_offset_us) is not int
            or isinstance(self.cancellation_offset_us, bool)
            or self.cancellation_offset_us < 0
        ):
            raise ValueError("formal request cancellation is invalid")
        if self.routed_dp_rank is not None and self.routed_dp_rank not in {0, 1}:
            raise ValueError("formal request DP route is invalid")
        FrozenSamplingParameters(items=self.sampling).validate()

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "formal request source row",
            value,
            {field.name for field in fields(cls)},
        )
        sampling = row.pop("sampling")
        if type(sampling) is not dict:
            raise TypeError("formal request sampling must be an object")
        return cls(
            **row,
            sampling=FrozenSamplingParameters.from_mapping(sampling).items,
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "sampling": dict(self.sampling)}

    @cached_property
    def source_request_key(self) -> str:
        return "source-" + _sha256(
            {
                "source_member_sha256": self.source_member_sha256,
                "source_raw_file_sha256": self.source_raw_file_sha256,
                "source_selected_rows_sha256": self.source_selected_rows_sha256,
                "source_sample_id": self.source_sample_id,
                "prompt_sha256": self.prompt_sha256,
                "phase": self.phase,
                "namespace": self.namespace,
                "split": self.split,
                "ordinal": self.ordinal,
            }
        )


@dataclass(frozen=True)
class FormalServingRequestScheduleSource:
    schema_version: Literal[3, 4, 5, 6, 7]
    kind: Literal["formal_serving_request_schedule_source"]
    protocol_sha256: str
    derivation_protocol_sha256: str
    subject_sha256: str
    materialization_receipt_sha256: str
    materialized_cell_id: str
    workload_authority_sha256: str | None
    workload_id: str
    workload_source_descriptor_sha256: str
    workload_source_authority_sha256: str | None
    tts_tuning_window_sha256: str | None
    tts_tuning_entry_ids: tuple[str, ...]
    sampling_profile_sha256: str
    load_protocol_sha256: str
    context_tokens: int
    regime: str
    arrival_policy: str
    max_running_requests: int
    cohort_count: int
    topology_mode: str
    tokenizer_content_member_id: str
    tokenizer_model_id: str
    tokenizer_revision: str
    tokenizer_content_authority_sha256: str | None
    requests: tuple[FormalServingRequestScheduleSourceRow, ...]
    e5_arrival_plan: CanonicalJsonProofBinding | None = None
    content_source_binding_sha256: str | None = None
    trusted_workload_member_sha256: str | None = None
    trusted_tts_calibration_authority_sha256: str | None = None
    trusted_task_native_workload_sha256: str | None = None
    requests_shard_index: CanonicalJsonProofBinding | None = None
    request_count: int | None = None
    controlled_context_uncompiled_source: CanonicalJsonProofBinding | None = None
    context_filler_artifact: CanonicalJsonProofBinding | None = None
    compiled_context_requests_shard_index: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {3, 4, 5, 6, 7}
            or self.kind != "formal_serving_request_schedule_source"
        ):
            raise ValueError("formal request schedule source schema differs")
        expected_protocol = (
            (
                TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
                if self.schema_version == 7
                else (
                    TRUSTED_SINGLE_OPERATOR_SHARDED_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
                    if self.schema_version == 6
                    else TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
                )
            )
            if self.schema_version in {5, 6, 7}
            else FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        )
        expected_derivation = (
            (
                TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
                if self.schema_version == 7
                else (
                    TRUSTED_SINGLE_OPERATOR_SHARDED_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
                    if self.schema_version == 6
                    else TRUSTED_SINGLE_OPERATOR_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
                )
            )
            if self.schema_version in {5, 6, 7}
            else FORMAL_SERVING_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        )
        if (
            self.protocol_sha256 != expected_protocol
            or self.derivation_protocol_sha256 != expected_derivation
        ):
            raise ValueError("formal request schedule source protocol differs")
        if self.schema_version == 3:
            if self.e5_arrival_plan is not None:
                raise ValueError("legacy formal schedule carries an E5 arrival plan")
        elif self.e5_arrival_plan is not None and (
            type(self.e5_arrival_plan) is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(self.e5_arrival_plan.absolute_path)
            != self.e5_arrival_plan
        ):
            raise ValueError("formal E5 arrival-plan binding changed")
        for label, value in (
            ("subject", self.subject_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("workload source descriptor", self.workload_source_descriptor_sha256),
            ("sampling profile", self.sampling_profile_sha256),
            ("load protocol", self.load_protocol_sha256),
        ):
            _require_sha256(f"formal schedule {label}", value)
        if self.schema_version in {3, 4}:
            for label, value in (
                ("workload authority", self.workload_authority_sha256),
                (
                    "workload source authority",
                    self.workload_source_authority_sha256,
                ),
                ("tokenizer authority", self.tokenizer_content_authority_sha256),
            ):
                _require_sha256(f"formal schedule {label}", value)
            if (
                self.content_source_binding_sha256 is not None
                or self.trusted_workload_member_sha256 is not None
                or self.trusted_tts_calibration_authority_sha256 is not None
                or self.trusted_task_native_workload_sha256 is not None
            ):
                raise ValueError("legacy formal schedule carries trusted lineage")
        else:
            _require_sha256(
                "formal schedule content source binding",
                self.content_source_binding_sha256,
            )
            _require_sha256(
                "formal schedule trusted workload member",
                self.trusted_workload_member_sha256,
            )
            if (
                self.workload_authority_sha256 is not None
                or self.workload_source_authority_sha256 is not None
                or self.tokenizer_content_authority_sha256 is not None
                or self.workload_source_descriptor_sha256
                != self.trusted_workload_member_sha256
                or any(
                    row.source_member_sha256 != self.trusted_workload_member_sha256
                    for row in self.requests
                )
            ):
                raise ValueError(
                    "trusted formal schedule carries misleading authorization lineage"
                )
            if (self.tts_tuning_window_sha256 is None) != (
                self.trusted_tts_calibration_authority_sha256 is None
            ):
                raise ValueError("trusted TTS schedule authority lineage is incomplete")
            if self.trusted_tts_calibration_authority_sha256 is not None:
                _require_sha256(
                    "trusted TTS calibration authority",
                    self.trusted_tts_calibration_authority_sha256,
                )
            if self.trusted_task_native_workload_sha256 is not None:
                _require_sha256(
                    "trusted task-native workload",
                    self.trusted_task_native_workload_sha256,
                )
        context_bindings = (
            self.controlled_context_uncompiled_source,
            self.context_filler_artifact,
            self.compiled_context_requests_shard_index,
        )
        if self.schema_version == 7:
            if (
                self.workload_id not in {"livecodebench_v6_hard", "math500_level5"}
                or self.trusted_task_native_workload_sha256 is not None
                or self.e5_arrival_plan is not None
                or self.regime
                not in {
                    "long_input_short_output",
                    "short_input_long_generation",
                    "multi_turn_shared_prefix",
                    "native_mtp_transfer",
                }
                or self.context_tokens > 40_928
            ):
                raise ValueError("controlled-context schedule scope differs")
            for binding in context_bindings:
                if (
                    type(binding) is not CanonicalJsonProofBinding
                    or CanonicalJsonProofBinding.bind(binding.absolute_path) != binding
                ):
                    raise ValueError("controlled-context schedule binding changed")
        elif any(binding is not None for binding in context_bindings):
            raise ValueError("non-context schedule carries context compilation")
        allowed_workloads = {
            "livecodebench_v6_hard",
            "math500_level5",
        } | (
            set(_TRUSTED_E0_WORKLOAD_IDS) if self.schema_version in {5, 6, 7} else set()
        )
        if self.workload_id not in allowed_workloads:
            raise ValueError("formal schedule workload is unsupported")
        if self.tts_tuning_window_sha256 is None:
            if self.tts_tuning_entry_ids:
                raise ValueError("non-TTS schedule carries tuning-window entries")
        else:
            _require_sha256(
                "formal schedule TTS tuning window",
                self.tts_tuning_window_sha256,
            )
            if not self.tts_tuning_entry_ids or self.tts_tuning_entry_ids != tuple(
                sorted(set(self.tts_tuning_entry_ids))
            ):
                raise ValueError("formal schedule TTS tuning entries differ")
            for entry_id in self.tts_tuning_entry_ids:
                _require_sha256("formal schedule TTS tuning entry", entry_id)
        if (
            type(self.context_tokens) is not int
            or self.context_tokens < 1
            or type(self.max_running_requests) is not int
            or self.max_running_requests < 1
            or type(self.cohort_count) is not int
            or self.cohort_count not in {1, 4, 16, 64}
        ):
            raise ValueError("formal schedule load cardinality is invalid")
        _require_text("formal schedule regime", self.regime)
        _require_text("formal schedule arrival policy", self.arrival_policy)
        if self.topology_mode not in _TOPOLOGIES:
            raise ValueError("formal schedule topology is unsupported")
        for label, value in (
            ("tokenizer member", self.tokenizer_content_member_id),
            ("tokenizer model", self.tokenizer_model_id),
            ("tokenizer revision", self.tokenizer_revision),
        ):
            _require_text(f"formal schedule {label}", value)
        if type(self.requests) is not tuple or any(
            type(row) is not FormalServingRequestScheduleSourceRow
            for row in self.requests
        ):
            raise ValueError("formal schedule request coverage is invalid")
        if self.schema_version in {6, 7}:
            if (
                self.requests
                or type(self.requests_shard_index) is not CanonicalJsonProofBinding
                or type(self.request_count) is not int
                or not 1 <= self.request_count <= _MAX_REQUEST_ROWS
            ):
                raise ValueError("sharded formal schedule source shape differs")
        elif (
            self.requests_shard_index is not None
            or self.request_count is not None
            or not 1 <= len(self.requests) <= _MAX_REQUEST_ROWS
        ):
            raise ValueError("formal schedule request coverage is invalid")
        if self.schema_version in {6, 7}:
            return
        request_keys = tuple(row.source_request_key for row in self.requests)
        ordinals = tuple(row.ordinal for row in self.requests)
        if (
            len(request_keys) != len(set(request_keys))
            or len(ordinals) != len(set(ordinals))
            or ordinals != tuple(sorted(ordinals))
            or not any(row.phase == "scored" for row in self.requests)
        ):
            raise ValueError("formal schedule request identity/order differs")
        cohort_routes: dict[str, int] = {}
        for row in self.requests:
            if self.topology_mode == "tp1_dp2":
                if row.routed_dp_rank not in {0, 1}:
                    raise ValueError("DP2 formal schedule lacks exact route")
                previous = cohort_routes.setdefault(row.cohort_id, row.routed_dp_rank)
                if previous != row.routed_dp_rank:
                    raise ValueError(
                        "DP2 formal schedule violates sticky cohort routing"
                    )
            elif row.routed_dp_rank is not None:
                raise ValueError("non-DP2 formal schedule carries a replica route")
        if self.topology_mode == "tp1_dp2" and set(cohort_routes.values()) != {0, 1}:
            raise ValueError("DP2 formal schedule must cover both replicas")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "derivation_protocol_sha256": self.derivation_protocol_sha256,
            "subject_sha256": self.subject_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "materialized_cell_id": self.materialized_cell_id,
            "workload_authority_sha256": self.workload_authority_sha256,
            "workload_id": self.workload_id,
            "workload_source_descriptor_sha256": (
                self.workload_source_descriptor_sha256
            ),
            "workload_source_authority_sha256": (self.workload_source_authority_sha256),
            "tts_tuning_window_sha256": self.tts_tuning_window_sha256,
            "tts_tuning_entry_ids": list(self.tts_tuning_entry_ids),
            "sampling_profile_sha256": self.sampling_profile_sha256,
            "load_protocol_sha256": self.load_protocol_sha256,
            "context_tokens": self.context_tokens,
            "regime": self.regime,
            "arrival_policy": self.arrival_policy,
            "max_running_requests": self.max_running_requests,
            "cohort_count": self.cohort_count,
            "topology_mode": self.topology_mode,
            "tokenizer_content_member_id": self.tokenizer_content_member_id,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_content_authority_sha256": (
                self.tokenizer_content_authority_sha256
            ),
        }
        if self.schema_version in {6, 7}:
            assert self.requests_shard_index is not None
            value["requests_shard_index"] = self.requests_shard_index.to_dict()
            value["request_count"] = self.request_count
        else:
            value["requests"] = [row.to_dict() for row in self.requests]
        if self.schema_version in {4, 5, 6, 7}:
            value["e5_arrival_plan"] = (
                None if self.e5_arrival_plan is None else self.e5_arrival_plan.to_dict()
            )
        if self.schema_version in {5, 6, 7}:
            value["content_source_binding_sha256"] = self.content_source_binding_sha256
            value["trusted_workload_member_sha256"] = (
                self.trusted_workload_member_sha256
            )
            value["trusted_tts_calibration_authority_sha256"] = (
                self.trusted_tts_calibration_authority_sha256
            )
            value["trusted_task_native_workload_sha256"] = (
                self.trusted_task_native_workload_sha256
            )
        if self.schema_version == 7:
            assert self.controlled_context_uncompiled_source is not None
            assert self.context_filler_artifact is not None
            assert self.compiled_context_requests_shard_index is not None
            value["controlled_context_uncompiled_source"] = (
                self.controlled_context_uncompiled_source.to_dict()
            )
            value["context_filler_artifact"] = self.context_filler_artifact.to_dict()
            value["compiled_context_requests_shard_index"] = (
                self.compiled_context_requests_shard_index.to_dict()
            )
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("formal request schedule source must be an object")
        schema_version = value.get("schema_version")
        expected = {field.name for field in fields(cls)}
        if schema_version == 3:
            expected.remove("e5_arrival_plan")
        if schema_version in {3, 4}:
            expected -= {
                "content_source_binding_sha256",
                "trusted_workload_member_sha256",
                "trusted_tts_calibration_authority_sha256",
                "trusted_task_native_workload_sha256",
            }
        if schema_version in {3, 4, 5, 6}:
            expected -= {
                "controlled_context_uncompiled_source",
                "context_filler_artifact",
                "compiled_context_requests_shard_index",
            }
        if schema_version in {3, 4, 5}:
            expected -= {"requests_shard_index", "request_count"}
        elif schema_version in {6, 7}:
            expected.remove("requests")
        row = _strict_object(
            "formal request schedule source",
            value,
            expected,
        )
        requests = row.pop("requests", [])
        tuning_entries = row.pop("tts_tuning_entry_ids")
        raw_e5_plan = row.pop("e5_arrival_plan", None)
        row.setdefault("content_source_binding_sha256", None)
        row.setdefault("trusted_workload_member_sha256", None)
        row.setdefault("trusted_tts_calibration_authority_sha256", None)
        row.setdefault("trusted_task_native_workload_sha256", None)
        raw_uncompiled_source = row.pop(
            "controlled_context_uncompiled_source",
            None,
        )
        raw_filler_artifact = row.pop("context_filler_artifact", None)
        raw_compiled_index = row.pop(
            "compiled_context_requests_shard_index",
            None,
        )
        row["controlled_context_uncompiled_source"] = (
            None
            if raw_uncompiled_source is None
            else CanonicalJsonProofBinding.from_dict(raw_uncompiled_source)
        )
        row["context_filler_artifact"] = (
            None
            if raw_filler_artifact is None
            else CanonicalJsonProofBinding.from_dict(raw_filler_artifact)
        )
        row["compiled_context_requests_shard_index"] = (
            None
            if raw_compiled_index is None
            else CanonicalJsonProofBinding.from_dict(raw_compiled_index)
        )
        raw_request_index = row.pop("requests_shard_index", None)
        row["requests_shard_index"] = (
            None
            if raw_request_index is None
            else CanonicalJsonProofBinding.from_dict(raw_request_index)
        )
        row.setdefault("request_count", None)
        if type(requests) is not list or type(tuning_entries) is not list:
            raise TypeError("formal request schedule requests must be an array")
        return cls(
            **row,
            tts_tuning_entry_ids=tuple(tuning_entries),
            requests=tuple(
                FormalServingRequestScheduleSourceRow.from_dict(item)
                for item in requests
            ),
            e5_arrival_plan=(
                None
                if raw_e5_plan is None
                else CanonicalJsonProofBinding.from_dict(raw_e5_plan)
            ),
        )


def trusted_schedule_source_rows_artifact_id(
    source: FormalServingRequestScheduleSource,
) -> str:
    """Non-circular row identity shared by schema-5 publishers and schema 6."""

    if type(
        source
    ) is not FormalServingRequestScheduleSource or source.schema_version not in {5, 6}:
        raise ValueError("trusted schedule source row identity requires schema 5/6")
    request_count = (
        len(source.requests) if source.schema_version == 5 else source.request_count
    )
    if type(request_count) is not int or request_count < 1:
        raise ValueError("trusted schedule source request count differs")
    return _sha256(
        {
            "schema_version": 1,
            "kind": FORMAL_TRUSTED_SCHEDULE_SOURCE_ROW_SHARD_ARTIFACT_KIND,
            "subject_sha256": source.subject_sha256,
            "materialization_receipt_sha256": (source.materialization_receipt_sha256),
            "materialized_cell_id": source.materialized_cell_id,
            "workload_id": source.workload_id,
            "workload_source_descriptor_sha256": (
                source.workload_source_descriptor_sha256
            ),
            "tts_tuning_window_sha256": source.tts_tuning_window_sha256,
            "tts_tuning_entry_ids": source.tts_tuning_entry_ids,
            "sampling_profile_sha256": source.sampling_profile_sha256,
            "load_protocol_sha256": source.load_protocol_sha256,
            "context_tokens": source.context_tokens,
            "regime": source.regime,
            "arrival_policy": source.arrival_policy,
            "max_running_requests": source.max_running_requests,
            "cohort_count": source.cohort_count,
            "topology_mode": source.topology_mode,
            "tokenizer_content_member_id": source.tokenizer_content_member_id,
            "tokenizer_model_id": source.tokenizer_model_id,
            "tokenizer_revision": source.tokenizer_revision,
            "e5_arrival_plan": (
                None
                if source.e5_arrival_plan is None
                else source.e5_arrival_plan.to_dict()
            ),
            "content_source_binding_sha256": (source.content_source_binding_sha256),
            "trusted_workload_member_sha256": (source.trusted_workload_member_sha256),
            "trusted_tts_calibration_authority_sha256": (
                source.trusted_tts_calibration_authority_sha256
            ),
            "trusted_task_native_workload_sha256": (
                source.trusted_task_native_workload_sha256
            ),
            "request_count": request_count,
        }
    )


def _trusted_controlled_context_compiled_rows_artifact_id(
    *,
    uncompiled_source: CanonicalJsonProofBinding,
    context_filler_artifact: CanonicalJsonProofBinding,
    context_tokens: int,
    regime: str,
    tokenizer_content_member_id: str,
    tokenizer_model_id: str,
    tokenizer_revision: str,
    request_count: int,
) -> str:
    return _sha256(
        {
            "schema_version": 1,
            "kind": FORMAL_TRUSTED_CONTROLLED_CONTEXT_COMPILED_ROW_SHARD_ARTIFACT_KIND,
            "uncompiled_source": uncompiled_source.to_dict(),
            "context_filler_artifact": context_filler_artifact.to_dict(),
            "context_tokens": context_tokens,
            "regime": regime,
            "tokenizer_content_member_id": tokenizer_content_member_id,
            "tokenizer_model_id": tokenizer_model_id,
            "tokenizer_revision": tokenizer_revision,
            "request_count": request_count,
        }
    )


def trusted_controlled_context_compiled_rows_artifact_id(
    source: FormalServingRequestScheduleSource,
) -> str:
    """Identity for exact compiled token rows without a circular header hash."""

    if source.schema_version != 7:
        raise ValueError("compiled context row identity requires schema 7")
    assert source.controlled_context_uncompiled_source is not None
    assert source.context_filler_artifact is not None
    assert source.request_count is not None
    return _trusted_controlled_context_compiled_rows_artifact_id(
        uncompiled_source=source.controlled_context_uncompiled_source,
        context_filler_artifact=source.context_filler_artifact,
        context_tokens=source.context_tokens,
        regime=source.regime,
        tokenizer_content_member_id=source.tokenizer_content_member_id,
        tokenizer_model_id=source.tokenizer_model_id,
        tokenizer_revision=source.tokenizer_revision,
        request_count=source.request_count,
    )


def _trusted_controlled_context_source_rows_artifact_id(
    *,
    uncompiled_source: CanonicalJsonProofBinding,
    context_filler_artifact: CanonicalJsonProofBinding,
    compiled_context_requests_shard_index: CanonicalJsonProofBinding,
    subject_sha256: str,
    materialized_cell_id: str,
    topology_mode: str,
    request_count: int,
) -> str:
    return _sha256(
        {
            "schema_version": 1,
            "kind": FORMAL_TRUSTED_CONTROLLED_CONTEXT_SOURCE_ROW_SHARD_ARTIFACT_KIND,
            "uncompiled_source": uncompiled_source.to_dict(),
            "context_filler_artifact": context_filler_artifact.to_dict(),
            "compiled_context_requests_shard_index": (
                compiled_context_requests_shard_index.to_dict()
            ),
            "subject_sha256": subject_sha256,
            "materialized_cell_id": materialized_cell_id,
            "topology_mode": topology_mode,
            "request_count": request_count,
        }
    )


def trusted_controlled_context_source_rows_artifact_id(
    source: FormalServingRequestScheduleSource,
) -> str:
    """Identity for schema-7 source rows after exact token compilation."""

    if source.schema_version != 7:
        raise ValueError("controlled context source identity requires schema 7")
    assert source.controlled_context_uncompiled_source is not None
    assert source.context_filler_artifact is not None
    assert source.compiled_context_requests_shard_index is not None
    assert source.request_count is not None
    return _trusted_controlled_context_source_rows_artifact_id(
        uncompiled_source=source.controlled_context_uncompiled_source,
        context_filler_artifact=source.context_filler_artifact,
        compiled_context_requests_shard_index=(
            source.compiled_context_requests_shard_index
        ),
        subject_sha256=source.subject_sha256,
        materialized_cell_id=source.materialized_cell_id,
        topology_mode=source.topology_mode,
        request_count=source.request_count,
    )


def formal_serving_request_schedule_source_rows(
    source: FormalServingRequestScheduleSource,
    *,
    revalidate: bool = True,
) -> Iterator[FormalServingRequestScheduleSourceRow]:
    """Iterate source rows while opening at most one bounded shard at a time."""

    if type(source) is not FormalServingRequestScheduleSource:
        raise TypeError("formal schedule source rows require an exact source")
    if source.schema_version not in {6, 7}:
        yield from source.requests
        return
    assert source.requests_shard_index is not None
    assert source.request_count is not None
    index = load_formal_canonical_sequence_shard_index(
        source.requests_shard_index.absolute_path,
        deep=False,
    )
    if (
        source.requests_shard_index.semantic_sha256 != _sha256(index.to_dict())
        or index.artifact_kind
        != (
            FORMAL_TRUSTED_CONTROLLED_CONTEXT_SOURCE_ROW_SHARD_ARTIFACT_KIND
            if source.schema_version == 7
            else FORMAL_TRUSTED_SCHEDULE_SOURCE_ROW_SHARD_ARTIFACT_KIND
        )
        or index.artifact_id
        != (
            trusted_controlled_context_source_rows_artifact_id(source)
            if source.schema_version == 7
            else trusted_schedule_source_rows_artifact_id(source)
        )
        or index.total_rows != source.request_count
    ):
        raise ValueError("trusted schedule source shard index identity differs")
    seen_keys: set[str] = set()
    seen_ordinals: set[int] = set()
    expected_ordinal = 0
    scored = False
    cohort_routes: dict[str, int] = {}
    for value in index.iter_rows():
        row = FormalServingRequestScheduleSourceRow.from_dict(value)
        if revalidate:
            if (
                row.source_request_key in seen_keys
                or row.ordinal in seen_ordinals
                or row.ordinal != expected_ordinal
                or row.source_member_sha256 != source.trusted_workload_member_sha256
            ):
                raise ValueError("trusted schedule source shard row order differs")
            seen_keys.add(row.source_request_key)
            seen_ordinals.add(row.ordinal)
            expected_ordinal += 1
            scored = scored or row.phase == "scored"
            if source.topology_mode == "tp1_dp2":
                if row.routed_dp_rank not in {0, 1}:
                    raise ValueError("sharded DP2 schedule lacks exact route")
                previous = cohort_routes.setdefault(
                    row.cohort_id,
                    row.routed_dp_rank,
                )
                if previous != row.routed_dp_rank:
                    raise ValueError("sharded DP2 schedule route changed")
            elif row.routed_dp_rank is not None:
                raise ValueError("sharded non-DP2 schedule carries a route")
        yield row
    if revalidate and (
        len(seen_ordinals) != source.request_count
        or not scored
        or (source.topology_mode == "tp1_dp2" and set(cohort_routes.values()) != {0, 1})
    ):
        raise ValueError("trusted schedule source shard coverage differs")


def formal_serving_controlled_context_requests(
    source: FormalServingRequestScheduleSource,
) -> Iterator[CompiledContextRequest]:
    """Stream schema-7 compiled requests while deep-checking the shard index."""

    if source.schema_version != 7:
        raise ValueError("controlled context requests require schema 7")
    assert source.compiled_context_requests_shard_index is not None
    assert source.request_count is not None
    index = load_formal_canonical_sequence_shard_index(
        source.compiled_context_requests_shard_index.absolute_path,
        deep=False,
    )
    if (
        source.compiled_context_requests_shard_index.semantic_sha256
        != _sha256(index.to_dict())
        or index.artifact_kind
        != FORMAL_TRUSTED_CONTROLLED_CONTEXT_COMPILED_ROW_SHARD_ARTIFACT_KIND
        or index.artifact_id
        != trusted_controlled_context_compiled_rows_artifact_id(source)
        or index.total_rows != source.request_count
    ):
        raise ValueError("controlled context compiled shard index differs")
    count = 0
    for value in index.iter_rows():
        row = CompiledContextRequest.from_dict(value)
        if (
            row.context_tokens != source.context_tokens
            or row.regime != source.regime
            or row.tokenizer_content_member_id != source.tokenizer_content_member_id
            or row.tokenizer_model_id != source.tokenizer_model_id
            or row.tokenizer_revision != source.tokenizer_revision
        ):
            raise ValueError("controlled context compiled row lineage differs")
        count += 1
        yield row
    if count != source.request_count:
        raise ValueError("controlled context compiled row coverage differs")


@dataclass(frozen=True)
class FormalServingRequestScheduleRow:
    source_member_sha256: str
    source_sample_id: str
    prompt_sha256: str
    phase: Literal["warmup", "scored"]
    routed_dp_rank: int | None
    request: BoundServingRequest
    tokenized_input_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("formal schedule row source", self.source_member_sha256)
        _require_text("formal schedule row sample", self.source_sample_id)
        _require_sha256("formal schedule row prompt", self.prompt_sha256)
        _require_sha256("formal schedule row tokens", self.tokenized_input_sha256)
        if self.phase not in _PHASES or type(self.request) is not BoundServingRequest:
            raise ValueError("formal schedule row phase/request differs")
        self.request.validate()
        if self.tokenized_input_sha256 != _sha256(list(self.request.input_token_ids)):
            raise ValueError("formal schedule row token digest differs")
        if self.routed_dp_rank is not None and self.routed_dp_rank not in {0, 1}:
            raise ValueError("formal schedule row DP route differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_member_sha256": self.source_member_sha256,
            "source_sample_id": self.source_sample_id,
            "prompt_sha256": self.prompt_sha256,
            "phase": self.phase,
            "routed_dp_rank": self.routed_dp_rank,
            "request": _bound_request_to_dict(self.request),
            "tokenized_input_sha256": self.tokenized_input_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "formal serving schedule row",
            value,
            {
                "source_member_sha256",
                "source_sample_id",
                "prompt_sha256",
                "phase",
                "routed_dp_rank",
                "request",
                "tokenized_input_sha256",
            },
        )
        request = _bound_request_from_dict(row.pop("request"))
        return cls(**row, request=request)


@dataclass(frozen=True)
class FormalServingRequestScheduleReceipt:
    """Path-bound result of the only allowed formal tokenization path."""

    schema_version: Literal[3, 4, 5, 6, 7]
    kind: Literal["formal_serving_request_schedule_receipt"]
    protocol_sha256: str
    formal_execution_authorized: Literal[False]
    execution_binding_sha256: str
    subject_sha256: str
    materialized_cell_id: str
    workload_authority_sha256: str | None
    content_verification_receipt_sha256: str | None
    topology_mode: str
    materialization: CanonicalJsonProofBinding
    content_verification_receipt: CanonicalJsonProofBinding | None
    workload_source: ContentJsonArtifactBinding
    compile_launch_manifest: CanonicalJsonProofBinding
    sampling_profile: CanonicalJsonProofBinding
    schedule_source: ContentJsonArtifactBinding
    tokenization_input: CanonicalJsonProofBinding
    tokenization_output: CanonicalJsonProofBinding
    tokenizer_worker_source_raw_sha256: str
    tokenizer_worker_source_size: int
    tokenizer_worker_argv_sha256: str
    tokenizer_model_id: str
    tokenizer_revision: str
    tokenizer_snapshot_path: str
    tokenizer_content_member_id: str
    tokenizer_content_authority_sha256: str | None
    transformers_version: str
    tokenizer_class: str
    tokenizer_vocab_size: int
    requests: tuple[FormalServingRequestScheduleRow, ...]
    e5_arrival_plan: CanonicalJsonProofBinding | None = None
    content_source_binding: FormalContentSourceBinding | None = None
    trusted_workload_member_sha256: str | None = None
    trusted_tts_calibration_authority: CanonicalJsonProofBinding | None = None
    requests_shard_index: CanonicalJsonProofBinding | None = None
    request_count: int | None = None
    controlled_context_uncompiled_source: CanonicalJsonProofBinding | None = None
    context_filler_artifact: CanonicalJsonProofBinding | None = None
    compiled_context_requests_shard_index: CanonicalJsonProofBinding | None = None
    livecodebench_tokenizer_authority: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {3, 4, 5, 6, 7}
            or self.kind != "formal_serving_request_schedule_receipt"
            or self.formal_execution_authorized is not False
        ):
            raise ValueError("formal schedule receipt schema differs")
        expected_protocol = (
            (
                TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
                if self.schema_version == 7
                else (
                    TRUSTED_SINGLE_OPERATOR_SHARDED_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
                    if self.schema_version == 6
                    else TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
                )
            )
            if self.schema_version in {5, 6, 7}
            else FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        )
        if self.protocol_sha256 != expected_protocol:
            raise ValueError("formal schedule receipt protocol differs")
        if self.schema_version == 3:
            if self.e5_arrival_plan is not None:
                raise ValueError("legacy formal receipt carries an E5 arrival plan")
        elif self.e5_arrival_plan is not None and (
            type(self.e5_arrival_plan) is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(self.e5_arrival_plan.absolute_path)
            != self.e5_arrival_plan
        ):
            raise ValueError("formal receipt E5 arrival-plan binding changed")
        for label, value in (
            ("execution binding", self.execution_binding_sha256),
            ("subject", self.subject_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("tokenizer worker", self.tokenizer_worker_source_raw_sha256),
            ("tokenizer argv", self.tokenizer_worker_argv_sha256),
        ):
            _require_sha256(f"formal schedule receipt {label}", value)
        if self.schema_version in {3, 4}:
            for label, value in (
                ("workload authority", self.workload_authority_sha256),
                ("content receipt", self.content_verification_receipt_sha256),
                ("tokenizer authority", self.tokenizer_content_authority_sha256),
            ):
                _require_sha256(f"formal schedule receipt {label}", value)
            if (
                type(self.content_verification_receipt) is not CanonicalJsonProofBinding
                or self.content_source_binding is not None
                or self.trusted_workload_member_sha256 is not None
                or self.trusted_tts_calibration_authority is not None
            ):
                raise ValueError("legacy schedule receipt content lineage differs")
        else:
            if (
                self.workload_authority_sha256 is not None
                or self.content_verification_receipt_sha256 is not None
                or self.content_verification_receipt is not None
                or self.tokenizer_content_authority_sha256 is not None
                or type(self.content_source_binding) is not FormalContentSourceBinding
                or self.content_source_binding.mode != "trusted_single_operator"
            ):
                raise ValueError(
                    "trusted schedule receipt carries offline authorization lineage"
                )
            _require_sha256(
                "trusted schedule receipt workload member",
                self.trusted_workload_member_sha256,
            )
            self.content_source_binding.reopen()
            if self.trusted_tts_calibration_authority is not None and (
                type(self.trusted_tts_calibration_authority)
                is not CanonicalJsonProofBinding
                or CanonicalJsonProofBinding.bind(
                    self.trusted_tts_calibration_authority.absolute_path
                )
                != self.trusted_tts_calibration_authority
            ):
                raise ValueError("trusted TTS authority binding changed")
        context_bindings = (
            self.controlled_context_uncompiled_source,
            self.context_filler_artifact,
            self.compiled_context_requests_shard_index,
        )
        if self.schema_version == 7:
            if any(
                type(value) is not CanonicalJsonProofBinding
                for value in context_bindings
            ):
                raise TypeError("controlled-context receipt binding differs")
            for binding in context_bindings:
                assert isinstance(binding, CanonicalJsonProofBinding)
                if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                    raise ValueError("controlled-context receipt binding changed")
        elif any(binding is not None for binding in context_bindings):
            raise ValueError("non-context receipt carries context compilation")
        if self.livecodebench_tokenizer_authority is not None and (
            type(self.livecodebench_tokenizer_authority)
            is not CanonicalJsonProofBinding
            or CanonicalJsonProofBinding.bind(
                self.livecodebench_tokenizer_authority.absolute_path
            )
            != self.livecodebench_tokenizer_authority
        ):
            raise ValueError("LiveCodeBench tokenizer authority binding changed")
        if self.topology_mode not in _TOPOLOGIES:
            raise ValueError("formal schedule receipt topology differs")
        if (
            type(self.workload_source) is not ContentJsonArtifactBinding
            or type(self.schedule_source) is not ContentJsonArtifactBinding
            or any(
                type(value) is not CanonicalJsonProofBinding
                for value in (
                    self.materialization,
                    self.compile_launch_manifest,
                    self.sampling_profile,
                    self.tokenization_input,
                    self.tokenization_output,
                )
            )
        ):
            raise TypeError("formal schedule receipt path binding differs")
        for label, value in (
            ("tokenizer model", self.tokenizer_model_id),
            ("tokenizer revision", self.tokenizer_revision),
            ("tokenizer snapshot", self.tokenizer_snapshot_path),
            ("tokenizer member", self.tokenizer_content_member_id),
            ("transformers version", self.transformers_version),
            ("tokenizer class", self.tokenizer_class),
        ):
            _require_text(f"formal schedule receipt {label}", value)
        tokenizer_path = Path(self.tokenizer_snapshot_path)
        if (
            not tokenizer_path.is_absolute()
            or tokenizer_path != tokenizer_path.resolve(strict=False)
            or tokenizer_path.name != self.tokenizer_revision
            or type(self.tokenizer_worker_source_size) is not int
            or self.tokenizer_worker_source_size < 1
            or type(self.tokenizer_vocab_size) is not int
            or self.tokenizer_vocab_size < 1
            or type(self.requests) is not tuple
            or any(
                type(row) is not FormalServingRequestScheduleRow
                for row in self.requests
            )
        ):
            raise ValueError("formal schedule receipt tokenizer/request shape differs")
        if self.schema_version in {6, 7}:
            if (
                self.requests
                or type(self.requests_shard_index) is not CanonicalJsonProofBinding
                or type(self.request_count) is not int
                or not 1 <= self.request_count <= _MAX_REQUEST_ROWS
            ):
                raise ValueError("sharded schedule receipt shape differs")
            return
        if (
            self.requests_shard_index is not None
            or self.request_count is not None
            or not self.requests
        ):
            raise ValueError("formal schedule receipt request coverage differs")
        request_ids = tuple(row.request.request_id for row in self.requests)
        if len(request_ids) != len(set(request_ids)) or not any(
            row.phase == "scored" for row in self.requests
        ):
            raise ValueError("formal schedule receipt request coverage differs")
        for row in self.requests:
            row.__post_init__()
            if self.topology_mode == "tp1_dp2":
                expected_route = f"dp{row.routed_dp_rank}"
                if (
                    row.routed_dp_rank not in {0, 1}
                    or row.request.route_id != expected_route
                ):
                    raise ValueError("DP2 formal schedule route binding differs")
            elif row.routed_dp_rank is not None:
                raise ValueError("non-DP2 formal receipt carries a replica route")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "formal_execution_authorized": self.formal_execution_authorized,
            "execution_binding_sha256": self.execution_binding_sha256,
            "subject_sha256": self.subject_sha256,
            "materialized_cell_id": self.materialized_cell_id,
            "workload_authority_sha256": self.workload_authority_sha256,
            "content_verification_receipt_sha256": (
                self.content_verification_receipt_sha256
            ),
            "topology_mode": self.topology_mode,
            "materialization": self.materialization.to_dict(),
            "content_verification_receipt": (
                None
                if self.content_verification_receipt is None
                else self.content_verification_receipt.to_dict()
            ),
            "workload_source": self.workload_source.to_dict(),
            "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
            "sampling_profile": self.sampling_profile.to_dict(),
            "schedule_source": self.schedule_source.to_dict(),
            "tokenization_input": self.tokenization_input.to_dict(),
            "tokenization_output": self.tokenization_output.to_dict(),
            "tokenizer_worker_source_raw_sha256": (
                self.tokenizer_worker_source_raw_sha256
            ),
            "tokenizer_worker_source_size": self.tokenizer_worker_source_size,
            "tokenizer_worker_argv_sha256": self.tokenizer_worker_argv_sha256,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_snapshot_path": self.tokenizer_snapshot_path,
            "tokenizer_content_member_id": self.tokenizer_content_member_id,
            "tokenizer_content_authority_sha256": (
                self.tokenizer_content_authority_sha256
            ),
            "transformers_version": self.transformers_version,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
        }
        if self.schema_version in {6, 7}:
            assert self.requests_shard_index is not None
            value["requests_shard_index"] = self.requests_shard_index.to_dict()
            value["request_count"] = self.request_count
        else:
            value["requests"] = [row.to_dict() for row in self.requests]
        if self.schema_version in {4, 5, 6, 7}:
            value["e5_arrival_plan"] = (
                None if self.e5_arrival_plan is None else self.e5_arrival_plan.to_dict()
            )
        if self.schema_version in {5, 6, 7}:
            assert self.content_source_binding is not None
            value["content_source_binding"] = self.content_source_binding.to_dict()
            value["trusted_workload_member_sha256"] = (
                self.trusted_workload_member_sha256
            )
            value["trusted_tts_calibration_authority"] = (
                None
                if self.trusted_tts_calibration_authority is None
                else self.trusted_tts_calibration_authority.to_dict()
            )
        if self.schema_version == 7:
            assert self.controlled_context_uncompiled_source is not None
            assert self.context_filler_artifact is not None
            assert self.compiled_context_requests_shard_index is not None
            value["controlled_context_uncompiled_source"] = (
                self.controlled_context_uncompiled_source.to_dict()
            )
            value["context_filler_artifact"] = self.context_filler_artifact.to_dict()
            value["compiled_context_requests_shard_index"] = (
                self.compiled_context_requests_shard_index.to_dict()
            )
        if self.livecodebench_tokenizer_authority is not None:
            value["livecodebench_tokenizer_authority"] = (
                self.livecodebench_tokenizer_authority.to_dict()
            )
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("formal serving request schedule receipt must be an object")
        schema_version = value.get("schema_version")
        expected = {field.name for field in fields(cls)}
        if "livecodebench_tokenizer_authority" not in value:
            expected.remove("livecodebench_tokenizer_authority")
        if schema_version == 3:
            expected.remove("e5_arrival_plan")
        if schema_version in {3, 4}:
            expected -= {
                "content_source_binding",
                "trusted_workload_member_sha256",
                "trusted_tts_calibration_authority",
            }
        if schema_version in {3, 4, 5, 6}:
            expected -= {
                "controlled_context_uncompiled_source",
                "context_filler_artifact",
                "compiled_context_requests_shard_index",
            }
        if schema_version in {3, 4, 5}:
            expected -= {"requests_shard_index", "request_count"}
        elif schema_version in {6, 7}:
            expected.remove("requests")
        row = _strict_object(
            "formal serving request schedule receipt",
            value,
            expected,
        )
        raw_e5_plan = row.pop("e5_arrival_plan", None)
        materialization = CanonicalJsonProofBinding.from_dict(
            row.pop("materialization")
        )
        raw_content_receipt = row.pop("content_verification_receipt")
        content_receipt = (
            None
            if raw_content_receipt is None
            else CanonicalJsonProofBinding.from_dict(raw_content_receipt)
        )
        raw_content_source = row.pop("content_source_binding", None)
        content_source = (
            None
            if raw_content_source is None
            else FormalContentSourceBinding.from_dict(raw_content_source)
        )
        row.setdefault("trusted_workload_member_sha256", None)
        raw_tts_authority = row.pop("trusted_tts_calibration_authority", None)
        raw_uncompiled_source = row.pop(
            "controlled_context_uncompiled_source",
            None,
        )
        raw_filler_artifact = row.pop("context_filler_artifact", None)
        raw_compiled_index = row.pop(
            "compiled_context_requests_shard_index",
            None,
        )
        raw_livecodebench_tokenizer_authority = row.pop(
            "livecodebench_tokenizer_authority",
            None,
        )
        row["controlled_context_uncompiled_source"] = (
            None
            if raw_uncompiled_source is None
            else CanonicalJsonProofBinding.from_dict(raw_uncompiled_source)
        )
        row["context_filler_artifact"] = (
            None
            if raw_filler_artifact is None
            else CanonicalJsonProofBinding.from_dict(raw_filler_artifact)
        )
        row["compiled_context_requests_shard_index"] = (
            None
            if raw_compiled_index is None
            else CanonicalJsonProofBinding.from_dict(raw_compiled_index)
        )
        row["livecodebench_tokenizer_authority"] = (
            None
            if raw_livecodebench_tokenizer_authority is None
            else CanonicalJsonProofBinding.from_dict(
                raw_livecodebench_tokenizer_authority
            )
        )
        workload_source = ContentJsonArtifactBinding.from_dict(
            row.pop("workload_source")
        )
        compile_launch_manifest = CanonicalJsonProofBinding.from_dict(
            row.pop("compile_launch_manifest")
        )
        sampling_profile = CanonicalJsonProofBinding.from_dict(
            row.pop("sampling_profile")
        )
        source = ContentJsonArtifactBinding.from_dict(row.pop("schedule_source"))
        token_input = CanonicalJsonProofBinding.from_dict(row.pop("tokenization_input"))
        token_output = CanonicalJsonProofBinding.from_dict(
            row.pop("tokenization_output")
        )
        requests = row.pop("requests", [])
        raw_request_index = row.pop("requests_shard_index", None)
        row["requests_shard_index"] = (
            None
            if raw_request_index is None
            else CanonicalJsonProofBinding.from_dict(raw_request_index)
        )
        row.setdefault("request_count", None)
        if type(requests) is not list:
            raise TypeError("formal schedule receipt requests must be an array")
        return cls(
            **row,
            materialization=materialization,
            content_verification_receipt=content_receipt,
            content_source_binding=content_source,
            trusted_tts_calibration_authority=(
                None
                if raw_tts_authority is None
                else CanonicalJsonProofBinding.from_dict(raw_tts_authority)
            ),
            workload_source=workload_source,
            compile_launch_manifest=compile_launch_manifest,
            sampling_profile=sampling_profile,
            schedule_source=source,
            tokenization_input=token_input,
            tokenization_output=token_output,
            requests=tuple(
                FormalServingRequestScheduleRow.from_dict(item) for item in requests
            ),
            e5_arrival_plan=(
                None
                if raw_e5_plan is None
                else CanonicalJsonProofBinding.from_dict(raw_e5_plan)
            ),
        )

    def reopen(self) -> None:
        source = FormalServingRequestScheduleSource.from_dict(
            self.schedule_source.load()
        )
        materialization = _reopen_stage_materialization(self.materialization)
        content: ContentVerificationReceipt | None = None
        authorization: VerifiedReleaseWorkloadSources | None = None
        workload: FormalWorkloadAuthority | None = None
        e0_workload: object | None = None
        trusted_bundle: object | None = None
        trusted_tts_window: TtsCalibrationTuningWindow | None = None
        trusted_tts_authority_sha256: str | None = None
        descriptor_sha256: str
        if self.schema_version in {3, 4}:
            assert self.content_verification_receipt is not None
            assert self.content_verification_receipt_sha256 is not None
            content = ContentVerificationReceipt.from_dict(
                self.content_verification_receipt.reopen()
            )
            if content.sha256 != self.content_verification_receipt_sha256:
                raise ValueError("formal schedule content receipt identity changed")
            _reject_master_derived_schedule_artifacts(content)
            authorization = _verified_workload_authorization(
                content,
                current_ns=content.verified_ns,
            )
            expected_workload_artifact_id = formal_workload_authority_artifact_id(
                source.workload_id  # type: ignore[arg-type]
            )
            if self.workload_source.artifact_id != expected_workload_artifact_id:
                raise ValueError(
                    "formal schedule workload binding names another source"
                )
            workload = formal_workload_authority_from_cli_artifact(
                self.workload_source.load()
            )
            workload = revalidate_authorized_formal_workload_authority(
                workload,
                authorization=authorization,
            )
            descriptor_sha256 = authorization.source(workload.workload_id).sha256
        else:
            from lightcone_spec.experiments.formal_single_operator_content import (
                TrustedSingleOperatorContentBundle,
            )

            assert self.content_source_binding is not None
            assert self.trusted_workload_member_sha256 is not None
            trusted_bundle = self.content_source_binding.reopen()
            if (
                type(trusted_bundle) is not TrustedSingleOperatorContentBundle
                or source.content_source_binding_sha256
                != self.content_source_binding.sha256
                or source.trusted_workload_member_sha256
                != self.trusted_workload_member_sha256
            ):
                raise ValueError("trusted schedule content source identity changed")
            if source.workload_id in {
                "livecodebench_v6_hard",
                "math500_level5",
            }:
                members = tuple(
                    row
                    for row in trusted_bundle.locked_workloads
                    if row.workload_id == source.workload_id
                    and row.sha256 == self.trusted_workload_member_sha256
                )
                if len(members) != 1:
                    raise ValueError("trusted schedule locked workload is not exact")
                member = members[0]
                if (
                    self.workload_source.artifact_id
                    != formal_workload_authority_artifact_id(member.workload_id)
                ):
                    raise ValueError("trusted schedule workload artifact ID differs")
                workload = formal_workload_authority_from_cli_artifact(
                    self.workload_source.load()
                )
                if (
                    workload.sha256 != member.authority_sha256
                    or workload.workload_id != member.workload_id
                    or workload.raw_source_path != member.raw_source_path
                    or workload.raw_file_sha256 != member.raw_file_sha256
                    or workload.repository_revision != member.repository_revision
                    or workload.raw_row_count != member.raw_row_count
                    or workload.selected_row_count != member.selected_row_count
                    or workload.selected_rows_sha256 != member.formal_samples_sha256
                    or workload.source_lock_sha256 != member.source_lock_sha256
                    or workload.protocol_sha256 != member.protocol_sha256
                ):
                    raise ValueError("trusted schedule workload authority differs")
                descriptor_sha256 = member.sha256
            else:
                from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
                    E0TaskNativeSourceAuthority,
                    load_e0_task_native_source_authority,
                )

                members = tuple(
                    row
                    for row in trusted_bundle.e0_task_native_descriptors
                    if row.task == source.workload_id
                    and row.sha256 == self.trusted_workload_member_sha256
                )
                if len(members) != 1:
                    raise ValueError("trusted E0 workload descriptor is not exact")
                member = members[0]
                if self.workload_source.artifact_id != (
                    f"trusted_e0_task_native:{source.workload_id}"
                ):
                    raise ValueError("trusted E0 workload artifact ID differs")
                if (
                    self.workload_source.path != member.source.absolute_path
                    or self.workload_source.raw_sha256 != member.source.raw_sha256
                    or self.workload_source.semantic_sha256
                    != member.source.semantic_sha256
                    or self.workload_source.size != member.source.size
                ):
                    raise ValueError("trusted E0 workload path binding differs")
                e0_workload = load_e0_task_native_source_authority(
                    self.workload_source.path
                )
                if (
                    type(e0_workload) is not E0TaskNativeSourceAuthority
                    or e0_workload.task != source.workload_id
                    or e0_workload.support_status != "READY"
                ):
                    raise ValueError("trusted E0 workload is not serving-ready")
                descriptor_sha256 = member.sha256
            if source.trusted_tts_calibration_authority_sha256 is None:
                if self.trusted_tts_calibration_authority is not None:
                    raise ValueError("non-TTS trusted schedule carries TTS authority")
            else:
                from lightcone_spec.experiments.formal_method_authority import (
                    load_tts_calibration_authority_artifact,
                )

                if self.trusted_tts_calibration_authority is None:
                    raise ValueError("trusted TTS schedule lacks authority binding")
                tts_artifact = load_tts_calibration_authority_artifact(
                    self.trusted_tts_calibration_authority.absolute_path
                )
                window = TtsCalibrationTuningWindow.from_dict(
                    tts_artifact.tuning_window_source.reopen()
                )
                if (
                    self.trusted_tts_calibration_authority.semantic_sha256
                    != tts_artifact.sha256
                    or tts_artifact.authority.sha256
                    != source.trusted_tts_calibration_authority_sha256
                    or window.sha256 != source.tts_tuning_window_sha256
                    or tuple(sorted(row.entry_id for row in window.tuning_entries))
                    != source.tts_tuning_entry_ids
                ):
                    raise ValueError("trusted TTS schedule authority changed")
                trusted_tts_window = window
                trusted_tts_authority_sha256 = tts_artifact.authority.sha256
        launch = CompileLaunchManifest.load(self.compile_launch_manifest.absolute_path)
        if (
            CanonicalJsonProofBinding.bind(
                self.compile_launch_manifest.absolute_path,
                semantic_sha256=launch.sha256,
            )
            != self.compile_launch_manifest
        ):
            raise ValueError("formal schedule launch manifest identity changed")
        config = load_run_config(launch.run_config_path)
        sampling = SamplingProfile.load(self.sampling_profile.absolute_path)
        if (
            CanonicalJsonProofBinding.bind(self.sampling_profile.absolute_path)
            != self.sampling_profile
            or sampling.sha256 != self.sampling_profile.semantic_sha256
        ):
            raise ValueError("formal schedule sampling profile identity changed")
        cell = _materialized_cell(
            materialization,
            cell_id=source.materialized_cell_id,
        )
        source_rows = tuple(formal_serving_request_schedule_source_rows(source))
        expected_controlled_rows: tuple[FormalServingRequestScheduleRow, ...] | None = (
            None
        )
        if self.schema_version in {5, 6, 7}:
            trusted_workload = workload if workload is not None else e0_workload
            if trusted_workload is None or self.content_source_binding is None:
                raise ValueError(
                    "trusted schedule exact workload replay is unavailable"
                )
            rebuilt = rebuild_trusted_single_operator_request_schedule_source(
                subject_sha256=source.subject_sha256,
                content_source_binding=self.content_source_binding,
                topology_mode=source.topology_mode,
                materialization=materialization,
                materialized_cell_id=source.materialized_cell_id,
                workload_source=trusted_workload,
                workload_source_binding=self.workload_source,
                sampling_profile=sampling,
                max_running_requests=source.max_running_requests,
                server_context_limit=config.runtime.context_length,
                tokenizer_content_member_id=source.tokenizer_content_member_id,
                tokenizer_model_id=source.tokenizer_model_id,
                tokenizer_revision=source.tokenizer_revision,
                tts_tuning_window=trusted_tts_window,
                trusted_tts_calibration_authority_sha256=(trusted_tts_authority_sha256),
                e5_arrival_plan=source.e5_arrival_plan,
            )
            if source.schema_version == 7:
                if (
                    self.controlled_context_uncompiled_source
                    != source.controlled_context_uncompiled_source
                    or self.context_filler_artifact != source.context_filler_artifact
                    or self.compiled_context_requests_shard_index
                    != source.compiled_context_requests_shard_index
                    or source.controlled_context_uncompiled_source is None
                    or source.context_filler_artifact is None
                    or source.compiled_context_requests_shard_index is None
                    or FormalServingRequestScheduleSource.from_dict(
                        source.controlled_context_uncompiled_source.reopen()
                    )
                    != rebuilt
                ):
                    raise ValueError("controlled context uncompiled source changed")
                adjusted, expected_compiled, _authority = (
                    _compile_trusted_controlled_context_source(
                        source=rebuilt,
                        content_source_binding=self.content_source_binding,
                        context_filler_artifact=source.context_filler_artifact,
                    )
                )
                if (
                    source_rows != adjusted.requests
                    or tuple(formal_serving_controlled_context_requests(source))
                    != expected_compiled
                ):
                    raise ValueError("controlled context compiler replay changed")
                expected_source = replace(
                    adjusted,
                    schema_version=7,
                    protocol_sha256=(
                        TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
                    ),
                    derivation_protocol_sha256=(
                        TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
                    ),
                    requests=(),
                    requests_shard_index=source.requests_shard_index,
                    request_count=source.request_count,
                    controlled_context_uncompiled_source=(
                        source.controlled_context_uncompiled_source
                    ),
                    context_filler_artifact=source.context_filler_artifact,
                    compiled_context_requests_shard_index=(
                        source.compiled_context_requests_shard_index
                    ),
                )
                if source != expected_source:
                    raise ValueError("controlled context source header changed")
                expected_controlled_rows = _materialized_controlled_context_rows(
                    source=source,
                    compiled=expected_compiled,
                )
            else:
                normalized_source = (
                    source
                    if source.schema_version == 5
                    else replace(
                        source,
                        schema_version=5,
                        protocol_sha256=(
                            TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
                        ),
                        derivation_protocol_sha256=(
                            TRUSTED_SINGLE_OPERATOR_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
                        ),
                        requests=source_rows,
                        requests_shard_index=None,
                        request_count=None,
                    )
                )
                if rebuilt != normalized_source:
                    raise ValueError(
                        "trusted schedule differs from exact source reducer"
                    )
        if (
            self.schema_version in {5, 6, 7}
            and cell.stage == "E0"
            and (
                source.trusted_task_native_workload_sha256
                != dict(cell.dimensions).get("task_native_workload_sha256")
            )
        ):
            raise ValueError("trusted E0 compatibility workload identity changed")
        tts_window = (
            _tts_calibration_window(content)
            if content is not None and cell.stage == "TTS-Cal"
            else None
        )
        if self.schema_version in {3, 4}:
            assert workload is not None
            revalidate_formal_serving_request_schedule_source(
                source,
                materialization=materialization,
                workload_source=workload,
                workload_source_descriptor_sha256=descriptor_sha256,
                tts_tuning_window=tts_window,
                sampling_profile=sampling,
                server_context_limit=config.runtime.context_length,
            )
        elif workload is not None:
            if source.workload_id != workload.workload_id or tuple(
                row.source_member_sha256 for row in source_rows
            ) != (descriptor_sha256,) * len(source_rows):
                raise ValueError("trusted schedule workload rows differ")
        else:
            assert e0_workload is not None
            e0_rows = {row.source_id: row for row in e0_workload.request_rows}
            if (
                cell.stage != "E0"
                or source.workload_id != _workload_id_for_cell(cell)
                or {row.source_sample_id for row in source_rows} - set(e0_rows)
            ):
                raise ValueError("trusted E0 schedule rows differ")
            for row in source_rows:
                expected = e0_rows[row.source_sample_id]
                if (
                    row.source_member_sha256 != descriptor_sha256
                    or row.source_raw_file_sha256 != e0_workload.raw_file_sha256
                    or row.source_selected_rows_sha256
                    != e0_workload.request_rows_sha256
                    or row.prompt != expected.turns[0]
                    or row.prompt_sha256 != _sha256(expected.turns[0])
                    or dict(row.sampling).get("seed") != expected.seed
                    or source.trusted_task_native_workload_sha256
                    != e0_workload.task_native_workload_sha256
                ):
                    raise ValueError("trusted E0 request row changed")
        if self.schema_version == 7:
            assert source.context_filler_artifact is not None
            assert source.compiled_context_requests_shard_index is not None
            if (
                self.tokenization_input != source.context_filler_artifact
                or self.tokenization_output
                != source.compiled_context_requests_shard_index
                or expected_controlled_rows is None
                or tuple(formal_serving_request_schedule_rows(self))
                != expected_controlled_rows
            ):
                raise ValueError("controlled context receipt rows changed")
            (
                worker_sha,
                worker_size,
                argv_sha,
                transformers_version,
                tokenizer_class,
                tokenizer_vocab_size,
                tokenizer_snapshot_path,
            ) = _controlled_context_tokenizer_metadata(source.context_filler_artifact)
            if (
                self.tokenizer_worker_source_raw_sha256 != worker_sha
                or self.tokenizer_worker_source_size != worker_size
                or self.tokenizer_worker_argv_sha256 != argv_sha
                or self.transformers_version != transformers_version
                or self.tokenizer_class != tokenizer_class
                or self.tokenizer_vocab_size != tokenizer_vocab_size
                or self.tokenizer_snapshot_path != tokenizer_snapshot_path
            ):
                raise ValueError("controlled context tokenizer metadata changed")
            token_input: dict[str, object] = {"schedule_source_sha256": source.sha256}
            token_output: dict[str, object] = {
                "schedule_source_sha256": source.sha256,
                "requests": True,
            }
        elif self.schema_version == 6:
            _reopen_sharded_tokenization(
                receipt=self,
                source=source,
                source_rows=source_rows,
                launch=launch,
            )
            token_input: dict[str, object] = {"schedule_source_sha256": source.sha256}
            token_output: dict[str, object] = {
                "schedule_source_sha256": source.sha256,
                "requests": True,
            }
        else:
            token_input = self.tokenization_input.reopen()
            token_output = self.tokenization_output.reopen()
        if (
            source.sha256 != self.schedule_source.semantic_sha256
            or source.e5_arrival_plan != self.e5_arrival_plan
            or materialization.sha256 != source.materialization_receipt_sha256
            or (
                self.schema_version in {3, 4}
                and (
                    workload is None
                    or workload.sha256 != source.workload_source_authority_sha256
                )
            )
            or (
                self.schema_version in {5, 6, 7}
                and source.workload_source_descriptor_sha256 != descriptor_sha256
            )
            or sampling.sha256 != source.sampling_profile_sha256
            or self.sampling_profile.absolute_path != launch.sampling_profile_path
            or sampling.sha256 != launch.sampling_profile_sha256
            or source.max_running_requests != config.runtime.max_running_requests
            or source.topology_mode != config.runtime.topology_mode
            or self.tokenizer_model_id != launch.tokenizer_model_id
            or self.tokenizer_revision != launch.tokenizer_revision
            or self.tokenizer_content_member_id != launch.tokenizer_content_member_id
            or self.tokenizer_content_authority_sha256
            != launch.tokenizer_content_authority_sha256
            or token_input.get("schedule_source_sha256") != source.sha256
            or token_output.get("schedule_source_sha256") != source.sha256
            or token_output.get("requests") is None
        ):
            raise ValueError("formal schedule receipt path identity changed")
        _revalidate_livecodebench_e1_e2_tokenizer_authority(
            receipt=self,
            stage=cell.stage,
            bundle=trusted_bundle,
            workload=workload,
            source=source,
        )


def trusted_schedule_receipt_rows_artifact_id(
    receipt: FormalServingRequestScheduleReceipt,
) -> str:
    """Return the non-circular identity of trusted tokenized request rows."""

    if type(
        receipt
    ) is not FormalServingRequestScheduleReceipt or receipt.schema_version not in {
        5,
        6,
    }:
        raise ValueError("trusted receipt row identity requires schema 5/6")
    request_count = (
        len(receipt.requests) if receipt.schema_version == 5 else receipt.request_count
    )
    if type(request_count) is not int or request_count < 1:
        raise ValueError("trusted receipt request count differs")
    return _sha256(
        {
            "schema_version": 1,
            "kind": FORMAL_TRUSTED_SCHEDULE_RECEIPT_ROW_SHARD_ARTIFACT_KIND,
            "execution_binding_sha256": receipt.execution_binding_sha256,
            "subject_sha256": receipt.subject_sha256,
            "materialized_cell_id": receipt.materialized_cell_id,
            "topology_mode": receipt.topology_mode,
            "materialization": receipt.materialization.to_dict(),
            "workload_source": receipt.workload_source.to_dict(),
            "compile_launch_manifest": receipt.compile_launch_manifest.to_dict(),
            "sampling_profile": receipt.sampling_profile.to_dict(),
            "schedule_source": receipt.schedule_source.to_dict(),
            "tokenization_input": receipt.tokenization_input.to_dict(),
            "tokenization_output": receipt.tokenization_output.to_dict(),
            "tokenizer_worker_source_raw_sha256": (
                receipt.tokenizer_worker_source_raw_sha256
            ),
            "tokenizer_worker_source_size": receipt.tokenizer_worker_source_size,
            "tokenizer_worker_argv_sha256": receipt.tokenizer_worker_argv_sha256,
            "tokenizer_model_id": receipt.tokenizer_model_id,
            "tokenizer_revision": receipt.tokenizer_revision,
            "tokenizer_snapshot_path": receipt.tokenizer_snapshot_path,
            "tokenizer_content_member_id": receipt.tokenizer_content_member_id,
            "transformers_version": receipt.transformers_version,
            "tokenizer_class": receipt.tokenizer_class,
            "tokenizer_vocab_size": receipt.tokenizer_vocab_size,
            "e5_arrival_plan": (
                None
                if receipt.e5_arrival_plan is None
                else receipt.e5_arrival_plan.to_dict()
            ),
            "content_source_binding": (
                None
                if receipt.content_source_binding is None
                else receipt.content_source_binding.to_dict()
            ),
            "trusted_workload_member_sha256": (receipt.trusted_workload_member_sha256),
            "trusted_tts_calibration_authority": (
                None
                if receipt.trusted_tts_calibration_authority is None
                else receipt.trusted_tts_calibration_authority.to_dict()
            ),
            "request_count": request_count,
        }
    )


def _trusted_controlled_context_receipt_rows_artifact_id(
    *,
    execution_binding_sha256: str,
    subject_sha256: str,
    materialized_cell_id: str,
    schedule_source: ContentJsonArtifactBinding,
    uncompiled_source: CanonicalJsonProofBinding,
    context_filler_artifact: CanonicalJsonProofBinding,
    compiled_context_requests_shard_index: CanonicalJsonProofBinding,
    topology_mode: str,
    request_count: int,
) -> str:
    return _sha256(
        {
            "schema_version": 1,
            "kind": FORMAL_TRUSTED_CONTROLLED_CONTEXT_RECEIPT_ROW_SHARD_ARTIFACT_KIND,
            "execution_binding_sha256": execution_binding_sha256,
            "subject_sha256": subject_sha256,
            "materialized_cell_id": materialized_cell_id,
            "schedule_source": schedule_source.to_dict(),
            "controlled_context_uncompiled_source": (uncompiled_source.to_dict()),
            "context_filler_artifact": context_filler_artifact.to_dict(),
            "compiled_context_requests_shard_index": (
                compiled_context_requests_shard_index.to_dict()
            ),
            "topology_mode": topology_mode,
            "request_count": request_count,
        }
    )


def trusted_controlled_context_receipt_rows_artifact_id(
    receipt: FormalServingRequestScheduleReceipt,
) -> str:
    """Identity for schema-7 tokenized rows without a circular receipt hash."""

    if receipt.schema_version != 7:
        raise ValueError("controlled context receipt identity requires schema 7")
    assert receipt.controlled_context_uncompiled_source is not None
    assert receipt.context_filler_artifact is not None
    assert receipt.compiled_context_requests_shard_index is not None
    assert receipt.request_count is not None
    return _trusted_controlled_context_receipt_rows_artifact_id(
        execution_binding_sha256=receipt.execution_binding_sha256,
        subject_sha256=receipt.subject_sha256,
        materialized_cell_id=receipt.materialized_cell_id,
        schedule_source=receipt.schedule_source,
        uncompiled_source=receipt.controlled_context_uncompiled_source,
        context_filler_artifact=receipt.context_filler_artifact,
        compiled_context_requests_shard_index=(
            receipt.compiled_context_requests_shard_index
        ),
        topology_mode=receipt.topology_mode,
        request_count=receipt.request_count,
    )


def formal_serving_request_schedule_rows(
    receipt: FormalServingRequestScheduleReceipt,
    *,
    revalidate: bool = True,
) -> Iterator[FormalServingRequestScheduleRow]:
    """Iterate tokenized requests with one bounded shard resident at a time."""

    if type(receipt) is not FormalServingRequestScheduleReceipt:
        raise TypeError("formal schedule rows require an exact receipt")
    if receipt.schema_version not in {6, 7}:
        yield from receipt.requests
        return
    assert receipt.requests_shard_index is not None
    assert receipt.request_count is not None
    index = load_formal_canonical_sequence_shard_index(
        receipt.requests_shard_index.absolute_path,
        deep=False,
    )
    if (
        receipt.requests_shard_index.semantic_sha256 != _sha256(index.to_dict())
        or index.artifact_kind
        != (
            FORMAL_TRUSTED_CONTROLLED_CONTEXT_RECEIPT_ROW_SHARD_ARTIFACT_KIND
            if receipt.schema_version == 7
            else FORMAL_TRUSTED_SCHEDULE_RECEIPT_ROW_SHARD_ARTIFACT_KIND
        )
        or index.artifact_id
        != (
            trusted_controlled_context_receipt_rows_artifact_id(receipt)
            if receipt.schema_version == 7
            else trusted_schedule_receipt_rows_artifact_id(receipt)
        )
        or index.total_rows != receipt.request_count
    ):
        raise ValueError("trusted receipt shard index identity differs")
    request_ids: set[str] = set()
    scored = False
    for value in index.iter_rows():
        row = FormalServingRequestScheduleRow.from_dict(value)
        if revalidate:
            if row.request.request_id in request_ids:
                raise ValueError("trusted receipt shard repeats a request")
            request_ids.add(row.request.request_id)
            scored = scored or row.phase == "scored"
            if receipt.topology_mode == "tp1_dp2":
                if (
                    row.routed_dp_rank not in {0, 1}
                    or row.request.route_id != f"dp{row.routed_dp_rank}"
                ):
                    raise ValueError("sharded DP2 receipt route differs")
            elif row.routed_dp_rank is not None:
                raise ValueError("sharded non-DP2 receipt carries a route")
        yield row
    if revalidate and (len(request_ids) != receipt.request_count or not scored):
        raise ValueError("trusted receipt shard coverage differs")


def _reopen_sharded_tokenization(
    *,
    receipt: FormalServingRequestScheduleReceipt,
    source: FormalServingRequestScheduleSource,
    source_rows: tuple[FormalServingRequestScheduleSourceRow, ...],
    launch: CompileLaunchManifest,
) -> None:
    if receipt.schema_version != 6 or source.schema_version != 6:
        raise ValueError("sharded tokenization replay requires schema 6")

    def index_rows(
        binding: CanonicalJsonProofBinding,
        *,
        kind: str,
    ) -> tuple[dict[str, object], ...]:
        value = _strict_object(
            "formal sharded tokenization index",
            binding.reopen(),
            {
                "schema_version",
                "kind",
                "protocol_sha256",
                "schedule_source_sha256",
                "batch_count",
                "batches",
            },
        )
        batches = value["batches"]
        if (
            value["schema_version"] != 1
            or value["kind"] != kind
            or value["protocol_sha256"]
            != TRUSTED_SINGLE_OPERATOR_SHARDED_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
            or value["schedule_source_sha256"] != source.sha256
            or type(batches) is not list
            or not batches
            or value["batch_count"] != len(batches)
            or any(type(row) is not dict for row in batches)
        ):
            raise ValueError("formal sharded tokenization index identity differs")
        return tuple(dict(row) for row in batches)

    input_rows = index_rows(
        receipt.tokenization_input,
        kind="formal_serving_tokenization_input_index",
    )
    output_rows = index_rows(
        receipt.tokenization_output,
        kind="formal_serving_tokenization_output_index",
    )
    if len(input_rows) != len(output_rows):
        raise ValueError("formal sharded tokenization batch coverage differs")
    receipt_rows = iter(formal_serving_request_schedule_rows(receipt))
    argv_sha256s: list[str] = []
    cursor = 0
    for ordinal, (input_row, output_row) in enumerate(
        zip(input_rows, output_rows, strict=True)
    ):
        input_value = _strict_object(
            "formal tokenization input batch",
            input_row,
            {
                "batch_ordinal",
                "start_ordinal",
                "end_ordinal_exclusive",
                "tokenization_input",
            },
        )
        output_value = _strict_object(
            "formal tokenization output batch",
            output_row,
            {
                "batch_ordinal",
                "start_ordinal",
                "end_ordinal_exclusive",
                "tokenization_output",
                "tokenizer_worker_argv_sha256",
            },
        )
        interval = (
            input_value["batch_ordinal"],
            input_value["start_ordinal"],
            input_value["end_ordinal_exclusive"],
        )
        if (
            interval
            != (
                output_value["batch_ordinal"],
                output_value["start_ordinal"],
                output_value["end_ordinal_exclusive"],
            )
            or interval[0] != ordinal
            or interval[1] != cursor
            or type(interval[2]) is not int
            or interval[2] <= cursor
            or interval[2] > len(source_rows)
        ):
            raise ValueError("formal sharded tokenization intervals differ")
        input_binding = CanonicalJsonProofBinding.from_dict(
            input_value["tokenization_input"]
        )
        output_binding = CanonicalJsonProofBinding.from_dict(
            output_value["tokenization_output"]
        )
        batch_source_rows = source_rows[cursor : interval[2]]
        derived = _materialized_schedule_rows(
            source=source,
            launch=launch,
            tokenization_input=input_binding,
            tokenization_output=output_binding,
            source_rows=batch_source_rows,
        )
        token_value = output_binding.reopen()
        if (
            token_value.get("transformers_version") != receipt.transformers_version
            or token_value.get("tokenizer_class") != receipt.tokenizer_class
            or token_value.get("tokenizer_vocab_size") != receipt.tokenizer_vocab_size
        ):
            raise ValueError("formal sharded tokenizer metadata changed")
        for row in derived:
            try:
                expected = next(receipt_rows)
            except StopIteration as error:
                raise ValueError("formal sharded receipt has too few rows") from error
            if row != expected:
                raise ValueError("formal sharded tokenized request row changed")
        argv_sha256s.append(
            _require_sha256(
                "formal tokenization batch argv",
                output_value["tokenizer_worker_argv_sha256"],
            )
        )
        cursor = interval[2]
    try:
        next(receipt_rows)
    except StopIteration:
        pass
    else:
        raise ValueError("formal sharded receipt has extra rows")
    worker, worker_raw_sha256, worker_size = _tokenizer_worker_source()
    del worker
    if (
        cursor != len(source_rows)
        or receipt.request_count != len(source_rows)
        or receipt.tokenizer_worker_source_raw_sha256 != worker_raw_sha256
        or receipt.tokenizer_worker_source_size != worker_size
        or receipt.tokenizer_worker_argv_sha256
        != _sha256({"ordered_batch_argv_sha256s": argv_sha256s})
    ):
        raise ValueError("formal sharded tokenizer worker identity changed")


def _reopen_stage_materialization(
    binding: CanonicalJsonProofBinding,
) -> StageMaterializationReceipt:
    # Import lazily: the formal registry also consumes physical result types.
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )

    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal schedule materialization path changed")
    materialization = stage_materialization_receipt_from_dict(binding.reopen())
    return materialization


def _bind_stage_materialization(
    path: str | Path,
) -> tuple[CanonicalJsonProofBinding, StageMaterializationReceipt]:
    binding = CanonicalJsonProofBinding.bind(path)
    return binding, _reopen_stage_materialization(binding)


def _materialized_cell(
    materialization: StageMaterializationReceipt, *, cell_id: str
) -> MaterializedCell:
    matches = tuple(row for row in materialization.cells if row.cell_id == cell_id)
    if len(matches) != 1:
        raise ValueError("formal schedule cell is outside exact materialization")
    return matches[0]


def _verified_workload_authorization(
    receipt: ContentVerificationReceipt, *, current_ns: int
) -> VerifiedReleaseWorkloadSources:
    verified = receipt.revalidate_formal_scope(current_ns=current_ns)
    matches = tuple(
        row for row in verified if type(row) is VerifiedReleaseWorkloadSources
    )
    if len(matches) != 1:
        raise ValueError("formal schedule lacks one root-verified workload authority")
    return matches[0]


def _reject_master_derived_schedule_artifacts(
    receipt: ContentVerificationReceipt,
) -> None:
    """A master/scoped content receipt predates every materialized-cell schedule."""

    if any(
        row.artifact_id.startswith(_SCHEDULE_ARTIFACT_PREFIX)
        for row in receipt.content_artifacts
    ):
        raise ValueError(
            "formal content receipt cannot authorize a derived per-cell schedule"
        )


def _workload_id_for_cell(cell: MaterializedCell) -> str:
    if cell.stage == "E0":
        task = (
            dict(cell.dimensions).get("deployment_task")
            if cell.task == "independent_onlinespec_tuning"
            else cell.task
        )
        if type(task) is not str or not task:
            raise ValueError(
                "formal request schedule is BLOCKED: "
                "task_native_deployment_task_missing"
            )
        if task == "MT-Bench":
            raise ValueError(
                "formal request schedule is BLOCKED: "
                "task_native_multi_turn_transport_unsupported"
            )
        if task == "LiveCodeBench":
            return "livecodebench_v6_hard"
        if task == "MATH-500":
            return "math500_level5"
        if task in _TRUSTED_E0_WORKLOAD_IDS:
            return task
        raise ValueError(
            "formal request schedule is BLOCKED: task_native_dataset_unsupported"
        )
    task = cell.task.casefold()
    regime = str(dict(cell.dimensions).get("regime", "")).casefold()
    if "math" in task or "long_input" in regime or "short_output" in regime:
        return "math500_level5"
    if cell.stage == "TTS-Cal":
        raise ValueError(
            "formal request schedule is BLOCKED: tuning_window_reducer_required"
        )
    return "livecodebench_v6_hard"


def _root_verified_workload_source(
    receipt: ContentVerificationReceipt,
    *,
    workload_id: FormalWorkloadId,
    workload_authority_path: str | Path,
    current_ns: int,
) -> tuple[ContentJsonArtifactBinding, FormalWorkloadAuthority, str]:
    """Deep-open a post-master workload reducer under the master's root authority."""

    _reject_master_derived_schedule_artifacts(receipt)
    authorization = _verified_workload_authorization(receipt, current_ns=current_ns)
    artifact_id = formal_workload_authority_artifact_id(workload_id)
    binding = ContentJsonArtifactBinding.from_path(
        artifact_id,
        workload_authority_path,
    )
    authority = formal_workload_authority_from_cli_artifact(binding.load())
    if authority.workload_id != workload_id:
        raise ValueError("formal workload artifact names another source")
    descriptor = authorization.source(workload_id)
    return (
        binding,
        revalidate_authorized_formal_workload_authority(
            authority,
            authorization=authorization,
        ),
        descriptor.sha256,
    )


def _tts_calibration_window(
    receipt: ContentVerificationReceipt,
) -> TtsCalibrationTuningWindow:
    artifacts = tuple(
        row
        for row in receipt.content_artifacts
        if row.artifact_id == "tts_calibration_tuning_window"
    )
    if len(artifacts) != 1:
        raise ValueError("TTS calibration lacks one path-bound tuning window")
    window = TtsCalibrationTuningWindow.from_dict(artifacts[0].load())
    if artifacts[0].semantic_sha256 != window.sha256:
        raise ValueError("TTS calibration tuning-window binding differs")
    return window


def _tts_calibration_window_samples(
    window: TtsCalibrationTuningWindow,
    *,
    workload: FormalWorkloadAuthority,
    workload_descriptor_sha256: str,
) -> tuple[FormalWorkloadSample, ...]:
    by_id = {row.sample_id: row for row in workload.samples}
    if len(by_id) != len(workload.samples):
        raise ValueError("TTS calibration workload repeats a sample identity")
    resolved: dict[str, FormalWorkloadSample] = {}
    for entry in window.entries:
        sample = by_id.get(entry.source_sample_id)
        if (
            entry.workload_id != workload.workload_id
            or entry.source_descriptor_sha256 != workload_descriptor_sha256
            or sample is None
            or _sha256(sample.prompt) != entry.prompt_sha256
        ):
            raise ValueError(
                "TTS calibration window entry differs from root-authorized prompt"
            )
        resolved[entry.entry_id] = sample
    selected = tuple(resolved[row.entry_id] for row in window.tuning_entries)
    if len(selected) != len(window.tuning_entries):
        raise ValueError("TTS calibration tuning prompt resolution differs")
    return selected


def _registry_cell_for_materialized(cell: MaterializedCell):
    registry_cell_id = dict(cell.dimensions).get("registry_cell_id")
    if registry_cell_id is None:
        return None
    if type(registry_cell_id) is not str:
        raise TypeError("formal schedule registry cell ID must be text")
    matches = tuple(
        row
        for row in build_industrial_registry().cells_for(cell.stage)
        if row.cell_id == registry_cell_id
    )
    if len(matches) != 1:
        raise ValueError("formal schedule registry source is not exact")
    return matches[0]


def _load_protocol_for_cell(
    *,
    cell: MaterializedCell,
    max_running_requests: int,
    server_context_limit: int,
    e5_arrival_plan: object | None = None,
) -> dict[str, object]:
    dimensions = dict(cell.dimensions)
    if cell.stage == "E4" and cell.task == "mechanism_profile_only":
        raise ValueError(
            "formal request schedule is BLOCKED: "
            "dedicated_e4_profiler_schedule_required"
        )
    registry_cell = _registry_cell_for_materialized(cell)
    identity = None if registry_cell is None else registry_cell.identity
    context = dimensions.get(
        "context",
        (None if identity is None else identity.context) or server_context_limit,
    )
    regime = dimensions.get("regime", None if identity is None else identity.regime)
    seed = None if identity is None else identity.seed
    cohort_count = dimensions.get(
        "cohort_count", 1 if identity is None else identity.cohort_count
    )
    cohort_distribution = dimensions.get("cohort_distribution", "uniform")
    raw_arrival = dimensions.get(
        "arrival", None if identity is None else identity.arrival
    )
    family = dimensions.get("family")
    load = dimensions.get("load")
    common_load = dimensions.get("common_load")
    declared_concurrency = dimensions.get(
        "concurrency", None if identity is None else identity.concurrency
    )
    if cell.stage == "E5" and cell.task == "deterministic_failure_injection":
        if e5_arrival_plan is not None:
            raise ValueError("E5 failure schedule carries a headline arrival plan")
        context = 40_928
        regime = "short_input_long_generation"
        declared_concurrency = max_running_requests
        raw_arrival = "closed_loop"
    elif cell.stage == "E5":
        from lightcone_spec.experiments.formal_single_operator_loads import (
            E5ArrivalPlan,
        )

        if type(e5_arrival_plan) is not E5ArrivalPlan:
            raise ValueError(
                "formal request schedule is BLOCKED: e5_arrival_plan_required"
            )
        expected_block = dimensions.get("block")
        if (
            e5_arrival_plan.cell_id != cell.cell_id
            or e5_arrival_plan.family != family
            or e5_arrival_plan.block != expected_block
        ):
            raise ValueError("formal E5 arrival plan differs from sealed cell")
        context = 40_928
        regime = "short_input_long_generation"
        declared_concurrency = e5_arrival_plan.concurrency
        raw_arrival = e5_arrival_plan.arrival_policy
    elif e5_arrival_plan is not None:
        raise ValueError("non-E5 request schedule carries an E5 arrival plan")
    if cell.stage == "E6":
        if cell.task == "immutable_metadata_interface_and_fit_preflight":
            raise ValueError(
                "formal request schedule is BLOCKED: E6_interface_preflight_is_not_serving"
            )
        regime = "native_mtp_transfer"
    if declared_concurrency is None and type(common_load) is int:
        declared_concurrency = common_load
    if load == "concurrency_one":
        declared_concurrency = 1
    elif load == "common_load":
        declared_concurrency = max_running_requests
    elif (
        cell.stage == "E4"
        and cell.task
        in {
            "mechanism_strength2_screen_headline",
            "winner_neighborhood_local_factorial_headline",
        }
        and load in {"low", "moderate", "saturation"}
    ):
        # Numeric E4 load is owned by the current-only execution mapper.  This
        # reducer merely joins the resulting RunConfig to the sealed label.
        context = 40_928
        regime = dimensions.get("traffic")
        declared_concurrency = max_running_requests
        raw_arrival = "closed_loop"
    if family == "closed_loop":
        raw_arrival = "closed_loop"
    if cell.stage == "TTS-Cal" and raw_arrival == "disjoint_tuning_window":
        # The registry label selects the root-authorized disjoint request set;
        # it is not an arrival process.  The caller cannot reach this path
        # without the exact typed tuning window checked by the reducer below.
        raw_arrival = "closed_loop"
    if raw_arrival in {None, "locked_reference_load"} and type(common_load) is int:
        raw_arrival = "closed_loop"
    if raw_arrival is None and cell.stage in {
        "E3a",
        "TTS-Cal",
        "E1",
        "E2",
        "E4",
        "E3b",
        "E1a",
        "E6",
    }:
        raw_arrival = "closed_loop"
    if cell.stage != "E5" and (
        family in {"open_loop", "trace_or_soak"}
        or raw_arrival not in {"closed_loop", "closed_loop_c1"}
    ):
        raise ValueError(
            "formal request schedule is BLOCKED: arrival_reducer_not_registered"
        )
    if (
        type(context) is not int
        or context < 1
        or type(regime) is not str
        or not regime
        or type(cohort_count) is not int
        or cohort_count not in {1, 4, 16, 64}
        or cohort_distribution not in {"uniform", "zipf"}
        or type(max_running_requests) is not int
        or max_running_requests < 1
        or (
            cell.stage != "E5"
            and declared_concurrency is not None
            and declared_concurrency != max_running_requests
        )
    ):
        raise ValueError("formal request schedule load differs from sealed cell")
    paired_request_pool_sha256 = _sha256(
        {
            "stage": cell.stage,
            "task": cell.task,
            "block": dimensions.get("block"),
            "context": context,
            "regime": regime,
            "family": family,
            "load": load,
            "arrival": raw_arrival,
            "concurrency": declared_concurrency,
            "width_panel": dimensions.get("width_panel"),
            "topology": dimensions.get("topology"),
            "cohort_count": cohort_count,
            "cohort_distribution": cohort_distribution,
        }
    )
    paired_seed = int(paired_request_pool_sha256[:16], 16)
    return {
        "schema_version": 1,
        "derivation_protocol_sha256": (
            FORMAL_SERVING_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        "stage": cell.stage,
        "materialized_cell_id": cell.cell_id,
        "context_tokens": context,
        "regime": regime,
        "arrival_policy": (
            e5_arrival_plan.arrival_policy
            if cell.stage == "E5" and e5_arrival_plan is not None
            else "closed_loop_zero_think"
        ),
        "max_running_requests": max_running_requests,
        "cohort_count": cohort_count,
        "cohort_distribution": cohort_distribution,
        # A missing registry seed is derived from method-independent scientific
        # axes.  Falling back to ``cell_id`` made otherwise identical role
        # traces receive different cohort assignments and request IDs.
        "seed": paired_seed if seed is None else seed,
        "paired_request_pool_sha256": paired_request_pool_sha256,
        "warmup_selection": "first_max_four_or_twenty_percent",
        "scored_selection": "all_remaining_rows_in_source_order",
        "cancellation": "none",
    }


def _requested_output_tokens(*, context: int, regime: str) -> int:
    folded = regime.casefold()
    if "long_input" in folded or "short_output" in folded:
        return max(1, min(256, context // 4))
    return max(1, min(2_048, context // 2))


def _scored_split(stage: str) -> str:
    if stage in {"E3a", "TTS-Cal", "E1", "E2", "E4"}:
        return "tuning"
    if stage in {"E3b", "E5", "E6"}:
        return "confirmation"
    return "broad_replication"


def _route_for_cohort(cohort_id: str, *, topology_mode: str) -> int | None:
    if topology_mode != "tp1_dp2":
        return None
    try:
        return int(cohort_id.rsplit("-", 1)[1]) % 2
    except (IndexError, ValueError) as error:
        raise ValueError("formal DP cohort identity is not canonical") from error


def rebuild_formal_serving_request_schedule_source(
    *,
    subject_sha256: str,
    workload_authority_sha256: str,
    topology_mode: str,
    materialization: StageMaterializationReceipt,
    materialized_cell_id: str,
    workload_source: FormalWorkloadAuthority,
    workload_source_descriptor_sha256: str,
    tts_tuning_window: TtsCalibrationTuningWindow | None,
    sampling_profile: SamplingProfile,
    max_running_requests: int,
    server_context_limit: int,
    tokenizer_content_member_id: str,
    tokenizer_model_id: str,
    tokenizer_revision: str,
    tokenizer_content_authority_sha256: str,
) -> FormalServingRequestScheduleSource:
    """Deterministically reduce root-verified rows; no schedule row is input."""

    cell = _materialized_cell(materialization, cell_id=materialized_cell_id)
    expected_workload = (
        workload_source.workload_id
        if cell.stage == "TTS-Cal"
        else _workload_id_for_cell(cell)
    )
    if workload_source.workload_id != expected_workload:
        raise ValueError("formal request workload differs from sealed cell")
    _require_sha256(
        "formal request workload descriptor", workload_source_descriptor_sha256
    )
    workload_source.__post_init__()
    sampling_profile.validate()
    load_protocol = _load_protocol_for_cell(
        cell=cell,
        max_running_requests=max_running_requests,
        server_context_limit=server_context_limit,
    )
    if cell.stage == "TTS-Cal":
        if type(tts_tuning_window) is not TtsCalibrationTuningWindow:
            raise ValueError("TTS calibration schedule lacks its typed window")
        samples = _tts_calibration_window_samples(
            tts_tuning_window,
            workload=workload_source,
            workload_descriptor_sha256=workload_source_descriptor_sha256,
        )
        tuning_window_sha256 = tts_tuning_window.sha256
        tuning_entry_ids = tuple(
            sorted(row.entry_id for row in tts_tuning_window.tuning_entries)
        )
    else:
        if tts_tuning_window is not None:
            raise ValueError("non-TTS schedule carries a tuning window")
        samples = workload_source.samples
        tuning_window_sha256 = None
        tuning_entry_ids = ()
    if len(samples) < max_running_requests + 1:
        raise ValueError("formal workload cannot cover warmup and closed-loop lanes")
    warmup_count = max(1, min(4, len(samples) // 5))
    if len(samples) - warmup_count < max_running_requests:
        warmup_count = 1
    cohort_count = int(load_protocol["cohort_count"])
    cohort_distribution = str(load_protocol["cohort_distribution"])
    seed = int(load_protocol["seed"])
    cohorts = cohort_assignments(
        len(samples),
        cohort_count=cohort_count,
        popularity=cohort_distribution,  # type: ignore[arg-type]
        seed=seed,
    )
    context = int(load_protocol["context_tokens"])
    regime = str(load_protocol["regime"])
    output_tokens = _requested_output_tokens(context=context, regime=regime)
    # Request identity is a paired scientific identity, not a method/cell
    # identity.  Every replay-relevant field is already part of
    # ``ImmutableRequest.request_id``; including ``cell_id`` here made the five
    # role schedules unpairable even when their source trace was byte-identical.
    namespace = (
        f"formal:{cell.stage}:{workload_source.workload_id}:"
        f"registered-request-pool:{load_protocol['paired_request_pool_sha256']}"
    )
    rows = tuple(
        FormalServingRequestScheduleSourceRow(
            source_member_sha256=workload_source_descriptor_sha256,
            source_raw_file_sha256=workload_source.raw_file_sha256,
            source_selected_rows_sha256=workload_source.selected_rows_sha256,
            source_sample_id=sample.sample_id,
            prompt=sample.prompt,
            prompt_sha256=_sha256(sample.prompt),
            phase="warmup" if ordinal < warmup_count else "scored",
            namespace=namespace,
            split=("warmup" if ordinal < warmup_count else _scored_split(cell.stage)),  # type: ignore[arg-type]
            ordinal=ordinal,
            requested_output_tokens=output_tokens,
            arrival_us=0,
            cancellation_offset_us=None,
            cohort_id=cohorts[ordinal],
            routed_dp_rank=_route_for_cohort(
                cohorts[ordinal], topology_mode=topology_mode
            ),
            sampling=FrozenSamplingParameters.from_mapping(
                sampling_profile.parameters(
                    seed=sample.seed,
                    max_new_tokens=output_tokens,
                )
            ).items,
        )
        for ordinal, sample in enumerate(samples)
    )
    return FormalServingRequestScheduleSource(
        schema_version=4,
        kind="formal_serving_request_schedule_source",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        derivation_protocol_sha256=(
            FORMAL_SERVING_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        subject_sha256=subject_sha256,
        materialization_receipt_sha256=materialization.sha256,
        materialized_cell_id=cell.cell_id,
        workload_authority_sha256=workload_authority_sha256,
        workload_id=workload_source.workload_id,
        workload_source_descriptor_sha256=workload_source_descriptor_sha256,
        workload_source_authority_sha256=workload_source.sha256,
        tts_tuning_window_sha256=tuning_window_sha256,
        tts_tuning_entry_ids=tuning_entry_ids,
        sampling_profile_sha256=sampling_profile.sha256,
        load_protocol_sha256=_sha256(load_protocol),
        context_tokens=context,
        regime=regime,
        arrival_policy=str(load_protocol["arrival_policy"]),
        max_running_requests=max_running_requests,
        cohort_count=cohort_count,
        topology_mode=topology_mode,
        tokenizer_content_member_id=tokenizer_content_member_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_content_authority_sha256=(tokenizer_content_authority_sha256),
        requests=rows,
    )


def rebuild_trusted_single_operator_request_schedule_source(
    *,
    subject_sha256: str,
    content_source_binding: FormalContentSourceBinding,
    topology_mode: str,
    materialization: StageMaterializationReceipt,
    materialized_cell_id: str,
    workload_source: object,
    workload_source_binding: ContentJsonArtifactBinding,
    sampling_profile: SamplingProfile,
    max_running_requests: int,
    server_context_limit: int,
    tokenizer_content_member_id: str,
    tokenizer_model_id: str,
    tokenizer_revision: str,
    tts_tuning_window: TtsCalibrationTuningWindow | None = None,
    trusted_tts_calibration_authority_sha256: str | None = None,
    e5_arrival_plan: CanonicalJsonProofBinding | None = None,
) -> FormalServingRequestScheduleSource:
    """Rebuild schema-5 requests from one BOUND tagged content bundle.

    This is deliberately separate from the signed reducer above.  It accepts
    no prompt, seed, arrival, load, or model path from its caller: those values
    are selected from the exact materialized cell, trusted content member,
    tokenizer member, and (for E5) a code-owned arrival plan that is rederived
    byte-for-byte.
    """

    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
    )
    from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
        E0TaskNativeSourceAuthority,
    )
    from lightcone_spec.experiments.formal_single_operator_loads import (
        E5ArrivalPlan,
        derive_e5_arrival_plan,
    )

    if (
        type(content_source_binding) is not FormalContentSourceBinding
        or content_source_binding.mode != "trusted_single_operator"
    ):
        raise ValueError("trusted request reducer requires tagged trusted content")
    bundle = content_source_binding.reopen()
    if (
        type(bundle) is not TrustedSingleOperatorContentBundle
        or bundle.runtime_binding_status != "BOUND"
    ):
        raise ValueError("trusted request reducer requires one BOUND content bundle")
    cell = _materialized_cell(materialization, cell_id=materialized_cell_id)
    expected_workload_id = _workload_id_for_cell(cell)
    task_native_workload_sha256 = (
        dict(cell.dimensions).get("task_native_workload_sha256")
        if cell.stage == "E0"
        else None
    )
    if cell.stage == "E0":
        _require_sha256(
            "trusted E0 task-native workload",
            task_native_workload_sha256,
        )
    tokenizer_members = tuple(
        row
        for row in bundle.model_members
        if row.role == "tokenizer"
        and cell.stage in row.stages
        and row.model_id == tokenizer_model_id
        and row.revision == tokenizer_revision
        and row.sha256 == tokenizer_content_member_id
    )
    if len(tokenizer_members) != 1:
        raise ValueError("trusted request tokenizer member is not exact")
    if type(workload_source_binding) is not ContentJsonArtifactBinding:
        raise TypeError("trusted request workload source is not path-bound")

    source_rows: tuple[tuple[str, str, int], ...]
    raw_file_sha256: str
    selected_rows_sha256: str
    if type(workload_source) is FormalWorkloadAuthority:
        if workload_source.workload_id != expected_workload_id:
            raise ValueError("trusted request workload differs from sealed cell")
        members = tuple(
            row
            for row in bundle.locked_workloads
            if row.workload_id == workload_source.workload_id
            and row.authority_sha256 == workload_source.sha256
            and row.raw_source_path == workload_source.raw_source_path
            and row.raw_file_sha256 == workload_source.raw_file_sha256
            and row.repository_revision == workload_source.repository_revision
            and row.raw_row_count == workload_source.raw_row_count
            and row.selected_row_count == workload_source.selected_row_count
            and row.formal_samples_sha256 == workload_source.selected_rows_sha256
            and row.source_lock_sha256 == workload_source.source_lock_sha256
            and row.protocol_sha256 == workload_source.protocol_sha256
        )
        if len(members) != 1:
            raise ValueError("trusted request locked workload member is not exact")
        member = members[0]
        if workload_source_binding.artifact_id != formal_workload_authority_artifact_id(
            workload_source.workload_id
        ) or workload_source_binding.load() != formal_workload_authority_cli_artifact(
            workload_source
        ):
            raise ValueError("trusted request workload path binding differs")
        samples = workload_source.samples
        if cell.stage == "TTS-Cal":
            if type(tts_tuning_window) is not TtsCalibrationTuningWindow:
                raise ValueError("trusted TTS schedule lacks its typed tuning window")
            _require_sha256(
                "trusted TTS calibration authority",
                trusted_tts_calibration_authority_sha256,
            )
            by_id = {row.sample_id: row for row in samples}
            selected: list[FormalWorkloadSample] = []
            for entry in tts_tuning_window.tuning_entries:
                sample = by_id.get(entry.source_sample_id)
                if (
                    entry.workload_id != workload_source.workload_id
                    or sample is None
                    or _sha256(sample.prompt) != entry.prompt_sha256
                ):
                    raise ValueError("trusted TTS tuning entry differs from workload")
                selected.append(sample)
            samples = tuple(selected)
            tuning_window_sha256 = tts_tuning_window.sha256
            tuning_entry_ids = tuple(
                sorted(row.entry_id for row in tts_tuning_window.tuning_entries)
            )
        else:
            if (
                tts_tuning_window is not None
                or trusted_tts_calibration_authority_sha256 is not None
            ):
                raise ValueError("non-TTS trusted schedule carries a tuning window")
            tuning_window_sha256 = None
            tuning_entry_ids = ()
        source_rows = tuple((row.sample_id, row.prompt, row.seed) for row in samples)
        raw_file_sha256 = workload_source.raw_file_sha256
        selected_rows_sha256 = workload_source.selected_rows_sha256
        descriptor_sha256 = member.sha256
    elif type(workload_source) is E0TaskNativeSourceAuthority:
        if (
            cell.stage != "E0"
            or workload_source.task != expected_workload_id
            or workload_source.support_status != "READY"
            or tts_tuning_window is not None
            or trusted_tts_calibration_authority_sha256 is not None
            or workload_source.task_native_workload_sha256
            != task_native_workload_sha256
        ):
            raise ValueError("trusted E0 request workload is not serving-ready")
        workload_source.revalidate()
        members = tuple(
            row
            for row in bundle.e0_task_native_descriptors
            if row.task == workload_source.task
            and row.source.absolute_path == workload_source_binding.path
            and row.source.raw_sha256 == workload_source_binding.raw_sha256
            and row.source.semantic_sha256 == workload_source_binding.semantic_sha256
            and row.source.size == workload_source_binding.size
        )
        if len(members) != 1:
            raise ValueError("trusted E0 request descriptor is not exact")
        member = members[0]
        if workload_source_binding.artifact_id != (
            f"trusted_e0_task_native:{workload_source.task}"
        ):
            raise ValueError("trusted E0 request artifact ID differs")
        source_rows = tuple(
            (row.source_id, row.turns[0], row.seed)
            for row in workload_source.request_rows
        )
        raw_file_sha256 = workload_source.raw_file_sha256
        selected_rows_sha256 = workload_source.request_rows_sha256
        descriptor_sha256 = member.sha256
        tuning_window_sha256 = None
        tuning_entry_ids = ()
    else:
        raise TypeError("trusted request workload source type is unsupported")
    if not source_rows:
        raise ValueError("trusted request workload contains no request rows")

    plan: E5ArrivalPlan | None = None
    if cell.stage == "E5" and cell.task == "production_slo_power_prefix":
        if type(e5_arrival_plan) is not CanonicalJsonProofBinding:
            raise ValueError("trusted E5 schedule lacks its arrival plan")
        plan = E5ArrivalPlan.from_dict(e5_arrival_plan.reopen())
        dimensions = dict(cell.dimensions)
        expected_plan = derive_e5_arrival_plan(
            cell_id=cell.cell_id,
            block=int(dimensions["block"]),
            family=str(dimensions["family"]),  # type: ignore[arg-type]
            dimensions=dimensions,
            lambda_star=plan.lambda_star,
            burstgpt_verification=(
                bundle.burstgpt_release.release_verification
                if dimensions.get("arrival") == "burstgpt_shape"
                else None
            ),
            burstgpt_active_asset_path=(
                next(
                    row.absolute_path
                    for row in bundle.burstgpt_release.assets
                    if row.name == bundle.burstgpt_release.active_asset
                )
                if dimensions.get("arrival") == "burstgpt_shape"
                else None
            ),
            selected_p99_anchor=("p99_extension_anchor_id" in dimensions),
        )
        if plan != expected_plan or e5_arrival_plan.semantic_sha256 != plan.sha256:
            raise ValueError("trusted E5 arrival plan failed exact rederivation")
    elif e5_arrival_plan is not None:
        raise ValueError("non-E5 trusted schedule carries an E5 arrival plan")

    load_protocol = _load_protocol_for_cell(
        cell=cell,
        max_running_requests=max_running_requests,
        server_context_limit=server_context_limit,
        e5_arrival_plan=plan,
    )
    if plan is not None and plan.concurrency != max_running_requests:
        raise ValueError("trusted E5 arrival concurrency differs from RunConfig")
    ordinary_warmup_count = max(1, min(4, len(source_rows) // 5))
    if plan is None:
        if len(source_rows) < max_running_requests + 1:
            raise ValueError("trusted workload cannot cover closed-loop lanes")
        warmup_count = ordinary_warmup_count
        if len(source_rows) - warmup_count < max_running_requests:
            warmup_count = 1
        scheduled = tuple(
            (row, 0, ordinal < warmup_count) for ordinal, row in enumerate(source_rows)
        )
    else:
        warmup_count = min(ordinary_warmup_count, len(source_rows))
        if len(plan.arrivals_us) + warmup_count > _MAX_REQUEST_ROWS:
            raise ValueError("trusted E5 schedule exceeds request-row ceiling")
        warmups = tuple((source_rows[index], 0, True) for index in range(warmup_count))
        scored = tuple(
            (
                source_rows[(warmup_count + index) % len(source_rows)],
                arrival,
                False,
            )
            for index, arrival in enumerate(plan.arrivals_us)
        )
        scheduled = warmups + scored

    cohort_count = int(load_protocol["cohort_count"])
    cohort_distribution = str(load_protocol["cohort_distribution"])
    seed = int(load_protocol["seed"])
    cohorts = cohort_assignments(
        len(scheduled),
        cohort_count=cohort_count,
        popularity=cohort_distribution,  # type: ignore[arg-type]
        seed=seed,
    )
    context = int(load_protocol["context_tokens"])
    regime = str(load_protocol["regime"])
    output_tokens = _requested_output_tokens(context=context, regime=regime)
    namespace = (
        f"formal:E5:{expected_workload_id}:{plan.paired_trace_sha256}"
        if plan is not None
        else (
            f"formal:{cell.stage}:{expected_workload_id}:registered-request-pool:"
            f"{load_protocol['paired_request_pool_sha256']}"
        )
    )
    rows = tuple(
        FormalServingRequestScheduleSourceRow(
            source_member_sha256=descriptor_sha256,
            source_raw_file_sha256=raw_file_sha256,
            source_selected_rows_sha256=selected_rows_sha256,
            source_sample_id=source_id,
            prompt=prompt,
            prompt_sha256=_sha256(prompt),
            phase="warmup" if is_warmup else "scored",
            namespace=namespace,
            split="warmup" if is_warmup else _scored_split(cell.stage),  # type: ignore[arg-type]
            ordinal=ordinal,
            requested_output_tokens=output_tokens,
            arrival_us=arrival_us,
            cancellation_offset_us=None,
            cohort_id=cohorts[ordinal],
            routed_dp_rank=_route_for_cohort(
                cohorts[ordinal], topology_mode=topology_mode
            ),
            sampling=FrozenSamplingParameters.from_mapping(
                sampling_profile.parameters(
                    seed=request_seed,
                    max_new_tokens=output_tokens,
                )
            ).items,
        )
        for ordinal, (
            (source_id, prompt, request_seed),
            arrival_us,
            is_warmup,
        ) in enumerate(scheduled)
    )
    return FormalServingRequestScheduleSource(
        schema_version=5,
        kind="formal_serving_request_schedule_source",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        derivation_protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        subject_sha256=subject_sha256,
        materialization_receipt_sha256=materialization.sha256,
        materialized_cell_id=cell.cell_id,
        workload_authority_sha256=None,
        workload_id=expected_workload_id,
        workload_source_descriptor_sha256=descriptor_sha256,
        workload_source_authority_sha256=None,
        tts_tuning_window_sha256=tuning_window_sha256,
        tts_tuning_entry_ids=tuning_entry_ids,
        sampling_profile_sha256=sampling_profile.sha256,
        load_protocol_sha256=(
            plan.sha256 if plan is not None else _sha256(load_protocol)
        ),
        context_tokens=context,
        regime=regime,
        arrival_policy=str(load_protocol["arrival_policy"]),
        max_running_requests=max_running_requests,
        cohort_count=cohort_count,
        topology_mode=topology_mode,
        tokenizer_content_member_id=tokenizer_content_member_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_content_authority_sha256=None,
        requests=rows,
        e5_arrival_plan=e5_arrival_plan,
        content_source_binding_sha256=content_source_binding.sha256,
        trusted_workload_member_sha256=descriptor_sha256,
        trusted_tts_calibration_authority_sha256=(
            trusted_tts_calibration_authority_sha256
        ),
        trusted_task_native_workload_sha256=task_native_workload_sha256,
    )


def revalidate_formal_serving_request_schedule_source(
    source: FormalServingRequestScheduleSource,
    *,
    materialization: StageMaterializationReceipt,
    workload_source: FormalWorkloadAuthority,
    workload_source_descriptor_sha256: str,
    tts_tuning_window: TtsCalibrationTuningWindow | None,
    sampling_profile: SamplingProfile,
    server_context_limit: int,
) -> FormalServingRequestScheduleSource:
    """Replay every row from trusted upstream inputs and require byte identity."""

    if type(source) is not FormalServingRequestScheduleSource:
        raise TypeError("formal schedule revalidation requires an exact source")
    rebuilt = rebuild_formal_serving_request_schedule_source(
        subject_sha256=source.subject_sha256,
        workload_authority_sha256=source.workload_authority_sha256,
        topology_mode=source.topology_mode,
        materialization=materialization,
        materialized_cell_id=source.materialized_cell_id,
        workload_source=workload_source,
        workload_source_descriptor_sha256=workload_source_descriptor_sha256,
        tts_tuning_window=tts_tuning_window,
        sampling_profile=sampling_profile,
        max_running_requests=source.max_running_requests,
        server_context_limit=server_context_limit,
        tokenizer_content_member_id=source.tokenizer_content_member_id,
        tokenizer_model_id=source.tokenizer_model_id,
        tokenizer_revision=source.tokenizer_revision,
        tokenizer_content_authority_sha256=(source.tokenizer_content_authority_sha256),
    )
    if rebuilt != source:
        raise ValueError("formal serving request schedule differs from source reducer")
    return source


def _tokenizer_worker_source() -> tuple[Path, str, int]:
    path = (
        Path(__file__).resolve().parent.parent
        / "sglang_bridge"
        / "formal_tokenize_worker.py"
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError("formal tokenizer worker source is unavailable")
    body = path.read_bytes()
    return path, hashlib.sha256(body).hexdigest(), len(body)


def _publish_tokenization_input(
    *,
    path: Path,
    source: FormalServingRequestScheduleSource,
    launch: CompileLaunchManifest,
) -> CanonicalJsonProofBinding:
    return _publish_tokenization_input_rows(
        path=path,
        schedule_source_sha256=source.sha256,
        rows=source.requests,
        launch=launch,
    )


def _publish_tokenization_input_rows(
    *,
    path: Path,
    schedule_source_sha256: str,
    rows: tuple[FormalServingRequestScheduleSourceRow, ...],
    launch: CompileLaunchManifest,
) -> CanonicalJsonProofBinding:
    if not rows:
        raise ValueError("formal tokenization batch must be non-empty")
    value = {
        "schema_version": 1,
        "kind": "formal_serving_tokenization_input",
        "protocol_sha256": FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        "schedule_source_sha256": schedule_source_sha256,
        "tokenizer_model_id": launch.tokenizer_model_id,
        "tokenizer_revision": launch.tokenizer_revision,
        "tokenizer_snapshot_path": launch.tokenizer_snapshot_path,
        "tokenizer_content_authority_sha256": (
            launch.tokenizer_content_authority_sha256
        ),
        "requests": [
            {
                "request_id": row.source_request_key,
                "ordinal": row.ordinal,
                "prompt": row.prompt,
                "prompt_sha256": row.prompt_sha256,
            }
            for row in rows
        ],
    }
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _invoke_tokenizer_worker(
    *, input_path: Path, output_path: Path
) -> tuple[CanonicalJsonProofBinding, str, int, str]:
    worker, worker_raw_sha256, worker_size = _tokenizer_worker_source()
    executable = Path(sys.executable).resolve()
    if not executable.is_file() or executable.is_symlink():
        raise ValueError("formal tokenizer interpreter is unavailable")
    argv = (
        str(executable),
        str(worker),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    )
    source_root = Path(__file__).resolve().parents[2]
    project_root = Path(__file__).resolve().parents[3]
    environment = {
        "PATH": str(executable.parent),
        "PYTHONPATH": str(source_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C",
        "LC_ALL": "C",
    }
    completed = subprocess.run(
        argv,
        cwd=project_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=600.0,
    )
    if (
        completed.returncode != 0
        or completed.stdout
        or completed.stderr
        or not output_path.is_file()
        or output_path.is_symlink()
    ):
        raise RuntimeError("first-party formal tokenizer worker failed")
    return (
        CanonicalJsonProofBinding.bind(output_path),
        worker_raw_sha256,
        worker_size,
        _sha256({"argv": list(argv)}),
    )


def _materialized_schedule_rows(
    *,
    source: FormalServingRequestScheduleSource,
    launch: CompileLaunchManifest,
    tokenization_input: CanonicalJsonProofBinding,
    tokenization_output: CanonicalJsonProofBinding,
    source_rows: tuple[FormalServingRequestScheduleSourceRow, ...] | None = None,
) -> tuple[FormalServingRequestScheduleRow, ...]:
    selected_source_rows = source.requests if source_rows is None else source_rows
    output = tokenization_output.reopen()
    input_value = tokenization_input.reopen()
    fields_expected = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "schedule_source_sha256",
        "tokenizer_model_id",
        "tokenizer_revision",
        "tokenizer_snapshot_path",
        "tokenizer_content_authority_sha256",
        "tokenizer_class",
        "tokenizer_vocab_size",
        "transformers_version",
        "requests",
    }
    if (
        set(output) != fields_expected
        or output["schema_version"] != 1
        or output["kind"] != "formal_serving_tokenization_output"
        or output["protocol_sha256"] != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or output["schedule_source_sha256"] != source.sha256
        or output["tokenizer_model_id"] != source.tokenizer_model_id
        or output["tokenizer_model_id"] != launch.tokenizer_model_id
        or output["tokenizer_revision"] != source.tokenizer_revision
        or output["tokenizer_revision"] != launch.tokenizer_revision
        or output["tokenizer_snapshot_path"] != launch.tokenizer_snapshot_path
        or output["tokenizer_content_authority_sha256"]
        != source.tokenizer_content_authority_sha256
        or output["tokenizer_content_authority_sha256"]
        != launch.tokenizer_content_authority_sha256
        or any(
            output[field] != input_value.get(field)
            for field in (
                "schedule_source_sha256",
                "tokenizer_model_id",
                "tokenizer_revision",
                "tokenizer_snapshot_path",
                "tokenizer_content_authority_sha256",
            )
        )
        or type(output["requests"]) is not list
        or len(output["requests"]) != len(selected_source_rows)
    ):
        raise ValueError("formal tokenizer output schema/coverage differs")
    result: list[FormalServingRequestScheduleRow] = []
    for source_row, token_row in zip(
        selected_source_rows,
        output["requests"],
        strict=True,
    ):
        token_value = _strict_object(
            "formal tokenizer output row",
            token_row,
            {
                "request_id",
                "ordinal",
                "prompt_sha256",
                "input_token_ids",
                "input_token_ids_sha256",
            },
        )
        input_ids = token_value["input_token_ids"]
        if (
            token_value["request_id"] != source_row.source_request_key
            or token_value["ordinal"] != source_row.ordinal
            or token_value["prompt_sha256"] != source_row.prompt_sha256
            or type(input_ids) is not list
            or token_value["input_token_ids_sha256"] != _sha256(input_ids)
        ):
            raise ValueError("formal tokenizer output row identity differs")
        immutable = ImmutableRequest.create(
            namespace=source_row.namespace,
            split=source_row.split,  # type: ignore[arg-type]
            ordinal=source_row.ordinal,
            template=RequestTemplate(
                input_token_ids=tuple(input_ids),
                requested_output_tokens=source_row.requested_output_tokens,
                sampling=FrozenSamplingParameters(items=source_row.sampling),
                cancellation_offset_us=source_row.cancellation_offset_us,
            ),
            arrival_us=source_row.arrival_us,
            cohort_id=source_row.cohort_id,
        )
        route_id = (
            f"dp{source_row.routed_dp_rank}"
            if source.topology_mode == "tp1_dp2"
            else source.topology_mode
        )
        bound = BoundServingRequest.create(immutable, route_id=route_id)
        result.append(
            FormalServingRequestScheduleRow(
                source_member_sha256=source_row.source_member_sha256,
                source_sample_id=source_row.source_sample_id,
                prompt_sha256=source_row.prompt_sha256,
                phase=source_row.phase,
                routed_dp_rank=source_row.routed_dp_rank,
                request=bound,
                tokenized_input_sha256=str(token_value["input_token_ids_sha256"]),
            )
        )
    return tuple(result)


def _livecodebench_tokenized_prompt_rows(
    *,
    workload: FormalWorkloadAuthority,
    rows: tuple[FormalServingRequestScheduleRow, ...],
) -> tuple[LiveCodeBenchTokenizedPrompt, ...]:
    """Translate exact receipt rows without importing the authority at startup."""

    from lightcone_spec.experiments.formal_single_operator_lcb_tokenizer import (
        LiveCodeBenchTokenizedPrompt,
    )

    if workload.workload_id != "livecodebench_v6_hard":
        raise ValueError("LiveCodeBench token rows require the locked hard workload")
    samples = {row.sample_id: row for row in workload.samples}
    if len(samples) != workload.selected_row_count:
        raise ValueError("LiveCodeBench workload repeats a sample identity")
    result: list[LiveCodeBenchTokenizedPrompt] = []
    for row in rows:
        sample = samples.get(row.source_sample_id)
        if (
            sample is None
            or row.prompt_sha256 != _sha256(sample.prompt)
            or row.tokenized_input_sha256 != _sha256(list(row.request.input_token_ids))
        ):
            raise ValueError("LiveCodeBench tokenized receipt row changed")
        result.append(
            LiveCodeBenchTokenizedPrompt(
                source_sample_id=sample.sample_id,
                source_row_id=sample.source_row_id,
                prompt_sha256=row.prompt_sha256,
                input_token_count=len(row.request.input_token_ids),
                input_token_ids_sha256=row.tokenized_input_sha256,
            )
        )
    if tuple(row.source_sample_id for row in result) != tuple(
        row.sample_id for row in workload.samples
    ):
        raise ValueError("LiveCodeBench tokenized hard-row order/coverage differs")
    return tuple(result)


def _publish_livecodebench_e1_e2_tokenizer_authority(
    *,
    root: Path,
    stage: str,
    content_source_binding: FormalContentSourceBinding,
    workload: FormalWorkloadAuthority,
    source: FormalServingRequestScheduleSource,
    rows: tuple[FormalServingRequestScheduleRow, ...],
    tokenizer_class: str,
    tokenizer_vocab_size: int,
    transformers_version: str,
) -> CanonicalJsonProofBinding | None:
    """Publish and enforce the hard-80 task-native authority before GPU use."""

    if stage not in {"E1", "E2"}:
        return None
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
        load_trusted_single_operator_content_bundle,
    )
    from lightcone_spec.experiments.formal_single_operator_lcb_tokenizer import (
        build_livecodebench_v6_hard_tokenizer_authority_from_revalidated_bundle,
        publish_livecodebench_v6_hard_tokenizer_authority,
        require_livecodebench_e1_e2_task_native_budget,
    )

    if source.workload_id != "livecodebench_v6_hard":
        raise ValueError("E1/E2 task-native schedule is not LiveCodeBench hard")
    # ``rebuild_trusted_single_operator_request_schedule_source`` immediately
    # above this boundary has already deep-reopened the same binding.  Loading
    # its immutable canonical bundle here avoids hashing multi-GB model trees a
    # second time solely to assemble a small derived sidecar.
    content_binding = content_source_binding.trusted_single_operator
    if content_binding is None:
        raise TypeError("E1/E2 tokenizer authority lacks trusted content binding")
    bundle = load_trusted_single_operator_content_bundle(content_binding.absolute_path)
    if (
        type(bundle) is not TrustedSingleOperatorContentBundle
        or bundle.semantic_sha256 != content_binding.semantic_sha256
        or bundle.runtime_binding_status != "BOUND"
    ):
        raise TypeError("E1/E2 tokenizer authority requires trusted content")
    locked = tuple(
        member
        for member in bundle.locked_workloads
        if member.workload_id == "livecodebench_v6_hard"
        and member.sha256 == source.trusted_workload_member_sha256
        and member.authority_sha256 == workload.sha256
    )
    tokenizers = tuple(
        member
        for member in bundle.model_members
        if member.role == "tokenizer"
        and {"E1", "E2"}.issubset(member.stages)
        and member.model_id == source.tokenizer_model_id
        and member.revision == source.tokenizer_revision
        and member.sha256 == source.tokenizer_content_member_id
    )
    if len(locked) != 1 or len(tokenizers) != 1:
        raise ValueError("E1/E2 tokenizer/workload content member is not exact")
    tokenized = _livecodebench_tokenized_prompt_rows(workload=workload, rows=rows)
    authority = build_livecodebench_v6_hard_tokenizer_authority_from_revalidated_bundle(
        content_bundle=bundle,
        workload_authority=workload,
        locked_workload=locked[0],
        tokenizer_member=tokenizers[0],
        tokenized_prompts=tokenized,
        tokenizer_class=tokenizer_class,
        tokenizer_vocab_size=tokenizer_vocab_size,
        transformers_version=transformers_version,
    )
    require_livecodebench_e1_e2_task_native_budget(
        authority,
        stage=stage,
        tokenized_prompts=tokenized,
        requested_output_tokens=tuple(
            row.request.requested_output_tokens for row in rows
        ),
    )
    return publish_livecodebench_v6_hard_tokenizer_authority(
        authority,
        root / "livecodebench-v6-hard-tokenizer-authority.json",
    )


def _revalidate_livecodebench_e1_e2_tokenizer_authority(
    *,
    receipt: FormalServingRequestScheduleReceipt,
    stage: str,
    bundle: object | None,
    workload: FormalWorkloadAuthority | None,
    source: FormalServingRequestScheduleSource,
) -> None:
    """Deep-replay the sidecar and its per-row budget during plan reopening."""

    expected = stage in {"E1", "E2"}
    binding = receipt.livecodebench_tokenizer_authority
    if not expected:
        if binding is not None:
            raise ValueError(
                "non-E1/E2 receipt carries LiveCodeBench tokenizer authority"
            )
        return
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
    )
    from lightcone_spec.experiments.formal_single_operator_lcb_tokenizer import (
        build_livecodebench_v6_hard_tokenizer_authority_from_revalidated_bundle,
        load_livecodebench_v6_hard_tokenizer_authority,
        require_livecodebench_e1_e2_task_native_budget,
    )

    if (
        type(binding) is not CanonicalJsonProofBinding
        or type(bundle) is not TrustedSingleOperatorContentBundle
        or type(workload) is not FormalWorkloadAuthority
        or source.workload_id != "livecodebench_v6_hard"
    ):
        raise ValueError("E1/E2 receipt lacks its LiveCodeBench tokenizer authority")
    locked = tuple(
        member
        for member in bundle.locked_workloads
        if member.workload_id == "livecodebench_v6_hard"
        and member.sha256 == source.trusted_workload_member_sha256
        and member.authority_sha256 == workload.sha256
    )
    tokenizers = tuple(
        member
        for member in bundle.model_members
        if member.role == "tokenizer"
        and {"E1", "E2"}.issubset(member.stages)
        and member.model_id == receipt.tokenizer_model_id
        and member.revision == receipt.tokenizer_revision
        and member.sha256 == receipt.tokenizer_content_member_id
    )
    if len(locked) != 1 or len(tokenizers) != 1:
        raise ValueError("E1/E2 tokenizer authority content member changed")
    rows = tuple(formal_serving_request_schedule_rows(receipt))
    tokenized = _livecodebench_tokenized_prompt_rows(workload=workload, rows=rows)
    observed = load_livecodebench_v6_hard_tokenizer_authority(binding)
    rebuilt = build_livecodebench_v6_hard_tokenizer_authority_from_revalidated_bundle(
        content_bundle=bundle,
        workload_authority=workload,
        locked_workload=locked[0],
        tokenizer_member=tokenizers[0],
        tokenized_prompts=tokenized,
        tokenizer_class=receipt.tokenizer_class,
        tokenizer_vocab_size=receipt.tokenizer_vocab_size,
        transformers_version=receipt.transformers_version,
    )
    if observed != rebuilt or observed.sha256 != rebuilt.sha256:
        raise ValueError("E1/E2 LiveCodeBench tokenizer authority changed")
    require_livecodebench_e1_e2_task_native_budget(
        observed,
        stage=stage,
        tokenized_prompts=tokenized,
        requested_output_tokens=tuple(
            row.request.requested_output_tokens for row in rows
        ),
    )


def _compile_trusted_controlled_context_source(
    *,
    source: FormalServingRequestScheduleSource,
    content_source_binding: FormalContentSourceBinding,
    context_filler_artifact: CanonicalJsonProofBinding,
) -> tuple[
    FormalServingRequestScheduleSource,
    tuple[CompiledContextRequest, ...],
    ContextFillerAuthority,
]:
    """Replace label-only E3b/E6 lengths with exact compiled token budgets."""

    from lightcone_spec.experiments.formal_single_operator_context_artifact import (
        load_trusted_context_filler_artifact,
    )

    if source.schema_version != 5 or not source.requests:
        raise ValueError("controlled context compilation requires populated schema 5")
    if source.regime not in {
        "long_input_short_output",
        "short_input_long_generation",
        "multi_turn_shared_prefix",
        "native_mtp_transfer",
    }:
        raise ValueError("controlled context schedule regime differs")
    authority = load_trusted_context_filler_artifact(
        context_filler_artifact.absolute_path,
        content_source_binding=content_source_binding,
        tokenizer_content_member_id=source.tokenizer_content_member_id,
        tokenizer_model_id=source.tokenizer_model_id,
        tokenizer_revision=source.tokenizer_revision,
    )
    by_key = {row.source_key: row for row in authority.rows}
    core_rows: list[TokenizedContextSourceRow] = []
    for row in source.requests:
        core = by_key.get((row.source_member_sha256, row.source_sample_id))
        if core is None or core.prompt_sha256 != row.prompt_sha256:
            raise ValueError("controlled context core row is outside filler authority")
        core_rows.append(core)
    compiled = compile_context_requests(
        regime=source.regime,  # type: ignore[arg-type]
        context_tokens=source.context_tokens,
        core_rows=tuple(core_rows),
        filler_authority=authority,
    )
    adjusted_rows: list[FormalServingRequestScheduleSourceRow] = []
    for row, compiled_row in zip(source.requests, compiled, strict=True):
        if (
            compiled_row.core_source_member_sha256 != row.source_member_sha256
            or compiled_row.core_source_sample_id != row.source_sample_id
            or compiled_row.core_prompt_sha256 != row.prompt_sha256
        ):
            raise ValueError("controlled context compiler changed core identity")
        sampling = dict(row.sampling)
        sampling["max_new_tokens"] = compiled_row.requested_output_tokens
        adjusted_rows.append(
            replace(
                row,
                requested_output_tokens=compiled_row.requested_output_tokens,
                sampling=FrozenSamplingParameters.from_mapping(sampling).items,
            )
        )
    return replace(source, requests=tuple(adjusted_rows)), compiled, authority


def _controlled_context_tokenizer_metadata(
    artifact_binding: CanonicalJsonProofBinding,
) -> tuple[str, int, str, str, str, int, str]:
    from lightcone_spec.experiments.formal_single_operator_context_artifact import (
        TrustedContextFillerArtifact,
    )

    artifact = TrustedContextFillerArtifact.from_dict(artifact_binding.reopen())
    if artifact_binding.semantic_sha256 != _sha256(artifact.to_dict()):
        raise ValueError("controlled context filler artifact identity differs")
    evidence = _strict_object(
        "controlled context tokenizer evidence",
        artifact.tokenization_evidence.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "source_identity_sha256",
            "tokenizer_worker_source_raw_sha256",
            "tokenizer_worker_source_size",
            "transformers_version",
            "tokenizer_class",
            "tokenizer_vocab_size",
            "batch_count",
            "batches",
        },
    )
    batches = evidence["batches"]
    if (
        type(batches) is not list
        or not batches
        or evidence["batch_count"] != len(batches)
        or any(type(row) is not dict for row in batches)
    ):
        raise ValueError("controlled context tokenizer batch coverage differs")
    argv_sha256s = tuple(
        _require_sha256(
            "controlled context tokenizer argv",
            row.get("tokenizer_worker_argv_sha256"),
        )
        for row in batches
    )
    return (
        _require_sha256(
            "controlled context tokenizer worker",
            evidence["tokenizer_worker_source_raw_sha256"],
        ),
        int(evidence["tokenizer_worker_source_size"]),
        _sha256({"ordered_batch_argv_sha256s": argv_sha256s}),
        str(evidence["transformers_version"]),
        str(evidence["tokenizer_class"]),
        int(evidence["tokenizer_vocab_size"]),
        artifact.tokenizer_snapshot_path,
    )


def _materialized_controlled_context_rows(
    *,
    source: FormalServingRequestScheduleSource,
    compiled: tuple[CompiledContextRequest, ...],
) -> tuple[FormalServingRequestScheduleRow, ...]:
    source_rows = tuple(formal_serving_request_schedule_source_rows(source))
    if source.schema_version != 7 or len(source_rows) != len(compiled):
        raise ValueError("controlled context materialization coverage differs")
    result: list[FormalServingRequestScheduleRow] = []
    for source_row, compiled_row in zip(source_rows, compiled, strict=True):
        if (
            compiled_row.core_source_member_sha256 != source_row.source_member_sha256
            or compiled_row.core_source_sample_id != source_row.source_sample_id
            or compiled_row.core_prompt_sha256 != source_row.prompt_sha256
            or compiled_row.requested_output_tokens
            != source_row.requested_output_tokens
        ):
            raise ValueError("controlled context source/compiler rows differ")
        immutable = ImmutableRequest.create(
            namespace=source_row.namespace,
            split=source_row.split,  # type: ignore[arg-type]
            ordinal=source_row.ordinal,
            template=RequestTemplate(
                input_token_ids=compiled_row.input_token_ids,
                requested_output_tokens=compiled_row.requested_output_tokens,
                sampling=FrozenSamplingParameters(items=source_row.sampling),
                cancellation_offset_us=source_row.cancellation_offset_us,
            ),
            arrival_us=source_row.arrival_us,
            cohort_id=source_row.cohort_id,
        )
        route_id = (
            f"dp{source_row.routed_dp_rank}"
            if source.topology_mode == "tp1_dp2"
            else source.topology_mode
        )
        result.append(
            FormalServingRequestScheduleRow(
                source_member_sha256=source_row.source_member_sha256,
                source_sample_id=source_row.source_sample_id,
                prompt_sha256=source_row.prompt_sha256,
                phase=source_row.phase,
                routed_dp_rank=source_row.routed_dp_rank,
                request=BoundServingRequest.create(immutable, route_id=route_id),
                tokenized_input_sha256=compiled_row.input_token_ids_sha256,
            )
        )
    return tuple(result)


def publish_trusted_controlled_context_schedule_source_shards(
    *,
    source: FormalServingRequestScheduleSource,
    uncompiled_source: CanonicalJsonProofBinding,
    context_filler_artifact: CanonicalJsonProofBinding,
    compiled: tuple[CompiledContextRequest, ...],
    compiled_output_directory: str | Path,
    source_output_directory: str | Path,
) -> FormalServingRequestScheduleSource:
    """Publish bounded compiler/source rows and return a schema-7 header."""

    if (
        source.schema_version != 5
        or not source.requests
        or len(source.requests) != len(compiled)
        or FormalServingRequestScheduleSource.from_dict(
            uncompiled_source.reopen()
        ).schema_version
        != 5
    ):
        raise ValueError("controlled context source inputs differ")
    request_count = len(source.requests)
    compiled_artifact_id = _trusted_controlled_context_compiled_rows_artifact_id(
        uncompiled_source=uncompiled_source,
        context_filler_artifact=context_filler_artifact,
        context_tokens=source.context_tokens,
        regime=source.regime,
        tokenizer_content_member_id=source.tokenizer_content_member_id,
        tokenizer_model_id=source.tokenizer_model_id,
        tokenizer_revision=source.tokenizer_revision,
        request_count=request_count,
    )
    compiled_binding, _compiled_index = publish_formal_canonical_sequence_shards(
        artifact_kind=(
            FORMAL_TRUSTED_CONTROLLED_CONTEXT_COMPILED_ROW_SHARD_ARTIFACT_KIND
        ),
        artifact_id=compiled_artifact_id,
        rows=tuple(row.to_dict() for row in compiled),
        output_directory=compiled_output_directory,
        maximum_shard_bytes=1_000_000,
        maximum_shard_rows=64,
    )
    source_artifact_id = _trusted_controlled_context_source_rows_artifact_id(
        uncompiled_source=uncompiled_source,
        context_filler_artifact=context_filler_artifact,
        compiled_context_requests_shard_index=compiled_binding,
        subject_sha256=source.subject_sha256,
        materialized_cell_id=source.materialized_cell_id,
        topology_mode=source.topology_mode,
        request_count=request_count,
    )
    source_binding, _source_index = publish_formal_canonical_sequence_shards(
        artifact_kind=(
            FORMAL_TRUSTED_CONTROLLED_CONTEXT_SOURCE_ROW_SHARD_ARTIFACT_KIND
        ),
        artifact_id=source_artifact_id,
        rows=tuple(row.to_dict() for row in source.requests),
        output_directory=source_output_directory,
        maximum_shard_bytes=256_000,
        maximum_shard_rows=128,
    )
    result = replace(
        source,
        schema_version=7,
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        derivation_protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        requests=(),
        requests_shard_index=source_binding,
        request_count=request_count,
        controlled_context_uncompiled_source=uncompiled_source,
        context_filler_artifact=context_filler_artifact,
        compiled_context_requests_shard_index=compiled_binding,
    )
    if (
        tuple(formal_serving_request_schedule_source_rows(result)) != source.requests
        or tuple(formal_serving_controlled_context_requests(result)) != compiled
    ):
        raise RuntimeError("controlled context source failed deep publication replay")
    return result


def publish_trusted_schedule_source_shards(
    *,
    source: FormalServingRequestScheduleSource,
    output_directory: str | Path,
) -> FormalServingRequestScheduleSource:
    """Publish schema-5 source rows and return a bounded schema-6 header."""

    if (
        type(source) is not FormalServingRequestScheduleSource
        or source.schema_version != 5
        or not source.requests
    ):
        raise ValueError("trusted source sharding requires populated schema 5")
    binding, _index = publish_formal_canonical_sequence_shards(
        artifact_kind=FORMAL_TRUSTED_SCHEDULE_SOURCE_ROW_SHARD_ARTIFACT_KIND,
        artifact_id=trusted_schedule_source_rows_artifact_id(source),
        rows=tuple(row.to_dict() for row in source.requests),
        output_directory=output_directory,
        maximum_shard_bytes=256_000,
        maximum_shard_rows=128,
    )
    result = replace(
        source,
        schema_version=6,
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SHARDED_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        derivation_protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SHARDED_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        requests=(),
        requests_shard_index=binding,
        request_count=len(source.requests),
    )
    tuple(formal_serving_request_schedule_source_rows(result))
    return result


def publish_trusted_schedule_receipt_shards(
    *,
    receipt: FormalServingRequestScheduleReceipt,
    output_directory: str | Path,
) -> FormalServingRequestScheduleReceipt:
    """Publish schema-5 tokenized rows and return a bounded schema-6 receipt."""

    if (
        type(receipt) is not FormalServingRequestScheduleReceipt
        or receipt.schema_version != 5
        or not receipt.requests
    ):
        raise ValueError("trusted receipt sharding requires populated schema 5")
    binding, _index = publish_formal_canonical_sequence_shards(
        artifact_kind=FORMAL_TRUSTED_SCHEDULE_RECEIPT_ROW_SHARD_ARTIFACT_KIND,
        artifact_id=trusted_schedule_receipt_rows_artifact_id(receipt),
        rows=tuple(row.to_dict() for row in receipt.requests),
        output_directory=output_directory,
        maximum_shard_bytes=1_500_000,
        maximum_shard_rows=128,
    )
    result = replace(
        receipt,
        schema_version=6,
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SHARDED_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        requests=(),
        requests_shard_index=binding,
        request_count=len(receipt.requests),
    )
    tuple(formal_serving_request_schedule_rows(result))
    return result


def publish_trusted_controlled_context_schedule_receipt_shards(
    *,
    receipt: FormalServingRequestScheduleReceipt,
    uncompiled_source: CanonicalJsonProofBinding,
    context_filler_artifact: CanonicalJsonProofBinding,
    compiled_context_requests_shard_index: CanonicalJsonProofBinding,
    output_directory: str | Path,
) -> FormalServingRequestScheduleReceipt:
    """Publish exact context-bound request rows and return schema-7 receipt."""

    if receipt.schema_version != 5 or not receipt.requests:
        raise ValueError("controlled context receipt requires populated schema 5")
    request_count = len(receipt.requests)
    artifact_id = _trusted_controlled_context_receipt_rows_artifact_id(
        execution_binding_sha256=receipt.execution_binding_sha256,
        subject_sha256=receipt.subject_sha256,
        materialized_cell_id=receipt.materialized_cell_id,
        schedule_source=receipt.schedule_source,
        uncompiled_source=uncompiled_source,
        context_filler_artifact=context_filler_artifact,
        compiled_context_requests_shard_index=(compiled_context_requests_shard_index),
        topology_mode=receipt.topology_mode,
        request_count=request_count,
    )
    binding, _index = publish_formal_canonical_sequence_shards(
        artifact_kind=(
            FORMAL_TRUSTED_CONTROLLED_CONTEXT_RECEIPT_ROW_SHARD_ARTIFACT_KIND
        ),
        artifact_id=artifact_id,
        rows=tuple(row.to_dict() for row in receipt.requests),
        output_directory=output_directory,
        maximum_shard_bytes=1_500_000,
        maximum_shard_rows=128,
    )
    result = replace(
        receipt,
        schema_version=7,
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        requests=(),
        requests_shard_index=binding,
        request_count=request_count,
        controlled_context_uncompiled_source=uncompiled_source,
        context_filler_artifact=context_filler_artifact,
        compiled_context_requests_shard_index=(compiled_context_requests_shard_index),
    )
    if tuple(formal_serving_request_schedule_rows(result)) != receipt.requests:
        raise RuntimeError("controlled context receipt failed publication replay")
    return result


def _publish_sharded_tokenization_index(
    *,
    path: Path,
    kind: str,
    schedule_source_sha256: str,
    batches: tuple[dict[str, object], ...],
) -> CanonicalJsonProofBinding:
    if (
        kind
        not in {
            "formal_serving_tokenization_input_index",
            "formal_serving_tokenization_output_index",
        }
        or not batches
    ):
        raise ValueError("formal sharded tokenization index differs")
    value = {
        "schema_version": 1,
        "kind": kind,
        "protocol_sha256": (
            TRUSTED_SINGLE_OPERATOR_SHARDED_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        "schedule_source_sha256": schedule_source_sha256,
        "batch_count": len(batches),
        "batches": list(batches),
    }
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _materialize_sharded_tokenization(
    *,
    root: Path,
    source: FormalServingRequestScheduleSource,
    launch: CompileLaunchManifest,
) -> tuple[
    tuple[FormalServingRequestScheduleRow, ...],
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    str,
    int,
    str,
    str,
    str,
    int,
]:
    if source.schema_version != 6 or source.requests_shard_index is None:
        raise ValueError("sharded tokenization requires schema-6 source")
    source_index = load_formal_canonical_sequence_shard_index(
        source.requests_shard_index.absolute_path,
        deep=False,
    )
    batch_root = root / "formal-tokenization-shards"
    batch_root.mkdir(mode=0o700)
    input_descriptors: list[dict[str, object]] = []
    output_descriptors: list[dict[str, object]] = []
    result_rows: list[FormalServingRequestScheduleRow] = []
    worker_identity: tuple[str, int] | None = None
    argv_sha256s: list[str] = []
    tokenizer_identity: tuple[str, str, int] | None = None
    for ordinal in range(source_index.shard_count):
        shard = source_index.reopen_shard(ordinal)
        source_rows = tuple(
            FormalServingRequestScheduleSourceRow.from_dict(row) for row in shard.rows
        )
        input_path = batch_root / f"input-{ordinal:06d}.json"
        output_path = batch_root / f"output-{ordinal:06d}.json"
        input_binding = _publish_tokenization_input_rows(
            path=input_path,
            schedule_source_sha256=source.sha256,
            rows=source_rows,
            launch=launch,
        )
        output_binding, worker_sha, worker_size, argv_sha = _invoke_tokenizer_worker(
            input_path=input_path,
            output_path=output_path,
        )
        batch_rows = _materialized_schedule_rows(
            source=source,
            launch=launch,
            tokenization_input=input_binding,
            tokenization_output=output_binding,
            source_rows=source_rows,
        )
        token_value = output_binding.reopen()
        current_worker = (worker_sha, worker_size)
        current_tokenizer = (
            str(token_value["transformers_version"]),
            str(token_value["tokenizer_class"]),
            int(token_value["tokenizer_vocab_size"]),
        )
        if worker_identity is None:
            worker_identity = current_worker
            tokenizer_identity = current_tokenizer
        elif (
            worker_identity != current_worker or tokenizer_identity != current_tokenizer
        ):
            raise ValueError(
                "formal sharded tokenizer identity changed between batches"
            )
        descriptor = {
            "batch_ordinal": ordinal,
            "start_ordinal": shard.start_ordinal,
            "end_ordinal_exclusive": shard.end_ordinal_exclusive,
        }
        input_descriptors.append(
            {**descriptor, "tokenization_input": input_binding.to_dict()}
        )
        output_descriptors.append(
            {
                **descriptor,
                "tokenization_output": output_binding.to_dict(),
                "tokenizer_worker_argv_sha256": argv_sha,
            }
        )
        argv_sha256s.append(argv_sha)
        result_rows.extend(batch_rows)
    assert worker_identity is not None and tokenizer_identity is not None
    input_index = _publish_sharded_tokenization_index(
        path=root / "formal-tokenization-input-index.json",
        kind="formal_serving_tokenization_input_index",
        schedule_source_sha256=source.sha256,
        batches=tuple(input_descriptors),
    )
    output_index = _publish_sharded_tokenization_index(
        path=root / "formal-tokenization-output-index.json",
        kind="formal_serving_tokenization_output_index",
        schedule_source_sha256=source.sha256,
        batches=tuple(output_descriptors),
    )
    return (
        tuple(result_rows),
        input_index,
        output_index,
        worker_identity[0],
        worker_identity[1],
        _sha256({"ordered_batch_argv_sha256s": argv_sha256s}),
        tokenizer_identity[0],
        tokenizer_identity[1],
        tokenizer_identity[2],
    )


def materialize_formal_serving_request_schedule(
    *,
    execution_binding: VerifiedFormalServingExecutionBinding,
    content_verification_receipt_path: str | Path,
    workload_authority_path: str | Path,
    materialization_path: str | Path,
    compile_launch_manifest_path: str | Path,
    private_output_root: str | Path,
    now_ns: int,
) -> FormalServingRequestScheduleReceipt:
    """Deep-open signed request text, tokenize it, and freeze exact requests."""

    verified = require_verified_formal_serving_execution_binding(execution_binding)
    subject = verified.subject
    content_binding = CanonicalJsonProofBinding.bind(content_verification_receipt_path)
    content_verification_receipt = ContentVerificationReceipt.from_dict(
        content_binding.reopen()
    )
    if (
        subject.content_verification_receipt_sha256
        != content_verification_receipt.sha256
        or content_binding.semantic_sha256 != content_verification_receipt.sha256
    ):
        raise ValueError(
            "formal request content receipt differs from execution binding"
        )
    content_verification_receipt.revalidate_formal_scope(current_ns=now_ns)
    root = _private_output_root(private_output_root)
    materialization_binding, materialization = _bind_stage_materialization(
        materialization_path
    )
    if (
        materialization.sha256 != subject.materialization_receipt_sha256
        or materialization.stage != subject.stage
    ):
        raise ValueError("formal request materialization differs from execution")
    cell = _materialized_cell(
        materialization,
        cell_id=subject.materialized_cell_id,
    )
    tts_window = (
        _tts_calibration_window(content_verification_receipt)
        if cell.stage == "TTS-Cal"
        else None
    )
    workload_id = (
        tts_window.entries[0].workload_id
        if tts_window is not None
        else _workload_id_for_cell(cell)
    )
    workload_binding, workload, workload_descriptor_sha256 = (
        _root_verified_workload_source(
            content_verification_receipt,
            workload_id=workload_id,
            workload_authority_path=workload_authority_path,
            current_ns=now_ns,
        )
    )
    launch = CompileLaunchManifest.load(compile_launch_manifest_path)
    launch_binding = CanonicalJsonProofBinding.bind(
        compile_launch_manifest_path,
        semantic_sha256=launch.sha256,
    )
    config = load_run_config(launch.run_config_path)
    sampling_profile = SamplingProfile.load(launch.sampling_profile_path)
    sampling_binding = CanonicalJsonProofBinding.bind(
        launch.sampling_profile_path,
        semantic_sha256=sampling_profile.sha256,
    )
    source = rebuild_formal_serving_request_schedule_source(
        subject_sha256=subject.sha256,
        workload_authority_sha256=subject.workload_authority_sha256,
        topology_mode=subject.topology_mode,
        materialization=materialization,
        materialized_cell_id=subject.materialized_cell_id,
        workload_source=workload,
        workload_source_descriptor_sha256=workload_descriptor_sha256,
        tts_tuning_window=tts_window,
        sampling_profile=sampling_profile,
        max_running_requests=config.runtime.max_running_requests,
        server_context_limit=config.runtime.context_length,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_content_authority_sha256=(launch.tokenizer_content_authority_sha256),
    )
    if (
        source.subject_sha256 != subject.sha256
        or source.materialized_cell_id != subject.materialized_cell_id
        or source.workload_authority_sha256 != subject.workload_authority_sha256
        or source.materialization_receipt_sha256
        != subject.materialization_receipt_sha256
        or source.workload_source_authority_sha256 != workload.sha256
        or source.sampling_profile_sha256 != config.runtime.sampling_profile_sha256
        or source.sampling_profile_sha256 != launch.sampling_profile_sha256
        or source.max_running_requests != config.runtime.max_running_requests
        or source.topology_mode != subject.topology_mode
        or source.tokenizer_content_member_id != launch.tokenizer_content_member_id
        or source.tokenizer_model_id != launch.tokenizer_model_id
        or source.tokenizer_revision != launch.tokenizer_revision
        or source.tokenizer_content_authority_sha256
        != launch.tokenizer_content_authority_sha256
        or run_config_sha256(config) != subject.run_config_sha256
        or config != verified.run_config
        or config.runtime.topology_mode != subject.topology_mode
        or launch.inventory_sha256 != subject.inventory_sha256
        or launch.gpu_uuids != subject.gpu_uuids
    ):
        raise ValueError("formal schedule/launch differs from sealed execution subject")
    allowed_workload_members = set(subject.workload_member_sha256s)
    if not allowed_workload_members:
        allowed_workload_members = {subject.workload_authority_sha256}
    if any(
        row.source_member_sha256 not in allowed_workload_members
        for row in source.requests
    ):
        raise ValueError("formal schedule row is outside verified workload content")
    if launch.tokenizer_content_authority_sha256 not in set(
        subject.prepared_model_member_sha256s
    ):
        raise ValueError("formal tokenizer is outside verified prepared-model content")
    input_path = root / "formal-tokenization-input.json"
    output_path = root / "formal-tokenization-output.json"
    source_path = root / "formal-request-schedule-source.json"
    receipt_path = root / "formal-request-schedule-receipt.json"
    if any(
        os.path.lexists(path)
        for path in (source_path, input_path, output_path, receipt_path)
    ):
        raise FileExistsError("formal request materialization output already exists")
    publish_canonical_json_no_replace(source_path, source.to_dict())
    source_artifact = ContentJsonArtifactBinding.from_path(
        f"{_SCHEDULE_ARTIFACT_PREFIX}{subject.materialized_cell_id}",
        source_path,
    )
    if source_artifact.semantic_sha256 != source.sha256:
        raise RuntimeError("derived formal request schedule source changed")
    token_input = _publish_tokenization_input(
        path=input_path,
        source=source,
        launch=launch,
    )
    token_output, worker_sha, worker_size, argv_sha = _invoke_tokenizer_worker(
        input_path=input_path,
        output_path=output_path,
    )
    rows = _materialized_schedule_rows(
        source=source,
        launch=launch,
        tokenization_input=token_input,
        tokenization_output=token_output,
    )
    token_value = token_output.reopen()
    receipt = FormalServingRequestScheduleReceipt(
        schema_version=4,
        kind="formal_serving_request_schedule_receipt",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=verified.sha256,
        subject_sha256=subject.sha256,
        materialized_cell_id=subject.materialized_cell_id,
        workload_authority_sha256=subject.workload_authority_sha256,
        content_verification_receipt_sha256=content_verification_receipt.sha256,
        topology_mode=subject.topology_mode,
        materialization=materialization_binding,
        content_verification_receipt=content_binding,
        workload_source=workload_binding,
        compile_launch_manifest=launch_binding,
        sampling_profile=sampling_binding,
        schedule_source=source_artifact,
        tokenization_input=token_input,
        tokenization_output=token_output,
        tokenizer_worker_source_raw_sha256=worker_sha,
        tokenizer_worker_source_size=worker_size,
        tokenizer_worker_argv_sha256=argv_sha,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_snapshot_path=launch.tokenizer_snapshot_path,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_content_authority_sha256=(launch.tokenizer_content_authority_sha256),
        transformers_version=str(token_value["transformers_version"]),
        tokenizer_class=str(token_value["tokenizer_class"]),
        tokenizer_vocab_size=int(token_value["tokenizer_vocab_size"]),
        requests=rows,
        e5_arrival_plan=source.e5_arrival_plan,
    )
    publish_canonical_json_no_replace(receipt_path, receipt.to_dict())
    rebound = FormalServingRequestScheduleReceipt.from_dict(
        CanonicalJsonProofBinding.bind(receipt_path).reopen()
    )
    if rebound != receipt or rebound.sha256 != receipt.sha256:
        raise RuntimeError("formal request schedule receipt failed replay")
    return receipt


def materialize_trusted_single_operator_request_schedule(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    compile_launch_manifest_path: str | Path,
    workload_source_path: str | Path,
    execution_binding_sha256: str,
    subject_sha256: str,
    private_output_root: str | Path,
    tts_calibration_authority_path: str | Path | None = None,
    e5_arrival_plan_path: str | Path | None = None,
    context_filler_artifact_path: str | Path | None = None,
) -> FormalServingRequestScheduleReceipt:
    """Publish one schema-5 schedule from current execution/content sources.

    The two supplied digests are the non-circular identities derived by the
    prepared-launch producer and are rechecked again by the prepared bundle.
    Every scientific value is rebuilt from path-bound current sources.
    """

    from lightcone_spec.experiments.formal_method_authority import (
        load_tts_calibration_authority_artifact,
    )
    from lightcone_spec.experiments.formal_registry import (
        protocol_lock_from_dict,
        stage_materialization_receipt_from_dict,
    )
    from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
        load_e0_task_native_source_authority,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        load_formal_single_operator_execution_source,
    )

    _require_sha256("trusted schedule execution binding", execution_binding_sha256)
    _require_sha256("trusted schedule subject", subject_sha256)
    source_execution = load_formal_single_operator_execution_source(
        execution_source_path
    )
    if (
        source_execution.schema_version != 3
        or type(source_execution.content_source_binding)
        is not FormalContentSourceBinding
        or source_execution.content_source_binding.mode != "trusted_single_operator"
    ):
        raise ValueError("trusted schedule execution source has another content mode")
    content_source = source_execution.content_source_binding
    protocol_lock = protocol_lock_from_dict(
        source_execution.protocol_lock_source.reopen(
            label="trusted schedule ProtocolLock"
        )
    )
    if (
        protocol_lock.schema_version != 5
        or protocol_lock.trusted_single_operator_content_bundle_sha256
        != content_source.content_sha256
    ):
        raise ValueError("trusted schedule ProtocolLock content differs")
    materialization_binding = CanonicalJsonProofBinding.bind(
        source_execution.materialization_source.absolute_path
    )
    materialization = stage_materialization_receipt_from_dict(
        materialization_binding.reopen()
    )
    if materialization.sha256 != source_execution.materialization_sha256:
        raise ValueError("trusted schedule materialization identity differs")
    cell = _materialized_cell(materialization, cell_id=materialized_cell_id)
    if cell.stage != source_execution.stage:
        raise ValueError("trusted schedule cell stage differs from execution source")
    controlled_context = cell.stage in {"E3b", "E6"}
    if cell.stage == "E6" and cell.task == (
        "immutable_metadata_interface_and_fit_preflight"
    ):
        raise ValueError("E6 interface preflight does not materialize serving requests")
    if controlled_context != (context_filler_artifact_path is not None):
        raise ValueError(
            "trusted schedule controlled-context artifact presence differs"
        )

    launch = CompileLaunchManifest.load(compile_launch_manifest_path)
    launch_binding = CanonicalJsonProofBinding.bind(
        compile_launch_manifest_path,
        semantic_sha256=launch.sha256,
    )
    if (
        launch.schema_version != 2
        or launch.content_source_binding != content_source
        or launch.formal_stage != cell.stage
    ):
        raise ValueError("trusted schedule compile launch lineage differs")
    config = load_run_config(launch.run_config_path)
    sampling = SamplingProfile.load(launch.sampling_profile_path)
    sampling_binding = CanonicalJsonProofBinding.bind(
        launch.sampling_profile_path,
        semantic_sha256=sampling.sha256,
    )
    if (
        config.runtime.topology_mode not in _TOPOLOGIES
        or run_config_sha256(config) != launch.run_config_semantic_sha256
        or sampling.sha256 != launch.sampling_profile_sha256
    ):
        raise ValueError("trusted schedule launch RunConfig changed")

    workload_id = _workload_id_for_cell(cell)
    if workload_id in {"livecodebench_v6_hard", "math500_level5"}:
        workload_binding = ContentJsonArtifactBinding.from_path(
            formal_workload_authority_artifact_id(workload_id),
            workload_source_path,
        )
        workload: object = formal_workload_authority_from_cli_artifact(
            workload_binding.load()
        )
    else:
        workload_binding = ContentJsonArtifactBinding.from_path(
            f"trusted_e0_task_native:{workload_id}",
            workload_source_path,
        )
        workload = load_e0_task_native_source_authority(workload_source_path)

    tts_window: TtsCalibrationTuningWindow | None = None
    tts_authority_binding: CanonicalJsonProofBinding | None = None
    trusted_tts_authority_sha256: str | None = None
    if cell.stage == "TTS-Cal":
        if tts_calibration_authority_path is None:
            raise ValueError("trusted TTS schedule lacks method authority")
        tts_authority_binding = CanonicalJsonProofBinding.bind(
            tts_calibration_authority_path
        )
        tts_artifact = load_tts_calibration_authority_artifact(
            tts_authority_binding.absolute_path
        )
        if (
            tts_authority_binding.semantic_sha256 != tts_artifact.sha256
            or tts_artifact.authority.sha256
            != protocol_lock.tts_calibration_authority_sha256
        ):
            raise ValueError("trusted TTS method authority differs from ProtocolLock")
        tts_window = TtsCalibrationTuningWindow.from_dict(
            tts_artifact.tuning_window_source.reopen()
        )
        if tts_window.sha256 != tts_artifact.authority.tuning_window_sha256:
            raise ValueError("trusted TTS tuning window changed")
        trusted_tts_authority_sha256 = tts_artifact.authority.sha256
    elif tts_calibration_authority_path is not None:
        raise ValueError("non-TTS trusted schedule carries TTS method authority")

    e5_binding = (
        None
        if e5_arrival_plan_path is None
        else CanonicalJsonProofBinding.bind(e5_arrival_plan_path)
    )
    if (cell.stage == "E5" and cell.task == "production_slo_power_prefix") != (
        e5_binding is not None
    ):
        raise ValueError("trusted schedule E5 arrival-plan presence differs")
    schedule_source = rebuild_trusted_single_operator_request_schedule_source(
        subject_sha256=subject_sha256,
        content_source_binding=content_source,
        topology_mode=config.runtime.topology_mode,
        materialization=materialization,
        materialized_cell_id=cell.cell_id,
        workload_source=workload,
        workload_source_binding=workload_binding,
        sampling_profile=sampling,
        max_running_requests=config.runtime.max_running_requests,
        server_context_limit=config.runtime.context_length,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tts_tuning_window=tts_window,
        trusted_tts_calibration_authority_sha256=(trusted_tts_authority_sha256),
        e5_arrival_plan=e5_binding,
    )

    root = _private_output_root(private_output_root)
    source_path = root / "formal-request-schedule-source.json"
    uncompiled_source_path = root / "formal-request-schedule-uncompiled-source.json"
    input_path = root / "formal-tokenization-input.json"
    output_path = root / "formal-tokenization-output.json"
    receipt_path = root / "formal-request-schedule-receipt.json"
    if any(
        os.path.lexists(path)
        for path in (
            source_path,
            input_path,
            output_path,
            receipt_path,
            uncompiled_source_path,
        )
    ):
        raise FileExistsError("trusted request schedule output already exists")
    context_filler_binding: CanonicalJsonProofBinding | None = None
    compiled_context: tuple[CompiledContextRequest, ...] | None = None
    uncompiled_source_binding: CanonicalJsonProofBinding | None = None
    if controlled_context:
        assert context_filler_artifact_path is not None
        context_filler_binding = CanonicalJsonProofBinding.bind(
            context_filler_artifact_path
        )
        publish_canonical_json_no_replace(
            uncompiled_source_path,
            schedule_source.to_dict(),
        )
        uncompiled_source_binding = CanonicalJsonProofBinding.bind(
            uncompiled_source_path,
        )
        schedule_source, compiled_context, _filler_authority = (
            _compile_trusted_controlled_context_source(
                source=schedule_source,
                content_source_binding=content_source,
                context_filler_artifact=context_filler_binding,
            )
        )
        source_row_root = root / "formal-request-schedule-source-rows"
        receipt_row_root = root / "formal-request-schedule-receipt-rows"
        compiled_row_root = root / "formal-request-schedule-compiled-context-rows"
        source_row_root.mkdir(mode=0o700)
        receipt_row_root.mkdir(mode=0o700)
        compiled_row_root.mkdir(mode=0o700)
        schedule_source = publish_trusted_controlled_context_schedule_source_shards(
            source=schedule_source,
            uncompiled_source=uncompiled_source_binding,
            context_filler_artifact=context_filler_binding,
            compiled=compiled_context,
            compiled_output_directory=compiled_row_root,
            source_output_directory=source_row_root,
        )
        sharded = True
    else:
        sharded = (
            len(schedule_source.requests) >= 2_048
            or len(_canonical_bytes(schedule_source.to_dict())) + 1 > 1_250_000
        )
    if sharded and not controlled_context:
        source_row_root = root / "formal-request-schedule-source-rows"
        receipt_row_root = root / "formal-request-schedule-receipt-rows"
        source_row_root.mkdir(mode=0o700)
        receipt_row_root.mkdir(mode=0o700)
        schedule_source = publish_trusted_schedule_source_shards(
            source=schedule_source,
            output_directory=source_row_root,
        )
    publish_canonical_json_no_replace(source_path, schedule_source.to_dict())
    schedule_binding = ContentJsonArtifactBinding.from_path(
        f"{_SCHEDULE_ARTIFACT_PREFIX}{cell.cell_id}",
        source_path,
    )
    if schedule_binding.semantic_sha256 != schedule_source.sha256:
        raise RuntimeError("trusted request schedule source changed")
    tokenizer_snapshot_path = launch.tokenizer_snapshot_path
    if controlled_context:
        assert context_filler_binding is not None
        assert compiled_context is not None
        assert schedule_source.compiled_context_requests_shard_index is not None
        rows = _materialized_controlled_context_rows(
            source=schedule_source,
            compiled=compiled_context,
        )
        token_input = context_filler_binding
        token_output = schedule_source.compiled_context_requests_shard_index
        (
            worker_sha,
            worker_size,
            argv_sha,
            transformers_version,
            tokenizer_class,
            tokenizer_vocab_size,
            tokenizer_snapshot_path,
        ) = _controlled_context_tokenizer_metadata(context_filler_binding)
        if tokenizer_snapshot_path != launch.tokenizer_snapshot_path:
            raise ValueError("controlled context tokenizer snapshot path changed")
    elif sharded:
        (
            rows,
            token_input,
            token_output,
            worker_sha,
            worker_size,
            argv_sha,
            transformers_version,
            tokenizer_class,
            tokenizer_vocab_size,
        ) = _materialize_sharded_tokenization(
            root=root,
            source=schedule_source,
            launch=launch,
        )
    else:
        token_input = _publish_tokenization_input(
            path=input_path,
            source=schedule_source,
            launch=launch,
        )
        token_output, worker_sha, worker_size, argv_sha = _invoke_tokenizer_worker(
            input_path=input_path,
            output_path=output_path,
        )
        rows = _materialized_schedule_rows(
            source=schedule_source,
            launch=launch,
            tokenization_input=token_input,
            tokenization_output=token_output,
        )
        token_value = token_output.reopen()
        transformers_version = str(token_value["transformers_version"])
        tokenizer_class = str(token_value["tokenizer_class"])
        tokenizer_vocab_size = int(token_value["tokenizer_vocab_size"])
    livecodebench_tokenizer_authority = (
        _publish_livecodebench_e1_e2_tokenizer_authority(
            root=root,
            stage=cell.stage,
            content_source_binding=content_source,
            workload=workload,
            source=schedule_source,
            rows=rows,
            tokenizer_class=tokenizer_class,
            tokenizer_vocab_size=tokenizer_vocab_size,
            transformers_version=transformers_version,
        )
        if type(workload) is FormalWorkloadAuthority
        else None
    )
    if cell.stage in {"E1", "E2"} and livecodebench_tokenizer_authority is None:
        raise ValueError("E1/E2 schedule lacks its LiveCodeBench tokenizer authority")
    receipt = FormalServingRequestScheduleReceipt(
        schema_version=5,
        kind="formal_serving_request_schedule_receipt",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        formal_execution_authorized=False,
        execution_binding_sha256=execution_binding_sha256,
        subject_sha256=subject_sha256,
        materialized_cell_id=cell.cell_id,
        workload_authority_sha256=None,
        content_verification_receipt_sha256=None,
        topology_mode=config.runtime.topology_mode,
        materialization=materialization_binding,
        content_verification_receipt=None,
        workload_source=workload_binding,
        compile_launch_manifest=launch_binding,
        sampling_profile=sampling_binding,
        schedule_source=schedule_binding,
        tokenization_input=token_input,
        tokenization_output=token_output,
        tokenizer_worker_source_raw_sha256=worker_sha,
        tokenizer_worker_source_size=worker_size,
        tokenizer_worker_argv_sha256=argv_sha,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_snapshot_path=tokenizer_snapshot_path,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_content_authority_sha256=None,
        transformers_version=transformers_version,
        tokenizer_class=tokenizer_class,
        tokenizer_vocab_size=tokenizer_vocab_size,
        requests=rows,
        e5_arrival_plan=e5_binding,
        content_source_binding=content_source,
        trusted_workload_member_sha256=(schedule_source.trusted_workload_member_sha256),
        trusted_tts_calibration_authority=tts_authority_binding,
        livecodebench_tokenizer_authority=livecodebench_tokenizer_authority,
    )
    if controlled_context:
        assert uncompiled_source_binding is not None
        assert context_filler_binding is not None
        assert schedule_source.compiled_context_requests_shard_index is not None
        receipt = publish_trusted_controlled_context_schedule_receipt_shards(
            receipt=receipt,
            uncompiled_source=uncompiled_source_binding,
            context_filler_artifact=context_filler_binding,
            compiled_context_requests_shard_index=(
                schedule_source.compiled_context_requests_shard_index
            ),
            output_directory=receipt_row_root,
        )
    elif sharded:
        receipt = publish_trusted_schedule_receipt_shards(
            receipt=receipt,
            output_directory=receipt_row_root,
        )
    publish_canonical_json_no_replace(receipt_path, receipt.to_dict())
    rebound = FormalServingRequestScheduleReceipt.from_dict(
        CanonicalJsonProofBinding.bind(receipt_path).reopen()
    )
    if rebound != receipt or rebound.sha256 != receipt.sha256:
        raise RuntimeError("trusted request schedule receipt failed replay")
    rebound.reopen()
    return receipt


def _materialize_single_operator_direct_schedule(
    *,
    inputs: object,
    preflight_inputs: object,
    input_binding: CanonicalJsonProofBinding,
    launch: CompileLaunchManifest,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    subject_sha256: str,
) -> FormalServingRequestScheduleReceipt:
    """Create a local request schedule without replaying content signatures."""

    root = _private_output_root(inputs.private_output_root)
    content = ContentVerificationReceipt.from_dict(
        preflight_inputs.content_receipt.reopen()
    )
    workload = formal_workload_authority_from_cli_artifact(
        preflight_inputs.workload_authority.load()
    )
    tts_window = _tts_calibration_window(content) if cell.stage == "TTS-Cal" else None
    workload_descriptor_sha256 = (
        tts_window.entries[0].source_descriptor_sha256
        if tts_window is not None
        else workload.sha256
    )
    config = load_run_config(launch.run_config_path)
    sampling = SamplingProfile.load(launch.sampling_profile_path)
    source = rebuild_formal_serving_request_schedule_source(
        subject_sha256=subject_sha256,
        workload_authority_sha256=workload.sha256,
        topology_mode=config.runtime.topology_mode,
        materialization=materialization,
        materialized_cell_id=cell.cell_id,
        workload_source=workload,
        workload_source_descriptor_sha256=workload_descriptor_sha256,
        tts_tuning_window=tts_window,
        sampling_profile=sampling,
        max_running_requests=config.runtime.max_running_requests,
        server_context_limit=config.runtime.context_length,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_content_authority_sha256=(launch.tokenizer_content_authority_sha256),
    )
    source_path = root / "formal-request-schedule-source.json"
    input_path = root / "formal-tokenization-input.json"
    output_path = root / "formal-tokenization-output.json"
    receipt_path = root / "formal-request-schedule-receipt.json"
    if any(
        os.path.lexists(path)
        for path in (source_path, input_path, output_path, receipt_path)
    ):
        raise FileExistsError("single-operator request schedule already exists")
    publish_canonical_json_no_replace(source_path, source.to_dict())
    source_artifact = ContentJsonArtifactBinding.from_path(
        f"{_SCHEDULE_ARTIFACT_PREFIX}{cell.cell_id}",
        source_path,
    )
    token_input = _publish_tokenization_input(
        path=input_path,
        source=source,
        launch=launch,
    )
    token_output, worker_sha, worker_size, argv_sha = _invoke_tokenizer_worker(
        input_path=input_path,
        output_path=output_path,
    )
    requests = _materialized_schedule_rows(
        source=source,
        launch=launch,
        tokenization_input=token_input,
        tokenization_output=token_output,
    )
    token_value = token_output.reopen()
    receipt = FormalServingRequestScheduleReceipt(
        schema_version=4,
        kind="formal_serving_request_schedule_receipt",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=input_binding.semantic_sha256,
        subject_sha256=subject_sha256,
        materialized_cell_id=cell.cell_id,
        workload_authority_sha256=workload.sha256,
        content_verification_receipt_sha256=content.sha256,
        topology_mode=config.runtime.topology_mode,
        materialization=inputs.materialization,
        content_verification_receipt=preflight_inputs.content_receipt,
        workload_source=preflight_inputs.workload_authority,
        compile_launch_manifest=inputs.compile_launch_manifest,
        sampling_profile=CanonicalJsonProofBinding.bind(
            launch.sampling_profile_path,
            semantic_sha256=sampling.sha256,
        ),
        schedule_source=source_artifact,
        tokenization_input=token_input,
        tokenization_output=token_output,
        tokenizer_worker_source_raw_sha256=worker_sha,
        tokenizer_worker_source_size=worker_size,
        tokenizer_worker_argv_sha256=argv_sha,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_snapshot_path=launch.tokenizer_snapshot_path,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_content_authority_sha256=(launch.tokenizer_content_authority_sha256),
        transformers_version=str(token_value["transformers_version"]),
        tokenizer_class=str(token_value["tokenizer_class"]),
        tokenizer_vocab_size=int(token_value["tokenizer_vocab_size"]),
        requests=requests,
        e5_arrival_plan=source.e5_arrival_plan,
    )
    publish_canonical_json_no_replace(receipt_path, receipt.to_dict())
    rebound = FormalServingRequestScheduleReceipt.from_dict(
        CanonicalJsonProofBinding.bind(receipt_path).reopen()
    )
    if rebound != receipt or rebound.sha256 != receipt.sha256:
        raise RuntimeError("single-operator request schedule changed")
    return receipt


def _native_binding_to_dict(value: NativeTerminalRunBinding) -> dict[str, object]:
    value.validate()
    return {
        "run_id": value.run_id,
        "run_nonce_sha256": value.run_nonce_sha256,
        "execution_plan_sha256": value.execution_plan_sha256,
        "rank_config_sha256": value.rank_config_sha256,
        "attempt_id": value.attempt_id,
        "session_id": value.session_id,
        "session_epoch": value.session_epoch,
        "previous_run_id": value.previous_run_id,
        "challenge_nonce_sha256": value.challenge_nonce_sha256,
        "method": value.method,
        "warmup_request_ids": list(value.warmup_request_ids),
        "scored_request_ids": list(value.scored_request_ids),
    }


def _native_binding_from_dict(value: object) -> NativeTerminalRunBinding:
    row = _strict_object(
        "formal physical native binding",
        value,
        {
            "run_id",
            "run_nonce_sha256",
            "execution_plan_sha256",
            "rank_config_sha256",
            "attempt_id",
            "session_id",
            "session_epoch",
            "previous_run_id",
            "challenge_nonce_sha256",
            "method",
            "warmup_request_ids",
            "scored_request_ids",
        },
    )
    warmup = row.pop("warmup_request_ids")
    scored = row.pop("scored_request_ids")
    if type(warmup) is not list or type(scored) is not list:
        raise TypeError("formal physical native request IDs must be arrays")
    result = NativeTerminalRunBinding(
        **row,
        warmup_request_ids=tuple(warmup),
        scored_request_ids=tuple(scored),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class FormalSingleOperatorExecutionRebuildSource:
    """Durable public inputs for rebuilding one private execution binding."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_execution_rebuild_source"]
    protocol_sha256: str
    execution_binding_sha256: str
    subject_sha256: str
    materialized_cell_id: str
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    formal_runtime_authority_manifest: CanonicalJsonProofBinding
    compile_launch_manifest: CanonicalJsonProofBinding
    inventory: CanonicalJsonProofBinding
    content_verification_receipt: CanonicalJsonProofBinding
    runtime_gpu_proof_artifacts: tuple[CanonicalJsonProofBinding, ...]
    tts_calibration_authority: CanonicalJsonProofBinding | None
    e1_recipe_anchor_authority: CanonicalJsonProofBinding | None
    formal_registry_verification_receipt: CanonicalJsonProofBinding | None
    repository_root: str | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_execution_rebuild_source"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_EXECUTION_REBUILD_SOURCE_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator execution rebuild source schema differs")
        for label, value in (
            ("execution binding", self.execution_binding_sha256),
            ("execution subject", self.subject_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("execution source", self.execution_source_sha256),
        ):
            _require_sha256(f"single-operator rebuild {label}", value)
        required = (
            self.execution_source,
            self.formal_runtime_authority_manifest,
            self.compile_launch_manifest,
            self.inventory,
            self.content_verification_receipt,
        )
        optional = (
            self.tts_calibration_authority,
            self.e1_recipe_anchor_authority,
            self.formal_registry_verification_receipt,
        )
        if any(type(value) is not CanonicalJsonProofBinding for value in required):
            raise TypeError("single-operator rebuild required source is not path-bound")
        if any(
            value is not None and type(value) is not CanonicalJsonProofBinding
            for value in optional
        ):
            raise TypeError("single-operator rebuild optional source is not path-bound")
        if (
            type(self.runtime_gpu_proof_artifacts) is not tuple
            or not self.runtime_gpu_proof_artifacts
            or any(
                type(value) is not CanonicalJsonProofBinding
                for value in self.runtime_gpu_proof_artifacts
            )
            or tuple(
                sorted(
                    value.semantic_sha256 for value in self.runtime_gpu_proof_artifacts
                )
            )
            != tuple(
                value.semantic_sha256 for value in self.runtime_gpu_proof_artifacts
            )
            or len(
                {value.semantic_sha256 for value in self.runtime_gpu_proof_artifacts}
            )
            != len(self.runtime_gpu_proof_artifacts)
        ):
            raise ValueError(
                "single-operator rebuild runtime proof set is not canonical"
            )
        for binding in (*required, *self.runtime_gpu_proof_artifacts, *optional):
            if binding is not None and (
                CanonicalJsonProofBinding.bind(binding.absolute_path) != binding
            ):
                raise ValueError("single-operator rebuild source changed")
        if self.repository_root is not None:
            requested = Path(self.repository_root)
            resolved = requested.resolve(strict=False)
            if (
                not requested.is_absolute()
                or requested != resolved
                or not resolved.is_dir()
                or resolved.is_symlink()
            ):
                raise ValueError(
                    "single-operator rebuild repository root is not a real "
                    "absolute directory"
                )

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "execution_binding_sha256": self.execution_binding_sha256,
            "subject_sha256": self.subject_sha256,
            "materialized_cell_id": self.materialized_cell_id,
            "execution_source": self.execution_source.to_dict(),
            "execution_source_sha256": self.execution_source_sha256,
            "formal_runtime_authority_manifest": (
                self.formal_runtime_authority_manifest.to_dict()
            ),
            "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
            "inventory": self.inventory.to_dict(),
            "content_verification_receipt": (
                self.content_verification_receipt.to_dict()
            ),
            "runtime_gpu_proof_artifacts": [
                value.to_dict() for value in self.runtime_gpu_proof_artifacts
            ],
            "tts_calibration_authority": (
                None
                if self.tts_calibration_authority is None
                else self.tts_calibration_authority.to_dict()
            ),
            "e1_recipe_anchor_authority": (
                None
                if self.e1_recipe_anchor_authority is None
                else self.e1_recipe_anchor_authority.to_dict()
            ),
            "formal_registry_verification_receipt": (
                None
                if self.formal_registry_verification_receipt is None
                else self.formal_registry_verification_receipt.to_dict()
            ),
            "repository_root": self.repository_root,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "single-operator execution rebuild source",
            value,
            {field.name for field in fields(cls)},
        )
        runtime_proofs = row.pop("runtime_gpu_proof_artifacts")
        if type(runtime_proofs) is not list:
            raise TypeError("single-operator rebuild runtime proofs must be an array")
        for name in (
            "execution_source",
            "formal_runtime_authority_manifest",
            "compile_launch_manifest",
            "inventory",
            "content_verification_receipt",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in (
            "tts_calibration_authority",
            "e1_recipe_anchor_authority",
            "formal_registry_verification_receipt",
        ):
            if row[name] is not None:
                row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(
            **row,
            runtime_gpu_proof_artifacts=tuple(
                CanonicalJsonProofBinding.from_dict(item) for item in runtime_proofs
            ),
        )


def _registered_serving_execution_policy(
    *,
    stage: str,
    schedule: FormalServingRequestScheduleReceipt,
) -> RegisteredServingExecutionPolicy | None:
    """Rebuild the current request timing contract from source-owned inputs."""

    if schedule.schema_version not in {4, 5, 6, 7}:
        return None
    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
    )

    if stage == "E5" and schedule.e5_arrival_plan is None:
        raise ValueError("E5 serving policy lacks its path-bound arrival plan")
    source = FormalServingRequestScheduleSource.from_dict(
        schedule.schedule_source.load()
    )
    if source.sha256 != schedule.schedule_source.semantic_sha256:
        raise ValueError("registered serving schedule source changed")
    if stage == "E5":
        assert schedule.e5_arrival_plan is not None
        from lightcone_spec.experiments.formal_single_operator_loads import (
            E5ArrivalPlan,
        )

        arrival = E5ArrivalPlan.from_dict(schedule.e5_arrival_plan.reopen())
        if (
            arrival.sha256 != schedule.e5_arrival_plan.semantic_sha256
            or source.e5_arrival_plan != schedule.e5_arrival_plan
            or source.load_protocol_sha256 != arrival.sha256
            or source.arrival_policy != arrival.arrival_policy
            or source.max_running_requests != arrival.concurrency
        ):
            raise ValueError("registered E5 serving policy differs from schedule")
        policy = RegisteredServingExecutionPolicy(
            schema_version=1,
            kind="registered_serving_execution_policy",
            source_kind=(
                "closed_loop"
                if arrival.arrival_policy in {"closed_loop", "closed_loop_zero_think"}
                else "scheduled"
            ),
            warmup_duration_us=arrival.warmup_duration_us,
            arrival_duration_us=arrival.arrival_duration_us,
            request_deadline_us=arrival.request_deadline_us,
            drain_duration_us=arrival.drain_duration_us,
            max_concurrency=arrival.concurrency,
            complete_closed_loop_pool=(
                arrival.p99_extension_minimum_completed is not None
            ),
        )
    else:
        if schedule.e5_arrival_plan is not None:
            raise ValueError("non-headline serving policy carries an E5 arrival plan")
        policy = RegisteredServingExecutionPolicy(
            schema_version=1,
            kind="registered_serving_execution_policy",
            source_kind=(
                "closed_loop"
                if source.arrival_policy
                in {"closed_loop_zero_think", "closed_loop", "closed_loop_c1"}
                else "scheduled"
            ),
            warmup_duration_us=_CURRENT_NON_E5_WARMUP_DURATION_US,
            arrival_duration_us=_CURRENT_NON_E5_ARRIVAL_DURATION_US,
            request_deadline_us=_CURRENT_NON_E5_REQUEST_DEADLINE_US,
            drain_duration_us=_CURRENT_NON_E5_DRAIN_DURATION_US,
            max_concurrency=source.max_running_requests,
            # Current non-E5 stages bind a finite task/request set rather than
            # an arrival-rate experiment, so every registered row is offered.
            complete_closed_loop_pool=(
                source.arrival_policy
                in {"closed_loop_zero_think", "closed_loop", "closed_loop_c1"}
            ),
        )
    policy.__post_init__()
    return policy


def _registered_process_hard_timeout_ns(
    *,
    policy: RegisteredServingExecutionPolicy,
    schedule: FormalServingRequestScheduleReceipt,
) -> int:
    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
    )

    if type(policy) is not RegisteredServingExecutionPolicy:
        raise TypeError("registered process timeout requires an exact policy")
    rows = tuple(formal_serving_request_schedule_rows(schedule))
    warmup_count = sum(row.phase == "warmup" for row in rows)
    scored_count = sum(row.phase == "scored" for row in rows)
    if warmup_count < 1 or scored_count < 1:
        raise ValueError("registered process timeout lacks phase coverage")
    # A deadline is followed by the source-owned abort request and then a
    # bounded wait for the submitted request's native terminal.  Both network
    # operations use the pinned abort timeout, so every sequential closed-loop
    # wave must reserve both intervals rather than ending at the client
    # deadline itself.
    abort_reconciliation_us = int(_ABORT_TIMEOUT_SECONDS * 2 * 1_000_000)
    if policy.complete_closed_loop_pool:
        warmup_waves = (warmup_count + policy.max_concurrency - 1) // (
            policy.max_concurrency
        )
        scored_waves = (scored_count + policy.max_concurrency - 1) // (
            policy.max_concurrency
        )
        request_execution_us = (warmup_waves + scored_waves) * (
            policy.request_deadline_us + abort_reconciliation_us
        )
    else:
        request_execution_us = (
            policy.minimum_process_timeout_us + abort_reconciliation_us
        )
    timeout_us = (
        request_execution_us
        + _CURRENT_PROCESS_STARTUP_RESERVE_US
        + _CURRENT_PROCESS_CLEANUP_RESERVE_US
    )
    timeout_ns = timeout_us * 1_000
    if (
        timeout_ns < policy.minimum_process_timeout_us * 1_000
        or timeout_ns > _MAX_CURRENT_PROCESS_TIMEOUT_NS
    ):
        raise ValueError("registered process hard timeout is outside source bounds")
    return timeout_ns


def _registered_plan_process_hard_timeout_seconds(
    plan: object,
    *,
    allowed_topologies: frozenset[str],
) -> float:
    """Project a deep-validated schema-4 cap into an inner runner timeout."""

    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
    )

    if (
        getattr(plan, "schema_version", None) != 4
        or getattr(plan, "topology_mode", None) not in allowed_topologies
        or type(getattr(plan, "serving_execution_policy", None))
        is not RegisteredServingExecutionPolicy
        or type(getattr(plan, "process_hard_timeout_ns", None)) is not int
    ):
        raise ValueError("current physical plan lacks its registered timing contract")
    policy = plan.serving_execution_policy
    timeout_ns = plan.process_hard_timeout_ns
    if (
        timeout_ns < policy.minimum_process_timeout_us * 1_000
        or timeout_ns > _MAX_CURRENT_PROCESS_TIMEOUT_NS
    ):
        raise ValueError("current physical plan hard timeout is outside source bounds")
    return timeout_ns / 1_000_000_000


@dataclass(frozen=True)
class FormalServingRunPlan:
    """Durable, non-authorizing physical plan rebuilt against a sealed token."""

    schema_version: Literal[1, 2, 3, 4]
    kind: Literal["formal_serving_run_plan"]
    protocol_sha256: str
    formal_execution_authorized: Literal[False]
    execution_binding_sha256: str
    subject_sha256: str
    materialized_cell_id: str
    stage: str
    method: str
    topology_mode: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    runtime_gpu_proof_sha256s: tuple[str, ...]
    runtime_gpu_proof_artifacts: tuple[CanonicalJsonProofBinding, ...]
    nextn_tp2_authority_sha256: str | None
    launch_manifest: CanonicalJsonProofBinding
    request_schedule_receipt: CanonicalJsonProofBinding
    native_terminal_binding: NativeTerminalRunBinding
    private_output_root: str
    terminal_output_path: str
    native_itl_pointer_output_path: str
    live_run_receipt_output_path: str
    lifecycle_timing_output_path: str
    server_log_output_path: str
    server_stdout_output_path: str
    server_stderr_output_path: str
    junit_output_path: str
    before_gpu_snapshot_output_path: str
    ready_gpu_snapshot_output_path: str
    after_gpu_snapshot_output_path: str
    formal_gang_terminal_output_path: str | None
    fatal_output_path: str
    single_operator_execution_rebuild_source: CanonicalJsonProofBinding | None = None
    nextn_mtp_mode: Literal["built_in_mtp"] | None = None
    target_snapshot_sha256: str | None = None
    mtp_component_sha256: str | None = None
    mtp_component: CanonicalJsonProofBinding | None = None
    serving_execution_policy: RegisteredServingExecutionPolicy | None = None
    process_hard_timeout_ns: int | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2, 3, 4}
            or self.kind != "formal_serving_run_plan"
            or self.protocol_sha256 != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
            or self.formal_execution_authorized is not False
        ):
            raise ValueError("formal physical run-plan schema differs")
        if self.schema_version == 4:
            from lightcone_spec.orchestration.executor import (
                RegisteredServingExecutionPolicy,
            )

            if (
                type(self.serving_execution_policy)
                is not RegisteredServingExecutionPolicy
                or type(self.process_hard_timeout_ns) is not int
                or self.process_hard_timeout_ns < 1
                or self.process_hard_timeout_ns > _MAX_CURRENT_PROCESS_TIMEOUT_NS
                or self.process_hard_timeout_ns
                < self.serving_execution_policy.minimum_process_timeout_us * 1_000
            ):
                raise ValueError("current physical plan execution timeout differs")
        elif (
            self.serving_execution_policy is not None
            or self.process_hard_timeout_ns is not None
        ):
            raise ValueError("legacy physical plan carries current timing policy")
        if self.schema_version == 1:
            if self.single_operator_execution_rebuild_source is not None:
                raise ValueError("legacy formal plan carries a current rebuild source")
        elif (
            type(self.single_operator_execution_rebuild_source)
            is not CanonicalJsonProofBinding
        ):
            raise ValueError("current formal plan lacks its execution rebuild source")
        for label, value in (
            ("execution binding", self.execution_binding_sha256),
            ("subject", self.subject_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"formal physical plan {label}", value)
        for label, value in (
            ("stage", self.stage),
            ("method", self.method),
        ):
            _require_text(f"formal physical plan {label}", value)
        if self.topology_mode not in _TOPOLOGIES:
            raise ValueError("formal physical plan topology differs")
        expected_gpus = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpus
            or len(set(self.gpu_uuids)) != expected_gpus
            or type(self.launch_manifest) is not CanonicalJsonProofBinding
            or type(self.request_schedule_receipt) is not CanonicalJsonProofBinding
            or type(self.native_terminal_binding) is not NativeTerminalRunBinding
        ):
            raise ValueError("formal physical plan GPU/path/run binding differs")
        if self.single_operator_execution_rebuild_source is not None and (
            CanonicalJsonProofBinding.bind(
                self.single_operator_execution_rebuild_source.absolute_path
            )
            != self.single_operator_execution_rebuild_source
        ):
            raise ValueError("formal physical plan execution rebuild source changed")
        if (
            type(self.runtime_gpu_proof_sha256s) is not tuple
            or self.runtime_gpu_proof_sha256s
            != tuple(sorted(set(self.runtime_gpu_proof_sha256s)))
            or (self.schema_version == 1 and not self.runtime_gpu_proof_sha256s)
        ):
            raise ValueError("formal physical plan GPU proof set is not canonical")
        for digest in self.runtime_gpu_proof_sha256s:
            _require_sha256("formal physical plan runtime GPU proof", digest)
        if (
            type(self.runtime_gpu_proof_artifacts) is not tuple
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.runtime_gpu_proof_artifacts
            )
            or tuple(
                sorted(row.semantic_sha256 for row in self.runtime_gpu_proof_artifacts)
            )
            != self.runtime_gpu_proof_sha256s
        ):
            raise ValueError("formal physical plan GPU proof paths differ")
        for binding in self.runtime_gpu_proof_artifacts:
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError("formal physical plan GPU proof path changed")
        if self.nextn_tp2_authority_sha256 is not None:
            _require_sha256(
                "formal physical plan NEXTN TP2 authority",
                self.nextn_tp2_authority_sha256,
            )
        if self.schema_version in {3, 4} and self.nextn_mtp_mode is not None:
            if (
                self.stage != "E6"
                or self.nextn_mtp_mode != "built_in_mtp"
                or self.target_snapshot_sha256 is None
                or self.mtp_component_sha256 is None
                or self.target_snapshot_sha256 == self.mtp_component_sha256
                or type(self.mtp_component) is not CanonicalJsonProofBinding
                or self.mtp_component.semantic_sha256 != self.mtp_component_sha256
            ):
                raise ValueError("formal built-in MTP physical identity differs")
            for label, digest in (
                ("target snapshot", self.target_snapshot_sha256),
                ("MTP component", self.mtp_component_sha256),
            ):
                _require_sha256(f"formal built-in MTP {label}", digest)
            if (
                CanonicalJsonProofBinding.bind(self.mtp_component.absolute_path)
                != self.mtp_component
            ):
                raise ValueError("formal built-in MTP component changed")
        elif any(
            value is not None
            for value in (
                self.nextn_mtp_mode,
                self.target_snapshot_sha256,
                self.mtp_component_sha256,
                self.mtp_component,
            )
        ):
            raise ValueError("legacy/external physical plan carries built-in MTP")
        self.native_terminal_binding.validate()
        root = _private_output_root(self.private_output_root)
        if self.schema_version in {2, 3, 4}:
            assert self.single_operator_execution_rebuild_source is not None
            if Path(
                self.single_operator_execution_rebuild_source.absolute_path
            ) not in {
                root / "formal-single-operator-execution-rebuild-source.json",
                root / "formal-single-operator-early-run-plan-inputs.json",
                root / "formal-single-operator-downstream-run-plan-inputs.json",
                root / "formal-single-operator-prepared-downstream-inputs.json",
                root / "formal-single-operator-e5-failure-execution.json",
                root / "formal-single-operator-profiler-subject-inputs.json",
            }:
                raise ValueError(
                    "current formal plan rebuild source path differs from run root"
                )
        values = (
            self.terminal_output_path,
            self.native_itl_pointer_output_path,
            self.live_run_receipt_output_path,
            self.lifecycle_timing_output_path,
            self.server_log_output_path,
            self.server_stdout_output_path,
            self.server_stderr_output_path,
            self.junit_output_path,
            self.before_gpu_snapshot_output_path,
            self.ready_gpu_snapshot_output_path,
            self.after_gpu_snapshot_output_path,
            self.fatal_output_path,
            *(
                ()
                if self.formal_gang_terminal_output_path is None
                else (self.formal_gang_terminal_output_path,)
            ),
        )
        paths = tuple(Path(value) for value in values)
        if len(paths) != len(set(paths)) or any(
            not path.is_absolute()
            or path != path.resolve(strict=False)
            or path.parent != root
            or not path.name
            for path in paths
        ):
            raise ValueError("formal physical plan output paths differ")
        if (self.topology_mode == "tp1_dp1") != (
            self.formal_gang_terminal_output_path is None
        ):
            raise ValueError("formal gang output path differs from topology")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "formal_execution_authorized": self.formal_execution_authorized,
            "execution_binding_sha256": self.execution_binding_sha256,
            "subject_sha256": self.subject_sha256,
            "materialized_cell_id": self.materialized_cell_id,
            "stage": self.stage,
            "method": self.method,
            "topology_mode": self.topology_mode,
            "inventory_sha256": self.inventory_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "runtime_gpu_proof_sha256s": list(self.runtime_gpu_proof_sha256s),
            "runtime_gpu_proof_artifacts": [
                row.to_dict() for row in self.runtime_gpu_proof_artifacts
            ],
            "nextn_tp2_authority_sha256": self.nextn_tp2_authority_sha256,
            "launch_manifest": self.launch_manifest.to_dict(),
            "request_schedule_receipt": self.request_schedule_receipt.to_dict(),
            "native_terminal_binding": _native_binding_to_dict(
                self.native_terminal_binding
            ),
            "private_output_root": self.private_output_root,
            "terminal_output_path": self.terminal_output_path,
            "native_itl_pointer_output_path": self.native_itl_pointer_output_path,
            "live_run_receipt_output_path": self.live_run_receipt_output_path,
            "lifecycle_timing_output_path": self.lifecycle_timing_output_path,
            "server_log_output_path": self.server_log_output_path,
            "server_stdout_output_path": self.server_stdout_output_path,
            "server_stderr_output_path": self.server_stderr_output_path,
            "junit_output_path": self.junit_output_path,
            "before_gpu_snapshot_output_path": self.before_gpu_snapshot_output_path,
            "ready_gpu_snapshot_output_path": self.ready_gpu_snapshot_output_path,
            "after_gpu_snapshot_output_path": self.after_gpu_snapshot_output_path,
            "formal_gang_terminal_output_path": (self.formal_gang_terminal_output_path),
            "fatal_output_path": self.fatal_output_path,
        }
        if self.schema_version in {2, 3, 4}:
            assert self.single_operator_execution_rebuild_source is not None
            value["single_operator_execution_rebuild_source"] = (
                self.single_operator_execution_rebuild_source.to_dict()
            )
        if self.schema_version in {3, 4} and self.nextn_mtp_mode is not None:
            assert self.mtp_component is not None
            value["nextn_mtp_mode"] = self.nextn_mtp_mode
            value["target_snapshot_sha256"] = self.target_snapshot_sha256
            value["mtp_component_sha256"] = self.mtp_component_sha256
            value["mtp_component"] = self.mtp_component.to_dict()
        if self.schema_version == 4:
            assert self.serving_execution_policy is not None
            value["serving_execution_policy"] = self.serving_execution_policy.to_dict()
            value["process_hard_timeout_ns"] = self.process_hard_timeout_ns
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("formal serving run plan must be an object")
        schema_version = value.get("schema_version")
        expected = {
            field.name
            for field in fields(cls)
            if field.name
            not in {
                "single_operator_execution_rebuild_source",
                "nextn_mtp_mode",
                "target_snapshot_sha256",
                "mtp_component_sha256",
                "mtp_component",
                "serving_execution_policy",
                "process_hard_timeout_ns",
            }
        }
        if schema_version in {2, 3, 4}:
            expected.add("single_operator_execution_rebuild_source")
        if schema_version in {3, 4} and "nextn_mtp_mode" in value:
            expected.update(
                {
                    "nextn_mtp_mode",
                    "target_snapshot_sha256",
                    "mtp_component_sha256",
                    "mtp_component",
                }
            )
        if schema_version == 4:
            expected.update({"serving_execution_policy", "process_hard_timeout_ns"})
        row = _strict_object(
            "formal serving run plan",
            value,
            expected,
        )
        gpu_uuids = row.pop("gpu_uuids")
        runtime_gpu_proof_sha256s = row.pop("runtime_gpu_proof_sha256s")
        runtime_gpu_proof_artifacts = row.pop("runtime_gpu_proof_artifacts")
        if (
            type(gpu_uuids) is not list
            or type(runtime_gpu_proof_sha256s) is not list
            or type(runtime_gpu_proof_artifacts) is not list
        ):
            raise TypeError("formal serving plan GPU UUID/proof values must be arrays")
        launch_manifest = CanonicalJsonProofBinding.from_dict(
            row.pop("launch_manifest")
        )
        request_schedule_receipt = CanonicalJsonProofBinding.from_dict(
            row.pop("request_schedule_receipt")
        )
        rebuild_source_value = row.pop("single_operator_execution_rebuild_source", None)
        rebuild_source = (
            None
            if rebuild_source_value is None
            else CanonicalJsonProofBinding.from_dict(rebuild_source_value)
        )
        mtp_component_value = row.pop("mtp_component", None)
        mtp_component = (
            None
            if mtp_component_value is None
            else CanonicalJsonProofBinding.from_dict(mtp_component_value)
        )
        native_terminal_binding = _native_binding_from_dict(
            row.pop("native_terminal_binding")
        )
        execution_policy_value = row.pop("serving_execution_policy", None)
        if execution_policy_value is not None:
            from lightcone_spec.orchestration.executor import (
                RegisteredServingExecutionPolicy,
            )

            execution_policy = RegisteredServingExecutionPolicy.from_dict(
                execution_policy_value
            )
        else:
            execution_policy = None
        return cls(
            **row,
            gpu_uuids=tuple(gpu_uuids),
            runtime_gpu_proof_sha256s=tuple(runtime_gpu_proof_sha256s),
            runtime_gpu_proof_artifacts=tuple(
                CanonicalJsonProofBinding.from_dict(item)
                for item in runtime_gpu_proof_artifacts
            ),
            launch_manifest=launch_manifest,
            request_schedule_receipt=request_schedule_receipt,
            single_operator_execution_rebuild_source=rebuild_source,
            mtp_component=mtp_component,
            serving_execution_policy=execution_policy,
            native_terminal_binding=native_terminal_binding,
        )


@dataclass(frozen=True)
class FormalServingProcessRuntimeContract:
    """Deep-replayed outer-watchdog contract for one current serving plan."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_process_runtime_contract"]
    plan_sha256: str
    scientific_command_sha256: str
    process_hard_timeout_ns: int
    terminal_publication_grace_ns: int
    outer_max_runtime_seconds: int
    progress_log_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_process_runtime_contract"
        ):
            raise ValueError("formal serving runtime contract schema differs")
        _require_sha256("formal serving runtime plan", self.plan_sha256)
        _require_sha256(
            "formal serving scientific command", self.scientific_command_sha256
        )
        if (
            type(self.process_hard_timeout_ns) is not int
            or self.process_hard_timeout_ns < 1
            or self.process_hard_timeout_ns > _MAX_CURRENT_PROCESS_TIMEOUT_NS
            or type(self.terminal_publication_grace_ns) is not int
            or self.terminal_publication_grace_ns
            != FORMAL_SERVING_TERMINAL_PUBLICATION_GRACE_NS
            or type(self.outer_max_runtime_seconds) is not int
            or self.outer_max_runtime_seconds
            != (
                self.process_hard_timeout_ns
                + self.terminal_publication_grace_ns
                + 1_000_000_000
                - 1
            )
            // 1_000_000_000
        ):
            raise ValueError("formal serving runtime timeout contract differs")
        if (
            type(self.progress_log_paths) is not tuple
            or len(self.progress_log_paths) != 3
            or len(set(self.progress_log_paths)) != 3
            or any(
                type(value) is not str
                or not value
                or not Path(value).is_absolute()
                or Path(value) != Path(value).resolve(strict=False)
                for value in self.progress_log_paths
            )
        ):
            raise ValueError("formal serving runtime progress logs differ")


def formal_serving_process_runtime_contract(
    plan_path: str | Path,
) -> FormalServingProcessRuntimeContract:
    """Return the sole source-owned outer timeout for a fresh serving plan.

    The plan, schedule, launch, materialization, inventory, and execution
    policy are deep-replayed before any value is returned.  The scientific
    digest excludes attempt-local output paths, so an infrastructure retry can
    use fresh evidence files without silently changing the experiment.
    """

    plan, launch, schedule = _load_formal_single_operator_trusted_run_plan(plan_path)
    if (
        plan.schema_version != 4
        or plan.serving_execution_policy is None
        or plan.process_hard_timeout_ns is None
    ):
        raise ValueError("formal serving runtime contract requires a schema-4 plan")
    policy = plan.serving_execution_policy
    _registered_plan_process_hard_timeout_seconds(
        plan,
        allowed_topologies=_TOPOLOGIES,
    )
    scientific_command_sha256 = _sha256(
        {
            "schema_version": 1,
            "kind": "formal_serving_scientific_command",
            "physical_dispatch_protocol_sha256": plan.protocol_sha256,
            "execution_binding_sha256": plan.execution_binding_sha256,
            "subject_sha256": plan.subject_sha256,
            "materialized_cell_id": plan.materialized_cell_id,
            "stage": plan.stage,
            "method": plan.method,
            "topology_mode": plan.topology_mode,
            "inventory_sha256": plan.inventory_sha256,
            "gpu_uuids": list(plan.gpu_uuids),
            "runtime_gpu_proof_sha256s": list(plan.runtime_gpu_proof_sha256s),
            "nextn_tp2_authority_sha256": plan.nextn_tp2_authority_sha256,
            "launch_manifest_sha256": launch.sha256,
            "request_schedule_sha256": schedule.sha256,
            "native_run_binding_sha256": _sha256(
                plan.native_terminal_binding.begin_payload()
            ),
            "serving_execution_policy_sha256": policy.sha256,
            "process_hard_timeout_ns": plan.process_hard_timeout_ns,
        }
    )
    return FormalServingProcessRuntimeContract(
        schema_version=1,
        kind="formal_serving_process_runtime_contract",
        plan_sha256=plan.sha256,
        scientific_command_sha256=scientific_command_sha256,
        process_hard_timeout_ns=plan.process_hard_timeout_ns,
        terminal_publication_grace_ns=(FORMAL_SERVING_TERMINAL_PUBLICATION_GRACE_NS),
        outer_max_runtime_seconds=(
            plan.process_hard_timeout_ns
            + FORMAL_SERVING_TERMINAL_PUBLICATION_GRACE_NS
            + 1_000_000_000
            - 1
        )
        // 1_000_000_000,
        progress_log_paths=(
            plan.server_log_output_path,
            plan.server_stdout_output_path,
            plan.server_stderr_output_path,
        ),
    )


def _validate_formal_serving_plan_mtp_identity(
    *,
    plan: FormalServingRunPlan,
    launch: CompileLaunchManifest,
    config: RunConfig,
) -> None:
    built_in = getattr(launch, "schema_version", 1) == 3
    if built_in:
        if (
            plan.schema_version not in {3, 4}
            or plan.stage != "E6"
            or plan.nextn_mtp_mode != "built_in_mtp"
            or plan.target_snapshot_sha256 != launch.target_snapshot_sha256
            or plan.mtp_component_sha256 != launch.mtp_component_sha256
            or plan.mtp_component != launch.mtp_component_binding
            or config.model.nextn_mtp_mode != "built_in_mtp"
            or config.model.target_snapshot_sha256 != plan.target_snapshot_sha256
            or config.model.mtp_component_sha256 != plan.mtp_component_sha256
            or launch.target_content_member_id != launch.drafter_content_member_id
            or launch.target_model_id != launch.drafter_model_id
            or launch.target_revision != launch.drafter_revision
            or launch.target_snapshot_path != launch.drafter_snapshot_path
            or "--speculative-draft-model-path" in launch.server_argv
        ):
            raise ValueError("formal built-in MTP physical binding differs")
    elif plan.schema_version == 3 or any(
        value is not None
        for value in (
            plan.nextn_mtp_mode,
            plan.target_snapshot_sha256,
            plan.mtp_component_sha256,
            plan.mtp_component,
        )
    ):
        raise ValueError("external physical launch carries built-in MTP state")


def _single_operator_execution_rebuild_source(
    execution_binding: FormalSingleOperatorExecutionBinding,
) -> FormalSingleOperatorExecutionRebuildSource:
    if type(execution_binding) is not FormalSingleOperatorExecutionBinding:
        raise TypeError("current formal plan requires an exact single-operator binding")
    verified = require_verified_formal_serving_execution_binding(execution_binding)
    assert type(verified) is FormalSingleOperatorExecutionBinding
    return FormalSingleOperatorExecutionRebuildSource(
        schema_version=1,
        kind="formal_single_operator_execution_rebuild_source",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_EXECUTION_REBUILD_SOURCE_PROTOCOL_SHA256
        ),
        execution_binding_sha256=verified.sha256,
        subject_sha256=verified.subject.sha256,
        materialized_cell_id=verified.subject.materialized_cell_id,
        execution_source=verified.execution_source,
        execution_source_sha256=verified.execution_source_sha256,
        formal_runtime_authority_manifest=(verified.runtime_authority_manifest_source),
        compile_launch_manifest=verified.compile_launch_manifest,
        inventory=verified.inventory_source,
        content_verification_receipt=(verified.content_verification_receipt_source),
        runtime_gpu_proof_artifacts=(verified.subject.runtime_gpu_proof_artifacts),
        tts_calibration_authority=(verified.tts_calibration_authority_source),
        e1_recipe_anchor_authority=(verified.e1_recipe_anchor_authority_source),
        formal_registry_verification_receipt=(
            verified.formal_registry_verification_receipt_source
        ),
        repository_root=verified.repository_root,
    )


def revalidate_formal_single_operator_execution_rebuild_source(
    path: str | Path,
) -> FormalSingleOperatorExecutionRebuildSource:
    """Deep-open one persisted clean-restart source without private state."""

    binding = CanonicalJsonProofBinding.bind(path)
    source = FormalSingleOperatorExecutionRebuildSource.from_dict(binding.reopen())
    if source.sha256 != binding.semantic_sha256:
        raise ValueError("single-operator execution rebuild source digest differs")
    source.__post_init__()
    return source


def _join_single_operator_rebuild_source(
    *,
    source: FormalSingleOperatorExecutionRebuildSource,
    execution_binding: FormalSingleOperatorExecutionBinding,
    plan: FormalServingRunPlan,
) -> None:
    if type(execution_binding) is not FormalSingleOperatorExecutionBinding:
        raise TypeError("current formal plan replay requires its exact binding type")
    expected_source = _single_operator_execution_rebuild_source(execution_binding)
    if source != expected_source or source.sha256 != expected_source.sha256:
        raise ValueError("current formal plan rebuild sources differ from verifier")
    if (
        plan.schema_version not in {2, 3, 4}
        or plan.single_operator_execution_rebuild_source is None
        or plan.single_operator_execution_rebuild_source.semantic_sha256
        != source.sha256
        or source.execution_binding_sha256 != plan.execution_binding_sha256
        or source.subject_sha256 != plan.subject_sha256
        or source.materialized_cell_id != plan.materialized_cell_id
        or source.compile_launch_manifest != plan.launch_manifest
        or source.runtime_gpu_proof_artifacts != plan.runtime_gpu_proof_artifacts
        or source.inventory.semantic_sha256 != plan.inventory_sha256
    ):
        raise ValueError("current formal plan differs from its rebuild source")


def _expected_native_terminal_binding(
    *,
    execution_binding: FormalServingExecutionBinding,
    schedule: FormalServingRequestScheduleReceipt,
) -> NativeTerminalRunBinding:
    identity = execution_binding.subject.execution_identity
    return NativeTerminalRunBinding(
        run_id=identity.run_id,
        run_nonce_sha256=identity.run_nonce_sha256,
        execution_plan_sha256=identity.execution_plan_sha256,
        rank_config_sha256=identity.rank_config_sha256,
        attempt_id=identity.attempt_id,
        session_id=f"formal-{execution_binding.subject.materialized_cell_id[:24]}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha256(
            {
                "execution_binding_sha256": execution_binding.sha256,
                "request_schedule_receipt_sha256": schedule.sha256,
            }
        ),
        method=execution_binding.subject.method,
        warmup_request_ids=tuple(
            row.request.request_id
            for row in formal_serving_request_schedule_rows(schedule)
            if row.phase == "warmup"
        ),
        scored_request_ids=tuple(
            row.request.request_id
            for row in formal_serving_request_schedule_rows(schedule)
            if row.phase == "scored"
        ),
    )


def _reopen_schedule_receipt(
    binding: CanonicalJsonProofBinding,
) -> FormalServingRequestScheduleReceipt:
    value = FormalServingRequestScheduleReceipt.from_dict(binding.reopen())
    if value.sha256 != binding.semantic_sha256:
        raise ValueError("formal request schedule receipt digest differs")
    value.reopen()
    return value


def _revalidate_backend_runtime_proofs(
    *,
    execution_binding: FormalServingExecutionBinding,
    plan: FormalServingRunPlan,
) -> None:
    """Consume the exact private proof tokens, not their caller-authored labels."""

    native = execution_binding.verified_native_gpu_proofs
    distributed = execution_binding.verified_distributed_gpu_proofs
    all_tokens = (*native, *distributed)
    if (
        tuple(sorted(row.sha256 for row in all_tokens))
        != plan.runtime_gpu_proof_sha256s
    ):
        raise ValueError("formal physical plan proof tokens differ from its digest set")
    if any(
        row.inventory_sha256 != plan.inventory_sha256 or row.gpu_uuids != plan.gpu_uuids
        for row in all_tokens
    ):
        raise ValueError("formal physical runtime proof belongs to another assignment")
    topology = plan.topology_mode
    algorithm = execution_binding.run_config.model.algorithm
    native_by_suite = {row.suite_id: row for row in native}
    if len(native_by_suite) != len(native):
        raise ValueError("formal physical runtime proof repeats a native suite")
    if topology == "tp1_dp1":
        if distributed:
            raise ValueError("TP1 physical plan carries a distributed proof")
        suite = {
            "DSPARK": "dspark_tp1",
            "NEXTN": "nextn_tp1",
            "EAGLE3": "eagle3_tp1",
        }.get(algorithm)
        if suite is not None and suite not in native_by_suite:
            raise ValueError("TP1 backend-specific native proof is absent")
        return
    if len(distributed) != 1 or distributed[0].topology_mode != topology:
        raise ValueError("distributed physical plan lacks exact topology proof")
    suite = {
        ("DSPARK", "tp2_dp1"): "dspark_tp2",
        ("DSPARK", "tp1_dp2"): "dspark_dp2",
        ("NEXTN", "tp2_dp1"): "nextn_tp2",
    }.get((algorithm, topology))
    if suite is not None:
        proof = native_by_suite.get(suite)
        if proof is None or proof.topology_mode != topology:
            raise ValueError("distributed backend-specific native proof is absent")
    elif algorithm != "DFLASH":
        raise ValueError("distributed backend/topology is not formally registered")


def revalidate_formal_serving_run_plan(
    plan: FormalServingRunPlan,
    *,
    execution_binding: FormalServingExecutionBinding,
    verified_nextn_tp2_authority: VerifiedNextNTp2Authority | None = None,
) -> tuple[CompileLaunchManifest, FormalServingRequestScheduleReceipt]:
    if type(plan) is not FormalServingRunPlan:
        raise TypeError("formal physical replay requires an exact run plan")
    plan.__post_init__()
    verified = require_verified_formal_serving_execution_binding(execution_binding)
    if plan.schema_version in {2, 3, 4}:
        if type(verified) is not FormalSingleOperatorExecutionBinding:
            raise TypeError(
                "current formal plan requires a rebuilt single-operator binding"
            )
        assert plan.single_operator_execution_rebuild_source is not None
        rebuild_source = revalidate_formal_single_operator_execution_rebuild_source(
            plan.single_operator_execution_rebuild_source.absolute_path
        )
        _join_single_operator_rebuild_source(
            source=rebuild_source,
            execution_binding=verified,
            plan=plan,
        )
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    if launch.sha256 != plan.launch_manifest.semantic_sha256:
        raise ValueError("formal physical launch manifest changed")
    schedule = _reopen_schedule_receipt(plan.request_schedule_receipt)
    if plan.schema_version == 4:
        expected_policy = _registered_serving_execution_policy(
            stage=plan.stage,
            schedule=schedule,
        )
        if expected_policy is None:
            raise ValueError("current physical plan lacks a registered schedule")
        expected_hard_timeout_ns = _registered_process_hard_timeout_ns(
            policy=expected_policy,
            schedule=schedule,
        )
        if (
            plan.serving_execution_policy != expected_policy
            or plan.process_hard_timeout_ns != expected_hard_timeout_ns
        ):
            raise ValueError("current physical plan timing policy failed replay")
    _worker, worker_sha, worker_size = _tokenizer_worker_source()
    expected_binding = _expected_native_terminal_binding(
        execution_binding=verified,
        schedule=schedule,
    )
    subject = verified.subject
    _revalidate_backend_runtime_proofs(execution_binding=verified, plan=plan)
    config = verified.run_config
    _validate_formal_serving_plan_mtp_identity(
        plan=plan,
        launch=launch,
        config=config,
    )
    nextn_tp2 = config.model.algorithm == "NEXTN" and subject.topology_mode == "tp2_dp1"
    if nextn_tp2:
        if type(verified_nextn_tp2_authority) is not VerifiedNextNTp2Authority:
            raise ValueError("NEXTN TP2 formal plan requires its verified authority")
        authority = verified_nextn_tp2_authority
        if (
            authority.sha256 != plan.nextn_tp2_authority_sha256
            or authority.inventory_sha256 != subject.inventory_sha256
            or authority.registry_sha256 != subject.execution_identity.registry_sha256
            or authority.gpu_uuids != subject.gpu_uuids
            or authority.target_model_id != config.model.target
            or authority.drafter_model_id != config.model.drafter
            or authority.target_revision != config.model.target_revision
            or authority.drafter_revision != config.model.drafter_revision
            or authority.native_gpu_proof_sha256
            not in verified.runtime_gpu_proof_sha256s
            or authority.distributed_gpu_proof_sha256
            not in verified.runtime_gpu_proof_sha256s
            or authority.content_verification_receipt_sha256
            != subject.content_verification_receipt_sha256
        ):
            raise ValueError("NEXTN TP2 authority differs from sealed execution")
    elif (
        verified_nextn_tp2_authority is not None
        or plan.nextn_tp2_authority_sha256 is not None
    ):
        raise ValueError("non-NEXTN-TP2 plan carries a NEXTN TP2 authority")
    if (
        plan.execution_binding_sha256 != verified.sha256
        or plan.subject_sha256 != subject.sha256
        or plan.materialized_cell_id != subject.materialized_cell_id
        or plan.stage != subject.stage
        or plan.method != subject.method
        or plan.topology_mode != subject.topology_mode
        or plan.inventory_sha256 != subject.inventory_sha256
        or plan.gpu_uuids != subject.gpu_uuids
        or plan.runtime_gpu_proof_sha256s != verified.runtime_gpu_proof_sha256s
        or plan.runtime_gpu_proof_artifacts
        != verified.subject.runtime_gpu_proof_artifacts
        or launch.inventory_sha256 != subject.inventory_sha256
        or launch.gpu_uuids != subject.gpu_uuids
        or schedule.execution_binding_sha256 != verified.sha256
        or schedule.subject_sha256 != subject.sha256
        or schedule.topology_mode != subject.topology_mode
        or schedule.tokenizer_worker_source_raw_sha256 != worker_sha
        or schedule.tokenizer_worker_source_size != worker_size
        or plan.native_terminal_binding != expected_binding
    ):
        raise ValueError("formal physical plan differs from sealed execution")
    return launch, schedule


def rebuild_formal_single_operator_execution_binding_from_plan(
    plan_path: str | Path,
    *,
    current_ns: int | None = None,
) -> FormalSingleOperatorExecutionBinding:
    """Rebuild a current binding from one schema-2 plan after process restart."""

    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if plan.sha256 != plan_binding.semantic_sha256:
        raise ValueError("current formal run plan semantic digest differs")
    if (
        plan.schema_version not in {2, 3, 4}
        or plan.single_operator_execution_rebuild_source is None
    ):
        raise ValueError("formal run plan has no clean-restart execution source")
    if Path(plan_binding.absolute_path) != (
        Path(plan.private_output_root) / "formal-serving-run-plan.json"
    ):
        raise ValueError("current formal run plan path differs from its run root")
    source = revalidate_formal_single_operator_execution_rebuild_source(
        plan.single_operator_execution_rebuild_source.absolute_path
    )
    rebuilt = verify_formal_single_operator_execution_binding(
        execution_source_path=source.execution_source.absolute_path,
        materialized_cell_id=source.materialized_cell_id,
        formal_runtime_authority_manifest_path=(
            source.formal_runtime_authority_manifest.absolute_path
        ),
        compile_launch_manifest_path=source.compile_launch_manifest.absolute_path,
        inventory_path=source.inventory.absolute_path,
        content_verification_receipt_path=(
            source.content_verification_receipt.absolute_path
        ),
        runtime_gpu_proof_artifact_paths=tuple(
            value.absolute_path for value in source.runtime_gpu_proof_artifacts
        ),
        tts_calibration_authority_artifact_path=(
            None
            if source.tts_calibration_authority is None
            else source.tts_calibration_authority.absolute_path
        ),
        e1_recipe_anchor_authority_artifact_path=(
            None
            if source.e1_recipe_anchor_authority is None
            else source.e1_recipe_anchor_authority.absolute_path
        ),
        formal_registry_verification_receipt_path=(
            None
            if source.formal_registry_verification_receipt is None
            else source.formal_registry_verification_receipt.absolute_path
        ),
        repository_root=source.repository_root,
        now_ns=current_ns,
    )
    _join_single_operator_rebuild_source(
        source=source,
        execution_binding=rebuilt,
        plan=plan,
    )
    revalidate_formal_serving_run_plan(
        plan,
        execution_binding=rebuilt,
        verified_nextn_tp2_authority=rebuilt.verified_nextn_tp2_authority,
    )
    if (
        CanonicalJsonProofBinding.bind(plan_path) != plan_binding
        or CanonicalJsonProofBinding.bind(
            plan.single_operator_execution_rebuild_source.absolute_path
        )
        != plan.single_operator_execution_rebuild_source
    ):
        raise RuntimeError("current formal plan changed while its binding rebuilt")
    return rebuilt


def _early_run_subject_sha(
    value: object,
    *,
    inventory_sha256: str,
) -> str:
    return _sha256(
        {
            "kind": "formal_single_operator_direct_run_subject",
            "execution_source_sha256": value.execution_source_sha256,
            "materialized_cell_id": value.materialized_cell_id,
            "materialization_sha256": value.materialization_sha256,
            "preflight_inputs_sha256": value.preflight_inputs.semantic_sha256,
            "compile_launch_manifest_sha256": (
                value.compile_launch_manifest.semantic_sha256
            ),
            "inventory_sha256": inventory_sha256,
        }
    )


def _load_formal_single_operator_trusted_run_plan(
    plan_path: str | Path,
) -> tuple[
    FormalServingRunPlan,
    CompileLaunchManifest,
    FormalServingRequestScheduleReceipt,
]:
    """Replay one current plan from its local source-owned files only.

    The trusted single-operator workflow does not reconstruct the older
    signed execution token.  It reopens the concrete plan, launch, request
    schedule, materialization, and inventory selected before the run, then
    checks the identities that the physical child actually consumes.
    """

    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if (
        plan.schema_version not in {2, 3, 4}
        or plan.sha256 != plan_binding.semantic_sha256
        or plan.single_operator_execution_rebuild_source is None
        or Path(plan_binding.absolute_path)
        != Path(plan.private_output_root) / "formal-serving-run-plan.json"
    ):
        raise ValueError("trusted single-operator run plan differs")
    source_binding = plan.single_operator_execution_rebuild_source
    trusted_execution_source_path: str | None = None
    trusted_content_source: FormalContentSourceBinding | None = None
    source_value = source_binding.reopen()
    source_kind = source_value.get("kind")
    if source_kind == "formal_single_operator_execution_rebuild_source":
        source = revalidate_formal_single_operator_execution_rebuild_source(
            source_binding.absolute_path
        )
        launch_source = source.compile_launch_manifest
        inventory_source = source.inventory
        materialized_cell_id = source.materialized_cell_id
        execution_binding_sha256 = source.execution_binding_sha256
        subject_sha256 = source.subject_sha256
    elif source_kind in {
        "formal_single_operator_early_run_plan_inputs",
        "formal_single_operator_downstream_run_plan_inputs",
        "formal_single_operator_prepared_downstream_run_plan_inputs",
        "formal_single_operator_e5_failure_execution_descriptor",
        "formal_single_operator_profiler_subject_run_plan_inputs",
    }:
        if source_kind == ("formal_single_operator_profiler_subject_run_plan_inputs"):
            from lightcone_spec.experiments.formal_single_operator_profiler import (
                revalidate_formal_single_operator_profiler_subject_inputs,
            )

            direct = revalidate_formal_single_operator_profiler_subject_inputs(
                source_binding.absolute_path,
                current_ns=time.time_ns(),
            )
            launch_source = direct.profile_compile_launch_manifest
            inventory_source = direct.inventory
            materialized_cell_id = direct.source_headline_cell_id
            execution_binding_sha256 = direct.sha256
            subject_sha256 = direct.subject_sha256
        elif source_kind == "formal_single_operator_e5_failure_execution_descriptor":
            from lightcone_spec.experiments.formal_failure_execution import (
                revalidate_formal_single_operator_e5_failure_execution_descriptor,
            )

            direct = revalidate_formal_single_operator_e5_failure_execution_descriptor(
                source_binding.absolute_path,
                current_ns=time.time_ns(),
            )
            launch_source = direct.compile_launch_manifest
            inventory_source = direct.inventory
            materialized_cell_id = direct.failure_subject.materialized_cell_id
            execution_binding_sha256 = direct.execution_binding_sha256
            subject_sha256 = direct.subject_sha256
        elif (
            source_kind == "formal_single_operator_prepared_downstream_run_plan_inputs"
        ):
            from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
                revalidate_formal_single_operator_prepared_downstream_run_plan_inputs,
            )

            direct = (
                revalidate_formal_single_operator_prepared_downstream_run_plan_inputs(
                    source_binding.absolute_path,
                    current_ns=time.time_ns(),
                )
            )
            launch_source = direct.compile_launch_manifest
            inventory_source = direct.inventory
            trusted_execution_source_path = direct.execution_source.absolute_path
            trusted_content_source = direct.content_source_binding
            materialized_cell_id = direct.materialized_cell_id
            execution_binding_sha256 = direct.execution_binding_sha256
            subject_sha256 = direct.subject_sha256
        elif source_kind == "formal_single_operator_early_run_plan_inputs":
            from lightcone_spec.experiments.formal_preflight_inputs import (
                FormalPreflightExecutionInputs,
            )
            from lightcone_spec.experiments.formal_single_operator_early_execution import (
                FormalSingleOperatorEarlyRunPlanInputs,
            )

            direct = FormalSingleOperatorEarlyRunPlanInputs.from_dict(source_value)
            if direct.sha256 != source_binding.semantic_sha256:
                raise ValueError("trusted direct run input digest differs")
            preflight_inputs = FormalPreflightExecutionInputs.from_dict(
                direct.preflight_inputs.reopen()
            )
            launch_source = direct.compile_launch_manifest
            inventory_source = preflight_inputs.inventory
            materialized_cell_id = direct.materialized_cell_id
            execution_binding_sha256 = direct.sha256
            subject_sha256 = _early_run_subject_sha(
                direct,
                inventory_sha256=inventory_source.semantic_sha256,
            )
        else:
            from lightcone_spec.experiments.formal_preflight_inputs import (
                FormalPreflightExecutionInputs,
            )
            from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
                FormalSingleOperatorDownstreamRunPlanInputs,
            )

            direct = FormalSingleOperatorDownstreamRunPlanInputs.from_dict(source_value)
            if direct.sha256 != source_binding.semantic_sha256:
                raise ValueError("trusted direct run input digest differs")
            preflight_inputs = FormalPreflightExecutionInputs.from_dict(
                direct.preflight_inputs.reopen()
            )
            launch_source = direct.compile_launch_manifest
            inventory_source = preflight_inputs.inventory
            materialized_cell_id = direct.materialized_cell_id
            execution_binding_sha256 = direct.sha256
            subject_sha256 = _early_run_subject_sha(
                direct,
                inventory_sha256=inventory_source.semantic_sha256,
            )
    else:
        raise ValueError("trusted plan source kind is unsupported")
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    schedule_binding = CanonicalJsonProofBinding.bind(
        plan.request_schedule_receipt.absolute_path
    )
    schedule = FormalServingRequestScheduleReceipt.from_dict(schedule_binding.reopen())
    expected_execution_policy = (
        _registered_serving_execution_policy(stage=plan.stage, schedule=schedule)
        if plan.schema_version == 4
        else None
    )
    expected_process_hard_timeout_ns = (
        None
        if expected_execution_policy is None
        else _registered_process_hard_timeout_ns(
            policy=expected_execution_policy,
            schedule=schedule,
        )
    )
    schedule_source = FormalServingRequestScheduleSource.from_dict(
        schedule.schedule_source.load()
    )
    materialization = _reopen_stage_materialization(schedule.materialization)
    cell = _materialized_cell(
        materialization,
        cell_id=plan.materialized_cell_id,
    )
    from lightcone_spec.experiments.gpu_pool import GpuInventory

    inventory = GpuInventory.from_dict(inventory_source.reopen())
    config = load_run_config(launch.run_config_path)
    _validate_formal_serving_plan_mtp_identity(
        plan=plan,
        launch=launch,
        config=config,
    )
    expected_warmup = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "warmup"
    )
    expected_scored = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    if (
        launch_source != plan.launch_manifest
        or inventory_source.semantic_sha256 != plan.inventory_sha256
        or materialized_cell_id != plan.materialized_cell_id
        or execution_binding_sha256 != plan.execution_binding_sha256
        or subject_sha256 != plan.subject_sha256
        or CanonicalJsonProofBinding.bind(plan.launch_manifest.absolute_path)
        != plan.launch_manifest
        or launch.sha256 != plan.launch_manifest.semantic_sha256
        or schedule_binding != plan.request_schedule_receipt
        or schedule.sha256 != schedule_binding.semantic_sha256
        or schedule.compile_launch_manifest != plan.launch_manifest
        or schedule.materialized_cell_id != plan.materialized_cell_id
        or schedule.execution_binding_sha256 != plan.execution_binding_sha256
        or schedule.subject_sha256 != plan.subject_sha256
        or schedule.topology_mode != plan.topology_mode
        or schedule_source.sha256 != schedule.schedule_source.semantic_sha256
        or schedule_source.e5_arrival_plan != schedule.e5_arrival_plan
        or schedule_source.materialization_receipt_sha256 != materialization.sha256
        or schedule_source.materialized_cell_id != plan.materialized_cell_id
        or schedule_source.subject_sha256 != plan.subject_sha256
        or schedule_source.topology_mode != plan.topology_mode
        or plan.serving_execution_policy != expected_execution_policy
        or plan.process_hard_timeout_ns != expected_process_hard_timeout_ns
        or cell.stage != plan.stage
        or inventory.sha256 != plan.inventory_sha256
        or launch.inventory_sha256 != plan.inventory_sha256
        or launch.gpu_uuids != plan.gpu_uuids
        or set(plan.gpu_uuids) - {row.uuid for row in inventory.devices}
        or config.method != plan.method
        or config.runtime.topology_mode != plan.topology_mode
        or plan.native_terminal_binding.method != plan.method
        or plan.native_terminal_binding.warmup_request_ids != expected_warmup
        or plan.native_terminal_binding.scored_request_ids != expected_scored
    ):
        raise ValueError("trusted single-operator plan inputs differ")
    nextn_tp2 = (
        plan.stage == "E6"
        and config.model.algorithm == "NEXTN"
        and config.runtime.topology_mode == "tp2_dp1"
    )
    if nextn_tp2:
        if (
            trusted_execution_source_path is None
            or trusted_content_source is None
            or trusted_content_source.mode != "trusted_single_operator"
        ):
            raise ValueError(
                "trusted E6 NEXTN TP2 plan lacks its empirical authority source"
            )
        from lightcone_spec.experiments.formal_single_operator_e6_interface import (
            derive_formal_single_operator_trusted_nextn_tp2_serving_authority,
        )

        authority = derive_formal_single_operator_trusted_nextn_tp2_serving_authority(
            execution_source_path=trusted_execution_source_path,
            materialized_cell_id=plan.materialized_cell_id,
            compile_launch_manifest=plan.launch_manifest,
            inventory=inventory_source,
            content_source=trusted_content_source,
        )
        if plan.nextn_tp2_authority_sha256 != authority.sha256:
            raise ValueError(
                "trusted E6 NEXTN TP2 empirical authority differs from plan"
            )
    elif plan.nextn_tp2_authority_sha256 is not None:
        raise ValueError("non-E6 trusted plan carries a NEXTN TP2 authority")
    return plan, launch, schedule


def _materialize_formal_single_operator_profiler_subject_schedule(
    *,
    inputs: object,
    input_binding: CanonicalJsonProofBinding,
    launch: CompileLaunchManifest,
) -> FormalServingRequestScheduleReceipt:
    """Retokenize the fixed E4-local subject under the profile-plan identity."""

    selected = FormalServingRequestScheduleReceipt.from_dict(
        inputs.selected_request_schedule.reopen()
    )
    selected.reopen()
    selected_source = FormalServingRequestScheduleSource.from_dict(
        selected.schedule_source.load()
    )
    if selected.schema_version != selected_source.schema_version:
        raise ValueError("profiler subject schedule/source schemas differ")
    if selected.schema_version == 7:
        raise ValueError("E4 profiler subject cannot be a controlled-context schedule")
    trusted = selected.schema_version in {5, 6}
    sharded = selected.schema_version == 6
    source_rows = tuple(formal_serving_request_schedule_source_rows(selected_source))
    source = replace(
        selected_source,
        schema_version=(5 if trusted else selected_source.schema_version),
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
            if trusted
            else selected_source.protocol_sha256
        ),
        derivation_protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
            if trusted
            else selected_source.derivation_protocol_sha256
        ),
        subject_sha256=inputs.subject_sha256,
        requests=source_rows,
        requests_shard_index=None,
        request_count=None,
    )
    root = _private_output_root(inputs.private_output_root)
    source_path = root / "formal-request-schedule-source.json"
    input_path = root / "formal-tokenization-input.json"
    output_path = root / "formal-tokenization-output.json"
    receipt_path = root / "formal-request-schedule-receipt.json"
    source_row_root = root / "formal-request-schedule-source-rows"
    receipt_row_root = root / "formal-request-schedule-receipt-rows"
    tokenization_root = root / "formal-tokenization-shards"
    input_index_path = root / "formal-tokenization-input-index.json"
    output_index_path = root / "formal-tokenization-output-index.json"
    if any(
        os.path.lexists(path)
        for path in (
            source_path,
            input_path,
            output_path,
            receipt_path,
            source_row_root,
            receipt_row_root,
            tokenization_root,
            input_index_path,
            output_index_path,
        )
    ):
        raise FileExistsError("profiler subject request schedule already exists")
    if sharded:
        source_row_root.mkdir(mode=0o700)
        receipt_row_root.mkdir(mode=0o700)
        source = publish_trusted_schedule_source_shards(
            source=source,
            output_directory=source_row_root,
        )
    publish_canonical_json_no_replace(source_path, source.to_dict())
    source_artifact = ContentJsonArtifactBinding.from_path(
        f"{_SCHEDULE_ARTIFACT_PREFIX}{inputs.source_headline_cell_id}-profiler",
        source_path,
    )
    if source_artifact.semantic_sha256 != source.sha256:
        raise RuntimeError("profiler subject schedule source changed")
    if sharded:
        (
            requests,
            token_input,
            token_output,
            worker_sha,
            worker_size,
            argv_sha,
            transformers_version,
            tokenizer_class,
            tokenizer_vocab_size,
        ) = _materialize_sharded_tokenization(
            root=root,
            source=source,
            launch=launch,
        )
    else:
        token_input = _publish_tokenization_input(
            path=input_path,
            source=source,
            launch=launch,
        )
        token_output, worker_sha, worker_size, argv_sha = _invoke_tokenizer_worker(
            input_path=input_path,
            output_path=output_path,
        )
        requests = _materialized_schedule_rows(
            source=source,
            launch=launch,
            tokenization_input=token_input,
            tokenization_output=token_output,
        )
        token_value = token_output.reopen()
        transformers_version = str(token_value["transformers_version"])
        tokenizer_class = str(token_value["tokenizer_class"])
        tokenizer_vocab_size = int(token_value["tokenizer_vocab_size"])
    if requests != tuple(formal_serving_request_schedule_rows(selected)):
        raise ValueError("profiler subject retokenization differs from headline")
    receipt = FormalServingRequestScheduleReceipt(
        schema_version=(5 if trusted else 4),
        kind="formal_serving_request_schedule_receipt",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
            if trusted
            else FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        formal_execution_authorized=False,
        execution_binding_sha256=input_binding.semantic_sha256,
        subject_sha256=inputs.subject_sha256,
        materialized_cell_id=inputs.source_headline_cell_id,
        workload_authority_sha256=(
            None if trusted else selected.workload_authority_sha256
        ),
        content_verification_receipt_sha256=(
            None if trusted else selected.content_verification_receipt_sha256
        ),
        topology_mode=selected.topology_mode,
        materialization=selected.materialization,
        content_verification_receipt=(
            None if trusted else selected.content_verification_receipt
        ),
        workload_source=selected.workload_source,
        compile_launch_manifest=inputs.profile_compile_launch_manifest,
        sampling_profile=selected.sampling_profile,
        schedule_source=source_artifact,
        tokenization_input=token_input,
        tokenization_output=token_output,
        tokenizer_worker_source_raw_sha256=worker_sha,
        tokenizer_worker_source_size=worker_size,
        tokenizer_worker_argv_sha256=argv_sha,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_snapshot_path=launch.tokenizer_snapshot_path,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_content_authority_sha256=(
            None if trusted else launch.tokenizer_content_authority_sha256
        ),
        transformers_version=transformers_version,
        tokenizer_class=tokenizer_class,
        tokenizer_vocab_size=tokenizer_vocab_size,
        requests=requests,
        e5_arrival_plan=selected.e5_arrival_plan,
        content_source_binding=(selected.content_source_binding if trusted else None),
        trusted_workload_member_sha256=(
            selected.trusted_workload_member_sha256 if trusted else None
        ),
        trusted_tts_calibration_authority=(
            selected.trusted_tts_calibration_authority if trusted else None
        ),
    )
    if sharded:
        receipt = publish_trusted_schedule_receipt_shards(
            receipt=receipt,
            output_directory=receipt_row_root,
        )
    publish_canonical_json_no_replace(receipt_path, receipt.to_dict())
    rebound = FormalServingRequestScheduleReceipt.from_dict(
        CanonicalJsonProofBinding.bind(receipt_path).reopen()
    )
    rebound.reopen()
    if rebound != receipt or rebound.sha256 != receipt.sha256:
        raise RuntimeError("profiler subject request schedule changed")
    return receipt


def materialize_formal_single_operator_profiler_subject_run_plan(
    *,
    profiler_subject_inputs_path: str | Path,
    current_ns: int,
) -> FormalServingRunPlan:
    """Publish one schema-2 profile-telemetry serving plan for E4 diagnostics."""

    from lightcone_spec.config import load_run_config
    from lightcone_spec.experiments.formal_single_operator_profiler import (
        revalidate_formal_single_operator_profiler_subject_inputs,
    )
    from lightcone_spec.experiments.gpu_pool import GpuInventory

    input_binding = CanonicalJsonProofBinding.bind(profiler_subject_inputs_path)
    inputs = revalidate_formal_single_operator_profiler_subject_inputs(
        input_binding.absolute_path,
        current_ns=current_ns,
    )
    root = _private_output_root(inputs.private_output_root)
    launch = CompileLaunchManifest.load(
        inputs.profile_compile_launch_manifest.absolute_path
    )
    config = load_run_config(launch.run_config_path)
    inventory = GpuInventory.from_dict(inputs.inventory.reopen())
    if (
        Path(input_binding.absolute_path)
        != root / "formal-single-operator-profiler-subject-inputs.json"
        or inputs.sha256 != input_binding.semantic_sha256
        or config.runtime.telemetry_detail != "profile"
        or config.runtime.topology_mode != "tp1_dp1"
        or launch.inventory_sha256 != inventory.sha256
        or launch.gpu_uuids != (config.runtime.device_identity,)
        or set(launch.gpu_uuids) - {row.uuid for row in inventory.devices}
    ):
        raise ValueError("profiler subject physical inputs differ")
    schedule = _materialize_formal_single_operator_profiler_subject_schedule(
        inputs=inputs,
        input_binding=input_binding,
        launch=launch,
    )
    schedule_binding = CanonicalJsonProofBinding.bind(
        root / "formal-request-schedule-receipt.json",
        semantic_sha256=schedule.sha256,
    )
    warmup_ids = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "warmup"
    )
    scored_ids = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    native_binding = NativeTerminalRunBinding(
        run_id=f"profile-{inputs.profiler_cell_id[:24]}",
        run_nonce_sha256=_sha256(
            {"profiler_subject_inputs_sha256": inputs.sha256, "attempt": 0}
        ),
        execution_plan_sha256=_sha256(
            {
                "compile_launch_manifest_sha256": launch.sha256,
                "request_schedule_sha256": schedule.sha256,
            }
        ),
        rank_config_sha256=_sha256(
            {
                "topology_mode": config.runtime.topology_mode,
                "gpu_uuids": list(launch.gpu_uuids),
            }
        ),
        attempt_id="attempt-0",
        session_id=f"profile-{inputs.profiler_cell_id[:24]}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha256(
            {
                "profiler_subject_inputs_sha256": inputs.sha256,
                "request_schedule_sha256": schedule.sha256,
            }
        ),
        method=config.method,
        warmup_request_ids=warmup_ids,
        scored_request_ids=scored_ids,
    )
    execution_policy = _registered_serving_execution_policy(
        stage="E4",
        schedule=schedule,
    )
    if execution_policy is None:
        raise ValueError("profiler subject lacks a registered serving policy")
    plan = FormalServingRunPlan(
        schema_version=4,
        kind="formal_serving_run_plan",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=inputs.sha256,
        subject_sha256=inputs.subject_sha256,
        materialized_cell_id=inputs.source_headline_cell_id,
        stage="E4",
        method=config.method,
        topology_mode=config.runtime.topology_mode,
        inventory_sha256=inventory.sha256,
        gpu_uuids=launch.gpu_uuids,
        runtime_gpu_proof_sha256s=(),
        runtime_gpu_proof_artifacts=(),
        nextn_tp2_authority_sha256=None,
        launch_manifest=inputs.profile_compile_launch_manifest,
        request_schedule_receipt=schedule_binding,
        native_terminal_binding=native_binding,
        private_output_root=str(root),
        terminal_output_path=str(root / "unsigned-native-terminal.json"),
        native_itl_pointer_output_path=str(root / "unsigned-native-itl.json"),
        live_run_receipt_output_path=str(root / "unsigned-live-run.json"),
        lifecycle_timing_output_path=str(root / "unsigned-lifecycle.json"),
        server_log_output_path=str(root / "server.log"),
        server_stdout_output_path=str(root / "stdout.log"),
        server_stderr_output_path=str(root / "stderr.log"),
        junit_output_path=str(root / "junit.xml"),
        before_gpu_snapshot_output_path=str(root / "before-gpu.json"),
        ready_gpu_snapshot_output_path=str(root / "ready-gpu.json"),
        after_gpu_snapshot_output_path=str(root / "after-gpu.json"),
        formal_gang_terminal_output_path=None,
        fatal_output_path=str(root / "fatal.json"),
        single_operator_execution_rebuild_source=input_binding,
        serving_execution_policy=execution_policy,
        process_hard_timeout_ns=_registered_process_hard_timeout_ns(
            policy=execution_policy,
            schedule=schedule,
        ),
    )
    plan_path = root / "formal-serving-run-plan.json"
    publish_canonical_json_no_replace(plan_path, plan.to_dict())
    rebound = revalidate_formal_single_operator_profiler_subject_run_plan(
        plan_path,
        current_ns=current_ns,
    )
    if rebound != plan:
        raise RuntimeError("profiler subject run plan changed")
    return plan


def revalidate_formal_single_operator_profiler_subject_run_plan(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalServingRunPlan:
    """Deep-reopen the subject plan and require byte-identical headline requests."""

    from lightcone_spec.config import load_run_config
    from lightcone_spec.experiments.formal_single_operator_profiler import (
        revalidate_formal_single_operator_profiler_subject_inputs,
    )

    plan, launch, schedule = _load_formal_single_operator_trusted_run_plan(path)
    assert plan.single_operator_execution_rebuild_source is not None
    inputs = revalidate_formal_single_operator_profiler_subject_inputs(
        plan.single_operator_execution_rebuild_source.absolute_path,
        current_ns=current_ns,
    )
    selected = FormalServingRequestScheduleReceipt.from_dict(
        inputs.selected_request_schedule.reopen()
    )
    selected.reopen()
    schedule.reopen()
    source = FormalServingRequestScheduleSource.from_dict(
        schedule.schedule_source.load()
    )
    selected_source = FormalServingRequestScheduleSource.from_dict(
        selected.schedule_source.load()
    )
    expected_source = replace(
        selected_source,
        subject_sha256=inputs.subject_sha256,
    )
    config = load_run_config(launch.run_config_path)
    if (
        Path(path) != Path(inputs.private_output_root) / "formal-serving-run-plan.json"
        or plan.execution_binding_sha256 != inputs.sha256
        or plan.subject_sha256 != inputs.subject_sha256
        or plan.materialized_cell_id != inputs.source_headline_cell_id
        or plan.launch_manifest != inputs.profile_compile_launch_manifest
        or schedule.execution_binding_sha256 != inputs.sha256
        or schedule.subject_sha256 != inputs.subject_sha256
        or tuple(formal_serving_request_schedule_rows(schedule))
        != tuple(formal_serving_request_schedule_rows(selected))
        or schedule.materialization != selected.materialization
        or schedule.content_verification_receipt
        != selected.content_verification_receipt
        or schedule.workload_source != selected.workload_source
        or source != expected_source
        or config.runtime.telemetry_detail != "profile"
    ):
        raise ValueError("profiler subject run plan differs from selected stratum")
    return plan


def _materialize_formal_single_operator_direct_serving_run_plan(
    *,
    inputs: object,
    input_binding: CanonicalJsonProofBinding,
    expected_input_name: str,
) -> FormalServingRunPlan:
    """Materialize one trusted plan from a validated direct input bundle."""

    if getattr(inputs, "kind", None) == (
        "formal_single_operator_prepared_downstream_run_plan_inputs"
    ):
        return _materialize_formal_single_operator_prepared_direct_serving_run_plan(
            inputs=inputs,
            input_binding=input_binding,
            expected_input_name=expected_input_name,
        )

    from lightcone_spec.experiments.formal_preflight_inputs import (
        FormalPreflightExecutionInputs,
    )
    from lightcone_spec.experiments.gpu_pool import GpuInventory

    if inputs.sha256 != input_binding.semantic_sha256:
        raise ValueError("single-operator direct plan input digest differs")
    root = _private_output_root(inputs.private_output_root)
    preflight_inputs = FormalPreflightExecutionInputs.from_dict(
        inputs.preflight_inputs.reopen()
    )
    launch = CompileLaunchManifest.load(inputs.compile_launch_manifest.absolute_path)
    materialization = _reopen_stage_materialization(inputs.materialization)
    cell = _materialized_cell(
        materialization,
        cell_id=inputs.materialized_cell_id,
    )
    inventory = GpuInventory.from_dict(preflight_inputs.inventory.reopen())
    config = load_run_config(launch.run_config_path)
    subject_sha256 = _early_run_subject_sha(
        inputs,
        inventory_sha256=inventory.sha256,
    )
    if (
        Path(input_binding.absolute_path) != root / expected_input_name
        or inputs.stage != materialization.stage
        or cell.stage != inputs.stage
        or inputs.compile_launch_manifest.semantic_sha256 != launch.sha256
        or inputs.preflight_inputs.semantic_sha256 != preflight_inputs.sha256
        or preflight_inputs.inventory.semantic_sha256 != inventory.sha256
        or launch.inventory_sha256 != inventory.sha256
        or set(launch.gpu_uuids) - {row.uuid for row in inventory.devices}
        or config.runtime.topology_mode not in _TOPOLOGIES
        or len(launch.gpu_uuids)
        != (1 if config.runtime.topology_mode == "tp1_dp1" else 2)
    ):
        raise ValueError("single-operator direct plan inputs differ")
    schedule = _materialize_single_operator_direct_schedule(
        inputs=inputs,
        preflight_inputs=preflight_inputs,
        input_binding=input_binding,
        launch=launch,
        materialization=materialization,
        cell=cell,
        subject_sha256=subject_sha256,
    )
    schedule_binding = CanonicalJsonProofBinding.bind(
        root / "formal-request-schedule-receipt.json",
        semantic_sha256=schedule.sha256,
    )
    warmup_ids = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "warmup"
    )
    scored_ids = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    native_binding = NativeTerminalRunBinding(
        run_id=f"single-{cell.cell_id[:24]}",
        run_nonce_sha256=_sha256(
            {"early_run_plan_inputs_sha256": inputs.sha256, "attempt": 0}
        ),
        execution_plan_sha256=_sha256(
            {
                "compile_launch_manifest_sha256": launch.sha256,
                "request_schedule_sha256": schedule.sha256,
            }
        ),
        rank_config_sha256=_sha256(
            {
                "topology_mode": config.runtime.topology_mode,
                "gpu_uuids": list(launch.gpu_uuids),
            }
        ),
        attempt_id="attempt-0",
        session_id=f"single-{cell.cell_id[:24]}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha256(
            {
                "early_run_plan_inputs_sha256": inputs.sha256,
                "request_schedule_sha256": schedule.sha256,
            }
        ),
        method=config.method,
        warmup_request_ids=warmup_ids,
        scored_request_ids=scored_ids,
    )
    execution_policy = _registered_serving_execution_policy(
        stage=cell.stage,
        schedule=schedule,
    )
    if execution_policy is None:
        raise ValueError("single-operator serving lacks a registered policy")
    plan = FormalServingRunPlan(
        schema_version=4,
        kind="formal_serving_run_plan",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=inputs.sha256,
        subject_sha256=subject_sha256,
        materialized_cell_id=cell.cell_id,
        stage=cell.stage,
        method=config.method,
        topology_mode=config.runtime.topology_mode,
        inventory_sha256=inventory.sha256,
        gpu_uuids=launch.gpu_uuids,
        runtime_gpu_proof_sha256s=(),
        runtime_gpu_proof_artifacts=(),
        nextn_tp2_authority_sha256=None,
        launch_manifest=inputs.compile_launch_manifest,
        request_schedule_receipt=schedule_binding,
        native_terminal_binding=native_binding,
        private_output_root=str(root),
        terminal_output_path=str(root / "unsigned-native-terminal.json"),
        native_itl_pointer_output_path=str(root / "unsigned-native-itl.json"),
        live_run_receipt_output_path=str(root / "unsigned-live-run.json"),
        lifecycle_timing_output_path=str(root / "unsigned-lifecycle.json"),
        server_log_output_path=str(root / "server.log"),
        server_stdout_output_path=str(root / "stdout.log"),
        server_stderr_output_path=str(root / "stderr.log"),
        junit_output_path=str(root / "junit.xml"),
        before_gpu_snapshot_output_path=str(root / "before-gpu.json"),
        ready_gpu_snapshot_output_path=str(root / "ready-gpu.json"),
        after_gpu_snapshot_output_path=str(root / "after-gpu.json"),
        formal_gang_terminal_output_path=(
            None
            if config.runtime.topology_mode == "tp1_dp1"
            else str(root / "unsigned-formal-gang-terminal.json")
        ),
        fatal_output_path=str(root / "fatal.json"),
        single_operator_execution_rebuild_source=input_binding,
        serving_execution_policy=execution_policy,
        process_hard_timeout_ns=_registered_process_hard_timeout_ns(
            policy=execution_policy,
            schedule=schedule,
        ),
    )
    plan_path = root / "formal-serving-run-plan.json"
    publish_canonical_json_no_replace(plan_path, plan.to_dict())
    rebound = FormalServingRunPlan.from_dict(
        CanonicalJsonProofBinding.bind(plan_path).reopen()
    )
    if rebound != plan or rebound.sha256 != plan.sha256:
        raise RuntimeError("single-operator serving run plan changed")
    return plan


def _materialize_formal_single_operator_prepared_direct_serving_run_plan(
    *,
    inputs: object,
    input_binding: CanonicalJsonProofBinding,
    expected_input_name: str,
) -> FormalServingRunPlan:
    """Materialize a trusted plan from an already tokenized source schedule."""

    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        revalidate_formal_single_operator_prepared_downstream_run_plan_inputs,
    )
    from lightcone_spec.experiments.gpu_pool import GpuInventory

    validated = revalidate_formal_single_operator_prepared_downstream_run_plan_inputs(
        input_binding.absolute_path,
        current_ns=time.time_ns(),
    )
    if validated != inputs or validated.sha256 != input_binding.semantic_sha256:
        raise ValueError("prepared downstream direct plan input differs")
    root = _private_output_root(inputs.private_output_root)
    launch = CompileLaunchManifest.load(inputs.compile_launch_manifest.absolute_path)
    materialization = _reopen_stage_materialization(inputs.materialization)
    cell = _materialized_cell(
        materialization,
        cell_id=inputs.materialized_cell_id,
    )
    inventory = GpuInventory.from_dict(inputs.inventory.reopen())
    config = load_run_config(launch.run_config_path)
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        inputs.request_schedule_receipt.reopen()
    )
    schedule_source = FormalServingRequestScheduleSource.from_dict(
        schedule.schedule_source.load()
    )
    if (
        Path(input_binding.absolute_path) != root / expected_input_name
        or inputs.stage != materialization.stage
        or cell.stage != inputs.stage
        or inputs.compile_launch_manifest.semantic_sha256 != launch.sha256
        or inputs.inventory.semantic_sha256 != inventory.sha256
        or launch.inventory_sha256 != inventory.sha256
        or set(launch.gpu_uuids) - {row.uuid for row in inventory.devices}
        or config.runtime.topology_mode not in _TOPOLOGIES
        or len(launch.gpu_uuids)
        != (1 if config.runtime.topology_mode == "tp1_dp1" else 2)
        or schedule.sha256 != inputs.request_schedule_receipt.semantic_sha256
        or schedule.execution_binding_sha256 != inputs.execution_binding_sha256
        or schedule.subject_sha256 != inputs.subject_sha256
        or schedule.materialized_cell_id != cell.cell_id
        or schedule.materialization != inputs.materialization
        or schedule.content_verification_receipt != inputs.content_verification_receipt
        or schedule.content_source_binding != inputs.content_source_binding
        or schedule.compile_launch_manifest != inputs.compile_launch_manifest
        or schedule.topology_mode != config.runtime.topology_mode
        or schedule_source.sha256 != schedule.schedule_source.semantic_sha256
        or schedule_source.e5_arrival_plan != schedule.e5_arrival_plan
        or schedule_source.materialization_receipt_sha256 != materialization.sha256
        or schedule_source.materialized_cell_id != cell.cell_id
        or schedule_source.subject_sha256 != inputs.subject_sha256
        or schedule_source.topology_mode != config.runtime.topology_mode
    ):
        raise ValueError("prepared downstream direct plan inputs differ")
    warmup_ids = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "warmup"
    )
    scored_ids = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    trusted_nextn_tp2_authority_sha256: str | None = None
    if (
        cell.stage == "E6"
        and config.model.algorithm == "NEXTN"
        and config.runtime.topology_mode == "tp2_dp1"
    ):
        if (
            inputs.schema_version != 2
            or inputs.content_source_binding is None
            or inputs.content_source_binding.mode != "trusted_single_operator"
        ):
            raise ValueError(
                "current E6 NEXTN TP2 serving requires trusted empirical authority"
            )
        from lightcone_spec.experiments.formal_single_operator_e6_interface import (
            derive_formal_single_operator_trusted_nextn_tp2_serving_authority,
        )

        trusted_nextn_tp2_authority_sha256 = (
            derive_formal_single_operator_trusted_nextn_tp2_serving_authority(
                execution_source_path=inputs.execution_source.absolute_path,
                materialized_cell_id=cell.cell_id,
                compile_launch_manifest=inputs.compile_launch_manifest,
                inventory=inputs.inventory,
                content_source=inputs.content_source_binding,
            ).sha256
        )
    native_binding = NativeTerminalRunBinding(
        run_id=f"single-{cell.cell_id[:24]}",
        run_nonce_sha256=_sha256(
            {"prepared_downstream_inputs_sha256": inputs.sha256, "attempt": 0}
        ),
        execution_plan_sha256=_sha256(
            {
                "compile_launch_manifest_sha256": launch.sha256,
                "request_schedule_sha256": schedule.sha256,
            }
        ),
        rank_config_sha256=_sha256(
            {
                "topology_mode": config.runtime.topology_mode,
                "gpu_uuids": list(launch.gpu_uuids),
            }
        ),
        attempt_id="attempt-0",
        session_id=f"single-{cell.cell_id[:24]}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha256(
            {
                "prepared_downstream_inputs_sha256": inputs.sha256,
                "request_schedule_sha256": schedule.sha256,
            }
        ),
        method=config.method,
        warmup_request_ids=warmup_ids,
        scored_request_ids=scored_ids,
    )
    launch_schema_version = getattr(launch, "schema_version", 2)
    execution_policy = _registered_serving_execution_policy(
        stage=cell.stage,
        schedule=schedule,
    )
    if execution_policy is None:
        raise ValueError("prepared serving lacks a registered policy")
    plan = FormalServingRunPlan(
        schema_version=4,
        kind="formal_serving_run_plan",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=inputs.execution_binding_sha256,
        subject_sha256=inputs.subject_sha256,
        materialized_cell_id=cell.cell_id,
        stage=cell.stage,
        method=config.method,
        topology_mode=config.runtime.topology_mode,
        inventory_sha256=inventory.sha256,
        gpu_uuids=launch.gpu_uuids,
        runtime_gpu_proof_sha256s=(),
        runtime_gpu_proof_artifacts=(),
        nextn_tp2_authority_sha256=trusted_nextn_tp2_authority_sha256,
        launch_manifest=inputs.compile_launch_manifest,
        request_schedule_receipt=inputs.request_schedule_receipt,
        native_terminal_binding=native_binding,
        private_output_root=str(root),
        terminal_output_path=str(root / "unsigned-native-terminal.json"),
        native_itl_pointer_output_path=str(root / "unsigned-native-itl.json"),
        live_run_receipt_output_path=str(root / "unsigned-live-run.json"),
        lifecycle_timing_output_path=str(root / "unsigned-lifecycle.json"),
        server_log_output_path=str(root / "server.log"),
        server_stdout_output_path=str(root / "stdout.log"),
        server_stderr_output_path=str(root / "stderr.log"),
        junit_output_path=str(root / "junit.xml"),
        before_gpu_snapshot_output_path=str(root / "before-gpu.json"),
        ready_gpu_snapshot_output_path=str(root / "ready-gpu.json"),
        after_gpu_snapshot_output_path=str(root / "after-gpu.json"),
        formal_gang_terminal_output_path=(
            None
            if config.runtime.topology_mode == "tp1_dp1"
            else str(root / "unsigned-formal-gang-terminal.json")
        ),
        fatal_output_path=str(root / "fatal.json"),
        single_operator_execution_rebuild_source=input_binding,
        nextn_mtp_mode=("built_in_mtp" if launch_schema_version == 3 else None),
        target_snapshot_sha256=getattr(launch, "target_snapshot_sha256", None),
        mtp_component_sha256=getattr(launch, "mtp_component_sha256", None),
        mtp_component=getattr(launch, "mtp_component_binding", None),
        serving_execution_policy=execution_policy,
        process_hard_timeout_ns=_registered_process_hard_timeout_ns(
            policy=execution_policy,
            schedule=schedule,
        ),
    )
    _validate_formal_serving_plan_mtp_identity(
        plan=plan,
        launch=launch,
        config=config,
    )
    plan_path = root / "formal-serving-run-plan.json"
    publish_canonical_json_no_replace(plan_path, plan.to_dict())
    rebound = FormalServingRunPlan.from_dict(
        CanonicalJsonProofBinding.bind(plan_path).reopen()
    )
    if rebound != plan or rebound.sha256 != plan.sha256:
        raise RuntimeError("prepared downstream serving run plan changed")
    return plan


def materialize_formal_single_operator_e5_failure_run_plan(
    *,
    failure_execution_descriptor_path: str | Path,
) -> FormalServingRunPlan:
    """Publish the serving plan consumed by one current E5 failure row."""

    from lightcone_spec.experiments.formal_failure_execution import (
        FormalSingleOperatorE5FailureExecutionDescriptor,
        formal_single_operator_e5_failure_native_identities,
        revalidate_formal_single_operator_e5_failure_execution_descriptor,
    )
    from lightcone_spec.experiments.gpu_pool import GpuInventory

    input_binding = CanonicalJsonProofBinding.bind(failure_execution_descriptor_path)
    inputs = FormalSingleOperatorE5FailureExecutionDescriptor.from_dict(
        input_binding.reopen()
    )
    validated = revalidate_formal_single_operator_e5_failure_execution_descriptor(
        input_binding.absolute_path,
        current_ns=time.time_ns(),
    )
    if validated != inputs or validated.sha256 != input_binding.semantic_sha256:
        raise ValueError("single-operator E5 failure descriptor changed")
    root = _private_output_root(inputs.private_output_root)
    launch = CompileLaunchManifest.load(inputs.compile_launch_manifest.absolute_path)
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        inputs.request_schedule_receipt.reopen()
    )
    materialization = _reopen_stage_materialization(inputs.materialization)
    cell = _materialized_cell(
        materialization,
        cell_id=inputs.failure_subject.materialized_cell_id,
    )
    inventory = GpuInventory.from_dict(inputs.inventory.reopen())
    config = load_run_config(launch.run_config_path)
    run_nonce, execution_plan, rank_config = (
        formal_single_operator_e5_failure_native_identities(
            prepared_launch_bundle_sha256=inputs.prepared_launch_bundle_sha256,
            prepared_launch_entry_sha256=inputs.prepared_launch_entry_sha256,
            compile_launch_manifest_sha256=launch.sha256,
            request_schedule_sha256=schedule.sha256,
            topology_mode=config.runtime.topology_mode,
            gpu_uuids=launch.gpu_uuids,
        )
    )
    subject = inputs.failure_subject
    if (
        Path(input_binding.absolute_path)
        != root / "formal-single-operator-e5-failure-execution.json"
        or cell.stage != "E5"
        or cell.task != "deterministic_failure_injection"
        or inputs.materialization_sha256 != materialization.sha256
        or inputs.compile_launch_manifest.semantic_sha256 != launch.sha256
        or inputs.request_schedule_receipt.semantic_sha256 != schedule.sha256
        or inputs.inventory.semantic_sha256 != inventory.sha256
        or launch.inventory_sha256 != inventory.sha256
        or launch.gpu_uuids != inputs.gpu_uuids
        or set(launch.gpu_uuids) - {row.uuid for row in inventory.devices}
        or config.runtime.topology_mode != subject.topology
        or config.runtime.topology_mode not in _TOPOLOGIES
        or schedule.execution_binding_sha256 != inputs.execution_binding_sha256
        or schedule.subject_sha256 != inputs.subject_sha256
        or schedule.materialized_cell_id != cell.cell_id
        or schedule.topology_mode != config.runtime.topology_mode
        or run_nonce != subject.run_nonce_sha256
        or execution_plan != subject.serving_execution_plan_sha256
        or rank_config != subject.serving_rank_config_sha256
    ):
        raise ValueError("single-operator E5 failure plan inputs differ")
    warmup_ids = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "warmup"
    )
    scored_ids = tuple(
        row.request.request_id
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    native_binding = NativeTerminalRunBinding(
        run_id=f"single-{cell.cell_id[:24]}",
        run_nonce_sha256=run_nonce,
        execution_plan_sha256=execution_plan,
        rank_config_sha256=rank_config,
        attempt_id="attempt-0",
        session_id=f"single-{cell.cell_id[:24]}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha256(
            {
                "failure_execution_descriptor_sha256": inputs.sha256,
                "request_schedule_sha256": schedule.sha256,
            }
        ),
        method=config.method,
        warmup_request_ids=warmup_ids,
        scored_request_ids=scored_ids,
    )
    execution_policy = _registered_serving_execution_policy(
        stage="E5",
        schedule=schedule,
    )
    if execution_policy is None:
        raise ValueError("E5 failure serving lacks a registered policy")
    plan = FormalServingRunPlan(
        schema_version=4,
        kind="formal_serving_run_plan",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=inputs.execution_binding_sha256,
        subject_sha256=inputs.subject_sha256,
        materialized_cell_id=cell.cell_id,
        stage="E5",
        method=config.method,
        topology_mode=config.runtime.topology_mode,
        inventory_sha256=inventory.sha256,
        gpu_uuids=launch.gpu_uuids,
        runtime_gpu_proof_sha256s=(),
        runtime_gpu_proof_artifacts=(),
        nextn_tp2_authority_sha256=None,
        launch_manifest=inputs.compile_launch_manifest,
        request_schedule_receipt=inputs.request_schedule_receipt,
        native_terminal_binding=native_binding,
        private_output_root=str(root),
        terminal_output_path=str(root / "unsigned-native-terminal.json"),
        native_itl_pointer_output_path=str(root / "unsigned-native-itl.json"),
        live_run_receipt_output_path=str(root / "unsigned-live-run.json"),
        lifecycle_timing_output_path=str(root / "unsigned-lifecycle.json"),
        server_log_output_path=str(root / "server.log"),
        server_stdout_output_path=str(root / "stdout.log"),
        server_stderr_output_path=str(root / "stderr.log"),
        junit_output_path=str(root / "junit.xml"),
        before_gpu_snapshot_output_path=str(root / "before-gpu.json"),
        ready_gpu_snapshot_output_path=str(root / "ready-gpu.json"),
        after_gpu_snapshot_output_path=str(root / "after-gpu.json"),
        formal_gang_terminal_output_path=(
            None
            if config.runtime.topology_mode == "tp1_dp1"
            else str(root / "unsigned-formal-gang-terminal.json")
        ),
        fatal_output_path=str(root / "fatal.json"),
        single_operator_execution_rebuild_source=input_binding,
        serving_execution_policy=execution_policy,
        process_hard_timeout_ns=_registered_process_hard_timeout_ns(
            policy=execution_policy,
            schedule=schedule,
        ),
    )
    output = root / "formal-serving-run-plan.json"
    publish_canonical_json_no_replace(output, plan.to_dict())
    replayed, _launch, _schedule = _load_formal_single_operator_trusted_run_plan(output)
    if replayed != plan:
        raise RuntimeError("single-operator E5 failure run plan changed")
    return plan


def materialize_formal_single_operator_serving_run_plan(
    *,
    early_run_plan_inputs_path: str | Path,
) -> FormalServingRunPlan:
    """Materialize one trusted plan from an early-stage input bundle."""

    from lightcone_spec.experiments.formal_single_operator_early_execution import (
        FormalSingleOperatorEarlyRunPlanInputs,
    )

    input_binding = CanonicalJsonProofBinding.bind(early_run_plan_inputs_path)
    inputs = FormalSingleOperatorEarlyRunPlanInputs.from_dict(input_binding.reopen())
    return _materialize_formal_single_operator_direct_serving_run_plan(
        inputs=inputs,
        input_binding=input_binding,
        expected_input_name="formal-single-operator-early-run-plan-inputs.json",
    )


def materialize_formal_single_operator_downstream_serving_run_plan(
    *,
    downstream_run_plan_inputs_path: str | Path,
) -> FormalServingRunPlan:
    """Materialize one trusted plan from a downstream source-owned bundle."""

    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        FormalSingleOperatorDownstreamRunPlanInputs,
    )

    input_binding = CanonicalJsonProofBinding.bind(downstream_run_plan_inputs_path)
    inputs = FormalSingleOperatorDownstreamRunPlanInputs.from_dict(
        input_binding.reopen()
    )
    return _materialize_formal_single_operator_direct_serving_run_plan(
        inputs=inputs,
        input_binding=input_binding,
        expected_input_name=("formal-single-operator-downstream-run-plan-inputs.json"),
    )


def materialize_formal_single_operator_prepared_downstream_serving_run_plan(
    *,
    prepared_downstream_run_plan_inputs_path: str | Path,
) -> FormalServingRunPlan:
    """Materialize one trusted plan from an exact prepared launch/schedule row."""

    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        FormalSingleOperatorPreparedDownstreamRunPlanInputs,
    )

    input_binding = CanonicalJsonProofBinding.bind(
        prepared_downstream_run_plan_inputs_path
    )
    inputs = FormalSingleOperatorPreparedDownstreamRunPlanInputs.from_dict(
        input_binding.reopen()
    )
    return _materialize_formal_single_operator_direct_serving_run_plan(
        inputs=inputs,
        input_binding=input_binding,
        expected_input_name=("formal-single-operator-prepared-downstream-inputs.json"),
    )


def materialize_formal_serving_run_plan(
    *,
    execution_binding: FormalServingExecutionBinding,
    content_verification_receipt_path: str | Path,
    workload_authority_path: str | Path,
    materialization_path: str | Path,
    compile_launch_manifest_path: str | Path,
    private_output_root: str | Path,
    now_ns: int,
    verified_nextn_tp2_authority: VerifiedNextNTp2Authority | None = None,
) -> FormalServingRunPlan:
    """Public operator materializer; no request/port/argv arguments exist."""

    verified = require_verified_formal_serving_execution_binding(execution_binding)
    root = _private_output_root(private_output_root)
    current_binding = (
        verified if type(verified) is FormalSingleOperatorExecutionBinding else None
    )
    if current_binding is not None:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            load_formal_single_operator_execution_source,
        )

        current_source = load_formal_single_operator_execution_source(
            current_binding.execution_source.absolute_path
        )
        if (
            CanonicalJsonProofBinding.bind(content_verification_receipt_path)
            != current_binding.content_verification_receipt_source
            or CanonicalJsonProofBinding.bind(compile_launch_manifest_path)
            != current_binding.compile_launch_manifest
            or CanonicalJsonProofBinding.bind(materialization_path)
            != current_source.materialization_source
        ):
            raise ValueError(
                "current formal plan inputs differ from the verified execution source"
            )
    schedule = materialize_formal_serving_request_schedule(
        execution_binding=verified,
        content_verification_receipt_path=content_verification_receipt_path,
        workload_authority_path=workload_authority_path,
        materialization_path=materialization_path,
        compile_launch_manifest_path=compile_launch_manifest_path,
        private_output_root=root,
        now_ns=now_ns,
    )
    schedule_path = root / "formal-request-schedule-receipt.json"
    schedule_binding = CanonicalJsonProofBinding.bind(
        schedule_path,
        semantic_sha256=schedule.sha256,
    )
    launch = CompileLaunchManifest.load(compile_launch_manifest_path)
    launch_binding = CanonicalJsonProofBinding.bind(
        compile_launch_manifest_path,
        semantic_sha256=launch.sha256,
    )
    rebuild_source_binding: CanonicalJsonProofBinding | None = None
    if current_binding is not None:
        rebuild_source = _single_operator_execution_rebuild_source(current_binding)
        rebuild_source_path = (
            root / "formal-single-operator-execution-rebuild-source.json"
        )
        publish_canonical_json_no_replace(
            rebuild_source_path,
            rebuild_source.to_dict(),
        )
        rebuild_source_binding = CanonicalJsonProofBinding.bind(
            rebuild_source_path,
            semantic_sha256=rebuild_source.sha256,
        )
        if (
            revalidate_formal_single_operator_execution_rebuild_source(
                rebuild_source_path
            )
            != rebuild_source
        ):
            raise RuntimeError(
                "single-operator execution rebuild source failed codec replay"
            )
    subject = verified.subject
    execution_policy = (
        _registered_serving_execution_policy(stage=subject.stage, schedule=schedule)
        if current_binding is not None
        else None
    )
    if current_binding is not None and execution_policy is None:
        raise ValueError("current serving plan lacks a registered policy")
    plan = FormalServingRunPlan(
        schema_version=4 if current_binding is not None else 1,
        kind="formal_serving_run_plan",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=verified.sha256,
        subject_sha256=subject.sha256,
        materialized_cell_id=subject.materialized_cell_id,
        stage=subject.stage,
        method=subject.method,
        topology_mode=subject.topology_mode,
        inventory_sha256=subject.inventory_sha256,
        gpu_uuids=subject.gpu_uuids,
        runtime_gpu_proof_sha256s=verified.runtime_gpu_proof_sha256s,
        runtime_gpu_proof_artifacts=subject.runtime_gpu_proof_artifacts,
        nextn_tp2_authority_sha256=(
            None
            if verified_nextn_tp2_authority is None
            else verified_nextn_tp2_authority.sha256
        ),
        launch_manifest=launch_binding,
        request_schedule_receipt=schedule_binding,
        single_operator_execution_rebuild_source=rebuild_source_binding,
        native_terminal_binding=_expected_native_terminal_binding(
            execution_binding=verified,
            schedule=schedule,
        ),
        private_output_root=str(root),
        terminal_output_path=str(root / "unsigned-native-terminal.json"),
        native_itl_pointer_output_path=str(root / "unsigned-native-itl.json"),
        live_run_receipt_output_path=str(root / "unsigned-live-run.json"),
        lifecycle_timing_output_path=str(root / "unsigned-lifecycle.json"),
        server_log_output_path=str(root / "server.log"),
        server_stdout_output_path=str(root / "stdout.log"),
        server_stderr_output_path=str(root / "stderr.log"),
        junit_output_path=str(root / "junit.xml"),
        before_gpu_snapshot_output_path=str(root / "gpu-before.json"),
        ready_gpu_snapshot_output_path=str(root / "gpu-ready.json"),
        after_gpu_snapshot_output_path=str(root / "gpu-after.json"),
        formal_gang_terminal_output_path=(
            None
            if subject.topology_mode == "tp1_dp1"
            else str(root / "unsigned-formal-gang-terminal.json")
        ),
        fatal_output_path=str(root / "fatal.json"),
        serving_execution_policy=execution_policy,
        process_hard_timeout_ns=(
            None
            if execution_policy is None
            else _registered_process_hard_timeout_ns(
                policy=execution_policy,
                schedule=schedule,
            )
        ),
    )
    revalidate_formal_serving_run_plan(
        plan,
        execution_binding=verified,
        verified_nextn_tp2_authority=verified_nextn_tp2_authority,
    )
    plan_path = root / "formal-serving-run-plan.json"
    publish_canonical_json_no_replace(plan_path, plan.to_dict())
    rebound = FormalServingRunPlan.from_dict(
        CanonicalJsonProofBinding.bind(plan_path).reopen()
    )
    if rebound != plan or rebound.sha256 != plan.sha256:
        raise RuntimeError("formal serving run plan failed codec replay")
    return plan


def load_formal_serving_run_plan(
    path: str | Path,
    *,
    execution_binding: FormalServingExecutionBinding,
    verified_nextn_tp2_authority: VerifiedNextNTp2Authority | None = None,
) -> FormalServingRunPlan:
    binding = CanonicalJsonProofBinding.bind(path)
    plan = FormalServingRunPlan.from_dict(binding.reopen())
    if plan.sha256 != binding.semantic_sha256:
        raise ValueError("formal serving run plan semantic digest differs")
    revalidate_formal_serving_run_plan(
        plan,
        execution_binding=execution_binding,
        verified_nextn_tp2_authority=verified_nextn_tp2_authority,
    )
    return plan


def _consume_physical_launch_admission(
    *,
    launch_admission_path: str | Path,
    plan_path: str | Path,
    execution_binding: FormalServingExecutionBinding | None,
) -> tuple[
    int,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
]:
    """Consume either the staged authority or the trusted single-operator gate."""

    candidate = CanonicalJsonProofBinding.bind(launch_admission_path)
    value = candidate.reopen()
    if value.get("kind") == "formal_single_operator_admission":
        from lightcone_spec.orchestration.formal_single_operator_admission import (
            consume_formal_single_operator_admission,
            validate_formal_single_operator_admission,
        )

        admission = validate_formal_single_operator_admission(
            launch_admission_path,
            plan_path=plan_path,
        )
        consumption = consume_formal_single_operator_admission(
            admission,
            consumed_ns=time.time_ns(),
        )
        return (
            admission.process_hard_timeout_ns,
            candidate,
            consumption,
            consumption,
        )
    from lightcone_spec.orchestration.formal_launch_admission import (
        consume_formal_stage_launch_admission,
        validate_formal_stage_launch_admission,
    )

    if execution_binding is None:
        raise ValueError(
            "trusted single-operator execution requires its local admission"
        )

    admission = validate_formal_stage_launch_admission(
        launch_admission_path,
        execution_binding=execution_binding,
        run_plan_path=plan_path,
        current_ns=time.time_ns(),
    )
    consumption = consume_formal_stage_launch_admission(
        admission,
        consumed_ns=time.time_ns(),
    )
    return (
        admission.artifact.hard_timeout_ns,
        CanonicalJsonProofBinding.bind(
            launch_admission_path,
            semantic_sha256=admission.artifact.sha256,
        ),
        consumption,
        admission.artifact.budget_consumption,
    )


def _publish_formal_serving_junit(
    *,
    output_path: str | Path,
    topology_mode: str,
    request_ids: tuple[str, ...],
) -> EvidenceFileBinding:
    """Publish deterministic successful request coverage after durable completion."""

    if not request_ids or len(request_ids) != len(set(request_ids)):
        raise ValueError("formal serving JUnit request coverage differs")
    cases = "".join(
        f"<testcase classname={quoteattr('lightcone.' + topology_mode)} "
        f"name={quoteattr(request_id)}/>"
        for request_id in request_ids
    )
    payload = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name="lightcone-formal-serving" tests="{len(request_ids)}" '
        'failures="0" errors="0" skipped="0">'
        f"{cases}</testsuite>\n"
    ).encode()
    destination = Path(output_path)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return EvidenceFileBinding.bind(destination, label="formal serving JUnit")


def _single_operator_tp1_runtime_authorities(
    binding: FormalSingleOperatorExecutionBinding,
) -> tuple[
    VerifiedNativeRuntimeGpuProof | None,
    str | None,
    VerifiedEagle3E0ExecutionAuthority | None,
    str | None,
    VerifiedNativeRuntimeGpuProof | None,
    str | None,
]:
    """Select all TP1 launch authorities from the rebuilt binding itself."""

    if type(binding) is not FormalSingleOperatorExecutionBinding:
        raise TypeError("single-operator authority projection requires current binding")
    config = binding.run_config
    proofs = binding.verified_native_gpu_proofs

    def exact_suite(suite_id: str) -> VerifiedNativeRuntimeGpuProof:
        rows = tuple(row for row in proofs if row.suite_id == suite_id)
        if len(rows) != 1:
            raise ValueError(
                f"single-operator binding lacks one exact {suite_id} GPU proof"
            )
        return rows[0]

    graph_proof: VerifiedNativeRuntimeGpuProof | None = None
    graph_source: str | None = None
    if config.runtime.cuda_graph_mode == "fixed_address_publication_v1":
        graph_proof = exact_suite("native_hot_path_tp1")
        graph_source = graph_proof.source_identity_sha256

    adaptation = config.adaptation
    chronobelief_proof: VerifiedNativeRuntimeGpuProof | None = None
    chronobelief_source: str | None = None
    if adaptation is not None and adaptation.optimizer.name == "chronobelief":
        chronobelief_proof = exact_suite("chronobelief_gpu_parity")
        chronobelief_source = chronobelief_proof.source_identity_sha256

    if adaptation is not None and config.model.algorithm == "EAGLE3":
        raise ValueError(
            "single-operator adaptive EAGLE3 execution authority is not yet "
            "durably mapped"
        )
    return (
        graph_proof,
        graph_source,
        None,
        None,
        chronobelief_proof,
        chronobelief_source,
    )


def _trusted_single_operator_eagle3_authority_from_plan(
    plan: FormalServingRunPlan,
) -> object | None:
    """Reopen the optional tagged empirical authority from a prepared plan."""

    source = plan.single_operator_execution_rebuild_source
    if source is None:
        return None
    raw = source.reopen()
    if raw.get("kind") != (
        "formal_single_operator_prepared_downstream_run_plan_inputs"
    ):
        return None
    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        load_trusted_single_operator_eagle3_execution_authority,
    )
    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        revalidate_formal_single_operator_prepared_downstream_run_plan_inputs,
    )

    direct = revalidate_formal_single_operator_prepared_downstream_run_plan_inputs(
        source.absolute_path,
        current_ns=time.time_ns(),
    )
    binding = direct.trusted_eagle3_execution_authority
    if binding is None:
        return None
    authority = load_trusted_single_operator_eagle3_execution_authority(
        binding.absolute_path
    )
    if (
        authority.compile_launch_manifest != plan.launch_manifest
        or authority.materialized_cell_id != plan.materialized_cell_id
        or authority.method != plan.method
        or authority.inventory_sha256 != plan.inventory_sha256
        or authority.gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("trusted EAGLE3 authority differs from physical plan")
    return authority


def _trusted_single_operator_chronobelief_proof_from_plan(
    plan: FormalServingRunPlan,
) -> object | None:
    """Deep-replay the optional empirical parity proof at physical launch."""

    source = plan.single_operator_execution_rebuild_source
    if source is None:
        return None
    raw = source.reopen()
    if raw.get("kind") != (
        "formal_single_operator_prepared_downstream_run_plan_inputs"
    ):
        return None
    from lightcone_spec.experiments.formal_single_operator_chronobelief import (
        revalidate_trusted_single_operator_chronobelief_for_prepared_launch,
    )
    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        revalidate_formal_single_operator_prepared_downstream_run_plan_inputs,
    )

    direct = revalidate_formal_single_operator_prepared_downstream_run_plan_inputs(
        source.absolute_path,
        current_ns=time.time_ns(),
    )
    binding = direct.trusted_chronobelief_gpu_parity_proof
    if binding is None:
        return None
    if (
        direct.compile_launch_manifest != plan.launch_manifest
        or direct.materialized_cell_id != plan.materialized_cell_id
        or direct.inventory.semantic_sha256 != plan.inventory_sha256
    ):
        raise ValueError("trusted ChronoBelief input differs from physical plan")
    proof = revalidate_trusted_single_operator_chronobelief_for_prepared_launch(
        proof_path=binding.absolute_path,
        execution_source_path=direct.execution_source.absolute_path,
        prepared_launch_path=plan.launch_manifest.absolute_path,
    )
    if (
        proof.inventory_sha256 != plan.inventory_sha256
        or len(plan.gpu_uuids) != 1
        or plan.gpu_uuids[0] not in proof.qualified_gpu_uuids
    ):
        raise ValueError("trusted ChronoBelief proof differs from physical plan")
    return proof


async def execute_formal_tp1_serving_run_plan(
    *,
    plan_path: str | Path,
    launch_admission_path: str | Path | None,
    execution_binding: FormalServingExecutionBinding | None = None,
    nvidia_smi_tool: PinnedNvidiaSmiTool,
    verified_native_graph_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    expected_graph_source_identity_sha256: str | None = None,
    verified_eagle3_e0_execution_authority: (
        VerifiedEagle3E0ExecutionAuthority | None
    ) = None,
    expected_eagle3_source_identity_sha256: str | None = None,
    verified_chronobelief_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    expected_chronobelief_source_identity_sha256: str | None = None,
) -> ValidatedUnsignedPinnedSglangServingRun:
    """Reach the real TP1 runner from a sealed plan without legacy diagnostics."""

    if execution_binding is None:
        plan, _launch, schedule = _load_formal_single_operator_trusted_run_plan(
            plan_path
        )
        resolved_binding = None
        trusted_eagle3_authority = _trusted_single_operator_eagle3_authority_from_plan(
            plan
        )
        trusted_chronobelief_proof = (
            _trusted_single_operator_chronobelief_proof_from_plan(plan)
        )
        if any(
            value is not None
            for value in (
                verified_native_graph_gpu_proof,
                expected_graph_source_identity_sha256,
                verified_eagle3_e0_execution_authority,
                expected_eagle3_source_identity_sha256,
                verified_chronobelief_gpu_proof,
                expected_chronobelief_source_identity_sha256,
            )
        ):
            raise ValueError(
                "trusted single-operator TP1 does not accept legacy authorities"
            )
    else:
        trusted_eagle3_authority = None
        trusted_chronobelief_proof = None
        resolved_binding = require_verified_formal_serving_execution_binding(
            execution_binding
        )
        plan = load_formal_serving_run_plan(
            plan_path,
            execution_binding=resolved_binding,
        )
        _launch, schedule = revalidate_formal_serving_run_plan(
            plan,
            execution_binding=resolved_binding,
        )
    if type(resolved_binding) is FormalSingleOperatorExecutionBinding:
        derived = _single_operator_tp1_runtime_authorities(resolved_binding)
        supplied = (
            verified_native_graph_gpu_proof,
            expected_graph_source_identity_sha256,
            verified_eagle3_e0_execution_authority,
            expected_eagle3_source_identity_sha256,
            verified_chronobelief_gpu_proof,
            expected_chronobelief_source_identity_sha256,
        )
        if any(
            value is not None and value != expected
            for value, expected in zip(supplied, derived, strict=True)
        ):
            raise ValueError(
                "single-operator TP1 caller authority differs from rebuilt binding"
            )
        (
            verified_native_graph_gpu_proof,
            expected_graph_source_identity_sha256,
            verified_eagle3_e0_execution_authority,
            expected_eagle3_source_identity_sha256,
            verified_chronobelief_gpu_proof,
            expected_chronobelief_source_identity_sha256,
        ) = derived
    if plan.topology_mode != "tp1_dp1":
        raise ValueError("TP1 formal operator rejects a distributed run plan")
    warmup = tuple(
        row.request
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "warmup"
    )
    scored = tuple(
        row.request
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    execution_policy = plan.serving_execution_policy
    if plan.schema_version == 4:
        timeout_seconds = _registered_plan_process_hard_timeout_seconds(
            plan,
            allowed_topologies=frozenset({"tp1_dp1"}),
        )
        if resolved_binding is None:
            launch_admission = None
            launch_consumption = None
            budget_consumption = None
        else:
            if launch_admission_path is None:
                raise ValueError("current bound TP1 execution requires admission")
            (
                admitted_timeout_ns,
                launch_admission,
                launch_consumption,
                budget_consumption,
            ) = _consume_physical_launch_admission(
                launch_admission_path=launch_admission_path,
                plan_path=plan_path,
                execution_binding=resolved_binding,
            )
            if admitted_timeout_ns != plan.process_hard_timeout_ns:
                raise ValueError("TP1 admission timeout differs from registered plan")
    elif resolved_binding is None:
        timeout_seconds = _SINGLE_OPERATOR_PROCESS_TIMEOUT_SECONDS
        launch_admission = None
        launch_consumption = None
        budget_consumption = None
    else:
        if launch_admission_path is None:
            raise ValueError("legacy formal TP1 execution requires admission")
        (
            hard_timeout_ns,
            launch_admission,
            launch_consumption,
            budget_consumption,
        ) = _consume_physical_launch_admission(
            launch_admission_path=launch_admission_path,
            plan_path=plan_path,
            execution_binding=resolved_binding,
        )
        timeout_seconds = hard_timeout_ns / 1_000_000_000
    result = await execute_unsigned_native_serving_run(
        launch_manifest_path=plan.launch_manifest.absolute_path,
        binding=plan.native_terminal_binding,
        warmup_requests=warmup,
        scored_requests=scored,
        terminal_output_path=plan.terminal_output_path,
        native_itl_pointer_output_path=plan.native_itl_pointer_output_path,
        live_run_receipt_output_path=plan.live_run_receipt_output_path,
        server_log_output_path=plan.server_log_output_path,
        server_stdout_output_path=plan.server_stdout_output_path,
        server_stderr_output_path=plan.server_stderr_output_path,
        nvidia_smi_tool=nvidia_smi_tool,
        before_gpu_snapshot_output_path=plan.before_gpu_snapshot_output_path,
        ready_gpu_snapshot_output_path=plan.ready_gpu_snapshot_output_path,
        after_gpu_snapshot_output_path=plan.after_gpu_snapshot_output_path,
        fatal_output_path=plan.fatal_output_path,
        timeout_seconds=timeout_seconds,
        formal_launch_admission=launch_admission,
        formal_launch_consumption=launch_consumption,
        budget_consumption=budget_consumption,
        verified_native_graph_gpu_proof=verified_native_graph_gpu_proof,
        expected_graph_source_identity_sha256=expected_graph_source_identity_sha256,
        verified_eagle3_e0_execution_authority=(verified_eagle3_e0_execution_authority),
        trusted_single_operator_eagle3_execution_authority=(trusted_eagle3_authority),
        expected_eagle3_source_identity_sha256=(expected_eagle3_source_identity_sha256),
        verified_chronobelief_gpu_proof=verified_chronobelief_gpu_proof,
        trusted_single_operator_chronobelief_gpu_parity_proof=(
            trusted_chronobelief_proof
        ),
        expected_chronobelief_source_identity_sha256=(
            expected_chronobelief_source_identity_sha256
        ),
        lifecycle_timing_output_path=plan.lifecycle_timing_output_path,
        execution_policy=execution_policy,
    )
    _publish_formal_serving_junit(
        output_path=plan.junit_output_path,
        topology_mode=plan.topology_mode,
        request_ids=tuple(row.request_id for row in scored),
    )
    return result


async def execute_formal_serving_run_plan(
    *,
    plan_path: str | Path,
    launch_admission_path: str | Path | None,
    execution_binding: FormalServingExecutionBinding | None = None,
    nvidia_smi_tool: PinnedNvidiaSmiTool,
    verified_nextn_tp2_authority: VerifiedNextNTp2Authority | None = None,
    verified_native_graph_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    expected_graph_source_identity_sha256: str | None = None,
    verified_eagle3_e0_execution_authority: (
        VerifiedEagle3E0ExecutionAuthority | None
    ) = None,
    expected_eagle3_source_identity_sha256: str | None = None,
    verified_chronobelief_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    expected_chronobelief_source_identity_sha256: str | None = None,
) -> ValidatedUnsignedPinnedSglangServingRun | ValidatedUnsignedFormalGangServingRun:
    """Closed-topology public operator adapter for every formal serving stage.

    The caller supplies only verifier-owned authorities and the sealed plan.
    Request rows, ports, paths, child argv, and live transports remain absent.
    """

    if execution_binding is None:
        plan, _launch, _schedule = _load_formal_single_operator_trusted_run_plan(
            plan_path
        )
        resolved_binding = None
        if verified_nextn_tp2_authority is not None:
            raise ValueError(
                "trusted single-operator execution does not accept legacy NEXTN authority"
            )
    else:
        resolved_binding = require_verified_formal_serving_execution_binding(
            execution_binding
        )
        rebuilt_nextn = resolved_binding.verified_nextn_tp2_authority
        if verified_nextn_tp2_authority is None:
            verified_nextn_tp2_authority = rebuilt_nextn
        elif (
            rebuilt_nextn is not None and verified_nextn_tp2_authority != rebuilt_nextn
        ):
            raise ValueError("formal operator NEXTN TP2 authority differs from rebuild")
        plan = load_formal_serving_run_plan(
            plan_path,
            execution_binding=resolved_binding,
            verified_nextn_tp2_authority=verified_nextn_tp2_authority,
        )
    if plan.topology_mode == "tp1_dp1":
        if verified_nextn_tp2_authority is not None:
            raise ValueError("TP1 formal operator rejects NEXTN TP2 authority")
        return await execute_formal_tp1_serving_run_plan(
            plan_path=plan_path,
            launch_admission_path=launch_admission_path,
            execution_binding=resolved_binding,
            nvidia_smi_tool=nvidia_smi_tool,
            verified_native_graph_gpu_proof=verified_native_graph_gpu_proof,
            expected_graph_source_identity_sha256=(
                expected_graph_source_identity_sha256
            ),
            verified_eagle3_e0_execution_authority=(
                verified_eagle3_e0_execution_authority
            ),
            expected_eagle3_source_identity_sha256=(
                expected_eagle3_source_identity_sha256
            ),
            verified_chronobelief_gpu_proof=verified_chronobelief_gpu_proof,
            expected_chronobelief_source_identity_sha256=(
                expected_chronobelief_source_identity_sha256
            ),
        )
    if any(
        value is not None
        for value in (
            verified_native_graph_gpu_proof,
            expected_graph_source_identity_sha256,
            verified_eagle3_e0_execution_authority,
            expected_eagle3_source_identity_sha256,
            verified_chronobelief_gpu_proof,
            expected_chronobelief_source_identity_sha256,
        )
    ):
        raise ValueError("distributed formal operator rejects TP1-only authorities")
    return await execute_formal_distributed_serving_run_plan(
        plan_path=plan_path,
        launch_admission_path=launch_admission_path,
        execution_binding=resolved_binding,
        nvidia_smi_tool=nvidia_smi_tool,
        verified_nextn_tp2_authority=verified_nextn_tp2_authority,
    )


async def execute_formal_single_operator_serving_run_plan(
    *,
    plan_path: str | Path,
    nvidia_smi_tool: PinnedNvidiaSmiTool,
) -> ValidatedUnsignedPinnedSglangServingRun | ValidatedUnsignedFormalGangServingRun:
    """Minimal current-mode operator; every execution authority comes from plan."""

    return await execute_formal_serving_run_plan(
        plan_path=plan_path,
        launch_admission_path=None,
        execution_binding=None,
        nvidia_smi_tool=nvidia_smi_tool,
    )


@dataclass(frozen=True)
class ValidatedUnsignedFormalGangServingRun:
    """Reopened unsigned distributed evidence; never a release authority."""

    receipt: CanonicalJsonProofBinding
    request_terminal: CanonicalJsonProofBinding
    native_itl_pointers: CanonicalJsonProofBinding
    formal_gang_terminal: CanonicalJsonProofBinding
    lifecycle_timing: CanonicalJsonProofBinding
    client_request_lifecycle: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        for value in (
            self.receipt,
            self.request_terminal,
            self.native_itl_pointers,
            self.formal_gang_terminal,
            self.lifecycle_timing,
            *(
                ()
                if self.client_request_lifecycle is None
                else (self.client_request_lifecycle,)
            ),
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("formal gang serving result lost a path binding")
            value.reopen()


def _formal_gang_schedule_rows(
    schedule: FormalServingRequestScheduleReceipt,
    *,
    phase: Literal["warmup", "scored"],
) -> list[dict[str, object]]:
    return [
        {
            "request_id": row.request.request_id,
            "cohort_sha256": row.request.cohort_sha256,
            "routed_dp_rank": row.routed_dp_rank,
        }
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == phase
    ]


def _validate_formal_gang_capability(
    value: object,
    *,
    plan: FormalServingRunPlan,
    launch: CompileLaunchManifest,
) -> dict[str, object]:
    row = _strict_object(
        "formal gang capability",
        value,
        {
            "schema_version",
            "kind",
            "hook",
            "protocol_sha256",
            "topology",
            "world_size",
            "method",
            "execution_plan_sha256",
            "rank_config_sha256",
            "run_nonce_sha256",
            "rank_capabilities",
            "rank_capability_sha256s",
        },
    )
    rank_rows = row["rank_capabilities"]
    rank_digests = row["rank_capability_sha256s"]
    if (
        row["schema_version"] != 1
        or row["kind"] != "sglang_formal_gang_capability"
        or row["hook"] != "sglang.lightcone_formal_gang_serving.v1"
        or row["protocol_sha256"] != FORMAL_GANG_SERVING_PROTOCOL_SHA256
        or row["topology"] != plan.topology_mode
        or row["world_size"] != 2
        or row["method"] != plan.method
        or row["execution_plan_sha256"]
        != plan.native_terminal_binding.execution_plan_sha256
        or row["rank_config_sha256"] != plan.native_terminal_binding.rank_config_sha256
        or row["run_nonce_sha256"] != plan.native_terminal_binding.run_nonce_sha256
        or type(rank_rows) is not list
        or type(rank_digests) is not list
        or len(rank_rows) != 2
        or len(rank_digests) != 2
    ):
        raise ValueError("formal gang capability differs from sealed plan")
    by_rank = {}
    for rank_row, digest in zip(rank_rows, rank_digests, strict=True):
        if (
            type(rank_row) is not dict
            or rank_row.get("rank") not in {0, 1}
            or rank_row.get("assignment_sha256") != launch.physical_assignment_sha256
            or rank_row.get("inventory_sha256") != plan.inventory_sha256
            or rank_row.get("gpu_uuid") not in plan.gpu_uuids
            or _sha256(rank_row) != digest
        ):
            raise ValueError("formal gang rank capability differs")
        by_rank[rank_row["rank"]] = rank_row
    if (
        set(by_rank) != {0, 1}
        or tuple(by_rank[index]["gpu_uuid"] for index in (0, 1)) != plan.gpu_uuids
    ):
        raise ValueError("formal gang rank/GPU order differs")
    return row


def _validate_formal_gang_transition(
    value: object,
    *,
    action: Literal["begin", "reset", "finalize"],
    plan: FormalServingRunPlan,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("formal gang transition must be an object")
    row = dict(value)
    if (
        row.get("schema_version") != 1
        or row.get("hook") != "sglang.lightcone_formal_gang_serving.v1"
        or row.get("protocol_sha256") != FORMAL_GANG_SERVING_PROTOCOL_SHA256
        or row.get("topology") != plan.topology_mode
        or row.get("world_size") != 2
    ):
        raise ValueError("formal gang transition identity differs")
    if action == "begin":
        if (
            set(row)
            != {
                "schema_version",
                "kind",
                "hook",
                "protocol_sha256",
                "topology",
                "world_size",
                "execution_plan_sha256",
                "schedule_sha256",
                "rank_begin_sha256s",
                "begin_sha256",
            }
            or row["kind"] != "sglang_formal_gang_begin"
            or row["execution_plan_sha256"]
            != plan.native_terminal_binding.execution_plan_sha256
            or type(row["rank_begin_sha256s"]) is not list
            or len(row["rank_begin_sha256s"]) != 2
        ):
            raise ValueError("formal gang begin differs")
        unsigned = dict(row)
        declared = unsigned.pop("begin_sha256")
        if _sha256(unsigned) != declared:
            raise ValueError("formal gang begin digest differs")
    else:
        if (
            row.get("kind") != "sglang_formal_gang_all_rank_terminal"
            or row.get("action") != f"formal_gang_{action}"
            or row.get("decision") != "COMMITTED"
            or row.get("published_ranks") != [0, 1]
            or row.get("reason_code") is not None
            or row.get("cross_replica_gradient_collective") is not False
            or type(row.get("rank_terminals")) is not list
            or len(row["rank_terminals"]) != 2
        ):
            raise RuntimeError("formal gang all-rank transition did not commit")
        unsigned = dict(row)
        declared = unsigned.pop("aggregate_sha256", None)
        if _sha256(unsigned) != declared:
            raise ValueError("formal gang aggregate digest differs")
    return row


def _spawn_formal_gang_server(
    launch: CompileLaunchManifest,
    *,
    binding: NativeTerminalRunBinding,
    stdout_file,
    stderr_file,
) -> subprocess.Popen[bytes]:
    environment = launch.child_environment()
    environment.update(
        {
            "LIGHTCONE_FORMAL_GANG_ENABLE": "1",
            "LIGHTCONE_FORMAL_GANG_ASSIGNMENT_SHA256": (
                launch.physical_assignment_sha256
            ),
            "LIGHTCONE_FORMAL_GANG_INVENTORY_SHA256": launch.inventory_sha256,
            "LIGHTCONE_FORMAL_GANG_EXECUTION_PLAN_SHA256": (
                binding.execution_plan_sha256
            ),
            "LIGHTCONE_FORMAL_GANG_RANK_CONFIG_SHA256": (binding.rank_config_sha256),
            "LIGHTCONE_FORMAL_GANG_RUN_NONCE_SHA256": binding.run_nonce_sha256,
        }
    )
    return subprocess.Popen(
        launch.server_argv,
        cwd=launch.patched_sglang_checkout,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,
        close_fds=True,
    )


def _phase_result_rows(phase_result) -> tuple[list[dict[str, object]], list[object]]:
    requests = []
    for row in phase_result.requests:
        row.validate()
        requests.append(
            {
                "request_id": row.request_id,
                "input_token_ids": list(row.input_token_ids),
                "output_token_ids": (
                    None if row.output_token_ids is None else list(row.output_token_ids)
                ),
                "terminal_status": row.terminal_status,
                "terminal_reason": row.terminal_reason,
                "submitted_to_server": row.submitted_to_server,
            }
        )
    pointers: list[object] = []
    for raw in phase_result.native_result_pointer_json:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("formal gang native pointer is not JSON") from error
        if type(value) is not dict or _sha256(
            {key: item for key, item in value.items() if key != "result_pointer_sha256"}
        ) != value.get("result_pointer_sha256"):
            raise ValueError("formal gang native pointer digest differs")
        pointers.append(value)
    return requests, pointers


def _publish_formal_gang_fatal(
    *,
    plan: FormalServingRunPlan,
    reason_code: str,
    error: BaseException,
    process: subprocess.Popen[bytes] | None,
    cleanup_error: BaseException | None,
    partial_paths: tuple[str, ...],
) -> CanonicalJsonProofBinding:
    partial = []
    for value in partial_paths:
        path = Path(value)
        if path.is_file() and not path.is_symlink():
            try:
                partial.append(CanonicalJsonProofBinding.bind(path).to_dict())
            except (OSError, TypeError, ValueError):
                pass
    payload = {
        "schema_version": 1,
        "kind": "unsigned_formal_gang_physical_fatal",
        "protocol_sha256": FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        "formal_execution_authorized": False,
        "plan_sha256": plan.sha256,
        "reason_code": reason_code,
        "error_type": type(error).__name__,
        "cleanup_error_type": (
            None if cleanup_error is None else type(cleanup_error).__name__
        ),
        "server_process_id": None if process is None else process.pid,
        "server_exit_code": None if process is None else process.poll(),
        "process_group_empty": (
            None
            if process is None
            else not _process_group_exists_for_formal_dispatch(process.pid)
        ),
        "partial_canonical_artifacts": partial,
    }
    publish_canonical_json_no_replace(plan.fatal_output_path, payload)
    return CanonicalJsonProofBinding.bind(plan.fatal_output_path)


def _process_group_exists_for_formal_dispatch(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def execute_formal_distributed_serving_run_plan(
    *,
    plan_path: str | Path,
    launch_admission_path: str | Path | None,
    execution_binding: FormalServingExecutionBinding | None = None,
    nvidia_smi_tool: PinnedNvidiaSmiTool,
    verified_nextn_tp2_authority: VerifiedNextNTp2Authority | None = None,
) -> ValidatedUnsignedFormalGangServingRun:
    """Run one TP2 or DP2 plan through the patched source-owned producer."""

    if execution_binding is None:
        plan, launch, schedule = _load_formal_single_operator_trusted_run_plan(
            plan_path
        )
        resolved_binding = None
        if verified_nextn_tp2_authority is not None:
            raise ValueError(
                "trusted single-operator distributed run rejects legacy authority"
            )
    else:
        resolved_binding = require_verified_formal_serving_execution_binding(
            execution_binding
        )
        rebuilt_nextn = resolved_binding.verified_nextn_tp2_authority
        if verified_nextn_tp2_authority is None:
            verified_nextn_tp2_authority = rebuilt_nextn
        elif (
            rebuilt_nextn is not None and verified_nextn_tp2_authority != rebuilt_nextn
        ):
            raise ValueError("distributed NEXTN TP2 authority differs from rebuild")
        plan = load_formal_serving_run_plan(
            plan_path,
            execution_binding=resolved_binding,
            verified_nextn_tp2_authority=verified_nextn_tp2_authority,
        )
        launch, schedule = revalidate_formal_serving_run_plan(
            plan,
            execution_binding=resolved_binding,
            verified_nextn_tp2_authority=verified_nextn_tp2_authority,
        )
    if plan.topology_mode not in {"tp2_dp1", "tp1_dp2"}:
        raise ValueError("distributed formal operator requires TP2 or DP2")
    config = load_run_config(launch.run_config_path)
    if config.runtime.topology_mode != plan.topology_mode or len(launch.gpu_uuids) != 2:
        raise ValueError("distributed formal launch topology differs")
    warmup = tuple(
        row.request
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "warmup"
    )
    scored = tuple(
        row.request
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    if not warmup or not scored:
        raise ValueError("distributed formal run requires warmup and scored coverage")
    client_lifecycle_path = (
        Path(plan.private_output_root) / "client-request-lifecycle.json"
    )
    output_paths = tuple(
        value
        for value in (
            plan.terminal_output_path,
            plan.native_itl_pointer_output_path,
            plan.live_run_receipt_output_path,
            plan.lifecycle_timing_output_path,
            plan.server_log_output_path,
            plan.server_stdout_output_path,
            plan.server_stderr_output_path,
            plan.junit_output_path,
            plan.before_gpu_snapshot_output_path,
            plan.ready_gpu_snapshot_output_path,
            plan.after_gpu_snapshot_output_path,
            plan.formal_gang_terminal_output_path,
            plan.fatal_output_path,
            str(client_lifecycle_path),
        )
        if value is not None
    )
    if any(os.path.lexists(value) for value in output_paths):
        raise FileExistsError("distributed formal output already exists")
    executable = Path(launch.server_argv[0])
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or executable.is_symlink()
    ):
        raise ValueError("distributed formal server executable is invalid")
    execution_policy = plan.serving_execution_policy
    if plan.schema_version == 4:
        timeout = _registered_plan_process_hard_timeout_seconds(
            plan,
            allowed_topologies=frozenset({"tp2_dp1", "tp1_dp2"}),
        )
        if resolved_binding is None:
            launch_admission = None
            launch_consumption = None
            budget_consumption = None
        else:
            if launch_admission_path is None:
                raise ValueError("current bound distributed run requires admission")
            (
                admitted_timeout_ns,
                launch_admission,
                launch_consumption,
                budget_consumption,
            ) = _consume_physical_launch_admission(
                launch_admission_path=launch_admission_path,
                plan_path=plan_path,
                execution_binding=resolved_binding,
            )
            if admitted_timeout_ns != plan.process_hard_timeout_ns:
                raise ValueError(
                    "distributed admission timeout differs from registered plan"
                )
    elif resolved_binding is None:
        timeout = _SINGLE_OPERATOR_PROCESS_TIMEOUT_SECONDS
        launch_admission = None
        launch_consumption = None
        budget_consumption = None
    else:
        if launch_admission_path is None:
            raise ValueError("legacy distributed execution requires admission")
        (
            hard_timeout_ns,
            launch_admission,
            launch_consumption,
            budget_consumption,
        ) = _consume_physical_launch_admission(
            launch_admission_path=launch_admission_path,
            plan_path=plan_path,
            execution_binding=resolved_binding,
        )
        timeout = hard_timeout_ns / 1_000_000_000
    launch_admission_value = (
        None if launch_admission is None else launch_admission.to_dict()
    )
    launch_consumption_value = (
        None if launch_consumption is None else launch_consumption.to_dict()
    )
    budget_consumption_value = (
        None if budget_consumption is None else budget_consumption.to_dict()
    )
    _require_port_unused(launch.localhost_port)
    before_path = Path(plan.before_gpu_snapshot_output_path)
    ready_path = Path(plan.ready_gpu_snapshot_output_path)
    after_path = Path(plan.after_gpu_snapshot_output_path)
    process: subprocess.Popen[bytes] | None = None
    transport: PinnedBenchServingTransport | None = None
    log_file = None
    stdout_file = None
    stderr_file = None
    before_snapshot: CanonicalJsonProofBinding | None = None
    ready_snapshot: CanonicalJsonProofBinding | None = None
    after_snapshot: CanonicalJsonProofBinding | None = None
    execution_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    process_exit_code: int | None = None
    cleanup_kind: str | None = None
    process_exited_ns: int | None = None
    client_lifecycle_binding: CanonicalJsonProofBinding | None = None
    phase_edges_ns: dict[str, int] = {"execution_started_ns": time.monotonic_ns()}
    execution_task = asyncio.current_task()
    if execution_task is None:  # pragma: no cover - asyncio always owns this call
        raise RuntimeError("distributed serving lacks an owning asyncio task")
    hard_timeout_state = {"fired": False}

    def trigger_hard_timeout() -> None:
        hard_timeout_state["fired"] = True
        execution_task.cancel()

    hard_timeout_handle = asyncio.get_running_loop().call_later(
        timeout,
        trigger_hard_timeout,
    )
    try:
        before_snapshot = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=nvidia_smi_tool,
            gpu_uuids=launch.gpu_uuids,
            inventory_sha256=launch.inventory_sha256,
            phase="before",
            output_path=before_path,
        )
        log_descriptor = os.open(
            plan.server_log_output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        log_file = os.fdopen(log_descriptor, "wb", buffering=0)
        log_file.write(b"source-owned formal gang serving started\n")
        stdout_descriptor = os.open(
            plan.server_stdout_output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        stderr_descriptor = os.open(
            plan.server_stderr_output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        stdout_file = os.fdopen(stdout_descriptor, "wb", buffering=0)
        stderr_file = os.fdopen(stderr_descriptor, "wb", buffering=0)
        stdout_file.write(b"source-owned formal gang stdout started\n")
        stderr_file.write(b"source-owned formal gang stderr started\n")
        process = await asyncio.to_thread(
            _spawn_formal_gang_server,
            launch,
            binding=plan.native_terminal_binding,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )
        await asyncio.to_thread(
            _wait_server_ready,
            process,
            port=launch.localhost_port,
            timeout_seconds=min(timeout, 600.0),
        )
        phase_edges_ns["server_ready_ns"] = time.monotonic_ns()
        ready_snapshot = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=nvidia_smi_tool,
            gpu_uuids=launch.gpu_uuids,
            inventory_sha256=launch.inventory_sha256,
            phase="ready",
            output_path=ready_path,
            shared_server_process_group_id=process.pid,
        )
        transport = PinnedBenchServingTransport.from_checkout(
            launch.patched_sglang_checkout
        )
        if type(transport) is not PinnedBenchServingTransport:
            raise TypeError("distributed formal run requires exact pinned transport")
        await transport.open(
            request_timeout_s=min(timeout, 600.0),
            abort_timeout_s=30.0,
        )
        base_url = f"http://127.0.0.1:{launch.localhost_port}"
        transport.bind_native_admin_base_url(base_url)
        await _observe_live_server_execution_policy(transport=transport, config=config)
        capability = _validate_formal_gang_capability(
            await transport.get_json("/v1/lightcone-spec/formal-gang/capability"),
            plan=plan,
            launch=launch,
        )
        phase_edges_ns["begin_started_ns"] = time.monotonic_ns()
        begin = _validate_formal_gang_transition(
            await transport.post_json(
                "/v1/lightcone-spec/formal-gang/begin",
                {
                    "assignment_sha256": launch.physical_assignment_sha256,
                    "inventory_sha256": launch.inventory_sha256,
                    "execution_plan_sha256": (
                        plan.native_terminal_binding.execution_plan_sha256
                    ),
                    "rank_config_sha256": (
                        plan.native_terminal_binding.rank_config_sha256
                    ),
                    "run_nonce_sha256": plan.native_terminal_binding.run_nonce_sha256,
                    "method": plan.method,
                    "topology": plan.topology_mode,
                    "warmup_requests": _formal_gang_schedule_rows(
                        schedule, phase="warmup"
                    ),
                    "scored_requests": _formal_gang_schedule_rows(
                        schedule, phase="scored"
                    ),
                },
            ),
            action="begin",
            plan=plan,
        )
        phase_edges_ns["begin_finished_ns"] = time.monotonic_ns()
        phase_edges_ns["warmup_started_ns"] = time.monotonic_ns()
        warmup_result = await _execute_source_owned_phase(
            "warmup",
            warmup,
            concurrency=config.runtime.max_running_requests,
            transport=transport,
            base_url=base_url,
            served_model=config.model.target,
            execution_policy=execution_policy,
        )
        phase_edges_ns["warmup_finished_ns"] = time.monotonic_ns()
        phase_edges_ns["reset_started_ns"] = time.monotonic_ns()
        reset = _validate_formal_gang_transition(
            await transport.post_json(
                "/v1/lightcone-spec/formal-gang/reset",
                {"rank_begin_sha256s": begin["rank_begin_sha256s"]},
            ),
            action="reset",
            plan=plan,
        )
        phase_edges_ns["reset_finished_ns"] = time.monotonic_ns()
        phase_edges_ns["scored_started_ns"] = time.monotonic_ns()
        scored_result = await _execute_source_owned_phase(
            "scored",
            scored,
            concurrency=config.runtime.max_running_requests,
            transport=transport,
            base_url=base_url,
            served_model=config.model.target,
            execution_policy=execution_policy,
        )
        phase_edges_ns["scored_finished_ns"] = time.monotonic_ns()
        client_lifecycle_rows = (
            None
            if execution_policy is None
            else [
                *warmup_result.client_lifecycle_rows,
                *scored_result.client_lifecycle_rows,
            ]
        )
        finalize_payload = {
            "rank_reset_sha256s": reset["rank_reset_sha256s"],
        }
        if client_lifecycle_rows is not None:
            finalize_payload.update(
                {
                    "client_lifecycle_rows": client_lifecycle_rows,
                    "client_lifecycle_sha256": _sha256(client_lifecycle_rows),
                }
            )
        phase_edges_ns["finalize_started_ns"] = time.monotonic_ns()
        final = _validate_formal_gang_transition(
            await transport.post_json(
                "/v1/lightcone-spec/formal-gang/finalize",
                finalize_payload,
            ),
            action="finalize",
            plan=plan,
        )
        phase_edges_ns["finalize_finished_ns"] = time.monotonic_ns()
        warmup_rows, warmup_pointers = _phase_result_rows(warmup_result)
        scored_rows, scored_pointers = _phase_result_rows(scored_result)
        if execution_policy is not None:
            client_lifecycle_binding = publish_scalable_client_request_lifecycle(
                output_path=client_lifecycle_path,
                run_binding_sha256=_sha256(
                    plan.native_terminal_binding.begin_payload()
                ),
                execution_policy_sha256=execution_policy.sha256,
                rows=list(client_lifecycle_rows),
            )
        terminal_value = {
            "schema_version": 1,
            "kind": "unsigned_formal_gang_request_terminal",
            "protocol_sha256": FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
            "formal_execution_authorized": False,
            "plan_sha256": plan.sha256,
            "formal_launch_admission": launch_admission_value,
            "formal_launch_consumption": launch_consumption_value,
            "budget_consumption": budget_consumption_value,
            "capability_sha256": _sha256(capability),
            "begin_sha256": begin["begin_sha256"],
            "reset_sha256": reset["aggregate_sha256"],
            "finalize_sha256": final["aggregate_sha256"],
            "warmup_requests": warmup_rows,
            "scored_requests": scored_rows,
        }
        publish_scalable_formal_gang_request_terminal(
            output_path=plan.terminal_output_path,
            legacy_terminal=terminal_value,
        )
        pointer_value = {
            "schema_version": 1,
            "kind": "unsigned_formal_gang_native_itl_pointer_bundle",
            "protocol_sha256": FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
            "formal_execution_authorized": False,
            "plan_sha256": plan.sha256,
            "formal_launch_admission": launch_admission_value,
            "formal_launch_consumption": launch_consumption_value,
            "budget_consumption": budget_consumption_value,
            "warmup_pointers": warmup_pointers,
            "scored_pointers": scored_pointers,
        }
        publish_scalable_formal_gang_itl_bundle(
            output_path=plan.native_itl_pointer_output_path,
            legacy_bundle=pointer_value,
        )
        assert plan.formal_gang_terminal_output_path is not None
        publish_scalable_formal_gang_terminal(
            output_path=plan.formal_gang_terminal_output_path,
            legacy_terminal=final,
        )
    except BaseException as error:  # noqa: BLE001 - cleanup must always run
        execution_error = (
            TimeoutError("registered distributed process hard timeout expired")
            if hard_timeout_state["fired"] and isinstance(error, asyncio.CancelledError)
            else error
        )
    finally:
        if transport is not None:
            try:
                await transport.close()
            except BaseException as error:  # noqa: BLE001 - preserve cleanup evidence
                cleanup_error = cleanup_error or error
        if process is not None:
            try:
                (
                    process_exit_code,
                    cleanup_kind,
                    process_exited_ns,
                ) = await asyncio.to_thread(_terminate_process_group, process)
                phase_edges_ns["process_exited_ns"] = process_exited_ns
                phase_edges_ns["process_group_empty_checked_ns"] = time.monotonic_ns()
            except BaseException as error:  # noqa: BLE001 - preserve cleanup evidence
                cleanup_error = cleanup_error or error
        if log_file is not None:
            try:
                log_file.flush()
                os.fsync(log_file.fileno())
                log_file.close()
            except BaseException as error:  # noqa: BLE001 - preserve log evidence
                cleanup_error = cleanup_error or error
        for stream in (stdout_file, stderr_file):
            if stream is None:
                continue
            try:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            except BaseException as error:  # noqa: BLE001 - preserve log evidence
                cleanup_error = cleanup_error or error
        try:
            after_snapshot = await asyncio.to_thread(
                _capture_gpu_process_snapshot,
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="after",
                output_path=after_path,
            )
        except BaseException as error:  # noqa: BLE001 - retain failed snapshot
            after_snapshot = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="after",
                output_path=after_path,
                error=error,
            )
            cleanup_error = cleanup_error or error
    hard_timeout_handle.cancel()
    failure = cleanup_error or execution_error
    if failure is not None:
        if before_snapshot is None and not before_path.exists():
            before_snapshot = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="before",
                output_path=before_path,
                error=failure,
            )
        if ready_snapshot is None and not ready_path.exists():
            ready_error_kwargs = {
                "tool": nvidia_smi_tool,
                "gpu_uuids": launch.gpu_uuids,
                "inventory_sha256": launch.inventory_sha256,
                "phase": "ready",
                "output_path": ready_path,
                "error": failure,
            }
            if process is not None:
                ready_error_kwargs["expected_server_process_group_ids"] = tuple(
                    process.pid for _gpu_uuid in launch.gpu_uuids
                )
            ready_snapshot = _publish_gpu_snapshot_error(
                **ready_error_kwargs,
            )
        fatal = _publish_formal_gang_fatal(
            plan=plan,
            reason_code="distributed_formal_execution_failed",
            error=failure,
            process=process,
            cleanup_error=cleanup_error,
            partial_paths=(
                plan.terminal_output_path,
                plan.native_itl_pointer_output_path,
                plan.formal_gang_terminal_output_path or "",
                plan.before_gpu_snapshot_output_path,
                plan.ready_gpu_snapshot_output_path,
                plan.after_gpu_snapshot_output_path,
            ),
        )
        raise FormalPhysicalDispatchError(
            "distributed_formal_execution_failed",
            fatal,
        ) from failure
    assert before_snapshot is not None
    assert ready_snapshot is not None
    assert after_snapshot is not None
    assert process is not None
    assert process_exit_code is not None
    assert cleanup_kind is not None
    assert process_exited_ns is not None
    terminal_binding = CanonicalJsonProofBinding.bind(plan.terminal_output_path)
    pointer_binding = CanonicalJsonProofBinding.bind(
        plan.native_itl_pointer_output_path
    )
    assert plan.formal_gang_terminal_output_path is not None
    gang_binding = CanonicalJsonProofBinding.bind(plan.formal_gang_terminal_output_path)
    phase_edges_ns["evidence_flush_started_ns"] = time.monotonic_ns()
    phase_edges_ns["evidence_flush_finished_ns"] = time.monotonic_ns()
    lifecycle_value = {
        "schema_version": 1,
        "kind": "unsigned_formal_gang_lifecycle_timing",
        "protocol_sha256": FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        "formal_execution_authorized": False,
        "plan_sha256": plan.sha256,
        "formal_launch_admission": launch_admission_value,
        "formal_launch_consumption": launch_consumption_value,
        "budget_consumption": budget_consumption_value,
        "topology_mode": plan.topology_mode,
        "terminal_sha256": terminal_binding.semantic_sha256,
        "native_itl_pointer_sha256": pointer_binding.semantic_sha256,
        "formal_gang_terminal_sha256": gang_binding.semantic_sha256,
        "phase_edges_ns": phase_edges_ns,
    }
    if execution_policy is not None:
        assert client_lifecycle_binding is not None
        lifecycle_value["serving_execution_policy_sha256"] = execution_policy.sha256
        lifecycle_value["client_request_lifecycle_sha256"] = (
            client_lifecycle_binding.semantic_sha256
        )
    publish_canonical_json_no_replace(
        plan.lifecycle_timing_output_path,
        lifecycle_value,
    )
    lifecycle_binding = CanonicalJsonProofBinding.bind(
        plan.lifecycle_timing_output_path
    )
    junit_binding = _publish_formal_serving_junit(
        output_path=plan.junit_output_path,
        topology_mode=plan.topology_mode,
        request_ids=tuple(row.request_id for row in scored),
    )
    log_binding = EvidenceFileBinding.bind(
        Path(plan.server_log_output_path),
        label="formal gang server log",
    )
    stdout_binding = EvidenceFileBinding.bind(
        Path(plan.server_stdout_output_path),
        label="formal gang server stdout",
    )
    stderr_binding = EvidenceFileBinding.bind(
        Path(plan.server_stderr_output_path),
        label="formal gang server stderr",
    )
    receipt_value = {
        "schema_version": 2 if execution_policy is not None else 1,
        "kind": "unsigned_formal_gang_physical_run_receipt",
        "protocol_sha256": FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        "formal_execution_authorized": False,
        "plan_sha256": plan.sha256,
        "execution_binding_sha256": plan.execution_binding_sha256,
        "formal_launch_admission": launch_admission_value,
        "formal_launch_consumption": launch_consumption_value,
        "budget_consumption": budget_consumption_value,
        "launch_manifest": plan.launch_manifest.to_dict(),
        "request_schedule_receipt": plan.request_schedule_receipt.to_dict(),
        "terminal": terminal_binding.to_dict(),
        "native_itl_pointers": pointer_binding.to_dict(),
        "formal_gang_terminal": gang_binding.to_dict(),
        "lifecycle_timing": lifecycle_binding.to_dict(),
        "before_gpu_snapshot": before_snapshot.to_dict(),
        "ready_gpu_snapshot": ready_snapshot.to_dict(),
        "after_gpu_snapshot": after_snapshot.to_dict(),
        "server_log": log_binding.to_dict(),
        "server_stdout": stdout_binding.to_dict(),
        "server_stderr": stderr_binding.to_dict(),
        "junit": junit_binding.to_dict(),
        "server_process_id": process.pid,
        "process_exit_code": process_exit_code,
        "cleanup_kind": cleanup_kind,
        "process_group_empty": not _process_group_exists_for_formal_dispatch(
            process.pid
        ),
        "phase_edges_ns": phase_edges_ns,
    }
    if execution_policy is not None:
        assert client_lifecycle_binding is not None
        receipt_value["serving_execution_policy"] = execution_policy.to_dict()
        receipt_value["client_request_lifecycle"] = client_lifecycle_binding.to_dict()
    if receipt_value["process_group_empty"] is not True:
        raise RuntimeError("distributed formal process group survived cleanup")
    publish_canonical_json_no_replace(plan.live_run_receipt_output_path, receipt_value)
    receipt_binding = CanonicalJsonProofBinding.bind(plan.live_run_receipt_output_path)
    return ValidatedUnsignedFormalGangServingRun(
        receipt=receipt_binding,
        request_terminal=terminal_binding,
        native_itl_pointers=pointer_binding,
        formal_gang_terminal=gang_binding,
        lifecycle_timing=lifecycle_binding,
        client_request_lifecycle=client_lifecycle_binding,
    )


__all__ = [
    "FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256",
    "FORMAL_SERVING_TERMINAL_PUBLICATION_GRACE_NS",
    "FORMAL_SINGLE_OPERATOR_EXECUTION_REBUILD_SOURCE_PROTOCOL_SHA256",
    "FORMAL_TRUSTED_CONTROLLED_CONTEXT_COMPILED_ROW_SHARD_ARTIFACT_KIND",
    "FORMAL_TRUSTED_CONTROLLED_CONTEXT_RECEIPT_ROW_SHARD_ARTIFACT_KIND",
    "FORMAL_TRUSTED_CONTROLLED_CONTEXT_SOURCE_ROW_SHARD_ARTIFACT_KIND",
    "FORMAL_TRUSTED_SCHEDULE_RECEIPT_ROW_SHARD_ARTIFACT_KIND",
    "FORMAL_TRUSTED_SCHEDULE_SOURCE_ROW_SHARD_ARTIFACT_KIND",
    "TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_CONTROLLED_CONTEXT_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_SHARDED_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_SHARDED_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256",
    "FormalPhysicalDispatchError",
    "FormalServingProcessRuntimeContract",
    "FormalServingRequestScheduleReceipt",
    "FormalServingRequestScheduleRow",
    "FormalServingRequestScheduleSource",
    "FormalServingRequestScheduleSourceRow",
    "FormalServingRunPlan",
    "FormalSingleOperatorExecutionRebuildSource",
    "ValidatedUnsignedFormalGangServingRun",
    "execute_formal_distributed_serving_run_plan",
    "execute_formal_serving_run_plan",
    "execute_formal_single_operator_serving_run_plan",
    "execute_formal_tp1_serving_run_plan",
    "formal_serving_controlled_context_requests",
    "formal_serving_process_runtime_contract",
    "formal_serving_request_schedule_rows",
    "formal_serving_request_schedule_source_rows",
    "load_formal_serving_run_plan",
    "materialize_formal_serving_request_schedule",
    "materialize_formal_serving_run_plan",
    "materialize_formal_single_operator_downstream_serving_run_plan",
    "materialize_formal_single_operator_e5_failure_run_plan",
    "materialize_formal_single_operator_prepared_downstream_serving_run_plan",
    "materialize_formal_single_operator_profiler_subject_run_plan",
    "materialize_formal_single_operator_serving_run_plan",
    "materialize_trusted_single_operator_request_schedule",
    "publish_trusted_controlled_context_schedule_receipt_shards",
    "publish_trusted_controlled_context_schedule_source_shards",
    "publish_trusted_schedule_receipt_shards",
    "publish_trusted_schedule_source_shards",
    "rebuild_formal_single_operator_execution_binding_from_plan",
    "revalidate_formal_serving_run_plan",
    "revalidate_formal_single_operator_execution_rebuild_source",
    "revalidate_formal_single_operator_profiler_subject_run_plan",
    "trusted_controlled_context_compiled_rows_artifact_id",
    "trusted_controlled_context_receipt_rows_artifact_id",
    "trusted_controlled_context_source_rows_artifact_id",
    "trusted_schedule_receipt_rows_artifact_id",
    "trusted_schedule_source_rows_artifact_id",
]
