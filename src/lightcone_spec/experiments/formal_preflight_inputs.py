"""Deterministic current-host inputs for the exact ten-cell preflight.

The primary path is the trusted bridge from one current preflight execution
source to the three first-party runners.  A sealed-dispatch compatibility
entry remains available, but it is not a prerequisite for
``formal_single_operator_v1``.  The operator chooses only immutable source
paths and a new private output directory.  Cell IDs, GPU assignments, ports,
run configs, server argv, request rows, token IDs, qualification rows,
attempts, and result paths are derived here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self
from xml.etree import ElementTree

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config import load_run_config, run_config_sha256
from lightcone_spec.config.schema import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.doctor import _nvcc_release
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_dispatch import (
    FormalPreflightDispatchReceipt,
    FormalPreflightExecutionBinding,
    VerifiedFormalPreflightDispatch,
)
from lightcone_spec.experiments.formal_preflight_execution import (
    FormalPreflightInterferenceExecutionManifest,
    FormalPreflightInterferenceRunInput,
    _execute_formal_preflight_interference_raw_core,
    _InterferenceExecutionAdmission,
    require_formal_preflight_compile_assignment,
    require_formal_preflight_exactness_assignment,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_from_dict,
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    build_source_formal_runtime_authority_manifest,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorExecutionSource,
    load_formal_single_operator_execution_source,
)
from lightcone_spec.experiments.formal_slo_metrics import formal_prompt_bucket
from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuInventory,
    registry_pool_work_item,
)
from lightcone_spec.experiments.preflight_interference import (
    FormalPreflightInterferenceQualificationRow,
    FormalPreflightInterferenceRawBatch,
)
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.workload_authority import (
    FormalWorkloadAuthority,
    formal_workload_authority_artifact_id,
    formal_workload_authority_from_cli_artifact,
    revalidate_authorized_formal_workload_authority,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.locking.prepared_models import (
    PREPARED_MODEL_CONTENT_PROTOCOL_SHA256,
    PreparedModelSnapshotContent,
)
from lightcone_spec.orchestration.formal_physical_dispatch import (
    _invoke_tokenizer_worker,
    _materialized_schedule_rows,
    _publish_tokenization_input,
    _root_verified_workload_source,
    rebuild_formal_serving_request_schedule_source,
    rebuild_trusted_single_operator_request_schedule_source,
)
from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool
from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding
from lightcone_spec.orchestration.runtime import _render_server
from lightcone_spec.runtime.compile_cache import (
    COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
    COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256,
    COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    CompileCacheKey,
    CompileCacheLaunchPlan,
    CompileOnlyAssignmentContract,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
)
from lightcone_spec.runtime.compile_runner import (
    COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileAssignmentPlan,
    CompileLaunchManifest,
    CompileResultPointer,
    CompileSubprocessLifecycleReceipt,
    CompileWorkerSourceDescriptor,
    _execute_compile_assignment_subprocess_path,
    write_compile_prewarm_manifest,
)
from lightcone_spec.runtime.content_authorization import (
    AuthorizedPreparedModel,
    ContentJsonArtifactBinding,
    ContentVerificationReceipt,
    VerifiedPreparedModelContentRelease,
    VerifiedReleaseWorkloadSources,
)
from lightcone_spec.runtime.distributed import (
    DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
)
from lightcone_spec.runtime.preflight_runner import (
    PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
    EvidenceFileBinding,
    ExactnessLoaderEnvironment,
    ExactnessPreflightAssignment,
    ExactnessPreflightResultPointer,
    ExactnessPreflightTerminal,
    PreflightInputLocks,
    _execute_exactness_preflight,
    derive_burstgpt_shape_authority_from_content_receipt,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_PREFLIGHT_INPUTS_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_exact_ten_preflight_inputs",
        "authority": "durable_dispatch_plus_root_verified_content_plus_PASS_doctor",
        "derivation": (
            "one_compile_one_exactness_eight_interference;source_owned_configs_"
            "argv_ports_requests_tokens_qualification_and_attempts"
        ),
        "caller_values": "paths_and_new_private_output_root_only",
        "publication": "atomic_no_replace_then_deep_reopen",
    }
)

FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_exact_ten_preflight_inputs",
        "mode": "formal_single_operator_v1",
        "source": "current_preflight_execution_source",
        "identity": (
            "clean_head_tree_patch_registry_protocol_lock_runtime_manifest_"
            "inventory_content_workload_doctor"
        ),
        "derivation": (
            "one_compile_one_exactness_eight_interference;code_owned_gpu_slot_"
            "port_budget_timeout_config_argv_request_token_and_output_policy"
        ),
        "control": "trusted_single_operator_without_signature_or_replay_gate",
        "publication": "external_private_root_atomic_no_replace",
    }
)

TRUSTED_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_single_operator_exact_ten_preflight_inputs",
        "mode": "trusted_single_operator",
        "content": (
            "runtime_BOUND_tagged_bundle_exact_preflight_model_members_LCB_"
            "and_BurstGPT_without_signed_authorization_claims"
        ),
        "compile": "schema2_trusted_compile_launch",
        "exactness": "tagged_preflight_input_locks",
        "interference": "schema5_trusted_request_sources",
        "legacy_schema2": "unchanged",
    }
)

TRUSTED_SINGLE_OPERATOR_QUALIFIED_PREFLIGHT_INPUTS_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "formal_single_operator_exact_ten_preflight_inputs",
        "mode": "trusted_single_operator",
        "base": TRUSTED_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
        "exactness_cell_physical_subtests": (
            "chronobelief_gpu_parity_dspark_tp1_tp2_dp2_tp2_dp1_tp1_dp2"
        ),
        "logical_cell_count": "unchanged_exact_1_plus_1_plus_8",
        "authority": "source_owned_exact_six_plan_index_no_signature",
    }
)

FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_exact_ten_preflight_completion",
        "input": "deep_revalidated_formal_single_operator_exact_ten_inputs",
        "actuals": (
            "compile_result_and_subprocess_lifecycle;exactness_result_terminal_"
            "junit_rank_terminals;eight_current_terminal_lifecycle_junit_DAGs"
        ),
        "status_and_timing": "derived_only_from_first_party_actual_evidence",
        "coverage": "one_compile_one_exactness_eight_interference",
        "publication": "canonical_atomic_no_replace_then_deep_reopen",
    }
)

TRUSTED_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_single_operator_exact_ten_preflight_completion",
        "base": FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256,
        "exactness_physical_subtests": (
            "exact_six_trusted_native_qualification_results_bound_into_one_"
            "exactness_row_digest_and_timing_without_new_logical_cells"
        ),
        "claim": "trusted_single_operator_no_signature_not_formal_MEASURED",
    }
)

FORMAL_SINGLE_OPERATOR_PREFLIGHT_EXECUTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_exact_ten_preflight_execution",
        "authority": "deep_revalidated_current_source_exact_ten_inputs",
        "execution": (
            "source_owned_compile_then_exactness_then_four_isolated_and_two_"
            "paired_interference_waves"
        ),
        "completion": "actual_evidence_only_or_explicit_error",
        "caller_values": "execution_inputs_path_and_current_time_only",
        "publication": "canonical_atomic_no_replace_then_deep_reopen",
    }
)

_TARGET_MODEL_ID = "Qwen/Qwen3-8B"
_DRAFTER_MODEL_ID = "z-lab/Qwen3-8B-DFlash-b16"
_FIXED_MEM_FRACTION_STATIC = 0.75
_FIXED_ATTEMPT_ID = "attempt-0"
_SINGLE_OPERATOR_PROCESS_TIMEOUT_NS = {
    "first_party_compile": 60 * 60 * 1_000_000_000,
    "first_party_exactness": 60 * 60 * 1_000_000_000,
    "first_party_interference": 30 * 60 * 1_000_000_000,
}


def _strict(label: str, value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _private_new_root(path: str | Path) -> Path:
    root = Path(path)
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or root.exists()
        or not root.parent.is_dir()
        or root.parent.is_symlink()
    ):
        raise ValueError(
            "formal preflight input root must be one new resolved directory"
        )
    root.mkdir(mode=0o700)
    status = root.stat(follow_symlinks=False)
    if status.st_uid != os.geteuid() or stat.S_IMODE(status.st_mode) & 0o077:
        raise ValueError("formal preflight input root is not private")
    return root


def _raw_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError("formal preflight source file is unavailable")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _publish_text_no_replace(path: Path, value: str) -> None:
    body = value.encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json_with_sidecar(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    binding = CanonicalJsonProofBinding.bind(path)
    _publish_text_no_replace(Path(f"{path}.sha256"), f"{binding.semantic_sha256}\n")
    return binding


def _publish_binding(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _dispatch_bindings(
    token: VerifiedFormalPreflightDispatch,
) -> tuple[FormalPreflightExecutionBinding, ...]:
    rows = token.subject.execution_bindings
    if (
        len(rows) != 10
        or rows != tuple(sorted(rows, key=lambda row: row.registry_cell_id))
        or len({row.registry_cell_id for row in rows}) != 10
    ):
        raise ValueError("formal preflight input dispatch is not exact ten")
    return rows


def _one_binding(
    token: VerifiedFormalPreflightDispatch, runner: str
) -> FormalPreflightExecutionBinding:
    rows = tuple(row for row in _dispatch_bindings(token) if row.runner_kind == runner)
    if len(rows) != 1:
        raise ValueError(f"formal preflight requires one {runner} row")
    return rows[0]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightAuthority:
    """Reopenable trusted projection of the current exact-ten materialization."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_preflight_authority"]
    protocol_sha256: str
    execution_source: CanonicalJsonProofBinding
    protocol_lock: CanonicalJsonProofBinding
    materialization: CanonicalJsonProofBinding
    runtime_authority_manifest: CanonicalJsonProofBinding
    inventory: CanonicalJsonProofBinding
    repository_root: str
    repository_head: str
    repository_tree: str
    patch_manifest_sha256: str
    registry_sha256: str
    budget_plan_sha256: str
    split_sha256: str
    execution_bindings: tuple[FormalPreflightExecutionBinding, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_preflight_authority"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator preflight authority schema differs")
        for value in (
            self.execution_source,
            self.protocol_lock,
            self.materialization,
            self.runtime_authority_manifest,
            self.inventory,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("single-operator preflight source is not path-bound")
        repository_root = Path(self.repository_root)
        if (
            not repository_root.is_absolute()
            or repository_root != repository_root.resolve(strict=False)
        ):
            raise ValueError("single-operator preflight repository root is invalid")
        for label, digest in (
            ("repository head", self.repository_head),
            ("repository tree", self.repository_tree),
        ):
            if (
                type(digest) is not str
                or len(digest) != 40
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"single-operator preflight {label} is invalid")
        for label, digest in (
            ("patch manifest", self.patch_manifest_sha256),
            ("registry", self.registry_sha256),
            ("budget plan", self.budget_plan_sha256),
            ("split", self.split_sha256),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"single-operator preflight {label} is invalid")
        if (
            len(self.execution_bindings) != 10
            or self.execution_bindings
            != tuple(
                sorted(self.execution_bindings, key=lambda row: row.registry_cell_id)
            )
            or len({row.registry_cell_id for row in self.execution_bindings}) != 10
        ):
            raise ValueError("single-operator preflight bindings are not exact ten")

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "execution_source": self.execution_source.to_dict(),
            "protocol_lock": self.protocol_lock.to_dict(),
            "materialization": self.materialization.to_dict(),
            "runtime_authority_manifest": self.runtime_authority_manifest.to_dict(),
            "inventory": self.inventory.to_dict(),
            "repository_root": self.repository_root,
            "repository_head": self.repository_head,
            "repository_tree": self.repository_tree,
            "patch_manifest_sha256": self.patch_manifest_sha256,
            "registry_sha256": self.registry_sha256,
            "budget_plan_sha256": self.budget_plan_sha256,
            "split_sha256": self.split_sha256,
            "execution_bindings": [row.to_dict() for row in self.execution_bindings],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator preflight authority",
            value,
            set(cls.__dataclass_fields__),
        )
        raw_bindings = row.pop("execution_bindings")
        if type(raw_bindings) is not list:
            raise TypeError("single-operator preflight bindings must be an array")
        for name in (
            "execution_source",
            "protocol_lock",
            "materialization",
            "runtime_authority_manifest",
            "inventory",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(
            **row,
            execution_bindings=tuple(
                FormalPreflightExecutionBinding.from_dict(item) for item in raw_bindings
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class _TrustedDigest:
    sha256: str


@dataclass(frozen=True)
class _TrustedActivation:
    runtime_sha256: str
    split_sha256: str

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_single_operator_preflight_activation",
                "runtime_sha256": self.runtime_sha256,
                "split_sha256": self.split_sha256,
            }
        )


@dataclass(frozen=True)
class _TrustedDispatchContext:
    inventory: GpuInventory
    registry: Any
    activation_artifact: _TrustedActivation


@dataclass(frozen=True)
class _TrustedSubject:
    inventory_sha256: str
    budget_plan_sha256: str
    execution_bindings: tuple[FormalPreflightExecutionBinding, ...]


@dataclass(frozen=True)
class _TrustedManifest:
    registry_sha256: str


@dataclass(frozen=True)
class _TrustedPreflightToken:
    sha256: str
    subject: _TrustedSubject
    manifest: _TrustedManifest
    dispatch_context: _TrustedDispatchContext
    dispatch_plan: _TrustedDigest


def _git_value(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _trusted_repository_identity(
    repository_root: str | Path,
    *,
    protocol_lock: ProtocolLock,
) -> tuple[Path, str, str]:
    root = Path(repository_root)
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or not (root / ".git").exists()
    ):
        raise ValueError("single-operator repository root is not one Git checkout")
    if _git_value(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError(
            "single-operator exact-ten builder requires a clean checkout"
        )
    head = _git_value(root, "rev-parse", "HEAD")
    tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    registry = build_industrial_registry()
    runtime = build_source_formal_runtime_authority_manifest(root)
    if (
        (head, tree) != (protocol_lock.code_git_head, protocol_lock.code_git_tree)
        or protocol_lock.patch_manifest_sha256 != PINNED_SGLANG_PATCH_MANIFEST_SHA256
        or protocol_lock.registry_sha256 != registry.sha256
        or protocol_lock.formal_runtime_authority_manifest_sha256 != runtime.sha256
    ):
        raise ValueError(
            "single-operator checkout/patch/registry/runtime differs from ProtocolLock"
        )
    return root, head, tree


def _runner_for_preflight_cell(cell: ExperimentCell) -> str:
    return {
        "environment_and_patch_preflight": "first_party_compile",
        "exactness_memory_telemetry_preflight": "first_party_exactness",
        "simultaneous_single_gpu_interference": "first_party_interference",
    }.get(cell.identity.task) or "unsupported"


def _trusted_preflight_bindings(
    *,
    source: FormalSingleOperatorExecutionSource,
    protocol_lock: ProtocolLock,
    inventory: GpuInventory,
) -> tuple[FormalPreflightExecutionBinding, ...]:
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="single-operator preflight materialization"
        )
    )
    registry = build_industrial_registry()
    if (
        source.node != "preflight"
        or materialization.stage != "preflight"
        or materialization.expected_cell_count != 10
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.source_decision_sha256 != registry.sha256
        or len(inventory.devices) != 2
        or any(not device.ready for device in inventory.devices)
    ):
        raise ValueError("single-operator preflight source is not exact/current/ready")
    registered = {row.cell_id: row for row in registry.cells_for("preflight")}
    linked = {
        dict(row.dimensions).get("registry_cell_id"): row.cell_id
        for row in materialization.cells
    }
    if set(linked) != set(registered) or None in linked:
        raise ValueError("single-operator preflight materialization link differs")
    physical = tuple(device.uuid for device in inventory.devices)
    rows = []
    for registry_cell_id, cell in sorted(registered.items()):
        runner = _runner_for_preflight_cell(cell)
        if runner == "unsupported":
            raise ValueError("single-operator preflight runner is unsupported")
        timeout_ns = _SINGLE_OPERATOR_PROCESS_TIMEOUT_NS[runner]
        work_item = registry_pool_work_item(
            cell,
            estimated_duration_seconds=timeout_ns / 1_000_000_000,
        )
        if runner in {"first_party_compile", "first_party_exactness"}:
            gpu_uuids = physical
        else:
            variant = cell.identity.variant
            if "_slot_" not in variant:
                raise ValueError("single-operator interference slot is missing")
            slot = int(variant.rsplit("_slot_", maxsplit=1)[1])
            if slot not in {0, 1}:
                raise ValueError("single-operator interference slot is invalid")
            gpu_uuids = (physical[slot],)
        shape = work_item.claim.gang_shape
        rank_groups = tuple(
            tuple(
                gpu_uuids[
                    index * shape.tensor_parallel_size : (index + 1)
                    * shape.tensor_parallel_size
                ]
            )
            for index in range(shape.data_parallel_size)
        )
        assignment = GpuAssignment(
            work_item=work_item,
            gpu_uuids=gpu_uuids,
            rank_groups=rank_groups,
            ports=cell.resources.ports,
        )
        budget_sha256 = content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_single_operator_preflight_cell_budget",
                "execution_source_sha256": source.sha256,
                "materialized_cell_id": linked[registry_cell_id],
                "registry_cell_id": registry_cell_id,
                "runner_kind": runner,
                "process_hard_timeout_ns": timeout_ns,
                "gpu_count": len(gpu_uuids),
                "attempts": 1,
            }
        )
        rows.append(
            FormalPreflightExecutionBinding(
                materialized_cell_id=linked[registry_cell_id],
                registry_cell_id=registry_cell_id,
                runner_kind=runner,  # type: ignore[arg-type]
                work_item_sha256=work_item.sha256,
                assignment_sha256=assignment.sha256,
                experiment_budget_sha256=budget_sha256,
                source_authority_bindings=(
                    protocol_lock.preflight_source_authority_bindings
                ),
                cell=cell,
                assignment=assignment,
                gpu_uuids=gpu_uuids,
                rank_groups=rank_groups,
            )
        )
    result = tuple(sorted(rows, key=lambda row: row.registry_cell_id))
    counts = {
        runner: sum(row.runner_kind == runner for row in result)
        for runner in _SINGLE_OPERATOR_PROCESS_TIMEOUT_NS
    }
    if counts != {
        "first_party_compile": 1,
        "first_party_exactness": 1,
        "first_party_interference": 8,
    }:
        raise ValueError("single-operator preflight runner coverage differs")
    return result


def _trusted_budget_plan_sha256(
    source: FormalSingleOperatorExecutionSource,
    inventory: GpuInventory,
    bindings: tuple[FormalPreflightExecutionBinding, ...],
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_preflight_budget_plan",
            "execution_source_sha256": source.sha256,
            "inventory_sha256": inventory.sha256,
            "cell_budgets": [
                [row.registry_cell_id, row.experiment_budget_sha256] for row in bindings
            ],
        }
    )


def _trusted_split_sha256(
    materialization: Any,
    bindings: tuple[FormalPreflightExecutionBinding, ...],
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_preflight_split",
            "materialization_sha256": materialization.sha256,
            "registry_cell_ids": [row.registry_cell_id for row in bindings],
        }
    )


def _trusted_token(
    *,
    authority: FormalSingleOperatorPreflightAuthority,
    source: FormalSingleOperatorExecutionSource,
    inventory: GpuInventory,
) -> _TrustedPreflightToken:
    registry = build_industrial_registry()
    return _TrustedPreflightToken(
        sha256=authority.sha256,
        subject=_TrustedSubject(
            inventory_sha256=inventory.sha256,
            budget_plan_sha256=authority.budget_plan_sha256,
            execution_bindings=authority.execution_bindings,
        ),
        manifest=_TrustedManifest(registry_sha256=registry.sha256),
        dispatch_context=_TrustedDispatchContext(
            inventory=inventory,
            registry=registry,
            activation_artifact=_TrustedActivation(
                runtime_sha256=source.runtime_authority_manifest_sha256,
                split_sha256=authority.split_sha256,
            ),
        ),
        dispatch_plan=_TrustedDigest(
            sha256=content_sha256(
                {
                    "schema_version": 1,
                    "kind": "formal_single_operator_preflight_execution_plan",
                    "authority_sha256": authority.sha256,
                    "assignments": [
                        row.assignment_sha256 for row in authority.execution_bindings
                    ],
                }
            )
        ),
    )


def _doctor_report(
    path: str | Path,
    *,
    token: VerifiedFormalPreflightDispatch,
) -> tuple[CanonicalJsonProofBinding, dict[str, Any], Path, Path, Path]:
    binding = CanonicalJsonProofBinding.bind(path)
    value = binding.reopen()
    if type(value) is not dict:
        raise TypeError("formal preflight doctor report must be an object")
    readiness = value.get("readiness")
    checks = value.get("checks")
    roots = value.get("roots")
    python = value.get("python")
    gpu = value.get("gpu")
    commands = value.get("commands")
    packages = value.get("packages")
    if (
        value.get("schema_version") != 2
        or value.get("status") != "PASS"
        or type(readiness) is not dict
        or readiness.get("status") != "PASS"
        or readiness.get("fail_count") != 0
        or readiness.get("unknown_count") != 0
        or type(checks) is not dict
        or not checks
        or any(
            type(row) is not dict or row.get("status") != "PASS"
            for row in checks.values()
        )
        or type(roots) is not dict
        or type(python) is not dict
        or type(gpu) is not dict
        or type(commands) is not dict
        or type(packages) is not dict
    ):
        raise ValueError("formal preflight requires one complete PASS doctor report")
    checkout = Path(str(roots.get("patched_sglang")))
    executable = Path(str(python.get("executable")))
    if (
        not checkout.is_absolute()
        or checkout != checkout.resolve(strict=False)
        or not checkout.is_dir()
        or checkout.is_symlink()
        or executable.resolve(strict=False) != Path(sys.executable).resolve()
        or not executable.is_file()
    ):
        raise ValueError("formal preflight doctor checkout/interpreter differs")
    parsed = gpu.get("parsed_inventory")
    devices = None if type(parsed) is not dict else parsed.get("devices")
    if type(devices) is not list or len(devices) != len(
        token.dispatch_context.inventory.devices
    ):
        raise ValueError("formal preflight doctor GPU coverage differs")
    by_uuid = {row.get("uuid"): row for row in devices if type(row) is dict}
    for device in token.dispatch_context.inventory.devices:
        row = by_uuid.get(device.uuid)
        if (
            type(row) is not dict
            or row.get("name") != device.model
            or row.get("compute_capability")
            != f"{device.compute_capability[0]}.{device.compute_capability[1]}"
        ):
            raise ValueError("formal preflight doctor GPU identity differs")
    nvidia_smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")
    if nvidia_smi is None or nvcc is None:
        raise ValueError("formal preflight doctor tools are unavailable")
    nvidia_path = Path(nvidia_smi).resolve()
    nvcc_path = Path(nvcc).resolve()
    cuda_home = nvcc_path.parent.parent
    if (
        not nvidia_path.is_file()
        or nvidia_path.is_symlink()
        or not nvcc_path.is_file()
        or nvcc_path.is_symlink()
        or not cuda_home.is_dir()
    ):
        raise ValueError("formal preflight doctor tool paths are invalid")
    return binding, value, checkout, nvidia_path, cuda_home


def _content_sources(
    *,
    content_receipt_path: str | Path,
    workload_authority_path: str | Path,
    current_ns: int,
) -> tuple[
    CanonicalJsonProofBinding,
    ContentVerificationReceipt,
    VerifiedPreparedModelContentRelease,
    VerifiedReleaseWorkloadSources,
    ContentJsonArtifactBinding,
    object,
    str,
    dict[str, tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent]],
]:
    content_binding = CanonicalJsonProofBinding.bind(content_receipt_path)
    receipt = ContentVerificationReceipt.from_dict(content_binding.reopen())
    if receipt.sha256 != content_binding.semantic_sha256:
        raise ValueError("formal preflight content receipt identity differs")
    verified = receipt.revalidate_formal_scope(current_ns=current_ns)
    prepared_rows = tuple(
        row for row in verified if type(row) is VerifiedPreparedModelContentRelease
    )
    workload_rows = tuple(
        row for row in verified if type(row) is VerifiedReleaseWorkloadSources
    )
    if len(prepared_rows) != 1 or len(workload_rows) != 1:
        raise ValueError("formal preflight content authority coverage differs")
    workload_binding, workload, descriptor_sha256 = _root_verified_workload_source(
        receipt,
        workload_id="livecodebench_v6_hard",
        workload_authority_path=workload_authority_path,
        current_ns=current_ns,
    )
    # Re-run the public reducer explicitly; the private schedule helper is not
    # allowed to turn a caller-authored workload JSON into authority.
    authorized_workload = formal_workload_authority_from_cli_artifact(
        workload_binding.load()
    )
    workload = revalidate_authorized_formal_workload_authority(
        authorized_workload,
        authorization=workload_rows[0],
    )
    members = {
        row.member_id: row for row in prepared_rows[0].require_stage("preflight")
    }
    snapshots: dict[
        str, tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent]
    ] = {}
    artifacts = {row.artifact_id: row for row in receipt.content_artifacts}
    for member_id, member in members.items():
        artifact = artifacts.get(f"snapshot:{member_id}")
        if artifact is None:
            raise ValueError("formal preflight prepared snapshot is not path-bound")
        snapshot = PreparedModelSnapshotContent.from_dict(artifact.load())
        if snapshot.model_id != member.model_id or snapshot.revision != member.revision:
            raise ValueError("formal preflight prepared snapshot identity differs")
        snapshots[member_id] = (member, snapshot)
    return (
        content_binding,
        receipt,
        prepared_rows[0],
        workload_rows[0],
        workload_binding,
        workload,
        descriptor_sha256,
        snapshots,
    )


def _trusted_content_sources(
    *,
    content_source_binding: FormalContentSourceBinding,
    workload_authority_path: str | Path,
) -> tuple[
    object,
    ContentJsonArtifactBinding,
    FormalWorkloadAuthority,
    object,
    object,
    object,
    object,
]:
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
    )

    bundle = content_source_binding.reopen()
    if (
        type(bundle) is not TrustedSingleOperatorContentBundle
        or bundle.runtime_binding_status != "BOUND"
    ):
        raise ValueError("trusted preflight requires one BOUND content bundle")
    workload_binding = ContentJsonArtifactBinding.from_path(
        formal_workload_authority_artifact_id("livecodebench_v6_hard"),
        workload_authority_path,
    )
    workload = formal_workload_authority_from_cli_artifact(workload_binding.load())
    workload_members = tuple(
        row
        for row in bundle.locked_workloads
        if row.workload_id == "livecodebench_v6_hard"
        and row.authority_sha256 == workload.sha256
        and row.raw_source_path == workload.raw_source_path
        and row.raw_file_sha256 == workload.raw_file_sha256
        and row.repository_revision == workload.repository_revision
        and row.raw_row_count == workload.raw_row_count
        and row.selected_row_count == workload.selected_row_count
        and row.formal_samples_sha256 == workload.selected_rows_sha256
        and row.source_lock_sha256 == workload.source_lock_sha256
        and row.protocol_sha256 == workload.protocol_sha256
    )
    if len(workload_members) != 1:
        raise ValueError("trusted preflight LiveCodeBench member is not exact")

    def select(role: str, model_id: str) -> object:
        rows = tuple(
            row
            for row in bundle.model_members
            if row.role == role
            and row.model_id == model_id
            and "preflight" in row.stages
        )
        if len(rows) != 1:
            raise ValueError(f"trusted preflight lacks one exact {role} member")
        return rows[0]

    return (
        bundle,
        workload_binding,
        workload,
        workload_members[0],
        select("target", _TARGET_MODEL_ID),
        select("drafter", _DRAFTER_MODEL_ID),
        select("tokenizer", _TARGET_MODEL_ID),
    )


def _select_model_sources(
    snapshots: dict[str, tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent]],
) -> tuple[
    tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent],
    tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent],
    tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent],
]:
    def select(
        role: str, model_id: str
    ) -> tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent]:
        rows = tuple(
            value
            for value in snapshots.values()
            if getattr(value[0], "role", None) == role
            and getattr(value[0], "model_id", None) == model_id
        )
        if len(rows) != 1:
            raise ValueError(f"formal preflight lacks one exact {role} model")
        return rows[0]

    return (
        select("target", _TARGET_MODEL_ID),
        select("drafter", _DRAFTER_MODEL_ID),
        select("tokenizer", _TARGET_MODEL_ID),
    )


def _rebuild_prepared_content_manifest(
    *,
    root: Path,
    receipt: ContentVerificationReceipt,
    prepared: VerifiedPreparedModelContentRelease,
) -> CanonicalJsonProofBinding:
    """Rebuild the root-authorized manifest from its bound snapshot members."""

    by_identity: dict[
        tuple[str, str], tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent]
    ] = {}
    for member in prepared.authorization.models:
        matches = tuple(
            artifact
            for artifact in receipt.content_artifacts
            if artifact.raw_sha256 == member.snapshot_manifest_raw_sha256
            and artifact.semantic_sha256 == member.snapshot_manifest_semantic_sha256
        )
        identities = {
            (artifact.path, artifact.raw_sha256, artifact.semantic_sha256)
            for artifact in matches
        }
        if len(identities) != 1:
            raise ValueError("formal preflight snapshot member is not exact")
        snapshot = PreparedModelSnapshotContent.from_dict(matches[0].load())
        if (snapshot.model_id, snapshot.revision) != (
            member.model_id,
            member.revision,
        ):
            raise ValueError("formal preflight snapshot member identity differs")
        key = (snapshot.model_id, snapshot.revision)
        previous = by_identity.get(key)
        if previous is not None and previous[1] != snapshot:
            raise ValueError("formal preflight snapshot identity is ambiguous")
        by_identity[key] = (member, snapshot)
    snapshots = tuple(
        sorted((row[1] for row in by_identity.values()), key=lambda row: row.model_id)
    )
    if not snapshots or tuple(row.model_id for row in snapshots) != tuple(
        sorted({row.model_id for row in snapshots})
    ):
        raise ValueError("formal preflight prepared snapshots are not model-ID exact")
    value = {
        "schema_version": 1,
        "kind": "lightcone_prepared_model_content_manifest",
        "protocol_sha256": PREPARED_MODEL_CONTENT_PROTOCOL_SHA256,
        "model_lock_sha256": prepared.authorization.model_lock_sha256,
        "prepared_model_set_sha256": (prepared.authorization.prepared_model_set_sha256),
        "snapshots": [row.to_dict() for row in snapshots],
    }
    binding = _publish_json_with_sidecar(
        root / "prepared-model-content-manifest.json", value
    )
    if (
        binding.raw_sha256 != prepared.authorization.content_manifest_raw_sha256
        or binding.semantic_sha256
        != prepared.authorization.content_manifest_semantic_sha256
        or binding.size != prepared.authorization.content_manifest_size
    ):
        raise ValueError(
            "formal preflight rebuilt prepared manifest differs from root authority"
        )
    return binding


def _toolchain(
    doctor: dict[str, Any], *, gpu_uuid: str
) -> tuple[str, str, str, str, str, str, str]:
    python = doctor["python"]
    gpu = doctor["gpu"]
    torch = gpu["torch"]
    parsed = gpu["parsed_inventory"]
    matches = tuple(row for row in parsed["devices"] if row.get("uuid") == gpu_uuid)
    if len(matches) != 1:
        raise ValueError("formal preflight toolchain GPU is not exact")
    device = matches[0]
    cuda = _nvcc_release(doctor["commands"].get("nvcc"))
    if cuda is None or cuda != torch.get("cuda_build"):
        raise ValueError("formal preflight Torch/nvcc CUDA versions differ")
    capability = str(device["compute_capability"]).replace(".", "")
    values = (
        python.get("version"),
        torch.get("version"),
        doctor["packages"].get("triton"),
        cuda,
        device.get("driver_version"),
        f"sm_{capability}",
        device.get("name"),
    )
    if any(type(value) is not str or not value for value in values):
        raise ValueError("formal preflight toolchain is incomplete")
    return values  # type: ignore[return-value]


def _run_config(
    *,
    sampling_profile: SamplingProfile,
    target_revision: str,
    drafter_revision: str,
    gpu_uuids: tuple[str, ...],
    runtime_qualification_sha256: str,
) -> RunConfig:
    distributed: dict[str, object] = {}
    if len(gpu_uuids) == 2:
        release = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES["tp2_dp1"]
        distributed = {
            "distributed_runtime_capability": "patched_two_gpu_v1",
            "distributed_release_capability_sha256": release.sha256,
            "distributed_capability_receipt_sha256": runtime_qualification_sha256,
            "process_group_backend": release.process_group_backend,
        }
    return RunConfig(
        method="static",
        model=ModelPair(
            target=_TARGET_MODEL_ID,
            drafter=_DRAFTER_MODEL_ID,
            target_revision=target_revision,
            drafter_revision=drafter_revision,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=sampling_profile.sha256,
            tensor_parallel_size=len(gpu_uuids),
            max_running_requests=1,
            device_identity=",".join(gpu_uuids),
            **distributed,
        ),
    )


def _compile_key(
    *,
    doctor: dict[str, Any],
    config: RunConfig,
    gpu_uuid: str,
) -> CompileCacheKey:
    python, torch, triton, cuda, driver, architecture, model = _toolchain(
        doctor, gpu_uuid=gpu_uuid
    )
    key = CompileCacheKey(
        patched_sglang_tree=PINNED_SGLANG_TREE,
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        patch_sha256=PINNED_SGLANG_PATCH_SHA256,
        source_sha256=PINNED_SGLANG_COMPILE_SOURCE_SHA256,
        python_version=python,
        torch_version=torch,
        triton_version=triton,
        cuda_version=cuda,
        driver_version=driver,
        sm_architecture=architecture,
        gpu_model=model,
        dtype="bfloat16",
        target_revision=config.model.target_revision,
        drafter_revision=config.model.drafter_revision,
        tensor_parallel_size=config.runtime.tensor_parallel_size,
        context_limit=config.runtime.context_length,
        max_running_requests=config.runtime.max_running_requests,
        graph_buckets=(1,),
        allocator="cuda_malloc_async",
        build_flags=(),
    )
    key.validate()
    return key


def _launch_manifest(
    *,
    root: Path,
    binding: FormalPreflightExecutionBinding,
    doctor: dict[str, Any],
    checkout: Path,
    cuda_home: Path,
    prepared: VerifiedPreparedModelContentRelease,
    prepared_manifest: CanonicalJsonProofBinding,
    sampling_profile: SamplingProfile,
    sampling_path: Path,
    target_source: tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent],
    drafter_source: tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent],
    tokenizer_source: tuple[AuthorizedPreparedModel, PreparedModelSnapshotContent],
    runtime_qualification_sha256: str,
    inventory_sha256: str,
    budget_materialization_authority_sha256: str,
) -> tuple[CompileLaunchManifest, Path, CompileCacheLaunchPlan, Path, RunConfig]:
    target_member, target = target_source
    drafter_member, drafter = drafter_source
    tokenizer_member, tokenizer = tokenizer_source
    config = _run_config(
        sampling_profile=sampling_profile,
        target_revision=target.revision,
        drafter_revision=drafter.revision,
        gpu_uuids=binding.gpu_uuids,
        runtime_qualification_sha256=runtime_qualification_sha256,
    )
    row_root = root / f"row-{binding.registry_cell_id}"
    row_root.mkdir(mode=0o700)
    cache_root = (root / "compile-cache" / binding.registry_cell_id).resolve()
    key = _compile_key(
        doctor=doctor,
        config=config,
        gpu_uuid=binding.gpu_uuids[0],
    )
    cache_plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=cache_root,
        cache_mode="build",
    )
    cache_path = row_root / "compile-cache-plan.json"
    cache_plan.write(cache_path)
    model_lock_sha256 = prepared.authorization.model_lock_sha256
    rendered = _render_server(
        output=row_root,
        method="static",
        config=config,
        verified_checkout=checkout,
        roots={_TARGET_MODEL_ID: target.root, _DRAFTER_MODEL_ID: drafter.root},
        target_id=_TARGET_MODEL_ID,
        drafter_id=_DRAFTER_MODEL_ID,
        adaptation_reserve_mb=0,
        mem_fraction_static=_FIXED_MEM_FRACTION_STATIC,
        host="127.0.0.1",
        port=binding.assignment.ports[0],
        compile_cache_plan_path=cache_path,
    )
    config_path = Path(rendered.run_config).resolve()
    launch = CompileLaunchManifest(
        schema_version=1,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        run_config_path=str(config_path),
        run_config_raw_sha256=_raw_sha256(config_path),
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=_raw_sha256(cache_path),
        compile_cache_plan_sha256=cache_plan.sha256,
        prewarm_manifest_path=str(row_root / "prewarm.json"),
        prewarm_manifest_raw_sha256="0" * 64,
        prewarm_manifest_sha256="0" * 64,
        sampling_profile_path=str(sampling_path),
        sampling_profile_raw_sha256=_raw_sha256(sampling_path),
        prepared_model_content_manifest_path=prepared_manifest.absolute_path,
        prepared_model_content_manifest_raw_sha256=prepared_manifest.raw_sha256,
        prepared_model_content_manifest_sha256=prepared_manifest.semantic_sha256,
        prepared_model_content_manifest_size=prepared_manifest.size,
        target_content_member_id=target_member.member_id,
        target_model_id=target.model_id,
        target_snapshot_path=target.root,
        target_revision=target.revision,
        target_content_authority_sha256=prepared.authorization_sha256,
        drafter_content_member_id=drafter_member.member_id,
        drafter_model_id=drafter.model_id,
        drafter_snapshot_path=drafter.root,
        drafter_revision=drafter.revision,
        drafter_content_authority_sha256=prepared.authorization_sha256,
        tokenizer_content_member_id=tokenizer_member.member_id,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_snapshot_path=tokenizer.root,
        tokenizer_revision=tokenizer.revision,
        tokenizer_content_authority_sha256=prepared.authorization_sha256,
        server_argv=rendered.argv,
        server_argv_sha256=content_sha256({"argv": list(rendered.argv)}),
        localhost_port=binding.assignment.ports[0],
        model_lock_sha256=model_lock_sha256,
        sampling_profile_sha256=sampling_profile.sha256,
        physical_assignment_sha256=binding.assignment_sha256,
        experiment_budget_sha256=binding.experiment_budget_sha256,
        budget_materialization_authority_sha256=(
            budget_materialization_authority_sha256
        ),
        inventory_sha256=inventory_sha256,
        gpu_uuids=binding.gpu_uuids,
        path_entries=tuple(
            dict.fromkeys(
                (
                    str(Path(sys.executable).resolve().parent),
                    str((cuda_home / "bin").resolve()),
                )
            )
        ),
        library_path_entries=(str((cuda_home / "lib64").resolve()),),
        cuda_home=str(cuda_home),
    )
    # The prewarm manifest is row-specific only because its immutable paths are.
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=model_lock_sha256,
        sampling_profile_sha256=sampling_profile.sha256,
        payloads=(CompileOnlyPrewarmPayload("graph-bucket-1", 1, (1, 2), 1, 1),),
    )
    prewarm_path = write_compile_prewarm_manifest(prewarm, row_root / "prewarm.json")
    launch = CompileLaunchManifest(
        **{
            **launch.__dict__,
            "prewarm_manifest_raw_sha256": _raw_sha256(prewarm_path),
            "prewarm_manifest_sha256": prewarm.sha256,
        }
    )
    launch_path = row_root / "compile-launch.json"
    launch.write(launch_path)
    return launch, launch_path, cache_plan, prewarm_path, config


def _trusted_launch_manifest(
    *,
    root: Path,
    binding: FormalPreflightExecutionBinding,
    doctor: dict[str, Any],
    checkout: Path,
    cuda_home: Path,
    content_source_binding: FormalContentSourceBinding,
    sampling_profile: SamplingProfile,
    sampling_path: Path,
    target: object,
    drafter: object,
    tokenizer: object,
    runtime_qualification_sha256: str,
    inventory_sha256: str,
    budget_materialization_authority_sha256: str,
) -> tuple[CompileLaunchManifest, Path, CompileCacheLaunchPlan, Path, RunConfig]:
    trusted_binding = content_source_binding.trusted_single_operator
    if trusted_binding is None:
        raise ValueError("trusted preflight content binding is not exact")
    config = _run_config(
        sampling_profile=sampling_profile,
        target_revision=target.revision,
        drafter_revision=drafter.revision,
        gpu_uuids=binding.gpu_uuids,
        runtime_qualification_sha256=runtime_qualification_sha256,
    )
    row_root = root / f"row-{binding.registry_cell_id}"
    row_root.mkdir(mode=0o700)
    cache_root = (root / "compile-cache" / binding.registry_cell_id).resolve()
    key = _compile_key(
        doctor=doctor,
        config=config,
        gpu_uuid=binding.gpu_uuids[0],
    )
    cache_plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=cache_root,
        cache_mode="build",
    )
    cache_path = row_root / "compile-cache-plan.json"
    cache_plan.write(cache_path)
    model_lock = ModelLock(
        schema_version=2,
        models=tuple(
            sorted(
                (
                    LockedModel(target.model_id, target.revision),
                    LockedModel(drafter.model_id, drafter.revision),
                ),
                key=lambda row: row.model_id,
            )
        ),
    )
    model_lock.write(row_root / "model-lock.json")
    rendered = _render_server(
        output=row_root,
        method="static",
        config=config,
        verified_checkout=checkout,
        roots={
            target.model_id: target.local_snapshot_path,
            drafter.model_id: drafter.local_snapshot_path,
        },
        target_id=target.model_id,
        drafter_id=drafter.model_id,
        adaptation_reserve_mb=0,
        mem_fraction_static=_FIXED_MEM_FRACTION_STATIC,
        host="127.0.0.1",
        port=binding.assignment.ports[0],
        compile_cache_plan_path=cache_path,
    )
    config_path = Path(rendered.run_config).resolve()
    launch = CompileLaunchManifest(
        schema_version=2,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
        ),
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        run_config_path=str(config_path),
        run_config_raw_sha256=_raw_sha256(config_path),
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=_raw_sha256(cache_path),
        compile_cache_plan_sha256=cache_plan.sha256,
        prewarm_manifest_path=str(row_root / "prewarm.json"),
        prewarm_manifest_raw_sha256="0" * 64,
        prewarm_manifest_sha256="0" * 64,
        sampling_profile_path=str(sampling_path),
        sampling_profile_raw_sha256=_raw_sha256(sampling_path),
        prepared_model_content_manifest_path=trusted_binding.absolute_path,
        prepared_model_content_manifest_raw_sha256=trusted_binding.raw_sha256,
        prepared_model_content_manifest_sha256=trusted_binding.semantic_sha256,
        prepared_model_content_manifest_size=trusted_binding.size,
        target_content_member_id=target.sha256,
        target_model_id=target.model_id,
        target_snapshot_path=target.local_snapshot_path,
        target_revision=target.revision,
        target_content_authority_sha256=None,
        drafter_content_member_id=drafter.sha256,
        drafter_model_id=drafter.model_id,
        drafter_snapshot_path=drafter.local_snapshot_path,
        drafter_revision=drafter.revision,
        drafter_content_authority_sha256=None,
        tokenizer_content_member_id=tokenizer.sha256,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_snapshot_path=tokenizer.local_snapshot_path,
        tokenizer_revision=tokenizer.revision,
        tokenizer_content_authority_sha256=None,
        server_argv=rendered.argv,
        server_argv_sha256=content_sha256({"argv": list(rendered.argv)}),
        localhost_port=binding.assignment.ports[0],
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=sampling_profile.sha256,
        physical_assignment_sha256=binding.assignment_sha256,
        experiment_budget_sha256=binding.experiment_budget_sha256,
        budget_materialization_authority_sha256=(
            budget_materialization_authority_sha256
        ),
        inventory_sha256=inventory_sha256,
        gpu_uuids=binding.gpu_uuids,
        path_entries=tuple(
            dict.fromkeys(
                (
                    str(Path(sys.executable).resolve().parent),
                    str((cuda_home / "bin").resolve()),
                )
            )
        ),
        library_path_entries=(str((cuda_home / "lib64").resolve()),),
        cuda_home=str(cuda_home),
        formal_stage="preflight",
        content_source_binding=content_source_binding,
    )
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=sampling_profile.sha256,
        payloads=(CompileOnlyPrewarmPayload("graph-bucket-1", 1, (1, 2), 1, 1),),
    )
    prewarm_path = write_compile_prewarm_manifest(prewarm, row_root / "prewarm.json")
    launch = CompileLaunchManifest(
        **{
            **launch.__dict__,
            "prewarm_manifest_raw_sha256": _raw_sha256(prewarm_path),
            "prewarm_manifest_sha256": prewarm.sha256,
        }
    )
    launch_path = row_root / "compile-launch.json"
    launch.write(launch_path)
    return launch, launch_path, cache_plan, prewarm_path, config


@dataclass(frozen=True)
class FormalPreflightExecutionInputs:
    schema_version: Literal[2, 3, 4]
    kind: Literal["formal_single_operator_exact_ten_preflight_inputs"]
    protocol_sha256: str
    authority_mode: Literal["formal_dispatch", "formal_single_operator_v1"]
    execution_authority: CanonicalJsonProofBinding
    inventory: CanonicalJsonProofBinding
    content_receipt: CanonicalJsonProofBinding | None
    workload_authority: ContentJsonArtifactBinding
    doctor_report: CanonicalJsonProofBinding
    compile_assignment_plan: CanonicalJsonProofBinding
    exactness_assignment: CanonicalJsonProofBinding
    interference_manifest: CanonicalJsonProofBinding
    request_schedule_sources: tuple[CanonicalJsonProofBinding, ...]
    tokenization_inputs: tuple[CanonicalJsonProofBinding, ...]
    tokenization_outputs: tuple[CanonicalJsonProofBinding, ...]
    content_source_binding: FormalContentSourceBinding | None = None
    qualification_plan_index: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {2, 3, 4}
            or self.kind != "formal_single_operator_exact_ten_preflight_inputs"
            or self.authority_mode
            not in {"formal_dispatch", "formal_single_operator_v1"}
        ):
            raise ValueError("formal preflight execution inputs schema differs")
        expected_protocol = (
            TRUSTED_SINGLE_OPERATOR_QUALIFIED_PREFLIGHT_INPUTS_PROTOCOL_SHA256
            if self.schema_version == 4
            else (
                TRUSTED_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256
                if self.schema_version == 3
                else FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256
            )
        )
        if self.protocol_sha256 != expected_protocol:
            raise ValueError("formal preflight execution inputs protocol differs")
        if self.schema_version == 2:
            if (
                type(self.content_receipt) is not CanonicalJsonProofBinding
                or self.content_source_binding is not None
                or self.qualification_plan_index is not None
            ):
                raise ValueError("legacy preflight content lineage differs")
        elif self.schema_version == 3:
            if (
                self.content_receipt is not None
                or type(self.content_source_binding) is not FormalContentSourceBinding
                or self.content_source_binding.mode != "trusted_single_operator"
                or self.qualification_plan_index is not None
            ):
                raise ValueError("trusted preflight content lineage differs")
            self.content_source_binding.reopen()
        elif (
            self.content_receipt is not None
            or type(self.content_source_binding) is not FormalContentSourceBinding
            or self.content_source_binding.mode != "trusted_single_operator"
            or type(self.qualification_plan_index) is not CanonicalJsonProofBinding
        ):
            raise ValueError("trusted preflight content lineage differs")
        else:
            self.content_source_binding.reopen()
            from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
                load_formal_single_operator_preflight_qualification_plan_index,
            )

            load_formal_single_operator_preflight_qualification_plan_index(
                self.qualification_plan_index.absolute_path
            )
        for value in (
            self.execution_authority,
            self.inventory,
            self.doctor_report,
            self.compile_assignment_plan,
            self.exactness_assignment,
            self.interference_manifest,
            *self.request_schedule_sources,
            *self.tokenization_inputs,
            *self.tokenization_outputs,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("formal preflight execution input is not path-bound")
        if type(self.workload_authority) is not ContentJsonArtifactBinding:
            raise TypeError("formal preflight workload input is not path-bound")
        for rows in (
            self.request_schedule_sources,
            self.tokenization_inputs,
            self.tokenization_outputs,
        ):
            if len(rows) != 8 or len(set(rows)) != 8:
                raise ValueError(
                    "formal preflight request derivation is not exact eight"
                )

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "authority_mode": self.authority_mode,
            "execution_authority": self.execution_authority.to_dict(),
            "inventory": self.inventory.to_dict(),
            "content_receipt": (
                None if self.content_receipt is None else self.content_receipt.to_dict()
            ),
            "workload_authority": self.workload_authority.to_dict(),
            "doctor_report": self.doctor_report.to_dict(),
            "compile_assignment_plan": self.compile_assignment_plan.to_dict(),
            "exactness_assignment": self.exactness_assignment.to_dict(),
            "interference_manifest": self.interference_manifest.to_dict(),
            "request_schedule_sources": [
                row.to_dict() for row in self.request_schedule_sources
            ],
            "tokenization_inputs": [row.to_dict() for row in self.tokenization_inputs],
            "tokenization_outputs": [
                row.to_dict() for row in self.tokenization_outputs
            ],
        }
        if self.schema_version in {3, 4}:
            assert self.content_source_binding is not None
            value["content_source_binding"] = self.content_source_binding.to_dict()
        if self.schema_version == 4:
            assert self.qualification_plan_index is not None
            value["qualification_plan_index"] = self.qualification_plan_index.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("formal preflight execution inputs must be an object")
        schema_version = value.get("schema_version")
        expected = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "authority_mode",
            "execution_authority",
            "inventory",
            "content_receipt",
            "workload_authority",
            "doctor_report",
            "compile_assignment_plan",
            "exactness_assignment",
            "interference_manifest",
            "request_schedule_sources",
            "tokenization_inputs",
            "tokenization_outputs",
        }
        if schema_version in {3, 4}:
            expected.add("content_source_binding")
        if schema_version == 4:
            expected.add("qualification_plan_index")
        row = _strict(
            "formal preflight execution inputs",
            value,
            expected,
        )
        tuple_fields: dict[str, tuple[CanonicalJsonProofBinding, ...]] = {}
        for name in (
            "request_schedule_sources",
            "tokenization_inputs",
            "tokenization_outputs",
        ):
            raw = row.pop(name)
            if type(raw) is not list:
                raise TypeError(f"formal preflight {name} must be an array")
            tuple_fields[name] = tuple(
                CanonicalJsonProofBinding.from_dict(item) for item in raw
            )
        singles = {
            "execution_authority": CanonicalJsonProofBinding.from_dict(
                row.pop("execution_authority")
            ),
            "inventory": CanonicalJsonProofBinding.from_dict(row.pop("inventory")),
            "content_receipt": (
                None
                if row["content_receipt"] is None
                else CanonicalJsonProofBinding.from_dict(row["content_receipt"])
            ),
            "workload_authority": ContentJsonArtifactBinding.from_dict(
                row.pop("workload_authority")
            ),
            "doctor_report": CanonicalJsonProofBinding.from_dict(
                row.pop("doctor_report")
            ),
            "compile_assignment_plan": CanonicalJsonProofBinding.from_dict(
                row.pop("compile_assignment_plan")
            ),
            "exactness_assignment": CanonicalJsonProofBinding.from_dict(
                row.pop("exactness_assignment")
            ),
            "interference_manifest": CanonicalJsonProofBinding.from_dict(
                row.pop("interference_manifest")
            ),
        }
        row.pop("content_receipt")
        raw_content_source = row.pop("content_source_binding", None)
        singles["content_source_binding"] = (
            None
            if raw_content_source is None
            else FormalContentSourceBinding.from_dict(raw_content_source)
        )
        raw_qualification = row.pop("qualification_plan_index", None)
        singles["qualification_plan_index"] = (
            None
            if raw_qualification is None
            else CanonicalJsonProofBinding.from_dict(raw_qualification)
        )
        return cls(**row, **singles, **tuple_fields)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightCompletionRow:
    materialized_cell_id: str
    registry_cell_id: str
    runner_kind: Literal[
        "first_party_compile",
        "first_party_exactness",
        "first_party_interference",
    ]
    status: Literal["COMPLETE", "FAILED"]
    started_ns: int
    finished_ns: int
    result_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
            ("result", self.result_sha256),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"single-operator preflight {label} digest is invalid")
        if self.runner_kind not in _SINGLE_OPERATOR_PROCESS_TIMEOUT_NS:
            raise ValueError("single-operator preflight completion runner differs")
        if self.status not in {"COMPLETE", "FAILED"}:
            raise ValueError("single-operator preflight completion status differs")
        if (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 1
            or self.finished_ns < self.started_ns
        ):
            raise ValueError("single-operator preflight completion timing is invalid")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict(
                "single-operator preflight completion row",
                value,
                set(cls.__dataclass_fields__),
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightInterferenceEvidence:
    materialized_cell_id: str
    registry_cell_id: str
    terminal_result_proof: CanonicalJsonProofBinding
    lifecycle_timing: CanonicalJsonProofBinding
    junit_xml: EvidenceFileBinding

    def __post_init__(self) -> None:
        for label, value in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"single-operator preflight interference {label} is invalid"
                )
        if (
            any(
                type(value) is not CanonicalJsonProofBinding
                for value in (self.terminal_result_proof, self.lifecycle_timing)
            )
            or type(self.junit_xml) is not EvidenceFileBinding
        ):
            raise TypeError(
                "single-operator preflight interference evidence is not path-bound"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "materialized_cell_id": self.materialized_cell_id,
            "registry_cell_id": self.registry_cell_id,
            "terminal_result_proof": self.terminal_result_proof.to_dict(),
            "lifecycle_timing": self.lifecycle_timing.to_dict(),
            "junit_xml": self.junit_xml.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator preflight interference evidence",
            value,
            set(cls.__dataclass_fields__),
        )
        return cls(
            materialized_cell_id=row["materialized_cell_id"],
            registry_cell_id=row["registry_cell_id"],
            terminal_result_proof=CanonicalJsonProofBinding.from_dict(
                row["terminal_result_proof"]
            ),
            lifecycle_timing=CanonicalJsonProofBinding.from_dict(
                row["lifecycle_timing"]
            ),
            junit_xml=EvidenceFileBinding.from_dict(
                row["junit_xml"], label="single-operator preflight JUnit"
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightInterferenceExecution:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_preflight_interference_execution"]
    execution_inputs: CanonicalJsonProofBinding
    raw_batch: CanonicalJsonProofBinding
    evidence: tuple[FormalSingleOperatorPreflightInterferenceEvidence, ...]
    status: Literal["WAITING_FOR_COMPLETION", "ERROR"]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_single_operator_preflight_interference_execution"
        ):
            raise ValueError("single-operator interference execution schema differs")
        if any(
            type(value) is not CanonicalJsonProofBinding
            for value in (self.execution_inputs, self.raw_batch)
        ):
            raise TypeError("single-operator interference execution is not path-bound")
        if (
            type(self.evidence) is not tuple
            or self.evidence
            != tuple(sorted(self.evidence, key=lambda row: row.registry_cell_id))
            or len({row.registry_cell_id for row in self.evidence})
            != len(self.evidence)
        ):
            raise ValueError("single-operator interference evidence ordering differs")
        if self.status == "WAITING_FOR_COMPLETION" and len(self.evidence) != 8:
            raise ValueError(
                "single-operator interference execution is not exact eight"
            )
        if self.status not in {"WAITING_FOR_COMPLETION", "ERROR"}:
            raise ValueError("single-operator interference execution status differs")

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "execution_inputs": self.execution_inputs.to_dict(),
            "raw_batch": self.raw_batch.to_dict(),
            "evidence": [row.to_dict() for row in self.evidence],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator interference execution",
            value,
            set(cls.__dataclass_fields__),
        )
        values = row.pop("evidence")
        if type(values) is not list:
            raise TypeError("single-operator interference evidence must be an array")
        execution_inputs = CanonicalJsonProofBinding.from_dict(
            row.pop("execution_inputs")
        )
        raw_batch = CanonicalJsonProofBinding.from_dict(row.pop("raw_batch"))
        return cls(
            **row,
            execution_inputs=execution_inputs,
            raw_batch=raw_batch,
            evidence=tuple(
                FormalSingleOperatorPreflightInterferenceEvidence.from_dict(item)
                for item in values
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightExecution:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_exact_ten_preflight_execution"]
    protocol_sha256: str
    execution_inputs: CanonicalJsonProofBinding
    compile_result: CanonicalJsonProofBinding
    exactness_result: CanonicalJsonProofBinding
    interference_execution: CanonicalJsonProofBinding
    completion: CanonicalJsonProofBinding | None
    status: Literal["COMPLETE", "FAILED", "ERROR"]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_exact_ten_preflight_execution"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREFLIGHT_EXECUTION_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator preflight execution schema differs")
        if any(
            type(value) is not CanonicalJsonProofBinding
            for value in (
                self.execution_inputs,
                self.compile_result,
                self.exactness_result,
                self.interference_execution,
            )
        ) or (
            self.completion is not None
            and type(self.completion) is not CanonicalJsonProofBinding
        ):
            raise TypeError("single-operator preflight execution is not path-bound")
        if self.status not in {"COMPLETE", "FAILED", "ERROR"}:
            raise ValueError("single-operator preflight execution status differs")
        if (self.status == "ERROR") != (self.completion is None):
            raise ValueError("single-operator preflight completion presence differs")

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "execution_inputs": self.execution_inputs.to_dict(),
            "compile_result": self.compile_result.to_dict(),
            "exactness_result": self.exactness_result.to_dict(),
            "interference_execution": self.interference_execution.to_dict(),
            "completion": (
                None if self.completion is None else self.completion.to_dict()
            ),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator preflight execution",
            value,
            set(cls.__dataclass_fields__),
        )
        for name in (
            "execution_inputs",
            "compile_result",
            "exactness_result",
            "interference_execution",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        completion = row.pop("completion")
        row["completion"] = (
            None
            if completion is None
            else CanonicalJsonProofBinding.from_dict(completion)
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightCompletion:
    schema_version: Literal[1, 2]
    kind: Literal["formal_single_operator_exact_ten_preflight_completion"]
    protocol_sha256: str
    execution_inputs: CanonicalJsonProofBinding
    compile_result: CanonicalJsonProofBinding
    exactness_result: CanonicalJsonProofBinding
    interference_evidence: tuple[FormalSingleOperatorPreflightInterferenceEvidence, ...]
    rows: tuple[FormalSingleOperatorPreflightCompletionRow, ...]
    status: Literal["COMPLETE", "FAILED"]
    started_ns: int
    finished_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2}
            or self.kind != "formal_single_operator_exact_ten_preflight_completion"
            or self.protocol_sha256
            != (
                TRUSTED_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256
                if self.schema_version == 2
                else FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256
            )
        ):
            raise ValueError("single-operator preflight completion schema differs")
        if any(
            type(value) is not CanonicalJsonProofBinding
            for value in (
                self.execution_inputs,
                self.compile_result,
                self.exactness_result,
            )
        ):
            raise TypeError("single-operator preflight completion source is not bound")
        if (
            len(self.interference_evidence) != 8
            or self.interference_evidence
            != tuple(
                sorted(
                    self.interference_evidence,
                    key=lambda row: row.registry_cell_id,
                )
            )
            or len({row.registry_cell_id for row in self.interference_evidence}) != 8
        ):
            raise ValueError(
                "single-operator preflight interference completion is not exact eight"
            )
        if (
            len(self.rows) != 10
            or self.rows
            != tuple(sorted(self.rows, key=lambda row: row.registry_cell_id))
            or len({row.registry_cell_id for row in self.rows}) != 10
            or [row.runner_kind for row in self.rows].count("first_party_compile") != 1
            or [row.runner_kind for row in self.rows].count("first_party_exactness")
            != 1
            or [row.runner_kind for row in self.rows].count("first_party_interference")
            != 8
        ):
            raise ValueError("single-operator preflight completion is not exact ten")
        expected_status = (
            "COMPLETE"
            if all(row.status == "COMPLETE" for row in self.rows)
            else "FAILED"
        )
        if (
            self.status != expected_status
            or self.started_ns != min(row.started_ns for row in self.rows)
            or self.finished_ns != max(row.finished_ns for row in self.rows)
        ):
            raise ValueError("single-operator preflight aggregate outcome differs")

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "execution_inputs": self.execution_inputs.to_dict(),
            "compile_result": self.compile_result.to_dict(),
            "exactness_result": self.exactness_result.to_dict(),
            "interference_evidence": [
                row.to_dict() for row in self.interference_evidence
            ],
            "rows": [row.to_dict() for row in self.rows],
            "status": self.status,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator preflight completion",
            value,
            set(cls.__dataclass_fields__),
        )
        raw_evidence = row.pop("interference_evidence")
        raw_rows = row.pop("rows")
        if type(raw_evidence) is not list or type(raw_rows) is not list:
            raise TypeError("single-operator preflight completion rows are malformed")
        execution_inputs = CanonicalJsonProofBinding.from_dict(
            row.pop("execution_inputs")
        )
        compile_result = CanonicalJsonProofBinding.from_dict(row.pop("compile_result"))
        exactness_result = CanonicalJsonProofBinding.from_dict(
            row.pop("exactness_result")
        )
        return cls(
            **row,
            execution_inputs=execution_inputs,
            compile_result=compile_result,
            exactness_result=exactness_result,
            interference_evidence=tuple(
                FormalSingleOperatorPreflightInterferenceEvidence.from_dict(item)
                for item in raw_evidence
            ),
            rows=tuple(
                FormalSingleOperatorPreflightCompletionRow.from_dict(item)
                for item in raw_rows
            ),
        )  # type: ignore[arg-type]


def _materialize_exact_ten_inputs(
    *,
    root: Path,
    execution_authority_binding: CanonicalJsonProofBinding,
    inventory_binding: CanonicalJsonProofBinding,
    token: Any,
    materialization: Any,
    authority_mode: Literal["formal_dispatch", "formal_single_operator_v1"],
    enforce_formal_dispatch_contracts: bool,
    content_verification_receipt_path: str | Path | None,
    content_source_binding: FormalContentSourceBinding | None,
    workload_authority_path: str | Path,
    doctor_report_path: str | Path,
    current_ns: int,
) -> CanonicalJsonProofBinding:
    """Shared deterministic publisher after one mode-specific source rebuild."""

    doctor_binding, doctor, checkout, nvidia_smi, cuda_home = _doctor_report(
        doctor_report_path, token=token
    )
    trusted_content = content_source_binding is not None
    if trusted_content:
        if content_verification_receipt_path is not None:
            raise ValueError("trusted preflight cannot carry a signed content receipt")
        assert content_source_binding is not None
        (
            trusted_bundle,
            workload_binding,
            workload,
            trusted_workload_member,
            trusted_target,
            trusted_drafter,
            trusted_tokenizer,
        ) = _trusted_content_sources(
            content_source_binding=content_source_binding,
            workload_authority_path=workload_authority_path,
        )
        content_binding = None
        content_receipt = None
        prepared = None
        workload_sources = None
        descriptor_sha256 = trusted_workload_member.sha256
        snapshots = None
    else:
        if content_verification_receipt_path is None:
            raise ValueError("signed preflight lacks its content receipt")
        (
            content_binding,
            content_receipt,
            prepared,
            workload_sources,
            workload_binding,
            workload,
            descriptor_sha256,
            snapshots,
        ) = _content_sources(
            content_receipt_path=content_verification_receipt_path,
            workload_authority_path=workload_authority_path,
            current_ns=current_ns,
        )
    source_authorities = dict(_dispatch_bindings(token)[0].source_authority_bindings)
    if trusted_content:
        assert content_source_binding is not None
        if source_authorities.get(
            "trusted_single_operator_content_bundle"
        ) != content_source_binding.content_sha256 or set(source_authorities) != {
            "trusted_single_operator_content_bundle",
            "compile_qualification",
            "exactness_qualification",
            "native_runtime_qualification",
        }:
            raise ValueError("trusted preflight content differs from ProtocolLock")
    else:
        assert prepared is not None and workload_sources is not None
        if (
            prepared.authorization_sha256
            != source_authorities["prepared_model_content"]
            or workload_sources.authorization_sha256
            != source_authorities["formal_workload_e3a"]
        ):
            raise ValueError(
                "formal preflight content differs from current execution authority"
            )
        assert snapshots is not None and content_receipt is not None
        target, drafter, tokenizer = _select_model_sources(snapshots)
        prepared_manifest = _rebuild_prepared_content_manifest(
            root=root,
            receipt=content_receipt,
            prepared=prepared,
        )
    sampling = SamplingProfile()
    sampling_path = root / "sampling-profile.json"
    sampling.write(sampling_path)
    runtime_qualification_sha256 = source_authorities["native_runtime_qualification"]
    launches: dict[
        str, tuple[CompileLaunchManifest, Path, CompileCacheLaunchPlan, Path, RunConfig]
    ] = {}
    for binding in _dispatch_bindings(token):
        if trusted_content:
            assert content_source_binding is not None
            launches[binding.registry_cell_id] = _trusted_launch_manifest(
                root=root,
                binding=binding,
                doctor=doctor,
                checkout=checkout,
                cuda_home=cuda_home,
                content_source_binding=content_source_binding,
                sampling_profile=sampling,
                sampling_path=sampling_path,
                target=trusted_target,
                drafter=trusted_drafter,
                tokenizer=trusted_tokenizer,
                runtime_qualification_sha256=runtime_qualification_sha256,
                inventory_sha256=token.subject.inventory_sha256,
                budget_materialization_authority_sha256=(
                    token.subject.budget_plan_sha256
                ),
            )
        else:
            assert prepared is not None
            launches[binding.registry_cell_id] = _launch_manifest(
                root=root,
                binding=binding,
                doctor=doctor,
                checkout=checkout,
                cuda_home=cuda_home,
                prepared=prepared,
                prepared_manifest=prepared_manifest,
                sampling_profile=sampling,
                sampling_path=sampling_path,
                target_source=target,
                drafter_source=drafter,
                tokenizer_source=tokenizer,
                runtime_qualification_sha256=runtime_qualification_sha256,
                inventory_sha256=token.subject.inventory_sha256,
                budget_materialization_authority_sha256=(
                    token.subject.budget_plan_sha256
                ),
            )

    compile_binding = _one_binding(token, "first_party_compile")
    compile_launch, compile_launch_path, cache_plan, prewarm_path, _config = launches[
        compile_binding.registry_cell_id
    ]
    assignment_root = root / "compile"
    assignment_root.mkdir(mode=0o700)
    prewarm = CompileOnlyPrewarmManifest.from_dict(
        json.loads(prewarm_path.read_text(encoding="utf-8"))
    )
    assignment = CompileOnlyAssignmentContract(
        schema_version=1,
        kind="compile_only_assignment_contract",
        assignment_protocol_sha256=COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
        cell_id=compile_binding.registry_cell_id,
        registry_sha256=token.manifest.registry_sha256,
        runtime_sha256=token.dispatch_context.activation_artifact.runtime_sha256,
        split_sha256=token.dispatch_context.activation_artifact.split_sha256,
        physical_assignment_sha256=compile_binding.assignment_sha256,
        experiment_budget_sha256=compile_binding.experiment_budget_sha256,
        budget_materialization_authority_sha256=(token.subject.budget_plan_sha256),
        inventory_sha256=token.subject.inventory_sha256,
        inventory_source_receipt_sha256=(
            token.dispatch_context.inventory.source_receipt_sha256
        ),
        gpu_uuids=compile_binding.gpu_uuids,
        host_id=token.dispatch_context.inventory.device(
            compile_binding.gpu_uuids[0]
        ).host_id,
        fixed_instance_gpu_count=len(token.dispatch_context.inventory.devices),
        compile_cache_plan=cache_plan,
        prewarm_manifest=prewarm,
        graceful_shutdown_protocol_sha256=COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256,
        result_pointer_protocol_sha256=COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
        result_pointer_path=str((assignment_root / "result.json").resolve()),
    )
    assignment_path = assignment.write(assignment_root / "assignment.json")
    compile_plan = CompileAssignmentPlan.issue(
        assignment_manifest_path=assignment_path,
        compile_cache_plan_path=compile_launch.compile_cache_plan_path,
        prewarm_manifest_path=prewarm_path,
        launch_manifest_path=compile_launch_path,
        result_pointer_path=assignment.result_pointer_path,
        attempt_id=_FIXED_ATTEMPT_ID,
    )
    compile_plan_path = root / "compile-assignment-plan.json"
    compile_plan.write(compile_plan_path)
    if enforce_formal_dispatch_contracts:
        assert content_binding is not None
        require_formal_preflight_compile_assignment(
            token,
            assignment_plan_path=compile_plan_path,
            prepared_content_verification_receipt_path=content_binding.absolute_path,
            now_ns=current_ns,
        )

    exact_binding = _one_binding(token, "first_party_exactness")
    exact_root = root / "exactness"
    exact_root.mkdir(mode=0o700)
    exact_devices = tuple(
        token.dispatch_context.inventory.device(uuid)
        for uuid in exact_binding.gpu_uuids
    )
    if (
        len(exact_devices) != 2
        or len({row.hardware_envelope_sha256 for row in exact_devices}) != 1
    ):
        raise ValueError("formal preflight exactness hardware envelope differs")
    python_path = Path(sys.executable).resolve()
    source = dict(exact_binding.source_authority_bindings)
    if trusted_content:
        assert content_source_binding is not None
        input_locks = PreflightInputLocks(
            prepared_model_set_sha256=None,
            prepared_model_content_authority_sha256=None,
            formal_workload_lock_sha256=None,
            burstgpt_shape_authority=None,
            content_source_binding=content_source_binding,
            trusted_model_member_sha256s=tuple(
                sorted(
                    (
                        trusted_target.sha256,
                        trusted_drafter.sha256,
                        trusted_tokenizer.sha256,
                    )
                )
            ),
            trusted_workload_member_sha256=trusted_workload_member.sha256,
            trusted_burstgpt_release_sha256=(trusted_bundle.burstgpt_release.sha256),
        )
    else:
        assert prepared is not None and content_binding is not None
        input_locks = PreflightInputLocks(
            prepared_model_set_sha256=(
                prepared.authorization.prepared_model_set_sha256
            ),
            prepared_model_content_authority_sha256=(prepared.authorization_sha256),
            formal_workload_lock_sha256=source["formal_workload_e3a"],
            burstgpt_shape_authority=(
                derive_burstgpt_shape_authority_from_content_receipt(
                    content_binding.absolute_path,
                    current_ns=current_ns,
                )
            ),
        )
    exact_assignment = ExactnessPreflightAssignment(
        schema_version=3,
        kind="formal_exactness_preflight_assignment",
        protocol_sha256=PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
        registry_sha256=token.manifest.registry_sha256,
        cell_id=exact_binding.registry_cell_id,
        runtime_sha256=token.dispatch_context.activation_artifact.runtime_sha256,
        split_sha256=token.dispatch_context.activation_artifact.split_sha256,
        inventory_sha256=token.subject.inventory_sha256,
        hardware_envelope_sha256=exact_devices[0].hardware_envelope_sha256,
        physical_assignment_sha256=exact_binding.assignment_sha256,
        experiment_budget_sha256=exact_binding.experiment_budget_sha256,
        gpu_uuids=exact_binding.gpu_uuids,  # type: ignore[arg-type]
        gpu_model=exact_devices[0].model,
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        python_executable=str(python_path),
        python_raw_sha256=_raw_sha256(python_path),
        nvidia_smi_executable=str(nvidia_smi),
        nvidia_smi_raw_sha256=_raw_sha256(nvidia_smi),
        python_version=_toolchain(doctor, gpu_uuid=exact_binding.gpu_uuids[0])[0],
        torch_version=_toolchain(doctor, gpu_uuid=exact_binding.gpu_uuids[0])[1],
        cuda_version=_toolchain(doctor, gpu_uuid=exact_binding.gpu_uuids[0])[3],
        driver_version=_toolchain(doctor, gpu_uuid=exact_binding.gpu_uuids[0])[4],
        input_locks=input_locks,
        loader_environment=ExactnessLoaderEnvironment(
            path_entries=(str(python_path.parent), str(nvidia_smi.parent)),
            library_path_entries=(str((cuda_home / "lib64").resolve()),),
            cuda_home=str(cuda_home),
        ),
        evidence_directory=str(exact_root),
    )
    exact_path = root / "exactness-assignment.json"
    exact_assignment.write(exact_path)
    if enforce_formal_dispatch_contracts:
        require_formal_preflight_exactness_assignment(token, assignment_path=exact_path)

    input_rows = []
    schedule_bindings = []
    token_inputs = []
    token_outputs = []
    for binding in sorted(
        (
            row
            for row in _dispatch_bindings(token)
            if row.runner_kind == "first_party_interference"
        ),
        key=lambda row: row.registry_cell_id,
    ):
        launch, launch_path, _cache, _prewarm_path, config = launches[
            binding.registry_cell_id
        ]
        if trusted_content:
            assert content_source_binding is not None
            schedule = rebuild_trusted_single_operator_request_schedule_source(
                subject_sha256=token.sha256,
                content_source_binding=content_source_binding,
                topology_mode="tp1_dp1",
                materialization=materialization,
                materialized_cell_id=binding.materialized_cell_id,
                workload_source=workload,
                workload_source_binding=workload_binding,
                sampling_profile=sampling,
                max_running_requests=config.runtime.max_running_requests,
                server_context_limit=config.runtime.context_length,
                tokenizer_content_member_id=launch.tokenizer_content_member_id,
                tokenizer_model_id=launch.tokenizer_model_id,
                tokenizer_revision=launch.tokenizer_revision,
            )
        else:
            schedule = rebuild_formal_serving_request_schedule_source(
                subject_sha256=token.sha256,
                workload_authority_sha256=workload_binding.semantic_sha256,
                topology_mode="tp1_dp1",
                materialization=materialization,
                materialized_cell_id=binding.materialized_cell_id,
                workload_source=workload,  # type: ignore[arg-type]
                workload_source_descriptor_sha256=descriptor_sha256,
                tts_tuning_window=None,
                sampling_profile=sampling,
                max_running_requests=config.runtime.max_running_requests,
                server_context_limit=config.runtime.context_length,
                tokenizer_content_member_id=launch.tokenizer_content_member_id,
                tokenizer_model_id=launch.tokenizer_model_id,
                tokenizer_revision=launch.tokenizer_revision,
                tokenizer_content_authority_sha256=(
                    launch.tokenizer_content_authority_sha256
                ),
            )
        row_root = root / f"row-{binding.registry_cell_id}"
        schedule_path = row_root / "request-schedule-source.json"
        schedule_binding = _publish_binding(schedule_path, schedule.to_dict())
        token_input = _publish_tokenization_input(
            path=row_root / "tokenization-input.json", source=schedule, launch=launch
        )
        token_output, _worker_sha, _worker_size, _argv_sha = _invoke_tokenizer_worker(
            input_path=Path(token_input.absolute_path),
            output_path=row_root / "tokenization-output.json",
        )
        rows = _materialized_schedule_rows(
            source=schedule,
            launch=launch,
            tokenization_input=token_input,
            tokenization_output=token_output,
        )
        warmup = tuple(row.request for row in rows if row.phase == "warmup")
        scored = tuple(row.request for row in rows if row.phase == "scored")
        run_id = f"preflight-{binding.materialized_cell_id[:16]}"
        run_binding = NativeTerminalRunBinding(
            run_id=run_id,
            run_nonce_sha256=content_sha256(
                {
                    "dispatch": token.sha256,
                    "cell": binding.materialized_cell_id,
                    "attempt": 0,
                }
            ),
            execution_plan_sha256=token.dispatch_plan.sha256,
            rank_config_sha256=content_sha256(
                {"launch": launch.sha256, "assignment": binding.assignment_sha256}
            ),
            attempt_id=_FIXED_ATTEMPT_ID,
            session_id=f"session-{binding.materialized_cell_id[:16]}",
            session_epoch=1,
            previous_run_id=None,
            challenge_nonce_sha256=content_sha256(
                {
                    "dispatch": token.sha256,
                    "cell": binding.materialized_cell_id,
                    "challenge": 0,
                }
            ),
            method="static",
            warmup_request_ids=tuple(row.request_id for row in warmup),
            scored_request_ids=tuple(row.request_id for row in scored),
        )
        input_rows.append(
            FormalPreflightInterferenceRunInput(
                registry_cell_id=binding.registry_cell_id,
                launch_manifest_path=str(launch_path),
                run_binding=run_binding,
                warmup_requests=warmup,
                scored_requests=scored,
                qualification_rows=tuple(
                    FormalPreflightInterferenceQualificationRow(
                        request_id=row.request_id,
                        prompt_bucket=formal_prompt_bucket(len(row.input_token_ids)),
                        eligible=True,
                    )
                    for row in scored
                ),
            )
        )
        schedule_bindings.append(schedule_binding)
        token_inputs.append(token_input)
        token_outputs.append(token_output)
    manifest = FormalPreflightInterferenceExecutionManifest(
        schema_version=1,
        kind="formal_preflight_interference_execution_manifest",
        dispatch_receipt_semantic_sha256=(execution_authority_binding.semantic_sha256),
        inputs=tuple(sorted(input_rows, key=lambda row: row.registry_cell_id)),
    )
    manifest_path = root / "interference-execution-manifest.json"
    manifest_binding = _publish_binding(manifest_path, manifest.to_dict())
    qualification_plan_index = None
    if trusted_content:
        from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
            materialize_formal_single_operator_preflight_qualification_plans,
            publish_formal_single_operator_preflight_qualification_launch_index,
        )

        authority = FormalSingleOperatorPreflightAuthority.from_dict(
            execution_authority_binding.reopen()
        )
        tp1_registry_cell_id = min(row.registry_cell_id for row in input_rows)
        qualification_launch_index = (
            publish_formal_single_operator_preflight_qualification_launch_index(
                protocol_lock_path=authority.protocol_lock.absolute_path,
                content_source_path=(
                    content_source_binding.trusted_single_operator.absolute_path
                ),
                inventory_path=inventory_binding.absolute_path,
                doctor_report_path=doctor_binding.absolute_path,
                exactness_assignment_path=exact_path,
                base_tp1_launch_path=launches[tp1_registry_cell_id][1],
                base_tp2_launch_path=launches[exact_binding.registry_cell_id][1],
                output_root=exact_root / "qualification-launches",
            )
        )
        qualification_plan_index = (
            materialize_formal_single_operator_preflight_qualification_plans(
                qualification_launch_index_path=(
                    qualification_launch_index.absolute_path
                ),
                output_root=exact_root / "qualification",
            )
        )
    artifact = FormalPreflightExecutionInputs(
        schema_version=4 if trusted_content else 2,
        kind="formal_single_operator_exact_ten_preflight_inputs",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_QUALIFIED_PREFLIGHT_INPUTS_PROTOCOL_SHA256
            if trusted_content
            else FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256
        ),
        authority_mode=authority_mode,
        execution_authority=execution_authority_binding,
        inventory=inventory_binding,
        content_receipt=content_binding,
        workload_authority=workload_binding,
        doctor_report=doctor_binding,
        compile_assignment_plan=CanonicalJsonProofBinding.bind(compile_plan_path),
        exactness_assignment=CanonicalJsonProofBinding.bind(exact_path),
        interference_manifest=manifest_binding,
        request_schedule_sources=tuple(schedule_bindings),
        tokenization_inputs=tuple(token_inputs),
        tokenization_outputs=tuple(token_outputs),
        content_source_binding=content_source_binding,
        qualification_plan_index=qualification_plan_index,
    )
    output_path = root / "formal-preflight-execution-inputs.json"
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    loaded = FormalPreflightExecutionInputs.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if loaded != artifact:
        raise RuntimeError("written formal preflight execution inputs changed")
    return CanonicalJsonProofBinding.bind(output_path, semantic_sha256=artifact.sha256)


def materialize_formal_preflight_execution_inputs(
    *,
    dispatch_receipt_path: str | Path,
    content_verification_receipt_path: str | Path,
    workload_authority_path: str | Path,
    doctor_report_path: str | Path,
    private_output_root: str | Path,
    current_ns: int,
) -> CanonicalJsonProofBinding:
    """Compatibility producer for an already sealed formal dispatch receipt."""

    root = _private_new_root(private_output_root)
    dispatch_binding = CanonicalJsonProofBinding.bind(dispatch_receipt_path)
    dispatch_receipt = FormalPreflightDispatchReceipt.from_dict(
        dispatch_binding.reopen()
    )
    token = dispatch_receipt.revalidate(current_ns=current_ns)
    if dispatch_binding.semantic_sha256 != dispatch_receipt.sha256:
        raise ValueError("formal preflight dispatch receipt identity differs")
    inventory_binding = _publish_binding(
        root / "verified-inventory.json",
        token.dispatch_context.inventory.to_dict(),
    )
    return _materialize_exact_ten_inputs(
        root=root,
        execution_authority_binding=dispatch_binding,
        inventory_binding=inventory_binding,
        token=token,
        materialization=dispatch_receipt.signed_materialization.payload,
        authority_mode="formal_dispatch",
        enforce_formal_dispatch_contracts=True,
        content_verification_receipt_path=content_verification_receipt_path,
        content_source_binding=None,
        workload_authority_path=workload_authority_path,
        doctor_report_path=doctor_report_path,
        current_ns=current_ns,
    )


def materialize_formal_single_operator_preflight_execution_inputs(
    *,
    execution_source_path: str | Path,
    repository_root: str | Path,
    formal_runtime_authority_manifest_path: str | Path,
    inventory_path: str | Path,
    content_verification_receipt_path: str | Path | None = None,
    content_source_path: str | Path | None = None,
    workload_authority_path: str | Path,
    doctor_report_path: str | Path,
    private_output_root: str | Path,
    current_ns: int,
) -> CanonicalJsonProofBinding:
    """Build exact 1+1+8 inputs from the trusted current preflight source."""

    source_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source = load_formal_single_operator_execution_source(execution_source_path)
    if source.schema_version == 3:
        if (
            content_source_path is None
            or content_verification_receipt_path is not None
            or source.content_source_binding is None
        ):
            raise ValueError("trusted preflight requires its tagged content source")
        selected_content_source = (
            FormalContentSourceBinding.bind_trusted_single_operator(
                str(content_source_path)
            )
        )
        if selected_content_source != source.content_source_binding:
            raise ValueError("trusted preflight content source differs from execution")
    else:
        if content_verification_receipt_path is None or content_source_path is not None:
            raise ValueError("legacy preflight requires its signed content receipt")
        selected_content_source = None
    protocol_lock_binding = CanonicalJsonProofBinding.bind(
        source.protocol_lock_source.absolute_path
    )
    protocol_lock = protocol_lock_from_dict(protocol_lock_binding.reopen())
    repository, head, tree = _trusted_repository_identity(
        repository_root,
        protocol_lock=protocol_lock,
    )
    requested_root = Path(private_output_root)
    if requested_root.is_relative_to(repository):
        raise ValueError("single-operator preflight output must be outside checkout")
    root = _private_new_root(requested_root)
    runtime_binding = CanonicalJsonProofBinding.bind(
        formal_runtime_authority_manifest_path
    )
    runtime = build_source_formal_runtime_authority_manifest(repository)
    from lightcone_spec.experiments.formal_registry import (
        formal_runtime_authority_manifest_from_dict,
    )

    persisted_runtime = formal_runtime_authority_manifest_from_dict(
        runtime_binding.reopen()
    )
    if (
        persisted_runtime != runtime
        or runtime.sha256 != source.runtime_authority_manifest_sha256
    ):
        raise ValueError("single-operator runtime manifest differs from current source")
    inventory_binding = CanonicalJsonProofBinding.bind(inventory_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    bindings = _trusted_preflight_bindings(
        source=source,
        protocol_lock=protocol_lock,
        inventory=inventory,
    )
    materialization_binding = CanonicalJsonProofBinding.bind(
        source.materialization_source.absolute_path
    )
    materialization = stage_materialization_receipt_from_dict(
        materialization_binding.reopen()
    )
    budget_plan_sha256 = _trusted_budget_plan_sha256(
        source,
        inventory,
        bindings,
    )
    split_sha256 = _trusted_split_sha256(materialization, bindings)
    authority = FormalSingleOperatorPreflightAuthority(
        schema_version=1,
        kind="formal_single_operator_preflight_authority",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
        execution_source=source_binding,
        protocol_lock=protocol_lock_binding,
        materialization=materialization_binding,
        runtime_authority_manifest=runtime_binding,
        inventory=inventory_binding,
        repository_root=str(repository),
        repository_head=head,
        repository_tree=tree,
        patch_manifest_sha256=protocol_lock.patch_manifest_sha256,
        registry_sha256=protocol_lock.registry_sha256,
        budget_plan_sha256=budget_plan_sha256,
        split_sha256=split_sha256,
        execution_bindings=bindings,
    )
    authority_path = root / "formal-single-operator-preflight-authority.json"
    authority_binding = _publish_binding(authority_path, authority.to_dict())
    token = _trusted_token(
        authority=authority,
        source=source,
        inventory=inventory,
    )
    return _materialize_exact_ten_inputs(
        root=root,
        execution_authority_binding=authority_binding,
        inventory_binding=inventory_binding,
        token=token,
        materialization=materialization,
        authority_mode="formal_single_operator_v1",
        enforce_formal_dispatch_contracts=False,
        content_verification_receipt_path=content_verification_receipt_path,
        content_source_binding=selected_content_source,
        workload_authority_path=workload_authority_path,
        doctor_report_path=doctor_report_path,
        current_ns=current_ns,
    )


def load_formal_preflight_execution_inputs(
    path: str | Path,
) -> FormalPreflightExecutionInputs:
    binding = CanonicalJsonProofBinding.bind(path)
    value = FormalPreflightExecutionInputs.from_dict(binding.reopen())
    if binding.semantic_sha256 != value.sha256:
        raise ValueError("formal preflight execution inputs identity differs")
    return value


def revalidate_formal_single_operator_preflight_execution_inputs(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalPreflightExecutionInputs:
    """Deep-rebuild trusted exact-ten inputs from their current source paths."""

    artifact_binding = CanonicalJsonProofBinding.bind(path)
    artifact = load_formal_preflight_execution_inputs(path)
    if artifact.authority_mode != "formal_single_operator_v1":
        raise ValueError("trusted preflight revalidator rejects formal-dispatch mode")
    authority = FormalSingleOperatorPreflightAuthority.from_dict(
        artifact.execution_authority.reopen()
    )
    if artifact.execution_authority.semantic_sha256 != authority.sha256:
        raise ValueError("single-operator preflight authority identity differs")
    source = load_formal_single_operator_execution_source(
        authority.execution_source.absolute_path
    )
    if authority.execution_source != CanonicalJsonProofBinding.bind(
        authority.execution_source.absolute_path
    ):
        raise ValueError("single-operator execution source bytes changed")
    protocol_lock = protocol_lock_from_dict(authority.protocol_lock.reopen())
    repository, head, tree = _trusted_repository_identity(
        authority.repository_root,
        protocol_lock=protocol_lock,
    )
    if (
        (head, tree) != (authority.repository_head, authority.repository_tree)
        or authority.patch_manifest_sha256 != protocol_lock.patch_manifest_sha256
        or authority.registry_sha256 != protocol_lock.registry_sha256
        or Path(artifact_binding.absolute_path).is_relative_to(repository)
    ):
        raise ValueError("single-operator preflight release identity changed")
    from lightcone_spec.experiments.formal_registry import (
        formal_runtime_authority_manifest_from_dict,
    )

    persisted_runtime = formal_runtime_authority_manifest_from_dict(
        authority.runtime_authority_manifest.reopen()
    )
    current_runtime = build_source_formal_runtime_authority_manifest(repository)
    if (
        persisted_runtime != current_runtime
        or persisted_runtime.sha256 != source.runtime_authority_manifest_sha256
    ):
        raise ValueError("single-operator preflight runtime authority changed")
    inventory = GpuInventory.from_dict(authority.inventory.reopen())
    if artifact.inventory != authority.inventory:
        raise ValueError("single-operator preflight inventory binding differs")
    materialization = stage_materialization_receipt_from_dict(
        authority.materialization.reopen()
    )
    expected_bindings = _trusted_preflight_bindings(
        source=source,
        protocol_lock=protocol_lock,
        inventory=inventory,
    )
    if (
        authority.execution_bindings != expected_bindings
        or authority.budget_plan_sha256
        != _trusted_budget_plan_sha256(source, inventory, expected_bindings)
        or authority.split_sha256
        != _trusted_split_sha256(materialization, expected_bindings)
    ):
        raise ValueError("single-operator preflight assignment projection changed")
    token = _trusted_token(
        authority=authority,
        source=source,
        inventory=inventory,
    )
    _doctor_report(artifact.doctor_report.absolute_path, token=token)
    source_authorities = dict(expected_bindings[0].source_authority_bindings)
    if artifact.schema_version in {3, 4}:
        if (
            artifact.content_receipt is not None
            or artifact.content_source_binding != source.content_source_binding
            or artifact.content_source_binding is None
        ):
            raise ValueError("trusted preflight content binding changed")
        (
            _trusted_bundle,
            workload_binding,
            _workload,
            _trusted_workload,
            _trusted_target,
            _trusted_drafter,
            _trusted_tokenizer,
        ) = _trusted_content_sources(
            content_source_binding=artifact.content_source_binding,
            workload_authority_path=artifact.workload_authority.path,
        )
        if (
            workload_binding != artifact.workload_authority
            or source_authorities.get("trusted_single_operator_content_bundle")
            != artifact.content_source_binding.content_sha256
        ):
            raise ValueError("trusted preflight content projection changed")
    else:
        assert artifact.content_receipt is not None
        (
            content_binding,
            _receipt,
            prepared,
            workload_sources,
            workload_binding,
            _workload,
            _descriptor,
            _snapshots,
        ) = _content_sources(
            content_receipt_path=artifact.content_receipt.absolute_path,
            workload_authority_path=artifact.workload_authority.path,
            current_ns=current_ns,
        )
        if (
            content_binding != artifact.content_receipt
            or workload_binding != artifact.workload_authority
            or prepared.authorization_sha256
            != source_authorities["prepared_model_content"]
            or workload_sources.authorization_sha256
            != source_authorities["formal_workload_e3a"]
        ):
            raise ValueError("single-operator preflight content projection changed")
    compile_plan = CompileAssignmentPlan.load(
        artifact.compile_assignment_plan.absolute_path
    )
    compile_assignment, _cache, _prewarm, compile_launch = compile_plan.revalidate()
    compile_binding = _one_binding(token, "first_party_compile")
    if (
        compile_assignment.cell_id != compile_binding.registry_cell_id
        or compile_assignment.physical_assignment_sha256
        != compile_binding.assignment_sha256
        or compile_assignment.experiment_budget_sha256
        != compile_binding.experiment_budget_sha256
        or compile_assignment.inventory_sha256 != inventory.sha256
        or compile_launch.physical_assignment_sha256
        != compile_binding.assignment_sha256
        or compile_launch.inventory_sha256 != inventory.sha256
        or compile_launch.gpu_uuids != compile_binding.gpu_uuids
    ):
        raise ValueError("single-operator compile input differs from exact binding")
    exactness = ExactnessPreflightAssignment.load(
        artifact.exactness_assignment.absolute_path
    )
    exact_binding = _one_binding(token, "first_party_exactness")
    if (
        exactness.cell_id != exact_binding.registry_cell_id
        or exactness.physical_assignment_sha256 != exact_binding.assignment_sha256
        or exactness.experiment_budget_sha256 != exact_binding.experiment_budget_sha256
        or exactness.inventory_sha256 != inventory.sha256
        or exactness.gpu_uuids != exact_binding.gpu_uuids
    ):
        raise ValueError("single-operator exactness input differs from exact binding")
    if artifact.schema_version in {3, 4}:
        assert artifact.content_source_binding is not None
        trusted_bundle = artifact.content_source_binding.reopen()
        expected_models = tuple(
            sorted(
                row.sha256
                for row in trusted_bundle.model_members
                if "preflight" in row.stages
                and (
                    (row.role == "target" and row.model_id == _TARGET_MODEL_ID)
                    or (row.role == "drafter" and row.model_id == _DRAFTER_MODEL_ID)
                    or (row.role == "tokenizer" and row.model_id == _TARGET_MODEL_ID)
                )
            )
        )
        locked = exactness.input_locks
        if (
            locked.content_source_binding != artifact.content_source_binding
            or locked.trusted_model_member_sha256s != expected_models
            or locked.trusted_workload_member_sha256 != _trusted_workload.sha256
            or locked.trusted_burstgpt_release_sha256
            != trusted_bundle.burstgpt_release.sha256
        ):
            raise ValueError("trusted exactness content locks changed")
    interference = FormalPreflightInterferenceExecutionManifest.from_dict(
        artifact.interference_manifest.reopen()
    )
    expected_interference = tuple(
        row.registry_cell_id
        for row in expected_bindings
        if row.runner_kind == "first_party_interference"
    )
    if (
        interference.dispatch_receipt_semantic_sha256
        != artifact.execution_authority.semantic_sha256
        or tuple(row.registry_cell_id for row in interference.inputs)
        != expected_interference
    ):
        raise ValueError("single-operator interference inputs differ from exact eight")
    binding_by_registry = {row.registry_cell_id: row for row in expected_bindings}
    for row in interference.inputs:
        expected = binding_by_registry[row.registry_cell_id]
        launch = CompileLaunchManifest.load(row.launch_manifest_path)
        if (
            launch.physical_assignment_sha256 != expected.assignment_sha256
            or launch.experiment_budget_sha256 != expected.experiment_budget_sha256
            or launch.inventory_sha256 != inventory.sha256
            or launch.gpu_uuids != expected.gpu_uuids
            or (
                artifact.schema_version in {3, 4}
                and (
                    launch.schema_version != 2
                    or launch.content_source_binding != artifact.content_source_binding
                    or launch.formal_stage != "preflight"
                )
            )
        ):
            raise ValueError("single-operator interference launch binding changed")
    if artifact.schema_version in {3, 4}:
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            FormalServingRequestScheduleSource,
        )

        assert artifact.content_source_binding is not None
        for binding in artifact.request_schedule_sources:
            schedule = FormalServingRequestScheduleSource.from_dict(binding.reopen())
            if (
                schedule.schema_version != 5
                or schedule.content_source_binding_sha256
                != artifact.content_source_binding.sha256
                or schedule.trusted_workload_member_sha256 != _trusted_workload.sha256
            ):
                raise ValueError("trusted preflight interference schedule changed")
    for binding in (
        *artifact.request_schedule_sources,
        *artifact.tokenization_inputs,
        *artifact.tokenization_outputs,
    ):
        binding.reopen()
    if artifact.schema_version == 4:
        from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
            load_formal_single_operator_preflight_qualification_plan_index,
        )

        assert artifact.qualification_plan_index is not None
        qualification = load_formal_single_operator_preflight_qualification_plan_index(
            artifact.qualification_plan_index.absolute_path
        )
        if (
            qualification.protocol_lock_sha256 != protocol_lock.sha256
            or qualification.exactness_cell_id != exactness.cell_id
        ):
            raise ValueError("trusted preflight qualification plan lineage changed")
    if artifact_binding.semantic_sha256 != artifact.sha256:
        raise ValueError("single-operator preflight input aggregate changed")
    return artifact


def _completion_context(
    execution_inputs_path: str | Path,
    *,
    current_ns: int,
) -> tuple[
    CanonicalJsonProofBinding,
    FormalPreflightExecutionInputs,
    FormalSingleOperatorPreflightAuthority,
    ProtocolLock,
    dict[str, FormalPreflightExecutionBinding],
    FormalPreflightInterferenceExecutionManifest,
]:
    inputs_binding = CanonicalJsonProofBinding.bind(execution_inputs_path)
    inputs = revalidate_formal_single_operator_preflight_execution_inputs(
        execution_inputs_path,
        current_ns=current_ns,
    )
    authority = FormalSingleOperatorPreflightAuthority.from_dict(
        inputs.execution_authority.reopen()
    )
    protocol_lock = protocol_lock_from_dict(authority.protocol_lock.reopen())
    bindings = {row.registry_cell_id: row for row in authority.execution_bindings}
    manifest = FormalPreflightInterferenceExecutionManifest.from_dict(
        inputs.interference_manifest.reopen()
    )
    if (
        inputs_binding.semantic_sha256 != inputs.sha256
        or inputs.execution_authority.semantic_sha256 != authority.sha256
        or len(bindings) != 10
        or manifest.dispatch_receipt_semantic_sha256
        != inputs.execution_authority.semantic_sha256
    ):
        raise ValueError("single-operator preflight completion input identity differs")
    return inputs_binding, inputs, authority, protocol_lock, bindings, manifest


def _single_operator_exactness_marker(
    *,
    execution_authority_sha256: str,
    assignment_sha256: str,
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_exactness_execution_marker",
            "execution_authority_sha256": execution_authority_sha256,
            "assignment_sha256": assignment_sha256,
        }
    )


def _single_operator_compile_worker(
    plan: CompileAssignmentPlan,
) -> CompileWorkerSourceDescriptor:
    _assignment, _cache, _prewarm, launch = plan.revalidate()
    return CompileWorkerSourceDescriptor.issue(
        patched_sglang_checkout=launch.patched_sglang_checkout,
    )


def execute_formal_single_operator_preflight_compile(
    execution_inputs_path: str | Path,
    *,
    current_ns: int,
) -> CompileResultPointer:
    """Execute the one source-owned compile row from trusted exact-ten inputs.

    No caller command, timeout, GPU, port, or cache identity is accepted.  All
    launch values are rebuilt from the current execution source before the
    child process is created.  The raw pointer remains schema 2; the trusted
    completion joins it back to the current-source authority.
    """

    (
        _inputs_binding,
        inputs,
        _authority,
        _protocol_lock,
        bindings,
        _manifest,
    ) = _completion_context(execution_inputs_path, current_ns=current_ns)
    binding = next(
        row for row in bindings.values() if row.runner_kind == "first_party_compile"
    )
    plan = CompileAssignmentPlan.load(inputs.compile_assignment_plan.absolute_path)
    worker = _single_operator_compile_worker(plan)
    pointer = _execute_compile_assignment_subprocess_path(
        inputs.compile_assignment_plan.absolute_path,
        argv=(worker.interpreter_path, worker.helper_path),
        timeout_seconds=(
            _SINGLE_OPERATOR_PROCESS_TIMEOUT_NS["first_party_compile"] / 1_000_000_000
        ),
        source_authority_sha256=None,
        control_verification_receipt_sha256=None,
        control_verification_receipt_path=None,
        formal_execution_authorized=False,
    )
    _validate_single_operator_compile_result(
        inputs=inputs,
        binding=binding,
        plan=plan,
        pointer=pointer,
        worker=worker,
    )
    return pointer


def _validate_single_operator_compile_result(
    *,
    inputs: FormalPreflightExecutionInputs,
    binding: FormalPreflightExecutionBinding,
    plan: CompileAssignmentPlan,
    pointer: CompileResultPointer,
    worker: CompileWorkerSourceDescriptor | None = None,
) -> CompileSubprocessLifecycleReceipt:
    source = worker or _single_operator_compile_worker(plan)
    if (
        binding.runner_kind != "first_party_compile"
        or pointer.schema_version != 2
        or pointer.formal_execution_authorized is not False
        or pointer.assignment_plan_source is None
        or pointer.subprocess_lifecycle_receipt is None
        or pointer.control_verification_receipt is not None
        or pointer.assignment_plan_sha256 != plan.sha256
        or pointer.assignment_plan_source.absolute_path
        != inputs.compile_assignment_plan.absolute_path
        or pointer.assignment_plan_source.raw_sha256
        != inputs.compile_assignment_plan.raw_sha256
        or pointer.assignment_plan_source.size != inputs.compile_assignment_plan.size
    ):
        raise ValueError("single-operator compile result differs from exact input")
    receipt = CompileSubprocessLifecycleReceipt.load(
        pointer.subprocess_lifecycle_receipt.absolute_path
    )
    expected_argv_sha256 = content_sha256(
        {"argv": [source.interpreter_path, source.helper_path]}
    )
    if (
        receipt.assignment_plan_sha256 != plan.sha256
        or receipt.formal_execution_authorized is not False
        or receipt.source_authority_sha256 is not None
        or receipt.control_verification_receipt_sha256 is not None
        or receipt.executable_path != source.interpreter_path
        or receipt.executable_raw_sha256 != source.interpreter_raw_sha256
        or receipt.executable_size != source.interpreter_size
        or receipt.argv_sha256 != expected_argv_sha256
        or receipt.launch_manifest_path != plan.launch_manifest_path
        or receipt.launch_manifest_raw_sha256 != plan.launch_manifest_raw_sha256
        or receipt.launch_manifest_sha256 != plan.launch_manifest_sha256
    ):
        raise ValueError("single-operator compile child source differs")
    return receipt


def execute_formal_single_operator_preflight_exactness(
    execution_inputs_path: str | Path,
    *,
    current_ns: int,
) -> ExactnessPreflightResultPointer:
    """Execute the one exactness row with a source-derived command and cap."""

    (
        _inputs_binding,
        inputs,
        authority,
        _protocol_lock,
        bindings,
        _manifest,
    ) = _completion_context(execution_inputs_path, current_ns=current_ns)
    binding = next(
        row for row in bindings.values() if row.runner_kind == "first_party_exactness"
    )
    assignment = ExactnessPreflightAssignment.load(
        inputs.exactness_assignment.absolute_path
    )
    pointer = _execute_exactness_preflight(
        inputs.exactness_assignment.absolute_path,
        dispatch_attestation=None,
        replay_store=None,
        now_ns=current_ns,
        timeout_seconds=(
            _SINGLE_OPERATOR_PROCESS_TIMEOUT_NS["first_party_exactness"] / 1_000_000_000
        ),
        single_operator_authority_sha256=authority.sha256,
    )
    _validate_single_operator_exactness_result(
        inputs=inputs,
        authority=authority,
        binding=binding,
        assignment=assignment,
        pointer=pointer,
    )
    return pointer


def _validate_single_operator_exactness_result(
    *,
    inputs: FormalPreflightExecutionInputs,
    authority: FormalSingleOperatorPreflightAuthority,
    binding: FormalPreflightExecutionBinding,
    assignment: ExactnessPreflightAssignment,
    pointer: ExactnessPreflightResultPointer,
) -> ExactnessPreflightTerminal:
    terminal = ExactnessPreflightTerminal.load(pointer.terminal.absolute_path)
    if (
        binding.runner_kind != "first_party_exactness"
        or pointer.schema_version != 2
        or pointer.control_verification_receipt is not None
        or pointer.qualification_proof_artifact is not None
        or pointer.assignment.absolute_path != inputs.exactness_assignment.absolute_path
        or pointer.assignment.raw_sha256 != inputs.exactness_assignment.raw_sha256
        or pointer.assignment.size != inputs.exactness_assignment.size
        or terminal.schema_version != 3
        or terminal.authority_mode != "formal_single_operator_v1"
        or terminal.assignment_sha256 != assignment.sha256
        or terminal.dispatch_attestation_sha256 != authority.sha256
        or terminal.replay_reservation_sha256
        != _single_operator_exactness_marker(
            execution_authority_sha256=authority.sha256,
            assignment_sha256=assignment.sha256,
        )
    ):
        raise ValueError("single-operator exactness result differs from exact input")
    return terminal


def _publish_single_operator_serving_junit(
    path: Path,
    *,
    request_ids: tuple[str, ...],
) -> EvidenceFileBinding:
    if not request_ids or request_ids != tuple(dict.fromkeys(request_ids)):
        raise ValueError("single-operator serving JUnit request coverage differs")
    suite = ElementTree.Element(
        "testsuite",
        {
            "name": "lightcone-formal-serving",
            "tests": str(len(request_ids)),
            "failures": "0",
            "errors": "0",
            "skipped": "0",
        },
    )
    for request_id in request_ids:
        ElementTree.SubElement(
            suite,
            "testcase",
            {"classname": "lightcone.tp1_dp1", "name": request_id},
        )
    body = (
        ElementTree.tostring(
            suite,
            encoding="utf-8",
            xml_declaration=True,
        )
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _validate_completion_junit(path, expected_request_ids=request_ids)


async def execute_formal_single_operator_preflight_interference(
    execution_inputs_path: str | Path,
    *,
    current_ns: int,
) -> CanonicalJsonProofBinding:
    """Run source-owned 4 isolated + 2 paired waves from trusted exact-ten inputs."""

    (
        inputs_binding,
        inputs,
        authority,
        protocol_lock,
        bindings,
        manifest,
    ) = _completion_context(execution_inputs_path, current_ns=current_ns)
    source = load_formal_single_operator_execution_source(
        authority.execution_source.absolute_path
    )
    inventory = GpuInventory.from_dict(authority.inventory.reopen())
    token = _trusted_token(authority=authority, source=source, inventory=inventory)
    admission = _InterferenceExecutionAdmission(
        authority_mode="formal_single_operator_v1",
        execution_authority_sha256=authority.sha256,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=token.dispatch_context.registry.sha256,
        activation_sha256=token.dispatch_context.activation_artifact.sha256,
        runtime_sha256=token.dispatch_context.activation_artifact.runtime_sha256,
        split_sha256=token.dispatch_context.activation_artifact.split_sha256,
        inventory_sha256=inventory.sha256,
        budget_plan_sha256=authority.budget_plan_sha256,
        execution_plan_sha256=token.dispatch_plan.sha256,
        process_hard_timeout_ns=_SINGLE_OPERATOR_PROCESS_TIMEOUT_NS[
            "first_party_interference"
        ],
        bindings=tuple(
            sorted(
                (
                    row
                    for row in authority.execution_bindings
                    if row.runner_kind == "first_party_interference"
                ),
                key=lambda row: row.registry_cell_id,
            )
        ),
    )
    exactness = ExactnessPreflightAssignment.load(
        inputs.exactness_assignment.absolute_path
    )
    tool = PinnedNvidiaSmiTool.bind(exactness.nvidia_smi_executable)
    root = _private_new_root(
        Path(inputs_binding.absolute_path).parent
        / "formal-single-operator-interference-execution"
    )
    execution_by_id = {row.registry_cell_id: row for row in manifest.inputs}
    expected_ids = {
        row.registry_cell_id
        for row in bindings.values()
        if row.runner_kind == "first_party_interference"
    }
    if set(execution_by_id) != expected_ids:
        raise ValueError("single-operator interference manifest is not exact eight")
    (
        raw_binding,
        raw_rows,
        consumptions,
    ) = await _execute_formal_preflight_interference_raw_core(
        admission,
        launch_cap_schedule_path=None,
        execution_inputs=execution_by_id,
        nvidia_smi_tool=tool,
        evidence_root=root,
        now_ns=current_ns,
    )
    if consumptions:
        raise RuntimeError("trusted interference unexpectedly consumed sealed control")
    batch = FormalPreflightInterferenceRawBatch.from_dict(raw_binding.reopen())
    batch.revalidate()
    from lightcone_spec.orchestration.formal_terminal_result import (
        publish_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact,
        validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact,
    )
    from lightcone_spec.orchestration.live_sglang import (
        validate_unsigned_pinned_sglang_lifecycle_timing_receipt,
    )

    evidence: list[FormalSingleOperatorPreflightInterferenceEvidence] = []
    promotable_rows = (
        sorted(raw_rows, key=lambda row: row.registry_cell_id)
        if batch.status == "WAITING_FOR_LOCAL_CONTROL"
        else ()
    )
    for raw_row in promotable_rows:
        if (
            raw_row.status != "WAITING_FOR_LOCAL_CONTROL"
            or raw_row.raw_terminal is None
            or raw_row.live_run_receipt is None
        ):
            continue
        manifest_row = execution_by_id[raw_row.registry_cell_id]
        directory = root / raw_row.registry_cell_id
        lifecycle_path = directory / "lifecycle.json"
        lifecycle_binding = CanonicalJsonProofBinding.bind(lifecycle_path)
        terminal_proof = (
            publish_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
                execution_manifest_path=inputs.interference_manifest.absolute_path,
                interference_raw_batch_path=raw_binding.absolute_path,
                raw_terminal_path=raw_row.raw_terminal.absolute_path,
                materialized_cell_id=raw_row.materialized_cell_id,
                registry_cell_id=raw_row.registry_cell_id,
                expected_inventory_sha256=inventory.sha256,
                expected_registry_sha256=protocol_lock.registry_sha256,
                expected_root_manifest_sha256=(
                    protocol_lock.offline_release_trust_root_sha256
                ),
                proof_artifact_path=str(
                    directory / "formal-single-operator-terminal-proof.json"
                ),
            )
        )
        projection = (
            validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
                terminal_proof.absolute_path,
                expected_inventory_sha256=inventory.sha256,
                expected_registry_sha256=protocol_lock.registry_sha256,
                expected_root_manifest_sha256=(
                    protocol_lock.offline_release_trust_root_sha256
                ),
                expected_execution_plan_sha256=(
                    manifest_row.run_binding.execution_plan_sha256
                ),
                expected_rank_config_sha256=(
                    manifest_row.run_binding.rank_config_sha256
                ),
                expected_run_id=manifest_row.run_binding.run_id,
                expected_run_nonce_sha256=(manifest_row.run_binding.run_nonce_sha256),
                expected_attempt_id=manifest_row.run_binding.attempt_id,
                expected_method=manifest_row.run_binding.method,
                now_ns=current_ns,
            )
        )
        assert raw_row.live_run_receipt is not None
        launch = CompileLaunchManifest.load(manifest_row.launch_manifest_path)
        config = load_run_config(launch.run_config_path)
        validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
            lifecycle_binding,
            expected_live_run_receipt=raw_row.live_run_receipt,
            expected_binding=manifest_row.run_binding,
            expected_telemetry_detail=config.runtime.telemetry_detail,
        )
        scored_ids = manifest_row.run_binding.scored_request_ids
        request_results = {
            row.request_id: row
            for row in projection.requests
            if row.request_id in set(scored_ids)
        }
        if (
            projection.scored_request_ids != scored_ids
            or set(request_results) != set(scored_ids)
            or any(
                not row.submitted_to_server
                or row.terminal_status != "completed"
                or row.output_token_ids is None
                for row in request_results.values()
            )
        ):
            continue
        junit = _publish_single_operator_serving_junit(
            directory / "junit.xml",
            request_ids=scored_ids,
        )
        evidence.append(
            FormalSingleOperatorPreflightInterferenceEvidence(
                materialized_cell_id=raw_row.materialized_cell_id,
                registry_cell_id=raw_row.registry_cell_id,
                terminal_result_proof=terminal_proof,
                lifecycle_timing=lifecycle_binding,
                junit_xml=junit,
            )
        )
    artifact = FormalSingleOperatorPreflightInterferenceExecution(
        schema_version=1,
        kind="formal_single_operator_preflight_interference_execution",
        execution_inputs=inputs_binding,
        raw_batch=raw_binding,
        evidence=tuple(sorted(evidence, key=lambda row: row.registry_cell_id)),
        status=(
            "WAITING_FOR_COMPLETION"
            if batch.status == "WAITING_FOR_LOCAL_CONTROL" and len(evidence) == 8
            else "ERROR"
        ),
    )
    output = root / "formal-single-operator-interference-execution.json"
    publish_canonical_json_no_replace(output, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)
    if (
        FormalSingleOperatorPreflightInterferenceExecution.from_dict(binding.reopen())
        != artifact
    ):
        raise RuntimeError("single-operator interference execution changed")
    return binding


def revalidate_formal_single_operator_preflight_interference_execution(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalSingleOperatorPreflightInterferenceExecution:
    """Deep-reopen one trusted exact-eight execution and all promoted evidence."""

    binding = CanonicalJsonProofBinding.bind(path)
    artifact = FormalSingleOperatorPreflightInterferenceExecution.from_dict(
        binding.reopen()
    )
    (
        inputs_binding,
        _inputs,
        authority,
        protocol_lock,
        bindings,
        manifest,
    ) = _completion_context(
        artifact.execution_inputs.absolute_path,
        current_ns=current_ns,
    )
    raw_binding = CanonicalJsonProofBinding.bind(artifact.raw_batch.absolute_path)
    batch = FormalPreflightInterferenceRawBatch.from_dict(raw_binding.reopen())
    batch.revalidate()
    source = load_formal_single_operator_execution_source(
        authority.execution_source.absolute_path
    )
    trusted_token = _trusted_token(
        authority=authority,
        source=source,
        inventory=GpuInventory.from_dict(authority.inventory.reopen()),
    )
    activation = trusted_token.dispatch_context.activation_artifact
    if (
        binding.semantic_sha256 != artifact.sha256
        or artifact.execution_inputs != inputs_binding
        or artifact.raw_batch != raw_binding
        or raw_binding.semantic_sha256 != batch.sha256
        or batch.dispatch_sha256 != authority.sha256
        or batch.registry_sha256 != protocol_lock.registry_sha256
        or batch.activation_sha256 != activation.sha256
        or batch.runtime_sha256 != activation.runtime_sha256
        or batch.split_sha256 != activation.split_sha256
        or batch.inventory_sha256 != authority.inventory.semantic_sha256
    ):
        raise ValueError("single-operator interference execution identity differs")
    if batch.status == "ERROR":
        if artifact.status != "ERROR" or artifact.evidence:
            raise ValueError("failed interference execution promoted partial evidence")
        return artifact
    if artifact.status != "WAITING_FOR_COMPLETION":
        raise ValueError("successful interference execution status differs")

    from lightcone_spec.orchestration.formal_terminal_result import (
        FormalSingleOperatorPreflightTp1RawTerminalProofArtifact,
        validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact,
    )
    from lightcone_spec.orchestration.live_sglang import (
        UnsignedPinnedSglangLifecycleTimingReceipt,
        validate_unsigned_pinned_sglang_lifecycle_timing_receipt,
    )

    manifest_by_id = {row.registry_cell_id: row for row in manifest.inputs}
    raw_by_id = {row.registry_cell_id: row for row in batch.rows}
    expected_ids = {
        row.registry_cell_id
        for row in bindings.values()
        if row.runner_kind == "first_party_interference"
    }
    evidence_by_id = {row.registry_cell_id: row for row in artifact.evidence}
    if (
        set(manifest_by_id) != expected_ids
        or set(raw_by_id) != expected_ids
        or set(evidence_by_id) != expected_ids
    ):
        raise ValueError("single-operator interference evidence is not exact eight")
    for registry_cell_id in sorted(expected_ids):
        expected = bindings[registry_cell_id]
        manifest_row = manifest_by_id[registry_cell_id]
        raw_row = raw_by_id[registry_cell_id]
        evidence = evidence_by_id[registry_cell_id]
        terminal_binding = CanonicalJsonProofBinding.bind(
            evidence.terminal_result_proof.absolute_path
        )
        terminal_artifact = (
            FormalSingleOperatorPreflightTp1RawTerminalProofArtifact.from_dict(
                terminal_binding.reopen()
            )
        )
        projection = (
            validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
                terminal_binding.absolute_path,
                expected_inventory_sha256=authority.inventory.semantic_sha256,
                expected_registry_sha256=protocol_lock.registry_sha256,
                expected_root_manifest_sha256=(
                    protocol_lock.offline_release_trust_root_sha256
                ),
                expected_execution_plan_sha256=(
                    manifest_row.run_binding.execution_plan_sha256
                ),
                expected_rank_config_sha256=(
                    manifest_row.run_binding.rank_config_sha256
                ),
                expected_run_id=manifest_row.run_binding.run_id,
                expected_run_nonce_sha256=(manifest_row.run_binding.run_nonce_sha256),
                expected_attempt_id=manifest_row.run_binding.attempt_id,
                expected_method=manifest_row.run_binding.method,
                now_ns=current_ns,
            )
        )
        if raw_row.live_run_receipt is None:
            raise ValueError("single-operator interference row lacks live receipt")
        lifecycle_binding = CanonicalJsonProofBinding.bind(
            evidence.lifecycle_timing.absolute_path
        )
        lifecycle = UnsignedPinnedSglangLifecycleTimingReceipt.from_dict(
            lifecycle_binding.reopen()
        )
        launch = CompileLaunchManifest.load(manifest_row.launch_manifest_path)
        config = load_run_config(launch.run_config_path)
        validated_lifecycle = validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
            lifecycle_binding,
            expected_live_run_receipt=raw_row.live_run_receipt,
            expected_binding=manifest_row.run_binding,
            expected_telemetry_detail=config.runtime.telemetry_detail,
        )
        scored_ids = manifest_row.run_binding.scored_request_ids
        requests = {
            row.request_id: row
            for row in projection.requests
            if row.request_id in set(scored_ids)
        }
        junit = _validate_completion_junit(
            evidence.junit_xml.absolute_path,
            expected_request_ids=scored_ids,
        )
        if (
            evidence.materialized_cell_id != expected.materialized_cell_id
            or evidence.registry_cell_id != registry_cell_id
            or raw_row.materialized_cell_id != expected.materialized_cell_id
            or raw_row.status != "WAITING_FOR_LOCAL_CONTROL"
            or terminal_artifact.materialized_cell_id != expected.materialized_cell_id
            or terminal_artifact.registry_cell_id != registry_cell_id
            or terminal_binding != evidence.terminal_result_proof
            or lifecycle_binding != evidence.lifecycle_timing
            or lifecycle_binding.semantic_sha256 != lifecycle.sha256
            or validated_lifecycle != lifecycle
            or junit != evidence.junit_xml
            or projection.scored_request_ids != scored_ids
            or set(requests) != set(scored_ids)
            or any(
                not row.submitted_to_server
                or row.terminal_status != "completed"
                or row.output_token_ids is None
                for row in requests.values()
            )
        ):
            raise ValueError("single-operator interference promoted evidence differs")
    return artifact


def _validate_completion_junit(
    path: str | Path,
    *,
    expected_request_ids: tuple[str, ...],
) -> EvidenceFileBinding:
    source = Path(path)
    if not source.is_absolute() or source != source.resolve(strict=False):
        raise ValueError("single-operator preflight JUnit path is not canonical")
    binding = EvidenceFileBinding.bind(source, label="single-operator preflight JUnit")
    try:
        suite = ElementTree.parse(source).getroot()
    except (ElementTree.ParseError, OSError) as error:
        raise ValueError("single-operator preflight JUnit is not valid XML") from error
    cases = tuple(suite)
    if (
        suite.tag != "testsuite"
        or set(suite.attrib)
        != {
            "name",
            "tests",
            "failures",
            "errors",
            "skipped",
        }
        or suite.attrib["name"] != "lightcone-formal-serving"
        or suite.attrib["tests"] != str(len(expected_request_ids))
        or any(suite.attrib[name] != "0" for name in ("failures", "errors", "skipped"))
        or len(cases) != len(expected_request_ids)
        or any(
            case.tag != "testcase"
            or set(case.attrib) != {"classname", "name"}
            or case.attrib["classname"] != "lightcone.tp1_dp1"
            or tuple(case)
            for case in cases
        )
        or tuple(case.attrib["name"] for case in cases) != expected_request_ids
    ):
        raise ValueError("single-operator preflight JUnit coverage differs")
    return binding


def _derive_formal_single_operator_preflight_completion(
    *,
    execution_inputs_path: str | Path,
    compile_result_path: str | Path,
    exactness_result_path: str | Path,
    interference_terminal_result_proof_paths: tuple[str | Path, ...],
    interference_lifecycle_timing_paths: tuple[str | Path, ...],
    interference_junit_paths: tuple[str | Path, ...],
    current_ns: int,
) -> FormalSingleOperatorPreflightCompletion:
    if type(current_ns) is not int or current_ns < 1:
        raise ValueError("single-operator preflight completion time is invalid")
    if any(
        type(rows) is not tuple or len(rows) != 8
        for rows in (
            interference_terminal_result_proof_paths,
            interference_lifecycle_timing_paths,
            interference_junit_paths,
        )
    ):
        raise ValueError(
            "single-operator preflight completion paths are not exact eight"
        )
    (
        inputs_binding,
        inputs,
        authority,
        protocol_lock,
        bindings,
        manifest,
    ) = _completion_context(execution_inputs_path, current_ns=current_ns)

    compile_binding = next(
        row for row in bindings.values() if row.runner_kind == "first_party_compile"
    )
    compile_plan = CompileAssignmentPlan.load(
        inputs.compile_assignment_plan.absolute_path
    )
    compile_result_binding = CanonicalJsonProofBinding.bind(compile_result_path)
    compile_result = CompileResultPointer.load(compile_result_path)
    if compile_result_binding.semantic_sha256 != compile_result.sha256 or Path(
        compile_plan.result_pointer_path
    ) != Path(compile_result_path):
        raise ValueError("single-operator compile completion differs from exact input")
    compile_lifecycle = _validate_single_operator_compile_result(
        inputs=inputs,
        binding=compile_binding,
        plan=compile_plan,
        pointer=compile_result,
    )
    compile_row = FormalSingleOperatorPreflightCompletionRow(
        materialized_cell_id=compile_binding.materialized_cell_id,
        registry_cell_id=compile_binding.registry_cell_id,
        runner_kind="first_party_compile",
        status="COMPLETE",
        started_ns=compile_lifecycle.process_started_ns,
        finished_ns=compile_lifecycle.process_exited_ns,
        result_sha256=compile_result.sha256,
    )

    exact_binding = next(
        row for row in bindings.values() if row.runner_kind == "first_party_exactness"
    )
    exact_assignment = ExactnessPreflightAssignment.load(
        inputs.exactness_assignment.absolute_path
    )
    exact_result_binding = CanonicalJsonProofBinding.bind(exactness_result_path)
    exact_result = ExactnessPreflightResultPointer.load(exactness_result_path)
    exact_terminal = _validate_single_operator_exactness_result(
        inputs=inputs,
        authority=authority,
        binding=exact_binding,
        assignment=exact_assignment,
        pointer=exact_result,
    )
    if (
        exact_result_binding.semantic_sha256 != exact_result.sha256
        or exact_result.junit_xml is None
        or len(exact_result.rank_terminals) != 2
    ):
        raise ValueError(
            "single-operator exactness completion differs from exact input"
        )
    qualification_results = ()
    if inputs.schema_version == 4:
        from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
            load_formal_single_operator_preflight_qualification_plan,
            load_formal_single_operator_preflight_qualification_plan_index,
            revalidate_formal_single_operator_preflight_qualification_result,
        )

        assert inputs.qualification_plan_index is not None
        qualification_index = (
            load_formal_single_operator_preflight_qualification_plan_index(
                inputs.qualification_plan_index.absolute_path
            )
        )
        qualification_results = tuple(
            revalidate_formal_single_operator_preflight_qualification_result(
                load_formal_single_operator_preflight_qualification_plan(
                    plan.absolute_path
                ).result_path
            )
            for plan in qualification_index.plans
        )
    exact_row = FormalSingleOperatorPreflightCompletionRow(
        materialized_cell_id=exact_binding.materialized_cell_id,
        registry_cell_id=exact_binding.registry_cell_id,
        runner_kind="first_party_exactness",
        status="COMPLETE" if exact_terminal.status == "PASSED" else "FAILED",
        started_ns=min(
            (exact_terminal.started_ns,)
            + tuple(row.started_ns for row in qualification_results)
        ),
        finished_ns=max(
            (exact_terminal.finished_ns,)
            + tuple(row.finished_ns for row in qualification_results)
        ),
        result_sha256=(
            content_sha256(
                {
                    "exactness_result_sha256": exact_result.sha256,
                    "qualification_result_sha256s": [
                        row.sha256 for row in qualification_results
                    ],
                }
            )
            if qualification_results
            else exact_result.sha256
        ),
    )

    from lightcone_spec.experiments.preflight_interference import (
        FormalPreflightInterferenceRawBatch,
    )
    from lightcone_spec.orchestration.formal_terminal_result import (
        FormalCurrentPreflightTp1TerminalResultProofArtifact,
        FormalSingleOperatorPreflightTp1RawTerminalProofArtifact,
        validate_formal_current_preflight_tp1_terminal_result_proof_artifact,
        validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact,
    )
    from lightcone_spec.orchestration.live_sglang import (
        UnsignedPinnedSglangLifecycleTimingReceipt,
        validate_unsigned_pinned_sglang_lifecycle_timing_receipt,
    )

    interference_rows: list[FormalSingleOperatorPreflightCompletionRow] = []
    interference_evidence: list[FormalSingleOperatorPreflightInterferenceEvidence] = []
    for manifest_row, terminal_path, lifecycle_path, junit_path in zip(
        manifest.inputs,
        interference_terminal_result_proof_paths,
        interference_lifecycle_timing_paths,
        interference_junit_paths,
        strict=True,
    ):
        expected = bindings[manifest_row.registry_cell_id]
        terminal_binding = CanonicalJsonProofBinding.bind(terminal_path)
        terminal_value = terminal_binding.reopen()
        if (
            type(terminal_value) is dict
            and terminal_value.get("kind")
            == "formal_current_preflight_tp1_terminal_result_proof_artifact"
        ):
            terminal_artifact = (
                FormalCurrentPreflightTp1TerminalResultProofArtifact.from_dict(
                    terminal_value
                )
            )
            validator = (
                validate_formal_current_preflight_tp1_terminal_result_proof_artifact
            )
        elif (
            type(terminal_value) is dict
            and terminal_value.get("kind")
            == "formal_single_operator_preflight_tp1_raw_terminal_proof_artifact"
        ):
            terminal_artifact = (
                FormalSingleOperatorPreflightTp1RawTerminalProofArtifact.from_dict(
                    terminal_value
                )
            )
            validator = validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact
        else:
            raise ValueError(
                "single-operator preflight terminal proof kind is unsupported"
            )
        projection = validator(
            terminal_binding.absolute_path,
            expected_inventory_sha256=authority.inventory.semantic_sha256,
            expected_registry_sha256=protocol_lock.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            expected_execution_plan_sha256=(
                manifest_row.run_binding.execution_plan_sha256
            ),
            expected_rank_config_sha256=manifest_row.run_binding.rank_config_sha256,
            expected_run_id=manifest_row.run_binding.run_id,
            expected_run_nonce_sha256=manifest_row.run_binding.run_nonce_sha256,
            expected_attempt_id=manifest_row.run_binding.attempt_id,
            expected_method=manifest_row.run_binding.method,
            now_ns=current_ns,
        )
        raw_batch = FormalPreflightInterferenceRawBatch.from_dict(
            terminal_artifact.interference_raw_batch.reopen()
        )
        raw_batch.revalidate()
        raw_matches = tuple(
            row
            for row in raw_batch.rows
            if row.registry_cell_id == manifest_row.registry_cell_id
            and row.materialized_cell_id == expected.materialized_cell_id
        )
        if len(raw_matches) != 1 or raw_matches[0].live_run_receipt is None:
            raise ValueError(
                "single-operator preflight lifecycle lacks one exact raw run"
            )
        raw_row = raw_matches[0]
        lifecycle_binding = CanonicalJsonProofBinding.bind(lifecycle_path)
        lifecycle = UnsignedPinnedSglangLifecycleTimingReceipt.from_dict(
            lifecycle_binding.reopen()
        )
        launch = CompileLaunchManifest.load(manifest_row.launch_manifest_path)
        config = load_run_config(launch.run_config_path)
        validated_lifecycle = validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
            lifecycle_binding,
            expected_live_run_receipt=raw_row.live_run_receipt,
            expected_binding=manifest_row.run_binding,
            expected_telemetry_detail=config.runtime.telemetry_detail,
        )
        scored_ids = manifest_row.run_binding.scored_request_ids
        junit_binding = _validate_completion_junit(
            junit_path,
            expected_request_ids=scored_ids,
        )
        request_results = {
            row.request_id: row
            for row in projection.requests
            if row.request_id in set(scored_ids)
        }
        complete = (
            set(request_results) == set(scored_ids)
            and projection.scored_request_ids == scored_ids
            and all(
                row.submitted_to_server
                and row.terminal_status == "completed"
                and row.output_token_ids is not None
                for row in request_results.values()
            )
        )
        if (
            terminal_binding.semantic_sha256 != terminal_artifact.sha256
            or terminal_artifact.materialized_cell_id != expected.materialized_cell_id
            or terminal_artifact.registry_cell_id != expected.registry_cell_id
            or lifecycle_binding.semantic_sha256 != lifecycle.sha256
            or validated_lifecycle != lifecycle
        ):
            raise ValueError(
                "single-operator preflight terminal/lifecycle identity differs"
            )
        evidence = FormalSingleOperatorPreflightInterferenceEvidence(
            materialized_cell_id=expected.materialized_cell_id,
            registry_cell_id=expected.registry_cell_id,
            terminal_result_proof=terminal_binding,
            lifecycle_timing=lifecycle_binding,
            junit_xml=junit_binding,
        )
        interference_evidence.append(evidence)
        interference_rows.append(
            FormalSingleOperatorPreflightCompletionRow(
                materialized_cell_id=expected.materialized_cell_id,
                registry_cell_id=expected.registry_cell_id,
                runner_kind="first_party_interference",
                status="COMPLETE" if complete else "FAILED",
                started_ns=lifecycle.phase_edges_ns["execution_started_ns"],
                finished_ns=lifecycle.phase_edges_ns["evidence_flush_finished_ns"],
                result_sha256=content_sha256(
                    {
                        "terminal_result_proof_sha256": terminal_binding.semantic_sha256,
                        "lifecycle_timing_sha256": lifecycle_binding.semantic_sha256,
                        "junit_xml_raw_sha256": junit_binding.raw_sha256,
                    }
                ),
            )
        )
    rows = tuple(
        sorted(
            (compile_row, exact_row, *interference_rows),
            key=lambda row: row.registry_cell_id,
        )
    )
    return FormalSingleOperatorPreflightCompletion(
        schema_version=2 if inputs.schema_version == 4 else 1,
        kind="formal_single_operator_exact_ten_preflight_completion",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256
            if inputs.schema_version == 4
            else FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256
        ),
        execution_inputs=inputs_binding,
        compile_result=compile_result_binding,
        exactness_result=exact_result_binding,
        interference_evidence=tuple(
            sorted(interference_evidence, key=lambda row: row.registry_cell_id)
        ),
        rows=rows,
        status=(
            "COMPLETE" if all(row.status == "COMPLETE" for row in rows) else "FAILED"
        ),
        started_ns=min(row.started_ns for row in rows),
        finished_ns=max(row.finished_ns for row in rows),
    )


def publish_formal_single_operator_preflight_completion(
    *,
    execution_inputs_path: str | Path,
    compile_result_path: str | Path,
    exactness_result_path: str | Path,
    interference_terminal_result_proof_paths: tuple[str | Path, ...],
    interference_lifecycle_timing_paths: tuple[str | Path, ...],
    interference_junit_paths: tuple[str | Path, ...],
    output_path: str | Path,
    current_ns: int,
) -> CanonicalJsonProofBinding:
    """Publish one exact-ten summary derived solely from actual runner evidence."""

    destination = Path(output_path)
    if not destination.is_absolute() or destination != destination.resolve(
        strict=False
    ):
        raise ValueError("single-operator preflight completion output is not canonical")
    artifact = _derive_formal_single_operator_preflight_completion(
        execution_inputs_path=execution_inputs_path,
        compile_result_path=compile_result_path,
        exactness_result_path=exactness_result_path,
        interference_terminal_result_proof_paths=(
            interference_terminal_result_proof_paths
        ),
        interference_lifecycle_timing_paths=interference_lifecycle_timing_paths,
        interference_junit_paths=interference_junit_paths,
        current_ns=current_ns,
    )
    publish_canonical_json_no_replace(destination, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(
        destination, semantic_sha256=artifact.sha256
    )
    if (
        revalidate_formal_single_operator_preflight_completion(
            destination,
            current_ns=current_ns,
        )
        != artifact
    ):
        raise RuntimeError(
            "single-operator preflight completion changed on publication"
        )
    return binding


def revalidate_formal_single_operator_preflight_completion(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalSingleOperatorPreflightCompletion:
    """Deep-reopen every actual result and byte-compare its derived summary."""

    binding = CanonicalJsonProofBinding.bind(path)
    artifact = FormalSingleOperatorPreflightCompletion.from_dict(binding.reopen())
    expected = _derive_formal_single_operator_preflight_completion(
        execution_inputs_path=artifact.execution_inputs.absolute_path,
        compile_result_path=artifact.compile_result.absolute_path,
        exactness_result_path=artifact.exactness_result.absolute_path,
        interference_terminal_result_proof_paths=tuple(
            row.terminal_result_proof.absolute_path
            for row in artifact.interference_evidence
        ),
        interference_lifecycle_timing_paths=tuple(
            row.lifecycle_timing.absolute_path for row in artifact.interference_evidence
        ),
        interference_junit_paths=tuple(
            row.junit_xml.absolute_path for row in artifact.interference_evidence
        ),
        current_ns=current_ns,
    )
    if binding.semantic_sha256 != artifact.sha256 or artifact != expected:
        raise ValueError("single-operator preflight completion evidence changed")
    return artifact


async def execute_formal_single_operator_preflight_exact_ten(
    execution_inputs_path: str | Path,
    *,
    current_ns: int,
) -> CanonicalJsonProofBinding:
    """Execute and durably summarize the callback-free current exact-ten path."""

    inputs_binding = CanonicalJsonProofBinding.bind(execution_inputs_path)
    inputs = revalidate_formal_single_operator_preflight_execution_inputs(
        inputs_binding.absolute_path,
        current_ns=current_ns,
    )
    compile_pointer = execute_formal_single_operator_preflight_compile(
        inputs_binding.absolute_path,
        current_ns=current_ns,
    )
    compile_plan = CompileAssignmentPlan.load(
        inputs.compile_assignment_plan.absolute_path
    )
    compile_result = CanonicalJsonProofBinding.bind(compile_plan.result_pointer_path)
    if compile_result.semantic_sha256 != compile_pointer.sha256:
        raise RuntimeError("single-operator compile result changed after execution")
    exact_pointer = execute_formal_single_operator_preflight_exactness(
        inputs_binding.absolute_path,
        current_ns=current_ns,
    )
    exact_assignment = ExactnessPreflightAssignment.load(
        inputs.exactness_assignment.absolute_path
    )
    exactness_result = CanonicalJsonProofBinding.bind(
        exact_assignment.result_pointer_path
    )
    if exactness_result.semantic_sha256 != exact_pointer.sha256:
        raise RuntimeError("single-operator exactness result changed after execution")
    interference_binding = await execute_formal_single_operator_preflight_interference(
        inputs_binding.absolute_path,
        current_ns=current_ns,
    )
    interference = revalidate_formal_single_operator_preflight_interference_execution(
        interference_binding.absolute_path,
        current_ns=current_ns,
    )
    # Keep the registered interference calibration cold with respect to the
    # auxiliary backend/topology qualification suites.  Those suites remain
    # evidence bound to the one logical exactness/memory/telemetry cell, but
    # execute only after all eight interference rows have reached a durable
    # terminal boundary.
    if inputs.schema_version == 4:
        from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
            execute_formal_single_operator_preflight_qualification_plan_index,
        )

        assert inputs.qualification_plan_index is not None
        qualification_results = (
            execute_formal_single_operator_preflight_qualification_plan_index(
                inputs.qualification_plan_index.absolute_path
            )
        )
        if len(qualification_results) != 6:
            raise RuntimeError(
                "single-operator exactness qualification fanout is not exact six"
            )
    completion_binding: CanonicalJsonProofBinding | None = None
    status: Literal["COMPLETE", "FAILED", "ERROR"] = "ERROR"
    if interference.status == "WAITING_FOR_COMPLETION":
        completion_binding = publish_formal_single_operator_preflight_completion(
            execution_inputs_path=inputs_binding.absolute_path,
            compile_result_path=compile_result.absolute_path,
            exactness_result_path=exactness_result.absolute_path,
            interference_terminal_result_proof_paths=tuple(
                row.terminal_result_proof.absolute_path for row in interference.evidence
            ),
            interference_lifecycle_timing_paths=tuple(
                row.lifecycle_timing.absolute_path for row in interference.evidence
            ),
            interference_junit_paths=tuple(
                row.junit_xml.absolute_path for row in interference.evidence
            ),
            output_path=(
                Path(inputs_binding.absolute_path).parent
                / "formal-single-operator-preflight-completion.json"
            ),
            current_ns=current_ns,
        )
        status = revalidate_formal_single_operator_preflight_completion(
            completion_binding.absolute_path,
            current_ns=current_ns,
        ).status
    artifact = FormalSingleOperatorPreflightExecution(
        schema_version=1,
        kind="formal_single_operator_exact_ten_preflight_execution",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_PREFLIGHT_EXECUTION_PROTOCOL_SHA256),
        execution_inputs=inputs_binding,
        compile_result=compile_result,
        exactness_result=exactness_result,
        interference_execution=interference_binding,
        completion=completion_binding,
        status=status,
    )
    output = (
        Path(inputs_binding.absolute_path).parent
        / "formal-single-operator-preflight-execution.json"
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    result = CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)
    if (
        revalidate_formal_single_operator_preflight_exact_ten_execution(
            result.absolute_path,
            current_ns=current_ns,
        )
        != artifact
    ):
        raise RuntimeError("single-operator exact-ten execution changed")
    return result


def revalidate_formal_single_operator_preflight_exact_ten_execution(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalSingleOperatorPreflightExecution:
    """Deep-reopen all actual outputs of one trusted exact-ten execution."""

    binding = CanonicalJsonProofBinding.bind(path)
    artifact = FormalSingleOperatorPreflightExecution.from_dict(binding.reopen())
    (
        inputs_binding,
        inputs,
        authority,
        _protocol_lock,
        bindings,
        _manifest,
    ) = _completion_context(
        artifact.execution_inputs.absolute_path,
        current_ns=current_ns,
    )
    compile_plan = CompileAssignmentPlan.load(
        inputs.compile_assignment_plan.absolute_path
    )
    compile_pointer_binding = CanonicalJsonProofBinding.bind(
        artifact.compile_result.absolute_path
    )
    compile_pointer = CompileResultPointer.load(compile_pointer_binding.absolute_path)
    compile_binding = next(
        row for row in bindings.values() if row.runner_kind == "first_party_compile"
    )
    _validate_single_operator_compile_result(
        inputs=inputs,
        binding=compile_binding,
        plan=compile_plan,
        pointer=compile_pointer,
    )
    exact_assignment = ExactnessPreflightAssignment.load(
        inputs.exactness_assignment.absolute_path
    )
    exact_pointer_binding = CanonicalJsonProofBinding.bind(
        artifact.exactness_result.absolute_path
    )
    exact_pointer = ExactnessPreflightResultPointer.load(
        exact_pointer_binding.absolute_path
    )
    exact_binding = next(
        row for row in bindings.values() if row.runner_kind == "first_party_exactness"
    )
    _validate_single_operator_exactness_result(
        inputs=inputs,
        authority=authority,
        binding=exact_binding,
        assignment=exact_assignment,
        pointer=exact_pointer,
    )
    interference_binding = CanonicalJsonProofBinding.bind(
        artifact.interference_execution.absolute_path
    )
    interference = revalidate_formal_single_operator_preflight_interference_execution(
        interference_binding.absolute_path,
        current_ns=current_ns,
    )
    if interference.status == "ERROR":
        if artifact.completion is not None or artifact.status != "ERROR":
            raise ValueError("failed exact-ten execution gained completion")
    else:
        if artifact.completion is None:
            raise ValueError("successful exact-ten execution lacks completion")
        completion_binding = CanonicalJsonProofBinding.bind(
            artifact.completion.absolute_path
        )
        completion = revalidate_formal_single_operator_preflight_completion(
            completion_binding.absolute_path,
            current_ns=current_ns,
        )
        if (
            completion_binding != artifact.completion
            or completion.execution_inputs != inputs_binding
            or completion.compile_result != compile_pointer_binding
            or completion.exactness_result != exact_pointer_binding
            or completion.interference_evidence != interference.evidence
            or artifact.status != completion.status
        ):
            raise ValueError("exact-ten execution completion lineage differs")
    if (
        binding.semantic_sha256 != artifact.sha256
        or artifact.execution_inputs != inputs_binding
        or artifact.compile_result != compile_pointer_binding
        or artifact.exactness_result != exact_pointer_binding
        or artifact.interference_execution != interference_binding
    ):
        raise ValueError("single-operator exact-ten execution identity differs")
    return artifact


__all__ = [
    "FORMAL_PREFLIGHT_INPUTS_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_PREFLIGHT_EXECUTION_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_QUALIFIED_PREFLIGHT_INPUTS_PROTOCOL_SHA256",
    "FormalPreflightExecutionInputs",
    "FormalSingleOperatorPreflightAuthority",
    "FormalSingleOperatorPreflightCompletion",
    "FormalSingleOperatorPreflightCompletionRow",
    "FormalSingleOperatorPreflightExecution",
    "FormalSingleOperatorPreflightInterferenceEvidence",
    "FormalSingleOperatorPreflightInterferenceExecution",
    "execute_formal_single_operator_preflight_compile",
    "execute_formal_single_operator_preflight_exact_ten",
    "execute_formal_single_operator_preflight_exactness",
    "execute_formal_single_operator_preflight_interference",
    "load_formal_preflight_execution_inputs",
    "materialize_formal_preflight_execution_inputs",
    "materialize_formal_single_operator_preflight_execution_inputs",
    "publish_formal_single_operator_preflight_completion",
    "revalidate_formal_single_operator_preflight_completion",
    "revalidate_formal_single_operator_preflight_exact_ten_execution",
    "revalidate_formal_single_operator_preflight_execution_inputs",
    "revalidate_formal_single_operator_preflight_interference_execution",
]
