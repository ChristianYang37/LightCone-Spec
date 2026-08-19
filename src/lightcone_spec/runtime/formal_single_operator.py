"""Minimal provenance for trusted single-operator formal experiments.

This mode is deliberately not an adversarial attestation protocol.  It is for
one trusted operator in a controlled, clean local checkout.  Scientific inputs
still come from the existing materialization, run-plan, compile-launch,
inventory, and request-schedule types; this module only records their exact
bytes and the actual run outputs in one canonical, no-replace manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FormalServingRequestScheduleReceipt,
    FormalServingRunPlan,
    formal_serving_request_schedule_rows,
)
from lightcone_spec.runtime.compile_cache import CompileCacheLaunchPlan
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SINGLE_OPERATOR_MODE = "formal_single_operator_v1"
TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_MODE = "formal_single_operator_v2"
FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS = (
    "one_trusted_operator",
    "controlled_local_workspace",
    "clean_git_checkout",
    "run_specific_no_replace_outputs",
    "canonical_sha256_provenance",
)


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


FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema": FORMAL_SINGLE_OPERATOR_MODE,
        "trust_assumptions": FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        "inputs": (
            "clean_git_head_and_tree",
            "registered_materialized_cell",
            "source_owned_run_config_launch_argv_and_request_schedule",
            "prepared_model_tokenizer_and_workload_content_sha256",
            "verified_gpu_inventory_assignment",
        ),
        "outputs": (
            "run_specific_directory_outside_git_checkout",
            "atomic_no_replace_canonical_manifest",
            "path_size_and_raw_sha256_for_every_observed_artifact",
        ),
        "physical_outcomes": (
            "tp1_unsigned_pinned_sglang_complete",
            "tp2_or_dp2_unsigned_formal_gang_complete",
            "tp1_source_owned_fatal_pointer_failed",
            "distributed_and_e5_failed_outcomes_not_yet_available",
        ),
        "manifest_fields": (
            "git_head_tree_and_sglang_patch_identity",
            "registry_protocol_materialization_and_inventory_identity",
            "run_plan_launch_execution_binding_and_subject_identity",
            "single_operator_direct_local_execution",
            "full_run_config_launch_argv_port_and_request_schedule",
            "model_drafter_tokenizer_and_workload_content_identity",
            "stage_cell_role_backend_topology_block_and_attempt",
            "gpu_uuid_model_driver_and_cuda",
            "run_directory_timing_exit_status_and_failure_reason",
            "junit_terminal_itl_lifecycle_stdout_stderr_paths_and_raw_sha256",
        ),
        "adversarial_security_claim": False,
    }
)
TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema": TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_MODE,
        "trust_assumptions": FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        "content_source": (
            "runtime_BOUND_trusted_single_operator_bundle_without_"
            "offline_authorization_claims"
        ),
        "models": ("exact_target_drafter_tokenizer_member_tree_and_content_sha256"),
        "workload": "exact_trusted_workload_member_and_raw_content_binding",
        "remaining_inputs_outputs": "formal_single_operator_v1_unchanged",
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_git_object_id(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case Git object ID")
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


def _strict_object(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    before = path.stat(follow_symlinks=False)
    body = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while hashing")
    return body


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_git_identity(repository_root: str | Path) -> tuple[str, str]:
    root = Path(repository_root)
    if not root.is_absolute() or root != root.resolve() or not (root / ".git").exists():
        raise ValueError(
            "single-operator repository root must be an absolute Git checkout"
        )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("formal_single_operator_v1 requires a clean Git checkout")
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    return (
        _require_git_object_id("single-operator Git HEAD", head),
        _require_git_object_id("single-operator Git tree", tree),
    )


def _patch_identity(repository_root: Path) -> tuple[str, str, str]:
    path = repository_root / "patches" / "sglang" / "manifest.json"
    body = _regular_file_bytes(path, label="SGLang patch manifest")
    value = json.loads(body)
    if type(value) is not dict or set(value) != {
        "schema_version",
        "upstream",
        "expected_tree",
        "patches",
    }:
        raise ValueError("SGLang patch manifest fields differ")
    upstream = value["upstream"]
    patches = value["patches"]
    if (
        value["schema_version"] != 2
        or type(upstream) is not dict
        or set(upstream) != {"repository", "commit"}
        or type(patches) is not list
        or not patches
    ):
        raise ValueError("SGLang patch manifest schema differs")
    for row in patches:
        if type(row) is not dict or set(row) != {"file", "sha256", "files"}:
            raise ValueError("SGLang patch row fields differ")
        patch_path = path.parent / _require_text("SGLang patch file", row["file"])
        observed = hashlib.sha256(
            _regular_file_bytes(patch_path, label="SGLang patch")
        ).hexdigest()
        if observed != _require_sha256("SGLang patch digest", row["sha256"]):
            raise ValueError("SGLang patch bytes differ from manifest")
    return (
        _require_git_object_id("SGLang upstream commit", upstream["commit"]),
        _require_git_object_id("SGLang final tree", value["expected_tree"]),
        _content_sha256(value),
    )


def _failed_tp1_outcome(
    *,
    plan: FormalServingRunPlan,
) -> tuple[int, int, int | None, str]:
    """Project one source-owned TP1 fatal pointer into honest run provenance."""

    fatal_binding = CanonicalJsonProofBinding.bind(plan.fatal_output_path)
    fatal = _strict_object(
        "single-operator TP1 fatal pointer",
        fatal_binding.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "status",
            "formal_execution_authorized",
            "reason_code",
            "error_type",
            "cleanup_error_type",
            "emitted_ns",
            "execution_started_ns",
            "run_binding_sha256",
            "requested_launch_manifest_path",
            "launch_manifest",
            "terminal_artifact",
            "native_itl_pointer_artifact",
            "live_run_receipt",
            "server_log",
            "server_process_id",
            "server_process_exit_code",
            "process_exited_ns",
            "process_group_empty",
            "process_group_empty_checked_ns",
            "before_gpu_snapshot",
            "ready_gpu_snapshot",
            "after_gpu_snapshot",
        },
    )
    from lightcone_spec.orchestration.live_sglang import (
        PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
    )

    expected_run_binding = _content_sha256(plan.native_terminal_binding.begin_payload())
    emitted_ns = fatal["emitted_ns"]
    execution_started_ns = fatal["execution_started_ns"]
    process_id = fatal["server_process_id"]
    exit_code = fatal["server_process_exit_code"]
    group_empty_checked_ns = fatal["process_group_empty_checked_ns"]
    if (
        fatal["schema_version"] != 1
        or fatal["kind"] != "unsigned_pinned_sglang_serving_fatal_pointer"
        or fatal["protocol_sha256"] != PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256
        or fatal["status"] != "ERROR"
        or fatal["formal_execution_authorized"] is not False
        or fatal["run_binding_sha256"] != expected_run_binding
        or fatal["requested_launch_manifest_path"] != plan.launch_manifest.absolute_path
        or type(emitted_ns) is not int
        or type(emitted_ns) is not int
        or emitted_ns < 1
        or (
            execution_started_ns is not None
            and (
                type(execution_started_ns) is not int
                or not 1 <= execution_started_ns < emitted_ns
            )
        )
        or (exit_code is not None and type(exit_code) is not int)
        or (
            process_id is not None
            and (
                type(process_id) is not int
                or process_id < 1
                or fatal["process_group_empty"] is not True
                or type(group_empty_checked_ns) is not int
                or group_empty_checked_ns > emitted_ns
            )
        )
    ):
        raise ValueError("single-operator TP1 fatal outcome differs from its plan")
    if fatal["launch_manifest"] is not None:
        launch_binding = CanonicalJsonProofBinding.from_dict(fatal["launch_manifest"])
        if (
            launch_binding != plan.launch_manifest
            or CanonicalJsonProofBinding.bind(launch_binding.absolute_path)
            != launch_binding
        ):
            raise ValueError("single-operator TP1 fatal launch changed")
    for field, expected_path in (
        ("terminal_artifact", plan.terminal_output_path),
        ("native_itl_pointer_artifact", plan.native_itl_pointer_output_path),
        ("live_run_receipt", plan.live_run_receipt_output_path),
        ("before_gpu_snapshot", plan.before_gpu_snapshot_output_path),
        ("ready_gpu_snapshot", plan.ready_gpu_snapshot_output_path),
        ("after_gpu_snapshot", plan.after_gpu_snapshot_output_path),
    ):
        value = fatal[field]
        if value is None:
            continue
        observed = CanonicalJsonProofBinding.from_dict(value)
        if (
            observed.absolute_path != expected_path
            or CanonicalJsonProofBinding.bind(observed.absolute_path) != observed
        ):
            raise ValueError(
                f"single-operator TP1 fatal {field} changed after publication"
            )
    reason_code = _require_text(
        "single-operator TP1 failure reason", fatal["reason_code"]
    )
    error_type = _require_text("single-operator TP1 failure type", fatal["error_type"])
    started_ns = (
        execution_started_ns
        if type(execution_started_ns) is int
        else max(1, emitted_ns - 1)
    )
    return started_ns, emitted_ns, exit_code, f"{reason_code}:{error_type}"


@dataclass(frozen=True)
class FormalSingleOperatorArtifact:
    """One expected output path and its observed raw-byte identity."""

    name: str
    relative_path: str
    status: Literal["PRESENT", "MISSING"]
    raw_sha256: str | None
    size_bytes: int | None

    def __post_init__(self) -> None:
        _require_text("single-operator artifact name", self.name)
        _require_text("single-operator artifact path", self.relative_path)
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("single-operator artifact path must be safe and relative")
        if self.status == "PRESENT":
            _require_sha256("single-operator artifact", self.raw_sha256)
            if type(self.size_bytes) is not int or self.size_bytes < 0:
                raise ValueError("present single-operator artifact size is invalid")
        elif self.status == "MISSING":
            if self.raw_sha256 is not None or self.size_bytes is not None:
                raise ValueError(
                    "missing single-operator artifact cannot carry a digest"
                )
        else:
            raise ValueError("single-operator artifact status differs")

    @classmethod
    def observe(
        cls,
        *,
        name: str,
        run_root: Path,
        path: str | Path,
    ) -> Self:
        candidate = Path(path)
        if not candidate.is_absolute() or candidate != candidate.resolve(strict=False):
            raise ValueError("single-operator observed artifact path must be absolute")
        try:
            relative = candidate.relative_to(run_root)
        except ValueError as error:
            raise ValueError(
                "single-operator artifact leaves its run directory"
            ) from error
        if not candidate.exists():
            return cls(
                name=name,
                relative_path=str(relative),
                status="MISSING",
                raw_sha256=None,
                size_bytes=None,
            )
        body = _regular_file_bytes(candidate, label=f"single-operator {name}")
        return cls(
            name=name,
            relative_path=str(relative),
            status="PRESENT",
            raw_sha256=hashlib.sha256(body).hexdigest(),
            size_bytes=len(body),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "relative_path": self.relative_path,
            "status": self.status,
            "raw_sha256": self.raw_sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "single-operator artifact",
                value,
                set(cls.__dataclass_fields__),
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorGpu:
    uuid: str
    model: str
    driver_version: str
    cuda_version: str

    def __post_init__(self) -> None:
        for label, value in (
            ("UUID", self.uuid),
            ("model", self.model),
            ("driver", self.driver_version),
            ("CUDA", self.cuda_version),
        ):
            _require_text(f"single-operator GPU {label}", value)

    def to_dict(self) -> dict[str, str]:
        return {
            "uuid": self.uuid,
            "model": self.model,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "single-operator GPU",
                value,
                set(cls.__dataclass_fields__),
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorGpuEnvironment:
    schema: Literal["formal_single_operator_gpu_environment_v1"]
    gpus: tuple[FormalSingleOperatorGpu, ...]

    def __post_init__(self) -> None:
        if self.schema != "formal_single_operator_gpu_environment_v1":
            raise ValueError("single-operator GPU environment schema differs")
        if (
            not self.gpus
            or any(type(row) is not FormalSingleOperatorGpu for row in self.gpus)
            or tuple(row.uuid for row in self.gpus)
            != tuple(sorted({row.uuid for row in self.gpus}))
        ):
            raise ValueError("single-operator GPU environment is not canonical")

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "gpus": [row.to_dict() for row in self.gpus]}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "single-operator GPU environment", value, {"schema", "gpus"}
        )
        gpus = row.pop("gpus")
        if type(gpus) is not list:
            raise TypeError("single-operator GPU environment rows must be an array")
        return cls(
            **row,  # type: ignore[arg-type]
            gpus=tuple(FormalSingleOperatorGpu.from_dict(item) for item in gpus),
        )


@dataclass(frozen=True)
class FormalSingleOperatorRunManifest:
    schema: Literal["formal_single_operator_v1", "formal_single_operator_v2"]
    protocol_sha256: str
    trust_assumptions: tuple[str, ...]
    git_head: str
    git_tree: str
    sglang_upstream_commit: str
    patch_manifest_sha256: str
    patched_sglang_tree: str
    registry_sha256: str
    physical_dispatch_protocol_sha256: str
    run_plan_sha256: str
    launch_manifest_sha256: str
    execution_binding_sha256: str
    execution_subject_sha256: str
    materialization_protocol_lock_sha256: str
    materialization_sha256: str
    inventory_sha256: str
    run_config_sha256: str
    run_config: dict[str, object]
    launch_argv_sha256: str
    launch_argv: tuple[str, ...]
    localhost_port: int
    request_schedule_sha256: str
    request_schedule: dict[str, object]
    target_model_id: str
    target_revision: str
    target_content_sha256: str | None
    drafter_model_id: str | None
    drafter_revision: str | None
    drafter_content_sha256: str | None
    tokenizer_model_id: str
    tokenizer_revision: str
    tokenizer_content_sha256: str | None
    workload_artifact_id: str
    workload_authority_sha256: str | None
    workload_member_sha256s: tuple[str, ...]
    workload_raw_sha256: str
    workload_semantic_sha256: str
    stage: str
    cell_id: str
    role: str
    backend: str
    topology: str
    block: int | None
    attempt: str
    run_directory: str
    gpu_environment: tuple[FormalSingleOperatorGpu, ...]
    started_ns: int
    finished_ns: int
    exit_code: int | None
    completion_status: Literal["COMPLETE", "FAILED"]
    failure_reason: str | None
    artifacts: tuple[FormalSingleOperatorArtifact, ...]
    content_source_mode: Literal["offline_root_signed", "trusted_single_operator"] = (
        "offline_root_signed"
    )
    trusted_content_source_binding_sha256: str | None = None
    trusted_content_bundle_sha256: str | None = None
    target_content_member_sha256: str | None = None
    target_tree_sha256: str | None = None
    target_snapshot_content_sha256: str | None = None
    drafter_content_member_sha256: str | None = None
    drafter_tree_sha256: str | None = None
    drafter_snapshot_content_sha256: str | None = None
    tokenizer_content_member_sha256: str | None = None
    tokenizer_tree_sha256: str | None = None
    tokenizer_snapshot_content_sha256: str | None = None
    trusted_workload_member_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.trust_assumptions != FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS:
            raise ValueError("single-operator manifest protocol differs")
        trusted_fields = (
            self.trusted_content_source_binding_sha256,
            self.trusted_content_bundle_sha256,
            self.target_content_member_sha256,
            self.target_tree_sha256,
            self.target_snapshot_content_sha256,
            self.drafter_content_member_sha256,
            self.drafter_tree_sha256,
            self.drafter_snapshot_content_sha256,
            self.tokenizer_content_member_sha256,
            self.tokenizer_tree_sha256,
            self.tokenizer_snapshot_content_sha256,
            self.trusted_workload_member_sha256,
        )
        if self.schema == FORMAL_SINGLE_OPERATOR_MODE:
            if (
                self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256
                or self.content_source_mode != "offline_root_signed"
                or any(value is not None for value in trusted_fields)
            ):
                raise ValueError("legacy single-operator content lineage differs")
        elif self.schema == TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_MODE:
            if (
                self.protocol_sha256
                != TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256
                or self.content_source_mode != "trusted_single_operator"
                or self.target_content_sha256 is not None
                or self.drafter_content_sha256 is not None
                or self.tokenizer_content_sha256 is not None
                or self.workload_authority_sha256 is not None
            ):
                raise ValueError("trusted single-operator content lineage differs")
            for label, value in (
                ("content source binding", self.trusted_content_source_binding_sha256),
                ("content bundle", self.trusted_content_bundle_sha256),
                ("target content member", self.target_content_member_sha256),
                ("target tree", self.target_tree_sha256),
                ("target snapshot content", self.target_snapshot_content_sha256),
                ("tokenizer content member", self.tokenizer_content_member_sha256),
                ("tokenizer tree", self.tokenizer_tree_sha256),
                (
                    "tokenizer snapshot content",
                    self.tokenizer_snapshot_content_sha256,
                ),
                ("trusted workload member", self.trusted_workload_member_sha256),
            ):
                _require_sha256(f"single-operator trusted {label}", value)
            drafter_trusted = (
                self.drafter_content_member_sha256,
                self.drafter_tree_sha256,
                self.drafter_snapshot_content_sha256,
            )
            if any(value is None for value in drafter_trusted) != all(
                value is None for value in drafter_trusted
            ):
                raise ValueError("trusted drafter content lineage is partial")
            for value in drafter_trusted:
                if value is not None:
                    _require_sha256("single-operator trusted drafter", value)
        else:
            raise ValueError("single-operator manifest protocol differs")
        for label, value in (
            ("Git HEAD", self.git_head),
            ("Git tree", self.git_tree),
            ("SGLang upstream commit", self.sglang_upstream_commit),
            ("patched SGLang tree", self.patched_sglang_tree),
        ):
            _require_git_object_id(f"single-operator {label}", value)
        for label, value in (
            ("patch manifest", self.patch_manifest_sha256),
            ("registry", self.registry_sha256),
            ("physical dispatch protocol", self.physical_dispatch_protocol_sha256),
            ("run plan", self.run_plan_sha256),
            ("launch manifest", self.launch_manifest_sha256),
            ("execution binding", self.execution_binding_sha256),
            ("execution subject", self.execution_subject_sha256),
            ("ProtocolLock", self.materialization_protocol_lock_sha256),
            ("materialization", self.materialization_sha256),
            ("inventory", self.inventory_sha256),
            ("RunConfig", self.run_config_sha256),
            ("launch argv", self.launch_argv_sha256),
            ("request schedule", self.request_schedule_sha256),
            ("workload raw", self.workload_raw_sha256),
            ("workload semantic", self.workload_semantic_sha256),
        ):
            _require_sha256(f"single-operator {label}", value)
        if self.schema == FORMAL_SINGLE_OPERATOR_MODE:
            for label, value in (
                ("target content", self.target_content_sha256),
                ("tokenizer content", self.tokenizer_content_sha256),
                ("workload authority", self.workload_authority_sha256),
            ):
                _require_sha256(f"single-operator {label}", value)
        if not self.workload_member_sha256s or self.workload_member_sha256s != tuple(
            sorted(set(self.workload_member_sha256s))
        ):
            raise ValueError("single-operator workload members are not canonical")
        for value in self.workload_member_sha256s:
            _require_sha256("single-operator workload member", value)
        if (
            run_config_sha256(RunConfig.model_validate(self.run_config))
            != self.run_config_sha256
        ):
            raise ValueError(
                "single-operator RunConfig content differs from its digest"
            )
        if (
            type(self.launch_argv) is not tuple
            or not self.launch_argv
            or any(type(value) is not str or not value for value in self.launch_argv)
            or _content_sha256({"argv": list(self.launch_argv)})
            != self.launch_argv_sha256
        ):
            raise ValueError("single-operator launch argv differs from its digest")
        if (
            type(self.localhost_port) is not int
            or not 1_024 <= self.localhost_port <= 65_535
            or _content_sha256(self.request_schedule) != self.request_schedule_sha256
        ):
            raise ValueError("single-operator port/request schedule differs")
        for optional_label, optional_value in (
            ("drafter content", self.drafter_content_sha256),
        ):
            if optional_value is not None:
                _require_sha256(f"single-operator {optional_label}", optional_value)
        for label, value in (
            ("stage", self.stage),
            ("cell", self.cell_id),
            ("role", self.role),
            ("backend", self.backend),
            ("topology", self.topology),
            ("attempt", self.attempt),
            ("target model", self.target_model_id),
            ("target revision", self.target_revision),
            ("tokenizer model", self.tokenizer_model_id),
            ("tokenizer revision", self.tokenizer_revision),
            ("workload artifact", self.workload_artifact_id),
            ("run directory", self.run_directory),
        ):
            _require_text(f"single-operator {label}", value)
        run_directory = Path(self.run_directory)
        if not run_directory.is_absolute() or run_directory != run_directory.resolve(
            strict=False
        ):
            raise ValueError("single-operator run directory must be absolute")
        if (self.drafter_model_id is None) != (self.drafter_revision is None):
            raise ValueError("single-operator drafter identity is partial")
        if self.schema == FORMAL_SINGLE_OPERATOR_MODE and (
            (self.drafter_model_id is None) != (self.drafter_content_sha256 is None)
        ):
            raise ValueError("single-operator drafter identity is partial")
        if self.schema == TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_MODE and (
            (self.drafter_model_id is None)
            != (self.drafter_content_member_sha256 is None)
        ):
            raise ValueError("trusted single-operator drafter identity is partial")
        if self.block is not None and (type(self.block) is not int or self.block < 0):
            raise ValueError("single-operator block must be a non-negative integer")
        if (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 0
            or self.finished_ns <= self.started_ns
            or (self.exit_code is not None and type(self.exit_code) is not int)
        ):
            raise ValueError("single-operator run outcome timing is invalid")
        if self.completion_status == "COMPLETE":
            if (
                self.exit_code not in {0, -signal.SIGTERM}
                or self.failure_reason is not None
            ):
                raise ValueError("completed single-operator run has a failure outcome")
            required = {
                "after_gpu_snapshot",
                "before_gpu_snapshot",
                "junit",
                "raw_terminal",
                "native_itl",
                "lifecycle",
                "live_run_receipt",
                "ready_gpu_snapshot",
                "request_schedule",
                "run_plan",
                "stdout",
                "stderr",
            }
            if self.topology in {"tp2_dp1", "tp1_dp2"}:
                required.add("formal_gang_terminal")
            present_rows = tuple(
                row for row in self.artifacts if row.status == "PRESENT"
            )
            present = {row.name for row in present_rows}
            if not required <= present:
                raise ValueError("completed single-operator run lacks required outputs")
            required_paths = tuple(
                row.relative_path for row in present_rows if row.name in required
            )
            if len(required_paths) != len(set(required_paths)):
                raise ValueError(
                    "completed single-operator outputs must be distinct files"
                )
        elif self.completion_status == "FAILED":
            _require_text("single-operator failure reason", self.failure_reason)
        else:
            raise ValueError("single-operator completion status differs")
        if (
            not self.gpu_environment
            or tuple(row.uuid for row in self.gpu_environment)
            != tuple(sorted({row.uuid for row in self.gpu_environment}))
            or not self.artifacts
            or tuple(row.name for row in self.artifacts)
            != tuple(sorted(row.name for row in self.artifacts))
            or len({row.name for row in self.artifacts}) != len(self.artifacts)
        ):
            raise ValueError("single-operator GPU/artifact rows are not canonical")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        if self.schema == FORMAL_SINGLE_OPERATOR_MODE:
            for name in (
                "content_source_mode",
                "trusted_content_source_binding_sha256",
                "trusted_content_bundle_sha256",
                "target_content_member_sha256",
                "target_tree_sha256",
                "target_snapshot_content_sha256",
                "drafter_content_member_sha256",
                "drafter_tree_sha256",
                "drafter_snapshot_content_sha256",
                "tokenizer_content_member_sha256",
                "tokenizer_tree_sha256",
                "tokenizer_snapshot_content_sha256",
                "trusted_workload_member_sha256",
            ):
                value.pop(name)
        value.update(
            {
                "trust_assumptions": list(self.trust_assumptions),
                "launch_argv": list(self.launch_argv),
                "workload_member_sha256s": list(self.workload_member_sha256s),
                "gpu_environment": [row.to_dict() for row in self.gpu_environment],
                "artifacts": [row.to_dict() for row in self.artifacts],
            }
        )
        if include_sha256:
            value["manifest_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict:
            raise TypeError("single-operator run manifest must be an object")
        if value.get("schema") == FORMAL_SINGLE_OPERATOR_MODE:
            fields -= {
                "content_source_mode",
                "trusted_content_source_binding_sha256",
                "trusted_content_bundle_sha256",
                "target_content_member_sha256",
                "target_tree_sha256",
                "target_snapshot_content_sha256",
                "drafter_content_member_sha256",
                "drafter_tree_sha256",
                "drafter_snapshot_content_sha256",
                "tokenizer_content_member_sha256",
                "tokenizer_tree_sha256",
                "tokenizer_snapshot_content_sha256",
                "trusted_workload_member_sha256",
            }
        row = _strict_object(
            "single-operator run manifest",
            value,
            fields | {"manifest_sha256"},
        )
        expected_sha256 = _require_sha256(
            "single-operator manifest", row.pop("manifest_sha256")
        )
        trust_assumptions = row.pop("trust_assumptions")
        launch_argv = row.pop("launch_argv")
        workload_member_sha256s = row.pop("workload_member_sha256s")
        gpu_environment = row.pop("gpu_environment")
        artifacts = row.pop("artifacts")
        if row.get("schema") == FORMAL_SINGLE_OPERATOR_MODE:
            row["content_source_mode"] = "offline_root_signed"
        if (
            type(trust_assumptions) is not list
            or type(launch_argv) is not list
            or type(workload_member_sha256s) is not list
            or type(gpu_environment) is not list
            or type(artifacts) is not list
        ):
            raise TypeError("single-operator manifest arrays differ")
        manifest = cls(
            **row,  # type: ignore[arg-type]
            trust_assumptions=tuple(trust_assumptions),
            launch_argv=tuple(launch_argv),
            workload_member_sha256s=tuple(workload_member_sha256s),
            gpu_environment=tuple(
                FormalSingleOperatorGpu.from_dict(item) for item in gpu_environment
            ),
            artifacts=tuple(
                FormalSingleOperatorArtifact.from_dict(item) for item in artifacts
            ),
        )
        if manifest.sha256 != expected_sha256:
            raise ValueError("single-operator manifest digest differs")
        return manifest


def create_formal_single_operator_run_directory(
    *,
    repository_root: str | Path,
    base_output_root: str | Path,
    stage: str,
    cell_id: str,
    attempt: str,
    started_ns: int,
) -> Path:
    """Create one deterministic private run directory without replacement."""

    repository = Path(repository_root)
    if (
        not repository.is_absolute()
        or repository != repository.resolve()
        or not (repository / ".git").exists()
    ):
        raise ValueError(
            "single-operator repository root must be an absolute Git checkout"
        )
    base = Path(base_output_root)
    if not base.is_absolute() or base != base.resolve(strict=False):
        raise ValueError("single-operator base output root must be absolute")
    if not base.is_dir() or base.is_symlink():
        raise ValueError("single-operator base output root must exist")
    try:
        base.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("single-operator output root must be outside the Git checkout")
    for label, value in (("stage", stage), ("cell", cell_id), ("attempt", attempt)):
        _require_text(f"single-operator run {label}", value)
    if type(started_ns) is not int or started_ns < 0:
        raise ValueError("single-operator run start must be non-negative")
    safe_stage = "".join(
        character if character.isalnum() else "-" for character in stage
    )
    safe_cell = "".join(
        character if character.isalnum() else "-" for character in cell_id
    )
    safe_attempt = "".join(
        character if character.isalnum() else "-" for character in attempt
    )
    if (
        not safe_stage.strip("-")
        or not safe_cell.strip("-")
        or not safe_attempt.strip("-")
    ):
        raise ValueError("single-operator run identity has no safe filename characters")
    destination = base / (f"{safe_stage}-{safe_cell[:16]}-{safe_attempt}-{started_ns}")
    os.mkdir(destination, 0o700)
    return destination


def finalize_formal_single_operator_run(
    *,
    repository_root: str | Path,
    run_plan_path: str | Path,
    execution_source_path: str | Path | None = None,
    inventory_path: str | Path | None = None,
) -> FormalSingleOperatorRunManifest:
    """Reopen one actual run and atomically publish its canonical provenance."""

    repository = Path(repository_root)
    git_head, git_tree = _clean_git_identity(repository)
    upstream, patched_tree, patch_manifest_sha256 = _patch_identity(repository)
    plan_binding = CanonicalJsonProofBinding.bind(run_plan_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if plan.sha256 != plan_binding.semantic_sha256:
        raise ValueError("single-operator run plan digest differs")
    current_downstream_rebuild_binding: CanonicalJsonProofBinding | None = None
    current_materialization_binding: CanonicalJsonProofBinding | None = None
    if plan.schema_version == 2:
        if plan.single_operator_execution_rebuild_source is None:
            raise ValueError("current single-operator plan lacks its rebuild source")
        # Rebuild the whole schema-2 plan before accepting any inferred path.
        # This rejoins the launch, request schedule, materialization, inventory,
        # native binding, and (for prepared rows) the exact prepared bundle
        # entry.  Merely decoding the small rebuild descriptor would otherwise
        # let a stale/foreign descriptor name individually well-formed files.
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            _load_formal_single_operator_trusted_run_plan,
        )

        replayed_plan, replayed_launch, replayed_schedule = (
            _load_formal_single_operator_trusted_run_plan(plan_binding.absolute_path)
        )
        if (
            replayed_plan != plan
            or replayed_launch.sha256 != plan.launch_manifest.semantic_sha256
            or replayed_schedule.sha256 != plan.request_schedule_receipt.semantic_sha256
        ):
            raise ValueError("current single-operator run plan replay differs")
        rebuild_binding = plan.single_operator_execution_rebuild_source
        rebuild_value = rebuild_binding.reopen()
        rebuild_kind = rebuild_value.get("kind")
        if rebuild_kind == "formal_single_operator_execution_rebuild_source":
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                revalidate_formal_single_operator_execution_rebuild_source,
            )

            rebuild_source = revalidate_formal_single_operator_execution_rebuild_source(
                plan.single_operator_execution_rebuild_source.absolute_path
            )
            inferred_execution_binding = rebuild_source.execution_source
            inferred_inventory_binding = rebuild_source.inventory
        elif rebuild_kind == "formal_single_operator_early_run_plan_inputs":
            from lightcone_spec.experiments.formal_preflight_inputs import (
                FormalPreflightExecutionInputs,
            )
            from lightcone_spec.experiments.formal_single_operator_early_execution import (
                FormalSingleOperatorEarlyRunPlanInputs,
            )

            direct = FormalSingleOperatorEarlyRunPlanInputs.from_dict(rebuild_value)
            if direct.sha256 != rebuild_binding.semantic_sha256:
                raise ValueError("current early run input digest differs")
            preflight = FormalPreflightExecutionInputs.from_dict(
                direct.preflight_inputs.reopen()
            )
            if preflight.sha256 != direct.preflight_inputs.semantic_sha256:
                raise ValueError("current early preflight input digest differs")
            inferred_execution_binding = direct.execution_source
            inferred_inventory_binding = preflight.inventory
            current_materialization_binding = direct.materialization
        elif rebuild_kind == "formal_single_operator_downstream_run_plan_inputs":
            from lightcone_spec.experiments.formal_preflight_inputs import (
                FormalPreflightExecutionInputs,
            )
            from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
                FormalSingleOperatorDownstreamRunPlanInputs,
            )

            direct = FormalSingleOperatorDownstreamRunPlanInputs.from_dict(
                rebuild_value
            )
            if direct.sha256 != rebuild_binding.semantic_sha256:
                raise ValueError("current downstream run input digest differs")
            preflight = FormalPreflightExecutionInputs.from_dict(
                direct.preflight_inputs.reopen()
            )
            if preflight.sha256 != direct.preflight_inputs.semantic_sha256:
                raise ValueError("current downstream preflight input digest differs")
            inferred_execution_binding = direct.execution_source
            inferred_inventory_binding = preflight.inventory
            current_downstream_rebuild_binding = rebuild_binding
            current_materialization_binding = direct.materialization
        elif rebuild_kind == (
            "formal_single_operator_prepared_downstream_run_plan_inputs"
        ):
            from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
                FormalSingleOperatorPreparedDownstreamRunPlanInputs,
            )

            # The trusted-plan replay above has already deep-opened this exact
            # descriptor and bundle.  Reopen it once more here to derive paths
            # from the typed object rather than caller-provided strings.
            prepared = FormalSingleOperatorPreparedDownstreamRunPlanInputs.from_dict(
                rebuild_binding.reopen()
            )
            if prepared.sha256 != rebuild_binding.semantic_sha256:
                raise ValueError("current prepared downstream input digest differs")
            inferred_execution_binding = prepared.execution_source
            inferred_inventory_binding = prepared.inventory
            current_downstream_rebuild_binding = rebuild_binding
            current_materialization_binding = prepared.materialization
        else:
            raise ValueError("current single-operator plan source is unsupported")
        inferred_execution_source = inferred_execution_binding.absolute_path
        inferred_inventory = inferred_inventory_binding.absolute_path
        if execution_source_path is not None and (
            CanonicalJsonProofBinding.bind(execution_source_path)
            != inferred_execution_binding
        ):
            raise ValueError("caller execution source differs from current run plan")
        if inventory_path is not None and (
            CanonicalJsonProofBinding.bind(inventory_path) != inferred_inventory_binding
        ):
            raise ValueError("caller inventory differs from current run plan")
        execution_source_path = inferred_execution_source
        inventory_path = inferred_inventory
    elif execution_source_path is None or inventory_path is None:
        raise ValueError(
            "legacy single-operator plan requires explicit execution source and "
            "inventory"
        )
    assert execution_source_path is not None
    assert inventory_path is not None
    run_root = Path(plan.private_output_root)
    if run_root != Path(run_plan_path).parent or run_root.is_symlink():
        raise ValueError("single-operator run plan is outside its run directory")
    try:
        run_root.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError(
            "single-operator run directory must be outside the Git checkout"
        )
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    config = load_run_config(launch.run_config_path)
    if run_config_sha256(config) != launch.run_config_semantic_sha256:
        raise ValueError("single-operator RunConfig differs from launch")
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        plan.request_schedule_receipt.reopen()
    )
    if schedule.sha256 != plan.request_schedule_receipt.semantic_sha256:
        raise ValueError("single-operator request schedule digest differs")
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )

    materialization = stage_materialization_receipt_from_dict(
        schedule.materialization.reopen()
    )
    from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
    from lightcone_spec.experiments.formal_single_operator_stages import (
        load_formal_single_operator_execution_source,
    )

    execution_source = load_formal_single_operator_execution_source(
        execution_source_path
    )
    protocol_lock = protocol_lock_from_dict(
        execution_source.protocol_lock_source.reopen(
            label="single-operator finalizer ProtocolLock"
        )
    )
    if (
        execution_source.stage != plan.stage
        or execution_source.materialization_sha256 != materialization.sha256
        or execution_source.materialization_source.absolute_path
        != schedule.materialization.absolute_path
        or execution_source.materialization_source.raw_sha256
        != schedule.materialization.raw_sha256
        or execution_source.materialization_source.semantic_sha256
        != schedule.materialization.semantic_sha256
        or protocol_lock.sha256 != materialization.protocol_lock_sha256
        or protocol_lock.code_git_head != git_head
        or protocol_lock.code_git_tree != git_tree
        or protocol_lock.patch_manifest_sha256 != patch_manifest_sha256
        or protocol_lock.registry_sha256 != build_industrial_registry().sha256
    ):
        raise ValueError(
            "single-operator execution source differs from the clean release"
        )
    cells = tuple(
        row for row in materialization.cells if row.cell_id == plan.materialized_cell_id
    )
    if len(cells) != 1:
        raise ValueError("single-operator run cell is outside materialization")
    cell = cells[0]
    if (
        cell.stage != plan.stage
        or cell.cell_id != plan.materialized_cell_id
        or launch.inventory_sha256 != plan.inventory_sha256
        or launch.gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("single-operator run plan differs from materialized cell")
    inventory_binding = CanonicalJsonProofBinding.bind(inventory_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    if inventory.sha256 != plan.inventory_sha256:
        raise ValueError("single-operator inventory differs from run assignment")
    devices = {row.uuid: row for row in inventory.devices}
    if set(plan.gpu_uuids) - set(devices):
        raise ValueError("single-operator GPU assignment leaves inventory")
    compile_cache = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
    compile_key = compile_cache.key
    environment = FormalSingleOperatorGpuEnvironment(
        schema="formal_single_operator_gpu_environment_v1",
        gpus=tuple(
            sorted(
                (
                    FormalSingleOperatorGpu(
                        uuid=uuid,
                        model=devices[uuid].model,
                        driver_version=compile_key.driver_version,
                        cuda_version=compile_key.cuda_version,
                    )
                    for uuid in plan.gpu_uuids
                ),
                key=lambda row: row.uuid,
            )
        ),
    )
    if any(devices[uuid].model != compile_key.gpu_model for uuid in plan.gpu_uuids):
        raise ValueError("single-operator compile GPU model differs from inventory")
    registry_sha256 = build_industrial_registry().sha256
    live_path = Path(plan.live_run_receipt_output_path)
    fatal_path = Path(plan.fatal_output_path)
    live_binding = (
        None if not live_path.is_file() else CanonicalJsonProofBinding.bind(live_path)
    )
    live_value = None if live_binding is None else live_binding.reopen()
    if live_binding is not None and type(live_value) is not dict:
        raise TypeError("single-operator live receipt must be an object")
    if (
        live_binding is not None
        and live_value.get("kind") == "unsigned_pinned_sglang_serving_run_receipt"
    ):
        from lightcone_spec.orchestration.live_sglang import (
            UnsignedPinnedSglangServingRunReceipt,
        )

        live = UnsignedPinnedSglangServingRunReceipt.from_dict(live_value)
        if (
            live.sha256 != live_binding.semantic_sha256
            or live.launch_manifest.semantic_sha256
            != plan.launch_manifest.semantic_sha256
            or live.inventory_sha256 != plan.inventory_sha256
            or live.gpu_uuids != plan.gpu_uuids
            or live.run_binding_sha256
            != _content_sha256(plan.native_terminal_binding.begin_payload())
            or live.process_group_empty is not True
        ):
            raise ValueError("single-operator TP1 outcome differs from its run plan")
        started_ns = live.server_process_started_ns
        finished_ns = live.process_group_empty_checked_ns
        exit_code = live.process_exit_code
        failure_reason = None
    elif (
        live_binding is not None
        and live_value.get("kind") == "unsigned_formal_gang_physical_run_receipt"
    ):
        from lightcone_spec.orchestration.formal_terminal_result import (
            validate_formal_distributed_physical_outcome,
        )

        gang_terminal_path = plan.formal_gang_terminal_output_path
        if gang_terminal_path is None:
            raise ValueError("single-operator distributed plan lacks gang output")
        distributed = validate_formal_distributed_physical_outcome(
            plan_path=str(Path(run_plan_path)),
            run_receipt_path=plan.live_run_receipt_output_path,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=registry_sha256,
        )
        if (
            distributed.run_receipt != live_binding
            or distributed.plan != plan_binding
            or distributed.request_terminal
            != CanonicalJsonProofBinding.bind(plan.terminal_output_path)
            or distributed.native_itl_pointers
            != CanonicalJsonProofBinding.bind(plan.native_itl_pointer_output_path)
            or distributed.lifecycle_timing
            != CanonicalJsonProofBinding.bind(plan.lifecycle_timing_output_path)
            or distributed.gang_terminal
            != CanonicalJsonProofBinding.bind(gang_terminal_path)
        ):
            raise ValueError(
                "single-operator distributed outcome paths differ from its plan"
            )
        started_ns = distributed.execution_started_ns
        finished_ns = distributed.finished_ns
        exit_code = distributed.process_exit_code
        failure_reason = None
    elif (
        live_binding is None
        and fatal_path.is_file()
        and plan.topology_mode == "tp1_dp1"
    ):
        started_ns, finished_ns, exit_code, failure_reason = _failed_tp1_outcome(
            plan=plan,
        )
    else:
        raise ValueError("single-operator physical outcome schema is unsupported")
    execution_source_snapshot_path = run_root / "execution-source.json"
    protocol_lock_snapshot_path = run_root / "protocol-lock.json"
    publish_canonical_json_no_replace(
        execution_source_snapshot_path,
        execution_source.to_dict(),
    )
    publish_canonical_json_no_replace(
        protocol_lock_snapshot_path,
        execution_source.protocol_lock_source.reopen(
            label="single-operator finalizer ProtocolLock snapshot"
        ),
    )
    current_snapshot_paths: dict[str, str] = {}
    if current_downstream_rebuild_binding is not None:
        if current_materialization_binding is None:
            raise AssertionError("current downstream materialization was not inferred")
        materialization_snapshot_path = run_root / "materialization.json"
        inventory_snapshot_path = run_root / "inventory.json"
        publish_canonical_json_no_replace(
            materialization_snapshot_path,
            current_materialization_binding.reopen(),
        )
        publish_canonical_json_no_replace(
            inventory_snapshot_path,
            inventory_binding.reopen(),
        )
        current_snapshot_paths = {
            "inventory": str(inventory_snapshot_path),
            "materialization": str(materialization_snapshot_path),
            "run_plan_inputs": current_downstream_rebuild_binding.absolute_path,
        }
    artifact_paths = {
        "after_gpu_snapshot": plan.after_gpu_snapshot_output_path,
        "before_gpu_snapshot": plan.before_gpu_snapshot_output_path,
        "execution_source": str(execution_source_snapshot_path),
        "fatal": plan.fatal_output_path,
        "junit": plan.junit_output_path,
        "lifecycle": plan.lifecycle_timing_output_path,
        "live_run_receipt": plan.live_run_receipt_output_path,
        "native_itl": plan.native_itl_pointer_output_path,
        "protocol_lock": str(protocol_lock_snapshot_path),
        "raw_terminal": plan.terminal_output_path,
        "ready_gpu_snapshot": plan.ready_gpu_snapshot_output_path,
        "request_schedule": plan.request_schedule_receipt.absolute_path,
        "run_plan": str(Path(run_plan_path)),
        "server_log": plan.server_log_output_path,
        "stderr": plan.server_stderr_output_path,
        "stdout": plan.server_stdout_output_path,
        **current_snapshot_paths,
    }
    if plan.formal_gang_terminal_output_path is not None:
        artifact_paths["formal_gang_terminal"] = plan.formal_gang_terminal_output_path
    artifacts = tuple(
        sorted(
            (
                FormalSingleOperatorArtifact.observe(
                    name=name,
                    run_root=run_root,
                    path=path,
                )
                for name, path in artifact_paths.items()
            ),
            key=lambda row: row.name,
        )
    )
    completion_status: Literal["COMPLETE", "FAILED"] = (
        "COMPLETE"
        if exit_code in {0, -signal.SIGTERM} and failure_reason is None
        else "FAILED"
    )
    dimensions = dict(cell.dimensions)
    run_config_value = config.model_dump(mode="json")
    trusted_content_source_binding_sha256: str | None = None
    trusted_content_bundle_sha256: str | None = None
    target_content_member_sha256: str | None = None
    target_tree_sha256: str | None = None
    target_snapshot_content_sha256: str | None = None
    drafter_content_member_sha256: str | None = None
    drafter_tree_sha256: str | None = None
    drafter_snapshot_content_sha256: str | None = None
    tokenizer_content_member_sha256: str | None = None
    tokenizer_tree_sha256: str | None = None
    tokenizer_snapshot_content_sha256: str | None = None
    trusted_workload_member_sha256: str | None = None
    if launch.schema_version == 2:
        from lightcone_spec.experiments.formal_content_source import (
            FormalContentSourceBinding,
        )
        from lightcone_spec.experiments.formal_single_operator_content import (
            TrustedModelSnapshotMember,
            TrustedSingleOperatorContentBundle,
        )

        content_source = launch.content_source_binding
        if (
            type(content_source) is not FormalContentSourceBinding
            or content_source.mode != "trusted_single_operator"
        ):
            raise ValueError("trusted run manifest lacks its tagged content source")
        trusted_bundle = content_source.reopen()
        if type(trusted_bundle) is not TrustedSingleOperatorContentBundle:
            raise TypeError("trusted run manifest content bundle is not exact")

        def trusted_member(member_id: str | None) -> TrustedModelSnapshotMember | None:
            if member_id is None:
                return None
            matches = tuple(
                row for row in trusted_bundle.model_members if row.sha256 == member_id
            )
            if len(matches) != 1:
                raise ValueError("trusted run manifest model member is not exact")
            return matches[0]

        target_member = trusted_member(launch.target_content_member_id)
        tokenizer_member = trusted_member(launch.tokenizer_content_member_id)
        drafter_member = trusted_member(launch.drafter_content_member_id)
        assert target_member is not None and tokenizer_member is not None
        trusted_content_source_binding_sha256 = content_source.sha256
        trusted_content_bundle_sha256 = content_source.content_sha256
        target_content_member_sha256 = target_member.sha256
        target_tree_sha256 = target_member.tree_sha256
        target_snapshot_content_sha256 = target_member.content_sha256
        tokenizer_content_member_sha256 = tokenizer_member.sha256
        tokenizer_tree_sha256 = tokenizer_member.tree_sha256
        tokenizer_snapshot_content_sha256 = tokenizer_member.content_sha256
        if drafter_member is not None:
            drafter_content_member_sha256 = drafter_member.sha256
            drafter_tree_sha256 = drafter_member.tree_sha256
            drafter_snapshot_content_sha256 = drafter_member.content_sha256
        trusted_workload_member_sha256 = schedule.trusted_workload_member_sha256
        if trusted_workload_member_sha256 is None:
            raise ValueError("trusted run manifest lacks workload member lineage")
    manifest = FormalSingleOperatorRunManifest(
        schema=(
            TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_MODE
            if launch.schema_version == 2
            else FORMAL_SINGLE_OPERATOR_MODE
        ),
        protocol_sha256=(
            TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256
            if launch.schema_version == 2
            else FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256
        ),
        trust_assumptions=FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        git_head=git_head,
        git_tree=git_tree,
        sglang_upstream_commit=upstream,
        patch_manifest_sha256=patch_manifest_sha256,
        patched_sglang_tree=patched_tree,
        registry_sha256=registry_sha256,
        physical_dispatch_protocol_sha256=plan.protocol_sha256,
        run_plan_sha256=plan.sha256,
        launch_manifest_sha256=launch.sha256,
        execution_binding_sha256=plan.execution_binding_sha256,
        execution_subject_sha256=plan.subject_sha256,
        materialization_protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        run_config_sha256=run_config_sha256(config),
        run_config=run_config_value,
        launch_argv_sha256=launch.server_argv_sha256,
        launch_argv=launch.server_argv,
        localhost_port=launch.localhost_port,
        request_schedule_sha256=schedule.sha256,
        request_schedule=schedule.to_dict(),
        target_model_id=launch.target_model_id,
        target_revision=launch.target_revision,
        target_content_sha256=(
            None
            if launch.schema_version == 2
            else launch.target_content_authority_sha256
        ),
        drafter_model_id=launch.drafter_model_id,
        drafter_revision=launch.drafter_revision,
        drafter_content_sha256=(
            None
            if launch.schema_version == 2
            else launch.drafter_content_authority_sha256
        ),
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_content_sha256=(
            None
            if launch.schema_version == 2
            else launch.tokenizer_content_authority_sha256
        ),
        workload_artifact_id=schedule.workload_source.artifact_id,
        workload_authority_sha256=schedule.workload_authority_sha256,
        workload_member_sha256s=tuple(
            sorted(
                {
                    row.source_member_sha256
                    for row in formal_serving_request_schedule_rows(schedule)
                }
            )
        ),
        workload_raw_sha256=schedule.workload_source.raw_sha256,
        workload_semantic_sha256=schedule.workload_source.semantic_sha256,
        stage=cell.stage,
        cell_id=cell.cell_id,
        role=cell.method_role,
        backend=cell.backend,
        topology=plan.topology_mode,
        block=dimensions.get("block") if type(dimensions.get("block")) is int else None,
        attempt=plan.native_terminal_binding.attempt_id,
        run_directory=str(run_root),
        gpu_environment=tuple(sorted(environment.gpus, key=lambda row: row.uuid)),
        started_ns=started_ns,
        finished_ns=finished_ns,
        exit_code=exit_code,
        completion_status=completion_status,
        failure_reason=failure_reason,
        artifacts=artifacts,
        content_source_mode=(
            "trusted_single_operator"
            if launch.schema_version == 2
            else "offline_root_signed"
        ),
        trusted_content_source_binding_sha256=(trusted_content_source_binding_sha256),
        trusted_content_bundle_sha256=trusted_content_bundle_sha256,
        target_content_member_sha256=target_content_member_sha256,
        target_tree_sha256=target_tree_sha256,
        target_snapshot_content_sha256=target_snapshot_content_sha256,
        drafter_content_member_sha256=drafter_content_member_sha256,
        drafter_tree_sha256=drafter_tree_sha256,
        drafter_snapshot_content_sha256=drafter_snapshot_content_sha256,
        tokenizer_content_member_sha256=tokenizer_content_member_sha256,
        tokenizer_tree_sha256=tokenizer_tree_sha256,
        tokenizer_snapshot_content_sha256=(tokenizer_snapshot_content_sha256),
        trusted_workload_member_sha256=trusted_workload_member_sha256,
    )
    output_path = run_root / "formal-single-operator-manifest.json"
    raw_sha256, size = publish_canonical_json_no_replace(
        output_path, manifest.to_dict()
    )
    rebound = CanonicalJsonProofBinding.bind(output_path)
    if rebound.raw_sha256 != raw_sha256 or rebound.size != size:
        raise RuntimeError("single-operator manifest changed during publication")
    publish_canonical_json_no_replace(
        run_root / "formal-single-operator-manifest.sha256.json",
        {
            "schema": "formal_single_operator_manifest_pointer_v1",
            "manifest_raw_sha256": raw_sha256,
            "manifest_semantic_sha256": manifest.sha256,
            "manifest_size": size,
        },
    )
    return manifest


def revalidate_formal_single_operator_run_manifest(
    *,
    repository_root: str | Path,
    manifest_path: str | Path,
) -> FormalSingleOperatorRunManifest:
    """Reopen the manifest, clean source identity, and every recorded output."""

    path = Path(manifest_path)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path.name != "formal-single-operator-manifest.json"
    ):
        raise ValueError("single-operator manifest path differs")
    binding = CanonicalJsonProofBinding.bind(path)
    manifest = FormalSingleOperatorRunManifest.from_dict(binding.reopen())
    pointer = CanonicalJsonProofBinding.bind(
        path.with_name("formal-single-operator-manifest.sha256.json")
    ).reopen()
    if pointer != {
        "schema": "formal_single_operator_manifest_pointer_v1",
        "manifest_raw_sha256": binding.raw_sha256,
        "manifest_semantic_sha256": manifest.sha256,
        "manifest_size": binding.size,
    }:
        raise ValueError("single-operator manifest pointer differs")
    repository = Path(repository_root)
    git_head, git_tree = _clean_git_identity(repository)
    upstream, patched_tree, patch_manifest_sha256 = _patch_identity(repository)
    if (
        manifest.git_head != git_head
        or manifest.git_tree != git_tree
        or manifest.sglang_upstream_commit != upstream
        or manifest.patched_sglang_tree != patched_tree
        or manifest.patch_manifest_sha256 != patch_manifest_sha256
        or manifest.registry_sha256 != build_industrial_registry().sha256
    ):
        raise ValueError("single-operator source identity changed after the run")
    run_root = path.parent
    for artifact in manifest.artifacts:
        observed = FormalSingleOperatorArtifact.observe(
            name=artifact.name,
            run_root=run_root,
            path=run_root / artifact.relative_path,
        )
        if observed != artifact:
            raise ValueError(
                f"single-operator output changed after the run: {artifact.name}"
            )
    return manifest


FORMAL_SINGLE_OPERATOR_RESIDENT_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_resident_run_manifest",
        "trust_assumptions": FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        "scope": "prepared_ordinary_tp1_schema4_trusted_content_only",
        "registered_identity": (
            "run_plan_launch_config_schedule_materialization_inventory_and_content"
        ),
        "physical_identity": (
            "shared_launch_reset_boundary_member_trace_and_shared_close"
        ),
        "cell_evidence": (
            "registered_raw_terminal_native_itl_client_lifecycle_junit_and_"
            "trace_lifecycle"
        ),
        "cell_outcome": "trace_interval_without_process_exit_or_pg_empty_claim",
        "publication": "after_shared_close_deep_revalidation_only",
        "claim": "trusted_single_operator_empirical_no_signature",
        "formal_measured": False,
    }
)


@dataclass(frozen=True)
class FormalSingleOperatorResidentRunManifest:
    """One scientific cell executed inside a sealed resident TP1 process."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_resident_run_manifest"]
    protocol_sha256: str
    trust_assumptions: tuple[str, ...]
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]
    git_head: str
    git_tree: str
    sglang_upstream_commit: str
    patch_manifest_sha256: str
    patched_sglang_tree: str
    registry_sha256: str
    run_plan: CanonicalJsonProofBinding
    launch_manifest: CanonicalJsonProofBinding
    materialization: CanonicalJsonProofBinding
    inventory: CanonicalJsonProofBinding
    request_schedule_binding: CanonicalJsonProofBinding
    group_plan: CanonicalJsonProofBinding
    reset_authority: CanonicalJsonProofBinding
    shared_launch: CanonicalJsonProofBinding
    reset_boundary: CanonicalJsonProofBinding
    resident_trace: CanonicalJsonProofBinding
    shared_close: CanonicalJsonProofBinding
    group_id: str
    group_session_binding_sha256: str
    member_index: int
    session_epoch: int
    physical_dispatch_protocol_sha256: str
    execution_binding_sha256: str
    execution_subject_sha256: str
    materialization_protocol_lock_sha256: str
    materialization_sha256: str
    inventory_sha256: str
    run_config_sha256: str
    run_config: dict[str, object]
    registered_launch_argv_sha256: str
    registered_launch_argv: tuple[str, ...]
    registered_localhost_port: int
    request_schedule_sha256: str
    request_schedule: dict[str, object]
    target_model_id: str
    target_revision: str
    target_snapshot_sha256: str
    drafter_model_id: str | None
    drafter_revision: str | None
    tokenizer_model_id: str
    tokenizer_revision: str
    workload_artifact_id: str
    workload_member_sha256s: tuple[str, ...]
    workload_raw_sha256: str
    workload_semantic_sha256: str
    stage: str
    cell_id: str
    role: str
    backend: str
    topology: Literal["tp1_dp1"]
    block: int | None
    attempt: str
    run_directory: str
    gpu_environment: tuple[FormalSingleOperatorGpu, ...]
    trace_started_ns: int
    scored_started_ns: int
    trace_finished_ns: int
    completion_status: Literal["COMPLETE"]
    artifacts: tuple[FormalSingleOperatorArtifact, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_resident_run_manifest"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_RESIDENT_PROTOCOL_SHA256
            or self.trust_assumptions != FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS
            or self.evidence_level != "trusted_single_operator_empirical_no_signature"
            or self.formal_measured is not False
            or self.topology != "tp1_dp1"
            or self.completion_status != "COMPLETE"
        ):
            raise ValueError("resident single-operator manifest schema differs")
        for label, value in (
            ("Git HEAD", self.git_head),
            ("Git tree", self.git_tree),
            ("SGLang upstream", self.sglang_upstream_commit),
            ("patched SGLang tree", self.patched_sglang_tree),
        ):
            _require_git_object_id(f"resident {label}", value)
        for label, value in (
            ("patch manifest", self.patch_manifest_sha256),
            ("registry", self.registry_sha256),
            ("group ID", self.group_id),
            ("session", self.group_session_binding_sha256),
            ("physical dispatch", self.physical_dispatch_protocol_sha256),
            ("execution binding", self.execution_binding_sha256),
            ("execution subject", self.execution_subject_sha256),
            ("ProtocolLock", self.materialization_protocol_lock_sha256),
            ("materialization", self.materialization_sha256),
            ("inventory", self.inventory_sha256),
            ("RunConfig", self.run_config_sha256),
            ("registered argv", self.registered_launch_argv_sha256),
            ("request schedule", self.request_schedule_sha256),
            ("target snapshot", self.target_snapshot_sha256),
            ("workload raw", self.workload_raw_sha256),
            ("workload semantic", self.workload_semantic_sha256),
        ):
            _require_sha256(f"resident {label}", value)
        for label, value in (
            ("run plan", self.run_plan),
            ("launch manifest", self.launch_manifest),
            ("materialization", self.materialization),
            ("inventory", self.inventory),
            ("request schedule", self.request_schedule_binding),
            ("group plan", self.group_plan),
            ("reset authority", self.reset_authority),
            ("shared launch", self.shared_launch),
            ("reset boundary", self.reset_boundary),
            ("resident trace", self.resident_trace),
            ("shared close", self.shared_close),
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError(f"resident {label} binding type differs")
        if (
            run_config_sha256(RunConfig.model_validate(self.run_config))
            != self.run_config_sha256
            or type(self.registered_launch_argv) is not tuple
            or not self.registered_launch_argv
            or _content_sha256({"argv": list(self.registered_launch_argv)})
            != self.registered_launch_argv_sha256
            or type(self.registered_localhost_port) is not int
            or not 1_024 <= self.registered_localhost_port <= 65_535
            or _content_sha256(self.request_schedule) != self.request_schedule_sha256
        ):
            raise ValueError("resident registered launch/config/schedule differs")
        if (
            type(self.member_index) is not int
            or self.member_index < 0
            or self.session_epoch != self.member_index + 1
            or any(
                type(value) is not int or value < 1
                for value in (
                    self.trace_started_ns,
                    self.scored_started_ns,
                    self.trace_finished_ns,
                )
            )
            or not (
                self.trace_started_ns
                <= self.scored_started_ns
                <= self.trace_finished_ns
            )
        ):
            raise ValueError("resident trace lifecycle differs")
        for label, value in (
            ("stage", self.stage),
            ("cell", self.cell_id),
            ("role", self.role),
            ("backend", self.backend),
            ("attempt", self.attempt),
            ("target model", self.target_model_id),
            ("target revision", self.target_revision),
            ("tokenizer model", self.tokenizer_model_id),
            ("tokenizer revision", self.tokenizer_revision),
            ("workload artifact", self.workload_artifact_id),
            ("run directory", self.run_directory),
        ):
            _require_text(f"resident {label}", value)
        if (self.drafter_model_id is None) != (self.drafter_revision is None):
            raise ValueError("resident drafter identity is partial")
        if self.block is not None and (type(self.block) is not int or self.block < 0):
            raise ValueError("resident block is invalid")
        if (
            not self.workload_member_sha256s
            or self.workload_member_sha256s
            != tuple(sorted(set(self.workload_member_sha256s)))
            or any(
                _require_sha256("resident workload member", item) != item
                for item in self.workload_member_sha256s
            )
            or not self.gpu_environment
            or tuple(row.uuid for row in self.gpu_environment)
            != tuple(sorted({row.uuid for row in self.gpu_environment}))
            or not self.artifacts
            or tuple(row.name for row in self.artifacts)
            != tuple(sorted({row.name for row in self.artifacts}))
        ):
            raise ValueError("resident workload/GPU/artifact rows are not canonical")
        required = {
            "client_lifecycle",
            "execution_source",
            "inventory",
            "junit",
            "lifecycle",
            "live_run_receipt",
            "materialization",
            "native_itl",
            "protocol_lock",
            "raw_terminal",
            "request_schedule",
            "run_plan",
            "run_plan_inputs",
        }
        present = {row.name for row in self.artifacts if row.status == "PRESENT"}
        if not required <= present:
            raise ValueError("resident manifest lacks complete per-cell evidence")

    @property
    def started_ns(self) -> int:
        return self.trace_started_ns

    @property
    def finished_ns(self) -> int:
        return self.trace_finished_ns

    @property
    def exit_code(self) -> None:
        """Resident cells do not own or claim the shared process exit."""

        return None

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        value["trust_assumptions"] = list(self.trust_assumptions)
        value["registered_launch_argv"] = list(self.registered_launch_argv)
        value["workload_member_sha256s"] = list(self.workload_member_sha256s)
        value["gpu_environment"] = [row.to_dict() for row in self.gpu_environment]
        value["artifacts"] = [row.to_dict() for row in self.artifacts]
        for name in (
            "run_plan",
            "launch_manifest",
            "materialization",
            "inventory",
            "request_schedule_binding",
            "group_plan",
            "reset_authority",
            "shared_launch",
            "reset_boundary",
            "resident_trace",
            "shared_close",
        ):
            value[name] = getattr(self, name).to_dict()
        if include_sha256:
            value["manifest_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "resident single-operator manifest",
            value,
            {*cls.__dataclass_fields__, "manifest_sha256"},
        )
        declared = _require_sha256(
            "resident single-operator manifest", row.pop("manifest_sha256")
        )
        for name in (
            "run_plan",
            "launch_manifest",
            "materialization",
            "inventory",
            "request_schedule_binding",
            "group_plan",
            "reset_authority",
            "shared_launch",
            "reset_boundary",
            "resident_trace",
            "shared_close",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        assumptions = row.pop("trust_assumptions")
        argv = row.pop("registered_launch_argv")
        members = row.pop("workload_member_sha256s")
        gpu = row.pop("gpu_environment")
        artifacts = row.pop("artifacts")
        if any(
            type(item) is not list
            for item in (assumptions, argv, members, gpu, artifacts)
        ):
            raise TypeError("resident manifest arrays differ")
        result = cls(
            **row,  # type: ignore[arg-type]
            trust_assumptions=tuple(assumptions),
            registered_launch_argv=tuple(argv),
            workload_member_sha256s=tuple(members),
            gpu_environment=tuple(
                FormalSingleOperatorGpu.from_dict(item) for item in gpu
            ),
            artifacts=tuple(
                FormalSingleOperatorArtifact.from_dict(item) for item in artifacts
            ),
        )
        if result.sha256 != declared:
            raise ValueError("resident manifest digest differs")
        return result


def _resident_manifest_pointer(path: Path) -> Path:
    return path.with_name("formal-single-operator-manifest.sha256.json")


def _validate_resident_trace_artifacts(
    *, plan: FormalServingRunPlan, trace: object
) -> None:
    """Join the trace receipt to the registered per-cell output paths."""

    from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding

    expected_canonical = {
        "raw_terminal": plan.terminal_output_path,
        "native_itl": plan.native_itl_pointer_output_path,
        "client_lifecycle": str(
            (Path(plan.private_output_root) / "client-request-lifecycle.json").resolve()
        ),
        "trace_lifecycle": plan.lifecycle_timing_output_path,
    }
    for name, path in expected_canonical.items():
        if CanonicalJsonProofBinding.bind(path) != getattr(trace, name, None):
            raise ValueError(f"resident trace {name} leaves registered output")
    expected_junit = EvidenceFileBinding.bind(
        Path(plan.junit_output_path), label="resident registered JUnit"
    )
    if expected_junit != getattr(trace, "junit", None):
        raise ValueError("resident trace JUnit leaves registered output")


def finalize_formal_single_operator_resident_run(
    *,
    repository_root: str | Path,
    run_plan_path: str | Path,
    group_plan_path: str | Path,
    reset_authority_path: str | Path,
    shared_launch_path: str | Path,
    reset_boundary_path: str | Path,
    trace_receipt_path: str | Path,
    shared_close_path: str | Path,
) -> FormalSingleOperatorResidentRunManifest:
    """Seal one resident cell only after the shared process is terminal."""

    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        revalidate_formal_single_operator_prepared_downstream_run_plan_inputs,
    )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        _load_formal_single_operator_trusted_run_plan,
        _reopen_stage_materialization,
    )
    from lightcone_spec.orchestration.formal_serving_session_group import (
        FormalServingSessionGroupPlan,
    )
    from lightcone_spec.orchestration.formal_serving_session_group_physical import (
        revalidate_formal_serving_resident_reset_boundary_receipt,
        revalidate_formal_serving_resident_shared_close_receipt,
        revalidate_formal_serving_resident_shared_launch_receipt,
        revalidate_formal_serving_resident_trace_receipt,
    )

    repository = Path(repository_root).resolve()
    git_head, git_tree = _clean_git_identity(repository)
    upstream, patched_tree, patch_manifest_sha256 = _patch_identity(repository)
    plan_binding = CanonicalJsonProofBinding.bind(run_plan_path)
    plan, launch, schedule = _load_formal_single_operator_trusted_run_plan(
        plan_binding.absolute_path
    )
    if (
        plan.schema_version != 4
        or plan.topology_mode != "tp1_dp1"
        or launch.schema_version != 2
        or plan.nextn_mtp_mode is not None
        or plan.single_operator_execution_rebuild_source is None
    ):
        raise ValueError("resident manifest only supports current ordinary TP1")
    inputs_binding = plan.single_operator_execution_rebuild_source
    inputs_value = inputs_binding.reopen()
    if (
        inputs_value.get("kind")
        != "formal_single_operator_prepared_downstream_run_plan_inputs"
    ):
        raise ValueError("resident manifest requires prepared downstream inputs")
    inputs = revalidate_formal_single_operator_prepared_downstream_run_plan_inputs(
        inputs_binding.absolute_path,
        current_ns=time.time_ns(),
    )
    if inputs.schema_version != 2 or inputs.content_source_binding is None:
        raise ValueError("resident manifest requires trusted prepared content")
    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        route_formal_single_operator_cell,
    )

    execution_source, _routed_cell, _route = route_formal_single_operator_cell(
        execution_source_path=inputs.execution_source.absolute_path,
        materialized_cell_id=plan.materialized_cell_id,
    )
    materialization = _reopen_stage_materialization(schedule.materialization)
    cells = tuple(
        row for row in materialization.cells if row.cell_id == plan.materialized_cell_id
    )
    if len(cells) != 1:
        raise ValueError("resident run leaves materialization")
    cell = cells[0]
    inventory = GpuInventory.from_dict(inputs.inventory.reopen())
    config = load_run_config(launch.run_config_path)
    group_binding = CanonicalJsonProofBinding.bind(group_plan_path)
    group = FormalServingSessionGroupPlan.from_dict(group_binding.reopen())
    authority_binding = CanonicalJsonProofBinding.bind(reset_authority_path)
    from lightcone_spec.experiments.formal_single_operator_session_reset import (
        revalidate_trusted_empirical_tp1_session_reset_authority,
    )

    authority_rebound, authority = (
        revalidate_trusted_empirical_tp1_session_reset_authority(
            authority_binding.absolute_path
        )
    )
    launch_binding, resident_launch = (
        revalidate_formal_serving_resident_shared_launch_receipt(shared_launch_path)
    )
    reset_binding, reset = revalidate_formal_serving_resident_reset_boundary_receipt(
        reset_boundary_path
    )
    trace_binding, trace = revalidate_formal_serving_resident_trace_receipt(
        trace_receipt_path
    )
    close_binding, close = revalidate_formal_serving_resident_shared_close_receipt(
        shared_close_path
    )
    matches = tuple(
        (index, member)
        for index, member in enumerate(group.members)
        if member.run_plan == plan_binding
    )
    if len(matches) != 1:
        raise ValueError("resident run plan is not one exact group member")
    member_index, member = matches[0]
    if (
        authority_rebound != authority_binding
        or group.execution_mode != "shared_session_tp1"
        or group.reset_authority_sha256 != authority.sha256
        or group_binding != resident_launch.group_plan
        or authority_binding != resident_launch.reset_authority
        or trace_binding not in close.member_trace_receipts
        or trace.member_index != member_index
        or trace.member_run_plan != plan_binding
        or trace.materialized_cell_id != plan.materialized_cell_id
        or trace.reset_boundary != reset_binding
        or reset.next_materialized_cell_id != plan.materialized_cell_id
        or close.shared_launch != launch_binding
        or close.process_group_empty is not True
        or member.assigned_gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("resident physical chain differs from registered cell")
    _validate_resident_trace_artifacts(plan=plan, trace=trace)
    run_root = Path(plan.private_output_root)
    if run_root != Path(run_plan_path).parent.resolve(strict=False):
        raise ValueError("resident run root differs from registered plan")
    snapshots = {
        "execution_source": inputs.execution_source.reopen(),
        "protocol_lock": execution_source.protocol_lock_source.reopen(),
        "materialization": schedule.materialization.reopen(),
        "inventory": inputs.inventory.reopen(),
        "run_plan_inputs": inputs_binding.reopen(),
    }
    snapshot_paths: dict[str, str] = {}
    for name, value in snapshots.items():
        path = run_root / f"{name.replace('_', '-')}.json"
        publish_canonical_json_no_replace(path, value)
        snapshot_paths[name] = str(path)
    artifact_paths = {
        **snapshot_paths,
        "client_lifecycle": str((run_root / "client-request-lifecycle.json").resolve()),
        "junit": plan.junit_output_path,
        "lifecycle": plan.lifecycle_timing_output_path,
        "live_run_receipt": plan.live_run_receipt_output_path,
        "native_itl": plan.native_itl_pointer_output_path,
        "raw_terminal": plan.terminal_output_path,
        "request_schedule": plan.request_schedule_receipt.absolute_path,
        "run_plan": plan_binding.absolute_path,
    }
    artifacts = tuple(
        sorted(
            (
                FormalSingleOperatorArtifact.observe(
                    name=name, run_root=run_root, path=path
                )
                for name, path in artifact_paths.items()
            ),
            key=lambda row: row.name,
        )
    )
    compile_cache = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
    devices = {row.uuid: row for row in inventory.devices}
    gpu_environment = tuple(
        FormalSingleOperatorGpu(
            uuid=uuid,
            model=devices[uuid].model,
            driver_version=compile_cache.key.driver_version,
            cuda_version=compile_cache.key.cuda_version,
        )
        for uuid in sorted(plan.gpu_uuids)
    )
    dimensions = dict(cell.dimensions)
    trusted_bundle = inputs.content_source_binding.reopen()
    target_members = tuple(
        row
        for row in trusted_bundle.model_members
        if row.sha256 == launch.target_content_member_id
    )
    if len(target_members) != 1:
        raise ValueError("resident target content member is not exact")
    target_snapshot_sha256 = target_members[0].content_sha256
    manifest = FormalSingleOperatorResidentRunManifest(
        schema_version=1,
        kind="formal_single_operator_resident_run_manifest",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_RESIDENT_PROTOCOL_SHA256,
        trust_assumptions=FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        evidence_level="trusted_single_operator_empirical_no_signature",
        formal_measured=False,
        git_head=git_head,
        git_tree=git_tree,
        sglang_upstream_commit=upstream,
        patch_manifest_sha256=patch_manifest_sha256,
        patched_sglang_tree=patched_tree,
        registry_sha256=build_industrial_registry().sha256,
        run_plan=plan_binding,
        launch_manifest=plan.launch_manifest,
        materialization=schedule.materialization,
        inventory=inputs.inventory,
        request_schedule_binding=plan.request_schedule_receipt,
        group_plan=group_binding,
        reset_authority=authority_binding,
        shared_launch=launch_binding,
        reset_boundary=reset_binding,
        resident_trace=trace_binding,
        shared_close=close_binding,
        group_id=group.group_id,
        group_session_binding_sha256=resident_launch.group_session_binding_sha256,
        member_index=member_index,
        session_epoch=trace.session_epoch,
        physical_dispatch_protocol_sha256=plan.protocol_sha256,
        execution_binding_sha256=plan.execution_binding_sha256,
        execution_subject_sha256=plan.subject_sha256,
        materialization_protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        run_config_sha256=run_config_sha256(config),
        run_config=config.model_dump(mode="json"),
        registered_launch_argv_sha256=launch.server_argv_sha256,
        registered_launch_argv=launch.server_argv,
        registered_localhost_port=launch.localhost_port,
        request_schedule_sha256=schedule.sha256,
        request_schedule=schedule.to_dict(),
        target_model_id=launch.target_model_id,
        target_revision=launch.target_revision,
        target_snapshot_sha256=target_snapshot_sha256,
        drafter_model_id=launch.drafter_model_id,
        drafter_revision=launch.drafter_revision,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        workload_artifact_id=schedule.workload_source.artifact_id,
        workload_member_sha256s=tuple(
            sorted(
                {
                    row.source_member_sha256
                    for row in formal_serving_request_schedule_rows(schedule)
                }
            )
        ),
        workload_raw_sha256=schedule.workload_source.raw_sha256,
        workload_semantic_sha256=schedule.workload_source.semantic_sha256,
        stage=cell.stage,
        cell_id=cell.cell_id,
        role=cell.method_role,
        backend=cell.backend,
        topology="tp1_dp1",
        block=dimensions.get("block") if type(dimensions.get("block")) is int else None,
        attempt=trace.effective_terminal_binding.attempt_id,
        run_directory=str(run_root),
        gpu_environment=gpu_environment,
        trace_started_ns=trace.trace_started_ns,
        scored_started_ns=trace.scored_started_ns,
        trace_finished_ns=trace.trace_finished_ns,
        completion_status="COMPLETE",
        artifacts=artifacts,
    )
    output_path = run_root / "formal-single-operator-manifest.json"
    raw_sha256, size = publish_canonical_json_no_replace(
        output_path, manifest.to_dict()
    )
    publish_canonical_json_no_replace(
        _resident_manifest_pointer(output_path),
        {
            "schema": "formal_single_operator_resident_manifest_pointer_v1",
            "manifest_raw_sha256": raw_sha256,
            "manifest_semantic_sha256": manifest.sha256,
            "manifest_size": size,
            "shared_close_sha256": close_binding.semantic_sha256,
        },
    )
    return manifest


def revalidate_formal_single_operator_resident_run_manifest(
    *, repository_root: str | Path, manifest_path: str | Path
) -> FormalSingleOperatorResidentRunManifest:
    """Deep-reopen a resident cell, including the terminal shared process."""

    from lightcone_spec.orchestration.formal_physical_dispatch import (
        _load_formal_single_operator_trusted_run_plan,
    )
    from lightcone_spec.orchestration.formal_serving_session_group import (
        FormalServingSessionGroupPlan,
    )
    from lightcone_spec.orchestration.formal_serving_session_group_physical import (
        revalidate_formal_serving_resident_reset_boundary_receipt,
        revalidate_formal_serving_resident_shared_close_receipt,
        revalidate_formal_serving_resident_shared_launch_receipt,
        revalidate_formal_serving_resident_trace_receipt,
    )

    path = Path(manifest_path)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path.name != "formal-single-operator-manifest.json"
    ):
        raise ValueError("resident manifest path differs")
    binding = CanonicalJsonProofBinding.bind(path)
    manifest = FormalSingleOperatorResidentRunManifest.from_dict(binding.reopen())
    pointer = CanonicalJsonProofBinding.bind(_resident_manifest_pointer(path)).reopen()
    if pointer != {
        "schema": "formal_single_operator_resident_manifest_pointer_v1",
        "manifest_raw_sha256": binding.raw_sha256,
        "manifest_semantic_sha256": manifest.sha256,
        "manifest_size": binding.size,
        "shared_close_sha256": manifest.shared_close.semantic_sha256,
    }:
        raise ValueError("resident manifest pointer differs")
    repository = Path(repository_root).resolve()
    git_head, git_tree = _clean_git_identity(repository)
    upstream, patched_tree, patch_manifest_sha256 = _patch_identity(repository)
    if (
        (manifest.git_head, manifest.git_tree) != (git_head, git_tree)
        or manifest.sglang_upstream_commit != upstream
        or manifest.patched_sglang_tree != patched_tree
        or manifest.patch_manifest_sha256 != patch_manifest_sha256
        or manifest.registry_sha256 != build_industrial_registry().sha256
    ):
        raise ValueError("resident source identity changed")
    plan, launch, schedule = _load_formal_single_operator_trusted_run_plan(
        manifest.run_plan.absolute_path
    )
    group = FormalServingSessionGroupPlan.from_dict(manifest.group_plan.reopen())
    _launch_binding, resident_launch = (
        revalidate_formal_serving_resident_shared_launch_receipt(
            manifest.shared_launch.absolute_path
        )
    )
    _reset_binding, reset = revalidate_formal_serving_resident_reset_boundary_receipt(
        manifest.reset_boundary.absolute_path
    )
    _trace_binding, trace = revalidate_formal_serving_resident_trace_receipt(
        manifest.resident_trace.absolute_path
    )
    _close_binding, close = revalidate_formal_serving_resident_shared_close_receipt(
        manifest.shared_close.absolute_path
    )
    if (
        plan.sha256 != manifest.run_plan.semantic_sha256
        or launch.sha256 != manifest.launch_manifest.semantic_sha256
        or schedule.sha256 != manifest.request_schedule_sha256
        or manifest.group_id != group.group_id
        or manifest.member_index != trace.member_index
        or manifest.session_epoch != trace.session_epoch
        or manifest.cell_id != trace.materialized_cell_id
        or manifest.reset_boundary != trace.reset_boundary
        or manifest.resident_trace not in close.member_trace_receipts
        or manifest.group_session_binding_sha256
        != resident_launch.group_session_binding_sha256
        or reset.group_session_binding_sha256 != manifest.group_session_binding_sha256
        or close.group_session_binding_sha256 != manifest.group_session_binding_sha256
        or manifest.trace_started_ns != trace.trace_started_ns
        or manifest.scored_started_ns != trace.scored_started_ns
        or manifest.trace_finished_ns != trace.trace_finished_ns
    ):
        raise ValueError("resident manifest physical/scientific join differs")
    _validate_resident_trace_artifacts(plan=plan, trace=trace)
    run_root = path.parent
    if manifest.run_directory != str(run_root):
        raise ValueError("resident manifest run directory differs")
    for artifact in manifest.artifacts:
        if (
            FormalSingleOperatorArtifact.observe(
                name=artifact.name,
                run_root=run_root,
                path=run_root / artifact.relative_path,
            )
            != artifact
        ):
            raise ValueError(f"resident output changed: {artifact.name}")
    return manifest


__all__ = [
    "FORMAL_SINGLE_OPERATOR_MODE",
    "FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_RESIDENT_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS",
    "TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_MODE",
    "TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256",
    "FormalSingleOperatorArtifact",
    "FormalSingleOperatorGpu",
    "FormalSingleOperatorGpuEnvironment",
    "FormalSingleOperatorResidentRunManifest",
    "FormalSingleOperatorRunManifest",
    "create_formal_single_operator_run_directory",
    "finalize_formal_single_operator_resident_run",
    "finalize_formal_single_operator_run",
    "revalidate_formal_single_operator_resident_run_manifest",
    "revalidate_formal_single_operator_run_manifest",
]
