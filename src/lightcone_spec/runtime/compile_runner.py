"""First-party COMPILE lifecycle and terminal-result contract.

The current release has no trusted GPU actuator for this lifecycle.  The
release entry point therefore remains a named pre-mutation BLOCK.  This module
freezes and CPU-tests the complete lifecycle behind that gate: exact assignment
inputs, graph-bucket prewarm coverage, graceful process finalization, immutable
cache sealing, and an atomic result pointer whose bindings are reopened rather
than trusted as serialized summaries.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Protocol, Self

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.runtime.compile_cache import (
    COMPILE_CACHE_ENVIRONMENT_VARIABLES,
    COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256,
    COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    CompileCacheAttemptReceipt,
    CompileCacheLaunchPlan,
    CompileCacheReceipt,
    CompileOnlyAssignmentContract,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
    ImmutableCompileCache,
    _content_sha256,
    _load_canonical_json_with_sidecar,
    _publish_json,
    _publish_text,
    _stable_regular_file_bytes,
    _strict_json_object,
    preflight_compile_cache_launch,
    start_compile_cache_launch,
)
from lightcone_spec.runtime.content_authorization import (
    ContentVerificationReceipt,
    VerifiedPreparedModelContentRelease,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
    VerifiedControlArtifact,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    relocated_evidence_path,
)

COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 2,
        "kind": "first_party_compile_assignment_plan",
        "inputs": (
            "assignment_manifest_path_and_sha256",
            "compile_cache_plan_path_and_sha256",
            "prewarm_manifest_path_and_sha256",
            "compile_key_sha256",
            "model_lock_sha256",
            "target_and_drafter_revisions",
            "attempt_id",
            "result_pointer_path",
            "source_owned_launch_manifest_path_raw_and_semantic_sha256",
        ),
        "prewarm_coverage": "every_graph_bucket_exactly_once_or_more",
        "terminal_order": (
            "start",
            "prewarm_all_registered_payloads",
            "graceful_shutdown_ack",
            "seal_cache",
            "publish_atomic_result_pointer",
            "reopen_all_pointer_bindings",
        ),
        "caller_supplied_compile_key_forbidden": True,
        "release_execution": (
            "root_signed_dynamic_control_plus_durable_prepared_content_"
            "verification_receipt_without_challenge_reuse"
        ),
    }
)
RELEASE_COMPILE_RUNNER_UNAVAILABLE = "release_first_party_compile_runner_unavailable"
RELEASE_COMPILE_DYNAMIC_CONTROL_UNAVAILABLE = (
    "release_dynamic_compile_control_unavailable"
)
RELEASE_COMPILE_ASSIGNMENT_PLAN_ALLOWLIST_EMPTY = (
    "release_compile_assignment_plan_allowlist_empty"
)
RELEASE_COMPILE_ASSIGNMENT_PLAN_UNTRUSTED = (
    "release_compile_assignment_plan_not_allowlisted"
)
RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S: tuple[str, ...] = ()

COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 2,
        "kind": "first_party_compile_subprocess_lifecycle",
        "transport": "canonical_json_lines_over_private_stdin_stdout",
        "child_must_delay_model_and_gpu_initialization_until_start": True,
        "ordered_messages": (
            "ready",
            "start_with_private_compile_environment_and_exact_launch_manifest",
            "started",
            "one_prewarm_request_and_completion_per_manifest_payload",
            "drain_and_shutdown",
            "drained",
            "parent_observed_zero_exit",
        ),
        "limits": {
            "maximum_message_bytes": 1024 * 1024,
            "bounded_deadline": True,
            "unexpected_stdout_forbidden": True,
        },
        "formal_authority": (
            "source_owned_exact_command_and_executable_digest",
            "source_owned_exact_assignment_plan_sha256",
            "source_owned_launch_manifest_path_raw_and_semantic_sha256",
            "root_signed_dynamic_compile_control_and_atomic_replay_reservation",
        ),
        "cpu_diagnostic_cannot_authorize_formal_execution": True,
    }
)
COMPILE_WORKER_IMPORT_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "first_party_compile_worker_imports",
        "module": "lightcone_spec.sglang_bridge.compile_worker",
        "required_imports": (
            "lightcone_spec.experiments.serving.PinnedBenchServingTransport",
            "lightcone_spec.runtime.compile_cache.CompileOnlyPrewarmPayload",
            "lightcone_spec.runtime.compile_runner.CompileAssignmentPlan",
        ),
        "native_transport": "same_pinned_official_bench_pool",
    }
)
RELEASE_COMPILE_GPU_SOURCE_REGISTRY_EMPTY = (
    "release_compile_gpu_vetted_source_registry_empty"
)
RELEASE_COMPILE_GPU_SOURCE_UNTRUSTED = "release_compile_source_not_gpu_vetted"
RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S: tuple[str, ...] = ()

_MAX_SUBPROCESS_MESSAGE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CompileWorkerSourceDescriptor:
    """Reopenable first-party helper, interpreter, and patched checkout."""

    schema_version: int
    kind: str
    helper_module: str
    helper_path: str
    helper_raw_sha256: str
    helper_size: int
    helper_import_protocol_sha256: str
    interpreter_path: str
    interpreter_raw_sha256: str
    interpreter_size: int
    patched_sglang_checkout: str
    patched_sglang_tree: str
    compile_source_sha256: str
    native_protocol_sha256: str

    @classmethod
    def issue(
        cls,
        *,
        patched_sglang_checkout: str | Path,
        interpreter_path: str | Path | None = None,
    ) -> Self:
        from lightcone_spec.sglang_bridge.checkout import verify_patched_checkout
        from lightcone_spec.sglang_bridge.compile_worker import (
            SOURCE_OWNED_COMPILE_PROTOCOL_SHA256,
        )

        checkout = verify_patched_checkout(patched_sglang_checkout)
        specification = importlib.util.find_spec(
            "lightcone_spec.sglang_bridge.compile_worker"
        )
        if specification is None or specification.origin is None:
            raise RuntimeError("compile worker helper module cannot be resolved")
        helper = Path(specification.origin).resolve()
        interpreter = Path(interpreter_path or sys.executable).resolve()
        helper_digest, helper_size = _raw_sha256(helper, label="compile worker helper")
        interpreter_digest, interpreter_size = _raw_sha256(
            interpreter, label="compile worker interpreter"
        )
        value = cls(
            schema_version=1,
            kind="first_party_compile_worker_source",
            helper_module="lightcone_spec.sglang_bridge.compile_worker",
            helper_path=str(helper),
            helper_raw_sha256=helper_digest,
            helper_size=helper_size,
            helper_import_protocol_sha256=COMPILE_WORKER_IMPORT_PROTOCOL_SHA256,
            interpreter_path=str(interpreter),
            interpreter_raw_sha256=interpreter_digest,
            interpreter_size=interpreter_size,
            patched_sglang_checkout=str(checkout),
            patched_sglang_tree=PINNED_SGLANG_TREE,
            compile_source_sha256=PINNED_SGLANG_COMPILE_SOURCE_SHA256,
            native_protocol_sha256=SOURCE_OWNED_COMPILE_PROTOCOL_SHA256,
        )
        value.validate(reopen_sources=True)
        return value

    def validate(self, *, reopen_sources: bool) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "first_party_compile_worker_source"
            or self.helper_module != "lightcone_spec.sglang_bridge.compile_worker"
        ):
            raise ValueError("compile worker source schema is unsupported")
        helper = _absolute_path("compile worker helper", self.helper_path)
        interpreter = _absolute_path(
            "compile worker interpreter", self.interpreter_path
        )
        checkout = _absolute_path(
            "compile worker patched checkout", self.patched_sglang_checkout
        )
        for label, digest in (
            ("compile worker helper", self.helper_raw_sha256),
            ("compile worker imports", self.helper_import_protocol_sha256),
            ("compile worker interpreter", self.interpreter_raw_sha256),
            ("compile worker source", self.compile_source_sha256),
            ("compile worker native protocol", self.native_protocol_sha256),
        ):
            _require_sha256(label, digest)
        for label, size in (
            ("compile worker helper", self.helper_size),
            ("compile worker interpreter", self.interpreter_size),
        ):
            if type(size) is not int or size < 1:
                raise ValueError(f"{label} size is invalid")
        if (
            self.helper_import_protocol_sha256 != COMPILE_WORKER_IMPORT_PROTOCOL_SHA256
            or self.patched_sglang_tree != PINNED_SGLANG_TREE
            or self.compile_source_sha256 != PINNED_SGLANG_COMPILE_SOURCE_SHA256
        ):
            raise ValueError("compile worker source identity differs from release")
        if not reopen_sources:
            return
        helper_digest, helper_size = _raw_sha256(helper, label="compile worker helper")
        interpreter_digest, interpreter_size = _raw_sha256(
            interpreter, label="compile worker interpreter"
        )
        if (
            helper_digest != self.helper_raw_sha256
            or helper_size != self.helper_size
            or interpreter_digest != self.interpreter_raw_sha256
            or interpreter_size != self.interpreter_size
        ):
            raise ValueError("compile worker helper or interpreter changed")
        specification = importlib.util.find_spec(self.helper_module)
        if specification is None or specification.origin is None:
            raise RuntimeError("compile worker helper module cannot be resolved")
        if Path(specification.origin).resolve() != helper:
            raise ValueError("compile worker module resolves to another helper")
        from lightcone_spec.sglang_bridge.checkout import verify_patched_checkout
        from lightcone_spec.sglang_bridge.compile_worker import (
            SOURCE_OWNED_COMPILE_PROTOCOL_SHA256,
        )

        if verify_patched_checkout(checkout) != checkout:
            raise ValueError("compile worker checkout identity differs")
        if SOURCE_OWNED_COMPILE_PROTOCOL_SHA256 != self.native_protocol_sha256:
            raise ValueError("compile worker native protocol changed")

    @property
    def sha256(self) -> str:
        self.validate(reopen_sources=False)
        return _content_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        self.validate(reopen_sources=False)
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or set(raw) != {
            field.name for field in dataclass_fields(cls)
        }:
            raise ValueError("compile worker source fields differ from schema")
        value = cls(**raw)
        value.validate(reopen_sources=False)
        return value


@dataclass(frozen=True)
class ReleaseCompileSubprocess:
    """One GPU-vetted source-owned command for future formal COMPILE work."""

    argv: tuple[str, ...]
    worker: CompileWorkerSourceDescriptor
    protocol_sha256: str
    gpu_qualification_sha256: str

    def validate(self, *, reopen_executable: bool) -> None:
        if type(self.argv) is not tuple or not self.argv:
            raise TypeError("release compile subprocess argv must be a non-empty tuple")
        for argument in self.argv:
            if type(argument) is not str or not argument or "\x00" in argument:
                raise ValueError("release compile subprocess argv contains NUL")
        if type(self.worker) is not CompileWorkerSourceDescriptor:
            raise TypeError("release compile subprocess lacks an exact worker source")
        self.worker.validate(reopen_sources=reopen_executable)
        if self.argv[:2] != (
            self.worker.interpreter_path,
            self.worker.helper_path,
        ):
            raise ValueError("release compile argv does not execute the bound helper")
        if self.protocol_sha256 != COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256:
            raise ValueError("release compile subprocess uses another protocol")
        _require_sha256(
            "release compile GPU qualification", self.gpu_qualification_sha256
        )

    @property
    def sha256(self) -> str:
        self.validate(reopen_executable=False)
        return _content_sha256(asdict(self))


# A future reviewed release must add exactly one command together with its
# executable digest and GPU-marked lifecycle tests.  Caller data cannot extend
# either this allowlist or the assignment-plan allowlist above.
RELEASE_COMPILE_SUBPROCESSES: tuple[ReleaseCompileSubprocess, ...] = ()


class CompileRunnerBlocked(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(f"COMPILE execution is BLOCKED: {reason_code}")
        self.reason_code = reason_code


def _require_formal_compile_receipt_authority(
    *,
    assignment_plan_sha256: str,
    executable_path: str,
    executable_raw_sha256: str,
    argv_sha256: str,
    source_authority_sha256: str | None,
    reopen_executable: bool,
) -> None:
    """Reopen source-owned authority before accepting formal raw evidence."""

    if len(RELEASE_COMPILE_SUBPROCESSES) != 1:
        raise CompileRunnerBlocked(RELEASE_COMPILE_RUNNER_UNAVAILABLE)
    if not RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_ASSIGNMENT_PLAN_ALLOWLIST_EMPTY)
    if assignment_plan_sha256 not in RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_ASSIGNMENT_PLAN_UNTRUSTED)
    source = RELEASE_COMPILE_SUBPROCESSES[0]
    source.validate(reopen_executable=reopen_executable)
    if not RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_GPU_SOURCE_REGISTRY_EMPTY)
    if source.sha256 not in RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_GPU_SOURCE_UNTRUSTED)
    expected_executable = _absolute_path(
        "release compile executable", source.worker.interpreter_path
    )
    if (
        source_authority_sha256 != source.sha256
        or Path(executable_path) != expected_executable
        or executable_raw_sha256 != source.worker.interpreter_raw_sha256
        or argv_sha256 != _content_sha256({"argv": list(source.argv)})
    ):
        raise ValueError("formal compile receipt differs from source authority")


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _absolute_path(label: str, value: object) -> Path:
    text = _require_text(label, value)
    path = Path(text)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    if path == Path(path.anchor):
        raise ValueError(f"{label} cannot be a filesystem root")
    return path


def _raw_sha256(path: Path, *, label: str) -> tuple[str, int]:
    body = _stable_regular_file_bytes(path, label=label)
    return hashlib.sha256(body).hexdigest(), len(body)


def write_compile_prewarm_manifest(
    manifest: CompileOnlyPrewarmManifest,
    path: str | Path,
) -> Path:
    if type(manifest) is not CompileOnlyPrewarmManifest:
        raise TypeError("compile prewarm publication requires an exact manifest")
    manifest.validate()
    destination = _absolute_path("compile prewarm manifest", str(path))
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("compile prewarm manifest parent must be a directory")
    _publish_json(destination, manifest.to_dict())
    _publish_text(Path(f"{destination}.sha256"), manifest.sha256)
    return destination


def load_compile_prewarm_manifest(path: str | Path) -> CompileOnlyPrewarmManifest:
    source = _absolute_path("compile prewarm manifest", str(path))
    raw, semantic_sha256 = _load_canonical_json_with_sidecar(
        source,
        label="compile prewarm manifest",
    )
    manifest = CompileOnlyPrewarmManifest.from_dict(raw)
    if semantic_sha256 != manifest.sha256:
        raise ValueError("compile prewarm manifest semantic digest differs")
    return manifest


COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "first_party_compile_launch_manifest",
        "source": (PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE),
        "bindings": (
            "patched_checkout",
            "exact_run_config_raw_and_semantic",
            "compile_cache_plan_raw_and_semantic",
            "prewarm_manifest_raw_and_semantic",
            "sampling_profile_raw_and_semantic",
            "prepared_model_content_manifest_raw_semantic_size",
            "target_drafter_tokenizer_prepared_snapshots_and_content_authorities",
            "exact_localhost_server_argv_and_port",
            "model_sampling_physical_budget_inventory_identities",
            "explicit_PATH_LD_LIBRARY_PATH_CUDA_HOME_without_inherited_environment",
        ),
        "child_start": "manifest_path_raw_semantic_sha256",
    }
)
TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 2,
        "kind": "first_party_compile_launch_manifest",
        "content_source": (
            "runtime_BOUND_trusted_single_operator_bundle_exact_stage_role_"
            "model_revision_snapshot_tree_without_offline_authorization_claims"
        ),
        "legacy_signed_schema": "schema1_unchanged",
        "remaining_bindings": "same_as_schema1",
    }
)
TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256 = (
    _content_sha256(
        {
            "schema_version": 3,
            "kind": "first_party_compile_launch_manifest",
            "content_source": "runtime_BOUND_trusted_single_operator_bundle",
            "nextn_mode": "built_in_mtp_same_target_snapshot_no_external_drafter",
            "mtp_component": (
                "path_bound_config_weight_index_and_safetensors_header_scan"
            ),
            "target_snapshot": "frozen_content_sha256",
            "legacy_signed_and_external_schemas": "schema1_and_schema2_unchanged",
        }
    )
)


@dataclass(frozen=True)
class CompileLaunchManifest:
    """Every non-cache input needed by the source-owned compile worker."""

    schema_version: int
    kind: str
    protocol_sha256: str
    patched_sglang_checkout: str
    patched_sglang_commit: str
    patched_sglang_tree: str
    run_config_path: str
    run_config_raw_sha256: str
    run_config_semantic_sha256: str
    compile_cache_plan_path: str
    compile_cache_plan_raw_sha256: str
    compile_cache_plan_sha256: str
    prewarm_manifest_path: str
    prewarm_manifest_raw_sha256: str
    prewarm_manifest_sha256: str
    sampling_profile_path: str
    sampling_profile_raw_sha256: str
    prepared_model_content_manifest_path: str
    prepared_model_content_manifest_raw_sha256: str
    prepared_model_content_manifest_sha256: str
    prepared_model_content_manifest_size: int
    target_content_member_id: str
    target_model_id: str
    target_snapshot_path: str
    target_revision: str
    target_content_authority_sha256: str | None
    drafter_content_member_id: str | None
    drafter_model_id: str | None
    drafter_snapshot_path: str | None
    drafter_revision: str | None
    drafter_content_authority_sha256: str | None
    tokenizer_content_member_id: str
    tokenizer_model_id: str
    tokenizer_snapshot_path: str
    tokenizer_revision: str
    tokenizer_content_authority_sha256: str | None
    server_argv: tuple[str, ...]
    server_argv_sha256: str
    localhost_port: int
    model_lock_sha256: str
    sampling_profile_sha256: str
    physical_assignment_sha256: str
    experiment_budget_sha256: str
    budget_materialization_authority_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    path_entries: tuple[str, ...]
    library_path_entries: tuple[str, ...]
    cuda_home: str
    formal_stage: str | None = None
    content_source_binding: object | None = None
    nextn_mtp_mode: str | None = None
    target_snapshot_sha256: str | None = None
    mtp_component_sha256: str | None = None
    mtp_component_binding: object | None = None

    def validate(self, *, reopen_inputs: bool) -> None:
        if (
            self.schema_version not in {1, 2, 3}
            or self.kind != "first_party_compile_launch_manifest"
            or self.patched_sglang_commit != PINNED_SGLANG_COMMIT
            or self.patched_sglang_tree != PINNED_SGLANG_TREE
        ):
            raise ValueError("compile launch manifest schema/source is unsupported")
        if self.schema_version == 1:
            if (
                self.protocol_sha256 != COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
                or self.formal_stage is not None
                or self.content_source_binding is not None
            ):
                raise ValueError("legacy compile launch content source differs")
        elif self.schema_version == 2:
            if self.protocol_sha256 != (
                TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
            ):
                raise ValueError("trusted compile launch protocol differs")
        elif self.protocol_sha256 != (
            TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
        ):
            raise ValueError("trusted built-in MTP compile launch protocol differs")
        for label, digest in (
            ("run config raw", self.run_config_raw_sha256),
            ("run config semantic", self.run_config_semantic_sha256),
            ("compile cache plan raw", self.compile_cache_plan_raw_sha256),
            ("compile cache plan", self.compile_cache_plan_sha256),
            ("prewarm manifest raw", self.prewarm_manifest_raw_sha256),
            ("prewarm manifest", self.prewarm_manifest_sha256),
            ("sampling profile raw", self.sampling_profile_raw_sha256),
            (
                "prepared-model content manifest raw",
                self.prepared_model_content_manifest_raw_sha256,
            ),
            (
                "prepared-model content manifest",
                self.prepared_model_content_manifest_sha256,
            ),
            ("server argv", self.server_argv_sha256),
            ("model lock", self.model_lock_sha256),
            ("sampling profile", self.sampling_profile_sha256),
            ("physical assignment", self.physical_assignment_sha256),
            ("experiment budget", self.experiment_budget_sha256),
            (
                "budget materialization authority",
                self.budget_materialization_authority_sha256,
            ),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"compile launch {label}", digest)
        if self.schema_version == 1:
            _require_sha256(
                "compile launch target content authority",
                self.target_content_authority_sha256,
            )
            _require_sha256(
                "compile launch tokenizer content authority",
                self.tokenizer_content_authority_sha256,
            )
        elif (
            self.target_content_authority_sha256 is not None
            or self.tokenizer_content_authority_sha256 is not None
            or self.drafter_content_authority_sha256 is not None
        ):
            raise ValueError(
                "trusted compile launch must not carry offline authorization claims"
            )
        for label, value in (
            ("target model ID", self.target_model_id),
            ("target revision", self.target_revision),
            ("target content member ID", self.target_content_member_id),
            ("tokenizer model ID", self.tokenizer_model_id),
            ("tokenizer revision", self.tokenizer_revision),
            ("tokenizer content member ID", self.tokenizer_content_member_id),
        ):
            _require_text(f"compile launch {label}", value)
        for label, value in (
            ("target", self.target_revision),
            ("tokenizer", self.tokenizer_revision),
        ):
            if len(value) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"compile launch {label} revision is not immutable")
        drafter_values = (
            self.drafter_content_member_id,
            self.drafter_model_id,
            self.drafter_snapshot_path,
            self.drafter_revision,
        )
        if any(value is None for value in drafter_values) != all(
            value is None for value in drafter_values
        ):
            raise ValueError(
                "compile launch drafter inputs must be all present or absent"
            )
        if self.schema_version == 1 and (
            (self.drafter_model_id is None)
            != (self.drafter_content_authority_sha256 is None)
        ):
            raise ValueError(
                "compile launch drafter authority must follow its model inputs"
            )
        if self.drafter_model_id is not None:
            _require_text(
                "compile launch drafter content member ID",
                self.drafter_content_member_id,
            )
            _require_text("compile launch drafter model ID", self.drafter_model_id)
            _require_text("compile launch drafter revision", self.drafter_revision)
            if len(self.drafter_revision) not in {40, 64} or any(
                character not in "0123456789abcdef"
                for character in self.drafter_revision
            ):
                raise ValueError("compile launch drafter revision is not immutable")
            if self.schema_version == 1:
                _require_sha256(
                    "compile launch drafter content authority",
                    self.drafter_content_authority_sha256,
                )
        checkout = _absolute_path(
            "compile launch patched checkout", self.patched_sglang_checkout
        )
        run_config = _absolute_path("compile launch run config", self.run_config_path)
        cache_plan_path = _absolute_path(
            "compile launch cache plan", self.compile_cache_plan_path
        )
        prewarm_path = _absolute_path(
            "compile launch prewarm manifest", self.prewarm_manifest_path
        )
        sampling_path = _absolute_path(
            "compile launch sampling profile", self.sampling_profile_path
        )
        prepared_content_path = _absolute_path(
            "compile launch prepared-model content manifest",
            self.prepared_model_content_manifest_path,
        )
        if (
            type(self.prepared_model_content_manifest_size) is not int
            or self.prepared_model_content_manifest_size < 1
        ):
            raise ValueError("compile launch prepared content size is invalid")
        target = _absolute_path(
            "compile launch target snapshot", self.target_snapshot_path
        )
        tokenizer = _absolute_path(
            "compile launch tokenizer snapshot", self.tokenizer_snapshot_path
        )
        drafter = (
            None
            if self.drafter_snapshot_path is None
            else _absolute_path(
                "compile launch drafter snapshot", self.drafter_snapshot_path
            )
        )
        if self.schema_version in {2, 3}:
            from lightcone_spec.experiments.formal_content_source import (
                FormalContentSourceBinding,
            )
            from lightcone_spec.experiments.formal_single_operator_content import (
                TrustedSingleOperatorContentBundle,
            )

            if (
                type(self.content_source_binding) is not FormalContentSourceBinding
                or self.content_source_binding.mode != "trusted_single_operator"
                or type(self.formal_stage) is not str
                or not self.formal_stage
            ):
                raise ValueError("trusted compile launch content lineage differs")
            bundle = self.content_source_binding.reopen()
            if type(bundle) is not TrustedSingleOperatorContentBundle:
                raise TypeError("trusted compile launch content bundle is not exact")
            trusted_binding = self.content_source_binding.trusted_single_operator
            assert trusted_binding is not None
            if (
                bundle.runtime_binding_status != "BOUND"
                or bundle.source_snapshot.sglang_upstream_commit
                != self.patched_sglang_commit
                or bundle.source_snapshot.patched_sglang_tree
                != self.patched_sglang_tree
                or prepared_content_path != Path(trusted_binding.absolute_path)
                or self.prepared_model_content_manifest_raw_sha256
                != trusted_binding.raw_sha256
                or self.prepared_model_content_manifest_sha256
                != trusted_binding.semantic_sha256
                or self.prepared_model_content_manifest_size != trusted_binding.size
            ):
                raise ValueError("trusted compile launch bundle identity differs")

            def require_trusted_member(
                *,
                member_id: str,
                role: str,
                model_id: str,
                revision: str,
                snapshot_path: Path,
            ) -> None:
                matches = tuple(
                    row
                    for row in bundle.model_members
                    if row.role == role
                    and row.model_id == model_id
                    and row.revision == revision
                    and self.formal_stage in row.stages
                    and Path(row.local_snapshot_path) == snapshot_path
                )
                if len(matches) != 1 or matches[0].sha256 != member_id:
                    raise ValueError("trusted compile launch model member is not exact")

            require_trusted_member(
                member_id=self.target_content_member_id,
                role="target",
                model_id=self.target_model_id,
                revision=self.target_revision,
                snapshot_path=target,
            )
            require_trusted_member(
                member_id=self.tokenizer_content_member_id,
                role="tokenizer",
                model_id=self.tokenizer_model_id,
                revision=self.tokenizer_revision,
                snapshot_path=tokenizer,
            )
            if drafter is not None and self.schema_version == 2:
                assert self.drafter_content_member_id is not None
                assert self.drafter_model_id is not None
                assert self.drafter_revision is not None
                require_trusted_member(
                    member_id=self.drafter_content_member_id,
                    role="drafter",
                    model_id=self.drafter_model_id,
                    revision=self.drafter_revision,
                    snapshot_path=drafter,
                )
            if self.schema_version == 3:
                from lightcone_spec.config import load_run_config
                from lightcone_spec.experiments.formal_single_operator_e6_builtin_mtp import (
                    FormalSingleOperatorE6BuiltInMtpComponent,
                    revalidate_formal_single_operator_e6_builtin_mtp_component,
                )
                from lightcone_spec.runtime.proof_artifact import (
                    CanonicalJsonProofBinding,
                )

                target_members = tuple(
                    row
                    for row in bundle.model_members
                    if row.sha256 == self.target_content_member_id
                    and row.role == "target"
                    and row.model_id == self.target_model_id
                    and row.revision == self.target_revision
                    and self.formal_stage in row.stages
                    and Path(row.local_snapshot_path) == target
                )
                if len(target_members) != 1:
                    raise ValueError("built-in MTP target member is not exact")
                target_member = target_members[0]
                if type(self.mtp_component_binding) is not CanonicalJsonProofBinding:
                    raise TypeError(
                        "built-in MTP compile launch lacks component binding"
                    )
                component = revalidate_formal_single_operator_e6_builtin_mtp_component(
                    self.mtp_component_binding.absolute_path,
                    member=target_member,
                )
                config = load_run_config(self.run_config_path)
                if (
                    type(component) is not FormalSingleOperatorE6BuiltInMtpComponent
                    or self.nextn_mtp_mode != "built_in_mtp"
                    or self.target_snapshot_sha256 != target_member.content_sha256
                    or self.target_snapshot_sha256 != component.target_snapshot_sha256
                    or self.mtp_component_sha256 != component.sha256
                    or self.mtp_component_binding.semantic_sha256 != component.sha256
                    or config.model.algorithm != "NEXTN"
                    or config.model.nextn_mtp_mode != "built_in_mtp"
                    or config.model.target_snapshot_sha256
                    != self.target_snapshot_sha256
                    or config.model.mtp_component_sha256 != self.mtp_component_sha256
                    or self.drafter_content_member_id != self.target_content_member_id
                    or self.drafter_model_id != self.target_model_id
                    or self.drafter_revision != self.target_revision
                    or drafter != target
                ):
                    raise ValueError("built-in MTP compile launch identity differs")
        if self.schema_version in {1, 2} and any(
            value is not None
            for value in (
                self.nextn_mtp_mode,
                self.target_snapshot_sha256,
                self.mtp_component_sha256,
                self.mtp_component_binding,
            )
        ):
            raise ValueError(
                "legacy/external compile launch carries built-in MTP state"
            )
        if target.name != self.target_revision or tokenizer.name != (
            self.tokenizer_revision
        ):
            raise ValueError("compile launch snapshot leaf differs from revision")
        if drafter is not None and drafter.name != self.drafter_revision:
            raise ValueError("compile launch drafter leaf differs from revision")
        if (
            type(self.server_argv) is not tuple
            or not self.server_argv
            or any(
                type(argument) is not str or not argument or "\x00" in argument
                for argument in self.server_argv
            )
            or self.server_argv_sha256
            != _content_sha256({"argv": list(self.server_argv)})
        ):
            raise ValueError("compile launch server argv is invalid")
        if type(self.localhost_port) is not int or not (
            1_024 <= self.localhost_port <= 65_535
        ):
            raise ValueError("compile launch localhost port is invalid")
        expected_pairs = {
            "--host": "127.0.0.1",
            "--port": str(self.localhost_port),
            "--model-path": str(target),
        }
        if self.drafter_snapshot_path is not None and self.schema_version != 3:
            expected_pairs["--speculative-draft-model-path"] = str(drafter)
        for flag, expected in expected_pairs.items():
            positions = tuple(
                index
                for index, argument in enumerate(self.server_argv)
                if argument == flag
            )
            if (
                len(positions) != 1
                or positions[0] + 1 >= len(self.server_argv)
                or self.server_argv[positions[0] + 1] != expected
            ):
                raise ValueError(f"compile launch server argv differs at {flag}")
        if self.schema_version == 3 and "--speculative-draft-model-path" in (
            self.server_argv
        ):
            raise ValueError("built-in MTP launch must not pass an external draft path")
        if (
            type(self.gpu_uuids) is not tuple
            or not self.gpu_uuids
            or len(set(self.gpu_uuids)) != len(self.gpu_uuids)
        ):
            raise ValueError("compile launch GPU UUIDs are invalid")
        for gpu_uuid in self.gpu_uuids:
            _require_text("compile launch GPU UUID", gpu_uuid)
        for label, values in (
            ("PATH", self.path_entries),
            ("LD_LIBRARY_PATH", self.library_path_entries),
        ):
            if (
                type(values) is not tuple
                or not values
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"compile launch {label} entries are invalid")
            for entry in values:
                path = _absolute_path(f"compile launch {label} entry", entry)
                if reopen_inputs and (not path.is_dir() or path.is_symlink()):
                    raise ValueError(f"compile launch {label} entry is unavailable")
        cuda_home = _absolute_path("compile launch CUDA home", self.cuda_home)
        if not reopen_inputs:
            return
        for label, directory in (
            ("patched checkout", checkout),
            ("target snapshot", target),
            ("tokenizer snapshot", tokenizer),
            ("CUDA home", cuda_home),
        ):
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError(f"compile launch {label} is unavailable")
        if drafter is not None and (not drafter.is_dir() or drafter.is_symlink()):
            raise ValueError("compile launch drafter snapshot is unavailable")
        raw_digest, raw_size = _raw_sha256(
            run_config, label="compile launch run config"
        )
        if raw_size < 1 or raw_digest != self.run_config_raw_sha256:
            raise ValueError("compile launch run-config raw bytes changed")
        _raw, semantic = _load_canonical_json_with_sidecar(
            run_config, label="compile launch run config"
        )
        if semantic != self.run_config_semantic_sha256:
            raise ValueError("compile launch run-config semantic identity changed")
        bound_files = [
            (
                "cache plan",
                cache_plan_path,
                self.compile_cache_plan_raw_sha256,
                self.compile_cache_plan_sha256,
            ),
            (
                "prewarm manifest",
                prewarm_path,
                self.prewarm_manifest_raw_sha256,
                self.prewarm_manifest_sha256,
            ),
            (
                "sampling profile",
                sampling_path,
                self.sampling_profile_raw_sha256,
                self.sampling_profile_sha256,
            ),
        ]
        if self.schema_version == 1:
            bound_files.append(
                (
                    "prepared-model content manifest",
                    prepared_content_path,
                    self.prepared_model_content_manifest_raw_sha256,
                    self.prepared_model_content_manifest_sha256,
                )
            )
        for label, path, expected_raw, expected_semantic in bound_files:
            file_raw, file_size = _raw_sha256(path, label=f"compile launch {label}")
            _value, file_semantic = _load_canonical_json_with_sidecar(
                path, label=f"compile launch {label}"
            )
            if (
                file_size < 1
                or file_raw != expected_raw
                or file_semantic != expected_semantic
                or (
                    label == "prepared-model content manifest"
                    and file_size != self.prepared_model_content_manifest_size
                )
            ):
                raise ValueError(f"compile launch {label} identity changed")
        cache_plan = CompileCacheLaunchPlan.load(cache_plan_path)
        prewarm = load_compile_prewarm_manifest(prewarm_path)
        if (
            cache_plan.sha256 != self.compile_cache_plan_sha256
            or prewarm.sha256 != self.prewarm_manifest_sha256
            or prewarm.sampling_profile_sha256 != self.sampling_profile_sha256
        ):
            raise ValueError("compile launch cache/prewarm/sampling bindings differ")

    def to_dict(self) -> dict[str, object]:
        self.validate(reopen_inputs=False)
        value = {
            **asdict(self),
            "server_argv": list(self.server_argv),
            "gpu_uuids": list(self.gpu_uuids),
            "path_entries": list(self.path_entries),
            "library_path_entries": list(self.library_path_entries),
        }
        if self.schema_version == 1:
            value.pop("formal_stage")
            value.pop("content_source_binding")
            value.pop("nextn_mtp_mode")
            value.pop("target_snapshot_sha256")
            value.pop("mtp_component_sha256")
            value.pop("mtp_component_binding")
        elif self.schema_version == 2:
            value.pop("nextn_mtp_mode")
            value.pop("target_snapshot_sha256")
            value.pop("mtp_component_sha256")
            value.pop("mtp_component_binding")
            from lightcone_spec.experiments.formal_content_source import (
                FormalContentSourceBinding,
            )

            assert type(self.content_source_binding) is FormalContentSourceBinding
            value["content_source_binding"] = self.content_source_binding.to_dict()
        else:
            from lightcone_spec.experiments.formal_content_source import (
                FormalContentSourceBinding,
            )

            assert type(self.content_source_binding) is FormalContentSourceBinding
            value["content_source_binding"] = self.content_source_binding.to_dict()
            from lightcone_spec.runtime.proof_artifact import (
                CanonicalJsonProofBinding,
            )

            assert type(self.mtp_component_binding) is CanonicalJsonProofBinding
            value["mtp_component_binding"] = self.mtp_component_binding.to_dict()
        return value

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict:
            raise TypeError("compile launch manifest must be a JSON object")
        schema_version = raw.get("schema_version")
        expected = {field.name for field in dataclass_fields(cls)}
        if schema_version == 1:
            expected -= {
                "formal_stage",
                "content_source_binding",
                "nextn_mtp_mode",
                "target_snapshot_sha256",
                "mtp_component_sha256",
                "mtp_component_binding",
            }
        elif schema_version == 2:
            expected -= {
                "nextn_mtp_mode",
                "target_snapshot_sha256",
                "mtp_component_sha256",
                "mtp_component_binding",
            }
        if set(raw) != expected:
            raise ValueError("compile launch manifest fields differ from schema")
        payload = dict(raw)
        for name in (
            "server_argv",
            "gpu_uuids",
            "path_entries",
            "library_path_entries",
        ):
            value = payload[name]
            if type(value) is not list:
                raise TypeError(f"compile launch {name} must be an array")
            payload[name] = tuple(value)
        if schema_version == 1:
            payload["formal_stage"] = None
            payload["content_source_binding"] = None
            payload["nextn_mtp_mode"] = None
            payload["target_snapshot_sha256"] = None
            payload["mtp_component_sha256"] = None
            payload["mtp_component_binding"] = None
        else:
            from lightcone_spec.experiments.formal_content_source import (
                FormalContentSourceBinding,
            )

            payload["content_source_binding"] = FormalContentSourceBinding.from_dict(
                payload["content_source_binding"]
            )
            if schema_version == 2:
                payload["nextn_mtp_mode"] = None
                payload["target_snapshot_sha256"] = None
                payload["mtp_component_sha256"] = None
                payload["mtp_component_binding"] = None
            else:
                from lightcone_spec.runtime.proof_artifact import (
                    CanonicalJsonProofBinding,
                )

                payload["mtp_component_binding"] = CanonicalJsonProofBinding.from_dict(
                    payload["mtp_component_binding"]
                )
        result = cls(**payload)
        result.validate(reopen_inputs=False)
        return result

    def write(self, path: str | Path) -> Path:
        destination = _absolute_path("compile launch manifest", str(path))
        _publish_json(destination, self.to_dict())
        _publish_text(Path(f"{destination}.sha256"), self.sha256)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("compile launch manifest", str(path))
        raw, semantic = _load_canonical_json_with_sidecar(
            source, label="compile launch manifest"
        )
        value = cls.from_dict(raw)
        if semantic != value.sha256:
            raise ValueError("compile launch manifest semantic digest differs")
        value.validate(reopen_inputs=True)
        return value

    def child_environment(self) -> dict[str, str]:
        self.validate(reopen_inputs=True)
        environment = {
            "PATH": os.pathsep.join(self.path_entries),
            "LD_LIBRARY_PATH": os.pathsep.join(self.library_path_entries),
            "CUDA_HOME": self.cuda_home,
            "CUDA_PATH": self.cuda_home,
            "CUDA_VISIBLE_DEVICES": ",".join(self.gpu_uuids),
            "NCCL_DEBUG": "INFO",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "LANG": "C",
            "LC_ALL": "C",
        }
        if self.schema_version == 3:
            assert self.target_snapshot_sha256 is not None
            assert self.mtp_component_sha256 is not None
            assert self.mtp_component_binding is not None
            environment.update(
                {
                    "LIGHTCONE_NEXTN_MTP_MODE": "built_in_mtp",
                    "LIGHTCONE_NEXTN_TARGET_SNAPSHOT_SHA256": (
                        self.target_snapshot_sha256
                    ),
                    "LIGHTCONE_NEXTN_MTP_COMPONENT_SHA256": (self.mtp_component_sha256),
                    "LIGHTCONE_NEXTN_MTP_COMPONENT_PATH": (
                        self.mtp_component_binding.absolute_path
                    ),
                }
            )
        return environment


@dataclass(frozen=True)
class CompileAssignmentPlan:
    schema_version: int
    kind: str
    protocol_sha256: str
    assignment_manifest_path: str
    assignment_sha256: str
    compile_cache_plan_path: str
    compile_cache_plan_sha256: str
    prewarm_manifest_path: str
    prewarm_manifest_sha256: str
    launch_manifest_path: str
    launch_manifest_raw_sha256: str
    launch_manifest_sha256: str
    compile_key_sha256: str
    model_lock_sha256: str
    target_revision: str
    drafter_revision: str | None
    physical_assignment_sha256: str
    experiment_budget_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    host_id: str
    tensor_parallel_size: int
    context_limit: int
    max_running_requests: int
    graph_buckets: tuple[int, ...]
    graceful_shutdown_protocol_sha256: str
    result_pointer_protocol_sha256: str
    attempt_id: str
    result_pointer_path: str

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 2
            or self.kind != "first_party_compile_assignment_plan"
        ):
            raise ValueError("compile assignment plan schema is unsupported")
        if self.protocol_sha256 != COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256:
            raise ValueError("compile assignment plan uses another protocol")
        for label, value in (
            ("assignment", self.assignment_sha256),
            ("cache plan", self.compile_cache_plan_sha256),
            ("prewarm manifest", self.prewarm_manifest_sha256),
            ("launch manifest raw", self.launch_manifest_raw_sha256),
            ("launch manifest", self.launch_manifest_sha256),
            ("compile key", self.compile_key_sha256),
            ("model lock", self.model_lock_sha256),
            ("physical assignment", self.physical_assignment_sha256),
            ("experiment budget", self.experiment_budget_sha256),
            ("inventory", self.inventory_sha256),
            ("shutdown protocol", self.graceful_shutdown_protocol_sha256),
            ("result pointer protocol", self.result_pointer_protocol_sha256),
        ):
            _require_sha256(label, value)
        for label, value in (
            ("assignment manifest", self.assignment_manifest_path),
            ("compile cache plan", self.compile_cache_plan_path),
            ("prewarm manifest", self.prewarm_manifest_path),
            ("launch manifest", self.launch_manifest_path),
            ("compile result pointer", self.result_pointer_path),
        ):
            _absolute_path(label, value)
        _require_text("target revision", self.target_revision)
        if self.drafter_revision is not None:
            _require_text("drafter revision", self.drafter_revision)
        if (
            not self.gpu_uuids
            or type(self.gpu_uuids) is not tuple
            or len(set(self.gpu_uuids)) != len(self.gpu_uuids)
        ):
            raise ValueError(
                "compile assignment GPU UUIDs must be unique and non-empty"
            )
        for gpu_uuid in self.gpu_uuids:
            _require_text("compile assignment GPU UUID", gpu_uuid)
        _require_text("compile assignment host", self.host_id)
        for label, value in (
            ("tensor parallel size", self.tensor_parallel_size),
            ("context limit", self.context_limit),
            ("maximum running requests", self.max_running_requests),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"compile assignment {label} must be positive")
        if (
            type(self.graph_buckets) is not tuple
            or not self.graph_buckets
            or tuple(sorted(set(self.graph_buckets))) != self.graph_buckets
            or any(type(value) is not int or value < 1 for value in self.graph_buckets)
        ):
            raise ValueError("compile assignment graph buckets are invalid")
        if self.graceful_shutdown_protocol_sha256 != (
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ):
            raise ValueError("compile assignment uses another shutdown protocol")
        if self.result_pointer_protocol_sha256 != (
            COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256
        ):
            raise ValueError("compile assignment uses another result-pointer protocol")
        attempt = _require_text("compile attempt ID", self.attempt_id)
        if any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in attempt
        ):
            raise ValueError("compile attempt ID is unsafe")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def write(self, path: str | Path) -> Path:
        destination = _absolute_path("compile assignment plan", str(path))
        if not destination.parent.is_dir() or destination.parent.is_symlink():
            raise ValueError("compile assignment plan parent must be a directory")
        _publish_json(destination, self.to_dict())
        _publish_text(Path(f"{destination}.sha256"), self.sha256)
        return destination

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        expected = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "assignment_manifest_path",
            "assignment_sha256",
            "compile_cache_plan_path",
            "compile_cache_plan_sha256",
            "prewarm_manifest_path",
            "prewarm_manifest_sha256",
            "launch_manifest_path",
            "launch_manifest_raw_sha256",
            "launch_manifest_sha256",
            "compile_key_sha256",
            "model_lock_sha256",
            "target_revision",
            "drafter_revision",
            "physical_assignment_sha256",
            "experiment_budget_sha256",
            "inventory_sha256",
            "gpu_uuids",
            "host_id",
            "tensor_parallel_size",
            "context_limit",
            "max_running_requests",
            "graph_buckets",
            "graceful_shutdown_protocol_sha256",
            "result_pointer_protocol_sha256",
            "attempt_id",
            "result_pointer_path",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ValueError("compile assignment plan fields differ from schema")
        payload = dict(raw)
        gpu_uuids = payload.pop("gpu_uuids")
        graph_buckets = payload.pop("graph_buckets")
        if type(gpu_uuids) is not list or type(graph_buckets) is not list:
            raise TypeError("compile assignment plan tuple fields must be JSON arrays")
        value = cls(
            **payload,
            gpu_uuids=tuple(gpu_uuids),
            graph_buckets=tuple(graph_buckets),
        )
        value.validate()
        return value

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("compile assignment plan", str(path))
        raw, semantic_sha256 = _load_canonical_json_with_sidecar(
            source,
            label="compile assignment plan",
        )
        value = cls.from_dict(raw)
        if semantic_sha256 != value.sha256:
            raise ValueError("compile assignment plan semantic digest differs")
        value.revalidate()
        return value

    @classmethod
    def issue(
        cls,
        *,
        assignment_manifest_path: str | Path,
        compile_cache_plan_path: str | Path,
        prewarm_manifest_path: str | Path,
        launch_manifest_path: str | Path,
        result_pointer_path: str | Path,
        attempt_id: str,
    ) -> Self:
        assignment_path = _absolute_path(
            "assignment manifest", str(assignment_manifest_path)
        )
        cache_plan_path = _absolute_path(
            "compile cache plan", str(compile_cache_plan_path)
        )
        prewarm_path = _absolute_path("prewarm manifest", str(prewarm_manifest_path))
        launch_path = _absolute_path(
            "compile launch manifest", str(launch_manifest_path)
        )
        pointer_path = _absolute_path(
            "compile result pointer", str(result_pointer_path)
        )
        assignment = CompileOnlyAssignmentContract.load(assignment_path)
        cache_plan = CompileCacheLaunchPlan.load(cache_plan_path)
        prewarm = load_compile_prewarm_manifest(prewarm_path)
        launch = CompileLaunchManifest.load(launch_path)
        if cache_plan != assignment.compile_cache_plan:
            raise ValueError("compile cache plan differs from assignment authority")
        if prewarm != assignment.prewarm_manifest:
            raise ValueError(
                "compile prewarm manifest differs from assignment authority"
            )
        if pointer_path != Path(assignment.result_pointer_path):
            raise ValueError("compile result pointer differs from assignment authority")
        key = cache_plan.key
        if (
            launch.patched_sglang_tree != key.patched_sglang_tree
            or launch.compile_cache_plan_path != str(cache_plan_path)
            or launch.compile_cache_plan_sha256 != cache_plan.sha256
            or launch.prewarm_manifest_path != str(prewarm_path)
            or launch.prewarm_manifest_sha256 != prewarm.sha256
            or launch.target_revision != key.target_revision
            or launch.drafter_revision != key.drafter_revision
            or launch.model_lock_sha256 != prewarm.model_lock_sha256
            or launch.sampling_profile_sha256 != prewarm.sampling_profile_sha256
            or launch.physical_assignment_sha256
            != assignment.physical_assignment_sha256
            or launch.experiment_budget_sha256 != assignment.experiment_budget_sha256
            or launch.budget_materialization_authority_sha256
            != assignment.budget_materialization_authority_sha256
            or launch.inventory_sha256 != assignment.inventory_sha256
            or launch.gpu_uuids != assignment.gpu_uuids
        ):
            raise ValueError(
                "compile launch manifest differs from assignment authority"
            )
        launch_raw_sha256, launch_size = _raw_sha256(
            launch_path, label="compile launch manifest"
        )
        if launch_size < 1:
            raise ValueError("compile launch manifest is empty")
        if prewarm.model_lock_sha256 != assignment.prewarm_manifest.model_lock_sha256:
            raise ValueError("compile model lock differs from assignment authority")
        value = cls(
            schema_version=2,
            kind="first_party_compile_assignment_plan",
            protocol_sha256=COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256,
            assignment_manifest_path=str(assignment_path),
            assignment_sha256=assignment.sha256,
            compile_cache_plan_path=str(cache_plan_path),
            compile_cache_plan_sha256=cache_plan.sha256,
            prewarm_manifest_path=str(prewarm_path),
            prewarm_manifest_sha256=prewarm.sha256,
            launch_manifest_path=str(launch_path),
            launch_manifest_raw_sha256=launch_raw_sha256,
            launch_manifest_sha256=launch.sha256,
            compile_key_sha256=key.sha256,
            model_lock_sha256=prewarm.model_lock_sha256,
            target_revision=key.target_revision,
            drafter_revision=key.drafter_revision,
            physical_assignment_sha256=assignment.physical_assignment_sha256,
            experiment_budget_sha256=assignment.experiment_budget_sha256,
            inventory_sha256=assignment.inventory_sha256,
            gpu_uuids=assignment.gpu_uuids,
            host_id=assignment.host_id,
            tensor_parallel_size=key.tensor_parallel_size,
            context_limit=key.context_limit,
            max_running_requests=key.max_running_requests,
            graph_buckets=key.graph_buckets,
            graceful_shutdown_protocol_sha256=(
                assignment.graceful_shutdown_protocol_sha256
            ),
            result_pointer_protocol_sha256=(assignment.result_pointer_protocol_sha256),
            attempt_id=attempt_id,
            result_pointer_path=str(pointer_path),
        )
        value.validate()
        value.revalidate()
        return value

    def revalidate(
        self,
    ) -> tuple[
        CompileOnlyAssignmentContract,
        CompileCacheLaunchPlan,
        CompileOnlyPrewarmManifest,
        CompileLaunchManifest,
    ]:
        self.validate()
        assignment = CompileOnlyAssignmentContract.load(self.assignment_manifest_path)
        cache_plan = CompileCacheLaunchPlan.load(self.compile_cache_plan_path)
        prewarm = load_compile_prewarm_manifest(self.prewarm_manifest_path)
        launch_path = Path(self.launch_manifest_path)
        launch = CompileLaunchManifest.load(launch_path)
        if assignment.sha256 != self.assignment_sha256:
            raise ValueError("compile assignment changed during revalidation")
        if cache_plan.sha256 != self.compile_cache_plan_sha256:
            raise ValueError("compile cache plan changed during revalidation")
        if prewarm.sha256 != self.prewarm_manifest_sha256:
            raise ValueError("compile prewarm manifest changed during revalidation")
        launch_raw_sha256, launch_size = _raw_sha256(
            launch_path, label="compile launch manifest"
        )
        if (
            launch_size < 1
            or launch_raw_sha256 != self.launch_manifest_raw_sha256
            or launch.sha256 != self.launch_manifest_sha256
        ):
            raise ValueError("compile launch manifest changed during revalidation")
        if (
            cache_plan != assignment.compile_cache_plan
            or prewarm != assignment.prewarm_manifest
        ):
            raise ValueError("compile inputs no longer agree with assignment authority")
        key = cache_plan.key
        covered_graph_buckets = tuple(
            sorted({payload.graph_bucket for payload in prewarm.payloads})
        )
        if covered_graph_buckets != key.graph_buckets:
            raise ValueError(
                "compile prewarm manifest does not cover every registered graph bucket"
            )
        if (
            key.sha256 != self.compile_key_sha256
            or launch.patched_sglang_tree != key.patched_sglang_tree
            or launch.compile_cache_plan_path != self.compile_cache_plan_path
            or launch.compile_cache_plan_sha256 != self.compile_cache_plan_sha256
            or launch.prewarm_manifest_path != self.prewarm_manifest_path
            or launch.prewarm_manifest_sha256 != self.prewarm_manifest_sha256
            or launch.target_revision != key.target_revision
            or launch.drafter_revision != key.drafter_revision
            or launch.model_lock_sha256 != self.model_lock_sha256
            or launch.sampling_profile_sha256 != prewarm.sampling_profile_sha256
            or launch.physical_assignment_sha256 != self.physical_assignment_sha256
            or launch.experiment_budget_sha256 != self.experiment_budget_sha256
            or launch.budget_materialization_authority_sha256
            != assignment.budget_materialization_authority_sha256
            or launch.inventory_sha256 != self.inventory_sha256
            or launch.gpu_uuids != self.gpu_uuids
            or key.target_revision != self.target_revision
            or key.drafter_revision != self.drafter_revision
            or prewarm.model_lock_sha256 != self.model_lock_sha256
            or assignment.result_pointer_path != self.result_pointer_path
            or assignment.physical_assignment_sha256 != self.physical_assignment_sha256
            or assignment.experiment_budget_sha256 != self.experiment_budget_sha256
            or assignment.inventory_sha256 != self.inventory_sha256
            or assignment.gpu_uuids != self.gpu_uuids
            or assignment.host_id != self.host_id
            or key.tensor_parallel_size != self.tensor_parallel_size
            or key.context_limit != self.context_limit
            or key.max_running_requests != self.max_running_requests
            or key.graph_buckets != self.graph_buckets
            or assignment.graceful_shutdown_protocol_sha256
            != self.graceful_shutdown_protocol_sha256
            or assignment.result_pointer_protocol_sha256
            != self.result_pointer_protocol_sha256
        ):
            raise ValueError("compile assignment identity changed during revalidation")
        return assignment, cache_plan, prewarm, launch


COMPILE_CONTROL_VERIFICATION_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 2,
        "kind": "compile_control_verification_receipt",
        "control": "root_verified_dynamic_deployment_policy",
        "bindings": (
            "assignment_plan_and_inventory",
            "source_descriptor_and_launch_manifest",
            "durable_prepared_content_verification_receipt",
            "compile_control_subject_and_lineage",
            "artifact_deployment_and_fresh_additional_challenges",
            "atomic_no_replace_replay_reservation",
        ),
        "reopen": (
            "verify_compile_signatures_at_recorded_decision_time_reopen_"
            "reservation_and_deep_revalidate_previously_reserved_content_receipt"
        ),
        "forbidden": "reserving_the_prepared_content_challenge_twice",
    }
)


def compile_control_lineage_sha256(
    plan: CompileAssignmentPlan,
    source: CompileWorkerSourceDescriptor,
) -> str:
    """Bind the compile control signature to every executable launch input."""

    if type(plan) is not CompileAssignmentPlan:
        raise TypeError("compile control lineage requires an exact plan")
    if type(source) is not CompileWorkerSourceDescriptor:
        raise TypeError("compile control lineage requires an exact source descriptor")
    assignment, cache_plan, prewarm, launch = plan.revalidate()
    source.validate(reopen_sources=True)
    if (
        source.patched_sglang_checkout != launch.patched_sglang_checkout
        or source.patched_sglang_tree != launch.patched_sglang_tree
    ):
        raise ValueError("compile control source differs from launch manifest")
    return _content_sha256(
        {
            "schema_version": 1,
            "kind": "compile_control_lineage",
            "assignment_plan_sha256": plan.sha256,
            "assignment_contract_sha256": assignment.sha256,
            "cache_plan_sha256": cache_plan.sha256,
            "prewarm_manifest_sha256": prewarm.sha256,
            "launch_manifest_raw_sha256": plan.launch_manifest_raw_sha256,
            "launch_manifest_sha256": launch.sha256,
            "source_descriptor_sha256": source.sha256,
            "subprocess_protocol_sha256": (
                COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256
            ),
        }
    )


@dataclass(frozen=True)
class CompileControlVerificationReceipt:
    """Reopenable proof that dynamic release control was atomically consumed."""

    schema_version: int
    kind: str
    protocol_sha256: str
    verified_ns: int
    assignment_plan_sha256: str
    inventory_sha256: str
    lineage_sha256: str
    control_envelope: ControlArtifactAttestation
    verified_control: VerifiedControlArtifact
    source_descriptor: CompileWorkerSourceDescriptor
    prepared_content_verification_receipt: CanonicalJsonProofBinding
    prepared_content_authorization_sha256: str
    additional_challenge_sha256s: tuple[str, ...]
    reservation_sha256: str
    reservation_record_path: str
    reservation_record_raw_sha256: str
    reservation_record_size: int

    def validate(self, *, reopen_sources: bool) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 2
            or self.kind != "compile_control_verification_receipt"
            or self.protocol_sha256 != COMPILE_CONTROL_VERIFICATION_PROTOCOL_SHA256
        ):
            raise ValueError("compile control verification schema is unsupported")
        if type(self.verified_ns) is not int or self.verified_ns < 0:
            raise ValueError("compile control verification time is invalid")
        for label, digest in (
            ("assignment plan", self.assignment_plan_sha256),
            ("inventory", self.inventory_sha256),
            ("lineage", self.lineage_sha256),
            (
                "prepared content authorization",
                self.prepared_content_authorization_sha256,
            ),
            ("reservation", self.reservation_sha256),
            ("reservation raw", self.reservation_record_raw_sha256),
        ):
            _require_sha256(f"compile control {label}", digest)
        if type(self.control_envelope) is not ControlArtifactAttestation:
            raise TypeError("compile control receipt requires an exact envelope")
        if type(self.verified_control) is not VerifiedControlArtifact:
            raise TypeError("compile control receipt requires an exact verification")
        if type(self.source_descriptor) is not CompileWorkerSourceDescriptor:
            raise TypeError("compile control receipt requires an exact source")
        self.source_descriptor.validate(reopen_sources=reopen_sources)
        if type(self.prepared_content_verification_receipt) is not (
            CanonicalJsonProofBinding
        ):
            raise TypeError(
                "compile control receipt requires a prepared-content receipt binding"
            )
        if reopen_sources:
            prepared = _verified_prepared_release_from_content_receipt(
                self.prepared_content_verification_receipt,
                current_ns=self.verified_ns,
            )
            if (
                prepared.authorization_sha256
                != self.prepared_content_authorization_sha256
            ):
                raise ValueError(
                    "compile control prepared-content authorization changed"
                )
        if (
            type(self.additional_challenge_sha256s) is not tuple
            or tuple(sorted(set(self.additional_challenge_sha256s)))
            != self.additional_challenge_sha256s
        ):
            raise ValueError("compile control additional challenges are not canonical")
        for digest in self.additional_challenge_sha256s:
            _require_sha256("compile control additional challenge", digest)
        replayed = verify_release_control_artifact_attestation(
            self.control_envelope,
            expected_inventory_sha256=self.inventory_sha256,
            now_ns=self.verified_ns,
            consumed_challenge_sha256s=(),
        )
        if replayed != self.verified_control:
            raise ValueError("compile control verification result changed")
        if (
            replayed.artifact_type != "compile"
            or replayed.artifact_sha256 != self.assignment_plan_sha256
            or self.control_envelope.subject.protocol_sha256
            != COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256
            or self.control_envelope.subject.lineage_sha256 != self.lineage_sha256
        ):
            raise ValueError("compile control subject differs from assignment")
        expected_reservation = control_challenge_reservation_sha256(
            (replayed,),
            reserved_ns=self.verified_ns,
            additional_challenge_sha256s=self.additional_challenge_sha256s,
        )
        if expected_reservation != self.reservation_sha256:
            raise ValueError("compile control replay reservation identity changed")
        reservation_path = _absolute_path(
            "compile control reservation", self.reservation_record_path
        )
        if reservation_path.name != f"reservation-{self.reservation_sha256}.json":
            raise ValueError("compile control reservation path is not exact")
        if type(self.reservation_record_size) is not int or (
            self.reservation_record_size < 1
        ):
            raise ValueError("compile control reservation size is invalid")
        if reopen_sources:
            raw, size = _raw_sha256(
                reservation_path, label="compile control reservation"
            )
            if (
                raw != self.reservation_record_raw_sha256
                or size != self.reservation_record_size
            ):
                raise ValueError("compile control reservation changed")
            expected_challenges = tuple(
                sorted(
                    {
                        replayed.challenge_sha256,
                        replayed.deployment_policy_challenge_sha256,
                        *self.additional_challenge_sha256s,
                    }
                )
            )
            row = _strict_json_object(
                _stable_regular_file_bytes(
                    reservation_path, label="compile control reservation"
                ),
                label="compile control reservation",
            )
            if row != {
                "schema_version": 2,
                "kind": "lightcone_control_challenge_reservation",
                "reserved_ns": self.verified_ns,
                "challenge_sha256s": list(expected_challenges),
            }:
                raise ValueError("compile control reservation content differs")

    def to_dict(self) -> dict[str, object]:
        self.validate(reopen_sources=False)
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "verified_ns": self.verified_ns,
            "assignment_plan_sha256": self.assignment_plan_sha256,
            "inventory_sha256": self.inventory_sha256,
            "lineage_sha256": self.lineage_sha256,
            "control_envelope": self.control_envelope.to_dict(),
            "verified_control": asdict(self.verified_control),
            "source_descriptor": self.source_descriptor.to_dict(),
            "prepared_content_verification_receipt": (
                self.prepared_content_verification_receipt.to_dict()
            ),
            "prepared_content_authorization_sha256": (
                self.prepared_content_authorization_sha256
            ),
            "additional_challenge_sha256s": list(self.additional_challenge_sha256s),
            "reservation_sha256": self.reservation_sha256,
            "reservation_record_path": self.reservation_record_path,
            "reservation_record_raw_sha256": self.reservation_record_raw_sha256,
            "reservation_record_size": self.reservation_record_size,
        }

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or set(raw) != {
            field.name for field in dataclass_fields(cls)
        }:
            raise ValueError("compile control verification fields differ from schema")
        payload = dict(raw)
        envelope = ControlArtifactAttestation.from_dict(payload.pop("control_envelope"))
        verified_row = payload.pop("verified_control")
        if type(verified_row) is not dict:
            raise TypeError("compile verified control must be an object")
        verified = VerifiedControlArtifact(**verified_row)
        source = CompileWorkerSourceDescriptor.from_dict(
            payload.pop("source_descriptor")
        )
        prepared_content_receipt = CanonicalJsonProofBinding.from_dict(
            payload.pop("prepared_content_verification_receipt")
        )
        additional = payload.pop("additional_challenge_sha256s")
        if type(additional) is not list:
            raise TypeError("compile control additional challenges must be an array")
        value = cls(
            **payload,
            control_envelope=envelope,
            verified_control=verified,
            source_descriptor=source,
            prepared_content_verification_receipt=prepared_content_receipt,
            additional_challenge_sha256s=tuple(additional),
        )
        value.validate(reopen_sources=False)
        return value

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("compile control verification", str(path))
        raw, semantic = _load_canonical_json_with_sidecar(
            source, label="compile control verification"
        )
        value = cls.from_dict(raw)
        if value.sha256 != semantic:
            raise ValueError("compile control verification digest differs")
        value.validate(reopen_sources=True)
        return value


def _verified_prepared_release_from_content_receipt(
    binding: CanonicalJsonProofBinding,
    *,
    current_ns: int,
) -> VerifiedPreparedModelContentRelease:
    """Deep-reopen one durable content receipt without re-consuming challenges."""

    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError("prepared content receipt binding must be exact")
    raw = binding.reopen()
    receipt = ContentVerificationReceipt.from_dict(raw)
    if receipt.sha256 != binding.semantic_sha256:
        raise ValueError("prepared content receipt semantic identity differs")
    verified = receipt.revalidate(current_ns=current_ns)
    prepared = tuple(
        row for row in verified if type(row) is VerifiedPreparedModelContentRelease
    )
    if len(prepared) != 1:
        raise ValueError(
            "content verification receipt must contain one prepared-model authority"
        )
    return prepared[0]


def revalidate_prepared_content_verification_receipt(
    path: str | Path,
    *,
    current_ns: int,
) -> tuple[CanonicalJsonProofBinding, VerifiedPreparedModelContentRelease]:
    """Bind and deep-reopen durable prepared content for formal execution."""

    binding = CanonicalJsonProofBinding.bind(path)
    return binding, _verified_prepared_release_from_content_receipt(
        binding,
        current_ns=current_ns,
    )


def verify_and_reserve_compile_control(
    plan: CompileAssignmentPlan,
    envelope: ControlArtifactAttestation,
    *,
    prepared_content_verification_receipt_path: str | Path,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    additional_challenge_sha256s: tuple[str, ...] = (),
) -> CompileControlVerificationReceipt:
    """Consume one root-authorized compile control before spawning GPU work."""

    if type(plan) is not CompileAssignmentPlan:
        raise TypeError("compile control requires an exact assignment plan")
    (
        prepared_content_receipt,
        prepared_model_authorization,
    ) = revalidate_prepared_content_verification_receipt(
        prepared_content_verification_receipt_path,
        current_ns=now_ns,
    )
    assignment, _cache, _prewarm, launch = plan.revalidate()
    prepared_release = prepared_model_authorization.authorization
    target = prepared_model_authorization.member(launch.target_content_member_id)
    tokenizer = prepared_model_authorization.member(launch.tokenizer_content_member_id)
    drafter = (
        None
        if launch.drafter_content_member_id is None
        else prepared_model_authorization.member(launch.drafter_content_member_id)
    )
    expected_content_authority = prepared_model_authorization.authorization_sha256
    content_value, content_semantic = _load_canonical_json_with_sidecar(
        Path(launch.prepared_model_content_manifest_path),
        label="compile prepared-model content manifest",
    )
    if type(content_value) is not dict or set(content_value) != {
        "schema_version",
        "kind",
        "protocol_sha256",
        "model_lock_sha256",
        "prepared_model_set_sha256",
        "snapshots",
    }:
        raise ValueError("compile prepared-model content manifest schema differs")
    snapshots = content_value["snapshots"]
    if type(snapshots) is not list or any(type(row) is not dict for row in snapshots):
        raise TypeError("compile prepared-model snapshots must be objects")
    snapshot_by_identity = {
        (row.get("model_id"), row.get("revision")): row for row in snapshots
    }
    if len(snapshot_by_identity) != len(snapshots):
        raise ValueError("compile prepared-model snapshots are duplicated")
    launch_paths = {
        "target": launch.target_snapshot_path,
        "tokenizer": launch.tokenizer_snapshot_path,
    }
    if launch.drafter_snapshot_path is not None:
        launch_paths["drafter"] = launch.drafter_snapshot_path
    selected_models = (target, tokenizer) + (() if drafter is None else (drafter,))
    for authorized_model in selected_models:
        snapshot = snapshot_by_identity.get(
            (authorized_model.model_id, authorized_model.revision)
        )
        snapshot_sha256 = None if snapshot is None else _content_sha256(snapshot)
        if (
            snapshot is None
            or snapshot.get("root") != launch_paths.get(authorized_model.role)
            or authorized_model.snapshot_manifest_raw_sha256 != snapshot_sha256
            or authorized_model.snapshot_manifest_semantic_sha256 != snapshot_sha256
        ):
            raise ValueError(
                "compile prepared-model role differs from signed content manifest"
            )
    if (
        prepared_release.model_lock_sha256 != launch.model_lock_sha256
        or content_value["model_lock_sha256"] != launch.model_lock_sha256
        or content_value["prepared_model_set_sha256"]
        != prepared_release.prepared_model_set_sha256
        or prepared_release.content_manifest_raw_sha256
        != launch.prepared_model_content_manifest_raw_sha256
        or prepared_release.content_manifest_semantic_sha256
        != launch.prepared_model_content_manifest_sha256
        or prepared_release.content_manifest_size
        != launch.prepared_model_content_manifest_size
        or content_semantic != launch.prepared_model_content_manifest_sha256
        or launch.target_content_authority_sha256 != expected_content_authority
        or launch.tokenizer_content_authority_sha256 != expected_content_authority
        or (launch.target_model_id, launch.target_revision)
        != (target.model_id, target.revision)
        or target.role != "target"
        or (launch.tokenizer_model_id, launch.tokenizer_revision)
        != (tokenizer.model_id, tokenizer.revision)
        or tokenizer.role != "tokenizer"
        or ((launch.drafter_model_id is None) != (drafter is None))
        or (
            drafter is not None
            and (
                launch.drafter_content_authority_sha256 != expected_content_authority
                or (launch.drafter_model_id, launch.drafter_revision)
                != (drafter.model_id, drafter.revision)
                or drafter.role != "drafter"
            )
        )
    ):
        raise ValueError(
            "compile launch differs from prepared-model content authorization"
        )
    all_additional_challenges = tuple(sorted(set(additional_challenge_sha256s)))
    if len(all_additional_challenges) != len(additional_challenge_sha256s):
        raise ValueError("compile control additional challenges are duplicated")
    if prepared_model_authorization.challenge_sha256 in all_additional_challenges:
        raise ValueError(
            "compile control must not reserve the prepared-content challenge twice"
        )
    source = CompileWorkerSourceDescriptor.issue(
        patched_sglang_checkout=launch.patched_sglang_checkout
    )
    lineage = compile_control_lineage_sha256(plan, source)
    subject = envelope.subject
    if (
        subject.artifact_type != "compile"
        or subject.artifact_sha256 != plan.sha256
        or subject.protocol_sha256 != COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256
        or subject.registry_sha256 != assignment.registry_sha256
        or subject.lineage_sha256 != lineage
    ):
        raise ValueError("compile control envelope does not authorize this launch")
    results = verify_and_reserve_release_control_artifact_attestations(
        (envelope,),
        expected_inventory_sha256=plan.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=all_additional_challenges,
    )
    verified = results[0]
    reservation = control_challenge_reservation_sha256(
        results,
        reserved_ns=now_ns,
        additional_challenge_sha256s=all_additional_challenges,
    )
    reservation_path = Path(replay_store.root) / f"reservation-{reservation}.json"
    raw_sha256, size = _raw_sha256(
        reservation_path, label="compile control reservation"
    )
    receipt = CompileControlVerificationReceipt(
        schema_version=2,
        kind="compile_control_verification_receipt",
        protocol_sha256=COMPILE_CONTROL_VERIFICATION_PROTOCOL_SHA256,
        verified_ns=now_ns,
        assignment_plan_sha256=plan.sha256,
        inventory_sha256=plan.inventory_sha256,
        lineage_sha256=lineage,
        control_envelope=envelope,
        verified_control=verified,
        source_descriptor=source,
        prepared_content_verification_receipt=prepared_content_receipt,
        prepared_content_authorization_sha256=(
            prepared_model_authorization.authorization_sha256
        ),
        additional_challenge_sha256s=all_additional_challenges,
        reservation_sha256=reservation,
        reservation_record_path=str(reservation_path),
        reservation_record_raw_sha256=raw_sha256,
        reservation_record_size=size,
    )
    receipt.validate(reopen_sources=True)
    return receipt


def require_release_compile_assignment_plan(
    plan: CompileAssignmentPlan | None = None,
) -> ReleaseCompileSubprocess:
    """Return exact source authority or block before cache/process mutation.

    The empty subprocess and plan allowlists are checked before any serialized
    plan path needs to be opened.  This ordering is deliberate: a diagnostic
    plan, however complete, cannot become formal launch authority.
    """

    if len(RELEASE_COMPILE_SUBPROCESSES) != 1:
        raise CompileRunnerBlocked(RELEASE_COMPILE_RUNNER_UNAVAILABLE)
    if not RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_ASSIGNMENT_PLAN_ALLOWLIST_EMPTY)
    command = RELEASE_COMPILE_SUBPROCESSES[0]
    command.validate(reopen_executable=True)
    if not RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_GPU_SOURCE_REGISTRY_EMPTY)
    if command.sha256 not in RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_GPU_SOURCE_UNTRUSTED)
    if plan is None:
        raise CompileRunnerBlocked(RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE)
    if type(plan) is not CompileAssignmentPlan:
        raise TypeError("release compile runner requires an exact assignment plan")
    plan.validate()
    if plan.sha256 not in RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_ASSIGNMENT_PLAN_UNTRUSTED)
    return command


@dataclass(frozen=True)
class CompilePrewarmObservation:
    request_id: str
    graph_bucket: int
    completed: bool
    provider_receipt_sha256: str

    def validate(self) -> None:
        _require_text("prewarm observation request", self.request_id)
        if type(self.graph_bucket) is not int or self.graph_bucket < 1:
            raise ValueError("prewarm observation graph bucket must be positive")
        if self.completed is not True:
            raise ValueError("compile prewarm request did not complete")
        _require_sha256("prewarm provider receipt", self.provider_receipt_sha256)


@dataclass(frozen=True)
class CompileShutdownObservation:
    process_id: int
    shutdown_requested_ns: int
    process_exited_ns: int
    exit_code: int
    active_requests: int
    queued_requests: int
    provider_ack_sha256: str

    def validate(self) -> None:
        for label, value in (
            ("process ID", self.process_id),
            ("shutdown request", self.shutdown_requested_ns),
            ("process exit", self.process_exited_ns),
            ("active requests", self.active_requests),
            ("queued requests", self.queued_requests),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"compile shutdown {label} is invalid")
        if self.process_id < 1 or self.process_exited_ns < self.shutdown_requested_ns:
            raise ValueError("compile shutdown ordering is invalid")
        if type(self.exit_code) is not int or self.exit_code != 0:
            raise ValueError("compile process did not exit successfully")
        if self.active_requests != 0 or self.queued_requests != 0:
            raise ValueError("compile shutdown acknowledgement is not drained")
        _require_sha256(
            "compile shutdown provider acknowledgement", self.provider_ack_sha256
        )


class CompileLifecycleDriver(Protocol):
    process_id: int

    def start(self, environment: Mapping[str, str]) -> None: ...

    def prewarm(
        self, payload: CompileOnlyPrewarmPayload
    ) -> CompilePrewarmObservation: ...

    def graceful_shutdown(self) -> CompileShutdownObservation: ...


@dataclass(frozen=True)
class CompileSubprocessEvent:
    sequence: int
    direction: str
    canonical_json: str
    raw_sha256: str

    def validate(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("compile subprocess event sequence is invalid")
        if self.direction not in {"parent_to_child", "child_to_parent"}:
            raise ValueError("compile subprocess event direction is invalid")
        if type(self.canonical_json) is not str or "\n" in self.canonical_json:
            raise ValueError("compile subprocess event must be one JSON line")
        raw = f"{self.canonical_json}\n".encode()
        _strict_json_object(raw, label="compile subprocess event")
        if hashlib.sha256(raw).hexdigest() != self.raw_sha256:
            raise ValueError("compile subprocess event raw SHA-256 differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or set(raw) != {
            "sequence",
            "direction",
            "canonical_json",
            "raw_sha256",
        }:
            raise ValueError("compile subprocess event fields differ from schema")
        value = cls(**raw)
        value.validate()
        return value


@dataclass(frozen=True)
class CompileSubprocessLifecycleReceipt:
    schema_version: int
    kind: str
    protocol_sha256: str
    assignment_plan_sha256: str
    executable_path: str
    executable_raw_sha256: str
    executable_size: int
    argv_sha256: str
    source_authority_sha256: str | None
    launch_manifest_path: str
    launch_manifest_raw_sha256: str
    launch_manifest_sha256: str
    control_verification_receipt_sha256: str | None
    process_id: int
    process_started_ns: int
    process_exited_ns: int
    exit_code: int
    events: tuple[CompileSubprocessEvent, ...]
    formal_execution_authorized: bool

    def validate(self, *, reopen_executable: bool) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 2
            or self.kind != "compile_subprocess_lifecycle_raw_receipt"
        ):
            raise ValueError("compile subprocess receipt schema is unsupported")
        if self.protocol_sha256 != COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256:
            raise ValueError("compile subprocess receipt uses another protocol")
        _require_sha256("compile subprocess plan", self.assignment_plan_sha256)
        executable = _absolute_path(
            "compile subprocess executable", self.executable_path
        )
        _require_sha256("compile subprocess executable", self.executable_raw_sha256)
        if type(self.executable_size) is not int or self.executable_size <= 0:
            raise ValueError("compile subprocess executable size is invalid")
        _require_sha256("compile subprocess argv", self.argv_sha256)
        if self.source_authority_sha256 is not None:
            _require_sha256(
                "compile subprocess source authority", self.source_authority_sha256
            )
        launch_path = _absolute_path(
            "compile subprocess launch manifest", self.launch_manifest_path
        )
        _require_sha256(
            "compile subprocess launch manifest raw", self.launch_manifest_raw_sha256
        )
        _require_sha256(
            "compile subprocess launch manifest", self.launch_manifest_sha256
        )
        if self.control_verification_receipt_sha256 is not None:
            _require_sha256(
                "compile subprocess control verification",
                self.control_verification_receipt_sha256,
            )
        for label, value in (
            ("process ID", self.process_id),
            ("process start", self.process_started_ns),
            ("process exit", self.process_exited_ns),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"compile subprocess {label} is invalid")
        if self.process_exited_ns < self.process_started_ns:
            raise ValueError("compile subprocess receipt time order is invalid")
        if type(self.exit_code) is not int or self.exit_code != 0:
            raise ValueError("compile subprocess receipt requires zero exit")
        if type(self.events) is not tuple or not self.events:
            raise TypeError("compile subprocess receipt requires exact events")
        for event in self.events:
            if type(event) is not CompileSubprocessEvent:
                raise TypeError("compile subprocess receipt event type is invalid")
            event.validate()
        if tuple(event.sequence for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("compile subprocess receipt event sequence is incomplete")
        event_rows = tuple(json.loads(event.canonical_json) for event in self.events)
        event_kinds = tuple(row.get("kind") for row in event_rows)
        if event_kinds[:3] != (
            "compile_subprocess_ready",
            "compile_subprocess_start",
            "compile_subprocess_started",
        ) or event_kinds[-2:] != (
            "compile_subprocess_shutdown",
            "compile_subprocess_drained",
        ):
            raise ValueError("compile subprocess receipt lifecycle is incomplete")
        if tuple(event.direction for event in self.events[:3]) != (
            "child_to_parent",
            "parent_to_child",
            "child_to_parent",
        ) or tuple(event.direction for event in self.events[-2:]) != (
            "parent_to_child",
            "child_to_parent",
        ):
            raise ValueError("compile subprocess receipt lifecycle direction differs")
        middle = self.events[3:-2]
        if not middle or len(middle) % 2:
            raise ValueError(
                "compile subprocess receipt prewarm exchange is incomplete"
            )
        for request, response in zip(middle[::2], middle[1::2], strict=True):
            if request.direction != "parent_to_child" or response.direction != (
                "child_to_parent"
            ):
                raise ValueError("compile subprocess prewarm direction differs")
            request_row = json.loads(request.canonical_json)
            response_row = json.loads(response.canonical_json)
            if (
                request_row.get("kind") != "compile_subprocess_prewarm"
                or response_row.get("kind") != "compile_subprocess_prewarm_complete"
                or request_row.get("request_id") != response_row.get("request_id")
                or request_row.get("graph_bucket") != response_row.get("graph_bucket")
            ):
                raise ValueError("compile subprocess prewarm exchange differs")
        if self.formal_execution_authorized is True:
            if (
                self.source_authority_sha256 is None
                or self.control_verification_receipt_sha256 is None
            ):
                raise ValueError(
                    "formal compile receipt lacks dynamic control authority"
                )
        elif self.formal_execution_authorized is not False:
            raise TypeError("compile subprocess formal flag must be boolean")
        elif (
            self.source_authority_sha256 is not None
            or self.control_verification_receipt_sha256 is not None
        ):
            raise ValueError(
                "diagnostic compile receipt cannot claim control authority"
            )
        start_row = event_rows[1]
        start_environment = start_row.get("cache_environment")
        start_launch = start_row.get("launch_manifest")
        if (
            type(start_environment) is not dict
            or set(start_environment) != set(COMPILE_CACHE_ENVIRONMENT_VARIABLES)
            or any(
                type(value) is not str
                or not Path(value).is_absolute()
                or Path(value) != Path(value).resolve(strict=False)
                for value in start_environment.values()
            )
        ):
            raise ValueError("compile subprocess receipt cache environment differs")
        if start_launch != {
            "path": self.launch_manifest_path,
            "raw_sha256": self.launch_manifest_raw_sha256,
            "semantic_sha256": self.launch_manifest_sha256,
        }:
            raise ValueError("compile subprocess receipt launch manifest differs")
        if reopen_executable:
            launch = CompileLaunchManifest.load(launch_path)
            launch_raw, launch_size = _raw_sha256(
                launch_path, label="compile subprocess launch manifest"
            )
            if (
                launch_size < 1
                or launch_raw != self.launch_manifest_raw_sha256
                or launch.sha256 != self.launch_manifest_sha256
            ):
                raise ValueError("compile subprocess launch manifest changed")
            digest, size = _raw_sha256(
                executable, label="compile subprocess executable"
            )
            if digest != self.executable_raw_sha256 or size != self.executable_size:
                raise ValueError(
                    "compile subprocess executable changed after execution"
                )

    def to_dict(self) -> dict[str, object]:
        self.validate(reopen_executable=False)
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "assignment_plan_sha256": self.assignment_plan_sha256,
            "executable_path": self.executable_path,
            "executable_raw_sha256": self.executable_raw_sha256,
            "executable_size": self.executable_size,
            "argv_sha256": self.argv_sha256,
            "source_authority_sha256": self.source_authority_sha256,
            "launch_manifest_path": self.launch_manifest_path,
            "launch_manifest_raw_sha256": self.launch_manifest_raw_sha256,
            "launch_manifest_sha256": self.launch_manifest_sha256,
            "control_verification_receipt_sha256": (
                self.control_verification_receipt_sha256
            ),
            "process_id": self.process_id,
            "process_started_ns": self.process_started_ns,
            "process_exited_ns": self.process_exited_ns,
            "exit_code": self.exit_code,
            "events": [event.to_dict() for event in self.events],
            "formal_execution_authorized": self.formal_execution_authorized,
        }

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        expected = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "assignment_plan_sha256",
            "executable_path",
            "executable_raw_sha256",
            "executable_size",
            "argv_sha256",
            "source_authority_sha256",
            "launch_manifest_path",
            "launch_manifest_raw_sha256",
            "launch_manifest_sha256",
            "control_verification_receipt_sha256",
            "process_id",
            "process_started_ns",
            "process_exited_ns",
            "exit_code",
            "events",
            "formal_execution_authorized",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ValueError("compile subprocess receipt fields differ from schema")
        payload = dict(raw)
        events = payload.pop("events")
        if type(events) is not list:
            raise TypeError("compile subprocess receipt events must be a JSON array")
        value = cls(
            **payload,
            events=tuple(CompileSubprocessEvent.from_dict(event) for event in events),
        )
        value.validate(reopen_executable=False)
        return value

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("compile subprocess receipt", str(path))
        raw, semantic_sha256 = _load_canonical_json_with_sidecar(
            source,
            label="compile subprocess receipt",
        )
        value = cls.from_dict(raw)
        if semantic_sha256 != value.sha256:
            raise ValueError("compile subprocess receipt semantic digest differs")
        value.validate(reopen_executable=True)
        return value


class _CompileSubprocessDriver:
    """Bounded JSON-lines client for the first-party compile wrapper."""

    def __init__(
        self,
        *,
        argv: tuple[str, ...],
        plan: CompileAssignmentPlan,
        timeout_seconds: float,
        source_authority_sha256: str | None,
        control_verification_receipt_sha256: str | None,
        formal_execution_authorized: bool,
    ) -> None:
        if type(argv) is not tuple or not argv:
            raise TypeError("compile subprocess argv must be a non-empty tuple")
        for argument in argv:
            if type(argument) is not str or not argument or "\x00" in argument:
                raise ValueError("compile subprocess argument contains NUL")
        executable = _absolute_path("compile subprocess executable", argv[0])
        digest, size = _raw_sha256(executable, label="compile subprocess executable")
        if type(plan) is not CompileAssignmentPlan:
            raise TypeError("compile subprocess requires an exact assignment plan")
        plan.validate()
        launch = CompileLaunchManifest.load(plan.launch_manifest_path)
        if launch.sha256 != plan.launch_manifest_sha256:
            raise ValueError("compile subprocess launch manifest differs from plan")
        if (
            type(timeout_seconds) not in {int, float}
            or isinstance(timeout_seconds, bool)
            or not (0 < float(timeout_seconds) <= 3_600)
        ):
            raise ValueError("compile subprocess timeout must be in (0, 3600] seconds")
        if formal_execution_authorized is True:
            if (
                source_authority_sha256 is None
                or control_verification_receipt_sha256 is None
            ):
                raise ValueError("formal compile subprocess lacks dynamic control")
            _require_sha256(
                "compile subprocess source authority", source_authority_sha256
            )
            _require_sha256(
                "compile subprocess control verification",
                control_verification_receipt_sha256,
            )
        elif formal_execution_authorized is not False:
            raise TypeError("compile subprocess formal flag must be boolean")
        elif (
            source_authority_sha256 is not None
            or control_verification_receipt_sha256 is not None
        ):
            raise ValueError("diagnostic compile subprocess cannot claim authority")
        self.argv = argv
        self.assignment_plan_sha256 = plan.sha256
        self.launch = launch
        self.launch_manifest_path = plan.launch_manifest_path
        self.launch_manifest_raw_sha256 = plan.launch_manifest_raw_sha256
        self.launch_manifest_sha256 = plan.launch_manifest_sha256
        self.timeout_seconds = float(timeout_seconds)
        self.executable_path = executable
        self.executable_raw_sha256 = digest
        self.executable_size = size
        self.argv_sha256 = _content_sha256({"argv": list(argv)})
        self.source_authority_sha256 = source_authority_sha256
        self.control_verification_receipt_sha256 = control_verification_receipt_sha256
        self.formal_execution_authorized = formal_execution_authorized
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_buffer = b""
        self._events: list[CompileSubprocessEvent] = []
        self._process_started_ns: int | None = None
        self._process_exited_ns: int | None = None
        self._exit_code: int | None = None
        self._deadline_monotonic: float | None = None

    @property
    def process_id(self) -> int:
        if self._process is None or self._process.pid is None:
            raise RuntimeError("compile subprocess has not been spawned")
        return self._process.pid

    @staticmethod
    def _encoded_message(value: Mapping[str, object]) -> bytes:
        return (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    def _record(self, direction: str, encoded: bytes) -> None:
        if len(encoded) > _MAX_SUBPROCESS_MESSAGE_BYTES:
            raise ValueError("compile subprocess protocol message is too large")
        row = _strict_json_object(encoded, label="compile subprocess protocol message")
        canonical_json = encoded[:-1].decode("utf-8")
        if row.get("protocol_sha256") != COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256:
            raise ValueError("compile subprocess message uses another protocol")
        if row.get("assignment_plan_sha256") != self.assignment_plan_sha256:
            raise ValueError("compile subprocess message names another plan")
        event = CompileSubprocessEvent(
            sequence=len(self._events),
            direction=direction,
            canonical_json=canonical_json,
            raw_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        event.validate()
        self._events.append(event)

    def _send(self, value: Mapping[str, object]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("compile subprocess stdin is unavailable")
        encoded = self._encoded_message(value)
        self._record("parent_to_child", encoded)
        try:
            self._process.stdin.write(encoded)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(
                "compile subprocess closed its command channel"
            ) from error

    def _read_line(self) -> bytes:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("compile subprocess stdout is unavailable")
        if self._deadline_monotonic is None:
            raise RuntimeError("compile subprocess deadline is unavailable")
        deadline = self._deadline_monotonic
        descriptor = self._process.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while b"\n" not in self._stdout_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("compile subprocess protocol response timed out")
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    raise RuntimeError(
                        "compile subprocess exited before its protocol response"
                    )
                self._stdout_buffer += chunk
                if len(self._stdout_buffer) > _MAX_SUBPROCESS_MESSAGE_BYTES:
                    raise ValueError(
                        "compile subprocess protocol response is too large"
                    )
        encoded, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
        encoded += b"\n"
        self._record("child_to_parent", encoded)
        return encoded

    def _receive(self, *, kind: str, fields: set[str]) -> dict[str, object]:
        encoded = self._read_line()
        row = _strict_json_object(encoded, label="compile subprocess response")
        expected = {
            "kind",
            "protocol_sha256",
            "assignment_plan_sha256",
            *fields,
        }
        if set(row) != expected:
            raise ValueError("compile subprocess response fields differ from protocol")
        if row["kind"] != kind:
            raise ValueError("compile subprocess response kind is out of order")
        return row

    def spawn(self) -> None:
        if self._process is not None:
            raise RuntimeError("compile subprocess was already spawned")
        # Never pass caller credentials, provider tokens, Python injection,
        # or unregistered cache paths into the compile child.  A future GPU
        # command needing another variable must bind it in source policy.
        environment = {
            **self.launch.child_environment(),
            # The worker must echo the exact routing identity before it may
            # receive the start manifest.  Putting this parent-validated digest
            # in the otherwise explicit environment avoids an assignment ↔
            # launch-manifest digest cycle; it is neither a secret nor launch
            # authority.
            "LIGHTCONE_COMPILE_ASSIGNMENT_PLAN_SHA256": (self.assignment_plan_sha256),
        }
        self._process_started_ns = time.monotonic_ns()
        self._deadline_monotonic = time.monotonic() + self.timeout_seconds
        try:
            self._process = subprocess.Popen(
                self.argv,
                executable=str(self.executable_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
            )
            ready = self._receive(
                kind="compile_subprocess_ready", fields={"process_id"}
            )
            if type(ready["process_id"]) is not int or ready["process_id"] != (
                self.process_id
            ):
                raise ValueError(
                    "compile subprocess ready message names another process"
                )
        except BaseException:
            self.abort()
            raise

    def start(self, environment: Mapping[str, str]) -> None:
        cache_environment: dict[str, str] = {}
        for name in COMPILE_CACHE_ENVIRONMENT_VARIABLES:
            value = environment.get(name)
            if type(value) is not str:
                raise ValueError("compile subprocess lacks private cache environment")
            path = _absolute_path(f"compile subprocess {name}", value)
            cache_environment[name] = str(path)
        self._send(
            {
                "kind": "compile_subprocess_start",
                "protocol_sha256": COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
                "assignment_plan_sha256": self.assignment_plan_sha256,
                "cache_environment": cache_environment,
                "launch_manifest": {
                    "path": self.launch_manifest_path,
                    "raw_sha256": self.launch_manifest_raw_sha256,
                    "semantic_sha256": self.launch_manifest_sha256,
                },
            }
        )
        started = self._receive(
            kind="compile_subprocess_started", fields={"process_id"}
        )
        if type(started["process_id"]) is not int or started["process_id"] != (
            self.process_id
        ):
            raise ValueError(
                "compile subprocess start acknowledgement names another process"
            )

    def prewarm(self, payload: CompileOnlyPrewarmPayload) -> CompilePrewarmObservation:
        payload.validate()
        self._send(
            {
                "kind": "compile_subprocess_prewarm",
                "protocol_sha256": COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
                "assignment_plan_sha256": self.assignment_plan_sha256,
                **payload.to_dict(),
            }
        )
        response = self._receive(
            kind="compile_subprocess_prewarm_complete",
            fields={
                "request_id",
                "graph_bucket",
                "completed",
                "provider_receipt_sha256",
            },
        )
        observation = CompilePrewarmObservation(
            request_id=response["request_id"],
            graph_bucket=response["graph_bucket"],
            completed=response["completed"],
            provider_receipt_sha256=response["provider_receipt_sha256"],
        )
        observation.validate()
        return observation

    def _assert_stdout_exhausted(self) -> None:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("compile subprocess stdout is unavailable")
        remainder = self._stdout_buffer + self._process.stdout.read()
        self._stdout_buffer = b""
        if remainder:
            raise ValueError(
                "compile subprocess emitted output after drain acknowledgement"
            )

    def graceful_shutdown(self) -> CompileShutdownObservation:
        if self._process is None:
            raise RuntimeError("compile subprocess was not spawned")
        requested_ns = time.monotonic_ns()
        self._send(
            {
                "kind": "compile_subprocess_shutdown",
                "protocol_sha256": COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
                "assignment_plan_sha256": self.assignment_plan_sha256,
            }
        )
        response = self._receive(
            kind="compile_subprocess_drained",
            fields={"active_requests", "queued_requests", "provider_ack_sha256"},
        )
        if self._deadline_monotonic is None:
            raise RuntimeError("compile subprocess deadline is unavailable")
        remaining = self._deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("compile subprocess exceeded its process hard cap")
        try:
            exit_code = self._process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("compile subprocess did not exit after drain") from error
        self._process_exited_ns = time.monotonic_ns()
        self._exit_code = exit_code
        self._assert_stdout_exhausted()
        try:
            os.killpg(self.process_id, 0)
        except ProcessLookupError:
            pass
        else:
            self.abort()
            raise ValueError("compile subprocess left a live child process group")
        observation = CompileShutdownObservation(
            process_id=self.process_id,
            shutdown_requested_ns=requested_ns,
            process_exited_ns=self._process_exited_ns,
            exit_code=exit_code,
            active_requests=response["active_requests"],
            queued_requests=response["queued_requests"],
            provider_ack_sha256=response["provider_ack_sha256"],
        )
        observation.validate()
        return observation

    def receipt(self) -> CompileSubprocessLifecycleReceipt:
        if (
            self._process is None
            or self._process_started_ns is None
            or self._process_exited_ns is None
            or self._exit_code is None
        ):
            raise RuntimeError("compile subprocess is not terminal")
        receipt = CompileSubprocessLifecycleReceipt(
            schema_version=2,
            kind="compile_subprocess_lifecycle_raw_receipt",
            protocol_sha256=COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
            assignment_plan_sha256=self.assignment_plan_sha256,
            executable_path=str(self.executable_path),
            executable_raw_sha256=self.executable_raw_sha256,
            executable_size=self.executable_size,
            argv_sha256=self.argv_sha256,
            source_authority_sha256=self.source_authority_sha256,
            launch_manifest_path=self.launch_manifest_path,
            launch_manifest_raw_sha256=self.launch_manifest_raw_sha256,
            launch_manifest_sha256=self.launch_manifest_sha256,
            control_verification_receipt_sha256=(
                self.control_verification_receipt_sha256
            ),
            process_id=self.process_id,
            process_started_ns=self._process_started_ns,
            process_exited_ns=self._process_exited_ns,
            exit_code=self._exit_code,
            events=tuple(self._events),
            formal_execution_authorized=self.formal_execution_authorized,
        )
        receipt.validate(reopen_executable=True)
        return receipt

    def abort(self) -> None:
        process = self._process
        if process is None:
            return

        def group_exists() -> bool:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True

        def wait_for_group(deadline: float) -> bool:
            while group_exists():
                process.poll()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(0.01, remaining))
            process.poll()
            return True

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.poll()
            return
        except OSError:
            pass
        term_deadline = time.monotonic() + min(self.timeout_seconds, 2.0)
        if wait_for_group(term_deadline):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.poll()
            return
        kill_deadline = time.monotonic() + 2.0
        if not wait_for_group(kill_deadline):
            raise RuntimeError("compile subprocess process group survived SIGKILL")


@dataclass(frozen=True)
class CompileResultBinding:
    absolute_path: str
    raw_sha256: str
    size: int

    @classmethod
    def bind(cls, path: Path, *, label: str) -> Self:
        normalized = _absolute_path(label, str(path))
        digest, size = _raw_sha256(normalized, label=label)
        return cls(str(normalized), digest, size)

    def reopen(self, *, label: str) -> None:
        digest, size = _raw_sha256(
            relocated_evidence_path(_absolute_path(label, self.absolute_path)),
            label=label,
        )
        if digest != self.raw_sha256 or size != self.size:
            raise ValueError(f"{label} changed after terminal publication")

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or set(raw) != {
            "absolute_path",
            "raw_sha256",
            "size",
        }:
            raise ValueError("compile result binding fields differ from schema")
        value = cls(**raw)
        _absolute_path("compile result binding", value.absolute_path)
        _require_sha256("compile result binding", value.raw_sha256)
        if type(value.size) is not int or value.size <= 0:
            raise ValueError("compile result binding size is invalid")
        return value


@dataclass(frozen=True)
class CompileResultPointer:
    schema_version: int
    kind: str
    result_pointer_protocol_sha256: str
    assignment_plan_sha256: str
    assignment_manifest: CompileResultBinding
    compile_cache_plan: CompileResultBinding
    prewarm_manifest: CompileResultBinding
    launch_manifest: CompileResultBinding
    attempt_receipt: CompileResultBinding
    graceful_shutdown_receipt: CompileResultBinding
    final_cache_receipt: CompileResultBinding
    immutable_cache_object_manifest: CompileResultBinding
    formal_execution_authorized: bool
    assignment_plan_source: CompileResultBinding | None = None
    subprocess_lifecycle_receipt: CompileResultBinding | None = None
    control_verification_receipt: CompileResultBinding | None = None

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version not in {1, 2, 3}
            or self.kind != "compile_atomic_result_pointer"
        ):
            raise ValueError("compile result pointer schema is unsupported")
        if (
            self.result_pointer_protocol_sha256
            != COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256
        ):
            raise ValueError("compile result pointer uses another protocol")
        _require_sha256("compile result assignment plan", self.assignment_plan_sha256)
        if self.schema_version == 1:
            if (
                self.assignment_plan_source is not None
                or self.subprocess_lifecycle_receipt is not None
                or self.control_verification_receipt is not None
            ):
                raise ValueError(
                    "legacy compile pointer cannot claim subprocess evidence"
                )
        elif (
            type(self.assignment_plan_source) is not CompileResultBinding
            or type(self.subprocess_lifecycle_receipt) is not CompileResultBinding
        ):
            raise TypeError("subprocess compile pointer lacks path-bound raw evidence")
        if self.schema_version == 2 and self.control_verification_receipt is not None:
            raise ValueError("diagnostic subprocess pointer cannot claim control")
        if (
            self.schema_version == 3
            and type(self.control_verification_receipt) is not CompileResultBinding
        ):
            raise TypeError("formal compile pointer lacks dynamic control evidence")
        if type(self.formal_execution_authorized) is not bool:
            raise TypeError("compile result formal flag must be boolean")
        if self.formal_execution_authorized is True and self.schema_version != 3:
            raise ValueError(
                "formal compile execution requires subprocess lifecycle evidence"
            )
        for label, binding in self.bindings().items():
            if type(binding) is not CompileResultBinding:
                raise TypeError(f"compile result {label} binding is invalid")
            _absolute_path(label, binding.absolute_path)
            _require_sha256(label, binding.raw_sha256)
            if type(binding.size) is not int or binding.size <= 0:
                raise ValueError(f"compile result {label} size is invalid")

    def bindings(self) -> dict[str, CompileResultBinding]:
        bindings = {
            "assignment_manifest": self.assignment_manifest,
            "compile_cache_plan": self.compile_cache_plan,
            "prewarm_manifest": self.prewarm_manifest,
            "launch_manifest": self.launch_manifest,
            "attempt_receipt": self.attempt_receipt,
            "graceful_shutdown_receipt": self.graceful_shutdown_receipt,
            "final_cache_receipt": self.final_cache_receipt,
            "immutable_cache_object_manifest": self.immutable_cache_object_manifest,
        }
        if self.assignment_plan_source is not None:
            bindings["assignment_plan_source"] = self.assignment_plan_source
        if self.subprocess_lifecycle_receipt is not None:
            bindings["subprocess_lifecycle_receipt"] = self.subprocess_lifecycle_receipt
        if self.control_verification_receipt is not None:
            bindings["control_verification_receipt"] = self.control_verification_receipt
        return bindings

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "result_pointer_protocol_sha256": self.result_pointer_protocol_sha256,
            "assignment_plan_sha256": self.assignment_plan_sha256,
            "formal_execution_authorized": self.formal_execution_authorized,
            **{label: asdict(binding) for label, binding in self.bindings().items()},
        }
        return payload

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def reopen(self) -> None:
        self.validate()
        for label, binding in self.bindings().items():
            binding.reopen(label=f"compile result {label}")
        if self.schema_version in {2, 3}:
            if self.assignment_plan_source is None:
                raise AssertionError("validated subprocess pointer lost its plan")
            if self.subprocess_lifecycle_receipt is None:
                raise AssertionError("validated subprocess pointer lost its receipt")
            plan = CompileAssignmentPlan.load(self.assignment_plan_source.absolute_path)
            if plan.sha256 != self.assignment_plan_sha256:
                raise ValueError("compile pointer assignment-plan binding differs")
            receipt = CompileSubprocessLifecycleReceipt.load(
                self.subprocess_lifecycle_receipt.absolute_path
            )
            if (
                receipt.assignment_plan_sha256 != self.assignment_plan_sha256
                or receipt.formal_execution_authorized
                is not self.formal_execution_authorized
            ):
                raise ValueError("compile pointer subprocess receipt differs")
            plan_launch = CompileResultBinding.bind(
                Path(plan.launch_manifest_path), label="compile launch manifest"
            )
            if self.launch_manifest != plan_launch:
                raise ValueError("compile pointer launch manifest differs")
            if self.schema_version == 3:
                if self.control_verification_receipt is None:
                    raise AssertionError("formal compile pointer lost dynamic control")
                control = CompileControlVerificationReceipt.load(
                    self.control_verification_receipt.absolute_path
                )
                if (
                    control.sha256 != receipt.control_verification_receipt_sha256
                    or control.assignment_plan_sha256 != self.assignment_plan_sha256
                ):
                    raise ValueError("compile pointer control verification differs")

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        legacy_binding_names = {
            "assignment_manifest",
            "compile_cache_plan",
            "prewarm_manifest",
            "launch_manifest",
            "attempt_receipt",
            "graceful_shutdown_receipt",
            "final_cache_receipt",
            "immutable_cache_object_manifest",
        }
        common = {
            "schema_version",
            "kind",
            "result_pointer_protocol_sha256",
            "assignment_plan_sha256",
            "formal_execution_authorized",
            *legacy_binding_names,
        }
        if type(raw) is not dict:
            raise TypeError("compile result pointer must be a JSON object")
        schema_version = raw.get("schema_version")
        binding_names = set(legacy_binding_names)
        expected = set(common)
        if schema_version in {2, 3}:
            binding_names.update(
                {"assignment_plan_source", "subprocess_lifecycle_receipt"}
            )
            expected.update({"assignment_plan_source", "subprocess_lifecycle_receipt"})
        if schema_version == 3:
            binding_names.add("control_verification_receipt")
            expected.add("control_verification_receipt")
        if set(raw) != expected:
            raise ValueError("compile result pointer fields differ from schema")
        scalar = {
            name: value for name, value in raw.items() if name not in binding_names
        }
        value = cls(
            **scalar,
            **{
                name: CompileResultBinding.from_dict(raw[name])
                for name in binding_names
            },
        )
        value.validate()
        return value

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("compile result pointer", str(path))
        raw, semantic_sha256 = _load_canonical_json_with_sidecar(
            source,
            label="compile result pointer",
        )
        value = cls.from_dict(raw)
        if semantic_sha256 != value.sha256:
            raise ValueError("compile result pointer semantic digest differs")
        value.reopen()
        if value.schema_version in {2, 3}:
            if value.assignment_plan_source is None:
                raise AssertionError("validated subprocess pointer lost its plan")
            plan = CompileAssignmentPlan.load(
                value.assignment_plan_source.absolute_path
            )
            if Path(plan.result_pointer_path) != source:
                raise ValueError("compile pointer was loaded from an unbound path")
        return value


def _terminal_path(cache_root: Path, name: str) -> Path:
    root = cache_root / "compile-terminal"
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("compile terminal root must be a regular directory")
    return root / name


def _publish_terminal(path: Path, value: object) -> Path:
    _publish_json(path, value)
    _publish_text(Path(f"{path}.sha256"), _content_sha256(value))
    return path


def _execute_compile_assignment(
    plan: CompileAssignmentPlan,
    driver: CompileLifecycleDriver,
    *,
    materialize_cache_files: Callable[[Path], None] | None,
    assignment_plan_source: Path | None,
    subprocess_driver: _CompileSubprocessDriver | None,
    control_verification_receipt_path: Path | None,
    formal_execution_authorized: bool,
) -> CompileResultPointer:
    if type(plan) is not CompileAssignmentPlan:
        raise TypeError("compile lifecycle requires an exact assignment plan")
    _, cache_plan, manifest, _launch = plan.revalidate()
    preflight_compile_cache_launch(cache_plan)
    if type(driver.process_id) is not int or driver.process_id < 1:
        raise ValueError("compile lifecycle driver process ID is invalid")
    if formal_execution_authorized is True:
        if (
            subprocess_driver is None
            or assignment_plan_source is None
            or control_verification_receipt_path is None
        ):
            raise ValueError(
                "formal compile lifecycle requires path-bound subprocess and control"
            )
    elif formal_execution_authorized is not False:
        raise TypeError("compile lifecycle formal flag must be boolean")
    session = start_compile_cache_launch(
        cache_plan,
        process_id=driver.process_id,
        attempt_id=plan.attempt_id,
    )
    try:
        environment = session.environment({})
        driver.start(environment)
        observations = tuple(driver.prewarm(payload) for payload in manifest.payloads)
        for observation in observations:
            observation.validate()
        expected = tuple(
            (payload.request_id, payload.graph_bucket) for payload in manifest.payloads
        )
        observed = tuple((row.request_id, row.graph_bucket) for row in observations)
        if observed != expected:
            raise ValueError(
                "compile prewarm observations do not exactly cover the manifest"
            )
        if materialize_cache_files is not None:
            materialize_cache_files(session.overlay.path)
        shutdown = driver.graceful_shutdown()
        shutdown.validate()
        if shutdown.process_id != driver.process_id:
            raise ValueError("compile shutdown acknowledgement names another process")
        object_path, receipt_path, attempt_path = session.complete()
    except BaseException as error:
        if not session._terminal:
            session.fail(error, reason_code="compile_lifecycle_failed")
        raise

    receipt = CompileCacheReceipt.load(receipt_path)
    attempt = CompileCacheAttemptReceipt.load(attempt_path)
    if (
        receipt.launch_plan_sha256 != cache_plan.sha256
        or receipt.key_sha256 != plan.compile_key_sha256
        or attempt.plan_sha256 != cache_plan.sha256
        or attempt.result_receipt_sha256 != receipt.receipt_sha256
        or attempt.base_receipt_sha256 != cache_plan.base_receipt_sha256
    ):
        raise ValueError("compile terminal receipts differ from the assignment plan")
    cache = ImmutableCompileCache._open_existing_read_only(cache_plan.cache_root)
    if cache.verify(cache_plan.key, receipt_path) != object_path:
        raise ValueError("compile immutable object path differs from cache receipt")

    terminal_root = Path(cache_plan.cache_root)
    subprocess_receipt_path: Path | None = None
    subprocess_receipt: CompileSubprocessLifecycleReceipt | None = None
    if subprocess_driver is not None:
        subprocess_receipt = subprocess_driver.receipt()
        if (
            subprocess_receipt.assignment_plan_sha256 != plan.sha256
            or subprocess_receipt.formal_execution_authorized
            is not formal_execution_authorized
            or subprocess_receipt.control_verification_receipt_sha256
            != (
                None
                if control_verification_receipt_path is None
                else CompileControlVerificationReceipt.load(
                    control_verification_receipt_path
                ).sha256
            )
        ):
            raise ValueError("compile subprocess receipt differs from execution")
        subprocess_receipt_path = _publish_terminal(
            _terminal_path(
                terminal_root,
                f"subprocess-{plan.attempt_id}-{subprocess_receipt.sha256}.json",
            ),
            subprocess_receipt.to_dict(),
        )
    shutdown_path = _publish_terminal(
        _terminal_path(terminal_root, f"shutdown-{plan.attempt_id}.json"),
        {
            "schema_version": 1,
            "kind": "compile_graceful_shutdown_receipt",
            "protocol_sha256": COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256,
            "assignment_plan_sha256": plan.sha256,
            "compile_plan_sha256": cache_plan.sha256,
            "prewarm_manifest_sha256": manifest.sha256,
            "attempt_id": plan.attempt_id,
            "process_id": shutdown.process_id,
            "shutdown_requested_ns": shutdown.shutdown_requested_ns,
            "process_exited_ns": shutdown.process_exited_ns,
            "exit_code": shutdown.exit_code,
            "active_requests": shutdown.active_requests,
            "queued_requests": shutdown.queued_requests,
            "provider_ack_sha256": shutdown.provider_ack_sha256,
            "final_cache_receipt_sha256": receipt.receipt_sha256,
            "prewarm_observations": [asdict(row) for row in observations],
            "subprocess_lifecycle_receipt_sha256": (
                None if subprocess_receipt is None else subprocess_receipt.sha256
            ),
            "formal_execution_authorized": formal_execution_authorized,
        },
    )
    object_manifest_path = _publish_terminal(
        _terminal_path(terminal_root, f"object-{receipt.content_sha256}.json"),
        {
            "schema_version": 1,
            "kind": "compile_immutable_cache_object_manifest",
            "assignment_plan_sha256": plan.sha256,
            "key_sha256": receipt.key_sha256,
            "content_sha256": receipt.content_sha256,
            "object_path": str(object_path),
            "files": [asdict(value) for value in receipt.files],
            "formal_execution_authorized": formal_execution_authorized,
        },
    )
    schema_version = (
        3
        if control_verification_receipt_path is not None
        else (2 if subprocess_receipt_path is not None else 1)
    )
    if schema_version in {2, 3} and assignment_plan_source is None:
        raise AssertionError("subprocess execution lost its path-bound plan")
    pointer = CompileResultPointer(
        schema_version=schema_version,
        kind="compile_atomic_result_pointer",
        result_pointer_protocol_sha256=COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
        assignment_plan_sha256=plan.sha256,
        assignment_manifest=CompileResultBinding.bind(
            Path(plan.assignment_manifest_path), label="assignment manifest"
        ),
        compile_cache_plan=CompileResultBinding.bind(
            Path(plan.compile_cache_plan_path), label="compile cache plan"
        ),
        prewarm_manifest=CompileResultBinding.bind(
            Path(plan.prewarm_manifest_path), label="prewarm manifest"
        ),
        launch_manifest=CompileResultBinding.bind(
            Path(plan.launch_manifest_path), label="compile launch manifest"
        ),
        attempt_receipt=CompileResultBinding.bind(
            attempt_path, label="compile attempt receipt"
        ),
        graceful_shutdown_receipt=CompileResultBinding.bind(
            shutdown_path, label="compile shutdown receipt"
        ),
        final_cache_receipt=CompileResultBinding.bind(
            receipt_path, label="final cache receipt"
        ),
        immutable_cache_object_manifest=CompileResultBinding.bind(
            object_manifest_path, label="immutable object manifest"
        ),
        formal_execution_authorized=formal_execution_authorized,
        assignment_plan_source=(
            None
            if assignment_plan_source is None
            else CompileResultBinding.bind(
                assignment_plan_source, label="compile assignment plan source"
            )
        ),
        subprocess_lifecycle_receipt=(
            None
            if subprocess_receipt_path is None
            else CompileResultBinding.bind(
                subprocess_receipt_path,
                label="compile subprocess lifecycle receipt",
            )
        ),
        control_verification_receipt=(
            None
            if control_verification_receipt_path is None
            else CompileResultBinding.bind(
                control_verification_receipt_path,
                label="compile control verification receipt",
            )
        ),
    )
    pointer.validate()
    result_path = Path(plan.result_pointer_path)
    if not result_path.parent.is_dir() or result_path.parent.is_symlink():
        raise ValueError("compile result pointer parent must be a directory")
    _publish_json(result_path, pointer.to_dict())
    _publish_text(Path(f"{result_path}.sha256"), pointer.sha256)
    pointer.reopen()
    return pointer


def execute_compile_assignment_for_cpu_test(
    plan: CompileAssignmentPlan,
    driver: CompileLifecycleDriver,
    *,
    materialize_cache_files: Callable[[Path], None],
) -> CompileResultPointer:
    """Exercise the lifecycle with a CPU fake; never formal authority."""

    return _execute_compile_assignment(
        plan,
        driver,
        materialize_cache_files=materialize_cache_files,
        assignment_plan_source=None,
        subprocess_driver=None,
        control_verification_receipt_path=None,
        formal_execution_authorized=False,
    )


def _preflight_subprocess_result(
    plan: CompileAssignmentPlan,
    *,
    assignment_plan_source: Path,
    formal_execution_authorized: bool,
    source_authority_sha256: str | None,
    control_verification_receipt_sha256: str | None,
    argv_sha256: str,
) -> CompileResultPointer | None:
    result_path = Path(plan.result_pointer_path)
    sidecar = Path(f"{result_path}.sha256")
    if result_path.exists() or sidecar.exists():
        if not result_path.is_file() or result_path.is_symlink():
            raise ValueError("compile result pointer is an incomplete prior attempt")
        if not sidecar.is_file() or sidecar.is_symlink():
            raise ValueError("compile result pointer commit marker is incomplete")
        pointer = CompileResultPointer.load(result_path)
        if (
            pointer.schema_version != (3 if formal_execution_authorized else 2)
            or pointer.assignment_plan_sha256 != plan.sha256
            or pointer.formal_execution_authorized is not formal_execution_authorized
            or pointer.assignment_plan_source is None
            or Path(pointer.assignment_plan_source.absolute_path)
            != assignment_plan_source
        ):
            raise ValueError("compile result pointer belongs to another execution")
        if pointer.subprocess_lifecycle_receipt is None:
            raise AssertionError("validated subprocess pointer lost its receipt")
        receipt = CompileSubprocessLifecycleReceipt.load(
            pointer.subprocess_lifecycle_receipt.absolute_path
        )
        if (
            receipt.source_authority_sha256 != source_authority_sha256
            or receipt.argv_sha256 != argv_sha256
            or receipt.control_verification_receipt_sha256
            != control_verification_receipt_sha256
        ):
            raise ValueError("compile result pointer uses another subprocess authority")
        return pointer
    parent = result_path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("compile result pointer parent must be a regular directory")
    return None


def _execute_compile_assignment_subprocess_path(
    assignment_plan_path: str | Path,
    *,
    argv: tuple[str, ...],
    timeout_seconds: float,
    source_authority_sha256: str | None,
    control_verification_receipt_sha256: str | None,
    control_verification_receipt_path: Path | None,
    formal_execution_authorized: bool,
) -> CompileResultPointer:
    plan_path = _absolute_path("compile assignment plan", str(assignment_plan_path))
    plan = CompileAssignmentPlan.load(plan_path)
    _assignment, cache_plan, _manifest, _launch = plan.revalidate()
    preflight_compile_cache_launch(cache_plan)
    argv_sha256 = _content_sha256({"argv": list(argv)})
    resumed = _preflight_subprocess_result(
        plan,
        assignment_plan_source=plan_path,
        formal_execution_authorized=formal_execution_authorized,
        source_authority_sha256=source_authority_sha256,
        control_verification_receipt_sha256=(control_verification_receipt_sha256),
        argv_sha256=argv_sha256,
    )
    if resumed is not None:
        return resumed
    driver = _CompileSubprocessDriver(
        argv=argv,
        plan=plan,
        timeout_seconds=timeout_seconds,
        source_authority_sha256=source_authority_sha256,
        control_verification_receipt_sha256=(control_verification_receipt_sha256),
        formal_execution_authorized=formal_execution_authorized,
    )
    driver.spawn()
    try:
        return _execute_compile_assignment(
            plan,
            driver,
            materialize_cache_files=None,
            assignment_plan_source=plan_path,
            subprocess_driver=driver,
            control_verification_receipt_path=control_verification_receipt_path,
            formal_execution_authorized=formal_execution_authorized,
        )
    finally:
        driver.abort()


def execute_compile_assignment_subprocess_for_cpu_test(
    assignment_plan_path: str | Path,
    argv: tuple[str, ...],
    *,
    timeout_seconds: float = 30.0,
) -> CompileResultPointer:
    """Run a real diagnostic child process without granting formal authority."""

    return _execute_compile_assignment_subprocess_path(
        assignment_plan_path,
        argv=argv,
        timeout_seconds=timeout_seconds,
        source_authority_sha256=None,
        control_verification_receipt_sha256=None,
        control_verification_receipt_path=None,
        formal_execution_authorized=False,
    )


def _execute_release_compile_assignment_plan_admitted(
    assignment_plan_path: str | Path,
    *,
    control_attestation: ControlArtifactAttestation | None = None,
    prepared_content_verification_receipt_path: str | Path | None = None,
    replay_store: ChallengeReplayStore | None = None,
    now_ns: int | None = None,
    additional_challenge_sha256s: tuple[str, ...] = (),
    timeout_seconds: float,
) -> CompileResultPointer:
    """Execute one exact plan under root-signed dynamic deployment control."""

    if (
        type(control_attestation) is not ControlArtifactAttestation
        or not isinstance(prepared_content_verification_receipt_path, (str, Path))
        or type(replay_store) is not ChallengeReplayStore
        or type(now_ns) is not int
        or now_ns < 0
    ):
        raise CompileRunnerBlocked(RELEASE_COMPILE_DYNAMIC_CONTROL_UNAVAILABLE)
    plan_path = _absolute_path("compile assignment plan", str(assignment_plan_path))
    plan = CompileAssignmentPlan.load(plan_path)
    (
        prepared_content_receipt,
        prepared_model_authorization,
    ) = revalidate_prepared_content_verification_receipt(
        prepared_content_verification_receipt_path,
        current_ns=now_ns,
    )
    expected_additional_challenges = tuple(sorted(set(additional_challenge_sha256s)))
    if len(expected_additional_challenges) != len(additional_challenge_sha256s):
        raise ValueError("compile control additional challenges are duplicated")
    if prepared_model_authorization.challenge_sha256 in expected_additional_challenges:
        raise ValueError(
            "compile control must not reserve the prepared-content challenge twice"
        )
    result_path = Path(plan.result_pointer_path)
    if result_path.exists() or Path(f"{result_path}.sha256").exists():
        pointer = CompileResultPointer.load(result_path)
        if (
            pointer.schema_version != 3
            or pointer.assignment_plan_sha256 != plan.sha256
            or pointer.control_verification_receipt is None
        ):
            raise ValueError("existing compile pointer is not this formal execution")
        prior = CompileControlVerificationReceipt.load(
            pointer.control_verification_receipt.absolute_path
        )
        if (
            prior.control_envelope != control_attestation
            or prior.additional_challenge_sha256s != expected_additional_challenges
            or prior.prepared_content_verification_receipt != prepared_content_receipt
        ):
            raise ValueError("existing compile pointer uses another control")
        return pointer
    control = verify_and_reserve_compile_control(
        plan,
        control_attestation,
        prepared_content_verification_receipt_path=(
            prepared_content_verification_receipt_path
        ),
        replay_store=replay_store,
        now_ns=now_ns,
        additional_challenge_sha256s=additional_challenge_sha256s,
    )
    control_path = Path(plan.result_pointer_path).parent / (
        f"compile-control-{plan.sha256}.json"
    )
    if control_path.exists() or Path(f"{control_path}.sha256").exists():
        existing = CompileControlVerificationReceipt.load(control_path)
        if existing != control:
            raise ValueError("compile control path belongs to another authorization")
    else:
        _publish_json(control_path, control.to_dict())
        _publish_text(Path(f"{control_path}.sha256"), control.sha256)
        CompileControlVerificationReceipt.load(control_path)
    source = control.source_descriptor
    argv = (source.interpreter_path, source.helper_path)
    return _execute_compile_assignment_subprocess_path(
        plan_path,
        argv=argv,
        timeout_seconds=timeout_seconds,
        source_authority_sha256=source.sha256,
        control_verification_receipt_sha256=control.sha256,
        control_verification_receipt_path=control_path,
        formal_execution_authorized=True,
    )


def execute_release_compile_assignment_plan(
    assignment_plan_path: str | Path,
    *,
    control_attestation: ControlArtifactAttestation | None = None,
    prepared_content_verification_receipt_path: str | Path | None = None,
    replay_store: ChallengeReplayStore | None = None,
    now_ns: int | None = None,
    additional_challenge_sha256s: tuple[str, ...] = (),
) -> CompileResultPointer:
    """Reject direct formal execution lacking a typed launch-cap admission.

    The admitted preflight operator calls the private boundary above only after
    atomically consuming its sealed per-cell cap.  Keeping this compatibility
    name fail-closed prevents callers from supplying an independent timeout.
    """

    del (
        assignment_plan_path,
        control_attestation,
        prepared_content_verification_receipt_path,
        replay_store,
        now_ns,
        additional_challenge_sha256s,
    )
    raise CompileRunnerBlocked(RELEASE_COMPILE_DYNAMIC_CONTROL_UNAVAILABLE)


__all__ = [
    "COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256",
    "COMPILE_CONTROL_VERIFICATION_PROTOCOL_SHA256",
    "COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256",
    "COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256",
    "COMPILE_WORKER_IMPORT_PROTOCOL_SHA256",
    "RELEASE_COMPILE_ASSIGNMENT_PLAN_ALLOWLIST_EMPTY",
    "RELEASE_COMPILE_ASSIGNMENT_PLAN_UNTRUSTED",
    "RELEASE_COMPILE_DYNAMIC_CONTROL_UNAVAILABLE",
    "RELEASE_COMPILE_GPU_SOURCE_REGISTRY_EMPTY",
    "RELEASE_COMPILE_GPU_SOURCE_UNTRUSTED",
    "RELEASE_COMPILE_RUNNER_UNAVAILABLE",
    "RELEASE_COMPILE_SUBPROCESSES",
    "RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S",
    "RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S",
    "CompileAssignmentPlan",
    "CompileControlVerificationReceipt",
    "CompileLaunchManifest",
    "CompileLifecycleDriver",
    "CompilePrewarmObservation",
    "CompileResultBinding",
    "CompileResultPointer",
    "CompileRunnerBlocked",
    "CompileShutdownObservation",
    "CompileSubprocessEvent",
    "CompileSubprocessLifecycleReceipt",
    "CompileWorkerSourceDescriptor",
    "ReleaseCompileSubprocess",
    "execute_compile_assignment_for_cpu_test",
    "execute_compile_assignment_subprocess_for_cpu_test",
    "execute_release_compile_assignment_plan",
    "load_compile_prewarm_manifest",
    "require_release_compile_assignment_plan",
    "revalidate_prepared_content_verification_receipt",
    "verify_and_reserve_compile_control",
    "write_compile_prewarm_manifest",
]
