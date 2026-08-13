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
from pathlib import Path
from typing import Protocol, Self

from lightcone_spec import PINNED_SGLANG_TREE
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

COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
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
        "release_execution_available": False,
    }
)
RELEASE_COMPILE_RUNNER_UNAVAILABLE = "release_first_party_compile_runner_unavailable"
RELEASE_COMPILE_ASSIGNMENT_PLAN_ALLOWLIST_EMPTY = (
    "release_compile_assignment_plan_allowlist_empty"
)
RELEASE_COMPILE_ASSIGNMENT_PLAN_UNTRUSTED = (
    "release_compile_assignment_plan_not_allowlisted"
)
RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S: tuple[str, ...] = ()

COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "first_party_compile_subprocess_lifecycle",
        "transport": "canonical_json_lines_over_private_stdin_stdout",
        "child_must_delay_model_and_gpu_initialization_until_start": True,
        "ordered_messages": (
            "ready",
            "start_with_private_compile_environment",
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
            or self.schema_version != 1
            or self.kind != "first_party_compile_assignment_plan"
        ):
            raise ValueError("compile assignment plan schema is unsupported")
        if self.protocol_sha256 != COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256:
            raise ValueError("compile assignment plan uses another protocol")
        for label, value in (
            ("assignment", self.assignment_sha256),
            ("cache plan", self.compile_cache_plan_sha256),
            ("prewarm manifest", self.prewarm_manifest_sha256),
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
        pointer_path = _absolute_path(
            "compile result pointer", str(result_pointer_path)
        )
        assignment = CompileOnlyAssignmentContract.load(assignment_path)
        cache_plan = CompileCacheLaunchPlan.load(cache_plan_path)
        prewarm = load_compile_prewarm_manifest(prewarm_path)
        if cache_plan != assignment.compile_cache_plan:
            raise ValueError("compile cache plan differs from assignment authority")
        if prewarm != assignment.prewarm_manifest:
            raise ValueError(
                "compile prewarm manifest differs from assignment authority"
            )
        if pointer_path != Path(assignment.result_pointer_path):
            raise ValueError("compile result pointer differs from assignment authority")
        key = cache_plan.key
        if prewarm.model_lock_sha256 != assignment.prewarm_manifest.model_lock_sha256:
            raise ValueError("compile model lock differs from assignment authority")
        value = cls(
            schema_version=1,
            kind="first_party_compile_assignment_plan",
            protocol_sha256=COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256,
            assignment_manifest_path=str(assignment_path),
            assignment_sha256=assignment.sha256,
            compile_cache_plan_path=str(cache_plan_path),
            compile_cache_plan_sha256=cache_plan.sha256,
            prewarm_manifest_path=str(prewarm_path),
            prewarm_manifest_sha256=prewarm.sha256,
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
    ]:
        self.validate()
        assignment = CompileOnlyAssignmentContract.load(self.assignment_manifest_path)
        cache_plan = CompileCacheLaunchPlan.load(self.compile_cache_plan_path)
        prewarm = load_compile_prewarm_manifest(self.prewarm_manifest_path)
        if assignment.sha256 != self.assignment_sha256:
            raise ValueError("compile assignment changed during revalidation")
        if cache_plan.sha256 != self.compile_cache_plan_sha256:
            raise ValueError("compile cache plan changed during revalidation")
        if prewarm.sha256 != self.prewarm_manifest_sha256:
            raise ValueError("compile prewarm manifest changed during revalidation")
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
        return assignment, cache_plan, prewarm


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
    process_id: int
    process_started_ns: int
    process_exited_ns: int
    exit_code: int
    events: tuple[CompileSubprocessEvent, ...]
    formal_execution_authorized: bool

    def validate(self, *, reopen_executable: bool) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
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
            if self.source_authority_sha256 is None:
                raise ValueError("formal compile receipt lacks source authority")
            _require_formal_compile_receipt_authority(
                assignment_plan_sha256=self.assignment_plan_sha256,
                executable_path=self.executable_path,
                executable_raw_sha256=self.executable_raw_sha256,
                argv_sha256=self.argv_sha256,
                source_authority_sha256=self.source_authority_sha256,
                reopen_executable=reopen_executable,
            )
        elif self.formal_execution_authorized is not False:
            raise TypeError("compile subprocess formal flag must be boolean")
        elif self.source_authority_sha256 is not None:
            raise ValueError("diagnostic compile receipt cannot claim source authority")
        start_row = event_rows[1]
        start_environment = start_row.get("cache_environment")
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
        if reopen_executable:
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
        assignment_plan_sha256: str,
        timeout_seconds: float,
        source_authority_sha256: str | None,
        formal_execution_authorized: bool,
    ) -> None:
        if type(argv) is not tuple or not argv:
            raise TypeError("compile subprocess argv must be a non-empty tuple")
        for argument in argv:
            if type(argument) is not str or not argument or "\x00" in argument:
                raise ValueError("compile subprocess argument contains NUL")
        executable = _absolute_path("compile subprocess executable", argv[0])
        digest, size = _raw_sha256(executable, label="compile subprocess executable")
        _require_sha256("compile subprocess plan", assignment_plan_sha256)
        if (
            type(timeout_seconds) not in {int, float}
            or isinstance(timeout_seconds, bool)
            or not (0 < float(timeout_seconds) <= 600)
        ):
            raise ValueError("compile subprocess timeout must be in (0, 600] seconds")
        if formal_execution_authorized is True:
            if source_authority_sha256 is None:
                raise ValueError("formal compile subprocess lacks source authority")
            _require_sha256(
                "compile subprocess source authority", source_authority_sha256
            )
        elif formal_execution_authorized is not False:
            raise TypeError("compile subprocess formal flag must be boolean")
        elif source_authority_sha256 is not None:
            raise ValueError("diagnostic compile subprocess cannot claim authority")
        self.argv = argv
        self.assignment_plan_sha256 = assignment_plan_sha256
        self.timeout_seconds = float(timeout_seconds)
        self.executable_path = executable
        self.executable_raw_sha256 = digest
        self.executable_size = size
        self.argv_sha256 = _content_sha256({"argv": list(argv)})
        self.source_authority_sha256 = source_authority_sha256
        self.formal_execution_authorized = formal_execution_authorized
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_buffer = b""
        self._events: list[CompileSubprocessEvent] = []
        self._process_started_ns: int | None = None
        self._process_exited_ns: int | None = None
        self._exit_code: int | None = None

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
        deadline = time.monotonic() + self.timeout_seconds
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
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        self._process_started_ns = time.monotonic_ns()
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
        try:
            exit_code = self._process.wait(timeout=self.timeout_seconds)
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
            schema_version=1,
            kind="compile_subprocess_lifecycle_raw_receipt",
            protocol_sha256=COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
            assignment_plan_sha256=self.assignment_plan_sha256,
            executable_path=str(self.executable_path),
            executable_raw_sha256=self.executable_raw_sha256,
            executable_size=self.executable_size,
            argv_sha256=self.argv_sha256,
            source_authority_sha256=self.source_authority_sha256,
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
            _absolute_path(label, self.absolute_path), label=label
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
    attempt_receipt: CompileResultBinding
    graceful_shutdown_receipt: CompileResultBinding
    final_cache_receipt: CompileResultBinding
    immutable_cache_object_manifest: CompileResultBinding
    formal_execution_authorized: bool
    assignment_plan_source: CompileResultBinding | None = None
    subprocess_lifecycle_receipt: CompileResultBinding | None = None

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version not in {1, 2}
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
            ):
                raise ValueError(
                    "legacy compile pointer cannot claim subprocess evidence"
                )
        elif (
            type(self.assignment_plan_source) is not CompileResultBinding
            or type(self.subprocess_lifecycle_receipt) is not CompileResultBinding
        ):
            raise TypeError("subprocess compile pointer lacks path-bound raw evidence")
        if type(self.formal_execution_authorized) is not bool:
            raise TypeError("compile result formal flag must be boolean")
        if self.formal_execution_authorized is True and self.schema_version != 2:
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
            "attempt_receipt": self.attempt_receipt,
            "graceful_shutdown_receipt": self.graceful_shutdown_receipt,
            "final_cache_receipt": self.final_cache_receipt,
            "immutable_cache_object_manifest": self.immutable_cache_object_manifest,
        }
        if self.assignment_plan_source is not None:
            bindings["assignment_plan_source"] = self.assignment_plan_source
        if self.subprocess_lifecycle_receipt is not None:
            bindings["subprocess_lifecycle_receipt"] = self.subprocess_lifecycle_receipt
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
        if self.schema_version == 2:
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

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        legacy_binding_names = {
            "assignment_manifest",
            "compile_cache_plan",
            "prewarm_manifest",
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
        if schema_version == 2:
            binding_names.update(
                {"assignment_plan_source", "subprocess_lifecycle_receipt"}
            )
            expected.update({"assignment_plan_source", "subprocess_lifecycle_receipt"})
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
        if value.schema_version == 2:
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
    formal_execution_authorized: bool,
) -> CompileResultPointer:
    if type(plan) is not CompileAssignmentPlan:
        raise TypeError("compile lifecycle requires an exact assignment plan")
    _, cache_plan, manifest = plan.revalidate()
    preflight_compile_cache_launch(cache_plan)
    if type(driver.process_id) is not int or driver.process_id < 1:
        raise ValueError("compile lifecycle driver process ID is invalid")
    if formal_execution_authorized is True:
        if subprocess_driver is None or assignment_plan_source is None:
            raise ValueError("formal compile lifecycle requires path-bound subprocess")
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
    schema_version = 2 if subprocess_receipt_path is not None else 1
    if schema_version == 2 and assignment_plan_source is None:
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
        formal_execution_authorized=False,
    )


def _preflight_subprocess_result(
    plan: CompileAssignmentPlan,
    *,
    assignment_plan_source: Path,
    formal_execution_authorized: bool,
    source_authority_sha256: str | None,
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
            pointer.schema_version != 2
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
    formal_execution_authorized: bool,
) -> CompileResultPointer:
    plan_path = _absolute_path("compile assignment plan", str(assignment_plan_path))
    plan = CompileAssignmentPlan.load(plan_path)
    _assignment, cache_plan, _manifest = plan.revalidate()
    preflight_compile_cache_launch(cache_plan)
    argv_sha256 = _content_sha256({"argv": list(argv)})
    resumed = _preflight_subprocess_result(
        plan,
        assignment_plan_source=plan_path,
        formal_execution_authorized=formal_execution_authorized,
        source_authority_sha256=source_authority_sha256,
        argv_sha256=argv_sha256,
    )
    if resumed is not None:
        return resumed
    driver = _CompileSubprocessDriver(
        argv=argv,
        assignment_plan_sha256=plan.sha256,
        timeout_seconds=timeout_seconds,
        source_authority_sha256=source_authority_sha256,
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
        formal_execution_authorized=False,
    )


def execute_release_compile_assignment_plan(
    assignment_plan_path: str | Path,
    *,
    timeout_seconds: float = 600.0,
) -> CompileResultPointer:
    """Execute a path-bound formal plan only under exact source allowlists."""

    # Both empty policies are checked before opening the caller-named plan path.
    if len(RELEASE_COMPILE_SUBPROCESSES) != 1:
        raise CompileRunnerBlocked(RELEASE_COMPILE_RUNNER_UNAVAILABLE)
    if not RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_ASSIGNMENT_PLAN_ALLOWLIST_EMPTY)
    source = RELEASE_COMPILE_SUBPROCESSES[0]
    source.validate(reopen_executable=True)
    if not RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_GPU_SOURCE_REGISTRY_EMPTY)
    if source.sha256 not in RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_GPU_SOURCE_UNTRUSTED)
    plan_path = _absolute_path("compile assignment plan", str(assignment_plan_path))
    plan = CompileAssignmentPlan.load(plan_path)
    source = require_release_compile_assignment_plan(plan)
    return _execute_compile_assignment_subprocess_path(
        plan_path,
        argv=source.argv,
        timeout_seconds=timeout_seconds,
        source_authority_sha256=source.sha256,
        formal_execution_authorized=True,
    )


__all__ = [
    "COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256",
    "COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256",
    "COMPILE_WORKER_IMPORT_PROTOCOL_SHA256",
    "RELEASE_COMPILE_ASSIGNMENT_PLAN_ALLOWLIST_EMPTY",
    "RELEASE_COMPILE_ASSIGNMENT_PLAN_UNTRUSTED",
    "RELEASE_COMPILE_GPU_SOURCE_REGISTRY_EMPTY",
    "RELEASE_COMPILE_GPU_SOURCE_UNTRUSTED",
    "RELEASE_COMPILE_RUNNER_UNAVAILABLE",
    "RELEASE_COMPILE_SUBPROCESSES",
    "RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S",
    "RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S",
    "CompileAssignmentPlan",
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
    "write_compile_prewarm_manifest",
]
