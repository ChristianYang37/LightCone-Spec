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
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn, Protocol, Self

from lightcone_spec.runtime.compile_cache import (
    COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256,
    COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
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
RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S: tuple[str, ...] = ()


class CompileRunnerBlocked(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(f"COMPILE execution is BLOCKED: {reason_code}")
        self.reason_code = reason_code


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
) -> NoReturn:
    """Block before reading paths or creating cache/GPU state in this release."""

    if plan is not None:
        if type(plan) is not CompileAssignmentPlan:
            raise TypeError("release compile runner requires an exact assignment plan")
        plan.validate()
    if not RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S:
        raise CompileRunnerBlocked(RELEASE_COMPILE_RUNNER_UNAVAILABLE)
    raise CompileRunnerBlocked(RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE)


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

    def validate(self) -> None:
        if self.schema_version != 1 or self.kind != "compile_atomic_result_pointer":
            raise ValueError("compile result pointer schema is unsupported")
        if (
            self.result_pointer_protocol_sha256
            != COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256
        ):
            raise ValueError("compile result pointer uses another protocol")
        _require_sha256("compile result assignment plan", self.assignment_plan_sha256)
        if self.formal_execution_authorized is not False:
            raise ValueError(
                "CPU compile lifecycle evidence cannot authorize formal execution"
            )
        for label, binding in self.bindings().items():
            if type(binding) is not CompileResultBinding:
                raise TypeError(f"compile result {label} binding is invalid")
            _absolute_path(label, binding.absolute_path)
            _require_sha256(label, binding.raw_sha256)
            if type(binding.size) is not int or binding.size <= 0:
                raise ValueError(f"compile result {label} size is invalid")

    def bindings(self) -> dict[str, CompileResultBinding]:
        return {
            "assignment_manifest": self.assignment_manifest,
            "compile_cache_plan": self.compile_cache_plan,
            "prewarm_manifest": self.prewarm_manifest,
            "attempt_receipt": self.attempt_receipt,
            "graceful_shutdown_receipt": self.graceful_shutdown_receipt,
            "final_cache_receipt": self.final_cache_receipt,
            "immutable_cache_object_manifest": self.immutable_cache_object_manifest,
        }

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "result_pointer_protocol_sha256": self.result_pointer_protocol_sha256,
            "assignment_plan_sha256": self.assignment_plan_sha256,
            "formal_execution_authorized": self.formal_execution_authorized,
            **{label: asdict(binding) for label, binding in self.bindings().items()},
        }

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def reopen(self) -> None:
        self.validate()
        for label, binding in self.bindings().items():
            binding.reopen(label=f"compile result {label}")

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        binding_names = {
            "assignment_manifest",
            "compile_cache_plan",
            "prewarm_manifest",
            "attempt_receipt",
            "graceful_shutdown_receipt",
            "final_cache_receipt",
            "immutable_cache_object_manifest",
        }
        expected = {
            "schema_version",
            "kind",
            "result_pointer_protocol_sha256",
            "assignment_plan_sha256",
            "formal_execution_authorized",
            *binding_names,
        }
        if type(raw) is not dict or set(raw) != expected:
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


def execute_compile_assignment_for_cpu_test(
    plan: CompileAssignmentPlan,
    driver: CompileLifecycleDriver,
    *,
    materialize_cache_files: Callable[[Path], None],
) -> CompileResultPointer:
    """Exercise the frozen lifecycle with a CPU fake; never formal authority."""

    if type(plan) is not CompileAssignmentPlan:
        raise TypeError("compile CPU lifecycle requires an exact assignment plan")
    _, cache_plan, manifest = plan.revalidate()
    if type(driver.process_id) is not int or driver.process_id < 1:
        raise ValueError("compile lifecycle driver process ID is invalid")
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
        shutdown = driver.graceful_shutdown()
        shutdown.validate()
        if shutdown.process_id != driver.process_id:
            raise ValueError("compile shutdown acknowledgement names another process")
        materialize_cache_files(session.overlay.path)
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
    ):
        raise ValueError("compile terminal receipts differ from the assignment plan")
    cache = ImmutableCompileCache._open_existing_read_only(cache_plan.cache_root)
    if cache.verify(cache_plan.key, receipt_path) != object_path:
        raise ValueError("compile immutable object path differs from cache receipt")

    terminal_root = Path(cache_plan.cache_root)
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
            "formal_execution_authorized": False,
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
        },
    )
    pointer = CompileResultPointer(
        schema_version=1,
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
        formal_execution_authorized=False,
    )
    pointer.validate()
    result_path = Path(plan.result_pointer_path)
    if not result_path.parent.is_dir() or result_path.parent.is_symlink():
        raise ValueError("compile result pointer parent must be a directory")
    _publish_json(result_path, pointer.to_dict())
    _publish_text(Path(f"{result_path}.sha256"), pointer.sha256)
    pointer.reopen()
    return pointer


__all__ = [
    "COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256",
    "RELEASE_COMPILE_RUNNER_UNAVAILABLE",
    "RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S",
    "CompileAssignmentPlan",
    "CompileLifecycleDriver",
    "CompilePrewarmObservation",
    "CompileResultBinding",
    "CompileResultPointer",
    "CompileRunnerBlocked",
    "CompileShutdownObservation",
    "execute_compile_assignment_for_cpu_test",
    "load_compile_prewarm_manifest",
    "require_release_compile_assignment_plan",
    "write_compile_prewarm_manifest",
]
