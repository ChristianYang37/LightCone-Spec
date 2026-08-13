"""Durable, non-authorizing publication for live session contract evidence.

The live runtime remains the sole parser and authority for the source-owned
session and native terminal lifecycle.  This module only persists the exact
validated result.  A close manifest is published last for a complete CPU
contract; interrupted or failed sessions retain any trace files but have no
commit marker and can never authorize process reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from lightcone_spec.orchestration.native_terminal import (
    ValidatedNativeTerminalEvidence,
    canonical_json_bytes,
)
from lightcone_spec.orchestration.session_live_runtime import (
    SessionLiveContractResult,
    SessionLiveStepBinding,
)

SESSION_LIVE_EVIDENCE_LEVEL = "CPU_CONTRACT_ONLY"
SESSION_LIVE_GPU_RESET_SEMANTICS = "PENDING"
SESSION_LIVE_CLOSE_MANIFEST = "session-close-manifest.json"
SESSION_LIVE_STEP_PREFIX = "live-step-"

_TRACE_ARTIFACT_KIND = "lightcone_session_live_trace_evidence"
_CLOSE_MANIFEST_KIND = "lightcone_session_live_close_manifest"
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class SessionLiveTraceArtifactBinding:
    trace_index: int
    execution_plan_sha256: str
    filename: str
    size: int
    raw_sha256: str
    complete: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SessionLiveStepArtifactBinding:
    """One raw source response durably retained before the next live action."""

    sequence: int
    step: str
    execution_plan_sha256: str | None
    content_sha256: str
    filename: str
    size: int
    raw_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SessionLiveEvidencePublication:
    """Paths and identities from a diagnostic-only durable publication."""

    status: str
    reason: str
    evidence_level: str
    gpu_reset_semantics: str
    reuse_authorized: bool
    trace_artifacts: tuple[SessionLiveTraceArtifactBinding, ...]
    close_manifest_path: Path | None
    close_manifest_sha256: str | None

    def validate(self) -> None:
        if self.reuse_authorized:
            raise ValueError("session live evidence cannot authorize reuse")
        if (
            self.evidence_level != SESSION_LIVE_EVIDENCE_LEVEL
            or self.gpu_reset_semantics != SESSION_LIVE_GPU_RESET_SEMANTICS
        ):
            raise ValueError("session live evidence overstated its evidence level")
        if self.status not in {"CPU_CONTRACT_ONLY", "FRESH_PROCESS_REQUIRED"}:
            raise ValueError("session live evidence status is unsupported")
        committed = self.close_manifest_path is not None
        if committed != (self.close_manifest_sha256 is not None):
            raise ValueError("session live close-manifest identity is incomplete")
        if committed != (self.status == "CPU_CONTRACT_ONLY"):
            raise ValueError("only a complete CPU contract may have a close manifest")
        if self.status == "CPU_CONTRACT_ONLY" and (
            not self.trace_artifacts
            or not all(binding.complete for binding in self.trace_artifacts)
        ):
            raise ValueError("committed session evidence lacks complete trace coverage")

    @property
    def committed(self) -> bool:
        self.validate()
        return self.close_manifest_path is not None


def _strict_json(raw: str | bytes, *, label: str) -> object:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-finite JSON value: {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if canonical_json_bytes(value) != text.encode("utf-8"):
        raise ValueError(f"{label} is not exact canonical JSON")
    return value


def _step_dict(step: SessionLiveStepBinding) -> dict[str, object]:
    return {
        "step": step.step,
        "execution_plan_sha256": step.execution_plan_sha256,
        "content_sha256": step.content_sha256,
        "raw_response": _strict_json(step.raw_json, label=f"{step.step} response"),
    }


def _terminal_dict(terminal: ValidatedNativeTerminalEvidence) -> dict[str, object]:
    if type(terminal) is not ValidatedNativeTerminalEvidence:
        raise TypeError("session trace requires exact validated terminal evidence")
    terminal.binding.validate()
    return {
        "binding": terminal.binding.begin_payload(),
        "begin_receipt": _strict_json(
            terminal.begin_receipt.raw_json,
            label="native terminal begin receipt",
        ),
        "reset_receipt": _strict_json(
            terminal.reset_receipt.raw_json,
            label="native terminal reset receipt",
        ),
        "terminal": _strict_json(
            terminal.raw_json,
            label="native terminal evidence",
        ),
        "terminal_sha256": terminal.terminal_sha256,
        "trusted_attestation": terminal.trusted_attestation,
        "trusted_attester_policy_sha256": (terminal.trusted_attester_policy_sha256),
    }


def _trace_artifact(
    result: SessionLiveContractResult,
    *,
    trace_index: int,
    execution_plan_sha256: str,
) -> dict[str, object] | None:
    steps = tuple(
        step
        for step in result.steps
        if step.execution_plan_sha256 == execution_plan_sha256
    )
    terminal = next(
        (
            item
            for item in result.native_terminals
            if item.binding.execution_plan_sha256 == execution_plan_sha256
        ),
        None,
    )
    if not steps and terminal is None:
        return None
    complete = (
        tuple(step.step for step in steps)
        == (
            "session_reset_boundary",
            "native_terminal_capability",
            "atomic_trace_begin",
            "atomic_trace_reset",
            "atomic_trace_finalize",
        )
        and terminal is not None
    )
    return {
        "schema_version": 1,
        "artifact_kind": _TRACE_ARTIFACT_KIND,
        "status": result.audit.status,
        "reason": result.audit.reason,
        "evidence_level": SESSION_LIVE_EVIDENCE_LEVEL,
        "gpu_reset_semantics": SESSION_LIVE_GPU_RESET_SEMANTICS,
        "reuse_authorized": False,
        "session_plan_sha256": result.audit.session_plan_sha256,
        "trace_index": trace_index,
        "execution_plan_sha256": execution_plan_sha256,
        "complete": complete,
        "steps": [_step_dict(step) for step in steps],
        "native_terminal": None if terminal is None else _terminal_dict(terminal),
    }


def _global_steps(result: SessionLiveContractResult) -> list[dict[str, object]]:
    return [
        _step_dict(step) for step in result.steps if step.execution_plan_sha256 is None
    ]


def _close_manifest(
    result: SessionLiveContractResult,
    bindings: tuple[SessionLiveTraceArtifactBinding, ...],
    *,
    incremental_steps: tuple[SessionLiveStepArtifactBinding, ...] = (),
) -> dict[str, object]:
    if result.audit.status != "CPU_CONTRACT_ONLY":
        raise ValueError("failed session evidence cannot produce a close manifest")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": _CLOSE_MANIFEST_KIND,
        "commit_marker": "COMPLETE_CPU_CONTRACT_ONLY",
        "status": result.audit.status,
        "reason": result.audit.reason,
        "evidence_level": SESSION_LIVE_EVIDENCE_LEVEL,
        "gpu_reset_semantics": SESSION_LIVE_GPU_RESET_SEMANTICS,
        "reuse_authorized": False,
        "session_plan_sha256": result.audit.session_plan_sha256,
        "audit_sha256": result.audit.sha256,
        "execution_plan_sha256s": list(result.execution_plan_sha256s),
        "global_steps": _global_steps(result),
        "incremental_step_artifacts": [
            binding.to_dict() for binding in incremental_steps
        ],
        "trace_artifacts": [binding.to_dict() for binding in bindings],
        "native_terminal_sha256s": [
            terminal.terminal_sha256 for terminal in result.native_terminals
        ],
        "close_receipt_sha256": result.audit.close_receipt_sha256,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


def _resolved_directory(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute() or value.resolve(strict=False) != value:
        raise ValueError(
            "session live evidence directory must be absolute, resolved, and symlink-free"
        )
    if value.is_symlink() or not value.is_dir():
        raise ValueError("session live evidence directory must already exist")
    return value


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    current = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(opened.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != (current.st_dev, current.st_ino):
        os.close(descriptor)
        raise RuntimeError("session live evidence directory identity changed")
    return descriptor


def _require_same_directory(descriptor: int, path: Path) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != (current.st_dev, current.st_ino):
        raise RuntimeError("session live evidence directory was replaced")


def _publish_exclusive_at(
    directory_fd: int,
    *,
    filename: str,
    body: bytes,
    label: str,
) -> None:
    if not filename or filename in {".", ".."} or "/" in filename:
        raise ValueError(f"{label} filename escapes its evidence directory")
    if not body or len(body) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"{label} has an unsupported byte size")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(body)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written < 1:
                raise RuntimeError(f"{label} write made no progress")
            offset += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError(f"{label} is not an exclusive single-link file")
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _read_bound_at(directory_fd: int, *, filename: str, label: str) -> bytes:
    if not filename or filename in {".", ".."} or "/" in filename:
        raise ValueError(f"{label} filename escapes its evidence directory")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(filename, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size < 1
            or opened.st_size > _MAX_ARTIFACT_BYTES
        ):
            raise RuntimeError(f"{label} is not one supported single-link file")
        body = os.read(descriptor, opened.st_size + 1)
        after = os.fstat(descriptor)
        current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            len(body) != opened.st_size
            or identity(opened) != identity(after)
            or identity(after) != identity(current)
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


def _trace_filename(index: int, execution_plan_sha256: str) -> str:
    return f"trace-{index:04d}-{execution_plan_sha256}.json"


def _step_filename(sequence: int, content_sha256: str) -> str:
    return f"{SESSION_LIVE_STEP_PREFIX}{sequence:06d}-{content_sha256}.json"


def _step_artifact(sequence: int, step: SessionLiveStepBinding) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "lightcone_session_live_incremental_step",
        "evidence_level": SESSION_LIVE_EVIDENCE_LEVEL,
        "gpu_reset_semantics": SESSION_LIVE_GPU_RESET_SEMANTICS,
        "reuse_authorized": False,
        "sequence": sequence,
        "step_binding": _step_dict(step),
    }


def _validate_incremental_steps(
    *,
    directory_fd: int,
    result: SessionLiveContractResult,
    bindings: tuple[SessionLiveStepArtifactBinding, ...],
) -> None:
    if len(bindings) != len(result.steps):
        raise RuntimeError("incremental live-step coverage is incomplete")
    for sequence, (binding, step) in enumerate(zip(bindings, result.steps, strict=True)):
        expected = canonical_json_bytes(_step_artifact(sequence, step))
        if (
            binding.sequence != sequence
            or binding.step != step.step
            or binding.execution_plan_sha256 != step.execution_plan_sha256
            or binding.content_sha256 != step.content_sha256
            or binding.filename != _step_filename(sequence, step.content_sha256)
            or binding.size != len(expected)
            or binding.raw_sha256 != hashlib.sha256(expected).hexdigest()
        ):
            raise RuntimeError("incremental live-step binding changed")
        observed = _read_bound_at(
            directory_fd,
            filename=binding.filename,
            label="incremental live-step artifact",
        )
        _strict_json(observed, label="incremental live-step artifact")
        if observed != expected:
            raise RuntimeError("incremental live-step artifact changed")


class IncrementalSessionLiveEvidenceSink:
    """Crash-durable diagnostic sink for the live source-response chain.

    Each step is published exclusively and fsynced before control returns to the
    runtime.  A failure or cancellation only closes this sink and deliberately
    leaves no session close manifest.  Successful finalization reopens every
    step, validates it against the typed terminal result, and then uses the
    normal trace-first/manifest-last publication path.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._directory = _resolved_directory(output_dir)
        opened = os.stat(self._directory, follow_symlinks=False)
        self._directory_identity = (opened.st_dev, opened.st_ino)
        self._bindings: list[SessionLiveStepArtifactBinding] = []
        self._closed = False
        self._publication: SessionLiveEvidencePublication | None = None

    def _open(self) -> int:
        descriptor = _open_directory(self._directory)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != self._directory_identity:
            os.close(descriptor)
            raise RuntimeError("session live evidence directory identity changed")
        return descriptor

    @property
    def publication(self) -> SessionLiveEvidencePublication | None:
        return self._publication

    @property
    def step_artifacts(self) -> tuple[SessionLiveStepArtifactBinding, ...]:
        return tuple(self._bindings)

    def record_step(self, step: SessionLiveStepBinding) -> None:
        if self._closed:
            raise RuntimeError("session live evidence sink is closed")
        if type(step) is not SessionLiveStepBinding:
            raise TypeError("session live evidence sink requires exact step bindings")
        sequence = len(self._bindings)
        body = canonical_json_bytes(_step_artifact(sequence, step))
        filename = _step_filename(sequence, step.content_sha256)
        descriptor = self._open()
        try:
            if _exists_at(descriptor, SESSION_LIVE_CLOSE_MANIFEST):
                raise FileExistsError("session live close manifest already exists")
            _publish_exclusive_at(
                descriptor,
                filename=filename,
                body=body,
                label="incremental live-step artifact",
            )
            _require_same_directory(descriptor, self._directory)
        finally:
            os.close(descriptor)
        self._bindings.append(
            SessionLiveStepArtifactBinding(
                sequence=sequence,
                step=step.step,
                execution_plan_sha256=step.execution_plan_sha256,
                content_sha256=step.content_sha256,
                filename=filename,
                size=len(body),
                raw_sha256=hashlib.sha256(body).hexdigest(),
            )
        )

    def finalize(self, result: SessionLiveContractResult) -> None:
        if self._closed:
            raise RuntimeError("session live evidence sink is closed")
        if type(result) is not SessionLiveContractResult:
            raise TypeError("session evidence requires an exact live contract result")
        result.validate()
        bindings = tuple(self._bindings)
        descriptor = self._open()
        try:
            _validate_incremental_steps(
                directory_fd=descriptor,
                result=result,
                bindings=bindings,
            )
        finally:
            os.close(descriptor)
        self._publication = publish_session_live_evidence(
            output_dir=self._directory,
            result=result,
            _incremental_steps=bindings,
        )
        self._closed = True

    def close_partial(self) -> None:
        self._closed = True


def _exists_at(directory_fd: int, filename: str) -> bool:
    try:
        os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def publish_session_live_evidence(
    *,
    output_dir: str | Path,
    result: SessionLiveContractResult,
    _incremental_steps: tuple[SessionLiveStepArtifactBinding, ...] = (),
) -> SessionLiveEvidencePublication:
    """Publish trace evidence and, only on complete success, a manifest last."""

    if type(result) is not SessionLiveContractResult:
        raise TypeError("session evidence requires an exact live contract result")
    result.validate()
    directory = _resolved_directory(output_dir)
    descriptor = _open_directory(directory)
    bindings: list[SessionLiveTraceArtifactBinding] = []
    try:
        if _exists_at(descriptor, SESSION_LIVE_CLOSE_MANIFEST):
            raise FileExistsError("session live close manifest already exists")
        if _incremental_steps:
            _validate_incremental_steps(
                directory_fd=descriptor,
                result=result,
                bindings=_incremental_steps,
            )
        for index, plan in enumerate(result.execution_plan_sha256s):
            artifact = _trace_artifact(
                result,
                trace_index=index,
                execution_plan_sha256=plan,
            )
            if artifact is None:
                continue
            body = canonical_json_bytes(artifact)
            filename = _trace_filename(index, plan)
            _publish_exclusive_at(
                descriptor,
                filename=filename,
                body=body,
                label="session live trace artifact",
            )
            bindings.append(
                SessionLiveTraceArtifactBinding(
                    trace_index=index,
                    execution_plan_sha256=plan,
                    filename=filename,
                    size=len(body),
                    raw_sha256=hashlib.sha256(body).hexdigest(),
                    complete=bool(artifact["complete"]),
                )
            )
        binding_values = tuple(bindings)
        manifest_path: Path | None = None
        manifest_sha256: str | None = None
        if result.audit.status == "CPU_CONTRACT_ONLY":
            if len(binding_values) != len(result.execution_plan_sha256s) or not all(
                binding.complete for binding in binding_values
            ):
                raise RuntimeError("complete session lost trace evidence coverage")
            manifest = _close_manifest(
                result,
                binding_values,
                incremental_steps=_incremental_steps,
            )
            body = canonical_json_bytes(manifest)
            _publish_exclusive_at(
                descriptor,
                filename=SESSION_LIVE_CLOSE_MANIFEST,
                body=body,
                label="session live close manifest",
            )
            manifest_path = directory / SESSION_LIVE_CLOSE_MANIFEST
            manifest_sha256 = str(manifest["manifest_sha256"])
        _require_same_directory(descriptor, directory)
    finally:
        os.close(descriptor)
    publication = SessionLiveEvidencePublication(
        status=result.audit.status,
        reason=result.audit.reason,
        evidence_level=SESSION_LIVE_EVIDENCE_LEVEL,
        gpu_reset_semantics=SESSION_LIVE_GPU_RESET_SEMANTICS,
        reuse_authorized=False,
        trace_artifacts=tuple(bindings),
        close_manifest_path=manifest_path,
        close_manifest_sha256=manifest_sha256,
    )
    publication.validate()
    return publication


def reopen_session_live_evidence(
    *,
    output_dir: str | Path,
    expected_result: SessionLiveContractResult,
) -> SessionLiveEvidencePublication:
    """Replay a publication against the exact typed live result.

    No digest-only reopen exists: callers must re-supply the validated native
    terminals and source-owned raw response chain that created the files.
    """

    if type(expected_result) is not SessionLiveContractResult:
        raise TypeError("session evidence reopen requires an exact live result")
    expected_result.validate()
    directory = _resolved_directory(output_dir)
    descriptor = _open_directory(directory)
    bindings: list[SessionLiveTraceArtifactBinding] = []
    try:
        observed_step_names = {
            name
            for name in os.listdir(descriptor)
            if name.startswith(SESSION_LIVE_STEP_PREFIX) and name.endswith(".json")
        }
        incremental_steps: tuple[SessionLiveStepArtifactBinding, ...] = ()
        if observed_step_names:
            rebuilt: list[SessionLiveStepArtifactBinding] = []
            for sequence, step in enumerate(expected_result.steps):
                expected = canonical_json_bytes(_step_artifact(sequence, step))
                filename = _step_filename(sequence, step.content_sha256)
                rebuilt.append(
                    SessionLiveStepArtifactBinding(
                        sequence=sequence,
                        step=step.step,
                        execution_plan_sha256=step.execution_plan_sha256,
                        content_sha256=step.content_sha256,
                        filename=filename,
                        size=len(expected),
                        raw_sha256=hashlib.sha256(expected).hexdigest(),
                    )
                )
            incremental_steps = tuple(rebuilt)
            if observed_step_names != {
                binding.filename for binding in incremental_steps
            }:
                raise RuntimeError("incremental live-step artifact coverage changed")
            _validate_incremental_steps(
                directory_fd=descriptor,
                result=expected_result,
                bindings=incremental_steps,
            )
        for index, plan in enumerate(expected_result.execution_plan_sha256s):
            expected_artifact = _trace_artifact(
                expected_result,
                trace_index=index,
                execution_plan_sha256=plan,
            )
            if expected_artifact is None:
                continue
            filename = _trace_filename(index, plan)
            observed = _read_bound_at(
                descriptor,
                filename=filename,
                label="session live trace artifact",
            )
            _strict_json(observed, label="session live trace artifact")
            expected = canonical_json_bytes(expected_artifact)
            if observed != expected:
                raise RuntimeError("session live trace artifact changed")
            bindings.append(
                SessionLiveTraceArtifactBinding(
                    trace_index=index,
                    execution_plan_sha256=plan,
                    filename=filename,
                    size=len(expected),
                    raw_sha256=hashlib.sha256(expected).hexdigest(),
                    complete=bool(expected_artifact["complete"]),
                )
            )
        binding_values = tuple(bindings)
        expected_trace_names = {binding.filename for binding in binding_values}
        observed_trace_names = {
            name
            for name in os.listdir(descriptor)
            if name.startswith("trace-") and name.endswith(".json")
        }
        if observed_trace_names != expected_trace_names:
            raise RuntimeError("session live trace artifact coverage changed")
        if expected_result.audit.status == "CPU_CONTRACT_ONLY":
            if len(binding_values) != len(expected_result.execution_plan_sha256s):
                raise RuntimeError("expected live result lacks trace evidence")
            expected_manifest = _close_manifest(
                expected_result,
                binding_values,
                incremental_steps=incremental_steps,
            )
            observed_manifest = _read_bound_at(
                descriptor,
                filename=SESSION_LIVE_CLOSE_MANIFEST,
                label="session live close manifest",
            )
            _strict_json(observed_manifest, label="session live close manifest")
            expected_manifest_body = canonical_json_bytes(expected_manifest)
            if observed_manifest != expected_manifest_body:
                raise RuntimeError("session live close manifest changed")
            manifest_path: Path | None = directory / SESSION_LIVE_CLOSE_MANIFEST
            manifest_sha256: str | None = str(expected_manifest["manifest_sha256"])
        else:
            if _exists_at(descriptor, SESSION_LIVE_CLOSE_MANIFEST):
                raise RuntimeError("partial session evidence forged a close marker")
            manifest_path = None
            manifest_sha256 = None
        _require_same_directory(descriptor, directory)
    finally:
        os.close(descriptor)
    publication = SessionLiveEvidencePublication(
        status=expected_result.audit.status,
        reason=expected_result.audit.reason,
        evidence_level=SESSION_LIVE_EVIDENCE_LEVEL,
        gpu_reset_semantics=SESSION_LIVE_GPU_RESET_SEMANTICS,
        reuse_authorized=False,
        trace_artifacts=binding_values,
        close_manifest_path=manifest_path,
        close_manifest_sha256=manifest_sha256,
    )
    publication.validate()
    return publication


__all__ = (
    "SESSION_LIVE_CLOSE_MANIFEST",
    "SESSION_LIVE_EVIDENCE_LEVEL",
    "SESSION_LIVE_GPU_RESET_SEMANTICS",
    "SESSION_LIVE_STEP_PREFIX",
    "IncrementalSessionLiveEvidenceSink",
    "SessionLiveEvidencePublication",
    "SessionLiveStepArtifactBinding",
    "SessionLiveTraceArtifactBinding",
    "publish_session_live_evidence",
    "reopen_session_live_evidence",
)
