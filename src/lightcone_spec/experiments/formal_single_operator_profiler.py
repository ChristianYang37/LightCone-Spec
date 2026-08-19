"""Trusted onsite profiler capture for ``formal_single_operator_v1``.

The adversarial release profiler path intentionally keeps an empty executable
allowlist.  A trusted single operator instead supplies an absolute onsite
``nsys`` or ``ncu`` executable.  This module hashes that executable, records
its real ``--version`` output, renders the existing code-owned profiler
template, and writes one run-specific raw report.  stdout/stderr are opened
with ``O_EXCL`` and the canonical terminal receipt is published last without
replacement.

This is raw diagnostic evidence only.  It cannot authorize headline timing,
and it does not turn a CPU fake into GPU evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
    load_formal_single_operator_execution_source,
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.experiments.profiler_authority import (
    ProfilerToolContract,
    registered_profiler_tool_contract,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_MAX_TOOL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_RAW_PROFILE_BYTES = 128 * 1024 * 1024 * 1024
_MAX_VERSION_BYTES = 64 * 1024
_MAX_SUBJECT_ARGUMENTS = 4096
_MAX_ARGUMENT_BYTES = 128 * 1024
_PROFILER_TIMEOUT_SECONDS = 60 * 60


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


FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema": "formal_single_operator_profiler_v1",
        "trust_mode": "formal_single_operator_v1",
        "current_source": (
            "deep_revalidated_e4_profiler_execution_source_and_materialized_cell"
        ),
        "tool": (
            "absolute_onsite_nsys_or_ncu_regular_executable_hashed_before_and_after"
        ),
        "version": "onsite_absolute_tool_path_plus_literal_--version",
        "subject_plan": (
            "schema2_profile_telemetry_serving_plan_derived_from_exact_e4_local_"
            "saturation_mixed_prefill_decode_schedule"
        ),
        "subject_command": (
            "code_owned_python_module_formal_single_operator_execute_run_argv"
        ),
        "command": "registered_profiler_tool_contract_template",
        "output": "new_private_directory_and_raw_report",
        "publication": "stdout_stderr_O_EXCL_then_terminal_O_EXCL_last",
        "headline_eligible": False,
    }
)


class FormalSingleOperatorProfilerBlocked(RuntimeError):
    """An onsite executable or actual raw report is unavailable."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("profiler BLOCKED reason must be text")
        self.reason = reason
        super().__init__(f"single-operator profiler is BLOCKED: {reason}")


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


def _strict_object(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _strict_array(label: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an array")
    return value


def _absolute_normalized(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != candidate.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    return candidate


def _safe_existing_parent(path: Path, *, label: str) -> Path:
    parent = path.parent
    try:
        resolved = parent.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} parent is missing") from error
    if resolved != parent or not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"{label} parent must be a symlink-free directory")
    current = parent.stat(follow_symlinks=False)
    if current.st_uid != os.geteuid() or stat.S_IMODE(current.st_mode) & 0o022:
        raise ValueError(f"{label} parent must be current-user-owned and non-writable")
    return parent


def _read_stable_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    allow_empty: bool,
    require_executable: bool = False,
) -> tuple[str, int]:
    candidate = _absolute_normalized(path, label=label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if resolved != candidate:
        raise ValueError(f"{label} must be a non-symlink path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > maximum_bytes
            or (before.st_size == 0 and not allow_empty)
            or (require_executable and not before.st_mode & 0o111)
        ):
            raise ValueError(f"{label} is not a bounded regular file")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"{label} exceeds its maximum size")
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = candidate.stat(follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if (
            observed != before.st_size
            or identity(before) != identity(after)
            or identity(after) != identity(current)
        ):
            raise RuntimeError(f"{label} changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), observed


def _stable_json_binding(
    binding: CanonicalJsonProofBinding,
    *,
    label: str,
) -> CanonicalJsonProofBinding:
    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} is not path-bound")
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError(f"{label} changed")
    return binding


@dataclass(frozen=True)
class FormalSingleOperatorProfilerSubjectRunPlanInputs:
    """Current-cell authority for one generated profiler subject run plan."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_profiler_subject_run_plan_inputs"]
    protocol_sha256: str
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    prepared_launch_bundle: CanonicalJsonProofBinding
    prepared_launch_bundle_sha256: str
    prepared_launch_entry_sha256: str
    profiler_cell_id: str
    source_headline_cell_id: str
    source_materialization: CanonicalJsonProofBinding
    source_materialization_sha256: str
    inventory: CanonicalJsonProofBinding
    inventory_sha256: str
    content_verification_receipt: CanonicalJsonProofBinding
    profile_compile_launch_manifest: CanonicalJsonProofBinding
    selected_request_schedule: CanonicalJsonProofBinding
    repository_root: str
    private_output_root: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_profiler_subject_run_plan_inputs"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256
        ):
            raise ValueError("profiler subject input schema differs")
        for label, value in (
            ("execution source", self.execution_source_sha256),
            ("prepared bundle", self.prepared_launch_bundle_sha256),
            ("prepared entry", self.prepared_launch_entry_sha256),
            ("profiler cell", self.profiler_cell_id),
            ("headline cell", self.source_headline_cell_id),
            ("source materialization", self.source_materialization_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"profiler subject {label}", value)
        for label, binding in (
            ("execution source", self.execution_source),
            ("prepared bundle", self.prepared_launch_bundle),
            ("source materialization", self.source_materialization),
            ("inventory", self.inventory),
            ("content receipt", self.content_verification_receipt),
            ("profile launch", self.profile_compile_launch_manifest),
            ("selected schedule", self.selected_request_schedule),
        ):
            _stable_json_binding(binding, label=f"profiler subject {label}")
        repository = _absolute_normalized(
            self.repository_root,
            label="profiler subject repository root",
        )
        root = _absolute_normalized(
            self.private_output_root,
            label="profiler subject output root",
        )
        if (
            not repository.is_dir()
            or repository.is_symlink()
            or not root.is_dir()
            or root.is_symlink()
        ):
            raise ValueError("profiler subject local roots differ")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    @cached_property
    def subject_sha256(self) -> str:
        return _content_sha256(
            {
                "kind": "formal_single_operator_profiler_subject",
                "inputs_sha256": self.sha256,
                "profile_compile_launch_manifest_sha256": (
                    self.profile_compile_launch_manifest.semantic_sha256
                ),
                "selected_request_schedule_sha256": (
                    self.selected_request_schedule.semantic_sha256
                ),
            }
        )

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        for name in (
            "execution_source",
            "prepared_launch_bundle",
            "source_materialization",
            "inventory",
            "content_verification_receipt",
            "profile_compile_launch_manifest",
            "selected_request_schedule",
        ):
            value[name] = getattr(self, name).to_dict()
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "profiler subject run-plan inputs",
            value,
            set(cls.__dataclass_fields__),
        )
        for name in (
            "execution_source",
            "prepared_launch_bundle",
            "source_materialization",
            "inventory",
            "content_verification_receipt",
            "profile_compile_launch_manifest",
            "selected_request_schedule",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorProfilerToolIdentity:
    tool: Literal["nsys", "ncu"]
    absolute_path: str
    raw_sha256: str
    size_bytes: int
    version_stdout: str
    version_stderr: str
    version_record_sha256: str

    def __post_init__(self) -> None:
        if self.tool not in {"nsys", "ncu"}:
            raise ValueError("single-operator profiler tool is unsupported")
        path = _absolute_normalized(self.absolute_path, label="profiler tool")
        if path.name != self.tool:
            raise ValueError("profiler tool basename differs from its contract")
        _require_sha256("profiler tool bytes", self.raw_sha256)
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("profiler tool size is invalid")
        for label, value in (
            ("profiler version stdout", self.version_stdout),
            ("profiler version stderr", self.version_stderr),
        ):
            if type(value) is not str or "\x00" in value:
                raise ValueError(f"{label} is invalid")
        if not self.version_stdout and not self.version_stderr:
            raise ValueError("profiler --version produced no output")
        expected = _content_sha256(
            {
                "argv": [self.absolute_path, "--version"],
                "stdout": self.version_stdout,
                "stderr": self.version_stderr,
            }
        )
        if self.version_record_sha256 != expected:
            raise ValueError("profiler version record digest differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "absolute_path": self.absolute_path,
            "raw_sha256": self.raw_sha256,
            "size_bytes": self.size_bytes,
            "version_stdout": self.version_stdout,
            "version_stderr": self.version_stderr,
            "version_record_sha256": self.version_record_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "single-operator profiler tool identity",
                value,
                set(cls.__dataclass_fields__),
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorProfilerCapturePlan:
    """Executable wrapper around one generated, source-owned subject plan."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_profiler_capture_plan"]
    protocol_sha256: str
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    prepared_launch_bundle: CanonicalJsonProofBinding
    prepared_launch_bundle_sha256: str
    prepared_launch_entry_sha256: str
    materialized_cell_id: str
    variant: Literal["nvtx", "nsight_systems", "nsight_compute"]
    subject_inputs: CanonicalJsonProofBinding
    subject_inputs_sha256: str
    subject_run_plan: CanonicalJsonProofBinding
    subject_run_plan_sha256: str
    subject_argv: tuple[str, ...]
    subject_argv_sha256: str
    tool_identity: FormalSingleOperatorProfilerToolIdentity
    repository_root: str
    output_directory: str
    created_ns: int
    headline_eligible: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_profiler_capture_plan"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256
            or self.variant not in {"nvtx", "nsight_systems", "nsight_compute"}
            or self.headline_eligible is not False
        ):
            raise ValueError("profiler capture plan schema differs")
        for label, value in (
            ("execution source", self.execution_source_sha256),
            ("prepared bundle", self.prepared_launch_bundle_sha256),
            ("prepared entry", self.prepared_launch_entry_sha256),
            ("cell", self.materialized_cell_id),
            ("subject inputs", self.subject_inputs_sha256),
            ("subject plan", self.subject_run_plan_sha256),
            ("subject argv", self.subject_argv_sha256),
        ):
            _require_sha256(f"profiler capture {label}", value)
        for label, binding in (
            ("execution source", self.execution_source),
            ("prepared bundle", self.prepared_launch_bundle),
            ("subject inputs", self.subject_inputs),
            ("subject plan", self.subject_run_plan),
        ):
            _stable_json_binding(binding, label=f"profiler capture {label}")
        subject = _validate_subject_argv(self.subject_argv)
        if self.subject_argv_sha256 != _content_sha256({"argv": list(subject)}):
            raise ValueError("profiler capture subject argv digest differs")
        if type(self.tool_identity) is not FormalSingleOperatorProfilerToolIdentity:
            raise TypeError("profiler capture tool identity differs")
        contract = registered_profiler_tool_contract(self.variant)
        if self.tool_identity.tool != contract.tool:
            raise ValueError("profiler capture tool differs from variant")
        repository = _absolute_normalized(
            self.repository_root,
            label="profiler capture repository root",
        )
        output = _absolute_normalized(
            self.output_directory,
            label="profiler capture output directory",
        )
        if (
            not repository.is_dir()
            or repository.is_symlink()
            or type(self.created_ns) is not int
            or self.created_ns < 1
        ):
            raise ValueError("profiler capture local plan differs")
        _safe_existing_parent(output, label="profiler capture output directory")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        for name in (
            "execution_source",
            "prepared_launch_bundle",
            "subject_inputs",
            "subject_run_plan",
        ):
            value[name] = getattr(self, name).to_dict()
        value["subject_argv"] = list(self.subject_argv)
        value["tool_identity"] = self.tool_identity.to_dict()
        if include_sha256:
            value["plan_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "profiler capture plan",
            value,
            {*cls.__dataclass_fields__, "plan_sha256"},
        )
        declared = _require_sha256("profiler capture plan", row.pop("plan_sha256"))
        for name in (
            "execution_source",
            "prepared_launch_bundle",
            "subject_inputs",
            "subject_run_plan",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_argv = row.pop("subject_argv")
        if type(raw_argv) is not list:
            raise TypeError("profiler capture subject argv must be an array")
        row["tool_identity"] = FormalSingleOperatorProfilerToolIdentity.from_dict(
            row["tool_identity"]
        )
        result = cls(**row, subject_argv=tuple(raw_argv))  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("profiler capture plan digest differs")
        return result


def probe_formal_single_operator_profiler_tool(
    *,
    expected_tool: Literal["nsys", "ncu"],
    tool_path: str | Path,
) -> FormalSingleOperatorProfilerToolIdentity:
    """Hash one absolute executable and record its actual ``--version`` output."""

    path = _absolute_normalized(tool_path, label="profiler tool")
    if path.name != expected_tool:
        raise ValueError("profiler executable basename differs from registered tool")
    raw_sha256, size_bytes = _read_stable_regular_file(
        path,
        label="profiler tool",
        maximum_bytes=_MAX_TOOL_BYTES,
        allow_empty=False,
        require_executable=True,
    )
    try:
        completed = subprocess.run(
            (str(path), "--version"),
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise FormalSingleOperatorProfilerBlocked(
            "profiler_tool_version_probe_failed"
        ) from error
    if completed.returncode != 0:
        raise FormalSingleOperatorProfilerBlocked("profiler_tool_version_probe_failed")
    if len(completed.stdout) + len(completed.stderr) > _MAX_VERSION_BYTES:
        raise ValueError("profiler --version output is too large")
    try:
        stdout = completed.stdout.decode("utf-8")
        stderr = completed.stderr.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("profiler --version output is not UTF-8") from error
    observed_sha256, observed_size = _read_stable_regular_file(
        path,
        label="profiler tool after version probe",
        maximum_bytes=_MAX_TOOL_BYTES,
        allow_empty=False,
        require_executable=True,
    )
    if (observed_sha256, observed_size) != (raw_sha256, size_bytes):
        raise RuntimeError("profiler tool changed during version probe")
    return FormalSingleOperatorProfilerToolIdentity(
        tool=expected_tool,
        absolute_path=str(path),
        raw_sha256=raw_sha256,
        size_bytes=size_bytes,
        version_stdout=stdout,
        version_stderr=stderr,
        version_record_sha256=_content_sha256(
            {
                "argv": [str(path), "--version"],
                "stdout": stdout,
                "stderr": stderr,
            }
        ),
    )


def _validate_subject_argv(subject_argv: tuple[str, ...]) -> tuple[str, ...]:
    if (
        type(subject_argv) is not tuple
        or not subject_argv
        or len(subject_argv) > _MAX_SUBJECT_ARGUMENTS
        or any(
            type(value) is not str
            or not value
            or "\x00" in value
            or len(value.encode("utf-8")) > _MAX_ARGUMENT_BYTES
            for value in subject_argv
        )
    ):
        raise ValueError("profiler subject argv is invalid")
    return subject_argv


def _render_formal_single_operator_profiler_command(
    *,
    contract: ProfilerToolContract,
    tool_path: str | Path,
    output_base: str | Path,
    subject_argv: tuple[str, ...],
) -> tuple[str, ...]:
    """Render the registered template without a shell or caller flags."""

    if type(contract) is not ProfilerToolContract:
        raise TypeError("profiler command requires an exact registered contract")
    expected = registered_profiler_tool_contract(contract.variant)
    if contract != expected:
        raise ValueError("profiler command contract is not code-owned")
    tool = _absolute_normalized(tool_path, label="profiler tool")
    if tool.name != contract.tool:
        raise ValueError("profiler tool path differs from registered contract")
    output = _absolute_normalized(output_base, label="profiler output base")
    subject = _validate_subject_argv(subject_argv)
    rendered: list[str] = []
    for index, argument in enumerate(contract.command_template):
        if index == 0:
            if argument != contract.tool:
                raise ValueError("profiler template tool token differs")
            rendered.append(str(tool))
        elif argument == "{subject_argv}":
            rendered.extend(subject)
        else:
            rendered.append(argument.replace("{output_base}", str(output)))
    if "{subject_argv}" not in contract.command_template:
        raise ValueError("profiler template lacks its subject placeholder")
    return tuple(rendered)


def _contract_from_dict(value: object) -> ProfilerToolContract:
    row = _strict_object(
        "single-operator profiler contract",
        value,
        {
            "variant",
            "tool",
            "command_template",
            "required_metrics",
            "raw_profile_role",
        },
    )
    return ProfilerToolContract(
        variant=row["variant"],  # type: ignore[arg-type]
        tool=row["tool"],  # type: ignore[arg-type]
        command_template=tuple(
            str(item)
            for item in _strict_array(
                "single-operator profiler command template",
                row["command_template"],
            )
        ),
        required_metrics=tuple(
            str(item)
            for item in _strict_array(
                "single-operator profiler required metrics",
                row["required_metrics"],
            )
        ),
        raw_profile_role=row["raw_profile_role"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class FormalSingleOperatorProfilerTerminal:
    schema_version: int
    kind: Literal["formal_single_operator_profiler_terminal"]
    protocol_sha256: str
    execution_source: FormalSingleOperatorJsonBinding
    execution_source_sha256: str
    protocol_lock_sha256: str
    materialization_sha256: str
    cell_id: str
    variant: Literal["nvtx", "nsight_systems", "nsight_compute"]
    tool_contract: ProfilerToolContract
    tool_identity: FormalSingleOperatorProfilerToolIdentity
    subject_argv: tuple[str, ...]
    subject_argv_sha256: str
    rendered_argv: tuple[str, ...]
    rendered_argv_sha256: str
    process_id: int
    started_ns: int
    finished_ns: int
    published_ns: int
    exit_code: int
    status: Literal["COMPLETE", "FAILED"]
    failure_reason: str | None
    stdout_relative_path: str
    stdout_raw_sha256: str
    stdout_size_bytes: int
    stderr_relative_path: str
    stderr_raw_sha256: str
    stderr_size_bytes: int
    raw_profile_role: Literal["nsys_report", "ncu_report"]
    raw_profile_relative_path: str | None
    raw_profile_sha256: str | None
    raw_profile_size_bytes: int | None
    headline_eligible: bool

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_profiler_terminal"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator profiler terminal schema differs")
        if type(self.execution_source) is not FormalSingleOperatorJsonBinding:
            raise TypeError("profiler terminal lacks its current execution source")
        for label, value in (
            ("execution source", self.execution_source_sha256),
            ("ProtocolLock", self.protocol_lock_sha256),
            ("materialization", self.materialization_sha256),
            ("cell", self.cell_id),
            ("subject argv", self.subject_argv_sha256),
            ("rendered argv", self.rendered_argv_sha256),
            ("stdout", self.stdout_raw_sha256),
            ("stderr", self.stderr_raw_sha256),
        ):
            _require_sha256(f"single-operator profiler {label}", value)
        expected_contract = registered_profiler_tool_contract(self.variant)
        if self.tool_contract != expected_contract:
            raise ValueError("profiler terminal contract differs from registration")
        if (
            type(self.tool_identity) is not FormalSingleOperatorProfilerToolIdentity
            or self.tool_identity.tool != self.tool_contract.tool
            or self.raw_profile_role != self.tool_contract.raw_profile_role
        ):
            raise ValueError("profiler terminal tool identity differs")
        subject = _validate_subject_argv(self.subject_argv)
        if self.subject_argv_sha256 != _content_sha256({"argv": list(subject)}):
            raise ValueError("profiler subject argv digest differs")
        expected_rendered = _render_formal_single_operator_profiler_command(
            contract=self.tool_contract,
            tool_path=self.tool_identity.absolute_path,
            output_base="/profiler/output/base",
            subject_argv=subject,
        )
        if len(expected_rendered) != len(self.rendered_argv):
            raise ValueError("profiler rendered argv shape differs")
        if self.rendered_argv_sha256 != _content_sha256(
            {"argv": list(self.rendered_argv)}
        ):
            raise ValueError("profiler rendered argv digest differs")
        if (
            type(self.process_id) is not int
            or self.process_id <= 0
            or type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or type(self.published_ns) is not int
            or self.started_ns <= 0
            or self.finished_ns <= self.started_ns
            or self.published_ns < self.finished_ns
            or type(self.exit_code) is not int
        ):
            raise ValueError("profiler terminal process/timing differs")
        for label, path_value, size in (
            ("stdout", self.stdout_relative_path, self.stdout_size_bytes),
            ("stderr", self.stderr_relative_path, self.stderr_size_bytes),
        ):
            _require_text(f"profiler {label} path", path_value)
            path = Path(path_value)
            if path.is_absolute() or ".." in path.parts or path == Path("."):
                raise ValueError(f"profiler {label} path is unsafe")
            if type(size) is not int or size < 0:
                raise ValueError(f"profiler {label} size is invalid")
        if (
            self.stdout_relative_path != "profiler.stdout"
            or self.stderr_relative_path != "profiler.stderr"
        ):
            raise ValueError("profiler stdout/stderr paths are not code-owned")
        raw_values = (
            self.raw_profile_relative_path,
            self.raw_profile_sha256,
            self.raw_profile_size_bytes,
        )
        if any(value is None for value in raw_values) != all(
            value is None for value in raw_values
        ):
            raise ValueError("profiler raw output identity is partial")
        if self.raw_profile_relative_path is not None:
            raw_path = Path(self.raw_profile_relative_path)
            if (
                raw_path.is_absolute()
                or ".." in raw_path.parts
                or raw_path == Path(".")
            ):
                raise ValueError("profiler raw output path is unsafe")
            expected_raw_name = (
                "profile.nsys-rep"
                if self.tool_contract.tool == "nsys"
                else "profile.ncu-rep"
            )
            if self.raw_profile_relative_path != expected_raw_name:
                raise ValueError("profiler raw output path is not code-owned")
            _require_sha256("profiler raw output", self.raw_profile_sha256)
            if (
                type(self.raw_profile_size_bytes) is not int
                or self.raw_profile_size_bytes <= 0
            ):
                raise ValueError("profiler raw output size is invalid")
        if self.status == "COMPLETE":
            if (
                self.exit_code != 0
                or self.failure_reason is not None
                or self.raw_profile_relative_path is None
            ):
                raise ValueError("complete profiler terminal is incomplete")
        elif self.status == "FAILED":
            _require_text("profiler failure reason", self.failure_reason)
        else:
            raise ValueError("profiler terminal status differs")
        if self.headline_eligible is not False:
            raise ValueError("profiler terminal cannot authorize headline evidence")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        value["execution_source"] = self.execution_source.to_dict()
        value["tool_contract"] = self.tool_contract.to_dict()
        value["tool_identity"] = self.tool_identity.to_dict()
        value["subject_argv"] = list(self.subject_argv)
        value["rendered_argv"] = list(self.rendered_argv)
        if include_sha256:
            value["terminal_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "single-operator profiler terminal",
            value,
            set(cls.__dataclass_fields__) | {"terminal_sha256"},
        )
        expected = _require_sha256(
            "single-operator profiler terminal",
            row.pop("terminal_sha256"),
        )
        row["execution_source"] = FormalSingleOperatorJsonBinding.from_dict(
            row["execution_source"]
        )
        row["tool_contract"] = _contract_from_dict(row["tool_contract"])
        row["tool_identity"] = FormalSingleOperatorProfilerToolIdentity.from_dict(
            row["tool_identity"]
        )
        row["subject_argv"] = tuple(
            str(item)
            for item in _strict_array("profiler subject argv", row["subject_argv"])
        )
        row["rendered_argv"] = tuple(
            str(item)
            for item in _strict_array("profiler rendered argv", row["rendered_argv"])
        )
        terminal = cls(**row)  # type: ignore[arg-type]
        if terminal.sha256 != expected:
            raise ValueError("single-operator profiler terminal digest differs")
        return terminal


def _current_profiler_context(
    execution_source_path: str | Path,
    *,
    materialized_cell_id: str,
) -> tuple[
    FormalSingleOperatorJsonBinding,
    object,
    object,
    ProfilerToolContract,
]:
    source_binding = FormalSingleOperatorJsonBinding.bind(
        execution_source_path,
        label="single-operator profiler execution source",
    )
    source = load_formal_single_operator_execution_source(source_binding.absolute_path)
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="single-operator profiler materialization"
        )
    )
    cells = tuple(
        cell for cell in materialization.cells if cell.cell_id == materialized_cell_id
    )
    if len(cells) != 1:
        raise ValueError("profiler cell is outside current materialization")
    cell = cells[0]
    dimensions = dict(cell.dimensions)
    variant = dimensions.get("profiler")
    if (
        source.node != "e4_profiler"
        or source.stage != "E4"
        or source.phase != "profiler"
        or materialization.stage != "E4"
        or materialization.materialization_rule
        != "three_profiler_only_rows_separate_from_headline"
        or len(materialization.cells) != 3
        or cell.stage != "E4"
        or cell.method_role != "LightCone"
        or cell.task != "mechanism_profile_only"
        or cell.publication_policy != "diagnostic_only"
        or variant not in {"nvtx", "nsight_systems", "nsight_compute"}
    ):
        raise ValueError("current source is not an exact E4 profiler cell")
    contract = registered_profiler_tool_contract(str(variant))
    return source_binding, source, materialization, contract


def _profiler_subject_argv(
    *,
    repository_root: str | Path,
    run_plan_path: str | Path,
) -> tuple[str, ...]:
    """Render the only command which may be placed behind a profiler tool."""

    repository = _absolute_normalized(
        repository_root,
        label="profiler subject repository root",
    )
    plan = _absolute_normalized(
        run_plan_path,
        label="profiler subject run plan",
    )
    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_file() or executable.is_symlink():
        raise ValueError("profiler subject Python executable is unavailable")
    return (
        str(executable),
        "-m",
        "lightcone_spec.cli.main",
        "formal-single-operator",
        "execute-run",
        "--repository-root",
        str(repository),
        "--run-plan",
        str(plan),
    )


def _revalidate_profiler_subject_argv(
    subject_argv: tuple[str, ...],
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    current_ns: int,
) -> None:
    """Reject any subject command not regenerated from the current plan."""

    subject = _validate_subject_argv(subject_argv)
    if (
        len(subject) != 9
        or subject[1:6]
        != (
            "-m",
            "lightcone_spec.cli.main",
            "formal-single-operator",
            "execute-run",
            "--repository-root",
        )
        or subject[7] != "--run-plan"
    ):
        raise ValueError("profiler subject command is not code-owned")
    repository_root = subject[6]
    run_plan_path = subject[8]
    if subject != _profiler_subject_argv(
        repository_root=repository_root,
        run_plan_path=run_plan_path,
    ):
        raise ValueError("profiler subject command differs from code-owned argv")
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        revalidate_formal_single_operator_profiler_subject_run_plan,
    )

    plan = revalidate_formal_single_operator_profiler_subject_run_plan(
        run_plan_path,
        current_ns=current_ns,
    )
    assert plan.single_operator_execution_rebuild_source is not None
    inputs = revalidate_formal_single_operator_profiler_subject_inputs(
        plan.single_operator_execution_rebuild_source.absolute_path,
        current_ns=current_ns,
    )
    if (
        inputs.execution_source != CanonicalJsonProofBinding.bind(execution_source_path)
        or inputs.profiler_cell_id != materialized_cell_id
        or inputs.repository_root != repository_root
        or plan.materialized_cell_id != inputs.source_headline_cell_id
    ):
        raise ValueError("profiler subject command names another current cell")


def revalidate_formal_single_operator_profiler_subject_inputs(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalSingleOperatorProfilerSubjectRunPlanInputs:
    """Deep-reopen one profiler subject descriptor against the current DAG."""

    if type(current_ns) is not int or current_ns < 1:
        raise ValueError("profiler subject revalidation time is invalid")
    binding = CanonicalJsonProofBinding.bind(path)
    inputs = FormalSingleOperatorProfilerSubjectRunPlanInputs.from_dict(
        binding.reopen()
    )
    if inputs.sha256 != binding.semantic_sha256:
        raise ValueError("profiler subject input digest differs")
    from lightcone_spec.config import load_run_config
    from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
        revalidate_formal_single_operator_prepared_launch_bundle,
    )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
    )
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

    source_binding, source, _materialization, _contract = _current_profiler_context(
        inputs.execution_source.absolute_path,
        materialized_cell_id=inputs.profiler_cell_id,
    )
    validated = revalidate_formal_single_operator_prepared_launch_bundle(
        execution_source_path=inputs.execution_source.absolute_path,
        prepared_launch_bundle_path=inputs.prepared_launch_bundle.absolute_path,
        materialized_cell_id=inputs.profiler_cell_id,
        current_ns=current_ns,
    )
    entry = validated.entry(inputs.profiler_cell_id)
    requirement = entry.profiler_subject
    if entry.physical_kind != "profiler" or requirement is None:
        raise ValueError("profiler subject entry route differs")
    schedule = FormalServingRequestScheduleReceipt.from_dict(
        inputs.selected_request_schedule.reopen()
    )
    schedule.reopen()
    launch = CompileLaunchManifest.load(
        inputs.profile_compile_launch_manifest.absolute_path
    )
    config = load_run_config(launch.run_config_path)
    source_materialization = stage_materialization_receipt_from_dict(
        inputs.source_materialization.reopen()
    )
    root = Path(inputs.private_output_root)
    if (
        Path(binding.absolute_path)
        != root / "formal-single-operator-profiler-subject-inputs.json"
        or source_binding.absolute_path != inputs.execution_source.absolute_path
        or source.sha256 != inputs.execution_source_sha256
        or validated.bundle.sha256 != inputs.prepared_launch_bundle_sha256
        or entry.sha256 != inputs.prepared_launch_entry_sha256
        or requirement.source_headline_cell_id != inputs.source_headline_cell_id
        or requirement.code_owned_request_schedule != inputs.selected_request_schedule
        or entry.compile_launch_manifest != inputs.profile_compile_launch_manifest
        or validated.bundle.inventory != inputs.inventory
        or validated.bundle.inventory.semantic_sha256 != inputs.inventory_sha256
        or validated.bundle.content_verification_receipt
        != inputs.content_verification_receipt
        or schedule.materialization != inputs.source_materialization
        or schedule.materialized_cell_id != inputs.source_headline_cell_id
        or source_materialization.sha256 != inputs.source_materialization_sha256
        or launch.sha256 != inputs.profile_compile_launch_manifest.semantic_sha256
        or launch.inventory_sha256 != inputs.inventory_sha256
        or config.runtime.telemetry_detail != "profile"
        or Path(inputs.repository_root).resolve(strict=True)
        != Path(inputs.repository_root)
    ):
        raise ValueError("profiler subject current authority differs")
    return inputs


def materialize_formal_single_operator_profiler_plan(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    prepared_launch_bundle_path: str | Path,
    repository_root: str | Path,
    private_output_root: str | Path,
    tool_path: str | Path,
    current_ns: int,
) -> FormalSingleOperatorProfilerCapturePlan:
    """Generate the subject serving plan and the outer profiler capture plan."""

    if type(current_ns) is not int or current_ns < 1:
        raise ValueError("profiler plan creation time is invalid")
    from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
        revalidate_formal_single_operator_prepared_launch_bundle,
    )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
        materialize_formal_single_operator_profiler_subject_run_plan,
    )

    source_binding, source, _materialization, contract = _current_profiler_context(
        execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    validated = revalidate_formal_single_operator_prepared_launch_bundle(
        execution_source_path=execution_source_path,
        prepared_launch_bundle_path=prepared_launch_bundle_path,
        materialized_cell_id=materialized_cell_id,
        current_ns=current_ns,
    )
    entry = validated.entry(materialized_cell_id)
    requirement = entry.profiler_subject
    if entry.physical_kind != "profiler" or requirement is None:
        raise ValueError("profiler plan requires one prepared profiler entry")
    selected_schedule = FormalServingRequestScheduleReceipt.from_dict(
        requirement.code_owned_request_schedule.reopen()
    )
    selected_schedule.reopen()
    root = _absolute_normalized(
        private_output_root,
        label="profiler run root",
    )
    if not root.is_dir() or root.is_symlink():
        raise ValueError("profiler run root is unavailable")
    subject_root = root / "subject"
    os.mkdir(subject_root, 0o700)
    inputs = FormalSingleOperatorProfilerSubjectRunPlanInputs(
        schema_version=1,
        kind="formal_single_operator_profiler_subject_run_plan_inputs",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256,
        execution_source=CanonicalJsonProofBinding.bind(source_binding.absolute_path),
        execution_source_sha256=source.sha256,
        prepared_launch_bundle=CanonicalJsonProofBinding.bind(
            prepared_launch_bundle_path
        ),
        prepared_launch_bundle_sha256=validated.bundle.sha256,
        prepared_launch_entry_sha256=entry.sha256,
        profiler_cell_id=materialized_cell_id,
        source_headline_cell_id=requirement.source_headline_cell_id,
        source_materialization=selected_schedule.materialization,
        source_materialization_sha256=(
            selected_schedule.materialization.semantic_sha256
        ),
        inventory=validated.bundle.inventory,
        inventory_sha256=validated.bundle.inventory.semantic_sha256,
        content_verification_receipt=(validated.bundle.content_verification_receipt),
        profile_compile_launch_manifest=entry.compile_launch_manifest,
        selected_request_schedule=requirement.code_owned_request_schedule,
        repository_root=str(
            _absolute_normalized(
                repository_root,
                label="profiler repository root",
            )
        ),
        private_output_root=str(subject_root),
    )
    inputs_path = subject_root / "formal-single-operator-profiler-subject-inputs.json"
    publish_canonical_json_no_replace(inputs_path, inputs.to_dict())
    rebound = revalidate_formal_single_operator_profiler_subject_inputs(
        inputs_path,
        current_ns=current_ns,
    )
    if rebound != inputs:
        raise RuntimeError("profiler subject inputs changed")
    subject_plan = materialize_formal_single_operator_profiler_subject_run_plan(
        profiler_subject_inputs_path=inputs_path,
        current_ns=current_ns,
    )
    subject_plan_path = subject_root / "formal-serving-run-plan.json"
    subject_argv = _profiler_subject_argv(
        repository_root=repository_root,
        run_plan_path=subject_plan_path,
    )
    identity = probe_formal_single_operator_profiler_tool(
        expected_tool=contract.tool,
        tool_path=tool_path,
    )
    plan = FormalSingleOperatorProfilerCapturePlan(
        schema_version=1,
        kind="formal_single_operator_profiler_capture_plan",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256,
        execution_source=inputs.execution_source,
        execution_source_sha256=inputs.execution_source_sha256,
        prepared_launch_bundle=inputs.prepared_launch_bundle,
        prepared_launch_bundle_sha256=inputs.prepared_launch_bundle_sha256,
        prepared_launch_entry_sha256=inputs.prepared_launch_entry_sha256,
        materialized_cell_id=materialized_cell_id,
        variant=contract.variant,
        subject_inputs=CanonicalJsonProofBinding.bind(inputs_path),
        subject_inputs_sha256=inputs.sha256,
        subject_run_plan=CanonicalJsonProofBinding.bind(subject_plan_path),
        subject_run_plan_sha256=subject_plan.sha256,
        subject_argv=subject_argv,
        subject_argv_sha256=_content_sha256({"argv": list(subject_argv)}),
        tool_identity=identity,
        repository_root=str(Path(repository_root)),
        output_directory=str(root / "capture"),
        created_ns=current_ns,
        headline_eligible=False,
    )
    plan_path = root / "formal-single-operator-profiler-plan.json"
    publish_canonical_json_no_replace(plan_path, plan.to_dict())
    reopened = revalidate_formal_single_operator_profiler_plan(
        plan_path,
        current_ns=current_ns,
    )
    if reopened != plan:
        raise RuntimeError("profiler capture plan changed")
    return plan


def revalidate_formal_single_operator_profiler_plan(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalSingleOperatorProfilerCapturePlan:
    """Deep-reopen one capture plan and regenerate its complete subject argv."""

    binding = CanonicalJsonProofBinding.bind(path)
    plan = FormalSingleOperatorProfilerCapturePlan.from_dict(binding.reopen())
    if plan.sha256 != binding.semantic_sha256:
        raise ValueError("profiler capture plan semantic digest differs")
    _source_binding, source, _materialization, contract = _current_profiler_context(
        plan.execution_source.absolute_path,
        materialized_cell_id=plan.materialized_cell_id,
    )
    inputs = revalidate_formal_single_operator_profiler_subject_inputs(
        plan.subject_inputs.absolute_path,
        current_ns=current_ns,
    )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        revalidate_formal_single_operator_profiler_subject_run_plan,
    )

    subject_plan = revalidate_formal_single_operator_profiler_subject_run_plan(
        plan.subject_run_plan.absolute_path,
        current_ns=current_ns,
    )
    expected_argv = _profiler_subject_argv(
        repository_root=plan.repository_root,
        run_plan_path=plan.subject_run_plan.absolute_path,
    )
    observed_tool = probe_formal_single_operator_profiler_tool(
        expected_tool=contract.tool,
        tool_path=plan.tool_identity.absolute_path,
    )
    root = Path(binding.absolute_path).parent
    if (
        Path(binding.absolute_path)
        != root / "formal-single-operator-profiler-plan.json"
        or source.sha256 != plan.execution_source_sha256
        or inputs.sha256 != plan.subject_inputs_sha256
        or inputs.prepared_launch_bundle != plan.prepared_launch_bundle
        or inputs.prepared_launch_bundle_sha256 != plan.prepared_launch_bundle_sha256
        or inputs.prepared_launch_entry_sha256 != plan.prepared_launch_entry_sha256
        or inputs.profiler_cell_id != plan.materialized_cell_id
        or subject_plan.sha256 != plan.subject_run_plan_sha256
        or expected_argv != plan.subject_argv
        or observed_tool != plan.tool_identity
        or plan.variant != contract.variant
        or Path(plan.output_directory) != root / "capture"
    ):
        raise ValueError("profiler capture plan current authority differs")
    return plan


def _exclusive_output_directory(path: str | Path) -> Path:
    destination = _absolute_normalized(path, label="profiler output directory")
    _safe_existing_parent(destination, label="profiler output directory")
    try:
        os.mkdir(destination, 0o700)
    except FileExistsError as error:
        raise RuntimeError("profiler output directory already exists") from error
    return destination


def _open_exclusive(path: Path, *, label: str) -> int:
    _safe_existing_parent(path, label=label)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"{label} already exists") from error


def _raw_profile_path(output_base: Path, tool: str) -> Path:
    if tool == "nsys":
        return Path(f"{output_base}.nsys-rep")
    if tool == "ncu":
        return Path(f"{output_base}.ncu-rep")
    raise ValueError("profiler raw output tool differs")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


def _run_formal_single_operator_profiler_capture(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    tool_path: str | Path,
    subject_argv: tuple[str, ...],
    output_directory: str | Path,
) -> FormalSingleOperatorProfilerTerminal:
    """Run one current E4 profiler cell and publish its terminal receipt last."""

    source_binding, source, materialization, contract = _current_profiler_context(
        execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    subject = _validate_subject_argv(subject_argv)
    _revalidate_profiler_subject_argv(
        subject,
        execution_source_path=source_binding.absolute_path,
        materialized_cell_id=materialized_cell_id,
        current_ns=time.time_ns(),
    )
    destination = _absolute_normalized(
        output_directory,
        label="profiler output directory",
    )
    if destination.exists():
        raise RuntimeError("profiler output directory already exists")
    identity = probe_formal_single_operator_profiler_tool(
        expected_tool=contract.tool,
        tool_path=tool_path,
    )
    run_root = _exclusive_output_directory(destination)
    output_base = run_root / "profile"
    command = _render_formal_single_operator_profiler_command(
        contract=contract,
        tool_path=identity.absolute_path,
        output_base=output_base,
        subject_argv=subject,
    )
    stdout_path = run_root / "profiler.stdout"
    stderr_path = run_root / "profiler.stderr"
    stdout_descriptor = _open_exclusive(stdout_path, label="profiler stdout")
    try:
        stderr_descriptor = _open_exclusive(stderr_path, label="profiler stderr")
    except BaseException:
        os.close(stdout_descriptor)
        raise
    started_ns = time.time_ns()
    timed_out = False
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_descriptor,
            stderr=stderr_descriptor,
            close_fds=True,
            start_new_session=True,
        )
        try:
            process.wait(timeout=_PROFILER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
    finally:
        os.close(stdout_descriptor)
        os.close(stderr_descriptor)
    finished_ns = max(time.time_ns(), started_ns + 1)
    tool_after = probe_formal_single_operator_profiler_tool(
        expected_tool=contract.tool,
        tool_path=identity.absolute_path,
    )
    if tool_after != identity:
        raise RuntimeError("profiler tool identity changed during execution")
    stdout_sha256, stdout_size = _read_stable_regular_file(
        stdout_path,
        label="profiler stdout",
        maximum_bytes=128 * 1024 * 1024,
        allow_empty=True,
    )
    stderr_sha256, stderr_size = _read_stable_regular_file(
        stderr_path,
        label="profiler stderr",
        maximum_bytes=128 * 1024 * 1024,
        allow_empty=True,
    )
    raw_path = _raw_profile_path(output_base, contract.tool)
    if raw_path.exists():
        raw_sha256, raw_size = _read_stable_regular_file(
            raw_path,
            label="profiler raw output",
            maximum_bytes=_MAX_RAW_PROFILE_BYTES,
            allow_empty=False,
        )
        raw_relative_path: str | None = raw_path.name
    else:
        raw_sha256 = None
        raw_size = None
        raw_relative_path = None
    exit_code = process.returncode
    if type(exit_code) is not int:
        raise RuntimeError("profiler process has no terminal exit code")
    if timed_out:
        status: Literal["COMPLETE", "FAILED"] = "FAILED"
        failure_reason = "profiler_process_timeout"
    elif exit_code != 0:
        status = "FAILED"
        failure_reason = "profiler_process_failed"
    elif raw_relative_path is None:
        status = "FAILED"
        failure_reason = "profiler_raw_profile_missing"
    else:
        status = "COMPLETE"
        failure_reason = None
    terminal = FormalSingleOperatorProfilerTerminal(
        schema_version=1,
        kind="formal_single_operator_profiler_terminal",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256,
        execution_source=source_binding,
        execution_source_sha256=source.sha256,
        protocol_lock_sha256=source.protocol_lock_sha256,
        materialization_sha256=materialization.sha256,
        cell_id=materialized_cell_id,
        variant=contract.variant,
        tool_contract=contract,
        tool_identity=identity,
        subject_argv=subject,
        subject_argv_sha256=_content_sha256({"argv": list(subject)}),
        rendered_argv=command,
        rendered_argv_sha256=_content_sha256({"argv": list(command)}),
        process_id=process.pid,
        started_ns=started_ns,
        finished_ns=finished_ns,
        published_ns=max(time.time_ns(), finished_ns),
        exit_code=exit_code,
        status=status,
        failure_reason=failure_reason,
        stdout_relative_path=stdout_path.name,
        stdout_raw_sha256=stdout_sha256,
        stdout_size_bytes=stdout_size,
        stderr_relative_path=stderr_path.name,
        stderr_raw_sha256=stderr_sha256,
        stderr_size_bytes=stderr_size,
        raw_profile_role=contract.raw_profile_role,
        raw_profile_relative_path=raw_relative_path,
        raw_profile_sha256=raw_sha256,
        raw_profile_size_bytes=raw_size,
        headline_eligible=False,
    )
    publish_formal_single_operator_json_artifact(
        run_root / "profiler-terminal.json",
        terminal.to_dict(),
    )
    return terminal


def run_formal_single_operator_profiler(
    *,
    profiler_plan_path: str | Path,
    current_ns: int | None = None,
) -> FormalSingleOperatorProfilerTerminal:
    """Execute only the subject argv regenerated from one deep-checked plan."""

    observed_ns = time.time_ns() if current_ns is None else current_ns
    plan = revalidate_formal_single_operator_profiler_plan(
        profiler_plan_path,
        current_ns=observed_ns,
    )
    return _run_formal_single_operator_profiler_capture(
        execution_source_path=plan.execution_source.absolute_path,
        materialized_cell_id=plan.materialized_cell_id,
        tool_path=plan.tool_identity.absolute_path,
        subject_argv=plan.subject_argv,
        output_directory=plan.output_directory,
    )


def load_formal_single_operator_profiler_terminal(
    path: str | Path,
) -> FormalSingleOperatorProfilerTerminal:
    """Deep-reopen one terminal and re-hash every onsite source/artifact."""

    binding = FormalSingleOperatorJsonBinding.bind(
        path,
        label="single-operator profiler terminal",
    )
    terminal = FormalSingleOperatorProfilerTerminal.from_dict(
        binding.reopen(label="single-operator profiler terminal")
    )
    source_binding, source, materialization, contract = _current_profiler_context(
        terminal.execution_source.absolute_path,
        materialized_cell_id=terminal.cell_id,
    )
    if (
        source_binding != terminal.execution_source
        or source.sha256 != terminal.execution_source_sha256
        or source.protocol_lock_sha256 != terminal.protocol_lock_sha256
        or materialization.sha256 != terminal.materialization_sha256
        or contract != terminal.tool_contract
    ):
        raise ValueError("profiler terminal current source changed")
    _revalidate_profiler_subject_argv(
        terminal.subject_argv,
        execution_source_path=terminal.execution_source.absolute_path,
        materialized_cell_id=terminal.cell_id,
        current_ns=time.time_ns(),
    )
    observed_tool = probe_formal_single_operator_profiler_tool(
        expected_tool=contract.tool,
        tool_path=terminal.tool_identity.absolute_path,
    )
    if observed_tool != terminal.tool_identity:
        raise ValueError("profiler terminal tool identity changed")
    run_root = Path(binding.absolute_path).parent
    expected_output_base = run_root / "profile"
    expected_command = _render_formal_single_operator_profiler_command(
        contract=contract,
        tool_path=terminal.tool_identity.absolute_path,
        output_base=expected_output_base,
        subject_argv=terminal.subject_argv,
    )
    if expected_command != terminal.rendered_argv:
        raise ValueError("profiler terminal rendered command changed")
    for label, relative, expected_sha256, expected_size in (
        (
            "profiler stdout",
            terminal.stdout_relative_path,
            terminal.stdout_raw_sha256,
            terminal.stdout_size_bytes,
        ),
        (
            "profiler stderr",
            terminal.stderr_relative_path,
            terminal.stderr_raw_sha256,
            terminal.stderr_size_bytes,
        ),
    ):
        observed = _read_stable_regular_file(
            run_root / relative,
            label=label,
            maximum_bytes=128 * 1024 * 1024,
            allow_empty=True,
        )
        if observed != (expected_sha256, expected_size):
            raise ValueError(f"{label} identity changed")
    if terminal.raw_profile_relative_path is not None:
        observed_raw = _read_stable_regular_file(
            run_root / terminal.raw_profile_relative_path,
            label="profiler raw output",
            maximum_bytes=_MAX_RAW_PROFILE_BYTES,
            allow_empty=False,
        )
        if observed_raw != (
            terminal.raw_profile_sha256,
            terminal.raw_profile_size_bytes,
        ):
            raise ValueError("profiler raw output identity changed")
    elif _raw_profile_path(expected_output_base, contract.tool).exists():
        raise ValueError("profiler raw output appeared after terminal publication")
    return terminal


__all__ = [
    "FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256",
    "FormalSingleOperatorProfilerBlocked",
    "FormalSingleOperatorProfilerCapturePlan",
    "FormalSingleOperatorProfilerSubjectRunPlanInputs",
    "FormalSingleOperatorProfilerTerminal",
    "FormalSingleOperatorProfilerToolIdentity",
    "load_formal_single_operator_profiler_terminal",
    "materialize_formal_single_operator_profiler_plan",
    "probe_formal_single_operator_profiler_tool",
    "revalidate_formal_single_operator_profiler_plan",
    "revalidate_formal_single_operator_profiler_subject_inputs",
    "run_formal_single_operator_profiler",
]
