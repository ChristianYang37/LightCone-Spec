"""Durable production bridge for the one physical exact-ten preflight run.

The scheduler represents preflight as ten logical attempts but launches only the
canonical leader.  This module binds those ten command identities to the
source-owned exact-ten executor, runs that executor once, publishes distinct
operator evidence for every logical cell, and finally publishes one shared
group receipt.  The shared receipt is publish-last and is the publication
identity returned for all ten ledger rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.orchestration.experiment_operator import (
    PhysicalAttemptGroupMemberSpec,
    ProcessObservation,
    QueuedCommandSpec,
    TerminalEvidence,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    OperatorTerminalContext,
)
from lightcone_spec.orchestration.formal_cell_worker import ChildHeartbeatPublisher
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_ENV = (
    "LIGHTCONE_PREFLIGHT_EXACT_TEN_GROUP_SPEC_PATH"
)

_SPEC_PROTOCOL_SOURCE = {
    "schema_version": 1,
    "kind": "formal_preflight_exact_ten_group_worker_spec_protocol",
    "coverage": "one_compile_one_exactness_eight_interference",
    "execution": "one_source_owned_exact_ten_parent",
    "gpu_binding": "exact_source_authority_dual_uuid_order",
    "publication": "ten_logical_results_then_one_shared_publish_last_receipt",
    "terminal_exit": "one_shared_wrapper_exit_receipt",
}
FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        _SPEC_PROTOCOL_SOURCE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

_PUBLICATION_PROTOCOL_SOURCE = {
    "schema_version": 1,
    "kind": "formal_preflight_exact_ten_group_publication_protocol",
    "source": FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_PROTOCOL_SHA256,
    "coverage": "exact_ten_distinct_terminal_junit_raw_pointer",
    "shared_completion": "deep_revalidated_exact_ten_completion",
    "atomicity": "group_publication_is_last",
}
FORMAL_PREFLIGHT_EXACT_TEN_GROUP_PUBLICATION_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        _PUBLICATION_PROTOCOL_SOURCE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

_RAW_PROTOCOL_SOURCE = {
    "schema_version": 1,
    "kind": "formal_preflight_exact_ten_logical_raw_protocol",
    "source": FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_PROTOCOL_SHA256,
    "projection": "one_completion_row_with_shared_execution_and_completion",
}
FORMAL_PREFLIGHT_EXACT_TEN_LOGICAL_RAW_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        _RAW_PROTOCOL_SOURCE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

_LOGICAL_TO_RUNNER = {
    "compile": "first_party_compile",
    "exactness": "first_party_exactness",
    "interference": "first_party_interference",
}
_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_sha256",
        "group_id",
        "leader_cell_id",
        "spec_path",
        "current_ns",
        "gpu_uuids",
        "parent_argv",
        "execution_inputs",
        "execution_output_path",
        "shared_publication_path",
        "members",
    }
)
_MEMBER_FIELDS = frozenset(
    {
        "cell_id",
        "attempt",
        "logical_kind",
        "command_sha256",
        "expected_terminal_path",
        "expected_junit_path",
        "expected_raw_log_path",
        "atomic_pointer_path",
        "child_exit_receipt_path",
    }
)
_PUBLICATION_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_sha256",
        "group_spec",
        "execution",
        "completion",
        "parent_exit_code",
        "status",
        "started_ns",
        "finished_ns",
        "members",
        "published_ns",
    }
)
_PUBLICATION_MEMBER_FIELDS = frozenset(
    {
        "cell_id",
        "attempt",
        "logical_kind",
        "command_sha256",
        "status",
        "result_sha256",
        "terminal_path",
        "terminal_sha256",
        "junit_path",
        "junit_sha256",
        "raw_log_path",
        "raw_log_sha256",
        "pointer_path",
        "pointer_sha256",
    }
)
_RAW_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_sha256",
        "group_spec",
        "group_id",
        "logical_kind",
        "execution",
        "completion",
        "parent_exit_code",
        "completion_row",
    }
)
_EXIT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "command_sha256",
        "wrapper_pid",
        "wrapper_pgid",
        "child_pid",
        "started_ns",
        "finished_ns",
        "exit_code",
        "launch_error_type",
        "receipt_sha256",
    }
)
_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "cell_id",
        "attempt",
        "command_sha256",
        "terminal_path",
        "terminal_sha256",
        "junit_path",
        "junit_sha256",
        "raw_log_path",
        "raw_log_sha256",
        "published_ns",
        "pointer_sha256",
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "cell_id",
        "attempt",
        "command_sha256",
        "status",
        "exit_code",
        "failure_class",
        "failure_code",
        "exclusion_reason",
        "included_in_analysis",
        "started_ns",
        "finished_ns",
    }
)


class FormalPreflightExactTenGroupError(RuntimeError):
    """Raised when a physical group cannot be proven exact and complete."""


@dataclass(frozen=True)
class FormalPreflightExactTenGroupMember:
    """One logical operator result written by the shared physical parent."""

    cell_id: str
    attempt: int
    logical_kind: Literal["compile", "exactness", "interference"]
    command_sha256: str
    expected_terminal_path: str
    expected_junit_path: str
    expected_raw_log_path: str
    atomic_pointer_path: str
    child_exit_receipt_path: str

    def __post_init__(self) -> None:
        _require_text(self.cell_id, "group member cell ID")
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("group member attempt must be positive")
        if self.logical_kind not in _LOGICAL_TO_RUNNER:
            raise ValueError("group member logical kind differs")
        _require_sha256(self.command_sha256, "group member command SHA-256")
        paths = self.evidence_paths
        if len(set(paths)) != len(paths):
            raise ValueError("group member evidence paths are not distinct")
        for value in paths:
            _canonical_absolute_path(value, "group member evidence")

    @property
    def evidence_paths(self) -> tuple[str, ...]:
        return (
            self.expected_terminal_path,
            self.expected_junit_path,
            self.expected_raw_log_path,
            self.atomic_pointer_path,
            self.child_exit_receipt_path,
        )

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_mapping(value, _MEMBER_FIELDS, "group member")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalPreflightExactTenGroupWorkerSpec:
    """Path-bound input for one restart-safe exact-ten physical worker."""

    schema_version: Literal[1]
    kind: Literal["formal_preflight_exact_ten_group_worker_spec"]
    protocol_sha256: str
    group_id: str
    leader_cell_id: str
    spec_path: str
    current_ns: int
    gpu_uuids: tuple[str, str]
    parent_argv: tuple[str, ...]
    execution_inputs: CanonicalJsonProofBinding
    execution_output_path: str
    shared_publication_path: str
    members: tuple[FormalPreflightExactTenGroupMember, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_preflight_exact_ten_group_worker_spec"
            or self.protocol_sha256
            != FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_PROTOCOL_SHA256
        ):
            raise ValueError("exact-ten group worker spec schema differs")
        _require_text(self.group_id, "exact-ten group ID")
        _require_text(self.leader_cell_id, "exact-ten group leader")
        _canonical_absolute_path(self.spec_path, "exact-ten group spec")
        if type(self.current_ns) is not int or self.current_ns < 1:
            raise ValueError("exact-ten group current time must be positive")
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
            or any(
                type(value) is not str or not value.startswith("GPU-")
                for value in self.gpu_uuids
            )
        ):
            raise ValueError("exact-ten group requires two source-bound GPU UUIDs")
        _parse_source_owned_parent_argv(
            self.parent_argv,
            expected_inputs_path=self.execution_inputs.absolute_path,
            expected_current_ns=self.current_ns,
        )
        _canonical_absolute_path(
            self.execution_output_path, "exact-ten execution output"
        )
        _canonical_absolute_path(
            self.shared_publication_path, "exact-ten shared publication"
        )
        expected_output = str(
            Path(self.execution_inputs.absolute_path).parent
            / "formal-single-operator-preflight-execution.json"
        )
        if self.execution_output_path != expected_output:
            raise ValueError("exact-ten execution output is not source-derived")
        if type(self.members) is not tuple or len(self.members) != 10:
            raise ValueError("exact-ten group worker spec requires ten members")
        identities = tuple((row.cell_id, row.attempt) for row in self.members)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("exact-ten group members are not uniquely sorted")
        kinds = tuple(row.logical_kind for row in self.members)
        if (
            kinds.count("compile") != 1
            or kinds.count("exactness") != 1
            or kinds.count("interference") != 8
        ):
            raise ValueError("exact-ten group member coverage is not 1+1+8")
        if self.leader_cell_id != self.members[0].cell_id:
            raise ValueError("exact-ten group leader is not canonical")
        if (
            len({row.attempt for row in self.members}) != 1
            or len({row.command_sha256 for row in self.members}) != 1
        ):
            raise ValueError("exact-ten group members do not share one command")
        evidence_paths = tuple(
            path for member in self.members for path in member.evidence_paths
        )
        if len(set(evidence_paths)) != len(evidence_paths):
            raise ValueError("exact-ten group evidence paths overlap")
        if self.shared_publication_path in set(evidence_paths) or (
            self.execution_output_path in set(evidence_paths)
        ):
            raise ValueError("exact-ten group shared paths overlap member evidence")

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "group_id": self.group_id,
            "leader_cell_id": self.leader_cell_id,
            "spec_path": self.spec_path,
            "current_ns": self.current_ns,
            "gpu_uuids": list(self.gpu_uuids),
            "parent_argv": list(self.parent_argv),
            "execution_inputs": self.execution_inputs.to_dict(),
            "execution_output_path": self.execution_output_path,
            "shared_publication_path": self.shared_publication_path,
            "members": [row.to_dict() for row in self.members],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_mapping(value, _SPEC_FIELDS, "exact-ten group worker spec")
        argv = row.pop("parent_argv")
        gpu_uuids = row.pop("gpu_uuids")
        members = row.pop("members")
        if type(argv) is not list or any(type(item) is not str for item in argv):
            raise TypeError("exact-ten parent argv is not an array of strings")
        if type(members) is not list:
            raise TypeError("exact-ten group members are not an array")
        if type(gpu_uuids) is not list or any(
            type(item) is not str for item in gpu_uuids
        ):
            raise TypeError("exact-ten group GPU UUIDs are not an array of strings")
        execution_inputs = CanonicalJsonProofBinding.from_dict(
            row.pop("execution_inputs")
        )
        return cls(
            **row,
            gpu_uuids=tuple(gpu_uuids),
            parent_argv=tuple(argv),
            execution_inputs=execution_inputs,
            members=tuple(
                FormalPreflightExactTenGroupMember.from_dict(item) for item in members
            ),
        )  # type: ignore[arg-type]


def formal_preflight_exact_ten_group_environment(
    spec_path: str | Path,
    *,
    base_environment: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[str, str], ...]:
    """Add the one scheduler-recognized group-spec binding canonically."""

    absolute = str(_canonical_absolute_path(spec_path, "exact-ten group spec"))
    if type(base_environment) is not tuple:
        raise TypeError("base group environment must be a tuple")
    values = dict(base_environment)
    if len(values) != len(base_environment) or any(
        type(name) is not str or type(value) is not str
        for name, value in base_environment
    ):
        raise ValueError("base group environment is not unique string pairs")
    existing = values.get(FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_ENV)
    if existing is not None and existing != absolute:
        raise ValueError("base environment binds another exact-ten group spec")
    values[FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_ENV] = absolute
    return tuple(sorted(values.items()))


def formal_preflight_exact_ten_group_spec_path(
    command: QueuedCommandSpec,
) -> str | None:
    """Return the exact group-spec path tagged on a queued command, if any."""

    if type(command) is not QueuedCommandSpec:
        raise TypeError("group-spec lookup requires an exact queued command")
    values = dict(command.environment)
    value = values.get(FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_ENV)
    if value is None:
        return None
    return str(_canonical_absolute_path(value, "exact-ten group spec"))


def publish_formal_preflight_exact_ten_group_worker_spec(
    *,
    group_id: str,
    members: tuple[PhysicalAttemptGroupMemberSpec, ...],
    leader_cell_id: str,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish a no-replace spec derived from one exact controller group."""

    destination = _canonical_absolute_path(output_path, "exact-ten group spec")
    if (
        type(members) is not tuple
        or len(members) != 10
        or any(type(row) is not PhysicalAttemptGroupMemberSpec for row in members)
    ):
        raise TypeError("exact-ten group spec requires exact operator members")
    commands = tuple(row.command for row in members)
    if any(
        command.required_gpu_count != 2 or command.timing_class != "EXCLUSIVE"
        for command in commands
    ):
        raise ValueError("exact-ten physical parent must be exclusive dual-GPU")
    tagged_paths = tuple(
        formal_preflight_exact_ten_group_spec_path(row) for row in commands
    )
    if set(tagged_paths) != {str(destination)}:
        raise ValueError("exact-ten commands do not bind this group spec path")
    parent_argv = commands[0].argv
    execution_inputs_path, current_ns = _parse_source_owned_parent_argv(parent_argv)
    if any(command.argv != parent_argv for command in commands):
        raise ValueError("exact-ten commands do not share one parent argv")
    source_kinds, gpu_uuids = _revalidate_source_projection(
        execution_inputs_path,
        current_ns=current_ns,
    )
    logical_members = tuple(
        FormalPreflightExactTenGroupMember(
            cell_id=row.attempt.cell_id,
            attempt=row.attempt.attempt,
            logical_kind=row.logical_kind,
            command_sha256=row.command.command_sha256,
            expected_terminal_path=row.command.expected_terminal_path,
            expected_junit_path=row.command.expected_junit_path,
            expected_raw_log_path=row.command.expected_raw_log_path,
            atomic_pointer_path=row.command.atomic_pointer_path,
            child_exit_receipt_path=row.command.child_exit_receipt_path,
        )
        for row in members
    )
    expected_source_kinds = {
        row.cell_id: _LOGICAL_TO_RUNNER[row.logical_kind] for row in logical_members
    }
    if source_kinds != expected_source_kinds:
        raise ValueError("exact-ten operator members differ from source authority")
    execution_inputs = CanonicalJsonProofBinding.bind(execution_inputs_path)
    artifact = FormalPreflightExactTenGroupWorkerSpec(
        schema_version=1,
        kind="formal_preflight_exact_ten_group_worker_spec",
        protocol_sha256=FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_PROTOCOL_SHA256,
        group_id=group_id,
        leader_cell_id=leader_cell_id,
        spec_path=str(destination),
        current_ns=current_ns,
        gpu_uuids=gpu_uuids,
        parent_argv=parent_argv,
        execution_inputs=execution_inputs,
        execution_output_path=str(
            Path(execution_inputs.absolute_path).parent
            / "formal-single-operator-preflight-execution.json"
        ),
        shared_publication_path=str(
            destination.parent / "formal-preflight-exact-ten-group-publication.json"
        ),
        members=logical_members,
    )
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    publish_canonical_json_no_replace(destination, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(
        destination,
        semantic_sha256=artifact.sha256,
    )
    if revalidate_formal_preflight_exact_ten_group_worker_spec(destination) != artifact:
        raise RuntimeError("exact-ten group worker spec changed on publication")
    return binding


def revalidate_formal_preflight_exact_ten_group_worker_spec(
    path: str | Path,
    *,
    expected_command: QueuedCommandSpec | None = None,
) -> FormalPreflightExactTenGroupWorkerSpec:
    """Deep-reopen a group spec and its exact current-source member universe."""

    binding = CanonicalJsonProofBinding.bind(path)
    artifact = FormalPreflightExactTenGroupWorkerSpec.from_dict(binding.reopen())
    if binding.semantic_sha256 != artifact.sha256:
        raise ValueError("exact-ten group worker spec identity differs")
    if binding.absolute_path != artifact.spec_path:
        raise ValueError("exact-ten group worker spec path differs")
    source_kinds, gpu_uuids = _revalidate_source_projection(
        artifact.execution_inputs.absolute_path,
        current_ns=artifact.current_ns,
    )
    expected_source_kinds = {
        row.cell_id: _LOGICAL_TO_RUNNER[row.logical_kind] for row in artifact.members
    }
    if source_kinds != expected_source_kinds:
        raise ValueError("exact-ten group spec source member universe changed")
    if gpu_uuids != artifact.gpu_uuids:
        raise ValueError("exact-ten group source GPU assignment changed")
    rebound_inputs = CanonicalJsonProofBinding.bind(
        artifact.execution_inputs.absolute_path
    )
    if rebound_inputs != artifact.execution_inputs:
        raise ValueError("exact-ten group execution inputs changed")
    if expected_command is not None:
        _require_command_member(artifact, expected_command)
    return artifact


def ensure_formal_preflight_exact_ten_group_outputs_unoccupied(
    spec: FormalPreflightExactTenGroupWorkerSpec,
) -> None:
    """Fail before allocation if any immutable group output already exists."""

    paths = [spec.execution_output_path, spec.shared_publication_path]
    paths.extend(path for row in spec.members for path in row.evidence_paths)
    occupied = [
        path for path in paths if Path(path).exists() or Path(path).is_symlink()
    ]
    if occupied:
        raise FormalPreflightExactTenGroupError(
            f"exact-ten group output path is already occupied: {occupied[0]}"
        )


def run_formal_preflight_exact_ten_group_worker(
    spec_path: str | Path,
    *,
    parent_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    execution_revalidator: Callable[..., object] | None = None,
    completion_revalidator: Callable[..., object] | None = None,
    assigned_gpu_uuids: tuple[str, ...] | None = None,
    heartbeat_interval_seconds: float = 30.0,
) -> int:
    """Run the bound parent once and atomically fan its exact completion."""

    spec_binding = CanonicalJsonProofBinding.bind(spec_path)
    spec = revalidate_formal_preflight_exact_ten_group_worker_spec(
        spec_binding.absolute_path
    )
    assigned = assigned_gpu_uuids
    if assigned is None:
        raw_assigned = os.environ.get("LIGHTCONE_ASSIGNED_GPU_UUIDS", "")
        assigned = tuple(raw_assigned.split(",")) if raw_assigned else ()
    if assigned != spec.gpu_uuids:
        raise FormalPreflightExactTenGroupError(
            "physical worker GPU UUIDs differ from source authority"
        )
    ensure_formal_preflight_exact_ten_group_outputs_unoccupied(spec)
    heartbeat: ChildHeartbeatPublisher | None = None
    heartbeat_path = os.environ.get("LIGHTCONE_OPERATOR_HEARTBEAT_PATH")
    if heartbeat_path:
        context = OperatorTerminalContext.from_environment()
        leader = spec.members[0]
        if (
            context.cell_id != leader.cell_id
            or context.attempt != leader.attempt
            or context.command_sha256 != leader.command_sha256
        ):
            raise FormalPreflightExactTenGroupError(
                "exact-ten heartbeat identity differs from the group leader"
            )
        heartbeat = ChildHeartbeatPublisher(
            path=_canonical_absolute_path(
                heartbeat_path,
                "exact-ten operator heartbeat",
            ),
            context=context,
            clock_ns=time.time_ns,
            interval_seconds=heartbeat_interval_seconds,
        )
        heartbeat.start()
    try:
        completed = parent_runner(
            list(spec.parent_argv),
            stdin=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
        if type(completed.returncode) is not int or completed.returncode not in {0, 42}:
            raise FormalPreflightExactTenGroupError(
                "source-owned exact-ten parent exited outside its registered contract"
            )
        execution_loader = execution_revalidator or _revalidate_execution
        completion_loader = completion_revalidator or _revalidate_completion
        execution = execution_loader(
            spec.execution_output_path,
            current_ns=spec.current_ns,
        )
        execution_binding = CanonicalJsonProofBinding.bind(spec.execution_output_path)
        status = getattr(execution, "status", None)
        expected_parent_exit = 0 if status == "COMPLETE" else 42
        if (
            status not in {"COMPLETE", "FAILED"}
            or completed.returncode != expected_parent_exit
        ):
            raise FormalPreflightExactTenGroupError(
                "exact-ten parent exit and deep-revalidated execution disagree"
            )
        completion_source = getattr(execution, "completion", None)
        if type(completion_source) is not CanonicalJsonProofBinding:
            raise FormalPreflightExactTenGroupError(
                "exact-ten execution lacks one path-bound completion"
            )
        completion = completion_loader(
            completion_source.absolute_path,
            current_ns=spec.current_ns,
        )
        completion_binding = CanonicalJsonProofBinding.bind(
            completion_source.absolute_path
        )
        if completion_binding != completion_source:
            raise FormalPreflightExactTenGroupError(
                "exact-ten completion binding changed"
            )
        if heartbeat is not None:
            # The immutable proof readers verify parent-directory stability.
            # Stop the mutable control file immediately before publish/rehydrate.
            heartbeat.stop()
            heartbeat = None
        _publish_group_results(
            spec_binding=spec_binding,
            spec=spec,
            execution_binding=execution_binding,
            execution=execution,
            completion_binding=completion_binding,
            completion=completion,
            parent_exit_code=completed.returncode,
        )
    finally:
        if heartbeat is not None:
            heartbeat.stop()
    return 0


def revalidate_formal_preflight_exact_ten_group_terminal(
    command: QueuedCommandSpec,
    attempt: Mapping[str, Any],
    observation: ProcessObservation,
) -> TerminalEvidence | None:
    """Validate one member while reopening the entire shared publication."""

    spec_path = formal_preflight_exact_ten_group_spec_path(command)
    if spec_path is None:
        raise FormalPreflightExactTenGroupError("command has no exact-ten group spec")
    spec_binding = CanonicalJsonProofBinding.bind(spec_path)
    spec = revalidate_formal_preflight_exact_ten_group_worker_spec(
        spec_path,
        expected_command=command,
    )
    leader = spec.members[0]
    receipt_path = Path(leader.child_exit_receipt_path)
    if not receipt_path.exists():
        return None
    if not Path(spec.shared_publication_path).exists():
        raise FormalPreflightExactTenGroupError(
            "shared group exit exists without publish-last group publication"
        )
    receipt_binding = CanonicalJsonProofBinding.bind(receipt_path)
    receipt = receipt_binding.reopen()
    _validate_shared_exit_receipt(
        receipt,
        expected_command_sha256=leader.command_sha256,
        expected_pid=attempt.get("pid"),
        expected_pgid=attempt.get("pgid"),
    )
    if observation.exit_code is not None and observation.exit_code != 0:
        raise FormalPreflightExactTenGroupError(
            "group process observation differs from successful bridge receipt"
        )
    publication_binding, publication, member_rows = _revalidate_group_publication(
        spec_binding=spec_binding,
        spec=spec,
    )
    if (
        receipt["exit_code"] != 0
        or publication["started_ns"] < receipt["started_ns"]
        or publication["finished_ns"] > receipt["finished_ns"]
        or publication["published_ns"] < receipt["started_ns"]
        or publication["published_ns"] > receipt["finished_ns"]
    ):
        raise FormalPreflightExactTenGroupError(
            "shared group publication lies outside its wrapper lifecycle"
        )
    member = next(row for row in spec.members if row.cell_id == command.cell_id)
    published = member_rows[member.cell_id]
    failed = published["status"] == "FAILED"
    evidence_files = {
        spec_binding.absolute_path: spec_binding.raw_sha256,
        publication_binding.absolute_path: publication_binding.raw_sha256,
        receipt_binding.absolute_path: receipt_binding.raw_sha256,
        publication["execution"]["absolute_path"]: publication["execution"][
            "raw_sha256"
        ],
        publication["completion"]["absolute_path"]: publication["completion"][
            "raw_sha256"
        ],
        member.atomic_pointer_path: published["pointer_sha256"],
    }
    return TerminalEvidence(
        status=published["status"],
        exit_code=0,
        atomic_publication_sha256=publication_binding.raw_sha256,
        terminal_sha256=published["terminal_sha256"],
        junit_sha256=published["junit_sha256"],
        raw_log_sha256=published["raw_log_sha256"],
        evidence_files=evidence_files,
        failure_class="SCIENTIFIC" if failed else None,
        failure_code="PREFLIGHT_LOGICAL_CELL_FAILED" if failed else None,
        exclusion_reason=(
            "source-owned exact-ten completion row failed" if failed else None
        ),
        included_in_analysis=not failed,
        started_ns=receipt["started_ns"],
        finished_ns=receipt["finished_ns"],
    )


def _publish_group_results(
    *,
    spec_binding: CanonicalJsonProofBinding,
    spec: FormalPreflightExactTenGroupWorkerSpec,
    execution_binding: CanonicalJsonProofBinding,
    execution: object,
    completion_binding: CanonicalJsonProofBinding,
    completion: object,
    parent_exit_code: int,
) -> CanonicalJsonProofBinding:
    rows = _validated_completion_rows(spec, execution, completion)
    publication_members: list[dict[str, object]] = []
    for member in spec.members:
        row = rows[member.cell_id]
        status = row.status
        raw = _logical_raw_value(
            spec_binding=spec_binding,
            spec=spec,
            member=member,
            execution_binding=execution_binding,
            completion_binding=completion_binding,
            parent_exit_code=parent_exit_code,
            completion_row=row,
        )
        _atomic_write_new_bytes(
            Path(member.expected_raw_log_path),
            _canonical_json_bytes(raw),
        )
        _atomic_write_new_bytes(
            Path(member.expected_junit_path),
            _logical_junit_bytes(member, status=status),
        )
        from lightcone_spec.orchestration.experiment_operator_production import (
            OperatorTerminalContext,
            publish_atomic_terminal_result,
        )

        failed = status == "FAILED"
        publish_atomic_terminal_result(
            OperatorTerminalContext(
                cell_id=member.cell_id,
                attempt=member.attempt,
                command_sha256=member.command_sha256,
                expected_terminal_path=member.expected_terminal_path,
                expected_junit_path=member.expected_junit_path,
                expected_raw_log_path=member.expected_raw_log_path,
                atomic_pointer_path=member.atomic_pointer_path,
            ),
            status=status,
            exit_code=0,
            started_ns=row.started_ns,
            finished_ns=row.finished_ns,
            failure_class="SCIENTIFIC" if failed else None,
            failure_code="PREFLIGHT_LOGICAL_CELL_FAILED" if failed else None,
            exclusion_reason=(
                "source-owned exact-ten completion row failed" if failed else None
            ),
            included_in_analysis=not failed,
        )
        publication_members.append(
            _publication_member_value(member=member, completion_row=row)
        )
    publication = {
        "schema_version": 1,
        "kind": "formal_preflight_exact_ten_group_publication",
        "protocol_sha256": (
            FORMAL_PREFLIGHT_EXACT_TEN_GROUP_PUBLICATION_PROTOCOL_SHA256
        ),
        "group_spec": spec_binding.to_dict(),
        "execution": execution_binding.to_dict(),
        "completion": completion_binding.to_dict(),
        "parent_exit_code": parent_exit_code,
        "status": completion.status,
        "started_ns": completion.started_ns,
        "finished_ns": completion.finished_ns,
        "members": publication_members,
        "published_ns": time.time_ns(),
    }
    publish_canonical_json_no_replace(spec.shared_publication_path, publication)
    binding, _value, _rows = _revalidate_group_publication(
        spec_binding=spec_binding,
        spec=spec,
    )
    return binding


def _revalidate_group_publication(
    *,
    spec_binding: CanonicalJsonProofBinding,
    spec: FormalPreflightExactTenGroupWorkerSpec,
) -> tuple[CanonicalJsonProofBinding, dict[str, Any], dict[str, dict[str, Any]]]:
    binding = CanonicalJsonProofBinding.bind(spec.shared_publication_path)
    value = binding.reopen()
    if frozenset(value) != _PUBLICATION_FIELDS:
        raise FormalPreflightExactTenGroupError(
            "exact-ten group publication fields differ"
        )
    if (
        value["schema_version"] != 1
        or value["kind"] != "formal_preflight_exact_ten_group_publication"
        or value["protocol_sha256"]
        != FORMAL_PREFLIGHT_EXACT_TEN_GROUP_PUBLICATION_PROTOCOL_SHA256
        or CanonicalJsonProofBinding.from_dict(value["group_spec"]) != spec_binding
        or type(value["parent_exit_code"]) is not int
        or value["parent_exit_code"] not in {0, 42}
        or value["status"] not in {"COMPLETE", "FAILED"}
        or type(value["started_ns"]) is not int
        or type(value["finished_ns"]) is not int
        or value["started_ns"] < 1
        or value["finished_ns"] < value["started_ns"]
        or type(value["published_ns"]) is not int
        or value["published_ns"] < 1
    ):
        raise FormalPreflightExactTenGroupError(
            "exact-ten group publication identity differs"
        )
    execution_binding = CanonicalJsonProofBinding.from_dict(value["execution"])
    completion_binding = CanonicalJsonProofBinding.from_dict(value["completion"])
    execution = _revalidate_execution(
        execution_binding.absolute_path,
        current_ns=spec.current_ns,
    )
    completion = _revalidate_completion(
        completion_binding.absolute_path,
        current_ns=spec.current_ns,
    )
    if (
        CanonicalJsonProofBinding.bind(execution_binding.absolute_path)
        != execution_binding
        or CanonicalJsonProofBinding.bind(completion_binding.absolute_path)
        != completion_binding
        or getattr(execution, "completion", None) != completion_binding
        or getattr(execution, "status", None) != value["status"]
        or getattr(completion, "status", None) != value["status"]
        or getattr(completion, "started_ns", None) != value["started_ns"]
        or getattr(completion, "finished_ns", None) != value["finished_ns"]
        or value["parent_exit_code"] != (0 if value["status"] == "COMPLETE" else 42)
    ):
        raise FormalPreflightExactTenGroupError(
            "exact-ten group publication completion lineage differs"
        )
    completion_rows = _validated_completion_rows(spec, execution, completion)
    raw_members = value["members"]
    if type(raw_members) is not list or len(raw_members) != 10:
        raise FormalPreflightExactTenGroupError(
            "exact-ten group publication member coverage differs"
        )
    published_rows: dict[str, dict[str, Any]] = {}
    for member, raw_member in zip(spec.members, raw_members, strict=True):
        published = _strict_mapping(
            raw_member,
            _PUBLICATION_MEMBER_FIELDS,
            "exact-ten publication member",
        )
        expected = _publication_member_value(
            member=member,
            completion_row=completion_rows[member.cell_id],
        )
        if published != expected:
            raise FormalPreflightExactTenGroupError(
                "exact-ten publication member evidence changed"
            )
        _revalidate_member_files(
            spec_binding=spec_binding,
            spec=spec,
            member=member,
            completion_row=completion_rows[member.cell_id],
            execution_binding=execution_binding,
            completion_binding=completion_binding,
            parent_exit_code=value["parent_exit_code"],
        )
        published_rows[member.cell_id] = published
    return binding, value, published_rows


def _validated_completion_rows(
    spec: FormalPreflightExactTenGroupWorkerSpec,
    execution: object,
    completion: object,
) -> dict[str, object]:
    raw_rows = getattr(completion, "rows", None)
    if type(raw_rows) is not tuple or len(raw_rows) != 10:
        raise FormalPreflightExactTenGroupError(
            "deep-revalidated completion is not exact ten"
        )
    rows = {getattr(row, "materialized_cell_id", None): row for row in raw_rows}
    if set(rows) != {member.cell_id for member in spec.members}:
        raise FormalPreflightExactTenGroupError(
            "deep-revalidated completion cell universe differs"
        )
    statuses: list[str] = []
    for member in spec.members:
        row = rows[member.cell_id]
        status = getattr(row, "status", None)
        if (
            getattr(row, "runner_kind", None) != _LOGICAL_TO_RUNNER[member.logical_kind]
            or status not in {"COMPLETE", "FAILED"}
            or type(getattr(row, "result_sha256", None)) is not str
            or type(getattr(row, "started_ns", None)) is not int
            or type(getattr(row, "finished_ns", None)) is not int
        ):
            raise FormalPreflightExactTenGroupError(
                "deep-revalidated completion row identity differs"
            )
        _require_sha256(row.result_sha256, "completion row result")
        if row.started_ns < 1 or row.finished_ns < row.started_ns:
            raise FormalPreflightExactTenGroupError(
                "deep-revalidated completion row timing differs"
            )
        statuses.append(status)
    aggregate = (
        "COMPLETE" if all(status == "COMPLETE" for status in statuses) else "FAILED"
    )
    if (
        getattr(completion, "status", None) != aggregate
        or getattr(execution, "status", None) != aggregate
        or getattr(completion, "started_ns", None)
        != min(row.started_ns for row in raw_rows)
        or getattr(completion, "finished_ns", None)
        != max(row.finished_ns for row in raw_rows)
    ):
        raise FormalPreflightExactTenGroupError(
            "exact-ten completion aggregate outcome differs"
        )
    return rows


def _logical_raw_value(
    *,
    spec_binding: CanonicalJsonProofBinding,
    spec: FormalPreflightExactTenGroupWorkerSpec,
    member: FormalPreflightExactTenGroupMember,
    execution_binding: CanonicalJsonProofBinding,
    completion_binding: CanonicalJsonProofBinding,
    parent_exit_code: int,
    completion_row: object,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "formal_preflight_exact_ten_logical_raw_evidence",
        "protocol_sha256": FORMAL_PREFLIGHT_EXACT_TEN_LOGICAL_RAW_PROTOCOL_SHA256,
        "group_spec": spec_binding.to_dict(),
        "group_id": spec.group_id,
        "logical_kind": member.logical_kind,
        "execution": execution_binding.to_dict(),
        "completion": completion_binding.to_dict(),
        "parent_exit_code": parent_exit_code,
        "completion_row": completion_row.to_dict(),
    }


def _publication_member_value(
    *,
    member: FormalPreflightExactTenGroupMember,
    completion_row: object,
) -> dict[str, object]:
    return {
        "cell_id": member.cell_id,
        "attempt": member.attempt,
        "logical_kind": member.logical_kind,
        "command_sha256": member.command_sha256,
        "status": completion_row.status,
        "result_sha256": completion_row.result_sha256,
        "terminal_path": member.expected_terminal_path,
        "terminal_sha256": _file_sha256(member.expected_terminal_path),
        "junit_path": member.expected_junit_path,
        "junit_sha256": _file_sha256(member.expected_junit_path),
        "raw_log_path": member.expected_raw_log_path,
        "raw_log_sha256": _file_sha256(member.expected_raw_log_path),
        "pointer_path": member.atomic_pointer_path,
        "pointer_sha256": _file_sha256(member.atomic_pointer_path),
    }


def _revalidate_member_files(
    *,
    spec_binding: CanonicalJsonProofBinding,
    spec: FormalPreflightExactTenGroupWorkerSpec,
    member: FormalPreflightExactTenGroupMember,
    completion_row: object,
    execution_binding: CanonicalJsonProofBinding,
    completion_binding: CanonicalJsonProofBinding,
    parent_exit_code: int,
) -> None:
    raw_binding = CanonicalJsonProofBinding.bind(member.expected_raw_log_path)
    expected_raw = _logical_raw_value(
        spec_binding=spec_binding,
        spec=spec,
        member=member,
        execution_binding=execution_binding,
        completion_binding=completion_binding,
        parent_exit_code=parent_exit_code,
        completion_row=completion_row,
    )
    raw = raw_binding.reopen()
    if frozenset(raw) != _RAW_FIELDS or raw != expected_raw:
        raise FormalPreflightExactTenGroupError(
            "exact-ten logical raw evidence changed"
        )
    expected_junit = _logical_junit_bytes(
        member,
        status=completion_row.status,
    )
    if _read_regular_bytes(member.expected_junit_path) != expected_junit:
        raise FormalPreflightExactTenGroupError("exact-ten logical JUnit changed")
    terminal_binding = CanonicalJsonProofBinding.bind(member.expected_terminal_path)
    terminal = terminal_binding.reopen()
    failed = completion_row.status == "FAILED"
    if frozenset(terminal) != _TERMINAL_FIELDS or terminal != {
        "schema_version": 1,
        "kind": "formal_experiment_terminal",
        "cell_id": member.cell_id,
        "attempt": member.attempt,
        "command_sha256": member.command_sha256,
        "status": completion_row.status,
        "exit_code": 0,
        "failure_class": "SCIENTIFIC" if failed else None,
        "failure_code": "PREFLIGHT_LOGICAL_CELL_FAILED" if failed else None,
        "exclusion_reason": (
            "source-owned exact-ten completion row failed" if failed else None
        ),
        "included_in_analysis": not failed,
        "started_ns": completion_row.started_ns,
        "finished_ns": completion_row.finished_ns,
    }:
        raise FormalPreflightExactTenGroupError("exact-ten logical terminal changed")
    pointer_binding = CanonicalJsonProofBinding.bind(member.atomic_pointer_path)
    pointer = pointer_binding.reopen()
    if frozenset(pointer) != _POINTER_FIELDS:
        raise FormalPreflightExactTenGroupError("exact-ten pointer fields differ")
    without_digest = dict(pointer)
    pointer_digest = without_digest.pop("pointer_sha256")
    if (
        pointer["schema_version"] != 1
        or pointer["kind"] != "formal_experiment_atomic_result_pointer"
        or pointer["cell_id"] != member.cell_id
        or pointer["attempt"] != member.attempt
        or pointer["command_sha256"] != member.command_sha256
        or pointer["terminal_path"] != member.expected_terminal_path
        or pointer["terminal_sha256"] != terminal_binding.raw_sha256
        or pointer["junit_path"] != member.expected_junit_path
        or pointer["junit_sha256"] != _file_sha256(member.expected_junit_path)
        or pointer["raw_log_path"] != member.expected_raw_log_path
        or pointer["raw_log_sha256"] != raw_binding.raw_sha256
        or type(pointer["published_ns"]) is not int
        or pointer["published_ns"] < 1
        or pointer_digest != _wire_sha256(without_digest)
    ):
        raise FormalPreflightExactTenGroupError("exact-ten logical pointer changed")


def _validate_shared_exit_receipt(
    receipt: Mapping[str, object],
    *,
    expected_command_sha256: str,
    expected_pid: object,
    expected_pgid: object,
) -> None:
    if frozenset(receipt) != _EXIT_RECEIPT_FIELDS:
        raise FormalPreflightExactTenGroupError("shared exit receipt fields differ")
    without_digest = dict(receipt)
    digest = without_digest.pop("receipt_sha256")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "formal_experiment_child_exit_receipt"
        or receipt["command_sha256"] != expected_command_sha256
        or receipt["wrapper_pid"] != expected_pid
        or receipt["wrapper_pgid"] != expected_pgid
        or type(receipt["started_ns"]) is not int
        or type(receipt["finished_ns"]) is not int
        or receipt["started_ns"] < 1  # type: ignore[operator]
        or receipt["finished_ns"] <= receipt["started_ns"]  # type: ignore[operator]
        or receipt["exit_code"] != 0
        or receipt["launch_error_type"] is not None
        or digest != _wire_sha256(without_digest)
    ):
        raise FormalPreflightExactTenGroupError("shared exit receipt identity differs")


def _require_command_member(
    spec: FormalPreflightExactTenGroupWorkerSpec,
    command: QueuedCommandSpec,
) -> FormalPreflightExactTenGroupMember:
    candidates = [row for row in spec.members if row.cell_id == command.cell_id]
    if len(candidates) != 1:
        raise ValueError("queued command is outside the exact-ten group")
    member = candidates[0]
    if (
        command.attempt != member.attempt
        or command.command_sha256 != member.command_sha256
        or command.argv != spec.parent_argv
        or command.required_gpu_count != 2
        or command.timing_class != "EXCLUSIVE"
        or command.expected_terminal_path != member.expected_terminal_path
        or command.expected_junit_path != member.expected_junit_path
        or command.expected_raw_log_path != member.expected_raw_log_path
        or command.atomic_pointer_path != member.atomic_pointer_path
        or command.child_exit_receipt_path != member.child_exit_receipt_path
        or formal_preflight_exact_ten_group_spec_path(command) != spec.spec_path
    ):
        raise ValueError("queued command differs from exact-ten group member")
    return member


def _parse_source_owned_parent_argv(
    argv: Sequence[str],
    *,
    expected_inputs_path: str | None = None,
    expected_current_ns: int | None = None,
) -> tuple[str, int]:
    if not argv or any(type(value) is not str or not value for value in argv):
        raise ValueError("source-owned exact-ten parent argv is invalid")
    values = tuple(argv)
    if Path(values[0]).name == "lightcone-spec":
        prefix_length = 1
    elif (
        len(values) >= 3
        and values[1] == "-m"
        and values[2] == "lightcone_spec.cli.main"
    ):
        prefix_length = 3
    else:
        raise ValueError("exact-ten parent is not the source-owned CLI")
    tail = values[prefix_length:]
    if (
        len(tail) != 6
        or tail[:2] != ("formal-single-operator", "execute-preflight")
        or tail[2] != "--execution-inputs"
        or tail[4] != "--current-ns"
    ):
        raise ValueError("exact-ten parent argv has runtime knobs or missing identity")
    inputs_path = str(_canonical_absolute_path(tail[3], "exact-ten execution inputs"))
    try:
        current_ns = int(tail[5])
    except ValueError as error:
        raise ValueError("exact-ten parent current time is invalid") from error
    if current_ns < 1 or str(current_ns) != tail[5]:
        raise ValueError("exact-ten parent current time is not canonical")
    if expected_inputs_path is not None and inputs_path != expected_inputs_path:
        raise ValueError("exact-ten parent execution inputs differ")
    if expected_current_ns is not None and current_ns != expected_current_ns:
        raise ValueError("exact-ten parent current time differs")
    return inputs_path, current_ns


def _revalidate_source_projection(
    execution_inputs_path: str | Path,
    *,
    current_ns: int,
) -> tuple[dict[str, str], tuple[str, str]]:
    from lightcone_spec.experiments.formal_preflight_inputs import (
        FormalSingleOperatorPreflightAuthority,
        revalidate_formal_single_operator_preflight_execution_inputs,
    )

    inputs = revalidate_formal_single_operator_preflight_execution_inputs(
        execution_inputs_path,
        current_ns=current_ns,
    )
    authority = FormalSingleOperatorPreflightAuthority.from_dict(
        inputs.execution_authority.reopen()
    )
    kinds = {
        row.materialized_cell_id: row.runner_kind
        for row in authority.execution_bindings
    }
    dual_rows = tuple(
        row.gpu_uuids
        for row in authority.execution_bindings
        if row.runner_kind in {"first_party_compile", "first_party_exactness"}
    )
    if len(dual_rows) != 2 or len(set(dual_rows)) != 1 or len(dual_rows[0]) != 2:
        raise ValueError("exact-ten source lacks one dual-GPU parent assignment")
    expected_gpus = dual_rows[0]
    if any(
        not set(row.gpu_uuids).issubset(set(expected_gpus))
        for row in authority.execution_bindings
    ):
        raise ValueError("exact-ten source member escapes the parent GPU assignment")
    return kinds, expected_gpus


def _revalidate_execution(path: str | Path, *, current_ns: int) -> object:
    from lightcone_spec.experiments.formal_preflight_inputs import (
        revalidate_formal_single_operator_preflight_exact_ten_execution,
    )

    return revalidate_formal_single_operator_preflight_exact_ten_execution(
        path,
        current_ns=current_ns,
    )


def _revalidate_completion(path: str | Path, *, current_ns: int) -> object:
    from lightcone_spec.experiments.formal_preflight_inputs import (
        revalidate_formal_single_operator_preflight_completion,
    )

    return revalidate_formal_single_operator_preflight_completion(
        path,
        current_ns=current_ns,
    )


def _logical_junit_bytes(
    member: FormalPreflightExactTenGroupMember,
    *,
    status: str,
) -> bytes:
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError("logical JUnit status differs")
    suite = ET.Element(
        "testsuite",
        {
            "name": "lightcone.preflight.exact_ten.physical_group",
            "tests": "1",
            "failures": "0" if status == "COMPLETE" else "1",
            "errors": "0",
            "skipped": "0",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": f"lightcone.preflight.{member.logical_kind}",
            "name": member.cell_id,
        },
    )
    if status == "FAILED":
        failure = ET.SubElement(
            case,
            "failure",
            {"type": "scientific", "message": "exact-ten completion row failed"},
        )
        failure.text = "The source-owned exact-ten completion marks this row FAILED."
    return ET.tostring(suite, encoding="utf-8", xml_declaration=True) + b"\n"


def _atomic_write_new_bytes(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError("exact-ten evidence path is not canonical")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite exact-ten evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_regular_bytes(path: str | Path) -> bytes:
    source = Path(path)
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode) or source.is_symlink():
        raise FormalPreflightExactTenGroupError(
            "exact-ten evidence is not a regular file"
        )
    return source.read_bytes()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)[:-1]).hexdigest()


def _wire_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not lowercase SHA-256")
    return value


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _canonical_absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} path must be absolute and normalized")
    return path


def _strict_mapping(
    value: object,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--group-spec", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return run_formal_preflight_exact_ten_group_worker(arguments.group_spec)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_PREFLIGHT_EXACT_TEN_GROUP_PUBLICATION_PROTOCOL_SHA256",
    "FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_ENV",
    "FORMAL_PREFLIGHT_EXACT_TEN_GROUP_SPEC_PROTOCOL_SHA256",
    "FORMAL_PREFLIGHT_EXACT_TEN_LOGICAL_RAW_PROTOCOL_SHA256",
    "FormalPreflightExactTenGroupError",
    "FormalPreflightExactTenGroupMember",
    "FormalPreflightExactTenGroupWorkerSpec",
    "ensure_formal_preflight_exact_ten_group_outputs_unoccupied",
    "formal_preflight_exact_ten_group_environment",
    "formal_preflight_exact_ten_group_spec_path",
    "publish_formal_preflight_exact_ten_group_worker_spec",
    "revalidate_formal_preflight_exact_ten_group_terminal",
    "revalidate_formal_preflight_exact_ten_group_worker_spec",
    "run_formal_preflight_exact_ten_group_worker",
]
