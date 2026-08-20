"""Current-only stage transitions for ``formal_single_operator_v1``.

This module is intentionally smaller than the adversarial formal evidence
stack.  It serves one trusted operator in one controlled local checkout.  A
node can be materialized only from the immediately preceding completed node,
and it can be reduced only from the exact current materialization plus one
source-validated actual result per materialized cell.

There is deliberately no argument for a future-stage registry, an E0
aggregate, a replayed source summary, or a placeholder completion.  Failed or
missing actual results stop the transition without publishing a completion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import statistics
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Literal, Protocol, Self

from lightcone_spec.experiments.formal_content_source import FormalContentSourceBinding
from lightcone_spec.experiments.formal_protocol import ProtocolLock
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_from_dict,
    stage_materialization_receipt_from_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
    materialize_preflight,
)

FORMAL_SINGLE_OPERATOR_STAGE_MODE = "formal_single_operator_v1"
FORMAL_SINGLE_OPERATOR_STAGE_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024

type FormalSingleOperatorNode = Literal[
    "preflight",
    "e3a",
    "tts_cal",
    "e1",
    "e2_r0",
    "e2_r1",
    "e2_r2",
    "e2_r3",
    "e4_screen",
    "e4_local",
    "e4_profiler",
    "e3b_pilot",
    "e3b_final",
    "e1a",
    "e5_pilot",
    "e5_final",
    "e6_pilot",
    "e6_final",
    "e0_tuning",
    "e0_pilot",
    "e0_final",
]

type FormalSingleOperatorActualValidatorKind = Literal[
    "preflight",
    "run_manifest",
    "profiler_terminal",
    "e5_failure_terminal",
    "e6_interface_preflight",
    "e0_compatibility_terminal",
    "onlinespec_run_manifest",
]

type FormalSingleOperatorAuxiliarySourceKind = Literal[
    "e6_interface_fit",
    "e0_compatibility",
]

FORMAL_SINGLE_OPERATOR_AUXILIARY_SOURCE_KINDS: tuple[
    FormalSingleOperatorAuxiliarySourceKind, ...
] = (
    "e6_interface_fit",
    "e0_compatibility",
)

FORMAL_SINGLE_OPERATOR_IMPLEMENTED_ACTUAL_VALIDATOR_KINDS: frozenset[
    FormalSingleOperatorActualValidatorKind
] = frozenset(
    {
        "preflight",
        "run_manifest",
        "profiler_terminal",
        "e5_failure_terminal",
        "e6_interface_preflight",
        "e0_compatibility_terminal",
        "onlinespec_run_manifest",
    }
)


def _canonical_payload(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_file(value: object) -> bytes:
    return _canonical_payload(value) + b"\n"


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


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
        raise ValueError(f"{label} must be exact non-empty single-line text")
    return value


def _strict(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _array(label: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


_FORBIDDEN_AUTHORITY_FRAGMENTS = (
    "source_replayed",
    "sentinel",
    "future_e0",
    "e0_aggregate",
)


def _reject_future_or_placeholder_authority(value: object, *, label: str) -> None:
    """Reject authority shortcuts that are outside the single-operator mode."""

    if type(value) is str:
        normalized = value.casefold().replace("-", "_").replace(" ", "_")
        if any(fragment in normalized for fragment in _FORBIDDEN_AUTHORITY_FRAGMENTS):
            raise ValueError(f"{label} contains a forbidden future/replay placeholder")
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} object keys must be strings")
        for key, item in value.items():
            _reject_future_or_placeholder_authority(key, label=label)
            _reject_future_or_placeholder_authority(item, label=label)
        return
    if type(value) in {list, tuple}:
        for item in value:
            _reject_future_or_placeholder_authority(item, label=label)


def _canonical_object(label: str, value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    _reject_future_or_placeholder_authority(value, label=label)
    try:
        cloned = json.loads(_canonical_payload(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite canonical JSON") from error
    if type(cloned) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    return cloned


def _absolute_normalized_path(label: str, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError(f"{label} path must be absolute and normalized")
    return path


def _open_safe_parent(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    if not path.name or path == Path(path.anchor):
        raise ValueError(f"{label} must name one file below a directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parent.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"{label} ancestors must be existing symlink-free directories"
                ) from error
            os.close(descriptor)
            descriptor = child
        parent = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
            or stat.S_IMODE(parent.st_mode) & 0o022
        ):
            raise ValueError(
                f"{label} parent must be current-user-owned and non-writable"
            )
        return descriptor, parent
    except Exception:
        os.close(descriptor)
        raise


def _stable_canonical_object(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, object], str, str, int]:
    parent_descriptor, parent_before = _open_safe_parent(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(f"{label} must be a symlink-free regular file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_size < 3
            or before.st_size > FORMAL_SINGLE_OPERATOR_STAGE_ARTIFACT_MAX_BYTES
        ):
            raise ValueError(
                f"{label} must be one bounded current-user-owned non-writable file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_descriptor)
        if (
            len(body) != before.st_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            or (
                parent_before.st_dev,
                parent_before.st_ino,
                parent_before.st_mode,
                parent_before.st_uid,
                parent_before.st_mtime_ns,
            )
            != (
                parent_after.st_dev,
                parent_after.st_ino,
                parent_after.st_mode,
                parent_after.st_uid,
                parent_after.st_mtime_ns,
            )
        ):
            raise RuntimeError(f"{label} changed while read")
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if type(decoded) is not dict or body != _canonical_file(decoded):
        raise ValueError(f"{label} is not one canonical JSON object")
    return (
        decoded,
        hashlib.sha256(body).hexdigest(),
        _content_sha256(decoded),
        len(body),
    )


def _publish_canonical_object_no_replace(path: Path, value: object) -> None:
    destination = _absolute_normalized_path("single-operator artifact", path)
    if type(value) is not dict:
        raise TypeError("single-operator artifact must be a JSON object")
    body = _canonical_file(value)
    if len(body) > FORMAL_SINGLE_OPERATOR_STAGE_ARTIFACT_MAX_BYTES:
        raise ValueError("single-operator artifact exceeds its bounded local schema")
    parent_descriptor, _parent = _open_safe_parent(
        destination,
        label="single-operator artifact",
    )
    temporary_name = f".{destination.name}.tmp.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
        try:
            offset = 0
            while offset < len(body):
                offset += os.write(descriptor, body[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise RuntimeError(
                "single-operator artifact target already exists"
            ) from error
        finally:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


@dataclass(frozen=True)
class FormalSingleOperatorJsonBinding:
    """Large-capable local binding for one canonical no-replace JSON object."""

    absolute_path: str
    raw_sha256: str
    semantic_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _absolute_normalized_path("single-operator binding", self.absolute_path)
        _require_sha256("single-operator binding raw digest", self.raw_sha256)
        _require_sha256(
            "single-operator binding semantic digest",
            self.semantic_sha256,
        )
        if (
            type(self.size_bytes) is not int
            or self.size_bytes < 3
            or self.size_bytes > FORMAL_SINGLE_OPERATOR_STAGE_ARTIFACT_MAX_BYTES
        ):
            raise ValueError("single-operator binding size is invalid")

    @classmethod
    def bind(cls, path: str | Path, *, label: str) -> Self:
        source = _absolute_normalized_path(label, path)
        _value, raw, semantic, size = _stable_canonical_object(source, label=label)
        return cls(str(source), raw, semantic, size)

    def reopen(self, *, label: str) -> dict[str, object]:
        value, raw, semantic, size = _stable_canonical_object(
            _absolute_normalized_path(label, self.absolute_path),
            label=label,
        )
        if (
            raw != self.raw_sha256
            or semantic != self.semantic_sha256
            or size != self.size_bytes
        ):
            raise ValueError(f"{label} changed")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_path": self.absolute_path,
            "raw_sha256": self.raw_sha256,
            "semantic_sha256": self.semantic_sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict(
                "single-operator JSON binding",
                value,
                set(cls.__dataclass_fields__),
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorAuxiliarySourceBinding:
    """Purpose-bound canonical input used by a code-owned stage adapter.

    Binding a source is not a scientific decision.  The consuming materializer
    or physical mapper must still deep-parse the named source and validate its
    typed lineage.  This object only prevents a caller from swapping bytes or
    silently changing the purpose for which those bytes were supplied.
    """

    source_kind: FormalSingleOperatorAuxiliarySourceKind
    source: FormalSingleOperatorJsonBinding

    def __post_init__(self) -> None:
        if self.source_kind not in FORMAL_SINGLE_OPERATOR_AUXILIARY_SOURCE_KINDS:
            raise ValueError("single-operator auxiliary source kind is not registered")
        if type(self.source) is not FormalSingleOperatorJsonBinding:
            raise TypeError("single-operator auxiliary source must be JSON-bound")

    def reopen(self) -> dict[str, object]:
        return self.source.reopen(
            label=f"single-operator {self.source_kind} auxiliary source"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator auxiliary source binding",
            value,
            set(cls.__dataclass_fields__),
        )
        row["source"] = FormalSingleOperatorJsonBinding.from_dict(row["source"])
        return cls(**row)  # type: ignore[arg-type]


def publish_formal_single_operator_json_artifact(
    path: str | Path,
    value: dict[str, object],
) -> FormalSingleOperatorJsonBinding:
    """Publish one local canonical object without replacing existing bytes."""

    destination = _absolute_normalized_path("single-operator artifact", path)
    _publish_canonical_object_no_replace(destination, value)
    return FormalSingleOperatorJsonBinding.bind(
        destination,
        label="single-operator published artifact",
    )


@dataclass(frozen=True)
class FormalSingleOperatorNodeSpec:
    node: FormalSingleOperatorNode
    ordinal: int
    stage: str
    phase: str


FORMAL_SINGLE_OPERATOR_NODE_SPECS: tuple[FormalSingleOperatorNodeSpec, ...] = (
    FormalSingleOperatorNodeSpec("preflight", 0, "preflight", "final"),
    FormalSingleOperatorNodeSpec("e3a", 1, "E3a", "selection"),
    FormalSingleOperatorNodeSpec("tts_cal", 2, "TTS-Cal", "calibration"),
    FormalSingleOperatorNodeSpec("e1", 3, "E1", "selection"),
    FormalSingleOperatorNodeSpec("e2_r0", 4, "E2", "round0"),
    FormalSingleOperatorNodeSpec("e2_r1", 5, "E2", "round1"),
    FormalSingleOperatorNodeSpec("e2_r2", 6, "E2", "round2"),
    FormalSingleOperatorNodeSpec("e2_r3", 7, "E2", "round3"),
    FormalSingleOperatorNodeSpec("e4_screen", 8, "E4", "screen"),
    FormalSingleOperatorNodeSpec("e4_local", 9, "E4", "local"),
    FormalSingleOperatorNodeSpec("e4_profiler", 10, "E4", "profiler"),
    FormalSingleOperatorNodeSpec("e3b_pilot", 11, "E3b", "excluded_pilot"),
    FormalSingleOperatorNodeSpec("e3b_final", 12, "E3b", "final"),
    FormalSingleOperatorNodeSpec("e1a", 13, "E1a", "verification"),
    FormalSingleOperatorNodeSpec("e5_pilot", 14, "E5", "excluded_pilot"),
    FormalSingleOperatorNodeSpec("e5_final", 15, "E5", "final"),
    FormalSingleOperatorNodeSpec("e6_pilot", 16, "E6", "excluded_pilot"),
    FormalSingleOperatorNodeSpec("e6_final", 17, "E6", "final"),
    FormalSingleOperatorNodeSpec("e0_tuning", 18, "E0", "tuning"),
    FormalSingleOperatorNodeSpec("e0_pilot", 19, "E0", "excluded_pilot"),
    FormalSingleOperatorNodeSpec("e0_final", 20, "E0", "final"),
)

FORMAL_SINGLE_OPERATOR_NODE_ORDER: tuple[FormalSingleOperatorNode, ...] = tuple(
    row.node for row in FORMAL_SINGLE_OPERATOR_NODE_SPECS
)

_NODE_BY_NAME = {row.node: row for row in FORMAL_SINGLE_OPERATOR_NODE_SPECS}

_E6_AUXILIARY_NODES = frozenset({"e6_pilot", "e6_final"})
_E0_AUXILIARY_NODES = frozenset({"e0_tuning", "e0_pilot", "e0_final"})


def formal_single_operator_required_auxiliary_source_kinds(
    node: FormalSingleOperatorNode | str,
) -> tuple[FormalSingleOperatorAuxiliarySourceKind, ...]:
    """Return sources that determine the node's scientific cell universe."""

    try:
        spec = _NODE_BY_NAME[node]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("single-operator node is outside the fixed DAG") from error
    kinds: list[FormalSingleOperatorAuxiliarySourceKind] = []
    if spec.node in _E6_AUXILIARY_NODES:
        kinds.append("e6_interface_fit")
    if spec.node in _E0_AUXILIARY_NODES:
        kinds.append("e0_compatibility")
    return tuple(kinds)


def bind_formal_single_operator_auxiliary_sources(
    *,
    node: FormalSingleOperatorNode | str,
    source_paths: Mapping[str, str | Path] | None,
) -> tuple[FormalSingleOperatorAuxiliarySourceBinding, ...]:
    """Bind exactly the auxiliary sources registered for ``node``.

    No source payload is interpreted here.  The stage-specific consumer owns
    that validation.  Requiring the exact key set nevertheless prevents an E0
    compatibility result or an E6 interface/fit result from being omitted or
    substituted under another label.  Post-materialization prepared launches
    are deliberately bound later so their materialization digest cannot form a
    cycle through this artifact.
    """

    required = formal_single_operator_required_auxiliary_source_kinds(node)
    raw = {} if source_paths is None else source_paths
    if type(raw) is not dict:
        raise TypeError("single-operator auxiliary source paths must be an exact map")
    if set(raw) != set(required):
        raise ValueError(
            "single-operator auxiliary source keys differ from the node contract"
        )
    rows = tuple(
        FormalSingleOperatorAuxiliarySourceBinding(
            source_kind=kind,
            source=FormalSingleOperatorJsonBinding.bind(
                raw[kind],
                label=f"single-operator {kind} auxiliary source",
            ),
        )
        for kind in required
    )
    if tuple(row.source_kind for row in rows) != required:
        raise AssertionError("single-operator auxiliary sources are not canonical")
    return rows


def _reopen_formal_single_operator_auxiliary_sources(
    sources: tuple[FormalSingleOperatorAuxiliarySourceBinding, ...],
    *,
    node: FormalSingleOperatorNode | str,
) -> None:
    required = formal_single_operator_required_auxiliary_source_kinds(node)
    if (
        type(sources) is not tuple
        or any(
            type(row) is not FormalSingleOperatorAuxiliarySourceBinding
            for row in sources
        )
        or tuple(row.source_kind for row in sources) != required
    ):
        raise ValueError("single-operator auxiliary source coverage differs")
    for row in sources:
        row.reopen()


_FORMAL_SINGLE_OPERATOR_SCIENTIFIC_STOP_STATUSES = (
    "NO_SAFE_SLO_WINNER",
    "NO_SAFE_GEOMETRY",
    "NO_SAFE_WINNER",
    "NO_SAFE_CONFIGURATION",
    "UNDERPOWERED",
    "POWER_UNRESOLVED",
)


FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema": "formal_single_operator_stage_sequence_v3",
        "mode": FORMAL_SINGLE_OPERATOR_STAGE_MODE,
        "nodes": [
            {
                "node": row.node,
                "ordinal": row.ordinal,
                "stage": row.stage,
                "phase": row.phase,
            }
            for row in FORMAL_SINGLE_OPERATOR_NODE_SPECS
        ],
        "transition": (
            "immediate_predecessor_actual_completion_to_current_materialization_"
            "to_exact_current_actual_results_to_current_reduction"
        ),
        "publication": "canonical_local_atomic_no_replace",
        "scientific_stop": {
            "statuses": list(_FORMAL_SINGLE_OPERATOR_SCIENTIFIC_STOP_STATUSES),
            "next_materialization_authority": "absent",
            "controller_disposition": "durable_blocked_before_downstream",
        },
        "materialization_auxiliary_sources": {
            row.node: list(
                formal_single_operator_required_auxiliary_source_kinds(row.node)
            )
            for row in FORMAL_SINGLE_OPERATOR_NODE_SPECS
            if formal_single_operator_required_auxiliary_source_kinds(row.node)
        },
        "forbidden_inputs": (
            "future_e0_aggregate",
            "source_replayed",
            "sentinel_completion",
            "caller_authored_complete_status",
        ),
        "adversarial_attestation": False,
    }
)


def formal_single_operator_node_spec(
    node: FormalSingleOperatorNode | str,
) -> FormalSingleOperatorNodeSpec:
    try:
        return _NODE_BY_NAME[node]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("single-operator node is outside the fixed DAG") from error


def next_formal_single_operator_node(
    predecessor: FormalSingleOperatorNode | None,
) -> FormalSingleOperatorNode | None:
    if predecessor is None:
        return FORMAL_SINGLE_OPERATOR_NODE_ORDER[0]
    spec = formal_single_operator_node_spec(predecessor)
    next_ordinal = spec.ordinal + 1
    return (
        None
        if next_ordinal == len(FORMAL_SINGLE_OPERATOR_NODE_SPECS)
        else FORMAL_SINGLE_OPERATOR_NODE_SPECS[next_ordinal].node
    )


class FormalSingleOperatorStageBlocked(RuntimeError):
    """The current node lacks a complete actual result and cannot advance."""


@dataclass(frozen=True)
class FormalSingleOperatorActualValidation:
    """Projection returned only by a source-specific actual-result validator."""

    status: Literal["COMPLETE", "FAILED"]
    started_ns: int
    finished_ns: int
    result_identity_sha256: str
    reducer_payload: dict[str, object]

    def __post_init__(self) -> None:
        if self.status not in {"COMPLETE", "FAILED"}:
            raise ValueError("single-operator actual status differs")
        if (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 0
            or self.finished_ns <= self.started_ns
        ):
            raise ValueError("single-operator actual timing is invalid")
        _require_sha256(
            "single-operator actual result identity",
            self.result_identity_sha256,
        )
        object.__setattr__(
            self,
            "reducer_payload",
            _canonical_object(
                "single-operator actual reducer payload",
                self.reducer_payload,
            ),
        )


class FormalSingleOperatorActualResultValidator(Protocol):
    """Source-specific validator used before a result may enter a reducer."""

    @property
    def validator_kind(self) -> str: ...

    @property
    def protocol_sha256(self) -> str: ...

    def validate(
        self,
        *,
        path: Path,
        node: FormalSingleOperatorNodeSpec,
        materialization: StageMaterializationReceipt,
        cell: MaterializedCell,
    ) -> FormalSingleOperatorActualValidation: ...


@dataclass(frozen=True)
class FormalSingleOperatorCellValidatorRoute:
    """One non-overlapping code-owned cell-to-validator route."""

    validator_kind: FormalSingleOperatorActualValidatorKind
    nodes: tuple[FormalSingleOperatorNode, ...]
    task: str | None = None
    method_role_prefix: str | None = None

    def __post_init__(self) -> None:
        if (
            self.validator_kind
            not in {
                "preflight",
                "run_manifest",
                "profiler_terminal",
                "e5_failure_terminal",
                "e6_interface_preflight",
                "e0_compatibility_terminal",
                "onlinespec_run_manifest",
            }
            or type(self.nodes) is not tuple
            or not self.nodes
            or tuple(sorted(set(self.nodes))) != tuple(sorted(self.nodes))
        ):
            raise ValueError("single-operator validator route identity differs")
        for node in self.nodes:
            formal_single_operator_node_spec(node)
        if self.task is not None:
            _require_text("single-operator validator route task", self.task)
        if self.method_role_prefix is not None:
            _require_text(
                "single-operator validator route role prefix",
                self.method_role_prefix,
            )
        if (self.task is None) == (self.method_role_prefix is None):
            raise ValueError(
                "single-operator validator route requires one exact selector"
            )

    def matches(
        self,
        *,
        node: FormalSingleOperatorNode,
        cell: MaterializedCell,
    ) -> bool:
        if node not in self.nodes:
            return False
        if self.task is not None:
            return cell.task == self.task
        assert self.method_role_prefix is not None
        return cell.method_role.startswith(self.method_role_prefix)


FORMAL_SINGLE_OPERATOR_CELL_VALIDATOR_ROUTES: tuple[
    FormalSingleOperatorCellValidatorRoute, ...
] = (
    FormalSingleOperatorCellValidatorRoute(
        validator_kind="profiler_terminal",
        nodes=("e4_profiler",),
        task="mechanism_profile_only",
    ),
    FormalSingleOperatorCellValidatorRoute(
        validator_kind="e5_failure_terminal",
        nodes=("e5_final",),
        task="deterministic_failure_injection",
    ),
    FormalSingleOperatorCellValidatorRoute(
        validator_kind="e6_interface_preflight",
        nodes=("e6_final", "e6_pilot"),
        task="immutable_metadata_interface_and_fit_preflight",
    ),
    FormalSingleOperatorCellValidatorRoute(
        validator_kind="e0_compatibility_terminal",
        nodes=("e0_tuning",),
        task="compatibility_decision",
    ),
    FormalSingleOperatorCellValidatorRoute(
        validator_kind="onlinespec_run_manifest",
        nodes=("e0_final", "e0_pilot", "e0_tuning"),
        method_role_prefix="OnlineSPEC-",
    ),
)


def formal_single_operator_cell_validator_kind(
    *,
    node: FormalSingleOperatorNode | str,
    cell: MaterializedCell,
) -> FormalSingleOperatorActualValidatorKind:
    """Select a validator from immutable cell identity, never caller input."""

    spec = formal_single_operator_node_spec(node)
    if type(cell) is not MaterializedCell:
        raise TypeError(
            "single-operator validator routing requires a materialized cell"
        )
    if cell.stage != spec.stage:
        raise ValueError("single-operator validator route cell stage differs")
    if spec.node == "preflight":
        return "preflight"
    matches = tuple(
        route.validator_kind
        for route in FORMAL_SINGLE_OPERATOR_CELL_VALIDATOR_ROUTES
        if route.matches(node=spec.node, cell=cell)
    )
    if len(matches) > 1:
        raise AssertionError("single-operator cell matches multiple validator routes")
    return "run_manifest" if not matches else matches[0]


def formal_single_operator_actual_validator_kinds(
    node: FormalSingleOperatorNode | str,
) -> tuple[FormalSingleOperatorActualValidatorKind, ...]:
    """Return every validator kind that the registered node matrix may emit."""

    spec = formal_single_operator_node_spec(node)
    if spec.node == "preflight":
        return ("preflight",)
    special = tuple(
        dict.fromkeys(
            route.validator_kind
            for route in FORMAL_SINGLE_OPERATOR_CELL_VALIDATOR_ROUTES
            if spec.node in route.nodes
        )
    )
    if spec.node == "e4_profiler":
        return special
    return ("run_manifest", *special)


@dataclass(frozen=True)
class FormalSingleOperatorPreflightActualReceipt:
    """Canonical bridge from the real preflight finalizer to this local DAG."""

    schema_version: int
    kind: Literal["formal_single_operator_preflight_actual"]
    protocol_sha256: str
    protocol_lock_sha256: str
    materialization_sha256: str
    final_evidence_source: FormalSingleOperatorJsonBinding
    final_evidence_sha256: str
    stage_coverage_sha256: str
    e3a_workload_authority_sha256: str
    verified_ns: int
    started_ns: int
    finished_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != "formal_single_operator_preflight_actual"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator preflight actual schema differs")
        for label, digest in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("materialization", self.materialization_sha256),
            ("final evidence", self.final_evidence_sha256),
            ("stage coverage", self.stage_coverage_sha256),
            ("E3a workload authority", self.e3a_workload_authority_sha256),
        ):
            _require_sha256(f"single-operator preflight {label}", digest)
        if type(self.final_evidence_source) is not FormalSingleOperatorJsonBinding:
            raise TypeError(
                "single-operator preflight requires a final-evidence source"
            )
        if (
            type(self.verified_ns) is not int
            or type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.verified_ns < 0
            or self.started_ns < 0
            or self.finished_ns <= self.started_ns
        ):
            raise ValueError("single-operator preflight timing is invalid")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_sha256": self.materialization_sha256,
            "final_evidence_source": self.final_evidence_source.to_dict(),
            "final_evidence_sha256": self.final_evidence_sha256,
            "stage_coverage_sha256": self.stage_coverage_sha256,
            "e3a_workload_authority_sha256": (self.e3a_workload_authority_sha256),
            "verified_ns": self.verified_ns,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
        }
        if include_sha256:
            value["receipt_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator preflight actual",
            value,
            set(cls.__dataclass_fields__) | {"receipt_sha256"},
        )
        expected = _require_sha256(
            "single-operator preflight actual",
            row.pop("receipt_sha256"),
        )
        row["final_evidence_source"] = FormalSingleOperatorJsonBinding.from_dict(
            row["final_evidence_source"]
        )
        receipt = cls(**row)  # type: ignore[arg-type]
        if receipt.sha256 != expected:
            raise ValueError("single-operator preflight actual digest differs")
        return receipt


def _revalidate_formal_single_operator_preflight_final_evidence(
    source: FormalSingleOperatorJsonBinding,
    *,
    verified_ns: int,
) -> object:
    """Replay the existing preflight finalizer graph from one bound source."""

    from lightcone_spec.experiments.formal_preflight_coverage import (
        revalidate_formal_preflight_stage_coverage_proof_artifact,
    )

    if type(source) is not FormalSingleOperatorJsonBinding:
        raise TypeError("single-operator preflight evidence source differs")
    before = source.reopen(label="single-operator preflight final-evidence source")
    evidence = revalidate_formal_preflight_stage_coverage_proof_artifact(
        source.absolute_path,
        now_ns=verified_ns,
    )
    after = source.reopen(label="single-operator preflight final-evidence source")
    if after != before:
        raise RuntimeError("single-operator preflight final evidence changed")
    return evidence


def _preflight_e3a_workload_authority_sha256(
    *,
    execution_inputs: object,
    protocol_lock: ProtocolLock,
) -> str:
    """Recover the E3a workload identity from the preflight's bound input.

    Schema-4 locks retain the root-signed workload authorization identity and
    therefore keep the legacy transition unchanged.  A trusted schema-5 lock
    intentionally carries no such signed field; in that mode the only valid
    identity is the exact ``FormalWorkloadAuthority`` already path-bound in
    the deeply replayed preflight inputs and matched by the locked content
    bundle.
    """

    from lightcone_spec.experiments.formal_preflight_inputs import (
        FormalPreflightExecutionInputs,
        _trusted_content_sources,
    )

    if type(execution_inputs) is not FormalPreflightExecutionInputs:
        raise TypeError("preflight workload derivation requires exact inputs")
    if protocol_lock.schema_version == 4:
        return _require_sha256(
            "legacy E3a workload authorization",
            protocol_lock.formal_workload_e3a_authorization_sha256,
        )
    if (
        protocol_lock.schema_version != 5
        or execution_inputs.schema_version not in {3, 4}
        or execution_inputs.content_source_binding is None
        or protocol_lock.trusted_single_operator_content_bundle_sha256
        != execution_inputs.content_source_binding.content_sha256
    ):
        raise ValueError("trusted preflight workload/content lineage differs")
    (
        _bundle,
        workload_binding,
        workload,
        _locked_workload,
        _target,
        _drafter,
        _tokenizer,
    ) = _trusted_content_sources(
        content_source_binding=execution_inputs.content_source_binding,
        workload_authority_path=execution_inputs.workload_authority.path,
    )
    if workload_binding != execution_inputs.workload_authority:
        raise ValueError("trusted preflight workload binding changed")
    return _require_sha256("trusted E3a workload authority", workload.sha256)


def publish_formal_single_operator_preflight_actual(
    *,
    final_evidence_source_path: str | Path,
    protocol_lock: object,
    verified_ns: int,
    started_ns: int,
    finished_ns: int,
    output_path: str | Path,
) -> FormalSingleOperatorPreflightActualReceipt:
    """Publish only a real source-finalized, all-COMPLETE preflight outcome."""

    from lightcone_spec.experiments.formal_protocol import ProtocolLock

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("single-operator preflight requires an exact ProtocolLock")
    if protocol_lock.schema_version != 4:
        raise ValueError(
            "trusted schema-5 preflight actuals require the exact-ten completion"
        )
    final_evidence_source = FormalSingleOperatorJsonBinding.bind(
        final_evidence_source_path,
        label="single-operator preflight final-evidence proof",
    )
    final_evidence = _revalidate_formal_single_operator_preflight_final_evidence(
        final_evidence_source,
        verified_ns=verified_ns,
    )
    if (
        final_evidence.materialization.protocol_lock_sha256 != protocol_lock.sha256
        or final_evidence.stage_coverage.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("single-operator preflight belongs to another ProtocolLock")
    final_evidence.stage_coverage.validate_against(final_evidence.materialization)
    receipt = FormalSingleOperatorPreflightActualReceipt(
        schema_version=2,
        kind="formal_single_operator_preflight_actual",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_sha256=final_evidence.materialization.sha256,
        final_evidence_source=final_evidence_source,
        final_evidence_sha256=final_evidence.sha256,
        stage_coverage_sha256=final_evidence.stage_coverage.sha256,
        e3a_workload_authority_sha256=(
            protocol_lock.formal_workload_e3a_authorization_sha256
        ),
        verified_ns=verified_ns,
        started_ns=started_ns,
        finished_ns=finished_ns,
    )
    _publish_canonical_object_no_replace(
        _absolute_normalized_path(
            "single-operator preflight actual output",
            output_path,
        ),
        receipt.to_dict(),
    )
    return receipt


@dataclass(frozen=True)
class _FormalSingleOperatorPreflightActualValidator:
    @property
    def validator_kind(self) -> str:
        return "formal_single_operator_preflight_actual_revalidator"

    @property
    def protocol_sha256(self) -> str:
        return FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256

    def validate(
        self,
        *,
        path: Path,
        node: FormalSingleOperatorNodeSpec,
        materialization: StageMaterializationReceipt,
        cell: MaterializedCell,
    ) -> FormalSingleOperatorActualValidation:
        if node.node != "preflight":
            raise ValueError("preflight validator cannot validate another node")
        source = FormalSingleOperatorJsonBinding.bind(
            path,
            label="single-operator preflight actual",
        )
        raw = source.reopen(label="single-operator preflight actual")
        if (
            type(raw) is dict
            and raw.get("kind")
            == "formal_single_operator_exact_ten_preflight_completion"
        ):
            from lightcone_spec.experiments.formal_preflight_inputs import (
                FormalPreflightExecutionInputs,
                FormalSingleOperatorPreflightAuthority,
                FormalSingleOperatorPreflightCompletion,
                revalidate_formal_single_operator_preflight_completion,
            )

            serialized = FormalSingleOperatorPreflightCompletion.from_dict(raw)
            completion = revalidate_formal_single_operator_preflight_completion(
                path,
                current_ns=serialized.finished_ns,
            )
            inputs = FormalPreflightExecutionInputs.from_dict(
                completion.execution_inputs.reopen()
            )
            authority = FormalSingleOperatorPreflightAuthority.from_dict(
                inputs.execution_authority.reopen()
            )
            completed_materialization = stage_materialization_receipt_from_dict(
                authority.materialization.reopen()
            )
            protocol_lock = protocol_lock_from_dict(authority.protocol_lock.reopen())
            e3a_workload_authority_sha256 = _preflight_e3a_workload_authority_sha256(
                execution_inputs=inputs,
                protocol_lock=protocol_lock,
            )
            matches = tuple(
                row
                for row in completion.rows
                if row.materialized_cell_id == cell.cell_id
            )
            if (
                completion.status != "COMPLETE"
                or completed_materialization != materialization
                or protocol_lock.sha256 != materialization.protocol_lock_sha256
                or len(matches) != 1
                or matches[0].status != "COMPLETE"
            ):
                raise ValueError("preflight completion differs from the current cell")
            completed = matches[0]
            return FormalSingleOperatorActualValidation(
                status="COMPLETE",
                started_ns=completed.started_ns,
                finished_ns=completed.finished_ns,
                result_identity_sha256=_content_sha256(
                    {
                        "completion_sha256": completion.sha256,
                        "cell_id": cell.cell_id,
                        "result_sha256": completed.result_sha256,
                    }
                ),
                reducer_payload={
                    "e3a_workload_authority_sha256": (e3a_workload_authority_sha256),
                    "preflight_completion_sha256": completion.sha256,
                    "preflight_result_sha256": completed.result_sha256,
                },
            )
        receipt = FormalSingleOperatorPreflightActualReceipt.from_dict(raw)
        final_evidence = _revalidate_formal_single_operator_preflight_final_evidence(
            receipt.final_evidence_source,
            verified_ns=receipt.verified_ns,
        )
        if (
            final_evidence.sha256 != receipt.final_evidence_sha256
            or final_evidence.stage_coverage.sha256 != receipt.stage_coverage_sha256
            or final_evidence.materialization.sha256 != receipt.materialization_sha256
        ):
            raise ValueError("single-operator preflight final evidence differs")
        final_evidence.stage_coverage.validate_against(materialization)
        matches = tuple(
            row
            for row in final_evidence.stage_coverage.dispositions
            if row.cell_id == cell.cell_id
        )
        if (
            receipt.materialization_sha256 != materialization.sha256
            or receipt.protocol_lock_sha256 != materialization.protocol_lock_sha256
            or len(matches) != 1
            or matches[0].status != "COMPLETE"
            or matches[0].terminal_receipt_sha256 is None
        ):
            raise ValueError("preflight actual differs from the current cell")
        return FormalSingleOperatorActualValidation(
            status="COMPLETE",
            started_ns=receipt.started_ns,
            finished_ns=receipt.finished_ns,
            result_identity_sha256=_content_sha256(
                {
                    "receipt_sha256": receipt.sha256,
                    "cell_id": cell.cell_id,
                    "terminal_receipt_sha256": matches[0].terminal_receipt_sha256,
                }
            ),
            reducer_payload={
                "e3a_workload_authority_sha256": (
                    receipt.e3a_workload_authority_sha256
                ),
                "final_evidence_sha256": receipt.final_evidence_sha256,
                "preflight_actual_receipt_sha256": receipt.sha256,
                "stage_coverage_sha256": receipt.stage_coverage_sha256,
                "terminal_receipt_sha256": matches[0].terminal_receipt_sha256,
            },
        )


def _single_operator_runtime_method(cell: MaterializedCell) -> str:
    try:
        return {
            "Target-only": "target_only",
            "Static": "static",
            "TTS": "tts",
            "L0-naive": "l0",
            "LightCone": "l0",
            "LightCone-candidate": "l0",
            "OnlineSPEC-OGD": "onlinespec_ogd",
            "OnlineSPEC-OGD-candidate": "onlinespec_ogd",
            "OnlineSPEC-OPT": "onlinespec_opt",
            "OnlineSPEC-OPT-candidate": "onlinespec_opt",
            "OnlineSPEC-Optimistic-OGD": "onlinespec_opt",
            "OnlineSPEC-Optimistic-OGD-candidate": "onlinespec_opt",
            "OnlineSPEC-ENS": "onlinespec_ens",
            "OnlineSPEC-ENS-candidate": "onlinespec_ens",
            "OnlineSPEC-Hedge": "onlinespec_ens",
            "OnlineSPEC-Hedge-candidate": "onlinespec_ens",
        }[cell.method_role]
    except KeyError as error:
        raise ValueError(
            "single-operator serving cell has no code-owned runtime method"
        ) from error


def _validated_single_operator_serving_payload(
    *,
    manifest: object,
    cell: MaterializedCell,
) -> dict[str, object]:
    """Deep-parse the manifest-bound terminal and native timestamp artifacts."""

    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
    )
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
        FormalServingRunPlan,
        formal_serving_request_schedule_rows,
    )
    from lightcone_spec.orchestration.formal_terminal_shards import (
        reopen_scalable_client_request_lifecycle,
        reopen_scalable_formal_gang_itl_bundle,
        reopen_scalable_formal_gang_request_terminal,
    )
    from lightcone_spec.orchestration.live_sglang import (
        UnsignedPinnedSglangServingRunReceipt,
    )
    from lightcone_spec.orchestration.native_terminal import (
        NO_TRUSTED_ATTESTERS,
        canonical_sha256,
        validate_native_terminal_artifact,
        validate_unsigned_native_itl_pointer_bundle,
    )
    from lightcone_spec.runtime.formal_single_operator import (
        FormalSingleOperatorResidentRunManifest,
        FormalSingleOperatorRunManifest,
    )
    from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

    resident = type(manifest) is FormalSingleOperatorResidentRunManifest
    if type(manifest) not in {
        FormalSingleOperatorRunManifest,
        FormalSingleOperatorResidentRunManifest,
    }:
        raise TypeError("single-operator serving payload requires a run manifest")
    if type(cell) is not MaterializedCell:
        raise TypeError("single-operator serving payload requires a materialized cell")
    if (
        manifest.role != cell.method_role
        or manifest.backend != cell.backend
        or manifest.target_model_id != cell.model
        or manifest.completion_status != "COMPLETE"
        or (not resident and manifest.exit_code not in {0, -15})
    ):
        raise ValueError("single-operator serving manifest differs from the cell")
    artifact_by_name = {row.name: row for row in manifest.artifacts}
    required = ("raw_terminal", "native_itl", "live_run_receipt", "run_plan")
    if any(
        name not in artifact_by_name or artifact_by_name[name].status != "PRESENT"
        for name in required
    ):
        raise ValueError("single-operator serving manifest lacks terminal timing")
    run_root = Path(manifest.run_directory)
    terminal_path = run_root / artifact_by_name["raw_terminal"].relative_path
    itl_path = run_root / artifact_by_name["native_itl"].relative_path
    live_path = run_root / artifact_by_name["live_run_receipt"].relative_path
    plan_path = run_root / artifact_by_name["run_plan"].relative_path
    terminal_binding = CanonicalJsonProofBinding.bind(terminal_path)
    timing_binding = CanonicalJsonProofBinding.bind(itl_path)
    live_binding = CanonicalJsonProofBinding.bind(live_path)
    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if plan.sha256 != plan_binding.semantic_sha256:
        raise ValueError("single-operator run plan identity differs")
    live_value = live_binding.reopen()
    if type(live_value) is not dict:
        raise TypeError("single-operator live receipt must be an object")
    schedule = FormalServingRequestScheduleReceipt.from_dict(manifest.request_schedule)
    if (
        schedule.sha256 != manifest.request_schedule_sha256
        or schedule.materialized_cell_id != cell.cell_id
        or schedule.topology_mode != manifest.topology
        or schedule.workload_authority_sha256
        != getattr(manifest, "workload_authority_sha256", None)
    ):
        raise ValueError("single-operator request schedule differs from the run")
    schedule_rows = formal_serving_request_schedule_rows(schedule)
    warmup = tuple(row.request for row in schedule_rows if row.phase == "warmup")
    scored = tuple(row.request for row in schedule_rows if row.phase == "scored")
    binding = plan.native_terminal_binding
    if resident:
        from lightcone_spec.orchestration.formal_serving_session_group_physical import (
            revalidate_formal_serving_resident_trace_receipt,
        )

        _resident_trace_binding, resident_trace = (
            revalidate_formal_serving_resident_trace_receipt(live_path)
        )
        if (
            resident_trace.raw_terminal != terminal_binding
            or resident_trace.native_itl != timing_binding
            or resident_trace.member_run_plan != plan_binding
            or resident_trace.materialized_cell_id != cell.cell_id
        ):
            raise ValueError("resident serving trace differs from manifest artifacts")
        binding = resident_trace.effective_terminal_binding
    if (
        plan.materialized_cell_id != cell.cell_id
        or binding.method != _single_operator_runtime_method(cell)
        or binding.warmup_request_ids != tuple(row.request_id for row in warmup)
        or binding.scored_request_ids != tuple(row.request_id for row in scored)
    ):
        raise ValueError("single-operator terminal binding differs from the schedule")

    policy: RegisteredServingExecutionPolicy | None = None
    lifecycle_binding: CanonicalJsonProofBinding | None = None
    scored_origin_ns: int
    terminal_rows: dict[str, dict[str, object]]
    timing_rows: dict[str, dict[str, object]]
    terminal_sha256: str
    if live_value.get("kind") == "formal_serving_resident_trace_receipt":
        if not resident:
            raise ValueError("fresh manifest cannot consume a resident trace")
        policy = plan.serving_execution_policy
        lifecycle_binding = resident_trace.client_lifecycle
        scored_origin_ns = resident_trace.scored_started_ns
        terminal = validate_native_terminal_artifact(
            terminal_binding.reopen(),
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        )
        if terminal.binding != binding:
            raise ValueError("resident TP1 terminal binding differs")
        terminal_rows = {
            row.request_id: {
                "request_id": row.request_id,
                "input_token_ids": row.input_token_ids,
                "output_token_ids": row.output_token_ids,
                "terminal_status": row.terminal_status,
                "terminal_reason": row.terminal_reason,
                "submitted_to_server": row.submitted_to_server,
            }
            for row in terminal.requests
        }
        completed_outputs = {
            request_id: tuple(row["output_token_ids"])
            for request_id, row in terminal_rows.items()
            if row["terminal_status"] == "completed"
            and row["output_token_ids"] is not None
        }
        timing = validate_unsigned_native_itl_pointer_bundle(
            timing_binding,
            expected_binding=binding,
            expected_terminal_artifact=terminal_binding,
            expected_scored_request_inputs_sha256=canonical_sha256(
                [request.sha256 for request in scored]
            ),
            expected_terminal_output_tokens=completed_outputs,
        )
        timing_rows = {
            row.request_id: {
                "request_started_ns": row.request_started_ns,
                "request_terminal_ns": row.request_terminal_ns,
                "terminal_status": row.terminal_status,
                "terminal_reason": row.terminal_reason,
                "output_token_ids": tuple(event.token_id for event in row.events),
                "token_observed_ns": tuple(event.observed_ns for event in row.events),
            }
            for row in timing.pointers
        }
        terminal_value = terminal.to_dict()
        performance = terminal_value.get("performance_counters")
        terminal_sha256 = terminal.terminal_sha256
    elif live_value.get("kind") == "unsigned_pinned_sglang_serving_run_receipt":
        if resident:
            raise ValueError("resident manifest cannot consume a fresh receipt")
        live = UnsignedPinnedSglangServingRunReceipt.from_dict(live_value)
        if (
            live.sha256 != live_binding.semantic_sha256
            or live.run_binding_sha256 != canonical_sha256(binding.begin_payload())
        ):
            raise ValueError("single-operator TP1 live receipt differs")
        policy = live.execution_policy
        lifecycle_binding = live.client_request_lifecycle
        scored_origin_ns = live.scored_started_ns
        terminal = validate_native_terminal_artifact(
            terminal_binding.reopen(),
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        )
        if terminal.binding != binding:
            raise ValueError("single-operator TP1 terminal binding differs")
        terminal_rows = {
            row.request_id: {
                "request_id": row.request_id,
                "input_token_ids": row.input_token_ids,
                "output_token_ids": row.output_token_ids,
                "terminal_status": row.terminal_status,
                "terminal_reason": row.terminal_reason,
                "submitted_to_server": row.submitted_to_server,
            }
            for row in terminal.requests
        }
        completed_outputs = {
            request_id: tuple(row["output_token_ids"])
            for request_id, row in terminal_rows.items()
            if row["terminal_status"] == "completed"
            and row["output_token_ids"] is not None
        }
        timing = validate_unsigned_native_itl_pointer_bundle(
            timing_binding,
            expected_binding=binding,
            expected_terminal_artifact=terminal_binding,
            expected_scored_request_inputs_sha256=canonical_sha256(
                [request.sha256 for request in scored]
            ),
            expected_terminal_output_tokens=completed_outputs,
        )
        timing_rows = {
            row.request_id: {
                "request_started_ns": row.request_started_ns,
                "request_terminal_ns": row.request_terminal_ns,
                "terminal_status": row.terminal_status,
                "terminal_reason": row.terminal_reason,
                "output_token_ids": tuple(event.token_id for event in row.events),
                "token_observed_ns": tuple(event.observed_ns for event in row.events),
            }
            for row in timing.pointers
        }
        terminal_value = terminal.to_dict()
        performance = terminal_value.get("performance_counters")
        terminal_sha256 = terminal.terminal_sha256
    elif live_value.get("kind") == "unsigned_formal_gang_physical_run_receipt":
        if live_value.get("schema_version") != 2:
            raise ValueError("current distributed live receipt schema differs")
        policy = RegisteredServingExecutionPolicy.from_dict(
            live_value.get("serving_execution_policy")
        )
        lifecycle_binding = CanonicalJsonProofBinding.from_dict(
            live_value.get("client_request_lifecycle")
        )
        edges = live_value.get("phase_edges_ns")
        if type(edges) is not dict or type(edges.get("scored_started_ns")) is not int:
            raise ValueError("distributed scored phase origin is unavailable")
        scored_origin_ns = int(edges["scored_started_ns"])
        gang_terminal = reopen_scalable_formal_gang_request_terminal(
            terminal_binding.reopen()
        )
        scored_values = gang_terminal.get("scored_requests")
        if type(scored_values) is not list:
            raise TypeError("distributed scored terminal rows must be an array")
        terminal_rows = {
            str(row["request_id"]): dict(row)
            for row in scored_values
            if type(row) is dict and type(row.get("request_id")) is str
        }
        pointer_bundle = reopen_scalable_formal_gang_itl_bundle(timing_binding.reopen())
        pointer_values = pointer_bundle.get("scored_pointers")
        if type(pointer_values) is not list:
            raise TypeError("distributed scored pointers must be an array")
        timing_rows = {
            str(row["request_id"]): {
                "request_started_ns": row["request_started_ns"],
                "request_terminal_ns": row["request_terminal_ns"],
                "terminal_status": row["terminal_status"],
                "terminal_reason": row["terminal_reason"],
                "output_token_ids": tuple(event["token_id"] for event in row["events"]),
                "token_observed_ns": tuple(
                    event["observed_ns"] for event in row["events"]
                ),
            }
            for row in pointer_values
            if type(row) is dict
            and type(row.get("request_id")) is str
            and type(row.get("events")) is list
        }
        from lightcone_spec.orchestration.formal_terminal_result import _performance
        from lightcone_spec.orchestration.formal_terminal_shards import (
            reopen_scalable_formal_gang_terminal,
        )

        gang_binding = CanonicalJsonProofBinding.from_dict(
            live_value.get("formal_gang_terminal")
        )
        gang = reopen_scalable_formal_gang_terminal(gang_binding.reopen())
        ranks = gang.get("rank_terminals")
        if type(ranks) is not list or len(ranks) != 2:
            raise ValueError("distributed rank terminal coverage differs")
        rank_performance = []
        for rank in ranks:
            native_state = None if type(rank) is not dict else rank.get("native_state")
            counters = (
                None
                if type(native_state) is not dict
                else native_state.get("performance_counters")
            )
            if type(counters) is not dict:
                raise ValueError("distributed performance counters are unavailable")
            rank_performance.append(dict(counters))
        performance = _performance(
            (rank_performance[0], rank_performance[1]),
            topology=plan.topology_mode,
        )
        terminal_sha256 = terminal_binding.semantic_sha256
    else:
        raise ValueError("single-operator live serving receipt kind differs")
    if type(performance) is not dict:
        raise ValueError("single-operator terminal lacks performance counters")
    if policy is None or lifecycle_binding is None:
        raise ValueError("fresh serving evidence lacks its registered lifecycle")
    if plan.serving_execution_policy != policy:
        raise ValueError("serving evidence policy differs from the run plan")
    lifecycle_rows = reopen_scalable_client_request_lifecycle(
        lifecycle_binding,
        expected_run_binding_sha256=canonical_sha256(binding.begin_payload()),
        expected_execution_policy_sha256=policy.sha256,
    )
    expected_all_ids = (*binding.warmup_request_ids, *binding.scored_request_ids)
    if tuple(row.get("request_id") for row in lifecycle_rows) != expected_all_ids:
        raise ValueError("serving client lifecycle coverage differs from schedule")
    lifecycle_by_id = {
        str(row["request_id"]): row
        for row in lifecycle_rows
        if row.get("phase") == "scored"
    }
    if set(terminal_rows) != {row.request_id for row in scored}:
        raise ValueError("single-operator terminal request coverage differs")
    schedule_by_id = {row.request_id: row for row in scored}
    requests = []
    outcome_counts = {
        status: 0
        for status in ("completed", "rejected", "timed_out", "cancelled", "unfinished")
    }
    for request_id in sorted(schedule_by_id):
        sealed = schedule_by_id[request_id]
        terminal_request = terminal_rows[request_id]
        lifecycle = lifecycle_by_id[request_id]
        if not bool(lifecycle.get("offered")):
            continue
        status = lifecycle.get("outcome_status")
        if status not in outcome_counts:
            raise ValueError("serving client outcome status differs")
        outcome_counts[str(status)] += 1
        inputs = terminal_request.get("input_token_ids")
        if tuple(inputs) != sealed.input_token_ids:  # type: ignore[arg-type]
            raise ValueError("single-operator terminal input tokens differ")
        native_status = terminal_request.get("terminal_status")
        submitted = bool(terminal_request.get("submitted_to_server"))
        clock = timing_rows.get(request_id)
        if status == "completed":
            outputs = terminal_request.get("output_token_ids")
            if (
                native_status != "completed"
                or not submitted
                or type(outputs) not in {list, tuple}
                or clock is None
                or tuple(outputs) != clock["output_token_ids"]
            ):
                raise ValueError("completed serving trajectory is not exact")
            output_ids = tuple(outputs)
            token_times = tuple(clock["token_observed_ns"])
            terminal_ns = int(clock["request_terminal_ns"])
        else:
            expected_native = lifecycle.get("native_terminal_status")
            if expected_native is not None and native_status != expected_native:
                raise ValueError("client/native negative terminal mapping differs")
            if clock is not None:
                raise ValueError("non-completed request carries headline ITL timing")
            output_ids = ()
            token_times = ()
            logical_terminal_us = (
                lifecycle.get("effective_deadline_us")
                if status == "unfinished"
                else lifecycle.get("terminal_at_us")
            )
            if type(logical_terminal_us) is not int:
                raise ValueError(
                    "negative serving request lacks its effective terminal"
                )
            terminal_ns = scored_origin_ns + logical_terminal_us * 1_000
        offered_at_us = lifecycle.get("offered_at_us")
        if type(offered_at_us) is not int:
            raise ValueError("offered serving request lacks its offer timestamp")
        started_ns = scored_origin_ns + offered_at_us * 1_000
        terminal_ns = max(started_ns + 1, terminal_ns)
        requests.append(
            {
                "request_id": request_id,
                "input_token_ids": list(sealed.input_token_ids),
                "output_token_ids": list(output_ids),
                "request_started_ns": started_ns,
                "request_terminal_ns": terminal_ns,
                "token_observed_ns": list(token_times),
                "terminal_status": status,
                "terminal_reason": lifecycle.get("outcome_code"),
                "submitted_to_server": lifecycle.get("submitted_to_server"),
                "native_terminal_status": lifecycle.get("native_terminal_status"),
                "offered_at_us": offered_at_us,
                "admitted_at_us": lifecycle.get("admitted_at_us"),
                "effective_deadline_us": lifecycle.get("effective_deadline_us"),
                "terminal_at_us": lifecycle.get("terminal_at_us"),
            }
        )
    if not requests or sum(outcome_counts.values()) != len(requests):
        raise ValueError("serving offered denominator coverage differs")
    return _canonical_object(
        "single-operator serving observation",
        {
            "schema_version": 2,
            "kind": "formal_single_operator_serving_observation",
            "materialized_cell_id": cell.cell_id,
            "inventory_sha256": manifest.inventory_sha256,
            "run_id": binding.run_id,
            "run_nonce_sha256": binding.run_nonce_sha256,
            "attempt_id": binding.attempt_id,
            "method": binding.method,
            "terminal_sha256": terminal_sha256,
            "terminal_artifact_sha256": terminal_binding.semantic_sha256,
            "native_itl_artifact_sha256": timing_binding.semantic_sha256,
            "request_schedule_sha256": schedule.sha256,
            "source_request_pool_sha256": _content_sha256(
                [
                    {
                        "request_id": row.request_id,
                        "input_token_ids": list(row.input_token_ids),
                    }
                    for row in scored
                ]
            ),
            "serving_execution_policy_sha256": policy.sha256,
            "client_lifecycle_artifact_sha256": (lifecycle_binding.semantic_sha256),
            "scored_phase_origin_ns": scored_origin_ns,
            "scheduled_request_count": len(scored),
            "offered_request_count": len(requests),
            "outcome_counts": outcome_counts,
            "requests": requests,
            "performance_counters": performance,
        },
    )


@dataclass(frozen=True)
class FormalSingleOperatorRunManifestActualValidator:
    """Deep validator for a serving cell's actual single-operator manifest."""

    repository_root: str

    def __post_init__(self) -> None:
        root = _absolute_normalized_path(
            "single-operator validator repository",
            self.repository_root,
        )
        if not root.is_dir() or root.is_symlink():
            raise ValueError("single-operator validator repository must be a directory")

    @property
    def validator_kind(self) -> str:
        return "formal_single_operator_run_manifest_revalidator"

    @property
    def protocol_sha256(self) -> str:
        from lightcone_spec.runtime.formal_single_operator import (
            FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256,
            FORMAL_SINGLE_OPERATOR_RESIDENT_PROTOCOL_SHA256,
        )

        return _content_sha256(
            {
                "kind": self.validator_kind,
                "fresh_protocol_sha256": FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256,
                "resident_protocol_sha256": (
                    FORMAL_SINGLE_OPERATOR_RESIDENT_PROTOCOL_SHA256
                ),
                "output": "formal_single_operator_serving_observation_v2",
            }
        )

    def validate(
        self,
        *,
        path: Path,
        node: FormalSingleOperatorNodeSpec,
        materialization: StageMaterializationReceipt,
        cell: MaterializedCell,
    ) -> FormalSingleOperatorActualValidation:
        from lightcone_spec.runtime.formal_single_operator import (
            revalidate_formal_single_operator_resident_run_manifest,
            revalidate_formal_single_operator_run_manifest,
        )
        from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

        root_value = CanonicalJsonProofBinding.bind(path).reopen()
        if type(root_value) is not dict:
            raise TypeError("single-operator actual manifest must be an object")
        if root_value.get("kind") == "formal_single_operator_resident_run_manifest":
            manifest = revalidate_formal_single_operator_resident_run_manifest(
                repository_root=self.repository_root,
                manifest_path=path,
            )
        else:
            manifest = revalidate_formal_single_operator_run_manifest(
                repository_root=self.repository_root,
                manifest_path=path,
            )
        if (
            manifest.stage != node.stage
            or manifest.cell_id != cell.cell_id
            or manifest.role != cell.method_role
            or manifest.backend != cell.backend
            or manifest.target_model_id != cell.model
            or manifest.materialization_sha256 != materialization.sha256
            or manifest.materialization_protocol_lock_sha256
            != materialization.protocol_lock_sha256
        ):
            raise ValueError(
                "single-operator run manifest differs from the current cell"
            )
        reducer_payload = _validated_single_operator_serving_payload(
            manifest=manifest,
            cell=cell,
        )
        return FormalSingleOperatorActualValidation(
            status=manifest.completion_status,
            started_ns=manifest.started_ns,
            finished_ns=manifest.finished_ns,
            result_identity_sha256=manifest.sha256,
            reducer_payload={
                "artifacts": [row.to_dict() for row in manifest.artifacts],
                "exit_code": manifest.exit_code,
                "manifest_sha256": manifest.sha256,
                "run_directory": manifest.run_directory,
                "serving_observation": reducer_payload,
            },
        )


@dataclass(frozen=True)
class FormalSingleOperatorOnlineSpecRunManifestActualValidator:
    """Serving-manifest validator with an explicit independent-baseline identity."""

    repository_root: str

    @property
    def validator_kind(self) -> str:
        return "formal_single_operator_onlinespec_run_manifest_revalidator"

    @property
    def protocol_sha256(self) -> str:
        return _content_sha256(
            {
                "kind": self.validator_kind,
                "source_validator": FormalSingleOperatorRunManifestActualValidator(
                    self.repository_root
                ).protocol_sha256,
                "roles": (
                    "OnlineSPEC-OGD",
                    "OnlineSPEC-OPT",
                    "OnlineSPEC-ENS",
                    "OnlineSPEC-Optimistic-OGD",
                    "OnlineSPEC-Hedge",
                ),
            }
        )

    def validate(
        self,
        *,
        path: Path,
        node: FormalSingleOperatorNodeSpec,
        materialization: StageMaterializationReceipt,
        cell: MaterializedCell,
    ) -> FormalSingleOperatorActualValidation:
        if not cell.method_role.startswith("OnlineSPEC-") or node.stage != "E0":
            raise ValueError("OnlineSPEC validator cannot validate a core-method cell")
        return FormalSingleOperatorRunManifestActualValidator(
            self.repository_root
        ).validate(
            path=path,
            node=node,
            materialization=materialization,
            cell=cell,
        )


@dataclass(frozen=True)
class FormalSingleOperatorProfilerActualValidator:
    """Deep validator for one descriptive-only E4 profiler terminal."""

    @property
    def validator_kind(self) -> str:
        return "formal_single_operator_profiler_terminal_revalidator"

    @property
    def protocol_sha256(self) -> str:
        from lightcone_spec.experiments.formal_single_operator_profiler import (
            FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256,
        )

        return FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256

    def validate(
        self,
        *,
        path: Path,
        node: FormalSingleOperatorNodeSpec,
        materialization: StageMaterializationReceipt,
        cell: MaterializedCell,
    ) -> FormalSingleOperatorActualValidation:
        from lightcone_spec.experiments.formal_single_operator_profiler import (
            load_formal_single_operator_profiler_terminal,
        )

        if node.node != "e4_profiler" or cell.task != "mechanism_profile_only":
            raise ValueError("profiler validator cannot validate a headline cell")
        terminal = load_formal_single_operator_profiler_terminal(path)
        variant = dict(cell.dimensions).get("profiler")
        if (
            terminal.protocol_lock_sha256 != materialization.protocol_lock_sha256
            or terminal.materialization_sha256 != materialization.sha256
            or terminal.cell_id != cell.cell_id
            or terminal.variant != variant
            or terminal.headline_eligible is not False
        ):
            raise ValueError("profiler terminal differs from the current cell")
        return FormalSingleOperatorActualValidation(
            status=terminal.status,
            started_ns=terminal.started_ns,
            finished_ns=terminal.finished_ns,
            result_identity_sha256=terminal.sha256,
            reducer_payload={
                "descriptive_only": True,
                "exit_code": terminal.exit_code,
                "profiler_variant": terminal.variant,
                "raw_profile_sha256": terminal.raw_profile_sha256,
                "raw_profile_size_bytes": terminal.raw_profile_size_bytes,
                "terminal_sha256": terminal.sha256,
            },
        )


@dataclass(frozen=True)
class FormalSingleOperatorE5FailureActualValidator:
    """Deep validator for the trusted single-operator E5 recovery lifecycle."""

    @property
    def validator_kind(self) -> str:
        return "formal_single_operator_e5_failure_terminal_revalidator"

    @property
    def protocol_sha256(self) -> str:
        from lightcone_spec.orchestration.formal_failure_physical import (
            FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256,
        )

        return FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256

    def validate(
        self,
        *,
        path: Path,
        node: FormalSingleOperatorNodeSpec,
        materialization: StageMaterializationReceipt,
        cell: MaterializedCell,
    ) -> FormalSingleOperatorActualValidation:
        from lightcone_spec.experiments.formal_failure_actuator import (
            FormalFailureActuationReceipt,
        )
        from lightcone_spec.orchestration.formal_failure_physical import (
            FormalE5FailureLifecycleRawReceipt,
            validate_formal_single_operator_e5_physical_outcome,
        )
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            FormalServingRunPlan,
            _load_formal_single_operator_trusted_run_plan,
            load_formal_serving_run_plan,
            rebuild_formal_single_operator_execution_binding_from_plan,
        )
        from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

        if node.node != "e5_final" or cell.task != "deterministic_failure_injection":
            raise ValueError("E5 failure validator cannot validate a headline cell")
        run = FormalSingleOperatorJsonBinding.bind(
            path,
            label="single-operator E5 failure run receipt",
        ).reopen(label="single-operator E5 failure run receipt")
        if run.get("kind") != "unsigned_formal_e5_failure_physical_run_receipt":
            raise ValueError("E5 failure actual is not a physical run receipt")
        lifecycle_source = CanonicalJsonProofBinding.from_dict(
            run.get("lifecycle_receipt")
        )
        lifecycle = FormalE5FailureLifecycleRawReceipt.from_dict(
            lifecycle_source.reopen()
        )
        plan_path = lifecycle.plan.absolute_path
        outcome = validate_formal_single_operator_e5_physical_outcome(
            plan_path=plan_path,
            run_receipt_path=path,
            lifecycle_receipt_path=lifecycle_source.absolute_path,
        )
        plan_binding = CanonicalJsonProofBinding.bind(plan_path)
        preliminary = FormalServingRunPlan.from_dict(plan_binding.reopen())
        source_kind = (
            None
            if preliminary.single_operator_execution_rebuild_source is None
            else preliminary.single_operator_execution_rebuild_source.reopen().get(
                "kind"
            )
        )
        current_failure = None
        if source_kind == ("formal_single_operator_e5_failure_execution_descriptor"):
            from lightcone_spec.experiments.formal_failure_execution import (
                revalidate_formal_single_operator_e5_failure_execution_descriptor,
            )

            assert preliminary.single_operator_execution_rebuild_source is not None
            current_failure = (
                revalidate_formal_single_operator_e5_failure_execution_descriptor(
                    preliminary.single_operator_execution_rebuild_source.absolute_path,
                    current_ns=max(outcome.finished_ns, 1),
                )
            )
            plan, _launch, _schedule = _load_formal_single_operator_trusted_run_plan(
                plan_path
            )
            materialization_sha256 = (
                current_failure.failure_subject.materialization_receipt_sha256
            )
        else:
            execution_binding = (
                rebuild_formal_single_operator_execution_binding_from_plan(plan_path)
            )
            plan = load_formal_serving_run_plan(
                plan_path,
                execution_binding=execution_binding,
                verified_nextn_tp2_authority=(
                    execution_binding.verified_nextn_tp2_authority
                ),
            )
            materialization_sha256 = (
                execution_binding.subject.materialization_receipt_sha256
            )
        raw_terminal = lifecycle.raw_failure_terminal.reopen()
        recovery = FormalFailureActuationReceipt.from_dict(
            raw_terminal.get("recovery_receipt")
        )
        dimensions = dict(cell.dimensions)
        if not {"cohort_count", "failure", "topology"} <= set(dimensions):
            raise ValueError("E5 failure cell lacks its diagnostic identity")
        if (
            plan.stage != node.stage
            or plan.materialized_cell_id != cell.cell_id
            or materialization_sha256 != materialization.sha256
            or recovery.materialized_cell_id != cell.cell_id
            or recovery.correctness_only is not True
            or (
                current_failure is not None
                and (
                    recovery.scenario != current_failure.failure_subject.scenario
                    or recovery.topology != current_failure.failure_subject.topology
                )
            )
        ):
            raise ValueError("E5 failure lifecycle differs from the current cell")
        return FormalSingleOperatorActualValidation(
            status="COMPLETE",
            started_ns=outcome.started_ns,
            finished_ns=outcome.finished_ns,
            result_identity_sha256=outcome.lifecycle_receipt.semantic_sha256,
            reducer_payload={
                "backend": cell.backend,
                "cohort_count": dimensions["cohort_count"],
                "descriptive_only": True,
                "diagnostic_status": "PASS" if recovery.recovered else "FAIL",
                "failure": dimensions["failure"],
                "junit_raw_sha256": outcome.junit.raw_sha256,
                "lifecycle_receipt_sha256": (outcome.lifecycle_receipt.semantic_sha256),
                "process_exit_code": outcome.process_exit_code,
                "recovered": recovery.recovered,
                "recovery_receipt_sha256": recovery.sha256,
                "run_receipt_sha256": outcome.run_receipt.semantic_sha256,
                "topology": dimensions["topology"],
            },
        )


FORMAL_SINGLE_OPERATOR_E6_INTERFACE_PREFLIGHT_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e6_interface_preflight_protocol",
        "source": "path_bound_e6_nextn_model_authority_input",
        "validation": "deep_nextn_tp2_dynamic_authority_revalidation",
        "timing": "publisher_observed_wall_clock",
        "signed_wrapper": False,
    }
)


def _revalidate_single_operator_e6_interface_source(
    *,
    source_path: str | Path,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    verified_ns: int,
) -> tuple[FormalSingleOperatorJsonBinding, object, object]:
    from lightcone_spec.experiments.e0_authority_artifact import (
        e6_nextn_model_authority_input_from_dict,
    )
    from lightcone_spec.experiments.formal_protocol import content_sha256
    from lightcone_spec.runtime.backend import (
        validate_nextn_tp2_dynamic_authority_artifact,
    )

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E6 interface revalidation requires an exact ProtocolLock")
    if materialization.stage != "E6" or materialization.protocol_lock_sha256 != (
        protocol_lock.sha256
    ):
        raise ValueError("E6 interface materialization differs from ProtocolLock")
    if cell.task != "immutable_metadata_interface_and_fit_preflight":
        raise ValueError("E6 interface source cannot validate a serving cell")
    source_binding = FormalSingleOperatorJsonBinding.bind(
        source_path,
        label="single-operator E6 interface source input",
    )
    source = e6_nextn_model_authority_input_from_dict(
        source_binding.reopen(label="single-operator E6 interface source input")
    )
    dimensions = dict(cell.dimensions)
    inventory_sha256 = _require_sha256(
        "single-operator E6 inventory",
        dimensions.get("inventory_sha256"),
    )
    if (
        source.model != cell.model
        or source.target_member_id != dimensions.get("target_member_id")
        or source.drafter_member_id != dimensions.get("drafter_member_id")
        or source.expected_interface_sha256 != dimensions.get("interface_sha256")
        or source.expected_topology_sha256
        != dimensions.get("topology_authority_sha256")
        or source.expected_source_adapter_version
        != dimensions.get("source_adapter_version")
    ):
        raise ValueError("E6 interface source differs from the current cell")
    verified = validate_nextn_tp2_dynamic_authority_artifact(
        source.artifact_path,
        expected_inventory_sha256=inventory_sha256,
        expected_registry_sha256=protocol_lock.registry_sha256,
        expected_root_manifest_sha256=(protocol_lock.offline_release_trust_root_sha256),
        expected_interface_sha256=source.expected_interface_sha256,
        expected_topology_sha256=source.expected_topology_sha256,
        expected_source_adapter_version=source.expected_source_adapter_version,
        expected_target_member_id=source.target_member_id,
        expected_drafter_member_id=source.drafter_member_id,
        now_ns=verified_ns,
    )
    expected = {
        "content_verification_receipt_sha256": (
            verified.content_verification_receipt_sha256
        ),
        "distributed_gpu_proof_sha256": verified.distributed_gpu_proof_sha256,
        "drafter_model_id": verified.drafter_model_id,
        "drafter_revision": verified.drafter_revision,
        "drafter_shard_manifest_sha256": verified.drafter_shard_manifest_sha256,
        "e6_verified_authority_sha256": verified.sha256,
        "gpu_uuid_order_sha256": content_sha256(verified.gpu_uuids),
        "interface_sha256": verified.interface_sha256,
        "inventory_sha256": verified.inventory_sha256,
        "native_gpu_proof_sha256": verified.native_gpu_proof_sha256,
        "source_adapter_version": verified.source_adapter_version,
        "target_model_id": verified.target_model_id,
        "target_revision": verified.target_revision,
        "target_shard_manifest_sha256": verified.target_shard_manifest_sha256,
        "topology_authority_sha256": verified.topology_sha256,
    }
    if any(dimensions.get(name) != value for name, value in expected.items()):
        raise ValueError("E6 verified interface/fit differs from materialization")
    return source_binding, source, verified


@dataclass(frozen=True)
class FormalSingleOperatorE6InterfacePreflightTerminal:
    schema_version: int
    kind: Literal["formal_single_operator_e6_interface_preflight_terminal"]
    protocol_sha256: str
    protocol_lock_sha256: str
    materialization_sha256: str
    cell_id: str
    source_input: FormalSingleOperatorJsonBinding
    source_input_sha256: str
    verified_authority_sha256: str
    started_ns: int
    finished_ns: int
    status: Literal["COMPLETE"]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e6_interface_preflight_terminal"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E6_INTERFACE_PREFLIGHT_PROTOCOL_SHA256
            or self.status != "COMPLETE"
        ):
            raise ValueError("E6 interface terminal schema/status differs")
        for label, value in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("materialization", self.materialization_sha256),
            ("cell", self.cell_id),
            ("source input", self.source_input_sha256),
            ("verified authority", self.verified_authority_sha256),
        ):
            _require_sha256(f"E6 interface terminal {label}", value)
        if type(self.source_input) is not FormalSingleOperatorJsonBinding:
            raise TypeError("E6 interface terminal source is not path-bound")
        if (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 1
            or self.finished_ns <= self.started_ns
        ):
            raise ValueError("E6 interface terminal timing is invalid")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_sha256": self.materialization_sha256,
            "cell_id": self.cell_id,
            "source_input": self.source_input.to_dict(),
            "source_input_sha256": self.source_input_sha256,
            "verified_authority_sha256": self.verified_authority_sha256,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "status": self.status,
        }
        if include_sha256:
            value["terminal_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator E6 interface terminal",
            value,
            set(cls.__dataclass_fields__) | {"terminal_sha256"},
        )
        expected = _require_sha256(
            "single-operator E6 interface terminal",
            row.pop("terminal_sha256"),
        )
        row["source_input"] = FormalSingleOperatorJsonBinding.from_dict(
            row["source_input"]
        )
        terminal = cls(**row)  # type: ignore[arg-type]
        if terminal.sha256 != expected:
            raise ValueError("E6 interface terminal digest differs")
        return terminal


def publish_formal_single_operator_e6_interface_preflight_terminal(
    *,
    node_materialization_path: str | Path,
    cell_id: str,
    source_input_path: str | Path,
    output_path: str | Path,
) -> FormalSingleOperatorE6InterfacePreflightTerminal:
    """Run and time one real immutable E6 interface/fit revalidation."""

    rebuilt = rebuild_formal_single_operator_node_materialization(
        node_materialization_path
    )
    if rebuilt.artifact.node not in _E6_AUXILIARY_NODES:
        raise ValueError("E6 interface terminal requires an E6 node")
    cells = {row.cell_id: row for row in rebuilt.materialization.cells}
    cell = cells.get(_require_sha256("E6 interface terminal cell", cell_id))
    if cell is None:
        raise ValueError("E6 interface terminal cell is not materialized")
    protocol_lock = protocol_lock_from_dict(
        rebuilt.artifact.protocol_lock_source.reopen(
            label="single-operator E6 interface ProtocolLock"
        )
    )
    started_ns = time.time_ns()
    source_binding, source, verified = _revalidate_single_operator_e6_interface_source(
        source_path=source_input_path,
        protocol_lock=protocol_lock,
        materialization=rebuilt.materialization,
        cell=cell,
        verified_ns=started_ns,
    )
    finished_ns = max(time.time_ns(), started_ns + 1)
    terminal = FormalSingleOperatorE6InterfacePreflightTerminal(
        schema_version=1,
        kind="formal_single_operator_e6_interface_preflight_terminal",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_E6_INTERFACE_PREFLIGHT_PROTOCOL_SHA256),
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_sha256=rebuilt.materialization.sha256,
        cell_id=cell.cell_id,
        source_input=source_binding,
        source_input_sha256=source.sha256,
        verified_authority_sha256=verified.sha256,
        started_ns=started_ns,
        finished_ns=finished_ns,
        status="COMPLETE",
    )
    _publish_canonical_object_no_replace(
        _absolute_normalized_path("E6 interface terminal output", output_path),
        terminal.to_dict(),
    )
    return terminal


@dataclass(frozen=True)
class FormalSingleOperatorE6InterfacePreflightActualValidator:
    """Revalidate one of the two immutable NEXTN TP2 interface/fit terminals."""

    protocol_lock: ProtocolLock

    def __post_init__(self) -> None:
        if type(self.protocol_lock) is not ProtocolLock:
            raise TypeError("E6 interface validator requires an exact ProtocolLock")

    @property
    def validator_kind(self) -> str:
        return "formal_single_operator_e6_interface_preflight_revalidator"

    @property
    def protocol_sha256(self) -> str:
        return FORMAL_SINGLE_OPERATOR_E6_INTERFACE_PREFLIGHT_PROTOCOL_SHA256

    def validate(
        self,
        *,
        path: Path,
        node: FormalSingleOperatorNodeSpec,
        materialization: StageMaterializationReceipt,
        cell: MaterializedCell,
    ) -> FormalSingleOperatorActualValidation:
        if node.node not in _E6_AUXILIARY_NODES:
            raise ValueError("E6 interface validator requires an E6 node")
        terminal_source = FormalSingleOperatorJsonBinding.bind(
            path,
            label="single-operator E6 interface terminal",
        )
        terminal_value = terminal_source.reopen(
            label="single-operator E6 interface terminal"
        )
        if terminal_value.get("kind") == (
            "formal_single_operator_e6_interface_fit_terminal"
        ):
            from lightcone_spec.experiments.formal_protocol import content_sha256
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                compatibility_row_for_terminal,
                revalidate_formal_single_operator_e6_interface_fit_plan,
                revalidate_formal_single_operator_e6_interface_fit_terminal,
            )

            terminal = revalidate_formal_single_operator_e6_interface_fit_terminal(path)
            plan = revalidate_formal_single_operator_e6_interface_fit_plan(
                terminal.plan.absolute_path
            )
            compatibility = compatibility_row_for_terminal(plan, terminal)
            dimensions = dict(cell.dimensions)
            expected = {
                "content_verification_receipt_sha256": (
                    compatibility.content_verification_receipt_sha256
                ),
                "distributed_gpu_proof_sha256": (
                    compatibility.distributed_gpu_proof_sha256
                ),
                "drafter_member_id": compatibility.drafter_member_id,
                "drafter_model_id": compatibility.drafter_model_id,
                "drafter_revision": compatibility.drafter_revision,
                "drafter_shard_manifest_sha256": (
                    compatibility.drafter_shard_manifest_sha256
                ),
                "e6_model_compatibility_row_sha256": compatibility.sha256,
                "e6_verified_authority_sha256": (
                    compatibility.verified_authority_sha256
                ),
                "gpu_uuid_order_sha256": content_sha256(compatibility.gpu_uuids),
                "interface_sha256": compatibility.interface_sha256,
                "inventory_sha256": compatibility.inventory_sha256,
                "native_gpu_proof_sha256": (compatibility.native_gpu_proof_sha256),
                "source_adapter_version": compatibility.source_adapter_version,
                "target_member_id": compatibility.target_member_id,
                "target_model_id": compatibility.target_model_id,
                "target_revision": compatibility.target_revision,
                "target_shard_manifest_sha256": (
                    compatibility.target_shard_manifest_sha256
                ),
                "topology_authority_sha256": compatibility.topology_sha256,
            }
            if (
                materialization.stage != "E6"
                or materialization.protocol_lock_sha256 != self.protocol_lock.sha256
                or cell.task != "immutable_metadata_interface_and_fit_preflight"
                or cell.model != terminal.model
                or any(
                    dimensions.get(name) != value for name, value in expected.items()
                )
                or terminal_source.semantic_sha256 != terminal.sha256
            ):
                raise ValueError(
                    "trusted E6 interface terminal differs from the current cell"
                )
            return FormalSingleOperatorActualValidation(
                status=terminal.status,
                started_ns=terminal.started_ns,
                finished_ns=terminal.finished_ns,
                result_identity_sha256=terminal.sha256,
                reducer_payload={
                    "artifact_sha256": terminal.live_observation.semantic_sha256,
                    "descriptive_only": True,
                    "distributed_gpu_proof_sha256": (
                        terminal.distributed_gpu_proof_sha256
                    ),
                    "interface_sha256": plan.interface_sha256,
                    "native_gpu_proof_sha256": terminal.native_gpu_proof_sha256,
                    "source_input_sha256": plan.sha256,
                    "terminal_sha256": terminal.sha256,
                    "trust_mode": ("trusted_single_operator_empirical_no_signature"),
                    "verified_authority_sha256": (terminal.trusted_authority_sha256),
                },
            )
        terminal = FormalSingleOperatorE6InterfacePreflightTerminal.from_dict(
            terminal_value
        )
        _source_binding, source, verified = (
            _revalidate_single_operator_e6_interface_source(
                source_path=terminal.source_input.absolute_path,
                protocol_lock=self.protocol_lock,
                materialization=materialization,
                cell=cell,
                verified_ns=terminal.finished_ns,
            )
        )
        if (
            terminal.protocol_lock_sha256 != self.protocol_lock.sha256
            or terminal.materialization_sha256 != materialization.sha256
            or terminal.cell_id != cell.cell_id
            or terminal.source_input_sha256 != source.sha256
            or terminal.verified_authority_sha256 != verified.sha256
        ):
            raise ValueError("E6 interface terminal differs from the current cell")
        return FormalSingleOperatorActualValidation(
            status=terminal.status,
            started_ns=terminal.started_ns,
            finished_ns=terminal.finished_ns,
            result_identity_sha256=terminal.sha256,
            reducer_payload={
                "artifact_sha256": verified.artifact_sha256,
                "distributed_gpu_proof_sha256": (verified.distributed_gpu_proof_sha256),
                "interface_sha256": verified.interface_sha256,
                "native_gpu_proof_sha256": verified.native_gpu_proof_sha256,
                "source_input_sha256": source.sha256,
                "terminal_sha256": terminal.sha256,
                "verified_authority_sha256": verified.sha256,
            },
        )


FORMAL_SINGLE_OPERATOR_E0_COMPATIBILITY_VALIDATOR_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_compatibility_validator",
        "source": "exact_materialization_e0_compatibility_auxiliary",
        "coverage": "one_of_108_typed_model_backend_task_decisions",
        "timing": "bundle_recorded_real_probe_boundaries",
        "result": "descriptive_valid_or_na_without_confidence_interval",
    }
)


@dataclass(frozen=True)
class FormalSingleOperatorE0CompatibilityActualValidator:
    """Deep-select one compatibility decision from the shared E0 source."""

    protocol_lock: ProtocolLock
    predecessor: RebuiltFormalSingleOperatorStageCompletion
    compatibility_source: FormalSingleOperatorJsonBinding

    def __post_init__(self) -> None:
        if type(self.protocol_lock) is not ProtocolLock:
            raise TypeError("E0 compatibility validator requires ProtocolLock")
        if (
            type(self.predecessor) is not RebuiltFormalSingleOperatorStageCompletion
            or self.predecessor.artifact.node != "e6_final"
        ):
            raise TypeError("E0 compatibility validator requires E6 completion")
        if type(self.compatibility_source) is not FormalSingleOperatorJsonBinding:
            raise TypeError("E0 compatibility validator source is not bound")
        self.compatibility_source.reopen(
            label="single-operator E0 compatibility validator source"
        )

    @property
    def validator_kind(self) -> str:
        return "formal_single_operator_e0_compatibility_revalidator"

    @property
    def protocol_sha256(self) -> str:
        return FORMAL_SINGLE_OPERATOR_E0_COMPATIBILITY_VALIDATOR_PROTOCOL_SHA256

    def validate(
        self,
        *,
        path: Path,
        node: FormalSingleOperatorNodeSpec,
        materialization: StageMaterializationReceipt,
        cell: MaterializedCell,
    ) -> FormalSingleOperatorActualValidation:
        from lightcone_spec.experiments.formal_single_operator_downstream import (
            _e0_compatibility_from_auxiliary,
        )

        if (
            node.node != "e0_tuning"
            or cell.task != "compatibility_decision"
            or cell.method_role != "Compatibility"
        ):
            raise ValueError(
                "E0 compatibility validator cannot validate a serving cell"
            )
        source = FormalSingleOperatorJsonBinding.bind(
            path,
            label="single-operator E0 compatibility result",
        )
        if source != self.compatibility_source:
            raise ValueError("E0 compatibility result differs from materialization")
        value = source.reopen(label="single-operator E0 compatibility result")
        compatibility, _authority, bundle_sha256, evidence_sha256 = (
            _e0_compatibility_from_auxiliary(
                self.predecessor,
                self.protocol_lock,
                value,
            )
        )
        if (
            materialization.protocol_lock_sha256 != self.protocol_lock.sha256
            or materialization.source_decision_sha256 != bundle_sha256
        ):
            raise ValueError("E0 compatibility materialization lineage differs")
        dimensions = dict(cell.dimensions)
        expected_keys = {
            "compatibility_decision_id",
            "compatibility_evidence_manifest_sha256",
            "compatibility_receipt_sha256",
            "deployment_task",
            "disposition",
            "e0_compatibility_bundle_sha256",
            "interface_sha256",
            "reason_code",
            "task_native_workload_sha256",
        }
        if set(dimensions) != expected_keys:
            raise ValueError("E0 compatibility cell dimensions differ")
        decisions = {
            decision.decision_id: decision for decision in compatibility.decisions
        }
        decision = decisions.get(dimensions["compatibility_decision_id"])
        if (
            decision is None
            or cell.model != decision.model
            or cell.backend != decision.backend
            or dimensions["deployment_task"] != decision.task
            or dimensions["disposition"] != decision.disposition
            or dimensions["reason_code"] != decision.reason_code
            or dimensions["interface_sha256"] != decision.interface_sha256
            or dimensions["task_native_workload_sha256"]
            != decision.task_native_workload_sha256
            or dimensions["compatibility_receipt_sha256"] != compatibility.sha256
            or dimensions["compatibility_evidence_manifest_sha256"] != evidence_sha256
            or dimensions["e0_compatibility_bundle_sha256"] != bundle_sha256
        ):
            raise ValueError("E0 compatibility result row differs from cell")
        started_ns = value.get("started_ns")
        finished_ns = value.get("finished_ns")
        if (
            type(started_ns) is not int
            or type(finished_ns) is not int
            or started_ns < 1
            or finished_ns <= started_ns
        ):
            raise ValueError("E0 compatibility result timing differs")
        decision_payload = {
            "schema_version": 1,
            "compatibility_decision_id": decision.decision_id,
            "disposition": decision.disposition,
            "reason_code": decision.reason_code,
            "interface_sha256": decision.interface_sha256,
            "task_native_workload_sha256": (decision.task_native_workload_sha256),
            "compatibility_evidence_manifest_sha256": evidence_sha256,
        }
        return FormalSingleOperatorActualValidation(
            status="COMPLETE",
            started_ns=started_ns,
            finished_ns=finished_ns,
            result_identity_sha256=_content_sha256(
                {
                    "e0_compatibility_bundle_sha256": bundle_sha256,
                    "compatibility_decision_id": decision.decision_id,
                }
            ),
            reducer_payload={
                "e0_compatibility_decision": decision_payload,
                "compatibility_receipt_sha256": compatibility.sha256,
                "decision_id": decision.decision_id,
                "descriptive_only": True,
                "disposition": decision.disposition,
                "evidence_manifest_sha256": evidence_sha256,
                "interface_sha256": decision.interface_sha256,
                "reason_code": decision.reason_code,
                "task_native_workload_sha256": (decision.task_native_workload_sha256),
            },
        )


@dataclass(frozen=True)
class FormalSingleOperatorValidatedActual:
    node: FormalSingleOperatorNode
    stage: str
    materialization_sha256: str
    cell_id: str
    status: Literal["COMPLETE", "FAILED"]
    started_ns: int
    finished_ns: int
    result_identity_sha256: str
    validator_kind: str
    validator_protocol_sha256: str
    source: FormalSingleOperatorJsonBinding
    reducer_payload: dict[str, object]

    def __post_init__(self) -> None:
        spec = formal_single_operator_node_spec(self.node)
        if self.stage != spec.stage:
            raise ValueError("single-operator actual stage differs from its node")
        for label, value in (
            ("materialization", self.materialization_sha256),
            ("result identity", self.result_identity_sha256),
            ("validator protocol", self.validator_protocol_sha256),
        ):
            _require_sha256(f"single-operator actual {label}", value)
        _require_sha256("single-operator actual cell", self.cell_id)
        _require_text("single-operator actual validator", self.validator_kind)
        if type(self.source) is not FormalSingleOperatorJsonBinding:
            raise TypeError("single-operator actual requires a JSON source binding")
        validation = FormalSingleOperatorActualValidation(
            status=self.status,
            started_ns=self.started_ns,
            finished_ns=self.finished_ns,
            result_identity_sha256=self.result_identity_sha256,
            reducer_payload=self.reducer_payload,
        )
        object.__setattr__(self, "reducer_payload", validation.reducer_payload)

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "node": self.node,
            "stage": self.stage,
            "materialization_sha256": self.materialization_sha256,
            "cell_id": self.cell_id,
            "status": self.status,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "result_identity_sha256": self.result_identity_sha256,
            "validator_kind": self.validator_kind,
            "validator_protocol_sha256": self.validator_protocol_sha256,
            "source": self.source.to_dict(),
            "reducer_payload": self.reducer_payload,
        }
        if include_sha256:
            value["actual_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator validated actual",
            value,
            {
                "node",
                "stage",
                "materialization_sha256",
                "cell_id",
                "status",
                "started_ns",
                "finished_ns",
                "result_identity_sha256",
                "validator_kind",
                "validator_protocol_sha256",
                "source",
                "reducer_payload",
                "actual_sha256",
            },
        )
        expected = _require_sha256(
            "single-operator validated actual",
            row.pop("actual_sha256"),
        )
        row["source"] = FormalSingleOperatorJsonBinding.from_dict(row["source"])
        actual = cls(**row)  # type: ignore[arg-type]
        if actual.sha256 != expected:
            raise ValueError("single-operator validated actual digest differs")
        return actual


@dataclass(frozen=True)
class FormalSingleOperatorDecisionDraft:
    decision_kind: str
    next_materialization_source_decision_sha256: str | None
    next_materialization_upstream_receipt_sha256s: tuple[str, ...]
    payload: dict[str, object]

    def __post_init__(self) -> None:
        _require_text("single-operator decision kind", self.decision_kind)
        _reject_future_or_placeholder_authority(
            self.decision_kind,
            label="single-operator decision kind",
        )
        if self.next_materialization_source_decision_sha256 is not None:
            _require_sha256(
                "single-operator next materialization source decision",
                self.next_materialization_source_decision_sha256,
            )
        if type(self.next_materialization_upstream_receipt_sha256s) is not tuple or len(
            set(self.next_materialization_upstream_receipt_sha256s)
        ) != len(self.next_materialization_upstream_receipt_sha256s):
            raise ValueError(
                "single-operator next materialization upstreams must be distinct"
            )
        for digest in self.next_materialization_upstream_receipt_sha256s:
            _require_sha256("single-operator next materialization upstream", digest)
        object.__setattr__(
            self,
            "payload",
            _canonical_object("single-operator decision payload", self.payload),
        )


class FormalSingleOperatorMaterializeAdapter(Protocol):
    def __call__(
        self,
        predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
        /,
    ) -> StageMaterializationReceipt: ...


class FormalSingleOperatorReduceAdapter(Protocol):
    def __call__(
        self,
        predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
        materialization: StageMaterializationReceipt,
        actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
        /,
    ) -> FormalSingleOperatorDecisionDraft: ...


class _FormalSingleOperatorClosedMaterializer(Protocol):
    def __call__(
        self,
        predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
        protocol_lock: ProtocolLock,
        auxiliary_sources: tuple[FormalSingleOperatorAuxiliarySourceBinding, ...],
        /,
    ) -> StageMaterializationReceipt: ...


class _FormalSingleOperatorMaterializerWithoutAuxiliarySources(Protocol):
    def __call__(
        self,
        predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
        protocol_lock: ProtocolLock,
        /,
    ) -> StageMaterializationReceipt: ...


@dataclass(frozen=True)
class _RejectingAuxiliaryMaterializerAdapter:
    """Make an early materializer's lack of auxiliary inputs explicit."""

    delegate: _FormalSingleOperatorMaterializerWithoutAuxiliarySources

    def __call__(
        self,
        predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
        protocol_lock: ProtocolLock,
        auxiliary_sources: tuple[FormalSingleOperatorAuxiliarySourceBinding, ...],
        /,
    ) -> StageMaterializationReceipt:
        if auxiliary_sources:
            raise ValueError("this single-operator materializer accepts no auxiliaries")
        return self.delegate(predecessor, protocol_lock)


def _without_auxiliary_sources(
    materializer: _FormalSingleOperatorMaterializerWithoutAuxiliarySources,
) -> _FormalSingleOperatorClosedMaterializer:
    return _RejectingAuxiliaryMaterializerAdapter(materializer)


@dataclass(frozen=True)
class _FormalSingleOperatorClosedNodeAdapter:
    materializer: _FormalSingleOperatorClosedMaterializer | None
    actual_validator_kind: FormalSingleOperatorActualValidatorKind | None
    reducer: FormalSingleOperatorReduceAdapter | None
    blocked_reason: str | None

    def __post_init__(self) -> None:
        if self.blocked_reason is not None:
            _require_text("single-operator adapter blocked reason", self.blocked_reason)


def _materialize_single_operator_preflight(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is not None:
        raise ValueError("preflight materializer cannot receive a predecessor")
    return materialize_preflight(
        protocol_lock_sha256=protocol_lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _reduce_single_operator_preflight(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    if predecessor is not None or materialization.stage != "preflight":
        raise ValueError("preflight reducer received another DAG node")
    completion_ids = {
        row.reducer_payload.get("preflight_completion_sha256") for row in actual_results
    }
    trusted_completion = None not in completion_ids
    evidence_ids = (
        completion_ids
        if trusted_completion
        else {row.reducer_payload["final_evidence_sha256"] for row in actual_results}
    )
    coverage_ids = (
        completion_ids
        if trusted_completion
        else {row.reducer_payload["stage_coverage_sha256"] for row in actual_results}
    )
    e3a_authorities = {
        row.reducer_payload["e3a_workload_authority_sha256"] for row in actual_results
    }
    if len(evidence_ids) != 1 or len(coverage_ids) != 1 or len(e3a_authorities) != 1:
        raise ValueError("preflight cell results do not share one final authority")
    evidence_sha256 = next(iter(evidence_ids))
    coverage_sha256 = next(iter(coverage_ids))
    e3a_authority_sha256 = next(iter(e3a_authorities))
    _require_sha256("single-operator preflight evidence", evidence_sha256)
    _require_sha256("single-operator preflight coverage", coverage_sha256)
    _require_sha256("single-operator E3a workload authority", e3a_authority_sha256)
    return FormalSingleOperatorDecisionDraft(
        decision_kind="preflight_all_complete",
        next_materialization_source_decision_sha256=e3a_authority_sha256,
        next_materialization_upstream_receipt_sha256s=(coverage_sha256,),
        payload={
            "actual_cell_count": len(actual_results),
            "final_evidence_sha256": evidence_sha256,
            "stage_coverage_sha256": coverage_sha256,
            "single_operator_completion": trusted_completion,
        },
    )


def _materialize_single_operator_e3a(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != "preflight":
        raise ValueError("E3a requires exact completed preflight")
    from lightcone_spec.experiments.stage_materialization import (
        _materialize_e3a_diagnostic,
    )

    decision = predecessor.decision
    if protocol_lock.schema_version == 4:
        workload_authority_sha256 = _require_sha256(
            "legacy E3a workload authorization",
            protocol_lock.formal_workload_e3a_authorization_sha256,
        )
    else:
        workload_authority_sha256 = _require_sha256(
            "trusted E3a workload authority",
            decision.next_materialization_source_decision_sha256,
        )
        validated_authorities = {
            row.reducer_payload.get("e3a_workload_authority_sha256")
            for row in predecessor.artifact.actual_results
        }
        if validated_authorities != {workload_authority_sha256}:
            raise ValueError(
                "trusted E3a predecessor actuals do not share the locked workload"
            )
    if (
        decision.next_materialization_source_decision_sha256
        != workload_authority_sha256
        or len(decision.next_materialization_upstream_receipt_sha256s) != 1
    ):
        raise ValueError("E3a predecessor did not authorize the exact workload")
    return _materialize_e3a_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_preflight_receipt_sha256=(
            decision.next_materialization_upstream_receipt_sha256s[0]
        ),
        workload_authority_sha256=(workload_authority_sha256),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _lock_from_completion(
    completion: RebuiltFormalSingleOperatorStageCompletion,
) -> ProtocolLock:
    lock = protocol_lock_from_dict(
        completion.node_materialization.protocol_lock_source.reopen(
            label="single-operator reducer ProtocolLock"
        )
    )
    if lock.sha256 != completion.artifact.protocol_lock_sha256:
        raise ValueError("single-operator reducer ProtocolLock differs")
    return lock


def _serving_observation(
    actual: FormalSingleOperatorValidatedActual,
    cell: MaterializedCell,
) -> dict[str, object]:
    payload = _strict(
        "single-operator serving reducer payload",
        actual.reducer_payload,
        {
            "artifacts",
            "exit_code",
            "manifest_sha256",
            "run_directory",
            "serving_observation",
        },
    )
    observation = _strict(
        "single-operator serving observation",
        payload["serving_observation"],
        {
            "schema_version",
            "kind",
            "materialized_cell_id",
            "inventory_sha256",
            "run_id",
            "run_nonce_sha256",
            "attempt_id",
            "method",
            "terminal_sha256",
            "terminal_artifact_sha256",
            "native_itl_artifact_sha256",
            "request_schedule_sha256",
            "source_request_pool_sha256",
            "serving_execution_policy_sha256",
            "client_lifecycle_artifact_sha256",
            "scored_phase_origin_ns",
            "scheduled_request_count",
            "offered_request_count",
            "outcome_counts",
            "requests",
            "performance_counters",
        },
    )
    if (
        observation["schema_version"] != 2
        or observation["kind"] != "formal_single_operator_serving_observation"
        or observation["materialized_cell_id"] != cell.cell_id
        or observation["method"] != _single_operator_runtime_method(cell)
    ):
        raise ValueError("single-operator serving observation differs from cell")
    for label in (
        "inventory_sha256",
        "run_nonce_sha256",
        "terminal_sha256",
        "terminal_artifact_sha256",
        "native_itl_artifact_sha256",
        "request_schedule_sha256",
        "source_request_pool_sha256",
        "serving_execution_policy_sha256",
        "client_lifecycle_artifact_sha256",
    ):
        _require_sha256(f"single-operator serving {label}", observation[label])
    _require_text("single-operator serving run", observation["run_id"])
    _require_text("single-operator serving attempt", observation["attempt_id"])
    if type(observation["performance_counters"]) is not dict:
        raise TypeError("single-operator serving counters must be an object")
    if (
        type(observation["scored_phase_origin_ns"]) is not int
        or observation["scored_phase_origin_ns"] < 1
        or type(observation["scheduled_request_count"]) is not int
        or type(observation["offered_request_count"]) is not int
        or observation["scheduled_request_count"] < observation["offered_request_count"]
        or observation["offered_request_count"] < 1
        or type(observation["outcome_counts"]) is not dict
        or set(observation["outcome_counts"])
        != {"completed", "rejected", "timed_out", "cancelled", "unfinished"}
        or any(
            type(value) is not int or value < 0
            for value in observation["outcome_counts"].values()
        )
        or sum(observation["outcome_counts"].values())
        != observation["offered_request_count"]
    ):
        raise ValueError("single-operator serving denominator accounting differs")
    requests = _array(
        "single-operator serving requests",
        observation["requests"],
    )
    if not requests:
        raise ValueError("single-operator serving observation has no requests")
    seen: set[str] = set()
    previous = None
    for request in requests:
        row = _strict(
            "single-operator serving request",
            request,
            {
                "request_id",
                "input_token_ids",
                "output_token_ids",
                "request_started_ns",
                "request_terminal_ns",
                "token_observed_ns",
                "terminal_status",
                "terminal_reason",
                "submitted_to_server",
                "native_terminal_status",
                "offered_at_us",
                "admitted_at_us",
                "effective_deadline_us",
                "terminal_at_us",
            },
        )
        request_id = _require_text(
            "single-operator serving request ID", row["request_id"]
        )
        if request_id in seen or (previous is not None and request_id <= previous):
            raise ValueError("single-operator serving requests are not canonical")
        seen.add(request_id)
        previous = request_id
        for name in ("input_token_ids", "output_token_ids", "token_observed_ns"):
            if type(row[name]) is not list:
                raise TypeError(f"single-operator serving {name} must be an array")
        if row["terminal_status"] not in {
            "completed",
            "rejected",
            "timed_out",
            "cancelled",
            "unfinished",
        }:
            raise ValueError("single-operator serving client status differs")
    return observation


def _single_operator_request_rows(
    observation: dict[str, object],
) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(row)
        for row in _array(
            "single-operator serving requests",
            observation["requests"],
        )
        if type(row) is dict
    )


def _single_operator_request_identity(
    observation: dict[str, object],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row["request_id"],
            tuple(row["input_token_ids"]),  # type: ignore[arg-type]
        )
        for row in _single_operator_request_rows(observation)
    )


def _counter(
    observation: dict[str, object],
    name: str,
    *,
    nullable: bool = False,
) -> int | None:
    counters = observation["performance_counters"]
    assert isinstance(counters, dict)
    if name not in counters:
        raise ValueError(f"single-operator performance counter {name} is missing")
    value = counters[name]
    if nullable and value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"single-operator performance counter {name} is unavailable")
    return value


def _single_operator_throughput(
    observation: dict[str, object],
) -> tuple[int, int]:
    rows = _single_operator_request_rows(observation)
    if any(
        row["terminal_status"] != "completed"
        or row["submitted_to_server"] is not True
        or not row["output_token_ids"]
        for row in rows
    ):
        raise ValueError("single-operator throughput contains an incomplete request")
    tokens = sum(len(row["output_token_ids"]) for row in rows)  # type: ignore[arg-type]
    window = max(int(row["request_terminal_ns"]) for row in rows) - min(
        int(row["request_started_ns"]) for row in rows
    )
    if tokens < 1 or window < 1:
        raise ValueError("single-operator throughput is not positive")
    return tokens, window


def _single_operator_e3a_lambda_star_request_rate(
    *,
    cell: MaterializedCell,
    observation: dict[str, object],
    matched_width: int,
    common_load: int,
) -> dict[str, object]:
    """Lock λ* from completed requests in the registered Static anchor."""

    dimensions = dict(cell.dimensions)
    if (
        cell.stage != "E3a"
        or cell.method_role != "Static"
        or dimensions.get("context") != 40_928
        or dimensions.get("regime") != "short_input_long_generation"
        or dimensions.get("width") != matched_width
        or dimensions.get("concurrency") != common_load
    ):
        raise ValueError("E3a lambda-star source is not the registered Static row")
    _tokens, window_ns = _single_operator_throughput(observation)
    request_count = len(_single_operator_request_rows(observation))
    if request_count < 1:
        raise ValueError("E3a lambda-star source has no completed requests")
    return {
        "numerator_requests_x_1e9": request_count * 1_000_000_000,
        "denominator_window_ns": window_ns,
        "source_cell_id": cell.cell_id,
        "source_observation_sha256": _content_sha256(observation),
        "rule": (
            "completed_requests_per_observed_window_at_static_40928_"
            "short_input_long_generation_matched_width_common_load"
        ),
    }


def _reduce_single_operator_e3a(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    if predecessor is None or predecessor.artifact.node != "preflight":
        raise ValueError("E3a reducer requires exact completed preflight")
    if materialization.stage != "E3a" or len(materialization.cells) != 360:
        raise ValueError("E3a reducer requires the exact 360-cell materialization")
    from lightcone_spec.experiments.registry import (
        CONTEXT_REGIMES,
        DRAFT_WIDTHS,
        E3A_CONCURRENCY_GRID,
        LONG_CONTEXT_ANCHORS,
    )

    lock = _lock_from_completion(predecessor)
    cells = {row.cell_id: row for row in materialization.cells}
    observations = {
        row.cell_id: _serving_observation(row, cells[row.cell_id])
        for row in actual_results
    }
    inventories = {row["inventory_sha256"] for row in observations.values()}
    if len(inventories) != 1:
        raise ValueError("E3a actual results do not share one inventory")
    safety = (
        "communicator_failures",
        "exactness_violations",
        "fallbacks",
        "nonfinite_updates",
        "oom_events",
        "retractions",
        "version_mismatches",
    )
    targets: dict[tuple[int, str, int], tuple[MaterializedCell, dict[str, object]]] = {}
    statics: dict[
        tuple[int, str, int, int], tuple[MaterializedCell, dict[str, object]]
    ] = {}
    rows: list[dict[str, object]] = []
    models = {cell.model for cell in cells.values()}
    for cell_id in sorted(cells):
        cell = cells[cell_id]
        observation = observations[cell_id]
        dimensions = dict(cell.dimensions)
        context = dimensions.get("context")
        regime = dimensions.get("regime")
        concurrency = dimensions.get("concurrency")
        width = dimensions.get("width")
        if (
            type(context) is not int
            or type(regime) is not str
            or type(concurrency) is not int
        ):
            raise ValueError("E3a cell lacks exact capacity axes")
        if cell.method_role == "Target-only":
            if any(
                _counter(observation, name, nullable=True) is not None
                for name in safety
            ):
                raise ValueError("E3a Target-only carries adaptation counters")
            if any(
                _counter(observation, name, nullable=True) is not None
                for name in ("updates_launched", "updates_published")
            ):
                raise ValueError("E3a Target-only launched an update")
            key = (context, regime, concurrency)
            if width is not None or key in targets:
                raise ValueError("E3a Target-only capacity grid differs")
            targets[key] = (cell, observation)
        elif cell.method_role == "Static":
            if any(_counter(observation, name) != 0 for name in safety):
                raise ValueError("E3a Static has a safety violation")
            if any(
                _counter(observation, name) != 0
                for name in ("updates_launched", "updates_published")
            ):
                raise ValueError("E3a Static unexpectedly launched an update")
            if type(width) is not int:
                raise ValueError("E3a Static lacks width")
            key = (context, regime, concurrency, width)
            if key in statics:
                raise ValueError("E3a Static capacity grid repeats a cell")
            statics[key] = (cell, observation)
        else:
            raise ValueError("E3a accepts only Target-only and Static")
    if len(models) != 1 or len(targets) != 96 or len(statics) != 264:
        raise ValueError("E3a role/axis cardinality differs")
    for context in LONG_CONTEXT_ANCHORS:
        for regime in CONTEXT_REGIMES:
            for concurrency in E3A_CONCURRENCY_GRID:
                target = targets.get((context, regime, concurrency))
                if target is None:
                    raise ValueError("E3a target grid is incomplete")
                target_tokens, target_window = _single_operator_throughput(target[1])
                for width in DRAFT_WIDTHS:
                    static = statics.get((context, regime, concurrency, width))
                    if static is None:
                        raise ValueError("E3a static grid is incomplete")
                    if _single_operator_request_identity(static[1]) != (
                        _single_operator_request_identity(target[1])
                    ):
                        raise ValueError(
                            "E3a Static/Target request trajectories differ"
                        )
                    tokens, window = _single_operator_throughput(static[1])
                    ratio = Fraction(tokens * target_window, window * target_tokens)
                    row = {
                        "cell_id": static[0].cell_id,
                        "context": context,
                        "regime": regime,
                        "concurrency": concurrency,
                        "width": width,
                        "throughput_tokens": tokens,
                        "throughput_window_ns": window,
                        "peak_hbm_bytes": _counter(static[1], "peak_hbm_bytes"),
                        "target_cell_id": target[0].cell_id,
                        "static_target_ratio": [ratio.numerator, ratio.denominator],
                    }
                    row["observation_sha256"] = _content_sha256(row)
                    rows.append(row)
    median_by_width_load: dict[tuple[int, int], Fraction] = {}
    for width in DRAFT_WIDTHS:
        for concurrency in E3A_CONCURRENCY_GRID:
            values = tuple(
                Fraction(
                    int(row["throughput_tokens"]) * 1_000_000_000,
                    int(row["throughput_window_ns"]),
                )
                for row in rows
                if row["width"] == width and row["concurrency"] == concurrency
            )
            if len(values) != len(LONG_CONTEXT_ANCHORS) * len(CONTEXT_REGIMES):
                raise ValueError("E3a width/load median lacks exact contexts")
            median_by_width_load[(width, concurrency)] = statistics.median(values)
    best_by_load = {
        load: max(median_by_width_load[(width, load)] for width in DRAFT_WIDTHS)
        for load in E3A_CONCURRENCY_GRID
    }
    threshold = Fraction(9, 10) * max(best_by_load.values())
    common_load = min(
        load for load, value in best_by_load.items() if value >= threshold
    )
    width_scores = []
    for width in DRAFT_WIDTHS:
        selected = tuple(
            row
            for row in rows
            if row["width"] == width and row["concurrency"] == common_load
        )
        ratios = tuple(Fraction(*row["static_target_ratio"]) for row in selected)  # type: ignore[arg-type]
        rates = tuple(
            Fraction(
                int(row["throughput_tokens"]) * 1_000_000_000,
                int(row["throughput_window_ns"]),
            )
            for row in selected
        )
        width_scores.append((min(ratios), statistics.median(rates), width))
    _worst, _median, matched_width = max(
        width_scores, key=lambda row: (row[0], row[1], -row[2])
    )
    lambda_star_source = statics.get(
        (
            40_928,
            "short_input_long_generation",
            common_load,
            matched_width,
        )
    )
    if lambda_star_source is None:
        raise ValueError("E3a lambda-star Static source row is missing")
    lambda_star_request_rate = _single_operator_e3a_lambda_star_request_rate(
        cell=lambda_star_source[0],
        observation=lambda_star_source[1],
        matched_width=matched_width,
        common_load=common_load,
    )
    capacity_sha256 = _content_sha256(
        {
            "kind": "single_operator_e3a_capacity",
            "observations": [row["observation_sha256"] for row in rows],
        }
    )
    locked_outputs = {
        "baseline_capacity_envelope": capacity_sha256,
        "e1_reference_load": _content_sha256(
            {
                "kind": "single_operator_e3a_reference_load",
                "common_load": common_load,
                "capacity_sha256": capacity_sha256,
            }
        ),
        "matched_width": _content_sha256(
            {
                "kind": "single_operator_e3a_matched_width",
                "matched_width": matched_width,
                "common_load": common_load,
                "capacity_sha256": capacity_sha256,
            }
        ),
        "width_selection_rule": _content_sha256(
            {"rule": "max_worst_ratio_then_median_throughput_then_min_width"}
        ),
        "static_target_crossover": _content_sha256(
            {
                "kind": "single_operator_e3a_crossover",
                "rows": [
                    {
                        "context": context,
                        "regime": regime,
                        "width": width,
                        "first_lte_one": next(
                            (
                                int(row["concurrency"])
                                for row in sorted(
                                    rows, key=lambda item: int(item["concurrency"])
                                )
                                if row["context"] == context
                                and row["regime"] == regime
                                and row["width"] == width
                                and Fraction(*row["static_target_ratio"]) <= 1  # type: ignore[arg-type]
                            ),
                            None,
                        ),
                    }
                    for context in LONG_CONTEXT_ANCHORS
                    for regime in CONTEXT_REGIMES
                    for width in DRAFT_WIDTHS
                ],
                "capacity_sha256": capacity_sha256,
            }
        ),
        "drift_witness": _content_sha256(
            {
                "kind": "single_operator_e3a_drift_witness",
                "matched_width": matched_width,
                "common_load": common_load,
                "observations": [
                    row["observation_sha256"]
                    for row in rows
                    if row["width"] == matched_width
                    and row["concurrency"] == common_load
                ],
            }
        ),
        "lambda_star_request_rate": _content_sha256(lambda_star_request_rate),
    }
    selection = {
        "schema_version": 1,
        "kind": "formal_single_operator_e3a_selection",
        "model": next(iter(models)),
        "matched_width": matched_width,
        "common_load": common_load,
        "inventory_sha256": next(iter(inventories)),
        "capacity_sha256": capacity_sha256,
        "lambda_star_request_rate": lambda_star_request_rate,
        "locked_outputs": locked_outputs,
    }
    selection_sha256 = _content_sha256(selection)
    return FormalSingleOperatorDecisionDraft(
        decision_kind="e3a_actual_360_reduced",
        next_materialization_source_decision_sha256=(
            lock.tts_calibration_authority_sha256
        ),
        next_materialization_upstream_receipt_sha256s=(selection_sha256,),
        payload={**selection, "selection_sha256": selection_sha256},
    )


def _materialize_single_operator_tts_calibration(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != "e3a":
        raise ValueError("TTS-Cal requires exact completed E3a")
    from lightcone_spec.experiments.stage_materialization import (
        _materialize_tts_calibration_diagnostic,
    )

    decision = predecessor.decision
    if (
        decision.next_materialization_source_decision_sha256
        != protocol_lock.tts_calibration_authority_sha256
        or len(decision.next_materialization_upstream_receipt_sha256s) != 1
        or decision.payload.get("selection_sha256")
        != decision.next_materialization_upstream_receipt_sha256s[0]
    ):
        raise ValueError("TTS-Cal predecessor lacks exact E3a selection")
    return _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_e3a_receipt_sha256=(
            decision.next_materialization_upstream_receipt_sha256s[0]
        ),
        calibration_authority_sha256=(protocol_lock.tts_calibration_authority_sha256),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _single_operator_slo_goodput(
    observation: dict[str, object],
) -> object:
    from lightcone_spec.experiments.formal_slo_metrics import (
        FormalSloRequestEvidence,
        reduce_formal_slo_goodput,
    )

    requests = []
    for row in _single_operator_request_rows(observation):
        status = str(row["terminal_status"])
        requests.append(
            FormalSloRequestEvidence(
                request_id=str(row["request_id"]),
                input_token_ids=tuple(row["input_token_ids"]),  # type: ignore[arg-type]
                output_token_ids=tuple(row["output_token_ids"]),  # type: ignore[arg-type]
                request_started_ns=int(row["request_started_ns"]),
                request_terminal_ns=int(row["request_terminal_ns"]),
                token_observed_ns=tuple(row["token_observed_ns"]),  # type: ignore[arg-type]
                eligible=True,
                completed=status == "completed",
                error=status == "unfinished",
            )
        )
    return reduce_formal_slo_goodput(
        tuple(requests),
        source_request_pool_sha256=str(observation["source_request_pool_sha256"]),
    )


def _adaptive_safety_reasons(
    observation: dict[str, object],
    *,
    require_published_update: bool,
) -> tuple[str, ...]:
    reasons = []
    for name in (
        "communicator_failures",
        "exactness_violations",
        "fallbacks",
        "nonfinite_updates",
        "oom_events",
        "retractions",
        "version_mismatches",
    ):
        if _counter(observation, name) != 0:
            reasons.append(name)
    if require_published_update and (
        _counter(observation, "updates_launched") in {None, 0}
        or _counter(observation, "updates_published") in {None, 0}
    ):
        reasons.append("no_published_update")
    return tuple(sorted(reasons))


def _selection_anchor_evaluations(
    roles: dict[str, list[MaterializedCell]],
    observed: dict[str, dict[str, object]],
    actual_by_id: dict[str, FormalSingleOperatorValidatedActual],
) -> tuple[list[dict[str, object]], set[str]]:
    """Record every anchor, while only Static/TTS gate LightCone ranking."""

    evaluations = []
    ranking_reasons = set()
    for role in ("Target-only", "Static", "TTS", "L0-naive"):
        cell = roles[role][0]
        reasons = _adaptive_safety_reasons(
            observed[cell.cell_id],
            require_published_update=role in {"TTS", "L0-naive"},
        )
        evaluations.append(
            {
                "method_role": role,
                "cell_id": cell.cell_id,
                "eligible": not reasons,
                "reason_codes": list(reasons),
                "evidence_ids": sorted(
                    {cell.cell_id, actual_by_id[cell.cell_id].result_identity_sha256}
                ),
            }
        )
        if role in {"Static", "TTS"}:
            ranking_reasons.update(f"{role}:{reason}" for reason in reasons)
    return evaluations, ranking_reasons


def _reduce_single_operator_tts_calibration(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    if predecessor is None or predecessor.artifact.node != "e3a":
        raise ValueError("TTS-Cal reducer requires exact completed E3a")
    if materialization.stage != "TTS-Cal" or len(materialization.cells) != 288:
        raise ValueError("TTS-Cal reducer requires exact 288-cell materialization")
    cells = {row.cell_id: row for row in materialization.cells}
    actual_by_id = {row.cell_id: row for row in actual_results}
    grouped: dict[str, list[tuple[MaterializedCell, object, dict[str, object]]]] = {}
    inventories: set[object] = set()
    for cell_id in sorted(cells):
        cell = cells[cell_id]
        observation = _serving_observation(actual_by_id[cell_id], cell)
        inventories.add(observation["inventory_sha256"])
        if cell.method_role != "TTS" or cell.recipe_sha256 is None:
            raise ValueError("TTS-Cal contains a non-TTS candidate")
        dimensions = dict(cell.dimensions)
        if (
            type(dimensions.get("learning_rate")) is not float
            or type(dimensions.get("stride")) is not int
            or type(dimensions.get("block")) is not int
        ):
            raise ValueError("TTS-Cal candidate dimensions differ")
        slo = _single_operator_slo_goodput(observation)
        grouped.setdefault(cell.recipe_sha256, []).append((cell, slo, observation))
    if (
        len(inventories) != 1
        or len(grouped) != 72
        or any(len(rows) != 4 for rows in grouped.values())
    ):
        raise ValueError("TTS-Cal candidate/technical-replicate coverage differs")
    feasible: dict[str, Fraction] = {}
    candidate_rows = []
    for candidate_id in sorted(grouped):
        rows = grouped[candidate_id]
        dimensions = [dict(row[0].dimensions) for row in rows]
        learning_rates = {row["learning_rate"] for row in dimensions}
        strides = {row["stride"] for row in dimensions}
        blocks = {row["block"] for row in dimensions}
        reasons = {
            reason
            for _cell, slo, observation in rows
            for reason in (
                *(_adaptive_safety_reasons(observation, require_published_update=True)),
                *(() if slo.status == "PASS" else ("slo_failed",)),
            )
        }
        if len(learning_rates) != 1 or len(strides) != 1 or blocks != {0, 1, 2, 3}:
            raise ValueError(
                "TTS-Cal candidate does not have four exact technical replicates"
            )
        scores = tuple(slo.goodput_tokens_per_second for _cell, slo, _obs in rows)
        mean = sum(scores, Fraction()) / len(scores)
        if not reasons:
            feasible[candidate_id] = mean
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "learning_rate": next(iter(learning_rates)),
                "stride": next(iter(strides)),
                "blocks": sorted(blocks),
                "mean_goodput": [mean.numerator, mean.denominator],
                "eligible": not reasons,
                "reason_codes": sorted(reasons),
                "slo_observation_sha256s": sorted(
                    slo.sha256 for _cell, slo, _obs in rows
                ),
                "evidence_ids": sorted(
                    {
                        *(cell.cell_id for cell, _slo, _observation in rows),
                        *(
                            actual_by_id[cell.cell_id].result_identity_sha256
                            for cell, _slo, _observation in rows
                        ),
                        *(slo.sha256 for _cell, slo, _observation in rows),
                    }
                ),
            }
        )
    if not feasible:
        negative = {
            "schema_version": 1,
            "kind": "formal_single_operator_tts_calibration_selection",
            "status": "NO_SAFE_SLO_WINNER",
            "candidate_evaluations": candidate_rows,
            "candidate_evaluations_sha256": _content_sha256(candidate_rows),
            "reason_codes": ["no_safe_slo_feasible_tts_candidate"],
            "materialization_sha256": materialization.sha256,
        }
        decision_sha256 = _content_sha256(negative)
        return FormalSingleOperatorDecisionDraft(
            decision_kind="tts_calibration_no_safe_slo_winner",
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**negative, "selection_sha256": decision_sha256},
        )
    selected_id = min(
        feasible,
        key=lambda candidate_id: (-feasible[candidate_id], candidate_id),
    )
    selected = next(row for row in candidate_rows if row["candidate_id"] == selected_id)
    e3a = predecessor.decision.payload
    e3a_selection_sha256 = _require_sha256(
        "TTS-Cal E3a selection", e3a.get("selection_sha256")
    )
    selection = {
        "schema_version": 1,
        "kind": "formal_single_operator_tts_calibration_selection",
        "candidate_id": selected_id,
        "learning_rate": selected["learning_rate"],
        "stride": selected["stride"],
        "mean_goodput": selected["mean_goodput"],
        "status": "READY",
        "inventory_sha256": next(iter(inventories)),
        "candidate_evaluations_sha256": _content_sha256(candidate_rows),
        "candidate_evaluations": candidate_rows,
        "e3a_selection_sha256": e3a_selection_sha256,
        "model": e3a["model"],
        "matched_width": e3a["matched_width"],
        "common_load": e3a["common_load"],
    }
    selection_sha256 = _content_sha256(selection)
    return FormalSingleOperatorDecisionDraft(
        decision_kind="tts_calibration_actual_288_reduced",
        next_materialization_source_decision_sha256=e3a_selection_sha256,
        next_materialization_upstream_receipt_sha256s=(
            materialization.sha256,
            selection_sha256,
            e3a_selection_sha256,
        ),
        payload={**selection, "selection_sha256": selection_sha256},
    )


def _materialize_single_operator_e1(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != "tts_cal":
        raise ValueError("E1 requires exact completed TTS-Cal")
    from lightcone_spec.experiments.stage_materialization import (
        _materialize_e1_first_slice_from_verified_decisions,
    )

    payload = predecessor.decision.payload
    status = payload.get("status")
    if status == "NO_SAFE_SLO_WINNER":
        raise FormalSingleOperatorStageBlocked(
            f"E1 cannot advance from TTS-Cal: {status}"
        )
    if status != "READY":
        raise ValueError("TTS-Cal selection status is malformed")
    upstream = predecessor.decision.next_materialization_upstream_receipt_sha256s
    if (
        len(upstream) != 3
        or payload.get("selection_sha256") != upstream[1]
        or payload.get("e3a_selection_sha256") != upstream[2]
        or predecessor.decision.next_materialization_source_decision_sha256
        != upstream[2]
    ):
        raise ValueError("E1 predecessor decisions differ")
    return _materialize_e1_first_slice_from_verified_decisions(
        protocol_lock_sha256=protocol_lock.sha256,
        tts_calibration_receipt_sha256=upstream[0],
        signed_tts_calibration_seal_sha256=upstream[1],
        e3a_selection_sha256=upstream[2],
        frozen_tts_recipe_sha256=_require_sha256(
            "single-operator frozen TTS recipe", payload.get("candidate_id")
        ),
        e1_recipe_anchor_authority_sha256=(
            protocol_lock.e1_recipe_anchor_authority_sha256
        ),
        model=_require_text("single-operator E1 model", payload.get("model")),
        matched_width=int(payload["matched_width"]),
        common_load=int(payload["common_load"]),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _request_metrics(
    observation: dict[str, object],
) -> tuple[dict[str, object], ...]:
    metrics = []
    for row in _single_operator_request_rows(observation):
        if (
            row["terminal_status"] != "completed"
            or row["submitted_to_server"] is not True
        ):
            raise ValueError("single-operator request metric is incomplete")
        timestamps = tuple(row["token_observed_ns"])  # type: ignore[arg-type]
        inter_token = tuple(right - left for left, right in pairwise(timestamps))
        if not inter_token:
            raise ValueError("single-operator request lacks an ITL sample")
        ordered = sorted(inter_token)
        position = Fraction((len(ordered) - 1) * 99, 100)
        lower = position.numerator // position.denominator
        upper = min(lower + 1, len(ordered) - 1)
        p99 = Fraction(ordered[lower]) * (1 - (position - lower)) + Fraction(
            ordered[upper]
        ) * (position - lower)
        latency = int(row["request_terminal_ns"]) - int(row["request_started_ns"])
        if latency < 1:
            raise ValueError("single-operator request latency is not positive")
        metrics.append(
            {
                "request_id": row["request_id"],
                "input_token_ids": tuple(row["input_token_ids"]),  # type: ignore[arg-type]
                "output_token_ids": tuple(row["output_token_ids"]),  # type: ignore[arg-type]
                "output_tokens": len(row["output_token_ids"]),  # type: ignore[arg-type]
                "latency_ns": latency,
                "p99_itl_ns": p99,
            }
        )
    return tuple(metrics)


def _paired_confidence_lower(
    numerator: tuple[dict[str, object], ...],
    denominator: tuple[dict[str, object], ...],
) -> float:
    left = {str(row["request_id"]): row for row in numerator}
    right = {str(row["request_id"]): row for row in denominator}
    if not left or set(left) != set(right):
        raise ValueError("single-operator paired confidence request IDs differ")
    ratios = []
    for request_id in sorted(left):
        lhs = left[request_id]
        rhs = right[request_id]
        if (
            lhs["input_token_ids"] != rhs["input_token_ids"]
            or lhs["output_token_ids"] != rhs["output_token_ids"]
        ):
            raise ValueError("single-operator paired token trajectories differ")
        left_rate = int(lhs["output_tokens"]) / int(lhs["latency_ns"])
        right_rate = int(rhs["output_tokens"]) / int(rhs["latency_ns"])
        if left_rate <= 0 or right_rate <= 0:
            raise ValueError("single-operator paired request rate is not positive")
        ratios.append(math.log(left_rate / right_rate))
    mean = statistics.fmean(ratios)
    lower = (
        mean
        if len(ratios) == 1
        else mean
        - 1.6448536269514722 * statistics.stdev(ratios) / math.sqrt(len(ratios))
    )
    result = math.exp(lower)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("single-operator confidence lower bound is invalid")
    return result


def _finite_counter(observation: dict[str, object], name: str) -> float:
    counters = observation["performance_counters"]
    assert isinstance(counters, dict)
    value = counters.get(name)
    if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"single-operator performance counter {name} is unavailable")
    return float(value)


def _geometry_payload(geometry: object) -> dict[str, object]:
    from lightcone_spec.experiments.stage_materialization import E1Geometry

    if type(geometry) is not E1Geometry:
        raise TypeError("single-operator E1 geometry differs")
    return {
        "scope": geometry.scope,
        "parameterization": geometry.parameterization,
        "rank": geometry.rank,
        "alpha_over_rank": geometry.alpha_over_rank,
        "geometry_sha256": geometry.sha256,
    }


def _geometry_from_payload(value: object) -> object:
    from lightcone_spec.experiments.stage_materialization import E1Geometry

    row = _strict(
        "single-operator E1 geometry",
        value,
        {"scope", "parameterization", "rank", "alpha_over_rank", "geometry_sha256"},
    )
    expected = _require_sha256(
        "single-operator E1 geometry", row.pop("geometry_sha256")
    )
    geometry = E1Geometry(**row)  # type: ignore[arg-type]
    if geometry.sha256 != expected:
        raise ValueError("single-operator E1 geometry digest differs")
    return geometry


def _reduce_single_operator_e1(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    if predecessor is None or predecessor.artifact.node != "tts_cal":
        raise ValueError("E1 reducer requires exact completed TTS-Cal")
    if materialization.stage != "E1" or len(materialization.cells) != 68:
        raise ValueError("E1 reducer requires exact 68-cell materialization")
    from lightcone_spec.experiments.stage_materialization import (
        E1_OPTIMIZER_ANCHORS,
        E1Geometry,
    )

    cells = {row.cell_id: row for row in materialization.cells}
    actual_by_id = {row.cell_id: row for row in actual_results}
    observed = {
        cell_id: _serving_observation(actual_by_id[cell_id], cell)
        for cell_id, cell in cells.items()
    }
    roles: dict[str, list[MaterializedCell]] = {}
    for cell in cells.values():
        roles.setdefault(cell.method_role, []).append(cell)
    if {role: len(rows) for role, rows in roles.items()} != {
        "Target-only": 1,
        "Static": 1,
        "TTS": 1,
        "L0-naive": 1,
        "LightCone-candidate": 64,
    }:
        raise ValueError("E1 role cardinality differs")
    target = observed[roles["Target-only"][0].cell_id]
    static = observed[roles["Static"][0].cell_id]
    tts = observed[roles["TTS"][0].cell_id]
    l0 = observed[roles["L0-naive"][0].cell_id]
    identity = _single_operator_request_identity(target)
    for label, row in (("Static", static), ("TTS", tts), ("L0-naive", l0)):
        if _single_operator_request_identity(row) != identity:
            raise ValueError(f"E1 {label} token trajectories differ")
    anchor_evaluations, reference_reasons = _selection_anchor_evaluations(
        roles, observed, actual_by_id
    )
    static_metrics = () if reference_reasons else _request_metrics(static)
    tts_metrics = () if reference_reasons else _request_metrics(tts)
    grouped: dict[str, tuple[object, dict[str, dict[str, object]]]] = {}
    for cell in roles["LightCone-candidate"]:
        dimensions = dict(cell.dimensions)
        geometry = E1Geometry(
            scope=str(dimensions["scope"]),
            parameterization=str(dimensions["parameterization"]),  # type: ignore[arg-type]
            rank=None if dimensions["rank"] == "none" else int(dimensions["rank"]),
            alpha_over_rank=(
                None
                if dimensions["alpha_over_rank"] == "none"
                else float(dimensions["alpha_over_rank"])
            ),
        )
        optimizer = str(dimensions["optimizer_anchor"])
        entry = grouped.setdefault(geometry.sha256, (geometry, {}))
        if optimizer in entry[1]:
            raise ValueError("E1 repeats a geometry/optimizer row")
        entry[1][optimizer] = {"cell": cell, "observation": observed[cell.cell_id]}
    if len(grouped) != 32 or any(
        set(rows) != set(E1_OPTIMIZER_ANCHORS) for _geometry, rows in grouped.values()
    ):
        raise ValueError("E1 geometry/optimizer grid differs")
    candidate_evaluations = []
    evaluations = []
    for geometry_sha256 in sorted(grouped):
        geometry, candidates = grouped[geometry_sha256]
        confidence = []
        peak = []
        p99 = []
        exposed = []
        unsafe = set(reference_reasons)
        evidence_ids = {
            roles["Target-only"][0].cell_id,
            roles["Static"][0].cell_id,
            roles["TTS"][0].cell_id,
            roles["L0-naive"][0].cell_id,
            actual_by_id[roles["Target-only"][0].cell_id].result_identity_sha256,
            actual_by_id[roles["Static"][0].cell_id].result_identity_sha256,
            actual_by_id[roles["TTS"][0].cell_id].result_identity_sha256,
            actual_by_id[roles["L0-naive"][0].cell_id].result_identity_sha256,
        }
        for optimizer in E1_OPTIMIZER_ANCHORS:
            candidate = candidates[optimizer]
            cell = candidate["cell"]
            assert isinstance(cell, MaterializedCell)
            observation = candidate["observation"]
            assert isinstance(observation, dict)
            evidence_ids.update(
                {
                    cell.cell_id,
                    actual_by_id[cell.cell_id].result_identity_sha256,
                }
            )
            unsafe.update(
                f"{optimizer}:{reason}"
                for reason in _adaptive_safety_reasons(
                    observation, require_published_update=True
                )
            )
            if _single_operator_request_identity(observation) != identity:
                unsafe.add(f"{optimizer}:target_token_trajectory_mismatch")
            if unsafe:
                continue
            metrics = _request_metrics(observation)
            confidence.extend(
                (
                    _paired_confidence_lower(metrics, static_metrics),
                    _paired_confidence_lower(metrics, tts_metrics),
                )
            )
            peak.append(_counter(observation, "peak_hbm_bytes"))
            p99.append(
                max(math.ceil(metric["p99_itl_ns"] / 1_000) for metric in metrics)
            )
            exposed.append(
                math.ceil(_finite_counter(observation, "exposed_update_ms") * 1_000)
            )
        eligible = not unsafe
        row = {
            "geometry": _geometry_payload(geometry),
            "eligible": eligible,
            "reason_codes": sorted(unsafe),
            "evidence_ids": sorted(evidence_ids),
            "confidence_lower_request_rate_ratio": (
                min(confidence) if eligible else None
            ),
            "peak_hbm_bytes": (max(int(value) for value in peak) if eligible else None),
            "p99_itl_us": max(p99) if eligible else None,
            "exposed_update_us": max(exposed) if eligible else None,
        }
        candidate_evaluations.append(row)
        if eligible:
            evaluations.append(row)
    if not evaluations:
        tts_payload = predecessor.decision.payload
        negative = {
            "schema_version": 1,
            "kind": "formal_single_operator_e1_pareto_selection",
            "status": "NO_SAFE_GEOMETRY",
            "model": tts_payload["model"],
            "matched_width": tts_payload["matched_width"],
            "common_load": tts_payload["common_load"],
            "frozen_tts_recipe_sha256": tts_payload["candidate_id"],
            "surviving_geometries": [],
            "anchor_evaluations": anchor_evaluations,
            "candidate_evaluations": candidate_evaluations,
            "evaluation_set_sha256": _content_sha256(candidate_evaluations),
            "reason_codes": ["no_safe_e1_geometry"],
        }
        selection_sha256 = _content_sha256(negative)
        return FormalSingleOperatorDecisionDraft(
            decision_kind="e1_no_safe_geometry",
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**negative, "selection_sha256": selection_sha256},
        )

    def dominates(left: dict[str, object], right: dict[str, object]) -> bool:
        weak = (
            float(left["confidence_lower_request_rate_ratio"])
            >= float(right["confidence_lower_request_rate_ratio"])
            and int(left["peak_hbm_bytes"]) <= int(right["peak_hbm_bytes"])
            and int(left["p99_itl_us"]) <= int(right["p99_itl_us"])
            and int(left["exposed_update_us"]) <= int(right["exposed_update_us"])
        )
        strict = (
            float(left["confidence_lower_request_rate_ratio"])
            > float(right["confidence_lower_request_rate_ratio"])
            or int(left["peak_hbm_bytes"]) < int(right["peak_hbm_bytes"])
            or int(left["p99_itl_us"]) < int(right["p99_itl_us"])
            or int(left["exposed_update_us"]) < int(right["exposed_update_us"])
        )
        return weak and strict

    survivors = tuple(
        row
        for row in evaluations
        if not any(other is not row and dominates(other, row) for other in evaluations)
    )
    geometries = tuple(
        _geometry_from_payload(row["geometry"])
        for row in sorted(
            survivors,
            key=lambda item: str(item["geometry"]["geometry_sha256"]),  # type: ignore[index]
        )
    )
    if not geometries:
        raise AssertionError("non-empty safe E1 set must have a Pareto survivor")
    tts_payload = predecessor.decision.payload
    selection = {
        "schema_version": 1,
        "kind": "formal_single_operator_e1_pareto_selection",
        "status": "READY",
        "model": tts_payload["model"],
        "matched_width": tts_payload["matched_width"],
        "common_load": tts_payload["common_load"],
        "frozen_tts_recipe_sha256": tts_payload["candidate_id"],
        "surviving_geometries": [_geometry_payload(row) for row in geometries],
        "anchor_evaluations": anchor_evaluations,
        "candidate_evaluations": candidate_evaluations,
        "evaluation_set_sha256": _content_sha256(candidate_evaluations),
    }
    selection_sha256 = _content_sha256(selection)
    return FormalSingleOperatorDecisionDraft(
        decision_kind="e1_actual_68_pareto_reduced",
        next_materialization_source_decision_sha256=selection_sha256,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**selection, "selection_sha256": selection_sha256},
    )


def _materialize_single_operator_e2_round0(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != "e1":
        raise ValueError("E2 round0 requires exact completed E1")
    from lightcone_spec.experiments.stage_materialization import (
        _materialize_e2_round_from_verified_values,
        default_e2_recipe_grid_authority,
    )

    decision = predecessor.decision
    payload = decision.payload
    status = payload.get("status")
    if status == "NO_SAFE_GEOMETRY":
        raise FormalSingleOperatorStageBlocked(
            f"E2 round0 cannot advance from E1: {status}"
        )
    if status != "READY":
        raise ValueError("E1 selection status is malformed")
    if (
        len(decision.next_materialization_upstream_receipt_sha256s) != 1
        or decision.next_materialization_upstream_receipt_sha256s[0]
        != predecessor.materialization.sha256
        or decision.next_materialization_source_decision_sha256
        != payload.get("selection_sha256")
    ):
        raise ValueError("E2 round0 predecessor selection differs")
    grid = default_e2_recipe_grid_authority()
    if grid.sha256 != protocol_lock.e2_recipe_grid_authority_sha256:
        raise ValueError("single-operator E2 grid differs from ProtocolLock")
    geometries = tuple(
        _geometry_from_payload(row)
        for row in _array(
            "single-operator E1 survivors",
            payload["surviving_geometries"],
        )
    )
    return _materialize_e2_round_from_verified_values(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=predecessor.materialization.sha256,
        source_selection_sha256=_require_sha256(
            "single-operator E1 selection", payload.get("selection_sha256")
        ),
        grid=grid,
        geometries=geometries,  # type: ignore[arg-type]
        round_index=0,
        model=_require_text("single-operator E2 model", payload.get("model")),
        matched_width=int(payload["matched_width"]),
        common_load=int(payload["common_load"]),
        frozen_tts_recipe_sha256=_require_sha256(
            "single-operator E2 frozen TTS recipe",
            payload.get("frozen_tts_recipe_sha256"),
        ),
        candidate_recipes=None,
        prior_round_materialization=None,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _e2_recipe_payload(recipe: object) -> dict[str, object]:
    from lightcone_spec.experiments.stage_materialization import E2CandidateRecipe

    if type(recipe) is not E2CandidateRecipe:
        raise TypeError("single-operator E2 recipe differs")
    return {
        "geometry": _geometry_payload(recipe.geometry),
        "optimizer": recipe.optimizer,
        "schedule": recipe.schedule,
        "learning_rate": recipe.learning_rate,
        "optimizer_recipe_authority_sha256": (recipe.optimizer_recipe_authority_sha256),
        "recipe_sha256": recipe.sha256,
    }


def _e2_recipe_from_payload(value: object) -> object:
    from lightcone_spec.experiments.stage_materialization import E2CandidateRecipe

    row = _strict(
        "single-operator E2 recipe",
        value,
        {
            "geometry",
            "optimizer",
            "schedule",
            "learning_rate",
            "optimizer_recipe_authority_sha256",
            "recipe_sha256",
        },
    )
    expected = _require_sha256("single-operator E2 recipe", row.pop("recipe_sha256"))
    geometry = _geometry_from_payload(row.pop("geometry"))
    recipe = E2CandidateRecipe(geometry=geometry, **row)  # type: ignore[arg-type]
    if recipe.sha256 != expected:
        raise ValueError("single-operator E2 recipe digest differs")
    return recipe


def _materialize_single_operator_e2_next(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
    *,
    prior_node: FormalSingleOperatorNode,
    round_index: int,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != prior_node:
        raise ValueError(f"E2 round{round_index} requires exact {prior_node}")
    from lightcone_spec.experiments.stage_materialization import (
        _materialize_e2_round_from_verified_values,
        default_e2_recipe_grid_authority,
    )

    payload = predecessor.decision.payload
    status = payload.get("status")
    if status == "NO_SAFE_WINNER":
        raise FormalSingleOperatorStageBlocked(
            f"E2 round{round_index} cannot advance: {status}"
        )
    if status != "READY":
        raise ValueError("prior E2 selection status is malformed")
    selection_sha256 = _require_sha256(
        "single-operator prior E2 selection", payload.get("selection_sha256")
    )
    if (
        predecessor.decision.next_materialization_source_decision_sha256
        != selection_sha256
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
        or payload.get("round_index") != round_index - 1
    ):
        raise ValueError("single-operator prior E2 transition differs")
    grid = default_e2_recipe_grid_authority()
    if grid.sha256 != protocol_lock.e2_recipe_grid_authority_sha256:
        raise ValueError("single-operator E2 grid differs from ProtocolLock")
    geometries = tuple(
        _geometry_from_payload(row)
        for row in _array(
            "single-operator E2 source geometries", payload["source_geometries"]
        )
    )
    recipes = tuple(
        _e2_recipe_from_payload(row)
        for row in _array("single-operator E2 survivors", payload["survivor_recipes"])
    )
    return _materialize_e2_round_from_verified_values(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=predecessor.materialization.sha256,
        source_selection_sha256=selection_sha256,
        grid=grid,
        geometries=geometries,  # type: ignore[arg-type]
        round_index=round_index,
        model=_require_text("single-operator E2 model", payload.get("model")),
        matched_width=int(payload["matched_width"]),
        common_load=int(payload["common_load"]),
        frozen_tts_recipe_sha256=_require_sha256(
            "single-operator frozen TTS recipe",
            payload.get("frozen_tts_recipe_sha256"),
        ),
        candidate_recipes=recipes,  # type: ignore[arg-type]
        prior_round_materialization=predecessor.materialization,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _materialize_single_operator_e2_round1(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    return _materialize_single_operator_e2_next(
        predecessor, protocol_lock, prior_node="e2_r0", round_index=1
    )


def _materialize_single_operator_e2_round2(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    return _materialize_single_operator_e2_next(
        predecessor, protocol_lock, prior_node="e2_r1", round_index=2
    )


def _materialize_single_operator_e2_round3(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    return _materialize_single_operator_e2_next(
        predecessor, protocol_lock, prior_node="e2_r2", round_index=3
    )


def _e2_recipe_from_cell(cell: MaterializedCell) -> object:
    from lightcone_spec.experiments.stage_materialization import (
        E1Geometry,
        E2CandidateRecipe,
    )

    dimensions = dict(cell.dimensions)
    geometry = E1Geometry(
        scope=str(dimensions["scope"]),
        parameterization=str(dimensions["parameterization"]),  # type: ignore[arg-type]
        rank=None if dimensions["rank"] == "none" else int(dimensions["rank"]),
        alpha_over_rank=(
            None
            if dimensions["alpha_over_rank"] == "none"
            else float(dimensions["alpha_over_rank"])
        ),
    )
    recipe = E2CandidateRecipe(
        geometry=geometry,
        optimizer=str(dimensions["optimizer"]),
        schedule=str(dimensions["schedule"]),
        learning_rate=float(dimensions["learning_rate"]),
        optimizer_recipe_authority_sha256=str(
            dimensions["optimizer_recipe_authority_sha256"]
        ),
    )
    if recipe.sha256 != cell.recipe_sha256:
        raise ValueError("single-operator E2 cell recipe differs")
    return recipe


def _reduce_single_operator_e2(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    if predecessor is None or materialization.stage != "E2":
        raise ValueError("E2 reducer requires an exact prior stage")
    from lightcone_spec.experiments.e2_stage_authority import (
        E2StagedCandidateEvaluation,
        _select_survivor_recipes,
    )

    round_values = {
        dict(cell.dimensions).get("round") for cell in materialization.cells
    }
    if len(round_values) != 1 or next(iter(round_values)) not in range(4):
        raise ValueError("single-operator E2 materialization round differs")
    round_index = int(next(iter(round_values)))
    expected_prior = "e1" if round_index == 0 else f"e2_r{round_index - 1}"
    if predecessor.artifact.node != expected_prior:
        raise ValueError("single-operator E2 predecessor differs")
    cells = {row.cell_id: row for row in materialization.cells}
    actual_by_id = {row.cell_id: row for row in actual_results}
    observed = {
        cell_id: _serving_observation(actual_by_id[cell_id], cell)
        for cell_id, cell in cells.items()
    }
    roles: dict[str, list[MaterializedCell]] = {}
    for cell in cells.values():
        roles.setdefault(cell.method_role, []).append(cell)
    if any(
        len(roles.get(role, ())) != 1
        for role in ("Target-only", "Static", "TTS", "L0-naive")
    ):
        raise ValueError("E2 fixed-anchor cardinality differs")
    candidates = roles.get("LightCone-candidate", [])
    if not candidates:
        raise ValueError("E2 has no candidate rows")
    target = observed[roles["Target-only"][0].cell_id]
    static = observed[roles["Static"][0].cell_id]
    tts = observed[roles["TTS"][0].cell_id]
    l0 = observed[roles["L0-naive"][0].cell_id]
    target_identity = _single_operator_request_identity(target)
    for label, row in (("Static", static), ("TTS", tts), ("L0-naive", l0)):
        if _single_operator_request_identity(row) != target_identity:
            raise ValueError(f"E2 {label} anchor is unpaired")
    anchor_evaluations, reference_reasons = _selection_anchor_evaluations(
        roles, observed, actual_by_id
    )
    static_metrics = () if reference_reasons else _request_metrics(static)
    tts_metrics = () if reference_reasons else _request_metrics(tts)
    source_recipes = tuple(
        sorted(
            (_e2_recipe_from_cell(cell) for cell in candidates),
            key=lambda row: row.sha256,
        )
    )
    evaluations = []
    candidate_evaluations = []
    anchor_evidence_ids = {
        cell.cell_id
        for role in ("Target-only", "Static", "TTS", "L0-naive")
        for cell in roles[role]
    } | {
        actual_by_id[cell.cell_id].result_identity_sha256
        for role in ("Target-only", "Static", "TTS", "L0-naive")
        for cell in roles[role]
    }
    for cell in candidates:
        observation = observed[cell.cell_id]
        reasons = set(reference_reasons)
        reasons.update(
            _adaptive_safety_reasons(observation, require_published_update=True)
        )
        if _single_operator_request_identity(observation) != target_identity:
            reasons.add("target_token_trajectory_mismatch")
        evaluation = None
        if not reasons:
            metrics = _request_metrics(observation)
            evaluation = E2StagedCandidateEvaluation(
                recipe=_e2_recipe_from_cell(cell),  # type: ignore[arg-type]
                cell_id=cell.cell_id,
                confidence_lower_request_rate_ratio=min(
                    _paired_confidence_lower(metrics, static_metrics),
                    _paired_confidence_lower(metrics, tts_metrics),
                ),
                peak_hbm_bytes=int(_counter(observation, "peak_hbm_bytes")),
                p99_itl_us=max(
                    math.ceil(metric["p99_itl_ns"] / 1_000) for metric in metrics
                ),
                exposed_update_us=math.ceil(
                    _finite_counter(observation, "exposed_update_ms") * 1_000
                ),
                launched_updates=int(_counter(observation, "updates_launched")),
                published_updates=int(_counter(observation, "updates_published")),
            )
            evaluations.append(evaluation)
        recipe = _e2_recipe_from_cell(cell)
        candidate_evaluations.append(
            {
                "recipe": _e2_recipe_payload(recipe),
                "cell_id": cell.cell_id,
                "eligible": not reasons,
                "reason_codes": sorted(reasons),
                "evidence_ids": sorted(
                    {
                        *anchor_evidence_ids,
                        cell.cell_id,
                        actual_by_id[cell.cell_id].result_identity_sha256,
                    }
                ),
                "evaluation_sha256": (
                    None if evaluation is None else evaluation.sha256
                ),
            }
        )
    canonical_evaluations = tuple(
        sorted(evaluations, key=lambda row: row.recipe.sha256)
    )
    candidate_evaluations.sort(
        key=lambda row: str(row["recipe"]["recipe_sha256"])  # type: ignore[index]
    )
    prior_payload = predecessor.decision.payload
    source_geometry_payloads = (
        prior_payload["surviving_geometries"]
        if round_index == 0
        else prior_payload["source_geometries"]
    )
    if round_index < 3:
        from lightcone_spec.experiments.stage_materialization import (
            E2_OPTIMIZERS,
            E2_SCHEDULES,
        )

        eligible_families = {
            (row.recipe.optimizer, row.recipe.schedule) for row in canonical_evaluations
        }
        expected_families = {
            (optimizer, schedule)
            for optimizer in E2_OPTIMIZERS
            for schedule in E2_SCHEDULES
        }
        enough = (
            len(canonical_evaluations) >= max(math.ceil(len(source_recipes) / 4), 21)
            and eligible_families == expected_families
        )
    else:
        enough = bool(canonical_evaluations)
    if not enough:
        negative = {
            "schema_version": 1,
            "kind": "formal_single_operator_e2_round_selection",
            "status": "NO_SAFE_WINNER",
            "round_index": round_index,
            "model": prior_payload["model"],
            "matched_width": prior_payload["matched_width"],
            "common_load": prior_payload["common_load"],
            "frozen_tts_recipe_sha256": prior_payload["frozen_tts_recipe_sha256"],
            "source_geometries": source_geometry_payloads,
            "source_candidate_count": len(source_recipes),
            "survivor_recipes": [],
            "final_recipe": None,
            "anchor_evaluations": anchor_evaluations,
            "candidate_evaluations": candidate_evaluations,
            "evaluation_sha256s": [row.sha256 for row in canonical_evaluations],
            "reason_codes": ["e2_safe_family_floor_or_winner_unavailable"],
        }
        selection_sha256 = _content_sha256(negative)
        return FormalSingleOperatorDecisionDraft(
            decision_kind=f"e2_round{round_index}_no_safe_winner",
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**negative, "selection_sha256": selection_sha256},
        )
    survivors, final_recipe = _select_survivor_recipes(
        source_recipes=source_recipes,  # type: ignore[arg-type]
        evaluations=canonical_evaluations,
        round_index=round_index,
    )
    selection = {
        "schema_version": 1,
        "kind": "formal_single_operator_e2_round_selection",
        "status": "READY",
        "round_index": round_index,
        "model": prior_payload["model"],
        "matched_width": prior_payload["matched_width"],
        "common_load": prior_payload["common_load"],
        "frozen_tts_recipe_sha256": prior_payload["frozen_tts_recipe_sha256"],
        "source_geometries": source_geometry_payloads,
        "source_candidate_count": len(source_recipes),
        "survivor_recipes": [_e2_recipe_payload(row) for row in survivors],
        "final_recipe": None
        if final_recipe is None
        else _e2_recipe_payload(final_recipe),
        "anchor_evaluations": anchor_evaluations,
        "candidate_evaluations": candidate_evaluations,
        "evaluation_sha256s": [row.sha256 for row in canonical_evaluations],
    }
    selection_sha256 = _content_sha256(selection)
    return FormalSingleOperatorDecisionDraft(
        decision_kind=f"e2_round{round_index}_actual_reduced",
        next_materialization_source_decision_sha256=selection_sha256,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**selection, "selection_sha256": selection_sha256},
    )


def _materialize_single_operator_e4_screen(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != "e2_r3":
        raise ValueError("E4 screen requires exact completed E2 round3")
    from lightcone_spec.experiments.stage_materialization import (
        _materialize_e4_strength2_screen_diagnostic,
    )

    payload = predecessor.decision.payload
    # Schema-1 E2 selections predate the explicit READY field.  They remain
    # replayable, while every explicit scientific negative is fail-closed.
    status = payload.get("status")
    if status == "NO_SAFE_WINNER":
        raise FormalSingleOperatorStageBlocked(
            f"E4 screen cannot advance from E2: {status}"
        )
    if status not in {None, "READY"}:
        raise ValueError("E2 final selection status is malformed")
    final_recipe = _e2_recipe_from_payload(payload.get("final_recipe"))
    selection_sha256 = _require_sha256(
        "single-operator E2 final selection", payload.get("selection_sha256")
    )
    if (
        payload.get("round_index") != 3
        or predecessor.decision.next_materialization_source_decision_sha256
        != selection_sha256
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E4 screen predecessor differs")
    return _materialize_e4_strength2_screen_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_e2_receipt_sha256=predecessor.materialization.sha256,
        source_decision_sha256=selection_sha256,
        model=_require_text("single-operator E4 model", payload.get("model")),
        lightcone_recipe_sha256=final_recipe.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _e4_configuration_payload(
    value: tuple[tuple[str, str | int], ...],
) -> list[list[str | int]]:
    return [[name, level] for name, level in value]


def _e4_configuration_from_payload(
    value: object,
) -> tuple[tuple[str, str | int], ...]:
    rows = _array("single-operator E4 configuration", value)
    result = []
    for row in rows:
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) not in {str, int}
        ):
            raise ValueError("single-operator E4 configuration row differs")
        result.append((row[0], row[1]))
    return tuple(result)


def _reduce_single_operator_e4_headline(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    if predecessor is None or materialization.stage != "E4":
        raise ValueError("E4 headline reducer requires its immediate predecessor")
    from lightcone_spec.experiments.e4_stage_authority import (
        E4ConfigurationEvaluation,
        _configuration,
        _neighborhoods,
        _ranking,
    )
    from lightcone_spec.experiments.stage_materialization import E4_LOADS, E4_TRAFFIC

    if materialization.materialization_rule == "strength2_8_rows_x_3_loads_x_2_traffic":
        phase = "screen"
        expected_node = "e2_r3"
        expected_count = 48
    elif (
        materialization.materialization_rule
        == "winner_neighborhood_2pow4_x_3_loads_x_2_traffic"
    ):
        phase = "local"
        expected_node = "e4_screen"
        expected_count = 96
    else:
        raise ValueError("E4 headline reducer cannot consume profiler rows")
    if (
        predecessor.artifact.node != expected_node
        or len(materialization.cells) != expected_count
    ):
        raise ValueError("E4 headline phase/predecessor differs")
    cells = {row.cell_id: row for row in materialization.cells}
    actual_by_id = {row.cell_id: row for row in actual_results}
    observed = {
        cell_id: _serving_observation(actual_by_id[cell_id], cell)
        for cell_id, cell in cells.items()
    }
    if any(cell.method_role != "LightCone" for cell in cells.values()):
        raise ValueError("E4 headline contains a non-LightCone row")
    recipes = {cell.recipe_sha256 for cell in cells.values()}
    models = {cell.model for cell in cells.values()}
    inventories = {row["inventory_sha256"] for row in observed.values()}
    if (
        len(recipes) != 1
        or None in recipes
        or len(models) != 1
        or len(inventories) != 1
    ):
        raise ValueError("E4 headline model/recipe/inventory differs")
    grouped: dict[
        tuple[tuple[str, str | int], ...],
        list[tuple[MaterializedCell, dict[str, object]]],
    ] = {}
    strata: dict[tuple[str, str], list[dict[str, object]]] = {}
    for cell in cells.values():
        observation = observed[cell.cell_id]
        configuration = _configuration(cell, phase)  # type: ignore[arg-type]
        grouped.setdefault(configuration, []).append((cell, observation))
        dimensions = dict(cell.dimensions)
        strata.setdefault(
            (str(dimensions["load"]), str(dimensions["traffic"])), []
        ).append(observation)
    expected_strata = {(load, traffic) for load in E4_LOADS for traffic in E4_TRAFFIC}
    if set(strata) != expected_strata:
        raise ValueError("E4 evidence lacks a registered stratum")
    if any(
        len({_single_operator_request_identity(row) for row in rows}) != 1
        for rows in strata.values()
    ):
        raise ValueError("E4 configurations use different request trajectories")
    evaluations = []
    candidate_evaluations = []
    for configuration, rows in grouped.items():
        if (
            len(rows) != 6
            or {
                (
                    str(dict(cell.dimensions)["load"]),
                    str(dict(cell.dimensions)["traffic"]),
                )
                for cell, _observation in rows
            }
            != expected_strata
        ):
            raise ValueError("E4 configuration lacks exact stratum coverage")
        reasons = {
            reason
            for _cell, observation in rows
            for reason in _adaptive_safety_reasons(
                observation,
                require_published_update=True,
            )
        }
        rates = []
        peak = []
        p99 = []
        exposed = []
        for _cell, observation in rows:
            if reasons:
                continue
            metrics = _request_metrics(observation)
            tokens = sum(int(metric["output_tokens"]) for metric in metrics)
            latency = sum(int(metric["latency_ns"]) for metric in metrics)
            if tokens < 1 or latency < 1:
                raise ValueError("E4 request-token rate is not positive")
            rates.append(Fraction(tokens * 1_000_000_000, latency))
            peak.append(int(_counter(observation, "peak_hbm_bytes")))
            p99.append(
                max(math.ceil(metric["p99_itl_ns"] / 1_000) for metric in metrics)
            )
            exposed.append(
                math.ceil(_finite_counter(observation, "exposed_update_ms") * 1_000)
            )
        evaluation = None
        if not reasons:
            worst = min(rates)
            evaluation = E4ConfigurationEvaluation(
                configuration=configuration,
                cell_ids=tuple(sorted(cell.cell_id for cell, _observation in rows)),
                minimum_request_rate_numerator=worst.numerator,
                minimum_request_rate_denominator=worst.denominator,
                peak_hbm_bytes=max(peak),
                p99_itl_us=max(p99),
                exposed_update_us=max(exposed),
            )
            evaluations.append(evaluation)
        candidate_evaluations.append(
            {
                "configuration": _e4_configuration_payload(configuration),
                "configuration_sha256": _content_sha256(configuration),
                "eligible": not reasons,
                "reason_codes": sorted(reasons),
                "evidence_ids": sorted(
                    {
                        *(cell.cell_id for cell, _observation in rows),
                        *(
                            actual_by_id[cell.cell_id].result_identity_sha256
                            for cell, _observation in rows
                        ),
                    }
                ),
                "evaluation_sha256": (
                    None if evaluation is None else evaluation.sha256
                ),
            }
        )
    eligible = tuple(sorted(evaluations, key=lambda row: row.configuration_sha256))
    if not eligible:
        negative = {
            "schema_version": 1,
            "kind": "formal_single_operator_e4_selection",
            "status": "NO_SAFE_CONFIGURATION",
            "phase": phase,
            "model": next(iter(models)),
            "lightcone_recipe_sha256": next(iter(recipes)),
            "inventory_sha256": next(iter(inventories)),
            "winner_configuration": None,
            "factor_neighborhoods": None,
            "candidate_evaluations": sorted(
                candidate_evaluations,
                key=lambda row: str(row["configuration_sha256"]),
            ),
            "evaluation_sha256s": [],
            "reason_codes": ["no_safe_complete_e4_configuration"],
        }
        selection_sha256 = _content_sha256(negative)
        return FormalSingleOperatorDecisionDraft(
            decision_kind=f"e4_{phase}_no_safe_configuration",
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**negative, "selection_sha256": selection_sha256},
        )
    winner = min(eligible, key=_ranking).configuration
    neighborhoods = _neighborhoods(winner) if phase == "screen" else None
    selection = {
        "schema_version": 1,
        "kind": "formal_single_operator_e4_selection",
        "status": "READY",
        "phase": phase,
        "model": next(iter(models)),
        "lightcone_recipe_sha256": next(iter(recipes)),
        "inventory_sha256": next(iter(inventories)),
        "winner_configuration": _e4_configuration_payload(winner),
        "factor_neighborhoods": (
            None
            if neighborhoods is None
            else [[name, left, right] for name, left, right in neighborhoods]
        ),
        "candidate_evaluations": sorted(
            candidate_evaluations,
            key=lambda row: str(row["configuration_sha256"]),
        ),
        "evaluation_sha256s": [
            _content_sha256(
                {
                    "configuration": _e4_configuration_payload(row.configuration),
                    "cell_ids": list(row.cell_ids),
                    "minimum_request_rate_numerator": (
                        row.minimum_request_rate_numerator
                    ),
                    "minimum_request_rate_denominator": (
                        row.minimum_request_rate_denominator
                    ),
                    "peak_hbm_bytes": row.peak_hbm_bytes,
                    "p99_itl_us": row.p99_itl_us,
                    "exposed_update_us": row.exposed_update_us,
                }
            )
            for row in eligible
        ],
    }
    selection_sha256 = _content_sha256(selection)
    return FormalSingleOperatorDecisionDraft(
        decision_kind=f"e4_{phase}_actual_reduced",
        next_materialization_source_decision_sha256=selection_sha256,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**selection, "selection_sha256": selection_sha256},
    )


def _materialize_single_operator_e4_local(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != "e4_screen":
        raise ValueError("E4 local requires exact completed screen")
    from lightcone_spec.experiments.stage_materialization import (
        _materialize_e4_winner_neighborhood_diagnostic,
    )

    payload = predecessor.decision.payload
    # Legacy schema-1 screen decisions omitted status; an explicit negative
    # disposition is still a typed scientific stop.
    status = payload.get("status")
    if status == "NO_SAFE_CONFIGURATION":
        raise FormalSingleOperatorStageBlocked(f"E4 local cannot advance: {status}")
    if status not in {None, "READY"}:
        raise ValueError("E4 screen selection status is malformed")
    neighborhoods = _array(
        "single-operator E4 factor neighborhoods", payload["factor_neighborhoods"]
    )
    typed_neighborhoods = tuple(
        (row[0], row[1], row[2])
        for row in neighborhoods
        if type(row) is list and len(row) == 3
    )
    if len(typed_neighborhoods) != len(neighborhoods):
        raise ValueError("single-operator E4 neighborhoods differ")
    selection_sha256 = _require_sha256(
        "single-operator E4 screen selection", payload.get("selection_sha256")
    )
    return _materialize_e4_winner_neighborhood_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_screen_receipt_sha256=predecessor.materialization.sha256,
        winner_decision_sha256=selection_sha256,
        model=_require_text("single-operator E4 model", payload.get("model")),
        lightcone_recipe_sha256=_require_sha256(
            "single-operator E4 recipe", payload.get("lightcone_recipe_sha256")
        ),
        factor_neighborhoods=typed_neighborhoods,  # type: ignore[arg-type]
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _materialize_single_operator_e4_profiler(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != "e4_local":
        raise ValueError("E4 profiler requires exact completed local selection")
    from lightcone_spec.experiments.stage_materialization import (
        _materialize_e4_profiler_diagnostic,
    )

    payload = predecessor.decision.payload
    # Preserve replay of schema-1 local selections while rejecting all
    # explicit non-ready scientific decisions.
    status = payload.get("status")
    if status == "NO_SAFE_CONFIGURATION":
        raise FormalSingleOperatorStageBlocked(f"E4 profiler cannot advance: {status}")
    if status not in {None, "READY"}:
        raise ValueError("E4 local selection status is malformed")
    selection_sha256 = _require_sha256(
        "single-operator E4 local selection", payload.get("selection_sha256")
    )
    selected_configuration_sha256 = _content_sha256(
        _e4_configuration_from_payload(payload["winner_configuration"])
    )
    return _materialize_e4_profiler_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_local_receipt_sha256=predecessor.materialization.sha256,
        source_decision_sha256=selection_sha256,
        selected_configuration_sha256=selected_configuration_sha256,
        model=_require_text("single-operator E4 model", payload.get("model")),
        lightcone_recipe_sha256=_require_sha256(
            "single-operator E4 recipe", payload.get("lightcone_recipe_sha256")
        ),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


_BLOCKED_NOT_CONNECTED = (
    "single-operator node has no code-owned current-only adapter yet"
)

_CONNECTED_DOWNSTREAM_NODES: tuple[FormalSingleOperatorNode, ...] = (
    "e3b_pilot",
    "e3b_final",
    "e1a",
    "e5_pilot",
    "e5_final",
    "e6_pilot",
    "e6_final",
    "e0_tuning",
    "e0_pilot",
    "e0_final",
)


def _single_operator_downstream_materializer(node: FormalSingleOperatorNode):
    from lightcone_spec.experiments import formal_single_operator_downstream

    return {
        "e3b_pilot": formal_single_operator_downstream.materialize_single_operator_e3b_pilot,
        "e3b_final": formal_single_operator_downstream.materialize_single_operator_e3b_final,
        "e1a": formal_single_operator_downstream.materialize_single_operator_e1a,
        "e5_pilot": formal_single_operator_downstream.materialize_single_operator_e5_pilot,
        "e5_final": formal_single_operator_downstream.materialize_single_operator_e5_final,
        "e6_pilot": formal_single_operator_downstream.materialize_single_operator_e6_pilot,
        "e6_final": formal_single_operator_downstream.materialize_single_operator_e6_final,
        "e0_tuning": formal_single_operator_downstream.materialize_single_operator_e0_tuning,
        "e0_pilot": formal_single_operator_downstream.materialize_single_operator_e0_pilot,
        "e0_final": formal_single_operator_downstream.materialize_single_operator_e0_final,
    }[node]


def _single_operator_downstream_reducer(node: FormalSingleOperatorNode):
    from lightcone_spec.experiments import formal_single_operator_downstream

    return {
        "e4_profiler": formal_single_operator_downstream.reduce_single_operator_e4_profiler,
        "e3b_pilot": formal_single_operator_downstream.reduce_single_operator_e3b_pilot,
        "e3b_final": formal_single_operator_downstream.reduce_single_operator_e3b_final,
        "e1a": formal_single_operator_downstream.reduce_single_operator_e1a,
        "e5_pilot": formal_single_operator_downstream.reduce_single_operator_e5_pilot,
        "e5_final": formal_single_operator_downstream.reduce_single_operator_e5_final,
        "e6_pilot": formal_single_operator_downstream.reduce_single_operator_e6_pilot,
        "e6_final": formal_single_operator_downstream.reduce_single_operator_e6_final,
        "e0_tuning": formal_single_operator_downstream.reduce_single_operator_e0_tuning,
        "e0_pilot": formal_single_operator_downstream.reduce_single_operator_e0_pilot,
        "e0_final": formal_single_operator_downstream.reduce_single_operator_e0_final,
    }[node]


@dataclass(frozen=True)
class _LazySingleOperatorDownstreamMaterializer:
    node: FormalSingleOperatorNode

    def __post_init__(self) -> None:
        if self.node not in _CONNECTED_DOWNSTREAM_NODES:
            raise ValueError("single-operator downstream materializer node differs")

    def __call__(
        self,
        predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
        protocol_lock: ProtocolLock,
        auxiliary_sources: tuple[FormalSingleOperatorAuxiliarySourceBinding, ...],
        /,
    ) -> StageMaterializationReceipt:
        materializer = _single_operator_downstream_materializer(self.node)
        if self.node in _E6_AUXILIARY_NODES:
            if tuple(row.source_kind for row in auxiliary_sources) != (
                "e6_interface_fit",
            ):
                raise ValueError("E6 materializer lacks exact interface/fit source")
            return materializer(
                predecessor,
                protocol_lock,
                auxiliary_sources[0].reopen(),
            )
        if self.node in _E0_AUXILIARY_NODES:
            if tuple(row.source_kind for row in auxiliary_sources) != (
                "e0_compatibility",
            ):
                raise ValueError("E0 materializer lacks exact compatibility source")
            return materializer(
                predecessor,
                protocol_lock,
                auxiliary_sources[0].reopen(),
            )
        if auxiliary_sources:
            raise ValueError("downstream materializer received an unexpected source")
        return materializer(predecessor, protocol_lock)


@dataclass(frozen=True)
class _LazySingleOperatorDownstreamReducer:
    node: FormalSingleOperatorNode

    def __post_init__(self) -> None:
        if self.node not in {"e4_profiler", *_CONNECTED_DOWNSTREAM_NODES}:
            raise ValueError("single-operator downstream reducer node differs")

    def __call__(
        self,
        predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
        materialization: StageMaterializationReceipt,
        actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
        /,
    ) -> FormalSingleOperatorDecisionDraft:
        return _single_operator_downstream_reducer(self.node)(
            predecessor,
            materialization,
            actual_results,
        )


_CLOSED_NODE_ADAPTERS: dict[
    FormalSingleOperatorNode,
    _FormalSingleOperatorClosedNodeAdapter,
] = {
    "preflight": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_preflight),
        actual_validator_kind="preflight",
        reducer=_reduce_single_operator_preflight,
        blocked_reason=None,
    ),
    "e3a": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_e3a),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_e3a,
        blocked_reason=None,
    ),
    "tts_cal": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(
            _materialize_single_operator_tts_calibration
        ),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_tts_calibration,
        blocked_reason=None,
    ),
    "e1": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_e1),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_e1,
        blocked_reason=None,
    ),
    "e2_r0": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_e2_round0),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_e2,
        blocked_reason=None,
    ),
    "e2_r1": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_e2_round1),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_e2,
        blocked_reason=None,
    ),
    "e2_r2": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_e2_round2),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_e2,
        blocked_reason=None,
    ),
    "e2_r3": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_e2_round3),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_e2,
        blocked_reason=None,
    ),
    "e4_screen": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_e4_screen),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_e4_headline,
        blocked_reason=None,
    ),
    "e4_local": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(_materialize_single_operator_e4_local),
        actual_validator_kind="run_manifest",
        reducer=_reduce_single_operator_e4_headline,
        blocked_reason=None,
    ),
    "e4_profiler": _FormalSingleOperatorClosedNodeAdapter(
        materializer=_without_auxiliary_sources(
            _materialize_single_operator_e4_profiler
        ),
        actual_validator_kind="profiler_terminal",
        reducer=_LazySingleOperatorDownstreamReducer("e4_profiler"),
        blocked_reason=None,
    ),
    **{
        node: _FormalSingleOperatorClosedNodeAdapter(
            materializer=_LazySingleOperatorDownstreamMaterializer(node),
            actual_validator_kind="run_manifest",
            reducer=_LazySingleOperatorDownstreamReducer(node),
            blocked_reason=None,
        )
        for node in _CONNECTED_DOWNSTREAM_NODES
    },
}

if tuple(_CLOSED_NODE_ADAPTERS) != FORMAL_SINGLE_OPERATOR_NODE_ORDER:
    raise AssertionError("single-operator closed adapter registry differs from the DAG")


@dataclass(frozen=True)
class FormalSingleOperatorNodeReadiness:
    """Code-path readiness for one node in the fixed local operator DAG."""

    node: FormalSingleOperatorNode
    ordinal: int
    stage: str
    phase: str
    status: Literal["READY", "BLOCKED"]
    materializer_available: bool
    actual_validator_kind: FormalSingleOperatorActualValidatorKind | None
    actual_validator_kinds: tuple[FormalSingleOperatorActualValidatorKind, ...]
    actual_validators_available: bool
    required_auxiliary_source_kinds: tuple[FormalSingleOperatorAuxiliarySourceKind, ...]
    reducer_available: bool
    blocker: str | None

    def __post_init__(self) -> None:
        spec = formal_single_operator_node_spec(self.node)
        if (
            (self.ordinal, self.stage, self.phase)
            != (spec.ordinal, spec.stage, spec.phase)
            or type(self.materializer_available) is not bool
            or type(self.actual_validators_available) is not bool
            or type(self.reducer_available) is not bool
        ):
            raise ValueError("single-operator readiness identity differs")
        expected_validator_kinds = formal_single_operator_actual_validator_kinds(
            self.node
        )
        if self.actual_validator_kinds != expected_validator_kinds:
            raise ValueError("single-operator readiness validator registry differs")
        if self.actual_validators_available != (
            set(self.actual_validator_kinds)
            <= FORMAL_SINGLE_OPERATOR_IMPLEMENTED_ACTUAL_VALIDATOR_KINDS
        ):
            raise ValueError("single-operator readiness validator availability differs")
        if self.actual_validator_kind not in self.actual_validator_kinds:
            raise ValueError("single-operator readiness primary validator differs")
        if self.required_auxiliary_source_kinds != (
            formal_single_operator_required_auxiliary_source_kinds(self.node)
        ):
            raise ValueError("single-operator readiness auxiliary contract differs")
        ready = (
            self.materializer_available
            and self.actual_validators_available
            and self.reducer_available
            and self.blocker is None
        )
        if (self.status == "READY") != ready:
            raise ValueError("single-operator readiness status differs")
        if ready:
            if self.blocker is not None:
                raise ValueError("ready single-operator node cannot carry a blocker")
        else:
            _require_text("single-operator readiness blocker", self.blocker)


def formal_single_operator_node_readiness() -> tuple[
    FormalSingleOperatorNodeReadiness, ...
]:
    """Return the closed 21-node implementation/readiness matrix."""

    rows = []
    for spec in FORMAL_SINGLE_OPERATOR_NODE_SPECS:
        adapter = _CLOSED_NODE_ADAPTERS[spec.node]
        validator_kinds = formal_single_operator_actual_validator_kinds(spec.node)
        validators_available = (
            set(validator_kinds)
            <= FORMAL_SINGLE_OPERATOR_IMPLEMENTED_ACTUAL_VALIDATOR_KINDS
        )
        ready = (
            adapter.materializer is not None
            and validators_available
            and adapter.reducer is not None
            and adapter.blocked_reason is None
        )
        rows.append(
            FormalSingleOperatorNodeReadiness(
                node=spec.node,
                ordinal=spec.ordinal,
                stage=spec.stage,
                phase=spec.phase,
                status="READY" if ready else "BLOCKED",
                materializer_available=adapter.materializer is not None,
                actual_validator_kind=adapter.actual_validator_kind,
                actual_validator_kinds=validator_kinds,
                actual_validators_available=validators_available,
                required_auxiliary_source_kinds=(
                    formal_single_operator_required_auxiliary_source_kinds(spec.node)
                ),
                reducer_available=adapter.reducer is not None,
                blocker=None if ready else adapter.blocked_reason,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class FormalSingleOperatorNodeMaterialization:
    schema_version: int
    kind: Literal["formal_single_operator_node_materialization"]
    protocol_sha256: str
    node: FormalSingleOperatorNode
    ordinal: int
    stage: str
    phase: str
    predecessor_source: FormalSingleOperatorJsonBinding | None
    predecessor_completion_sha256: str | None
    protocol_lock_source: FormalSingleOperatorJsonBinding
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    prepared_model_content_authorization_sha256: str | None
    formal_workload_e3a_authorization_sha256: str | None
    formal_workload_e0_authorization_sha256: str | None
    burstgpt_shape_authorization_sha256: str | None
    materialization_source: FormalSingleOperatorJsonBinding
    materialization_sha256: str
    created_ns: int
    auxiliary_sources: tuple[FormalSingleOperatorAuxiliarySourceBinding, ...] = ()
    content_source_binding: FormalContentSourceBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2, 3}
            or self.kind != "formal_single_operator_node_materialization"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator materialization artifact schema differs")
        spec = formal_single_operator_node_spec(self.node)
        if (self.ordinal, self.stage, self.phase) != (
            spec.ordinal,
            spec.stage,
            spec.phase,
        ):
            raise ValueError("single-operator materialization node metadata differs")
        if spec.ordinal == 0:
            if (
                self.predecessor_source is not None
                or self.predecessor_completion_sha256 is not None
            ):
                raise ValueError("preflight cannot have a predecessor")
        elif (
            type(self.predecessor_source) is not FormalSingleOperatorJsonBinding
            or self.predecessor_completion_sha256 is None
        ):
            raise ValueError("non-root materialization requires its predecessor")
        if self.predecessor_completion_sha256 is not None:
            _require_sha256(
                "single-operator predecessor completion",
                self.predecessor_completion_sha256,
            )
        if type(self.protocol_lock_source) is not FormalSingleOperatorJsonBinding:
            raise TypeError(
                "single-operator materialization requires ProtocolLock source"
            )
        for label, digest in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_manifest_sha256),
        ):
            _require_sha256(f"single-operator materialization {label}", digest)
        signed_digests = (
            self.prepared_model_content_authorization_sha256,
            self.formal_workload_e3a_authorization_sha256,
            self.formal_workload_e0_authorization_sha256,
            self.burstgpt_shape_authorization_sha256,
        )
        if self.schema_version in {1, 2}:
            for digest in signed_digests:
                _require_sha256(
                    "single-operator materialization signed content authority",
                    digest,
                )
            if self.content_source_binding is not None:
                raise ValueError("legacy materialization carries trusted content")
        elif (
            any(value is not None for value in signed_digests)
            or type(self.content_source_binding) is not FormalContentSourceBinding
            or self.content_source_binding.mode != "trusted_single_operator"
        ):
            raise ValueError("trusted materialization content lineage differs")
        else:
            self.content_source_binding.reopen()
        if type(self.materialization_source) is not FormalSingleOperatorJsonBinding:
            raise TypeError("single-operator materialization requires a JSON source")
        _require_sha256(
            "single-operator materialization",
            self.materialization_sha256,
        )
        if type(self.created_ns) is not int or self.created_ns < 0:
            raise ValueError("single-operator materialization time is invalid")
        if self.schema_version == 1:
            if self.auxiliary_sources:
                raise ValueError(
                    "legacy single-operator materialization cannot carry auxiliary sources"
                )
            if formal_single_operator_required_auxiliary_source_kinds(self.node):
                raise ValueError(
                    "legacy single-operator materialization lacks required auxiliary sources"
                )
        else:
            _reopen_formal_single_operator_auxiliary_sources(
                self.auxiliary_sources,
                node=self.node,
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
            "ordinal": self.ordinal,
            "stage": self.stage,
            "phase": self.phase,
            "predecessor_source": (
                None
                if self.predecessor_source is None
                else self.predecessor_source.to_dict()
            ),
            "predecessor_completion_sha256": self.predecessor_completion_sha256,
            "protocol_lock_source": self.protocol_lock_source.to_dict(),
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "runtime_authority_manifest_sha256": (
                self.runtime_authority_manifest_sha256
            ),
            "prepared_model_content_authorization_sha256": (
                self.prepared_model_content_authorization_sha256
            ),
            "formal_workload_e3a_authorization_sha256": (
                self.formal_workload_e3a_authorization_sha256
            ),
            "formal_workload_e0_authorization_sha256": (
                self.formal_workload_e0_authorization_sha256
            ),
            "burstgpt_shape_authorization_sha256": (
                self.burstgpt_shape_authorization_sha256
            ),
            "materialization_source": self.materialization_source.to_dict(),
            "materialization_sha256": self.materialization_sha256,
            "created_ns": self.created_ns,
        }
        if self.schema_version in {2, 3}:
            value["auxiliary_sources"] = [
                row.to_dict() for row in self.auxiliary_sources
            ]
        if self.schema_version == 3:
            assert self.content_source_binding is not None
            value["content_source_binding"] = self.content_source_binding.to_dict()
        if include_sha256:
            value["node_materialization_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("single-operator node materialization must be an object")
        schema_version = value.get("schema_version")
        fields = set(cls.__dataclass_fields__) | {"node_materialization_sha256"}
        if schema_version == 1:
            fields.remove("auxiliary_sources")
        if schema_version in {1, 2}:
            fields.remove("content_source_binding")
        row = _strict(
            "single-operator node materialization",
            value,
            fields,
        )
        expected = _require_sha256(
            "single-operator node materialization",
            row.pop("node_materialization_sha256"),
        )
        predecessor = row.pop("predecessor_source")
        row["predecessor_source"] = (
            None
            if predecessor is None
            else FormalSingleOperatorJsonBinding.from_dict(predecessor)
        )
        row["protocol_lock_source"] = FormalSingleOperatorJsonBinding.from_dict(
            row["protocol_lock_source"]
        )
        row["materialization_source"] = FormalSingleOperatorJsonBinding.from_dict(
            row["materialization_source"]
        )
        row["auxiliary_sources"] = tuple(
            FormalSingleOperatorAuxiliarySourceBinding.from_dict(item)
            for item in _array(
                "single-operator materialization auxiliary sources",
                row.get("auxiliary_sources", []),
            )
        )
        raw_content_source = row.get("content_source_binding")
        row["content_source_binding"] = (
            None
            if raw_content_source is None
            else FormalContentSourceBinding.from_dict(raw_content_source)
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != expected:
            raise ValueError("single-operator node materialization digest differs")
        return artifact


@dataclass(frozen=True)
class FormalSingleOperatorStageDecision:
    schema_version: int
    kind: Literal["formal_single_operator_stage_decision"]
    protocol_sha256: str
    node: FormalSingleOperatorNode
    ordinal: int
    stage: str
    phase: str
    predecessor_completion_sha256: str | None
    materialization_sha256: str
    actual_result_set_sha256: str
    decision_kind: str
    next_materialization_source_decision_sha256: str | None
    next_materialization_upstream_receipt_sha256s: tuple[str, ...]
    payload: dict[str, object]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_stage_decision"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator decision schema differs")
        spec = formal_single_operator_node_spec(self.node)
        if (self.ordinal, self.stage, self.phase) != (
            spec.ordinal,
            spec.stage,
            spec.phase,
        ):
            raise ValueError("single-operator decision node metadata differs")
        if (spec.ordinal == 0) != (self.predecessor_completion_sha256 is None):
            raise ValueError("single-operator decision predecessor differs")
        if self.predecessor_completion_sha256 is not None:
            _require_sha256(
                "single-operator decision predecessor",
                self.predecessor_completion_sha256,
            )
        for label, digest in (
            ("materialization", self.materialization_sha256),
            ("actual result set", self.actual_result_set_sha256),
        ):
            _require_sha256(f"single-operator decision {label}", digest)
        draft = FormalSingleOperatorDecisionDraft(
            decision_kind=self.decision_kind,
            next_materialization_source_decision_sha256=(
                self.next_materialization_source_decision_sha256
            ),
            next_materialization_upstream_receipt_sha256s=(
                self.next_materialization_upstream_receipt_sha256s
            ),
            payload=self.payload,
        )
        terminal = spec.ordinal == len(FORMAL_SINGLE_OPERATOR_NODE_SPECS) - 1
        scientific_stop = (
            draft.payload.get("status")
            in _FORMAL_SINGLE_OPERATOR_SCIENTIFIC_STOP_STATUSES
        )
        if scientific_stop:
            reasons = draft.payload.get("reason_codes")
            if (
                type(reasons) is not list
                or not reasons
                or reasons != sorted(set(reasons))
                or any(type(reason) is not str or not reason for reason in reasons)
            ):
                raise ValueError(
                    "scientifically blocked decision needs canonical reason codes"
                )
        if terminal:
            if (
                draft.next_materialization_source_decision_sha256 is not None
                or draft.next_materialization_upstream_receipt_sha256s
            ):
                raise ValueError("terminal E0 decision cannot authorize a future node")
        elif scientific_stop:
            if (
                draft.next_materialization_source_decision_sha256 is not None
                or draft.next_materialization_upstream_receipt_sha256s
            ):
                raise ValueError(
                    "scientifically blocked decision cannot authorize a future node"
                )
        elif (
            draft.next_materialization_source_decision_sha256 is None
            or not draft.next_materialization_upstream_receipt_sha256s
        ):
            raise ValueError(
                "non-terminal decision must bind the exact next materialization"
            )
        object.__setattr__(self, "payload", draft.payload)

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "node": self.node,
            "ordinal": self.ordinal,
            "stage": self.stage,
            "phase": self.phase,
            "predecessor_completion_sha256": self.predecessor_completion_sha256,
            "materialization_sha256": self.materialization_sha256,
            "actual_result_set_sha256": self.actual_result_set_sha256,
            "decision_kind": self.decision_kind,
            "next_materialization_source_decision_sha256": (
                self.next_materialization_source_decision_sha256
            ),
            "next_materialization_upstream_receipt_sha256s": list(
                self.next_materialization_upstream_receipt_sha256s
            ),
            "payload": self.payload,
        }
        if include_sha256:
            value["decision_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator stage decision",
            value,
            set(cls.__dataclass_fields__) | {"decision_sha256"},
        )
        expected = _require_sha256(
            "single-operator stage decision",
            row.pop("decision_sha256"),
        )
        row["next_materialization_upstream_receipt_sha256s"] = tuple(
            _array(
                "single-operator next materialization upstreams",
                row["next_materialization_upstream_receipt_sha256s"],
            )
        )
        decision = cls(**row)  # type: ignore[arg-type]
        if decision.sha256 != expected:
            raise ValueError("single-operator stage decision digest differs")
        return decision


@dataclass(frozen=True)
class FormalSingleOperatorStageCompletion:
    schema_version: int
    kind: Literal["formal_single_operator_stage_completion"]
    protocol_sha256: str
    node: FormalSingleOperatorNode
    ordinal: int
    stage: str
    phase: str
    predecessor_source: FormalSingleOperatorJsonBinding | None
    predecessor_completion_sha256: str | None
    protocol_lock_sha256: str
    node_materialization_source: FormalSingleOperatorJsonBinding
    node_materialization_sha256: str
    materialization_sha256: str
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...]
    actual_result_set_sha256: str
    decision_source: FormalSingleOperatorJsonBinding
    decision_sha256: str
    completed_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_stage_completion"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator completion schema differs")
        spec = formal_single_operator_node_spec(self.node)
        if (self.ordinal, self.stage, self.phase) != (
            spec.ordinal,
            spec.stage,
            spec.phase,
        ):
            raise ValueError("single-operator completion node metadata differs")
        if spec.ordinal == 0:
            if (
                self.predecessor_source is not None
                or self.predecessor_completion_sha256 is not None
            ):
                raise ValueError("preflight completion cannot have a predecessor")
        elif (
            type(self.predecessor_source) is not FormalSingleOperatorJsonBinding
            or self.predecessor_completion_sha256 is None
        ):
            raise ValueError("non-root completion requires its predecessor")
        for label, digest in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("node materialization", self.node_materialization_sha256),
            ("materialization", self.materialization_sha256),
            ("actual result set", self.actual_result_set_sha256),
            ("decision", self.decision_sha256),
        ):
            _require_sha256(f"single-operator completion {label}", digest)
        if self.predecessor_completion_sha256 is not None:
            _require_sha256(
                "single-operator completion predecessor",
                self.predecessor_completion_sha256,
            )
        if (
            type(self.node_materialization_source)
            is not FormalSingleOperatorJsonBinding
            or type(self.decision_source) is not FormalSingleOperatorJsonBinding
        ):
            raise TypeError("single-operator completion sources must be JSON bindings")
        if (
            type(self.actual_results) is not tuple
            or any(
                type(row) is not FormalSingleOperatorValidatedActual
                for row in self.actual_results
            )
            or tuple(row.cell_id for row in self.actual_results)
            != tuple(sorted({row.cell_id for row in self.actual_results}))
        ):
            raise ValueError("single-operator actual results are not canonical")
        expected_actual_set = _content_sha256(
            [row.to_dict() for row in self.actual_results]
        )
        if expected_actual_set != self.actual_result_set_sha256:
            raise ValueError("single-operator actual result-set digest differs")
        if any(row.status != "COMPLETE" for row in self.actual_results):
            raise ValueError("a completed stage cannot contain failed actual results")
        if type(self.completed_ns) is not int or self.completed_ns < 0:
            raise ValueError("single-operator completion time is invalid")

    @cached_property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "node": self.node,
            "ordinal": self.ordinal,
            "stage": self.stage,
            "phase": self.phase,
            "predecessor_source": (
                None
                if self.predecessor_source is None
                else self.predecessor_source.to_dict()
            ),
            "predecessor_completion_sha256": self.predecessor_completion_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "node_materialization_source": (self.node_materialization_source.to_dict()),
            "node_materialization_sha256": self.node_materialization_sha256,
            "materialization_sha256": self.materialization_sha256,
            "actual_results": [row.to_dict() for row in self.actual_results],
            "actual_result_set_sha256": self.actual_result_set_sha256,
            "decision_source": self.decision_source.to_dict(),
            "decision_sha256": self.decision_sha256,
            "completed_ns": self.completed_ns,
        }
        if include_sha256:
            value["completion_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator stage completion",
            value,
            set(cls.__dataclass_fields__) | {"completion_sha256"},
        )
        expected = _require_sha256(
            "single-operator stage completion",
            row.pop("completion_sha256"),
        )
        predecessor = row.pop("predecessor_source")
        row["predecessor_source"] = (
            None
            if predecessor is None
            else FormalSingleOperatorJsonBinding.from_dict(predecessor)
        )
        row["node_materialization_source"] = FormalSingleOperatorJsonBinding.from_dict(
            row["node_materialization_source"]
        )
        row["actual_results"] = tuple(
            FormalSingleOperatorValidatedActual.from_dict(item)
            for item in _array(
                "single-operator actual results",
                row["actual_results"],
            )
        )
        row["decision_source"] = FormalSingleOperatorJsonBinding.from_dict(
            row["decision_source"]
        )
        completion = cls(**row)  # type: ignore[arg-type]
        if completion.sha256 != expected:
            raise ValueError("single-operator stage completion digest differs")
        return completion


@dataclass(frozen=True)
class RebuiltFormalSingleOperatorNodeMaterialization:
    artifact: FormalSingleOperatorNodeMaterialization
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None
    materialization: StageMaterializationReceipt


@dataclass(frozen=True)
class RebuiltFormalSingleOperatorStageCompletion:
    artifact: FormalSingleOperatorStageCompletion
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None
    node_materialization: FormalSingleOperatorNodeMaterialization
    materialization: StageMaterializationReceipt
    decision: FormalSingleOperatorStageDecision


@dataclass(frozen=True)
class FormalSingleOperatorExecutionSource:
    """Exact current-only source consumed by the physical runtime mapper."""

    schema_version: int
    kind: Literal["formal_single_operator_execution_source"]
    protocol_sha256: str
    node: FormalSingleOperatorNode
    ordinal: int
    stage: str
    phase: str
    protocol_lock_source: FormalSingleOperatorJsonBinding
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    prepared_model_content_authorization_sha256: str | None
    formal_workload_e3a_authorization_sha256: str | None
    formal_workload_e0_authorization_sha256: str | None
    burstgpt_shape_authorization_sha256: str | None
    predecessor_completion_source: FormalSingleOperatorJsonBinding | None
    predecessor_completion_sha256: str | None
    predecessor_decision_sha256: str | None
    materialization_source: FormalSingleOperatorJsonBinding
    materialization_sha256: str
    materialization_source_decision_sha256: str
    materialization_upstream_receipt_sha256s: tuple[str, ...]
    auxiliary_sources: tuple[FormalSingleOperatorAuxiliarySourceBinding, ...] = ()
    content_source_binding: FormalContentSourceBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2, 3}
            or self.kind != "formal_single_operator_execution_source"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256
        ):
            raise ValueError("single-operator execution source schema differs")
        spec = formal_single_operator_node_spec(self.node)
        if (self.ordinal, self.stage, self.phase) != (
            spec.ordinal,
            spec.stage,
            spec.phase,
        ):
            raise ValueError("single-operator execution source node differs")
        if type(self.protocol_lock_source) is not FormalSingleOperatorJsonBinding:
            raise TypeError("single-operator execution source lacks ProtocolLock")
        for label, digest in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_manifest_sha256),
            ("materialization", self.materialization_sha256),
            (
                "materialization source decision",
                self.materialization_source_decision_sha256,
            ),
        ):
            _require_sha256(f"single-operator execution source {label}", digest)
        signed_digests = (
            self.prepared_model_content_authorization_sha256,
            self.formal_workload_e3a_authorization_sha256,
            self.formal_workload_e0_authorization_sha256,
            self.burstgpt_shape_authorization_sha256,
        )
        if self.schema_version in {1, 2}:
            for digest in signed_digests:
                _require_sha256(
                    "single-operator execution signed content authority",
                    digest,
                )
            if self.content_source_binding is not None:
                raise ValueError("legacy execution source carries trusted content")
        elif (
            any(value is not None for value in signed_digests)
            or type(self.content_source_binding) is not FormalContentSourceBinding
            or self.content_source_binding.mode != "trusted_single_operator"
        ):
            raise ValueError("trusted execution content lineage differs")
        else:
            self.content_source_binding.reopen()
        for digest in self.materialization_upstream_receipt_sha256s:
            _require_sha256("single-operator execution source upstream", digest)
        if type(self.materialization_source) is not FormalSingleOperatorJsonBinding:
            raise TypeError("single-operator execution source lacks materialization")
        if spec.ordinal == 0:
            if any(
                value is not None
                for value in (
                    self.predecessor_completion_source,
                    self.predecessor_completion_sha256,
                    self.predecessor_decision_sha256,
                )
            ):
                raise ValueError("preflight execution source cannot have predecessor")
        else:
            if (
                type(self.predecessor_completion_source)
                is not FormalSingleOperatorJsonBinding
                or self.predecessor_completion_sha256 is None
                or self.predecessor_decision_sha256 is None
            ):
                raise ValueError("downstream execution source lacks predecessor")
            _require_sha256(
                "single-operator execution source predecessor",
                self.predecessor_completion_sha256,
            )
            _require_sha256(
                "single-operator execution source predecessor decision",
                self.predecessor_decision_sha256,
            )
        if self.schema_version == 1:
            if self.auxiliary_sources:
                raise ValueError(
                    "legacy single-operator execution source cannot carry auxiliaries"
                )
            if formal_single_operator_required_auxiliary_source_kinds(self.node):
                raise ValueError(
                    "legacy single-operator execution source lacks required auxiliaries"
                )
        else:
            _reopen_formal_single_operator_auxiliary_sources(
                self.auxiliary_sources,
                node=self.node,
            )

    def auxiliary_source_binding(
        self,
        source_kind: FormalSingleOperatorAuxiliarySourceKind,
    ) -> FormalSingleOperatorJsonBinding:
        """Return one purpose-bound source, failing closed when it is absent."""

        if source_kind not in FORMAL_SINGLE_OPERATOR_AUXILIARY_SOURCE_KINDS:
            raise ValueError("single-operator auxiliary source kind is not registered")
        matches = tuple(
            row.source
            for row in self.auxiliary_sources
            if row.source_kind == source_kind
        )
        if len(matches) != 1:
            raise ValueError("single-operator execution auxiliary source is absent")
        return matches[0]

    def reopen_auxiliary_source(
        self,
        source_kind: FormalSingleOperatorAuxiliarySourceKind,
    ) -> dict[str, object]:
        binding = self.auxiliary_source_binding(source_kind)
        return binding.reopen(
            label=f"single-operator {source_kind} execution auxiliary source"
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
            "ordinal": self.ordinal,
            "stage": self.stage,
            "phase": self.phase,
            "protocol_lock_source": self.protocol_lock_source.to_dict(),
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "runtime_authority_manifest_sha256": (
                self.runtime_authority_manifest_sha256
            ),
            "prepared_model_content_authorization_sha256": (
                self.prepared_model_content_authorization_sha256
            ),
            "formal_workload_e3a_authorization_sha256": (
                self.formal_workload_e3a_authorization_sha256
            ),
            "formal_workload_e0_authorization_sha256": (
                self.formal_workload_e0_authorization_sha256
            ),
            "burstgpt_shape_authorization_sha256": (
                self.burstgpt_shape_authorization_sha256
            ),
            "predecessor_completion_source": (
                None
                if self.predecessor_completion_source is None
                else self.predecessor_completion_source.to_dict()
            ),
            "predecessor_completion_sha256": self.predecessor_completion_sha256,
            "predecessor_decision_sha256": self.predecessor_decision_sha256,
            "materialization_source": self.materialization_source.to_dict(),
            "materialization_sha256": self.materialization_sha256,
            "materialization_source_decision_sha256": (
                self.materialization_source_decision_sha256
            ),
            "materialization_upstream_receipt_sha256s": list(
                self.materialization_upstream_receipt_sha256s
            ),
        }
        if self.schema_version in {2, 3}:
            value["auxiliary_sources"] = [
                row.to_dict() for row in self.auxiliary_sources
            ]
        if self.schema_version == 3:
            assert self.content_source_binding is not None
            value["content_source_binding"] = self.content_source_binding.to_dict()
        if include_sha256:
            value["execution_source_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("single-operator execution source must be an object")
        schema_version = value.get("schema_version")
        fields = set(cls.__dataclass_fields__) | {"execution_source_sha256"}
        if schema_version == 1:
            fields.remove("auxiliary_sources")
        if schema_version in {1, 2}:
            fields.remove("content_source_binding")
        row = _strict(
            "single-operator execution source",
            value,
            fields,
        )
        expected = _require_sha256(
            "single-operator execution source",
            row.pop("execution_source_sha256"),
        )
        row["protocol_lock_source"] = FormalSingleOperatorJsonBinding.from_dict(
            row["protocol_lock_source"]
        )
        predecessor = row.pop("predecessor_completion_source")
        row["predecessor_completion_source"] = (
            None
            if predecessor is None
            else FormalSingleOperatorJsonBinding.from_dict(predecessor)
        )
        row["materialization_source"] = FormalSingleOperatorJsonBinding.from_dict(
            row["materialization_source"]
        )
        row["materialization_upstream_receipt_sha256s"] = tuple(
            _array(
                "single-operator execution source upstreams",
                row["materialization_upstream_receipt_sha256s"],
            )
        )
        row["auxiliary_sources"] = tuple(
            FormalSingleOperatorAuxiliarySourceBinding.from_dict(item)
            for item in _array(
                "single-operator execution auxiliary sources",
                row.get("auxiliary_sources", []),
            )
        )
        raw_content_source = row.get("content_source_binding")
        row["content_source_binding"] = (
            None
            if raw_content_source is None
            else FormalContentSourceBinding.from_dict(raw_content_source)
        )
        source = cls(**row)  # type: ignore[arg-type]
        if source.sha256 != expected:
            raise ValueError("single-operator execution source digest differs")
        return source


def build_formal_single_operator_execution_source(
    node_materialization_path: str | Path,
) -> FormalSingleOperatorExecutionSource:
    """Deep-reopen one node and project its exact physical execution source."""

    rebuilt = rebuild_formal_single_operator_node_materialization(
        node_materialization_path
    )
    artifact = rebuilt.artifact
    predecessor = rebuilt.predecessor
    return FormalSingleOperatorExecutionSource(
        schema_version=(3 if artifact.schema_version == 3 else 2),
        kind="formal_single_operator_execution_source",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node=artifact.node,
        ordinal=artifact.ordinal,
        stage=artifact.stage,
        phase=artifact.phase,
        protocol_lock_source=artifact.protocol_lock_source,
        protocol_lock_sha256=artifact.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(artifact.runtime_authority_manifest_sha256),
        prepared_model_content_authorization_sha256=(
            artifact.prepared_model_content_authorization_sha256
        ),
        formal_workload_e3a_authorization_sha256=(
            artifact.formal_workload_e3a_authorization_sha256
        ),
        formal_workload_e0_authorization_sha256=(
            artifact.formal_workload_e0_authorization_sha256
        ),
        burstgpt_shape_authorization_sha256=(
            artifact.burstgpt_shape_authorization_sha256
        ),
        predecessor_completion_source=artifact.predecessor_source,
        predecessor_completion_sha256=artifact.predecessor_completion_sha256,
        predecessor_decision_sha256=(
            None if predecessor is None else predecessor.decision.sha256
        ),
        materialization_source=artifact.materialization_source,
        materialization_sha256=artifact.materialization_sha256,
        materialization_source_decision_sha256=(
            rebuilt.materialization.source_decision_sha256
        ),
        materialization_upstream_receipt_sha256s=(
            rebuilt.materialization.upstream_receipt_sha256s
        ),
        auxiliary_sources=artifact.auxiliary_sources,
        content_source_binding=artifact.content_source_binding,
    )


def publish_formal_single_operator_execution_source(
    *,
    node_materialization_path: str | Path,
    output_path: str | Path,
) -> FormalSingleOperatorExecutionSource:
    source = build_formal_single_operator_execution_source(node_materialization_path)
    _publish_canonical_object_no_replace(
        _absolute_normalized_path(
            "single-operator execution source output",
            output_path,
        ),
        source.to_dict(),
    )
    return source


def load_formal_single_operator_execution_source(
    path: str | Path,
) -> FormalSingleOperatorExecutionSource:
    binding = FormalSingleOperatorJsonBinding.bind(
        path,
        label="single-operator execution source",
    )
    source = FormalSingleOperatorExecutionSource.from_dict(
        binding.reopen(label="single-operator execution source")
    )
    protocol_lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(
            label="single-operator execution ProtocolLock"
        )
    )
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="single-operator execution materialization"
        )
    )
    predecessor = (
        None
        if source.predecessor_completion_source is None
        else _rebuild_completion_from_binding(
            source.predecessor_completion_source,
            visited=frozenset({binding.absolute_path}),
        )
    )
    _reopen_formal_single_operator_auxiliary_sources(
        source.auxiliary_sources,
        node=source.node,
    )
    _validate_single_operator_materialization_transition(
        node=source.node,
        predecessor=predecessor,
        protocol_lock=protocol_lock,
        auxiliary_sources=source.auxiliary_sources,
        materialization=materialization,
        message="single-operator execution source changed",
    )
    if (
        protocol_lock.sha256 != source.protocol_lock_sha256
        or protocol_lock.formal_runtime_authority_manifest_sha256
        != source.runtime_authority_manifest_sha256
        or protocol_lock.prepared_model_content_authorization_sha256
        != source.prepared_model_content_authorization_sha256
        or materialization.sha256 != source.materialization_sha256
        or materialization.source_decision_sha256
        != source.materialization_source_decision_sha256
        or materialization.upstream_receipt_sha256s
        != source.materialization_upstream_receipt_sha256s
        or next_formal_single_operator_node(
            None if predecessor is None else predecessor.artifact.node
        )
        != source.node
        or source.predecessor_decision_sha256
        != (None if predecessor is None else predecessor.decision.sha256)
    ):
        raise ValueError("single-operator execution source changed")
    return source


def _rebuild_completion_from_binding(
    source: FormalSingleOperatorJsonBinding,
    *,
    visited: frozenset[str],
) -> RebuiltFormalSingleOperatorStageCompletion:
    path = source.absolute_path
    if path in visited:
        raise ValueError("single-operator completion chain contains a cycle")
    value = source.reopen(label="single-operator predecessor completion")
    completion = FormalSingleOperatorStageCompletion.from_dict(value)
    if _content_sha256(completion.to_dict()) != source.semantic_sha256:
        raise ValueError("single-operator predecessor binding differs")
    return _rebuild_stage_completion(
        completion,
        visited=visited | {path},
    )


def _validate_single_operator_materialization_transition(
    *,
    node: FormalSingleOperatorNode,
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
    auxiliary_sources: tuple[FormalSingleOperatorAuxiliarySourceBinding, ...],
    materialization: StageMaterializationReceipt,
    message: str,
) -> None:
    """Validate the immediate decision edge, including E0's probe edge.

    Every ordinary node consumes the source decision committed by its immediate
    predecessor.  E0 tuning is the sole registered exception: its scientific
    universe is decided by a compatibility probe completed after E6, so the
    receipt source is the deeply validated compatibility-bundle digest.  The
    bundle itself remains bound to the immediate E6 materialization and
    confirmation by the downstream codec.
    """

    if predecessor is None:
        return
    expected_source_decision_sha256 = (
        predecessor.decision.next_materialization_source_decision_sha256
    )
    if node == "e0_tuning":
        if tuple(row.source_kind for row in auxiliary_sources) != ("e0_compatibility",):
            raise ValueError("E0 transition lacks its exact compatibility source")
        from lightcone_spec.experiments.formal_single_operator_downstream import (
            _e0_compatibility_from_auxiliary,
        )

        _compatibility, _authority, bundle_sha256, _evidence_sha256 = (
            _e0_compatibility_from_auxiliary(
                predecessor,
                protocol_lock,
                auxiliary_sources[0].reopen(),
            )
        )
        expected_source_decision_sha256 = bundle_sha256
    if (
        materialization.protocol_lock_sha256
        != predecessor.artifact.protocol_lock_sha256
        or materialization.source_decision_sha256 != expected_source_decision_sha256
        or materialization.upstream_receipt_sha256s
        != predecessor.decision.next_materialization_upstream_receipt_sha256s
    ):
        raise ValueError(message)


def _rebuild_node_materialization(
    artifact: FormalSingleOperatorNodeMaterialization,
    *,
    visited: frozenset[str],
) -> RebuiltFormalSingleOperatorNodeMaterialization:
    protocol_lock = protocol_lock_from_dict(
        artifact.protocol_lock_source.reopen(
            label="single-operator ProtocolLock source"
        )
    )
    if (
        protocol_lock.sha256 != artifact.protocol_lock_sha256
        or protocol_lock.formal_runtime_authority_manifest_sha256
        != artifact.runtime_authority_manifest_sha256
        or protocol_lock.prepared_model_content_authorization_sha256
        != artifact.prepared_model_content_authorization_sha256
        or protocol_lock.formal_workload_e3a_authorization_sha256
        != artifact.formal_workload_e3a_authorization_sha256
        or protocol_lock.formal_workload_e0_authorization_sha256
        != artifact.formal_workload_e0_authorization_sha256
        or protocol_lock.burstgpt_shape_authorization_sha256
        != artifact.burstgpt_shape_authorization_sha256
    ):
        raise ValueError("single-operator ProtocolLock source changed")
    materialization = stage_materialization_receipt_from_dict(
        artifact.materialization_source.reopen(
            label="single-operator materialization receipt"
        )
    )
    if (
        materialization.sha256 != artifact.materialization_sha256
        or materialization.protocol_lock_sha256 != artifact.protocol_lock_sha256
        or materialization.stage != artifact.stage
    ):
        raise ValueError("single-operator materialization receipt changed")
    predecessor = (
        None
        if artifact.predecessor_source is None
        else _rebuild_completion_from_binding(
            artifact.predecessor_source,
            visited=visited,
        )
    )
    expected_node = next_formal_single_operator_node(
        None if predecessor is None else predecessor.artifact.node
    )
    if expected_node != artifact.node:
        raise ValueError("single-operator materialization skips the fixed DAG")
    if predecessor is not None and (
        predecessor.artifact.sha256 != artifact.predecessor_completion_sha256
        or predecessor.artifact.protocol_lock_sha256 != artifact.protocol_lock_sha256
        or predecessor.node_materialization.protocol_lock_source
        != artifact.protocol_lock_source
    ):
        raise ValueError(
            "single-operator materialization differs from its immediate decision"
        )
    _validate_single_operator_materialization_transition(
        node=artifact.node,
        predecessor=predecessor,
        protocol_lock=protocol_lock,
        auxiliary_sources=artifact.auxiliary_sources,
        materialization=materialization,
        message=("single-operator materialization differs from its immediate decision"),
    )
    return RebuiltFormalSingleOperatorNodeMaterialization(
        artifact=artifact,
        predecessor=predecessor,
        materialization=materialization,
    )


def rebuild_formal_single_operator_node_materialization(
    path: str | Path,
) -> RebuiltFormalSingleOperatorNodeMaterialization:
    source = FormalSingleOperatorJsonBinding.bind(
        path,
        label="single-operator node materialization",
    )
    artifact = FormalSingleOperatorNodeMaterialization.from_dict(
        source.reopen(label="single-operator node materialization")
    )
    if _content_sha256(artifact.to_dict()) != source.semantic_sha256:
        raise ValueError("single-operator node materialization binding differs")
    return _rebuild_node_materialization(
        artifact,
        visited=frozenset({source.absolute_path}),
    )


def _rebuild_stage_completion(
    completion: FormalSingleOperatorStageCompletion,
    *,
    visited: frozenset[str],
) -> RebuiltFormalSingleOperatorStageCompletion:
    node_materialization_value = completion.node_materialization_source.reopen(
        label="single-operator completion materialization source"
    )
    node_materialization = FormalSingleOperatorNodeMaterialization.from_dict(
        node_materialization_value
    )
    if (
        node_materialization.sha256 != completion.node_materialization_sha256
        or node_materialization.node != completion.node
        or node_materialization.materialization_sha256
        != completion.materialization_sha256
    ):
        raise ValueError("single-operator completion materialization changed")
    rebuilt_materialization = _rebuild_node_materialization(
        node_materialization,
        visited=visited,
    )
    predecessor = rebuilt_materialization.predecessor
    if (
        completion.predecessor_completion_sha256
        != node_materialization.predecessor_completion_sha256
        or (
            (completion.predecessor_source is None)
            != (node_materialization.predecessor_source is None)
        )
        or (
            completion.predecessor_source is not None
            and completion.predecessor_source != node_materialization.predecessor_source
        )
        or completion.protocol_lock_sha256 != node_materialization.protocol_lock_sha256
    ):
        raise ValueError("single-operator completion predecessor changed")
    expected_cell_ids = tuple(
        cell.cell_id for cell in rebuilt_materialization.materialization.cells
    )
    if tuple(row.cell_id for row in completion.actual_results) != expected_cell_ids:
        raise ValueError("single-operator completion lacks exact cell-result coverage")
    for row in completion.actual_results:
        row.source.reopen(label="single-operator actual result")
        if (
            row.node != completion.node
            or row.stage != completion.stage
            or row.materialization_sha256 != completion.materialization_sha256
        ):
            raise ValueError("single-operator actual result lineage changed")
    decision = FormalSingleOperatorStageDecision.from_dict(
        completion.decision_source.reopen(label="single-operator stage decision")
    )
    if (
        decision.sha256 != completion.decision_sha256
        or decision.node != completion.node
        or decision.predecessor_completion_sha256
        != completion.predecessor_completion_sha256
        or decision.materialization_sha256 != completion.materialization_sha256
        or decision.actual_result_set_sha256 != completion.actual_result_set_sha256
    ):
        raise ValueError("single-operator stage decision changed")
    if completion.completed_ns < node_materialization.created_ns:
        raise ValueError("single-operator completion predates materialization")
    return RebuiltFormalSingleOperatorStageCompletion(
        artifact=completion,
        predecessor=predecessor,
        node_materialization=node_materialization,
        materialization=rebuilt_materialization.materialization,
        decision=decision,
    )


def rebuild_formal_single_operator_stage_completion(
    path: str | Path,
) -> RebuiltFormalSingleOperatorStageCompletion:
    source = FormalSingleOperatorJsonBinding.bind(
        path,
        label="single-operator stage completion",
    )
    completion = FormalSingleOperatorStageCompletion.from_dict(
        source.reopen(label="single-operator stage completion")
    )
    if _content_sha256(completion.to_dict()) != source.semantic_sha256:
        raise ValueError("single-operator stage completion binding differs")
    return _rebuild_stage_completion(
        completion,
        visited=frozenset({source.absolute_path}),
    )


def _materialize_formal_single_operator_node_with_adapter(
    *,
    node: FormalSingleOperatorNode,
    predecessor_completion_path: str | Path | None,
    protocol_lock_source: FormalSingleOperatorJsonBinding,
    protocol_lock: ProtocolLock,
    content_source_binding: FormalContentSourceBinding | None,
    auxiliary_sources: tuple[FormalSingleOperatorAuxiliarySourceBinding, ...],
    adapter: _FormalSingleOperatorClosedMaterializer,
    materialization_output_path: str | Path,
    node_materialization_output_path: str | Path,
    created_ns: int,
) -> RebuiltFormalSingleOperatorNodeMaterialization:
    """Materialize exactly the next node from only its immediate predecessor."""

    spec = formal_single_operator_node_spec(node)
    predecessor = (
        None
        if predecessor_completion_path is None
        else rebuild_formal_single_operator_stage_completion(
            predecessor_completion_path
        )
    )
    if (
        next_formal_single_operator_node(
            None if predecessor is None else predecessor.artifact.node
        )
        != node
    ):
        raise ValueError("single-operator materialization is not the next DAG node")
    if type(created_ns) is not int or created_ns < 0:
        raise ValueError("single-operator materialization time is invalid")
    if type(protocol_lock_source) is not FormalSingleOperatorJsonBinding:
        raise TypeError("single-operator materializer requires ProtocolLock source")
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("single-operator materializer requires exact ProtocolLock")
    if protocol_lock.schema_version == 4:
        if content_source_binding is not None:
            raise ValueError("signed materializer cannot carry trusted content")
    elif (
        type(content_source_binding) is not FormalContentSourceBinding
        or content_source_binding.mode != "trusted_single_operator"
        or content_source_binding.content_sha256
        != protocol_lock.trusted_single_operator_content_bundle_sha256
    ):
        raise ValueError("trusted materializer content differs from ProtocolLock")
    else:
        content_source_binding.reopen()
    _reopen_formal_single_operator_auxiliary_sources(
        auxiliary_sources,
        node=node,
    )
    rebound_lock = protocol_lock_from_dict(
        protocol_lock_source.reopen(label="single-operator ProtocolLock source")
    )
    if rebound_lock != protocol_lock:
        raise ValueError("single-operator ProtocolLock source differs")
    if predecessor is not None and (
        predecessor.node_materialization.protocol_lock_source != protocol_lock_source
        or predecessor.node_materialization.content_source_binding
        != content_source_binding
    ):
        raise ValueError("single-operator node switched ProtocolLock source")
    materialization = adapter(predecessor, protocol_lock, auxiliary_sources)
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("single-operator materializer must return an exact receipt")
    if materialization.stage != spec.stage:
        raise ValueError("single-operator materializer returned another stage")
    _validate_single_operator_materialization_transition(
        node=node,
        predecessor=predecessor,
        protocol_lock=protocol_lock,
        auxiliary_sources=auxiliary_sources,
        materialization=materialization,
        message=("single-operator materializer did not consume the immediate decision"),
    )
    materialization_path = _absolute_normalized_path(
        "single-operator materialization output",
        materialization_output_path,
    )
    artifact_path = _absolute_normalized_path(
        "single-operator node materialization output",
        node_materialization_output_path,
    )
    if materialization_path == artifact_path:
        raise ValueError("single-operator materialization outputs must be distinct")
    _publish_canonical_object_no_replace(
        materialization_path,
        stage_materialization_receipt_to_dict(materialization),
    )
    materialization_source = FormalSingleOperatorJsonBinding.bind(
        materialization_path,
        label="single-operator materialization receipt",
    )
    predecessor_source = (
        None
        if predecessor_completion_path is None
        else FormalSingleOperatorJsonBinding.bind(
            predecessor_completion_path,
            label="single-operator predecessor completion",
        )
    )
    artifact = FormalSingleOperatorNodeMaterialization(
        schema_version=(3 if content_source_binding is not None else 2),
        kind="formal_single_operator_node_materialization",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node=node,
        ordinal=spec.ordinal,
        stage=spec.stage,
        phase=spec.phase,
        predecessor_source=predecessor_source,
        predecessor_completion_sha256=(
            None if predecessor is None else predecessor.artifact.sha256
        ),
        protocol_lock_source=protocol_lock_source,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(
            protocol_lock.formal_runtime_authority_manifest_sha256
        ),
        prepared_model_content_authorization_sha256=(
            protocol_lock.prepared_model_content_authorization_sha256
        ),
        formal_workload_e3a_authorization_sha256=(
            protocol_lock.formal_workload_e3a_authorization_sha256
        ),
        formal_workload_e0_authorization_sha256=(
            protocol_lock.formal_workload_e0_authorization_sha256
        ),
        burstgpt_shape_authorization_sha256=(
            protocol_lock.burstgpt_shape_authorization_sha256
        ),
        materialization_source=materialization_source,
        materialization_sha256=materialization.sha256,
        created_ns=created_ns,
        auxiliary_sources=auxiliary_sources,
        content_source_binding=content_source_binding,
    )
    _publish_canonical_object_no_replace(artifact_path, artifact.to_dict())
    return rebuild_formal_single_operator_node_materialization(artifact_path)


def materialize_formal_single_operator_node(
    *,
    node: FormalSingleOperatorNode,
    predecessor_completion_path: str | Path | None,
    protocol_lock_path: str | Path | None,
    content_source_path: str | Path | None = None,
    materialization_output_path: str | Path,
    node_materialization_output_path: str | Path,
    created_ns: int,
    auxiliary_source_paths: Mapping[str, str | Path] | None = None,
    require_capacity_available: bool = True,
    revalidate_runtime_observations: bool = True,
) -> RebuiltFormalSingleOperatorNodeMaterialization:
    """Dispatch the next node through the closed code-owned materializer map."""

    if (
        type(require_capacity_available) is not bool
        or type(revalidate_runtime_observations) is not bool
    ):
        raise TypeError("single-operator runtime replay policies must be boolean")
    spec = formal_single_operator_node_spec(node)
    predecessor = (
        None
        if predecessor_completion_path is None
        else rebuild_formal_single_operator_stage_completion(
            predecessor_completion_path
        )
    )
    if (
        next_formal_single_operator_node(
            None if predecessor is None else predecessor.artifact.node
        )
        != node
    ):
        raise ValueError("single-operator materialization is not the next DAG node")
    adapter = _CLOSED_NODE_ADAPTERS[spec.node]
    if adapter.materializer is None or adapter.blocked_reason is not None:
        raise FormalSingleOperatorStageBlocked(
            adapter.blocked_reason or _BLOCKED_NOT_CONNECTED
        )
    if spec.ordinal == 0:
        if predecessor is not None or protocol_lock_path is None:
            raise ValueError("preflight requires only its explicit ProtocolLock source")
        protocol_lock_source = FormalSingleOperatorJsonBinding.bind(
            protocol_lock_path,
            label="single-operator initial ProtocolLock",
        )
    else:
        if predecessor is None:
            raise ValueError("non-root node requires its immediate predecessor")
        if protocol_lock_path is not None:
            raise ValueError(
                "downstream node cannot replace the inherited ProtocolLock"
            )
        protocol_lock_source = predecessor.node_materialization.protocol_lock_source
    protocol_lock = protocol_lock_from_dict(
        protocol_lock_source.reopen(label="single-operator ProtocolLock")
    )
    if protocol_lock.schema_version != 5:
        raise ValueError(
            "legacy ProtocolLock schema 4 is read-only and cannot materialize nodes"
        )
    if protocol_lock.schema_version == 5 and spec.ordinal == 0:
        from lightcone_spec.experiments.formal_single_operator_protocol_lock import (
            revalidate_trusted_single_operator_protocol_lock,
        )

        if content_source_path is None:
            raise ValueError("trusted preflight requires --content-source")
        revalidate_trusted_single_operator_protocol_lock(
            protocol_lock,
            expected_content_bundle_path=content_source_path,
            require_capacity_available=require_capacity_available,
            revalidate_runtime_observations=revalidate_runtime_observations,
        )
    if spec.ordinal == 0:
        if protocol_lock.schema_version == 5:
            assert content_source_path is not None
            content_source_binding = (
                FormalContentSourceBinding.bind_trusted_single_operator(
                    str(content_source_path)
                )
            )
        else:
            if content_source_path is not None:
                raise ValueError("signed preflight does not accept --content-source")
            content_source_binding = None
    else:
        if content_source_path is not None:
            raise ValueError("downstream node cannot replace inherited content")
        assert predecessor is not None
        content_source_binding = predecessor.node_materialization.content_source_binding
    if protocol_lock.schema_version == 5 and spec.ordinal != 0:
        from lightcone_spec.experiments.formal_single_operator_protocol_lock import (
            revalidate_trusted_single_operator_protocol_lock,
        )

        if (
            type(content_source_binding) is not FormalContentSourceBinding
            or content_source_binding.trusted_single_operator is None
        ):
            raise ValueError("trusted materializer lacks path-bound content")
        revalidate_trusted_single_operator_protocol_lock(
            protocol_lock,
            expected_content_bundle_path=(
                content_source_binding.trusted_single_operator.absolute_path
            ),
            require_capacity_available=require_capacity_available,
            revalidate_runtime_observations=revalidate_runtime_observations,
        )
    auxiliary_sources = bind_formal_single_operator_auxiliary_sources(
        node=node,
        source_paths=auxiliary_source_paths,
    )
    return _materialize_formal_single_operator_node_with_adapter(
        node=node,
        predecessor_completion_path=predecessor_completion_path,
        protocol_lock_source=protocol_lock_source,
        protocol_lock=protocol_lock,
        content_source_binding=content_source_binding,
        auxiliary_sources=auxiliary_sources,
        adapter=adapter.materializer,
        materialization_output_path=materialization_output_path,
        node_materialization_output_path=node_materialization_output_path,
        created_ns=created_ns,
    )


def _reduce_formal_single_operator_node_with_adapters(
    *,
    node_materialization_path: str | Path,
    actual_result_paths: Mapping[str, str | Path],
    actual_validators: Mapping[
        FormalSingleOperatorActualValidatorKind,
        FormalSingleOperatorActualResultValidator,
    ],
    adapter: FormalSingleOperatorReduceAdapter,
    decision_output_path: str | Path,
    completion_output_path: str | Path,
    completed_ns: int,
) -> RebuiltFormalSingleOperatorStageCompletion:
    """Reduce one node from exact source-validated current actual results."""

    rebuilt = rebuild_formal_single_operator_node_materialization(
        node_materialization_path
    )
    artifact = rebuilt.artifact
    materialization = rebuilt.materialization
    if type(actual_result_paths) is not dict:
        raise TypeError("single-operator actual-result paths must be an exact mapping")
    if type(actual_validators) is not dict:
        raise TypeError("single-operator actual validators must be an exact mapping")
    expected_ids = tuple(cell.cell_id for cell in materialization.cells)
    if set(actual_result_paths) != set(expected_ids):
        raise FormalSingleOperatorStageBlocked(
            f"{artifact.node} lacks exact current actual-result coverage"
        )
    routed_kinds = tuple(
        dict.fromkeys(
            formal_single_operator_cell_validator_kind(
                node=artifact.node,
                cell=cell,
            )
            for cell in materialization.cells
        )
    )
    if set(actual_validators) != set(routed_kinds):
        raise ValueError("single-operator actual validator coverage differs")
    validated: list[FormalSingleOperatorValidatedActual] = []
    cells = {cell.cell_id: cell for cell in materialization.cells}
    for cell_id in expected_ids:
        cell = cells[cell_id]
        route_kind = formal_single_operator_cell_validator_kind(
            node=artifact.node,
            cell=cell,
        )
        actual_validator = actual_validators[route_kind]
        validator_kind = _require_text(
            "single-operator actual validator",
            actual_validator.validator_kind,
        )
        validator_protocol = _require_sha256(
            "single-operator actual validator protocol",
            actual_validator.protocol_sha256,
        )
        path = _absolute_normalized_path(
            "single-operator actual result",
            actual_result_paths[cell_id],
        )
        before = FormalSingleOperatorJsonBinding.bind(
            path,
            label="single-operator actual result",
        )
        validation = actual_validator.validate(
            path=path,
            node=formal_single_operator_node_spec(artifact.node),
            materialization=materialization,
            cell=cell,
        )
        if type(validation) is not FormalSingleOperatorActualValidation:
            raise TypeError("single-operator validator must return an exact validation")
        after = FormalSingleOperatorJsonBinding.bind(
            path,
            label="single-operator actual result after validation",
        )
        if after != before:
            raise RuntimeError(
                "single-operator actual result changed during validation"
            )
        validated.append(
            FormalSingleOperatorValidatedActual(
                node=artifact.node,
                stage=artifact.stage,
                materialization_sha256=materialization.sha256,
                cell_id=cell_id,
                status=validation.status,
                started_ns=validation.started_ns,
                finished_ns=validation.finished_ns,
                result_identity_sha256=validation.result_identity_sha256,
                validator_kind=validator_kind,
                validator_protocol_sha256=validator_protocol,
                source=before,
                reducer_payload=validation.reducer_payload,
            )
        )
    actual_results = tuple(validated)
    failed = tuple(row.cell_id for row in actual_results if row.status != "COMPLETE")
    if failed:
        raise FormalSingleOperatorStageBlocked(
            f"{artifact.node} has failed actual results: {','.join(failed)}"
        )
    actual_set_sha256 = _content_sha256([row.to_dict() for row in actual_results])
    draft = adapter(
        rebuilt.predecessor,
        materialization,
        actual_results,
    )
    if type(draft) is not FormalSingleOperatorDecisionDraft:
        raise TypeError("single-operator reducer must return an exact decision draft")
    decision = FormalSingleOperatorStageDecision(
        schema_version=1,
        kind="formal_single_operator_stage_decision",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node=artifact.node,
        ordinal=artifact.ordinal,
        stage=artifact.stage,
        phase=artifact.phase,
        predecessor_completion_sha256=artifact.predecessor_completion_sha256,
        materialization_sha256=materialization.sha256,
        actual_result_set_sha256=actual_set_sha256,
        decision_kind=draft.decision_kind,
        next_materialization_source_decision_sha256=(
            draft.next_materialization_source_decision_sha256
        ),
        next_materialization_upstream_receipt_sha256s=(
            draft.next_materialization_upstream_receipt_sha256s
        ),
        payload=draft.payload,
    )
    decision_path = _absolute_normalized_path(
        "single-operator decision output",
        decision_output_path,
    )
    completion_path = _absolute_normalized_path(
        "single-operator completion output",
        completion_output_path,
    )
    node_materialization_source = FormalSingleOperatorJsonBinding.bind(
        node_materialization_path,
        label="single-operator node materialization",
    )
    if (
        len(
            {
                decision_path,
                completion_path,
                Path(node_materialization_source.absolute_path),
            }
        )
        != 3
    ):
        raise ValueError("single-operator stage output paths must be distinct")
    _publish_canonical_object_no_replace(decision_path, decision.to_dict())
    decision_source = FormalSingleOperatorJsonBinding.bind(
        decision_path,
        label="single-operator stage decision",
    )
    if type(completed_ns) is not int or completed_ns < artifact.created_ns:
        raise ValueError("single-operator completion time is invalid")
    completion = FormalSingleOperatorStageCompletion(
        schema_version=1,
        kind="formal_single_operator_stage_completion",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node=artifact.node,
        ordinal=artifact.ordinal,
        stage=artifact.stage,
        phase=artifact.phase,
        predecessor_source=artifact.predecessor_source,
        predecessor_completion_sha256=artifact.predecessor_completion_sha256,
        protocol_lock_sha256=artifact.protocol_lock_sha256,
        node_materialization_source=node_materialization_source,
        node_materialization_sha256=artifact.sha256,
        materialization_sha256=materialization.sha256,
        actual_results=actual_results,
        actual_result_set_sha256=actual_set_sha256,
        decision_source=decision_source,
        decision_sha256=decision.sha256,
        completed_ns=completed_ns,
    )
    _publish_canonical_object_no_replace(completion_path, completion.to_dict())
    return rebuild_formal_single_operator_stage_completion(completion_path)


def _formal_single_operator_actual_validator(
    *,
    rebuilt: RebuiltFormalSingleOperatorNodeMaterialization,
    kind: FormalSingleOperatorActualValidatorKind,
    repository_root: str | Path | None,
) -> FormalSingleOperatorActualResultValidator:
    """Construct the one code-owned validator for an actual-result route.

    Keeping this factory shared by per-cell admission and whole-node reduction
    prevents the resident operator from accepting a result under a weaker
    contract than the later reducer.
    """

    protocol_lock = protocol_lock_from_dict(
        rebuilt.artifact.protocol_lock_source.reopen(
            label="single-operator actual ProtocolLock"
        )
    )
    if kind == "preflight":
        return _FormalSingleOperatorPreflightActualValidator()
    if kind == "run_manifest":
        if repository_root is None:
            raise ValueError(
                "serving actual validation requires an explicit repository root"
            )
        return FormalSingleOperatorRunManifestActualValidator(str(repository_root))
    if kind == "onlinespec_run_manifest":
        if repository_root is None:
            raise ValueError(
                "OnlineSPEC actual validation requires an explicit repository root"
            )
        return FormalSingleOperatorOnlineSpecRunManifestActualValidator(
            str(repository_root)
        )
    if kind == "profiler_terminal":
        return FormalSingleOperatorProfilerActualValidator()
    if kind == "e5_failure_terminal":
        return FormalSingleOperatorE5FailureActualValidator()
    if kind == "e6_interface_preflight":
        return FormalSingleOperatorE6InterfacePreflightActualValidator(protocol_lock)
    if kind == "e0_compatibility_terminal":
        if rebuilt.predecessor is None:
            raise ValueError("E0 compatibility validator lacks E6 predecessor")
        compatibility_sources = tuple(
            row.source
            for row in rebuilt.artifact.auxiliary_sources
            if row.source_kind == "e0_compatibility"
        )
        if len(compatibility_sources) != 1:
            raise ValueError("E0 compatibility source coverage differs")
        return FormalSingleOperatorE0CompatibilityActualValidator(
            protocol_lock=protocol_lock,
            predecessor=rebuilt.predecessor,
            compatibility_source=compatibility_sources[0],
        )
    raise AssertionError("single-operator actual validator is not implemented")


def validate_formal_single_operator_cell_actual(
    *,
    node_materialization_path: str | Path,
    cell_id: str,
    actual_result_path: str | Path,
    repository_root: str | Path | None = None,
) -> FormalSingleOperatorValidatedActual:
    """Deep-validate one materialized cell without reducing its whole node.

    This is the admission boundary used by the non-LLM cell worker.  It binds
    the exact current materialization, routes by the cell task, reopens the
    result before and after validation, and returns the same immutable row
    later consumed by :func:`reduce_formal_single_operator_node`.
    """

    rebuilt = rebuild_formal_single_operator_node_materialization(
        node_materialization_path
    )
    matches = tuple(
        cell for cell in rebuilt.materialization.cells if cell.cell_id == cell_id
    )
    if len(matches) != 1:
        raise ValueError("single-operator actual names another or duplicate cell")
    cell = matches[0]
    kind = formal_single_operator_cell_validator_kind(
        node=rebuilt.artifact.node,
        cell=cell,
    )
    validator = _formal_single_operator_actual_validator(
        rebuilt=rebuilt,
        kind=kind,
        repository_root=repository_root,
    )
    path = _absolute_normalized_path(
        "single-operator actual result",
        actual_result_path,
    )
    before = FormalSingleOperatorJsonBinding.bind(
        path,
        label="single-operator actual result",
    )
    validation = validator.validate(
        path=path,
        node=formal_single_operator_node_spec(rebuilt.artifact.node),
        materialization=rebuilt.materialization,
        cell=cell,
    )
    if type(validation) is not FormalSingleOperatorActualValidation:
        raise TypeError("single-operator validator must return an exact validation")
    after = FormalSingleOperatorJsonBinding.bind(
        path,
        label="single-operator actual result after validation",
    )
    if after != before:
        raise RuntimeError("single-operator actual result changed during validation")
    return FormalSingleOperatorValidatedActual(
        node=rebuilt.artifact.node,
        stage=rebuilt.artifact.stage,
        materialization_sha256=rebuilt.materialization.sha256,
        cell_id=cell_id,
        status=validation.status,
        started_ns=validation.started_ns,
        finished_ns=validation.finished_ns,
        result_identity_sha256=validation.result_identity_sha256,
        validator_kind=_require_text(
            "single-operator actual validator",
            validator.validator_kind,
        ),
        validator_protocol_sha256=_require_sha256(
            "single-operator actual validator protocol",
            validator.protocol_sha256,
        ),
        source=before,
        reducer_payload=validation.reducer_payload,
    )


def reduce_formal_single_operator_node(
    *,
    node_materialization_path: str | Path,
    actual_result_paths: Mapping[str, str | Path],
    repository_root: str | Path | None = None,
    decision_output_path: str | Path,
    completion_output_path: str | Path,
    completed_ns: int,
) -> RebuiltFormalSingleOperatorStageCompletion:
    """Dispatch the current result through the closed validator/reducer map."""

    rebuilt = rebuild_formal_single_operator_node_materialization(
        node_materialization_path
    )
    adapter = _CLOSED_NODE_ADAPTERS[rebuilt.artifact.node]
    if adapter.reducer is None or adapter.blocked_reason is not None:
        raise FormalSingleOperatorStageBlocked(
            adapter.blocked_reason or _BLOCKED_NOT_CONNECTED
        )
    routed_kinds = tuple(
        dict.fromkeys(
            formal_single_operator_cell_validator_kind(
                node=rebuilt.artifact.node,
                cell=cell,
            )
            for cell in rebuilt.materialization.cells
        )
    )
    requires_repository = bool(
        {"run_manifest", "onlinespec_run_manifest"} & set(routed_kinds)
    )
    if requires_repository and repository_root is None:
        raise ValueError("serving-node reduction requires an explicit repository root")
    validators: dict[
        FormalSingleOperatorActualValidatorKind,
        FormalSingleOperatorActualResultValidator,
    ] = {}
    for kind in routed_kinds:
        validators[kind] = _formal_single_operator_actual_validator(
            rebuilt=rebuilt,
            kind=kind,
            repository_root=repository_root,
        )
    return _reduce_formal_single_operator_node_with_adapters(
        node_materialization_path=node_materialization_path,
        actual_result_paths=actual_result_paths,
        actual_validators=validators,
        adapter=adapter.reducer,
        decision_output_path=decision_output_path,
        completion_output_path=completion_output_path,
        completed_ns=completed_ns,
    )


__all__ = [
    "FORMAL_SINGLE_OPERATOR_NODE_ORDER",
    "FORMAL_SINGLE_OPERATOR_NODE_SPECS",
    "FORMAL_SINGLE_OPERATOR_STAGE_ARTIFACT_MAX_BYTES",
    "FORMAL_SINGLE_OPERATOR_STAGE_MODE",
    "FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256",
    "FormalSingleOperatorExecutionSource",
    "FormalSingleOperatorJsonBinding",
    "FormalSingleOperatorNode",
    "FormalSingleOperatorNodeMaterialization",
    "FormalSingleOperatorNodeReadiness",
    "FormalSingleOperatorNodeSpec",
    "FormalSingleOperatorPreflightActualReceipt",
    "FormalSingleOperatorStageBlocked",
    "FormalSingleOperatorStageCompletion",
    "FormalSingleOperatorStageDecision",
    "FormalSingleOperatorValidatedActual",
    "RebuiltFormalSingleOperatorNodeMaterialization",
    "RebuiltFormalSingleOperatorStageCompletion",
    "build_formal_single_operator_execution_source",
    "formal_single_operator_node_readiness",
    "formal_single_operator_node_spec",
    "load_formal_single_operator_execution_source",
    "materialize_formal_single_operator_node",
    "next_formal_single_operator_node",
    "publish_formal_single_operator_execution_source",
    "publish_formal_single_operator_json_artifact",
    "publish_formal_single_operator_preflight_actual",
    "rebuild_formal_single_operator_node_materialization",
    "rebuild_formal_single_operator_stage_completion",
    "reduce_formal_single_operator_node",
    "validate_formal_single_operator_cell_actual",
]
