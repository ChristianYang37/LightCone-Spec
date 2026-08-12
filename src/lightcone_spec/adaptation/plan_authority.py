"""Path-bound raw authority for backend-native trainable parameter plans.

The serialized plan in the authority manifest is never trusted by itself.
Every bind and replay reopens the exact model lock, serialized inventory mirror,
manifest, and their canonical SHA-256 sidecars without following symlinks.  The
formal parameter metadata instead comes from the first-party prepared-snapshot
content authority's safetensors index/header replay.  The existing backend
selector is rerun over that allocation-free metadata and exact-compared against
the resulting entries, frozen names, state layout, and allocation-memory digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.locking.prepared_models import (
    PreparedModelContentAuthorityBinding,
    PreparedModelContentAuthorityBlocked,
    PreparedModelContentAuthorityResult,
    prepared_model_content_authority_from_dict,
    prepared_model_content_authority_to_dict,
    require_prepared_model_content_release_manifest_sha256,
    revalidate_prepared_model_content_authority,
)

from .parameters import (
    TRAINABLE_PLAN_OPTIMIZERS,
    TRAINABLE_PLAN_REDUCER_PROTOCOL_SHA256,
    DFlashParameterPlan,
    DSparkParameterPlan,
    NativeLayerParameterPlan,
    ParameterEntry,
    TrainablePlan,
)

_SHA256_LENGTH = 64
_REVISION_LENGTH = 40
_RAW_ROLES = frozenset(
    {
        "trainable_plan_authority_manifest",
        "trainable_plan_model_lock",
        "prepared_drafter_parameter_inventory",
        "trainable_plan_run_config",
        "trainable_plan_split",
        "trainable_plan_cell",
    }
)
_FLOAT_DTYPES = frozenset(
    {
        "torch.float16",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
    }
)
_PREPARED_DTYPES = _FLOAT_DTYPES | {
    "torch.bool",
    "torch.uint8",
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.complex64",
    "torch.complex128",
}
_RAW_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "path",
        "sidecar_path",
        "semantic_sha256",
        "file_sha256",
        "sidecar_file_sha256",
        "size",
        "sidecar_size",
    }
)
_PARAMETER_FIELDS = frozenset({"name", "shape", "dtype", "ownership"})
_PREPARED_INVENTORY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "model_lock_sha256",
        "drafter_model_id",
        "prepared_drafter_revision",
        "dspark_native_heads",
        "parameters",
    }
)
_MODEL_LOCK_FIELDS = frozenset({"schema_version", "models"})
_LOCKED_MODEL_FIELDS = frozenset({"model_id", "revision"})
_DSPARK_HEAD_FIELDS = frozenset({"w1_name", "w2_name", "acceptance_name"})
_DSPARK_W1 = re.compile(r"(?:^|\.)markov_head\.markov_w1\.weight$")
_DSPARK_W2 = re.compile(r"(?:^|\.)markov_head\.markov_w2\.weight$")
_DSPARK_ACCEPTANCE_WEIGHT = re.compile(r"(?:^|\.)confidence_head\.proj\.weight$")
_MEMORY_PREDICTION_FIELDS = frozenset(
    {
        "optimizer",
        "active_merged",
        "masters",
        "gradients",
        "optimizer_first",
        "optimizer_second",
        "candidate",
        "staging",
        "merge_scratch",
        "resident_bytes",
        "peak_bytes",
    }
)
_CELL_FIELDS = frozenset({"identity", "resources", "status", "reason_code", "reason"})
_RESOURCE_FIELDS = frozenset(
    {"gpu_uuids", "ports", "cache_root", "evidence_root", "workload_class"}
)
_CELL_IDENTITY_FIELDS = frozenset(
    {
        "experiment",
        "model",
        "backend",
        "task",
        "method",
        "scope",
        "rank",
        "alpha_over_rank",
        "optimizer",
        "learning_rate",
        "schedule",
        "context",
        "regime",
        "width",
        "arrival",
        "slo",
        "cohort",
        "topology",
        "seed",
        "block",
        "gpu_uuids",
        "parameterization",
        "variant",
        "concurrency",
        "load_factor",
        "cohort_count",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "reducer_protocol_sha256",
        "model_lock_artifact",
        "prepared_drafter_artifact",
        "run_config_artifact",
        "split_artifact",
        "cell_artifact",
        "model_lock_sha256",
        "prepared_model_content_authority_sha256",
        "prepared_model_content_manifest_sha256",
        "prepared_drafter_inventory_sha256",
        "run_config_sha256",
        "split_sha256",
        "cell_id",
        "cell_declaration_sha256",
        "target_model_id",
        "target_revision",
        "drafter_model_id",
        "prepared_drafter_revision",
        "method",
        "backend",
        "mode",
        "scope",
        "rank",
        "lora_alpha",
        "optimizer",
        "dspark_native_heads",
        "entries",
        "frozen_names",
        "state_layout",
        "trainable_parameter_count",
        "entries_sha256",
        "frozen_names_sha256",
        "state_layout_sha256",
        "optimizer_memory_prediction",
        "optimizer_memory_sha256",
        "allocation_memory_sha256",
        "trainable_plan_sha256",
    }
)
_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "manifest",
        "model_lock",
        "prepared_drafter",
        "run_config",
        "split",
        "cell",
        "prepared_model_content_authority",
        "reducer_protocol_sha256",
        "model_lock_sha256",
        "prepared_model_content_manifest_sha256",
        "prepared_drafter_inventory_sha256",
        "run_config_sha256",
        "split_sha256",
        "cell_id",
        "cell_declaration_sha256",
        "target_model_id",
        "target_revision",
        "drafter_model_id",
        "prepared_drafter_revision",
        "method",
        "backend",
        "mode",
        "scope",
        "rank",
        "lora_alpha",
        "optimizer",
        "dspark_native_heads",
        "entries_sha256",
        "frozen_names_sha256",
        "state_layout_sha256",
        "optimizer_memory_sha256",
        "allocation_memory_sha256",
        "trainable_plan_sha256",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: object) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be lower-case SHA-256")
    return value


def _require_revision(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _REVISION_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be an immutable lower-case Git revision")
    return value


def _strict_text(name: str, value: object) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _strict_object(
    name: str, value: object, expected_fields: frozenset[str]
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    if set(value) != expected_fields:
        missing = sorted(expected_fields - set(value))
        unknown = sorted(set(value) - expected_fields)
        raise ValueError(f"{name} fields differ: missing={missing}, unknown={unknown}")
    return value


def _strict_list(name: str, value: object) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


def _nullable_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be null or a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nullable_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _strict_text(name, value)


def _nullable_positive_float(name: str, value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be null or a finite positive number")
    return float(value)


def _strict_text_list(name: str, value: object) -> tuple[str, ...]:
    result = tuple(
        _strict_text(f"{name} item", item) for item in _strict_list(name, value)
    )
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _strict_int_list(name: str, value: object) -> tuple[int, ...]:
    result = tuple(
        _positive_int(f"{name} item", item) for item in _strict_list(name, value)
    )
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        value[key] = item
    return value


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
        return
    if type(value) is str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("unpaired JSON surrogate is forbidden")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            _validate_json_value(key)
            _validate_json_value(item)
        return
    raise TypeError(f"unsupported strict JSON value {type(value).__name__}")


def _parse_json(body: bytes, *, label: str) -> object:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    _validate_json_value(value)
    return value


def _resolved_absolute_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise ValueError(f"{label} path must be absolute, resolved, and symlink-free")
    return path


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    _resolved_absolute_path(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is not a readable regular file") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        reopened = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(reopened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or _stat_identity(opened) != _stat_identity(reopened)
            or _stat_identity(reopened) != _stat_identity(current)
            or reopened.st_size != len(body)
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class TrainablePlanRawJsonBinding:
    """Raw-byte and canonical identity for one JSON file and exact sidecar."""

    schema_version: int
    role: str
    path: str
    sidecar_path: str
    semantic_sha256: str
    file_sha256: str
    sidecar_file_sha256: str
    size: int
    sidecar_size: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only trainable-plan raw binding schema v1 is supported")
        if type(self.role) is not str or self.role not in _RAW_ROLES:
            raise ValueError("trainable-plan raw binding role is unsupported")
        if type(self.path) is not str or type(self.sidecar_path) is not str:
            raise TypeError("bound JSON paths must be strings")
        source = _resolved_absolute_path(self.path, label="bound JSON source")
        sidecar = _resolved_absolute_path(self.sidecar_path, label="bound JSON sidecar")
        if sidecar != Path(f"{source}.sha256"):
            raise ValueError("bound JSON sidecar path is not exact")
        for name in ("semantic_sha256", "file_sha256", "sidecar_file_sha256"):
            _require_sha256(f"raw binding {name}", getattr(self, name))
        if type(self.size) is not int or self.size < 1:
            raise ValueError("bound JSON size must be positive")
        if type(self.sidecar_size) is not int or self.sidecar_size != 65:
            raise ValueError("bound JSON sidecar must be one SHA-256 line")

    @classmethod
    def from_path(cls, path: str | Path, *, role: str) -> TrainablePlanRawJsonBinding:
        if role not in _RAW_ROLES:
            raise ValueError("trainable-plan raw binding role is unsupported")
        source = _resolved_absolute_path(path, label=role)
        sidecar = _resolved_absolute_path(
            Path(f"{source}.sha256"), label=f"{role} sidecar"
        )
        body = _regular_file_bytes(source, label=role)
        sidecar_body = _regular_file_bytes(sidecar, label=f"{role} sidecar")
        value = _parse_json(body, label=role)
        semantic_sha256 = _content_sha256(value)
        if sidecar_body != f"{semantic_sha256}\n".encode("ascii"):
            raise ValueError(f"{role} sidecar differs from canonical JSON")
        return cls(
            schema_version=1,
            role=role,
            path=str(source),
            sidecar_path=str(sidecar),
            semantic_sha256=semantic_sha256,
            file_sha256=hashlib.sha256(body).hexdigest(),
            sidecar_file_sha256=hashlib.sha256(sidecar_body).hexdigest(),
            size=len(body),
            sidecar_size=len(sidecar_body),
        )

    def load(self) -> object:
        source = Path(self.path)
        sidecar = Path(self.sidecar_path)
        body = _regular_file_bytes(source, label=f"bound {self.role}")
        sidecar_body = _regular_file_bytes(sidecar, label=f"bound {self.role} sidecar")
        value = _parse_json(body, label=f"bound {self.role}")
        semantic_sha256 = _content_sha256(value)
        if (
            len(body) != self.size
            or len(sidecar_body) != self.sidecar_size
            or hashlib.sha256(body).hexdigest() != self.file_sha256
            or hashlib.sha256(sidecar_body).hexdigest() != self.sidecar_file_sha256
            or semantic_sha256 != self.semantic_sha256
            or sidecar_body != f"{semantic_sha256}\n".encode("ascii")
        ):
            raise RuntimeError(f"bound {self.role} or sidecar changed")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "path": self.path,
            "sidecar_path": self.sidecar_path,
            "semantic_sha256": self.semantic_sha256,
            "file_sha256": self.file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "size": self.size,
            "sidecar_size": self.sidecar_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrainablePlanRawJsonBinding:
        row = _strict_object("trainable-plan raw binding", value, _RAW_BINDING_FIELDS)
        return cls(
            schema_version=row["schema_version"],
            role=row["role"],
            path=row["path"],
            sidecar_path=row["sidecar_path"],
            semantic_sha256=row["semantic_sha256"],
            file_sha256=row["file_sha256"],
            sidecar_file_sha256=row["sidecar_file_sha256"],
            size=row["size"],
            sidecar_size=row["sidecar_size"],
        )

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())


@dataclass(frozen=True)
class PreparedParameterMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    ownership: Literal["sharded", "replicated"]

    def __post_init__(self) -> None:
        _strict_text("prepared parameter name", self.name)
        if type(self.shape) is not tuple or any(
            type(dimension) is not int or dimension < 1 for dimension in self.shape
        ):
            raise ValueError(
                "prepared parameter shape must contain only positive integers"
            )
        if self.dtype not in _PREPARED_DTYPES:
            raise ValueError("prepared parameter dtype is unsupported")
        if self.ownership not in {"sharded", "replicated"}:
            raise ValueError("prepared parameter ownership is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "ownership": self.ownership,
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedParameterMetadata:
        row = _strict_object("prepared parameter", value, _PARAMETER_FIELDS)
        shape = _strict_list("prepared parameter shape", row["shape"])
        return cls(
            name=row["name"],
            shape=tuple(shape),
            dtype=row["dtype"],
            ownership=row["ownership"],
        )


@dataclass(frozen=True)
class DSparkNativeHeadNames:
    w1_name: str
    w2_name: str
    acceptance_name: str

    def __post_init__(self) -> None:
        names = (self.w1_name, self.w2_name, self.acceptance_name)
        for name in names:
            _strict_text("DSpark native head name", name)
        if len(set(names)) != 3:
            raise ValueError("DSpark native head names must be unique")

    def to_dict(self) -> dict[str, str]:
        return {
            "w1_name": self.w1_name,
            "w2_name": self.w2_name,
            "acceptance_name": self.acceptance_name,
        }

    @classmethod
    def from_dict(cls, value: object) -> DSparkNativeHeadNames:
        row = _strict_object("DSpark native heads", value, _DSPARK_HEAD_FIELDS)
        return cls(
            w1_name=row["w1_name"],
            w2_name=row["w2_name"],
            acceptance_name=row["acceptance_name"],
        )


@dataclass(frozen=True)
class PreparedDrafterParameterInventory:
    schema_version: int
    kind: Literal["prepared_drafter_parameter_inventory"]
    model_lock_sha256: str
    drafter_model_id: str
    prepared_drafter_revision: str
    dspark_native_heads: DSparkNativeHeadNames | None
    parameters: tuple[PreparedParameterMetadata, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only prepared-drafter inventory schema v1 is supported")
        if (
            type(self.kind) is not str
            or self.kind != "prepared_drafter_parameter_inventory"
        ):
            raise ValueError("prepared-drafter inventory kind is invalid")
        _require_sha256("prepared-drafter model lock", self.model_lock_sha256)
        _strict_text("prepared-drafter model ID", self.drafter_model_id)
        _require_revision("prepared-drafter revision", self.prepared_drafter_revision)
        if (
            self.dspark_native_heads is not None
            and type(self.dspark_native_heads) is not DSparkNativeHeadNames
        ):
            raise TypeError("prepared-drafter DSpark native heads are invalid")
        if (
            type(self.parameters) is not tuple
            or not self.parameters
            or any(
                type(row) is not PreparedParameterMetadata for row in self.parameters
            )
        ):
            raise TypeError("prepared-drafter parameters must be a non-empty tuple")
        names = tuple(row.name for row in self.parameters)
        if names != tuple(sorted(set(names))):
            raise ValueError(
                "prepared-drafter parameter names must be sorted and unique"
            )
        if self.dspark_native_heads is not None and not {
            self.dspark_native_heads.w1_name,
            self.dspark_native_heads.w2_name,
            self.dspark_native_heads.acceptance_name,
        }.issubset(names):
            raise ValueError(
                "prepared-drafter DSpark native heads must name exact parameters"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "model_lock_sha256": self.model_lock_sha256,
            "drafter_model_id": self.drafter_model_id,
            "prepared_drafter_revision": self.prepared_drafter_revision,
            "dspark_native_heads": (
                None
                if self.dspark_native_heads is None
                else self.dspark_native_heads.to_dict()
            ),
            "parameters": [row.to_dict() for row in self.parameters],
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedDrafterParameterInventory:
        row = _strict_object(
            "prepared-drafter inventory", value, _PREPARED_INVENTORY_FIELDS
        )
        return cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            model_lock_sha256=row["model_lock_sha256"],
            drafter_model_id=row["drafter_model_id"],
            prepared_drafter_revision=row["prepared_drafter_revision"],
            dspark_native_heads=(
                None
                if row["dspark_native_heads"] is None
                else DSparkNativeHeadNames.from_dict(row["dspark_native_heads"])
            ),
            parameters=tuple(
                PreparedParameterMetadata.from_dict(item)
                for item in _strict_list(
                    "prepared-drafter parameters", row["parameters"]
                )
            ),
        )

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())


@dataclass(frozen=True)
class _MetadataTensor:
    shape: tuple[int, ...]
    dtype: str

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        value = 1
        for dimension in self.shape:
            value *= dimension
        return value

    def is_floating_point(self) -> bool:
        return self.dtype in _FLOAT_DTYPES


@dataclass(frozen=True)
class _ExecutionIdentity:
    method: str
    backend: str
    mode: str
    scope: str
    rank: int | None
    lora_alpha: int | None
    optimizer: str
    target_model_id: str
    target_revision: str
    drafter_model_id: str
    drafter_revision: str
    run_config_sha256: str
    split_sha256: str
    cell_id: str
    cell_declaration_sha256: str


def _execution_identity(
    *,
    run_config: TrainablePlanRawJsonBinding,
    split: TrainablePlanRawJsonBinding,
    cell: TrainablePlanRawJsonBinding,
) -> _ExecutionIdentity:
    # Import lazily: config.schema imports adaptation.parameters while the
    # adaptation package is initialized.
    from lightcone_spec.config import run_config_sha256
    from lightcone_spec.config.schema import RunConfig
    from lightcone_spec.experiments.registry import (
        CellIdentity,
        CellStatus,
        ExperimentCell,
        ResourceClaim,
        WorkloadClass,
    )

    raw_config = run_config.load()
    if type(raw_config) is not dict:
        raise TypeError("trainable-plan run config must be a JSON object")
    try:
        config = RunConfig.model_validate(raw_config)
    except ValueError as error:
        raise ValueError("trainable-plan run config is invalid") from error
    if (
        config.model_dump(mode="json") != raw_config
        or run_config_sha256(config) != run_config.semantic_sha256
    ):
        raise ValueError("trainable-plan run config is not fully materialized")
    if config.method not in {"tts", "l0"} or config.adaptation is None:
        raise ValueError("only TTS/L0 may carry trainable-plan authority")
    adaptation = config.adaptation
    optimizer = adaptation.optimizer.name
    if optimizer not in TRAINABLE_PLAN_OPTIMIZERS:
        raise ValueError("trainable-plan optimizer has no registered state layout")

    raw_cell = _strict_object("trainable-plan cell", cell.load(), _CELL_FIELDS)
    identity = _strict_object(
        "trainable-plan cell identity", raw_cell["identity"], _CELL_IDENTITY_FIELDS
    )
    resources = _strict_object(
        "trainable-plan cell resources", raw_cell["resources"], _RESOURCE_FIELDS
    )
    raw_identity_gpus = _strict_text_list(
        "trainable-plan cell identity GPU UUIDs", identity["gpu_uuids"]
    )
    raw_resource_gpus = _strict_text_list(
        "trainable-plan cell resource GPU UUIDs", resources["gpu_uuids"]
    )
    try:
        registered_cell = ExperimentCell(
            identity=CellIdentity(
                experiment=_strict_text("cell experiment", identity["experiment"]),
                model=_strict_text("cell model", identity["model"]),
                backend=_strict_text("cell backend", identity["backend"]),
                task=_strict_text("cell task", identity["task"]),
                method=_strict_text("cell method", identity["method"]),
                scope=_nullable_text("cell scope", identity["scope"]),
                rank=_nullable_int("cell rank", identity["rank"]),
                alpha_over_rank=_nullable_positive_float(
                    "cell alpha_over_rank", identity["alpha_over_rank"]
                ),
                optimizer=_nullable_text("cell optimizer", identity["optimizer"]),
                learning_rate=_nullable_positive_float(
                    "cell learning_rate", identity["learning_rate"]
                ),
                schedule=_nullable_text("cell schedule", identity["schedule"]),
                context=_nullable_int("cell context", identity["context"]),
                regime=_strict_text("cell regime", identity["regime"]),
                width=_nullable_int("cell width", identity["width"]),
                arrival=_strict_text("cell arrival", identity["arrival"]),
                slo=_strict_text("cell SLO", identity["slo"]),
                cohort=_strict_text("cell cohort", identity["cohort"]),
                topology=_strict_text("cell topology", identity["topology"]),
                seed=_nonnegative_int("cell seed", identity["seed"]),
                block=_nonnegative_int("cell block", identity["block"]),
                gpu_uuids=raw_identity_gpus,
                parameterization=_strict_text(
                    "cell parameterization", identity["parameterization"]
                ),
                variant=_strict_text("cell variant", identity["variant"]),
                concurrency=_nullable_int("cell concurrency", identity["concurrency"]),
                load_factor=_nullable_positive_float(
                    "cell load_factor", identity["load_factor"]
                ),
                cohort_count=_positive_int(
                    "cell cohort_count", identity["cohort_count"]
                ),
            ),
            resources=ResourceClaim(
                gpu_uuids=raw_resource_gpus,
                ports=_strict_int_list(
                    "trainable-plan cell resource ports", resources["ports"]
                ),
                cache_root=_strict_text(
                    "cell resource cache_root", resources["cache_root"]
                ),
                evidence_root=_strict_text(
                    "cell resource evidence_root", resources["evidence_root"]
                ),
                workload_class=WorkloadClass(
                    _strict_text(
                        "cell resource workload_class", resources["workload_class"]
                    )
                ),
            ),
            status=CellStatus(_strict_text("cell status", raw_cell["status"])),
            reason_code=_strict_text("cell reason_code", raw_cell["reason_code"]),
            reason=_strict_text("cell reason", raw_cell["reason"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("trainable-plan cell declaration is invalid") from error
    if registered_cell.sha256 != cell.semantic_sha256:
        raise ValueError("trainable-plan cell declaration is not canonical")
    if not registered_cell.runnable:
        raise ValueError("trainable-plan authority requires a runnable registry cell")
    expected_alpha_over_rank = (
        None
        if adaptation.weight_update_mode == "full"
        else adaptation.lora_alpha / adaptation.rank
    )
    if (
        identity["method"] != config.method
        or identity["model"] != config.model.target
        or identity["backend"] != config.model.algorithm
        or identity["scope"] != adaptation.parameter_scope
        or identity["rank"] != adaptation.rank
        or identity["alpha_over_rank"] != expected_alpha_over_rank
        or identity["optimizer"] != optimizer
        or identity["learning_rate"] != adaptation.optimizer.learning_rate
        or identity["schedule"] != adaptation.optimizer.schedule
        or identity["context"] != config.runtime.context_length
        or identity["width"] != config.runtime.speculative_num_draft_tokens
        or identity["seed"] != config.runtime.random_seed
        or identity["concurrency"] != config.runtime.max_running_requests
        or identity["parameterization"] != adaptation.weight_update_mode
    ):
        raise ValueError("registry cell differs from its exact adaptive run config")
    # Force the split through the strict raw loader even though its scientific
    # schema remains owned by the execution bundle.
    split.load()
    return _ExecutionIdentity(
        method=config.method,
        backend=config.model.algorithm,
        mode=adaptation.weight_update_mode,
        scope=adaptation.parameter_scope,
        rank=adaptation.rank,
        lora_alpha=adaptation.lora_alpha,
        optimizer=optimizer,
        target_model_id=config.model.target,
        target_revision=config.model.target_revision,
        drafter_model_id=config.model.drafter,
        drafter_revision=config.model.drafter_revision,
        run_config_sha256=run_config.semantic_sha256,
        split_sha256=split.semantic_sha256,
        cell_id=registered_cell.cell_id,
        cell_declaration_sha256=registered_cell.sha256,
    )


def _validate_model_lock(
    value: object,
    *,
    source: TrainablePlanRawJsonBinding,
    inventory: PreparedDrafterParameterInventory,
    execution: _ExecutionIdentity,
) -> ModelLock:
    row = _strict_object("trainable-plan model lock", value, _MODEL_LOCK_FIELDS)
    if type(row["schema_version"]) is not int or row["schema_version"] != 2:
        raise ValueError("trainable-plan model lock must use schema v2")
    models: dict[str, str] = {}
    locked_rows: list[LockedModel] = []
    for item in _strict_list("trainable-plan locked models", row["models"]):
        model = _strict_object(
            "trainable-plan locked model", item, _LOCKED_MODEL_FIELDS
        )
        model_id = _strict_text("locked model ID", model["model_id"])
        revision = _require_revision("locked model revision", model["revision"])
        if model_id in models:
            raise ValueError("trainable-plan model lock contains duplicate model IDs")
        models[model_id] = revision
        locked_rows.append(LockedModel(model_id=model_id, revision=revision))
    if not models:
        raise ValueError("trainable-plan model lock is empty")
    if (
        inventory.model_lock_sha256 != source.semantic_sha256
        or inventory.drafter_model_id != execution.drafter_model_id
        or inventory.prepared_drafter_revision != execution.drafter_revision
        or models.get(inventory.drafter_model_id) != inventory.prepared_drafter_revision
        or models.get(execution.target_model_id) != execution.target_revision
    ):
        raise ValueError(
            "target/prepared drafter differs from its exact model lock and revision"
        )
    lock = ModelLock(schema_version=2, models=tuple(locked_rows))
    lock.validate()
    if lock.sha256 != source.semantic_sha256:
        raise ValueError("trainable-plan model lock canonical identity differs")
    return lock


def _release_dspark_heads(
    content: PreparedModelContentAuthorityResult,
    *,
    drafter_model_id: str,
) -> DSparkNativeHeadNames:
    tensors = content.snapshot(drafter_model_id).tensors
    w1 = tuple(item.name for item in tensors if _DSPARK_W1.search(item.name))
    w2 = tuple(item.name for item in tensors if _DSPARK_W2.search(item.name))
    acceptance = tuple(
        item for item in tensors if _DSPARK_ACCEPTANCE_WEIGHT.search(item.name)
    )
    if len(w1) != 1 or len(w2) != 1 or len(acceptance) != 1:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_parameter_inventory_first_party_extractor_unavailable",
            "release-owned DSpark native-head names do not resolve exactly",
        )
    if len(acceptance[0].shape) > 1 or _tensor_numel(acceptance[0].shape) != 1:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_parameter_inventory_first_party_extractor_unavailable",
            "release-owned DSpark acceptance parameter is not scalar",
        )
    return DSparkNativeHeadNames(
        w1_name=w1[0],
        w2_name=w2[0],
        acceptance_name=acceptance[0].name,
    )


def _tensor_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _first_party_inventory(
    *,
    model_lock: ModelLock,
    content_authority: PreparedModelContentAuthorityBinding | None,
    execution: _ExecutionIdentity,
) -> tuple[
    PreparedDrafterParameterInventory,
    PreparedModelContentAuthorityResult,
]:
    if type(content_authority) is not PreparedModelContentAuthorityBinding:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_parameter_inventory_first_party_extractor_unavailable",
            "adapted execution lacks prepared snapshot content authority",
        )
    try:
        content = revalidate_prepared_model_content_authority(
            model_lock,
            content_authority,
            expected_release_manifest_sha256=(
                content_authority.release_manifest_sha256
            ),
        )
        snapshot = content.snapshot(execution.drafter_model_id)
    except PreparedModelContentAuthorityBlocked as error:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_parameter_inventory_first_party_extractor_unavailable",
            str(error),
        ) from error
    if snapshot.revision != execution.drafter_revision:
        raise ValueError("prepared content drafter revision differs from RunConfig")
    heads = (
        _release_dspark_heads(
            content,
            drafter_model_id=execution.drafter_model_id,
        )
        if execution.backend == "DSPARK"
        else None
    )
    replicated = (
        frozenset()
        if heads is None
        else frozenset((heads.w1_name, heads.w2_name, heads.acceptance_name))
    )
    inventory = PreparedDrafterParameterInventory(
        schema_version=1,
        kind="prepared_drafter_parameter_inventory",
        model_lock_sha256=model_lock.sha256,
        drafter_model_id=execution.drafter_model_id,
        prepared_drafter_revision=execution.drafter_revision,
        dspark_native_heads=heads,
        parameters=tuple(
            PreparedParameterMetadata(
                name=item.name,
                shape=item.shape,
                dtype=item.dtype,
                ownership="replicated" if item.name in replicated else "sharded",
            )
            for item in snapshot.tensors
        ),
    )
    return inventory, content


def _derive_plan(
    inventory: PreparedDrafterParameterInventory,
    *,
    backend: str,
    mode: str,
    scope: str,
    rank: int | None,
    lora_alpha: int | None,
    dspark_native_heads: DSparkNativeHeadNames | None,
) -> TrainablePlan:
    if mode == "lora":
        if rank is None or lora_alpha != rank:
            raise ValueError("LoRA authority requires exact registered rank and alpha")
    elif mode == "full":
        if rank is not None or lora_alpha is not None:
            raise ValueError("Full authority requires null rank and alpha")
    else:
        raise ValueError("trainable-plan authority mode is unsupported")
    items = tuple(
        (row.name, _MetadataTensor(shape=row.shape, dtype=row.dtype))
        for row in inventory.parameters
    )
    replicated = tuple(
        row.name for row in inventory.parameters if row.ownership == "replicated"
    )
    if backend == "DFLASH":
        if dspark_native_heads is not None:
            raise ValueError("DFLASH authority cannot name DSpark native heads")
        plan: TrainablePlan = DFlashParameterPlan.build(
            items,
            mode=mode,
            scope=scope,
            rank=rank,
            replicated_names=replicated,
        )
    elif backend == "DSPARK":
        if dspark_native_heads is None:
            raise ValueError("DSPARK authority requires exact native head names")
        plan = DSparkParameterPlan.build(
            items,
            mode=mode,
            scope=scope,
            rank=rank,
            w1_name=dspark_native_heads.w1_name,
            w2_name=dspark_native_heads.w2_name,
            acceptance_name=dspark_native_heads.acceptance_name,
            replicated_names=replicated,
        )
    elif backend in {"EAGLE", "EAGLE3", "NEXTN"}:
        if dspark_native_heads is not None:
            raise ValueError("native-layer authority cannot name DSpark native heads")
        plan = NativeLayerParameterPlan.build(
            items,
            backend=backend,
            mode=mode,
            scope=scope,
            rank=rank,
            replicated_names=replicated,
        )
    else:
        raise ValueError("trainable-plan authority backend is unsupported")
    if plan.lora_alpha != lora_alpha:
        raise RuntimeError("trainable-plan reducer changed its alpha contract")
    return plan


def _entry_to_dict(entry: ParameterEntry) -> dict[str, object]:
    return {
        "name": entry.name,
        "shape": list(entry.shape),
        "dtype": entry.dtype,
        "parameterization": entry.parameterization,
        "ownership": entry.ownership,
    }


def _state_layout_to_list(plan: TrainablePlan) -> list[dict[str, object]]:
    return [
        {
            "name": row["name"],
            "parameterization": row["parameterization"],
            "ownership": row["ownership"],
            "state_shapes": [list(shape) for shape in row["state_shapes"]],
        }
        for row in plan.state_layout
    ]


def _manifest_value(
    *,
    plan: TrainablePlan,
    model_lock: TrainablePlanRawJsonBinding,
    prepared_drafter: TrainablePlanRawJsonBinding,
    run_config: TrainablePlanRawJsonBinding,
    split: TrainablePlanRawJsonBinding,
    cell: TrainablePlanRawJsonBinding,
    content_authority: PreparedModelContentAuthorityBinding,
    inventory: PreparedDrafterParameterInventory,
    execution: _ExecutionIdentity,
    dspark_native_heads: DSparkNativeHeadNames | None,
) -> dict[str, object]:
    entries = [_entry_to_dict(entry) for entry in plan.entries]
    frozen_names = list(plan.frozen_names)
    state_layout = _state_layout_to_list(plan)
    optimizer_memory_prediction: dict[str, object] = {
        "optimizer": execution.optimizer,
        **plan.predict_memory(execution.optimizer).to_dict(),
    }
    return {
        "schema_version": 1,
        "kind": "trainable_plan_authority_manifest",
        "reducer_protocol_sha256": TRAINABLE_PLAN_REDUCER_PROTOCOL_SHA256,
        "model_lock_artifact": model_lock.path,
        "prepared_drafter_artifact": prepared_drafter.path,
        "run_config_artifact": run_config.path,
        "split_artifact": split.path,
        "cell_artifact": cell.path,
        "model_lock_sha256": model_lock.semantic_sha256,
        "prepared_model_content_authority_sha256": content_authority.sha256,
        "prepared_model_content_manifest_sha256": (
            content_authority.release_manifest_sha256
        ),
        "prepared_drafter_inventory_sha256": inventory.sha256,
        "run_config_sha256": execution.run_config_sha256,
        "split_sha256": execution.split_sha256,
        "cell_id": execution.cell_id,
        "cell_declaration_sha256": execution.cell_declaration_sha256,
        "target_model_id": execution.target_model_id,
        "target_revision": execution.target_revision,
        "drafter_model_id": inventory.drafter_model_id,
        "prepared_drafter_revision": inventory.prepared_drafter_revision,
        "method": execution.method,
        "backend": plan.backend,
        "mode": plan.mode,
        "scope": plan.scope,
        "rank": plan.rank,
        "lora_alpha": plan.lora_alpha,
        "optimizer": execution.optimizer,
        "dspark_native_heads": (
            None if dspark_native_heads is None else dspark_native_heads.to_dict()
        ),
        "entries": entries,
        "frozen_names": frozen_names,
        "state_layout": state_layout,
        "trainable_parameter_count": plan.trainable_parameter_count,
        "entries_sha256": _content_sha256(entries),
        "frozen_names_sha256": _content_sha256(frozen_names),
        "state_layout_sha256": _content_sha256(state_layout),
        "optimizer_memory_prediction": optimizer_memory_prediction,
        "optimizer_memory_sha256": _content_sha256(optimizer_memory_prediction),
        "allocation_memory_sha256": plan.allocation_memory_sha256,
        "trainable_plan_sha256": plan.sha256,
    }


def _selector_from_manifest(
    value: object,
) -> tuple[dict[str, Any], DSparkNativeHeadNames | None]:
    row = _strict_object("trainable-plan authority manifest", value, _MANIFEST_FIELDS)
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != 1
        or type(row["kind"]) is not str
        or row["kind"] != "trainable_plan_authority_manifest"
        or row["reducer_protocol_sha256"] != TRAINABLE_PLAN_REDUCER_PROTOCOL_SHA256
    ):
        raise ValueError("trainable-plan authority manifest identity is invalid")
    for name in (
        "model_lock_sha256",
        "prepared_model_content_authority_sha256",
        "prepared_model_content_manifest_sha256",
        "prepared_drafter_inventory_sha256",
        "run_config_sha256",
        "split_sha256",
        "cell_id",
        "cell_declaration_sha256",
        "entries_sha256",
        "frozen_names_sha256",
        "state_layout_sha256",
        "optimizer_memory_sha256",
        "allocation_memory_sha256",
        "trainable_plan_sha256",
    ):
        _require_sha256(f"manifest {name}", row[name])
    _strict_text("manifest backend", row["backend"])
    _strict_text("manifest mode", row["mode"])
    _strict_text("manifest scope", row["scope"])
    _strict_text("manifest target model ID", row["target_model_id"])
    _require_revision("manifest target revision", row["target_revision"])
    _strict_text("manifest drafter model ID", row["drafter_model_id"])
    _require_revision(
        "manifest prepared-drafter revision", row["prepared_drafter_revision"]
    )
    _strict_text("manifest method", row["method"])
    _nullable_int("manifest rank", row["rank"])
    _nullable_int("manifest LoRA alpha", row["lora_alpha"])
    _strict_text("manifest optimizer", row["optimizer"])
    _nonnegative_int(
        "manifest trainable parameter count", row["trainable_parameter_count"]
    )
    _strict_list("manifest entries", row["entries"])
    _strict_list("manifest frozen names", row["frozen_names"])
    _strict_list("manifest state layout", row["state_layout"])
    prediction = _strict_object(
        "manifest optimizer memory prediction",
        row["optimizer_memory_prediction"],
        _MEMORY_PREDICTION_FIELDS,
    )
    _strict_text("manifest memory optimizer", prediction["optimizer"])
    for name in _MEMORY_PREDICTION_FIELDS - {"optimizer"}:
        _nonnegative_int(f"manifest memory {name}", prediction[name])
    for name in (
        "model_lock_artifact",
        "prepared_drafter_artifact",
        "run_config_artifact",
        "split_artifact",
        "cell_artifact",
    ):
        _resolved_absolute_path(row[name], label=f"manifest {name}")
    heads = (
        None
        if row["dspark_native_heads"] is None
        else DSparkNativeHeadNames.from_dict(row["dspark_native_heads"])
    )
    return row, heads


def materialize_trainable_plan_authority_manifest(
    *,
    model_lock_artifact: str | Path,
    prepared_drafter_artifact: str | Path,
    run_config_artifact: str | Path,
    split_artifact: str | Path,
    cell_artifact: str | Path,
    prepared_model_content_authority: (
        PreparedModelContentAuthorityBinding | None
    ) = None,
) -> dict[str, object]:
    """Derive the canonical manifest; callers still publish it immutably."""

    model_lock = TrainablePlanRawJsonBinding.from_path(
        model_lock_artifact, role="trainable_plan_model_lock"
    )
    prepared = TrainablePlanRawJsonBinding.from_path(
        prepared_drafter_artifact,
        role="prepared_drafter_parameter_inventory",
    )
    run_config = TrainablePlanRawJsonBinding.from_path(
        run_config_artifact, role="trainable_plan_run_config"
    )
    split = TrainablePlanRawJsonBinding.from_path(
        split_artifact, role="trainable_plan_split"
    )
    cell = TrainablePlanRawJsonBinding.from_path(
        cell_artifact, role="trainable_plan_cell"
    )
    serialized_inventory = PreparedDrafterParameterInventory.from_dict(prepared.load())
    execution = _execution_identity(run_config=run_config, split=split, cell=cell)
    locked = _validate_model_lock(
        model_lock.load(),
        source=model_lock,
        inventory=serialized_inventory,
        execution=execution,
    )
    if (
        type(prepared_model_content_authority)
        is not PreparedModelContentAuthorityBinding
    ):
        raise PreparedModelContentAuthorityBlocked(
            "prepared_parameter_inventory_first_party_extractor_unavailable",
            "adapted execution lacks prepared snapshot content authority",
        )
    inventory, _ = _first_party_inventory(
        model_lock=locked,
        content_authority=prepared_model_content_authority,
        execution=execution,
    )
    if serialized_inventory != inventory:
        raise ValueError(
            "serialized prepared inventory differs from first-party snapshot extraction"
        )
    plan = _derive_plan(
        inventory,
        backend=execution.backend,
        mode=execution.mode,
        scope=execution.scope,
        rank=execution.rank,
        lora_alpha=execution.lora_alpha,
        dspark_native_heads=inventory.dspark_native_heads,
    )
    return _manifest_value(
        plan=plan,
        model_lock=model_lock,
        prepared_drafter=prepared,
        run_config=run_config,
        split=split,
        cell=cell,
        inventory=inventory,
        content_authority=prepared_model_content_authority,
        execution=execution,
        dspark_native_heads=inventory.dspark_native_heads,
    )


@dataclass(frozen=True)
class TrainablePlanAuthorityBinding:
    schema_version: int
    kind: Literal["trainable_plan_authority_binding"]
    manifest: TrainablePlanRawJsonBinding
    model_lock: TrainablePlanRawJsonBinding
    prepared_drafter: TrainablePlanRawJsonBinding
    run_config: TrainablePlanRawJsonBinding
    split: TrainablePlanRawJsonBinding
    cell: TrainablePlanRawJsonBinding
    prepared_model_content_authority: PreparedModelContentAuthorityBinding
    reducer_protocol_sha256: str
    model_lock_sha256: str
    prepared_model_content_manifest_sha256: str
    prepared_drafter_inventory_sha256: str
    run_config_sha256: str
    split_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    target_model_id: str
    target_revision: str
    drafter_model_id: str
    prepared_drafter_revision: str
    method: str
    backend: str
    mode: str
    scope: str
    rank: int | None
    lora_alpha: int | None
    optimizer: str
    dspark_native_heads: DSparkNativeHeadNames | None
    entries_sha256: str
    frozen_names_sha256: str
    state_layout_sha256: str
    optimizer_memory_sha256: str
    allocation_memory_sha256: str
    trainable_plan_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.kind) is not str
            or self.kind != "trainable_plan_authority_binding"
        ):
            raise ValueError("trainable-plan authority binding identity is invalid")
        for value, role in (
            (self.manifest, "trainable_plan_authority_manifest"),
            (self.model_lock, "trainable_plan_model_lock"),
            (self.prepared_drafter, "prepared_drafter_parameter_inventory"),
            (self.run_config, "trainable_plan_run_config"),
            (self.split, "trainable_plan_split"),
            (self.cell, "trainable_plan_cell"),
        ):
            if type(value) is not TrainablePlanRawJsonBinding or value.role != role:
                raise TypeError("trainable-plan authority raw source role is invalid")
        if (
            type(self.prepared_model_content_authority)
            is not PreparedModelContentAuthorityBinding
        ):
            raise TypeError(
                "trainable-plan authority requires exact prepared content authority"
            )
        if self.reducer_protocol_sha256 != TRAINABLE_PLAN_REDUCER_PROTOCOL_SHA256:
            raise ValueError("trainable-plan authority uses another reducer protocol")
        for name in (
            "model_lock_sha256",
            "prepared_model_content_manifest_sha256",
            "prepared_drafter_inventory_sha256",
            "run_config_sha256",
            "split_sha256",
            "cell_id",
            "cell_declaration_sha256",
            "entries_sha256",
            "frozen_names_sha256",
            "state_layout_sha256",
            "optimizer_memory_sha256",
            "allocation_memory_sha256",
            "trainable_plan_sha256",
        ):
            _require_sha256(f"authority {name}", getattr(self, name))
        _strict_text("authority target model ID", self.target_model_id)
        _require_revision("authority target revision", self.target_revision)
        _strict_text("authority drafter model ID", self.drafter_model_id)
        _require_revision(
            "authority prepared-drafter revision", self.prepared_drafter_revision
        )
        if self.method not in {"tts", "l0"}:
            raise ValueError("trainable-plan authority method must be TTS/L0")
        if self.backend not in {"DFLASH", "DSPARK", "EAGLE", "EAGLE3", "NEXTN"}:
            raise ValueError("trainable-plan authority backend is unsupported")
        if self.mode not in {"lora", "full"}:
            raise ValueError("trainable-plan authority mode is unsupported")
        _strict_text("authority scope", self.scope)
        if self.optimizer not in TRAINABLE_PLAN_OPTIMIZERS:
            raise ValueError("trainable-plan authority optimizer is unsupported")
        rank = _nullable_int("authority rank", self.rank)
        alpha = _nullable_int("authority LoRA alpha", self.lora_alpha)
        if (self.mode == "lora" and (rank is None or alpha != rank)) or (
            self.mode == "full" and (rank is not None or alpha is not None)
        ):
            raise ValueError("trainable-plan authority rank/alpha differ from mode")
        if (
            self.model_lock_sha256 != self.model_lock.semantic_sha256
            or self.model_lock_sha256
            != self.prepared_model_content_authority.model_lock_sha256
            or self.prepared_model_content_manifest_sha256
            != self.prepared_model_content_authority.release_manifest_sha256
            or self.prepared_drafter_inventory_sha256
            != self.prepared_drafter.semantic_sha256
            or self.run_config_sha256 != self.run_config.semantic_sha256
            or self.split_sha256 != self.split.semantic_sha256
            or self.cell_declaration_sha256 != self.cell.semantic_sha256
        ):
            raise ValueError("authority identity differs from a bound raw source")
        if (
            self.dspark_native_heads is not None
            and type(self.dspark_native_heads) is not DSparkNativeHeadNames
        ):
            raise TypeError("authority DSpark native heads are invalid")
        if (self.backend == "DSPARK") != (self.dspark_native_heads is not None):
            raise ValueError(
                "authority DSpark backend and raw native-head names differ"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "manifest": self.manifest.to_dict(),
            "model_lock": self.model_lock.to_dict(),
            "prepared_drafter": self.prepared_drafter.to_dict(),
            "run_config": self.run_config.to_dict(),
            "split": self.split.to_dict(),
            "cell": self.cell.to_dict(),
            "prepared_model_content_authority": (
                prepared_model_content_authority_to_dict(
                    self.prepared_model_content_authority
                )
            ),
            "reducer_protocol_sha256": self.reducer_protocol_sha256,
            "model_lock_sha256": self.model_lock_sha256,
            "prepared_model_content_manifest_sha256": (
                self.prepared_model_content_manifest_sha256
            ),
            "prepared_drafter_inventory_sha256": (
                self.prepared_drafter_inventory_sha256
            ),
            "run_config_sha256": self.run_config_sha256,
            "split_sha256": self.split_sha256,
            "cell_id": self.cell_id,
            "cell_declaration_sha256": self.cell_declaration_sha256,
            "target_model_id": self.target_model_id,
            "target_revision": self.target_revision,
            "drafter_model_id": self.drafter_model_id,
            "prepared_drafter_revision": self.prepared_drafter_revision,
            "method": self.method,
            "backend": self.backend,
            "mode": self.mode,
            "scope": self.scope,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "optimizer": self.optimizer,
            "dspark_native_heads": (
                None
                if self.dspark_native_heads is None
                else self.dspark_native_heads.to_dict()
            ),
            "entries_sha256": self.entries_sha256,
            "frozen_names_sha256": self.frozen_names_sha256,
            "state_layout_sha256": self.state_layout_sha256,
            "optimizer_memory_sha256": self.optimizer_memory_sha256,
            "allocation_memory_sha256": self.allocation_memory_sha256,
            "trainable_plan_sha256": self.trainable_plan_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrainablePlanAuthorityBinding:
        row = _strict_object(
            "trainable-plan authority binding", value, _AUTHORITY_FIELDS
        )
        return cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            manifest=TrainablePlanRawJsonBinding.from_dict(row["manifest"]),
            model_lock=TrainablePlanRawJsonBinding.from_dict(row["model_lock"]),
            prepared_drafter=TrainablePlanRawJsonBinding.from_dict(
                row["prepared_drafter"]
            ),
            run_config=TrainablePlanRawJsonBinding.from_dict(row["run_config"]),
            split=TrainablePlanRawJsonBinding.from_dict(row["split"]),
            cell=TrainablePlanRawJsonBinding.from_dict(row["cell"]),
            prepared_model_content_authority=(
                prepared_model_content_authority_from_dict(
                    row["prepared_model_content_authority"]
                )
            ),
            reducer_protocol_sha256=row["reducer_protocol_sha256"],
            model_lock_sha256=row["model_lock_sha256"],
            prepared_model_content_manifest_sha256=row[
                "prepared_model_content_manifest_sha256"
            ],
            prepared_drafter_inventory_sha256=row["prepared_drafter_inventory_sha256"],
            run_config_sha256=row["run_config_sha256"],
            split_sha256=row["split_sha256"],
            cell_id=row["cell_id"],
            cell_declaration_sha256=row["cell_declaration_sha256"],
            target_model_id=row["target_model_id"],
            target_revision=row["target_revision"],
            drafter_model_id=row["drafter_model_id"],
            prepared_drafter_revision=row["prepared_drafter_revision"],
            method=row["method"],
            backend=row["backend"],
            mode=row["mode"],
            scope=row["scope"],
            rank=row["rank"],
            lora_alpha=row["lora_alpha"],
            optimizer=row["optimizer"],
            dspark_native_heads=(
                None
                if row["dspark_native_heads"] is None
                else DSparkNativeHeadNames.from_dict(row["dspark_native_heads"])
            ),
            entries_sha256=row["entries_sha256"],
            frozen_names_sha256=row["frozen_names_sha256"],
            state_layout_sha256=row["state_layout_sha256"],
            optimizer_memory_sha256=row["optimizer_memory_sha256"],
            allocation_memory_sha256=row["allocation_memory_sha256"],
            trainable_plan_sha256=row["trainable_plan_sha256"],
        )

    @property
    def sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def revalidate(self) -> TrainablePlanAuthorityResult:
        return replay_trainable_plan_authority(self)


@dataclass(frozen=True)
class TrainablePlanAuthorityResult:
    binding: TrainablePlanAuthorityBinding
    plan: TrainablePlan
    prepared_drafter: PreparedDrafterParameterInventory

    def __post_init__(self) -> None:
        if type(self.binding) is not TrainablePlanAuthorityBinding:
            raise TypeError("trainable-plan result requires an exact authority binding")
        if not isinstance(self.plan, TrainablePlan):
            raise TypeError("trainable-plan result requires a TrainablePlan")
        if type(self.prepared_drafter) is not PreparedDrafterParameterInventory:
            raise TypeError("trainable-plan result requires exact prepared metadata")
        if (
            self.plan.sha256 != self.binding.trainable_plan_sha256
            or self.plan.state_layout_sha256 != self.binding.state_layout_sha256
            or _content_sha256(
                {
                    "optimizer": self.binding.optimizer,
                    **self.plan.predict_memory(self.binding.optimizer).to_dict(),
                }
            )
            != self.binding.optimizer_memory_sha256
            or self.plan.allocation_memory_sha256
            != self.binding.allocation_memory_sha256
            or self.prepared_drafter.sha256
            != self.binding.prepared_drafter_inventory_sha256
        ):
            raise ValueError("trainable-plan result differs from its authority")


def _binding_from_replay(
    *,
    manifest: TrainablePlanRawJsonBinding,
    model_lock: TrainablePlanRawJsonBinding,
    prepared_drafter: TrainablePlanRawJsonBinding,
    run_config: TrainablePlanRawJsonBinding,
    split: TrainablePlanRawJsonBinding,
    cell: TrainablePlanRawJsonBinding,
    content_authority: PreparedModelContentAuthorityBinding,
    inventory: PreparedDrafterParameterInventory,
    execution: _ExecutionIdentity,
    plan: TrainablePlan,
    dspark_native_heads: DSparkNativeHeadNames | None,
) -> TrainablePlanAuthorityBinding:
    entries = [_entry_to_dict(entry) for entry in plan.entries]
    frozen_names = list(plan.frozen_names)
    state_layout = _state_layout_to_list(plan)
    optimizer_memory_prediction = {
        "optimizer": execution.optimizer,
        **plan.predict_memory(execution.optimizer).to_dict(),
    }
    return TrainablePlanAuthorityBinding(
        schema_version=1,
        kind="trainable_plan_authority_binding",
        manifest=manifest,
        model_lock=model_lock,
        prepared_drafter=prepared_drafter,
        run_config=run_config,
        split=split,
        cell=cell,
        prepared_model_content_authority=content_authority,
        reducer_protocol_sha256=TRAINABLE_PLAN_REDUCER_PROTOCOL_SHA256,
        model_lock_sha256=model_lock.semantic_sha256,
        prepared_model_content_manifest_sha256=(
            content_authority.release_manifest_sha256
        ),
        prepared_drafter_inventory_sha256=inventory.sha256,
        run_config_sha256=execution.run_config_sha256,
        split_sha256=execution.split_sha256,
        cell_id=execution.cell_id,
        cell_declaration_sha256=execution.cell_declaration_sha256,
        target_model_id=execution.target_model_id,
        target_revision=execution.target_revision,
        drafter_model_id=inventory.drafter_model_id,
        prepared_drafter_revision=inventory.prepared_drafter_revision,
        method=execution.method,
        backend=plan.backend,
        mode=plan.mode,
        scope=plan.scope,
        rank=plan.rank,
        lora_alpha=plan.lora_alpha,
        optimizer=execution.optimizer,
        dspark_native_heads=dspark_native_heads,
        entries_sha256=_content_sha256(entries),
        frozen_names_sha256=_content_sha256(frozen_names),
        state_layout_sha256=_content_sha256(state_layout),
        optimizer_memory_sha256=_content_sha256(optimizer_memory_prediction),
        allocation_memory_sha256=plan.allocation_memory_sha256,
        trainable_plan_sha256=plan.sha256,
    )


def _replay(
    *,
    manifest: TrainablePlanRawJsonBinding,
    model_lock: TrainablePlanRawJsonBinding,
    prepared_drafter: TrainablePlanRawJsonBinding,
    run_config: TrainablePlanRawJsonBinding,
    split: TrainablePlanRawJsonBinding,
    cell: TrainablePlanRawJsonBinding,
    content_authority: PreparedModelContentAuthorityBinding | None,
    expected_binding: TrainablePlanAuthorityBinding | None,
) -> TrainablePlanAuthorityResult:
    manifest_row, _ = _selector_from_manifest(manifest.load())
    if (
        manifest_row["model_lock_artifact"] != model_lock.path
        or manifest_row["prepared_drafter_artifact"] != prepared_drafter.path
        or manifest_row["run_config_artifact"] != run_config.path
        or manifest_row["split_artifact"] != split.path
        or manifest_row["cell_artifact"] != cell.path
    ):
        raise ValueError("trainable-plan manifest swaps a raw source path")
    serialized_inventory = PreparedDrafterParameterInventory.from_dict(
        prepared_drafter.load()
    )
    execution = _execution_identity(run_config=run_config, split=split, cell=cell)
    locked = _validate_model_lock(
        model_lock.load(),
        source=model_lock,
        inventory=serialized_inventory,
        execution=execution,
    )
    inventory, _ = _first_party_inventory(
        model_lock=locked,
        content_authority=content_authority,
        execution=execution,
    )
    if serialized_inventory != inventory:
        raise ValueError(
            "serialized prepared inventory differs from first-party snapshot extraction"
        )
    if type(content_authority) is not PreparedModelContentAuthorityBinding:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_parameter_inventory_first_party_extractor_unavailable",
            "adapted execution lacks prepared snapshot content authority",
        )
    plan = _derive_plan(
        inventory,
        backend=execution.backend,
        mode=execution.mode,
        scope=execution.scope,
        rank=execution.rank,
        lora_alpha=execution.lora_alpha,
        dspark_native_heads=inventory.dspark_native_heads,
    )
    expected_manifest = _manifest_value(
        plan=plan,
        model_lock=model_lock,
        prepared_drafter=prepared_drafter,
        run_config=run_config,
        split=split,
        cell=cell,
        inventory=inventory,
        content_authority=content_authority,
        execution=execution,
        dspark_native_heads=inventory.dspark_native_heads,
    )
    if manifest_row != expected_manifest:
        raise ValueError(
            "serialized trainable plan differs from the raw parameter-plan reducer"
        )
    binding = _binding_from_replay(
        manifest=manifest,
        model_lock=model_lock,
        prepared_drafter=prepared_drafter,
        run_config=run_config,
        split=split,
        cell=cell,
        content_authority=content_authority,
        inventory=inventory,
        execution=execution,
        plan=plan,
        dspark_native_heads=inventory.dspark_native_heads,
    )
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("trainable-plan authority binding differs from raw replay")
    return TrainablePlanAuthorityResult(
        binding=binding,
        plan=plan,
        prepared_drafter=inventory,
    )


def bind_trainable_plan_authority(
    manifest_path: str | Path,
    *,
    prepared_model_content_authority: (
        PreparedModelContentAuthorityBinding | None
    ) = None,
) -> TrainablePlanAuthorityBinding:
    """Bind and replay a path manifest before returning formal authority."""

    manifest = TrainablePlanRawJsonBinding.from_path(
        manifest_path, role="trainable_plan_authority_manifest"
    )
    row, _ = _selector_from_manifest(manifest.load())
    model_lock = TrainablePlanRawJsonBinding.from_path(
        _resolved_absolute_path(
            row["model_lock_artifact"], label="manifest model lock"
        ),
        role="trainable_plan_model_lock",
    )
    prepared_drafter = TrainablePlanRawJsonBinding.from_path(
        _resolved_absolute_path(
            row["prepared_drafter_artifact"], label="manifest prepared drafter"
        ),
        role="prepared_drafter_parameter_inventory",
    )
    run_config = TrainablePlanRawJsonBinding.from_path(
        _resolved_absolute_path(
            row["run_config_artifact"], label="manifest run config"
        ),
        role="trainable_plan_run_config",
    )
    split = TrainablePlanRawJsonBinding.from_path(
        _resolved_absolute_path(row["split_artifact"], label="manifest split"),
        role="trainable_plan_split",
    )
    cell = TrainablePlanRawJsonBinding.from_path(
        _resolved_absolute_path(row["cell_artifact"], label="manifest cell"),
        role="trainable_plan_cell",
    )
    return _replay(
        manifest=manifest,
        model_lock=model_lock,
        prepared_drafter=prepared_drafter,
        run_config=run_config,
        split=split,
        cell=cell,
        content_authority=prepared_model_content_authority,
        expected_binding=None,
    ).binding


def replay_trainable_plan_authority(
    binding: TrainablePlanAuthorityBinding,
) -> TrainablePlanAuthorityResult:
    """Reopen every raw source and rerun the parameter-plan reducer."""

    if type(binding) is not TrainablePlanAuthorityBinding:
        raise TypeError("trainable-plan replay requires an exact authority binding")
    return _replay(
        manifest=binding.manifest,
        model_lock=binding.model_lock,
        prepared_drafter=binding.prepared_drafter,
        run_config=binding.run_config,
        split=binding.split,
        cell=binding.cell,
        content_authority=binding.prepared_model_content_authority,
        expected_binding=binding,
    )


def trainable_plan_authority_binding_to_dict(
    binding: TrainablePlanAuthorityBinding,
) -> dict[str, object]:
    if type(binding) is not TrainablePlanAuthorityBinding:
        raise TypeError("trainable-plan codec requires an exact authority binding")
    return binding.to_dict()


def trainable_plan_authority_binding_from_dict(
    value: object,
) -> TrainablePlanAuthorityBinding:
    return TrainablePlanAuthorityBinding.from_dict(value)


def audit_trainable_plan_authority_for_method(
    method: str,
    authority: TrainablePlanAuthorityBinding | None,
    *,
    expected_model_lock_sha256: str | None = None,
    expected_prepared_model_content_manifest_sha256: str | None = None,
    expected_run_config_sha256: str | None = None,
    expected_split_sha256: str | None = None,
    expected_cell_id: str | None = None,
    expected_cell_declaration_sha256: str | None = None,
    expected_target_model_id: str | None = None,
    expected_target_revision: str | None = None,
    expected_drafter_model_id: str | None = None,
    expected_prepared_drafter_revision: str | None = None,
    expected_backend: str | None = None,
    expected_mode: str | None = None,
    expected_scope: str | None = None,
    expected_optimizer: str | None = None,
    expected_rank: int | None = None,
    expected_lora_alpha: int | None = None,
) -> TrainablePlan | None:
    """Replay raw closure and caller mirror without granting release trust."""

    if type(method) is not str:
        raise TypeError("trainable-plan method must be a string")
    if method in {"target_only", "static"}:
        if authority is not None:
            raise ValueError(f"{method} must not carry trainable-plan authority")
        return None
    if method not in {"tts", "l0"}:
        raise ValueError("trainable-plan authority supports only core TTS/L0 methods")
    if type(authority) is not TrainablePlanAuthorityBinding:
        raise ValueError(f"{method} requires exact path-bound trainable-plan authority")
    for label, value in (
        ("expected model lock", expected_model_lock_sha256),
        (
            "expected prepared model content manifest",
            expected_prepared_model_content_manifest_sha256,
        ),
        ("expected run config", expected_run_config_sha256),
        ("expected split", expected_split_sha256),
        ("expected cell ID", expected_cell_id),
        ("expected cell declaration", expected_cell_declaration_sha256),
        ("expected target model ID", expected_target_model_id),
        ("expected target revision", expected_target_revision),
        ("expected drafter model ID", expected_drafter_model_id),
        ("expected prepared drafter revision", expected_prepared_drafter_revision),
        ("expected backend", expected_backend),
        ("expected mode", expected_mode),
        ("expected scope", expected_scope),
        ("expected optimizer", expected_optimizer),
    ):
        if value is None:
            raise ValueError(f"{label} is required for adapted execution")
    _require_sha256("expected model lock", expected_model_lock_sha256)
    _require_sha256(
        "expected prepared model content manifest",
        expected_prepared_model_content_manifest_sha256,
    )
    _require_sha256("expected run config", expected_run_config_sha256)
    _require_sha256("expected split", expected_split_sha256)
    _require_sha256("expected cell ID", expected_cell_id)
    _require_sha256("expected cell declaration", expected_cell_declaration_sha256)
    _require_revision("expected target revision", expected_target_revision)
    _require_revision(
        "expected prepared drafter revision", expected_prepared_drafter_revision
    )
    result = replay_trainable_plan_authority(authority)
    actual = (
        result.binding.model_lock_sha256,
        result.binding.prepared_model_content_manifest_sha256,
        result.binding.run_config_sha256,
        result.binding.split_sha256,
        result.binding.cell_id,
        result.binding.cell_declaration_sha256,
        result.binding.target_model_id,
        result.binding.target_revision,
        result.binding.drafter_model_id,
        result.binding.prepared_drafter_revision,
        result.binding.method,
        result.plan.backend,
        result.plan.mode,
        result.plan.scope,
        result.binding.optimizer,
        result.plan.rank,
        result.plan.lora_alpha,
    )
    expected = (
        expected_model_lock_sha256,
        expected_prepared_model_content_manifest_sha256,
        expected_run_config_sha256,
        expected_split_sha256,
        expected_cell_id,
        expected_cell_declaration_sha256,
        expected_target_model_id,
        expected_target_revision,
        expected_drafter_model_id,
        expected_prepared_drafter_revision,
        method,
        expected_backend,
        expected_mode,
        expected_scope,
        expected_optimizer,
        expected_rank,
        expected_lora_alpha,
    )
    if actual != expected:
        raise ValueError("adapted method differs from its trainable-plan authority")
    return result.plan


def require_trainable_plan_authority_for_method(
    method: str,
    authority: TrainablePlanAuthorityBinding | None,
    *,
    expected_model_lock_sha256: str | None = None,
    expected_prepared_model_content_manifest_sha256: str | None = None,
    expected_run_config_sha256: str | None = None,
    expected_split_sha256: str | None = None,
    expected_cell_id: str | None = None,
    expected_cell_declaration_sha256: str | None = None,
    expected_target_model_id: str | None = None,
    expected_target_revision: str | None = None,
    expected_drafter_model_id: str | None = None,
    expected_prepared_drafter_revision: str | None = None,
    expected_backend: str | None = None,
    expected_mode: str | None = None,
    expected_scope: str | None = None,
    expected_optimizer: str | None = None,
    expected_rank: int | None = None,
    expected_lora_alpha: int | None = None,
) -> TrainablePlan | None:
    """Require a source-owned release pin before returning an executable plan."""

    if method in {"target_only", "static"}:
        return audit_trainable_plan_authority_for_method(method, authority)
    if method not in {"tts", "l0"}:
        raise ValueError("trainable-plan authority supports only core TTS/L0 methods")
    if type(authority) is not TrainablePlanAuthorityBinding:
        raise ValueError(f"{method} requires exact path-bound trainable-plan authority")
    if expected_prepared_model_content_manifest_sha256 is None:
        raise PreparedModelContentAuthorityBlocked(
            "prepared_model_content_release_manifest_pin_unavailable",
            "adapted execution lacks a claimed release-manifest mirror",
        )
    require_prepared_model_content_release_manifest_sha256(
        model_lock_sha256=authority.model_lock_sha256,
        prepared=authority.prepared_model_content_authority.prepared_model_set,
        claimed_manifest_sha256=expected_prepared_model_content_manifest_sha256,
    )
    return audit_trainable_plan_authority_for_method(
        method,
        authority,
        expected_model_lock_sha256=expected_model_lock_sha256,
        expected_prepared_model_content_manifest_sha256=(
            expected_prepared_model_content_manifest_sha256
        ),
        expected_run_config_sha256=expected_run_config_sha256,
        expected_split_sha256=expected_split_sha256,
        expected_cell_id=expected_cell_id,
        expected_cell_declaration_sha256=expected_cell_declaration_sha256,
        expected_target_model_id=expected_target_model_id,
        expected_target_revision=expected_target_revision,
        expected_drafter_model_id=expected_drafter_model_id,
        expected_prepared_drafter_revision=expected_prepared_drafter_revision,
        expected_backend=expected_backend,
        expected_mode=expected_mode,
        expected_scope=expected_scope,
        expected_optimizer=expected_optimizer,
        expected_rank=expected_rank,
        expected_lora_alpha=expected_lora_alpha,
    )


__all__ = [
    "DSparkNativeHeadNames",
    "PreparedDrafterParameterInventory",
    "PreparedParameterMetadata",
    "TrainablePlanAuthorityBinding",
    "TrainablePlanAuthorityResult",
    "TrainablePlanRawJsonBinding",
    "audit_trainable_plan_authority_for_method",
    "bind_trainable_plan_authority",
    "materialize_trainable_plan_authority_manifest",
    "replay_trainable_plan_authority",
    "require_trainable_plan_authority_for_method",
    "trainable_plan_authority_binding_from_dict",
    "trainable_plan_authority_binding_to_dict",
]
