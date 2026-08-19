"""Frozen built-in MTP component identity for the two E6 Qwen checkpoints.

Qwen3.5/Qwen3.6 NEXTN does not use an independently downloaded draft model.
The MTP module is embedded in the target checkpoint under ``mtp.*`` tensor
keys.  This module binds that component without reading tensor payloads: it
replays the locked Hugging Face config, the safetensors index, and every
safetensors header named by the index.  The already-frozen target snapshot
content digest remains the authority for payload bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from lightcone_spec.experiments.formal_protocol import (
    E6_MODELS,
    content_sha256,
    reject_banned_model_identity,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedContentFile,
    TrustedModelSnapshotMember,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

NEXTN_BUILT_IN_MTP_MODE = "built_in_mtp"
_CONFIG_NAME = "config.json"
_INDEX_NAME = "model.safetensors.index.json"
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024
_MTP_LAYER = re.compile(r"^mtp\.layers\.([0-9]+)\.")


class FormalSingleOperatorE6BuiltInMtpBlocked(RuntimeError):
    """The frozen target does not expose the registered built-in MTP module."""


def _sha(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be canonical text")
    return value


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _json_bytes(raw: bytes, *, label: str) -> object:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains a forbidden JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _member_file(
    member: TrustedModelSnapshotMember,
    relative_path: str,
) -> TrustedContentFile:
    matches = tuple(row for row in member.files if row.relative_path == relative_path)
    if len(matches) != 1:
        raise FormalSingleOperatorE6BuiltInMtpBlocked(
            f"frozen_snapshot_missing_{relative_path.replace('.', '_')}"
        )
    return matches[0]


def _safe_snapshot_file(
    member: TrustedModelSnapshotMember,
    relative_path: str,
) -> tuple[Path, TrustedContentFile]:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != relative_path
    ):
        raise ValueError("E6 MTP snapshot relative path is unsafe")
    metadata = _member_file(member, relative_path)
    root = Path(member.local_snapshot_path)
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise FormalSingleOperatorE6BuiltInMtpBlocked(
            "frozen_target_snapshot_root_unavailable"
        )
    path = root / relative
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode):
        if member.storage_mode != "huggingface_cache_symlinks":
            raise ValueError("E6 MTP snapshot contains an unauthorized symlink")
        cache = member.content_cache_root
        if cache is None:
            raise ValueError("E6 MTP snapshot cache root is absent")
        cache_root = Path(cache).resolve(strict=True)
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(cache_root)
        except ValueError as error:
            raise ValueError("E6 MTP snapshot symlink leaves its cache root") from error
        if metadata.storage_kind != "symlinked_blob":
            raise ValueError("E6 MTP snapshot symlink metadata differs")
        path = resolved
    elif not stat.S_ISREG(status.st_mode) or metadata.storage_kind != "regular":
        raise ValueError("E6 MTP snapshot file type differs")
    if not path.is_file() or path.is_symlink():
        raise FormalSingleOperatorE6BuiltInMtpBlocked(
            "frozen_target_snapshot_file_unavailable"
        )
    return path, metadata


def _stable_prefix(
    path: Path,
    *,
    size: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < size:
        raise ValueError(f"{label} is not a sufficiently large regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"{label} changed before read")
        raw = bytearray()
        while len(raw) < size:
            chunk = os.read(descriptor, size - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    identity = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise RuntimeError(f"{label} changed during read")
    if len(raw) != size:
        raise ValueError(f"{label} ended before its declared prefix")
    return bytes(raw), after


def _small_json(
    member: TrustedModelSnapshotMember,
    relative_path: str,
    *,
    label: str,
) -> tuple[dict[str, object], str, int]:
    path, metadata = _safe_snapshot_file(member, relative_path)
    if metadata.size < 2 or metadata.size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} size is outside the registered bound")
    raw, observed = _stable_prefix(path, size=metadata.size, label=label)
    digest = hashlib.sha256(raw).hexdigest()
    if observed.st_size != metadata.size or digest != metadata.sha256:
        raise ValueError(f"{label} differs from the frozen snapshot")
    value = _json_bytes(raw, label=label)
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    return dict(value), digest, len(raw)


@dataclass(frozen=True, order=True)
class BuiltInMtpTensorMetadata:
    name: str
    shard_relative_path: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.name.startswith("mtp."):
            raise ValueError("built-in MTP tensor leaves the mtp.* namespace")
        _text("built-in MTP tensor name", self.name)
        _text("built-in MTP shard path", self.shard_relative_path)
        _text("built-in MTP tensor dtype", self.dtype)
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(value) is not int or value < 0 for value in self.shape)
            or type(self.data_offsets) is not tuple
            or len(self.data_offsets) != 2
            or any(type(value) is not int or value < 0 for value in self.data_offsets)
            or self.data_offsets[1] <= self.data_offsets[0]
        ):
            raise ValueError("built-in MTP tensor metadata is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shard_relative_path": self.shard_relative_path,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offsets": list(self.data_offsets),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict("built-in MTP tensor", value, set(cls.__dataclass_fields__))
        shape = row.pop("shape")
        offsets = row.pop("data_offsets")
        if type(shape) is not list or type(offsets) is not list:
            raise TypeError("built-in MTP tensor arrays differ")
        return cls(
            **row,
            shape=tuple(shape),
            data_offsets=tuple(offsets),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, order=True)
class BuiltInMtpSafetensorsHeader:
    shard_relative_path: str
    file_size: int
    file_sha256: str
    header_size: int
    header_sha256: str
    indexed_tensor_count: int

    def __post_init__(self) -> None:
        _text("built-in MTP shard path", self.shard_relative_path)
        _sha("built-in MTP shard file", self.file_sha256)
        _sha("built-in MTP shard header", self.header_sha256)
        if (
            type(self.file_size) is not int
            or self.file_size < 9
            or type(self.header_size) is not int
            or not 2 <= self.header_size <= _MAX_SAFETENSORS_HEADER_BYTES
            or type(self.indexed_tensor_count) is not int
            or self.indexed_tensor_count < 1
        ):
            raise ValueError("built-in MTP safetensors header metadata is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict("built-in MTP header", value, set(cls.__dataclass_fields__))
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorE6BuiltInMtpComponent:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e6_builtin_mtp_component"]
    mode: Literal["built_in_mtp"]
    model_id: str
    revision: str
    target_member_sha256: str
    target_snapshot_tree_sha256: str
    target_snapshot_sha256: str
    snapshot_root: str
    config_relative_path: Literal["config.json"]
    config_raw_sha256: str
    config_size: int
    model_type: Literal["qwen3_5_moe"]
    text_model_type: Literal["qwen3_5_moe_text"]
    mtp_num_hidden_layers: Literal[1]
    mtp_use_dedicated_embeddings: Literal[False]
    weight_index_relative_path: Literal["model.safetensors.index.json"]
    weight_index_raw_sha256: str
    weight_index_size: int
    headers: tuple[BuiltInMtpSafetensorsHeader, ...]
    tensors: tuple[BuiltInMtpTensorMetadata, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e6_builtin_mtp_component"
            or self.mode != NEXTN_BUILT_IN_MTP_MODE
            or self.model_id not in E6_MODELS
            or self.model_type != "qwen3_5_moe"
            or self.text_model_type != "qwen3_5_moe_text"
            or self.mtp_num_hidden_layers != 1
            or self.mtp_use_dedicated_embeddings is not False
            or self.config_relative_path != _CONFIG_NAME
            or self.weight_index_relative_path != _INDEX_NAME
        ):
            raise ValueError("E6 built-in MTP component identity differs")
        _text("E6 built-in MTP revision", self.revision)
        for label, digest in (
            ("target member", self.target_member_sha256),
            ("target tree", self.target_snapshot_tree_sha256),
            ("target snapshot", self.target_snapshot_sha256),
            ("config raw", self.config_raw_sha256),
            ("weight index raw", self.weight_index_raw_sha256),
        ):
            _sha(f"E6 built-in MTP {label}", digest)
        root = Path(self.snapshot_root)
        if not root.is_absolute() or root != root.resolve(strict=False):
            raise ValueError("E6 built-in MTP snapshot root is not normalized")
        if (
            type(self.config_size) is not int
            or self.config_size < 2
            or type(self.weight_index_size) is not int
            or self.weight_index_size < 2
            or not self.headers
            or self.headers != tuple(sorted(set(self.headers)))
            or not self.tensors
            or self.tensors != tuple(sorted(set(self.tensors)))
            or any(type(row) is not BuiltInMtpSafetensorsHeader for row in self.headers)
            or any(type(row) is not BuiltInMtpTensorMetadata for row in self.tensors)
        ):
            raise ValueError("E6 built-in MTP component coverage differs")
        names = {row.name for row in self.tensors}
        required = {
            "mtp.fc.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        }
        layer_ids = {
            int(match.group(1))
            for name in names
            if (match := _MTP_LAYER.match(name)) is not None
        }
        if not required <= names or layer_ids != {0}:
            raise ValueError("E6 built-in MTP structural tensors are incomplete")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "mode": self.mode,
            "model_id": self.model_id,
            "revision": self.revision,
            "target_member_sha256": self.target_member_sha256,
            "target_snapshot_tree_sha256": self.target_snapshot_tree_sha256,
            "target_snapshot_sha256": self.target_snapshot_sha256,
            "snapshot_root": self.snapshot_root,
            "config_relative_path": self.config_relative_path,
            "config_raw_sha256": self.config_raw_sha256,
            "config_size": self.config_size,
            "model_type": self.model_type,
            "text_model_type": self.text_model_type,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
            "mtp_use_dedicated_embeddings": self.mtp_use_dedicated_embeddings,
            "weight_index_relative_path": self.weight_index_relative_path,
            "weight_index_raw_sha256": self.weight_index_raw_sha256,
            "weight_index_size": self.weight_index_size,
            "headers": [row.to_dict() for row in self.headers],
            "tensors": [row.to_dict() for row in self.tensors],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict("E6 built-in MTP component", value, set(cls.__dataclass_fields__))
        headers = row.pop("headers")
        tensors = row.pop("tensors")
        if type(headers) is not list or type(tensors) is not list:
            raise TypeError("E6 built-in MTP component arrays differ")
        return cls(
            **row,
            headers=tuple(
                BuiltInMtpSafetensorsHeader.from_dict(item) for item in headers
            ),
            tensors=tuple(BuiltInMtpTensorMetadata.from_dict(item) for item in tensors),
        )  # type: ignore[arg-type]


def _header_metadata(
    member: TrustedModelSnapshotMember,
    relative_path: str,
    *,
    indexed_names: tuple[str, ...],
) -> tuple[BuiltInMtpSafetensorsHeader, dict[str, object]]:
    path, frozen = _safe_snapshot_file(member, relative_path)
    prefix, status = _stable_prefix(path, size=8, label="E6 safetensors prefix")
    header_size = int.from_bytes(prefix, byteorder="little", signed=False)
    if not 2 <= header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError("E6 safetensors header length is outside the safe bound")
    raw_header, repeated = _stable_prefix(
        path,
        size=8 + header_size,
        label="E6 safetensors header",
    )
    if (
        status.st_size != repeated.st_size
        or repeated.st_size != frozen.size
        or int.from_bytes(raw_header[:8], byteorder="little", signed=False)
        != header_size
    ):
        raise ValueError("E6 safetensors file differs from frozen metadata")
    value = _json_bytes(raw_header[8:], label="E6 safetensors header")
    if type(value) is not dict:
        raise TypeError("E6 safetensors header must be a JSON object")
    header = dict(value)
    tensor_names = tuple(sorted(name for name in header if name != "__metadata__"))
    if tensor_names != indexed_names:
        raise ValueError("E6 safetensors header/index tensor sets differ")
    return (
        BuiltInMtpSafetensorsHeader(
            shard_relative_path=relative_path,
            file_size=repeated.st_size,
            file_sha256=frozen.sha256,
            header_size=header_size,
            header_sha256=hashlib.sha256(raw_header[8:]).hexdigest(),
            indexed_tensor_count=len(tensor_names),
        ),
        header,
    )


def scan_formal_single_operator_e6_builtin_mtp_component(
    member: TrustedModelSnapshotMember,
) -> FormalSingleOperatorE6BuiltInMtpComponent:
    """Scan one frozen E6 target using only config/index/header metadata."""

    if type(member) is not TrustedModelSnapshotMember:
        raise TypeError("E6 built-in MTP scan requires an exact trusted member")
    bindings = tuple(
        row
        for row in member.runtime_bindings
        if row.stage == "E6"
        and row.backend == "NEXTN"
        and row.target_model_id == member.model_id
    )
    if (
        member.role != "target"
        or member.model_id not in E6_MODELS
        or "E6" not in member.stages
        or len(bindings) != 1
    ):
        raise FormalSingleOperatorE6BuiltInMtpBlocked(
            "exact_e6_target_runtime_binding_unavailable"
        )
    config, config_sha, config_size = _small_json(
        member,
        _CONFIG_NAME,
        label="E6 target config",
    )
    text_config = config.get("text_config")
    if (
        config.get("model_type") != "qwen3_5_moe"
        or type(text_config) is not dict
        or text_config.get("model_type") != "qwen3_5_moe_text"
        or text_config.get("mtp_num_hidden_layers") != 1
        or text_config.get("mtp_use_dedicated_embeddings") is not False
    ):
        raise FormalSingleOperatorE6BuiltInMtpBlocked(
            "registered_qwen35_builtin_mtp_config_unavailable"
        )
    index, index_sha, index_size = _small_json(
        member,
        _INDEX_NAME,
        label="E6 safetensors index",
    )
    if (
        set(index) != {"metadata", "weight_map"}
        or type(index["weight_map"]) is not dict
    ):
        raise ValueError("E6 safetensors index schema differs")
    weight_map = index["weight_map"]
    assert type(weight_map) is dict
    if not weight_map:
        raise FormalSingleOperatorE6BuiltInMtpBlocked("empty_safetensors_weight_index")
    by_shard: dict[str, list[str]] = {}
    for raw_name, raw_shard in weight_map.items():
        name = _text("E6 indexed tensor", raw_name)
        shard = _text("E6 indexed shard", raw_shard)
        shard_path = PurePosixPath(shard)
        if (
            shard_path.is_absolute()
            or shard_path.as_posix() != shard
            or not shard_path.parts
            or any(part in {"", ".", ".."} for part in shard_path.parts)
            or shard_path.suffix != ".safetensors"
        ):
            raise ValueError("E6 safetensors index contains an unsafe shard path")
        by_shard.setdefault(shard, []).append(name)
    headers: list[BuiltInMtpSafetensorsHeader] = []
    tensor_rows: list[BuiltInMtpTensorMetadata] = []
    mtp_shards = {
        shard
        for shard, names in by_shard.items()
        if any(name.startswith("mtp.") for name in names)
    }
    if not mtp_shards:
        raise FormalSingleOperatorE6BuiltInMtpBlocked(
            "safetensors_index_has_no_mtp_component"
        )
    for shard in sorted(mtp_shards):
        indexed = tuple(sorted(by_shard[shard]))
        header_receipt, header = _header_metadata(
            member,
            shard,
            indexed_names=indexed,
        )
        headers.append(header_receipt)
        data_capacity = header_receipt.file_size - 8 - header_receipt.header_size
        intervals: list[tuple[int, int, str]] = []
        for name in indexed:
            metadata = header[name]
            row = _strict(
                "E6 safetensors tensor metadata",
                metadata,
                {"dtype", "shape", "data_offsets"},
            )
            shape = row["shape"]
            offsets = row["data_offsets"]
            if type(shape) is not list or type(offsets) is not list:
                raise TypeError("E6 safetensors tensor arrays differ")
            if (
                len(offsets) != 2
                or any(type(value) is not int or value < 0 for value in offsets)
                or offsets[1] <= offsets[0]
                or offsets[1] > data_capacity
            ):
                raise ValueError("E6 safetensors tensor offsets are invalid")
            intervals.append((offsets[0], offsets[1], name))
            if name.startswith("mtp."):
                tensor_rows.append(
                    BuiltInMtpTensorMetadata(
                        name=name,
                        shard_relative_path=shard,
                        dtype=_text("E6 safetensors dtype", row["dtype"]),
                        shape=tuple(shape),
                        data_offsets=tuple(offsets),  # type: ignore[arg-type]
                    )
                )
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in pairwise(ordered)):
            raise ValueError("E6 safetensors tensor ranges overlap")
    return FormalSingleOperatorE6BuiltInMtpComponent(
        schema_version=1,
        kind="formal_single_operator_e6_builtin_mtp_component",
        mode="built_in_mtp",
        model_id=member.model_id,
        revision=member.revision,
        target_member_sha256=member.sha256,
        target_snapshot_tree_sha256=member.tree_sha256,
        target_snapshot_sha256=member.content_sha256,
        snapshot_root=member.local_snapshot_path,
        config_relative_path="config.json",
        config_raw_sha256=config_sha,
        config_size=config_size,
        model_type="qwen3_5_moe",
        text_model_type="qwen3_5_moe_text",
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        weight_index_relative_path="model.safetensors.index.json",
        weight_index_raw_sha256=index_sha,
        weight_index_size=index_size,
        headers=tuple(sorted(headers)),
        tensors=tuple(sorted(tensor_rows)),
    )


def publish_formal_single_operator_e6_builtin_mtp_component(
    member: TrustedModelSnapshotMember,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    component = scan_formal_single_operator_e6_builtin_mtp_component(member)
    publish_canonical_json_no_replace(output_path, component.to_dict())
    return CanonicalJsonProofBinding.bind(output_path, semantic_sha256=component.sha256)


def revalidate_formal_single_operator_e6_builtin_mtp_component(
    path: str | Path,
    *,
    member: TrustedModelSnapshotMember,
) -> FormalSingleOperatorE6BuiltInMtpComponent:
    binding = CanonicalJsonProofBinding.bind(path)
    declared = FormalSingleOperatorE6BuiltInMtpComponent.from_dict(binding.reopen())
    observed = scan_formal_single_operator_e6_builtin_mtp_component(member)
    if declared != observed or binding.semantic_sha256 != observed.sha256:
        raise ValueError("E6 built-in MTP component changed")
    return observed


__all__ = [
    "NEXTN_BUILT_IN_MTP_MODE",
    "BuiltInMtpSafetensorsHeader",
    "BuiltInMtpTensorMetadata",
    "FormalSingleOperatorE6BuiltInMtpBlocked",
    "FormalSingleOperatorE6BuiltInMtpComponent",
    "publish_formal_single_operator_e6_builtin_mtp_component",
    "revalidate_formal_single_operator_e6_builtin_mtp_component",
    "scan_formal_single_operator_e6_builtin_mtp_component",
]
