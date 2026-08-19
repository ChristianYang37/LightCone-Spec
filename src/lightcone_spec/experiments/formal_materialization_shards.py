"""Bounded path graph for large formal stage materializations.

The global GPU-proof primitive intentionally caps one canonical JSON object at
2 MiB.  Large E3b/E5/E6/E0 cell matrices therefore use a small index plus
canonical ordered cell shards.  Revalidation reconstructs the original typed
``StageMaterializationReceipt`` and requires its exact preregistered SHA-256;
the shard graph is only a transport representation and never a new scientific
identity.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    SignedStageMaterializationReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_MATERIALIZATION_SHARD_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_materialization_shard_protocol",
        "header": "all_non_cell_stage_materialization_fields",
        "cells": "canonical_cell_id_order_bounded_shards",
        "coverage": "every_and_only_expected_cell_once",
        "identity": "reconstructed_stage_materialization_receipt_sha256",
        "publication": "atomic_no_replace_per_file_new_bundle_on_partial_failure",
    }
)


def _sha(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _canonical_output_path(label: str, value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError(f"{label} must be absolute and normalized")
    return str(path)


def _cell_to_dict(cell: MaterializedCell) -> dict[str, object]:
    if type(cell) is not MaterializedCell:
        raise TypeError("materialization shard requires exact cells")
    return {
        "stage": cell.stage,
        "method_role": cell.method_role,
        "model": cell.model,
        "backend": cell.backend,
        "task": cell.task,
        "publication_policy": cell.publication_policy,
        "recipe_sha256": cell.recipe_sha256,
        "dimensions": [list(row) for row in cell.dimensions],
        "cell_id": cell.cell_id,
    }


def _cell_from_dict(value: object) -> MaterializedCell:
    fields = {*MaterializedCell.__dataclass_fields__, "cell_id"}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("materialization shard cell fields differ")
    row = dict(value)
    declared = _sha("materialization shard cell", row.pop("cell_id"))
    dimensions = row["dimensions"]
    if type(dimensions) is not list or any(
        type(item) is not list or len(item) != 2 for item in dimensions
    ):
        raise TypeError("materialization shard dimensions must be pair arrays")
    row["dimensions"] = tuple((item[0], item[1]) for item in dimensions)
    cell = MaterializedCell(**row)  # type: ignore[arg-type]
    if cell.cell_id != declared:
        raise ValueError("materialization shard cell digest differs")
    return cell


@dataclass(frozen=True)
class FormalMaterializationShardHeader:
    schema_version: Literal[1]
    stage: str
    protocol_lock_sha256: str
    upstream_receipt_sha256s: tuple[str, ...]
    source_decision_sha256: str
    materialization_rule: str
    expected_cell_count: int
    gpu_hours: GpuHourEstimate

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("materialization shard header schema differs")
        _sha("materialization shard protocol lock", self.protocol_lock_sha256)
        _sha("materialization shard source decision", self.source_decision_sha256)
        if type(self.upstream_receipt_sha256s) is not tuple:
            raise TypeError("materialization shard upstream rows must be a tuple")
        for digest in self.upstream_receipt_sha256s:
            _sha("materialization shard upstream", digest)
        if (
            type(self.stage) is not str
            or not self.stage
            or type(self.materialization_rule) is not str
            or not self.materialization_rule
            or type(self.expected_cell_count) is not int
            or self.expected_cell_count < 0
            or type(self.gpu_hours) is not GpuHourEstimate
        ):
            raise ValueError("materialization shard header fields are invalid")

    @classmethod
    def from_receipt(cls, receipt: StageMaterializationReceipt) -> Self:
        if type(receipt) is not StageMaterializationReceipt:
            raise TypeError("materialization shard header requires exact receipt")
        return cls(
            schema_version=receipt.schema_version,
            stage=receipt.stage,
            protocol_lock_sha256=receipt.protocol_lock_sha256,
            upstream_receipt_sha256s=receipt.upstream_receipt_sha256s,
            source_decision_sha256=receipt.source_decision_sha256,
            materialization_rule=receipt.materialization_rule,
            expected_cell_count=receipt.expected_cell_count,
            gpu_hours=receipt.gpu_hours,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "upstream_receipt_sha256s": list(self.upstream_receipt_sha256s),
            "source_decision_sha256": self.source_decision_sha256,
            "materialization_rule": self.materialization_rule,
            "expected_cell_count": self.expected_cell_count,
            "gpu_hours": asdict(self.gpu_hours),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("materialization shard header fields differ")
        row = dict(value)
        upstream = row["upstream_receipt_sha256s"]
        gpu_hours = row["gpu_hours"]
        if (
            type(upstream) is not list
            or type(gpu_hours) is not dict
            or set(gpu_hours) != set(GpuHourEstimate.__dataclass_fields__)
        ):
            raise TypeError("materialization shard header collections differ")
        row["upstream_receipt_sha256s"] = tuple(upstream)
        row["gpu_hours"] = GpuHourEstimate(**gpu_hours)
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalMaterializationCellShard:
    schema_version: Literal[1]
    kind: Literal["formal_materialization_cell_shard"]
    protocol_sha256: str
    materialization_receipt_sha256: str
    stage: str
    shard_index: int
    shard_count: int
    cells: tuple[MaterializedCell, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_materialization_cell_shard"
            or self.protocol_sha256 != FORMAL_MATERIALIZATION_SHARD_PROTOCOL_SHA256
        ):
            raise ValueError("materialization cell shard identity differs")
        _sha("materialization cell shard receipt", self.materialization_receipt_sha256)
        ids = tuple(row.cell_id for row in self.cells)
        if (
            type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or self.shard_index not in range(self.shard_count)
            or not self.cells
            or any(type(row) is not MaterializedCell for row in self.cells)
            or any(row.stage != self.stage for row in self.cells)
            or ids != tuple(sorted(set(ids)))
        ):
            raise ValueError("materialization cell shard rows are not canonical")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "stage": self.stage,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "cells": [_cell_to_dict(row) for row in self.cells],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "shard_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("materialization cell shard fields differ")
        row = dict(value)
        declared = _sha("materialization cell shard", row.pop("shard_sha256"))
        cells = row["cells"]
        if type(cells) is not list:
            raise TypeError("materialization cell shard cells must be an array")
        row["cells"] = tuple(_cell_from_dict(item) for item in cells)
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("materialization cell shard digest differs")
        return shard


@dataclass(frozen=True)
class FormalMaterializationShardIndex:
    schema_version: Literal[1]
    kind: Literal["formal_materialization_shard_index"]
    protocol_sha256: str
    materialization_receipt_sha256: str
    header: FormalMaterializationShardHeader
    cell_shard_sources: tuple[CanonicalJsonProofBinding, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_materialization_shard_index"
            or self.protocol_sha256 != FORMAL_MATERIALIZATION_SHARD_PROTOCOL_SHA256
        ):
            raise ValueError("materialization shard index identity differs")
        _sha("materialization shard index receipt", self.materialization_receipt_sha256)
        if (
            type(self.header) is not FormalMaterializationShardHeader
            or type(self.cell_shard_sources) is not tuple
            or not self.cell_shard_sources
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.cell_shard_sources
            )
        ):
            raise TypeError("materialization shard index sources are not exact")
        paths = tuple(row.absolute_path for row in self.cell_shard_sources)
        if len(paths) != len(set(paths)):
            raise ValueError("materialization shard index aliases a source path")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "header": self.header.to_dict(),
            "cell_shard_sources": [row.to_dict() for row in self.cell_shard_sources],
        }
        if include_sha256:
            value["index_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "index_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("materialization shard index fields differ")
        row = dict(value)
        declared = _sha("materialization shard index", row.pop("index_sha256"))
        sources = row["cell_shard_sources"]
        if type(sources) is not list:
            raise TypeError("materialization shard sources must be an array")
        row["header"] = FormalMaterializationShardHeader.from_dict(row["header"])
        row["cell_shard_sources"] = tuple(
            CanonicalJsonProofBinding.from_dict(item) for item in sources
        )
        index = cls(**row)  # type: ignore[arg-type]
        if index.sha256 != declared:
            raise ValueError("materialization shard index digest differs")
        return index


@dataclass(frozen=True)
class FormalSignedMaterializationShardWrapper:
    """Compact signature header for a sharded materialization payload."""

    schema_version: Literal[1]
    kind: Literal["formal_signed_materialization_shard_wrapper"]
    protocol_sha256: str
    materialization_receipt_sha256: str
    signed_materialization_receipt_sha256: str
    materialization_index_source: CanonicalJsonProofBinding
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_signed_materialization_shard_wrapper"
            or self.protocol_sha256 != FORMAL_MATERIALIZATION_SHARD_PROTOCOL_SHA256
        ):
            raise ValueError("signed materialization shard wrapper identity differs")
        _sha("signed materialization shard payload", self.payload_sha256)
        _sha(
            "signed materialization shard receipt",
            self.materialization_receipt_sha256,
        )
        _sha(
            "signed materialization wrapper",
            self.signed_materialization_receipt_sha256,
        )
        if type(self.materialization_index_source) is not CanonicalJsonProofBinding:
            raise TypeError("signed materialization shard index must be path-bound")
        if type(self.challenge) is not AttestationChallenge:
            raise TypeError("signed materialization shard challenge must be exact")
        if type(self.attestation) is not SignedAttestation:
            raise TypeError("signed materialization shard attestation must be exact")
        self.challenge.validate()
        self.attestation.validate()
        if (
            self.payload_sha256 != self.materialization_receipt_sha256
            or self.challenge.subject_sha256 != self.materialization_receipt_sha256
            or self.attestation.payload_sha256 != self.payload_sha256
            or self.attestation.challenge_sha256 != self.challenge.sha256
        ):
            raise ValueError("signed materialization shard signature lineage differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "signed_materialization_receipt_sha256": (
                self.signed_materialization_receipt_sha256
            ),
            "materialization_index_source": (
                self.materialization_index_source.to_dict()
            ),
            "payload_sha256": self.payload_sha256,
            "challenge": asdict(self.challenge),
            "attestation": asdict(self.attestation),
        }
        if include_sha256:
            value["wrapper_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "wrapper_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("signed materialization shard wrapper fields differ")
        row = dict(value)
        declared = _sha(
            "signed materialization shard wrapper", row.pop("wrapper_sha256")
        )
        challenge = row["challenge"]
        attestation = row["attestation"]
        if (
            type(challenge) is not dict
            or set(challenge) != set(AttestationChallenge.__dataclass_fields__)
            or type(attestation) is not dict
            or set(attestation) != set(SignedAttestation.__dataclass_fields__)
        ):
            raise ValueError("signed materialization signature fields differ")
        row["challenge"] = AttestationChallenge(**challenge)
        row["attestation"] = SignedAttestation(**attestation)
        row["materialization_index_source"] = CanonicalJsonProofBinding.from_dict(
            row["materialization_index_source"]
        )
        wrapper = cls(**row)  # type: ignore[arg-type]
        if wrapper.sha256 != declared:
            raise ValueError("signed materialization shard wrapper digest differs")
        return wrapper


def publish_formal_materialization_shard_index(
    receipt: StageMaterializationReceipt,
    *,
    cell_shard_output_paths: tuple[str | Path, ...],
    index_output_path: str | Path,
    maximum_cells_per_shard: int = 256,
) -> CanonicalJsonProofBinding:
    """Publish one bounded no-replace graph for an existing receipt."""

    if type(receipt) is not StageMaterializationReceipt:
        raise TypeError("materialization shard publisher requires exact receipt")
    if (
        type(maximum_cells_per_shard) is not int
        or maximum_cells_per_shard < 1
        or maximum_cells_per_shard > 512
    ):
        raise ValueError("materialization shard cell bound is outside [1,512]")
    chunks = tuple(
        receipt.cells[index : index + maximum_cells_per_shard]
        for index in range(0, len(receipt.cells), maximum_cells_per_shard)
    )
    if not chunks:
        raise ValueError("zero-cell materialization uses its dedicated N/A proof")
    if type(cell_shard_output_paths) is not tuple or len(
        cell_shard_output_paths
    ) != len(chunks):
        raise ValueError("materialization shard output path count differs")
    output_paths = tuple(
        _canonical_output_path("materialization cell shard", path)
        for path in cell_shard_output_paths
    )
    index_path = _canonical_output_path(
        "materialization shard index", index_output_path
    )
    if len({*output_paths, index_path}) != len(output_paths) + 1:
        raise ValueError("materialization shard output paths alias")
    sources = []
    for index, (chunk, output_path) in enumerate(
        zip(chunks, output_paths, strict=True)
    ):
        shard = FormalMaterializationCellShard(
            schema_version=1,
            kind="formal_materialization_cell_shard",
            protocol_sha256=FORMAL_MATERIALIZATION_SHARD_PROTOCOL_SHA256,
            materialization_receipt_sha256=receipt.sha256,
            stage=receipt.stage,
            shard_index=index,
            shard_count=len(chunks),
            cells=chunk,
        )
        publish_canonical_json_no_replace(output_path, shard.to_dict())
        sources.append(CanonicalJsonProofBinding.bind(output_path))
    index = FormalMaterializationShardIndex(
        schema_version=1,
        kind="formal_materialization_shard_index",
        protocol_sha256=FORMAL_MATERIALIZATION_SHARD_PROTOCOL_SHA256,
        materialization_receipt_sha256=receipt.sha256,
        header=FormalMaterializationShardHeader.from_receipt(receipt),
        cell_shard_sources=tuple(sources),
    )
    publish_canonical_json_no_replace(index_path, index.to_dict())
    binding = CanonicalJsonProofBinding.bind(index_path)
    if revalidate_formal_materialization_shard_index(index_path) != receipt:
        raise RuntimeError("published materialization shard graph changed")
    return binding


def publish_formal_signed_materialization_shard_wrapper(
    signed_materialization: SignedStageMaterializationReceipt,
    *,
    materialization_index_source: CanonicalJsonProofBinding,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish a bounded signature header without embedding the cell payload."""

    if type(signed_materialization) is not SignedStageMaterializationReceipt:
        raise TypeError("signed materialization shard publisher requires exact input")
    if type(materialization_index_source) is not CanonicalJsonProofBinding:
        raise TypeError("signed materialization shard publisher requires an index")
    observed = CanonicalJsonProofBinding.bind(
        materialization_index_source.absolute_path
    )
    if observed != materialization_index_source:
        raise ValueError("materialization shard index path identity changed")
    rebuilt = revalidate_formal_materialization_shard_index(
        observed.absolute_path,
        expected_materialization_receipt_sha256=signed_materialization.payload.sha256,
    )
    if (
        rebuilt != signed_materialization.payload
        or signed_materialization.payload_sha256 != rebuilt.sha256
    ):
        raise ValueError("signed materialization differs from its shard graph")
    wrapper = FormalSignedMaterializationShardWrapper(
        schema_version=1,
        kind="formal_signed_materialization_shard_wrapper",
        protocol_sha256=FORMAL_MATERIALIZATION_SHARD_PROTOCOL_SHA256,
        materialization_receipt_sha256=rebuilt.sha256,
        signed_materialization_receipt_sha256=signed_materialization.sha256,
        materialization_index_source=observed,
        payload_sha256=signed_materialization.payload_sha256,
        challenge=signed_materialization.challenge,
        attestation=signed_materialization.attestation,
    )
    destination = _canonical_output_path(
        "signed materialization shard wrapper", output_path
    )
    publish_canonical_json_no_replace(destination, wrapper.to_dict())
    return CanonicalJsonProofBinding.bind(destination)


def rebuild_formal_signed_materialization_shard_wrapper(
    wrapper_path: str | Path,
) -> SignedStageMaterializationReceipt:
    """Deep-rebuild the original signed wrapper from its compact path graph."""

    binding = CanonicalJsonProofBinding.bind(wrapper_path)
    wrapper = FormalSignedMaterializationShardWrapper.from_dict(binding.reopen())
    observed_index = CanonicalJsonProofBinding.bind(
        wrapper.materialization_index_source.absolute_path
    )
    if observed_index != wrapper.materialization_index_source:
        raise ValueError("signed materialization shard index identity changed")
    receipt = revalidate_formal_materialization_shard_index(
        observed_index.absolute_path,
        expected_materialization_receipt_sha256=(
            wrapper.materialization_receipt_sha256
        ),
    )
    signed = SignedStageMaterializationReceipt(
        payload=receipt,
        payload_sha256=wrapper.payload_sha256,
        challenge=wrapper.challenge,
        attestation=wrapper.attestation,
    )
    if signed.sha256 != wrapper.signed_materialization_receipt_sha256:
        raise ValueError("signed materialization shard reconstruction differs")
    return signed


def revalidate_formal_signed_materialization_shard_wrapper(
    wrapper_path: str | Path,
    *,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
) -> SignedStageMaterializationReceipt:
    """Deep-rebuild and cryptographically revalidate a compact wrapper."""

    signed = rebuild_formal_signed_materialization_shard_wrapper(wrapper_path)
    signed.verify(
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    return signed


def revalidate_formal_materialization_shard_index(
    index_path: str | Path,
    *,
    expected_materialization_receipt_sha256: str | None = None,
) -> StageMaterializationReceipt:
    """Deep-open every shard and reconstruct the original receipt identity."""

    if expected_materialization_receipt_sha256 is not None:
        _sha(
            "expected materialization shard receipt",
            expected_materialization_receipt_sha256,
        )
    index_binding = CanonicalJsonProofBinding.bind(index_path)
    index = FormalMaterializationShardIndex.from_dict(index_binding.reopen())
    shards = []
    for source in index.cell_shard_sources:
        observed = CanonicalJsonProofBinding.bind(source.absolute_path)
        if observed != source:
            raise ValueError("materialization cell shard path identity changed")
        shards.append(FormalMaterializationCellShard.from_dict(observed.reopen()))
    rows = tuple(shards)
    if (
        tuple(row.shard_index for row in rows) != tuple(range(len(rows)))
        or any(row.shard_count != len(rows) for row in rows)
        or any(
            row.materialization_receipt_sha256 != index.materialization_receipt_sha256
            or row.stage != index.header.stage
            for row in rows
        )
    ):
        raise ValueError("materialization cell shard graph is incomplete")
    cells = tuple(cell for shard in rows for cell in shard.cells)
    ids = tuple(row.cell_id for row in cells)
    if len(cells) != index.header.expected_cell_count or ids != tuple(sorted(set(ids))):
        raise ValueError("materialization shard union is missing/extra/duplicated")
    header = index.header
    receipt = StageMaterializationReceipt(
        schema_version=header.schema_version,
        stage=header.stage,
        protocol_lock_sha256=header.protocol_lock_sha256,
        upstream_receipt_sha256s=header.upstream_receipt_sha256s,
        source_decision_sha256=header.source_decision_sha256,
        materialization_rule=header.materialization_rule,
        expected_cell_count=header.expected_cell_count,
        cells=cells,
        gpu_hours=header.gpu_hours,
    )
    if receipt.sha256 != index.materialization_receipt_sha256 or (
        expected_materialization_receipt_sha256 is not None
        and receipt.sha256 != expected_materialization_receipt_sha256
    ):
        raise ValueError("materialization shard reconstruction digest differs")
    return receipt


__all__ = (
    "FORMAL_MATERIALIZATION_SHARD_PROTOCOL_SHA256",
    "FormalMaterializationCellShard",
    "FormalMaterializationShardHeader",
    "FormalMaterializationShardIndex",
    "FormalSignedMaterializationShardWrapper",
    "publish_formal_materialization_shard_index",
    "publish_formal_signed_materialization_shard_wrapper",
    "rebuild_formal_signed_materialization_shard_wrapper",
    "revalidate_formal_materialization_shard_index",
    "revalidate_formal_signed_materialization_shard_wrapper",
)
