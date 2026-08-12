"""Strict bindings for revision-addressed local model snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .models import ModelLock


def _canonical_sha256(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


PREPARED_MODEL_BINDING_PROTOCOL_SHA256 = _canonical_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_prepared_model_binding_protocol",
        "requirements": [
            "exact_schema_v2_model_lock",
            "model_id_sorted_unique_complete_coverage",
            "absolute_resolved_regular_snapshot_directory",
            "snapshot_parent_component_is_snapshots",
            "snapshot_leaf_equals_locked_revision",
        ],
    }
)


@dataclass(frozen=True)
class PreparedModelSnapshot:
    """One model ID bound to its immutable local revision directory."""

    model_id: str
    revision: str
    root: str

    def validate(self) -> None:
        if not self.model_id:
            raise ValueError("prepared model ID must be non-empty")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("prepared model revision must be an immutable Git SHA")
        root = Path(self.root)
        if not root.is_absolute() or root.resolve() != root:
            raise ValueError("prepared model root must be absolute and resolved")
        if root.is_symlink() or not root.is_dir():
            raise ValueError("prepared model root must be a regular directory")
        if root.name != self.revision or root.parent.name != "snapshots":
            raise ValueError(
                "prepared model root must be the locked revision snapshot directory"
            )

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "root": self.root,
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelSnapshot:
        if type(value) is not dict or set(value) != {"model_id", "revision", "root"}:
            raise ValueError("prepared model snapshot fields are invalid")
        if any(type(value[name]) is not str for name in value):
            raise TypeError("prepared model snapshot values must be strings")
        result = cls(
            model_id=value["model_id"],
            revision=value["revision"],
            root=value["root"],
        )
        result.validate()
        return result


@dataclass(frozen=True)
class PreparedModelSet:
    """Complete local materialization of one exact :class:`ModelLock`."""

    schema_version: int
    kind: str
    model_lock_sha256: str
    snapshots: tuple[PreparedModelSnapshot, ...]
    protocol_sha256: str = PREPARED_MODEL_BINDING_PROTOCOL_SHA256

    def validate(self) -> None:
        if self.schema_version != 1 or self.kind != "lightcone_prepared_model_set":
            raise ValueError("prepared model set schema is unsupported")
        for name, value in (
            ("model lock", self.model_lock_sha256),
            ("protocol", self.protocol_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"prepared model {name} digest is invalid")
        if self.protocol_sha256 != PREPARED_MODEL_BINDING_PROTOCOL_SHA256:
            raise ValueError("prepared model binding protocol is unsupported")
        if not self.snapshots or any(
            type(snapshot) is not PreparedModelSnapshot for snapshot in self.snapshots
        ):
            raise TypeError("prepared model set requires exact snapshot bindings")
        for snapshot in self.snapshots:
            snapshot.validate()
        identities = tuple(snapshot.model_id for snapshot in self.snapshots)
        if identities != tuple(sorted(set(identities))):
            raise ValueError(
                "prepared model snapshots must be model-ID sorted and unique"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "model_lock_sha256": self.model_lock_sha256,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "protocol_sha256": self.protocol_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> PreparedModelSet:
        fields = {
            "schema_version",
            "kind",
            "model_lock_sha256",
            "snapshots",
            "protocol_sha256",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("prepared model set fields are invalid")
        if type(value["snapshots"]) is not list:
            raise TypeError("prepared model snapshots must be a list")
        result = cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            model_lock_sha256=value["model_lock_sha256"],
            snapshots=tuple(
                PreparedModelSnapshot.from_dict(item) for item in value["snapshots"]
            ),
            protocol_sha256=value["protocol_sha256"],
        )
        result.validate()
        return result


def bind_prepared_models(
    model_lock: ModelLock,
    roots: Mapping[str, str | Path],
) -> PreparedModelSet:
    """Bind exact Hugging Face snapshot roots to a validated model lock."""

    if type(model_lock) is not ModelLock:
        raise TypeError("prepared model binding requires an exact ModelLock")
    model_lock.validate()
    locked = {model.model_id: model.revision for model in model_lock.models}
    if set(roots) != set(locked):
        raise ValueError("prepared model roots do not cover the model lock exactly")
    canonical_roots: dict[str, str] = {}
    for model_id in locked:
        root = Path(roots[model_id])
        if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
            raise ValueError(
                "prepared model root must be the locked revision snapshot directory"
            )
        canonical_roots[model_id] = str(root)
    result = PreparedModelSet(
        schema_version=1,
        kind="lightcone_prepared_model_set",
        model_lock_sha256=model_lock.sha256,
        snapshots=tuple(
            PreparedModelSnapshot(
                model_id=model_id,
                revision=locked[model_id],
                root=canonical_roots[model_id],
            )
            for model_id in sorted(locked)
        ),
    )
    result.validate()
    return result


def revalidate_prepared_models(
    model_lock: ModelLock,
    prepared: PreparedModelSet,
) -> dict[str, str]:
    """Reopen and exact-compare a prepared model binding before launch."""

    if type(model_lock) is not ModelLock or type(prepared) is not PreparedModelSet:
        raise TypeError("prepared model revalidation requires exact authority types")
    model_lock.validate()
    prepared.validate()
    if prepared.model_lock_sha256 != model_lock.sha256:
        raise ValueError("prepared models bind a different model lock")
    locked = {model.model_id: model.revision for model in model_lock.models}
    observed = {snapshot.model_id: snapshot.revision for snapshot in prepared.snapshots}
    if observed != locked:
        raise ValueError("prepared model revisions differ from the model lock")
    return {snapshot.model_id: snapshot.root for snapshot in prepared.snapshots}


__all__ = [
    "PREPARED_MODEL_BINDING_PROTOCOL_SHA256",
    "PreparedModelSet",
    "PreparedModelSnapshot",
    "bind_prepared_models",
    "revalidate_prepared_models",
]
