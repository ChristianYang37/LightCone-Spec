"""Resolve and verify Hugging Face revisions without embedding credentials."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LockedModel:
    model_id: str
    revision: str


@dataclass(frozen=True)
class ModelLock:
    schema_version: int
    models: tuple[LockedModel, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "models": [asdict(model) for model in self.models],
        }

    def validate(self) -> None:
        if self.schema_version != 2 or not self.models:
            raise ValueError("model lock must be non-empty schema-v2")
        identifiers = [model.model_id for model in self.models]
        if any(not value for value in identifiers) or len(identifiers) != len(
            set(identifiers)
        ):
            raise ValueError("locked model IDs must be non-empty and unique")
        for model in self.models:
            if len(model.revision) != 40 or any(
                character not in "0123456789abcdef"
                for character in model.revision
            ):
                raise ValueError("model revisions must be immutable Git SHAs")

    @property
    def sha256(self) -> str:
        body = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(body).hexdigest()

    def write(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("model lock is immutable; choose a new output path")
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(
            self.sha256 + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> ModelLock:
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if set(data) != {"schema_version", "models"}:
            raise ValueError("model lock fields do not match schema-v2")
        if data.get("schema_version") != 2 or not isinstance(
            data.get("models"), list
        ):
            raise ValueError("model lock must use schema version 2")
        if any(
            not isinstance(model, dict)
            or set(model) != {"model_id", "revision"}
            for model in data["models"]
        ):
            raise ValueError("locked model records are malformed")
        lock = cls(
            schema_version=2,
            models=tuple(
                LockedModel(**model) for model in data["models"]
            ),
        )
        lock.validate()
        sidecar = Path(f"{source}.sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != lock.sha256:
            raise ValueError("model-lock sidecar is missing or invalid")
        return lock


def resolve_model_lock(model_ids: tuple[str, ...], token: str | None = None) -> ModelLock:
    from huggingface_hub import HfApi

    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("model IDs must be non-empty and unique")
    api = HfApi(token=token)
    models = tuple(
        LockedModel(
            model_id=model_id,
            revision=str(api.model_info(model_id).sha),
        )
        for model_id in model_ids
    )
    lock = ModelLock(schema_version=2, models=models)
    lock.validate()
    return lock


def prepare_models(
    lock: ModelLock,
    cache_dir: str | Path,
    *,
    token: str | None = None,
    local_files_only: bool = False,
) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    lock.validate()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    roots: dict[str, str] = {}
    for model in lock.models:
        root = snapshot_download(
            repo_id=model.model_id,
            revision=model.revision,
            cache_dir=cache,
            token=token,
            local_files_only=local_files_only,
        )
        roots[model.model_id] = str(Path(root).resolve())
    return roots
