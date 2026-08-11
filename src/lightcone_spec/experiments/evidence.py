"""Content-bound GPU evidence attestation for the formal speed gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.data import DFLASH_SAFE_CONTEXT_LIMIT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_files_sha256(paths: Iterable[str | Path]) -> str:
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    entries = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"evidence file does not exist: {path}")
        entries.append((_sha256(path), path.stat().st_size))
    if not entries:
        raise ValueError("at least one evidence file is required")
    body = json.dumps(sorted(entries), separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class GpuEvidenceAttestation:
    schema_version: int
    status: str
    manifest_sha256: str
    selection_sha256: str
    model_lock_sha256: str
    performance_sha256: str
    patched_sglang_tree: str
    target_revision: str
    drafter_revision: str
    hardware_sha256: str
    methods: tuple[str, ...]
    repetitions: int
    context_start: int
    context_limit: int

    def validate(self) -> None:
        if self.schema_version != 2 or self.status != "MEASURED":
            raise ValueError("formal GPU evidence must be schema-v2 MEASURED")
        if self.methods != ("static", "tts", "l0"):
            raise ValueError("GPU evidence methods do not match the formal study")
        if self.repetitions != 8:
            raise ValueError("GPU evidence requires eight repetition blocks")
        if (
            self.context_start != 16384
            or self.context_limit != DFLASH_SAFE_CONTEXT_LIMIT
        ):
            raise ValueError("GPU evidence context region is outside the protocol")
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("GPU evidence uses a different patched SGLang tree")
        for name in (
            "manifest_sha256",
            "selection_sha256",
            "model_lock_sha256",
            "performance_sha256",
            "hardware_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in ("target_revision", "drafter_revision"):
            value = getattr(self, name)
            if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be an immutable Git revision")

    @property
    def sha256(self) -> str:
        body = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(body).hexdigest()

    def write(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("attestation is immutable; choose a new output path")
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(self.sha256 + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GpuEvidenceAttestation:
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        attestation = cls(
            **{**value, "methods": tuple(value.get("methods", ()))}
        )
        attestation.validate()
        sidecar = Path(f"{source}.sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != attestation.sha256:
            raise ValueError("GPU attestation sidecar is missing or invalid")
        return attestation

    def verify_performance(self, paths: Iterable[str | Path]) -> None:
        if evidence_files_sha256(paths) != self.performance_sha256:
            raise ValueError("GPU attestation does not bind these performance files")
