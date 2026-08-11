"""Content-bound GPU evidence attestation for the formal speed gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.data import DFLASH_SAFE_CONTEXT_LIMIT
from lightcone_spec.telemetry.records import OUTPUT_HASH_FORMAT


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


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


@dataclass(frozen=True)
class TargetOutput:
    prompt_id: str
    input_tokens: int
    output_tokens: int
    output_sha256: str

    def validate(self, *, context_limit: int) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("target-reference prompt IDs must be non-empty")
        if (
            isinstance(self.input_tokens, bool)
            or not isinstance(self.input_tokens, int)
            or isinstance(self.output_tokens, bool)
            or not isinstance(self.output_tokens, int)
            or self.input_tokens < 1
            or self.output_tokens < 1
        ):
            raise ValueError("target-reference token counts must be positive")
        if self.input_tokens + self.output_tokens != context_limit:
            raise ValueError("target-reference requests must reach the safe limit")
        if not _is_sha256(self.output_sha256):
            raise ValueError("target-reference outputs require a SHA-256")


@dataclass(frozen=True)
class GreedyTargetReference:
    """One target-only greedy trajectory shared by every formal method."""

    schema_version: int
    status: str
    model_lock_sha256: str
    target_model_id: str
    target_revision: str
    sampling_profile_sha256: str
    window_sha256: str
    runtime_config_sha256: str
    hardware_sha256: str
    patched_sglang_tree: str
    concurrency: int
    context_limit: int
    output_hash_format: str
    outputs: tuple[TargetOutput, ...]

    def validate(self) -> None:
        if self.schema_version != 2 or self.status != "MEASURED":
            raise ValueError("target reference must be schema-v2 MEASURED")
        if self.target_model_id != "Qwen/Qwen3-8B":
            raise ValueError("target reference uses the wrong target model")
        if not isinstance(self.target_revision, str) or len(self.target_revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.target_revision
        ):
            raise ValueError("target reference requires an immutable revision")
        for value in (
            self.model_lock_sha256,
            self.sampling_profile_sha256,
            self.window_sha256,
            self.runtime_config_sha256,
            self.hardware_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("target-reference identities must be SHA-256 values")
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("target reference uses another SGLang runtime tree")
        if self.output_hash_format != OUTPUT_HASH_FORMAT:
            raise ValueError("target reference uses an unknown output hash format")
        if (
            isinstance(self.concurrency, bool)
            or not isinstance(self.concurrency, int)
            or self.concurrency < 1
            or isinstance(self.context_limit, bool)
            or not isinstance(self.context_limit, int)
            or self.context_limit != DFLASH_SAFE_CONTEXT_LIMIT
        ):
            raise ValueError("target reference uses the wrong load or context limit")
        if len(self.outputs) != 32:
            raise ValueError("target reference requires 32 confirmation prompts")
        prompt_ids = [row.prompt_id for row in self.outputs]
        if prompt_ids != sorted(prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("target-reference outputs must have sorted unique prompts")
        for row in self.outputs:
            row.validate(context_limit=self.context_limit)

    def verify_study(
        self,
        *,
        model_lock_sha256: str,
        target_revision: str,
        sampling_profile_sha256: str,
        window_sha256: str,
        concurrency: int,
    ) -> None:
        """Require this trajectory to be the exact formal counterfactual."""
        self.validate()
        expected = {
            "model_lock_sha256": model_lock_sha256,
            "target_revision": target_revision,
            "sampling_profile_sha256": sampling_profile_sha256,
            "window_sha256": window_sha256,
            "concurrency": concurrency,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("target reference belongs to a different formal study")

    @property
    def sha256(self) -> str:
        self.validate()
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
            raise ValueError("target reference is immutable; choose a new output path")
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(self.sha256 + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GreedyTargetReference:
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "status",
            "model_lock_sha256",
            "target_model_id",
            "target_revision",
            "sampling_profile_sha256",
            "window_sha256",
            "runtime_config_sha256",
            "hardware_sha256",
            "patched_sglang_tree",
            "concurrency",
            "context_limit",
            "output_hash_format",
            "outputs",
        }
        if set(value) != expected or not isinstance(value["outputs"], list):
            raise ValueError("target-reference fields are malformed")
        output_fields = {
            "prompt_id",
            "input_tokens",
            "output_tokens",
            "output_sha256",
        }
        if any(
            not isinstance(row, dict) or set(row) != output_fields
            for row in value["outputs"]
        ):
            raise ValueError("target-reference output fields are malformed")
        artifact = cls(
            **{
                **value,
                "outputs": tuple(TargetOutput(**row) for row in value["outputs"]),
            }
        )
        artifact.validate()
        sidecar = Path(f"{source}.sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != artifact.sha256:
            raise ValueError("target-reference sidecar is missing or invalid")
        return artifact


@dataclass(frozen=True)
class GpuEvidenceAttestation:
    schema_version: int
    status: str
    manifest_sha256: str
    selection_sha256: str
    model_lock_sha256: str
    performance_sha256: str
    target_reference_sha256: str
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
        if self.methods != ("static", "tts", "naive_async"):
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
            "target_reference_sha256",
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

    def verify_target_reference(self, reference: GreedyTargetReference) -> None:
        if reference.sha256 != self.target_reference_sha256:
            raise ValueError("GPU attestation does not bind this target reference")
