"""Historical diagnostic evidence types outside the industrial gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NoReturn

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.execution import ControlledExecutionPolicy
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
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _strict_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{label} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error

    def reject_nonfinite(item: object) -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError(f"{label} contains a non-finite JSON number")
        if type(item) is dict:
            for nested in item.values():
                reject_nonfinite(nested)
        elif type(item) is list:
            for nested in item:
                reject_nonfinite(nested)

    reject_nonfinite(value)
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    return value


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
    """One content-bound target-only trajectory for diagnostic comparison.

    A structurally valid capture remains explicitly
    ``PRELIMINARY_DIAGNOSTIC_ONLY``. Its hashes can bind legacy comparisons,
    but it can never enter the industrial evidence path.
    """

    schema_version: int
    status: str
    model_lock_sha256: str
    target_model_id: str
    target_revision: str
    sampling_profile_sha256: str
    execution_policy_sha256: str
    window_sha256: str
    runtime_config_sha256: str
    hardware_sha256: str
    patched_sglang_tree: str
    concurrency: int
    context_limit: int
    output_hash_format: str
    outputs: tuple[TargetOutput, ...]
    _historical_source_sha256: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 2
            or self.status != "PRELIMINARY_DIAGNOSTIC_ONLY"
        ):
            raise ValueError(
                "target reference must be schema-v2 PRELIMINARY_DIAGNOSTIC_ONLY"
            )
        if self.target_model_id != "Qwen/Qwen3-8B":
            raise ValueError("target reference uses the wrong target model")
        if (
            not isinstance(self.target_revision, str)
            or len(self.target_revision) != 40
            or any(char not in "0123456789abcdef" for char in self.target_revision)
        ):
            raise ValueError("target reference requires an immutable revision")
        for value in (
            self.model_lock_sha256,
            self.sampling_profile_sha256,
            self.execution_policy_sha256,
            self.window_sha256,
            self.runtime_config_sha256,
            self.hardware_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("target-reference identities must be SHA-256 values")
        if self.execution_policy_sha256 != ControlledExecutionPolicy().sha256:
            raise ValueError("target reference uses an unregistered execution policy")
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
        execution_policy_sha256: str,
        window_sha256: str,
        concurrency: int,
    ) -> None:
        """Require this trajectory to be the exact diagnostic counterfactual."""
        self.validate()
        expected = {
            "model_lock_sha256": model_lock_sha256,
            "target_revision": target_revision,
            "sampling_profile_sha256": sampling_profile_sha256,
            "execution_policy_sha256": execution_policy_sha256,
            "window_sha256": window_sha256,
            "concurrency": concurrency,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError(
                "target reference belongs to a different preliminary diagnostic study"
            )

    @property
    def sha256(self) -> str:
        self.validate()
        if self._historical_source_sha256 is not None:
            return self._historical_source_sha256
        value = asdict(self)
        value.pop("_historical_source_sha256")
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()

    def write(self, path: str | Path) -> None:
        self.validate()
        if self._historical_source_sha256 is not None:
            raise ValueError(
                "historical target references are read-only preliminary evidence"
            )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        value = asdict(self)
        value.pop("_historical_source_sha256")
        body = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("target reference is immutable; choose a new output path")
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(self.sha256 + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GreedyTargetReference:
        source = Path(path)
        value = _strict_json_object(source, label="target reference")
        source_sha256 = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        expected = {
            "schema_version",
            "status",
            "model_lock_sha256",
            "target_model_id",
            "target_revision",
            "sampling_profile_sha256",
            "execution_policy_sha256",
            "window_sha256",
            "runtime_config_sha256",
            "hardware_sha256",
            "patched_sglang_tree",
            "concurrency",
            "context_limit",
            "output_hash_format",
            "outputs",
        }
        if set(value) != expected or type(value["outputs"]) is not list:
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
        sidecar = Path(f"{source}.sha256")
        if (
            sidecar.is_symlink()
            or not sidecar.is_file()
            or sidecar.read_bytes() != f"{source_sha256}\n".encode("ascii")
        ):
            raise ValueError("target-reference sidecar is missing or invalid")
        historical = (
            type(value.get("schema_version")) is int
            and value.get("schema_version") == 2
            and value.get("status") == "UNMEASURED"
        )
        artifact = cls(
            **{
                **value,
                "status": (
                    "PRELIMINARY_DIAGNOSTIC_ONLY" if historical else value.get("status")
                ),
                "outputs": tuple(TargetOutput(**row) for row in value["outputs"]),
                "_historical_source_sha256": (source_sha256 if historical else None),
            }
        )
        artifact.validate()
        if not historical and artifact.sha256 != source_sha256:
            raise ValueError("target-reference sidecar is missing or invalid")
        return artifact


@dataclass(frozen=True)
class GpuEvidenceAttestation:
    """Disabled legacy attestation schema retained only for clear rejection."""

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
        raise RuntimeError(
            "legacy_gpu_attestation_api_disabled: formal evidence requires the "
            "industrial executor and native terminal authority"
        )

    @property
    def sha256(self) -> str:
        self.validate()
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
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
        del path
        raise RuntimeError(
            "legacy_gpu_attestation_api_disabled: formal evidence requires the "
            "industrial executor and native terminal authority"
        )

    def verify_performance(self, paths: Iterable[str | Path]) -> None:
        self.validate()
        if evidence_files_sha256(paths) != self.performance_sha256:
            raise ValueError("GPU attestation does not bind these performance files")

    def verify_target_reference(self, reference: GreedyTargetReference) -> None:
        self.validate()
        if reference.sha256 != self.target_reference_sha256:
            raise ValueError("GPU attestation does not bind this target reference")
