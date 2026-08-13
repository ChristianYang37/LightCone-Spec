"""Preliminary-only manifest for the historical focused speed study.

This schema predates the industrial registry, budget, activation, completion,
and execution-bundle authorities.  It remains reproducible as a diagnostic
workflow, but it is categorically outside the formal industrial surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NoReturn

from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.data import (
    DFLASH_SAFE_CONTEXT_LIMIT,
    LongContinuationAdapter,
)
from lightcone_spec.experiments.protocol import TUNING_STAGES, tuning_candidates
from lightcone_spec.experiments.sampling import SamplingProfile

PRELIMINARY_DIAGNOSTIC_ONLY = "PRELIMINARY_DIAGNOSTIC_ONLY"
PRELIMINARY_SPEED_STUDY_MANIFEST_KIND = "preliminary_diagnostic_speed_study_manifest"


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


def _tuning_grid_sha256() -> str:
    rows = [asdict(candidate) for candidate in tuning_candidates()]
    body = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _historical_payload(current: dict[str, object]) -> dict[str, object]:
    value = current.copy()
    for field_name in (
        "kind",
        "evidence_scope",
        "formal_execution_authorized",
        "industrial_authority_consumption",
    ):
        value.pop(field_name)
    value["name"] = "static-tts-l0-speed-study"
    value["formal_context_start"] = value.pop("diagnostic_context_start")
    value["gpu_evidence"] = "UNMEASURED"
    return value


@dataclass(frozen=True)
class PreliminarySpeedStudyManifest:
    schema_version: int = 2
    kind: str = PRELIMINARY_SPEED_STUDY_MANIFEST_KIND
    name: str = "preliminary-diagnostic-static-tts-l0-speed-study"
    evidence_scope: str = PRELIMINARY_DIAGNOSTIC_ONLY
    formal_execution_authorized: bool = False
    industrial_authority_consumption: str = "FORBIDDEN"
    model_pair: str = "qwen3_8b_dflash16"
    methods: tuple[str, ...] = ("static", "tts", "l0")
    phases: tuple[str, ...] = (
        "static_load_screen",
        "shared_config_tuning",
        "controlled_confirmation",
        "natural_task_replication",
        "independent_profiler",
    )
    diagnostic_context_start: int = 16384
    safe_context_limit: int = DFLASH_SAFE_CONTEXT_LIMIT
    context_limit_definition: str = "prompt_tokens_plus_generated_tokens"
    concurrency_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 48)
    load_screen_context_limit: int = 4096
    tuning_stages: tuple[tuple[int, int], ...] = TUNING_STAGES
    generated_buckets: tuple[int, ...] = (
        0,
        2048,
        4096,
        8192,
        16384,
        24576,
        32768,
        DFLASH_SAFE_CONTEXT_LIMIT,
    )
    confirmation_repetitions: int = 8
    confirmation_schedule_seed: int = 20260809
    request_scheduling: str = "ordered_native_batch_cohort_queue"
    headline_timing_unit: str = "method_repetition_batch"
    inference_cluster_unit: str = "repetition_block"
    controlled_window_hashes: dict[str, str] = field(default_factory=dict)
    tuning_grid_sha256: str = ""
    sampling_profile_sha256: str = ""
    execution_policy_sha256: str = ""
    natural_side_tables: tuple[str, ...] = ("livecodebench", "math500")
    gpu_evidence: str = PRELIMINARY_DIAGNOSTIC_ONLY
    _historical_source_sha256: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def default(cls) -> PreliminarySpeedStudyManifest:
        adapter = LongContinuationAdapter()
        adapter.assert_disjoint()
        return cls(
            controlled_window_hashes={
                name: adapter.window_sha256(name)
                for name in ("load", "tune", "confirm")
            },
            tuning_grid_sha256=_tuning_grid_sha256(),
            sampling_profile_sha256=SamplingProfile().sha256,
            execution_policy_sha256=ControlledExecutionPolicy().sha256,
        )

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError(
                "only schema-v2 preliminary speed-study manifests are valid"
            )
        if self.kind != PRELIMINARY_SPEED_STUDY_MANIFEST_KIND:
            raise ValueError("preliminary speed-study manifest kind mismatch")
        if self.name != "preliminary-diagnostic-static-tts-l0-speed-study":
            raise ValueError("preliminary speed-study name mismatch")
        if self.evidence_scope != PRELIMINARY_DIAGNOSTIC_ONLY:
            raise ValueError("legacy speed-study evidence must remain preliminary")
        if self.formal_execution_authorized is not False:
            raise ValueError("legacy speed-study cannot authorize formal execution")
        if self.industrial_authority_consumption != "FORBIDDEN":
            raise ValueError("legacy speed-study cannot consume industrial authorities")
        if self.model_pair != "qwen3_8b_dflash16":
            raise ValueError("preliminary speed-study model pair mismatch")
        if self.methods != ("static", "tts", "l0"):
            raise ValueError(
                "preliminary diagnostic methods must remain Static, TTS, and L0"
            )
        if self.phases != (
            "static_load_screen",
            "shared_config_tuning",
            "controlled_confirmation",
            "natural_task_replication",
            "independent_profiler",
        ):
            raise ValueError("speed-study phase order is immutable")
        exact_integer_fields = {
            "diagnostic_context_start": self.diagnostic_context_start,
            "safe_context_limit": self.safe_context_limit,
            "load_screen_context_limit": self.load_screen_context_limit,
            "confirmation_repetitions": self.confirmation_repetitions,
            "confirmation_schedule_seed": self.confirmation_schedule_seed,
        }
        if any(type(value) is not int for value in exact_integer_fields.values()):
            raise TypeError(
                "preliminary speed-study scalar identities must be exact integers"
            )
        for name, values in (
            ("concurrency_grid", self.concurrency_grid),
            ("generated_buckets", self.generated_buckets),
        ):
            if type(values) is not tuple or any(
                type(value) is not int for value in values
            ):
                raise TypeError(f"{name} must contain only exact integers")
        if type(self.tuning_stages) is not tuple or any(
            type(stage) is not tuple
            or len(stage) != 2
            or any(type(value) is not int for value in stage)
            for stage in self.tuning_stages
        ):
            raise TypeError("tuning_stages must contain exact integer pairs")
        if self.diagnostic_context_start != 16384:
            raise ValueError("preliminary diagnostic speed region must begin at 16K")
        if self.gpu_evidence != PRELIMINARY_DIAGNOSTIC_ONLY:
            raise ValueError("legacy manifests cannot contain formal GPU claims")
        if self.safe_context_limit != DFLASH_SAFE_CONTEXT_LIMIT:
            raise ValueError(
                "the pinned DFlash study stops at the registered safe limit"
            )
        if self.context_limit_definition != "prompt_tokens_plus_generated_tokens":
            raise ValueError("context limit must include the tokenized prompt")
        if self.concurrency_grid != (1, 2, 4, 8, 16, 32, 48):
            raise ValueError("Static load grid identity mismatch")
        if self.load_screen_context_limit != 4096:
            raise ValueError("Static load screen context identity mismatch")
        if self.tuning_stages != TUNING_STAGES:
            raise ValueError("successive-halving stage identity mismatch")
        if self.generated_buckets != (
            0,
            2048,
            4096,
            8192,
            16384,
            24576,
            32768,
            DFLASH_SAFE_CONTEXT_LIMIT,
        ):
            raise ValueError("generated-token bucket identity mismatch")
        if self.confirmation_repetitions != 8:
            raise ValueError(
                "preliminary confirmation requires eight independent blocks"
            )
        if self.confirmation_schedule_seed != 20260809:
            raise ValueError("confirmation schedule identity mismatch")
        if self.request_scheduling != "ordered_native_batch_cohort_queue":
            raise ValueError("request scheduling identity mismatch")
        if self.headline_timing_unit != "method_repetition_batch":
            raise ValueError("headline timing identity mismatch")
        if self.inference_cluster_unit != "repetition_block":
            raise ValueError("inference cluster identity mismatch")
        if self.natural_side_tables != ("livecodebench", "math500"):
            raise ValueError("natural side-table identity mismatch")
        expected_windows = LongContinuationAdapter.default_hashes()
        if self.controlled_window_hashes != expected_windows:
            raise ValueError("controlled dataset window identity mismatch")
        if self.tuning_grid_sha256 != _tuning_grid_sha256():
            raise ValueError("tuning grid identity mismatch")
        if self.sampling_profile_sha256 != SamplingProfile().sha256:
            raise ValueError("sampling profile identity mismatch")
        if self.execution_policy_sha256 != ControlledExecutionPolicy().sha256:
            raise ValueError("execution policy identity mismatch")
        if self._historical_source_sha256 is not None:
            expected_historical_sha256 = hashlib.sha256(
                json.dumps(
                    _historical_payload(self.to_dict()),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if self._historical_source_sha256 != expected_historical_sha256:
                raise ValueError("historical manifest source identity mismatch")

    def to_dict(self) -> dict:
        value = asdict(self)
        value.pop("_historical_source_sha256")
        return value

    @property
    def sha256(self) -> str:
        self.validate()
        if self._historical_source_sha256 is not None:
            return self._historical_source_sha256
        body = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(body).hexdigest()

    def write(self, path: str | Path) -> None:
        self.validate()
        if self._historical_source_sha256 is not None:
            raise ValueError(
                "historical unscoped manifests are read-only; build a new "
                "preliminary manifest"
            )
        output = Path(path)
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("manifest is immutable; choose a new output path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(self.sha256 + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PreliminarySpeedStudyManifest:
        source = Path(path)
        data = _strict_json_object(source, label="preliminary speed-study manifest")
        source_sha256 = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        sidecar = Path(f"{source}.sha256")
        if (
            sidecar.is_symlink()
            or not sidecar.is_file()
            or sidecar.read_bytes() != f"{source_sha256}\n".encode("ascii")
        ):
            raise ValueError("manifest sidecar is missing or does not match")
        current_fields = set(cls.default().to_dict())
        historical_fields = set(_historical_payload(cls.default().to_dict()))
        if set(data) == current_fields:
            historical = False
        elif set(data) == historical_fields:
            historical = True
        else:
            raise ValueError(
                "preliminary speed-study manifest fields do not match schema"
            )
        if historical and not (
            type(data.get("schema_version")) is int
            and data.get("schema_version") == 2
            and data.get("name") == "static-tts-l0-speed-study"
            and data.get("gpu_evidence") == "UNMEASURED"
            and "kind" not in data
            and "evidence_scope" not in data
            and "formal_execution_authorized" not in data
            and "industrial_authority_consumption" not in data
            and "formal_context_start" in data
            and "diagnostic_context_start" not in data
        ):
            raise ValueError("historical speed-study manifest identity mismatch")
        sequence_fields = (
            "methods",
            "phases",
            "concurrency_grid",
            "generated_buckets",
            "natural_side_tables",
            "tuning_stages",
        )
        if any(type(data[field_name]) is not list for field_name in sequence_fields):
            raise TypeError("manifest sequence fields must be JSON arrays")
        if type(data["controlled_window_hashes"]) is not dict:
            raise TypeError("controlled window hashes must be a JSON object")
        if any(
            type(stage) is not list or len(stage) != 2
            for stage in data["tuning_stages"]
        ):
            raise TypeError("tuning stages must be two-element JSON arrays")
        if historical:
            data["kind"] = PRELIMINARY_SPEED_STUDY_MANIFEST_KIND
            data["name"] = "preliminary-diagnostic-static-tts-l0-speed-study"
            data["evidence_scope"] = PRELIMINARY_DIAGNOSTIC_ONLY
            data["formal_execution_authorized"] = False
            data["industrial_authority_consumption"] = "FORBIDDEN"
            data["diagnostic_context_start"] = data.pop("formal_context_start")
            data["gpu_evidence"] = PRELIMINARY_DIAGNOSTIC_ONLY
        for field_name in sequence_fields[:-1]:
            data[field_name] = tuple(data[field_name])
        data["tuning_stages"] = tuple(tuple(stage) for stage in data["tuning_stages"])
        manifest = cls(**data)
        if historical:
            object.__setattr__(
                manifest,
                "_historical_source_sha256",
                source_sha256,
            )
        manifest.validate()
        if not historical and source_sha256 != manifest.sha256:
            raise ValueError("manifest sidecar is missing or does not match")
        return manifest
