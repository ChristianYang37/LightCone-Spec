"""Single resumable protocol manifest for the focused speed study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lightcone_spec.experiments.data import LongContinuationAdapter
from lightcone_spec.experiments.protocol import TUNING_STAGES, tuning_candidates
from lightcone_spec.experiments.sampling import SamplingProfile


def _tuning_grid_sha256() -> str:
    rows = [asdict(candidate) for candidate in tuning_candidates()]
    body = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class SpeedStudyManifest:
    schema_version: int = 2
    name: str = "static-tts-l0-speed-study"
    model_pair: str = "qwen3_8b_dflash16"
    methods: tuple[str, ...] = ("static", "tts", "naive_async")
    phases: tuple[str, ...] = (
        "static_load_screen",
        "shared_config_tuning",
        "controlled_confirmation",
        "natural_task_replication",
        "independent_profiler",
    )
    formal_context_start: int = 16384
    safe_context_limit: int = 40960
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
        40960,
    )
    confirmation_repetitions: int = 8
    confirmation_schedule_seed: int = 20260809
    controlled_window_hashes: dict[str, str] = field(default_factory=dict)
    tuning_grid_sha256: str = ""
    sampling_profile_sha256: str = ""
    natural_side_tables: tuple[str, ...] = ("livecodebench", "math500")
    gpu_evidence: str = "UNMEASURED"

    @classmethod
    def default(cls) -> SpeedStudyManifest:
        adapter = LongContinuationAdapter()
        adapter.assert_disjoint()
        return cls(
            controlled_window_hashes={
                name: adapter.window_sha256(name)
                for name in ("load", "tune", "confirm")
            },
            tuning_grid_sha256=_tuning_grid_sha256(),
            sampling_profile_sha256=SamplingProfile().sha256,
        )

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only schema-v2 speed-study manifests are valid")
        if self.methods != ("static", "tts", "naive_async"):
            raise ValueError("formal methods must remain Static, TTS, and L0")
        if self.phases != (
            "static_load_screen",
            "shared_config_tuning",
            "controlled_confirmation",
            "natural_task_replication",
            "independent_profiler",
        ):
            raise ValueError("speed-study phase order is immutable")
        if self.formal_context_start != 16384:
            raise ValueError("formal speed region must begin at 16K")
        if self.gpu_evidence != "UNMEASURED":
            raise ValueError("source manifests cannot contain GPU claims")
        if self.safe_context_limit != 40960:
            raise ValueError("the pinned DFlash study stops at 40,960 tokens")
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
            40960,
        ):
            raise ValueError("generated-token bucket identity mismatch")
        if self.confirmation_repetitions != 8:
            raise ValueError("formal confirmation requires eight independent blocks")
        if self.confirmation_schedule_seed != 20260809:
            raise ValueError("confirmation schedule identity mismatch")
        if self.natural_side_tables != ("livecodebench", "math500"):
            raise ValueError("natural side-table identity mismatch")
        expected_windows = LongContinuationAdapter.default_hashes()
        if self.controlled_window_hashes != expected_windows:
            raise ValueError("controlled dataset window identity mismatch")
        if self.tuning_grid_sha256 != _tuning_grid_sha256():
            raise ValueError("tuning grid identity mismatch")
        if self.sampling_profile_sha256 != SamplingProfile().sha256:
            raise ValueError("sampling profile identity mismatch")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        body = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(body).hexdigest()

    def write(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        body = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("manifest is immutable; choose a new output path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(
            self.sha256 + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> SpeedStudyManifest:
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        for field_name in (
            "methods",
            "phases",
            "concurrency_grid",
            "generated_buckets",
            "natural_side_tables",
        ):
            if field_name in data:
                data[field_name] = tuple(data[field_name])
        if "tuning_stages" in data:
            data["tuning_stages"] = tuple(
                tuple(stage) for stage in data["tuning_stages"]
            )
        manifest = cls(**data)
        manifest.validate()
        sidecar = Path(f"{source}.sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != manifest.sha256:
            raise ValueError("manifest sidecar is missing or does not match")
        return manifest
