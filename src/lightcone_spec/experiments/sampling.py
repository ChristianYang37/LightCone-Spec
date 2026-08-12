"""Immutable sampling profile shared by all formal methods."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SamplingProfile:
    schema_version: int = 2
    purpose: Literal["controlled", "natural"] = "controlled"
    # Formal controlled timing is greedy so every method follows the same
    # target-token trajectory. Stochastic exactness remains a separate test and
    # natural-task profile rather than a source of paired-timing path variance.
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = True

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("sampling profile must use schema version 2")
        if (
            not math.isfinite(self.temperature)
            or not math.isfinite(self.top_p)
            or self.temperature < 0
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError("sampling temperature/top_p are invalid")
        if self.purpose == "controlled" and not self.ignore_eos:
            raise ValueError("controlled long continuation requires ignore_eos=true")
        if self.purpose == "natural" and self.ignore_eos:
            raise ValueError("natural replication requires ignore_eos=false")

    @property
    def sha256(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()

    def parameters(self, *, seed: int, max_new_tokens: int) -> dict:
        self.validate()
        if seed < 0 or max_new_tokens < 1:
            raise ValueError(
                "sampling seed must be non-negative and generation limit positive"
            )
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            # The native SGLang ``/generate`` endpoint accepts
            # ``sampling_seed``.  ``seed`` belongs to its OpenAI-compatible
            # endpoint and is rejected by the native SamplingParams schema.
            "sampling_seed": seed,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": self.ignore_eos,
        }

    def write(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("sampling profile is immutable")
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(self.sha256 + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SamplingProfile:
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if set(value) != {
            "schema_version",
            "purpose",
            "temperature",
            "top_p",
            "ignore_eos",
        }:
            raise ValueError("sampling profile fields do not match schema-v2")
        profile = cls(**value)
        profile.validate()
        sidecar = Path(f"{source}.sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != profile.sha256:
            raise ValueError("sampling profile sidecar is missing or invalid")
        return profile
