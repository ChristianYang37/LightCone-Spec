"""Immutable server execution policy for formal GPU evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ExecutionRole = Literal["target_reference", "speculative"]


@dataclass(frozen=True)
class ControlledExecutionPolicy:
    """Server controls shared by target and speculative endpoint roles."""

    schema_version: int = 2
    context_length: int = 40960
    random_seed: int = 1
    disable_radix_cache: bool = True
    disable_cuda_graph: bool = True
    target_reference_disable_overlap_schedule: bool = True
    speculative_disable_overlap_schedule: bool = False
    enable_deterministic_inference: bool = False
    incremental_streaming_output: bool = False

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("execution policy must use schema version 2")
        if type(self.context_length) is not int or self.context_length != 40960:
            raise ValueError("execution context must equal the locked model limit")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("execution random seed must be non-negative")
        boolean_fields = (
            self.disable_radix_cache,
            self.disable_cuda_graph,
            self.target_reference_disable_overlap_schedule,
            self.speculative_disable_overlap_schedule,
            self.enable_deterministic_inference,
            self.incremental_streaming_output,
        )
        if any(type(value) is not bool for value in boolean_fields):
            raise ValueError("execution switches must be booleans")
        if not self.disable_radix_cache or not self.disable_cuda_graph:
            raise ValueError(
                "formal DFlash exactness requires radix cache and CUDA graph disabled"
            )
        if not self.target_reference_disable_overlap_schedule:
            raise ValueError("target reference must disable overlap scheduling")
        if self.speculative_disable_overlap_schedule:
            raise ValueError("formal speculative runs must retain overlap scheduling")
        if self.enable_deterministic_inference:
            raise ValueError("unregistered deterministic execution variant")
        if self.incremental_streaming_output:
            raise ValueError("formal evidence requires complete output token IDs")

    @property
    def sha256(self) -> str:
        self.validate()
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()

    def overlap_disabled(self, *, role: ExecutionRole) -> bool:
        """Return the registered overlap switch for one endpoint role."""
        self.validate()
        if role == "target_reference":
            return self.target_reference_disable_overlap_schedule
        if role == "speculative":
            return self.speculative_disable_overlap_schedule
        raise ValueError(f"unknown execution-policy role: {role}")

    def server_info_fields(
        self, *, role: ExecutionRole
    ) -> dict[str, int | bool]:
        """Return the exact public ``/server_info`` values to attest."""
        self.validate()
        return {
            "context_length": self.context_length,
            "random_seed": self.random_seed,
            "disable_radix_cache": self.disable_radix_cache,
            "disable_cuda_graph": self.disable_cuda_graph,
            "disable_overlap_schedule": self.overlap_disabled(role=role),
            "enable_deterministic_inference": self.enable_deterministic_inference,
            "incremental_streaming_output": self.incremental_streaming_output,
        }

    def validate_server_info(
        self, server_info: dict, *, role: ExecutionRole
    ) -> None:
        """Fail closed unless a live server reports this exact policy."""
        if not isinstance(server_info, dict):
            raise TypeError("server_info must be an object")
        for name, expected in self.server_info_fields(role=role).items():
            value = server_info.get(name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"server execution policy mismatch: {name}")

    def write(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("execution policy is immutable")
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(self.sha256 + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ControlledExecutionPolicy:
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if set(value) != set(asdict(cls())):
            raise ValueError("execution policy fields do not match schema-v2")
        policy = cls(**value)
        policy.validate()
        sidecar = Path(f"{source}.sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != policy.sha256:
            raise ValueError("execution policy sidecar is missing or invalid")
        return policy
