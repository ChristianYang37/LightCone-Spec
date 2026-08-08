"""Hardware execution profiles (spec 2.4, 9.2).

- local_1x80gb: local smoke / P1 reference, one 80GB GPU; supports the
  Qwen3-4B and Qwen3-8B adapter paths.
- reference_8x80gb: single node 8x H100/H200 80GB for the full matrix.
- cpu_reference: toy/synthetic units only (this machine class).

Any GPU / CUDA / kernel / power-limit change requires recalibration of
baseline SPS, contention, energy and controller overhead before results
may be compared.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    gpus: int
    gpu_memory_gb: int
    supports_real_models: bool
    allowed_pairs: tuple[str, ...] = ()
    notes: str = ""


PROFILES = {
    "cpu_reference": HardwareProfile(
        name="cpu_reference",
        gpus=0,
        gpu_memory_gb=0,
        supports_real_models=False,
        notes="toy/synthetic units only; used for P0 and CPU test suites",
    ),
    "local_1x80gb": HardwareProfile(
        name="local_1x80gb",
        gpus=1,
        gpu_memory_gb=80,
        supports_real_models=True,
        allowed_pairs=("qwen3_4b_dspark7", "qwen3_8b_dspark7"),
        notes="local smoke / P1 reference; cache-safe tail updates only",
    ),
    "reference_8x80gb": HardwareProfile(
        name="reference_8x80gb",
        gpus=8,
        gpu_memory_gb=80,
        supports_real_models=True,
        allowed_pairs=(
            "qwen3_4b_dspark7",
            "qwen3_8b_dspark7",
            "qwen3_14b_dspark7",
            "gemma4_12b_dspark7",
        ),
        notes="single-node full-matrix reference",
    ),
}


def get_profile(name: str) -> HardwareProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown hardware profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]
