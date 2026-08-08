"""Resource budget generation (spec 17.1).

GPUh = sum_c N_c * L_c * g_c / (3600 * v_c_baseline)

computed from measured baseline throughput, plus HBM, storage, replay
shards, log volume, a 30% retry margin and optional monetary cost.
Budgets must exist before any GPU run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetLine:
    condition: str
    num_requests: int  # N_c
    avg_output_tokens: float  # L_c
    gpus: int  # g_c
    baseline_tps: float  # v_c^{baseline}, measured

    @property
    def gpu_hours(self) -> float:
        if self.baseline_tps <= 0:
            raise ValueError(f"baseline TPS must be measured and positive: {self.condition}")
        return self.num_requests * self.avg_output_tokens * self.gpus / (
            3600.0 * self.baseline_tps
        )


@dataclass
class Budget:
    lines: list[BudgetLine] = field(default_factory=list)
    peak_hbm_gb: float = 0.0
    local_storage_gb: float = 0.0
    object_storage_gb: float = 0.0
    replay_shard_gb: float = 0.0
    log_volume_gb: float = 0.0
    retry_margin: float = 0.30
    cost_per_gpu_hour: float | None = None

    @property
    def gpu_hours_raw(self) -> float:
        return sum(line.gpu_hours for line in self.lines)

    @property
    def gpu_hours_with_margin(self) -> float:
        return self.gpu_hours_raw * (1.0 + self.retry_margin)

    @property
    def estimated_cost(self) -> float | None:
        if self.cost_per_gpu_hour is None:
            return None
        return self.gpu_hours_with_margin * self.cost_per_gpu_hour

    def to_dict(self) -> dict:
        return {
            "lines": [
                {
                    "condition": line.condition,
                    "num_requests": line.num_requests,
                    "avg_output_tokens": line.avg_output_tokens,
                    "gpus": line.gpus,
                    "baseline_tps": line.baseline_tps,
                    "gpu_hours": line.gpu_hours,
                }
                for line in self.lines
            ],
            "gpu_hours_raw": self.gpu_hours_raw,
            "retry_margin": self.retry_margin,
            "gpu_hours_with_margin": self.gpu_hours_with_margin,
            "peak_hbm_gb": self.peak_hbm_gb,
            "local_storage_gb": self.local_storage_gb,
            "object_storage_gb": self.object_storage_gb,
            "replay_shard_gb": self.replay_shard_gb,
            "log_volume_gb": self.log_volume_gb,
            "estimated_cost": self.estimated_cost,
        }
