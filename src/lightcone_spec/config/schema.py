"""Schema-v2 configuration for the focused Static/TTS/L0 runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lightcone_spec import PINNED_SGLANG_COMMIT

CoreMethod = Literal["static", "tts", "naive_async"]
BaselineMethod = Literal[
    "onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"
]
Method = CoreMethod | BaselineMethod
CORE_METHODS = ("static", "tts", "naive_async")
EXTERNAL_BASELINES = (
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
)
METHODS = CORE_METHODS + EXTERNAL_BASELINES


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class ModelPair(StrictModel):
    key: str = "qwen3_8b_dflash16"
    target: str = "Qwen/Qwen3-8B"
    drafter: str = "z-lab/Qwen3-8B-DFlash-b16"
    target_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    drafter_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    algorithm: Literal["DFLASH", "DSPARK", "EAGLE", "EAGLE3"] = "DFLASH"
    max_context_length: int = Field(default=40960, ge=1)
    draft_depth: int = Field(default=15, ge=1)


class OptimizerConfig(StrictModel):
    name: Literal["adam", "adamw", "sgd", "none"]
    learning_rate: float = Field(default=0.0, ge=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.999, gt=0.0, lt=1.0)
    epsilon: float = Field(default=1e-8, gt=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def validate_disabled(self) -> OptimizerConfig:
        if self.name == "none":
            if self.learning_rate != 0 or self.weight_decay != 0:
                raise ValueError(
                    "optimizer=none requires zero lr and weight decay"
                )
        elif self.learning_rate <= 0:
            raise ValueError(
                "an enabled optimizer requires a positive learning rate"
            )
        if self.name != "adamw" and self.weight_decay != 0:
            raise ValueError("weight_decay is only defined for adamw")
        return self


class AdaptationConfig(StrictModel):
    weight_update_mode: Literal["residual", "lora", "full"]
    parameter_scope: Literal["tail", "drafter"]
    kv_history_policy: Literal["frozen"] = "frozen"
    adaptation_scope: Literal["cohort"] = "cohort"
    adaptation_group_id: str = Field(min_length=1, max_length=128)
    optimizer: OptimizerConfig
    rank: int | None = Field(default=None, ge=1)
    stride: int = Field(default=10, ge=1)
    max_in_flight: Literal[1] = 1
    canvas_tokens: int = Field(default=16, ge=2)
    loss_position_decay: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_mode(self) -> AdaptationConfig:
        if (
            self.weight_update_mode == "residual"
            and self.parameter_scope != "tail"
        ):
            raise ValueError(
                "residual updates are only valid for parameter_scope=tail"
            )
        if self.weight_update_mode == "full" and self.rank is not None:
            raise ValueError("full updates require rank=null")
        if (
            self.weight_update_mode in {"lora", "residual"}
            and self.rank is None
        ):
            raise ValueError(
                f"{self.weight_update_mode} updates require an explicit rank"
            )
        if self.optimizer.name == "none":
            raise ValueError("adaptation requires an enabled optimizer")
        return self


class RuntimeConfig(StrictModel):
    sglang_commit: Literal[PINNED_SGLANG_COMMIT] = PINNED_SGLANG_COMMIT
    sampling_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tensor_parallel_size: int = Field(default=1, ge=1)
    speculative_num_draft_tokens: int = Field(default=16, ge=2)
    max_running_requests: int = Field(default=1, ge=1)
    telemetry_detail: Literal["headline", "profile"] = "headline"


class RunConfig(StrictModel):
    schema_version: Literal[2] = 2
    method: Method
    model: ModelPair
    runtime: RuntimeConfig
    adaptation: AdaptationConfig | None = None
    tenant_id: str = Field(
        default="research", min_length=1, max_length=128
    )

    @model_validator(mode="after")
    def validate_method_contract(self) -> RunConfig:
        if self.method == "static":
            if self.adaptation is not None:
                raise ValueError("static must not allocate adaptation state")
            return self
        if self.adaptation is None:
            raise ValueError(
                f"method={self.method} requires adaptation configuration"
            )
        if self.method.startswith("onlinespec_"):
            if self.adaptation.optimizer.name != "sgd":
                raise ValueError(
                    "OnlineSpec baselines preserve their SGD update"
                )
            if self.adaptation.parameter_scope != "tail":
                raise ValueError(
                    "OnlineSpec baselines are isolated tail baselines"
                )
            return self
        if self.model.algorithm != "DFLASH":
            raise ValueError(
                "schema-v2 TTS/L0 adaptation is certified only for DFlash"
            )
        if self.runtime.tensor_parallel_size != 1:
            raise ValueError(
                "schema-v2 TTS/L0 adaptation currently requires TP=1"
            )
        if self.adaptation.optimizer.name not in {"adam", "adamw"}:
            raise ValueError("TTS/L0 speed tuning supports Adam or AdamW")
        if (
            self.runtime.speculative_num_draft_tokens
            != self.model.draft_depth + 1
        ):
            raise ValueError(
                "DFlash runtime block size must equal draft_depth + 1"
            )
        if (
            self.adaptation.canvas_tokens
            != self.runtime.speculative_num_draft_tokens
        ):
            raise ValueError(
                "current-canvas width must equal the DFlash block size"
            )
        return self
