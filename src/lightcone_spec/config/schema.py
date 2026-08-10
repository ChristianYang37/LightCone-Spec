"""Schema-v2 configuration for the focused Static/TTS/L0 runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lightcone_spec import PINNED_SGLANG_COMMIT

CoreMethod = Literal["static", "tts", "naive_async"]
BaselineMethod = Literal["onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"]
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
    name: Literal[
        "adam",
        "adamw",
        "sgd",
        "sgdm",
        "nag",
        "muon",
        "lion",
        "none",
    ]
    learning_rate: float = Field(default=0.0, ge=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.999, gt=0.0, lt=1.0)
    epsilon: float = Field(default=1e-8, gt=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    momentum: float | None = Field(default=None, gt=0.0, lt=1.0)
    muon_ns_steps: int | None = Field(default=None, ge=1, le=20)
    muon_auxiliary_learning_rate: float | None = Field(default=None, gt=0.0)
    muon_auxiliary_weight_decay: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_disabled(self) -> OptimizerConfig:
        if self.name == "none":
            if self.learning_rate != 0 or self.weight_decay != 0:
                raise ValueError("optimizer=none requires zero lr and weight decay")
        elif self.learning_rate <= 0:
            raise ValueError("an enabled optimizer requires a positive learning rate")
        decay_optimizers = {"adamw", "sgdm", "nag", "muon", "lion"}
        if self.name not in decay_optimizers and self.weight_decay != 0:
            raise ValueError("weight_decay is unsupported for this optimizer")
        momentum_optimizers = {"sgdm", "nag", "muon"}
        if self.name in momentum_optimizers and self.momentum is None:
            raise ValueError(f"{self.name} requires explicit momentum")
        if self.name not in momentum_optimizers and self.momentum is not None:
            raise ValueError(f"momentum is not a parameter of optimizer={self.name}")
        if self.name in {"sgd", "sgdm", "nag", "none"} and (
            self.beta1 != 0.9 or self.beta2 != 0.999 or self.epsilon != 1e-8
        ):
            raise ValueError(
                f"unused Adam fields must stay canonical for optimizer={self.name}"
            )
        if self.name == "lion" and self.epsilon != 1e-8:
            raise ValueError("unused epsilon must stay canonical for optimizer=lion")
        if self.name == "muon" and self.muon_ns_steps is None:
            raise ValueError("muon requires explicit muon_ns_steps")
        if self.name != "muon" and self.muon_ns_steps is not None:
            raise ValueError("muon_ns_steps is only defined for optimizer=muon")
        auxiliary = (
            self.muon_auxiliary_learning_rate,
            self.muon_auxiliary_weight_decay,
        )
        if self.name == "muon" and any(value is None for value in auxiliary):
            raise ValueError(
                "muon requires explicit auxiliary AdamW lr and weight decay"
            )
        if self.name != "muon" and any(value is not None for value in auxiliary):
            raise ValueError("Muon auxiliary AdamW fields require optimizer=muon")
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
        if self.weight_update_mode == "residual" and self.parameter_scope != "tail":
            raise ValueError("residual updates are only valid for parameter_scope=tail")
        if self.weight_update_mode == "full" and self.rank is not None:
            raise ValueError("full updates require rank=null")
        if self.weight_update_mode in {"lora", "residual"} and self.rank is None:
            raise ValueError(
                f"{self.weight_update_mode} updates require an explicit rank"
            )
        if self.optimizer.name == "none":
            raise ValueError("adaptation requires an enabled optimizer")
        return self


class OnlineSpecConfig(StrictModel):
    """Algorithm state not shared with the Static/TTS/L0 hypothesis."""

    projection_radius: float | None = Field(default=None, gt=0.0)
    additional_learning_rates: tuple[float, ...] = ()
    hedge_learning_rate: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_finite_grid(self) -> OnlineSpecConfig:
        rates = self.additional_learning_rates
        if any(rate <= 0 for rate in rates):
            raise ValueError("OnlineSPEC learning rates must be positive")
        if len(set(rates)) != len(rates) or tuple(sorted(rates)) != rates:
            raise ValueError(
                "OnlineSPEC additional learning rates must be unique and increasing"
            )
        return self


class RuntimeConfig(StrictModel):
    sglang_commit: Literal[PINNED_SGLANG_COMMIT] = PINNED_SGLANG_COMMIT
    sampling_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tensor_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    speculative_num_draft_tokens: int = Field(default=16, ge=2)
    speculative_eagle_topk: int | None = Field(default=None, ge=1)
    use_rejection_sampling: bool = True
    max_running_requests: int = Field(default=1, ge=1)
    telemetry_detail: Literal["headline", "profile"] = "headline"


class RunConfig(StrictModel):
    schema_version: Literal[2] = 2
    method: Method
    model: ModelPair
    runtime: RuntimeConfig
    adaptation: AdaptationConfig | None = None
    online_spec: OnlineSpecConfig | None = None
    tenant_id: str = Field(default="research", min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_method_contract(self) -> RunConfig:
        algorithm = self.model.algorithm
        if self.runtime.speculative_num_draft_tokens != self.model.draft_depth + 1:
            raise ValueError(f"{algorithm} verify width must equal draft_depth + 1")
        if algorithm in {"DFLASH", "DSPARK"}:
            if self.runtime.speculative_eagle_topk is not None:
                raise ValueError(
                    "speculative_eagle_topk is only valid for EAGLE backends"
                )
        elif self.runtime.speculative_eagle_topk is None:
            raise ValueError("EAGLE backends require speculative_eagle_topk")
        if self.method == "static":
            if self.adaptation is not None or self.online_spec is not None:
                raise ValueError("static must not allocate adaptation state")
            return self
        if self.adaptation is None:
            raise ValueError(f"method={self.method} requires adaptation configuration")
        if self.method.startswith("onlinespec_"):
            if self.online_spec is None:
                raise ValueError("OnlineSPEC baselines require online_spec state")
            if self.adaptation.optimizer.name != "sgd":
                raise ValueError("OnlineSpec baselines preserve their SGD update")
            if algorithm != "DFLASH" and self.adaptation.parameter_scope != "tail":
                raise ValueError(
                    f"{algorithm} OnlineSPEC currently supports parameter_scope=tail"
                )
            if self.method == "onlinespec_ens":
                rates = (
                    self.adaptation.optimizer.learning_rate,
                    *self.online_spec.additional_learning_rates,
                )
                if len(rates) < 2 or tuple(sorted(rates)) != rates:
                    raise ValueError(
                        "OnlineSPEC Hedge requires an increasing multi-rate grid"
                    )
                if self.online_spec.hedge_learning_rate is None:
                    raise ValueError("OnlineSPEC Hedge requires hedge_learning_rate")
            elif (
                self.online_spec.additional_learning_rates
                or self.online_spec.hedge_learning_rate is not None
            ):
                raise ValueError("ensemble fields are only valid for onlinespec_ens")
            if (
                self.runtime.tensor_parallel_size != 1
                or self.runtime.data_parallel_size != 1
            ):
                raise ValueError("OnlineSPEC currently requires TP=DP=1")
            if not self.runtime.use_rejection_sampling:
                raise ValueError("OnlineSPEC requires exact rejection sampling")
            if (
                algorithm in {"EAGLE", "EAGLE3"}
                and self.runtime.speculative_eagle_topk != 1
            ):
                raise ValueError("adapted EAGLE/EAGLE3 currently requires topk=1")
            if (
                self.adaptation.canvas_tokens
                != self.runtime.speculative_num_draft_tokens
            ):
                raise ValueError(
                    "adaptation canvas width must equal the speculative width"
                )
            return self
        if self.online_spec is not None:
            raise ValueError("online_spec state is only valid for OnlineSPEC methods")
        if (
            self.runtime.tensor_parallel_size != 1
            or self.runtime.data_parallel_size != 1
        ):
            raise ValueError("schema-v2 TTS/L0 adaptation currently requires TP=DP=1")
        if not self.runtime.use_rejection_sampling:
            raise ValueError("TTS/L0 requires exact full-vocabulary rejection sampling")
        if self.adaptation.optimizer.name not in {
            "adam",
            "adamw",
            "sgdm",
            "nag",
            "muon",
            "lion",
        }:
            raise ValueError("unsupported optimizer for TTS/L0 speed tuning")
        if (
            algorithm in {"EAGLE", "EAGLE3"}
            and self.runtime.speculative_eagle_topk != 1
        ):
            raise ValueError("adapted EAGLE/EAGLE3 currently requires topk=1")
        if algorithm != "DFLASH" and self.adaptation.parameter_scope != "tail":
            raise ValueError(
                f"{algorithm} adaptation currently supports parameter_scope=tail"
            )
        if self.adaptation.canvas_tokens != self.runtime.speculative_num_draft_tokens:
            raise ValueError("adaptation canvas width must equal the speculative width")
        return self
