"""Strict configuration for native Target-only, Static, TTS, and L0."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lightcone_spec import PINNED_SGLANG_COMMIT
from lightcone_spec.adaptation.parameters import (
    DSPARK_HYBRID_SCOPES,
    LAYER_SCOPES,
    LORA_RANKS,
)

CoreMethod = Literal["target_only", "static", "tts", "l0"]
BaselineMethod = Literal["onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"]
Method = CoreMethod | BaselineMethod
CORE_METHODS = ("target_only", "static", "tts", "l0")
EXTERNAL_BASELINES = (
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
)
METHODS = CORE_METHODS + EXTERNAL_BASELINES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelPair(StrictModel):
    key: str = "qwen3_8b_dflash16"
    target: str = "Qwen/Qwen3-8B"
    drafter: str = "z-lab/Qwen3-8B-DFlash-b16"
    target_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    drafter_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    algorithm: Literal["DFLASH", "DSPARK", "EAGLE", "EAGLE3", "NEXTN"] = (
        "DFLASH"
    )
    max_context_length: int = Field(default=40960, ge=1)
    draft_depth: int = Field(default=15, ge=1)


class OptimizerConfig(StrictModel):
    name: Literal["adam", "adamw", "sgd", "sgdm", "nag", "muon", "lion", "none"]
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
    schedule: Literal[
        "constant", "inverse_sqrt_published_update", "cosine_to_zero"
    ] = "constant"
    schedule_total_published_updates: int | None = Field(default=None, ge=2)

    @model_validator(mode="after")
    def validate_optimizer(self) -> OptimizerConfig:
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
            raise ValueError("muon requires explicit auxiliary AdamW lr and weight decay")
        if self.name != "muon" and any(value is not None for value in auxiliary):
            raise ValueError("Muon auxiliary AdamW fields require optimizer=muon")
        if (self.schedule == "cosine_to_zero") != (
            self.schedule_total_published_updates is not None
        ):
            raise ValueError(
                "cosine_to_zero requires exactly one published-update horizon"
            )
        return self


class AdaptationConfig(StrictModel):
    weight_update_mode: Literal["lora", "full"]
    parameter_scope: str = Field(min_length=1, max_length=64)
    kv_history_policy: Literal["frozen"] = "frozen"
    adaptation_scope: Literal["cohort"] = "cohort"
    adaptation_group_id: str = Field(min_length=1, max_length=128)
    optimizer: OptimizerConfig
    rank: int | None = Field(default=None, ge=1)
    lora_alpha: int | None = Field(default=None, ge=1)
    lora_matrix_policy: Literal["registered_matrices_v1"] = "registered_matrices_v1"
    native_head_policy: Literal["frozen", "full"] = "frozen"
    stride: int = Field(default=10, ge=1)
    max_in_flight: Literal[1] = 1
    canvas_tokens: int = Field(default=16, ge=2)
    loss_position_decay: float = Field(default=1.0, gt=0.0, le=1.0)
    extra_logical_delay: int = Field(default=0, ge=0)
    teacher_row_policy: Literal["update_round", "quota_shadow"] = "update_round"
    verification_mode: Literal["native_scheduler", "fixed_budget"] = (
        "native_scheduler"
    )
    fixed_verification_budget: int | None = Field(default=None, ge=1)
    confidence_loss_weight: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_mode(self) -> AdaptationConfig:
        if self.weight_update_mode == "full":
            if self.rank is not None or self.lora_alpha is not None:
                raise ValueError("full updates require rank and lora_alpha to be null")
        elif self.rank not in LORA_RANKS or self.lora_alpha != self.rank:
            raise ValueError("LoRA requires a registered rank and alpha/r=1")
        if self.optimizer.name == "none":
            raise ValueError("adaptation requires an enabled optimizer")
        if (self.verification_mode == "fixed_budget") != (
            self.fixed_verification_budget is not None
        ):
            raise ValueError("fixed-budget verification requires exactly one budget")
        return self


class OnlineSpecConfig(StrictModel):
    """Algorithm state isolated from the Static/TTS/L0 hypothesis."""

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
    speculation_enabled: bool = True
    tensor_parallel_size: int = Field(default=1, ge=1, le=2)
    data_parallel_size: int = Field(default=1, ge=1, le=2)
    tp_rank: int = Field(default=0, ge=0)
    dp_rank: int = Field(default=0, ge=0)
    node_count: int = Field(default=1, ge=1)
    node_rank: int = Field(default=0, ge=0)
    device_identity: str = Field(default="local-device-0", min_length=1)
    rendezvous_identity: str = Field(default="local-rendezvous", min_length=1)
    router_identity: str = Field(default="single-replica", min_length=1)
    clock_identity: str = Field(default="monotonic", min_length=1)
    process_group_backend: Literal["nccl", "gloo"] = "nccl"
    distributed_runtime_capability: Literal[
        "single_rank", "patched_two_gpu_v1"
    ] = "single_rank"
    distributed_capability_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    speculative_num_draft_tokens: int = Field(default=16, ge=2)
    speculative_eagle_topk: int | None = Field(default=None, ge=1)
    use_rejection_sampling: bool = True
    max_running_requests: int = Field(default=1, ge=1)
    telemetry_detail: Literal["headline", "profile"] = "headline"
    prefill_decode_disaggregation: Literal[False] = False
    two_batch_overlap: Literal[False] = False

    @model_validator(mode="after")
    def validate_topology(self) -> RuntimeConfig:
        if self.tp_rank >= self.tensor_parallel_size:
            raise ValueError("tp_rank is outside tensor_parallel_size")
        if self.dp_rank >= self.data_parallel_size:
            raise ValueError("dp_rank is outside data_parallel_size")
        if self.node_rank >= self.node_count:
            raise ValueError("node_rank is outside node_count")
        if self.node_count != 1:
            raise ValueError("multi-node LightCone remains UNMEASURED and fail-closed")
        if self.tensor_parallel_size * self.data_parallel_size > 2:
            raise ValueError("the registered two-GPU topology supports at most two ranks")
        if self.data_parallel_size > 1 and self.router_identity == "single-replica":
            raise ValueError("DP replicas require an explicit sticky router identity")
        distributed = self.tensor_parallel_size * self.data_parallel_size > 1
        if distributed:
            raise ValueError(
                "the pinned schema-v3 release does not expose TP2/DP2 execution; "
                "a caller-authored capability digest cannot enable it"
            )
        elif (
            self.distributed_runtime_capability != "single_rank"
            or self.distributed_capability_receipt_sha256 is not None
        ):
            raise ValueError(
                "single-rank runs cannot claim a distributed runtime capability"
            )
        return self


class RunConfig(StrictModel):
    schema_version: Literal[3] = 3
    method: Method
    model: ModelPair
    runtime: RuntimeConfig
    adaptation: AdaptationConfig | None = None
    online_spec: OnlineSpecConfig | None = None
    tenant_id: str = Field(default="research", min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_method_contract(self) -> RunConfig:
        algorithm = self.model.algorithm
        if self.method == "target_only":
            if self.runtime.speculation_enabled:
                raise ValueError("target_only requires speculation_enabled=false")
            if self.adaptation is not None or self.online_spec is not None:
                raise ValueError("target_only must not allocate adaptation state")
            return self
        if not self.runtime.speculation_enabled:
            raise ValueError("speculative methods require speculation_enabled=true")
        if self.runtime.speculative_num_draft_tokens != self.model.draft_depth + 1:
            raise ValueError(f"{algorithm} verify width must equal draft_depth + 1")
        if algorithm in {"DFLASH", "DSPARK", "NEXTN"}:
            if self.runtime.speculative_eagle_topk is not None:
                raise ValueError("speculative_eagle_topk is only valid for EAGLE backends")
        elif self.runtime.speculative_eagle_topk is None:
            raise ValueError("EAGLE backends require speculative_eagle_topk")
        if self.method == "static":
            if self.adaptation is not None or self.online_spec is not None:
                raise ValueError("static must not allocate adaptation state")
            return self
        if self.adaptation is None:
            raise ValueError(f"method={self.method} requires adaptation configuration")
        if algorithm != "DFLASH":
            raise ValueError(
                "the pinned schema-v3 patch executes adaptation only for DFLASH"
            )
        if self.adaptation.optimizer.schedule != "constant":
            raise ValueError(
                "the pinned schema-v3 patch executes only a constant optimizer schedule"
            )
        if self.adaptation.extra_logical_delay != 0:
            raise ValueError(
                "the pinned schema-v3 patch does not execute positive extra logical delay"
            )
        if self.adaptation.teacher_row_policy != "update_round":
            raise ValueError(
                "the pinned schema-v3 patch does not execute quota-shadow teacher rows"
            )
        self._validate_backend_scope()
        if self.method.startswith("onlinespec_"):
            self._validate_onlinespec()
            return self
        if self.online_spec is not None:
            raise ValueError("online_spec state is only valid for OnlineSPEC methods")
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
        if self.adaptation.canvas_tokens != self.runtime.speculative_num_draft_tokens:
            raise ValueError("adaptation canvas width must equal the speculative width")
        return self

    def _validate_backend_scope(self) -> None:
        assert self.adaptation is not None
        algorithm = self.model.algorithm
        scope = self.adaptation.parameter_scope
        if algorithm == "DSPARK":
            valid = {*LAYER_SCOPES, *DSPARK_HYBRID_SCOPES}
        else:
            valid = set(LAYER_SCOPES)
        if scope not in valid:
            raise ValueError(f"{algorithm} parameter scope is not registered")
        hybrid = scope in DSPARK_HYBRID_SCOPES
        expected_head_policy = "full" if hybrid else "frozen"
        if self.adaptation.native_head_policy != expected_head_policy:
            raise ValueError(
                "DSpark hybrid scopes require Full W1/W2/acceptance; layer-only "
                "scopes freeze every native head"
            )
        if algorithm != "DSPARK" and (
            self.adaptation.verification_mode != "native_scheduler"
            or self.adaptation.confidence_loss_weight is not None
        ):
            raise ValueError("DSpark verification/confidence fields are backend-owned")
        if algorithm == "DSPARK":
            if hybrid and self.adaptation.confidence_loss_weight is None:
                raise ValueError("composite DSpark candidates require confidence loss")
            if not hybrid and self.adaptation.confidence_loss_weight is not None:
                raise ValueError("layer-only DSpark keeps the confidence head frozen")
        if algorithm in {"EAGLE", "EAGLE3"} and self.runtime.speculative_eagle_topk != 1:
            raise ValueError("adapted EAGLE/EAGLE3 currently requires topk=1")

    def _validate_onlinespec(self) -> None:
        assert self.adaptation is not None
        if self.online_spec is None:
            raise ValueError("OnlineSPEC baselines require online_spec state")
        if self.adaptation.optimizer.name != "sgd":
            raise ValueError("OnlineSPEC baselines preserve their SGD update")
        if (
            self.runtime.tensor_parallel_size != 1
            or self.runtime.data_parallel_size != 1
        ):
            raise ValueError("OnlineSPEC comparisons retain separate TP1/DP1 tuning")
        if not self.runtime.use_rejection_sampling:
            raise ValueError("OnlineSPEC requires exact rejection sampling")
        if self.method == "onlinespec_ens":
            rates = (
                self.adaptation.optimizer.learning_rate,
                *self.online_spec.additional_learning_rates,
            )
            if len(rates) < 2 or tuple(sorted(rates)) != rates:
                raise ValueError("OnlineSPEC Hedge requires an increasing multi-rate grid")
            if self.online_spec.hedge_learning_rate is None:
                raise ValueError("OnlineSPEC Hedge requires hedge_learning_rate")
        elif (
            self.online_spec.additional_learning_rates
            or self.online_spec.hedge_learning_rate is not None
        ):
            raise ValueError("ensemble fields are only valid for onlinespec_ens")
