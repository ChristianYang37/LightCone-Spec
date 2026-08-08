"""Versioned, strict AdaptationConfig (spec section 10).

Every config is `extra="forbid"`: unknown fields are a schema error.
Cross-field compatibility constraints from spec 10.2 are enforced in
model validators so that an invalid method/optimizer/lifecycle
combination can never reach GPU initialization.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lightcone_spec.exit_codes import ConfigError

# ---------------------------------------------------------------------------
# Canonical enumerations (spec 2.2, 6.4, 6.14, 12.1)
# ---------------------------------------------------------------------------

METHOD_KEYS = (
    "static",
    "sync_fresh",
    "tts",
    "naive_async",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
    "lc_gate",
    "lc_damp",
    "lc_transport",
    "oracle_current",
)

# Report names must be used verbatim (spec 6.4).
METHOD_REPORT_NAMES = {
    "static": "DSpark-Static",
    "sync_fresh": "Sync-Fresh",
    "tts": "TTS-DSpark",
    "naive_async": "L0-NaiveAsync",
    "onlinespec_ogd": "OnlineSpec-OGD-DSpark",
    "onlinespec_opt": "OnlineSpec-Opt-DSpark",
    "onlinespec_ens": "OnlineSpec-Ens-DSpark",
    "lc_gate": "L1-LC-Gate",
    "lc_damp": "L2-LC-Damp",
    "lc_transport": "L3-LC-Transport",
    "oracle_current": "Oracle-Current",
}

# Diagnostic negative controls (spec 6.14). They enter P0/P1/P3
# mechanism figures only, never the P2 eleven-method main table.
DIAGNOSTIC_METHOD_KEYS = (
    "round_discard",
    "wall_damp",
    "endpoint_gate",
    "parameter_only",
    "random_transport",
)

DIAGNOSTIC_REPORT_NAMES = {
    "round_discard": "Round-Discard",
    "wall_damp": "Wall-Damp",
    "endpoint_gate": "Endpoint-Gate",
    "parameter_only": "Parameter-Only",
    "random_transport": "Random-Transport",
}

ALL_METHOD_KEYS = METHOD_KEYS + DIAGNOSTIC_METHOD_KEYS

# Updating *how* a parameter is represented and choosing *which* parameters
# are trainable are orthogonal decisions.  Early schema-v1 configs collapsed
# both decisions into names such as ``tail_lora``.  Keep those spellings
# readable, but never expose them as the public weight-update mode.
WEIGHT_UPDATE_MODE_CHOICES = ("residual", "lora", "full")
CANONICAL_WEIGHT_UPDATE_MODES = WEIGHT_UPDATE_MODE_CHOICES
PARAMETER_SCOPE_CHOICES = ("tail", "all", "allowlist")
TAIL_LAYOUT_MODES = (
    "output_residual",
    "tail_lora",
    "full_rank_tail",
)
_WEIGHT_UPDATE_MODE_ALIASES = {
    "adapter": "residual",
    "residual": "residual",
    "output-residual": "residual",
    "output_residual": "residual",
    "lora": "lora",
    "tail-lora": "lora",
    "tail_lora": "lora",
    "full": "full",
    "full-rank-tail": "full",
    "full_rank_tail": "full",
}

_TAIL_LAYOUT_BY_WEIGHT_UPDATE_MODE = {
    "residual": "output_residual",
    "lora": "tail_lora",
    "full": "full_rank_tail",
}


def canonical_weight_update_mode(value: str) -> str:
    """Resolve a public or schema-v1 spelling to ``residual|lora|full``."""
    try:
        return _WEIGHT_UPDATE_MODE_ALIASES[str(value)]
    except KeyError as exc:
        allowed = sorted(_WEIGHT_UPDATE_MODE_ALIASES)
        raise ValueError(
            f"unknown weight update mode {value!r}; allowed: {allowed}"
        ) from exc


def canonical_parameter_scope(value: str) -> str:
    """Return the independent trainable-parameter scope."""
    scope = str(value).replace("-", "_").lower()
    if scope not in PARAMETER_SCOPE_CHOICES:
        raise ValueError(
            f"unknown parameter scope {value!r}; allowed: "
            f"{list(PARAMETER_SCOPE_CHOICES)}"
        )
    return scope


def canonical_tail_layout_mode(value: str) -> str:
    """Resolve a public/schema-v1 mode to the frozen tail-bank layout name.

    The internal names are retained only for compatibility with existing
    parameter-layout and controller artifacts.  They are not public update
    modes and say nothing about the independent parameter scope.
    """
    return _TAIL_LAYOUT_BY_WEIGHT_UPDATE_MODE[canonical_weight_update_mode(value)]


def effective_proposal_depth(pair: dict, speculative_num_draft_tokens: int) -> int:
    """Return the backend-visible proposal-row count for one verify window.

    DSpark and DFlash expose a verify-window width: one row is the anchor/bonus
    position, so only ``n - 1`` rows are draft proposals.  EAGLE/EAGLE3 expose
    their proposal width directly.  The pair declaration is an upper bound in
    both cases; smaller runtime windows remain legal and must size every bank
    and graph buffer consistently.
    """

    declared = int(pair["draft_depth"])
    if declared < 1:
        raise ValueError(f"model pair draft_depth must be >= 1, got {declared}")
    window = int(speculative_num_draft_tokens)
    algorithm = str(pair["speculative_algorithm"]).upper()
    if algorithm in ("DSPARK", "DFLASH"):
        if window < 2:
            raise ValueError(
                f"{algorithm} speculative_num_draft_tokens must be >= 2 "
                "(one anchor plus at least one proposal)"
            )
        return min(declared, window - 1)
    if algorithm in ("EAGLE", "EAGLE3"):
        if window < 1:
            raise ValueError(
                f"{algorithm} speculative_num_draft_tokens must be >= 1"
            )
        return min(declared, window)
    raise ValueError(f"unsupported speculative algorithm {algorithm!r}")


# Model matrix (spec 2.2).  Capability declarations are consumed before model
# load; unsupported algorithm/mode combinations fail closed instead of falling
# through to a DSpark worker.
MODEL_PAIRS = {
    "qwen3_4b_dspark7": {
        "target": "Qwen/Qwen3-4B",
        "drafter": "deepseek-ai/dspark_qwen3_4b_block7",
        "draft_depth": 7,
        "speculative_algorithm": "DSPARK",
        "default_num_draft_tokens": 8,
        "max_context_length": 40960,
        "capabilities": {
            "tail_adaptation": True,
            "markov_features": True,
            "confidence_head": True,
            "linear_topk_one": True,
            "multi_layer": False,
            "enable_thinking": True,
            "reasoning_parser": "qwen3",
        },
    },
    "qwen3_8b_dspark7": {
        "target": "Qwen/Qwen3-8B",
        "drafter": "deepseek-ai/dspark_qwen3_8b_block7",
        "draft_depth": 7,
        "speculative_algorithm": "DSPARK",
        "default_num_draft_tokens": 8,
        "max_context_length": 40960,
        "capabilities": {
            "tail_adaptation": True,
            "markov_features": True,
            "confidence_head": True,
            "linear_topk_one": True,
            "multi_layer": False,
            "enable_thinking": True,
            "reasoning_parser": "qwen3",
        },
    },
    "qwen3_14b_dspark7": {
        "target": "Qwen/Qwen3-14B",
        "drafter": "deepseek-ai/dspark_qwen3_14b_block7",
        "draft_depth": 7,
        "speculative_algorithm": "DSPARK",
        "default_num_draft_tokens": 8,
        "max_context_length": 40960,
        "capabilities": {
            "tail_adaptation": True,
            "markov_features": True,
            "confidence_head": True,
            "linear_topk_one": True,
            "multi_layer": False,
            "enable_thinking": True,
            "reasoning_parser": "qwen3",
        },
    },
    "gemma4_12b_dspark7": {
        "target": "google/gemma-4-12B-it",
        "drafter": "deepseek-ai/dspark_gemma4_12b_block7",
        "draft_depth": 7,
        "speculative_algorithm": "DSPARK",
        "default_num_draft_tokens": 8,
        "max_context_length": None,
        "capabilities": {
            "tail_adaptation": True,
            "markov_features": True,
            "confidence_head": True,
            "linear_topk_one": True,
            "multi_layer": False,
            "enable_thinking": True,
            "reasoning_parser": "gemma4",
        },
    },
    "qwen3_4b_dflash16": {
        "target": "Qwen/Qwen3-4B",
        "drafter": "z-lab/Qwen3-4B-DFlash-b16",
        "draft_depth": 15,
        "speculative_algorithm": "DFLASH",
        "default_num_draft_tokens": 16,
        "max_context_length": 40960,
        "capabilities": {
            "tail_adaptation": True,
            "markov_features": False,
            "confidence_head": False,
            "linear_topk_one": True,
            "multi_layer": False,
            "enable_thinking": True,
            "reasoning_parser": "qwen3",
        },
    },
    "qwen3_8b_dflash16": {
        "target": "Qwen/Qwen3-8B",
        "drafter": "z-lab/Qwen3-8B-DFlash-b16",
        "draft_depth": 15,
        "speculative_algorithm": "DFLASH",
        "default_num_draft_tokens": 16,
        "max_context_length": 40960,
        "capabilities": {
            "tail_adaptation": True,
            "markov_features": False,
            "confidence_head": False,
            "linear_topk_one": True,
            "multi_layer": False,
            "enable_thinking": True,
            "reasoning_parser": "qwen3",
        },
    },
    "qwen3_8b_eagle3": {
        "target": "Qwen/Qwen3-8B",
        "drafter": "AngelSlim/Qwen3-8B_eagle3",
        "draft_depth": 8,
        "speculative_algorithm": "EAGLE3",
        "default_num_draft_tokens": 8,
        "max_context_length": 40960,
        "capabilities": {
            "tail_adaptation": True,
            "markov_features": False,
            "confidence_head": False,
            "linear_topk_one": True,
            "multi_layer": False,
            "enable_thinking": True,
            "reasoning_parser": "qwen3",
        },
    },
    "llama2_7b_eagle": {
        "target": "meta-llama/Llama-2-7b-chat-hf",
        "drafter": "lmsys/sglang-EAGLE-llama2-chat-7B",
        "draft_depth": 8,
        "speculative_algorithm": "EAGLE",
        "default_num_draft_tokens": 8,
        "max_context_length": 4096,
        "capabilities": {
            "tail_adaptation": True,
            "markov_features": False,
            "confidence_head": False,
            "linear_topk_one": True,
            "multi_layer": False,
            "enable_thinking": False,
            "reasoning_parser": None,
        },
    },
}


def pair_thinking_config(pair: dict) -> dict:
    """Resolve thinking-mode defaults for a MODEL_PAIRS entry.

    Qwen3 and Gemma instruct pairs default to thinking ON. Explicit capability
    flags win when present. Non-reasoning pairs (e.g. Llama-2) stay off.
    """
    caps = pair.get("capabilities") or {}
    target = str(pair.get("target", "")).lower()
    if "enable_thinking" in caps:
        enable = bool(caps["enable_thinking"])
    else:
        enable = ("qwen3" in target) or ("gemma" in target)
    if not enable:
        return {
            "enable_thinking": False,
            "reasoning_parser": None,
            "chat_template_kwargs": None,
        }
    parser = caps.get("reasoning_parser")
    if not parser:
        if "gemma" in target:
            parser = "gemma4"
        elif "qwen3" in target:
            parser = "qwen3"
        else:
            parser = None
    return {
        "enable_thinking": True,
        "reasoning_parser": parser,
        "chat_template_kwargs": {"enable_thinking": True},
    }

DSPARK_MODEL_PAIR_IDS = tuple(
    pair_id
    for pair_id, pair in MODEL_PAIRS.items()
    if pair["speculative_algorithm"] == "DSPARK"
)

BENCHMARK_ADAPTER_KEYS = (
    "gsm8k",
    "math500",
    "aime25",
    "mbpp",
    "humaneval",
    "livecodebench",
    "mt_bench",
    "alpaca",
    "arena_hard_v2",
    "aime24",
    "olympiadbench_math",
    "olympiadbench_physics",
    "gpqa_diamond",
    "theoremqa",
)

SYNTHETIC_DATASET_KEYS = (
    "markov4_world",
    "phase_switch",
    "aba_recurrence",
    "idle_insertion_twins",
    "wall_only_twins",
    "state_only_twins",
)

# Methods that use the common single-step AdamW TTS candidate generator
# (spec 6.8 / 10.2).
ADAMW_CANDIDATE_METHODS = (
    "tts",
    "naive_async",
    "lc_gate",
    "lc_damp",
    "lc_transport",
    "oracle_current",
    # Diagnostic controls share the L1/L2/L3 candidate path (spec 6.14).
    "round_discard",
    "wall_damp",
    "endpoint_gate",
    "parameter_only",
    "random_transport",
)

SGD_METHODS = ("onlinespec_ogd", "onlinespec_opt", "onlinespec_ens")

CONTROLLER_METHODS = (
    "lc_gate",
    "lc_damp",
    "lc_transport",
    "round_discard",
    "wall_damp",
    "endpoint_gate",
    "parameter_only",
    "random_transport",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AsyncConfig(_StrictModel):
    enabled: bool = True
    logical_delay_rounds: int = Field(0, ge=0)
    max_in_flight: int = Field(1, ge=1, le=2)
    stream_priority: int = 0


class TrajectoryConfig(_StrictModel):
    topk: int = 64
    hidden_proj_dim: int = 128
    clock_variant: Literal["target_only", "draft_aware", "gradient_oracle"] = (
        "target_only"
    )
    probability_weight: float = Field(0.3333333333, ge=0.0)
    hidden_weight: float = Field(0.3333333333, ge=0.0)
    event_weight: float = Field(0.3333333334, ge=0.0)


class ControllerConfig(_StrictModel):
    artifact_path: Optional[str] = None
    threshold: Optional[float] = None
    damping_kernel: Literal["exponential", "clipped_linear"] = "exponential"
    unsafe_apply_limit: float = Field(0.10, gt=0.0, lt=1.0)


class TransportConfig(_StrictModel):
    rank: Literal[8, 16, 32] = 16
    basis_path: Optional[str] = None
    ridge_alpha: Optional[float] = None
    fisher_decay: float = Field(0.99, gt=0.0, lt=1.0)


class TraceConfig(_StrictModel):
    level: Literal["full", "light"] = "full"
    privacy_mode: Literal["benchmark", "private"] = "benchmark"
    artifact_root: str
    telemetry_path: Optional[str] = None
    trace_capture_max_bytes: int = Field(0, ge=0)
    trace_capture_max_records_per_request: int = Field(4, ge=1)
    trace_capture_sampling: Literal["first", "staged"] = "first"
    # Explicit two-pass L3 evidence collection.  This permits a controller
    # artifact whose production L3 gate is still closed to run *only* as a
    # bounded benchmark trace producer.  It is never inferred from a positive
    # trace budget, so ordinary serving cannot accidentally bypass the gate.
    l3_evaluation_only: bool = False


class ModelConfig(_StrictModel):
    pair_id: str
    target_revision: str = "locked"
    drafter_revision: str = "locked"
    tokenizer_revision: str = "locked"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    projection_artifact_path: Optional[str] = None

    @model_validator(mode="after")
    def _check_pair(self) -> "ModelConfig":
        if self.pair_id not in MODEL_PAIRS and not self.pair_id.startswith("toy_"):
            raise ValueError(
                f"unknown model pair_id {self.pair_id!r}; "
                f"allowed: {sorted(MODEL_PAIRS)} or toy_* pairs"
            )
        return self


class DatasetConfig(_StrictModel):
    adapter: str
    revision: str = "locked"
    split: str = "test"

    @model_validator(mode="after")
    def _check_adapter(self) -> "DatasetConfig":
        allowed = set(BENCHMARK_ADAPTER_KEYS) | set(SYNTHETIC_DATASET_KEYS)
        if self.adapter not in allowed and not self.adapter.startswith("toy_"):
            raise ValueError(
                f"unknown dataset adapter {self.adapter!r}; allowed: {sorted(allowed)}"
            )
        return self


class SamplingConfig(_StrictModel):
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(32768, ge=1)
    ignore_eos: bool = False
    # Thinking is a model-pair default (Qwen3/Gemma ON). Recorded here so the
    # materialized runtime yaml is auditable; the SGLang bridge is the enforcer.
    enable_thinking: bool = False
    reasoning_parser: Optional[str] = None


class RuntimeConfig(_StrictModel):
    seed: int = 0
    concurrency: int = Field(1, ge=1, le=512)
    adaptation_slots: Optional[int] = Field(None, ge=1, le=512)
    # Materialized GPU runs bind this to the same resolved request/graph row
    # capacity used during pre-KV memory sizing.  It remains optional only for
    # schema-v1 replay fixtures and direct legacy configurations.
    adapter_row_capacity: Optional[int] = Field(None, ge=1)
    memory_safety_factor: float = Field(1.25, ge=1.0, le=2.0)
    calibrated_reserve_mb: int = Field(0, ge=0)
    tensor_parallel_size: int = Field(1, ge=1)
    speculative_num_draft_tokens: int = 8

    def resolved_adaptation_slots(self) -> int:
        return int(self.adaptation_slots or self.concurrency)


class AdaptationConfig(_StrictModel):
    """Top-level config (spec 10.1). schema_version must be 1."""

    schema_version: Literal[1]
    method: str
    lifecycle: Literal["request", "stream"] = "request"
    # ``trainable_scope`` is the schema-v1 wire key.  Its three historical
    # values encode only the tail-bank layout; new callers should provide
    # ``weight_update_mode`` (accepted by the migration validator below) plus
    # the independent ``parameter_scope``.
    trainable_scope: Literal[
        "output_residual", "tail_lora", "full_rank_tail"
    ] = "output_residual"
    parameter_scope: Literal["tail", "all", "allowlist"] = "tail"
    parameter_allowlist: tuple[str, ...] = ()
    update_stride: int = Field(10, ge=1)
    optimizer: Literal["adamw", "sgd", "none"] = "adamw"
    lr: float = Field(1e-4, gt=0.0)
    # Decoupled AdamW decay.  Schema-v1 configs predate this field and
    # therefore retain their historical numerical behaviour through the
    # explicit zero default.
    weight_decay: float = Field(0.0, ge=0.0)
    lambda_prox: float = Field(0.1, ge=0.0)
    confidence_loss_weight: float = 1.0
    grad_clip: float = Field(1.0, gt=0.0)
    trust_region_radius: float = Field(1.0, gt=0.0)
    adapter_rank: Literal[8, 16, 32] = 16

    async_: AsyncConfig = Field(default_factory=AsyncConfig, alias="async")
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    trace: TraceConfig
    model: ModelConfig
    dataset: DatasetConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _read_public_update_mode(cls, value: object) -> object:
        """Accept the new public key without invalidating schema-v1 files.

        Keeping the legacy field in the Pydantic model lets old frozen runtime
        YAML round-trip byte-for-byte at the semantic level.  New materialized
        configs may carry both keys; a disagreement is rejected instead of
        silently choosing one.
        """
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        public = raw.pop("weight_update_mode", None)
        if public is None:
            return raw
        public_mode = canonical_weight_update_mode(str(public))
        legacy = raw.get("trainable_scope")
        if (
            legacy is not None
            and canonical_weight_update_mode(str(legacy)) != public_mode
        ):
            raise ValueError(
                "weight_update_mode conflicts with deprecated "
                f"trainable_scope: {public!r} != {legacy!r}"
            )
        raw["trainable_scope"] = canonical_tail_layout_mode(public_mode)
        return raw

    @field_validator("trainable_scope", mode="before")
    @classmethod
    def _canonicalize_trainable_scope(cls, value: object) -> str:
        return canonical_tail_layout_mode(str(value))

    @field_validator("parameter_scope", mode="before")
    @classmethod
    def _canonicalize_parameter_scope(cls, value: object) -> str:
        return canonical_parameter_scope(str(value))

    @property
    def effective_adapter_rank(self) -> int | None:
        """Rank is not part of a full-update runtime/artifact identity."""
        return None if self.weight_update_mode == "full" else self.adapter_rank

    @property
    def weight_update_mode(self) -> str:
        """Public update representation: exactly residual, lora, or full."""
        return canonical_weight_update_mode(self.trainable_scope)

    @property
    def tail_layout_mode(self) -> str:
        """Frozen internal tail-bank spelling used by existing artifacts."""
        return self.trainable_scope

    # -- spec 10.2 compatibility constraints -------------------------------

    @model_validator(mode="after")
    def _compat(self) -> "AdaptationConfig":
        m = self.method
        if m not in ALL_METHOD_KEYS:
            raise ValueError(f"unknown method {m!r}; allowed: {sorted(ALL_METHOD_KEYS)}")

        allowlist = tuple(str(name).strip() for name in self.parameter_allowlist)
        if any(not name for name in allowlist):
            raise ValueError("parameter_allowlist entries must be non-empty")
        if len(set(allowlist)) != len(allowlist):
            raise ValueError("parameter_allowlist entries must be unique")
        if self.parameter_scope == "allowlist" and not allowlist:
            raise ValueError(
                "parameter_scope=allowlist requires a non-empty "
                "parameter_allowlist"
            )
        if self.parameter_scope != "allowlist" and allowlist:
            raise ValueError(
                "parameter_allowlist is only valid with "
                "parameter_scope=allowlist"
            )
        if self.weight_update_mode == "residual" and self.parameter_scope != "tail":
            raise ValueError(
                "weight_update_mode=residual is an output-tail update and "
                "requires parameter_scope=tail"
            )

        if m == "static":
            if self.optimizer != "none":
                raise ValueError("static forbids an optimizer (use optimizer: none)")
            if self.async_.enabled:
                raise ValueError("static forbids async updates")
            if self.controller.artifact_path is not None:
                raise ValueError("static forbids a controller artifact")

        if m == "sync_fresh":
            if self.optimizer != "adamw":
                raise ValueError("sync_fresh requires AdamW")
            if self.async_.enabled:
                raise ValueError("sync_fresh requires async.enabled=false")

        if m in ADAMW_CANDIDATE_METHODS and self.optimizer != "adamw":
            raise ValueError(f"{m} requires AdamW (single-step candidate generator)")

        if self.optimizer != "adamw" and self.weight_decay != 0.0:
            raise ValueError(
                "weight_decay is only valid with optimizer=adamw"
            )

        if m == "tts" and self.async_.max_in_flight != 1:
            raise ValueError("tts requires max_in_flight=1")

        if m in SGD_METHODS and self.optimizer != "sgd":
            raise ValueError(f"{m} requires SGD")

        if m in ("lc_gate", "lc_damp", "lc_transport") and (
            self.controller.artifact_path is None
        ):
            raise ValueError(f"{m} requires a frozen controller artifact")

        if m == "lc_transport" and self.transport.basis_path is None:
            raise ValueError("lc_transport requires a transport artifact")

        if self.trace.l3_evaluation_only:
            if m != "lc_transport":
                raise ValueError(
                    "trace.l3_evaluation_only is only valid for lc_transport"
                )
            if self.trace.trace_capture_max_bytes <= 0:
                raise ValueError(
                    "trace.l3_evaluation_only requires a positive bounded "
                    "trace_capture_max_bytes budget"
                )
            if self.trace.privacy_mode != "benchmark":
                raise ValueError(
                    "trace.l3_evaluation_only is restricted to benchmark traces"
                )

        if self.async_.max_in_flight == 2 and m != "lc_transport":
            raise ValueError(
                "max_in_flight=2 is only allowed for the lc_transport "
                "parameter-staleness manifest"
            )

        if self.model.pair_id in MODEL_PAIRS:
            pair = MODEL_PAIRS[self.model.pair_id]
            capabilities = pair["capabilities"]
            if m != "static" and not capabilities["tail_adaptation"]:
                raise ValueError(
                    f"{self.model.pair_id} does not support tail adaptation"
                )

        if self.sampling.temperature not in (0.0, 1.0):
            raise ValueError(
                "sampling.temperature must be 1.0 (main) or 0.0 (greedy parity)"
            )
        if self.sampling.temperature == 1.0 and self.sampling.top_p != 1.0:
            raise ValueError("main sampling requires top_p=1.0")

        if self.trajectory.topk != 64 and self.trajectory.clock_variant == "target_only":
            raise ValueError("trajectory.topk main value must be 64 for target_only")

        return self

    def report_name(self) -> str:
        return METHOD_REPORT_NAMES.get(self.method) or DIAGNOSTIC_REPORT_NAMES[self.method]


def validation_error_to_config_error(exc: Exception) -> ConfigError:
    return ConfigError(f"invalid AdaptationConfig: {exc}")
