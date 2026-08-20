"""Signed scientific authorities for the formal LightCone-Spec protocol.

This module is deliberately independent from the legacy, eagerly expanded
industrial registry.  It defines the immutable scientific identity that a
staged registry must bind before any formal cell can be materialized.  Private
signing keys are never accepted or stored here: callers provision signatures
out of band and this module verifies them against an explicitly pinned public
policy digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)

FORMAL_METHOD_ROLES = (
    "Target-only",
    "Static",
    "TTS",
    "L0-naive",
    "LightCone",
)
E0_METHOD_ROLES = FORMAL_METHOD_ROLES + (
    "OnlineSPEC-OGD",
    "OnlineSPEC-OPT",
    "OnlineSPEC-ENS",
)
FORMAL_STAGE_DAG = (
    "preflight",
    "E3a",
    "TTS-Cal",
    "E1",
    "E2",
    "E4",
    "E3b",
    "E1a",
    "E5",
    "E6",
    "E0",
)
FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS = (
    "all_stage_execution_mapper",
    "download_completion_reducer",
    "e0_compatibility_reducer",
    "e0_fdr_reducer",
    "e0_power_prefix_reducer",
    "e1_pareto_reducer",
    "e1a_verification_reducer",
    "e2_successive_halving_reducer",
    "e3a_selection_reducer",
    "e3b_confirmation_reducer",
    "e3b_power_prefix_reducer",
    "e4_local_factorial_reducer",
    "e4_strength2_screen_reducer",
    "e4_winner_neighborhood_reducer",
    "e5_anchor_selection_reducer",
    "e5_confirmation_reducer",
    "e5_failure_reducer",
    "e5_power_prefix_reducer",
    "e6_confirmation_reducer",
    "e6_model_compatibility_reducer",
    "e6_power_prefix_reducer",
    "failure_actuator",
    "gpu_hour_budget_reducer",
    "onlinespec_learner",
    "onlinespec_tuning_reducer",
    "power_energy_sampler",
    "profiler_runner",
    "stage_coverage_reducer",
    "stage_materialization_reducer",
    "tts_calibration_reducer",
)
TTS_L0_CANDIDATE_STATE_COVERAGE_STAGES = (
    "preflight",
    "E1",
    "E2",
    "E3b",
    "E5",
    "E6",
    "E0",
)
PRIMARY_HOLM_FAMILY = ("LightCone-Static", "LightCone-TTS")
SECONDARY_MECHANISM_CONTRASTS = (
    "L0-naive-TTS",
    "LightCone-L0-naive",
)
DEPLOYMENT_CONTRAST = "LightCone-Target-only-SLO-feasible"

E6_MODELS = (
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B-FP8",
)
BANNED_MODEL = "Qwen/Qwen3.5-35B-A3B"

TTS_LEARNING_RATES = (
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
)
TTS_STRIDES = (1, 5, 10, 15, 20, 30, 40, 50)
TTS_PRIMARY_SOURCE_ID = "arXiv:2605.09329"
TTS_PRIMARY_SOURCE_VERSION = "v2"

_SHA256_LENGTH = 64
_CHRONOBELIEF_EQUATIONS = (
    "m_r=beta1*m_(r-1)+(1-beta1)*g_r",
    "s_r=beta2*s_(r-1)+(1-beta2)*(g_r-m_r)^{odot2}",
    "kappa(d_r)=min(1,(beta1/sqrt(beta2))^d_r)",
    ("theta_(r+1)=(1-eta*lambda)*theta_r-eta*kappa(d_r)*mhat_r/(sqrt(shat_r)+epsilon)"),
)


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return _canonical(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            if int(field.metadata.get("canonical_since_schema", 1))
            <= int(getattr(value, "schema_version", 1))
        }
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("canonical mappings require exact string keys")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical content cannot contain non-finite floats")
        return 0.0 if value == 0 else value
    if value is None or type(value) in {str, int, bool}:
        return value
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode authority content without accepting NaN or ambiguous key types."""

    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def code_owned_qualification_source_identities() -> dict[str, tuple[str, str, str]]:
    """Return the three closed qualification identities used by ProtocolLock.

    This is intentionally a code-owned constructor.  The command that creates
    a lock must call it in-process; accepting any of these nine values from a
    caller would let an arbitrary qualification implementation become part of
    the scientific identity.
    """

    from lightcone_spec.runtime.compile_runner import (
        COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256,
        COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
    )
    from lightcone_spec.runtime.preflight_runner import (
        PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256,
        PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
        PREFLIGHT_EXACTNESS_TEST_NAMES,
    )
    from lightcone_spec.runtime.readiness import (
        NATIVE_RUNTIME_QUALIFICATION_PROTOCOL_SHA256,
        NATIVE_RUNTIME_QUALIFICATION_RUNNER_SHA256,
        NATIVE_RUNTIME_QUALIFICATION_TEST_SET_SHA256,
    )
    from lightcone_spec.sglang_bridge.compile_worker import (
        SOURCE_OWNED_COMPILE_PROTOCOL_SHA256,
    )

    return {
        "native_runtime": (
            NATIVE_RUNTIME_QUALIFICATION_PROTOCOL_SHA256,
            NATIVE_RUNTIME_QUALIFICATION_RUNNER_SHA256,
            NATIVE_RUNTIME_QUALIFICATION_TEST_SET_SHA256,
        ),
        "compile": (
            SOURCE_OWNED_COMPILE_PROTOCOL_SHA256,
            COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
            content_sha256(
                {
                    "schema_version": 1,
                    "kind": "formal_compile_qualification_test_set",
                    "assignment_protocol_sha256": (
                        COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256
                    ),
                    "worker_protocol_sha256": SOURCE_OWNED_COMPILE_PROTOCOL_SHA256,
                }
            ),
        ),
        "exactness": (
            PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256,
            PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
            content_sha256(PREFLIGHT_EXACTNESS_TEST_NAMES),
        ),
    }


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lower-case SHA-256")
    return value


def _require_git_oid(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lower-case 40-hex Git object ID")
    return value


def _require_text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be exact non-empty single-line text")
    return value


def reject_banned_model_identity(value: object, *, path: str = "root") -> None:
    """Reject the banned model anywhere in an authority or disposition tree.

    The check is substring based so a model cannot be hidden inside a variant,
    reason, selector, download cell, or other free-form field.  This validator
    is intentionally public and is called by every formal materialization and
    coverage payload.
    """

    if type(value) is str:
        if BANNED_MODEL in value:
            raise ValueError(f"banned E6 model appears at {path}")
        return
    if isinstance(value, Enum):
        reject_banned_model_identity(value.value, path=path)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            reject_banned_model_identity(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_banned_model_identity(key, path=f"{path}.key")
            reject_banned_model_identity(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            reject_banned_model_identity(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class ChronoBeliefAuthority:
    """Project-owned preregistration of equations 5.5--5.8."""

    schema_version: int
    authority_id: str
    paper_pdf_sha256: str
    tex_source_sha256: str
    equations: tuple[str, ...] = _CHRONOBELIEF_EQUATIONS
    bias_correction: str = "standard_update_count"
    weight_decay_semantics: str = "decoupled"
    age_semantics: str = "safe_boundary_age"
    skipped_transition_semantics: str = "moments_and_update_count_unchanged"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only ChronoBelief authority schema 1 is supported")
        _require_text("ChronoBelief authority ID", self.authority_id)
        _require_sha256("ChronoBelief PDF digest", self.paper_pdf_sha256)
        _require_sha256("ChronoBelief TeX digest", self.tex_source_sha256)
        if self.equations != _CHRONOBELIEF_EQUATIONS:
            raise ValueError("ChronoBelief equations differ from 5.5--5.8")
        if (
            self.bias_correction != "standard_update_count"
            or self.weight_decay_semantics != "decoupled"
            or self.age_semantics != "safe_boundary_age"
            or self.skipped_transition_semantics != "moments_and_update_count_unchanged"
        ):
            raise ValueError("ChronoBelief transition semantics are not canonical")
        reject_banned_model_identity(self)

    @cached_property
    def equation_sha256(self) -> str:
        return content_sha256(self.equations)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ChronoBeliefState:
    parameters: tuple[float, ...]
    first_moments: tuple[float, ...]
    second_moments: tuple[float, ...]
    update_count: int

    def __post_init__(self) -> None:
        if not self.parameters or not (
            len(self.parameters) == len(self.first_moments) == len(self.second_moments)
        ):
            raise ValueError("ChronoBelief state vectors must be non-empty and aligned")
        if any(
            not math.isfinite(value)
            for vector in (
                self.parameters,
                self.first_moments,
                self.second_moments,
            )
            for value in vector
        ):
            raise ValueError("ChronoBelief state must be finite")
        if type(self.update_count) is not int or self.update_count < 0:
            raise ValueError("ChronoBelief update count must be non-negative")


def chronobelief_reference_transition(
    state: ChronoBeliefState,
    gradients: tuple[float, ...],
    *,
    safe_boundary_age: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    outcome: Literal["commit", "skip", "abort"] = "commit",
) -> ChronoBeliefState:
    """Evaluate one exact scalar-vector CPU reference transition.

    ``skip`` and ``abort`` return the original immutable state exactly.  This
    makes it impossible for an unpublishable proposal to advance either
    moments or the bias-correction update count.
    """

    if type(state) is not ChronoBeliefState:
        raise TypeError("ChronoBelief transition requires an exact state")
    if outcome not in {"commit", "skip", "abort"}:
        raise ValueError("ChronoBelief outcome must be commit, skip, or abort")
    if outcome != "commit":
        return state
    if len(gradients) != len(state.parameters) or any(
        not math.isfinite(value) for value in gradients
    ):
        raise ValueError("ChronoBelief gradients must be finite and aligned")
    if type(safe_boundary_age) is not int or safe_boundary_age < 0:
        raise ValueError("ChronoBelief safe-boundary age must be non-negative")
    for name, value in (
        ("learning rate", learning_rate),
        ("epsilon", epsilon),
        ("weight decay", weight_decay),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"ChronoBelief {name} must be finite and non-negative")
    if learning_rate == 0 or epsilon == 0:
        raise ValueError("ChronoBelief learning rate and epsilon must be positive")
    if not 0 < beta1 < 1 or not 0 < beta2 < 1:
        raise ValueError("ChronoBelief betas must be in (0, 1)")

    update_count = state.update_count + 1
    try:
        first = tuple(
            beta1 * old + (1.0 - beta1) * gradient
            for old, gradient in zip(state.first_moments, gradients, strict=True)
        )
        second = tuple(
            beta2 * old + (1.0 - beta2) * (gradient - moment) ** 2
            for old, gradient, moment in zip(
                state.second_moments, gradients, first, strict=True
            )
        )
        correction1 = 1.0 - beta1**update_count
        correction2 = 1.0 - beta2**update_count
        age_ratio = beta1 / math.sqrt(beta2)
        kappa = 1.0 if age_ratio >= 1.0 else age_ratio**safe_boundary_age
        parameters = tuple(
            (1.0 - learning_rate * weight_decay) * parameter
            - learning_rate
            * kappa
            * (moment1 / correction1)
            / (math.sqrt(moment2 / correction2) + epsilon)
            for parameter, moment1, moment2 in zip(
                state.parameters, first, second, strict=True
            )
        )
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError(
            "ChronoBelief transition produced a non-finite state"
        ) from error
    if any(
        not math.isfinite(value)
        for vector in (first, second, parameters)
        for value in vector
    ):
        raise ValueError("ChronoBelief transition produced a non-finite state")
    return ChronoBeliefState(parameters, first, second, update_count)


@dataclass(frozen=True)
class TtsCalibrationAuthority:
    """Project-calibrated TTS grid bound to the pinned DFlash runtime recipe."""

    schema_version: int
    authority_id: str
    primary_source_id: str
    primary_source_version: str
    paper_pdf_sha256: str
    paper_source_sha256: str
    tuning_window_sha256: str
    trainable_plan_sha256: str
    drafter_native_loss_recipe_sha256: str
    learning_rates: tuple[float, ...] = TTS_LEARNING_RATES
    strides: tuple[int, ...] = TTS_STRIDES
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    gradient_clipping: str = "none"
    trainable_scope: str = "full_drafter"
    optimization_steps_per_update: int = 1
    teacher_rows: str = "latest_update_round_only"
    loss_objective: str = "masked_position_weighted_target_to_draft_forward_kl"
    loss_accumulation_precision: str = "float32"
    loss_temperature: float = 1.0
    position_weight_formula: str = "exp(-(k-1)/7)"
    loss_position_decay: float = DFLASH_LOSS_POSITION_DECAY
    loss_normalization: str = "masked_weighted_mean_denominator_clamped_min_1"
    source_point_value_correction: str = (
        "inference_forward_value_with_differentiable_surrogate_jacobian"
    )
    proximal_penalty: str = "absent"
    reset_scope: str = "request"
    execution_stream: str = "side_stream"
    excluded_pilot_blocks: tuple[int, ...] = (0, 1, 2, 3)
    selection_rule: str = "safety_first_then_maximize_slo_goodput"
    result_class: str = "tuning_only_not_formal"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("only TTS calibration authority schema 2 is supported")
        _require_text("TTS calibration authority ID", self.authority_id)
        if (
            self.primary_source_id != TTS_PRIMARY_SOURCE_ID
            or self.primary_source_version != TTS_PRIMARY_SOURCE_VERSION
            or f"{self.primary_source_id}{self.primary_source_version}"
            != "arXiv:2605.09329v2"
        ):
            raise ValueError("TTS primary source must be exactly arXiv:2605.09329v2")
        for name in (
            "paper_pdf_sha256",
            "paper_source_sha256",
            "tuning_window_sha256",
            "trainable_plan_sha256",
            "drafter_native_loss_recipe_sha256",
        ):
            _require_sha256(f"TTS {name}", getattr(self, name))
        if self.learning_rates != TTS_LEARNING_RATES or self.strides != TTS_STRIDES:
            raise ValueError("TTS calibration grid differs from the protocol")
        if (
            self.optimizer != "adam"
            or self.beta1 != 0.9
            or self.beta2 != 0.999
            or self.epsilon != 1e-8
            or self.weight_decay != 0.0
            or self.gradient_clipping != "none"
            or self.trainable_scope != "full_drafter"
            or self.optimization_steps_per_update != 1
            or self.teacher_rows != "latest_update_round_only"
            or self.loss_objective
            != "masked_position_weighted_target_to_draft_forward_kl"
            or self.loss_accumulation_precision != "float32"
            or self.loss_temperature != 1.0
            or self.position_weight_formula != "exp(-(k-1)/7)"
            or self.loss_position_decay != DFLASH_LOSS_POSITION_DECAY
            or self.loss_normalization
            != "masked_weighted_mean_denominator_clamped_min_1"
            or self.source_point_value_correction
            != "inference_forward_value_with_differentiable_surrogate_jacobian"
            or self.proximal_penalty != "absent"
            or self.reset_scope != "request"
            or self.execution_stream != "side_stream"
            or self.excluded_pilot_blocks != (0, 1, 2, 3)
            or self.selection_rule != "safety_first_then_maximize_slo_goodput"
            or self.result_class != "tuning_only_not_formal"
        ):
            raise ValueError("TTS calibration semantics differ from the protocol")
        reject_banned_model_identity(self)

    def candidate_id(self, *, learning_rate: float, stride: int) -> str:
        if learning_rate not in self.learning_rates or stride not in self.strides:
            raise ValueError("TTS candidate lies outside the sealed grid")
        return content_sha256(
            {
                "authority_sha256": self.sha256,
                "learning_rate": learning_rate,
                "stride": stride,
            }
        )

    def validate_runtime_optimizer_config(self, config: object) -> None:
        """Require the frozen TTS/L0 numeric recipe, including literal no-clip."""

        from lightcone_spec.config.schema import OptimizerConfig

        if type(config) is not OptimizerConfig:
            raise TypeError("TTS runtime recipe requires an exact OptimizerConfig")
        if (
            config.name != "adam"
            or config.learning_rate not in self.learning_rates
            or config.beta1 != self.beta1
            or config.beta2 != self.beta2
            or config.epsilon != self.epsilon
            or config.weight_decay != self.weight_decay
            or config.grad_clip is not None
            or config.schedule != "constant"
            or config.schedule_total_published_updates is not None
        ):
            raise ValueError(
                "TTS runtime optimizer differs from the frozen no-clip recipe"
            )

    def validate_runtime_adaptation_config(
        self,
        config: object,
        *,
        learning_rate: float,
        stride: int,
        canvas_tokens: int,
    ) -> None:
        """Require the complete fixed TTS/L0 adaptation and loss configuration."""

        from lightcone_spec.config.schema import AdaptationConfig

        if type(config) is not AdaptationConfig:
            raise TypeError("TTS runtime recipe requires an exact AdaptationConfig")
        if learning_rate not in self.learning_rates or stride not in self.strides:
            raise ValueError("TTS runtime candidate lies outside the sealed grid")
        self.validate_runtime_optimizer_config(config.optimizer)
        if (
            config.optimizer.learning_rate != learning_rate
            or config.weight_update_mode != "full"
            or config.parameter_scope != "all"
            or config.kv_history_policy != "frozen"
            or config.adaptation_scope != "cohort"
            or config.reset_scope != "request"
            or config.request_admission_policy != "serialized_native_scheduler_v1"
            or config.rank is not None
            or config.lora_alpha is not None
            or config.lora_matrix_policy != "registered_matrices_v1"
            or config.native_head_policy != "frozen"
            or config.stride != stride
            or config.max_in_flight != 1
            or config.canvas_tokens != canvas_tokens
            or config.loss_position_decay != self.loss_position_decay
            or config.extra_logical_delay != 0
            or config.teacher_row_policy != "update_round"
            or config.verification_mode != "native_scheduler"
            or config.fixed_verification_budget is not None
            or config.confidence_loss_weight is not None
            or config.chronobelief_release_capability_sha256 is not None
            or config.chronobelief_gpu_proof_sha256 is not None
        ):
            raise ValueError(
                "TTS runtime adaptation differs from the frozen DFlash recipe"
            )

    @cached_property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            self.candidate_id(learning_rate=learning_rate, stride=stride)
            for learning_rate in self.learning_rates
            for stride in self.strides
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


_TTS_CALIBRATION_SEAL_CONSTRUCTION = object()


@dataclass(frozen=True, init=False)
class TtsCalibrationSeal:
    """Offline-selected recipe derived only from the exact raw TTS reduction."""

    schema_version: int
    authority_sha256: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    reduction_receipt_sha256: str
    raw_manifest_sha256: str
    tuning_window_sha256: str
    selected_learning_rate: float
    selected_stride: int
    selected_candidate_id: str
    selected_pilot_run_binding_sha256s: tuple[str, ...]
    selection_rule: str = "safety_first_then_maximize_slo_goodput"
    result_class: str = "tuning_only_not_formal"

    def __init__(
        self,
        *,
        schema_version: int,
        authority_sha256: str,
        protocol_lock_sha256: str,
        materialization_receipt_sha256: str,
        coverage_receipt_sha256: str,
        reduction_receipt_sha256: str,
        raw_manifest_sha256: str,
        tuning_window_sha256: str,
        selected_learning_rate: float,
        selected_stride: int,
        selected_candidate_id: str,
        selected_pilot_run_binding_sha256s: tuple[str, ...],
        selection_rule: str = "safety_first_then_maximize_slo_goodput",
        result_class: str = "tuning_only_not_formal",
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _TTS_CALIBRATION_SEAL_CONSTRUCTION:
            raise TypeError(
                "TTS calibration seal must come from the raw 288-cell reducer"
            )
        for name, value in locals().copy().items():
            if name not in {"self", "_construction_seal"}:
                object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("only TTS calibration seal schema 2 is supported")
        for name in (
            "authority_sha256",
            "protocol_lock_sha256",
            "materialization_receipt_sha256",
            "coverage_receipt_sha256",
            "reduction_receipt_sha256",
            "raw_manifest_sha256",
            "tuning_window_sha256",
            "selected_candidate_id",
        ):
            _require_sha256(f"TTS seal {name}", getattr(self, name))
        if (
            type(self.selected_pilot_run_binding_sha256s) is not tuple
            or len(self.selected_pilot_run_binding_sha256s) != 4
            or len(set(self.selected_pilot_run_binding_sha256s)) != 4
        ):
            raise ValueError("TTS seal requires four distinct selected pilot runs")
        for digest in self.selected_pilot_run_binding_sha256s:
            _require_sha256("TTS selected pilot run binding", digest)
        if (
            self.selection_rule != "safety_first_then_maximize_slo_goodput"
            or self.result_class != "tuning_only_not_formal"
        ):
            raise ValueError("TTS seal cannot claim a formal result")
        reject_banned_model_identity(self)

    @classmethod
    def _from_reduction(
        cls,
        *,
        reduction: object,
        authority: TtsCalibrationAuthority,
        materialization: object,
        coverage: object,
    ) -> TtsCalibrationSeal:
        from lightcone_spec.experiments.stage_materialization import (
            StageCoverageReceipt,
            StageMaterializationReceipt,
        )
        from lightcone_spec.experiments.tts_calibration_authority import (
            TtsCalibrationReductionReceipt,
        )

        if type(reduction) is not TtsCalibrationReductionReceipt:
            raise TypeError("TTS seal requires an exact first-party reduction")
        if type(materialization) is not StageMaterializationReceipt:
            raise TypeError("TTS seal requires an exact materialization")
        if type(coverage) is not StageCoverageReceipt:
            raise TypeError("TTS seal requires an exact coverage receipt")
        reduction.validate_against(
            authority=authority,
            materialization=materialization,
            coverage=coverage,
        )
        return cls(
            schema_version=2,
            authority_sha256=authority.sha256,
            protocol_lock_sha256=materialization.protocol_lock_sha256,
            materialization_receipt_sha256=materialization.sha256,
            coverage_receipt_sha256=coverage.sha256,
            reduction_receipt_sha256=reduction.sha256,
            raw_manifest_sha256=reduction.raw_manifest_sha256,
            tuning_window_sha256=reduction.tuning_window_sha256,
            selected_learning_rate=reduction.selected_learning_rate,
            selected_stride=reduction.selected_stride,
            selected_candidate_id=reduction.selected_candidate_id,
            selected_pilot_run_binding_sha256s=(
                reduction.selected_pilot_run_binding_sha256s
            ),
            _construction_seal=_TTS_CALIBRATION_SEAL_CONSTRUCTION,
        )

    def validate_against(self, authority: TtsCalibrationAuthority) -> None:
        if type(authority) is not TtsCalibrationAuthority:
            raise TypeError("TTS seal requires an exact calibration authority")
        if self.authority_sha256 != authority.sha256:
            raise ValueError("TTS seal belongs to another calibration authority")
        if self.tuning_window_sha256 != authority.tuning_window_sha256:
            raise ValueError("TTS seal belongs to another tuning window")
        expected = authority.candidate_id(
            learning_rate=self.selected_learning_rate,
            stride=self.selected_stride,
        )
        if self.selected_candidate_id != expected:
            raise ValueError("TTS seal candidate identity is inconsistent")
        if (
            self.selection_rule != authority.selection_rule
            or self.result_class != "tuning_only_not_formal"
        ):
            raise ValueError("TTS seal cannot claim a formal result")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedTtsCalibrationSeal:
    payload: TtsCalibrationSeal
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        authority: TtsCalibrationAuthority,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> TtsCalibrationSeal:
        if type(self.payload) is not TtsCalibrationSeal:
            raise TypeError("signed TTS calibration seal has the wrong payload type")
        self.payload.validate_against(authority)
        verify_signed_payload(
            self.payload,
            payload_sha256=self.payload_sha256,
            challenge=self.challenge,
            attestation=self.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        return self.payload

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "payload": asdict(self.payload),
                "payload_sha256": self.payload_sha256,
                "challenge": asdict(self.challenge),
                "attestation": asdict(self.attestation),
            }
        )


@dataclass(frozen=True)
class CandidateStateReplay:
    """One controlled TTS/L0-naive mechanism replay observation."""

    method_role: Literal["TTS", "L0-naive"]
    cell_id: str
    run_id: str
    native_replay_pointer_sha256: str
    source_round: int
    source_version: int
    source_state_sha256: str
    trainable_plan_sha256: str
    candidate_bytes_sha256: str
    optimizer_state_bytes_sha256: str
    proposal_evidence_sha256: str
    publication_policy: Literal["fixed_barrier", "first_ready"]

    def __post_init__(self) -> None:
        _require_text("candidate replay run ID", self.run_id)
        _require_sha256("candidate replay cell ID", self.cell_id)
        _require_sha256(
            "candidate native replay pointer",
            self.native_replay_pointer_sha256,
        )
        if type(self.source_round) is not int or self.source_round < 1:
            raise ValueError("candidate replay source round must be positive")
        if type(self.source_version) is not int or self.source_version < 0:
            raise ValueError("candidate replay source version must be non-negative")
        for name in (
            "source_state_sha256",
            "trainable_plan_sha256",
            "candidate_bytes_sha256",
            "optimizer_state_bytes_sha256",
            "proposal_evidence_sha256",
        ):
            _require_sha256(f"candidate replay {name}", getattr(self, name))
        expected_policy = (
            "fixed_barrier" if self.method_role == "TTS" else "first_ready"
        )
        if self.publication_policy != expected_policy:
            raise ValueError("candidate replay publication policy differs from role")
        reject_banned_model_identity(self)


def assert_tts_l0_candidate_state_equality(
    tts: CandidateStateReplay,
    l0_naive: CandidateStateReplay,
) -> None:
    """Require byte equality only for a controlled equal-source replay."""

    if (
        type(tts) is not CandidateStateReplay
        or type(l0_naive) is not CandidateStateReplay
    ):
        raise TypeError("candidate equality requires exact replay observations")
    if tts.method_role != "TTS" or l0_naive.method_role != "L0-naive":
        raise ValueError("candidate equality requires TTS then L0-naive")
    if tts.run_id == l0_naive.run_id:
        raise ValueError("TTS and L0-naive must keep distinct live state identities")
    fields_to_match = (
        "source_round",
        "source_version",
        "source_state_sha256",
        "trainable_plan_sha256",
        "candidate_bytes_sha256",
        "optimizer_state_bytes_sha256",
        "proposal_evidence_sha256",
    )
    mismatches = tuple(
        name
        for name in fields_to_match
        if getattr(tts, name) != getattr(l0_naive, name)
    )
    if mismatches:
        raise ValueError(
            "TTS/L0-naive controlled replay differs in " + ",".join(mismatches)
        )


@dataclass(frozen=True)
class CandidateStateTerminalPair:
    """Terminal evidence for both publications of one source round."""

    source_round: int
    tts_cell_id: str
    l0_naive_cell_id: str
    tts_run_id: str
    l0_naive_run_id: str
    tts_native_replay_pointer_sha256: str
    l0_naive_native_replay_pointer_sha256: str
    proposal_evidence_sha256: str
    tts_terminal_receipt_sha256: str
    l0_naive_terminal_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.source_round) is not int or self.source_round < 1:
            raise ValueError("candidate terminal source round must be positive")
        for label, digest in (
            ("TTS cell", self.tts_cell_id),
            ("L0-naive cell", self.l0_naive_cell_id),
            ("proposal evidence", self.proposal_evidence_sha256),
            ("TTS replay pointer", self.tts_native_replay_pointer_sha256),
            (
                "L0-naive replay pointer",
                self.l0_naive_native_replay_pointer_sha256,
            ),
        ):
            _require_sha256(f"candidate terminal {label}", digest)
        _require_text("candidate terminal TTS run ID", self.tts_run_id)
        _require_text("candidate terminal L0-naive run ID", self.l0_naive_run_id)
        if self.tts_cell_id == self.l0_naive_cell_id:
            raise ValueError("candidate terminal requires distinct TTS/L0 cell IDs")
        if self.tts_run_id == self.l0_naive_run_id:
            raise ValueError("candidate terminal requires distinct TTS/L0 run IDs")
        _require_sha256(
            "TTS candidate terminal receipt",
            self.tts_terminal_receipt_sha256,
        )
        _require_sha256(
            "L0-naive candidate terminal receipt",
            self.l0_naive_terminal_receipt_sha256,
        )
        if self.tts_terminal_receipt_sha256 == (self.l0_naive_terminal_receipt_sha256):
            raise ValueError("TTS and L0-naive require distinct terminal receipts")


@dataclass(frozen=True)
class TtsL0CandidateStateCoverage:
    """Complete source-round evidence for one exact TTS/L0 matched pair.

    The enclosing :class:`StageCoverageReceipt` is the signed terminal
    authority.  A preflight fixture is explicitly distinguished from a formal
    materialized pair so qualification evidence cannot masquerade as stage
    coverage.
    """

    schema_version: int
    stage: str
    scope: Literal["preflight_exactness_qualification", "materialized_pair"]
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    pair_id: str
    tts_cell_id: str
    l0_naive_cell_id: str
    tts_native_replay_pointer_sha256: str
    l0_naive_native_replay_pointer_sha256: str
    qualification_cell_id: str | None
    source_round_plan_sha256: str
    trainable_plan_sha256: str
    expected_source_rounds: tuple[int, ...]
    tts_observations: tuple[CandidateStateReplay, ...]
    l0_naive_observations: tuple[CandidateStateReplay, ...]
    terminal_pairs: tuple[CandidateStateTerminalPair, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only TTS/L0 candidate coverage schema 1 is supported")
        if self.stage not in TTS_L0_CANDIDATE_STATE_COVERAGE_STAGES:
            raise ValueError("TTS/L0 candidate coverage names an unsupported stage")
        for name, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("matched pair", self.pair_id),
            ("TTS cell", self.tts_cell_id),
            ("L0-naive cell", self.l0_naive_cell_id),
            ("TTS replay pointer", self.tts_native_replay_pointer_sha256),
            (
                "L0-naive replay pointer",
                self.l0_naive_native_replay_pointer_sha256,
            ),
            ("source-round plan", self.source_round_plan_sha256),
            ("trainable plan", self.trainable_plan_sha256),
        ):
            _require_sha256(f"TTS/L0 candidate coverage {name}", digest)
        if self.tts_cell_id == self.l0_naive_cell_id:
            raise ValueError("TTS/L0 candidate coverage requires distinct cell IDs")
        if self.scope == "preflight_exactness_qualification":
            if self.stage != "preflight" or self.qualification_cell_id is None:
                raise ValueError(
                    "preflight candidate coverage requires its exact qualification cell"
                )
            _require_sha256(
                "TTS/L0 preflight qualification cell",
                self.qualification_cell_id,
            )
        elif self.scope == "materialized_pair":
            if self.stage == "preflight" or self.qualification_cell_id is not None:
                raise ValueError(
                    "materialized candidate pair cannot claim preflight qualification"
                )
        else:
            raise ValueError("TTS/L0 candidate coverage scope is unsupported")
        rounds = self.expected_source_rounds
        if (
            type(rounds) is not tuple
            or not rounds
            or rounds != tuple(range(1, len(rounds) + 1))
        ):
            raise ValueError(
                "TTS/L0 candidate coverage rounds must be the complete one-based prefix"
            )
        if (
            type(self.tts_observations) is not tuple
            or type(self.l0_naive_observations) is not tuple
            or type(self.terminal_pairs) is not tuple
            or len(self.tts_observations) != len(rounds)
            or len(self.l0_naive_observations) != len(rounds)
            or len(self.terminal_pairs) != len(rounds)
        ):
            raise ValueError("TTS/L0 candidate coverage is not source-round complete")
        if any(
            type(row) is not CandidateStateReplay
            for row in (*self.tts_observations, *self.l0_naive_observations)
        ) or any(
            type(row) is not CandidateStateTerminalPair for row in self.terminal_pairs
        ):
            raise TypeError("TTS/L0 candidate coverage requires exact typed rows")
        if tuple(row.source_round for row in self.tts_observations) != rounds:
            raise ValueError("TTS observations do not cover the exact source rounds")
        if tuple(row.source_round for row in self.l0_naive_observations) != rounds:
            raise ValueError(
                "L0-naive observations do not cover the exact source rounds"
            )
        if tuple(row.source_round for row in self.terminal_pairs) != rounds:
            raise ValueError("candidate terminals do not cover the exact source rounds")
        if any(
            row.trainable_plan_sha256 != self.trainable_plan_sha256
            for row in (*self.tts_observations, *self.l0_naive_observations)
        ):
            raise ValueError("candidate observations use another trainable plan")
        if any(row.cell_id != self.tts_cell_id for row in self.tts_observations):
            raise ValueError("TTS observations belong to another materialized cell")
        if any(
            row.cell_id != self.l0_naive_cell_id for row in self.l0_naive_observations
        ):
            raise ValueError(
                "L0-naive observations belong to another materialized cell"
            )
        if (
            len({row.run_id for row in self.tts_observations}) != 1
            or len({row.run_id for row in self.l0_naive_observations}) != 1
        ):
            raise ValueError("each candidate role must bind one exact native run")
        if any(
            row.native_replay_pointer_sha256 != self.tts_native_replay_pointer_sha256
            for row in self.tts_observations
        ) or any(
            row.native_replay_pointer_sha256
            != self.l0_naive_native_replay_pointer_sha256
            for row in self.l0_naive_observations
        ):
            raise ValueError("candidate observation uses another native replay pointer")
        if len(
            {(row.source_round, row.source_version) for row in self.tts_observations}
        ) != len(rounds) or len(
            {
                (row.source_round, row.source_version)
                for row in self.l0_naive_observations
            }
        ) != len(rounds):
            raise ValueError("candidate replay source identities are duplicated")
        if (
            len({row.tts_terminal_receipt_sha256 for row in self.terminal_pairs}) != 1
            or len(
                {row.l0_naive_terminal_receipt_sha256 for row in self.terminal_pairs}
            )
            != 1
        ):
            raise ValueError("each candidate role must bind one exact terminal")
        for tts, l0_naive, terminal in zip(
            self.tts_observations,
            self.l0_naive_observations,
            self.terminal_pairs,
            strict=True,
        ):
            assert_tts_l0_candidate_state_equality(tts, l0_naive)
            if (
                terminal.tts_cell_id != tts.cell_id
                or terminal.l0_naive_cell_id != l0_naive.cell_id
                or terminal.tts_run_id != tts.run_id
                or terminal.l0_naive_run_id != l0_naive.run_id
                or terminal.proposal_evidence_sha256 != tts.proposal_evidence_sha256
                or terminal.tts_native_replay_pointer_sha256
                != tts.native_replay_pointer_sha256
                or terminal.l0_naive_native_replay_pointer_sha256
                != l0_naive.native_replay_pointer_sha256
            ):
                raise ValueError(
                    "candidate terminal pointer differs from its exact replay rows"
                )
        reject_banned_model_identity(self)

    def validate_native_replay_pointers(self, pointers: tuple[object, ...]) -> None:
        """Deep-bind rows to reservation-independent terminal commitments.

        The complete pointer also records the replay reservation that made an
        externally controlled terminal durable.  That reservation is a later
        evidence-layer identity and cannot be included in the commitment that
        this already-signed coverage receipt names without creating a digest
        cycle.  Durable proof reopening verifies the reservation separately;
        scientific coverage therefore binds the pointer's exact semantic
        commitment.
        """

        from lightcone_spec.orchestration.native_terminal import (
            CandidateStateReplayPointer,
        )

        if type(pointers) is not tuple or any(
            type(pointer) is not CandidateStateReplayPointer for pointer in pointers
        ):
            raise TypeError("candidate coverage requires sealed replay pointers")
        by_sha = {pointer.semantic_commitment_sha256: pointer for pointer in pointers}
        if len(by_sha) != len(pointers):
            raise ValueError("candidate replay pointer set is duplicated")
        for role, expected_method, pointer_sha256, observations in (
            (
                "TTS",
                "tts",
                self.tts_native_replay_pointer_sha256,
                self.tts_observations,
            ),
            (
                "L0-naive",
                "l0",
                self.l0_naive_native_replay_pointer_sha256,
                self.l0_naive_observations,
            ),
        ):
            pointer = by_sha.get(pointer_sha256)
            if pointer is None or pointer.method != expected_method:
                raise ValueError(
                    f"{role} candidate replay pointer is missing or foreign"
                )
            if {row.run_id for row in observations} != {pointer.run_id}:
                raise ValueError(f"{role} candidate rows use another native run")
            terminal_sha256s = (
                {row.tts_terminal_receipt_sha256 for row in self.terminal_pairs}
                if role == "TTS"
                else {
                    row.l0_naive_terminal_receipt_sha256 for row in self.terminal_pairs
                }
            )
            if terminal_sha256s != {pointer.terminal_sha256}:
                raise ValueError(f"{role} candidate rows use another native terminal")
            updates = tuple(
                sorted(
                    pointer.updates,
                    key=lambda row: (row.source_round, row.source_version),
                )
            )
            rows = tuple(
                sorted(
                    observations,
                    key=lambda row: (row.source_round, row.source_version),
                )
            )
            if len(updates) != len(rows):
                raise ValueError(f"{role} candidate pointer update coverage differs")
            for update, row in zip(updates, rows, strict=True):
                if (
                    update.source_round != row.source_round
                    or update.source_version != row.source_version
                    or update.source_state_sha256 != row.source_state_sha256
                    or update.candidate_bytes_sha256 != row.candidate_bytes_sha256
                    or update.optimizer_state_bytes_sha256
                    != row.optimizer_state_bytes_sha256
                    or update.proposal_evidence_sha256 != row.proposal_evidence_sha256
                ):
                    raise ValueError(
                        f"{role} candidate row differs from first-party terminal"
                    )

    def validate_identity(
        self,
        *,
        stage: str,
        protocol_lock_sha256: str,
        materialization_receipt_sha256: str,
    ) -> None:
        if (
            self.stage != stage
            or self.protocol_lock_sha256 != protocol_lock_sha256
            or self.materialization_receipt_sha256 != materialization_receipt_sha256
        ):
            raise ValueError(
                "TTS/L0 candidate coverage differs from stage coverage identity"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class FormalRuntimeAuthorityMember:
    """Semantic identity plus raw-source commitment for one formal path."""

    member_id: str
    protocol_sha256: str
    runner_sha256: str
    test_set_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.member_id not in FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS:
            raise ValueError("formal runtime authority member is unregistered")
        for label, digest in (
            ("protocol", self.protocol_sha256),
            ("runner", self.runner_sha256),
            ("test set", self.test_set_sha256),
            ("source", self.source_sha256),
        ):
            _require_sha256(f"formal runtime {self.member_id} {label}", digest)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class FormalRuntimeAuthorityManifest:
    """ProtocolLock-bound enumeration of every mutable formal runtime surface."""

    schema_version: int
    authority_id: str
    members: tuple[FormalRuntimeAuthorityMember, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(
                "only formal runtime authority manifest schema 2 is supported"
            )
        _require_text("formal runtime authority ID", self.authority_id)
        if (
            type(self.members) is not tuple
            or any(
                type(row) is not FormalRuntimeAuthorityMember for row in self.members
            )
            or tuple(row.member_id for row in self.members)
            != FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
        ):
            raise ValueError(
                "formal runtime authority must cover every named member exactly"
            )
        for row in self.members:
            row.__post_init__()

    def member(self, member_id: str) -> FormalRuntimeAuthorityMember:
        matches = tuple(row for row in self.members if row.member_id == member_id)
        if len(matches) != 1:
            raise ValueError("formal runtime authority member is not exact")
        return matches[0]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class TrustedSingleOperatorProtocolSourceBinding:
    """Path/raw/semantic identity for one trusted ProtocolLock source."""

    absolute_path: str
    raw_sha256: str
    semantic_sha256: str
    size: int

    def __post_init__(self) -> None:
        if type(self.absolute_path) is not str:
            raise TypeError("trusted ProtocolLock source path must be exact text")
        path = Path(self.absolute_path)
        if not path.is_absolute() or path != Path(os.path.abspath(path)):
            raise ValueError(
                "trusted ProtocolLock source path must be absolute and normalized"
            )
        _require_sha256("trusted ProtocolLock source raw digest", self.raw_sha256)
        _require_sha256(
            "trusted ProtocolLock source semantic digest",
            self.semantic_sha256,
        )
        if type(self.size) is not int or not 2 <= self.size <= 256 * 1024 * 1024:
            raise ValueError("trusted ProtocolLock source size is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_path": self.absolute_path,
            "raw_sha256": self.raw_sha256,
            "semantic_sha256": self.semantic_sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrustedSingleOperatorProtocolSourceBinding:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("trusted ProtocolLock source binding fields differ")
        return cls(**value)


@dataclass(frozen=True)
class TrustedSingleOperatorProtocolSourceBindings:
    """Complete path-bound replay inputs for a trusted schema-5 lock."""

    trusted_content_bundle_source: TrustedSingleOperatorProtocolSourceBinding
    formal_runtime_authority_manifest_source: TrustedSingleOperatorProtocolSourceBinding
    tts_calibration_authority_source: TrustedSingleOperatorProtocolSourceBinding
    chronobelief_authority_source: TrustedSingleOperatorProtocolSourceBinding
    e1_recipe_anchor_authority_source: TrustedSingleOperatorProtocolSourceBinding

    def __post_init__(self) -> None:
        bindings = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(
            type(binding) is not TrustedSingleOperatorProtocolSourceBinding
            for binding in bindings
        ):
            raise TypeError("trusted ProtocolLock sources must be exact bindings")
        if len({binding.absolute_path for binding in bindings}) != len(bindings):
            raise ValueError("trusted ProtocolLock source bindings alias inputs")

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name).to_dict() for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> TrustedSingleOperatorProtocolSourceBindings:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("trusted ProtocolLock source bindings fields differ")
        return cls(
            **{
                name: TrustedSingleOperatorProtocolSourceBinding.from_dict(value[name])
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class ProtocolLock:
    """Content identity for the complete formal protocol and code release."""

    schema_version: int
    protocol_id: str
    code_git_head: str
    code_git_tree: str
    patch_manifest_sha256: str
    registry_sha256: str
    english_protocol_sha256: str
    chinese_protocol_sha256: str
    tts_calibration_authority_sha256: str
    chronobelief_authority_sha256: str
    e1_recipe_anchor_authority_sha256: str
    e2_recipe_grid_authority_sha256: str
    formal_runtime_authority_manifest_sha256: str
    offline_release_trust_root_sha256: str | None
    prepared_model_content_authorization_sha256: str | None
    formal_workload_e3a_authorization_sha256: str | None
    formal_workload_e0_authorization_sha256: str | None
    burstgpt_shape_authorization_sha256: str | None
    native_runtime_qualification_protocol_sha256: str
    native_runtime_qualification_runner_sha256: str
    native_runtime_qualification_test_set_sha256: str
    compile_qualification_protocol_sha256: str
    compile_qualification_runner_sha256: str
    compile_qualification_test_set_sha256: str
    exactness_qualification_protocol_sha256: str
    exactness_qualification_runner_sha256: str
    exactness_qualification_test_set_sha256: str
    method_roles: tuple[str, ...] = FORMAL_METHOD_ROLES
    stage_dag: tuple[str, ...] = FORMAL_STAGE_DAG
    primary_holm_family: tuple[str, ...] = PRIMARY_HOLM_FAMILY
    secondary_mechanism_contrasts: tuple[str, ...] = SECONDARY_MECHANISM_CONTRASTS
    deployment_contrast: str = DEPLOYMENT_CONTRAST
    e6_models: tuple[str, ...] = E6_MODELS
    content_source_mode: Literal["offline_root_signed", "trusted_single_operator"] = (
        field(
            default="offline_root_signed",
            metadata={"canonical_since_schema": 5},
        )
    )
    trusted_single_operator_content_bundle_sha256: str | None = field(
        default=None,
        metadata={"canonical_since_schema": 5},
    )
    trusted_single_operator_source_bindings: (
        TrustedSingleOperatorProtocolSourceBindings | None
    ) = field(
        default=None,
        metadata={"canonical_since_schema": 5},
    )

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {4, 5}:
            raise ValueError(
                "only formal ProtocolLock schema 4 or trusted schema 5 is supported"
            )
        _require_text("protocol ID", self.protocol_id)
        _require_git_oid("ProtocolLock code_git_head", self.code_git_head)
        _require_git_oid("ProtocolLock code_git_tree", self.code_git_tree)
        for name in (
            "patch_manifest_sha256",
            "registry_sha256",
            "english_protocol_sha256",
            "chinese_protocol_sha256",
            "tts_calibration_authority_sha256",
            "chronobelief_authority_sha256",
            "e1_recipe_anchor_authority_sha256",
            "e2_recipe_grid_authority_sha256",
            "formal_runtime_authority_manifest_sha256",
            "native_runtime_qualification_protocol_sha256",
            "native_runtime_qualification_runner_sha256",
            "native_runtime_qualification_test_set_sha256",
            "compile_qualification_protocol_sha256",
            "compile_qualification_runner_sha256",
            "compile_qualification_test_set_sha256",
            "exactness_qualification_protocol_sha256",
            "exactness_qualification_runner_sha256",
            "exactness_qualification_test_set_sha256",
        ):
            _require_sha256(f"ProtocolLock {name}", getattr(self, name))
        signed_content_fields = (
            "offline_release_trust_root_sha256",
            "prepared_model_content_authorization_sha256",
            "formal_workload_e3a_authorization_sha256",
            "formal_workload_e0_authorization_sha256",
            "burstgpt_shape_authorization_sha256",
        )
        if self.schema_version == 4:
            if (
                self.content_source_mode != "offline_root_signed"
                or self.trusted_single_operator_content_bundle_sha256 is not None
                or self.trusted_single_operator_source_bindings is not None
            ):
                raise ValueError("legacy ProtocolLock content source tag differs")
            for name in signed_content_fields:
                _require_sha256(f"ProtocolLock {name}", getattr(self, name))
        else:
            if (
                self.content_source_mode != "trusted_single_operator"
                or self.trusted_single_operator_content_bundle_sha256 is None
                or type(self.trusted_single_operator_source_bindings)
                is not TrustedSingleOperatorProtocolSourceBindings
            ):
                raise ValueError("trusted ProtocolLock content source tag differs")
            _require_sha256(
                "ProtocolLock trusted single-operator content bundle",
                self.trusted_single_operator_content_bundle_sha256,
            )
            assert self.trusted_single_operator_source_bindings is not None
            if (
                self.trusted_single_operator_source_bindings.trusted_content_bundle_source.semantic_sha256
                != self.trusted_single_operator_content_bundle_sha256
            ):
                raise ValueError("trusted ProtocolLock content source digest differs")
            if any(getattr(self, name) is not None for name in signed_content_fields):
                raise ValueError(
                    "trusted ProtocolLock must not carry offline authorization claims"
                )
        if self.method_roles != FORMAL_METHOD_ROLES:
            raise ValueError("ProtocolLock must contain exactly the five formal roles")
        if self.stage_dag != FORMAL_STAGE_DAG:
            raise ValueError("ProtocolLock stage DAG differs from the preregistration")
        if (
            self.primary_holm_family != PRIMARY_HOLM_FAMILY
            or self.secondary_mechanism_contrasts != SECONDARY_MECHANISM_CONTRASTS
            or self.deployment_contrast != DEPLOYMENT_CONTRAST
        ):
            raise ValueError(
                "ProtocolLock contrast family differs from preregistration"
            )
        if self.e6_models != E6_MODELS:
            raise ValueError("ProtocolLock E6 model set is not exact")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def _qualification_source_identity(self, kind: str) -> str:
        fields = {
            "native_runtime": (
                self.native_runtime_qualification_protocol_sha256,
                self.native_runtime_qualification_runner_sha256,
                self.native_runtime_qualification_test_set_sha256,
            ),
            "compile": (
                self.compile_qualification_protocol_sha256,
                self.compile_qualification_runner_sha256,
                self.compile_qualification_test_set_sha256,
            ),
            "exactness": (
                self.exactness_qualification_protocol_sha256,
                self.exactness_qualification_runner_sha256,
                self.exactness_qualification_test_set_sha256,
            ),
        }
        return content_sha256(
            {
                "schema_version": 1,
                "kind": f"{kind}_qualification_source_identity",
                "protocol_sha256": fields[kind][0],
                "runner_sha256": fields[kind][1],
                "test_set_sha256": fields[kind][2],
                "patch_manifest_sha256": self.patch_manifest_sha256,
            }
        )

    @cached_property
    def native_runtime_qualification_source_identity_sha256(self) -> str:
        return self._qualification_source_identity("native_runtime")

    @cached_property
    def native_runtime_qualification_authority_sha256(self) -> str:
        """Stable authority identity, distinct from a dynamic GPU assignment."""

        from lightcone_spec import PINNED_SGLANG_TREE
        from lightcone_spec.runtime.qualification_spec import (
            build_formal_runtime_qualification_authority_sha256,
        )
        from lightcone_spec.runtime.readiness import (
            NATIVE_RUNTIME_RELEASE_CAPABILITY,
        )

        return build_formal_runtime_qualification_authority_sha256(
            native_runtime_release_capability_sha256=(
                NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
            ),
            qualification_protocol_sha256=(
                self.native_runtime_qualification_protocol_sha256
            ),
            qualification_runner_sha256=(
                self.native_runtime_qualification_runner_sha256
            ),
            qualification_test_set_sha256=(
                self.native_runtime_qualification_test_set_sha256
            ),
            patched_sglang_tree=PINNED_SGLANG_TREE,
            patch_manifest_sha256=self.patch_manifest_sha256,
        )

    @cached_property
    def compile_qualification_source_identity_sha256(self) -> str:
        return self._qualification_source_identity("compile")

    @cached_property
    def exactness_qualification_source_identity_sha256(self) -> str:
        return self._qualification_source_identity("exactness")

    @property
    def preflight_source_authority_bindings(self) -> tuple[tuple[str, str], ...]:
        """Named source authorities every preflight assignment must carry."""

        if self.schema_version == 5:
            assert self.trusted_single_operator_content_bundle_sha256 is not None
            return tuple(
                sorted(
                    (
                        (
                            "compile_qualification",
                            self.compile_qualification_source_identity_sha256,
                        ),
                        (
                            "exactness_qualification",
                            self.exactness_qualification_source_identity_sha256,
                        ),
                        (
                            "native_runtime_qualification",
                            self.native_runtime_qualification_source_identity_sha256,
                        ),
                        (
                            "trusted_single_operator_content_bundle",
                            self.trusted_single_operator_content_bundle_sha256,
                        ),
                    )
                )
            )

        assert self.burstgpt_shape_authorization_sha256 is not None
        assert self.formal_workload_e0_authorization_sha256 is not None
        assert self.formal_workload_e3a_authorization_sha256 is not None
        assert self.offline_release_trust_root_sha256 is not None
        assert self.prepared_model_content_authorization_sha256 is not None

        return tuple(
            sorted(
                (
                    (
                        "burstgpt_shape",
                        self.burstgpt_shape_authorization_sha256,
                    ),
                    (
                        "compile_qualification",
                        self.compile_qualification_source_identity_sha256,
                    ),
                    (
                        "exactness_qualification",
                        self.exactness_qualification_source_identity_sha256,
                    ),
                    (
                        "formal_workload_e0",
                        self.formal_workload_e0_authorization_sha256,
                    ),
                    (
                        "formal_workload_e3a",
                        self.formal_workload_e3a_authorization_sha256,
                    ),
                    (
                        "native_runtime_qualification",
                        self.native_runtime_qualification_source_identity_sha256,
                    ),
                    (
                        "offline_release_trust_root",
                        self.offline_release_trust_root_sha256,
                    ),
                    (
                        "prepared_model_content",
                        self.prepared_model_content_authorization_sha256,
                    ),
                )
            )
        )


def verify_signed_payload(
    payload: object,
    *,
    payload_sha256: str,
    challenge: AttestationChallenge,
    attestation: SignedAttestation,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int | None = None,
) -> None:
    """Verify one payload under an externally pinned Ed25519 trust root."""

    reject_banned_model_identity(payload)
    observed = content_sha256(payload)
    if payload_sha256 != observed:
        raise ValueError("signed payload digest differs from its content")
    _require_sha256("expected signing policy digest", expected_policy_sha256)
    if type(policy) is not TrustedAttesterPolicy:
        raise TypeError("signed authority requires an exact trusted policy")
    if policy.sha256 != expected_policy_sha256:
        raise ValueError("signing policy differs from the externally pinned trust root")
    if challenge.subject_sha256 != payload_sha256:
        raise ValueError("signed authority challenge is not payload-bound")
    policy.verify_release(
        challenge,
        attestation,
        payload_sha256=payload_sha256,
        now_ns=now_ns,
    )


@dataclass(frozen=True)
class SignedProtocolLock:
    payload: ProtocolLock
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> ProtocolLock:
        if type(self.payload) is not ProtocolLock:
            raise TypeError("signed ProtocolLock payload has the wrong type")
        verify_signed_payload(
            self.payload,
            payload_sha256=self.payload_sha256,
            challenge=self.challenge,
            attestation=self.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        return self.payload

    @cached_property
    def sha256(self) -> str:
        from lightcone_spec.experiments.formal_registry import protocol_lock_to_dict

        return content_sha256(
            {
                "payload": protocol_lock_to_dict(self.payload),
                "payload_sha256": self.payload_sha256,
                "challenge": asdict(self.challenge),
                "attestation": asdict(self.attestation),
            }
        )
