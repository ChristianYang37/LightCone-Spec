"""Declarative industrial experiment registry and two-GPU dispatch planning.

This module contains protocol data only.  It deliberately does not launch a
server, execute a cell, reduce evidence, or mutate a selection.  The existing
runner can consume the immutable cells after their dependency receipts have
been validated.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from functools import cached_property
from typing import Any

INDUSTRIAL_EXPERIMENT_ORDER = (
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

LORA_RANKS = (1, 2, 4, 8, 16, 32, 64)
CONTEXT_GRID = (1024, 2048, 4096, 8192, 16384, 24576, 32768, 40928)
LONG_CONTEXT_ANCHORS = (4096, 16384, 32768, 40928)
CONTEXT_REGIMES = (
    "long_input_short_output",
    "short_input_long_generation",
    "multi_turn_shared_prefix",
)
DRAFT_WIDTHS = (4, 8, 16)
# E2 is downstream of the sealed E3a width decision.  Its declarative cells are
# one template per optimizer recipe, not three speculative-width cells.  The
# exact width is materialized from this selector; an integer in an E2 template
# would silently turn today's middle grid value into a scientific decision.
E2_DRAFT_WIDTH_SELECTOR = "sealed_e3a_selection.matched_width"
E3A_CONCURRENCY_GRID = (1, 2, 4, 8, 16, 32, 64)
E1_SCOPES = ("last1", "last3", "last5", "all")
E1_OPTIMIZER_ANCHORS = ("adamw", "sgdm")
E2_OPTIMIZERS = (
    "adam",
    "adamw",
    "sgdm",
    "nag",
    "muon",
    "lion",
    "chronobelief",
)
E2_SCHEDULES = ("constant", "inverse_sqrt_published_update", "cosine_to_zero")
E2_HALVING_STAGES = ((2, 4096), (4, 8192), (8, 16384), (16, 40928))
PILOT_BLOCKS = (0, 1, 2, 3)
MAXIMUM_FINAL_BLOCKS = 20
FINAL_BLOCKS = tuple(range(len(PILOT_BLOCKS), len(PILOT_BLOCKS) + MAXIMUM_FINAL_BLOCKS))
REGISTERED_CONFIRMATION_BLOCKS = PILOT_BLOCKS + FINAL_BLOCKS
# Ports are reused only across sequential waves: one per independent GPU and
# three reserved for an exclusive two-GPU server/router group.
INDUSTRIAL_PORT_SPAN = 5


def _industrial_port_span(logical_gpu_count: int) -> int:
    if logical_gpu_count < 2:
        raise ValueError(
            "the registered industrial protocol requires at least two logical GPU slots"
        )
    # One single-rank port per logical slot, followed by the registered TP2
    # server/router group's two rank ports and one coordinator port.
    return logical_gpu_count + 3


E5_CLOSED_LOOP_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64, 128, 256)
E5_OPEN_LOOP_LOAD_FACTORS = (0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.25)
E5_COHORT_COUNTS = (1, 4, 16, 64)
E5_COHORT_DISTRIBUTIONS = ("uniform", "zipf")
E5_TOPOLOGIES = ("tp1_dp1", "tp2_dp1", "two_replica_tp1_dp2")
E5_FAILURES = (
    "queue_saturation",
    "cancellation",
    "duplicate_retry",
    "nonfinite_candidate",
    "oom_candidate",
    "evidence_backpressure",
    "disk_quota",
    "slow_rank",
    "communicator_failure",
    "replica_drain",
    "replica_restart",
)
E6_CANDIDATE_MODELS = (
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B-FP8",
)
E0_MODELS = (
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Gemma4-12B",
)
E0_BACKENDS = ("EAGLE3", "DFLASH", "DSPARK")
E0_TASKS = (
    "GSM8K",
    "MATH-500",
    "AIME-2025",
    "MBPP",
    "HumanEval",
    "LiveCodeBench",
    "MT-Bench",
    "Alpaca",
    "Arena-Hard",
)
# E0 is a preregistered breadth confirmation, not a one-shot compatibility
# sweep.  Its load and paired-block axes make every eventual breadth contrast
# independently auditable.  The concrete arrival trace remains an external
# content-bound authority; neither value is selected from observed gain.
E0_LOADS = ("concurrency_one", "common_slo_load")
# The breadth universe stays compact.  These are the only E0 slots that may
# later enter a reported interaction reducer: every candidate model is named
# now, the backend/task anchors are fixed, and the block count is the
# preregistered minimum for a 95% paired interval (four excluded pilots plus
# twelve final blocks).  The remaining compatibility templates are not silently
# multiplied into a formal timing grid.
E0_INTERACTION_MODELS = E0_MODELS
E0_INTERACTION_BACKENDS = ("DFLASH",)
E0_INTERACTION_TASKS = ("GSM8K", "LiveCodeBench")
E0_INTERACTION_FINAL_BLOCKS = FINAL_BLOCKS[:12]
CORE_METHODS = ("target_only", "static", "tts", "l0")
E0_METHODS = CORE_METHODS + (
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
)


class ScientificMethodRole(str, Enum):
    """Reported method identity derived from runtime policy plus recipe authority."""

    TARGET_ONLY = "target_only"
    STATIC = "static"
    TTS_CALIBRATION_CANDIDATE = "tts_calibration_candidate"
    TTS = "tts"
    L0_NAIVE = "l0_naive"
    LC_CANDIDATE = "lc_candidate"
    LIGHTCONE_TEMPLATE = "lightcone_template"
    LIGHTCONE = "lightcone"
    ONLINESPEC_OGD = "onlinespec_ogd"
    ONLINESPEC_OPT = "onlinespec_opt"
    ONLINESPEC_ENS = "onlinespec_ens"


CONFIRMATION_METHOD_ROLES = (
    ScientificMethodRole.TARGET_ONLY.value,
    ScientificMethodRole.STATIC.value,
    ScientificMethodRole.TTS.value,
    ScientificMethodRole.L0_NAIVE.value,
    ScientificMethodRole.LIGHTCONE.value,
)
E0_METHOD_ROLES = CONFIRMATION_METHOD_ROLES + E0_METHODS[-3:]

FROZEN_TTS_RECIPE_SENTINEL = "frozen_tts_recipe"
SEALED_E2_RECIPE_SENTINEL = "sealed_e2_recipe"

_ADAPTIVE_METHODS = frozenset(E0_METHODS) - {"target_only", "static"}
_PATCH_UNSUPPORTED_ADAPTIVE_BACKENDS = frozenset(
    {"DSPARK", "EAGLE", "EAGLE3", "NEXTN", "DFLASH+DSPARK"}
)
_PATCH_UNSUPPORTED_ADAPTIVE_TOPOLOGIES = frozenset(
    {"tp2_dp1", "two_replica_tp1_dp2", "tp2_and_two_replica"}
)
_SERVING_BACKENDS = frozenset({"DFLASH", "DSPARK", "EAGLE", "EAGLE3", "NEXTN"})
_SERVING_TOPOLOGIES = frozenset({"tp1_dp1", "tp2_dp1", "two_replica_tp1_dp2"})
_UNRESOLVED_SEMANTIC_FRAGMENTS = (
    "selected",
    "transferred",
    "locked_",
    "common_load",
    "common_feasible_load",
    "matched",
    "deployment_optimal",
    "_grid",
)

PRODUCTION_SLO = (
    "ttft_short_medium_long<=2/5/10s;within_request_p99_itl<=100ms;"
    "qualified>=99%;error<=0.1%;completion>=99.9%"
)

_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}").fullmatch
_REASON_CODE = re.compile(r"[a-z0-9][a-z0-9_.-]*").fullmatch


@dataclass(frozen=True)
class FrozenTtsRecipeAuthority:
    """Primary-source-bound TTS facts without invented numeric values."""

    schema_version: int
    authority_id: str
    provenance_status: str
    status: str
    formal_eligible: bool
    paper_arxiv_id: str
    paper_url: str
    paper_pdf_sha256: str
    paper_source_sha256: str
    official_implementation_status: str
    known_semantics: tuple[str, ...]
    unresolved_fields: tuple[str, ...]
    blocker_codes: tuple[str, ...]
    historical_diagnostic_manifest_canonical_sha256: str
    historical_diagnostic_classification: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only frozen TTS recipe authority schema 1 is supported")
        for name in (
            "authority_id",
            "provenance_status",
            "status",
            "paper_arxiv_id",
            "paper_url",
            "official_implementation_status",
            "historical_diagnostic_classification",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value.strip()
                or "\n" in value
                or "\r" in value
            ):
                raise ValueError(f"{name} must be exact non-empty single-line text")
        for name in (
            "paper_pdf_sha256",
            "paper_source_sha256",
            "historical_diagnostic_manifest_canonical_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or not _LOWER_SHA256(value):
                raise ValueError(f"{name} must be a lower-case SHA-256")
        for name in ("known_semantics", "unresolved_fields", "blocker_codes"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not str for item in value):
                raise TypeError(f"{name} must be an exact text tuple")
            if value != tuple(sorted(set(value))):
                raise ValueError(f"{name} must be sorted and unique")
        if self.provenance_status != "TTS-paper-reconstruction":
            raise ValueError("the release has no official TTS code/config authority")
        if self.status != "BLOCKED" or self.formal_eligible is not False:
            raise ValueError("paper reconstruction must remain formally blocked")
        if not self.unresolved_fields or not self.blocker_codes:
            raise ValueError(
                "blocked TTS authority requires explicit unresolved fields"
            )
        if any(not _REASON_CODE(code) for code in self.blocker_codes):
            raise ValueError("TTS authority blocker codes must be stable tokens")
        if (
            self.historical_diagnostic_classification
            != "matched_recipe_publication_policy_diagnostic_not_tts_reproduction"
        ):
            raise ValueError("historical shared-recipe evidence is diagnostic only")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, Any]:
        return _canonical(self)


LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY = FrozenTtsRecipeAuthority(
    schema_version=1,
    authority_id="tts-paper-reconstruction-v1",
    provenance_status="TTS-paper-reconstruction",
    status="BLOCKED",
    formal_eligible=False,
    paper_arxiv_id="2605.09329v2",
    paper_url="https://arxiv.org/abs/2605.09329v2",
    paper_pdf_sha256="7688b05bab7696f4a47a5987f2fcad13d46f1d84cec9f90caf661fb397f3ee20",
    paper_source_sha256="22c549c0297fc0a2a71af002c3721f71ddfd06d86bc46b2f41592bd6748afe59",
    official_implementation_status="not_found_as_of_2026-08-15",
    known_semantics=tuple(
        sorted(
            (
                "adam_optimizer",
                "exactly_one_optimization_step_per_update",
                "fixed_update_stride",
                "latest_round_only_supervision",
                "position_weighted_distillation_plus_source_point_proximal_loss",
                "request_local_state_reset",
                "strided_side_cuda_stream_execution",
            )
        )
    ),
    unresolved_fields=tuple(
        sorted(
            (
                "adam_beta1",
                "adam_beta2",
                "adam_epsilon",
                "gradient_clip",
                "learning_rate",
                "learning_rate_schedule",
                "loss_normalization",
                "loss_precision",
                "loss_temperature",
                "official_implementation_commit",
                "optimizer_state_reset_semantics",
                "position_weight_values",
                "proximal_lambda",
                "trainable_parameter_manifest",
                "update_stride_selection_rule",
                "weight_decay",
            )
        )
    ),
    blocker_codes=("tts_official_recipe_unavailable",),
    historical_diagnostic_manifest_canonical_sha256=(
        "e21748a923d5c16206164a6b007ec9365a7654267b1afef7480387d6c7e4d09d"
    ),
    historical_diagnostic_classification=(
        "matched_recipe_publication_policy_diagnostic_not_tts_reproduction"
    ),
)


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return _canonical(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical content cannot contain a non-finite float")
        return 0.0 if value == 0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def content_sha256(value: Any) -> str:
    """Return a stable SHA-256 over canonical JSON content."""

    body = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{name} must be non-empty single-line text")


def _require_exact_text(name: str, value: object) -> None:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be exact non-empty single-line text")


class CellStatus(str, Enum):
    """Truthful source-registry state; no status implies a measured value."""

    UNMEASURED = "UNMEASURED"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "N/A"


class WorkloadClass(str, Enum):
    """Resource-isolation class used by the pure dispatch planner."""

    TUNING = "tuning"
    CORRECTNESS = "correctness"
    HEADLINE = "headline"
    PROFILE = "profile"
    DOWNLOAD = "download"
    COMPILE = "compile"


_EXCLUSIVE_WORKLOADS = {
    WorkloadClass.PROFILE,
    WorkloadClass.DOWNLOAD,
    WorkloadClass.COMPILE,
}


@dataclass(frozen=True)
class AxisSpec:
    name: str
    values: tuple[str | int | float, ...]

    def __post_init__(self) -> None:
        _require_text("axis name", self.name)
        if not self.values:
            raise ValueError("an experiment axis cannot be empty")
        identities = [content_sha256(value) for value in self.values]
        if len(identities) != len(set(identities)):
            raise ValueError(f"axis {self.name!r} contains duplicate values")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    dependencies: tuple[str, ...]
    locked_outputs: tuple[str, ...]
    axes: tuple[AxisSpec, ...]

    def __post_init__(self) -> None:
        _require_text("experiment name", self.name)
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("experiment dependencies must be unique")
        if not self.locked_outputs or len(self.locked_outputs) != len(
            set(self.locked_outputs)
        ):
            raise ValueError("locked outputs must be non-empty and unique")
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("experiment axis names must be unique")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class CellIdentity:
    """Every registered field that can change a scientific cell identity."""

    experiment: str
    model: str
    backend: str
    task: str
    method: str
    scope: str | None
    rank: int | None
    alpha_over_rank: float | None
    optimizer: str | None
    learning_rate: float | None
    schedule: str | None
    context: int | None
    regime: str
    width: int | None
    arrival: str
    slo: str
    cohort: str
    topology: str
    seed: int
    block: int
    gpu_uuids: tuple[str, ...]
    parameterization: str = "none"
    variant: str = "default"
    concurrency: int | None = None
    load_factor: float | None = None
    cohort_count: int = 1

    def __post_init__(self) -> None:
        for name in (
            "experiment",
            "model",
            "backend",
            "task",
            "method",
            "regime",
            "arrival",
            "slo",
            "cohort",
            "topology",
            "parameterization",
            "variant",
        ):
            _require_text(name, getattr(self, name))
        if self.scope is not None:
            _require_text("scope", self.scope)
        if self.optimizer is not None:
            _require_text("optimizer", self.optimizer)
        if self.schedule is not None:
            _require_text("schedule", self.schedule)
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be positive")
        if self.alpha_over_rank is not None and (
            not math.isfinite(self.alpha_over_rank) or self.alpha_over_rank <= 0
        ):
            raise ValueError("alpha_over_rank must be finite and positive")
        if self.parameterization == "lora":
            if self.rank is None or self.alpha_over_rank is None:
                raise ValueError("LoRA identities require rank and alpha_over_rank")
        elif self.rank is not None or self.alpha_over_rank is not None:
            raise ValueError("rank fields are valid only for LoRA identities")
        if self.learning_rate is not None and (
            not math.isfinite(self.learning_rate) or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if self.context is not None and self.context < 1:
            raise ValueError("context must be positive")
        if self.width is not None and self.width < 1:
            raise ValueError("width must be positive")
        if self.seed < 0 or self.block < 0:
            raise ValueError("seed and block must be non-negative")
        if self.concurrency is not None and self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.load_factor is not None and (
            not math.isfinite(self.load_factor) or self.load_factor <= 0
        ):
            raise ValueError("load_factor must be finite and positive")
        if self.cohort_count < 1:
            raise ValueError("cohort_count must be positive")
        if not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("a cell must identify one or more distinct GPU slots")
        for gpu_uuid in self.gpu_uuids:
            _require_text("GPU UUID", gpu_uuid)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ResourceClaim:
    gpu_uuids: tuple[str, ...]
    ports: tuple[int, ...]
    cache_root: str
    evidence_root: str
    workload_class: WorkloadClass

    def __post_init__(self) -> None:
        if not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("a resource claim must reserve one or more GPUs")
        if not self.ports or len(self.ports) != len(set(self.ports)):
            raise ValueError("resource ports must be non-empty and unique")
        if any(port < 1024 or port > 65535 for port in self.ports):
            raise ValueError("resource ports must be in [1024, 65535]")
        _require_text("cache_root", self.cache_root)
        _require_text("evidence_root", self.evidence_root)
        if not isinstance(self.workload_class, WorkloadClass):
            raise TypeError("workload_class must be a WorkloadClass")

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_uuids)

    @property
    def exclusive(self) -> bool:
        return self.gpu_count > 1 or self.workload_class in _EXCLUSIVE_WORKLOADS

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ExperimentCell:
    identity: CellIdentity
    resources: ResourceClaim
    status: CellStatus
    reason_code: str
    reason: str

    def __post_init__(self) -> None:
        if self.identity.gpu_uuids != self.resources.gpu_uuids:
            raise ValueError("cell identity and resource GPU assignments differ")
        if not isinstance(self.status, CellStatus):
            raise TypeError("status must be a CellStatus")
        if not isinstance(self.reason_code, str) or not _REASON_CODE(self.reason_code):
            raise ValueError("reason_code must be a stable lower-case token")
        _require_text("reason", self.reason)

    @cached_property
    def cell_id(self) -> str:
        return self.identity.sha256

    @cached_property
    def sha256(self) -> str:
        """Digest the declaration, including resources and truthful status."""

        return content_sha256(self)

    @property
    def runnable(self) -> bool:
        return self.status is CellStatus.UNMEASURED

    def with_status(
        self, status: CellStatus, *, reason_code: str, reason: str
    ) -> ExperimentCell:
        """Return a new cell while preserving the old reason in the old object."""

        return replace(
            self,
            status=status,
            reason_code=reason_code,
            reason=reason,
        )


def _has_recipe_sentinel(identity: CellIdentity, sentinel: str) -> bool:
    return identity.optimizer == sentinel and identity.schedule == sentinel


def _require_zero_adaptation_identity(identity: CellIdentity, *, label: str) -> None:
    if (
        identity.scope not in {None, "none"}
        or identity.parameterization != "none"
        or any(
            value is not None
            for value in (
                identity.optimizer,
                identity.learning_rate,
                identity.schedule,
                identity.rank,
                identity.alpha_over_rank,
            )
        )
    ):
        raise ValueError(f"{label} cannot carry adaptation recipe state")


def _require_frozen_tts_recipe_identity(identity: CellIdentity) -> None:
    if (
        identity.scope != FROZEN_TTS_RECIPE_SENTINEL
        or identity.optimizer != FROZEN_TTS_RECIPE_SENTINEL
        or identity.schedule != FROZEN_TTS_RECIPE_SENTINEL
        or identity.parameterization != FROZEN_TTS_RECIPE_SENTINEL
        or identity.learning_rate is not None
        or identity.rank is not None
        or identity.alpha_over_rank is not None
    ):
        raise ValueError("frozen TTS recipe identity is not exact")


def _require_sealed_e2_recipe_identity(identity: CellIdentity) -> None:
    if (
        identity.scope != SEALED_E2_RECIPE_SENTINEL
        or identity.optimizer != SEALED_E2_RECIPE_SENTINEL
        or identity.schedule != SEALED_E2_RECIPE_SENTINEL
        or identity.parameterization != SEALED_E2_RECIPE_SENTINEL
        or identity.learning_rate is not None
        or identity.rank is not None
        or identity.alpha_over_rank is not None
    ):
        raise ValueError("sealed E2 recipe identity is not exact")


def _require_e1a_candidate_identity(identity: CellIdentity) -> None:
    configurations = {
        (row.scope, row.parameterization, row.rank, row.alpha_over_rank)
        for row in e1a_adaptive_configurations()
    }
    if (
        identity.experiment != "E1a"
        or identity.backend != "DSPARK"
        or identity.optimizer != SEALED_E2_RECIPE_SENTINEL
        or identity.schedule != SEALED_E2_RECIPE_SENTINEL
        or identity.learning_rate is not None
        or (
            identity.scope,
            identity.parameterization,
            identity.rank,
            identity.alpha_over_rank,
        )
        not in configurations
        or not identity.variant.startswith("sealed_lightcone_recipe:native_heads:")
    ):
        raise ValueError("E1a candidate identity is not preregistered")


def _derived_scientific_method_role(cell: ExperimentCell) -> ScientificMethodRole:
    """Derive a role from one cell after registry ownership is established."""

    identity = cell.identity
    if identity.experiment == "TTS-Cal":
        try:
            stride = int(identity.variant.removeprefix("tts_calibration:stride="))
        except ValueError as error:
            raise ValueError("TTS-Cal stride identity is not exact") from error
        from lightcone_spec.experiments.formal_protocol import (
            TTS_LEARNING_RATES,
            TTS_STRIDES,
        )

        if (
            identity.method != "tts"
            or identity.backend != "DFLASH"
            or identity.scope != "full_drafter"
            or identity.parameterization != "full"
            or identity.optimizer != "adam"
            or identity.schedule != "constant"
            or identity.learning_rate not in TTS_LEARNING_RATES
            or stride not in TTS_STRIDES
            or identity.rank is not None
            or identity.alpha_over_rank is not None
        ):
            raise ValueError("TTS-Cal candidate identity is not preregistered")
        return ScientificMethodRole.TTS_CALIBRATION_CANDIDATE
    if identity.method == "target_only":
        _require_zero_adaptation_identity(identity, label="Target-only")
        return ScientificMethodRole.TARGET_ONLY
    if identity.method == "static":
        _require_zero_adaptation_identity(identity, label="Static")
        return ScientificMethodRole.STATIC
    if identity.method == "tts":
        _require_frozen_tts_recipe_identity(identity)
        return ScientificMethodRole.TTS
    if identity.method == "l0":
        if _has_recipe_sentinel(identity, FROZEN_TTS_RECIPE_SENTINEL):
            _require_frozen_tts_recipe_identity(identity)
            return ScientificMethodRole.L0_NAIVE
        if _has_recipe_sentinel(identity, SEALED_E2_RECIPE_SENTINEL):
            if identity.experiment == "E1a":
                _require_e1a_candidate_identity(identity)
                return ScientificMethodRole.LC_CANDIDATE
            _require_sealed_e2_recipe_identity(identity)
            return ScientificMethodRole.LIGHTCONE_TEMPLATE
        if identity.experiment in {"E1", "E2"}:
            return ScientificMethodRole.LC_CANDIDATE
        raise ValueError("L0 identity lacks a registered recipe authority")
    if identity.method in E0_METHODS[-3:]:
        return ScientificMethodRole(identity.method)
    raise ValueError("cell method has no registered scientific role")


def unresolved_semantic_placeholder(cell: ExperimentCell) -> str | None:
    """Return the first unresolved execution value without choosing a replacement."""

    identity = cell.identity
    for value in (
        identity.scope,
        identity.optimizer,
        identity.schedule,
        identity.parameterization,
        identity.arrival,
    ):
        if value is not None and any(
            fragment in value.lower() for fragment in _UNRESOLVED_SEMANTIC_FRAGMENTS
        ):
            return value
    return None


def serving_cell_rejection_reason(cell: ExperimentCell) -> str | None:
    """Mirror the serving renderer's allocation-free cell preflight."""

    identity = cell.identity
    if not cell.runnable:
        return "only UNMEASURED registry cells are runnable"
    if identity.experiment == "preflight":
        return "preflight cells use their dedicated non-serving executor"
    if cell.resources.workload_class in {
        WorkloadClass.DOWNLOAD,
        WorkloadClass.COMPILE,
    }:
        return "DOWNLOAD/COMPILE cells are not serving runtime cells"
    if identity.method not in E0_METHODS:
        return "cell method is outside the canonical method set"
    placeholder = unresolved_semantic_placeholder(cell)
    if placeholder is not None:
        return f"cell contains unresolved semantic placeholder {placeholder!r}"
    if identity.context is None:
        return "serving cells require an exact context"
    if identity.concurrency is None:
        return "serving cells require an exact concurrency/admission cap"
    if identity.method != "target_only" and identity.width is None:
        return "speculative serving cells require an exact draft width"
    if identity.backend not in _SERVING_BACKENDS | {"NONE"}:
        return "cell backend is not one native serving backend"
    if identity.method != "target_only" and identity.backend == "NONE":
        return "speculative methods cannot use backend=NONE"
    if identity.topology not in _SERVING_TOPOLOGIES:
        return "cell topology has no exact serving runtime mapping"
    return None


@dataclass(frozen=True)
class LockedOutput:
    name: str
    content_sha256: str

    def __post_init__(self) -> None:
        _require_text("locked output name", self.name)
        if not isinstance(self.content_sha256, str) or not _LOWER_SHA256(
            self.content_sha256
        ):
            raise ValueError("locked output content_sha256 must be lower-case SHA-256")


@dataclass(frozen=True)
class ExperimentReceipt:
    experiment: str
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    completed_cells_sha256: str
    dependency_receipts: tuple[LockedOutput, ...]
    outputs: tuple[LockedOutput, ...]
    selection_state: str = "sealed_before_downstream_unblinding"

    def __post_init__(self) -> None:
        _require_text("receipt experiment", self.experiment)
        if not isinstance(self.registry_sha256, str) or not _LOWER_SHA256(
            self.registry_sha256
        ):
            raise ValueError("receipt registry_sha256 must be lower-case SHA-256")
        for name, digest in (
            ("runtime_sha256", self.runtime_sha256),
            ("split_sha256", self.split_sha256),
            ("completed_cells_sha256", self.completed_cells_sha256),
        ):
            if not isinstance(digest, str) or not _LOWER_SHA256(digest):
                raise ValueError(f"receipt {name} must be lower-case SHA-256")
        if len({row.name for row in self.dependency_receipts}) != len(
            self.dependency_receipts
        ):
            raise ValueError("receipt dependency identities must be unique")
        if not self.outputs or len({output.name for output in self.outputs}) != len(
            self.outputs
        ):
            raise ValueError("receipt outputs must be non-empty and unique")
        if self.selection_state != "sealed_before_downstream_unblinding":
            raise ValueError("selection outputs must be sealed before downstream use")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "registry_sha256": self.registry_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "completed_cells_sha256": self.completed_cells_sha256,
            "dependency_receipts": [
                _canonical(row)
                for row in sorted(self.dependency_receipts, key=lambda row: row.name)
            ],
            "outputs": [
                _canonical(output)
                for output in sorted(self.outputs, key=lambda row: row.name)
            ],
            "selection_state": self.selection_state,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class StageActivationPlan:
    """Immutable reducer output selecting one dependency-bound execution wave."""

    registry_sha256: str
    experiment: str
    dependency_receipt_sha256: str
    runtime_sha256: str
    split_sha256: str
    source_selection_sha256: str
    activation_round: str
    status: str
    activated_cell_ids: tuple[str, ...]
    not_applicable_cell_ids: tuple[str, ...]
    blocked_cell_ids: tuple[str, ...]
    deferred_cell_ids: tuple[str, ...]
    reason_code: str

    def __post_init__(self) -> None:
        _require_text("activation experiment", self.experiment)
        _require_text("activation round", self.activation_round)
        for name in (
            "registry_sha256",
            "dependency_receipt_sha256",
            "runtime_sha256",
            "split_sha256",
            "source_selection_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _LOWER_SHA256(value):
                raise ValueError(f"{name} must be lower-case SHA-256")
        partitions = (
            self.activated_cell_ids,
            self.not_applicable_cell_ids,
            self.blocked_cell_ids,
            self.deferred_cell_ids,
        )
        flattened = tuple(cell_id for partition in partitions for cell_id in partition)
        if any(
            not isinstance(cell_id, str) or not _LOWER_SHA256(cell_id)
            for cell_id in flattened
        ):
            raise ValueError("activation dispositions require lower-case cell SHA-256")
        if len(flattened) != len(set(flattened)):
            raise ValueError("activation dispositions must be disjoint")
        if self.status == "AVAILABLE":
            if not self.activated_cell_ids:
                raise ValueError("AVAILABLE activation plans require runnable cells")
        elif self.status == "BLOCKED":
            if self.activated_cell_ids:
                raise ValueError("BLOCKED activation plans cannot activate cells")
        else:
            raise ValueError("activation status must be AVAILABLE or BLOCKED")
        if not isinstance(self.reason_code, str) or not _REASON_CODE(self.reason_code):
            raise ValueError("activation reason_code is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ParameterConfiguration:
    scope: str
    parameterization: str
    rank: int | None
    alpha_over_rank: float | None
    native_head_policy: str

    def __post_init__(self) -> None:
        _require_text("scope", self.scope)
        _require_text("parameterization", self.parameterization)
        _require_text("native_head_policy", self.native_head_policy)
        if self.parameterization == "lora":
            if self.rank not in LORA_RANKS or self.alpha_over_rank != 1.0:
                raise ValueError(
                    "registered LoRA configurations require rank grid and alpha/r=1"
                )
        elif self.parameterization == "full":
            if self.rank is not None or self.alpha_over_rank is not None:
                raise ValueError("Full configurations cannot carry LoRA fields")
        else:
            raise ValueError("parameterization must be full or lora")

    @property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class AdaptationRecipeLookupKey:
    """Scientific fields that select one source-owned adaptation recipe.

    L0-policy LightCone candidates have E1/E2 search keys. Frozen TTS and
    L0-naive anchors use one model/backend-compatible singleton key that binds
    the primary-source authority without inheriting stage, load, or geometry.
    E1 search keys carry an exact width; E2 search keys carry the sealed E3a
    selector slot.
    """

    authority_kind: str
    model: str
    experiment: str
    backend: str
    scope: str
    parameterization: str
    rank: int | None
    alpha_over_rank: float | None
    optimizer: str
    learning_rate: float | None
    schedule: str
    cohort: str
    draft_width: int | None
    draft_width_selector: str | None

    def __post_init__(self) -> None:
        for name in (
            "authority_kind",
            "model",
            "experiment",
            "backend",
            "scope",
            "parameterization",
            "optimizer",
            "schedule",
            "cohort",
        ):
            _require_exact_text(f"adaptation recipe {name}", getattr(self, name))
        for name in ("rank", "draft_width"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise TypeError(f"adaptation recipe {name} must be an exact integer")
        if self.alpha_over_rank is not None and type(self.alpha_over_rank) is not float:
            raise TypeError("adaptation recipe alpha_over_rank must be an exact float")
        if self.learning_rate is not None and type(self.learning_rate) is not float:
            raise TypeError("adaptation recipe learning_rate must be an exact float")
        if self.draft_width_selector is not None:
            _require_exact_text(
                "adaptation recipe draft-width selector", self.draft_width_selector
            )
        if self.authority_kind == "frozen_tts":
            if (
                self.experiment != "frozen_tts"
                or self.backend not in _SERVING_BACKENDS
                or self.scope != FROZEN_TTS_RECIPE_SENTINEL
                or self.parameterization != FROZEN_TTS_RECIPE_SENTINEL
                or self.rank is not None
                or self.alpha_over_rank is not None
                or self.optimizer != FROZEN_TTS_RECIPE_SENTINEL
                or self.learning_rate is not None
                or self.schedule != FROZEN_TTS_RECIPE_SENTINEL
                or self.cohort != FROZEN_TTS_RECIPE_SENTINEL
                or self.draft_width is not None
                or self.draft_width_selector != FROZEN_TTS_RECIPE_SENTINEL
            ):
                raise ValueError("frozen TTS recipe lookup key is not canonical")
            return
        if self.authority_kind != "lc_candidate":
            raise ValueError("adaptation recipe authority kind is unsupported")
        if self.experiment not in {"E1", "E2"}:
            raise ValueError("adaptation recipes are registered only for E1/E2")
        if self.backend != "DFLASH":
            raise ValueError("E1/E2 adaptation recipes require DFLASH")
        if self.optimizer not in E2_OPTIMIZERS or self.schedule not in E2_SCHEDULES:
            raise ValueError("LC candidate recipe lies outside its registered grid")
        if self.parameterization == "full":
            if self.rank is not None or self.alpha_over_rank is not None:
                raise ValueError("Full recipe keys cannot carry LoRA fields")
        elif self.parameterization == "lora":
            if self.rank not in LORA_RANKS or self.alpha_over_rank != 1.0:
                raise ValueError("LoRA recipe keys require the registered rank grid")
        else:
            raise ValueError("recipe parameterization must be full or lora")
        if self.learning_rate is not None and (
            not math.isfinite(self.learning_rate) or self.learning_rate <= 0
        ):
            raise ValueError("recipe learning rate must be finite and positive")
        exact_width = self.draft_width is not None
        selected_width = self.draft_width_selector is not None
        if exact_width == selected_width:
            raise ValueError("recipe key requires exactly one draft-width authority")
        if exact_width:
            if self.experiment != "E1" or self.draft_width not in DRAFT_WIDTHS:
                raise ValueError("only E1 recipe keys may carry an exact grid width")
        elif (
            self.experiment != "E2"
            or self.draft_width_selector != E2_DRAFT_WIDTH_SELECTOR
        ):
            raise ValueError("E2 recipe keys require the sealed E3a width selector")

    @classmethod
    def from_cell(cls, cell: ExperimentCell) -> AdaptationRecipeLookupKey:
        identity = cell.identity
        if _has_recipe_sentinel(identity, FROZEN_TTS_RECIPE_SENTINEL):
            if identity.method not in {"tts", "l0"}:
                raise ValueError("frozen TTS recipe requires TTS or L0-naive")
            _require_frozen_tts_recipe_identity(identity)
            return cls(
                authority_kind="frozen_tts",
                model=identity.model,
                experiment="frozen_tts",
                backend=identity.backend,
                scope=FROZEN_TTS_RECIPE_SENTINEL,
                parameterization=FROZEN_TTS_RECIPE_SENTINEL,
                rank=None,
                alpha_over_rank=None,
                optimizer=FROZEN_TTS_RECIPE_SENTINEL,
                learning_rate=None,
                schedule=FROZEN_TTS_RECIPE_SENTINEL,
                cohort=FROZEN_TTS_RECIPE_SENTINEL,
                draft_width=None,
                draft_width_selector=FROZEN_TTS_RECIPE_SENTINEL,
            )
        if identity.experiment not in {"E1", "E2"} or identity.method != "l0":
            raise ValueError("adaptation recipe lookup requires an E1/E2 LC candidate")
        if (
            identity.scope is None
            or identity.optimizer is None
            or identity.schedule is None
        ):
            raise ValueError("adaptation recipe cell contains unresolved key fields")
        if identity.experiment == "E1":
            if identity.width is None:
                raise ValueError("E1 adaptation recipes require an exact width")
            selector = None
        else:
            if identity.width is not None:
                raise ValueError(
                    "E2 registry cells are width templates, not fixed-width cells"
                )
            selector = E2_DRAFT_WIDTH_SELECTOR
        return cls(
            authority_kind="lc_candidate",
            model=identity.model,
            experiment=identity.experiment,
            backend=identity.backend,
            scope=identity.scope,
            parameterization=identity.parameterization,
            rank=identity.rank,
            alpha_over_rank=identity.alpha_over_rank,
            optimizer=identity.optimizer,
            learning_rate=identity.learning_rate,
            schedule=identity.schedule,
            cohort=identity.cohort,
            draft_width=identity.width,
            draft_width_selector=selector,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


_OPTIMIZER_RECIPE_FIELDS = frozenset(
    {
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "grad_clip",
        "momentum",
        "muon_ns_steps",
        "muon_auxiliary_learning_rate",
        "muon_auxiliary_weight_decay",
        "schedule",
        "schedule_total_published_updates",
    }
)

_E2_RECIPE_BLOCKER_FIELD_BY_CODE = {
    "e2_beta1_unregistered": "optimizer.beta1",
    "e2_beta2_unregistered": "optimizer.beta2",
    "e2_cosine_horizon_unregistered": ("optimizer.schedule_total_published_updates"),
    "e2_epsilon_unregistered": "optimizer.epsilon",
    "e2_extra_logical_delay_unregistered": "extra_logical_delay",
    "e2_fixed_verification_budget_unregistered": "fixed_verification_budget",
    "e2_grad_clip_unregistered": "optimizer.grad_clip",
    "e2_learning_rate_unregistered": "optimizer.learning_rate",
    "e2_momentum_unregistered": "optimizer.momentum",
    "e2_muon_auxiliary_learning_rate_unregistered": (
        "optimizer.muon_auxiliary_learning_rate"
    ),
    "e2_muon_auxiliary_weight_decay_unregistered": (
        "optimizer.muon_auxiliary_weight_decay"
    ),
    "e2_muon_ns_steps_unregistered": "optimizer.muon_ns_steps",
    "e2_teacher_row_policy_unregistered": "teacher_row_policy",
    "e2_update_stride_unregistered": "stride",
    "e2_verification_mode_unregistered": "verification_mode",
    "e2_weight_decay_unregistered": "optimizer.weight_decay",
}


@dataclass(frozen=True)
class AdaptationRecipeBlocker:
    """One missing source-owned value in a blocked recipe declaration."""

    field: str
    reason_code: str

    def __post_init__(self) -> None:
        _require_exact_text("adaptation recipe blocker field", self.field)
        if type(self.reason_code) is not str or not _REASON_CODE(self.reason_code):
            raise ValueError("adaptation recipe blocker reason code is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class OptimizerRecipeDeclaration:
    """Every OptimizerConfig field, without an implicit schema default."""

    name: str
    learning_rate: float | None
    weight_decay: float | None
    beta1: float | None
    beta2: float | None
    epsilon: float | None
    grad_clip: float | None
    momentum: float | None
    muon_ns_steps: int | None
    muon_auxiliary_learning_rate: float | None
    muon_auxiliary_weight_decay: float | None
    schedule: str | None
    schedule_total_published_updates: int | None
    unresolved_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_exact_text("optimizer recipe name", self.name)
        if self.schedule is not None:
            _require_exact_text("optimizer recipe schedule", self.schedule)
        if type(self.unresolved_fields) is not tuple or any(
            type(value) is not str for value in self.unresolved_fields
        ):
            raise TypeError("optimizer unresolved fields must be exact text tuples")
        if self.unresolved_fields != tuple(sorted(set(self.unresolved_fields))):
            raise ValueError("optimizer unresolved fields must be sorted and unique")
        if not set(self.unresolved_fields) <= _OPTIMIZER_RECIPE_FIELDS:
            raise ValueError("optimizer recipe names an unknown unresolved field")
        for name in (
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "grad_clip",
            "momentum",
            "muon_auxiliary_learning_rate",
            "muon_auxiliary_weight_decay",
        ):
            value = getattr(self, name)
            if value is not None:
                if type(value) is not float:
                    raise TypeError(f"optimizer recipe {name} must be an exact float")
                if not math.isfinite(value):
                    raise ValueError(f"optimizer recipe {name} must be finite")
        for name in ("muon_ns_steps", "schedule_total_published_updates"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise TypeError(f"optimizer recipe {name} must be an exact integer")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("optimizer recipe learning rate must be positive")
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError("optimizer recipe weight decay must be non-negative")
        if self.beta1 is not None and not 0 < self.beta1 < 1:
            raise ValueError("optimizer recipe beta1 must be in (0, 1)")
        if self.beta2 is not None and not 0 < self.beta2 < 1:
            raise ValueError("optimizer recipe beta2 must be in (0, 1)")
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError("optimizer recipe epsilon must be positive")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("optimizer recipe grad_clip must be positive")
        if self.momentum is not None and not 0 < self.momentum < 1:
            raise ValueError("optimizer recipe momentum must be in (0, 1)")
        if self.muon_ns_steps is not None and not 1 <= self.muon_ns_steps <= 20:
            raise ValueError("optimizer recipe Muon steps must be in [1, 20]")
        if (
            self.schedule_total_published_updates is not None
            and self.schedule_total_published_updates < 2
        ):
            raise ValueError("cosine horizon must cover at least two publications")
        for name in self.unresolved_fields:
            if getattr(self, name) is not None:
                raise ValueError("an unresolved optimizer field must remain null")

    def to_optimizer_config(self):
        """Build only a completely declared config, passing every field."""

        if self.unresolved_fields:
            raise ValueError(
                "optimizer recipe is BLOCKED by unresolved fields: "
                + ",".join(self.unresolved_fields)
            )
        if None in (
            self.learning_rate,
            self.weight_decay,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.grad_clip,
            self.schedule,
        ):
            raise ValueError("optimizer recipe is not fully declared")
        from lightcone_spec.config.schema import OptimizerConfig

        return OptimizerConfig(
            name=self.name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            grad_clip=self.grad_clip,
            momentum=self.momentum,
            muon_ns_steps=self.muon_ns_steps,
            muon_auxiliary_learning_rate=self.muon_auxiliary_learning_rate,
            muon_auxiliary_weight_decay=self.muon_auxiliary_weight_decay,
            schedule=self.schedule,
            schedule_total_published_updates=(self.schedule_total_published_updates),
        )


_ADAPTATION_RECIPE_FIELDS = frozenset(
    {
        "adaptation_group_id",
        "adaptation_scope",
        "stride",
        "canvas_tokens",
        "extra_logical_delay",
        "lora_matrix_policy",
        "loss_position_decay",
        "max_in_flight",
        "native_head_policy",
        "parameter_scope",
        "teacher_row_policy",
        "verification_mode",
        "fixed_verification_budget",
        "weight_update_mode",
    }
)

_FROZEN_TTS_DECLARATION_UNRESOLVED_FIELDS = frozenset(
    {
        "adaptation_group_id",
        "adaptation_scope",
        "canvas_tokens",
        "extra_logical_delay",
        "fixed_verification_budget",
        "lora_matrix_policy",
        "loss_position_decay",
        "max_in_flight",
        "native_head_policy",
        "parameter_scope",
        "stride",
        "verification_mode",
        "weight_update_mode",
    }
)

_E2_REQUIRED_ADAPTATION_UNRESOLVED_FIELDS = frozenset(
    {
        "stride",
        "extra_logical_delay",
        "teacher_row_policy",
        "verification_mode",
        "fixed_verification_budget",
    }
)


def _e2_required_optimizer_unresolved_fields(
    *, optimizer: str, schedule: str
) -> frozenset[str]:
    fields = {
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "grad_clip",
    }
    if optimizer in {"sgdm", "nag", "muon"}:
        fields.add("momentum")
    if optimizer == "muon":
        fields.update(
            {
                "muon_ns_steps",
                "muon_auxiliary_learning_rate",
                "muon_auxiliary_weight_decay",
            }
        )
    if schedule == "cosine_to_zero":
        fields.add("schedule_total_published_updates")
    return frozenset(fields)


@dataclass(frozen=True)
class AdaptationRecipeDeclaration:
    """Registry-owned, content-bound declaration of update-side semantics.

    ``extra_logical_delay`` is a source-readiness latency bound before applying
    the method's publication policy; it is not a second publication policy.
    """

    schema_version: int
    lookup_key: AdaptationRecipeLookupKey
    source_authority: str
    source_authority_sha256: str
    weight_update_mode: str | None
    parameter_scope: str | None
    kv_history_policy: str
    adaptation_scope: str | None
    adaptation_group_id: str | None
    optimizer: OptimizerRecipeDeclaration
    rank: int | None
    lora_alpha: int | None
    lora_matrix_policy: str | None
    native_head_policy: str | None
    stride: int | None
    max_in_flight: int | None
    canvas_tokens: int | None
    loss_position_decay: float | None
    extra_logical_delay: int | None
    teacher_row_policy: str | None
    verification_mode: str | None
    fixed_verification_budget: int | None
    confidence_loss_weight: float | None
    chronobelief_release_capability_sha256: str | None
    chronobelief_gpu_proof_sha256: str | None
    eagle3_e0_execution_authority_sha256: str | None
    eagle3_compatibility_authority_sha256: str | None
    eagle3_model_selector_sha256: str | None
    eagle3_native_gpu_proof_sha256: str | None
    eagle3_qualification_compatibility_authority_sha256: str | None
    eagle3_qualification_model_selector_sha256: str | None
    status: str
    blocker_codes: tuple[str, ...]
    unresolved_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only adaptation recipe declaration schema 1 is supported")
        if type(self.lookup_key) is not AdaptationRecipeLookupKey:
            raise TypeError("adaptation recipe requires an exact lookup key")
        if type(self.optimizer) is not OptimizerRecipeDeclaration:
            raise TypeError("adaptation recipe requires an exact optimizer declaration")
        for name in (
            "source_authority",
            "kv_history_policy",
        ):
            _require_exact_text(f"adaptation recipe {name}", getattr(self, name))
        for name in (
            "weight_update_mode",
            "parameter_scope",
            "adaptation_scope",
            "adaptation_group_id",
            "lora_matrix_policy",
            "native_head_policy",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_exact_text(f"adaptation recipe {name}", value)
        for name in ("teacher_row_policy", "verification_mode"):
            value = getattr(self, name)
            if value is not None:
                _require_exact_text(f"adaptation recipe {name}", value)
        if type(self.source_authority_sha256) is not str or not _LOWER_SHA256(
            self.source_authority_sha256
        ):
            raise ValueError("recipe source authority must be content-bound")
        for name in (
            "rank",
            "lora_alpha",
            "stride",
            "max_in_flight",
            "canvas_tokens",
            "extra_logical_delay",
            "fixed_verification_budget",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise TypeError(f"adaptation recipe {name} must be an exact integer")
        if (
            self.loss_position_decay is not None
            and type(self.loss_position_decay) is not float
        ):
            raise TypeError(
                "adaptation recipe loss_position_decay must be an exact float"
            )
        if (
            self.confidence_loss_weight is not None
            and type(self.confidence_loss_weight) is not float
        ):
            raise TypeError(
                "adaptation recipe confidence_loss_weight must be an exact float"
            )
        for name in (
            "chronobelief_release_capability_sha256",
            "chronobelief_gpu_proof_sha256",
            "eagle3_e0_execution_authority_sha256",
            "eagle3_compatibility_authority_sha256",
            "eagle3_model_selector_sha256",
            "eagle3_native_gpu_proof_sha256",
            "eagle3_qualification_compatibility_authority_sha256",
            "eagle3_qualification_model_selector_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not _LOWER_SHA256(value):
                raise ValueError(f"adaptation recipe {name} must be a SHA-256")
        if type(self.status) is not str:
            raise TypeError("adaptation recipe status must be exact text")
        for name in ("blocker_codes", "unresolved_fields"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not str for item in value):
                raise TypeError(f"adaptation recipe {name} must be exact text tuples")
        if self.lookup_key.authority_kind == "frozen_tts":
            self._validate_frozen_tts_declaration()
            return
        if self.lookup_key.scope != self.parameter_scope:
            raise ValueError("recipe scope differs from its lookup key")
        if self.lookup_key.parameterization != self.weight_update_mode:
            raise ValueError("recipe parameterization differs from its lookup key")
        if self.lookup_key.rank != self.rank:
            raise ValueError("recipe rank differs from its lookup key")
        expected_alpha = self.rank if self.weight_update_mode == "lora" else None
        if self.lora_alpha != expected_alpha:
            raise ValueError("recipe LoRA alpha must bind alpha/r=1")
        if self.lookup_key.optimizer != self.optimizer.name:
            raise ValueError("recipe optimizer differs from its lookup key")
        if self.lookup_key.experiment == "E1":
            if self.lookup_key.learning_rate is not None:
                raise ValueError("E1 anchor LR belongs to the recipe declaration")
        elif "learning_rate" in self.optimizer.unresolved_fields:
            if self.optimizer.learning_rate is not None:
                raise ValueError("unregistered E2 learning rate must remain null")
        elif self.lookup_key.learning_rate != self.optimizer.learning_rate:
            raise ValueError("recipe learning rate differs from its lookup key")
        if self.lookup_key.schedule != self.optimizer.schedule:
            raise ValueError("recipe schedule differs from its lookup key")
        if self.lookup_key.cohort != self.adaptation_group_id:
            raise ValueError("recipe cohort differs from its lookup key")
        if self.lookup_key.draft_width is not None:
            if self.canvas_tokens != self.lookup_key.draft_width:
                raise ValueError("exact-width recipe canvas differs from its key")
        elif self.canvas_tokens is not None:
            raise ValueError("selected-width recipe declarations retain a null canvas")
        if self.unresolved_fields != tuple(sorted(set(self.unresolved_fields))):
            raise ValueError("adaptation unresolved fields must be sorted and unique")
        if not set(self.unresolved_fields) <= _ADAPTATION_RECIPE_FIELDS:
            raise ValueError("adaptation recipe names an unknown unresolved field")
        for name in self.unresolved_fields:
            if getattr(self, name) is not None:
                raise ValueError("an unresolved adaptation field must remain null")
        if (
            self.loss_position_decay is None
            or not math.isfinite(self.loss_position_decay)
            or not (0 < self.loss_position_decay <= 1)
        ):
            raise ValueError("recipe loss position decay must be in (0, 1]")
        if self.extra_logical_delay is not None and self.extra_logical_delay < 0:
            raise ValueError("recipe extra logical delay must be non-negative")
        if self.max_in_flight != 1:
            raise ValueError("registered adaptation recipes permit one candidate")
        if self.status == "AVAILABLE":
            if (
                self.blocker_codes
                or self.unresolved_fields
                or self.optimizer.unresolved_fields
            ):
                raise ValueError("AVAILABLE recipes cannot retain unresolved semantics")
            if self.lookup_key.draft_width is None:
                raise ValueError("AVAILABLE recipes require an exact canvas authority")
            # This deliberately validates every config field.  No selector or
            # Pydantic default can turn a blocked template into an executable
            # recipe.
            self.to_adaptation_config()
        elif self.status == "BLOCKED":
            if not self.blocker_codes:
                raise ValueError("BLOCKED recipes require named blocker codes")
        else:
            raise ValueError("recipe status must be AVAILABLE or BLOCKED")
        if self.blocker_codes != tuple(sorted(set(self.blocker_codes))) or any(
            not _REASON_CODE(code) for code in self.blocker_codes
        ):
            raise ValueError("recipe blocker codes must be sorted stable tokens")
        if self.lookup_key.experiment == "E2":
            required_optimizer_fields = _e2_required_optimizer_unresolved_fields(
                optimizer=self.optimizer.name,
                schedule=self.optimizer.schedule,
            )
            if set(self.optimizer.unresolved_fields) != required_optimizer_fields:
                raise ValueError(
                    "E2 optimizer unresolved fields differ from source protocol"
                )
            if set(self.unresolved_fields) != (
                _E2_REQUIRED_ADAPTATION_UNRESOLVED_FIELDS
            ):
                raise ValueError(
                    "E2 adaptation unresolved fields differ from source protocol"
                )
            try:
                blocker_fields = tuple(
                    _E2_RECIPE_BLOCKER_FIELD_BY_CODE[code]
                    for code in self.blocker_codes
                )
            except KeyError as error:
                raise ValueError("E2 recipe blocker lacks a field mapping") from error
            expected_fields = {
                *(f"optimizer.{field}" for field in self.optimizer.unresolved_fields),
                *self.unresolved_fields,
            }
            if (
                len(blocker_fields) != len(set(blocker_fields))
                or set(blocker_fields) != expected_fields
            ):
                raise ValueError(
                    "E2 recipe blockers differ from its unresolved field set"
                )

    def _validate_frozen_tts_declaration(self) -> None:
        """Validate one blocked paper-reconstruction row in the shared lifecycle."""

        expected_optimizer_unresolved = frozenset(
            {
                "beta1",
                "beta2",
                "epsilon",
                "grad_clip",
                "learning_rate",
                "schedule",
                "weight_decay",
            }
        )
        if (
            self.source_authority != "tts_recipe_authority_v1"
            or self.source_authority_sha256
            != LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY.sha256
            or self.weight_update_mode is not None
            or self.parameter_scope is not None
            or self.kv_history_policy != "frozen"
            or self.adaptation_scope is not None
            or self.adaptation_group_id is not None
            or self.optimizer.name != "adam"
            or set(self.optimizer.unresolved_fields) != expected_optimizer_unresolved
            or any(
                getattr(self.optimizer, name) is not None
                for name in expected_optimizer_unresolved
            )
            or self.optimizer.momentum is not None
            or self.optimizer.muon_ns_steps is not None
            or self.optimizer.muon_auxiliary_learning_rate is not None
            or self.optimizer.muon_auxiliary_weight_decay is not None
            or self.optimizer.schedule_total_published_updates is not None
            or self.rank is not None
            or self.lora_alpha is not None
            or self.lora_matrix_policy is not None
            or self.native_head_policy is not None
            or self.stride is not None
            or self.max_in_flight is not None
            or self.canvas_tokens is not None
            or self.loss_position_decay is not None
            or self.extra_logical_delay is not None
            or self.teacher_row_policy != "latest_round_only"
            or self.verification_mode is not None
            or self.fixed_verification_budget is not None
            or self.confidence_loss_weight is not None
            or any(
                getattr(self, name) is not None
                for name in (
                    "chronobelief_release_capability_sha256",
                    "chronobelief_gpu_proof_sha256",
                    "eagle3_e0_execution_authority_sha256",
                    "eagle3_compatibility_authority_sha256",
                    "eagle3_model_selector_sha256",
                    "eagle3_native_gpu_proof_sha256",
                    "eagle3_qualification_compatibility_authority_sha256",
                    "eagle3_qualification_model_selector_sha256",
                )
            )
            or self.status != "BLOCKED"
            or self.blocker_codes
            != LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY.blocker_codes
            or self.unresolved_fields
            != tuple(sorted(_FROZEN_TTS_DECLARATION_UNRESOLVED_FIELDS))
        ):
            raise ValueError("frozen TTS declaration differs from source authority")

    def to_adaptation_config(self):
        """Materialize an exact-width declaration with no implicit values."""

        if self.status != "AVAILABLE":
            raise ValueError(
                "adaptation recipe is BLOCKED: " + ",".join(self.blocker_codes)
            )
        if (
            self.unresolved_fields
            or self.stride is None
            or self.canvas_tokens is None
            or self.extra_logical_delay is None
            or self.teacher_row_policy is None
            or self.verification_mode is None
        ):
            raise ValueError("adaptation recipe is not fully declared")
        from lightcone_spec.config.schema import AdaptationConfig

        return AdaptationConfig(
            weight_update_mode=self.weight_update_mode,
            parameter_scope=self.parameter_scope,
            kv_history_policy=self.kv_history_policy,
            adaptation_scope=self.adaptation_scope,
            adaptation_group_id=self.adaptation_group_id,
            optimizer=self.optimizer.to_optimizer_config(),
            rank=self.rank,
            lora_alpha=self.lora_alpha,
            lora_matrix_policy=self.lora_matrix_policy,
            native_head_policy=self.native_head_policy,
            stride=self.stride,
            max_in_flight=self.max_in_flight,
            canvas_tokens=self.canvas_tokens,
            loss_position_decay=self.loss_position_decay,
            extra_logical_delay=self.extra_logical_delay,
            teacher_row_policy=self.teacher_row_policy,
            verification_mode=self.verification_mode,
            fixed_verification_budget=self.fixed_verification_budget,
            confidence_loss_weight=self.confidence_loss_weight,
            chronobelief_release_capability_sha256=(
                self.chronobelief_release_capability_sha256
            ),
            chronobelief_gpu_proof_sha256=self.chronobelief_gpu_proof_sha256,
            eagle3_e0_execution_authority_sha256=(
                self.eagle3_e0_execution_authority_sha256
            ),
            eagle3_compatibility_authority_sha256=(
                self.eagle3_compatibility_authority_sha256
            ),
            eagle3_model_selector_sha256=self.eagle3_model_selector_sha256,
            eagle3_native_gpu_proof_sha256=self.eagle3_native_gpu_proof_sha256,
            eagle3_qualification_compatibility_authority_sha256=(
                self.eagle3_qualification_compatibility_authority_sha256
            ),
            eagle3_qualification_model_selector_sha256=(
                self.eagle3_qualification_model_selector_sha256
            ),
        )

    @cached_property
    def blocker_matrix(self) -> tuple[AdaptationRecipeBlocker, ...]:
        """Return every unresolved E2 value as one stable field/code row."""

        if self.lookup_key.experiment != "E2":
            return ()
        try:
            rows = tuple(
                AdaptationRecipeBlocker(
                    field=_E2_RECIPE_BLOCKER_FIELD_BY_CODE[reason_code],
                    reason_code=reason_code,
                )
                for reason_code in self.blocker_codes
            )
        except KeyError as error:  # pragma: no cover - declaration invariant
            raise ValueError("E2 recipe blocker lacks a field mapping") from error
        return tuple(sorted(rows, key=lambda row: (row.field, row.reason_code)))

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def _e1_recipe_declaration(
    key: AdaptationRecipeLookupKey,
) -> AdaptationRecipeDeclaration:
    from lightcone_spec.experiments.protocol import (
        DFLASH_LOSS_POSITION_DECAY,
        tuning_candidates,
    )

    matches = tuple(
        candidate
        for candidate in tuning_candidates()
        if (
            candidate.parameter_scope,
            candidate.weight_update_mode,
            candidate.rank,
            candidate.optimizer,
        )
        == (key.scope, key.parameterization, key.rank, key.optimizer)
    )
    if len(matches) != 1:
        raise ValueError("E1 recipe key does not resolve one registered candidate")
    candidate = matches[0]
    if key.learning_rate is not None or key.schedule != candidate.schedule:
        raise ValueError("E1 cell identity conflicts with its registered anchor")
    # TuningCandidate predates the explicit recipe codec and therefore does not
    # carry this canonical OptimizerConfig field.  Bind it here as source
    # semantics instead of silently inheriting the Pydantic default.
    optimizer_epsilon = 1e-8
    optimizer = OptimizerRecipeDeclaration(
        name=candidate.optimizer,
        learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
        beta1=candidate.beta1,
        beta2=candidate.beta2,
        epsilon=optimizer_epsilon,
        grad_clip=candidate.grad_clip,
        momentum=candidate.momentum,
        muon_ns_steps=candidate.muon_ns_steps,
        muon_auxiliary_learning_rate=candidate.muon_auxiliary_learning_rate,
        muon_auxiliary_weight_decay=candidate.muon_auxiliary_weight_decay,
        schedule=candidate.schedule,
        schedule_total_published_updates=(candidate.schedule_total_published_updates),
    )
    fixed_semantics = {
        "kv_history_policy": "frozen",
        "adaptation_scope": "cohort",
        "lora_matrix_policy": "registered_matrices_v1",
        "native_head_policy": "frozen",
        "max_in_flight": 1,
        "loss_position_decay": DFLASH_LOSS_POSITION_DECAY,
        "extra_logical_delay": 0,
        "teacher_row_policy": "update_round",
        "verification_mode": "native_scheduler",
        "fixed_verification_budget": None,
        "confidence_loss_weight": None,
        "optimizer_epsilon": optimizer_epsilon,
    }
    return AdaptationRecipeDeclaration(
        schema_version=1,
        lookup_key=key,
        source_authority="registered_e1_tuning_grid_v1",
        source_authority_sha256=content_sha256(
            {
                "candidate_id": candidate.candidate_id,
                "fixed_semantics": fixed_semantics,
            }
        ),
        weight_update_mode=candidate.weight_update_mode,
        parameter_scope=candidate.parameter_scope,
        kv_history_policy="frozen",
        adaptation_scope="cohort",
        adaptation_group_id=key.cohort,
        optimizer=optimizer,
        rank=candidate.rank,
        lora_alpha=candidate.lora_alpha,
        lora_matrix_policy="registered_matrices_v1",
        native_head_policy="frozen",
        stride=candidate.stride,
        max_in_flight=1,
        canvas_tokens=key.draft_width,
        loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
        extra_logical_delay=0,
        teacher_row_policy="update_round",
        verification_mode="native_scheduler",
        fixed_verification_budget=None,
        confidence_loss_weight=None,
        chronobelief_release_capability_sha256=None,
        chronobelief_gpu_proof_sha256=None,
        eagle3_e0_execution_authority_sha256=None,
        eagle3_compatibility_authority_sha256=None,
        eagle3_model_selector_sha256=None,
        eagle3_native_gpu_proof_sha256=None,
        eagle3_qualification_compatibility_authority_sha256=None,
        eagle3_qualification_model_selector_sha256=None,
        status="AVAILABLE",
        blocker_codes=(),
    )


def _e2_optimizer_declaration(
    key: AdaptationRecipeLookupKey,
) -> tuple[OptimizerRecipeDeclaration, tuple[str, ...]]:
    optimizer = key.optimizer
    unresolved: set[str] = {
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "grad_clip",
    }
    blockers: set[str] = {
        "e2_learning_rate_unregistered",
        "e2_weight_decay_unregistered",
        "e2_beta1_unregistered",
        "e2_beta2_unregistered",
        "e2_epsilon_unregistered",
        "e2_grad_clip_unregistered",
    }
    weight_decay: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    epsilon: float | None = None
    momentum: float | None = None
    muon_ns_steps: int | None = None
    auxiliary_lr: float | None = None
    auxiliary_decay: float | None = None

    if optimizer in {"sgdm", "nag", "muon"}:
        unresolved.add("momentum")
        blockers.add("e2_momentum_unregistered")
    if optimizer == "muon":
        unresolved.update(
            {
                "muon_ns_steps",
                "muon_auxiliary_learning_rate",
                "muon_auxiliary_weight_decay",
            }
        )
        blockers.update(
            {
                "e2_muon_ns_steps_unregistered",
                "e2_muon_auxiliary_learning_rate_unregistered",
                "e2_muon_auxiliary_weight_decay_unregistered",
            }
        )
    if key.schedule == "cosine_to_zero":
        unresolved.add("schedule_total_published_updates")
        blockers.add("e2_cosine_horizon_unregistered")
    values = {
        "name": optimizer,
        # The numeric cell axis describes the intended logarithmic template,
        # but neither specification fixes its endpoints as execution authority.
        "learning_rate": None,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "grad_clip": None,
        "momentum": momentum,
        "muon_ns_steps": muon_ns_steps,
        "muon_auxiliary_learning_rate": auxiliary_lr,
        "muon_auxiliary_weight_decay": auxiliary_decay,
        "schedule": key.schedule,
        "schedule_total_published_updates": None,
    }
    return (
        OptimizerRecipeDeclaration(
            **values,
            unresolved_fields=tuple(sorted(unresolved)),
        ),
        tuple(sorted(blockers)),
    )


def _e2_recipe_declaration(
    key: AdaptationRecipeLookupKey,
) -> AdaptationRecipeDeclaration:
    from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY

    optimizer, optimizer_blockers = _e2_optimizer_declaration(key)
    blockers = tuple(
        sorted(
            {
                *optimizer_blockers,
                "e2_extra_logical_delay_unregistered",
                "e2_fixed_verification_budget_unregistered",
                "e2_teacher_row_policy_unregistered",
                "e2_update_stride_unregistered",
                "e2_verification_mode_unregistered",
            }
        )
    )
    return AdaptationRecipeDeclaration(
        schema_version=1,
        lookup_key=key,
        source_authority="registered_e2_optimizer_template_v1",
        source_authority_sha256=content_sha256(
            {
                "lookup_key": key,
                "optimizer": optimizer,
                "blocker_codes": blockers,
                "draft_width_selector": E2_DRAFT_WIDTH_SELECTOR,
            }
        ),
        weight_update_mode=key.parameterization,
        parameter_scope=key.scope,
        kv_history_policy="frozen",
        adaptation_scope="cohort",
        adaptation_group_id=key.cohort,
        optimizer=optimizer,
        rank=key.rank,
        lora_alpha=key.rank if key.parameterization == "lora" else None,
        lora_matrix_policy="registered_matrices_v1",
        native_head_policy="frozen",
        stride=None,
        max_in_flight=1,
        canvas_tokens=None,
        loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
        extra_logical_delay=None,
        teacher_row_policy=None,
        verification_mode=None,
        fixed_verification_budget=None,
        confidence_loss_weight=None,
        chronobelief_release_capability_sha256=None,
        chronobelief_gpu_proof_sha256=None,
        eagle3_e0_execution_authority_sha256=None,
        eagle3_compatibility_authority_sha256=None,
        eagle3_model_selector_sha256=None,
        eagle3_native_gpu_proof_sha256=None,
        eagle3_qualification_compatibility_authority_sha256=None,
        eagle3_qualification_model_selector_sha256=None,
        status="BLOCKED",
        blocker_codes=blockers,
        unresolved_fields=(
            "extra_logical_delay",
            "fixed_verification_budget",
            "stride",
            "teacher_row_policy",
            "verification_mode",
        ),
    )


def _frozen_tts_recipe_declaration(
    key: AdaptationRecipeLookupKey,
) -> AdaptationRecipeDeclaration:
    """Project the paper authority into the sole adaptation-recipe lifecycle."""

    if key.authority_kind != "frozen_tts":
        raise ValueError("frozen TTS declaration requires its canonical lookup key")
    optimizer = OptimizerRecipeDeclaration(
        name="adam",
        learning_rate=None,
        weight_decay=None,
        beta1=None,
        beta2=None,
        epsilon=None,
        grad_clip=None,
        momentum=None,
        muon_ns_steps=None,
        muon_auxiliary_learning_rate=None,
        muon_auxiliary_weight_decay=None,
        schedule=None,
        schedule_total_published_updates=None,
        unresolved_fields=tuple(
            sorted(
                (
                    "beta1",
                    "beta2",
                    "epsilon",
                    "grad_clip",
                    "learning_rate",
                    "schedule",
                    "weight_decay",
                )
            )
        ),
    )
    return AdaptationRecipeDeclaration(
        schema_version=1,
        lookup_key=key,
        source_authority="tts_recipe_authority_v1",
        source_authority_sha256=(LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY.sha256),
        weight_update_mode=None,
        parameter_scope=None,
        kv_history_policy="frozen",
        adaptation_scope=None,
        adaptation_group_id=None,
        optimizer=optimizer,
        rank=None,
        lora_alpha=None,
        lora_matrix_policy=None,
        native_head_policy=None,
        stride=None,
        max_in_flight=None,
        canvas_tokens=None,
        loss_position_decay=None,
        extra_logical_delay=None,
        teacher_row_policy="latest_round_only",
        verification_mode=None,
        fixed_verification_budget=None,
        confidence_loss_weight=None,
        chronobelief_release_capability_sha256=None,
        chronobelief_gpu_proof_sha256=None,
        eagle3_e0_execution_authority_sha256=None,
        eagle3_compatibility_authority_sha256=None,
        eagle3_model_selector_sha256=None,
        eagle3_native_gpu_proof_sha256=None,
        eagle3_qualification_compatibility_authority_sha256=None,
        eagle3_qualification_model_selector_sha256=None,
        status="BLOCKED",
        blocker_codes=LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY.blocker_codes,
        unresolved_fields=tuple(sorted(_FROZEN_TTS_DECLARATION_UNRESOLVED_FIELDS)),
    )


def _build_adaptation_recipe_declarations(
    cells: Sequence[ExperimentCell],
) -> tuple[AdaptationRecipeDeclaration, ...]:
    keys: dict[str, AdaptationRecipeLookupKey] = {}
    for cell in cells:
        if _has_recipe_sentinel(cell.identity, FROZEN_TTS_RECIPE_SENTINEL):
            key = AdaptationRecipeLookupKey.from_cell(cell)
            keys[key.sha256] = key
            continue
        if cell.identity.experiment not in {"E1", "E2"} or cell.identity.method != "l0":
            continue
        key = AdaptationRecipeLookupKey.from_cell(cell)
        keys[key.sha256] = key
    declarations = tuple(
        _frozen_tts_recipe_declaration(key)
        if key.authority_kind == "frozen_tts"
        else (
            _e1_recipe_declaration(key)
            if key.experiment == "E1"
            else _e2_recipe_declaration(key)
        )
        for _, key in sorted(keys.items())
    )
    identities = tuple(row.lookup_key.sha256 for row in declarations)
    if identities != tuple(sorted(set(identities))):
        raise AssertionError("adaptation recipe declarations are not canonical")
    return declarations


def e1a_adaptive_configurations() -> tuple[ParameterConfiguration, ...]:
    """Return the preregistered 56 DSpark adaptive configurations."""

    configurations: list[ParameterConfiguration] = []
    for scope in E1_SCOPES:
        configurations.append(
            ParameterConfiguration(scope, "full", None, None, "frozen")
        )
        configurations.extend(
            ParameterConfiguration(scope, "lora", rank, 1.0, "frozen")
            for rank in LORA_RANKS
        )
    for depth in ("last1", "last3", "last5"):
        scope = f"{depth}_native_heads"
        configurations.append(ParameterConfiguration(scope, "full", None, None, "full"))
        configurations.extend(
            ParameterConfiguration(scope, "lora", rank, 1.0, "full")
            for rank in LORA_RANKS
        )
    if (
        len(configurations) != 56
        or len({configuration.sha256 for configuration in configurations}) != 56
    ):
        raise AssertionError(
            "E1a must contain exactly 56 unique adaptive configurations"
        )
    return tuple(configurations)


def _axis(name: str, values: Sequence[str | int | float]) -> AxisSpec:
    return AxisSpec(name=name, values=tuple(values))


def _industrial_definitions() -> tuple[ExperimentDefinition, ...]:
    configurations = e1a_adaptive_configurations()
    return (
        ExperimentDefinition(
            name="preflight",
            dependencies=(),
            locked_outputs=("runtime_envelope",),
            axes=(
                _axis(
                    "gate",
                    (
                        "identity",
                        "exactness",
                        "memory",
                        "telemetry",
                        "two_gpu_interference",
                    ),
                ),
            ),
        ),
        ExperimentDefinition(
            name="E3a",
            dependencies=("preflight",),
            locked_outputs=(
                "baseline_capacity_envelope",
                "e1_reference_load",
                "matched_width",
                "width_selection_rule",
                "static_target_crossover",
                "drift_witness",
            ),
            axes=(
                _axis("method", ("target_only", "static")),
                _axis("context", CONTEXT_GRID),
                _axis("concurrency", E3A_CONCURRENCY_GRID),
                _axis("width", DRAFT_WIDTHS),
                _axis("regime", CONTEXT_REGIMES),
            ),
        ),
        ExperimentDefinition(
            name="TTS-Cal",
            dependencies=("E3a",),
            locked_outputs=("frozen_tts_recipe",),
            axes=(
                _axis("method_role", ("tts_calibration_candidate",)),
                _axis("optimizer", ("adam",)),
                _axis(
                    "learning_rate",
                    (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
                ),
                _axis("stride", (1, 5, 10, 15, 20, 30, 40, 50)),
                _axis("pilot_block", PILOT_BLOCKS),
                _axis("result_class", ("tuning_only_not_formal",)),
            ),
        ),
        ExperimentDefinition(
            name="E1",
            dependencies=("TTS-Cal",),
            locked_outputs=("dflash_pareto_set", "common_downstream_load"),
            axes=(
                _axis("search_method_role", (ScientificMethodRole.LC_CANDIDATE.value,)),
                _axis(
                    "anchor_method_role",
                    (
                        ScientificMethodRole.TTS.value,
                        ScientificMethodRole.L0_NAIVE.value,
                    ),
                ),
                _axis("scope", E1_SCOPES),
                _axis("parameterization", ("full", "lora")),
                _axis("rank", LORA_RANKS),
                _axis("alpha_over_rank", (1.0,)),
                _axis("optimizer_anchor", E1_OPTIMIZER_ANCHORS),
            ),
        ),
        ExperimentDefinition(
            name="E2",
            dependencies=("E1",),
            locked_outputs=("dflash_recipe",),
            axes=(
                _axis("search_method_role", (ScientificMethodRole.LC_CANDIDATE.value,)),
                _axis(
                    "anchor_method_role",
                    (
                        ScientificMethodRole.TTS.value,
                        ScientificMethodRole.L0_NAIVE.value,
                    ),
                ),
                _axis("scope", E1_SCOPES),
                _axis("parameterization", ("full", "lora")),
                _axis("rank", LORA_RANKS),
                _axis("optimizer", E2_OPTIMIZERS),
                _axis("learning_rate", ("optimizer_specific_log_grid",)),
                _axis("schedule", E2_SCHEDULES),
                _axis("halving_context", (4096, 8192, 16384, 40928)),
                _axis("halving_batch", (2, 4, 8, 16)),
            ),
        ),
        ExperimentDefinition(
            name="E4",
            dependencies=("E2",),
            locked_outputs=("systems_mechanism_gate",),
            axes=(
                _axis(
                    "cumulative_ablation",
                    (
                        "synchronous_main_stream",
                        "side_stream_tts",
                        "l0_first_ready",
                        "cohort_batch",
                        "fixed_buffer_graph",
                        "bounded_telemetry",
                    ),
                ),
                _axis("load", ("low", "moderate", "saturation")),
                _axis("traffic", ("pure_decode", "mixed_prefill_decode")),
                _axis("chunked_prefill", ("disabled", "enabled")),
                _axis("prefix_reuse", ("none", "shared")),
                _axis("update_stride", ("locked_e2_operational_grid",)),
                _axis("microbatch", ("locked_e2_operational_grid",)),
                _axis("coalescing", ("locked_e2_operational_grid",)),
                _axis("stream_priority", ("locked_e2_operational_grid",)),
                _axis("profile", ("nvtx", "nsight_systems", "nsight_compute")),
            ),
        ),
        ExperimentDefinition(
            name="E3b",
            dependencies=("E4",),
            locked_outputs=("long_context_confirmation",),
            axes=(
                _axis("method_role", CONFIRMATION_METHOD_ROLES),
                _axis("context", CONTEXT_GRID),
                _axis("regime", CONTEXT_REGIMES),
                _axis("load", ("concurrency_one", "common_load")),
                _axis("width_panel", ("matched", "deployment_optimal")),
                _axis("block", REGISTERED_CONFIRMATION_BLOCKS),
                _axis("block_phase", ("excluded_pilot", "final_candidate")),
            ),
        ),
        ExperimentDefinition(
            name="E1a",
            dependencies=("E3b",),
            locked_outputs=("dspark_recipe",),
            axes=(
                _axis(
                    "configuration_sha256", tuple(row.sha256 for row in configurations)
                ),
                _axis("rank", LORA_RANKS),
                _axis(
                    "verification_mode",
                    ("fixed_verification_budget", "native_scheduler"),
                ),
                _axis("baseline", ("target_only", "static")),
            ),
        ),
        ExperimentDefinition(
            name="E5",
            dependencies=("E1a",),
            locked_outputs=("production_slo_surface", "topology_failure_surface"),
            axes=(
                _axis("backend", ("DFLASH", "DSPARK")),
                _axis("method_role", CONFIRMATION_METHOD_ROLES),
                _axis("closed_loop_concurrency", E5_CLOSED_LOOP_CONCURRENCY),
                _axis("open_loop_lambda_star", E5_OPEN_LOOP_LOAD_FACTORS),
                _axis(
                    "arrival",
                    (
                        "poisson",
                        "immediate_burst",
                        "burstgpt_shape",
                        "moderate_soak",
                        "saturation_soak",
                        "overload_soak",
                    ),
                ),
                _axis("cohort_count", E5_COHORT_COUNTS),
                _axis("cohort_distribution", E5_COHORT_DISTRIBUTIONS),
                _axis("topology", E5_TOPOLOGIES),
                _axis("failure", E5_FAILURES),
                _axis("block", REGISTERED_CONFIRMATION_BLOCKS),
                _axis("block_phase", ("excluded_pilot", "final_candidate")),
            ),
        ),
        ExperimentDefinition(
            name="E6",
            dependencies=("E5",),
            locked_outputs=("native_mtp_transfer_surface",),
            axes=(
                _axis("model", E6_CANDIDATE_MODELS),
                _axis(
                    "headline_method_role",
                    (
                        ScientificMethodRole.TARGET_ONLY.value,
                        ScientificMethodRole.STATIC.value,
                        ScientificMethodRole.TTS.value,
                        ScientificMethodRole.LIGHTCONE.value,
                    ),
                ),
                _axis("mechanism_anchor_role", (ScientificMethodRole.L0_NAIVE.value,)),
                _axis("task", ("LiveCodeBench", "MATH-500")),
                _axis("context", (4096, 16384, 32768)),
                _axis("load", ("concurrency_one", "common_slo_load")),
                _axis("block", REGISTERED_CONFIRMATION_BLOCKS),
                _axis("block_phase", ("excluded_pilot", "final_candidate")),
                _axis("topology", ("tp2_dp1",)),
            ),
        ),
        ExperimentDefinition(
            name="E0",
            dependencies=("E6",),
            locked_outputs=("breadth_surface",),
            axes=(
                _axis("model", E0_MODELS),
                _axis("backend", E0_BACKENDS),
                _axis("task", E0_TASKS),
                _axis("method_role", E0_METHOD_ROLES),
                _axis("load", E0_LOADS),
                _axis("block", REGISTERED_CONFIRMATION_BLOCKS),
                _axis("block_phase", ("excluded_pilot", "final_candidate")),
            ),
        ),
    )


@dataclass(frozen=True)
class ExperimentRegistry:
    schema_version: int
    name: str
    gpu_uuids: tuple[str, ...]
    definitions: tuple[ExperimentDefinition, ...]
    cells: tuple[ExperimentCell, ...]
    materialization_mode: str = "signed_staged"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 3:
            raise ValueError("only industrial registry schema version 3 is supported")
        _require_text("registry name", self.name)
        if self.materialization_mode not in {
            "signed_staged",
            "legacy_diagnostic",
        }:
            raise ValueError("registry materialization mode is unsupported")
        if not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError(
                "the industrial registry requires unique logical GPU slots"
            )
        definition_names = tuple(definition.name for definition in self.definitions)
        if definition_names != INDUSTRIAL_EXPERIMENT_ORDER:
            raise ValueError("industrial experiment dependency order is immutable")
        for index, definition in enumerate(self.definitions):
            expected = () if index == 0 else (definition_names[index - 1],)
            if definition.dependencies != expected:
                raise ValueError(
                    f"experiment {definition.name} must depend exactly on {expected}"
                )
        if not self.cells:
            raise ValueError("registry must contain cells")
        identity_ids = [cell.identity.sha256 for cell in self.cells]
        if len(identity_ids) != len(set(identity_ids)):
            raise ValueError("registry contains duplicate cell identities")
        known = set(definition_names)
        seen_stages: set[str] = set()
        for cell in self.cells:
            if cell.identity.experiment not in known:
                raise ValueError("cell names an unknown experiment")
            if not set(cell.resources.gpu_uuids).issubset(self.gpu_uuids):
                raise ValueError("cell reserves a GPU outside the registry inventory")
            seen_stages.add(cell.identity.experiment)
        if self.materialization_mode == "legacy_diagnostic":
            if seen_stages != known:
                raise ValueError("legacy registry must declare every experiment")
        else:
            staged_prefix = set(INDUSTRIAL_EXPERIMENT_ORDER[:3])
            if seen_stages != staged_prefix:
                raise ValueError(
                    "staged registry must contain exactly preflight, E3a, and TTS-Cal"
                )
        for cell in self.cells:
            identity = cell.identity
            _derived_scientific_method_role(cell)
            if self.materialization_mode == "signed_staged" and any(
                value in {FROZEN_TTS_RECIPE_SENTINEL, SEALED_E2_RECIPE_SENTINEL}
                for value in (
                    identity.scope,
                    identity.optimizer,
                    identity.schedule,
                    identity.parameterization,
                )
            ):
                raise ValueError("staged registry cannot contain recipe sentinels")
            if (
                identity.optimizer
                in {
                    FROZEN_TTS_RECIPE_SENTINEL,
                    SEALED_E2_RECIPE_SENTINEL,
                }
            ) != (
                identity.schedule
                in {
                    FROZEN_TTS_RECIPE_SENTINEL,
                    SEALED_E2_RECIPE_SENTINEL,
                }
            ):
                raise ValueError("recipe sentinels must bind optimizer and schedule")
        declarations = {
            row.lookup_key.sha256: row for row in self.adaptation_recipe_declarations
        }
        for cell in self.cells:
            identity = cell.identity
            if identity.method in {"tts", "l0"} and _has_recipe_sentinel(
                identity, FROZEN_TTS_RECIPE_SENTINEL
            ):
                key = AdaptationRecipeLookupKey.from_cell(cell)
                declaration = declarations.get(key.sha256)
                if declaration is None:
                    raise ValueError("frozen TTS anchor lacks a recipe declaration")
                if (
                    declaration.source_authority_sha256
                    != self.legacy_diagnostic_tts_reconstruction_authority.sha256
                ):
                    raise ValueError("frozen TTS recipe source authority differs")
                if (
                    declaration.status == "BLOCKED"
                    and cell.status is not CellStatus.BLOCKED
                ):
                    raise ValueError(
                        "blocked frozen TTS recipe must block formal cells"
                    )
                continue
            if identity.method == "l0" and _has_recipe_sentinel(
                identity, SEALED_E2_RECIPE_SENTINEL
            ):
                if (
                    cell.status is not CellStatus.BLOCKED
                    or cell.reason_code != "sealed_e2_recipe_receipt_required"
                ):
                    raise ValueError(
                        "unmaterialized LightCone templates require an E2 seal"
                    )
                continue
            if identity.experiment not in {"E1", "E2"} or identity.method != "l0":
                continue
            key = AdaptationRecipeLookupKey.from_cell(cell)
            declaration = declarations.get(key.sha256)
            if declaration is None:
                raise ValueError("E1/E2 adaptive cell lacks a recipe declaration")
            if (
                declaration.status == "BLOCKED"
                and cell.status is not CellStatus.BLOCKED
            ):
                raise ValueError(
                    "blocked adaptation recipe must block its registry cell"
                )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "materialization_mode": self.materialization_mode,
            "gpu_uuids": list(self.gpu_uuids),
            "definitions": [_canonical(row) for row in self.definitions],
            "adaptation_recipe_declarations": [
                _canonical(row) for row in self.adaptation_recipe_declarations
            ],
            "adaptation_recipe_declarations_sha256": (
                self.adaptation_recipe_declarations_sha256
            ),
            "cells": [
                _canonical(cell)
                for cell in sorted(self.cells, key=lambda row: row.cell_id)
            ],
        }
        if self.materialization_mode == "legacy_diagnostic":
            value["legacy_frozen_tts_recipe_authority"] = _canonical(
                self.legacy_diagnostic_tts_reconstruction_authority
            )
            value["legacy_frozen_tts_recipe_authority_sha256"] = (
                self.legacy_diagnostic_tts_reconstruction_authority.sha256
            )
        return value

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def definition(self, experiment: str) -> ExperimentDefinition:
        for definition in self.definitions:
            if definition.name == experiment:
                return definition
        raise ValueError(f"unknown experiment {experiment!r}")

    def cells_for(self, experiment: str) -> tuple[ExperimentCell, ...]:
        self.definition(experiment)
        return tuple(
            sorted(
                (cell for cell in self.cells if cell.identity.experiment == experiment),
                key=lambda row: row.cell_id,
            )
        )

    @cached_property
    def _cells_by_id(self) -> dict[str, ExperimentCell]:
        return {cell.cell_id: cell for cell in self.cells}

    @property
    def legacy_diagnostic_tts_reconstruction_authority(
        self,
    ) -> FrozenTtsRecipeAuthority:
        if self.materialization_mode != "legacy_diagnostic":
            raise ValueError("paper-only frozen TTS authority is a legacy diagnostic")
        return LEGACY_DIAGNOSTIC_TTS_RECONSTRUCTION_AUTHORITY

    def scientific_method_role_for_cell(
        self, cell_or_id: ExperimentCell | str
    ) -> ScientificMethodRole:
        """Derive a scientific role only from an exact registry-owned cell."""

        if type(cell_or_id) is str:
            cell = self._cells_by_id.get(cell_or_id)
            if cell is None:
                raise ValueError("scientific-role cell ID is absent from the registry")
        elif type(cell_or_id) is ExperimentCell:
            cell = self._cells_by_id.get(cell_or_id.cell_id)
            if cell is None or cell != cell_or_id:
                raise ValueError("scientific-role cell is not registry-owned")
        else:
            raise TypeError("scientific-role lookup requires a cell or cell ID")
        return _derived_scientific_method_role(cell)

    @cached_property
    def adaptation_recipe_declarations(
        self,
    ) -> tuple[AdaptationRecipeDeclaration, ...]:
        return _build_adaptation_recipe_declarations(self.cells)

    @cached_property
    def adaptation_recipe_declarations_sha256(self) -> str:
        return content_sha256(self.adaptation_recipe_declarations)

    def adaptation_recipe_for_cell(
        self, cell_or_id: ExperimentCell | str
    ) -> AdaptationRecipeDeclaration:
        """Resolve only a cell owned by this registry to one exact declaration."""

        if type(cell_or_id) is str:
            cell = self._cells_by_id.get(cell_or_id)
            if cell is None:
                raise ValueError(
                    "adaptation recipe cell ID is absent from the registry"
                )
        elif type(cell_or_id) is ExperimentCell:
            cell = self._cells_by_id.get(cell_or_id.cell_id)
            if cell is None or cell != cell_or_id:
                raise ValueError("adaptation recipe cell is not registry-owned")
        else:
            raise TypeError("adaptation recipe lookup requires a cell or cell ID")
        key = AdaptationRecipeLookupKey.from_cell(cell)
        declarations = {
            row.lookup_key.sha256: row for row in self.adaptation_recipe_declarations
        }
        try:
            return declarations[key.sha256]
        except KeyError as exc:
            raise ValueError("registry cell lacks an adaptation recipe") from exc

    def make_receipt(
        self,
        experiment: str,
        outputs: Mapping[str, str],
        *,
        runtime_sha256: str,
        split_sha256: str,
        completed_cells_sha256: str,
        dependencies: Sequence[ExperimentReceipt] = (),
    ) -> ExperimentReceipt:
        definition = self.definition(experiment)
        if set(outputs) != set(definition.locked_outputs):
            raise ValueError(
                "receipt must bind every and only registered locked output"
            )
        dependency_by_name = self.validate_receipts(dependencies)
        if not set(definition.dependencies) <= set(dependency_by_name):
            raise ValueError(
                "receipt dependencies must include every direct dependency"
            )
        receipt = ExperimentReceipt(
            experiment=experiment,
            registry_sha256=self.sha256,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            completed_cells_sha256=completed_cells_sha256,
            dependency_receipts=tuple(
                LockedOutput(name=name, content_sha256=dependency_by_name[name].sha256)
                for name in definition.dependencies
            ),
            outputs=tuple(
                LockedOutput(name=name, content_sha256=outputs[name])
                for name in definition.locked_outputs
            ),
        )
        return receipt

    def validate_receipts(
        self, receipts: Sequence[ExperimentReceipt]
    ) -> dict[str, ExperimentReceipt]:
        by_experiment: dict[str, ExperimentReceipt] = {}
        for receipt in receipts:
            if receipt.registry_sha256 != self.sha256:
                raise ValueError("dependency receipt belongs to another registry")
            if receipt.experiment in by_experiment:
                raise ValueError("duplicate dependency receipt")
            definition = self.definition(receipt.experiment)
            if {output.name for output in receipt.outputs} != set(
                definition.locked_outputs
            ):
                raise ValueError("dependency receipt has incomplete or extra outputs")
            by_experiment[receipt.experiment] = receipt
        for definition in self.definitions:
            if definition.name not in by_experiment:
                continue
            missing = set(definition.dependencies) - by_experiment.keys()
            if missing:
                raise ValueError(
                    f"receipt for {definition.name} is missing dependencies {sorted(missing)}"
                )
            expected_dependencies = {
                name: by_experiment[name].sha256 for name in definition.dependencies
            }
            actual_dependencies = {
                row.name: row.content_sha256
                for row in by_experiment[definition.name].dependency_receipts
            }
            if actual_dependencies != expected_dependencies:
                raise ValueError(
                    f"receipt for {definition.name} does not bind exact dependencies"
                )
        return by_experiment

    def ready_experiment(self, receipts: Sequence[ExperimentReceipt]) -> str | None:
        completed = self.validate_receipts(receipts)
        for definition in self.definitions:
            if definition.name in completed:
                continue
            if all(dependency in completed for dependency in definition.dependencies):
                return definition.name
        return None


def scientific_role_for_cell(
    registry: ExperimentRegistry, cell_or_id: ExperimentCell | str
) -> str:
    """Return the role value for an exact cell owned by ``registry``."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("scientific-role resolution requires an ExperimentRegistry")
    return registry.scientific_method_role_for_cell(cell_or_id).value


class _CellFactory:
    def __init__(
        self,
        *,
        gpu_uuids: tuple[str, ...],
        base_port: int,
        cache_root: str,
        evidence_root: str,
        seed: int,
    ) -> None:
        if len(gpu_uuids) < 2 or len(set(gpu_uuids)) != len(gpu_uuids):
            raise ValueError("two or more distinct logical GPU slots are required")
        port_span = _industrial_port_span(len(gpu_uuids))
        if base_port < 1024 or base_port + port_span - 1 > 65_535:
            raise ValueError("base_port must leave room for the dispatch port pool")
        _require_text("cache_root", cache_root)
        _require_text("evidence_root", evidence_root)
        self.gpu_uuids = gpu_uuids
        self.base_port = base_port
        self.cache_root = cache_root.rstrip("/")
        self.evidence_root = evidence_root.rstrip("/")
        self.seed = seed
        self.single_gpu_index = 0
        self.cells: list[ExperimentCell] = []

    def add(
        self,
        *,
        experiment: str,
        model: str,
        backend: str,
        task: str,
        method: str,
        workload_class: WorkloadClass,
        gpu_count: int = 1,
        scope: str | None = None,
        rank: int | None = None,
        alpha_over_rank: float | None = None,
        optimizer: str | None = None,
        learning_rate: float | None = None,
        schedule: str | None = None,
        context: int | None = None,
        regime: str = "not_applicable",
        width: int | None = None,
        arrival: str = "not_applicable",
        slo: str = "not_applicable",
        cohort: str = "K=1:uniform",
        topology: str = "tp1_dp1",
        block: int = 0,
        parameterization: str = "none",
        variant: str = "default",
        concurrency: int | None = None,
        load_factor: float | None = None,
        cohort_count: int = 1,
        gpu_index: int | None = None,
        status: CellStatus = CellStatus.UNMEASURED,
        reason_code: str = "awaiting_registered_measurement",
        reason: str = "No complete content-bound measurement exists.",
    ) -> ExperimentCell:
        if (
            status is CellStatus.UNMEASURED
            and topology in _PATCH_UNSUPPORTED_ADAPTIVE_TOPOLOGIES
            and workload_class is not WorkloadClass.DOWNLOAD
        ):
            status = CellStatus.BLOCKED
            reason_code = "release_topology_executor_unsupported"
            reason = (
                "This release has no executable TP2/DP2 serving path; a "
                "caller-authored capability digest cannot enable one."
            )
        elif status is CellStatus.UNMEASURED and method in _ADAPTIVE_METHODS:
            if backend in _PATCH_UNSUPPORTED_ADAPTIVE_BACKENDS:
                status = CellStatus.BLOCKED
                reason_code = "patched_runtime_backend_unsupported"
                reason = (
                    "The pinned schema-v3 patch rejects adaptive execution for this "
                    "backend before model loading."
                )
        if optimizer == schedule == FROZEN_TTS_RECIPE_SENTINEL:
            status = CellStatus.BLOCKED
            reason_code = "tts_official_recipe_unavailable"
            reason = (
                "The paper fixes TTS mechanisms but does not disclose the complete "
                "numeric recipe and no official implementation/config was found."
            )
        elif optimizer == schedule == SEALED_E2_RECIPE_SENTINEL:
            status = CellStatus.BLOCKED
            reason_code = "sealed_e2_recipe_receipt_required"
            reason = (
                "LightCone materialization requires the exact sealed E2 final-recipe "
                "receipt before execution."
            )
        if gpu_count == 1:
            if gpu_index is not None and gpu_index not in range(len(self.gpu_uuids)):
                raise ValueError("gpu_index lies outside the logical GPU slots")
            selected_gpu = (
                self.single_gpu_index % len(self.gpu_uuids)
                if gpu_index is None
                else gpu_index
            )
            assigned = (self.gpu_uuids[selected_gpu],)
            self.single_gpu_index += 1
            ports = (self.base_port + selected_gpu,)
        elif 1 < gpu_count <= len(self.gpu_uuids):
            if gpu_index is not None:
                raise ValueError("gang cells cannot select one gpu_index")
            assigned = self.gpu_uuids[:gpu_count]
            gang_port_start = (
                self.base_port + 2
                if gpu_count == 2 and len(self.gpu_uuids) == 2
                else self.base_port + len(self.gpu_uuids)
            )
            ports = tuple(range(gang_port_start, gang_port_start + gpu_count + 1))
        else:
            raise ValueError("gpu_count must fit the logical GPU-slot inventory")
        identity = CellIdentity(
            experiment=experiment,
            model=model,
            backend=backend,
            task=task,
            method=method,
            scope=scope,
            rank=rank,
            alpha_over_rank=alpha_over_rank,
            optimizer=optimizer,
            learning_rate=learning_rate,
            schedule=schedule,
            context=context,
            regime=regime,
            width=width,
            arrival=arrival,
            slo=slo,
            cohort=cohort,
            topology=topology,
            seed=self.seed,
            block=block,
            gpu_uuids=assigned,
            parameterization=parameterization,
            variant=variant,
            concurrency=concurrency,
            load_factor=load_factor,
            cohort_count=cohort_count,
        )
        prefix = identity.sha256
        resources = ResourceClaim(
            gpu_uuids=assigned,
            ports=ports,
            cache_root=f"{self.cache_root}/{experiment}/{prefix}",
            evidence_root=f"{self.evidence_root}/{experiment}/{prefix}",
            workload_class=workload_class,
        )
        cell = ExperimentCell(
            identity=identity,
            resources=resources,
            status=status,
            reason_code=reason_code,
            reason=reason,
        )
        self.cells.append(cell)
        return cell


def _add_preflight_cells(factory: _CellFactory) -> None:
    factory.add(
        experiment="preflight",
        model="registered_runtime",
        backend="SGLANG",
        task="environment_and_patch_preflight",
        method="target_only",
        workload_class=WorkloadClass.COMPILE,
        gpu_count=2,
        topology="two_gpu_host",
        variant="identity_environment_patch_compile",
    )
    factory.add(
        experiment="preflight",
        model="registered_runtime",
        backend="ALL_REGISTERED",
        task="exactness_memory_telemetry_preflight",
        method="static",
        workload_class=WorkloadClass.CORRECTNESS,
        gpu_count=2,
        topology="two_gpu_host",
        variant="exactness_memory_telemetry",
    )
    # The calibration reducer consumes matched single-GPU observations, not a
    # synthetic two-rank serving run.  Two repetitions are the registered
    # minimum for the paired BCa gate; every slot is an independent terminal
    # cell whose mode is fixed in the immutable identity.
    for repetition in range(2):
        for mode in ("isolated", "concurrent"):
            for slot in range(2):
                factory.add(
                    experiment="preflight",
                    model="Qwen/Qwen3-8B",
                    backend="DFLASH",
                    task="simultaneous_single_gpu_interference",
                    method="static",
                    workload_class=WorkloadClass.CORRECTNESS,
                    gpu_count=1,
                    context=4096,
                    regime="short_input_long_generation",
                    arrival="closed_loop_c1",
                    concurrency=1,
                    topology="tp1_dp1",
                    block=repetition,
                    variant=f"{mode}_slot_{slot}",
                    gpu_index=slot,
                )


def _add_e3a_cells(factory: _CellFactory) -> None:
    non_anchor_contexts = tuple(
        context for context in CONTEXT_GRID if context not in LONG_CONTEXT_ANCHORS
    )
    for context in non_anchor_contexts:
        for regime in CONTEXT_REGIMES:
            for method in ("target_only", "static"):
                factory.add(
                    experiment="E3a",
                    model="Qwen/Qwen3-8B",
                    backend="NONE" if method == "target_only" else "DFLASH",
                    task="controlled_baseline",
                    method=method,
                    workload_class=WorkloadClass.TUNING,
                    scope="none",
                    context=context,
                    regime=regime,
                    width=None if method == "target_only" else DRAFT_WIDTHS[1],
                    arrival="closed_loop",
                    slo="capacity_envelope",
                    concurrency=1,
                    variant="context_c1",
                )
    for context in LONG_CONTEXT_ANCHORS:
        for regime in CONTEXT_REGIMES:
            for concurrency in E3A_CONCURRENCY_GRID:
                factory.add(
                    experiment="E3a",
                    model="Qwen/Qwen3-8B",
                    backend="NONE",
                    task="controlled_baseline",
                    method="target_only",
                    workload_class=WorkloadClass.TUNING,
                    scope="none",
                    context=context,
                    regime=regime,
                    arrival="closed_loop",
                    slo="capacity_envelope",
                    concurrency=concurrency,
                    variant="context_load",
                )
                for width in DRAFT_WIDTHS:
                    factory.add(
                        experiment="E3a",
                        model="Qwen/Qwen3-8B",
                        backend="DFLASH",
                        task="controlled_baseline",
                        method="static",
                        workload_class=WorkloadClass.TUNING,
                        scope="none",
                        context=context,
                        regime=regime,
                        width=width,
                        arrival="closed_loop",
                        slo="capacity_envelope",
                        concurrency=concurrency,
                        variant="context_load_width",
                    )


def _add_tts_cal_cells(factory: _CellFactory) -> None:
    """Declare the disjoint 9-by-8 numeric TTS calibration grid.

    These are tuning-only rows.  Their signed offline reducer freezes one
    candidate before E1 materialization; no row is a publication result.
    """

    from lightcone_spec.experiments.formal_protocol import (
        TTS_LEARNING_RATES,
        TTS_STRIDES,
    )

    for learning_rate in TTS_LEARNING_RATES:
        for stride in TTS_STRIDES:
            for pilot_block in PILOT_BLOCKS:
                factory.add(
                    experiment="TTS-Cal",
                    model="Qwen/Qwen3-8B",
                    backend="DFLASH",
                    task="disjoint_tts_numeric_calibration",
                    method="tts",
                    workload_class=WorkloadClass.TUNING,
                    scope="full_drafter",
                    optimizer="adam",
                    learning_rate=learning_rate,
                    schedule="constant",
                    context=40928,
                    regime="short_input_long_generation",
                    width=16,
                    arrival="disjoint_tuning_window",
                    slo="safety_first_then_maximize_slo_goodput",
                    parameterization="full",
                    block=pilot_block,
                    variant=f"tts_calibration:stride={stride}",
                )


def _paired_gpu_index(seed: int, payload: Mapping[str, Any]) -> int:
    """Assign every member of a scientific pair to the same rotating GPU."""

    return int(content_sha256({"seed": seed, "pair": payload})[:16], 16) % 2


def _add_reference_baselines(
    factory: _CellFactory,
    *,
    experiment: str,
    backend: str,
    task: str,
    context: int,
    arrival: str = "locked_reference_load",
    variant: str = "reference_baseline",
    width: int | None = DRAFT_WIDTHS[1],
    concurrency: int | None = None,
    gpu_index: int | None = None,
) -> None:
    for method in ("target_only", "static"):
        factory.add(
            experiment=experiment,
            model="Qwen/Qwen3-8B",
            backend="NONE" if method == "target_only" else backend,
            task=task,
            method=method,
            workload_class=WorkloadClass.TUNING,
            scope="none",
            context=context,
            regime="short_input_long_generation",
            width=None if method == "target_only" else width,
            arrival=arrival,
            slo="tuning_safety",
            concurrency=concurrency,
            variant=variant,
            gpu_index=gpu_index,
        )


def _scientific_role_identity_fields(role: str) -> dict[str, Any]:
    """Map one reported role onto the existing runtime publication method."""

    parsed = ScientificMethodRole(role)
    if parsed is ScientificMethodRole.TARGET_ONLY:
        return {"method": "target_only", "scope": "none", "parameterization": "none"}
    if parsed is ScientificMethodRole.STATIC:
        return {"method": "static", "scope": "none", "parameterization": "none"}
    if parsed in {ScientificMethodRole.TTS, ScientificMethodRole.L0_NAIVE}:
        return {
            "method": "tts" if parsed is ScientificMethodRole.TTS else "l0",
            "scope": FROZEN_TTS_RECIPE_SENTINEL,
            "optimizer": FROZEN_TTS_RECIPE_SENTINEL,
            "schedule": FROZEN_TTS_RECIPE_SENTINEL,
            "parameterization": FROZEN_TTS_RECIPE_SENTINEL,
        }
    if parsed is ScientificMethodRole.LIGHTCONE:
        return {
            "method": "l0",
            "scope": SEALED_E2_RECIPE_SENTINEL,
            "optimizer": SEALED_E2_RECIPE_SENTINEL,
            "schedule": SEALED_E2_RECIPE_SENTINEL,
            "parameterization": SEALED_E2_RECIPE_SENTINEL,
        }
    if parsed in {
        ScientificMethodRole.ONLINESPEC_OGD,
        ScientificMethodRole.ONLINESPEC_OPT,
        ScientificMethodRole.ONLINESPEC_ENS,
    }:
        return {
            "method": parsed.value,
            "scope": "external_baseline_recipe",
            "parameterization": "external_baseline_recipe",
        }
    raise ValueError("LC candidates require an E1/E2 search geometry")


def _add_e1_cells(factory: _CellFactory) -> None:
    for width in DRAFT_WIDTHS:
        for concurrency in E3A_CONCURRENCY_GRID:
            selection = f"width={width}:concurrency={concurrency}"
            reference_gpu = _paired_gpu_index(
                factory.seed,
                {"experiment": "E1", "selection": selection, "kind": "reference"},
            )
            _add_reference_baselines(
                factory,
                experiment="E1",
                backend="DFLASH",
                task="LiveCodeBench_tuning",
                context=40928,
                variant=f"reference_baseline:{selection}",
                width=width,
                concurrency=concurrency,
                gpu_index=reference_gpu,
            )
            anchor_gpu = _paired_gpu_index(
                factory.seed,
                {
                    "experiment": "E1",
                    "selection": selection,
                    "kind": "frozen_tts_anchor",
                },
            )
            for role in (
                ScientificMethodRole.TTS.value,
                ScientificMethodRole.L0_NAIVE.value,
            ):
                factory.add(
                    experiment="E1",
                    model="Qwen/Qwen3-8B",
                    backend="DFLASH",
                    task="LiveCodeBench_tuning",
                    workload_class=WorkloadClass.TUNING,
                    context=40928,
                    regime="short_input_long_generation",
                    width=width,
                    arrival="locked_reference_load",
                    slo="tuning_safety",
                    concurrency=concurrency,
                    variant=f"frozen_tts_anchor:{selection}:role={role}",
                    gpu_index=anchor_gpu,
                    **_scientific_role_identity_fields(role),
                )
            for configuration in _dflash_parameter_configurations():
                geometry_gpu = _paired_gpu_index(
                    factory.seed,
                    {
                        "experiment": "E1",
                        "selection": selection,
                        "configuration": configuration.sha256,
                    },
                )
                for optimizer in E1_OPTIMIZER_ANCHORS:
                    factory.add(
                        experiment="E1",
                        model="Qwen/Qwen3-8B",
                        backend="DFLASH",
                        task="LiveCodeBench_tuning",
                        method="l0",
                        workload_class=WorkloadClass.TUNING,
                        scope=configuration.scope,
                        rank=configuration.rank,
                        alpha_over_rank=configuration.alpha_over_rank,
                        optimizer=optimizer,
                        schedule="constant",
                        context=40928,
                        regime="short_input_long_generation",
                        width=width,
                        arrival="locked_reference_load",
                        slo="tuning_safety",
                        concurrency=concurrency,
                        parameterization=configuration.parameterization,
                        variant=f"lc_candidate:optimizer_anchor:{selection}",
                        gpu_index=geometry_gpu,
                    )


def _dflash_parameter_configurations() -> tuple[ParameterConfiguration, ...]:
    configurations: list[ParameterConfiguration] = []
    for scope in E1_SCOPES:
        configurations.append(
            ParameterConfiguration(scope, "full", None, None, "not_applicable")
        )
        configurations.extend(
            ParameterConfiguration(scope, "lora", rank, 1.0, "not_applicable")
            for rank in LORA_RANKS
        )
    return tuple(configurations)


def _optimizer_learning_rates(
    optimizer: str, parameterization: str
) -> tuple[float, ...]:
    if parameterization not in {"full", "lora"}:
        raise ValueError("unknown parameterization for optimizer grid")
    if parameterization == "lora":
        return {
            "sgdm": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
            "nag": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
            "lion": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
            "muon": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        }.get(optimizer, (1e-5, 3e-5, 1e-4, 3e-4, 1e-3))
    return {
        "sgdm": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
        "nag": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
        "lion": (1e-8, 3e-8, 1e-7, 3e-7, 1e-6),
        "muon": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
    }.get(optimizer, (1e-7, 3e-7, 1e-6, 3e-6, 1e-5))


def _add_e2_cells(factory: _CellFactory) -> None:
    for stage_index, (batch_size, context) in enumerate(E2_HALVING_STAGES):
        stage = f"halving_stage={stage_index}:batch={batch_size}:min_updates=locked_e1"
        reference_gpu = _paired_gpu_index(
            factory.seed,
            {"experiment": "E2", "stage": stage, "kind": "reference"},
        )
        _add_reference_baselines(
            factory,
            experiment="E2",
            backend="DFLASH",
            task="LiveCodeBench_tuning",
            context=context,
            arrival="e1_common_load",
            variant=f"{stage}:reference_baseline",
            width=None,
            gpu_index=reference_gpu,
        )
        anchor_gpu = _paired_gpu_index(
            factory.seed,
            {"experiment": "E2", "stage": stage, "kind": "frozen_tts_anchor"},
        )
        for role in (
            ScientificMethodRole.TTS.value,
            ScientificMethodRole.L0_NAIVE.value,
        ):
            factory.add(
                experiment="E2",
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="LiveCodeBench_tuning",
                workload_class=WorkloadClass.TUNING,
                context=context,
                regime="short_input_long_generation",
                width=None,
                arrival="e1_common_load",
                slo="tuning_safety",
                variant=f"{stage}:frozen_tts_anchor:role={role}",
                gpu_index=anchor_gpu,
                **_scientific_role_identity_fields(role),
            )
        for configuration in _dflash_parameter_configurations():
            for optimizer in E2_OPTIMIZERS:
                for schedule in E2_SCHEDULES:
                    learning_rates = _optimizer_learning_rates(
                        optimizer, configuration.parameterization
                    )
                    for learning_rate in learning_rates:
                        pair_gpu = _paired_gpu_index(
                            factory.seed,
                            {
                                "experiment": "E2",
                                "stage": stage,
                                "configuration": configuration.sha256,
                                "optimizer": optimizer,
                                "schedule": schedule,
                                "learning_rate": learning_rate,
                            },
                        )
                        factory.add(
                            experiment="E2",
                            model="Qwen/Qwen3-8B",
                            backend="DFLASH",
                            task="LiveCodeBench_tuning",
                            method="l0",
                            workload_class=WorkloadClass.TUNING,
                            scope=configuration.scope,
                            rank=configuration.rank,
                            alpha_over_rank=configuration.alpha_over_rank,
                            optimizer=optimizer,
                            learning_rate=learning_rate,
                            schedule=schedule,
                            context=context,
                            regime="short_input_long_generation",
                            width=None,
                            arrival="e1_common_load",
                            slo="tuning_safety",
                            parameterization=configuration.parameterization,
                            variant=f"{stage}:lc_candidate:optimizer_specific_log_lr",
                            gpu_index=pair_gpu,
                            status=CellStatus.BLOCKED,
                            reason_code="adaptation_recipe_values_unregistered",
                            reason=(
                                "The E2 recipe declaration retains named unresolved "
                                "optimizer, stride, or E3a-selected width semantics; "
                                "schema defaults cannot authorize execution."
                            ),
                        )


def _add_e4_cells(factory: _CellFactory) -> None:
    for variant, role in (
        ("synchronous_main_stream", ScientificMethodRole.TTS.value),
        ("side_stream_tts", ScientificMethodRole.TTS.value),
        ("l0_first_ready", ScientificMethodRole.L0_NAIVE.value),
        ("cohort_batch", ScientificMethodRole.LIGHTCONE.value),
        ("fixed_buffer_graph", ScientificMethodRole.LIGHTCONE.value),
        ("bounded_telemetry", ScientificMethodRole.LIGHTCONE.value),
    ):
        identity_fields = _scientific_role_identity_fields(role)
        for load in ("low", "moderate", "saturation"):
            for traffic in ("pure_decode", "mixed_prefill_decode"):
                for chunked_prefill in ("disabled", "enabled"):
                    for prefix_reuse in ("none", "shared"):
                        factory.add(
                            experiment="E4",
                            model="Qwen/Qwen3-8B",
                            backend="DFLASH",
                            task="systems_ablation",
                            workload_class=WorkloadClass.TUNING,
                            context=40928,
                            regime=traffic,
                            width=DRAFT_WIDTHS[1],
                            arrival=(
                                f"load={load}:chunked_prefill={chunked_prefill}:"
                                f"prefix_reuse={prefix_reuse}:"
                                "operational_grid=locked_e2"
                            ),
                            slo="critical_path",
                            variant=variant,
                            **identity_fields,
                        )
    for profiler in ("nvtx", "nsight_systems", "nsight_compute"):
        factory.add(
            experiment="E4",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="isolated_profile",
            workload_class=WorkloadClass.PROFILE,
            gpu_count=2,
            context=40928,
            regime="mixed_prefill_decode",
            width=DRAFT_WIDTHS[1],
            arrival="isolated_profile",
            slo="headline_evidence_forbidden",
            topology="two_gpu_host_exclusive",
            variant=profiler,
            **_scientific_role_identity_fields(ScientificMethodRole.LIGHTCONE.value),
        )


def _add_e3b_cells(factory: _CellFactory) -> None:
    for block in REGISTERED_CONFIRMATION_BLOCKS:
        phase = "excluded_pilot" if block in PILOT_BLOCKS else "final_candidate"
        for role_index, role in enumerate(CONFIRMATION_METHOD_ROLES):
            identity_fields = _scientific_role_identity_fields(role)
            for context in CONTEXT_GRID:
                for regime in CONTEXT_REGIMES:
                    for load in ("concurrency_one", "common_load"):
                        for width_panel in ("matched", "deployment_optimal"):
                            factory.add(
                                experiment="E3b",
                                model="Qwen/Qwen3-8B",
                                backend=(
                                    "NONE"
                                    if role == ScientificMethodRole.TARGET_ONLY.value
                                    else "DFLASH"
                                ),
                                task="heldout_long_context_confirmation",
                                workload_class=WorkloadClass.HEADLINE,
                                gpu_index=(role_index + block) % 2,
                                context=context,
                                regime=regime,
                                arrival=(
                                    "closed_loop_c1"
                                    if load == "concurrency_one"
                                    else "closed_loop_common_load"
                                ),
                                slo="paired_long_context_confirmation",
                                block=block,
                                variant=f"{phase}:{load}:{width_panel}:role={role}",
                                **identity_fields,
                            )


def _add_e1a_cells(factory: _CellFactory) -> None:
    for configuration in e1a_adaptive_configurations():
        for method in ("l0",):
            factory.add(
                experiment="E1a",
                model="Qwen/Qwen3-8B",
                backend="DSPARK",
                task="LiveCodeBench_tuning_disjoint_from_E5",
                method=method,
                workload_class=WorkloadClass.TUNING,
                scope=configuration.scope,
                rank=configuration.rank,
                alpha_over_rank=configuration.alpha_over_rank,
                optimizer=SEALED_E2_RECIPE_SENTINEL,
                schedule=SEALED_E2_RECIPE_SENTINEL,
                context=40928,
                regime="short_input_long_generation",
                width=DRAFT_WIDTHS[1],
                arrival="fixed_budget_then_native_scheduler",
                slo="confidence_head_guard",
                parameterization=configuration.parameterization,
                variant=(
                    "sealed_lightcone_recipe:"
                    f"native_heads:{configuration.native_head_policy}"
                ),
            )
    _add_reference_baselines(
        factory,
        experiment="E1a",
        backend="DSPARK",
        task="LiveCodeBench_tuning_disjoint_from_E5",
        context=40928,
    )


def _add_e5_block_cells(factory: _CellFactory, block: int) -> None:
    phase = "excluded_pilot" if block in PILOT_BLOCKS else "final_candidate"
    for backend in ("DFLASH", "DSPARK"):
        for role_index, role in enumerate(CONFIRMATION_METHOD_ROLES):
            identity_fields = _scientific_role_identity_fields(role)
            for concurrency in E5_CLOSED_LOOP_CONCURRENCY:
                factory.add(
                    experiment="E5",
                    model="Qwen/Qwen3-8B",
                    backend=backend,
                    task="production_crossover",
                    workload_class=WorkloadClass.HEADLINE,
                    gpu_index=(role_index + block) % 2,
                    context=40928,
                    regime="production_mix",
                    arrival="closed_loop",
                    slo=PRODUCTION_SLO,
                    block=block,
                    concurrency=concurrency,
                    variant=f"{phase}:E5a_closed_loop:role={role}",
                    **identity_fields,
                )
            for load_factor in E5_OPEN_LOOP_LOAD_FACTORS:
                factory.add(
                    experiment="E5",
                    model="Qwen/Qwen3-8B",
                    backend=backend,
                    task="production_crossover",
                    workload_class=WorkloadClass.HEADLINE,
                    gpu_index=(role_index + block) % 2,
                    context=40928,
                    regime="production_mix",
                    arrival="poisson",
                    slo=PRODUCTION_SLO,
                    block=block,
                    load_factor=load_factor,
                    variant=f"{phase}:E5a_open_loop_lambda_star:role={role}",
                    **identity_fields,
                )
            for arrival in (
                "immediate_burst",
                "burstgpt_shape",
                "moderate_soak",
                "saturation_soak",
                "overload_soak",
            ):
                factory.add(
                    experiment="E5",
                    model="Qwen/Qwen3-8B",
                    backend=backend,
                    task="production_crossover",
                    workload_class=WorkloadClass.HEADLINE,
                    gpu_index=(role_index + block) % 2,
                    context=40928,
                    regime="production_mix",
                    arrival=arrival,
                    slo=PRODUCTION_SLO,
                    block=block,
                    variant=f"{phase}:E5a_trace_or_soak:role={role}",
                    status=CellStatus.UNMEASURED,
                    reason_code="awaiting_registered_measurement",
                    reason="No complete content-bound measurement exists.",
                    **identity_fields,
                )

    for backend in ("DFLASH", "DSPARK"):
        for role_index, role in enumerate(CONFIRMATION_METHOD_ROLES):
            identity_fields = _scientific_role_identity_fields(role)
            for topology in E5_TOPOLOGIES:
                for cohort_count in E5_COHORT_COUNTS:
                    for distribution in E5_COHORT_DISTRIBUTIONS:
                        factory.add(
                            experiment="E5",
                            model="Qwen/Qwen3-8B",
                            backend=backend,
                            task="topology_cohort_capacity",
                            workload_class=WorkloadClass.HEADLINE,
                            gpu_count=1 if topology == "tp1_dp1" else 2,
                            gpu_index=(
                                (role_index + block) % 2
                                if topology == "tp1_dp1"
                                else None
                            ),
                            context=40928,
                            regime="production_mix",
                            arrival="saturation_anchor",
                            slo=PRODUCTION_SLO,
                            block=block,
                            cohort=f"K={cohort_count}:{distribution}",
                            topology=topology,
                            variant=f"{phase}:E5b_topology_cohort:role={role}",
                            cohort_count=cohort_count,
                            **identity_fields,
                        )
    for failure in E5_FAILURES:
        factory.add(
            experiment="E5",
            model="Qwen/Qwen3-8B",
            backend="DFLASH+DSPARK",
            task="failure_injection",
            workload_class=WorkloadClass.CORRECTNESS,
            gpu_count=2,
            context=40928,
            regime="production_mix",
            arrival=f"failure:{failure}",
            slo="excluded_from_headline",
            topology="tp2_and_two_replica",
            block=block,
            variant=f"{phase}:E5b_failure:role=lightcone",
            **_scientific_role_identity_fields(ScientificMethodRole.LIGHTCONE.value),
        )


def _add_e5_cells(factory: _CellFactory) -> None:
    for block in REGISTERED_CONFIRMATION_BLOCKS:
        _add_e5_block_cells(factory, block)


def _add_e6_cells(factory: _CellFactory) -> None:
    headline_roles = (
        ScientificMethodRole.TARGET_ONLY.value,
        ScientificMethodRole.STATIC.value,
        ScientificMethodRole.TTS.value,
        ScientificMethodRole.LIGHTCONE.value,
    )
    for model in E6_CANDIDATE_MODELS:
        factory.add(
            experiment="E6",
            model=model,
            backend="NEXTN",
            task="immutable_metadata_and_fit_preflight",
            method="target_only",
            workload_class=WorkloadClass.DOWNLOAD,
            gpu_count=2,
            topology="tp2_dp1",
            variant="metadata_interface_hash_and_download_gate",
        )
        for task in ("LiveCodeBench", "MATH-500"):
            for context in (4096, 16384, 32768):
                for load in ("concurrency_one", "common_slo_load"):
                    for block in REGISTERED_CONFIRMATION_BLOCKS:
                        phase = (
                            "excluded_pilot"
                            if block in PILOT_BLOCKS
                            else "final_candidate"
                        )
                        for role in headline_roles:
                            factory.add(
                                experiment="E6",
                                model=model,
                                backend=(
                                    "NONE"
                                    if role == ScientificMethodRole.TARGET_ONLY.value
                                    else "NEXTN"
                                ),
                                task=task,
                                workload_class=WorkloadClass.HEADLINE,
                                gpu_count=2,
                                context=context,
                                regime="native_mtp_transfer",
                                arrival=(
                                    "closed_loop_c1"
                                    if load == "concurrency_one"
                                    else "common_slo_load"
                                ),
                                slo=PRODUCTION_SLO,
                                topology="tp2_dp1",
                                block=block,
                                variant=(
                                    f"{phase}:{load}:compatibility_transfer:role={role}"
                                ),
                                status=CellStatus.BLOCKED,
                                reason_code="native_nextn_preflight_required",
                                reason=(
                                    "The exact NEXTN interface and two-rank memory fit "
                                    "have not passed their registered preflight."
                                ),
                                **_scientific_role_identity_fields(role),
                            )
        for context in (16384, 32768):
            for load in ("concurrency_one", "common_slo_load"):
                for block in REGISTERED_CONFIRMATION_BLOCKS:
                    phase = (
                        "excluded_pilot" if block in PILOT_BLOCKS else "final_candidate"
                    )
                    factory.add(
                        experiment="E6",
                        model=model,
                        backend="NEXTN",
                        task="LiveCodeBench",
                        workload_class=WorkloadClass.HEADLINE,
                        gpu_count=2,
                        context=context,
                        regime="native_mtp_transfer",
                        arrival=(
                            "closed_loop_c1"
                            if load == "concurrency_one"
                            else "common_slo_load"
                        ),
                        slo=PRODUCTION_SLO,
                        topology="tp2_dp1",
                        block=block,
                        variant=(
                            f"{phase}:{load}:largest_feasible_model_anchor_template:"
                            "role=l0_naive"
                        ),
                        status=CellStatus.BLOCKED,
                        reason_code="largest_feasible_model_selection_required",
                        reason=(
                            "The preregistered feasibility reducer must select the "
                            "largest feasible model before this L0-naive anchor."
                        ),
                        **_scientific_role_identity_fields(
                            ScientificMethodRole.L0_NAIVE.value
                        ),
                    )


def _add_e0_cells(factory: _CellFactory) -> None:
    # Keep the complete breadth compatibility universe structural.  It is not
    # an interval-bearing interaction grid and must not be promoted merely
    # because a later selected anchor is measured.
    for model in E0_MODELS:
        for backend in E0_BACKENDS:
            for task in E0_TASKS:
                for role in E0_METHOD_ROLES:
                    factory.add(
                        experiment="E0",
                        model=model,
                        backend=backend,
                        task=task,
                        workload_class=WorkloadClass.HEADLINE,
                        context=4096,
                        regime="breadth_replication",
                        arrival="deterministic_stratified_requests",
                        slo="breadth_timing",
                        variant=f"compatibility_template:role={role}",
                        status=CellStatus.BLOCKED,
                        reason_code="compatibility_template_only",
                        reason=(
                            "This breadth template cannot be reported before its exact "
                            "recipe and compatibility authorities are materialized."
                        ),
                        **_scientific_role_identity_fields(role),
                    )

    for model in E0_INTERACTION_MODELS:
        for backend in E0_INTERACTION_BACKENDS:
            for task in E0_INTERACTION_TASKS:
                for load in E0_LOADS:
                    for block in (*PILOT_BLOCKS, *E0_INTERACTION_FINAL_BLOCKS):
                        phase = (
                            "excluded_pilot"
                            if block in PILOT_BLOCKS
                            else "final_candidate"
                        )
                        for role in CONFIRMATION_METHOD_ROLES:
                            factory.add(
                                experiment="E0",
                                model=model,
                                backend=backend,
                                task=task,
                                workload_class=WorkloadClass.HEADLINE,
                                context=4096,
                                regime="breadth_replication",
                                arrival=(
                                    "closed_loop_c1"
                                    if load == "concurrency_one"
                                    else "common_slo_load"
                                ),
                                slo="breadth_timing",
                                block=block,
                                variant=(
                                    f"{phase}:{load}:compatibility_template:role={role}"
                                ),
                                status=CellStatus.BLOCKED,
                                reason_code="compatibility_template_only",
                                reason=(
                                    "This preregistered breadth slot cannot be reported "
                                    "before its exact recipe, load, compatibility, and "
                                    "content-bound evidence authorities are materialized."
                                ),
                                **_scientific_role_identity_fields(role),
                            )


def build_legacy_industrial_registry(
    *,
    gpu_uuids: tuple[str, ...] = (
        "logical-rank-slot-0",
        "logical-rank-slot-1",
    ),
    base_port: int = 24000,
    cache_root: str = "runtime-cache/industrial",
    evidence_root: str = "artifacts/industrial",
    seed: int = 20260811,
) -> ExperimentRegistry:
    """Build the eager historical registry for explicit diagnostics only.

    Formal execution must use signed staged materialization receipts.  This
    compatibility builder remains available to inspect and migrate historical
    schema-v3 evidence; it is not a formal experiment authority.
    """

    port_span = _industrial_port_span(len(gpu_uuids))
    if base_port + port_span - 1 > 65_535:
        raise ValueError(
            "base_port cannot fit the complete collision-free industrial port span"
        )

    factory = _CellFactory(
        gpu_uuids=gpu_uuids,
        base_port=base_port,
        cache_root=cache_root,
        evidence_root=evidence_root,
        seed=seed,
    )
    _add_preflight_cells(factory)
    _add_e3a_cells(factory)
    _add_tts_cal_cells(factory)
    _add_e1_cells(factory)
    _add_e2_cells(factory)
    _add_e4_cells(factory)
    _add_e3b_cells(factory)
    _add_e1a_cells(factory)
    _add_e5_cells(factory)
    _add_e6_cells(factory)
    _add_e0_cells(factory)
    registry = ExperimentRegistry(
        schema_version=3,
        name="lightcone-industrial-experiment-registry",
        gpu_uuids=gpu_uuids,
        definitions=_industrial_definitions(),
        cells=tuple(factory.cells),
        materialization_mode="legacy_diagnostic",
    )
    used_ports = {port for cell in registry.cells for port in cell.resources.ports}
    if used_ports != set(range(base_port, base_port + port_span)):
        raise AssertionError("industrial port-pool declaration is out of date")
    return registry


def build_industrial_registry(
    *,
    gpu_uuids: tuple[str, ...] = (
        "logical-rank-slot-0",
        "logical-rank-slot-1",
    ),
    base_port: int = 24000,
    cache_root: str = "runtime-cache/industrial",
    evidence_root: str = "artifacts/industrial",
    seed: int = 20260811,
) -> ExperimentRegistry:
    """Build only the concrete preregistration prefix for signed staging.

    Future stages are absent rather than represented by BLOCKED/sentinel rows.
    They can exist only in signed :class:`StageMaterializationReceipt` values.
    """

    port_span = _industrial_port_span(len(gpu_uuids))
    if base_port + port_span - 1 > 65_535:
        raise ValueError(
            "base_port cannot fit the complete collision-free industrial port span"
        )
    factory = _CellFactory(
        gpu_uuids=gpu_uuids,
        base_port=base_port,
        cache_root=cache_root,
        evidence_root=evidence_root,
        seed=seed,
    )
    _add_preflight_cells(factory)
    _add_e3a_cells(factory)
    _add_tts_cal_cells(factory)
    registry = ExperimentRegistry(
        schema_version=3,
        name="lightcone-formal-staged-registry",
        gpu_uuids=gpu_uuids,
        definitions=_industrial_definitions(),
        cells=tuple(factory.cells),
        materialization_mode="signed_staged",
    )
    used_ports = {port for cell in registry.cells for port in cell.resources.ports}
    if used_ports != set(range(base_port, base_port + port_span)):
        raise AssertionError("staged registry port-pool declaration is out of date")
    return registry
