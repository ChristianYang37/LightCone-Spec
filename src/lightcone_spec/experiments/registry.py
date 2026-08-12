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
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "Qwen/Qwen3.6-35B-A3B",
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
CORE_METHODS = ("target_only", "static", "tts", "l0")
E0_METHODS = CORE_METHODS + (
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
)

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

    TTS and L0 intentionally share a key.  E1 keys carry an exact width.  E2
    keys carry a selector slot instead, because the width is an E3a output and
    is not an optimizer-grid axis.
    """

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
            "experiment",
            "backend",
            "scope",
            "parameterization",
            "optimizer",
            "schedule",
            "cohort",
        ):
            _require_text(f"adaptation recipe {name}", getattr(self, name))
        if self.experiment not in {"E1", "E2"}:
            raise ValueError("adaptation recipes are registered only for E1/E2")
        if self.backend != "DFLASH":
            raise ValueError("E1/E2 adaptation recipes require DFLASH")
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
        if identity.experiment not in {"E1", "E2"} or identity.method not in {
            "tts",
            "l0",
        }:
            raise ValueError("adaptation recipe lookup requires an E1/E2 TTS/L0 cell")
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
        "schedule_total_published_updates",
    }
)


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
    schedule: str
    schedule_total_published_updates: int | None
    unresolved_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("optimizer recipe name", self.name)
        _require_text("optimizer recipe schedule", self.schedule)
        if self.schedule not in E2_SCHEDULES:
            raise ValueError("optimizer recipe schedule is outside the registry")
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
            if value is not None and not math.isfinite(value):
                raise ValueError(f"optimizer recipe {name} must be finite")
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
    {"stride", "canvas_tokens", "extra_logical_delay"}
)


@dataclass(frozen=True)
class AdaptationRecipeDeclaration:
    """Registry-owned, content-bound declaration of full adaptation semantics."""

    schema_version: int
    lookup_key: AdaptationRecipeLookupKey
    source_authority: str
    source_authority_sha256: str
    weight_update_mode: str
    parameter_scope: str
    kv_history_policy: str
    adaptation_scope: str
    adaptation_group_id: str
    optimizer: OptimizerRecipeDeclaration
    rank: int | None
    lora_alpha: int | None
    lora_matrix_policy: str
    native_head_policy: str
    stride: int | None
    max_in_flight: int
    canvas_tokens: int | None
    loss_position_decay: float
    extra_logical_delay: int | None
    teacher_row_policy: str
    verification_mode: str
    fixed_verification_budget: int | None
    confidence_loss_weight: float | None
    status: str
    blocker_codes: tuple[str, ...]
    unresolved_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only adaptation recipe declaration schema 1 is supported")
        for name in (
            "source_authority",
            "weight_update_mode",
            "parameter_scope",
            "kv_history_policy",
            "adaptation_scope",
            "adaptation_group_id",
            "lora_matrix_policy",
            "native_head_policy",
            "teacher_row_policy",
            "verification_mode",
        ):
            _require_text(f"adaptation recipe {name}", getattr(self, name))
        if not _LOWER_SHA256(self.source_authority_sha256):
            raise ValueError("recipe source authority must be content-bound")
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
        elif self.lookup_key.learning_rate != self.optimizer.learning_rate:
            raise ValueError("recipe learning rate differs from its lookup key")
        if self.lookup_key.schedule != self.optimizer.schedule:
            raise ValueError("recipe schedule differs from its lookup key")
        if self.lookup_key.cohort != self.adaptation_group_id:
            raise ValueError("recipe cohort differs from its lookup key")
        if self.lookup_key.draft_width is not None:
            if self.canvas_tokens != self.lookup_key.draft_width:
                raise ValueError("exact-width recipe canvas differs from its key")
        elif (
            self.canvas_tokens is not None
            or "canvas_tokens" not in self.unresolved_fields
        ):
            raise ValueError("selected-width recipes must retain a null canvas slot")
        if self.unresolved_fields != tuple(sorted(set(self.unresolved_fields))):
            raise ValueError("adaptation unresolved fields must be sorted and unique")
        if not set(self.unresolved_fields) <= _ADAPTATION_RECIPE_FIELDS:
            raise ValueError("adaptation recipe names an unknown unresolved field")
        for name in self.unresolved_fields:
            if getattr(self, name) is not None:
                raise ValueError("an unresolved adaptation field must remain null")
        if not math.isfinite(self.loss_position_decay) or not (
            0 < self.loss_position_decay <= 1
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
            # This is deliberately an all-fields call.  It proves that no
            # Pydantic default is needed to turn a declaration into a config.
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

    def to_adaptation_config(self):
        """Materialize an available declaration with no implicit field values."""

        if self.status != "AVAILABLE":
            raise ValueError(
                "adaptation recipe is BLOCKED: " + ",".join(self.blocker_codes)
            )
        if (
            self.unresolved_fields
            or self.stride is None
            or self.canvas_tokens is None
            or self.extra_logical_delay is None
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
        )

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
    optimizer = OptimizerRecipeDeclaration(
        name=candidate.optimizer,
        learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
        beta1=candidate.beta1,
        beta2=candidate.beta2,
        epsilon=1e-8,
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
        status="AVAILABLE",
        blocker_codes=(),
    )


def _e2_optimizer_declaration(
    key: AdaptationRecipeLookupKey,
) -> tuple[OptimizerRecipeDeclaration, tuple[str, ...]]:
    optimizer = key.optimizer
    unresolved: set[str] = {
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "grad_clip",
    }
    blockers: set[str] = {
        "e2_weight_decay_unregistered",
        "e2_beta_values_unregistered",
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

    if optimizer == "chronobelief":
        unresolved.update(_OPTIMIZER_RECIPE_FIELDS)
        blockers.add("chronobelief_equation_unregistered")
    else:
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
            blockers.add("e2_muon_parameters_unregistered")
        if key.schedule == "cosine_to_zero":
            unresolved.add("schedule_total_published_updates")
            blockers.add("e2_cosine_horizon_unregistered")
    values = {
        "name": optimizer,
        "learning_rate": key.learning_rate,
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
    # ChronoBelief has no registered LR grid.  Other E2 keys must already bind
    # the exact optimizer-specific logarithmic value from the registry.
    if optimizer == "chronobelief":
        values["learning_rate"] = None
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
                "e2_draft_width_selector_unresolved",
                "e2_extra_logical_delay_unregistered",
                "e2_update_stride_unregistered",
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
        teacher_row_policy="update_round",
        verification_mode="native_scheduler",
        fixed_verification_budget=None,
        confidence_loss_weight=None,
        status="BLOCKED",
        blocker_codes=blockers,
        unresolved_fields=("canvas_tokens", "extra_logical_delay", "stride"),
    )


def _build_adaptation_recipe_declarations(
    cells: Sequence[ExperimentCell],
) -> tuple[AdaptationRecipeDeclaration, ...]:
    keys: dict[str, AdaptationRecipeLookupKey] = {}
    for cell in cells:
        if cell.identity.experiment not in {"E1", "E2"} or cell.identity.method not in {
            "tts",
            "l0",
        }:
            continue
        key = AdaptationRecipeLookupKey.from_cell(cell)
        keys[key.sha256] = key
    declarations = tuple(
        _e1_recipe_declaration(key)
        if key.experiment == "E1"
        else _e2_recipe_declaration(key)
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
            name="E1",
            dependencies=("E3a",),
            locked_outputs=("dflash_pareto_set", "common_downstream_load"),
            axes=(
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
                _axis("method", CORE_METHODS),
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
                _axis("method", CORE_METHODS),
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
                _axis("method", CORE_METHODS),
                _axis("task", ("LiveCodeBench", "MATH-500")),
                _axis("context", (4096, 16384, 32768)),
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
                _axis("method", E0_METHODS),
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

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only industrial registry schema version 2 is supported")
        _require_text("registry name", self.name)
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
        if seen_stages != known:
            raise ValueError("every experiment must have at least one declared cell")
        declarations = {
            row.lookup_key.sha256: row for row in self.adaptation_recipe_declarations
        }
        for cell in self.cells:
            if cell.identity.experiment not in {
                "E1",
                "E2",
            } or cell.identity.method not in {
                "tts",
                "l0",
            }:
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
        return {
            "schema_version": self.schema_version,
            "name": self.name,
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

        if isinstance(cell_or_id, str):
            matches = tuple(cell for cell in self.cells if cell.cell_id == cell_or_id)
            if len(matches) != 1:
                raise ValueError(
                    "adaptation recipe cell ID is absent from the registry"
                )
            cell = matches[0]
        elif isinstance(cell_or_id, ExperimentCell):
            matches = tuple(
                candidate
                for candidate in self.cells
                if candidate.cell_id == cell_or_id.cell_id
            )
            if len(matches) != 1 or matches[0] != cell_or_id:
                raise ValueError("adaptation recipe cell is not registry-owned")
            cell = cell_or_id
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
            for scope in E1_SCOPES:
                for optimizer in E1_OPTIMIZER_ANCHORS:
                    full_gpu = _paired_gpu_index(
                        factory.seed,
                        {
                            "experiment": "E1",
                            "selection": selection,
                            "scope": scope,
                            "optimizer": optimizer,
                            "parameterization": "full",
                        },
                    )
                    for method in ("tts", "l0"):
                        factory.add(
                            experiment="E1",
                            model="Qwen/Qwen3-8B",
                            backend="DFLASH",
                            task="LiveCodeBench_tuning",
                            method=method,
                            workload_class=WorkloadClass.TUNING,
                            scope=scope,
                            optimizer=optimizer,
                            schedule="constant",
                            context=40928,
                            regime="short_input_long_generation",
                            width=width,
                            arrival="locked_reference_load",
                            slo="tuning_safety",
                            concurrency=concurrency,
                            parameterization="full",
                            variant=f"optimizer_anchor:{selection}",
                            gpu_index=full_gpu,
                        )
                    for rank in LORA_RANKS:
                        lora_gpu = _paired_gpu_index(
                            factory.seed,
                            {
                                "experiment": "E1",
                                "selection": selection,
                                "scope": scope,
                                "optimizer": optimizer,
                                "parameterization": "lora",
                                "rank": rank,
                            },
                        )
                        for method in ("tts", "l0"):
                            factory.add(
                                experiment="E1",
                                model="Qwen/Qwen3-8B",
                                backend="DFLASH",
                                task="LiveCodeBench_tuning",
                                method=method,
                                workload_class=WorkloadClass.TUNING,
                                scope=scope,
                                rank=rank,
                                alpha_over_rank=1.0,
                                optimizer=optimizer,
                                schedule="constant",
                                context=40928,
                                regime="short_input_long_generation",
                                width=width,
                                arrival="locked_reference_load",
                                slo="tuning_safety",
                                concurrency=concurrency,
                                parameterization="lora",
                                variant=f"optimizer_anchor:{selection}",
                                gpu_index=lora_gpu,
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
    if optimizer == "chronobelief":
        return ()
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
        for configuration in _dflash_parameter_configurations():
            for optimizer in E2_OPTIMIZERS:
                for schedule in E2_SCHEDULES:
                    learning_rates = _optimizer_learning_rates(
                        optimizer, configuration.parameterization
                    )
                    if not learning_rates:
                        pair_gpu = _paired_gpu_index(
                            factory.seed,
                            {
                                "experiment": "E2",
                                "stage": stage,
                                "configuration": configuration.sha256,
                                "optimizer": optimizer,
                                "schedule": schedule,
                                "learning_rate": None,
                            },
                        )
                        for method in ("tts", "l0"):
                            factory.add(
                                experiment="E2",
                                model="Qwen/Qwen3-8B",
                                backend="DFLASH",
                                task="LiveCodeBench_tuning",
                                method=method,
                                workload_class=WorkloadClass.TUNING,
                                scope=configuration.scope,
                                rank=configuration.rank,
                                alpha_over_rank=configuration.alpha_over_rank,
                                optimizer=optimizer,
                                schedule=schedule,
                                context=context,
                                regime="short_input_long_generation",
                                width=None,
                                arrival="e1_common_load",
                                slo="tuning_safety",
                                parameterization=configuration.parameterization,
                                variant=f"{stage}:optimizer_equation_unresolved",
                                gpu_index=pair_gpu,
                                status=CellStatus.BLOCKED,
                                reason_code="optimizer_equation_unresolved",
                                reason=(
                                    "No authoritative ChronoBelief update equation or "
                                    "source identity is registered; substitution is "
                                    "forbidden."
                                ),
                            )
                        continue
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
                        for method in ("tts", "l0"):
                            factory.add(
                                experiment="E2",
                                model="Qwen/Qwen3-8B",
                                backend="DFLASH",
                                task="LiveCodeBench_tuning",
                                method=method,
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
                                variant=f"{stage}:optimizer_specific_log_lr",
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
    for variant, method in (
        ("synchronous_main_stream", "tts"),
        ("side_stream_tts", "tts"),
        ("l0_first_ready", "l0"),
        ("cohort_batch", "l0"),
        ("fixed_buffer_graph", "l0"),
        ("bounded_telemetry", "l0"),
    ):
        for load in ("low", "moderate", "saturation"):
            for traffic in ("pure_decode", "mixed_prefill_decode"):
                for chunked_prefill in ("disabled", "enabled"):
                    for prefix_reuse in ("none", "shared"):
                        factory.add(
                            experiment="E4",
                            model="Qwen/Qwen3-8B",
                            backend="DFLASH",
                            task="systems_ablation",
                            method=method,
                            workload_class=WorkloadClass.TUNING,
                            scope="selected_e2",
                            optimizer="selected_e2",
                            schedule="selected_e2",
                            context=40928,
                            regime=traffic,
                            width=DRAFT_WIDTHS[1],
                            arrival=(
                                f"load={load}:chunked_prefill={chunked_prefill}:"
                                f"prefix_reuse={prefix_reuse}:"
                                "operational_grid=locked_e2"
                            ),
                            slo="critical_path",
                            parameterization="selected",
                            variant=variant,
                        )
    for profiler in ("nvtx", "nsight_systems", "nsight_compute"):
        factory.add(
            experiment="E4",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="isolated_profile",
            method="l0",
            workload_class=WorkloadClass.PROFILE,
            gpu_count=2,
            scope="selected_e2",
            optimizer="selected_e2",
            schedule="selected_e2",
            context=40928,
            regime="mixed_prefill_decode",
            width=DRAFT_WIDTHS[1],
            arrival="isolated_profile",
            slo="headline_evidence_forbidden",
            topology="two_gpu_host_exclusive",
            parameterization="selected",
            variant=profiler,
        )


def _add_e3b_cells(factory: _CellFactory) -> None:
    for block in REGISTERED_CONFIRMATION_BLOCKS:
        phase = "excluded_pilot" if block in PILOT_BLOCKS else "final_candidate"
        for method in CORE_METHODS:
            for context in CONTEXT_GRID:
                for regime in CONTEXT_REGIMES:
                    for load in ("concurrency_one", "common_load"):
                        for width_panel in ("matched", "deployment_optimal"):
                            factory.add(
                                experiment="E3b",
                                model="Qwen/Qwen3-8B",
                                backend=(
                                    "NONE" if method == "target_only" else "DFLASH"
                                ),
                                task="heldout_long_context_confirmation",
                                method=method,
                                workload_class=WorkloadClass.HEADLINE,
                                gpu_index=(CORE_METHODS.index(method) + block) % 2,
                                scope=(
                                    "selected_e2" if method in {"tts", "l0"} else "none"
                                ),
                                optimizer=(
                                    "selected_e2" if method in {"tts", "l0"} else None
                                ),
                                schedule=(
                                    "selected_e2" if method in {"tts", "l0"} else None
                                ),
                                context=context,
                                regime=regime,
                                arrival=(
                                    "closed_loop_c1"
                                    if load == "concurrency_one"
                                    else "closed_loop_common_load"
                                ),
                                slo="paired_long_context_confirmation",
                                block=block,
                                parameterization=(
                                    "selected" if method in {"tts", "l0"} else "none"
                                ),
                                variant=f"{phase}:{load}:{width_panel}",
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
                optimizer="transferred_e2",
                schedule="transferred_e2",
                context=40928,
                regime="short_input_long_generation",
                width=DRAFT_WIDTHS[1],
                arrival="fixed_budget_then_native_scheduler",
                slo="confidence_head_guard",
                parameterization=configuration.parameterization,
                variant=f"native_heads:{configuration.native_head_policy}",
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
        for method in CORE_METHODS:
            for concurrency in E5_CLOSED_LOOP_CONCURRENCY:
                factory.add(
                    experiment="E5",
                    model="Qwen/Qwen3-8B",
                    backend=backend,
                    task="production_crossover",
                    method=method,
                    workload_class=WorkloadClass.HEADLINE,
                    gpu_index=(CORE_METHODS.index(method) + block) % 2,
                    scope="selected_recipe" if method in {"tts", "l0"} else "none",
                    context=40928,
                    regime="production_mix",
                    arrival="closed_loop",
                    slo=PRODUCTION_SLO,
                    block=block,
                    concurrency=concurrency,
                    parameterization=(
                        "selected" if method in {"tts", "l0"} else "none"
                    ),
                    variant=f"{phase}:E5a_closed_loop",
                )
            for load_factor in E5_OPEN_LOOP_LOAD_FACTORS:
                factory.add(
                    experiment="E5",
                    model="Qwen/Qwen3-8B",
                    backend=backend,
                    task="production_crossover",
                    method=method,
                    workload_class=WorkloadClass.HEADLINE,
                    gpu_index=(CORE_METHODS.index(method) + block) % 2,
                    scope="selected_recipe" if method in {"tts", "l0"} else "none",
                    context=40928,
                    regime="production_mix",
                    arrival="poisson",
                    slo=PRODUCTION_SLO,
                    block=block,
                    load_factor=load_factor,
                    parameterization=(
                        "selected" if method in {"tts", "l0"} else "none"
                    ),
                    variant=f"{phase}:E5a_open_loop_lambda_star",
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
                    method=method,
                    workload_class=WorkloadClass.HEADLINE,
                    gpu_index=(CORE_METHODS.index(method) + block) % 2,
                    scope="selected_recipe" if method in {"tts", "l0"} else "none",
                    context=40928,
                    regime="production_mix",
                    arrival=arrival,
                    slo=PRODUCTION_SLO,
                    block=block,
                    parameterization=(
                        "selected" if method in {"tts", "l0"} else "none"
                    ),
                    variant=f"{phase}:E5a_trace_or_soak",
                    status=(
                        CellStatus.BLOCKED
                        if arrival == "burstgpt_shape"
                        else CellStatus.UNMEASURED
                    ),
                    reason_code=(
                        "burstgpt_source_lock_missing"
                        if arrival == "burstgpt_shape"
                        else "awaiting_registered_measurement"
                    ),
                    reason=(
                        "No reviewed BurstGPT asset revision and row digest are "
                        "pinned in the external source lock."
                        if arrival == "burstgpt_shape"
                        else "No complete content-bound measurement exists."
                    ),
                )

    for backend in ("DFLASH", "DSPARK"):
        for method in CORE_METHODS:
            for topology in E5_TOPOLOGIES:
                for cohort_count in E5_COHORT_COUNTS:
                    for distribution in E5_COHORT_DISTRIBUTIONS:
                        factory.add(
                            experiment="E5",
                            model="Qwen/Qwen3-8B",
                            backend=backend,
                            task="topology_cohort_capacity",
                            method=method,
                            workload_class=WorkloadClass.HEADLINE,
                            gpu_count=1 if topology == "tp1_dp1" else 2,
                            gpu_index=(
                                (CORE_METHODS.index(method) + block) % 2
                                if topology == "tp1_dp1"
                                else None
                            ),
                            scope=(
                                "selected_recipe" if method in {"tts", "l0"} else "none"
                            ),
                            context=40928,
                            regime="production_mix",
                            arrival="saturation_anchor",
                            slo=PRODUCTION_SLO,
                            block=block,
                            cohort=f"K={cohort_count}:{distribution}",
                            topology=topology,
                            parameterization=(
                                "selected" if method in {"tts", "l0"} else "none"
                            ),
                            variant=f"{phase}:E5b_topology_cohort",
                            cohort_count=cohort_count,
                        )
    for failure in E5_FAILURES:
        factory.add(
            experiment="E5",
            model="Qwen/Qwen3-8B",
            backend="DFLASH+DSPARK",
            task="failure_injection",
            method="l0",
            workload_class=WorkloadClass.CORRECTNESS,
            gpu_count=2,
            scope="selected_recipe",
            context=40928,
            regime="production_mix",
            arrival=f"failure:{failure}",
            slo="excluded_from_headline",
            topology="tp2_and_two_replica",
            block=block,
            parameterization="selected",
            variant=f"{phase}:E5b_failure",
        )


def _add_e5_cells(factory: _CellFactory) -> None:
    for block in REGISTERED_CONFIRMATION_BLOCKS:
        _add_e5_block_cells(factory, block)


def _add_e6_cells(factory: _CellFactory) -> None:
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
                for method in CORE_METHODS:
                    factory.add(
                        experiment="E6",
                        model=model,
                        backend="NONE" if method == "target_only" else "NEXTN",
                        task=task,
                        method=method,
                        workload_class=WorkloadClass.HEADLINE,
                        gpu_count=2,
                        scope="transferred_e1_e2",
                        optimizer=(
                            "transferred_e2" if method in {"tts", "l0"} else None
                        ),
                        schedule=(
                            "transferred_e2" if method in {"tts", "l0"} else None
                        ),
                        context=context,
                        regime="native_mtp_transfer",
                        arrival="common_feasible_load",
                        slo=PRODUCTION_SLO,
                        topology="tp2_dp1",
                        parameterization=(
                            "transferred" if method in {"tts", "l0"} else "none"
                        ),
                        variant="compatibility_fit_then_transfer",
                        status=CellStatus.BLOCKED,
                        reason_code="native_nextn_preflight_required",
                        reason=(
                            "The exact NEXTN interface and two-rank memory fit have not "
                            "passed their registered preflight."
                        ),
                    )


def _add_e0_cells(factory: _CellFactory) -> None:
    for model in E0_MODELS:
        for backend in E0_BACKENDS:
            for task in E0_TASKS:
                for method in E0_METHODS:
                    selected = method in {
                        "tts",
                        "l0",
                        "onlinespec_ogd",
                        "onlinespec_opt",
                        "onlinespec_ens",
                    }
                    factory.add(
                        experiment="E0",
                        model=model,
                        backend=backend,
                        task=task,
                        method=method,
                        workload_class=WorkloadClass.HEADLINE,
                        scope="selected_recipe" if selected else "none",
                        context=4096,
                        regime="breadth_replication",
                        arrival="deterministic_stratified_requests",
                        slo="breadth_timing",
                        parameterization="selected" if selected else "none",
                        variant="compatibility_template",
                        reason_code="compatibility_unmeasured",
                        reason=(
                            "The model-backend-task interface has not yet produced a "
                            "compatibility receipt."
                        ),
                    )


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
    """Build the immutable Phase-II registry without allocating device state."""

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
    _add_e1_cells(factory)
    _add_e2_cells(factory)
    _add_e4_cells(factory)
    _add_e3b_cells(factory)
    _add_e1a_cells(factory)
    _add_e5_cells(factory)
    _add_e6_cells(factory)
    _add_e0_cells(factory)
    registry = ExperimentRegistry(
        schema_version=2,
        name="lightcone-industrial-experiment-registry",
        gpu_uuids=gpu_uuids,
        definitions=_industrial_definitions(),
        cells=tuple(factory.cells),
    )
    used_ports = {port for cell in registry.cells for port in cell.resources.ports}
    if used_ports != set(range(base_port, base_port + port_span)):
        raise AssertionError("industrial port-pool declaration is out of date")
    return registry
