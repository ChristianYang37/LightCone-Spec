"""Content-bound industrial planning, activation, budget, and alias contracts.

The registry intentionally declares the complete scientific envelope.  This
module is the reducer-owned boundary that turns sealed upstream decisions into
the much smaller set of cells that may actually be materialized.  Nothing in
this module launches a server or interprets confirmation results.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import pairwise
from typing import Literal

from lightcone_spec.experiments.registry import (
    CORE_METHODS,
    DRAFT_WIDTHS,
    E2_HALVING_STAGES,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    CellStatus,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    StageActivationPlan,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.experiments.statistics import (
    MAXIMUM_FINAL_BLOCKS,
    MINIMUM_FINAL_BLOCKS,
    P99_MINIMUM_COMPLETIONS,
    PRIMARY_CONTRASTS,
    PRIMARY_FAMILY_ALPHA,
    PRIMARY_MINIMUM_RELATIVE_EFFECT,
    PRIMARY_TARGET_POWER,
    PowerSizingPlan,
)

_SHA256_LENGTH = 64
_E1_SLICE_CELLS = 130
_E2_RETENTION_NUMERATOR = 1
_E2_RETENTION_DENOMINATOR = 4
_E2_FAMILY_FLOOR = 1
E2_HALVING_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "e2_successive_halving_protocol",
        "stages": E2_HALVING_STAGES,
        "retention_numerator": _E2_RETENTION_NUMERATOR,
        "retention_denominator": _E2_RETENTION_DENOMINATOR,
        "optimizer_schedule_family_floor": _E2_FAMILY_FLOOR,
        "minimum_published_updates_per_adapted_method": 1,
        "confidence_goodput": ("paired_request_log_ratio_normal_95pct_lower_bound_v1"),
        "confidence_pareto": (
            "non_dominated_confidence_lower_goodput_hbm_p99_exposed_update_v1"
        ),
        "ranking": (
            "safety_gate",
            "confidence_pareto",
            "descending_min_tts_l0_static_goodput_ratio",
            "ascending_hbm_bytes",
            "ascending_p99_itl_us",
            "ascending_exposed_update_us",
            "candidate_sha256",
        ),
        "confirmation_data_forbidden": True,
        "inventory_authority": (
            "content_bound_gpu_inventory_sha_source_receipt_count_host_v1"
        ),
    }
)
CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "confirmation_family_power_raw_reducer",
        "inputs": (
            "schema_v3_terminal_receipts",
            "hardware_receipts",
            "budget_observations",
            "qualification_locks",
            "content_bound_gpu_inventory",
        ),
        "pilot_blocks": PILOT_BLOCKS,
        "confirmation_data_forbidden": True,
        "power_constants": {
            "minimum_final_blocks": MINIMUM_FINAL_BLOCKS,
            "maximum_final_blocks": MAXIMUM_FINAL_BLOCKS,
            "target_power": PRIMARY_TARGET_POWER,
            "family_alpha": PRIMARY_FAMILY_ALPHA,
            "minimum_relative_effect": PRIMARY_MINIMUM_RELATIVE_EFFECT,
        },
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: object) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be lower-case SHA-256")


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{name} must be non-empty single-line text")


def _require_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _receipt_outputs(receipt: ExperimentReceipt) -> dict[str, str]:
    return {output.name: output.content_sha256 for output in receipt.outputs}


def _validate_direct_receipt(
    registry: ExperimentRegistry, receipt: ExperimentReceipt, experiment: str
) -> None:
    definition = registry.definition(experiment)
    if receipt.experiment != experiment or receipt.registry_sha256 != registry.sha256:
        raise ValueError(f"{experiment} receipt identity mismatch")
    if set(_receipt_outputs(receipt)) != set(definition.locked_outputs):
        raise ValueError(f"{experiment} receipt has incomplete or extra locked outputs")
    if {row.name for row in receipt.dependency_receipts} != set(
        definition.dependencies
    ):
        raise ValueError(f"{experiment} receipt has incomplete dependency lineage")


class BudgetScenario(str, Enum):
    OPTIMISTIC = "optimistic"
    REGISTERED = "registered"
    QUOTA_ENVELOPE = "quota_envelope"


@dataclass(frozen=True)
class ScenarioMilliseconds:
    """Exact integral millisecond limits for all three planning scenarios."""

    optimistic: int
    registered: int
    quota_envelope: int

    def __post_init__(self) -> None:
        for name in ("optimistic", "registered", "quota_envelope"):
            _require_nonnegative_int(name, getattr(self, name))
        if not self.optimistic <= self.registered <= self.quota_envelope:
            raise ValueError("budget scenarios must be monotone")

    def value(self, scenario: BudgetScenario) -> int:
        return int(getattr(self, scenario.value))

    def __add__(self, other: ScenarioMilliseconds) -> ScenarioMilliseconds:
        if not isinstance(other, ScenarioMilliseconds):
            return NotImplemented
        return ScenarioMilliseconds(
            self.optimistic + other.optimistic,
            self.registered + other.registered,
            self.quota_envelope + other.quota_envelope,
        )

    def scale(self, multiplier: int) -> ScenarioMilliseconds:
        _require_nonnegative_int("scenario multiplier", multiplier)
        return ScenarioMilliseconds(
            self.optimistic * multiplier,
            self.registered * multiplier,
            self.quota_envelope * multiplier,
        )


ZERO_MILLISECONDS = ScenarioMilliseconds(0, 0, 0)


@dataclass(frozen=True)
class ExpectedMaximumCount:
    expected: int
    maximum: int

    def __post_init__(self) -> None:
        _require_nonnegative_int("expected count", self.expected)
        _require_nonnegative_int("maximum count", self.maximum)
        if self.expected > self.maximum:
            raise ValueError("expected count cannot exceed maximum count")

    def __add__(self, other: ExpectedMaximumCount) -> ExpectedMaximumCount:
        if not isinstance(other, ExpectedMaximumCount):
            return NotImplemented
        return ExpectedMaximumCount(
            expected=self.expected + other.expected,
            maximum=self.maximum + other.maximum,
        )


ZERO_COUNT = ExpectedMaximumCount(0, 0)


@dataclass(frozen=True)
class ExactScenarioHours:
    """Exact hours represented as integral millisecond numerators."""

    optimistic_millisecond_numerator: int
    registered_millisecond_numerator: int
    quota_envelope_millisecond_numerator: int
    denominator_milliseconds_per_hour: int = 3_600_000

    def __post_init__(self) -> None:
        for name in (
            "optimistic_millisecond_numerator",
            "registered_millisecond_numerator",
            "quota_envelope_millisecond_numerator",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        if self.denominator_milliseconds_per_hour != 3_600_000:
            raise ValueError("hour accounting denominator is fixed at 3,600,000 ms")
        if not (
            self.optimistic_millisecond_numerator
            <= self.registered_millisecond_numerator
            <= self.quota_envelope_millisecond_numerator
        ):
            raise ValueError("exact hour scenarios must be monotone")

    @classmethod
    def from_milliseconds(cls, value: ScenarioMilliseconds) -> ExactScenarioHours:
        return cls(value.optimistic, value.registered, value.quota_envelope)


class BudgetJobKind(str, Enum):
    STANDARD = "standard"
    SHORT = "short"
    P99_ANCHOR = "p99_anchor"
    SOAK = "soak"
    PROFILER = "profiler"
    FAILURE = "failure"
    COMPILE = "compile"
    DOWNLOAD = "download"


class P99AnchorStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    REQUIRED_UNRESOLVED = "required_unresolved"
    LOCKED = "locked"


_ADDITIVE_DURATION_FIELDS = (
    "startup_model_load",
    "compile_jit_graph_prewarm",
    "excluded_warmup",
    "scored_arrival",
    "drain",
    "reset_finalization",
    "evidence_flush_shutdown",
    "soak",
    "failure_injection",
    "retry",
    "profiler",
    "download_compile_reservation",
)


@dataclass(frozen=True)
class ExperimentBudget:
    """One complete, immutable, per-cell budget with no duration defaults.

    Every duration is explicitly supplied in integral milliseconds.  A zero is
    therefore an explicit not-applicable decision rather than a hidden default.
    ``measured_gpu_ms`` is explicitly ``None`` on a pre-run registration and is
    populated only in an immutable observed copy or observation receipt.
    """

    schema_version: int
    cell_id: str
    experiment: str
    method: str
    workload_class: WorkloadClass
    job_kind: BudgetJobKind
    startup_model_load: ScenarioMilliseconds
    compile_jit_graph_prewarm: ScenarioMilliseconds
    excluded_warmup: ScenarioMilliseconds
    excluded_warmup_requests: ExpectedMaximumCount
    scored_arrival: ScenarioMilliseconds
    request_deadline: ScenarioMilliseconds
    drain: ScenarioMilliseconds
    reset_finalization: ScenarioMilliseconds
    evidence_flush_shutdown: ScenarioMilliseconds
    output_tokens: ExpectedMaximumCount
    minimum_completed_requests: int
    p99_anchor_status: P99AnchorStatus
    soak: ScenarioMilliseconds
    failure_injection: ScenarioMilliseconds
    retry: ScenarioMilliseconds
    retry_allowance: int
    profiler: ScenarioMilliseconds
    download_compile_reservation: ScenarioMilliseconds
    gpu_count: int
    topology: str
    reserved_gpu_ms: ScenarioMilliseconds
    measured_gpu_ms: int | None
    fixed_instance_billed_gpu_ms: ScenarioMilliseconds

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only ExperimentBudget schema version 1 is supported")
        _require_sha256("budget cell_id", self.cell_id)
        for name in ("experiment", "method", "topology"):
            _require_text(name, getattr(self, name))
        if not isinstance(self.workload_class, WorkloadClass):
            raise TypeError("workload_class must be a WorkloadClass")
        if not isinstance(self.job_kind, BudgetJobKind):
            raise TypeError("job_kind must be a BudgetJobKind")
        if not isinstance(self.p99_anchor_status, P99AnchorStatus):
            raise TypeError("p99_anchor_status must be a P99AnchorStatus")
        _require_nonnegative_int(
            "minimum_completed_requests", self.minimum_completed_requests
        )
        _require_nonnegative_int("retry_allowance", self.retry_allowance)
        if not isinstance(self.gpu_count, int) or isinstance(self.gpu_count, bool):
            raise TypeError("gpu_count must be an integer")
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be positive")
        if self.measured_gpu_ms is not None:
            _require_nonnegative_int("measured_gpu_ms", self.measured_gpu_ms)
        if self.retry_allowance == 0 and self.retry.registered != 0:
            raise ValueError("retry time requires a positive retry allowance")
        if self.retry_allowance > 0 and self.retry.registered == 0:
            raise ValueError("retry allowance requires an explicit retry duration")
        if self.excluded_warmup.registered > 0 and (
            self.excluded_warmup_requests.maximum == 0
        ):
            raise ValueError("excluded warm-up duration requires a request pool")
        if self.excluded_warmup.registered == 0 and (
            self.excluded_warmup_requests.maximum > 0
        ):
            raise ValueError("excluded warm-up requests require an explicit duration")
        if self.scored_arrival.registered > 0:
            if self.request_deadline.registered <= 0:
                raise ValueError("scored arrivals require an explicit request deadline")
            if self.output_tokens.maximum <= 0:
                raise ValueError("scored arrivals require an output-token budget")
            if self.minimum_completed_requests <= 0:
                raise ValueError("scored arrivals require a minimum completion count")
        if self.job_kind is BudgetJobKind.P99_ANCHOR:
            if self.p99_anchor_status is P99AnchorStatus.NOT_REQUIRED:
                raise ValueError("p99-anchor jobs require an explicit anchor status")
            if self.minimum_completed_requests < P99_MINIMUM_COMPLETIONS:
                raise ValueError(
                    "p99-anchor jobs require at least 10,000 completed requests"
                )
        elif self.p99_anchor_status is not P99AnchorStatus.NOT_REQUIRED:
            raise ValueError(
                "p99 anchor status is valid only for preregistered p99-anchor jobs"
            )
        required_component = {
            BudgetJobKind.SOAK: self.soak,
            BudgetJobKind.PROFILER: self.profiler,
            BudgetJobKind.FAILURE: self.failure_injection,
            BudgetJobKind.COMPILE: self.compile_jit_graph_prewarm,
            BudgetJobKind.DOWNLOAD: self.download_compile_reservation,
        }.get(self.job_kind)
        if required_component is not None and required_component.registered <= 0:
            raise ValueError(
                f"{self.job_kind.value} jobs require their explicit duration component"
            )
        compute_gpu_ms = self.wall_time.scale(self.gpu_count)
        for scenario in BudgetScenario:
            if self.reserved_gpu_ms.value(scenario) < compute_gpu_ms.value(scenario):
                raise ValueError("reserved GPU time cannot be below compute GPU time")
            if self.fixed_instance_billed_gpu_ms.value(scenario) < compute_gpu_ms.value(
                scenario
            ):
                raise ValueError(
                    "fixed-instance billed GPU time cannot be below compute GPU time"
                )

    @cached_property
    def wall_time(self) -> ScenarioMilliseconds:
        total = ZERO_MILLISECONDS
        for name in _ADDITIVE_DURATION_FIELDS:
            total += getattr(self, name)
        return total

    @cached_property
    def compute_gpu_ms(self) -> ScenarioMilliseconds:
        return self.wall_time.scale(self.gpu_count)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class BudgetInventoryIdentity:
    schema_version: int
    host_sha256: str
    gpu_uuids: tuple[str, ...]
    topology_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only budget inventory schema version 1 is supported")
        _require_sha256("inventory host_sha256", self.host_sha256)
        _require_sha256("inventory topology_sha256", self.topology_sha256)
        if not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("inventory requires unique GPU UUIDs")
        for value in self.gpu_uuids:
            _require_text("inventory GPU UUID", value)

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_uuids)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class BudgetGroupTotal:
    experiment: str
    method: str
    workload_class: str
    topology: str
    cells: int
    gpu_cell_units: int
    component_ms: tuple[tuple[str, ScenarioMilliseconds], ...]
    request_deadline_ms: ScenarioMilliseconds
    excluded_warmup_requests: ExpectedMaximumCount
    output_tokens: ExpectedMaximumCount
    minimum_completed_requests: int
    retry_allowance: int
    p99_anchor_status_counts: tuple[tuple[str, int], ...]
    compute_gpu_ms: ScenarioMilliseconds
    compute_gpu_hours: ExactScenarioHours
    reserved_gpu_ms: ScenarioMilliseconds
    reserved_gpu_hours: ExactScenarioHours
    fixed_instance_billed_gpu_ms: ScenarioMilliseconds
    fixed_instance_billed_gpu_hours: ExactScenarioHours


@dataclass(frozen=True)
class IndustrialBudgetReport:
    schema_version: int
    registry_sha256: str
    activation_sha256: str
    inventory: BudgetInventoryIdentity
    budget_sha256s: tuple[str, ...]
    groups: tuple[BudgetGroupTotal, ...]
    cells: int
    gpu_cell_units: int
    component_ms: tuple[tuple[str, ScenarioMilliseconds], ...]
    request_deadline_ms: ScenarioMilliseconds
    excluded_warmup_requests: ExpectedMaximumCount
    output_tokens: ExpectedMaximumCount
    minimum_completed_requests: int
    retry_allowance: int
    p99_anchor_status_counts: tuple[tuple[str, int], ...]
    compute_gpu_ms: ScenarioMilliseconds
    compute_gpu_hours: ExactScenarioHours
    reserved_gpu_ms: ScenarioMilliseconds
    reserved_gpu_hours: ExactScenarioHours
    fixed_instance_billed_gpu_ms: ScenarioMilliseconds
    fixed_instance_billed_gpu_hours: ExactScenarioHours
    estimated_wall_ms: ScenarioMilliseconds | None
    estimated_wall_hours: ExactScenarioHours | None
    schedule_fixed_instance_billed_gpu_ms: ScenarioMilliseconds | None
    schedule_fixed_instance_billed_gpu_hours: ExactScenarioHours | None
    unresolved_assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only industrial budget report schema 1 is supported")
        _require_sha256("budget registry_sha256", self.registry_sha256)
        _require_sha256("budget activation_sha256", self.activation_sha256)
        if len(self.budget_sha256s) != len(set(self.budget_sha256s)):
            raise ValueError("budget report contains duplicate budget identities")
        if self.unresolved_assumptions != tuple(
            sorted(set(self.unresolved_assumptions))
        ):
            raise ValueError("unresolved assumptions must be sorted and unique")
        if (self.estimated_wall_ms is None) != (self.estimated_wall_hours is None):
            raise ValueError(
                "wall millisecond and hour estimates must be present or absent together"
            )
        if (self.schedule_fixed_instance_billed_gpu_ms is None) != (
            self.schedule_fixed_instance_billed_gpu_hours is None
        ):
            raise ValueError(
                "schedule billed millisecond and hour totals must be present together"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def _sum_scenarios(values: Iterable[ScenarioMilliseconds]) -> ScenarioMilliseconds:
    total = ZERO_MILLISECONDS
    for value in values:
        total += value
    return total


def _sum_counts(values: Iterable[ExpectedMaximumCount]) -> ExpectedMaximumCount:
    total = ZERO_COUNT
    for value in values:
        total += value
    return total


def _p99_counts(
    budgets: Sequence[ExperimentBudget],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            status.value,
            sum(budget.p99_anchor_status is status for budget in budgets),
        )
        for status in P99AnchorStatus
    )


def _component_totals(
    budgets: Sequence[ExperimentBudget],
) -> tuple[tuple[str, ScenarioMilliseconds], ...]:
    return tuple(
        (name, _sum_scenarios(getattr(budget, name) for budget in budgets))
        for name in _ADDITIVE_DURATION_FIELDS
    )


def _estimate_wall_time(
    budgets: Sequence[ExperimentBudget], inventory_gpus: int
) -> ScenarioMilliseconds:
    """Deterministic gang-aware list schedule, exact for the declared order."""

    results: list[int] = []
    ordered = tuple(sorted(budgets, key=lambda budget: budget.cell_id))
    for scenario in BudgetScenario:
        available = [0] * inventory_gpus
        for budget in ordered:
            selected = sorted(
                range(inventory_gpus), key=lambda index: (available[index], index)
            )[: budget.gpu_count]
            start = max(available[index] for index in selected)
            finish = start + budget.wall_time.value(scenario)
            for index in selected:
                available[index] = finish
        results.append(max(available, default=0))
    return ScenarioMilliseconds(*results)


def estimate_industrial_budget(
    registry: ExperimentRegistry,
    *,
    activated_cell_ids: Sequence[str],
    activation_sha256: str,
    budgets: Sequence[ExperimentBudget],
    inventory: BudgetInventoryIdentity,
) -> IndustrialBudgetReport:
    """Reduce exact per-cell budgets; missing or extra coverage fails closed."""

    _require_sha256("activation_sha256", activation_sha256)
    active = tuple(activated_cell_ids)
    if len(active) != len(set(active)):
        raise ValueError("activated cell identities must be unique")
    known = {cell.cell_id: cell for cell in registry.cells}
    if set(active) - known.keys():
        raise ValueError("activated budget set contains cells outside the registry")
    rows = tuple(budgets)
    by_cell = {budget.cell_id: budget for budget in rows}
    if len(by_cell) != len(rows):
        raise ValueError("budgets must have unique cell identities")
    if set(by_cell) != set(active):
        missing = sorted(set(active) - by_cell.keys())
        extra = sorted(by_cell.keys() - set(active))
        raise ValueError(f"budget coverage mismatch: missing={missing}, extra={extra}")
    for budget in rows:
        cell = known[budget.cell_id]
        if budget.measured_gpu_ms is not None:
            raise ValueError(
                "pre-run budget estimates cannot contain measured GPU time"
            )
        if (
            budget.experiment != cell.identity.experiment
            or budget.method != cell.identity.method
            or budget.workload_class is not cell.resources.workload_class
            or budget.gpu_count != cell.resources.gpu_count
            or budget.topology != cell.identity.topology
        ):
            raise ValueError("budget metadata does not match its registry cell")
        expected_billing = budget.wall_time.scale(inventory.gpu_count)
        if (
            budget.gpu_count <= inventory.gpu_count
            and budget.fixed_instance_billed_gpu_ms != expected_billing
        ):
            raise ValueError(
                "fixed-instance budget must bill the exact supplied inventory"
            )

    group_rows: list[BudgetGroupTotal] = []
    group_keys = sorted(
        {
            (
                budget.experiment,
                budget.method,
                budget.workload_class.value,
                budget.topology,
            )
            for budget in rows
        }
    )
    for experiment, method, workload, topology in group_keys:
        members = tuple(
            budget
            for budget in rows
            if (
                budget.experiment,
                budget.method,
                budget.workload_class.value,
                budget.topology,
            )
            == (experiment, method, workload, topology)
        )
        group_compute = _sum_scenarios(row.compute_gpu_ms for row in members)
        group_reserved = _sum_scenarios(row.reserved_gpu_ms for row in members)
        group_billed = _sum_scenarios(
            row.fixed_instance_billed_gpu_ms for row in members
        )
        group_rows.append(
            BudgetGroupTotal(
                experiment=experiment,
                method=method,
                workload_class=workload,
                topology=topology,
                cells=len(members),
                gpu_cell_units=sum(row.gpu_count for row in members),
                component_ms=_component_totals(members),
                request_deadline_ms=_sum_scenarios(
                    row.request_deadline for row in members
                ),
                excluded_warmup_requests=_sum_counts(
                    row.excluded_warmup_requests for row in members
                ),
                output_tokens=_sum_counts(row.output_tokens for row in members),
                minimum_completed_requests=sum(
                    row.minimum_completed_requests for row in members
                ),
                retry_allowance=sum(row.retry_allowance for row in members),
                p99_anchor_status_counts=_p99_counts(members),
                compute_gpu_ms=group_compute,
                compute_gpu_hours=ExactScenarioHours.from_milliseconds(group_compute),
                reserved_gpu_ms=group_reserved,
                reserved_gpu_hours=ExactScenarioHours.from_milliseconds(group_reserved),
                fixed_instance_billed_gpu_ms=group_billed,
                fixed_instance_billed_gpu_hours=ExactScenarioHours.from_milliseconds(
                    group_billed
                ),
            )
        )
    unresolved_values = {
        f"cell_requires_{budget.gpu_count}_gpus_but_inventory_has_"
        f"{inventory.gpu_count}:{budget.cell_id}"
        for budget in rows
        if budget.gpu_count > inventory.gpu_count
    }
    unresolved_values.update(
        f"p99_anchor_unresolved:{budget.cell_id}"
        for budget in rows
        if budget.p99_anchor_status is P99AnchorStatus.REQUIRED_UNRESOLVED
    )
    unresolved = tuple(sorted(unresolved_values))
    resources_fit = all(budget.gpu_count <= inventory.gpu_count for budget in rows)
    wall = _estimate_wall_time(rows, inventory.gpu_count) if resources_fit else None
    schedule_billed = None if wall is None else wall.scale(inventory.gpu_count)
    compute = _sum_scenarios(row.compute_gpu_ms for row in rows)
    reserved = _sum_scenarios(row.reserved_gpu_ms for row in rows)
    billed = _sum_scenarios(row.fixed_instance_billed_gpu_ms for row in rows)
    return IndustrialBudgetReport(
        schema_version=1,
        registry_sha256=registry.sha256,
        activation_sha256=activation_sha256,
        inventory=inventory,
        budget_sha256s=tuple(sorted(budget.sha256 for budget in rows)),
        groups=tuple(group_rows),
        cells=len(rows),
        gpu_cell_units=sum(row.gpu_count for row in rows),
        component_ms=_component_totals(rows),
        request_deadline_ms=_sum_scenarios(row.request_deadline for row in rows),
        excluded_warmup_requests=_sum_counts(
            row.excluded_warmup_requests for row in rows
        ),
        output_tokens=_sum_counts(row.output_tokens for row in rows),
        minimum_completed_requests=sum(row.minimum_completed_requests for row in rows),
        retry_allowance=sum(row.retry_allowance for row in rows),
        p99_anchor_status_counts=_p99_counts(rows),
        compute_gpu_ms=compute,
        compute_gpu_hours=ExactScenarioHours.from_milliseconds(compute),
        reserved_gpu_ms=reserved,
        reserved_gpu_hours=ExactScenarioHours.from_milliseconds(reserved),
        fixed_instance_billed_gpu_ms=billed,
        fixed_instance_billed_gpu_hours=ExactScenarioHours.from_milliseconds(billed),
        estimated_wall_ms=wall,
        estimated_wall_hours=(
            None if wall is None else ExactScenarioHours.from_milliseconds(wall)
        ),
        schedule_fixed_instance_billed_gpu_ms=schedule_billed,
        schedule_fixed_instance_billed_gpu_hours=(
            None
            if schedule_billed is None
            else ExactScenarioHours.from_milliseconds(schedule_billed)
        ),
        unresolved_assumptions=unresolved,
    )


@dataclass(frozen=True)
class BudgetObservationReceipt:
    """Content-bound estimated-versus-observed accounting for one execution."""

    schema_version: int
    budget: ExperimentBudget
    observed_component_ms: tuple[tuple[str, int], ...]
    measured_gpu_ms: int
    fixed_instance_billed_gpu_ms: int
    terminal_evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only budget observation schema version 1 is supported")
        if self.budget.measured_gpu_ms is not None:
            raise ValueError("observations must bind the immutable pre-run budget")
        names = tuple(name for name, _ in self.observed_component_ms)
        if names != _ADDITIVE_DURATION_FIELDS:
            raise ValueError("observed timing must cover every component in order")
        for name, value in self.observed_component_ms:
            _require_nonnegative_int(f"observed {name}", value)
        _require_nonnegative_int("measured_gpu_ms", self.measured_gpu_ms)
        _require_nonnegative_int(
            "fixed_instance_billed_gpu_ms", self.fixed_instance_billed_gpu_ms
        )
        _require_sha256("terminal_evidence_sha256", self.terminal_evidence_sha256)
        if self.measured_gpu_ms > self.observed_wall_ms * self.budget.gpu_count:
            raise ValueError("measured GPU time exceeds the observed gang wall time")
        if self.fixed_instance_billed_gpu_ms < self.measured_gpu_ms:
            raise ValueError(
                "fixed-instance billed GPU time is below measured GPU time"
            )

    @property
    def observed_wall_ms(self) -> int:
        return sum(value for _, value in self.observed_component_ms)

    @property
    def registered_wall_delta_ms(self) -> int:
        return self.observed_wall_ms - self.budget.wall_time.registered

    @property
    def registered_gpu_delta_ms(self) -> int:
        return self.measured_gpu_ms - self.budget.compute_gpu_ms.registered

    @property
    def registered_billed_delta_ms(self) -> int:
        return (
            self.fixed_instance_billed_gpu_ms
            - self.budget.fixed_instance_billed_gpu_ms.registered
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


class DispositionStatus(str, Enum):
    ACTIVATED = "ACTIVATED"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "N/A"
    DEFERRED = "DEFERRED"
    COMPLETED_PRIOR_ROUND = "COMPLETED_PRIOR_ROUND"


@dataclass(frozen=True)
class CellDisposition:
    cell_id: str
    status: DispositionStatus
    reason_code: str

    def __post_init__(self) -> None:
        _require_sha256("disposition cell_id", self.cell_id)
        if not isinstance(self.status, DispositionStatus):
            raise TypeError("disposition status must be a DispositionStatus")
        _require_text("disposition reason_code", self.reason_code)
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in self.reason_code
        ):
            raise ValueError("disposition reason_code is invalid")


@dataclass(frozen=True)
class ReducerActivationArtifact:
    """Reducer-owned activation plus an immutable reason for every template."""

    schema_version: int
    plan: StageActivationPlan
    reducer_protocol_sha256: str
    dispositions: tuple[CellDisposition, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only reducer activation schema version 1 is supported")
        _require_sha256("reducer_protocol_sha256", self.reducer_protocol_sha256)
        if (
            tuple(sorted(self.dispositions, key=lambda row: row.cell_id))
            != self.dispositions
        ):
            raise ValueError("activation dispositions must be sorted by cell_id")
        if len({row.cell_id for row in self.dispositions}) != len(self.dispositions):
            raise ValueError("activation dispositions must have unique cell IDs")
        by_status = {
            status: tuple(
                row.cell_id for row in self.dispositions if row.status is status
            )
            for status in DispositionStatus
        }
        if by_status[DispositionStatus.ACTIVATED] != self.plan.activated_cell_ids:
            raise ValueError("activated dispositions do not match the stage plan")
        if by_status[DispositionStatus.BLOCKED] != self.plan.blocked_cell_ids:
            raise ValueError("blocked dispositions do not match the stage plan")
        legacy_na = tuple(
            sorted(
                by_status[DispositionStatus.NOT_APPLICABLE]
                + by_status[DispositionStatus.COMPLETED_PRIOR_ROUND]
            )
        )
        if legacy_na != self.plan.not_applicable_cell_ids:
            raise ValueError("N/A dispositions do not match the stage plan")
        if by_status[DispositionStatus.DEFERRED] != self.plan.deferred_cell_ids:
            raise ValueError("deferred dispositions do not match the stage plan")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SealedE3aSelection:
    schema_version: int
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    width: int
    concurrency: int
    reducer_evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E3a selection schema version 1 is supported")
        for name in (
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "reducer_evidence_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.width not in DRAFT_WIDTHS:
            raise ValueError("E3a selected width is outside the registered grid")
        if not isinstance(self.concurrency, int) or isinstance(self.concurrency, bool):
            raise TypeError("E3a selected concurrency must be an integer")
        if self.concurrency < 1:
            raise ValueError("E3a selected concurrency must be positive")

    @cached_property
    def matched_width_output_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "e3a_matched_width",
                "width": self.width,
                "reducer_evidence_sha256": self.reducer_evidence_sha256,
            }
        )

    @cached_property
    def reference_load_output_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "e3a_reference_load",
                "concurrency": self.concurrency,
                "reducer_evidence_sha256": self.reducer_evidence_sha256,
            }
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def _make_activation(
    *,
    registry: ExperimentRegistry,
    experiment: str,
    dependency_receipt: ExperimentReceipt,
    runtime_sha256: str,
    split_sha256: str,
    source_selection_sha256: str,
    activation_round: str,
    rows: Sequence[CellDisposition],
    reason_code: str,
    reducer_protocol_sha256: str,
) -> ReducerActivationArtifact:
    ordered = tuple(sorted(rows, key=lambda row: row.cell_id))
    by_status = {
        status: tuple(row.cell_id for row in ordered if row.status is status)
        for status in DispositionStatus
    }
    active = by_status[DispositionStatus.ACTIVATED]
    status = "AVAILABLE" if active else "BLOCKED"
    plan = StageActivationPlan(
        registry_sha256=registry.sha256,
        experiment=experiment,
        dependency_receipt_sha256=dependency_receipt.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        source_selection_sha256=source_selection_sha256,
        activation_round=activation_round,
        status=status,
        activated_cell_ids=active,
        not_applicable_cell_ids=tuple(
            sorted(
                by_status[DispositionStatus.NOT_APPLICABLE]
                + by_status[DispositionStatus.COMPLETED_PRIOR_ROUND]
            )
        ),
        blocked_cell_ids=by_status[DispositionStatus.BLOCKED],
        deferred_cell_ids=by_status[DispositionStatus.DEFERRED],
        reason_code=reason_code,
    )
    return ReducerActivationArtifact(
        schema_version=1,
        plan=plan,
        reducer_protocol_sha256=reducer_protocol_sha256,
        dispositions=ordered,
    )


def reduce_e1_activation(
    registry: ExperimentRegistry,
    *,
    e3a_receipt: ExperimentReceipt,
    selection: SealedE3aSelection,
) -> ReducerActivationArtifact:
    """Materialize the one sealed 130-cell E1 width/load slice."""

    _validate_direct_receipt(registry, e3a_receipt, "E3a")
    if (
        selection.registry_sha256 != registry.sha256
        or e3a_receipt.runtime_sha256 != selection.runtime_sha256
        or e3a_receipt.split_sha256 != selection.split_sha256
    ):
        raise ValueError("E1 activation identity does not match the E3a receipt")
    outputs = _receipt_outputs(e3a_receipt)
    if outputs.get("matched_width") != selection.matched_width_output_sha256:
        raise ValueError("E3a receipt does not bind the selected width artifact")
    if outputs.get("e1_reference_load") != selection.reference_load_output_sha256:
        raise ValueError("E3a receipt does not bind the selected load artifact")
    tag = f"width={selection.width}:concurrency={selection.concurrency}"
    cells = registry.cells_for("E1")
    selected = {cell.cell_id for cell in cells if tag in cell.identity.variant}
    if len(cells) != 2730 or len(selected) != _E1_SLICE_CELLS:
        raise ValueError("registered E1 envelope must reduce from 2,730 to 130 cells")
    methods = {cell.identity.method for cell in cells if cell.cell_id in selected}
    if methods != set(CORE_METHODS):
        raise ValueError("E1 slice must retain Target-only, Static, TTS, and L0")
    rows: list[CellDisposition] = []
    for cell in cells:
        if not cell.runnable:
            status = (
                DispositionStatus.NOT_APPLICABLE
                if cell.status is CellStatus.NOT_APPLICABLE
                else DispositionStatus.BLOCKED
            )
            reason = cell.reason_code
        elif cell.cell_id in selected:
            status = DispositionStatus.ACTIVATED
            reason = "e3a_selected_width_load_slice"
        else:
            status = DispositionStatus.DEFERRED
            reason = "outside_e3a_selected_width_load_slice"
        rows.append(CellDisposition(cell.cell_id, status, reason))
    protocol_sha = content_sha256(
        {
            "schema_version": 1,
            "kind": "e1_single_slice_reducer",
            "declared_cells": 2730,
            "activated_cells": _E1_SLICE_CELLS,
            "source": "sealed_e3a_selection",
        }
    )
    return _make_activation(
        registry=registry,
        experiment="E1",
        dependency_receipt=e3a_receipt,
        runtime_sha256=selection.runtime_sha256,
        split_sha256=selection.split_sha256,
        source_selection_sha256=selection.sha256,
        activation_round="e3a_locked_reference",
        rows=rows,
        reason_code="e3a_reducer_selected_slice",
        reducer_protocol_sha256=protocol_sha,
    )


def verify_e1_activation(
    registry: ExperimentRegistry,
    *,
    e3a_receipt: ExperimentReceipt,
    selection: SealedE3aSelection,
    artifact: ReducerActivationArtifact,
) -> None:
    """Reject serialized E1 activations that differ from reducer output."""

    expected = reduce_e1_activation(
        registry,
        e3a_receipt=e3a_receipt,
        selection=selection,
    )
    if artifact != expected:
        raise ValueError("E1 activation is not the exact reducer-generated artifact")


@dataclass(frozen=True)
class E1GeometryIdentity:
    scope: str
    parameterization: str
    rank: int | None
    alpha_over_rank: float | None

    def __post_init__(self) -> None:
        _require_text("E1 geometry scope", self.scope)
        if self.parameterization == "full":
            if self.rank is not None or self.alpha_over_rank is not None:
                raise ValueError("full E1 geometry cannot carry LoRA fields")
        elif self.parameterization == "lora":
            if self.rank is None or self.alpha_over_rank != 1.0:
                raise ValueError("LoRA E1 geometry requires rank and alpha/r=1")
        else:
            raise ValueError("E1 geometry parameterization must be full or lora")

    @classmethod
    def from_cell(cls, cell: ExperimentCell) -> E1GeometryIdentity:
        identity = cell.identity
        if identity.method not in {"tts", "l0"}:
            raise ValueError("E1 geometry can be derived only from TTS/L0")
        if identity.scope is None:
            raise ValueError("adaptive E1 geometry requires an exact scope")
        return cls(
            scope=identity.scope,
            parameterization=identity.parameterization,
            rank=identity.rank,
            alpha_over_rank=identity.alpha_over_rank,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E1ParetoArtifact:
    schema_version: int
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    e1_activation_sha256: str
    reducer_evidence_sha256: str
    common_load_sha256: str
    surviving_geometries: tuple[E1GeometryIdentity, ...]
    selection_state: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E1 Pareto schema version 1 is supported")
        for name in (
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "e1_activation_sha256",
            "reducer_evidence_sha256",
            "common_load_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not self.surviving_geometries:
            raise ValueError("E1 Pareto artifact requires surviving geometries")
        identities = tuple(row.sha256 for row in self.surviving_geometries)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("E1 Pareto geometries must be sorted and unique")
        if self.selection_state != "sealed_before_e2_unblinding":
            raise ValueError("E1 Pareto selection must be sealed before E2")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2CandidateIdentity:
    model: str
    backend: str
    task: str
    scope: str
    parameterization: str
    rank: int | None
    alpha_over_rank: float | None
    optimizer: str
    learning_rate: float
    schedule: str
    width: int

    def __post_init__(self) -> None:
        for name in ("model", "backend", "task", "scope", "optimizer", "schedule"):
            _require_text(name, getattr(self, name))
        if self.parameterization not in {"full", "lora"}:
            raise ValueError("E2 parameterization must be full or lora")
        if self.parameterization == "full" and (
            self.rank is not None or self.alpha_over_rank is not None
        ):
            raise ValueError("full E2 candidates cannot carry LoRA fields")
        if self.parameterization == "lora" and (
            self.rank is None or self.alpha_over_rank != 1.0
        ):
            raise ValueError("LoRA E2 candidates require rank and alpha/r=1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("E2 learning rate must be finite and positive")
        if self.width < 1:
            raise ValueError("E2 width must be positive")

    @classmethod
    def from_cell(cls, cell: ExperimentCell) -> E2CandidateIdentity:
        identity = cell.identity
        if identity.experiment != "E2" or identity.method not in {"tts", "l0"}:
            raise ValueError("E2 candidates can be derived only from E2 TTS/L0 cells")
        if (
            identity.scope is None
            or identity.optimizer is None
            or identity.learning_rate is None
            or identity.schedule is None
            or identity.width is None
        ):
            raise ValueError("E2 candidate identity contains unresolved fields")
        return cls(
            model=identity.model,
            backend=identity.backend,
            task=identity.task,
            scope=identity.scope,
            parameterization=identity.parameterization,
            rank=identity.rank,
            alpha_over_rank=identity.alpha_over_rank,
            optimizer=identity.optimizer,
            learning_rate=identity.learning_rate,
            schedule=identity.schedule,
            width=identity.width,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    @property
    def family(self) -> tuple[str, str]:
        return self.optimizer, self.schedule


def _e2_stage(cell: ExperimentCell) -> int:
    for stage_index in range(len(E2_HALVING_STAGES)):
        if f"halving_stage={stage_index}:" in cell.identity.variant:
            return stage_index
    raise ValueError("E2 cell does not identify a registered halving stage")


def _e2_candidate_pairs(
    cells: Sequence[ExperimentCell],
) -> dict[str, tuple[E2CandidateIdentity, tuple[ExperimentCell, ...]]]:
    grouped: dict[str, tuple[E2CandidateIdentity, list[ExperimentCell]]] = {}
    for cell in cells:
        if cell.identity.method not in {"tts", "l0"}:
            continue
        candidate = E2CandidateIdentity.from_cell(cell)
        current = grouped.setdefault(candidate.sha256, (candidate, []))
        current[1].append(cell)
    result: dict[str, tuple[E2CandidateIdentity, tuple[ExperimentCell, ...]]] = {}
    for candidate_id, (candidate, members) in grouped.items():
        if {member.identity.method for member in members} != {"tts", "l0"} or len(
            members
        ) != 2:
            raise ValueError("every E2 candidate requires one matched TTS/L0 pair")
        if len({member.resources.gpu_uuids for member in members}) != 1:
            raise ValueError("E2 TTS/L0 candidate pair must use the same GPU")
        if len({member.identity.block for member in members}) != 1:
            raise ValueError("E2 TTS/L0 candidate pair must use one scientific block")
        result[candidate_id] = (candidate, tuple(members))
    return result


@dataclass(frozen=True)
class E2SurvivorReceipt:
    schema_version: int
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    halving_protocol_sha256: str
    stage_index: int
    source_activation_sha256: str
    prior_stage_reduction_sha256: str | None
    completed_cells_sha256: str
    completed_stage_cell_ids: tuple[str, ...]
    completed_lineage_cell_ids: tuple[str, ...]
    tuning_evidence_sha256: str
    source_candidate_ids: tuple[str, ...]
    survivor_candidate_ids: tuple[str, ...]
    final_recipe_candidate_id: str | None
    status: Literal["SURVIVORS", "FINAL_RECIPE", "BLOCKED"]
    reason_code: str
    selection_state: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only E2 survivor schema version 2 is supported")
        for name in (
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "halving_protocol_sha256",
            "source_activation_sha256",
            "completed_cells_sha256",
            "tuning_evidence_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.halving_protocol_sha256 != E2_HALVING_PROTOCOL_SHA256:
            raise ValueError("E2 receipt uses an unregistered halving protocol")
        if self.stage_index not in range(len(E2_HALVING_STAGES)):
            raise ValueError("E2 survivor stage is outside the registered stages")
        if self.stage_index == 0:
            if self.prior_stage_reduction_sha256 is not None:
                raise ValueError("E2 stage zero cannot bind a prior raw reduction")
        else:
            _require_sha256(
                "prior_stage_reduction_sha256",
                self.prior_stage_reduction_sha256,
            )
        for name, values in (
            ("completed stage", self.completed_stage_cell_ids),
            ("completed lineage", self.completed_lineage_cell_ids),
            ("source", self.source_candidate_ids),
            ("survivor", self.survivor_candidate_ids),
        ):
            if values != tuple(sorted(set(values))) or any(
                not _is_sha256(value) for value in values
            ):
                raise ValueError(
                    f"E2 {name} candidate IDs must be sorted unique SHA-256"
                )
        if self.completed_cells_sha256 != content_sha256(self.completed_stage_cell_ids):
            raise ValueError("E2 completed-cell digest does not match its identities")
        if not set(self.completed_stage_cell_ids) <= set(
            self.completed_lineage_cell_ids
        ):
            raise ValueError("E2 completed stage must be contained in its lineage")
        if not set(self.survivor_candidate_ids) <= set(self.source_candidate_ids):
            raise ValueError("E2 survivors must be a subset of the source stage")
        if self.status == "SURVIVORS":
            if self.stage_index >= len(E2_HALVING_STAGES) - 1:
                raise ValueError("the final E2 stage must lock one final recipe")
            if (
                not self.survivor_candidate_ids
                or self.final_recipe_candidate_id is not None
            ):
                raise ValueError("intermediate E2 receipts require survivors only")
        elif self.status == "FINAL_RECIPE":
            if self.stage_index != len(E2_HALVING_STAGES) - 1:
                raise ValueError("only the final E2 stage can lock a recipe")
            if (
                self.final_recipe_candidate_id is None
                or self.survivor_candidate_ids != (self.final_recipe_candidate_id,)
            ):
                raise ValueError("final E2 receipt must lock exactly one survivor")
        elif self.status == "BLOCKED":
            if (
                self.survivor_candidate_ids
                or self.final_recipe_candidate_id is not None
            ):
                raise ValueError("blocked E2 receipts cannot promote candidates")
        else:
            raise ValueError("invalid E2 survivor status")
        _require_text("E2 survivor reason_code", self.reason_code)
        if self.selection_state != "sealed_before_next_stage_unblinding":
            raise ValueError("E2 survivor selection must precede next-stage unblinding")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def reduce_e2_activation(
    registry: ExperimentRegistry,
    *,
    e1_receipt: ExperimentReceipt,
    pareto: E1ParetoArtifact,
    stage_index: int,
    prior_reduction: E2StageReductionArtifact | None = None,
) -> ReducerActivationArtifact:
    """Materialize only stage zero or the prior receipt's exact survivors."""

    if stage_index not in range(len(E2_HALVING_STAGES)):
        raise ValueError("E2 activation stage is outside the registered grid")
    _validate_direct_receipt(registry, e1_receipt, "E1")
    if (
        pareto.registry_sha256 != registry.sha256
        or pareto.runtime_sha256 != e1_receipt.runtime_sha256
        or pareto.split_sha256 != e1_receipt.split_sha256
    ):
        raise ValueError("E2 activation identity does not match the E1 receipt")
    outputs = _receipt_outputs(e1_receipt)
    if outputs.get("dflash_pareto_set") != pareto.sha256:
        raise ValueError("E1 receipt does not bind the supplied Pareto artifact")
    if outputs.get("common_downstream_load") != pareto.common_load_sha256:
        raise ValueError("E1 receipt does not bind the E2 common load")
    prior_survivors: E2SurvivorReceipt | None = None
    if stage_index == 0:
        if prior_reduction is not None:
            raise ValueError("E2 stage zero cannot consume a prior raw reduction")
        geometry_ids = {row.sha256 for row in pareto.surviving_geometries}
        selected_candidate_ids: set[str] | None = None
        source_selection_sha256 = pareto.sha256
    else:
        if not isinstance(prior_reduction, E2StageReductionArtifact):
            raise TypeError("later E2 stages require the prior raw reduction")
        prior_survivors = prior_reduction.survivor_receipt
        if (
            prior_reduction.registry_sha256 != registry.sha256
            or prior_reduction.runtime_sha256 != pareto.runtime_sha256
            or prior_reduction.split_sha256 != pareto.split_sha256
            or prior_reduction.stage_index != stage_index - 1
            or prior_reduction.reducer_protocol_sha256 != E2_HALVING_PROTOCOL_SHA256
            or prior_survivors.registry_sha256 != registry.sha256
            or prior_survivors.runtime_sha256 != pareto.runtime_sha256
            or prior_survivors.split_sha256 != pareto.split_sha256
            or prior_survivors.stage_index != stage_index - 1
            or prior_survivors.status != "SURVIVORS"
        ):
            raise ValueError("E2 prior raw reduction has wrong lineage or round")
        geometry_ids = set()
        selected_candidate_ids = set(prior_survivors.survivor_candidate_ids)
        source_selection_sha256 = prior_reduction.sha256

    cells = registry.cells_for("E2")
    current = tuple(cell for cell in cells if _e2_stage(cell) == stage_index)
    current_runnable = tuple(cell for cell in current if cell.runnable)
    pairs = _e2_candidate_pairs(current_runnable)
    selected_cells: set[str] = {
        cell.cell_id
        for cell in current_runnable
        if cell.identity.method in {"target_only", "static"}
    }
    if stage_index == 0:
        available_geometry_ids = {
            E1GeometryIdentity(
                candidate.scope,
                candidate.parameterization,
                candidate.rank,
                candidate.alpha_over_rank,
            ).sha256
            for candidate, _ in pairs.values()
        }
        if not geometry_ids <= available_geometry_ids:
            raise ValueError("E1 Pareto geometry is absent from the runnable E2 grid")
        selected_pair_ids = {
            candidate_id
            for candidate_id, (candidate, _) in pairs.items()
            if E1GeometryIdentity(
                candidate.scope,
                candidate.parameterization,
                candidate.rank,
                candidate.alpha_over_rank,
            ).sha256
            in geometry_ids
        }
    else:
        if selected_candidate_ids is None:
            raise ValueError("later E2 activation lost its survivor identities")
        if not selected_candidate_ids <= pairs.keys():
            raise ValueError("prior E2 survivors are absent from the next-stage grid")
        selected_pair_ids = selected_candidate_ids
    for candidate_id in selected_pair_ids:
        selected_cells.update(cell.cell_id for cell in pairs[candidate_id][1])
    if not selected_pair_ids:
        raise ValueError("E2 activation cannot materialize an empty adaptive stage")
    reference_methods = {
        cell.identity.method
        for cell in current_runnable
        if cell.cell_id in selected_cells
        and cell.identity.method in {"target_only", "static"}
    }
    if reference_methods != {"target_only", "static"}:
        raise ValueError("E2 activation requires both registered reference baselines")

    rows: list[CellDisposition] = []
    for cell in cells:
        cell_stage = _e2_stage(cell)
        if not cell.runnable:
            status = (
                DispositionStatus.NOT_APPLICABLE
                if cell.status is CellStatus.NOT_APPLICABLE
                else DispositionStatus.BLOCKED
            )
            reason = cell.reason_code
        elif cell.cell_id in selected_cells:
            status = DispositionStatus.ACTIVATED
            reason = f"halving_stage_{stage_index}_selected"
        elif cell_stage < stage_index:
            if prior_reduction is None:
                raise ValueError("prior E2 completion lineage is missing")
            if cell.cell_id in set(prior_survivors.completed_lineage_cell_ids):
                status = DispositionStatus.COMPLETED_PRIOR_ROUND
                reason = "completed_prior_halving_round"
            else:
                status = DispositionStatus.NOT_APPLICABLE
                reason = "not_selected_in_prior_halving_round"
        elif cell_stage > stage_index:
            status = DispositionStatus.DEFERRED
            reason = "awaiting_prior_survivor_receipt"
        else:
            status = DispositionStatus.NOT_APPLICABLE
            reason = "not_selected_by_e1_or_prior_halving_round"
        rows.append(CellDisposition(cell.cell_id, status, reason))
    return _make_activation(
        registry=registry,
        experiment="E2",
        dependency_receipt=e1_receipt,
        runtime_sha256=pareto.runtime_sha256,
        split_sha256=pareto.split_sha256,
        source_selection_sha256=source_selection_sha256,
        activation_round=f"halving_{stage_index}",
        rows=rows,
        reason_code="successive_halving_reducer_activation",
        reducer_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
    )


def verify_e2_activation(
    registry: ExperimentRegistry,
    *,
    e1_receipt: ExperimentReceipt,
    pareto: E1ParetoArtifact,
    stage_index: int,
    artifact: ReducerActivationArtifact,
    prior_reduction: E2StageReductionArtifact | None = None,
) -> None:
    """Reject hand-authored cell SHA lists and cross-round activation edits."""

    expected = reduce_e2_activation(
        registry,
        e1_receipt=e1_receipt,
        pareto=pareto,
        stage_index=stage_index,
        prior_reduction=prior_reduction,
    )
    if artifact != expected:
        raise ValueError("E2 activation is not the exact reducer-generated artifact")


@dataclass(frozen=True)
class E2CandidateEvaluation:
    candidate_id: str
    evidence_sha256: str
    safety_passed: bool
    confidence_pareto: bool
    min_tts_l0_static_goodput_ratio: float
    confidence_lower_goodput_ratio: float
    hbm_bytes: int
    p99_itl_us: int
    exposed_update_us: int
    minimum_published_updates: int
    safety_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256("E2 evaluation candidate_id", self.candidate_id)
        _require_sha256("E2 evaluation evidence_sha256", self.evidence_sha256)
        if not isinstance(self.safety_passed, bool) or not isinstance(
            self.confidence_pareto, bool
        ):
            raise TypeError("E2 safety and Pareto flags must be booleans")
        if (
            not math.isfinite(self.min_tts_l0_static_goodput_ratio)
            or self.min_tts_l0_static_goodput_ratio <= 0
            or not math.isfinite(self.confidence_lower_goodput_ratio)
            or self.confidence_lower_goodput_ratio <= 0
            or self.confidence_lower_goodput_ratio
            > self.min_tts_l0_static_goodput_ratio
        ):
            raise ValueError("E2 goodput ratios must be finite, positive, and ordered")
        for name in (
            "hbm_bytes",
            "p99_itl_us",
            "exposed_update_us",
            "minimum_published_updates",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        if self.safety_reason_codes != tuple(sorted(set(self.safety_reason_codes))):
            raise ValueError("E2 safety reasons must be sorted and unique")
        if self.safety_passed and self.safety_reason_codes:
            raise ValueError("safe E2 evaluations cannot carry failure reasons")
        if not self.safety_passed and not self.safety_reason_codes:
            raise ValueError("unsafe E2 evaluations require a reason")
        if self.safety_passed and self.minimum_published_updates < 1:
            raise ValueError("safe E2 evaluations require a published update")


@dataclass(frozen=True)
class RawEvidenceRunBinding:
    """Substantive receipt-bound run provenance emitted only by raw reducers."""

    schema_version: int
    cell_id: str
    experiment: str
    method: str
    scientific_unit: str
    config_sha256: str
    rank_config_sha256s: tuple[str, ...]
    run_id: str
    rank_count: int
    model_pair: str
    runtime_sha256: str
    split_sha256: str
    corpus_sha256: str
    arrival_trace_sha256: str
    request_ids_sha256: str
    sampling_profile_sha256: str
    model_lock_sha256: str
    patched_sglang_tree: str
    run_nonce_sha256: str
    topology_sha256: str
    experiment_budget_sha256: str
    physical_gpu_uuids: tuple[str, ...]
    terminal_receipt_sha256s: tuple[str, ...]
    hardware_receipt_sha256: str
    budget_observation_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only raw run-binding schema version 1 is supported")
        for name in ("experiment", "method", "scientific_unit", "run_id", "model_pair"):
            _require_text(f"raw run {name}", getattr(self, name))
        for name in (
            "cell_id",
            "config_sha256",
            "runtime_sha256",
            "split_sha256",
            "corpus_sha256",
            "arrival_trace_sha256",
            "request_ids_sha256",
            "sampling_profile_sha256",
            "model_lock_sha256",
            "run_nonce_sha256",
            "topology_sha256",
            "experiment_budget_sha256",
            "hardware_receipt_sha256",
            "budget_observation_sha256",
        ):
            _require_sha256(f"raw run {name}", getattr(self, name))
        if (
            not isinstance(self.patched_sglang_tree, str)
            or len(self.patched_sglang_tree) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.patched_sglang_tree
            )
        ):
            raise ValueError("raw run patched tree must be a lower-case Git tree")
        if (
            not isinstance(self.rank_count, int)
            or isinstance(self.rank_count, bool)
            or self.rank_count < 1
            or len(self.rank_config_sha256s) != self.rank_count
            or len(self.terminal_receipt_sha256s) != self.rank_count
            or len(self.physical_gpu_uuids) != self.rank_count
        ):
            raise ValueError("raw run binding lacks exact rank coverage")
        for name, values in (
            ("rank configs", self.rank_config_sha256s),
            ("terminal receipts", self.terminal_receipt_sha256s),
        ):
            if len(set(values)) != len(values) or any(
                not _is_sha256(value) for value in values
            ):
                raise ValueError(f"raw run {name} must be unique SHA-256")
        if len(set(self.physical_gpu_uuids)) != self.rank_count or any(
            not isinstance(value, str) or not value.strip()
            for value in self.physical_gpu_uuids
        ):
            raise ValueError("raw run physical GPU identities are incomplete")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2StageEvidenceArtifact:
    """Tuning-only reducer output reconstructed from receipt-bound raw evidence."""

    schema_version: int
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    fixed_instance_gpu_count: int
    inventory_host_id: str
    activation_sha256: str
    stage_index: int
    prior_stage_reduction_sha256: str | None
    raw_evidence_manifest_sha256: str
    completed_cell_ids: tuple[str, ...]
    terminal_receipt_sha256s: tuple[str, ...]
    hardware_receipt_sha256s: tuple[str, ...]
    budget_observation_sha256s: tuple[str, ...]
    run_bindings: tuple[RawEvidenceRunBinding, ...]
    evaluations: tuple[E2CandidateEvaluation, ...]
    reducer_protocol_sha256: str
    data_source: Literal["tuning_only"]
    confirmation_data_visible: bool

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("only E2 stage-evidence schema version 3 is supported")
        for name in (
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "activation_sha256",
            "raw_evidence_manifest_sha256",
            "reducer_protocol_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            not isinstance(self.fixed_instance_gpu_count, int)
            or isinstance(self.fixed_instance_gpu_count, bool)
            or self.fixed_instance_gpu_count < 1
        ):
            raise ValueError("E2 stage inventory GPU count must be positive")
        _require_text("E2 stage inventory host", self.inventory_host_id)
        if self.reducer_protocol_sha256 != E2_HALVING_PROTOCOL_SHA256:
            raise ValueError("E2 stage evidence uses an unregistered reducer")
        if self.stage_index not in range(len(E2_HALVING_STAGES)):
            raise ValueError("E2 stage evidence names an invalid round")
        if self.stage_index == 0:
            if self.prior_stage_reduction_sha256 is not None:
                raise ValueError("E2 stage zero cannot bind a prior raw reduction")
        else:
            _require_sha256(
                "prior_stage_reduction_sha256",
                self.prior_stage_reduction_sha256,
            )
        for name, values in (
            ("completed cells", self.completed_cell_ids),
            ("terminal receipts", self.terminal_receipt_sha256s),
            ("hardware receipts", self.hardware_receipt_sha256s),
            ("budget observations", self.budget_observation_sha256s),
        ):
            if values != tuple(sorted(set(values))) or any(
                not _is_sha256(value) for value in values
            ):
                raise ValueError(f"E2 {name} must be sorted unique SHA-256")
            if not values:
                raise ValueError(f"E2 stage evidence requires {name}")
        if len(self.hardware_receipt_sha256s) != len(self.completed_cell_ids) or len(
            self.budget_observation_sha256s
        ) != len(self.completed_cell_ids):
            raise ValueError(
                "E2 hardware/budget evidence must cover every completed cell"
            )
        if len(self.run_bindings) != len(self.completed_cell_ids):
            raise ValueError("E2 run bindings must cover every completed cell")
        evaluation_count = len(self.evaluations)
        expected_method_counts = {
            "target_only": 1,
            "static": 1,
            "tts": evaluation_count,
            "l0": evaluation_count,
        }
        observed_method_counts = {
            method: sum(binding.method == method for binding in self.run_bindings)
            for method in expected_method_counts
        }
        if (
            tuple(binding.cell_id for binding in self.run_bindings)
            != self.completed_cell_ids
            or len({binding.sha256 for binding in self.run_bindings})
            != len(self.run_bindings)
            or len({binding.run_id for binding in self.run_bindings})
            != len(self.run_bindings)
            or len({binding.run_nonce_sha256 for binding in self.run_bindings})
            != len(self.run_bindings)
            or evaluation_count < 1
            or observed_method_counts != expected_method_counts
            or any(
                binding.experiment != "E2"
                or binding.runtime_sha256 != self.runtime_sha256
                or binding.split_sha256 != self.split_sha256
                or len(binding.physical_gpu_uuids) > self.fixed_instance_gpu_count
                or binding.scientific_unit != f"halving_{self.stage_index}"
                for binding in self.run_bindings
            )
            or tuple(
                sorted(
                    digest
                    for binding in self.run_bindings
                    for digest in binding.terminal_receipt_sha256s
                )
            )
            != self.terminal_receipt_sha256s
            or tuple(
                sorted(binding.hardware_receipt_sha256 for binding in self.run_bindings)
            )
            != self.hardware_receipt_sha256s
            or tuple(
                sorted(
                    binding.budget_observation_sha256 for binding in self.run_bindings
                )
            )
            != self.budget_observation_sha256s
        ):
            raise ValueError("E2 substantive run provenance differs from its stage")
        candidate_ids = tuple(row.candidate_id for row in self.evaluations)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("E2 evaluations must be sorted by unique candidate ID")
        if self.data_source != "tuning_only" or self.confirmation_data_visible:
            raise ValueError(
                "E2 stage evidence must be tuning-only and pre-confirmation"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    @property
    def run_binding_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(binding.sha256 for binding in self.run_bindings))


@dataclass(frozen=True)
class E2StageReductionArtifact:
    """Raw-evidence reduction and its inseparable successive-halving decision."""

    schema_version: int
    activation: ReducerActivationArtifact
    stage_evidence: E2StageEvidenceArtifact
    survivor_receipt: E2SurvivorReceipt

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E2 stage-reduction schema version 1 is supported")
        activation = self.activation
        evidence = self.stage_evidence
        receipt = self.survivor_receipt
        if (
            activation.plan.experiment != "E2"
            or activation.reducer_protocol_sha256 != E2_HALVING_PROTOCOL_SHA256
            or activation.plan.registry_sha256 != evidence.registry_sha256
            or activation.plan.runtime_sha256 != evidence.runtime_sha256
            or activation.plan.split_sha256 != evidence.split_sha256
            or activation.sha256 != evidence.activation_sha256
            or activation.plan.activation_round != f"halving_{evidence.stage_index}"
            or tuple(sorted(activation.plan.activated_cell_ids))
            != evidence.completed_cell_ids
            or (
                evidence.stage_index > 0
                and activation.plan.source_selection_sha256
                != evidence.prior_stage_reduction_sha256
            )
            or evidence.reducer_protocol_sha256 != E2_HALVING_PROTOCOL_SHA256
            or receipt.halving_protocol_sha256 != E2_HALVING_PROTOCOL_SHA256
            or receipt.registry_sha256 != evidence.registry_sha256
            or receipt.runtime_sha256 != evidence.runtime_sha256
            or receipt.split_sha256 != evidence.split_sha256
            or receipt.stage_index != evidence.stage_index
            or receipt.source_activation_sha256 != evidence.activation_sha256
            or receipt.prior_stage_reduction_sha256
            != evidence.prior_stage_reduction_sha256
            or receipt.completed_stage_cell_ids != evidence.completed_cell_ids
            or receipt.tuning_evidence_sha256 != evidence.sha256
        ):
            raise ValueError("E2 stage decision is not bound to its raw reduction")

    @property
    def registry_sha256(self) -> str:
        return self.stage_evidence.registry_sha256

    @property
    def runtime_sha256(self) -> str:
        return self.stage_evidence.runtime_sha256

    @property
    def split_sha256(self) -> str:
        return self.stage_evidence.split_sha256

    @property
    def stage_index(self) -> int:
        return self.stage_evidence.stage_index

    @property
    def reducer_protocol_sha256(self) -> str:
        return self.stage_evidence.reducer_protocol_sha256

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2FinalRecipeArtifact:
    """The only sealable E2 output, materialized from raw halving-stage four."""

    schema_version: int
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    final_stage_reduction_sha256: str
    source_activation_sha256: str
    candidate_id: str
    candidate: E2CandidateIdentity
    selection_state: Literal["locked_from_raw_halving_3"]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E2 final-recipe schema version 1 is supported")
        for name in (
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "final_stage_reduction_sha256",
            "source_activation_sha256",
            "candidate_id",
        ):
            _require_sha256(name, getattr(self, name))
        if self.candidate.sha256 != self.candidate_id:
            raise ValueError("E2 final recipe candidate identity is inconsistent")
        if self.selection_state != "locked_from_raw_halving_3":
            raise ValueError("E2 final recipe must be locked from raw halving_3")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def materialize_e2_final_recipe(
    registry: ExperimentRegistry,
    reduction: E2StageReductionArtifact,
) -> E2FinalRecipeArtifact:
    """Rebuild the exact recipe output authorized by raw final-stage evidence."""

    if not isinstance(reduction, E2StageReductionArtifact):
        raise TypeError("E2 final recipe requires an exact stage reduction")
    receipt = reduction.survivor_receipt
    final_stage = len(E2_HALVING_STAGES) - 1
    if (
        reduction.registry_sha256 != registry.sha256
        or reduction.stage_index != final_stage
        or receipt.status != "FINAL_RECIPE"
        or receipt.final_recipe_candidate_id is None
    ):
        raise ValueError("E2 recipe requires a raw FINAL_RECIPE halving_3 reduction")
    active = {
        cell.cell_id: cell
        for cell in registry.cells_for("E2")
        if cell.cell_id in set(reduction.activation.plan.activated_cell_ids)
    }
    pairs = _e2_candidate_pairs(tuple(active.values()))
    candidate_id = receipt.final_recipe_candidate_id
    try:
        candidate, _ = pairs[candidate_id]
    except KeyError as exc:
        raise ValueError(
            "E2 final candidate is absent from the activated stage"
        ) from exc
    if receipt.survivor_candidate_ids != (candidate_id,):
        raise ValueError("E2 final reduction does not lock exactly one candidate")
    return E2FinalRecipeArtifact(
        schema_version=1,
        registry_sha256=registry.sha256,
        runtime_sha256=reduction.runtime_sha256,
        split_sha256=reduction.split_sha256,
        final_stage_reduction_sha256=reduction.sha256,
        source_activation_sha256=reduction.activation.sha256,
        candidate_id=candidate_id,
        candidate=candidate,
        selection_state="locked_from_raw_halving_3",
    )


def _reduce_e2_successive_halving(
    activation: ReducerActivationArtifact,
    *,
    registry: ExperimentRegistry,
    stage_evidence: E2StageEvidenceArtifact,
) -> E2StageReductionArtifact:
    """Apply the registered 25% safety/Pareto rule and per-family floor."""

    if activation.plan.experiment != "E2":
        raise ValueError("successive halving requires an E2 activation")
    if activation.reducer_protocol_sha256 != E2_HALVING_PROTOCOL_SHA256:
        raise ValueError("E2 activation does not use the registered halving protocol")
    if activation.plan.registry_sha256 != registry.sha256:
        raise ValueError("E2 activation belongs to another registry")
    stage_index = int(activation.plan.activation_round.removeprefix("halving_"))
    if (
        stage_evidence.registry_sha256 != registry.sha256
        or stage_evidence.runtime_sha256 != activation.plan.runtime_sha256
        or stage_evidence.split_sha256 != activation.plan.split_sha256
        or stage_evidence.activation_sha256 != activation.sha256
        or stage_evidence.stage_index != stage_index
        or (
            stage_index > 0
            and activation.plan.source_selection_sha256
            != stage_evidence.prior_stage_reduction_sha256
        )
    ):
        raise ValueError("E2 raw-evidence reduction has wrong activation lineage")
    active_ids = set(activation.plan.activated_cell_ids)
    completed = stage_evidence.completed_cell_ids
    if len(completed) != len(set(completed)) or set(completed) != active_ids:
        raise ValueError("E2 sealing requires exact completion of every activated cell")
    known = {cell.cell_id: cell for cell in registry.cells_for("E2")}
    active_cells = tuple(known[cell_id] for cell_id in active_ids)
    pairs = _e2_candidate_pairs(active_cells)
    source_ids = tuple(sorted(pairs))
    by_candidate = {row.candidate_id: row for row in stage_evidence.evaluations}
    if set(by_candidate) != set(source_ids):
        raise ValueError("E2 evaluations must exactly cover activated candidates")
    eligible = tuple(
        row
        for row in by_candidate.values()
        if row.safety_passed and row.confidence_pareto
    )
    families = {
        candidate.family
        for candidate, _ in (pairs[candidate_id] for candidate_id in source_ids)
    }
    eligible_by_family: dict[tuple[str, str], list[E2CandidateEvaluation]] = {
        family: [] for family in families
    }
    for row in eligible:
        eligible_by_family[pairs[row.candidate_id][0].family].append(row)
    missing_families = tuple(
        sorted(family for family, rows in eligible_by_family.items() if not rows)
    )
    completed_sha = content_sha256(tuple(sorted(completed)))
    prior_completed = tuple(
        row.cell_id
        for row in activation.dispositions
        if row.status is DispositionStatus.COMPLETED_PRIOR_ROUND
    )
    completed_stage = tuple(sorted(completed))
    completed_lineage = tuple(sorted(set(prior_completed) | set(completed_stage)))
    common = {
        "schema_version": 2,
        "registry_sha256": registry.sha256,
        "runtime_sha256": activation.plan.runtime_sha256,
        "split_sha256": activation.plan.split_sha256,
        "halving_protocol_sha256": E2_HALVING_PROTOCOL_SHA256,
        "stage_index": stage_index,
        "source_activation_sha256": activation.sha256,
        "prior_stage_reduction_sha256": (stage_evidence.prior_stage_reduction_sha256),
        "completed_cells_sha256": completed_sha,
        "completed_stage_cell_ids": completed_stage,
        "completed_lineage_cell_ids": completed_lineage,
        "tuning_evidence_sha256": stage_evidence.sha256,
        "source_candidate_ids": source_ids,
        "selection_state": "sealed_before_next_stage_unblinding",
    }

    def finish(receipt: E2SurvivorReceipt) -> E2StageReductionArtifact:
        return E2StageReductionArtifact(
            schema_version=1,
            activation=activation,
            stage_evidence=stage_evidence,
            survivor_receipt=receipt,
        )

    if missing_families:
        return finish(
            E2SurvivorReceipt(
                **common,
                survivor_candidate_ids=(),
                final_recipe_candidate_id=None,
                status="BLOCKED",
                reason_code="optimizer_schedule_family_floor_unmet",
            )
        )
    ranking = lambda row: (
        -row.min_tts_l0_static_goodput_ratio,
        row.hbm_bytes,
        row.p99_itl_us,
        row.exposed_update_us,
        row.candidate_id,
    )
    ranked = tuple(sorted(eligible, key=ranking))
    if stage_index == len(E2_HALVING_STAGES) - 1:
        survivors = (ranked[0].candidate_id,)
        return finish(
            E2SurvivorReceipt(
                **common,
                survivor_candidate_ids=survivors,
                final_recipe_candidate_id=survivors[0],
                status="FINAL_RECIPE",
                reason_code="registered_final_recipe_locked",
            )
        )
    floor_ids = {
        min(rows, key=ranking).candidate_id for rows in eligible_by_family.values()
    }
    fraction_count = math.ceil(
        len(source_ids) * _E2_RETENTION_NUMERATOR / _E2_RETENTION_DENOMINATOR
    )
    survivor_count = max(fraction_count, len(floor_ids) * _E2_FAMILY_FLOOR)
    selected = set(floor_ids)
    for row in ranked:
        if len(selected) >= survivor_count:
            break
        selected.add(row.candidate_id)
    survivors = tuple(sorted(selected))
    return finish(
        E2SurvivorReceipt(
            **common,
            survivor_candidate_ids=survivors,
            final_recipe_candidate_id=None,
            status="SURVIVORS",
            reason_code="registered_quarter_retention_with_family_floor",
        )
    )


def reduce_e2_successive_halving(
    activation: ReducerActivationArtifact,
    *,
    registry: ExperimentRegistry,
    stage_evidence: E2StageEvidenceArtifact,
) -> E2StageReductionArtifact:
    """Reject the retired caller-summary E2 selection path."""

    del activation, registry, stage_evidence
    raise ValueError(
        "E2 halving requires the first-party raw terminal-evidence reducer"
    )


def _without_block_phase(variant: str) -> str:
    for prefix in ("excluded_pilot:", "final_candidate:"):
        if variant.startswith(prefix):
            return variant.removeprefix(prefix)
    raise ValueError("confirmation cell variant lacks a block phase")


def _normalized_confirmation_backend(cell: ExperimentCell) -> str:
    if cell.identity.experiment == "E3b" and cell.identity.backend == "NONE":
        return "DFLASH"
    return cell.identity.backend


def _confirmation_load_sha256(cell: ExperimentCell) -> str:
    identity = cell.identity
    return content_sha256(
        {
            "arrival": identity.arrival,
            "slo": identity.slo,
            "variant": _without_block_phase(identity.variant),
            "concurrency": identity.concurrency,
            "load_factor": identity.load_factor,
            "cohort": identity.cohort,
            "cohort_count": identity.cohort_count,
        }
    )


def _width_panel(cell: ExperimentCell) -> str:
    variant = _without_block_phase(cell.identity.variant)
    if cell.identity.experiment == "E3b":
        panel = variant.rsplit(":", maxsplit=1)[-1]
        if panel not in {"matched", "deployment_optimal"}:
            raise ValueError("E3b cell has an invalid width panel")
        return panel
    return "not_applicable"


@dataclass(frozen=True)
class ConfirmationFamilyIdentity:
    schema_version: int
    registry_sha256: str
    experiment: str
    model: str
    backend: str
    task: str
    context: int
    regime: str
    arrival: str
    load_arrival_sha256: str
    width_panel: str
    topology: str
    cohort_family: str
    cohort_count: int
    method_family: tuple[str, ...]
    runtime_sha256: str
    split_sha256: str
    trace_sha256: str
    sampling_sha256: str
    hardware_envelope_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only confirmation-family schema version 1 is supported")
        if self.experiment not in {"E3b", "E5"}:
            raise ValueError("confirmation families are defined only for E3b/E5")
        for name in (
            "registry_sha256",
            "load_arrival_sha256",
            "runtime_sha256",
            "split_sha256",
            "trace_sha256",
            "sampling_sha256",
            "hardware_envelope_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "model",
            "backend",
            "task",
            "regime",
            "arrival",
            "width_panel",
            "topology",
            "cohort_family",
        ):
            _require_text(name, getattr(self, name))
        if (
            not isinstance(self.context, int)
            or isinstance(self.context, bool)
            or self.context < 1
        ):
            raise ValueError("confirmation context must be a positive integer")
        if (
            not isinstance(self.cohort_count, int)
            or isinstance(self.cohort_count, bool)
            or self.cohort_count < 1
        ):
            raise ValueError("confirmation cohort_count must be positive")
        if self.method_family != CORE_METHODS:
            raise ValueError(
                "primary confirmation family must bind all four core methods"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def derive_confirmation_family(
    registry: ExperimentRegistry,
    *,
    cell_id: str,
    runtime_sha256: str,
    split_sha256: str,
    trace_sha256: str,
    sampling_sha256: str,
    hardware_envelope_sha256: str,
) -> ConfirmationFamilyIdentity:
    cells = {cell.cell_id: cell for cell in registry.cells}
    try:
        cell = cells[cell_id]
    except KeyError as exc:
        raise ValueError(
            "confirmation family source cell is outside the registry"
        ) from exc
    identity = cell.identity
    if identity.experiment not in {"E3b", "E5"}:
        raise ValueError("confirmation family source must be an E3b/E5 cell")
    return ConfirmationFamilyIdentity(
        schema_version=1,
        registry_sha256=registry.sha256,
        experiment=identity.experiment,
        model=identity.model,
        backend=_normalized_confirmation_backend(cell),
        task=identity.task,
        context=identity.context or 0,
        regime=identity.regime,
        arrival=identity.arrival,
        load_arrival_sha256=_confirmation_load_sha256(cell),
        width_panel=_width_panel(cell),
        topology=identity.topology,
        cohort_family=identity.cohort,
        cohort_count=identity.cohort_count,
        method_family=CORE_METHODS,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        trace_sha256=trace_sha256,
        sampling_sha256=sampling_sha256,
        hardware_envelope_sha256=hardware_envelope_sha256,
    )


def _family_cells(
    registry: ExperimentRegistry, family: ConfirmationFamilyIdentity
) -> tuple[ExperimentCell, ...]:
    if family.registry_sha256 != registry.sha256:
        raise ValueError("confirmation family belongs to another registry")
    matches = tuple(
        cell
        for cell in registry.cells_for(family.experiment)
        if (
            cell.identity.model == family.model
            and _normalized_confirmation_backend(cell) == family.backend
            and cell.identity.task == family.task
            and cell.identity.context == family.context
            and cell.identity.regime == family.regime
            and cell.identity.arrival == family.arrival
            and _confirmation_load_sha256(cell) == family.load_arrival_sha256
            and _width_panel(cell) == family.width_panel
            and cell.identity.topology == family.topology
            and cell.identity.cohort == family.cohort_family
            and cell.identity.cohort_count == family.cohort_count
            and cell.identity.method in family.method_family
        )
    )
    expected = len(PILOT_BLOCKS + FINAL_BLOCKS) * len(CORE_METHODS)
    if len(matches) != expected:
        raise ValueError(
            "confirmation family must contain every method in every registered block"
        )
    for block in PILOT_BLOCKS + FINAL_BLOCKS:
        methods = {
            cell.identity.method for cell in matches if cell.identity.block == block
        }
        if methods != set(CORE_METHODS):
            raise ValueError(
                "confirmation family block is not a complete paired method set"
            )
    return tuple(sorted(matches, key=lambda cell: cell.cell_id))


@dataclass(frozen=True)
class FamilyActivationArtifact:
    schema_version: int
    family: ConfirmationFamilyIdentity
    activation_round: Literal["excluded_pilots", "final_prefix"]
    power_plan_sha256: str | None
    dispositions: tuple[CellDisposition, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only family activation schema version 1 is supported")
        if self.activation_round == "excluded_pilots":
            if self.power_plan_sha256 is not None:
                raise ValueError("pilot activation cannot consume a power plan")
        elif self.activation_round == "final_prefix":
            _require_sha256("family power_plan_sha256", self.power_plan_sha256)
        else:
            raise ValueError("invalid family activation round")
        if (
            tuple(sorted(self.dispositions, key=lambda row: row.cell_id))
            != self.dispositions
        ):
            raise ValueError("family dispositions must be sorted")
        if len({row.cell_id for row in self.dispositions}) != len(self.dispositions):
            raise ValueError("family dispositions must have unique cell IDs")

    @property
    def activated_cell_ids(self) -> tuple[str, ...]:
        return tuple(
            row.cell_id
            for row in self.dispositions
            if row.status is DispositionStatus.ACTIVATED
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def materialize_confirmation_pilots(
    registry: ExperimentRegistry, family: ConfirmationFamilyIdentity
) -> FamilyActivationArtifact:
    cells = _family_cells(registry, family)
    pilot_cells = tuple(cell for cell in cells if cell.identity.block in PILOT_BLOCKS)
    family_runnable = all(cell.runnable for cell in pilot_cells)
    rows: list[CellDisposition] = []
    for cell in cells:
        if not cell.runnable:
            status = (
                DispositionStatus.NOT_APPLICABLE
                if cell.status is CellStatus.NOT_APPLICABLE
                else DispositionStatus.BLOCKED
            )
            reason = cell.reason_code
        elif cell.identity.block in PILOT_BLOCKS and family_runnable:
            status = DispositionStatus.ACTIVATED
            reason = "family_excluded_pilot"
        elif cell.identity.block in PILOT_BLOCKS:
            status = DispositionStatus.BLOCKED
            reason = "family_pilot_pair_incomplete"
        else:
            status = DispositionStatus.DEFERRED
            reason = "awaiting_family_power_lock"
        rows.append(CellDisposition(cell.cell_id, status, reason))
    artifact = FamilyActivationArtifact(
        schema_version=1,
        family=family,
        activation_round="excluded_pilots",
        power_plan_sha256=None,
        dispositions=tuple(sorted(rows, key=lambda row: row.cell_id)),
    )
    expected = len(PILOT_BLOCKS) * len(CORE_METHODS)
    if len(artifact.activated_cell_ids) not in {0, expected}:
        raise ValueError("family pilot activation must be empty or four paired blocks")
    return artifact


def verify_confirmation_pilot_activation(
    registry: ExperimentRegistry,
    *,
    family: ConfirmationFamilyIdentity,
    artifact: FamilyActivationArtifact,
) -> None:
    expected = materialize_confirmation_pilots(registry, family)
    if artifact != expected:
        raise ValueError("family pilot activation is not reducer-generated")


def family_pilot_block_id(family: ConfirmationFamilyIdentity, block: int) -> str:
    if block not in PILOT_BLOCKS:
        raise ValueError("family pilot block is outside the excluded pilot prefix")
    return content_sha256(
        {"schema_version": 1, "family_sha256": family.sha256, "pilot_block": block}
    )


@dataclass(frozen=True)
class ConfirmationFamilyPowerPlan:
    schema_version: int
    family: ConfirmationFamilyIdentity
    pilot_activation_sha256: str
    completed_pilot_cells_sha256: str
    pilot_evidence_sha256: str
    power_sizing: PowerSizingPlan
    status: Literal["POWERED", "UNDERPOWERED"]
    selected_final_blocks: int | None
    selected_final_prefix: tuple[int, ...]
    reason_code: str
    selection_state: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only family power schema version 1 is supported")
        for name in (
            "pilot_activation_sha256",
            "completed_pilot_cells_sha256",
            "pilot_evidence_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        expected_block_ids = tuple(
            family_pilot_block_id(self.family, block) for block in PILOT_BLOCKS
        )
        _validate_power_sizing(self.power_sizing, expected_block_ids)
        if self.status == "POWERED":
            if (
                not isinstance(self.selected_final_blocks, int)
                or isinstance(self.selected_final_blocks, bool)
                or not MINIMUM_FINAL_BLOCKS
                <= self.selected_final_blocks
                <= MAXIMUM_FINAL_BLOCKS
            ):
                raise ValueError("POWERED family plans require 12--20 final blocks")
            if self.selected_final_prefix != FINAL_BLOCKS[: self.selected_final_blocks]:
                raise ValueError(
                    "family confirmation must activate the exact final prefix"
                )
            if self.power_sizing.status != "READY" or (
                self.power_sizing.selected_final_blocks != self.selected_final_blocks
            ):
                raise ValueError("family POWERED status differs from power sizing")
        elif self.status == "UNDERPOWERED":
            if self.selected_final_blocks is not None or self.selected_final_prefix:
                raise ValueError(
                    "UNDERPOWERED family plans cannot activate final blocks"
                )
            if self.power_sizing.status != "UNDERPOWERED":
                raise ValueError("family UNDERPOWERED status differs from power sizing")
        else:
            raise ValueError("invalid family power status")
        _require_text("family power reason_code", self.reason_code)
        if self.selection_state != "sealed_before_confirmation_unblinding":
            raise ValueError("family power plan must be sealed before confirmation")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ConfirmationFamilyPowerReductionArtifact:
    """Power decision inseparably bound to reducer-validated raw pilot evidence."""

    schema_version: int
    plan: ConfirmationFamilyPowerPlan
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    fixed_instance_gpu_count: int
    inventory_host_id: str
    raw_evidence_manifest_sha256: str
    terminal_receipt_sha256s: tuple[str, ...]
    hardware_receipt_sha256s: tuple[str, ...]
    budget_observation_sha256s: tuple[str, ...]
    run_bindings: tuple[RawEvidenceRunBinding, ...]
    reducer_protocol_sha256: str
    data_source: Literal["excluded_pilots_only"]
    confirmation_data_visible: bool

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(
                "only confirmation family-power reduction schema version 2 is supported"
            )
        for name in ("inventory_sha256", "inventory_source_receipt_sha256"):
            _require_sha256(f"family {name}", getattr(self, name))
        if (
            not isinstance(self.fixed_instance_gpu_count, int)
            or isinstance(self.fixed_instance_gpu_count, bool)
            or self.fixed_instance_gpu_count < 1
        ):
            raise ValueError("family inventory GPU count must be positive")
        _require_text("family inventory host", self.inventory_host_id)
        _require_sha256(
            "family raw_evidence_manifest_sha256",
            self.raw_evidence_manifest_sha256,
        )
        if self.raw_evidence_manifest_sha256 != self.plan.pilot_evidence_sha256:
            raise ValueError("family power plan is not bound to its raw pilot manifest")
        if (
            self.reducer_protocol_sha256
            != CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256
        ):
            raise ValueError("family power reduction uses an unregistered protocol")
        expected_cells = len(PILOT_BLOCKS) * len(CORE_METHODS)
        for name, values in (
            ("terminal receipts", self.terminal_receipt_sha256s),
            ("hardware receipts", self.hardware_receipt_sha256s),
            ("budget observations", self.budget_observation_sha256s),
        ):
            if values != tuple(sorted(set(values))) or any(
                not _is_sha256(value) for value in values
            ):
                raise ValueError(f"family power {name} must be sorted unique SHA-256")
            if not values:
                raise ValueError(f"family power reduction requires {name}")
        if (
            len(self.hardware_receipt_sha256s) != expected_cells
            or len(self.budget_observation_sha256s) != expected_cells
            or len(self.run_bindings) != expected_cells
            or len(self.terminal_receipt_sha256s) < expected_cells
        ):
            raise ValueError("family power provenance lacks exact pilot-cell coverage")
        if (
            len({binding.cell_id for binding in self.run_bindings}) != expected_cells
            or len({binding.sha256 for binding in self.run_bindings}) != expected_cells
            or tuple(binding.cell_id for binding in self.run_bindings)
            != tuple(sorted(binding.cell_id for binding in self.run_bindings))
            or len({binding.run_id for binding in self.run_bindings}) != expected_cells
            or len({binding.run_nonce_sha256 for binding in self.run_bindings})
            != expected_cells
            or content_sha256(tuple(binding.cell_id for binding in self.run_bindings))
            != self.plan.completed_pilot_cells_sha256
            or {
                (binding.scientific_unit, binding.method)
                for binding in self.run_bindings
            }
            != {
                (f"excluded_pilot_{block}", method)
                for block in PILOT_BLOCKS
                for method in CORE_METHODS
            }
            or any(
                binding.experiment != self.family.experiment
                or binding.runtime_sha256 != self.family.runtime_sha256
                or binding.split_sha256 != self.family.split_sha256
                or len(binding.physical_gpu_uuids) > self.fixed_instance_gpu_count
                or binding.scientific_unit
                not in {f"excluded_pilot_{block}" for block in PILOT_BLOCKS}
                for binding in self.run_bindings
            )
            or tuple(
                sorted(
                    digest
                    for binding in self.run_bindings
                    for digest in binding.terminal_receipt_sha256s
                )
            )
            != self.terminal_receipt_sha256s
            or tuple(
                sorted(binding.hardware_receipt_sha256 for binding in self.run_bindings)
            )
            != self.hardware_receipt_sha256s
            or tuple(
                sorted(
                    binding.budget_observation_sha256 for binding in self.run_bindings
                )
            )
            != self.budget_observation_sha256s
        ):
            raise ValueError(
                "family substantive run provenance differs from pilot evidence"
            )
        if self.data_source != "excluded_pilots_only" or self.confirmation_data_visible:
            raise ValueError(
                "family power reduction must use hidden excluded-pilot evidence"
            )

    @property
    def family(self) -> ConfirmationFamilyIdentity:
        return self.plan.family

    @property
    def status(self) -> Literal["POWERED", "UNDERPOWERED"]:
        return self.plan.status

    @property
    def selected_final_blocks(self) -> int | None:
        return self.plan.selected_final_blocks

    @property
    def selected_final_prefix(self) -> tuple[int, ...]:
        return self.plan.selected_final_prefix

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    @property
    def run_binding_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(binding.sha256 for binding in self.run_bindings))


def _validate_power_sizing(
    power_sizing: PowerSizingPlan, expected_block_ids: tuple[str, ...]
) -> None:
    if power_sizing.pilot_block_ids != expected_block_ids:
        raise ValueError("pilot variance belongs to another confirmation family")
    if (
        power_sizing.minimum_final_blocks != MINIMUM_FINAL_BLOCKS
        or power_sizing.maximum_final_blocks != MAXIMUM_FINAL_BLOCKS
        or power_sizing.target_power != PRIMARY_TARGET_POWER
        or power_sizing.family_alpha != PRIMARY_FAMILY_ALPHA
        or power_sizing.adjusted_alpha != PRIMARY_FAMILY_ALPHA / len(PRIMARY_CONTRASTS)
        or power_sizing.minimum_relative_effect != PRIMARY_MINIMUM_RELATIVE_EFFECT
        or power_sizing.minimum_log_effect
        != math.log1p(PRIMARY_MINIMUM_RELATIVE_EFFECT)
    ):
        raise ValueError("family power plan changes a preregistered power constant")
    deviations = dict(power_sizing.pilot_log_standard_deviations)
    if set(deviations) != set(PRIMARY_CONTRASTS) or any(
        not math.isfinite(value) or value <= 0 for value in deviations.values()
    ):
        raise ValueError("family pilot variance is incomplete or invalid")
    grid = {
        (cell.contrast, cell.final_blocks): cell.power
        for cell in power_sizing.power_grid
    }
    expected_grid = {
        (contrast, blocks)
        for blocks in range(MINIMUM_FINAL_BLOCKS, MAXIMUM_FINAL_BLOCKS + 1)
        for contrast in PRIMARY_CONTRASTS
    }
    if len(grid) != len(power_sizing.power_grid) or set(grid) != expected_grid:
        raise ValueError("family power grid is incomplete or duplicated")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in grid.values()):
        raise ValueError("family power grid contains an invalid probability")
    for contrast in PRIMARY_CONTRASTS:
        values = tuple(
            grid[(contrast, blocks)]
            for blocks in range(MINIMUM_FINAL_BLOCKS, MAXIMUM_FINAL_BLOCKS + 1)
        )
        if any(later < earlier for earlier, later in pairwise(values)):
            raise ValueError("family power must be monotone in final block count")
    selected = next(
        (
            blocks
            for blocks in range(MINIMUM_FINAL_BLOCKS, MAXIMUM_FINAL_BLOCKS + 1)
            if all(
                grid[(contrast, blocks)] >= PRIMARY_TARGET_POWER
                for contrast in PRIMARY_CONTRASTS
            )
        ),
        None,
    )
    expected_status = "READY" if selected is not None else "UNDERPOWERED"
    if (
        power_sizing.status != expected_status
        or power_sizing.selected_final_blocks != selected
    ):
        raise ValueError("family power decision does not match its registered grid")


def _seal_confirmation_family_power(
    *,
    registry: ExperimentRegistry,
    family: ConfirmationFamilyIdentity,
    pilot_activation: FamilyActivationArtifact,
    completed_pilot_cell_ids: Sequence[str],
    pilot_evidence_sha256: str,
    power_sizing: PowerSizingPlan,
    confirmation_data_visible: bool,
) -> ConfirmationFamilyPowerPlan:
    if confirmation_data_visible:
        raise ValueError(
            "family power cannot be selected after confirmation unblinding"
        )
    verify_confirmation_pilot_activation(
        registry, family=family, artifact=pilot_activation
    )
    if (
        pilot_activation.family != family
        or pilot_activation.activation_round != "excluded_pilots"
    ):
        raise ValueError("family power plan requires its own pilot activation")
    if len(pilot_activation.activated_cell_ids) != len(PILOT_BLOCKS) * len(
        CORE_METHODS
    ):
        raise ValueError("a blocked family cannot enter power sizing")
    completed = tuple(completed_pilot_cell_ids)
    if len(completed) != len(set(completed)) or set(completed) != set(
        pilot_activation.activated_cell_ids
    ):
        raise ValueError("family power lock requires exact completion of its pilots")
    _require_sha256("pilot_evidence_sha256", pilot_evidence_sha256)
    expected_block_ids = tuple(
        family_pilot_block_id(family, block) for block in PILOT_BLOCKS
    )
    _validate_power_sizing(power_sizing, expected_block_ids)
    if power_sizing.status == "READY":
        if power_sizing.selected_final_blocks is None:
            raise ValueError("READY family power sizing lacks a final block count")
        status: Literal["POWERED", "UNDERPOWERED"] = "POWERED"
        selected = power_sizing.selected_final_blocks
        prefix = FINAL_BLOCKS[:selected]
        reason = "registered_family_power_target_met"
    elif power_sizing.status == "UNDERPOWERED":
        status = "UNDERPOWERED"
        selected = None
        prefix = ()
        reason = "registered_family_underpowered"
    else:
        raise ValueError("power sizing plan has an invalid status")
    return ConfirmationFamilyPowerPlan(
        schema_version=1,
        family=family,
        pilot_activation_sha256=pilot_activation.sha256,
        completed_pilot_cells_sha256=content_sha256(tuple(sorted(completed))),
        pilot_evidence_sha256=pilot_evidence_sha256,
        power_sizing=power_sizing,
        status=status,
        selected_final_blocks=selected,
        selected_final_prefix=prefix,
        reason_code=reason,
        selection_state="sealed_before_confirmation_unblinding",
    )


def seal_confirmation_family_power(
    *,
    registry: ExperimentRegistry,
    family: ConfirmationFamilyIdentity,
    pilot_activation: FamilyActivationArtifact,
    completed_pilot_cell_ids: Sequence[str],
    pilot_evidence_sha256: str,
    power_sizing: PowerSizingPlan,
    confirmation_data_visible: bool,
) -> ConfirmationFamilyPowerPlan:
    """Reject the retired caller-summary family-power construction path."""

    del (
        registry,
        family,
        pilot_activation,
        completed_pilot_cell_ids,
        pilot_evidence_sha256,
        power_sizing,
        confirmation_data_visible,
    )
    raise ValueError(
        "family power requires the first-party raw terminal-evidence reducer"
    )


def materialize_confirmation_prefix(
    registry: ExperimentRegistry,
    *,
    family: ConfirmationFamilyIdentity,
    reduction: ConfirmationFamilyPowerReductionArtifact,
    pilot_activation: FamilyActivationArtifact,
) -> FamilyActivationArtifact:
    if not isinstance(reduction, ConfirmationFamilyPowerReductionArtifact):
        raise TypeError("confirmation prefix requires a raw family-power reduction")
    plan = reduction.plan
    if plan.family != family:
        raise ValueError("confirmation power plan belongs to another family")
    verify_confirmation_pilot_activation(
        registry, family=family, artifact=pilot_activation
    )
    if (
        pilot_activation.family != family
        or pilot_activation.activation_round != "excluded_pilots"
        or plan.pilot_activation_sha256 != pilot_activation.sha256
        or plan.completed_pilot_cells_sha256
        != content_sha256(tuple(sorted(pilot_activation.activated_cell_ids)))
    ):
        raise ValueError("confirmation plan does not bind this family's exact pilots")
    cells = _family_cells(registry, family)
    selected_blocks = set(plan.selected_final_prefix)
    rows: list[CellDisposition] = []
    for cell in cells:
        block = cell.identity.block
        if block in PILOT_BLOCKS:
            status = DispositionStatus.COMPLETED_PRIOR_ROUND
            reason = "excluded_pilot_not_reused_for_confirmation"
        elif not cell.runnable:
            status = (
                DispositionStatus.NOT_APPLICABLE
                if cell.status is CellStatus.NOT_APPLICABLE
                else DispositionStatus.BLOCKED
            )
            reason = cell.reason_code
        elif block in selected_blocks:
            status = DispositionStatus.ACTIVATED
            reason = "family_power_selected_final_prefix"
        else:
            status = (
                DispositionStatus.BLOCKED
                if plan.status == "UNDERPOWERED"
                else DispositionStatus.DEFERRED
            )
            reason = (
                "family_underpowered"
                if plan.status == "UNDERPOWERED"
                else "outside_family_selected_final_prefix"
            )
        rows.append(CellDisposition(cell.cell_id, status, reason))
    artifact = FamilyActivationArtifact(
        schema_version=1,
        family=family,
        activation_round="final_prefix",
        power_plan_sha256=reduction.sha256,
        dispositions=tuple(sorted(rows, key=lambda row: row.cell_id)),
    )
    expected = (
        0
        if plan.selected_final_blocks is None
        else plan.selected_final_blocks * len(CORE_METHODS)
    )
    if len(artifact.activated_cell_ids) != expected:
        raise ValueError("family final activation is not the exact selected prefix")
    return artifact


@dataclass(frozen=True)
class ExecutionSemanticsIdentity:
    """Every execution field that must remain equal across an evidence alias."""

    target_model: str
    target_revision: str
    runtime_sha256: str
    patched_tree_identity: str
    sampling_sha256: str
    seed: int
    request_corpus_sha256: str
    arrival_trace_sha256: str
    maximum_context_tokens: int
    maximum_output_tokens: int
    hardware_envelope_sha256: str
    topology: str
    rank_layout_sha256: str
    method: str
    method_implementation_sha256: str
    server_config_sha256: str
    evidence_schema: str
    output_token_contract_sha256: str
    timing_contract_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "target_model",
            "target_revision",
            "patched_tree_identity",
            "topology",
            "evidence_schema",
        ):
            _require_text(name, getattr(self, name))
        for name in (
            "runtime_sha256",
            "sampling_sha256",
            "request_corpus_sha256",
            "arrival_trace_sha256",
            "hardware_envelope_sha256",
            "rank_layout_sha256",
            "method_implementation_sha256",
            "server_config_sha256",
            "output_token_contract_sha256",
            "timing_contract_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_nonnegative_int("alias seed", self.seed)
        if self.maximum_context_tokens < 1 or self.maximum_output_tokens < 1:
            raise ValueError("alias context and output limits must be positive")
        if self.method != "target_only":
            raise ValueError(
                "only Target-only is initially eligible for evidence aliases"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class PresentationAxis:
    name: str
    value: str

    def __post_init__(self) -> None:
        _require_text("presentation axis name", self.name)
        _require_text("presentation axis value", self.value)


@dataclass(frozen=True)
class EvidenceAliasCandidate:
    cell_id: str
    semantics: ExecutionSemanticsIdentity
    presentation_axes: tuple[PresentationAxis, ...]

    def __post_init__(self) -> None:
        _require_sha256("alias candidate cell_id", self.cell_id)
        names = tuple(axis.name for axis in self.presentation_axes)
        if names != tuple(sorted(set(names))):
            raise ValueError("presentation axes must be sorted and unique")


_ELIGIBLE_ALIAS_REASONS = frozenset(
    {
        "target_only_backend_label_only",
        "target_only_breadth_reference",
        "identical_selected_width_panel",
        "identical_materialized_load_plan",
        "target_only_cross_analysis_reference",
    }
)
_PRESENTATION_ONLY_AXES = frozenset(
    {
        "analysis_panel",
        "backend_label",
        "breadth_panel_label",
        "load_panel_label",
        "width_panel_label",
    }
)


@dataclass(frozen=True)
class EvidenceAliasReceipt:
    schema_version: int
    source: EvidenceAliasCandidate
    target: EvidenceAliasCandidate
    source_evidence_sha256: str
    removed_presentation_axis: str
    reason_code: str
    analysis_state: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only evidence alias schema version 1 is supported")
        if self.source.cell_id == self.target.cell_id:
            raise ValueError(
                "an evidence alias requires distinct source and target cells"
            )
        _require_sha256("source_evidence_sha256", self.source_evidence_sha256)
        _require_text("removed_presentation_axis", self.removed_presentation_axis)
        if self.reason_code not in _ELIGIBLE_ALIAS_REASONS:
            raise ValueError("evidence alias reason is not registered")
        if self.removed_presentation_axis not in _PRESENTATION_ONLY_AXES:
            raise ValueError("evidence alias may remove only a presentation-only axis")
        if self.analysis_state != "sealed_before_analysis":
            raise ValueError("evidence alias must be sealed before analysis")
        if self.source.semantics != self.target.semantics:
            raise ValueError(
                "evidence alias execution semantics are not byte-equivalent"
            )
        source_axes = {axis.name: axis.value for axis in self.source.presentation_axes}
        target_axes = {axis.name: axis.value for axis in self.target.presentation_axes}
        if set(source_axes) != set(target_axes):
            raise ValueError("alias candidates expose different presentation axes")
        if self.removed_presentation_axis not in source_axes:
            raise ValueError("removed presentation axis is absent")
        if (
            source_axes[self.removed_presentation_axis]
            == target_axes[self.removed_presentation_axis]
        ):
            raise ValueError("removed presentation axis must actually differ")
        if {
            name: value
            for name, value in source_axes.items()
            if name != self.removed_presentation_axis
        } != {
            name: value
            for name, value in target_axes.items()
            if name != self.removed_presentation_axis
        }:
            raise ValueError(
                "alias candidates differ on more than one presentation axis"
            )

    @cached_property
    def dependence_unit_sha256(self) -> str:
        return content_sha256(
            {
                "source_cell_id": self.source.cell_id,
                "source_evidence_sha256": self.source_evidence_sha256,
                "semantics_sha256": self.source.semantics.sha256,
            }
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class AnalysisDependenceUnit:
    unit_sha256: str
    source_cell_id: str
    member_cell_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceDependenceMap:
    schema_version: int
    units: tuple[AnalysisDependenceUnit, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only evidence dependence schema version 1 is supported")
        if tuple(sorted(self.units, key=lambda row: row.unit_sha256)) != self.units:
            raise ValueError("dependence units must be sorted")
        if len({unit.unit_sha256 for unit in self.units}) != len(self.units):
            raise ValueError("dependence-unit identities must be unique")
        for unit in self.units:
            _require_sha256("dependence unit_sha256", unit.unit_sha256)
            _require_sha256("dependence source_cell_id", unit.source_cell_id)
            if unit.member_cell_ids != tuple(sorted(set(unit.member_cell_ids))):
                raise ValueError("dependence members must be sorted and unique")
            if unit.source_cell_id not in unit.member_cell_ids:
                raise ValueError("dependence source must be one of its members")
        members = tuple(cell for unit in self.units for cell in unit.member_cell_ids)
        if len(members) != len(set(members)):
            raise ValueError("one cell cannot belong to two dependence units")

    @property
    def independent_unit_count(self) -> int:
        return len(self.units)

    def unit_for(self, cell_id: str) -> str:
        matches = [
            unit.unit_sha256 for unit in self.units if cell_id in unit.member_cell_ids
        ]
        if len(matches) != 1:
            raise ValueError("cell does not resolve to exactly one dependence unit")
        return matches[0]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def build_evidence_dependence_map(
    *,
    direct_observation_cell_ids: Sequence[str],
    aliases: Sequence[EvidenceAliasReceipt],
) -> EvidenceDependenceMap:
    """Collapse aliases onto source observations for bootstrap/covariance units."""

    direct = tuple(direct_observation_cell_ids)
    if len(direct) != len(set(direct)) or any(
        not _is_sha256(value) for value in direct
    ):
        raise ValueError("direct observations must be unique cell SHA-256 values")
    alias_rows = tuple(aliases)
    targets = tuple(row.target.cell_id for row in alias_rows)
    if len(targets) != len(set(targets)):
        raise ValueError("an alias target can be defined only once")
    if set(targets) & set(direct):
        raise ValueError("an alias target cannot also be an independent observation")
    if any(row.source.cell_id not in set(direct) for row in alias_rows):
        raise ValueError(
            "alias chains are forbidden; every source needs direct evidence"
        )
    members: dict[str, list[str]] = {source: [source] for source in direct}
    unit_ids: dict[str, str] = {
        source: content_sha256({"direct_observation_cell_id": source})
        for source in direct
    }
    for row in alias_rows:
        members[row.source.cell_id].append(row.target.cell_id)
        existing = unit_ids[row.source.cell_id]
        if len(members[row.source.cell_id]) == 2:
            unit_ids[row.source.cell_id] = row.dependence_unit_sha256
        elif unit_ids[row.source.cell_id] != row.dependence_unit_sha256:
            raise ValueError("aliases from one source bind inconsistent evidence")
        if (
            len(members[row.source.cell_id]) > 2
            and existing != row.dependence_unit_sha256
        ):
            raise ValueError("aliases from one source bind inconsistent evidence")
    units = tuple(
        sorted(
            (
                AnalysisDependenceUnit(
                    unit_sha256=unit_ids[source],
                    source_cell_id=source,
                    member_cell_ids=tuple(sorted(values)),
                )
                for source, values in members.items()
            ),
            key=lambda row: row.unit_sha256,
        )
    )
    return EvidenceDependenceMap(schema_version=1, units=units)
