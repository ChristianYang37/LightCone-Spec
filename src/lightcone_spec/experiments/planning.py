"""Content-bound industrial planning, activation, budget, and alias contracts.

The registry intentionally declares the complete scientific envelope.  This
module is the reducer-owned boundary that turns sealed upstream decisions into
the much smaller set of cells that may actually be materialized.  Nothing in
this module launches a server or interprets confirmation results.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lightcone_spec.experiments.gpu_pool import GpuInventory, InterferenceEnvelope

from lightcone_spec.experiments.itl_authority import (
    ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256,
)
from lightcone_spec.experiments.load import ProductionLoadPlan
from lightcone_spec.experiments.registry import (
    CONFIRMATION_METHOD_ROLES,
    DRAFT_WIDTHS,
    E0_METHOD_ROLES,
    E2_DRAFT_WIDTH_SELECTOR,
    E2_HALVING_STAGES,
    FINAL_BLOCKS,
    INDUSTRIAL_EXPERIMENT_ORDER,
    PILOT_BLOCKS,
    AdaptationRecipeDeclaration,
    CellStatus,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    ScientificMethodRole,
    StageActivationPlan,
    WorkloadClass,
    content_sha256,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
    verify_registry_stage_activation,
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
_E1_SLICE_CELLS = 68
_E2_RETENTION_NUMERATOR = 1
_E2_RETENTION_DENOMINATOR = 4
_E2_FAMILY_FLOOR = 1
_E2_PROMOTION_MINIMA_BLOCKER = "e2_promotion_minima_unregistered"
E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON = "e1_common_load_authority_unregistered"


@dataclass(frozen=True)
class _E2PromotionMinimumAuthority:
    """Source-owned per-stage gate; callers can observe but never supply it."""

    stage_index: int
    minimum_launched_updates_per_adapted_method: int | None
    minimum_published_updates_per_adapted_method: int | None
    blocker_reason_code: str | None

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or self.stage_index < 0:
            raise ValueError("E2 promotion-minimum stage must be non-negative")
        launched = self.minimum_launched_updates_per_adapted_method
        published = self.minimum_published_updates_per_adapted_method
        if launched is None or published is None:
            if launched is not None or published is not None:
                raise ValueError("E2 promotion minima cannot be partially registered")
            if self.blocker_reason_code != _E2_PROMOTION_MINIMA_BLOCKER:
                raise ValueError("unregistered E2 promotion minima require the blocker")
            return
        if (
            type(launched) is not int
            or type(published) is not int
            or launched < 1
            or published < 1
            or published > launched
        ):
            raise ValueError(
                "registered E2 promotion minima must be positive integer counts "
                "with published no greater than launched"
            )
        if self.blocker_reason_code is not None:
            raise ValueError("registered E2 promotion minima cannot retain a blocker")

    @property
    def registered(self) -> bool:
        return self.minimum_launched_updates_per_adapted_method is not None


# The specifications require these values to be registered before E2, but do
# not provide them.  Keeping every stage explicit prevents stage zero (or a
# later round replay) from inheriting an invented default such as one update.
_E2_PROMOTION_MINIMA = tuple(
    _E2PromotionMinimumAuthority(
        stage_index=stage_index,
        minimum_launched_updates_per_adapted_method=None,
        minimum_published_updates_per_adapted_method=None,
        blocker_reason_code=_E2_PROMOTION_MINIMA_BLOCKER,
    )
    for stage_index in range(len(E2_HALVING_STAGES))
)
E2_PROMOTION_MINIMA_AUTHORITY_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e2_source_owned_promotion_minima",
        "stages": _E2_PROMOTION_MINIMA,
        "caller_override": "forbidden",
    }
)
E3A_RAW_SELECTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e3a_raw_capacity_selection_protocol",
        "coverage": "all_runnable_e3a_target_static_cells",
        "reference_load": (
            "smallest_concurrency_reaching_90pct_of_maximum_median_static_goodput"
        ),
        "matched_width": (
            "highest_worst_static_target_goodput_ratio_then_median_goodput_"
            "then_smallest_width"
        ),
        "primary_context_floor": 4096,
        "terminal_hardware_budget_evidence_required": True,
        "confirmation_data_forbidden": True,
    }
)
E1_RAW_PARETO_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 4,
        "kind": "e1_raw_geometry_pareto_protocol",
        "coverage": "exact_67_cell_selection_subset_of_68_cell_mechanism_slice",
        "candidate_grid": "two_l0_policy_lc_optimizer_anchors_per_geometry",
        "fixed_references": "one_target_one_static_one_frozen_tts_shared_per_slice",
        "mechanism_anchor": (
            "one_l0_naive_shared_per_slice_excluded_from_selection_evidence"
        ),
        "reference_safety": (
            "target_static_or_frozen_tts_invalid_or_incomplete_blocks_the_slice"
        ),
        "adaptive_safety": (
            "any_lc_candidate_safety_hardware_token_completion_or_zero_publish_"
            "failure_excludes_the_candidate_geometry"
        ),
        "objectives": (
            "maximize_worst_lc_confidence_lower_static_and_frozen_tts_goodput_ratio",
            "minimize_peak_hbm",
            "minimize_p99_itl",
            "minimize_exposed_update",
        ),
        "terminal_hardware_budget_evidence_required": True,
        "common_load_selection": "forbidden_separate_typed_authority_required",
        "e2_data_forbidden": True,
    }
)
E2_HALVING_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 8,
        "kind": "e2_successive_halving_protocol",
        "stages": E2_HALVING_STAGES,
        "draft_width_authority": E2_DRAFT_WIDTH_SELECTOR,
        "adaptation_recipe_authority": "registry_declaration_schema_v1",
        "common_load_authority": E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON,
        "retention_numerator": _E2_RETENTION_NUMERATOR,
        "retention_denominator": _E2_RETENTION_DENOMINATOR,
        "optimizer_schedule_family_floor": _E2_FAMILY_FLOOR,
        "promotion_minima_authority_sha256": (E2_PROMOTION_MINIMA_AUTHORITY_SHA256),
        "confidence_goodput": (
            "lc_vs_frozen_tts_and_static_paired_request_log_ratio_normal_"
            "95pct_lower_bounds_v1"
        ),
        "confidence_pareto": (
            "non_dominated_confidence_lower_goodput_hbm_p99_exposed_update_v1"
        ),
        "selection_references": ("target_only", "static", "frozen_tts"),
        "l0_naive_anchor": "mechanism_only_excluded_from_selection_evidence",
        "raw_run_binding_schema": 3,
        "stage_evidence_schema": 5,
        "ranking": (
            "safety_gate",
            "confidence_pareto",
            "descending_lc_vs_frozen_tts_confidence_lower_goodput_ratio",
            "descending_lc_vs_static_confidence_lower_goodput_ratio",
            "ascending_hbm_bytes",
            "ascending_p99_itl_us",
            "ascending_exposed_update_us",
            "candidate_sha256",
        ),
        "confirmation_data_forbidden": True,
        "inventory_authority": (
            "content_bound_gpu_inventory_sha_source_receipt_count_host_v1"
        ),
        "formal_itl_timestamp_authority_protocol_sha256": (
            ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256
        ),
        "formal_itl_timestamp_source": (
            "source_owned_result_pointer_then_path_bound_raw_receipt_full_coverage_v1"
        ),
        "current_release_itl_result_pointer": "BLOCKED_unavailable",
        "cpu_contract_only_itl": "BLOCKED",
        "sse_chunk_gap_interpolation": "forbidden",
    }
)
CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "confirmation_family_power_raw_reducer",
        "inputs": (
            "schema_v4_native_completion_receipts",
            "hardware_receipts",
            "budget_observations",
            "qualification_locks",
            "content_bound_gpu_inventory",
        ),
        "scientific_roles": CONFIRMATION_METHOD_ROLES,
        "primary_contrasts": ("lightcone_vs_tts", "lightcone_vs_static"),
        "secondary_contrasts": (
            "l0_naive_vs_tts",
            "lightcone_vs_l0_naive",
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
CONFIRMATION_AUXILIARY_ACTIVATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "confirmation_auxiliary_registry_activation",
        "stages": ("E3b", "E5"),
        "predicate": (
            "runnable_serving_cells_not_owned_by_any_complete_core_method_family"
        ),
        "caller_cell_ids": "forbidden",
        "all_auxiliary_cells_dispositioned": True,
        "schema_v4_native_completion_required": True,
    }
)
CAPACITY_MAXIMUM_SOURCE_AGE_NS = 300_000_000_000
CELL_CAPACITY_SIZING_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "industrial_cell_capacity_sizing_protocol",
        "inputs": (
            "path_bound_evidence_contract",
            "path_bound_model_staging_manifest",
            "path_bound_compile_overlay_plan",
        ),
        "maximum_attempt_bytes": (
            "maximum_evidence_bytes_plus_model_staging_bytes_plus_compile_overlay_bytes"
        ),
        "missing_provenance": "unresolved",
    }
)
CAPACITY_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "industrial_capacity_raw_authority_protocol",
        "inputs": (
            "path_bound_capacity_envelope",
            "path_bound_gpu_inventory",
            "path_bound_gpu_inventory_source_receipt",
            "path_bound_provider_quota_receipt",
            "path_bound_host_capacity_receipt",
            "path_bound_per_cell_sizing_receipts_and_provenance",
        ),
        "verification": "source_owned_release_ed25519_policy",
        "maximum_source_age_ns": CAPACITY_MAXIMUM_SOURCE_AGE_NS,
        "caller_selected_trust_root": "forbidden",
        "raw_revalidation_per_consumer": True,
        "current_release_without_verifier": "unresolved",
    }
)
BUDGET_MATERIALIZATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "industrial_budget_materialization_protocol",
        "activation_authority": "reducer_artifact_bundle_v1",
        "load_authority": "three_scenario_production_load_plan_v1",
        "capacity_authority_sha256": CAPACITY_AUTHORITY_PROTOCOL_SHA256,
        "maximum_attempts": "retry_allowance_plus_one",
        "duration_defaults_forbidden": True,
        "unresolved_disposition_required": True,
        "whole_inventory_billing": "wall_ms_times_inventory_gpu_count",
    }
)
BUDGET_MATERIALIZATION_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "industrial_budget_materialization_raw_authority_protocol",
        "materialization_protocol_sha256": BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
        "activation_authority": (
            "strict_path_bound_tagged_union_with_recursive_completion_sources_v3"
        ),
        "registry_authority": "generated_registry_replay_v3",
        "policy_authority": "path_bound_budget_policy_v1",
        "load_authority": "path_bound_cell_sorted_budget_load_bindings_v1",
        "capacity_authority": "path_bound_capacity_authority_v1",
        "declared_plan": "exact_first_party_rematerialization_only",
        "raw_revalidation_per_consumer": True,
        "serialized_activation_summary": "forbidden",
        "dependency_completion_authority": (
            "schema_v4_completed_cells_plus_recursive_raw_activation_"
            "terminal_hardware_budget_inventory_source_and_locked_outputs"
        ),
        "bare_dependency_receipts": "forbidden_for_formal_execution",
        "specialized_variants": (
            "e1_from_raw_e3a_selection",
            "e2_from_raw_e1_pareto_and_successive_halving",
            "confirmation_family_pilot_from_registered_raw_identity",
            "confirmation_family_final_from_four_raw_excluded_pilots",
            "confirmation_auxiliary_from_registry_non_family_predicate",
            "confirmation_stage_from_sorted_complete_family_raw_authorities",
        ),
        "confirmation_lock_scope": "per_family_incremental_after_exact_four_pilots",
        "confirmation_stage_completion": (
            "family_sha_sorted_exact_registry_coverage_aggregation_v1"
        ),
        "confirmation_auxiliary_protocol_sha256": (
            CONFIRMATION_AUXILIARY_ACTIVATION_PROTOCOL_SHA256
        ),
        "e2_prior_round_completion": (
            "stage_sorted_schema_v4_completed_cells_plus_native_trusted_terminal_"
            "authority_required"
        ),
        "serialized_activation_power_summary_or_cell_ids": "forbidden",
    }
)
EVIDENCE_ALIAS_REDUCER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "execution_derived_evidence_alias_reducer",
        "authority": (
            "registry_execution_plan_load_run_config_split_model_lock_"
            "terminal_native_budget_hardware_inventory_v1"
        ),
        "eligible_method": "target_only",
        "presentation_axis_count": 1,
        "target_independent_result": "forbidden",
        "legacy_self_described_receipts": "non_authoritative",
        "formal_analysis": "rerun_raw_reducer_and_compare_exact_artifact",
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


def _e2_promotion_minimum(stage_index: int) -> _E2PromotionMinimumAuthority:
    if type(stage_index) is not int or stage_index not in range(len(E2_HALVING_STAGES)):
        raise ValueError("E2 promotion-minimum stage is outside the registered grid")
    authority = _E2_PROMOTION_MINIMA[stage_index]
    if authority.stage_index != stage_index:
        raise RuntimeError("E2 source-owned promotion-minimum table is misordered")
    return authority


def _require_e2_promotion_authority(
    *, stage_index: int, status: str, reason_code: str
) -> _E2PromotionMinimumAuthority:
    authority = _e2_promotion_minimum(stage_index)
    if not authority.registered and (
        status != "BLOCKED" or reason_code != authority.blocker_reason_code
    ):
        raise ValueError(
            f"{_E2_PROMOTION_MINIMA_BLOCKER}: unregistered E2 promotion minima "
            "categorically forbid survivors"
        )
    return authority


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
class CapacityRawJsonBinding:
    """Durable name, bytes, and semantic identity for one raw JSON source.

    A digest without a path is not a capacity source.  The authority reducer
    reopens both names with no-follow semantics on every use and compares the
    bytes as well as the canonical JSON identity recorded here.
    """

    schema_version: int
    path: str
    sidecar_path: str
    semantic_sha256: str
    file_sha256: str
    sidecar_file_sha256: str
    size: int
    sidecar_size: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only raw capacity JSON binding schema 1 is supported")
        source = Path(self.path)
        sidecar = Path(self.sidecar_path)
        if not source.is_absolute() or source.resolve() != source:
            raise ValueError("raw capacity JSON path must be absolute and resolved")
        if sidecar != Path(f"{source}.sha256"):
            raise ValueError("raw capacity JSON sidecar path is not exact")
        if not sidecar.is_absolute() or sidecar.resolve() != sidecar:
            raise ValueError(
                "raw capacity JSON sidecar path must be absolute and resolved"
            )
        for name in (
            "semantic_sha256",
            "file_sha256",
            "sidecar_file_sha256",
        ):
            _require_sha256(f"raw capacity JSON {name}", getattr(self, name))
        if type(self.size) is not int or self.size < 1:
            raise ValueError("raw capacity JSON size must be positive")
        if type(self.sidecar_size) is not int or self.sidecar_size != 65:
            raise ValueError("raw capacity JSON sidecar must be one SHA-256 line")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "sidecar_path": self.sidecar_path,
            "semantic_sha256": self.semantic_sha256,
            "file_sha256": self.file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "size": self.size,
            "sidecar_size": self.sidecar_size,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class CapacityAuthorityBinding:
    """Serializable pointer to raw capacity sources and verifier receipt.

    This value never authorizes by itself.  ``BudgetPlan.require_ready`` calls
    the raw authority reducer, which reopens both bindings and every source
    named by the manifest under the source-owned release verifier policy.
    """

    schema_version: int
    source_manifest: CapacityRawJsonBinding
    verification_receipt: CapacityRawJsonBinding
    registry_sha256: str
    budget_inventory_sha256: str
    capacity_envelope_sha256: str
    gpu_inventory_sha256: str
    inventory_source_receipt_sha256: str
    trusted_verifier_policy_sha256: str
    authority_protocol_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only capacity authority binding schema 1 is supported")
        if (
            type(self.source_manifest) is not CapacityRawJsonBinding
            or type(self.verification_receipt) is not CapacityRawJsonBinding
        ):
            raise TypeError("capacity authority requires exact raw JSON bindings")
        if self.source_manifest.path == self.verification_receipt.path:
            raise ValueError("capacity manifest and verifier receipt must be distinct")
        for name in (
            "registry_sha256",
            "budget_inventory_sha256",
            "capacity_envelope_sha256",
            "gpu_inventory_sha256",
            "inventory_source_receipt_sha256",
            "trusted_verifier_policy_sha256",
        ):
            _require_sha256(f"capacity authority {name}", getattr(self, name))
        if self.authority_protocol_sha256 != CAPACITY_AUTHORITY_PROTOCOL_SHA256:
            raise ValueError("capacity authority binding uses another protocol")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


_BUDGET_RAW_JSON_ROLES = frozenset(
    {
        "generated_registry",
        "registry_stage_activation_manifest",
        "activation_runtime",
        "activation_split",
        "activation_dependency_receipt",
        "dependency_completed_cells",
        "dependency_gpu_inventory",
        "dependency_gpu_inventory_source_receipt",
        "dependency_locked_output",
        "e1_activation_authority_manifest",
        "e2_activation_authority_manifest",
        "confirmation_pilot_activation_authority_manifest",
        "confirmation_final_activation_authority_manifest",
        "confirmation_stage_aggregate_authority_manifest",
        "confirmation_auxiliary_activation_authority_manifest",
        "e3a_selection_raw_manifest",
        "e1_pareto_raw_manifest",
        "e2_stage_raw_manifest",
        "e2_stage_completed_cells",
        "confirmation_family_power_raw_manifest",
        "activation_hardware_envelope",
        "activation_trace",
        "activation_sampling",
        "family_pilot_completed_cells",
        "confirmation_family_completed_cells",
        "confirmation_auxiliary_completed_cells",
        "budget_policy",
        "budget_load_binding",
        "capacity_envelope",
        "declared_budget_plan",
    }
)


@dataclass(frozen=True)
class BudgetRawJsonBinding:
    """Immutable path, byte, canonical, and semantic identity for raw JSON.

    The existing CLI sidecar is a single canonical-JSON SHA-256 line at
    ``<path>.sha256``.  The semantic digest is deliberately separate: planning
    wrappers bind their ``artifact_sha256``, while manifests and generic
    runtime/split inputs bind their canonical JSON identity.
    """

    schema_version: int
    role: str
    path: str
    sidecar_path: str
    canonical_sha256: str
    semantic_sha256: str
    file_sha256: str
    sidecar_file_sha256: str
    size: int
    sidecar_size: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only budget raw JSON binding schema 1 is supported")
        if type(self.role) is not str or self.role not in _BUDGET_RAW_JSON_ROLES:
            raise ValueError("budget raw JSON binding role is unsupported")
        source = Path(self.path)
        sidecar = Path(self.sidecar_path)
        if not source.is_absolute() or source.resolve() != source:
            raise ValueError("budget raw JSON path must be absolute and resolved")
        if sidecar != Path(f"{source}.sha256"):
            raise ValueError("budget raw JSON sidecar path is not exact")
        if not sidecar.is_absolute() or sidecar.resolve() != sidecar:
            raise ValueError(
                "budget raw JSON sidecar path must be absolute and resolved"
            )
        for name in (
            "canonical_sha256",
            "semantic_sha256",
            "file_sha256",
            "sidecar_file_sha256",
        ):
            _require_sha256(f"budget raw JSON {name}", getattr(self, name))
        if (
            self.role
            in {
                "registry_stage_activation_manifest",
                "activation_runtime",
                "activation_split",
                "activation_dependency_receipt",
                "dependency_completed_cells",
                "dependency_locked_output",
                "e1_activation_authority_manifest",
                "e2_activation_authority_manifest",
                "confirmation_pilot_activation_authority_manifest",
                "confirmation_final_activation_authority_manifest",
                "confirmation_stage_aggregate_authority_manifest",
                "confirmation_auxiliary_activation_authority_manifest",
                "e3a_selection_raw_manifest",
                "e1_pareto_raw_manifest",
                "e2_stage_raw_manifest",
                "e2_stage_completed_cells",
                "confirmation_family_power_raw_manifest",
                "activation_hardware_envelope",
                "activation_trace",
                "activation_sampling",
                "family_pilot_completed_cells",
                "confirmation_family_completed_cells",
                "confirmation_auxiliary_completed_cells",
            }
            and self.semantic_sha256 != self.canonical_sha256
        ):
            raise ValueError(
                "unwrapped budget raw JSON semantic identity must be canonical"
            )
        if type(self.size) is not int or self.size < 1:
            raise ValueError("budget raw JSON size must be positive")
        if type(self.sidecar_size) is not int or self.sidecar_size != 65:
            raise ValueError("budget raw JSON sidecar must be one SHA-256 line")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "path": self.path,
            "sidecar_path": self.sidecar_path,
            "canonical_sha256": self.canonical_sha256,
            "semantic_sha256": self.semantic_sha256,
            "file_sha256": self.file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "size": self.size,
            "sidecar_size": self.sidecar_size,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class DependencyGpuInventoryAuthorityBinding:
    """Full raw GPU inventory plus the probe receipt that issued its source ID."""

    schema_version: int
    inventory: BudgetRawJsonBinding
    source_receipt: BudgetRawJsonBinding
    inventory_sha256: str
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only dependency GPU inventory authority schema 1 is supported"
            )
        if (
            type(self.inventory) is not BudgetRawJsonBinding
            or self.inventory.role != "dependency_gpu_inventory"
        ):
            raise TypeError("dependency completion requires a full raw GPU inventory")
        if (
            type(self.source_receipt) is not BudgetRawJsonBinding
            or self.source_receipt.role != "dependency_gpu_inventory_source_receipt"
        ):
            raise TypeError(
                "dependency completion requires the raw inventory source receipt"
            )
        _require_sha256("dependency GPU inventory", self.inventory_sha256)
        _require_sha256(
            "dependency GPU inventory source receipt", self.source_receipt_sha256
        )
        if (
            self.inventory_sha256 != self.inventory.semantic_sha256
            or self.source_receipt_sha256 != self.source_receipt.semantic_sha256
        ):
            raise ValueError("dependency GPU inventory redundant identities differ")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class DependencyLockedOutputAuthorityBinding:
    """One registered locked output reopened from its exact JSON artifact."""

    name: str
    artifact: BudgetRawJsonBinding

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or "\n" in self.name:
            raise ValueError("dependency locked-output name is invalid")
        if (
            type(self.artifact) is not BudgetRawJsonBinding
            or self.artifact.role != "dependency_locked_output"
        ):
            raise TypeError("dependency locked output requires an exact raw binding")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class RegistryStageDependencyCompletionAuthorityBinding:
    """Path closure that reconstructs one prior completion from raw activation."""

    schema_version: int
    receipt: BudgetRawJsonBinding
    completed_cells: BudgetRawJsonBinding
    activation: BudgetActivationAuthorityBinding
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    locked_outputs: tuple[DependencyLockedOutputAuthorityBinding, ...]
    receipt_sha256: str
    completed_authority_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only registry-stage dependency completion schema 1 is supported"
            )
        if (
            type(self.receipt) is not BudgetRawJsonBinding
            or self.receipt.role != "activation_dependency_receipt"
        ):
            raise TypeError("dependency completion requires its exact raw receipt")
        if (
            type(self.completed_cells) is not BudgetRawJsonBinding
            or self.completed_cells.role != "dependency_completed_cells"
        ):
            raise TypeError(
                "dependency completion requires its schema-v4 raw completed cells"
            )
        if not _is_budget_activation_authority_binding(self.activation):
            raise TypeError(
                "dependency completion requires its prior raw activation lineage"
            )
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("dependency completion requires exact inventory authority")
        if any(
            type(value) is not DependencyLockedOutputAuthorityBinding
            for value in self.locked_outputs
        ):
            raise TypeError("dependency completion locked outputs must be exact")
        names = tuple(value.name for value in self.locked_outputs)
        if names != tuple(sorted(set(names))):
            raise ValueError(
                "dependency completion locked outputs must be name-sorted and unique"
            )
        _require_sha256("dependency completion receipt", self.receipt_sha256)
        _require_sha256(
            "dependency completed authority", self.completed_authority_sha256
        )
        if self.receipt_sha256 != self.receipt.semantic_sha256:
            raise ValueError("dependency completion receipt identity differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class RegistryStageActivationAuthorityBinding:
    """Raw generic-stage reducer manifest plus every nested source it names."""

    schema_version: int
    kind: Literal["registry_stage_activation_manifest"]
    manifest: BudgetRawJsonBinding
    generated_registry: BudgetRawJsonBinding
    runtime: BudgetRawJsonBinding
    split: BudgetRawJsonBinding
    dependency_receipts: tuple[BudgetRawJsonBinding, ...]
    dependency_completion_authorities: tuple[
        RegistryStageDependencyCompletionAuthorityBinding, ...
    ]
    activation_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only registry-stage activation authority schema 1 is supported"
            )
        if self.kind != "registry_stage_activation_manifest":
            raise ValueError("budget activation authority kind is unsupported")
        for value, role in (
            (self.manifest, "registry_stage_activation_manifest"),
            (self.generated_registry, "generated_registry"),
            (self.runtime, "activation_runtime"),
            (self.split, "activation_split"),
        ):
            if type(value) is not BudgetRawJsonBinding or value.role != role:
                raise TypeError(
                    f"registry-stage activation authority requires exact {role}"
                )
        if any(
            type(value) is not BudgetRawJsonBinding
            or value.role != "activation_dependency_receipt"
            for value in self.dependency_receipts
        ):
            raise TypeError(
                "registry-stage activation dependencies require exact raw bindings"
            )
        dependency_paths = tuple(value.path for value in self.dependency_receipts)
        if len(dependency_paths) != len(set(dependency_paths)):
            raise ValueError("registry-stage activation dependencies are duplicated")
        if self.dependency_completion_authorities and (
            len(self.dependency_completion_authorities) != len(self.dependency_receipts)
            or tuple(value.receipt for value in self.dependency_completion_authorities)
            != self.dependency_receipts
        ):
            raise ValueError(
                "registry-stage dependency completions must exactly cover receipts"
            )
        sources = (
            self.manifest,
            self.generated_registry,
            self.runtime,
            self.split,
            *self.dependency_receipts,
        )
        raw_paths = tuple(
            path for source in sources for path in (source.path, source.sidecar_path)
        )
        if len(raw_paths) != len(set(raw_paths)):
            raise ValueError("registry-stage activation raw source paths alias")
        _require_sha256("registry-stage activation identity", self.activation_sha256)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E1ActivationAuthorityBinding:
    """Raw E3a selection evidence and completion lineage for one E1 slice."""

    schema_version: int
    kind: Literal["e1_activation_manifest"]
    manifest: BudgetRawJsonBinding
    generated_registry: BudgetRawJsonBinding
    runtime: BudgetRawJsonBinding
    split: BudgetRawJsonBinding
    dependency_receipt: BudgetRawJsonBinding
    dependency_completion_authority: RegistryStageDependencyCompletionAuthorityBinding
    selection_manifest: BudgetRawJsonBinding
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    hardware_envelope: BudgetRawJsonBinding
    activation_sha256: str
    selection_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E1 activation authority schema 1 is supported")
        if self.kind != "e1_activation_manifest":
            raise ValueError("E1 activation authority kind is invalid")
        for value, role in (
            (self.manifest, "e1_activation_authority_manifest"),
            (self.generated_registry, "generated_registry"),
            (self.runtime, "activation_runtime"),
            (self.split, "activation_split"),
            (self.dependency_receipt, "activation_dependency_receipt"),
            (self.selection_manifest, "e3a_selection_raw_manifest"),
            (self.hardware_envelope, "activation_hardware_envelope"),
        ):
            if type(value) is not BudgetRawJsonBinding or value.role != role:
                raise TypeError(f"E1 activation authority requires exact {role}")
        if (
            type(self.dependency_completion_authority)
            is not RegistryStageDependencyCompletionAuthorityBinding
            or self.dependency_completion_authority.receipt != self.dependency_receipt
        ):
            raise TypeError("E1 activation requires exact E3a completion authority")
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("E1 activation requires exact GPU inventory authority")
        if (
            self.dependency_completion_authority.activation.generated_registry
            != self.generated_registry
            or self.dependency_completion_authority.inventory_authority
            != self.inventory_authority
        ):
            raise ValueError("E1 activation swaps dependency registry or inventory")
        _require_sha256("E1 activation", self.activation_sha256)
        _require_sha256("E3a selection", self.selection_sha256)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2ActivationAuthorityBinding:
    """Raw E1 Pareto plus every prior raw halving round for one E2 stage."""

    schema_version: int
    kind: Literal["e2_activation_manifest"]
    manifest: BudgetRawJsonBinding
    generated_registry: BudgetRawJsonBinding
    runtime: BudgetRawJsonBinding
    split: BudgetRawJsonBinding
    dependency_receipt: BudgetRawJsonBinding
    dependency_completion_authority: RegistryStageDependencyCompletionAuthorityBinding
    pareto_manifest: BudgetRawJsonBinding
    prior_stage_manifests: tuple[BudgetRawJsonBinding, ...]
    prior_stage_completion_authorities: tuple[E2StageCompletionAuthorityBinding, ...]
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    hardware_envelope: BudgetRawJsonBinding
    stage_index: int
    activation_sha256: str
    pareto_sha256: str
    prior_stage_reduction_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E2 activation authority schema 1 is supported")
        if self.kind != "e2_activation_manifest":
            raise ValueError("E2 activation authority kind is invalid")
        for value, role in (
            (self.manifest, "e2_activation_authority_manifest"),
            (self.generated_registry, "generated_registry"),
            (self.runtime, "activation_runtime"),
            (self.split, "activation_split"),
            (self.dependency_receipt, "activation_dependency_receipt"),
            (self.pareto_manifest, "e1_pareto_raw_manifest"),
            (self.hardware_envelope, "activation_hardware_envelope"),
        ):
            if type(value) is not BudgetRawJsonBinding or value.role != role:
                raise TypeError(f"E2 activation authority requires exact {role}")
        if (
            type(self.dependency_completion_authority)
            is not RegistryStageDependencyCompletionAuthorityBinding
            or self.dependency_completion_authority.receipt != self.dependency_receipt
        ):
            raise TypeError("E2 activation requires exact E1 completion authority")
        if any(
            type(value) is not BudgetRawJsonBinding
            or value.role != "e2_stage_raw_manifest"
            for value in self.prior_stage_manifests
        ):
            raise TypeError("E2 prior rounds require exact raw stage manifests")
        if (
            type(self.stage_index) is not int
            or self.stage_index not in range(len(E2_HALVING_STAGES))
            or len(self.prior_stage_manifests) != self.stage_index
            or len(self.prior_stage_completion_authorities) != self.stage_index
        ):
            raise ValueError(
                "E2 raw stage and completion lineage must cover every prior round"
            )
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("E2 activation requires exact GPU inventory authority")
        if (
            self.dependency_completion_authority.activation.generated_registry
            != self.generated_registry
            or self.dependency_completion_authority.inventory_authority
            != self.inventory_authority
        ):
            raise ValueError("E2 activation swaps dependency registry or inventory")
        if any(
            type(authority) is not E2StageCompletionAuthorityBinding
            or authority.stage_activation.stage_index != expected_stage
            or authority.stage_activation.generated_registry != self.generated_registry
            or authority.inventory_authority != self.inventory_authority
            for expected_stage, authority in enumerate(
                self.prior_stage_completion_authorities
            )
        ):
            raise ValueError(
                "E2 prior completion authorities differ from the exact stage prefix"
            )
        _require_sha256("E2 activation", self.activation_sha256)
        _require_sha256("E1 Pareto", self.pareto_sha256)
        if self.stage_index == 0:
            if self.prior_stage_reduction_sha256 is not None:
                raise ValueError("E2 stage zero cannot bind a prior reduction")
        else:
            _require_sha256(
                "E2 prior stage reduction", self.prior_stage_reduction_sha256
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2StageCompletionAuthorityBinding:
    """Schema-v4 completion and native-terminal authority for one E2 round."""

    schema_version: int
    completed_cells: BudgetRawJsonBinding
    stage_activation: E2ActivationAuthorityBinding
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    completed_authority_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E2 stage completion schema 1 is supported")
        if (
            type(self.completed_cells) is not BudgetRawJsonBinding
            or self.completed_cells.role != "e2_stage_completed_cells"
        ):
            raise TypeError("E2 stage completion requires schema-v4 raw cells")
        if type(self.stage_activation) is not E2ActivationAuthorityBinding:
            raise TypeError("E2 stage completion requires raw E2 activation")
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("E2 stage completion requires exact inventory")
        if self.inventory_authority != self.stage_activation.inventory_authority:
            raise ValueError("E2 stage completion swaps its activation inventory")
        _require_sha256("E2 stage completed authority", self.completed_authority_sha256)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ConfirmationAuxiliaryActivationAuthorityBinding:
    """Raw authority for the deterministic non-family E3b/E5 remainder."""

    schema_version: int
    kind: Literal["confirmation_auxiliary_activation_manifest"]
    manifest: BudgetRawJsonBinding
    generated_registry: BudgetRawJsonBinding
    runtime: BudgetRawJsonBinding
    split: BudgetRawJsonBinding
    trace: BudgetRawJsonBinding
    sampling: BudgetRawJsonBinding
    dependency_receipts: tuple[BudgetRawJsonBinding, ...]
    dependency_completion_authorities: tuple[
        RegistryStageDependencyCompletionAuthorityBinding, ...
    ]
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    hardware_envelope: BudgetRawJsonBinding
    experiment: Literal["E3b", "E5"]
    activation_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only confirmation auxiliary activation authority schema 1 is supported"
            )
        if self.kind != "confirmation_auxiliary_activation_manifest":
            raise ValueError("confirmation auxiliary authority kind is invalid")
        for value, role in (
            (self.manifest, "confirmation_auxiliary_activation_authority_manifest"),
            (self.generated_registry, "generated_registry"),
            (self.runtime, "activation_runtime"),
            (self.split, "activation_split"),
            (self.trace, "activation_trace"),
            (self.sampling, "activation_sampling"),
            (self.hardware_envelope, "activation_hardware_envelope"),
        ):
            if type(value) is not BudgetRawJsonBinding or value.role != role:
                raise TypeError(
                    f"confirmation auxiliary authority requires exact {role}"
                )
        if self.experiment not in {"E3b", "E5"}:
            raise ValueError("confirmation auxiliary authority names another stage")
        if any(
            type(value) is not BudgetRawJsonBinding
            or value.role != "activation_dependency_receipt"
            for value in self.dependency_receipts
        ):
            raise TypeError("confirmation auxiliary dependencies require raw receipts")
        if (
            len(self.dependency_completion_authorities) != len(self.dependency_receipts)
            or tuple(value.receipt for value in self.dependency_completion_authorities)
            != self.dependency_receipts
        ):
            raise ValueError(
                "confirmation auxiliary completion authorities must cover dependencies"
            )
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("confirmation auxiliary requires exact inventory")
        if any(
            value.activation.generated_registry != self.generated_registry
            or value.inventory_authority != self.inventory_authority
            for value in self.dependency_completion_authorities
        ):
            raise ValueError(
                "confirmation auxiliary swaps dependency registry or inventory"
            )
        _require_sha256("confirmation auxiliary activation", self.activation_sha256)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ConfirmationAuxiliaryCompletionAuthorityBinding:
    """Schema-v4 completion for the deterministic non-family remainder."""

    schema_version: int
    completed_cells: BudgetRawJsonBinding
    activation: ConfirmationAuxiliaryActivationAuthorityBinding
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    completed_authority_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only confirmation auxiliary completion schema 1 is supported"
            )
        if (
            type(self.completed_cells) is not BudgetRawJsonBinding
            or self.completed_cells.role != "confirmation_auxiliary_completed_cells"
        ):
            raise TypeError(
                "confirmation auxiliary completion requires schema-v4 raw cells"
            )
        if type(self.activation) is not ConfirmationAuxiliaryActivationAuthorityBinding:
            raise TypeError("confirmation auxiliary completion requires raw activation")
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("confirmation auxiliary completion requires inventory")
        if self.inventory_authority != self.activation.inventory_authority:
            raise ValueError("confirmation auxiliary completion swaps inventory")
        _require_sha256(
            "confirmation auxiliary completed authority",
            self.completed_authority_sha256,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ConfirmationPilotActivationAuthorityBinding:
    """Raw registered family identity and dependency closure for pilot activation."""

    schema_version: int
    kind: Literal["confirmation_pilot_activation_manifest"]
    manifest: BudgetRawJsonBinding
    generated_registry: BudgetRawJsonBinding
    runtime: BudgetRawJsonBinding
    split: BudgetRawJsonBinding
    trace: BudgetRawJsonBinding
    sampling: BudgetRawJsonBinding
    dependency_receipts: tuple[BudgetRawJsonBinding, ...]
    dependency_completion_authorities: tuple[
        RegistryStageDependencyCompletionAuthorityBinding, ...
    ]
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    hardware_envelope: BudgetRawJsonBinding
    family_sha256: str
    activation_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only confirmation pilot activation authority schema 1 is supported"
            )
        if self.kind != "confirmation_pilot_activation_manifest":
            raise ValueError("confirmation pilot authority kind is invalid")
        for value, role in (
            (self.manifest, "confirmation_pilot_activation_authority_manifest"),
            (self.generated_registry, "generated_registry"),
            (self.runtime, "activation_runtime"),
            (self.split, "activation_split"),
            (self.trace, "activation_trace"),
            (self.sampling, "activation_sampling"),
            (self.hardware_envelope, "activation_hardware_envelope"),
        ):
            if type(value) is not BudgetRawJsonBinding or value.role != role:
                raise TypeError(f"confirmation pilot authority requires exact {role}")
        if any(
            type(value) is not BudgetRawJsonBinding
            or value.role != "activation_dependency_receipt"
            for value in self.dependency_receipts
        ):
            raise TypeError("confirmation pilot dependencies require raw receipts")
        if (
            len(self.dependency_completion_authorities) != len(self.dependency_receipts)
            or tuple(value.receipt for value in self.dependency_completion_authorities)
            != self.dependency_receipts
        ):
            raise ValueError(
                "confirmation pilot completion authorities must cover dependencies"
            )
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("confirmation pilot requires exact GPU inventory authority")
        if any(
            value.activation.generated_registry != self.generated_registry
            or value.inventory_authority != self.inventory_authority
            for value in self.dependency_completion_authorities
        ):
            raise ValueError(
                "confirmation pilot swaps dependency registry or inventory"
            )
        _require_sha256("confirmation family", self.family_sha256)
        _require_sha256("confirmation pilot activation", self.activation_sha256)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class FamilyPilotCompletionAuthorityBinding:
    """Raw schema-v4 pilot completion tied to its raw pilot activation."""

    schema_version: int
    completed_cells: BudgetRawJsonBinding
    pilot_activation: ConfirmationPilotActivationAuthorityBinding
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    completed_authority_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only family pilot completion schema 1 is supported")
        if (
            type(self.completed_cells) is not BudgetRawJsonBinding
            or self.completed_cells.role != "family_pilot_completed_cells"
        ):
            raise TypeError("family pilot completion requires schema-v4 raw cells")
        if (
            type(self.pilot_activation)
            is not ConfirmationPilotActivationAuthorityBinding
        ):
            raise TypeError("family pilot completion requires raw pilot activation")
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("family pilot completion requires exact inventory")
        if self.inventory_authority != self.pilot_activation.inventory_authority:
            raise ValueError("family pilot completion swaps pilot inventory")
        _require_sha256(
            "family pilot completed authority", self.completed_authority_sha256
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ConfirmationFinalActivationAuthorityBinding:
    """Four raw excluded pilots rerun into power and the exact final prefix."""

    schema_version: int
    kind: Literal["confirmation_final_activation_manifest"]
    manifest: BudgetRawJsonBinding
    generated_registry: BudgetRawJsonBinding
    pilot_activation_authority: ConfirmationPilotActivationAuthorityBinding
    pilot_completion_authority: FamilyPilotCompletionAuthorityBinding
    power_manifest: BudgetRawJsonBinding
    family_sha256: str
    power_reduction_sha256: str
    activation_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only confirmation final activation authority schema 1 is supported"
            )
        if self.kind != "confirmation_final_activation_manifest":
            raise ValueError("confirmation final authority kind is invalid")
        for value, role in (
            (self.manifest, "confirmation_final_activation_authority_manifest"),
            (self.generated_registry, "generated_registry"),
            (self.power_manifest, "confirmation_family_power_raw_manifest"),
        ):
            if type(value) is not BudgetRawJsonBinding or value.role != role:
                raise TypeError(f"confirmation final authority requires exact {role}")
        if (
            type(self.pilot_activation_authority)
            is not ConfirmationPilotActivationAuthorityBinding
            or type(self.pilot_completion_authority)
            is not FamilyPilotCompletionAuthorityBinding
        ):
            raise TypeError(
                "confirmation final authority requires pilot activation/completion"
            )
        if (
            self.pilot_completion_authority.pilot_activation
            != self.pilot_activation_authority
        ):
            raise ValueError("confirmation final authority swaps its pilot lineage")
        if (
            self.generated_registry
            != self.pilot_activation_authority.generated_registry
            or self.family_sha256 != self.pilot_activation_authority.family_sha256
        ):
            raise ValueError("confirmation final authority swaps registry or family")
        for name in (
            "family_sha256",
            "power_reduction_sha256",
            "activation_sha256",
        ):
            _require_sha256(f"confirmation final {name}", getattr(self, name))

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ConfirmationFamilyCompletionAuthorityBinding:
    """Schema-v4 final-prefix completion for one raw confirmation family."""

    schema_version: int
    completed_cells: BudgetRawJsonBinding
    final_activation: ConfirmationFinalActivationAuthorityBinding
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    completed_authority_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only confirmation family completion schema 1 is supported"
            )
        if (
            type(self.completed_cells) is not BudgetRawJsonBinding
            or self.completed_cells.role != "confirmation_family_completed_cells"
        ):
            raise TypeError(
                "confirmation family completion requires schema-v4 raw cells"
            )
        if (
            type(self.final_activation)
            is not ConfirmationFinalActivationAuthorityBinding
        ):
            raise TypeError(
                "confirmation family completion requires raw final activation"
            )
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("confirmation family completion requires exact inventory")
        if (
            self.inventory_authority
            != self.final_activation.pilot_activation_authority.inventory_authority
        ):
            raise ValueError("confirmation family completion swaps final inventory")
        _require_sha256(
            "confirmation family completed authority",
            self.completed_authority_sha256,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ConfirmationStageFamilyAuthorityBinding:
    """One family entry in a sorted exact-coverage stage aggregate."""

    schema_version: int
    family_sha256: str
    final_activation_authority: ConfirmationFinalActivationAuthorityBinding
    completion_authority: ConfirmationFamilyCompletionAuthorityBinding

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only confirmation stage-family schema 1 is supported")
        _require_sha256("confirmation aggregate family", self.family_sha256)
        if (
            type(self.final_activation_authority)
            is not ConfirmationFinalActivationAuthorityBinding
            or type(self.completion_authority)
            is not ConfirmationFamilyCompletionAuthorityBinding
        ):
            raise TypeError("confirmation aggregate family authorities must be exact")
        if (
            self.final_activation_authority.family_sha256 != self.family_sha256
            or self.completion_authority.final_activation
            != self.final_activation_authority
        ):
            raise ValueError("confirmation aggregate family lineage was swapped")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ConfirmationStageAggregateAuthorityBinding:
    """Independent all-family authority for one completed E3b/E5 stage."""

    schema_version: int
    kind: Literal["confirmation_stage_aggregate_manifest"]
    manifest: BudgetRawJsonBinding
    generated_registry: BudgetRawJsonBinding
    stage_receipt: BudgetRawJsonBinding
    stage_completed_cells: BudgetRawJsonBinding
    runtime: BudgetRawJsonBinding
    split: BudgetRawJsonBinding
    inventory_authority: DependencyGpuInventoryAuthorityBinding
    experiment: Literal["E3b", "E5"]
    families: tuple[ConfirmationStageFamilyAuthorityBinding, ...]
    auxiliary_completion_authority: (
        ConfirmationAuxiliaryCompletionAuthorityBinding | None
    )
    stage_receipt_sha256: str
    family_sha256s: tuple[str, ...]
    activated_cell_ids: tuple[str, ...]
    dispositions_sha256: str
    activation_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only confirmation stage aggregate authority schema 1 is supported"
            )
        if self.kind != "confirmation_stage_aggregate_manifest":
            raise ValueError("confirmation stage aggregate kind is invalid")
        for value, role in (
            (self.manifest, "confirmation_stage_aggregate_authority_manifest"),
            (self.generated_registry, "generated_registry"),
            (self.stage_receipt, "activation_dependency_receipt"),
            (self.stage_completed_cells, "dependency_completed_cells"),
            (self.runtime, "activation_runtime"),
            (self.split, "activation_split"),
        ):
            if type(value) is not BudgetRawJsonBinding or value.role != role:
                raise TypeError(f"confirmation stage aggregate requires exact {role}")
        if type(self.inventory_authority) is not DependencyGpuInventoryAuthorityBinding:
            raise TypeError("confirmation stage aggregate requires exact inventory")
        if self.experiment not in {"E3b", "E5"}:
            raise ValueError("confirmation stage aggregate names another stage")
        if any(
            type(value) is not ConfirmationStageFamilyAuthorityBinding
            for value in self.families
        ):
            raise TypeError("confirmation aggregate family entries must be exact")
        if (
            self.auxiliary_completion_authority is not None
            and type(self.auxiliary_completion_authority)
            is not ConfirmationAuxiliaryCompletionAuthorityBinding
        ):
            raise TypeError("confirmation aggregate auxiliary completion must be exact")
        family_sha256s = tuple(value.family_sha256 for value in self.families)
        if (
            not family_sha256s
            or family_sha256s != tuple(sorted(set(family_sha256s)))
            or self.family_sha256s != family_sha256s
        ):
            raise ValueError(
                "confirmation aggregate families must be SHA-sorted and unique"
            )
        if any(
            value.final_activation_authority.generated_registry
            != self.generated_registry
            or value.final_activation_authority.pilot_activation_authority.runtime
            != self.runtime
            or value.final_activation_authority.pilot_activation_authority.split
            != self.split
            or value.completion_authority.inventory_authority
            != self.inventory_authority
            for value in self.families
        ):
            raise ValueError(
                "confirmation aggregate family registry/runtime/split/inventory differs"
            )
        if self.auxiliary_completion_authority is not None and (
            self.auxiliary_completion_authority.activation.generated_registry
            != self.generated_registry
            or self.auxiliary_completion_authority.activation.runtime != self.runtime
            or self.auxiliary_completion_authority.activation.split != self.split
            or self.auxiliary_completion_authority.inventory_authority
            != self.inventory_authority
            or self.auxiliary_completion_authority.activation.experiment
            != self.experiment
        ):
            raise ValueError("confirmation aggregate auxiliary stage identity differs")
        if self.activated_cell_ids != tuple(sorted(set(self.activated_cell_ids))):
            raise ValueError(
                "confirmation aggregate activated cells must be sorted and unique"
            )
        for name in ("dispositions_sha256", "activation_sha256"):
            _require_sha256(f"confirmation aggregate {name}", getattr(self, name))
        _require_sha256(
            "confirmation aggregate stage receipt", self.stage_receipt_sha256
        )
        if self.stage_receipt_sha256 != self.stage_receipt.semantic_sha256:
            raise ValueError("confirmation aggregate stage receipt identity differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


type BudgetActivationAuthorityBinding = (
    RegistryStageActivationAuthorityBinding
    | E1ActivationAuthorityBinding
    | E2ActivationAuthorityBinding
    | ConfirmationAuxiliaryActivationAuthorityBinding
    | ConfirmationPilotActivationAuthorityBinding
    | ConfirmationFinalActivationAuthorityBinding
    | ConfirmationStageAggregateAuthorityBinding
)


def _is_budget_activation_authority_binding(value: object) -> bool:
    return type(value) in {
        RegistryStageActivationAuthorityBinding,
        E1ActivationAuthorityBinding,
        E2ActivationAuthorityBinding,
        ConfirmationAuxiliaryActivationAuthorityBinding,
        ConfirmationPilotActivationAuthorityBinding,
        ConfirmationFinalActivationAuthorityBinding,
        ConfirmationStageAggregateAuthorityBinding,
    }


def _budget_activation_raw_sources(
    binding: BudgetActivationAuthorityBinding,
) -> tuple[BudgetRawJsonBinding, ...]:
    """Return the de-duplicated path closure of one tagged activation binding."""

    collected: list[BudgetRawJsonBinding] = []

    def add(source: BudgetRawJsonBinding) -> None:
        if type(source) is not BudgetRawJsonBinding:
            raise TypeError("activation authority raw source must be exact")
        prior = next((row for row in collected if row.path == source.path), None)
        if prior is None:
            collected.append(source)
        elif prior != source:
            raise ValueError(
                "activation authority aliases one path under two identities"
            )

    def add_inventory(authority: DependencyGpuInventoryAuthorityBinding) -> None:
        add(authority.inventory)
        add(authority.source_receipt)

    def add_completion(
        authority: RegistryStageDependencyCompletionAuthorityBinding,
    ) -> None:
        add(authority.receipt)
        add(authority.completed_cells)
        add_inventory(authority.inventory_authority)
        for output in authority.locked_outputs:
            add(output.artifact)
        add_activation(authority.activation)

    def add_activation(authority: BudgetActivationAuthorityBinding) -> None:
        add(authority.manifest)
        add(authority.generated_registry)
        if type(authority) is RegistryStageActivationAuthorityBinding:
            add(authority.runtime)
            add(authority.split)
            for receipt in authority.dependency_receipts:
                add(receipt)
            for completion in authority.dependency_completion_authorities:
                add_completion(completion)
        elif type(authority) is E1ActivationAuthorityBinding:
            add(authority.runtime)
            add(authority.split)
            add(authority.dependency_receipt)
            add(authority.selection_manifest)
            add_inventory(authority.inventory_authority)
            add(authority.hardware_envelope)
            add_completion(authority.dependency_completion_authority)
        elif type(authority) is E2ActivationAuthorityBinding:
            add(authority.runtime)
            add(authority.split)
            add(authority.dependency_receipt)
            add(authority.pareto_manifest)
            for stage in authority.prior_stage_manifests:
                add(stage)
            for completion in authority.prior_stage_completion_authorities:
                add(completion.completed_cells)
                add_inventory(completion.inventory_authority)
                add_activation(completion.stage_activation)
            add_inventory(authority.inventory_authority)
            add(authority.hardware_envelope)
            add_completion(authority.dependency_completion_authority)
        elif (
            type(authority) is ConfirmationAuxiliaryActivationAuthorityBinding
            or type(authority) is ConfirmationPilotActivationAuthorityBinding
        ):
            add(authority.runtime)
            add(authority.split)
            add(authority.trace)
            add(authority.sampling)
            for receipt in authority.dependency_receipts:
                add(receipt)
            for completion in authority.dependency_completion_authorities:
                add_completion(completion)
            add_inventory(authority.inventory_authority)
            add(authority.hardware_envelope)
        elif type(authority) is ConfirmationFinalActivationAuthorityBinding:
            add(authority.power_manifest)
            add_activation(authority.pilot_activation_authority)
            add(authority.pilot_completion_authority.completed_cells)
            add_inventory(authority.pilot_completion_authority.inventory_authority)
        elif type(authority) is ConfirmationStageAggregateAuthorityBinding:
            add(authority.stage_receipt)
            add(authority.stage_completed_cells)
            add(authority.runtime)
            add(authority.split)
            add_inventory(authority.inventory_authority)
            for family in authority.families:
                add_activation(family.final_activation_authority)
                add(family.completion_authority.completed_cells)
            if authority.auxiliary_completion_authority is not None:
                add_activation(authority.auxiliary_completion_authority.activation)
                add(authority.auxiliary_completion_authority.completed_cells)
        else:  # pragma: no cover - guarded by exact tagged union
            raise TypeError("unsupported budget activation authority")

    add_activation(binding)
    return tuple(collected)


@dataclass(frozen=True)
class BudgetLoadRawBinding:
    """Cell-keyed raw load source; top-level tuples are cell-sorted."""

    cell_id: str
    source: BudgetRawJsonBinding

    def __post_init__(self) -> None:
        _require_sha256("budget raw load cell", self.cell_id)
        if (
            type(self.source) is not BudgetRawJsonBinding
            or self.source.role != "budget_load_binding"
        ):
            raise TypeError("budget raw load requires an exact load binding source")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class BudgetMaterializationAuthorityBinding:
    """Serializable pointer to every raw input of one exact BudgetPlan."""

    schema_version: int
    activation: BudgetActivationAuthorityBinding
    policy: BudgetRawJsonBinding
    load_bindings: tuple[BudgetLoadRawBinding, ...]
    capacity_envelope: BudgetRawJsonBinding
    capacity_authority: CapacityAuthorityBinding
    declared_plan: BudgetRawJsonBinding
    registry_sha256: str
    budget_inventory_sha256: str
    activation_sha256: str
    budget_policy_sha256: str
    budget_load_binding_sha256s: tuple[str, ...]
    capacity_envelope_sha256: str
    capacity_authority_sha256: str
    declared_plan_sha256: str
    authority_protocol_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only budget materialization authority schema 1 is supported"
            )
        if not _is_budget_activation_authority_binding(self.activation):
            raise TypeError(
                "budget materialization requires exact activation authority"
            )
        for value, role in (
            (self.policy, "budget_policy"),
            (self.capacity_envelope, "capacity_envelope"),
            (self.declared_plan, "declared_budget_plan"),
        ):
            if type(value) is not BudgetRawJsonBinding or value.role != role:
                raise TypeError(
                    f"budget materialization authority requires exact {role}"
                )
        if type(self.capacity_authority) is not CapacityAuthorityBinding:
            raise TypeError("budget materialization requires exact capacity authority")
        if any(type(value) is not BudgetLoadRawBinding for value in self.load_bindings):
            raise TypeError("budget materialization loads require exact raw bindings")
        cell_ids = tuple(value.cell_id for value in self.load_bindings)
        if cell_ids != tuple(sorted(set(cell_ids))):
            raise ValueError(
                "budget materialization load bindings must be cell-sorted and unique"
            )
        for name in (
            "registry_sha256",
            "budget_inventory_sha256",
            "activation_sha256",
            "budget_policy_sha256",
            "capacity_envelope_sha256",
            "capacity_authority_sha256",
            "declared_plan_sha256",
        ):
            _require_sha256(
                f"budget materialization authority {name}", getattr(self, name)
            )
        if self.budget_load_binding_sha256s != tuple(
            value.source.semantic_sha256 for value in self.load_bindings
        ):
            raise ValueError(
                "budget materialization load identities differ from sources"
            )
        if (
            self.registry_sha256 != self.activation.generated_registry.semantic_sha256
            or self.activation_sha256 != self.activation.activation_sha256
            or self.budget_policy_sha256 != self.policy.semantic_sha256
            or self.capacity_envelope_sha256 != self.capacity_envelope.semantic_sha256
            or self.capacity_authority_sha256 != self.capacity_authority.sha256
            or self.declared_plan_sha256 != self.declared_plan.semantic_sha256
            or self.registry_sha256 != self.capacity_authority.registry_sha256
            or self.budget_inventory_sha256
            != self.capacity_authority.budget_inventory_sha256
            or self.capacity_envelope_sha256
            != self.capacity_authority.capacity_envelope_sha256
        ):
            raise ValueError("budget materialization redundant identities differ")
        budget_sources = (
            self.policy,
            self.capacity_envelope,
            self.declared_plan,
            *(value.source for value in self.load_bindings),
        )
        activation_sources = _budget_activation_raw_sources(self.activation)
        capacity_sources = (
            self.capacity_authority.source_manifest,
            self.capacity_authority.verification_receipt,
        )
        budget_paths = {
            path
            for source in budget_sources
            for path in (source.path, source.sidecar_path)
        }
        activation_paths = {
            path
            for source in activation_sources
            for path in (source.path, source.sidecar_path)
        }
        capacity_paths = {
            path
            for source in capacity_sources
            for path in (source.path, source.sidecar_path)
        }
        if (
            len(budget_paths) != 2 * len(budget_sources)
            or len(activation_paths) != 2 * len(activation_sources)
            or len(capacity_paths) != 2 * len(capacity_sources)
            or budget_paths & activation_paths
            or budget_paths & capacity_paths
            or activation_paths & capacity_paths
        ):
            raise ValueError("budget materialization raw source paths alias")
        if (
            self.authority_protocol_sha256
            != BUDGET_MATERIALIZATION_AUTHORITY_PROTOCOL_SHA256
        ):
            raise ValueError("budget materialization authority uses another protocol")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class BudgetJobPolicy:
    """Explicit non-load timing policy for one registered job kind.

    A policy row exists for every :class:`BudgetJobKind`; sharing an implicit
    duration default between job kinds is therefore impossible.  Load-window
    durations and request counts are deliberately absent from this type and
    must be derived from :class:`ProductionLoadPlan` objects.
    """

    job_kind: BudgetJobKind
    startup_model_load: ScenarioMilliseconds
    compile_jit_graph_prewarm: ScenarioMilliseconds
    reset_finalization: ScenarioMilliseconds
    evidence_flush_shutdown: ScenarioMilliseconds
    retry: ScenarioMilliseconds
    retry_allowance: int
    download_compile_reservation: ScenarioMilliseconds
    reserved_gpu_overhead: ScenarioMilliseconds

    def __post_init__(self) -> None:
        if not isinstance(self.job_kind, BudgetJobKind):
            raise TypeError("budget job policy kind must be a BudgetJobKind")
        for name in (
            "startup_model_load",
            "compile_jit_graph_prewarm",
            "reset_finalization",
            "evidence_flush_shutdown",
            "retry",
            "download_compile_reservation",
            "reserved_gpu_overhead",
        ):
            if not isinstance(getattr(self, name), ScenarioMilliseconds):
                raise TypeError(
                    f"budget job policy {name} must be scenario milliseconds"
                )
        _require_nonnegative_int("budget job retry allowance", self.retry_allowance)
        if self.retry_allowance == 0 and self.retry != ZERO_MILLISECONDS:
            raise ValueError(
                "budget job retry duration and allowance must be explicitly aligned"
            )
        if self.retry_allowance > 0 and self.retry.registered <= 0:
            raise ValueError(
                "budget job retry duration and allowance must be explicitly aligned"
            )
        if (
            self.job_kind is BudgetJobKind.COMPILE
            and self.compile_jit_graph_prewarm.registered <= 0
        ):
            raise ValueError("compile policy requires an explicit compile duration")
        if (
            self.job_kind is not BudgetJobKind.COMPILE
            and self.compile_jit_graph_prewarm != ZERO_MILLISECONDS
        ):
            raise ValueError(
                "compile/prewarm duration is valid only for the compile policy"
            )
        if (
            self.job_kind is BudgetJobKind.DOWNLOAD
            and self.download_compile_reservation.registered <= 0
        ):
            raise ValueError(
                "download policy requires an explicit reservation duration"
            )
        if (
            self.job_kind is not BudgetJobKind.DOWNLOAD
            and self.download_compile_reservation != ZERO_MILLISECONDS
        ):
            raise ValueError(
                "download reservation is valid only for the explicit download policy"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class BudgetPolicy:
    """Reducer-owned, complete timing policy for every industrial job kind."""

    schema_version: int
    policy_name: str
    reducer_protocol_sha256: str
    job_policies: tuple[BudgetJobPolicy, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only BudgetPolicy schema version 1 is supported")
        _require_text("budget policy name", self.policy_name)
        if self.reducer_protocol_sha256 != BUDGET_MATERIALIZATION_PROTOCOL_SHA256:
            raise ValueError("budget policy uses an unregistered reducer protocol")
        if any(not isinstance(row, BudgetJobPolicy) for row in self.job_policies):
            raise TypeError("budget policy rows must be BudgetJobPolicy values")
        kinds = tuple(row.job_kind for row in self.job_policies)
        expected = tuple(sorted(BudgetJobKind, key=lambda value: value.value))
        if kinds != expected:
            raise ValueError(
                "budget policy must cover every job kind once in canonical order"
            )

    def for_job(self, job_kind: BudgetJobKind) -> BudgetJobPolicy:
        matches = tuple(row for row in self.job_policies if row.job_kind is job_kind)
        if len(matches) != 1:
            raise ValueError("budget policy does not resolve one exact job kind")
        return matches[0]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class BudgetLoadBinding:
    """Three exact load-plan scenarios for one activated serving cell.

    The materializer consumes these real ``ProductionLoadPlan`` objects and
    derives every load duration, request count, and output-token count.  It
    never accepts a per-cell duration dictionary.
    """

    cell_id: str
    job_kind: BudgetJobKind
    optimistic_load: ProductionLoadPlan
    registered_load: ProductionLoadPlan
    quota_envelope_load: ProductionLoadPlan
    minimum_completed_requests: int
    p99_anchor_status: P99AnchorStatus

    def __post_init__(self) -> None:
        _require_sha256("budget load cell_id", self.cell_id)
        if not isinstance(self.job_kind, BudgetJobKind):
            raise TypeError("budget load job kind must be a BudgetJobKind")
        if self.job_kind in {BudgetJobKind.COMPILE, BudgetJobKind.DOWNLOAD}:
            raise ValueError("compile/download jobs do not consume serving load plans")
        if not isinstance(self.p99_anchor_status, P99AnchorStatus):
            raise TypeError("budget load p99 status must be a P99AnchorStatus")
        _require_nonnegative_int(
            "budget load minimum completions", self.minimum_completed_requests
        )
        plans = self.load_plans
        if any(type(plan) is not ProductionLoadPlan for plan in plans):
            raise TypeError(
                "budget load scenarios must be exact ProductionLoadPlan values"
            )
        for plan in plans:
            plan.validate()
        scored_hashes = tuple(plan.scored.hashes for plan in plans)
        warmup_hashes = tuple(
            None if plan.warmup is None else plan.warmup.hashes for plan in plans
        )
        if len(set(scored_hashes)) != 1 or len(set(warmup_hashes)) != 1:
            raise ValueError(
                "budget load scenarios must retain identical request/corpus semantics"
            )
        for name in (
            "warmup_duration_us",
            "arrival_duration_us",
            "request_deadline_us",
            "drain_duration_us",
        ):
            values = tuple(getattr(plan.window, name) for plan in plans)
            if values != tuple(sorted(values)):
                raise ValueError("budget load scenario windows must be monotone")
        if self.job_kind is BudgetJobKind.P99_ANCHOR:
            if self.p99_anchor_status is P99AnchorStatus.NOT_REQUIRED:
                raise ValueError("p99 load binding requires an explicit anchor status")
            if self.minimum_completed_requests < P99_MINIMUM_COMPLETIONS:
                raise ValueError(
                    "p99 load binding requires at least 10,000 completions"
                )
        elif self.p99_anchor_status is not P99AnchorStatus.NOT_REQUIRED:
            raise ValueError("p99 status is valid only for a p99-anchor load binding")

    @property
    def load_plans(self) -> tuple[ProductionLoadPlan, ...]:
        return (
            self.optimistic_load,
            self.registered_load,
            self.quota_envelope_load,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "cell_id": self.cell_id,
                "job_kind": self.job_kind,
                "load_plan_sha256s": tuple(
                    plan.paired_replay_sha256 for plan in self.load_plans
                ),
                "minimum_completed_requests": self.minimum_completed_requests,
                "p99_anchor_status": self.p99_anchor_status,
            }
        )


@dataclass(frozen=True)
class CellCapacityRequirement:
    """Maximum durable bytes retained by one attempt of one activated cell."""

    cell_id: str
    maximum_evidence_bytes: int
    model_staging_bytes: int
    compile_overlay_bytes: int

    def __post_init__(self) -> None:
        _require_sha256("capacity requirement cell_id", self.cell_id)
        for name in (
            "maximum_evidence_bytes",
            "model_staging_bytes",
            "compile_overlay_bytes",
        ):
            _require_nonnegative_int(
                f"capacity requirement {name}", getattr(self, name)
            )

    @property
    def maximum_attempt_bytes(self) -> int:
        return (
            self.maximum_evidence_bytes
            + self.model_staging_bytes
            + self.compile_overlay_bytes
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class CapacityEnvelope:
    """Diagnostic provider/host declaration for one budget inventory.

    The values become execution authority only when a
    :class:`CapacityAuthorityBinding` reopens their raw provider, host,
    inventory, and per-cell sizing provenance under the source-owned release
    verifier.  ``source_receipt_sha256`` alone is never authority.
    """

    schema_version: int
    budget_inventory_sha256: str
    provider_quota_gpu_ms: int
    host_free_bytes: int
    host_quota_bytes: int
    cell_requirements: tuple[CellCapacityRequirement, ...]
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only CapacityEnvelope schema version 1 is supported")
        _require_sha256("capacity budget inventory", self.budget_inventory_sha256)
        _require_sha256("capacity source receipt", self.source_receipt_sha256)
        for name in (
            "provider_quota_gpu_ms",
            "host_free_bytes",
            "host_quota_bytes",
        ):
            _require_nonnegative_int(f"capacity {name}", getattr(self, name))
        if any(
            type(row) is not CellCapacityRequirement for row in self.cell_requirements
        ):
            raise TypeError(
                "capacity rows must be exact CellCapacityRequirement values"
            )
        cell_ids = tuple(row.cell_id for row in self.cell_requirements)
        if cell_ids != tuple(sorted(set(cell_ids))):
            raise ValueError("capacity requirements must be cell-sorted and unique")

    @property
    def effective_host_bytes(self) -> int:
        return min(self.host_free_bytes, self.host_quota_bytes)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def _ready_capacity_rejection_reason(
    envelope: CapacityEnvelope,
    budgets: Sequence[ExperimentBudget],
) -> str | None:
    requirement_by_cell = {row.cell_id: row for row in envelope.cell_requirements}
    budget_by_cell = {budget.cell_id: budget for budget in budgets}
    if set(requirement_by_cell) != set(budget_by_cell):
        return "capacity_requirement_coverage_incomplete"
    provider_gpu_ms = sum(
        budget.fixed_instance_billed_gpu_ms.quota_envelope
        * (budget.retry_allowance + 1)
        for budget in budget_by_cell.values()
    )
    if provider_gpu_ms > envelope.provider_quota_gpu_ms:
        return "capacity_provider_quota_exceeded"
    durable_bytes = sum(
        requirement_by_cell[cell_id].maximum_attempt_bytes
        * (budget.retry_allowance + 1)
        for cell_id, budget in budget_by_cell.items()
    )
    if durable_bytes > envelope.effective_host_bytes:
        return "capacity_host_disk_exceeded"
    return None


class BudgetDispositionStatus(str, Enum):
    BUDGETED = "BUDGETED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class BudgetDisposition:
    cell_id: str
    status: BudgetDispositionStatus
    reason_code: str
    source_semantics_sha256: str
    experiment_budget_sha256: str | None

    def __post_init__(self) -> None:
        _require_sha256("budget disposition cell_id", self.cell_id)
        _require_sha256(
            "budget disposition source semantics", self.source_semantics_sha256
        )
        if not isinstance(self.status, BudgetDispositionStatus):
            raise TypeError("budget disposition status is invalid")
        _require_text("budget disposition reason", self.reason_code)
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in self.reason_code
        ):
            raise ValueError("budget disposition reason code is invalid")
        if self.status is BudgetDispositionStatus.BUDGETED:
            _require_sha256(
                "budget disposition ExperimentBudget", self.experiment_budget_sha256
            )
            if self.reason_code != "first_party_budget_materialized":
                raise ValueError("budgeted disposition has an unexpected reason")
        elif self.experiment_budget_sha256 is not None:
            raise ValueError("unresolved budget disposition cannot bind a budget")


def _budget_activation_sha256(
    reducer_activation_sha256s: tuple[str, ...],
    family_activation_sha256s: tuple[str, ...],
    family_power_reduction_sha256s: tuple[str, ...],
) -> str:
    return content_sha256(
        {
            "reducer_activation_sha256s": reducer_activation_sha256s,
            "family_activation_sha256s": family_activation_sha256s,
            "family_power_reduction_sha256s": family_power_reduction_sha256s,
        }
    )


@dataclass(frozen=True)
class BudgetPlan:
    """Canonical reducer output for one exact activated set and inventory."""

    schema_version: int
    registry_sha256: str
    activation_sha256: str
    reducer_activation_sha256s: tuple[str, ...]
    family_activation_sha256s: tuple[str, ...]
    family_power_reduction_sha256s: tuple[str, ...]
    policy: BudgetPolicy
    inventory: BudgetInventoryIdentity
    capacity_envelope: CapacityEnvelope | None
    capacity_authority: CapacityAuthorityBinding | None
    activated_cell_ids: tuple[str, ...]
    budgets: tuple[ExperimentBudget, ...]
    dispositions: tuple[BudgetDisposition, ...]
    status: Literal["READY", "UNRESOLVED"]
    reducer_protocol_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only BudgetPlan schema version 2 is supported")
        _require_sha256("budget plan registry", self.registry_sha256)
        if self.reducer_protocol_sha256 != BUDGET_MATERIALIZATION_PROTOCOL_SHA256:
            raise ValueError("budget plan uses an unregistered reducer protocol")
        if not isinstance(self.policy, BudgetPolicy):
            raise TypeError("budget plan policy must be a BudgetPolicy")
        if not isinstance(self.inventory, BudgetInventoryIdentity):
            raise TypeError("budget plan inventory must be a BudgetInventoryIdentity")
        if self.capacity_envelope is not None:
            if type(self.capacity_envelope) is not CapacityEnvelope:
                raise TypeError(
                    "budget plan capacity must be an exact CapacityEnvelope"
                )
            if self.capacity_envelope.budget_inventory_sha256 != self.inventory.sha256:
                raise ValueError(
                    "budget plan capacity belongs to another budget inventory"
                )
        if self.capacity_authority is not None:
            if type(self.capacity_authority) is not CapacityAuthorityBinding:
                raise TypeError(
                    "budget plan capacity authority must be an exact binding"
                )
            if self.capacity_envelope is None:
                raise ValueError("capacity authority requires a capacity envelope")
            if (
                self.capacity_authority.registry_sha256 != self.registry_sha256
                or self.capacity_authority.budget_inventory_sha256
                != self.inventory.sha256
                or self.capacity_authority.capacity_envelope_sha256
                != self.capacity_envelope.sha256
            ):
                raise ValueError(
                    "budget plan capacity authority belongs to another exact plan"
                )
        for name in (
            "reducer_activation_sha256s",
            "family_activation_sha256s",
            "family_power_reduction_sha256s",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"budget plan {name} must be sorted and unique")
            for value in values:
                _require_sha256(f"budget plan {name}", value)
        if not (self.reducer_activation_sha256s or self.family_activation_sha256s):
            raise ValueError("budget plan requires reducer-owned activation authority")
        expected_activation = _budget_activation_sha256(
            self.reducer_activation_sha256s,
            self.family_activation_sha256s,
            self.family_power_reduction_sha256s,
        )
        if self.activation_sha256 != expected_activation:
            raise ValueError("budget plan activation bundle identity mismatch")
        if not self.activated_cell_ids:
            raise ValueError("budget plan requires at least one activated cell")
        if self.activated_cell_ids != tuple(sorted(set(self.activated_cell_ids))):
            raise ValueError("budget plan activated cells must be sorted and unique")
        for cell_id in self.activated_cell_ids:
            _require_sha256("budget plan activated cell", cell_id)
        if any(type(row) is not ExperimentBudget for row in self.budgets):
            raise TypeError("budget plan budgets must be exact ExperimentBudget values")
        if any(type(row) is not BudgetDisposition for row in self.dispositions):
            raise TypeError(
                "budget plan dispositions must be exact BudgetDisposition values"
            )
        if tuple(row.cell_id for row in self.budgets) != tuple(
            sorted(row.cell_id for row in self.budgets)
        ) or len({row.cell_id for row in self.budgets}) != len(self.budgets):
            raise ValueError("budget plan budgets must be cell-sorted and unique")
        if {row.cell_id for row in self.budgets} - set(self.activated_cell_ids):
            raise ValueError("budget plan names a non-activated diagnostic budget")
        if tuple(row.cell_id for row in self.dispositions) != self.activated_cell_ids:
            raise ValueError("budget dispositions must exactly cover activated cells")
        if len({row.cell_id for row in self.dispositions}) != len(self.dispositions):
            raise ValueError("budget dispositions contain duplicate cells")
        if self.capacity_envelope is not None and (
            {row.cell_id for row in self.capacity_envelope.cell_requirements}
            - set(self.activated_cell_ids)
        ):
            raise ValueError("budget plan capacity names a non-activated cell")
        budget_by_cell = {budget.cell_id: budget for budget in self.budgets}
        budgeted = {
            row.cell_id: row
            for row in self.dispositions
            if row.status is BudgetDispositionStatus.BUDGETED
        }
        if self.status == "READY":
            if set(budget_by_cell) != set(budgeted) or any(
                budgeted[cell_id].experiment_budget_sha256 != budget.sha256
                for cell_id, budget in budget_by_cell.items()
            ):
                raise ValueError(
                    "ready budget dispositions differ from materialized budgets"
                )
        elif budgeted:
            raise ValueError("unresolved budget plan cannot claim budgeted cells")
        if any(
            budget.fixed_instance_billed_gpu_ms
            != budget.wall_time.scale(self.inventory.gpu_count)
            for budget in self.budgets
        ):
            raise ValueError("budget plan billing differs from its exact inventory")
        expected_status = (
            "UNRESOLVED"
            if any(
                row.status is BudgetDispositionStatus.UNRESOLVED
                for row in self.dispositions
            )
            else "READY"
        )
        if self.status != expected_status:
            raise ValueError("budget plan status differs from its dispositions")
        if self.status == "READY":
            if len(self.budgets) != len(self.activated_cell_ids):
                raise ValueError("ready budget plan lacks complete budget coverage")
            if self.capacity_envelope is None:
                raise ValueError("ready budget plan lacks capacity authority")
            if self.capacity_authority is None:
                raise ValueError("ready budget plan lacks raw capacity authority")
            capacity_rejection = _ready_capacity_rejection_reason(
                self.capacity_envelope, self.budgets
            )
            if capacity_rejection is not None:
                raise ValueError(
                    "ready budget plan exceeds or lacks its capacity authority: "
                    f"{capacity_rejection}"
                )
            self._revalidate_capacity_authority()

    def _revalidate_capacity_authority(self) -> None:
        if self.capacity_envelope is None or self.capacity_authority is None:
            raise ValueError("budget plan lacks raw capacity authority")
        from lightcone_spec.experiments.capacity_authority import (
            revalidate_capacity_authority_binding,
        )

        revalidate_capacity_authority_binding(
            self.capacity_authority,
            expected_registry_sha256=self.registry_sha256,
            expected_inventory=self.inventory,
            expected_envelope=self.capacity_envelope,
        )

    def require_ready(self) -> tuple[ExperimentBudget, ...]:
        if self.status != "READY":
            unresolved = tuple(
                (row.cell_id, row.reason_code)
                for row in self.dispositions
                if row.status is BudgetDispositionStatus.UNRESOLVED
            )
            raise ValueError(f"budget plan contains unresolved cells: {unresolved}")
        self._revalidate_capacity_authority()
        return self.budgets

    @property
    def diagnostic_budgets(self) -> tuple[ExperimentBudget, ...]:
        """Return exact arithmetic for read-only reporting, never execution.

        An unresolved plan may retain these rows so blocked quota, disk, or
        trust assumptions do not erase the truthful GPU-hour estimate.  Every
        scheduler/executor boundary must call :meth:`require_ready` instead.
        """

        return self.budgets

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
    scheduler_gpu_inventory_sha256: str | None
    interference_envelope_sha256: str | None

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
        schedule_authority = (
            self.scheduler_gpu_inventory_sha256,
            self.interference_envelope_sha256,
        )
        if (schedule_authority[0] is None) != (schedule_authority[1] is None):
            raise ValueError(
                "scheduler inventory and interference identities must be paired"
            )
        for name, value in zip(
            (
                "scheduler GPU inventory",
                "interference envelope",
            ),
            schedule_authority,
            strict=True,
        ):
            if value is not None:
                _require_sha256(name, value)
        if self.estimated_wall_ms is not None and schedule_authority[0] is None:
            raise ValueError("an exact wall estimate requires scheduler authority")

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


def budget_inventory_identity_from_gpu_inventory(
    gpu_inventory: GpuInventory,
) -> BudgetInventoryIdentity:
    """Project a complete physical inventory onto the budget identity contract."""

    from lightcone_spec.experiments.gpu_pool import GpuInventory

    if type(gpu_inventory) is not GpuInventory:
        raise TypeError("budget inventory projection requires an exact GpuInventory")
    return BudgetInventoryIdentity(
        schema_version=1,
        host_sha256=content_sha256(list(gpu_inventory.host_ids)),
        gpu_uuids=tuple(device.uuid for device in gpu_inventory.devices),
        topology_sha256=content_sha256(
            [group.to_dict() for group in gpu_inventory.topology_groups]
        ),
    )


def _estimate_wall_time_from_scheduler(
    registry: ExperimentRegistry,
    *,
    activation_sha256: str,
    budgets: Sequence[ExperimentBudget],
    gpu_inventory: GpuInventory,
    interference_envelope: InterferenceEnvelope,
) -> tuple[ScenarioMilliseconds | None, str | None]:
    """Replay the production scheduler and account its exact frozen waves."""

    from lightcone_spec.experiments.gpu_pool import (
        CapabilityRejectionError,
        GpuInventory,
        GpuPoolScheduler,
        InterferenceEnvelope,
        registry_pool_work_item,
    )

    if type(gpu_inventory) is not GpuInventory:
        raise TypeError("exact budget estimation requires an exact GpuInventory")
    if type(interference_envelope) is not InterferenceEnvelope:
        raise TypeError(
            "exact budget estimation requires an exact InterferenceEnvelope"
        )
    ordered = tuple(sorted(budgets, key=lambda row: row.cell_id))
    if any(budget.wall_time.optimistic <= 0 for budget in ordered):
        return None, "nonpositive_budget_duration"
    if not ordered:
        return ZERO_MILLISECONDS, None
    if len(gpu_inventory.host_ids) != 1:
        return None, "multi_host_inventory_unsupported"
    cell_by_id = {cell.cell_id: cell for cell in registry.cells}
    if any(
        not GpuPoolScheduler._dispatchable(cell_by_id[budget.cell_id])
        for budget in ordered
    ):
        return None, "registry_cell_not_dispatchable"
    scheduler = GpuPoolScheduler(
        registry=registry,
        inventory=gpu_inventory,
        interference_envelope=interference_envelope,
    )
    budget_by_cell = {budget.cell_id: budget for budget in ordered}
    budget_sha256_by_cell = {
        cell_id: budget.sha256 for cell_id, budget in budget_by_cell.items()
    }
    values: list[int] = []
    for scenario in BudgetScenario:
        items = tuple(
            registry_pool_work_item(
                cell_by_id[budget.cell_id],
                estimated_duration_seconds=budget.wall_time.value(scenario) / 1_000.0,
            )
            for budget in ordered
        )
        try:
            dispatch = scheduler.schedule_work_items(
                items,
                receipts_sha256=content_sha256(
                    {
                        "schema_version": 1,
                        "kind": "industrial_budget_exact_schedule",
                        "activation_sha256": activation_sha256,
                        "scenario": scenario,
                        "inventory_sha256": gpu_inventory.sha256,
                        "interference_envelope_sha256": interference_envelope.sha256,
                    }
                ),
                budget_sha256_by_cell=budget_sha256_by_cell,
            )
        except CapabilityRejectionError:
            return None, "capability_or_topology_rejected"
        values.append(
            sum(
                max(
                    budget_by_cell[assignment.work_item.item_id].wall_time.value(
                        scenario
                    )
                    for assignment in wave.assignments
                )
                for wave in dispatch.waves
            )
        )
    return ScenarioMilliseconds(*values), None


def estimate_industrial_budget(
    registry: ExperimentRegistry,
    *,
    activated_cell_ids: Sequence[str],
    activation_sha256: str,
    budgets: Sequence[ExperimentBudget],
    inventory: BudgetInventoryIdentity,
    gpu_inventory: GpuInventory | None = None,
    interference_envelope: InterferenceEnvelope | None = None,
    unresolved_assumptions: Sequence[str] = (),
) -> IndustrialBudgetReport:
    """Reduce exact budgets and, when authoritative inputs exist, exact waves."""

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
    unresolved_values = set(unresolved_assumptions)
    if any(
        type(reason) is not str
        or not reason.strip()
        or "\n" in reason
        or "\r" in reason
        for reason in unresolved_values
    ):
        raise ValueError("budget report unresolved assumptions are invalid")
    unresolved_values.update(
        {
            f"cell_requires_{budget.gpu_count}_gpus_but_inventory_has_"
            f"{inventory.gpu_count}:{budget.cell_id}"
            for budget in rows
            if budget.gpu_count > inventory.gpu_count
        }
    )
    unresolved_values.update(
        f"p99_anchor_unresolved:{budget.cell_id}"
        for budget in rows
        if budget.p99_anchor_status is P99AnchorStatus.REQUIRED_UNRESOLVED
    )
    resources_fit = all(budget.gpu_count <= inventory.gpu_count for budget in rows)
    wall: ScenarioMilliseconds | None = None
    scheduler_inventory_sha256: str | None = None
    interference_envelope_sha256: str | None = None
    if (gpu_inventory is None) != (interference_envelope is None):
        raise ValueError(
            "physical GPU inventory and interference envelope must be supplied together"
        )
    if gpu_inventory is not None and interference_envelope is not None:
        from lightcone_spec.experiments.gpu_pool import InterferenceEnvelope

        if type(interference_envelope) is not InterferenceEnvelope:
            raise TypeError(
                "exact budget estimation requires an exact InterferenceEnvelope"
            )
        projected_inventory = budget_inventory_identity_from_gpu_inventory(
            gpu_inventory
        )
        if projected_inventory != inventory:
            raise ValueError(
                "physical GPU inventory differs from the budget inventory identity"
            )
        scheduler_inventory_sha256 = gpu_inventory.sha256
        interference_envelope_sha256 = interference_envelope.sha256
    if resources_fit:
        if gpu_inventory is None or interference_envelope is None:
            unresolved_values.add(
                "exact_inventory_schedule_unresolved:"
                "full_inventory_and_interference_required"
            )
        else:
            wall, schedule_rejection = _estimate_wall_time_from_scheduler(
                registry,
                activation_sha256=activation_sha256,
                budgets=rows,
                gpu_inventory=gpu_inventory,
                interference_envelope=interference_envelope,
            )
            if schedule_rejection is not None:
                unresolved_values.add(
                    f"exact_inventory_schedule_unresolved:{schedule_rejection}"
                )
    unresolved = tuple(sorted(unresolved_values))
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
        scheduler_gpu_inventory_sha256=scheduler_inventory_sha256,
        interference_envelope_sha256=interference_envelope_sha256,
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


def materialize_confirmation_auxiliary_activation(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    dependency_receipts: Sequence[ExperimentReceipt],
    runtime_sha256: str,
    split_sha256: str,
    trace_sha256: str,
    sampling_sha256: str,
    hardware_envelope_sha256: str,
) -> ReducerActivationArtifact:
    """Activate every and only registered cells outside complete core families."""

    if experiment not in {"E3b", "E5"}:
        raise ValueError("confirmation auxiliary activation supports only E3b/E5")
    stage_index = INDUSTRIAL_EXPERIMENT_ORDER.index(experiment)
    receipts = tuple(dependency_receipts)
    if (
        tuple(receipt.experiment for receipt in receipts)
        != (INDUSTRIAL_EXPERIMENT_ORDER[:stage_index])
    ):
        raise ValueError(
            "confirmation auxiliary activation lacks exact dependency prefix"
        )
    registry.validate_receipts(receipts)
    if not receipts:
        raise ValueError("confirmation auxiliary activation requires dependencies")
    _, auxiliary = derive_confirmation_stage_partition(
        registry,
        experiment=experiment,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        trace_sha256=trace_sha256,
        sampling_sha256=sampling_sha256,
        hardware_envelope_sha256=hardware_envelope_sha256,
    )
    if not auxiliary:
        raise ValueError("confirmation stage has no auxiliary registry cells")
    rows = tuple(
        CellDisposition(
            cell_id=cell.cell_id,
            status=(
                DispositionStatus.ACTIVATED
                if cell.runnable
                and cell.resources.workload_class
                not in {WorkloadClass.COMPILE, WorkloadClass.DOWNLOAD}
                else DispositionStatus.NOT_APPLICABLE
                if cell.status is CellStatus.NOT_APPLICABLE
                else DispositionStatus.BLOCKED
            ),
            reason_code=(
                "confirmation_auxiliary_registry_cell"
                if cell.runnable
                and cell.resources.workload_class
                not in {WorkloadClass.COMPILE, WorkloadClass.DOWNLOAD}
                else "confirmation_auxiliary_non_serving_contract_unavailable"
                if cell.runnable
                else cell.reason_code
            ),
        )
        for cell in auxiliary
    )
    source_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "confirmation_auxiliary_registry_activation_source",
            "registry_sha256": registry.sha256,
            "experiment": experiment,
            "dependency_receipt_sha256s": tuple(receipt.sha256 for receipt in receipts),
            "runtime_sha256": runtime_sha256,
            "split_sha256": split_sha256,
            "trace_sha256": trace_sha256,
            "sampling_sha256": sampling_sha256,
            "hardware_envelope_sha256": hardware_envelope_sha256,
            "auxiliary_cell_ids": tuple(cell.cell_id for cell in auxiliary),
            "reducer_protocol_sha256": (
                CONFIRMATION_AUXILIARY_ACTIVATION_PROTOCOL_SHA256
            ),
        }
    )
    return _make_activation(
        registry=registry,
        experiment=experiment,
        dependency_receipt=receipts[-1],
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        source_selection_sha256=source_sha256,
        activation_round="confirmation_auxiliary_registry_v1",
        rows=rows,
        reason_code="confirmation_auxiliary_registry_activation",
        reducer_protocol_sha256=CONFIRMATION_AUXILIARY_ACTIVATION_PROTOCOL_SHA256,
    )


def reduce_e1_activation(
    registry: ExperimentRegistry,
    *,
    e3a_receipt: ExperimentReceipt,
    selection: SealedE3aSelection,
) -> ReducerActivationArtifact:
    """Materialize the one sealed 68-cell E1 width/load slice."""

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
    if len(cells) != 1428 or len(selected) != _E1_SLICE_CELLS:
        raise ValueError("registered E1 envelope must reduce from 1,428 to 68 cells")
    selected_cells = tuple(cell for cell in cells if cell.cell_id in selected)
    role_counts = {
        role: sum(
            scientific_role_for_cell(registry, cell) == role for cell in selected_cells
        )
        for role in ("target_only", "static", "tts", "l0_naive", "lc_candidate")
    }
    if role_counts != {
        "target_only": 1,
        "static": 1,
        "tts": 1,
        "l0_naive": 1,
        "lc_candidate": 64,
    }:
        raise ValueError(
            "E1 slice must retain two LC candidates per search geometry and one "
            "frozen TTS/L0-naive reference pair"
        )
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
            "declared_cells": 1428,
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
    def from_cell(
        cls, cell: ExperimentCell, *, registry: ExperimentRegistry
    ) -> E1GeometryIdentity:
        identity = cell.identity
        if scientific_role_for_cell(registry, cell) != "lc_candidate":
            raise ValueError("E1 geometry can be derived only from LC-candidates")
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
    surviving_geometries: tuple[E1GeometryIdentity, ...]
    selection_state: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("only E1 Pareto schema version 2 is supported")
        for name in (
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "e1_activation_sha256",
            "reducer_evidence_sha256",
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
    width: int | None
    draft_width_selector: str | None = None

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
        exact_width = self.width is not None
        selected_width = self.draft_width_selector is not None
        if exact_width == selected_width:
            raise ValueError("E2 candidate requires exactly one width authority")
        if exact_width:
            if self.width is None or self.width < 1:
                raise ValueError("E2 width must be positive")
        elif self.draft_width_selector != E2_DRAFT_WIDTH_SELECTOR:
            raise ValueError("E2 template requires the sealed E3a width selector")

    @classmethod
    def from_cell(
        cls, cell: ExperimentCell, *, registry: ExperimentRegistry
    ) -> E2CandidateIdentity:
        identity = cell.identity
        if (
            identity.experiment != "E2"
            or identity.method != "l0"
            or scientific_role_for_cell(registry, cell) != "lc_candidate"
        ):
            raise ValueError(
                "E2 candidates can be derived only from E2 LC-candidate cells"
            )
        if (
            identity.scope is None
            or identity.optimizer is None
            or identity.learning_rate is None
            or identity.schedule is None
        ):
            raise ValueError("E2 candidate identity contains unresolved fields")
        if identity.width is not None:
            raise ValueError("E2 registry candidate must remain a width template")
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
            width=None,
            draft_width_selector=E2_DRAFT_WIDTH_SELECTOR,
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


def _e2_candidates(
    registry: ExperimentRegistry,
    cells: Sequence[ExperimentCell],
) -> dict[str, tuple[E2CandidateIdentity, ExperimentCell]]:
    result: dict[str, tuple[E2CandidateIdentity, ExperimentCell]] = {}
    for cell in cells:
        if scientific_role_for_cell(registry, cell) != "lc_candidate":
            continue
        identity = cell.identity
        if (
            identity.scope is None
            or identity.optimizer is None
            or identity.learning_rate is None
            or identity.schedule is None
        ):
            if cell.runnable:
                raise ValueError("runnable E2 LC-candidate has unresolved identity")
            continue
        candidate = E2CandidateIdentity.from_cell(cell, registry=registry)
        if candidate.sha256 in result:
            raise ValueError("E2 repeats one LC-candidate identity")
        result[candidate.sha256] = (candidate, cell)
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
        _require_e2_promotion_authority(
            stage_index=self.stage_index,
            status=self.status,
            reason_code=self.reason_code,
        )
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
    if type(e1_receipt) is not ExperimentReceipt:
        raise TypeError("E2 activation requires an exact E1 receipt")
    _validate_direct_receipt(registry, e1_receipt, "E1")
    if type(pareto) is not E1ParetoArtifact:
        raise TypeError("E2 activation requires an exact E1 Pareto artifact")
    pareto = replace(pareto)
    if (
        pareto.registry_sha256 != registry.sha256
        or pareto.runtime_sha256 != e1_receipt.runtime_sha256
        or pareto.split_sha256 != e1_receipt.split_sha256
    ):
        raise ValueError("E2 activation identity does not match the E1 receipt")
    outputs = _receipt_outputs(e1_receipt)
    if outputs.get("dflash_pareto_set") != pareto.sha256:
        raise ValueError("E1 receipt does not bind the supplied Pareto artifact")
    if stage_index == 0:
        if prior_reduction is not None:
            raise ValueError("E2 stage zero cannot consume a prior reduction")
        source_candidate_ids: set[str] | None = None
        prior_completed: set[str] = set()
        source_selection_sha256 = pareto.sha256
    else:
        if (
            type(prior_reduction) is not E2StageReductionArtifact
            or prior_reduction.registry_sha256 != registry.sha256
            or prior_reduction.runtime_sha256 != pareto.runtime_sha256
            or prior_reduction.split_sha256 != pareto.split_sha256
            or prior_reduction.stage_index != stage_index - 1
            or prior_reduction.survivor_receipt.status != "SURVIVORS"
        ):
            raise ValueError("E2 prior reduction has the wrong lineage or round")
        source_candidate_ids = set(
            prior_reduction.survivor_receipt.survivor_candidate_ids
        )
        prior_completed = set(
            prior_reduction.survivor_receipt.completed_lineage_cell_ids
        )
        source_selection_sha256 = prior_reduction.sha256

    cells = registry.cells_for("E2")
    pareto_geometry_ids = {geometry.sha256 for geometry in pareto.surviving_geometries}
    rows: list[CellDisposition] = []
    for cell in cells:
        cell_stage = _e2_stage(cell)
        role = scientific_role_for_cell(registry, cell)
        if cell.cell_id in prior_completed:
            status = DispositionStatus.COMPLETED_PRIOR_ROUND
            reason = "completed_prior_halving_round"
        elif cell_stage != stage_index:
            status = DispositionStatus.DEFERRED
            reason = "awaiting_registered_e2_halving_round"
        elif role == "lc_candidate":
            identity = cell.identity
            if identity.scope is None:
                raise ValueError("E2 LC-candidate lacks a geometry")
            cell_geometry = E1GeometryIdentity(
                scope=identity.scope,
                parameterization=identity.parameterization,
                rank=identity.rank,
                alpha_over_rank=identity.alpha_over_rank,
            )
            exact_candidate = not any(
                value is None
                for value in (
                    identity.optimizer,
                    identity.learning_rate,
                    identity.schedule,
                )
            )
            candidate_id = (
                E2CandidateIdentity.from_cell(cell, registry=registry).sha256
                if exact_candidate
                else None
            )
            selected_for_round = cell_geometry.sha256 in pareto_geometry_ids and (
                source_candidate_ids is None or candidate_id in source_candidate_ids
            )
            if not selected_for_round:
                status = DispositionStatus.DEFERRED
                reason = "outside_e1_pareto_or_prior_survivor_set"
            elif not cell.runnable:
                status = (
                    DispositionStatus.NOT_APPLICABLE
                    if cell.status is CellStatus.NOT_APPLICABLE
                    else DispositionStatus.BLOCKED
                )
                reason = cell.reason_code
            else:
                status = DispositionStatus.BLOCKED
                reason = E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON
        elif role in {"tts", "l0_naive"}:
            if not cell.runnable:
                status = (
                    DispositionStatus.NOT_APPLICABLE
                    if cell.status is CellStatus.NOT_APPLICABLE
                    else DispositionStatus.BLOCKED
                )
                reason = cell.reason_code
            else:
                status = DispositionStatus.BLOCKED
                reason = E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON
        elif role in {"target_only", "static"} and not cell.runnable:
            status = (
                DispositionStatus.NOT_APPLICABLE
                if cell.status is CellStatus.NOT_APPLICABLE
                else DispositionStatus.BLOCKED
            )
            reason = cell.reason_code
        elif role in {"target_only", "static"}:
            status = DispositionStatus.BLOCKED
            reason = E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON
        else:
            raise ValueError("E2 registry contains an unsupported scientific role")
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
        reason_code=E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON,
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
    lc_vs_tts_goodput_ratio: float
    lc_vs_tts_confidence_lower_goodput_ratio: float
    lc_vs_static_goodput_ratio: float
    lc_vs_static_confidence_lower_goodput_ratio: float
    hbm_bytes: int
    p99_itl_us: int
    exposed_update_us: int
    minimum_launched_updates: int
    minimum_published_updates: int
    safety_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256("E2 evaluation candidate_id", self.candidate_id)
        _require_sha256("E2 evaluation evidence_sha256", self.evidence_sha256)
        if not isinstance(self.safety_passed, bool) or not isinstance(
            self.confidence_pareto, bool
        ):
            raise TypeError("E2 safety and Pareto flags must be booleans")
        for reference in ("tts", "static"):
            point = getattr(self, f"lc_vs_{reference}_goodput_ratio")
            lower = getattr(self, f"lc_vs_{reference}_confidence_lower_goodput_ratio")
            if (
                not math.isfinite(point)
                or point <= 0
                or not math.isfinite(lower)
                or lower <= 0
                or lower > point
            ):
                raise ValueError(
                    "E2 LC/reference goodput ratios must be finite, positive, "
                    "and confidence-ordered"
                )
        for name in (
            "hbm_bytes",
            "p99_itl_us",
            "exposed_update_us",
            "minimum_launched_updates",
            "minimum_published_updates",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        if self.safety_reason_codes != tuple(sorted(set(self.safety_reason_codes))):
            raise ValueError("E2 safety reasons must be sorted and unique")
        if self.safety_passed and self.safety_reason_codes:
            raise ValueError("safe E2 evaluations cannot carry failure reasons")
        if not self.safety_passed and not self.safety_reason_codes:
            raise ValueError("unsafe E2 evaluations require a reason")
        if self.minimum_published_updates > self.minimum_launched_updates:
            raise ValueError("E2 published updates cannot exceed launched updates")


@dataclass(frozen=True)
class RawEvidenceRunBinding:
    """Substantive receipt-bound activation and execution provenance.

    ``runtime_sha256`` and ``split_sha256`` bind the shared reducer activation
    lineage.  ``execution_plan_sha256`` and ``execution_split_sha256`` bind the
    independently materialized per-cell execution recovered from schema-v4
    completion evidence.  They are deliberately separate domains: a locked
    split contains the per-cell execution split and therefore cannot also be
    that cell's execution split digest without a cryptographic fixed point.
    """

    schema_version: int
    cell_id: str
    experiment: str
    method: str
    scientific_role: str
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
    execution_plan_sha256: str | None = None
    execution_split_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError(
                "only formal raw run-binding schema version 3 is supported"
            )
        if self.execution_plan_sha256 is None or self.execution_split_sha256 is None:
            raise ValueError(
                "raw run-binding schema version 3 requires per-cell execution identity"
            )
        for name in (
            "experiment",
            "method",
            "scientific_role",
            "scientific_unit",
            "run_id",
            "model_pair",
        ):
            _require_text(f"raw run {name}", getattr(self, name))
        allowed_roles = {*E0_METHOD_ROLES, "lc_candidate"}
        if self.scientific_role not in allowed_roles:
            raise ValueError("raw run scientific role is outside the registry")
        if self.scientific_role == "lightcone":
            raise ValueError(
                "formal LightCone run bindings require a path-replayed E2 seal "
                "authority, which is unavailable in this release"
            )
        expected_runtime_method = {
            "target_only": "target_only",
            "static": "static",
            "tts": "tts",
            "l0_naive": "l0",
            "lc_candidate": "l0",
            "lightcone": "l0",
            "onlinespec_ogd": "onlinespec_ogd",
            "onlinespec_opt": "onlinespec_opt",
            "onlinespec_ens": "onlinespec_ens",
        }[self.scientific_role]
        if self.method != expected_runtime_method:
            raise ValueError("raw run method differs from its scientific role")
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
            "execution_plan_sha256",
            "execution_split_sha256",
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
    excluded_mechanism_anchor_cell_ids: tuple[str, ...]
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
        if self.schema_version != 5:
            raise ValueError("only E2 stage-evidence schema version 5 is supported")
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
            ("excluded mechanism anchors", self.excluded_mechanism_anchor_cell_ids),
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
        if len(self.excluded_mechanism_anchor_cell_ids) != 1:
            raise ValueError("E2 stage evidence requires one excluded L0-naive anchor")
        if set(self.excluded_mechanism_anchor_cell_ids) & set(self.completed_cell_ids):
            raise ValueError("E2 mechanism anchor cannot enter selection evidence")
        if len(self.hardware_receipt_sha256s) != len(self.completed_cell_ids) or len(
            self.budget_observation_sha256s
        ) != len(self.completed_cell_ids):
            raise ValueError(
                "E2 hardware/budget evidence must cover every completed cell"
            )
        if len(self.run_bindings) != len(self.completed_cell_ids):
            raise ValueError("E2 run bindings must cover every completed cell")
        evaluation_count = len(self.evaluations)
        role_counts = {
            role: sum(binding.scientific_role == role for binding in self.run_bindings)
            for role in ("target_only", "static", "tts", "l0_naive", "lc_candidate")
        }
        frozen_anchor_count = role_counts["tts"]
        expected_role_counts = {
            "target_only": 1,
            "static": 1,
            "tts": frozen_anchor_count,
            "l0_naive": 0,
            "lc_candidate": evaluation_count,
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
            or frozen_anchor_count < 1
            or role_counts != expected_role_counts
            or sum(role_counts.values()) != len(self.run_bindings)
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
            type(activation) is not ReducerActivationArtifact
            or type(evidence) is not E2StageEvidenceArtifact
            or type(receipt) is not E2SurvivorReceipt
        ):
            raise TypeError("E2 stage reduction requires exact nested artifacts")
        replace(receipt)
        _require_e2_promotion_authority(
            stage_index=receipt.stage_index,
            status=receipt.status,
            reason_code=receipt.reason_code,
        )
        if (
            activation.plan.experiment != "E2"
            or activation.reducer_protocol_sha256 != E2_HALVING_PROTOCOL_SHA256
            or activation.plan.registry_sha256 != evidence.registry_sha256
            or activation.plan.runtime_sha256 != evidence.runtime_sha256
            or activation.plan.split_sha256 != evidence.split_sha256
            or activation.sha256 != evidence.activation_sha256
            or activation.plan.activation_round != f"halving_{evidence.stage_index}"
            or tuple(sorted(activation.plan.activated_cell_ids))
            != tuple(
                sorted(
                    (
                        *evidence.completed_cell_ids,
                        *evidence.excluded_mechanism_anchor_cell_ids,
                    )
                )
            )
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
    recipe_sha256: str
    recipe: AdaptationRecipeDeclaration
    selection_state: Literal["locked_from_raw_halving_3"]

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("only E2 final-recipe schema version 3 is supported")
        for name in (
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "final_stage_reduction_sha256",
            "source_activation_sha256",
            "candidate_id",
            "recipe_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.candidate.sha256 != self.candidate_id:
            raise ValueError("E2 final recipe candidate identity is inconsistent")
        if type(self.recipe) is not AdaptationRecipeDeclaration:
            raise TypeError("E2 final recipe requires its exact registry declaration")
        if self.recipe.sha256 != self.recipe_sha256:
            raise ValueError("E2 final recipe declaration identity is inconsistent")
        key = self.recipe.lookup_key
        candidate = self.candidate
        if (
            key.experiment != "E2"
            or key.backend != candidate.backend
            or key.scope != candidate.scope
            or key.parameterization != candidate.parameterization
            or key.rank != candidate.rank
            or key.alpha_over_rank != candidate.alpha_over_rank
            or key.optimizer != candidate.optimizer
            or key.learning_rate != candidate.learning_rate
            or key.schedule != candidate.schedule
            or key.draft_width != candidate.width
            or key.draft_width_selector != candidate.draft_width_selector
        ):
            raise ValueError(
                "E2 final recipe declaration differs from its candidate identity"
            )
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

    if type(reduction) is not E2StageReductionArtifact:
        raise TypeError("E2 final recipe requires an exact stage reduction")
    receipt = reduction.survivor_receipt
    if type(receipt) is not E2SurvivorReceipt:
        raise TypeError("E2 final recipe requires an exact survivor receipt")
    replace(receipt)
    _require_e2_promotion_authority(
        stage_index=receipt.stage_index,
        status=receipt.status,
        reason_code=receipt.reason_code,
    )
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
    candidates = _e2_candidates(registry, tuple(active.values()))
    candidate_id = receipt.final_recipe_candidate_id
    try:
        candidate, candidate_cell = candidates[candidate_id]
    except KeyError as exc:
        raise ValueError(
            "E2 final candidate is absent from the activated stage"
        ) from exc
    if receipt.survivor_candidate_ids != (candidate_id,):
        raise ValueError("E2 final reduction does not lock exactly one candidate")
    recipe = registry.adaptation_recipe_for_cell(candidate_cell)
    return E2FinalRecipeArtifact(
        schema_version=3,
        registry_sha256=registry.sha256,
        runtime_sha256=reduction.runtime_sha256,
        split_sha256=reduction.split_sha256,
        final_stage_reduction_sha256=reduction.sha256,
        source_activation_sha256=reduction.activation.sha256,
        candidate_id=candidate_id,
        candidate=candidate,
        recipe_sha256=recipe.sha256,
        recipe=recipe,
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
    excluded_mechanism_ids = stage_evidence.excluded_mechanism_anchor_cell_ids
    if (
        len(completed) != len(set(completed))
        or set(completed) & set(excluded_mechanism_ids)
        or set(completed) | set(excluded_mechanism_ids) != active_ids
    ):
        raise ValueError(
            "E2 selection sealing requires exact selection evidence and one "
            "separately planned mechanism anchor"
        )
    known = {cell.cell_id: cell for cell in registry.cells_for("E2")}
    active_cells = tuple(known[cell_id] for cell_id in active_ids)
    candidates = _e2_candidates(registry, active_cells)
    source_ids = tuple(sorted(candidates))
    by_candidate = {row.candidate_id: row for row in stage_evidence.evaluations}
    if set(by_candidate) != set(source_ids):
        raise ValueError("E2 evaluations must exactly cover activated candidates")
    promotion_minimum = _e2_promotion_minimum(stage_index)
    if promotion_minimum.registered:
        minimum_launched = promotion_minimum.minimum_launched_updates_per_adapted_method
        minimum_published = (
            promotion_minimum.minimum_published_updates_per_adapted_method
        )
        if minimum_launched is None or minimum_published is None:
            raise RuntimeError("registered E2 promotion minima are incomplete")
        eligible = tuple(
            row
            for row in by_candidate.values()
            if row.safety_passed
            and row.confidence_pareto
            and row.minimum_launched_updates >= minimum_launched
            and row.minimum_published_updates >= minimum_published
        )
    else:
        eligible = ()
    families = {
        candidate.family
        for candidate, _ in (candidates[candidate_id] for candidate_id in source_ids)
    }
    eligible_by_family: dict[tuple[str, str], list[E2CandidateEvaluation]] = {
        family: [] for family in families
    }
    for row in eligible:
        eligible_by_family[candidates[row.candidate_id][0].family].append(row)
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

    if not promotion_minimum.registered:
        reason_code = promotion_minimum.blocker_reason_code
        if reason_code is None:
            raise RuntimeError("unregistered E2 promotion minima lost their blocker")
        return finish(
            E2SurvivorReceipt(
                **common,
                survivor_candidate_ids=(),
                final_recipe_candidate_id=None,
                status="BLOCKED",
                reason_code=reason_code,
            )
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
        -row.lc_vs_tts_confidence_lower_goodput_ratio,
        -row.lc_vs_static_confidence_lower_goodput_ratio,
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
            parts = tuple(
                part
                for part in variant.removeprefix(prefix).split(":")
                if not part.startswith("role=")
            )
            return ":".join(parts)
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
        panels = {
            part
            for part in variant.split(":")
            if part in {"matched", "deployment_optimal"}
        }
        if len(panels) != 1:
            raise ValueError("E3b cell has an invalid width panel")
        return panels.pop()
    return "not_applicable"


def _confirmation_structural_role(
    registry: ExperimentRegistry, cell: ExperimentCell
) -> str:
    """Map an unmaterialized LightCone template to its paired-family slot.

    A template completes the *registered structure* of a confirmation block,
    but it is not a reportable LightCone run.  The registry owns the exact E2
    seal/materialization boundary; planning may therefore use this mapping
    only to check paired coverage.  Activation continues to use ``runnable``
    and cannot enable the template.
    """

    role = scientific_role_for_cell(registry, cell)
    if role != ScientificMethodRole.LIGHTCONE_TEMPLATE.value:
        return role
    if (
        cell.status is not CellStatus.BLOCKED
        or cell.reason_code != "sealed_e2_recipe_receipt_required"
    ):
        raise ValueError("unmaterialized LightCone template lacks its E2-seal block")
    return ScientificMethodRole.LIGHTCONE.value


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
        if self.method_family != CONFIRMATION_METHOD_ROLES:
            raise ValueError(
                "primary confirmation family must bind all five scientific roles"
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
        method_family=CONFIRMATION_METHOD_ROLES,
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
            and _confirmation_structural_role(registry, cell) in family.method_family
        )
    )
    expected = len(PILOT_BLOCKS + FINAL_BLOCKS) * len(CONFIRMATION_METHOD_ROLES)
    if len(matches) != expected:
        raise ValueError(
            "confirmation family must contain every method in every registered block"
        )
    for block in PILOT_BLOCKS + FINAL_BLOCKS:
        methods = {
            _confirmation_structural_role(registry, cell)
            for cell in matches
            if cell.identity.block == block
        }
        if methods != set(CONFIRMATION_METHOD_ROLES):
            raise ValueError(
                "confirmation family block is not a complete paired method set"
            )
    return tuple(sorted(matches, key=lambda cell: cell.cell_id))


def derive_confirmation_stage_partition(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    runtime_sha256: str,
    split_sha256: str,
    trace_sha256: str,
    sampling_sha256: str,
    hardware_envelope_sha256: str,
) -> tuple[tuple[ConfirmationFamilyIdentity, ...], tuple[ExperimentCell, ...]]:
    """Derive complete primary families and the exact non-family remainder."""

    if experiment not in {"E3b", "E5"}:
        raise ValueError("confirmation stage partition supports only E3b/E5")
    candidates: dict[str, ConfirmationFamilyIdentity] = {}
    for cell in registry.cells_for(experiment):
        family = derive_confirmation_family(
            registry,
            cell_id=cell.cell_id,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            trace_sha256=trace_sha256,
            sampling_sha256=sampling_sha256,
            hardware_envelope_sha256=hardware_envelope_sha256,
        )
        candidates[family.sha256] = family
    families: list[ConfirmationFamilyIdentity] = []
    primary_cell_ids: set[str] = set()
    for family_sha256 in sorted(candidates):
        family = candidates[family_sha256]
        try:
            cells = _family_cells(registry, family)
        except ValueError as error:
            if str(error) not in {
                "confirmation family must contain every method in every registered block",
                "confirmation family block is not a complete paired method set",
            }:
                raise
            continue
        cell_ids = {cell.cell_id for cell in cells}
        if primary_cell_ids & cell_ids:
            raise ValueError("confirmation primary families overlap registry cells")
        primary_cell_ids.update(cell_ids)
        families.append(family)
    stage_cells = registry.cells_for(experiment)
    auxiliary = tuple(
        cell for cell in stage_cells if cell.cell_id not in primary_cell_ids
    )
    if (
        not families
        or len(primary_cell_ids) + len(auxiliary) != len(stage_cells)
        or primary_cell_ids & {cell.cell_id for cell in auxiliary}
    ):
        raise ValueError("confirmation stage partition is not exact")
    return tuple(families), auxiliary


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
    expected = len(PILOT_BLOCKS) * len(CONFIRMATION_METHOD_ROLES)
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
        expected_cells = len(PILOT_BLOCKS) * len(CONFIRMATION_METHOD_ROLES)
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
                (binding.scientific_unit, binding.scientific_role)
                for binding in self.run_bindings
            }
            != {
                (f"excluded_pilot_{block}", method)
                for block in PILOT_BLOCKS
                for method in CONFIRMATION_METHOD_ROLES
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
        CONFIRMATION_METHOD_ROLES
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
        else plan.selected_final_blocks * len(CONFIRMATION_METHOD_ROLES)
    )
    if len(artifact.activated_cell_ids) != expected:
        raise ValueError("family final activation is not the exact selected prefix")
    return artifact


def _budget_activated_cells(
    registry: ExperimentRegistry,
    *,
    activations: Sequence[ReducerActivationArtifact | RegistryStageActivationArtifact],
    family_activations: Sequence[FamilyActivationArtifact],
    family_power_reductions: Sequence[ConfirmationFamilyPowerReductionArtifact],
) -> tuple[
    tuple[ExperimentCell, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    reducer_rows = tuple(activations)
    family_rows = tuple(family_activations)
    power_rows = tuple(family_power_reductions)
    if any(
        type(row) not in {ReducerActivationArtifact, RegistryStageActivationArtifact}
        for row in reducer_rows
    ):
        raise TypeError("budget activations must be reducer artifacts")
    if any(type(row) is not FamilyActivationArtifact for row in family_rows):
        raise TypeError("budget family activations must be reducer artifacts")
    if any(
        type(row) is not ConfirmationFamilyPowerReductionArtifact for row in power_rows
    ):
        raise TypeError("budget family power inputs must be raw reduction artifacts")
    for name, rows in (
        ("activation", reducer_rows),
        ("family activation", family_rows),
        ("family power", power_rows),
    ):
        identities = tuple(row.sha256 for row in rows)
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate budget {name} artifact")

    activated_ids: list[str] = []
    auxiliary_rows: list[ReducerActivationArtifact] = []
    for artifact in reducer_rows:
        if type(artifact) is RegistryStageActivationArtifact:
            verify_registry_stage_activation(registry, artifact)
            activated_ids.extend(artifact.activated_cell_ids)
            continue
        plan = artifact.plan
        if plan.registry_sha256 != registry.sha256:
            raise ValueError("budget activation belongs to another registry")
        stage_ids = {cell.cell_id for cell in registry.cells_for(plan.experiment)}
        disposition_ids = {row.cell_id for row in artifact.dispositions}
        if (
            artifact.reducer_protocol_sha256
            == CONFIRMATION_AUXILIARY_ACTIVATION_PROTOCOL_SHA256
        ):
            if plan.experiment not in {"E3b", "E5"}:
                raise ValueError("budget auxiliary activation names another stage")
            auxiliary_rows.append(artifact)
        elif disposition_ids != stage_ids:
            raise ValueError(
                "budget activation does not disposition its complete stage"
            )
        by_id = {cell.cell_id: cell for cell in registry.cells_for(plan.experiment)}
        if any(not by_id[cell_id].runnable for cell_id in plan.activated_cell_ids):
            raise ValueError("budget activation contains a registry-blocked cell")
        activated_ids.extend(plan.activated_cell_ids)

    pilot_by_family: dict[str, FamilyActivationArtifact] = {}
    final_by_family: dict[str, FamilyActivationArtifact] = {}
    for artifact in family_rows:
        if artifact.family.registry_sha256 != registry.sha256:
            raise ValueError("budget family activation belongs to another registry")
        target = (
            pilot_by_family
            if artifact.activation_round == "excluded_pilots"
            else final_by_family
        )
        family_sha256 = artifact.family.sha256
        if family_sha256 in target:
            raise ValueError("duplicate family activation round")
        target[family_sha256] = artifact
    power_by_family: dict[str, ConfirmationFamilyPowerReductionArtifact] = {}
    for reduction in power_rows:
        if reduction.family.registry_sha256 != registry.sha256:
            raise ValueError("budget family power belongs to another registry")
        family_sha256 = reduction.family.sha256
        if family_sha256 in power_by_family:
            raise ValueError("duplicate family power reduction")
        power_by_family[family_sha256] = reduction
    if set(final_by_family) != set(power_by_family):
        raise ValueError(
            "budget final-prefix activations require one exact family power reduction"
        )
    if set(final_by_family) - pilot_by_family.keys():
        raise ValueError("budget final-prefix activation lacks its pilot activation")
    for family_sha256, pilot in pilot_by_family.items():
        verify_confirmation_pilot_activation(
            registry, family=pilot.family, artifact=pilot
        )
        activated_ids.extend(pilot.activated_cell_ids)
        final = final_by_family.get(family_sha256)
        if final is None:
            continue
        expected_final = materialize_confirmation_prefix(
            registry,
            family=pilot.family,
            reduction=power_by_family[family_sha256],
            pilot_activation=pilot,
        )
        if final != expected_final:
            raise ValueError("budget final-prefix activation is not reducer-generated")
        activated_ids.extend(final.activated_cell_ids)

    if auxiliary_rows:
        if len(auxiliary_rows) != 1 or not family_rows:
            raise ValueError(
                "budget confirmation auxiliary requires one complete family aggregate"
            )
        auxiliary = auxiliary_rows[0]
        identity = family_rows[0].family
        families, auxiliary_cells = derive_confirmation_stage_partition(
            registry,
            experiment=auxiliary.plan.experiment,
            runtime_sha256=auxiliary.plan.runtime_sha256,
            split_sha256=auxiliary.plan.split_sha256,
            trace_sha256=identity.trace_sha256,
            sampling_sha256=identity.sampling_sha256,
            hardware_envelope_sha256=identity.hardware_envelope_sha256,
        )
        if tuple(sorted(pilot_by_family)) != tuple(
            family.sha256 for family in families
        ) or tuple(row.cell_id for row in auxiliary.dispositions) != tuple(
            cell.cell_id for cell in auxiliary_cells
        ):
            raise ValueError(
                "budget confirmation family/auxiliary partition differs from registry"
            )

    if not activated_ids:
        raise ValueError("budget materialization requires at least one activated cell")
    if len(activated_ids) != len(set(activated_ids)):
        raise ValueError("budget activation artifacts overlap activated cells")
    known = {cell.cell_id: cell for cell in registry.cells}
    if set(activated_ids) - known.keys():
        raise ValueError("budget activation contains a cell outside the registry")
    ordered_ids = tuple(sorted(activated_ids))
    return (
        tuple(known[cell_id] for cell_id in ordered_ids),
        tuple(sorted(row.sha256 for row in reducer_rows)),
        tuple(sorted(row.sha256 for row in family_rows)),
        tuple(sorted(row.sha256 for row in power_rows)),
    )


def _load_window_scenarios(
    binding: BudgetLoadBinding, field: str
) -> ScenarioMilliseconds | None:
    values_us = tuple(getattr(plan.window, field) for plan in binding.load_plans)
    if any(value % 1_000 for value in values_us):
        return None
    return ScenarioMilliseconds(*(value // 1_000 for value in values_us))


def _load_semantics_rejection_reason(
    cell: ExperimentCell, binding: BudgetLoadBinding
) -> str | None:
    plans = binding.load_plans
    expected_split = {
        "preflight": "tuning",
        "E3a": "tuning",
        "E1": "tuning",
        "E2": "tuning",
        "E4": "tuning",
        "E3b": "pilot" if cell.identity.block in PILOT_BLOCKS else "confirmation",
        "E1a": "tuning",
        "E5": "pilot" if cell.identity.block in PILOT_BLOCKS else "confirmation",
        "E0": "broad_replication",
    }.get(cell.identity.experiment)
    if expected_split is None or any(
        plan.scored.split != expected_split for plan in plans
    ):
        return "load_split_unresolved"
    if any(
        len(request.input_token_ids) + request.requested_output_tokens
        > (cell.identity.context or 0)
        for plan in plans
        for request in plan.scored.requests
    ):
        return "load_context_unresolved"
    source = dict(binding.registered_load.scored.source_parameters)
    if source.get("cohort_count") != cell.identity.cohort_count:
        return "load_cohort_unresolved"
    arrival = cell.identity.arrival
    expected_source = None
    if arrival.startswith("closed_loop"):
        expected_source = "closed_loop"
    elif arrival in {"immediate_burst", "deterministic_stratified_requests"}:
        expected_source = "immediate_burst"
    elif arrival in {
        "poisson",
        "moderate_soak",
        "saturation_soak",
        "overload_soak",
    }:
        expected_source = "poisson"
    elif arrival == "burstgpt_shape":
        expected_source = "external_shape"
    if expected_source is not None and any(
        plan.scored.source_kind != expected_source for plan in plans
    ):
        return "load_arrival_unresolved"
    if (
        expected_source == "closed_loop"
        and source.get("concurrency") != cell.identity.concurrency
    ):
        return "load_concurrency_unresolved"
    if cell.identity.load_factor is not None and (
        expected_source != "poisson"
        or source.get("registered_load_factor") != cell.identity.load_factor
    ):
        return "load_factor_unresolved"
    if binding.minimum_completed_requests > len(
        binding.registered_load.scored.requests
    ):
        return "minimum_completion_pool_unresolved"
    if any(
        _load_window_scenarios(binding, field) is None
        for field in (
            "warmup_duration_us",
            "arrival_duration_us",
            "request_deadline_us",
            "drain_duration_us",
        )
    ):
        return "load_window_not_integral_milliseconds"
    return None


def _serving_budget(
    *,
    cell: ExperimentCell,
    binding: BudgetLoadBinding,
    job_policy: BudgetJobPolicy,
    inventory: BudgetInventoryIdentity,
) -> ExperimentBudget:
    warmup = _load_window_scenarios(binding, "warmup_duration_us")
    active = _load_window_scenarios(binding, "arrival_duration_us")
    deadline = _load_window_scenarios(binding, "request_deadline_us")
    drain = _load_window_scenarios(binding, "drain_duration_us")
    if warmup is None or active is None or deadline is None or drain is None:
        raise ValueError("unresolved load windows cannot produce an ExperimentBudget")
    active_fields = {
        "scored_arrival": ZERO_MILLISECONDS,
        "soak": ZERO_MILLISECONDS,
        "failure_injection": ZERO_MILLISECONDS,
        "profiler": ZERO_MILLISECONDS,
    }
    field = {
        BudgetJobKind.STANDARD: "scored_arrival",
        BudgetJobKind.SHORT: "scored_arrival",
        BudgetJobKind.P99_ANCHOR: "scored_arrival",
        BudgetJobKind.SOAK: "soak",
        BudgetJobKind.FAILURE: "failure_injection",
        BudgetJobKind.PROFILER: "profiler",
    }[binding.job_kind]
    active_fields[field] = active
    warmup_counts = tuple(
        0 if plan.warmup is None else len(plan.warmup.requests)
        for plan in binding.load_plans
    )
    scored_output_tokens = tuple(
        sum(request.requested_output_tokens for request in plan.scored.requests)
        for plan in binding.load_plans
    )
    wall_time = _sum_scenarios(
        (
            job_policy.startup_model_load,
            job_policy.compile_jit_graph_prewarm,
            warmup,
            active_fields["scored_arrival"],
            drain,
            job_policy.reset_finalization,
            job_policy.evidence_flush_shutdown,
            active_fields["soak"],
            active_fields["failure_injection"],
            job_policy.retry,
            active_fields["profiler"],
            job_policy.download_compile_reservation,
        )
    )
    compute_gpu_ms = wall_time.scale(cell.resources.gpu_count)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=binding.job_kind,
        startup_model_load=job_policy.startup_model_load,
        compile_jit_graph_prewarm=job_policy.compile_jit_graph_prewarm,
        excluded_warmup=warmup,
        excluded_warmup_requests=ExpectedMaximumCount(
            warmup_counts[1], warmup_counts[2]
        ),
        scored_arrival=active_fields["scored_arrival"],
        request_deadline=deadline,
        drain=drain,
        reset_finalization=job_policy.reset_finalization,
        evidence_flush_shutdown=job_policy.evidence_flush_shutdown,
        output_tokens=ExpectedMaximumCount(
            scored_output_tokens[1], scored_output_tokens[2]
        ),
        minimum_completed_requests=binding.minimum_completed_requests,
        p99_anchor_status=binding.p99_anchor_status,
        soak=active_fields["soak"],
        failure_injection=active_fields["failure_injection"],
        retry=job_policy.retry,
        retry_allowance=job_policy.retry_allowance,
        profiler=active_fields["profiler"],
        download_compile_reservation=job_policy.download_compile_reservation,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=compute_gpu_ms
        + job_policy.reserved_gpu_overhead.scale(cell.resources.gpu_count),
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=wall_time.scale(inventory.gpu_count),
    )


def _nonserving_budget(
    *,
    cell: ExperimentCell,
    job_kind: BudgetJobKind,
    job_policy: BudgetJobPolicy,
    inventory: BudgetInventoryIdentity,
) -> ExperimentBudget:
    wall_time = _sum_scenarios(
        (
            job_policy.startup_model_load,
            job_policy.compile_jit_graph_prewarm,
            job_policy.reset_finalization,
            job_policy.evidence_flush_shutdown,
            job_policy.retry,
            job_policy.download_compile_reservation,
        )
    )
    compute_gpu_ms = wall_time.scale(cell.resources.gpu_count)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=job_kind,
        startup_model_load=job_policy.startup_model_load,
        compile_jit_graph_prewarm=job_policy.compile_jit_graph_prewarm,
        excluded_warmup=ZERO_MILLISECONDS,
        excluded_warmup_requests=ZERO_COUNT,
        scored_arrival=ZERO_MILLISECONDS,
        request_deadline=ZERO_MILLISECONDS,
        drain=ZERO_MILLISECONDS,
        reset_finalization=job_policy.reset_finalization,
        evidence_flush_shutdown=job_policy.evidence_flush_shutdown,
        output_tokens=ZERO_COUNT,
        minimum_completed_requests=0,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=ZERO_MILLISECONDS,
        failure_injection=ZERO_MILLISECONDS,
        retry=job_policy.retry,
        retry_allowance=job_policy.retry_allowance,
        profiler=ZERO_MILLISECONDS,
        download_compile_reservation=job_policy.download_compile_reservation,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=compute_gpu_ms
        + job_policy.reserved_gpu_overhead.scale(cell.resources.gpu_count),
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=wall_time.scale(inventory.gpu_count),
    )


def materialize_industrial_budgets(
    registry: ExperimentRegistry,
    *,
    activations: Sequence[
        ReducerActivationArtifact | RegistryStageActivationArtifact
    ] = (),
    family_activations: Sequence[FamilyActivationArtifact] = (),
    family_power_reductions: Sequence[ConfirmationFamilyPowerReductionArtifact] = (),
    load_bindings: Sequence[BudgetLoadBinding] = (),
    policy: BudgetPolicy,
    inventory: BudgetInventoryIdentity,
    capacity_envelope: CapacityEnvelope | None = None,
    capacity_authority: CapacityAuthorityBinding | None = None,
    require_complete: bool = False,
) -> BudgetPlan:
    """Derive complete per-cell budgets from sealed activation and load semantics.

    Missing/non-integral load authority or missing/exceeded capacity becomes an
    immutable ``UNRESOLVED`` disposition.  No duration or request field is
    supplied as an ad-hoc table, and no missing component is silently replaced
    with zero.
    """

    if not isinstance(policy, BudgetPolicy):
        raise TypeError("budget materialization requires a BudgetPolicy")
    if not isinstance(inventory, BudgetInventoryIdentity):
        raise TypeError("budget materialization requires a BudgetInventoryIdentity")
    if capacity_envelope is not None:
        if type(capacity_envelope) is not CapacityEnvelope:
            raise TypeError(
                "budget materialization capacity must be an exact CapacityEnvelope"
            )
        if capacity_envelope.budget_inventory_sha256 != inventory.sha256:
            raise ValueError("capacity envelope belongs to another budget inventory")
    if capacity_authority is not None:
        if type(capacity_authority) is not CapacityAuthorityBinding:
            raise TypeError(
                "budget materialization capacity authority must be an exact binding"
            )
        if capacity_envelope is None:
            raise ValueError("capacity authority requires a capacity envelope")
        if (
            capacity_authority.registry_sha256 != registry.sha256
            or capacity_authority.budget_inventory_sha256 != inventory.sha256
            or capacity_authority.capacity_envelope_sha256 != capacity_envelope.sha256
        ):
            raise ValueError(
                "capacity authority belongs to another registry/inventory/envelope"
            )
    (
        cells,
        activation_sha256s,
        family_activation_sha256s,
        family_power_sha256s,
    ) = _budget_activated_cells(
        registry,
        activations=activations,
        family_activations=family_activations,
        family_power_reductions=family_power_reductions,
    )
    binding_rows = tuple(load_bindings)
    if any(type(row) is not BudgetLoadBinding for row in binding_rows):
        raise TypeError("load bindings must be exact BudgetLoadBinding values")
    binding_by_cell = {row.cell_id: row for row in binding_rows}
    if len(binding_by_cell) != len(binding_rows):
        raise ValueError("budget load bindings contain duplicate cell IDs")
    activated_ids = {cell.cell_id for cell in cells}
    if set(binding_by_cell) - activated_ids:
        raise ValueError("budget load binding names a non-activated cell")
    if capacity_envelope is not None and (
        {row.cell_id for row in capacity_envelope.cell_requirements} - activated_ids
    ):
        raise ValueError("capacity envelope names a non-activated cell")

    budgets: list[ExperimentBudget] = []
    dispositions: list[BudgetDisposition] = []
    for cell in cells:
        binding = binding_by_cell.get(cell.cell_id)
        source_sha256 = (
            binding.sha256
            if binding is not None
            else content_sha256(
                {
                    "schema_version": 1,
                    "cell_id": cell.cell_id,
                    "policy_sha256": policy.sha256,
                    "source": "missing_load_semantics",
                }
            )
        )
        reason: str | None = None
        budget: ExperimentBudget | None = None
        if cell.resources.gpu_count > inventory.gpu_count:
            reason = "insufficient_inventory_gpus"
        elif cell.resources.workload_class in {
            WorkloadClass.COMPILE,
            WorkloadClass.DOWNLOAD,
        }:
            if binding is not None:
                raise ValueError(
                    "non-serving budget cell cannot consume a load binding"
                )
            job_kind = (
                BudgetJobKind.COMPILE
                if cell.resources.workload_class is WorkloadClass.COMPILE
                else BudgetJobKind.DOWNLOAD
            )
            job_policy = policy.for_job(job_kind)
            source_sha256 = content_sha256(
                {
                    "schema_version": 1,
                    "cell_id": cell.cell_id,
                    "job_kind": job_kind,
                    "job_policy_sha256": job_policy.sha256,
                }
            )
            budget = _nonserving_budget(
                cell=cell,
                job_kind=job_kind,
                job_policy=job_policy,
                inventory=inventory,
            )
        elif binding is None:
            reason = "missing_load_semantics"
        else:
            if (cell.resources.workload_class is WorkloadClass.PROFILE) != (
                binding.job_kind is BudgetJobKind.PROFILER
            ):
                reason = "job_workload_class_unresolved"
            else:
                reason = _load_semantics_rejection_reason(cell, binding)
            if reason is None:
                budget = _serving_budget(
                    cell=cell,
                    binding=binding,
                    job_policy=policy.for_job(binding.job_kind),
                    inventory=inventory,
                )
        if budget is None:
            dispositions.append(
                BudgetDisposition(
                    cell_id=cell.cell_id,
                    status=BudgetDispositionStatus.UNRESOLVED,
                    reason_code=reason or "budget_semantics_unresolved",
                    source_semantics_sha256=source_sha256,
                    experiment_budget_sha256=None,
                )
            )
        else:
            budgets.append(budget)
            dispositions.append(
                BudgetDisposition(
                    cell_id=cell.cell_id,
                    status=BudgetDispositionStatus.BUDGETED,
                    reason_code="first_party_budget_materialized",
                    source_semantics_sha256=source_sha256,
                    experiment_budget_sha256=budget.sha256,
                )
            )

    capacity_by_cell = (
        {}
        if capacity_envelope is None
        else {row.cell_id: row for row in capacity_envelope.cell_requirements}
    )
    budget_by_cell = {budget.cell_id: budget for budget in budgets}
    capacity_authority_reason: str | None = None
    if capacity_envelope is not None:
        if capacity_authority is None:
            capacity_authority_reason = "capacity_raw_authority_missing"
        else:
            from lightcone_spec.experiments.capacity_authority import (
                CapacityAuthorityUnavailableError,
                revalidate_capacity_authority_binding,
            )

            try:
                revalidate_capacity_authority_binding(
                    capacity_authority,
                    expected_registry_sha256=registry.sha256,
                    expected_inventory=inventory,
                    expected_envelope=capacity_envelope,
                )
            except CapacityAuthorityUnavailableError as error:
                capacity_authority_reason = error.reason_code
    capacity_reason_by_cell: dict[str, str] = {}
    if capacity_envelope is None:
        capacity_reason_by_cell = {
            cell_id: "capacity_envelope_missing" for cell_id in budget_by_cell
        }
    elif set(capacity_by_cell) != activated_ids:
        capacity_reason_by_cell = {
            cell_id: (
                "capacity_requirement_missing"
                if cell_id not in capacity_by_cell
                else "capacity_requirement_coverage_incomplete"
            )
            for cell_id in budget_by_cell
        }
    elif set(budget_by_cell) != activated_ids:
        capacity_reason_by_cell = {
            cell_id: "capacity_budget_coverage_incomplete" for cell_id in budget_by_cell
        }
    else:
        capacity_rejection = _ready_capacity_rejection_reason(
            capacity_envelope, budgets
        )
        if capacity_rejection is not None:
            capacity_reason_by_cell = {
                cell_id: capacity_rejection for cell_id in budget_by_cell
            }
        elif capacity_authority_reason is not None:
            capacity_reason_by_cell = {
                cell_id: capacity_authority_reason for cell_id in budget_by_cell
            }

    capacity_sha256 = None if capacity_envelope is None else capacity_envelope.sha256
    adjusted_dispositions: list[BudgetDisposition] = []
    for disposition in dispositions:
        requirement = capacity_by_cell.get(disposition.cell_id)
        source_sha256 = content_sha256(
            {
                "schema_version": 1,
                "cell_source_semantics_sha256": (disposition.source_semantics_sha256),
                "capacity_envelope_sha256": capacity_sha256,
                "capacity_requirement_sha256": (
                    None if requirement is None else requirement.sha256
                ),
            }
        )
        capacity_reason = capacity_reason_by_cell.get(disposition.cell_id)
        adjusted_dispositions.append(
            BudgetDisposition(
                cell_id=disposition.cell_id,
                status=(
                    BudgetDispositionStatus.UNRESOLVED
                    if capacity_reason is not None
                    else disposition.status
                ),
                reason_code=capacity_reason or disposition.reason_code,
                source_semantics_sha256=source_sha256,
                experiment_budget_sha256=(
                    None
                    if capacity_reason is not None
                    else disposition.experiment_budget_sha256
                ),
            )
        )
    dispositions = adjusted_dispositions
    plan = BudgetPlan(
        schema_version=2,
        registry_sha256=registry.sha256,
        activation_sha256=_budget_activation_sha256(
            activation_sha256s,
            family_activation_sha256s,
            family_power_sha256s,
        ),
        reducer_activation_sha256s=activation_sha256s,
        family_activation_sha256s=family_activation_sha256s,
        family_power_reduction_sha256s=family_power_sha256s,
        policy=policy,
        inventory=inventory,
        capacity_envelope=capacity_envelope,
        capacity_authority=capacity_authority,
        activated_cell_ids=tuple(cell.cell_id for cell in cells),
        budgets=tuple(sorted(budgets, key=lambda row: row.cell_id)),
        dispositions=tuple(sorted(dispositions, key=lambda row: row.cell_id)),
        status=(
            "UNRESOLVED"
            if any(
                row.status is BudgetDispositionStatus.UNRESOLVED for row in dispositions
            )
            else "READY"
        ),
        reducer_protocol_sha256=BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
    )
    if require_complete:
        plan.require_ready()
    return plan


@dataclass(frozen=True)
class ExecutionSemanticsIdentity:
    """Legacy self-described semantics retained only for wire rejection.

    Constructing this value does not establish alias authority.  Formal
    analysis accepts only :class:`EvidenceAliasReductionArtifact` values that
    it reproduces from raw execution and terminal artifacts.
    """

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
    """Legacy caller-described receipt; never formal alias authority."""

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
                "evidence alias execution semantics are not content-identical"
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
class ExecutionDerivedAliasSemantics:
    """Execution semantics reconstructed by the first-party raw reducer."""

    schema_version: int
    target_model: str
    target_revision: str
    runtime_authority_sha256: str
    patched_tree_identity: str
    run_config_sha256: str
    sampling_profile_sha256: str
    seed: int
    load_plan_sha256: str
    warmup_corpus_sha256: str | None
    request_corpus_sha256: str
    arrival_trace_sha256: str
    request_ids_sha256: str
    maximum_context_tokens: int
    maximum_output_tokens: int
    split_semantics_sha256: str
    model_lock_sha256: str
    experiment_budget_semantics_sha256: str
    hardware_envelope_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    fixed_instance_gpu_count: int
    topology: str
    rank_layout_sha256: str
    method: Literal["target_only"]
    method_implementation_sha256: str
    server_config_sha256: str
    evidence_schema: Literal["schema_v3_native_terminal_v1"]
    output_token_contract_sha256: str
    timing_contract_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "only execution-derived alias semantics schema version 1 is supported"
            )
        for name in (
            "target_model",
            "target_revision",
            "patched_tree_identity",
            "topology",
        ):
            _require_text(f"execution-derived alias {name}", getattr(self, name))
        for name in (
            "runtime_authority_sha256",
            "run_config_sha256",
            "sampling_profile_sha256",
            "load_plan_sha256",
            "request_corpus_sha256",
            "arrival_trace_sha256",
            "request_ids_sha256",
            "split_semantics_sha256",
            "model_lock_sha256",
            "experiment_budget_semantics_sha256",
            "hardware_envelope_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "rank_layout_sha256",
            "method_implementation_sha256",
            "server_config_sha256",
            "output_token_contract_sha256",
            "timing_contract_sha256",
        ):
            _require_sha256(f"execution-derived alias {name}", getattr(self, name))
        if self.warmup_corpus_sha256 is not None:
            _require_sha256(
                "execution-derived alias warmup corpus",
                self.warmup_corpus_sha256,
            )
        _require_nonnegative_int("execution-derived alias seed", self.seed)
        if (
            not isinstance(self.maximum_context_tokens, int)
            or isinstance(self.maximum_context_tokens, bool)
            or self.maximum_context_tokens < 1
            or not isinstance(self.maximum_output_tokens, int)
            or isinstance(self.maximum_output_tokens, bool)
            or self.maximum_output_tokens < 1
            or not isinstance(self.fixed_instance_gpu_count, int)
            or isinstance(self.fixed_instance_gpu_count, bool)
            or self.fixed_instance_gpu_count < 1
        ):
            raise ValueError(
                "execution-derived alias limits and inventory count must be positive"
            )
        if self.method != "target_only":
            raise ValueError("execution-derived aliases are Target-only only")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


_ALIAS_REASON_BY_PRESENTATION_AXIS = {
    "analysis_panel": "target_only_cross_analysis_reference",
    "backend_label": "target_only_backend_label_only",
    "breadth_panel_label": "target_only_breadth_reference",
    "load_panel_label": "identical_materialized_load_plan",
    "width_panel_label": "identical_selected_width_panel",
}


@dataclass(frozen=True)
class EvidenceAliasReductionArtifact:
    """First-party proof that one Target-only observation may be reused."""

    schema_version: int
    registry_sha256: str
    source_cell_id: str
    target_cell_id: str
    source_cell_declaration_sha256: str
    target_cell_declaration_sha256: str
    source_execution_plan_file_sha256: str
    source_execution_plan_sha256: str
    target_execution_plan_file_sha256: str
    target_execution_plan_sha256: str
    raw_manifest_sha256: str
    source_semantics: ExecutionDerivedAliasSemantics
    target_semantics: ExecutionDerivedAliasSemantics
    source_run_binding: RawEvidenceRunBinding
    source_native_terminal_sha256s: tuple[str, ...]
    removed_presentation_axis: str
    source_presentation_value: str
    target_presentation_value: str
    reason_code: str
    target_result_status: Literal["ABSENT_REUSED_SOURCE"]
    reducer_protocol_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "only evidence alias reduction schema version 1 is supported"
            )
        for name in (
            "registry_sha256",
            "source_cell_id",
            "target_cell_id",
            "source_cell_declaration_sha256",
            "target_cell_declaration_sha256",
            "source_execution_plan_file_sha256",
            "source_execution_plan_sha256",
            "target_execution_plan_file_sha256",
            "target_execution_plan_sha256",
            "raw_manifest_sha256",
        ):
            _require_sha256(f"evidence alias reduction {name}", getattr(self, name))
        if self.source_cell_id == self.target_cell_id:
            raise ValueError("evidence alias reduction requires distinct cells")
        if (
            type(self.source_semantics) is not ExecutionDerivedAliasSemantics
            or type(self.target_semantics) is not ExecutionDerivedAliasSemantics
        ):
            raise TypeError(
                "evidence alias reduction requires execution-derived semantics"
            )
        if self.source_semantics != self.target_semantics:
            raise ValueError(
                "source and target execution-derived semantics are not equivalent"
            )
        if type(self.source_run_binding) is not RawEvidenceRunBinding:
            raise TypeError("evidence alias reduction requires a raw run binding")
        if (
            self.source_run_binding.cell_id != self.source_cell_id
            or self.source_run_binding.method != "target_only"
        ):
            raise ValueError(
                "evidence alias source run differs from its Target-only registry cell"
            )
        if (
            not self.source_native_terminal_sha256s
            or len(self.source_native_terminal_sha256s)
            != self.source_run_binding.rank_count
            or len(set(self.source_native_terminal_sha256s))
            != len(self.source_native_terminal_sha256s)
        ):
            raise ValueError(
                "evidence alias reduction lacks unique native terminal rank coverage"
            )
        for digest in self.source_native_terminal_sha256s:
            _require_sha256("evidence alias native terminal", digest)
        expected_reason = _ALIAS_REASON_BY_PRESENTATION_AXIS.get(
            self.removed_presentation_axis
        )
        if expected_reason is None or self.reason_code != expected_reason:
            raise ValueError(
                "evidence alias reason does not match one registered presentation axis"
            )
        _require_text(
            "evidence alias source presentation value",
            self.source_presentation_value,
        )
        _require_text(
            "evidence alias target presentation value",
            self.target_presentation_value,
        )
        if self.source_presentation_value == self.target_presentation_value:
            raise ValueError("the removed presentation axis must actually differ")
        if self.target_result_status != "ABSENT_REUSED_SOURCE":
            raise ValueError(
                "an evidence alias target cannot have an independent result"
            )
        if self.reducer_protocol_sha256 != EVIDENCE_ALIAS_REDUCER_PROTOCOL_SHA256:
            raise ValueError(
                "evidence alias reduction uses an unknown reducer protocol"
            )

    @cached_property
    def dependence_unit_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "execution_derived_evidence_alias_dependence_unit",
                "source_cell_id": self.source_cell_id,
                "source_run_binding": self.source_run_binding,
                "source_semantics_sha256": self.source_semantics.sha256,
                "native_terminal_sha256s": self.source_native_terminal_sha256s,
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
    aliases: Sequence[EvidenceAliasReductionArtifact],
) -> EvidenceDependenceMap:
    """Collapse reducer-proven aliases for bootstrap/covariance units.

    Legacy :class:`EvidenceAliasReceipt` values and caller-authored semantic
    identities are intentionally rejected at this boundary.
    """

    direct = tuple(direct_observation_cell_ids)
    if len(direct) != len(set(direct)) or any(
        not _is_sha256(value) for value in direct
    ):
        raise ValueError("direct observations must be unique cell SHA-256 values")
    alias_rows = tuple(aliases)
    if any(type(row) is not EvidenceAliasReductionArtifact for row in alias_rows):
        raise TypeError(
            "dependence aliases must be first-party EvidenceAliasReductionArtifact "
            "values"
        )
    targets = tuple(row.target_cell_id for row in alias_rows)
    if len(targets) != len(set(targets)):
        raise ValueError("an alias target can be defined only once")
    if set(targets) & set(direct):
        raise ValueError("an alias target cannot also be an independent observation")
    if any(row.source_cell_id not in set(direct) for row in alias_rows):
        raise ValueError(
            "alias chains are forbidden; every source needs direct evidence"
        )
    members: dict[str, list[str]] = {source: [source] for source in direct}
    unit_ids: dict[str, str] = {
        source: content_sha256({"direct_observation_cell_id": source})
        for source in direct
    }
    for row in alias_rows:
        members[row.source_cell_id].append(row.target_cell_id)
        existing = unit_ids[row.source_cell_id]
        if len(members[row.source_cell_id]) == 2:
            unit_ids[row.source_cell_id] = row.dependence_unit_sha256
        elif unit_ids[row.source_cell_id] != row.dependence_unit_sha256:
            raise ValueError("aliases from one source bind inconsistent evidence")
        if (
            len(members[row.source_cell_id]) > 2
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
