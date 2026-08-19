"""Two-layer GPU-hour reporting for ``formal_single_operator_v1``.

The trusted single-operator workflow needs a useful launch plan before any
duration has been observed, but it must not turn an assumed duration into an
experiment result.  This module therefore has exactly two outputs:

* a pre-pilot count plan with ``duration_unmeasured`` and deterministic
  minimum pilot IDs; and
* a post-pilot report derived from the existing lifecycle GPU-hour reducers.

There is intentionally no registry/signature/coverage gate in this adapter.
The post-pilot path still deep-revalidates the lifecycle source so numeric
durations cannot enter through caller-provided scalars.  E5's 264 diagnostic
rows use only the dedicated integrated failure-lifecycle source and are added
exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments import gpu_hour_authority as _gpu_hours
from lightcone_spec.experiments.formal_protocol import (
    FormalRuntimeAuthorityManifest,
    ProtocolLock,
    content_sha256,
)
from lightcone_spec.experiments.gpu_hour_authority import (
    LifecycleGpuHourSourceManifest,
    ProspectiveGpuHourCost,
    StagedProspectiveGpuHourCost,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.stage_materialization import (
    StageGpuHourEnvelope,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_single_operator_gpu_hour_protocol",
        "pre_pilot": (
            "fixed_materialized_counts_duration_unmeasured_and_deterministic_"
            "minimum_pilot_ids"
        ),
        "post_pilot": (
            "deep_revalidated_lifecycle_actual_plus_same_stratum_projection_"
            "from_old_formal_source_or_current_single_operator_run_manifests"
        ),
        "current_run_manifest_source": (
            "exact_materialization_plus_root_revalidated_manifest_per_cell_"
            "plus_recorded_lifecycle_integer_boundaries_and_gpu_counts_no_scalars"
        ),
        "root_manifest_join": (
            "immutable_complete_manifest",
            "stage_cell_and_inventory_lineage",
            "manifest_gpu_environment_subset_of_inventory",
            "admission_artifacts_and_digests_are_not_gpu_hour_inputs",
        ),
        "e5_one_shot": "exact_264_dedicated_integrated_failure_lifecycles_once",
        "stage_disambiguation": (
            "stage_plus_pilot_and_final_materialization_rules_and_receipt_sha256s"
        ),
        "forbidden_inputs": (
            "caller_duration",
            "caller_gpu_hours",
            "caller_complete_status",
            "coverage_or_signature_ceremony",
        ),
    }
)
FORMAL_SINGLE_OPERATOR_SERVING_GPU_HOUR_SOURCE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_formal_single_operator_serving_gpu_hour_source_protocol",
        "actual_inputs": (
            "root_revalidated_fresh_run_manifest",
            "root_revalidated_resident_run_manifest",
        ),
        "physical_execution_identity": (
            "fresh_process_once_per_run_manifest",
            "resident_process_once_per_shared_session_receipt",
        ),
        "actual_cost": (
            "compute_union_per_gpu_uuid",
            "reject_distinct_physical_execution_overlap_on_same_gpu",
            "provider_core_global_interval_union",
            "evidence_tail_global_union_minus_core_union",
        ),
        "projection": (
            "fresh_process_lifecycle",
            "resident_member_trace_lifecycle",
        ),
        "forbidden_inputs": (
            "caller_duration",
            "caller_gpu_hours",
            "caller_complete_status",
        ),
    }
)

_EARLY_STAGED_STAGES = frozenset({"E3a", "TTS-Cal", "E1", "E2", "E4", "E1a"})
_DOWNSTREAM_PILOT_STAGE_BY_RULE = {
    "e3b_exact_480_rows_x_4_excluded_pilot_blocks": "E3b",
    "e5_exact_450_headline_rows_x_4_excluded_pilot_blocks": "E5",
    "e6_exact_two_model_preflights_plus_60_rows_x_4_excluded_pilot_blocks": "E6",
    "e0_full_registered_onlinespec_grid_per_valid_combination_tuning_only": "E0",
    "e0_exact_16_rows_per_valid_combination_x_4_excluded_pilot_blocks": "E0",
}
_DOWNSTREAM_PROJECTION_STAGES = frozenset({"E3b", "E5", "E0"})


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _strict(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


class FormalSingleOperatorGpuHourBlocked(RuntimeError):
    """The trusted workflow still lacks a required measured lifecycle."""

    def __init__(
        self,
        reason_code: str,
        *,
        minimum_pilot_cell_ids: tuple[str, ...] = (),
    ) -> None:
        if type(reason_code) is not str or not reason_code:
            raise ValueError("single-operator GPU-hour reason must be text")
        if type(minimum_pilot_cell_ids) is not tuple or minimum_pilot_cell_ids != tuple(
            sorted(set(minimum_pilot_cell_ids))
        ):
            raise ValueError("single-operator missing pilot IDs are not canonical")
        self.reason_code = reason_code
        self.minimum_pilot_cell_ids = minimum_pilot_cell_ids
        super().__init__(f"single-operator GPU hours are BLOCKED: {reason_code}")


@dataclass(frozen=True)
class FormalSingleOperatorPrePilotGpuHours:
    """Exact work count before any duration is available."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_pre_pilot_gpu_hours"]
    protocol_sha256: str
    stage: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    materialization_rule: str
    fixed_cell_count: int
    projection_stratum_count: int
    minimum_pilot_cell_ids: tuple[str, ...]
    duration_status: Literal["duration_unmeasured"]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_pre_pilot_gpu_hours"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256
            or self.duration_status != "duration_unmeasured"
        ):
            raise ValueError("single-operator pre-pilot GPU-hour identity differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
        ):
            _sha256(f"single-operator pre-pilot {label}", digest)
        if (
            type(self.stage) is not str
            or not self.stage
            or type(self.materialization_rule) is not str
            or not self.materialization_rule
            or type(self.fixed_cell_count) is not int
            or self.fixed_cell_count < 0
            or type(self.projection_stratum_count) is not int
            or self.projection_stratum_count < 0
            or self.projection_stratum_count > self.fixed_cell_count
            or type(self.minimum_pilot_cell_ids) is not tuple
            or len(self.minimum_pilot_cell_ids) != self.projection_stratum_count
            or self.minimum_pilot_cell_ids
            != tuple(sorted(set(self.minimum_pilot_cell_ids)))
        ):
            raise ValueError("single-operator pre-pilot counts are not canonical")
        if self.fixed_cell_count == 0 and (
            self.stage != "E0"
            or self.projection_stratum_count != 0
            or self.minimum_pilot_cell_ids
        ):
            raise ValueError("only an all-N/A E0 plan may have no GPU-hour work")
        if self.fixed_cell_count > 0 and self.projection_stratum_count < 1:
            raise ValueError("non-empty GPU-hour work requires a pilot stratum")
        for cell_id in self.minimum_pilot_cell_ids:
            _sha256("single-operator minimum pilot cell", cell_id)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "stage": self.stage,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "materialization_rule": self.materialization_rule,
            "fixed_cell_count": self.fixed_cell_count,
            "projection_stratum_count": self.projection_stratum_count,
            "minimum_pilot_cell_ids": list(self.minimum_pilot_cell_ids),
            "duration_status": self.duration_status,
        }
        if include_sha256:
            value["output_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator pre-pilot GPU hours",
            value,
            {*cls.__dataclass_fields__, "output_sha256"},
        )
        declared = _sha256("single-operator pre-pilot output", row.pop("output_sha256"))
        minimum = row.pop("minimum_pilot_cell_ids")
        if type(minimum) is not list:
            raise TypeError("single-operator minimum pilot IDs must be an array")
        output = cls(**row, minimum_pilot_cell_ids=tuple(minimum))  # type: ignore[arg-type]
        if output.sha256 != declared:
            raise ValueError("single-operator pre-pilot output digest differs")
        return output


FormalSingleOperatorGpuHourCategory = Literal[
    "actual_pilot",
    "projected_remaining",
    "actual_one_shot",
    "total",
]


@dataclass(frozen=True)
class FormalSingleOperatorGpuHourCost:
    """Exact nanosecond accounting for one non-overlapping cost component."""

    category: FormalSingleOperatorGpuHourCategory
    cell_count: int
    compute_gpu_ns: int
    provider_base_reserved_gpu_ns: int
    wall_ns: int
    retry_reserve_gpu_ns: int
    profile_reserve_gpu_ns: int
    evidence_reserve_gpu_ns: int

    def __post_init__(self) -> None:
        if self.category not in {
            "actual_pilot",
            "projected_remaining",
            "actual_one_shot",
            "total",
        }:
            raise ValueError("single-operator GPU-hour cost category differs")
        for label, value in (
            ("cell count", self.cell_count),
            ("compute", self.compute_gpu_ns),
            ("provider base", self.provider_base_reserved_gpu_ns),
            ("wall", self.wall_ns),
            ("retry", self.retry_reserve_gpu_ns),
            ("profile", self.profile_reserve_gpu_ns),
            ("evidence", self.evidence_reserve_gpu_ns),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"single-operator GPU-hour {label} must be non-negative"
                )

    @property
    def reserved_gpu_ns(self) -> int:
        return (
            self.provider_base_reserved_gpu_ns
            + self.retry_reserve_gpu_ns
            + self.profile_reserve_gpu_ns
            + self.evidence_reserve_gpu_ns
        )

    @property
    def compute_gpu_hours(self) -> float:
        return self.compute_gpu_ns / _gpu_hours.NANOSECONDS_PER_HOUR

    @property
    def reserved_gpu_hours(self) -> float:
        return self.reserved_gpu_ns / _gpu_hours.NANOSECONDS_PER_HOUR

    @property
    def estimated_wall_hours(self) -> float:
        return self.wall_ns / _gpu_hours.NANOSECONDS_PER_HOUR

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "cell_count": self.cell_count,
            "compute_gpu_ns": self.compute_gpu_ns,
            "provider_base_reserved_gpu_ns": self.provider_base_reserved_gpu_ns,
            "wall_ns": self.wall_ns,
            "retry_reserve_gpu_ns": self.retry_reserve_gpu_ns,
            "profile_reserve_gpu_ns": self.profile_reserve_gpu_ns,
            "evidence_reserve_gpu_ns": self.evidence_reserve_gpu_ns,
            "compute_gpu_hours": self.compute_gpu_hours,
            "reserved_gpu_hours": self.reserved_gpu_hours,
            "estimated_wall_hours": self.estimated_wall_hours,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        computed = {
            "compute_gpu_hours",
            "reserved_gpu_hours",
            "estimated_wall_hours",
        }
        row = _strict(
            "single-operator GPU-hour cost",
            value,
            {*cls.__dataclass_fields__, *computed},
        )
        declared = {name: row.pop(name) for name in computed}
        output = cls(**row)  # type: ignore[arg-type]
        expected = output.to_dict()
        if any(declared[name] != expected[name] for name in computed):
            raise ValueError("single-operator GPU-hour decimal projection differs")
        return output


@dataclass(frozen=True)
class FormalSingleOperatorPostPilotGpuHours:
    """Actual pilot cost plus source-derived projection and optional E5 cost."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_post_pilot_gpu_hours"]
    protocol_sha256: str
    stage: str
    protocol_lock_sha256: str
    pilot_materialization_receipt_sha256: str
    pilot_materialization_rule: str
    final_materialization_receipt_sha256: str | None
    final_materialization_rule: str | None
    inventory_sha256: str
    hardware_envelope_sha256: str
    duration_status: Literal["measured_and_projected"]
    pilot_lifecycle_source: CanonicalJsonProofBinding
    one_shot_lifecycle_source: CanonicalJsonProofBinding | None
    mapping_sha256: str
    actual_pilot: FormalSingleOperatorGpuHourCost
    projected_remaining: FormalSingleOperatorGpuHourCost
    actual_one_shot: FormalSingleOperatorGpuHourCost | None
    total: FormalSingleOperatorGpuHourCost

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_post_pilot_gpu_hours"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256
            or self.duration_status != "measured_and_projected"
        ):
            raise ValueError("single-operator post-pilot GPU-hour identity differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("pilot materialization", self.pilot_materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware", self.hardware_envelope_sha256),
            ("mapping", self.mapping_sha256),
        ):
            _sha256(f"single-operator post-pilot {label}", digest)
        if self.final_materialization_receipt_sha256 is not None:
            _sha256(
                "single-operator final materialization",
                self.final_materialization_receipt_sha256,
            )
        if (
            type(self.pilot_materialization_rule) is not str
            or not self.pilot_materialization_rule
            or (
                self.final_materialization_receipt_sha256 is None
                and self.final_materialization_rule is not None
            )
            or (
                self.final_materialization_receipt_sha256 is not None
                and (
                    type(self.final_materialization_rule) is not str
                    or not self.final_materialization_rule
                )
            )
        ):
            raise ValueError("single-operator post-pilot materialization rules differ")
        if type(self.pilot_lifecycle_source) is not CanonicalJsonProofBinding:
            raise TypeError("single-operator pilot lifecycle source is not path-bound")
        if (
            self.one_shot_lifecycle_source is not None
            and type(self.one_shot_lifecycle_source) is not CanonicalJsonProofBinding
        ):
            raise TypeError("single-operator one-shot source is not path-bound")
        if (
            self.one_shot_lifecycle_source is not None
            and self.one_shot_lifecycle_source.absolute_path
            == self.pilot_lifecycle_source.absolute_path
        ):
            raise ValueError("single-operator pilot and one-shot sources are reused")
        if (
            type(self.actual_pilot) is not FormalSingleOperatorGpuHourCost
            or type(self.projected_remaining) is not FormalSingleOperatorGpuHourCost
            or type(self.total) is not FormalSingleOperatorGpuHourCost
            or (
                self.actual_one_shot is not None
                and type(self.actual_one_shot) is not FormalSingleOperatorGpuHourCost
            )
        ):
            raise TypeError("single-operator GPU-hour component types differ")
        if (
            self.actual_pilot.category != "actual_pilot"
            or self.projected_remaining.category != "projected_remaining"
            or self.total.category != "total"
            or (self.actual_one_shot is None)
            != (self.one_shot_lifecycle_source is None)
            or (
                self.actual_one_shot is not None
                and self.actual_one_shot.category != "actual_one_shot"
            )
        ):
            raise ValueError("single-operator GPU-hour component union differs")
        components = (self.actual_pilot, self.projected_remaining) + (
            () if self.actual_one_shot is None else (self.actual_one_shot,)
        )
        expected = _sum_costs(*components)
        if self.total != expected:
            raise ValueError(
                "single-operator GPU-hour total double-counts or omits work"
            )
        if self.stage == "E5":
            if self.actual_one_shot is None or self.actual_one_shot.cell_count != 264:
                raise ValueError(
                    "single-operator E5 requires exact 264 actual one-shots"
                )
        elif self.actual_one_shot is not None:
            raise ValueError("only E5 carries one-shot failure GPU hours")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "stage": self.stage,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "pilot_materialization_receipt_sha256": (
                self.pilot_materialization_receipt_sha256
            ),
            "pilot_materialization_rule": self.pilot_materialization_rule,
            "final_materialization_receipt_sha256": (
                self.final_materialization_receipt_sha256
            ),
            "final_materialization_rule": self.final_materialization_rule,
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "duration_status": self.duration_status,
            "pilot_lifecycle_source": self.pilot_lifecycle_source.to_dict(),
            "one_shot_lifecycle_source": (
                None
                if self.one_shot_lifecycle_source is None
                else self.one_shot_lifecycle_source.to_dict()
            ),
            "mapping_sha256": self.mapping_sha256,
            "actual_pilot": self.actual_pilot.to_dict(),
            "projected_remaining": self.projected_remaining.to_dict(),
            "actual_one_shot": (
                None if self.actual_one_shot is None else self.actual_one_shot.to_dict()
            ),
            "total": self.total.to_dict(),
        }
        if include_sha256:
            value["output_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "single-operator post-pilot GPU hours",
            value,
            {*cls.__dataclass_fields__, "output_sha256"},
        )
        declared = _sha256(
            "single-operator post-pilot output", row.pop("output_sha256")
        )
        row["pilot_lifecycle_source"] = CanonicalJsonProofBinding.from_dict(
            row["pilot_lifecycle_source"]
        )
        if row["one_shot_lifecycle_source"] is not None:
            row["one_shot_lifecycle_source"] = CanonicalJsonProofBinding.from_dict(
                row["one_shot_lifecycle_source"]
            )
        for name in ("actual_pilot", "projected_remaining", "total"):
            row[name] = FormalSingleOperatorGpuHourCost.from_dict(row[name])
        if row["actual_one_shot"] is not None:
            row["actual_one_shot"] = FormalSingleOperatorGpuHourCost.from_dict(
                row["actual_one_shot"]
            )
        output = cls(**row)  # type: ignore[arg-type]
        if output.sha256 != declared:
            raise ValueError("single-operator post-pilot output digest differs")
        return output


FormalSingleOperatorGpuHourOutput = (
    FormalSingleOperatorPrePilotGpuHours | FormalSingleOperatorPostPilotGpuHours
)


def _cost(
    category: FormalSingleOperatorGpuHourCategory,
    source: ProspectiveGpuHourCost | StagedProspectiveGpuHourCost,
) -> FormalSingleOperatorGpuHourCost:
    return FormalSingleOperatorGpuHourCost(
        category=category,
        cell_count=source.cell_count,
        compute_gpu_ns=source.compute_gpu_ns,
        provider_base_reserved_gpu_ns=source.provider_base_reserved_gpu_ns,
        wall_ns=source.wall_ns,
        retry_reserve_gpu_ns=source.retry_reserve_gpu_ns,
        profile_reserve_gpu_ns=source.profile_reserve_gpu_ns,
        evidence_reserve_gpu_ns=source.evidence_reserve_gpu_ns,
    )


def _sum_costs(
    *components: FormalSingleOperatorGpuHourCost,
) -> FormalSingleOperatorGpuHourCost:
    return FormalSingleOperatorGpuHourCost(
        category="total",
        cell_count=sum(row.cell_count for row in components),
        compute_gpu_ns=sum(row.compute_gpu_ns for row in components),
        provider_base_reserved_gpu_ns=sum(
            row.provider_base_reserved_gpu_ns for row in components
        ),
        wall_ns=sum(row.wall_ns for row in components),
        retry_reserve_gpu_ns=sum(row.retry_reserve_gpu_ns for row in components),
        profile_reserve_gpu_ns=sum(row.profile_reserve_gpu_ns for row in components),
        evidence_reserve_gpu_ns=sum(row.evidence_reserve_gpu_ns for row in components),
    )


@dataclass(frozen=True)
class _SingleOperatorLifecycleObservation:
    cell_id: str
    run_manifest: CanonicalJsonProofBinding
    run_manifest_sha256: str
    lifecycle: CanonicalJsonProofBinding
    topology: str
    gang_gpu_count: int
    provider_reserved_gpu_count: int
    scored_request_count: int
    phase_edges_ns: tuple[tuple[str, int], ...]

    def to_source_row(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "run_manifest": self.run_manifest.to_dict(),
            "run_manifest_sha256": self.run_manifest_sha256,
            "lifecycle": self.lifecycle.to_dict(),
            "topology": self.topology,
            "gang_gpu_count": self.gang_gpu_count,
            "provider_reserved_gpu_count": self.provider_reserved_gpu_count,
            "scored_request_count": self.scored_request_count,
            "phase_edges_ns": dict(self.phase_edges_ns),
        }


_UNIFIED_PHYSICAL_EDGE_NAMES = (
    "server_process_started_ns",
    "process_exited_ns",
    "process_group_empty_checked_ns",
    "evidence_flush_completed_ns",
)


@dataclass(frozen=True)
class _UnifiedPhysicalExecution:
    physical_execution_id: str
    execution_kind: Literal["fresh_process", "resident_session"]
    source: CanonicalJsonProofBinding
    gpu_uuids: tuple[str, ...]
    phase_edges_ns: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _sha256(
            "single-operator physical execution",
            self.physical_execution_id,
        )
        if self.execution_kind not in {"fresh_process", "resident_session"}:
            raise ValueError("single-operator physical execution kind differs")
        if type(self.source) is not CanonicalJsonProofBinding:
            raise TypeError("single-operator physical execution source is not bound")
        if CanonicalJsonProofBinding.bind(self.source.absolute_path) != self.source:
            raise ValueError("single-operator physical execution source changed")
        if (
            type(self.gpu_uuids) is not tuple
            or not self.gpu_uuids
            or self.gpu_uuids != tuple(sorted(set(self.gpu_uuids)))
        ):
            raise ValueError("single-operator physical GPU UUIDs are not canonical")
        if tuple(name for name, _value in self.phase_edges_ns) != (
            _UNIFIED_PHYSICAL_EDGE_NAMES
        ):
            raise ValueError("single-operator physical lifecycle fields differ")
        edges = dict(self.phase_edges_ns)
        if any(type(value) is not int for value in edges.values()):
            raise TypeError("single-operator physical lifecycle must be integral")
        started, exited, empty, flushed = (
            edges[name] for name in _UNIFIED_PHYSICAL_EDGE_NAMES
        )
        if not (0 < started < exited <= empty <= flushed):
            raise ValueError("single-operator physical lifecycle is not ordered")

    def to_source_row(self) -> dict[str, object]:
        return {
            "physical_execution_id": self.physical_execution_id,
            "execution_kind": self.execution_kind,
            "source": self.source.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
            "phase_edges_ns": dict(self.phase_edges_ns),
        }


@dataclass(frozen=True)
class _UnifiedCellObservation:
    cell_id: str
    actual_result: CanonicalJsonProofBinding
    actual_result_sha256: str
    member_lifecycle: CanonicalJsonProofBinding
    physical_execution_id: str
    topology: str
    gang_gpu_count: int
    provider_reserved_gpu_count: int
    scored_request_count: int
    projection_process_ns: int
    projection_core_wall_ns: int
    projection_evidence_tail_ns: int
    projection_source: Literal["fresh_process", "resident_member_trace"]

    def __post_init__(self) -> None:
        _sha256("single-operator unified cell", self.cell_id)
        _sha256(
            "single-operator unified actual result",
            self.actual_result_sha256,
        )
        _sha256(
            "single-operator unified physical execution",
            self.physical_execution_id,
        )
        for label, binding in (
            ("actual result", self.actual_result),
            ("member lifecycle", self.member_lifecycle),
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError(f"single-operator unified {label} is not bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError(f"single-operator unified {label} changed")
        if self.projection_source not in {
            "fresh_process",
            "resident_member_trace",
        }:
            raise ValueError("single-operator projection source differs")
        for label, value in (
            ("gang GPU count", self.gang_gpu_count),
            ("provider GPU count", self.provider_reserved_gpu_count),
            ("scored request count", self.scored_request_count),
            ("projection process", self.projection_process_ns),
            ("projection core wall", self.projection_core_wall_ns),
            ("projection evidence tail", self.projection_evidence_tail_ns),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"single-operator unified {label} is invalid")
        if (
            self.gang_gpu_count < 1
            or self.provider_reserved_gpu_count < self.gang_gpu_count
            or self.scored_request_count < 1
            or self.projection_process_ns < 1
            or self.projection_core_wall_ns < self.projection_process_ns
        ):
            raise ValueError("single-operator unified cell cost shape differs")

    def to_source_row(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "actual_result": self.actual_result.to_dict(),
            "actual_result_sha256": self.actual_result_sha256,
            "member_lifecycle": self.member_lifecycle.to_dict(),
            "physical_execution_id": self.physical_execution_id,
            "topology": self.topology,
            "gang_gpu_count": self.gang_gpu_count,
            "provider_reserved_gpu_count": self.provider_reserved_gpu_count,
            "scored_request_count": self.scored_request_count,
            "projection_process_ns": self.projection_process_ns,
            "projection_core_wall_ns": self.projection_core_wall_ns,
            "projection_evidence_tail_ns": self.projection_evidence_tail_ns,
            "projection_source": self.projection_source,
        }


def _merged_intervals(
    intervals: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if any(
        type(started) is not int
        or type(finished) is not int
        or started < 0
        or finished < started
        for started, finished in intervals
    ):
        raise ValueError("single-operator accounting interval is invalid")
    nonempty = sorted(
        (started, finished) for started, finished in intervals if started < finished
    )
    merged: list[tuple[int, int]] = []
    for started, finished in nonempty:
        if not merged or started > merged[-1][1]:
            merged.append((started, finished))
            continue
        prior_started, prior_finished = merged[-1]
        merged[-1] = (prior_started, max(prior_finished, finished))
    return tuple(merged)


def _interval_measure(intervals: tuple[tuple[int, int], ...]) -> int:
    return sum(finished - started for started, finished in _merged_intervals(intervals))


def _interval_intersection_measure(
    left: tuple[tuple[int, int], ...],
    right: tuple[tuple[int, int], ...],
) -> int:
    left_rows = _merged_intervals(left)
    right_rows = _merged_intervals(right)
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left_rows) and right_index < len(right_rows):
        left_started, left_finished = left_rows[left_index]
        right_started, right_finished = right_rows[right_index]
        total += max(
            0,
            min(left_finished, right_finished) - max(left_started, right_started),
        )
        if left_finished <= right_finished:
            left_index += 1
        else:
            right_index += 1
    return total


def _deduplicate_physical_executions(
    rows: tuple[_UnifiedPhysicalExecution, ...],
) -> tuple[_UnifiedPhysicalExecution, ...]:
    if not rows:
        raise ValueError("single-operator unified accounting has no physical execution")
    by_id: dict[str, _UnifiedPhysicalExecution] = {}
    for row in rows:
        if type(row) is not _UnifiedPhysicalExecution:
            raise TypeError("single-operator unified physical row type differs")
        prior = by_id.get(row.physical_execution_id)
        if prior is not None and prior != row:
            raise ValueError("single-operator physical execution identity is ambiguous")
        by_id[row.physical_execution_id] = row
    return tuple(by_id[key] for key in sorted(by_id))


def _actual_cost_from_unified_observations(
    cells: tuple[_UnifiedCellObservation, ...],
    physical_rows: tuple[_UnifiedPhysicalExecution, ...],
    *,
    inventory_gpu_count: int,
) -> FormalSingleOperatorGpuHourCost:
    if not cells or type(inventory_gpu_count) is not int or inventory_gpu_count < 1:
        raise ValueError("single-operator unified actual-cost inputs differ")
    physical = _deduplicate_physical_executions(physical_rows)
    physical_by_id = {row.physical_execution_id: row for row in physical}
    if (
        len({row.cell_id for row in cells}) != len(cells)
        or len({row.actual_result.absolute_path for row in cells}) != len(cells)
        or {row.physical_execution_id for row in cells} != set(physical_by_id)
    ):
        raise ValueError("single-operator unified cell/physical coverage differs")
    if any(
        row.provider_reserved_gpu_count != inventory_gpu_count
        or row.gang_gpu_count
        != len(physical_by_id[row.physical_execution_id].gpu_uuids)
        for row in cells
    ):
        raise ValueError("single-operator unified cell GPU shape differs")

    by_gpu: dict[str, list[tuple[int, int, int, str]]] = {}
    core_intervals: list[tuple[int, int]] = []
    evidence_intervals: list[tuple[int, int]] = []
    for row in physical:
        edges = dict(row.phase_edges_ns)
        started = edges["server_process_started_ns"]
        exited = edges["process_exited_ns"]
        empty = edges["process_group_empty_checked_ns"]
        flushed = edges["evidence_flush_completed_ns"]
        core_intervals.append((started, empty))
        evidence_intervals.append((empty, flushed))
        for gpu_uuid in row.gpu_uuids:
            by_gpu.setdefault(gpu_uuid, []).append(
                (started, exited, empty, row.physical_execution_id)
            )

    compute_gpu_ns = 0
    for gpu_uuid, intervals in by_gpu.items():
        ordered = sorted(intervals)
        for prior, following in pairwise(ordered):
            # GPU ownership is not released merely because the parent process
            # exited.  A following execution on the same device is admissible
            # only after the prior process group was observed empty.  Evidence
            # flushing happens after that boundary and may overlap safely.
            if following[0] < prior[2]:
                raise FormalSingleOperatorGpuHourBlocked(
                    f"overlapping_physical_execution_on_gpu:{gpu_uuid}"
                )
        compute_gpu_ns += _interval_measure(
            tuple((started, exited) for started, exited, _empty, _identity in ordered)
        )

    core = tuple(core_intervals)
    evidence = tuple(evidence_intervals)
    wall_ns = _interval_measure(core)
    provider_ns = wall_ns * inventory_gpu_count
    evidence_only_ns = _interval_measure(evidence) - _interval_intersection_measure(
        evidence,
        core,
    )
    if evidence_only_ns < 0:
        raise AssertionError(
            "single-operator evidence interval subtraction underflowed"
        )
    evidence_ns = evidence_only_ns * inventory_gpu_count
    retry_ns = (
        compute_gpu_ns * _gpu_hours.FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + _gpu_hours.FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // _gpu_hours.FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
    return FormalSingleOperatorGpuHourCost(
        category="actual_pilot",
        cell_count=len(cells),
        compute_gpu_ns=compute_gpu_ns,
        provider_base_reserved_gpu_ns=provider_ns,
        wall_ns=wall_ns,
        retry_reserve_gpu_ns=retry_ns,
        profile_reserve_gpu_ns=0,
        evidence_reserve_gpu_ns=evidence_ns,
    )


def _fresh_unified_observation(
    *,
    repository_root: str | Path,
    run_manifest_path: str | Path,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
) -> tuple[_UnifiedCellObservation, _UnifiedPhysicalExecution]:
    from lightcone_spec.runtime.formal_single_operator import (
        FormalSingleOperatorRunManifest,
    )

    legacy = _single_operator_lifecycle_observation(
        repository_root=repository_root,
        run_manifest_path=run_manifest_path,
        materialization=materialization,
        inventory=inventory,
    )
    manifest = FormalSingleOperatorRunManifest.from_dict(legacy.run_manifest.reopen())
    if manifest.sha256 != legacy.run_manifest_sha256:
        raise ValueError("single-operator fresh manifest identity differs")
    gpu_uuids = tuple(sorted(row.uuid for row in manifest.gpu_environment))
    edges = dict(legacy.phase_edges_ns)
    process_ns = edges["process_exited_ns"] - edges["execution_started_ns"]
    core_wall_ns = (
        edges["process_group_empty_checked_ns"] - edges["execution_started_ns"]
    )
    evidence_tail_ns = (
        edges["evidence_flush_finished_ns"] - edges["process_group_empty_checked_ns"]
    )
    physical = _UnifiedPhysicalExecution(
        physical_execution_id=manifest.sha256,
        execution_kind="fresh_process",
        source=legacy.run_manifest,
        gpu_uuids=gpu_uuids,
        phase_edges_ns=(
            ("server_process_started_ns", edges["execution_started_ns"]),
            ("process_exited_ns", edges["process_exited_ns"]),
            (
                "process_group_empty_checked_ns",
                edges["process_group_empty_checked_ns"],
            ),
            (
                "evidence_flush_completed_ns",
                edges["evidence_flush_finished_ns"],
            ),
        ),
    )
    cell = _UnifiedCellObservation(
        cell_id=legacy.cell_id,
        actual_result=legacy.run_manifest,
        actual_result_sha256=manifest.sha256,
        member_lifecycle=legacy.lifecycle,
        physical_execution_id=physical.physical_execution_id,
        topology=legacy.topology,
        gang_gpu_count=legacy.gang_gpu_count,
        provider_reserved_gpu_count=legacy.provider_reserved_gpu_count,
        scored_request_count=legacy.scored_request_count,
        projection_process_ns=process_ns,
        projection_core_wall_ns=core_wall_ns,
        projection_evidence_tail_ns=evidence_tail_ns,
        projection_source="fresh_process",
    )
    return cell, physical


def _resident_unified_observation(
    *,
    repository_root: str | Path,
    resident_manifest_path: str | Path,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
) -> tuple[_UnifiedCellObservation, _UnifiedPhysicalExecution]:
    from lightcone_spec.runtime.formal_single_operator import (
        revalidate_formal_single_operator_resident_run_manifest,
    )

    # The resident runtime module owns the exact schema and every source join.
    # This adapter consumes only that verifier-created typed projection; it
    # never accepts lifecycle scalars from a caller or an untyped JSON object.
    resident = revalidate_formal_single_operator_resident_run_manifest(
        repository_root=repository_root,
        manifest_path=resident_manifest_path,
    )
    return _unified_observation_from_revalidated_resident(
        resident=resident,
        resident_manifest_path=resident_manifest_path,
        materialization=materialization,
        inventory=inventory,
    )


def _unified_observation_from_revalidated_resident(
    *,
    resident: object,
    resident_manifest_path: str | Path,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
) -> tuple[_UnifiedCellObservation, _UnifiedPhysicalExecution]:
    """Project the exact resident runtime type after its root revalidator."""

    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
        formal_serving_request_schedule_rows,
    )
    from lightcone_spec.orchestration.formal_serving_session_group_physical import (
        revalidate_formal_serving_resident_shared_close_receipt,
        revalidate_formal_serving_resident_trace_receipt,
    )
    from lightcone_spec.runtime.formal_single_operator import (
        FormalSingleOperatorResidentRunManifest,
    )

    if type(resident) is not FormalSingleOperatorResidentRunManifest:
        raise TypeError("single-operator resident GPU-hour manifest type differs")
    path = Path(resident_manifest_path)
    before = CanonicalJsonProofBinding.bind(path)
    if (
        resident.completion_status != "COMPLETE"
        or resident.stage != materialization.stage
        or resident.materialization_sha256 != materialization.sha256
        or resident.materialization_protocol_lock_sha256
        != materialization.protocol_lock_sha256
        or resident.inventory_sha256 != inventory.sha256
        or resident.inventory.semantic_sha256 != inventory.sha256
    ):
        raise ValueError("single-operator resident GPU-hour lineage differs")
    matches = tuple(
        row for row in materialization.cells if row.cell_id == resident.cell_id
    )
    if len(matches) != 1:
        raise ValueError("single-operator resident GPU-hour cell differs")
    cell = matches[0]
    inventory_uuids = {row.uuid for row in inventory.devices}
    gpu_uuids = tuple(sorted(row.uuid for row in resident.gpu_environment))
    if (
        resident.role != cell.method_role
        or resident.backend != cell.backend
        or resident.topology != "tp1_dp1"
        or len(gpu_uuids) != 1
        or not set(gpu_uuids).issubset(inventory_uuids)
    ):
        raise ValueError("single-operator resident GPU-hour GPU/cell join differs")

    trace_binding, trace = revalidate_formal_serving_resident_trace_receipt(
        resident.resident_trace.absolute_path
    )
    close_binding, close = revalidate_formal_serving_resident_shared_close_receipt(
        resident.shared_close.absolute_path
    )
    if (
        trace_binding != resident.resident_trace
        or close_binding != resident.shared_close
        or trace_binding not in close.member_trace_receipts
        or trace.materialized_cell_id != resident.cell_id
        or trace.trace_started_ns != resident.trace_started_ns
        or trace.trace_finished_ns != resident.trace_finished_ns
        or close.gpu_uuid != gpu_uuids[0]
        or close.group_session_binding_sha256 != resident.group_session_binding_sha256
        or close.process_group_empty is not True
    ):
        raise ValueError("single-operator resident physical lifecycle differs")
    schedule = FormalServingRequestScheduleReceipt.from_dict(resident.request_schedule)
    if schedule.sha256 != resident.request_schedule_sha256:
        raise ValueError("single-operator resident request schedule differs")
    scored_request_count = sum(
        row.phase == "scored" for row in formal_serving_request_schedule_rows(schedule)
    )
    if scored_request_count < 1:
        raise ValueError("single-operator resident scored schedule is empty")
    after = CanonicalJsonProofBinding.bind(path)
    if after != before:
        raise RuntimeError(
            "single-operator resident manifest changed during projection"
        )

    physical = _UnifiedPhysicalExecution(
        physical_execution_id=close_binding.semantic_sha256,
        execution_kind="resident_session",
        source=close_binding,
        gpu_uuids=gpu_uuids,
        phase_edges_ns=(
            ("server_process_started_ns", close.server_process_started_ns),
            ("process_exited_ns", close.process_exited_ns),
            (
                "process_group_empty_checked_ns",
                close.process_group_empty_checked_ns,
            ),
            (
                "evidence_flush_completed_ns",
                close.evidence_flush_completed_ns,
            ),
        ),
    )
    trace_ns = trace.trace_finished_ns - trace.trace_started_ns
    observation = _UnifiedCellObservation(
        cell_id=resident.cell_id,
        actual_result=before,
        actual_result_sha256=resident.sha256,
        member_lifecycle=trace.trace_lifecycle,
        physical_execution_id=physical.physical_execution_id,
        topology=resident.topology,
        gang_gpu_count=len(gpu_uuids),
        provider_reserved_gpu_count=len(inventory.devices),
        scored_request_count=scored_request_count,
        projection_process_ns=trace_ns,
        projection_core_wall_ns=trace_ns,
        projection_evidence_tail_ns=0,
        projection_source="resident_member_trace",
    )
    return observation, physical


def _unified_observations_from_actual_results(
    *,
    repository_root: str | Path,
    actual_result_paths: tuple[str | Path, ...],
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
) -> tuple[
    tuple[_UnifiedCellObservation, ...],
    tuple[_UnifiedPhysicalExecution, ...],
]:
    if (
        type(actual_result_paths) is not tuple
        or not actual_result_paths
        or any(not isinstance(value, (str, Path)) for value in actual_result_paths)
    ):
        raise TypeError("single-operator unified GPU hours require actual-result paths")
    cells: list[_UnifiedCellObservation] = []
    physical: list[_UnifiedPhysicalExecution] = []
    for value in actual_result_paths:
        path = Path(value)
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise ValueError(
                "single-operator unified actual-result path must be absolute"
            )
        root_binding = CanonicalJsonProofBinding.bind(path)
        root = root_binding.reopen()
        if root.get("kind") == "formal_single_operator_resident_run_manifest":
            cell, execution = _resident_unified_observation(
                repository_root=repository_root,
                resident_manifest_path=path,
                materialization=materialization,
                inventory=inventory,
            )
        elif root.get("schema") in {
            "formal_single_operator_v1",
            "formal_single_operator_v2",
        }:
            cell, execution = _fresh_unified_observation(
                repository_root=repository_root,
                run_manifest_path=path,
                materialization=materialization,
                inventory=inventory,
            )
        else:
            raise ValueError("single-operator unified actual-result kind differs")
        if cell.actual_result != root_binding:
            raise ValueError("single-operator unified actual-result root differs")
        cells.append(cell)
        physical.append(execution)
    ordered_cells = tuple(sorted(cells, key=lambda row: row.cell_id))
    if len({row.cell_id for row in ordered_cells}) != len(ordered_cells) or len(
        {row.actual_result.absolute_path for row in ordered_cells}
    ) != len(ordered_cells):
        raise ValueError("single-operator unified actual results repeat cell/path")
    executions = _deduplicate_physical_executions(tuple(physical))
    if {row.physical_execution_id for row in ordered_cells} != {
        row.physical_execution_id for row in executions
    }:
        raise ValueError("single-operator unified physical mapping differs")
    return ordered_cells, executions


def _unified_source_value(
    *,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    cells: tuple[_UnifiedCellObservation, ...],
    physical: tuple[_UnifiedPhysicalExecution, ...],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "formal_single_operator_serving_gpu_hour_source",
        "protocol_sha256": (
            FORMAL_SINGLE_OPERATOR_SERVING_GPU_HOUR_SOURCE_PROTOCOL_SHA256
        ),
        "stage": materialization.stage,
        "materialization_receipt_sha256": materialization.sha256,
        "materialization_rule": materialization.materialization_rule,
        "protocol_lock_sha256": materialization.protocol_lock_sha256,
        "inventory_sha256": inventory.sha256,
        "hardware_envelope_sha256": _hardware_envelope_sha256(inventory),
        "actual_results": [row.actual_result.to_dict() for row in cells],
        "cell_observations": [row.to_source_row() for row in cells],
        "physical_executions": [row.to_source_row() for row in physical],
        "physical_execution_count": len(physical),
        "schedule": "physical_execution_deduplicated_per_gpu_interval_union",
    }


def _projection_observations_from_unified(
    cells: tuple[_UnifiedCellObservation, ...],
) -> tuple[_SingleOperatorLifecycleObservation, ...]:
    observations = []
    for row in cells:
        started_ns = 1
        process_exited_ns = started_ns + row.projection_process_ns
        process_group_empty_checked_ns = started_ns + row.projection_core_wall_ns
        evidence_flush_finished_ns = (
            process_group_empty_checked_ns + row.projection_evidence_tail_ns
        )
        observations.append(
            _SingleOperatorLifecycleObservation(
                cell_id=row.cell_id,
                run_manifest=row.actual_result,
                run_manifest_sha256=row.actual_result_sha256,
                lifecycle=row.member_lifecycle,
                topology=row.topology,
                gang_gpu_count=row.gang_gpu_count,
                provider_reserved_gpu_count=row.provider_reserved_gpu_count,
                scored_request_count=row.scored_request_count,
                phase_edges_ns=(
                    ("execution_started_ns", started_ns),
                    ("process_exited_ns", process_exited_ns),
                    (
                        "process_group_empty_checked_ns",
                        process_group_empty_checked_ns,
                    ),
                    ("evidence_flush_finished_ns", evidence_flush_finished_ns),
                ),
            )
        )
    return tuple(observations)


def _recorded_artifact_path(
    *,
    manifest_path: Path,
    manifest: object,
    name: str,
) -> tuple[Path, str, int]:
    from lightcone_spec.runtime.formal_single_operator import (
        FormalSingleOperatorRunManifest,
    )

    if type(manifest) is not FormalSingleOperatorRunManifest:
        raise TypeError("single-operator GPU-hour manifest type differs")
    rows = tuple(row for row in manifest.artifacts if row.name == name)
    if len(rows) != 1 or rows[0].status != "PRESENT":
        raise ValueError(f"single-operator GPU-hour manifest lacks {name}")
    row = rows[0]
    assert row.raw_sha256 is not None
    assert row.size_bytes is not None
    path = manifest_path.parent / row.relative_path
    if (
        manifest.run_directory != str(manifest_path.parent)
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=False) != path
    ):
        raise ValueError("single-operator GPU-hour artifact path differs")
    return path, row.raw_sha256, row.size_bytes


def _recorded_json_binding(
    *,
    manifest_path: Path,
    manifest: object,
    name: str,
) -> CanonicalJsonProofBinding:
    path, raw_sha256, size = _recorded_artifact_path(
        manifest_path=manifest_path,
        manifest=manifest,
        name=name,
    )
    binding = CanonicalJsonProofBinding.bind(path)
    if binding.raw_sha256 != raw_sha256 or binding.size != size:
        raise ValueError("single-operator GPU-hour JSON artifact changed")
    return binding


def _hardware_envelope_sha256(inventory: GpuInventory) -> str:
    if type(inventory) is not GpuInventory or len(inventory.devices) != 2:
        raise ValueError("single-operator GPU hours require an exact two-GPU inventory")
    hardware = {row.hardware_envelope_sha256 for row in inventory.devices}
    if len(hardware) != 1:
        raise ValueError("single-operator inventory mixes hardware envelopes")
    return next(iter(hardware))


def _single_operator_lifecycle_observation(
    *,
    repository_root: str | Path,
    run_manifest_path: str | Path,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
) -> _SingleOperatorLifecycleObservation:
    from lightcone_spec.runtime.formal_single_operator import (
        revalidate_formal_single_operator_run_manifest,
    )

    path = Path(run_manifest_path)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError("single-operator GPU-hour manifest path must be absolute")
    before = CanonicalJsonProofBinding.bind(path)
    manifest = revalidate_formal_single_operator_run_manifest(
        repository_root=repository_root,
        manifest_path=path,
    )
    after = CanonicalJsonProofBinding.bind(path)
    if before != after or manifest.completion_status != "COMPLETE":
        raise ValueError("single-operator GPU-hour run is not immutable COMPLETE")
    if (
        manifest.stage != materialization.stage
        or manifest.materialization_sha256 != materialization.sha256
        or manifest.materialization_protocol_lock_sha256
        != materialization.protocol_lock_sha256
        or manifest.inventory_sha256 != inventory.sha256
    ):
        raise ValueError("single-operator GPU-hour run has foreign stage lineage")
    cells = tuple(
        row for row in materialization.cells if row.cell_id == manifest.cell_id
    )
    if len(cells) != 1:
        raise ValueError("single-operator GPU-hour run names a foreign cell")
    cell = cells[0]
    inventory_uuids = {row.uuid for row in inventory.devices}
    manifest_gpu_uuids = tuple(row.uuid for row in manifest.gpu_environment)
    expected_gang_gpu_count = {
        "tp1_dp1": 1,
        "tp2_dp1": 2,
        "tp1_dp2": 2,
    }.get(manifest.topology)
    if (
        manifest.role != cell.method_role
        or manifest.backend != cell.backend
        or expected_gang_gpu_count is None
        or len(manifest_gpu_uuids) != expected_gang_gpu_count
        or not set(manifest_gpu_uuids).issubset(inventory_uuids)
    ):
        raise ValueError("single-operator GPU-hour manifest GPU/cell join differs")

    lifecycle_binding = _recorded_json_binding(
        manifest_path=path,
        manifest=manifest,
        name="lifecycle",
    )
    raw_lifecycle = lifecycle_binding.reopen()
    if (
        type(raw_lifecycle) is not dict
        or type(raw_lifecycle.get("phase_edges_ns")) is not dict
    ):
        raise TypeError("single-operator lifecycle cost source differs")
    raw_edges = raw_lifecycle["phase_edges_ns"]

    required_edges = (
        "execution_started_ns",
        "process_exited_ns",
        "process_group_empty_checked_ns",
        "evidence_flush_finished_ns",
    )
    if any(type(raw_edges.get(name)) is not int for name in required_edges):
        raise ValueError("single-operator lifecycle lacks integer cost boundaries")
    edges = {name: raw_edges[name] for name in required_edges}
    started, exited, empty, finished = (edges[name] for name in required_edges)
    if not (0 < started < exited <= empty <= finished):
        raise ValueError("single-operator lifecycle cost boundaries are not ordered")

    schedule_requests = manifest.request_schedule.get("requests")
    if type(schedule_requests) is not list or not schedule_requests:
        raise ValueError("single-operator manifest request schedule differs")
    phases = tuple(
        row.get("phase") if type(row) is dict else None for row in schedule_requests
    )
    if any(type(phase) is not str for phase in phases) or "scored" not in phases:
        raise ValueError("single-operator manifest scored schedule differs")
    return _SingleOperatorLifecycleObservation(
        cell_id=cell.cell_id,
        run_manifest=before,
        run_manifest_sha256=manifest.sha256,
        lifecycle=lifecycle_binding,
        topology=manifest.topology,
        gang_gpu_count=len(manifest_gpu_uuids),
        provider_reserved_gpu_count=len(inventory.devices),
        scored_request_count=sum(phase == "scored" for phase in phases),
        phase_edges_ns=tuple((name, edges[name]) for name in required_edges),
    )


def _single_operator_source_value(
    *,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    observations: tuple[_SingleOperatorLifecycleObservation, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "formal_single_operator_run_manifest_gpu_hour_source",
        "protocol_sha256": FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256,
        "stage": materialization.stage,
        "materialization_receipt_sha256": materialization.sha256,
        "materialization_rule": materialization.materialization_rule,
        "protocol_lock_sha256": materialization.protocol_lock_sha256,
        "inventory_sha256": inventory.sha256,
        "hardware_envelope_sha256": _hardware_envelope_sha256(inventory),
        "run_manifests": [row.run_manifest.to_dict() for row in observations],
        "observations": [row.to_source_row() for row in observations],
        "schedule": "trusted_single_operator_sequential_no_overlap",
    }


def _observations_from_run_manifests(
    *,
    repository_root: str | Path,
    run_manifest_paths: tuple[str | Path, ...],
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
) -> tuple[_SingleOperatorLifecycleObservation, ...]:
    if (
        type(run_manifest_paths) is not tuple
        or not run_manifest_paths
        or any(not isinstance(value, (str, Path)) for value in run_manifest_paths)
    ):
        raise TypeError("single-operator GPU hours require manifest paths")
    observations = tuple(
        _single_operator_lifecycle_observation(
            repository_root=repository_root,
            run_manifest_path=value,
            materialization=materialization,
            inventory=inventory,
        )
        for value in run_manifest_paths
    )
    if len({row.cell_id for row in observations}) != len(observations) or len(
        {row.run_manifest.absolute_path for row in observations}
    ) != len(observations):
        raise ValueError("single-operator GPU-hour manifests repeat cell/path")
    ordered = tuple(sorted(observations, key=lambda row: row.cell_id))
    chronological = tuple(
        sorted(
            ordered,
            key=lambda row: (
                dict(row.phase_edges_ns)["execution_started_ns"],
                row.cell_id,
            ),
        )
    )
    for prior, following in pairwise(chronological):
        if (
            dict(following.phase_edges_ns)["execution_started_ns"]
            < dict(prior.phase_edges_ns)["evidence_flush_finished_ns"]
        ):
            raise FormalSingleOperatorGpuHourBlocked(
                "overlapping_single_operator_wave_identity_unavailable"
            )
    return ordered


def _actual_cost_from_run_manifests(
    observations: tuple[_SingleOperatorLifecycleObservation, ...],
) -> FormalSingleOperatorGpuHourCost:
    compute_ns = 0
    provider_ns = 0
    wall_ns = 0
    evidence_ns = 0
    for row in observations:
        edges = dict(row.phase_edges_ns)
        process_ns = edges["process_exited_ns"] - edges["execution_started_ns"]
        core_wall_ns = (
            edges["process_group_empty_checked_ns"] - edges["execution_started_ns"]
        )
        evidence_tail_ns = (
            edges["evidence_flush_finished_ns"]
            - edges["process_group_empty_checked_ns"]
        )
        compute_ns += process_ns * row.gang_gpu_count
        provider_ns += core_wall_ns * row.provider_reserved_gpu_count
        wall_ns += core_wall_ns
        evidence_ns += evidence_tail_ns * row.provider_reserved_gpu_count
    retry_ns = (
        compute_ns * _gpu_hours.FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + _gpu_hours.FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // _gpu_hours.FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
    return FormalSingleOperatorGpuHourCost(
        category="actual_pilot",
        cell_count=len(observations),
        compute_gpu_ns=compute_ns,
        provider_base_reserved_gpu_ns=provider_ns,
        wall_ns=wall_ns,
        retry_reserve_gpu_ns=retry_ns,
        profile_reserve_gpu_ns=0,
        evidence_reserve_gpu_ns=evidence_ns,
    )


def _project_staged_run_manifest_cost(
    *,
    materialization: StageMaterializationReceipt,
    observations: tuple[_SingleOperatorLifecycleObservation, ...],
) -> tuple[FormalSingleOperatorGpuHourCost, tuple[tuple[object, ...], ...]]:
    by_id = {row.cell_id: row for row in observations}
    strata = _gpu_hours._staged_strata(
        materialization,
        completed_cell_ids=tuple(sorted(by_id)),
    )
    missing = tuple(
        sorted(
            row.minimum_pilot_cell_id
            for row in strata
            if row.minimum_pilot_cell_id is not None
        )
    )
    if missing:
        raise FormalSingleOperatorGpuHourBlocked(
            "scientific_strata_duration_unmeasured",
            minimum_pilot_cell_ids=missing,
        )
    compute_ns = 0
    provider_ns = 0
    wall_ns = 0
    evidence_ns = 0
    projected_count = 0
    mapping: list[tuple[object, ...]] = []
    for stratum in strata:
        pilots = tuple(by_id[cell_id] for cell_id in stratum.completed_cell_ids)
        gang_counts = {row.gang_gpu_count for row in pilots}
        provider_counts = {row.provider_reserved_gpu_count for row in pilots}
        if len(gang_counts) != 1 or provider_counts != {2}:
            raise ValueError("single-operator stratum GPU shape differs")
        gang_count = next(iter(gang_counts))
        process_ns = _gpu_hours._ceiling_mean(
            tuple(
                dict(row.phase_edges_ns)["process_exited_ns"]
                - dict(row.phase_edges_ns)["execution_started_ns"]
                for row in pilots
            )
        )
        core_wall_ns = _gpu_hours._ceiling_mean(
            tuple(
                dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                - dict(row.phase_edges_ns)["execution_started_ns"]
                for row in pilots
            )
        )
        evidence_tail_ns = _gpu_hours._ceiling_mean(
            tuple(
                dict(row.phase_edges_ns)["evidence_flush_finished_ns"]
                - dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                for row in pilots
            )
        )
        for cell_id in stratum.projected_cell_ids:
            projected_count += 1
            compute_ns += process_ns * gang_count
            provider_ns += core_wall_ns * 2
            wall_ns += core_wall_ns
            evidence_ns += evidence_tail_ns * 2
            mapping.append(
                (
                    cell_id,
                    stratum.stratum_sha256,
                    stratum.completed_cell_ids,
                    process_ns,
                    core_wall_ns,
                    evidence_tail_ns,
                    gang_count,
                )
            )
    retry_ns = (
        compute_ns * _gpu_hours.FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + _gpu_hours.FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // _gpu_hours.FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
    return (
        FormalSingleOperatorGpuHourCost(
            category="projected_remaining",
            cell_count=projected_count,
            compute_gpu_ns=compute_ns,
            provider_base_reserved_gpu_ns=provider_ns,
            wall_ns=wall_ns,
            retry_reserve_gpu_ns=retry_ns,
            profile_reserve_gpu_ns=0,
            evidence_reserve_gpu_ns=evidence_ns,
        ),
        tuple(mapping),
    )


def _project_downstream_run_manifest_cost(
    *,
    pilot_materialization: StageMaterializationReceipt,
    final_materialization: StageMaterializationReceipt,
    observations: tuple[_SingleOperatorLifecycleObservation, ...],
) -> tuple[FormalSingleOperatorGpuHourCost, tuple[tuple[object, ...], ...]]:
    pilot_cells = {row.cell_id: row for row in pilot_materialization.cells}
    by_id = {row.cell_id: row for row in observations}
    if set(pilot_cells) != set(by_id):
        raise ValueError("single-operator excluded-pilot coverage is not exact")
    pilot_by_stratum: dict[
        str, list[tuple[int, _SingleOperatorLifecycleObservation]]
    ] = {}
    for cell in pilot_materialization.cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if (
            type(block) is not int
            or block not in range(4)
            or dimensions.get("block_phase") != "excluded_pilot"
        ):
            raise ValueError("single-operator source is not four excluded pilots")
        pilot_by_stratum.setdefault(_gpu_hours._projection_stratum(cell), []).append(
            (block, by_id[cell.cell_id])
        )
    if any(
        tuple(sorted(block for block, _row in rows)) != (0, 1, 2, 3)
        for rows in pilot_by_stratum.values()
    ):
        raise ValueError("single-operator stratum lacks four excluded pilots")

    projected_cells = tuple(
        row
        for row in final_materialization.cells
        if type(dict(row.dimensions).get("block")) is int
    )
    if not projected_cells:
        raise ValueError("single-operator final materialization has no projected rows")
    compute_ns = 0
    provider_ns = 0
    wall_ns = 0
    evidence_ns = 0
    mapping: list[tuple[object, ...]] = []
    for cell in projected_cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if (
            type(block) is not int
            or block not in range(4, 24)
            or dimensions.get("block_phase") != "final"
        ):
            raise ValueError("single-operator projected row is outside final prefix")
        stratum_sha256 = _gpu_hours._projection_stratum(cell)
        pilots = tuple(
            row
            for _pilot_block, row in sorted(pilot_by_stratum.get(stratum_sha256, ()))
        )
        if len(pilots) != 4:
            raise ValueError("single-operator final row lacks exact four pilots")
        expected_gang = 2 if dimensions.get("topology") in {"tp2_dp1", "tp1_dp2"} else 1
        if {row.gang_gpu_count for row in pilots} != {expected_gang} or {
            row.provider_reserved_gpu_count for row in pilots
        } != {2}:
            raise ValueError("single-operator final topology differs from pilots")
        process_ns = _gpu_hours._ceiling_mean(
            tuple(
                dict(row.phase_edges_ns)["process_exited_ns"]
                - dict(row.phase_edges_ns)["execution_started_ns"]
                for row in pilots
            )
        )
        core_wall_ns = _gpu_hours._ceiling_mean(
            tuple(
                dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                - dict(row.phase_edges_ns)["execution_started_ns"]
                for row in pilots
            )
        )
        evidence_tail_ns = _gpu_hours._ceiling_mean(
            tuple(
                dict(row.phase_edges_ns)["evidence_flush_finished_ns"]
                - dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                for row in pilots
            )
        )
        required_requests = dimensions.get(
            "p99_extension_minimum_completions",
            dimensions.get("p99_minimum_completions"),
        )
        if required_requests is not None:
            if type(required_requests) is not int or required_requests != 10_000:
                raise ValueError("single-operator p99 completion target differs")
            offered_requests = dimensions.get(
                "p99_extension_offered_requests",
                required_requests,
            )
            if type(offered_requests) is not int or offered_requests not in {
                required_requests,
                11_000,
            }:
                raise ValueError("single-operator p99 extension offer target differs")
            denominator = min(row.scored_request_count for row in pilots)
            process_ns = _gpu_hours._scaled_ceiling(
                process_ns,
                numerator=offered_requests,
                denominator=denominator,
            )
            core_wall_ns = _gpu_hours._scaled_ceiling(
                core_wall_ns,
                numerator=offered_requests,
                denominator=denominator,
            )
            evidence_tail_ns = _gpu_hours._scaled_ceiling(
                evidence_tail_ns,
                numerator=offered_requests,
                denominator=denominator,
            )
        compute_ns += process_ns * expected_gang
        provider_ns += core_wall_ns * 2
        wall_ns += core_wall_ns
        evidence_ns += evidence_tail_ns * 2
        mapping.append(
            (
                cell.cell_id,
                stratum_sha256,
                tuple(row.cell_id for row in pilots),
                process_ns,
                core_wall_ns,
                evidence_tail_ns,
            )
        )
    retry_ns = (
        compute_ns * _gpu_hours.FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + _gpu_hours.FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // _gpu_hours.FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
    return (
        FormalSingleOperatorGpuHourCost(
            category="projected_remaining",
            cell_count=len(projected_cells),
            compute_gpu_ns=compute_ns,
            provider_base_reserved_gpu_ns=provider_ns,
            wall_ns=wall_ns,
            retry_reserve_gpu_ns=retry_ns,
            profile_reserve_gpu_ns=0,
            evidence_reserve_gpu_ns=evidence_ns,
        ),
        tuple(mapping),
    )


def revalidate_formal_single_operator_run_gpu_hour_source(
    *,
    repository_root: str | Path,
    source_manifest_path: str | Path,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
) -> CanonicalJsonProofBinding:
    """Deep-rebuild one manifest source without signatures or caller durations."""

    binding = CanonicalJsonProofBinding.bind(source_manifest_path)
    raw = _strict(
        "single-operator run GPU-hour source",
        binding.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "stage",
            "materialization_receipt_sha256",
            "materialization_rule",
            "protocol_lock_sha256",
            "inventory_sha256",
            "hardware_envelope_sha256",
            "run_manifests",
            "observations",
            "schedule",
        },
    )
    manifest_values = raw.get("run_manifests")
    if type(manifest_values) is not list or not manifest_values:
        raise ValueError("single-operator run GPU-hour source has no manifests")
    manifest_bindings = tuple(
        CanonicalJsonProofBinding.from_dict(value) for value in manifest_values
    )
    observations = _observations_from_run_manifests(
        repository_root=repository_root,
        run_manifest_paths=tuple(row.absolute_path for row in manifest_bindings),
        materialization=materialization,
        inventory=inventory,
    )
    expected = _single_operator_source_value(
        materialization=materialization,
        inventory=inventory,
        observations=observations,
    )
    if (
        raw != expected
        or CanonicalJsonProofBinding.bind(binding.absolute_path) != binding
    ):
        raise ValueError("single-operator run GPU-hour source replay differs")
    return binding


def revalidate_formal_single_operator_serving_gpu_hour_source(
    *,
    repository_root: str | Path,
    source_manifest_path: str | Path,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
) -> CanonicalJsonProofBinding:
    """Deep-rebuild a mixed fresh/resident source and its physical intervals."""

    binding = CanonicalJsonProofBinding.bind(source_manifest_path)
    raw = _strict(
        "single-operator serving GPU-hour source",
        binding.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "stage",
            "materialization_receipt_sha256",
            "materialization_rule",
            "protocol_lock_sha256",
            "inventory_sha256",
            "hardware_envelope_sha256",
            "actual_results",
            "cell_observations",
            "physical_executions",
            "physical_execution_count",
            "schedule",
        },
    )
    actual_values = raw.get("actual_results")
    if type(actual_values) is not list or not actual_values:
        raise ValueError(
            "single-operator serving GPU-hour source has no actual results"
        )
    actual_bindings = tuple(
        CanonicalJsonProofBinding.from_dict(value) for value in actual_values
    )
    cells, physical = _unified_observations_from_actual_results(
        repository_root=repository_root,
        actual_result_paths=tuple(row.absolute_path for row in actual_bindings),
        materialization=materialization,
        inventory=inventory,
    )
    _actual_cost_from_unified_observations(
        cells,
        physical,
        inventory_gpu_count=len(inventory.devices),
    )
    expected = _unified_source_value(
        materialization=materialization,
        inventory=inventory,
        cells=cells,
        physical=physical,
    )
    if (
        raw != expected
        or CanonicalJsonProofBinding.bind(binding.absolute_path) != binding
    ):
        raise ValueError("single-operator serving GPU-hour source replay differs")
    return binding


def derive_formal_single_operator_pre_pilot_gpu_hours(
    materialization: StageMaterializationReceipt,
) -> FormalSingleOperatorPrePilotGpuHours:
    """Return counts only; this function accepts no duration or hour scalar."""

    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("single-operator pre-pilot planning requires materialization")
    if materialization.expected_cell_count == 0:
        if materialization.stage != "E0":
            raise FormalSingleOperatorGpuHourBlocked("no_gpu_cells_materialized")
        return FormalSingleOperatorPrePilotGpuHours(
            schema_version=1,
            kind="formal_single_operator_pre_pilot_gpu_hours",
            protocol_sha256=FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256,
            stage="E0",
            protocol_lock_sha256=materialization.protocol_lock_sha256,
            materialization_receipt_sha256=materialization.sha256,
            materialization_rule=materialization.materialization_rule,
            fixed_cell_count=0,
            projection_stratum_count=0,
            minimum_pilot_cell_ids=(),
            duration_status="duration_unmeasured",
        )
    if materialization.stage in _EARLY_STAGED_STAGES:
        strata = _gpu_hours._staged_strata(
            materialization,
            completed_cell_ids=(),
        )
        minimum = tuple(
            sorted(
                row.minimum_pilot_cell_id
                for row in strata
                if row.minimum_pilot_cell_id is not None
            )
        )
    elif (
        materialization.stage == "preflight"
        or _DOWNSTREAM_PILOT_STAGE_BY_RULE.get(materialization.materialization_rule)
        == materialization.stage
    ):
        minimum = tuple(cell.cell_id for cell in materialization.cells)
        strata = minimum
    else:
        raise FormalSingleOperatorGpuHourBlocked(
            "pre_pilot_plan_requires_current_pilot_materialization"
        )
    return FormalSingleOperatorPrePilotGpuHours(
        schema_version=1,
        kind="formal_single_operator_pre_pilot_gpu_hours",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256,
        stage=materialization.stage,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        materialization_rule=materialization.materialization_rule,
        fixed_cell_count=materialization.expected_cell_count,
        projection_stratum_count=len(strata),
        minimum_pilot_cell_ids=minimum,
        duration_status="duration_unmeasured",
    )


def _revalidate_pilot_source(
    *,
    source_manifest_path: str | Path,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    now_ns: int,
) -> tuple[CanonicalJsonProofBinding, LifecycleGpuHourSourceManifest]:
    binding = CanonicalJsonProofBinding.bind(source_manifest_path)
    unverified = LifecycleGpuHourSourceManifest.from_dict(binding.reopen())
    envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        signed_pilot_receipt_sha256=unverified.sha256,
        schedule_sha256=unverified.schedule_sha256,
        estimate=_gpu_hours._estimate(unverified),
    )
    source = _gpu_hours.revalidate_persisted_stage_gpu_hour_source_manifest(
        binding.absolute_path,
        envelope=envelope,
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        inventory=inventory,
        now_ns=now_ns,
        expected_cell_ids=tuple(
            sorted(row.materialized_cell_id for row in unverified.observations)
        ),
    )
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("single-operator pilot lifecycle source changed")
    return binding, source


def derive_formal_single_operator_post_pilot_gpu_hours(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    pilot_materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    pilot_lifecycle_source_manifest_path: str | Path,
    now_ns: int,
    final_materialization: StageMaterializationReceipt | None = None,
    e5_one_shot_source_manifest_path: str | Path | None = None,
) -> FormalSingleOperatorPostPilotGpuHours:
    """Derive actual/projected cost from lifecycle evidence, never scalars."""

    pilot_binding, pilot_source = _revalidate_pilot_source(
        source_manifest_path=pilot_lifecycle_source_manifest_path,
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=pilot_materialization,
        inventory=inventory,
        now_ns=now_ns,
    )
    if final_materialization is None:
        if pilot_materialization.stage not in _EARLY_STAGED_STAGES:
            raise FormalSingleOperatorGpuHourBlocked(
                "downstream_projection_requires_final_materialization"
            )
        staged = _gpu_hours._derive_staged_prospective_gpu_hour_source(
            protocol_lock=protocol_lock,
            runtime_authority_member_sha256=formal_runtime_authority_manifest.member(
                "gpu_hour_budget_reducer"
            ).sha256,
            materialization=pilot_materialization,
            inventory=inventory,
            completed_source_binding=pilot_binding,
            completed_source=pilot_source,
        )
        if staged.status != "READY" or staged.projected_remaining is None:
            raise FormalSingleOperatorGpuHourBlocked(
                "scientific_strata_duration_unmeasured",
                minimum_pilot_cell_ids=staged.minimum_pilot_cell_ids,
            )
        actual = _cost("actual_pilot", staged.actual_completed)
        projected = _cost("projected_remaining", staged.projected_remaining)
        one_shot = None
        one_binding = None
        mapping_sha256 = staged.mapping_sha256
        final_sha256 = None
    else:
        if (
            type(final_materialization) is not StageMaterializationReceipt
            or final_materialization.stage != pilot_materialization.stage
            or final_materialization.protocol_lock_sha256 != protocol_lock.sha256
            or pilot_materialization.protocol_lock_sha256 != protocol_lock.sha256
        ):
            raise ValueError("single-operator final/pilot materialization differs")
        stage = pilot_materialization.stage
        if stage == "E6":
            raise FormalSingleOperatorGpuHourBlocked(
                "e6_model_preflight_lifecycle_cost_unavailable"
            )
        if stage not in _DOWNSTREAM_PROJECTION_STAGES:
            raise ValueError("single-operator downstream GPU-hour stage differs")
        actual_source = _gpu_hours._cost_from_actual_observations(
            category="actual_tuning",
            observations=pilot_source.observations,
            inventory_gpu_count=2,
        )
        projected_source, projection_sha256 = _gpu_hours._project_final_cost(
            pilot_materialization=pilot_materialization,
            pilot_source=pilot_source,
            final_materialization=final_materialization,
        )
        actual = _cost("actual_pilot", actual_source)
        projected = _cost("projected_remaining", projected_source)
        one_shot = None
        one_binding = None
        if stage == "E5":
            if e5_one_shot_source_manifest_path is None:
                raise FormalSingleOperatorGpuHourBlocked(
                    "e5_264_actual_lifecycle_source_missing"
                )
            one_binding = CanonicalJsonProofBinding.bind(
                e5_one_shot_source_manifest_path
            )
            if one_binding.absolute_path == pilot_binding.absolute_path:
                raise ValueError("E5 pilot lifecycle cannot double as one-shot source")
            one_source = (
                _gpu_hours.revalidate_persisted_e5_failure_gpu_hour_source_manifest(
                    one_binding.absolute_path,
                    protocol_lock=protocol_lock,
                    formal_runtime_authority_manifest=(
                        formal_runtime_authority_manifest
                    ),
                    materialization=final_materialization,
                    inventory=inventory,
                    now_ns=now_ns,
                )
            )
            if (
                CanonicalJsonProofBinding.bind(one_binding.absolute_path) != one_binding
                or one_source.cost.cell_count != 264
                or one_source.hardware_envelope_sha256
                != pilot_source.hardware_envelope_sha256
            ):
                raise ValueError("E5 one-shot lifecycle cost/hardware differs")
            one_shot = _cost("actual_one_shot", one_source.cost)
        elif e5_one_shot_source_manifest_path is not None:
            raise ValueError("only E5 accepts a one-shot lifecycle source")
        mapping_sha256 = content_sha256(
            {
                "schema_version": 1,
                "kind": "single_operator_post_pilot_mapping",
                "pilot_source_sha256": pilot_source.sha256,
                "projection_sha256": projection_sha256,
                "one_shot_source_sha256": (
                    None if one_binding is None else one_binding.semantic_sha256
                ),
            }
        )
        final_sha256 = final_materialization.sha256
    total = _sum_costs(
        actual,
        projected,
        *((one_shot,) if one_shot is not None else ()),
    )
    return FormalSingleOperatorPostPilotGpuHours(
        schema_version=1,
        kind="formal_single_operator_post_pilot_gpu_hours",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256,
        stage=pilot_materialization.stage,
        protocol_lock_sha256=protocol_lock.sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_materialization_rule=pilot_materialization.materialization_rule,
        final_materialization_receipt_sha256=final_sha256,
        final_materialization_rule=(
            None
            if final_materialization is None
            else final_materialization.materialization_rule
        ),
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        duration_status="measured_and_projected",
        pilot_lifecycle_source=pilot_binding,
        one_shot_lifecycle_source=one_binding,
        mapping_sha256=mapping_sha256,
        actual_pilot=actual,
        projected_remaining=projected,
        actual_one_shot=one_shot,
        total=total,
    )


def derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests(
    *,
    repository_root: str | Path,
    pilot_materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    pilot_run_manifest_paths: tuple[str | Path, ...],
    source_manifest_output_path: str | Path,
    final_materialization: StageMaterializationReceipt | None = None,
) -> FormalSingleOperatorPostPilotGpuHours:
    """Charge root-revalidated single-operator runs, never caller durations.

    Ordinary serving manifests deliberately cannot stand in for E5's dedicated
    failure-run union, E4 profiler evidence, or E6 model-preflight evidence.
    Those branches stay BLOCKED until their single-operator manifests record
    the corresponding physical lifecycle rather than a serving lifecycle.
    """

    if type(pilot_materialization) is not StageMaterializationReceipt:
        raise TypeError("single-operator manifest GPU hours require materialization")
    if pilot_materialization.stage == "preflight":
        raise FormalSingleOperatorGpuHourBlocked(
            "preflight_requires_preflight_lifecycle_manifest"
        )
    if (
        pilot_materialization.stage == "E4"
        and pilot_materialization.materialization_rule
        == "three_profiler_only_rows_separate_from_headline"
    ):
        raise FormalSingleOperatorGpuHourBlocked(
            "e4_profiler_requires_dedicated_profiler_lifecycle_manifest"
        )
    if pilot_materialization.stage == "E6":
        raise FormalSingleOperatorGpuHourBlocked(
            "e6_model_preflight_lifecycle_cost_unavailable"
        )
    if pilot_materialization.stage == "E5":
        raise FormalSingleOperatorGpuHourBlocked(
            "e5_dedicated_failure_run_manifest_union_required"
        )
    if pilot_materialization.expected_cell_count == 0:
        raise FormalSingleOperatorGpuHourBlocked("no_measured_gpu_lifecycle")
    hardware_sha256 = _hardware_envelope_sha256(inventory)

    if final_materialization is None:
        if pilot_materialization.stage not in _EARLY_STAGED_STAGES:
            raise FormalSingleOperatorGpuHourBlocked(
                "downstream_projection_requires_final_materialization"
            )
    elif (
        type(final_materialization) is not StageMaterializationReceipt
        or final_materialization.stage != pilot_materialization.stage
        or final_materialization.protocol_lock_sha256
        != pilot_materialization.protocol_lock_sha256
    ):
        raise ValueError("single-operator final/pilot materialization differs")
    elif pilot_materialization.stage not in {"E3b", "E0"}:
        raise ValueError("single-operator downstream manifest stage differs")

    observations = _observations_from_run_manifests(
        repository_root=repository_root,
        run_manifest_paths=pilot_run_manifest_paths,
        materialization=pilot_materialization,
        inventory=inventory,
    )
    actual = _actual_cost_from_run_manifests(observations)
    if final_materialization is None:
        projected, mapping_rows = _project_staged_run_manifest_cost(
            materialization=pilot_materialization,
            observations=observations,
        )
        final_sha256 = None
        final_rule = None
    else:
        projected, mapping_rows = _project_downstream_run_manifest_cost(
            pilot_materialization=pilot_materialization,
            final_materialization=final_materialization,
            observations=observations,
        )
        final_sha256 = final_materialization.sha256
        final_rule = final_materialization.materialization_rule

    source_value = _single_operator_source_value(
        materialization=pilot_materialization,
        inventory=inventory,
        observations=observations,
    )
    publish_canonical_json_no_replace(source_manifest_output_path, source_value)
    source_binding = revalidate_formal_single_operator_run_gpu_hour_source(
        repository_root=repository_root,
        source_manifest_path=source_manifest_output_path,
        materialization=pilot_materialization,
        inventory=inventory,
    )
    mapping_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "single_operator_run_manifest_gpu_hour_mapping",
            "source_raw_sha256": source_binding.raw_sha256,
            "source_semantic_sha256": source_binding.semantic_sha256,
            "pilot_materialization_receipt_sha256": pilot_materialization.sha256,
            "final_materialization_receipt_sha256": final_sha256,
            "rows": mapping_rows,
            "schedule": "trusted_single_operator_sequential_no_overlap",
        }
    )
    total = _sum_costs(actual, projected)
    return FormalSingleOperatorPostPilotGpuHours(
        schema_version=1,
        kind="formal_single_operator_post_pilot_gpu_hours",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256,
        stage=pilot_materialization.stage,
        protocol_lock_sha256=pilot_materialization.protocol_lock_sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_materialization_rule=pilot_materialization.materialization_rule,
        final_materialization_receipt_sha256=final_sha256,
        final_materialization_rule=final_rule,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=hardware_sha256,
        duration_status="measured_and_projected",
        pilot_lifecycle_source=source_binding,
        one_shot_lifecycle_source=None,
        mapping_sha256=mapping_sha256,
        actual_pilot=actual,
        projected_remaining=projected,
        actual_one_shot=None,
        total=total,
    )


def derive_formal_single_operator_post_pilot_gpu_hours_from_serving_actuals(
    *,
    repository_root: str | Path,
    pilot_materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    pilot_actual_result_paths: tuple[str | Path, ...],
    source_manifest_output_path: str | Path,
    final_materialization: StageMaterializationReceipt | None = None,
) -> FormalSingleOperatorPostPilotGpuHours:
    """Charge mixed fresh/resident serving evidence once per physical execution.

    Actual intervals are derived only from root-revalidated result manifests.
    Resident members that share one session receipt therefore contribute one
    physical process interval, while their independent trace durations remain
    available to the registered same-stratum projection.
    """

    if type(pilot_materialization) is not StageMaterializationReceipt:
        raise TypeError("single-operator serving GPU hours require materialization")
    if pilot_materialization.stage == "preflight":
        raise FormalSingleOperatorGpuHourBlocked(
            "preflight_requires_preflight_lifecycle_manifest"
        )
    if (
        pilot_materialization.stage == "E4"
        and pilot_materialization.materialization_rule
        == "three_profiler_only_rows_separate_from_headline"
    ):
        raise FormalSingleOperatorGpuHourBlocked(
            "e4_profiler_requires_dedicated_profiler_lifecycle_manifest"
        )
    if pilot_materialization.stage == "E6":
        raise FormalSingleOperatorGpuHourBlocked(
            "e6_model_preflight_lifecycle_cost_unavailable"
        )
    if pilot_materialization.stage == "E5":
        raise FormalSingleOperatorGpuHourBlocked(
            "e5_dedicated_failure_run_manifest_union_required"
        )
    if pilot_materialization.expected_cell_count == 0:
        raise FormalSingleOperatorGpuHourBlocked("no_measured_gpu_lifecycle")
    hardware_sha256 = _hardware_envelope_sha256(inventory)

    if final_materialization is None:
        if pilot_materialization.stage not in _EARLY_STAGED_STAGES:
            raise FormalSingleOperatorGpuHourBlocked(
                "downstream_projection_requires_final_materialization"
            )
    elif (
        type(final_materialization) is not StageMaterializationReceipt
        or final_materialization.stage != pilot_materialization.stage
        or final_materialization.protocol_lock_sha256
        != pilot_materialization.protocol_lock_sha256
    ):
        raise ValueError("single-operator final/pilot materialization differs")
    elif pilot_materialization.stage not in {"E3b", "E0"}:
        raise ValueError("single-operator downstream serving stage differs")

    cells, physical = _unified_observations_from_actual_results(
        repository_root=repository_root,
        actual_result_paths=pilot_actual_result_paths,
        materialization=pilot_materialization,
        inventory=inventory,
    )
    actual = _actual_cost_from_unified_observations(
        cells,
        physical,
        inventory_gpu_count=len(inventory.devices),
    )
    projection_observations = _projection_observations_from_unified(cells)
    if final_materialization is None:
        projected, mapping_rows = _project_staged_run_manifest_cost(
            materialization=pilot_materialization,
            observations=projection_observations,
        )
        final_sha256 = None
        final_rule = None
    else:
        projected, mapping_rows = _project_downstream_run_manifest_cost(
            pilot_materialization=pilot_materialization,
            final_materialization=final_materialization,
            observations=projection_observations,
        )
        final_sha256 = final_materialization.sha256
        final_rule = final_materialization.materialization_rule

    source_value = _unified_source_value(
        materialization=pilot_materialization,
        inventory=inventory,
        cells=cells,
        physical=physical,
    )
    publish_canonical_json_no_replace(source_manifest_output_path, source_value)
    source_binding = revalidate_formal_single_operator_serving_gpu_hour_source(
        repository_root=repository_root,
        source_manifest_path=source_manifest_output_path,
        materialization=pilot_materialization,
        inventory=inventory,
    )
    mapping_sha256 = content_sha256(
        {
            "schema_version": 2,
            "kind": "single_operator_serving_gpu_hour_mapping",
            "source_raw_sha256": source_binding.raw_sha256,
            "source_semantic_sha256": source_binding.semantic_sha256,
            "pilot_materialization_receipt_sha256": pilot_materialization.sha256,
            "final_materialization_receipt_sha256": final_sha256,
            "rows": mapping_rows,
            "schedule": "physical_execution_deduplicated_per_gpu_interval_union",
        }
    )
    total = _sum_costs(actual, projected)
    return FormalSingleOperatorPostPilotGpuHours(
        schema_version=1,
        kind="formal_single_operator_post_pilot_gpu_hours",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256,
        stage=pilot_materialization.stage,
        protocol_lock_sha256=pilot_materialization.protocol_lock_sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_materialization_rule=pilot_materialization.materialization_rule,
        final_materialization_receipt_sha256=final_sha256,
        final_materialization_rule=final_rule,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=hardware_sha256,
        duration_status="measured_and_projected",
        pilot_lifecycle_source=source_binding,
        one_shot_lifecycle_source=None,
        mapping_sha256=mapping_sha256,
        actual_pilot=actual,
        projected_remaining=projected,
        actual_one_shot=None,
        total=total,
    )


def publish_formal_single_operator_gpu_hours(
    output: FormalSingleOperatorGpuHourOutput,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Persist one count/report output atomically without a signing ceremony."""

    if type(output) not in {
        FormalSingleOperatorPrePilotGpuHours,
        FormalSingleOperatorPostPilotGpuHours,
    }:
        raise TypeError("single-operator GPU-hour publisher requires exact output")
    publish_canonical_json_no_replace(output_path, output.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


def load_formal_single_operator_gpu_hours(
    output_path: str | Path,
) -> FormalSingleOperatorGpuHourOutput:
    binding = CanonicalJsonProofBinding.bind(output_path)
    value = binding.reopen()
    if type(value) is not dict:
        raise TypeError("single-operator GPU-hour output must be an object")
    kind = value.get("kind")
    if kind == "formal_single_operator_pre_pilot_gpu_hours":
        return FormalSingleOperatorPrePilotGpuHours.from_dict(value)
    if kind == "formal_single_operator_post_pilot_gpu_hours":
        return FormalSingleOperatorPostPilotGpuHours.from_dict(value)
    raise ValueError("single-operator GPU-hour output kind differs")


__all__ = (
    "FORMAL_SINGLE_OPERATOR_GPU_HOUR_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_SERVING_GPU_HOUR_SOURCE_PROTOCOL_SHA256",
    "FormalSingleOperatorGpuHourBlocked",
    "FormalSingleOperatorGpuHourCost",
    "FormalSingleOperatorPostPilotGpuHours",
    "FormalSingleOperatorPrePilotGpuHours",
    "derive_formal_single_operator_post_pilot_gpu_hours",
    "derive_formal_single_operator_post_pilot_gpu_hours_from_run_manifests",
    "derive_formal_single_operator_post_pilot_gpu_hours_from_serving_actuals",
    "derive_formal_single_operator_pre_pilot_gpu_hours",
    "load_formal_single_operator_gpu_hours",
    "publish_formal_single_operator_gpu_hours",
    "revalidate_formal_single_operator_run_gpu_hour_source",
    "revalidate_formal_single_operator_serving_gpu_hour_source",
)
