"""Stage-specific disk gate with an explicit 100 GB fail-closed fallback.

The signed path separates durable and transient capacity.  Evidence from every
attempt is retained, while model staging and compile overlays are counted only
for assignments that the immutable dispatch schedule can run concurrently.
This avoids both unsafe single-cell sizing and the opposite error of charging
one shared transient workspace once for every sequential cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, Self

from lightcone_spec.experiments.capacity_authority import (
    CapacityAuthorityResult,
    CapacityAuthorityUnavailableError,
    UnsignedCapacitySourceReplay,
    replay_unsigned_capacity_source_manifest,
    revalidate_capacity_authority_binding,
)
from lightcone_spec.experiments.planning import (
    BudgetInventoryIdentity,
    CapacityAuthorityBinding,
    CapacityEnvelope,
    CapacityRawJsonBinding,
)
from lightcone_spec.experiments.registry import (
    INDUSTRIAL_EXPERIMENT_ORDER,
    ExperimentRegistry,
    content_sha256,
)

LEGACY_MINIMUM_FREE_BYTES = 100_000_000_000
STAGE_CAPACITY_SAFETY_MARGIN_BPS = 2_000
STAGE_CAPACITY_MINIMUM_SAFETY_MARGIN_BYTES = 1_000_000_000

STAGE_CAPACITY_GATE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "industrial_stage_capacity_gate",
        "preferred_authority": (
            "fresh_path_bound_raw_capacity_replay_exact_stage_schedule_then_"
            "dynamic_root_authorized_capacity_control"
        ),
        "high_water_inputs": {
            "retained_evidence": "sum(maximum_evidence_bytes*(retry_allowance+1))",
            "transient": (
                "max_dispatch_wave(sum(model_staging_bytes+compile_overlay_bytes))"
            ),
            "schedule": "dispatch_and_budget_identity_plus_wave_topology",
        },
        "safety_margin_basis_points": STAGE_CAPACITY_SAFETY_MARGIN_BPS,
        "minimum_safety_margin_bytes": (STAGE_CAPACITY_MINIMUM_SAFETY_MARGIN_BYTES),
        "fallback": {
            "minimum_free_bytes": LEGACY_MINIMUM_FREE_BYTES,
            "use_only_when_signed_authority_absent_or_unavailable": True,
        },
        "tamper_never_falls_back": True,
        "caller_authored_observed_or_high_water_bytes": "forbidden",
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_nonnegative(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _strict_object(
    label: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object with string keys")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return value


@dataclass(frozen=True)
class StageCapacityRetryBinding:
    """One budget-owned retry allowance used by durable evidence sizing."""

    cell_id: str
    experiment_budget_sha256: str
    retry_allowance: int

    def __post_init__(self) -> None:
        _require_sha256("stage capacity retry cell", self.cell_id)
        _require_sha256("stage capacity retry budget", self.experiment_budget_sha256)
        _require_nonnegative("stage capacity retry allowance", self.retry_allowance)


@dataclass(frozen=True)
class StageCapacityWaveBinding:
    """One immutable concurrency wave and its physical topology identity."""

    wave_index: int
    cell_ids: tuple[str, ...]
    topology_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative("stage capacity wave index", self.wave_index)
        if not self.cell_ids or self.cell_ids != tuple(sorted(set(self.cell_ids))):
            raise ValueError("stage capacity wave cells must be sorted and unique")
        for cell_id in self.cell_ids:
            _require_sha256("stage capacity wave cell", cell_id)
        _require_sha256("stage capacity wave topology", self.topology_sha256)


@dataclass(frozen=True)
class StageCapacitySchedule:
    """Source-derived dispatch/budget schedule used by the capacity reducer.

    Construction should use :func:`bind_stage_capacity_schedule`; downstream
    acceptance additionally signs the resulting gate with the release control
    root, so a caller-authored schedule is never authority by itself.
    """

    schema_version: int
    kind: Literal["industrial_stage_capacity_schedule"]
    registry_sha256: str
    experiment: str
    activated_cell_ids: tuple[str, ...]
    gpu_inventory_sha256: str
    dispatch_plan_sha256: str
    budget_plan_sha256: str
    capacity_envelope_sha256: str
    capacity_authority_sha256: str
    waves: tuple[StageCapacityWaveBinding, ...]
    retries: tuple[StageCapacityRetryBinding, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "industrial_stage_capacity_schedule"
        ):
            raise ValueError("stage capacity schedule schema is unsupported")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("GPU inventory", self.gpu_inventory_sha256),
            ("dispatch plan", self.dispatch_plan_sha256),
            ("budget plan", self.budget_plan_sha256),
            ("capacity envelope", self.capacity_envelope_sha256),
            ("capacity authority", self.capacity_authority_sha256),
        ):
            _require_sha256(f"stage capacity schedule {label}", digest)
        if self.experiment not in INDUSTRIAL_EXPERIMENT_ORDER:
            raise ValueError("stage capacity schedule names an unknown experiment")
        if not self.activated_cell_ids or self.activated_cell_ids != tuple(
            sorted(set(self.activated_cell_ids))
        ):
            raise ValueError("stage capacity schedule cells must be sorted and unique")
        if not self.waves or tuple(row.wave_index for row in self.waves) != tuple(
            range(len(self.waves))
        ):
            raise ValueError("stage capacity waves must be nonempty and contiguous")
        scheduled = tuple(cell_id for wave in self.waves for cell_id in wave.cell_ids)
        if tuple(sorted(scheduled)) != self.activated_cell_ids:
            raise ValueError("stage capacity waves do not exactly cover stage cells")
        retry_cells = tuple(row.cell_id for row in self.retries)
        if retry_cells != self.activated_cell_ids:
            raise ValueError("stage capacity retries do not exactly cover stage cells")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule_sha256": self.sha256,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "experiment": self.experiment,
            "activated_cell_ids": list(self.activated_cell_ids),
            "gpu_inventory_sha256": self.gpu_inventory_sha256,
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "budget_plan_sha256": self.budget_plan_sha256,
            "capacity_envelope_sha256": self.capacity_envelope_sha256,
            "capacity_authority_sha256": self.capacity_authority_sha256,
            "waves": [
                {
                    "wave_index": row.wave_index,
                    "cell_ids": list(row.cell_ids),
                    "topology_sha256": row.topology_sha256,
                }
                for row in self.waves
            ],
            "retries": [
                {
                    "cell_id": row.cell_id,
                    "experiment_budget_sha256": row.experiment_budget_sha256,
                    "retry_allowance": row.retry_allowance,
                }
                for row in self.retries
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "schedule_sha256",
                "schema_version",
                "kind",
                "registry_sha256",
                "experiment",
                "activated_cell_ids",
                "gpu_inventory_sha256",
                "dispatch_plan_sha256",
                "budget_plan_sha256",
                "capacity_envelope_sha256",
                "capacity_authority_sha256",
                "waves",
                "retries",
            }
        )
        row = _strict_object("stage capacity schedule", value, fields)
        declared = row.pop("schedule_sha256")
        raw_cells = row.pop("activated_cell_ids")
        raw_waves = row.pop("waves")
        raw_retries = row.pop("retries")
        if (
            type(raw_cells) is not list
            or type(raw_waves) is not list
            or type(raw_retries) is not list
        ):
            raise TypeError("stage capacity schedule collections must be JSON arrays")
        wave_fields = frozenset({"wave_index", "cell_ids", "topology_sha256"})
        waves = []
        for index, raw_wave in enumerate(raw_waves):
            wave = _strict_object(
                f"stage capacity schedule wave {index}", raw_wave, wave_fields
            )
            cell_ids = wave["cell_ids"]
            if type(cell_ids) is not list:
                raise TypeError("stage capacity schedule wave cells must be an array")
            waves.append(
                StageCapacityWaveBinding(
                    wave_index=wave["wave_index"],
                    cell_ids=tuple(cell_ids),
                    topology_sha256=wave["topology_sha256"],
                )
            )
        retry_fields = frozenset(
            {"cell_id", "experiment_budget_sha256", "retry_allowance"}
        )
        retries = tuple(
            StageCapacityRetryBinding(
                **_strict_object(
                    f"stage capacity schedule retry {index}", raw_retry, retry_fields
                )
            )
            for index, raw_retry in enumerate(raw_retries)
        )
        schedule = cls(
            activated_cell_ids=tuple(raw_cells),
            waves=tuple(waves),
            retries=retries,
            **row,
        )
        if declared != schedule.sha256:
            raise ValueError("stage capacity schedule SHA-256 mismatch")
        return schedule


def _capacity_binding_from_dict(value: object) -> CapacityRawJsonBinding:
    fields = frozenset(
        {
            "schema_version",
            "path",
            "sidecar_path",
            "semantic_sha256",
            "file_sha256",
            "sidecar_file_sha256",
            "size",
            "sidecar_size",
        }
    )
    return CapacityRawJsonBinding(
        **_strict_object("stage capacity source-manifest binding", value, fields)
    )


@dataclass(frozen=True)
class StageCapacitySourceAuthority:
    """Path-bound raw capacity replay joined to one exact stage schedule.

    This object is intentionally not an authorization token.  It makes the
    reducer reproducible and prevents callers from supplying free-space or
    high-water numbers.  GPU dispatch still requires a fresh root-authorized
    ``capacity`` control whose subject is the derived gate.
    """

    schema_version: int
    kind: Literal["industrial_stage_capacity_source_authority"]
    registry_sha256: str
    experiment: str
    activated_cell_ids: tuple[str, ...]
    source_manifest: CapacityRawJsonBinding
    source_replay_sha256: str
    schedule_sha256: str
    capacity_envelope_sha256: str
    budget_inventory_sha256: str
    gpu_inventory_sha256: str
    captured_at_ns: int
    provider_quota_receipt_sha256: str
    host_capacity_receipt_sha256: str
    cell_sizing_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != "industrial_stage_capacity_source_authority"
        ):
            raise ValueError("stage capacity source authority schema is unsupported")
        if self.experiment not in INDUSTRIAL_EXPERIMENT_ORDER:
            raise ValueError("stage capacity source authority names an unknown stage")
        if not self.activated_cell_ids or self.activated_cell_ids != tuple(
            sorted(set(self.activated_cell_ids))
        ):
            raise ValueError("stage capacity source cells must be sorted and unique")
        if type(self.source_manifest) is not CapacityRawJsonBinding:
            raise TypeError("stage capacity source requires an exact raw binding")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("source replay", self.source_replay_sha256),
            ("schedule", self.schedule_sha256),
            ("capacity envelope", self.capacity_envelope_sha256),
            ("budget inventory", self.budget_inventory_sha256),
            ("GPU inventory", self.gpu_inventory_sha256),
            ("provider quota receipt", self.provider_quota_receipt_sha256),
            ("host capacity receipt", self.host_capacity_receipt_sha256),
        ):
            _require_sha256(f"stage capacity source {label}", digest)
        _require_nonnegative("stage capacity source capture time", self.captured_at_ns)
        if not self.cell_sizing_receipt_sha256s or any(
            _require_sha256("stage capacity sizing receipt", digest) != digest
            for digest in self.cell_sizing_receipt_sha256s
        ):
            raise ValueError("stage capacity source sizing receipts are invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    @classmethod
    def from_replay(
        cls,
        replay: UnsignedCapacitySourceReplay,
        *,
        schedule: StageCapacitySchedule,
    ) -> Self:
        if type(replay) is not UnsignedCapacitySourceReplay:
            raise TypeError("stage capacity source requires an exact raw replay")
        if type(schedule) is not StageCapacitySchedule:
            raise TypeError("stage capacity source requires an exact schedule")
        requirements = tuple(
            row.cell_id for row in replay.capacity_envelope.cell_requirements
        )
        if (
            replay.registry_sha256 != schedule.registry_sha256
            or replay.capacity_envelope.sha256 != schedule.capacity_envelope_sha256
            or replay.gpu_inventory.sha256 != schedule.gpu_inventory_sha256
            or requirements != schedule.activated_cell_ids
        ):
            raise ValueError(
                "raw capacity replay differs from the exact stage schedule"
            )
        return cls(
            schema_version=1,
            kind="industrial_stage_capacity_source_authority",
            registry_sha256=schedule.registry_sha256,
            experiment=schedule.experiment,
            activated_cell_ids=schedule.activated_cell_ids,
            source_manifest=replay.source_manifest,
            source_replay_sha256=replay.sha256,
            schedule_sha256=schedule.sha256,
            capacity_envelope_sha256=replay.capacity_envelope.sha256,
            budget_inventory_sha256=replay.budget_inventory.sha256,
            gpu_inventory_sha256=replay.gpu_inventory.sha256,
            captured_at_ns=replay.captured_at_ns,
            provider_quota_receipt_sha256=(replay.provider_quota_receipt_sha256),
            host_capacity_receipt_sha256=replay.host_capacity_receipt_sha256,
            cell_sizing_receipt_sha256s=(replay.cell_sizing_receipt_sha256s),
        )

    def revalidate(
        self,
        *,
        schedule: StageCapacitySchedule,
        now_ns: int,
    ) -> UnsignedCapacitySourceReplay:
        if type(schedule) is not StageCapacitySchedule:
            raise TypeError("stage capacity revalidation requires an exact schedule")
        if (
            schedule.sha256 != self.schedule_sha256
            or schedule.registry_sha256 != self.registry_sha256
            or schedule.experiment != self.experiment
            or schedule.activated_cell_ids != self.activated_cell_ids
        ):
            raise ValueError("stage capacity source belongs to another schedule")
        replay = replay_unsigned_capacity_source_manifest(
            self.source_manifest.path,
            expected_registry_sha256=self.registry_sha256,
            now_ns=now_ns,
        )
        expected = type(self).from_replay(replay, schedule=schedule)
        if expected != self:
            raise ValueError("stage capacity raw source replay changed")
        return replay

    def to_dict(self) -> dict[str, object]:
        return {
            "authority_sha256": self.sha256,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "experiment": self.experiment,
            "activated_cell_ids": list(self.activated_cell_ids),
            "source_manifest": self.source_manifest.to_dict(),
            "source_replay_sha256": self.source_replay_sha256,
            "schedule_sha256": self.schedule_sha256,
            "capacity_envelope_sha256": self.capacity_envelope_sha256,
            "budget_inventory_sha256": self.budget_inventory_sha256,
            "gpu_inventory_sha256": self.gpu_inventory_sha256,
            "captured_at_ns": self.captured_at_ns,
            "provider_quota_receipt_sha256": (self.provider_quota_receipt_sha256),
            "host_capacity_receipt_sha256": self.host_capacity_receipt_sha256,
            "cell_sizing_receipt_sha256s": list(self.cell_sizing_receipt_sha256s),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "authority_sha256",
                "schema_version",
                "kind",
                "registry_sha256",
                "experiment",
                "activated_cell_ids",
                "source_manifest",
                "source_replay_sha256",
                "schedule_sha256",
                "capacity_envelope_sha256",
                "budget_inventory_sha256",
                "gpu_inventory_sha256",
                "captured_at_ns",
                "provider_quota_receipt_sha256",
                "host_capacity_receipt_sha256",
                "cell_sizing_receipt_sha256s",
            }
        )
        row = _strict_object("stage capacity source authority", value, fields)
        declared = row.pop("authority_sha256")
        raw_cells = row.pop("activated_cell_ids")
        raw_manifest = row.pop("source_manifest")
        raw_sizing = row.pop("cell_sizing_receipt_sha256s")
        if type(raw_cells) is not list or type(raw_sizing) is not list:
            raise TypeError("stage capacity source collections must be JSON arrays")
        authority = cls(
            activated_cell_ids=tuple(raw_cells),
            source_manifest=_capacity_binding_from_dict(raw_manifest),
            cell_sizing_receipt_sha256s=tuple(raw_sizing),
            **row,
        )
        if declared != authority.sha256:
            raise ValueError("stage capacity source authority SHA-256 mismatch")
        return authority


def bind_stage_capacity_schedule(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    activated_cell_ids: tuple[str, ...],
    dispatch_plan: object,
    budget_plan: object,
) -> StageCapacitySchedule:
    """Bind the gate schedule to exact first-party dispatch and budget objects."""

    # Local imports avoid making planning/gpu-pool import stage-capacity during
    # their own module initialization.
    from lightcone_spec.experiments.gpu_pool import GpuDispatchPlan
    from lightcone_spec.experiments.planning import BudgetPlan

    cells = _validate_stage_cells(
        registry,
        experiment=experiment,
        activated_cell_ids=activated_cell_ids,
    )
    if type(dispatch_plan) is not GpuDispatchPlan:
        raise TypeError("stage capacity schedule requires an exact dispatch plan")
    if type(budget_plan) is not BudgetPlan:
        raise TypeError("stage capacity schedule requires an exact budget plan")
    if (
        dispatch_plan.registry_sha256 != registry.sha256
        or budget_plan.registry_sha256 != registry.sha256
        or budget_plan.activated_cell_ids != cells
        or dispatch_plan.completed_cell_ids
    ):
        raise ValueError("stage capacity schedule lineage differs from the stage")
    scheduled = tuple(
        assignment for wave in dispatch_plan.waves for assignment in wave.assignments
    )
    if tuple(sorted(row.work_item.item_id for row in scheduled)) != cells:
        raise ValueError("dispatch plan does not exactly cover the stage capacity set")
    registered = {cell.cell_id: cell for cell in registry.cells_for(experiment)}
    if any(
        assignment.work_item.cell != registered[assignment.work_item.item_id]
        for assignment in scheduled
    ):
        raise ValueError("dispatch plan changes a registered stage cell")
    budgets = {row.cell_id: row for row in budget_plan.budgets}
    if set(budgets) != set(cells):
        raise ValueError("stage capacity budget coverage differs from stage cells")
    dispatch_budget = dict(dispatch_plan.budget_sha256_by_cell)
    if dispatch_budget != {cell_id: row.sha256 for cell_id, row in budgets.items()}:
        raise ValueError("dispatch plan changes its stage budget bindings")
    envelope = budget_plan.capacity_envelope
    authority = budget_plan.capacity_authority
    if envelope is None or authority is None:
        raise ValueError("stage capacity schedule lacks signed capacity inputs")
    waves = tuple(
        StageCapacityWaveBinding(
            wave_index=wave.wave_index,
            cell_ids=tuple(
                sorted(assignment.work_item.item_id for assignment in wave.assignments)
            ),
            topology_sha256=content_sha256(
                tuple(
                    sorted(
                        (
                            assignment.work_item.item_id,
                            assignment.work_item.cell.sha256,
                            assignment.work_item.cell.identity.topology,
                            assignment.gpu_uuids,
                            assignment.rank_groups,
                        )
                        for assignment in wave.assignments
                    )
                )
            ),
        )
        for wave in dispatch_plan.waves
    )
    return StageCapacitySchedule(
        schema_version=1,
        kind="industrial_stage_capacity_schedule",
        registry_sha256=registry.sha256,
        experiment=experiment,
        activated_cell_ids=cells,
        gpu_inventory_sha256=dispatch_plan.inventory_sha256,
        dispatch_plan_sha256=dispatch_plan.sha256,
        budget_plan_sha256=budget_plan.sha256,
        capacity_envelope_sha256=envelope.sha256,
        capacity_authority_sha256=authority.sha256,
        waves=waves,
        retries=tuple(
            StageCapacityRetryBinding(
                cell_id=cell_id,
                experiment_budget_sha256=budgets[cell_id].sha256,
                retry_allowance=budgets[cell_id].retry_allowance,
            )
            for cell_id in cells
        ),
    )


@dataclass(frozen=True)
class StageCapacityGate:
    schema_version: int
    kind: Literal["industrial_stage_capacity_gate"]
    protocol_sha256: str
    registry_sha256: str
    experiment: str
    activated_cell_ids: tuple[str, ...]
    mode: Literal["SIGNED_STAGE_ENVELOPE", "LEGACY_100GB_FALLBACK"]
    status: Literal["AVAILABLE", "BLOCKED"]
    reason_code: str
    observed_free_bytes: int
    retained_evidence_bytes: int
    maximum_concurrent_transient_bytes: int
    high_water_bytes: int
    safety_margin_bytes: int
    required_free_bytes: int
    capacity_envelope_sha256: str | None
    capacity_verification_receipt_sha256: str | None
    capacity_source_authority: StageCapacitySourceAuthority | None
    schedule_sha256: str | None
    dispatch_plan_sha256: str | None
    budget_plan_sha256: str | None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version not in {2, 3}
            or self.kind != "industrial_stage_capacity_gate"
        ):
            raise ValueError("stage capacity gate schema is unsupported")
        if self.protocol_sha256 != STAGE_CAPACITY_GATE_PROTOCOL_SHA256:
            raise ValueError("stage capacity gate uses another protocol")
        _require_sha256("stage capacity registry", self.registry_sha256)
        if self.experiment not in INDUSTRIAL_EXPERIMENT_ORDER:
            raise ValueError("stage capacity gate names an unknown experiment")
        if not self.activated_cell_ids or self.activated_cell_ids != tuple(
            sorted(set(self.activated_cell_ids))
        ):
            raise ValueError("stage capacity cells must be non-empty, sorted, unique")
        if self.mode not in {"SIGNED_STAGE_ENVELOPE", "LEGACY_100GB_FALLBACK"}:
            raise ValueError("stage capacity mode is unsupported")
        if self.status not in {"AVAILABLE", "BLOCKED"}:
            raise ValueError("stage capacity status is unsupported")
        if (
            type(self.reason_code) is not str
            or not self.reason_code
            or any(character.isspace() for character in self.reason_code)
        ):
            raise ValueError("stage capacity reason code is invalid")
        for label, amount in (
            ("observed free bytes", self.observed_free_bytes),
            ("retained evidence bytes", self.retained_evidence_bytes),
            (
                "maximum concurrent transient bytes",
                self.maximum_concurrent_transient_bytes,
            ),
            ("high water bytes", self.high_water_bytes),
            ("safety margin bytes", self.safety_margin_bytes),
            ("required free bytes", self.required_free_bytes),
        ):
            _require_nonnegative(f"stage capacity {label}", amount)
        expected_status = (
            "AVAILABLE"
            if self.observed_free_bytes >= self.required_free_bytes
            else "BLOCKED"
        )
        if self.status != expected_status:
            raise ValueError("stage capacity status differs from byte envelope")
        if self.mode == "SIGNED_STAGE_ENVELOPE":
            for label, digest in (
                ("envelope", self.capacity_envelope_sha256),
                ("schedule", self.schedule_sha256),
                ("dispatch plan", self.dispatch_plan_sha256),
                ("budget plan", self.budget_plan_sha256),
            ):
                _require_sha256(f"stage capacity {label}", digest)
            if self.schema_version == 2:
                _require_sha256(
                    "stage capacity verification receipt",
                    self.capacity_verification_receipt_sha256,
                )
                if self.capacity_source_authority is not None:
                    raise ValueError(
                        "legacy signed capacity cannot embed raw authority"
                    )
            else:
                if self.capacity_verification_receipt_sha256 is not None:
                    raise ValueError(
                        "dynamic capacity must not inherit a legacy verification receipt"
                    )
                if (
                    type(self.capacity_source_authority)
                    is not StageCapacitySourceAuthority
                ):
                    raise TypeError(
                        "dynamic capacity requires exact path-bound source authority"
                    )
                authority = self.capacity_source_authority
                if (
                    authority.registry_sha256 != self.registry_sha256
                    or authority.experiment != self.experiment
                    or authority.activated_cell_ids != self.activated_cell_ids
                    or authority.schedule_sha256 != self.schedule_sha256
                    or authority.capacity_envelope_sha256
                    != self.capacity_envelope_sha256
                ):
                    raise ValueError(
                        "dynamic capacity source authority differs from the gate"
                    )
            if (
                self.high_water_bytes
                != self.retained_evidence_bytes
                + self.maximum_concurrent_transient_bytes
                or self.required_free_bytes
                != self.high_water_bytes + self.safety_margin_bytes
            ):
                raise ValueError("signed stage capacity total is inconsistent")
        else:
            if (
                self.capacity_envelope_sha256 is not None
                or self.capacity_verification_receipt_sha256 is not None
                or self.capacity_source_authority is not None
                or self.schedule_sha256 is not None
                or self.dispatch_plan_sha256 is not None
                or self.budget_plan_sha256 is not None
                or self.retained_evidence_bytes != 0
                or self.maximum_concurrent_transient_bytes != 0
                or self.high_water_bytes != 0
                or self.safety_margin_bytes != 0
                or self.required_free_bytes != LEGACY_MINIMUM_FREE_BYTES
            ):
                raise ValueError("legacy stage capacity fallback is not canonical")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "registry_sha256": self.registry_sha256,
            "experiment": self.experiment,
            "activated_cell_ids": list(self.activated_cell_ids),
            "mode": self.mode,
            "status": self.status,
            "reason_code": self.reason_code,
            "observed_free_bytes": self.observed_free_bytes,
            "retained_evidence_bytes": self.retained_evidence_bytes,
            "maximum_concurrent_transient_bytes": (
                self.maximum_concurrent_transient_bytes
            ),
            "high_water_bytes": self.high_water_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "required_free_bytes": self.required_free_bytes,
            "capacity_envelope_sha256": self.capacity_envelope_sha256,
            "capacity_verification_receipt_sha256": (
                self.capacity_verification_receipt_sha256
            ),
            "capacity_source_authority": (
                None
                if self.capacity_source_authority is None
                else self.capacity_source_authority.to_dict()
            ),
            "schedule_sha256": self.schedule_sha256,
            "dispatch_plan_sha256": self.dispatch_plan_sha256,
            "budget_plan_sha256": self.budget_plan_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {"gate_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "gate_sha256",
                "schema_version",
                "kind",
                "protocol_sha256",
                "registry_sha256",
                "experiment",
                "activated_cell_ids",
                "mode",
                "status",
                "reason_code",
                "observed_free_bytes",
                "retained_evidence_bytes",
                "maximum_concurrent_transient_bytes",
                "high_water_bytes",
                "safety_margin_bytes",
                "required_free_bytes",
                "capacity_envelope_sha256",
                "capacity_verification_receipt_sha256",
                "capacity_source_authority",
                "schedule_sha256",
                "dispatch_plan_sha256",
                "budget_plan_sha256",
            }
        )
        row = _strict_object("stage capacity gate", value, fields)
        declared = row.pop("gate_sha256")
        raw_cells = row.pop("activated_cell_ids")
        raw_source_authority = row.pop("capacity_source_authority")
        if type(raw_cells) is not list:
            raise TypeError("stage capacity gate cells must be a JSON array")
        gate = cls(
            activated_cell_ids=tuple(raw_cells),
            capacity_source_authority=(
                None
                if raw_source_authority is None
                else StageCapacitySourceAuthority.from_dict(raw_source_authority)
            ),
            **row,
        )
        if declared != gate.sha256:
            raise ValueError("stage capacity gate SHA-256 mismatch")
        return gate


def stage_capacity_control_lineage_sha256(
    *,
    activation_sha256: str,
    inventory_sha256: str,
    gate: StageCapacityGate,
) -> str:
    """Canonical control-subject lineage for one signed capacity gate."""

    _require_sha256("stage capacity control activation", activation_sha256)
    _require_sha256("stage capacity control inventory", inventory_sha256)
    if type(gate) is not StageCapacityGate:
        raise TypeError("stage capacity control lineage requires an exact gate")
    return content_sha256(
        {
            "schema_version": 2,
            "kind": "industrial_stage_capacity_control_lineage",
            "activation_sha256": activation_sha256,
            "inventory_sha256": inventory_sha256,
            "schedule_sha256": gate.schedule_sha256,
            "dispatch_plan_sha256": gate.dispatch_plan_sha256,
            "budget_plan_sha256": gate.budget_plan_sha256,
            "capacity_envelope_sha256": gate.capacity_envelope_sha256,
            "capacity_verification_receipt_sha256": (
                gate.capacity_verification_receipt_sha256
            ),
            "capacity_source_authority_sha256": (
                None
                if gate.capacity_source_authority is None
                else gate.capacity_source_authority.sha256
            ),
            "mode": gate.mode,
        }
    )


def _validate_stage_cells(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    activated_cell_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if type(registry) is not ExperimentRegistry:
        raise TypeError("stage capacity requires an exact registry")
    if experiment not in INDUSTRIAL_EXPERIMENT_ORDER:
        raise ValueError("stage capacity experiment is not registered")
    if not activated_cell_ids or activated_cell_ids != tuple(
        sorted(set(activated_cell_ids))
    ):
        raise ValueError("activated stage cells must be non-empty, sorted, unique")
    stage_cell_ids = {cell.cell_id for cell in registry.cells_for(experiment)}
    if not set(activated_cell_ids) <= stage_cell_ids:
        raise ValueError("stage capacity names a cell from another experiment")
    return activated_cell_ids


def _signed_gate(
    *,
    registry: ExperimentRegistry,
    experiment: str,
    activated_cell_ids: tuple[str, ...],
    result: CapacityAuthorityResult,
    schedule: StageCapacitySchedule,
) -> StageCapacityGate:
    if result.registry_sha256 != registry.sha256:
        raise ValueError("stage capacity authority belongs to another registry")
    if type(schedule) is not StageCapacitySchedule:
        raise TypeError("signed stage capacity requires an exact schedule")
    if (
        schedule.registry_sha256 != registry.sha256
        or schedule.experiment != experiment
        or schedule.activated_cell_ids != activated_cell_ids
        or schedule.gpu_inventory_sha256 != result.gpu_inventory.sha256
        or schedule.capacity_envelope_sha256 != result.capacity_envelope.sha256
    ):
        raise ValueError("stage capacity schedule differs from signed capacity lineage")
    requirements = result.capacity_envelope.cell_requirements
    if tuple(row.cell_id for row in requirements) != activated_cell_ids:
        raise ValueError("signed capacity envelope does not exactly cover the stage")
    return _derive_signed_gate(
        registry=registry,
        experiment=experiment,
        activated_cell_ids=activated_cell_ids,
        envelope=result.capacity_envelope,
        schedule=schedule,
        schema_version=2,
        verification_receipt_sha256=result.verification_receipt_sha256,
        source_authority=None,
    )


def _derive_signed_gate(
    *,
    registry: ExperimentRegistry,
    experiment: str,
    activated_cell_ids: tuple[str, ...],
    envelope: CapacityEnvelope,
    schedule: StageCapacitySchedule,
    schema_version: int,
    verification_receipt_sha256: str | None,
    source_authority: StageCapacitySourceAuthority | None,
) -> StageCapacityGate:
    """Derive every byte count from verified raw inputs and the exact schedule."""

    if type(envelope) is not CapacityEnvelope:
        raise TypeError("stage capacity reducer requires an exact envelope")
    if type(schedule) is not StageCapacitySchedule:
        raise TypeError("stage capacity reducer requires an exact schedule")
    if (
        registry.sha256 != schedule.registry_sha256
        or experiment != schedule.experiment
        or activated_cell_ids != schedule.activated_cell_ids
        or envelope.sha256 != schedule.capacity_envelope_sha256
    ):
        raise ValueError("stage capacity reducer inputs belong to another schedule")
    requirements = envelope.cell_requirements
    if tuple(row.cell_id for row in requirements) != activated_cell_ids:
        raise ValueError("capacity envelope does not exactly cover the stage")
    requirement_by_cell = {row.cell_id: row for row in requirements}
    retained = sum(
        requirement_by_cell[row.cell_id].maximum_evidence_bytes
        * (row.retry_allowance + 1)
        for row in schedule.retries
    )
    transient_by_wave = tuple(
        sum(
            requirement_by_cell[cell_id].model_staging_bytes
            + requirement_by_cell[cell_id].compile_overlay_bytes
            for cell_id in wave.cell_ids
        )
        for wave in schedule.waves
    )
    maximum_transient = max(transient_by_wave)
    high_water = retained + maximum_transient
    proportional_margin = (
        high_water * STAGE_CAPACITY_SAFETY_MARGIN_BPS + 9_999
    ) // 10_000
    safety_margin = max(
        STAGE_CAPACITY_MINIMUM_SAFETY_MARGIN_BYTES,
        proportional_margin,
    )
    required = high_water + safety_margin
    observed = envelope.effective_host_bytes
    available = observed >= required
    return StageCapacityGate(
        schema_version=schema_version,
        kind="industrial_stage_capacity_gate",
        protocol_sha256=STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
        registry_sha256=registry.sha256,
        experiment=experiment,
        activated_cell_ids=activated_cell_ids,
        mode="SIGNED_STAGE_ENVELOPE",
        status="AVAILABLE" if available else "BLOCKED",
        reason_code=(
            "signed_stage_capacity_verified"
            if available
            else "signed_stage_capacity_insufficient"
        ),
        observed_free_bytes=observed,
        retained_evidence_bytes=retained,
        maximum_concurrent_transient_bytes=maximum_transient,
        high_water_bytes=high_water,
        safety_margin_bytes=safety_margin,
        required_free_bytes=required,
        capacity_envelope_sha256=envelope.sha256,
        capacity_verification_receipt_sha256=verification_receipt_sha256,
        capacity_source_authority=source_authority,
        schedule_sha256=schedule.sha256,
        dispatch_plan_sha256=schedule.dispatch_plan_sha256,
        budget_plan_sha256=schedule.budget_plan_sha256,
    )


def materialize_stage_capacity_gate_from_raw_sources(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    activated_cell_ids: tuple[str, ...],
    source_manifest_path: str,
    schedule: StageCapacitySchedule,
    now_ns: int,
) -> StageCapacityGate:
    """Deep-replay raw capacity bytes and derive an unsigned dynamic gate.

    The result is safe to present to the offline control signer, but it does
    not authorize dispatch until a root-authorized ``capacity`` control is
    verified and its challenge atomically reserved.
    """

    cells = _validate_stage_cells(
        registry,
        experiment=experiment,
        activated_cell_ids=activated_cell_ids,
    )
    if type(schedule) is not StageCapacitySchedule:
        raise TypeError("raw stage capacity materialization requires a schedule")
    if (
        schedule.registry_sha256 != registry.sha256
        or schedule.experiment != experiment
        or schedule.activated_cell_ids != cells
    ):
        raise ValueError("raw stage capacity schedule belongs to another stage")
    replay = replay_unsigned_capacity_source_manifest(
        source_manifest_path,
        expected_registry_sha256=registry.sha256,
        now_ns=now_ns,
    )
    authority = StageCapacitySourceAuthority.from_replay(
        replay,
        schedule=schedule,
    )
    return _derive_signed_gate(
        registry=registry,
        experiment=experiment,
        activated_cell_ids=cells,
        envelope=replay.capacity_envelope,
        schedule=schedule,
        schema_version=3,
        verification_receipt_sha256=None,
        source_authority=authority,
    )


def revalidate_stage_capacity_gate_sources(
    registry: ExperimentRegistry,
    gate: StageCapacityGate,
    *,
    schedule: StageCapacitySchedule,
    now_ns: int,
) -> UnsignedCapacitySourceReplay:
    """Deep-reopen the raw inputs behind a dynamic gate before control use."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("raw source revalidation requires an exact registry")
    if type(gate) is not StageCapacityGate or gate.schema_version != 3:
        raise TypeError("raw source revalidation requires a dynamic capacity gate")
    if type(gate.capacity_source_authority) is not StageCapacitySourceAuthority:
        raise TypeError("dynamic capacity gate lacks path-bound source authority")
    replay = gate.capacity_source_authority.revalidate(
        schedule=schedule,
        now_ns=now_ns,
    )
    expected = _derive_signed_gate(
        registry=registry,
        experiment=gate.experiment,
        activated_cell_ids=gate.activated_cell_ids,
        envelope=replay.capacity_envelope,
        schedule=schedule,
        schema_version=3,
        verification_receipt_sha256=None,
        source_authority=gate.capacity_source_authority,
    )
    if expected != gate:
        raise ValueError("stage capacity gate differs from rederived raw capacity")
    return replay


def evaluate_stage_capacity(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    activated_cell_ids: tuple[str, ...],
    capacity_authority: CapacityAuthorityBinding | None = None,
    capacity_inventory: BudgetInventoryIdentity | None = None,
    capacity_envelope: CapacityEnvelope | None = None,
    schedule: StageCapacitySchedule | None = None,
    legacy_host_free_bytes: int | None = None,
) -> StageCapacityGate:
    """Prefer fresh signed sizing; otherwise retain the 100 GB floor.

    Invalid or tampered signed input raises and never falls back.  Only an
    absent bundle or the named source-verifier-unavailable state may use the
    legacy floor.
    """

    cells = _validate_stage_cells(
        registry,
        experiment=experiment,
        activated_cell_ids=activated_cell_ids,
    )
    supplied = (
        capacity_authority is not None,
        capacity_inventory is not None,
        capacity_envelope is not None,
        schedule is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "signed stage capacity and schedule inputs must be supplied together"
        )
    if all(supplied):
        assert capacity_authority is not None
        assert capacity_inventory is not None
        assert capacity_envelope is not None
        assert schedule is not None
        try:
            result = revalidate_capacity_authority_binding(
                capacity_authority,
                expected_registry_sha256=registry.sha256,
                expected_inventory=capacity_inventory,
                expected_envelope=capacity_envelope,
            )
        except CapacityAuthorityUnavailableError:
            result = None
        if result is not None:
            return _signed_gate(
                registry=registry,
                experiment=experiment,
                activated_cell_ids=cells,
                result=result,
                schedule=schedule,
            )
    if type(legacy_host_free_bytes) is not int or legacy_host_free_bytes < 0:
        raise ValueError(
            "legacy host free bytes are required when signed capacity is unavailable"
        )
    available = legacy_host_free_bytes >= LEGACY_MINIMUM_FREE_BYTES
    return StageCapacityGate(
        schema_version=2,
        kind="industrial_stage_capacity_gate",
        protocol_sha256=STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
        registry_sha256=registry.sha256,
        experiment=experiment,
        activated_cell_ids=cells,
        mode="LEGACY_100GB_FALLBACK",
        status="AVAILABLE" if available else "BLOCKED",
        reason_code=(
            "legacy_100gb_capacity_verified"
            if available
            else "legacy_100gb_capacity_insufficient"
        ),
        observed_free_bytes=legacy_host_free_bytes,
        retained_evidence_bytes=0,
        maximum_concurrent_transient_bytes=0,
        high_water_bytes=0,
        safety_margin_bytes=0,
        required_free_bytes=LEGACY_MINIMUM_FREE_BYTES,
        capacity_envelope_sha256=None,
        capacity_verification_receipt_sha256=None,
        capacity_source_authority=None,
        schedule_sha256=None,
        dispatch_plan_sha256=None,
        budget_plan_sha256=None,
    )


__all__ = [
    "LEGACY_MINIMUM_FREE_BYTES",
    "STAGE_CAPACITY_GATE_PROTOCOL_SHA256",
    "STAGE_CAPACITY_MINIMUM_SAFETY_MARGIN_BYTES",
    "STAGE_CAPACITY_SAFETY_MARGIN_BPS",
    "StageCapacityGate",
    "StageCapacityRetryBinding",
    "StageCapacitySchedule",
    "StageCapacitySourceAuthority",
    "StageCapacityWaveBinding",
    "bind_stage_capacity_schedule",
    "evaluate_stage_capacity",
    "materialize_stage_capacity_gate_from_raw_sources",
    "revalidate_stage_capacity_gate_sources",
    "stage_capacity_control_lineage_sha256",
]
