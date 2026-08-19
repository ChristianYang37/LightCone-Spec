"""Registered pre-allocation budget for E5's 264 one-shot diagnostics.

The final E5 GPU-hour envelope cannot exist until every correctness-only
failure row has run.  This module closes that bootstrap cycle without treating
the rows as pilots or inventing a prospective estimate.  It derives an upper
bound from the signed E5 materialization, the code-owned failure lifecycle,
and an AVAILABLE preflight receipt.  A release control reserves the budget
once; individual launches still require fresh capacity controls and the shared
append-only stage spend ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.experiments.formal_gpu_hour_registry import (
    FormalStageGpuHourVerificationReceipt,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_registry_layers import (
    load_formal_registry_verification_receipt_path,
)
from lightcone_spec.experiments.stage_materialization import (
    E5_BACKENDS,
    E5_COHORT_COUNTS,
    E5_FAILURES,
    E5_TOPOLOGIES,
    StageMaterializationReceipt,
    default_e5_failure_diagnostic_authority,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
    VerifiedControlArtifact,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_E5_ONE_SHOT_PROCESS_HARD_TIMEOUT_NS = 600_000_000_000
FORMAL_E5_ONE_SHOT_PROVIDER_HARD_TIMEOUT_NS = 660_000_000_000
FORMAL_E5_ONE_SHOT_BUDGET_MAXIMUM_LIFETIME_NS = 86_400_000_000_000

FORMAL_E5_ONE_SHOT_LAUNCH_BUDGET_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_e5_one_shot_launch_budget_protocol",
        "scope": "exact_signed_264_correctness_only_failure_rows",
        "matrix": "11_scenarios_x_2_backends_x_3_topologies_x_4_cohorts",
        "process_timeout_ns": FORMAL_E5_ONE_SHOT_PROCESS_HARD_TIMEOUT_NS,
        "provider_timeout_ns": FORMAL_E5_ONE_SHOT_PROVIDER_HARD_TIMEOUT_NS,
        "attempts": 1,
        "estimate": "forbidden_actual_lifecycle_cost_only",
        "authorization": "one_durable_budget_control_then_fresh_cell_controls",
    }
)
FORMAL_E5_ONE_SHOT_BUDGET_VERIFICATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_e5_one_shot_budget_verification_protocol",
        "budget_protocol_sha256": (FORMAL_E5_ONE_SHOT_LAUNCH_BUDGET_PROTOCOL_SHA256),
        "reservation": "release_control_challenge_reserved_once",
        "cell_spend": "append_only_external_ledger",
    }
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _strict(label: str, value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


@dataclass(frozen=True)
class FormalE5OneShotCellCap:
    materialized_cell_id: str
    backend: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    scenario: str
    cohort_count: int
    failure_member_id: str
    gpu_count: Literal[1, 2]
    provider_reserved_gpu_count: Literal[2]
    allowed_attempts: Literal[1]
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    maximum_compute_gpu_ns: int
    maximum_provider_reserved_gpu_ns: int

    def __post_init__(self) -> None:
        _sha256("formal E5 one-shot cell", self.materialized_cell_id)
        _sha256("formal E5 one-shot member", self.failure_member_id)
        expected_gpus = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            self.backend not in E5_BACKENDS
            or self.topology_mode not in E5_TOPOLOGIES
            or self.scenario not in E5_FAILURES
            or self.cohort_count not in E5_COHORT_COUNTS
            or self.gpu_count != expected_gpus
            or self.provider_reserved_gpu_count != 2
            or self.allowed_attempts != 1
            or self.process_hard_timeout_ns
            != FORMAL_E5_ONE_SHOT_PROCESS_HARD_TIMEOUT_NS
            or self.provider_wave_hard_timeout_ns
            != FORMAL_E5_ONE_SHOT_PROVIDER_HARD_TIMEOUT_NS
            or self.maximum_compute_gpu_ns
            != self.process_hard_timeout_ns * self.gpu_count
            or self.maximum_provider_reserved_gpu_ns
            != self.provider_wave_hard_timeout_ns * self.provider_reserved_gpu_count
        ):
            raise ValueError("formal E5 one-shot cell cap differs from protocol")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict(
                "formal E5 one-shot cell cap", value, set(cls.__dataclass_fields__)
            )
        )  # type: ignore[arg-type]


def _failure_cells(
    materialization: StageMaterializationReceipt,
) -> tuple[FormalE5OneShotCellCap, ...]:
    if (
        type(materialization) is not StageMaterializationReceipt
        or materialization.stage != "E5"
        or materialization.materialization_rule
        != "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
    ):
        raise ValueError("formal E5 one-shot budget requires final E5 materialization")
    authority = default_e5_failure_diagnostic_authority()
    rows = tuple(
        row
        for row in materialization.cells
        if row.task == "deterministic_failure_injection"
    )
    if (
        len(rows) != 264
        or {dict(row.dimensions).get("failure_member_id") for row in rows}
        != {member.member_id for member in authority.members}
        or any(
            row.method_role != "LightCone"
            or row.publication_policy != "diagnostic_only"
            or dict(row.dimensions).get("diagnostic_only") != "true"
            or dict(row.dimensions).get("failure_authority_sha256") != authority.sha256
            for row in rows
        )
    ):
        raise ValueError("formal E5 one-shot failure matrix is not exact 264")
    caps = []
    for row in rows:
        dimensions = dict(row.dimensions)
        topology = dimensions["topology"]
        cap = FormalE5OneShotCellCap(
            materialized_cell_id=row.cell_id,
            backend=row.backend,
            topology_mode=topology,  # type: ignore[arg-type]
            scenario=dimensions["failure"],
            cohort_count=dimensions["cohort_count"],
            failure_member_id=dimensions["failure_member_id"],
            gpu_count=1 if topology == "tp1_dp1" else 2,
            provider_reserved_gpu_count=2,
            allowed_attempts=1,
            process_hard_timeout_ns=FORMAL_E5_ONE_SHOT_PROCESS_HARD_TIMEOUT_NS,
            provider_wave_hard_timeout_ns=(FORMAL_E5_ONE_SHOT_PROVIDER_HARD_TIMEOUT_NS),
            maximum_compute_gpu_ns=(
                FORMAL_E5_ONE_SHOT_PROCESS_HARD_TIMEOUT_NS
                * (1 if topology == "tp1_dp1" else 2)
            ),
            maximum_provider_reserved_gpu_ns=(
                FORMAL_E5_ONE_SHOT_PROVIDER_HARD_TIMEOUT_NS * 2
            ),
        )
        caps.append(cap)
    result = tuple(sorted(caps, key=lambda row: row.materialized_cell_id))
    if len({row.failure_member_id for row in result}) != 264:
        raise ValueError("formal E5 one-shot member identities are not unique")
    return result


@dataclass(frozen=True)
class FormalE5OneShotLaunchBudget:
    schema_version: Literal[1]
    kind: Literal["lightcone_formal_e5_one_shot_launch_budget"]
    protocol_sha256: str
    registry_layer: CanonicalJsonProofBinding
    registry_receipt_sha256: str
    preflight_budget_receipt: CanonicalJsonProofBinding
    preflight_budget_receipt_sha256: str
    materialization: CanonicalJsonProofBinding
    materialization_receipt_sha256: str
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    registry_sha256: str
    inventory_sha256: str
    cell_caps: tuple[FormalE5OneShotCellCap, ...]
    maximum_compute_gpu_ns: int
    maximum_provider_reserved_gpu_ns: int
    issued_ns: int
    expires_ns: int
    launch_nonce_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_e5_one_shot_launch_budget"
            or self.protocol_sha256 != FORMAL_E5_ONE_SHOT_LAUNCH_BUDGET_PROTOCOL_SHA256
        ):
            raise ValueError("formal E5 one-shot budget schema differs")
        for label, value in (
            ("registry receipt", self.registry_receipt_sha256),
            ("preflight budget", self.preflight_budget_receipt_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime", self.runtime_authority_manifest_sha256),
            ("registry", self.registry_sha256),
            ("inventory", self.inventory_sha256),
            ("launch nonce", self.launch_nonce_sha256),
        ):
            _sha256(f"formal E5 one-shot budget {label}", value)
        if any(
            type(value) is not CanonicalJsonProofBinding
            for value in (
                self.registry_layer,
                self.preflight_budget_receipt,
                self.materialization,
            )
        ):
            raise TypeError("formal E5 one-shot budget sources are not path-bound")
        ids = tuple(row.materialized_cell_id for row in self.cell_caps)
        if (
            len(self.cell_caps) != 264
            or ids != tuple(sorted(set(ids)))
            or self.maximum_compute_gpu_ns
            != sum(row.maximum_compute_gpu_ns for row in self.cell_caps)
            or self.maximum_provider_reserved_gpu_ns
            != sum(row.maximum_provider_reserved_gpu_ns for row in self.cell_caps)
            or type(self.issued_ns) is not int
            or type(self.expires_ns) is not int
            or self.issued_ns < 1
            or not self.issued_ns < self.expires_ns
            or self.expires_ns - self.issued_ns
            > FORMAL_E5_ONE_SHOT_BUDGET_MAXIMUM_LIFETIME_NS
        ):
            raise ValueError("formal E5 one-shot budget coverage/cap differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "registry_layer": self.registry_layer.to_dict(),
            "preflight_budget_receipt": self.preflight_budget_receipt.to_dict(),
            "materialization": self.materialization.to_dict(),
            "cell_caps": [row.to_dict() for row in self.cell_caps],
            "budget_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal E5 one-shot launch budget",
            value,
            {*cls.__dataclass_fields__, "budget_sha256"},
        )
        declared = _sha256("formal E5 one-shot budget", row.pop("budget_sha256"))
        for name in (
            "registry_layer",
            "preflight_budget_receipt",
            "materialization",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_caps = row.pop("cell_caps")
        if type(raw_caps) is not list:
            raise TypeError("formal E5 one-shot caps are not an array")
        budget = cls(
            **row,
            cell_caps=tuple(
                FormalE5OneShotCellCap.from_dict(item) for item in raw_caps
            ),
        )
        if budget.sha256 != declared:
            raise ValueError("formal E5 one-shot budget digest differs")
        return budget

    def cap_for(self, materialized_cell_id: str) -> FormalE5OneShotCellCap:
        rows = tuple(
            row
            for row in self.cell_caps
            if row.materialized_cell_id == materialized_cell_id
        )
        if len(rows) != 1:
            raise ValueError("formal E5 one-shot budget lacks exact cell")
        return rows[0]


def _reopen_materialization(
    binding: CanonicalJsonProofBinding,
) -> StageMaterializationReceipt:
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )

    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal E5 one-shot materialization binding changed")
    result = stage_materialization_receipt_from_dict(binding.reopen())
    if result.sha256 != binding.semantic_sha256:
        raise ValueError("formal E5 one-shot materialization identity changed")
    return result


def revalidate_formal_e5_one_shot_launch_budget(
    budget: FormalE5OneShotLaunchBudget,
    *,
    current_ns: int,
) -> tuple[object, FormalStageGpuHourVerificationReceipt, StageMaterializationReceipt]:
    budget.__post_init__()
    if (
        type(current_ns) is not int
        or not budget.issued_ns <= current_ns <= budget.expires_ns
    ):
        raise ValueError("formal E5 one-shot budget is not fresh")
    for source in (
        budget.registry_layer,
        budget.preflight_budget_receipt,
        budget.materialization,
    ):
        if CanonicalJsonProofBinding.bind(source.absolute_path) != source:
            raise ValueError("formal E5 one-shot budget source changed")
    registry = load_formal_registry_verification_receipt_path(
        budget.registry_layer.absolute_path,
        now_ns=current_ns,
    )
    preflight = FormalStageGpuHourVerificationReceipt.from_dict(
        budget.preflight_budget_receipt.reopen()
    )
    preflight.revalidate(current_ns=current_ns)
    materialization = _reopen_materialization(budget.materialization)
    lock = registry.signed_protocol_lock.payload
    registered = tuple(
        row.payload
        for row in registry.cumulative_signed_materializations
        if row.payload.sha256 == materialization.sha256
    )
    if (
        len(registered) != 1
        or registered[0] != materialization
        or budget.registry_receipt_sha256 != registry.sha256
        or budget.preflight_budget_receipt_sha256 != preflight.sha256
        or preflight.stage != "preflight"
        or preflight.inventory.sha256 != registry.inventory_sha256
        or preflight.registry_receipt.signed_protocol_lock.payload.sha256 != lock.sha256
        or budget.materialization_receipt_sha256 != materialization.sha256
        or budget.protocol_lock_sha256 != lock.sha256
        or budget.runtime_authority_manifest_sha256
        != preflight.formal_runtime_authority_manifest.sha256
        or budget.runtime_authority_manifest_sha256
        != lock.formal_runtime_authority_manifest_sha256
        or budget.registry_sha256 != lock.registry_sha256
        or budget.inventory_sha256 != registry.inventory_sha256
    ):
        raise ValueError("formal E5 one-shot budget immutable lineage differs")
    caps = _failure_cells(materialization)
    nonce = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_e5_one_shot_launch_nonce",
            "registry_receipt_sha256": registry.sha256,
            "preflight_budget_receipt_sha256": preflight.sha256,
            "materialization_receipt_sha256": materialization.sha256,
            "cell_ids": [row.materialized_cell_id for row in caps],
            "process_hard_timeout_ns": FORMAL_E5_ONE_SHOT_PROCESS_HARD_TIMEOUT_NS,
            "provider_hard_timeout_ns": FORMAL_E5_ONE_SHOT_PROVIDER_HARD_TIMEOUT_NS,
            "issued_ns": budget.issued_ns,
            "expires_ns": budget.expires_ns,
        }
    )
    if (
        budget.cell_caps != caps
        or budget.maximum_compute_gpu_ns
        != sum(row.maximum_compute_gpu_ns for row in caps)
        or budget.maximum_provider_reserved_gpu_ns
        != sum(row.maximum_provider_reserved_gpu_ns for row in caps)
        or budget.launch_nonce_sha256 != nonce
    ):
        raise ValueError("formal E5 one-shot budget cap derivation differs")
    return registry, preflight, materialization


def materialize_formal_e5_one_shot_launch_budget(
    *,
    registry_layer_path: str | Path,
    preflight_budget_receipt_path: str | Path,
    issued_ns: int,
    expires_ns: int,
    output_path: str | Path,
) -> FormalE5OneShotLaunchBudget:
    """Derive the exact 264-cell cap; the caller supplies no IDs or timeout."""

    registry_binding = CanonicalJsonProofBinding.bind(registry_layer_path)
    registry = load_formal_registry_verification_receipt_path(
        registry_binding.absolute_path,
        now_ns=issued_ns,
    )
    preflight_binding = CanonicalJsonProofBinding.bind(preflight_budget_receipt_path)
    preflight = FormalStageGpuHourVerificationReceipt.from_dict(
        preflight_binding.reopen()
    )
    preflight.revalidate(current_ns=issued_ns)
    candidates = tuple(
        row.payload
        for row in registry.cumulative_signed_materializations
        if row.payload.stage == "E5"
        and row.payload.materialization_rule
        == "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
    )
    if len(candidates) != 1:
        raise ValueError(
            "formal E5 one-shot budget lacks one signed final materialization"
        )
    materialization = candidates[0]
    materialization_path = Path(output_path).parent / (
        "formal-e5-one-shot-materialization.json"
    )
    publish_canonical_json_no_replace(
        materialization_path,
        stage_materialization_receipt_to_dict(materialization),
    )
    materialization_binding = CanonicalJsonProofBinding.bind(materialization_path)
    caps = _failure_cells(materialization)
    lock = registry.signed_protocol_lock.payload
    nonce = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_e5_one_shot_launch_nonce",
            "registry_receipt_sha256": registry.sha256,
            "preflight_budget_receipt_sha256": preflight.sha256,
            "materialization_receipt_sha256": materialization.sha256,
            "cell_ids": [row.materialized_cell_id for row in caps],
            "process_hard_timeout_ns": FORMAL_E5_ONE_SHOT_PROCESS_HARD_TIMEOUT_NS,
            "provider_hard_timeout_ns": FORMAL_E5_ONE_SHOT_PROVIDER_HARD_TIMEOUT_NS,
            "issued_ns": issued_ns,
            "expires_ns": expires_ns,
        }
    )
    budget = FormalE5OneShotLaunchBudget(
        schema_version=1,
        kind="lightcone_formal_e5_one_shot_launch_budget",
        protocol_sha256=FORMAL_E5_ONE_SHOT_LAUNCH_BUDGET_PROTOCOL_SHA256,
        registry_layer=registry_binding,
        registry_receipt_sha256=registry.sha256,
        preflight_budget_receipt=preflight_binding,
        preflight_budget_receipt_sha256=preflight.sha256,
        materialization=materialization_binding,
        materialization_receipt_sha256=materialization.sha256,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_manifest_sha256=(
            preflight.formal_runtime_authority_manifest.sha256
        ),
        registry_sha256=lock.registry_sha256,
        inventory_sha256=registry.inventory_sha256,
        cell_caps=caps,
        maximum_compute_gpu_ns=sum(row.maximum_compute_gpu_ns for row in caps),
        maximum_provider_reserved_gpu_ns=sum(
            row.maximum_provider_reserved_gpu_ns for row in caps
        ),
        issued_ns=issued_ns,
        expires_ns=expires_ns,
        launch_nonce_sha256=nonce,
    )
    revalidate_formal_e5_one_shot_launch_budget(budget, current_ns=issued_ns)
    publish_canonical_json_no_replace(output_path, budget.to_dict())
    reopened = FormalE5OneShotLaunchBudget.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != budget:
        raise RuntimeError("formal E5 one-shot budget changed during publication")
    return budget


def formal_e5_one_shot_budget_control_subject(
    budget: FormalE5OneShotLaunchBudget,
) -> ControlArtifactSubject:
    return ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="rank_aggregate",
        artifact_sha256=budget.sha256,
        protocol_sha256=FORMAL_E5_ONE_SHOT_LAUNCH_BUDGET_PROTOCOL_SHA256,
        registry_sha256=budget.registry_sha256,
        lineage_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_formal_e5_one_shot_budget_control_lineage",
                "registry_layer": budget.registry_layer.to_dict(),
                "preflight_budget_receipt": budget.preflight_budget_receipt.to_dict(),
                "materialization": budget.materialization.to_dict(),
                "launch_nonce_sha256": budget.launch_nonce_sha256,
            }
        ),
    )


@dataclass(frozen=True)
class FormalE5OneShotBudgetVerificationReceipt:
    schema_version: Literal[1]
    kind: Literal["lightcone_formal_e5_one_shot_budget_verification_receipt"]
    protocol_sha256: str
    budget: CanonicalJsonProofBinding
    budget_sha256: str
    control: ControlArtifactAttestation
    reservation: ChallengeReplayReservationBinding
    registry_sha256: str
    inventory_sha256: str
    root_manifest_sha256: str
    trusted_attester_policy_sha256: str
    verified_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_e5_one_shot_budget_verification_receipt"
            or self.protocol_sha256
            != FORMAL_E5_ONE_SHOT_BUDGET_VERIFICATION_PROTOCOL_SHA256
        ):
            raise ValueError("formal E5 one-shot verification schema differs")
        if (
            type(self.budget) is not CanonicalJsonProofBinding
            or type(self.control) is not ControlArtifactAttestation
            or type(self.reservation) is not ChallengeReplayReservationBinding
            or type(self.verified_ns) is not int
            or self.verified_ns < 1
        ):
            raise TypeError("formal E5 one-shot verification sources differ")
        for label, value in (
            ("budget", self.budget_sha256),
            ("registry", self.registry_sha256),
            ("inventory", self.inventory_sha256),
            ("root", self.root_manifest_sha256),
            ("policy", self.trusted_attester_policy_sha256),
        ):
            _sha256(f"formal E5 one-shot verification {label}", value)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "budget": self.budget.to_dict(),
            "control": self.control.to_dict(),
            "reservation": self.reservation.to_dict(),
            "receipt_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal E5 one-shot verification receipt",
            value,
            {*cls.__dataclass_fields__, "receipt_sha256"},
        )
        declared = _sha256(
            "formal E5 one-shot verification receipt",
            row.pop("receipt_sha256"),
        )
        row["budget"] = CanonicalJsonProofBinding.from_dict(row["budget"])
        row["control"] = ControlArtifactAttestation.from_dict(row["control"])
        row["reservation"] = ChallengeReplayReservationBinding.from_dict(
            row["reservation"]
        )
        receipt = cls(**row)
        if receipt.sha256 != declared:
            raise ValueError("formal E5 one-shot verification digest differs")
        return receipt


def reserve_formal_e5_one_shot_budget_verification(
    *,
    budget_path: str | Path,
    control: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    current_ns: int,
    output_path: str | Path,
) -> FormalE5OneShotBudgetVerificationReceipt:
    binding = CanonicalJsonProofBinding.bind(budget_path)
    budget = FormalE5OneShotLaunchBudget.from_dict(binding.reopen())
    registry, preflight, _materialization = revalidate_formal_e5_one_shot_launch_budget(
        budget,
        current_ns=current_ns,
    )
    if control.subject != formal_e5_one_shot_budget_control_subject(budget):
        raise ValueError("formal E5 one-shot budget control subject differs")
    lock = registry.signed_protocol_lock.payload
    policy_sha256 = registry.trusted_release_policy(current_ns=current_ns).sha256
    if (
        Path(replay_store.root) != Path(registry.reservation.path).parent
        or control.deployment_policy_authorization.root_manifest_sha256
        != lock.offline_release_trust_root_sha256
        or control.trusted_attester_policy_sha256 != policy_sha256
        or preflight.inventory.sha256 != budget.inventory_sha256
    ):
        raise ValueError("formal E5 one-shot budget control authority differs")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control,),
        expected_inventory_sha256=budget.inventory_sha256,
        now_ns=current_ns,
        replay_store=replay_store,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified,
        reserved_ns=current_ns,
    )
    receipt = FormalE5OneShotBudgetVerificationReceipt(
        schema_version=1,
        kind="lightcone_formal_e5_one_shot_budget_verification_receipt",
        protocol_sha256=(FORMAL_E5_ONE_SHOT_BUDGET_VERIFICATION_PROTOCOL_SHA256),
        budget=binding,
        budget_sha256=budget.sha256,
        control=control,
        reservation=replay_store.bind_reservation(reservation_sha256),
        registry_sha256=budget.registry_sha256,
        inventory_sha256=budget.inventory_sha256,
        root_manifest_sha256=lock.offline_release_trust_root_sha256,
        trusted_attester_policy_sha256=policy_sha256,
        verified_ns=current_ns,
    )
    publish_canonical_json_no_replace(output_path, receipt.to_dict())
    reopened = FormalE5OneShotBudgetVerificationReceipt.from_dict(
        CanonicalJsonProofBinding.bind(output_path).reopen()
    )
    if reopened != receipt:
        raise RuntimeError("formal E5 one-shot verification changed during publication")
    return receipt


def revalidate_formal_e5_one_shot_budget_verification(
    binding: CanonicalJsonProofBinding,
    *,
    current_ns: int,
) -> tuple[
    FormalE5OneShotBudgetVerificationReceipt,
    FormalE5OneShotLaunchBudget,
    object,
]:
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("formal E5 one-shot verification binding changed")
    receipt = FormalE5OneShotBudgetVerificationReceipt.from_dict(binding.reopen())
    if CanonicalJsonProofBinding.bind(receipt.budget.absolute_path) != receipt.budget:
        raise ValueError("formal E5 one-shot verified budget changed")
    budget = FormalE5OneShotLaunchBudget.from_dict(receipt.budget.reopen())
    registry, _preflight, _materialization = (
        revalidate_formal_e5_one_shot_launch_budget(
            budget,
            current_ns=current_ns,
        )
    )
    lock = registry.signed_protocol_lock.payload
    policy_sha256 = registry.trusted_release_policy(
        current_ns=receipt.verified_ns
    ).sha256
    verified: VerifiedControlArtifact = verify_release_control_artifact_attestation(
        receipt.control,
        expected_inventory_sha256=budget.inventory_sha256,
        now_ns=receipt.verified_ns,
        consumed_challenge_sha256s=(),
    )
    expected_reservation = control_challenge_reservation_sha256(
        (verified,),
        reserved_ns=receipt.verified_ns,
    )
    expected_challenges = tuple(
        sorted(
            {
                verified.challenge_sha256,
                verified.deployment_policy_challenge_sha256,
            }
        )
    )
    if (
        receipt.budget_sha256 != budget.sha256
        or receipt.control.subject != formal_e5_one_shot_budget_control_subject(budget)
        or receipt.registry_sha256 != budget.registry_sha256
        or receipt.inventory_sha256 != budget.inventory_sha256
        or receipt.root_manifest_sha256 != lock.offline_release_trust_root_sha256
        or receipt.trusted_attester_policy_sha256 != policy_sha256
        or receipt.reservation.reservation_sha256 != expected_reservation
        or receipt.reservation.revalidate() != expected_challenges
    ):
        raise ValueError("formal E5 one-shot verification lineage differs")
    return receipt, budget, registry


__all__ = [
    "FORMAL_E5_ONE_SHOT_BUDGET_VERIFICATION_PROTOCOL_SHA256",
    "FORMAL_E5_ONE_SHOT_LAUNCH_BUDGET_PROTOCOL_SHA256",
    "FORMAL_E5_ONE_SHOT_PROCESS_HARD_TIMEOUT_NS",
    "FORMAL_E5_ONE_SHOT_PROVIDER_HARD_TIMEOUT_NS",
    "FormalE5OneShotBudgetVerificationReceipt",
    "FormalE5OneShotCellCap",
    "FormalE5OneShotLaunchBudget",
    "formal_e5_one_shot_budget_control_subject",
    "materialize_formal_e5_one_shot_launch_budget",
    "reserve_formal_e5_one_shot_budget_verification",
    "revalidate_formal_e5_one_shot_budget_verification",
    "revalidate_formal_e5_one_shot_launch_budget",
]
