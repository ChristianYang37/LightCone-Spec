"""Path-bound GPU-hour authority derived from first-party lifecycle proofs.

The historical pilot receipt accepts caller-authored millisecond scalars.  It
is useful for diagnostic arithmetic, but it is not evidence.  This module is
the formal path: every duration is reopened from a locally controlled
``PinnedSglangLifecycleTimingProofArtifact`` and joined to a verifier-sealed
materialized-cell execution binding.  Wave placement is reconstructed from
the proof timestamps, gang size comes from the exact GPU UUID assignment, and
the fixed two-GPU provider reservation is charged even for isolated TP1 work.

The immutable materialization remains ``UNMEASURED``.  The reducer publishes a
separate source manifest and returns a schema-2 ``StageGpuHourEnvelope``.  A
later signer may attest that envelope, but a signature over the legacy scalar
receipt can never enter this path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from lightcone_spec.experiments.formal_protocol import (
    FormalRuntimeAuthorityManifest,
    ProtocolLock,
    content_sha256,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageGpuHourEnvelope,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

if TYPE_CHECKING:
    from lightcone_spec.orchestration.formal_failure_physical import (
        FormalE5FailureLifecycleCostProjection,
    )
    from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding


NANOSECONDS_PER_HOUR = 3_600_000_000_000
FORMAL_GPU_HOUR_RETRY_RESERVE_BPS = 1_000
FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR = 10_000
_PHASE_EDGE_NAMES = (
    "execution_started_ns",
    "server_ready_ns",
    "begin_started_ns",
    "begin_finished_ns",
    "warmup_started_ns",
    "warmup_finished_ns",
    "reset_started_ns",
    "reset_finished_ns",
    "scored_executor_started_ns",
    "scored_request_started_ns",
    "scored_request_finished_ns",
    "scored_executor_finished_ns",
    "finalize_started_ns",
    "finalize_finished_ns",
    "terminal_published_ns",
    "itl_pointer_published_ns",
    "native_terminal_finished_ns",
    "process_exited_ns",
    "process_group_empty_checked_ns",
    "evidence_flush_started_ns",
    "evidence_flush_finished_ns",
)
_PHASE_DURATION_NAMES = (
    "startup_ns",
    "warmup_ns",
    "adaptation_reset_ns",
    "scored_request_window_ns",
    "drain_ns",
    "process_cleanup_ns",
    "evidence_flush_ns",
    "reserved_wall_ns",
    "profile_reserved_ns",
)

FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "lightcone_formal_gpu_hour_lifecycle_budget_protocol",
        "inputs": (
            "verifier_sealed_materialized_cell_execution_binding",
            "path_bound_root_controlled_pinned_sglang_lifecycle_timing_proof",
            "preflight_scientific_rows_plus_resolved_qualification_lifecycle",
            "exact_two_gpu_inventory",
        ),
        "duration_unit": "integer_nanoseconds",
        "schedule": "overlap_components_derived_from_first_party_phase_edges",
        "compute": "sum_execution_start_to_process_exit_times_exact_gang",
        "wall": "sum_wave_start_to_process_group_empty_spans",
        "provider_base_reservation": "core_wall_times_exact_two_gpu_inventory",
        "retry_reserve_basis_points": FORMAL_GPU_HOUR_RETRY_RESERVE_BPS,
        "retry_reserve_basis": "measured_compute_gpu_nanoseconds",
        "profile_reserve": "zero_when_profile_cells_are_already_materialized",
        "evidence_reserve": "post_process_empty_wave_tail_times_two_gpu_inventory",
        "no_double_count": "compute_profile_evidence_intervals_are_disjoint",
        "legacy_scalar_duration_receipts": "diagnostic_only_never_formal",
        "nonserving_cost_union": (
            "preflight_requires_admission_bound_core10_or_eagle11_qualification",
            "E4_profiler_requires_dedicated_nsys_ncu_lifecycle_cost",
            "E6_model_preflight_requires_dedicated_compatibility_lifecycle_cost",
        ),
    }
)
FORMAL_GPU_HOUR_BUDGET_RUNNER_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_formal_gpu_hour_lifecycle_budget_runner",
        "module": "lightcone_spec.experiments.gpu_hour_authority",
        "entrypoints": (
            "materialize_stage_gpu_hour_envelope_from_lifecycle_proofs",
            "materialize_preflight_gpu_hour_envelope",
            "materialize_staged_prospective_gpu_hour_envelope",
            "materialize_e5_failure_gpu_hour_source_manifest",
            "revalidate_stage_gpu_hour_source_manifest",
            "revalidate_persisted_preflight_gpu_hour_source_manifest",
            "revalidate_persisted_staged_prospective_gpu_hour_source_manifest",
            "revalidate_persisted_e5_failure_gpu_hour_source_manifest",
        ),
    }
)
FORMAL_GPU_HOUR_BUDGET_TEST_SET_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_formal_gpu_hour_lifecycle_budget_acceptance_set",
        "requirements": (
            "single_tp1_fixed_two_gpu_reservation",
            "two_simultaneous_tp1_wave",
            "three_sequential_tp2_gangs",
            "foreign_cell_execution_plan_inventory_root_and_hardware_rejected",
            "changed_phase_timestamps_and_reused_proof_rejected",
            "preflight_exact_1_compile_1_exactness_8_interference",
            "preflight_missing_phase_timing_or_lifecycle_proof_blocked",
            "preflight_legacy_1_plus_1_plus_8_source_never_available",
            "preflight_resolved_qualification_process_costs_required_once",
            "early_stage_unmeasured_stratum_blocked_with_minimum_pilot",
            "early_stage_projection_uses_only_same_stratum_lifecycle_proofs",
            "actual_and_projected_compute_share_full_gpu_process_interval",
            "profile_cells_are_explicit_and_never_double_reserved",
            "e5_exact_264_uses_integrated_failure_lifecycle_only",
            "e5_failure_subject_control_replay_and_consumptions_are_unique",
            "e5_failure_actual_cost_has_zero_retry_and_projection",
            "e4_profiler_rejects_ordinary_serving_lifecycle_and_launch_caps",
            "e6_model_preflight_rejects_ordinary_serving_lifecycle_and_launch_caps",
            "legacy_scalar_receipt_never_authorizes",
        ),
    }
)
FORMAL_SERVING_EXECUTION_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_serving_execution_proof_protocol",
        "source": FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
        "trust_lift": "root_authorized_dispatch_control_and_atomic_replay",
        "durable_identity": (
            "materialization_cell_stage_method_recipe",
            "run_config_execution_plan_rank_topology_gpu",
            "runtime_gpu_proofs_hardware_run_nonce_attempt",
        ),
        "consumer": "formal_gpu_hour_lifecycle_source_manifest",
    }
)
PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_prospective_gpu_hour_protocol",
        "pilot_source": "excluded_four_blocks_with_first_party_lifecycle_proofs",
        "projection": "exact_scientific_stratum_ceiling_mean_isolated_schedule",
        "power": "typed_signed_stage_power_prefix_only",
        "categories": (
            "actual_tuning",
            "projected_final",
            "actual_one_shot",
        ),
        "e5_failure": (
            "all_264_rows_require_dedicated_failure_subject_joined_"
            "actuation_recovery_and_lifecycle_cost_proofs"
        ),
        "e6_model_preflight": (
            "blocked_until_dedicated_compatibility_lifecycle_cost_proof"
        ),
        "profile": "explicit_profile_cells_never_reserved_twice",
        "scalar_duration_inputs": "forbidden",
    }
)
E5_FAILURE_GPU_HOUR_SOURCE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e5_failure_gpu_hour_source_protocol",
        "matrix": "exact_264_correctness_only_failure_cells",
        "input": (
            "path_bound_formal_failure_execution_rebuild_input_plus_"
            "controlled_integrated_failure_lifecycle_cost_proof"
        ),
        "join": (
            "failure_subject_binding_assignment_run_nonce_plan_admission_"
            "launch_and_budget_consumptions_raw_terminal_recovery_pgid_gpu"
        ),
        "compute": "execution_started_through_process_exited_times_exact_gang",
        "provider": "execution_started_through_gpu_release_times_provider_gang",
        "evidence": "gpu_release_through_evidence_flush_times_inventory",
        "ordinary_serving_lifecycle": "forbidden",
        "projection": "actual_only_never_scaled_or_retried",
    }
)
STAGED_PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_staged_prospective_gpu_hour_protocol",
        "stages": ("E3a", "TTS-Cal", "E1", "E2", "E4", "E1a"),
        "actual_source": (
            "path_bound_root_controlled_lifecycle_proofs_for_completed_cells"
        ),
        "stratum": ("all_scientific_cell_identity_except_registered_repeat_dimensions"),
        "registered_repeat_dimensions": (
            ("E3a", ("registry_cell_id",)),
            ("TTS-Cal", ("block", "pilot_phase", "registry_cell_id")),
            ("E1", ()),
            ("E2", ()),
            ("E4", ()),
            ("E1a", ()),
        ),
        "projection": (
            "same_stratum_ceiling_mean_with_isolated_two_gpu_provider_reservation"
        ),
        "compute_interval": (
            "execution_started_through_process_exited_including_startup_warmup_"
            "reset_scored_arrivals_drain_and_native_finalize_times_exact_gang"
        ),
        "post_process_evidence_flush": "separate_provider_reserve_not_compute",
        "uncovered_stratum": "BLOCKED_with_deterministic_minimum_pilot_cell",
        "e4_profiler": "blocked_until_dedicated_nsys_ncu_lifecycle_cost_proof",
        "categories": (
            "actual_completed",
            "projected_remaining",
            "total",
        ),
        "profile": "explicit_profile_cells_never_reserved_twice",
        "legacy_smoke_or_caller_duration": "forbidden",
    }
)
PREFLIGHT_GPU_HOUR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_preflight_gpu_hour_protocol",
        "authority": "typed_two_phase_formal_preflight_final_evidence",
        "scientific_coverage": "exact_1_compile_1_exactness_8_interference_zero_skip",
        "qualification": (
            "root_resolved_core10_or_core10_plus_eagle11_with_admission_bound_"
            "process_lifecycle_cost_counted_once"
        ),
        "eagle_resolution": (
            "root_replayed_signed108_all_na_zero_cost_else_exact_eagle_suite"
        ),
        "compile_timing": "first_party_subprocess_process_started_to_exited",
        "exactness_timing": "first_party_terminal_and_two_rank_process_bounds",
        "interference_timing": "root_controlled_pinned_sglang_lifecycle",
        "schedule": "phase_interval_overlap_components_exact_gpu_ownership",
        "provider_reservation": "fixed_two_gpu_wall_plus_evidence_tail",
        "legacy_schema1_status": "blocked_never_available_or_signable",
        "scalar_duration_inputs": "forbidden",
    }
)


class FormalGpuHourLifecycleBlocked(RuntimeError):
    """A formal budget lacks source-owned lifecycle authority."""

    def __init__(self, reason_code: str) -> None:
        if type(reason_code) is not str or not reason_code:
            raise ValueError("formal GPU-hour BLOCKED reason must be text")
        self.reason_code = reason_code
        super().__init__(
            f"formal GPU-hour lifecycle authority is BLOCKED: {reason_code}"
        )


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _absolute_path(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f"{label} must be absolute and resolved")
    return value


def _strict_dict(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _run_binding_to_dict(value: NativeTerminalRunBinding) -> dict[str, object]:
    from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding

    if type(value) is not NativeTerminalRunBinding:
        raise TypeError("lifecycle GPU-hour input requires an exact native run binding")
    value.validate()
    return {
        "run_id": value.run_id,
        "run_nonce_sha256": value.run_nonce_sha256,
        "execution_plan_sha256": value.execution_plan_sha256,
        "rank_config_sha256": value.rank_config_sha256,
        "attempt_id": value.attempt_id,
        "session_id": value.session_id,
        "session_epoch": value.session_epoch,
        "previous_run_id": value.previous_run_id,
        "challenge_nonce_sha256": value.challenge_nonce_sha256,
        "method": value.method,
        "warmup_request_ids": list(value.warmup_request_ids),
        "scored_request_ids": list(value.scored_request_ids),
    }


def _run_binding_from_dict(value: object) -> NativeTerminalRunBinding:
    from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding

    fields = {
        "run_id",
        "run_nonce_sha256",
        "execution_plan_sha256",
        "rank_config_sha256",
        "attempt_id",
        "session_id",
        "session_epoch",
        "previous_run_id",
        "challenge_nonce_sha256",
        "method",
        "warmup_request_ids",
        "scored_request_ids",
    }
    row = _strict_dict("lifecycle native run binding", value, fields)
    warmup = row.pop("warmup_request_ids")
    scored = row.pop("scored_request_ids")
    if type(warmup) is not list or type(scored) is not list:
        raise TypeError("lifecycle native request IDs must be arrays")
    binding = NativeTerminalRunBinding(
        **row,
        warmup_request_ids=tuple(warmup),
        scored_request_ids=tuple(scored),
    )
    binding.validate()
    return binding


def _execution_proof_lineage_sha256(
    *,
    protocol_lock_sha256: str,
    runtime_authority_manifest_sha256: str,
    materialization_receipt_sha256: str,
    materialized_cell_id: str,
    inventory_sha256: str,
    execution_binding_sha256: str,
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_serving_execution_proof_lineage",
            "protocol_lock_sha256": protocol_lock_sha256,
            "runtime_authority_manifest_sha256": (runtime_authority_manifest_sha256),
            "materialization_receipt_sha256": materialization_receipt_sha256,
            "materialized_cell_id": materialized_cell_id,
            "inventory_sha256": inventory_sha256,
            "execution_binding_sha256": execution_binding_sha256,
        }
    )


@dataclass(frozen=True)
class FormalServingExecutionProofPayload:
    """Durable public projection of one private-sealed execution binding."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_execution_proof_payload"]
    protocol_lock_sha256: str
    runtime_authority_manifest_sha256: str
    materialization_receipt_sha256: str
    materialized_cell_id: str
    stage: str
    method: str
    run_config_sha256: str
    recipe_authority_sha256s: tuple[str, ...]
    subject_sha256: str
    execution_binding_sha256: str
    inventory_sha256: str
    topology_mode: str
    gpu_uuids: tuple[str, ...]
    runtime_gpu_proof_artifacts: tuple[CanonicalJsonProofBinding, ...]
    runtime_gpu_proof_sha256s: tuple[str, ...]
    hardware_envelope_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    run_id: str
    run_nonce_sha256: str
    attempt_id: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_execution_proof_payload"
        ):
            raise ValueError("formal serving execution proof schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_manifest_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("run config", self.run_config_sha256),
            ("subject", self.subject_sha256),
            ("execution binding", self.execution_binding_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("rank config", self.rank_config_sha256),
            ("run nonce", self.run_nonce_sha256),
        ):
            _sha256(f"formal serving execution proof {label}", digest)
        for label, value in (
            ("stage", self.stage),
            ("method", self.method),
            ("topology", self.topology_mode),
            ("run ID", self.run_id),
            ("attempt ID", self.attempt_id),
        ):
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"formal serving execution proof {label} is invalid")
        if self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("formal serving execution proof topology is unsupported")
        expected_gpu_count = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpu_count
            or len(set(self.gpu_uuids)) != expected_gpu_count
        ):
            raise ValueError("formal serving execution proof GPU coverage differs")
        if (
            type(self.recipe_authority_sha256s) is not tuple
            or not self.recipe_authority_sha256s
            or self.recipe_authority_sha256s
            != tuple(sorted(set(self.recipe_authority_sha256s)))
        ):
            raise ValueError("formal serving recipe authorities are not canonical")
        if (
            type(self.runtime_gpu_proof_artifacts) is not tuple
            or not self.runtime_gpu_proof_artifacts
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.runtime_gpu_proof_artifacts
            )
            or tuple(row.absolute_path for row in self.runtime_gpu_proof_artifacts)
            != tuple(
                sorted(row.absolute_path for row in self.runtime_gpu_proof_artifacts)
            )
        ):
            raise ValueError("formal serving runtime proof artifacts are not canonical")
        if (
            type(self.runtime_gpu_proof_sha256s) is not tuple
            or self.runtime_gpu_proof_sha256s
            != tuple(sorted(set(self.runtime_gpu_proof_sha256s)))
            or len(self.runtime_gpu_proof_sha256s)
            != len(self.runtime_gpu_proof_artifacts)
        ):
            raise ValueError("formal serving verified runtime proofs are not exact")
        for digest in (
            *self.recipe_authority_sha256s,
            *self.runtime_gpu_proof_sha256s,
        ):
            _sha256("formal serving execution proof member", digest)
        expected_binding = content_sha256(
            {
                "protocol_sha256": FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
                "subject_sha256": self.subject_sha256,
                "runtime_gpu_proof_sha256s": self.runtime_gpu_proof_sha256s,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
            }
        )
        if self.execution_binding_sha256 != expected_binding:
            raise ValueError("formal serving execution binding identity differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "runtime_authority_manifest_sha256": (
                self.runtime_authority_manifest_sha256
            ),
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "materialized_cell_id": self.materialized_cell_id,
            "stage": self.stage,
            "method": self.method,
            "run_config_sha256": self.run_config_sha256,
            "recipe_authority_sha256s": list(self.recipe_authority_sha256s),
            "subject_sha256": self.subject_sha256,
            "execution_binding_sha256": self.execution_binding_sha256,
            "inventory_sha256": self.inventory_sha256,
            "topology_mode": self.topology_mode,
            "gpu_uuids": list(self.gpu_uuids),
            "runtime_gpu_proof_artifacts": [
                row.to_dict() for row in self.runtime_gpu_proof_artifacts
            ],
            "runtime_gpu_proof_sha256s": list(self.runtime_gpu_proof_sha256s),
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "run_id": self.run_id,
            "run_nonce_sha256": self.run_nonce_sha256,
            "attempt_id": self.attempt_id,
            "payload_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {
            *cls.__dataclass_fields__,
            "payload_sha256",
        }
        row = _strict_dict("formal serving execution proof payload", value, fields)
        declared = row.pop("payload_sha256")
        for name in (
            "recipe_authority_sha256s",
            "gpu_uuids",
            "runtime_gpu_proof_sha256s",
        ):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(
                    f"formal serving execution proof {name} must be an array"
                )
            row[name] = tuple(raw)
        raw_artifacts = row["runtime_gpu_proof_artifacts"]
        if type(raw_artifacts) is not list:
            raise TypeError("formal serving runtime proof artifacts must be an array")
        row["runtime_gpu_proof_artifacts"] = tuple(
            CanonicalJsonProofBinding.from_dict(item) for item in raw_artifacts
        )
        payload = cls(**row)
        if payload.sha256 != declared:
            raise ValueError("formal serving execution proof payload digest differs")
        return payload


@dataclass(frozen=True)
class FormalServingExecutionProofArtifact:
    """Root-controlled, replay-bound durable execution/cell authority."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_execution_proof_artifact"]
    payload: FormalServingExecutionProofPayload
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_execution_proof_artifact"
            or type(self.payload) is not FormalServingExecutionProofPayload
            or type(self.control_attestation) is not ControlArtifactAttestation
            or type(self.replay_reservation) is not ChallengeReplayReservationBinding
        ):
            raise ValueError("formal serving execution proof artifact is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "payload": self.payload.to_dict(),
                "control_attestation": self.control_attestation.to_dict(),
                "replay_reservation": self.replay_reservation.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "payload": self.payload.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_dict(
            "formal serving execution proof artifact",
            value,
            {
                "schema_version",
                "kind",
                "payload",
                "control_attestation",
                "replay_reservation",
            },
        )
        artifact = cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            payload=FormalServingExecutionProofPayload.from_dict(row["payload"]),
            control_attestation=ControlArtifactAttestation.from_dict(
                row["control_attestation"]
            ),
            replay_reservation=ChallengeReplayReservationBinding.from_dict(
                row["replay_reservation"]
            ),
        )
        return artifact


def _load_formal_serving_execution_proof_artifact(
    path: str,
) -> FormalServingExecutionProofArtifact:
    binding = CanonicalJsonProofBinding.bind(path)
    artifact = FormalServingExecutionProofArtifact.from_dict(binding.reopen())
    if artifact.sha256 != binding.semantic_sha256:
        raise ValueError("formal serving execution proof semantic identity differs")
    return artifact


def _load_lifecycle_timing_proof_artifact(path: str) -> object:
    from lightcone_spec.orchestration.live_sglang import (
        PinnedSglangLifecycleTimingProofArtifact,
    )

    return PinnedSglangLifecycleTimingProofArtifact.from_dict(
        CanonicalJsonProofBinding.bind(path).reopen()
    )


def _execution_proof_payload(
    binding: VerifiedFormalServingExecutionBinding,
) -> FormalServingExecutionProofPayload:
    from lightcone_spec.config import run_config_sha256

    verified = require_verified_formal_serving_execution_binding(binding)
    subject = verified.subject
    identity = subject.execution_identity
    return FormalServingExecutionProofPayload(
        schema_version=1,
        kind="formal_serving_execution_proof_payload",
        protocol_lock_sha256=subject.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(
            subject.formal_runtime_authority_manifest_sha256
        ),
        materialization_receipt_sha256=subject.materialization_receipt_sha256,
        materialized_cell_id=subject.materialized_cell_id,
        stage=subject.stage,
        method=subject.method,
        run_config_sha256=run_config_sha256(verified.run_config),
        recipe_authority_sha256s=subject.recipe_authority_sha256s,
        subject_sha256=subject.sha256,
        execution_binding_sha256=verified.sha256,
        inventory_sha256=subject.inventory_sha256,
        topology_mode=subject.topology_mode,
        gpu_uuids=subject.gpu_uuids,
        runtime_gpu_proof_artifacts=tuple(
            sorted(
                subject.runtime_gpu_proof_artifacts,
                key=lambda row: row.absolute_path,
            )
        ),
        runtime_gpu_proof_sha256s=verified.runtime_gpu_proof_sha256s,
        hardware_envelope_sha256=verified.hardware_envelope_sha256,
        execution_plan_sha256=subject.execution_plan_sha256,
        rank_config_sha256=subject.rank_config_sha256,
        run_id=identity.run_id,
        run_nonce_sha256=identity.run_nonce_sha256,
        attempt_id=identity.attempt_id,
    )


def publish_formal_serving_execution_proof_artifact(
    binding: VerifiedFormalServingExecutionBinding,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    output_path: str,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Trust-lift one verifier-sealed execution binding for durable consumers."""

    payload = _execution_proof_payload(binding)
    control = control_attestation
    if type(control) is not ControlArtifactAttestation:
        raise TypeError("formal serving execution proof requires exact control")
    if type(replay_store) is not ChallengeReplayStore:
        raise TypeError("formal serving execution proof requires replay store")
    lineage = _execution_proof_lineage_sha256(
        protocol_lock_sha256=payload.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(payload.runtime_authority_manifest_sha256),
        materialization_receipt_sha256=payload.materialization_receipt_sha256,
        materialized_cell_id=payload.materialized_cell_id,
        inventory_sha256=payload.inventory_sha256,
        execution_binding_sha256=payload.execution_binding_sha256,
    )
    subject = control.subject
    if (
        subject.artifact_type != "dispatch"
        or subject.artifact_sha256 != payload.sha256
        or subject.protocol_sha256 != FORMAL_SERVING_EXECUTION_PROOF_PROTOCOL_SHA256
        or subject.registry_sha256 != binding.subject.execution_identity.registry_sha256
        or subject.lineage_sha256 != lineage
        or control.hardware_envelope_sha256 != payload.hardware_envelope_sha256
    ):
        raise ValueError("formal serving execution proof control subject differs")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control,),
        expected_inventory_sha256=payload.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified, reserved_ns=now_ns
    )
    artifact = FormalServingExecutionProofArtifact(
        schema_version=1,
        kind="formal_serving_execution_proof_artifact",
        payload=payload,
        control_attestation=control,
        replay_reservation=replay_store.bind_reservation(reservation_sha256),
    )
    destination = _absolute_path("formal serving execution proof", output_path)
    publish_canonical_json_no_replace(destination, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(destination, semantic_sha256=artifact.sha256)


def validate_formal_serving_execution_proof_artifact(
    artifact_path: str,
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    expected_cell_id: str,
    now_ns: int,
) -> FormalServingExecutionProofPayload:
    """Deep-open a root-controlled exact cell/execution mapping."""

    path = _absolute_path("formal serving execution proof", artifact_path)
    binding = CanonicalJsonProofBinding.bind(path)
    artifact = _load_formal_serving_execution_proof_artifact(path)
    payload = artifact.payload
    cells = {row.cell_id: row for row in materialization.cells}
    cell = cells.get(expected_cell_id)
    expected_method = {
        "Target-only": "target_only",
        "Static": "static",
        "TTS": "tts",
        "L0-naive": "l0",
        "LightCone-candidate": "l0",
        "LightCone": "l0",
        "TTS-calibration-candidate": "tts",
        "OnlineSPEC-OGD": "onlinespec_ogd",
        "OnlineSPEC-OPT": "onlinespec_opt",
        "OnlineSPEC-ENS": "onlinespec_ens",
        "OnlineSPEC-Optimistic-OGD": "onlinespec_opt",
        "OnlineSPEC-Hedge": "onlinespec_ens",
    }.get(cell.method_role if cell is not None else "")
    if (
        artifact.sha256 != binding.semantic_sha256
        or cell is None
        or expected_method is None
        or payload.protocol_lock_sha256 != protocol_lock.sha256
        or payload.runtime_authority_manifest_sha256
        != formal_runtime_authority_manifest.sha256
        or payload.materialization_receipt_sha256 != materialization.sha256
        or payload.materialized_cell_id != cell.cell_id
        or payload.stage != materialization.stage
        or payload.method != expected_method
        or payload.inventory_sha256 != inventory.sha256
        or any(
            uuid not in {device.uuid for device in inventory.devices}
            for uuid in payload.gpu_uuids
        )
    ):
        raise ValueError("formal serving execution proof lineage differs")
    for proof in payload.runtime_gpu_proof_artifacts:
        if CanonicalJsonProofBinding.bind(proof.absolute_path) != proof:
            raise ValueError("formal serving runtime GPU proof changed")
    lineage = _execution_proof_lineage_sha256(
        protocol_lock_sha256=payload.protocol_lock_sha256,
        runtime_authority_manifest_sha256=(payload.runtime_authority_manifest_sha256),
        materialization_receipt_sha256=payload.materialization_receipt_sha256,
        materialized_cell_id=payload.materialized_cell_id,
        inventory_sha256=payload.inventory_sha256,
        execution_binding_sha256=payload.execution_binding_sha256,
    )
    control = artifact.control_attestation
    control_subject = control.subject
    if (
        control_subject.artifact_type != "dispatch"
        or control_subject.artifact_sha256 != payload.sha256
        or control_subject.protocol_sha256
        != FORMAL_SERVING_EXECUTION_PROOF_PROTOCOL_SHA256
        or control_subject.registry_sha256 != protocol_lock.registry_sha256
        or control_subject.lineage_sha256 != lineage
        or control.hardware_envelope_sha256 != payload.hardware_envelope_sha256
        or control.deployment_policy_authorization.root_manifest_sha256
        != protocol_lock.offline_release_trust_root_sha256
    ):
        raise ValueError("formal serving execution proof control differs")
    reservation = artifact.replay_reservation
    if type(now_ns) is not int or now_ns < reservation.reserved_ns:
        raise ValueError(
            "formal serving execution proof validation predates acceptance"
        )
    verified = verify_release_control_artifact_attestation(
        control,
        expected_inventory_sha256=inventory.sha256,
        now_ns=reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    expected_reservation = control_challenge_reservation_sha256(
        (verified,), reserved_ns=reservation.reserved_ns
    )
    reserved = reservation.revalidate()
    if expected_reservation != reservation.reservation_sha256 or reserved != tuple(
        sorted(
            {
                verified.challenge_sha256,
                verified.deployment_policy_challenge_sha256,
            }
        )
    ):
        raise ValueError("formal serving execution proof replay differs")
    return payload


@dataclass(frozen=True)
class LifecycleGpuHourProofInput:
    """Ephemeral join of a sealed cell binding and one durable timing proof."""

    execution_binding: VerifiedFormalServingExecutionBinding
    native_run_binding: NativeTerminalRunBinding
    lifecycle_proof_artifact_path: str
    execution_proof_artifact_path: str

    def __post_init__(self) -> None:
        require_verified_formal_serving_execution_binding(self.execution_binding)
        _run_binding_to_dict(self.native_run_binding)
        _absolute_path("lifecycle GPU-hour proof", self.lifecycle_proof_artifact_path)
        _absolute_path(
            "formal serving execution proof", self.execution_proof_artifact_path
        )


@dataclass(frozen=True)
class LifecycleGpuHourObservation:
    """Durable, proof-derived timing for one exact materialized cell."""

    materialized_cell_id: str
    execution_binding_sha256: str
    execution_proof: CanonicalJsonProofBinding
    execution_proof_payload_sha256: str
    execution_control_envelope_sha256: str
    execution_replay_reservation: ChallengeReplayReservationBinding
    native_run_binding: NativeTerminalRunBinding
    lifecycle_proof: CanonicalJsonProofBinding
    verified_lifecycle_proof_sha256: str
    raw_timing_sha256: str
    live_run_receipt_sha256: str
    native_result_proof_sha256: str
    run_binding_sha256: str
    control_envelope_sha256: str
    lifecycle_replay_reservation: ChallengeReplayReservationBinding
    telemetry_detail: Literal["headline", "profile"]
    gpu_uuids: tuple[str, ...]
    phase_edges_ns: tuple[tuple[str, int], ...]
    phase_durations_ns: tuple[tuple[str, int], ...]
    wave_index: int

    def __post_init__(self) -> None:
        for label, digest in (
            ("materialized cell", self.materialized_cell_id),
            ("execution binding", self.execution_binding_sha256),
            ("execution proof payload", self.execution_proof_payload_sha256),
            ("execution control", self.execution_control_envelope_sha256),
            ("verified lifecycle proof", self.verified_lifecycle_proof_sha256),
            ("raw timing", self.raw_timing_sha256),
            ("live run", self.live_run_receipt_sha256),
            ("native result proof", self.native_result_proof_sha256),
            ("run binding", self.run_binding_sha256),
            ("control envelope", self.control_envelope_sha256),
        ):
            _sha256(f"lifecycle GPU-hour {label}", digest)
        if (
            type(self.execution_proof) is not CanonicalJsonProofBinding
            or type(self.execution_replay_reservation)
            is not ChallengeReplayReservationBinding
            or type(self.lifecycle_replay_reservation)
            is not ChallengeReplayReservationBinding
        ):
            raise TypeError("lifecycle GPU-hour durable proof bindings are invalid")
        _run_binding_to_dict(self.native_run_binding)
        if type(self.lifecycle_proof) is not CanonicalJsonProofBinding:
            raise TypeError("lifecycle GPU-hour proof binding is not path-bound")
        if self.telemetry_detail not in {"headline", "profile"}:
            raise ValueError("lifecycle GPU-hour telemetry detail is unsupported")
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) not in {1, 2}
            or len(set(self.gpu_uuids)) != len(self.gpu_uuids)
            or any(not value.startswith("GPU-") for value in self.gpu_uuids)
        ):
            raise ValueError("lifecycle GPU-hour gang identity is invalid")
        if (
            tuple(name for name, _value in self.phase_edges_ns) != _PHASE_EDGE_NAMES
            or tuple(name for name, _value in self.phase_durations_ns)
            != _PHASE_DURATION_NAMES
            or any(
                type(value) is not int or value < 0
                for _name, value in (*self.phase_edges_ns, *self.phase_durations_ns)
            )
        ):
            raise ValueError("lifecycle GPU-hour phase coverage is not exact")
        edges = dict(self.phase_edges_ns)
        durations = dict(self.phase_durations_ns)
        if (
            tuple(edges.values()) != tuple(sorted(edges.values()))
            or durations["reserved_wall_ns"]
            != edges["evidence_flush_finished_ns"] - edges["execution_started_ns"]
            or durations["reserved_wall_ns"] <= 0
            or durations["scored_request_window_ns"] <= 0
            or durations["profile_reserved_ns"]
            != (
                durations["reserved_wall_ns"]
                if self.telemetry_detail == "profile"
                else 0
            )
        ):
            raise ValueError("lifecycle GPU-hour phase arithmetic differs")
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise ValueError("lifecycle GPU-hour wave index is invalid")

    @property
    def start_ns(self) -> int:
        return dict(self.phase_edges_ns)["execution_started_ns"]

    @property
    def finish_ns(self) -> int:
        return dict(self.phase_edges_ns)["evidence_flush_finished_ns"]

    @property
    def gang_gpu_count(self) -> int:
        return len(self.gpu_uuids)

    @cached_property
    def schedule_assignment_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "lifecycle_gpu_hour_schedule_assignment",
                "materialized_cell_id": self.materialized_cell_id,
                "execution_binding_sha256": self.execution_binding_sha256,
                "verified_lifecycle_proof_sha256": (
                    self.verified_lifecycle_proof_sha256
                ),
                "wave_index": self.wave_index,
                "gpu_uuids": self.gpu_uuids,
                "start_ns": self.start_ns,
                "finish_ns": self.finish_ns,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "materialized_cell_id": self.materialized_cell_id,
            "execution_binding_sha256": self.execution_binding_sha256,
            "execution_proof": self.execution_proof.to_dict(),
            "execution_proof_payload_sha256": (self.execution_proof_payload_sha256),
            "execution_control_envelope_sha256": (
                self.execution_control_envelope_sha256
            ),
            "execution_replay_reservation": (
                self.execution_replay_reservation.to_dict()
            ),
            "native_run_binding": _run_binding_to_dict(self.native_run_binding),
            "lifecycle_proof": self.lifecycle_proof.to_dict(),
            "verified_lifecycle_proof_sha256": (self.verified_lifecycle_proof_sha256),
            "raw_timing_sha256": self.raw_timing_sha256,
            "live_run_receipt_sha256": self.live_run_receipt_sha256,
            "native_result_proof_sha256": self.native_result_proof_sha256,
            "run_binding_sha256": self.run_binding_sha256,
            "control_envelope_sha256": self.control_envelope_sha256,
            "lifecycle_replay_reservation": (
                self.lifecycle_replay_reservation.to_dict()
            ),
            "telemetry_detail": self.telemetry_detail,
            "gpu_uuids": list(self.gpu_uuids),
            "phase_edges_ns": dict(self.phase_edges_ns),
            "phase_durations_ns": dict(self.phase_durations_ns),
            "wave_index": self.wave_index,
            "schedule_assignment_sha256": self.schedule_assignment_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> LifecycleGpuHourObservation:
        fields = {
            "materialized_cell_id",
            "execution_binding_sha256",
            "execution_proof",
            "execution_proof_payload_sha256",
            "execution_control_envelope_sha256",
            "execution_replay_reservation",
            "native_run_binding",
            "lifecycle_proof",
            "verified_lifecycle_proof_sha256",
            "raw_timing_sha256",
            "live_run_receipt_sha256",
            "native_result_proof_sha256",
            "run_binding_sha256",
            "control_envelope_sha256",
            "lifecycle_replay_reservation",
            "telemetry_detail",
            "gpu_uuids",
            "phase_edges_ns",
            "phase_durations_ns",
            "wave_index",
            "schedule_assignment_sha256",
        }
        row = _strict_dict("lifecycle GPU-hour observation", value, fields)
        declared_assignment = row.pop("schedule_assignment_sha256")
        gpu_uuids = row.pop("gpu_uuids")
        edges = row.pop("phase_edges_ns")
        durations = row.pop("phase_durations_ns")
        if (
            type(gpu_uuids) is not list
            or type(edges) is not dict
            or type(durations) is not dict
        ):
            raise TypeError("lifecycle GPU-hour collections have invalid JSON types")
        if set(edges) != set(_PHASE_EDGE_NAMES) or set(durations) != set(
            _PHASE_DURATION_NAMES
        ):
            raise ValueError("lifecycle GPU-hour phase fields differ")
        native_run_binding = _run_binding_from_dict(row.pop("native_run_binding"))
        execution_proof = CanonicalJsonProofBinding.from_dict(
            row.pop("execution_proof")
        )
        execution_reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("execution_replay_reservation")
        )
        lifecycle_proof = CanonicalJsonProofBinding.from_dict(
            row.pop("lifecycle_proof")
        )
        lifecycle_reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("lifecycle_replay_reservation")
        )
        observation = cls(
            **row,
            native_run_binding=native_run_binding,
            execution_proof=execution_proof,
            execution_replay_reservation=execution_reservation,
            lifecycle_proof=lifecycle_proof,
            lifecycle_replay_reservation=lifecycle_reservation,
            gpu_uuids=tuple(gpu_uuids),
            phase_edges_ns=tuple((name, edges[name]) for name in _PHASE_EDGE_NAMES),
            phase_durations_ns=tuple(
                (name, durations[name]) for name in _PHASE_DURATION_NAMES
            ),
        )
        if observation.schedule_assignment_sha256 != declared_assignment:
            raise ValueError("lifecycle GPU-hour schedule assignment digest differs")
        return observation


@dataclass(frozen=True)
class LifecycleGpuHourSourceManifest:
    """Immutable proof manifest used as the estimate's non-circular source."""

    schema_version: Literal[1]
    kind: Literal["lifecycle_gpu_hour_source_manifest"]
    protocol_sha256: str
    protocol_lock_sha256: str
    runtime_authority_member_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    inventory_gpu_count: Literal[2]
    hardware_envelope_sha256: str
    observations: tuple[LifecycleGpuHourObservation, ...]
    schedule_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lifecycle_gpu_hour_source_manifest"
            or self.protocol_sha256 != FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256
        ):
            raise ValueError("lifecycle GPU-hour source schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority member", self.runtime_authority_member_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("schedule", self.schedule_sha256),
        ):
            _sha256(f"lifecycle GPU-hour {label}", digest)
        if self.inventory_gpu_count != 2:
            raise ValueError("formal GPU-hour source requires exactly two GPUs")
        keys = tuple(
            (row.wave_index, row.materialized_cell_id) for row in self.observations
        )
        if (
            not keys
            or keys != tuple(sorted(keys))
            or len({row.materialized_cell_id for row in self.observations})
            != len(self.observations)
        ):
            raise ValueError("lifecycle GPU-hour observations are not canonical")
        evidence_sets = (
            tuple(row.execution_binding_sha256 for row in self.observations),
            tuple(row.execution_proof.raw_sha256 for row in self.observations),
            tuple(row.execution_control_envelope_sha256 for row in self.observations),
            tuple(
                row.execution_replay_reservation.raw_sha256 for row in self.observations
            ),
            tuple(row.verified_lifecycle_proof_sha256 for row in self.observations),
            tuple(row.raw_timing_sha256 for row in self.observations),
            tuple(row.live_run_receipt_sha256 for row in self.observations),
            tuple(row.native_result_proof_sha256 for row in self.observations),
            tuple(row.control_envelope_sha256 for row in self.observations),
            tuple(
                row.lifecycle_replay_reservation.raw_sha256 for row in self.observations
            ),
            tuple(row.native_run_binding.run_nonce_sha256 for row in self.observations),
        )
        if any(len(values) != len(set(values)) for values in evidence_sets):
            raise ValueError("lifecycle GPU-hour source reuses run/control evidence")
        waves = sorted({row.wave_index for row in self.observations})
        if waves != list(range(len(waves))):
            raise ValueError("lifecycle GPU-hour waves are not contiguous")
        for wave in waves:
            rows = tuple(row for row in self.observations if row.wave_index == wave)
            if sum(row.gang_gpu_count for row in rows) > self.inventory_gpu_count:
                raise ValueError("lifecycle GPU-hour wave exceeds two GPUs")
            gpus = tuple(gpu for row in rows for gpu in row.gpu_uuids)
            if len(gpus) != len(set(gpus)):
                raise ValueError("lifecycle GPU-hour wave reuses a GPU")
        if self.schedule_sha256 != _schedule_sha256(self.observations):
            raise ValueError("lifecycle GPU-hour schedule differs from observations")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "runtime_authority_member_sha256": self.runtime_authority_member_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "inventory_sha256": self.inventory_sha256,
            "inventory_gpu_count": self.inventory_gpu_count,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "observations": [row.to_dict() for row in self.observations],
            "schedule_sha256": self.schedule_sha256,
            "manifest_sha256": self.sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "protocol_sha256": self.protocol_sha256,
                "protocol_lock_sha256": self.protocol_lock_sha256,
                "runtime_authority_member_sha256": (
                    self.runtime_authority_member_sha256
                ),
                "materialization_receipt_sha256": (self.materialization_receipt_sha256),
                "inventory_sha256": self.inventory_sha256,
                "inventory_gpu_count": self.inventory_gpu_count,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "observations": tuple(row.to_dict() for row in self.observations),
                "schedule_sha256": self.schedule_sha256,
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> LifecycleGpuHourSourceManifest:
        fields = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "protocol_lock_sha256",
            "runtime_authority_member_sha256",
            "materialization_receipt_sha256",
            "inventory_sha256",
            "inventory_gpu_count",
            "hardware_envelope_sha256",
            "observations",
            "schedule_sha256",
            "manifest_sha256",
        }
        row = _strict_dict("lifecycle GPU-hour source manifest", value, fields)
        declared = row.pop("manifest_sha256")
        raw_observations = row.pop("observations")
        if type(raw_observations) is not list:
            raise TypeError("lifecycle GPU-hour observations must be an array")
        manifest = cls(
            **row,
            observations=tuple(
                LifecycleGpuHourObservation.from_dict(item) for item in raw_observations
            ),
        )
        if manifest.sha256 != declared:
            raise ValueError("lifecycle GPU-hour source digest differs")
        return manifest


@dataclass(frozen=True)
class PreflightGpuHourLifecycleProofInput:
    """One path-bound lifecycle proof for an exact interference cell."""

    materialized_cell_id: str
    lifecycle_proof_artifact_path: str

    def __post_init__(self) -> None:
        _sha256("preflight GPU-hour materialized cell", self.materialized_cell_id)
        _absolute_path(
            "preflight GPU-hour lifecycle proof",
            self.lifecycle_proof_artifact_path,
        )


@dataclass(frozen=True)
class PreflightGpuHourObservation:
    """First-party timing and replay identity for one mandatory preflight row."""

    materialized_cell_id: str
    registry_cell_id: str
    phase_kind: Literal["compile", "exactness", "interference"]
    timing_proof: CanonicalJsonProofBinding
    timing_authority_sha256: str
    execution_identity_sha256: str
    control_envelope_sha256: str
    replay_reservation: ChallengeReplayReservationBinding
    gpu_uuids: tuple[str, ...]
    process_started_ns: int
    process_finished_ns: int
    gpu_released_ns: int
    evidence_finished_ns: int
    wave_index: int

    def __post_init__(self) -> None:
        for label, digest in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
            ("timing authority", self.timing_authority_sha256),
            ("execution identity", self.execution_identity_sha256),
            ("control envelope", self.control_envelope_sha256),
        ):
            _sha256(f"preflight GPU-hour {label}", digest)
        if self.phase_kind not in {"compile", "exactness", "interference"}:
            raise ValueError("preflight GPU-hour phase is unsupported")
        if type(self.timing_proof) is not CanonicalJsonProofBinding:
            raise TypeError("preflight GPU-hour timing proof is not path-bound")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("preflight GPU-hour replay reservation is not exact")
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) not in {1, 2}
            or len(set(self.gpu_uuids)) != len(self.gpu_uuids)
            or any(not value.startswith("GPU-") for value in self.gpu_uuids)
        ):
            raise ValueError("preflight GPU-hour GPU assignment is invalid")
        if (
            type(self.process_started_ns) is not int
            or type(self.process_finished_ns) is not int
            or type(self.gpu_released_ns) is not int
            or type(self.evidence_finished_ns) is not int
            or not (
                0
                < self.process_started_ns
                <= self.process_finished_ns
                <= self.gpu_released_ns
                <= self.evidence_finished_ns
            )
        ):
            raise ValueError("preflight GPU-hour phase timing is invalid")
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise ValueError("preflight GPU-hour wave index is invalid")

    @cached_property
    def schedule_assignment_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "preflight_gpu_hour_schedule_assignment",
                "materialized_cell_id": self.materialized_cell_id,
                "phase_kind": self.phase_kind,
                "timing_authority_sha256": self.timing_authority_sha256,
                "execution_identity_sha256": self.execution_identity_sha256,
                "gpu_uuids": self.gpu_uuids,
                "process_started_ns": self.process_started_ns,
                "process_finished_ns": self.process_finished_ns,
                "gpu_released_ns": self.gpu_released_ns,
                "evidence_finished_ns": self.evidence_finished_ns,
                "wave_index": self.wave_index,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "materialized_cell_id": self.materialized_cell_id,
            "registry_cell_id": self.registry_cell_id,
            "phase_kind": self.phase_kind,
            "timing_proof": self.timing_proof.to_dict(),
            "timing_authority_sha256": self.timing_authority_sha256,
            "execution_identity_sha256": self.execution_identity_sha256,
            "control_envelope_sha256": self.control_envelope_sha256,
            "replay_reservation": self.replay_reservation.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
            "process_started_ns": self.process_started_ns,
            "process_finished_ns": self.process_finished_ns,
            "gpu_released_ns": self.gpu_released_ns,
            "evidence_finished_ns": self.evidence_finished_ns,
            "wave_index": self.wave_index,
            "schedule_assignment_sha256": self.schedule_assignment_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_dict(
            "preflight GPU-hour observation",
            value,
            {
                *cls.__dataclass_fields__,
                "schedule_assignment_sha256",
            },
        )
        declared = row.pop("schedule_assignment_sha256")
        gpus = row.pop("gpu_uuids")
        if type(gpus) is not list:
            raise TypeError("preflight GPU-hour GPUs must be an array")
        timing_proof = CanonicalJsonProofBinding.from_dict(row.pop("timing_proof"))
        replay_reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("replay_reservation")
        )
        observation = cls(
            **row,
            timing_proof=timing_proof,
            replay_reservation=replay_reservation,
            gpu_uuids=tuple(gpus),
        )
        if observation.schedule_assignment_sha256 != declared:
            raise ValueError("preflight GPU-hour schedule assignment differs")
        return observation


def _preflight_schedule_sha256(
    observations: tuple[PreflightGpuHourObservation, ...],
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "preflight_gpu_hour_reconstructed_schedule",
            "assignments": tuple(
                row.schedule_assignment_sha256 for row in observations
            ),
            "waves": tuple(
                (
                    wave,
                    tuple(
                        row.materialized_cell_id
                        for row in observations
                        if row.wave_index == wave
                    ),
                )
                for wave in sorted({row.wave_index for row in observations})
            ),
        }
    )


@dataclass(frozen=True)
class PreflightGpuHourSourceManifest:
    """Durable non-serving budget authority for all ten preflight cells."""

    schema_version: Literal[1]
    kind: Literal["preflight_gpu_hour_source_manifest"]
    protocol_sha256: str
    protocol_lock_sha256: str
    runtime_authority_member_sha256: str
    materialization_receipt_sha256: str
    stage_coverage_receipt_sha256: str
    final_evidence_sha256: str
    remote_raw_receipt: CanonicalJsonProofBinding
    source_authority: object
    activation_sha256: str
    pointer_coverage_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    observations: tuple[PreflightGpuHourObservation, ...]
    schedule_sha256: str

    def __post_init__(self) -> None:
        from lightcone_spec.experiments.preflight_authority import (
            PreflightExecutionSourceAuthority,
        )

        if (
            self.schema_version != 1
            or self.kind != "preflight_gpu_hour_source_manifest"
            or self.protocol_sha256 != PREFLIGHT_GPU_HOUR_PROTOCOL_SHA256
        ):
            raise ValueError("preflight GPU-hour source schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_member_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("stage coverage", self.stage_coverage_receipt_sha256),
            ("final evidence", self.final_evidence_sha256),
            ("activation", self.activation_sha256),
            ("pointer coverage", self.pointer_coverage_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("schedule", self.schedule_sha256),
        ):
            _sha256(f"preflight GPU-hour {label}", digest)
        if type(self.remote_raw_receipt) is not CanonicalJsonProofBinding:
            raise TypeError("preflight GPU-hour raw receipt is not path-bound")
        if type(self.source_authority) is not PreflightExecutionSourceAuthority:
            raise TypeError("preflight GPU-hour source authority is not exact")
        keys = tuple(
            (row.wave_index, row.materialized_cell_id) for row in self.observations
        )
        if (
            len(self.observations) != 10
            or keys != tuple(sorted(keys))
            or len({row.materialized_cell_id for row in self.observations}) != 10
            or tuple(sorted(row.phase_kind for row in self.observations)).count(
                "compile"
            )
            != 1
            or tuple(sorted(row.phase_kind for row in self.observations)).count(
                "exactness"
            )
            != 1
            or tuple(sorted(row.phase_kind for row in self.observations)).count(
                "interference"
            )
            != 8
        ):
            raise ValueError("preflight GPU-hour observation coverage is not 1+1+8")
        unique_sets = (
            tuple(row.timing_proof.raw_sha256 for row in self.observations),
            tuple(row.timing_authority_sha256 for row in self.observations),
            tuple(row.execution_identity_sha256 for row in self.observations),
            tuple(row.control_envelope_sha256 for row in self.observations),
            tuple(
                row.replay_reservation.reservation_sha256 for row in self.observations
            ),
            tuple(
                challenge
                for row in self.observations
                for challenge in row.replay_reservation.challenge_sha256s
            ),
        )
        if any(len(values) != len(set(values)) for values in unique_sets):
            raise ValueError("preflight GPU-hour source reuses proof/control evidence")
        if self.schedule_sha256 != _preflight_schedule_sha256(self.observations):
            raise ValueError("preflight GPU-hour schedule differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "runtime_authority_member_sha256": (self.runtime_authority_member_sha256),
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "stage_coverage_receipt_sha256": self.stage_coverage_receipt_sha256,
            "final_evidence_sha256": self.final_evidence_sha256,
            "remote_raw_receipt": self.remote_raw_receipt.to_dict(),
            "source_authority": self.source_authority.to_dict(),
            "activation_sha256": self.activation_sha256,
            "pointer_coverage_sha256": self.pointer_coverage_sha256,
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "observations": [row.to_dict() for row in self.observations],
            "schedule_sha256": self.schedule_sha256,
        }
        if include_sha256:
            value["manifest_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        from lightcone_spec.experiments.preflight_authority import (
            PreflightExecutionSourceAuthority,
        )

        row = _strict_dict(
            "preflight GPU-hour source",
            value,
            {"manifest_sha256", *cls.__dataclass_fields__},
        )
        declared = row.pop("manifest_sha256")
        raw_observations = row.pop("observations")
        if type(raw_observations) is not list:
            raise TypeError("preflight GPU-hour observations must be an array")
        remote_raw_receipt = CanonicalJsonProofBinding.from_dict(
            row.pop("remote_raw_receipt")
        )
        source_authority = PreflightExecutionSourceAuthority.from_dict(
            row.pop("source_authority")
        )
        manifest = cls(
            **row,
            remote_raw_receipt=remote_raw_receipt,
            source_authority=source_authority,
            observations=tuple(
                PreflightGpuHourObservation.from_dict(item) for item in raw_observations
            ),
        )
        if manifest.sha256 != declared:
            raise ValueError("preflight GPU-hour source digest differs")
        return manifest


_VERIFIED_PROSPECTIVE_GPU_HOUR_AUTHORITY_SEAL = object()


@dataclass(frozen=True, init=False)
class VerifiedProspectiveGpuHourAuthority:
    """Private-sealed pilot-to-final source authority.

    Only the stage-specific reducers below may construct this value.  The
    selected block prefix comes from a verified signed power receipt; callers
    cannot pass a block count or duration scalar.
    """

    stage: str
    signed_authority_sha256: str
    signed_authority_challenge_sha256: str
    pilot_materialization_receipt_sha256: str
    final_materialization_receipt_sha256: str
    selected_final_prefix: tuple[int, ...]
    _construction_seal: object

    def __init__(
        self,
        *,
        stage: str,
        signed_authority_sha256: str,
        signed_authority_challenge_sha256: str,
        pilot_materialization_receipt_sha256: str,
        final_materialization_receipt_sha256: str,
        selected_final_prefix: tuple[int, ...],
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _VERIFIED_PROSPECTIVE_GPU_HOUR_AUTHORITY_SEAL:
            raise TypeError("prospective GPU-hour authority is verifier-constructed")
        if stage not in {"E3b", "E5", "E6", "E0"}:
            raise ValueError("prospective GPU-hour stage is unsupported")
        for label, digest in (
            ("signed authority", signed_authority_sha256),
            ("signed challenge", signed_authority_challenge_sha256),
            ("pilot materialization", pilot_materialization_receipt_sha256),
            ("final materialization", final_materialization_receipt_sha256),
        ):
            _sha256(f"prospective GPU-hour {label}", digest)
        if (
            type(selected_final_prefix) is not tuple
            or not 12 <= len(selected_final_prefix) <= 20
            or selected_final_prefix != tuple(range(4, 4 + len(selected_final_prefix)))
        ):
            raise ValueError("prospective GPU-hour final prefix is not exact")
        for name, value in (
            ("stage", stage),
            ("signed_authority_sha256", signed_authority_sha256),
            (
                "signed_authority_challenge_sha256",
                signed_authority_challenge_sha256,
            ),
            (
                "pilot_materialization_receipt_sha256",
                pilot_materialization_receipt_sha256,
            ),
            (
                "final_materialization_receipt_sha256",
                final_materialization_receipt_sha256,
            ),
            ("selected_final_prefix", selected_final_prefix),
            ("_construction_seal", _construction_seal),
        ):
            object.__setattr__(self, name, value)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "verified_prospective_gpu_hour_authority",
                "stage": self.stage,
                "signed_authority_sha256": self.signed_authority_sha256,
                "signed_authority_challenge_sha256": (
                    self.signed_authority_challenge_sha256
                ),
                "pilot_materialization_receipt_sha256": (
                    self.pilot_materialization_receipt_sha256
                ),
                "final_materialization_receipt_sha256": (
                    self.final_materialization_receipt_sha256
                ),
                "selected_final_prefix": self.selected_final_prefix,
            }
        )


@dataclass(frozen=True)
class E5FailureGpuHourProofInput:
    """Ephemeral paths joining one public failure subject to its lifecycle."""

    failure_execution_rebuild_input_path: str
    lifecycle_proof_artifact_path: str

    def __post_init__(self) -> None:
        _absolute_path(
            "E5 failure execution rebuild input",
            self.failure_execution_rebuild_input_path,
        )
        _absolute_path(
            "E5 failure lifecycle cost proof",
            self.lifecycle_proof_artifact_path,
        )
        if self.failure_execution_rebuild_input_path == (
            self.lifecycle_proof_artifact_path
        ):
            raise ValueError("E5 failure GPU-hour input aliases its proof paths")


@dataclass(frozen=True)
class E5FailureGpuHourObservation:
    """Durable exact cost projection for one correctness-only E5 row."""

    materialized_cell_id: str
    failure_execution_rebuild_input: CanonicalJsonProofBinding
    lifecycle_proof: CanonicalJsonProofBinding
    projection: FormalE5FailureLifecycleCostProjection
    control_envelope_sha256: str
    replay_reservation: ChallengeReplayReservationBinding
    wave_index: int

    def __post_init__(self) -> None:
        from lightcone_spec.orchestration.formal_failure_physical import (
            FormalE5FailureLifecycleCostProjection,
        )

        _sha256("E5 failure GPU-hour cell", self.materialized_cell_id)
        if (
            type(self.failure_execution_rebuild_input) is not CanonicalJsonProofBinding
            or type(self.lifecycle_proof) is not CanonicalJsonProofBinding
        ):
            raise TypeError("E5 failure GPU-hour sources must be path-bound")
        if type(self.projection) is not FormalE5FailureLifecycleCostProjection:
            raise TypeError("E5 failure GPU-hour projection is not exact")
        self.projection.__post_init__()
        _sha256("E5 failure GPU-hour control", self.control_envelope_sha256)
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("E5 failure GPU-hour replay reservation is not exact")
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise ValueError("E5 failure GPU-hour wave index is invalid")
        if (
            self.projection.materialized_cell_id != self.materialized_cell_id
            or self.lifecycle_proof.semantic_sha256
            != self.projection.proof_artifact_sha256
        ):
            raise ValueError("E5 failure GPU-hour projection identity differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "materialized_cell_id": self.materialized_cell_id,
            "failure_execution_rebuild_input": (
                self.failure_execution_rebuild_input.to_dict()
            ),
            "lifecycle_proof": self.lifecycle_proof.to_dict(),
            "projection": self.projection.to_dict(),
            "control_envelope_sha256": self.control_envelope_sha256,
            "replay_reservation": self.replay_reservation.to_dict(),
            "wave_index": self.wave_index,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        from lightcone_spec.orchestration.formal_failure_physical import (
            FormalE5FailureLifecycleCostProjection,
        )

        row = _strict_dict(
            "E5 failure GPU-hour observation",
            value,
            set(cls.__dataclass_fields__),
        )
        raw_projection = row.pop("projection")
        if type(raw_projection) is not dict or set(raw_projection) != set(
            FormalE5FailureLifecycleCostProjection.__dataclass_fields__
        ):
            raise ValueError("E5 failure GPU-hour projection fields differ")
        projection = dict(raw_projection)
        gpu_uuids = projection.pop("gpu_uuids")
        if type(gpu_uuids) is not list:
            raise TypeError("E5 failure GPU-hour projection GPUs must be an array")
        raw_rebuild_input = row.pop("failure_execution_rebuild_input")
        raw_lifecycle_proof = row.pop("lifecycle_proof")
        raw_reservation = row.pop("replay_reservation")
        return cls(
            **row,
            failure_execution_rebuild_input=(
                CanonicalJsonProofBinding.from_dict(raw_rebuild_input)
            ),
            lifecycle_proof=CanonicalJsonProofBinding.from_dict(raw_lifecycle_proof),
            projection=FormalE5FailureLifecycleCostProjection(
                **projection,
                gpu_uuids=tuple(gpu_uuids),
            ),
            replay_reservation=ChallengeReplayReservationBinding.from_dict(
                raw_reservation
            ),
        )


@dataclass(frozen=True)
class E5FailureGpuHourSourceManifest:
    """Actual-only GPU-hour authority for every exact E5 failure run."""

    schema_version: Literal[1]
    kind: Literal["e5_failure_gpu_hour_source_manifest"]
    protocol_sha256: str
    protocol_lock_sha256: str
    runtime_authority_member_sha256: str
    runtime_authority_manifest_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    registry_sha256: str
    root_manifest_sha256: str
    hardware_envelope_sha256: str
    observations: tuple[E5FailureGpuHourObservation, ...]
    cost: ProspectiveGpuHourCost
    schedule_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "e5_failure_gpu_hour_source_manifest"
            or self.protocol_sha256 != E5_FAILURE_GPU_HOUR_SOURCE_PROTOCOL_SHA256
        ):
            raise ValueError("E5 failure GPU-hour source schema differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime member", self.runtime_authority_member_sha256),
            ("runtime manifest", self.runtime_authority_manifest_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("registry", self.registry_sha256),
            ("release root", self.root_manifest_sha256),
            ("hardware", self.hardware_envelope_sha256),
            ("schedule", self.schedule_sha256),
        ):
            _sha256(f"E5 failure GPU-hour {label}", digest)
        if (
            type(self.observations) is not tuple
            or len(self.observations) != 264
            or any(
                type(row) is not E5FailureGpuHourObservation
                for row in self.observations
            )
            or tuple(row.wave_index for row in self.observations) != tuple(range(264))
            or len({row.materialized_cell_id for row in self.observations}) != 264
        ):
            raise ValueError("E5 failure GPU-hour source is not exact 264 coverage")
        if (
            type(self.cost) is not ProspectiveGpuHourCost
            or self.cost.category != "actual_one_shot"
            or self.cost.cell_count != 264
            or self.cost.retry_reserve_gpu_ns != 0
            or self.cost.profile_reserve_gpu_ns != 0
        ):
            raise ValueError("E5 failure GPU-hour actual cost differs")
        if any(
            row.projection.inventory_sha256 != self.inventory_sha256
            or row.projection.registry_sha256 != self.registry_sha256
            or row.projection.root_manifest_sha256 != self.root_manifest_sha256
            or row.projection.materialized_cell_id != row.materialized_cell_id
            for row in self.observations
        ):
            raise ValueError("E5 failure GPU-hour projection lineage differs")
        unique_sets = (
            tuple(row.materialized_cell_id for row in self.observations),
            tuple(
                row.failure_execution_rebuild_input.absolute_path
                for row in self.observations
            ),
            tuple(
                row.failure_execution_rebuild_input.semantic_sha256
                for row in self.observations
            ),
            tuple(row.lifecycle_proof.absolute_path for row in self.observations),
            tuple(row.lifecycle_proof.semantic_sha256 for row in self.observations),
            tuple(
                row.projection.raw_lifecycle_receipt_sha256 for row in self.observations
            ),
            tuple(row.projection.failure_subject_sha256 for row in self.observations),
            tuple(
                row.projection.formal_failure_execution_binding_sha256
                for row in self.observations
            ),
            tuple(
                row.projection.serving_execution_binding_sha256
                for row in self.observations
            ),
            tuple(
                row.projection.serving_execution_plan_sha256
                for row in self.observations
            ),
            tuple(row.projection.assignment_sha256 for row in self.observations),
            tuple(row.projection.run_nonce_sha256 for row in self.observations),
            tuple(
                row.projection.formal_launch_admission_sha256
                for row in self.observations
            ),
            tuple(
                row.projection.formal_launch_consumption_sha256
                for row in self.observations
            ),
            tuple(
                row.projection.budget_consumption_sha256 for row in self.observations
            ),
            tuple(
                row.projection.raw_failure_terminal_sha256 for row in self.observations
            ),
            tuple(row.projection.recovery_receipt_sha256 for row in self.observations),
            tuple(row.control_envelope_sha256 for row in self.observations),
            tuple(
                row.replay_reservation.reservation_sha256 for row in self.observations
            ),
            tuple(
                challenge
                for row in self.observations
                for challenge in row.replay_reservation.challenge_sha256s
            ),
        )
        if any(len(values) != len(set(values)) for values in unique_sets):
            raise ValueError(
                "E5 failure GPU-hour source reuses proof/control authority"
            )
        if self.schedule_sha256 != _e5_failure_schedule_sha256(self.observations):
            raise ValueError("E5 failure GPU-hour schedule differs")
        expected = _e5_failure_cost(self.observations)
        if self.cost != expected:
            raise ValueError("E5 failure GPU-hour cost differs from projections")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        from dataclasses import asdict

        value = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in {"observations", "cost"}
        }
        value["observations"] = [row.to_dict() for row in self.observations]
        value["cost"] = asdict(self.cost)
        if include_sha256:
            value["manifest_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_dict(
            "E5 failure GPU-hour source",
            value,
            {*cls.__dataclass_fields__, "manifest_sha256"},
        )
        declared = _sha256("E5 failure GPU-hour source", row.pop("manifest_sha256"))
        raw_observations = row.pop("observations")
        if type(raw_observations) is not list:
            raise TypeError("E5 failure GPU-hour observations must be an array")
        raw_cost = row.pop("cost")
        if type(raw_cost) is not dict:
            raise TypeError("E5 failure GPU-hour cost must be an object")
        manifest = cls(
            **row,
            observations=tuple(
                E5FailureGpuHourObservation.from_dict(item) for item in raw_observations
            ),
            cost=ProspectiveGpuHourCost(**raw_cost),
        )
        if manifest.sha256 != declared:
            raise ValueError("E5 failure GPU-hour source digest differs")
        return manifest


def _seal_prospective_authority(
    *,
    stage: str,
    signed_authority_sha256: str,
    signed_authority_challenge_sha256: str,
    pilot_materialization: StageMaterializationReceipt,
    final_materialization: StageMaterializationReceipt,
    selected_final_prefix: tuple[int, ...],
) -> VerifiedProspectiveGpuHourAuthority:
    if (
        pilot_materialization.stage != stage
        or final_materialization.stage != stage
        or final_materialization.source_decision_sha256 != signed_authority_sha256
    ):
        raise ValueError("prospective GPU-hour materialization/source differs")
    final_blocks = tuple(
        sorted(
            {
                block
                for cell in final_materialization.cells
                if type(block := dict(cell.dimensions).get("block")) is int
            }
        )
    )
    if final_blocks != selected_final_prefix:
        raise ValueError("prospective GPU-hour final block prefix differs")
    return VerifiedProspectiveGpuHourAuthority(
        stage=stage,
        signed_authority_sha256=signed_authority_sha256,
        signed_authority_challenge_sha256=signed_authority_challenge_sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        final_materialization_receipt_sha256=final_materialization.sha256,
        selected_final_prefix=selected_final_prefix,
        _construction_seal=_VERIFIED_PROSPECTIVE_GPU_HOUR_AUTHORITY_SEAL,
    )


def verify_registered_prospective_gpu_hour_authority(
    *,
    registry_receipt: object,
    pilot_materialization: StageMaterializationReceipt,
    final_materialization: StageMaterializationReceipt,
    current_ns: int,
) -> VerifiedProspectiveGpuHourAuthority:
    """Rebuild the private authority from one verified registry power wrapper.

    The formal registry already owns signature/replay validation for the
    proof-derived power receipt and binds that exact signed receipt to the main
    materialization.  This adapter deliberately accepts no caller-authored
    prefix or digest: it revalidates the durable registry prefix, locates its
    one typed stage power wrapper, and joins it to the path-reopened excluded
    pilot materialization.
    """

    from lightcone_spec.experiments.downstream_stage_authority import (
        SignedE3bPowerPrefixReceipt,
        SignedE5PowerAndAnchorReceipt,
    )
    from lightcone_spec.experiments.e0_stage_authority import (
        SignedE0PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        SignedE6PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("prospective GPU hours require a formal registry receipt")
    if (
        type(pilot_materialization) is not StageMaterializationReceipt
        or type(final_materialization) is not StageMaterializationReceipt
    ):
        raise TypeError("prospective GPU hours require exact materializations")
    if type(current_ns) is not int or current_ns < 1:
        raise ValueError("prospective GPU-hour verification time is invalid")
    registry_receipt.revalidate(current_ns=current_ns)
    protocol_lock = registry_receipt.signed_protocol_lock.payload
    stage = final_materialization.stage
    if stage == "E3b":
        signed_rows = registry_receipt.cumulative_signed_e3b_power_prefixes
        expected_type = SignedE3bPowerPrefixReceipt
    elif stage == "E5":
        signed_rows = registry_receipt.cumulative_signed_e5_power_and_anchor_prefixes
        expected_type = SignedE5PowerAndAnchorReceipt
    elif stage == "E6":
        signed_rows = registry_receipt.cumulative_signed_e6_power_prefixes
        expected_type = SignedE6PowerPrefixReceipt
    elif stage == "E0":
        signed_rows = registry_receipt.cumulative_signed_e0_power_prefixes
        expected_type = SignedE0PowerPrefixReceipt
    else:
        raise ValueError("prospective GPU-hour stage is unsupported")
    if len(signed_rows) != 1 or type(signed_rows[0]) is not expected_type:
        raise ValueError("prospective GPU-hour registry power source is not exact")
    signed_power = signed_rows[0]
    if signed_power.sha256 != final_materialization.source_decision_sha256:
        raise ValueError("prospective GPU-hour signed power differs from final source")

    registered_materializations = tuple(
        row.payload
        for row in registry_receipt.cumulative_signed_materializations
        if row.payload.sha256 == final_materialization.sha256
    )
    if registered_materializations != (final_materialization,):
        raise ValueError("prospective GPU-hour final registry lineage is not exact")

    power = signed_power.payload
    power.__post_init__()
    pilot_blocks = tuple(
        sorted(
            {
                block
                for cell in pilot_materialization.cells
                if type(block := dict(cell.dimensions).get("block")) is int
            }
        )
    )
    if (
        pilot_materialization.stage != stage
        or pilot_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or final_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or power.protocol_lock_sha256 != protocol_lock.sha256
        or power.registry_sha256 != protocol_lock.registry_sha256
        or power.inventory_sha256 != registry_receipt.inventory_sha256
        or power.pilot_materialization_receipt_sha256 != pilot_materialization.sha256
        or pilot_blocks != tuple(range(4))
    ):
        raise ValueError("prospective GPU-hour registered authority differs")
    return _seal_prospective_authority(
        stage=stage,
        signed_authority_sha256=signed_power.sha256,
        signed_authority_challenge_sha256=signed_power.challenge.sha256,
        pilot_materialization=pilot_materialization,
        final_materialization=final_materialization,
        selected_final_prefix=power.selected_final_prefix,
    )


def verify_e3b_prospective_gpu_hour_authority(
    *,
    protocol_lock: ProtocolLock,
    signed_power_prefix: object,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: object,
    pilot_evidence_manifest: object,
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    final_materialization: StageMaterializationReceipt,
    policy: object,
    expected_policy_sha256: str,
    now_ns: int,
) -> VerifiedProspectiveGpuHourAuthority:
    from lightcone_spec.experiments.downstream_stage_authority import (
        FormalDownstreamEvidenceManifest,
        SignedE3bPowerPrefixReceipt,
    )
    from lightcone_spec.experiments.stage_materialization import StageCoverageReceipt
    from lightcone_spec.runtime.attestation import TrustedAttesterPolicy

    if type(signed_power_prefix) is not SignedE3bPowerPrefixReceipt:
        raise TypeError("E3b prospective GPU hours require signed power authority")
    if type(pilot_coverage) is not StageCoverageReceipt:
        raise TypeError("E3b prospective GPU hours require exact pilot coverage")
    if type(pilot_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E3b prospective GPU hours require exact pilot evidence")
    if type(policy) is not TrustedAttesterPolicy:
        raise TypeError("E3b prospective GPU hours require exact signer policy")
    power = signed_power_prefix.verify(
        protocol_lock=protocol_lock,
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        manifest=pilot_evidence_manifest,
        execution_bindings=pilot_execution_bindings,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    return _seal_prospective_authority(
        stage="E3b",
        signed_authority_sha256=signed_power_prefix.sha256,
        signed_authority_challenge_sha256=signed_power_prefix.challenge.sha256,
        pilot_materialization=pilot_materialization,
        final_materialization=final_materialization,
        selected_final_prefix=power.selected_final_prefix,
    )


def verify_e5_prospective_gpu_hour_authority(
    *,
    protocol_lock: ProtocolLock,
    signed_power_and_anchor_prefix: object,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: object,
    pilot_evidence_manifest: object,
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    final_materialization: StageMaterializationReceipt,
    policy: object,
    expected_policy_sha256: str,
    now_ns: int,
) -> VerifiedProspectiveGpuHourAuthority:
    from lightcone_spec.experiments.downstream_stage_authority import (
        FormalDownstreamEvidenceManifest,
        SignedE5PowerAndAnchorReceipt,
    )
    from lightcone_spec.experiments.stage_materialization import StageCoverageReceipt
    from lightcone_spec.runtime.attestation import TrustedAttesterPolicy

    if type(signed_power_and_anchor_prefix) is not SignedE5PowerAndAnchorReceipt:
        raise TypeError("E5 prospective GPU hours require signed power authority")
    if type(pilot_coverage) is not StageCoverageReceipt:
        raise TypeError("E5 prospective GPU hours require exact pilot coverage")
    if type(pilot_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E5 prospective GPU hours require exact pilot evidence")
    if type(policy) is not TrustedAttesterPolicy:
        raise TypeError("E5 prospective GPU hours require exact signer policy")
    power = signed_power_and_anchor_prefix.verify(
        protocol_lock=protocol_lock,
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        manifest=pilot_evidence_manifest,
        execution_bindings=pilot_execution_bindings,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    return _seal_prospective_authority(
        stage="E5",
        signed_authority_sha256=signed_power_and_anchor_prefix.sha256,
        signed_authority_challenge_sha256=(
            signed_power_and_anchor_prefix.challenge.sha256
        ),
        pilot_materialization=pilot_materialization,
        final_materialization=final_materialization,
        selected_final_prefix=power.selected_final_prefix,
    )


def verify_e6_prospective_gpu_hour_authority(
    *,
    protocol_lock: ProtocolLock,
    signed_model_compatibility: object,
    compatibility_sources: tuple[object, ...],
    signed_power_prefix: object,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: object,
    pilot_evidence_manifest: object,
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    final_materialization: StageMaterializationReceipt,
    policy: object,
    expected_policy_sha256: str,
    now_ns: int,
) -> VerifiedProspectiveGpuHourAuthority:
    from lightcone_spec.experiments.downstream_stage_authority import (
        FormalDownstreamEvidenceManifest,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        E6NextnModelAuthorityInput,
        SignedE6ModelCompatibilityReceipt,
        SignedE6PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.stage_materialization import StageCoverageReceipt
    from lightcone_spec.runtime.attestation import TrustedAttesterPolicy

    if type(signed_model_compatibility) is not SignedE6ModelCompatibilityReceipt:
        raise TypeError("E6 prospective GPU hours require signed compatibility")
    if type(signed_power_prefix) is not SignedE6PowerPrefixReceipt:
        raise TypeError("E6 prospective GPU hours require signed power authority")
    if type(compatibility_sources) is not tuple or any(
        type(row) is not E6NextnModelAuthorityInput for row in compatibility_sources
    ):
        raise TypeError("E6 prospective GPU hours require exact model sources")
    if type(pilot_coverage) is not StageCoverageReceipt:
        raise TypeError("E6 prospective GPU hours require exact pilot coverage")
    if type(pilot_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E6 prospective GPU hours require exact pilot evidence")
    if type(policy) is not TrustedAttesterPolicy:
        raise TypeError("E6 prospective GPU hours require exact signer policy")
    power = signed_power_prefix.verify(
        protocol_lock=protocol_lock,
        signed_model_compatibility=signed_model_compatibility,
        compatibility_sources=compatibility_sources,
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        manifest=pilot_evidence_manifest,
        execution_bindings=pilot_execution_bindings,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    return _seal_prospective_authority(
        stage="E6",
        signed_authority_sha256=signed_power_prefix.sha256,
        signed_authority_challenge_sha256=signed_power_prefix.challenge.sha256,
        pilot_materialization=pilot_materialization,
        final_materialization=final_materialization,
        selected_final_prefix=power.selected_final_prefix,
    )


def verify_e0_prospective_gpu_hour_authority(
    *,
    protocol_lock: ProtocolLock,
    signed_e6_confirmation: object,
    signed_compatibility: object,
    signed_tuning_seals: tuple[object, ...],
    source_authority: object,
    tuning_proof_set: object,
    pilot_proof_set: object,
    signed_power_prefix: object,
    pilot_materialization: StageMaterializationReceipt,
    final_materialization: StageMaterializationReceipt,
    policy: object,
    expected_policy_sha256: str,
    now_ns: int,
) -> VerifiedProspectiveGpuHourAuthority:
    """Deep-open E0's signed 4-pilot to 12--20-block power authority."""

    from lightcone_spec.experiments.e0_stage_authority import (
        E0OnlineSpecSourceAuthority,
        E0OnlineSpecTuningProofSet,
        SignedE0OnlineSpecTuningSeal,
        SignedE0PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        SignedE6ConfirmationReceipt,
    )
    from lightcone_spec.experiments.stage_materialization import (
        SignedE0CompatibilityReceipt,
    )
    from lightcone_spec.runtime.attestation import TrustedAttesterPolicy

    if type(signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
        raise TypeError("E0 prospective GPU hours require signed E6 confirmation")
    if type(signed_compatibility) is not SignedE0CompatibilityReceipt:
        raise TypeError("E0 prospective GPU hours require signed compatibility")
    if type(signed_tuning_seals) is not tuple or any(
        type(row) is not SignedE0OnlineSpecTuningSeal for row in signed_tuning_seals
    ):
        raise TypeError("E0 prospective GPU hours require exact tuning seals")
    if type(source_authority) is not E0OnlineSpecSourceAuthority:
        raise TypeError("E0 prospective GPU hours require source checkout authority")
    if (
        type(tuning_proof_set) is not E0OnlineSpecTuningProofSet
        or type(pilot_proof_set) is not E0OnlineSpecTuningProofSet
    ):
        raise TypeError("E0 prospective GPU hours require exact proof sets")
    if type(signed_power_prefix) is not SignedE0PowerPrefixReceipt:
        raise TypeError("E0 prospective GPU hours require signed power authority")
    if type(policy) is not TrustedAttesterPolicy:
        raise TypeError("E0 prospective GPU hours require exact signer policy")
    if pilot_proof_set.materialization != pilot_materialization:
        raise ValueError("E0 prospective pilot proof set names another materialization")
    power = signed_power_prefix.verify(
        protocol_lock=protocol_lock,
        signed_e6_confirmation=signed_e6_confirmation,
        signed_compatibility=signed_compatibility,
        signed_tuning_seals=signed_tuning_seals,
        source_authority=source_authority,
        tuning_proof_set=tuning_proof_set,
        pilot_proof_set=pilot_proof_set,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    return _seal_prospective_authority(
        stage="E0",
        signed_authority_sha256=signed_power_prefix.sha256,
        signed_authority_challenge_sha256=signed_power_prefix.challenge.sha256,
        pilot_materialization=pilot_materialization,
        final_materialization=final_materialization,
        selected_final_prefix=power.selected_final_prefix,
    )


@dataclass(frozen=True)
class ProspectiveGpuHourCost:
    category: Literal["actual_tuning", "projected_final", "actual_one_shot"]
    cell_count: int
    compute_gpu_ns: int
    provider_base_reserved_gpu_ns: int
    wall_ns: int
    retry_reserve_gpu_ns: int
    profile_reserve_gpu_ns: int
    evidence_reserve_gpu_ns: int

    def __post_init__(self) -> None:
        if self.category not in {
            "actual_tuning",
            "projected_final",
            "actual_one_shot",
        }:
            raise ValueError("prospective GPU-hour cost category is unsupported")
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
                raise ValueError(f"prospective GPU-hour {label} must be non-negative")
        if self.cell_count < 1:
            raise ValueError("prospective GPU-hour cost must cover cells")

    @property
    def reserved_gpu_ns(self) -> int:
        return (
            self.provider_base_reserved_gpu_ns
            + self.retry_reserve_gpu_ns
            + self.profile_reserve_gpu_ns
            + self.evidence_reserve_gpu_ns
        )


def _e5_failure_schedule_sha256(
    observations: tuple[E5FailureGpuHourObservation, ...],
) -> str:
    """Bind the exact chronological, actual-only E5 failure schedule."""

    return content_sha256(
        {
            "schema_version": 1,
            "kind": "e5_failure_gpu_hour_actual_schedule",
            "protocol_sha256": E5_FAILURE_GPU_HOUR_SOURCE_PROTOCOL_SHA256,
            "rows": tuple(
                {
                    "wave_index": row.wave_index,
                    "materialized_cell_id": row.materialized_cell_id,
                    "failure_execution_rebuild_input_sha256": (
                        row.failure_execution_rebuild_input.semantic_sha256
                    ),
                    "lifecycle_proof_sha256": (row.lifecycle_proof.semantic_sha256),
                    "projection_sha256": row.projection.sha256,
                    "control_envelope_sha256": row.control_envelope_sha256,
                    "replay_reservation_sha256": (
                        row.replay_reservation.reservation_sha256
                    ),
                    "execution_started_ns": row.projection.execution_started_ns,
                    "process_exited_ns": row.projection.process_exited_ns,
                    "process_group_empty_checked_ns": (
                        row.projection.process_group_empty_checked_ns
                    ),
                    "gpu_release_ns": row.projection.gpu_release_ns,
                    "evidence_flush_finished_ns": (
                        row.projection.evidence_flush_finished_ns
                    ),
                    "gpu_uuids": row.projection.gpu_uuids,
                }
                for row in observations
            ),
            "schedule": "sequential_actual_only_provider_release_boundary",
        }
    )


def _e5_failure_cost(
    observations: tuple[E5FailureGpuHourObservation, ...],
) -> ProspectiveGpuHourCost:
    if len(observations) != 264:
        raise ValueError("E5 failure GPU-hour cost requires exact 264 rows")
    ordered = tuple(
        sorted(
            observations,
            key=lambda row: (
                row.projection.execution_started_ns,
                row.materialized_cell_id,
            ),
        )
    )
    for prior, following in pairwise(ordered):
        if following.projection.execution_started_ns < prior.projection.gpu_release_ns:
            raise ValueError("E5 failure GPU-hour provider intervals overlap")
    for row in observations:
        projection = row.projection
        provider_wall_ns = projection.gpu_release_ns - projection.execution_started_ns
        evidence_wall_ns = (
            projection.evidence_flush_finished_ns - projection.gpu_release_ns
        )
        if (
            projection.provider_reserved_gpu_ns != provider_wall_ns * 2
            or projection.evidence_gpu_ns != evidence_wall_ns * 2
        ):
            raise ValueError("E5 failure GPU-hour fixed provider charge differs")
    return ProspectiveGpuHourCost(
        category="actual_one_shot",
        cell_count=264,
        compute_gpu_ns=sum(row.projection.compute_gpu_ns for row in observations),
        provider_base_reserved_gpu_ns=sum(
            row.projection.provider_reserved_gpu_ns for row in observations
        ),
        wall_ns=sum(
            row.projection.gpu_release_ns - row.projection.execution_started_ns
            for row in observations
        ),
        retry_reserve_gpu_ns=0,
        profile_reserve_gpu_ns=0,
        evidence_reserve_gpu_ns=sum(
            row.projection.evidence_gpu_ns for row in observations
        ),
    )


@dataclass(frozen=True)
class ProspectiveGpuHourSourceManifest:
    """Typed hybrid of actual pilots, projected finals, and actual one-shots."""

    schema_version: Literal[1]
    kind: Literal["prospective_gpu_hour_source_manifest"]
    protocol_sha256: str
    protocol_lock_sha256: str
    runtime_authority_member_sha256: str
    stage: Literal["E3b", "E5", "E6", "E0"]
    final_materialization_receipt_sha256: str
    pilot_materialization_receipt_sha256: str
    pilot_source_manifest: CanonicalJsonProofBinding
    one_shot_source_manifest: CanonicalJsonProofBinding | None
    prospective_authority_sha256: str
    signed_power_authority_sha256: str
    signed_power_challenge_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    mapping_sha256: str
    costs: tuple[ProspectiveGpuHourCost, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "prospective_gpu_hour_source_manifest"
            or self.stage not in {"E3b", "E5", "E6", "E0"}
            or self.protocol_sha256 != PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256
        ):
            raise ValueError("prospective GPU-hour source schema is unsupported")
        for label, digest in (
            ("protocol", self.protocol_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_member_sha256),
            ("final materialization", self.final_materialization_receipt_sha256),
            ("pilot materialization", self.pilot_materialization_receipt_sha256),
            ("prospective authority", self.prospective_authority_sha256),
            ("signed power authority", self.signed_power_authority_sha256),
            ("signed power challenge", self.signed_power_challenge_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("mapping", self.mapping_sha256),
        ):
            _sha256(f"prospective GPU-hour {label}", digest)
        if type(self.pilot_source_manifest) is not CanonicalJsonProofBinding:
            raise TypeError("prospective GPU-hour pilot source is not path-bound")
        if (
            self.one_shot_source_manifest is not None
            and type(self.one_shot_source_manifest) is not CanonicalJsonProofBinding
        ):
            raise TypeError("prospective GPU-hour one-shot source is not path-bound")
        expected_categories = (
            ("actual_tuning", "projected_final", "actual_one_shot")
            if self.stage in {"E5", "E6"}
            else ("actual_tuning", "projected_final")
        )
        if (
            type(self.costs) is not tuple
            or tuple(row.category for row in self.costs) != expected_categories
            or any(type(row) is not ProspectiveGpuHourCost for row in self.costs)
        ):
            raise ValueError("prospective GPU-hour component coverage is not exact")
        if (self.stage == "E5") != (self.one_shot_source_manifest is not None):
            raise ValueError("only E5 requires a separate actual one-shot source")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "runtime_authority_member_sha256": self.runtime_authority_member_sha256,
            "stage": self.stage,
            "final_materialization_receipt_sha256": (
                self.final_materialization_receipt_sha256
            ),
            "pilot_materialization_receipt_sha256": (
                self.pilot_materialization_receipt_sha256
            ),
            "pilot_source_manifest": self.pilot_source_manifest.to_dict(),
            "one_shot_source_manifest": (
                None
                if self.one_shot_source_manifest is None
                else self.one_shot_source_manifest.to_dict()
            ),
            "prospective_authority_sha256": self.prospective_authority_sha256,
            "signed_power_authority_sha256": self.signed_power_authority_sha256,
            "signed_power_challenge_sha256": self.signed_power_challenge_sha256,
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "mapping_sha256": self.mapping_sha256,
            "costs": [asdict(row) for row in self.costs],
            "manifest_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_dict(
            "prospective GPU-hour source",
            value,
            {
                *cls.__dataclass_fields__,
                "manifest_sha256",
            },
        )
        declared = row.pop("manifest_sha256")
        pilot = CanonicalJsonProofBinding.from_dict(row.pop("pilot_source_manifest"))
        raw_one_shot = row.pop("one_shot_source_manifest")
        one_shot = (
            None
            if raw_one_shot is None
            else CanonicalJsonProofBinding.from_dict(raw_one_shot)
        )
        raw_costs = row.pop("costs")
        if type(raw_costs) is not list:
            raise TypeError("prospective GPU-hour costs must be an array")
        manifest = cls(
            **row,
            pilot_source_manifest=pilot,
            one_shot_source_manifest=one_shot,
            costs=tuple(ProspectiveGpuHourCost(**item) for item in raw_costs),
        )
        if manifest.sha256 != declared:
            raise ValueError("prospective GPU-hour source digest differs")
        return manifest


_STAGED_PROSPECTIVE_STAGES = {"E3a", "TTS-Cal", "E1", "E2", "E4", "E1a"}


@dataclass(frozen=True)
class StagedProspectiveGpuHourCost:
    """One exact component of an honest staged estimate, in integer ns."""

    category: Literal["actual_completed", "projected_remaining", "total"]
    cell_count: int
    compute_gpu_ns: int
    provider_base_reserved_gpu_ns: int
    wall_ns: int
    retry_reserve_gpu_ns: int
    profile_reserve_gpu_ns: int
    evidence_reserve_gpu_ns: int

    def __post_init__(self) -> None:
        if self.category not in {
            "actual_completed",
            "projected_remaining",
            "total",
        }:
            raise ValueError("staged prospective GPU-hour category is unsupported")
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
                    f"staged prospective GPU-hour {label} must be non-negative"
                )

    @property
    def reserved_gpu_ns(self) -> int:
        return (
            self.provider_base_reserved_gpu_ns
            + self.retry_reserve_gpu_ns
            + self.profile_reserve_gpu_ns
            + self.evidence_reserve_gpu_ns
        )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_dict("staged GPU-hour cost", value, set(cls.__dataclass_fields__))
        )


@dataclass(frozen=True)
class StagedGpuHourStratum:
    """Deterministic projection domain and its minimum missing pilot."""

    stratum_sha256: str
    cell_ids: tuple[str, ...]
    completed_cell_ids: tuple[str, ...]
    projected_cell_ids: tuple[str, ...]
    status: Literal["MEASURED", "UNMEASURED"]
    minimum_pilot_cell_id: str | None

    def __post_init__(self) -> None:
        _sha256("staged GPU-hour stratum", self.stratum_sha256)
        for label, values in (
            ("cells", self.cell_ids),
            ("completed cells", self.completed_cell_ids),
            ("projected cells", self.projected_cell_ids),
        ):
            if (
                type(values) is not tuple
                or values != tuple(sorted(set(values)))
                or any(type(value) is not str for value in values)
            ):
                raise ValueError(f"staged GPU-hour stratum {label} are not canonical")
        if (
            not self.cell_ids
            or set(self.completed_cell_ids) | set(self.projected_cell_ids)
            != set(self.cell_ids)
            or set(self.completed_cell_ids) & set(self.projected_cell_ids)
        ):
            raise ValueError("staged GPU-hour stratum cell partition differs")
        if self.status == "MEASURED":
            if not self.completed_cell_ids or self.minimum_pilot_cell_id is not None:
                raise ValueError("measured staged GPU-hour stratum pilot state differs")
        elif self.status == "UNMEASURED":
            if (
                self.completed_cell_ids
                or self.minimum_pilot_cell_id != self.cell_ids[0]
            ):
                raise ValueError("unmeasured staged GPU-hour stratum pilot differs")
        else:
            raise ValueError("staged GPU-hour stratum status is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "stratum_sha256": self.stratum_sha256,
            "cell_ids": list(self.cell_ids),
            "completed_cell_ids": list(self.completed_cell_ids),
            "projected_cell_ids": list(self.projected_cell_ids),
            "status": self.status,
            "minimum_pilot_cell_id": self.minimum_pilot_cell_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_dict(
            "staged GPU-hour stratum", value, set(cls.__dataclass_fields__)
        )
        cell_ids = row.pop("cell_ids")
        completed_cell_ids = row.pop("completed_cell_ids")
        projected_cell_ids = row.pop("projected_cell_ids")
        return cls(
            **row,
            cell_ids=tuple(cell_ids),
            completed_cell_ids=tuple(completed_cell_ids),
            projected_cell_ids=tuple(projected_cell_ids),
        )


@dataclass(frozen=True)
class StagedProspectiveGpuHourSourceManifest:
    """Actual completed cost plus same-stratum projection, or explicit BLOCKED."""

    schema_version: Literal[1]
    kind: Literal["staged_prospective_gpu_hour_source_manifest"]
    protocol_sha256: str
    protocol_lock_sha256: str
    runtime_authority_member_sha256: str
    stage: str
    materialization_receipt_sha256: str
    completed_source_manifest: CanonicalJsonProofBinding | None
    inventory_sha256: str
    hardware_envelope_sha256: str | None
    status: Literal["READY", "BLOCKED"]
    strata: tuple[StagedGpuHourStratum, ...]
    minimum_pilot_cell_ids: tuple[str, ...]
    actual_completed: StagedProspectiveGpuHourCost
    projected_remaining: StagedProspectiveGpuHourCost | None
    total: StagedProspectiveGpuHourCost | None
    mapping_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "staged_prospective_gpu_hour_source_manifest"
            or self.protocol_sha256 != STAGED_PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256
            or self.stage not in _STAGED_PROSPECTIVE_STAGES
        ):
            raise ValueError("staged prospective GPU-hour schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_member_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("mapping", self.mapping_sha256),
        ):
            _sha256(f"staged prospective GPU-hour {label}", digest)
        if (
            self.completed_source_manifest is not None
            and type(self.completed_source_manifest) is not CanonicalJsonProofBinding
        ):
            raise TypeError("staged prospective completed source is not path-bound")
        if self.hardware_envelope_sha256 is not None:
            _sha256(
                "staged prospective GPU-hour hardware",
                self.hardware_envelope_sha256,
            )
        if (
            type(self.strata) is not tuple
            or not self.strata
            or any(type(row) is not StagedGpuHourStratum for row in self.strata)
            or tuple(row.stratum_sha256 for row in self.strata)
            != tuple(sorted(row.stratum_sha256 for row in self.strata))
            or len({row.stratum_sha256 for row in self.strata}) != len(self.strata)
        ):
            raise ValueError("staged prospective GPU-hour strata are not canonical")
        all_cell_ids = tuple(cell_id for row in self.strata for cell_id in row.cell_ids)
        if len(all_cell_ids) != len(set(all_cell_ids)):
            raise ValueError("staged prospective GPU-hour strata overlap")
        expected_missing = tuple(
            sorted(
                row.minimum_pilot_cell_id
                for row in self.strata
                if row.minimum_pilot_cell_id is not None
            )
        )
        if self.minimum_pilot_cell_ids != expected_missing:
            raise ValueError("staged prospective minimum pilot set differs")
        if type(self.actual_completed) is not StagedProspectiveGpuHourCost or (
            self.actual_completed.category != "actual_completed"
        ):
            raise TypeError("staged prospective actual component is invalid")
        completed_count = sum(len(row.completed_cell_ids) for row in self.strata)
        projected_count = sum(len(row.projected_cell_ids) for row in self.strata)
        if self.actual_completed.cell_count != completed_count:
            raise ValueError("staged prospective actual coverage differs")
        if (self.completed_source_manifest is None) != (completed_count == 0) or (
            (self.hardware_envelope_sha256 is None) != (completed_count == 0)
        ):
            raise ValueError("staged prospective completed proof authority differs")
        if self.status == "BLOCKED":
            if (
                not self.minimum_pilot_cell_ids
                or self.projected_remaining is not None
                or self.total is not None
            ):
                raise ValueError("BLOCKED staged GPU-hour source exposes a projection")
        elif self.status == "READY":
            if (
                self.minimum_pilot_cell_ids
                or type(self.projected_remaining) is not StagedProspectiveGpuHourCost
                or type(self.total) is not StagedProspectiveGpuHourCost
                or self.projected_remaining.category != "projected_remaining"
                or self.total.category != "total"
                or self.completed_source_manifest is None
                or self.hardware_envelope_sha256 is None
                or any(row.status != "MEASURED" for row in self.strata)
                or self.projected_remaining.cell_count != projected_count
            ):
                raise ValueError("READY staged GPU-hour source is incomplete")
            expected_total = _sum_staged_costs(
                "total", self.actual_completed, self.projected_remaining
            )
            if self.total != expected_total or self.total.cell_count != len(
                all_cell_ids
            ):
                raise ValueError("staged prospective GPU-hour total differs")
        else:
            raise ValueError("staged prospective GPU-hour status is unsupported")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "runtime_authority_member_sha256": self.runtime_authority_member_sha256,
            "stage": self.stage,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "completed_source_manifest": (
                None
                if self.completed_source_manifest is None
                else self.completed_source_manifest.to_dict()
            ),
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "status": self.status,
            "strata": [row.to_dict() for row in self.strata],
            "minimum_pilot_cell_ids": list(self.minimum_pilot_cell_ids),
            "actual_completed": self.actual_completed.to_dict(),
            "projected_remaining": (
                None
                if self.projected_remaining is None
                else self.projected_remaining.to_dict()
            ),
            "total": None if self.total is None else self.total.to_dict(),
            "mapping_sha256": self.mapping_sha256,
            "manifest_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_dict(
            "staged prospective GPU-hour source",
            value,
            {*cls.__dataclass_fields__, "manifest_sha256"},
        )
        declared = row.pop("manifest_sha256")
        raw_source = row.pop("completed_source_manifest")
        raw_strata = row.pop("strata")
        raw_missing = row.pop("minimum_pilot_cell_ids")
        raw_actual = row.pop("actual_completed")
        raw_projected = row.pop("projected_remaining")
        raw_total = row.pop("total")
        if type(raw_strata) is not list or type(raw_missing) is not list:
            raise TypeError("staged prospective GPU-hour arrays are invalid")
        manifest = cls(
            **row,
            completed_source_manifest=(
                None
                if raw_source is None
                else CanonicalJsonProofBinding.from_dict(raw_source)
            ),
            strata=tuple(StagedGpuHourStratum.from_dict(item) for item in raw_strata),
            minimum_pilot_cell_ids=tuple(raw_missing),
            actual_completed=StagedProspectiveGpuHourCost.from_dict(raw_actual),
            projected_remaining=(
                None
                if raw_projected is None
                else StagedProspectiveGpuHourCost.from_dict(raw_projected)
            ),
            total=(
                None
                if raw_total is None
                else StagedProspectiveGpuHourCost.from_dict(raw_total)
            ),
        )
        if manifest.sha256 != declared:
            raise ValueError("staged prospective GPU-hour source digest differs")
        return manifest


@dataclass(frozen=True)
class _ValidatedLifecycle:
    execution_binding: VerifiedFormalServingExecutionBinding
    execution_proof_binding: CanonicalJsonProofBinding
    execution_proof_payload: FormalServingExecutionProofPayload
    execution_proof_artifact: FormalServingExecutionProofArtifact
    native_run_binding: NativeTerminalRunBinding
    proof_binding: CanonicalJsonProofBinding
    lifecycle_replay_reservation: ChallengeReplayReservationBinding
    verified: _ValidatedLifecycleTiming


@dataclass(frozen=True)
class _ValidatedLifecycleTiming:
    """Closed TP1-or-distributed lifecycle projection used by cost reducers."""

    proof_kind: Literal["tp1", "distributed"]
    sha256: str
    raw_timing_sha256: str
    live_run_receipt_sha256: str
    native_result_proof_sha256: str
    run_binding_sha256: str
    run_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    method: str
    inventory_sha256: str
    registry_sha256: str
    root_manifest_sha256: str
    hardware_envelope_sha256: str
    gpu_uuids: tuple[str, ...]
    telemetry_detail: Literal["headline", "profile"]
    phase_edges_ns: tuple[tuple[str, int], ...]
    phase_durations_ns: tuple[tuple[str, int], ...]
    control_envelope_sha256: str
    replay_reservation_sha256: str


def _validate_serving_lifecycle_timing(
    *,
    proof_path: str,
    native: NativeTerminalRunBinding,
    topology_mode: str,
    gpu_uuids: tuple[str, ...],
    hardware_envelope_sha256: str,
    telemetry_detail: Literal["headline", "profile"],
    protocol_lock: ProtocolLock,
    inventory: GpuInventory,
    now_ns: int,
) -> tuple[
    CanonicalJsonProofBinding,
    ChallengeReplayReservationBinding,
    _ValidatedLifecycleTiming,
]:
    """Deep-open the topology-specific formal lifecycle proof closed union."""

    from lightcone_spec.orchestration.formal_serving_lift import (
        FormalDistributedLifecycleTimingProofArtifact,
        validate_formal_distributed_lifecycle_timing_proof_artifact,
    )
    from lightcone_spec.orchestration.formal_terminal_result import (
        FormalDistributedTerminalResultProofArtifact,
    )
    from lightcone_spec.orchestration.live_sglang import (
        UnsignedPinnedSglangLifecycleTimingReceipt,
        VerifiedPinnedSglangLifecycleTimingProof,
        validate_pinned_sglang_lifecycle_timing_proof_artifact,
    )

    proof_binding = CanonicalJsonProofBinding.bind(
        _absolute_path("formal GPU-hour lifecycle proof", proof_path)
    )
    if topology_mode == "tp1_dp1":
        verified = validate_pinned_sglang_lifecycle_timing_proof_artifact(
            proof_binding.absolute_path,
            expected_binding=native,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=protocol_lock.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            expected_gpu_uuids=gpu_uuids,
            expected_telemetry_detail=telemetry_detail,
            now_ns=now_ns,
        )
        if type(verified) is not VerifiedPinnedSglangLifecycleTimingProof:
            raise TypeError("TP1 lifecycle validator returned an invalid proof")
        artifact = _load_lifecycle_timing_proof_artifact(proof_binding.absolute_path)
        projection = _ValidatedLifecycleTiming(
            proof_kind="tp1",
            **{
                field: getattr(verified, field)
                for field in _ValidatedLifecycleTiming.__dataclass_fields__
                if field != "proof_kind"
            },
        )
        reservation = artifact.replay_reservation
    elif topology_mode in {"tp2_dp1", "tp1_dp2"}:
        artifact = validate_formal_distributed_lifecycle_timing_proof_artifact(
            proof_binding.absolute_path,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=protocol_lock.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            now_ns=now_ns,
        )
        if (
            type(artifact) is not FormalDistributedLifecycleTimingProofArtifact
            or artifact.sha256 != proof_binding.semantic_sha256
            or artifact.topology_mode != topology_mode
        ):
            raise ValueError("distributed GPU-hour lifecycle identity differs")
        timing = UnsignedPinnedSglangLifecycleTimingReceipt.from_dict(
            artifact.raw_lifecycle_timing.reopen()
        )
        terminal = FormalDistributedTerminalResultProofArtifact.from_dict(
            artifact.terminal_result_proof.reopen()
        )
        if (
            timing.sha256 != artifact.raw_lifecycle_timing.semantic_sha256
            or terminal.sha256 != artifact.terminal_result_proof.semantic_sha256
            or timing.formal_launch_admission != artifact.launch_admission
            or timing.formal_launch_consumption != artifact.launch_consumption
            or timing.budget_consumption != artifact.budget_consumption
            or timing.run_binding_sha256 != content_sha256(_run_binding_to_dict(native))
        ):
            raise ValueError("distributed GPU-hour lifecycle DAG differs")
        reservation = terminal.replay_reservation
        projection = _ValidatedLifecycleTiming(
            proof_kind="distributed",
            sha256=artifact.sha256,
            raw_timing_sha256=timing.sha256,
            live_run_receipt_sha256=timing.live_run_receipt.semantic_sha256,
            native_result_proof_sha256=(artifact.terminal_result_proof.semantic_sha256),
            run_binding_sha256=timing.run_binding_sha256,
            run_id=timing.run_id,
            run_nonce_sha256=timing.run_nonce_sha256,
            execution_plan_sha256=timing.execution_plan_sha256,
            rank_config_sha256=timing.rank_config_sha256,
            attempt_id=timing.attempt_id,
            method=timing.method,
            inventory_sha256=timing.inventory_sha256,
            registry_sha256=terminal.expected_registry_sha256,
            root_manifest_sha256=terminal.expected_root_manifest_sha256,
            hardware_envelope_sha256=(
                terminal.control_attestation.hardware_envelope_sha256
            ),
            gpu_uuids=timing.gpu_uuids,
            telemetry_detail=timing.telemetry_detail,
            phase_edges_ns=tuple(timing.phase_edges_ns.items()),
            phase_durations_ns=tuple(timing.phase_durations_ns.items()),
            control_envelope_sha256=terminal.control_attestation.sha256,
            replay_reservation_sha256=reservation.reservation_sha256,
        )
    else:  # pragma: no cover - sealed execution topology is closed upstream
        raise AssertionError("formal GPU-hour topology is unsupported")
    if (
        projection.execution_plan_sha256 != native.execution_plan_sha256
        or projection.rank_config_sha256 != native.rank_config_sha256
        or projection.run_id != native.run_id
        or projection.run_nonce_sha256 != native.run_nonce_sha256
        or projection.attempt_id != native.attempt_id
        or projection.method != native.method
        or projection.inventory_sha256 != inventory.sha256
        or projection.registry_sha256 != protocol_lock.registry_sha256
        or projection.root_manifest_sha256
        != protocol_lock.offline_release_trust_root_sha256
        or projection.gpu_uuids != gpu_uuids
        or projection.hardware_envelope_sha256 != hardware_envelope_sha256
        or projection.telemetry_detail != telemetry_detail
        or projection.replay_reservation_sha256 != reservation.reservation_sha256
    ):
        raise ValueError("lifecycle GPU-hour proof differs from sealed execution")
    return proof_binding, reservation, projection


def _schedule_sha256(
    observations: tuple[LifecycleGpuHourObservation, ...],
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "lifecycle_gpu_hour_reconstructed_schedule",
            "assignments": tuple(
                row.schedule_assignment_sha256 for row in observations
            ),
            "waves": tuple(
                (
                    wave,
                    tuple(
                        row.materialized_cell_id
                        for row in observations
                        if row.wave_index == wave
                    ),
                )
                for wave in sorted({row.wave_index for row in observations})
            ),
        }
    )


def _overlap(left: _ValidatedLifecycle, right: _ValidatedLifecycle) -> bool:
    left_edges = dict(left.verified.phase_edges_ns)  # type: ignore[attr-defined]
    right_edges = dict(right.verified.phase_edges_ns)  # type: ignore[attr-defined]
    return (
        left_edges["execution_started_ns"] < right_edges["evidence_flush_finished_ns"]
        and right_edges["execution_started_ns"]
        < left_edges["evidence_flush_finished_ns"]
    )


def _wave_indexes(rows: tuple[_ValidatedLifecycle, ...]) -> dict[str, int]:
    """Return connected overlap components and reject impossible co-tenancy."""

    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                dict(row.verified.phase_edges_ns)["execution_started_ns"],  # type: ignore[attr-defined]
                row.execution_binding.subject.materialized_cell_id,
            ),
        )
    )
    components: list[list[_ValidatedLifecycle]] = []
    remaining = list(ordered)
    while remaining:
        component = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in tuple(remaining):
                if any(_overlap(candidate, member) for member in component):
                    component.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        components.append(component)
    mapping: dict[str, int] = {}
    for wave_index, component in enumerate(components):
        events: list[tuple[int, int, tuple[str, ...]]] = []
        for row in component:
            edges = dict(row.verified.phase_edges_ns)  # type: ignore[attr-defined]
            gpus = row.execution_binding.subject.gpu_uuids
            events.append((edges["execution_started_ns"], 1, gpus))
            events.append((edges["evidence_flush_finished_ns"], -1, gpus))
        # End events precede start events at the same timestamp.
        active: set[str] = set()
        for _timestamp, direction, gpus in sorted(
            events, key=lambda item: (item[0], item[1])
        ):
            if direction < 0:
                if not set(gpus).issubset(active):
                    raise ValueError(
                        "lifecycle GPU-hour interval events are inconsistent"
                    )
                active.difference_update(gpus)
            else:
                if active & set(gpus) or len(active) + len(gpus) > 2:
                    raise ValueError("lifecycle GPU-hour proofs overlap a physical GPU")
                active.update(gpus)
        if active:
            raise ValueError("lifecycle GPU-hour interval did not terminate")
        for row in component:
            mapping[row.execution_binding.subject.materialized_cell_id] = wave_index
    return mapping


def _require_runtime_authority(
    protocol_lock: ProtocolLock,
    manifest: FormalRuntimeAuthorityManifest,
) -> str:
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal GPU-hour reducer requires exact ProtocolLock")
    if type(manifest) is not FormalRuntimeAuthorityManifest:
        raise TypeError("formal GPU-hour reducer requires exact runtime authority")
    if manifest.sha256 != protocol_lock.formal_runtime_authority_manifest_sha256:
        raise ValueError("formal GPU-hour runtime authority differs from ProtocolLock")
    member = manifest.member("gpu_hour_budget_reducer")
    if (
        member.protocol_sha256 != FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256
        or member.runner_sha256 != FORMAL_GPU_HOUR_BUDGET_RUNNER_SHA256
        or member.test_set_sha256 != FORMAL_GPU_HOUR_BUDGET_TEST_SET_SHA256
    ):
        raise FormalGpuHourLifecycleBlocked(
            "gpu_hour_budget_reducer_source_identity_mismatch"
        )
    return member.sha256


def _reject_untyped_special_gpu_hour_cells(
    materialization: StageMaterializationReceipt,
    *,
    expected_cell_ids: tuple[str, ...] | None = None,
    allow_e5_dedicated_failure_source: bool = False,
) -> None:
    """Fail closed where serving lifecycles are not the registered cost proof."""

    if (
        materialization.stage == "E4"
        and materialization.materialization_rule
        == "three_profiler_only_rows_separate_from_headline"
    ):
        raise FormalGpuHourLifecycleBlocked(
            "e4_profiler_dedicated_lifecycle_cost_proof_missing"
        )
    selected = (
        {cell.cell_id for cell in materialization.cells}
        if expected_cell_ids is None
        else set(expected_cell_ids)
    )
    if (
        not allow_e5_dedicated_failure_source
        and materialization.stage == "E5"
        and any(
            cell.cell_id in selected and cell.task == "deterministic_failure_injection"
            for cell in materialization.cells
        )
    ):
        raise FormalGpuHourLifecycleBlocked(
            "e5_dedicated_failure_lifecycle_cost_proof_required"
        )
    if materialization.stage == "E6" and any(
        cell.cell_id in selected
        and cell.task == "immutable_metadata_interface_and_fit_preflight"
        for cell in materialization.cells
    ):
        raise FormalGpuHourLifecycleBlocked(
            "e6_model_compatibility_lifecycle_cost_proof_missing"
        )


def _validate_inputs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    proof_inputs: tuple[LifecycleGpuHourProofInput, ...],
    now_ns: int,
    expected_cell_ids: tuple[str, ...] | None = None,
) -> tuple[tuple[_ValidatedLifecycle, ...], str]:
    from lightcone_spec.config import run_config_sha256

    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal GPU-hour reducer requires exact materialization")
    if materialization.protocol_lock_sha256 != protocol_lock.sha256:
        raise ValueError("formal GPU-hour materialization differs from ProtocolLock")
    _reject_untyped_special_gpu_hour_cells(
        materialization,
        expected_cell_ids=expected_cell_ids,
    )
    if type(inventory) is not GpuInventory or len(inventory.devices) != 2:
        raise ValueError("formal GPU-hour reducer requires exact two-GPU inventory")
    if type(proof_inputs) is not tuple or not proof_inputs:
        raise TypeError("formal GPU-hour reducer requires lifecycle proof inputs")
    materialized_cell_ids = tuple(row.cell_id for row in materialization.cells)
    cell_ids = materialized_cell_ids if expected_cell_ids is None else expected_cell_ids
    if cell_ids != tuple(sorted(set(cell_ids))) or set(cell_ids) - set(
        materialized_cell_ids
    ):
        raise ValueError("formal GPU-hour requested cell subset is not exact")
    input_cell_ids = tuple(
        row.execution_binding.subject.materialized_cell_id for row in proof_inputs
    )
    if tuple(sorted(input_cell_ids)) != cell_ids or len(set(input_cell_ids)) != len(
        input_cell_ids
    ):
        raise ValueError(
            "formal GPU-hour proofs must cover every materialized cell once"
        )
    validated: list[_ValidatedLifecycle] = []
    hardware: set[str] = set()
    for row in proof_inputs:
        if type(row) is not LifecycleGpuHourProofInput:
            raise TypeError("formal GPU-hour proof inputs must be exact typed joins")
        binding = require_verified_formal_serving_execution_binding(
            row.execution_binding
        )
        subject = binding.subject
        identity = subject.execution_identity
        native = row.native_run_binding
        native.validate()
        if (
            subject.protocol_lock_sha256 != protocol_lock.sha256
            or subject.formal_runtime_authority_manifest_sha256
            != formal_runtime_authority_manifest.sha256
            or subject.materialization_receipt_sha256 != materialization.sha256
            or subject.inventory_sha256 != inventory.sha256
            or any(
                gpu not in {device.uuid for device in inventory.devices}
                for gpu in subject.gpu_uuids
            )
            or native.run_id != identity.run_id
            or native.run_nonce_sha256 != identity.run_nonce_sha256
            or native.execution_plan_sha256 != subject.execution_plan_sha256
            or native.rank_config_sha256 != subject.rank_config_sha256
            or native.attempt_id != identity.attempt_id
            or native.method != subject.method
        ):
            raise ValueError("formal GPU-hour execution/run identity differs")
        execution_proof_binding = CanonicalJsonProofBinding.bind(
            row.execution_proof_artifact_path
        )
        execution_proof_artifact = _load_formal_serving_execution_proof_artifact(
            row.execution_proof_artifact_path
        )
        execution_proof_payload = validate_formal_serving_execution_proof_artifact(
            row.execution_proof_artifact_path,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=(formal_runtime_authority_manifest),
            materialization=materialization,
            inventory=inventory,
            expected_cell_id=subject.materialized_cell_id,
            now_ns=now_ns,
        )
        if (
            execution_proof_payload.execution_binding_sha256 != binding.sha256
            or execution_proof_payload.subject_sha256 != subject.sha256
            or execution_proof_payload.run_config_sha256
            != run_config_sha256(binding.run_config)
            or execution_proof_payload.execution_plan_sha256
            != subject.execution_plan_sha256
            or execution_proof_payload.rank_config_sha256 != subject.rank_config_sha256
            or execution_proof_payload.run_id != identity.run_id
            or execution_proof_payload.run_nonce_sha256 != identity.run_nonce_sha256
            or execution_proof_payload.attempt_id != identity.attempt_id
        ):
            raise ValueError("formal serving execution proof differs from binding")
        proof_binding, lifecycle_reservation, verified = (
            _validate_serving_lifecycle_timing(
                proof_path=row.lifecycle_proof_artifact_path,
                native=native,
                topology_mode=subject.topology_mode,
                gpu_uuids=subject.gpu_uuids,
                hardware_envelope_sha256=binding.hardware_envelope_sha256,
                telemetry_detail=binding.run_config.runtime.telemetry_detail,
                protocol_lock=protocol_lock,
                inventory=inventory,
                now_ns=now_ns,
            )
        )
        hardware.add(verified.hardware_envelope_sha256)
        validated.append(
            _ValidatedLifecycle(
                execution_binding=binding,
                execution_proof_binding=execution_proof_binding,
                execution_proof_payload=execution_proof_payload,
                execution_proof_artifact=execution_proof_artifact,
                native_run_binding=native,
                proof_binding=proof_binding,
                lifecycle_replay_reservation=lifecycle_reservation,
                verified=verified,
            )
        )
    if len(hardware) != 1:
        raise ValueError("formal GPU-hour proofs use different hardware envelopes")
    return tuple(validated), member_sha256


def _source_manifest(
    *,
    protocol_lock: ProtocolLock,
    runtime_authority_member_sha256: str,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    validated: tuple[_ValidatedLifecycle, ...],
) -> LifecycleGpuHourSourceManifest:
    wave_by_cell = _wave_indexes(validated)
    observations = tuple(
        sorted(
            (
                LifecycleGpuHourObservation(
                    materialized_cell_id=row.execution_binding.subject.materialized_cell_id,
                    execution_binding_sha256=row.execution_binding.sha256,
                    execution_proof=row.execution_proof_binding,
                    execution_proof_payload_sha256=(row.execution_proof_payload.sha256),
                    execution_control_envelope_sha256=(
                        row.execution_proof_artifact.control_attestation.sha256
                    ),
                    execution_replay_reservation=(
                        row.execution_proof_artifact.replay_reservation
                    ),
                    native_run_binding=row.native_run_binding,
                    lifecycle_proof=row.proof_binding,
                    verified_lifecycle_proof_sha256=row.verified.sha256,  # type: ignore[attr-defined]
                    raw_timing_sha256=row.verified.raw_timing_sha256,  # type: ignore[attr-defined]
                    live_run_receipt_sha256=row.verified.live_run_receipt_sha256,  # type: ignore[attr-defined]
                    native_result_proof_sha256=row.verified.native_result_proof_sha256,  # type: ignore[attr-defined]
                    run_binding_sha256=row.verified.run_binding_sha256,  # type: ignore[attr-defined]
                    control_envelope_sha256=row.verified.control_envelope_sha256,  # type: ignore[attr-defined]
                    lifecycle_replay_reservation=row.lifecycle_replay_reservation,
                    telemetry_detail=row.verified.telemetry_detail,  # type: ignore[attr-defined]
                    gpu_uuids=row.verified.gpu_uuids,  # type: ignore[attr-defined]
                    phase_edges_ns=row.verified.phase_edges_ns,  # type: ignore[attr-defined]
                    phase_durations_ns=row.verified.phase_durations_ns,  # type: ignore[attr-defined]
                    wave_index=wave_by_cell[
                        row.execution_binding.subject.materialized_cell_id
                    ],
                )
                for row in validated
            ),
            key=lambda item: (item.wave_index, item.materialized_cell_id),
        )
    )
    schedule_sha256 = _schedule_sha256(observations)
    hardware = {row.execution_binding.hardware_envelope_sha256 for row in validated}
    assert len(hardware) == 1
    return LifecycleGpuHourSourceManifest(
        schema_version=1,
        kind="lifecycle_gpu_hour_source_manifest",
        protocol_sha256=FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_member_sha256=runtime_authority_member_sha256,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        inventory_gpu_count=2,
        hardware_envelope_sha256=next(iter(hardware)),
        observations=observations,
        schedule_sha256=schedule_sha256,
    )


def _gpu_process_occupied_ns(row: LifecycleGpuHourObservation) -> int:
    """Charge the complete GPU-process lifetime once for one gang.

    ``execution_started_ns`` precedes server readiness, warmup, reset, scored
    arrivals, drain, and native finalization.  ``process_exited_ns`` closes
    that GPU-owning interval.  Post-exit evidence flushing is intentionally a
    separate provider reserve and must not be counted again as GPU compute.
    """

    edges = dict(row.phase_edges_ns)
    if row.start_ns != edges["execution_started_ns"]:
        raise ValueError("lifecycle GPU-hour process start identity differs")
    occupied_ns = edges["process_exited_ns"] - row.start_ns
    if occupied_ns <= 0:
        raise ValueError("lifecycle GPU-hour process occupancy is not positive")
    return occupied_ns


def _estimate(manifest: LifecycleGpuHourSourceManifest) -> GpuHourEstimate:
    compute_ns = sum(
        _gpu_process_occupied_ns(row) * row.gang_gpu_count
        for row in manifest.observations
    )
    wall_ns = 0
    evidence_ns = 0
    for wave in sorted({row.wave_index for row in manifest.observations}):
        rows = tuple(row for row in manifest.observations if row.wave_index == wave)
        wave_start_ns = min(row.start_ns for row in rows)
        process_group_empty_ns = max(
            dict(row.phase_edges_ns)["process_group_empty_checked_ns"] for row in rows
        )
        evidence_finished_ns = max(row.finish_ns for row in rows)
        wall_ns += process_group_empty_ns - wave_start_ns
        evidence_ns += (
            evidence_finished_ns - process_group_empty_ns
        ) * manifest.inventory_gpu_count
    retry_ns = (
        compute_ns * FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
    # Profile rows are already explicit materialized cells and therefore part
    # of compute/base wall.  Adding their whole wall again would double count.
    profile_ns = 0

    def hours(value: int) -> float:
        return float(value / NANOSECONDS_PER_HOUR)

    compute_hours = hours(compute_ns)
    wall_hours = hours(wall_ns)
    retry_hours = hours(retry_ns)
    profile_hours = hours(profile_ns)
    evidence_hours = hours(evidence_ns)
    # Build the public float total from the exact same float components used by
    # GpuHourEstimate's minimum-reservation check.  The integer-nanosecond
    # ``reserved_ns`` identity remains covered below, while this avoids a one-
    # ulp underflow caused by dividing the pre-summed integer independently.
    reserved_hours = (
        wall_hours * manifest.inventory_gpu_count
        + retry_hours
        + profile_hours
        + evidence_hours
    )
    values = {
        "source_pilot_receipt_sha256": manifest.sha256,
        "source_schedule_sha256": manifest.schedule_sha256,
        "source_materialization_receipt_sha256": (
            manifest.materialization_receipt_sha256
        ),
        "source_inventory_gpu_count": manifest.inventory_gpu_count,
        "compute_gpu_hours": compute_hours,
        "reserved_gpu_hours": reserved_hours,
        "estimated_wall_hours": wall_hours,
        "retry_reserve_gpu_hours": retry_hours,
        "profile_reserve_gpu_hours": profile_hours,
        "evidence_reserve_gpu_hours": evidence_hours,
    }
    return GpuHourEstimate(
        status="AVAILABLE",
        derivation_sha256=content_sha256(values),
        **values,
    )


def _cost_from_actual_observations(
    *,
    category: Literal["actual_tuning", "actual_one_shot"],
    observations: tuple[LifecycleGpuHourObservation, ...],
    inventory_gpu_count: int,
) -> ProspectiveGpuHourCost:
    if not observations:
        raise ValueError("prospective actual cost requires lifecycle observations")
    compute_ns = sum(
        _gpu_process_occupied_ns(row) * row.gang_gpu_count for row in observations
    )
    wall_ns = 0
    evidence_ns = 0
    for wave in sorted({row.wave_index for row in observations}):
        rows = tuple(row for row in observations if row.wave_index == wave)
        start_ns = min(row.start_ns for row in rows)
        empty_ns = max(
            dict(row.phase_edges_ns)["process_group_empty_checked_ns"] for row in rows
        )
        finished_ns = max(row.finish_ns for row in rows)
        wall_ns += empty_ns - start_ns
        evidence_ns += (finished_ns - empty_ns) * inventory_gpu_count
    retry_ns = (
        compute_ns * FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
    return ProspectiveGpuHourCost(
        category=category,
        cell_count=len(observations),
        compute_gpu_ns=compute_ns,
        provider_base_reserved_gpu_ns=wall_ns * inventory_gpu_count,
        wall_ns=wall_ns,
        retry_reserve_gpu_ns=retry_ns,
        profile_reserve_gpu_ns=0,
        evidence_reserve_gpu_ns=evidence_ns,
    )


def _sum_staged_costs(
    category: Literal["total"],
    *costs: StagedProspectiveGpuHourCost,
) -> StagedProspectiveGpuHourCost:
    return StagedProspectiveGpuHourCost(
        category=category,
        cell_count=sum(row.cell_count for row in costs),
        compute_gpu_ns=sum(row.compute_gpu_ns for row in costs),
        provider_base_reserved_gpu_ns=sum(
            row.provider_base_reserved_gpu_ns for row in costs
        ),
        wall_ns=sum(row.wall_ns for row in costs),
        retry_reserve_gpu_ns=sum(row.retry_reserve_gpu_ns for row in costs),
        profile_reserve_gpu_ns=sum(row.profile_reserve_gpu_ns for row in costs),
        evidence_reserve_gpu_ns=sum(row.evidence_reserve_gpu_ns for row in costs),
    )


def _staged_actual_cost(
    observations: tuple[LifecycleGpuHourObservation, ...],
    *,
    inventory_gpu_count: int,
) -> StagedProspectiveGpuHourCost:
    if not observations:
        return StagedProspectiveGpuHourCost(
            category="actual_completed",
            cell_count=0,
            compute_gpu_ns=0,
            provider_base_reserved_gpu_ns=0,
            wall_ns=0,
            retry_reserve_gpu_ns=0,
            profile_reserve_gpu_ns=0,
            evidence_reserve_gpu_ns=0,
        )
    existing = _cost_from_actual_observations(
        category="actual_tuning",
        observations=observations,
        inventory_gpu_count=inventory_gpu_count,
    )
    return StagedProspectiveGpuHourCost(
        category="actual_completed",
        cell_count=existing.cell_count,
        compute_gpu_ns=existing.compute_gpu_ns,
        provider_base_reserved_gpu_ns=existing.provider_base_reserved_gpu_ns,
        wall_ns=existing.wall_ns,
        retry_reserve_gpu_ns=existing.retry_reserve_gpu_ns,
        profile_reserve_gpu_ns=existing.profile_reserve_gpu_ns,
        evidence_reserve_gpu_ns=existing.evidence_reserve_gpu_ns,
    )


_STAGED_REPEAT_DIMENSIONS_BY_STAGE = {
    "E3a": frozenset({"registry_cell_id"}),
    "TTS-Cal": frozenset({"block", "pilot_phase", "registry_cell_id"}),
    "E1": frozenset(),
    "E2": frozenset(),
    "E4": frozenset(),
    "E1a": frozenset(),
}


def _staged_projection_stratum(cell: MaterializedCell) -> str:
    """Keep every scientific factor and remove only registered repeats."""

    ignored = _STAGED_REPEAT_DIMENSIONS_BY_STAGE.get(cell.stage)
    if ignored is None:
        raise ValueError("staged projection cell names an unsupported stage")

    return content_sha256(
        {
            "stage": cell.stage,
            "method_role": cell.method_role,
            "model": cell.model,
            "backend": cell.backend,
            "task": cell.task,
            "publication_policy": cell.publication_policy,
            "recipe_sha256": cell.recipe_sha256,
            "dimensions": tuple(
                (name, value) for name, value in cell.dimensions if name not in ignored
            ),
        }
    )


def _staged_strata(
    materialization: StageMaterializationReceipt,
    *,
    completed_cell_ids: tuple[str, ...],
) -> tuple[StagedGpuHourStratum, ...]:
    completed = set(completed_cell_ids)
    groups: dict[str, list[str]] = {}
    for cell in materialization.cells:
        groups.setdefault(_staged_projection_stratum(cell), []).append(cell.cell_id)
    rows = []
    for stratum_sha256, raw_cell_ids in sorted(groups.items()):
        cell_ids = tuple(sorted(raw_cell_ids))
        completed_ids = tuple(value for value in cell_ids if value in completed)
        projected_ids = tuple(value for value in cell_ids if value not in completed)
        rows.append(
            StagedGpuHourStratum(
                stratum_sha256=stratum_sha256,
                cell_ids=cell_ids,
                completed_cell_ids=completed_ids,
                projected_cell_ids=projected_ids,
                status="MEASURED" if completed_ids else "UNMEASURED",
                minimum_pilot_cell_id=None if completed_ids else cell_ids[0],
            )
        )
    return tuple(rows)


def _staged_projected_cost(
    *,
    materialization: StageMaterializationReceipt,
    observations: tuple[LifecycleGpuHourObservation, ...],
    strata: tuple[StagedGpuHourStratum, ...],
    inventory_gpu_count: int,
) -> tuple[StagedProspectiveGpuHourCost, tuple[tuple[object, ...], ...]]:
    by_id = {row.materialized_cell_id: row for row in observations}
    cells = {cell.cell_id: cell for cell in materialization.cells}
    compute_ns = 0
    wall_ns = 0
    evidence_ns = 0
    mapping_rows: list[tuple[object, ...]] = []
    projected_count = 0
    for stratum in strata:
        if stratum.status != "MEASURED":
            raise ValueError("cannot project an unmeasured scientific stratum")
        pilots = tuple(by_id[cell_id] for cell_id in stratum.completed_cell_ids)
        gang_counts = {row.gang_gpu_count for row in pilots}
        if len(gang_counts) != 1:
            raise ValueError("staged scientific stratum mixes GPU gang sizes")
        gang_count = next(iter(gang_counts))
        execution_ns = _ceiling_mean(
            tuple(_gpu_process_occupied_ns(row) for row in pilots)
        )
        core_wall_ns = _ceiling_mean(
            tuple(
                dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                - row.start_ns
                for row in pilots
            )
        )
        evidence_tail_ns = _ceiling_mean(
            tuple(
                row.finish_ns
                - dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                for row in pilots
            )
        )
        for cell_id in stratum.projected_cell_ids:
            # Reopen the target cell from the materialization so projection
            # cannot be applied to a caller-invented count or identity.
            if cells[cell_id].stage != materialization.stage:
                raise ValueError("staged projection target changed stage")
            projected_count += 1
            compute_ns += execution_ns * gang_count
            wall_ns += core_wall_ns
            evidence_ns += evidence_tail_ns * inventory_gpu_count
            mapping_rows.append(
                (
                    cell_id,
                    stratum.stratum_sha256,
                    stratum.completed_cell_ids,
                    execution_ns,
                    core_wall_ns,
                    evidence_tail_ns,
                    gang_count,
                )
            )
    retry_ns = (
        compute_ns * FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
    return (
        StagedProspectiveGpuHourCost(
            category="projected_remaining",
            cell_count=projected_count,
            compute_gpu_ns=compute_ns,
            provider_base_reserved_gpu_ns=wall_ns * inventory_gpu_count,
            wall_ns=wall_ns,
            retry_reserve_gpu_ns=retry_ns,
            profile_reserve_gpu_ns=0,
            evidence_reserve_gpu_ns=evidence_ns,
        ),
        tuple(mapping_rows),
    )


_PROJECTION_IGNORED_DIMENSIONS = {
    "block",
    "block_phase",
    "pilot_materialization_receipt_sha256",
    "pilot_coverage_receipt_sha256",
    "signed_power_prefix_sha256",
    "signed_power_and_anchor_prefix_sha256",
    "p99_anchor_id",
    "p99_minimum_completions",
    "p99_selection_receipt_sha256",
    "p99_extension_anchor_id",
    "p99_extension_minimum_completions",
    "p99_extension_offered_requests",
    "p99_extension_selection_receipt_sha256",
}


def _projection_stratum(cell: MaterializedCell) -> str:
    dimensions = tuple(
        (name, value)
        for name, value in cell.dimensions
        if name not in _PROJECTION_IGNORED_DIMENSIONS
    )
    return content_sha256(
        {
            "stage": cell.stage,
            "method_role": cell.method_role,
            "model": cell.model,
            "backend": cell.backend,
            "task": cell.task,
            "publication_policy": cell.publication_policy,
            "recipe_sha256": cell.recipe_sha256,
            "dimensions": dimensions,
        }
    )


def _ceiling_mean(values: tuple[int, ...]) -> int:
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("prospective GPU-hour mean inputs are invalid")
    return (sum(values) + len(values) - 1) // len(values)


def _scaled_ceiling(value: int, *, numerator: int, denominator: int) -> int:
    if (
        type(value) is not int
        or value < 0
        or type(numerator) is not int
        or numerator < 1
        or type(denominator) is not int
        or denominator < 1
    ):
        raise ValueError("prospective GPU-hour scaling values are invalid")
    if value == 0:
        return 0
    return (value * numerator + denominator - 1) // denominator


def _project_final_cost(
    *,
    pilot_materialization: StageMaterializationReceipt,
    pilot_source: LifecycleGpuHourSourceManifest,
    final_materialization: StageMaterializationReceipt,
) -> tuple[ProspectiveGpuHourCost, str]:
    pilot_cells = {cell.cell_id: cell for cell in pilot_materialization.cells}
    pilot_observations = {
        row.materialized_cell_id: row for row in pilot_source.observations
    }
    if set(pilot_cells) != set(pilot_observations):
        raise ValueError("prospective pilot lifecycle coverage is not exact")
    pilot_by_stratum: dict[str, list[tuple[int, LifecycleGpuHourObservation]]] = {}
    for cell in pilot_materialization.cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if type(block) is not int:
            continue
        if block not in range(4) or dimensions.get("block_phase") != "excluded_pilot":
            raise ValueError("prospective tuning source is not four excluded pilots")
        pilot_by_stratum.setdefault(_projection_stratum(cell), []).append(
            (block, pilot_observations[cell.cell_id])
        )
    for values in pilot_by_stratum.values():
        if tuple(sorted(block for block, _row in values)) != (0, 1, 2, 3):
            raise ValueError("prospective stratum lacks one row in every pilot block")

    projected_cells = tuple(
        cell
        for cell in final_materialization.cells
        if type(dict(cell.dimensions).get("block")) is int
    )
    if not projected_cells:
        raise ValueError("prospective final materialization has no projected rows")
    compute_ns = 0
    wall_ns = 0
    evidence_ns = 0
    mapping_rows: list[tuple[str, str, tuple[str, ...], int, int]] = []
    for cell in projected_cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if (
            type(block) is not int
            or block not in range(4, 24)
            or dimensions.get("block_phase") != "final"
        ):
            raise ValueError("prospective projected row is outside final prefix")
        stratum = _projection_stratum(cell)
        pilots = tuple(
            row for _pilot_block, row in sorted(pilot_by_stratum.get(stratum, ()))
        )
        if len(pilots) != 4:
            raise ValueError("prospective final row lacks exact four-pilot mapping")
        gang_counts = {row.gang_gpu_count for row in pilots}
        expected_gang = 2 if dimensions.get("topology") in {"tp2_dp1", "tp1_dp2"} else 1
        if gang_counts != {expected_gang}:
            raise ValueError("prospective final topology differs from pilot gang")
        execution_ns = _ceiling_mean(
            tuple(_gpu_process_occupied_ns(row) for row in pilots)
        )
        core_wall_ns = _ceiling_mean(
            tuple(
                dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                - dict(row.phase_edges_ns)["execution_started_ns"]
                for row in pilots
            )
        )
        evidence_tail_ns = _ceiling_mean(
            tuple(
                row.finish_ns
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
                raise ValueError("prospective p99 anchor completion target differs")
            offered_requests = dimensions.get(
                "p99_extension_offered_requests",
                required_requests,
            )
            if type(offered_requests) is not int or offered_requests not in {
                required_requests,
                11_000,
            }:
                raise ValueError("prospective p99 extension offer target differs")
            pilot_requests = tuple(
                len(row.native_run_binding.scored_request_ids) for row in pilots
            )
            denominator = min(pilot_requests)
            execution_ns = _scaled_ceiling(
                execution_ns,
                numerator=offered_requests,
                denominator=denominator,
            )
            core_wall_ns = _scaled_ceiling(
                core_wall_ns,
                numerator=offered_requests,
                denominator=denominator,
            )
            evidence_tail_ns = _scaled_ceiling(
                evidence_tail_ns,
                numerator=offered_requests,
                denominator=denominator,
            )
        compute_ns += execution_ns * expected_gang
        wall_ns += core_wall_ns
        evidence_ns += evidence_tail_ns * 2
        mapping_rows.append(
            (
                cell.cell_id,
                stratum,
                tuple(row.materialized_cell_id for row in pilots),
                execution_ns,
                core_wall_ns,
            )
        )
    retry_ns = (
        compute_ns * FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
    mapping_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "exact_excluded_pilot_to_final_gpu_hour_mapping",
            "pilot_materialization_receipt_sha256": pilot_materialization.sha256,
            "pilot_source_manifest_sha256": pilot_source.sha256,
            "final_materialization_receipt_sha256": final_materialization.sha256,
            "rows": tuple(mapping_rows),
            "schedule": "isolated_fail_closed_two_gpu_provider_reservation",
        }
    )
    return (
        ProspectiveGpuHourCost(
            category="projected_final",
            cell_count=len(projected_cells),
            compute_gpu_ns=compute_ns,
            provider_base_reserved_gpu_ns=wall_ns * 2,
            wall_ns=wall_ns,
            retry_reserve_gpu_ns=retry_ns,
            profile_reserve_gpu_ns=0,
            evidence_reserve_gpu_ns=evidence_ns,
        ),
        mapping_sha256,
    )


def _estimate_prospective_manifest(
    manifest: ProspectiveGpuHourSourceManifest,
) -> GpuHourEstimate:
    def hours(value: int) -> float:
        return float(value / NANOSECONDS_PER_HOUR)

    compute = sum(row.compute_gpu_ns for row in manifest.costs)
    wall = sum(row.wall_ns for row in manifest.costs)
    retry = sum(row.retry_reserve_gpu_ns for row in manifest.costs)
    profile = sum(row.profile_reserve_gpu_ns for row in manifest.costs)
    evidence = sum(row.evidence_reserve_gpu_ns for row in manifest.costs)
    base = sum(row.provider_base_reserved_gpu_ns for row in manifest.costs)
    values = {
        "source_pilot_receipt_sha256": manifest.sha256,
        "source_schedule_sha256": manifest.mapping_sha256,
        "source_materialization_receipt_sha256": (
            manifest.final_materialization_receipt_sha256
        ),
        "source_inventory_gpu_count": 2,
        "compute_gpu_hours": hours(compute),
        "reserved_gpu_hours": (
            hours(base) + hours(retry) + hours(profile) + hours(evidence)
        ),
        "estimated_wall_hours": hours(wall),
        "retry_reserve_gpu_hours": hours(retry),
        "profile_reserve_gpu_hours": hours(profile),
        "evidence_reserve_gpu_hours": hours(evidence),
    }
    return GpuHourEstimate(
        status="AVAILABLE",
        derivation_sha256=content_sha256(values),
        **values,
    )


def _reservation_binding_from_record(
    *,
    path: str,
    reservation_sha256: str,
    expected_raw_sha256: str | None = None,
    expected_size: int | None = None,
) -> ChallengeReplayReservationBinding:
    reservation_path, body, reserved_ns, challenges = (
        ChallengeReplayReservationBinding._observe(
            Path(path).parent,
            reservation_sha256,
        )
    )
    binding = ChallengeReplayReservationBinding(
        schema_version=1,
        kind="lightcone_challenge_replay_reservation_binding",
        path=str(reservation_path),
        reservation_sha256=reservation_sha256,
        raw_sha256=__import__("hashlib").sha256(body).hexdigest(),
        size=len(body),
        reserved_ns=reserved_ns,
        challenge_sha256s=challenges,
    )
    if expected_raw_sha256 is not None and binding.raw_sha256 != expected_raw_sha256:
        raise ValueError("preflight control reservation raw identity differs")
    if expected_size is not None and binding.size != expected_size:
        raise ValueError("preflight control reservation size differs")
    return binding


def _preflight_wave_indexes(
    observations: tuple[PreflightGpuHourObservation, ...],
) -> dict[str, int]:
    def overlaps(
        left: PreflightGpuHourObservation,
        right: PreflightGpuHourObservation,
    ) -> bool:
        return (
            left.process_started_ns < right.evidence_finished_ns
            and right.process_started_ns < left.evidence_finished_ns
        )

    remaining = sorted(
        observations,
        key=lambda row: (row.process_started_ns, row.materialized_cell_id),
    )
    components: list[list[PreflightGpuHourObservation]] = []
    while remaining:
        component = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in tuple(remaining):
                if any(overlaps(candidate, member) for member in component):
                    component.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        components.append(component)
    result: dict[str, int] = {}
    for wave_index, component in enumerate(components):
        events: list[tuple[int, int, tuple[str, ...]]] = []
        for row in component:
            events.append((row.process_started_ns, 1, row.gpu_uuids))
            events.append((row.gpu_released_ns, -1, row.gpu_uuids))
        active: set[str] = set()
        for _timestamp, direction, gpu_uuids in sorted(
            events, key=lambda value: (value[0], value[1])
        ):
            gpus = set(gpu_uuids)
            if direction < 0:
                if not gpus <= active:
                    raise ValueError("preflight GPU-hour interval events differ")
                active.difference_update(gpus)
            else:
                if active & gpus or len(active) + len(gpus) > 2:
                    raise ValueError("preflight GPU-hour proofs overlap a physical GPU")
                active.update(gpus)
        if active:
            raise ValueError("preflight GPU-hour GPU interval did not terminate")
        result.update((row.materialized_cell_id, wave_index) for row in component)
    return result


def _preflight_observations_from_sources(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    stage_coverage: object,
    source_authority: object,
    inventory: GpuInventory,
    lifecycle_proof_inputs: tuple[PreflightGpuHourLifecycleProofInput, ...],
    now_ns: int,
) -> tuple[tuple[PreflightGpuHourObservation, ...], str]:
    from dataclasses import replace

    from lightcone_spec.experiments.preflight_authority import (
        PreflightExecutionSourceAuthority,
    )
    from lightcone_spec.experiments.preflight_interference import (
        FormalPreflightInterferenceProofArtifact,
    )
    from lightcone_spec.experiments.registry import build_industrial_registry
    from lightcone_spec.experiments.stage_materialization import StageCoverageReceipt
    from lightcone_spec.orchestration.live_sglang import (
        VerifiedPinnedSglangLifecycleTimingProof,
        validate_pinned_sglang_lifecycle_timing_proof_artifact,
    )
    from lightcone_spec.runtime.compile_runner import (
        CompileAssignmentPlan,
        CompileControlVerificationReceipt,
        CompileResultPointer,
        CompileSubprocessLifecycleReceipt,
    )
    from lightcone_spec.runtime.preflight_runner import (
        ExactnessPreflightAssignment,
        ExactnessPreflightResultPointer,
        ExactnessPreflightTerminal,
        ExactnessQualificationProofArtifact,
        ExactnessRankTerminal,
    )

    if type(materialization) is not StageMaterializationReceipt or (
        materialization.stage != "preflight"
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("preflight GPU hours require exact preflight materialization")
    if type(stage_coverage) is not StageCoverageReceipt:
        raise TypeError("preflight GPU hours require exact stage coverage")
    stage_coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in stage_coverage.dispositions):
        raise FormalGpuHourLifecycleBlocked("preflight_coverage_not_complete")
    if type(source_authority) is not PreflightExecutionSourceAuthority:
        raise TypeError("preflight GPU hours require exact source authority")
    if (
        type(inventory) is not GpuInventory
        or len(inventory.devices) != 2
        or source_authority.inventory_sha256 != inventory.sha256
        or source_authority.registry_sha256 != protocol_lock.registry_sha256
        or source_authority.release_root_manifest_sha256
        != protocol_lock.offline_release_trust_root_sha256
    ):
        raise ValueError("preflight GPU-hour immutable source identity differs")
    terminals = source_authority.revalidate(build_industrial_registry(), now_ns=now_ns)
    if (
        len(terminals) != 10
        or any(not row.passed for row in terminals)
        or tuple(sorted(row.terminal_kind for row in terminals)).count("compile") != 1
        or tuple(sorted(row.terminal_kind for row in terminals)).count("exactness") != 1
        or tuple(sorted(row.terminal_kind for row in terminals)).count("interference")
        != 8
    ):
        raise FormalGpuHourLifecycleBlocked("preflight_terminal_coverage_not_1_1_8")
    cells_by_registry: dict[str, MaterializedCell] = {}
    for cell in materialization.cells:
        registry_cell_id = dict(cell.dimensions).get("registry_cell_id")
        if type(registry_cell_id) is not str:
            raise ValueError("preflight cell lacks registry source identity")
        cells_by_registry[registry_cell_id] = cell
    if len(cells_by_registry) != 10 or set(cells_by_registry) != {
        row.cell_id for row in terminals
    }:
        raise ValueError("preflight materialization/source cell coverage differs")
    inventory_gpus = tuple(row.uuid for row in inventory.devices)
    hardware = {row.hardware_envelope_sha256 for row in inventory.devices}
    if len(hardware) != 1:
        raise ValueError("preflight GPU-hour inventory is not homogeneous")
    hardware_sha256 = next(iter(hardware))
    observations: list[PreflightGpuHourObservation] = []

    compile_pointer = CompileResultPointer.load(source_authority.compile_result.path)
    if (
        compile_pointer.assignment_plan_source is None
        or compile_pointer.subprocess_lifecycle_receipt is None
        or compile_pointer.control_verification_receipt is None
    ):
        raise FormalGpuHourLifecycleBlocked("preflight_compile_phase_timing_missing")
    compile_plan = CompileAssignmentPlan.load(
        compile_pointer.assignment_plan_source.absolute_path
    )
    compile_receipt = CompileSubprocessLifecycleReceipt.load(
        compile_pointer.subprocess_lifecycle_receipt.absolute_path
    )
    compile_control = CompileControlVerificationReceipt.load(
        compile_pointer.control_verification_receipt.absolute_path
    )
    compile_control.validate(reopen_sources=True)
    compile_cell = cells_by_registry.get(compile_plan.revalidate()[0].cell_id)
    if (
        compile_cell is None
        or set(compile_plan.gpu_uuids) != set(inventory_gpus)
        or len(compile_plan.gpu_uuids) != 2
        or compile_receipt.process_exited_ns <= compile_receipt.process_started_ns
    ):
        raise ValueError("preflight compile timing/assignment differs")
    compile_reservation = _reservation_binding_from_record(
        path=compile_control.reservation_record_path,
        reservation_sha256=compile_control.reservation_sha256,
        expected_raw_sha256=compile_control.reservation_record_raw_sha256,
        expected_size=compile_control.reservation_record_size,
    )
    compile_binding = CanonicalJsonProofBinding.bind(
        compile_pointer.subprocess_lifecycle_receipt.absolute_path,
        semantic_sha256=compile_receipt.sha256,
    )
    observations.append(
        PreflightGpuHourObservation(
            materialized_cell_id=compile_cell.cell_id,
            registry_cell_id=compile_plan.revalidate()[0].cell_id,
            phase_kind="compile",
            timing_proof=compile_binding,
            timing_authority_sha256=compile_receipt.sha256,
            execution_identity_sha256=compile_plan.sha256,
            control_envelope_sha256=compile_control.control_envelope.sha256,
            replay_reservation=compile_reservation,
            gpu_uuids=compile_plan.gpu_uuids,
            process_started_ns=compile_receipt.process_started_ns,
            process_finished_ns=compile_receipt.process_exited_ns,
            gpu_released_ns=compile_receipt.process_exited_ns,
            evidence_finished_ns=compile_receipt.process_exited_ns,
            wave_index=0,
        )
    )

    exact_pointer = ExactnessPreflightResultPointer.load(
        source_authority.exactness_result.path
    )
    if exact_pointer.qualification_proof_artifact is None:
        raise FormalGpuHourLifecycleBlocked("preflight_exactness_phase_timing_missing")
    exact_assignment = ExactnessPreflightAssignment.load(
        exact_pointer.assignment.absolute_path
    )
    exact_terminal = ExactnessPreflightTerminal.load(
        exact_pointer.terminal.absolute_path
    )
    rank_terminals = tuple(
        ExactnessRankTerminal.load(binding.absolute_path)
        for binding in exact_pointer.rank_terminals
    )
    exact_qualification = ExactnessQualificationProofArtifact.load(
        exact_pointer.qualification_proof_artifact.absolute_path
    )
    exact_qualification.revalidate(now_ns=now_ns)
    exact_cell = cells_by_registry.get(exact_assignment.cell_id)
    exact_start_ns = min(
        (exact_terminal.started_ns, *(row.process_started_ns for row in rank_terminals))
    )
    exact_finish_ns = max(
        (exact_terminal.finished_ns, *(row.finished_ns for row in rank_terminals))
    )
    if (
        exact_cell is None
        or exact_terminal.status != "PASSED"
        or len(rank_terminals) != 2
        or set(exact_assignment.gpu_uuids) != set(inventory_gpus)
        or exact_finish_ns <= exact_start_ns
        or exact_qualification.payload.hardware_envelope_sha256 != hardware_sha256
    ):
        raise ValueError("preflight exactness timing/assignment differs")
    exact_binding = CanonicalJsonProofBinding.bind(
        exact_pointer.terminal.absolute_path,
        semantic_sha256=exact_terminal.sha256,
    )
    observations.append(
        PreflightGpuHourObservation(
            materialized_cell_id=exact_cell.cell_id,
            registry_cell_id=exact_assignment.cell_id,
            phase_kind="exactness",
            timing_proof=exact_binding,
            timing_authority_sha256=exact_terminal.sha256,
            execution_identity_sha256=exact_assignment.sha256,
            control_envelope_sha256=exact_qualification.control_attestation.sha256,
            replay_reservation=exact_qualification.replay_reservation,
            gpu_uuids=exact_assignment.gpu_uuids,
            process_started_ns=exact_start_ns,
            process_finished_ns=exact_finish_ns,
            gpu_released_ns=exact_finish_ns,
            evidence_finished_ns=exact_finish_ns,
            wave_index=0,
        )
    )

    interference_artifact = FormalPreflightInterferenceProofArtifact.from_dict(
        source_authority.interference_proof_artifact.reopen()
    )
    input_by_cell = {row.materialized_cell_id: row for row in lifecycle_proof_inputs}
    interference_ids = {row.materialized_cell_id for row in interference_artifact.rows}
    if (
        type(lifecycle_proof_inputs) is not tuple
        or len(lifecycle_proof_inputs) != 8
        or len(input_by_cell) != 8
        or set(input_by_cell) != interference_ids
    ):
        raise FormalGpuHourLifecycleBlocked(
            "preflight_eight_interference_lifecycle_proofs_missing"
        )
    for proof_row in interference_artifact.rows:
        proof_input = input_by_cell[proof_row.materialized_cell_id]
        cell = cells_by_registry.get(proof_row.registry_cell_id)
        if cell is None or cell.cell_id != proof_row.materialized_cell_id:
            raise ValueError("preflight interference cell mapping differs")
        lifecycle_artifact = _load_lifecycle_timing_proof_artifact(
            proof_input.lifecycle_proof_artifact_path
        )
        verified = validate_pinned_sglang_lifecycle_timing_proof_artifact(
            proof_input.lifecycle_proof_artifact_path,
            expected_binding=proof_row.run_binding,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=protocol_lock.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            expected_gpu_uuids=(proof_row.gpu_uuid,),
            expected_telemetry_detail="headline",
            now_ns=now_ns,
        )
        if type(verified) is not VerifiedPinnedSglangLifecycleTimingProof:
            raise TypeError("preflight lifecycle validator returned invalid proof")
        edges = dict(verified.phase_edges_ns)
        timing_binding = CanonicalJsonProofBinding.bind(
            proof_input.lifecycle_proof_artifact_path
        )
        if (
            timing_binding.semantic_sha256 != lifecycle_artifact.sha256
            or verified.hardware_envelope_sha256 != hardware_sha256
            or verified.gpu_uuids != (proof_row.gpu_uuid,)
            or edges["process_exited_ns"] <= edges["execution_started_ns"]
        ):
            raise ValueError("preflight interference lifecycle timing differs")
        observations.append(
            PreflightGpuHourObservation(
                materialized_cell_id=cell.cell_id,
                registry_cell_id=proof_row.registry_cell_id,
                phase_kind="interference",
                timing_proof=timing_binding,
                timing_authority_sha256=verified.sha256,
                execution_identity_sha256=content_sha256(
                    _run_binding_to_dict(proof_row.run_binding)
                ),
                control_envelope_sha256=(lifecycle_artifact.control_attestation.sha256),
                replay_reservation=lifecycle_artifact.replay_reservation,
                gpu_uuids=(proof_row.gpu_uuid,),
                process_started_ns=edges["execution_started_ns"],
                process_finished_ns=edges["process_exited_ns"],
                gpu_released_ns=edges["process_group_empty_checked_ns"],
                evidence_finished_ns=edges["evidence_flush_finished_ns"],
                wave_index=0,
            )
        )
    provisional = tuple(observations)
    waves = _preflight_wave_indexes(provisional)
    finalized = tuple(
        sorted(
            (
                replace(row, wave_index=waves[row.materialized_cell_id])
                for row in provisional
            ),
            key=lambda row: (row.wave_index, row.materialized_cell_id),
        )
    )
    return finalized, hardware_sha256


def _estimate_preflight_manifest(
    manifest: PreflightGpuHourSourceManifest,
) -> GpuHourEstimate:
    compute_ns = sum(
        (row.process_finished_ns - row.process_started_ns) * len(row.gpu_uuids)
        for row in manifest.observations
    )
    wall_ns = 0
    evidence_ns = 0
    for wave in sorted({row.wave_index for row in manifest.observations}):
        rows = tuple(row for row in manifest.observations if row.wave_index == wave)
        start_ns = min(row.process_started_ns for row in rows)
        released_ns = max(row.gpu_released_ns for row in rows)
        finished_ns = max(row.evidence_finished_ns for row in rows)
        wall_ns += released_ns - start_ns
        evidence_ns += (finished_ns - released_ns) * 2
    retry_ns = (
        compute_ns * FORMAL_GPU_HOUR_RETRY_RESERVE_BPS
        + FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR
        - 1
    ) // FORMAL_GPU_HOUR_RESERVE_BPS_DENOMINATOR

    def hours(value: int) -> float:
        return float(value / NANOSECONDS_PER_HOUR)

    values = {
        "source_pilot_receipt_sha256": manifest.sha256,
        "source_schedule_sha256": manifest.schedule_sha256,
        "source_materialization_receipt_sha256": (
            manifest.materialization_receipt_sha256
        ),
        "source_inventory_gpu_count": 2,
        "compute_gpu_hours": hours(compute_ns),
        "reserved_gpu_hours": hours(wall_ns * 2 + retry_ns + evidence_ns),
        "estimated_wall_hours": hours(wall_ns),
        "retry_reserve_gpu_hours": hours(retry_ns),
        "profile_reserve_gpu_hours": 0.0,
        "evidence_reserve_gpu_hours": hours(evidence_ns),
    }
    return GpuHourEstimate(
        status="AVAILABLE",
        derivation_sha256=content_sha256(values),
        **values,
    )


def materialize_preflight_gpu_hour_envelope(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    final_evidence: object,
    inventory: GpuInventory,
    interference_lifecycle_proof_inputs: tuple[
        PreflightGpuHourLifecycleProofInput, ...
    ],
    source_manifest_output_path: str,
    now_ns: int,
) -> StageGpuHourEnvelope:
    """Fail closed until qualification lifecycle costs join the 1+1+8 rows."""

    # The schema-1 source below remains readable only so historical local
    # diagnostics cannot be silently reinterpreted.  It omits the resolved
    # core-10/optional-Eagle qualification process set and its launch
    # admission, so it must never publish an AVAILABLE formal envelope.
    raise FormalGpuHourLifecycleBlocked(
        "preflight_qualification_lifecycle_cost_schedule_missing"
    )

    from lightcone_spec.experiments.formal_preflight_execution import (
        FormalPreflightFinalEvidence,
    )

    if type(final_evidence) is not FormalPreflightFinalEvidence:
        raise TypeError("preflight GPU hours require exact final evidence")
    final_evidence.__post_init__()
    if (
        final_evidence.materialization.protocol_lock_sha256 != protocol_lock.sha256
        or final_evidence.source_authority.inventory_sha256 != inventory.sha256
        or final_evidence.source_authority.registry_sha256
        != protocol_lock.registry_sha256
    ):
        raise ValueError("preflight GPU-hour final evidence identity differs")
    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    observations, hardware_sha256 = _preflight_observations_from_sources(
        protocol_lock=protocol_lock,
        materialization=final_evidence.materialization,
        stage_coverage=final_evidence.stage_coverage,
        source_authority=final_evidence.source_authority,
        inventory=inventory,
        lifecycle_proof_inputs=interference_lifecycle_proof_inputs,
        now_ns=now_ns,
    )
    remote_binding = CanonicalJsonProofBinding.bind(
        final_evidence.remote_raw_receipt.absolute_path
    )
    if remote_binding != final_evidence.remote_raw_receipt:
        raise ValueError("preflight GPU-hour raw receipt binding changed")
    manifest = PreflightGpuHourSourceManifest(
        schema_version=1,
        kind="preflight_gpu_hour_source_manifest",
        protocol_sha256=PREFLIGHT_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_member_sha256=member_sha256,
        materialization_receipt_sha256=final_evidence.materialization.sha256,
        stage_coverage_receipt_sha256=final_evidence.stage_coverage.sha256,
        final_evidence_sha256=final_evidence.sha256,
        remote_raw_receipt=remote_binding,
        source_authority=final_evidence.source_authority,
        activation_sha256=final_evidence.activation.sha256,
        pointer_coverage_sha256=final_evidence.coverage.sha256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=hardware_sha256,
        observations=observations,
        schedule_sha256=_preflight_schedule_sha256(observations),
    )
    destination = _absolute_path(
        "preflight GPU-hour source", source_manifest_output_path
    )
    publish_canonical_json_no_replace(destination, manifest.to_dict())
    binding = CanonicalJsonProofBinding.bind(destination)
    reopened = PreflightGpuHourSourceManifest.from_dict(binding.reopen())
    if reopened != manifest:
        raise RuntimeError("preflight GPU-hour source changed during publication")
    return StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=final_evidence.materialization.sha256,
        signed_pilot_receipt_sha256=manifest.sha256,
        schedule_sha256=manifest.schedule_sha256,
        estimate=_estimate_preflight_manifest(manifest),
    )


def revalidate_persisted_preflight_gpu_hour_source_manifest(
    source_manifest_path: str,
    *,
    envelope: StageGpuHourEnvelope,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    stage_coverage: object,
    inventory: GpuInventory,
    now_ns: int,
) -> PreflightGpuHourSourceManifest:
    """Reject legacy 1+1+8 sources until the qualification union exists."""

    raise FormalGpuHourLifecycleBlocked(
        "preflight_qualification_lifecycle_cost_schedule_missing"
    )

    from lightcone_spec.experiments.formal_preflight_execution import (
        FORMAL_PREFLIGHT_EXECUTION_PROTOCOL_SHA256,
        FormalPreflightRemoteRawEvidenceReceipt,
    )
    from lightcone_spec.experiments.preflight_authority import (
        materialize_pointer_preflight_coverage,
    )
    from lightcone_spec.experiments.preflight_interference import (
        FormalPreflightInterferenceProofArtifact,
    )
    from lightcone_spec.experiments.registry import build_industrial_registry

    if type(envelope) is not StageGpuHourEnvelope or envelope.schema_version != 2:
        raise TypeError("preflight GPU-hour revalidation requires schema-2 envelope")
    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    destination = _absolute_path(
        "persisted preflight GPU-hour source", source_manifest_path
    )
    before = CanonicalJsonProofBinding.bind(destination)
    persisted = PreflightGpuHourSourceManifest.from_dict(before.reopen())
    if (
        persisted.protocol_lock_sha256 != protocol_lock.sha256
        or persisted.runtime_authority_member_sha256 != member_sha256
        or persisted.materialization_receipt_sha256 != materialization.sha256
        or persisted.stage_coverage_receipt_sha256 != stage_coverage.sha256
        or persisted.inventory_sha256 != inventory.sha256
    ):
        raise ValueError("persisted preflight GPU-hour lineage differs")
    remote_binding = CanonicalJsonProofBinding.bind(
        persisted.remote_raw_receipt.absolute_path
    )
    remote = FormalPreflightRemoteRawEvidenceReceipt.from_dict(remote_binding.reopen())
    if (
        remote_binding != persisted.remote_raw_receipt
        or remote_binding.semantic_sha256 != remote.sha256
    ):
        raise ValueError("persisted preflight raw receipt differs")
    interference_artifact = FormalPreflightInterferenceProofArtifact.from_dict(
        persisted.source_authority.interference_proof_artifact.reopen()
    )
    if remote.interference_raw_batch != interference_artifact.raw_batch:
        raise ValueError("preflight raw/final interference evidence differs")
    activation, pointer_coverage = materialize_pointer_preflight_coverage(
        build_industrial_registry(), persisted.source_authority
    )
    expected_final_evidence_sha256 = content_sha256(
        {
            "schema_version": 2,
            "kind": "formal_preflight_final_evidence",
            "protocol_sha256": FORMAL_PREFLIGHT_EXECUTION_PROTOCOL_SHA256,
            "remote_raw_receipt_sha256": remote_binding.semantic_sha256,
            "source_authority_sha256": persisted.source_authority.sha256,
            "activation_sha256": activation.sha256,
            "coverage_sha256": pointer_coverage.sha256,
            "materialization_receipt_sha256": materialization.sha256,
            "stage_coverage_sha256": persisted.stage_coverage_receipt_sha256,
        }
    )
    if (
        persisted.activation_sha256 != activation.sha256
        or persisted.pointer_coverage_sha256 != pointer_coverage.sha256
        or persisted.final_evidence_sha256 != expected_final_evidence_sha256
    ):
        raise ValueError("persisted preflight final evidence identity differs")
    lifecycle_inputs = tuple(
        PreflightGpuHourLifecycleProofInput(
            materialized_cell_id=row.materialized_cell_id,
            lifecycle_proof_artifact_path=row.timing_proof.absolute_path,
        )
        for row in persisted.observations
        if row.phase_kind == "interference"
    )
    observations, hardware_sha256 = _preflight_observations_from_sources(
        protocol_lock=protocol_lock,
        materialization=materialization,
        stage_coverage=stage_coverage,
        source_authority=persisted.source_authority,
        inventory=inventory,
        lifecycle_proof_inputs=lifecycle_inputs,
        now_ns=now_ns,
    )
    expected = PreflightGpuHourSourceManifest(
        schema_version=1,
        kind="preflight_gpu_hour_source_manifest",
        protocol_sha256=PREFLIGHT_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_member_sha256=member_sha256,
        materialization_receipt_sha256=materialization.sha256,
        stage_coverage_receipt_sha256=persisted.stage_coverage_receipt_sha256,
        final_evidence_sha256=expected_final_evidence_sha256,
        remote_raw_receipt=remote_binding,
        source_authority=persisted.source_authority,
        activation_sha256=activation.sha256,
        pointer_coverage_sha256=pointer_coverage.sha256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=hardware_sha256,
        observations=observations,
        schedule_sha256=_preflight_schedule_sha256(observations),
    )
    expected_envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        signed_pilot_receipt_sha256=expected.sha256,
        schedule_sha256=expected.schedule_sha256,
        estimate=_estimate_preflight_manifest(expected),
    )
    after = CanonicalJsonProofBinding.bind(destination)
    if before != after:
        raise RuntimeError("preflight GPU-hour source changed while reopened")
    if persisted != expected:
        raise ValueError("preflight GPU-hour source differs from first-party timing")
    if envelope != expected_envelope:
        raise ValueError("preflight GPU-hour envelope differs from source derivation")
    return persisted


def materialize_stage_gpu_hour_envelope_from_lifecycle_proofs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    proof_inputs: tuple[LifecycleGpuHourProofInput, ...],
    source_manifest_output_path: str,
    now_ns: int,
) -> StageGpuHourEnvelope:
    """Deep-open exact lifecycle evidence and publish a schema-2 budget source."""

    validated, member_sha256 = _validate_inputs(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        inventory=inventory,
        proof_inputs=proof_inputs,
        now_ns=now_ns,
    )
    manifest = _source_manifest(
        protocol_lock=protocol_lock,
        runtime_authority_member_sha256=member_sha256,
        materialization=materialization,
        inventory=inventory,
        validated=validated,
    )
    destination = _absolute_path(
        "formal GPU-hour source manifest", source_manifest_output_path
    )
    publish_canonical_json_no_replace(destination, manifest.to_dict())
    binding = CanonicalJsonProofBinding.bind(destination)
    reopened = LifecycleGpuHourSourceManifest.from_dict(binding.reopen())
    if reopened != manifest:
        raise RuntimeError("formal GPU-hour source changed during publication")
    estimate = _estimate(manifest)
    return StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        signed_pilot_receipt_sha256=manifest.sha256,
        schedule_sha256=manifest.schedule_sha256,
        estimate=estimate,
    )


def materialize_lifecycle_gpu_hour_subset_source(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    expected_cell_ids: tuple[str, ...],
    proof_inputs: tuple[LifecycleGpuHourProofInput, ...],
    source_manifest_output_path: str,
    now_ns: int,
) -> LifecycleGpuHourSourceManifest:
    """Publish actual lifecycle evidence for one exact materialized subset.

    This generic subset path intentionally excludes profiler, model-preflight,
    and E5 fault-actuation rows.  Those rows require their dedicated lifecycle
    reducers and can never be relabelled as ordinary serving observations.
    """

    validated, member_sha256 = _validate_inputs(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        inventory=inventory,
        proof_inputs=proof_inputs,
        now_ns=now_ns,
        expected_cell_ids=expected_cell_ids,
    )
    manifest = _source_manifest(
        protocol_lock=protocol_lock,
        runtime_authority_member_sha256=member_sha256,
        materialization=materialization,
        inventory=inventory,
        validated=validated,
    )
    destination = _absolute_path(
        "formal GPU-hour subset source manifest", source_manifest_output_path
    )
    publish_canonical_json_no_replace(destination, manifest.to_dict())
    reopened = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(destination).reopen()
    )
    if reopened != manifest:
        raise RuntimeError("formal GPU-hour subset source changed during publication")
    return manifest


def _validate_e5_failure_gpu_hour_inputs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    proof_inputs: tuple[E5FailureGpuHourProofInput, ...],
    now_ns: int,
) -> tuple[tuple[E5FailureGpuHourObservation, ...], str, str]:
    """Deep-open the public E5 failure subject/lifecycle closed join."""

    from lightcone_spec.experiments.formal_failure_execution import (
        FormalFailureExecutionRebuildInput,
    )
    from lightcone_spec.orchestration.formal_failure_physical import (
        FormalE5FailureLifecycleProofArtifact,
        validate_formal_e5_failure_lifecycle_cost_proof_artifact,
    )

    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    if (
        type(materialization) is not StageMaterializationReceipt
        or materialization.stage != "E5"
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.materialization_rule
        != ("450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics")
    ):
        raise ValueError("E5 failure GPU-hour source requires the final matrix")
    if type(inventory) is not GpuInventory or len(inventory.devices) != 2:
        raise ValueError("E5 failure GPU-hour source requires exact two-GPU inventory")
    hardware = {device.hardware_envelope_sha256 for device in inventory.devices}
    if len(hardware) != 1:
        raise ValueError("E5 failure GPU-hour inventory mixes hardware envelopes")
    hardware_sha256 = next(iter(hardware))
    failure_ids = tuple(
        sorted(
            cell.cell_id
            for cell in materialization.cells
            if cell.task == "deterministic_failure_injection"
        )
    )
    if len(failure_ids) != 264:
        raise ValueError("E5 failure GPU-hour source lacks exact 264 failure rows")
    if (
        type(proof_inputs) is not tuple
        or len(proof_inputs) != 264
        or any(type(row) is not E5FailureGpuHourProofInput for row in proof_inputs)
    ):
        raise TypeError("E5 failure GPU-hour inputs must cover exact 264 typed rows")
    path_sets = (
        tuple(row.failure_execution_rebuild_input_path for row in proof_inputs),
        tuple(row.lifecycle_proof_artifact_path for row in proof_inputs),
    )
    if any(len(paths) != len(set(paths)) for paths in path_sets) or set(
        path_sets[0]
    ) & set(path_sets[1]):
        raise ValueError("E5 failure GPU-hour input paths repeat or alias")

    opened: list[
        tuple[
            CanonicalJsonProofBinding,
            CanonicalJsonProofBinding,
            object,
            str,
            ChallengeReplayReservationBinding,
        ]
    ] = []
    for item in proof_inputs:
        rebuild_binding = CanonicalJsonProofBinding.bind(
            item.failure_execution_rebuild_input_path
        )
        rebuild_input = FormalFailureExecutionRebuildInput.from_dict(
            rebuild_binding.reopen()
        )
        subject = rebuild_input.subject
        lifecycle_binding = CanonicalJsonProofBinding.bind(
            item.lifecycle_proof_artifact_path
        )
        lifecycle_artifact = FormalE5FailureLifecycleProofArtifact.from_dict(
            lifecycle_binding.reopen()
        )
        if lifecycle_artifact.sha256 != lifecycle_binding.semantic_sha256:
            raise ValueError("E5 failure GPU-hour lifecycle proof identity differs")
        projection = validate_formal_e5_failure_lifecycle_cost_proof_artifact(
            lifecycle_binding.absolute_path,
            failure_subject=subject,
            expected_protocol_lock_sha256=protocol_lock.sha256,
            expected_runtime_authority_manifest_sha256=(
                formal_runtime_authority_manifest.sha256
            ),
            expected_materialization_receipt_sha256=materialization.sha256,
            expected_materialized_cell_id=subject.materialized_cell_id,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=protocol_lock.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            now_ns=now_ns,
        )
        if (
            rebuild_input.expected_failure_execution_binding_sha256
            != projection.formal_failure_execution_binding_sha256
            or projection.failure_subject_sha256 != subject.sha256
            or projection.materialized_cell_id not in set(failure_ids)
            or projection.serving_execution_binding_sha256
            != subject.serving_execution_binding_sha256
            or projection.serving_execution_plan_sha256
            != subject.serving_execution_plan_sha256
            or projection.assignment_sha256 != subject.assignment_sha256
            or projection.run_nonce_sha256 != subject.run_nonce_sha256
            or projection.topology_mode != subject.topology
            or lifecycle_artifact.control_attestation.hardware_envelope_sha256
            != hardware_sha256
        ):
            raise ValueError("E5 failure GPU-hour subject/projection join differs")
        opened.append(
            (
                rebuild_binding,
                lifecycle_binding,
                projection,
                lifecycle_artifact.control_attestation.sha256,
                lifecycle_artifact.replay_reservation,
            )
        )
    opened.sort(
        key=lambda row: (
            row[2].execution_started_ns,
            row[2].materialized_cell_id,
        )
    )
    observations = tuple(
        E5FailureGpuHourObservation(
            materialized_cell_id=projection.materialized_cell_id,
            failure_execution_rebuild_input=rebuild_binding,
            lifecycle_proof=lifecycle_binding,
            projection=projection,
            control_envelope_sha256=control_sha256,
            replay_reservation=replay_reservation,
            wave_index=wave_index,
        )
        for wave_index, (
            rebuild_binding,
            lifecycle_binding,
            projection,
            control_sha256,
            replay_reservation,
        ) in enumerate(opened)
    )
    if tuple(sorted(row.materialized_cell_id for row in observations)) != failure_ids:
        raise ValueError("E5 failure GPU-hour proofs do not cover exact 264 rows")
    # Constructor-level checks close path/control/replay/run reuse, schedule,
    # fixed-provider arithmetic, and overlapping GPU reservation intervals.
    _e5_failure_cost(observations)
    return observations, member_sha256, hardware_sha256


def materialize_e5_failure_gpu_hour_source_manifest(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    proof_inputs: tuple[E5FailureGpuHourProofInput, ...],
    source_manifest_output_path: str,
    now_ns: int,
) -> E5FailureGpuHourSourceManifest:
    """Publish all 264 integrated fault-run costs without any projection."""

    observations, member_sha256, hardware_sha256 = _validate_e5_failure_gpu_hour_inputs(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        inventory=inventory,
        proof_inputs=proof_inputs,
        now_ns=now_ns,
    )
    manifest = E5FailureGpuHourSourceManifest(
        schema_version=1,
        kind="e5_failure_gpu_hour_source_manifest",
        protocol_sha256=E5_FAILURE_GPU_HOUR_SOURCE_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_member_sha256=member_sha256,
        runtime_authority_manifest_sha256=formal_runtime_authority_manifest.sha256,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        root_manifest_sha256=protocol_lock.offline_release_trust_root_sha256,
        hardware_envelope_sha256=hardware_sha256,
        observations=observations,
        cost=_e5_failure_cost(observations),
        schedule_sha256=_e5_failure_schedule_sha256(observations),
    )
    destination = _absolute_path(
        "E5 failure GPU-hour source manifest", source_manifest_output_path
    )
    publish_canonical_json_no_replace(destination, manifest.to_dict())
    reopened = E5FailureGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(destination).reopen()
    )
    if reopened != manifest:
        raise RuntimeError("E5 failure GPU-hour source changed during publication")
    return manifest


def revalidate_persisted_e5_failure_gpu_hour_source_manifest(
    source_manifest_path: str,
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    now_ns: int,
) -> E5FailureGpuHourSourceManifest:
    """Reopen all public E5 subjects and integrated lifecycle proofs."""

    destination = _absolute_path(
        "persisted E5 failure GPU-hour source", source_manifest_path
    )
    before = CanonicalJsonProofBinding.bind(destination)
    persisted = E5FailureGpuHourSourceManifest.from_dict(before.reopen())
    proof_inputs = tuple(
        E5FailureGpuHourProofInput(
            failure_execution_rebuild_input_path=(
                row.failure_execution_rebuild_input.absolute_path
            ),
            lifecycle_proof_artifact_path=row.lifecycle_proof.absolute_path,
        )
        for row in persisted.observations
    )
    observations, member_sha256, hardware_sha256 = _validate_e5_failure_gpu_hour_inputs(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        inventory=inventory,
        proof_inputs=proof_inputs,
        now_ns=now_ns,
    )
    expected = E5FailureGpuHourSourceManifest(
        schema_version=1,
        kind="e5_failure_gpu_hour_source_manifest",
        protocol_sha256=E5_FAILURE_GPU_HOUR_SOURCE_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_member_sha256=member_sha256,
        runtime_authority_manifest_sha256=formal_runtime_authority_manifest.sha256,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        root_manifest_sha256=protocol_lock.offline_release_trust_root_sha256,
        hardware_envelope_sha256=hardware_sha256,
        observations=observations,
        cost=_e5_failure_cost(observations),
        schedule_sha256=_e5_failure_schedule_sha256(observations),
    )
    after = CanonicalJsonProofBinding.bind(destination)
    if before != after:
        raise RuntimeError(
            "persisted E5 failure GPU-hour source changed while reopened"
        )
    if persisted != expected:
        raise ValueError("persisted E5 failure GPU-hour source differs from proofs")
    return persisted


def _derive_staged_prospective_gpu_hour_source(
    *,
    protocol_lock: ProtocolLock,
    runtime_authority_member_sha256: str,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    completed_source_binding: CanonicalJsonProofBinding | None,
    completed_source: LifecycleGpuHourSourceManifest | None,
) -> StagedProspectiveGpuHourSourceManifest:
    if materialization.stage not in _STAGED_PROSPECTIVE_STAGES:
        raise ValueError("staged prospective GPU hours do not cover this stage")
    _reject_untyped_special_gpu_hour_cells(materialization)
    if materialization.protocol_lock_sha256 != protocol_lock.sha256:
        raise ValueError("staged prospective materialization root differs")
    if type(inventory) is not GpuInventory or len(inventory.devices) != 2:
        raise ValueError("staged prospective GPU hours require exactly two GPUs")
    if (completed_source_binding is None) != (completed_source is None):
        raise ValueError("staged prospective completed source binding is incomplete")
    if completed_source is not None and (
        completed_source.protocol_lock_sha256 != protocol_lock.sha256
        or completed_source.runtime_authority_member_sha256
        != runtime_authority_member_sha256
        or completed_source.materialization_receipt_sha256 != materialization.sha256
        or completed_source.inventory_sha256 != inventory.sha256
        or completed_source.inventory_gpu_count != 2
    ):
        raise ValueError("staged prospective completed source lineage differs")
    observations = () if completed_source is None else completed_source.observations
    completed_ids = tuple(sorted(row.materialized_cell_id for row in observations))
    materialized_ids = {cell.cell_id for cell in materialization.cells}
    if set(completed_ids) - materialized_ids:
        raise ValueError("staged prospective source covers a foreign cell")
    strata = _staged_strata(
        materialization,
        completed_cell_ids=completed_ids,
    )
    missing = tuple(
        sorted(
            row.minimum_pilot_cell_id
            for row in strata
            if row.minimum_pilot_cell_id is not None
        )
    )
    actual = _staged_actual_cost(observations, inventory_gpu_count=2)
    projected: StagedProspectiveGpuHourCost | None = None
    total: StagedProspectiveGpuHourCost | None = None
    mapping_rows: tuple[tuple[object, ...], ...] = ()
    if not missing:
        projected, mapping_rows = _staged_projected_cost(
            materialization=materialization,
            observations=observations,
            strata=strata,
            inventory_gpu_count=2,
        )
        total = _sum_staged_costs("total", actual, projected)
    mapping_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "staged_scientific_stratum_gpu_hour_mapping",
            "protocol_sha256": STAGED_PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256,
            "protocol_lock_sha256": protocol_lock.sha256,
            "runtime_authority_member_sha256": runtime_authority_member_sha256,
            "materialization_receipt_sha256": materialization.sha256,
            "completed_source_manifest": (
                None
                if completed_source_binding is None
                else completed_source_binding.to_dict()
            ),
            "strata": tuple(row.to_dict() for row in strata),
            "minimum_pilot_cell_ids": missing,
            "mapping_rows": mapping_rows,
            "schedule": "isolated_fail_closed_two_gpu_provider_reservation",
        }
    )
    return StagedProspectiveGpuHourSourceManifest(
        schema_version=1,
        kind="staged_prospective_gpu_hour_source_manifest",
        protocol_sha256=STAGED_PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_member_sha256=runtime_authority_member_sha256,
        stage=materialization.stage,
        materialization_receipt_sha256=materialization.sha256,
        completed_source_manifest=completed_source_binding,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=(
            None
            if completed_source is None
            else completed_source.hardware_envelope_sha256
        ),
        status="BLOCKED" if missing else "READY",
        strata=strata,
        minimum_pilot_cell_ids=missing,
        actual_completed=actual,
        projected_remaining=projected,
        total=total,
        mapping_sha256=mapping_sha256,
    )


def _staged_prospective_envelope(
    manifest: StagedProspectiveGpuHourSourceManifest,
) -> StageGpuHourEnvelope:
    if manifest.status != "READY" or manifest.total is None:
        raise FormalGpuHourLifecycleBlocked("scientific_strata_unmeasured")
    total = manifest.total

    def hours(value: int) -> float:
        return float(value / NANOSECONDS_PER_HOUR)

    values = {
        "source_pilot_receipt_sha256": manifest.sha256,
        "source_schedule_sha256": manifest.mapping_sha256,
        "source_materialization_receipt_sha256": (
            manifest.materialization_receipt_sha256
        ),
        "source_inventory_gpu_count": 2,
        "compute_gpu_hours": hours(total.compute_gpu_ns),
        "reserved_gpu_hours": hours(total.reserved_gpu_ns),
        "estimated_wall_hours": hours(total.wall_ns),
        "retry_reserve_gpu_hours": hours(total.retry_reserve_gpu_ns),
        "profile_reserve_gpu_hours": hours(total.profile_reserve_gpu_ns),
        "evidence_reserve_gpu_hours": hours(total.evidence_reserve_gpu_ns),
    }
    estimate = GpuHourEstimate(
        status="AVAILABLE",
        derivation_sha256=content_sha256(values),
        **values,
    )
    return StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=manifest.protocol_lock_sha256,
        materialization_receipt_sha256=(manifest.materialization_receipt_sha256),
        signed_pilot_receipt_sha256=manifest.sha256,
        schedule_sha256=manifest.mapping_sha256,
        estimate=estimate,
    )


def _reopen_completed_staged_source(
    *,
    source_manifest_path: str,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    now_ns: int,
) -> tuple[CanonicalJsonProofBinding, LifecycleGpuHourSourceManifest]:
    binding = CanonicalJsonProofBinding.bind(source_manifest_path)
    unverified = LifecycleGpuHourSourceManifest.from_dict(binding.reopen())
    completed_ids = tuple(
        sorted(row.materialized_cell_id for row in unverified.observations)
    )
    subset_envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        signed_pilot_receipt_sha256=unverified.sha256,
        schedule_sha256=unverified.schedule_sha256,
        estimate=_estimate(unverified),
    )
    source = revalidate_persisted_stage_gpu_hour_source_manifest(
        binding.absolute_path,
        envelope=subset_envelope,
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        inventory=inventory,
        now_ns=now_ns,
        expected_cell_ids=completed_ids,
    )
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("staged prospective completed source changed")
    return binding, source


def materialize_staged_prospective_gpu_hour_envelope(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    source_manifest_output_path: str,
    now_ns: int,
    completed_source_manifest_path: str | None = None,
) -> tuple[
    StagedProspectiveGpuHourSourceManifest,
    StageGpuHourEnvelope | None,
]:
    """Publish an honest early-stage projection or an explicit BLOCKED plan.

    No duration or cell-count scalar enters this API.  A missing completed
    source is a valid planning call and yields the minimum pilot cell for every
    uncovered scientific stratum; it can never yield a signable envelope.
    """

    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    binding: CanonicalJsonProofBinding | None = None
    source: LifecycleGpuHourSourceManifest | None = None
    if completed_source_manifest_path is not None:
        binding, source = _reopen_completed_staged_source(
            source_manifest_path=completed_source_manifest_path,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=formal_runtime_authority_manifest,
            materialization=materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
    manifest = _derive_staged_prospective_gpu_hour_source(
        protocol_lock=protocol_lock,
        runtime_authority_member_sha256=member_sha256,
        materialization=materialization,
        inventory=inventory,
        completed_source_binding=binding,
        completed_source=source,
    )
    destination = _absolute_path(
        "staged prospective GPU-hour source", source_manifest_output_path
    )
    publish_canonical_json_no_replace(destination, manifest.to_dict())
    reopened = StagedProspectiveGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(destination).reopen()
    )
    if reopened != manifest:
        raise RuntimeError("staged prospective GPU-hour source changed")
    return (
        manifest,
        None
        if manifest.status == "BLOCKED"
        else _staged_prospective_envelope(manifest),
    )


def revalidate_persisted_staged_prospective_gpu_hour_source_manifest(
    source_manifest_path: str,
    *,
    envelope: StageGpuHourEnvelope,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    now_ns: int,
) -> StagedProspectiveGpuHourSourceManifest:
    """Deep-rebuild a READY staged source before registry consumption."""

    path = _absolute_path(
        "persisted staged prospective GPU-hour source", source_manifest_path
    )
    before = CanonicalJsonProofBinding.bind(path)
    persisted = StagedProspectiveGpuHourSourceManifest.from_dict(before.reopen())
    if persisted.status != "READY" or persisted.completed_source_manifest is None:
        raise FormalGpuHourLifecycleBlocked("scientific_strata_unmeasured")
    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    completed_binding, completed_source = _reopen_completed_staged_source(
        source_manifest_path=persisted.completed_source_manifest.absolute_path,
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        inventory=inventory,
        now_ns=now_ns,
    )
    if completed_binding != persisted.completed_source_manifest:
        raise ValueError("staged prospective nested source binding differs")
    expected = _derive_staged_prospective_gpu_hour_source(
        protocol_lock=protocol_lock,
        runtime_authority_member_sha256=member_sha256,
        materialization=materialization,
        inventory=inventory,
        completed_source_binding=completed_binding,
        completed_source=completed_source,
    )
    after = CanonicalJsonProofBinding.bind(path)
    if before != after:
        raise RuntimeError("staged prospective GPU-hour source changed while reopened")
    if persisted != expected or envelope != _staged_prospective_envelope(expected):
        raise ValueError("staged prospective GPU-hour source differs from proofs")
    return persisted


def materialize_prospective_stage_gpu_hour_envelope(
    *,
    authority: VerifiedProspectiveGpuHourAuthority,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    pilot_materialization: StageMaterializationReceipt,
    pilot_envelope: StageGpuHourEnvelope,
    pilot_source_manifest_path: str,
    final_materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    prospective_source_manifest_output_path: str,
    now_ns: int,
    one_shot_proof_inputs: tuple[
        E5FailureGpuHourProofInput | LifecycleGpuHourProofInput, ...
    ] = (),
    one_shot_source_manifest_output_path: str | None = None,
    existing_one_shot_source_manifest_path: str | None = None,
) -> tuple[ProspectiveGpuHourSourceManifest, StageGpuHourEnvelope]:
    """Project a powered final prefix without caller-authored durations.

    E3b has actual tuning plus projected final rows.  E6 additionally charges
    its two already-executed model preflights once.  E5 requires all 264
    diagnostic rows to carry their dedicated integrated failure lifecycles.
    """

    if (
        type(authority) is not VerifiedProspectiveGpuHourAuthority
        or authority._construction_seal
        is not _VERIFIED_PROSPECTIVE_GPU_HOUR_AUTHORITY_SEAL
    ):
        raise TypeError("prospective GPU hours require sealed power authority")
    if (
        authority.stage != pilot_materialization.stage
        or authority.stage != final_materialization.stage
    ):
        raise ValueError("prospective GPU-hour stage differs")
    if (
        authority.pilot_materialization_receipt_sha256 != pilot_materialization.sha256
        or authority.final_materialization_receipt_sha256
        != final_materialization.sha256
        or pilot_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or final_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or type(inventory) is not GpuInventory
        or len(inventory.devices) != 2
    ):
        raise ValueError("prospective GPU-hour immutable lineage differs")
    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    pilot_source = revalidate_persisted_stage_gpu_hour_source_manifest(
        pilot_source_manifest_path,
        envelope=pilot_envelope,
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=pilot_materialization,
        inventory=inventory,
        now_ns=now_ns,
    )
    pilot_cells = {cell.cell_id: cell for cell in pilot_materialization.cells}
    pilot_tuning_ids = tuple(
        sorted(
            cell.cell_id
            for cell in pilot_materialization.cells
            if type(dict(cell.dimensions).get("block")) is int
        )
    )
    pilot_one_shot_ids = tuple(sorted(set(pilot_cells) - set(pilot_tuning_ids)))
    pilot_by_id = {row.materialized_cell_id: row for row in pilot_source.observations}
    tuning_observations = tuple(pilot_by_id[cell_id] for cell_id in pilot_tuning_ids)
    one_shot_pilot_observations = tuple(
        pilot_by_id[cell_id] for cell_id in pilot_one_shot_ids
    )
    if {row.wave_index for row in tuning_observations} & {
        row.wave_index for row in one_shot_pilot_observations
    }:
        raise ValueError(
            "prospective tuning and one-shot cells overlap one reservation wave"
        )
    tuning_cost = _cost_from_actual_observations(
        category="actual_tuning",
        observations=tuning_observations,
        inventory_gpu_count=2,
    )
    projected_cost, mapping_sha256 = _project_final_cost(
        pilot_materialization=pilot_materialization,
        pilot_source=pilot_source,
        final_materialization=final_materialization,
    )

    one_shot_binding: CanonicalJsonProofBinding | None = None
    if authority.stage == "E5":
        failure_ids = tuple(
            sorted(
                cell.cell_id
                for cell in final_materialization.cells
                if cell.task == "deterministic_failure_injection"
            )
        )
        if len(failure_ids) != 264:
            raise ValueError("E5 prospective source lacks exact 264 failure rows")
        if existing_one_shot_source_manifest_path is not None:
            if (
                one_shot_proof_inputs
                or one_shot_source_manifest_output_path is not None
            ):
                raise ValueError("E5 one-shot lifecycle sources are ambiguous")
            one_shot_binding = CanonicalJsonProofBinding.bind(
                existing_one_shot_source_manifest_path
            )
            E5FailureGpuHourSourceManifest.from_dict(one_shot_binding.reopen())
            one_shot_source = revalidate_persisted_e5_failure_gpu_hour_source_manifest(
                one_shot_binding.absolute_path,
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=(formal_runtime_authority_manifest),
                materialization=final_materialization,
                inventory=inventory,
                now_ns=now_ns,
            )
        else:
            if one_shot_source_manifest_output_path is None:
                raise FormalGpuHourLifecycleBlocked(
                    "e5_264_integrated_failure_lifecycle_source_path_missing"
                )
            if any(
                type(item) is not E5FailureGpuHourProofInput
                for item in one_shot_proof_inputs
            ):
                raise TypeError(
                    "E5 ordinary serving lifecycle cannot authorize failure cost"
                )
            one_shot_source = materialize_e5_failure_gpu_hour_source_manifest(
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=(formal_runtime_authority_manifest),
                materialization=final_materialization,
                inventory=inventory,
                proof_inputs=one_shot_proof_inputs,  # type: ignore[arg-type]
                source_manifest_output_path=one_shot_source_manifest_output_path,
                now_ns=now_ns,
            )
            one_shot_binding = CanonicalJsonProofBinding.bind(
                one_shot_source_manifest_output_path
            )
        if (
            one_shot_source.hardware_envelope_sha256
            != pilot_source.hardware_envelope_sha256
        ):
            raise ValueError("E5 one-shot lifecycle hardware envelope differs")
        one_shot_cost = one_shot_source.cost
        if one_shot_cost.cell_count != 264:
            raise ValueError("E5 failure lifecycle cost is not 264 actual rows")
    elif authority.stage == "E6":
        if (
            one_shot_proof_inputs
            or one_shot_source_manifest_output_path is not None
            or existing_one_shot_source_manifest_path is not None
        ):
            raise ValueError("E6 preflight is already actual in its pilot source")
        if len(one_shot_pilot_observations) != 2:
            raise ValueError("E6 prospective source requires two actual preflights")
        one_shot_cost = _cost_from_actual_observations(
            category="actual_one_shot",
            observations=one_shot_pilot_observations,
            inventory_gpu_count=2,
        )
    else:
        if (
            one_shot_pilot_observations
            or one_shot_proof_inputs
            or one_shot_source_manifest_output_path is not None
            or existing_one_shot_source_manifest_path is not None
        ):
            raise ValueError("stage has unregistered prospective one-shot rows")
        one_shot_cost = None

    costs = (
        (tuning_cost, projected_cost, one_shot_cost)
        if one_shot_cost is not None
        else (tuning_cost, projected_cost)
    )
    manifest = ProspectiveGpuHourSourceManifest(
        schema_version=1,
        kind="prospective_gpu_hour_source_manifest",
        protocol_sha256=PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_member_sha256=member_sha256,
        stage=authority.stage,  # type: ignore[arg-type]
        final_materialization_receipt_sha256=final_materialization.sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_source_manifest=CanonicalJsonProofBinding.bind(
            pilot_source_manifest_path
        ),
        one_shot_source_manifest=one_shot_binding,
        prospective_authority_sha256=authority.sha256,
        signed_power_authority_sha256=authority.signed_authority_sha256,
        signed_power_challenge_sha256=(authority.signed_authority_challenge_sha256),
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        mapping_sha256=mapping_sha256,
        costs=costs,  # type: ignore[arg-type]
    )
    destination = _absolute_path(
        "prospective GPU-hour source", prospective_source_manifest_output_path
    )
    publish_canonical_json_no_replace(destination, manifest.to_dict())
    reopened = ProspectiveGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(destination).reopen()
    )
    if reopened != manifest:
        raise RuntimeError("prospective GPU-hour source changed during publication")
    estimate = _estimate_prospective_manifest(manifest)
    envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=final_materialization.sha256,
        signed_pilot_receipt_sha256=manifest.sha256,
        schedule_sha256=manifest.mapping_sha256,
        estimate=estimate,
    )
    return manifest, envelope


def revalidate_persisted_prospective_gpu_hour_source_manifest(
    source_manifest_path: str,
    *,
    envelope: StageGpuHourEnvelope,
    authority: VerifiedProspectiveGpuHourAuthority,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    pilot_materialization: StageMaterializationReceipt,
    final_materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    now_ns: int,
) -> ProspectiveGpuHourSourceManifest:
    """Reopen every actual proof and recompute a prospective stage budget.

    The signed envelope is only a summary.  Pilot timing, E5 one-shot timing,
    the exact pilot-to-final mapping, and every reserve component are rebuilt
    from the path-bound manifests before the summary can enter the registry.
    """

    if type(envelope) is not StageGpuHourEnvelope or envelope.schema_version != 2:
        raise TypeError("prospective GPU-hour revalidation requires schema-2 envelope")
    if (
        type(authority) is not VerifiedProspectiveGpuHourAuthority
        or authority._construction_seal
        is not _VERIFIED_PROSPECTIVE_GPU_HOUR_AUTHORITY_SEAL
    ):
        raise TypeError("prospective GPU-hour revalidation requires sealed authority")
    if (
        authority.stage != pilot_materialization.stage
        or authority.stage != final_materialization.stage
        or authority.pilot_materialization_receipt_sha256
        != pilot_materialization.sha256
        or authority.final_materialization_receipt_sha256
        != final_materialization.sha256
        or final_materialization.source_decision_sha256
        != authority.signed_authority_sha256
        or pilot_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or final_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or type(inventory) is not GpuInventory
        or len(inventory.devices) != 2
    ):
        raise ValueError("prospective GPU-hour immutable authority differs")
    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    destination = _absolute_path(
        "persisted prospective GPU-hour source", source_manifest_path
    )
    before = CanonicalJsonProofBinding.bind(destination)
    persisted = ProspectiveGpuHourSourceManifest.from_dict(before.reopen())
    if (
        persisted.protocol_lock_sha256 != protocol_lock.sha256
        or persisted.runtime_authority_member_sha256 != member_sha256
        or persisted.stage != authority.stage
        or persisted.final_materialization_receipt_sha256
        != final_materialization.sha256
        or persisted.pilot_materialization_receipt_sha256
        != pilot_materialization.sha256
        or persisted.prospective_authority_sha256 != authority.sha256
        or persisted.signed_power_authority_sha256 != authority.signed_authority_sha256
        or persisted.signed_power_challenge_sha256
        != authority.signed_authority_challenge_sha256
        or persisted.inventory_sha256 != inventory.sha256
    ):
        raise ValueError("persisted prospective GPU-hour lineage differs")

    pilot_binding = CanonicalJsonProofBinding.bind(
        persisted.pilot_source_manifest.absolute_path
    )
    if pilot_binding != persisted.pilot_source_manifest:
        raise ValueError("prospective pilot source binding changed")
    unverified_pilot = LifecycleGpuHourSourceManifest.from_dict(pilot_binding.reopen())
    pilot_envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=pilot_materialization.sha256,
        signed_pilot_receipt_sha256=unverified_pilot.sha256,
        schedule_sha256=unverified_pilot.schedule_sha256,
        estimate=_estimate(unverified_pilot),
    )
    pilot_source = revalidate_persisted_stage_gpu_hour_source_manifest(
        pilot_binding.absolute_path,
        envelope=pilot_envelope,
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=pilot_materialization,
        inventory=inventory,
        now_ns=now_ns,
    )
    pilot_cells = {cell.cell_id: cell for cell in pilot_materialization.cells}
    pilot_by_id = {row.materialized_cell_id: row for row in pilot_source.observations}
    tuning_ids = tuple(
        sorted(
            cell.cell_id
            for cell in pilot_materialization.cells
            if type(dict(cell.dimensions).get("block")) is int
        )
    )
    one_shot_pilot_ids = tuple(sorted(set(pilot_cells) - set(tuning_ids)))
    tuning_observations = tuple(pilot_by_id[cell_id] for cell_id in tuning_ids)
    one_shot_pilot_observations = tuple(
        pilot_by_id[cell_id] for cell_id in one_shot_pilot_ids
    )
    if {row.wave_index for row in tuning_observations} & {
        row.wave_index for row in one_shot_pilot_observations
    }:
        raise ValueError(
            "prospective tuning and one-shot cells overlap one reservation wave"
        )
    tuning_cost = _cost_from_actual_observations(
        category="actual_tuning",
        observations=tuning_observations,
        inventory_gpu_count=2,
    )
    projected_cost, mapping_sha256 = _project_final_cost(
        pilot_materialization=pilot_materialization,
        pilot_source=pilot_source,
        final_materialization=final_materialization,
    )

    if authority.stage == "E5":
        failure_ids = tuple(
            sorted(
                cell.cell_id
                for cell in final_materialization.cells
                if cell.task == "deterministic_failure_injection"
            )
        )
        if len(failure_ids) != 264 or persisted.one_shot_source_manifest is None:
            raise ValueError("E5 prospective source lacks exact 264 one-shot rows")
        one_binding = CanonicalJsonProofBinding.bind(
            persisted.one_shot_source_manifest.absolute_path
        )
        if one_binding != persisted.one_shot_source_manifest:
            raise ValueError("E5 one-shot source binding changed")
        E5FailureGpuHourSourceManifest.from_dict(one_binding.reopen())
        one_source = revalidate_persisted_e5_failure_gpu_hour_source_manifest(
            one_binding.absolute_path,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=formal_runtime_authority_manifest,
            materialization=final_materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
        if one_source.hardware_envelope_sha256 != pilot_source.hardware_envelope_sha256:
            raise ValueError("E5 failure lifecycle hardware envelope differs")
        one_shot_cost = one_source.cost
        costs: tuple[ProspectiveGpuHourCost, ...] = (
            tuning_cost,
            projected_cost,
            one_shot_cost,
        )
    elif authority.stage == "E6":
        if (
            persisted.one_shot_source_manifest is not None
            or len(one_shot_pilot_observations) != 2
        ):
            raise ValueError("E6 prospective source requires two actual preflights")
        one_shot_cost = _cost_from_actual_observations(
            category="actual_one_shot",
            observations=one_shot_pilot_observations,
            inventory_gpu_count=2,
        )
        costs = (tuning_cost, projected_cost, one_shot_cost)
    else:
        if (
            persisted.one_shot_source_manifest is not None
            or one_shot_pilot_observations
        ):
            raise ValueError("stage has unregistered prospective one-shot rows")
        costs = (tuning_cost, projected_cost)

    expected = ProspectiveGpuHourSourceManifest(
        schema_version=1,
        kind="prospective_gpu_hour_source_manifest",
        protocol_sha256=PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_member_sha256=member_sha256,
        stage=authority.stage,  # type: ignore[arg-type]
        final_materialization_receipt_sha256=final_materialization.sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_source_manifest=pilot_binding,
        one_shot_source_manifest=persisted.one_shot_source_manifest,
        prospective_authority_sha256=authority.sha256,
        signed_power_authority_sha256=authority.signed_authority_sha256,
        signed_power_challenge_sha256=(authority.signed_authority_challenge_sha256),
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        mapping_sha256=mapping_sha256,
        costs=costs,  # type: ignore[arg-type]
    )
    after = CanonicalJsonProofBinding.bind(destination)
    expected_envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=final_materialization.sha256,
        signed_pilot_receipt_sha256=expected.sha256,
        schedule_sha256=expected.mapping_sha256,
        estimate=_estimate_prospective_manifest(expected),
    )
    if before != after:
        raise RuntimeError("prospective GPU-hour source changed while reopened")
    if persisted != expected:
        raise ValueError("prospective GPU-hour source differs from actual proofs")
    if envelope != expected_envelope:
        raise ValueError("prospective GPU-hour envelope differs from source derivation")
    return persisted


def revalidate_stage_gpu_hour_source_manifest(
    source_manifest_path: str,
    *,
    envelope: StageGpuHourEnvelope,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    proof_inputs: tuple[LifecycleGpuHourProofInput, ...],
    now_ns: int,
) -> LifecycleGpuHourSourceManifest:
    """Rebuild the formal budget from sources; never trust the signed summary."""

    if type(envelope) is not StageGpuHourEnvelope or envelope.schema_version != 2:
        raise TypeError("formal GPU-hour revalidation requires schema-2 envelope")
    path = _absolute_path("formal GPU-hour source manifest", source_manifest_path)
    before = CanonicalJsonProofBinding.bind(path)
    persisted = LifecycleGpuHourSourceManifest.from_dict(before.reopen())
    validated, member_sha256 = _validate_inputs(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        inventory=inventory,
        proof_inputs=proof_inputs,
        now_ns=now_ns,
    )
    expected = _source_manifest(
        protocol_lock=protocol_lock,
        runtime_authority_member_sha256=member_sha256,
        materialization=materialization,
        inventory=inventory,
        validated=validated,
    )
    after = CanonicalJsonProofBinding.bind(path)
    expected_estimate = _estimate(expected)
    expected_envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        signed_pilot_receipt_sha256=expected.sha256,
        schedule_sha256=expected.schedule_sha256,
        estimate=expected_estimate,
    )
    if before != after:
        raise RuntimeError("formal GPU-hour source changed while reopened")
    if persisted != expected:
        raise ValueError("formal GPU-hour source differs from lifecycle proofs")
    if envelope != expected_envelope:
        raise ValueError("formal GPU-hour envelope differs from source derivation")
    return persisted


def revalidate_persisted_stage_gpu_hour_source_manifest(
    source_manifest_path: str,
    *,
    envelope: StageGpuHourEnvelope,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    now_ns: int,
    expected_cell_ids: tuple[str, ...] | None = None,
) -> LifecycleGpuHourSourceManifest:
    """Reopen a signed source manifest without trusting its numeric summary.

    Every observation carries a root-controlled durable projection of the
    private-sealed execution binding.  The projection is reopened before the
    lifecycle proof, closing the otherwise possible relabeling of a genuine
    timing proof onto another materialized cell.
    """

    if type(envelope) is not StageGpuHourEnvelope or envelope.schema_version != 2:
        raise TypeError("persisted formal GPU-hour source requires schema-2 envelope")
    member_sha256 = _require_runtime_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("persisted formal GPU-hour source requires materialization")
    if (
        materialization.protocol_lock_sha256 != protocol_lock.sha256
        or type(inventory) is not GpuInventory
        or len(inventory.devices) != 2
    ):
        raise ValueError("persisted formal GPU-hour source identity differs")
    path = _absolute_path("persisted formal GPU-hour source", source_manifest_path)
    before = CanonicalJsonProofBinding.bind(path)
    persisted = LifecycleGpuHourSourceManifest.from_dict(before.reopen())
    materialized_cell_ids = tuple(row.cell_id for row in materialization.cells)
    required_cell_ids = (
        materialized_cell_ids if expected_cell_ids is None else expected_cell_ids
    )
    _reject_untyped_special_gpu_hour_cells(
        materialization,
        expected_cell_ids=required_cell_ids,
    )
    if required_cell_ids != tuple(sorted(set(required_cell_ids))) or set(
        required_cell_ids
    ) - set(materialized_cell_ids):
        raise ValueError("persisted formal GPU-hour requested subset is not exact")
    observed_cell_ids = tuple(
        sorted(row.materialized_cell_id for row in persisted.observations)
    )
    if observed_cell_ids != required_cell_ids:
        raise ValueError("persisted formal GPU-hour source cell coverage differs")
    if (
        persisted.protocol_lock_sha256 != protocol_lock.sha256
        or persisted.runtime_authority_member_sha256 != member_sha256
        or persisted.materialization_receipt_sha256 != materialization.sha256
        or persisted.inventory_sha256 != inventory.sha256
        or persisted.inventory_gpu_count != len(inventory.devices)
    ):
        raise ValueError("persisted formal GPU-hour source lineage differs")

    hardware: set[str] = set()
    for observation in persisted.observations:
        native = observation.native_run_binding
        execution = validate_formal_serving_execution_proof_artifact(
            observation.execution_proof.absolute_path,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=formal_runtime_authority_manifest,
            materialization=materialization,
            inventory=inventory,
            expected_cell_id=observation.materialized_cell_id,
            now_ns=now_ns,
        )
        execution_artifact = _load_formal_serving_execution_proof_artifact(
            observation.execution_proof.absolute_path
        )
        _proof_binding, lifecycle_reservation, verified = (
            _validate_serving_lifecycle_timing(
                proof_path=observation.lifecycle_proof.absolute_path,
                native=native,
                topology_mode=execution.topology_mode,
                gpu_uuids=execution.gpu_uuids,
                hardware_envelope_sha256=execution.hardware_envelope_sha256,
                telemetry_detail=observation.telemetry_detail,
                protocol_lock=protocol_lock,
                inventory=inventory,
                now_ns=now_ns,
            )
        )
        if (
            CanonicalJsonProofBinding.bind(observation.execution_proof.absolute_path)
            != observation.execution_proof
            or execution.sha256 != observation.execution_proof_payload_sha256
            or execution.execution_binding_sha256
            != observation.execution_binding_sha256
            or execution.materialized_cell_id != observation.materialized_cell_id
            or execution.materialization_receipt_sha256 != materialization.sha256
            or execution.stage != materialization.stage
            or execution_artifact.control_attestation.sha256
            != observation.execution_control_envelope_sha256
            or execution_artifact.replay_reservation
            != observation.execution_replay_reservation
            or execution.execution_plan_sha256 != native.execution_plan_sha256
            or execution.rank_config_sha256 != native.rank_config_sha256
            or execution.run_id != native.run_id
            or execution.run_nonce_sha256 != native.run_nonce_sha256
            or execution.attempt_id != native.attempt_id
            or execution.method != native.method
            or execution.gpu_uuids != observation.gpu_uuids
            or execution.hardware_envelope_sha256 != persisted.hardware_envelope_sha256
            or lifecycle_reservation != observation.lifecycle_replay_reservation
            or CanonicalJsonProofBinding.bind(observation.lifecycle_proof.absolute_path)
            != observation.lifecycle_proof
            or verified.sha256 != observation.verified_lifecycle_proof_sha256
            or verified.raw_timing_sha256 != observation.raw_timing_sha256
            or verified.live_run_receipt_sha256 != observation.live_run_receipt_sha256
            or verified.native_result_proof_sha256
            != observation.native_result_proof_sha256
            or verified.run_binding_sha256 != observation.run_binding_sha256
            or verified.control_envelope_sha256 != observation.control_envelope_sha256
            or verified.replay_reservation_sha256
            != observation.lifecycle_replay_reservation.reservation_sha256
            or verified.run_id != native.run_id
            or verified.run_nonce_sha256 != native.run_nonce_sha256
            or verified.execution_plan_sha256 != native.execution_plan_sha256
            or verified.rank_config_sha256 != native.rank_config_sha256
            or verified.attempt_id != native.attempt_id
            or verified.method != native.method
            or verified.inventory_sha256 != inventory.sha256
            or verified.registry_sha256 != protocol_lock.registry_sha256
            or verified.root_manifest_sha256
            != protocol_lock.offline_release_trust_root_sha256
            or verified.hardware_envelope_sha256 != persisted.hardware_envelope_sha256
            or verified.gpu_uuids != observation.gpu_uuids
            or verified.telemetry_detail != observation.telemetry_detail
            or verified.phase_edges_ns != observation.phase_edges_ns
            or verified.phase_durations_ns != observation.phase_durations_ns
        ):
            raise ValueError("persisted formal GPU-hour proof differs from source")
        hardware.add(verified.hardware_envelope_sha256)
    if hardware != {persisted.hardware_envelope_sha256}:
        raise ValueError("persisted formal GPU-hour hardware coverage differs")

    expected_envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        signed_pilot_receipt_sha256=persisted.sha256,
        schedule_sha256=persisted.schedule_sha256,
        estimate=_estimate(persisted),
    )
    after = CanonicalJsonProofBinding.bind(path)
    if before != after:
        raise RuntimeError("persisted formal GPU-hour source changed while reopened")
    if envelope != expected_envelope:
        raise ValueError("persisted formal GPU-hour envelope differs from sources")
    return persisted


FORMAL_LAUNCH_CAP_SCHEDULE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 4,
        "kind": "lightcone_formal_launch_cap_schedule_protocol",
        "authority": "deep_rebuilt_gpu_hour_source_and_exact_materialization",
        "cell_state": "preconsumed_or_launchable_closed_union",
        "process_cap": "per_cell_process_occupied_hard_timeout",
        "provider_cap": "separate_wave_atomic_fixed_two_gpu_reservation_timeout",
        "retry": "one_additional_attempt_subject_to_global_retry_reserve",
        "projected": "same_stratum_pilot_ceiling_without_caller_durations",
        "nonserving": (
            "E4_profiler_and_E6_model_preflight_require_dedicated_cost_schedules"
        ),
        "preflight": (
            "blocked_until_scientific_1_plus_1_plus_8_and_resolved_"
            "qualification_lifecycles_share_admission_bound_caps"
        ),
    }
)
FORMAL_LAUNCH_CAP_ALLOWED_ATTEMPTS = 2


@dataclass(frozen=True)
class FormalLaunchCellCap:
    """One process cap and its atomic two-GPU provider wave allocation."""

    materialized_cell_id: str
    disposition: Literal["PRECONSUMED", "LAUNCHABLE"]
    stratum_sha256: str
    wave_group_sha256: str
    gpu_count: Literal[1, 2]
    provider_reserved_gpu_count: Literal[1, 2]
    allowed_attempts: Literal[2]
    process_hard_timeout_ns_per_attempt: int
    provider_wave_hard_timeout_ns_per_attempt: int
    maximum_compute_gpu_ns_per_attempt: int
    maximum_provider_reserved_gpu_ns_per_attempt: int

    def __post_init__(self) -> None:
        for label, digest in (
            ("cell", self.materialized_cell_id),
            ("stratum", self.stratum_sha256),
            ("wave", self.wave_group_sha256),
        ):
            _sha256(f"formal launch cap {label}", digest)
        if self.disposition not in {"PRECONSUMED", "LAUNCHABLE"}:
            raise ValueError("formal launch cap disposition is unsupported")
        if (
            self.gpu_count not in {1, 2}
            or self.provider_reserved_gpu_count not in {1, 2}
            or self.allowed_attempts != FORMAL_LAUNCH_CAP_ALLOWED_ATTEMPTS
        ):
            raise ValueError("formal launch cap GPU/attempt identity differs")
        for label, value in (
            ("process timeout", self.process_hard_timeout_ns_per_attempt),
            (
                "provider wave timeout",
                self.provider_wave_hard_timeout_ns_per_attempt,
            ),
            ("compute", self.maximum_compute_gpu_ns_per_attempt),
            ("provider", self.maximum_provider_reserved_gpu_ns_per_attempt),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"formal launch cap {label} must be positive")
        if (
            self.maximum_compute_gpu_ns_per_attempt
            != self.process_hard_timeout_ns_per_attempt * self.gpu_count
            or self.maximum_provider_reserved_gpu_ns_per_attempt
            != self.provider_wave_hard_timeout_ns_per_attempt
            * self.provider_reserved_gpu_count
            or self.process_hard_timeout_ns_per_attempt
            > self.provider_wave_hard_timeout_ns_per_attempt
        ):
            raise ValueError("formal launch cap per-attempt charge differs")

    @property
    def hard_timeout_ns_per_attempt(self) -> int:
        """Compatibility alias for the process kill deadline, never provider wall."""

        return self.process_hard_timeout_ns_per_attempt

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_dict(
                "formal launch cell cap",
                value,
                set(cls.__dataclass_fields__),
            )
        )


@dataclass(frozen=True)
class FormalLaunchCapSchedule:
    """Deterministic launch limits with already-spent work made explicit."""

    schema_version: Literal[2]
    kind: Literal["lightcone_formal_launch_cap_schedule"]
    protocol_sha256: str
    source_kind: str
    source_manifest_sha256: str
    source_schedule_sha256: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    cell_caps: tuple[FormalLaunchCellCap, ...]
    preconsumed_compute_gpu_ns: int
    preconsumed_provider_reserved_gpu_ns: int
    preconsumed_wall_ns: int
    launchable_compute_gpu_ns: int
    launchable_provider_reserved_gpu_ns: int
    launchable_wall_ns: int
    retry_reserve_gpu_ns: int
    retry_reserve_wall_ns: int
    profile_reserve_gpu_ns: int
    evidence_reserve_gpu_ns: int
    maximum_compute_gpu_ns: int
    maximum_reserved_gpu_ns: int
    maximum_wall_ns: int
    derivation_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != "lightcone_formal_launch_cap_schedule"
            or self.protocol_sha256 != FORMAL_LAUNCH_CAP_SCHEDULE_PROTOCOL_SHA256
            or self.source_kind
            not in {
                "lifecycle_gpu_hour_source_manifest",
                "preflight_gpu_hour_source_manifest",
                "prospective_gpu_hour_source_manifest",
                "staged_prospective_gpu_hour_source_manifest",
            }
        ):
            raise ValueError("formal launch cap schedule schema is unsupported")
        for label, digest in (
            ("protocol", self.protocol_sha256),
            ("source", self.source_manifest_sha256),
            ("source schedule", self.source_schedule_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware", self.hardware_envelope_sha256),
            ("derivation", self.derivation_sha256),
        ):
            _sha256(f"formal launch schedule {label}", digest)
        keys = tuple(row.materialized_cell_id for row in self.cell_caps)
        if (
            type(self.cell_caps) is not tuple
            or not self.cell_caps
            or any(type(row) is not FormalLaunchCellCap for row in self.cell_caps)
            or keys != tuple(sorted(set(keys)))
        ):
            raise ValueError("formal launch schedule cell caps are not canonical")
        for label, value in (
            ("preconsumed compute", self.preconsumed_compute_gpu_ns),
            ("preconsumed provider", self.preconsumed_provider_reserved_gpu_ns),
            ("preconsumed wall", self.preconsumed_wall_ns),
            ("launchable compute", self.launchable_compute_gpu_ns),
            ("launchable provider", self.launchable_provider_reserved_gpu_ns),
            ("launchable wall", self.launchable_wall_ns),
            ("retry", self.retry_reserve_gpu_ns),
            ("retry wall", self.retry_reserve_wall_ns),
            ("profile", self.profile_reserve_gpu_ns),
            ("evidence", self.evidence_reserve_gpu_ns),
            ("maximum compute", self.maximum_compute_gpu_ns),
            ("maximum reserved", self.maximum_reserved_gpu_ns),
            ("maximum wall", self.maximum_wall_ns),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"formal launch schedule {label} must be non-negative")
        if (
            self.maximum_compute_gpu_ns
            != self.preconsumed_compute_gpu_ns
            + self.launchable_compute_gpu_ns
            + self.retry_reserve_gpu_ns
            or self.maximum_reserved_gpu_ns
            != self.preconsumed_provider_reserved_gpu_ns
            + self.launchable_provider_reserved_gpu_ns
            + self.retry_reserve_gpu_ns
            + self.profile_reserve_gpu_ns
            + self.evidence_reserve_gpu_ns
            or self.maximum_wall_ns
            != self.preconsumed_wall_ns
            + self.launchable_wall_ns
            + self.retry_reserve_wall_ns
            or self.retry_reserve_wall_ns != self.retry_reserve_gpu_ns // 2
        ):
            raise ValueError("formal launch schedule aggregate caps differ")
        for wave in {row.wave_group_sha256 for row in self.cell_caps}:
            rows = tuple(row for row in self.cell_caps if row.wave_group_sha256 == wave)
            if (
                len(rows) not in {1, 2}
                or len({row.provider_wave_hard_timeout_ns_per_attempt for row in rows})
                != 1
                or sum(row.gpu_count for row in rows) > 2
                or sum(row.provider_reserved_gpu_count for row in rows) != 2
                or (len(rows) == 2 and any(row.gpu_count != 1 for row in rows))
                or len({row.disposition for row in rows}) != 1
            ):
                raise ValueError("formal launch schedule provider wave differs")
        expected_derivation = content_sha256(
            {
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "derivation_sha256"
            }
        )
        if self.derivation_sha256 != expected_derivation:
            raise ValueError("formal launch schedule derivation differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    @property
    def launchable_cell_ids(self) -> tuple[str, ...]:
        return tuple(
            row.materialized_cell_id
            for row in self.cell_caps
            if row.disposition == "LAUNCHABLE"
        )

    def cap_for(self, materialized_cell_id: str) -> FormalLaunchCellCap:
        rows = tuple(
            row
            for row in self.cell_caps
            if row.materialized_cell_id == materialized_cell_id
        )
        if len(rows) != 1:
            raise ValueError("formal launch schedule lacks an exact cell cap")
        return rows[0]

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "cell_caps"
        }
        value["cell_caps"] = [row.to_dict() for row in self.cell_caps]
        if include_sha256:
            value["schedule_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_dict(
            "formal launch cap schedule",
            value,
            {*cls.__dataclass_fields__, "schedule_sha256"},
        )
        declared = _sha256(
            "formal launch cap schedule",
            row.pop("schedule_sha256"),
        )
        raw_caps = row.pop("cell_caps")
        if type(raw_caps) is not list:
            raise TypeError("formal launch cap cells must be an array")
        schedule = cls(
            **row,
            cell_caps=tuple(FormalLaunchCellCap.from_dict(item) for item in raw_caps),
        )
        if schedule.sha256 != declared:
            raise ValueError("formal launch cap schedule digest differs")
        return schedule


def _launch_cap_stratum(cell: MaterializedCell) -> str:
    if cell.stage in _STAGED_PROSPECTIVE_STAGES:
        return _staged_projection_stratum(cell)
    if cell.stage in {"E3b", "E5", "E6", "E0"}:
        return _projection_stratum(cell)
    return cell.cell_id


def _wave_group_sha256(
    *,
    source_manifest_sha256: str,
    wave_label: str,
    cell_ids: tuple[str, ...],
) -> str:
    return content_sha256(
        {
            "kind": "lightcone_formal_launch_cap_wave",
            "source_manifest_sha256": source_manifest_sha256,
            "wave_label": wave_label,
            "cell_ids": cell_ids,
        }
    )


def _lifecycle_caps(
    observations: tuple[LifecycleGpuHourObservation, ...],
    *,
    cells: dict[str, MaterializedCell],
    source_manifest_sha256: str,
    disposition_by_cell: dict[str, Literal["PRECONSUMED", "LAUNCHABLE"]],
    remap_cell_id: dict[str, str] | None = None,
    wave_prefix: str = "observed",
) -> tuple[tuple[FormalLaunchCellCap, ...], int, int, int]:
    remap = {} if remap_cell_id is None else remap_cell_id
    caps: list[FormalLaunchCellCap] = []
    wall_ns = 0
    evidence_ns = 0
    for wave in sorted({row.wave_index for row in observations}):
        rows = tuple(row for row in observations if row.wave_index == wave)
        target_ids = tuple(
            sorted(
                remap.get(row.materialized_cell_id, row.materialized_cell_id)
                for row in rows
            )
        )
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("formal launch lifecycle cell remap repeats a target")
        start_ns = min(row.start_ns for row in rows)
        released_ns = max(
            dict(row.phase_edges_ns)["process_group_empty_checked_ns"] for row in rows
        )
        finished_ns = max(row.finish_ns for row in rows)
        wave_wall_ns = released_ns - start_ns
        if wave_wall_ns < 1:
            raise ValueError("formal launch lifecycle wave is not positive")
        wall_ns += wave_wall_ns
        evidence_ns += (finished_ns - released_ns) * 2
        wave_group = _wave_group_sha256(
            source_manifest_sha256=source_manifest_sha256,
            wave_label=f"{wave_prefix}:{wave}",
            cell_ids=target_ids,
        )
        provider_count = 2 if len(rows) == 1 else 1
        for observation in rows:
            target_id = remap.get(
                observation.materialized_cell_id,
                observation.materialized_cell_id,
            )
            cell = cells.get(target_id)
            if cell is None:
                raise ValueError("formal launch lifecycle cap targets a foreign cell")
            caps.append(
                FormalLaunchCellCap(
                    materialized_cell_id=target_id,
                    disposition=disposition_by_cell[target_id],
                    stratum_sha256=_launch_cap_stratum(cell),
                    wave_group_sha256=wave_group,
                    gpu_count=observation.gang_gpu_count,
                    provider_reserved_gpu_count=provider_count,
                    allowed_attempts=FORMAL_LAUNCH_CAP_ALLOWED_ATTEMPTS,
                    process_hard_timeout_ns_per_attempt=(
                        _gpu_process_occupied_ns(observation)
                    ),
                    provider_wave_hard_timeout_ns_per_attempt=wave_wall_ns,
                    maximum_compute_gpu_ns_per_attempt=(
                        _gpu_process_occupied_ns(observation)
                        * observation.gang_gpu_count
                    ),
                    maximum_provider_reserved_gpu_ns_per_attempt=(
                        wave_wall_ns * provider_count
                    ),
                )
            )
    return (
        tuple(sorted(caps, key=lambda row: row.materialized_cell_id)),
        wall_ns,
        evidence_ns,
        sum(row.maximum_compute_gpu_ns_per_attempt for row in caps),
    )


def _preflight_caps(
    source: PreflightGpuHourSourceManifest,
    *,
    cells: dict[str, MaterializedCell],
) -> tuple[tuple[FormalLaunchCellCap, ...], int, int, int]:
    caps: list[FormalLaunchCellCap] = []
    wall_ns = 0
    evidence_ns = 0
    for wave in sorted({row.wave_index for row in source.observations}):
        rows = tuple(row for row in source.observations if row.wave_index == wave)
        cell_ids = tuple(sorted(row.materialized_cell_id for row in rows))
        start_ns = min(row.process_started_ns for row in rows)
        released_ns = max(row.gpu_released_ns for row in rows)
        finished_ns = max(row.evidence_finished_ns for row in rows)
        wave_wall_ns = released_ns - start_ns
        wall_ns += wave_wall_ns
        evidence_ns += (finished_ns - released_ns) * 2
        wave_group = _wave_group_sha256(
            source_manifest_sha256=source.sha256,
            wave_label=f"preflight:{wave}",
            cell_ids=cell_ids,
        )
        provider_count = 2 if len(rows) == 1 else 1
        for observation in rows:
            cell = cells.get(observation.materialized_cell_id)
            if cell is None:
                raise ValueError("formal preflight cap targets a foreign cell")
            caps.append(
                FormalLaunchCellCap(
                    materialized_cell_id=cell.cell_id,
                    disposition="PRECONSUMED",
                    stratum_sha256=_launch_cap_stratum(cell),
                    wave_group_sha256=wave_group,
                    gpu_count=len(observation.gpu_uuids),
                    provider_reserved_gpu_count=provider_count,
                    allowed_attempts=FORMAL_LAUNCH_CAP_ALLOWED_ATTEMPTS,
                    process_hard_timeout_ns_per_attempt=(
                        observation.process_finished_ns - observation.process_started_ns
                    ),
                    provider_wave_hard_timeout_ns_per_attempt=wave_wall_ns,
                    maximum_compute_gpu_ns_per_attempt=(
                        observation.process_finished_ns - observation.process_started_ns
                    )
                    * len(observation.gpu_uuids),
                    maximum_provider_reserved_gpu_ns_per_attempt=(
                        wave_wall_ns * provider_count
                    ),
                )
            )
    return (
        tuple(sorted(caps, key=lambda row: row.materialized_cell_id)),
        wall_ns,
        evidence_ns,
        sum(row.maximum_compute_gpu_ns_per_attempt for row in caps),
    )


def _e5_failure_caps(
    source: E5FailureGpuHourSourceManifest,
    *,
    cells: dict[str, MaterializedCell],
    source_manifest_sha256: str,
) -> tuple[tuple[FormalLaunchCellCap, ...], int, int, int, int]:
    caps: list[FormalLaunchCellCap] = []
    wall_ns = 0
    evidence_ns = 0
    compute_ns = 0
    provider_ns = 0
    for observation in source.observations:
        projection = observation.projection
        cell = cells.get(observation.materialized_cell_id)
        if cell is None or cell.task != "deterministic_failure_injection":
            raise ValueError("E5 failure launch cap targets a foreign cell")
        process_ns = projection.process_exited_ns - projection.execution_started_ns
        provider_wall_ns = projection.gpu_release_ns - projection.execution_started_ns
        wave_group = _wave_group_sha256(
            source_manifest_sha256=source_manifest_sha256,
            wave_label=f"e5-failure-actual:{observation.wave_index}",
            cell_ids=(observation.materialized_cell_id,),
        )
        cap = FormalLaunchCellCap(
            materialized_cell_id=observation.materialized_cell_id,
            disposition="PRECONSUMED",
            stratum_sha256=_launch_cap_stratum(cell),
            wave_group_sha256=wave_group,
            gpu_count=len(projection.gpu_uuids),
            provider_reserved_gpu_count=2,
            allowed_attempts=FORMAL_LAUNCH_CAP_ALLOWED_ATTEMPTS,
            process_hard_timeout_ns_per_attempt=process_ns,
            provider_wave_hard_timeout_ns_per_attempt=provider_wall_ns,
            maximum_compute_gpu_ns_per_attempt=projection.compute_gpu_ns,
            maximum_provider_reserved_gpu_ns_per_attempt=(
                projection.provider_reserved_gpu_ns
            ),
        )
        caps.append(cap)
        wall_ns += provider_wall_ns
        evidence_ns += projection.evidence_gpu_ns
        compute_ns += projection.compute_gpu_ns
        provider_ns += projection.provider_reserved_gpu_ns
    if (
        compute_ns != source.cost.compute_gpu_ns
        or provider_ns != source.cost.provider_base_reserved_gpu_ns
        or wall_ns != source.cost.wall_ns
        or evidence_ns != source.cost.evidence_reserve_gpu_ns
    ):
        raise ValueError("E5 failure launch cap cost differs from source")
    return (
        tuple(sorted(caps, key=lambda row: row.materialized_cell_id)),
        wall_ns,
        evidence_ns,
        compute_ns,
        provider_ns,
    )


def _isolated_projected_cap(
    *,
    cell: MaterializedCell,
    source_manifest_sha256: str,
    stratum_sha256: str,
    execution_ns: int,
    wall_ns: int,
    gang_count: int,
    disposition: Literal["PRECONSUMED", "LAUNCHABLE"],
    label: str,
) -> FormalLaunchCellCap:
    wave_group = _wave_group_sha256(
        source_manifest_sha256=source_manifest_sha256,
        wave_label=label,
        cell_ids=(cell.cell_id,),
    )
    return FormalLaunchCellCap(
        materialized_cell_id=cell.cell_id,
        disposition=disposition,
        stratum_sha256=stratum_sha256,
        wave_group_sha256=wave_group,
        gpu_count=gang_count,
        provider_reserved_gpu_count=2,
        allowed_attempts=FORMAL_LAUNCH_CAP_ALLOWED_ATTEMPTS,
        process_hard_timeout_ns_per_attempt=execution_ns,
        provider_wave_hard_timeout_ns_per_attempt=wall_ns,
        maximum_compute_gpu_ns_per_attempt=execution_ns * gang_count,
        maximum_provider_reserved_gpu_ns_per_attempt=wall_ns * 2,
    )


def derive_and_validate_formal_launch_cap_schedule(
    source: (
        LifecycleGpuHourSourceManifest
        | PreflightGpuHourSourceManifest
        | ProspectiveGpuHourSourceManifest
        | StagedProspectiveGpuHourSourceManifest
    ),
    materialization: StageMaterializationReceipt,
    *,
    pilot_materialization: StageMaterializationReceipt | None = None,
) -> FormalLaunchCapSchedule:
    """Rebuild per-cell/wave launch caps without accepting caller durations."""

    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal launch caps require an exact materialization")
    _reject_untyped_special_gpu_hour_cells(
        materialization,
        allow_e5_dedicated_failure_source=(
            type(source) is ProspectiveGpuHourSourceManifest
            and source.stage == "E5"
            and source.one_shot_source_manifest is not None
        ),
    )
    cells = {cell.cell_id: cell for cell in materialization.cells}
    if len(cells) != len(materialization.cells):  # pragma: no cover - receipt gate
        raise AssertionError("formal launch cap materialization repeats cells")
    if (
        source.protocol_lock_sha256 != materialization.protocol_lock_sha256
        or source.inventory_sha256 == ""
    ):
        raise ValueError("formal launch cap immutable lineage differs")
    caps: tuple[FormalLaunchCellCap, ...]
    pre_compute = pre_provider = pre_wall = 0
    launch_compute = launch_provider = launch_wall = 0
    retry_ns = profile_ns = evidence_ns = 0
    hardware_sha256 = source.hardware_envelope_sha256

    if type(source) is LifecycleGpuHourSourceManifest:
        if pilot_materialization is not None or {
            row.materialized_cell_id for row in source.observations
        } != set(cells):
            raise ValueError("formal lifecycle launch caps require exact cell coverage")
        caps, pre_wall, evidence_ns, pre_compute = _lifecycle_caps(
            source.observations,
            cells=cells,
            source_manifest_sha256=source.sha256,
            disposition_by_cell={cell_id: "PRECONSUMED" for cell_id in cells},
        )
        pre_provider = pre_wall * 2
        retry_ns = 0
    elif type(source) is PreflightGpuHourSourceManifest:
        # Schema-1 preflight manifests contain only the scientific 1+1+8
        # rows.  They omit the resolved native/distributed qualification
        # process set and therefore cannot bound the physical preflight wave.
        # The source-owned qualification scheduler will replace this branch
        # with a single admission-bound cap union.
        raise FormalGpuHourLifecycleBlocked(
            "preflight_qualification_lifecycle_cost_schedule_missing"
        )
    elif type(source) is StagedProspectiveGpuHourSourceManifest:
        if (
            source.status != "READY"
            or source.completed_source_manifest is None
            or source.total is None
            or pilot_materialization is not None
        ):
            raise FormalGpuHourLifecycleBlocked("scientific_strata_unmeasured")
        binding = source.completed_source_manifest
        if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
            raise ValueError("formal launch staged source binding changed")
        completed = LifecycleGpuHourSourceManifest.from_dict(binding.reopen())
        completed_ids = {row.materialized_cell_id for row in completed.observations}
        if (
            completed.materialization_receipt_sha256 != materialization.sha256
            or completed.protocol_lock_sha256 != source.protocol_lock_sha256
            or completed.inventory_sha256 != source.inventory_sha256
            or completed.sha256 != binding.reopen().get("manifest_sha256")
            or completed_ids - set(cells)
        ):
            raise ValueError("formal launch staged completed source differs")
        observed_caps, pre_wall, actual_evidence, pre_compute = _lifecycle_caps(
            completed.observations,
            cells=cells,
            source_manifest_sha256=source.sha256,
            disposition_by_cell={cell_id: "PRECONSUMED" for cell_id in completed_ids},
            wave_prefix="staged-completed",
        )
        projected_caps: list[FormalLaunchCellCap] = []
        by_id = {row.materialized_cell_id: row for row in completed.observations}
        projected_evidence = 0
        for stratum in source.strata:
            pilots = tuple(by_id[cell_id] for cell_id in stratum.completed_cell_ids)
            if stratum.status != "MEASURED" or not pilots:
                raise FormalGpuHourLifecycleBlocked("scientific_strata_unmeasured")
            gang_counts = {row.gang_gpu_count for row in pilots}
            if len(gang_counts) != 1:
                raise ValueError("formal launch staged stratum mixes gang sizes")
            execution_ns = _ceiling_mean(
                tuple(_gpu_process_occupied_ns(row) for row in pilots)
            )
            wall_bound_ns = _ceiling_mean(
                tuple(
                    dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                    - row.start_ns
                    for row in pilots
                )
            )
            evidence_tail_ns = _ceiling_mean(
                tuple(
                    row.finish_ns
                    - dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                    for row in pilots
                )
            )
            for cell_id in stratum.projected_cell_ids:
                projected_caps.append(
                    _isolated_projected_cap(
                        cell=cells[cell_id],
                        source_manifest_sha256=source.sha256,
                        stratum_sha256=stratum.stratum_sha256,
                        execution_ns=execution_ns,
                        wall_ns=wall_bound_ns,
                        gang_count=next(iter(gang_counts)),
                        disposition="LAUNCHABLE",
                        label=f"staged-projected:{cell_id}",
                    )
                )
                projected_evidence += evidence_tail_ns * 2
        caps = tuple(
            sorted(
                (*observed_caps, *projected_caps),
                key=lambda row: row.materialized_cell_id,
            )
        )
        pre_provider = pre_wall * 2
        launch_compute = sum(
            row.maximum_compute_gpu_ns_per_attempt for row in projected_caps
        )
        launch_provider = sum(
            row.maximum_provider_reserved_gpu_ns_per_attempt for row in projected_caps
        )
        launch_wall = sum(
            row.provider_wave_hard_timeout_ns_per_attempt for row in projected_caps
        )
        if source.projected_remaining is None:  # pragma: no cover - READY gate
            raise AssertionError("READY staged source lost projected costs")
        retry_ns = source.projected_remaining.retry_reserve_gpu_ns
        profile_ns = source.total.profile_reserve_gpu_ns
        evidence_ns = actual_evidence + projected_evidence
        if (
            pre_compute + launch_compute != source.total.compute_gpu_ns
            or pre_provider + launch_provider
            != source.total.provider_base_reserved_gpu_ns
            or pre_wall + launch_wall != source.total.wall_ns
            or evidence_ns != source.total.evidence_reserve_gpu_ns
        ):
            raise ValueError("formal launch staged cap totals differ from source")
    elif type(source) is ProspectiveGpuHourSourceManifest:
        if source.stage == "E5":
            if source.one_shot_source_manifest is None:
                raise FormalGpuHourLifecycleBlocked(
                    "e5_dedicated_failure_launch_cap_required"
                )
            raw_failure_source = source.one_shot_source_manifest.reopen()
            if raw_failure_source.get("kind") != (
                "e5_failure_gpu_hour_source_manifest"
            ):
                raise FormalGpuHourLifecycleBlocked(
                    "e5_dedicated_failure_launch_cap_required"
                )
        if (
            pilot_materialization is None
            or pilot_materialization.sha256
            != source.pilot_materialization_receipt_sha256
            or source.stage != materialization.stage
            or source.final_materialization_receipt_sha256 != materialization.sha256
        ):
            raise ValueError("formal prospective launch caps require exact pilots")
        binding = source.pilot_source_manifest
        if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
            raise ValueError("formal launch prospective pilot binding changed")
        pilot_source = LifecycleGpuHourSourceManifest.from_dict(binding.reopen())
        if (
            pilot_source.materialization_receipt_sha256 != pilot_materialization.sha256
            or pilot_source.protocol_lock_sha256 != source.protocol_lock_sha256
            or pilot_source.inventory_sha256 != source.inventory_sha256
            or {row.materialized_cell_id for row in pilot_source.observations}
            != {cell.cell_id for cell in pilot_materialization.cells}
        ):
            raise ValueError("formal launch prospective pilot source differs")
        pilot_by_id = {
            row.materialized_cell_id: row for row in pilot_source.observations
        }
        pilot_by_stratum: dict[str, list[LifecycleGpuHourObservation]] = {}
        for pilot_cell in pilot_materialization.cells:
            if type(dict(pilot_cell.dimensions).get("block")) is int:
                pilot_by_stratum.setdefault(_projection_stratum(pilot_cell), []).append(
                    pilot_by_id[pilot_cell.cell_id]
                )
        projected_caps = []
        projected_evidence = 0
        nonblock_cells = []
        for cell in materialization.cells:
            if type(dict(cell.dimensions).get("block")) is not int:
                nonblock_cells.append(cell)
                continue
            stratum = _projection_stratum(cell)
            pilots = tuple(pilot_by_stratum.get(stratum, ()))
            if len(pilots) != 4:
                raise ValueError("formal launch prospective stratum lacks four pilots")
            gang_counts = {row.gang_gpu_count for row in pilots}
            if len(gang_counts) != 1:
                raise ValueError("formal launch prospective stratum mixes gang sizes")
            execution_ns = _ceiling_mean(
                tuple(_gpu_process_occupied_ns(row) for row in pilots)
            )
            wall_bound_ns = _ceiling_mean(
                tuple(
                    dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                    - row.start_ns
                    for row in pilots
                )
            )
            evidence_tail_ns = _ceiling_mean(
                tuple(
                    row.finish_ns
                    - dict(row.phase_edges_ns)["process_group_empty_checked_ns"]
                    for row in pilots
                )
            )
            projected_caps.append(
                _isolated_projected_cap(
                    cell=cell,
                    source_manifest_sha256=source.sha256,
                    stratum_sha256=stratum,
                    execution_ns=execution_ns,
                    wall_ns=wall_bound_ns,
                    gang_count=next(iter(gang_counts)),
                    disposition="LAUNCHABLE",
                    label=f"prospective-final:{cell.cell_id}",
                )
            )
            projected_evidence += evidence_tail_ns * 2
        preconsumed_caps: tuple[FormalLaunchCellCap, ...] = ()
        if nonblock_cells:
            if source.stage == "E5":
                nonblock_ids = tuple(sorted(cell.cell_id for cell in nonblock_cells))
                if len(nonblock_ids) != 264 or source.one_shot_source_manifest is None:
                    raise FormalGpuHourLifecycleBlocked(
                        "e5_dedicated_failure_launch_cap_required"
                    )
                failure_binding = source.one_shot_source_manifest
                if (
                    CanonicalJsonProofBinding.bind(failure_binding.absolute_path)
                    != failure_binding
                ):
                    raise ValueError("E5 failure launch source binding changed")
                failure_source = E5FailureGpuHourSourceManifest.from_dict(
                    failure_binding.reopen()
                )
                if (
                    failure_source.protocol_lock_sha256 != source.protocol_lock_sha256
                    or failure_source.materialization_receipt_sha256
                    != materialization.sha256
                    or failure_source.inventory_sha256 != source.inventory_sha256
                    or failure_source.hardware_envelope_sha256
                    != source.hardware_envelope_sha256
                    or tuple(
                        sorted(
                            row.materialized_cell_id
                            for row in failure_source.observations
                        )
                    )
                    != nonblock_ids
                    or len(source.costs) != 3
                    or source.costs[2] != failure_source.cost
                ):
                    raise ValueError("E5 failure launch source lineage differs")
                (
                    preconsumed_caps,
                    one_shot_wall,
                    one_shot_evidence,
                    one_shot_compute,
                    one_shot_provider,
                ) = _e5_failure_caps(
                    failure_source,
                    cells=cells,
                    source_manifest_sha256=source.sha256,
                )
                if (
                    one_shot_compute != source.costs[2].compute_gpu_ns
                    or one_shot_provider
                    != source.costs[2].provider_base_reserved_gpu_ns
                    or one_shot_wall != source.costs[2].wall_ns
                    or one_shot_evidence != source.costs[2].evidence_reserve_gpu_ns
                ):
                    raise ValueError("E5 failure launch cap arithmetic differs")
            elif source.stage != "E6" or len(nonblock_cells) != 2:
                raise FormalGpuHourLifecycleBlocked(
                    "prospective_nonblock_launch_cap_requires_typed_actual_join"
                )
            else:
                nonblock_ids = tuple(sorted(cell.cell_id for cell in nonblock_cells))
                if any(cell_id not in pilot_by_id for cell_id in nonblock_ids):
                    raise ValueError(
                        "E6 actual preflight launch cap lacks pilot evidence"
                    )
                one_shot_observations = tuple(
                    pilot_by_id[cell_id] for cell_id in nonblock_ids
                )
                (
                    preconsumed_caps,
                    one_shot_wall,
                    one_shot_evidence,
                    one_shot_compute,
                ) = _lifecycle_caps(
                    one_shot_observations,
                    cells=cells,
                    source_manifest_sha256=source.sha256,
                    disposition_by_cell={
                        cell_id: "PRECONSUMED" for cell_id in nonblock_ids
                    },
                    wave_prefix="prospective-actual-preflight",
                )
                if len(source.costs) != 3:
                    raise ValueError("E6 prospective costs lack actual preflights")
                one_shot_cost = _cost_from_actual_observations(
                    category="actual_one_shot",
                    observations=one_shot_observations,
                    inventory_gpu_count=2,
                )
                if (
                    one_shot_cost != source.costs[2]
                    or one_shot_compute != one_shot_cost.compute_gpu_ns
                    or one_shot_wall != one_shot_cost.wall_ns
                    or one_shot_evidence != one_shot_cost.evidence_reserve_gpu_ns
                ):
                    raise ValueError("E6 actual preflight launch cap cost differs")
        elif source.stage == "E6":
            raise ValueError("E6 launch cap lacks its two actual preflights")
        caps = tuple(
            sorted(
                (*preconsumed_caps, *projected_caps),
                key=lambda row: row.materialized_cell_id,
            )
        )
        launch_compute = sum(
            row.maximum_compute_gpu_ns_per_attempt for row in projected_caps
        )
        launch_provider = sum(
            row.maximum_provider_reserved_gpu_ns_per_attempt for row in projected_caps
        )
        launch_wall = sum(
            row.provider_wave_hard_timeout_ns_per_attempt for row in projected_caps
        )
        actual = source.costs[0]
        projected = source.costs[1]
        actual_costs = (
            (actual, source.costs[2]) if source.stage in {"E5", "E6"} else (actual,)
        )
        pre_compute = sum(row.compute_gpu_ns for row in actual_costs)
        pre_provider = sum(row.provider_base_reserved_gpu_ns for row in actual_costs)
        pre_wall = sum(row.wall_ns for row in actual_costs)
        retry_ns = projected.retry_reserve_gpu_ns
        profile_ns = sum(row.profile_reserve_gpu_ns for row in source.costs)
        evidence_ns = sum(row.evidence_reserve_gpu_ns for row in source.costs)
        if (
            launch_compute != projected.compute_gpu_ns
            or launch_provider != projected.provider_base_reserved_gpu_ns
            or launch_wall != projected.wall_ns
            or projected_evidence != projected.evidence_reserve_gpu_ns
        ):
            raise ValueError("formal launch prospective cap totals differ from source")
    else:  # pragma: no cover - exact type union above
        raise TypeError("formal launch cap source kind is unsupported")

    if tuple(row.materialized_cell_id for row in caps) != tuple(sorted(cells)):
        raise ValueError("formal launch cap schedule lacks exact materialization cells")
    maximum_compute = pre_compute + launch_compute + retry_ns
    maximum_reserved = (
        pre_provider + launch_provider + retry_ns + profile_ns + evidence_ns
    )
    fields = {
        "schema_version": 2,
        "kind": "lightcone_formal_launch_cap_schedule",
        "protocol_sha256": FORMAL_LAUNCH_CAP_SCHEDULE_PROTOCOL_SHA256,
        "source_kind": source.kind,
        "source_manifest_sha256": source.sha256,
        "source_schedule_sha256": (
            source.mapping_sha256
            if type(source)
            in {
                ProspectiveGpuHourSourceManifest,
                StagedProspectiveGpuHourSourceManifest,
            }
            else source.schedule_sha256
        ),
        "protocol_lock_sha256": source.protocol_lock_sha256,
        "materialization_receipt_sha256": materialization.sha256,
        "inventory_sha256": source.inventory_sha256,
        "hardware_envelope_sha256": hardware_sha256,
        "cell_caps": caps,
        "preconsumed_compute_gpu_ns": pre_compute,
        "preconsumed_provider_reserved_gpu_ns": pre_provider,
        "preconsumed_wall_ns": pre_wall,
        "launchable_compute_gpu_ns": launch_compute,
        "launchable_provider_reserved_gpu_ns": launch_provider,
        "launchable_wall_ns": launch_wall,
        "retry_reserve_gpu_ns": retry_ns,
        "retry_reserve_wall_ns": retry_ns // 2,
        "profile_reserve_gpu_ns": profile_ns,
        "evidence_reserve_gpu_ns": evidence_ns,
        "maximum_compute_gpu_ns": maximum_compute,
        "maximum_reserved_gpu_ns": maximum_reserved,
        "maximum_wall_ns": pre_wall + launch_wall + retry_ns // 2,
    }
    schedule = FormalLaunchCapSchedule(
        **fields,
        derivation_sha256=content_sha256(fields),
    )
    if FormalLaunchCapSchedule.from_dict(schedule.to_dict()) != schedule:
        raise RuntimeError("formal launch cap schedule codec changed")
    return schedule


__all__ = [
    "E5_FAILURE_GPU_HOUR_SOURCE_PROTOCOL_SHA256",
    "FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256",
    "FORMAL_GPU_HOUR_BUDGET_RUNNER_SHA256",
    "FORMAL_GPU_HOUR_BUDGET_TEST_SET_SHA256",
    "FORMAL_GPU_HOUR_RETRY_RESERVE_BPS",
    "FORMAL_LAUNCH_CAP_ALLOWED_ATTEMPTS",
    "FORMAL_LAUNCH_CAP_SCHEDULE_PROTOCOL_SHA256",
    "FORMAL_SERVING_EXECUTION_PROOF_PROTOCOL_SHA256",
    "PREFLIGHT_GPU_HOUR_PROTOCOL_SHA256",
    "PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256",
    "STAGED_PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256",
    "E5FailureGpuHourObservation",
    "E5FailureGpuHourProofInput",
    "E5FailureGpuHourSourceManifest",
    "FormalGpuHourLifecycleBlocked",
    "FormalLaunchCapSchedule",
    "FormalLaunchCellCap",
    "FormalServingExecutionProofArtifact",
    "FormalServingExecutionProofPayload",
    "LifecycleGpuHourObservation",
    "LifecycleGpuHourProofInput",
    "LifecycleGpuHourSourceManifest",
    "PreflightGpuHourLifecycleProofInput",
    "PreflightGpuHourObservation",
    "PreflightGpuHourSourceManifest",
    "ProspectiveGpuHourCost",
    "ProspectiveGpuHourSourceManifest",
    "StagedGpuHourStratum",
    "StagedProspectiveGpuHourCost",
    "StagedProspectiveGpuHourSourceManifest",
    "VerifiedProspectiveGpuHourAuthority",
    "derive_and_validate_formal_launch_cap_schedule",
    "materialize_e5_failure_gpu_hour_source_manifest",
    "materialize_lifecycle_gpu_hour_subset_source",
    "materialize_preflight_gpu_hour_envelope",
    "materialize_prospective_stage_gpu_hour_envelope",
    "materialize_stage_gpu_hour_envelope_from_lifecycle_proofs",
    "materialize_staged_prospective_gpu_hour_envelope",
    "publish_formal_serving_execution_proof_artifact",
    "revalidate_persisted_e5_failure_gpu_hour_source_manifest",
    "revalidate_persisted_preflight_gpu_hour_source_manifest",
    "revalidate_persisted_prospective_gpu_hour_source_manifest",
    "revalidate_persisted_stage_gpu_hour_source_manifest",
    "revalidate_persisted_staged_prospective_gpu_hour_source_manifest",
    "revalidate_stage_gpu_hour_source_manifest",
    "validate_formal_serving_execution_proof_artifact",
    "verify_e0_prospective_gpu_hour_authority",
    "verify_e3b_prospective_gpu_hour_authority",
    "verify_e5_prospective_gpu_hour_authority",
    "verify_e6_prospective_gpu_hour_authority",
    "verify_registered_prospective_gpu_hour_authority",
]
