"""Deterministic staged materialization for the formal experiment DAG.

Only concrete cells are emitted.  Downstream waves are derived from sealed
upstream decisions and signed as receipts; this module never expands blocked
sentinels to stand in for a future matrix.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

from lightcone_spec.experiments.formal_protocol import (
    E0_METHOD_ROLES,
    E6_MODELS,
    FORMAL_METHOD_ROLES,
    FORMAL_STAGE_DAG,
    TTS_L0_CANDIDATE_STATE_COVERAGE_STAGES,
    ProtocolLock,
    SignedTtsCalibrationSeal,
    TtsCalibrationAuthority,
    TtsL0CandidateStateCoverage,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.statistics import PILOT_BLOCK_COUNT

if TYPE_CHECKING:
    from lightcone_spec.experiments.selection_authority import (
        E3aSelectionReductionAuthority,
    )
    from lightcone_spec.experiments.stage_decisions import (
        SignedE3aSelectionReceipt,
    )
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)

E1_SCOPES = ("last1", "last3", "last5", "all")
LORA_RANKS = (1, 2, 4, 8, 16, 32, 64)
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
E3B_CONTEXTS = (1024, 2048, 4096, 8192, 16384, 24576, 32768, 40928)
E3B_REGIMES = (
    "long_input_short_output",
    "short_input_long_generation",
    "multi_turn_shared_prefix",
)
E3B_LOADS = ("concurrency_one", "common_load")
E3B_WIDTH_PANELS = ("matched", "deployment_optimal")
E1A_VERIFICATION_MODES = ("fixed_verification_budget", "native_scheduler")
E1A_FIXED_VERIFICATION_BUDGET = 8
E1A_NATIVE_VERIFICATION_BUDGET = "not_applicable"
E6_TASKS = ("LiveCodeBench", "MATH-500")
E6_CONTEXTS = (4096, 16384, 32768)
E0_MODELS = ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Gemma4-12B")
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
E0_LOADS = ("concurrency_one", "common_slo_load")
E0_ALL_NA_MATERIALIZATION_RULE = "all_proof_backed_combinations_are_na"
E0_ALL_NA_MATERIALIZATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e0_all_na_zero_cell_materialization_protocol",
        "compatibility_universe": "exact_signed_108_decisions",
        "required_disposition": "all_N/A",
        "upstream": "proof_derived_confirmed_E6",
        "materialized_cells": 0,
        "tuning_pilots_power": "not_applicable_and_not_fabricated",
    }
)
E4_LOADS = ("low", "moderate", "saturation")
E4_TRAFFIC = ("pure_decode", "mixed_prefill_decode")
E4_SCREEN_FACTOR_LEVELS = (
    ("update_stride", (1, 50)),
    ("microbatch", (1, 8)),
    ("coalescing", (1, 8)),
    ("stream_priority", ("default", "high")),
)
E5_BACKENDS = ("DFLASH", "DSPARK")
E5_TOPOLOGIES = ("tp1_dp1", "tp2_dp1", "tp1_dp2")
E5_LOADS = ("concurrency_one", "common_slo_load")
E5_CLOSED_LOOP_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64, 128, 256)
E5_OPEN_LOOP_LOAD_FACTORS = (0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.25)
E5_TRACE_AND_SOAK_ARRIVALS = (
    "immediate_burst",
    "burstgpt_shape",
    "moderate_soak",
    "saturation_soak",
    "overload_soak",
)
E5_COHORT_COUNTS = (1, 4, 16, 64)
E5_COHORT_DISTRIBUTIONS = ("uniform", "zipf")
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
E5_FAILURE_DIAGNOSTIC_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e5_failure_diagnostic_matrix",
        "failures": E5_FAILURES,
        "backends": E5_BACKENDS,
        "topologies": E5_TOPOLOGIES,
        "cohort_counts": E5_COHORT_COUNTS,
        "execution": "one_correctness_only_diagnostic_per_matrix_member",
        "block_multiplier": False,
        "p99_anchor_materialization": "headline_requirement_not_extra_cell",
    }
)

_SHA256_LENGTH = 64
_UNRESOLVED_FRAGMENTS = ("template", "sentinel", "blocked", "to_be_selected")


class FormalStageMaterializationBlocked(RuntimeError):
    """A typed upstream authority required by a formal stage is unavailable."""


class FormalGpuHourAuthorityBlocked(RuntimeError):
    """Formal GPU-hour authority lacks a complete lifecycle timing proof."""


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lower-case SHA-256")
    return value


def _require_text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be exact non-empty single-line text")
    return value


@dataclass(frozen=True)
class GpuHourEstimate:
    """Pilot-derived resource output attached to every materialization."""

    status: Literal["UNMEASURED", "AVAILABLE"]
    source_pilot_receipt_sha256: str | None
    compute_gpu_hours: float | None
    reserved_gpu_hours: float | None
    estimated_wall_hours: float | None
    retry_reserve_gpu_hours: float | None
    profile_reserve_gpu_hours: float | None
    evidence_reserve_gpu_hours: float | None
    source_schedule_sha256: str | None = None
    source_materialization_receipt_sha256: str | None = None
    source_inventory_gpu_count: int | None = None
    derivation_sha256: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.compute_gpu_hours,
            self.reserved_gpu_hours,
            self.estimated_wall_hours,
            self.retry_reserve_gpu_hours,
            self.profile_reserve_gpu_hours,
            self.evidence_reserve_gpu_hours,
        )
        if self.status == "UNMEASURED":
            if any(
                value is not None
                for value in (
                    self.source_pilot_receipt_sha256,
                    *values,
                    self.source_schedule_sha256,
                    self.source_materialization_receipt_sha256,
                    self.source_inventory_gpu_count,
                    self.derivation_sha256,
                )
            ):
                raise ValueError("UNMEASURED GPU hours cannot contain estimates")
            return
        if self.status != "AVAILABLE":
            raise ValueError("GPU-hour estimate status must be UNMEASURED or AVAILABLE")
        _require_sha256("GPU-hour pilot receipt", self.source_pilot_receipt_sha256)
        _require_sha256("GPU-hour schedule", self.source_schedule_sha256)
        _require_sha256(
            "GPU-hour materialization",
            self.source_materialization_receipt_sha256,
        )
        _require_sha256("GPU-hour derivation", self.derivation_sha256)
        if self.source_inventory_gpu_count != 2:
            raise ValueError("formal GPU-hour estimate requires a two-GPU inventory")
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0
            for value in values
        ):
            raise ValueError("GPU-hour estimates must be finite non-negative floats")
        assert self.compute_gpu_hours is not None
        assert self.reserved_gpu_hours is not None
        assert self.estimated_wall_hours is not None
        assert self.retry_reserve_gpu_hours is not None
        assert self.profile_reserve_gpu_hours is not None
        assert self.evidence_reserve_gpu_hours is not None
        if self.compute_gpu_hours <= 0 or self.estimated_wall_hours <= 0:
            raise ValueError("available compute and wall hours must be positive")
        minimum_reserved = (
            self.estimated_wall_hours * self.source_inventory_gpu_count
            + self.retry_reserve_gpu_hours
            + self.profile_reserve_gpu_hours
            + self.evidence_reserve_gpu_hours
        )
        if self.reserved_gpu_hours < minimum_reserved:
            raise ValueError("reserved GPU hours do not cover all registered reserves")
        expected_derivation = content_sha256(
            {
                "source_pilot_receipt_sha256": self.source_pilot_receipt_sha256,
                "source_schedule_sha256": self.source_schedule_sha256,
                "source_materialization_receipt_sha256": (
                    self.source_materialization_receipt_sha256
                ),
                "source_inventory_gpu_count": self.source_inventory_gpu_count,
                "compute_gpu_hours": self.compute_gpu_hours,
                "reserved_gpu_hours": self.reserved_gpu_hours,
                "estimated_wall_hours": self.estimated_wall_hours,
                "retry_reserve_gpu_hours": self.retry_reserve_gpu_hours,
                "profile_reserve_gpu_hours": self.profile_reserve_gpu_hours,
                "evidence_reserve_gpu_hours": self.evidence_reserve_gpu_hours,
            }
        )
        if self.derivation_sha256 != expected_derivation:
            raise ValueError(
                "GPU-hour estimate differs from its deterministic derivation"
            )

    @classmethod
    def unmeasured(cls) -> GpuHourEstimate:
        return cls("UNMEASURED", None, None, None, None, None, None, None)


@dataclass(frozen=True)
class PilotDurationObservation:
    cell_id: str
    wave_index: int
    gang_gpu_count: int
    terminal_receipt_sha256: str
    schedule_assignment_sha256: str
    startup_ms: int
    warmup_ms: int
    arrival_window_ms: int
    drain_ms: int
    reset_ms: int
    evidence_flush_ms: int
    retry_ms: int
    profile_ms: int
    wall_time_ms: int

    def __post_init__(self) -> None:
        _require_sha256("pilot duration cell", self.cell_id)
        _require_sha256("pilot duration terminal receipt", self.terminal_receipt_sha256)
        _require_sha256(
            "pilot duration schedule assignment", self.schedule_assignment_sha256
        )
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise ValueError("pilot wave index must be a non-negative integer")
        if type(self.gang_gpu_count) is not int or self.gang_gpu_count not in {1, 2}:
            raise ValueError("pilot gang count must be one or two GPUs")
        if type(self.wall_time_ms) is not int or self.wall_time_ms <= 0:
            raise ValueError("pilot wall duration must be a positive integer")
        phase_fields = (
            "startup_ms",
            "warmup_ms",
            "arrival_window_ms",
            "drain_ms",
            "reset_ms",
            "evidence_flush_ms",
            "retry_ms",
            "profile_ms",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in phase_fields
        ):
            raise ValueError("pilot phase durations must be non-negative integers")
        if self.arrival_window_ms <= 0:
            raise ValueError("pilot arrival window must be positive")
        if self.wall_time_ms != sum(getattr(self, name) for name in phase_fields):
            raise ValueError("pilot wall duration differs from complete phase coverage")


@dataclass(frozen=True)
class PilotDurationReceipt:
    schema_version: int
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    schedule_sha256: str
    inventory_gpu_count: int
    observations: tuple[PilotDurationObservation, ...]
    retry_reserve_fraction: float
    profile_reserve_gpu_hours: float
    evidence_reserve_gpu_hours: float

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("only pilot-duration receipt schema 2 is supported")
        for name, digest in (
            ("pilot protocol lock", self.protocol_lock_sha256),
            ("pilot materialization", self.materialization_receipt_sha256),
            ("pilot schedule", self.schedule_sha256),
        ):
            _require_sha256(name, digest)
        if type(self.inventory_gpu_count) is not int or self.inventory_gpu_count != 2:
            raise ValueError("formal pilot duration authority requires two GPUs")
        keys = tuple((row.wave_index, row.cell_id) for row in self.observations)
        if not keys or keys != tuple(sorted(set(keys))):
            raise ValueError(
                "pilot duration observations must be non-empty and canonical"
            )
        terminal_receipts = tuple(
            row.terminal_receipt_sha256 for row in self.observations
        )
        if len(terminal_receipts) != len(set(terminal_receipts)):
            raise ValueError("pilot observations reuse a first-party terminal receipt")
        for wave in {row.wave_index for row in self.observations}:
            if (
                sum(
                    row.gang_gpu_count
                    for row in self.observations
                    if row.wave_index == wave
                )
                > self.inventory_gpu_count
            ):
                raise ValueError("pilot wave gang counts exceed the signed inventory")
        if (
            type(self.retry_reserve_fraction) is not float
            or not math.isfinite(self.retry_reserve_fraction)
            or not 0 <= self.retry_reserve_fraction <= 1
        ):
            raise ValueError("pilot retry reserve fraction must be in [0, 1]")
        for name in (
            "profile_reserve_gpu_hours",
            "evidence_reserve_gpu_hours",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError(f"pilot {name} must be finite and non-negative")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedPilotDurationReceipt:
    payload: PilotDurationReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> PilotDurationReceipt:
        if type(self.payload) is not PilotDurationReceipt:
            raise TypeError("signed pilot-duration payload has the wrong type")
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
class StageGpuHourEnvelope:
    """Independent pilot-derived budget for one immutable materialization.

    Schema 1 is the historical signer-authored scalar-duration envelope and is
    retained only for diagnostic decoding.  Schema 2 is reserved for the
    path-bound lifecycle reducer in :mod:`lightcone_spec.experiments.gpu_hour_authority`.
    Callers cannot promote a schema-1 envelope by changing this integer: the
    formal reducer and registry deep-reopen the schema-2 source manifest and
    every first-party lifecycle proof.
    """

    schema_version: int
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    signed_pilot_receipt_sha256: str
    schedule_sha256: str
    estimate: GpuHourEstimate

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {1, 2}:
            raise ValueError("stage GPU-hour envelope schema is unsupported")
        for name, digest in (
            ("GPU-hour protocol lock", self.protocol_lock_sha256),
            ("GPU-hour materialization", self.materialization_receipt_sha256),
            ("GPU-hour signed pilot", self.signed_pilot_receipt_sha256),
            ("GPU-hour schedule", self.schedule_sha256),
        ):
            _require_sha256(name, digest)
        if type(self.estimate) is not GpuHourEstimate or self.estimate.status != (
            "AVAILABLE"
        ):
            raise ValueError("stage GPU-hour envelope requires an AVAILABLE estimate")
        if (
            self.estimate.source_pilot_receipt_sha256
            != self.signed_pilot_receipt_sha256
            or self.estimate.source_schedule_sha256 != self.schedule_sha256
            or self.estimate.source_materialization_receipt_sha256
            != self.materialization_receipt_sha256
        ):
            raise ValueError("stage GPU-hour estimate differs from envelope lineage")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedStageGpuHourEnvelope:
    payload: StageGpuHourEnvelope
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> StageGpuHourEnvelope:
        if type(self.payload) is not StageGpuHourEnvelope:
            raise TypeError("signed stage GPU-hour payload has the wrong type")
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


def _reduce_gpu_hours_from_signed_pilots_diagnostic(
    signed: SignedPilotDurationReceipt,
    *,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    protocol_lock_sha256: str,
    materialization_receipt_sha256: str,
    schedule_sha256: str,
    now_ns: int | None = None,
) -> GpuHourEstimate:
    """Legacy arithmetic check over signer-authored durations; non-authorizing."""

    receipt = signed.verify(
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    if (
        receipt.protocol_lock_sha256 != protocol_lock_sha256
        or receipt.materialization_receipt_sha256 != materialization_receipt_sha256
        or receipt.schedule_sha256 != schedule_sha256
    ):
        raise ValueError("pilot duration receipt differs from requested stage lineage")
    compute = (
        sum(row.wall_time_ms * row.gang_gpu_count for row in receipt.observations)
        / 3_600_000
    )
    wall = (
        sum(
            max(
                row.wall_time_ms
                for row in receipt.observations
                if row.wave_index == wave
            )
            for wave in sorted({row.wave_index for row in receipt.observations})
        )
        / 3_600_000
    )
    retry = compute * receipt.retry_reserve_fraction
    reserved = (
        wall * receipt.inventory_gpu_count
        + retry
        + receipt.profile_reserve_gpu_hours
        + receipt.evidence_reserve_gpu_hours
    )
    derivation = {
        "source_pilot_receipt_sha256": signed.sha256,
        "source_schedule_sha256": receipt.schedule_sha256,
        "source_materialization_receipt_sha256": (
            receipt.materialization_receipt_sha256
        ),
        "source_inventory_gpu_count": receipt.inventory_gpu_count,
        "compute_gpu_hours": compute,
        "reserved_gpu_hours": reserved,
        "estimated_wall_hours": wall,
        "retry_reserve_gpu_hours": retry,
        "profile_reserve_gpu_hours": receipt.profile_reserve_gpu_hours,
        "evidence_reserve_gpu_hours": receipt.evidence_reserve_gpu_hours,
    }
    return GpuHourEstimate(
        status="AVAILABLE",
        source_pilot_receipt_sha256=signed.sha256,
        compute_gpu_hours=compute,
        reserved_gpu_hours=reserved,
        estimated_wall_hours=wall,
        retry_reserve_gpu_hours=retry,
        profile_reserve_gpu_hours=receipt.profile_reserve_gpu_hours,
        evidence_reserve_gpu_hours=receipt.evidence_reserve_gpu_hours,
        source_schedule_sha256=receipt.schedule_sha256,
        source_materialization_receipt_sha256=(receipt.materialization_receipt_sha256),
        source_inventory_gpu_count=receipt.inventory_gpu_count,
        derivation_sha256=content_sha256(derivation),
    )


def _reduce_stage_gpu_hour_envelope_from_signed_pilots_diagnostic(
    signed: SignedPilotDurationReceipt,
    *,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    protocol_lock_sha256: str,
    materialization_receipt_sha256: str,
    schedule_sha256: str,
    now_ns: int | None = None,
) -> StageGpuHourEnvelope:
    estimate = _reduce_gpu_hours_from_signed_pilots_diagnostic(
        signed,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        protocol_lock_sha256=protocol_lock_sha256,
        materialization_receipt_sha256=materialization_receipt_sha256,
        schedule_sha256=schedule_sha256,
        now_ns=now_ns,
    )
    return StageGpuHourEnvelope(
        schema_version=1,
        protocol_lock_sha256=protocol_lock_sha256,
        materialization_receipt_sha256=materialization_receipt_sha256,
        signed_pilot_receipt_sha256=signed.sha256,
        schedule_sha256=schedule_sha256,
        estimate=estimate,
    )


def reduce_gpu_hours_from_signed_pilots(
    signed: SignedPilotDurationReceipt,
    *,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    protocol_lock_sha256: str,
    materialization_receipt_sha256: str,
    schedule_sha256: str,
    now_ns: int | None = None,
) -> GpuHourEstimate:
    """Reject signer-authored phase durations on the formal path.

    Schema-2 pilot receipts bind only scalar phase milliseconds and a bare
    terminal digest.  Until a source-owned lifecycle proof exposes every
    phase timestamp, they cannot authorize a budget or launch.
    """

    del (
        signed,
        policy,
        expected_policy_sha256,
        protocol_lock_sha256,
        materialization_receipt_sha256,
        schedule_sha256,
        now_ns,
    )
    raise FormalGpuHourAuthorityBlocked(
        "formal_lifecycle_phase_timing_proof_unregistered"
    )


def reduce_stage_gpu_hour_envelope_from_signed_pilots(
    signed: SignedPilotDurationReceipt,
    *,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    protocol_lock_sha256: str,
    materialization_receipt_sha256: str,
    schedule_sha256: str,
    now_ns: int | None = None,
) -> StageGpuHourEnvelope:
    """Fail closed rather than sign a caller-authored formal budget."""

    del (
        signed,
        policy,
        expected_policy_sha256,
        protocol_lock_sha256,
        materialization_receipt_sha256,
        schedule_sha256,
        now_ns,
    )
    raise FormalGpuHourAuthorityBlocked(
        "formal_lifecycle_phase_timing_proof_unregistered"
    )


@dataclass(frozen=True)
class E1Geometry:
    scope: str
    parameterization: Literal["full", "lora"]
    rank: int | None
    alpha_over_rank: float | None

    def __post_init__(self) -> None:
        if self.scope not in E1_SCOPES:
            raise ValueError("E1 geometry scope is not registered")
        if self.parameterization == "full":
            if self.rank is not None or self.alpha_over_rank is not None:
                raise ValueError("Full E1 geometry cannot carry LoRA fields")
        elif self.rank not in LORA_RANKS or self.alpha_over_rank != 1.0:
            raise ValueError("LoRA E1 geometry requires rank grid and alpha/r=1")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def e1_geometries() -> tuple[E1Geometry, ...]:
    rows = tuple(
        geometry
        for scope in E1_SCOPES
        for geometry in (
            E1Geometry(scope, "full", None, None),
            *(E1Geometry(scope, "lora", rank, 1.0) for rank in LORA_RANKS),
        )
    )
    if len(rows) != 32 or len({row.sha256 for row in rows}) != 32:
        raise AssertionError("E1 must define exactly 32 unique geometries")
    return rows


@dataclass(frozen=True)
class MaterializedCell:
    stage: str
    method_role: str
    model: str
    backend: str
    task: str
    publication_policy: str
    recipe_sha256: str | None
    dimensions: tuple[tuple[str, str | int | float], ...]

    def __post_init__(self) -> None:
        if self.stage not in FORMAL_STAGE_DAG:
            raise ValueError("materialized cell names an unknown formal stage")
        for name in ("method_role", "model", "backend", "task", "publication_policy"):
            value = _require_text(f"materialized cell {name}", getattr(self, name))
            if any(fragment in value.lower() for fragment in _UNRESOLVED_FRAGMENTS):
                raise ValueError("materialized cell contains an unresolved placeholder")
        if self.recipe_sha256 is not None:
            _require_sha256("materialized recipe digest", self.recipe_sha256)
        keys = tuple(key for key, _ in self.dimensions)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("materialized dimensions must be sorted and unique")
        for key, value in self.dimensions:
            _require_text("materialized dimension name", key)
            if type(value) not in {str, int, float}:
                raise TypeError("materialized dimensions must be scalar")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("materialized dimensions must be finite")
            if type(value) is str and any(
                fragment in value.lower() for fragment in _UNRESOLVED_FRAGMENTS
            ):
                raise ValueError("materialized cell contains an unresolved placeholder")
        reject_banned_model_identity(self)

    @cached_property
    def cell_id(self) -> str:
        return content_sha256(self)


def _cell(
    *,
    stage: str,
    method_role: str,
    model: str,
    backend: str,
    task: str,
    publication_policy: str,
    recipe_sha256: str | None,
    dimensions: dict[str, str | int | float],
) -> MaterializedCell:
    if method_role in {"TTS", "L0-naive"} and "tts_l0_pair_id" not in dimensions:
        if recipe_sha256 is None:
            raise ValueError("TTS/L0 matched pair requires the frozen recipe")
        dimensions = {
            **dimensions,
            "tts_l0_pair_id": content_sha256(
                {
                    "kind": "lightcone_tts_l0_materialized_pair",
                    "stage": stage,
                    "model": model,
                    "backend": backend,
                    "task": task,
                    "recipe_sha256": recipe_sha256,
                    "dimensions": tuple(sorted(dimensions.items())),
                }
            ),
        }
    return MaterializedCell(
        stage=stage,
        method_role=method_role,
        model=model,
        backend=backend,
        task=task,
        publication_policy=publication_policy,
        recipe_sha256=recipe_sha256,
        dimensions=tuple(sorted(dimensions.items())),
    )


@dataclass(frozen=True)
class StageMaterializationReceipt:
    schema_version: int
    stage: str
    protocol_lock_sha256: str
    upstream_receipt_sha256s: tuple[str, ...]
    source_decision_sha256: str
    materialization_rule: str
    expected_cell_count: int
    cells: tuple[MaterializedCell, ...]
    gpu_hours: GpuHourEstimate

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only StageMaterializationReceipt schema 1 is supported")
        if self.stage not in FORMAL_STAGE_DAG:
            raise ValueError("materialization receipt names an unknown stage")
        for name in ("protocol_lock_sha256", "source_decision_sha256"):
            _require_sha256(f"materialization {name}", getattr(self, name))
        if type(self.upstream_receipt_sha256s) is not tuple or len(
            set(self.upstream_receipt_sha256s)
        ) != len(self.upstream_receipt_sha256s):
            raise ValueError("materialization requires distinct upstream receipts")
        if self.stage == "preflight":
            if self.upstream_receipt_sha256s:
                raise ValueError("preflight materialization cannot have an upstream")
        elif not self.upstream_receipt_sha256s:
            raise ValueError("non-root materialization requires an upstream receipt")
        for digest in self.upstream_receipt_sha256s:
            _require_sha256("materialization upstream receipt", digest)
        _require_text("materialization rule", self.materialization_rule)
        if any(
            fragment in self.materialization_rule.lower()
            for fragment in _UNRESOLVED_FRAGMENTS
        ):
            raise ValueError("materialization rule cannot be a future placeholder")
        if type(self.expected_cell_count) is not int or self.expected_cell_count < 0:
            raise ValueError("materialization expected-cell count must be non-negative")
        if self.expected_cell_count == 0 and self.stage != "E0":
            raise ValueError(
                "only an all-N/A E0 compatibility receipt may emit no rows"
            )
        if len(self.cells) != self.expected_cell_count:
            raise ValueError("materialization does not contain its exact cell count")
        if any(type(cell) is not MaterializedCell for cell in self.cells):
            raise TypeError(
                "materialization cells must be exact MaterializedCell values"
            )
        if any(cell.stage != self.stage for cell in self.cells):
            raise ValueError("materialization contains a cell from another stage")
        ids = tuple(cell.cell_id for cell in self.cells)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("materialized cells must be unique and canonically sorted")
        if type(self.gpu_hours) is not GpuHourEstimate:
            raise TypeError("materialization requires a GPU-hour output")
        if self.gpu_hours.status != "UNMEASURED":
            raise ValueError(
                "materialization identity is immutable; GPU hours use a separate envelope"
            )
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def _receipt(
    *,
    stage: str,
    protocol_lock_sha256: str,
    upstream_receipt_sha256s: tuple[str, ...],
    source_decision_sha256: str,
    materialization_rule: str,
    cells: tuple[MaterializedCell, ...],
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    ordered = tuple(sorted(cells, key=lambda row: row.cell_id))
    return StageMaterializationReceipt(
        schema_version=1,
        stage=stage,
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=upstream_receipt_sha256s,
        source_decision_sha256=source_decision_sha256,
        materialization_rule=materialization_rule,
        expected_cell_count=len(ordered),
        cells=ordered,
        gpu_hours=gpu_hours,
    )


@dataclass(frozen=True)
class SignedStageMaterializationReceipt:
    payload: StageMaterializationReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> StageMaterializationReceipt:
        if type(self.payload) is not StageMaterializationReceipt:
            raise TypeError("signed materialization payload has the wrong type")
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


def _staged_registry_cells(stage: str):
    from lightcone_spec.experiments.registry import build_industrial_registry

    registry = build_industrial_registry()
    if registry.materialization_mode != "signed_staged":
        raise AssertionError("default registry is not signed-staged")
    cells = registry.cells_for(stage)
    expected = {"preflight": 10, "E3a": 360, "TTS-Cal": 288}.get(stage)
    if expected is None or len(cells) != expected:
        raise AssertionError("staged registry prefix cardinality changed")
    reject_banned_model_identity(cells)
    return registry, cells


def _prefix_cell(stage: str, cell, *, recipe_sha256: str | None = None):
    identity = cell.identity
    role = {
        "target_only": "Target-only",
        "static": "Static",
        "tts": "TTS-calibration-candidate",
    }.get(identity.method)
    if role is None:
        raise ValueError("staged prefix cell has an unknown method role")
    dimensions: dict[str, str | int | float] = {
        "registry_cell_id": cell.cell_id,
    }
    for name in (
        "block",
        "concurrency",
        "context",
        "learning_rate",
        "regime",
        "width",
    ):
        value = getattr(identity, name)
        if value is not None:
            dimensions[name] = value
    if stage == "TTS-Cal":
        dimensions["stride"] = int(
            identity.variant.removeprefix("tts_calibration:stride=")
        )
        dimensions["pilot_phase"] = "excluded"
    return _cell(
        stage=stage,
        method_role=role,
        model=identity.model,
        backend=identity.backend,
        task=identity.task,
        publication_policy=(
            "fixed_barrier"
            if stage == "TTS-Cal"
            else "tuning_only"
            if stage != "preflight"
            else "none"
        ),
        recipe_sha256=recipe_sha256,
        dimensions=dimensions,
    )


def materialize_preflight(
    *,
    protocol_lock_sha256: str,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize every mandatory concrete preflight assignment (10 rows)."""

    _require_sha256("preflight protocol lock", protocol_lock_sha256)
    registry, source_cells = _staged_registry_cells("preflight")
    cells = tuple(_prefix_cell("preflight", cell) for cell in source_cells)
    return _receipt(
        stage="preflight",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(),
        source_decision_sha256=registry.sha256,
        materialization_rule="all_10_mandatory_compile_exactness_interference_assignments",
        cells=cells,
        gpu_hours=gpu_hours,
    )


def _materialize_e3a_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_preflight_receipt_sha256: str,
    workload_authority_sha256: str,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Build a non-authorizing E3a fixture from scalar identities."""

    for name, digest in (
        ("E3a protocol lock", protocol_lock_sha256),
        ("E3a preflight receipt", upstream_preflight_receipt_sha256),
        ("E3a workload authority", workload_authority_sha256),
    ):
        _require_sha256(name, digest)
    _, source_cells = _staged_registry_cells("E3a")
    cells = tuple(_prefix_cell("E3a", cell) for cell in source_cells)
    return _receipt(
        stage="E3a",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_preflight_receipt_sha256,),
        source_decision_sha256=workload_authority_sha256,
        materialization_rule="exact_360_row_capacity_width_and_drift_grid",
        cells=cells,
        gpu_hours=gpu_hours,
    )


def materialize_e3a(
    *,
    registry_verification_receipt: object,
    protocol_lock: ProtocolLock,
    preflight_materialization: StageMaterializationReceipt,
    preflight_coverage: StageCoverageReceipt,
    now_ns: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize E3a only from durable all-COMPLETE preflight authority."""

    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError(
            "formal E3a materialization requires a durable registry verification "
            "receipt"
        )
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal E3a materialization requires an exact ProtocolLock")
    if type(preflight_materialization) is not StageMaterializationReceipt:
        raise TypeError("formal E3a materialization requires exact preflight cells")
    if type(preflight_coverage) is not StageCoverageReceipt:
        raise TypeError("formal E3a materialization requires exact preflight coverage")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("formal E3a materialization time must be non-negative")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    if (
        registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("formal E3a registry receipt belongs to another ProtocolLock")
    signed_materializations = tuple(
        row
        for row in registry_verification_receipt.cumulative_signed_materializations
        if row.payload.stage == "preflight"
        and row.payload.sha256 == preflight_materialization.sha256
    )
    signed_coverages = tuple(
        row
        for row in registry_verification_receipt.cumulative_signed_coverage
        if row.payload.stage == "preflight"
        and row.payload.sha256 == preflight_coverage.sha256
        and row.payload.materialization_receipt_sha256
        == preflight_materialization.sha256
    )
    if len(signed_materializations) != 1 or len(signed_coverages) != 1:
        raise ValueError(
            "formal E3a registry receipt lacks exact signed preflight coverage"
        )
    if tuple(row.stage for row in manifest.materializations) != ("preflight",):
        raise ValueError("formal E3a materialization requires the exact DAG prefix")
    preflight_coverage.validate_against(preflight_materialization)
    if any(row.status != "COMPLETE" for row in preflight_coverage.dispositions):
        raise ValueError("formal E3a requires all-COMPLETE preflight coverage")
    return _materialize_e3a_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_preflight_receipt_sha256=signed_coverages[0].sha256,
        workload_authority_sha256=(
            protocol_lock.formal_workload_e3a_authorization_sha256
        ),
        gpu_hours=gpu_hours,
    )


def _materialize_tts_calibration_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_e3a_receipt_sha256: str,
    calibration_authority_sha256: str,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Build a non-authorizing TTS-Cal fixture from scalar identities."""

    for name, digest in (
        ("TTS-Cal protocol lock", protocol_lock_sha256),
        ("TTS-Cal E3a receipt", upstream_e3a_receipt_sha256),
        ("TTS-Cal authority", calibration_authority_sha256),
    ):
        _require_sha256(name, digest)
    _, source_cells = _staged_registry_cells("TTS-Cal")
    cells = tuple(
        _prefix_cell(
            "TTS-Cal",
            cell,
            recipe_sha256=content_sha256(
                {
                    "authority_sha256": calibration_authority_sha256,
                    "learning_rate": cell.identity.learning_rate,
                    "stride": int(
                        cell.identity.variant.removeprefix("tts_calibration:stride=")
                    ),
                }
            ),
        )
        for cell in source_cells
    )
    return _receipt(
        stage="TTS-Cal",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_e3a_receipt_sha256,),
        source_decision_sha256=calibration_authority_sha256,
        materialization_rule="72_candidates_x_4_disjoint_excluded_pilots",
        cells=cells,
        gpu_hours=gpu_hours,
    )


def materialize_tts_calibration(
    *,
    registry_verification_receipt: object,
    protocol_lock: ProtocolLock,
    tts_calibration_authority: TtsCalibrationAuthority,
    e3a_materialization: StageMaterializationReceipt,
    e3a_coverage: StageCoverageReceipt,
    now_ns: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize TTS-Cal from the durable signed E3a six-output source."""

    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError(
            "formal TTS-Cal materialization requires a durable registry verification "
            "receipt"
        )
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal TTS-Cal requires an exact ProtocolLock")
    if type(tts_calibration_authority) is not TtsCalibrationAuthority:
        raise TypeError("formal TTS-Cal requires an exact calibration authority")
    if type(e3a_materialization) is not StageMaterializationReceipt:
        raise TypeError("formal TTS-Cal requires exact E3a cells")
    if type(e3a_coverage) is not StageCoverageReceipt:
        raise TypeError("formal TTS-Cal requires exact E3a coverage")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("formal TTS-Cal materialization time must be non-negative")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    if (
        registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or tts_calibration_authority.sha256
        != protocol_lock.tts_calibration_authority_sha256
    ):
        raise ValueError("formal TTS-Cal authority differs from ProtocolLock")
    if tuple(row.stage for row in manifest.materializations) != (
        "preflight",
        "E3a",
    ):
        raise ValueError("formal TTS-Cal requires the exact covered E3a DAG prefix")
    if e3a_materialization.sha256 not in {
        row.payload.sha256
        for row in registry_verification_receipt.cumulative_signed_materializations
        if row.payload.stage == "E3a"
    } or e3a_coverage.sha256 not in {
        row.payload.sha256
        for row in registry_verification_receipt.cumulative_signed_coverage
        if row.payload.stage == "E3a"
    }:
        raise ValueError("formal TTS-Cal registry receipt lacks exact E3a coverage")
    e3a_coverage.validate_against(e3a_materialization)
    if any(row.status != "COMPLETE" for row in e3a_coverage.dispositions):
        raise ValueError("formal TTS-Cal requires all-COMPLETE E3a coverage")
    artifacts = registry_verification_receipt.cumulative_e3a_staged_selection_artifacts
    signed_selections = (
        registry_verification_receipt.cumulative_signed_e3a_staged_selections
    )
    if len(artifacts) != 1 or len(signed_selections) != 1:
        raise ValueError("formal TTS-Cal requires one exact signed E3a selection")
    selection = signed_selections[0].payload
    selection.validate_artifact(artifacts[0])
    if (
        selection.protocol_lock_sha256 != protocol_lock.sha256
        or selection.e3a_materialization_receipt_sha256 != e3a_materialization.sha256
        or selection.e3a_coverage_receipt_sha256 != e3a_coverage.sha256
        or not any(
            row.stage == "E3a"
            and row.authority_kind == "e3a_staged_selection"
            and row.signed_authority_sha256 == signed_selections[0].sha256
            for row in manifest.source_authorities
        )
    ):
        raise ValueError("formal TTS-Cal E3a selection lineage is not exact")
    return _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_e3a_receipt_sha256=signed_selections[0].sha256,
        calibration_authority_sha256=tts_calibration_authority.sha256,
        gpu_hours=gpu_hours,
    )


def _materialize_e1_first_slice_from_verified_decisions(
    *,
    protocol_lock_sha256: str,
    tts_calibration_receipt_sha256: str,
    signed_tts_calibration_seal_sha256: str,
    e3a_selection_sha256: str,
    frozen_tts_recipe_sha256: str,
    e1_recipe_anchor_authority_sha256: str,
    model: str,
    matched_width: int,
    common_load: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize values already obtained from reopened signed decisions."""

    for name, digest in (
        ("protocol lock", protocol_lock_sha256),
        ("TTS calibration receipt", tts_calibration_receipt_sha256),
        ("signed TTS calibration seal", signed_tts_calibration_seal_sha256),
        ("E3a selection", e3a_selection_sha256),
        ("frozen TTS recipe", frozen_tts_recipe_sha256),
        ("E1 recipe-anchor authority", e1_recipe_anchor_authority_sha256),
    ):
        _require_sha256(name, digest)
    if type(matched_width) is not int or matched_width < 1:
        raise ValueError("E1 matched width must be positive")
    if type(common_load) is not int or common_load < 1:
        raise ValueError("E1 common load must be positive")
    reject_banned_model_identity(model)
    shared = {"common_load": common_load, "matched_width": matched_width}
    anchor_pair_id = content_sha256(
        {
            "kind": "lightcone_tts_l0_materialized_pair",
            "stage": "E1",
            "model": model,
            "backend": "DFLASH",
            "task": "LiveCodeBench_tuning",
            "recipe_sha256": frozen_tts_recipe_sha256,
            "dimensions": tuple(sorted(shared.items())),
        }
    )
    cells = [
        _cell(
            stage="E1",
            method_role="Target-only",
            model=model,
            backend="NONE",
            task="LiveCodeBench_tuning",
            publication_policy="none",
            recipe_sha256=None,
            dimensions={**shared, "anchor": "target_only"},
        ),
        _cell(
            stage="E1",
            method_role="Static",
            model=model,
            backend="DFLASH",
            task="LiveCodeBench_tuning",
            publication_policy="none",
            recipe_sha256=None,
            dimensions={**shared, "anchor": "static"},
        ),
        _cell(
            stage="E1",
            method_role="TTS",
            model=model,
            backend="DFLASH",
            task="LiveCodeBench_tuning",
            publication_policy="fixed_barrier",
            recipe_sha256=frozen_tts_recipe_sha256,
            dimensions={
                **shared,
                "anchor": "frozen_tts",
                "tts_l0_pair_id": anchor_pair_id,
            },
        ),
        _cell(
            stage="E1",
            method_role="L0-naive",
            model=model,
            backend="DFLASH",
            task="LiveCodeBench_tuning",
            publication_policy="first_ready",
            recipe_sha256=frozen_tts_recipe_sha256,
            dimensions={
                **shared,
                "anchor": "frozen_l0_naive",
                "tts_l0_pair_id": anchor_pair_id,
            },
        ),
    ]
    for geometry in e1_geometries():
        for optimizer in E1_OPTIMIZER_ANCHORS:
            recipe_sha256 = content_sha256(
                {
                    "kind": "e1_lightcone_candidate",
                    "geometry": geometry,
                    "optimizer_anchor": optimizer,
                    "matched_width": matched_width,
                    "recipe_anchor_authority_sha256": (
                        e1_recipe_anchor_authority_sha256
                    ),
                }
            )
            cells.append(
                _cell(
                    stage="E1",
                    method_role="LightCone-candidate",
                    model=model,
                    backend="DFLASH",
                    task="LiveCodeBench_tuning",
                    publication_policy="first_ready",
                    recipe_sha256=recipe_sha256,
                    dimensions={
                        **shared,
                        "alpha_over_rank": (
                            "none"
                            if geometry.alpha_over_rank is None
                            else geometry.alpha_over_rank
                        ),
                        "optimizer_anchor": optimizer,
                        "parameterization": geometry.parameterization,
                        "rank": "none" if geometry.rank is None else geometry.rank,
                        "scope": geometry.scope,
                    },
                )
            )
    if len(cells) != 68:
        raise AssertionError("E1 first slice must contain exactly 68 concrete cells")
    return _receipt(
        stage="E1",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(
            tts_calibration_receipt_sha256,
            signed_tts_calibration_seal_sha256,
            e3a_selection_sha256,
        ),
        source_decision_sha256=e3a_selection_sha256,
        materialization_rule="four_fixed_anchors_plus_32_geometries_x_2_optimizers",
        cells=tuple(cells),
        gpu_hours=gpu_hours,
    )


def _materialize_e1_first_slice_with_verified_policy(
    *,
    protocol_lock: ProtocolLock,
    tts_calibration_materialization: StageMaterializationReceipt,
    tts_calibration_coverage: StageCoverageReceipt,
    signed_tts_calibration_seal: SignedTtsCalibrationSeal,
    tts_calibration_authority: TtsCalibrationAuthority,
    tts_seal_policy: TrustedAttesterPolicy,
    expected_tts_seal_policy_sha256: str,
    e3a_materialization: StageMaterializationReceipt,
    e3a_coverage: StageCoverageReceipt,
    signed_e3a_selection: SignedE3aSelectionReceipt,
    e3a_reduction_authority: E3aSelectionReductionAuthority,
    e3a_selection_policy: TrustedAttesterPolicy,
    expected_e3a_selection_policy_sha256: str,
    now_ns: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Internal reducer after the caller's signing policy is root-authorized.

    The public formal entry point intentionally has no scalar model, width,
    load, recipe, or arbitrary decision-digest arguments.  Those values are
    recovered from upstream authorities and checked again immediately before
    constructing the immutable 68-cell receipt.
    """

    from lightcone_spec.experiments.selection_authority import (
        E3aSelectionReductionAuthority,
    )
    from lightcone_spec.experiments.stage_decisions import (
        SignedE3aSelectionReceipt,
    )

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E1 materialization requires an exact ProtocolLock")
    if type(tts_calibration_materialization) is not StageMaterializationReceipt:
        raise TypeError("E1 requires the exact TTS-Cal materialization")
    if type(tts_calibration_coverage) is not StageCoverageReceipt:
        raise TypeError("E1 requires the exact TTS-Cal coverage")
    if type(signed_tts_calibration_seal) is not SignedTtsCalibrationSeal:
        raise TypeError("E1 requires the exact signed TTS calibration seal")
    if type(tts_calibration_authority) is not TtsCalibrationAuthority:
        raise TypeError("E1 requires the exact TTS calibration authority")
    if type(e3a_materialization) is not StageMaterializationReceipt:
        raise TypeError("E1 requires the exact E3a materialization")
    if type(e3a_coverage) is not StageCoverageReceipt:
        raise TypeError("E1 requires the exact E3a coverage")
    if type(signed_e3a_selection) is not SignedE3aSelectionReceipt:
        raise TypeError("E1 requires the exact signed E3a selection")
    if type(e3a_reduction_authority) is not E3aSelectionReductionAuthority:
        raise TypeError("E1 requires the exact E3a reduction authority")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E1 authority verification time must be non-negative")
    if (
        tts_calibration_materialization.stage != "TTS-Cal"
        or tts_calibration_materialization.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("E1 TTS-Cal materialization differs from ProtocolLock")
    tts_calibration_coverage.validate_against(tts_calibration_materialization)
    if any(row.status != "COMPLETE" for row in tts_calibration_coverage.dispositions):
        raise ValueError("E1 requires all-COMPLETE TTS-Cal coverage")
    if (
        protocol_lock.tts_calibration_authority_sha256
        != tts_calibration_authority.sha256
        or tts_calibration_materialization.source_decision_sha256
        != tts_calibration_authority.sha256
    ):
        raise ValueError("E1 TTS authority differs from ProtocolLock or TTS-Cal")
    tts_seal = signed_tts_calibration_seal.verify(
        authority=tts_calibration_authority,
        policy=tts_seal_policy,
        expected_policy_sha256=expected_tts_seal_policy_sha256,
        now_ns=now_ns,
    )
    if (
        tts_seal.protocol_lock_sha256 != protocol_lock.sha256
        or tts_seal.materialization_receipt_sha256
        != tts_calibration_materialization.sha256
        or tts_seal.coverage_receipt_sha256 != tts_calibration_coverage.sha256
    ):
        raise ValueError("signed TTS seal differs from exact TTS-Cal lineage")
    e3a_selection = signed_e3a_selection.verify(
        protocol_lock=protocol_lock,
        materialization=e3a_materialization,
        coverage=e3a_coverage,
        reduction_authority=e3a_reduction_authority,
        policy=e3a_selection_policy,
        expected_policy_sha256=expected_e3a_selection_policy_sha256,
        now_ns=now_ns,
    )
    return _materialize_e1_first_slice_from_verified_decisions(
        protocol_lock_sha256=protocol_lock.sha256,
        tts_calibration_receipt_sha256=tts_calibration_materialization.sha256,
        signed_tts_calibration_seal_sha256=(signed_tts_calibration_seal.sha256),
        e3a_selection_sha256=signed_e3a_selection.sha256,
        frozen_tts_recipe_sha256=tts_seal.selected_candidate_id,
        e1_recipe_anchor_authority_sha256=(
            protocol_lock.e1_recipe_anchor_authority_sha256
        ),
        model=e3a_selection.model,
        matched_width=e3a_selection.matched_width,
        common_load=e3a_selection.common_load,
        gpu_hours=gpu_hours,
    )


def materialize_e1_first_slice(
    *,
    registry_verification_receipt: object,
    protocol_lock: ProtocolLock,
    tts_calibration_materialization: StageMaterializationReceipt,
    tts_calibration_coverage: StageCoverageReceipt,
    e3a_materialization: StageMaterializationReceipt,
    e3a_coverage: StageCoverageReceipt,
    now_ns: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize E1 from the staged-native E3a and TTS-Cal source rows.

    The public path deliberately has no legacy ``E3aSelectionReductionAuthority``
    or caller-supplied signer policy.  Both the six-output E3a receipt and the
    TTS-Cal seal are reopened from the durable append-only registry receipt.
    """

    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError(
            "formal E1 materialization requires a durable registry verification receipt"
        )
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal E1 materialization requires an exact ProtocolLock")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("formal E1 materialization time must be non-negative")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    if (
        registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("formal E1 registry receipt belongs to another ProtocolLock")
    if tuple(row.stage for row in manifest.materializations) != (
        "preflight",
        "E3a",
        "TTS-Cal",
    ):
        raise ValueError("formal E1 materialization requires exact covered DAG prefix")
    required_materializations = {
        tts_calibration_materialization.sha256,
        e3a_materialization.sha256,
    }
    required_coverage = {
        tts_calibration_coverage.sha256,
        e3a_coverage.sha256,
    }
    if required_materializations - {
        row.materialization_receipt_sha256 for row in manifest.materializations
    } or required_coverage - {row.coverage_receipt_sha256 for row in manifest.coverage}:
        raise ValueError("formal E1 registry receipt lacks exact upstream coverage")
    tts_calibration_coverage.validate_against(tts_calibration_materialization)
    e3a_coverage.validate_against(e3a_materialization)
    if any(
        row.status != "COMPLETE"
        for coverage in (tts_calibration_coverage, e3a_coverage)
        for row in coverage.dispositions
    ):
        raise ValueError("formal E1 requires all-COMPLETE upstream coverage")
    authorities = registry_verification_receipt.cumulative_tts_calibration_authorities
    signed_seals = registry_verification_receipt.cumulative_signed_tts_calibration_seals
    e3a_artifacts = (
        registry_verification_receipt.cumulative_e3a_staged_selection_artifacts
    )
    signed_e3a_selections = (
        registry_verification_receipt.cumulative_signed_e3a_staged_selections
    )
    if (
        len(authorities) != 1
        or len(signed_seals) != 1
        or len(e3a_artifacts) != 1
        or len(signed_e3a_selections) != 1
    ):
        raise ValueError("formal E1 registry receipt lacks exact staged sources")
    tts_authority = authorities[0]
    signed_tts_seal = signed_seals[0]
    tts_seal = signed_tts_seal.payload
    e3a_artifact = e3a_artifacts[0]
    signed_e3a_selection = signed_e3a_selections[0]
    e3a_selection = signed_e3a_selection.payload
    tts_seal.validate_against(tts_authority)
    e3a_selection.validate_artifact(e3a_artifact)
    signed_tts_coverage = tuple(
        row
        for row in registry_verification_receipt.cumulative_signed_coverage
        if row.payload.sha256 == tts_calibration_coverage.sha256
    )
    if (
        len(signed_tts_coverage) != 1
        or tts_authority.sha256 != protocol_lock.tts_calibration_authority_sha256
        or tts_seal.protocol_lock_sha256 != protocol_lock.sha256
        or tts_seal.materialization_receipt_sha256
        != tts_calibration_materialization.sha256
        or tts_seal.coverage_receipt_sha256 != tts_calibration_coverage.sha256
        or e3a_selection.protocol_lock_sha256 != protocol_lock.sha256
        or e3a_selection.e3a_materialization_receipt_sha256
        != e3a_materialization.sha256
        or e3a_selection.e3a_coverage_receipt_sha256 != e3a_coverage.sha256
        or not {
            ("E3a", "e3a_staged_selection", signed_e3a_selection.sha256),
            ("TTS-Cal", "tts_calibration_seal", signed_tts_seal.sha256),
        }
        <= {
            (row.stage, row.authority_kind, row.signed_authority_sha256)
            for row in manifest.source_authorities
        }
    ):
        raise ValueError("formal E1 staged source lineage is not exact")
    return _materialize_e1_first_slice_from_verified_decisions(
        protocol_lock_sha256=protocol_lock.sha256,
        tts_calibration_receipt_sha256=signed_tts_coverage[0].sha256,
        signed_tts_calibration_seal_sha256=signed_tts_seal.sha256,
        e3a_selection_sha256=signed_e3a_selection.sha256,
        frozen_tts_recipe_sha256=tts_seal.selected_candidate_id,
        e1_recipe_anchor_authority_sha256=(
            protocol_lock.e1_recipe_anchor_authority_sha256
        ),
        model=e3a_selection.model,
        matched_width=e3a_selection.matched_width,
        common_load=e3a_selection.common_load,
        gpu_hours=gpu_hours,
    )


@dataclass(frozen=True)
class OptimizerRateGrid:
    optimizer: str
    parameterization: Literal["full", "lora"]
    learning_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.optimizer not in E2_OPTIMIZERS:
            raise ValueError("E2 rate grid names an unknown optimizer")
        if (
            type(self.learning_rates) is not tuple
            or len(self.learning_rates) != 5
            or tuple(sorted(set(self.learning_rates))) != self.learning_rates
            or any(
                type(value) is not float or not math.isfinite(value) or value <= 0
                for value in self.learning_rates
            )
        ):
            raise ValueError(
                "E2 rate grid must contain five increasing positive floats"
            )


@dataclass(frozen=True)
class E2OptimizerNumericRecipe:
    """Every optimizer field that is not an E2 learning-rate/schedule axis.

    Keeping these values outside ``OptimizerConfig`` avoids inheriting a future
    schema default.  :meth:`optimizer_config` passes every optimizer field
    explicitly after joining the registered learning-rate and schedule axes.
    """

    optimizer: str
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    grad_clip: float | None
    momentum: float | None
    muon_ns_steps: int | None
    muon_auxiliary_learning_rate: float | None
    muon_auxiliary_weight_decay: float | None
    stride: int
    decay_semantics: Literal["none", "coupled_l2", "decoupled"]

    def __post_init__(self) -> None:
        if self.optimizer not in E2_OPTIMIZERS:
            raise ValueError("E2 numeric recipe names an unknown optimizer")
        for name in ("weight_decay", "beta1", "beta2", "epsilon"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"E2 numeric recipe {name} must be a finite float")
        if self.weight_decay < 0 or not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("E2 decay/beta numerics are outside their domains")
        if self.epsilon <= 0:
            raise ValueError("E2 optimizer epsilon must be positive")
        if self.grad_clip is not None and (
            type(self.grad_clip) is not float
            or not math.isfinite(self.grad_clip)
            or self.grad_clip <= 0
        ):
            raise ValueError("E2 gradient clipping must be null or positive")
        needs_momentum = self.optimizer in {"sgdm", "nag", "muon"}
        if needs_momentum != (self.momentum is not None):
            raise ValueError("E2 momentum presence differs from optimizer semantics")
        if self.momentum is not None and (
            type(self.momentum) is not float or not 0 < self.momentum < 1
        ):
            raise ValueError("E2 optimizer momentum must be in (0, 1)")
        muon_values = (
            self.muon_ns_steps,
            self.muon_auxiliary_learning_rate,
            self.muon_auxiliary_weight_decay,
        )
        if (self.optimizer == "muon") != all(
            value is not None for value in muon_values
        ):
            raise ValueError("E2 Muon auxiliary numerics are not exact")
        if self.optimizer != "muon" and any(value is not None for value in muon_values):
            raise ValueError("E2 Muon numerics are forbidden for another optimizer")
        if self.muon_ns_steps is not None and (
            type(self.muon_ns_steps) is not int or not 1 <= self.muon_ns_steps <= 20
        ):
            raise ValueError("E2 Muon Newton--Schulz steps are outside [1, 20]")
        for name in (
            "muon_auxiliary_learning_rate",
            "muon_auxiliary_weight_decay",
        ):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not float or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"E2 {name} must be finite and non-negative")
        if self.muon_auxiliary_learning_rate is not None and (
            self.muon_auxiliary_learning_rate <= 0
        ):
            raise ValueError("E2 Muon auxiliary learning rate must be positive")
        if type(self.stride) is not int or self.stride < 1:
            raise ValueError("E2 optimizer stride must be positive")
        expected_decay = {
            "adam": "none",
            "adamw": "decoupled",
            "sgdm": "coupled_l2",
            "nag": "coupled_l2",
            "muon": "decoupled",
            "lion": "decoupled",
            "chronobelief": "decoupled",
        }[self.optimizer]
        if self.decay_semantics != expected_decay:
            raise ValueError("E2 optimizer decay semantics differ from runtime")
        if (self.decay_semantics == "none") != (self.weight_decay == 0):
            raise ValueError("E2 no-decay recipe must use zero weight decay")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2ScheduleNumericRecipe:
    schedule: str
    total_published_updates: int | None
    counter_semantics: Literal["committed_published_updates_only"] = (
        "committed_published_updates_only"
    )

    def __post_init__(self) -> None:
        if self.schedule not in E2_SCHEDULES:
            raise ValueError("E2 schedule recipe names an unknown schedule")
        if (self.schedule == "cosine_to_zero") != (
            self.total_published_updates is not None
        ):
            raise ValueError("only E2 cosine schedule carries a finite horizon")
        if self.total_published_updates is not None and (
            type(self.total_published_updates) is not int
            or self.total_published_updates < 2
        ):
            raise ValueError("E2 cosine horizon must cover at least two updates")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2OptimizerRecipeAuthority:
    """Protocol-owned complete numerics for all seven E2 optimizers."""

    schema_version: int
    authority_id: str
    optimizer_recipes: tuple[E2OptimizerNumericRecipe, ...]
    schedule_recipes: tuple[E2ScheduleNumericRecipe, ...]
    clipping_semantics: Literal["global_l2_before_optimizer_moments"] = (
        "global_l2_before_optimizer_moments"
    )
    skipped_or_aborted_update_semantics: Literal["do_not_advance_state_or_counter"] = (
        "do_not_advance_state_or_counter"
    )
    chronobelief_age_semantics: Literal[
        "safe_boundary_version_minus_source_version"
    ] = "safe_boundary_version_minus_source_version"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E2 optimizer-recipe authority schema 1 is supported")
        _require_text("E2 optimizer-recipe authority ID", self.authority_id)
        if (
            type(self.optimizer_recipes) is not tuple
            or any(
                type(row) is not E2OptimizerNumericRecipe
                for row in self.optimizer_recipes
            )
            or tuple(row.optimizer for row in self.optimizer_recipes) != E2_OPTIMIZERS
        ):
            raise ValueError("E2 numeric authority must contain seven exact optimizers")
        if (
            type(self.schedule_recipes) is not tuple
            or any(
                type(row) is not E2ScheduleNumericRecipe
                for row in self.schedule_recipes
            )
            or tuple(row.schedule for row in self.schedule_recipes) != E2_SCHEDULES
        ):
            raise ValueError("E2 numeric authority must contain three exact schedules")
        for row in (*self.optimizer_recipes, *self.schedule_recipes):
            row.__post_init__()
        reject_banned_model_identity(self)

    def optimizer_recipe(self, optimizer: str) -> E2OptimizerNumericRecipe:
        matches = tuple(
            row for row in self.optimizer_recipes if row.optimizer == optimizer
        )
        if len(matches) != 1:
            raise ValueError("E2 optimizer numeric recipe is not unique")
        return matches[0]

    def schedule_recipe(self, schedule: str) -> E2ScheduleNumericRecipe:
        matches = tuple(
            row for row in self.schedule_recipes if row.schedule == schedule
        )
        if len(matches) != 1:
            raise ValueError("E2 schedule numeric recipe is not unique")
        return matches[0]

    def optimizer_config(
        self,
        *,
        optimizer: str,
        learning_rate: float,
        schedule: str,
    ):
        """Construct a complete config without consulting schema defaults."""

        from lightcone_spec.config.schema import OptimizerConfig

        numeric = self.optimizer_recipe(optimizer)
        schedule_numeric = self.schedule_recipe(schedule)
        return OptimizerConfig(
            name=numeric.optimizer,
            learning_rate=learning_rate,
            weight_decay=numeric.weight_decay,
            beta1=numeric.beta1,
            beta2=numeric.beta2,
            epsilon=numeric.epsilon,
            grad_clip=numeric.grad_clip,
            momentum=numeric.momentum,
            muon_ns_steps=numeric.muon_ns_steps,
            muon_auxiliary_learning_rate=numeric.muon_auxiliary_learning_rate,
            muon_auxiliary_weight_decay=numeric.muon_auxiliary_weight_decay,
            schedule=schedule_numeric.schedule,
            schedule_total_published_updates=(schedule_numeric.total_published_updates),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2RecipeGridAuthority:
    schema_version: int
    authority_id: str
    rate_grids: tuple[OptimizerRateGrid, ...]
    optimizer_recipe_authority: E2OptimizerRecipeAuthority
    optimizers: tuple[str, ...] = E2_OPTIMIZERS
    schedules: tuple[str, ...] = E2_SCHEDULES
    learning_rates_per_optimizer: int = 5

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("only E2 recipe-grid authority schema 2 is supported")
        _require_text("E2 recipe-grid authority ID", self.authority_id)
        if self.optimizers != E2_OPTIMIZERS or self.schedules != E2_SCHEDULES:
            raise ValueError("E2 optimizer or schedule grid differs from protocol")
        if self.learning_rates_per_optimizer != 5:
            raise ValueError("E2 requires exactly five learning rates")
        if type(self.optimizer_recipe_authority) is not E2OptimizerRecipeAuthority:
            raise TypeError("E2 grid requires complete optimizer numeric authority")
        self.optimizer_recipe_authority.__post_init__()
        keys = tuple((row.parameterization, row.optimizer) for row in self.rate_grids)
        expected = tuple(
            sorted(
                (parameterization, optimizer)
                for parameterization in ("full", "lora")
                for optimizer in E2_OPTIMIZERS
            )
        )
        if tuple(sorted(keys)) != expected or len(keys) != len(set(keys)):
            raise ValueError(
                "E2 authority must bind 14 parameterization/optimizer grids"
            )
        reject_banned_model_identity(self)

    def rates(self, *, optimizer: str, parameterization: str) -> tuple[float, ...]:
        matches = tuple(
            row.learning_rates
            for row in self.rate_grids
            if row.optimizer == optimizer and row.parameterization == parameterization
        )
        if len(matches) != 1:
            raise ValueError("E2 rate lookup is not unique")
        return matches[0]

    def optimizer_config_for(self, candidate: E2CandidateRecipe):
        """Rebuild the candidate config only from this complete authority."""

        if type(candidate) is not E2CandidateRecipe:
            raise TypeError("E2 config lookup requires an exact candidate recipe")
        if (
            candidate.optimizer_recipe_authority_sha256
            != self.optimizer_recipe_authority.sha256
            or candidate.learning_rate
            not in self.rates(
                optimizer=candidate.optimizer,
                parameterization=candidate.geometry.parameterization,
            )
        ):
            raise ValueError("E2 candidate differs from its numeric/grid authority")
        return self.optimizer_recipe_authority.optimizer_config(
            optimizer=candidate.optimizer,
            learning_rate=candidate.learning_rate,
            schedule=candidate.schedule,
        )

    def adaptation_config_for(
        self,
        candidate: E2CandidateRecipe,
        *,
        canvas_tokens: int,
        adaptation_group_id: str,
        chronobelief_gpu_proof_sha256: str | None = None,
    ):
        """Build every AdaptationConfig field from registered E2 authority."""

        from lightcone_spec.config.schema import AdaptationConfig
        from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY
        from lightcone_spec.runtime.readiness import (
            NATIVE_RUNTIME_RELEASE_CAPABILITY,
        )

        numeric = self.optimizer_recipe_authority.optimizer_recipe(candidate.optimizer)
        chronobelief = candidate.optimizer == "chronobelief"
        if chronobelief != (chronobelief_gpu_proof_sha256 is not None):
            raise ValueError("E2 ChronoBelief candidate lacks exact GPU proof identity")
        return AdaptationConfig(
            weight_update_mode=candidate.geometry.parameterization,
            parameter_scope=candidate.geometry.scope,
            kv_history_policy="frozen",
            adaptation_scope="cohort",
            adaptation_group_id=adaptation_group_id,
            optimizer=self.optimizer_config_for(candidate),
            rank=candidate.geometry.rank,
            lora_alpha=candidate.geometry.rank,
            lora_matrix_policy="registered_matrices_v1",
            native_head_policy="frozen",
            stride=numeric.stride,
            max_in_flight=1,
            canvas_tokens=canvas_tokens,
            loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
            extra_logical_delay=0,
            teacher_row_policy="update_round",
            verification_mode="native_scheduler",
            fixed_verification_budget=None,
            confidence_loss_weight=None,
            chronobelief_release_capability_sha256=(
                NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256 if chronobelief else None
            ),
            chronobelief_gpu_proof_sha256=chronobelief_gpu_proof_sha256,
            eagle3_qualification_model_selector_sha256=None,
            eagle3_qualification_compatibility_authority_sha256=None,
            eagle3_e0_execution_authority_sha256=None,
            eagle3_compatibility_authority_sha256=None,
            eagle3_model_selector_sha256=None,
            eagle3_native_gpu_proof_sha256=None,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def default_e2_recipe_grid_authority() -> E2RecipeGridAuthority:
    full = {
        "adam": (1e-7, 3e-7, 1e-6, 3e-6, 1e-5),
        "adamw": (1e-7, 3e-7, 1e-6, 3e-6, 1e-5),
        "sgdm": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
        "nag": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
        "muon": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
        "lion": (1e-8, 3e-8, 1e-7, 3e-7, 1e-6),
        "chronobelief": (1e-7, 3e-7, 1e-6, 3e-6, 1e-5),
    }
    lora = {
        "adam": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
        "adamw": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
        "sgdm": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        "nag": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        "muon": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        "lion": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
        "chronobelief": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    }
    numerics = E2OptimizerRecipeAuthority(
        schema_version=1,
        authority_id="lightcone-e2-seven-optimizer-numerics-v1",
        optimizer_recipes=(
            E2OptimizerNumericRecipe(
                "adam", 0.0, 0.9, 0.999, 1e-8, 1.0, None, None, None, None, 10, "none"
            ),
            E2OptimizerNumericRecipe(
                "adamw",
                0.01,
                0.9,
                0.999,
                1e-8,
                1.0,
                None,
                None,
                None,
                None,
                10,
                "decoupled",
            ),
            E2OptimizerNumericRecipe(
                "sgdm",
                0.01,
                0.9,
                0.999,
                1e-8,
                1.0,
                0.9,
                None,
                None,
                None,
                10,
                "coupled_l2",
            ),
            E2OptimizerNumericRecipe(
                "nag",
                0.01,
                0.9,
                0.999,
                1e-8,
                1.0,
                0.9,
                None,
                None,
                None,
                10,
                "coupled_l2",
            ),
            E2OptimizerNumericRecipe(
                "muon",
                0.01,
                0.9,
                0.999,
                1e-8,
                1.0,
                0.9,
                5,
                0.001,
                0.01,
                10,
                "decoupled",
            ),
            E2OptimizerNumericRecipe(
                "lion",
                0.01,
                0.9,
                0.99,
                1e-8,
                1.0,
                None,
                None,
                None,
                None,
                10,
                "decoupled",
            ),
            E2OptimizerNumericRecipe(
                "chronobelief",
                0.01,
                0.9,
                0.999,
                1e-8,
                1.0,
                None,
                None,
                None,
                None,
                10,
                "decoupled",
            ),
        ),
        schedule_recipes=(
            E2ScheduleNumericRecipe("constant", None),
            E2ScheduleNumericRecipe("inverse_sqrt_published_update", None),
            E2ScheduleNumericRecipe("cosine_to_zero", 64),
        ),
    )
    return E2RecipeGridAuthority(
        schema_version=2,
        authority_id="lightcone-e2-recipe-grid-v2",
        rate_grids=tuple(
            OptimizerRateGrid(optimizer, parameterization, rates)
            for parameterization, mapping in (("full", full), ("lora", lora))
            for optimizer, rates in mapping.items()
        ),
        optimizer_recipe_authority=numerics,
    )


@dataclass(frozen=True)
class E2CandidateRecipe:
    geometry: E1Geometry
    optimizer: str
    schedule: str
    learning_rate: float
    optimizer_recipe_authority_sha256: str

    def __post_init__(self) -> None:
        if self.optimizer not in E2_OPTIMIZERS or self.schedule not in E2_SCHEDULES:
            raise ValueError("E2 candidate lies outside optimizer/schedule grid")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("E2 candidate learning rate must be positive and finite")
        _require_sha256(
            "E2 candidate optimizer-recipe authority",
            self.optimizer_recipe_authority_sha256,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def e2_candidate_recipes(
    geometries: tuple[E1Geometry, ...],
    *,
    grid: E2RecipeGridAuthority,
) -> tuple[E2CandidateRecipe, ...]:
    if not geometries or len({row.sha256 for row in geometries}) != len(geometries):
        raise ValueError("E2 survivor geometries must be non-empty and unique")
    rows = tuple(
        E2CandidateRecipe(
            geometry,
            optimizer,
            schedule,
            learning_rate,
            grid.optimizer_recipe_authority.sha256,
        )
        for geometry in geometries
        for optimizer in E2_OPTIMIZERS
        for schedule in E2_SCHEDULES
        for learning_rate in grid.rates(
            optimizer=optimizer,
            parameterization=geometry.parameterization,
        )
    )
    if len(rows) != 105 * len(geometries):
        raise AssertionError("E2 round zero must contain 105 recipes per geometry")
    for row in rows:
        grid.optimizer_config_for(row)
    return tuple(sorted(rows, key=lambda row: row.sha256))


def e2_round_candidate_counts(geometry_count: int) -> tuple[int, int, int, int]:
    if type(geometry_count) is not int or not 1 <= geometry_count <= 32:
        raise ValueError("E2 geometry count must be in [1, 32]")
    counts = [105 * geometry_count]
    for _ in range(3):
        counts.append(max(math.ceil(counts[-1] / 4), 21))
    return tuple(counts)  # type: ignore[return-value]


def e2_total_cell_count(geometry_count: int) -> int:
    return 16 + sum(e2_round_candidate_counts(geometry_count))


def _materialize_e2_round_from_verified_values(
    *,
    protocol_lock_sha256: str,
    upstream_receipt_sha256: str,
    source_selection_sha256: str,
    grid: E2RecipeGridAuthority,
    geometries: tuple[E1Geometry, ...],
    round_index: int,
    model: str,
    matched_width: int,
    common_load: int,
    frozen_tts_recipe_sha256: str,
    candidate_recipes: tuple[E2CandidateRecipe, ...] | None,
    prior_round_materialization: StageMaterializationReceipt | None,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize one E2 round, including exactly four fixed anchors."""

    if type(round_index) is not int or round_index not in range(4):
        raise ValueError("E2 round index must be in [0, 4)")
    if type(matched_width) is not int or matched_width < 1:
        raise ValueError("E2 matched width must be positive")
    if type(common_load) is not int or common_load < 1:
        raise ValueError("E2 common load must be positive")
    for name, digest in (
        ("protocol lock", protocol_lock_sha256),
        ("E2 upstream receipt", upstream_receipt_sha256),
        ("E2 selection", source_selection_sha256),
        ("frozen TTS recipe", frozen_tts_recipe_sha256),
    ):
        _require_sha256(name, digest)
    expected_count = e2_round_candidate_counts(len(geometries))[round_index]
    universe = e2_candidate_recipes(geometries, grid=grid)
    if round_index == 0:
        if candidate_recipes is not None or prior_round_materialization is not None:
            raise ValueError(
                "E2 round zero candidates are derived without a prior E2 round"
            )
        selected = universe
    else:
        if (
            candidate_recipes is None
            or type(prior_round_materialization) is not StageMaterializationReceipt
        ):
            raise ValueError(
                "later E2 rounds require a concrete prior materialization and survivors"
            )
        if (
            prior_round_materialization.stage != "E2"
            or prior_round_materialization.sha256 != upstream_receipt_sha256
        ):
            raise ValueError(
                "later E2 round does not bind its exact prior materialization"
            )
        prior_candidates = {
            cell.recipe_sha256
            for cell in prior_round_materialization.cells
            if cell.method_role == "LightCone-candidate"
        }
        if None in prior_candidates:
            raise ValueError("prior E2 candidate lacks a concrete recipe identity")
        selected = tuple(sorted(candidate_recipes, key=lambda row: row.sha256))
        for row in selected:
            grid.optimizer_config_for(row)
        universe_by_id = {row.sha256: row for row in universe}
        if (
            len(selected) != expected_count
            or len({row.sha256 for row in selected}) != len(selected)
            or any(
                row.sha256 not in universe_by_id or universe_by_id[row.sha256] != row
                for row in selected
            )
            or any(row.sha256 not in prior_candidates for row in selected)
        ):
            raise ValueError(
                "E2 survivor set has wrong count, foreign recipes, or re-entered recipes"
            )
        selected_families = {(row.optimizer, row.schedule) for row in selected}
        expected_families = {
            (optimizer, schedule)
            for optimizer in E2_OPTIMIZERS
            for schedule in E2_SCHEDULES
        }
        if selected_families != expected_families:
            raise ValueError(
                "E2 survivor set must retain every optimizer/schedule family"
            )
    anchor_pair_id = content_sha256(
        {
            "kind": "lightcone_tts_l0_materialized_pair",
            "stage": "E2",
            "model": model,
            "backend": "DFLASH",
            "task": "LiveCodeBench_tuning",
            "recipe_sha256": frozen_tts_recipe_sha256,
            "round": round_index,
            "matched_width": matched_width,
            "common_load": common_load,
        }
    )
    shared_dimensions = {
        "common_load": common_load,
        "matched_width": matched_width,
        "round": round_index,
    }
    cells = [
        _cell(
            stage="E2",
            method_role=role,
            model=model,
            backend="NONE" if role == "Target-only" else "DFLASH",
            task="LiveCodeBench_tuning",
            publication_policy=(
                "fixed_barrier"
                if role == "TTS"
                else "first_ready"
                if role == "L0-naive"
                else "none"
            ),
            recipe_sha256=(
                frozen_tts_recipe_sha256 if role in {"TTS", "L0-naive"} else None
            ),
            dimensions={
                **shared_dimensions,
                "anchor": role.lower(),
                **(
                    {"tts_l0_pair_id": anchor_pair_id}
                    if role in {"TTS", "L0-naive"}
                    else {}
                ),
            },
        )
        for role in ("Target-only", "Static", "TTS", "L0-naive")
    ]
    cells.extend(
        _cell(
            stage="E2",
            method_role="LightCone-candidate",
            model=model,
            backend="DFLASH",
            task="LiveCodeBench_tuning",
            publication_policy="first_ready",
            recipe_sha256=row.sha256,
            dimensions={
                **shared_dimensions,
                "alpha_over_rank": (
                    "none"
                    if row.geometry.alpha_over_rank is None
                    else row.geometry.alpha_over_rank
                ),
                "geometry_sha256": row.geometry.sha256,
                "learning_rate": row.learning_rate,
                "optimizer": row.optimizer,
                "optimizer_numeric_recipe_sha256": (
                    grid.optimizer_recipe_authority.optimizer_recipe(
                        row.optimizer
                    ).sha256
                ),
                "optimizer_recipe_authority_sha256": (
                    row.optimizer_recipe_authority_sha256
                ),
                "parameterization": row.geometry.parameterization,
                "rank": "none" if row.geometry.rank is None else row.geometry.rank,
                "schedule": row.schedule,
                "schedule_numeric_recipe_sha256": (
                    grid.optimizer_recipe_authority.schedule_recipe(row.schedule).sha256
                ),
                "scope": row.geometry.scope,
                "stride": grid.optimizer_recipe_authority.optimizer_recipe(
                    row.optimizer
                ).stride,
            },
        )
        for row in selected
    )
    return _receipt(
        stage="E2",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_receipt_sha256,),
        source_decision_sha256=source_selection_sha256,
        materialization_rule=(
            "e2_round_0_105_per_geometry_plus_four_anchors"
            if round_index == 0
            else "e2_quarter_retention_floor_21_plus_four_anchors"
        ),
        cells=tuple(cells),
        gpu_hours=gpu_hours,
    )


def materialize_e2_round(
    *,
    registry_verification_receipt: object,
    protocol_lock: ProtocolLock,
    signed_e1_selection: object,
    e1_materialization: StageMaterializationReceipt,
    e1_coverage: StageCoverageReceipt,
    pareto_evidence_manifest: object,
    execution_bindings: tuple[object, ...],
    now_ns: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize E2 round zero from the signed, raw-reopened E1 survivors.

    Later rounds intentionally have no scalar/candidate-set fallback.  They are
    unlocked only by the path-bearing per-round E2 reduction receipt, which is
    a distinct downstream authority.
    """

    from lightcone_spec.experiments.e1_stage_authority import (
        E1StagedParetoEvidenceManifest,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )
    from lightcone_spec.experiments.stage_decisions import (
        SignedE1SurvivorSelectionReceipt,
    )

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E2 materialization requires an exact ProtocolLock")
    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError(
            "E2 materialization requires durable registry verification receipt"
        )
    if type(signed_e1_selection) is not SignedE1SurvivorSelectionReceipt:
        raise TypeError("E2 round zero requires a signed E1 survivor receipt")
    if type(pareto_evidence_manifest) is not E1StagedParetoEvidenceManifest:
        raise TypeError("E2 round zero requires staged-native E1 Pareto evidence")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("E2 round zero requires sealed E1 execution bindings")
    registry_manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    if (
        registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
        or registry_manifest.protocol_lock_sha256 != protocol_lock.sha256
        or e1_materialization.sha256
        not in {
            row.materialization_receipt_sha256
            for row in registry_manifest.materializations
        }
        or e1_coverage.sha256
        not in {row.coverage_receipt_sha256 for row in registry_manifest.coverage}
    ):
        raise ValueError("E2 registry receipt lacks exact E1 lineage")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    e3a_artifacts = (
        registry_verification_receipt.cumulative_e3a_staged_selection_artifacts
    )
    signed_e3a_selections = (
        registry_verification_receipt.cumulative_signed_e3a_staged_selections
    )
    if len(e3a_artifacts) != 1 or len(signed_e3a_selections) != 1:
        raise ValueError("E2 registry receipt lacks staged E3a source")
    selection = signed_e1_selection.verify(
        protocol_lock=protocol_lock,
        e1_materialization=e1_materialization,
        e1_coverage=e1_coverage,
        e3a_selection_artifact=e3a_artifacts[0],
        signed_e3a_selection=signed_e3a_selections[0],
        e3a_policy=policy,
        expected_e3a_policy_sha256=policy.sha256,
        pareto_evidence_manifest=pareto_evidence_manifest,
        execution_bindings=execution_bindings,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    grid = default_e2_recipe_grid_authority()
    if protocol_lock.e2_recipe_grid_authority_sha256 != grid.sha256:
        raise ValueError("E2 recipe grid differs from ProtocolLock")
    e1_dimensions = tuple(dict(cell.dimensions) for cell in e1_materialization.cells)
    matched_widths = {row.get("matched_width") for row in e1_dimensions}
    common_loads = {row.get("common_load") for row in e1_dimensions}
    if (
        len(matched_widths) != 1
        or len(common_loads) != 1
        or type(next(iter(matched_widths))) is not int
        or type(next(iter(common_loads))) is not int
    ):
        raise ValueError("E1 source lacks exact matched width/common load")
    return _materialize_e2_round_from_verified_values(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=e1_materialization.sha256,
        source_selection_sha256=signed_e1_selection.sha256,
        grid=grid,
        geometries=selection.surviving_geometries,
        round_index=0,
        model=selection.model,
        matched_width=next(iter(matched_widths)),  # type: ignore[arg-type]
        common_load=next(iter(common_loads)),  # type: ignore[arg-type]
        frozen_tts_recipe_sha256=selection.frozen_tts_recipe_sha256,
        candidate_recipes=None,
        prior_round_materialization=None,
        gpu_hours=gpu_hours,
    )


def materialize_e2_next_round(
    *,
    registry_verification_receipt: object,
    protocol_lock: ProtocolLock,
    signed_prior_selection: object,
    prior_materialization: StageMaterializationReceipt,
    prior_coverage: StageCoverageReceipt,
    source_recipes: tuple[E2CandidateRecipe, ...],
    evidence_manifest: object,
    execution_bindings: tuple[object, ...],
    now_ns: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize E2 round ``k+1`` only from its signed staged-native reducer.

    The caller cannot supply the survivor set.  It is recomputed from the
    path-bound result/timestamp evidence, verified under the release-root
    policy already sealed by the durable registry receipt, and then copied
    verbatim from the verified signed selection.
    """

    from lightcone_spec.experiments.e2_stage_authority import (
        E2StagedRoundEvidenceManifest,
        SignedE2StagedRoundSelectionReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError(
            "later E2 materialization requires durable registry verification"
        )
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("later E2 materialization requires exact ProtocolLock")
    if type(signed_prior_selection) is not SignedE2StagedRoundSelectionReceipt:
        raise TypeError("later E2 materialization requires signed staged selection")
    if type(evidence_manifest) is not E2StagedRoundEvidenceManifest:
        raise TypeError("later E2 materialization requires path-bound E2 evidence")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("later E2 materialization requires sealed E2 bindings")
    registry_manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    if (
        registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
        or registry_manifest.protocol_lock_sha256 != protocol_lock.sha256
        or prior_materialization.sha256
        not in {
            row.materialization_receipt_sha256
            for row in registry_manifest.materializations
        }
        or prior_coverage.sha256
        not in {row.coverage_receipt_sha256 for row in registry_manifest.coverage}
    ):
        raise ValueError("later E2 registry receipt lacks exact prior round lineage")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    selection = signed_prior_selection.verify(
        protocol_lock=protocol_lock,
        materialization=prior_materialization,
        coverage=prior_coverage,
        source_recipes=source_recipes,
        manifest=evidence_manifest,
        execution_bindings=execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if selection.round_index >= 3 or selection.final_recipe is not None:
        raise ValueError("E2 round three selection has no downstream round")
    models = {cell.model for cell in prior_materialization.cells}
    frozen_recipes = {
        cell.recipe_sha256
        for cell in prior_materialization.cells
        if cell.method_role in {"TTS", "L0-naive"}
    }
    geometries = tuple(
        sorted(
            {row.geometry for row in source_recipes},
            key=lambda row: row.sha256,
        )
    )
    if len(models) != 1 or len(frozen_recipes) != 1 or None in frozen_recipes:
        raise ValueError("prior E2 round lacks exact model/frozen TTS anchors")
    prior_dimensions = tuple(
        dict(cell.dimensions) for cell in prior_materialization.cells
    )
    matched_widths = {row.get("matched_width") for row in prior_dimensions}
    common_loads = {row.get("common_load") for row in prior_dimensions}
    if (
        len(matched_widths) != 1
        or len(common_loads) != 1
        or type(next(iter(matched_widths))) is not int
        or type(next(iter(common_loads))) is not int
    ):
        raise ValueError("prior E2 round lacks exact matched width/common load")
    grid = default_e2_recipe_grid_authority()
    if protocol_lock.e2_recipe_grid_authority_sha256 != grid.sha256:
        raise ValueError("E2 recipe grid differs from ProtocolLock")
    return _materialize_e2_round_from_verified_values(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=prior_materialization.sha256,
        source_selection_sha256=signed_prior_selection.sha256,
        grid=grid,
        geometries=geometries,
        round_index=selection.round_index + 1,
        model=next(iter(models)),
        matched_width=next(iter(matched_widths)),  # type: ignore[arg-type]
        common_load=next(iter(common_loads)),  # type: ignore[arg-type]
        frozen_tts_recipe_sha256=next(iter(frozen_recipes)),  # type: ignore[arg-type]
        candidate_recipes=selection.survivor_recipes,
        prior_round_materialization=prior_materialization,
        gpu_hours=gpu_hours,
    )


def _e4_strength2_screen_rows() -> tuple[tuple[tuple[str, str | int], ...], ...]:
    """Return an eight-run binary orthogonal array with pairwise balance."""

    factors = dict(E4_SCREEN_FACTOR_LEVELS)
    rows = []
    for a, b, c in itertools.product((0, 1), repeat=3):
        d = a ^ b ^ c
        rows.append(
            tuple(
                sorted(
                    (
                        ("update_stride", factors["update_stride"][a]),
                        ("microbatch", factors["microbatch"][b]),
                        ("coalescing", factors["coalescing"][c]),
                        ("stream_priority", factors["stream_priority"][d]),
                    )
                )
            )
        )
    result = tuple(rows)
    if len(result) != 8 or len(set(result)) != 8:
        raise AssertionError("E4 strength-2 screen must have eight unique rows")
    for left, right in itertools.combinations(range(4), 2):
        pairs = [(row[left][1], row[right][1]) for row in result]
        if len(set(pairs)) != 4 or any(pairs.count(pair) != 2 for pair in set(pairs)):
            raise AssertionError("E4 screen is not pairwise balanced")
    return result


def _materialize_e4_strength2_screen_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_e2_receipt_sha256: str,
    source_decision_sha256: str,
    model: str,
    lightcone_recipe_sha256: str,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize 8 orthogonal rows x 3 loads x 2 traffic classes."""

    for name, digest in (
        ("E4 protocol lock", protocol_lock_sha256),
        ("E4 E2 receipt", upstream_e2_receipt_sha256),
        ("E4 source decision", source_decision_sha256),
        ("E4 LightCone recipe", lightcone_recipe_sha256),
    ):
        _require_sha256(name, digest)
    reject_banned_model_identity(model)
    cells = tuple(
        _cell(
            stage="E4",
            method_role="LightCone",
            model=model,
            backend="DFLASH",
            task="mechanism_strength2_screen_headline",
            publication_policy="first_ready",
            recipe_sha256=lightcone_recipe_sha256,
            dimensions={
                **dict(factors),
                "load": load,
                "screen_row": screen_row,
                "traffic": traffic,
            },
        )
        for screen_row, factors in enumerate(_e4_strength2_screen_rows())
        for load in E4_LOADS
        for traffic in E4_TRAFFIC
    )
    if len(cells) != 48:
        raise AssertionError("E4 strength-2 screen must contain exactly 48 cells")
    return _receipt(
        stage="E4",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_e2_receipt_sha256,),
        source_decision_sha256=source_decision_sha256,
        materialization_rule="strength2_8_rows_x_3_loads_x_2_traffic",
        cells=cells,
        gpu_hours=gpu_hours,
    )


def _validate_e4_neighborhoods(
    neighborhoods: tuple[tuple[str, str | int | float, str | int | float], ...],
) -> tuple[tuple[str, str | int | float, str | int | float], ...]:
    expected_names = tuple(name for name, _ in E4_SCREEN_FACTOR_LEVELS)
    if type(neighborhoods) is not tuple or len(neighborhoods) != len(expected_names):
        raise ValueError("E4 local neighborhoods must bind two levels for four factors")
    if any(type(row) is not tuple or len(row) != 3 for row in neighborhoods):
        raise ValueError("E4 local neighborhoods must bind two levels for four factors")
    if tuple(row[0] for row in neighborhoods) != expected_names or any(
        row[1] == row[2]
        or type(row[1]) not in {str, int, float}
        or type(row[2]) not in {str, int, float}
        or (type(row[1]) is float and not math.isfinite(row[1]))
        or (type(row[2]) is float and not math.isfinite(row[2]))
        for row in neighborhoods
    ):
        raise ValueError("E4 local neighborhoods must bind two levels for four factors")
    reject_banned_model_identity(neighborhoods)
    return neighborhoods


def _materialize_e4_winner_neighborhood_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_screen_receipt_sha256: str,
    winner_decision_sha256: str,
    model: str,
    lightcone_recipe_sha256: str,
    factor_neighborhoods: tuple[tuple[str, str | int | float, str | int | float], ...],
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize the signed winner's 2^4 local full-factorial neighborhood."""

    for name, digest in (
        ("E4 protocol lock", protocol_lock_sha256),
        ("E4 screen receipt", upstream_screen_receipt_sha256),
        ("E4 winner decision", winner_decision_sha256),
        ("E4 LightCone recipe", lightcone_recipe_sha256),
    ):
        _require_sha256(name, digest)
    reject_banned_model_identity(model)
    neighborhoods = _validate_e4_neighborhoods(factor_neighborhoods)
    cells = tuple(
        _cell(
            stage="E4",
            method_role="LightCone",
            model=model,
            backend="DFLASH",
            task="winner_neighborhood_local_factorial_headline",
            publication_policy="first_ready",
            recipe_sha256=lightcone_recipe_sha256,
            dimensions={
                **{
                    neighborhoods[index][0]: level for index, level in enumerate(levels)
                },
                "load": load,
                "local_row": local_row,
                "traffic": traffic,
            },
        )
        for local_row, levels in enumerate(
            itertools.product(*(row[1:] for row in neighborhoods))
        )
        for load in E4_LOADS
        for traffic in E4_TRAFFIC
    )
    if len(cells) != 96:
        raise AssertionError("E4 local full factorial must contain exactly 96 cells")
    return _receipt(
        stage="E4",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_screen_receipt_sha256,),
        source_decision_sha256=winner_decision_sha256,
        materialization_rule="winner_neighborhood_2pow4_x_3_loads_x_2_traffic",
        cells=cells,
        gpu_hours=gpu_hours,
    )


def _materialize_e4_profiler_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_local_receipt_sha256: str,
    source_decision_sha256: str,
    selected_configuration_sha256: str,
    model: str,
    lightcone_recipe_sha256: str,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    """Materialize three profiler-only rows, separate from headline timing."""

    for name, digest in (
        ("E4 protocol lock", protocol_lock_sha256),
        ("E4 local receipt", upstream_local_receipt_sha256),
        ("E4 source decision", source_decision_sha256),
        ("E4 selected configuration", selected_configuration_sha256),
        ("E4 LightCone recipe", lightcone_recipe_sha256),
    ):
        _require_sha256(name, digest)
    reject_banned_model_identity(model)
    cells = tuple(
        _cell(
            stage="E4",
            method_role="LightCone",
            model=model,
            backend="DFLASH",
            task="mechanism_profile_only",
            publication_policy="diagnostic_only",
            recipe_sha256=lightcone_recipe_sha256,
            dimensions={
                "profiler": profiler,
                "selected_configuration_sha256": selected_configuration_sha256,
            },
        )
        for profiler in ("nvtx", "nsight_systems", "nsight_compute")
    )
    return _receipt(
        stage="E4",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_local_receipt_sha256,),
        source_decision_sha256=source_decision_sha256,
        materialization_rule="three_profiler_only_rows_separate_from_headline",
        cells=cells,
        gpu_hours=gpu_hours,
    )


def _validate_block_count(final_blocks: int) -> int:
    if type(final_blocks) is not int or not 12 <= final_blocks <= 20:
        raise ValueError("final block count must be in [12, 20]")
    return 4 + final_blocks


def _select_exact_final_prefix(
    cells: tuple[MaterializedCell, ...],
    *,
    selected_final_prefix: tuple[int, ...],
    expected_cells_per_block: int,
) -> tuple[MaterializedCell, ...]:
    """Project a powered fixture onto the non-overlapping formal prefix.

    Blocks ``0..3`` are tuning-only pilot authority and never become main
    registry rows.  The source block numbers remain ``4..`` so the signed
    power-prefix receipt and every later proof reducer share one exact mapping.
    """

    if (
        type(selected_final_prefix) is not tuple
        or not 12 <= len(selected_final_prefix) <= 20
        or selected_final_prefix
        != tuple(
            range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + len(selected_final_prefix))
        )
    ):
        raise ValueError("formal final prefix must be the exact powered 12--20 suffix")
    if type(expected_cells_per_block) is not int or expected_cells_per_block < 1:
        raise ValueError("formal final-prefix width must be positive")
    selected = tuple(
        cell
        for cell in cells
        if dict(cell.dimensions).get("block") in selected_final_prefix
    )
    if (
        len(selected) != expected_cells_per_block * len(selected_final_prefix)
        or {dict(cell.dimensions).get("block") for cell in selected}
        != set(selected_final_prefix)
        or any(dict(cell.dimensions).get("block_phase") != "final" for cell in selected)
    ):
        raise ValueError("formal final-prefix projection is not exact")
    return selected


def _materialize_e3b_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_receipt_sha256: str,
    source_decision_sha256: str,
    model: str,
    frozen_tts_recipe_sha256: str,
    lightcone_recipe_sha256: str,
    final_blocks: int,
    gpu_hours: GpuHourEstimate,
    lineage_dimensions: dict[str, str] | None = None,
) -> StageMaterializationReceipt:
    blocks = _validate_block_count(final_blocks)
    for name, digest in (
        ("protocol lock", protocol_lock_sha256),
        ("E3b upstream", upstream_receipt_sha256),
        ("E3b decision", source_decision_sha256),
        ("frozen TTS recipe", frozen_tts_recipe_sha256),
        ("LightCone recipe", lightcone_recipe_sha256),
    ):
        _require_sha256(name, digest)
    extra_dimensions = {} if lineage_dimensions is None else lineage_dimensions
    if any(
        type(key) is not str or type(value) is not str
        for key, value in extra_dimensions.items()
    ):
        raise TypeError("E3b lineage dimensions must be exact strings")
    cells = tuple(
        _cell(
            stage="E3b",
            method_role=role,
            model=model,
            backend="NONE" if role == "Target-only" else "DFLASH",
            task="heldout_long_context_confirmation",
            publication_policy=(
                "fixed_barrier"
                if role == "TTS"
                else "first_ready"
                if role in {"L0-naive", "LightCone"}
                else "none"
            ),
            recipe_sha256=(
                frozen_tts_recipe_sha256
                if role in {"TTS", "L0-naive"}
                else lightcone_recipe_sha256
                if role == "LightCone"
                else None
            ),
            dimensions={
                "block": block,
                "block_phase": "excluded_pilot" if block < 4 else "final",
                "context": context,
                "load": load,
                "regime": regime,
                "width_panel": width_panel,
                **extra_dimensions,
            },
        )
        for block in range(blocks)
        for role in FORMAL_METHOD_ROLES
        for context in E3B_CONTEXTS
        for regime in E3B_REGIMES
        for load in E3B_LOADS
        for width_panel in E3B_WIDTH_PANELS
    )
    if len(cells) != 480 * blocks:
        raise AssertionError("E3b must materialize exactly 480 cells per block")
    return _receipt(
        stage="E3b",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_receipt_sha256,),
        source_decision_sha256=source_decision_sha256,
        materialization_rule="five_roles_x_8_contexts_x_3_regimes_x_2_loads_x_2_widths",
        cells=cells,
        gpu_hours=gpu_hours,
    )


def e1a_configurations() -> tuple[tuple[str, str, str | int], ...]:
    adaptive = tuple(
        (scope, parameterization, "none" if rank is None else rank)
        for scope in E1_SCOPES
        for parameterization, rank in (
            ("full", None),
            *(("lora", rank) for rank in LORA_RANKS),
        )
    ) + tuple(
        (f"{depth}_native_heads", parameterization, "none" if rank is None else rank)
        for depth in ("last1", "last3", "last5")
        for parameterization, rank in (
            ("full", None),
            *(("lora", rank) for rank in LORA_RANKS),
        )
    )
    rows = (("target_only", "none", "none"), ("static", "none", "none")) + adaptive
    if len(rows) != 58 or len(set(rows)) != 58:
        raise AssertionError("E1a must contain exactly 58 unique configurations")
    return rows


def _materialize_e1a_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_receipt_sha256: str,
    source_decision_sha256: str,
    model: str,
    frozen_tts_recipe_sha256: str,
    lightcone_recipe_sha256: str,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    _require_sha256("E1a frozen TTS recipe", frozen_tts_recipe_sha256)
    cells = []
    for scope, parameterization, rank in e1a_configurations():
        for verification_mode in E1A_VERIFICATION_MODES:
            if scope == "target_only":
                role, backend, policy, recipe = "Target-only", "NONE", "none", None
            elif scope == "static":
                role, backend, policy, recipe = "Static", "DSPARK", "none", None
            else:
                role, backend, policy, recipe = (
                    "LightCone-candidate",
                    "DSPARK",
                    "first_ready",
                    lightcone_recipe_sha256,
                )
            cells.append(
                _cell(
                    stage="E1a",
                    method_role=role,
                    model=model,
                    backend=backend,
                    task="LiveCodeBench_tuning_disjoint_from_E5",
                    publication_policy=policy,
                    recipe_sha256=recipe,
                    dimensions={
                        "frozen_tts_recipe_sha256": frozen_tts_recipe_sha256,
                        "parameterization": parameterization,
                        "rank": rank,
                        "scope": scope,
                        "verification_mode": verification_mode,
                        "fixed_verification_budget": (
                            E1A_FIXED_VERIFICATION_BUDGET
                            if verification_mode == "fixed_verification_budget"
                            else E1A_NATIVE_VERIFICATION_BUDGET
                        ),
                    },
                )
            )
    if len(cells) != 116:
        raise AssertionError("E1a must materialize exactly 116 execution rows")
    return _receipt(
        stage="E1a",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_receipt_sha256,),
        source_decision_sha256=source_decision_sha256,
        materialization_rule="58_configurations_x_2_verification_modes",
        cells=tuple(cells),
        gpu_hours=gpu_hours,
    )


@dataclass(frozen=True)
class E5SelectedP99Anchor:
    backend: Literal["DFLASH", "DSPARK"]
    topology: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    family_id: str
    minimum_completions: int

    def __post_init__(self) -> None:
        if self.backend not in E5_BACKENDS or self.topology not in E5_TOPOLOGIES:
            raise ValueError("E5 p99 anchor backend or topology is not registered")
        _require_text("E5 p99 anchor family", self.family_id)
        if (
            type(self.minimum_completions) is not int
            or self.minimum_completions < 10_000
        ):
            raise ValueError("E5 p99 anchor requires at least 10,000 completions")
        reject_banned_model_identity(self)

    @cached_property
    def anchor_id(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E5AnchorSelectionReceipt:
    schema_version: int
    protocol_lock_sha256: str
    upstream_e1a_receipt_sha256: str
    power_prefix_decision_sha256: str
    anchors: tuple[E5SelectedP99Anchor, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E5 anchor-selection schema 1 is supported")
        for name, digest in (
            ("E5 anchor protocol lock", self.protocol_lock_sha256),
            ("E5 anchor E1a receipt", self.upstream_e1a_receipt_sha256),
            ("E5 anchor power-prefix decision", self.power_prefix_decision_sha256),
        ):
            _require_sha256(name, digest)
        ids = tuple(anchor.anchor_id for anchor in self.anchors)
        if not ids or ids != tuple(sorted(set(ids))):
            raise ValueError("E5 selected p99 anchors must be non-empty and canonical")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE5AnchorSelectionReceipt:
    payload: E5AnchorSelectionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> E5AnchorSelectionReceipt:
        if type(self.payload) is not E5AnchorSelectionReceipt:
            raise TypeError("signed E5 anchor-selection payload has the wrong type")
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
class E5FailureDiagnosticMember:
    failure: str
    backend: Literal["DFLASH", "DSPARK"]
    topology: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    cohort_count: int

    def __post_init__(self) -> None:
        if (
            self.failure not in E5_FAILURES
            or self.backend not in E5_BACKENDS
            or self.topology not in E5_TOPOLOGIES
            or self.cohort_count not in E5_COHORT_COUNTS
        ):
            raise ValueError("E5 failure diagnostic lies outside the locked matrix")

    @cached_property
    def member_id(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E5FailureDiagnosticAuthority:
    schema_version: int
    protocol_sha256: str
    members: tuple[E5FailureDiagnosticMember, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "only E5 failure diagnostic authority schema 1 is supported"
            )
        if self.protocol_sha256 != E5_FAILURE_DIAGNOSTIC_PROTOCOL_SHA256:
            raise ValueError("E5 failure diagnostic protocol identity differs")
        expected = {
            (failure, backend, topology, cohort_count)
            for failure in E5_FAILURES
            for backend in E5_BACKENDS
            for topology in E5_TOPOLOGIES
            for cohort_count in E5_COHORT_COUNTS
        }
        observed = {
            (row.failure, row.backend, row.topology, row.cohort_count)
            for row in self.members
        }
        ids = tuple(row.member_id for row in self.members)
        if len(self.members) != 264 or observed != expected:
            raise ValueError(
                "E5 failure diagnostic authority must cover exactly 264 rows"
            )
        if ids != tuple(sorted(set(ids))):
            raise ValueError("E5 failure diagnostic members are not canonical")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def default_e5_failure_diagnostic_authority() -> E5FailureDiagnosticAuthority:
    return E5FailureDiagnosticAuthority(
        schema_version=1,
        protocol_sha256=E5_FAILURE_DIAGNOSTIC_PROTOCOL_SHA256,
        members=tuple(
            sorted(
                (
                    E5FailureDiagnosticMember(
                        failure=failure,
                        backend=backend,
                        topology=topology,
                        cohort_count=cohort_count,
                    )
                    for failure in E5_FAILURES
                    for backend in E5_BACKENDS
                    for topology in E5_TOPOLOGIES
                    for cohort_count in E5_COHORT_COUNTS
                ),
                key=lambda row: row.member_id,
            )
        ),
    )


def _e5_family_id(family: str, dimensions: dict[str, str | int | float]) -> str:
    if family == "closed_loop":
        return f"closed_loop_c{dimensions['concurrency']}"
    if family == "open_loop":
        return f"open_loop_{dimensions['load_factor']}"
    if family == "trace_or_soak":
        return f"trace_or_soak_{dimensions['arrival']}"
    if family == "topology_cohort":
        return (
            f"topology_cohort_{dimensions['topology']}_"
            f"k{dimensions['cohort_count']}_{dimensions['cohort_distribution']}"
        )
    raise ValueError("E5 family is not registered")


def _e5_families() -> tuple[tuple[str, dict[str, str | int | float]], ...]:
    families: list[tuple[str, dict[str, str | int | float]]] = []
    families.extend(
        ("closed_loop", {"concurrency": concurrency})
        for concurrency in E5_CLOSED_LOOP_CONCURRENCY
    )
    families.extend(
        ("open_loop", {"load_factor": load_factor})
        for load_factor in E5_OPEN_LOOP_LOAD_FACTORS
    )
    families.extend(
        ("trace_or_soak", {"arrival": arrival})
        for arrival in E5_TRACE_AND_SOAK_ARRIVALS
    )
    families.extend(
        (
            "topology_cohort",
            {
                "cohort_count": cohort_count,
                "cohort_distribution": distribution,
                "topology": topology,
            },
        )
        for topology in E5_TOPOLOGIES
        for cohort_count in E5_COHORT_COUNTS
        for distribution in E5_COHORT_DISTRIBUTIONS
    )
    result = tuple(families)
    if len(result) != 45:
        raise AssertionError("E5 must register exactly 45 families per backend")
    return result


def _e5_headline_cells(
    *,
    model: str,
    frozen_tts_recipe_sha256: str,
    dflash_lightcone_recipe_sha256: str,
    dspark_lightcone_recipe_sha256: str,
    blocks: int,
    anchors: tuple[E5SelectedP99Anchor, ...],
    anchor_receipt_sha256: str | None,
    lineage_dimensions: dict[str, str] | None = None,
) -> tuple[MaterializedCell, ...]:
    for name, digest in (
        ("E5 frozen TTS recipe", frozen_tts_recipe_sha256),
        ("E5 DFlash LightCone recipe", dflash_lightcone_recipe_sha256),
        ("E5 DSpark LightCone recipe", dspark_lightcone_recipe_sha256),
    ):
        _require_sha256(name, digest)
    if type(anchors) is not tuple or any(
        type(row) is not E5SelectedP99Anchor for row in anchors
    ):
        raise TypeError("E5 headline anchors must be exact typed rows")
    if anchors and anchor_receipt_sha256 is None:
        raise ValueError("E5 selected anchors require their signed receipt identity")
    if anchor_receipt_sha256 is not None:
        _require_sha256("E5 anchor receipt", anchor_receipt_sha256)
    extra_dimensions = {} if lineage_dimensions is None else lineage_dimensions
    if any(
        type(key) is not str or type(value) is not str
        for key, value in extra_dimensions.items()
    ):
        raise TypeError("E5 headline lineage dimensions must be exact strings")
    families = _e5_families()
    registered_families = {
        (
            _e5_family_id(family, dimensions),
            backend,
            str(dimensions.get("topology", "tp1_dp1")),
        )
        for backend in E5_BACKENDS
        for family, dimensions in families
    }
    selected_by_family = {
        (anchor.family_id, anchor.backend, anchor.topology): anchor
        for anchor in anchors
    }
    if len(selected_by_family) != len(anchors):
        raise ValueError("E5 selected p99 anchors are duplicated")
    if set(selected_by_family) - registered_families:
        raise ValueError("E5 selected p99 anchor is foreign to the headline matrix")
    cells = tuple(
        _cell(
            stage="E5",
            method_role=role,
            model=model,
            backend="NONE" if role == "Target-only" else backend,
            task="production_slo_power_prefix",
            publication_policy=(
                "fixed_barrier"
                if role == "TTS"
                else "first_ready"
                if role in {"L0-naive", "LightCone"}
                else "none"
            ),
            recipe_sha256=(
                frozen_tts_recipe_sha256
                if role in {"TTS", "L0-naive"}
                else (
                    dflash_lightcone_recipe_sha256
                    if backend == "DFLASH"
                    else dspark_lightcone_recipe_sha256
                )
                if role == "LightCone"
                else None
            ),
            dimensions={
                "backend_authority": backend,
                "block": block,
                "block_phase": "excluded_pilot" if block < 4 else "final",
                "family": family,
                "family_id": _e5_family_id(family, dimensions),
                "topology": dimensions.get("topology", "tp1_dp1"),
                **(
                    {
                        "p99_extension_anchor_id": selected.anchor_id,
                        "p99_extension_minimum_completions": (
                            selected.minimum_completions
                        ),
                        "p99_extension_offered_requests": 11_000,
                        "p99_extension_selection_receipt_sha256": (
                            anchor_receipt_sha256
                        ),
                    }
                    if (
                        selected := selected_by_family.get(
                            (
                                _e5_family_id(family, dimensions),
                                backend,
                                str(dimensions.get("topology", "tp1_dp1")),
                            )
                        )
                    )
                    is not None
                    else {}
                ),
                **(
                    {
                        "p99_anchor_id": selected.anchor_id,
                        "p99_minimum_completions": selected.minimum_completions,
                        "p99_selection_receipt_sha256": anchor_receipt_sha256,
                    }
                    if selected is not None and role == "LightCone"
                    else {}
                ),
                **dimensions,
                **extra_dimensions,
            },
        )
        for block in range(blocks)
        for role in FORMAL_METHOD_ROLES
        for backend in E5_BACKENDS
        for family, dimensions in families
    )
    if len(cells) != 450 * blocks:
        raise AssertionError("E5 headline prefix must contain exactly 450B cells")
    return cells


def _materialize_e5_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_e1a_receipt_sha256: str,
    power_prefix_decision_sha256: str,
    model: str,
    frozen_tts_recipe_sha256: str,
    lightcone_recipe_sha256: str,
    final_blocks: int,
    signed_anchor_selection: SignedE5AnchorSelectionReceipt,
    failure_diagnostic_authority: E5FailureDiagnosticAuthority,
    anchor_policy: TrustedAttesterPolicy,
    expected_anchor_policy_sha256: str,
    now_ns: int | None,
    gpu_hours: GpuHourEstimate,
    dspark_lightcone_recipe_sha256: str | None = None,
) -> StageMaterializationReceipt:
    """Materialize the full powered E5 surface plus one-shot diagnostics.

    Failure injections are the locked 11 x 2 x 3 x 4 correctness-only matrix
    and therefore never multiply the pilot/final block prefix.  A selected p99
    anchor is a >=10,000-completion requirement on an existing headline cell;
    it never creates an additional materialized row.
    """

    blocks = _validate_block_count(final_blocks)
    dspark_recipe = (
        lightcone_recipe_sha256
        if dspark_lightcone_recipe_sha256 is None
        else dspark_lightcone_recipe_sha256
    )
    for name, digest in (
        ("E5 protocol lock", protocol_lock_sha256),
        ("E5 E1a receipt", upstream_e1a_receipt_sha256),
        ("E5 power-prefix decision", power_prefix_decision_sha256),
        ("E5 frozen TTS recipe", frozen_tts_recipe_sha256),
        ("E5 LightCone recipe", lightcone_recipe_sha256),
        ("E5 DSpark LightCone recipe", dspark_recipe),
    ):
        _require_sha256(name, digest)
    reject_banned_model_identity(model)
    selection = signed_anchor_selection.verify(
        policy=anchor_policy,
        expected_policy_sha256=expected_anchor_policy_sha256,
        now_ns=now_ns,
    )
    if type(failure_diagnostic_authority) is not E5FailureDiagnosticAuthority:
        raise TypeError("E5 requires the exact failure diagnostic authority")
    failure_diagnostic_authority.__post_init__()
    if (
        selection.protocol_lock_sha256 != protocol_lock_sha256
        or selection.upstream_e1a_receipt_sha256 != upstream_e1a_receipt_sha256
        or selection.power_prefix_decision_sha256 != power_prefix_decision_sha256
    ):
        raise ValueError("E5 selected-anchor receipt differs from stage lineage")

    headline = _e5_headline_cells(
        model=model,
        frozen_tts_recipe_sha256=frozen_tts_recipe_sha256,
        dflash_lightcone_recipe_sha256=lightcone_recipe_sha256,
        dspark_lightcone_recipe_sha256=dspark_recipe,
        blocks=blocks,
        anchors=selection.anchors,
        anchor_receipt_sha256=signed_anchor_selection.sha256,
    )

    failures = tuple(
        _cell(
            stage="E5",
            method_role="LightCone",
            model=model,
            backend=member.backend,
            task="deterministic_failure_injection",
            publication_policy="diagnostic_only",
            recipe_sha256=(
                lightcone_recipe_sha256 if member.backend == "DFLASH" else dspark_recipe
            ),
            dimensions={
                "diagnostic_only": "true",
                "failure": member.failure,
                "failure_authority_sha256": failure_diagnostic_authority.sha256,
                "failure_member_id": member.member_id,
                "topology": member.topology,
                "cohort_count": member.cohort_count,
            },
        )
        for member in failure_diagnostic_authority.members
    )
    if len(failures) != 264:
        raise AssertionError("E5 failure diagnostics must contain exactly 264 cells")
    cells = headline + failures
    if len(cells) != 450 * blocks + 264:
        raise AssertionError("E5 total differs from 450B plus 264 diagnostics")
    return _receipt(
        stage="E5",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_e1a_receipt_sha256,),
        source_decision_sha256=content_sha256(
            {
                "power_prefix_decision_sha256": power_prefix_decision_sha256,
                "signed_anchor_selection_sha256": signed_anchor_selection.sha256,
                "failure_diagnostic_authority_sha256": (
                    failure_diagnostic_authority.sha256
                ),
            }
        ),
        materialization_rule=(
            "450_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
        ),
        cells=cells,
        gpu_hours=gpu_hours,
    )


def _materialize_e6_diagnostic(
    *,
    protocol_lock_sha256: str,
    upstream_receipt_sha256: str,
    source_decision_sha256: str,
    frozen_tts_recipe_sha256: str,
    lightcone_recipe_sha256: str,
    final_blocks: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    blocks = _validate_block_count(final_blocks)
    cells = [
        _cell(
            stage="E6",
            method_role="Target-only",
            model=model,
            backend="NEXTN",
            task="immutable_metadata_interface_and_fit_preflight",
            publication_policy="none",
            recipe_sha256=None,
            dimensions={"topology": "tp2_dp1"},
        )
        for model in E6_MODELS
    ]
    cells.extend(
        _cell(
            stage="E6",
            method_role=role,
            model=model,
            backend="NONE" if role == "Target-only" else "NEXTN",
            task=task,
            publication_policy=(
                "fixed_barrier"
                if role == "TTS"
                else "first_ready"
                if role in {"L0-naive", "LightCone"}
                else "none"
            ),
            recipe_sha256=(
                frozen_tts_recipe_sha256
                if role in {"TTS", "L0-naive"}
                else lightcone_recipe_sha256
                if role == "LightCone"
                else None
            ),
            dimensions={
                "block": block,
                "block_phase": "excluded_pilot" if block < 4 else "final",
                "context": context,
                "load": "common_slo_load",
                "topology": "tp2_dp1",
            },
        )
        for block in range(blocks)
        for model in E6_MODELS
        for role in FORMAL_METHOD_ROLES
        for task in E6_TASKS
        for context in E6_CONTEXTS
    )
    if len(cells) != 2 + 60 * blocks:
        raise AssertionError("E6 must materialize exactly 2 + 60B rows")
    return _receipt(
        stage="E6",
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_receipt_sha256s=(upstream_receipt_sha256,),
        source_decision_sha256=source_decision_sha256,
        materialization_rule="two_model_preflights_plus_60_rows_per_block",
        cells=tuple(cells),
        gpu_hours=gpu_hours,
    )


def _e6_cells_from_verified_sources(
    *,
    signed_e5_confirmation_sha256: str,
    signed_model_compatibility_sha256: str,
    model_compatibility: object,
    frozen_tts_recipe_sha256: str,
    lightcone_recipe_sha256: str,
    block_indices: tuple[int, ...],
    power_prefix_source_sha256: str | None = None,
    pilot_materialization_receipt_sha256: str | None = None,
    pilot_coverage_receipt_sha256: str | None = None,
) -> tuple[MaterializedCell, ...]:
    """Build the exact E6 universe from verified typed source rows."""

    from lightcone_spec.experiments.e6_stage_authority import (
        E6ModelCompatibilityReceipt,
    )
    from lightcone_spec.experiments.formal_single_operator_e6_interface import (
        FormalSingleOperatorE6CompatibilityReceipt,
    )

    if type(model_compatibility) not in {
        E6ModelCompatibilityReceipt,
        FormalSingleOperatorE6CompatibilityReceipt,
    }:
        raise TypeError("E6 cells require exact verified model compatibility")
    if type(block_indices) is not tuple or any(
        type(block) is not int for block in block_indices
    ):
        raise TypeError("E6 block indices must be exact integers")
    is_pilot_prefix = block_indices == tuple(range(PILOT_BLOCK_COUNT))
    is_final_prefix = 12 <= len(block_indices) <= 20 and block_indices == tuple(
        range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + len(block_indices))
    )
    if not is_pilot_prefix and not is_final_prefix:
        raise ValueError("E6 blocks must be the exact pilot or final powered prefix")
    for label, digest in (
        ("signed E5 confirmation", signed_e5_confirmation_sha256),
        ("signed model compatibility", signed_model_compatibility_sha256),
        ("frozen TTS recipe", frozen_tts_recipe_sha256),
        ("LightCone recipe", lightcone_recipe_sha256),
    ):
        _require_sha256(f"E6 {label}", digest)
    optional_lineage = (
        power_prefix_source_sha256,
        pilot_materialization_receipt_sha256,
        pilot_coverage_receipt_sha256,
    )
    if any(value is not None for value in optional_lineage):
        if any(value is None for value in optional_lineage):
            raise ValueError("E6 power-prefix lineage must be complete")
        _require_sha256("E6 power-prefix source", power_prefix_source_sha256)
        _require_sha256(
            "E6 pilot materialization", pilot_materialization_receipt_sha256
        )
        _require_sha256("E6 pilot coverage", pilot_coverage_receipt_sha256)
    compatibility_by_model = {row.model: row for row in model_compatibility.models}
    if tuple(compatibility_by_model) != E6_MODELS:
        raise ValueError("E6 model compatibility order/panel is not exact")

    def dimensions(
        model: str,
        *,
        include_power_lineage: bool,
    ) -> dict[str, str | int | float]:
        row = compatibility_by_model[model]
        return {
            "content_verification_receipt_sha256": (
                row.content_verification_receipt_sha256
            ),
            "distributed_gpu_proof_sha256": row.distributed_gpu_proof_sha256,
            "drafter_member_id": row.drafter_member_id,
            "drafter_model_id": row.drafter_model_id,
            "drafter_revision": row.drafter_revision,
            "drafter_shard_manifest_sha256": row.drafter_shard_manifest_sha256,
            "e6_model_compatibility_row_sha256": row.sha256,
            "e6_verified_authority_sha256": row.verified_authority_sha256,
            "frozen_tts_recipe_sha256": frozen_tts_recipe_sha256,
            "gpu_uuid_order_sha256": content_sha256(row.gpu_uuids),
            "interface_sha256": row.interface_sha256,
            "inventory_sha256": row.inventory_sha256,
            "lightcone_recipe_sha256": lightcone_recipe_sha256,
            "native_gpu_proof_sha256": row.native_gpu_proof_sha256,
            "signed_e6_model_compatibility_sha256": (signed_model_compatibility_sha256),
            "source_adapter_version": row.source_adapter_version,
            "target_member_id": row.target_member_id,
            "target_model_id": row.target_model_id,
            "target_revision": row.target_revision,
            "target_shard_manifest_sha256": row.target_shard_manifest_sha256,
            "topology": "tp2_dp1",
            "topology_authority_sha256": row.topology_sha256,
            "upstream_e5_confirmation_sha256": (signed_e5_confirmation_sha256),
            **(
                {
                    "nextn_mtp_mode": row.nextn_mtp_mode,
                    "target_snapshot_sha256": row.target_snapshot_sha256,
                    "mtp_component_sha256": row.mtp_component_sha256,
                }
                if getattr(row, "nextn_mtp_mode", "external_drafter") == "built_in_mtp"
                else {}
            ),
            **(
                {"signed_power_prefix_sha256": power_prefix_source_sha256}
                if include_power_lineage and power_prefix_source_sha256 is not None
                else {}
            ),
            **(
                {
                    "pilot_materialization_receipt_sha256": (
                        pilot_materialization_receipt_sha256
                    ),
                    "pilot_coverage_receipt_sha256": (pilot_coverage_receipt_sha256),
                }
                if include_power_lineage
                and pilot_materialization_receipt_sha256 is not None
                and pilot_coverage_receipt_sha256 is not None
                else {}
            ),
        }

    # The two immutable interface/fit cells are globally executed exactly once
    # with the excluded-pilot prefix.  A powered final prefix reuses their
    # sealed authority but must not materialize or bill those cells again.
    cells = (
        [
            _cell(
                stage="E6",
                method_role="Target-only",
                model=model,
                backend="NEXTN",
                task="immutable_metadata_interface_and_fit_preflight",
                publication_policy="none",
                recipe_sha256=None,
                dimensions=dimensions(model, include_power_lineage=False),
            )
            for model in E6_MODELS
        ]
        if is_pilot_prefix
        else []
    )
    cells.extend(
        _cell(
            stage="E6",
            method_role=role,
            model=model,
            backend="NONE" if role == "Target-only" else "NEXTN",
            task=task,
            publication_policy=(
                "fixed_barrier"
                if role == "TTS"
                else "first_ready"
                if role in {"L0-naive", "LightCone"}
                else "none"
            ),
            recipe_sha256=(
                frozen_tts_recipe_sha256
                if role in {"TTS", "L0-naive"}
                else lightcone_recipe_sha256
                if role == "LightCone"
                else None
            ),
            dimensions={
                **dimensions(model, include_power_lineage=True),
                "block": block,
                "block_phase": "excluded_pilot" if block < 4 else "final",
                "context": context,
                "load": "common_slo_load",
            },
        )
        for block in block_indices
        for model in E6_MODELS
        for role in FORMAL_METHOD_ROLES
        for task in E6_TASKS
        for context in E6_CONTEXTS
    )
    expected = (len(E6_MODELS) if is_pilot_prefix else 0) + 60 * len(block_indices)
    if len(cells) != expected:
        raise AssertionError("E6 typed materialization cardinality differs")
    return tuple(cells)


@dataclass(frozen=True)
class E0CompatibilityDecision:
    model: str
    backend: str
    task: str
    disposition: Literal["VALID", "N/A"]
    reason_code: str
    interface_sha256: str
    task_native_workload_sha256: str

    def __post_init__(self) -> None:
        if (
            self.model not in E0_MODELS
            or self.backend not in E0_BACKENDS
            or self.task not in E0_TASKS
        ):
            raise ValueError(
                "E0 compatibility decision lies outside the 108-cell universe"
            )
        _require_text("E0 compatibility reason", self.reason_code)
        _require_sha256("E0 interface digest", self.interface_sha256)
        _require_sha256(
            "E0 task-native workload digest", self.task_native_workload_sha256
        )
        reject_banned_model_identity(self)

    @cached_property
    def decision_id(self) -> str:
        return content_sha256((self.model, self.backend, self.task))


@dataclass(frozen=True)
class E0CompatibilityReceipt:
    schema_version: int
    protocol_lock_sha256: str
    upstream_e6_receipt_sha256: str
    decisions: tuple[E0CompatibilityDecision, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E0 compatibility receipt schema 1 is supported")
        _require_sha256("E0 protocol lock", self.protocol_lock_sha256)
        _require_sha256("E0 upstream E6 receipt", self.upstream_e6_receipt_sha256)
        expected = {
            (model, backend, task)
            for model in E0_MODELS
            for backend in E0_BACKENDS
            for task in E0_TASKS
        }
        observed = {(row.model, row.backend, row.task) for row in self.decisions}
        ids = tuple(row.decision_id for row in self.decisions)
        if len(self.decisions) != 108 or observed != expected:
            raise ValueError(
                "E0 compatibility receipt must cover exactly 108 decisions"
            )
        if ids != tuple(sorted(set(ids))):
            raise ValueError("E0 compatibility decisions must be canonical and unique")
        reject_banned_model_identity(self)

    @property
    def valid_count(self) -> int:
        return sum(row.disposition == "VALID" for row in self.decisions)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE0CompatibilityReceipt:
    payload: E0CompatibilityReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> E0CompatibilityReceipt:
        if type(self.payload) is not E0CompatibilityReceipt:
            raise TypeError("signed E0 compatibility payload has the wrong type")
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


def _e0_tuning_cells_from_verified_sources(
    *,
    compatibility: E0CompatibilityReceipt,
    signed_compatibility_sha256: str,
    signed_e6_confirmation_sha256: str,
    onlinespec_source_authority_sha256: str,
    frozen_tts_recipe_sha256: str,
) -> tuple[MaterializedCell, ...]:
    """Return the exact source-owned OnlineSPEC grid outside the main DAG."""

    from lightcone_spec.experiments.e0_stage_authority import _role_for_method
    from lightcone_spec.experiments.onlinespec import onlinespec_candidates

    for label, digest in (
        ("signed compatibility", signed_compatibility_sha256),
        ("signed E6 confirmation", signed_e6_confirmation_sha256),
        ("OnlineSPEC source authority", onlinespec_source_authority_sha256),
        ("frozen TTS recipe", frozen_tts_recipe_sha256),
    ):
        _require_sha256(f"E0 tuning {label}", digest)
    compatibility.__post_init__()
    valid = tuple(row for row in compatibility.decisions if row.disposition == "VALID")
    if not valid:
        raise ValueError("E0 tuning has no proof-backed compatible combination")
    candidates = tuple(
        sorted(onlinespec_candidates(), key=lambda row: row.candidate_id)
    )
    common = {
        "signed_e0_compatibility_sha256": signed_compatibility_sha256,
        "signed_e6_confirmation_sha256": signed_e6_confirmation_sha256,
        "e0_onlinespec_source_authority_sha256": (onlinespec_source_authority_sha256),
    }
    cells: list[MaterializedCell] = []
    for decision in valid:
        pair_id = content_sha256(
            {
                "stage": "E0",
                "scope": "independent_onlinespec_tuning",
                "compatibility_decision_id": decision.decision_id,
                "frozen_tts_recipe_sha256": frozen_tts_recipe_sha256,
            }
        )
        dimensions = {
            **common,
            "compatibility_decision_id": decision.decision_id,
            "deployment_task": decision.task,
            "interface_sha256": decision.interface_sha256,
            "task_native_workload_sha256": (decision.task_native_workload_sha256),
            "tuning_window": "task_native_disjoint",
        }
        for role in ("Static", "TTS", "L0-naive"):
            cells.append(
                _cell(
                    stage="E0",
                    method_role=role,
                    model=decision.model,
                    backend=decision.backend,
                    task="independent_onlinespec_tuning",
                    publication_policy=(
                        "fixed_barrier"
                        if role == "TTS"
                        else "first_ready"
                        if role == "L0-naive"
                        else "none"
                    ),
                    recipe_sha256=(
                        frozen_tts_recipe_sha256
                        if role in {"TTS", "L0-naive"}
                        else None
                    ),
                    dimensions={
                        **dimensions,
                        **(
                            {"tts_l0_pair_id": pair_id}
                            if role in {"TTS", "L0-naive"}
                            else {}
                        ),
                    },
                )
            )
        cells.extend(
            _cell(
                stage="E0",
                method_role=_role_for_method(candidate.method, candidate=True),
                model=decision.model,
                backend=decision.backend,
                task="independent_onlinespec_tuning",
                publication_policy="tuning_only",
                recipe_sha256=candidate.candidate_id,
                dimensions={
                    **dimensions,
                    "candidate_id": candidate.candidate_id,
                    "onlinespec_method": candidate.method,
                },
            )
            for candidate in candidates
        )
    expected = len(valid) * (len(candidates) + 3)
    if len(cells) != expected:
        raise AssertionError("E0 tuning grid cardinality changed")
    return tuple(cells)


def _e0_cells_from_verified_sources(
    *,
    compatibility: E0CompatibilityReceipt,
    signed_compatibility_sha256: str,
    signed_e6_confirmation_sha256: str,
    signed_tuning_seals: tuple[object, ...],
    frozen_tts_recipe_sha256: str,
    lightcone_recipe_sha256: str,
    block_indices: tuple[int, ...],
    power_prefix_source_sha256: str | None = None,
    pilot_materialization_receipt_sha256: str | None = None,
    pilot_coverage_receipt_sha256: str | None = None,
) -> tuple[MaterializedCell, ...]:
    """Materialize the exact ``16VB`` pilot or ``16VN`` final E0 matrix."""

    from lightcone_spec.experiments.e0_stage_authority import (
        E0OnlineSpecTuningSeal,
        SignedE0OnlineSpecTuningSeal,
    )

    compatibility.__post_init__()
    valid = tuple(row for row in compatibility.decisions if row.disposition == "VALID")
    if not valid:
        raise ValueError("E0 has no proof-backed compatible combination")
    if type(signed_tuning_seals) is not tuple or any(
        type(row) is not SignedE0OnlineSpecTuningSeal for row in signed_tuning_seals
    ):
        raise TypeError("E0 cells require exact signed OnlineSPEC tuning seals")
    seals = tuple(row.payload for row in signed_tuning_seals)
    if any(type(row) is not E0OnlineSpecTuningSeal for row in seals):
        raise TypeError("E0 cells require exact OnlineSPEC tuning payloads")
    if tuple(row.decision_id for row in seals) != tuple(
        sorted(row.decision_id for row in valid)
    ):
        raise ValueError("E0 tuning seals do not cover every VALID decision")
    signed_by_decision = {
        signed.payload.decision_id: signed for signed in signed_tuning_seals
    }
    for label, digest in (
        ("signed compatibility", signed_compatibility_sha256),
        ("signed E6 confirmation", signed_e6_confirmation_sha256),
        ("frozen TTS recipe", frozen_tts_recipe_sha256),
        ("LightCone recipe", lightcone_recipe_sha256),
    ):
        _require_sha256(f"E0 cells {label}", digest)
    is_pilot = block_indices == tuple(range(PILOT_BLOCK_COUNT))
    is_final = 12 <= len(block_indices) <= 20 and block_indices == tuple(
        range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + len(block_indices))
    )
    if not is_pilot and not is_final:
        raise ValueError("E0 block indices must be exact pilot or powered final prefix")
    power_values = (
        power_prefix_source_sha256,
        pilot_materialization_receipt_sha256,
        pilot_coverage_receipt_sha256,
    )
    if is_final and any(value is None for value in power_values):
        raise ValueError("E0 final cells require complete pilot/power lineage")
    if is_pilot and any(value is not None for value in power_values):
        raise ValueError("E0 excluded pilots cannot claim a power decision")
    if is_final:
        for label, digest in (
            ("power prefix", power_prefix_source_sha256),
            ("pilot materialization", pilot_materialization_receipt_sha256),
            ("pilot coverage", pilot_coverage_receipt_sha256),
        ):
            _require_sha256(f"E0 final {label}", digest)
    tuning_set_sha256 = content_sha256(
        tuple(sorted(row.sha256 for row in signed_tuning_seals))
    )
    cells = []
    for block in block_indices:
        for decision in valid:
            signed_seal = signed_by_decision[decision.decision_id]
            recipes = {
                row.method_role: row.candidate_id
                for row in signed_seal.payload.selected_recipes
            }
            for role in E0_METHOD_ROLES:
                for load in E0_LOADS:
                    pair_id = content_sha256(
                        {
                            "stage": "E0",
                            "block": block,
                            "compatibility_decision_id": decision.decision_id,
                            "load": load,
                        }
                    )
                    lineage = {
                        "signed_e0_compatibility_sha256": (signed_compatibility_sha256),
                        "signed_e0_tuning_seal_sha256": signed_seal.sha256,
                        "signed_e0_tuning_seal_set_sha256": tuning_set_sha256,
                        "signed_e6_confirmation_sha256": (
                            signed_e6_confirmation_sha256
                        ),
                    }
                    if is_final:
                        lineage.update(
                            {
                                "pilot_coverage_receipt_sha256": (
                                    pilot_coverage_receipt_sha256
                                ),
                                "pilot_materialization_receipt_sha256": (
                                    pilot_materialization_receipt_sha256
                                ),
                                "signed_power_prefix_sha256": (
                                    power_prefix_source_sha256
                                ),
                            }
                        )
                    cells.append(
                        _cell(
                            stage="E0",
                            method_role=role,
                            model=decision.model,
                            backend=decision.backend,
                            task=decision.task,
                            publication_policy=(
                                "fixed_barrier"
                                if role == "TTS"
                                else "first_ready"
                                if role in {"L0-naive", "LightCone"}
                                else "independent_online"
                                if role.startswith("OnlineSPEC-")
                                else "none"
                            ),
                            recipe_sha256=(
                                frozen_tts_recipe_sha256
                                if role in {"TTS", "L0-naive"}
                                else lightcone_recipe_sha256
                                if role == "LightCone"
                                else recipes[role]
                                if role in E0_METHOD_ROLES[-3:]
                                else None
                            ),
                            dimensions={
                                **lineage,
                                "block": block,
                                "block_phase": (
                                    "excluded_pilot" if is_pilot else "final"
                                ),
                                "compatibility_decision_id": (decision.decision_id),
                                "interface_sha256": decision.interface_sha256,
                                "load": load,
                                "task_native_workload_sha256": (
                                    decision.task_native_workload_sha256
                                ),
                                **(
                                    {"tts_l0_pair_id": pair_id}
                                    if role in {"TTS", "L0-naive"}
                                    else {}
                                ),
                            },
                        )
                    )
    if len(cells) != 16 * len(valid) * len(block_indices):
        raise AssertionError("E0 serving matrix differs from exact 16VB")
    return tuple(cells)


def _materialize_e0_from_signed_compatibility_diagnostic(
    signed_compatibility: SignedE0CompatibilityReceipt,
    *,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
    source_decision_sha256: str,
    frozen_tts_recipe_sha256: str,
    lightcone_recipe_sha256: str,
    online_spec_recipe_sha256s: tuple[tuple[str, str, str], ...],
    final_blocks: int,
    gpu_hours: GpuHourEstimate,
) -> StageMaterializationReceipt:
    compatibility = signed_compatibility.verify(
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    if source_decision_sha256 != signed_compatibility.sha256:
        raise ValueError("E0 source decision must be the signed compatibility receipt")
    blocks = _validate_block_count(final_blocks)
    valid = tuple(row for row in compatibility.decisions if row.disposition == "VALID")
    online_roles = E0_METHOD_ROLES[-3:]
    expected_online_keys = {
        (decision.decision_id, role) for decision in valid for role in online_roles
    }
    observed_online_keys = {
        (decision_id, role) for decision_id, role, _ in online_spec_recipe_sha256s
    }
    if (
        len(online_spec_recipe_sha256s) != len(expected_online_keys)
        or observed_online_keys != expected_online_keys
    ):
        raise ValueError(
            "E0 requires one independent OnlineSPEC recipe seal per valid combination"
        )
    online_recipe_by_key: dict[tuple[str, str], str] = {}
    for decision_id, role, digest in online_spec_recipe_sha256s:
        _require_sha256("E0 OnlineSPEC recipe seal", digest)
        key = (decision_id, role)
        if key in online_recipe_by_key:
            raise ValueError("E0 OnlineSPEC recipe seals are duplicated")
        online_recipe_by_key[key] = digest
    cells = tuple(
        _cell(
            stage="E0",
            method_role=role,
            model=decision.model,
            backend=decision.backend,
            task=decision.task,
            publication_policy=(
                "fixed_barrier"
                if role == "TTS"
                else "first_ready"
                if role in {"L0-naive", "LightCone"}
                else "independent_online"
                if role.startswith("OnlineSPEC-")
                else "none"
            ),
            recipe_sha256=(
                frozen_tts_recipe_sha256
                if role in {"TTS", "L0-naive"}
                else lightcone_recipe_sha256
                if role == "LightCone"
                else online_recipe_by_key[(decision.decision_id, role)]
                if role.startswith("OnlineSPEC-")
                else None
            ),
            dimensions={
                "block": block,
                "block_phase": "excluded_pilot" if block < 4 else "final",
                "interface_sha256": decision.interface_sha256,
                "load": load,
                "task_native_workload_sha256": decision.task_native_workload_sha256,
            },
        )
        for decision in valid
        for block in range(blocks)
        for role in E0_METHOD_ROLES
        for load in E0_LOADS
    )
    if len(cells) != 16 * len(valid) * blocks:
        raise AssertionError("E0 must materialize exactly 16VB concrete rows")
    return _receipt(
        stage="E0",
        protocol_lock_sha256=compatibility.protocol_lock_sha256,
        upstream_receipt_sha256s=(
            compatibility.upstream_e6_receipt_sha256,
            signed_compatibility.sha256,
        ),
        source_decision_sha256=signed_compatibility.sha256,
        materialization_rule="valid_compatibilities_x_8_roles_x_2_loads_x_B_blocks",
        cells=cells,
        gpu_hours=gpu_hours,
    )


def _raise_unregistered_typed_stage_authority(stage: str) -> None:
    raise FormalStageMaterializationBlocked(
        f"{stage}_typed_source_authority_unregistered"
    )


def materialize_e4_strength2_screen(
    *,
    registry_verification_receipt: object,
    signed_e2_final_selection: object,
    e2_materialization: StageMaterializationReceipt,
    e2_coverage: StageCoverageReceipt,
    e2_source_recipes: tuple[E2CandidateRecipe, ...],
    e2_evidence_manifest: object,
    e2_execution_bindings: tuple[object, ...],
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize the E4 screen from the proof-reduced final E2 recipe.

    GPU-hour authority is deliberately downstream of this immutable
    UNMEASURED receipt.  Passing a caller-built estimate or envelope here is
    forbidden; dispatch later joins the independent lifecycle-derived budget.
    """

    from lightcone_spec.experiments.e2_stage_authority import (
        E2StagedRoundEvidenceManifest,
        SignedE2StagedRoundSelectionReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E4 screen requires durable registry verification")
    if type(signed_e2_final_selection) is not SignedE2StagedRoundSelectionReceipt:
        raise TypeError("E4 screen requires a signed E2 final selection")
    if type(e2_materialization) is not StageMaterializationReceipt:
        raise TypeError("E4 screen requires exact E2 materialization")
    if type(e2_coverage) is not StageCoverageReceipt:
        raise TypeError("E4 screen requires exact E2 coverage")
    if type(e2_evidence_manifest) is not E2StagedRoundEvidenceManifest:
        raise TypeError("E4 screen requires path-bound E2 evidence")
    if type(e2_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in e2_execution_bindings
    ):
        raise TypeError("E4 screen requires sealed E2 execution bindings")
    if gpu_hour_envelope is not None:
        raise TypeError(
            "E4 screen materialization accepts no caller-authored GPU-hour envelope"
        )
    registry_manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    if (
        signed_e2_final_selection
        not in registry_verification_receipt.cumulative_signed_e2_staged_selections
        or e2_materialization.sha256
        not in {
            row.materialization_receipt_sha256
            for row in registry_manifest.materializations
        }
        or e2_coverage.sha256
        not in {row.coverage_receipt_sha256 for row in registry_manifest.coverage}
    ):
        raise ValueError("E4 screen registry receipt lacks final E2 lineage")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    selection = signed_e2_final_selection.verify(
        protocol_lock=protocol_lock,
        materialization=e2_materialization,
        coverage=e2_coverage,
        source_recipes=e2_source_recipes,
        manifest=e2_evidence_manifest,
        execution_bindings=e2_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if selection.round_index != 3 or selection.final_recipe is None:
        raise ValueError("E4 screen requires the unique E2 round-three winner")
    dimensions = tuple(dict(cell.dimensions) for cell in e2_materialization.cells)
    rounds = {row.get("round") for row in dimensions}
    models = {cell.model for cell in e2_materialization.cells}
    candidate_ids = {
        cell.recipe_sha256
        for cell in e2_materialization.cells
        if cell.method_role == "LightCone-candidate"
    }
    if (
        e2_materialization.stage != "E2"
        or rounds != {3}
        or len(models) != 1
        or selection.final_recipe.sha256 not in candidate_ids
    ):
        raise ValueError("E4 screen source is not the exact final E2 round")
    return _materialize_e4_strength2_screen_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_e2_receipt_sha256=e2_materialization.sha256,
        source_decision_sha256=signed_e2_final_selection.sha256,
        model=next(iter(models)),
        lightcone_recipe_sha256=selection.final_recipe.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e4_winner_neighborhood(
    *,
    registry_verification_receipt: object,
    signed_e4_screen_selection: object,
    screen_materialization: StageMaterializationReceipt,
    screen_coverage: StageCoverageReceipt,
    screen_evidence_manifest: object,
    screen_execution_bindings: tuple[object, ...],
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize the local factorial from a proof-reduced screen winner."""

    from lightcone_spec.experiments.e4_stage_authority import (
        E4StagedEvidenceManifest,
        SignedE4StageSelectionReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E4 local factorial requires durable registry verification")
    if type(signed_e4_screen_selection) is not SignedE4StageSelectionReceipt:
        raise TypeError("E4 local factorial requires signed screen selection")
    if type(screen_materialization) is not StageMaterializationReceipt:
        raise TypeError("E4 local factorial requires exact screen materialization")
    if type(screen_coverage) is not StageCoverageReceipt:
        raise TypeError("E4 local factorial requires exact screen coverage")
    if type(screen_evidence_manifest) is not E4StagedEvidenceManifest:
        raise TypeError("E4 local factorial requires path-bound screen evidence")
    if type(screen_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in screen_execution_bindings
    ):
        raise TypeError("E4 local factorial requires sealed screen bindings")
    if gpu_hour_envelope is not None:
        raise TypeError(
            "E4 local materialization accepts no caller-authored GPU-hour envelope"
        )
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    if (
        signed_e4_screen_selection
        not in registry_verification_receipt.cumulative_signed_e4_stage_selections
        or screen_materialization.sha256
        not in {row.materialization_receipt_sha256 for row in manifest.materializations}
        or screen_coverage.sha256
        not in {row.coverage_receipt_sha256 for row in manifest.coverage}
    ):
        raise ValueError("E4 local registry receipt lacks screen lineage")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    selection = signed_e4_screen_selection.verify(
        protocol_lock=protocol_lock,
        materialization=screen_materialization,
        coverage=screen_coverage,
        manifest=screen_evidence_manifest,
        execution_bindings=screen_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if selection.phase != "screen" or selection.factor_neighborhoods is None:
        raise ValueError("E4 local factorial requires the screen-phase winner")
    return _materialize_e4_winner_neighborhood_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_screen_receipt_sha256=screen_materialization.sha256,
        winner_decision_sha256=signed_e4_screen_selection.sha256,
        model=selection.model,
        lightcone_recipe_sha256=selection.lightcone_recipe_sha256,
        factor_neighborhoods=selection.factor_neighborhoods,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e4_profiler(
    *,
    registry_verification_receipt: object,
    signed_e4_final_selection: object,
    local_materialization: StageMaterializationReceipt,
    local_coverage: StageCoverageReceipt,
    local_evidence_manifest: object,
    local_execution_bindings: tuple[object, ...],
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize three isolated profilers from the local-factorial winner."""

    from lightcone_spec.experiments.e4_stage_authority import (
        E4StagedEvidenceManifest,
        SignedE4StageSelectionReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E4 profiler requires durable registry verification")
    if type(signed_e4_final_selection) is not SignedE4StageSelectionReceipt:
        raise TypeError("E4 profiler requires signed local-factorial selection")
    if type(local_materialization) is not StageMaterializationReceipt:
        raise TypeError("E4 profiler requires exact local materialization")
    if type(local_coverage) is not StageCoverageReceipt:
        raise TypeError("E4 profiler requires exact local coverage")
    if type(local_evidence_manifest) is not E4StagedEvidenceManifest:
        raise TypeError("E4 profiler requires path-bound local evidence")
    if type(local_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in local_execution_bindings
    ):
        raise TypeError("E4 profiler requires sealed local bindings")
    if gpu_hour_envelope is not None:
        raise TypeError(
            "E4 profiler materialization accepts no caller-authored GPU-hour envelope"
        )
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    if (
        signed_e4_final_selection
        not in registry_verification_receipt.cumulative_signed_e4_stage_selections
        or local_materialization.sha256
        not in {row.materialization_receipt_sha256 for row in manifest.materializations}
        or local_coverage.sha256
        not in {row.coverage_receipt_sha256 for row in manifest.coverage}
    ):
        raise ValueError("E4 profiler registry receipt lacks local-factorial lineage")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    selection = signed_e4_final_selection.verify(
        protocol_lock=protocol_lock,
        materialization=local_materialization,
        coverage=local_coverage,
        manifest=local_evidence_manifest,
        execution_bindings=local_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if selection.phase != "local" or selection.factor_neighborhoods is not None:
        raise ValueError("E4 profiler requires the local-factorial winner")
    selected_configuration_sha256 = content_sha256(selection.winner_configuration)
    return _materialize_e4_profiler_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_local_receipt_sha256=local_materialization.sha256,
        source_decision_sha256=signed_e4_final_selection.sha256,
        selected_configuration_sha256=selected_configuration_sha256,
        model=selection.model,
        lightcone_recipe_sha256=selection.lightcone_recipe_sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _require_registered_e4_profiler_completion(
    registry_verification_receipt: object,
    *,
    profiler_materialization: StageMaterializationReceipt,
    now_ns: int,
) -> object:
    """Reopen the registered profiler prefix and derive its typed completion."""

    from lightcone_spec.experiments.e4_stage_authority import (
        reduce_e4_profiler_completion_from_registry,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_prefix import (
        load_and_rebuild_formal_stage_prefix,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E4 profiler completion requires durable registry")
    rebuilt = tuple(
        load_and_rebuild_formal_stage_prefix(row.absolute_path, now_ns=now_ns)
        for row in registry_verification_receipt.cumulative_formal_stage_prefix_artifacts
    )
    matches = tuple(
        row
        for row in rebuilt
        if row.artifact.phase == "e4_profiler"
        and row.materialization == profiler_materialization
        and row.coverage.sha256
        in {
            signed.payload.sha256
            for signed in registry_verification_receipt.cumulative_signed_coverage
        }
    )
    if len(matches) != 1:
        raise ValueError("E3b requires one registered proof-derived profiler prefix")
    return reduce_e4_profiler_completion_from_registry(
        registry_verification_receipt=registry_verification_receipt,
        materialization=profiler_materialization,
        now_ns=now_ns,
    )


def materialize_e3b(
    *,
    registry_verification_receipt: object,
    signed_power_prefix: object,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    pilot_evidence_manifest: object,
    pilot_execution_bindings: tuple[object, ...],
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Append the powered E3b final prefix from proof-derived excluded pilots."""

    from lightcone_spec.experiments.downstream_stage_authority import (
        FormalDownstreamEvidenceManifest,
        SignedE3bPowerPrefixReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E3b requires durable registry verification")
    if type(signed_power_prefix) is not SignedE3bPowerPrefixReceipt:
        raise TypeError("E3b requires a signed proof-derived power prefix")
    if type(pilot_materialization) is not StageMaterializationReceipt:
        raise TypeError("E3b requires exact excluded-pilot materialization")
    if type(pilot_coverage) is not StageCoverageReceipt:
        raise TypeError("E3b requires exact excluded-pilot coverage")
    if type(pilot_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E3b requires path-bound excluded-pilot evidence")
    if type(pilot_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in pilot_execution_bindings
    ):
        raise TypeError("E3b requires sealed excluded-pilot execution bindings")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E3b materialization time must be non-negative")
    if gpu_hour_envelope is not None:
        raise TypeError("E3b materialization accepts no caller GPU-hour envelope")
    registry_manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    power = signed_power_prefix.verify(
        protocol_lock=protocol_lock,
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        manifest=pilot_evidence_manifest,
        execution_bindings=pilot_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    models = {cell.model for cell in pilot_materialization.cells}
    tts_recipes = {
        cell.recipe_sha256
        for cell in pilot_materialization.cells
        if cell.method_role in {"TTS", "L0-naive"}
    }
    lightcone_recipes = {
        cell.recipe_sha256
        for cell in pilot_materialization.cells
        if cell.method_role == "LightCone"
    }
    if (
        len(models) != 1
        or len(tts_recipes) != 1
        or None in tts_recipes
        or len(lightcone_recipes) != 1
        or None in lightcone_recipes
    ):
        raise ValueError("E3b pilot source has ambiguous model or recipe identity")
    tts_recipe = next(iter(tts_recipes))
    lightcone_recipe = next(iter(lightcone_recipes))
    assert tts_recipe is not None and lightcone_recipe is not None
    prior_rows = tuple(
        row.payload
        for row in registry_verification_receipt.cumulative_signed_materializations
        if row.payload.stage == "E4"
        and row.payload.materialization_rule
        == "three_profiler_only_rows_separate_from_headline"
    )
    if len(prior_rows) != 1 or prior_rows[0].sha256 not in {
        row.materialization_receipt_sha256 for row in registry_manifest.materializations
    }:
        raise ValueError("E3b full materialization lacks exact E4 DAG predecessor")
    _require_registered_e4_profiler_completion(
        registry_verification_receipt,
        profiler_materialization=prior_rows[0],
        now_ns=now_ns,
    )
    fixture = _materialize_e3b_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=prior_rows[0].sha256,
        source_decision_sha256=signed_power_prefix.sha256,
        model=next(iter(models)),
        frozen_tts_recipe_sha256=tts_recipe,
        lightcone_recipe_sha256=lightcone_recipe,
        final_blocks=power.selected_final_blocks,
        gpu_hours=GpuHourEstimate.unmeasured(),
        lineage_dimensions={
            "pilot_coverage_receipt_sha256": power.pilot_coverage_receipt_sha256,
            "pilot_materialization_receipt_sha256": (
                power.pilot_materialization_receipt_sha256
            ),
            "signed_power_prefix_sha256": signed_power_prefix.sha256,
        },
    )
    final_cells = _select_exact_final_prefix(
        fixture.cells,
        selected_final_prefix=power.selected_final_prefix,
        expected_cells_per_block=480,
    )
    return _receipt(
        stage="E3b",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(prior_rows[0].sha256,),
        source_decision_sha256=signed_power_prefix.sha256,
        materialization_rule=(
            "five_roles_x_8_contexts_x_3_regimes_x_2_loads_x_2_widths_final_only"
        ),
        cells=final_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e3b_excluded_pilots(
    *,
    registry_verification_receipt: object,
    signed_e4_final_selection: object,
    local_materialization: StageMaterializationReceipt,
    local_coverage: StageCoverageReceipt,
    local_evidence_manifest: object,
    local_execution_bindings: tuple[object, ...],
    profiler_materialization: StageMaterializationReceipt,
    profiler_coverage: StageCoverageReceipt,
    tts_calibration_authority: TtsCalibrationAuthority,
    signed_tts_calibration_seal: SignedTtsCalibrationSeal,
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize exactly four E3b tuning-only pilots before power sizing."""

    from lightcone_spec.experiments.e4_stage_authority import (
        E4StagedEvidenceManifest,
        SignedE4StageSelectionReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E3b pilots require durable registry verification")
    if type(signed_e4_final_selection) is not SignedE4StageSelectionReceipt:
        raise TypeError("E3b pilots require a signed E4 local selection")
    if type(local_materialization) is not StageMaterializationReceipt:
        raise TypeError("E3b pilots require exact E4 local materialization")
    if type(local_coverage) is not StageCoverageReceipt:
        raise TypeError("E3b pilots require exact E4 local coverage")
    if type(local_evidence_manifest) is not E4StagedEvidenceManifest:
        raise TypeError("E3b pilots require path-bound E4 local evidence")
    if type(local_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in local_execution_bindings
    ):
        raise TypeError("E3b pilots require sealed E4 local execution bindings")
    if type(profiler_materialization) is not StageMaterializationReceipt:
        raise TypeError("E3b pilots require exact E4 profiler materialization")
    if type(profiler_coverage) is not StageCoverageReceipt:
        raise TypeError("E3b pilots require exact E4 profiler coverage")
    if type(tts_calibration_authority) is not TtsCalibrationAuthority:
        raise TypeError("E3b pilots require exact TTS calibration authority")
    if type(signed_tts_calibration_seal) is not SignedTtsCalibrationSeal:
        raise TypeError("E3b pilots require the signed frozen TTS seal")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E3b pilot materialization time must be non-negative")
    if gpu_hour_envelope is not None:
        raise TypeError("E3b pilot materialization accepts no GPU-hour envelope")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    if (
        signed_e4_final_selection
        not in registry_verification_receipt.cumulative_signed_e4_stage_selections
        or profiler_materialization.sha256
        not in {row.materialization_receipt_sha256 for row in manifest.materializations}
        or profiler_coverage.sha256
        not in {row.coverage_receipt_sha256 for row in manifest.coverage}
        or profiler_materialization.stage != "E4"
        or profiler_materialization.materialization_rule
        != "three_profiler_only_rows_separate_from_headline"
        or profiler_materialization.source_decision_sha256
        != signed_e4_final_selection.sha256
    ):
        raise ValueError("E3b pilots lack exact covered E4 profiler lineage")
    profiler_coverage.validate_against(profiler_materialization)
    if any(row.status != "COMPLETE" for row in profiler_coverage.dispositions):
        raise ValueError("E3b pilots require all-COMPLETE E4 profiler coverage")
    profiler_completion = _require_registered_e4_profiler_completion(
        registry_verification_receipt,
        profiler_materialization=profiler_materialization,
        now_ns=now_ns,
    )
    if (
        profiler_completion.coverage_receipt_sha256 != profiler_coverage.sha256
        or profiler_completion.signed_local_selection_sha256
        != signed_e4_final_selection.sha256
    ):
        raise ValueError("E3b pilots use a foreign profiler completion")
    selection = signed_e4_final_selection.verify(
        protocol_lock=protocol_lock,
        materialization=local_materialization,
        coverage=local_coverage,
        manifest=local_evidence_manifest,
        execution_bindings=local_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if selection.phase != "local" or selection.factor_neighborhoods is not None:
        raise ValueError("E3b pilots require the final local-factorial winner")
    if (
        tts_calibration_authority.sha256
        != protocol_lock.tts_calibration_authority_sha256
        or signed_tts_calibration_seal
        not in registry_verification_receipt.cumulative_signed_tts_calibration_seals
    ):
        raise ValueError("E3b pilots lack ProtocolLock-bound TTS calibration")
    seal = signed_tts_calibration_seal.verify(
        authority=tts_calibration_authority,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if seal.protocol_lock_sha256 != protocol_lock.sha256:
        raise ValueError("E3b frozen TTS seal belongs to another ProtocolLock")
    full_fixture = _materialize_e3b_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=profiler_materialization.sha256,
        source_decision_sha256=signed_e4_final_selection.sha256,
        model=selection.model,
        frozen_tts_recipe_sha256=seal.selected_candidate_id,
        lightcone_recipe_sha256=selection.lightcone_recipe_sha256,
        final_blocks=12,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    pilot_cells = tuple(
        cell for cell in full_fixture.cells if dict(cell.dimensions)["block"] < 4
    )
    if len(pilot_cells) != 480 * 4:
        raise AssertionError(
            "E3b excluded pilot materialization must contain 1,920 rows"
        )
    signed_profiler_coverage = tuple(
        row
        for row in registry_verification_receipt.cumulative_signed_coverage
        if row.payload.sha256 == profiler_coverage.sha256
    )
    if len(signed_profiler_coverage) != 1:
        raise ValueError("E3b pilots require one signed E4 profiler coverage receipt")
    return _receipt(
        stage="E3b",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(
            signed_profiler_coverage[0].sha256,
            signed_tts_calibration_seal.sha256,
        ),
        source_decision_sha256=signed_e4_final_selection.sha256,
        materialization_rule="e3b_exact_480_rows_x_4_excluded_pilot_blocks",
        cells=pilot_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e1a(
    *,
    registry_verification_receipt: object,
    signed_e3b_confirmation: object,
    e3b_materialization: StageMaterializationReceipt,
    e3b_coverage: StageCoverageReceipt,
    e3b_evidence_manifest: object,
    e3b_execution_bindings: tuple[object, ...],
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize E1a only after proof-derived E3b primary confirmation."""

    from lightcone_spec.experiments.downstream_stage_authority import (
        FormalDownstreamEvidenceManifest,
        SignedE3bConfirmationReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E1a requires durable registry verification")
    if type(signed_e3b_confirmation) is not SignedE3bConfirmationReceipt:
        raise TypeError("E1a requires a signed proof-derived E3b confirmation")
    if type(e3b_materialization) is not StageMaterializationReceipt:
        raise TypeError("E1a requires exact E3b materialization")
    if type(e3b_coverage) is not StageCoverageReceipt:
        raise TypeError("E1a requires exact E3b coverage")
    if type(e3b_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E1a requires path-bound E3b evidence")
    if type(e3b_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in e3b_execution_bindings
    ):
        raise TypeError("E1a requires sealed E3b execution bindings")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E1a materialization time must be non-negative")
    if gpu_hour_envelope is not None:
        raise TypeError("E1a materialization accepts no caller GPU-hour envelope")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    if e3b_materialization.sha256 not in {
        row.materialization_receipt_sha256 for row in manifest.materializations
    } or e3b_coverage.sha256 not in {
        row.coverage_receipt_sha256 for row in manifest.coverage
    }:
        raise ValueError("E1a registry receipt lacks exact E3b lineage")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    confirmation = signed_e3b_confirmation.verify(
        protocol_lock=protocol_lock,
        materialization=e3b_materialization,
        coverage=e3b_coverage,
        manifest=e3b_evidence_manifest,
        execution_bindings=e3b_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if confirmation.status != "CONFIRMED":
        raise ValueError("E1a cannot start before both E3b primary contrasts pass")
    return _materialize_e1a_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=e3b_materialization.sha256,
        source_decision_sha256=signed_e3b_confirmation.sha256,
        model=confirmation.model,
        frozen_tts_recipe_sha256=confirmation.frozen_tts_recipe_sha256,
        lightcone_recipe_sha256=confirmation.lightcone_recipe_sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e5_excluded_pilots(
    *,
    registry_verification_receipt: object,
    signed_e1a_verification: object,
    e1a_materialization: StageMaterializationReceipt,
    e1a_coverage: StageCoverageReceipt,
    e1a_evidence_manifest: object,
    e1a_execution_bindings: tuple[object, ...],
    formal_runtime_authority_manifest: object,
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize the exact 1,800 E5 excluded headline-pilot cells."""

    from lightcone_spec.experiments.downstream_stage_authority import (
        E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
        FormalDownstreamEvidenceManifest,
        SignedE1aVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_protocol import (
        FormalRuntimeAuthorityManifest,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E5 pilots require durable registry verification")
    if type(signed_e1a_verification) is not SignedE1aVerificationReceipt:
        raise TypeError("E5 pilots require signed proof-derived E1a verification")
    if type(e1a_materialization) is not StageMaterializationReceipt:
        raise TypeError("E5 pilots require exact E1a materialization")
    if type(e1a_coverage) is not StageCoverageReceipt:
        raise TypeError("E5 pilots require exact E1a coverage")
    if type(e1a_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E5 pilots require path-bound E1a evidence")
    if type(e1a_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in e1a_execution_bindings
    ):
        raise TypeError("E5 pilots require sealed E1a execution bindings")
    if type(formal_runtime_authority_manifest) is not FormalRuntimeAuthorityManifest:
        raise TypeError("E5 pilots require the exact formal runtime authority manifest")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E5 pilot materialization time must be non-negative")
    if gpu_hour_envelope is not None:
        raise TypeError("E5 pilot materialization accepts no caller GPU-hour envelope")
    registry_manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    if e1a_materialization.sha256 not in {
        row.materialization_receipt_sha256 for row in registry_manifest.materializations
    } or e1a_coverage.sha256 not in {
        row.coverage_receipt_sha256 for row in registry_manifest.coverage
    }:
        raise ValueError("E5 pilots lack exact E1a registry lineage")
    if formal_runtime_authority_manifest.sha256 != (
        protocol_lock.formal_runtime_authority_manifest_sha256
    ):
        raise ValueError("E5 pilots use another formal runtime authority manifest")
    if (
        formal_runtime_authority_manifest.member(
            "e5_power_prefix_reducer"
        ).protocol_sha256
        != E5_POWER_AND_ANCHOR_PROTOCOL_SHA256
        or formal_runtime_authority_manifest.member(
            "e5_anchor_selection_reducer"
        ).protocol_sha256
        != E5_POWER_AND_ANCHOR_PROTOCOL_SHA256
    ):
        raise ValueError("E5 power/anchor reducer differs from ProtocolLock")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    e1a = signed_e1a_verification.verify(
        protocol_lock=protocol_lock,
        materialization=e1a_materialization,
        coverage=e1a_coverage,
        manifest=e1a_evidence_manifest,
        execution_bindings=e1a_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    cells = _e5_headline_cells(
        model=e1a.model,
        frozen_tts_recipe_sha256=e1a.frozen_tts_recipe_sha256,
        dflash_lightcone_recipe_sha256=e1a.source_lightcone_recipe_sha256,
        dspark_lightcone_recipe_sha256=e1a.selected_dspark_recipe_sha256,
        blocks=PILOT_BLOCK_COUNT,
        anchors=(),
        anchor_receipt_sha256=None,
        lineage_dimensions={
            "upstream_e1a_verification_sha256": signed_e1a_verification.sha256,
            "frozen_tts_recipe_sha256": e1a.frozen_tts_recipe_sha256,
            "dflash_lightcone_recipe_sha256": e1a.source_lightcone_recipe_sha256,
            "dspark_lightcone_recipe_sha256": e1a.selected_dspark_recipe_sha256,
        },
    )
    if len(cells) != 450 * PILOT_BLOCK_COUNT:
        raise AssertionError("E5 excluded pilots must contain exactly 1,800 rows")
    return _receipt(
        stage="E5",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(e1a_materialization.sha256,),
        source_decision_sha256=signed_e1a_verification.sha256,
        materialization_rule=("e5_exact_450_headline_rows_x_4_excluded_pilot_blocks"),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e5(
    *,
    registry_verification_receipt: object,
    signed_power_and_anchor_prefix: object,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    pilot_evidence_manifest: object,
    pilot_execution_bindings: tuple[object, ...],
    formal_runtime_authority_manifest: object,
    failure_diagnostic_authority: E5FailureDiagnosticAuthority,
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize exact ``450N + 264`` final rows from the E5 pilot reducer."""

    from lightcone_spec.experiments.downstream_stage_authority import (
        E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
        FormalDownstreamEvidenceManifest,
        SignedE5PowerAndAnchorReceipt,
    )
    from lightcone_spec.experiments.formal_protocol import (
        FormalRuntimeAuthorityManifest,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E5 requires durable registry verification")
    if type(signed_power_and_anchor_prefix) is not SignedE5PowerAndAnchorReceipt:
        raise TypeError("E5 requires a signed proof-derived power/anchor receipt")
    if type(pilot_materialization) is not StageMaterializationReceipt:
        raise TypeError("E5 requires exact pilot materialization")
    if type(pilot_coverage) is not StageCoverageReceipt:
        raise TypeError("E5 requires exact pilot coverage")
    if type(pilot_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E5 requires path-bound pilot evidence")
    if type(pilot_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in pilot_execution_bindings
    ):
        raise TypeError("E5 requires sealed pilot execution bindings")
    if type(formal_runtime_authority_manifest) is not FormalRuntimeAuthorityManifest:
        raise TypeError("E5 requires the exact formal runtime authority manifest")
    if type(failure_diagnostic_authority) is not E5FailureDiagnosticAuthority:
        raise TypeError("E5 requires the exact failure diagnostic authority")
    failure_diagnostic_authority.__post_init__()
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E5 materialization time must be non-negative")
    if gpu_hour_envelope is not None:
        raise TypeError("E5 materialization accepts no caller GPU-hour envelope")
    registry_manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    prior_rows = tuple(
        row.payload
        for row in registry_verification_receipt.cumulative_signed_materializations
        if row.payload.stage == "E1a"
    )
    if len(prior_rows) != 1 or prior_rows[0].sha256 not in {
        row.materialization_receipt_sha256 for row in registry_manifest.materializations
    }:
        raise ValueError("E5 full materialization lacks exact E1a DAG predecessor")
    if formal_runtime_authority_manifest.sha256 != (
        protocol_lock.formal_runtime_authority_manifest_sha256
    ):
        raise ValueError("E5 uses another formal runtime authority manifest")
    if (
        formal_runtime_authority_manifest.member(
            "e5_power_prefix_reducer"
        ).protocol_sha256
        != E5_POWER_AND_ANCHOR_PROTOCOL_SHA256
        or formal_runtime_authority_manifest.member(
            "e5_anchor_selection_reducer"
        ).protocol_sha256
        != E5_POWER_AND_ANCHOR_PROTOCOL_SHA256
        or formal_runtime_authority_manifest.member(
            "e5_failure_reducer"
        ).protocol_sha256
        != failure_diagnostic_authority.protocol_sha256
    ):
        raise ValueError("E5 reducer identities differ from ProtocolLock")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    decision = signed_power_and_anchor_prefix.verify(
        protocol_lock=protocol_lock,
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        manifest=pilot_evidence_manifest,
        execution_bindings=pilot_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    blocks = PILOT_BLOCK_COUNT + decision.selected_final_blocks
    lineage = {
        "upstream_e1a_verification_sha256": (decision.upstream_e1a_verification_sha256),
        "frozen_tts_recipe_sha256": decision.frozen_tts_recipe_sha256,
        "dflash_lightcone_recipe_sha256": (decision.dflash_lightcone_recipe_sha256),
        "dspark_lightcone_recipe_sha256": decision.dspark_lightcone_recipe_sha256,
        "pilot_coverage_receipt_sha256": decision.pilot_coverage_receipt_sha256,
        "pilot_materialization_receipt_sha256": (
            decision.pilot_materialization_receipt_sha256
        ),
        "signed_power_and_anchor_prefix_sha256": (
            signed_power_and_anchor_prefix.sha256
        ),
    }
    headline = _e5_headline_cells(
        model=decision.model,
        frozen_tts_recipe_sha256=decision.frozen_tts_recipe_sha256,
        dflash_lightcone_recipe_sha256=decision.dflash_lightcone_recipe_sha256,
        dspark_lightcone_recipe_sha256=decision.dspark_lightcone_recipe_sha256,
        blocks=blocks,
        anchors=decision.p99_anchors,
        anchor_receipt_sha256=signed_power_and_anchor_prefix.sha256,
        lineage_dimensions=lineage,
    )
    headline = _select_exact_final_prefix(
        headline,
        selected_final_prefix=decision.selected_final_prefix,
        expected_cells_per_block=450,
    )
    failures = tuple(
        _cell(
            stage="E5",
            method_role="LightCone",
            model=decision.model,
            backend=member.backend,
            task="deterministic_failure_injection",
            publication_policy="diagnostic_only",
            recipe_sha256=(
                decision.dflash_lightcone_recipe_sha256
                if member.backend == "DFLASH"
                else decision.dspark_lightcone_recipe_sha256
            ),
            dimensions={
                "diagnostic_only": "true",
                "failure": member.failure,
                "failure_authority_sha256": failure_diagnostic_authority.sha256,
                "failure_member_id": member.member_id,
                "topology": member.topology,
                "cohort_count": member.cohort_count,
                **lineage,
            },
        )
        for member in failure_diagnostic_authority.members
    )
    if len(failures) != 264:
        raise AssertionError("E5 failure diagnostics must contain exactly 264 cells")
    cells = headline + failures
    if len(cells) != 450 * decision.selected_final_blocks + 264:
        raise AssertionError(
            "E5 total differs from 450 final rows per block plus 264 diagnostics"
        )
    return _receipt(
        stage="E5",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(prior_rows[0].sha256,),
        source_decision_sha256=signed_power_and_anchor_prefix.sha256,
        materialization_rule=(
            "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
        ),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e6_excluded_pilots(
    *,
    registry_verification_receipt: object,
    signed_e5_confirmation: object,
    e5_materialization: StageMaterializationReceipt,
    e5_coverage: StageCoverageReceipt,
    e5_headline_evidence_manifest: object,
    e5_headline_execution_bindings: tuple[object, ...],
    e5_failure_evidence_manifest: object,
    e5_failure_execution_bindings: tuple[object, ...],
    signed_model_compatibility: object,
    compatibility_sources: tuple[object, ...],
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Build the signed E6 tuning authority outside the main registry.

    The two immutable NEXTN fit checks and four excluded pilot blocks are
    replay inputs for power sizing.  This receipt is deliberately rejected by
    the main formal registry and can only be consumed by the typed E6 power
    reducer.
    """

    from lightcone_spec.experiments.downstream_stage_authority import (
        E5FailureEvidenceManifest,
        FormalDownstreamEvidenceManifest,
        SignedE5ConfirmationReceipt,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        E6NextnModelAuthorityInput,
        SignedE6ModelCompatibilityReceipt,
    )
    from lightcone_spec.experiments.formal_failure_execution import (
        VerifiedFormalFailureExecutionBinding,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E6 pilots require durable registry verification")
    if type(signed_e5_confirmation) is not SignedE5ConfirmationReceipt:
        raise TypeError("E6 pilots require a signed E5 confirmation")
    if type(e5_materialization) is not StageMaterializationReceipt:
        raise TypeError("E6 pilots require exact E5 materialization")
    if type(e5_coverage) is not StageCoverageReceipt:
        raise TypeError("E6 pilots require exact E5 coverage")
    if type(e5_headline_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E6 pilots require exact E5 headline evidence")
    if type(e5_headline_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in e5_headline_execution_bindings
    ):
        raise TypeError("E6 pilots require sealed E5 headline bindings")
    if type(e5_failure_evidence_manifest) is not E5FailureEvidenceManifest:
        raise TypeError("E6 pilots require exact E5 failure evidence")
    if type(e5_failure_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalFailureExecutionBinding
        for row in e5_failure_execution_bindings
    ):
        raise TypeError("E6 pilots require sealed E5 failure bindings")
    if type(signed_model_compatibility) is not SignedE6ModelCompatibilityReceipt:
        raise TypeError("E6 pilots require signed two-model compatibility")
    if type(compatibility_sources) is not tuple or any(
        type(row) is not E6NextnModelAuthorityInput for row in compatibility_sources
    ):
        raise TypeError("E6 pilots require exact NEXTN compatibility sources")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E6 pilot materialization time must be positive")
    if gpu_hour_envelope is not None:
        raise TypeError("E6 pilot materialization accepts no GPU-hour envelope")

    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    if e5_materialization.sha256 not in {
        row.materialization_receipt_sha256 for row in manifest.materializations
    } or e5_coverage.sha256 not in {
        row.coverage_receipt_sha256 for row in manifest.coverage
    }:
        raise ValueError("E6 pilots lack exact covered E5 registry lineage")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    e5 = signed_e5_confirmation.verify(
        protocol_lock=protocol_lock,
        materialization=e5_materialization,
        coverage=e5_coverage,
        headline_manifest=e5_headline_evidence_manifest,
        headline_execution_bindings=e5_headline_execution_bindings,  # type: ignore[arg-type]
        failure_manifest=e5_failure_evidence_manifest,
        failure_execution_bindings=e5_failure_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if e5.status != "CONFIRMED":
        raise ValueError("E6 pilots cannot start before E5 confirmation")
    compatibility = signed_model_compatibility.verify(
        protocol_lock=protocol_lock,
        sources=compatibility_sources,  # type: ignore[arg-type]
        expected_inventory_sha256=manifest.inventory_sha256,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    cells = _e6_cells_from_verified_sources(
        signed_e5_confirmation_sha256=signed_e5_confirmation.sha256,
        signed_model_compatibility_sha256=signed_model_compatibility.sha256,
        model_compatibility=compatibility,
        frozen_tts_recipe_sha256=e5.frozen_tts_recipe_sha256,
        lightcone_recipe_sha256=e5.dflash_lightcone_recipe_sha256,
        block_indices=tuple(range(PILOT_BLOCK_COUNT)),
    )
    return _receipt(
        stage="E6",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(e5_materialization.sha256,),
        source_decision_sha256=content_sha256(
            {
                "signed_e5_confirmation_sha256": signed_e5_confirmation.sha256,
                "signed_e6_model_compatibility_sha256": (
                    signed_model_compatibility.sha256
                ),
            }
        ),
        materialization_rule=(
            "e6_exact_two_model_preflights_plus_60_rows_x_4_excluded_pilot_blocks"
        ),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e6(
    *,
    registry_verification_receipt: object,
    signed_e5_confirmation: object,
    signed_model_compatibility: object,
    compatibility_sources: tuple[object, ...],
    signed_power_prefix: object,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    pilot_evidence_manifest: object,
    pilot_execution_bindings: tuple[object, ...],
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize only the powered E6 final prefix in the main registry."""

    from lightcone_spec.experiments.downstream_stage_authority import (
        FormalDownstreamEvidenceManifest,
        SignedE5ConfirmationReceipt,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        E6NextnModelAuthorityInput,
        SignedE6ModelCompatibilityReceipt,
        SignedE6PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        VerifiedFormalServingExecutionBinding,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E6 requires durable registry verification")
    if type(signed_e5_confirmation) is not SignedE5ConfirmationReceipt:
        raise TypeError("E6 requires a signed E5 confirmation")
    if type(signed_model_compatibility) is not SignedE6ModelCompatibilityReceipt:
        raise TypeError("E6 requires signed two-model compatibility")
    if type(compatibility_sources) is not tuple or any(
        type(row) is not E6NextnModelAuthorityInput for row in compatibility_sources
    ):
        raise TypeError("E6 requires exact NEXTN compatibility sources")
    if type(signed_power_prefix) is not SignedE6PowerPrefixReceipt:
        raise TypeError("E6 requires a signed proof-derived power prefix")
    if type(pilot_materialization) is not StageMaterializationReceipt:
        raise TypeError("E6 requires exact excluded-pilot materialization")
    if type(pilot_coverage) is not StageCoverageReceipt:
        raise TypeError("E6 requires exact excluded-pilot coverage")
    if type(pilot_evidence_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E6 requires path-bound excluded-pilot evidence")
    if type(pilot_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in pilot_execution_bindings
    ):
        raise TypeError("E6 requires sealed excluded-pilot execution bindings")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E6 materialization time must be positive")
    if gpu_hour_envelope is not None:
        raise TypeError("E6 materialization accepts no caller GPU-hour envelope")

    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    prior_rows = tuple(
        row.payload
        for row in registry_verification_receipt.cumulative_signed_materializations
        if row.payload.stage == "E5"
    )
    if (
        len(prior_rows) != 1
        or prior_rows[0].sha256
        not in {row.materialization_receipt_sha256 for row in manifest.materializations}
        or signed_e5_confirmation.payload.materialization_receipt_sha256
        != prior_rows[0].sha256
        or signed_e5_confirmation.payload.status != "CONFIRMED"
    ):
        raise ValueError("E6 full materialization lacks exact confirmed E5 predecessor")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    power = signed_power_prefix.verify(
        protocol_lock=protocol_lock,
        signed_model_compatibility=signed_model_compatibility,
        compatibility_sources=compatibility_sources,  # type: ignore[arg-type]
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        manifest=pilot_evidence_manifest,
        execution_bindings=pilot_execution_bindings,  # type: ignore[arg-type]
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if (
        power.upstream_e5_confirmation_sha256 != signed_e5_confirmation.sha256
        or power.signed_model_compatibility_sha256 != signed_model_compatibility.sha256
    ):
        raise ValueError("E6 power prefix changes its typed source decisions")
    compatibility = signed_model_compatibility.payload
    cells = _e6_cells_from_verified_sources(
        signed_e5_confirmation_sha256=signed_e5_confirmation.sha256,
        signed_model_compatibility_sha256=signed_model_compatibility.sha256,
        model_compatibility=compatibility,
        frozen_tts_recipe_sha256=power.frozen_tts_recipe_sha256,
        lightcone_recipe_sha256=power.lightcone_recipe_sha256,
        block_indices=power.selected_final_prefix,
        power_prefix_source_sha256=signed_power_prefix.sha256,
        pilot_materialization_receipt_sha256=(
            power.pilot_materialization_receipt_sha256
        ),
        pilot_coverage_receipt_sha256=power.pilot_coverage_receipt_sha256,
    )
    return _receipt(
        stage="E6",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(prior_rows[0].sha256,),
        source_decision_sha256=signed_power_prefix.sha256,
        materialization_rule=(
            "60_final_rows_per_block_reusing_global_model_preflights"
        ),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e0_onlinespec_tuning(
    *,
    registry_verification_receipt: object,
    signed_e6_confirmation: object,
    e6_confirmation_proof_bundle: object,
    signed_compatibility_receipt: object,
    onlinespec_source_authority: object,
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize the exact registered OnlineSPEC grid outside the main DAG."""

    from lightcone_spec.experiments.e0_stage_authority import (
        E0_ONLINESPEC_TUNING_RULE,
        E0OnlineSpecSourceAuthority,
        E6ConfirmationProofBundle,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        SignedE6ConfirmationReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E0 tuning requires durable registry verification")
    if type(signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
        raise TypeError("E0 tuning requires a signed E6 confirmation")
    if type(e6_confirmation_proof_bundle) is not E6ConfirmationProofBundle:
        raise TypeError("E0 tuning requires exact E6 proof inputs")
    if type(signed_compatibility_receipt) is not SignedE0CompatibilityReceipt:
        raise TypeError("E0 tuning requires signed 108-row compatibility")
    if type(onlinespec_source_authority) is not E0OnlineSpecSourceAuthority:
        raise TypeError("E0 tuning requires source-owned OnlineSPEC authority")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E0 tuning materialization time must be positive")
    if gpu_hour_envelope is not None:
        raise TypeError("E0 tuning materialization accepts no GPU-hour envelope")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    e6 = e6_confirmation_proof_bundle.verify(
        signed_e6_confirmation,
        protocol_lock=protocol_lock,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if (
        e6.status != "CONFIRMED"
        or e6.materialization_receipt_sha256
        not in {row.materialization_receipt_sha256 for row in manifest.materializations}
        or e6.coverage_receipt_sha256
        not in {row.coverage_receipt_sha256 for row in manifest.coverage}
    ):
        raise ValueError("E0 tuning lacks exact confirmed E6 registry lineage")
    compatibility = signed_compatibility_receipt.verify(
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if (
        compatibility.protocol_lock_sha256 != protocol_lock.sha256
        or compatibility.upstream_e6_receipt_sha256 != e6.materialization_receipt_sha256
    ):
        raise ValueError("E0 compatibility differs from confirmed E6")
    onlinespec_source_authority.revalidate()
    cells = _e0_tuning_cells_from_verified_sources(
        compatibility=compatibility,
        signed_compatibility_sha256=signed_compatibility_receipt.sha256,
        signed_e6_confirmation_sha256=signed_e6_confirmation.sha256,
        onlinespec_source_authority_sha256=onlinespec_source_authority.sha256,
        frozen_tts_recipe_sha256=e6.frozen_tts_recipe_sha256,
    )
    source = content_sha256(
        {
            "signed_e6_confirmation_sha256": signed_e6_confirmation.sha256,
            "signed_e0_compatibility_sha256": signed_compatibility_receipt.sha256,
            "e0_onlinespec_source_authority_sha256": (
                onlinespec_source_authority.sha256
            ),
        }
    )
    return _receipt(
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(e6.materialization_receipt_sha256,),
        source_decision_sha256=source,
        materialization_rule=E0_ONLINESPEC_TUNING_RULE,
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e0_all_na_from_signed_compatibility(
    *,
    registry_verification_receipt: object,
    signed_e6_confirmation: object,
    e6_confirmation_proof_bundle: object,
    signed_compatibility_receipt: object,
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize the legitimate zero-cell E0 branch from 108 signed N/A rows.

    This branch deliberately skips OnlineSPEC tuning, excluded pilots, and power
    sizing.  It is available only after the E6 confirmation is deep-replayed and
    every member of the closed compatibility universe is signed ``N/A``.
    """

    from lightcone_spec.experiments.e0_stage_authority import (
        E6ConfirmationProofBundle,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        SignedE6ConfirmationReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E0 all-N/A requires durable registry verification")
    if type(signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
        raise TypeError("E0 all-N/A requires a signed E6 confirmation")
    if type(e6_confirmation_proof_bundle) is not E6ConfirmationProofBundle:
        raise TypeError("E0 all-N/A requires exact E6 proof inputs")
    if type(signed_compatibility_receipt) is not SignedE0CompatibilityReceipt:
        raise TypeError("E0 all-N/A requires signed 108-row compatibility")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E0 all-N/A materialization time must be positive")
    if gpu_hour_envelope is not None:
        raise TypeError("E0 all-N/A accepts no caller GPU-hour envelope")

    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    e6 = e6_confirmation_proof_bundle.verify(
        signed_e6_confirmation,
        protocol_lock=protocol_lock,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if (
        e6.status != "CONFIRMED"
        or e6.materialization_receipt_sha256
        not in {row.materialization_receipt_sha256 for row in manifest.materializations}
        or e6.coverage_receipt_sha256
        not in {row.coverage_receipt_sha256 for row in manifest.coverage}
    ):
        raise ValueError("E0 all-N/A lacks exact confirmed E6 registry lineage")
    compatibility = signed_compatibility_receipt.verify(
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if (
        compatibility.protocol_lock_sha256 != protocol_lock.sha256
        or compatibility.upstream_e6_receipt_sha256 != e6.materialization_receipt_sha256
        or compatibility.valid_count != 0
        or any(row.disposition != "N/A" for row in compatibility.decisions)
    ):
        raise ValueError(
            "E0 all-N/A requires every signed compatibility decision to be N/A"
        )
    source = content_sha256(
        {
            "protocol_sha256": E0_ALL_NA_MATERIALIZATION_PROTOCOL_SHA256,
            "signed_e6_confirmation_sha256": signed_e6_confirmation.sha256,
            "signed_e0_compatibility_sha256": signed_compatibility_receipt.sha256,
        }
    )
    return _receipt(
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(e6.materialization_receipt_sha256,),
        source_decision_sha256=source,
        materialization_rule=E0_ALL_NA_MATERIALIZATION_RULE,
        cells=(),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e0_excluded_pilots(
    *,
    registry_verification_receipt: object,
    signed_e6_confirmation: object,
    e6_confirmation_proof_bundle: object,
    signed_compatibility_receipt: object,
    signed_onlinespec_tuning_seals: tuple[object, ...],
    onlinespec_source_authority: object,
    tuning_proof_set: object,
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize exactly four excluded ``16V`` E0 pilot blocks."""

    from lightcone_spec.experiments.e0_stage_authority import (
        E0_EXCLUDED_PILOT_RULE,
        E0OnlineSpecSourceAuthority,
        E0OnlineSpecTuningProofSet,
        E6ConfirmationProofBundle,
        SignedE0OnlineSpecTuningSeal,
        _verify_tuning_seal_set,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        SignedE6ConfirmationReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E0 pilots require durable registry verification")
    if type(signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
        raise TypeError("E0 pilots require a signed E6 confirmation")
    if type(e6_confirmation_proof_bundle) is not E6ConfirmationProofBundle:
        raise TypeError("E0 pilots require exact E6 proof inputs")
    if type(signed_compatibility_receipt) is not SignedE0CompatibilityReceipt:
        raise TypeError("E0 pilots require signed 108-row compatibility")
    if type(signed_onlinespec_tuning_seals) is not tuple or any(
        type(row) is not SignedE0OnlineSpecTuningSeal
        for row in signed_onlinespec_tuning_seals
    ):
        raise TypeError("E0 pilots require exact signed OnlineSPEC tuning seals")
    if type(onlinespec_source_authority) is not E0OnlineSpecSourceAuthority:
        raise TypeError("E0 pilots require source-owned OnlineSPEC authority")
    if type(tuning_proof_set) is not E0OnlineSpecTuningProofSet:
        raise TypeError("E0 pilots require exact tuning proofs")
    if tuning_proof_set.e6_confirmation_proof_bundle != e6_confirmation_proof_bundle:
        raise ValueError("E0 tuning proof set reopens another E6 authority")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E0 pilot materialization time must be positive")
    if gpu_hour_envelope is not None:
        raise TypeError("E0 pilot materialization accepts no GPU-hour envelope")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    e6 = e6_confirmation_proof_bundle.verify(
        signed_e6_confirmation,
        protocol_lock=protocol_lock,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if e6.materialization_receipt_sha256 not in {
        row.materialization_receipt_sha256 for row in manifest.materializations
    }:
        raise ValueError("E0 pilots lack the exact E6 DAG predecessor")
    compatibility = signed_compatibility_receipt.verify(
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if compatibility.upstream_e6_receipt_sha256 != e6.materialization_receipt_sha256:
        raise ValueError("E0 pilot compatibility names another E6 receipt")
    _verify_tuning_seal_set(
        protocol_lock=protocol_lock,
        signed_e6_confirmation=signed_e6_confirmation,
        signed_compatibility=signed_compatibility_receipt,
        signed_tuning_seals=signed_onlinespec_tuning_seals,
        source_authority=onlinespec_source_authority,
        tuning_proof_set=tuning_proof_set,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    cells = _e0_cells_from_verified_sources(
        compatibility=compatibility,
        signed_compatibility_sha256=signed_compatibility_receipt.sha256,
        signed_e6_confirmation_sha256=signed_e6_confirmation.sha256,
        signed_tuning_seals=signed_onlinespec_tuning_seals,
        frozen_tts_recipe_sha256=e6.frozen_tts_recipe_sha256,
        lightcone_recipe_sha256=e6.lightcone_recipe_sha256,
        block_indices=tuple(range(PILOT_BLOCK_COUNT)),
    )
    source = content_sha256(
        {
            "signed_e6_confirmation_sha256": signed_e6_confirmation.sha256,
            "signed_e0_compatibility_sha256": signed_compatibility_receipt.sha256,
            "signed_e0_tuning_seal_sha256s": tuple(
                sorted(row.sha256 for row in signed_onlinespec_tuning_seals)
            ),
        }
    )
    return _receipt(
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(e6.materialization_receipt_sha256,),
        source_decision_sha256=source,
        materialization_rule=E0_EXCLUDED_PILOT_RULE,
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def materialize_e0_from_signed_compatibility(
    *,
    registry_verification_receipt: object,
    signed_e6_confirmation: object,
    e6_confirmation_proof_bundle: object,
    signed_compatibility_receipt: object,
    signed_onlinespec_tuning_seals: tuple[object, ...],
    onlinespec_source_authority: object,
    tuning_proof_set: object,
    signed_power_prefix: object,
    pilot_proof_set: object,
    now_ns: int,
    gpu_hour_envelope: None = None,
) -> StageMaterializationReceipt:
    """Materialize only the proof-derived ``16VN`` final E0 prefix."""

    from lightcone_spec.experiments.e0_stage_authority import (
        E0_FINAL_MATERIALIZATION_RULE,
        E0OnlineSpecSourceAuthority,
        E0OnlineSpecTuningProofSet,
        E6ConfirmationProofBundle,
        SignedE0OnlineSpecTuningSeal,
        SignedE0PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        SignedE6ConfirmationReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E0 requires durable registry verification")
    if type(signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
        raise TypeError("E0 requires a signed E6 confirmation")
    if type(e6_confirmation_proof_bundle) is not E6ConfirmationProofBundle:
        raise TypeError("E0 requires exact E6 proof inputs")
    if type(signed_compatibility_receipt) is not SignedE0CompatibilityReceipt:
        raise TypeError("E0 requires signed 108-row compatibility")
    if type(signed_onlinespec_tuning_seals) is not tuple or any(
        type(row) is not SignedE0OnlineSpecTuningSeal
        for row in signed_onlinespec_tuning_seals
    ):
        raise TypeError("E0 requires exact signed OnlineSPEC tuning seals")
    if type(onlinespec_source_authority) is not E0OnlineSpecSourceAuthority:
        raise TypeError("E0 requires source-owned OnlineSPEC authority")
    if type(tuning_proof_set) is not E0OnlineSpecTuningProofSet:
        raise TypeError("E0 requires exact tuning proofs")
    if type(signed_power_prefix) is not SignedE0PowerPrefixReceipt:
        raise TypeError("E0 requires a signed proof-derived power prefix")
    if type(pilot_proof_set) is not E0OnlineSpecTuningProofSet:
        raise TypeError("E0 requires exact excluded-pilot proofs")
    if (
        tuning_proof_set.e6_confirmation_proof_bundle != e6_confirmation_proof_bundle
        or pilot_proof_set.e6_confirmation_proof_bundle != e6_confirmation_proof_bundle
    ):
        raise ValueError("E0 proof sets reopen another E6 authority")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E0 materialization time must be positive")
    if gpu_hour_envelope is not None:
        raise TypeError("E0 materialization accepts no caller GPU-hour envelope")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    e6 = e6_confirmation_proof_bundle.verify(
        signed_e6_confirmation,
        protocol_lock=protocol_lock,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    prior_rows = tuple(
        row.payload
        for row in registry_verification_receipt.cumulative_signed_materializations
        if row.payload.stage == "E6"
    )
    if (
        len(prior_rows) != 1
        or prior_rows[0].sha256 != e6.materialization_receipt_sha256
        or prior_rows[0].sha256
        not in {row.materialization_receipt_sha256 for row in manifest.materializations}
        or e6.status != "CONFIRMED"
    ):
        raise ValueError("E0 lacks the exact confirmed E6 DAG predecessor")
    compatibility = signed_compatibility_receipt.verify(
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if compatibility.upstream_e6_receipt_sha256 != prior_rows[0].sha256:
        raise ValueError("E0 compatibility names another E6 materialization")
    power = signed_power_prefix.verify(
        protocol_lock=protocol_lock,
        signed_e6_confirmation=signed_e6_confirmation,
        signed_compatibility=signed_compatibility_receipt,
        signed_tuning_seals=signed_onlinespec_tuning_seals,
        source_authority=onlinespec_source_authority,
        tuning_proof_set=tuning_proof_set,
        pilot_proof_set=pilot_proof_set,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    cells = _e0_cells_from_verified_sources(
        compatibility=compatibility,
        signed_compatibility_sha256=signed_compatibility_receipt.sha256,
        signed_e6_confirmation_sha256=signed_e6_confirmation.sha256,
        signed_tuning_seals=signed_onlinespec_tuning_seals,
        frozen_tts_recipe_sha256=e6.frozen_tts_recipe_sha256,
        lightcone_recipe_sha256=e6.lightcone_recipe_sha256,
        block_indices=power.selected_final_prefix,
        power_prefix_source_sha256=signed_power_prefix.sha256,
        pilot_materialization_receipt_sha256=(
            power.pilot_materialization_receipt_sha256
        ),
        pilot_coverage_receipt_sha256=power.pilot_coverage_receipt_sha256,
    )
    return _receipt(
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(prior_rows[0].sha256,),
        source_decision_sha256=signed_power_prefix.sha256,
        materialization_rule=E0_FINAL_MATERIALIZATION_RULE,
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


@dataclass(frozen=True)
class StageCellDisposition:
    stage: str
    cell_id: str
    status: Literal["COMPLETE", "BLOCKED", "N/A"]
    reason_code: str
    terminal_receipt_sha256: str | None

    def __post_init__(self) -> None:
        if self.stage not in FORMAL_STAGE_DAG:
            raise ValueError("stage disposition names an unknown stage")
        _require_sha256("stage disposition cell ID", self.cell_id)
        _require_text("stage disposition reason", self.reason_code)
        if self.status == "COMPLETE":
            _require_sha256("stage terminal receipt", self.terminal_receipt_sha256)
        elif self.terminal_receipt_sha256 is not None:
            raise ValueError("non-complete disposition cannot carry terminal evidence")
        reject_banned_model_identity(self)


@dataclass(frozen=True)
class StageCoverageReceipt:
    schema_version: int
    stage: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    dispositions: tuple[StageCellDisposition, ...]
    tts_l0_candidate_state_coverages: tuple[TtsL0CandidateStateCoverage, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("only StageCoverageReceipt schema 2 is supported")
        if self.stage not in FORMAL_STAGE_DAG:
            raise ValueError("coverage receipt names an unknown stage")
        _require_sha256("coverage protocol lock", self.protocol_lock_sha256)
        _require_sha256("coverage materialization", self.materialization_receipt_sha256)
        ids = tuple(row.cell_id for row in self.dispositions)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("coverage dispositions must be canonical and unique")
        if any(row.stage != self.stage for row in self.dispositions):
            raise ValueError("coverage contains a disposition from another stage")
        candidate_coverages = self.tts_l0_candidate_state_coverages
        if (
            type(candidate_coverages) is not tuple
            or any(
                type(row) is not TtsL0CandidateStateCoverage
                for row in candidate_coverages
            )
            or tuple(row.pair_id for row in candidate_coverages)
            != tuple(sorted({row.pair_id for row in candidate_coverages}))
        ):
            raise ValueError(
                "TTS/L0 candidate-state pair coverages must be canonical and unique"
            )
        if self.stage in TTS_L0_CANDIDATE_STATE_COVERAGE_STAGES:
            e0_all_na_empty = (
                self.stage == "E0" and not self.dispositions and not candidate_coverages
            )
            if not candidate_coverages and not e0_all_na_empty:
                raise ValueError(
                    "stage coverage requires complete TTS/L0 candidate-state evidence"
                )
            for candidate_coverage in candidate_coverages:
                candidate_coverage.validate_identity(
                    stage=self.stage,
                    protocol_lock_sha256=self.protocol_lock_sha256,
                    materialization_receipt_sha256=(
                        self.materialization_receipt_sha256
                    ),
                )
        elif candidate_coverages:
            raise ValueError(
                "stage without TTS/L0 publication cannot attach candidate evidence"
            )
        run_ids = tuple(
            next(iter({row.run_id for row in observations}))
            for coverage in candidate_coverages
            for observations in (
                coverage.tts_observations,
                coverage.l0_naive_observations,
            )
        )
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("stage candidate-state coverage reuses a run identity")
        proposal_digests = tuple(
            observation.proposal_evidence_sha256
            for coverage in candidate_coverages
            for observation in coverage.tts_observations
        )
        if len(proposal_digests) != len(set(proposal_digests)):
            raise ValueError(
                "stage candidate-state coverage reuses proposal evidence across "
                "matched pair source rounds"
            )
        terminal_digests = tuple(
            digest
            for coverage in candidate_coverages
            for digest in (
                coverage.terminal_pairs[0].tts_terminal_receipt_sha256,
                coverage.terminal_pairs[0].l0_naive_terminal_receipt_sha256,
            )
        )
        if len(terminal_digests) != len(set(terminal_digests)):
            raise ValueError("stage candidate-state coverage reuses a terminal receipt")
        reject_banned_model_identity(self)

    def validate_against(self, materialization: StageMaterializationReceipt) -> None:
        if type(materialization) is not StageMaterializationReceipt:
            raise TypeError("coverage requires an exact materialization receipt")
        if (
            self.stage != materialization.stage
            or self.protocol_lock_sha256 != materialization.protocol_lock_sha256
            or self.materialization_receipt_sha256 != materialization.sha256
        ):
            raise ValueError("coverage identity differs from materialization")
        expected = {cell.cell_id for cell in materialization.cells}
        observed = {row.cell_id for row in self.dispositions}
        if observed != expected or len(self.dispositions) != len(expected):
            raise ValueError(
                "coverage must disposition every and only materialized cell"
            )
        candidate_coverages = self.tts_l0_candidate_state_coverages
        if self.stage == "E0" and not expected:
            if (
                materialization.materialization_rule != E0_ALL_NA_MATERIALIZATION_RULE
                or materialization.expected_cell_count != 0
                or self.dispositions
                or candidate_coverages
            ):
                raise ValueError(
                    "empty E0 coverage is reserved for the signed all-N/A branch"
                )
            return
        if self.stage == "preflight":
            exactness_cells = tuple(
                cell.cell_id
                for cell in materialization.cells
                if cell.task == "exactness_memory_telemetry_preflight"
            )
            if (
                len(candidate_coverages) != 1
                or len(exactness_cells) != 1
                or candidate_coverages[0].scope != "preflight_exactness_qualification"
                or candidate_coverages[0].qualification_cell_id != exactness_cells[0]
            ):
                raise ValueError(
                    "preflight candidate evidence does not bind the exactness cell"
                )
            return

        tts_by_pair: dict[str, MaterializedCell] = {}
        l0_by_pair: dict[str, MaterializedCell] = {}
        for cell in materialization.cells:
            if cell.method_role not in {"TTS", "L0-naive"}:
                continue
            pair_id = dict(cell.dimensions).get("tts_l0_pair_id")
            _require_sha256("materialized TTS/L0 pair ID", pair_id)
            destination = tts_by_pair if cell.method_role == "TTS" else l0_by_pair
            if pair_id in destination:
                raise ValueError("materialization repeats a TTS/L0 matched pair role")
            destination[pair_id] = cell
        if set(tts_by_pair) != set(l0_by_pair):
            raise ValueError("materialization has an unmatched TTS/L0 cell")
        if not tts_by_pair:
            if candidate_coverages:
                raise ValueError(
                    "stage without materialized TTS/L0 pairs cannot attach pair evidence"
                )
            return
        coverage_by_pair = {row.pair_id: row for row in candidate_coverages}
        if set(coverage_by_pair) != set(tts_by_pair):
            raise ValueError(
                "candidate-state evidence does not cover every materialized TTS/L0 pair"
            )
        for pair_id, candidate_coverage in coverage_by_pair.items():
            if (
                candidate_coverage.scope != "materialized_pair"
                or candidate_coverage.tts_cell_id != tts_by_pair[pair_id].cell_id
                or candidate_coverage.l0_naive_cell_id != l0_by_pair[pair_id].cell_id
            ):
                raise ValueError(
                    "candidate-state evidence differs from its materialized TTS/L0 pair"
                )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedStageCoverageReceipt:
    payload: StageCoverageReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        materialization: StageMaterializationReceipt,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> StageCoverageReceipt:
        if type(self.payload) is not StageCoverageReceipt:
            raise TypeError("signed coverage payload has the wrong type")
        self.payload.validate_against(materialization)
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


def validate_formal_registry_no_banned_models(value: object) -> None:
    """Fail-closed validator for registry, activation, and coverage payloads."""

    reject_banned_model_identity(value)
