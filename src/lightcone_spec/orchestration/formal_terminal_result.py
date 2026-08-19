"""Durable result authority for source-owned TP2/DP2 serving terminals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.orchestration.formal_terminal_shards import (
    reopen_scalable_client_request_lifecycle,
    reopen_scalable_formal_gang_itl_bundle,
    reopen_scalable_formal_gang_request_terminal,
    reopen_scalable_formal_gang_terminal,
    reopen_scalable_unsigned_native_itl_bundle,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalResultProjection,
    NativeTerminalResultProofArtifact,
    TerminalRequestExpectation,
    build_native_terminal_external_control_binding,
    canonical_sha256,
    derive_native_terminal_result_projection_from_verified_formal_control,
    validate_native_terminal_artifact,
    validate_native_terminal_result_proof_artifact,
)
from lightcone_spec.runtime.attestation import NO_TRUSTED_ATTESTERS
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

from .formal_physical_dispatch import (
    FORMAL_GANG_SERVING_PROTOCOL_SHA256,
    FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
    FormalServingRunPlan,
    _reopen_schedule_receipt,
    formal_serving_request_schedule_rows,
)

FORMAL_DISTRIBUTED_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_distributed_terminal_external_control_protocol",
        "source": "callback_free_patched_sglang_all_rank_terminal",
        "topologies": ["tp2_dp1", "tp1_dp2"],
        "authority": "offline_external_control_after_immutable_pull",
        "tp2": "all_rank_atomic_publication_and_rank_native_state",
        "dp2": "sticky_disjoint_replicas_without_cross_replica_gradient",
        "itl": "exact_first_party_request_tokens_and_native_pointer_bundle",
        "client_lifecycle": (
            "complete_ordered_digest_with_exact_non_submitted_scored_IDs_"
            "and_native_terminal_for_every_submitted_request"
        ),
        "warmup": "all_rows_offered_submitted_and_native_completed",
    }
)
FORMAL_TP1_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_tp1_terminal_external_control_protocol",
        "native_terminal": "exact_unsigned_pinned_sglang_terminal",
        "launch": "admission_consumption_and_budget_ledger_entry",
        "lifecycle": "run_receipt_gpu_snapshots_and_process_group_empty",
        "client_terminal": (
            "five_state_offered_denominator_with_native_completed_or_aborted_"
            "proof_for_submitted_requests"
        ),
        "authority": "offline_external_control_after_immutable_pull",
    }
)
FORMAL_PREFLIGHT_TP1_TERMINAL_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_preflight_tp1_terminal_result_proof_protocol",
        "terminal": "existing_external_controlled_native_terminal_proof",
        "launch": ("schema2_preflight_dispatch_budget_cap_and_atomic_wave_consumption"),
        "join": "exact_raw_batch_row_run_binding_assignment_budget_and_cell",
        "scope": "preflight_interference_only",
        "legacy_native_proof_alone": "forbidden",
    }
)
FORMAL_CURRENT_PREFLIGHT_TP1_TERMINAL_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_current_preflight_tp1_terminal_result_proof_protocol",
        "mode": "formal_single_operator_v1",
        "terminal": "existing_external_controlled_native_terminal_proof",
        "execution": "path_bound_exact_eight_execution_manifest_and_raw_batch",
        "join": "exact_materialized_cell_registry_cell_and_run_binding",
        "scope": "preflight_interference_only",
        "legacy_native_proof_alone": "forbidden",
    }
)
FORMAL_SINGLE_OPERATOR_PREFLIGHT_TP1_RAW_TERMINAL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_preflight_tp1_raw_terminal_protocol",
        "mode": "formal_single_operator_v1",
        "terminal": "source_owned_canonical_raw_native_terminal",
        "execution": "exact_eight_manifest_and_raw_batch",
        "validation": "NO_TRUSTED_ATTESTERS_plus_exact_run_binding",
        "external_control_or_replay": "not_required_in_trusted_single_operator_mode",
        "scope": "preflight_interference_only",
    }
)

_LEGACY_RANK_TERMINAL_FIELDS = {
    "schema_version",
    "kind",
    "hook",
    "protocol_sha256",
    "topology",
    "rank",
    "world_size",
    "gpu_uuid",
    "execution_plan_sha256",
    "rank_config_sha256",
    "run_nonce_sha256",
    "method",
    "phase",
    "full_schedule_sha256",
    "local_request_routes_sha256",
    "sticky_cohort_routes_sha256",
    "expected_request_ids_sha256",
    "request_terminals",
    "request_terminal_sha256s",
    "native_state",
    "native_state_sha256",
    "status",
    "reason_code",
    "terminal_sha256",
}
_CURRENT_RANK_TERMINAL_FIELDS = _LEGACY_RANK_TERMINAL_FIELDS | {
    "client_lifecycle_sha256",
    "non_submitted_request_ids_sha256",
}
_NATIVE_ITL_POINTER_FIELDS = {
    "schema_version",
    "kind",
    "hook",
    "semantics",
    "release_status",
    "request_id",
    "request_started_ns",
    "request_terminal_ns",
    "terminal_status",
    "terminal_reason",
    "events",
    "result_pointer_sha256",
}
_UPDATE_FIELDS = {
    "update_index",
    "cohort_sha256",
    "cohort_epoch",
    "parameter_layout_sha256",
    "source_round",
    "source_version",
    "request_ids",
    "prefix_len_before",
    "optimizer_step",
    "published_version",
    "status",
    "loss",
    "gradient_norm",
    "reconstruction_ok",
    "reconstruction_max_abs",
    "reconstruction_relative_rms",
    "reconstruction_top1_match",
    "supervision_nonempty",
    "reconstruction_mean_kl",
    "online_hint_error",
    "online_ensemble_entropy",
    "online_effective_experts",
    "online_expert_probabilities",
    "online_cumulative_losses",
    "online_expert_gradient_norms",
    "source_state_sha256",
    "candidate_bytes_sha256",
    "optimizer_state_bytes_sha256",
    "proposal_evidence_sha256",
    "update_sha256",
}


def _sha(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _new_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or Path(os.path.abspath(path)) != path
        or path.exists()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise ValueError(f"{label} must be a new absolute path")
    return path


def _object(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _rank_terminal_digest(
    value: dict[str, object],
    *,
    require_client_lifecycle: bool,
) -> str:
    observed_fields = set(value)
    expected_fields = (
        _CURRENT_RANK_TERMINAL_FIELDS
        if require_client_lifecycle
        else _LEGACY_RANK_TERMINAL_FIELDS
    )
    legacy_with_current_codec = (
        not require_client_lifecycle
        and observed_fields == _CURRENT_RANK_TERMINAL_FIELDS
        and value.get("client_lifecycle_sha256") is None
        and value.get("non_submitted_request_ids_sha256") is None
    )
    if observed_fields != expected_fields and not legacy_with_current_codec:
        raise ValueError("formal rank terminal fields differ")
    row = dict(value)
    declared = _sha("formal rank terminal", row.pop("terminal_sha256", None))
    if content_sha256(row) != declared:
        raise ValueError("formal rank terminal content digest differs")
    return declared


@dataclass(frozen=True)
class FormalTp1TerminalExternalControlBinding:
    """Exact TP1 terminal plus admitted physical-lifecycle signing subject."""

    schema_version: Literal[1]
    kind: Literal["formal_tp1_terminal_external_control_binding"]
    plan_raw_sha256: str
    plan_semantic_sha256: str
    live_run_receipt_raw_sha256: str
    live_run_receipt_semantic_sha256: str
    raw_terminal_raw_sha256: str
    raw_terminal_semantic_sha256: str
    native_terminal_binding_sha256: str
    lifecycle_timing_raw_sha256: str
    lifecycle_timing_semantic_sha256: str
    launch_admission_raw_sha256: str
    launch_admission_semantic_sha256: str
    launch_consumption_raw_sha256: str
    launch_consumption_semantic_sha256: str
    budget_consumption_raw_sha256: str
    budget_consumption_semantic_sha256: str
    inventory_sha256: str
    registry_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    run_nonce_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_tp1_terminal_external_control_binding"
        ):
            raise ValueError("formal TP1 control binding schema differs")
        for name in self.__dataclass_fields__:
            if name.endswith("sha256"):
                _sha(f"formal TP1 control {name}", getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @cached_property
    def lineage_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_tp1_terminal_external_control_lineage",
                "binding_sha256": self.sha256,
                "execution_plan_sha256": self.execution_plan_sha256,
                "rank_config_sha256": self.rank_config_sha256,
                "run_nonce_sha256": self.run_nonce_sha256,
                "launch_admission_semantic_sha256": (
                    self.launch_admission_semantic_sha256
                ),
                "launch_consumption_semantic_sha256": (
                    self.launch_consumption_semantic_sha256
                ),
                "budget_consumption_semantic_sha256": (
                    self.budget_consumption_semantic_sha256
                ),
            }
        )


@dataclass(frozen=True)
class FormalTp1TerminalResultProofArtifact:
    schema_version: Literal[1]
    kind: Literal["formal_tp1_terminal_result_proof_artifact"]
    plan: CanonicalJsonProofBinding
    live_run_receipt: CanonicalJsonProofBinding
    raw_terminal: CanonicalJsonProofBinding
    lifecycle_timing: CanonicalJsonProofBinding
    launch_admission: CanonicalJsonProofBinding
    launch_consumption: CanonicalJsonProofBinding
    budget_consumption: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    result: dict[str, object]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_tp1_terminal_result_proof_artifact"
            or type(self.result) is not dict
            or self.result.get("kind") != "native_terminal_result_projection"
        ):
            raise ValueError("formal TP1 proof artifact schema/result differs")
        for value in (
            self.plan,
            self.live_run_receipt,
            self.raw_terminal,
            self.lifecycle_timing,
            self.launch_admission,
            self.launch_consumption,
            self.budget_consumption,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("formal TP1 proof lost a path binding")
        for label, digest in (
            ("inventory", self.expected_inventory_sha256),
            ("registry", self.expected_registry_sha256),
            ("release root", self.expected_root_manifest_sha256),
        ):
            _sha(f"formal TP1 proof {label}", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "plan": self.plan.to_dict(),
            "live_run_receipt": self.live_run_receipt.to_dict(),
            "raw_terminal": self.raw_terminal.to_dict(),
            "lifecycle_timing": self.lifecycle_timing.to_dict(),
            "launch_admission": self.launch_admission.to_dict(),
            "launch_consumption": self.launch_consumption.to_dict(),
            "budget_consumption": self.budget_consumption.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalTp1TerminalResultProofArtifact:
        row = _object("formal TP1 terminal proof", value, set(cls.__dataclass_fields__))
        for name in (
            "plan",
            "live_run_receipt",
            "raw_terminal",
            "lifecycle_timing",
            "launch_admission",
            "launch_consumption",
            "budget_consumption",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["control_attestation"] = ControlArtifactAttestation.from_dict(
            row["control_attestation"]
        )
        row["replay_reservation"] = ChallengeReplayReservationBinding.from_dict(
            row["replay_reservation"]
        )
        return cls(**row)  # type: ignore[arg-type]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class FormalPreflightTp1TerminalResultProofArtifact:
    """Admission-bound wrapper for a preflight interference TP1 terminal."""

    schema_version: Literal[1]
    kind: Literal["formal_preflight_tp1_terminal_result_proof_artifact"]
    protocol_sha256: str
    launch_cap_schedule: CanonicalJsonProofBinding
    launch_consumption: CanonicalJsonProofBinding
    interference_raw_batch: CanonicalJsonProofBinding
    native_result_proof: CanonicalJsonProofBinding
    materialized_cell_id: str
    registry_cell_id: str
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    result: dict[str, object]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_preflight_tp1_terminal_result_proof_artifact"
            or self.protocol_sha256
            != FORMAL_PREFLIGHT_TP1_TERMINAL_PROOF_PROTOCOL_SHA256
            or type(self.result) is not dict
            or self.result.get("kind") != "native_terminal_result_projection"
        ):
            raise ValueError("formal preflight TP1 proof schema/result differs")
        for value in (
            self.launch_cap_schedule,
            self.launch_consumption,
            self.interference_raw_batch,
            self.native_result_proof,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("formal preflight TP1 proof lost a path binding")
        for label, digest in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
            ("inventory", self.expected_inventory_sha256),
            ("registry", self.expected_registry_sha256),
            ("release root", self.expected_root_manifest_sha256),
        ):
            _sha(f"formal preflight TP1 proof {label}", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "launch_cap_schedule": self.launch_cap_schedule.to_dict(),
            "launch_consumption": self.launch_consumption.to_dict(),
            "interference_raw_batch": self.interference_raw_batch.to_dict(),
            "native_result_proof": self.native_result_proof.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalPreflightTp1TerminalResultProofArtifact:
        row = _object(
            "formal preflight TP1 terminal proof",
            value,
            set(cls.__dataclass_fields__),
        )
        for name in (
            "launch_cap_schedule",
            "launch_consumption",
            "interference_raw_batch",
            "native_result_proof",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(**row)  # type: ignore[arg-type]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class FormalCurrentPreflightTp1TerminalResultProofArtifact:
    """Current single-operator wrapper for one exact preflight TP1 terminal.

    This intentionally does not claim the signed staged-launch authority used by
    :class:`FormalPreflightTp1TerminalResultProofArtifact`.  It instead binds the
    exact-eight source-owned execution manifest, immutable raw batch, and the
    external-controlled native terminal.  The closed kind lets current trusted
    single-operator runs use the formal terminal/ITL pipeline without accepting
    a bare legacy native proof as formal evidence.
    """

    schema_version: Literal[1]
    kind: Literal["formal_current_preflight_tp1_terminal_result_proof_artifact"]
    protocol_sha256: str
    execution_manifest: CanonicalJsonProofBinding
    interference_raw_batch: CanonicalJsonProofBinding
    native_result_proof: CanonicalJsonProofBinding
    materialized_cell_id: str
    registry_cell_id: str
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    result: dict[str, object]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind
            != "formal_current_preflight_tp1_terminal_result_proof_artifact"
            or self.protocol_sha256
            != FORMAL_CURRENT_PREFLIGHT_TP1_TERMINAL_PROOF_PROTOCOL_SHA256
            or type(self.result) is not dict
            or self.result.get("kind") != "native_terminal_result_projection"
        ):
            raise ValueError("formal current preflight TP1 proof schema/result differs")
        for value in (
            self.execution_manifest,
            self.interference_raw_batch,
            self.native_result_proof,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError(
                    "formal current preflight TP1 proof lost a path binding"
                )
        for label, digest in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
            ("inventory", self.expected_inventory_sha256),
            ("registry", self.expected_registry_sha256),
            ("release root", self.expected_root_manifest_sha256),
        ):
            _sha(f"formal current preflight TP1 proof {label}", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "execution_manifest": self.execution_manifest.to_dict(),
            "interference_raw_batch": self.interference_raw_batch.to_dict(),
            "native_result_proof": self.native_result_proof.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, value: object
    ) -> FormalCurrentPreflightTp1TerminalResultProofArtifact:
        row = _object(
            "formal current preflight TP1 terminal proof",
            value,
            set(cls.__dataclass_fields__),
        )
        for name in (
            "execution_manifest",
            "interference_raw_batch",
            "native_result_proof",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(**row)  # type: ignore[arg-type]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class FormalSingleOperatorPreflightRawRequestResult:
    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...] | None
    terminal_status: str
    terminal_reason: str
    submitted_to_server: bool

    @classmethod
    def from_expectation(
        cls,
        value: TerminalRequestExpectation,
    ) -> FormalSingleOperatorPreflightRawRequestResult:
        if type(value) is not TerminalRequestExpectation:
            raise TypeError("single-operator raw request result is not validated")
        value.validate()
        return cls(**value.__dict__)

    def __post_init__(self) -> None:
        TerminalRequestExpectation(**self.__dict__).validate()

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "input_token_ids": list(self.input_token_ids),
            "output_token_ids": (
                None if self.output_token_ids is None else list(self.output_token_ids)
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalSingleOperatorPreflightRawRequestResult:
        row = _object(
            "single-operator preflight raw request result",
            value,
            set(cls.__dataclass_fields__),
        )
        inputs = row.pop("input_token_ids")
        outputs = row.pop("output_token_ids")
        if type(inputs) is not list or (
            outputs is not None and type(outputs) is not list
        ):
            raise TypeError("single-operator raw request token rows are malformed")
        return cls(
            **row,
            input_token_ids=tuple(inputs),
            output_token_ids=None if outputs is None else tuple(outputs),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightRawTerminalProjection:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_preflight_raw_terminal_projection"]
    run_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    method: str
    terminal_sha256: str
    requests: tuple[FormalSingleOperatorPreflightRawRequestResult, ...]
    scored_request_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_preflight_raw_terminal_projection"
        ):
            raise ValueError("single-operator raw terminal projection schema differs")
        for label, digest in (
            ("run nonce", self.run_nonce_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("rank config", self.rank_config_sha256),
            ("terminal", self.terminal_sha256),
        ):
            _sha(f"single-operator raw projection {label}", digest)
        for label, value in (
            ("run ID", self.run_id),
            ("attempt ID", self.attempt_id),
            ("method", self.method),
        ):
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"single-operator raw projection {label} is invalid")
        if (
            type(self.requests) is not tuple
            or any(
                type(row) is not FormalSingleOperatorPreflightRawRequestResult
                for row in self.requests
            )
            or tuple(row.request_id for row in self.requests) != self.scored_request_ids
            or len(set(self.scored_request_ids)) != len(self.scored_request_ids)
        ):
            raise ValueError("single-operator raw terminal request coverage differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "requests": [row.to_dict() for row in self.requests],
            "scored_request_ids": list(self.scored_request_ids),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> FormalSingleOperatorPreflightRawTerminalProjection:
        row = _object(
            "single-operator preflight raw terminal projection",
            value,
            set(cls.__dataclass_fields__),
        )
        requests = row.pop("requests")
        scored = row.pop("scored_request_ids")
        if type(requests) is not list or type(scored) is not list:
            raise TypeError("single-operator raw terminal arrays are malformed")
        return cls(
            **row,
            requests=tuple(
                FormalSingleOperatorPreflightRawRequestResult.from_dict(item)
                for item in requests
            ),
            scored_request_ids=tuple(scored),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightTp1RawTerminalProofArtifact:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_preflight_tp1_raw_terminal_proof_artifact"]
    protocol_sha256: str
    execution_manifest: CanonicalJsonProofBinding
    interference_raw_batch: CanonicalJsonProofBinding
    raw_terminal: CanonicalJsonProofBinding
    materialized_cell_id: str
    registry_cell_id: str
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    result: dict[str, object]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind
            != "formal_single_operator_preflight_tp1_raw_terminal_proof_artifact"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREFLIGHT_TP1_RAW_TERMINAL_PROTOCOL_SHA256
            or type(self.result) is not dict
            or self.result.get("kind")
            != "formal_single_operator_preflight_raw_terminal_projection"
        ):
            raise ValueError("single-operator raw terminal proof schema differs")
        for value in (
            self.execution_manifest,
            self.interference_raw_batch,
            self.raw_terminal,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("single-operator raw terminal source is not path-bound")
        for label, digest in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
            ("inventory", self.expected_inventory_sha256),
            ("registry", self.expected_registry_sha256),
            ("release root", self.expected_root_manifest_sha256),
        ):
            _sha(f"single-operator raw terminal {label}", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "execution_manifest": self.execution_manifest.to_dict(),
            "interference_raw_batch": self.interference_raw_batch.to_dict(),
            "raw_terminal": self.raw_terminal.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> FormalSingleOperatorPreflightTp1RawTerminalProofArtifact:
        row = _object(
            "single-operator preflight TP1 raw terminal proof",
            value,
            set(cls.__dataclass_fields__),
        )
        for name in ("execution_manifest", "interference_raw_batch", "raw_terminal"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(**row)  # type: ignore[arg-type]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _derive_formal_single_operator_preflight_tp1_raw_terminal(
    artifact: FormalSingleOperatorPreflightTp1RawTerminalProofArtifact,
) -> FormalSingleOperatorPreflightRawTerminalProjection:
    from lightcone_spec.experiments.formal_preflight_execution import (
        FormalPreflightInterferenceExecutionManifest,
    )
    from lightcone_spec.experiments.preflight_interference import (
        FormalPreflightInterferenceRawBatch,
    )

    manifest_binding = CanonicalJsonProofBinding.bind(
        artifact.execution_manifest.absolute_path
    )
    manifest = FormalPreflightInterferenceExecutionManifest.from_dict(
        manifest_binding.reopen()
    )
    batch_binding = CanonicalJsonProofBinding.bind(
        artifact.interference_raw_batch.absolute_path
    )
    batch = FormalPreflightInterferenceRawBatch.from_dict(batch_binding.reopen())
    batch.revalidate()
    manifest_rows = tuple(
        row
        for row in manifest.inputs
        if row.registry_cell_id == artifact.registry_cell_id
    )
    raw_rows = tuple(
        row
        for row in batch.rows
        if row.materialized_cell_id == artifact.materialized_cell_id
        and row.registry_cell_id == artifact.registry_cell_id
    )
    if len(manifest_rows) != 1 or len(raw_rows) != 1:
        raise ValueError("single-operator raw terminal lacks one exact preflight row")
    manifest_row = manifest_rows[0]
    raw = raw_rows[0]
    if (
        artifact.execution_manifest != manifest_binding
        or artifact.interference_raw_batch != batch_binding
        or manifest_binding.semantic_sha256 != manifest.sha256
        or batch_binding.semantic_sha256 != batch.sha256
        or manifest.dispatch_receipt_semantic_sha256 != batch.dispatch_sha256
        or batch.status != "WAITING_FOR_LOCAL_CONTROL"
        or batch.inventory_sha256 != artifact.expected_inventory_sha256
        or batch.registry_sha256 != artifact.expected_registry_sha256
        or raw.status != "WAITING_FOR_LOCAL_CONTROL"
        or raw.raw_terminal is None
        or raw.raw_terminal != artifact.raw_terminal
        or raw.run_binding != manifest_row.run_binding
    ):
        raise ValueError("single-operator raw terminal source lineage differs")
    raw_binding = CanonicalJsonProofBinding.bind(artifact.raw_terminal.absolute_path)
    evidence = validate_native_terminal_artifact(
        raw_binding.reopen(),
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        expected_binding=manifest_row.run_binding,
    )
    if (
        raw_binding != artifact.raw_terminal
        or evidence.authority_kind != "untrusted_raw_terminal"
        or tuple(row.request_id for row in evidence.requests)
        != manifest_row.run_binding.scored_request_ids
    ):
        raise ValueError("single-operator raw terminal content identity differs")
    return FormalSingleOperatorPreflightRawTerminalProjection(
        schema_version=1,
        kind="formal_single_operator_preflight_raw_terminal_projection",
        run_id=evidence.binding.run_id,
        run_nonce_sha256=evidence.binding.run_nonce_sha256,
        execution_plan_sha256=evidence.binding.execution_plan_sha256,
        rank_config_sha256=evidence.binding.rank_config_sha256,
        attempt_id=evidence.binding.attempt_id,
        method=evidence.binding.method,
        terminal_sha256=evidence.terminal_sha256,
        requests=tuple(
            FormalSingleOperatorPreflightRawRequestResult.from_expectation(row)
            for row in evidence.requests
        ),
        scored_request_ids=evidence.binding.scored_request_ids,
    )


def publish_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
    *,
    execution_manifest_path: str,
    interference_raw_batch_path: str,
    raw_terminal_path: str,
    materialized_cell_id: str,
    registry_cell_id: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    """Bind one trusted current preflight raw terminal without signing it."""

    output = _new_path(
        proof_artifact_path,
        label="single-operator preflight raw terminal proof output",
    )
    artifact = FormalSingleOperatorPreflightTp1RawTerminalProofArtifact(
        schema_version=1,
        kind="formal_single_operator_preflight_tp1_raw_terminal_proof_artifact",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_PREFLIGHT_TP1_RAW_TERMINAL_PROTOCOL_SHA256
        ),
        execution_manifest=CanonicalJsonProofBinding.bind(execution_manifest_path),
        interference_raw_batch=CanonicalJsonProofBinding.bind(
            interference_raw_batch_path
        ),
        raw_terminal=CanonicalJsonProofBinding.bind(raw_terminal_path),
        materialized_cell_id=materialized_cell_id,
        registry_cell_id=registry_cell_id,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        result={"kind": "formal_single_operator_preflight_raw_terminal_projection"},
    )
    projection = _derive_formal_single_operator_preflight_tp1_raw_terminal(artifact)
    artifact = FormalSingleOperatorPreflightTp1RawTerminalProofArtifact(
        **{**artifact.__dict__, "result": projection.to_dict()}
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)
    if (
        FormalSingleOperatorPreflightTp1RawTerminalProofArtifact.from_dict(
            binding.reopen()
        )
        != artifact
    ):
        raise RuntimeError("single-operator raw terminal proof changed on publication")
    return binding


def validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_execution_plan_sha256: str,
    expected_rank_config_sha256: str,
    expected_run_id: str,
    expected_run_nonce_sha256: str,
    expected_attempt_id: str,
    expected_method: str,
    now_ns: int,
) -> FormalSingleOperatorPreflightRawTerminalProjection:
    """Reopen a trusted raw terminal and exact current execution lineage."""

    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("single-operator raw terminal verification time is invalid")
    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalSingleOperatorPreflightTp1RawTerminalProofArtifact.from_dict(
        binding.reopen()
    )
    if (
        binding.semantic_sha256 != artifact.sha256
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("single-operator raw terminal proof identity differs")
    projection = _derive_formal_single_operator_preflight_tp1_raw_terminal(artifact)
    if projection.to_dict() != artifact.result:
        raise ValueError("single-operator raw terminal projection changed")
    expected = (
        (projection.execution_plan_sha256, expected_execution_plan_sha256),
        (projection.rank_config_sha256, expected_rank_config_sha256),
        (projection.run_id, expected_run_id),
        (projection.run_nonce_sha256, expected_run_nonce_sha256),
        (projection.attempt_id, expected_attempt_id),
        (projection.method, expected_method),
    )
    if any(observed != wanted for observed, wanted in expected):
        raise ValueError("single-operator raw terminal execution identity differs")
    return projection


def _derive_formal_current_preflight_tp1_terminal_result(
    artifact: FormalCurrentPreflightTp1TerminalResultProofArtifact,
    *,
    now_ns: int,
) -> NativeTerminalResultProjection:
    """Deep-rebuild one trusted current preflight terminal wrapper."""

    from lightcone_spec.experiments.formal_preflight_execution import (
        FormalPreflightInterferenceExecutionManifest,
    )
    from lightcone_spec.experiments.preflight_interference import (
        FormalPreflightInterferenceRawBatch,
    )

    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("formal current preflight TP1 proof time is invalid")
    manifest_binding = CanonicalJsonProofBinding.bind(
        artifact.execution_manifest.absolute_path
    )
    manifest = FormalPreflightInterferenceExecutionManifest.from_dict(
        manifest_binding.reopen()
    )
    batch_binding = CanonicalJsonProofBinding.bind(
        artifact.interference_raw_batch.absolute_path
    )
    batch = FormalPreflightInterferenceRawBatch.from_dict(batch_binding.reopen())
    batch.revalidate()
    manifest_rows = tuple(
        row
        for row in manifest.inputs
        if row.registry_cell_id == artifact.registry_cell_id
    )
    raw_rows = tuple(
        row
        for row in batch.rows
        if row.materialized_cell_id == artifact.materialized_cell_id
        and row.registry_cell_id == artifact.registry_cell_id
    )
    if len(manifest_rows) != 1 or len(raw_rows) != 1:
        raise ValueError("formal current preflight TP1 proof lacks one exact row")
    manifest_row = manifest_rows[0]
    raw = raw_rows[0]
    if (
        artifact.execution_manifest != manifest_binding
        or artifact.interference_raw_batch != batch_binding
        or manifest_binding.semantic_sha256 != manifest.sha256
        or batch_binding.semantic_sha256 != batch.sha256
        or manifest.dispatch_receipt_semantic_sha256 != batch.dispatch_sha256
        or batch.status != "WAITING_FOR_LOCAL_CONTROL"
        or batch.inventory_sha256 != artifact.expected_inventory_sha256
        or batch.registry_sha256 != artifact.expected_registry_sha256
        or raw.status != "WAITING_FOR_LOCAL_CONTROL"
        or raw.raw_terminal is None
        or raw.run_binding != manifest_row.run_binding
    ):
        raise ValueError("formal current preflight TP1 proof source lineage differs")
    native_binding = CanonicalJsonProofBinding.bind(
        artifact.native_result_proof.absolute_path
    )
    native = NativeTerminalResultProofArtifact.from_dict(native_binding.reopen())
    if (
        artifact.native_result_proof != native_binding
        or native_binding.semantic_sha256 != native.sha256
        or native.raw_terminal != raw.raw_terminal
        or native.expected_inventory_sha256 != artifact.expected_inventory_sha256
        or native.expected_registry_sha256 != artifact.expected_registry_sha256
        or native.expected_root_manifest_sha256
        != artifact.expected_root_manifest_sha256
    ):
        raise ValueError("formal current preflight TP1 native proof lineage differs")
    return validate_native_terminal_result_proof_artifact(
        native_binding.absolute_path,
        expected_inventory_sha256=artifact.expected_inventory_sha256,
        expected_registry_sha256=artifact.expected_registry_sha256,
        expected_root_manifest_sha256=artifact.expected_root_manifest_sha256,
        expected_execution_plan_sha256=raw.run_binding.execution_plan_sha256,
        expected_rank_config_sha256=raw.run_binding.rank_config_sha256,
        expected_run_id=raw.run_binding.run_id,
        expected_run_nonce_sha256=raw.run_binding.run_nonce_sha256,
        expected_attempt_id=raw.run_binding.attempt_id,
        expected_method=raw.run_binding.method,
        now_ns=now_ns,
    )


def publish_formal_current_preflight_tp1_terminal_result_proof_artifact(
    *,
    execution_manifest_path: str,
    interference_raw_batch_path: str,
    native_result_proof_path: str,
    materialized_cell_id: str,
    registry_cell_id: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    """Publish one current single-operator preflight terminal wrapper."""

    output = _new_path(
        proof_artifact_path,
        label="formal current preflight TP1 proof output",
    )
    native = NativeTerminalResultProofArtifact.from_dict(
        CanonicalJsonProofBinding.bind(native_result_proof_path).reopen()
    )
    artifact = FormalCurrentPreflightTp1TerminalResultProofArtifact(
        schema_version=1,
        kind="formal_current_preflight_tp1_terminal_result_proof_artifact",
        protocol_sha256=(FORMAL_CURRENT_PREFLIGHT_TP1_TERMINAL_PROOF_PROTOCOL_SHA256),
        execution_manifest=CanonicalJsonProofBinding.bind(execution_manifest_path),
        interference_raw_batch=CanonicalJsonProofBinding.bind(
            interference_raw_batch_path
        ),
        native_result_proof=CanonicalJsonProofBinding.bind(native_result_proof_path),
        materialized_cell_id=materialized_cell_id,
        registry_cell_id=registry_cell_id,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        result=native.result,
    )
    projection = _derive_formal_current_preflight_tp1_terminal_result(
        artifact,
        now_ns=now_ns,
    )
    artifact = FormalCurrentPreflightTp1TerminalResultProofArtifact(
        **{**artifact.__dict__, "result": projection.to_dict()}
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)
    if (
        FormalCurrentPreflightTp1TerminalResultProofArtifact.from_dict(binding.reopen())
        != artifact
    ):
        raise RuntimeError(
            "formal current preflight TP1 proof changed during publication"
        )
    return binding


def validate_formal_current_preflight_tp1_terminal_result_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_execution_plan_sha256: str,
    expected_rank_config_sha256: str,
    expected_run_id: str,
    expected_run_nonce_sha256: str,
    expected_attempt_id: str,
    expected_method: str,
    now_ns: int,
) -> NativeTerminalResultProjection:
    """Reopen the current exact-eight/raw/native terminal proof DAG."""

    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalCurrentPreflightTp1TerminalResultProofArtifact.from_dict(
        binding.reopen()
    )
    if (
        binding.semantic_sha256 != artifact.sha256
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("formal current preflight TP1 proof identity differs")
    projection = _derive_formal_current_preflight_tp1_terminal_result(
        artifact,
        now_ns=now_ns,
    )
    if projection.to_dict() != artifact.result:
        raise ValueError("formal current preflight TP1 proof projection changed")
    expected_values = (
        (projection.execution_plan_sha256, expected_execution_plan_sha256),
        (projection.rank_config_sha256, expected_rank_config_sha256),
        (projection.run_id, expected_run_id),
        (projection.run_nonce_sha256, expected_run_nonce_sha256),
        (projection.attempt_id, expected_attempt_id),
        (projection.method, expected_method),
    )
    if any(observed != expected for observed, expected in expected_values):
        raise ValueError("formal current preflight TP1 execution identity differs")
    return projection


def _derive_formal_preflight_tp1_terminal_result(
    artifact: FormalPreflightTp1TerminalResultProofArtifact,
    *,
    now_ns: int,
) -> NativeTerminalResultProjection:
    """Deep-rebuild one preflight-only TP1 wrapper from its sealed sources."""

    from lightcone_spec.experiments.formal_dispatch import (
        FormalPreflightDispatchReceipt,
    )
    from lightcone_spec.experiments.formal_preflight_launch import (
        revalidate_formal_preflight_launch_cap_schedule,
        validate_formal_preflight_launch_wave_consumption,
    )
    from lightcone_spec.experiments.preflight_interference import (
        FormalPreflightInterferenceRawBatch,
    )

    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("formal preflight TP1 proof time is invalid")
    schedule = revalidate_formal_preflight_launch_cap_schedule(
        artifact.launch_cap_schedule.absolute_path,
        current_ns=now_ns,
    )
    schedule_binding = CanonicalJsonProofBinding.bind(
        artifact.launch_cap_schedule.absolute_path
    )
    consumption = validate_formal_preflight_launch_wave_consumption(
        artifact.launch_consumption.absolute_path,
        current_ns=now_ns,
    )
    receipt = FormalPreflightDispatchReceipt.from_dict(
        schedule.dispatch_receipt.reopen()
    )
    token = receipt.revalidate(current_ns=now_ns)
    batch_binding = CanonicalJsonProofBinding.bind(
        artifact.interference_raw_batch.absolute_path
    )
    batch = FormalPreflightInterferenceRawBatch.from_dict(batch_binding.reopen())
    batch.revalidate()
    raw_rows = tuple(
        row
        for row in batch.rows
        if row.materialized_cell_id == artifact.materialized_cell_id
        and row.registry_cell_id == artifact.registry_cell_id
    )
    if len(raw_rows) != 1:
        raise ValueError("formal preflight TP1 proof lacks one exact raw row")
    raw = raw_rows[0]
    sealed_rows = tuple(
        row
        for row in token.subject.execution_bindings
        if row.materialized_cell_id == artifact.materialized_cell_id
        and row.registry_cell_id == artifact.registry_cell_id
    )
    if len(sealed_rows) != 1:
        raise ValueError("formal preflight TP1 proof lacks one sealed assignment")
    sealed = sealed_rows[0]
    cap = schedule.cap_for_registry_cell(artifact.registry_cell_id)
    activation = token.dispatch_context.activation_artifact
    if (
        artifact.launch_cap_schedule != schedule_binding
        or artifact.launch_consumption
        != CanonicalJsonProofBinding.bind(artifact.launch_consumption.absolute_path)
        or consumption.schedule != schedule_binding
        or artifact.materialized_cell_id not in consumption.materialized_cell_ids
        or artifact.registry_cell_id not in consumption.registry_cell_ids
        or cap.materialized_cell_id != artifact.materialized_cell_id
        or cap.runner_kind != "first_party_interference"
        or cap.topology_mode != "tp1_dp1"
        or cap.wave_index != consumption.wave_index
        or cap.process_hard_timeout_ns != consumption.process_hard_timeout_ns
        or batch_binding.semantic_sha256 != batch.sha256
        or artifact.interference_raw_batch != batch_binding
        or batch.status != "WAITING_FOR_LOCAL_CONTROL"
        or batch.dispatch_sha256 != token.sha256
        or batch.registry_sha256 != token.manifest.registry_sha256
        or batch.inventory_sha256 != token.subject.inventory_sha256
        or batch.activation_sha256 != activation.sha256
        or batch.runtime_sha256 != activation.runtime_sha256
        or batch.split_sha256 != activation.split_sha256
        or raw.status != "WAITING_FOR_LOCAL_CONTROL"
        or raw.raw_terminal is None
        or raw.assignment_sha256 != sealed.assignment_sha256
        or raw.experiment_budget_sha256 != sealed.experiment_budget_sha256
        or raw.inventory_sha256 != token.subject.inventory_sha256
        or raw.gpu_uuid != sealed.gpu_uuids[0]
        or artifact.expected_inventory_sha256 != token.subject.inventory_sha256
        or artifact.expected_registry_sha256 != token.manifest.registry_sha256
        or artifact.expected_root_manifest_sha256
        != token.protocol_lock.offline_release_trust_root_sha256
    ):
        raise ValueError("formal preflight TP1 proof source lineage differs")
    native_binding = CanonicalJsonProofBinding.bind(
        artifact.native_result_proof.absolute_path
    )
    native = NativeTerminalResultProofArtifact.from_dict(native_binding.reopen())
    if (
        artifact.native_result_proof != native_binding
        or native_binding.semantic_sha256 != native.sha256
        or native.raw_terminal != raw.raw_terminal
        or native.expected_inventory_sha256 != artifact.expected_inventory_sha256
        or native.expected_registry_sha256 != artifact.expected_registry_sha256
        or native.expected_root_manifest_sha256
        != artifact.expected_root_manifest_sha256
    ):
        raise ValueError("formal preflight TP1 native proof lineage differs")
    return validate_native_terminal_result_proof_artifact(
        native_binding.absolute_path,
        expected_inventory_sha256=artifact.expected_inventory_sha256,
        expected_registry_sha256=artifact.expected_registry_sha256,
        expected_root_manifest_sha256=artifact.expected_root_manifest_sha256,
        expected_execution_plan_sha256=raw.run_binding.execution_plan_sha256,
        expected_rank_config_sha256=raw.run_binding.rank_config_sha256,
        expected_run_id=raw.run_binding.run_id,
        expected_run_nonce_sha256=raw.run_binding.run_nonce_sha256,
        expected_attempt_id=raw.run_binding.attempt_id,
        expected_method=raw.run_binding.method,
        now_ns=now_ns,
    )


def publish_formal_preflight_tp1_terminal_result_proof_artifact(
    *,
    launch_cap_schedule_path: str,
    launch_consumption_path: str,
    interference_raw_batch_path: str,
    native_result_proof_path: str,
    materialized_cell_id: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    """Publish an admission-bound terminal proof for preflight TP1 only."""

    output = _new_path(proof_artifact_path, label="formal preflight TP1 proof output")
    schedule = CanonicalJsonProofBinding.bind(launch_cap_schedule_path)
    from lightcone_spec.experiments.formal_preflight_launch import (
        revalidate_formal_preflight_launch_cap_schedule,
    )

    rebuilt = revalidate_formal_preflight_launch_cap_schedule(
        launch_cap_schedule_path,
        current_ns=now_ns,
    )
    cap_rows = tuple(
        row
        for row in rebuilt.cell_caps
        if row.materialized_cell_id == materialized_cell_id
    )
    if len(cap_rows) != 1:
        raise ValueError("formal preflight TP1 output cell is not exact")
    native_result = NativeTerminalResultProofArtifact.from_dict(
        CanonicalJsonProofBinding.bind(native_result_proof_path).reopen()
    ).result
    artifact = FormalPreflightTp1TerminalResultProofArtifact(
        schema_version=1,
        kind="formal_preflight_tp1_terminal_result_proof_artifact",
        protocol_sha256=FORMAL_PREFLIGHT_TP1_TERMINAL_PROOF_PROTOCOL_SHA256,
        launch_cap_schedule=schedule,
        launch_consumption=CanonicalJsonProofBinding.bind(launch_consumption_path),
        interference_raw_batch=CanonicalJsonProofBinding.bind(
            interference_raw_batch_path
        ),
        native_result_proof=CanonicalJsonProofBinding.bind(native_result_proof_path),
        materialized_cell_id=materialized_cell_id,
        registry_cell_id=cap_rows[0].registry_cell_id,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        result=native_result,
    )
    projection = _derive_formal_preflight_tp1_terminal_result(
        artifact,
        now_ns=now_ns,
    )
    artifact = FormalPreflightTp1TerminalResultProofArtifact(
        **{**artifact.__dict__, "result": projection.to_dict()}
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)
    if (
        FormalPreflightTp1TerminalResultProofArtifact.from_dict(binding.reopen())
        != artifact
    ):
        raise RuntimeError("formal preflight TP1 proof changed during publication")
    return binding


def validate_formal_preflight_tp1_terminal_result_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_execution_plan_sha256: str,
    expected_rank_config_sha256: str,
    expected_run_id: str,
    expected_run_nonce_sha256: str,
    expected_attempt_id: str,
    expected_method: str,
    now_ns: int,
) -> NativeTerminalResultProjection:
    """Reopen the full preflight launch/raw/native authority DAG."""

    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalPreflightTp1TerminalResultProofArtifact.from_dict(binding.reopen())
    if (
        binding.semantic_sha256 != artifact.sha256
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("formal preflight TP1 proof identity differs")
    projection = _derive_formal_preflight_tp1_terminal_result(
        artifact,
        now_ns=now_ns,
    )
    if projection.to_dict() != artifact.result:
        raise ValueError("formal preflight TP1 proof projection changed")
    expected_values = (
        (projection.execution_plan_sha256, expected_execution_plan_sha256),
        (projection.rank_config_sha256, expected_rank_config_sha256),
        (projection.run_id, expected_run_id),
        (projection.run_nonce_sha256, expected_run_nonce_sha256),
        (projection.attempt_id, expected_attempt_id),
        (projection.method, expected_method),
    )
    if any(observed != expected for observed, expected in expected_values):
        raise ValueError("formal preflight TP1 execution identity differs")
    return projection


@dataclass(frozen=True)
class _ValidatedTp1Raw:
    plan: FormalServingRunPlan
    binding: FormalTp1TerminalExternalControlBinding
    live_run_receipt: CanonicalJsonProofBinding
    raw_terminal: CanonicalJsonProofBinding
    lifecycle_timing: CanonicalJsonProofBinding
    launch_admission: CanonicalJsonProofBinding
    launch_consumption: CanonicalJsonProofBinding
    budget_consumption: CanonicalJsonProofBinding


def _nonnegative_integer(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_pointer(
    value: object,
    *,
    expected: FormalDistributedTerminalRequestResult,
) -> tuple[str, str]:
    pointer = _object(
        "formal distributed native ITL pointer", value, _NATIVE_ITL_POINTER_FIELDS
    )
    declared = _sha(
        "formal distributed native ITL pointer",
        pointer["result_pointer_sha256"],
    )
    unsigned = dict(pointer)
    unsigned.pop("result_pointer_sha256")
    if (
        canonical_sha256(unsigned) != declared
        or pointer["schema_version"] != 1
        or pointer["kind"] != "sglang_native_itl_result_pointer"
        or pointer["hook"] != "sglang.schema_v3.native_per_token_timestamp.v2"
        or pointer["semantics"] != "scheduler_committed_token_at_result_processor_v1"
        or pointer["release_status"] != "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"
        or pointer["request_id"] != expected.request_id
        or pointer["terminal_status"] != expected.terminal_status
        or pointer["terminal_reason"] != expected.terminal_reason
    ):
        raise ValueError("formal distributed native ITL pointer identity differs")
    started = _nonnegative_integer(
        "formal distributed native ITL request start",
        pointer["request_started_ns"],
    )
    terminal = _nonnegative_integer(
        "formal distributed native ITL request terminal",
        pointer["request_terminal_ns"],
    )
    events = pointer["events"]
    if type(events) is not list or len(events) != len(expected.output_token_ids):
        raise ValueError("formal distributed native ITL token coverage differs")
    previous = started
    for index, (event_value, token_id) in enumerate(
        zip(events, expected.output_token_ids, strict=True)
    ):
        event = _object(
            "formal distributed native ITL event",
            event_value,
            {"token_index", "token_id", "observed_ns"},
        )
        observed = _nonnegative_integer(
            "formal distributed native ITL observation",
            event["observed_ns"],
        )
        if (
            event["token_index"] != index
            or event["token_id"] != token_id
            or observed <= previous
            or observed > terminal
        ):
            raise ValueError("formal distributed native ITL event order differs")
        previous = observed
    if terminal < previous:
        raise ValueError("formal distributed native ITL terminal precedes tokens")
    return declared, canonical_sha256(events)


@dataclass(frozen=True)
class FormalDistributedTerminalRequestResult:
    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    terminal_status: Literal["completed"]
    terminal_reason: str
    submitted_to_server: Literal[True]
    request_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not self.request_id
            or type(self.input_token_ids) is not tuple
            or not self.input_token_ids
            or type(self.output_token_ids) is not tuple
            or not self.output_token_ids
            or any(
                type(token) is not int or token < 0
                for token in (*self.input_token_ids, *self.output_token_ids)
            )
            or self.terminal_status != "completed"
            or type(self.terminal_reason) is not str
            or not self.terminal_reason
            or self.submitted_to_server is not True
        ):
            raise ValueError("formal distributed request terminal is invalid")
        _sha("formal distributed request result", self.request_sha256)
        if self.request_sha256 != content_sha256(
            {
                "request_id": self.request_id,
                "input_token_ids": list(self.input_token_ids),
                "output_token_ids": list(self.output_token_ids),
                "terminal_status": self.terminal_status,
                "terminal_reason": self.terminal_reason,
                "submitted_to_server": self.submitted_to_server,
            }
        ):
            raise ValueError("formal distributed request result digest differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "input_token_ids": list(self.input_token_ids),
            "output_token_ids": list(self.output_token_ids),
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
            "submitted_to_server": self.submitted_to_server,
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True)
class FormalDistributedTerminalUpdateResult:
    rank: int
    source_row: dict[str, object]

    def __post_init__(self) -> None:
        if (
            self.rank not in {0, 1}
            or type(self.source_row) is not dict
            or set(self.source_row) != _UPDATE_FIELDS
        ):
            raise ValueError("formal distributed update row is invalid")
        if (
            type(self.source_row.get("status")) is not str
            or type(self.source_row.get("reconstruction_ok")) is not bool
            or type(self.source_row.get("update_sha256")) is not str
        ):
            raise ValueError("formal distributed update row lacks native identity")
        declared = _sha("formal distributed update", self.source_row["update_sha256"])
        unsigned = dict(self.source_row)
        unsigned.pop("update_sha256")
        if content_sha256(unsigned) != declared:
            raise ValueError("formal distributed update digest differs")
        for field, minimum in (
            ("update_index", 0),
            ("cohort_epoch", 0),
            ("source_round", 1),
            ("source_version", 0),
            ("optimizer_step", 1),
        ):
            value = self.source_row[field]
            if type(value) is not int or value < minimum:
                raise ValueError("formal distributed update integer differs")
        for field in ("cohort_sha256", "parameter_layout_sha256"):
            _sha(f"formal distributed update {field}", self.source_row[field])
        request_ids = self.source_row["request_ids"]
        prefixes = self.source_row["prefix_len_before"]
        if (
            type(request_ids) is not list
            or not request_ids
            or type(prefixes) is not list
            or len(prefixes) != len(request_ids)
            or any(type(value) is not str or not value for value in request_ids)
            or any(type(value) is not int or value < 0 for value in prefixes)
        ):
            raise ValueError("formal distributed update source coverage differs")

    @property
    def status(self) -> str:
        return str(self.source_row["status"])

    @property
    def published_version(self) -> int | None:
        value = self.source_row.get("published_version")
        if value is not None and type(value) is not int:
            raise TypeError("formal distributed published version is invalid")
        return value

    @property
    def reconstruction_ok(self) -> bool:
        return bool(self.source_row["reconstruction_ok"])

    def to_dict(self) -> dict[str, object]:
        return {"rank": self.rank, "source_row": dict(self.source_row)}


@dataclass(frozen=True)
class FormalDistributedTerminalResultProjection:
    schema_version: Literal[1]
    kind: Literal["formal_distributed_terminal_result_projection"]
    stage: str
    topology_mode: Literal["tp2_dp1", "tp1_dp2"]
    run_id: str
    method: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    terminal_sha256: str
    native_itl_pointer_bundle_sha256: str
    lifecycle_timing_sha256: str
    launch_admission_sha256: str
    launch_consumption_sha256: str
    budget_consumption_sha256: str
    authority_kind: Literal["external_release_control"]
    external_control_binding_sha256: str
    external_control_envelope_sha256: str
    external_control_reservation_sha256: str
    requests: tuple[FormalDistributedTerminalRequestResult, ...]
    updates: tuple[FormalDistributedTerminalUpdateResult, ...]
    scored_request_ids: tuple[str, ...]
    output_token_count: int
    performance_counters: dict[str, object]
    rank_terminal_sha256s: tuple[str, str]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_distributed_terminal_result_projection"
            or self.topology_mode not in {"tp2_dp1", "tp1_dp2"}
            or self.authority_kind != "external_release_control"
            or type(self.requests) is not tuple
            or not self.requests
            or any(
                type(row) is not FormalDistributedTerminalRequestResult
                for row in self.requests
            )
            or type(self.updates) is not tuple
            or any(
                type(row) is not FormalDistributedTerminalUpdateResult
                for row in self.updates
            )
            or self.scored_request_ids != tuple(row.request_id for row in self.requests)
            or self.output_token_count
            != sum(len(row.output_token_ids) for row in self.requests)
            or type(self.performance_counters) is not dict
            or type(self.rank_terminal_sha256s) is not tuple
            or len(self.rank_terminal_sha256s) != 2
        ):
            raise ValueError("formal distributed result projection is invalid")
        for label, value in (
            ("run nonce", self.run_nonce_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("rank config", self.rank_config_sha256),
            ("terminal", self.terminal_sha256),
            ("native ITL pointer bundle", self.native_itl_pointer_bundle_sha256),
            ("lifecycle timing", self.lifecycle_timing_sha256),
            ("launch admission", self.launch_admission_sha256),
            ("launch consumption", self.launch_consumption_sha256),
            ("budget consumption", self.budget_consumption_sha256),
            ("control binding", self.external_control_binding_sha256),
            ("control envelope", self.external_control_envelope_sha256),
            ("replay reservation", self.external_control_reservation_sha256),
            ("rank terminal zero", self.rank_terminal_sha256s[0]),
            ("rank terminal one", self.rank_terminal_sha256s[1]),
        ):
            _sha(f"formal distributed result {label}", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "stage": self.stage,
            "topology_mode": self.topology_mode,
            "run_id": self.run_id,
            "method": self.method,
            "run_nonce_sha256": self.run_nonce_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "attempt_id": self.attempt_id,
            "terminal_sha256": self.terminal_sha256,
            "native_itl_pointer_bundle_sha256": (self.native_itl_pointer_bundle_sha256),
            "lifecycle_timing_sha256": self.lifecycle_timing_sha256,
            "launch_admission_sha256": self.launch_admission_sha256,
            "launch_consumption_sha256": self.launch_consumption_sha256,
            "budget_consumption_sha256": self.budget_consumption_sha256,
            "authority_kind": self.authority_kind,
            "external_control_binding_sha256": self.external_control_binding_sha256,
            "external_control_envelope_sha256": self.external_control_envelope_sha256,
            "external_control_reservation_sha256": (
                self.external_control_reservation_sha256
            ),
            "requests": [row.to_dict() for row in self.requests],
            "updates": [row.to_dict() for row in self.updates],
            "scored_request_ids": list(self.scored_request_ids),
            "output_token_count": self.output_token_count,
            "performance_counters": dict(self.performance_counters),
            "rank_terminal_sha256s": list(self.rank_terminal_sha256s),
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class FormalDistributedTerminalExternalControlBinding:
    schema_version: Literal[1]
    kind: Literal["formal_distributed_terminal_external_control_binding"]
    plan_raw_sha256: str
    plan_semantic_sha256: str
    run_receipt_raw_sha256: str
    run_receipt_semantic_sha256: str
    request_terminal_raw_sha256: str
    request_terminal_semantic_sha256: str
    gang_terminal_raw_sha256: str
    gang_terminal_semantic_sha256: str
    pointer_bundle_raw_sha256: str
    pointer_bundle_semantic_sha256: str
    lifecycle_timing_raw_sha256: str
    lifecycle_timing_semantic_sha256: str
    launch_admission_raw_sha256: str
    launch_admission_semantic_sha256: str
    launch_consumption_raw_sha256: str
    launch_consumption_semantic_sha256: str
    budget_consumption_raw_sha256: str
    budget_consumption_semantic_sha256: str
    inventory_sha256: str
    registry_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    run_nonce_sha256: str
    topology_mode: Literal["tp2_dp1", "tp1_dp2"]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_distributed_terminal_external_control_binding"
            or self.topology_mode not in {"tp2_dp1", "tp1_dp2"}
        ):
            raise ValueError("formal distributed control binding is invalid")
        for name in self.__dataclass_fields__:
            if name.endswith("sha256"):
                _sha(f"formal distributed control {name}", getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @cached_property
    def lineage_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_distributed_terminal_external_control_lineage",
                "binding_sha256": self.sha256,
                "execution_plan_sha256": self.execution_plan_sha256,
                "rank_config_sha256": self.rank_config_sha256,
                "run_nonce_sha256": self.run_nonce_sha256,
                "topology_mode": self.topology_mode,
                "launch_admission_semantic_sha256": (
                    self.launch_admission_semantic_sha256
                ),
                "launch_consumption_semantic_sha256": (
                    self.launch_consumption_semantic_sha256
                ),
                "budget_consumption_semantic_sha256": (
                    self.budget_consumption_semantic_sha256
                ),
            }
        )


@dataclass(frozen=True)
class FormalDistributedTerminalResultProofArtifact:
    schema_version: Literal[1]
    kind: Literal["formal_distributed_terminal_result_proof_artifact"]
    plan: CanonicalJsonProofBinding
    run_receipt: CanonicalJsonProofBinding
    request_terminal: CanonicalJsonProofBinding
    gang_terminal: CanonicalJsonProofBinding
    pointer_bundle: CanonicalJsonProofBinding
    lifecycle_timing: CanonicalJsonProofBinding
    launch_admission: CanonicalJsonProofBinding
    launch_consumption: CanonicalJsonProofBinding
    budget_consumption: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    result: dict[str, object]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_distributed_terminal_result_proof_artifact"
            or type(self.result) is not dict
            or self.result.get("kind")
            != "formal_distributed_terminal_result_projection"
        ):
            raise ValueError("formal distributed proof artifact is invalid")
        for value in (
            self.plan,
            self.run_receipt,
            self.request_terminal,
            self.gang_terminal,
            self.pointer_bundle,
            self.lifecycle_timing,
            self.launch_admission,
            self.launch_consumption,
            self.budget_consumption,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("formal distributed proof lost a path binding")
            value.reopen()
        for label, value in (
            ("inventory", self.expected_inventory_sha256),
            ("registry", self.expected_registry_sha256),
            ("release root", self.expected_root_manifest_sha256),
        ):
            _sha(f"formal distributed proof {label}", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "plan": self.plan.to_dict(),
            "run_receipt": self.run_receipt.to_dict(),
            "request_terminal": self.request_terminal.to_dict(),
            "gang_terminal": self.gang_terminal.to_dict(),
            "pointer_bundle": self.pointer_bundle.to_dict(),
            "lifecycle_timing": self.lifecycle_timing.to_dict(),
            "launch_admission": self.launch_admission.to_dict(),
            "launch_consumption": self.launch_consumption.to_dict(),
            "budget_consumption": self.budget_consumption.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "expected_inventory_sha256": self.expected_inventory_sha256,
            "expected_registry_sha256": self.expected_registry_sha256,
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalDistributedTerminalResultProofArtifact:
        row = _object(
            "formal distributed proof",
            value,
            set(cls.__dataclass_fields__),
        )
        for name in (
            "plan",
            "run_receipt",
            "request_terminal",
            "gang_terminal",
            "pointer_bundle",
            "lifecycle_timing",
            "launch_admission",
            "launch_consumption",
            "budget_consumption",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["control_attestation"] = ControlArtifactAttestation.from_dict(
            row["control_attestation"]
        )
        row["replay_reservation"] = ChallengeReplayReservationBinding.from_dict(
            row["replay_reservation"]
        )
        return cls(**row)  # type: ignore[arg-type]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class _ValidatedDistributedRaw:
    plan: FormalServingRunPlan
    binding: FormalDistributedTerminalExternalControlBinding
    requests: tuple[FormalDistributedTerminalRequestResult, ...]
    updates_by_rank: tuple[
        tuple[FormalDistributedTerminalUpdateResult, ...],
        tuple[FormalDistributedTerminalUpdateResult, ...],
    ]
    performance_by_rank: tuple[dict[str, object], dict[str, object]]
    rank_terminal_sha256s: tuple[str, str]
    aggregate_sha256: str


@dataclass(frozen=True)
class FormalDistributedPhysicalOutcome:
    """Source-derived COMPLETE outcome for one callback-free TP2/DP2 run."""

    status: Literal["COMPLETE"]
    plan: CanonicalJsonProofBinding
    run_receipt: CanonicalJsonProofBinding
    request_terminal: CanonicalJsonProofBinding
    gang_terminal: CanonicalJsonProofBinding
    native_itl_pointers: CanonicalJsonProofBinding
    lifecycle_timing: CanonicalJsonProofBinding
    before_gpu_snapshot: CanonicalJsonProofBinding
    ready_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    server_log: EvidenceFileBinding
    server_stdout: EvidenceFileBinding
    server_stderr: EvidenceFileBinding
    junit: EvidenceFileBinding
    execution_started_ns: int
    process_group_empty_checked_ns: int
    process_exit_code: int
    cleanup_kind: Literal["already_exited_clean", "sigterm_clean"]

    @property
    def finished_ns(self) -> int:
        return self.process_group_empty_checked_ns


def _request_result(value: object) -> FormalDistributedTerminalRequestResult:
    row = _object(
        "formal distributed request",
        value,
        {
            "request_id",
            "input_token_ids",
            "output_token_ids",
            "terminal_status",
            "terminal_reason",
            "submitted_to_server",
        },
    )
    inputs = row.pop("input_token_ids")
    outputs = row.pop("output_token_ids")
    if type(inputs) is not list or type(outputs) is not list:
        raise TypeError("formal distributed request tokens must be arrays")
    request_value = {
        **row,
        "input_token_ids": inputs,
        "output_token_ids": outputs,
    }
    return FormalDistributedTerminalRequestResult(
        **row,  # type: ignore[arg-type]
        input_token_ids=tuple(inputs),
        output_token_ids=tuple(outputs),
        request_sha256=content_sha256(request_value),
    )


def _performance(
    rows: tuple[dict[str, object], dict[str, object]],
    *,
    topology: str,
) -> dict[str, object]:
    for row in rows:
        if type(row) is not dict:
            raise TypeError("formal distributed performance row must be an object")
    logical = dict(rows[0])
    peaks = tuple(row.get("peak_hbm_bytes") for row in rows)
    if any(type(value) is not int or value < 0 for value in peaks):
        raise ValueError("formal distributed peak HBM counter is invalid")
    logical["peak_hbm_bytes"] = max(peaks)
    safety = (
        "exactness_violations",
        "version_mismatches",
        "fallbacks",
        "nonfinite_updates",
        "oom_events",
        "retractions",
        "communicator_failures",
    )
    for field in safety:
        values = tuple(row.get(field) for row in rows)
        if any(value is None for value in values):
            logical[field] = None
        elif all(type(value) is int and value >= 0 for value in values):
            logical[field] = sum(values) if topology == "tp1_dp2" else max(values)
        else:
            raise ValueError("formal distributed safety counter is invalid")
    if topology == "tp1_dp2":
        for field in ("target_calls", "updates_launched", "updates_published"):
            values = tuple(row.get(field) for row in rows)
            if all(value is None for value in values):
                logical[field] = None
            elif any(type(value) is not int or value < 0 for value in values):
                raise ValueError("formal DP2 logical counter is invalid")
            else:
                logical[field] = sum(values)
        exposed = tuple(row.get("exposed_update_ms") for row in rows)
        if all(type(value) in {int, float} and value >= 0 for value in exposed):
            logical["exposed_update_ms"] = max(exposed)
    else:
        for field in ("updates_launched", "updates_published"):
            if rows[0].get(field) != rows[1].get(field):
                raise ValueError("formal TP2 logical update counters differ by rank")
    logical["formal_topology_mode"] = topology
    logical["rank_performance_sha256s"] = [content_sha256(row) for row in rows]
    return logical


def _phase_requests(
    value: object,
    *,
    schedule,
    phase: str,
) -> tuple[FormalDistributedTerminalRequestResult, ...]:
    if type(value) is not list:
        raise TypeError("formal distributed request phase must be an array")
    rows = tuple(_request_result(row) for row in value)
    expected = tuple(
        row
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == phase
    )
    if len(rows) != len(expected):
        raise ValueError("formal distributed request phase coverage differs")
    for terminal, sealed in zip(rows, expected, strict=True):
        if (
            terminal.request_id != sealed.request.request_id
            or terminal.input_token_ids != sealed.request.input_token_ids
        ):
            raise ValueError("formal distributed request input identity differs")
    return rows


def _validate_pointer_bundle(
    value: object,
    *,
    plan: FormalServingRunPlan,
    warmup: tuple[FormalDistributedTerminalRequestResult, ...],
    scored: tuple[FormalDistributedTerminalRequestResult, ...],
    launch_admission: CanonicalJsonProofBinding,
    launch_consumption: CanonicalJsonProofBinding,
    budget_consumption: CanonicalJsonProofBinding,
) -> dict[str, tuple[str, str]]:
    bundle = _object(
        "formal distributed native ITL pointer bundle",
        value,
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "plan_sha256",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "warmup_pointers",
            "scored_pointers",
        },
    )
    if (
        bundle["schema_version"] != 1
        or bundle["kind"] != "unsigned_formal_gang_native_itl_pointer_bundle"
        or bundle["protocol_sha256"] != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or bundle["formal_execution_authorized"] is not False
        or bundle["plan_sha256"] != plan.sha256
        or CanonicalJsonProofBinding.from_dict(bundle["formal_launch_admission"])
        != launch_admission
        or CanonicalJsonProofBinding.from_dict(bundle["formal_launch_consumption"])
        != launch_consumption
        or CanonicalJsonProofBinding.from_dict(bundle["budget_consumption"])
        != budget_consumption
    ):
        raise ValueError("formal distributed native ITL pointer bundle differs")
    pointers_by_request: dict[str, tuple[str, str]] = {}
    for label, pointer_values, expected_rows in (
        ("warmup", bundle["warmup_pointers"], warmup),
        ("scored", bundle["scored_pointers"], scored),
    ):
        if type(pointer_values) is not list or len(pointer_values) != len(
            expected_rows
        ):
            raise ValueError(f"formal distributed {label} pointer coverage differs")
        for pointer, expected in zip(pointer_values, expected_rows, strict=True):
            if expected.request_id in pointers_by_request:
                raise ValueError("formal distributed native ITL request repeats")
            pointers_by_request[expected.request_id] = _validate_pointer(
                pointer,
                expected=expected,
            )
    digests = tuple(value[0] for value in pointers_by_request.values())
    if len(digests) != len(set(digests)):
        raise ValueError("formal distributed native ITL pointers repeat")
    return pointers_by_request


def _validate_physical_launch_evidence(
    *,
    admission: CanonicalJsonProofBinding,
    launch_consumption: CanonicalJsonProofBinding,
    budget_consumption: CanonicalJsonProofBinding,
    plan_path: str,
    expected_registry_sha256: str,
) -> int:
    """Deep-open the staged/single-operator launch evidence union."""

    value = admission.reopen()
    if value.get("kind") == "formal_single_operator_admission":
        from lightcone_spec.orchestration.formal_single_operator_admission import (
            validate_formal_single_operator_admission,
            validate_formal_single_operator_admission_consumption,
        )

        artifact = validate_formal_single_operator_admission(
            admission.absolute_path,
            plan_path=plan_path,
        )
        consumption = validate_formal_single_operator_admission_consumption(
            launch_consumption.absolute_path,
            admission_path=admission.absolute_path,
            plan_path=plan_path,
        )
        if (
            artifact.registry_sha256 != expected_registry_sha256
            or budget_consumption != launch_consumption
        ):
            raise ValueError("single-operator launch evidence identity differs")
        return consumption.consumed_ns
    from lightcone_spec.orchestration.formal_launch_admission import (
        FormalStageLaunchConsumption,
        validate_formal_stage_launch_evidence_lineage,
    )

    consumption = FormalStageLaunchConsumption.from_dict(launch_consumption.reopen())
    validate_formal_stage_launch_evidence_lineage(
        admission=admission,
        launch_consumption=launch_consumption,
        budget_consumption=budget_consumption,
        run_plan_path=plan_path,
        current_ns=max(consumption.consumed_ns, 1),
    )
    return consumption.consumed_ns


def _validate_distributed_raw(
    *,
    plan_binding: CanonicalJsonProofBinding,
    receipt_binding: CanonicalJsonProofBinding,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> _ValidatedDistributedRaw:
    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    _sha("formal distributed registry", expected_registry_sha256)
    if plan.sha256 != plan_binding.semantic_sha256:
        raise ValueError("formal distributed plan path identity differs")
    if plan.topology_mode not in {"tp2_dp1", "tp1_dp2"}:
        raise ValueError("formal distributed proof requires TP2 or DP2")
    if plan.inventory_sha256 != expected_inventory_sha256:
        raise ValueError("formal distributed plan uses another inventory")
    receipt = _object(
        "formal distributed run receipt",
        receipt_binding.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "plan_sha256",
            "execution_binding_sha256",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "launch_manifest",
            "request_schedule_receipt",
            "terminal",
            "native_itl_pointers",
            "formal_gang_terminal",
            "lifecycle_timing",
            "before_gpu_snapshot",
            "ready_gpu_snapshot",
            "after_gpu_snapshot",
            "server_log",
            "server_stdout",
            "server_stderr",
            "junit",
            "server_process_id",
            "process_exit_code",
            "cleanup_kind",
            "process_group_empty",
            "phase_edges_ns",
        },
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "unsigned_formal_gang_physical_run_receipt"
        or receipt["protocol_sha256"]
        != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or receipt["formal_execution_authorized"] is not False
        or receipt["plan_sha256"] != plan.sha256
        or receipt["execution_binding_sha256"] != plan.execution_binding_sha256
        or receipt["process_group_empty"] is not True
        or receipt["launch_manifest"] != plan.launch_manifest.to_dict()
        or receipt["request_schedule_receipt"]
        != plan.request_schedule_receipt.to_dict()
    ):
        raise ValueError("formal distributed run receipt identity/status differs")
    plan.launch_manifest.reopen()
    schedule = _reopen_schedule_receipt(plan.request_schedule_receipt)
    launch_admission = CanonicalJsonProofBinding.from_dict(
        receipt["formal_launch_admission"]
    )
    launch_consumption = CanonicalJsonProofBinding.from_dict(
        receipt["formal_launch_consumption"]
    )
    budget_consumption = CanonicalJsonProofBinding.from_dict(
        receipt["budget_consumption"]
    )
    _validate_physical_launch_evidence(
        admission=launch_admission,
        launch_consumption=launch_consumption,
        budget_consumption=budget_consumption,
        plan_path=plan_binding.absolute_path,
        expected_registry_sha256=expected_registry_sha256,
    )
    request_terminal = CanonicalJsonProofBinding.from_dict(receipt["terminal"])
    gang_terminal = CanonicalJsonProofBinding.from_dict(receipt["formal_gang_terminal"])
    pointer_bundle = CanonicalJsonProofBinding.from_dict(receipt["native_itl_pointers"])
    lifecycle_binding = CanonicalJsonProofBinding.from_dict(receipt["lifecycle_timing"])
    before_binding = CanonicalJsonProofBinding.from_dict(receipt["before_gpu_snapshot"])
    ready_binding = CanonicalJsonProofBinding.from_dict(receipt["ready_gpu_snapshot"])
    after_binding = CanonicalJsonProofBinding.from_dict(receipt["after_gpu_snapshot"])
    log_binding = EvidenceFileBinding.from_dict(
        receipt["server_log"],
        label="formal distributed server log",
    )
    log_binding.reopen(label="formal distributed server log")
    stdout_binding = EvidenceFileBinding.from_dict(
        receipt["server_stdout"],
        label="formal distributed server stdout",
    )
    stdout_binding.reopen(label="formal distributed server stdout")
    stderr_binding = EvidenceFileBinding.from_dict(
        receipt["server_stderr"],
        label="formal distributed server stderr",
    )
    stderr_binding.reopen(label="formal distributed server stderr")
    junit_binding = EvidenceFileBinding.from_dict(
        receipt["junit"],
        label="formal distributed JUnit",
    )
    junit_binding.reopen(label="formal distributed JUnit")
    if (
        log_binding.absolute_path != plan.server_log_output_path
        or stdout_binding.absolute_path != plan.server_stdout_output_path
        or stderr_binding.absolute_path != plan.server_stderr_output_path
        or junit_binding.absolute_path != plan.junit_output_path
    ):
        raise ValueError("formal distributed fixed run artifact path differs")
    lifecycle = lifecycle_binding.reopen()
    before = before_binding.reopen()
    ready = ready_binding.reopen()
    after = after_binding.reopen()
    server_process_id = receipt["server_process_id"]
    process_exit_code = receipt["process_exit_code"]
    cleanup_kind = receipt["cleanup_kind"]
    phase_edges_ns = receipt["phase_edges_ns"]
    ready_processes = ready.get("compute_process_rows")
    if type(ready_processes) is not list or any(
        type(row) is not dict for row in ready_processes
    ):
        raise ValueError("formal distributed ready GPU processes are malformed")
    if (
        lifecycle.get("kind") != "unsigned_formal_gang_lifecycle_timing"
        or lifecycle.get("protocol_sha256")
        != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or lifecycle.get("formal_execution_authorized") is not False
        or lifecycle.get("plan_sha256") != plan.sha256
        or lifecycle.get("formal_launch_admission") != launch_admission.to_dict()
        or lifecycle.get("formal_launch_consumption") != launch_consumption.to_dict()
        or lifecycle.get("budget_consumption") != budget_consumption.to_dict()
        or lifecycle.get("topology_mode") != plan.topology_mode
        or lifecycle.get("terminal_sha256") != request_terminal.semantic_sha256
        or lifecycle.get("native_itl_pointer_sha256") != pointer_bundle.semantic_sha256
        or lifecycle.get("formal_gang_terminal_sha256") != gang_terminal.semantic_sha256
        or lifecycle.get("phase_edges_ns") != receipt["phase_edges_ns"]
        or type(server_process_id) is not int
        or server_process_id < 1
        or type(process_exit_code) is not int
        or process_exit_code not in {0, -15}
        or cleanup_kind not in {"already_exited_clean", "sigterm_clean"}
        or type(phase_edges_ns) is not dict
        or type(phase_edges_ns.get("process_exited_ns")) is not int
        or type(phase_edges_ns.get("process_group_empty_checked_ns")) is not int
        or phase_edges_ns["process_group_empty_checked_ns"]
        < phase_edges_ns["process_exited_ns"]
        or any(
            snapshot.get("kind") != "unsigned_pinned_sglang_gpu_process_snapshot"
            or snapshot.get("inventory_sha256") != plan.inventory_sha256
            or snapshot.get("gpu_uuids") != list(plan.gpu_uuids)
            for snapshot in (before, ready, after)
        )
        or tuple(snapshot.get("phase") for snapshot in (before, ready, after))
        != ("before", "ready", "after")
        or before.get("compute_process_rows") != []
        or after.get("compute_process_rows") != []
        or ready.get("server_process_group_ids")
        != [server_process_id] * len(plan.gpu_uuids)
        or {row.get("gpu_uuid") for row in ready_processes} != set(plan.gpu_uuids)
        or any(
            row.get("process_group_id") != server_process_id for row in ready_processes
        )
    ):
        raise ValueError("formal distributed lifecycle/GPU lineage differs")
    terminal = _object(
        "formal distributed request terminal",
        reopen_scalable_formal_gang_request_terminal(request_terminal.reopen()),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "plan_sha256",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "capability_sha256",
            "begin_sha256",
            "reset_sha256",
            "finalize_sha256",
            "warmup_requests",
            "scored_requests",
        },
    )
    gang = reopen_scalable_formal_gang_terminal(gang_terminal.reopen())
    if type(gang) is not dict:
        raise TypeError("formal distributed gang terminal must be an object")
    aggregate = _sha("formal distributed gang aggregate", gang.get("aggregate_sha256"))
    unsigned_gang = dict(gang)
    unsigned_gang.pop("aggregate_sha256")
    if (
        content_sha256(unsigned_gang) != aggregate
        or terminal["schema_version"] != 1
        or terminal["kind"] != "unsigned_formal_gang_request_terminal"
        or terminal["protocol_sha256"]
        != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or terminal["formal_execution_authorized"] is not False
        or terminal["plan_sha256"] != plan.sha256
        or CanonicalJsonProofBinding.from_dict(terminal["formal_launch_admission"])
        != launch_admission
        or CanonicalJsonProofBinding.from_dict(terminal["formal_launch_consumption"])
        != launch_consumption
        or CanonicalJsonProofBinding.from_dict(terminal["budget_consumption"])
        != budget_consumption
        or terminal["finalize_sha256"] != aggregate
        or gang.get("kind") != "sglang_formal_gang_all_rank_terminal"
        or gang.get("protocol_sha256") != FORMAL_GANG_SERVING_PROTOCOL_SHA256
        or gang.get("action") != "formal_gang_finalize"
        or gang.get("topology") != plan.topology_mode
        or gang.get("decision") != "COMMITTED"
        or gang.get("published_ranks") != [0, 1]
        or gang.get("reason_code") is not None
        or gang.get("cross_replica_gradient_collective") is not False
    ):
        raise ValueError("formal distributed terminal/gang lineage differs")
    rank_values = gang.get("rank_terminals")
    rank_digests = gang.get("rank_terminal_sha256s")
    if (
        type(rank_values) is not list
        or type(rank_digests) is not list
        or len(rank_values) != 2
        or len(rank_digests) != 2
    ):
        raise ValueError("formal distributed rank terminal coverage differs")
    scored_schedule = tuple(
        row
        for row in formal_serving_request_schedule_rows(schedule)
        if row.phase == "scored"
    )
    warmup_requests = _phase_requests(
        terminal["warmup_requests"],
        schedule=schedule,
        phase="warmup",
    )
    requests = _phase_requests(
        terminal["scored_requests"],
        schedule=schedule,
        phase="scored",
    )
    native_itl_by_request = _validate_pointer_bundle(
        reopen_scalable_formal_gang_itl_bundle(pointer_bundle.reopen()),
        plan=plan,
        warmup=warmup_requests,
        scored=requests,
        launch_admission=launch_admission,
        launch_consumption=launch_consumption,
        budget_consumption=budget_consumption,
    )
    by_request = {row.request_id: row for row in requests}
    full_schedule_rows = {
        phase: [
            {
                "request_id": row.request.request_id,
                "cohort_sha256": row.request.cohort_sha256,
                "routed_dp_rank": row.routed_dp_rank,
            }
            for row in formal_serving_request_schedule_rows(schedule)
            if row.phase == phase
        ]
        for phase in ("warmup", "scored")
    }
    full_schedule_sha256 = content_sha256(full_schedule_rows)
    sticky_sha256 = content_sha256(
        sorted(
            {
                row.request.cohort_sha256: row.routed_dp_rank
                for row in formal_serving_request_schedule_rows(schedule)
            }.items()
        )
    )
    updates_by_rank: list[tuple[FormalDistributedTerminalUpdateResult, ...]] = []
    performance_by_rank: list[dict[str, object]] = []
    observed_rank_digests: list[str] = []
    for expected_rank, raw_rank in enumerate(rank_values):
        if type(raw_rank) is not dict:
            raise TypeError("formal distributed rank terminal is not an object")
        rank = dict(raw_rank)
        digest = _rank_terminal_digest(rank, require_client_lifecycle=False)
        if digest != rank_digests[expected_rank]:
            raise ValueError("formal distributed rank terminal list differs")
        observed_rank_digests.append(digest)
        native_state = rank.get("native_state")
        native_sha = rank.get("native_state_sha256")
        local_schedule = tuple(
            row
            for row in scored_schedule
            if plan.topology_mode == "tp2_dp1" or row.routed_dp_rank == expected_rank
        )
        local_routes = [
            {
                "request_id": row.request.request_id,
                "cohort_sha256": row.request.cohort_sha256,
                "routed_dp_rank": row.routed_dp_rank,
            }
            for row in local_schedule
        ]
        local_request_ids = [row.request.request_id for row in local_schedule]
        if (
            rank.get("schema_version") != 1
            or rank.get("kind") != "sglang_formal_gang_rank_terminal"
            or rank.get("hook") != "sglang.lightcone_formal_gang_serving.v1"
            or rank.get("protocol_sha256") != FORMAL_GANG_SERVING_PROTOCOL_SHA256
            or rank.get("topology") != plan.topology_mode
            or rank.get("rank") != expected_rank
            or rank.get("world_size") != 2
            or rank.get("gpu_uuid") != plan.gpu_uuids[expected_rank]
            or rank.get("status") != "COMPLETE"
            or rank.get("reason_code") is not None
            or rank.get("execution_plan_sha256")
            != plan.native_terminal_binding.execution_plan_sha256
            or rank.get("rank_config_sha256")
            != plan.native_terminal_binding.rank_config_sha256
            or rank.get("run_nonce_sha256")
            != plan.native_terminal_binding.run_nonce_sha256
            or rank.get("method") != plan.method
            or rank.get("phase") != "scored"
            or rank.get("full_schedule_sha256") != full_schedule_sha256
            or rank.get("local_request_routes_sha256") != content_sha256(local_routes)
            or rank.get("sticky_cohort_routes_sha256") != sticky_sha256
            or rank.get("expected_request_ids_sha256")
            != content_sha256(local_request_ids)
            or type(native_state) is not dict
            or set(native_state)
            != {
                "scheduler",
                "round_rows",
                "update_rows",
                "performance_counters",
                "historical_kv_source_versions",
                "adaptation",
            }
            or content_sha256(native_state) != native_sha
        ):
            raise ValueError("formal distributed rank/native identity differs")
        expected_terminals = [
            {
                "request_id": by_request[row.request.request_id].request_id,
                "input_token_ids": list(
                    by_request[row.request.request_id].input_token_ids
                ),
                "output_token_ids": list(
                    by_request[row.request.request_id].output_token_ids
                ),
                "native_itl_semantics": (
                    "scheduler_committed_token_at_result_processor_v1"
                ),
                "native_itl_event_count": len(
                    by_request[row.request.request_id].output_token_ids
                ),
                "native_itl_events_sha256": native_itl_by_request[
                    row.request.request_id
                ][1],
                "terminal_status": "completed",
                "terminal_reason": by_request[row.request.request_id].terminal_reason,
            }
            for row in local_schedule
        ]
        if rank.get("request_terminals") != expected_terminals or rank.get(
            "request_terminal_sha256s"
        ) != [content_sha256(row) for row in expected_terminals]:
            raise ValueError("formal distributed rank request content differs")
        performance = native_state.get("performance_counters")
        update_rows = native_state.get("update_rows")
        if (
            type(performance) is not dict
            or type(update_rows) is not list
            or native_state.get("round_rows") is None
            or type(native_state.get("round_rows")) is not list
            or type(native_state.get("scheduler")) is not dict
            or type(native_state.get("historical_kv_source_versions")) is not dict
        ):
            raise ValueError("formal distributed rank native evidence is incomplete")
        allocation_free = plan.method in {"target_only", "static"}
        if allocation_free != (not update_rows):
            raise ValueError("formal distributed update allocation identity differs")
        if plan.topology_mode == "tp2_dp1":
            adaptation = native_state.get("adaptation")
            if type(adaptation) is not dict:
                raise ValueError("formal TP2 adaptation evidence is unavailable")
            publication = adaptation.get("tp2_last_publication_receipt")
            if plan.method not in {"target_only", "static"} and (
                type(publication) is not dict
                or publication.get("decision") != "COMMITTED"
                or publication.get("published_ranks") != [0, 1]
            ):
                raise ValueError("formal TP2 adaptation publication is not atomic")
        else:
            adaptation = native_state.get("adaptation")
            replica = (
                None
                if type(adaptation) is not dict
                else adaptation.get("dp2_replica_state")
            )
            if (
                type(replica) is not dict
                or replica.get("cross_replica_gradient_collective") is not False
                or replica.get("cross_replica_optimizer_state") is not False
            ):
                raise ValueError("formal DP2 replica isolation is not exact")
        updates_by_rank.append(
            tuple(
                FormalDistributedTerminalUpdateResult(
                    rank=expected_rank,
                    source_row=dict(row),
                )
                for row in update_rows
                if type(row) is dict
            )
        )
        if len(updates_by_rank[-1]) != len(update_rows):
            raise TypeError("formal distributed update row is not an object")
        performance_by_rank.append(dict(performance))
    if plan.topology_mode == "tp2_dp1":
        for field in (
            "target_calls",
            "updates_launched",
            "updates_published",
            "exactness_violations",
            "version_mismatches",
            "fallbacks",
            "nonfinite_updates",
            "oom_events",
            "retractions",
            "communicator_failures",
        ):
            if performance_by_rank[0].get(field) != performance_by_rank[1].get(field):
                raise ValueError("formal TP2 rank performance identity differs")
        left_updates = updates_by_rank[0]
        right_updates = updates_by_rank[1]
        if len(left_updates) != len(right_updates):
            raise ValueError("formal TP2 update coverage differs by rank")
        logical_fields = (
            "update_index",
            "cohort_sha256",
            "cohort_epoch",
            "parameter_layout_sha256",
            "source_round",
            "source_version",
            "request_ids",
            "prefix_len_before",
            "optimizer_step",
            "published_version",
            "status",
            "reconstruction_ok",
        )
        if any(
            any(
                left.source_row[field] != right.source_row[field]
                for field in logical_fields
            )
            for left, right in zip(left_updates, right_updates, strict=True)
        ):
            raise ValueError("formal TP2 logical update identity differs by rank")
    binding = FormalDistributedTerminalExternalControlBinding(
        schema_version=1,
        kind="formal_distributed_terminal_external_control_binding",
        plan_raw_sha256=plan_binding.raw_sha256,
        plan_semantic_sha256=plan_binding.semantic_sha256,
        run_receipt_raw_sha256=receipt_binding.raw_sha256,
        run_receipt_semantic_sha256=receipt_binding.semantic_sha256,
        request_terminal_raw_sha256=request_terminal.raw_sha256,
        request_terminal_semantic_sha256=request_terminal.semantic_sha256,
        gang_terminal_raw_sha256=gang_terminal.raw_sha256,
        gang_terminal_semantic_sha256=gang_terminal.semantic_sha256,
        pointer_bundle_raw_sha256=pointer_bundle.raw_sha256,
        pointer_bundle_semantic_sha256=pointer_bundle.semantic_sha256,
        lifecycle_timing_raw_sha256=lifecycle_binding.raw_sha256,
        lifecycle_timing_semantic_sha256=lifecycle_binding.semantic_sha256,
        launch_admission_raw_sha256=launch_admission.raw_sha256,
        launch_admission_semantic_sha256=launch_admission.semantic_sha256,
        launch_consumption_raw_sha256=launch_consumption.raw_sha256,
        launch_consumption_semantic_sha256=launch_consumption.semantic_sha256,
        budget_consumption_raw_sha256=budget_consumption.raw_sha256,
        budget_consumption_semantic_sha256=budget_consumption.semantic_sha256,
        inventory_sha256=expected_inventory_sha256,
        registry_sha256=expected_registry_sha256,
        execution_plan_sha256=plan.native_terminal_binding.execution_plan_sha256,
        rank_config_sha256=plan.native_terminal_binding.rank_config_sha256,
        run_nonce_sha256=plan.native_terminal_binding.run_nonce_sha256,
        topology_mode=plan.topology_mode,
    )
    return _ValidatedDistributedRaw(
        plan=plan,
        binding=binding,
        requests=requests,
        updates_by_rank=(updates_by_rank[0], updates_by_rank[1]),
        performance_by_rank=(performance_by_rank[0], performance_by_rank[1]),
        rank_terminal_sha256s=(observed_rank_digests[0], observed_rank_digests[1]),
        aggregate_sha256=aggregate,
    )


def _validate_current_distributed_raw(
    *,
    plan_binding: CanonicalJsonProofBinding,
    receipt_binding: CanonicalJsonProofBinding,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> FormalServingRunPlan:
    """Deep-open current trusted TP2/DP2 evidence with client denominators.

    Current single-operator runs intentionally carry no offline launch
    attestation.  They therefore cannot reuse the legacy external-control
    binding builder, but every raw source is still path-bound and verified
    here before the run can become a scientific COMPLETE attempt.
    """

    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
        RegisteredServingRequestLifecycle,
    )

    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    _sha("current distributed registry", expected_registry_sha256)
    if (
        plan.schema_version != 4
        or plan.sha256 != plan_binding.semantic_sha256
        or plan.topology_mode not in {"tp2_dp1", "tp1_dp2"}
        or plan.inventory_sha256 != expected_inventory_sha256
        or type(plan.serving_execution_policy) is not RegisteredServingExecutionPolicy
    ):
        raise ValueError("current distributed plan identity differs")
    receipt = _object(
        "current distributed run receipt",
        receipt_binding.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "plan_sha256",
            "execution_binding_sha256",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "launch_manifest",
            "request_schedule_receipt",
            "serving_execution_policy",
            "client_request_lifecycle",
            "terminal",
            "native_itl_pointers",
            "formal_gang_terminal",
            "lifecycle_timing",
            "before_gpu_snapshot",
            "ready_gpu_snapshot",
            "after_gpu_snapshot",
            "server_log",
            "server_stdout",
            "server_stderr",
            "junit",
            "server_process_id",
            "process_exit_code",
            "cleanup_kind",
            "process_group_empty",
            "phase_edges_ns",
        },
    )
    policy = RegisteredServingExecutionPolicy.from_dict(
        receipt["serving_execution_policy"]
    )
    if (
        receipt["schema_version"] != 2
        or receipt["kind"] != "unsigned_formal_gang_physical_run_receipt"
        or receipt["protocol_sha256"]
        != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or receipt["formal_execution_authorized"] is not False
        or receipt["plan_sha256"] != plan.sha256
        or receipt["execution_binding_sha256"] != plan.execution_binding_sha256
        or receipt["process_group_empty"] is not True
        or receipt["launch_manifest"] != plan.launch_manifest.to_dict()
        or receipt["request_schedule_receipt"]
        != plan.request_schedule_receipt.to_dict()
        or policy != plan.serving_execution_policy
    ):
        raise ValueError("current distributed run receipt identity/status differs")
    launch_lineage = (
        receipt["formal_launch_admission"],
        receipt["formal_launch_consumption"],
        receipt["budget_consumption"],
    )
    if len({value is None for value in launch_lineage}) != 1:
        raise ValueError("current distributed launch lineage is partial")
    if launch_lineage[0] is not None:
        admission = CanonicalJsonProofBinding.from_dict(launch_lineage[0])
        consumption = CanonicalJsonProofBinding.from_dict(launch_lineage[1])
        budget = CanonicalJsonProofBinding.from_dict(launch_lineage[2])
        _validate_physical_launch_evidence(
            admission=admission,
            launch_consumption=consumption,
            budget_consumption=budget,
            plan_path=plan_binding.absolute_path,
            expected_registry_sha256=expected_registry_sha256,
        )

    schedule = _reopen_schedule_receipt(plan.request_schedule_receipt)
    schedule_rows = tuple(formal_serving_request_schedule_rows(schedule))
    expected_ids = tuple(row.request.request_id for row in schedule_rows)
    client_binding = CanonicalJsonProofBinding.from_dict(
        receipt["client_request_lifecycle"]
    )
    client_values = reopen_scalable_client_request_lifecycle(
        client_binding,
        expected_run_binding_sha256=canonical_sha256(
            plan.native_terminal_binding.begin_payload()
        ),
        expected_execution_policy_sha256=policy.sha256,
    )
    client_rows = tuple(
        RegisteredServingRequestLifecycle(
            **_object(
                "current distributed client lifecycle",
                value,
                set(RegisteredServingRequestLifecycle.__dataclass_fields__),
            )
        )
        for value in client_values
    )
    if tuple(row.request_id for row in client_rows) != expected_ids:
        raise ValueError("current distributed client lifecycle coverage differs")
    if any(
        row.phase == "warmup"
        and (
            not row.offered
            or not row.submitted_to_server
            or row.outcome_status != "completed"
            or row.native_terminal_status != "completed"
        )
        for row in client_rows
    ):
        raise ValueError("current distributed warmup is not strict completed")
    client_by_id = {row.request_id: row for row in client_rows}
    client_rows_sha256 = canonical_sha256(client_values)

    terminal_binding = CanonicalJsonProofBinding.from_dict(receipt["terminal"])
    terminal = _object(
        "current distributed request terminal",
        reopen_scalable_formal_gang_request_terminal(terminal_binding.reopen()),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "plan_sha256",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "capability_sha256",
            "begin_sha256",
            "reset_sha256",
            "finalize_sha256",
            "warmup_requests",
            "scored_requests",
        },
    )
    if (
        terminal["schema_version"] != 1
        or terminal["kind"] != "unsigned_formal_gang_request_terminal"
        or terminal["protocol_sha256"]
        != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or terminal["formal_execution_authorized"] is not False
        or terminal["plan_sha256"] != plan.sha256
        or tuple(
            terminal[name]
            for name in (
                "formal_launch_admission",
                "formal_launch_consumption",
                "budget_consumption",
            )
        )
        != launch_lineage
    ):
        raise ValueError("current distributed request terminal lineage differs")
    native_by_id: dict[str, dict[str, object]] = {}
    for phase in ("warmup", "scored"):
        raw_phase = terminal[f"{phase}_requests"]
        expected_phase = tuple(row for row in schedule_rows if row.phase == phase)
        if type(raw_phase) is not list or len(raw_phase) != len(expected_phase):
            raise ValueError("current distributed terminal phase coverage differs")
        for raw, sealed in zip(raw_phase, expected_phase, strict=True):
            row = _object(
                "current distributed terminal request",
                raw,
                {
                    "request_id",
                    "input_token_ids",
                    "output_token_ids",
                    "terminal_status",
                    "terminal_reason",
                    "submitted_to_server",
                },
            )
            request_id = sealed.request.request_id
            lifecycle = client_by_id[request_id]
            outputs = row["output_token_ids"]
            if (
                row["request_id"] != request_id
                or tuple(row["input_token_ids"]) != sealed.request.input_token_ids
                or row["submitted_to_server"] is not lifecycle.submitted_to_server
                or (outputs is not None and type(outputs) is not list)
            ):
                raise ValueError("current distributed terminal request differs")
            if lifecycle.submitted_to_server:
                if row["terminal_status"] != lifecycle.native_terminal_status:
                    raise ValueError(
                        "current distributed native terminal status differs"
                    )
            elif row["terminal_status"] != (
                "rejected" if not lifecycle.offered else lifecycle.outcome_status
            ):
                raise ValueError("current distributed non-submitted terminal differs")
            native_by_id[request_id] = row

    pointer_binding = CanonicalJsonProofBinding.from_dict(
        receipt["native_itl_pointers"]
    )
    pointer_bundle = _object(
        "current distributed native ITL pointer bundle",
        reopen_scalable_formal_gang_itl_bundle(pointer_binding.reopen()),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "plan_sha256",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "warmup_pointers",
            "scored_pointers",
        },
    )
    if (
        pointer_bundle["schema_version"] != 1
        or pointer_bundle["kind"] != "unsigned_formal_gang_native_itl_pointer_bundle"
        or pointer_bundle["protocol_sha256"]
        != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or pointer_bundle["formal_execution_authorized"] is not False
        or pointer_bundle["plan_sha256"] != plan.sha256
        or tuple(
            pointer_bundle[name]
            for name in (
                "formal_launch_admission",
                "formal_launch_consumption",
                "budget_consumption",
            )
        )
        != launch_lineage
    ):
        raise ValueError("current distributed pointer bundle lineage differs")
    pointer_by_id: dict[str, tuple[str, str]] = {}
    for phase in ("warmup", "scored"):
        raw_pointers = pointer_bundle[f"{phase}_pointers"]
        if type(raw_pointers) is not list:
            raise TypeError("current distributed pointers must be arrays")
        for pointer in raw_pointers:
            if type(pointer) is not dict or type(pointer.get("request_id")) is not str:
                raise TypeError("current distributed pointer is malformed")
            request_id = str(pointer["request_id"])
            native = native_by_id.get(request_id)
            lifecycle = client_by_id.get(request_id)
            if (
                native is None
                or lifecycle is None
                or lifecycle.phase != phase
                or lifecycle.outcome_status != "completed"
                or request_id in pointer_by_id
            ):
                raise ValueError("current distributed pointer request differs")
            outputs = native["output_token_ids"]
            if type(outputs) is not list or not outputs:
                raise ValueError("completed distributed request lacks output")
            expected = FormalDistributedTerminalRequestResult(
                request_id=request_id,
                input_token_ids=tuple(native["input_token_ids"]),
                output_token_ids=tuple(outputs),
                terminal_status="completed",
                terminal_reason=str(native["terminal_reason"]),
                submitted_to_server=True,
                request_sha256=content_sha256(native),
            )
            pointer_by_id[request_id] = _validate_pointer(
                pointer,
                expected=expected,
            )
    completed_ids = {
        row.request_id for row in client_rows if row.outcome_status == "completed"
    }
    if set(pointer_by_id) != completed_ids:
        raise ValueError("current distributed completed pointer coverage differs")

    gang_binding = CanonicalJsonProofBinding.from_dict(receipt["formal_gang_terminal"])
    gang = reopen_scalable_formal_gang_terminal(gang_binding.reopen())
    if type(gang) is not dict:
        raise TypeError("current distributed gang terminal must be an object")
    aggregate = gang.get("aggregate_sha256")
    unsigned_gang = dict(gang)
    unsigned_gang.pop("aggregate_sha256", None)
    if (
        type(aggregate) is not str
        or canonical_sha256(unsigned_gang) != aggregate
        or gang.get("protocol_sha256") != FORMAL_GANG_SERVING_PROTOCOL_SHA256
        or gang.get("action") != "formal_gang_finalize"
        or gang.get("topology") != plan.topology_mode
        or gang.get("decision") != "COMMITTED"
        or gang.get("published_ranks") != [0, 1]
        or gang.get("reason_code") is not None
        or terminal["finalize_sha256"] != aggregate
    ):
        raise ValueError("current distributed gang aggregate differs")
    rank_values = gang.get("rank_terminals")
    rank_digests = gang.get("rank_terminal_sha256s")
    if (
        type(rank_values) is not list
        or type(rank_digests) is not list
        or len(rank_values) != 2
        or len(rank_digests) != 2
    ):
        raise ValueError("current distributed rank coverage differs")
    full_schedule = {
        phase: [
            {
                "request_id": row.request.request_id,
                "cohort_sha256": row.request.cohort_sha256,
                "routed_dp_rank": row.routed_dp_rank,
            }
            for row in schedule_rows
            if row.phase == phase
        ]
        for phase in ("warmup", "scored")
    }
    full_schedule_sha256 = content_sha256(full_schedule)
    sticky_sha256 = content_sha256(
        sorted(
            {
                row.request.cohort_sha256: row.routed_dp_rank for row in schedule_rows
            }.items()
        )
    )
    scored_schedule = tuple(row for row in schedule_rows if row.phase == "scored")
    for rank_index, raw_rank in enumerate(rank_values):
        rank = _object(
            "current distributed rank terminal",
            raw_rank,
            _CURRENT_RANK_TERMINAL_FIELDS,
        )
        digest = _rank_terminal_digest(rank, require_client_lifecycle=True)
        local_schedule = tuple(
            row
            for row in scored_schedule
            if plan.topology_mode == "tp2_dp1" or row.routed_dp_rank == rank_index
        )
        local_routes = [
            {
                "request_id": row.request.request_id,
                "cohort_sha256": row.request.cohort_sha256,
                "routed_dp_rank": row.routed_dp_rank,
            }
            for row in local_schedule
        ]
        local_ids = tuple(row.request.request_id for row in local_schedule)
        submitted_ids = tuple(
            request_id
            for request_id in local_ids
            if client_by_id[request_id].submitted_to_server
        )
        non_submitted_ids = tuple(
            request_id
            for request_id in local_ids
            if not client_by_id[request_id].submitted_to_server
        )
        native_state = rank["native_state"]
        if (
            digest != rank_digests[rank_index]
            or rank["protocol_sha256"] != FORMAL_GANG_SERVING_PROTOCOL_SHA256
            or rank["topology"] != plan.topology_mode
            or rank["rank"] != rank_index
            or rank["world_size"] != 2
            or rank["gpu_uuid"] != plan.gpu_uuids[rank_index]
            or rank["execution_plan_sha256"]
            != plan.native_terminal_binding.execution_plan_sha256
            or rank["rank_config_sha256"]
            != plan.native_terminal_binding.rank_config_sha256
            or rank["run_nonce_sha256"] != plan.native_terminal_binding.run_nonce_sha256
            or rank["method"] != plan.method
            or rank["phase"] != "scored"
            or rank["full_schedule_sha256"] != full_schedule_sha256
            or rank["local_request_routes_sha256"] != content_sha256(local_routes)
            or rank["sticky_cohort_routes_sha256"] != sticky_sha256
            or rank["expected_request_ids_sha256"] != content_sha256(list(local_ids))
            or rank["client_lifecycle_sha256"] != client_rows_sha256
            or rank["non_submitted_request_ids_sha256"]
            != content_sha256(list(non_submitted_ids))
            or rank["status"] != "COMPLETE"
            or rank["reason_code"] is not None
            or type(native_state) is not dict
            or content_sha256(native_state) != rank["native_state_sha256"]
        ):
            raise ValueError("current distributed rank identity differs")
        rank_requests = rank["request_terminals"]
        rank_request_digests = rank["request_terminal_sha256s"]
        if (
            type(rank_requests) is not list
            or type(rank_request_digests) is not list
            or tuple(row.get("request_id") for row in rank_requests) != submitted_ids
            or [content_sha256(row) for row in rank_requests] != rank_request_digests
        ):
            raise ValueError("current distributed rank request coverage differs")
        for rank_request in rank_requests:
            if type(rank_request) is not dict:
                raise TypeError("current distributed rank request is malformed")
            request_id = str(rank_request.get("request_id"))
            native = native_by_id[request_id]
            outputs = native["output_token_ids"]
            if (
                rank_request.get("input_token_ids") != native["input_token_ids"]
                or rank_request.get("output_token_ids") != outputs
                or rank_request.get("terminal_status") != native["terminal_status"]
                or rank_request.get("terminal_reason") != native["terminal_reason"]
                or rank_request.get("native_itl_event_count")
                != (0 if outputs is None else len(outputs))
            ):
                raise ValueError("current distributed rank request content differs")
            if (
                request_id in pointer_by_id
                and rank_request.get("native_itl_events_sha256")
                != pointer_by_id[request_id][1]
            ):
                raise ValueError("current distributed rank/pointer ITL differs")
            _sha(
                "current distributed rank ITL events",
                rank_request.get("native_itl_events_sha256"),
            )
        if set(native_state) != {
            "scheduler",
            "round_rows",
            "update_rows",
            "performance_counters",
            "historical_kv_source_versions",
            "adaptation",
        }:
            raise ValueError("current distributed native state is incomplete")

    lifecycle_binding = CanonicalJsonProofBinding.from_dict(receipt["lifecycle_timing"])
    lifecycle = lifecycle_binding.reopen()
    if (
        lifecycle.get("serving_execution_policy_sha256") != policy.sha256
        or lifecycle.get("client_request_lifecycle_sha256")
        != client_binding.semantic_sha256
        or lifecycle.get("terminal_sha256") != terminal_binding.semantic_sha256
        or lifecycle.get("native_itl_pointer_sha256") != pointer_binding.semantic_sha256
        or lifecycle.get("formal_gang_terminal_sha256") != gang_binding.semantic_sha256
    ):
        raise ValueError("current distributed lifecycle lineage differs")
    for label, receipt_name, expected_path in (
        ("server log", "server_log", plan.server_log_output_path),
        ("server stdout", "server_stdout", plan.server_stdout_output_path),
        ("server stderr", "server_stderr", plan.server_stderr_output_path),
        ("JUnit", "junit", plan.junit_output_path),
    ):
        binding = EvidenceFileBinding.from_dict(receipt[receipt_name], label=label)
        binding.reopen(label=label)
        if binding.absolute_path != expected_path:
            raise ValueError("current distributed fixed output path differs")
    for receipt_name, expected_path in (
        ("before_gpu_snapshot", plan.before_gpu_snapshot_output_path),
        ("ready_gpu_snapshot", plan.ready_gpu_snapshot_output_path),
        ("after_gpu_snapshot", plan.after_gpu_snapshot_output_path),
    ):
        binding = CanonicalJsonProofBinding.from_dict(receipt[receipt_name])
        if binding.absolute_path != expected_path:
            raise ValueError("current distributed GPU snapshot path differs")
        binding.reopen()
    if (
        type(receipt["server_process_id"]) is not int
        or receipt["server_process_id"] < 1
        or receipt["process_exit_code"] not in {0, -15}
        or receipt["cleanup_kind"] not in {"already_exited_clean", "sigterm_clean"}
        or type(receipt["phase_edges_ns"]) is not dict
    ):
        raise ValueError("current distributed process terminal differs")
    return plan


def validate_formal_distributed_physical_outcome(
    *,
    plan_path: str,
    run_receipt_path: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> FormalDistributedPhysicalOutcome:
    """Deep-reopen a completed TP2/DP2 run for single-operator provenance."""

    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    receipt_binding = CanonicalJsonProofBinding.bind(run_receipt_path)
    receipt = receipt_binding.reopen()
    if receipt.get("schema_version") == 2:
        _validate_current_distributed_raw(
            plan_binding=plan_binding,
            receipt_binding=receipt_binding,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_registry_sha256=expected_registry_sha256,
        )
    else:
        _validate_distributed_raw(
            plan_binding=plan_binding,
            receipt_binding=receipt_binding,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_registry_sha256=expected_registry_sha256,
        )
    edges = receipt["phase_edges_ns"]
    if type(edges) is not dict:
        raise TypeError("formal distributed outcome timing is not an object")
    started = edges.get("execution_started_ns")
    finished = edges.get("process_group_empty_checked_ns")
    exit_code = receipt["process_exit_code"]
    cleanup_kind = receipt["cleanup_kind"]
    if (
        type(started) is not int
        or type(finished) is not int
        or started < 1
        or finished < started
        or type(exit_code) is not int
        or exit_code not in {0, -15}
        or cleanup_kind not in {"already_exited_clean", "sigterm_clean"}
    ):
        raise ValueError("formal distributed outcome process result differs")
    return FormalDistributedPhysicalOutcome(
        status="COMPLETE",
        plan=plan_binding,
        run_receipt=receipt_binding,
        request_terminal=CanonicalJsonProofBinding.from_dict(receipt["terminal"]),
        gang_terminal=CanonicalJsonProofBinding.from_dict(
            receipt["formal_gang_terminal"]
        ),
        native_itl_pointers=CanonicalJsonProofBinding.from_dict(
            receipt["native_itl_pointers"]
        ),
        lifecycle_timing=CanonicalJsonProofBinding.from_dict(
            receipt["lifecycle_timing"]
        ),
        before_gpu_snapshot=CanonicalJsonProofBinding.from_dict(
            receipt["before_gpu_snapshot"]
        ),
        ready_gpu_snapshot=CanonicalJsonProofBinding.from_dict(
            receipt["ready_gpu_snapshot"]
        ),
        after_gpu_snapshot=CanonicalJsonProofBinding.from_dict(
            receipt["after_gpu_snapshot"]
        ),
        server_log=EvidenceFileBinding.from_dict(
            receipt["server_log"], label="formal distributed server log"
        ),
        server_stdout=EvidenceFileBinding.from_dict(
            receipt["server_stdout"], label="formal distributed server stdout"
        ),
        server_stderr=EvidenceFileBinding.from_dict(
            receipt["server_stderr"], label="formal distributed server stderr"
        ),
        junit=EvidenceFileBinding.from_dict(
            receipt["junit"], label="formal distributed JUnit"
        ),
        execution_started_ns=started,
        process_group_empty_checked_ns=finished,
        process_exit_code=exit_code,
        cleanup_kind=cleanup_kind,
    )


def build_formal_distributed_terminal_external_control_binding(
    run_receipt_path: str,
    *,
    plan_path: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> FormalDistributedTerminalExternalControlBinding:
    validated = _validate_distributed_raw(
        plan_binding=CanonicalJsonProofBinding.bind(plan_path),
        receipt_binding=CanonicalJsonProofBinding.bind(run_receipt_path),
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    return validated.binding


def _validate_tp1_raw(
    *,
    plan_binding: CanonicalJsonProofBinding,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> _ValidatedTp1Raw:
    from lightcone_spec.config import load_run_config
    from lightcone_spec.orchestration.live_sglang import (
        UnsignedPinnedSglangServingRunReceipt,
        validate_unsigned_pinned_sglang_lifecycle_timing_receipt,
    )
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

    plan = FormalServingRunPlan.from_dict(plan_binding.reopen())
    if (
        plan.sha256 != plan_binding.semantic_sha256
        or plan.topology_mode != "tp1_dp1"
        or plan.inventory_sha256 != expected_inventory_sha256
    ):
        raise ValueError("formal TP1 plan identity/topology differs")
    live_binding = CanonicalJsonProofBinding.bind(plan.live_run_receipt_output_path)
    live = UnsignedPinnedSglangServingRunReceipt.from_dict(live_binding.reopen())
    raw_terminal = CanonicalJsonProofBinding.bind(plan.terminal_output_path)
    lifecycle = CanonicalJsonProofBinding.bind(plan.lifecycle_timing_output_path)
    if (
        live.sha256 != live_binding.semantic_sha256
        or live.launch_manifest != plan.launch_manifest
        or live.terminal_artifact != raw_terminal
        or live.native_itl_pointer_artifact
        != CanonicalJsonProofBinding.bind(plan.native_itl_pointer_output_path)
        or live.inventory_sha256 != plan.inventory_sha256
        or live.gpu_uuids != plan.gpu_uuids
        or live.run_binding_sha256
        != canonical_sha256(plan.native_terminal_binding.begin_payload())
        or live.formal_launch_admission is None
        or live.formal_launch_consumption is None
        or live.budget_consumption is None
    ):
        raise ValueError("formal TP1 run receipt differs from sealed plan/admission")
    _validate_physical_launch_evidence(
        admission=live.formal_launch_admission,
        launch_consumption=live.formal_launch_consumption,
        budget_consumption=live.budget_consumption,
        plan_path=plan_binding.absolute_path,
        expected_registry_sha256=expected_registry_sha256,
    )
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    telemetry_detail = load_run_config(launch.run_config_path).runtime.telemetry_detail
    validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
        lifecycle,
        expected_live_run_receipt=live_binding,
        expected_binding=plan.native_terminal_binding,
        expected_telemetry_detail=telemetry_detail,
    )
    native = build_native_terminal_external_control_binding(
        raw_terminal.reopen(),
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        inventory_sha256=expected_inventory_sha256,
        expected_binding=plan.native_terminal_binding,
    )
    binding = FormalTp1TerminalExternalControlBinding(
        schema_version=1,
        kind="formal_tp1_terminal_external_control_binding",
        plan_raw_sha256=plan_binding.raw_sha256,
        plan_semantic_sha256=plan_binding.semantic_sha256,
        live_run_receipt_raw_sha256=live_binding.raw_sha256,
        live_run_receipt_semantic_sha256=live_binding.semantic_sha256,
        raw_terminal_raw_sha256=raw_terminal.raw_sha256,
        raw_terminal_semantic_sha256=raw_terminal.semantic_sha256,
        native_terminal_binding_sha256=native.sha256,
        lifecycle_timing_raw_sha256=lifecycle.raw_sha256,
        lifecycle_timing_semantic_sha256=lifecycle.semantic_sha256,
        launch_admission_raw_sha256=live.formal_launch_admission.raw_sha256,
        launch_admission_semantic_sha256=(live.formal_launch_admission.semantic_sha256),
        launch_consumption_raw_sha256=live.formal_launch_consumption.raw_sha256,
        launch_consumption_semantic_sha256=(
            live.formal_launch_consumption.semantic_sha256
        ),
        budget_consumption_raw_sha256=live.budget_consumption.raw_sha256,
        budget_consumption_semantic_sha256=live.budget_consumption.semantic_sha256,
        inventory_sha256=expected_inventory_sha256,
        registry_sha256=expected_registry_sha256,
        execution_plan_sha256=plan.native_terminal_binding.execution_plan_sha256,
        rank_config_sha256=plan.native_terminal_binding.rank_config_sha256,
        run_nonce_sha256=plan.native_terminal_binding.run_nonce_sha256,
    )
    return _ValidatedTp1Raw(
        plan=plan,
        binding=binding,
        live_run_receipt=live_binding,
        raw_terminal=raw_terminal,
        lifecycle_timing=lifecycle,
        launch_admission=live.formal_launch_admission,
        launch_consumption=live.formal_launch_consumption,
        budget_consumption=live.budget_consumption,
    )


def build_formal_tp1_terminal_external_control_binding(
    *,
    plan_path: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> FormalTp1TerminalExternalControlBinding:
    return _validate_tp1_raw(
        plan_binding=CanonicalJsonProofBinding.bind(plan_path),
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    ).binding


def _load_bound_formal_serving_plan(plan_path: str) -> FormalServingRunPlan:
    binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = FormalServingRunPlan.from_dict(binding.reopen())
    if plan.sha256 != binding.semantic_sha256:
        raise ValueError("formal serving plan semantic identity changed")
    return plan


def build_formal_tp1_terminal_control_subject(
    *,
    plan_path: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> ControlArtifactSubject:
    """Derive a TP1 terminal subject solely from one immutable run plan."""

    plan = _load_bound_formal_serving_plan(plan_path)
    if plan.topology_mode != "tp1_dp1":
        raise ValueError("TP1 terminal subject requires a TP1 run plan")
    if plan.inventory_sha256 != expected_inventory_sha256:
        raise ValueError("TP1 terminal subject inventory differs from run plan")
    control_binding = build_formal_tp1_terminal_external_control_binding(
        plan_path=plan_path,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    return ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=control_binding.sha256,
        protocol_sha256=FORMAL_TP1_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256,
        registry_sha256=expected_registry_sha256,
        lineage_sha256=control_binding.lineage_sha256,
    )


def build_formal_terminal_control_subject(
    *,
    plan_path: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> ControlArtifactSubject:
    """Closed-topology terminal subject builder for TP1, TP2, and DP2."""

    plan = _load_bound_formal_serving_plan(plan_path)
    if plan.topology_mode == "tp1_dp1":
        return build_formal_tp1_terminal_control_subject(
            plan_path=plan_path,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_registry_sha256=expected_registry_sha256,
        )
    return build_formal_distributed_terminal_control_subject(
        plan.live_run_receipt_output_path,
        plan_path=plan_path,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )


def build_formal_distributed_terminal_control_subject(
    run_receipt_path: str,
    *,
    plan_path: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> ControlArtifactSubject:
    """Derive the exact offline-signing subject from immutable raw evidence."""

    binding = build_formal_distributed_terminal_external_control_binding(
        run_receipt_path,
        plan_path=plan_path,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    return ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=binding.sha256,
        protocol_sha256=(FORMAL_DISTRIBUTED_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256),
        registry_sha256=expected_registry_sha256,
        lineage_sha256=binding.lineage_sha256,
    )


def formal_distributed_scored_native_itl_pointers(
    run_receipt_path: str,
    *,
    plan_path: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Return only the deeply validated scored first-party pointer rows."""

    validated = _validate_distributed_raw(
        plan_binding=CanonicalJsonProofBinding.bind(plan_path),
        receipt_binding=CanonicalJsonProofBinding.bind(run_receipt_path),
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    receipt = CanonicalJsonProofBinding.bind(run_receipt_path).reopen()
    if type(receipt) is not dict:
        raise TypeError("formal distributed receipt must be an object")
    pointer_binding = CanonicalJsonProofBinding.from_dict(
        receipt["native_itl_pointers"]
    )
    bundle = reopen_scalable_formal_gang_itl_bundle(pointer_binding.reopen())
    if type(bundle) is not dict or type(bundle.get("scored_pointers")) is not list:
        raise ValueError("formal distributed scored pointer bundle changed")
    rows = tuple(dict(row) for row in bundle["scored_pointers"] if type(row) is dict)
    if len(rows) != len(validated.requests):
        raise ValueError("formal distributed scored pointer coverage changed")
    return rows


def formal_scored_native_itl_pointers(
    *,
    plan_path: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Reopen the plan-owned scored pointer bundle for every topology."""

    plan = _load_bound_formal_serving_plan(plan_path)
    if plan.inventory_sha256 != expected_inventory_sha256:
        raise ValueError("formal ITL pointer inventory differs from run plan")
    if plan.topology_mode != "tp1_dp1":
        return formal_distributed_scored_native_itl_pointers(
            plan.live_run_receipt_output_path,
            plan_path=plan_path,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_registry_sha256=expected_registry_sha256,
        )
    pointer_binding = CanonicalJsonProofBinding.bind(
        plan.native_itl_pointer_output_path
    )
    value = reopen_scalable_unsigned_native_itl_bundle(pointer_binding.reopen())
    if (
        type(value) is not dict
        or value.get("kind") != "unsigned_native_itl_result_pointer_bundle"
        or type(value.get("native_result_pointers")) is not list
    ):
        raise ValueError("formal TP1 native ITL pointer bundle differs")
    rows = tuple(
        dict(row) for row in value["native_result_pointers"] if type(row) is dict
    )
    if len(rows) != len(value["native_result_pointers"]) or not rows:
        raise ValueError("formal TP1 native ITL pointer coverage differs")
    return rows


def _projection(
    validated: _ValidatedDistributedRaw,
    *,
    control: ControlArtifactAttestation,
    reservation: ChallengeReplayReservationBinding,
) -> FormalDistributedTerminalResultProjection:
    plan = validated.plan
    updates = (
        validated.updates_by_rank[0]
        if plan.topology_mode == "tp2_dp1"
        else validated.updates_by_rank[0] + validated.updates_by_rank[1]
    )
    return FormalDistributedTerminalResultProjection(
        schema_version=1,
        kind="formal_distributed_terminal_result_projection",
        stage=plan.stage,
        topology_mode=plan.topology_mode,
        run_id=plan.native_terminal_binding.run_id,
        method=plan.method,
        run_nonce_sha256=plan.native_terminal_binding.run_nonce_sha256,
        execution_plan_sha256=plan.native_terminal_binding.execution_plan_sha256,
        rank_config_sha256=plan.native_terminal_binding.rank_config_sha256,
        attempt_id=plan.native_terminal_binding.attempt_id,
        terminal_sha256=validated.aggregate_sha256,
        native_itl_pointer_bundle_sha256=(
            validated.binding.pointer_bundle_semantic_sha256
        ),
        lifecycle_timing_sha256=(validated.binding.lifecycle_timing_semantic_sha256),
        launch_admission_sha256=(validated.binding.launch_admission_semantic_sha256),
        launch_consumption_sha256=(
            validated.binding.launch_consumption_semantic_sha256
        ),
        budget_consumption_sha256=(
            validated.binding.budget_consumption_semantic_sha256
        ),
        authority_kind="external_release_control",
        external_control_binding_sha256=validated.binding.sha256,
        external_control_envelope_sha256=control.sha256,
        external_control_reservation_sha256=reservation.reservation_sha256,
        requests=validated.requests,
        updates=updates,
        scored_request_ids=tuple(row.request_id for row in validated.requests),
        output_token_count=sum(len(row.output_token_ids) for row in validated.requests),
        performance_counters=_performance(
            validated.performance_by_rank,
            topology=plan.topology_mode,
        ),
        rank_terminal_sha256s=validated.rank_terminal_sha256s,
    )


def publish_formal_distributed_terminal_result_proof_artifact(
    run_receipt_path: str,
    *,
    plan_path: str,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    output = _new_path(proof_artifact_path, label="formal distributed proof output")
    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    receipt_binding = CanonicalJsonProofBinding.bind(run_receipt_path)
    validated = _validate_distributed_raw(
        plan_binding=plan_binding,
        receipt_binding=receipt_binding,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    subject = control_attestation.subject
    if (
        control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
        or subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != validated.binding.sha256
        or subject.protocol_sha256
        != FORMAL_DISTRIBUTED_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_registry_sha256
        or subject.lineage_sha256 != validated.binding.lineage_sha256
    ):
        raise ValueError("formal distributed external control differs")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=(
            validated.plan.native_terminal_binding.run_nonce_sha256,
        ),
    )
    reservation_sha = control_challenge_reservation_sha256(
        verified,
        reserved_ns=now_ns,
        additional_challenge_sha256s=(
            validated.plan.native_terminal_binding.run_nonce_sha256,
        ),
    )
    reservation = replay_store.bind_reservation(reservation_sha)
    projection = _projection(
        validated,
        control=control_attestation,
        reservation=reservation,
    )
    receipt = receipt_binding.reopen()
    assert type(receipt) is dict
    artifact = FormalDistributedTerminalResultProofArtifact(
        schema_version=1,
        kind="formal_distributed_terminal_result_proof_artifact",
        plan=plan_binding,
        run_receipt=receipt_binding,
        request_terminal=CanonicalJsonProofBinding.from_dict(receipt["terminal"]),
        gang_terminal=CanonicalJsonProofBinding.from_dict(
            receipt["formal_gang_terminal"]
        ),
        pointer_bundle=CanonicalJsonProofBinding.from_dict(
            receipt["native_itl_pointers"]
        ),
        lifecycle_timing=CanonicalJsonProofBinding.from_dict(
            receipt["lifecycle_timing"]
        ),
        launch_admission=CanonicalJsonProofBinding.from_dict(
            receipt["formal_launch_admission"]
        ),
        launch_consumption=CanonicalJsonProofBinding.from_dict(
            receipt["formal_launch_consumption"]
        ),
        budget_consumption=CanonicalJsonProofBinding.from_dict(
            receipt["budget_consumption"]
        ),
        control_attestation=control_attestation,
        replay_reservation=reservation,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        result=projection.to_dict(),
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)


def publish_formal_tp1_terminal_result_proof_artifact(
    *,
    plan_path: str,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    """Trust-lift one admitted TP1 terminal and its exact physical lifecycle."""

    output = _new_path(proof_artifact_path, label="formal TP1 proof output")
    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    validated = _validate_tp1_raw(
        plan_binding=plan_binding,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    subject = control_attestation.subject
    if (
        control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
        or subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != validated.binding.sha256
        or subject.protocol_sha256
        != FORMAL_TP1_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_registry_sha256
        or subject.lineage_sha256 != validated.binding.lineage_sha256
    ):
        raise ValueError("formal TP1 external control differs")
    verified_rows = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=(
            validated.plan.native_terminal_binding.run_nonce_sha256,
        ),
    )
    reservation_sha = control_challenge_reservation_sha256(
        verified_rows,
        reserved_ns=now_ns,
        additional_challenge_sha256s=(
            validated.plan.native_terminal_binding.run_nonce_sha256,
        ),
    )
    reservation = replay_store.bind_reservation(reservation_sha)
    projection = derive_native_terminal_result_projection_from_verified_formal_control(
        validated.raw_terminal.reopen(),
        expected_binding=validated.plan.native_terminal_binding,
        expected_inventory_sha256=expected_inventory_sha256,
        formal_control_binding_sha256=validated.binding.sha256,
        verified_control=verified_rows[0],
        replay_reservation=reservation,
    )
    artifact = FormalTp1TerminalResultProofArtifact(
        schema_version=1,
        kind="formal_tp1_terminal_result_proof_artifact",
        plan=plan_binding,
        live_run_receipt=validated.live_run_receipt,
        raw_terminal=validated.raw_terminal,
        lifecycle_timing=validated.lifecycle_timing,
        launch_admission=validated.launch_admission,
        launch_consumption=validated.launch_consumption,
        budget_consumption=validated.budget_consumption,
        control_attestation=control_attestation,
        replay_reservation=reservation,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        result=projection.to_dict(),
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)


def validate_formal_tp1_terminal_result_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_execution_plan_sha256: str,
    expected_rank_config_sha256: str,
    expected_run_id: str,
    expected_run_nonce_sha256: str,
    expected_attempt_id: str,
    expected_method: str,
    now_ns: int,
    expected_stage: str | None = None,
) -> NativeTerminalResultProjection:
    proof = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalTp1TerminalResultProofArtifact.from_dict(proof.reopen())
    if (
        proof.semantic_sha256 != artifact.sha256
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
        or type(now_ns) is not int
        or now_ns < artifact.replay_reservation.reserved_ns
    ):
        raise ValueError("formal TP1 proof identity, release root, or time differs")
    validated = _validate_tp1_raw(
        plan_binding=artifact.plan,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    for name in (
        "live_run_receipt",
        "raw_terminal",
        "lifecycle_timing",
        "launch_admission",
        "launch_consumption",
        "budget_consumption",
    ):
        if getattr(artifact, name) != getattr(validated, name):
            raise ValueError("formal TP1 proof member binding changed")
    subject = artifact.control_attestation.subject
    if (
        artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
        or subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != validated.binding.sha256
        or subject.protocol_sha256
        != FORMAL_TP1_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_registry_sha256
        or subject.lineage_sha256 != validated.binding.lineage_sha256
    ):
        raise ValueError("formal TP1 proof external control differs")
    verified = verify_release_control_artifact_attestation(
        artifact.control_attestation,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=artifact.replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    reserved = artifact.replay_reservation.revalidate()
    expected_reservation = control_challenge_reservation_sha256(
        (verified,),
        reserved_ns=artifact.replay_reservation.reserved_ns,
        additional_challenge_sha256s=(
            validated.plan.native_terminal_binding.run_nonce_sha256,
        ),
    )
    required = {
        validated.plan.native_terminal_binding.run_nonce_sha256,
        verified.challenge_sha256,
        verified.deployment_policy_challenge_sha256,
    }
    if (
        not required.issubset(set(reserved))
        or artifact.replay_reservation.reservation_sha256 != expected_reservation
    ):
        raise ValueError("formal TP1 proof replay reservation differs")
    projection = derive_native_terminal_result_projection_from_verified_formal_control(
        validated.raw_terminal.reopen(),
        expected_binding=validated.plan.native_terminal_binding,
        expected_inventory_sha256=expected_inventory_sha256,
        formal_control_binding_sha256=validated.binding.sha256,
        verified_control=verified,
        replay_reservation=artifact.replay_reservation,
    )
    if projection.to_dict() != artifact.result:
        raise ValueError("formal TP1 proof derived result changed")
    expected_values = (
        (projection.execution_plan_sha256, expected_execution_plan_sha256),
        (projection.rank_config_sha256, expected_rank_config_sha256),
        (projection.run_id, expected_run_id),
        (projection.run_nonce_sha256, expected_run_nonce_sha256),
        (projection.attempt_id, expected_attempt_id),
        (projection.method, expected_method),
    )
    if any(observed != expected for observed, expected in expected_values):
        raise ValueError("formal TP1 proof execution identity differs")
    if expected_stage is not None and validated.plan.stage != expected_stage:
        raise ValueError("formal TP1 proof stage differs")
    return projection


def publish_formal_terminal_result_proof_artifact(
    *,
    plan_path: str,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    """Closed-topology durable terminal publisher for every serving plan."""

    plan = _load_bound_formal_serving_plan(plan_path)
    if plan.inventory_sha256 != expected_inventory_sha256:
        raise ValueError("formal terminal proof inventory differs from run plan")
    if plan.topology_mode == "tp1_dp1":
        return publish_formal_tp1_terminal_result_proof_artifact(
            plan_path=plan_path,
            control_attestation=control_attestation,
            replay_store=replay_store,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_registry_sha256=expected_registry_sha256,
            expected_root_manifest_sha256=expected_root_manifest_sha256,
            now_ns=now_ns,
            proof_artifact_path=proof_artifact_path,
        )
    return publish_formal_distributed_terminal_result_proof_artifact(
        plan.live_run_receipt_output_path,
        plan_path=plan_path,
        control_attestation=control_attestation,
        replay_store=replay_store,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
        proof_artifact_path=proof_artifact_path,
    )


def validate_formal_distributed_terminal_result_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_execution_plan_sha256: str,
    expected_rank_config_sha256: str,
    expected_run_id: str,
    expected_run_nonce_sha256: str,
    expected_attempt_id: str,
    expected_method: str,
    expected_stage: str | None = None,
    expected_topology: Literal["tp2_dp1", "tp1_dp2"] | None = None,
    now_ns: int,
) -> FormalDistributedTerminalResultProjection:
    proof = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalDistributedTerminalResultProofArtifact.from_dict(proof.reopen())
    if (
        proof.semantic_sha256 != artifact.sha256
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("formal distributed proof identity/root differs")
    validated = _validate_distributed_raw(
        plan_binding=artifact.plan,
        receipt_binding=artifact.run_receipt,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
    )
    receipt = artifact.run_receipt.reopen()
    assert type(receipt) is dict
    for name, expected in (
        ("request_terminal", CanonicalJsonProofBinding.from_dict(receipt["terminal"])),
        (
            "gang_terminal",
            CanonicalJsonProofBinding.from_dict(receipt["formal_gang_terminal"]),
        ),
        (
            "pointer_bundle",
            CanonicalJsonProofBinding.from_dict(receipt["native_itl_pointers"]),
        ),
        (
            "lifecycle_timing",
            CanonicalJsonProofBinding.from_dict(receipt["lifecycle_timing"]),
        ),
        (
            "launch_admission",
            CanonicalJsonProofBinding.from_dict(receipt["formal_launch_admission"]),
        ),
        (
            "launch_consumption",
            CanonicalJsonProofBinding.from_dict(receipt["formal_launch_consumption"]),
        ),
        (
            "budget_consumption",
            CanonicalJsonProofBinding.from_dict(receipt["budget_consumption"]),
        ),
    ):
        if getattr(artifact, name) != expected:
            raise ValueError("formal distributed proof member binding changed")
    subject = artifact.control_attestation.subject
    if (
        artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
        or subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != validated.binding.sha256
        or subject.protocol_sha256
        != FORMAL_DISTRIBUTED_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_registry_sha256
        or subject.lineage_sha256 != validated.binding.lineage_sha256
        or type(now_ns) is not int
        or now_ns < artifact.replay_reservation.reserved_ns
    ):
        raise ValueError("formal distributed proof external control differs")
    reserved = artifact.replay_reservation.revalidate()
    control = verify_release_control_artifact_attestation(
        artifact.control_attestation,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=artifact.replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    required = {
        validated.plan.native_terminal_binding.run_nonce_sha256,
        control.challenge_sha256,
        control.deployment_policy_challenge_sha256,
    }
    expected_reservation = control_challenge_reservation_sha256(
        (control,),
        reserved_ns=artifact.replay_reservation.reserved_ns,
        additional_challenge_sha256s=(
            validated.plan.native_terminal_binding.run_nonce_sha256,
        ),
    )
    if (
        not required.issubset(set(reserved))
        or artifact.replay_reservation.reservation_sha256 != expected_reservation
    ):
        raise ValueError("formal distributed proof reservation differs")
    projection = _projection(
        validated,
        control=artifact.control_attestation,
        reservation=artifact.replay_reservation,
    )
    if projection.to_dict() != artifact.result:
        raise ValueError("formal distributed proof derived result changed")
    expected_values = (
        (projection.execution_plan_sha256, expected_execution_plan_sha256),
        (projection.rank_config_sha256, expected_rank_config_sha256),
        (projection.run_id, expected_run_id),
        (projection.run_nonce_sha256, expected_run_nonce_sha256),
        (projection.attempt_id, expected_attempt_id),
        (projection.method, expected_method),
    )
    if any(observed != expected for observed, expected in expected_values):
        raise ValueError("formal distributed proof execution identity differs")
    if expected_stage is not None and projection.stage != expected_stage:
        raise ValueError("formal distributed proof stage differs")
    if expected_topology is not None and projection.topology_mode != expected_topology:
        raise ValueError("formal distributed proof topology differs")
    return projection


def validate_formal_terminal_result_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_execution_plan_sha256: str,
    expected_rank_config_sha256: str,
    expected_run_id: str,
    expected_run_nonce_sha256: str,
    expected_attempt_id: str,
    expected_method: str,
    now_ns: int,
    expected_stage: str | None = None,
    expected_topology: str | None = None,
) -> (
    NativeTerminalResultProjection
    | FormalDistributedTerminalResultProjection
    | FormalSingleOperatorPreflightRawTerminalProjection
):
    """Closed-kind dispatcher for TP1 native and TP2/DP2 gang proofs."""

    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    value = binding.reopen()
    if type(value) is not dict:
        raise TypeError("formal terminal proof artifact must be an object")
    kind = value.get("kind")
    common = {
        "expected_inventory_sha256": expected_inventory_sha256,
        "expected_registry_sha256": expected_registry_sha256,
        "expected_root_manifest_sha256": expected_root_manifest_sha256,
        "expected_execution_plan_sha256": expected_execution_plan_sha256,
        "expected_rank_config_sha256": expected_rank_config_sha256,
        "expected_run_id": expected_run_id,
        "expected_run_nonce_sha256": expected_run_nonce_sha256,
        "expected_attempt_id": expected_attempt_id,
        "expected_method": expected_method,
        "now_ns": now_ns,
    }
    if kind == "formal_tp1_terminal_result_proof_artifact":
        if expected_topology not in {None, "tp1_dp1"}:
            raise ValueError("TP1 native proof cannot satisfy a distributed topology")
        return validate_formal_tp1_terminal_result_proof_artifact(
            proof_artifact_path,
            expected_stage=expected_stage,
            **common,
        )
    if kind == "formal_preflight_tp1_terminal_result_proof_artifact":
        if expected_topology not in {None, "tp1_dp1"}:
            raise ValueError("preflight TP1 proof cannot satisfy another topology")
        if expected_stage not in {None, "preflight"}:
            raise ValueError("preflight TP1 proof cannot satisfy a formal stage")
        return validate_formal_preflight_tp1_terminal_result_proof_artifact(
            proof_artifact_path,
            **common,
        )
    if kind == "formal_current_preflight_tp1_terminal_result_proof_artifact":
        if expected_topology not in {None, "tp1_dp1"}:
            raise ValueError(
                "current preflight TP1 proof cannot satisfy another topology"
            )
        if expected_stage not in {None, "preflight"}:
            raise ValueError(
                "current preflight TP1 proof cannot satisfy a formal stage"
            )
        return validate_formal_current_preflight_tp1_terminal_result_proof_artifact(
            proof_artifact_path,
            **common,
        )
    if kind == ("formal_single_operator_preflight_tp1_raw_terminal_proof_artifact"):
        if expected_topology not in {None, "tp1_dp1"}:
            raise ValueError("single-operator raw TP1 proof has another topology")
        if expected_stage not in {None, "preflight"}:
            raise ValueError("single-operator raw TP1 proof has another stage")
        return (
            validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
                proof_artifact_path,
                **common,
            )
        )
    if kind == "native_terminal_result_proof_artifact":
        raise ValueError("legacy native terminal proof cannot satisfy formal coverage")
    if kind == "formal_distributed_terminal_result_proof_artifact":
        if expected_topology not in {None, "tp2_dp1", "tp1_dp2"}:
            raise ValueError("distributed proof topology is unsupported")
        return validate_formal_distributed_terminal_result_proof_artifact(
            proof_artifact_path,
            expected_stage=expected_stage,
            expected_topology=expected_topology,  # type: ignore[arg-type]
            **common,
        )
    raise ValueError("formal terminal proof artifact kind is unsupported")


__all__ = [
    "FORMAL_CURRENT_PREFLIGHT_TP1_TERMINAL_PROOF_PROTOCOL_SHA256",
    "FORMAL_DISTRIBUTED_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256",
    "FORMAL_PREFLIGHT_TP1_TERMINAL_PROOF_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_PREFLIGHT_TP1_RAW_TERMINAL_PROTOCOL_SHA256",
    "FORMAL_TP1_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256",
    "FormalCurrentPreflightTp1TerminalResultProofArtifact",
    "FormalDistributedPhysicalOutcome",
    "FormalDistributedTerminalExternalControlBinding",
    "FormalDistributedTerminalRequestResult",
    "FormalDistributedTerminalResultProjection",
    "FormalDistributedTerminalResultProofArtifact",
    "FormalDistributedTerminalUpdateResult",
    "FormalPreflightTp1TerminalResultProofArtifact",
    "FormalSingleOperatorPreflightRawRequestResult",
    "FormalSingleOperatorPreflightRawTerminalProjection",
    "FormalSingleOperatorPreflightTp1RawTerminalProofArtifact",
    "FormalTp1TerminalExternalControlBinding",
    "FormalTp1TerminalResultProofArtifact",
    "build_formal_distributed_terminal_control_subject",
    "build_formal_distributed_terminal_external_control_binding",
    "build_formal_terminal_control_subject",
    "build_formal_tp1_terminal_control_subject",
    "build_formal_tp1_terminal_external_control_binding",
    "formal_distributed_scored_native_itl_pointers",
    "formal_scored_native_itl_pointers",
    "publish_formal_current_preflight_tp1_terminal_result_proof_artifact",
    "publish_formal_distributed_terminal_result_proof_artifact",
    "publish_formal_preflight_tp1_terminal_result_proof_artifact",
    "publish_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact",
    "publish_formal_terminal_result_proof_artifact",
    "publish_formal_tp1_terminal_result_proof_artifact",
    "validate_formal_current_preflight_tp1_terminal_result_proof_artifact",
    "validate_formal_distributed_physical_outcome",
    "validate_formal_distributed_terminal_result_proof_artifact",
    "validate_formal_preflight_tp1_terminal_result_proof_artifact",
    "validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact",
    "validate_formal_terminal_result_proof_artifact",
    "validate_formal_tp1_terminal_result_proof_artifact",
]
