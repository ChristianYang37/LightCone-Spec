"""Two-phase execution boundary for the sealed ten-cell formal preflight.

The remote GPU host receives no offline signing key.  Phase one consumes an
in-memory :class:`VerifiedFormalPreflightDispatch` and runs only source-owned
compile or exactness runners, each under its own fresh external control.  The
remote runner publishes raw terminal evidence even on a test failure.  Phase
two runs locally after stable evidence pull: it qualifies exactness with a
fresh rank-aggregate control, deep-reopens the eight serving terminal
authorities, and derives the sole complete coverage receipt.

This module deliberately has no callback-based execution escape hatch.  The
remaining eight serving rows come from the first-party unsigned pinned-SGLang
runner, then a local dynamic-control aggregate.  A raw batch remains
``WAITING_FOR_LOCAL_CONTROL`` and can never complete preflight.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lightcone_spec.experiments.formal_dispatch import (
    FormalPreflightExecutionBinding,
    VerifiedFormalPreflightDispatch,
    require_verified_formal_preflight_dispatch,
)
from lightcone_spec.experiments.formal_preflight_launch import (
    FormalPreflightLaunchWaveConsumption,
    consume_formal_preflight_launch_wave,
    revalidate_formal_preflight_launch_cap_schedule,
    validate_formal_preflight_launch_wave_consumption,
)
from lightcone_spec.experiments.load import FrozenSamplingParameters
from lightcone_spec.experiments.preflight_authority import (
    PreflightCoverageReceipt,
    PreflightExecutionSourceAuthority,
    materialize_formal_preflight_stage_coverage,
    materialize_pointer_preflight_coverage,
    require_complete_preflight_coverage,
)
from lightcone_spec.experiments.preflight_interference import (
    FormalPreflightInterferenceProofArtifact,
    FormalPreflightInterferenceQualificationRow,
    FormalPreflightInterferenceRawBatch,
    FormalPreflightInterferenceRawRow,
    VerifiedFormalPreflightInterferenceProof,
    _publish_single_operator_preflight_interference_qualification_lock,
    _publish_single_operator_preflight_interference_raw_batch,
    build_formal_preflight_interference_raw_row,
    publish_formal_preflight_interference_fatal_terminal,
    publish_formal_preflight_interference_proof_artifact,
    validate_formal_preflight_interference_proof_artifact,
)
from lightcone_spec.experiments.registry import ExperimentRegistry, content_sha256
from lightcone_spec.experiments.serving import BoundServingRequest
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
)
from lightcone_spec.runtime.compile_runner import (
    CompileAssignmentPlan,
    CompileResultPointer,
    _execute_release_compile_assignment_plan_admitted,
    revalidate_prepared_content_verification_receipt,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
)
from lightcone_spec.runtime.preflight_runner import (
    ExactnessPreflightAssignment,
    ExactnessPreflightResultPointer,
    ExactnessQualificationProofArtifact,
    execute_release_exactness_preflight,
    finalize_release_exactness_preflight,
    publish_release_exactness_qualification_proof,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

if TYPE_CHECKING:
    from lightcone_spec.experiments.formal_protocol import (
        TtsL0CandidateStateCoverage,
    )
    from lightcone_spec.experiments.stage_materialization import (
        StageCoverageReceipt,
        StageMaterializationReceipt,
    )

FORMAL_PREFLIGHT_EXECUTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "sealed_formal_preflight_two_phase_execution",
        "sealed_input": "VerifiedFormalPreflightDispatch",
        "remote_unsigned": (
            "one_first_party_compile_result",
            "one_first_party_exactness_schema3_result",
            "eight_first_party_unsigned_interference_terminal_pointer_pairs",
        ),
        "local_controlled": (
            "compile_control_before_gpu_mutation",
            "exactness_non_serving_control_before_gpu_mutation",
            "post_pull_exactness_rank_aggregate_control",
            "deep_reopened_dynamic_interference_result_itl_aggregate",
            "eight_distinct_suite_specific_native_distributed_session_proofs",
        ),
        "terminal_coverage": "exact_1_plus_1_plus_8_zero_skip",
        "stable_pull": (
            "every_local_qualification_and_final_reducer_reopens_the_same_"
            "remote_waiting_receipt"
        ),
        "formal_registry_bridge": (
            "exact_pointer_coverage_to_signable_stage_coverage_with_native_"
            "tts_l0_replay_proofs"
        ),
        "remote_private_key_forbidden": True,
        "callback_or_bare_digest_execution_forbidden": True,
    }
)

FormalPreflightRunnerKind = Literal[
    "first_party_compile",
    "first_party_exactness",
    "first_party_interference",
]


class FormalPreflightExecutionBlocked(RuntimeError):
    """Named fail-closed state before a formal runner mutates GPU state."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"formal preflight execution is BLOCKED: {reason_code}")


@dataclass(frozen=True)
class FormalPreflightInterferenceRunInput:
    """One source-materialized serving input for a sealed interference row."""

    registry_cell_id: str
    launch_manifest_path: str
    run_binding: object
    warmup_requests: tuple[BoundServingRequest, ...]
    scored_requests: tuple[BoundServingRequest, ...]
    qualification_rows: tuple[FormalPreflightInterferenceQualificationRow, ...]

    def validate(self) -> None:
        from lightcone_spec.orchestration.native_terminal import (
            NativeTerminalRunBinding,
        )

        if (
            type(self.registry_cell_id) is not str
            or len(self.registry_cell_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.registry_cell_id
            )
        ):
            raise ValueError("formal preflight interference cell ID is invalid")
        path = Path(self.launch_manifest_path)
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise ValueError("formal preflight launch manifest path is not canonical")
        if type(self.run_binding) is not NativeTerminalRunBinding:
            raise TypeError("formal preflight interference run binding is not exact")
        self.run_binding.validate()
        if self.run_binding.method != "static":
            raise ValueError("formal preflight interference must run Static")
        for label, requests in (
            ("warmup", self.warmup_requests),
            ("scored", self.scored_requests),
        ):
            if type(requests) is not tuple or any(
                type(request) is not BoundServingRequest for request in requests
            ):
                raise TypeError(f"formal preflight {label} requests are not exact")
            for request in requests:
                request.validate()
            request_ids = tuple(request.request_id for request in requests)
            if len(request_ids) != len(set(request_ids)):
                raise ValueError(f"formal preflight {label} request IDs repeat")
        if (
            tuple(request.request_id for request in self.warmup_requests)
            != self.run_binding.warmup_request_ids
            or tuple(request.request_id for request in self.scored_requests)
            != self.run_binding.scored_request_ids
        ):
            raise ValueError("formal preflight run/request coverage differs")
        if (
            type(self.qualification_rows) is not tuple
            or any(
                type(row) is not FormalPreflightInterferenceQualificationRow
                for row in self.qualification_rows
            )
            or {row.request_id for row in self.qualification_rows}
            != {request.request_id for request in self.scored_requests}
        ):
            raise ValueError("formal preflight request qualification differs")

    def to_dict(self) -> dict[str, object]:
        from lightcone_spec.experiments.preflight_interference import (
            _run_binding_dict,
        )

        self.validate()
        return {
            "registry_cell_id": self.registry_cell_id,
            "launch_manifest_path": self.launch_manifest_path,
            "run_binding": _run_binding_dict(self.run_binding),
            "warmup_requests": [
                _bound_serving_request_to_dict(row) for row in self.warmup_requests
            ],
            "scored_requests": [
                _bound_serving_request_to_dict(row) for row in self.scored_requests
            ],
            "qualification_rows": [row.to_dict() for row in self.qualification_rows],
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalPreflightInterferenceRunInput:
        from lightcone_spec.experiments.preflight_interference import (
            _run_binding_from_dict,
        )

        fields = {
            "registry_cell_id",
            "launch_manifest_path",
            "run_binding",
            "warmup_requests",
            "scored_requests",
            "qualification_rows",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal preflight interference input fields differ")
        for name in ("warmup_requests", "scored_requests", "qualification_rows"):
            if type(value[name]) is not list:
                raise TypeError(f"formal preflight {name} must be an array")
        result = cls(
            registry_cell_id=value["registry_cell_id"],
            launch_manifest_path=value["launch_manifest_path"],
            run_binding=_run_binding_from_dict(value["run_binding"]),
            warmup_requests=tuple(
                _bound_serving_request_from_dict(row)
                for row in value["warmup_requests"]
            ),
            scored_requests=tuple(
                _bound_serving_request_from_dict(row)
                for row in value["scored_requests"]
            ),
            qualification_rows=tuple(
                FormalPreflightInterferenceQualificationRow.from_dict(row)
                for row in value["qualification_rows"]
            ),
        )
        result.validate()
        return result


def _bound_serving_request_to_dict(value: BoundServingRequest) -> dict[str, object]:
    value.validate()
    return {
        "request_id": value.request_id,
        "namespace": value.namespace,
        "split": value.split,
        "ordinal": value.ordinal,
        "input_token_ids": list(value.input_token_ids),
        "requested_output_tokens": value.requested_output_tokens,
        "arrival_us": value.arrival_us,
        "cancellation_offset_us": value.cancellation_offset_us,
        "cohort_id": value.cohort_id,
        "cohort_sha256": value.cohort_sha256,
        "route_id": value.route_id,
        "sampling": dict(value.sampling.items),
    }


def _bound_serving_request_from_dict(value: object) -> BoundServingRequest:
    fields = {
        "request_id",
        "namespace",
        "split",
        "ordinal",
        "input_token_ids",
        "requested_output_tokens",
        "arrival_us",
        "cancellation_offset_us",
        "cohort_id",
        "cohort_sha256",
        "route_id",
        "sampling",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("formal preflight bound request fields differ")
    raw_tokens = value["input_token_ids"]
    raw_sampling = value["sampling"]
    if type(raw_tokens) is not list or type(raw_sampling) is not dict:
        raise TypeError("formal preflight bound request collections are invalid")
    result = BoundServingRequest(
        request_id=value["request_id"],
        namespace=value["namespace"],
        split=value["split"],
        ordinal=value["ordinal"],
        input_token_ids=tuple(raw_tokens),
        requested_output_tokens=value["requested_output_tokens"],
        arrival_us=value["arrival_us"],
        cancellation_offset_us=value["cancellation_offset_us"],
        cohort_id=value["cohort_id"],
        cohort_sha256=value["cohort_sha256"],
        route_id=value["route_id"],
        sampling=FrozenSamplingParameters.from_mapping(raw_sampling),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class FormalPreflightInterferenceExecutionManifest:
    """Path-bound exact-eight input manifest; it grants no execution authority."""

    schema_version: int
    kind: Literal["formal_preflight_interference_execution_manifest"]
    dispatch_receipt_semantic_sha256: str
    inputs: tuple[FormalPreflightInterferenceRunInput, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_preflight_interference_execution_manifest"
        ):
            raise ValueError("formal preflight interference manifest is unsupported")
        if (
            type(self.dispatch_receipt_semantic_sha256) is not str
            or len(self.dispatch_receipt_semantic_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.dispatch_receipt_semantic_sha256
            )
        ):
            raise ValueError("formal preflight dispatch receipt digest is invalid")
        if (
            len(self.inputs) != 8
            or self.inputs
            != tuple(sorted(self.inputs, key=lambda row: row.registry_cell_id))
            or len({row.registry_cell_id for row in self.inputs}) != 8
        ):
            raise ValueError(
                "formal preflight interference manifest is not exact eight"
            )
        for row in self.inputs:
            row.validate()

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "dispatch_receipt_semantic_sha256": (self.dispatch_receipt_semantic_sha256),
            "inputs": [row.to_dict() for row in self.inputs],
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalPreflightInterferenceExecutionManifest:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "dispatch_receipt_semantic_sha256",
            "inputs",
        }:
            raise ValueError("formal preflight interference manifest fields differ")
        rows = value["inputs"]
        if type(rows) is not list:
            raise TypeError("formal preflight interference inputs are not an array")
        return cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            dispatch_receipt_semantic_sha256=value["dispatch_receipt_semantic_sha256"],
            inputs=tuple(
                FormalPreflightInterferenceRunInput.from_dict(row) for row in rows
            ),
        )


@dataclass(frozen=True)
class FormalPreflightRemoteInterferenceEvidence:
    """The remote phase is explicitly incomplete until local controls exist."""

    status: Literal["WAITING_FOR_LOCAL_CONTROL", "ERROR"]
    raw_batch: CanonicalJsonProofBinding
    rows: tuple[FormalPreflightInterferenceRawRow, ...]
    launch_consumptions: tuple[CanonicalJsonProofBinding, ...]

    def __post_init__(self) -> None:
        if self.status not in {"WAITING_FOR_LOCAL_CONTROL", "ERROR"}:
            raise ValueError("formal preflight remote status is invalid")
        if type(self.raw_batch) is not CanonicalJsonProofBinding:
            raise TypeError("formal preflight remote batch is not path-bound")
        if (
            type(self.launch_consumptions) is not tuple
            or not self.launch_consumptions
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.launch_consumptions
            )
            or self.launch_consumptions
            != tuple(
                sorted(
                    set(self.launch_consumptions),
                    key=lambda row: row.absolute_path,
                )
            )
        ):
            raise ValueError("formal preflight interference launch lineage differs")
        batch = FormalPreflightInterferenceRawBatch.from_dict(self.raw_batch.reopen())
        batch.revalidate()
        if (
            batch.sha256 != self.raw_batch.semantic_sha256
            or batch.status != self.status
            or batch.rows != self.rows
        ):
            raise ValueError("formal preflight remote evidence identity differs")


@dataclass(frozen=True)
class FormalPreflightRemoteRawEvidenceReceipt:
    """Explicit stable-pull boundary; it is never terminal coverage."""

    schema_version: int
    kind: Literal["formal_preflight_remote_raw_evidence_receipt"]
    status: Literal["WAITING_FOR_LOCAL_CONTROL"]
    dispatch_sha256: str
    launch_cap_schedule: CanonicalJsonProofBinding
    launch_consumptions: tuple[CanonicalJsonProofBinding, ...]
    compile_result: CanonicalJsonProofBinding
    exactness_result: CanonicalJsonProofBinding
    interference_raw_batch: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.kind != (
            "formal_preflight_remote_raw_evidence_receipt"
        ):
            raise ValueError("formal preflight raw receipt schema is unsupported")
        if self.status != "WAITING_FOR_LOCAL_CONTROL":
            raise ValueError("formal preflight raw receipt cannot claim completion")
        if (
            type(self.dispatch_sha256) is not str
            or len(self.dispatch_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.dispatch_sha256
            )
        ):
            raise ValueError("formal preflight raw dispatch digest is invalid")
        for value in (
            self.launch_cap_schedule,
            self.compile_result,
            self.exactness_result,
            self.interference_raw_batch,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("formal preflight raw source is not path-bound")
        if (
            type(self.launch_consumptions) is not tuple
            or not self.launch_consumptions
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.launch_consumptions
            )
            or self.launch_consumptions
            != tuple(
                sorted(
                    set(self.launch_consumptions),
                    key=lambda row: row.absolute_path,
                )
            )
        ):
            raise ValueError("formal preflight raw launch consumptions differ")

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "status": self.status,
            "dispatch_sha256": self.dispatch_sha256,
            "launch_cap_schedule": self.launch_cap_schedule.to_dict(),
            "launch_consumptions": [row.to_dict() for row in self.launch_consumptions],
            "compile_result": self.compile_result.to_dict(),
            "exactness_result": self.exactness_result.to_dict(),
            "interference_raw_batch": self.interference_raw_batch.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> FormalPreflightRemoteRawEvidenceReceipt:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "status",
            "dispatch_sha256",
            "launch_cap_schedule",
            "launch_consumptions",
            "compile_result",
            "exactness_result",
            "interference_raw_batch",
        }:
            raise ValueError("formal preflight raw receipt fields differ")
        raw_consumptions = value["launch_consumptions"]
        if type(raw_consumptions) is not list:
            raise TypeError("formal preflight raw launch consumptions are not an array")
        return cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            status=value["status"],
            dispatch_sha256=value["dispatch_sha256"],
            launch_cap_schedule=CanonicalJsonProofBinding.from_dict(
                value["launch_cap_schedule"]
            ),
            launch_consumptions=tuple(
                CanonicalJsonProofBinding.from_dict(row) for row in raw_consumptions
            ),
            compile_result=CanonicalJsonProofBinding.from_dict(value["compile_result"]),
            exactness_result=CanonicalJsonProofBinding.from_dict(
                value["exactness_result"]
            ),
            interference_raw_batch=CanonicalJsonProofBinding.from_dict(
                value["interference_raw_batch"]
            ),
        )

    def revalidate(self, dispatch: object) -> None:
        token = _verified(dispatch)
        decoded_consumptions = tuple(
            FormalPreflightLaunchWaveConsumption.from_dict(row.reopen())
            for row in self.launch_consumptions
        )
        evidence_ns = max(row.consumed_ns for row in decoded_consumptions)
        schedule = revalidate_formal_preflight_launch_cap_schedule(
            self.launch_cap_schedule.absolute_path,
            current_ns=evidence_ns,
        )
        consumptions = tuple(
            validate_formal_preflight_launch_wave_consumption(
                row.absolute_path,
                current_ns=evidence_ns,
            )
            for row in self.launch_consumptions
        )
        compile_pointer = CompileResultPointer.load(self.compile_result.absolute_path)
        exactness_pointer = ExactnessPreflightResultPointer.load(
            self.exactness_result.absolute_path
        )
        batch = FormalPreflightInterferenceRawBatch.from_dict(
            self.interference_raw_batch.reopen()
        )
        batch.revalidate()
        if (
            self.dispatch_sha256 != token.sha256
            or self.launch_cap_schedule
            != CanonicalJsonProofBinding.bind(
                self.launch_cap_schedule.absolute_path,
                semantic_sha256=schedule.sha256,
            )
            or compile_pointer.schema_version != 3
            or compile_pointer.formal_execution_authorized is not True
            or self.compile_result.semantic_sha256 != compile_pointer.sha256
            or exactness_pointer.schema_version != 3
            or self.exactness_result.semantic_sha256 != exactness_pointer.sha256
            or batch.status != "WAITING_FOR_LOCAL_CONTROL"
            or batch.dispatch_sha256 != token.sha256
            or self.interference_raw_batch.semantic_sha256 != batch.sha256
        ):
            raise ValueError("formal preflight raw receipt upstream evidence differs")
        consumed_cells = tuple(
            sorted(
                cell_id
                for consumption in consumptions
                for cell_id in consumption.materialized_cell_ids
            )
        )
        if (
            consumed_cells
            != tuple(row.materialized_cell_id for row in schedule.cell_caps)
            or len(consumed_cells) != len(set(consumed_cells))
            or any(
                consumption.schedule != self.launch_cap_schedule
                or consumption.attempt_index != 0
                for consumption in consumptions
            )
        ):
            raise ValueError("formal preflight raw receipt launch coverage differs")


def publish_formal_preflight_remote_raw_evidence_receipt(
    dispatch: object,
    *,
    launch_cap_schedule_path: str | Path,
    launch_consumption_paths: tuple[str | Path, ...],
    compile_result_path: str | Path,
    exactness_result_path: str | Path,
    interference_raw_batch_path: str | Path,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish the only successful remote terminal: waiting for local control."""

    from lightcone_spec.runtime.proof_artifact import (
        publish_canonical_json_no_replace,
    )

    token = _verified(dispatch)
    receipt = FormalPreflightRemoteRawEvidenceReceipt(
        schema_version=2,
        kind="formal_preflight_remote_raw_evidence_receipt",
        status="WAITING_FOR_LOCAL_CONTROL",
        dispatch_sha256=token.sha256,
        launch_cap_schedule=CanonicalJsonProofBinding.bind(launch_cap_schedule_path),
        launch_consumptions=tuple(
            sorted(
                (
                    CanonicalJsonProofBinding.bind(path)
                    for path in launch_consumption_paths
                ),
                key=lambda row: row.absolute_path,
            )
        ),
        compile_result=CanonicalJsonProofBinding.bind(compile_result_path),
        exactness_result=CanonicalJsonProofBinding.bind(exactness_result_path),
        interference_raw_batch=CanonicalJsonProofBinding.bind(
            interference_raw_batch_path
        ),
    )
    receipt.revalidate(token)
    publish_canonical_json_no_replace(output_path, receipt.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    reloaded = FormalPreflightRemoteRawEvidenceReceipt.from_dict(binding.reopen())
    reloaded.revalidate(token)
    if reloaded != receipt:
        raise RuntimeError("written formal preflight raw receipt changed")
    return binding


def _load_formal_preflight_remote_raw_evidence_receipt(
    dispatch: object,
    path: str | Path,
) -> tuple[CanonicalJsonProofBinding, FormalPreflightRemoteRawEvidenceReceipt]:
    """Deep-reopen the stable-pull receipt and bind it to one sealed dispatch."""

    token = _verified(dispatch)
    binding = CanonicalJsonProofBinding.bind(path)
    receipt = FormalPreflightRemoteRawEvidenceReceipt.from_dict(binding.reopen())
    receipt.revalidate(token)
    if binding.semantic_sha256 != receipt.sha256:
        raise ValueError("formal preflight raw receipt semantic identity differs")
    return binding, receipt


@dataclass(frozen=True)
class FormalPreflightFinalEvidence:
    """Verifier-derived final evidence; no serialized summary is trusted."""

    remote_raw_receipt: CanonicalJsonProofBinding
    source_authority: PreflightExecutionSourceAuthority
    activation: RegistryStageActivationArtifact
    coverage: PreflightCoverageReceipt
    materialization: StageMaterializationReceipt
    stage_coverage: StageCoverageReceipt

    def __post_init__(self) -> None:
        from lightcone_spec.experiments.stage_materialization import (
            StageCoverageReceipt,
            StageMaterializationReceipt,
        )

        if type(self.remote_raw_receipt) is not CanonicalJsonProofBinding:
            raise TypeError("formal preflight final evidence lacks its raw receipt")
        if type(self.source_authority) is not PreflightExecutionSourceAuthority:
            raise TypeError(
                "formal preflight final evidence lacks raw source authority"
            )
        if type(self.activation) is not RegistryStageActivationArtifact:
            raise TypeError("formal preflight final evidence lacks typed activation")
        if type(self.coverage) is not PreflightCoverageReceipt:
            raise TypeError("formal preflight final evidence lacks typed coverage")
        if type(self.materialization) is not StageMaterializationReceipt:
            raise TypeError("formal preflight final evidence lacks materialization")
        if type(self.stage_coverage) is not StageCoverageReceipt:
            raise TypeError("formal preflight final evidence lacks stage coverage")
        if (
            self.coverage.source_authority != self.source_authority
            or self.coverage.activation_sha256 != self.activation.sha256
        ):
            raise ValueError("formal preflight final evidence identities differ")
        require_complete_preflight_coverage(self.coverage)
        self.stage_coverage.validate_against(self.materialization)
        if self.stage_coverage.stage != "preflight":
            raise ValueError("formal preflight final stage coverage is not preflight")

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 2,
                "kind": "formal_preflight_final_evidence",
                "protocol_sha256": FORMAL_PREFLIGHT_EXECUTION_PROTOCOL_SHA256,
                "remote_raw_receipt_sha256": (self.remote_raw_receipt.semantic_sha256),
                "source_authority_sha256": self.source_authority.sha256,
                "activation_sha256": self.activation.sha256,
                "coverage_sha256": self.coverage.sha256,
                "materialization_receipt_sha256": self.materialization.sha256,
                "stage_coverage_sha256": self.stage_coverage.sha256,
            }
        )


def _verified(
    value: object,
) -> VerifiedFormalPreflightDispatch:
    token = require_verified_formal_preflight_dispatch(value)
    if (
        token.manifest.registry_sha256 != token.dispatch_context.registry.sha256
        or token.subject.manifest_sha256 != token.manifest.sha256
        or token.subject.inventory_sha256 != token.dispatch_context.inventory.sha256
        or token.subject.dispatch_context_sha256 != token.dispatch_context.sha256
        or token.subject.dispatch_plan_sha256 != token.dispatch_plan.sha256
        or token.protocol_lock.sha256 != token.manifest.protocol_lock_sha256
    ):
        raise ValueError("sealed preflight dispatch internal identity differs")
    return token


def _bindings(
    token: VerifiedFormalPreflightDispatch,
) -> tuple[FormalPreflightExecutionBinding, ...]:
    rows = token.subject.execution_bindings
    counts = {
        kind: sum(row.runner_kind == kind for row in rows)
        for kind in (
            "first_party_compile",
            "first_party_exactness",
            "first_party_interference",
        )
    }
    if len(rows) != 10 or counts != {
        "first_party_compile": 1,
        "first_party_exactness": 1,
        "first_party_interference": 8,
    }:
        raise ValueError("sealed preflight dispatch runner coverage is not 1+1+8")
    return rows


def _one_binding(
    token: VerifiedFormalPreflightDispatch,
    runner_kind: FormalPreflightRunnerKind,
) -> FormalPreflightExecutionBinding:
    matches = tuple(row for row in _bindings(token) if row.runner_kind == runner_kind)
    if len(matches) != 1:
        raise ValueError(f"formal preflight {runner_kind} cardinality is not one")
    return matches[0]


def _activation(
    token: VerifiedFormalPreflightDispatch,
) -> RegistryStageActivationArtifact:
    activation = token.dispatch_context.activation_artifact
    if (
        type(activation) is not RegistryStageActivationArtifact
        or activation.experiment != "preflight"
        or activation.registry_sha256 != token.dispatch_context.registry.sha256
    ):
        raise ValueError("sealed preflight dispatch lacks its exact activation")
    return activation


def _source_bindings(
    binding: FormalPreflightExecutionBinding,
) -> dict[str, str]:
    value = dict(binding.source_authority_bindings)
    if set(value) != {
        "burstgpt_shape",
        "compile_qualification",
        "exactness_qualification",
        "formal_workload_e0",
        "formal_workload_e3a",
        "native_runtime_qualification",
        "offline_release_trust_root",
        "prepared_model_content",
    }:
        raise ValueError("formal preflight source-authority names differ")
    return value


def require_formal_preflight_compile_assignment(
    dispatch: object,
    *,
    assignment_plan_path: str | Path,
    prepared_content_verification_receipt_path: str | Path,
    now_ns: int,
) -> CompileAssignmentPlan:
    """Deep-bind one compile plan to the sealed scheduled compile row."""

    token = _verified(dispatch)
    binding = _one_binding(token, "first_party_compile")
    activation = _activation(token)
    plan = CompileAssignmentPlan.load(assignment_plan_path)
    assignment, _cache, _prewarm, launch = plan.revalidate()
    source = _source_bindings(binding)
    _content_binding, prepared = revalidate_prepared_content_verification_receipt(
        prepared_content_verification_receipt_path,
        current_ns=now_ns,
    )
    if (
        assignment.cell_id != binding.registry_cell_id
        or assignment.registry_sha256 != token.dispatch_context.registry.sha256
        or assignment.runtime_sha256 != activation.runtime_sha256
        or assignment.split_sha256 != activation.split_sha256
        or assignment.inventory_sha256 != token.subject.inventory_sha256
        or assignment.physical_assignment_sha256 != binding.assignment_sha256
        or assignment.experiment_budget_sha256 != binding.experiment_budget_sha256
        or assignment.gpu_uuids != binding.gpu_uuids
        or launch.physical_assignment_sha256 != binding.assignment_sha256
        or launch.experiment_budget_sha256 != binding.experiment_budget_sha256
        or launch.inventory_sha256 != token.subject.inventory_sha256
        or launch.gpu_uuids != binding.gpu_uuids
        or prepared.authorization_sha256 != source["prepared_model_content"]
        or launch.target_content_authority_sha256 != prepared.authorization_sha256
        or launch.tokenizer_content_authority_sha256 != prepared.authorization_sha256
        or (
            launch.drafter_content_authority_sha256 is not None
            and launch.drafter_content_authority_sha256 != prepared.authorization_sha256
        )
    ):
        raise ValueError("compile assignment differs from sealed preflight dispatch")
    return plan


def execute_formal_preflight_compile_raw(
    dispatch: object,
    *,
    launch_cap_schedule_path: str | Path,
    assignment_plan_path: str | Path,
    prepared_content_verification_receipt_path: str | Path,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> tuple[CompileResultPointer, CanonicalJsonProofBinding]:
    """Run the sole compile row after sealed and content identity replay."""

    token = _verified(dispatch)
    binding = _one_binding(token, "first_party_compile")
    require_formal_preflight_compile_assignment(
        dispatch,
        assignment_plan_path=assignment_plan_path,
        prepared_content_verification_receipt_path=(
            prepared_content_verification_receipt_path
        ),
        now_ns=now_ns,
    )
    cap_schedule = revalidate_formal_preflight_launch_cap_schedule(
        launch_cap_schedule_path,
        current_ns=now_ns,
    )
    cap = cap_schedule.cap_for_registry_cell(binding.registry_cell_id)
    if cap.runner_kind != "first_party_compile" or cap.wave_cell_ids != (
        binding.materialized_cell_id,
    ):
        raise ValueError("formal preflight compile cap is not one singleton row")
    launch_consumption = consume_formal_preflight_launch_wave(
        launch_cap_schedule_path,
        registry_cell_id=binding.registry_cell_id,
        consumed_ns=now_ns,
        current_ns=now_ns,
    )
    pointer = _execute_release_compile_assignment_plan_admitted(
        assignment_plan_path,
        control_attestation=control_attestation,
        prepared_content_verification_receipt_path=(
            prepared_content_verification_receipt_path
        ),
        replay_store=replay_store,
        now_ns=now_ns,
        timeout_seconds=cap.process_hard_timeout_ns / 1_000_000_000,
    )
    return pointer, launch_consumption


def require_formal_preflight_exactness_assignment(
    dispatch: object,
    *,
    assignment_path: str | Path,
) -> ExactnessPreflightAssignment:
    """Deep-bind one exactness assignment to the sealed scheduled row."""

    token = _verified(dispatch)
    binding = _one_binding(token, "first_party_exactness")
    activation = _activation(token)
    source = _source_bindings(binding)
    assignment = ExactnessPreflightAssignment.load(assignment_path)
    if (
        assignment.cell_id != binding.registry_cell_id
        or assignment.registry_sha256 != token.dispatch_context.registry.sha256
        or assignment.runtime_sha256 != activation.runtime_sha256
        or assignment.split_sha256 != activation.split_sha256
        or assignment.inventory_sha256 != token.subject.inventory_sha256
        or assignment.physical_assignment_sha256 != binding.assignment_sha256
        or assignment.experiment_budget_sha256 != binding.experiment_budget_sha256
        or assignment.gpu_uuids != binding.gpu_uuids
        or assignment.input_locks.prepared_model_content_authority_sha256
        != source["prepared_model_content"]
        or assignment.input_locks.formal_workload_lock_sha256
        != source["formal_workload_e3a"]
        or assignment.input_locks.burstgpt_shape_authority.sha256
        != source["burstgpt_shape"]
    ):
        raise ValueError("exactness assignment differs from sealed preflight dispatch")
    return assignment


def execute_formal_preflight_exactness_raw(
    dispatch: object,
    *,
    launch_cap_schedule_path: str | Path,
    assignment_path: str | Path,
    dispatch_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> tuple[ExactnessPreflightResultPointer, CanonicalJsonProofBinding]:
    """Run the sole exactness row and retain its raw schema-3 evidence."""

    token = _verified(dispatch)
    binding = _one_binding(token, "first_party_exactness")
    require_formal_preflight_exactness_assignment(
        dispatch,
        assignment_path=assignment_path,
    )
    cap_schedule = revalidate_formal_preflight_launch_cap_schedule(
        launch_cap_schedule_path,
        current_ns=now_ns,
    )
    cap = cap_schedule.cap_for_registry_cell(binding.registry_cell_id)
    if cap.runner_kind != "first_party_exactness" or cap.wave_cell_ids != (
        binding.materialized_cell_id,
    ):
        raise ValueError("formal preflight exactness cap is not one singleton row")
    launch_consumption = consume_formal_preflight_launch_wave(
        launch_cap_schedule_path,
        registry_cell_id=binding.registry_cell_id,
        consumed_ns=now_ns,
        current_ns=now_ns,
    )
    pointer = execute_release_exactness_preflight(
        assignment_path,
        dispatch_attestation=dispatch_attestation,
        replay_store=replay_store,
        now_ns=now_ns,
        timeout_seconds=cap.process_hard_timeout_ns / 1_000_000_000,
    )
    return pointer, launch_consumption


def qualify_formal_preflight_exactness_locally(
    dispatch: object,
    *,
    remote_raw_receipt_path: str | Path,
    rank_aggregate_control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    proof_artifact_path: str | Path,
    qualified_result_pointer_path: str | Path,
) -> ExactnessPreflightResultPointer:
    """Post-pull local qualification; no signing key is present on the GPU host."""

    token = _verified(dispatch)
    _raw_receipt_binding, remote = _load_formal_preflight_remote_raw_evidence_receipt(
        token,
        remote_raw_receipt_path,
    )
    raw_result_pointer_path = remote.exactness_result.absolute_path
    raw = ExactnessPreflightResultPointer.load(raw_result_pointer_path)
    assignment = require_formal_preflight_exactness_assignment(
        token,
        assignment_path=raw.assignment.absolute_path,
    )
    if raw.schema_version != 3 or assignment.sha256 != (
        ExactnessPreflightAssignment.load(raw.assignment.absolute_path).sha256
    ):
        raise ValueError("exactness local qualification requires sealed raw evidence")
    publish_release_exactness_qualification_proof(
        raw_result_pointer_path,
        control_attestation=rank_aggregate_control_attestation,
        replay_store=replay_store,
        now_ns=now_ns,
        proof_artifact_path=proof_artifact_path,
    )
    return finalize_release_exactness_preflight(
        raw_result_pointer_path,
        proof_artifact_path=proof_artifact_path,
        qualified_result_pointer_path=qualified_result_pointer_path,
        now_ns=now_ns,
    )


def qualify_formal_preflight_interference_locally(
    dispatch: object,
    *,
    remote_raw_receipt_path: str | Path,
    native_result_proof_paths: Mapping[str, str | Path],
    native_itl_proof_paths: Mapping[str, str | Path],
    aggregate_control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    proof_artifact_path: str | Path,
) -> VerifiedFormalPreflightInterferenceProof:
    """Authorize only the exact eight rows named by the stable-pull receipt.

    Failed paired diagnostics are still published durably and returned with a
    ``FAILED`` status.  They cannot pass the final coverage reducer.
    """

    token = _verified(dispatch)
    _raw_receipt_binding, remote = _load_formal_preflight_remote_raw_evidence_receipt(
        token,
        remote_raw_receipt_path,
    )
    activation = _activation(token)
    expected_root = _source_bindings(_one_binding(token, "first_party_exactness"))[
        "offline_release_trust_root"
    ]
    if (
        aggregate_control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root
    ):
        raise ValueError("interference aggregate control uses another release root")
    publish_formal_preflight_interference_proof_artifact(
        token,
        raw_batch_path=remote.interference_raw_batch.absolute_path,
        native_result_proof_paths=native_result_proof_paths,
        native_itl_proof_paths=native_itl_proof_paths,
        control_attestation=aggregate_control_attestation,
        replay_store=replay_store,
        now_ns=now_ns,
        proof_artifact_path=proof_artifact_path,
    )
    verified = validate_formal_preflight_interference_proof_artifact(
        proof_artifact_path,
        registry=token.dispatch_context.registry,
        expected_activation_sha256=activation.sha256,
        expected_runtime_sha256=activation.runtime_sha256,
        expected_split_sha256=activation.split_sha256,
        expected_inventory_sha256=token.subject.inventory_sha256,
        now_ns=now_ns,
    )
    artifact = FormalPreflightInterferenceProofArtifact.from_dict(
        CanonicalJsonProofBinding.bind(proof_artifact_path).reopen()
    )
    if artifact.raw_batch != remote.interference_raw_batch:
        raise ValueError("interference qualification belongs to another raw receipt")
    return verified


def _empty_safe_evidence_root(value: str | Path) -> Path:
    root = Path(value)
    if not root.is_absolute() or root != root.resolve(strict=False):
        raise ValueError("formal preflight evidence root is not canonical")
    status = root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(status.st_mode)
        or root.is_symlink()
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ValueError("formal preflight evidence root is not private")
    if any(root.iterdir()):
        raise ValueError("formal preflight evidence root must start empty")
    return root


@dataclass(frozen=True)
class _InterferenceExecutionAdmission:
    """Exact, mode-closed identity consumed by the shared exact-eight core."""

    authority_mode: Literal["formal_dispatch", "formal_single_operator_v1"]
    execution_authority_sha256: str
    protocol_lock_sha256: str
    registry_sha256: str
    activation_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    budget_plan_sha256: str
    execution_plan_sha256: str
    process_hard_timeout_ns: int
    bindings: tuple[FormalPreflightExecutionBinding, ...]

    def __post_init__(self) -> None:
        if self.authority_mode not in {
            "formal_dispatch",
            "formal_single_operator_v1",
        }:
            raise ValueError("preflight interference admission mode differs")
        for label, value in (
            ("execution authority", self.execution_authority_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("activation", self.activation_sha256),
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
            ("inventory", self.inventory_sha256),
            ("budget plan", self.budget_plan_sha256),
            ("execution plan", self.execution_plan_sha256),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"preflight interference {label} digest is invalid")
        if (
            type(self.process_hard_timeout_ns) is not int
            or self.process_hard_timeout_ns < 1_000_000_000
            or self.process_hard_timeout_ns > 3_600_000_000_000
        ):
            raise ValueError("preflight interference process timeout differs")
        if (
            type(self.bindings) is not tuple
            or len(self.bindings) != 8
            or self.bindings
            != tuple(sorted(self.bindings, key=lambda row: row.registry_cell_id))
            or len({row.registry_cell_id for row in self.bindings}) != 8
            or any(
                type(row) is not FormalPreflightExecutionBinding
                or row.runner_kind != "first_party_interference"
                for row in self.bindings
            )
        ):
            raise ValueError("preflight interference admission is not exact eight")


def _interference_admission_from_verified_dispatch(
    token: VerifiedFormalPreflightDispatch,
) -> _InterferenceExecutionAdmission:
    activation = token.dispatch_context.activation_artifact
    return _InterferenceExecutionAdmission(
        authority_mode="formal_dispatch",
        execution_authority_sha256=token.sha256,
        protocol_lock_sha256=token.protocol_lock.sha256,
        registry_sha256=token.dispatch_context.registry.sha256,
        activation_sha256=activation.sha256,
        runtime_sha256=activation.runtime_sha256,
        split_sha256=activation.split_sha256,
        inventory_sha256=token.subject.inventory_sha256,
        budget_plan_sha256=token.subject.budget_plan_sha256,
        execution_plan_sha256=token.dispatch_plan.sha256,
        process_hard_timeout_ns=30 * 60 * 1_000_000_000,
        bindings=tuple(
            sorted(
                (
                    row
                    for row in _bindings(token)
                    if row.runner_kind == "first_party_interference"
                ),
                key=lambda row: row.registry_cell_id,
            )
        ),
    )


def _interference_execution_bindings(
    admission: _InterferenceExecutionAdmission,
) -> dict[str, FormalPreflightExecutionBinding]:
    rows = {row.registry_cell_id: row for row in admission.bindings}
    if len(rows) != 8:
        raise ValueError("sealed preflight interference coverage is not eight")
    return rows


def _interference_mode_repetition_slot(
    binding: FormalPreflightExecutionBinding,
) -> tuple[Literal["isolated", "concurrent"], int, int]:
    variant = str(binding.cell.identity.variant)
    for mode in ("isolated", "concurrent"):
        for slot in range(2):
            if variant == f"{mode}_slot_{slot}":
                repetition = int(binding.cell.identity.block)
                if repetition not in {0, 1}:
                    break
                return mode, repetition, slot  # type: ignore[return-value]
    raise ValueError("sealed preflight interference schedule is unsupported")


def _validate_interference_run_input(
    admission: _InterferenceExecutionAdmission,
    binding: FormalPreflightExecutionBinding,
    execution_input: FormalPreflightInterferenceRunInput,
) -> None:
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

    execution_input.validate()
    launch = CompileLaunchManifest.load(execution_input.launch_manifest_path)
    if (
        execution_input.registry_cell_id != binding.registry_cell_id
        or launch.physical_assignment_sha256 != binding.assignment_sha256
        or launch.experiment_budget_sha256 != binding.experiment_budget_sha256
        or launch.inventory_sha256 != admission.inventory_sha256
        or launch.gpu_uuids != binding.gpu_uuids
        or launch.localhost_port not in binding.assignment.ports
        or execution_input.run_binding.execution_plan_sha256
        != admission.execution_plan_sha256
    ):
        raise ValueError("interference run input differs from sealed dispatch")


async def _execute_formal_preflight_interference_raw_core(
    admission: _InterferenceExecutionAdmission,
    *,
    launch_cap_schedule_path: str | Path | None,
    execution_inputs: dict[str, FormalPreflightInterferenceRunInput],
    nvidia_smi_tool: object,
    evidence_root: str | Path,
    now_ns: int,
) -> tuple[
    CanonicalJsonProofBinding,
    tuple[FormalPreflightInterferenceRawRow, ...],
    tuple[CanonicalJsonProofBinding, ...],
]:
    """Execute exact 4 isolated + 2 shared-barrier groups without a signer.

    This API intentionally returns only ``WAITING_FOR_LOCAL_CONTROL`` or
    ``ERROR``.  The local publisher must later create the eight native-result
    and ITL proofs plus the aggregate control before coverage can be complete.
    """

    from lightcone_spec.orchestration.live_sglang import (
        PinnedNvidiaSmiTool,
        PinnedSglangServingRunError,
        PinnedSglangServingRunSpec,
        execute_unsigned_native_serving_group,
        execute_unsigned_native_serving_run,
    )

    if type(admission) is not _InterferenceExecutionAdmission:
        raise TypeError("preflight interference admission is not exact")
    expected = _interference_execution_bindings(admission)
    cap_schedule = (
        None
        if launch_cap_schedule_path is None
        else revalidate_formal_preflight_launch_cap_schedule(
            launch_cap_schedule_path,
            current_ns=now_ns,
        )
    )
    if cap_schedule is not None and (
        admission.authority_mode != "formal_dispatch"
        or cap_schedule.protocol_lock_sha256 != admission.protocol_lock_sha256
        or cap_schedule.registry_sha256 != admission.registry_sha256
        or cap_schedule.inventory_sha256 != admission.inventory_sha256
        or cap_schedule.budget_plan_sha256 != admission.budget_plan_sha256
    ):
        raise ValueError(
            "formal preflight interference cap belongs to another dispatch"
        )
    if type(execution_inputs) is not dict or set(execution_inputs) != set(expected):
        raise ValueError("formal preflight execution inputs are not exact eight")
    if type(nvidia_smi_tool) is not PinnedNvidiaSmiTool:
        raise TypeError("formal preflight requires a pinned nvidia-smi tool")
    nvidia_smi_tool.revalidate()
    root = _empty_safe_evidence_root(evidence_root)
    inputs = dict(execution_inputs)
    for cell_id, value in inputs.items():
        if type(value) is not FormalPreflightInterferenceRunInput:
            raise TypeError("formal preflight execution input is not exact")
        _validate_interference_run_input(admission, expected[cell_id], value)

    row_directories: dict[str, Path] = {}
    qualification_paths: dict[str, Path] = {}
    for cell_id in sorted(expected):
        directory = root / cell_id
        directory.mkdir(mode=0o700)
        row_directories[cell_id] = directory
        qualification_path = directory / "qualification.json"
        qualification_paths[cell_id] = qualification_path
        _publish_single_operator_preflight_interference_qualification_lock(
            binding=expected[cell_id],
            registry_sha256=admission.registry_sha256,
            runtime_sha256=admission.runtime_sha256,
            split_sha256=admission.split_sha256,
            scored_requests=inputs[cell_id].scored_requests,
            rows=inputs[cell_id].qualification_rows,
            output_path=qualification_path,
        )

    rows: list[FormalPreflightInterferenceRawRow] = []
    launch_consumptions: list[CanonicalJsonProofBinding] = []

    async def execute_isolated(
        binding: FormalPreflightExecutionBinding,
        execution_input: FormalPreflightInterferenceRunInput,
    ) -> FormalPreflightInterferenceRawRow:
        if cap_schedule is None:
            if admission.authority_mode != "formal_single_operator_v1":
                raise ValueError("uncontrolled interference requires trusted authority")
            timeout_seconds = admission.process_hard_timeout_ns / 1_000_000_000
        else:
            cap = cap_schedule.cap_for_registry_cell(binding.registry_cell_id)
            if cap.runner_kind != "first_party_interference" or cap.wave_cell_ids != (
                binding.materialized_cell_id,
            ):
                raise ValueError("formal preflight isolated cap is not a singleton")
            assert launch_cap_schedule_path is not None
            launch_consumptions.append(
                consume_formal_preflight_launch_wave(
                    launch_cap_schedule_path,
                    registry_cell_id=binding.registry_cell_id,
                    consumed_ns=now_ns,
                    current_ns=now_ns,
                )
            )
            timeout_seconds = cap.process_hard_timeout_ns / 1_000_000_000
        directory = row_directories[binding.registry_cell_id]
        terminal_path = directory / "terminal.json"
        pointer_path = directory / "native-itl.json"
        receipt_path = directory / "live-run.json"
        source_fatal_path = directory / "live-fatal.json"
        try:
            result = await execute_unsigned_native_serving_run(
                launch_manifest_path=execution_input.launch_manifest_path,
                binding=execution_input.run_binding,
                warmup_requests=execution_input.warmup_requests,
                scored_requests=execution_input.scored_requests,
                terminal_output_path=terminal_path,
                native_itl_pointer_output_path=pointer_path,
                live_run_receipt_output_path=receipt_path,
                server_log_output_path=directory / "server.log",
                nvidia_smi_tool=nvidia_smi_tool,
                before_gpu_snapshot_output_path=directory / "gpu-before.json",
                ready_gpu_snapshot_output_path=directory / "gpu-ready.json",
                after_gpu_snapshot_output_path=directory / "gpu-after.json",
                fatal_output_path=source_fatal_path,
                timeout_seconds=timeout_seconds,
                lifecycle_timing_output_path=(
                    directory / "lifecycle.json"
                    if admission.authority_mode == "formal_single_operator_v1"
                    else None
                ),
            )
        except PinnedSglangServingRunError as error:
            if error.fatal_pointer is None:
                raise RuntimeError(
                    "live runner failed without a fatal pointer"
                ) from error
            fatal_path = directory / "row-fatal.json"
            publish_formal_preflight_interference_fatal_terminal(
                fatal_path,
                binding=binding,
                inventory_sha256=admission.inventory_sha256,
                run_binding=execution_input.run_binding,
                error_code=error.reason_code,
                source_fatal_terminal_path=error.fatal_pointer.absolute_path,
            )
            return build_formal_preflight_interference_raw_row(
                binding,
                inventory_sha256=admission.inventory_sha256,
                run_binding=execution_input.run_binding,
                fatal_terminal_path=fatal_path,
            )
        return build_formal_preflight_interference_raw_row(
            binding,
            inventory_sha256=admission.inventory_sha256,
            run_binding=execution_input.run_binding,
            nvidia_smi_tool=nvidia_smi_tool,
            launch_manifest_path=execution_input.launch_manifest_path,
            live_run_receipt_path=result.receipt_binding.absolute_path,
            raw_terminal_path=terminal_path,
            native_itl_pointer_artifact_path=pointer_path,
            qualification_lock_path=qualification_paths[binding.registry_cell_id],
        )

    schedule = {
        _interference_mode_repetition_slot(binding): binding
        for binding in expected.values()
    }
    for repetition in range(2):
        for slot in range(2):
            binding = schedule[("isolated", repetition, slot)]
            rows.append(
                await execute_isolated(binding, inputs[binding.registry_cell_id])
            )

    for repetition in range(2):
        pair = tuple(schedule[("concurrent", repetition, slot)] for slot in range(2))
        if cap_schedule is None:
            if admission.authority_mode != "formal_single_operator_v1":
                raise ValueError("uncontrolled interference requires trusted authority")
            timeout_seconds = admission.process_hard_timeout_ns / 1_000_000_000
        else:
            pair_caps = tuple(
                cap_schedule.cap_for_registry_cell(binding.registry_cell_id)
                for binding in pair
            )
            expected_wave = tuple(
                sorted(binding.materialized_cell_id for binding in pair)
            )
            if (
                any(cap.runner_kind != "first_party_interference" for cap in pair_caps)
                or any(cap.wave_cell_ids != expected_wave for cap in pair_caps)
                or len({cap.wave_index for cap in pair_caps}) != 1
                or len({cap.process_hard_timeout_ns for cap in pair_caps}) != 1
            ):
                raise ValueError(
                    "formal preflight concurrent cap is not one exact wave"
                )
            assert launch_cap_schedule_path is not None
            launch_consumptions.append(
                consume_formal_preflight_launch_wave(
                    launch_cap_schedule_path,
                    registry_cell_id=pair[0].registry_cell_id,
                    consumed_ns=now_ns,
                    current_ns=now_ns,
                )
            )
            timeout_seconds = pair_caps[0].process_hard_timeout_ns / 1_000_000_000
        group_directory = root / f"concurrent-group-{repetition}"
        group_directory.mkdir(mode=0o700)
        specs: list[PinnedSglangServingRunSpec] = []
        for binding in pair:
            execution_input = inputs[binding.registry_cell_id]
            directory = row_directories[binding.registry_cell_id]
            specs.append(
                PinnedSglangServingRunSpec(
                    launch_manifest=CanonicalJsonProofBinding.bind(
                        execution_input.launch_manifest_path
                    ),
                    binding=execution_input.run_binding,
                    warmup_requests=execution_input.warmup_requests,
                    scored_requests=execution_input.scored_requests,
                    terminal_output_path=str(directory / "terminal.json"),
                    native_itl_pointer_output_path=str(directory / "native-itl.json"),
                    live_run_receipt_output_path=str(directory / "live-run.json"),
                    server_log_output_path=str(directory / "server.log"),
                    lifecycle_timing_output_path=(
                        str(directory / "lifecycle.json")
                        if admission.authority_mode == "formal_single_operator_v1"
                        else None
                    ),
                )
            )
        try:
            group = await execute_unsigned_native_serving_group(
                specs=(specs[0], specs[1]),
                nvidia_smi_tool=nvidia_smi_tool,
                inventory_sha256=admission.inventory_sha256,
                before_gpu_snapshot_output_path=group_directory / "gpu-before.json",
                ready_gpu_snapshot_output_path=group_directory / "gpu-ready.json",
                after_gpu_snapshot_output_path=group_directory / "gpu-after.json",
                group_receipt_output_path=group_directory / "group-receipt.json",
                fatal_output_path=group_directory / "group-fatal.json",
                timeout_seconds=timeout_seconds,
            )
        except PinnedSglangServingRunError as error:
            if error.fatal_pointer is None:
                raise RuntimeError(
                    "live group failed without a fatal pointer"
                ) from error
            for binding in pair:
                execution_input = inputs[binding.registry_cell_id]
                directory = row_directories[binding.registry_cell_id]
                fatal_path = directory / "row-fatal.json"
                publish_formal_preflight_interference_fatal_terminal(
                    fatal_path,
                    binding=binding,
                    inventory_sha256=admission.inventory_sha256,
                    run_binding=execution_input.run_binding,
                    error_code=error.reason_code,
                    source_fatal_terminal_path=error.fatal_pointer.absolute_path,
                )
                rows.append(
                    build_formal_preflight_interference_raw_row(
                        binding,
                        inventory_sha256=admission.inventory_sha256,
                        run_binding=execution_input.run_binding,
                        fatal_terminal_path=fatal_path,
                    )
                )
            continue
        for index, binding in enumerate(pair):
            execution_input = inputs[binding.registry_cell_id]
            run = group.runs[index]
            rows.append(
                build_formal_preflight_interference_raw_row(
                    binding,
                    inventory_sha256=admission.inventory_sha256,
                    run_binding=execution_input.run_binding,
                    nvidia_smi_tool=nvidia_smi_tool,
                    launch_manifest_path=execution_input.launch_manifest_path,
                    live_run_receipt_path=run.receipt_binding.absolute_path,
                    raw_terminal_path=specs[index].terminal_output_path,
                    native_itl_pointer_artifact_path=(
                        specs[index].native_itl_pointer_output_path
                    ),
                    qualification_lock_path=qualification_paths[
                        binding.registry_cell_id
                    ],
                    concurrent_group_receipt_path=(group.receipt_binding.absolute_path),
                )
            )

    raw_path = root / "interference-raw-batch.json"
    raw_binding = _publish_single_operator_preflight_interference_raw_batch(
        execution_authority_sha256=admission.execution_authority_sha256,
        registry_sha256=admission.registry_sha256,
        activation_sha256=admission.activation_sha256,
        runtime_sha256=admission.runtime_sha256,
        split_sha256=admission.split_sha256,
        inventory_sha256=admission.inventory_sha256,
        expected_bindings=tuple(expected.values()),
        rows=tuple(rows),
        nvidia_smi_tool=nvidia_smi_tool,
        output_path=raw_path,
    )
    batch = FormalPreflightInterferenceRawBatch.from_dict(raw_binding.reopen())
    batch.revalidate()
    return (
        raw_binding,
        batch.rows,
        tuple(sorted(launch_consumptions, key=lambda row: row.absolute_path)),
    )


async def execute_formal_preflight_interference_raw(
    dispatch: object,
    *,
    launch_cap_schedule_path: str | Path,
    execution_inputs: dict[str, FormalPreflightInterferenceRunInput],
    nvidia_smi_tool: object,
    evidence_root: str | Path,
    now_ns: int,
) -> FormalPreflightRemoteInterferenceEvidence:
    """Execute exact 4 isolated + 2 paired waves from a sealed dispatch."""

    token = _verified(dispatch)
    admission = _interference_admission_from_verified_dispatch(token)
    (
        raw_binding,
        rows,
        consumptions,
    ) = await _execute_formal_preflight_interference_raw_core(
        admission,
        launch_cap_schedule_path=launch_cap_schedule_path,
        execution_inputs=execution_inputs,
        nvidia_smi_tool=nvidia_smi_tool,
        evidence_root=evidence_root,
        now_ns=now_ns,
    )
    batch = FormalPreflightInterferenceRawBatch.from_dict(raw_binding.reopen())
    return FormalPreflightRemoteInterferenceEvidence(
        status=batch.status,
        raw_batch=raw_binding,
        rows=rows,
        launch_consumptions=consumptions,
    )


def _require_results_match_sealed_dispatch(
    token: VerifiedFormalPreflightDispatch,
    *,
    remote_raw_receipt: FormalPreflightRemoteRawEvidenceReceipt,
    exactness_result_path: str | Path,
    interference_proof_artifact_path: str | Path,
    now_ns: int,
) -> tuple[str, str]:
    compile_result_path = remote_raw_receipt.compile_result.absolute_path
    compile_binding = _one_binding(token, "first_party_compile")
    compile_pointer = CompileResultPointer.load(compile_result_path)
    if compile_pointer.assignment_plan_source is None:
        raise ValueError("formal preflight compile result lost its assignment")
    compile_plan = CompileAssignmentPlan.load(
        compile_pointer.assignment_plan_source.absolute_path
    )
    compile_assignment, _cache, _prewarm, _launch = compile_plan.revalidate()
    exactness_binding = _one_binding(token, "first_party_exactness")
    exactness_pointer = ExactnessPreflightResultPointer.load(exactness_result_path)
    exactness_assignment = ExactnessPreflightAssignment.load(
        exactness_pointer.assignment.absolute_path
    )
    activation = _activation(token)
    if exactness_pointer.qualification_proof_artifact is None:
        raise ValueError("formal preflight exactness result lacks local qualification")
    exactness_proof = ExactnessQualificationProofArtifact.load(
        exactness_pointer.qualification_proof_artifact.absolute_path
    )
    exactness_proof.revalidate(now_ns=now_ns)
    if (
        compile_pointer.schema_version != 3
        or compile_pointer.formal_execution_authorized is not True
        or compile_assignment.cell_id != compile_binding.registry_cell_id
        or compile_assignment.physical_assignment_sha256
        != compile_binding.assignment_sha256
        or compile_assignment.experiment_budget_sha256
        != compile_binding.experiment_budget_sha256
        or compile_assignment.gpu_uuids != compile_binding.gpu_uuids
        or exactness_pointer.schema_version != 4
        or exactness_assignment.cell_id != exactness_binding.registry_cell_id
        or exactness_assignment.physical_assignment_sha256
        != exactness_binding.assignment_sha256
        or exactness_assignment.experiment_budget_sha256
        != exactness_binding.experiment_budget_sha256
        or exactness_assignment.gpu_uuids != exactness_binding.gpu_uuids
        or exactness_proof.payload.raw_result_pointer.absolute_path
        != remote_raw_receipt.exactness_result.absolute_path
        or exactness_proof.payload.raw_result_pointer_semantic_sha256
        != remote_raw_receipt.exactness_result.semantic_sha256
    ):
        raise ValueError("preflight compile/exactness results differ from sealed rows")

    interference = validate_formal_preflight_interference_proof_artifact(
        interference_proof_artifact_path,
        registry=token.dispatch_context.registry,
        expected_activation_sha256=activation.sha256,
        expected_runtime_sha256=activation.runtime_sha256,
        expected_split_sha256=activation.split_sha256,
        expected_inventory_sha256=token.subject.inventory_sha256,
        now_ns=now_ns,
    )
    if (
        interference.binding.dispatch_sha256 != token.sha256
        or interference.binding.status != "PASSED"
    ):
        raise ValueError("preflight interference aggregate is not sealed/pass")
    interference_artifact = FormalPreflightInterferenceProofArtifact.from_dict(
        CanonicalJsonProofBinding.bind(interference_proof_artifact_path).reopen()
    )
    if interference_artifact.raw_batch != remote_raw_receipt.interference_raw_batch:
        raise ValueError("preflight interference proof belongs to another raw receipt")
    expected = {
        row.registry_cell_id: row
        for row in _bindings(token)
        if row.runner_kind == "first_party_interference"
    }
    seen: set[str] = set()
    for proof_row in interference.rows:
        binding = expected.get(proof_row.registry_cell_id)
        if (
            binding is None
            or proof_row.registry_cell_id in seen
            or proof_row.assignment_sha256 != binding.assignment_sha256
            or proof_row.experiment_budget_sha256 != binding.experiment_budget_sha256
            or (proof_row.gpu_uuid,) != binding.gpu_uuids
        ):
            raise ValueError(
                "preflight interference result differs from sealed assignment"
            )
        seen.add(proof_row.registry_cell_id)
    if seen != set(expected) or len(seen) != 8:
        raise ValueError("preflight interference results are not exact eight")
    return activation.runtime_sha256, activation.split_sha256


def finalize_formal_preflight_evidence(
    dispatch: object,
    *,
    remote_raw_receipt_path: str | Path,
    exactness_result_path: str | Path,
    interference_proof_artifact_path: str | Path,
    qualification_proof_paths: dict[str, tuple[str | Path, str | Path]],
    materialization: StageMaterializationReceipt,
    candidate_state_coverage: TtsL0CandidateStateCoverage,
    candidate_replay_proof_paths: tuple[str | Path, str | Path],
    now_ns: int,
) -> FormalPreflightFinalEvidence:
    """Derive pointer and signable stage coverage from the stable-pull receipt."""

    from lightcone_spec.experiments.stage_materialization import (
        StageMaterializationReceipt,
    )

    token = _verified(dispatch)
    raw_receipt_binding, raw_receipt = (
        _load_formal_preflight_remote_raw_evidence_receipt(
            token,
            remote_raw_receipt_path,
        )
    )
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal preflight finalization requires exact materialization")
    if (
        materialization.stage != "preflight"
        or materialization.protocol_lock_sha256 != token.protocol_lock.sha256
        or {cell.cell_id for cell in materialization.cells}
        != {row.materialized_cell_id for row in _bindings(token)}
    ):
        raise ValueError("preflight materialization differs from sealed dispatch")
    runtime_sha256, split_sha256 = _require_results_match_sealed_dispatch(
        token,
        remote_raw_receipt=raw_receipt,
        exactness_result_path=exactness_result_path,
        interference_proof_artifact_path=interference_proof_artifact_path,
        now_ns=now_ns,
    )
    registry: ExperimentRegistry = token.dispatch_context.registry
    source = PreflightExecutionSourceAuthority.bind(
        registry=registry,
        dispatch_activation_sha256=_activation(token).sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        release_root_manifest_sha256=_source_bindings(
            _one_binding(token, "first_party_exactness")
        )["offline_release_trust_root"],
        compile_result_path=raw_receipt.compile_result.absolute_path,
        exactness_result_path=exactness_result_path,
        interference_proof_artifact_path=interference_proof_artifact_path,
        qualification_proof_paths=qualification_proof_paths,
        now_ns=now_ns,
    )
    activation, coverage = materialize_pointer_preflight_coverage(registry, source)
    stage_coverage = materialize_formal_preflight_stage_coverage(
        materialization,
        coverage,
        candidate_state_coverage=candidate_state_coverage,
        candidate_replay_proof_paths=candidate_replay_proof_paths,
        now_ns=now_ns,
    )
    evidence = FormalPreflightFinalEvidence(
        remote_raw_receipt=raw_receipt_binding,
        source_authority=source,
        activation=activation,
        coverage=coverage,
        materialization=materialization,
        stage_coverage=stage_coverage,
    )
    require_complete_preflight_coverage(evidence.coverage)
    return evidence


__all__ = [
    "FORMAL_PREFLIGHT_EXECUTION_PROTOCOL_SHA256",
    "FormalPreflightExecutionBlocked",
    "FormalPreflightFinalEvidence",
    "FormalPreflightInterferenceExecutionManifest",
    "FormalPreflightInterferenceRunInput",
    "FormalPreflightRemoteInterferenceEvidence",
    "FormalPreflightRemoteRawEvidenceReceipt",
    "execute_formal_preflight_compile_raw",
    "execute_formal_preflight_exactness_raw",
    "execute_formal_preflight_interference_raw",
    "finalize_formal_preflight_evidence",
    "publish_formal_preflight_remote_raw_evidence_receipt",
    "qualify_formal_preflight_exactness_locally",
    "qualify_formal_preflight_interference_locally",
    "require_formal_preflight_compile_assignment",
    "require_formal_preflight_exactness_assignment",
]
