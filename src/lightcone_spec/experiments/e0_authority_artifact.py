"""Durable, path-bound reconstruction artifact for the formal E0 append.

The artifact deliberately stores no verifier-private execution or stage-source
seal.  It contains only typed signed inputs and immutable bindings to the
public reconstruction inputs and raw proof DAG.  Loading is a deep operation:
every reference is reopened, private bindings are rebuilt by the public
verifiers, and the source/tuning/power reductions are recomputed before the
bundle may enter the formal registry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.downstream_stage_authority import (
    E1aConfigurationEvaluation,
    E1aVerificationReceipt,
    E3bConfirmationReceipt,
    E5ConfirmationReceipt,
    E5FailureCellEvidence,
    E5FailureEvidenceManifest,
    E5P99AnchorCompletion,
    FormalDownstreamCellEvidence,
    FormalDownstreamEvidenceManifest,
    FormalFamilyConfirmationResult,
    SignedE1aVerificationReceipt,
    SignedE3bConfirmationReceipt,
    SignedE5ConfirmationReceipt,
)
from lightcone_spec.experiments.e0_stage_authority import (
    E0FormalRegistryAuthorityBundle,
    E0OnlineSpecSourceAuthority,
    E0OnlineSpecTuningProofSet,
    E6ConfirmationProofBundle,
    SignedE0OnlineSpecTuningSeal,
    SignedE0PowerPrefixReceipt,
)
from lightcone_spec.experiments.e1_stage_authority import (
    _request_identity,
    _validated_cell,
)
from lightcone_spec.experiments.e2_stage_authority import (
    E2StagedRoundEvidenceManifest,
)
from lightcone_spec.experiments.e4_stage_authority import E4StagedEvidenceManifest
from lightcone_spec.experiments.e6_stage_authority import (
    E6ConfirmationReceipt,
    E6ModelCompatibilityReceipt,
    E6NextnModelAuthorityInput,
    E6NextnModelCompatibility,
    SignedE6ConfirmationReceipt,
    SignedE6ModelCompatibilityReceipt,
)
from lightcone_spec.experiments.formal_failure_execution import (
    FormalFailureExecutionRebuildInput,
    VerifiedFormalFailureExecutionBinding,
    rebuild_formal_failure_execution_binding,
)
from lightcone_spec.experiments.formal_protocol import (
    content_sha256,
    verify_signed_payload,
)
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
    e0_onlinespec_source_authority_from_dict,
    e0_onlinespec_source_authority_to_dict,
    e2_staged_evidence_manifest_from_dict,
    e4_staged_evidence_manifest_from_dict,
    formal_registry_verification_receipt_from_dict,
    formal_runtime_authority_manifest_from_dict,
    signed_e0_compatibility_from_dict,
    signed_e0_compatibility_to_dict,
    signed_e0_onlinespec_tuning_seal_from_dict,
    signed_e0_onlinespec_tuning_seal_to_dict,
    signed_e0_power_prefix_from_dict,
    signed_e0_power_prefix_to_dict,
    signed_stage_materialization_from_dict,
    stage_coverage_receipt_from_dict,
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_stage_execution import (
    E0FinalStageSourceRebuildInputs,
    E0PilotStageSourceRebuildInputs,
    E0TuningStageSourceRebuildInputs,
    E1aStageSourceRebuildInputs,
    E3bFinalStageSourceRebuildInputs,
    E3bPilotStageSourceRebuildInputs,
    E4LocalStageSourceRebuildInputs,
    E4ScreenStageSourceRebuildInputs,
    E5FinalStageSourceRebuildInputs,
    E5PilotStageSourceRebuildInputs,
    E6FinalStageSourceRebuildInputs,
    E6PilotStageSourceRebuildInputs,
    FormalServingExecutionRebuildInput,
    FormalStageSourceRebuildInput,
    VerifiedFormalServingExecutionBinding,
    VerifiedFormalStageMaterializationSource,
    rebuild_formal_serving_execution_binding,
    rebuild_formal_stage_materialization_source,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.stage_materialization import (
    E0CompatibilityDecision,
    E1Geometry,
    E2CandidateRecipe,
    SignedE0CompatibilityReceipt,
    SignedStageMaterializationReceipt,
    StageCoverageReceipt,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
    default_e5_failure_diagnostic_authority,
)
from lightcone_spec.experiments.statistics import (
    TTFT_LIMIT_MS,
    WITHIN_REQUEST_P99_ITL_LIMIT_MS,
    MultiplicityDecision,
    PairedBcaContrast,
    SloRequest,
    account_slo,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

E0_AUTHORITY_BUNDLE_ARTIFACT_KIND = (
    "lightcone_e0_formal_registry_authority_bundle_artifact"
)
E0_EXECUTION_REBUILD_SHARD_KIND = "lightcone_e0_execution_rebuild_shard"
E5_FAILURE_EXECUTION_REBUILD_SHARD_KIND = "lightcone_e5_failure_execution_rebuild_shard"
E6_RECURSIVE_SOURCE_DAG_ARTIFACT_KIND = "lightcone_e6_recursive_source_dag_artifact"
FORMAL_STAGE_PROOF_NODE_KIND = "lightcone_formal_stage_proof_node"
E0_FINAL_RESULT_REBUILD_ARTIFACT_KIND = "lightcone_e0_final_result_rebuild_artifact"
E0_FINAL_COMPLETION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e0_final_completion_protocol",
        "materialization": "exact_16VN_final_rows_no_pilots",
        "coverage": "current_registry_signed_all_complete",
        "source": "deep_rebuilt_E2_through_E0_typed_authority_DAG",
        "execution": "one_private_sealed_binding_and_two_path_bound_proofs_per_cell",
        "safety": "zero_safety_reasons_and_adaptive_publication_present",
        "claim": "completion_only_no_statistical_or_performance_claim",
    }
)
E0_FINAL_SLO_GOODPUT_POLICY_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e0_task_native_slo_goodput_policy",
        "eligibility": "all_scored_task_native_requests",
        "prompt_bucket_by_input_tokens": {
            "short": "1..2048",
            "medium": "2049..8192",
            "long": "8193_plus",
        },
        "ttft_limits_ms": sorted(TTFT_LIMIT_MS.items()),
        "within_request_p99_itl_limit_ms": WITHIN_REQUEST_P99_ITL_LIMIT_MS,
        "qualified_tokens": "output_tokens_from_individually_qualified_requests",
        "scored_window": "max_request_terminal_ns_minus_min_request_started_ns",
        "source": "deep_opened_native_terminal_plus_stage_itl",
    }
)

FormalProofNodeId = Literal[
    "e2_final",
    "e4_screen",
    "e4_local",
    "e4_profiler",
    "e3b_pilot",
    "e3b_final",
    "e1a",
    "e5_pilot",
    "e5_final",
    "e6_pilot",
]

_FORMAL_PROOF_NODE_ORDER: tuple[FormalProofNodeId, ...] = (
    "e2_final",
    "e4_screen",
    "e4_local",
    "e4_profiler",
    "e3b_pilot",
    "e3b_final",
    "e1a",
    "e5_pilot",
    "e5_final",
    "e6_pilot",
)

_NODE_STAGE: dict[FormalProofNodeId, str] = {
    "e2_final": "E2",
    "e4_screen": "E4",
    "e4_local": "E4",
    "e4_profiler": "E4",
    "e3b_pilot": "E3b",
    "e3b_final": "E3b",
    "e1a": "E1a",
    "e5_pilot": "E5",
    "e5_final": "E5",
    "e6_pilot": "E6",
}


def _strict(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _array(label: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _challenge_from_dict(value: object) -> AttestationChallenge:
    return AttestationChallenge(
        **_strict(
            "attestation challenge",
            value,
            set(AttestationChallenge.__dataclass_fields__),
        )
    )


def _attestation_from_dict(value: object) -> SignedAttestation:
    return SignedAttestation(
        **_strict(
            "signed attestation",
            value,
            set(SignedAttestation.__dataclass_fields__),
        )
    )


def formal_downstream_evidence_manifest_to_dict(
    value: FormalDownstreamEvidenceManifest,
) -> dict[str, object]:
    if type(value) is not FormalDownstreamEvidenceManifest:
        raise TypeError("downstream evidence codec requires an exact manifest")
    return {**asdict(value), "manifest_sha256": value.sha256}


def formal_downstream_evidence_manifest_from_dict(
    value: object,
) -> FormalDownstreamEvidenceManifest:
    row = _strict(
        "formal downstream evidence manifest",
        value,
        {*FormalDownstreamEvidenceManifest.__dataclass_fields__, "manifest_sha256"},
    )
    declared = _sha256(
        "formal downstream evidence manifest", row.pop("manifest_sha256")
    )
    cells = []
    for item in _array("formal downstream evidence cells", row["cells"]):
        cell = _strict(
            "formal downstream evidence cell",
            item,
            set(FormalDownstreamCellEvidence.__dataclass_fields__),
        )
        cell["execution_identity"] = StageItlExecutionIdentity.from_dict(
            cell["execution_identity"]
        )
        cells.append(FormalDownstreamCellEvidence(**cell))  # type: ignore[arg-type]
    row["cells"] = tuple(cells)
    manifest = FormalDownstreamEvidenceManifest(**row)  # type: ignore[arg-type]
    if manifest.sha256 != declared:
        raise ValueError("downstream evidence manifest digest differs from content")
    return manifest


def _family_confirmation_from_dict(
    label: str,
    value: object,
) -> FormalFamilyConfirmationResult:
    family = _strict(
        f"{label} family confirmation",
        value,
        set(FormalFamilyConfirmationResult.__dataclass_fields__),
    )
    family["family_dimensions"] = tuple(
        tuple(row)
        for row in _array(f"{label} family dimensions", family["family_dimensions"])
    )
    family["final_block_ids"] = tuple(
        _array(f"{label} family final blocks", family["final_block_ids"])
    )
    family["final_goodput_observation_sha256s"] = tuple(
        tuple(row)
        for row in _array(
            f"{label} family observations",
            family["final_goodput_observation_sha256s"],
        )
    )
    contrasts = []
    for contrast_value in _array(
        f"{label} family primary contrasts", family["primary_contrasts"]
    ):
        contrast = _strict(
            f"{label} family paired contrast",
            contrast_value,
            set(PairedBcaContrast.__dataclass_fields__),
        )
        contrast["block_ids"] = tuple(
            _array(
                f"{label} family paired contrast blocks",
                contrast["block_ids"],
            )
        )
        contrasts.append(PairedBcaContrast(**contrast))
    family["primary_contrasts"] = tuple(contrasts)
    family["holm_decisions"] = tuple(
        MultiplicityDecision(
            **_strict(
                f"{label} family Holm decision",
                decision,
                set(MultiplicityDecision.__dataclass_fields__),
            )
        )
        for decision in _array(
            f"{label} family Holm decisions", family["holm_decisions"]
        )
    )
    return FormalFamilyConfirmationResult(**family)


def signed_e5_confirmation_to_dict(
    value: SignedE5ConfirmationReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE5ConfirmationReceipt:
        raise TypeError("signed E5 confirmation codec requires an exact wrapper")
    return {
        "payload": asdict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e5_confirmation_from_dict(value: object) -> SignedE5ConfirmationReceipt:
    row = _strict(
        "signed E5 confirmation",
        value,
        {
            "payload",
            "payload_sha256",
            "challenge",
            "attestation",
            "signed_receipt_sha256",
        },
    )
    declared = _sha256("signed E5 confirmation", row.pop("signed_receipt_sha256"))
    payload_row = _strict(
        "E5 confirmation payload",
        row.pop("payload"),
        set(E5ConfirmationReceipt.__dataclass_fields__),
    )
    payload_row["family_confirmations"] = tuple(
        _family_confirmation_from_dict("E5", item)
        for item in _array(
            "E5 family confirmations", payload_row["family_confirmations"]
        )
    )
    payload_row["p99_anchor_completions"] = tuple(
        E5P99AnchorCompletion(
            **_strict(
                "E5 p99 anchor completion",
                item,
                set(E5P99AnchorCompletion.__dataclass_fields__),
            )
        )
        for item in _array(
            "E5 p99 anchor completions", payload_row["p99_anchor_completions"]
        )
    )
    payload_row["failure_result_sha256s"] = tuple(
        _array("E5 failure result digests", payload_row["failure_result_sha256s"])
    )
    signed = SignedE5ConfirmationReceipt(
        payload=E5ConfirmationReceipt(**payload_row),  # type: ignore[arg-type]
        payload_sha256=row["payload_sha256"],  # type: ignore[arg-type]
        challenge=_challenge_from_dict(row["challenge"]),
        attestation=_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E5 confirmation digest differs from content")
    return signed


def signed_e3b_confirmation_to_dict(
    value: SignedE3bConfirmationReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE3bConfirmationReceipt:
        raise TypeError("signed E3b confirmation codec requires an exact wrapper")
    return {
        "payload": asdict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e3b_confirmation_from_dict(
    value: object,
) -> SignedE3bConfirmationReceipt:
    row = _strict(
        "signed E3b confirmation",
        value,
        {
            "payload",
            "payload_sha256",
            "challenge",
            "attestation",
            "signed_receipt_sha256",
        },
    )
    declared = _sha256("signed E3b confirmation", row.pop("signed_receipt_sha256"))
    payload_row = _strict(
        "E3b confirmation payload",
        row.pop("payload"),
        set(E3bConfirmationReceipt.__dataclass_fields__),
    )
    payload_row["final_block_ids"] = tuple(
        _array("E3b confirmation final blocks", payload_row["final_block_ids"])
    )
    payload_row["family_confirmations"] = tuple(
        _family_confirmation_from_dict("E3b", item)
        for item in _array(
            "E3b family confirmations", payload_row["family_confirmations"]
        )
    )
    signed = SignedE3bConfirmationReceipt(
        payload=E3bConfirmationReceipt(**payload_row),  # type: ignore[arg-type]
        payload_sha256=row["payload_sha256"],  # type: ignore[arg-type]
        challenge=_challenge_from_dict(row["challenge"]),
        attestation=_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E3b confirmation digest differs from content")
    return signed


def signed_e1a_verification_to_dict(
    value: SignedE1aVerificationReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE1aVerificationReceipt:
        raise TypeError("signed E1a verification codec requires an exact wrapper")
    return {
        "payload": asdict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def _configuration_from_json(
    label: str,
    value: object,
) -> tuple[tuple[str, str | int], ...]:
    result: list[tuple[str, str | int]] = []
    for item in _array(label, value):
        if (
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) not in {str, int}
        ):
            raise TypeError(f"{label} entry is not a typed name/value pair")
        result.append((item[0], item[1]))
    return tuple(result)


def signed_e1a_verification_from_dict(
    value: object,
) -> SignedE1aVerificationReceipt:
    row = _strict(
        "signed E1a verification",
        value,
        {
            "payload",
            "payload_sha256",
            "challenge",
            "attestation",
            "signed_receipt_sha256",
        },
    )
    declared = _sha256("signed E1a verification", row.pop("signed_receipt_sha256"))
    payload_row = _strict(
        "E1a verification payload",
        row.pop("payload"),
        set(E1aVerificationReceipt.__dataclass_fields__),
    )
    evaluations = []
    for item in _array("E1a evaluations", payload_row["evaluations"]):
        evaluation = _strict(
            "E1a evaluation",
            item,
            set(E1aConfigurationEvaluation.__dataclass_fields__),
        )
        evaluation["configuration"] = _configuration_from_json(
            "E1a evaluation configuration", evaluation["configuration"]
        )
        evaluation["cell_ids"] = tuple(
            _array("E1a evaluation cells", evaluation["cell_ids"])
        )
        evaluations.append(E1aConfigurationEvaluation(**evaluation))  # type: ignore[arg-type]
    payload_row["evaluations"] = tuple(evaluations)
    payload_row["selected_configuration"] = _configuration_from_json(
        "E1a selected configuration", payload_row["selected_configuration"]
    )
    signed = SignedE1aVerificationReceipt(
        payload=E1aVerificationReceipt(**payload_row),  # type: ignore[arg-type]
        payload_sha256=row["payload_sha256"],  # type: ignore[arg-type]
        challenge=_challenge_from_dict(row["challenge"]),
        attestation=_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E1a verification digest differs from content")
    return signed


def e5_failure_evidence_manifest_to_dict(
    value: E5FailureEvidenceManifest,
) -> dict[str, object]:
    if type(value) is not E5FailureEvidenceManifest:
        raise TypeError("E5 failure evidence codec requires an exact manifest")
    return {**asdict(value), "manifest_sha256": value.sha256}


def e5_failure_evidence_manifest_from_dict(
    value: object,
) -> E5FailureEvidenceManifest:
    row = _strict(
        "E5 failure evidence manifest",
        value,
        {*E5FailureEvidenceManifest.__dataclass_fields__, "manifest_sha256"},
    )
    declared = _sha256("E5 failure evidence manifest", row.pop("manifest_sha256"))
    row["cells"] = tuple(
        E5FailureCellEvidence(
            **_strict(
                "E5 failure evidence cell",
                item,
                set(E5FailureCellEvidence.__dataclass_fields__),
            )
        )
        for item in _array("E5 failure evidence cells", row["cells"])
    )
    manifest = E5FailureEvidenceManifest(**row)  # type: ignore[arg-type]
    if manifest.sha256 != declared:
        raise ValueError("E5 failure evidence manifest digest differs from content")
    return manifest


def e6_nextn_model_authority_input_to_dict(
    value: E6NextnModelAuthorityInput,
) -> dict[str, object]:
    if type(value) is not E6NextnModelAuthorityInput:
        raise TypeError("E6 NEXTN source codec requires an exact typed input")
    return {**asdict(value), "source_input_sha256": value.sha256}


def e6_nextn_model_authority_input_from_dict(
    value: object,
) -> E6NextnModelAuthorityInput:
    row = _strict(
        "E6 NEXTN source input",
        value,
        {*E6NextnModelAuthorityInput.__dataclass_fields__, "source_input_sha256"},
    )
    declared = _sha256("E6 NEXTN source input", row.pop("source_input_sha256"))
    source = E6NextnModelAuthorityInput(**row)  # type: ignore[arg-type]
    if source.sha256 != declared:
        raise ValueError("E6 NEXTN source input digest differs from content")
    return source


def signed_e6_model_compatibility_to_dict(
    value: SignedE6ModelCompatibilityReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE6ModelCompatibilityReceipt:
        raise TypeError("signed E6 compatibility codec requires an exact wrapper")
    return {
        "payload": asdict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e6_model_compatibility_from_dict(
    value: object,
) -> SignedE6ModelCompatibilityReceipt:
    row = _strict(
        "signed E6 model compatibility",
        value,
        {
            "payload",
            "payload_sha256",
            "challenge",
            "attestation",
            "signed_receipt_sha256",
        },
    )
    declared = _sha256(
        "signed E6 model compatibility", row.pop("signed_receipt_sha256")
    )
    payload_row = _strict(
        "E6 model compatibility payload",
        row.pop("payload"),
        set(E6ModelCompatibilityReceipt.__dataclass_fields__),
    )
    payload_row["models"] = tuple(
        E6NextnModelCompatibility(
            **{
                **_strict(
                    "E6 model compatibility row",
                    item,
                    set(E6NextnModelCompatibility.__dataclass_fields__),
                ),
                "gpu_uuids": tuple(
                    _array(
                        "E6 model compatibility GPU UUIDs",
                        _strict(
                            "E6 model compatibility row",
                            item,
                            set(E6NextnModelCompatibility.__dataclass_fields__),
                        )["gpu_uuids"],
                    )
                ),
            }
        )
        for item in _array("E6 model compatibility rows", payload_row["models"])
    )
    signed = SignedE6ModelCompatibilityReceipt(
        payload=E6ModelCompatibilityReceipt(**payload_row),  # type: ignore[arg-type]
        payload_sha256=row["payload_sha256"],  # type: ignore[arg-type]
        challenge=_challenge_from_dict(row["challenge"]),
        attestation=_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E6 model compatibility digest differs from content")
    return signed


def signed_e6_confirmation_to_dict(
    value: SignedE6ConfirmationReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE6ConfirmationReceipt:
        raise TypeError("signed E6 confirmation codec requires an exact wrapper")
    return {
        "payload": asdict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e6_confirmation_from_dict(value: object) -> SignedE6ConfirmationReceipt:
    row = _strict(
        "signed E6 confirmation",
        value,
        {
            "payload",
            "payload_sha256",
            "challenge",
            "attestation",
            "signed_receipt_sha256",
        },
    )
    declared = _sha256("signed E6 confirmation", row.pop("signed_receipt_sha256"))
    payload_row = _strict(
        "E6 confirmation payload",
        row.pop("payload"),
        set(E6ConfirmationReceipt.__dataclass_fields__),
    )
    payload_row["models"] = tuple(
        _array("E6 confirmation models", payload_row["models"])
    )
    payload_row["final_block_ids"] = tuple(
        _array("E6 confirmation final blocks", payload_row["final_block_ids"])
    )
    payload_row["primary_contrasts"] = tuple(
        PairedBcaContrast(
            **{
                **_strict(
                    "E6 paired contrast",
                    item,
                    set(PairedBcaContrast.__dataclass_fields__),
                ),
                "block_ids": tuple(
                    _array(
                        "E6 paired contrast blocks",
                        _strict(
                            "E6 paired contrast",
                            item,
                            set(PairedBcaContrast.__dataclass_fields__),
                        )["block_ids"],
                    )
                ),
            }
        )
        for item in _array(
            "E6 confirmation primary contrasts", payload_row["primary_contrasts"]
        )
    )
    payload_row["holm_decisions"] = tuple(
        MultiplicityDecision(
            **_strict(
                "E6 Holm decision",
                item,
                set(MultiplicityDecision.__dataclass_fields__),
            )
        )
        for item in _array(
            "E6 confirmation Holm decisions", payload_row["holm_decisions"]
        )
    )
    signed = SignedE6ConfirmationReceipt(
        payload=E6ConfirmationReceipt(**payload_row),  # type: ignore[arg-type]
        payload_sha256=row["payload_sha256"],  # type: ignore[arg-type]
        challenge=_challenge_from_dict(row["challenge"]),
        attestation=_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E6 confirmation digest differs from content")
    return signed


@dataclass(frozen=True)
class E0ExecutionRebuildShard:
    """Bounded shard of public execution descriptors for one exact phase."""

    schema_version: Literal[1]
    kind: Literal["lightcone_e0_execution_rebuild_shard"]
    phase: Literal[
        "e2_final",
        "e4_screen",
        "e4_local",
        "e3b_pilot",
        "e3b_final",
        "e1a",
        "e5_pilot",
        "e5_final",
        "e6_pilot",
        "e6_final",
        "e0_tuning",
        "e0_pilot",
        "e0_final",
    ]
    materialization_receipt_sha256: str
    stage_source_rebuild_input_sha256: str | None
    descriptors: tuple[FormalServingExecutionRebuildInput, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != E0_EXECUTION_REBUILD_SHARD_KIND:
            raise ValueError("E0 execution rebuild shard schema is unsupported")
        if self.phase not in {
            "e2_final",
            "e4_screen",
            "e4_local",
            "e3b_pilot",
            "e3b_final",
            "e1a",
            "e5_pilot",
            "e5_final",
            "e6_pilot",
            "e6_final",
            "e0_tuning",
            "e0_pilot",
            "e0_final",
        }:
            raise ValueError("E0 execution rebuild shard phase is unsupported")
        _sha256(
            "E0 execution rebuild shard materialization",
            self.materialization_receipt_sha256,
        )
        if self.phase == "e2_final":
            if self.stage_source_rebuild_input_sha256 is not None:
                raise ValueError("E2 execution shard cannot name a stage-source token")
        else:
            _sha256(
                "E0 execution rebuild shard stage source",
                self.stage_source_rebuild_input_sha256,
            )
        expected_stage = {
            "e2_final": "E2",
            "e4_screen": "E4",
            "e4_local": "E4",
            "e3b_pilot": "E3b",
            "e3b_final": "E3b",
            "e1a": "E1a",
            "e5_pilot": "E5",
            "e5_final": "E5",
            "e6_pilot": "E6",
            "e6_final": "E6",
            "e0_tuning": "E0",
            "e0_pilot": "E0",
            "e0_final": "E0",
        }[self.phase]
        if (
            type(self.descriptors) is not tuple
            or not self.descriptors
            or any(
                type(row) is not FormalServingExecutionRebuildInput
                for row in self.descriptors
            )
            or tuple(row.subject.materialized_cell_id for row in self.descriptors)
            != tuple(
                sorted({row.subject.materialized_cell_id for row in self.descriptors})
            )
            or any(
                row.subject.stage != expected_stage
                or row.subject.materialization_receipt_sha256
                != self.materialization_receipt_sha256
                for row in self.descriptors
            )
            or len({row.execution_binding_sha256 for row in self.descriptors})
            != len(self.descriptors)
        ):
            raise ValueError("E0 execution rebuild shard descriptor set is not exact")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "phase": self.phase,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "stage_source_rebuild_input_sha256": (
                self.stage_source_rebuild_input_sha256
            ),
            "descriptors": [row.to_dict() for row in self.descriptors],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E0 execution rebuild shard",
            value,
            {*cls.__dataclass_fields__, "shard_sha256"},
        )
        declared = _sha256("E0 execution rebuild shard", row.pop("shard_sha256"))
        row["descriptors"] = tuple(
            FormalServingExecutionRebuildInput.from_dict(item)
            for item in _array("E0 execution rebuild descriptors", row["descriptors"])
        )
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("E0 execution rebuild shard digest differs from content")
        return shard


def publish_e0_execution_rebuild_shard(
    shard: E0ExecutionRebuildShard,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(shard) is not E0ExecutionRebuildShard:
        raise TypeError("E0 shard publisher requires an exact typed shard")
    shard.__post_init__()
    publish_canonical_json_no_replace(str(output_path), shard.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


def _load_execution_rebuild_shards(
    bindings: tuple[CanonicalJsonProofBinding, ...],
    *,
    expected_phase: Literal[
        "e2_final",
        "e4_screen",
        "e4_local",
        "e3b_pilot",
        "e3b_final",
        "e1a",
        "e5_pilot",
        "e5_final",
        "e6_pilot",
        "e6_final",
        "e0_tuning",
        "e0_pilot",
        "e0_final",
    ],
    materialization: StageMaterializationReceipt,
    stage_source_rebuild_input_sha256: str | None,
) -> tuple[FormalServingExecutionRebuildInput, ...]:
    descriptors: list[FormalServingExecutionRebuildInput] = []
    for binding in bindings:
        before = CanonicalJsonProofBinding.bind(binding.absolute_path)
        if before != binding:
            raise ValueError("E0 execution shard path identity changed")
        shard = E0ExecutionRebuildShard.from_dict(before.reopen())
        after = CanonicalJsonProofBinding.bind(binding.absolute_path)
        if (
            after != before
            or shard.phase != expected_phase
            or shard.materialization_receipt_sha256 != materialization.sha256
            or shard.stage_source_rebuild_input_sha256
            != stage_source_rebuild_input_sha256
        ):
            raise ValueError("E0 execution shard has foreign or changed lineage")
        descriptors.extend(shard.descriptors)
    result = tuple(descriptors)
    cell_ids = tuple(row.subject.materialized_cell_id for row in result)
    if (
        not result
        or cell_ids != tuple(sorted(set(cell_ids)))
        or len({row.execution_binding_sha256 for row in result}) != len(result)
    ):
        raise ValueError("E0 execution descriptor union is not exact")
    return result


@dataclass(frozen=True)
class E5FailureExecutionRebuildShard:
    """Public rebuild descriptors for the exact 264 E5 failure rows."""

    schema_version: Literal[1]
    kind: Literal["lightcone_e5_failure_execution_rebuild_shard"]
    materialization_receipt_sha256: str
    stage_source_rebuild_input_sha256: str
    descriptors: tuple[FormalFailureExecutionRebuildInput, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != E5_FAILURE_EXECUTION_REBUILD_SHARD_KIND
        ):
            raise ValueError("E5 failure rebuild shard schema is unsupported")
        _sha256(
            "E5 failure rebuild shard materialization",
            self.materialization_receipt_sha256,
        )
        _sha256(
            "E5 failure rebuild shard stage source",
            self.stage_source_rebuild_input_sha256,
        )
        if (
            type(self.descriptors) is not tuple
            or not self.descriptors
            or any(
                type(row) is not FormalFailureExecutionRebuildInput
                for row in self.descriptors
            )
            or tuple(row.subject.materialized_cell_id for row in self.descriptors)
            != tuple(
                sorted({row.subject.materialized_cell_id for row in self.descriptors})
            )
            or len(
                {
                    row.expected_failure_execution_binding_sha256
                    for row in self.descriptors
                }
            )
            != len(self.descriptors)
        ):
            raise ValueError("E5 failure rebuild descriptor set is not exact")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "stage_source_rebuild_input_sha256": (
                self.stage_source_rebuild_input_sha256
            ),
            "descriptors": [row.to_dict() for row in self.descriptors],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E5 failure rebuild shard",
            value,
            {*cls.__dataclass_fields__, "shard_sha256"},
        )
        declared = _sha256("E5 failure rebuild shard", row.pop("shard_sha256"))
        row["descriptors"] = tuple(
            FormalFailureExecutionRebuildInput.from_dict(item)
            for item in _array("E5 failure rebuild descriptors", row["descriptors"])
        )
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("E5 failure rebuild shard digest differs from content")
        return shard


def publish_e5_failure_execution_rebuild_shard(
    shard: E5FailureExecutionRebuildShard,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(shard) is not E5FailureExecutionRebuildShard:
        raise TypeError("E5 failure shard publisher requires an exact typed shard")
    shard.__post_init__()
    publish_canonical_json_no_replace(str(output_path), shard.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


def _load_failure_execution_rebuild_shards(
    bindings: tuple[CanonicalJsonProofBinding, ...],
    *,
    materialization: StageMaterializationReceipt,
    stage_source_rebuild_input_sha256: str,
) -> tuple[FormalFailureExecutionRebuildInput, ...]:
    descriptors: list[FormalFailureExecutionRebuildInput] = []
    for binding in bindings:
        before = CanonicalJsonProofBinding.bind(binding.absolute_path)
        if before != binding:
            raise ValueError("E5 failure execution shard path identity changed")
        shard = E5FailureExecutionRebuildShard.from_dict(before.reopen())
        after = CanonicalJsonProofBinding.bind(binding.absolute_path)
        if (
            after != before
            or shard.materialization_receipt_sha256 != materialization.sha256
            or shard.stage_source_rebuild_input_sha256
            != stage_source_rebuild_input_sha256
        ):
            raise ValueError("E5 failure rebuild shard has foreign lineage")
        descriptors.extend(shard.descriptors)
    result = tuple(descriptors)
    cell_ids = tuple(row.subject.materialized_cell_id for row in result)
    if (
        not result
        or cell_ids != tuple(sorted(set(cell_ids)))
        or len({row.expected_failure_execution_binding_sha256 for row in result})
        != len(result)
    ):
        raise ValueError("E5 failure descriptor union is not exact")
    return result


@dataclass(frozen=True)
class FormalStageProofNode:
    """Path-bound public proof inputs for one closed formal DAG phase."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_stage_proof_node"]
    node_id: FormalProofNodeId
    materialization_source: CanonicalJsonProofBinding
    coverage_source: CanonicalJsonProofBinding
    evidence_manifest_source: CanonicalJsonProofBinding | None
    stage_source_rebuild_source: CanonicalJsonProofBinding | None
    execution_rebuild_shards: tuple[CanonicalJsonProofBinding, ...]
    failure_evidence_manifest_source: CanonicalJsonProofBinding | None = None
    failure_execution_rebuild_shards: tuple[CanonicalJsonProofBinding, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != FORMAL_STAGE_PROOF_NODE_KIND:
            raise ValueError("formal stage proof node schema is unsupported")
        if self.node_id not in _FORMAL_PROOF_NODE_ORDER:
            raise ValueError("formal stage proof node ID is unsupported")
        if any(
            type(row) is not CanonicalJsonProofBinding
            for row in (self.materialization_source, self.coverage_source)
        ):
            raise TypeError("formal stage proof node core sources are not path-bound")
        profiler = self.node_id == "e4_profiler"
        e2 = self.node_id == "e2_final"
        if profiler:
            if (
                self.evidence_manifest_source is not None
                or self.stage_source_rebuild_source is not None
                or self.execution_rebuild_shards
            ):
                raise ValueError("E4 profiler node may only carry signed coverage")
        elif (
            type(self.evidence_manifest_source) is not CanonicalJsonProofBinding
            or (
                not e2
                and type(self.stage_source_rebuild_source)
                is not CanonicalJsonProofBinding
            )
            or (e2 and self.stage_source_rebuild_source is not None)
            or type(self.execution_rebuild_shards) is not tuple
            or not self.execution_rebuild_shards
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.execution_rebuild_shards
            )
        ):
            raise ValueError("formal stage proof node public sources are not exact")
        e5_final = self.node_id == "e5_final"
        if (
            e5_final
            and (
                type(self.failure_evidence_manifest_source)
                is not CanonicalJsonProofBinding
                or type(self.failure_execution_rebuild_shards) is not tuple
                or not self.failure_execution_rebuild_shards
                or any(
                    type(row) is not CanonicalJsonProofBinding
                    for row in self.failure_execution_rebuild_shards
                )
            )
        ) or (
            not e5_final
            and (
                self.failure_evidence_manifest_source is not None
                or self.failure_execution_rebuild_shards
            )
        ):
            raise ValueError("formal stage proof node failure sources differ")
        paths = tuple(
            row.absolute_path
            for row in (
                self.materialization_source,
                self.coverage_source,
                *(
                    ()
                    if self.evidence_manifest_source is None
                    else (self.evidence_manifest_source,)
                ),
                *(
                    ()
                    if self.stage_source_rebuild_source is None
                    else (self.stage_source_rebuild_source,)
                ),
                *self.execution_rebuild_shards,
                *(
                    ()
                    if self.failure_evidence_manifest_source is None
                    else (self.failure_evidence_manifest_source,)
                ),
                *self.failure_execution_rebuild_shards,
            )
        )
        if len(paths) != len(set(paths)):
            raise ValueError("formal stage proof node reuses a source path")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "node_id": self.node_id,
            "materialization_source": self.materialization_source.to_dict(),
            "coverage_source": self.coverage_source.to_dict(),
            "evidence_manifest_source": (
                None
                if self.evidence_manifest_source is None
                else self.evidence_manifest_source.to_dict()
            ),
            "stage_source_rebuild_source": (
                None
                if self.stage_source_rebuild_source is None
                else self.stage_source_rebuild_source.to_dict()
            ),
            "execution_rebuild_shards": [
                row.to_dict() for row in self.execution_rebuild_shards
            ],
            "failure_evidence_manifest_source": (
                None
                if self.failure_evidence_manifest_source is None
                else self.failure_evidence_manifest_source.to_dict()
            ),
            "failure_execution_rebuild_shards": [
                row.to_dict() for row in self.failure_execution_rebuild_shards
            ],
        }
        if include_sha256:
            value["node_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage proof node",
            value,
            {*cls.__dataclass_fields__, "node_sha256"},
        )
        declared = _sha256("formal stage proof node", row.pop("node_sha256"))
        for name in ("materialization_source", "coverage_source"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in (
            "evidence_manifest_source",
            "stage_source_rebuild_source",
            "failure_evidence_manifest_source",
        ):
            row[name] = (
                None
                if row[name] is None
                else CanonicalJsonProofBinding.from_dict(row[name])
            )
        for name in (
            "execution_rebuild_shards",
            "failure_execution_rebuild_shards",
        ):
            row[name] = tuple(
                CanonicalJsonProofBinding.from_dict(item)
                for item in _array(f"formal stage proof node {name}", row[name])
            )
        node = cls(**row)  # type: ignore[arg-type]
        if node.sha256 != declared:
            raise ValueError("formal stage proof node digest differs from content")
        return node


@dataclass(frozen=True)
class E6RecursiveSourceDagArtifact:
    """Complete public proof DAG needed to recreate the E6 final source."""

    schema_version: Literal[1]
    kind: Literal["lightcone_e6_recursive_source_dag_artifact"]
    protocol_lock_sha256: str
    registry_verification_receipt_sha256: str
    signed_e3b_confirmation: SignedE3bConfirmationReceipt
    signed_e1a_verification: SignedE1aVerificationReceipt
    signed_e5_confirmation: SignedE5ConfirmationReceipt
    nodes: tuple[FormalStageProofNode, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != E6_RECURSIVE_SOURCE_DAG_ARTIFACT_KIND
        ):
            raise ValueError("E6 recursive source DAG schema is unsupported")
        _sha256("E6 recursive DAG ProtocolLock", self.protocol_lock_sha256)
        _sha256(
            "E6 recursive DAG registry receipt",
            self.registry_verification_receipt_sha256,
        )
        if (
            type(self.signed_e3b_confirmation) is not SignedE3bConfirmationReceipt
            or type(self.signed_e1a_verification) is not SignedE1aVerificationReceipt
            or type(self.signed_e5_confirmation) is not SignedE5ConfirmationReceipt
        ):
            raise TypeError("E6 recursive DAG signed decisions are not exact")
        if (
            type(self.nodes) is not tuple
            or tuple(row.node_id for row in self.nodes) != _FORMAL_PROOF_NODE_ORDER
            or any(type(row) is not FormalStageProofNode for row in self.nodes)
        ):
            raise ValueError("E6 recursive DAG node coverage is not exact")
        paths = tuple(
            source.absolute_path
            for node in self.nodes
            for source in (
                node.materialization_source,
                node.coverage_source,
                *(
                    ()
                    if node.evidence_manifest_source is None
                    else (node.evidence_manifest_source,)
                ),
                *(
                    ()
                    if node.stage_source_rebuild_source is None
                    else (node.stage_source_rebuild_source,)
                ),
                *node.execution_rebuild_shards,
                *(
                    ()
                    if node.failure_evidence_manifest_source is None
                    else (node.failure_evidence_manifest_source,)
                ),
                *node.failure_execution_rebuild_shards,
            )
        )
        if len(paths) != len(set(paths)):
            raise ValueError("E6 recursive DAG reuses a proof path")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "registry_verification_receipt_sha256": (
                self.registry_verification_receipt_sha256
            ),
            "signed_e3b_confirmation": signed_e3b_confirmation_to_dict(
                self.signed_e3b_confirmation
            ),
            "signed_e1a_verification": signed_e1a_verification_to_dict(
                self.signed_e1a_verification
            ),
            "signed_e5_confirmation": signed_e5_confirmation_to_dict(
                self.signed_e5_confirmation
            ),
            "nodes": [row.to_dict() for row in self.nodes],
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E6 recursive source DAG",
            value,
            {*cls.__dataclass_fields__, "artifact_sha256"},
        )
        declared = _sha256("E6 recursive source DAG", row.pop("artifact_sha256"))
        row["signed_e3b_confirmation"] = signed_e3b_confirmation_from_dict(
            row["signed_e3b_confirmation"]
        )
        row["signed_e1a_verification"] = signed_e1a_verification_from_dict(
            row["signed_e1a_verification"]
        )
        row["signed_e5_confirmation"] = signed_e5_confirmation_from_dict(
            row["signed_e5_confirmation"]
        )
        row["nodes"] = tuple(
            FormalStageProofNode.from_dict(item)
            for item in _array("E6 recursive source DAG nodes", row["nodes"])
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("E6 recursive source DAG digest differs from content")
        return artifact


def publish_e6_recursive_source_dag_artifact(
    artifact: E6RecursiveSourceDagArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not E6RecursiveSourceDagArtifact:
        raise TypeError("E6 recursive DAG publisher requires an exact artifact")
    artifact.__post_init__()
    publish_canonical_json_no_replace(str(output_path), artifact.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


@dataclass(frozen=True)
class E0FormalRegistryAuthorityArtifact:
    """Top-level durable E0 reconstruction index (no private verifier seals)."""

    schema_version: Literal[1]
    kind: Literal["lightcone_e0_formal_registry_authority_bundle_artifact"]
    protocol_lock_sha256: str
    prior_registry_verification_receipt_sha256: str
    main_materialization_receipt_sha256: str
    formal_runtime_authority_manifest_source: CanonicalJsonProofBinding
    inventory_source: CanonicalJsonProofBinding
    signed_e6_confirmation: SignedE6ConfirmationReceipt
    signed_e6_model_compatibility: SignedE6ModelCompatibilityReceipt
    e6_compatibility_sources: tuple[E6NextnModelAuthorityInput, ...]
    signed_e0_compatibility: SignedE0CompatibilityReceipt
    onlinespec_source_authority: E0OnlineSpecSourceAuthority
    signed_e0_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...]
    signed_e0_power_prefix: SignedE0PowerPrefixReceipt
    e6_recursive_source_dag_source: CanonicalJsonProofBinding
    e6_materialization_source: CanonicalJsonProofBinding
    e6_coverage_source: CanonicalJsonProofBinding
    e6_evidence_manifest_source: CanonicalJsonProofBinding
    e6_stage_source_rebuild_source: CanonicalJsonProofBinding
    e6_execution_rebuild_shards: tuple[CanonicalJsonProofBinding, ...]
    e0_tuning_materialization_source: CanonicalJsonProofBinding
    e0_tuning_coverage_source: CanonicalJsonProofBinding
    e0_tuning_evidence_manifest_source: CanonicalJsonProofBinding
    e0_tuning_stage_source_rebuild_source: CanonicalJsonProofBinding
    e0_tuning_execution_rebuild_shards: tuple[CanonicalJsonProofBinding, ...]
    e0_pilot_materialization_source: CanonicalJsonProofBinding
    e0_pilot_coverage_source: CanonicalJsonProofBinding
    e0_pilot_evidence_manifest_source: CanonicalJsonProofBinding
    e0_pilot_stage_source_rebuild_source: CanonicalJsonProofBinding
    e0_pilot_execution_rebuild_shards: tuple[CanonicalJsonProofBinding, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != E0_AUTHORITY_BUNDLE_ARTIFACT_KIND:
            raise ValueError("E0 authority bundle artifact schema is unsupported")
        for label, value in (
            ("protocol lock", self.protocol_lock_sha256),
            (
                "prior registry verification receipt",
                self.prior_registry_verification_receipt_sha256,
            ),
            ("main materialization", self.main_materialization_receipt_sha256),
        ):
            _sha256(f"E0 authority artifact {label}", value)
        if type(self.signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
            raise TypeError("E0 authority artifact requires signed E6 confirmation")
        if type(self.signed_e6_model_compatibility) is not (
            SignedE6ModelCompatibilityReceipt
        ):
            raise TypeError("E0 authority artifact requires E6 compatibility")
        if type(self.e6_compatibility_sources) is not tuple or any(
            type(row) is not E6NextnModelAuthorityInput
            for row in self.e6_compatibility_sources
        ):
            raise TypeError("E0 authority artifact E6 source set is not exact")
        if type(self.signed_e0_compatibility) is not SignedE0CompatibilityReceipt:
            raise TypeError("E0 authority artifact requires signed E0 compatibility")
        if type(self.onlinespec_source_authority) is not E0OnlineSpecSourceAuthority:
            raise TypeError("E0 authority artifact requires OnlineSPEC source")
        if (
            type(self.signed_e0_tuning_seals) is not tuple
            or not self.signed_e0_tuning_seals
            or any(
                type(row) is not SignedE0OnlineSpecTuningSeal
                for row in self.signed_e0_tuning_seals
            )
            or tuple(row.payload.decision_id for row in self.signed_e0_tuning_seals)
            != tuple(
                sorted(row.payload.decision_id for row in self.signed_e0_tuning_seals)
            )
        ):
            raise ValueError("E0 authority artifact tuning-seal set is not canonical")
        if type(self.signed_e0_power_prefix) is not SignedE0PowerPrefixReceipt:
            raise TypeError("E0 authority artifact requires signed power prefix")
        scalar_bindings = (
            self.formal_runtime_authority_manifest_source,
            self.inventory_source,
            self.e6_recursive_source_dag_source,
            self.e6_materialization_source,
            self.e6_coverage_source,
            self.e6_evidence_manifest_source,
            self.e6_stage_source_rebuild_source,
            self.e0_tuning_materialization_source,
            self.e0_tuning_coverage_source,
            self.e0_tuning_evidence_manifest_source,
            self.e0_tuning_stage_source_rebuild_source,
            self.e0_pilot_materialization_source,
            self.e0_pilot_coverage_source,
            self.e0_pilot_evidence_manifest_source,
            self.e0_pilot_stage_source_rebuild_source,
        )
        collections = (
            self.e6_execution_rebuild_shards,
            self.e0_tuning_execution_rebuild_shards,
            self.e0_pilot_execution_rebuild_shards,
        )
        if any(type(row) is not CanonicalJsonProofBinding for row in scalar_bindings):
            raise TypeError("E0 authority artifact has a non-path-bound source")
        for rows in collections:
            if (
                type(rows) is not tuple
                or not rows
                or any(type(row) is not CanonicalJsonProofBinding for row in rows)
                or tuple(row.absolute_path for row in rows)
                != tuple(sorted({row.absolute_path for row in rows}))
            ):
                raise ValueError("E0 execution rebuild shards are not canonical")
        all_paths = tuple(
            row.absolute_path
            for row in (
                *scalar_bindings,
                *(item for rows in collections for item in rows),
            )
        )
        if len(all_paths) != len(set(all_paths)):
            raise ValueError("E0 authority artifact reuses a proof reference")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "prior_registry_verification_receipt_sha256": (
                self.prior_registry_verification_receipt_sha256
            ),
            "main_materialization_receipt_sha256": (
                self.main_materialization_receipt_sha256
            ),
            "formal_runtime_authority_manifest_source": (
                self.formal_runtime_authority_manifest_source.to_dict()
            ),
            "inventory_source": self.inventory_source.to_dict(),
            "signed_e6_confirmation": signed_e6_confirmation_to_dict(
                self.signed_e6_confirmation
            ),
            "signed_e6_model_compatibility": (
                signed_e6_model_compatibility_to_dict(
                    self.signed_e6_model_compatibility
                )
            ),
            "e6_compatibility_sources": [
                e6_nextn_model_authority_input_to_dict(row)
                for row in self.e6_compatibility_sources
            ],
            "signed_e0_compatibility": signed_e0_compatibility_to_dict(
                self.signed_e0_compatibility
            ),
            "onlinespec_source_authority": (
                e0_onlinespec_source_authority_to_dict(self.onlinespec_source_authority)
            ),
            "signed_e0_tuning_seals": [
                signed_e0_onlinespec_tuning_seal_to_dict(row)
                for row in self.signed_e0_tuning_seals
            ],
            "signed_e0_power_prefix": signed_e0_power_prefix_to_dict(
                self.signed_e0_power_prefix
            ),
        }
        for name in (
            "e6_recursive_source_dag_source",
            "e6_materialization_source",
            "e6_coverage_source",
            "e6_evidence_manifest_source",
            "e6_stage_source_rebuild_source",
            "e0_tuning_materialization_source",
            "e0_tuning_coverage_source",
            "e0_tuning_evidence_manifest_source",
            "e0_tuning_stage_source_rebuild_source",
            "e0_pilot_materialization_source",
            "e0_pilot_coverage_source",
            "e0_pilot_evidence_manifest_source",
            "e0_pilot_stage_source_rebuild_source",
        ):
            value[name] = getattr(self, name).to_dict()
        for name in (
            "e6_execution_rebuild_shards",
            "e0_tuning_execution_rebuild_shards",
            "e0_pilot_execution_rebuild_shards",
        ):
            value[name] = [row.to_dict() for row in getattr(self, name)]
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E0 authority bundle artifact",
            value,
            {*cls.__dataclass_fields__, "artifact_sha256"},
        )
        declared = _sha256("E0 authority bundle artifact", row.pop("artifact_sha256"))
        row["formal_runtime_authority_manifest_source"] = (
            CanonicalJsonProofBinding.from_dict(
                row["formal_runtime_authority_manifest_source"]
            )
        )
        row["inventory_source"] = CanonicalJsonProofBinding.from_dict(
            row["inventory_source"]
        )
        row["signed_e6_confirmation"] = signed_e6_confirmation_from_dict(
            row["signed_e6_confirmation"]
        )
        row["signed_e6_model_compatibility"] = signed_e6_model_compatibility_from_dict(
            row["signed_e6_model_compatibility"]
        )
        row["e6_compatibility_sources"] = tuple(
            e6_nextn_model_authority_input_from_dict(item)
            for item in _array(
                "E0 artifact E6 compatibility sources",
                row["e6_compatibility_sources"],
            )
        )
        row["signed_e0_compatibility"] = signed_e0_compatibility_from_dict(
            row["signed_e0_compatibility"]
        )
        row["onlinespec_source_authority"] = e0_onlinespec_source_authority_from_dict(
            row["onlinespec_source_authority"]
        )
        row["signed_e0_tuning_seals"] = tuple(
            signed_e0_onlinespec_tuning_seal_from_dict(item)
            for item in _array(
                "E0 artifact signed tuning seals", row["signed_e0_tuning_seals"]
            )
        )
        row["signed_e0_power_prefix"] = signed_e0_power_prefix_from_dict(
            row["signed_e0_power_prefix"]
        )
        for name in (
            "e6_recursive_source_dag_source",
            "e6_materialization_source",
            "e6_coverage_source",
            "e6_evidence_manifest_source",
            "e6_stage_source_rebuild_source",
            "e0_tuning_materialization_source",
            "e0_tuning_coverage_source",
            "e0_tuning_evidence_manifest_source",
            "e0_tuning_stage_source_rebuild_source",
            "e0_pilot_materialization_source",
            "e0_pilot_coverage_source",
            "e0_pilot_evidence_manifest_source",
            "e0_pilot_stage_source_rebuild_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in (
            "e6_execution_rebuild_shards",
            "e0_tuning_execution_rebuild_shards",
            "e0_pilot_execution_rebuild_shards",
        ):
            row[name] = tuple(
                CanonicalJsonProofBinding.from_dict(item)
                for item in _array(f"E0 artifact {name}", row[name])
            )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("E0 authority bundle artifact digest differs from content")
        return artifact


def publish_e0_formal_registry_authority_artifact(
    artifact: E0FormalRegistryAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish one canonical aggregate with atomic no-replace semantics."""

    if type(artifact) is not E0FormalRegistryAuthorityArtifact:
        raise TypeError("E0 authority publisher requires an exact typed artifact")
    artifact.__post_init__()
    path = str(output_path)
    publish_canonical_json_no_replace(path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(path)


def _reopen_typed_source(
    binding: CanonicalJsonProofBinding,
    *,
    label: str,
    decoder,
):
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError(f"{label} path identity changed")
    value = decoder(before.reopen())
    after = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if after != before:
        raise RuntimeError(f"{label} changed while validated")
    return value


def _e0_rebuild_recipe_authorities(
    receipt: FormalRegistryVerificationReceipt,
):
    tts_authorities = receipt.cumulative_tts_calibration_authorities
    signed_tts_seals = receipt.cumulative_signed_tts_calibration_seals
    final_e2 = tuple(
        row.payload.final_recipe
        for row in receipt.cumulative_signed_e2_staged_selections
        if row.payload.round_index == 3
    )
    if (
        len(tts_authorities) != 1
        or len(signed_tts_seals) != 1
        or len(final_e2) != 1
        or final_e2[0] is None
    ):
        raise ValueError("E0 rebuild lacks one frozen TTS and final E2 recipe")
    grid = default_e2_recipe_grid_authority()
    lock = receipt.signed_protocol_lock.payload
    if (
        tts_authorities[0].sha256 != lock.tts_calibration_authority_sha256
        or grid.sha256 != lock.e2_recipe_grid_authority_sha256
    ):
        raise ValueError("E0 rebuild recipe authorities differ from ProtocolLock")
    return tts_authorities[0], signed_tts_seals[0], grid, final_e2[0]


def _rebuild_serving_bindings(
    descriptors: tuple[FormalServingExecutionRebuildInput, ...],
    *,
    materialization: StageMaterializationReceipt,
    stage_source: VerifiedFormalStageMaterializationSource | None,
    artifact: E0FormalRegistryAuthorityArtifact,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    now_ns: int,
) -> tuple[VerifiedFormalServingExecutionBinding, ...]:
    runtime_manifest = _reopen_typed_source(
        artifact.formal_runtime_authority_manifest_source,
        label="E0 formal runtime authority",
        decoder=formal_runtime_authority_manifest_from_dict,
    )
    inventory = _reopen_typed_source(
        artifact.inventory_source,
        label="E0 GPU inventory",
        decoder=GpuInventory.from_dict,
    )
    if inventory.sha256 != registry_verification_receipt.inventory_sha256:
        raise ValueError("E0 rebuild inventory differs from durable registry")
    lock = registry_verification_receipt.signed_protocol_lock.payload
    if runtime_manifest.sha256 != lock.formal_runtime_authority_manifest_sha256:
        raise ValueError("E0 rebuild runtime authority differs from ProtocolLock")
    tts, signed_tts, grid, lightcone_recipe = _e0_rebuild_recipe_authorities(
        registry_verification_receipt
    )
    return tuple(
        rebuild_formal_serving_execution_binding(
            descriptor,
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            tts_authority=tts,
            signed_tts_seal=signed_tts,
            e2_recipe_grid_authority=grid,
            lightcone_recipe=lightcone_recipe,
            stage_source=stage_source,
            now_ns=now_ns,
            registry_verification_receipt=registry_verification_receipt,
        )
        for descriptor in descriptors
    )


def _validate_manifest_bindings(
    manifest: FormalDownstreamEvidenceManifest,
    bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
) -> None:
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    binding_by_cell = {row.subject.materialized_cell_id: row for row in bindings}
    if (
        len(evidence_by_cell) != len(manifest.cells)
        or len(binding_by_cell) != len(bindings)
        or set(evidence_by_cell) != set(binding_by_cell)
    ):
        raise ValueError("E0 evidence and rebuilt binding coverage differ")
    for cell_id, evidence in evidence_by_cell.items():
        binding = binding_by_cell[cell_id]
        if (
            evidence.execution_binding_sha256 != binding.sha256
            or evidence.execution_identity != binding.subject.execution_identity
        ):
            raise ValueError("E0 downstream evidence names a foreign binding")


def _validate_manifest_binding_subset(
    manifest: FormalDownstreamEvidenceManifest,
    bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
) -> None:
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    binding_by_cell = {row.subject.materialized_cell_id: row for row in bindings}
    if len(evidence_by_cell) != len(manifest.cells) or not set(evidence_by_cell) <= set(
        binding_by_cell
    ):
        raise ValueError("formal evidence is not a unique subset of rebuilt bindings")
    for cell_id, evidence in evidence_by_cell.items():
        binding = binding_by_cell[cell_id]
        if (
            evidence.execution_binding_sha256 != binding.sha256
            or evidence.execution_identity != binding.subject.execution_identity
        ):
            raise ValueError("formal evidence subset names a foreign binding")


def _e2_source_recipes_from_materialization(
    materialization: StageMaterializationReceipt,
) -> tuple[E2CandidateRecipe, ...]:
    grid = default_e2_recipe_grid_authority()
    recipes: dict[str, E2CandidateRecipe] = {}
    for cell in materialization.cells:
        if cell.method_role != "LightCone-candidate":
            continue
        dimensions = dict(cell.dimensions)
        rank_value = dimensions.get("rank")
        alpha_value = dimensions.get("alpha_over_rank")
        geometry = E1Geometry(
            scope=str(dimensions.get("scope")),
            parameterization=str(dimensions.get("parameterization")),  # type: ignore[arg-type]
            rank=None if rank_value == "none" else int(rank_value),  # type: ignore[arg-type]
            alpha_over_rank=(
                None if alpha_value == "none" else float(alpha_value)  # type: ignore[arg-type]
            ),
        )
        recipe = E2CandidateRecipe(
            geometry=geometry,
            optimizer=str(dimensions.get("optimizer")),
            schedule=str(dimensions.get("schedule")),
            learning_rate=float(dimensions.get("learning_rate")),  # type: ignore[arg-type]
            optimizer_recipe_authority_sha256=(grid.optimizer_recipe_authority.sha256),
        )
        if (
            recipe.sha256 != cell.recipe_sha256
            or dimensions.get("geometry_sha256") != geometry.sha256
            or dimensions.get("optimizer_recipe_authority_sha256")
            != grid.optimizer_recipe_authority.sha256
        ):
            raise ValueError("E2 durable cell differs from its typed source recipe")
        recipes[recipe.sha256] = recipe
    result = tuple(recipes[digest] for digest in sorted(recipes))
    if not result or len(result) != len(
        [
            cell
            for cell in materialization.cells
            if cell.method_role == "LightCone-candidate"
        ]
    ):
        raise ValueError("E2 durable source recipe universe is not exact")
    return result


@dataclass(frozen=True)
class _RebuiltStageProofNode:
    node: FormalStageProofNode
    materialization: StageMaterializationReceipt
    coverage: StageCoverageReceipt
    manifest: object | None
    stage_source: VerifiedFormalStageMaterializationSource | None
    serving_descriptors: tuple[FormalServingExecutionRebuildInput, ...]
    serving_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]
    failure_manifest: E5FailureEvidenceManifest | None = None
    failure_bindings: tuple[VerifiedFormalFailureExecutionBinding, ...] = ()


def _load_e6_recursive_source_dag(
    binding: CanonicalJsonProofBinding,
    *,
    artifact: E0FormalRegistryAuthorityArtifact,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
) -> E6RecursiveSourceDagArtifact:
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError("E6 recursive DAG path identity changed")
    dag = E6RecursiveSourceDagArtifact.from_dict(before.reopen())
    after = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if (
        after != before
        or dag.protocol_lock_sha256 != artifact.protocol_lock_sha256
        or dag.registry_verification_receipt_sha256
        != registry_verification_receipt.sha256
    ):
        raise ValueError("E6 recursive DAG has foreign or changed lineage")
    nested_paths = {
        source.absolute_path
        for node in dag.nodes
        for source in (
            node.materialization_source,
            node.coverage_source,
            *(
                ()
                if node.evidence_manifest_source is None
                else (node.evidence_manifest_source,)
            ),
            *(
                ()
                if node.stage_source_rebuild_source is None
                else (node.stage_source_rebuild_source,)
            ),
            *node.execution_rebuild_shards,
            *(
                ()
                if node.failure_evidence_manifest_source is None
                else (node.failure_evidence_manifest_source,)
            ),
            *node.failure_execution_rebuild_shards,
        )
    }
    if before.absolute_path in nested_paths:
        raise ValueError("E6 recursive DAG aliases its own index path")
    top_level_paths = {
        source.absolute_path
        for source in (
            artifact.formal_runtime_authority_manifest_source,
            artifact.inventory_source,
            artifact.e6_materialization_source,
            artifact.e6_coverage_source,
            artifact.e6_evidence_manifest_source,
            artifact.e6_stage_source_rebuild_source,
            *artifact.e6_execution_rebuild_shards,
            artifact.e0_tuning_materialization_source,
            artifact.e0_tuning_coverage_source,
            artifact.e0_tuning_evidence_manifest_source,
            artifact.e0_tuning_stage_source_rebuild_source,
            *artifact.e0_tuning_execution_rebuild_shards,
            artifact.e0_pilot_materialization_source,
            artifact.e0_pilot_coverage_source,
            artifact.e0_pilot_evidence_manifest_source,
            artifact.e0_pilot_stage_source_rebuild_source,
            *artifact.e0_pilot_execution_rebuild_shards,
        )
    }
    if nested_paths & top_level_paths:
        raise ValueError("E6 recursive DAG aliases an E0 aggregate source path")
    return dag


def _load_formal_stage_proof_node(
    node: FormalStageProofNode,
    *,
    protocol_lock_sha256: str,
) -> tuple[StageMaterializationReceipt, StageCoverageReceipt, object | None]:
    materialization = _reopen_typed_source(
        node.materialization_source,
        label=f"{node.node_id} materialization",
        decoder=stage_materialization_receipt_from_dict,
    )
    coverage = _reopen_typed_source(
        node.coverage_source,
        label=f"{node.node_id} coverage",
        decoder=stage_coverage_receipt_from_dict,
    )
    if (
        materialization.stage != _NODE_STAGE[node.node_id]
        or materialization.protocol_lock_sha256 != protocol_lock_sha256
    ):
        raise ValueError(f"{node.node_id} materialization has foreign lineage")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError(f"{node.node_id} coverage is not all COMPLETE")
    if node.node_id == "e4_profiler":
        return materialization, coverage, None
    assert node.evidence_manifest_source is not None
    if node.node_id == "e2_final":
        manifest = _reopen_typed_source(
            node.evidence_manifest_source,
            label="E2 final evidence",
            decoder=e2_staged_evidence_manifest_from_dict,
        )
        if type(manifest) is not E2StagedRoundEvidenceManifest:
            raise TypeError("E2 final evidence manifest is not exact")
    elif node.node_id in {"e4_screen", "e4_local"}:
        manifest = _reopen_typed_source(
            node.evidence_manifest_source,
            label=f"{node.node_id} evidence",
            decoder=e4_staged_evidence_manifest_from_dict,
        )
        if type(manifest) is not E4StagedEvidenceManifest:
            raise TypeError("E4 evidence manifest is not exact")
    else:
        manifest = _reopen_typed_source(
            node.evidence_manifest_source,
            label=f"{node.node_id} evidence",
            decoder=formal_downstream_evidence_manifest_from_dict,
        )
        if type(manifest) is not FormalDownstreamEvidenceManifest:
            raise TypeError("downstream evidence manifest is not exact")
    if (
        manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
    ):
        raise ValueError(f"{node.node_id} evidence lineage differs")
    return materialization, coverage, manifest


def _one_registry_source(
    rows: tuple[object, ...],
    *,
    label: str,
    predicate,
):
    selected = tuple(row for row in rows if predicate(row))
    if len(selected) != 1:
        raise ValueError(f"durable registry lacks one exact {label}")
    return selected[0]


def _stage_source_descriptor_for_node(
    node: FormalStageProofNode,
) -> FormalStageSourceRebuildInput:
    if node.stage_source_rebuild_source is None:
        raise ValueError(f"{node.node_id} lacks its stage-source descriptor")
    return _load_stage_source_descriptor(node.stage_source_rebuild_source)


def _serving_bindings_for_node(
    node: FormalStageProofNode,
    *,
    materialization: StageMaterializationReceipt,
    stage_source_descriptor: FormalStageSourceRebuildInput | None,
    stage_source: VerifiedFormalStageMaterializationSource | None,
    artifact: E0FormalRegistryAuthorityArtifact,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    now_ns: int,
) -> tuple[
    tuple[FormalServingExecutionRebuildInput, ...],
    tuple[VerifiedFormalServingExecutionBinding, ...],
]:
    descriptors = _load_execution_rebuild_shards(
        node.execution_rebuild_shards,
        expected_phase=node.node_id,  # type: ignore[arg-type]
        materialization=materialization,
        stage_source_rebuild_input_sha256=(
            None if stage_source_descriptor is None else stage_source_descriptor.sha256
        ),
    )
    bindings = _rebuild_serving_bindings(
        descriptors,
        materialization=materialization,
        stage_source=stage_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    expected_cells = {cell.cell_id for cell in materialization.cells}
    observed_cells = {row.subject.materialized_cell_id for row in bindings}
    if len(observed_cells) != len(bindings) or observed_cells != expected_cells:
        raise ValueError(f"{node.node_id} serving rebuild coverage is not exact")
    return descriptors, bindings


def _rebuild_e6_confirmation_proof_bundle(
    artifact: E0FormalRegistryAuthorityArtifact,
    *,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    now_ns: int,
) -> E6ConfirmationProofBundle:
    """Recursively deep-open E2→E6 without deserializing a private seal."""

    dag = _load_e6_recursive_source_dag(
        artifact.e6_recursive_source_dag_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
    )
    nodes = {row.node_id: row for row in dag.nodes}
    lock = registry_verification_receipt.signed_protocol_lock.payload
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    tts, signed_tts, _grid, _lightcone = _e0_rebuild_recipe_authorities(
        registry_verification_receipt
    )

    e2_node = nodes["e2_final"]
    e2_mat, e2_cov, e2_manifest = _load_formal_stage_proof_node(
        e2_node, protocol_lock_sha256=lock.sha256
    )
    if type(e2_manifest) is not E2StagedRoundEvidenceManifest:
        raise TypeError("E2 final recursive evidence is not exact")
    e2_recipes = _e2_source_recipes_from_materialization(e2_mat)
    e2_selection = _one_registry_source(
        registry_verification_receipt.cumulative_signed_e2_staged_selections,
        label="E2 round-three selection",
        predicate=lambda row: (
            row.payload.round_index == 3
            and row.payload.materialization_receipt_sha256 == e2_mat.sha256
        ),
    )
    _e2_descriptors, e2_bindings = _serving_bindings_for_node(
        e2_node,
        materialization=e2_mat,
        stage_source_descriptor=None,
        stage_source=None,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(e2_manifest, e2_bindings)  # type: ignore[arg-type]

    e4_screen_node = nodes["e4_screen"]
    e4_screen_mat, e4_screen_cov, e4_screen_manifest = _load_formal_stage_proof_node(
        e4_screen_node, protocol_lock_sha256=lock.sha256
    )
    if type(e4_screen_manifest) is not E4StagedEvidenceManifest:
        raise TypeError("E4 screen recursive evidence is not exact")
    e4_screen_descriptor = _stage_source_descriptor_for_node(e4_screen_node)
    e4_screen_source = rebuild_formal_stage_materialization_source(
        e4_screen_descriptor,
        materialization=e4_screen_mat,
        source_inputs=E4ScreenStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e2_final_selection=e2_selection,  # type: ignore[arg-type]
            e2_materialization=e2_mat,
            e2_coverage=e2_cov,
            e2_source_recipes=e2_recipes,
            e2_evidence_manifest=e2_manifest,
            e2_execution_bindings=e2_bindings,
        ),
        now_ns=now_ns,
    )
    _screen_descriptors, e4_screen_bindings = _serving_bindings_for_node(
        e4_screen_node,
        materialization=e4_screen_mat,
        stage_source_descriptor=e4_screen_descriptor,
        stage_source=e4_screen_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(  # type: ignore[arg-type]
        e4_screen_manifest, e4_screen_bindings
    )
    e4_screen_selection = _one_registry_source(
        registry_verification_receipt.cumulative_signed_e4_stage_selections,
        label="E4 screen selection",
        predicate=lambda row: (
            row.payload.phase == "screen"
            and row.payload.materialization_receipt_sha256 == e4_screen_mat.sha256
        ),
    )

    e4_local_node = nodes["e4_local"]
    e4_local_mat, e4_local_cov, e4_local_manifest = _load_formal_stage_proof_node(
        e4_local_node, protocol_lock_sha256=lock.sha256
    )
    if type(e4_local_manifest) is not E4StagedEvidenceManifest:
        raise TypeError("E4 local recursive evidence is not exact")
    e4_local_descriptor = _stage_source_descriptor_for_node(e4_local_node)
    e4_local_source = rebuild_formal_stage_materialization_source(
        e4_local_descriptor,
        materialization=e4_local_mat,
        source_inputs=E4LocalStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e4_screen_selection=e4_screen_selection,  # type: ignore[arg-type]
            screen_materialization=e4_screen_mat,
            screen_coverage=e4_screen_cov,
            screen_evidence_manifest=e4_screen_manifest,
            screen_execution_bindings=e4_screen_bindings,
        ),
        now_ns=now_ns,
    )
    _local_descriptors, e4_local_bindings = _serving_bindings_for_node(
        e4_local_node,
        materialization=e4_local_mat,
        stage_source_descriptor=e4_local_descriptor,
        stage_source=e4_local_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(  # type: ignore[arg-type]
        e4_local_manifest, e4_local_bindings
    )
    e4_final_selection = _one_registry_source(
        registry_verification_receipt.cumulative_signed_e4_stage_selections,
        label="E4 local selection",
        predicate=lambda row: (
            row.payload.phase == "local"
            and row.payload.materialization_receipt_sha256 == e4_local_mat.sha256
        ),
    )

    e4_profiler_node = nodes["e4_profiler"]
    e4_profiler_mat, e4_profiler_cov, profiler_manifest = _load_formal_stage_proof_node(
        e4_profiler_node, protocol_lock_sha256=lock.sha256
    )
    if profiler_manifest is not None:
        raise ValueError("E4 profiler node unexpectedly carries headline evidence")

    e3b_pilot_node = nodes["e3b_pilot"]
    e3b_pilot_mat, e3b_pilot_cov, e3b_pilot_manifest = _load_formal_stage_proof_node(
        e3b_pilot_node, protocol_lock_sha256=lock.sha256
    )
    if type(e3b_pilot_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E3b pilot recursive evidence is not exact")
    e3b_pilot_descriptor = _stage_source_descriptor_for_node(e3b_pilot_node)
    e3b_pilot_source = rebuild_formal_stage_materialization_source(
        e3b_pilot_descriptor,
        materialization=e3b_pilot_mat,
        source_inputs=E3bPilotStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e4_final_selection=e4_final_selection,  # type: ignore[arg-type]
            local_materialization=e4_local_mat,
            local_coverage=e4_local_cov,
            local_evidence_manifest=e4_local_manifest,
            local_execution_bindings=e4_local_bindings,
            profiler_materialization=e4_profiler_mat,
            profiler_coverage=e4_profiler_cov,
            tts_calibration_authority=tts,
            signed_tts_calibration_seal=signed_tts,
        ),
        now_ns=now_ns,
    )
    _e3b_pilot_descriptors, e3b_pilot_bindings = _serving_bindings_for_node(
        e3b_pilot_node,
        materialization=e3b_pilot_mat,
        stage_source_descriptor=e3b_pilot_descriptor,
        stage_source=e3b_pilot_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(e3b_pilot_manifest, e3b_pilot_bindings)
    e3b_power = _one_registry_source(
        registry_verification_receipt.cumulative_signed_e3b_power_prefixes,
        label="E3b power prefix",
        predicate=lambda row: (
            row.payload.pilot_materialization_receipt_sha256 == e3b_pilot_mat.sha256
            and row.payload.pilot_coverage_receipt_sha256 == e3b_pilot_cov.sha256
        ),
    )

    e3b_final_node = nodes["e3b_final"]
    e3b_final_mat, e3b_final_cov, e3b_final_manifest = _load_formal_stage_proof_node(
        e3b_final_node, protocol_lock_sha256=lock.sha256
    )
    if type(e3b_final_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E3b final recursive evidence is not exact")
    e3b_final_descriptor = _stage_source_descriptor_for_node(e3b_final_node)
    e3b_final_source = rebuild_formal_stage_materialization_source(
        e3b_final_descriptor,
        materialization=e3b_final_mat,
        source_inputs=E3bFinalStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_power_prefix=e3b_power,  # type: ignore[arg-type]
            pilot_materialization=e3b_pilot_mat,
            pilot_coverage=e3b_pilot_cov,
            pilot_evidence_manifest=e3b_pilot_manifest,
            pilot_execution_bindings=e3b_pilot_bindings,
        ),
        now_ns=now_ns,
    )
    _e3b_final_descriptors, e3b_final_bindings = _serving_bindings_for_node(
        e3b_final_node,
        materialization=e3b_final_mat,
        stage_source_descriptor=e3b_final_descriptor,
        stage_source=e3b_final_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(e3b_final_manifest, e3b_final_bindings)
    dag.signed_e3b_confirmation.verify(
        protocol_lock=lock,
        materialization=e3b_final_mat,
        coverage=e3b_final_cov,
        manifest=e3b_final_manifest,
        execution_bindings=e3b_final_bindings,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )

    e1a_node = nodes["e1a"]
    e1a_mat, e1a_cov, e1a_manifest = _load_formal_stage_proof_node(
        e1a_node, protocol_lock_sha256=lock.sha256
    )
    if type(e1a_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E1a recursive evidence is not exact")
    e1a_descriptor = _stage_source_descriptor_for_node(e1a_node)
    e1a_source = rebuild_formal_stage_materialization_source(
        e1a_descriptor,
        materialization=e1a_mat,
        source_inputs=E1aStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e3b_confirmation=dag.signed_e3b_confirmation,
            e3b_materialization=e3b_final_mat,
            e3b_coverage=e3b_final_cov,
            e3b_evidence_manifest=e3b_final_manifest,
            e3b_execution_bindings=e3b_final_bindings,
        ),
        now_ns=now_ns,
    )
    _e1a_descriptors, e1a_bindings = _serving_bindings_for_node(
        e1a_node,
        materialization=e1a_mat,
        stage_source_descriptor=e1a_descriptor,
        stage_source=e1a_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(e1a_manifest, e1a_bindings)
    dag.signed_e1a_verification.verify(
        protocol_lock=lock,
        materialization=e1a_mat,
        coverage=e1a_cov,
        manifest=e1a_manifest,
        execution_bindings=e1a_bindings,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )

    e5_pilot_node = nodes["e5_pilot"]
    e5_pilot_mat, e5_pilot_cov, e5_pilot_manifest = _load_formal_stage_proof_node(
        e5_pilot_node, protocol_lock_sha256=lock.sha256
    )
    if type(e5_pilot_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E5 pilot recursive evidence is not exact")
    e5_pilot_descriptor = _stage_source_descriptor_for_node(e5_pilot_node)
    runtime_manifest = _reopen_typed_source(
        artifact.formal_runtime_authority_manifest_source,
        label="E5 formal runtime authority",
        decoder=formal_runtime_authority_manifest_from_dict,
    )
    e5_pilot_source = rebuild_formal_stage_materialization_source(
        e5_pilot_descriptor,
        materialization=e5_pilot_mat,
        source_inputs=E5PilotStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e1a_verification=dag.signed_e1a_verification,
            e1a_materialization=e1a_mat,
            e1a_coverage=e1a_cov,
            e1a_evidence_manifest=e1a_manifest,
            e1a_execution_bindings=e1a_bindings,
            formal_runtime_authority_manifest=runtime_manifest,
        ),
        now_ns=now_ns,
    )
    _e5_pilot_descriptors, e5_pilot_bindings = _serving_bindings_for_node(
        e5_pilot_node,
        materialization=e5_pilot_mat,
        stage_source_descriptor=e5_pilot_descriptor,
        stage_source=e5_pilot_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(e5_pilot_manifest, e5_pilot_bindings)
    e5_power = _one_registry_source(
        registry_verification_receipt.cumulative_signed_e5_power_and_anchor_prefixes,
        label="E5 power and anchor prefix",
        predicate=lambda row: (
            row.payload.pilot_materialization_receipt_sha256 == e5_pilot_mat.sha256
            and row.payload.pilot_coverage_receipt_sha256 == e5_pilot_cov.sha256
        ),
    )

    e5_final_node = nodes["e5_final"]
    e5_final_mat, e5_final_cov, e5_final_manifest = _load_formal_stage_proof_node(
        e5_final_node, protocol_lock_sha256=lock.sha256
    )
    if type(e5_final_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E5 final recursive evidence is not exact")
    e5_final_descriptor = _stage_source_descriptor_for_node(e5_final_node)
    failure_authority = default_e5_failure_diagnostic_authority()
    e5_final_source = rebuild_formal_stage_materialization_source(
        e5_final_descriptor,
        materialization=e5_final_mat,
        source_inputs=E5FinalStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_power_and_anchor_prefix=e5_power,  # type: ignore[arg-type]
            pilot_materialization=e5_pilot_mat,
            pilot_coverage=e5_pilot_cov,
            pilot_evidence_manifest=e5_pilot_manifest,
            pilot_execution_bindings=e5_pilot_bindings,
            formal_runtime_authority_manifest=runtime_manifest,
            failure_diagnostic_authority=failure_authority,
        ),
        now_ns=now_ns,
    )
    e5_serving_descriptors, e5_serving_bindings = _serving_bindings_for_node(
        e5_final_node,
        materialization=e5_final_mat,
        stage_source_descriptor=e5_final_descriptor,
        stage_source=e5_final_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_binding_subset(e5_final_manifest, e5_serving_bindings)
    assert e5_final_node.failure_evidence_manifest_source is not None
    failure_manifest = _reopen_typed_source(
        e5_final_node.failure_evidence_manifest_source,
        label="E5 final failure evidence",
        decoder=e5_failure_evidence_manifest_from_dict,
    )
    if (
        type(failure_manifest) is not E5FailureEvidenceManifest
        or failure_manifest.materialization_receipt_sha256 != e5_final_mat.sha256
        or failure_manifest.coverage_receipt_sha256 != e5_final_cov.sha256
    ):
        raise ValueError("E5 failure evidence lineage differs")
    failure_descriptors = _load_failure_execution_rebuild_shards(
        e5_final_node.failure_execution_rebuild_shards,
        materialization=e5_final_mat,
        stage_source_rebuild_input_sha256=e5_final_descriptor.sha256,
    )
    serving_descriptor_by_sha = {row.sha256: row for row in e5_serving_descriptors}
    serving_binding_by_sha = {row.sha256: row for row in e5_serving_bindings}
    failure_bindings = []
    for failure_descriptor in failure_descriptors:
        serving_descriptor = serving_descriptor_by_sha.get(
            failure_descriptor.serving_execution_rebuild_input_sha256
        )
        if serving_descriptor is None:
            raise ValueError("E5 failure descriptor lacks its serving descriptor")
        serving_binding = serving_binding_by_sha.get(
            serving_descriptor.execution_binding_sha256
        )
        if serving_binding is None:
            raise ValueError("E5 failure descriptor lacks its rebuilt serving token")
        failure_bindings.append(
            rebuild_formal_failure_execution_binding(
                failure_descriptor,
                serving_execution_rebuild_input=serving_descriptor,
                serving_execution=serving_binding,
                protocol_lock=lock,
                formal_runtime_authority_manifest=runtime_manifest,
                materialization=e5_final_mat,
            )
        )
    rebuilt_failure_bindings = tuple(failure_bindings)
    failure_evidence_by_cell = {
        row.materialized_cell_id: row for row in failure_manifest.cells
    }
    failure_binding_by_cell = {
        row.subject.materialized_cell_id: row for row in rebuilt_failure_bindings
    }
    if (
        len(rebuilt_failure_bindings) != 264
        or set(failure_evidence_by_cell) != set(failure_binding_by_cell)
        or any(
            failure_evidence_by_cell[cell_id].failure_execution_binding_sha256
            != binding.sha256
            for cell_id, binding in failure_binding_by_cell.items()
        )
    ):
        raise ValueError("E5 failure proof/binding coverage differs")
    dag.signed_e5_confirmation.verify(
        protocol_lock=lock,
        materialization=e5_final_mat,
        coverage=e5_final_cov,
        headline_manifest=e5_final_manifest,
        headline_execution_bindings=tuple(
            row
            for row in e5_serving_bindings
            if row.subject.materialized_cell_id
            in {item.materialized_cell_id for item in e5_final_manifest.cells}
        ),
        failure_manifest=failure_manifest,
        failure_execution_bindings=rebuilt_failure_bindings,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )

    e6_pilot_node = nodes["e6_pilot"]
    e6_pilot_mat, e6_pilot_cov, e6_pilot_manifest = _load_formal_stage_proof_node(
        e6_pilot_node, protocol_lock_sha256=lock.sha256
    )
    if type(e6_pilot_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E6 pilot recursive evidence is not exact")
    e6_pilot_descriptor = _stage_source_descriptor_for_node(e6_pilot_node)
    e6_pilot_source = rebuild_formal_stage_materialization_source(
        e6_pilot_descriptor,
        materialization=e6_pilot_mat,
        source_inputs=E6PilotStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e5_confirmation=dag.signed_e5_confirmation,
            e5_materialization=e5_final_mat,
            e5_coverage=e5_final_cov,
            e5_headline_evidence_manifest=e5_final_manifest,
            e5_headline_execution_bindings=tuple(
                row
                for row in e5_serving_bindings
                if row.subject.materialized_cell_id
                in {item.materialized_cell_id for item in e5_final_manifest.cells}
            ),
            e5_failure_evidence_manifest=failure_manifest,
            e5_failure_execution_bindings=rebuilt_failure_bindings,
            signed_model_compatibility=artifact.signed_e6_model_compatibility,
            compatibility_sources=artifact.e6_compatibility_sources,
        ),
        now_ns=now_ns,
    )
    _e6_pilot_descriptors, e6_pilot_bindings = _serving_bindings_for_node(
        e6_pilot_node,
        materialization=e6_pilot_mat,
        stage_source_descriptor=e6_pilot_descriptor,
        stage_source=e6_pilot_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(e6_pilot_manifest, e6_pilot_bindings)
    e6_power = _one_registry_source(
        registry_verification_receipt.cumulative_signed_e6_power_prefixes,
        label="E6 power prefix",
        predicate=lambda row: (
            row.payload.pilot_materialization_receipt_sha256 == e6_pilot_mat.sha256
            and row.payload.pilot_coverage_receipt_sha256 == e6_pilot_cov.sha256
        ),
    )

    e6_mat = _reopen_typed_source(
        artifact.e6_materialization_source,
        label="E6 final materialization",
        decoder=stage_materialization_receipt_from_dict,
    )
    e6_cov = _reopen_typed_source(
        artifact.e6_coverage_source,
        label="E6 final coverage",
        decoder=stage_coverage_receipt_from_dict,
    )
    e6_manifest = _reopen_typed_source(
        artifact.e6_evidence_manifest_source,
        label="E6 final evidence",
        decoder=formal_downstream_evidence_manifest_from_dict,
    )
    if (
        type(e6_mat) is not StageMaterializationReceipt
        or type(e6_cov) is not StageCoverageReceipt
        or type(e6_manifest) is not FormalDownstreamEvidenceManifest
        or e6_mat.stage != "E6"
        or e6_mat.protocol_lock_sha256 != lock.sha256
        or e6_manifest.materialization_receipt_sha256 != e6_mat.sha256
        or e6_manifest.coverage_receipt_sha256 != e6_cov.sha256
    ):
        raise ValueError("E6 final durable proof lineage differs")
    e6_cov.validate_against(e6_mat)
    if any(row.status != "COMPLETE" for row in e6_cov.dispositions):
        raise ValueError("E6 final coverage is not all COMPLETE")
    e6_descriptor = _load_stage_source_descriptor(
        artifact.e6_stage_source_rebuild_source
    )
    e6_source = rebuild_formal_stage_materialization_source(
        e6_descriptor,
        materialization=e6_mat,
        source_inputs=E6FinalStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e5_confirmation=dag.signed_e5_confirmation,
            signed_model_compatibility=artifact.signed_e6_model_compatibility,
            compatibility_sources=artifact.e6_compatibility_sources,
            signed_power_prefix=e6_power,  # type: ignore[arg-type]
            pilot_materialization=e6_pilot_mat,
            pilot_coverage=e6_pilot_cov,
            pilot_evidence_manifest=e6_pilot_manifest,
            pilot_execution_bindings=e6_pilot_bindings,
        ),
        now_ns=now_ns,
    )
    e6_descriptors = _load_execution_rebuild_shards(
        artifact.e6_execution_rebuild_shards,
        expected_phase="e6_final",
        materialization=e6_mat,
        stage_source_rebuild_input_sha256=e6_descriptor.sha256,
    )
    e6_bindings = _rebuild_serving_bindings(
        e6_descriptors,
        materialization=e6_mat,
        stage_source=e6_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(e6_manifest, e6_bindings)
    bundle = E6ConfirmationProofBundle(
        signed_model_compatibility=artifact.signed_e6_model_compatibility,
        compatibility_sources=artifact.e6_compatibility_sources,
        materialization=e6_mat,
        coverage=e6_cov,
        manifest=e6_manifest,
        execution_bindings=e6_bindings,
    )
    bundle.verify(
        artifact.signed_e6_confirmation,
        protocol_lock=lock,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    return bundle


def _load_stage_source_descriptor(
    binding: CanonicalJsonProofBinding,
) -> FormalStageSourceRebuildInput:
    return _reopen_typed_source(
        binding,
        label="formal stage-source rebuild descriptor",
        decoder=FormalStageSourceRebuildInput.from_dict,
    )


def _load_e0_proof_set_sources(
    artifact: E0FormalRegistryAuthorityArtifact,
    *,
    prefix: Literal["e0_tuning", "e0_pilot"],
):
    materialization = _reopen_typed_source(
        getattr(artifact, f"{prefix}_materialization_source"),
        label=f"{prefix} materialization",
        decoder=stage_materialization_receipt_from_dict,
    )
    coverage = _reopen_typed_source(
        getattr(artifact, f"{prefix}_coverage_source"),
        label=f"{prefix} coverage",
        decoder=stage_coverage_receipt_from_dict,
    )
    manifest = _reopen_typed_source(
        getattr(artifact, f"{prefix}_evidence_manifest_source"),
        label=f"{prefix} evidence manifest",
        decoder=formal_downstream_evidence_manifest_from_dict,
    )
    if (
        type(materialization) is not StageMaterializationReceipt
        or type(coverage) is not StageCoverageReceipt
        or type(manifest) is not FormalDownstreamEvidenceManifest
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
    ):
        raise ValueError(f"{prefix} durable proof lineage differs")
    coverage.validate_against(materialization)
    return materialization, coverage, manifest


def _rebuild_e0_tuning_and_pilot_proofs(
    artifact: E0FormalRegistryAuthorityArtifact,
    *,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    e6_confirmation_proof_bundle: E6ConfirmationProofBundle,
    now_ns: int,
) -> tuple[E0OnlineSpecTuningProofSet, E0OnlineSpecTuningProofSet]:
    tuning_materialization, tuning_coverage, tuning_manifest = (
        _load_e0_proof_set_sources(artifact, prefix="e0_tuning")
    )
    tuning_source_descriptor = _load_stage_source_descriptor(
        artifact.e0_tuning_stage_source_rebuild_source
    )
    tuning_source = rebuild_formal_stage_materialization_source(
        tuning_source_descriptor,
        materialization=tuning_materialization,
        source_inputs=E0TuningStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e6_confirmation=artifact.signed_e6_confirmation,
            e6_confirmation_proof_bundle=e6_confirmation_proof_bundle,
            signed_compatibility_receipt=artifact.signed_e0_compatibility,
            onlinespec_source_authority=artifact.onlinespec_source_authority,
        ),
        now_ns=now_ns,
    )
    tuning_descriptors = _load_execution_rebuild_shards(
        artifact.e0_tuning_execution_rebuild_shards,
        expected_phase="e0_tuning",
        materialization=tuning_materialization,
        stage_source_rebuild_input_sha256=tuning_source_descriptor.sha256,
    )
    tuning_bindings = _rebuild_serving_bindings(
        tuning_descriptors,
        materialization=tuning_materialization,
        stage_source=tuning_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(tuning_manifest, tuning_bindings)
    tuning = E0OnlineSpecTuningProofSet(
        e6_confirmation_proof_bundle=e6_confirmation_proof_bundle,
        materialization=tuning_materialization,
        coverage=tuning_coverage,
        manifest=tuning_manifest,
        execution_bindings=tuning_bindings,
    )

    pilot_materialization, pilot_coverage, pilot_manifest = _load_e0_proof_set_sources(
        artifact, prefix="e0_pilot"
    )
    pilot_source_descriptor = _load_stage_source_descriptor(
        artifact.e0_pilot_stage_source_rebuild_source
    )
    pilot_source = rebuild_formal_stage_materialization_source(
        pilot_source_descriptor,
        materialization=pilot_materialization,
        source_inputs=E0PilotStageSourceRebuildInputs(
            registry_verification_receipt=registry_verification_receipt,
            signed_e6_confirmation=artifact.signed_e6_confirmation,
            e6_confirmation_proof_bundle=e6_confirmation_proof_bundle,
            signed_compatibility_receipt=artifact.signed_e0_compatibility,
            signed_onlinespec_tuning_seals=artifact.signed_e0_tuning_seals,
            onlinespec_source_authority=artifact.onlinespec_source_authority,
            tuning_proof_set=tuning,
        ),
        now_ns=now_ns,
    )
    pilot_descriptors = _load_execution_rebuild_shards(
        artifact.e0_pilot_execution_rebuild_shards,
        expected_phase="e0_pilot",
        materialization=pilot_materialization,
        stage_source_rebuild_input_sha256=pilot_source_descriptor.sha256,
    )
    pilot_bindings = _rebuild_serving_bindings(
        pilot_descriptors,
        materialization=pilot_materialization,
        stage_source=pilot_source,
        artifact=artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(pilot_manifest, pilot_bindings)
    pilot = E0OnlineSpecTuningProofSet(
        e6_confirmation_proof_bundle=e6_confirmation_proof_bundle,
        materialization=pilot_materialization,
        coverage=pilot_coverage,
        manifest=pilot_manifest,
        execution_bindings=pilot_bindings,
    )
    return tuning, pilot


def _assert_expected_e0_artifact_lineage(
    artifact: E0FormalRegistryAuthorityArtifact,
    *,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    materialization: StageMaterializationReceipt,
) -> None:
    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E0 artifact requires exact prior registry receipt")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E0 artifact requires exact main materialization")
    protocol_lock = registry_verification_receipt.signed_protocol_lock.payload
    if (
        artifact.protocol_lock_sha256 != protocol_lock.sha256
        or artifact.prior_registry_verification_receipt_sha256
        != registry_verification_receipt.sha256
        or artifact.main_materialization_receipt_sha256 != materialization.sha256
        or materialization.stage != "E0"
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("E0 authority artifact has foreign or replayed DAG lineage")


def load_e0_formal_registry_authority_artifact_index(
    artifact_path: str | Path,
    *,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    materialization: StageMaterializationReceipt,
) -> E0FormalRegistryAuthorityArtifact:
    """Strictly reopen the aggregate index before deep proof reconstruction."""

    before = CanonicalJsonProofBinding.bind(str(artifact_path))
    artifact = E0FormalRegistryAuthorityArtifact.from_dict(before.reopen())
    after = CanonicalJsonProofBinding.bind(before.absolute_path)
    if before != after:
        raise RuntimeError("E0 authority artifact changed while read")
    _assert_expected_e0_artifact_lineage(
        artifact,
        registry_verification_receipt=registry_verification_receipt,
        materialization=materialization,
    )
    return artifact


def load_e0_formal_registry_authority_bundle(
    artifact_path: str | Path,
    *,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    materialization: StageMaterializationReceipt,
    now_ns: int,
) -> E0FormalRegistryAuthorityBundle:
    """Deep-rebuild the E0 bundle from its durable artifact.

    The recursive public stage-source interface is intentionally required here;
    this function never accepts a digest-only or private-token fallback.
    """

    artifact = load_e0_formal_registry_authority_artifact_index(
        artifact_path,
        registry_verification_receipt=registry_verification_receipt,
        materialization=materialization,
    )
    e6_confirmation_proof_bundle = _rebuild_e6_confirmation_proof_bundle(
        artifact,
        registry_verification_receipt=registry_verification_receipt,
        now_ns=now_ns,
    )
    tuning_proof_set, pilot_proof_set = _rebuild_e0_tuning_and_pilot_proofs(
        artifact,
        registry_verification_receipt=registry_verification_receipt,
        e6_confirmation_proof_bundle=e6_confirmation_proof_bundle,
        now_ns=now_ns,
    )
    bundle = E0FormalRegistryAuthorityBundle(
        signed_e6_confirmation=artifact.signed_e6_confirmation,
        e6_confirmation_proof_bundle=e6_confirmation_proof_bundle,
        signed_compatibility=artifact.signed_e0_compatibility,
        source_authority=artifact.onlinespec_source_authority,
        tuning_proof_set=tuning_proof_set,
        signed_tuning_seals=artifact.signed_e0_tuning_seals,
        pilot_proof_set=pilot_proof_set,
        signed_power_prefix=artifact.signed_e0_power_prefix,
    )
    bundle.verify_against(
        registry_verification_receipt=registry_verification_receipt,
        materialization=materialization,
        now_ns=now_ns,
    )
    return bundle


@dataclass(frozen=True)
class E0FinalResultRebuildArtifact:
    """Path-bound public inputs for reducing the terminal E0 completion."""

    schema_version: Literal[1]
    kind: Literal["lightcone_e0_final_result_rebuild_artifact"]
    protocol_lock_sha256: str
    prior_registry_verification_receipt_sha256: str
    current_registry_verification_receipt_sha256: str
    main_materialization_receipt_sha256: str
    prior_registry_source: CanonicalJsonProofBinding
    current_registry_source: CanonicalJsonProofBinding
    signed_main_materialization_source: CanonicalJsonProofBinding
    e0_authority_artifact_source: CanonicalJsonProofBinding
    evidence_manifest_source: CanonicalJsonProofBinding
    stage_source_rebuild_source: CanonicalJsonProofBinding
    execution_rebuild_shards: tuple[CanonicalJsonProofBinding, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != E0_FINAL_RESULT_REBUILD_ARTIFACT_KIND
        ):
            raise ValueError("E0 final result rebuild artifact is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("prior registry", self.prior_registry_verification_receipt_sha256),
            ("current registry", self.current_registry_verification_receipt_sha256),
            ("main materialization", self.main_materialization_receipt_sha256),
        ):
            _sha256(f"E0 final result {label}", digest)
        bindings = (
            self.prior_registry_source,
            self.current_registry_source,
            self.signed_main_materialization_source,
            self.e0_authority_artifact_source,
            self.evidence_manifest_source,
            self.stage_source_rebuild_source,
            *self.execution_rebuild_shards,
        )
        if (
            not self.execution_rebuild_shards
            or any(type(row) is not CanonicalJsonProofBinding for row in bindings)
            or len({row.absolute_path for row in bindings}) != len(bindings)
            or self.execution_rebuild_shards
            != tuple(
                sorted(
                    self.execution_rebuild_shards,
                    key=lambda row: row.absolute_path,
                )
            )
        ):
            raise ValueError("E0 final result sources are not exact unique bindings")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "prior_registry_verification_receipt_sha256": (
                self.prior_registry_verification_receipt_sha256
            ),
            "current_registry_verification_receipt_sha256": (
                self.current_registry_verification_receipt_sha256
            ),
            "main_materialization_receipt_sha256": (
                self.main_materialization_receipt_sha256
            ),
            "prior_registry_source": self.prior_registry_source.to_dict(),
            "current_registry_source": self.current_registry_source.to_dict(),
            "signed_main_materialization_source": (
                self.signed_main_materialization_source.to_dict()
            ),
            "e0_authority_artifact_source": (
                self.e0_authority_artifact_source.to_dict()
            ),
            "evidence_manifest_source": self.evidence_manifest_source.to_dict(),
            "stage_source_rebuild_source": (self.stage_source_rebuild_source.to_dict()),
            "execution_rebuild_shards": [
                row.to_dict() for row in self.execution_rebuild_shards
            ],
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E0 final result rebuild artifact",
            value,
            {*cls.__dataclass_fields__, "artifact_sha256"},
        )
        declared = _sha256("E0 final result artifact", row.pop("artifact_sha256"))
        for field in (
            "prior_registry_source",
            "current_registry_source",
            "signed_main_materialization_source",
            "e0_authority_artifact_source",
            "evidence_manifest_source",
            "stage_source_rebuild_source",
        ):
            row[field] = CanonicalJsonProofBinding.from_dict(row[field])
        row["execution_rebuild_shards"] = tuple(
            CanonicalJsonProofBinding.from_dict(item)
            for item in _array(
                "E0 final execution rebuild shards",
                row["execution_rebuild_shards"],
            )
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("E0 final result artifact digest differs from content")
        return artifact


def publish_e0_final_result_rebuild_artifact(
    artifact: E0FinalResultRebuildArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not E0FinalResultRebuildArtifact:
        raise TypeError("E0 final result publisher requires an exact artifact")
    artifact.__post_init__()
    publish_canonical_json_no_replace(str(output_path), artifact.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


@dataclass(frozen=True)
class E0FinalCellCompletion:
    materialized_cell_id: str
    execution_binding_sha256: str
    terminal_receipt_sha256: str
    native_result_proof_semantic_sha256: str
    stage_itl_proof_semantic_sha256: str

    def __post_init__(self) -> None:
        for label, digest in (
            ("cell", self.materialized_cell_id),
            ("execution binding", self.execution_binding_sha256),
            ("terminal", self.terminal_receipt_sha256),
            ("native result", self.native_result_proof_semantic_sha256),
            ("stage ITL", self.stage_itl_proof_semantic_sha256),
        ):
            _sha256(f"E0 final completion {label}", digest)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E0FinalAnalysisCell:
    """Public metric projection emitted only after full E0 proof validation."""

    materialized_cell_id: str
    compatibility_decision_id: str
    model: str
    backend: str
    task: str
    method_role: str
    block: int
    load: str
    terminal_receipt_sha256: str
    request_identity_sha256: str
    completed_output_tokens: int
    slo_goodput_numerator_tokens: int
    scored_window_ns: int
    slo_accounting_sha256: str
    slo_policy_sha256: str

    def __post_init__(self) -> None:
        for label, digest in (
            ("cell", self.materialized_cell_id),
            ("compatibility decision", self.compatibility_decision_id),
            ("terminal", self.terminal_receipt_sha256),
            ("request identity", self.request_identity_sha256),
            ("SLO accounting", self.slo_accounting_sha256),
            ("SLO policy", self.slo_policy_sha256),
        ):
            _sha256(f"E0 final analysis {label}", digest)
        if (
            type(self.model) is not str
            or not self.model
            or type(self.backend) is not str
            or not self.backend
            or type(self.task) is not str
            or not self.task
            or self.method_role
            not in {
                "Target-only",
                "Static",
                "TTS",
                "L0-naive",
                "LightCone",
                "OnlineSPEC-OGD",
                "OnlineSPEC-OPT",
                "OnlineSPEC-ENS",
                "OnlineSPEC-Optimistic-OGD",
                "OnlineSPEC-Hedge",
            }
            or type(self.block) is not int
            or self.block < 4
            or self.load not in {"concurrency_one", "common_slo_load"}
            or type(self.completed_output_tokens) is not int
            or self.completed_output_tokens < 1
            or type(self.slo_goodput_numerator_tokens) is not int
            or not 0
            <= self.slo_goodput_numerator_tokens
            <= self.completed_output_tokens
            or type(self.scored_window_ns) is not int
            or self.scored_window_ns < 1
            or self.slo_policy_sha256 != E0_FINAL_SLO_GOODPUT_POLICY_SHA256
        ):
            raise ValueError("E0 final analysis cell fields are not exact")

    @property
    def slo_goodput_tps(self) -> float:
        return self.slo_goodput_numerator_tokens / (
            self.scored_window_ns / 1_000_000_000.0
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E0FinalAnalysisProjection:
    """Deep-rebuilt E0 inputs safe for a proof-derived statistical reducer."""

    schema_version: Literal[1]
    completion_receipt: E0FinalCompletionReceipt
    compatibility_decisions: tuple[E0CompatibilityDecision, ...]
    cells: tuple[E0FinalAnalysisCell, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("E0 final analysis projection schema is unsupported")
        if type(self.completion_receipt) is not E0FinalCompletionReceipt:
            raise TypeError("E0 final analysis requires an exact completion receipt")
        if (
            type(self.compatibility_decisions) is not tuple
            or len(self.compatibility_decisions) != 108
            or any(
                type(row) is not E0CompatibilityDecision
                for row in self.compatibility_decisions
            )
            or tuple(row.decision_id for row in self.compatibility_decisions)
            != tuple(sorted({row.decision_id for row in self.compatibility_decisions}))
        ):
            raise ValueError("E0 final analysis compatibility coverage is not exact")
        valid_ids = {
            row.decision_id
            for row in self.compatibility_decisions
            if row.disposition == "VALID"
        }
        cell_ids = tuple(row.materialized_cell_id for row in self.cells)
        expected_blocks = (
            set(self.completion_receipt.selected_final_prefix) if valid_ids else set()
        )
        if (
            type(self.cells) is not tuple
            or any(type(row) is not E0FinalAnalysisCell for row in self.cells)
            or cell_ids != tuple(sorted(set(cell_ids)))
            or {row.compatibility_decision_id for row in self.cells} != valid_ids
            or len(self.cells)
            != 16 * len(valid_ids) * len(self.completion_receipt.selected_final_prefix)
            or {row.block for row in self.cells} != expected_blocks
            or self.completion_receipt.valid_compatibility_count != len(valid_ids)
        ):
            raise ValueError("E0 final analysis cell universe is not exact")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E0FinalCompletionReceipt:
    schema_version: Literal[1]
    protocol_lock_sha256: str
    registry_sha256: str
    prior_registry_verification_receipt_sha256: str
    current_registry_verification_receipt_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    stage_source_binding_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    rebuild_artifact_sha256: str
    selected_final_prefix: tuple[int, ...]
    valid_compatibility_count: int
    cells: tuple[E0FinalCellCompletion, ...]
    protocol_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.protocol_sha256 != (
            E0_FINAL_COMPLETION_PROTOCOL_SHA256
        ):
            raise ValueError("E0 final completion receipt is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("prior registry receipt", self.prior_registry_verification_receipt_sha256),
            (
                "current registry receipt",
                self.current_registry_verification_receipt_sha256,
            ),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("stage source", self.stage_source_binding_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("rebuild artifact", self.rebuild_artifact_sha256),
        ):
            _sha256(f"E0 final completion {label}", digest)
        all_na = self.valid_compatibility_count == 0
        if (
            type(self.valid_compatibility_count) is not int
            or self.valid_compatibility_count < 0
            or (all_na and (self.selected_final_prefix != () or self.cells != ()))
            or (
                not all_na
                and (
                    not 12 <= len(self.selected_final_prefix) <= 20
                    or self.selected_final_prefix
                    != tuple(range(4, 4 + len(self.selected_final_prefix)))
                )
            )
            or len(self.cells)
            != 16 * self.valid_compatibility_count * len(self.selected_final_prefix)
            or any(type(row) is not E0FinalCellCompletion for row in self.cells)
            or tuple(row.materialized_cell_id for row in self.cells)
            != tuple(sorted({row.materialized_cell_id for row in self.cells}))
        ):
            raise ValueError("E0 final completion cardinality/prefix is not exact")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE0FinalCompletionReceipt:
    payload: E0FinalCompletionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        rebuild_artifact_path: str | Path,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E0FinalCompletionReceipt:
        if type(self.payload) is not E0FinalCompletionReceipt:
            raise TypeError("signed E0 final completion payload is not exact")
        expected = reduce_e0_final_completion_from_artifact(
            rebuild_artifact_path,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E0 final completion differs from proof reducer")
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


def _registry_contains_ancestor(
    current: FormalRegistryVerificationReceipt,
    expected: FormalRegistryVerificationReceipt,
) -> bool:
    cursor: FormalRegistryVerificationReceipt | None = current
    while cursor is not None:
        if cursor.sha256 == expected.sha256:
            return cursor == expected
        cursor = cursor.prior_receipt
    return False


def _load_e0_final_result_artifact(
    artifact_path: str | Path,
) -> tuple[CanonicalJsonProofBinding, E0FinalResultRebuildArtifact]:
    before = CanonicalJsonProofBinding.bind(str(artifact_path))
    artifact = E0FinalResultRebuildArtifact.from_dict(before.reopen())
    if CanonicalJsonProofBinding.bind(before.absolute_path) != before:
        raise RuntimeError("E0 final result artifact changed while read")
    return before, artifact


def _e0_prompt_bucket(input_token_count: int) -> str:
    if type(input_token_count) is not int or input_token_count < 1:
        raise ValueError("E0 SLO prompt length must be positive")
    if input_token_count <= 2_048:
        return "short"
    if input_token_count <= 8_192:
        return "medium"
    return "long"


def _e0_slo_goodput_inputs(validated) -> tuple[int, int, str]:
    """Derive SLO-qualified tokens/window only from terminal and native ITL."""

    timing_by_id = {row.request_id: row for row in validated.timing.requests}
    metrics_by_id = {row.request_id: row for row in validated.metrics}
    if (
        not timing_by_id
        or set(timing_by_id) != set(metrics_by_id)
        or len(timing_by_id) != len(validated.timing.requests)
    ):
        raise ValueError("E0 SLO timing/terminal request coverage differs")
    slo_rows = []
    qualified_ids = set()
    for request_id in sorted(metrics_by_id):
        metric = metrics_by_id[request_id]
        timing = timing_by_id[request_id]
        ttft_ms = (
            timing.token_observed_ns[0] - timing.request_started_ns
        ) / 1_000_000.0
        p99_itl_ms = float(metric.p99_itl_ns) / 1_000_000.0
        bucket = _e0_prompt_bucket(len(metric.input_token_ids))
        row = SloRequest(
            request_id=request_id,
            prompt_bucket=bucket,
            eligible=True,
            completed=True,
            error=False,
            ttft_ms=ttft_ms,
            within_request_p99_itl_ms=p99_itl_ms,
        )
        slo_rows.append(row)
        if (
            ttft_ms <= TTFT_LIMIT_MS[bucket]
            and p99_itl_ms <= WITHIN_REQUEST_P99_ITL_LIMIT_MS
        ):
            qualified_ids.add(request_id)
    slo = account_slo(tuple(slo_rows))
    qualified_tokens = sum(
        metric.output_tokens
        for metric in validated.metrics
        if metric.request_id in qualified_ids
    )
    scored_window_ns = max(
        row.request_terminal_ns for row in validated.timing.requests
    ) - min(row.request_started_ns for row in validated.timing.requests)
    if qualified_tokens < 0 or scored_window_ns < 1:
        raise ValueError("E0 final cell SLO-goodput accounting is invalid")
    return qualified_tokens, scored_window_ns, content_sha256(slo)


def rebuild_e0_final_analysis_projection_from_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> E0FinalAnalysisProjection:
    """Deep-replay E2→E0 and expose only validated final analysis inputs."""

    artifact_binding, artifact = _load_e0_final_result_artifact(artifact_path)
    prior = _reopen_typed_source(
        artifact.prior_registry_source,
        label="E0 final prior registry",
        decoder=formal_registry_verification_receipt_from_dict,
    )
    current = _reopen_typed_source(
        artifact.current_registry_source,
        label="E0 final current registry",
        decoder=formal_registry_verification_receipt_from_dict,
    )
    if (
        type(prior) is not FormalRegistryVerificationReceipt
        or type(current) is not FormalRegistryVerificationReceipt
    ):
        raise TypeError("E0 final result registries are not exact")
    prior.revalidate(current_ns=now_ns)
    current.revalidate(current_ns=now_ns)
    if (
        prior.sha256 != artifact.prior_registry_verification_receipt_sha256
        or current.sha256 != artifact.current_registry_verification_receipt_sha256
        or not _registry_contains_ancestor(current, prior)
    ):
        raise ValueError("E0 final result registry lineage is not append-only")
    signed_materialization = _reopen_typed_source(
        artifact.signed_main_materialization_source,
        label="E0 final signed materialization",
        decoder=signed_stage_materialization_from_dict,
    )
    if type(signed_materialization) is not SignedStageMaterializationReceipt:
        raise TypeError("E0 final materialization wrapper is not exact")
    materialization = signed_materialization.payload
    signed_materializations = current.cumulative_signed_materializations
    if (
        materialization.stage != "E0"
        or materialization.sha256 != artifact.main_materialization_receipt_sha256
        or signed_materialization not in signed_materializations
    ):
        raise ValueError("E0 final materialization is absent from current registry")
    signed_coverages = tuple(
        row
        for row in current.cumulative_signed_coverage
        if row.payload.materialization_receipt_sha256 == materialization.sha256
    )
    if len(signed_coverages) != 1:
        raise ValueError("E0 final current registry lacks one exact signed coverage")
    coverage = signed_coverages[0].payload
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E0 final result requires all-COMPLETE coverage")
    authority_before = CanonicalJsonProofBinding.bind(
        artifact.e0_authority_artifact_source.absolute_path
    )
    if authority_before != artifact.e0_authority_artifact_source:
        raise ValueError("E0 final authority artifact path identity changed")
    bundle = load_e0_formal_registry_authority_bundle(
        authority_before.absolute_path,
        registry_verification_receipt=prior,
        materialization=materialization,
        now_ns=now_ns,
    )
    stage_descriptor = _reopen_typed_source(
        artifact.stage_source_rebuild_source,
        label="E0 final stage-source descriptor",
        decoder=FormalStageSourceRebuildInput.from_dict,
    )
    stage_source = rebuild_formal_stage_materialization_source(
        stage_descriptor,
        materialization=materialization,
        source_inputs=E0FinalStageSourceRebuildInputs(
            registry_verification_receipt=prior,
            authority_bundle=bundle,
        ),
        now_ns=now_ns,
    )
    manifest = _reopen_typed_source(
        artifact.evidence_manifest_source,
        label="E0 final evidence manifest",
        decoder=formal_downstream_evidence_manifest_from_dict,
    )
    if (
        type(manifest) is not FormalDownstreamEvidenceManifest
        or manifest.stage != "E0"
        or manifest.protocol_lock_sha256 != artifact.protocol_lock_sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.source_authority_sha256 != stage_source.sha256
    ):
        raise ValueError("E0 final evidence differs from exact lineage")
    descriptors = _load_execution_rebuild_shards(
        artifact.execution_rebuild_shards,
        expected_phase="e0_final",
        materialization=materialization,
        stage_source_rebuild_input_sha256=stage_descriptor.sha256,
    )
    e0_artifact = load_e0_formal_registry_authority_artifact_index(
        authority_before.absolute_path,
        registry_verification_receipt=prior,
        materialization=materialization,
    )
    bindings = _rebuild_serving_bindings(
        descriptors,
        materialization=materialization,
        stage_source=stage_source,
        artifact=e0_artifact,
        registry_verification_receipt=prior,
        now_ns=now_ns,
    )
    _validate_manifest_bindings(manifest, bindings)
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell = {row.subject.materialized_cell_id: row for row in bindings}
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    expected_ids = {cell.cell_id for cell in materialization.cells}
    if (
        set(evidence_by_cell) != expected_ids
        or set(bindings_by_cell) != expected_ids
        or set(terminal_by_cell) != expected_ids
    ):
        raise ValueError("E0 final evidence/binding/coverage universe is not exact")
    rows: list[E0FinalCellCompletion] = []
    analysis_rows: list[E0FinalAnalysisCell] = []
    for cell in materialization.cells:
        evidence = evidence_by_cell[cell.cell_id]
        validated = _validated_cell(
            cell=cell,
            evidence=evidence,  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],
            protocol_lock=current.signed_protocol_lock.payload,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E0",
        )
        if validated.safety_reasons or (
            cell.method_role not in {"Target-only", "Static"}
            and validated.published_updates < 1
        ):
            raise ValueError("E0 final result contains unsafe or inactive evidence")
        rows.append(
            E0FinalCellCompletion(
                materialized_cell_id=cell.cell_id,
                execution_binding_sha256=bindings_by_cell[cell.cell_id].sha256,
                terminal_receipt_sha256=terminal_by_cell[cell.cell_id],
                native_result_proof_semantic_sha256=(
                    evidence.native_result_proof_semantic_sha256
                ),
                stage_itl_proof_semantic_sha256=(
                    evidence.stage_itl_proof_semantic_sha256
                ),
            )
        )
        dimensions = dict(cell.dimensions)
        decision_id = dimensions.get("compatibility_decision_id")
        block = dimensions.get("block")
        load = dimensions.get("load")
        if (
            type(decision_id) is not str
            or type(block) is not int
            or type(load) is not str
        ):
            raise ValueError("E0 final analysis dimensions are not exact")
        (
            slo_goodput_numerator_tokens,
            scored_window_ns,
            slo_accounting_sha256,
        ) = _e0_slo_goodput_inputs(validated)
        analysis_rows.append(
            E0FinalAnalysisCell(
                materialized_cell_id=cell.cell_id,
                compatibility_decision_id=decision_id,
                model=cell.model,
                backend=cell.backend,
                task=cell.task,
                method_role=cell.method_role,
                block=block,
                load=load,
                terminal_receipt_sha256=terminal_by_cell[cell.cell_id],
                request_identity_sha256=content_sha256(
                    _request_identity(validated.metrics)
                ),
                completed_output_tokens=sum(
                    metric.output_tokens for metric in validated.metrics
                ),
                slo_goodput_numerator_tokens=slo_goodput_numerator_tokens,
                scored_window_ns=scored_window_ns,
                slo_accounting_sha256=slo_accounting_sha256,
                slo_policy_sha256=E0_FINAL_SLO_GOODPUT_POLICY_SHA256,
            )
        )
    block_values = {
        dict(cell.dimensions).get("block") for cell in materialization.cells
    }
    decision_values = {
        dict(cell.dimensions).get("compatibility_decision_id")
        for cell in materialization.cells
    }
    prefix = bundle.signed_power_prefix.payload.selected_final_prefix
    if (
        block_values != set(prefix)
        or None in decision_values
        or len(materialization.cells) != 16 * len(decision_values) * len(prefix)
    ):
        raise ValueError("E0 final result block/compatibility coverage is not exact")
    receipt = E0FinalCompletionReceipt(
        schema_version=1,
        protocol_lock_sha256=current.signed_protocol_lock.payload.sha256,
        registry_sha256=current.signed_protocol_lock.payload.registry_sha256,
        prior_registry_verification_receipt_sha256=prior.sha256,
        current_registry_verification_receipt_sha256=current.sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        stage_source_binding_sha256=stage_source.sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        rebuild_artifact_sha256=artifact_binding.semantic_sha256,
        selected_final_prefix=prefix,
        valid_compatibility_count=len(decision_values),
        cells=tuple(sorted(rows, key=lambda row: row.materialized_cell_id)),
        protocol_sha256=E0_FINAL_COMPLETION_PROTOCOL_SHA256,
    )
    receipt.__post_init__()
    projection = E0FinalAnalysisProjection(
        schema_version=1,
        completion_receipt=receipt,
        compatibility_decisions=bundle.signed_compatibility.payload.decisions,
        cells=tuple(sorted(analysis_rows, key=lambda row: row.materialized_cell_id)),
    )
    projection.__post_init__()
    return projection


def reduce_e0_final_completion_from_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> E0FinalCompletionReceipt:
    """Deep-replay E2→E0 and reduce one completion-only terminal receipt."""

    return rebuild_e0_final_analysis_projection_from_artifact(
        artifact_path,
        now_ns=now_ns,
    ).completion_receipt


def e0_final_completion_receipt_to_dict(
    value: E0FinalCompletionReceipt,
) -> dict[str, object]:
    if type(value) is not E0FinalCompletionReceipt:
        raise TypeError("E0 final completion codec requires an exact receipt")
    row = asdict(value)
    row["selected_final_prefix"] = list(value.selected_final_prefix)
    row["cells"] = [asdict(item) for item in value.cells]
    return {**row, "receipt_sha256": value.sha256}


def e0_final_completion_receipt_from_dict(value: object) -> E0FinalCompletionReceipt:
    row = _strict(
        "E0 final completion receipt",
        value,
        {*E0FinalCompletionReceipt.__dataclass_fields__, "receipt_sha256"},
    )
    declared = _sha256("E0 final completion receipt", row.pop("receipt_sha256"))
    row["selected_final_prefix"] = tuple(
        _array("E0 final completion prefix", row["selected_final_prefix"])
    )
    row["cells"] = tuple(
        E0FinalCellCompletion(
            **_strict(
                "E0 final cell completion",
                item,
                set(E0FinalCellCompletion.__dataclass_fields__),
            )
        )
        for item in _array("E0 final completion cells", row["cells"])
    )
    receipt = E0FinalCompletionReceipt(**row)  # type: ignore[arg-type]
    if receipt.sha256 != declared:
        raise ValueError("E0 final completion receipt digest differs")
    return receipt


def signed_e0_final_completion_to_dict(
    value: SignedE0FinalCompletionReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE0FinalCompletionReceipt:
        raise TypeError("signed E0 final completion codec requires exact wrapper")
    return {
        "payload": e0_final_completion_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e0_final_completion_from_dict(
    value: object,
) -> SignedE0FinalCompletionReceipt:
    row = _strict(
        "signed E0 final completion",
        value,
        {
            "payload",
            "payload_sha256",
            "challenge",
            "attestation",
            "signed_receipt_sha256",
        },
    )
    declared = _sha256("signed E0 final completion", row.pop("signed_receipt_sha256"))
    signed = SignedE0FinalCompletionReceipt(
        payload=e0_final_completion_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=_challenge_from_dict(row["challenge"]),
        attestation=_attestation_from_dict(row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E0 final completion digest differs")
    return signed


__all__ = [
    "E0_AUTHORITY_BUNDLE_ARTIFACT_KIND",
    "E0_EXECUTION_REBUILD_SHARD_KIND",
    "E0_FINAL_COMPLETION_PROTOCOL_SHA256",
    "E0_FINAL_RESULT_REBUILD_ARTIFACT_KIND",
    "E0_FINAL_SLO_GOODPUT_POLICY_SHA256",
    "E0ExecutionRebuildShard",
    "E0FinalAnalysisCell",
    "E0FinalAnalysisProjection",
    "E0FinalCellCompletion",
    "E0FinalCompletionReceipt",
    "E0FinalResultRebuildArtifact",
    "E0FormalRegistryAuthorityArtifact",
    "E5FailureExecutionRebuildShard",
    "E6RecursiveSourceDagArtifact",
    "FormalStageProofNode",
    "SignedE0FinalCompletionReceipt",
    "e0_final_completion_receipt_from_dict",
    "e0_final_completion_receipt_to_dict",
    "e5_failure_evidence_manifest_from_dict",
    "e5_failure_evidence_manifest_to_dict",
    "e6_nextn_model_authority_input_from_dict",
    "e6_nextn_model_authority_input_to_dict",
    "load_e0_formal_registry_authority_artifact_index",
    "load_e0_formal_registry_authority_bundle",
    "publish_e0_execution_rebuild_shard",
    "publish_e0_final_result_rebuild_artifact",
    "publish_e0_formal_registry_authority_artifact",
    "publish_e5_failure_execution_rebuild_shard",
    "publish_e6_recursive_source_dag_artifact",
    "reduce_e0_final_completion_from_artifact",
    "signed_e0_final_completion_from_dict",
    "signed_e0_final_completion_to_dict",
    "signed_e1a_verification_from_dict",
    "signed_e1a_verification_to_dict",
    "signed_e3b_confirmation_from_dict",
    "signed_e3b_confirmation_to_dict",
    "signed_e5_confirmation_from_dict",
    "signed_e5_confirmation_to_dict",
    "signed_e6_confirmation_from_dict",
    "signed_e6_confirmation_to_dict",
    "signed_e6_model_compatibility_from_dict",
    "signed_e6_model_compatibility_to_dict",
]
