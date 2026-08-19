from __future__ import annotations

import json

import pytest
from test_e6_stage_authority import _compatibility, _source
from test_formal_registry_power_sources import _signature_parts

from lightcone_spec.experiments.downstream_stage_authority import (
    E1A_VERIFICATION_PROTOCOL_SHA256,
    E3B_CONFIRMATION_PROTOCOL_SHA256,
    E5_CONFIRMATION_PROTOCOL_SHA256,
    E1aConfigurationEvaluation,
    E1aVerificationReceipt,
    E3bConfirmationReceipt,
    E5ConfirmationReceipt,
    E5P99AnchorCompletion,
    FormalFamilyConfirmationResult,
    SignedE1aVerificationReceipt,
    SignedE3bConfirmationReceipt,
    SignedE5ConfirmationReceipt,
)
from lightcone_spec.experiments.e0_authority_artifact import (
    E6RecursiveSourceDagArtifact,
    FormalStageProofNode,
    e6_nextn_model_authority_input_from_dict,
    e6_nextn_model_authority_input_to_dict,
    publish_e6_recursive_source_dag_artifact,
    signed_e1a_verification_from_dict,
    signed_e1a_verification_to_dict,
    signed_e3b_confirmation_from_dict,
    signed_e3b_confirmation_to_dict,
    signed_e5_confirmation_from_dict,
    signed_e5_confirmation_to_dict,
    signed_e6_confirmation_from_dict,
    signed_e6_confirmation_to_dict,
    signed_e6_model_compatibility_from_dict,
    signed_e6_model_compatibility_to_dict,
)
from lightcone_spec.experiments.e6_stage_authority import (
    E6_CONFIRMATION_PROTOCOL_SHA256,
    E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
    E6ConfirmationReceipt,
    E6ModelCompatibilityReceipt,
    SignedE6ConfirmationReceipt,
    SignedE6ModelCompatibilityReceipt,
)
from lightcone_spec.experiments.formal_protocol import E6_MODELS, content_sha256
from lightcone_spec.experiments.formal_slo_metrics import (
    FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
)
from lightcone_spec.experiments.statistics import (
    PRIMARY_CONTRASTS,
    MultiplicityDecision,
    PairedBcaContrast,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return content_sha256({"e0-artifact-test": label})


def _signed_model_compatibility(tmp_path) -> SignedE6ModelCompatibilityReceipt:
    sources = tuple(
        _source(tmp_path, model, index) for index, model in enumerate(E6_MODELS)
    )
    payload = E6ModelCompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_sha256=_sha("registry"),
        release_root_manifest_sha256=_sha("release-root"),
        protocol_sha256=E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        models=tuple(
            _compatibility(source, index) for index, source in enumerate(sources)
        ),
    )
    payload_sha256, challenge, attestation = _signature_parts(
        payload, key_id="e6-compatibility-key"
    )
    return SignedE6ModelCompatibilityReceipt(
        payload=payload,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )


def _signed_confirmation(
    compatibility: SignedE6ModelCompatibilityReceipt,
) -> SignedE6ConfirmationReceipt:
    blocks = tuple(sorted(f"E6:final:{index}" for index in range(4, 16)))
    contrasts = tuple(
        PairedBcaContrast(
            name=name,
            block_ids=blocks,
            mean_log_ratio=0.1,
            mean_relative_gain=0.105,
            ci_lower_relative_gain=0.01,
            ci_upper_relative_gain=0.2,
            raw_p_value=0.001,
            confidence=0.95,
        )
        for name in PRIMARY_CONTRASTS
    )
    decisions = tuple(
        MultiplicityDecision(
            name=name,
            raw_p_value=0.001,
            adjusted_p_value=0.002,
            rejected=True,
            procedure="holm",
        )
        for name in PRIMARY_CONTRASTS
    )
    payload = E6ConfirmationReceipt(
        schema_version=1,
        protocol_lock_sha256=compatibility.payload.protocol_lock_sha256,
        registry_sha256=compatibility.payload.registry_sha256,
        materialization_receipt_sha256=_sha("e6-materialization"),
        coverage_receipt_sha256=_sha("e6-coverage"),
        evidence_manifest_sha256=_sha("e6-evidence"),
        inventory_sha256=_sha("inventory"),
        protocol_sha256=E6_CONFIRMATION_PROTOCOL_SHA256,
        upstream_e5_confirmation_sha256=_sha("e5-confirmation"),
        signed_model_compatibility_sha256=compatibility.sha256,
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        models=E6_MODELS,
        final_block_ids=blocks,
        primary_contrasts=contrasts,
        holm_decisions=decisions,
        status="CONFIRMED",
    )
    payload_sha256, challenge, attestation = _signature_parts(
        payload, key_id="e6-confirmation-key"
    )
    return SignedE6ConfirmationReceipt(
        payload=payload,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )


def _primary_contrasts(prefix: str) -> tuple[PairedBcaContrast, ...]:
    blocks = tuple(f"{prefix}:final:{index:02d}" for index in range(12))
    return tuple(
        PairedBcaContrast(
            name=name,
            block_ids=blocks,
            mean_log_ratio=0.1,
            mean_relative_gain=0.105,
            ci_lower_relative_gain=0.01,
            ci_upper_relative_gain=0.2,
            raw_p_value=0.001,
            confidence=0.95,
        )
        for name in PRIMARY_CONTRASTS
    )


def _holm_decisions() -> tuple[MultiplicityDecision, ...]:
    return tuple(
        MultiplicityDecision(
            name=name,
            raw_p_value=0.001,
            adjusted_p_value=0.002,
            rejected=True,
            procedure="holm",
        )
        for name in PRIMARY_CONTRASTS
    )


def _signed_e3b_confirmation() -> SignedE3bConfirmationReceipt:
    contrasts = _primary_contrasts("E3b")
    decisions = _holm_decisions()
    family_confirmations = []
    for context in (1024, 2048, 4096, 8192, 16384, 24576, 32768, 40928):
        for regime in (
            "long_input_short_output",
            "short_input_long_generation",
            "multi_turn_shared_prefix",
        ):
            for load in ("concurrency_one", "common_load"):
                for width_panel in ("matched", "deployment_optimal"):
                    dimensions = tuple(
                        sorted(
                            {
                                "context": context,
                                "load": load,
                                "regime": regime,
                                "width_panel": width_panel,
                            }.items()
                        )
                    )
                    family_sha256 = content_sha256(
                        {
                            "stage": "E3b",
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "task": "heldout_long_context_confirmation",
                            "dimensions": list(dimensions),
                        }
                    )
                    family_confirmations.append(
                        FormalFamilyConfirmationResult(
                            schema_version=1,
                            stage="E3b",
                            model="meta-llama/Llama-3.1-8B-Instruct",
                            task="heldout_long_context_confirmation",
                            family_dimensions=dimensions,  # type: ignore[arg-type]
                            family_sha256=family_sha256,
                            slo_goodput_protocol_sha256=(
                                FORMAL_SLO_GOODPUT_PROTOCOL_SHA256
                            ),
                            final_block_ids=contrasts[0].block_ids,
                            final_goodput_observation_sha256s=tuple(
                                sorted(
                                    (
                                        block_id,
                                        role,
                                        _sha(
                                            f"e3b-observation-{family_sha256}-"
                                            f"{block_id}-{role}"
                                        ),
                                    )
                                    for block_id in contrasts[0].block_ids
                                    for role in ("Static", "TTS", "LightCone")
                                )
                            ),
                            primary_contrasts=contrasts,
                            holm_decisions=decisions,
                            status="CONFIRMED",
                        )
                    )
    payload = E3bConfirmationReceipt(
        schema_version=2,
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_sha256=_sha("registry"),
        materialization_receipt_sha256=_sha("e3b-materialization"),
        coverage_receipt_sha256=_sha("e3b-coverage"),
        evidence_manifest_sha256=_sha("e3b-evidence"),
        inventory_sha256=_sha("inventory"),
        protocol_sha256=E3B_CONFIRMATION_PROTOCOL_SHA256,
        model="meta-llama/Llama-3.1-8B-Instruct",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        final_block_ids=contrasts[0].block_ids,
        family_confirmations=tuple(
            sorted(family_confirmations, key=lambda row: row.family_sha256)
        ),
        status="CONFIRMED",
    )
    payload_sha256, challenge, attestation = _signature_parts(
        payload, key_id="e3b-confirmation-key"
    )
    return SignedE3bConfirmationReceipt(
        payload=payload,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )


def _signed_e1a_verification() -> SignedE1aVerificationReceipt:
    evaluations = tuple(
        sorted(
            (
                E1aConfigurationEvaluation(
                    configuration=(
                        ("parameterization", f"parameterization-{index}"),
                        ("rank", index + 1),
                        ("scope", f"scope-{index}"),
                    ),
                    cell_ids=tuple(
                        sorted(
                            (
                                _sha(f"e1a-cell-{index}-0"),
                                _sha(f"e1a-cell-{index}-1"),
                            )
                        )
                    ),
                    minimum_confidence_lower_request_rate_ratio=1.01,
                    peak_hbm_bytes=1_000 + index,
                    p99_itl_us=100 + index,
                    exposed_update_us=10 + index,
                )
                for index in range(56)
            ),
            key=lambda row: row.configuration_sha256,
        )
    )
    selected = evaluations[0].configuration
    source_recipe = _sha("lightcone")
    selected_recipe = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_e1a_selected_dspark_recipe",
            "source_lightcone_recipe_sha256": source_recipe,
            "configuration": selected,
            "verification_protocol_sha256": E1A_VERIFICATION_PROTOCOL_SHA256,
        }
    )
    payload = E1aVerificationReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_sha256=_sha("registry"),
        materialization_receipt_sha256=_sha("e1a-materialization"),
        coverage_receipt_sha256=_sha("e1a-coverage"),
        evidence_manifest_sha256=_sha("e1a-evidence"),
        inventory_sha256=_sha("inventory"),
        protocol_sha256=E1A_VERIFICATION_PROTOCOL_SHA256,
        model="meta-llama/Llama-3.1-8B-Instruct",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        source_lightcone_recipe_sha256=source_recipe,
        evaluations=evaluations,
        selected_configuration=selected,
        selected_dspark_recipe_sha256=selected_recipe,
    )
    payload_sha256, challenge, attestation = _signature_parts(
        payload, key_id="e1a-verification-key"
    )
    return SignedE1aVerificationReceipt(
        payload=payload,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )


def _signed_e5_confirmation() -> SignedE5ConfirmationReceipt:
    anchors = tuple(
        E5P99AnchorCompletion(
            anchor_id=_sha(f"e5-anchor-{index}"),
            completed_requests=10_000,
        )
        for index in range(6)
    )
    anchors = tuple(sorted(anchors, key=lambda row: row.anchor_id))
    contrasts = _primary_contrasts("E5")
    decisions = _holm_decisions()
    family_confirmations = []
    for backend in ("DFLASH", "DSPARK"):
        for index in range(45):
            dimensions = (
                ("backend_authority", backend),
                ("family", f"family-{index:02d}"),
            )
            family_sha256 = content_sha256(
                {
                    "stage": "E5",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "task": "production_slo_power_prefix",
                    "dimensions": list(dimensions),
                }
            )
            family_confirmations.append(
                FormalFamilyConfirmationResult(
                    schema_version=1,
                    stage="E5",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    task="production_slo_power_prefix",
                    family_dimensions=dimensions,
                    family_sha256=family_sha256,
                    slo_goodput_protocol_sha256=FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
                    final_block_ids=contrasts[0].block_ids,
                    final_goodput_observation_sha256s=tuple(
                        sorted(
                            (
                                block_id,
                                role,
                                _sha(
                                    f"e5-observation-{family_sha256}-{block_id}-{role}"
                                ),
                            )
                            for block_id in contrasts[0].block_ids
                            for role in ("Static", "TTS", "LightCone")
                        )
                    ),
                    primary_contrasts=contrasts,
                    holm_decisions=decisions,
                    status="CONFIRMED",
                )
            )
    payload = E5ConfirmationReceipt(
        schema_version=2,
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_sha256=_sha("registry"),
        materialization_receipt_sha256=_sha("e5-materialization"),
        coverage_receipt_sha256=_sha("e5-coverage"),
        headline_evidence_manifest_sha256=_sha("e5-headline-evidence"),
        failure_evidence_manifest_sha256=_sha("e5-failure-evidence"),
        inventory_sha256=_sha("inventory"),
        protocol_sha256=E5_CONFIRMATION_PROTOCOL_SHA256,
        model="meta-llama/Llama-3.1-8B-Instruct",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        dflash_lightcone_recipe_sha256=_sha("dflash-lightcone"),
        dspark_lightcone_recipe_sha256=_sha("dspark-lightcone"),
        block_count=12,
        headline_cell_count=450 * 12,
        failure_cell_count=264,
        family_confirmations=tuple(
            sorted(family_confirmations, key=lambda row: row.family_sha256)
        ),
        p99_anchor_completions=anchors,
        failure_result_sha256s=tuple(
            sorted(_sha(f"e5-failure-result-{index}") for index in range(264))
        ),
        status="CONFIRMED",
    )
    payload_sha256, challenge, attestation = _signature_parts(
        payload, key_id="e5-confirmation-key"
    )
    return SignedE5ConfirmationReceipt(
        payload=payload,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )


def _binding(tmp_path, label: str) -> CanonicalJsonProofBinding:
    path = tmp_path / f"{label}.json"
    publish_canonical_json_no_replace(
        path,
        {"schema_version": 1, "kind": "e0_artifact_test_source", "label": label},
    )
    return CanonicalJsonProofBinding.bind(str(path))


def _node(tmp_path, node_id: str) -> FormalStageProofNode:
    profiler = node_id == "e4_profiler"
    e2 = node_id == "e2_final"
    e5 = node_id == "e5_final"
    return FormalStageProofNode(
        schema_version=1,
        kind="lightcone_formal_stage_proof_node",
        node_id=node_id,  # type: ignore[arg-type]
        materialization_source=_binding(tmp_path, f"{node_id}-materialization"),
        coverage_source=_binding(tmp_path, f"{node_id}-coverage"),
        evidence_manifest_source=(
            None if profiler else _binding(tmp_path, f"{node_id}-evidence")
        ),
        stage_source_rebuild_source=(
            None if profiler or e2 else _binding(tmp_path, f"{node_id}-stage-source")
        ),
        execution_rebuild_shards=(
            () if profiler else (_binding(tmp_path, f"{node_id}-serving-shard"),)
        ),
        failure_evidence_manifest_source=(
            _binding(tmp_path, f"{node_id}-failure-evidence") if e5 else None
        ),
        failure_execution_rebuild_shards=(
            (_binding(tmp_path, f"{node_id}-failure-shard"),) if e5 else ()
        ),
    )


def test_e6_typed_authority_codecs_round_trip(tmp_path) -> None:
    source = _source(tmp_path, E6_MODELS[0], 0)
    source_json = json.loads(json.dumps(e6_nextn_model_authority_input_to_dict(source)))
    assert e6_nextn_model_authority_input_from_dict(source_json) == source

    compatibility = _signed_model_compatibility(tmp_path)
    compatibility_json = json.loads(
        json.dumps(signed_e6_model_compatibility_to_dict(compatibility))
    )
    assert signed_e6_model_compatibility_from_dict(compatibility_json) == compatibility

    confirmation = _signed_confirmation(compatibility)
    confirmation_json = json.loads(
        json.dumps(signed_e6_confirmation_to_dict(confirmation))
    )
    assert signed_e6_confirmation_from_dict(confirmation_json) == confirmation


def test_e6_typed_authority_codecs_reject_foreign_or_changed_sources(tmp_path) -> None:
    source = _source(tmp_path, E6_MODELS[0], 0)
    encoded = e6_nextn_model_authority_input_to_dict(source)
    encoded["source_input_sha256"] = _sha("foreign-source")
    with pytest.raises(ValueError, match="digest differs"):
        e6_nextn_model_authority_input_from_dict(encoded)

    compatibility = _signed_model_compatibility(tmp_path)
    encoded_compatibility = signed_e6_model_compatibility_to_dict(compatibility)
    encoded_compatibility["digest_only_fallback"] = compatibility.sha256
    with pytest.raises(ValueError, match="fields differ"):
        signed_e6_model_compatibility_from_dict(encoded_compatibility)


def test_recursive_signed_decision_codecs_round_trip() -> None:
    e3b = _signed_e3b_confirmation()
    e1a = _signed_e1a_verification()
    e5 = _signed_e5_confirmation()
    assert (
        signed_e3b_confirmation_from_dict(
            json.loads(json.dumps(signed_e3b_confirmation_to_dict(e3b)))
        )
        == e3b
    )
    assert (
        signed_e1a_verification_from_dict(
            json.loads(json.dumps(signed_e1a_verification_to_dict(e1a)))
        )
        == e1a
    )
    assert (
        signed_e5_confirmation_from_dict(
            json.loads(json.dumps(signed_e5_confirmation_to_dict(e5)))
        )
        == e5
    )

    tampered = signed_e1a_verification_to_dict(e1a)
    tampered["digest_only_fallback"] = e1a.sha256
    with pytest.raises(ValueError, match="fields differ"):
        signed_e1a_verification_from_dict(tampered)


def test_recursive_source_dag_is_closed_path_bound_and_no_replace(tmp_path) -> None:
    nodes = tuple(
        _node(tmp_path, node_id)
        for node_id in (
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
    )
    artifact = E6RecursiveSourceDagArtifact(
        schema_version=1,
        kind="lightcone_e6_recursive_source_dag_artifact",
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_verification_receipt_sha256=_sha("registry-receipt"),
        signed_e3b_confirmation=_signed_e3b_confirmation(),
        signed_e1a_verification=_signed_e1a_verification(),
        signed_e5_confirmation=_signed_e5_confirmation(),
        nodes=nodes,
    )
    encoded = json.loads(json.dumps(artifact.to_dict()))
    assert E6RecursiveSourceDagArtifact.from_dict(encoded) == artifact

    output = tmp_path / "e6-recursive-dag.json"
    binding = publish_e6_recursive_source_dag_artifact(artifact, output)
    assert E6RecursiveSourceDagArtifact.from_dict(binding.reopen()) == artifact
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_e6_recursive_source_dag_artifact(artifact, output)

    encoded["nodes"] = encoded["nodes"][:-1]
    encoded["artifact_sha256"] = content_sha256(
        {key: value for key, value in encoded.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="node coverage"):
        E6RecursiveSourceDagArtifact.from_dict(encoded)


def test_stage_proof_node_rejects_path_alias_and_digest_only_field(tmp_path) -> None:
    node = _node(tmp_path, "e4_profiler")
    encoded = json.loads(json.dumps(node.to_dict()))
    encoded["digest_only_fallback"] = node.sha256
    with pytest.raises(ValueError, match="fields differ"):
        FormalStageProofNode.from_dict(encoded)

    with pytest.raises(ValueError, match="reuses a source path"):
        FormalStageProofNode(
            schema_version=1,
            kind="lightcone_formal_stage_proof_node",
            node_id="e4_profiler",
            materialization_source=node.materialization_source,
            coverage_source=node.materialization_source,
            evidence_manifest_source=None,
            stage_source_rebuild_source=None,
            execution_rebuild_shards=(),
        )
