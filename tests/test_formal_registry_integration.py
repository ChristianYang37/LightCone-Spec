from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.experiments.e3a_stage_authority import (
    E3A_STAGED_REDUCTION_PROTOCOL_SHA256,
    E3aCapacityObservation,
    E3aLockedOutput,
    E3aStagedSelectionArtifact,
    SignedE3aStagedSelectionReceipt,
    build_e3a_staged_selection_receipt,
)
from lightcone_spec.experiments.e4_stage_authority import (
    E4CellExecutionEvidence,
    E4ConfigurationEvaluation,
    E4StagedEvidenceManifest,
    E4StageSelectionReceipt,
    SignedE4StageSelectionReceipt,
)
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_STAGE_DAG,
    TTS_PRIMARY_SOURCE_ID,
    TTS_PRIMARY_SOURCE_VERSION,
    CandidateStateReplay,
    CandidateStateTerminalPair,
    ProtocolLock,
    SignedProtocolLock,
    SignedTtsCalibrationSeal,
    TtsCalibrationAuthority,
    TtsCalibrationSeal,
    TtsL0CandidateStateCoverage,
    content_sha256,
    reject_banned_model_identity,
)
from lightcone_spec.experiments.formal_registry import (
    FormalMaterializationBinding,
    _build_formal_registry_manifest_with_policy,
    _ordered_materializations,
    _validate_downstream_main_materialization,
    e3a_staged_selection_artifact_from_dict,
    e3a_staged_selection_artifact_to_dict,
    e4_staged_evidence_manifest_from_dict,
    e4_staged_evidence_manifest_to_dict,
    signed_e3a_staged_selection_from_dict,
    signed_e3a_staged_selection_to_dict,
    signed_e4_stage_selection_from_dict,
    signed_e4_stage_selection_to_dict,
    signed_protocol_lock_from_dict,
    signed_protocol_lock_to_dict,
    signed_stage_coverage_from_dict,
    signed_stage_coverage_to_dict,
    signed_stage_materialization_from_dict,
    signed_stage_materialization_to_dict,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.registry import (
    INDUSTRIAL_EXPERIMENT_ORDER,
    CellStatus,
    build_industrial_registry,
    build_legacy_industrial_registry,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    SignedStageCoverageReceipt,
    SignedStageMaterializationReceipt,
    StageCellDisposition,
    StageCoverageReceipt,
    _materialize_e1_first_slice_from_verified_decisions,
    _materialize_e3a_diagnostic,
    _materialize_e3b_diagnostic,
    _materialize_tts_calibration_diagnostic,
    _select_exact_final_prefix,
    materialize_preflight,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.mark.parametrize(
    "pilot_rule",
    (
        "e3b_exact_480_rows_x_4_excluded_pilot_blocks",
        "e5_exact_450_headline_rows_x_4_excluded_pilot_blocks",
        ("e6_exact_two_model_preflights_plus_60_rows_x_4_excluded_pilot_blocks"),
    ),
)
def test_main_registry_rejects_excluded_pilot_stage_receipts(
    pilot_rule: str,
) -> None:
    fixture = _materialize_e3b_diagnostic(
        protocol_lock_sha256=_sha("protocol"),
        upstream_receipt_sha256=_sha("e4-profiler"),
        source_decision_sha256=_sha("e4-selection"),
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        final_blocks=12,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    tuning_only = replace(
        fixture,
        materialization_rule=pilot_rule,
    )
    with pytest.raises(ValueError, match="tuning materialization"):
        _ordered_materializations((tuning_only,))


def test_main_registry_requires_final_only_typed_pilot_power_lineage() -> None:
    source = _sha("signed-power-prefix")
    fixture = _materialize_e3b_diagnostic(
        protocol_lock_sha256=_sha("protocol"),
        upstream_receipt_sha256=_sha("e4-profiler"),
        source_decision_sha256=source,
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        final_blocks=12,
        gpu_hours=GpuHourEstimate.unmeasured(),
        lineage_dimensions={
            "pilot_coverage_receipt_sha256": _sha("pilot-coverage"),
            "pilot_materialization_receipt_sha256": _sha("pilot-materialization"),
            "signed_power_prefix_sha256": source,
        },
    )
    final_cells = _select_exact_final_prefix(
        fixture.cells,
        selected_final_prefix=tuple(range(4, 16)),
        expected_cells_per_block=480,
    )
    final = replace(
        fixture,
        materialization_rule=(
            "five_roles_x_8_contexts_x_3_regimes_x_2_loads_x_2_widths_final_only"
        ),
        expected_cell_count=len(final_cells),
        cells=tuple(sorted(final_cells, key=lambda cell: cell.cell_id)),
    )
    _validate_downstream_main_materialization(final)

    with pytest.raises(ValueError, match="typed pilot/power lineage"):
        _validate_downstream_main_materialization(
            replace(final, source_decision_sha256=_sha("foreign-power-prefix"))
        )

    with pytest.raises(ValueError, match="lacks its tuning-only pilot lineage"):
        FormalMaterializationBinding(
            stage="E3b",
            materialization_receipt_sha256=final.sha256,
            signed_receipt_sha256=_sha("signed-final"),
            source_decision_sha256=source,
            expected_cell_count=final.expected_cell_count,
            materialization_rule=final.materialization_rule,
        )


def _json_round_trip(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def test_e4_source_authority_codecs_bind_paths_and_signed_winner(
    tmp_path: Path,
) -> None:
    native_path = (tmp_path / "e4-native.json").resolve()
    timing_path = (tmp_path / "e4-timing.json").resolve()
    publish_canonical_json_no_replace(native_path, {"kind": "e4_native_fixture"})
    publish_canonical_json_no_replace(timing_path, {"kind": "e4_timing_fixture"})
    native = CanonicalJsonProofBinding.bind(native_path)
    timing = CanonicalJsonProofBinding.bind(timing_path)
    cell_id = _sha("e4-cell")
    identity = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id=cell_id,
        inventory_sha256=_sha("inventory"),
        registry_sha256=_sha("registry"),
        execution_plan_sha256=_sha("e4-plan"),
        rank_config_sha256=_sha("e4-ranks"),
        run_id="e4-screen-run",
        run_nonce_sha256=_sha("e4-run-nonce"),
        attempt_id="e4-screen-attempt",
        method="l0",
    )
    evidence = E4CellExecutionEvidence(
        schema_version=1,
        materialized_cell_id=cell_id,
        execution_binding_sha256=_sha("e4-binding"),
        execution_identity=identity,
        native_result_proof_path=native.absolute_path,
        native_result_proof_raw_sha256=native.raw_sha256,
        native_result_proof_semantic_sha256=native.semantic_sha256,
        stage_itl_proof_path=timing.absolute_path,
        stage_itl_proof_raw_sha256=timing.raw_sha256,
        stage_itl_proof_semantic_sha256=timing.semantic_sha256,
    )
    manifest = E4StagedEvidenceManifest(
        schema_version=1,
        phase="screen",
        protocol_lock_sha256=_sha("lock"),
        materialization_receipt_sha256=_sha("e4-screen-materialization"),
        coverage_receipt_sha256=_sha("e4-screen-coverage"),
        upstream_signed_authority_sha256=_sha("e2-final-selection"),
        inventory_sha256=_sha("inventory"),
        cells=(evidence,),
    )
    decoded_manifest = e4_staged_evidence_manifest_from_dict(
        _json_round_trip(e4_staged_evidence_manifest_to_dict(manifest))
    )
    assert decoded_manifest == manifest

    winner = (
        ("update_stride", 1),
        ("microbatch", 1),
        ("coalescing", 1),
        ("stream_priority", "default"),
    )
    evaluation = E4ConfigurationEvaluation(
        configuration=winner,
        cell_ids=tuple(sorted(_sha(f"e4-stratum-{index}") for index in range(6))),
        minimum_request_rate_numerator=2,
        minimum_request_rate_denominator=1,
        peak_hbm_bytes=1,
        p99_itl_us=1,
        exposed_update_us=1,
    )
    selection = E4StageSelectionReceipt(
        schema_version=1,
        phase="screen",
        protocol_lock_sha256=_sha("lock"),
        registry_sha256=_sha("registry"),
        materialization_receipt_sha256=_sha("e4-screen-materialization"),
        coverage_receipt_sha256=_sha("e4-screen-coverage"),
        upstream_signed_authority_sha256=_sha("e2-final-selection"),
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=_sha("inventory"),
        model="Qwen/Qwen3-8B",
        lightcone_recipe_sha256=_sha("e2-winner"),
        evaluations=(evaluation,),
        winner_configuration=winner,
        factor_neighborhoods=(
            ("update_stride", 1, 5),
            ("microbatch", 1, 2),
            ("coalescing", 1, 2),
            ("stream_priority", "default", "high"),
        ),
    )
    signer = _Signer()
    signed = SignedE4StageSelectionReceipt(selection, *signer.sign(selection))
    decoded_signed = signed_e4_stage_selection_from_dict(
        _json_round_trip(signed_e4_stage_selection_to_dict(signed))
    )
    assert decoded_signed == signed


class _Signer:
    def __init__(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        public_bytes = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.public_base64 = base64.b64encode(public_bytes).decode()
        self.public_sha256 = hashlib.sha256(public_bytes).hexdigest()
        self.policy = TrustedAttesterPolicy(
            policy_id="formal-registry-integration-v1",
            trusted_attesters=(
                ("formal-registry-signer", "formal-registry-key", self.public_sha256),
            ),
            public_keys=((self.public_sha256, self.public_base64),),
        )
        self.now_ns = 30_000_000_000
        self.counter = 0

    def sign(self, payload: object):
        self.counter += 1
        payload_sha256 = content_sha256(payload)
        challenge = AttestationChallenge.issue(
            challenge_id=f"formal-registry-challenge-{self.counter}",
            subject_sha256=payload_sha256,
            lifetime_s=60,
            now_ns=self.now_ns,
        )
        signature = self.private_key.sign(
            attestation_message(challenge, payload_sha256=payload_sha256)
        )
        attestation = SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="formal-registry-signer",
            key_id="formal-registry-key",
            environment="release",
            public_key_base64=self.public_base64,
            challenge_sha256=challenge.sha256,
            payload_sha256=payload_sha256,
            signature_base64=base64.b64encode(signature).decode(),
        )
        return payload_sha256, challenge, attestation


def _protocol_lock() -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="lightcone-formal-protocol-v1",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=build_industrial_registry().sha256,
        english_protocol_sha256=_sha("existing-english-protocol"),
        chinese_protocol_sha256=_sha("existing-chinese-protocol"),
        tts_calibration_authority_sha256=_sha("tts-authority"),
        chronobelief_authority_sha256=_sha("chronobelief-authority"),
        e1_recipe_anchor_authority_sha256=_sha("e1-recipe-anchors"),
        e2_recipe_grid_authority_sha256=_sha("e2-grid"),
        formal_runtime_authority_manifest_sha256=_sha("formal-runtime"),
        offline_release_trust_root_sha256=_sha("release-root"),
        prepared_model_content_authorization_sha256=_sha("prepared-model"),
        formal_workload_e3a_authorization_sha256=_sha("e3a-workload"),
        formal_workload_e0_authorization_sha256=_sha("e0-workload"),
        burstgpt_shape_authorization_sha256=_sha("burstgpt-shape"),
        native_runtime_qualification_protocol_sha256=_sha("native-protocol"),
        native_runtime_qualification_runner_sha256=_sha("native-runner"),
        native_runtime_qualification_test_set_sha256=_sha("native-tests"),
        compile_qualification_protocol_sha256=_sha("compile-protocol"),
        compile_qualification_runner_sha256=_sha("compile-runner"),
        compile_qualification_test_set_sha256=_sha("compile-tests"),
        exactness_qualification_protocol_sha256=_sha("exactness-protocol"),
        exactness_qualification_runner_sha256=_sha("exactness-runner"),
        exactness_qualification_test_set_sha256=_sha("exactness-tests"),
    )


def _tts_authority() -> TtsCalibrationAuthority:
    return TtsCalibrationAuthority(
        schema_version=1,
        authority_id="formal-registry-tts-v2",
        primary_source_id=TTS_PRIMARY_SOURCE_ID,
        primary_source_version=TTS_PRIMARY_SOURCE_VERSION,
        paper_pdf_sha256=_sha("tts-paper-pdf"),
        paper_source_sha256=_sha("tts-paper-source"),
        tuning_window_sha256=_sha("tts-tuning-window"),
        trainable_plan_sha256=_sha("tts-trainable-plan"),
        drafter_native_loss_recipe_sha256=_sha("tts-native-loss"),
    )


def _candidate_coverage(materialization) -> TtsL0CandidateStateCoverage:
    exactness = next(
        cell
        for cell in materialization.cells
        if cell.task == "exactness_memory_telemetry_preflight"
    )
    plan = _sha("candidate-plan")
    shared = {
        "source_round": 1,
        "source_version": 0,
        "source_state_sha256": _sha("candidate-source"),
        "trainable_plan_sha256": plan,
        "candidate_bytes_sha256": _sha("candidate-bytes"),
        "optimizer_state_bytes_sha256": _sha("candidate-optimizer-state"),
        "proposal_evidence_sha256": _sha("candidate-proposal"),
    }
    return TtsL0CandidateStateCoverage(
        schema_version=1,
        stage="preflight",
        scope="preflight_exactness_qualification",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        pair_id=_sha("candidate-pair"),
        tts_cell_id=_sha("candidate-tts-fixture"),
        l0_naive_cell_id=_sha("candidate-l0-fixture"),
        tts_native_replay_pointer_sha256=_sha("candidate-tts-pointer"),
        l0_naive_native_replay_pointer_sha256=_sha("candidate-l0-pointer"),
        qualification_cell_id=exactness.cell_id,
        source_round_plan_sha256=_sha("candidate-round-plan"),
        trainable_plan_sha256=plan,
        expected_source_rounds=(1,),
        tts_observations=(
            CandidateStateReplay(
                method_role="TTS",
                cell_id=_sha("candidate-tts-fixture"),
                native_replay_pointer_sha256=_sha("candidate-tts-pointer"),
                run_id="preflight-tts-round-0",
                publication_policy="fixed_barrier",
                **shared,
            ),
        ),
        l0_naive_observations=(
            CandidateStateReplay(
                method_role="L0-naive",
                cell_id=_sha("candidate-l0-fixture"),
                native_replay_pointer_sha256=_sha("candidate-l0-pointer"),
                run_id="preflight-l0-round-0",
                publication_policy="first_ready",
                **shared,
            ),
        ),
        terminal_pairs=(
            CandidateStateTerminalPair(
                source_round=1,
                tts_cell_id=_sha("candidate-tts-fixture"),
                l0_naive_cell_id=_sha("candidate-l0-fixture"),
                tts_run_id="preflight-tts-round-0",
                l0_naive_run_id="preflight-l0-round-0",
                tts_native_replay_pointer_sha256=_sha("candidate-tts-pointer"),
                l0_naive_native_replay_pointer_sha256=_sha("candidate-l0-pointer"),
                proposal_evidence_sha256=_sha("candidate-proposal"),
                tts_terminal_receipt_sha256=_sha("candidate-tts-terminal"),
                l0_naive_terminal_receipt_sha256=_sha("candidate-l0-terminal"),
            ),
        ),
    )


def _diagnostic_manifest(
    signed_lock: SignedProtocolLock,
    *,
    signed_materializations: tuple[SignedStageMaterializationReceipt, ...],
    signed_coverage: tuple[SignedStageCoverageReceipt, ...],
    signer: _Signer,
    e3a_staged_selection_artifacts: tuple[E3aStagedSelectionArtifact, ...] = (),
    signed_e3a_staged_selections: tuple[SignedE3aStagedSelectionReceipt, ...] = (),
    tts_calibration_authorities: tuple[TtsCalibrationAuthority, ...] = (),
    signed_tts_calibration_seals: tuple[SignedTtsCalibrationSeal, ...] = (),
):
    return _build_formal_registry_manifest_with_policy(
        signed_lock,
        signed_materializations=signed_materializations,
        signed_coverage=signed_coverage,
        e3a_staged_selection_artifacts=e3a_staged_selection_artifacts,
        signed_e3a_staged_selections=signed_e3a_staged_selections,
        tts_calibration_authorities=tts_calibration_authorities,
        signed_tts_calibration_seals=signed_tts_calibration_seals,
        policy=signer.policy,
        expected_policy_sha256=signer.policy.sha256,
        inventory_sha256=_sha("inventory"),
        deployment_policy_authorization_sha256=_sha("deployment"),
        control_lineage_sha256=_sha("control-lineage"),
        control_envelope_sha256s=(_sha("control-envelope"),),
        challenge_reservation_sha256=_sha("challenge-reservation"),
        now_ns=signer.now_ns,
    )


def _complete_coverage(
    materialization,
    *,
    candidate_coverages: tuple[TtsL0CandidateStateCoverage, ...] = (),
) -> StageCoverageReceipt:
    return StageCoverageReceipt(
        schema_version=2,
        stage=materialization.stage,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            sorted(
                (
                    StageCellDisposition(
                        stage=materialization.stage,
                        cell_id=cell.cell_id,
                        status="COMPLETE",
                        reason_code="terminal_complete",
                        terminal_receipt_sha256=_sha(
                            f"{materialization.stage}-terminal-{cell.cell_id}"
                        ),
                    )
                    for cell in materialization.cells
                ),
                key=lambda row: row.cell_id,
            )
        ),
        tts_l0_candidate_state_coverages=candidate_coverages,
    )


def _synthetic_e3a_artifact(
    *,
    lock: ProtocolLock,
    materialization,
    coverage: StageCoverageReceipt,
) -> E3aStagedSelectionArtifact:
    targets = {
        (
            dimensions["context"],
            dimensions["regime"],
            dimensions["concurrency"],
        ): cell.cell_id
        for cell in materialization.cells
        if cell.method_role == "Target-only"
        for dimensions in (dict(cell.dimensions),)
    }
    observations = []
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        target_cell_id = targets[
            (
                dimensions["context"],
                dimensions["regime"],
                dimensions["concurrency"],
            )
        ]
        observations.append(
            E3aCapacityObservation(
                cell_id=cell.cell_id,
                method_role=cell.method_role,
                context=dimensions["context"],
                regime=dimensions["regime"],
                concurrency=dimensions["concurrency"],
                width=dimensions.get("width"),
                throughput_tokens=2,
                throughput_window_ns=100,
                peak_hbm_bytes=1,
                target_cell_id=(
                    None if cell.method_role == "Target-only" else target_cell_id
                ),
                static_target_ratio_numerator=(
                    None if cell.method_role == "Target-only" else 1
                ),
                static_target_ratio_denominator=(
                    None if cell.method_role == "Target-only" else 1
                ),
                execution_evidence_sha256=_sha(f"e3a-evidence-{cell.cell_id}"),
                terminal_sha256=_sha(f"e3a-terminal-{cell.cell_id}"),
                timing_authority_sha256=_sha(f"e3a-timing-{cell.cell_id}"),
            )
        )
    locked_outputs = tuple(
        E3aLockedOutput(name=name, content_sha256=_sha(f"e3a-locked-{name}"))
        for name in (
            "baseline_capacity_envelope",
            "drift_witness",
            "e1_reference_load",
            "matched_width",
            "static_target_crossover",
            "width_selection_rule",
        )
    )
    return E3aStagedSelectionArtifact(
        schema_version=1,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        inventory_sha256=_sha("inventory"),
        evidence_manifest_sha256=_sha("e3a-evidence-manifest"),
        reducer_authority_member_sha256=_sha("e3a-reducer-member"),
        reducer_protocol_sha256=E3A_STAGED_REDUCTION_PROTOCOL_SHA256,
        model="Qwen/Qwen3-8B",
        matched_width=8,
        common_load=1,
        observations=tuple(sorted(observations, key=lambda row: row.cell_id)),
        locked_outputs=locked_outputs,
    )


def test_default_registry_is_concrete_signed_prefix_not_eager_templates() -> None:
    registry = build_industrial_registry()
    assert registry.materialization_mode == "signed_staged"
    assert (
        tuple(definition.name for definition in registry.definitions)
        == (INDUSTRIAL_EXPERIMENT_ORDER)
        == FORMAL_STAGE_DAG
    )
    assert {
        stage: len(registry.cells_for(stage)) for stage in INDUSTRIAL_EXPERIMENT_ORDER
    } == {
        "preflight": 10,
        "E3a": 360,
        "TTS-Cal": 288,
        "E1": 0,
        "E2": 0,
        "E4": 0,
        "E3b": 0,
        "E1a": 0,
        "E5": 0,
        "E6": 0,
        "E0": 0,
    }
    assert {cell.status for cell in registry.cells} == {CellStatus.UNMEASURED}
    serialized = json.dumps(registry.to_dict(), sort_keys=True).lower()
    assert "compatibility_template" not in serialized
    assert all(
        value not in {"frozen_tts_recipe", "sealed_e2_recipe"}
        for cell in registry.cells
        for value in (
            cell.identity.scope,
            cell.identity.optimizer,
            cell.identity.schedule,
            cell.identity.parameterization,
        )
    )
    reject_banned_model_identity(registry)

    legacy = build_legacy_industrial_registry()
    assert legacy.materialization_mode == "legacy_diagnostic"
    assert len(legacy.cells) > len(registry.cells)
    reject_banned_model_identity(legacy)


def test_signed_codecs_and_manifest_bind_exact_staged_registry() -> None:
    signer = _Signer()
    lock = _protocol_lock()
    lock_sha, lock_challenge, lock_attestation = signer.sign(lock)
    signed_lock = SignedProtocolLock(
        lock,
        lock_sha,
        lock_challenge,
        lock_attestation,
    )
    lock_codec = signed_protocol_lock_to_dict(signed_lock)
    decoded_lock = signed_protocol_lock_from_dict(_json_round_trip(lock_codec))
    assert decoded_lock.sha256 == signed_lock.sha256

    materialization = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    mat_sha, mat_challenge, mat_attestation = signer.sign(materialization)
    signed_materialization = SignedStageMaterializationReceipt(
        materialization,
        mat_sha,
        mat_challenge,
        mat_attestation,
    )
    mat_codec = signed_stage_materialization_to_dict(signed_materialization)
    decoded_materialization = signed_stage_materialization_from_dict(
        _json_round_trip(mat_codec)
    )
    assert decoded_materialization.sha256 == signed_materialization.sha256

    dispositions = tuple(
        StageCellDisposition(
            stage="preflight",
            cell_id=cell.cell_id,
            status="COMPLETE",
            reason_code="terminal_complete",
            terminal_receipt_sha256=_sha(f"terminal-{cell.cell_id}"),
        )
        for cell in materialization.cells
    )
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="preflight",
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(sorted(dispositions, key=lambda row: row.cell_id)),
        tts_l0_candidate_state_coverages=(_candidate_coverage(materialization),),
    )
    coverage_sha, coverage_challenge, coverage_attestation = signer.sign(coverage)
    signed_coverage = SignedStageCoverageReceipt(
        coverage,
        coverage_sha,
        coverage_challenge,
        coverage_attestation,
    )
    coverage_codec = signed_stage_coverage_to_dict(signed_coverage)
    decoded_coverage = signed_stage_coverage_from_dict(_json_round_trip(coverage_codec))
    assert decoded_coverage.sha256 == signed_coverage.sha256

    manifest = _diagnostic_manifest(
        decoded_lock,
        signed_materializations=(decoded_materialization,),
        signed_coverage=(decoded_coverage,),
        signer=signer,
    )
    assert manifest.status == "COVERED"
    assert manifest.formal_dispatch_authorized is False
    assert manifest.materializations[0].expected_cell_count == 10
    assert manifest.coverage[0].disposition_count == 10

    tampered = _json_round_trip(mat_codec)
    assert isinstance(tampered, dict)
    tampered["payload"]["cells"][0]["cell_id"] = _sha("tampered-cell")
    with pytest.raises(ValueError, match="cell ID differs"):
        signed_stage_materialization_from_dict(tampered)


def test_manifest_rejects_protocol_lock_for_another_registry() -> None:
    signer = _Signer()
    lock = _protocol_lock()
    wrong_lock = ProtocolLock(
        **{
            **lock.__dict__,
            "registry_sha256": _sha("different-registry"),
        }
    )
    payload_sha256, challenge, attestation = signer.sign(wrong_lock)
    with pytest.raises(ValueError, match="staged registry identity"):
        _diagnostic_manifest(
            SignedProtocolLock(
                wrong_lock,
                payload_sha256,
                challenge,
                attestation,
            ),
            signed_materializations=(),
            signed_coverage=(),
            signer=signer,
        )


def test_e3a_coverage_requires_exact_signed_staged_selection_source() -> None:
    signer = _Signer()
    lock = _protocol_lock()
    lock_sha, lock_challenge, lock_attestation = signer.sign(lock)
    signed_lock = SignedProtocolLock(
        lock,
        lock_sha,
        lock_challenge,
        lock_attestation,
    )

    preflight = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    preflight_sha, preflight_challenge, preflight_attestation = signer.sign(preflight)
    signed_preflight = SignedStageMaterializationReceipt(
        preflight,
        preflight_sha,
        preflight_challenge,
        preflight_attestation,
    )
    preflight_coverage = _complete_coverage(
        preflight,
        candidate_coverages=(_candidate_coverage(preflight),),
    )
    preflight_coverage_sha, challenge, attestation = signer.sign(preflight_coverage)
    signed_preflight_coverage = SignedStageCoverageReceipt(
        preflight_coverage,
        preflight_coverage_sha,
        challenge,
        attestation,
    )

    e3a = _materialize_e3a_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_preflight_receipt_sha256=signed_preflight_coverage.sha256,
        workload_authority_sha256=lock.formal_workload_e3a_authorization_sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    e3a_sha, challenge, attestation = signer.sign(e3a)
    signed_e3a = SignedStageMaterializationReceipt(
        e3a,
        e3a_sha,
        challenge,
        attestation,
    )
    e3a_coverage = _complete_coverage(e3a)
    e3a_coverage_sha, challenge, attestation = signer.sign(e3a_coverage)
    signed_e3a_coverage = SignedStageCoverageReceipt(
        e3a_coverage,
        e3a_coverage_sha,
        challenge,
        attestation,
    )
    artifact = _synthetic_e3a_artifact(
        lock=lock,
        materialization=e3a,
        coverage=e3a_coverage,
    )
    selection = build_e3a_staged_selection_receipt(artifact)
    selection_sha, challenge, attestation = signer.sign(selection)
    signed_selection = SignedE3aStagedSelectionReceipt(
        selection,
        selection_sha,
        challenge,
        attestation,
    )

    decoded_artifact = e3a_staged_selection_artifact_from_dict(
        _json_round_trip(e3a_staged_selection_artifact_to_dict(artifact))
    )
    decoded_selection = signed_e3a_staged_selection_from_dict(
        _json_round_trip(signed_e3a_staged_selection_to_dict(signed_selection))
    )
    manifest = _diagnostic_manifest(
        signed_lock,
        signed_materializations=(signed_preflight, signed_e3a),
        signed_coverage=(signed_preflight_coverage, signed_e3a_coverage),
        e3a_staged_selection_artifacts=(decoded_artifact,),
        signed_e3a_staged_selections=(decoded_selection,),
        signer=signer,
    )
    assert manifest.status == "COVERED"
    assert tuple(
        (
            row.stage,
            row.authority_kind,
            row.authority_sha256,
            row.signed_authority_sha256,
        )
        for row in manifest.source_authorities
    ) == (
        (
            "E3a",
            "e3a_staged_selection",
            artifact.sha256,
            signed_selection.sha256,
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires one exact staged selection artifact/receipt",
    ):
        _diagnostic_manifest(
            signed_lock,
            signed_materializations=(signed_preflight, signed_e3a),
            signed_coverage=(signed_preflight_coverage, signed_e3a_coverage),
            signer=signer,
        )

    foreign_artifact = replace(
        artifact,
        evidence_manifest_sha256=_sha("foreign-e3a-evidence-manifest"),
    )
    with pytest.raises(ValueError, match="differs from reducer artifact"):
        _diagnostic_manifest(
            signed_lock,
            signed_materializations=(signed_preflight, signed_e3a),
            signed_coverage=(signed_preflight_coverage, signed_e3a_coverage),
            e3a_staged_selection_artifacts=(foreign_artifact,),
            signed_e3a_staged_selections=(signed_selection,),
            signer=signer,
        )


def test_e1_manifest_uses_staged_e3a_and_tts_cal_sources_only() -> None:
    signer = _Signer()
    authority = _tts_authority()
    lock = replace(
        _protocol_lock(),
        tts_calibration_authority_sha256=authority.sha256,
    )
    lock_sha, lock_challenge, lock_attestation = signer.sign(lock)
    signed_lock = SignedProtocolLock(
        lock,
        lock_sha,
        lock_challenge,
        lock_attestation,
    )

    preflight = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    signed_preflight = SignedStageMaterializationReceipt(
        preflight,
        *signer.sign(preflight),
    )
    preflight_coverage = _complete_coverage(
        preflight,
        candidate_coverages=(_candidate_coverage(preflight),),
    )
    signed_preflight_coverage = SignedStageCoverageReceipt(
        preflight_coverage,
        *signer.sign(preflight_coverage),
    )

    e3a = _materialize_e3a_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_preflight_receipt_sha256=signed_preflight_coverage.sha256,
        workload_authority_sha256=lock.formal_workload_e3a_authorization_sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    signed_e3a = SignedStageMaterializationReceipt(e3a, *signer.sign(e3a))
    e3a_coverage = _complete_coverage(e3a)
    signed_e3a_coverage = SignedStageCoverageReceipt(
        e3a_coverage,
        *signer.sign(e3a_coverage),
    )
    e3a_artifact = _synthetic_e3a_artifact(
        lock=lock,
        materialization=e3a,
        coverage=e3a_coverage,
    )
    e3a_selection = build_e3a_staged_selection_receipt(e3a_artifact)
    signed_e3a_selection = SignedE3aStagedSelectionReceipt(
        e3a_selection,
        *signer.sign(e3a_selection),
    )

    tts_cal = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_e3a_receipt_sha256=signed_e3a_selection.sha256,
        calibration_authority_sha256=authority.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    signed_tts_cal = SignedStageMaterializationReceipt(
        tts_cal,
        *signer.sign(tts_cal),
    )
    tts_cal_coverage = _complete_coverage(tts_cal)
    signed_tts_cal_coverage = SignedStageCoverageReceipt(
        tts_cal_coverage,
        *signer.sign(tts_cal_coverage),
    )
    seal = object.__new__(TtsCalibrationSeal)
    selected_learning_rate = authority.learning_rates[0]
    selected_stride = authority.strides[0]
    for name, value in {
        "schema_version": 2,
        "authority_sha256": authority.sha256,
        "protocol_lock_sha256": lock.sha256,
        "materialization_receipt_sha256": tts_cal.sha256,
        "coverage_receipt_sha256": tts_cal_coverage.sha256,
        "reduction_receipt_sha256": _sha("tts-reduction"),
        "raw_manifest_sha256": _sha("tts-raw-manifest"),
        "tuning_window_sha256": authority.tuning_window_sha256,
        "selected_learning_rate": selected_learning_rate,
        "selected_stride": selected_stride,
        "selected_candidate_id": authority.candidate_id(
            learning_rate=selected_learning_rate,
            stride=selected_stride,
        ),
        "selected_pilot_run_binding_sha256s": tuple(
            _sha(f"tts-selected-pilot-{block}") for block in range(4)
        ),
        "selection_rule": "safety_first_then_maximize_slo_goodput",
        "result_class": "tuning_only_not_formal",
    }.items():
        object.__setattr__(seal, name, value)
    seal.__post_init__()
    signed_seal = SignedTtsCalibrationSeal(seal, *signer.sign(seal))

    e1 = _materialize_e1_first_slice_from_verified_decisions(
        protocol_lock_sha256=lock.sha256,
        tts_calibration_receipt_sha256=signed_tts_cal_coverage.sha256,
        signed_tts_calibration_seal_sha256=signed_seal.sha256,
        e3a_selection_sha256=signed_e3a_selection.sha256,
        frozen_tts_recipe_sha256=seal.selected_candidate_id,
        e1_recipe_anchor_authority_sha256=(lock.e1_recipe_anchor_authority_sha256),
        model=e3a_selection.model,
        matched_width=e3a_selection.matched_width,
        common_load=e3a_selection.common_load,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    signed_e1 = SignedStageMaterializationReceipt(e1, *signer.sign(e1))
    manifest = _diagnostic_manifest(
        signed_lock,
        signed_materializations=(
            signed_preflight,
            signed_e3a,
            signed_tts_cal,
            signed_e1,
        ),
        signed_coverage=(
            signed_preflight_coverage,
            signed_e3a_coverage,
            signed_tts_cal_coverage,
        ),
        e3a_staged_selection_artifacts=(e3a_artifact,),
        signed_e3a_staged_selections=(signed_e3a_selection,),
        signer=signer,
        tts_calibration_authorities=(authority,),
        signed_tts_calibration_seals=(signed_seal,),
    )
    assert tuple(row.stage for row in manifest.materializations) == (
        "preflight",
        "E3a",
        "TTS-Cal",
        "E1",
    )
    assert manifest.materializations[-1].expected_cell_count == 68
    assert {(row.stage, row.authority_kind) for row in manifest.source_authorities} == {
        ("E3a", "e3a_staged_selection"),
        ("TTS-Cal", "tts_calibration_seal"),
    }

    with pytest.raises(ValueError, match="staged source lineage"):
        _diagnostic_manifest(
            signed_lock,
            signed_materializations=(
                signed_preflight,
                signed_e3a,
                signed_tts_cal,
                SignedStageMaterializationReceipt(
                    replace(
                        e1,
                        upstream_receipt_sha256s=(
                            _sha("foreign-tts-coverage"),
                            signed_seal.sha256,
                            signed_e3a_selection.sha256,
                        ),
                    ),
                    *signer.sign(
                        replace(
                            e1,
                            upstream_receipt_sha256s=(
                                _sha("foreign-tts-coverage"),
                                signed_seal.sha256,
                                signed_e3a_selection.sha256,
                            ),
                        )
                    ),
                ),
            ),
            signed_coverage=(
                signed_preflight_coverage,
                signed_e3a_coverage,
                signed_tts_cal_coverage,
            ),
            e3a_staged_selection_artifacts=(e3a_artifact,),
            signed_e3a_staged_selections=(signed_e3a_selection,),
            signer=signer,
            tts_calibration_authorities=(authority,),
            signed_tts_calibration_seals=(signed_seal,),
        )
