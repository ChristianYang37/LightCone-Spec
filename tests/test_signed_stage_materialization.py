from __future__ import annotations

import base64
import hashlib
import inspect
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.experiments.downstream_stage_authority import (
    E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
    E5PowerAndAnchorReceipt,
    FormalFamilyPowerCommitment,
)
from lightcone_spec.experiments.formal_failure_execution import (
    _require_exact_e5_final_failure_materialization,
)
from lightcone_spec.experiments.formal_protocol import (
    BANNED_MODEL,
    E0_METHOD_ROLES,
    E6_MODELS,
    FORMAL_METHOD_ROLES,
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
)
from lightcone_spec.experiments.formal_registry import (
    _build_formal_registry_manifest_with_policy,
    _validate_candidate_coverage_replay_uniqueness,
    signed_tts_calibration_seal_from_dict,
    signed_tts_calibration_seal_to_dict,
    tts_calibration_authority_from_dict,
    tts_calibration_authority_to_dict,
)
from lightcone_spec.experiments.formal_slo_metrics import (
    FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
)
from lightcone_spec.experiments.planning import SealedE3aSelection
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    build_legacy_industrial_registry,
)
from lightcone_spec.experiments.selection_authority import (
    E3aSelectionReductionAuthority,
)
from lightcone_spec.experiments.stage_decisions import (
    E3aSelectionReceipt,
    SignedE3aSelectionReceipt,
    _require_signed_staged_source_registry,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_ALL_NA_MATERIALIZATION_RULE,
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
    E1A_VERIFICATION_MODES,
    E2_OPTIMIZERS,
    E2_SCHEDULES,
    E4_SCREEN_FACTOR_LEVELS,
    E5_FAILURES,
    E0CompatibilityDecision,
    E0CompatibilityReceipt,
    E2CandidateRecipe,
    E5AnchorSelectionReceipt,
    E5FailureDiagnosticAuthority,
    E5SelectedP99Anchor,
    FormalGpuHourAuthorityBlocked,
    GpuHourEstimate,
    MaterializedCell,
    PilotDurationObservation,
    PilotDurationReceipt,
    SignedE0CompatibilityReceipt,
    SignedE5AnchorSelectionReceipt,
    SignedPilotDurationReceipt,
    SignedStageCoverageReceipt,
    SignedStageMaterializationReceipt,
    StageCellDisposition,
    StageCoverageReceipt,
    StageMaterializationReceipt,
    _materialize_e0_from_signed_compatibility_diagnostic,
    _materialize_e1_first_slice_with_verified_policy,
    _materialize_e1a_diagnostic,
    _materialize_e2_round_from_verified_values,
    _materialize_e3a_diagnostic,
    _materialize_e3b_diagnostic,
    _materialize_e4_profiler_diagnostic,
    _materialize_e4_strength2_screen_diagnostic,
    _materialize_e4_winner_neighborhood_diagnostic,
    _materialize_e5_diagnostic,
    _materialize_e6_diagnostic,
    _materialize_tts_calibration_diagnostic,
    _reduce_gpu_hours_from_signed_pilots_diagnostic,
    _select_exact_final_prefix,
    default_e2_recipe_grid_authority,
    default_e5_failure_diagnostic_authority,
    e1_geometries,
    e1a_configurations,
    e2_candidate_recipes,
    e2_round_candidate_counts,
    e2_total_cell_count,
    materialize_e0_from_signed_compatibility,
    materialize_e1_first_slice,
    materialize_e1a,
    materialize_e2_round,
    materialize_e3a,
    materialize_e3b,
    materialize_e4_profiler,
    materialize_e4_strength2_screen,
    materialize_e4_winner_neighborhood,
    materialize_e5,
    materialize_e6,
    materialize_preflight,
    materialize_tts_calibration,
    reduce_gpu_hours_from_signed_pilots,
    validate_formal_registry_no_banned_models,
)
from lightcone_spec.experiments.statistics import PilotBlock, preregister_power_sizing
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _sign(payload: object, *, now_ns: int = 20_000_000_000):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_base64 = base64.b64encode(public_bytes).decode()
    public_sha256 = hashlib.sha256(public_bytes).hexdigest()
    policy = TrustedAttesterPolicy(
        policy_id="formal-stage-authority-v1",
        trusted_attesters=(("formal-stage-signer", "formal-stage-key", public_sha256),),
        public_keys=((public_sha256, public_base64),),
    )
    payload_sha256 = content_sha256(payload)
    challenge = AttestationChallenge.issue(
        challenge_id="formal-stage-challenge",
        subject_sha256=payload_sha256,
        lifetime_s=60,
        now_ns=now_ns,
    )
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=payload_sha256)
    )
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="formal-stage-signer",
        key_id="formal-stage-key",
        environment="release",
        public_key_base64=public_base64,
        challenge_sha256=challenge.sha256,
        payload_sha256=payload_sha256,
        signature_base64=base64.b64encode(signature).decode(),
    )
    return payload_sha256, challenge, attestation, policy, now_ns


def _unmeasured() -> GpuHourEstimate:
    return GpuHourEstimate.unmeasured()


def test_empty_e0_coverage_is_only_valid_for_signed_all_na_materialization() -> None:
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=_sha("all-na-lock"),
        upstream_receipt_sha256s=(_sha("all-na-e6"),),
        source_decision_sha256=_sha("all-na-compatibility"),
        materialization_rule=E0_ALL_NA_MATERIALIZATION_RULE,
        expected_cell_count=0,
        cells=(),
        gpu_hours=_unmeasured(),
    )
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="E0",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=(),
        tts_l0_candidate_state_coverages=(),
    )
    coverage.validate_against(materialization)

    foreign = replace(materialization, materialization_rule="caller_empty_e0")
    foreign_coverage = replace(
        coverage,
        materialization_receipt_sha256=foreign.sha256,
    )
    with pytest.raises(ValueError, match="all-N/A branch"):
        foreign_coverage.validate_against(foreign)
    with pytest.raises(ValueError, match="candidate-state evidence"):
        StageCoverageReceipt(
            schema_version=2,
            stage="E1",
            protocol_lock_sha256=_sha("empty-e1-lock"),
            materialization_receipt_sha256=_sha("empty-e1-materialization"),
            dispositions=(),
        )


def _pilot_observation(
    label: str,
    wave_index: int,
    gang_gpu_count: int,
    wall_time_ms: int,
) -> PilotDurationObservation:
    return PilotDurationObservation(
        cell_id=_sha(label),
        wave_index=wave_index,
        gang_gpu_count=gang_gpu_count,
        terminal_receipt_sha256=_sha(f"{label}-terminal"),
        schedule_assignment_sha256=_sha(f"{label}-schedule-assignment"),
        startup_ms=0,
        warmup_ms=0,
        arrival_window_ms=wall_time_ms,
        drain_ms=0,
        reset_ms=0,
        evidence_flush_ms=0,
        retry_ms=0,
        profile_ms=0,
        wall_time_ms=wall_time_ms,
    )


def _candidate_state_coverages(
    materialization,
) -> tuple[TtsL0CandidateStateCoverage, ...]:
    grouped: dict[str, dict[str, object]] = {}
    for cell in materialization.cells:
        if cell.method_role in {"TTS", "L0-naive"}:
            pair_id = dict(cell.dimensions)["tts_l0_pair_id"]
            grouped.setdefault(pair_id, {})[cell.method_role] = cell
    if materialization.stage == "preflight":
        exactness = next(
            cell
            for cell in materialization.cells
            if cell.task == "exactness_memory_telemetry_preflight"
        )
        grouped = {
            _sha("preflight-qualification-pair"): {
                "TTS": _sha("preflight-tts-fixture"),
                "L0-naive": _sha("preflight-l0-fixture"),
                "qualification": exactness.cell_id,
            }
        }
    rounds = (1, 2)

    def replay(
        role: str,
        source_round: int,
        *,
        cell_id: str,
        native_replay_pointer_sha256: str,
        pair_id: str,
        trainable_plan: str,
    ) -> CandidateStateReplay:
        return CandidateStateReplay(
            method_role=role,  # type: ignore[arg-type]
            cell_id=cell_id,
            run_id=f"{role}-{materialization.stage}-{pair_id[:12]}",
            native_replay_pointer_sha256=native_replay_pointer_sha256,
            source_round=source_round,
            source_version=source_round - 1,
            source_state_sha256=_sha(
                f"source-state-{materialization.stage}-{pair_id}-{source_round}"
            ),
            trainable_plan_sha256=trainable_plan,
            candidate_bytes_sha256=_sha(
                f"candidate-{materialization.stage}-{pair_id}-{source_round}"
            ),
            optimizer_state_bytes_sha256=_sha(
                f"optimizer-state-{materialization.stage}-{pair_id}-{source_round}"
            ),
            proposal_evidence_sha256=_sha(
                f"proposal-{materialization.stage}-{pair_id}-{source_round}"
            ),
            publication_policy=("fixed_barrier" if role == "TTS" else "first_ready"),
        )

    coverages = []
    for pair_id, cells in sorted(grouped.items()):
        trainable_plan = _sha(f"trainable-plan-{materialization.stage}-{pair_id}")
        tts_cell = cells["TTS"]
        l0_cell = cells["L0-naive"]
        tts_cell_id = tts_cell if type(tts_cell) is str else tts_cell.cell_id
        l0_cell_id = l0_cell if type(l0_cell) is str else l0_cell.cell_id
        tts_pointer = _sha(f"tts-pointer-{materialization.stage}-{pair_id}")
        l0_pointer = _sha(f"l0-pointer-{materialization.stage}-{pair_id}")
        coverages.append(
            TtsL0CandidateStateCoverage(
                schema_version=1,
                stage=materialization.stage,
                scope=(
                    "preflight_exactness_qualification"
                    if materialization.stage == "preflight"
                    else "materialized_pair"
                ),
                protocol_lock_sha256=materialization.protocol_lock_sha256,
                materialization_receipt_sha256=materialization.sha256,
                pair_id=pair_id,
                tts_cell_id=tts_cell_id,
                l0_naive_cell_id=l0_cell_id,
                tts_native_replay_pointer_sha256=tts_pointer,
                l0_naive_native_replay_pointer_sha256=l0_pointer,
                qualification_cell_id=cells.get("qualification"),
                source_round_plan_sha256=_sha(
                    f"source-round-plan-{materialization.stage}-{pair_id}"
                ),
                trainable_plan_sha256=trainable_plan,
                expected_source_rounds=rounds,
                tts_observations=tuple(
                    replay(
                        "TTS",
                        row,
                        cell_id=tts_cell_id,
                        native_replay_pointer_sha256=tts_pointer,
                        pair_id=pair_id,
                        trainable_plan=trainable_plan,
                    )
                    for row in rounds
                ),
                l0_naive_observations=tuple(
                    replay(
                        "L0-naive",
                        row,
                        cell_id=l0_cell_id,
                        native_replay_pointer_sha256=l0_pointer,
                        pair_id=pair_id,
                        trainable_plan=trainable_plan,
                    )
                    for row in rounds
                ),
                terminal_pairs=tuple(
                    CandidateStateTerminalPair(
                        source_round=row,
                        tts_cell_id=tts_cell_id,
                        l0_naive_cell_id=l0_cell_id,
                        tts_run_id=(f"TTS-{materialization.stage}-{pair_id[:12]}"),
                        l0_naive_run_id=(
                            f"L0-naive-{materialization.stage}-{pair_id[:12]}"
                        ),
                        tts_native_replay_pointer_sha256=tts_pointer,
                        l0_naive_native_replay_pointer_sha256=l0_pointer,
                        proposal_evidence_sha256=_sha(
                            f"proposal-{materialization.stage}-{pair_id}-{row}"
                        ),
                        tts_terminal_receipt_sha256=_sha(
                            f"tts-terminal-{materialization.stage}-{pair_id}"
                        ),
                        l0_naive_terminal_receipt_sha256=_sha(
                            f"l0-terminal-{materialization.stage}-{pair_id}"
                        ),
                    )
                    for row in rounds
                ),
            )
        )
    return tuple(coverages)


def _complete_stage_coverage(materialization) -> StageCoverageReceipt:
    dispositions = tuple(
        sorted(
            (
                StageCellDisposition(
                    stage=materialization.stage,
                    cell_id=cell.cell_id,
                    status="COMPLETE",
                    reason_code="terminal_complete",
                    terminal_receipt_sha256=_sha(
                        f"terminal-{materialization.stage}-{cell.cell_id}"
                    ),
                )
                for cell in materialization.cells
            ),
            key=lambda row: row.cell_id,
        )
    )
    return StageCoverageReceipt(
        schema_version=2,
        stage=materialization.stage,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=dispositions,
        tts_l0_candidate_state_coverages=_candidate_state_coverages(materialization),
    )


def _protocol_lock(*, registry_sha256: str, tts_authority_sha256: str) -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="lightcone-formal-test",
        code_git_head=_sha("git-head")[:40],
        code_git_tree=_sha("git-tree")[:40],
        patch_manifest_sha256=_sha("patch-manifest"),
        registry_sha256=registry_sha256,
        english_protocol_sha256=_sha("english-protocol"),
        chinese_protocol_sha256=_sha("chinese-protocol"),
        tts_calibration_authority_sha256=tts_authority_sha256,
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1-recipe-anchor-authority"),
        e2_recipe_grid_authority_sha256=_sha("e2-recipe-grid-authority"),
        formal_runtime_authority_manifest_sha256=_sha("formal-runtime"),
        offline_release_trust_root_sha256=_sha("release-root"),
        prepared_model_content_authorization_sha256=_sha("prepared-models"),
        formal_workload_e3a_authorization_sha256=_sha("e3a-workload-auth"),
        formal_workload_e0_authorization_sha256=_sha("e0-workload-auth"),
        burstgpt_shape_authorization_sha256=_sha("burstgpt-shape-auth"),
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
        authority_id="tts-arxiv-v2-numeric-calibration",
        primary_source_id=TTS_PRIMARY_SOURCE_ID,
        primary_source_version=TTS_PRIMARY_SOURCE_VERSION,
        paper_pdf_sha256=_sha("tts-paper-pdf"),
        paper_source_sha256=_sha("tts-paper-source"),
        tuning_window_sha256=_sha("tts-tuning-window"),
        trainable_plan_sha256=_sha("tts-trainable-plan"),
        drafter_native_loss_recipe_sha256=_sha("tts-native-loss"),
    )


def _e1_authority_inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Build signed inputs while stubbing only the heavy raw E3a file replay."""

    authority = _tts_authority()
    registry_sha256 = build_industrial_registry().sha256
    protocol_lock = _protocol_lock(
        registry_sha256=registry_sha256,
        tts_authority_sha256=authority.sha256,
    )
    e3a_materialization = _materialize_e3a_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_preflight_receipt_sha256=_sha("preflight"),
        workload_authority_sha256=_sha("e3a-workload"),
        gpu_hours=_unmeasured(),
    )
    e3a_coverage = _complete_stage_coverage(e3a_materialization)
    selection = SealedE3aSelection(
        schema_version=1,
        registry_sha256=registry_sha256,
        runtime_sha256=_sha("e3a-runtime"),
        split_sha256=_sha("e3a-split"),
        width=16,
        concurrency=8,
        reducer_evidence_sha256=_sha("e3a-reducer-evidence"),
    )
    reduction_authority = object.__new__(E3aSelectionReductionAuthority)
    reduction_sha256 = _sha("e3a-reduction-authority")
    monkeypatch.setattr(
        E3aSelectionReductionAuthority,
        "revalidate",
        lambda self: selection,
    )
    monkeypatch.setattr(
        E3aSelectionReductionAuthority,
        "sha256",
        property(lambda self: reduction_sha256),
    )
    e3a_receipt = E3aSelectionReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=registry_sha256,
        e3a_materialization_receipt_sha256=e3a_materialization.sha256,
        e3a_coverage_receipt_sha256=e3a_coverage.sha256,
        e3a_workload_authority_sha256=e3a_materialization.source_decision_sha256,
        reduction_authority_sha256=reduction_sha256,
        source_selection_sha256=selection.sha256,
        model="Qwen/Qwen3-8B",
        matched_width=selection.width,
        common_load=selection.concurrency,
    )
    e3a_sha, e3a_challenge, e3a_attestation, e3a_policy, now_ns = _sign(e3a_receipt)
    signed_e3a = SignedE3aSelectionReceipt(
        e3a_receipt,
        e3a_sha,
        e3a_challenge,
        e3a_attestation,
    )

    tts_materialization = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_e3a_receipt_sha256=e3a_materialization.sha256,
        calibration_authority_sha256=authority.sha256,
        gpu_hours=_unmeasured(),
    )
    tts_coverage = _complete_stage_coverage(tts_materialization)
    seal = object.__new__(TtsCalibrationSeal)
    for name, value in {
        "schema_version": 2,
        "authority_sha256": authority.sha256,
        "protocol_lock_sha256": protocol_lock.sha256,
        "materialization_receipt_sha256": tts_materialization.sha256,
        "coverage_receipt_sha256": tts_coverage.sha256,
        "reduction_receipt_sha256": _sha("tts-reduction"),
        "raw_manifest_sha256": _sha("tts-raw-manifest"),
        "tuning_window_sha256": authority.tuning_window_sha256,
        "selected_learning_rate": authority.learning_rates[0],
        "selected_stride": authority.strides[0],
        "selected_candidate_id": authority.candidate_ids[0],
        "selected_pilot_run_binding_sha256s": tuple(
            _sha(f"tts-pilot-{block}") for block in range(4)
        ),
        "selection_rule": authority.selection_rule,
        "result_class": authority.result_class,
    }.items():
        object.__setattr__(seal, name, value)
    seal.__post_init__()
    tts_sha, tts_challenge, tts_attestation, tts_policy, _ = _sign(seal, now_ns=now_ns)
    signed_tts = SignedTtsCalibrationSeal(
        seal,
        tts_sha,
        tts_challenge,
        tts_attestation,
    )
    return {
        "protocol_lock": protocol_lock,
        "tts_calibration_materialization": tts_materialization,
        "tts_calibration_coverage": tts_coverage,
        "signed_tts_calibration_seal": signed_tts,
        "tts_calibration_authority": authority,
        "tts_seal_policy": tts_policy,
        "expected_tts_seal_policy_sha256": tts_policy.sha256,
        "e3a_materialization": e3a_materialization,
        "e3a_coverage": e3a_coverage,
        "signed_e3a_selection": signed_e3a,
        "e3a_reduction_authority": reduction_authority,
        "e3a_selection_policy": e3a_policy,
        "expected_e3a_selection_policy_sha256": e3a_policy.sha256,
        "now_ns": now_ns,
        "gpu_hours": _unmeasured(),
    }


def test_prefix_materializers_cover_every_concrete_registered_cell() -> None:
    protocol_lock_sha256 = _sha("protocol")
    preflight = materialize_preflight(
        protocol_lock_sha256=protocol_lock_sha256,
        gpu_hours=_unmeasured(),
    )
    assert preflight.expected_cell_count == len(preflight.cells) == 10
    assert preflight.upstream_receipt_sha256s == ()

    e3a = _materialize_e3a_diagnostic(
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_preflight_receipt_sha256=preflight.sha256,
        workload_authority_sha256=_sha("e3a-workload"),
        gpu_hours=_unmeasured(),
    )
    assert e3a.expected_cell_count == len(e3a.cells) == 360
    assert e3a.upstream_receipt_sha256s == (preflight.sha256,)

    calibration = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=protocol_lock_sha256,
        upstream_e3a_receipt_sha256=e3a.sha256,
        calibration_authority_sha256=_sha("tts-calibration-authority"),
        gpu_hours=_unmeasured(),
    )
    assert calibration.expected_cell_count == len(calibration.cells) == 288
    assert len({cell.recipe_sha256 for cell in calibration.cells}) == 72
    assert {dict(cell.dimensions)["block"] for cell in calibration.cells} == {
        0,
        1,
        2,
        3,
    }
    assert all(
        dict(cell.dimensions)["pilot_phase"] == "excluded"
        and cell.publication_policy == "fixed_barrier"
        for cell in calibration.cells
    )


def test_e4_two_stage_design_and_profiler_are_separate_exact_receipts() -> None:
    common = {
        "protocol_lock_sha256": _sha("protocol"),
        "model": "Qwen/Qwen3-8B",
        "lightcone_recipe_sha256": _sha("lightcone"),
        "gpu_hours": _unmeasured(),
    }
    screen = _materialize_e4_strength2_screen_diagnostic(
        **common,
        upstream_e2_receipt_sha256=_sha("e2"),
        source_decision_sha256=_sha("e4-screen-decision"),
    )
    assert screen.expected_cell_count == len(screen.cells) == 48
    assert {dict(cell.dimensions)["screen_row"] for cell in screen.cells} == set(
        range(8)
    )
    assert all(
        cell.task == "mechanism_strength2_screen_headline" for cell in screen.cells
    )

    neighborhoods = tuple(
        (name, levels[0], levels[1]) for name, levels in E4_SCREEN_FACTOR_LEVELS
    )
    local = _materialize_e4_winner_neighborhood_diagnostic(
        **common,
        upstream_screen_receipt_sha256=screen.sha256,
        winner_decision_sha256=_sha("e4-winner"),
        factor_neighborhoods=neighborhoods,
    )
    assert local.expected_cell_count == len(local.cells) == 96
    assert local.upstream_receipt_sha256s == (screen.sha256,)
    assert all(
        cell.task == "winner_neighborhood_local_factorial_headline"
        for cell in local.cells
    )

    profiler = _materialize_e4_profiler_diagnostic(
        **common,
        upstream_local_receipt_sha256=local.sha256,
        source_decision_sha256=_sha("e4-profiler-decision"),
        selected_configuration_sha256=_sha("e4-selected"),
    )
    assert profiler.expected_cell_count == len(profiler.cells) == 3
    assert all(
        cell.task == "mechanism_profile_only"
        and cell.publication_policy == "diagnostic_only"
        for cell in profiler.cells
    )

    with pytest.raises(ValueError, match="two levels for four factors"):
        _materialize_e4_winner_neighborhood_diagnostic(
            **common,
            upstream_screen_receipt_sha256=screen.sha256,
            winner_decision_sha256=_sha("e4-malformed"),
            factor_neighborhoods=((),),  # type: ignore[arg-type]
        )


def test_e5_power_prefix_does_not_multiply_one_shot_failures() -> None:
    selected = E5SelectedP99Anchor(
        backend="DFLASH",
        topology="tp1_dp1",
        family_id="closed_loop_c8",
        minimum_completions=10_000,
    )
    selection = E5AnchorSelectionReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol"),
        upstream_e1a_receipt_sha256=_sha("e1a"),
        power_prefix_decision_sha256=_sha("power-prefix"),
        anchors=(selected,),
    )
    payload_sha256, challenge, attestation, policy, now_ns = _sign(selection)
    receipt = _materialize_e5_diagnostic(
        protocol_lock_sha256=_sha("protocol"),
        upstream_e1a_receipt_sha256=_sha("e1a"),
        power_prefix_decision_sha256=_sha("power-prefix"),
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        final_blocks=12,
        signed_anchor_selection=SignedE5AnchorSelectionReceipt(
            selection,
            payload_sha256,
            challenge,
            attestation,
        ),
        failure_diagnostic_authority=default_e5_failure_diagnostic_authority(),
        anchor_policy=policy,
        expected_anchor_policy_sha256=policy.sha256,
        now_ns=now_ns,
        gpu_hours=_unmeasured(),
    )
    headline = [
        cell for cell in receipt.cells if cell.task == "production_slo_power_prefix"
    ]
    failures = [
        cell for cell in receipt.cells if cell.task == "deterministic_failure_injection"
    ]
    assert receipt.expected_cell_count == len(receipt.cells) == 7_464
    assert len(headline) == 450 * 16
    assert len(failures) == 264 == len(E5_FAILURES) * 2 * 3 * 4
    assert all("block" not in dict(cell.dimensions) for cell in failures)
    selected_headline = [
        cell
        for cell in headline
        if dict(cell.dimensions).get("p99_anchor_id") == selected.anchor_id
    ]
    assert len(selected_headline) == 16
    assert all(
        dict(cell.dimensions)["p99_minimum_completions"] == 10_000
        for cell in selected_headline
    )
    selected_family = [
        cell
        for cell in headline
        if dict(cell.dimensions).get("p99_extension_anchor_id") == selected.anchor_id
    ]
    assert len(selected_family) == 5 * 16
    assert {cell.method_role for cell in selected_family} == set(FORMAL_METHOD_ROLES)
    assert all(
        dict(cell.dimensions)["p99_extension_minimum_completions"] == 10_000
        and dict(cell.dimensions)["p99_extension_offered_requests"] == 11_000
        for cell in selected_family
    )
    assert not any(
        cell.task == "selected_anchor_native_p99_itl" for cell in receipt.cells
    )
    final_only_cells = tuple(
        sorted(
            (
                cell
                for cell in receipt.cells
                if cell.task == "deterministic_failure_injection"
                or dict(cell.dimensions).get("block") in range(4, 16)
            ),
            key=lambda cell: cell.cell_id,
        )
    )
    final_only = replace(
        receipt,
        materialization_rule=(
            "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
        ),
        expected_cell_count=len(final_only_cells),
        cells=final_only_cells,
    )
    _authority, exact_failures = _require_exact_e5_final_failure_materialization(
        final_only,
        protocol_lock_sha256=_sha("protocol"),
    )
    assert len(exact_failures) == 264
    with pytest.raises(ValueError, match="final-only E5"):
        _require_exact_e5_final_failure_materialization(
            replace(
                final_only,
                materialization_rule=(
                    "e5_exact_450_headline_rows_x_4_excluded_pilot_blocks"
                ),
            ),
            protocol_lock_sha256=_sha("protocol"),
        )

    b24 = _materialize_e5_diagnostic(
        protocol_lock_sha256=_sha("protocol"),
        upstream_e1a_receipt_sha256=_sha("e1a"),
        power_prefix_decision_sha256=_sha("power-prefix"),
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        final_blocks=20,
        signed_anchor_selection=SignedE5AnchorSelectionReceipt(
            selection,
            payload_sha256,
            challenge,
            attestation,
        ),
        failure_diagnostic_authority=default_e5_failure_diagnostic_authority(),
        anchor_policy=policy,
        expected_anchor_policy_sha256=policy.sha256,
        now_ns=now_ns,
        gpu_hours=_unmeasured(),
    )
    assert b24.expected_cell_count == len(b24.cells) == 11_064

    authority = default_e5_failure_diagnostic_authority()
    assert isinstance(authority, E5FailureDiagnosticAuthority)
    with pytest.raises(ValueError, match="exactly 264"):
        replace(authority, members=authority.members[:-1])

    with pytest.raises(ValueError, match="at least 10,000 completions"):
        E5SelectedP99Anchor(
            backend="DFLASH",
            topology="tp1_dp1",
            family_id="closed_loop_c8",
            minimum_completions=9_999,
        )


def test_e5_power_and_anchor_receipt_requires_exact_six_anchor_pairs() -> None:
    power = preregister_power_sizing(
        tuple(
            PilotBlock(
                block_id=f"pilot-{index}",
                static_goodput=static,
                tts_goodput=tts,
                lightcone_goodput=lightcone,
            )
            for index, (static, tts, lightcone) in enumerate(
                (
                    (100.0, 100.0, 104.0),
                    (100.0, 100.0, 105.0),
                    (100.0, 100.0, 103.0),
                    (100.0, 100.0, 104.0),
                )
            )
        )
    )
    assert power.selected_final_blocks == 12
    anchors = tuple(
        sorted(
            (
                E5SelectedP99Anchor(
                    backend=backend,
                    topology=topology,
                    family_id=(
                        "closed_loop_c8"
                        if topology == "tp1_dp1"
                        else f"topology_cohort_{topology}_k1_uniform"
                    ),
                    minimum_completions=10_000,
                )
                for backend in ("DFLASH", "DSPARK")
                for topology in ("tp1_dp1", "tp2_dp1", "tp1_dp2")
            ),
            key=lambda row: row.anchor_id,
        )
    )
    receipt = E5PowerAndAnchorReceipt(
        schema_version=2,
        protocol_lock_sha256=_sha("e5-lock"),
        registry_sha256=_sha("e5-registry"),
        upstream_e1a_verification_sha256=_sha("e5-e1a"),
        pilot_materialization_receipt_sha256=_sha("e5-pilot-materialization"),
        pilot_coverage_receipt_sha256=_sha("e5-pilot-coverage"),
        evidence_manifest_sha256=_sha("e5-pilot-evidence"),
        inventory_sha256=_sha("e5-inventory"),
        protocol_sha256=E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("e5-tts"),
        dflash_lightcone_recipe_sha256=_sha("e5-dflash"),
        dspark_lightcone_recipe_sha256=_sha("e5-dspark"),
        family_power_commitments=tuple(
            sorted(
                (
                    FormalFamilyPowerCommitment(
                        schema_version=1,
                        stage="E5",
                        model="Qwen/Qwen3-8B",
                        task="production_slo_power_prefix",
                        family_dimensions=(("family", f"family-{index:03d}"),),
                        family_sha256=content_sha256(
                            {
                                "stage": "E5",
                                "model": "Qwen/Qwen3-8B",
                                "task": "production_slo_power_prefix",
                                "dimensions": [["family", f"family-{index:03d}"]],
                            }
                        ),
                        slo_goodput_protocol_sha256=(
                            FORMAL_SLO_GOODPUT_PROTOCOL_SHA256
                        ),
                        pilot_goodput_observation_sha256s=tuple(
                            sorted(
                                (
                                    block,
                                    role,
                                    _sha(f"e5-family-{index}-{block}-{role}"),
                                )
                                for block in range(4)
                                for role in ("Static", "TTS", "LightCone")
                            )
                        ),
                        power_sizing=power,
                    )
                    for index in range(90)
                ),
                key=lambda row: row.family_sha256,
            )
        ),
        selected_final_blocks=12,
        selected_final_prefix=tuple(range(4, 16)),
        p99_anchors=anchors,
    )
    assert len(receipt.p99_anchors) == 6
    with pytest.raises(ValueError, match="exact six anchors"):
        replace(receipt, p99_anchors=receipt.p99_anchors[:-1])


def test_e1_materializes_exact_68_concrete_rows_with_frozen_anchor_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _e1_authority_inputs(monkeypatch)
    receipt = _materialize_e1_first_slice_with_verified_policy(**inputs)
    authority = inputs["tts_calibration_authority"]
    signed_tts = inputs["signed_tts_calibration_seal"]
    assert isinstance(authority, TtsCalibrationAuthority)
    assert isinstance(signed_tts, SignedTtsCalibrationSeal)
    assert (
        tts_calibration_authority_from_dict(
            tts_calibration_authority_to_dict(authority)
        )
        == authority
    )
    assert (
        signed_tts_calibration_seal_from_dict(
            signed_tts_calibration_seal_to_dict(signed_tts)
        )
        == signed_tts
    )
    frozen = authority.candidate_ids[0]

    assert receipt.expected_cell_count == len(receipt.cells) == 68
    assert len(e1_geometries()) == 32
    assert sum(row.method_role == "LightCone-candidate" for row in receipt.cells) == 64
    anchors = [row for row in receipt.cells if row.method_role != "LightCone-candidate"]
    assert {row.method_role for row in anchors} == set(FORMAL_METHOD_ROLES[:-1])
    tts = next(row for row in anchors if row.method_role == "TTS")
    naive = next(row for row in anchors if row.method_role == "L0-naive")
    assert tts.recipe_sha256 == naive.recipe_sha256 == frozen
    assert tts.publication_policy == "fixed_barrier"
    assert naive.publication_policy == "first_ready"
    assert all("template" not in str(row.dimensions).lower() for row in receipt.cells)
    assert receipt.upstream_receipt_sha256s == (
        inputs["tts_calibration_materialization"].sha256,
        signed_tts.sha256,
        inputs["signed_e3a_selection"].sha256,
    )

    payload_sha256, challenge, attestation, policy, now_ns = _sign(signed_tts.payload)
    resealed = SignedTtsCalibrationSeal(
        signed_tts.payload,
        payload_sha256,
        challenge,
        attestation,
    )
    resealed_receipt = _materialize_e1_first_slice_with_verified_policy(
        **{
            **inputs,
            "signed_tts_calibration_seal": resealed,
            "tts_seal_policy": policy,
            "expected_tts_seal_policy_sha256": policy.sha256,
            "now_ns": now_ns,
        }
    )
    assert resealed.sha256 != signed_tts.sha256
    assert resealed_receipt.sha256 != receipt.sha256
    assert resealed_receipt.upstream_receipt_sha256s[1] == resealed.sha256


def test_formal_e1_rejects_caller_supplied_independent_signing_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _e1_authority_inputs(monkeypatch)
    formal_parameters = inspect.signature(materialize_e1_first_slice).parameters
    assert {
        "signed_e3a_selection",
        "e3a_reduction_authority",
        "tts_seal_policy",
        "e3a_selection_policy",
        "expected_tts_seal_policy_sha256",
        "expected_e3a_selection_policy_sha256",
    }.isdisjoint(formal_parameters)
    assert set(formal_parameters) == {
        "registry_verification_receipt",
        "protocol_lock",
        "tts_calibration_materialization",
        "tts_calibration_coverage",
        "e3a_materialization",
        "e3a_coverage",
        "now_ns",
        "gpu_hours",
    }
    with pytest.raises(TypeError, match="durable registry verification receipt"):
        materialize_e1_first_slice(
            registry_verification_receipt=object(),
            protocol_lock=inputs["protocol_lock"],
            tts_calibration_materialization=inputs["tts_calibration_materialization"],
            tts_calibration_coverage=inputs["tts_calibration_coverage"],
            e3a_materialization=inputs["e3a_materialization"],
            e3a_coverage=inputs["e3a_coverage"],
            now_ns=inputs["now_ns"],
            gpu_hours=inputs["gpu_hours"],
        )


def test_formal_e2_round_zero_rejects_legacy_e3a_authority_inputs() -> None:
    parameters = inspect.signature(materialize_e2_round).parameters
    assert set(parameters) == {
        "registry_verification_receipt",
        "protocol_lock",
        "signed_e1_selection",
        "e1_materialization",
        "e1_coverage",
        "pareto_evidence_manifest",
        "execution_bindings",
        "now_ns",
        "gpu_hours",
    }
    assert {
        "signed_e3a_selection",
        "e3a_materialization",
        "e3a_coverage",
        "e3a_reduction_authority",
        "candidate_recipes",
        "source_selection_sha256",
    }.isdisjoint(parameters)
    with pytest.raises(TypeError, match="exact ProtocolLock"):
        materialize_e2_round(
            registry_verification_receipt=object(),
            protocol_lock=object(),  # type: ignore[arg-type]
            signed_e1_selection=object(),
            e1_materialization=object(),  # type: ignore[arg-type]
            e1_coverage=object(),  # type: ignore[arg-type]
            pareto_evidence_manifest=object(),
            execution_bindings=(),
            now_ns=1,
            gpu_hours=_unmeasured(),
        )


def test_e3a_and_tts_cal_public_materializers_reject_scalar_authority() -> None:
    e3a_parameters = inspect.signature(materialize_e3a).parameters
    assert {
        "registry_verification_receipt",
        "protocol_lock",
        "preflight_materialization",
        "preflight_coverage",
        "now_ns",
        "gpu_hours",
    } == set(e3a_parameters)
    assert {
        "protocol_lock_sha256",
        "upstream_preflight_receipt_sha256",
        "workload_authority_sha256",
    }.isdisjoint(e3a_parameters)
    with pytest.raises(TypeError, match="durable registry verification receipt"):
        materialize_e3a(
            registry_verification_receipt=object(),
            protocol_lock=object(),  # type: ignore[arg-type]
            preflight_materialization=object(),  # type: ignore[arg-type]
            preflight_coverage=object(),  # type: ignore[arg-type]
            now_ns=1,
            gpu_hours=_unmeasured(),
        )

    calibration_parameters = inspect.signature(materialize_tts_calibration).parameters
    assert {
        "registry_verification_receipt",
        "protocol_lock",
        "tts_calibration_authority",
        "e3a_materialization",
        "e3a_coverage",
        "now_ns",
        "gpu_hours",
    } == set(calibration_parameters)
    assert {
        "protocol_lock_sha256",
        "upstream_e3a_receipt_sha256",
        "calibration_authority_sha256",
    }.isdisjoint(calibration_parameters)
    with pytest.raises(TypeError, match="durable registry verification receipt"):
        materialize_tts_calibration(
            registry_verification_receipt=object(),
            protocol_lock=object(),  # type: ignore[arg-type]
            tts_calibration_authority=object(),  # type: ignore[arg-type]
            e3a_materialization=object(),  # type: ignore[arg-type]
            e3a_coverage=object(),  # type: ignore[arg-type]
            now_ns=1,
            gpu_hours=_unmeasured(),
        )


def test_e1_rejects_signed_scalar_override_and_noncomplete_tts_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _e1_authority_inputs(monkeypatch)
    signed_e3a = inputs["signed_e3a_selection"]
    assert isinstance(signed_e3a, SignedE3aSelectionReceipt)
    forged_payload = replace(signed_e3a.payload, matched_width=32)
    payload_sha256, challenge, attestation, policy, now_ns = _sign(forged_payload)
    with pytest.raises(ValueError, match="differs from reopened evidence"):
        _materialize_e1_first_slice_with_verified_policy(
            **{
                **inputs,
                "signed_e3a_selection": SignedE3aSelectionReceipt(
                    forged_payload,
                    payload_sha256,
                    challenge,
                    attestation,
                ),
                "e3a_selection_policy": policy,
                "expected_e3a_selection_policy_sha256": policy.sha256,
                "now_ns": now_ns,
            }
        )

    coverage = inputs["tts_calibration_coverage"]
    assert isinstance(coverage, StageCoverageReceipt)
    first = coverage.dispositions[0]
    blocked = replace(
        coverage,
        dispositions=(
            replace(
                first,
                status="BLOCKED",
                reason_code="terminal_failed",
                terminal_receipt_sha256=None,
            ),
            *coverage.dispositions[1:],
        ),
    )
    with pytest.raises(ValueError, match="all-COMPLETE TTS-Cal coverage"):
        _materialize_e1_first_slice_with_verified_policy(
            **{**inputs, "tts_calibration_coverage": blocked}
        )


def test_formal_manifest_rejects_legacy_e3a_lineage_even_with_signed_tts_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _e1_authority_inputs(monkeypatch)
    protocol_lock = inputs["protocol_lock"]
    authority = inputs["tts_calibration_authority"]
    reduction_authority = inputs["e3a_reduction_authority"]
    assert isinstance(protocol_lock, ProtocolLock)
    assert isinstance(authority, TtsCalibrationAuthority)
    assert isinstance(reduction_authority, E3aSelectionReductionAuthority)

    preflight = materialize_preflight(
        protocol_lock_sha256=protocol_lock.sha256,
        gpu_hours=_unmeasured(),
    )
    preflight_coverage = _complete_stage_coverage(preflight)
    e3a = _materialize_e3a_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_preflight_receipt_sha256=preflight.sha256,
        workload_authority_sha256=_sha("e3a-workload"),
        gpu_hours=_unmeasured(),
    )
    e3a_coverage = _complete_stage_coverage(e3a)
    selection = reduction_authority.revalidate()
    e3a_selection = E3aSelectionReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        e3a_materialization_receipt_sha256=e3a.sha256,
        e3a_coverage_receipt_sha256=e3a_coverage.sha256,
        e3a_workload_authority_sha256=e3a.source_decision_sha256,
        reduction_authority_sha256=reduction_authority.sha256,
        source_selection_sha256=selection.sha256,
        model="Qwen/Qwen3-8B",
        matched_width=selection.width,
        common_load=selection.concurrency,
    )
    tts_cal = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_e3a_receipt_sha256=e3a.sha256,
        calibration_authority_sha256=authority.sha256,
        gpu_hours=_unmeasured(),
    )
    tts_cal_coverage = _complete_stage_coverage(tts_cal)
    old_signed_seal = inputs["signed_tts_calibration_seal"]
    assert isinstance(old_signed_seal, SignedTtsCalibrationSeal)
    seal = object.__new__(TtsCalibrationSeal)
    for name, value in {
        **old_signed_seal.payload.__dict__,
        "materialization_receipt_sha256": tts_cal.sha256,
        "coverage_receipt_sha256": tts_cal_coverage.sha256,
    }.items():
        object.__setattr__(seal, name, value)
    seal.__post_init__()

    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_base64 = base64.b64encode(public_bytes).decode()
    public_sha256 = hashlib.sha256(public_bytes).hexdigest()
    policy = TrustedAttesterPolicy(
        policy_id="formal-manifest-tts-lineage-v1",
        trusted_attesters=(("manifest-signer", "manifest-key", public_sha256),),
        public_keys=((public_sha256, public_base64),),
    )
    now_ns = 30_000_000_000

    def sign_shared(payload: object):
        payload_sha256 = content_sha256(payload)
        challenge = AttestationChallenge.issue(
            challenge_id="formal-manifest-tts-lineage",
            subject_sha256=payload_sha256,
            lifetime_s=60,
            now_ns=now_ns,
        )
        attestation = SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="manifest-signer",
            key_id="manifest-key",
            environment="release",
            public_key_base64=public_base64,
            challenge_sha256=challenge.sha256,
            payload_sha256=payload_sha256,
            signature_base64=base64.b64encode(
                private_key.sign(
                    attestation_message(
                        challenge,
                        payload_sha256=payload_sha256,
                    )
                )
            ).decode(),
        )
        return payload_sha256, challenge, attestation

    e3a_sha, e3a_challenge, e3a_attestation = sign_shared(e3a_selection)
    signed_e3a = SignedE3aSelectionReceipt(
        e3a_selection,
        e3a_sha,
        e3a_challenge,
        e3a_attestation,
    )
    seal_sha, seal_challenge, seal_attestation = sign_shared(seal)
    signed_seal = SignedTtsCalibrationSeal(
        seal,
        seal_sha,
        seal_challenge,
        seal_attestation,
    )
    e1 = _materialize_e1_first_slice_with_verified_policy(
        protocol_lock=protocol_lock,
        tts_calibration_materialization=tts_cal,
        tts_calibration_coverage=tts_cal_coverage,
        signed_tts_calibration_seal=signed_seal,
        tts_calibration_authority=authority,
        tts_seal_policy=policy,
        expected_tts_seal_policy_sha256=policy.sha256,
        e3a_materialization=e3a,
        e3a_coverage=e3a_coverage,
        signed_e3a_selection=signed_e3a,
        e3a_reduction_authority=reduction_authority,
        e3a_selection_policy=policy,
        expected_e3a_selection_policy_sha256=policy.sha256,
        now_ns=now_ns,
        gpu_hours=_unmeasured(),
    )

    lock_parts = sign_shared(protocol_lock)
    signed_lock = SignedProtocolLock(protocol_lock, *lock_parts)

    def signed_materialization(value):
        return SignedStageMaterializationReceipt(value, *sign_shared(value))

    def signed_coverage(value):
        return SignedStageCoverageReceipt(value, *sign_shared(value))

    signed_materializations = tuple(
        signed_materialization(row) for row in (preflight, e3a, tts_cal, e1)
    )
    signed_coverages = tuple(
        signed_coverage(row)
        for row in (preflight_coverage, e3a_coverage, tts_cal_coverage)
    )
    with pytest.raises(ValueError, match="signed preflight/workload lineage"):
        _build_formal_registry_manifest_with_policy(
            signed_lock,
            signed_materializations=signed_materializations,
            signed_coverage=signed_coverages,
            tts_calibration_authorities=(authority,),
            signed_tts_calibration_seals=(signed_seal,),
            policy=policy,
            expected_policy_sha256=policy.sha256,
            inventory_sha256=_sha("inventory"),
            deployment_policy_authorization_sha256=_sha("deployment"),
            control_lineage_sha256=_sha("control-lineage"),
            control_envelope_sha256s=(_sha("control-envelope"),),
            challenge_reservation_sha256=_sha("reservation"),
            now_ns=now_ns,
        )


def _e2_family_preserving_selection(
    rows: tuple[E2CandidateRecipe, ...], count: int
) -> tuple[E2CandidateRecipe, ...]:
    by_family: dict[tuple[str, str], list[E2CandidateRecipe]] = {}
    for row in rows:
        by_family.setdefault((row.optimizer, row.schedule), []).append(row)
    selected = [
        min(by_family[family], key=lambda row: row.sha256)
        for family in sorted(by_family)
    ]
    selected_ids = {row.sha256 for row in selected}
    selected.extend(
        row
        for row in sorted(rows, key=lambda row: row.sha256)
        if row.sha256 not in selected_ids
    )
    return tuple(selected[:count])


def test_legacy_eager_registry_cannot_authorize_staged_e1_or_e2_selection() -> None:
    legacy = build_legacy_industrial_registry()
    assert legacy.materialization_mode == "legacy_diagnostic"
    with pytest.raises(ValueError, match="diagnostic and non-authorizing"):
        _require_signed_staged_source_registry(
            legacy,
            expected_registry_sha256=legacy.sha256,
            stage="E1 Pareto",
        )


def test_e2_grid_and_successive_halving_counts_are_exact() -> None:
    grid = default_e2_recipe_grid_authority()
    geometry = (e1_geometries()[0],)
    universe = e2_candidate_recipes(geometry, grid=grid)
    assert len(universe) == 7 * 3 * 5 == 105
    assert e2_round_candidate_counts(1) == (105, 27, 21, 21)
    assert e2_total_cell_count(1) == 190
    assert e2_round_candidate_counts(32) == (3360, 840, 210, 53)
    assert e2_total_cell_count(32) == 4479

    round_zero = _materialize_e2_round_from_verified_values(
        protocol_lock_sha256=_sha("protocol"),
        upstream_receipt_sha256=_sha("e1-receipt"),
        source_selection_sha256=_sha("e1-survivors"),
        grid=grid,
        geometries=geometry,
        round_index=0,
        model="Qwen/Qwen3-8B",
        matched_width=16,
        common_load=8,
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        candidate_recipes=None,
        prior_round_materialization=None,
        gpu_hours=_unmeasured(),
    )
    assert len(round_zero.cells) == 105 + 4
    assert (
        sum(row.method_role == "LightCone-candidate" for row in round_zero.cells) == 105
    )
    round_zero_coverage = _complete_stage_coverage(round_zero)
    round_zero_coverage.validate_against(round_zero)
    assert len(round_zero_coverage.tts_l0_candidate_state_coverages) == 1

    round_one_candidates = _e2_family_preserving_selection(universe, 27)
    round_one = _materialize_e2_round_from_verified_values(
        protocol_lock_sha256=_sha("protocol"),
        upstream_receipt_sha256=round_zero.sha256,
        source_selection_sha256=_sha("round-one-survivors"),
        grid=grid,
        geometries=geometry,
        round_index=1,
        model="Qwen/Qwen3-8B",
        matched_width=16,
        common_load=8,
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        candidate_recipes=round_one_candidates,
        prior_round_materialization=round_zero,
        gpu_hours=_unmeasured(),
    )
    assert len(round_one.cells) == 27 + 4
    round_one_coverage = _complete_stage_coverage(round_one)
    round_one_coverage.validate_against(round_one)
    zero_evidence = round_zero_coverage.tts_l0_candidate_state_coverages[0]
    one_evidence = round_one_coverage.tts_l0_candidate_state_coverages[0]
    assert zero_evidence.pair_id != one_evidence.pair_id
    assert {
        row.run_id
        for row in (
            *zero_evidence.tts_observations,
            *zero_evidence.l0_naive_observations,
        )
    }.isdisjoint(
        {
            row.run_id
            for row in (
                *one_evidence.tts_observations,
                *one_evidence.l0_naive_observations,
            )
        }
    )
    _validate_candidate_coverage_replay_uniqueness(
        (round_zero_coverage, round_one_coverage)
    )
    assert {
        row.proposal_evidence_sha256 for row in zero_evidence.tts_observations
    }.isdisjoint(
        {row.proposal_evidence_sha256 for row in one_evidence.tts_observations}
    )
    assert {
        digest
        for row in zero_evidence.terminal_pairs
        for digest in (
            row.tts_terminal_receipt_sha256,
            row.l0_naive_terminal_receipt_sha256,
        )
    }.isdisjoint(
        {
            digest
            for row in one_evidence.terminal_pairs
            for digest in (
                row.tts_terminal_receipt_sha256,
                row.l0_naive_terminal_receipt_sha256,
            )
        }
    )
    reused_run = replace(
        one_evidence,
        tts_observations=tuple(
            replace(
                observation,
                run_id=zero_evidence.tts_observations[0].run_id,
            )
            for observation in one_evidence.tts_observations
        ),
        terminal_pairs=tuple(
            replace(
                terminal,
                tts_run_id=zero_evidence.tts_observations[0].run_id,
            )
            for terminal in one_evidence.terminal_pairs
        ),
    )
    with pytest.raises(ValueError, match="run identity"):
        _validate_candidate_coverage_replay_uniqueness(
            (
                round_zero_coverage,
                replace(
                    round_one_coverage,
                    tts_l0_candidate_state_coverages=(reused_run,),
                ),
            )
        )
    reused_proposal = replace(
        one_evidence,
        tts_observations=(
            replace(
                one_evidence.tts_observations[0],
                proposal_evidence_sha256=(
                    zero_evidence.tts_observations[0].proposal_evidence_sha256
                ),
            ),
            *one_evidence.tts_observations[1:],
        ),
        l0_naive_observations=(
            replace(
                one_evidence.l0_naive_observations[0],
                proposal_evidence_sha256=(
                    zero_evidence.tts_observations[0].proposal_evidence_sha256
                ),
            ),
            *one_evidence.l0_naive_observations[1:],
        ),
        terminal_pairs=(
            replace(
                one_evidence.terminal_pairs[0],
                proposal_evidence_sha256=(
                    zero_evidence.tts_observations[0].proposal_evidence_sha256
                ),
            ),
            *one_evidence.terminal_pairs[1:],
        ),
    )
    with pytest.raises(ValueError, match="proposal evidence"):
        _validate_candidate_coverage_replay_uniqueness(
            (
                round_zero_coverage,
                replace(
                    round_one_coverage,
                    tts_l0_candidate_state_coverages=(reused_proposal,),
                ),
            )
        )
    reused_terminal = replace(
        one_evidence,
        terminal_pairs=tuple(
            replace(
                terminal,
                tts_terminal_receipt_sha256=(
                    zero_evidence.terminal_pairs[0].tts_terminal_receipt_sha256
                ),
            )
            for terminal in one_evidence.terminal_pairs
        ),
    )
    with pytest.raises(ValueError, match="terminal receipt"):
        _validate_candidate_coverage_replay_uniqueness(
            (
                round_zero_coverage,
                replace(
                    round_one_coverage,
                    tts_l0_candidate_state_coverages=(reused_terminal,),
                ),
            )
        )
    with pytest.raises(ValueError, match="stage coverage identity"):
        replace(
            round_one_coverage,
            tts_l0_candidate_state_coverages=(zero_evidence,),
        ).validate_against(round_one)
    with pytest.raises(ValueError, match="wrong count"):
        _materialize_e2_round_from_verified_values(
            protocol_lock_sha256=_sha("protocol"),
            upstream_receipt_sha256=round_zero.sha256,
            source_selection_sha256=_sha("wrong-survivors"),
            grid=grid,
            geometries=geometry,
            round_index=1,
            model="Qwen/Qwen3-8B",
            matched_width=16,
            common_load=8,
            frozen_tts_recipe_sha256=_sha("frozen-tts"),
            candidate_recipes=universe[:26],
            prior_round_materialization=round_zero,
            gpu_hours=_unmeasured(),
        )
    missing_family = (
        round_one_candidates[0].optimizer,
        round_one_candidates[0].schedule,
    )
    without_family = tuple(
        row for row in universe if (row.optimizer, row.schedule) != missing_family
    )
    with pytest.raises(ValueError, match="every optimizer/schedule family"):
        _materialize_e2_round_from_verified_values(
            protocol_lock_sha256=_sha("protocol"),
            upstream_receipt_sha256=round_zero.sha256,
            source_selection_sha256=_sha("missing-family"),
            grid=grid,
            geometries=geometry,
            round_index=1,
            model="Qwen/Qwen3-8B",
            matched_width=16,
            common_load=8,
            frozen_tts_recipe_sha256=_sha("frozen-tts"),
            candidate_recipes=without_family[:27],
            prior_round_materialization=round_zero,
            gpu_hours=_unmeasured(),
        )
    with pytest.raises(ValueError, match="re-entered"):
        _materialize_e2_round_from_verified_values(
            protocol_lock_sha256=_sha("protocol"),
            upstream_receipt_sha256=round_one.sha256,
            source_selection_sha256=_sha("round-two-reentry"),
            grid=grid,
            geometries=geometry,
            round_index=2,
            model="Qwen/Qwen3-8B",
            matched_width=16,
            common_load=8,
            frozen_tts_recipe_sha256=_sha("frozen-tts"),
            candidate_recipes=universe[27:48],
            prior_round_materialization=round_one,
            gpu_hours=_unmeasured(),
        )


def test_e2_family_floor_rejects_loss_for_32_geometry_grid() -> None:
    grid = default_e2_recipe_grid_authority()
    geometries = e1_geometries()
    universe = e2_candidate_recipes(geometries, grid=grid)
    round_zero = _materialize_e2_round_from_verified_values(
        protocol_lock_sha256=_sha("protocol"),
        upstream_receipt_sha256=_sha("e1-receipt"),
        source_selection_sha256=_sha("e1-survivors"),
        grid=grid,
        geometries=geometries,
        round_index=0,
        model="Qwen/Qwen3-8B",
        matched_width=16,
        common_load=8,
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        candidate_recipes=None,
        prior_round_materialization=None,
        gpu_hours=_unmeasured(),
    )
    missing_family = (E2_OPTIMIZERS[0], E2_SCHEDULES[0])
    invalid = tuple(
        row for row in universe if (row.optimizer, row.schedule) != missing_family
    )[: e2_round_candidate_counts(32)[1]]
    assert len(invalid) == 840
    with pytest.raises(ValueError, match="every optimizer/schedule family"):
        _materialize_e2_round_from_verified_values(
            protocol_lock_sha256=_sha("protocol"),
            upstream_receipt_sha256=round_zero.sha256,
            source_selection_sha256=_sha("missing-family-32"),
            grid=grid,
            geometries=geometries,
            round_index=1,
            model="Qwen/Qwen3-8B",
            matched_width=16,
            common_load=8,
            frozen_tts_recipe_sha256=_sha("frozen-tts"),
            candidate_recipes=invalid,
            prior_round_materialization=round_zero,
            gpu_hours=_unmeasured(),
        )


def test_e3b_e1a_and_e6_cardinalities_match_signed_plan() -> None:
    common = {
        "protocol_lock_sha256": _sha("protocol"),
        "upstream_receipt_sha256": _sha("upstream"),
        "source_decision_sha256": _sha("decision"),
        "gpu_hours": _unmeasured(),
    }
    e3b = _materialize_e3b_diagnostic(
        **common,
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        final_blocks=12,
    )
    assert len(e3b.cells) == 480 * (4 + 12) == 7680
    assert {row.method_role for row in e3b.cells} == set(FORMAL_METHOD_ROLES)

    e1a = _materialize_e1a_diagnostic(
        **common,
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
    )
    assert len(e1a_configurations()) == 58
    assert len(e1a.cells) == 58 * len(E1A_VERIFICATION_MODES) == 116

    e6 = _materialize_e6_diagnostic(
        **common,
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        final_blocks=12,
    )
    assert len(e6.cells) == 2 + 60 * (4 + 12) == 962
    assert {row.model for row in e6.cells} == set(E6_MODELS)
    assert all(BANNED_MODEL not in row.model for row in e6.cells)


def test_powered_final_projection_excludes_all_four_tuning_blocks() -> None:
    fixture = _materialize_e3b_diagnostic(
        protocol_lock_sha256=_sha("protocol"),
        upstream_receipt_sha256=_sha("upstream"),
        source_decision_sha256=_sha("signed-power-prefix"),
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        final_blocks=12,
        gpu_hours=_unmeasured(),
    )
    final = _select_exact_final_prefix(
        fixture.cells,
        selected_final_prefix=tuple(range(4, 16)),
        expected_cells_per_block=480,
    )
    assert len(final) == 480 * 12
    assert {dict(cell.dimensions)["block"] for cell in final} == set(range(4, 16))
    assert all(dict(cell.dimensions)["block_phase"] == "final" for cell in final)

    with pytest.raises(ValueError, match="exact powered"):
        _select_exact_final_prefix(
            fixture.cells,
            selected_final_prefix=tuple(range(3, 15)),
            expected_cells_per_block=480,
        )


def _compatibility_receipt(*, valid_count: int) -> E0CompatibilityReceipt:
    rows = []
    index = 0
    for model in E0_MODELS:
        for backend in E0_BACKENDS:
            for task in E0_TASKS:
                rows.append(
                    E0CompatibilityDecision(
                        model=model,
                        backend=backend,
                        task=task,
                        disposition="VALID" if index < valid_count else "N/A",
                        reason_code="compatible"
                        if index < valid_count
                        else "not_applicable",
                        interface_sha256=_sha(f"interface-{model}-{backend}-{task}"),
                        task_native_workload_sha256=_sha(f"workload-{task}"),
                    )
                )
                index += 1
    return E0CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol"),
        upstream_e6_receipt_sha256=_sha("e6-receipt"),
        decisions=tuple(sorted(rows, key=lambda row: row.decision_id)),
    )


def test_e0_requires_signed_108_decisions_then_materializes_16vb() -> None:
    compatibility = _compatibility_receipt(valid_count=2)
    assert len(compatibility.decisions) == 108
    assert compatibility.valid_count == 2
    payload_sha256, challenge, attestation, policy, now_ns = _sign(compatibility)
    signed = SignedE0CompatibilityReceipt(
        compatibility,
        payload_sha256,
        challenge,
        attestation,
    )
    online_spec_recipes = tuple(
        (decision.decision_id, role, _sha(f"{decision.decision_id}-{role}-recipe"))
        for decision in compatibility.decisions
        if decision.disposition == "VALID"
        for role in E0_METHOD_ROLES[-3:]
    )
    e0 = _materialize_e0_from_signed_compatibility_diagnostic(
        signed,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
        source_decision_sha256=signed.sha256,
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        online_spec_recipe_sha256s=online_spec_recipes,
        final_blocks=12,
        gpu_hours=_unmeasured(),
    )
    block_count = 4 + 12
    assert len(e0.cells) == 16 * compatibility.valid_count * block_count
    assert {row.method_role for row in e0.cells} == set(E0_METHOD_ROLES)
    assert all(
        "context" not in dict(row.dimensions)
        and "task_native_workload_sha256" in dict(row.dimensions)
        for row in e0.cells
    )
    with pytest.raises(ValueError, match="independent OnlineSPEC recipe"):
        _materialize_e0_from_signed_compatibility_diagnostic(
            signed,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
            source_decision_sha256=signed.sha256,
            frozen_tts_recipe_sha256=_sha("frozen-tts"),
            lightcone_recipe_sha256=_sha("lightcone"),
            online_spec_recipe_sha256s=online_spec_recipes[:-1],
            final_blocks=12,
            gpu_hours=_unmeasured(),
        )

    no_valid = _compatibility_receipt(valid_count=0)
    n_sha, n_challenge, n_attestation, n_policy, n_now = _sign(no_valid)
    signed_no_valid = SignedE0CompatibilityReceipt(
        no_valid,
        n_sha,
        n_challenge,
        n_attestation,
    )
    empty = _materialize_e0_from_signed_compatibility_diagnostic(
        signed_no_valid,
        policy=n_policy,
        expected_policy_sha256=n_policy.sha256,
        now_ns=n_now,
        source_decision_sha256=signed_no_valid.sha256,
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        online_spec_recipe_sha256s=(),
        final_blocks=12,
        gpu_hours=_unmeasured(),
    )
    assert empty.expected_cell_count == 0
    assert empty.cells == ()


def test_materialization_and_coverage_signatures_require_exact_full_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization = _materialize_e1_first_slice_with_verified_policy(
        **_e1_authority_inputs(monkeypatch)
    )
    payload_sha256, challenge, attestation, policy, now_ns = _sign(materialization)
    signed_materialization = SignedStageMaterializationReceipt(
        materialization,
        payload_sha256,
        challenge,
        attestation,
    )
    assert (
        signed_materialization.verify(
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
        is materialization
    )

    dispositions = tuple(
        StageCellDisposition(
            stage="E1",
            cell_id=cell.cell_id,
            status="COMPLETE",
            reason_code="terminal_complete",
            terminal_receipt_sha256=_sha(f"terminal-{cell.cell_id}"),
        )
        for cell in materialization.cells
    )
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="E1",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(sorted(dispositions, key=lambda row: row.cell_id)),
        tts_l0_candidate_state_coverages=_candidate_state_coverages(materialization),
    )
    coverage.validate_against(materialization)
    c_sha, c_challenge, c_attestation, c_policy, c_now = _sign(coverage)
    signed_coverage = SignedStageCoverageReceipt(
        coverage,
        c_sha,
        c_challenge,
        c_attestation,
    )
    assert (
        signed_coverage.verify(
            materialization=materialization,
            policy=c_policy,
            expected_policy_sha256=c_policy.sha256,
            now_ns=c_now,
        )
        is coverage
    )

    incomplete = replace(coverage, dispositions=coverage.dispositions[:-1])
    with pytest.raises(ValueError, match="every and only"):
        incomplete.validate_against(materialization)

    with pytest.raises(ValueError, match="requires complete"):
        replace(coverage, tts_l0_candidate_state_coverages=())
    evidence = coverage.tts_l0_candidate_state_coverages[0]
    with pytest.raises(ValueError, match="canonical and unique"):
        replace(
            coverage,
            tts_l0_candidate_state_coverages=(evidence, evidence),
        )
    with pytest.raises(ValueError, match="source-round complete"):
        replace(
            evidence,
            l0_naive_observations=evidence.l0_naive_observations[:-1],
        )
    with pytest.raises(ValueError, match="exact source rounds"):
        replace(
            evidence,
            l0_naive_observations=(
                evidence.l0_naive_observations[0],
                replace(evidence.l0_naive_observations[1], source_round=1),
            ),
        )
    with pytest.raises(ValueError, match="differs from stage coverage identity"):
        replace(
            coverage,
            tts_l0_candidate_state_coverages=(
                replace(
                    evidence,
                    materialization_receipt_sha256=_sha("foreign-materialization"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="candidate_bytes_sha256"):
        replace(
            evidence,
            l0_naive_observations=(
                replace(
                    evidence.l0_naive_observations[0],
                    candidate_bytes_sha256=_sha("foreign-candidate"),
                ),
                *evidence.l0_naive_observations[1:],
            ),
        )
    with pytest.raises(ValueError, match="another materialized cell"):
        replace(
            evidence,
            tts_observations=(
                replace(evidence.tts_observations[0], cell_id=_sha("foreign-cell")),
                *evidence.tts_observations[1:],
            ),
        )
    with pytest.raises(ValueError, match="terminal pointer"):
        replace(
            evidence,
            terminal_pairs=(
                replace(evidence.terminal_pairs[0], tts_run_id="foreign-run"),
                *evidence.terminal_pairs[1:],
            ),
        )


def test_gpu_hour_output_and_global_banned_model_gate_fail_closed() -> None:
    pilot = PilotDurationReceipt(
        schema_version=2,
        protocol_lock_sha256=_sha("protocol"),
        materialization_receipt_sha256=_sha("materialization"),
        schedule_sha256=_sha("schedule"),
        inventory_gpu_count=2,
        observations=tuple(
            sorted(
                (
                    _pilot_observation("cell-a", 0, 1, 3_600),
                    _pilot_observation("cell-b", 0, 1, 7_200),
                    _pilot_observation("cell-c", 1, 2, 3_600),
                ),
                key=lambda row: (row.wave_index, row.cell_id),
            )
        ),
        retry_reserve_fraction=0.1,
        profile_reserve_gpu_hours=0.001,
        evidence_reserve_gpu_hours=0.002,
    )
    payload_sha256, challenge, attestation, policy, now_ns = _sign(pilot)
    available = _reduce_gpu_hours_from_signed_pilots_diagnostic(
        SignedPilotDurationReceipt(
            pilot,
            payload_sha256,
            challenge,
            attestation,
        ),
        policy=policy,
        expected_policy_sha256=policy.sha256,
        protocol_lock_sha256=pilot.protocol_lock_sha256,
        materialization_receipt_sha256=pilot.materialization_receipt_sha256,
        schedule_sha256=pilot.schedule_sha256,
        now_ns=now_ns,
    )
    assert available.compute_gpu_hours == pytest.approx(0.005)
    assert available.estimated_wall_hours == pytest.approx(0.003)
    assert available.compute_gpu_hours > available.estimated_wall_hours
    with pytest.raises(
        FormalGpuHourAuthorityBlocked,
        match="formal_lifecycle_phase_timing_proof_unregistered",
    ):
        reduce_gpu_hours_from_signed_pilots(
            SignedPilotDurationReceipt(
                pilot,
                payload_sha256,
                challenge,
                attestation,
            ),
            policy=policy,
            expected_policy_sha256=policy.sha256,
            protocol_lock_sha256=pilot.protocol_lock_sha256,
            materialization_receipt_sha256=pilot.materialization_receipt_sha256,
            schedule_sha256=pilot.schedule_sha256,
            now_ns=now_ns,
        )
    with pytest.raises(ValueError, match="cover all registered reserves"):
        replace(available, reserved_gpu_hours=0.0)
    with pytest.raises(ValueError, match="deterministic derivation"):
        replace(available, compute_gpu_hours=0.004)
    with pytest.raises(ValueError, match="complete phase coverage"):
        replace(pilot.observations[0], wall_time_ms=3_601)
    with pytest.raises(ValueError, match="reuse a first-party terminal"):
        replace(
            pilot,
            observations=tuple(
                sorted(
                    (
                        pilot.observations[0],
                        replace(
                            pilot.observations[1],
                            terminal_receipt_sha256=(
                                pilot.observations[0].terminal_receipt_sha256
                            ),
                        ),
                        pilot.observations[2],
                    ),
                    key=lambda row: (row.wave_index, row.cell_id),
                )
            ),
        )
    with pytest.raises(ValueError, match="gang counts exceed"):
        PilotDurationReceipt(
            schema_version=2,
            protocol_lock_sha256=_sha("protocol"),
            materialization_receipt_sha256=_sha("materialization"),
            schedule_sha256=_sha("schedule"),
            inventory_gpu_count=2,
            observations=tuple(
                sorted(
                    (
                        _pilot_observation("cell-a", 0, 2, 3_600),
                        _pilot_observation("cell-b", 0, 1, 7_200),
                    ),
                    key=lambda row: (row.wave_index, row.cell_id),
                )
            ),
            retry_reserve_fraction=0.1,
            profile_reserve_gpu_hours=0.0,
            evidence_reserve_gpu_hours=0.0,
        )


@pytest.mark.parametrize(
    ("observations", "expected_compute", "expected_wall", "expected_reserved"),
    (
        (
            (_pilot_observation("isolated", 0, 1, 3_600_000),),
            1.0,
            1.0,
            2.0,
        ),
        (
            tuple(
                sorted(
                    (
                        _pilot_observation("dp-a", 0, 1, 3_600_000),
                        _pilot_observation("dp-b", 0, 1, 3_600_000),
                    ),
                    key=lambda row: (row.wave_index, row.cell_id),
                )
            ),
            2.0,
            1.0,
            2.0,
        ),
        (
            tuple(
                _pilot_observation(f"tp2-{wave}", wave, 2, 3_600_000)
                for wave in range(3)
            ),
            6.0,
            3.0,
            6.0,
        ),
    ),
)
def test_gpu_hour_reserved_cost_uses_fixed_two_gpu_instance_capacity(
    observations: tuple[PilotDurationObservation, ...],
    expected_compute: float,
    expected_wall: float,
    expected_reserved: float,
) -> None:
    receipt = PilotDurationReceipt(
        schema_version=2,
        protocol_lock_sha256=_sha("protocol"),
        materialization_receipt_sha256=_sha("materialization"),
        schedule_sha256=_sha("schedule"),
        inventory_gpu_count=2,
        observations=observations,
        retry_reserve_fraction=0.0,
        profile_reserve_gpu_hours=0.0,
        evidence_reserve_gpu_hours=0.0,
    )
    payload_sha256, challenge, attestation, policy, now_ns = _sign(receipt)
    estimate = _reduce_gpu_hours_from_signed_pilots_diagnostic(
        SignedPilotDurationReceipt(
            receipt,
            payload_sha256,
            challenge,
            attestation,
        ),
        policy=policy,
        expected_policy_sha256=policy.sha256,
        protocol_lock_sha256=receipt.protocol_lock_sha256,
        materialization_receipt_sha256=receipt.materialization_receipt_sha256,
        schedule_sha256=receipt.schedule_sha256,
        now_ns=now_ns,
    )
    assert estimate.compute_gpu_hours == pytest.approx(expected_compute)
    assert estimate.estimated_wall_hours == pytest.approx(expected_wall)
    assert estimate.reserved_gpu_hours == pytest.approx(expected_reserved)


def test_global_banned_model_and_unresolved_placeholder_gates_fail_closed() -> None:
    with pytest.raises(ValueError, match="banned E6 model"):
        validate_formal_registry_no_banned_models(
            {"blocked_cells": [{"model": BANNED_MODEL, "status": "N/A"}]}
        )
    with pytest.raises(ValueError, match="banned E6 model"):
        MaterializedCell(
            stage="E6",
            method_role="Target-only",
            model=BANNED_MODEL,
            backend="NEXTN",
            task="download",
            publication_policy="none",
            recipe_sha256=None,
            dimensions=(),
        )
    with pytest.raises(ValueError, match="unresolved placeholder"):
        MaterializedCell(
            stage="E1",
            method_role="LightCone-candidate",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="tuning",
            publication_policy="first_ready",
            recipe_sha256=_sha("recipe"),
            dimensions=(("recipe", "future_template"),),
        )


def test_e0_public_materializer_requires_full_typed_proof_lineage() -> None:
    forbidden = {
        "final_blocks",
        "frozen_tts_recipe_sha256",
        "lightcone_recipe_sha256",
        "model",
        "online_spec_recipe_sha256s",
        "source_decision_sha256",
    }
    assert forbidden.isdisjoint(
        inspect.signature(materialize_e0_from_signed_compatibility).parameters
    )
    with pytest.raises(TypeError, match="durable registry verification"):
        materialize_e0_from_signed_compatibility(
            registry_verification_receipt=object(),
            signed_e6_confirmation=object(),
            e6_confirmation_proof_bundle=object(),
            signed_compatibility_receipt=object(),
            signed_onlinespec_tuning_seals=(),
            onlinespec_source_authority=object(),
            tuning_proof_set=object(),
            signed_power_prefix=object(),
            pilot_proof_set=object(),
            now_ns=1,
        )


def test_e6_public_materializer_requires_registered_typed_authorities() -> None:
    forbidden = {
        "final_blocks",
        "frozen_tts_recipe_sha256",
        "lightcone_recipe_sha256",
        "model",
        "source_decision_sha256",
    }
    assert forbidden.isdisjoint(inspect.signature(materialize_e6).parameters)
    with pytest.raises(TypeError, match="durable registry verification"):
        materialize_e6(
            registry_verification_receipt=object(),
            signed_e5_confirmation=object(),
            signed_model_compatibility=object(),
            compatibility_sources=(),
            signed_power_prefix=object(),
            pilot_materialization=object(),  # type: ignore[arg-type]
            pilot_coverage=object(),  # type: ignore[arg-type]
            pilot_evidence_manifest=object(),
            pilot_execution_bindings=(),
            now_ns=1,
        )


def test_e3b_public_materializer_requires_proof_derived_power_prefix() -> None:
    forbidden = {
        "final_blocks",
        "frozen_tts_recipe_sha256",
        "lightcone_recipe_sha256",
        "model",
        "source_decision_sha256",
    }
    assert forbidden.isdisjoint(inspect.signature(materialize_e3b).parameters)
    with pytest.raises(TypeError, match="durable registry verification"):
        materialize_e3b(
            registry_verification_receipt=object(),
            signed_power_prefix=object(),
            pilot_materialization=object(),  # type: ignore[arg-type]
            pilot_coverage=object(),  # type: ignore[arg-type]
            pilot_evidence_manifest=object(),
            pilot_execution_bindings=(),
            now_ns=1,
        )


def test_e1a_public_materializer_requires_proof_derived_e3b_confirmation() -> None:
    forbidden = {
        "frozen_tts_recipe_sha256",
        "lightcone_recipe_sha256",
        "model",
        "source_decision_sha256",
    }
    assert forbidden.isdisjoint(inspect.signature(materialize_e1a).parameters)
    with pytest.raises(TypeError, match="durable registry verification"):
        materialize_e1a(
            registry_verification_receipt=object(),
            signed_e3b_confirmation=object(),
            e3b_materialization=object(),  # type: ignore[arg-type]
            e3b_coverage=object(),  # type: ignore[arg-type]
            e3b_evidence_manifest=object(),
            e3b_execution_bindings=(),
            now_ns=1,
        )


def test_e4_screen_public_surface_requires_complete_typed_e2_lineage() -> None:
    forbidden = {
        "final_blocks",
        "frozen_tts_recipe_sha256",
        "lightcone_recipe_sha256",
        "model",
        "source_decision_sha256",
    }
    assert forbidden.isdisjoint(
        inspect.signature(materialize_e4_strength2_screen).parameters
    )


def test_e5_public_materializer_requires_proof_derived_power_and_anchor_receipt() -> (
    None
):
    forbidden = {
        "final_blocks",
        "frozen_tts_recipe_sha256",
        "lightcone_recipe_sha256",
        "model",
        "signed_anchor_selection",
        "signed_e1a_verification",
        "source_decision_sha256",
    }
    assert forbidden.isdisjoint(inspect.signature(materialize_e5).parameters)
    with pytest.raises(TypeError, match="durable registry verification"):
        materialize_e5(
            registry_verification_receipt=object(),
            signed_power_and_anchor_prefix=object(),
            pilot_materialization=object(),  # type: ignore[arg-type]
            pilot_coverage=object(),  # type: ignore[arg-type]
            pilot_evidence_manifest=object(),
            pilot_execution_bindings=(),
            formal_runtime_authority_manifest=object(),
            failure_diagnostic_authority=object(),  # type: ignore[arg-type]
            now_ns=1,
        )
    with pytest.raises(TypeError, match="durable registry verification"):
        materialize_e4_strength2_screen(
            registry_verification_receipt=object(),
            signed_e2_final_selection=object(),
            e2_materialization=object(),  # type: ignore[arg-type]
            e2_coverage=object(),  # type: ignore[arg-type]
            e2_source_recipes=(),
            e2_evidence_manifest=object(),
            e2_execution_bindings=(),
            now_ns=1,
        )


@pytest.mark.parametrize(
    ("materializer", "kwargs", "message"),
    (
        (
            materialize_e4_winner_neighborhood,
            {
                "registry_verification_receipt": object(),
                "signed_e4_screen_selection": object(),
                "screen_materialization": object(),
                "screen_coverage": object(),
                "screen_evidence_manifest": object(),
                "screen_execution_bindings": (),
                "now_ns": 1,
            },
            "durable registry verification",
        ),
        (
            materialize_e4_profiler,
            {
                "registry_verification_receipt": object(),
                "signed_e4_final_selection": object(),
                "local_materialization": object(),
                "local_coverage": object(),
                "local_evidence_manifest": object(),
                "local_execution_bindings": (),
                "now_ns": 1,
            },
            "durable registry verification",
        ),
    ),
)
def test_later_e4_public_surfaces_require_path_bound_typed_lineage(
    materializer: object,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        materializer(**kwargs)
