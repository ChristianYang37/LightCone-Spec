from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import stage_activation
from lightcone_spec.experiments.formal_protocol import (
    CandidateStateReplay,
    CandidateStateTerminalPair,
    TtsL0CandidateStateCoverage,
)
from lightcone_spec.experiments.preflight_authority import (
    PREFLIGHT_POINTER_COVERAGE_PROTOCOL_SHA256,
    PREFLIGHT_REQUIRED_QUALIFICATION_SUITES,
    PreflightCellTerminal,
    PreflightCoverageBlocked,
    PreflightCoverageReceipt,
    PreflightExecutionSourceAuthority,
    PreflightQualificationProofSource,
    materialize_formal_preflight_stage_coverage,
    materialize_preflight_coverage,
    require_complete_preflight_coverage,
    verify_preflight_coverage,
)
from lightcone_spec.experiments.registry import (
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    materialize_preflight,
)
from lightcone_spec.orchestration.execution_bundle import BoundJsonSource
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding


def _sha(label: str) -> str:
    return content_sha256({"preflight-authority-test": label})


def _terminal(cell, *, status: str = "PASSED") -> PreflightCellTerminal:
    if cell.resources.workload_class is WorkloadClass.COMPILE:
        kind = "compile"
    elif cell.identity.task == "exactness_memory_telemetry_preflight":
        kind = "exactness"
    else:
        kind = "interference"
    return PreflightCellTerminal(
        cell_id=cell.cell_id,
        terminal_kind=kind,
        terminal_authority_sha256=_sha(f"terminal-{cell.cell_id}"),
        status=status,
        expected_rank_count=len(cell.resources.gpu_uuids),
        terminal_rank_count=len(cell.resources.gpu_uuids),
        failure_count=1 if status == "FAILED" else 0,
        error_count=1 if status == "ERROR" else 0,
        skip_count=1 if status == "SKIPPED" else 0,
    )


def _proof_binding(label: str) -> CanonicalJsonProofBinding:
    return CanonicalJsonProofBinding(
        absolute_path=f"/validation/{label}.json",
        raw_sha256=_sha(f"{label}-raw"),
        semantic_sha256=_sha(f"{label}-semantic"),
        size=2,
    )


def _bound_source(label: str) -> BoundJsonSource:
    return BoundJsonSource(
        path=f"/validation/{label}.json",
        canonical_sha256=_sha(f"{label}-canonical"),
        semantic_sha256=_sha(f"{label}-semantic"),
        file_sha256=_sha(f"{label}-file"),
        sidecar_file_sha256=_sha(f"{label}-sidecar"),
        size=2,
    )


def _formal_source(registry_sha256: str) -> PreflightExecutionSourceAuthority:
    return PreflightExecutionSourceAuthority(
        schema_version=3,
        kind="formal_preflight_execution_source_authority",
        registry_sha256=registry_sha256,
        dispatch_activation_sha256=_sha("pointer-activation"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        inventory_sha256=_sha("inventory"),
        release_root_manifest_sha256=_sha("root"),
        compile_result=_bound_source("compile"),
        exactness_result=_bound_source("exactness"),
        interference_proof_artifact=_proof_binding("interference"),
        qualification_proofs=tuple(
            PreflightQualificationProofSource(
                suite_id=suite_id,
                result_pointer=_proof_binding(f"{suite_id}-result"),
                proof_artifact=_proof_binding(f"{suite_id}-proof"),
            )
            for suite_id in PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
        ),
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
        "optimizer_state_bytes_sha256": _sha("candidate-optimizer"),
        "proposal_evidence_sha256": _sha("candidate-proposal"),
    }
    tts_cell = _sha("candidate-tts-cell")
    l0_cell = _sha("candidate-l0-cell")
    tts_pointer = _sha("candidate-tts-pointer")
    l0_pointer = _sha("candidate-l0-pointer")
    return TtsL0CandidateStateCoverage(
        schema_version=1,
        stage="preflight",
        scope="preflight_exactness_qualification",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        pair_id=_sha("candidate-pair"),
        tts_cell_id=tts_cell,
        l0_naive_cell_id=l0_cell,
        tts_native_replay_pointer_sha256=tts_pointer,
        l0_naive_native_replay_pointer_sha256=l0_pointer,
        qualification_cell_id=exactness.cell_id,
        source_round_plan_sha256=_sha("candidate-round-plan"),
        trainable_plan_sha256=plan,
        expected_source_rounds=(1,),
        tts_observations=(
            CandidateStateReplay(
                method_role="TTS",
                cell_id=tts_cell,
                run_id="preflight-candidate-tts",
                native_replay_pointer_sha256=tts_pointer,
                publication_policy="fixed_barrier",
                **shared,
            ),
        ),
        l0_naive_observations=(
            CandidateStateReplay(
                method_role="L0-naive",
                cell_id=l0_cell,
                run_id="preflight-candidate-l0",
                native_replay_pointer_sha256=l0_pointer,
                publication_policy="first_ready",
                **shared,
            ),
        ),
        terminal_pairs=(
            CandidateStateTerminalPair(
                source_round=1,
                tts_cell_id=tts_cell,
                l0_naive_cell_id=l0_cell,
                tts_run_id="preflight-candidate-tts",
                l0_naive_run_id="preflight-candidate-l0",
                tts_native_replay_pointer_sha256=tts_pointer,
                l0_naive_native_replay_pointer_sha256=l0_pointer,
                proposal_evidence_sha256=_sha("candidate-proposal"),
                tts_terminal_receipt_sha256=_sha("candidate-tts-terminal"),
                l0_naive_terminal_receipt_sha256=_sha("candidate-l0-terminal"),
            ),
        ),
    )


def test_available_preflight_without_terminals_cannot_materialize_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    monkeypatch.setattr(
        stage_activation,
        "release_dispatch_rejection_reason",
        lambda _cell: None,
    )
    activation = stage_activation.materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    assert activation.status == "AVAILABLE"
    with pytest.raises(ValueError, match="terminal coverage differs"):
        materialize_preflight_coverage(registry, activation, ())


def test_complete_coverage_requires_every_terminal_and_zero_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    monkeypatch.setattr(
        stage_activation,
        "release_dispatch_rejection_reason",
        lambda _cell: None,
    )
    activation = stage_activation.materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    assert activation.status == "AVAILABLE"
    cells = registry.cells_for("preflight")
    terminals = tuple(_terminal(cell) for cell in reversed(cells))

    receipt = materialize_preflight_coverage(registry, activation, terminals)

    assert receipt.status == "COMPLETE"
    assert tuple(row.cell_id for row in receipt.terminals) == tuple(
        sorted(cell.cell_id for cell in cells)
    )
    assert PreflightCoverageReceipt.from_dict(receipt.to_dict()) == receipt
    verify_preflight_coverage(registry, activation, receipt)
    with pytest.raises(
        PreflightCoverageBlocked,
        match="mandatory_preflight_terminal_not_clean",
    ):
        require_complete_preflight_coverage(receipt)

    skipped_cell = cells[0]
    skipped = tuple(
        _terminal(cell, status="SKIPPED" if cell == skipped_cell else "PASSED")
        for cell in cells
    )
    blocked = materialize_preflight_coverage(registry, activation, skipped)
    assert blocked.status == "BLOCKED"
    with pytest.raises(PreflightCoverageBlocked, match="terminal_not_clean"):
        require_complete_preflight_coverage(blocked)


def test_preflight_coverage_rejects_missing_rank_cell_kind_and_digest_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    monkeypatch.setattr(
        stage_activation,
        "release_dispatch_rejection_reason",
        lambda _cell: None,
    )
    activation = stage_activation.materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    cells = registry.cells_for("preflight")
    terminals = tuple(_terminal(cell) for cell in cells)

    with pytest.raises(ValueError, match="coverage differs"):
        materialize_preflight_coverage(registry, activation, terminals[:-1])

    first = terminals[0]
    with pytest.raises(ValueError, match="rank count differs"):
        materialize_preflight_coverage(
            registry,
            activation,
            (
                replace(
                    first,
                    expected_rank_count=first.expected_rank_count + 1,
                    terminal_rank_count=first.terminal_rank_count + 1,
                ),
            )
            + terminals[1:],
        )

    replacement_kind = "compile" if first.terminal_kind != "compile" else "exactness"
    with pytest.raises(ValueError, match="kind differs"):
        materialize_preflight_coverage(
            registry,
            activation,
            (replace(first, terminal_kind=replacement_kind),) + terminals[1:],
        )

    receipt = materialize_preflight_coverage(registry, activation, terminals)
    raw = receipt.to_dict()
    raw["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PreflightCoverageReceipt.from_dict(raw)


def test_preflight_source_persists_exact_ten_core_suite_specific_proofs() -> None:
    assert PREFLIGHT_REQUIRED_QUALIFICATION_SUITES == (
        "chronobelief_gpu_parity",
        "dspark_tp1",
        "dspark_tp2",
        "dspark_dp2",
        "native_hot_path_tp1",
        "nextn_tp1",
        "nextn_tp2",
        "session_reset_tp1",
        "tp1_dp2",
        "tp2_dp1",
    )
    proofs = tuple(
        PreflightQualificationProofSource(
            suite_id=suite_id,
            result_pointer=_proof_binding(f"{suite_id}-result"),
            proof_artifact=_proof_binding(f"{suite_id}-proof"),
        )
        for suite_id in PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
    )
    source = PreflightExecutionSourceAuthority(
        schema_version=3,
        kind="formal_preflight_execution_source_authority",
        registry_sha256=_sha("registry"),
        dispatch_activation_sha256=_sha("activation"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        inventory_sha256=_sha("inventory"),
        release_root_manifest_sha256=_sha("root"),
        compile_result=_bound_source("compile"),
        exactness_result=_bound_source("exactness"),
        interference_proof_artifact=_proof_binding("interference"),
        qualification_proofs=proofs,
    )

    assert PreflightExecutionSourceAuthority.from_dict(source.to_dict()) == source
    with pytest.raises(ValueError, match="exact ten core suites"):
        replace(source, qualification_proofs=proofs[:-1])
    with pytest.raises(ValueError, match="exact ten core suites"):
        replace(source, qualification_proofs=tuple(reversed(proofs)))
    with pytest.raises(ValueError, match="unsupported"):
        replace(proofs[0], suite_id="eagle3_tp1")


def test_pointer_coverage_bridge_materializes_exact_signable_stage_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry()
    materialization = materialize_preflight(
        protocol_lock_sha256=_sha("protocol-lock"),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    source = _formal_source(registry.sha256)
    terminals = tuple(
        sorted(
            (_terminal(cell) for cell in registry.cells_for("preflight")),
            key=lambda row: row.cell_id,
        )
    )
    pointer_coverage = PreflightCoverageReceipt(
        schema_version=2,
        kind="formal_preflight_coverage_receipt",
        protocol_sha256=PREFLIGHT_POINTER_COVERAGE_PROTOCOL_SHA256,
        registry_sha256=registry.sha256,
        activation_sha256=source.dispatch_activation_sha256,
        runtime_sha256=source.runtime_sha256,
        split_sha256=source.split_sha256,
        status="COMPLETE",
        terminals=terminals,
        source_authority=source,
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.preflight_authority."
        "materialize_pointer_preflight_coverage",
        lambda observed_registry, observed_source: (
            (
                SimpleNamespace(sha256=source.dispatch_activation_sha256),
                pointer_coverage,
            )
            if observed_registry == registry and observed_source == source
            else (_ for _ in ()).throw(AssertionError("unexpected pointer source"))
        ),
    )
    candidate = _candidate_coverage(materialization)
    monkeypatch.setattr(
        TtsL0CandidateStateCoverage,
        "validate_native_replay_pointers",
        lambda self, pointers: None,
    )
    monkeypatch.setattr(
        "lightcone_spec.orchestration.native_terminal."
        "validate_candidate_state_replay_proof_artifact",
        lambda path, **_kwargs: SimpleNamespace(
            method="tts" if Path(path).name.startswith("tts") else "l0"
        ),
    )
    tts_proof = (tmp_path / "tts-proof.json").resolve()
    l0_proof = (tmp_path / "l0-proof.json").resolve()

    stage_coverage = materialize_formal_preflight_stage_coverage(
        materialization,
        pointer_coverage,
        candidate_state_coverage=candidate,
        candidate_replay_proof_paths=(tts_proof, l0_proof),
        now_ns=1,
    )

    assert stage_coverage.stage == "preflight"
    assert len(stage_coverage.dispositions) == 10
    assert {row.status for row in stage_coverage.dispositions} == {"COMPLETE"}
    assert stage_coverage.tts_l0_candidate_state_coverages == (candidate,)
    stage_coverage.validate_against(materialization)

    incomplete = replace(
        pointer_coverage,
        terminals=(replace(terminals[0], status="SKIPPED", skip_count=1),)
        + terminals[1:],
        status="BLOCKED",
    )
    with pytest.raises(PreflightCoverageBlocked, match="terminal_not_clean"):
        materialize_formal_preflight_stage_coverage(
            materialization,
            incomplete,
            candidate_state_coverage=candidate,
            candidate_replay_proof_paths=(tts_proof, l0_proof),
            now_ns=1,
        )
    with pytest.raises(ValueError, match="exact deep reducer"):
        materialize_formal_preflight_stage_coverage(
            materialization,
            replace(pointer_coverage, activation_sha256=_sha("tampered-activation")),
            candidate_state_coverage=candidate,
            candidate_replay_proof_paths=(tts_proof, l0_proof),
            now_ns=1,
        )
    with pytest.raises(ValueError, match="canonical"):
        materialize_formal_preflight_stage_coverage(
            materialization,
            pointer_coverage,
            candidate_state_coverage=candidate,
            candidate_replay_proof_paths=("tts.json", l0_proof),
            now_ns=1,
        )
