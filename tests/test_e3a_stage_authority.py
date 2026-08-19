from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_dispatch import _protocol_lock
from test_tts_calibration_authority import _unmeasured

import lightcone_spec.experiments.e3a_stage_authority as e3a_authority
from lightcone_spec.experiments.e3a_stage_authority import (
    E3A_STAGED_REDUCTION_PROTOCOL_SHA256,
    E3A_STAGED_REDUCTION_RUNNER_SHA256,
    E3A_STAGED_REDUCTION_TEST_SET_SHA256,
    E3aCapacityObservation,
    E3aCellExecutionEvidence,
    E3aStagedEvidenceManifest,
    build_e3a_staged_selection_receipt,
    reduce_e3a_staged_selection_from_proofs,
)
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS,
    FormalRuntimeAuthorityManifest,
    FormalRuntimeAuthorityMember,
    content_sha256,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.stage_materialization import (
    StageCellDisposition,
    StageCoverageReceipt,
    _materialize_e3a_diagnostic,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _runtime_manifest() -> FormalRuntimeAuthorityManifest:
    return FormalRuntimeAuthorityManifest(
        schema_version=2,
        authority_id="e3a-stage-test-v1",
        members=tuple(
            FormalRuntimeAuthorityMember(
                member_id=member_id,
                protocol_sha256=(
                    E3A_STAGED_REDUCTION_PROTOCOL_SHA256
                    if member_id == "e3a_selection_reducer"
                    else _sha(f"{member_id}:protocol")
                ),
                runner_sha256=(
                    E3A_STAGED_REDUCTION_RUNNER_SHA256
                    if member_id == "e3a_selection_reducer"
                    else _sha(f"{member_id}:runner")
                ),
                test_set_sha256=(
                    E3A_STAGED_REDUCTION_TEST_SET_SHA256
                    if member_id == "e3a_selection_reducer"
                    else _sha(f"{member_id}:tests")
                ),
                source_sha256=_sha(f"{member_id}:source"),
            )
            for member_id in FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
        ),
    )


def _materialization_and_coverage(runtime_manifest: FormalRuntimeAuthorityManifest):
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    materialization = _materialize_e3a_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_preflight_receipt_sha256=_sha("preflight"),
        workload_authority_sha256=lock.formal_workload_e3a_authorization_sha256,
        gpu_hours=_unmeasured(),
    )
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="E3a",
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage="E3a",
                cell_id=cell.cell_id,
                status="COMPLETE",
                reason_code="proof_derived_complete",
                terminal_receipt_sha256=_sha(f"terminal:{cell.cell_id}"),
            )
            for cell in materialization.cells
        ),
    )
    return lock, materialization, coverage


def _evidence_rows(
    tmp_path: Path, materialization
) -> tuple[E3aCellExecutionEvidence, ...]:
    rows = []
    for index, cell in enumerate(materialization.cells):
        result_path = (tmp_path / f"result-{index:03d}.json").resolve()
        timing_path = (tmp_path / f"timing-{index:03d}.json").resolve()
        publish_canonical_json_no_replace(
            result_path,
            {"cell_id": cell.cell_id, "kind": "test-result-binding"},
        )
        publish_canonical_json_no_replace(
            timing_path,
            {"cell_id": cell.cell_id, "kind": "test-timing-binding"},
        )
        result = CanonicalJsonProofBinding.bind(result_path)
        timing = CanonicalJsonProofBinding.bind(timing_path)
        method = "target_only" if cell.method_role == "Target-only" else "static"
        identity = StageItlExecutionIdentity(
            schema_version=1,
            kind="stage_itl_execution_identity",
            materialized_cell_id=cell.cell_id,
            inventory_sha256=_sha("inventory"),
            registry_sha256=_protocol_lock().registry_sha256,
            execution_plan_sha256=_sha(f"plan:{cell.cell_id}"),
            rank_config_sha256=_sha(f"rank:{cell.cell_id}"),
            run_id=f"e3a-run-{index:03d}",
            run_nonce_sha256=_sha(f"nonce:{cell.cell_id}"),
            attempt_id="attempt-0",
            method=method,
        )
        rows.append(
            E3aCellExecutionEvidence(
                schema_version=1,
                materialized_cell_id=cell.cell_id,
                execution_binding_sha256=_sha(f"binding:{cell.cell_id}"),
                execution_identity=identity,
                native_result_proof_path=result.absolute_path,
                native_result_proof_raw_sha256=result.raw_sha256,
                native_result_proof_semantic_sha256=result.semantic_sha256,
                stage_itl_proof_path=timing.absolute_path,
                stage_itl_proof_raw_sha256=timing.raw_sha256,
                stage_itl_proof_semantic_sha256=timing.semantic_sha256,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.materialized_cell_id))


def test_e3a_manifest_is_exact_360_path_bound_rows_and_requires_sealed_execution(
    tmp_path: Path,
) -> None:
    runtime_manifest = _runtime_manifest()
    lock, materialization, coverage = _materialization_and_coverage(runtime_manifest)
    assert len(materialization.cells) == 360
    assert {cell.method_role for cell in materialization.cells} == {
        "Static",
        "Target-only",
    }
    assert all("regime" in dict(cell.dimensions) for cell in materialization.cells)
    rows = _evidence_rows(tmp_path, materialization)
    manifest = E3aStagedEvidenceManifest(
        schema_version=1,
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        inventory_sha256=_sha("inventory"),
        reducer_authority_member_sha256=runtime_manifest.member(
            "e3a_selection_reducer"
        ).sha256,
        cells=rows,
    )
    assert len(manifest.cells) == 360
    with pytest.raises(ValueError, match="360 sorted unique"):
        replace(manifest, cells=manifest.cells[:-1])
    with pytest.raises(ValueError, match="reuses a terminal result proof"):
        replace(
            manifest,
            cells=tuple(
                sorted(
                    (
                        *manifest.cells[:-1],
                        replace(
                            manifest.cells[-1],
                            native_result_proof_path=(
                                manifest.cells[0].native_result_proof_path
                            ),
                            native_result_proof_raw_sha256=(
                                manifest.cells[0].native_result_proof_raw_sha256
                            ),
                            native_result_proof_semantic_sha256=(
                                manifest.cells[0].native_result_proof_semantic_sha256
                            ),
                        ),
                    ),
                    key=lambda row: row.materialized_cell_id,
                )
            ),
        )
    with pytest.raises(TypeError, match="sealed execution bindings"):
        reduce_e3a_staged_selection_from_proofs(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            coverage=coverage,
            manifest=manifest,
            execution_bindings=(object(),),  # type: ignore[arg-type]
            now_ns=1,
        )


def test_e3a_reducer_identity_and_six_output_receipt_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_manifest = _runtime_manifest()
    lock, materialization, coverage = _materialization_and_coverage(runtime_manifest)
    manifest = E3aStagedEvidenceManifest(
        schema_version=1,
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        inventory_sha256=_sha("inventory"),
        reducer_authority_member_sha256=runtime_manifest.member(
            "e3a_selection_reducer"
        ).sha256,
        cells=_evidence_rows(tmp_path, materialization),
    )
    foreign_member = replace(
        runtime_manifest.member("e3a_selection_reducer"),
        runner_sha256=_sha("foreign-e3a-runner"),
    )
    foreign_manifest = replace(
        runtime_manifest,
        members=tuple(
            foreign_member if row.member_id == foreign_member.member_id else row
            for row in runtime_manifest.members
        ),
    )
    foreign_lock = replace(
        lock,
        formal_runtime_authority_manifest_sha256=foreign_manifest.sha256,
    )
    with pytest.raises(ValueError, match="source identity"):
        reduce_e3a_staged_selection_from_proofs(
            protocol_lock=foreign_lock,
            formal_runtime_authority_manifest=foreign_manifest,
            materialization=materialization,
            coverage=coverage,
            manifest=manifest,
            execution_bindings=(),
            now_ns=1,
        )

    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    width_throughput = {4: 90, 8: 110, 16: 105}
    validated = {}
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        width = dimensions.get("width")
        throughput = 100 if width is None else width_throughput[int(width)]
        request_identity = (
            (
                "request",
                dimensions["context"],
                dimensions["regime"],
                dimensions["concurrency"],
                (1, 2, 3),
            ),
        )
        validated[cell.cell_id] = e3a_authority._ValidatedCell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],
            result=SimpleNamespace(terminal_sha256=_sha(f"terminal:{cell.cell_id}")),
            timing=SimpleNamespace(
                throughput_numerator_tokens=throughput,
                throughput_window_ns=100,
                sha256=_sha(f"timing:{cell.cell_id}"),
            ),
            request_identity=request_identity,
            peak_hbm_bytes=1_000 + (0 if width is None else int(width)),
        )
    proof_coverage = e3a_authority._coverage_from_validated(
        protocol_lock=lock,
        materialization=materialization,
        validated=validated,
    )
    proof_manifest = replace(
        manifest,
        coverage_receipt_sha256=proof_coverage.sha256,
    )
    monkeypatch.setattr(
        e3a_authority,
        "_validate_e3a_execution_rows",
        lambda **_kwargs: validated,
    )
    artifact = reduce_e3a_staged_selection_from_proofs(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        coverage=proof_coverage,
        manifest=proof_manifest,
        execution_bindings=(),
        now_ns=1,
    )
    names = (
        "baseline_capacity_envelope",
        "drift_witness",
        "e1_reference_load",
        "matched_width",
        "static_target_crossover",
        "width_selection_rule",
    )
    assert len(artifact.observations) == 360
    assert artifact.matched_width == 8
    assert artifact.common_load == 1
    assert tuple(row.name for row in artifact.locked_outputs) == names
    receipt = build_e3a_staged_selection_receipt(artifact)
    receipt.validate_artifact(artifact)
    assert tuple(row.name for row in receipt.locked_outputs) == names
    with pytest.raises(ValueError, match="differs from reducer"):
        receipt.validate_artifact(replace(artifact, matched_width=4))


def test_e3a_cell_evidence_rejects_unsealed_execution_binding() -> None:
    with pytest.raises(TypeError, match="sealed execution binding"):
        E3aCellExecutionEvidence.bind(
            execution_binding=object(),  # type: ignore[arg-type]
            native_result_proof_path="/does/not/matter.json",
            stage_itl_proof_path="/does/not/matter-either.json",
        )


def test_e3a_capacity_observation_rejects_missing_static_pair() -> None:
    with pytest.raises(ValueError, match="paired target"):
        E3aCapacityObservation(
            cell_id=_sha("cell"),
            method_role="Static",
            context=4096,
            regime="long_input_short_output",
            concurrency=1,
            width=8,
            throughput_tokens=2,
            throughput_window_ns=100,
            peak_hbm_bytes=1,
            target_cell_id=None,
            static_target_ratio_numerator=1,
            static_target_ratio_denominator=1,
            execution_evidence_sha256=_sha("evidence"),
            terminal_sha256=_sha("terminal"),
            timing_authority_sha256=_sha("timing"),
        )


def test_e3a_runtime_reducer_member_commitment_is_named_not_opaque() -> None:
    manifest = _runtime_manifest()
    member = manifest.member("e3a_selection_reducer")
    assert member.protocol_sha256 == E3A_STAGED_REDUCTION_PROTOCOL_SHA256
    assert member.runner_sha256 == E3A_STAGED_REDUCTION_RUNNER_SHA256
    assert member.test_set_sha256 == E3A_STAGED_REDUCTION_TEST_SET_SHA256
    assert content_sha256(member) == member.sha256
