from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import formal_single_operator_stages as stages
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_STAGE_DAG,
    CandidateStateReplay,
    CandidateStateTerminalPair,
    ProtocolLock,
    TtsL0CandidateStateCoverage,
)
from lightcone_spec.experiments.formal_registry import protocol_lock_to_dict
from lightcone_spec.experiments.formal_single_operator_stages import (
    FORMAL_SINGLE_OPERATOR_NODE_ORDER,
    FORMAL_SINGLE_OPERATOR_NODE_SPECS,
    FormalSingleOperatorPreflightActualReceipt,
    FormalSingleOperatorStageBlocked,
    FormalSingleOperatorStageDecision,
    build_formal_single_operator_execution_source,
    formal_single_operator_node_readiness,
    load_formal_single_operator_execution_source,
    materialize_formal_single_operator_node,
    next_formal_single_operator_node,
    publish_formal_single_operator_execution_source,
    publish_formal_single_operator_json_artifact,
    publish_formal_single_operator_preflight_actual,
    rebuild_formal_single_operator_stage_completion,
    reduce_formal_single_operator_node,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageCellDisposition,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def test_l0_anchor_safety_is_recorded_but_does_not_gate_lightcone_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = {
        role: [SimpleNamespace(cell_id=_sha(f"cell:{role}"))]
        for role in ("Target-only", "Static", "TTS", "L0-naive")
    }
    observations = {rows[0].cell_id: {"role": role} for role, rows in roles.items()}
    actuals = {
        rows[0].cell_id: SimpleNamespace(result_identity_sha256=_sha(f"actual:{role}"))
        for role, rows in roles.items()
    }
    monkeypatch.setattr(
        stages,
        "_adaptive_safety_reasons",
        lambda observation, **_kwargs: (
            ("nonfinite_updates",) if observation["role"] == "L0-naive" else ()
        ),
    )

    evaluations, ranking_reasons = stages._selection_anchor_evaluations(
        roles,  # type: ignore[arg-type]
        observations,
        actuals,  # type: ignore[arg-type]
    )

    l0 = next(row for row in evaluations if row["method_role"] == "L0-naive")
    assert l0["eligible"] is False
    assert l0["reason_codes"] == ["nonfinite_updates"]
    assert ranking_reasons == set()


@pytest.mark.parametrize(
    "status",
    (
        "NO_SAFE_SLO_WINNER",
        "NO_SAFE_GEOMETRY",
        "NO_SAFE_WINNER",
        "NO_SAFE_CONFIGURATION",
        "UNDERPOWERED",
        "POWER_UNRESOLVED",
    ),
)
def test_registered_scientific_stop_is_a_sealed_nonterminal_decision(
    status: str,
) -> None:
    decision = FormalSingleOperatorStageDecision(
        schema_version=1,
        kind="formal_single_operator_stage_decision",
        protocol_sha256=stages.FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node="e1",
        ordinal=3,
        stage="E1",
        phase="selection",
        predecessor_completion_sha256=_sha("predecessor"),
        materialization_sha256=_sha("materialization"),
        actual_result_set_sha256=_sha("actuals"),
        decision_kind="registered_scientific_stop",
        next_materialization_source_decision_sha256=None,
        next_materialization_upstream_receipt_sha256s=(),
        payload={"status": status, "reason_codes": ["measured_negative"]},
    )

    assert decision.payload["status"] == status
    assert decision.next_materialization_source_decision_sha256 is None
    assert decision.next_materialization_upstream_receipt_sha256s == ()
    assert FormalSingleOperatorStageDecision.from_dict(decision.to_dict()) == decision


def test_scientific_stop_cannot_authorize_downstream_and_unknown_status_is_malformed() -> (
    None
):
    common = {
        "schema_version": 1,
        "kind": "formal_single_operator_stage_decision",
        "protocol_sha256": stages.FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        "node": "e1",
        "ordinal": 3,
        "stage": "E1",
        "phase": "selection",
        "predecessor_completion_sha256": _sha("predecessor"),
        "materialization_sha256": _sha("materialization"),
        "actual_result_set_sha256": _sha("actuals"),
        "decision_kind": "registered_scientific_stop",
    }
    with pytest.raises(ValueError, match="cannot authorize a future node"):
        FormalSingleOperatorStageDecision(
            **common,
            next_materialization_source_decision_sha256=_sha("future"),
            next_materialization_upstream_receipt_sha256s=(_sha("receipt"),),
            payload={
                "status": "NO_SAFE_GEOMETRY",
                "reason_codes": ["no_safe_e1_geometry"],
            },
        )
    with pytest.raises(ValueError, match="must bind the exact next materialization"):
        FormalSingleOperatorStageDecision(
            **common,
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={"status": "UNKNOWN_SELECTION_STATUS"},
        )
    with pytest.raises(ValueError, match="canonical reason codes"):
        FormalSingleOperatorStageDecision(
            **common,
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={"status": "NO_SAFE_GEOMETRY", "reason_codes": []},
        )


@pytest.mark.parametrize(
    ("materializer", "predecessor_node", "negative_status"),
    (
        (
            stages._materialize_single_operator_e1,
            "tts_cal",
            "NO_SAFE_SLO_WINNER",
        ),
        (
            stages._materialize_single_operator_e2_round0,
            "e1",
            "NO_SAFE_GEOMETRY",
        ),
        (
            stages._materialize_single_operator_e4_screen,
            "e2_r3",
            "NO_SAFE_WINNER",
        ),
    ),
)
def test_negative_selection_is_a_typed_scientific_stage_block(
    materializer: object,
    predecessor_node: str,
    negative_status: str,
) -> None:
    predecessor = SimpleNamespace(
        artifact=SimpleNamespace(node=predecessor_node),
        decision=SimpleNamespace(payload={"status": negative_status}),
    )

    with pytest.raises(FormalSingleOperatorStageBlocked, match=negative_status):
        materializer(predecessor, _protocol_lock())  # type: ignore[operator]

    predecessor.decision.payload["status"] = "UNKNOWN_SELECTION_STATUS"
    with pytest.raises(ValueError, match="status is malformed"):
        materializer(predecessor, _protocol_lock())  # type: ignore[operator]


def _protocol_lock() -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="formal-single-operator-test",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=_sha("registry"),
        english_protocol_sha256=_sha("english"),
        chinese_protocol_sha256=_sha("chinese"),
        tts_calibration_authority_sha256=_sha("tts"),
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1-anchor"),
        e2_recipe_grid_authority_sha256=_sha("e2-grid"),
        formal_runtime_authority_manifest_sha256=_sha("runtime"),
        offline_release_trust_root_sha256=_sha("trust-root"),
        prepared_model_content_authorization_sha256=_sha("prepared-models"),
        formal_workload_e3a_authorization_sha256=_sha("e3a-workload"),
        formal_workload_e0_authorization_sha256=_sha("e0-workload"),
        burstgpt_shape_authorization_sha256=_sha("burstgpt"),
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


def test_trusted_preflight_derives_real_e3a_workload_sha_from_bound_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import formal_preflight_inputs as inputs_module
    from lightcone_spec.experiments.formal_preflight_inputs import (
        FormalPreflightExecutionInputs,
    )

    content_sha256 = _sha("trusted-content")
    workload_sha256 = _sha("trusted-workload")
    protocol_lock = replace(
        _protocol_lock(),
        schema_version=5,
        offline_release_trust_root_sha256=None,
        prepared_model_content_authorization_sha256=None,
        formal_workload_e3a_authorization_sha256=None,
        formal_workload_e0_authorization_sha256=None,
        burstgpt_shape_authorization_sha256=None,
        content_source_mode="trusted_single_operator",
        trusted_single_operator_content_bundle_sha256=content_sha256,
    )
    workload_binding = SimpleNamespace(
        identity="workload-binding",
        path="/unused/workload-authority.json",
    )
    inputs = object.__new__(FormalPreflightExecutionInputs)
    object.__setattr__(inputs, "schema_version", 3)
    object.__setattr__(
        inputs,
        "content_source_binding",
        SimpleNamespace(content_sha256=content_sha256),
    )
    object.__setattr__(inputs, "workload_authority", workload_binding)
    monkeypatch.setattr(
        inputs_module,
        "_trusted_content_sources",
        lambda **_kwargs: (
            object(),
            workload_binding,
            SimpleNamespace(sha256=workload_sha256),
            object(),
            object(),
            object(),
            object(),
        ),
    )

    assert (
        stages._preflight_e3a_workload_authority_sha256(
            execution_inputs=inputs,
            protocol_lock=protocol_lock,
        )
        == workload_sha256
    )
    object.__setattr__(
        inputs,
        "content_source_binding",
        SimpleNamespace(content_sha256=_sha("wrong-content")),
    )
    with pytest.raises(ValueError, match="lineage"):
        stages._preflight_e3a_workload_authority_sha256(
            execution_inputs=inputs,
            protocol_lock=protocol_lock,
        )


def _candidate_coverage(
    *,
    protocol_lock_sha256: str,
    materialization_sha256: str,
    qualification_cell_id: str,
) -> TtsL0CandidateStateCoverage:
    tts_cell = _sha("preflight-tts-cell")
    l0_cell = _sha("preflight-l0-cell")
    tts_pointer = _sha("preflight-tts-pointer")
    l0_pointer = _sha("preflight-l0-pointer")
    trainable = _sha("preflight-trainable")
    source_state = _sha("preflight-source-state")
    candidate = _sha("preflight-candidate")
    optimizer = _sha("preflight-optimizer")
    proposal = _sha("preflight-proposal")
    tts = CandidateStateReplay(
        method_role="TTS",
        cell_id=tts_cell,
        run_id="preflight-tts-run",
        native_replay_pointer_sha256=tts_pointer,
        source_round=1,
        source_version=0,
        source_state_sha256=source_state,
        trainable_plan_sha256=trainable,
        candidate_bytes_sha256=candidate,
        optimizer_state_bytes_sha256=optimizer,
        proposal_evidence_sha256=proposal,
        publication_policy="fixed_barrier",
    )
    l0 = CandidateStateReplay(
        method_role="L0-naive",
        cell_id=l0_cell,
        run_id="preflight-l0-run",
        native_replay_pointer_sha256=l0_pointer,
        source_round=1,
        source_version=0,
        source_state_sha256=source_state,
        trainable_plan_sha256=trainable,
        candidate_bytes_sha256=candidate,
        optimizer_state_bytes_sha256=optimizer,
        proposal_evidence_sha256=proposal,
        publication_policy="first_ready",
    )
    terminal = CandidateStateTerminalPair(
        source_round=1,
        tts_cell_id=tts_cell,
        l0_naive_cell_id=l0_cell,
        tts_run_id=tts.run_id,
        l0_naive_run_id=l0.run_id,
        tts_native_replay_pointer_sha256=tts_pointer,
        l0_naive_native_replay_pointer_sha256=l0_pointer,
        proposal_evidence_sha256=proposal,
        tts_terminal_receipt_sha256=_sha("preflight-tts-terminal"),
        l0_naive_terminal_receipt_sha256=_sha("preflight-l0-terminal"),
    )
    return TtsL0CandidateStateCoverage(
        schema_version=1,
        stage="preflight",
        scope="preflight_exactness_qualification",
        protocol_lock_sha256=protocol_lock_sha256,
        materialization_receipt_sha256=materialization_sha256,
        pair_id=_sha("preflight-pair"),
        tts_cell_id=tts_cell,
        l0_naive_cell_id=l0_cell,
        tts_native_replay_pointer_sha256=tts_pointer,
        l0_naive_native_replay_pointer_sha256=l0_pointer,
        qualification_cell_id=qualification_cell_id,
        source_round_plan_sha256=_sha("preflight-round-plan"),
        trainable_plan_sha256=trainable,
        expected_source_rounds=(1,),
        tts_observations=(tts,),
        l0_naive_observations=(l0,),
        terminal_pairs=(terminal,),
    )


def _preflight_coverage(materialization) -> StageCoverageReceipt:
    exactness_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task == "exactness_memory_telemetry_preflight"
    )
    assert len(exactness_cells) == 1
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="preflight",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage="preflight",
                cell_id=cell.cell_id,
                status="COMPLETE",
                reason_code="source_owned_actual_complete",
                terminal_receipt_sha256=_sha(f"terminal:{cell.cell_id}"),
            )
            for cell in materialization.cells
        ),
        tts_l0_candidate_state_coverages=(
            _candidate_coverage(
                protocol_lock_sha256=materialization.protocol_lock_sha256,
                materialization_sha256=materialization.sha256,
                qualification_cell_id=exactness_cells[0].cell_id,
            ),
        ),
    )
    coverage.validate_against(materialization)
    return coverage


def _initial_materialization(tmp_path: Path):
    protocol_lock = _protocol_lock()
    lock_path = tmp_path / "protocol-lock.json"
    publish_formal_single_operator_json_artifact(
        lock_path,
        protocol_lock_to_dict(protocol_lock),
    )
    rebuilt = materialize_formal_single_operator_node(
        node="preflight",
        predecessor_completion_path=None,
        protocol_lock_path=lock_path,
        materialization_output_path=tmp_path / "preflight-materialization.json",
        node_materialization_output_path=(
            tmp_path / "preflight-node-materialization.json"
        ),
        created_ns=10,
    )
    return protocol_lock, lock_path, rebuilt


def _publish_preflight_actual(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    protocol_lock: ProtocolLock,
    materialization,
    coverage: StageCoverageReceipt,
    name: str = "preflight-actual",
) -> Path:
    proof_path = (tmp_path / f"{name}-final-evidence-proof.json").resolve()
    publish_formal_single_operator_json_artifact(
        proof_path,
        {"kind": "test-final-evidence-source"},
    )
    evidence = SimpleNamespace(
        materialization=materialization,
        stage_coverage=coverage,
        sha256=_sha(f"{name}:final-evidence"),
    )
    reopened: list[tuple[str, int]] = []

    def _revalidate(source, *, verified_ns: int):
        reopened.append((source.absolute_path, verified_ns))
        return evidence

    monkeypatch.setattr(
        stages,
        "_revalidate_formal_single_operator_preflight_final_evidence",
        _revalidate,
    )
    actual_path = (tmp_path / f"{name}.json").resolve()
    publish_formal_single_operator_preflight_actual(
        final_evidence_source_path=proof_path,
        protocol_lock=protocol_lock,
        verified_ns=19,
        started_ns=11,
        finished_ns=19,
        output_path=actual_path,
    )
    assert reopened == [(str(proof_path), 19)]
    return actual_path


def test_fixed_atomic_dag_has_no_future_aggregate_or_callback_entrypoint() -> None:
    assert tuple(row.ordinal for row in FORMAL_SINGLE_OPERATOR_NODE_SPECS) == tuple(
        range(21)
    )
    assert tuple(row.stage for row in FORMAL_SINGLE_OPERATOR_NODE_SPECS) == (
        "preflight",
        "E3a",
        "TTS-Cal",
        "E1",
        "E2",
        "E2",
        "E2",
        "E2",
        "E4",
        "E4",
        "E4",
        "E3b",
        "E3b",
        "E1a",
        "E5",
        "E5",
        "E6",
        "E6",
        "E0",
        "E0",
        "E0",
    )
    assert {row.stage for row in FORMAL_SINGLE_OPERATOR_NODE_SPECS} == set(
        FORMAL_STAGE_DAG
    )
    assert next_formal_single_operator_node(None) == "preflight"
    assert next_formal_single_operator_node("e4_profiler") == "e3b_pilot"
    assert next_formal_single_operator_node("e0_final") is None
    assert (
        "adapter"
        not in inspect.signature(materialize_formal_single_operator_node).parameters
    )
    assert (
        "adapter"
        not in inspect.signature(reduce_formal_single_operator_node).parameters
    )
    assert (
        "actual_validator"
        not in inspect.signature(reduce_formal_single_operator_node).parameters
    )
    assert all("aggregate" not in node for node in FORMAL_SINGLE_OPERATOR_NODE_ORDER)


def test_preflight_rejects_caller_authored_complete_json(tmp_path: Path) -> None:
    _lock, _lock_path, rebuilt = _initial_materialization(tmp_path)
    fake = tmp_path / "caller-complete.json"
    publish_formal_single_operator_json_artifact(
        fake,
        {"status": "COMPLETE", "cell_id": rebuilt.materialization.cells[0].cell_id},
    )
    with pytest.raises(ValueError, match="preflight actual fields differ"):
        reduce_formal_single_operator_node(
            node_materialization_path=(
                tmp_path / "preflight-node-materialization.json"
            ),
            actual_result_paths={
                cell.cell_id: fake for cell in rebuilt.materialization.cells
            },
            decision_output_path=tmp_path / "must-not-exist-decision.json",
            completion_output_path=tmp_path / "must-not-exist-completion.json",
            completed_ns=20,
        )
    assert not (tmp_path / "must-not-exist-decision.json").exists()
    assert not (tmp_path / "must-not-exist-completion.json").exists()


def test_actual_preflight_unlocks_only_e3a_and_exports_current_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_lock, lock_path, preflight = _initial_materialization(tmp_path)
    coverage = _preflight_coverage(preflight.materialization)
    actual_path = _publish_preflight_actual(
        monkeypatch,
        tmp_path=tmp_path,
        protocol_lock=protocol_lock,
        materialization=preflight.materialization,
        coverage=coverage,
    )
    preflight_completion_path = tmp_path / "preflight-completion.json"
    completed = reduce_formal_single_operator_node(
        node_materialization_path=tmp_path / "preflight-node-materialization.json",
        actual_result_paths={
            cell.cell_id: actual_path for cell in preflight.materialization.cells
        },
        decision_output_path=tmp_path / "preflight-decision.json",
        completion_output_path=preflight_completion_path,
        completed_ns=20,
    )
    assert completed.artifact.node == "preflight"
    assert completed.decision.decision_kind == "preflight_all_complete"
    assert completed.decision.next_materialization_source_decision_sha256 == (
        protocol_lock.formal_workload_e3a_authorization_sha256
    )
    assert completed.decision.next_materialization_upstream_receipt_sha256s == (
        coverage.sha256,
    )

    e3a = materialize_formal_single_operator_node(
        node="e3a",
        predecessor_completion_path=preflight_completion_path,
        protocol_lock_path=None,
        materialization_output_path=tmp_path / "e3a-materialization.json",
        node_materialization_output_path=tmp_path / "e3a-node-materialization.json",
        created_ns=21,
    )
    assert e3a.materialization.stage == "E3a"
    assert e3a.materialization.expected_cell_count == 360
    assert e3a.materialization.source_decision_sha256 == (
        protocol_lock.formal_workload_e3a_authorization_sha256
    )
    assert e3a.materialization.upstream_receipt_sha256s == (coverage.sha256,)

    source = build_formal_single_operator_execution_source(
        tmp_path / "e3a-node-materialization.json"
    )
    assert source.schema_version == 2
    assert source.auxiliary_sources == ()
    assert source.predecessor_completion_sha256 == completed.artifact.sha256
    assert source.predecessor_decision_sha256 == completed.decision.sha256
    assert source.protocol_lock_source.absolute_path == str(lock_path)
    assert source.runtime_authority_manifest_sha256 == (
        protocol_lock.formal_runtime_authority_manifest_sha256
    )
    source_path = tmp_path / "e3a-execution-source.json"
    publish_formal_single_operator_execution_source(
        node_materialization_path=tmp_path / "e3a-node-materialization.json",
        output_path=source_path,
    )
    assert load_formal_single_operator_execution_source(source_path) == source

    with pytest.raises(
        FormalSingleOperatorStageBlocked,
        match="lacks exact current actual-result coverage",
    ):
        reduce_formal_single_operator_node(
            node_materialization_path=tmp_path / "e3a-node-materialization.json",
            actual_result_paths={},
            repository_root=tmp_path,
            decision_output_path=tmp_path / "e3a-decision.json",
            completion_output_path=tmp_path / "e3a-completion.json",
            completed_ns=30,
        )
    with pytest.raises(ValueError, match="not the next DAG node"):
        materialize_formal_single_operator_node(
            node="tts_cal",
            predecessor_completion_path=preflight_completion_path,
            protocol_lock_path=None,
            materialization_output_path=tmp_path / "skip-materialization.json",
            node_materialization_output_path=tmp_path / "skip-node.json",
            created_ns=30,
        )


def test_stage_artifacts_are_no_replace_and_tamper_evident(tmp_path: Path) -> None:
    _lock, lock_path, _preflight = _initial_materialization(tmp_path)
    with pytest.raises(RuntimeError, match="target already exists"):
        materialize_formal_single_operator_node(
            node="preflight",
            predecessor_completion_path=None,
            protocol_lock_path=lock_path,
            materialization_output_path=tmp_path / "preflight-materialization.json",
            node_materialization_output_path=(
                tmp_path / "preflight-node-materialization-2.json"
            ),
            created_ns=10,
        )
    materialization = tmp_path / "preflight-materialization.json"
    materialization.chmod(0o600)
    body = materialization.read_bytes()
    materialization.write_bytes(body.replace(b'"preflight"', b'"E3a"', 1))
    with pytest.raises(ValueError, match="changed"):
        build_formal_single_operator_execution_source(
            tmp_path / "preflight-node-materialization.json"
        )


def test_rebuilt_completion_reopens_actual_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_lock, _lock_path, preflight = _initial_materialization(tmp_path)
    coverage = _preflight_coverage(preflight.materialization)
    actual_path = _publish_preflight_actual(
        monkeypatch,
        tmp_path=tmp_path,
        protocol_lock=protocol_lock,
        materialization=preflight.materialization,
        coverage=coverage,
        name="actual",
    )
    completion_path = tmp_path / "completion.json"
    reduce_formal_single_operator_node(
        node_materialization_path=tmp_path / "preflight-node-materialization.json",
        actual_result_paths={
            cell.cell_id: actual_path for cell in preflight.materialization.cells
        },
        decision_output_path=tmp_path / "decision.json",
        completion_output_path=completion_path,
        completed_ns=20,
    )
    assert (
        rebuild_formal_single_operator_stage_completion(completion_path).artifact.node
        == "preflight"
    )
    actual_path.chmod(0o600)
    body = actual_path.read_bytes()
    changed = body.replace(b'"finished_ns":19', b'"finished_ns":18')
    assert changed != body
    actual_path.write_bytes(changed)
    with pytest.raises(ValueError, match="actual result changed"):
        rebuild_formal_single_operator_stage_completion(completion_path)


def test_reduced_stage_can_rebuild_after_nonretained_transitive_raw_is_archived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rolling archive boundary retains wrappers, not their bulky raw DAG."""

    protocol_lock, _lock_path, preflight = _initial_materialization(tmp_path)
    coverage = _preflight_coverage(preflight.materialization)
    actual_path = _publish_preflight_actual(
        monkeypatch,
        tmp_path=tmp_path,
        protocol_lock=protocol_lock,
        materialization=preflight.materialization,
        coverage=coverage,
        name="archive-safe-actual",
    )
    completion_path = tmp_path / "archive-safe-completion.json"
    reduce_formal_single_operator_node(
        node_materialization_path=tmp_path / "preflight-node-materialization.json",
        actual_result_paths={
            cell.cell_id: actual_path for cell in preflight.materialization.cells
        },
        decision_output_path=tmp_path / "archive-safe-decision.json",
        completion_output_path=completion_path,
        completed_ns=20,
    )
    raw_proof = tmp_path / "archive-safe-actual-final-evidence-proof.json"
    local_archive = tmp_path / "local-archive"
    local_archive.mkdir()
    raw_proof.rename(local_archive / raw_proof.name)

    assert (
        rebuild_formal_single_operator_stage_completion(completion_path).artifact.node
        == "preflight"
    )
    e3a = materialize_formal_single_operator_node(
        node="e3a",
        predecessor_completion_path=completion_path,
        protocol_lock_path=None,
        materialization_output_path=tmp_path / "archive-safe-e3a-materialization.json",
        node_materialization_output_path=tmp_path / "archive-safe-e3a-node.json",
        created_ns=21,
    )
    assert e3a.materialization.expected_cell_count == 360


def test_structurally_valid_preflight_receipt_cannot_self_authorize(
    tmp_path: Path,
) -> None:
    protocol_lock, _lock_path, preflight = _initial_materialization(tmp_path)
    proof_path = (tmp_path / "fabricated-final-evidence-proof.json").resolve()
    proof_binding = publish_formal_single_operator_json_artifact(
        proof_path,
        {"kind": "caller-authored-complete-proof"},
    )
    coverage = _preflight_coverage(preflight.materialization)
    receipt = FormalSingleOperatorPreflightActualReceipt(
        schema_version=2,
        kind="formal_single_operator_preflight_actual",
        protocol_sha256=preflight.artifact.protocol_sha256,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_sha256=preflight.materialization.sha256,
        final_evidence_source=proof_binding,
        final_evidence_sha256=_sha("fabricated-final-evidence"),
        stage_coverage_sha256=coverage.sha256,
        e3a_workload_authority_sha256=(
            protocol_lock.formal_workload_e3a_authorization_sha256
        ),
        verified_ns=19,
        started_ns=11,
        finished_ns=19,
    )
    actual_path = (tmp_path / "fabricated-preflight-actual.json").resolve()
    publish_formal_single_operator_json_artifact(actual_path, receipt.to_dict())
    with pytest.raises((TypeError, ValueError)):
        reduce_formal_single_operator_node(
            node_materialization_path=(
                tmp_path / "preflight-node-materialization.json"
            ),
            actual_result_paths={
                cell.cell_id: actual_path for cell in preflight.materialization.cells
            },
            decision_output_path=tmp_path / "fabricated-decision.json",
            completion_output_path=tmp_path / "fabricated-completion.json",
            completed_ns=20,
        )
    assert not (tmp_path / "fabricated-decision.json").exists()
    assert not (tmp_path / "fabricated-completion.json").exists()


def test_closed_stage_map_reports_connected_trusted_downstream_adapters() -> None:
    expected_connected = (
        "preflight",
        "e3a",
        "tts_cal",
        "e1",
        "e2_r0",
        "e2_r1",
        "e2_r2",
        "e2_r3",
    )
    for node in expected_connected:
        adapter = stages._CLOSED_NODE_ADAPTERS[node]
        assert adapter.materializer is not None
        assert adapter.actual_validator_kind is not None
        assert adapter.reducer is not None
        assert adapter.blocked_reason is None
    for node in ("e4_screen", "e4_local"):
        adapter = stages._CLOSED_NODE_ADAPTERS[node]
        assert adapter.materializer is not None
        assert adapter.actual_validator_kind == "run_manifest"
        assert adapter.reducer is not None
        assert adapter.blocked_reason is None
    profiler = stages._CLOSED_NODE_ADAPTERS["e4_profiler"]
    assert profiler.materializer is not None
    assert profiler.actual_validator_kind == "profiler_terminal"
    assert profiler.reducer is not None
    assert profiler.blocked_reason is None
    for node in stages._CONNECTED_DOWNSTREAM_NODES:
        adapter = stages._CLOSED_NODE_ADAPTERS[node]
        assert adapter.materializer is not None
        assert adapter.actual_validator_kind == "run_manifest"
        assert adapter.reducer is not None
        assert adapter.blocked_reason is None

    readiness = formal_single_operator_node_readiness()
    assert tuple(row.node for row in readiness) == FORMAL_SINGLE_OPERATOR_NODE_ORDER
    assert tuple(row.status for row in readiness) == ("READY",) * 21
    assert readiness[10].status == "READY"
    assert readiness[10].materializer_available is True
    assert readiness[10].actual_validator_kinds == ("profiler_terminal",)
    assert readiness[10].actual_validators_available is True
    assert readiness[10].blocker is None
    assert all(row.blocker is None for row in readiness)
    by_node = {row.node: row for row in readiness}
    assert by_node["e5_final"].actual_validator_kinds == (
        "run_manifest",
        "e5_failure_terminal",
    )
    assert by_node["e6_final"].actual_validator_kinds == (
        "run_manifest",
        "e6_interface_preflight",
    )
    assert by_node["e0_final"].actual_validator_kinds == (
        "run_manifest",
        "onlinespec_run_manifest",
    )
    assert by_node["e0_tuning"].actual_validator_kinds == (
        "run_manifest",
        "e0_compatibility_terminal",
        "onlinespec_run_manifest",
    )
    assert by_node["e3b_pilot"].required_auxiliary_source_kinds == ()
    assert by_node["e6_pilot"].required_auxiliary_source_kinds == ("e6_interface_fit",)
    assert by_node["e0_tuning"].required_auxiliary_source_kinds == ("e0_compatibility",)


def _route_cell(
    *,
    stage: str,
    role: str,
    task: str,
) -> MaterializedCell:
    return MaterializedCell(
        stage=stage,
        method_role=role,
        model="Qwen/test",
        backend="DFLASH",
        task=task,
        publication_policy="first_ready",
        recipe_sha256=None,
        dimensions=(),
    )


def test_cell_validator_registry_routes_mixed_late_stage_cells() -> None:
    assert (
        stages.formal_single_operator_cell_validator_kind(
            node="e4_profiler",
            cell=_route_cell(
                stage="E4",
                role="LightCone",
                task="mechanism_profile_only",
            ),
        )
        == "profiler_terminal"
    )
    assert (
        stages.formal_single_operator_cell_validator_kind(
            node="e5_final",
            cell=_route_cell(
                stage="E5",
                role="LightCone",
                task="deterministic_failure_injection",
            ),
        )
        == "e5_failure_terminal"
    )
    assert (
        stages.formal_single_operator_cell_validator_kind(
            node="e5_final",
            cell=_route_cell(stage="E5", role="Static", task="LiveCodeBench"),
        )
        == "run_manifest"
    )
    assert (
        stages.formal_single_operator_cell_validator_kind(
            node="e6_pilot",
            cell=_route_cell(
                stage="E6",
                role="Target-only",
                task="immutable_metadata_interface_and_fit_preflight",
            ),
        )
        == "e6_interface_preflight"
    )
    assert (
        stages.formal_single_operator_cell_validator_kind(
            node="e0_tuning",
            cell=_route_cell(stage="E0", role="OnlineSPEC-OPT", task="MATH-500"),
        )
        == "onlinespec_run_manifest"
    )
    assert (
        stages.formal_single_operator_cell_validator_kind(
            node="e0_tuning",
            cell=_route_cell(
                stage="E0",
                role="Compatibility",
                task="compatibility_decision",
            ),
        )
        == "e0_compatibility_terminal"
    )
    assert (
        stages.formal_single_operator_cell_validator_kind(
            node="e0_final",
            cell=_route_cell(stage="E0", role="Static", task="MATH-500"),
        )
        == "run_manifest"
    )
    with pytest.raises(ValueError, match="cell stage differs"):
        stages.formal_single_operator_cell_validator_kind(
            node="e6_final",
            cell=_route_cell(stage="E0", role="Static", task="MATH-500"),
        )


def test_auxiliary_source_contract_is_exact_and_reopens_canonical_bytes(
    tmp_path: Path,
) -> None:
    metadata = publish_formal_single_operator_json_artifact(
        (tmp_path / "metadata.json").resolve(),
        {"kind": "test_metadata", "members": []},
    )
    e6 = publish_formal_single_operator_json_artifact(
        (tmp_path / "e6.json").resolve(),
        {"kind": "test_e6_interface_fit", "models": []},
    )
    compatibility = publish_formal_single_operator_json_artifact(
        (tmp_path / "e0.json").resolve(),
        {"kind": "test_e0_compatibility", "decisions": []},
    )
    e6_sources = stages.bind_formal_single_operator_auxiliary_sources(
        node="e6_pilot",
        source_paths={
            "e6_interface_fit": e6.absolute_path,
        },
    )
    assert tuple(row.source_kind for row in e6_sources) == ("e6_interface_fit",)
    assert e6_sources[0].reopen()["kind"] == "test_e6_interface_fit"
    e6_spec = stages.formal_single_operator_node_spec("e6_pilot")
    execution_source = stages.FormalSingleOperatorExecutionSource(
        schema_version=2,
        kind="formal_single_operator_execution_source",
        protocol_sha256=stages.FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node=e6_spec.node,
        ordinal=e6_spec.ordinal,
        stage=e6_spec.stage,
        phase=e6_spec.phase,
        protocol_lock_source=metadata,
        protocol_lock_sha256=_sha("lock"),
        runtime_authority_manifest_sha256=_sha("runtime"),
        prepared_model_content_authorization_sha256=_sha("models"),
        formal_workload_e3a_authorization_sha256=_sha("e3a"),
        formal_workload_e0_authorization_sha256=_sha("e0"),
        burstgpt_shape_authorization_sha256=_sha("burst"),
        predecessor_completion_source=e6,
        predecessor_completion_sha256=_sha("predecessor"),
        predecessor_decision_sha256=_sha("decision"),
        materialization_source=compatibility,
        materialization_sha256=_sha("materialization"),
        materialization_source_decision_sha256=_sha("source-decision"),
        materialization_upstream_receipt_sha256s=(_sha("upstream"),),
        auxiliary_sources=e6_sources,
    )
    round_trip = stages.FormalSingleOperatorExecutionSource.from_dict(
        execution_source.to_dict()
    )
    assert round_trip == execution_source
    assert round_trip.reopen_auxiliary_source("e6_interface_fit")["kind"] == (
        "test_e6_interface_fit"
    )
    with pytest.raises(ValueError, match="is absent"):
        round_trip.auxiliary_source_binding("e0_compatibility")
    e0_sources = stages.bind_formal_single_operator_auxiliary_sources(
        node="e0_final",
        source_paths={
            "e0_compatibility": compatibility.absolute_path,
        },
    )
    assert tuple(row.source_kind for row in e0_sources) == ("e0_compatibility",)
    with pytest.raises(ValueError, match="keys differ"):
        stages.bind_formal_single_operator_auxiliary_sources(
            node="e6_final",
            source_paths={},
        )
    with pytest.raises(ValueError, match="keys differ"):
        stages.bind_formal_single_operator_auxiliary_sources(
            node="e3b_final",
            source_paths={
                "e0_compatibility": compatibility.absolute_path,
            },
        )


def test_e0_transition_uses_probe_bundle_without_dropping_immediate_e6_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_downstream

    protocol_lock = _protocol_lock()
    compatibility = publish_formal_single_operator_json_artifact(
        (tmp_path / "compatibility-bundle.json").resolve(),
        {"kind": "test_e0_compatibility_bundle"},
    )
    auxiliary_sources = stages.bind_formal_single_operator_auxiliary_sources(
        node="e0_tuning",
        source_paths={"e0_compatibility": compatibility.absolute_path},
    )
    e6_materialization_sha256 = _sha("E6 materialization")
    e6_confirmation_sha256 = _sha("E6 confirmation")
    bundle_sha256 = _sha("E0 compatibility bundle")
    predecessor = SimpleNamespace(
        artifact=SimpleNamespace(protocol_lock_sha256=protocol_lock.sha256),
        materialization=SimpleNamespace(sha256=e6_materialization_sha256),
        decision=SimpleNamespace(
            next_materialization_source_decision_sha256=e6_confirmation_sha256,
            next_materialization_upstream_receipt_sha256s=(e6_materialization_sha256,),
        ),
    )
    reopened: list[object] = []

    def _reopen_e0(predecessor_value, lock_value, value):
        assert predecessor_value is predecessor
        assert lock_value is protocol_lock
        reopened.append(value)
        return None, None, bundle_sha256, _sha("E0 evidence")

    monkeypatch.setattr(
        formal_single_operator_downstream,
        "_e0_compatibility_from_auxiliary",
        _reopen_e0,
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(e6_materialization_sha256,),
        source_decision_sha256=bundle_sha256,
        materialization_rule="compatibility_probe_derived_test",
        expected_cell_count=0,
        cells=(),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    stages._validate_single_operator_materialization_transition(
        node="e0_tuning",
        predecessor=predecessor,
        protocol_lock=protocol_lock,
        auxiliary_sources=auxiliary_sources,
        materialization=materialization,
        message="transition differs",
    )
    assert reopened == [{"kind": "test_e0_compatibility_bundle"}]
    assert materialization.source_decision_sha256 != e6_confirmation_sha256

    wrong_source = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(e6_materialization_sha256,),
        source_decision_sha256=e6_confirmation_sha256,
        materialization_rule="compatibility_probe_derived_test",
        expected_cell_count=0,
        cells=(),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    with pytest.raises(ValueError, match="transition differs"):
        stages._validate_single_operator_materialization_transition(
            node="e0_tuning",
            predecessor=predecessor,
            protocol_lock=protocol_lock,
            auxiliary_sources=auxiliary_sources,
            materialization=wrong_source,
            message="transition differs",
        )


def test_registered_validator_implementations_have_distinct_protocols(
    tmp_path: Path,
) -> None:
    validators = (
        stages.FormalSingleOperatorRunManifestActualValidator(str(tmp_path)),
        stages.FormalSingleOperatorOnlineSpecRunManifestActualValidator(str(tmp_path)),
        stages.FormalSingleOperatorProfilerActualValidator(),
        stages.FormalSingleOperatorE5FailureActualValidator(),
        stages.FormalSingleOperatorE6InterfacePreflightActualValidator(
            _protocol_lock()
        ),
    )
    assert len({row.validator_kind for row in validators}) == len(validators)
    assert all(len(row.protocol_sha256) == 64 for row in validators)
    assert (
        stages._single_operator_runtime_method(
            _route_cell(stage="E0", role="OnlineSPEC-OGD", task="MATH-500")
        )
        == "onlinespec_ogd"
    )
    assert (
        stages._single_operator_runtime_method(
            _route_cell(stage="E0", role="OnlineSPEC-OPT", task="MATH-500")
        )
        == "onlinespec_opt"
    )
    assert (
        stages._single_operator_runtime_method(
            _route_cell(stage="E0", role="OnlineSPEC-ENS", task="MATH-500")
        )
        == "onlinespec_ens"
    )


def test_e3a_lambda_star_uses_completed_request_rate_not_token_throughput() -> None:
    cell = MaterializedCell(
        stage="E3a",
        method_role="Static",
        model="Qwen/test",
        backend="DFLASH",
        task="controlled_capacity",
        publication_policy="none",
        recipe_sha256=None,
        dimensions=tuple(
            sorted(
                {
                    "concurrency": 8,
                    "context": 40_928,
                    "regime": "short_input_long_generation",
                    "width": 16,
                }.items()
            )
        ),
    )
    observation = {
        "requests": [
            {
                "request_id": "request-0",
                "output_token_ids": [1, 2, 3, 4, 5],
                "request_started_ns": 100,
                "request_terminal_ns": 400,
                "submitted_to_server": True,
                "terminal_status": "completed",
            },
            {
                "request_id": "request-1",
                "output_token_ids": [6],
                "request_started_ns": 200,
                "request_terminal_ns": 600,
                "submitted_to_server": True,
                "terminal_status": "completed",
            },
        ]
    }
    locked = stages._single_operator_e3a_lambda_star_request_rate(
        cell=cell,
        observation=observation,
        matched_width=16,
        common_load=8,
    )
    assert locked["numerator_requests_x_1e9"] == 2_000_000_000
    assert locked["denominator_window_ns"] == 500
    assert locked["source_cell_id"] == cell.cell_id
    assert len(locked["source_observation_sha256"]) == 64
    incomplete = {
        **observation,
        "requests": [
            {**observation["requests"][0], "terminal_status": "timed_out"},
            observation["requests"][1],
        ],
    }
    with pytest.raises(ValueError, match="incomplete request"):
        stages._single_operator_e3a_lambda_star_request_rate(
            cell=cell,
            observation=incomplete,
            matched_width=16,
            common_load=8,
        )
