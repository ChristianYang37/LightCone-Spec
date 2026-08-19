from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_dispatch import _protocol_lock
from test_formal_registry_integration import _Signer
from test_tts_calibration_authority import _authority, _inventory, _unmeasured

from lightcone_spec.config import (
    AdaptationConfig,
    ModelPair,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from lightcone_spec.experiments import e1_stage_authority, formal_stage_execution
from lightcone_spec.experiments.e1_stage_authority import E1CellExecutionEvidence
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS,
    FormalRuntimeAuthorityManifest,
    FormalRuntimeAuthorityMember,
    SignedTtsCalibrationSeal,
    TtsCalibrationSeal,
    content_sha256,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    build_source_formal_runtime_authority_manifest,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
    FORMAL_SERVING_EXECUTION_RUNNER_SHA256,
    FORMAL_SERVING_EXECUTION_TEST_SET_SHA256,
    E1RecipeAnchorAuthority,
    FormalOptimizerRecipe,
    FormalStageExecutionBlocked,
    VerifiedFormalServingExecutionBinding,
    prepare_formal_serving_execution_subject,
    verify_formal_serving_execution_binding,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    MaterializedCell,
    _materialize_e2_round_from_verified_values,
    _materialize_e3a_diagnostic,
    _materialize_tts_calibration_diagnostic,
    default_e2_recipe_grid_authority,
    e1_geometries,
    e2_candidate_recipes,
)
from lightcone_spec.orchestration.formal_serving_lift import (
    validate_formal_serving_itl_proof,
)
from lightcone_spec.orchestration.formal_terminal_result import (
    FormalDistributedTerminalRequestResult,
)
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace


def _runtime_manifest() -> FormalRuntimeAuthorityManifest:
    return FormalRuntimeAuthorityManifest(
        schema_version=2,
        authority_id="formal-runtime-test-v1",
        members=tuple(
            FormalRuntimeAuthorityMember(
                member_id=member_id,
                protocol_sha256=(
                    FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256
                    if member_id == "all_stage_execution_mapper"
                    else hashlib.sha256(f"{member_id}-protocol".encode()).hexdigest()
                ),
                runner_sha256=(
                    FORMAL_SERVING_EXECUTION_RUNNER_SHA256
                    if member_id == "all_stage_execution_mapper"
                    else hashlib.sha256(f"{member_id}-runner".encode()).hexdigest()
                ),
                test_set_sha256=(
                    FORMAL_SERVING_EXECUTION_TEST_SET_SHA256
                    if member_id == "all_stage_execution_mapper"
                    else hashlib.sha256(f"{member_id}-tests".encode()).hexdigest()
                ),
                source_sha256=hashlib.sha256(
                    f"{member_id}-source".encode()
                ).hexdigest(),
            )
            for member_id in FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
        ),
    )


def test_source_runtime_manifest_matches_serving_mapper_consumer() -> None:
    manifest = build_source_formal_runtime_authority_manifest(
        Path(__file__).resolve().parents[1]
    )
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=manifest.sha256,
    )
    assert (
        formal_stage_execution._require_execution_mapper_authority(
            lock,
            manifest,
        )
        == manifest.member("all_stage_execution_mapper").sha256
    )


def test_staged_reducer_dispatches_distributed_terminal_with_exact_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _protocol_lock()
    inventory_sha256 = hashlib.sha256(b"distributed-inventory").hexdigest()
    cell = MaterializedCell(
        stage="E6",
        method_role="Static",
        model="Qwen/Qwen3.6-35B-A3B",
        backend="nextn",
        task="formal-distributed-terminal-test",
        publication_policy="first_ready",
        recipe_sha256=None,
        dimensions=(("topology", "tp2_dp1"),),
    )
    identity = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id=cell.cell_id,
        inventory_sha256=inventory_sha256,
        registry_sha256=lock.registry_sha256,
        execution_plan_sha256=hashlib.sha256(b"distributed-plan").hexdigest(),
        rank_config_sha256=hashlib.sha256(b"distributed-ranks").hexdigest(),
        run_id="distributed-stage-run",
        run_nonce_sha256=hashlib.sha256(b"distributed-nonce").hexdigest(),
        attempt_id="attempt-0",
        method="static",
    )
    binding_sha256 = hashlib.sha256(b"distributed-binding").hexdigest()
    binding = SimpleNamespace(
        sha256=binding_sha256,
        subject=SimpleNamespace(
            stage="E6",
            protocol_lock_sha256=lock.sha256,
            materialized_cell_id=cell.cell_id,
            inventory_sha256=inventory_sha256,
            execution_identity=identity,
            topology_mode="tp2_dp1",
        ),
    )
    evidence = SimpleNamespace(
        execution_identity=identity,
        execution_binding_sha256=binding_sha256,
        native_result_proof_path="/proofs/distributed-result.json",
        native_result_proof_raw_sha256=hashlib.sha256(
            b"distributed-result-raw"
        ).hexdigest(),
        native_result_proof_semantic_sha256=hashlib.sha256(
            b"distributed-result-semantic"
        ).hexdigest(),
        stage_itl_proof_path="/proofs/distributed-itl.json",
    )
    request_payload = {
        "request_id": "request-0",
        "input_token_ids": [1],
        "output_token_ids": [2, 3],
        "terminal_status": "completed",
        "terminal_reason": "completed",
        "submitted_to_server": True,
    }
    request = FormalDistributedTerminalRequestResult(
        request_id="request-0",
        input_token_ids=(1,),
        output_token_ids=(2, 3),
        terminal_status="completed",
        terminal_reason="completed",
        submitted_to_server=True,
        request_sha256=content_sha256(request_payload),
    )
    terminal_sha256 = hashlib.sha256(b"distributed-terminal").hexdigest()
    result = SimpleNamespace(
        terminal_sha256=terminal_sha256,
        requests=(request,),
        scored_request_ids=("request-0",),
        updates=(),
        performance_counters={
            "exactness_violations": 0,
            "version_mismatches": 0,
            "fallbacks": 0,
            "nonfinite_updates": 0,
            "oom_events": 0,
            "retractions": 0,
            "communicator_failures": 0,
            "updates_published": None,
            "peak_hbm_bytes": 1,
            "exposed_update_ms": 0,
        },
    )
    timing = SimpleNamespace(
        execution_identity=identity,
        native_result_proof_path=evidence.native_result_proof_path,
        native_result_proof_raw_sha256=evidence.native_result_proof_raw_sha256,
        native_result_proof_semantic_sha256=(
            evidence.native_result_proof_semantic_sha256
        ),
        requests=(
            SimpleNamespace(
                request_id="request-0",
                output_token_ids=(2, 3),
                request_started_ns=10,
                request_terminal_ns=30,
                inter_token_ns=(10,),
            ),
        ),
    )
    observed: dict[str, object] = {}

    def validate_terminal(_path: str, **kwargs):
        observed.update(kwargs)
        return result

    monkeypatch.setattr(
        e1_stage_authority,
        "require_verified_formal_serving_execution_binding",
        lambda value: value,
    )
    monkeypatch.setattr(
        e1_stage_authority,
        "validate_formal_terminal_result_proof_artifact",
        validate_terminal,
    )
    monkeypatch.setattr(
        e1_stage_authority,
        "validate_formal_serving_itl_proof",
        lambda *_args, **_kwargs: timing,
    )
    validated = e1_stage_authority._validated_cell(
        cell=cell,
        evidence=evidence,
        execution_binding=binding,
        coverage_terminal_sha256=terminal_sha256,
        protocol_lock=lock,
        inventory_sha256=inventory_sha256,
        now_ns=10,
        expected_stage="E6",
    )
    assert observed["expected_stage"] == "E6"
    assert observed["expected_topology"] == "tp2_dp1"
    assert validated.metrics[0].output_token_ids == (2, 3)

    foreign_identity = replace(identity, run_nonce_sha256="f" * 64)
    monkeypatch.setattr(
        e1_stage_authority,
        "validate_formal_serving_itl_proof",
        lambda *_args, **_kwargs: SimpleNamespace(
            **{
                **timing.__dict__,
                "execution_identity": foreign_identity,
            }
        ),
    )
    with pytest.raises(ValueError, match="terminal/timing proof lineage"):
        e1_stage_authority._validated_cell(
            cell=cell,
            evidence=evidence,
            execution_binding=binding,
            coverage_terminal_sha256=terminal_sha256,
            protocol_lock=lock,
            inventory_sha256=inventory_sha256,
            now_ns=10,
            expected_stage="E6",
        )


def test_staged_reducer_rejects_legacy_inner_itl_as_formal_authority(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy-stage-itl.json"
    publish_canonical_json_no_replace(
        legacy,
        {
            "schema_version": 1,
            "kind": "stage_itl_timestamp_proof_artifact",
        },
    )
    with pytest.raises(ValueError, match="formal serving ITL proof fields differ"):
        validate_formal_serving_itl_proof(
            legacy,
            expected_registry_sha256="1" * 64,
            expected_root_manifest_sha256="2" * 64,
            now_ns=1,
        )


def _candidate_config(*, learning_rate: float, stride: int, gpu_uuid: str) -> RunConfig:
    return RunConfig(
        method="tts",
        model=ModelPair(
            target_revision="1" * 40,
            drafter_revision="2" * 40,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256="3" * 64,
            device_identity=gpu_uuid,
            speculative_num_draft_tokens=16,
        ),
        adaptation=AdaptationConfig(
            weight_update_mode="full",
            parameter_scope="all",
            adaptation_group_id="tts-cal-test",
            optimizer=OptimizerConfig(
                name="adam",
                learning_rate=learning_rate,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                grad_clip=None,
            ),
            stride=stride,
            canvas_tokens=16,
        ),
    )


def _slot_zero_cell(materialization):
    registry = {
        row.cell_id: row for row in build_industrial_registry().cells_for("TTS-Cal")
    }
    for cell in materialization.cells:
        source = registry[dict(cell.dimensions)["registry_cell_id"]]
        if source.identity.gpu_uuids == ("logical-rank-slot-0",):
            return cell
    raise AssertionError("test registry lost TTS-Cal logical slot zero")


def _e3a_static_cell(materialization):
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        if cell.method_role == "Static" and dimensions.get("width") == 8:
            return cell
    raise AssertionError("test registry lost the E3a Static width-8 row")


def _signed_tts_seal(*, lock, authority, signer: _Signer):
    seal = object.__new__(TtsCalibrationSeal)
    learning_rate = authority.learning_rates[0]
    stride = authority.strides[0]
    for name, value in {
        "schema_version": 2,
        "authority_sha256": authority.sha256,
        "protocol_lock_sha256": lock.sha256,
        "materialization_receipt_sha256": "4" * 64,
        "coverage_receipt_sha256": "5" * 64,
        "reduction_receipt_sha256": "6" * 64,
        "raw_manifest_sha256": "7" * 64,
        "tuning_window_sha256": authority.tuning_window_sha256,
        "selected_learning_rate": learning_rate,
        "selected_stride": stride,
        "selected_candidate_id": authority.candidate_id(
            learning_rate=learning_rate,
            stride=stride,
        ),
        "selected_pilot_run_binding_sha256s": tuple(
            hashlib.sha256(f"e2-pilot-{index}".encode()).hexdigest()
            for index in range(4)
        ),
        "selection_rule": "safety_first_then_maximize_slo_goodput",
        "result_class": "tuning_only_not_formal",
    }.items():
        object.__setattr__(seal, name, value)
    seal.__post_init__()
    return SignedTtsCalibrationSeal(seal, *signer.sign(seal))


def test_e3a_execution_subject_rebuilds_exact_staged_capacity_row(tmp_path) -> None:
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    materialization = _materialize_e3a_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_preflight_receipt_sha256="4" * 64,
        workload_authority_sha256=lock.formal_workload_e3a_authorization_sha256,
        gpu_hours=_unmeasured(),
    )
    cell = _e3a_static_cell(materialization)
    dimensions = dict(cell.dimensions)
    inventory = _inventory()
    registry = {
        row.cell_id: row for row in build_industrial_registry().cells_for("E3a")
    }
    source = registry[dimensions["registry_cell_id"]]
    gpu_index = int(source.identity.gpu_uuids[0].removeprefix("logical-rank-slot-"))
    gpu_uuid = inventory.devices[gpu_index].uuid
    width = int(dimensions["width"])
    config = RunConfig(
        method="static",
        model=ModelPair(
            target_revision="1" * 40,
            drafter_revision="2" * 40,
            draft_depth=width - 1,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256="3" * 64,
            device_identity=gpu_uuid,
            speculative_num_draft_tokens=width,
            max_running_requests=int(dimensions["concurrency"]),
        ),
    )
    proof_path = tmp_path / "runtime-proof.json"
    publish_canonical_json_no_replace(
        proof_path,
        {"kind": "non_authorizing_test_placeholder", "schema_version": 1},
    )
    subject = prepare_formal_serving_execution_subject(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        materialized_cell_id=cell.cell_id,
        run_config=config,
        inventory=inventory,
        gpu_uuids=(gpu_uuid,),
        runtime_gpu_proof_artifact_paths=(str(proof_path),),
        run_id="e3a-static-width8",
        run_nonce_sha256="5" * 64,
        attempt_id="attempt-0",
        tts_authority=None,
        now_ns=2_000_000_000,
    )
    assert subject.stage == "E3a"
    assert subject.method == "static"
    assert subject.workload_authority_sha256 == (
        lock.formal_workload_e3a_authorization_sha256
    )
    assert subject.recipe_authority_sha256s == (
        lock.formal_workload_e3a_authorization_sha256,
    )

    altered = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"max_running_requests": int(dimensions["concurrency"]) + 1}
            )
        }
    )
    with pytest.raises(ValueError, match="materialized load|staged registry row"):
        prepare_formal_serving_execution_subject(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            materialized_cell_id=cell.cell_id,
            run_config=altered,
            inventory=inventory,
            gpu_uuids=(gpu_uuid,),
            runtime_gpu_proof_artifact_paths=(str(proof_path),),
            run_id="e3a-static-wrong-load",
            run_nonce_sha256="6" * 64,
            attempt_id="attempt-0",
            tts_authority=None,
            now_ns=2_000_000_000,
        )


def test_tts_cal_execution_subject_derives_config_and_terminal_identity(
    tmp_path,
) -> None:
    authority = _authority()
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        tts_calibration_authority_sha256=authority.sha256,
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    materialization = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_e3a_receipt_sha256="4" * 64,
        calibration_authority_sha256=authority.sha256,
        gpu_hours=_unmeasured(),
    )
    cell = _slot_zero_cell(materialization)
    dimensions = dict(cell.dimensions)
    inventory = _inventory()
    config = _candidate_config(
        learning_rate=float(dimensions["learning_rate"]),
        stride=int(dimensions["stride"]),
        gpu_uuid=inventory.devices[0].uuid,
    )
    proof_path = tmp_path / "runtime-proof.json"
    publish_canonical_json_no_replace(
        proof_path,
        {
            "kind": "non_authorizing_test_placeholder",
            "schema_version": 1,
        },
    )
    subject = prepare_formal_serving_execution_subject(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        materialized_cell_id=cell.cell_id,
        run_config=config,
        inventory=inventory,
        gpu_uuids=(inventory.devices[0].uuid,),
        runtime_gpu_proof_artifact_paths=(str(proof_path),),
        run_id="tts-cal-run-0",
        run_nonce_sha256="5" * 64,
        attempt_id="attempt-0",
        tts_authority=authority,
        now_ns=2_000_000_000,
    )
    assert subject.materialized_cell_id == cell.cell_id
    assert subject.execution_identity.execution_plan_sha256 == (
        subject.execution_plan_sha256
    )
    assert subject.execution_identity.rank_config_sha256 == subject.rank_config_sha256
    assert subject.recipe_authority_sha256s == (authority.sha256,)
    assert subject.workload_authority_sha256 == authority.tuning_window_sha256
    assert subject.formal_runtime_authority_manifest_sha256 == (runtime_manifest.sha256)
    assert (
        subject.execution_mapper_authority_sha256
        == runtime_manifest.member("all_stage_execution_mapper").sha256
    )

    foreign_mapper = replace(
        runtime_manifest.members[0],
        runner_sha256=hashlib.sha256(b"foreign-mapper").hexdigest(),
    )
    foreign_manifest = replace(
        runtime_manifest,
        members=(foreign_mapper, *runtime_manifest.members[1:]),
    )
    foreign_lock = replace(
        lock,
        formal_runtime_authority_manifest_sha256=foreign_manifest.sha256,
    )
    foreign_materialization = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=foreign_lock.sha256,
        upstream_e3a_receipt_sha256="4" * 64,
        calibration_authority_sha256=authority.sha256,
        gpu_hours=_unmeasured(),
    )
    foreign_cell = _slot_zero_cell(foreign_materialization)
    foreign_dimensions = dict(foreign_cell.dimensions)
    foreign_config = _candidate_config(
        learning_rate=float(foreign_dimensions["learning_rate"]),
        stride=int(foreign_dimensions["stride"]),
        gpu_uuid=inventory.devices[0].uuid,
    )
    with pytest.raises(
        FormalStageExecutionBlocked,
        match="formal_execution_mapper_source_identity_mismatch",
    ):
        prepare_formal_serving_execution_subject(
            protocol_lock=foreign_lock,
            formal_runtime_authority_manifest=foreign_manifest,
            materialization=foreign_materialization,
            materialized_cell_id=foreign_cell.cell_id,
            run_config=foreign_config,
            inventory=inventory,
            gpu_uuids=(inventory.devices[0].uuid,),
            runtime_gpu_proof_artifact_paths=(str(proof_path),),
            run_id="tts-cal-run-foreign-mapper",
            run_nonce_sha256="a" * 64,
            attempt_id="attempt-0",
            tts_authority=authority,
            now_ns=2_000_000_000,
        )


def test_tts_cal_execution_subject_rejects_clip_stride_and_foreign_gpu(
    tmp_path,
) -> None:
    authority = _authority()
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        tts_calibration_authority_sha256=authority.sha256,
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    materialization = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_e3a_receipt_sha256="4" * 64,
        calibration_authority_sha256=authority.sha256,
        gpu_hours=_unmeasured(),
    )
    cell = _slot_zero_cell(materialization)
    dimensions = dict(cell.dimensions)
    inventory = _inventory()
    config = _candidate_config(
        learning_rate=float(dimensions["learning_rate"]),
        stride=int(dimensions["stride"]),
        gpu_uuid=inventory.devices[0].uuid,
    )
    proof_path = tmp_path / "runtime-proof.json"
    publish_canonical_json_no_replace(
        proof_path,
        {"kind": "non_authorizing_test_placeholder", "schema_version": 1},
    )
    clipped = config.model_copy(
        update={
            "adaptation": config.adaptation.model_copy(
                update={
                    "optimizer": config.adaptation.optimizer.model_copy(
                        update={"grad_clip": 1.0}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="frozen no-clip recipe"):
        prepare_formal_serving_execution_subject(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            materialized_cell_id=cell.cell_id,
            run_config=clipped,
            inventory=inventory,
            gpu_uuids=(inventory.devices[0].uuid,),
            runtime_gpu_proof_artifact_paths=(str(proof_path),),
            run_id="tts-cal-run-clipped",
            run_nonce_sha256="6" * 64,
            attempt_id="attempt-0",
            tts_authority=authority,
            now_ns=2_000_000_000,
        )
    with pytest.raises(ValueError, match="unknown GPU UUID"):
        prepare_formal_serving_execution_subject(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            materialized_cell_id=cell.cell_id,
            run_config=config,
            inventory=inventory,
            gpu_uuids=("GPU-foreign",),
            runtime_gpu_proof_artifact_paths=(str(proof_path),),
            run_id="tts-cal-run-foreign",
            run_nonce_sha256="7" * 64,
            attempt_id="attempt-0",
            tts_authority=authority,
            now_ns=2_000_000_000,
        )


def test_verifier_rebuilds_subject_and_rejects_directly_altered_workload(
    tmp_path,
) -> None:
    authority = _authority()
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        tts_calibration_authority_sha256=authority.sha256,
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    materialization = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_e3a_receipt_sha256="4" * 64,
        calibration_authority_sha256=authority.sha256,
        gpu_hours=_unmeasured(),
    )
    cell = _slot_zero_cell(materialization)
    dimensions = dict(cell.dimensions)
    inventory = _inventory()
    config = _candidate_config(
        learning_rate=float(dimensions["learning_rate"]),
        stride=int(dimensions["stride"]),
        gpu_uuid=inventory.devices[0].uuid,
    )
    proof_path = tmp_path / "runtime-proof.json"
    publish_canonical_json_no_replace(
        proof_path,
        {"kind": "non_authorizing_test_placeholder", "schema_version": 1},
    )
    subject = prepare_formal_serving_execution_subject(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        materialized_cell_id=cell.cell_id,
        run_config=config,
        inventory=inventory,
        gpu_uuids=(inventory.devices[0].uuid,),
        runtime_gpu_proof_artifact_paths=(str(proof_path),),
        run_id="tts-cal-run-altered",
        run_nonce_sha256="7" * 64,
        attempt_id="attempt-0",
        tts_authority=authority,
        now_ns=2_000_000_000,
    )
    altered = replace(subject, workload_authority_sha256="f" * 64)
    with pytest.raises(ValueError, match="deterministically rebuilt"):
        verify_formal_serving_execution_binding(
            altered,
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            run_config=config,
            inventory=inventory,
            tts_authority=authority,
            now_ns=2_000_000_000,
        )
    with pytest.raises(
        FormalStageExecutionBlocked,
        match="durable_content_verification_receipt_missing",
    ):
        verify_formal_serving_execution_binding(
            subject,
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            run_config=config,
            inventory=inventory,
            tts_authority=authority,
            now_ns=2_000_000_000,
        )


def test_e1_anchor_authority_is_complete_and_binding_is_unforgeable() -> None:
    authority = E1RecipeAnchorAuthority(
        schema_version=1,
        authority_id="e1-anchor-test",
        trainable_plan_sha256="8" * 64,
        anchors=tuple(
            sorted(
                (
                    FormalOptimizerRecipe(
                        anchor_name="adamw",
                        optimizer=OptimizerConfig(
                            name="adamw",
                            learning_rate=1e-4,
                            weight_decay=0.01,
                        ),
                        stride=10,
                    ),
                    FormalOptimizerRecipe(
                        anchor_name="sgdm",
                        optimizer=OptimizerConfig(
                            name="sgdm",
                            learning_rate=1e-3,
                            weight_decay=0.0,
                            momentum=0.9,
                        ),
                        stride=10,
                    ),
                ),
                key=lambda row: row.anchor_name,
            )
        ),
    )
    assert authority.anchor("adamw").optimizer.name == "adamw"
    with pytest.raises(TypeError, match="verifier-constructed only"):
        VerifiedFormalServingExecutionBinding(
            subject=None,  # type: ignore[arg-type]
            run_config=None,  # type: ignore[arg-type]
            runtime_gpu_proof_sha256s=("9" * 64,),
            verified_native_gpu_proofs=(),
            verified_distributed_gpu_proofs=(),
            verified_nextn_tp2_authority=None,
            hardware_envelope_sha256="a" * 64,
            _construction_seal=object(),
        )
    with pytest.raises(TypeError, match="sealed execution binding"):
        E1CellExecutionEvidence.bind(
            execution_binding=object(),  # type: ignore[arg-type]
            native_result_proof_path="/does/not/matter.json",
            stage_itl_proof_path="/does/not/matter-either.json",
        )


def test_e2_mapper_rebuilds_complete_numeric_candidate_and_rejects_drift() -> None:
    authority = _authority()
    grid = default_e2_recipe_grid_authority()
    lock = replace(
        _protocol_lock(),
        tts_calibration_authority_sha256=authority.sha256,
        e2_recipe_grid_authority_sha256=grid.sha256,
    )
    signer = _Signer()
    signed_seal = _signed_tts_seal(lock=lock, authority=authority, signer=signer)
    geometry = e1_geometries()[0]
    candidate = next(
        row
        for row in e2_candidate_recipes((geometry,), grid=grid)
        if row.optimizer == "adamw" and row.schedule == "cosine_to_zero"
    )
    materialization = _materialize_e2_round_from_verified_values(
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256="8" * 64,
        source_selection_sha256="9" * 64,
        grid=grid,
        geometries=(geometry,),
        round_index=0,
        model="Qwen/Qwen3-8B",
        matched_width=16,
        common_load=8,
        frozen_tts_recipe_sha256=signed_seal.payload.selected_candidate_id,
        candidate_recipes=None,
        prior_round_materialization=None,
        gpu_hours=_unmeasured(),
    )
    cell = next(
        row for row in materialization.cells if row.recipe_sha256 == candidate.sha256
    )
    config = RunConfig(
        method="l0",
        model=ModelPair(
            target="Qwen/Qwen3-8B",
            target_revision="1" * 40,
            drafter_revision="2" * 40,
            draft_depth=15,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256="3" * 64,
            device_identity="GPU-e2",
            speculative_num_draft_tokens=16,
            max_running_requests=8,
        ),
        adaptation=grid.adaptation_config_for(
            candidate,
            canvas_tokens=16,
            adaptation_group_id=f"e2:{cell.cell_id}",
        ),
    )
    authorities = formal_stage_execution._validate_e2_config(
        protocol_lock=lock,
        materialization=materialization,
        cell=cell,
        config=config,
        tts_authority=authority,
        signed_tts_seal=signed_seal,
        tts_seal_policy=signer.policy,
        grid=grid,
        now_ns=signer.now_ns,
    )
    assert candidate.sha256 in authorities
    assert grid.optimizer_recipe_authority.optimizer_recipe("adamw").sha256 in (
        authorities
    )
    assert (
        grid.optimizer_recipe_authority.schedule_recipe("cosine_to_zero").sha256
        in authorities
    )

    altered = config.model_copy(
        update={
            "adaptation": config.adaptation.model_copy(
                update={"adaptation_group_id": "caller-invented-e2-group"}
            )
        }
    )
    with pytest.raises(ValueError, match="complete recipe"):
        formal_stage_execution._validate_e2_config(
            protocol_lock=lock,
            materialization=materialization,
            cell=cell,
            config=altered,
            tts_authority=authority,
            signed_tts_seal=signed_seal,
            tts_seal_policy=signer.policy,
            grid=grid,
            now_ns=signer.now_ns,
        )
