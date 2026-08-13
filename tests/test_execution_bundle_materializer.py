from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuDispatchPlan,
    GpuDispatchWave,
    registry_pool_work_item,
)
from lightcone_spec.experiments.itl_authority import (
    ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON,
    release_e2_itl_timestamp_plan,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    content_sha256,
    serving_cell_rejection_reason,
)
from lightcone_spec.orchestration.execution_bundle_materializer import (
    AssignmentRunNonceReceipt,
    DispatchBundleMaterializationBlocked,
    DispatchBundleMaterializationRequest,
    DispatchExecutionBundleManifest,
    MaterializedAssignmentBundleReceipt,
    bind_dispatch_bundle_materialization_inputs,
)


def _write_bound(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(f"{path}.sha256").write_text(content_sha256(value) + "\n", encoding="ascii")
    return path


def _request_value(
    tmp_path: Path,
    *,
    method: str = "target_only",
) -> tuple[dict[str, object], GpuAssignment, GpuDispatchPlan]:
    registry = build_industrial_registry(
        gpu_uuids=("GPU-logical-a", "GPU-logical-b"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
        base_port=28_000,
    )
    experiment = "E3a" if method in {"target_only", "static"} else "E1"
    cell = next(
        cell
        for cell in registry.cells_for(experiment)
        if cell.identity.method == method
        and (
            method not in {"target_only", "static"}
            or serving_cell_rejection_reason(cell) is None
        )
    )
    item = registry_pool_work_item(cell, estimated_duration_seconds=1.0)
    assignment = GpuAssignment(
        work_item=item,
        gpu_uuids=("GPU-physical-a",),
        rank_groups=(("GPU-physical-a",),),
        ports=tuple(range(31_000, 31_000 + item.claim.port_count)),
    )
    interference_sha256 = content_sha256("materializer-interference")
    wave = GpuDispatchWave(
        wave_index=0,
        assignments=(assignment,),
        interference_envelope_sha256=interference_sha256,
    )
    dispatch = GpuDispatchPlan(
        schema_version=1,
        registry_sha256=registry.sha256,
        inventory_sha256=content_sha256("materializer-inventory"),
        receipts_sha256=content_sha256("materializer-receipts"),
        interference_envelope_sha256=interference_sha256,
        budget_sha256_by_cell=((cell.cell_id, content_sha256("materializer-budget")),),
        seed=20260811,
        waves=(wave,),
        completed_cell_ids=(),
    )

    shared_roles = (
        "registry",
        "inventory",
        "interference_envelope",
        "interference_source_receipt",
        "budget_plan",
        "budget_policy",
        "capacity_envelope",
        "capacity_source_manifest",
        "capacity_verification_receipt",
        "activation",
        "activation_runtime",
        "activation_split",
        "dispatch_context",
    )
    shared = {
        role: str(
            _write_bound(tmp_path / f"shared-{role}.json", {"role": role}).resolve()
        )
        for role in shared_roles
    }
    dispatch_path = _write_bound(tmp_path / "dispatch-plan.json", dispatch.to_dict())
    shared["dispatch_plan"] = str(dispatch_path.resolve())

    checkout = tmp_path / "patched-sglang"
    checkout.mkdir()
    nonce = AssignmentRunNonceReceipt.issue(
        dispatch_plan_sha256=dispatch.sha256,
        assignment=assignment,
    )

    required_assignment_roles = (
        "topology_receipts",
        "production_load",
        "run_config",
        "launch_policy",
        "run_nonce_receipt",
        "split_artifact",
        "sampling_artifact",
        "model_lock_artifact",
        "prepared_models",
        "compile_cache_plan",
        "inventory_source_artifact",
        "runtime_envelope_artifact",
        "execution_policy",
    )
    assignment_sources = {
        role: str(
            _write_bound(tmp_path / f"assignment-{role}.json", {"role": role}).resolve()
        )
        for role in required_assignment_roles
        if role not in {"launch_policy", "run_nonce_receipt"}
    }
    launch_policy = _write_bound(
        tmp_path / "assignment-launch-policy.json",
        {
            "schema_version": 1,
            "kind": "industrial_server_launch_materialization_policy",
            "patched_sglang_checkout": str(checkout.resolve()),
            "adaptation_reserve_mb": 0 if method in {"target_only", "static"} else 1,
            "mem_fraction_static": 0.8,
            "host": "127.0.0.1",
        },
    )
    nonce_path = _write_bound(tmp_path / "assignment-run-nonce.json", nonce.to_dict())
    assignment_sources["launch_policy"] = str(launch_policy.resolve())
    assignment_sources["run_nonce_receipt"] = str(nonce_path.resolve())
    if method in {"tts", "l0"}:
        trainable = _write_bound(
            tmp_path / "assignment-trainable-binding.json",
            {"role": "trainable-binding"},
        )
        prepared_release = _write_bound(
            tmp_path / "assignment-prepared-release.json",
            {"role": "prepared-release"},
        )
        assignment_sources["trainable_plan_authority_binding"] = str(
            trainable.resolve()
        )
        assignment_sources["prepared_model_content_release_manifest"] = str(
            prepared_release.resolve()
        )
    budget_load = _write_bound(tmp_path / "budget-load.json", {"role": "budget"})
    receipt = _write_bound(tmp_path / "dependency-receipt.json", {"role": "receipt"})
    dependency = _write_bound(
        tmp_path / "dependency-runtime-envelope.json", {"role": "runtime_envelope"}
    )
    assignment_sources["runtime_envelope_artifact"] = str(dependency.resolve())
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "industrial_dispatch_bundle_materialization_request",
        "shared_sources": shared,
        "shared_multi_sources": {
            "budget_load_bindings": [str(budget_load.resolve())],
            "dependency_receipts": [str(receipt.resolve())],
        },
        "shared_optional_sources": {
            "interference_calibration_execution_authority": None,
        },
        "assignments": [
            {
                "cell_id": cell.cell_id,
                "sources": [
                    {"role": role, "path": path}
                    for role, path in sorted(assignment_sources.items())
                ],
                "dependency_artifacts": [
                    {
                        "experiment": "preflight",
                        "name": "runtime_envelope",
                        "path": str(dependency.resolve()),
                    }
                ],
            }
        ],
    }
    return value, assignment, dispatch


def test_binds_complete_plan_coverage_and_derives_assignment_identity(
    tmp_path: Path,
) -> None:
    value, expected_assignment, dispatch = _request_value(tmp_path)
    request_path = _write_bound(tmp_path / "materialization-request.json", value)

    bound = bind_dispatch_bundle_materialization_inputs(request_path)

    assert len(bound.assignments) == 1
    assignment = bound.assignments[0]
    assert assignment.cell_id == expected_assignment.work_item.item_id
    assert assignment.assignment_sha256 == expected_assignment.assignment_id
    nonce_value = next(
        source.load()
        for role, source in assignment.sources
        if role == "run_nonce_receipt"
    )
    assert (
        assignment.run_nonce_sha256
        == AssignmentRunNonceReceipt.from_dict(nonce_value).run_nonce_sha256
    )
    assert assignment.output_root == str(
        Path(expected_assignment.work_item.cell.resources.evidence_root).resolve()
    )
    assert bound.dispatch_plan.semantic_sha256 == dispatch.sha256
    assert bound.request.semantic_sha256 == bound.request.canonical_sha256
    assert "assignment_sha256" not in value["assignments"][0]
    assert "execution_plan_sha256" not in value["assignments"][0]
    assert "execution_plan_summary" not in value["assignments"][0]
    assert DispatchBundleMaterializationRequest.from_dict(value).to_dict() == value
    assert nonce_value["assignment_sha256"] == assignment.assignment_sha256


def test_rejects_incomplete_assignment_coverage(tmp_path: Path) -> None:
    value, _, _ = _request_value(tmp_path)
    value["assignments"][0]["cell_id"] = "0" * 64
    request_path = _write_bound(tmp_path / "wrong-coverage.json", value)

    with pytest.raises(
        DispatchBundleMaterializationBlocked,
        match="bundle_assignment_runtime_source_coverage_incomplete",
    ):
        bind_dispatch_bundle_materialization_inputs(request_path)


def test_rejects_missing_release_runtime_role_with_named_block(tmp_path: Path) -> None:
    value, _, _ = _request_value(tmp_path)
    sources = value["assignments"][0]["sources"]
    value["assignments"][0]["sources"] = [
        source for source in sources if source["role"] != "launch_policy"
    ]

    with pytest.raises(
        DispatchBundleMaterializationBlocked,
        match="bundle_runtime_launch_policy_source_missing",
    ):
        DispatchBundleMaterializationRequest.from_dict(value)


def test_e2_provisional_materialization_blocks_empty_producer_before_runtime_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightcone_spec.orchestration.execution_bundle_materializer as module

    registry = build_industrial_registry(
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
    )
    cell = registry.cells_for("E2")[0]
    plan = release_e2_itl_timestamp_plan(registry, cell)
    plan_path = _write_bound(tmp_path / "e2-itl-plan.json", plan.to_dict()).resolve()
    plan_source = module.BoundJsonSource.bind(
        plan_path,
        semantic_sha256=plan.sha256,
    )
    assignment = SimpleNamespace(
        assignment_id="a" * 64,
        work_item=SimpleNamespace(item_id=cell.cell_id, cell=cell),
    )
    bound_assignment = SimpleNamespace(
        assignment_sha256=assignment.assignment_id,
        cell_id=cell.cell_id,
        sources=(("itl_timestamp_authority_plan", plan_source),),
    )
    authority = SimpleNamespace(registry=registry)
    monkeypatch.setattr(
        module.AssignmentLaunchMaterializationPolicy,
        "from_dict",
        lambda _value: pytest.fail("launch-policy read was reached"),
    )

    with pytest.raises(
        DispatchBundleMaterializationBlocked,
        match=ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON,
    ):
        module._materialize_assignment_provisional(
            assignment=assignment,
            bound_assignment=bound_assignment,
            authority=authority,
        )

    assert not Path(cell.resources.evidence_root).exists()


def test_rejects_caller_execution_summary_and_hash(tmp_path: Path) -> None:
    value, _, _ = _request_value(tmp_path)
    boolean_schema = deepcopy(value)
    boolean_schema["schema_version"] = True
    with pytest.raises(TypeError, match="schema.*integer"):
        DispatchBundleMaterializationRequest.from_dict(boolean_schema)

    forged = deepcopy(value)
    forged["assignments"][0]["execution_plan_summary"] = "/tmp/forged.json"

    with pytest.raises(ValueError, match="fields differ"):
        DispatchBundleMaterializationRequest.from_dict(forged)

    forged = deepcopy(value)
    forged["assignments"][0]["assignment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fields differ"):
        DispatchBundleMaterializationRequest.from_dict(forged)


def test_bound_inputs_detect_post_bind_tamper(tmp_path: Path) -> None:
    value, _, _ = _request_value(tmp_path)
    request_path = _write_bound(tmp_path / "tamper-request.json", value)
    bound = bind_dispatch_bundle_materialization_inputs(request_path)
    run_config = next(
        source for role, source in bound.assignments[0].sources if role == "run_config"
    )

    Path(run_config.path).write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        run_config.load()


def test_rejects_caller_or_foreign_run_nonce(tmp_path: Path) -> None:
    value, assignment, dispatch = _request_value(tmp_path)
    nonce_source = next(
        source
        for source in value["assignments"][0]["sources"]
        if source["role"] == "run_nonce_receipt"
    )
    foreign = AssignmentRunNonceReceipt.issue(
        dispatch_plan_sha256=dispatch.sha256,
        assignment=assignment,
    ).to_dict()
    foreign["assignment_sha256"] = "0" * 64
    _write_bound(Path(nonce_source["path"]), foreign)
    request_path = _write_bound(tmp_path / "foreign-nonce-request.json", value)

    with pytest.raises(ValueError, match="differs from the dispatch assignment"):
        bind_dispatch_bundle_materialization_inputs(request_path)


def test_adapted_binding_role_passes_the_structural_gate(tmp_path: Path) -> None:
    value, assignment, _ = _request_value(tmp_path, method="tts")
    request_path = _write_bound(tmp_path / "adapted-request.json", value)

    bound = bind_dispatch_bundle_materialization_inputs(request_path)

    assert bound.assignments[0].assignment_sha256 == assignment.assignment_id
    assert {role for role, _ in bound.assignments[0].sources} >= {
        "trainable_plan_authority_binding",
        "prepared_model_content_release_manifest",
    }


def test_optional_calibration_authority_round_trips_as_a_path(tmp_path: Path) -> None:
    value, _, _ = _request_value(tmp_path)
    calibration = _write_bound(
        tmp_path / "calibration-execution-authority.json",
        {"source-owned": "calibration"},
    )
    value["shared_optional_sources"]["interference_calibration_execution_authority"] = (
        str(calibration.resolve())
    )

    decoded = DispatchBundleMaterializationRequest.from_dict(value)

    assert decoded.to_dict() == value


def test_runtime_envelope_allows_only_the_locked_dependency_alias(
    tmp_path: Path,
) -> None:
    value, _, _ = _request_value(tmp_path)
    alternate = _write_bound(
        tmp_path / "alternate-runtime-envelope.json",
        {"role": "runtime_envelope"},
    )
    runtime_role = next(
        source
        for source in value["assignments"][0]["sources"]
        if source["role"] == "runtime_envelope_artifact"
    )
    runtime_role["path"] = str(alternate.resolve())

    with pytest.raises(
        DispatchBundleMaterializationBlocked,
        match="runtime_envelope_locked_dependency_binding_required",
    ):
        DispatchBundleMaterializationRequest.from_dict(value)


def test_semantic_failure_does_not_create_publication_directory(
    tmp_path: Path,
) -> None:
    from lightcone_spec.orchestration.execution_bundle_materializer import (
        materialize_dispatch_execution_bundles,
    )

    value, _, _ = _request_value(tmp_path)
    request_path = _write_bound(tmp_path / "invalid-semantic-request.json", value)
    output = (tmp_path / "must-not-be-published").resolve()

    with pytest.raises(ValueError, match="industrial registry artifact"):
        materialize_dispatch_execution_bundles(
            request_path,
            output_directory=output,
        )

    assert not output.exists()


def test_interrupted_publication_is_retained_without_a_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightcone_spec.orchestration.execution_bundle_materializer as module

    cell_id = "1" * 64
    output = (tmp_path / "interrupted-publication").resolve()
    request_path = _write_bound(tmp_path / "request.json", {"request": "bound"})
    dispatch_source = SimpleNamespace(load=lambda: {"dispatch": "bound"})
    evidence_root = (tmp_path / "registered-evidence").resolve()
    bound = SimpleNamespace(
        dispatch_plan=dispatch_source,
        assignments=(SimpleNamespace(cell_id=cell_id, output_root=str(evidence_root)),),
        sha256="2" * 64,
        request=SimpleNamespace(),
    )
    assignment = SimpleNamespace(
        work_item=SimpleNamespace(
            item_id=cell_id,
            cell=SimpleNamespace(identity=SimpleNamespace(method="target_only")),
        )
    )
    provisional = SimpleNamespace(
        cell_id=cell_id,
        preflight_execution_plan_materialization=lambda policy, **kwargs: None,
        reconstruct_execution_plan_for_materialization=(
            lambda policy, **kwargs: SimpleNamespace(server_launch=SimpleNamespace())
        ),
    )
    monkeypatch.setattr(
        module, "bind_dispatch_bundle_materialization_inputs", lambda path: bound
    )
    monkeypatch.setattr(
        module,
        "_reconstruct_materialization_authority",
        lambda value: SimpleNamespace(dispatch_plan_source=SimpleNamespace()),
    )
    monkeypatch.setattr(module, "_dispatch_assignments", lambda value: (assignment,))
    monkeypatch.setattr(
        module,
        "_materialize_assignment_provisional",
        lambda **kwargs: (
            provisional,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(module, "server_launch_to_dict", lambda value: {})
    monkeypatch.setattr(
        module,
        "_write_exclusive_bound_json",
        lambda path, value: (_ for _ in ()).throw(RuntimeError("injected crash")),
    )

    with pytest.raises(RuntimeError, match="injected crash"):
        module.materialize_dispatch_execution_bundles(
            request_path,
            output_directory=output,
        )

    assert output.is_dir()
    assert not (output / "dispatch-execution-bundle-manifest.json.sha256").exists()
    with pytest.raises(
        DispatchBundleMaterializationBlocked,
        match="fresh_bundle_publication_directory_required",
    ):
        module.materialize_dispatch_execution_bundles(
            request_path,
            output_directory=output,
        )


def test_publication_directory_rejects_a_broken_symlink(tmp_path: Path) -> None:
    from lightcone_spec.orchestration.execution_bundle_materializer import (
        _preflight_publication_directory,
    )

    output = tmp_path / "publication-symlink"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(
        DispatchBundleMaterializationBlocked,
        match="fresh_bundle_publication_directory_required",
    ):
        _preflight_publication_directory(output)

    assert output.is_symlink()
    assert not (tmp_path / "missing-target").exists()


def test_private_render_method_symlink_is_rejected_without_following(
    tmp_path: Path,
) -> None:
    from lightcone_spec.orchestration.execution_bundle_materializer import (
        _create_fresh_directory,
    )

    publication = (tmp_path / "publication").resolve()
    _create_fresh_directory(publication, label="bundle publication directory")
    render_root = publication / "assignment-runtime"
    _create_fresh_directory(render_root, label="assignment render root")
    victim = (tmp_path / "victim").resolve()
    victim.mkdir()
    (render_root / "target_only").symlink_to(victim, target_is_directory=True)

    with pytest.raises(
        DispatchBundleMaterializationBlocked,
        match="fresh_bundle_publication_directory_required",
    ):
        _create_fresh_directory(
            render_root / "target_only",
            label="assignment method render root",
        )

    assert not tuple(victim.iterdir())


def test_manifest_loader_rejects_missing_commit_marker_and_symlink_path(
    tmp_path: Path,
) -> None:
    from lightcone_spec.orchestration.execution_bundle_materializer import (
        load_materialized_dispatch_execution_bundle_publication,
    )

    manifest = (tmp_path / "dispatch-execution-bundle-manifest.json").resolve()
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source sidecar"):
        load_materialized_dispatch_execution_bundle_publication(manifest)

    target = _write_bound(tmp_path / "target-manifest.json", {})
    link = (tmp_path / "linked-manifest.json").resolve()
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_materialized_dispatch_execution_bundle_publication(link)


def test_schema_v5_manifest_binds_request_nonce_policy_and_bundle_member(
    tmp_path: Path,
) -> None:
    value, _, dispatch = _request_value(tmp_path)
    request_path = _write_bound(tmp_path / "manifest-request.json", value)
    bound = bind_dispatch_bundle_materialization_inputs(request_path)
    assignment = bound.assignments[0]
    sources = {role: source for role, source in assignment.sources}
    receipt = MaterializedAssignmentBundleReceipt(
        assignment_sha256=assignment.assignment_sha256,
        cell_id=assignment.cell_id,
        run_nonce_sha256=assignment.run_nonce_sha256,
        execution_plan_sha256=content_sha256({"plan": "materialized"}),
        launch_policy=sources["launch_policy"],
        run_nonce_receipt=sources["run_nonce_receipt"],
        bundle=bound.request,
    )
    manifest = DispatchExecutionBundleManifest(
        schema_version=1,
        kind="industrial_dispatch_execution_bundle_manifest",
        bundle_schema_version=5,
        materialization_inputs_sha256=bound.sha256,
        request=bound.request,
        dispatch_plan=bound.dispatch_plan,
        assignments=(receipt,),
    )

    assert DispatchExecutionBundleManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.dispatch_plan.semantic_sha256 == dispatch.sha256
    forged = manifest.to_dict()
    forged["bundle_schema_version"] = 4
    with pytest.raises(ValueError, match="manifest is unsupported"):
        DispatchExecutionBundleManifest.from_dict(forged)

    forged = manifest.to_dict()
    forged["bundle_schema_version"] = True
    with pytest.raises(TypeError, match="bundle schema.*integer"):
        DispatchExecutionBundleManifest.from_dict(forged)


def test_itl_construction_source_requires_the_exact_request_path(
    tmp_path: Path,
) -> None:
    import lightcone_spec.orchestration.execution_bundle_materializer as module
    from lightcone_spec.orchestration.execution_bundle import BoundJsonSource

    original = _write_bound(tmp_path / "requested-itl-plan.json", {"plan": "same"})
    alternate = _write_bound(tmp_path / "alternate-itl-plan.json", {"plan": "same"})
    original_source = BoundJsonSource.bind(original)
    alternate_source = BoundJsonSource.bind(alternate)

    assert original_source.canonical_sha256 == alternate_source.canonical_sha256
    assert original_source.file_sha256 == alternate_source.file_sha256
    assert original_source != alternate_source
    published_bundle = SimpleNamespace(itl_timestamp_plan=alternate_source)
    with pytest.raises(ValueError, match="swapped its path-bound ITL"):
        module._require_published_bundle_itl_source(
            published_bundle,
            {"itl_timestamp_authority_plan": original_source},
        )
