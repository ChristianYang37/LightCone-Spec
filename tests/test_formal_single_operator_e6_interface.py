from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightcone_spec.experiments.formal_single_operator_e6_interface as e6
from lightcone_spec.cli.formal_single_operator import _e6_launch_paths
from lightcone_spec.cli.main import main
from lightcone_spec.experiments import formal_single_operator_downstream as downstream
from lightcone_spec.experiments.formal_protocol import E6_MODELS, ProtocolLock
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorE6InterfacePreflightActualValidator,
    formal_single_operator_node_spec,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    StageMaterializationReceipt,
    _e6_cells_from_verified_sources,
)
from lightcone_spec.runtime.native_qualification_runner import (
    NATIVE_RUNTIME_GPU_TEST_NAMES,
    NativeRuntimeQualificationObservation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_e6_campaign_process_timeout_is_source_owned_exact_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        e6,
        "_load_campaign",
        lambda _path: SimpleNamespace(physical_run_count=2, plans=(object(), object())),
    )

    assert (
        e6.formal_single_operator_e6_interface_fit_process_hard_timeout_ns(
            "/bound/campaign.json"
        )
        == 4_620 * 1_000_000_000
    )

    monkeypatch.setattr(
        e6,
        "_load_campaign",
        lambda _path: SimpleNamespace(physical_run_count=1, plans=(object(),)),
    )
    with pytest.raises(ValueError, match="exact-two"):
        e6.formal_single_operator_e6_interface_fit_process_hard_timeout_ns(
            "/bound/campaign.json"
        )


def _binding(path: Path, value: dict[str, object]) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path.resolve(), value)
    return CanonicalJsonProofBinding.bind(path.resolve())


def _row(model: str, index: int) -> e6.FormalSingleOperatorE6CompatibilityRow:
    return e6.FormalSingleOperatorE6CompatibilityRow(
        model=model,
        source_input_sha256=_sha(f"source-{index}"),
        dynamic_artifact_sha256=_sha(f"dynamic-{index}"),
        verified_authority_sha256=_sha(f"authority-{index}"),
        interface_sha256=e6.NEXTN_MTP_INTERFACE_SHA256,
        target_member_id=_sha(f"target-member-{index}"),
        drafter_member_id=_sha(f"drafter-member-{index}"),
        target_model_id=model,
        drafter_model_id=f"fixture/nextn-drafter-{index}",
        target_revision=str(index + 1) * 40,
        drafter_revision=str(index + 3) * 40,
        target_shard_manifest_sha256=_sha(f"target-shards-{index}"),
        drafter_shard_manifest_sha256=_sha(f"drafter-shards-{index}"),
        topology_sha256=_sha("tp2-topology"),
        source_adapter_version=0,
        native_gpu_proof_sha256=_sha(f"native-{index}"),
        distributed_gpu_proof_sha256=_sha(f"distributed-{index}"),
        content_verification_receipt_sha256=_sha("trusted-content"),
        inventory_sha256=_sha("inventory"),
        gpu_uuids=("GPU-0", "GPU-1"),
        terminal_sha256=_sha(f"terminal-{index}"),
    )


def _compatibility() -> e6.FormalSingleOperatorE6CompatibilityReceipt:
    return e6.FormalSingleOperatorE6CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("lock"),
        registry_sha256=_sha("registry"),
        trusted_content_bundle_sha256=_sha("trusted-content"),
        protocol_sha256=e6.E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        inventory_sha256=_sha("inventory"),
        models=tuple(_row(model, index) for index, model in enumerate(E6_MODELS)),
    )


def _protocol_lock() -> ProtocolLock:
    fields = {
        name: _sha(name)
        for name in (
            "patch_manifest_sha256",
            "registry_sha256",
            "english_protocol_sha256",
            "chinese_protocol_sha256",
            "tts_calibration_authority_sha256",
            "chronobelief_authority_sha256",
            "e1_recipe_anchor_authority_sha256",
            "e2_recipe_grid_authority_sha256",
            "formal_runtime_authority_manifest_sha256",
            "native_runtime_qualification_protocol_sha256",
            "native_runtime_qualification_runner_sha256",
            "native_runtime_qualification_test_set_sha256",
            "compile_qualification_protocol_sha256",
            "compile_qualification_runner_sha256",
            "compile_qualification_test_set_sha256",
            "exactness_qualification_protocol_sha256",
            "exactness_qualification_runner_sha256",
            "exactness_qualification_test_set_sha256",
        )
    }
    return ProtocolLock(
        schema_version=5,
        protocol_id="trusted-e6-test",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        offline_release_trust_root_sha256=None,
        prepared_model_content_authorization_sha256=None,
        formal_workload_e3a_authorization_sha256=None,
        formal_workload_e0_authorization_sha256=None,
        burstgpt_shape_authorization_sha256=None,
        content_source_mode="trusted_single_operator",
        trusted_single_operator_content_bundle_sha256=_sha("trusted-content"),
        **fields,
    )


def test_trusted_compatibility_materializes_exact_panel_and_reuses_preflights() -> None:
    compatibility = _compatibility()
    common = {
        "signed_e5_confirmation_sha256": _sha("e5"),
        "signed_model_compatibility_sha256": _sha("auxiliary"),
        "model_compatibility": compatibility,
        "frozen_tts_recipe_sha256": _sha("tts"),
        "lightcone_recipe_sha256": _sha("lightcone"),
    }

    pilot = _e6_cells_from_verified_sources(
        **common,
        block_indices=(0, 1, 2, 3),
    )
    final = _e6_cells_from_verified_sources(
        **common,
        block_indices=tuple(range(4, 16)),
        power_prefix_source_sha256=_sha("power"),
        pilot_materialization_receipt_sha256=_sha("pilot-materialization"),
        pilot_coverage_receipt_sha256=_sha("pilot-coverage"),
    )
    pilot_preflights = tuple(
        row
        for row in pilot
        if row.task == "immutable_metadata_interface_and_fit_preflight"
    )
    final_preflights = tuple(
        row
        for row in final
        if row.task == "immutable_metadata_interface_and_fit_preflight"
    )

    assert len(pilot) == 2 + 60 * 4
    assert len(final) == 60 * 12
    assert tuple(row.model for row in pilot_preflights) == E6_MODELS
    assert final_preflights == ()
    assert all(dict(row.dimensions)["topology"] == "tp2_dp1" for row in pilot)


def test_downstream_schema2_auxiliary_uses_deep_trusted_revalidator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _protocol_lock()
    compatibility = _compatibility()
    value = {
        "schema_version": 2,
        "kind": "formal_single_operator_e6_interface_fit_bundle",
        "trust_mode": "trusted_single_operator_empirical_no_signature",
    }
    observed: dict[str, object] = {}

    def revalidate(candidate: object, *, protocol_lock: ProtocolLock):
        observed["candidate"] = candidate
        observed["lock"] = protocol_lock
        return SimpleNamespace(compatibility=compatibility)

    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_bundle_value",
        revalidate,
    )
    receipt, digest = downstream._e6_compatibility_from_auxiliary(lock, value)
    assert receipt is compatibility
    assert digest == downstream._digest(value)
    assert observed == {"candidate": value, "lock": lock}


def test_exact_model_set_banned_model_gpu_placement_and_cli_launch_gate(
    tmp_path: Path,
) -> None:
    compatibility = _compatibility()
    with pytest.raises(ValueError, match="coverage"):
        e6.FormalSingleOperatorE6CompatibilityReceipt(
            **{
                **compatibility.__dict__,
                "models": tuple(reversed(compatibility.models)),
            }
        )
    with pytest.raises(ValueError, match="identity"):
        e6.FormalSingleOperatorE6CompatibilityRow(
            **{
                **compatibility.models[0].__dict__,
                "model": "Qwen/Qwen3.5-35B-A3B",
                "target_model_id": "Qwen/Qwen3.5-35B-A3B",
            }
        )

    launches = {
        model: str((tmp_path / f"launch-{index}.json").resolve())
        for index, model in enumerate(E6_MODELS)
    }
    assert (
        _e6_launch_paths([f"{model}={launches[model]}" for model in E6_MODELS])
        == launches
    )
    with pytest.raises(ValueError, match="both exact"):
        _e6_launch_paths([f"{E6_MODELS[0]}={launches[E6_MODELS[0]]}"])
    with pytest.raises(ValueError, match="each exact"):
        _e6_launch_paths(
            [
                f"{E6_MODELS[0]}={launches[E6_MODELS[0]]}",
                f"Qwen/Qwen3.5-35B-A3B={launches[E6_MODELS[1]]}",
            ]
        )

    first = _binding(tmp_path / "plan-0.json", {"plan": 0})
    second = _binding(tmp_path / "plan-1.json", {"plan": 1})
    campaign = e6.FormalSingleOperatorE6InterfaceFitCampaign(
        schema_version=1,
        kind="formal_single_operator_e6_interface_fit_campaign",
        protocol_sha256=e6.FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256,
        protocol_lock_sha256=_sha("lock"),
        predecessor_completion_sha256=_sha("predecessor"),
        trusted_content_sha256=_sha("trusted-content"),
        inventory_sha256=_sha("inventory"),
        models=E6_MODELS,
        gpu_uuids=("GPU-0", "GPU-1"),
        plans=(first, second),
        physical_run_count=2,
    )
    assert (
        e6.FormalSingleOperatorE6InterfaceFitCampaign.from_dict(campaign.to_dict())
        == campaign
    )
    with pytest.raises(ValueError, match="campaign identity"):
        e6.FormalSingleOperatorE6InterfaceFitCampaign(
            **{**campaign.__dict__, "gpu_uuids": ("GPU-0", "GPU-0")}
        )


def test_bundle_codec_rejects_terminal_alias_and_mutation(tmp_path: Path) -> None:
    campaign = _binding(tmp_path / "campaign.json", {"campaign": True})
    terminal_0 = _binding(tmp_path / "terminal-0.json", {"terminal": 0})
    terminal_1 = _binding(tmp_path / "terminal-1.json", {"terminal": 1})
    compatibility = _compatibility()
    bundle = e6.FormalSingleOperatorE6InterfaceFitBundle(
        schema_version=2,
        kind="formal_single_operator_e6_interface_fit_bundle",
        trust_mode="trusted_single_operator_empirical_no_signature",
        protocol_sha256=e6.FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256,
        campaign=campaign,
        protocol_lock_sha256=compatibility.protocol_lock_sha256,
        expected_inventory_sha256=compatibility.inventory_sha256,
        verified_ns=10,
        models=E6_MODELS,
        terminals=(terminal_0, terminal_1),
        compatibility=compatibility,
        compatibility_sha256=compatibility.sha256,
        physical_execution_count=2,
        reuse_scope="e6_pilot_and_e6_final",
    )
    assert (
        e6.FormalSingleOperatorE6InterfaceFitBundle.from_dict(bundle.to_dict())
        == bundle
    )
    with pytest.raises(ValueError, match="bundle identity"):
        e6.FormalSingleOperatorE6InterfaceFitBundle(
            **{**bundle.__dict__, "terminals": (terminal_0, terminal_0)}
        )
    mutated = bundle.to_dict()
    mutated["physical_execution_count"] = 3
    with pytest.raises(ValueError, match="bundle identity"):
        e6.FormalSingleOperatorE6InterfaceFitBundle.from_dict(mutated)

    output = (tmp_path / "published-bundle.json").resolve()
    publish_canonical_json_no_replace(output, bundle.to_dict())
    with pytest.raises(RuntimeError, match="already exists"):
        publish_canonical_json_no_replace(output, bundle.to_dict())


def test_pilot_and_final_replay_same_terminal_without_gpu_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _binding(tmp_path / "shared-terminal.json", {"terminal": True})
    auxiliary_path = (tmp_path / "auxiliary.json").resolve()
    auxiliary = _binding(auxiliary_path, {"auxiliary": True})
    lock_path = (tmp_path / "lock.json").resolve()
    lock_binding = _binding(lock_path, {"lock": True})
    source_paths = {
        "e6_pilot": _binding(tmp_path / "pilot-source.json", {"node": "pilot"}),
        "e6_final": _binding(tmp_path / "final-source.json", {"node": "final"}),
    }
    cell_id = _sha("shared-preflight-cell")
    fake_bundle = SimpleNamespace(terminals=(terminal,))
    lock = SimpleNamespace(sha256=_sha("lock"))

    class Source:
        def __init__(self, node: str) -> None:
            self.node = node
            self.protocol_lock_source = SimpleNamespace(
                reopen=lambda **_kwargs: {"lock": True}
            )

        def auxiliary_source_binding(self, kind: str):
            assert kind == "e6_interface_fit"
            from lightcone_spec.experiments.formal_single_operator_stages import (
                FormalSingleOperatorJsonBinding,
            )

            return FormalSingleOperatorJsonBinding.bind(
                auxiliary.absolute_path,
                label="test E6 auxiliary",
            )

    def route(*, execution_source_path: str | Path, materialized_cell_id: str):
        assert materialized_cell_id == cell_id
        node = (
            "e6_pilot"
            if str(execution_source_path) == source_paths["e6_pilot"].absolute_path
            else "e6_final"
        )
        return (
            Source(node),
            SimpleNamespace(
                cell_id=cell_id,
                model=E6_MODELS[0],
                task="immutable_metadata_interface_and_fit_preflight",
            ),
            SimpleNamespace(physical_kind="e6_interface_preflight"),
        )

    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_single_operator_run_dispatch."
        "route_formal_single_operator_cell",
        route,
    )
    monkeypatch.setattr(e6, "protocol_lock_from_dict", lambda _value: lock)
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_bundle_value",
        lambda _value, *, protocol_lock: fake_bundle,
    )
    monkeypatch.setattr(e6, "terminal_for_model", lambda _bundle, _model: terminal)

    plans = tuple(
        e6.materialize_formal_single_operator_e6_interface_replay_plan(
            execution_source_path=source_paths[node].absolute_path,
            materialized_cell_id=cell_id,
            output_path=(tmp_path / f"{node}-replay.json").resolve(),
        )
        for node in ("e6_pilot", "e6_final")
    )
    assert all(row.shared_terminal == terminal for row in plans)
    assert all(row.additional_gpu_runs == 0 for row in plans)
    assert all(row.physical_execution_reused is True for row in plans)
    for node, plan in zip(("e6_pilot", "e6_final"), plans, strict=True):
        assert (
            e6.revalidate_formal_single_operator_e6_interface_replay_plan(
                tmp_path / f"{node}-replay.json"
            )
            == plan
        )
    assert lock_binding.semantic_sha256 != auxiliary.semantic_sha256


def _junit(names: tuple[str, ...]) -> bytes:
    cases = "".join(f'<testcase name="{name}"/>' for name in names)
    return f'<testsuite tests="8">{cases}</testsuite>'.encode()


def test_physical_executor_routes_exact_eight_tests_and_is_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "evidence").resolve()
    evidence.mkdir()
    plan_path = (evidence / "plan.json").resolve()
    plan_binding = _binding(plan_path, {"plan": "fixture"})
    assignment_binding = _binding(
        evidence / "assignment.json", {"assignment": "fixture"}
    )
    launch_binding = _binding(evidence / "launch.json", {"launch": "fixture"})
    gpu_uuids = ("GPU-0", "GPU-1")

    assignment = SimpleNamespace(
        suite_id="nextn_tp2",
        runner_protocol_sha256=(
            e6.NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S["nextn_tp2"]
        ),
        sha256=_sha("assignment"),
        launch_manifest=launch_binding,
        inventory_sha256=_sha("inventory"),
        gpu_uuids=gpu_uuids,
        python_executable="/usr/bin/python3",
        evidence_directory=str(evidence),
        evidence_path=lambda suffix: evidence / f"native-nextn_tp2-fixture.{suffix}",
    )
    launch = SimpleNamespace(
        sha256=launch_binding.semantic_sha256,
        patched_sglang_checkout=str(tmp_path.resolve()),
        prepared_model_content_manifest_sha256=_sha("prepared-content"),
        run_config_semantic_sha256=_sha("run-config"),
        child_environment=dict,
    )
    plan = SimpleNamespace(
        sha256=plan_binding.semantic_sha256,
        native_assignment=assignment_binding,
        launch_manifest=launch_binding,
        model=E6_MODELS[0],
        evidence_directory=str(evidence),
        target_member_sha256=_sha("target-member"),
        drafter_member_sha256=_sha("drafter-member"),
        target_shard_manifest_sha256=_sha("target-shards"),
        drafter_shard_manifest_sha256=_sha("drafter-shards"),
        interface_sha256=e6.NEXTN_MTP_INTERFACE_SHA256,
        topology_sha256=_sha("topology"),
        gpu_uuids=gpu_uuids,
    )
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_plan",
        lambda _path: plan,
    )
    monkeypatch.setattr(
        e6.NativeRuntimeQualificationAssignment,
        "load",
        classmethod(lambda _cls, _path: assignment),
    )
    monkeypatch.setattr(
        e6.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: launch),
    )
    snapshot_index = 0

    def publish_snapshot(_assignment: object, *, phase: str):
        nonlocal snapshot_index
        snapshot_index += 1
        return _binding(
            evidence / f"snapshot-{snapshot_index}.json",
            {"phase": phase},
        )

    monkeypatch.setattr(e6, "_publish_snapshot", publish_snapshot)
    monkeypatch.setattr(
        e6,
        "_validate_gpu_snapshot",
        lambda _value, **_kwargs: {
            "status": "AVAILABLE",
            "compute_process_rows": [],
        },
    )
    monkeypatch.setattr(e6, "_process_group_exists", lambda _pid: False)
    observed: dict[str, object] = {}

    class Process:
        pid = 43210
        returncode = 0

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["environment"] = kwargs["env"]

        def wait(self, *, timeout: float) -> int:
            assert timeout == 1_800.0
            environment = observed["environment"]
            names = NATIVE_RUNTIME_GPU_TEST_NAMES["nextn_tp2"]
            junit = next(
                item for item in observed["command"] if item.startswith("--junitxml=")
            )
            Path(junit.split("=", 1)[1]).write_bytes(_junit(names))
            artifacts = {
                "native": _binding(
                    evidence / "nextn_tp2.live-native-terminal.json",
                    {"kind": "native"},
                ),
                "itl": _binding(
                    evidence / "nextn_tp2.live-native-itl.json", {"kind": "itl"}
                ),
                "graph": _binding(
                    evidence / "nextn_tp2.live-graph.json", {"kind": "graph"}
                ),
                "worker": _binding(
                    evidence / "nextn_tp2.live-worker-hook.json",
                    {"kind": "worker"},
                ),
                "rank0": _binding(evidence / "nextn_tp2.rank-0.json", {"rank": 0}),
                "rank1": _binding(evidence / "nextn_tp2.rank-1.json", {"rank": 1}),
                "server": _binding(
                    evidence / "nextn_tp2.live-server-receipt.json",
                    {"kind": "server"},
                ),
            }
            server_log = assignment.evidence_path("live-observation.json").with_suffix(
                ".live-server.log"
            )
            server_log.write_bytes(b"server\n")
            observation = NativeRuntimeQualificationObservation(
                schema_version=1,
                kind="source_owned_native_runtime_live_observation",
                suite_id="nextn_tp2",
                runner_protocol_sha256=assignment.runner_protocol_sha256,
                assignment_sha256=assignment.sha256,
                source_capability_sha256=e6._source_capability_sha256("nextn_tp2"),
                launch_manifest_sha256=launch_binding.semantic_sha256,
                prepared_model_content_manifest_sha256=(
                    launch.prepared_model_content_manifest_sha256
                ),
                run_config_sha256=launch.run_config_semantic_sha256,
                inventory_sha256=assignment.inventory_sha256,
                gpu_uuids=gpu_uuids,
                completed_test_names=names,
                server_process_ids=(10, 11),
                rank_terminal_sha256s=(
                    artifacts["rank0"].semantic_sha256,
                    artifacts["rank1"].semantic_sha256,
                ),
                live_server_receipt_sha256=artifacts["server"].semantic_sha256,
                native_terminal_sha256=artifacts["native"].semantic_sha256,
                native_itl_pointer_sha256=artifacts["itl"].semantic_sha256,
                graph_observation_sha256=artifacts["graph"].semantic_sha256,
                worker_hook_observation_sha256=artifacts["worker"].semantic_sha256,
                scored_request_inputs_sha256=_sha("requests"),
                completed_request_count=8,
                worker_hook_invocation_count=8,
                graph_replay_count=8,
                native_timestamp_count=32,
                started_ns=10,
                finished_ns=20,
                actual_sglang_server=True,
                component_only=False,
            )
            publish_canonical_json_no_replace(
                Path(environment["LIGHTCONE_NATIVE_QUALIFICATION_OBSERVATION_PATH"]),
                observation.to_dict(),
            )
            return 0

    monkeypatch.setattr(e6.subprocess, "Popen", Process)
    terminal = e6.execute_formal_single_operator_e6_interface_fit_plan(plan_path)
    command = observed["command"]
    assert command[:4] == (assignment.python_executable, "-m", "pytest", "-q")
    assert (
        tuple(item.split("::")[-1] for item in command[4:12])
        == (NATIVE_RUNTIME_GPU_TEST_NAMES["nextn_tp2"])
    )
    assert terminal.model == E6_MODELS[0]
    assert terminal.physical_execution_count == 1
    assert terminal.status == "COMPLETE"
    with pytest.raises((FileExistsError, RuntimeError)):
        e6.execute_formal_single_operator_e6_interface_fit_plan(plan_path)
    (evidence / "nextn_tp2.live-worker-hook.json").write_text(
        '{"kind":"mutated"}\n',
        encoding="utf-8",
    )
    with pytest.raises((ValueError, RuntimeError), match="changed|differs"):
        e6.revalidate_formal_single_operator_e6_interface_fit_terminal(
            evidence / "e6-interface-fit-terminal.json"
        )


def test_physical_executor_refuses_competing_dual_gpu_campaign_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "model-0").resolve()
    evidence.mkdir()
    plan_path = (evidence / "plan.json").resolve()
    plan_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_plan",
        lambda _path: SimpleNamespace(evidence_directory=str(evidence)),
    )

    def busy(_descriptor: int, _operation: int) -> None:
        raise BlockingIOError

    monkeypatch.setattr(e6.fcntl, "flock", busy)
    with pytest.raises(
        e6.FormalSingleOperatorE6InterfaceFitBlocked,
        match="campaign_gang_busy",
    ):
        e6.execute_formal_single_operator_e6_interface_fit_plan(plan_path)


def test_current_actual_validator_accepts_empirical_terminal_and_rejects_foreign_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _protocol_lock()
    compatibility = _compatibility().models[0]
    cells = _e6_cells_from_verified_sources(
        signed_e5_confirmation_sha256=_sha("e5"),
        signed_model_compatibility_sha256=_sha("auxiliary"),
        model_compatibility=_compatibility(),
        frozen_tts_recipe_sha256=_sha("tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        block_indices=(0, 1, 2, 3),
    )
    cell = next(
        row
        for row in cells
        if row.task == "immutable_metadata_interface_and_fit_preflight"
        and row.model == E6_MODELS[0]
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E6",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("upstream"),),
        source_decision_sha256=_sha("source"),
        materialization_rule="trusted_e6_interface_validator_test",
        expected_cell_count=1,
        cells=(cell,),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    terminal_path = (tmp_path / "terminal.json").resolve()
    terminal_binding = _binding(
        terminal_path,
        {"kind": "formal_single_operator_e6_interface_fit_terminal"},
    )
    observation = _binding(tmp_path / "observation.json", {"observation": True})
    plan = SimpleNamespace(
        sha256=compatibility.source_input_sha256,
        interface_sha256=compatibility.interface_sha256,
    )
    terminal = SimpleNamespace(
        sha256=terminal_binding.semantic_sha256,
        model=cell.model,
        status="COMPLETE",
        started_ns=10,
        finished_ns=20,
        live_observation=observation,
        distributed_gpu_proof_sha256=(compatibility.distributed_gpu_proof_sha256),
        native_gpu_proof_sha256=compatibility.native_gpu_proof_sha256,
        trusted_authority_sha256=compatibility.verified_authority_sha256,
        plan=SimpleNamespace(absolute_path=str((tmp_path / "plan.json").resolve())),
    )
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_terminal",
        lambda _path: terminal,
    )
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_plan",
        lambda _path: plan,
    )
    monkeypatch.setattr(
        e6,
        "compatibility_row_for_terminal",
        lambda _plan, _terminal: compatibility,
    )

    validator = FormalSingleOperatorE6InterfacePreflightActualValidator(lock)
    result = validator.validate(
        path=terminal_path,
        node=formal_single_operator_node_spec("e6_pilot"),
        materialization=materialization,
        cell=cell,
    )
    assert result.status == "COMPLETE"
    assert result.reducer_payload["descriptive_only"] is True
    assert result.reducer_payload["trust_mode"] == (
        "trusted_single_operator_empirical_no_signature"
    )
    foreign = next(
        row
        for row in cells
        if row.task == "immutable_metadata_interface_and_fit_preflight"
        and row.model == E6_MODELS[1]
    )
    with pytest.raises(ValueError, match="differs from the current cell"):
        validator.validate(
            path=terminal_path,
            node=formal_single_operator_node_spec("e6_pilot"),
            materialization=materialization,
            cell=foreign,
        )


def test_standalone_e6_prepare_execute_finalize_cli_routes_exact_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    plans = tuple(
        SimpleNamespace(absolute_path=str((tmp_path / f"plan-{index}.json").resolve()))
        for index in range(2)
    )
    campaign = SimpleNamespace(
        sha256=_sha("campaign"),
        gpu_uuids=("GPU-0", "GPU-1"),
        models=E6_MODELS,
        physical_run_count=2,
        plans=plans,
    )
    terminal = SimpleNamespace(
        model=E6_MODELS[0],
        physical_execution_count=1,
        status="COMPLETE",
        sha256=_sha("terminal"),
    )
    bundle = SimpleNamespace(
        sha256=_sha("bundle"),
        models=E6_MODELS,
        physical_execution_count=2,
        reuse_scope="e6_pilot_and_e6_final",
        trust_mode="trusted_single_operator_empirical_no_signature",
    )
    observed: dict[str, object] = {}

    def prepare(**kwargs):
        observed["prepare"] = kwargs
        return campaign

    def execute(path: str):
        observed["execute"] = path
        return terminal

    def finalize(**kwargs):
        observed["finalize"] = kwargs
        return bundle

    monkeypatch.setattr(
        e6, "materialize_formal_single_operator_e6_interface_fit_campaign", prepare
    )
    monkeypatch.setattr(
        e6, "execute_formal_single_operator_e6_interface_fit_plan", execute
    )
    monkeypatch.setattr(
        e6, "finalize_formal_single_operator_e6_interface_fit_bundle", finalize
    )
    launch_paths = {
        model: str((tmp_path / f"launch-{index}.json").resolve())
        for index, model in enumerate(E6_MODELS)
    }
    prepare_argv = [
        "formal-single-operator",
        "prepare-e6-interface-fit",
        "--protocol-lock",
        str((tmp_path / "lock.json").resolve()),
        "--predecessor-completion",
        str((tmp_path / "e5.json").resolve()),
        "--content-source",
        str((tmp_path / "content.json").resolve()),
        "--output-root",
        str((tmp_path / "campaign-root").resolve()),
    ]
    for model in E6_MODELS:
        prepare_argv.extend(("--launch", f"{model}={launch_paths[model]}"))
    assert main(prepare_argv) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["physical_run_count"] == 2
    assert observed["prepare"]["launch_manifest_paths"] == launch_paths

    plan_path = str((tmp_path / "plan.json").resolve())
    assert (
        main(
            [
                "formal-single-operator",
                "execute-e6-interface-fit",
                "--plan",
                plan_path,
            ]
        )
        == 0
    )
    executed = json.loads(capsys.readouterr().out)
    assert executed["status"] == "COMPLETE"
    assert observed["execute"] == plan_path

    output_path = str((tmp_path / "bundle.json").resolve())
    assert (
        main(
            [
                "formal-single-operator",
                "finalize-e6-interface-fit",
                "--campaign",
                str((tmp_path / "campaign.json").resolve()),
                "--output",
                output_path,
            ]
        )
        == 0
    )
    finalized = json.loads(capsys.readouterr().out)
    assert finalized["physical_execution_count"] == 2
    assert observed["finalize"]["output_path"] == output_path


def test_generic_run_cli_prepares_and_replays_shared_e6_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_run_dispatch as dispatch,
    )
    from lightcone_spec.runtime import formal_single_operator as runtime

    cell_id = _sha("e6-preflight-cell")
    source = SimpleNamespace(node="e6_pilot", stage="E6")
    route = SimpleNamespace(physical_kind="e6_interface_preflight")
    monkeypatch.setattr(
        dispatch,
        "route_formal_single_operator_cell",
        lambda **_kwargs: (source, SimpleNamespace(cell_id=cell_id), route),
    )
    run_root = (tmp_path / "run").resolve()

    def create_run_directory(**_kwargs):
        run_root.mkdir(mode=0o700)
        return run_root

    monkeypatch.setattr(
        runtime, "create_formal_single_operator_run_directory", create_run_directory
    )
    shared_terminal = str((tmp_path / "shared-terminal.json").resolve())
    replay = SimpleNamespace(
        additional_gpu_runs=0,
        physical_execution_reused=True,
        shared_terminal=SimpleNamespace(absolute_path=shared_terminal),
        sha256=_sha("replay"),
    )
    observed: dict[str, object] = {}

    def materialize(**kwargs):
        observed.update(kwargs)
        return replay

    monkeypatch.setattr(
        e6,
        "materialize_formal_single_operator_e6_interface_replay_plan",
        materialize,
    )
    source_path = str((tmp_path / "source.json").resolve())
    assert (
        main(
            [
                "formal-single-operator",
                "prepare-run",
                "--repository-root",
                str(tmp_path.resolve()),
                "--execution-source",
                source_path,
                "--cell",
                cell_id,
                "--output-root",
                str((tmp_path / "output").resolve()),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["additional_gpu_runs"] == 0
    assert output["shared_terminal"] == shared_terminal
    assert observed["execution_source_path"] == source_path
    assert observed["materialized_cell_id"] == cell_id

    replay_path = (tmp_path / "replay-plan.json").resolve()
    replay_binding = _binding(
        replay_path,
        {"kind": "formal_single_operator_e6_interface_replay_plan"},
    )
    replay.materialized_cell_id = cell_id
    terminal = SimpleNamespace(status="COMPLETE", sha256=_sha("terminal"))
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_replay_plan",
        lambda _path: replay,
    )
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_terminal",
        lambda _path: terminal,
    )
    for operation in ("execute-run", "finalize-run"):
        assert (
            main(
                [
                    "formal-single-operator",
                    operation,
                    "--repository-root",
                    str(tmp_path.resolve()),
                    "--run-plan",
                    replay_binding.absolute_path,
                ]
            )
            == 0
        )
        raw = capsys.readouterr().out.strip()
        if operation == "execute-run":
            assert json.loads(raw)["additional_gpu_runs"] == 0
        else:
            assert raw == terminal.sha256


def test_trusted_serving_authority_codec_is_explicitly_unmeasured_and_tamper_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
        TrustedSingleOperatorContentBundleBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorJsonBinding,
    )

    bindings = {
        name: _binding(tmp_path / f"{name}.json", {"kind": name})
        for name in (
            "execution-source",
            "auxiliary",
            "interface-plan",
            "interface-terminal",
            "inventory",
            "doctor",
        )
    }
    content_path = (tmp_path / "content.json").resolve()
    content_path.write_text('{"kind":"content"}\n', encoding="utf-8")
    fake_content = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(fake_content, "runtime_binding_status", "BOUND")
    object.__setattr__(fake_content, "semantic_sha256", _sha("content"))
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: fake_content,
    )
    trusted_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(content_path),
        size=content_path.stat().st_size,
        raw_sha256=hashlib.sha256(content_path.read_bytes()).hexdigest(),
        semantic_sha256=_sha("content"),
        runtime_binding_status="BOUND",
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=trusted_binding,
    )
    row = _compatibility().models[0]
    authority = e6.FormalSingleOperatorTrustedNextnTp2ServingAuthority(
        schema_version=1,
        kind="formal_single_operator_e6_trusted_serving_authority",
        protocol_sha256=(
            e6.FORMAL_SINGLE_OPERATOR_E6_TRUSTED_SERVING_AUTHORITY_PROTOCOL_SHA256
        ),
        trust_mode="trusted_single_operator_empirical_no_signature",
        formal_measured_authorization=False,
        execution_source=bindings["execution-source"],
        execution_source_sha256=_sha("execution-source-identity"),
        materialized_cell_id=_sha("cell"),
        node="e6_pilot",
        auxiliary_bundle=FormalSingleOperatorJsonBinding.bind(
            bindings["auxiliary"].absolute_path,
            label="test auxiliary",
        ),
        auxiliary_bundle_sha256=bindings["auxiliary"].semantic_sha256,
        protocol_lock_sha256=_sha("lock"),
        compatibility_receipt_sha256=_compatibility().sha256,
        compatibility_row_sha256=row.sha256,
        interface_fit_plan=bindings["interface-plan"],
        interface_fit_plan_sha256=bindings["interface-plan"].semantic_sha256,
        interface_fit_terminal=bindings["interface-terminal"],
        interface_fit_terminal_sha256=(bindings["interface-terminal"].semantic_sha256),
        content_source=content_source,
        trusted_content_bundle_sha256=_sha("content"),
        inventory=bindings["inventory"],
        inventory_sha256=bindings["inventory"].semantic_sha256,
        doctor=bindings["doctor"],
        doctor_sha256=bindings["doctor"].semantic_sha256,
        model=row.model,
        target_model_id=row.target_model_id,
        drafter_model_id=row.drafter_model_id,
        target_revision=row.target_revision,
        drafter_revision=row.drafter_revision,
        target_member_sha256=row.target_member_id,
        drafter_member_sha256=row.drafter_member_id,
        target_shard_manifest_sha256=row.target_shard_manifest_sha256,
        drafter_shard_manifest_sha256=row.drafter_shard_manifest_sha256,
        interface_sha256=row.interface_sha256,
        topology_sha256=row.topology_sha256,
        source_adapter_version=0,
        gpu_uuids=row.gpu_uuids,
        native_gpu_proof_sha256=row.native_gpu_proof_sha256,
        distributed_gpu_proof_sha256=row.distributed_gpu_proof_sha256,
        empirical_authority_sha256=row.verified_authority_sha256,
        junit_raw_sha256=_sha("junit"),
        qualified_test_count=8,
        passed_test_count=8,
        failed_test_count=0,
        error_test_count=0,
        skipped_test_count=0,
        physical_execution_count=1,
    )
    assert (
        e6.FormalSingleOperatorTrustedNextnTp2ServingAuthority.from_dict(
            authority.to_dict()
        )
        == authority
    )
    assert authority.formal_measured_authorization is False
    assert authority.trust_mode == "trusted_single_operator_empirical_no_signature"
    mutated = authority.to_dict()
    mutated["formal_measured_authorization"] = True
    with pytest.raises(ValueError, match="authority identity"):
        e6.FormalSingleOperatorTrustedNextnTp2ServingAuthority.from_dict(mutated)
    Path(bindings["interface-terminal"].absolute_path).write_text(
        '{"kind":"mutated"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed"):
        e6.FormalSingleOperatorTrustedNextnTp2ServingAuthority.from_dict(
            authority.to_dict()
        )


def test_prepared_e6_serving_plan_binds_derived_empirical_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_run_dispatch
    from lightcone_spec.orchestration import formal_physical_dispatch as physical

    root = (tmp_path / "run").resolve()
    root.mkdir()
    root.chmod(0o700)
    input_path = root / "formal-single-operator-prepared-downstream-inputs.json"
    launch_binding = _binding(root / "launch.json", {"kind": "launch"})
    inventory_binding = _binding(root / "inventory.json", {"kind": "inventory"})
    schedule_binding = _binding(root / "schedule.json", {"kind": "schedule"})
    execution_binding = _binding(root / "execution.json", {"kind": "execution"})
    input_binding = _binding(input_path, {"kind": "prepared-input"})
    cell = SimpleNamespace(
        cell_id=_sha("e6-serving-cell"),
        stage="E6",
        task="LiveCodeBench",
    )
    materialization = SimpleNamespace(stage="E6", sha256=_sha("materialization"))
    inventory = SimpleNamespace(
        sha256=inventory_binding.semantic_sha256,
        devices=(SimpleNamespace(uuid="GPU-0"), SimpleNamespace(uuid="GPU-1")),
    )
    content_source = SimpleNamespace(mode="trusted_single_operator")
    inputs = SimpleNamespace(
        kind="formal_single_operator_prepared_downstream_run_plan_inputs",
        schema_version=2,
        sha256=input_binding.semantic_sha256,
        private_output_root=str(root),
        stage="E6",
        materialized_cell_id=cell.cell_id,
        materialization=SimpleNamespace(),
        materialization_sha256=materialization.sha256,
        compile_launch_manifest=launch_binding,
        inventory=inventory_binding,
        request_schedule_receipt=schedule_binding,
        execution_binding_sha256=_sha("execution-binding"),
        subject_sha256=_sha("subject"),
        content_verification_receipt=None,
        content_source_binding=content_source,
        execution_source=execution_binding,
    )
    launch = SimpleNamespace(
        sha256=launch_binding.semantic_sha256,
        run_config_path=str((root / "config.json").resolve()),
        inventory_sha256=inventory.sha256,
        gpu_uuids=("GPU-0", "GPU-1"),
    )
    config = SimpleNamespace(
        method="target_only",
        model=SimpleNamespace(algorithm="NEXTN"),
        runtime=SimpleNamespace(topology_mode="tp2_dp1"),
    )
    warmup_request = SimpleNamespace(request_id="warmup-request-0")
    request = SimpleNamespace(request_id="request-0")
    schedule_rows = (
        SimpleNamespace(phase="warmup", request=warmup_request),
        SimpleNamespace(phase="scored", request=request),
    )
    schedule = SimpleNamespace(
        schema_version=7,
        sha256=schedule_binding.semantic_sha256,
        execution_binding_sha256=inputs.execution_binding_sha256,
        subject_sha256=inputs.subject_sha256,
        materialized_cell_id=cell.cell_id,
        materialization=inputs.materialization,
        content_verification_receipt=None,
        content_source_binding=content_source,
        compile_launch_manifest=launch_binding,
        topology_mode="tp2_dp1",
        schedule_source=SimpleNamespace(load=lambda: {"kind": "source"}),
        e5_arrival_plan=None,
    )
    schedule_source = SimpleNamespace(
        sha256=_sha("schedule-source"),
        e5_arrival_plan=None,
        arrival_policy="closed_loop_zero_think",
        max_running_requests=1,
        materialization_receipt_sha256=materialization.sha256,
        materialized_cell_id=cell.cell_id,
        subject_sha256=inputs.subject_sha256,
        topology_mode="tp2_dp1",
    )
    schedule.schedule_source.semantic_sha256 = schedule_source.sha256
    authority_sha = _sha("trusted-nextn-serving-authority")
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        formal_single_operator_run_dispatch,
        "revalidate_formal_single_operator_prepared_downstream_run_plan_inputs",
        lambda _path, *, current_ns: inputs,
    )
    monkeypatch.setattr(
        physical.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: launch),
    )
    monkeypatch.setattr(
        physical,
        "_reopen_stage_materialization",
        lambda _value: materialization,
    )
    monkeypatch.setattr(
        physical,
        "_materialized_cell",
        lambda *_args, **_kwargs: cell,
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.gpu_pool.GpuInventory.from_dict",
        classmethod(lambda _cls, _value: inventory),
    )
    monkeypatch.setattr(physical, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        physical.FormalServingRequestScheduleReceipt,
        "from_dict",
        classmethod(lambda _cls, _value: schedule),
    )
    monkeypatch.setattr(
        physical.FormalServingRequestScheduleSource,
        "from_dict",
        classmethod(lambda _cls, _value: schedule_source),
    )
    monkeypatch.setattr(
        physical,
        "formal_serving_request_schedule_rows",
        lambda _schedule: schedule_rows,
    )

    def derive(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(sha256=authority_sha)

    monkeypatch.setattr(
        e6,
        "derive_formal_single_operator_trusted_nextn_tp2_serving_authority",
        derive,
    )
    plan = (
        physical._materialize_formal_single_operator_prepared_direct_serving_run_plan(
            inputs=inputs,
            input_binding=input_binding,
            expected_input_name=input_path.name,
        )
    )
    assert plan.nextn_tp2_authority_sha256 == authority_sha
    assert observed["execution_source_path"] == execution_binding.absolute_path
    assert observed["materialized_cell_id"] == cell.cell_id
    assert observed["compile_launch_manifest"] == launch_binding
    assert observed["inventory"] == inventory_binding
    assert observed["content_source"] is content_source


def test_derive_trusted_e6_serving_authority_joins_cell_content_and_8_of_8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_run_dispatch
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
        TrustedSingleOperatorContentBundleBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorJsonBinding,
    )
    from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding

    execution = _binding(tmp_path / "execution.json", {"kind": "execution"})
    auxiliary_binding = _binding(tmp_path / "auxiliary.json", {"kind": "aux"})
    auxiliary = FormalSingleOperatorJsonBinding.bind(
        auxiliary_binding.absolute_path,
        label="test E6 serving auxiliary",
    )
    inventory_binding = _binding(tmp_path / "inventory.json", {"kind": "inventory"})
    doctor_binding = _binding(tmp_path / "doctor.json", {"kind": "doctor"})
    launch_binding = _binding(tmp_path / "launch.json", {"kind": "launch"})
    plan_binding = _binding(tmp_path / "interface-plan.json", {"kind": "plan"})
    terminal_binding = _binding(
        tmp_path / "interface-terminal.json", {"kind": "terminal"}
    )
    junit_path = (tmp_path / "junit.xml").resolve()
    junit_path.write_text('<testsuite tests="8" failures="0"/>\n', encoding="utf-8")
    junit = EvidenceFileBinding.bind(junit_path, label="test E6 JUnit")

    content_path = (tmp_path / "content.json").resolve()
    content_path.write_text('{"kind":"content"}\n', encoding="utf-8")
    content_sha = _sha("trusted-content")
    fake_content = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(fake_content, "runtime_binding_status", "BOUND")
    object.__setattr__(fake_content, "semantic_sha256", content_sha)
    object.__setattr__(
        fake_content,
        "runtime_observations",
        SimpleNamespace(
            inventory=SimpleNamespace(
                absolute_path=inventory_binding.absolute_path,
                raw_sha256=inventory_binding.raw_sha256,
                semantic_sha256=inventory_binding.semantic_sha256,
                size=inventory_binding.size,
            ),
            doctor=SimpleNamespace(
                absolute_path=doctor_binding.absolute_path,
                raw_sha256=doctor_binding.raw_sha256,
                semantic_sha256=doctor_binding.semantic_sha256,
                size=doctor_binding.size,
            ),
        ),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: fake_content,
    )
    trusted_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(content_path),
        size=content_path.stat().st_size,
        raw_sha256=hashlib.sha256(content_path.read_bytes()).hexdigest(),
        semantic_sha256=content_sha,
        runtime_binding_status="BOUND",
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=trusted_binding,
    )
    base = _row(E6_MODELS[0], 0)
    row = e6.FormalSingleOperatorE6CompatibilityRow(
        **{
            **base.__dict__,
            "source_input_sha256": plan_binding.semantic_sha256,
            "terminal_sha256": terminal_binding.semantic_sha256,
            "content_verification_receipt_sha256": content_sha,
            "inventory_sha256": inventory_binding.semantic_sha256,
        }
    )
    other = _row(E6_MODELS[1], 1)
    compatibility = e6.FormalSingleOperatorE6CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_protocol_lock().sha256,
        registry_sha256=_protocol_lock().registry_sha256,
        trusted_content_bundle_sha256=content_sha,
        protocol_sha256=e6.E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        inventory_sha256=inventory_binding.semantic_sha256,
        models=(
            row,
            e6.FormalSingleOperatorE6CompatibilityRow(
                **{
                    **other.__dict__,
                    "content_verification_receipt_sha256": content_sha,
                    "inventory_sha256": inventory_binding.semantic_sha256,
                }
            ),
        ),
    )
    cells = _e6_cells_from_verified_sources(
        signed_e5_confirmation_sha256=_sha("e5"),
        signed_model_compatibility_sha256=auxiliary.semantic_sha256,
        model_compatibility=compatibility,
        frozen_tts_recipe_sha256=_sha("tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        block_indices=(0, 1, 2, 3),
    )
    cell = next(
        candidate
        for candidate in cells
        if candidate.model == E6_MODELS[0]
        and candidate.task == "LiveCodeBench"
        and candidate.method_role == "Target-only"
    )
    lock = _protocol_lock()

    class Source:
        node = "e6_pilot"
        schema_version = 3
        stage = "E6"
        sha256 = _sha("execution-source")
        content_source_binding = content_source
        protocol_lock_source = SimpleNamespace(
            reopen=lambda **_kwargs: {"kind": "lock"}
        )

        @staticmethod
        def auxiliary_source_binding(kind: str):
            assert kind == "e6_interface_fit"
            return auxiliary

    source = Source()
    terminal = SimpleNamespace(
        sha256=terminal_binding.semantic_sha256,
        plan=plan_binding,
        trusted_authority_sha256=row.verified_authority_sha256,
        native_gpu_proof_sha256=row.native_gpu_proof_sha256,
        distributed_gpu_proof_sha256=row.distributed_gpu_proof_sha256,
        status="COMPLETE",
        physical_execution_count=1,
        junit_xml=junit,
    )
    plan = SimpleNamespace(
        sha256=plan_binding.semantic_sha256,
        model=row.model,
        content_source=content_source,
        inventory=inventory_binding,
        doctor=doctor_binding,
    )
    bundle = SimpleNamespace(
        sha256=auxiliary.semantic_sha256,
        compatibility=compatibility,
        expected_inventory_sha256=inventory_binding.semantic_sha256,
    )
    inventory = SimpleNamespace(sha256=inventory_binding.semantic_sha256)
    launch = SimpleNamespace(
        sha256=launch_binding.semantic_sha256,
        schema_version=2,
        formal_stage="E6",
        content_source_binding=content_source,
        inventory_sha256=inventory.sha256,
        gpu_uuids=row.gpu_uuids,
        run_config_path=str((tmp_path / "config.json").resolve()),
        target_content_member_id=row.target_member_id,
        drafter_content_member_id=row.drafter_member_id,
    )
    config = SimpleNamespace(
        model=SimpleNamespace(
            algorithm="NEXTN",
            target=row.target_model_id,
            drafter=row.drafter_model_id,
            target_revision=row.target_revision,
            drafter_revision=row.drafter_revision,
        ),
        runtime=SimpleNamespace(
            topology_mode="tp2_dp1",
            tensor_parallel_size=2,
            data_parallel_size=1,
        ),
    )
    monkeypatch.setattr(
        formal_single_operator_run_dispatch,
        "route_formal_single_operator_cell",
        lambda **_kwargs: (
            source,
            cell,
            SimpleNamespace(physical_kind="serving"),
        ),
    )
    monkeypatch.setattr(e6, "protocol_lock_from_dict", lambda _value: lock)
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_bundle_value",
        lambda _value, *, protocol_lock: bundle,
    )
    monkeypatch.setattr(
        e6, "terminal_for_model", lambda _bundle, _model: terminal_binding
    )
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_terminal",
        lambda _path: terminal,
    )
    monkeypatch.setattr(
        e6,
        "revalidate_formal_single_operator_e6_interface_fit_plan",
        lambda _path: plan,
    )
    monkeypatch.setattr(
        e6,
        "compatibility_row_for_terminal",
        lambda _plan, _terminal: row,
    )
    monkeypatch.setattr(
        e6.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: launch),
    )
    monkeypatch.setattr(
        e6.GpuInventory,
        "from_dict",
        classmethod(lambda _cls, _value: inventory),
    )
    monkeypatch.setattr(e6, "load_run_config", lambda _path: config)

    authority = e6.derive_formal_single_operator_trusted_nextn_tp2_serving_authority(
        execution_source_path=execution.absolute_path,
        materialized_cell_id=cell.cell_id,
        compile_launch_manifest=launch_binding,
        inventory=inventory_binding,
        content_source=content_source,
    )
    assert authority.formal_measured_authorization is False
    assert authority.interface_fit_terminal_sha256 == terminal.sha256
    assert authority.qualified_test_count == authority.passed_test_count == 8
    config.model.drafter = "foreign/drafter"
    with pytest.raises(ValueError, match="empirical authority differs"):
        e6.derive_formal_single_operator_trusted_nextn_tp2_serving_authority(
            execution_source_path=execution.absolute_path,
            materialized_cell_id=cell.cell_id,
            compile_launch_manifest=launch_binding,
            inventory=inventory_binding,
            content_source=content_source,
        )
