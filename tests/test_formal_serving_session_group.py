from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config import (
    AdaptationConfig,
    ModelPair,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    run_config_sha256,
)
from lightcone_spec.experiments import formal_single_operator_session_reset as reset
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundleBinding,
)
from lightcone_spec.experiments.formal_single_operator_session_reset import (
    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PROTOCOL_SHA256,
    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_QUALIFICATION_SPEC_PROTOCOL_SHA256,
    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE,
    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS,
    TrustedEmpiricalTp1SessionResetQualificationSpec,
    publish_trusted_empirical_tp1_session_reset_authority,
    revalidate_trusted_empirical_tp1_session_reset_authority,
)
from lightcone_spec.orchestration.formal_serving_session_group import (
    FormalServingSessionGroupPlan,
    FormalServingSessionGroupSpec,
    build_formal_serving_session_group_spec,
    formal_serving_session_reuse_exclusion_reason,
    normalized_formal_serving_process_key,
    partition_formal_serving_session_groups,
)
from lightcone_spec.orchestration.runtime import _render_server
from lightcone_spec.runtime.compile_cache import (
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    CompileCacheKey,
    CompileCacheLaunchPlan,
)
from lightcone_spec.runtime.compile_runner import (
    TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileLaunchManifest,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _publish(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _config(
    *,
    label: str,
    method: str = "static",
    adaptation_group_id: str | None = None,
    learning_rate: float = 1e-5,
) -> RunConfig:
    adaptation = None
    if method == "l0":
        adaptation = AdaptationConfig(
            weight_update_mode="full",
            parameter_scope="last1",
            reset_scope="cohort",
            request_admission_policy="cohort_batching_v1",
            adaptation_group_id=adaptation_group_id or f"cell-{label}",
            optimizer=OptimizerConfig(
                name="adam",
                learning_rate=learning_rate,
                weight_decay=0.0,
            ),
            stride=10,
            canvas_tokens=16,
        )
    return RunConfig(
        method=method,  # type: ignore[arg-type]
        model=ModelPair(
            target="Qwen/Qwen3-8B",
            drafter="z-lab/Qwen3-8B-DFlash-b16",
            target_revision="1" * 40,
            drafter_revision="2" * 40,
            algorithm="DFLASH",
            draft_depth=15,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=_sha(f"sampling:{label}"),
            device_identity="GPU-A",
            rendezvous_identity=f"rendezvous-{label}",
            router_identity=f"router-{label}",
            speculative_num_draft_tokens=16,
            max_running_requests=4,
        ),
        adaptation=adaptation,
    )


def _cache_plan(
    tmp_path: Path,
    *,
    label: str,
    config: RunConfig,
) -> tuple[CompileCacheLaunchPlan, Path]:
    key = CompileCacheKey(
        patched_sglang_tree=PINNED_SGLANG_TREE,
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        patch_sha256=PINNED_SGLANG_PATCH_SHA256,
        source_sha256=PINNED_SGLANG_COMPILE_SOURCE_SHA256,
        python_version="3.11.13",
        torch_version="2.8.0",
        triton_version="3.4.0",
        cuda_version="12.8",
        driver_version="580.95.05",
        sm_architecture="sm_120",
        gpu_model="RTX-PRO-6000",
        dtype="bfloat16",
        target_revision=config.model.target_revision,
        drafter_revision=config.model.drafter_revision,
        tensor_parallel_size=1,
        context_limit=config.runtime.context_length,
        max_running_requests=config.runtime.max_running_requests,
        graph_buckets=(1,),
        allocator="cuda_malloc_async",
        build_flags=(),
    )
    cache_root = (tmp_path / "shared-cache").resolve()
    cache_root.mkdir(exist_ok=True)
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=cache_root,
        cache_mode="build",
    )
    path = (tmp_path / label / "compile-cache-plan.json").resolve()
    path.parent.mkdir()
    plan.write(path)
    return plan, path


def _producer_generated_launch(
    tmp_path: Path,
    *,
    label: str,
    config: RunConfig,
    port: int,
    mem_fraction_static: float = 0.82,
) -> CompileLaunchManifest:
    """Use the real server renderer used by the prepared-launch producer."""

    plan, plan_path = _cache_plan(tmp_path, label=label, config=config)
    root = (tmp_path / label).resolve()
    target = (tmp_path / "models" / config.model.target_revision).resolve()
    drafter = (tmp_path / "models" / config.model.drafter_revision).resolve()
    target.mkdir(parents=True, exist_ok=True)
    drafter.mkdir(parents=True, exist_ok=True)
    checkout = (tmp_path / "patched-sglang").resolve()
    checkout.mkdir(exist_ok=True)
    server = _render_server(
        output=root,
        method=config.method,
        config=config,
        verified_checkout=checkout,
        roots={
            config.model.target: str(target),
            config.model.drafter: str(drafter),
        },
        target_id=config.model.target,
        drafter_id=config.model.drafter,
        adaptation_reserve_mb=0 if config.adaptation is None else 4096,
        mem_fraction_static=mem_fraction_static,
        host="127.0.0.1",
        port=port,
        compile_cache_plan_path=plan_path,
    )
    config_binding = CanonicalJsonProofBinding.bind(server.run_config)
    return CompileLaunchManifest(
        schema_version=2,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
        ),
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        run_config_path=server.run_config,
        run_config_raw_sha256=config_binding.raw_sha256,
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(plan_path),
        compile_cache_plan_raw_sha256=CanonicalJsonProofBinding.bind(
            plan_path
        ).raw_sha256,
        compile_cache_plan_sha256=plan.sha256,
        prewarm_manifest_path=str((root / "prewarm.json").resolve()),
        prewarm_manifest_raw_sha256=_sha(f"prewarm-raw:{label}"),
        prewarm_manifest_sha256=_sha(f"prewarm-semantic:{label}"),
        sampling_profile_path=str((root / "sampling.json").resolve()),
        sampling_profile_raw_sha256=_sha(f"sampling-raw:{label}"),
        prepared_model_content_manifest_path=str((tmp_path / "content.json").resolve()),
        prepared_model_content_manifest_raw_sha256=_sha("content-raw"),
        prepared_model_content_manifest_sha256=_sha("content-semantic"),
        prepared_model_content_manifest_size=10,
        target_content_member_id=_sha("target-member"),
        target_model_id=config.model.target,
        target_snapshot_path=str(target),
        target_revision=config.model.target_revision,
        target_content_authority_sha256=None,
        drafter_content_member_id=_sha("drafter-member"),
        drafter_model_id=config.model.drafter,
        drafter_snapshot_path=str(drafter),
        drafter_revision=config.model.drafter_revision,
        drafter_content_authority_sha256=None,
        tokenizer_content_member_id=_sha("tokenizer-member"),
        tokenizer_model_id=config.model.target,
        tokenizer_snapshot_path=str(target),
        tokenizer_revision=config.model.target_revision,
        tokenizer_content_authority_sha256=None,
        server_argv=server.argv,
        server_argv_sha256=content_sha256({"argv": list(server.argv)}),
        localhost_port=port,
        model_lock_sha256=_sha("model-lock"),
        sampling_profile_sha256=config.runtime.sampling_profile_sha256,
        physical_assignment_sha256=_sha(f"assignment:{label}"),
        experiment_budget_sha256=_sha(f"budget:{label}"),
        budget_materialization_authority_sha256=_sha("budget-authority"),
        inventory_sha256=_sha("inventory"),
        gpu_uuids=("GPU-A",),
        path_entries=("/usr/bin",),
        library_path_entries=("/usr/lib",),
        cuda_home="/usr/local/cuda",
        formal_stage="E3b",
        content_source_binding=None,
    )


def _qualification_common(
    *,
    kind: str,
    run_id: str,
    method_family: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": kind,
        "suite_id": TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE,
        "topology_mode": "tp1_dp1",
        "gpu_uuid": "GPU-A",
        "backend": "DFLASH",
        "method_family": method_family,
        "qualification_run_id": run_id,
    }


def _published_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    method_family: str,
) -> tuple[CanonicalJsonProofBinding, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol_path = (tmp_path / "protocol-lock.json").resolve()
    protocol_binding = _publish(protocol_path, {"fixture": "protocol-lock"})
    content_path = (tmp_path / "content.json").resolve()
    content_path.write_text("{}\n", encoding="utf-8")
    content_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(content_path),
        size=content_path.stat().st_size,
        raw_sha256=hashlib.sha256(content_path.read_bytes()).hexdigest(),
        semantic_sha256=_sha("content-semantic"),
        runtime_binding_status="BOUND",
    )
    inventory_path = (tmp_path / "inventory.json").resolve()
    inventory_binding = _publish(inventory_path, {"fixture": "inventory"})
    source = SimpleNamespace(
        source_snapshot_sha256=_sha("source-snapshot"),
        patched_sglang_tree=PINNED_SGLANG_TREE,
    )
    content = SimpleNamespace(runtime_binding_status="BOUND", source_snapshot=source)
    lock = SimpleNamespace(
        schema_version=5,
        content_source_mode="trusted_single_operator",
        trusted_single_operator_content_bundle_sha256=(content_binding.semantic_sha256),
        sha256=protocol_binding.semantic_sha256,
    )
    inventory = SimpleNamespace(
        sha256=inventory_binding.semantic_sha256,
        device=lambda uuid: SimpleNamespace(uuid=uuid),
    )
    monkeypatch.setattr(reset, "protocol_lock_from_dict", lambda _value: lock)
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: content_binding),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: content,
    )
    monkeypatch.setattr(
        reset.GpuInventory,
        "from_dict",
        classmethod(lambda _cls, _value: inventory),
    )

    run_id = _sha(f"qualification:{method_family}")
    junit_path = (tmp_path / "session-reset.xml").resolve()
    cases = "".join(
        f'<testcase name="{name}"/>'
        for name in TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS
    )
    junit_path.write_text(
        f'<testsuite tests="8" failures="0" errors="0" skipped="0">{cases}</testsuite>',
        encoding="utf-8",
    )
    terminal_path = (tmp_path / "raw-terminal.log").resolve()
    terminal_path.write_text("same pid; exact reset; complete\n", encoding="utf-8")
    lifecycle_path = (tmp_path / "native-lifecycle.json").resolve()
    _publish(
        lifecycle_path,
        {
            **_qualification_common(
                kind="trusted_empirical_tp1_session_reset_native_lifecycle",
                run_id=run_id,
                method_family=method_family,
            ),
            "server_pid": 4321,
            "session_epochs": [1, 2],
            "execution_plan_sha256s": [_sha(f"trace:{index}") for index in range(2)],
            "exact_output_token_trajectory": True,
            "native_timestamp_coverage": True,
        },
    )
    reset_path = (tmp_path / "reset-state.json").resolve()
    _publish(
        reset_path,
        {
            **_qualification_common(
                kind="trusted_empirical_tp1_session_reset_state_evidence",
                run_id=run_id,
                method_family=method_family,
            ),
            "reset_boundary_count": 2,
            "request_queue_empty": True,
            "optimizer_state_reset": True,
            "candidate_state_reset": True,
            "adaptation_state_reset": True,
            "registered_cache_policy_restored": True,
            "terminal_writer_flushed": True,
            "previous_requests_fully_terminal": True,
        },
    )
    hbm_path = (tmp_path / "hbm.json").resolve()
    _publish(
        hbm_path,
        {
            **_qualification_common(
                kind="trusted_empirical_tp1_session_reset_hbm_evidence",
                run_id=run_id,
                method_family=method_family,
            ),
            "initial_memory_bytes": 1000,
            "memory_after_reset_bytes": [1000, 1000],
            "allowed_growth_bytes": 0,
            "monotonic_growth_detected": False,
        },
    )
    spec = TrustedEmpiricalTp1SessionResetQualificationSpec(
        schema_version=1,
        kind="trusted_empirical_tp1_session_reset_qualification_spec",
        protocol_sha256=(
            TRUSTED_EMPIRICAL_TP1_SESSION_RESET_QUALIFICATION_SPEC_PROTOCOL_SHA256
        ),
        topology_mode="tp1_dp1",
        suite_id=TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE,
        gpu_uuid="GPU-A",
        backend="DFLASH",
        method_family=method_family,  # type: ignore[arg-type]
        qualification_run_id=run_id,
        protocol_lock_path=str(protocol_path),
        content_bundle_path=str(content_path),
        inventory_path=str(inventory_path),
        junit_xml_path=str(junit_path),
        raw_terminal_path=str(terminal_path),
        native_lifecycle_path=str(lifecycle_path),
        reset_state_evidence_path=str(reset_path),
        hbm_evidence_path=str(hbm_path),
    )
    spec_path = (tmp_path / "qualification-spec.json").resolve()
    _publish(spec_path, spec.to_dict())
    authority_path = (tmp_path / "session-reset-authority.json").resolve()
    binding = publish_trusted_empirical_tp1_session_reset_authority(
        qualification_spec_path=spec_path,
        output_path=authority_path,
    )
    rebound, authority = revalidate_trusted_empirical_tp1_session_reset_authority(
        authority_path
    )
    assert rebound == binding
    return binding, authority


def test_path_only_empirical_authority_deep_reopens_every_evidence_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, authority = _published_authority(
        tmp_path,
        monkeypatch,
        method_family="lightcone",
    )

    assert authority.to_dict()["authority_sha256"] == authority.sha256
    assert binding == CanonicalJsonProofBinding.bind(binding.absolute_path)
    assert authority.protocol_sha256 == (
        TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PROTOCOL_SHA256
    )
    assert authority.formal_measured is False
    assert authority.operational_reuse_allowed
    assert authority.method_family == "lightcone"
    assert authority.tests_collected == authority.tests_passed == 8

    Path(authority.raw_terminal.absolute_path).write_text("changed\n", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="changed|replay differs"):
        revalidate_trusted_empirical_tp1_session_reset_authority(binding.absolute_path)


def test_real_server_renderer_cell_paths_ports_prewarm_and_sampling_do_not_split_key(
    tmp_path: Path,
) -> None:
    first_config = _config(label="cell-a")
    second_config = _config(label="cell-b")
    first_launch = _producer_generated_launch(
        tmp_path,
        label="cell-a",
        config=first_config,
        port=21001,
    )
    second_launch = _producer_generated_launch(
        tmp_path,
        label="cell-b",
        config=second_config,
        port=21002,
    )

    first = normalized_formal_serving_process_key(
        launch=first_launch,
        config=first_config,
    )
    second = normalized_formal_serving_process_key(
        launch=second_launch,
        config=second_config,
    )

    assert first == second
    assert first.sha256 == second.sha256
    assert first_launch.server_argv_sha256 != second_launch.server_argv_sha256
    assert first_launch.prewarm_manifest_sha256 != (
        second_launch.prewarm_manifest_sha256
    )
    assert first_launch.sampling_profile_sha256 != (
        second_launch.sampling_profile_sha256
    )
    rendered = "\n".join(first.normalized_server_argv)
    assert "21001" not in rendered
    assert first_launch.run_config_path not in rendered
    assert first_launch.compile_cache_plan_path not in rendered

    different_memory_launch = _producer_generated_launch(
        tmp_path,
        label="cell-c",
        config=_config(label="cell-c"),
        port=21003,
        mem_fraction_static=0.80,
    )
    assert (
        normalized_formal_serving_process_key(
            launch=different_memory_launch,
            config=_config(label="cell-c"),
        )
        != first
    )


def _group_spec(
    tmp_path: Path,
    *,
    index: int,
    config: RunConfig,
    launch: CompileLaunchManifest,
    method_family: str,
    source_snapshot_sha256: str,
    protocol_lock_sha256: str,
    inventory_sha256: str,
) -> FormalServingSessionGroupSpec:
    run_plan = _publish(
        (tmp_path / f"run-plan-{index}.json").resolve(),
        {"cell": index},
    )
    return build_formal_serving_session_group_spec(
        node="e3b_final",
        stage="E3b",
        phase="final",
        materialized_cell_id=_sha(f"cell:{index}"),
        attempt=1,
        physical_kind="serving",
        method_family=method_family,
        protocol_lock_sha256=protocol_lock_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        inventory_sha256=inventory_sha256,
        run_plan=run_plan,
        prepared_launch_entry_sha256=_sha(f"entry:{index}"),
        compile_launch_manifest_sha256=_sha(f"launch:{index}"),
        request_schedule_sha256=_sha(f"schedule:{index}"),
        launch=launch,
        config=config,
        output_directory=str((tmp_path / f"output-{index}").resolve()),
        estimated_duration_seconds=10.0,
        dispatch_order_key=(f"{index:04d}",),
    )


def test_partitioner_requires_reopened_gate_and_roundtrips_bounded_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_binding, authority = _published_authority(
        tmp_path / "authority",
        monkeypatch,
        method_family="static",
    )
    specs = []
    for index in range(5):
        config = _config(label=f"cell-{index}")
        launch = _producer_generated_launch(
            tmp_path,
            label=f"cell-{index}",
            config=config,
            port=22000 + index,
        )
        specs.append(
            _group_spec(
                tmp_path,
                index=index,
                config=config,
                launch=launch,
                method_family="static",
                source_snapshot_sha256=authority.source_snapshot_sha256,
                protocol_lock_sha256=authority.protocol_lock_sha256,
                inventory_sha256=authority.inventory_sha256,
            )
        )

    missing = partition_formal_serving_session_groups(specs)
    assert len(missing) == 5
    assert {plan.reason_code for plan in missing} == {"session_reset_gate_missing"}

    plans = partition_formal_serving_session_groups(
        tuple(reversed(specs)),
        reset_authorities=(authority_binding,),
        max_member_count=2,
        max_estimated_duration_seconds=25.0,
    )
    assert [len(plan.members) for plan in plans] == [2, 2, 1]
    assert [plan.execution_mode for plan in plans] == [
        "shared_session_tp1",
        "shared_session_tp1",
        "fresh_process_per_cell",
    ]
    assert plans[-1].reason_code == "session_group_singleton"
    assert all(plan.formal_measured is False for plan in plans)
    assert (
        partition_formal_serving_session_groups(
            specs,
            reset_authorities=(authority_binding,),
            max_member_count=2,
            max_estimated_duration_seconds=25.0,
        )
        == plans
    )
    assert all(
        FormalServingSessionGroupPlan.from_dict(plan.to_dict()) == plan
        for plan in plans
    )
    assert all(
        FormalServingSessionGroupSpec.from_dict(spec.to_dict()) == spec
        for spec in specs
    )


def test_lightcone_recipe_isolation_and_explicit_fresh_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_binding, authority = _published_authority(
        tmp_path / "authority",
        monkeypatch,
        method_family="lightcone",
    )
    configs = (
        _config(
            label="a",
            method="l0",
            adaptation_group_id="per-cell-a",
            learning_rate=1e-5,
        ),
        _config(
            label="b",
            method="l0",
            adaptation_group_id="per-cell-b",
            learning_rate=1e-5,
        ),
        _config(
            label="c",
            method="l0",
            adaptation_group_id="per-cell-c",
            learning_rate=3e-5,
        ),
    )
    launches = tuple(
        _producer_generated_launch(
            tmp_path,
            label=f"adaptive-{index}",
            config=config,
            port=23000 + index,
        )
        for index, config in enumerate(configs)
    )
    keys = tuple(
        normalized_formal_serving_process_key(launch=launch, config=config)
        for launch, config in zip(launches, configs, strict=True)
    )
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]
    specs = tuple(
        _group_spec(
            tmp_path,
            index=10 + index,
            config=config,
            launch=launch,
            method_family="lightcone",
            source_snapshot_sha256=authority.source_snapshot_sha256,
            protocol_lock_sha256=authority.protocol_lock_sha256,
            inventory_sha256=authority.inventory_sha256,
        )
        for index, (launch, config) in enumerate(zip(launches, configs, strict=True))
    )
    plans = partition_formal_serving_session_groups(
        specs,
        reset_authorities=(authority_binding,),
    )
    assert [len(plan.members) for plan in plans] == [2, 1]
    assert plans[0].session_adaptation_group_id == (
        f"formal-session-{plans[0].group_id[:32]}"
    )
    assert plans[1].reason_code == "session_group_singleton"

    static_config = _config(label="excluded")
    static_launch = _producer_generated_launch(
        tmp_path,
        label="excluded",
        config=static_config,
        port=24000,
    )
    assert (
        formal_serving_session_reuse_exclusion_reason(
            physical_kind="profiler",
            launch=static_launch,
            config=static_config,
        )
        == "profiler_requires_fresh_process"
    )
    assert (
        formal_serving_session_reuse_exclusion_reason(
            physical_kind="e5_failure",
            launch=static_launch,
            config=static_config,
        )
        == "failure_injection_requires_fresh_process"
    )
    assert (
        formal_serving_session_reuse_exclusion_reason(
            physical_kind="serving",
            launch=replace(static_launch, gpu_uuids=("GPU-A", "GPU-B")),
            config=static_config,
        )
        == "distributed_topology_requires_fresh_process"
    )
    assert (
        formal_serving_session_reuse_exclusion_reason(
            physical_kind="serving",
            launch=replace(static_launch, schema_version=3),
            config=static_config,
        )
        == "nextn_requires_fresh_process"
    )
