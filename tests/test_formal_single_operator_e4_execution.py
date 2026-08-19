from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config import ModelPair, RunConfig, RuntimeConfig, run_config_sha256
from lightcone_spec.experiments.formal_preflight_inputs import (
    FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
    FormalPreflightExecutionInputs,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_to_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_single_operator_e4_execution import (
    materialize_formal_single_operator_e4_compile_launch,
    revalidate_formal_single_operator_e4_compile_launch,
)
from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
    FormalSingleOperatorDownstreamRunPlanInputs,
    FormalSingleOperatorPreparedDownstreamRunPlanInputs,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
    FormalSingleOperatorJsonBinding,
    FormalSingleOperatorNodeMaterialization,
    FormalSingleOperatorRunManifestActualValidator,
    FormalSingleOperatorStageCompletion,
    FormalSingleOperatorStageDecision,
    FormalSingleOperatorValidatedActual,
    _e2_recipe_payload,
    formal_single_operator_node_spec,
    materialize_formal_single_operator_node,
    publish_formal_single_operator_execution_source,
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
)
from lightcone_spec.experiments.load import FrozenSamplingParameters
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.serving import BoundServingRequest
from lightcone_spec.experiments.stage_materialization import (
    E4_SCREEN_FACTOR_LEVELS,
    E2CandidateRecipe,
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
    e1_geometries,
    e2_candidate_recipes,
)
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
    FormalServingRequestScheduleReceipt,
    FormalServingRequestScheduleRow,
    FormalServingRunPlan,
    _load_protocol_for_cell,
)
from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding
from lightcone_spec.runtime import formal_single_operator as single
from lightcone_spec.runtime.compile_cache import (
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    CompileCacheKey,
    CompileCacheLaunchPlan,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
)
from lightcone_spec.runtime.compile_runner import (
    COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileLaunchManifest,
    write_compile_prewarm_manifest,
)
from lightcone_spec.runtime.content_authorization import ContentJsonArtifactBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str, str, str]:
    repository = (tmp_path / "source").resolve()
    patch_root = repository / "patches" / "sglang"
    patch_root.mkdir(parents=True)
    patch = patch_root / "0001.patch"
    patch.write_text("test patch\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "upstream": {
            "repository": "https://example.invalid/sglang",
            "commit": "6" * 40,
        },
        "expected_tree": "8" * 40,
        "patches": [
            {
                "file": patch.name,
                "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                "files": ["python/example.py"],
            }
        ],
    }
    (patch_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test Operator")
    _git(repository, "config", "user.email", "operator@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "source identity")
    return (
        repository,
        _git(repository, "rev-parse", "HEAD"),
        _git(repository, "rev-parse", "HEAD^{tree}"),
        single._content_sha256(manifest),
    )


def _protocol_lock(*, head: str, tree: str, patch_sha256: str) -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="formal-single-operator-e4-test",
        code_git_head=head,
        code_git_tree=tree,
        patch_manifest_sha256=patch_sha256,
        registry_sha256=build_industrial_registry().sha256,
        english_protocol_sha256=_sha("english"),
        chinese_protocol_sha256=_sha("chinese"),
        tts_calibration_authority_sha256=_sha("tts"),
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1-anchor"),
        e2_recipe_grid_authority_sha256=(default_e2_recipe_grid_authority().sha256),
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


def _inventory() -> GpuInventory:
    return GpuInventory(
        schema_version=1,
        devices=(
            GpuDevice(
                uuid="GPU-e4-test",
                host_id="host-e4-test",
                model="NVIDIA H200",
                memory_bytes=141_000_000_000,
                compute_capability=(9, 0),
                pci_bus_id="0000:01:00.0",
                pci_root="root-0",
                numa_node=0,
                interconnects=("pcie",),
                peer_access_class="single_gpu",
                clock_policy="locked",
                power_limit_watts=700.0,
                thermal_limit_celsius=83.0,
                availability=GpuAvailability.READY,
                reserved_processes=(),
                allowed_topology_groups=(),
            ),
        ),
        topology_groups=(),
        source_receipt_sha256=_sha("inventory-source"),
    )


def _canonical_with_sidecar(path: Path, value: object) -> tuple[str, int]:
    body = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    path.write_bytes(body)
    semantic = single._content_sha256(value)
    Path(f"{path}.sha256").write_text(f"{semantic}\n", encoding="ascii")
    return hashlib.sha256(body).hexdigest(), len(body)


def _recipe() -> E2CandidateRecipe:
    grid = default_e2_recipe_grid_authority()
    return next(
        row
        for row in e2_candidate_recipes((e1_geometries()[0],), grid=grid)
        if row.optimizer == "adamw"
        and row.schedule == "constant"
        and row.learning_rate
        == grid.rates(optimizer="adamw", parameterization="full")[0]
    )


def _e2_config(
    *, recipe: E2CandidateRecipe, inventory: GpuInventory, sampling: SamplingProfile
) -> RunConfig:
    grid = default_e2_recipe_grid_authority()
    return RunConfig(
        method="l0",
        model=ModelPair(
            target="Qwen/Qwen3-8B",
            target_revision="1" * 40,
            drafter_revision="2" * 40,
            draft_depth=15,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=sampling.sha256,
            device_identity=inventory.devices[0].uuid,
            speculative_num_draft_tokens=16,
            max_running_requests=8,
        ),
        adaptation=grid.adaptation_config_for(
            recipe,
            canvas_tokens=16,
            adaptation_group_id="formal-single-e2-winner",
        ),
    )


def _base_launch(
    *,
    root: Path,
    config: RunConfig,
    inventory: GpuInventory,
    content_sha256: str,
) -> CompileLaunchManifest:
    sampling = SamplingProfile()
    sampling_path = root / "sampling.json"
    sampling.write(sampling_path)
    config_path = root / "run-config.json"
    config_raw, _config_size = _canonical_with_sidecar(
        config_path, config.model_dump(mode="json")
    )
    checkout = root / "patched-sglang"
    target = root / "models" / "target" / config.model.target_revision
    drafter = root / "models" / "drafter" / config.model.drafter_revision
    tokenizer = root / "models" / "tokenizer" / ("3" * 40)
    cuda_home = root / "cuda"
    library = root / "lib"
    bin_path = root / "bin"
    cache_root = root / "cache"
    for directory in (
        checkout,
        target,
        drafter,
        tokenizer,
        cuda_home,
        library,
        bin_path,
        cache_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    key = CompileCacheKey(
        patched_sglang_tree=PINNED_SGLANG_TREE,
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        patch_sha256=PINNED_SGLANG_PATCH_SHA256,
        source_sha256=PINNED_SGLANG_COMPILE_SOURCE_SHA256,
        python_version="3.12",
        torch_version="2.8",
        triton_version="3.4",
        cuda_version="12.8",
        driver_version="570.00",
        sm_architecture="sm_90",
        gpu_model=inventory.devices[0].model,
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
    cache_plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=cache_root,
        cache_mode="build",
    )
    cache_path = root / "cache-plan.json"
    cache_plan.write(cache_path)
    cache_binding = CanonicalJsonProofBinding.bind(cache_path)
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=_sha("model-lock"),
        sampling_profile_sha256=sampling.sha256,
        payloads=(
            CompileOnlyPrewarmPayload(
                request_id="prewarm-0",
                graph_bucket=1,
                input_token_ids=(1, 2),
                requested_output_tokens=2,
                sampling_seed=1,
            ),
        ),
    )
    prewarm_path = root / "prewarm.json"
    write_compile_prewarm_manifest(prewarm, prewarm_path)
    prewarm_binding = CanonicalJsonProofBinding.bind(prewarm_path)
    prepared_path = root / "prepared.json"
    prepared_raw, prepared_size = _canonical_with_sidecar(
        prepared_path, {"kind": "prepared-model-content-test"}
    )
    server_argv = (
        "/usr/bin/env",
        "python",
        "--model-path",
        str(target),
        "--speculative-draft-model-path",
        str(drafter),
        "--host",
        "127.0.0.1",
        "--port",
        "31001",
        "--mem-fraction-static",
        "0.8",
        "--speculative-adaptation-reserve-mb",
        "2048",
    )
    launch = CompileLaunchManifest(
        schema_version=1,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        run_config_path=str(config_path),
        run_config_raw_sha256=config_raw,
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=cache_binding.raw_sha256,
        compile_cache_plan_sha256=cache_plan.sha256,
        prewarm_manifest_path=str(prewarm_path),
        prewarm_manifest_raw_sha256=prewarm_binding.raw_sha256,
        prewarm_manifest_sha256=prewarm.sha256,
        sampling_profile_path=str(sampling_path),
        sampling_profile_raw_sha256=hashlib.sha256(
            sampling_path.read_bytes()
        ).hexdigest(),
        prepared_model_content_manifest_path=str(prepared_path),
        prepared_model_content_manifest_raw_sha256=prepared_raw,
        prepared_model_content_manifest_sha256=single._content_sha256(
            {"kind": "prepared-model-content-test"}
        ),
        prepared_model_content_manifest_size=prepared_size,
        target_content_member_id="target-member",
        target_model_id=config.model.target,
        target_snapshot_path=str(target),
        target_revision=config.model.target_revision,
        target_content_authority_sha256=content_sha256,
        drafter_content_member_id="drafter-member",
        drafter_model_id=config.model.drafter,
        drafter_snapshot_path=str(drafter),
        drafter_revision=config.model.drafter_revision,
        drafter_content_authority_sha256=content_sha256,
        tokenizer_content_member_id="tokenizer-member",
        tokenizer_model_id=config.model.target,
        tokenizer_snapshot_path=str(tokenizer),
        tokenizer_revision=tokenizer.name,
        tokenizer_content_authority_sha256=content_sha256,
        server_argv=server_argv,
        server_argv_sha256=single._content_sha256({"argv": list(server_argv)}),
        localhost_port=31_001,
        model_lock_sha256=prewarm.model_lock_sha256,
        sampling_profile_sha256=sampling.sha256,
        physical_assignment_sha256=_sha("e2-assignment"),
        experiment_budget_sha256=_sha("e2-budget"),
        budget_materialization_authority_sha256=_sha("e2-budget-authority"),
        inventory_sha256=inventory.sha256,
        gpu_uuids=(inventory.devices[0].uuid,),
        path_entries=(str(bin_path),),
        library_path_entries=(str(library),),
        cuda_home=str(cuda_home),
    )
    launch.write(root / "compile-launch.json")
    return launch


def _run_plan(
    *,
    root: Path,
    launch_path: Path,
    launch: CompileLaunchManifest,
    cell_id: str,
    stage: str,
) -> FormalServingRunPlan:
    proof_path = root / "runtime-gpu-proof.json"
    publish_canonical_json_no_replace(proof_path, {"kind": "gpu-proof-test"})
    proof = CanonicalJsonProofBinding.bind(proof_path)
    schedule_value = {"kind": "request-schedule-test", "rows": []}
    schedule_path = root / "request-schedule.json"
    publish_canonical_json_no_replace(schedule_path, schedule_value)
    schedule = CanonicalJsonProofBinding.bind(schedule_path)
    binding = NativeTerminalRunBinding(
        run_id=f"run-{cell_id[:16]}",
        run_nonce_sha256=_sha(f"nonce:{cell_id}"),
        execution_plan_sha256=_sha(f"plan:{cell_id}"),
        rank_config_sha256=_sha(f"rank:{cell_id}"),
        attempt_id="attempt-0000",
        session_id=f"session-{cell_id[:16]}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha(f"challenge:{cell_id}"),
        method="l0",
        warmup_request_ids=(),
        scored_request_ids=("request-0",),
    )
    return FormalServingRunPlan(
        schema_version=1,
        kind="formal_serving_run_plan",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=_sha(f"execution-binding:{cell_id}"),
        subject_sha256=_sha(f"execution-subject:{cell_id}"),
        materialized_cell_id=cell_id,
        stage=stage,
        method="l0",
        topology_mode="tp1_dp1",
        inventory_sha256=launch.inventory_sha256,
        gpu_uuids=launch.gpu_uuids,
        runtime_gpu_proof_sha256s=(proof.semantic_sha256,),
        runtime_gpu_proof_artifacts=(proof,),
        nextn_tp2_authority_sha256=None,
        launch_manifest=CanonicalJsonProofBinding.bind(launch_path),
        request_schedule_receipt=schedule,
        native_terminal_binding=binding,
        private_output_root=str(root),
        terminal_output_path=str(root / "terminal.json"),
        native_itl_pointer_output_path=str(root / "native-itl.json"),
        live_run_receipt_output_path=str(root / "live-run.json"),
        lifecycle_timing_output_path=str(root / "lifecycle.json"),
        server_log_output_path=str(root / "server.log"),
        server_stdout_output_path=str(root / "stdout.json"),
        server_stderr_output_path=str(root / "stderr.json"),
        junit_output_path=str(root / "junit.xml"),
        before_gpu_snapshot_output_path=str(root / "before-gpu.json"),
        ready_gpu_snapshot_output_path=str(root / "ready-gpu.json"),
        after_gpu_snapshot_output_path=str(root / "after-gpu.json"),
        formal_gang_terminal_output_path=None,
        fatal_output_path=str(root / "fatal.json"),
    )


def _publish_manifest(
    *,
    repository: Path,
    head: str,
    tree: str,
    patch_sha256: str,
    root: Path,
    launch_path: Path,
    launch: CompileLaunchManifest,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
) -> Path:
    plan = _run_plan(
        root=root,
        launch_path=launch_path,
        launch=launch,
        cell_id=cell.cell_id,
        stage=cell.stage,
    )
    plan_path = root / "run-plan.json"
    publish_canonical_json_no_replace(plan_path, plan.to_dict())
    schedule = plan.request_schedule_receipt.reopen()
    required_paths = {
        "admission": root / "admission.json",
        "admission_consumption": root / "admission-consumption.json",
        "after_gpu_snapshot": root / "after-gpu.json",
        "before_gpu_snapshot": root / "before-gpu.json",
        "junit": root / "junit.xml",
        "raw_terminal": root / "terminal.json",
        "native_itl": root / "native-itl.json",
        "lifecycle": root / "lifecycle.json",
        "live_run_receipt": root / "live-run.json",
        "ready_gpu_snapshot": root / "ready-gpu.json",
        "request_schedule": Path(plan.request_schedule_receipt.absolute_path),
        "run_plan": plan_path,
        "stdout": root / "stdout.json",
        "stderr": root / "stderr.json",
    }
    for name, path in required_paths.items():
        if not path.exists():
            path.write_text(f"{name}\n", encoding="utf-8")
    artifacts = tuple(
        sorted(
            (
                single.FormalSingleOperatorArtifact.observe(
                    name=name,
                    run_root=root,
                    path=path,
                )
                for name, path in required_paths.items()
            ),
            key=lambda row: row.name,
        )
    )
    config = RunConfig.model_validate(
        json.loads(Path(launch.run_config_path).read_text(encoding="utf-8"))
    )
    manifest = single.FormalSingleOperatorRunManifest(
        schema="formal_single_operator_v1",
        protocol_sha256=single.FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256,
        trust_assumptions=single.FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        git_head=head,
        git_tree=tree,
        sglang_upstream_commit="6" * 40,
        patch_manifest_sha256=patch_sha256,
        patched_sglang_tree="8" * 40,
        registry_sha256=build_industrial_registry().sha256,
        physical_dispatch_protocol_sha256=(
            FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        run_plan_sha256=plan.sha256,
        launch_manifest_sha256=launch.sha256,
        execution_binding_sha256=plan.execution_binding_sha256,
        execution_subject_sha256=plan.subject_sha256,
        materialization_protocol_lock_sha256=(materialization.protocol_lock_sha256),
        materialization_sha256=materialization.sha256,
        inventory_sha256=launch.inventory_sha256,
        run_config_sha256=run_config_sha256(config),
        run_config=config.model_dump(mode="json"),
        launch_argv_sha256=launch.server_argv_sha256,
        launch_argv=launch.server_argv,
        localhost_port=launch.localhost_port,
        request_schedule_sha256=single._content_sha256(schedule),
        request_schedule=schedule,
        target_model_id=launch.target_model_id,
        target_revision=launch.target_revision,
        target_content_sha256=launch.target_content_authority_sha256,
        drafter_model_id=launch.drafter_model_id,
        drafter_revision=launch.drafter_revision,
        drafter_content_sha256=launch.drafter_content_authority_sha256,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_content_sha256=launch.tokenizer_content_authority_sha256,
        workload_artifact_id="e4-test-workload",
        workload_authority_sha256=_sha("workload-authority"),
        workload_member_sha256s=(_sha("workload-member"),),
        workload_raw_sha256=_sha("workload-raw"),
        workload_semantic_sha256=_sha("workload-semantic"),
        stage=cell.stage,
        cell_id=cell.cell_id,
        role=cell.method_role,
        backend=cell.backend,
        topology="tp1_dp1",
        block=None,
        attempt="attempt-0000",
        run_directory=str(root),
        gpu_environment=(
            single.FormalSingleOperatorGpu(
                uuid=launch.gpu_uuids[0],
                model="NVIDIA H200",
                driver_version="570.00",
                cuda_version="12.8",
            ),
        ),
        started_ns=10,
        finished_ns=20,
        exit_code=0,
        completion_status="COMPLETE",
        failure_reason=None,
        artifacts=artifacts,
    )
    manifest_path = root / "formal-single-operator-manifest.json"
    raw_sha256, size = publish_canonical_json_no_replace(
        manifest_path, manifest.to_dict()
    )
    publish_canonical_json_no_replace(
        root / "formal-single-operator-manifest.sha256.json",
        {
            "schema": "formal_single_operator_manifest_pointer_v1",
            "manifest_raw_sha256": raw_sha256,
            "manifest_semantic_sha256": manifest.sha256,
            "manifest_size": size,
        },
    )
    assert (
        single.revalidate_formal_single_operator_run_manifest(
            repository_root=repository,
            manifest_path=manifest_path,
        )
        == manifest
    )
    return manifest_path


def _dummy_cell(node: str, stage: str) -> MaterializedCell:
    return MaterializedCell(
        stage=stage,
        method_role="Target-only",
        model="Qwen/Qwen3-8B",
        backend="NONE",
        task=f"single_operator_fixture_{node}",
        publication_policy="none",
        recipe_sha256=None,
        dimensions=(("fixture", node),),
    )


def _fixture_materializations(
    *, protocol_lock: ProtocolLock, recipe: E2CandidateRecipe
) -> dict[str, StageMaterializationReceipt]:
    result: dict[str, StageMaterializationReceipt] = {}
    for node in (
        "preflight",
        "e3a",
        "tts_cal",
        "e1",
        "e2_r0",
        "e2_r1",
        "e2_r2",
        "e2_r3",
    ):
        spec = formal_single_operator_node_spec(node)
        prior = None if not result else result[next(reversed(result))]
        cell = (
            MaterializedCell(
                stage="E2",
                method_role="LightCone-candidate",
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="LiveCodeBench_tuning",
                publication_policy="first_ready",
                recipe_sha256=recipe.sha256,
                dimensions=(("round", 3),),
            )
            if node == "e2_r3"
            else _dummy_cell(node, spec.stage)
        )
        result[node] = StageMaterializationReceipt(
            schema_version=1,
            stage=spec.stage,
            protocol_lock_sha256=protocol_lock.sha256,
            upstream_receipt_sha256s=(() if prior is None else (prior.sha256,)),
            source_decision_sha256=_sha(f"source:{node}"),
            materialization_rule=f"single_operator_fixture_{node}",
            expected_cell_count=1,
            cells=(cell,),
            gpu_hours=GpuHourEstimate.unmeasured(),
        )
    return result


def _publish_chain_to_e2(
    *,
    root: Path,
    protocol_lock: ProtocolLock,
    recipe: E2CandidateRecipe,
    e2_manifest_path: Path,
    materializations: dict[str, StageMaterializationReceipt],
) -> Path:
    lock_path = root / "protocol-lock.json"
    publish_formal_single_operator_json_artifact(
        lock_path, protocol_lock_to_dict(protocol_lock)
    )
    lock_binding = FormalSingleOperatorJsonBinding.bind(
        lock_path, label="test ProtocolLock"
    )
    nodes = (
        "preflight",
        "e3a",
        "tts_cal",
        "e1",
        "e2_r0",
        "e2_r1",
        "e2_r2",
        "e2_r3",
    )
    predecessor_path: Path | None = None
    predecessor_completion: FormalSingleOperatorStageCompletion | None = None
    predecessor_decision: FormalSingleOperatorStageDecision | None = None
    predecessor_materialization: StageMaterializationReceipt | None = None
    for index, node in enumerate(nodes):
        spec = formal_single_operator_node_spec(node)
        materialization = materializations[node]
        cell = materialization.cells[0]
        source_decision = (
            _sha(f"source:{node}")
            if predecessor_decision is None
            else predecessor_decision.next_materialization_source_decision_sha256
        )
        assert source_decision is not None
        assert materialization.source_decision_sha256 == source_decision
        assert materialization.upstream_receipt_sha256s == (
            ()
            if predecessor_materialization is None
            else (predecessor_materialization.sha256,)
        )
        materialization_path = root / f"{node}-materialization.json"
        publish_formal_single_operator_json_artifact(
            materialization_path,
            stage_materialization_receipt_to_dict(materialization),
        )
        node_materialization = FormalSingleOperatorNodeMaterialization(
            schema_version=1,
            kind="formal_single_operator_node_materialization",
            protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
            node=node,  # type: ignore[arg-type]
            ordinal=spec.ordinal,
            stage=spec.stage,
            phase=spec.phase,
            predecessor_source=(
                None
                if predecessor_path is None
                else FormalSingleOperatorJsonBinding.bind(
                    predecessor_path, label="test predecessor"
                )
            ),
            predecessor_completion_sha256=(
                None
                if predecessor_completion is None
                else predecessor_completion.sha256
            ),
            protocol_lock_source=lock_binding,
            protocol_lock_sha256=protocol_lock.sha256,
            runtime_authority_manifest_sha256=(
                protocol_lock.formal_runtime_authority_manifest_sha256
            ),
            prepared_model_content_authorization_sha256=(
                protocol_lock.prepared_model_content_authorization_sha256
            ),
            formal_workload_e3a_authorization_sha256=(
                protocol_lock.formal_workload_e3a_authorization_sha256
            ),
            formal_workload_e0_authorization_sha256=(
                protocol_lock.formal_workload_e0_authorization_sha256
            ),
            burstgpt_shape_authorization_sha256=(
                protocol_lock.burstgpt_shape_authorization_sha256
            ),
            materialization_source=FormalSingleOperatorJsonBinding.bind(
                materialization_path, label="test materialization"
            ),
            materialization_sha256=materialization.sha256,
            created_ns=100 + index * 10,
        )
        node_path = root / f"{node}-node.json"
        publish_formal_single_operator_json_artifact(
            node_path, node_materialization.to_dict()
        )
        actual_source = (
            e2_manifest_path if node == "e2_r3" else root / f"{node}-actual-source.json"
        )
        if node != "e2_r3":
            publish_formal_single_operator_json_artifact(
                actual_source, {"kind": "fixture-actual", "node": node}
            )
        actual = FormalSingleOperatorValidatedActual(
            node=node,  # type: ignore[arg-type]
            stage=spec.stage,
            materialization_sha256=materialization.sha256,
            cell_id=cell.cell_id,
            status="COMPLETE",
            started_ns=110 + index * 10,
            finished_ns=115 + index * 10,
            result_identity_sha256=(
                single.FormalSingleOperatorRunManifest.from_dict(
                    json.loads(e2_manifest_path.read_text(encoding="utf-8"))
                ).sha256
                if node == "e2_r3"
                else _sha(f"result:{node}")
            ),
            validator_kind="run_manifest",
            validator_protocol_sha256=_sha("fixture-validator"),
            source=FormalSingleOperatorJsonBinding.bind(
                actual_source, label="test actual source"
            ),
            reducer_payload={"kind": "fixture-reducer-payload"},
        )
        actual_set_sha256 = single._content_sha256([actual.to_dict()])
        if node == "e2_r3":
            selection = _sha("e2-r3-selection")
            next_source = selection
            payload = {
                "schema_version": 1,
                "kind": "formal_single_operator_e2_round_selection",
                "round_index": 3,
                "model": "Qwen/Qwen3-8B",
                "matched_width": 16,
                "common_load": 8,
                "frozen_tts_recipe_sha256": _sha("frozen-tts"),
                "source_geometries": [],
                "source_candidate_count": 1,
                "survivor_recipes": [],
                "final_recipe": _e2_recipe_payload(recipe),
                "evaluation_sha256s": [],
                "selection_sha256": selection,
            }
        else:
            next_source = _sha(f"source:{nodes[index + 1]}")
            payload = {"kind": "fixture-decision", "node": node}
        decision = FormalSingleOperatorStageDecision(
            schema_version=1,
            kind="formal_single_operator_stage_decision",
            protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
            node=node,  # type: ignore[arg-type]
            ordinal=spec.ordinal,
            stage=spec.stage,
            phase=spec.phase,
            predecessor_completion_sha256=(
                None
                if predecessor_completion is None
                else predecessor_completion.sha256
            ),
            materialization_sha256=materialization.sha256,
            actual_result_set_sha256=actual_set_sha256,
            decision_kind=f"fixture-{node}",
            next_materialization_source_decision_sha256=next_source,
            next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
            payload=payload,
        )
        decision_path = root / f"{node}-decision.json"
        publish_formal_single_operator_json_artifact(decision_path, decision.to_dict())
        completion = FormalSingleOperatorStageCompletion(
            schema_version=1,
            kind="formal_single_operator_stage_completion",
            protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
            node=node,  # type: ignore[arg-type]
            ordinal=spec.ordinal,
            stage=spec.stage,
            phase=spec.phase,
            predecessor_source=node_materialization.predecessor_source,
            predecessor_completion_sha256=(
                None
                if predecessor_completion is None
                else predecessor_completion.sha256
            ),
            protocol_lock_sha256=protocol_lock.sha256,
            node_materialization_source=FormalSingleOperatorJsonBinding.bind(
                node_path, label="test node materialization"
            ),
            node_materialization_sha256=node_materialization.sha256,
            materialization_sha256=materialization.sha256,
            actual_results=(actual,),
            actual_result_set_sha256=actual_set_sha256,
            decision_source=FormalSingleOperatorJsonBinding.bind(
                decision_path, label="test decision"
            ),
            decision_sha256=decision.sha256,
            completed_ns=119 + index * 10,
        )
        completion_path = root / f"{node}-completion.json"
        publish_formal_single_operator_json_artifact(
            completion_path, completion.to_dict()
        )
        predecessor_path = completion_path
        predecessor_completion = completion
        predecessor_decision = decision
        predecessor_materialization = materialization
    assert predecessor_path is not None
    return predecessor_path


def _fixture(tmp_path: Path):
    repository, head, tree, patch_sha256 = _source_repository(tmp_path)
    protocol_lock = _protocol_lock(head=head, tree=tree, patch_sha256=patch_sha256)
    inventory = _inventory()
    inventory_path = (tmp_path / "inventory.json").resolve()
    publish_formal_single_operator_json_artifact(inventory_path, inventory.to_dict())
    recipe = _recipe()
    materializations = _fixture_materializations(
        protocol_lock=protocol_lock,
        recipe=recipe,
    )
    sampling = SamplingProfile()
    e2_config = _e2_config(recipe=recipe, inventory=inventory, sampling=sampling)
    chain_root = (tmp_path / "chain").resolve()
    chain_root.mkdir()
    # The E2 materialization identity is determined before its actual manifest.
    e2_materialization = materializations["e2_r3"]
    e2_cell = e2_materialization.cells[0]
    e2_run_root = (tmp_path / "e2-run").resolve()
    e2_run_root.mkdir()
    e2_run_root.chmod(0o700)
    e2_launch = _base_launch(
        root=e2_run_root,
        config=e2_config,
        inventory=inventory,
        content_sha256=protocol_lock.prepared_model_content_authorization_sha256,
    )
    e2_manifest_path = _publish_manifest(
        repository=repository,
        head=head,
        tree=tree,
        patch_sha256=patch_sha256,
        root=e2_run_root,
        launch_path=e2_run_root / "compile-launch.json",
        launch=e2_launch,
        materialization=e2_materialization,
        cell=e2_cell,
    )
    # The helper reconstructs the same E2 materialization identity from its chain.
    e2_completion_path = _publish_chain_to_e2(
        root=chain_root,
        protocol_lock=protocol_lock,
        recipe=recipe,
        e2_manifest_path=e2_manifest_path,
        materializations=materializations,
    )
    return {
        "repository": repository,
        "head": head,
        "tree": tree,
        "patch_sha256": patch_sha256,
        "protocol_lock": protocol_lock,
        "inventory": inventory,
        "inventory_path": inventory_path,
        "recipe": recipe,
        "e2_completion_path": e2_completion_path,
        "chain_root": chain_root,
    }


def test_e4_screen_mapper_derives_exact_config_argv_and_port_from_current_chain(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    chain_root = fixture["chain_root"]
    screen = materialize_formal_single_operator_node(
        node="e4_screen",
        predecessor_completion_path=fixture["e2_completion_path"],
        protocol_lock_path=None,
        materialization_output_path=chain_root / "e4-screen-materialization.json",
        node_materialization_output_path=chain_root / "e4-screen-node.json",
        created_ns=300,
    )
    source_path = chain_root / "e4-screen-execution-source.json"
    publish_formal_single_operator_execution_source(
        node_materialization_path=chain_root / "e4-screen-node.json",
        output_path=source_path,
    )
    cell = next(
        row
        for row in screen.materialization.cells
        if dict(row.dimensions)["load"] == "moderate"
        and dict(row.dimensions)["traffic"] == "mixed_prefill_decode"
    )
    output = (tmp_path / "e4-screen-run").resolve()
    output.mkdir()
    output.chmod(0o700)
    context = materialize_formal_single_operator_e4_compile_launch(
        execution_source_path=source_path,
        materialized_cell_id=cell.cell_id,
        repository_root=fixture["repository"],
        inventory_path=fixture["inventory_path"],
        private_output_root=output,
    )
    dimensions = dict(cell.dimensions)
    assert context.lightcone_recipe == fixture["recipe"]
    assert context.run_config.runtime.max_running_requests == 8
    assert (
        context.run_config.runtime.adaptation_microbatch_size
        == dimensions["microbatch"]
    )
    assert (
        context.run_config.runtime.adaptation_publication_coalescing
        == dimensions["coalescing"]
    )
    assert (
        context.run_config.runtime.adaptation_stream_priority
        == dimensions["stream_priority"]
    )
    assert context.run_config.adaptation is not None
    assert context.run_config.adaptation.stride == dimensions["update_stride"]
    assert context.traffic == "mixed_prefill_decode"
    assert context.launch.localhost_port == int(
        context.launch.server_argv[context.launch.server_argv.index("--port") + 1]
    )
    assert (
        revalidate_formal_single_operator_e4_compile_launch(
            execution_source_path=source_path,
            materialized_cell_id=cell.cell_id,
            repository_root=fixture["repository"],
            inventory_path=fixture["inventory_path"],
            compile_launch_manifest_path=(
                output / "formal-single-operator-e4-compile-launch.json"
            ),
        )
        == context
    )
    with pytest.raises(RuntimeError, match="refuses to replace"):
        materialize_formal_single_operator_e4_compile_launch(
            execution_source_path=source_path,
            materialized_cell_id=cell.cell_id,
            repository_root=fixture["repository"],
            inventory_path=fixture["inventory_path"],
            private_output_root=output,
        )


def test_e4_local_mapper_uses_same_stratum_screen_winner_actual(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    chain_root = fixture["chain_root"]
    screen = materialize_formal_single_operator_node(
        node="e4_screen",
        predecessor_completion_path=fixture["e2_completion_path"],
        protocol_lock_path=None,
        materialization_output_path=chain_root / "screen-materialization.json",
        node_materialization_output_path=chain_root / "screen-node.json",
        created_ns=300,
    )
    screen_source_path = chain_root / "screen-source.json"
    publish_formal_single_operator_execution_source(
        node_materialization_path=chain_root / "screen-node.json",
        output_path=screen_source_path,
    )
    winner_cell = next(
        row
        for row in screen.materialization.cells
        if dict(row.dimensions)["load"] == "low"
        and dict(row.dimensions)["traffic"] == "pure_decode"
    )
    screen_run = (tmp_path / "screen-actual-run").resolve()
    screen_run.mkdir()
    screen_run.chmod(0o700)
    screen_context = materialize_formal_single_operator_e4_compile_launch(
        execution_source_path=screen_source_path,
        materialized_cell_id=winner_cell.cell_id,
        repository_root=fixture["repository"],
        inventory_path=fixture["inventory_path"],
        private_output_root=screen_run,
    )
    winner_manifest = _publish_manifest(
        repository=fixture["repository"],
        head=fixture["head"],
        tree=fixture["tree"],
        patch_sha256=fixture["patch_sha256"],
        root=screen_run,
        launch_path=screen_run / "formal-single-operator-e4-compile-launch.json",
        launch=screen_context.launch,
        materialization=screen.materialization,
        cell=winner_cell,
    )
    actuals = []
    for index, cell in enumerate(screen.materialization.cells):
        if cell.cell_id == winner_cell.cell_id:
            source_path = winner_manifest
            result_sha = single.FormalSingleOperatorRunManifest.from_dict(
                json.loads(winner_manifest.read_text(encoding="utf-8"))
            ).sha256
        else:
            source_path = chain_root / f"screen-dummy-{index}.json"
            publish_formal_single_operator_json_artifact(
                source_path, {"kind": "unselected-screen-actual", "index": index}
            )
            result_sha = _sha(f"screen-result:{index}")
        actuals.append(
            FormalSingleOperatorValidatedActual(
                node="e4_screen",
                stage="E4",
                materialization_sha256=screen.materialization.sha256,
                cell_id=cell.cell_id,
                status="COMPLETE",
                started_ns=310,
                finished_ns=320,
                result_identity_sha256=result_sha,
                validator_kind="run_manifest",
                validator_protocol_sha256=_sha("fixture-validator"),
                source=FormalSingleOperatorJsonBinding.bind(
                    source_path, label="screen actual"
                ),
                reducer_payload={"kind": "fixture-screen-payload"},
            )
        )
    actual_rows = tuple(sorted(actuals, key=lambda row: row.cell_id))
    actual_set_sha = single._content_sha256([row.to_dict() for row in actual_rows])
    winner_configuration = [
        [name, dict(winner_cell.dimensions)[name]]
        for name, _levels in E4_SCREEN_FACTOR_LEVELS
    ]
    neighborhoods = [
        [name, levels[0], levels[1]] for name, levels in E4_SCREEN_FACTOR_LEVELS
    ]
    selection_sha = _sha("screen-selection")
    decision = FormalSingleOperatorStageDecision(
        schema_version=1,
        kind="formal_single_operator_stage_decision",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node="e4_screen",
        ordinal=8,
        stage="E4",
        phase="screen",
        predecessor_completion_sha256=screen.artifact.predecessor_completion_sha256,
        materialization_sha256=screen.materialization.sha256,
        actual_result_set_sha256=actual_set_sha,
        decision_kind="e4_screen_actual_reduced",
        next_materialization_source_decision_sha256=selection_sha,
        next_materialization_upstream_receipt_sha256s=(screen.materialization.sha256,),
        payload={
            "schema_version": 1,
            "kind": "formal_single_operator_e4_selection",
            "phase": "screen",
            "model": winner_cell.model,
            "lightcone_recipe_sha256": fixture["recipe"].sha256,
            "inventory_sha256": fixture["inventory"].sha256,
            "winner_configuration": winner_configuration,
            "factor_neighborhoods": neighborhoods,
            "evaluation_sha256s": [],
            "selection_sha256": selection_sha,
        },
    )
    decision_path = chain_root / "screen-decision.json"
    publish_formal_single_operator_json_artifact(decision_path, decision.to_dict())
    completion = FormalSingleOperatorStageCompletion(
        schema_version=1,
        kind="formal_single_operator_stage_completion",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node="e4_screen",
        ordinal=8,
        stage="E4",
        phase="screen",
        predecessor_source=screen.artifact.predecessor_source,
        predecessor_completion_sha256=screen.artifact.predecessor_completion_sha256,
        protocol_lock_sha256=fixture["protocol_lock"].sha256,
        node_materialization_source=FormalSingleOperatorJsonBinding.bind(
            chain_root / "screen-node.json", label="screen node"
        ),
        node_materialization_sha256=screen.artifact.sha256,
        materialization_sha256=screen.materialization.sha256,
        actual_results=actual_rows,
        actual_result_set_sha256=actual_set_sha,
        decision_source=FormalSingleOperatorJsonBinding.bind(
            decision_path, label="screen decision"
        ),
        decision_sha256=decision.sha256,
        completed_ns=330,
    )
    completion_path = chain_root / "screen-completion.json"
    publish_formal_single_operator_json_artifact(completion_path, completion.to_dict())
    local = materialize_formal_single_operator_node(
        node="e4_local",
        predecessor_completion_path=completion_path,
        protocol_lock_path=None,
        materialization_output_path=chain_root / "local-materialization.json",
        node_materialization_output_path=chain_root / "local-node.json",
        created_ns=340,
    )
    local_source = chain_root / "local-source.json"
    publish_formal_single_operator_execution_source(
        node_materialization_path=chain_root / "local-node.json",
        output_path=local_source,
    )
    local_cell = next(
        row
        for row in local.materialization.cells
        if dict(row.dimensions)["load"] == "low"
        and dict(row.dimensions)["traffic"] == "pure_decode"
    )
    output = (tmp_path / "local-run").resolve()
    output.mkdir()
    output.chmod(0o700)
    context = materialize_formal_single_operator_e4_compile_launch(
        execution_source_path=local_source,
        materialized_cell_id=local_cell.cell_id,
        repository_root=fixture["repository"],
        inventory_path=fixture["inventory_path"],
        private_output_root=output,
    )
    assert context.assignment_actual.cell_id == winner_cell.cell_id
    assert context.assignment_manifest.cell_id == winner_cell.cell_id
    assert context.run_config.runtime.max_running_requests == 1


def test_e4_mapper_rejects_profiler_foreign_cell_and_tampered_launch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    chain_root = fixture["chain_root"]
    screen = materialize_formal_single_operator_node(
        node="e4_screen",
        predecessor_completion_path=fixture["e2_completion_path"],
        protocol_lock_path=None,
        materialization_output_path=chain_root / "screen-materialization.json",
        node_materialization_output_path=chain_root / "screen-node.json",
        created_ns=300,
    )
    source_path = chain_root / "screen-source.json"
    publish_formal_single_operator_execution_source(
        node_materialization_path=chain_root / "screen-node.json",
        output_path=source_path,
    )
    cell = screen.materialization.cells[0]
    output = (tmp_path / "run").resolve()
    output.mkdir()
    output.chmod(0o700)
    context = materialize_formal_single_operator_e4_compile_launch(
        execution_source_path=source_path,
        materialized_cell_id=cell.cell_id,
        repository_root=fixture["repository"],
        inventory_path=fixture["inventory_path"],
        private_output_root=output,
    )
    foreign_output = (tmp_path / "foreign").resolve()
    foreign_output.mkdir()
    foreign_output.chmod(0o700)
    with pytest.raises(ValueError, match="outside current materialization"):
        materialize_formal_single_operator_e4_compile_launch(
            execution_source_path=source_path,
            materialized_cell_id=_sha("foreign-cell"),
            repository_root=fixture["repository"],
            inventory_path=fixture["inventory_path"],
            private_output_root=foreign_output,
        )
    launch_path = output / "formal-single-operator-e4-compile-launch.json"
    launch_path.chmod(0o600)
    value = json.loads(launch_path.read_text(encoding="utf-8"))
    value["localhost_port"] = context.launch.localhost_port + 1
    launch_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    Path(f"{launch_path}.sha256").write_text(
        f"{single._content_sha256(value)}\n", encoding="ascii"
    )
    with pytest.raises(ValueError):
        revalidate_formal_single_operator_e4_compile_launch(
            execution_source_path=source_path,
            materialized_cell_id=cell.cell_id,
            repository_root=fixture["repository"],
            inventory_path=fixture["inventory_path"],
            compile_launch_manifest_path=launch_path,
        )


def test_e4_physical_schedule_uses_mapped_load_and_rejects_profiler() -> None:
    headline = MaterializedCell(
        stage="E4",
        method_role="LightCone",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="mechanism_strength2_screen_headline",
        publication_policy="first_ready",
        recipe_sha256=_sha("recipe"),
        dimensions=tuple(
            sorted(
                {
                    "coalescing": 1,
                    "load": "saturation",
                    "microbatch": 1,
                    "screen_row": 0,
                    "stream_priority": "default",
                    "traffic": "pure_decode",
                    "update_stride": 1,
                }.items()
            )
        ),
    )
    schedule = _load_protocol_for_cell(
        cell=headline,
        max_running_requests=64,
        server_context_limit=40_960,
    )
    assert schedule["context_tokens"] == 40_928
    assert schedule["regime"] == "pure_decode"
    assert schedule["arrival_policy"] == "closed_loop_zero_think"
    assert schedule["max_running_requests"] == 64

    profiler = replace(
        headline,
        task="mechanism_profile_only",
        dimensions=(("profile", "nsys"),),
    )
    with pytest.raises(ValueError, match="dedicated_e4_profiler_schedule_required"):
        _load_protocol_for_cell(
            cell=profiler,
            max_running_requests=1,
            server_context_limit=40_960,
        )


def test_e4_mapper_has_no_caller_runtime_or_recipe_scalar() -> None:
    assert tuple(
        inspect.signature(
            materialize_formal_single_operator_e4_compile_launch
        ).parameters
    ) == (
        "execution_source_path",
        "materialized_cell_id",
        "repository_root",
        "inventory_path",
        "private_output_root",
    )
    forbidden = {
        "argv",
        "port",
        "recipe",
        "recipe_sha256",
        "run_config",
        "optimizer",
        "update_stride",
        "microbatch",
        "coalescing",
        "stream_priority",
    }
    assert forbidden.isdisjoint(
        inspect.signature(
            materialize_formal_single_operator_e4_compile_launch
        ).parameters
    )


def _downstream_finalizer_preflight_inputs(*, root: Path, inventory_path: Path) -> Path:
    common_path = root / "preflight-common.json"
    publish_canonical_json_no_replace(common_path, {"kind": "fixture"})
    common = CanonicalJsonProofBinding.bind(common_path)
    workload_path = root / "preflight-workload.json"
    publish_canonical_json_no_replace(workload_path, {"kind": "fixture-workload"})
    request_bindings = []
    for index in range(8):
        path = root / f"preflight-request-{index}.json"
        publish_canonical_json_no_replace(path, {"index": index})
        request_bindings.append(CanonicalJsonProofBinding.bind(path))
    inputs = FormalPreflightExecutionInputs(
        schema_version=2,
        kind="formal_single_operator_exact_ten_preflight_inputs",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
        authority_mode="formal_single_operator_v1",
        execution_authority=common,
        inventory=CanonicalJsonProofBinding.bind(inventory_path),
        content_receipt=common,
        workload_authority=ContentJsonArtifactBinding.from_path(
            "fixture-workload", workload_path
        ),
        doctor_report=common,
        compile_assignment_plan=common,
        exactness_assignment=common,
        interference_manifest=common,
        request_schedule_sources=tuple(request_bindings),
        tokenization_inputs=tuple(request_bindings),
        tokenization_outputs=tuple(request_bindings),
    )
    path = root / "preflight-inputs.json"
    publish_canonical_json_no_replace(path, inputs.to_dict())
    return path


def _downstream_finalizer_schedule(
    *,
    root: Path,
    cell: MaterializedCell,
    materialization_path: Path,
    launch_path: Path,
    launch: CompileLaunchManifest,
    execution_binding_sha256: str,
    subject_sha256: str,
    workload_authority_sha256: str,
) -> FormalServingRequestScheduleReceipt:
    workload_path = root / "workload.json"
    schedule_source_path = root / "schedule-source.json"
    token_input_path = root / "token-input.json"
    token_output_path = root / "token-output.json"
    content_path = root / "content-receipt.json"
    for path, value in (
        (workload_path, {"kind": "fixture-workload", "rows": ["sample-0"]}),
        (schedule_source_path, {"kind": "fixture-schedule-source"}),
        (token_input_path, {"kind": "fixture-token-input"}),
        (token_output_path, {"kind": "fixture-token-output"}),
        (content_path, {"kind": "fixture-content-receipt"}),
    ):
        publish_canonical_json_no_replace(path, value)
    request = BoundServingRequest(
        request_id="scored-0",
        namespace="downstream-finalizer-test",
        split="test",
        ordinal=0,
        input_token_ids=(1, 2),
        requested_output_tokens=2,
        arrival_us=0,
        cancellation_offset_us=None,
        cohort_id="cohort-0",
        cohort_sha256=single._content_sha256("cohort-0"),
        route_id="single-replica",
        sampling=FrozenSamplingParameters.from_mapping(
            {"temperature": 0.0, "max_new_tokens": 2}
        ),
    )
    source_member_sha256 = _sha("source-member")
    row = FormalServingRequestScheduleRow(
        source_member_sha256=source_member_sha256,
        source_sample_id="sample-0",
        prompt_sha256=_sha("prompt"),
        phase="scored",
        routed_dp_rank=None,
        request=request,
        tokenized_input_sha256=single._content_sha256([1, 2]),
    )
    return FormalServingRequestScheduleReceipt(
        schema_version=3,
        kind="formal_serving_request_schedule_receipt",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=execution_binding_sha256,
        subject_sha256=subject_sha256,
        materialized_cell_id=cell.cell_id,
        workload_authority_sha256=workload_authority_sha256,
        content_verification_receipt_sha256=CanonicalJsonProofBinding.bind(
            content_path
        ).semantic_sha256,
        topology_mode="tp1_dp1",
        materialization=CanonicalJsonProofBinding.bind(materialization_path),
        content_verification_receipt=CanonicalJsonProofBinding.bind(content_path),
        workload_source=ContentJsonArtifactBinding.from_path(
            "fixture-workload", workload_path
        ),
        compile_launch_manifest=CanonicalJsonProofBinding.bind(launch_path),
        sampling_profile=CanonicalJsonProofBinding.bind(launch.sampling_profile_path),
        schedule_source=ContentJsonArtifactBinding.from_path(
            "fixture-schedule", schedule_source_path
        ),
        tokenization_input=CanonicalJsonProofBinding.bind(token_input_path),
        tokenization_output=CanonicalJsonProofBinding.bind(token_output_path),
        tokenizer_worker_source_raw_sha256=_sha("tokenizer-worker"),
        tokenizer_worker_source_size=1,
        tokenizer_worker_argv_sha256=_sha("tokenizer-argv"),
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_snapshot_path=launch.tokenizer_snapshot_path,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_content_authority_sha256=(launch.tokenizer_content_authority_sha256),
        transformers_version="4.test",
        tokenizer_class="FixtureTokenizer",
        tokenizer_vocab_size=32,
        requests=(row,),
    )


def _downstream_finalizer_case(
    tmp_path: Path,
    *,
    prepared: bool,
) -> dict[str, object]:
    fixture = _fixture(tmp_path)
    chain_root = fixture["chain_root"]
    screen = materialize_formal_single_operator_node(
        node="e4_screen",
        predecessor_completion_path=fixture["e2_completion_path"],
        protocol_lock_path=None,
        materialization_output_path=chain_root / "finalizer-materialization.json",
        node_materialization_output_path=chain_root / "finalizer-node.json",
        created_ns=300,
    )
    source_path = chain_root / "finalizer-execution-source.json"
    publish_formal_single_operator_execution_source(
        node_materialization_path=chain_root / "finalizer-node.json",
        output_path=source_path,
    )
    cell = screen.materialization.cells[0]
    run_root = (tmp_path / ("prepared-run" if prepared else "direct-run")).resolve()
    run_root.mkdir()
    run_root.chmod(0o700)
    context = materialize_formal_single_operator_e4_compile_launch(
        execution_source_path=source_path,
        materialized_cell_id=cell.cell_id,
        repository_root=fixture["repository"],
        inventory_path=fixture["inventory_path"],
        private_output_root=run_root,
    )
    launch_path = run_root / "formal-single-operator-e4-compile-launch.json"
    preflight_path = _downstream_finalizer_preflight_inputs(
        root=run_root,
        inventory_path=fixture["inventory_path"],
    )
    source_binding = CanonicalJsonProofBinding.bind(source_path)
    execution_source_sha256 = json.loads(source_path.read_text(encoding="utf-8"))[
        "execution_source_sha256"
    ]
    materialization_binding = CanonicalJsonProofBinding.bind(
        chain_root / "finalizer-materialization.json"
    )
    if prepared:
        prepared_bundle_path = run_root / "prepared-bundle.json"
        content_path = run_root / "prepared-content.json"
        publish_canonical_json_no_replace(
            prepared_bundle_path, {"kind": "fixture-prepared-bundle"}
        )
        publish_canonical_json_no_replace(
            content_path, {"kind": "fixture-prepared-content"}
        )
        execution_binding_sha256 = _sha("prepared-execution-binding")
        subject_sha256 = _sha("prepared-subject")
    else:
        descriptor_seed = FormalSingleOperatorDownstreamRunPlanInputs(
            schema_version=1,
            kind="formal_single_operator_downstream_run_plan_inputs",
            execution_source=source_binding,
            execution_source_sha256=execution_source_sha256,
            materialized_cell_id=cell.cell_id,
            stage="E4",
            materialization=materialization_binding,
            materialization_sha256=screen.materialization.sha256,
            preflight_inputs=CanonicalJsonProofBinding.bind(preflight_path),
            compile_launch_manifest=CanonicalJsonProofBinding.bind(launch_path),
            private_output_root=str(run_root),
        )
        execution_binding_sha256 = descriptor_seed.sha256
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            _early_run_subject_sha,
        )

        subject_sha256 = _early_run_subject_sha(
            descriptor_seed,
            inventory_sha256=fixture["inventory"].sha256,
        )
    schedule = _downstream_finalizer_schedule(
        root=run_root,
        cell=cell,
        materialization_path=chain_root / "finalizer-materialization.json",
        launch_path=launch_path,
        launch=context.launch,
        execution_binding_sha256=execution_binding_sha256,
        subject_sha256=subject_sha256,
        workload_authority_sha256=(
            fixture["protocol_lock"].formal_workload_e3a_authorization_sha256
        ),
    )
    schedule_path = run_root / "formal-request-schedule-receipt.json"
    publish_canonical_json_no_replace(schedule_path, schedule.to_dict())
    if prepared:
        descriptor = FormalSingleOperatorPreparedDownstreamRunPlanInputs(
            schema_version=1,
            kind="formal_single_operator_prepared_downstream_run_plan_inputs",
            execution_source=source_binding,
            execution_source_sha256=execution_source_sha256,
            prepared_launch_bundle=CanonicalJsonProofBinding.bind(prepared_bundle_path),
            prepared_launch_bundle_sha256=_sha("prepared-bundle"),
            prepared_launch_entry_sha256=_sha("prepared-entry"),
            materialized_cell_id=cell.cell_id,
            stage="E4",  # type: ignore[arg-type]
            materialization=materialization_binding,
            materialization_sha256=screen.materialization.sha256,
            inventory=CanonicalJsonProofBinding.bind(fixture["inventory_path"]),
            content_verification_receipt=CanonicalJsonProofBinding.bind(content_path),
            compile_launch_manifest=CanonicalJsonProofBinding.bind(launch_path),
            request_schedule_receipt=CanonicalJsonProofBinding.bind(schedule_path),
            execution_binding_sha256=execution_binding_sha256,
            subject_sha256=subject_sha256,
            private_output_root=str(run_root),
        )
        descriptor_path = (
            run_root / "formal-single-operator-prepared-downstream-inputs.json"
        )
    else:
        descriptor = descriptor_seed
        descriptor_path = (
            run_root / "formal-single-operator-downstream-run-plan-inputs.json"
        )
    publish_canonical_json_no_replace(descriptor_path, descriptor.to_dict())
    native_binding = NativeTerminalRunBinding(
        run_id=f"single-{cell.cell_id[:24]}",
        run_nonce_sha256=_sha("finalizer-run-nonce"),
        execution_plan_sha256=_sha("finalizer-execution-plan"),
        rank_config_sha256=_sha("finalizer-rank-config"),
        attempt_id="attempt-0",
        session_id=f"single-{cell.cell_id[:24]}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha("finalizer-challenge"),
        method=context.run_config.method,
        warmup_request_ids=(),
        scored_request_ids=("scored-0",),
    )
    outputs = {
        "terminal_output_path": run_root / "unsigned-native-terminal.json",
        "native_itl_pointer_output_path": run_root / "unsigned-native-itl.json",
        "live_run_receipt_output_path": run_root / "unsigned-live-run.json",
        "lifecycle_timing_output_path": run_root / "unsigned-lifecycle.json",
        "server_log_output_path": run_root / "server.log",
        "server_stdout_output_path": run_root / "stdout.log",
        "server_stderr_output_path": run_root / "stderr.log",
        "junit_output_path": run_root / "junit.xml",
        "before_gpu_snapshot_output_path": run_root / "before-gpu.json",
        "ready_gpu_snapshot_output_path": run_root / "ready-gpu.json",
        "after_gpu_snapshot_output_path": run_root / "after-gpu.json",
        "fatal_output_path": run_root / "fatal.json",
    }
    plan = FormalServingRunPlan(
        schema_version=2,
        kind="formal_serving_run_plan",
        protocol_sha256=FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        execution_binding_sha256=execution_binding_sha256,
        subject_sha256=subject_sha256,
        materialized_cell_id=cell.cell_id,
        stage="E4",
        method=context.run_config.method,
        topology_mode="tp1_dp1",
        inventory_sha256=fixture["inventory"].sha256,
        gpu_uuids=context.launch.gpu_uuids,
        runtime_gpu_proof_sha256s=(),
        runtime_gpu_proof_artifacts=(),
        nextn_tp2_authority_sha256=None,
        launch_manifest=CanonicalJsonProofBinding.bind(launch_path),
        request_schedule_receipt=CanonicalJsonProofBinding.bind(schedule_path),
        native_terminal_binding=native_binding,
        private_output_root=str(run_root),
        formal_gang_terminal_output_path=None,
        single_operator_execution_rebuild_source=(
            CanonicalJsonProofBinding.bind(descriptor_path)
        ),
        **{name: str(path) for name, path in outputs.items()},
    )
    plan_path = run_root / "formal-serving-run-plan.json"
    publish_canonical_json_no_replace(plan_path, plan.to_dict())
    for name, path in outputs.items():
        if name == "fatal_output_path":
            continue
        if path.suffix == ".xml":
            path.write_text('<testsuite tests="1" failures="0"/>\n')
        elif path.suffix == ".log":
            path.write_text(f"{name}\n")
        else:
            publish_canonical_json_no_replace(
                path,
                {
                    "kind": (
                        "unsigned_pinned_sglang_serving_run_receipt"
                        if name == "live_run_receipt_output_path"
                        else name
                    )
                },
            )
    return {
        **fixture,
        "cell": cell,
        "materialization": screen.materialization,
        "descriptor_path": descriptor_path,
        "run_root": run_root,
        "plan": plan,
        "plan_path": plan_path,
        "launch": context.launch,
        "schedule": schedule,
    }


@pytest.mark.parametrize("prepared", (False, True))
def test_schema2_downstream_finalizer_publishes_replayable_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: bool,
) -> None:
    case = _downstream_finalizer_case(tmp_path, prepared=prepared)
    plan = case["plan"]
    launch = case["launch"]
    schedule = case["schedule"]
    from lightcone_spec.experiments import formal_single_operator_stages as stages
    from lightcone_spec.orchestration import formal_physical_dispatch as dispatch
    from lightcone_spec.orchestration import live_sglang

    monkeypatch.setattr(
        dispatch,
        "_load_formal_single_operator_trusted_run_plan",
        lambda _path: (plan, launch, schedule),
    )
    live_path = Path(plan.live_run_receipt_output_path)
    live_binding = CanonicalJsonProofBinding.bind(live_path)
    fake_live = SimpleNamespace(
        sha256=live_binding.semantic_sha256,
        launch_manifest=plan.launch_manifest,
        inventory_sha256=plan.inventory_sha256,
        gpu_uuids=plan.gpu_uuids,
        run_binding_sha256=single._content_sha256(
            plan.native_terminal_binding.begin_payload()
        ),
        process_group_empty=True,
        server_process_started_ns=10,
        process_group_empty_checked_ns=20,
        process_exit_code=0,
    )
    monkeypatch.setattr(
        live_sglang.UnsignedPinnedSglangServingRunReceipt,
        "from_dict",
        classmethod(lambda _cls, _value: fake_live),
    )
    manifest = single.finalize_formal_single_operator_run(
        repository_root=case["repository"],
        run_plan_path=case["plan_path"],
    )
    manifest_path = case["run_root"] / "formal-single-operator-manifest.json"
    assert (
        single.revalidate_formal_single_operator_run_manifest(
            repository_root=case["repository"],
            manifest_path=manifest_path,
        )
        == manifest
    )
    assert {"inventory", "materialization", "run_plan_inputs"} <= {
        row.name for row in manifest.artifacts
    }
    monkeypatch.setattr(
        stages,
        "_validated_single_operator_serving_payload",
        lambda **_kwargs: {"kind": "validated-fixture-serving"},
    )
    validation = FormalSingleOperatorRunManifestActualValidator(
        str(case["repository"])
    ).validate(
        path=manifest_path,
        node=formal_single_operator_node_spec("e4_screen"),
        materialization=case["materialization"],
        cell=case["cell"],
    )
    assert validation.status == "COMPLETE"
    assert validation.result_identity_sha256 == manifest.sha256


def test_schema2_downstream_finalizer_rejects_foreign_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _downstream_finalizer_case(tmp_path, prepared=False)
    from lightcone_spec.orchestration import formal_physical_dispatch as dispatch

    monkeypatch.setattr(
        dispatch,
        "_load_formal_single_operator_trusted_run_plan",
        lambda _path: (case["plan"], case["launch"], case["schedule"]),
    )
    foreign = tmp_path / "foreign-inventory.json"
    publish_canonical_json_no_replace(foreign, {"kind": "foreign-inventory"})
    with pytest.raises(ValueError, match="caller inventory differs"):
        single.finalize_formal_single_operator_run(
            repository_root=case["repository"],
            run_plan_path=case["plan_path"],
            inventory_path=foreign,
        )


def test_schema2_prepared_finalizer_rejects_mutated_descriptor(
    tmp_path: Path,
) -> None:
    case = _downstream_finalizer_case(tmp_path, prepared=True)
    descriptor_path = case["descriptor_path"]
    descriptor_path.write_text(
        descriptor_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed|canonical JSON"):
        single.finalize_formal_single_operator_run(
            repository_root=case["repository"],
            run_plan_path=case["plan_path"],
        )
