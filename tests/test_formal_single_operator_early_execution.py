from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments.formal_preflight_execution import (
    FormalPreflightInterferenceExecutionManifest,
    FormalPreflightInterferenceRunInput,
)
from lightcone_spec.experiments.formal_preflight_inputs import (
    FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256,
    FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
    FormalPreflightExecutionInputs,
    FormalSingleOperatorPreflightCompletion,
    FormalSingleOperatorPreflightCompletionRow,
    FormalSingleOperatorPreflightInterferenceEvidence,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock, content_sha256
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_to_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_single_operator_early_execution import (
    materialize_formal_single_operator_early_compile_launch,
    materialize_formal_single_operator_early_run_plan_inputs,
    revalidate_formal_single_operator_early_compile_launch,
    revalidate_formal_single_operator_early_run_plan_inputs,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
    FormalSingleOperatorExecutionSource,
    FormalSingleOperatorNodeMaterialization,
    FormalSingleOperatorStageCompletion,
    FormalSingleOperatorStageDecision,
    FormalSingleOperatorValidatedActual,
    formal_single_operator_node_spec,
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
)
from lightcone_spec.experiments.load import FrozenSamplingParameters
from lightcone_spec.experiments.preflight_interference import (
    FormalPreflightInterferenceQualificationRow,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.serving import BoundServingRequest
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
)
from lightcone_spec.experiments.workload_authority import (
    FORMAL_WORKLOAD_PROTOCOLS,
    FormalWorkloadAuthority,
    FormalWorkloadSample,
    formal_workload_authority_artifact_id,
    formal_workload_authority_cli_artifact,
    formal_workload_samples_sha256,
)
from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding
from lightcone_spec.orchestration.runtime import _render_server
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
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical_with_sidecar(path: Path, value: object) -> CanonicalJsonProofBinding:
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
    Path(f"{path}.sha256").write_text(f"{content_sha256(value)}\n", encoding="ascii")
    return CanonicalJsonProofBinding.bind(path)


def _inventory() -> GpuInventory:
    return GpuInventory(
        schema_version=1,
        devices=(
            GpuDevice(
                uuid="GPU-early-test",
                host_id="host-early-test",
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


def _protocol_lock() -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="formal-single-operator-early-test",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=_sha("registry"),
        english_protocol_sha256=_sha("english"),
        chinese_protocol_sha256=_sha("chinese"),
        tts_calibration_authority_sha256=_sha("tts"),
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1-anchor"),
        e2_recipe_grid_authority_sha256=default_e2_recipe_grid_authority().sha256,
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


def _workload(path: Path) -> ContentJsonArtifactBinding:
    raw = path.parent / "livecodebench-raw.json"
    raw.write_text("{}\n", encoding="utf-8")
    sample = FormalWorkloadSample(
        source_row_id="question-0",
        sample_id="livecodebench-v6-hard-0",
        prompt="Write a deterministic function.",
        seed=1,
    )
    samples = (sample,)
    protocol = FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"]
    authority = FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id="livecodebench_v6_hard",
        raw_source_path=str(raw.resolve()),
        raw_file_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        repository_revision="3" * 40,
        raw_row_count=1,
        selected_row_count=1,
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=_sha("workload-lock"),
        protocol_sha256=protocol.sha256,
        samples=samples,
    )
    publish_canonical_json_no_replace(
        path, formal_workload_authority_cli_artifact(authority)
    )
    return ContentJsonArtifactBinding.from_path(
        formal_workload_authority_artifact_id("livecodebench_v6_hard"),
        path,
    )


def _template_launch(
    root: Path,
    *,
    inventory: GpuInventory,
    content_authority_sha256: str,
) -> CompileLaunchManifest:
    sampling = SamplingProfile()
    sampling_path = root / "sampling.json"
    sampling.write(sampling_path)
    config = RunConfig(
        method="static",
        model=ModelPair(
            target="Qwen/Qwen3-8B",
            target_revision="4" * 40,
            drafter_revision="5" * 40,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=sampling.sha256,
            device_identity=inventory.devices[0].uuid,
            max_running_requests=1,
        ),
    )
    checkout = root / "patched-sglang"
    target = root / "models" / "target" / config.model.target_revision
    drafter = root / "models" / "drafter" / config.model.drafter_revision
    tokenizer = root / "models" / "tokenizer" / ("6" * 40)
    cuda = root / "cuda"
    bin_path = root / "bin"
    library = root / "lib"
    for directory in (checkout, target, drafter, tokenizer, cuda, bin_path, library):
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
        max_running_requests=1,
        graph_buckets=(1,),
        allocator="cuda_malloc_async",
        build_flags=(),
    )
    cache_plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=root / "cache",
        cache_mode="build",
    )
    cache_path = root / "cache-plan.json"
    cache_plan.write(cache_path)
    rendered = _render_server(
        output=root,
        method="static",
        config=config,
        verified_checkout=checkout,
        roots={config.model.target: str(target), config.model.drafter: str(drafter)},
        target_id=config.model.target,
        drafter_id=config.model.drafter,
        adaptation_reserve_mb=0,
        mem_fraction_static=0.8,
        host="127.0.0.1",
        port=31_001,
        compile_cache_plan_path=cache_path,
    )
    config_binding = CanonicalJsonProofBinding.bind(rendered.run_config)
    cache_binding = CanonicalJsonProofBinding.bind(cache_path)
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=_sha("model-lock"),
        sampling_profile_sha256=sampling.sha256,
        payloads=(CompileOnlyPrewarmPayload("prewarm-0", 1, (1, 2), 2, 1),),
    )
    prewarm_path = root / "prewarm.json"
    write_compile_prewarm_manifest(prewarm, prewarm_path)
    prewarm_binding = CanonicalJsonProofBinding.bind(prewarm_path)
    prepared = _canonical_with_sidecar(
        root / "prepared.json", {"kind": "prepared-model-content-test"}
    )
    launch = CompileLaunchManifest(
        schema_version=1,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        run_config_path=config_binding.absolute_path,
        run_config_raw_sha256=config_binding.raw_sha256,
        run_config_semantic_sha256=config_binding.semantic_sha256,
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
        prepared_model_content_manifest_path=prepared.absolute_path,
        prepared_model_content_manifest_raw_sha256=prepared.raw_sha256,
        prepared_model_content_manifest_sha256=prepared.semantic_sha256,
        prepared_model_content_manifest_size=prepared.size,
        target_content_member_id="target-member",
        target_model_id=config.model.target,
        target_snapshot_path=str(target),
        target_revision=config.model.target_revision,
        target_content_authority_sha256=content_authority_sha256,
        drafter_content_member_id="drafter-member",
        drafter_model_id=config.model.drafter,
        drafter_snapshot_path=str(drafter),
        drafter_revision=config.model.drafter_revision,
        drafter_content_authority_sha256=content_authority_sha256,
        tokenizer_content_member_id="tokenizer-member",
        tokenizer_model_id=config.model.target,
        tokenizer_snapshot_path=str(tokenizer),
        tokenizer_revision=tokenizer.name,
        tokenizer_content_authority_sha256=content_authority_sha256,
        server_argv=rendered.argv,
        server_argv_sha256=content_sha256({"argv": list(rendered.argv)}),
        localhost_port=31_001,
        model_lock_sha256=prewarm.model_lock_sha256,
        sampling_profile_sha256=sampling.sha256,
        physical_assignment_sha256=_sha("preflight-assignment"),
        experiment_budget_sha256=_sha("preflight-budget"),
        budget_materialization_authority_sha256=_sha("preflight-budget-authority"),
        inventory_sha256=inventory.sha256,
        gpu_uuids=(inventory.devices[0].uuid,),
        path_entries=(str(bin_path),),
        library_path_entries=(str(library),),
        cuda_home=str(cuda),
    )
    launch.write(root / "compile-launch.json")
    return launch


def _preflight_cells() -> tuple[MaterializedCell, ...]:
    return tuple(
        sorted(
            (
                MaterializedCell(
                    stage="preflight",
                    method_role="Target-only",
                    model="Qwen/Qwen3-8B",
                    backend="NONE",
                    task=f"preflight-fixture-{index}",
                    publication_policy="none",
                    recipe_sha256=None,
                    dimensions=(("index", index),),
                )
                for index in range(10)
            ),
            key=lambda cell: cell.cell_id,
        )
    )


def _publish_current_e3a_source(tmp_path: Path) -> dict[str, Path | str]:
    lock = _protocol_lock()
    lock_path = tmp_path / "protocol-lock.json"
    lock_binding = publish_formal_single_operator_json_artifact(
        lock_path, protocol_lock_to_dict(lock)
    )
    inventory = _inventory()
    inventory_path = tmp_path / "inventory.json"
    publish_canonical_json_no_replace(inventory_path, inventory.to_dict())
    inventory_binding = CanonicalJsonProofBinding.bind(inventory_path)
    content_path = tmp_path / "content.json"
    publish_canonical_json_no_replace(content_path, {"kind": "content-fixture"})
    content_binding = CanonicalJsonProofBinding.bind(content_path)
    workload_path = tmp_path / "workload.json"
    workload_binding = _workload(workload_path)
    template_root = tmp_path / "preflight-launch"
    template_root.mkdir()
    launch = _template_launch(
        template_root,
        inventory=inventory,
        content_authority_sha256=lock.prepared_model_content_authorization_sha256,
    )
    rows = []
    for index in range(8):
        registry_cell_id = _sha(f"interference-{index}")
        request_id = f"score-{index}"
        request = BoundServingRequest(
            request_id=request_id,
            namespace="early-launch-test",
            split="preflight",
            ordinal=index,
            input_token_ids=(1, 2),
            requested_output_tokens=2,
            arrival_us=index,
            cancellation_offset_us=None,
            cohort_id="preflight-cohort",
            cohort_sha256=content_sha256("preflight-cohort"),
            route_id="single-replica",
            sampling=FrozenSamplingParameters.from_mapping(
                {"temperature": 0.0, "max_new_tokens": 2}
            ),
        )
        rows.append(
            FormalPreflightInterferenceRunInput(
                registry_cell_id=registry_cell_id,
                launch_manifest_path=str(template_root / "compile-launch.json"),
                run_binding=NativeTerminalRunBinding(
                    run_id=f"preflight-{index}",
                    run_nonce_sha256=_sha(f"nonce-{index}"),
                    execution_plan_sha256=_sha("preflight-plan"),
                    rank_config_sha256=_sha(f"rank-{index}"),
                    attempt_id="attempt-0000",
                    session_id=f"session-{index}",
                    session_epoch=1,
                    previous_run_id=None,
                    challenge_nonce_sha256=_sha(f"challenge-{index}"),
                    method="static",
                    warmup_request_ids=(),
                    scored_request_ids=(request_id,),
                ),
                warmup_requests=(),
                scored_requests=(request,),
                qualification_rows=(
                    FormalPreflightInterferenceQualificationRow(
                        request_id=request_id,
                        prompt_bucket="short",
                        eligible=True,
                    ),
                ),
            )
        )
    interference = FormalPreflightInterferenceExecutionManifest(
        schema_version=1,
        kind="formal_preflight_interference_execution_manifest",
        dispatch_receipt_semantic_sha256=_sha("dispatch"),
        inputs=tuple(sorted(rows, key=lambda row: row.registry_cell_id)),
    )
    interference_path = tmp_path / "interference.json"
    publish_canonical_json_no_replace(interference_path, interference.to_dict())
    common_path = tmp_path / "common.json"
    publish_canonical_json_no_replace(common_path, {"kind": "common"})
    common = CanonicalJsonProofBinding.bind(common_path)
    request_sources = []
    for index in range(8):
        path = tmp_path / f"request-source-{index}.json"
        publish_canonical_json_no_replace(path, {"index": index})
        request_sources.append(CanonicalJsonProofBinding.bind(path))
    inputs = FormalPreflightExecutionInputs(
        schema_version=2,
        kind="formal_single_operator_exact_ten_preflight_inputs",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
        authority_mode="formal_single_operator_v1",
        execution_authority=common,
        inventory=inventory_binding,
        content_receipt=content_binding,
        workload_authority=workload_binding,
        doctor_report=common,
        compile_assignment_plan=common,
        exactness_assignment=common,
        interference_manifest=CanonicalJsonProofBinding.bind(interference_path),
        request_schedule_sources=tuple(request_sources),
        tokenization_inputs=tuple(request_sources),
        tokenization_outputs=tuple(request_sources),
    )
    inputs_path = tmp_path / "preflight-inputs.json"
    publish_canonical_json_no_replace(inputs_path, inputs.to_dict())
    inputs_binding = CanonicalJsonProofBinding.bind(inputs_path)
    preflight_cells = _preflight_cells()
    preflight_materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="preflight",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(),
        source_decision_sha256=_sha("preflight-source"),
        materialization_rule="test_exact_ten",
        expected_cell_count=10,
        cells=preflight_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    completion_rows = []
    evidence = []
    runner_kinds = (
        "first_party_compile",
        "first_party_exactness",
        *("first_party_interference" for _ in range(8)),
    )
    for index, (cell, runner_kind) in enumerate(zip(preflight_cells, runner_kinds)):
        registry_cell_id = _sha(f"preflight-registry-{index}")
        completion_rows.append(
            FormalSingleOperatorPreflightCompletionRow(
                materialized_cell_id=cell.cell_id,
                registry_cell_id=registry_cell_id,
                runner_kind=runner_kind,  # type: ignore[arg-type]
                status="COMPLETE",
                started_ns=index + 1,
                finished_ns=index + 2,
                result_sha256=_sha(f"preflight-result-{index}"),
            )
        )
        if runner_kind == "first_party_interference":
            terminal_path = tmp_path / f"terminal-{index}.json"
            lifecycle_path = tmp_path / f"lifecycle-{index}.json"
            publish_canonical_json_no_replace(terminal_path, {"index": index})
            publish_canonical_json_no_replace(lifecycle_path, {"index": index})
            junit_path = tmp_path / f"junit-{index}.xml"
            junit_path.write_text("<testsuite/>\n", encoding="utf-8")
            evidence.append(
                FormalSingleOperatorPreflightInterferenceEvidence(
                    materialized_cell_id=cell.cell_id,
                    registry_cell_id=registry_cell_id,
                    terminal_result_proof=CanonicalJsonProofBinding.bind(terminal_path),
                    lifecycle_timing=CanonicalJsonProofBinding.bind(lifecycle_path),
                    junit_xml=EvidenceFileBinding.bind(junit_path, label="test JUnit"),
                )
            )
    completion_rows = sorted(completion_rows, key=lambda row: row.registry_cell_id)
    exact_ten = FormalSingleOperatorPreflightCompletion(
        schema_version=1,
        kind="formal_single_operator_exact_ten_preflight_completion",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_COMPLETION_PROTOCOL_SHA256,
        execution_inputs=inputs_binding,
        compile_result=common,
        exactness_result=common,
        interference_evidence=tuple(
            sorted(evidence, key=lambda row: row.registry_cell_id)
        ),
        rows=tuple(completion_rows),
        status="COMPLETE",
        started_ns=min(row.started_ns for row in completion_rows),
        finished_ns=max(row.finished_ns for row in completion_rows),
    )
    exact_path = tmp_path / "exact-ten.json"
    exact_binding = publish_formal_single_operator_json_artifact(
        exact_path, exact_ten.to_dict()
    )
    by_cell = {row.materialized_cell_id: row for row in completion_rows}
    actuals = tuple(
        sorted(
            (
                FormalSingleOperatorValidatedActual(
                    node="preflight",
                    stage="preflight",
                    materialization_sha256=preflight_materialization.sha256,
                    cell_id=cell.cell_id,
                    status="COMPLETE",
                    started_ns=by_cell[cell.cell_id].started_ns,
                    finished_ns=by_cell[cell.cell_id].finished_ns,
                    result_identity_sha256=content_sha256(
                        {
                            "completion_sha256": exact_ten.sha256,
                            "cell_id": cell.cell_id,
                            "result_sha256": by_cell[cell.cell_id].result_sha256,
                        }
                    ),
                    validator_kind="typed_source_closure_fixture",
                    validator_protocol_sha256=_sha("fixture-validator"),
                    source=exact_binding,
                    reducer_payload={},
                )
                for cell in preflight_cells
            ),
            key=lambda row: row.cell_id,
        )
    )
    actual_set_sha256 = content_sha256([row.to_dict() for row in actuals])
    preflight_materialization_path = tmp_path / "preflight-materialization.json"
    preflight_materialization_binding = publish_formal_single_operator_json_artifact(
        preflight_materialization_path,
        stage_materialization_receipt_to_dict(preflight_materialization),
    )
    preflight_spec = formal_single_operator_node_spec("preflight")
    preflight_node = FormalSingleOperatorNodeMaterialization(
        schema_version=1,
        kind="formal_single_operator_node_materialization",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node="preflight",
        ordinal=preflight_spec.ordinal,
        stage="preflight",
        phase=preflight_spec.phase,
        predecessor_source=None,
        predecessor_completion_sha256=None,
        protocol_lock_source=lock_binding,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_manifest_sha256=lock.formal_runtime_authority_manifest_sha256,
        prepared_model_content_authorization_sha256=lock.prepared_model_content_authorization_sha256,
        formal_workload_e3a_authorization_sha256=lock.formal_workload_e3a_authorization_sha256,
        formal_workload_e0_authorization_sha256=lock.formal_workload_e0_authorization_sha256,
        burstgpt_shape_authorization_sha256=lock.burstgpt_shape_authorization_sha256,
        materialization_source=preflight_materialization_binding,
        materialization_sha256=preflight_materialization.sha256,
        created_ns=1,
    )
    preflight_node_path = tmp_path / "preflight-node.json"
    preflight_node_binding = publish_formal_single_operator_json_artifact(
        preflight_node_path, preflight_node.to_dict()
    )
    decision = FormalSingleOperatorStageDecision(
        schema_version=1,
        kind="formal_single_operator_stage_decision",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node="preflight",
        ordinal=preflight_spec.ordinal,
        stage="preflight",
        phase=preflight_spec.phase,
        predecessor_completion_sha256=None,
        materialization_sha256=preflight_materialization.sha256,
        actual_result_set_sha256=actual_set_sha256,
        decision_kind="preflight_all_complete",
        next_materialization_source_decision_sha256=lock.formal_workload_e3a_authorization_sha256,
        next_materialization_upstream_receipt_sha256s=(exact_ten.sha256,),
        payload={"kind": "typed_source_closure_fixture"},
    )
    decision_path = tmp_path / "preflight-decision.json"
    decision_binding = publish_formal_single_operator_json_artifact(
        decision_path, decision.to_dict()
    )
    preflight_completion = FormalSingleOperatorStageCompletion(
        schema_version=1,
        kind="formal_single_operator_stage_completion",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node="preflight",
        ordinal=preflight_spec.ordinal,
        stage="preflight",
        phase=preflight_spec.phase,
        predecessor_source=None,
        predecessor_completion_sha256=None,
        protocol_lock_sha256=lock.sha256,
        node_materialization_source=preflight_node_binding,
        node_materialization_sha256=preflight_node.sha256,
        materialization_sha256=preflight_materialization.sha256,
        actual_results=actuals,
        actual_result_set_sha256=actual_set_sha256,
        decision_source=decision_binding,
        decision_sha256=decision.sha256,
        completed_ns=20,
    )
    preflight_completion_path = tmp_path / "preflight-completion.json"
    preflight_completion_binding = publish_formal_single_operator_json_artifact(
        preflight_completion_path, preflight_completion.to_dict()
    )
    e3a_cell = MaterializedCell(
        stage="E3a",
        method_role="Static",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="capacity_screen",
        publication_policy="tuning_only",
        recipe_sha256=None,
        dimensions=(
            ("concurrency", 4),
            ("context", 4096),
            ("regime", "short_input_long_generation"),
            ("width", 8),
        ),
    )
    e3a_materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E3a",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(exact_ten.sha256,),
        source_decision_sha256=lock.formal_workload_e3a_authorization_sha256,
        materialization_rule="typed_source_closure_e3a",
        expected_cell_count=1,
        cells=(e3a_cell,),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    e3a_materialization_path = tmp_path / "e3a-materialization.json"
    e3a_materialization_binding = publish_formal_single_operator_json_artifact(
        e3a_materialization_path,
        stage_materialization_receipt_to_dict(e3a_materialization),
    )
    e3a_spec = formal_single_operator_node_spec("e3a")
    source = FormalSingleOperatorExecutionSource(
        schema_version=1,
        kind="formal_single_operator_execution_source",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_STAGE_SEQUENCE_PROTOCOL_SHA256,
        node="e3a",
        ordinal=e3a_spec.ordinal,
        stage="E3a",
        phase=e3a_spec.phase,
        protocol_lock_source=lock_binding,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_manifest_sha256=lock.formal_runtime_authority_manifest_sha256,
        prepared_model_content_authorization_sha256=lock.prepared_model_content_authorization_sha256,
        formal_workload_e3a_authorization_sha256=lock.formal_workload_e3a_authorization_sha256,
        formal_workload_e0_authorization_sha256=lock.formal_workload_e0_authorization_sha256,
        burstgpt_shape_authorization_sha256=lock.burstgpt_shape_authorization_sha256,
        predecessor_completion_source=preflight_completion_binding,
        predecessor_completion_sha256=preflight_completion.sha256,
        predecessor_decision_sha256=decision.sha256,
        materialization_source=e3a_materialization_binding,
        materialization_sha256=e3a_materialization.sha256,
        materialization_source_decision_sha256=e3a_materialization.source_decision_sha256,
        materialization_upstream_receipt_sha256s=e3a_materialization.upstream_receipt_sha256s,
    )
    source_path = tmp_path / "e3a-execution-source.json"
    publish_canonical_json_no_replace(source_path, source.to_dict())
    assert launch.inventory_sha256 == inventory.sha256
    return {
        "source": source_path,
        "cell_id": e3a_cell.cell_id,
        "inventory": inventory_path,
        "content": content_path,
        "workload": workload_path,
        "preflight_inputs": inputs_path,
    }


def test_early_mapper_has_no_caller_runtime_scalars_and_materializes_e3a(
    tmp_path: Path,
) -> None:
    fixture = _publish_current_e3a_source(tmp_path)
    output = (tmp_path / "run").resolve()
    output.mkdir()
    signature = inspect.signature(
        materialize_formal_single_operator_early_compile_launch
    )
    assert not {
        "run_config",
        "argv",
        "port",
        "recipe_sha256",
        "gpu_uuid",
        "method",
        "inventory_path",
        "content_verification_receipt_path",
        "workload_authority_path",
        "runtime_gpu_proof_artifact_paths",
    } & set(signature.parameters)
    assert set(signature.parameters) == {
        "execution_source_path",
        "materialized_cell_id",
        "preflight_inputs_path",
        "private_output_root",
    }
    context = materialize_formal_single_operator_early_compile_launch(
        execution_source_path=fixture["source"],
        materialized_cell_id=str(fixture["cell_id"]),
        preflight_inputs_path=fixture["preflight_inputs"],
        private_output_root=output,
    )
    assert context.run_config.method == "static"
    assert context.run_config.runtime.max_running_requests == 4
    assert context.run_config.runtime.speculative_num_draft_tokens == 8
    assert context.launch.localhost_port != 31_001
    assert (
        revalidate_formal_single_operator_early_compile_launch(
            execution_source_path=fixture["source"],
            materialized_cell_id=str(fixture["cell_id"]),
            preflight_inputs_path=fixture["preflight_inputs"],
            compile_launch_manifest_path=(
                output / "formal-single-operator-early-compile-launch.json"
            ),
        )
        == context
    )
    with pytest.raises(RuntimeError, match="refuses to replace"):
        materialize_formal_single_operator_early_compile_launch(
            execution_source_path=fixture["source"],
            materialized_cell_id=str(fixture["cell_id"]),
            preflight_inputs_path=fixture["preflight_inputs"],
            private_output_root=output,
        )


def test_early_plan_inputs_reopen_and_reject_foreign_or_tampered_sources(
    tmp_path: Path,
) -> None:
    fixture = _publish_current_e3a_source(tmp_path)
    output = (tmp_path / "run").resolve()
    output.mkdir()
    value = materialize_formal_single_operator_early_run_plan_inputs(
        execution_source_path=fixture["source"],
        materialized_cell_id=str(fixture["cell_id"]),
        preflight_inputs_path=fixture["preflight_inputs"],
        private_output_root=output,
    )
    descriptor = output / "formal-single-operator-early-run-plan-inputs.json"
    assert revalidate_formal_single_operator_early_run_plan_inputs(descriptor) == value
    original_inputs = FormalPreflightExecutionInputs.from_dict(
        CanonicalJsonProofBinding.bind(fixture["preflight_inputs"]).reopen()
    )
    foreign_inventory_value = replace(
        _inventory(), source_receipt_sha256=_sha("foreign-inventory-source")
    )
    foreign_inventory = tmp_path / "foreign-inventory.json"
    publish_canonical_json_no_replace(
        foreign_inventory, foreign_inventory_value.to_dict()
    )
    foreign_inputs = replace(
        original_inputs,
        inventory=CanonicalJsonProofBinding.bind(foreign_inventory),
    )
    foreign_inputs_path = tmp_path / "foreign-preflight-inputs.json"
    publish_canonical_json_no_replace(foreign_inputs_path, foreign_inputs.to_dict())
    other_output = (tmp_path / "foreign-run").resolve()
    other_output.mkdir()
    with pytest.raises(ValueError, match="no reusable TP1 launch"):
        materialize_formal_single_operator_early_compile_launch(
            execution_source_path=fixture["source"],
            materialized_cell_id=str(fixture["cell_id"]),
            preflight_inputs_path=foreign_inputs_path,
            private_output_root=other_output,
        )
    launch_path = output / "formal-single-operator-early-compile-launch.json"
    body = launch_path.read_bytes()
    launch_path.write_bytes(body.replace(b'"localhost_port":', b'"localhost_port":1'))
    with pytest.raises((ValueError, RuntimeError)):
        revalidate_formal_single_operator_early_run_plan_inputs(descriptor)


def test_early_plan_inputs_are_not_a_legacy_sealed_run_plan() -> None:
    assert set(
        inspect.signature(
            materialize_formal_single_operator_early_run_plan_inputs
        ).parameters
    ) == {
        "execution_source_path",
        "materialized_cell_id",
        "preflight_inputs_path",
        "private_output_root",
    }
