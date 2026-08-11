from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

import lightcone_spec.orchestration.executor as executor_module
import lightcone_spec.orchestration.session as session_module
import lightcone_spec.telemetry.writer as writer_module
from lightcone_spec.config.schema import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuDispatchExecutionContext,
    GpuInventory,
    GpuPoolScheduler,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ProductionLoadPlan,
    ProductionWindow,
    RequestOutcome,
    RequestTemplate,
    TokenChunkTiming,
    closed_loop_corpus,
    controlled_poisson_corpus,
    immediate_burst_corpus,
)
from lightcone_spec.experiments.planning import (
    BudgetJobKind,
    BudgetObservationReceipt,
    ExpectedMaximumCount,
    ExperimentBudget,
    P99AnchorStatus,
    ScenarioMilliseconds,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.serving import (
    BenchServingResult,
    BoundServingRequest,
    PinnedBenchServingTransport,
)
from lightcone_spec.orchestration.executor import (
    NATIVE_TERMINAL_EVIDENCE_HOOK,
    ArtifactBinding,
    IndustrialExecutionPlan,
    NativeEvidenceUnavailableError,
    build_industrial_execution_plan,
    execute_industrial_plan,
    industrial_execution_split_contract,
    launch_server_subprocess,
    native_evidence_preflight,
)
from lightcone_spec.orchestration.industrial import (
    render_assigned_industrial_cell_runtime_plan,
)
from lightcone_spec.orchestration.native_terminal import NativeTerminalProvider
from lightcone_spec.orchestration.runtime import ServerLaunch
from lightcone_spec.orchestration.session import (
    SHARED_SESSION_UNAVAILABLE_REASON,
    IndustrialServerSessionPlan,
    IndustrialSessionOpenReceipt,
    SharedSessionUnavailableError,
    execute_industrial_server_session,
)
from lightcone_spec.runtime.distributed import (
    RankTopologyReceipt,
    TopologyIdentity,
    TopologyReceiptSet,
)
from lightcone_spec.telemetry.records import OUTPUT_HASH_FORMAT, RequestRecord
from lightcone_spec.telemetry.writer import EvidenceWriter, load_completed_evidence


def _write_artifact(
    root: Path,
    name: str,
    body: bytes,
    *,
    experiment: str | None = None,
) -> ArtifactBinding:
    path = root / f"{experiment or 'root'}-{name}.json"
    path.write_bytes(body)
    return ArtifactBinding.from_path(
        name=name,
        path=path,
        experiment=experiment,
    )


def _topology(device_id: str) -> TopologyReceiptSet:
    return TopologyReceiptSet(
        (
            RankTopologyReceipt(
                topology=TopologyIdentity(
                    tensor_parallel_size=1,
                    data_parallel_size=1,
                    node_count=1,
                    node_id="executor-host",
                    node_rank=0,
                    global_rank=0,
                    local_rank=0,
                    tensor_parallel_rank=0,
                    data_parallel_rank=0,
                    device_id=device_id,
                    rendezvous_id="executor-rendezvous",
                    router_id="single-replica",
                    clock_id="executor-clock",
                ),
                process_id="executor-process",
                observed_world_size=1,
            ),
        )
    )


@dataclass(frozen=True)
class _Fixture:
    plan: IndustrialExecutionPlan
    dependency_artifacts: tuple[ArtifactBinding, ...]


def _fixture_budget(
    *,
    cell,
    load: ProductionLoadPlan,
    request_count: int,
) -> ExperimentBudget:
    arrival_ms = load.window.arrival_duration_us // 1000
    deadline_ms = load.window.request_deadline_us // 1000
    drain_ms = load.window.drain_duration_us // 1000
    zero = ScenarioMilliseconds(0, 0, 0)
    startup = ScenarioMilliseconds(1_000, 1_000, 1_000)
    scored_arrival = ScenarioMilliseconds(arrival_ms, arrival_ms, arrival_ms)
    request_deadline = ScenarioMilliseconds(deadline_ms, deadline_ms, deadline_ms)
    drain = ScenarioMilliseconds(drain_ms, drain_ms, drain_ms)
    evidence = ScenarioMilliseconds(1_000, 1_000, 1_000)
    wall_ms = 1_000 + arrival_ms + drain_ms + 1_000
    gpu_ms = ScenarioMilliseconds(wall_ms, wall_ms, wall_ms)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=BudgetJobKind.SHORT,
        startup_model_load=startup,
        compile_jit_graph_prewarm=zero,
        excluded_warmup=zero,
        excluded_warmup_requests=ExpectedMaximumCount(0, 0),
        scored_arrival=scored_arrival,
        request_deadline=request_deadline,
        drain=drain,
        reset_finalization=zero,
        evidence_flush_shutdown=evidence,
        output_tokens=ExpectedMaximumCount(request_count * 2, request_count * 2),
        minimum_completed_requests=1,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=zero,
        failure_injection=zero,
        retry=zero,
        retry_allowance=0,
        profiler=zero,
        download_compile_reservation=zero,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=gpu_ms,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=gpu_ms.scale(2),
    )


def _execution_fixture(
    tmp_path: Path,
    *,
    method: str = "target_only",
    request_count: int = 2,
    cancelled: bool = False,
    request_deadline_us: int = 100_000,
    budget_mutator: Callable[[ExperimentBudget], ExperimentBudget] | None = None,
) -> _Fixture:
    registry = build_industrial_registry(
        gpu_uuids=("GPU-executor-a", "GPU-executor-b"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
        base_port=28000,
    )
    cell = next(
        value
        for value in registry.cells_for("E3a")
        if value.identity.method == method
        and value.identity.context == 1024
        and value.identity.concurrency == 1
        and value.identity.regime == "long_input_short_output"
        and (method == "target_only" or value.identity.width == 8)
    )
    physical_gpu_uuid = "GPU-physical-executor"
    physical_port = 31_000
    sampling_profile = SamplingProfile()
    sampling_path = tmp_path / "root-sampling.json"
    sampling_profile.write(sampling_path)
    sampling = ArtifactBinding.from_path(
        name="sampling",
        path=sampling_path,
        semantic_sha256=sampling_profile.sha256,
    )
    model_lock = _write_artifact(tmp_path, "model-lock", b'{"models":[]}\n')
    config = RunConfig.model_validate(
        RunConfig(
            method=method,
            model=ModelPair(
                target=cell.identity.model,
                drafter="test/drafter",
                target_revision="1" * 40,
                drafter_revision="2" * 40,
                algorithm="DFLASH",
                max_context_length=cell.identity.context,
                draft_depth=7,
            ),
            runtime=RuntimeConfig(
                sampling_profile_sha256=sampling.content_sha256,
                speculation_enabled=method != "target_only",
                tensor_parallel_size=1,
                data_parallel_size=1,
                tp_rank=0,
                dp_rank=0,
                node_count=1,
                node_rank=0,
                device_identity=physical_gpu_uuid,
                rendezvous_identity="executor-rendezvous",
                router_identity="single-replica",
                clock_identity="executor-clock",
                process_group_backend="nccl",
                distributed_runtime_capability="single_rank",
                distributed_capability_receipt_sha256=None,
                speculative_num_draft_tokens=8,
                speculative_eagle_topk=None,
                use_rejection_sampling=True,
                max_running_requests=1,
                telemetry_detail="headline",
                prefill_decode_disaggregation=False,
                two_batch_overlap=False,
            ),
            adaptation=None,
            online_spec=None,
            tenant_id="executor-test",
        ).model_dump(mode="json")
    )
    sampling_parameters = FrozenSamplingParameters.from_mapping(
        sampling_profile.parameters(
            seed=cell.identity.seed,
            max_new_tokens=2,
        )
    )
    templates = tuple(
        RequestTemplate(
            input_token_ids=tuple(
                10_000 * (index + 1) + offset for offset in range(768)
            ),
            requested_output_tokens=2,
            sampling=sampling_parameters,
            cancellation_offset_us=5_000 if cancelled else None,
        )
        for index in range(request_count)
    )
    scored = closed_loop_corpus(
        templates,
        namespace="executor-score",
        split="tuning",
        concurrency=1,
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=7,
    )
    load = ProductionLoadPlan(
        warmup=None,
        scored=scored,
        window=ProductionWindow(
            warmup_duration_us=0,
            arrival_duration_us=request_count * 3_000,
            request_deadline_us=request_deadline_us,
            drain_duration_us=100_000,
        ),
    )
    budget = _fixture_budget(cell=cell, load=load, request_count=request_count)
    if budget_mutator is not None:
        budget = budget_mutator(budget)
    split_value = industrial_execution_split_contract(
        registry_sha256=registry.sha256,
        cell=cell,
        load_plan=load,
        sampling_profile_sha256=sampling.content_sha256,
        model_lock_sha256=model_lock.content_sha256,
    )
    split = _write_artifact(
        tmp_path,
        "split",
        json.dumps(split_value, sort_keys=True, separators=(",", ":")).encode(),
    )
    preflight_outputs: dict[str, str] = {}
    dependencies: list[ArtifactBinding] = []
    for name in registry.definition("preflight").locked_outputs:
        artifact = _write_artifact(
            tmp_path,
            name,
            json.dumps({"output": name}, sort_keys=True).encode(),
            experiment="preflight",
        )
        dependencies.append(artifact)
        preflight_outputs[name] = artifact.content_sha256
    receipt = registry.make_receipt(
        "preflight",
        preflight_outputs,
        runtime_sha256=content_sha256({"runtime": "preflight"}),
        split_sha256=split.content_sha256,
        completed_cells_sha256=content_sha256({"completed": "preflight"}),
    )
    inventory = GpuInventory(
        schema_version=1,
        devices=(
            GpuDevice(
                uuid=physical_gpu_uuid,
                host_id="executor-host",
                model="executor-gpu",
                memory_bytes=80 * 1024**3,
                compute_capability=(9, 0),
                pci_bus_id="0000:01:00.0",
                pci_root="executor-root",
                numa_node=0,
                interconnects=("pcie",),
                peer_access_class="executor-peer",
                clock_policy="locked",
                power_limit_watts=700.0,
                thermal_limit_celsius=83.0,
                availability=GpuAvailability.READY,
                reserved_processes=(),
                allowed_topology_groups=(),
            ),
            GpuDevice(
                uuid="GPU-physical-idle",
                host_id="executor-host",
                model="executor-gpu",
                memory_bytes=80 * 1024**3,
                compute_capability=(9, 0),
                pci_bus_id="0000:02:00.0",
                pci_root="executor-root",
                numa_node=0,
                interconnects=("pcie",),
                peer_access_class="executor-peer",
                clock_policy="locked",
                power_limit_watts=700.0,
                thermal_limit_celsius=83.0,
                availability=GpuAvailability.RESERVED,
                reserved_processes=(),
                allowed_topology_groups=(),
            ),
        ),
        topology_groups=(),
        source_receipt_sha256=content_sha256({"inventory": "executor"}),
    )
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256({"interference": "executor"})
    )
    dispatch_budgets = tuple(
        sorted(
            (
                budget
                if candidate.cell_id == cell.cell_id
                else _fixture_budget(
                    cell=candidate,
                    load=load,
                    request_count=request_count,
                )
                for candidate in registry.cells_for("E3a")
                if GpuPoolScheduler._dispatchable(candidate)
            ),
            key=lambda row: row.cell_id,
        )
    )
    dispatch_context = GpuDispatchExecutionContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=envelope,
        budgets=dispatch_budgets,
        receipts=(receipt,),
        port_start=physical_port,
        port_end=physical_port + 999,
        seed=20260811,
    )
    dispatch_plan = dispatch_context.issue_plan()
    matching_assignments = tuple(
        assignment
        for wave in dispatch_plan.waves
        for assignment in wave.assignments
        if assignment.work_item.item_id == cell.cell_id
    )
    if len(matching_assignments) != 1:
        raise ValueError("cell has no exact scheduler-issued physical assignment")
    assignment = matching_assignments[0]
    runtime = render_assigned_industrial_cell_runtime_plan(
        registry=registry,
        cell_id=cell.cell_id,
        assignment=assignment,
        dispatch_plan=dispatch_plan,
        dispatch_context=dispatch_context,
        budget=budget,
        inventory=inventory,
        dispatch_inventory_sha256=inventory.sha256,
        rank_configs=(config,),
        topology_receipts=_topology(physical_gpu_uuid),
        dependency_receipts=(receipt,),
    )
    config_path = tmp_path / "run-config.json"
    config_path.write_text(
        json.dumps(config.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    checkout = tmp_path / "verified-checkout"
    checkout.mkdir()
    target_root = tmp_path / "target-model"
    target_root.mkdir()
    drafter_root = tmp_path / "drafter-model"
    drafter_root.mkdir()
    server_argv = (
        sys.executable,
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--checkout",
        str(checkout),
        "--",
        "--model-path",
        str(target_root),
        "--max-running-requests",
        "1",
        "--mem-fraction-static",
        "0.8",
        "--tp-size",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(physical_port),
    )
    if method == "static":
        server_argv += (
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            str(drafter_root),
            "--speculative-num-draft-tokens",
            "8",
            "--speculative-draft-window-size",
            "8",
            "--speculative-accept-threshold-single",
            "1.0",
            "--speculative-accept-threshold-acc",
            "1.0",
            "--speculative-use-rejection-sampling",
            "--speculative-speed-study-metrics",
        )
    launch = ServerLaunch(
        method=method,
        base_url=f"http://127.0.0.1:{physical_port}",
        exclusive_device=True,
        run_config=str(config_path),
        adaptation_config=None,
        telemetry_path=None,
        argv=server_argv,
    )
    plan = build_industrial_execution_plan(
        runtime_plan=runtime,
        dispatch_plan=dispatch_plan,
        dispatch_context=dispatch_context,
        budget=budget,
        load_plan=load,
        server_launch=launch,
        dependency_receipts=(receipt,),
        dependency_artifacts=tuple(dependencies),
        split_artifact=split,
        sampling_artifact=sampling,
        model_lock_artifact=model_lock,
        startup_timeout_s=1.0,
        shutdown_timeout_s=1.0,
        abort_grace_s=1.0,
    )
    return _Fixture(plan=plan, dependency_artifacts=tuple(dependencies))


class _FakeHandle:
    def __init__(self) -> None:
        self.ready = 0
        self.terminated = 0

    async def wait_ready(self, timeout_s: float) -> None:
        assert timeout_s == 1.0
        self.ready += 1

    async def terminate(self, timeout_s: float) -> None:
        assert timeout_s == 1.0
        self.terminated += 1


class _FakeTransport:
    def __init__(self, *, delay_s: float = 0.003) -> None:
        self.requests: list[str] = []
        self.aborts: list[str] = []
        self.delay_s = delay_s
        self.opened = 0
        self.closed = 0

    async def open(
        self,
        *,
        request_timeout_s: float,
        abort_timeout_s: float,
    ) -> None:
        assert request_timeout_s > 0 and abort_timeout_s > 0
        self.opened += 1

    async def close(self) -> None:
        self.closed += 1

    def metrics(self) -> dict[str, int]:
        return {
            "connections_created": self.opened,
            "submitted_requests": len(self.requests),
            "reused_requests": max(0, len(self.requests) - self.opened),
        }

    async def submit(
        self,
        request: BoundServingRequest,
        *,
        base_url: str,
        served_model: str,
    ) -> BenchServingResult:
        assert base_url.startswith("http://127.0.0.1:")
        assert served_model == "Qwen/Qwen3-8B"
        self.requests.append(request.request_id)
        await asyncio.sleep(self.delay_s)
        result = BenchServingResult(
            request_id=request.request_id,
            success=True,
            generated_text=f"result:{request.request_id}",
            output_tokens=2,
            latency_us=900,
            stop_reason="length",
            error_code=None,
            chunks=(
                TokenChunkTiming(
                    request_id=request.request_id,
                    first_token_index=0,
                    token_count=2,
                    chunk_observed_at_us=900,
                    per_token_observed_at_us=(300, 900),
                ),
            ),
            generated_token_ids=(101, 102),
        )
        result.validate(request)
        return result

    async def abort(self, request_id: str, *, base_url: str) -> None:
        self.aborts.append(request_id)


def test_official_adapter_preserves_exact_request_and_marks_coalescing() -> None:
    observed: dict[str, object] = {}

    class RequestInput:
        def __init__(self, **kwargs) -> None:
            observed.update(kwargs)

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    session = Session()

    async def request_callable(*, request_func_input, pbar, client_session):
        assert pbar is None
        assert request_func_input is not None
        assert client_session is session
        return SimpleNamespace(
            success=True,
            generated_text="two tokens",
            output_len=2,
            latency=0.002,
            ttft=0.0005,
            generated_token_ids=(101, 102),
            # Upstream's distributed per-token ITLs must not cross the adapter.
            itl=[0.001],
        )

    transport = PinnedBenchServingTransport(
        request_type=RequestInput,
        request_callable=request_callable,
        set_global_args=lambda value: observed.setdefault("args", value),
        session_factory=lambda: session,
        module_identity="sglang.benchmark.serving.async_request_sglang_generate",
    )
    corpus = immediate_burst_corpus(
        (
            RequestTemplate(
                input_token_ids=(1, 2),
                requested_output_tokens=2,
                sampling=FrozenSamplingParameters.from_mapping({"temperature": 0.0}),
            ),
        ),
        namespace="adapter",
        split="confirmation",
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=1,
    )
    request = BoundServingRequest.create(corpus.requests[0], route_id="single-replica")

    async def exercise() -> BenchServingResult:
        await transport.open(request_timeout_s=1.0, abort_timeout_s=1.0)
        result = await transport.submit(
            request,
            base_url="http://127.0.0.1:30000",
            served_model="model",
        )
        await transport.close()
        return result

    result = asyncio.run(exercise())

    assert observed["prompt"] == [1, 2]
    assert observed["routing_key"] == "single-replica"
    assert observed["extra_request_body"] == {
        "rid": request.request_id,
        "sampling_params": {"max_new_tokens": 2, "temperature": 0.0},
    }
    assert result.generated_token_ids == (101, 102)
    assert result.ttft_us == 500
    assert len(result.chunks) == 1
    assert result.chunks[0].per_token_observed_at_us is None
    assert result.chunks[0].token_count == 2
    assert transport.metrics() == {
        "connections_created": 1,
        "submitted_requests": 1,
        "reused_requests": 0,
    }


def test_official_adapter_rejects_missing_ordered_token_ids() -> None:
    class RequestInput:
        def __init__(self, **kwargs) -> None:
            self.request_id = kwargs["extra_request_body"]["rid"]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def request_callable(*, request_func_input, pbar, client_session):
        assert request_func_input is not None and pbar is None
        assert client_session is not None
        return SimpleNamespace(
            success=True,
            generated_text="decoded text is not an exact token identity",
            output_len=1,
            latency=0.001,
            ttft=0.001,
        )

    transport = PinnedBenchServingTransport(
        request_type=RequestInput,
        request_callable=request_callable,
        set_global_args=lambda value: None,
        session_factory=Session,
        module_identity="sglang.benchmark.serving.async_request_sglang_generate",
    )
    corpus = immediate_burst_corpus(
        (
            RequestTemplate(
                input_token_ids=(1,),
                requested_output_tokens=1,
                sampling=FrozenSamplingParameters.from_mapping({"temperature": 0.0}),
            ),
        ),
        namespace="missing-token-ids",
        split="confirmation",
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=1,
    )
    request = BoundServingRequest.create(corpus.requests[0], route_id="replica")

    async def exercise() -> None:
        await transport.open(request_timeout_s=1.0, abort_timeout_s=1.0)
        try:
            await transport.submit(
                request,
                base_url="http://127.0.0.1:30000",
                served_model="model",
            )
        finally:
            await transport.close()

    with pytest.raises(RuntimeError, match="lacks exact ordered generated token IDs"):
        asyncio.run(exercise())


def test_official_adapter_reuses_one_pool_for_submit_and_abort() -> None:
    class RequestInput:
        def __init__(self, **kwargs) -> None:
            self.request_id = kwargs["extra_request_body"]["rid"]

    class Response:
        status = 200

        def __init__(self, value: object) -> None:
            self.value = value

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def json(self, *, content_type):
            assert content_type is None
            return self.value

    class Session:
        def __init__(self) -> None:
            self.enters = 0
            self.exits = 0
            self.abort_posts = 0
            self.admin_gets = 0
            self.admin_posts = 0

        async def __aenter__(self):
            self.enters += 1
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            self.exits += 1

        def get(self, **kwargs):
            assert kwargs["url"].endswith(
                "/v1/lightcone-spec/terminal-evidence/capability"
            )
            self.admin_gets += 1
            return Response({"kind": "capability"})

        def post(self, **kwargs):
            if kwargs["url"].endswith("/abort_request"):
                self.abort_posts += 1
                return Response({})
            assert kwargs["url"].endswith("/v1/lightcone-spec/terminal-evidence")
            self.admin_posts += 1
            return Response({"kind": "terminal"})

    session = Session()

    async def request_callable(*, request_func_input, pbar, client_session):
        assert pbar is None and client_session is session
        return SimpleNamespace(
            success=True,
            generated_text="token",
            output_len=1,
            latency=0.001,
            ttft=0.001,
            generated_token_ids=(101,),
        )

    transport = PinnedBenchServingTransport(
        request_type=RequestInput,
        request_callable=request_callable,
        set_global_args=lambda value: None,
        session_factory=lambda: session,
        headers_factory=lambda: {"x-test": "1"},
        module_identity="sglang.benchmark.serving.async_request_sglang_generate",
    )
    corpus = immediate_burst_corpus(
        tuple(
            RequestTemplate(
                input_token_ids=(index + 1,),
                requested_output_tokens=1,
                sampling=FrozenSamplingParameters.from_mapping({"temperature": 0.0}),
            )
            for index in range(2)
        ),
        namespace="pool",
        split="tuning",
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=1,
    )
    requests = tuple(
        BoundServingRequest.create(value, route_id="single-replica")
        for value in corpus.requests
    )

    async def exercise() -> None:
        await transport.open(request_timeout_s=1.0, abort_timeout_s=1.0)
        transport.bind_native_admin_base_url("http://127.0.0.1:30000")
        assert await transport.get_json(
            "/v1/lightcone-spec/terminal-evidence/capability"
        ) == {"kind": "capability"}
        assert await transport.post_json(
            "/v1/lightcone-spec/terminal-evidence",
            {"action": "begin"},
        ) == {"kind": "terminal"}
        for request in requests:
            await transport.submit(
                request,
                base_url="http://127.0.0.1:30000",
                served_model="model",
            )
        await transport.abort(requests[0].request_id, base_url="http://127.0.0.1:30000")
        await transport.close()

    asyncio.run(exercise())
    assert session.enters == 1 and session.exits == 1 and session.abort_posts == 1
    assert session.admin_gets == 1 and session.admin_posts == 1
    assert transport.metrics() == {
        "connections_created": 1,
        "submitted_requests": 2,
        "reused_requests": 1,
    }


def test_official_adapter_enforces_registered_abort_timeout() -> None:
    class SlowResponse:
        status = 200

        async def __aenter__(self):
            await asyncio.sleep(0.1)
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        def post(self, **kwargs):
            return SlowResponse()

    session = Session()
    transport = PinnedBenchServingTransport(
        request_type=SimpleNamespace,
        request_callable=lambda **kwargs: None,
        set_global_args=lambda value: None,
        session_factory=lambda: session,
        module_identity="sglang.benchmark.serving.async_request_sglang_generate",
    )

    async def exercise() -> None:
        await transport.open(request_timeout_s=1.0, abort_timeout_s=0.001)
        with pytest.raises(TimeoutError):
            await transport.abort("request", base_url="http://127.0.0.1:30000")
        await transport.close()

    asyncio.run(exercise())


def test_native_provider_cannot_use_a_different_pool_from_serving() -> None:
    def transport() -> PinnedBenchServingTransport:
        return PinnedBenchServingTransport(
            request_type=SimpleNamespace,
            request_callable=lambda **kwargs: None,
            set_global_args=lambda value: None,
            session_factory=lambda: SimpleNamespace(),
            module_identity=("sglang.benchmark.serving.async_request_sglang_generate"),
        )

    admin_pool = transport()
    serving_pool = transport()
    provider = NativeTerminalProvider(admin_pool)
    with pytest.raises(ValueError, match="different HTTP pools"):
        executor_module._bind_native_terminal_transport(
            provider=provider,
            transport=serving_pool,
            base_url="http://127.0.0.1:31000",
        )


def test_terminal_reconciliation_is_exact_and_rejects_short_unknown_finish(
    tmp_path: Path,
) -> None:
    request = _execution_fixture(tmp_path, request_count=1).plan.scored_requests[0]
    rejected = executor_module.RequestExecution(
        request=request,
        outcome=RequestOutcome(
            request_id=request.request_id,
            status="rejected",
            admitted_at_us=None,
            terminal_at_us=10,
            code="admission_deadline",
            offered_at_us=0,
        ),
        result=None,
    )
    expectation = executor_module._terminal_request_expectation(rejected)
    assert expectation.submitted_to_server is False
    assert expectation.terminal_status == "rejected"
    assert expectation.terminal_reason == "admission_deadline"

    short = BenchServingResult(
        request_id=request.request_id,
        success=True,
        generated_text="short",
        output_tokens=1,
        latency_us=100,
        stop_reason="server_stop",
        error_code=None,
        chunks=(
            TokenChunkTiming(
                request_id=request.request_id,
                first_token_index=0,
                token_count=1,
                chunk_observed_at_us=100,
                per_token_observed_at_us=(100,),
            ),
        ),
        generated_token_ids=(7,),
        ttft_us=100,
    )
    completed = executor_module.RequestExecution(
        request=request,
        outcome=RequestOutcome(
            request_id=request.request_id,
            status="completed",
            admitted_at_us=0,
            terminal_at_us=100,
            code="completed",
            offered_at_us=0,
        ),
        result=short,
    )
    with pytest.raises(RuntimeError, match="FINISH_LENGTH"):
        executor_module._terminal_request_expectation(completed)


def test_executor_orders_native_lifecycle_and_buckets_reset_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _execution_fixture(tmp_path, method="target_only", request_count=1).plan
    events: list[str] = []

    class SyntheticPerfCounter:
        def __init__(self) -> None:
            self.now_ns = 1_000_000_000

        def __call__(self) -> int:
            return self.now_ns

        def advance_ms(self, value: int) -> None:
            self.now_ns += value * 1_000_000

    perf_counter = SyntheticPerfCounter()
    monkeypatch.setattr(executor_module.time, "perf_counter_ns", perf_counter)

    class RequestInput:
        def __init__(self, **kwargs) -> None:
            self.request_id = kwargs["extra_request_body"]["rid"]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    session = Session()

    async def request_callable(*, request_func_input, pbar, client_session):
        assert pbar is None and client_session is session
        events.append("submit")
        await asyncio.sleep(0.004)
        return SimpleNamespace(
            success=True,
            generated_text="native-static",
            output_len=2,
            latency=0.001,
            ttft=0.0005,
            generated_token_ids=(101, 102),
        )

    transport = PinnedBenchServingTransport(
        request_type=RequestInput,
        request_callable=request_callable,
        set_global_args=lambda value: None,
        session_factory=lambda: session,
        module_identity="sglang.benchmark.serving.async_request_sglang_generate",
    )
    provider = NativeTerminalProvider(transport)
    state: dict[str, object] = {}

    async def capability(self, *, expected_method):
        assert self is provider and expected_method == "target_only"
        events.append("capability")
        return SimpleNamespace(trusted_attester_configured=True)

    async def begin(self, binding):
        assert self is provider and binding.method == "target_only"
        state["binding"] = binding
        events.append("begin")

    async def reset(self, *, warmup_requests=()):
        assert self is provider and tuple(warmup_requests) == ()
        events.append("reset")
        perf_counter.advance_ms(7)

    async def finalize(self, *, requests):
        assert self is provider and len(requests) == 1
        request = requests[0]
        assert request.submitted_to_server
        assert request.terminal_status == "completed"
        assert request.terminal_reason == "FINISH_LENGTH"
        assert request.output_token_ids == (101, 102)
        events.append("finalize")
        perf_counter.advance_ms(11)
        return executor_module.ValidatedNativeTerminalEvidence(
            binding=state["binding"],
            begin_receipt=SimpleNamespace(),
            reset_receipt=SimpleNamespace(),
            requests=tuple(requests),
            attestation=SimpleNamespace(trusted=False),
            terminal_sha256="a" * 64,
            raw_json="{}",
        )

    monkeypatch.setattr(NativeTerminalProvider, "capability", capability)
    monkeypatch.setattr(NativeTerminalProvider, "begin", begin)
    monkeypatch.setattr(NativeTerminalProvider, "reset", reset)
    monkeypatch.setattr(NativeTerminalProvider, "finalize", finalize)
    monkeypatch.setattr(
        executor_module.ValidatedNativeTerminalEvidence,
        "to_native_evidence_batch",
        lambda self: executor_module.NativeEvidenceBatch(),
    )

    async def launch(server: ServerLaunch) -> _FakeHandle:
        return _FakeHandle()

    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    execution = execute_industrial_plan(
        plan,
        output_root=output,
        run_nonce_sha256="c" * 64,
        launch_server=launch,
        transport=transport,
        native_evidence=provider,
    )
    result = asyncio.run(execution)
    observation = json.loads(Path(result.budget_observation).read_text())
    components = dict(observation["observed_component_ms"])
    assert components["excluded_warmup"] == 0
    assert components["scored_arrival"] == 0
    assert components["reset_finalization"] == 18
    assert observation["measured_gpu_ms"] == 18
    assert observation["fixed_instance_billed_gpu_ms"] == 36
    assert events == ["capability", "begin", "reset", "submit", "finalize"]


def test_target_only_execution_writes_terminal_receipt_and_resumes(
    tmp_path: Path,
) -> None:
    fixture = _execution_fixture(tmp_path)
    plan = fixture.plan
    handle = _FakeHandle()
    launch_count = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launch_count
        assert server == plan.server_launch
        launch_count += 1
        return handle

    transport = _FakeTransport()
    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    first = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=output,
            run_nonce_sha256="9" * 64,
            launch_server=launch,
            transport=transport,
        )
    )
    assert not first.resumed
    assert first.accounting is not None
    assert first.accounting.completed == 2
    assert first.accounting.offered == 2
    assert (
        first.rank_config_sha256
        == plan.runtime_plan.to_dict()["rank_config_sha256s"][0]
    )
    assert first.topology_sha256 == plan.topology_sha256
    assert plan.runtime_plan.physical_gpu_uuids == ("GPU-physical-executor",)
    assert plan.runtime_plan.physical_rank_groups == (("GPU-physical-executor",),)
    assert plan.runtime_plan.physical_ports == (31_000,)
    assert plan.server_launch.base_url == "http://127.0.0.1:31000"
    assert plan.to_dict()["topology_receipt_sha256"] == (
        plan.runtime_plan.topology_receipt_sha256
    )
    assert Path(first.terminal_receipt).is_file()
    assert handle.ready == 1 and handle.terminated == 1
    assert launch_count == 1

    completed = load_completed_evidence(output, run_id=first.run_id, rank=0)
    assert completed is not None
    run = pq.read_table(completed["run"]).to_pylist()[0]
    requests = pq.read_table(completed["request"]).to_pylist()
    performance = pq.read_table(completed["performance"]).to_pylist()[0]
    assert run["runtime_sha256"] == plan.sha256
    assert run["industrial_cell_id"] == plan.runtime_plan.cell_id
    assert run["rank_config_sha256"] == plan.rank_config_sha256
    assert run["topology_sha256"] == plan.topology_sha256
    assert run["experiment_budget_sha256"] == plan.budget.sha256
    terminal = json.loads(Path(first.terminal_receipt).read_text(encoding="utf-8"))
    assert terminal["experiment_budget_sha256"] == plan.budget.sha256
    assert first.experiment_budget_sha256 == plan.budget.sha256
    assert (
        first.terminal_receipt_sha256
        == hashlib.sha256(Path(first.terminal_receipt).read_bytes()).hexdigest()
    )
    observation_artifact = json.loads(
        Path(first.budget_observation).read_text(encoding="utf-8")
    )
    observation = BudgetObservationReceipt(
        schema_version=observation_artifact["schema_version"],
        budget=plan.budget,
        observed_component_ms=tuple(
            (name, value)
            for name, value in observation_artifact["observed_component_ms"]
        ),
        measured_gpu_ms=observation_artifact["measured_gpu_ms"],
        fixed_instance_billed_gpu_ms=observation_artifact[
            "fixed_instance_billed_gpu_ms"
        ],
        terminal_evidence_sha256=observation_artifact["terminal_evidence_sha256"],
    )
    assert observation_artifact["budget_observation_sha256"] == observation.sha256
    assert first.budget_observation_sha256 == observation.sha256
    assert (
        Path(first.budget_observation_sidecar).read_text(encoding="utf-8").strip()
        == observation.sha256
    )
    assert observation.terminal_evidence_sha256 == terminal["prepared_receipt_sha256"]
    assert terminal["budget_observation"]["budget_observation_sha256"] == (
        observation.sha256
    )
    assert (
        terminal["budget_observation"]["receipt_sha256"]
        == hashlib.sha256(Path(first.budget_observation).read_bytes()).hexdigest()
    )
    assert observation.measured_gpu_ms == (
        observation.observed_wall_ms * plan.budget.gpu_count
    )
    assert plan.runtime_plan.physical_fixed_instance_gpu_count == 2
    assert observation.measured_gpu_ms == observation.observed_wall_ms
    assert observation.fixed_instance_billed_gpu_ms == 2 * observation.observed_wall_ms
    assert (
        observation_artifact["fixed_instance_billing_semantics"]
        == "whole_inventory_wall_clock_v1"
    )
    assert observation_artifact["registered_wall_delta_ms"] == (
        observation.registered_wall_delta_ms
    )
    assert observation_artifact["registered_gpu_delta_ms"] == (
        observation.registered_gpu_delta_ms
    )
    assert observation_artifact["registered_billed_delta_ms"] == (
        observation.registered_billed_delta_ms
    )
    observed_components = dict(observation.observed_component_ms)
    assert observed_components["excluded_warmup"] == 0
    assert all(
        observed_components[name] == 0
        for name in executor_module._STRUCTURALLY_ABSENT_SERVING_COMPONENTS
    )
    assert {row["request_id"] for row in requests} == {
        request.request_id for request in plan.scored_requests
    }
    assert all(row["token_timing_coverage"] == 1.0 for row in requests)
    assert all(row["coalesced_intervals"] == 0 for row in requests)
    assert all(row["outcome_status"] == "completed" for row in requests)
    ordered_requests = sorted(requests, key=lambda row: row["prompt_id"])
    assert ordered_requests[1]["arrival_ns"] == ordered_requests[0]["completed_ns"]
    assert all(
        row["ttft_ms"]
        == pytest.approx((row["first_token_ns"] - row["arrival_ns"]) / 1_000_000)
        for row in requests
    )
    assert performance["output_tokens"] == 4
    assert performance["offered_requests"] == 2
    assert performance["admitted_requests"] == 2
    assert performance["completed_requests"] == 2
    assert performance["unfinished_requests"] == 0
    assert performance["optimizer_bytes"] == 0
    assert performance["trainable_parameters"] == 0
    assert performance["updates_launched"] == 0
    assert performance["updates_published"] == 0
    assert performance["communicator_failures"] == 0
    assert performance["itl_p99_ms"] is not None
    assert performance["evidence_dropped_rows"] == 0
    assert performance["evidence_backpressure_events"] == 0

    resumed = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=output,
            run_nonce_sha256="9" * 64,
            launch_server=launch,
            transport=transport,
        )
    )
    assert resumed.resumed
    assert resumed.run_id == first.run_id
    assert resumed.budget_observation_sha256 == first.budget_observation_sha256
    assert resumed.terminal_receipt_sha256 == first.terminal_receipt_sha256
    assert launch_count == 1


def test_logical_runtime_plan_cannot_cross_the_execution_boundary(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    logical = replace(
        plan,
        runtime_plan=replace(plan.runtime_plan, physical_assignment=None),
    )
    with pytest.raises(ValueError, match="physical scheduler assignment"):
        logical.validate()


@pytest.mark.parametrize(
    "field",
    (
        "inventory_sha256",
        "inventory_source_receipt_sha256",
        "dispatch_plan_sha256",
        "assignment_sha256",
        "work_item_sha256",
    ),
)
def test_launch_replays_scheduler_and_rejects_replaced_physical_authority(
    tmp_path: Path,
    field: str,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    physical = plan.runtime_plan.physical_assignment
    assert physical is not None
    forged_physical = replace(
        physical,
        **{field: content_sha256({"forged_launch_field": field})},
    )
    forged = replace(
        plan,
        runtime_plan=replace(
            plan.runtime_plan,
            physical_assignment=forged_physical,
        ),
    )
    launched = False

    async def launch(_server: ServerLaunch) -> _FakeHandle:
        nonlocal launched
        launched = True
        return _FakeHandle()

    with pytest.raises(ValueError, match="exact scheduler replay"):
        asyncio.run(
            execute_industrial_plan(
                forged,
                output_root=forged.runtime_plan.cell.resources.evidence_root,
                run_nonce_sha256="4" * 64,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    assert not launched


def test_execution_blocks_an_unobservable_registered_budget_component_before_launch(
    tmp_path: Path,
) -> None:
    def register_compile(budget: ExperimentBudget) -> ExperimentBudget:
        registered = ScenarioMilliseconds(1, 1, 1)
        reserved = ScenarioMilliseconds(
            budget.reserved_gpu_ms.optimistic + budget.gpu_count,
            budget.reserved_gpu_ms.registered + budget.gpu_count,
            budget.reserved_gpu_ms.quota_envelope + budget.gpu_count,
        )
        return replace(
            budget,
            compile_jit_graph_prewarm=registered,
            reserved_gpu_ms=reserved,
            fixed_instance_billed_gpu_ms=ScenarioMilliseconds(
                budget.fixed_instance_billed_gpu_ms.optimistic + 2,
                budget.fixed_instance_billed_gpu_ms.registered + 2,
                budget.fixed_instance_billed_gpu_ms.quota_envelope + 2,
            ),
        )

    plan = _execution_fixture(
        tmp_path,
        request_count=1,
        budget_mutator=register_compile,
    ).plan
    launched = False

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launched
        launched = True
        return _FakeHandle()

    with pytest.raises(ValueError, match="cannot observe registered budget component"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=plan.runtime_plan.cell.resources.evidence_root,
                run_nonce_sha256="7" * 64,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    assert not launched
    assert not Path(plan.runtime_plan.cell.resources.evidence_root).exists()


def test_soak_job_records_its_own_scoring_clock(tmp_path: Path) -> None:
    def register_soak(budget: ExperimentBudget) -> ExperimentBudget:
        return replace(
            budget,
            job_kind=BudgetJobKind.SOAK,
            scored_arrival=ScenarioMilliseconds(0, 0, 0),
            soak=budget.scored_arrival,
        )

    plan = _execution_fixture(
        tmp_path,
        request_count=1,
        budget_mutator=register_soak,
    ).plan

    async def launch(server: ServerLaunch) -> _FakeHandle:
        return _FakeHandle()

    result = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=plan.runtime_plan.cell.resources.evidence_root,
            run_nonce_sha256="a" * 64,
            launch_server=launch,
            transport=_FakeTransport(),
        )
    )
    observation = json.loads(Path(result.budget_observation).read_text())
    components = dict(observation["observed_component_ms"])
    assert components["scored_arrival"] == 0
    assert components["soak"] > 0
    assert components["failure_injection"] == 0
    assert components["profiler"] == 0


def test_retry_budget_is_an_envelope_but_first_attempt_observes_zero(
    tmp_path: Path,
) -> None:
    def register_retry(budget: ExperimentBudget) -> ExperimentBudget:
        retry = ScenarioMilliseconds(10, 10, 10)
        reserved = ScenarioMilliseconds(
            budget.reserved_gpu_ms.optimistic + 10 * budget.gpu_count,
            budget.reserved_gpu_ms.registered + 10 * budget.gpu_count,
            budget.reserved_gpu_ms.quota_envelope + 10 * budget.gpu_count,
        )
        return replace(
            budget,
            retry=retry,
            retry_allowance=1,
            reserved_gpu_ms=reserved,
            fixed_instance_billed_gpu_ms=ScenarioMilliseconds(
                budget.fixed_instance_billed_gpu_ms.optimistic + 20,
                budget.fixed_instance_billed_gpu_ms.registered + 20,
                budget.fixed_instance_billed_gpu_ms.quota_envelope + 20,
            ),
        )

    plan = _execution_fixture(
        tmp_path,
        request_count=1,
        budget_mutator=register_retry,
    ).plan

    async def launch(server: ServerLaunch) -> _FakeHandle:
        return _FakeHandle()

    result = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=plan.runtime_plan.cell.resources.evidence_root,
            run_nonce_sha256="b" * 64,
            launch_server=launch,
            transport=_FakeTransport(),
        )
    )
    observation = json.loads(Path(result.budget_observation).read_text())
    assert dict(observation["observed_component_ms"])["retry"] == 0
    assert observation["budget"]["retry_allowance"] == 1
    assert observation["budget"]["retry"]["registered"] == 10


def test_inactive_job_duration_must_be_zero_in_every_scenario(
    tmp_path: Path,
) -> None:
    def hide_quota_only_soak(budget: ExperimentBudget) -> ExperimentBudget:
        quota_only = ScenarioMilliseconds(0, 0, 1)
        reserved = ScenarioMilliseconds(
            budget.reserved_gpu_ms.optimistic,
            budget.reserved_gpu_ms.registered,
            budget.reserved_gpu_ms.quota_envelope + budget.gpu_count,
        )
        return replace(
            budget,
            soak=quota_only,
            reserved_gpu_ms=reserved,
            fixed_instance_billed_gpu_ms=ScenarioMilliseconds(
                budget.fixed_instance_billed_gpu_ms.optimistic,
                budget.fixed_instance_billed_gpu_ms.registered,
                budget.fixed_instance_billed_gpu_ms.quota_envelope + 2,
            ),
        )

    with pytest.raises(ValueError, match="scored duration components"):
        _execution_fixture(
            tmp_path,
            request_count=1,
            budget_mutator=hide_quota_only_soak,
        )


def test_failure_and_profiler_jobs_fail_closed_without_their_runtime_contract(
    tmp_path: Path,
) -> None:
    def register_failure(budget: ExperimentBudget) -> ExperimentBudget:
        return replace(
            budget,
            job_kind=BudgetJobKind.FAILURE,
            scored_arrival=ScenarioMilliseconds(0, 0, 0),
            failure_injection=budget.scored_arrival,
        )

    with pytest.raises(ValueError, match="fault actuator"):
        _execution_fixture(
            tmp_path / "failure",
            request_count=1,
            budget_mutator=register_failure,
        )

    def register_profiler(budget: ExperimentBudget) -> ExperimentBudget:
        return replace(
            budget,
            job_kind=BudgetJobKind.PROFILER,
            scored_arrival=ScenarioMilliseconds(0, 0, 0),
            profiler=budget.scored_arrival,
        )

    with pytest.raises(ValueError, match="PROFILE isolation"):
        _execution_fixture(
            tmp_path / "profiler",
            request_count=1,
            budget_mutator=register_profiler,
        )


def test_resume_requires_immutable_budget_observation(tmp_path: Path) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan

    async def launch(server: ServerLaunch) -> _FakeHandle:
        return _FakeHandle()

    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    first = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=output,
            run_nonce_sha256="6" * 64,
            launch_server=launch,
            transport=_FakeTransport(),
        )
    )
    Path(first.budget_observation_sidecar).write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="budget observation content binding"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256="6" * 64,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )


def test_resume_rejects_rehashed_underreported_gpu_observation(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    launches = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launches
        launches += 1
        return _FakeHandle()

    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    nonce = "c" * 64
    first = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=output,
            run_nonce_sha256=nonce,
            launch_server=launch,
            transport=_FakeTransport(),
        )
    )
    observation_path = Path(first.budget_observation)
    artifact = json.loads(observation_path.read_text(encoding="utf-8"))
    terminal = json.loads(Path(first.terminal_receipt).read_text(encoding="utf-8"))
    prepared_receipt_sha256 = terminal["prepared_receipt_sha256"]
    forged = BudgetObservationReceipt(
        schema_version=1,
        budget=plan.budget,
        observed_component_ms=tuple(
            (row[0], row[1]) for row in artifact["observed_component_ms"]
        ),
        measured_gpu_ms=0,
        fixed_instance_billed_gpu_ms=0,
        terminal_evidence_sha256=prepared_receipt_sha256,
    )
    artifact.update(
        {
            "budget_observation_sha256": forged.sha256,
            "measured_gpu_ms": forged.measured_gpu_ms,
            "fixed_instance_billed_gpu_ms": forged.fixed_instance_billed_gpu_ms,
            "observed_wall_ms": forged.observed_wall_ms,
            "registered_wall_delta_ms": forged.registered_wall_delta_ms,
            "registered_gpu_delta_ms": forged.registered_gpu_delta_ms,
            "registered_billed_delta_ms": forged.registered_billed_delta_ms,
        }
    )
    observation_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    Path(first.budget_observation_sidecar).write_text(
        forged.sha256 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="accounting is inconsistent"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )

    zeroed = BudgetObservationReceipt(
        schema_version=1,
        budget=plan.budget,
        observed_component_ms=tuple(
            (name, 0) for name, _ in forged.observed_component_ms
        ),
        measured_gpu_ms=0,
        fixed_instance_billed_gpu_ms=0,
        terminal_evidence_sha256=prepared_receipt_sha256,
    )
    artifact.update(
        {
            "budget_observation_sha256": zeroed.sha256,
            "observed_component_ms": [
                list(row) for row in zeroed.observed_component_ms
            ],
            "measured_gpu_ms": zeroed.measured_gpu_ms,
            "fixed_instance_billed_gpu_ms": zeroed.fixed_instance_billed_gpu_ms,
            "observed_wall_ms": zeroed.observed_wall_ms,
            "registered_wall_delta_ms": zeroed.registered_wall_delta_ms,
            "registered_gpu_delta_ms": zeroed.registered_gpu_delta_ms,
            "registered_billed_delta_ms": zeroed.registered_billed_delta_ms,
        }
    )
    observation_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    Path(first.budget_observation_sidecar).write_text(
        zeroed.sha256 + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not bind its observation"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    assert launches == 1


def test_resume_promotes_observation_bound_prepared_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    launches = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launches
        launches += 1
        return _FakeHandle()

    original_publish = EvidenceWriter.publish_close

    def fail_terminal_publish(
        self: EvidenceWriter,
        **_kwargs: object,
    ) -> dict[str, Path]:
        raise OSError("injected terminal publication failure")

    monkeypatch.setattr(EvidenceWriter, "publish_close", fail_terminal_publish)
    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    nonce = "d" * 64
    with pytest.raises(OSError, match="terminal publication failure"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    run_id = executor_module.industrial_run_id(plan, nonce)
    assert not (output / f"{run_id}.rank0.complete.json").exists()
    assert (output / f"{run_id}.rank0.prepared.json").is_file()
    observation_directory = output / f"{run_id}.rank0.budget-observation"
    assert observation_directory.is_dir()

    held_observation = output / f".{run_id}.held-budget-observation"
    observation_directory.rename(held_observation)
    with pytest.raises(RuntimeError, match="durable budget observation"):
        writer_module.publish_prepared_evidence_completion(
            output,
            run_id=run_id,
            rank=0,
            validate_post_binding=lambda: None,
        )
    held_observation.rename(observation_directory)

    monkeypatch.setattr(EvidenceWriter, "publish_close", original_publish)
    resumed = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=output,
            run_nonce_sha256=nonce,
            launch_server=launch,
            transport=_FakeTransport(),
        )
    )
    assert resumed.resumed
    assert launches == 1
    assert Path(resumed.terminal_receipt).is_file()


def test_post_terminal_publish_failure_still_has_observation_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    launches = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launches
        launches += 1
        return _FakeHandle()

    original_publish = EvidenceWriter.publish_close

    def publish_then_fail(
        self: EvidenceWriter,
        **kwargs: object,
    ) -> dict[str, Path]:
        original_publish(self, **kwargs)  # type: ignore[arg-type]
        raise OSError("injected post-terminal failure")

    monkeypatch.setattr(EvidenceWriter, "publish_close", publish_then_fail)
    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    nonce = "f" * 64
    with pytest.raises(OSError, match="post-terminal failure"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    run_id = executor_module.industrial_run_id(plan, nonce)
    assert (output / f"{run_id}.rank0.complete.json").is_file()
    observation_directory = output / f"{run_id}.rank0.budget-observation"
    assert observation_directory.is_dir()

    held_observation = output / f".{run_id}.held-budget-observation"
    observation_directory.rename(held_observation)
    with pytest.raises(RuntimeError, match="durable budget observation"):
        load_completed_evidence(output, run_id=run_id, rank=0)
    held_observation.rename(observation_directory)

    prepared_receipt = output / f"{run_id}.rank0.prepared.json"
    held_prepared = output / f".{run_id}.held-prepared.json"
    prepared_receipt.rename(held_prepared)
    with pytest.raises(RuntimeError, match="does not bind its preparation"):
        load_completed_evidence(output, run_id=run_id, rank=0)
    held_prepared.rename(prepared_receipt)

    monkeypatch.setattr(EvidenceWriter, "publish_close", original_publish)
    resumed = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=output,
            run_nonce_sha256=nonce,
            launch_server=launch,
            transport=_FakeTransport(),
        )
    )
    assert resumed.resumed
    assert launches == 1


def test_prepared_only_completion_fails_closed_before_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    launches = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launches
        launches += 1
        return _FakeHandle()

    def fail_observation_publish(**kwargs):
        raise OSError("injected observation publication failure")

    original_publish = executor_module._publish_budget_observation
    monkeypatch.setattr(
        executor_module,
        "_publish_budget_observation",
        fail_observation_publish,
    )
    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    nonce = "e" * 64
    with pytest.raises(OSError, match="observation publication failure"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    run_id = executor_module.industrial_run_id(plan, nonce)
    assert (output / f"{run_id}.rank0.prepared.json").is_file()
    assert not (output / f"{run_id}.rank0.budget-observation").exists()
    assert not (output / f"{run_id}.rank0.complete.json").exists()

    monkeypatch.setattr(
        executor_module,
        "_publish_budget_observation",
        original_publish,
    )
    with pytest.raises(RuntimeError, match="incomplete and non-resumable"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    assert launches == 1
    assert not (output / f"{run_id}.rank0.complete.json").exists()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("prepared_symlink", "lacks a budget observation"),
        ("observation_symlink", "lacks a budget observation"),
        ("prepared_bytes", "content binding is invalid"),
        ("observation_sidecar", "content binding is invalid"),
    ],
)
def test_prepared_recovery_rejects_symlink_and_tamper_before_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    launches = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launches
        launches += 1
        return _FakeHandle()

    original_publish = EvidenceWriter.publish_close

    def fail_terminal_publish(
        self: EvidenceWriter,
        **_kwargs: object,
    ) -> dict[str, Path]:
        raise OSError("injected terminal publication failure")

    monkeypatch.setattr(EvidenceWriter, "publish_close", fail_terminal_publish)
    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    nonce = {
        "prepared_symlink": "7",
        "prepared_bytes": "8",
        "observation_symlink": "b",
    }.get(tamper, "9") * 64
    with pytest.raises(OSError, match="terminal publication failure"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    run_id = executor_module.industrial_run_id(plan, nonce)
    prepared = output / f"{run_id}.rank0.prepared.json"
    observation = output / f"{run_id}.rank0.budget-observation"
    if tamper == "prepared_symlink":
        target = output / f"{run_id}.rank0.prepared.real.json"
        prepared.rename(target)
        prepared.symlink_to(target.name)
    elif tamper == "observation_symlink":
        target = output / f"{run_id}.rank0.budget-observation.real"
        observation.rename(target)
        observation.symlink_to(target.name, target_is_directory=True)
    elif tamper == "prepared_bytes":
        prepared.write_bytes(prepared.read_bytes() + b" ")
    else:
        (observation / "observation.json.sha256").write_text(
            "0" * 64 + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(EvidenceWriter, "publish_close", original_publish)
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    assert launches == 1
    assert not (output / f"{run_id}.rank0.complete.json").exists()


def test_prepared_recovery_validates_plan_before_canonical_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    launches = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launches
        launches += 1
        return _FakeHandle()

    original_publish = EvidenceWriter.publish_close

    def fail_terminal_publish(
        self: EvidenceWriter,
        **_kwargs: object,
    ) -> dict[str, Path]:
        raise OSError("injected terminal publication failure")

    monkeypatch.setattr(EvidenceWriter, "publish_close", fail_terminal_publish)
    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    nonce = "a" * 64
    with pytest.raises(OSError, match="terminal publication failure"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    monkeypatch.setattr(EvidenceWriter, "publish_close", original_publish)
    run_id = executor_module.industrial_run_id(plan, nonce)
    canonical = output / f"{run_id}.rank0.complete.json"

    foreign_plan = replace(plan, abort_grace_s=2.0)
    with pytest.raises(RuntimeError, match="mismatched run identity"):
        executor_module._recover_prepared_completion(
            root=output,
            run_id=run_id,
            run_nonce_sha256=nonce,
            plan=foreign_plan,
        )
    assert not canonical.exists()

    resumed = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=output,
            run_nonce_sha256=nonce,
            launch_server=launch,
            transport=_FakeTransport(),
        )
    )
    assert resumed.resumed
    assert launches == 1
    assert canonical.is_file()


def test_execution_plan_rejects_caller_authored_measured_gpu_time(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    with pytest.raises(ValueError, match="budget differs from the scheduler authority"):
        replace(
            plan,
            budget=replace(plan.budget, measured_gpu_ms=1),
        ).validate()


def test_execution_plan_rejects_budget_outside_dispatch_binding(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    increased = ScenarioMilliseconds(
        plan.budget.reserved_gpu_ms.optimistic + 1,
        plan.budget.reserved_gpu_ms.registered + 1,
        plan.budget.reserved_gpu_ms.quota_envelope + 1,
    )
    forged = replace(
        plan,
        budget=replace(
            plan.budget,
            reserved_gpu_ms=increased,
            fixed_instance_billed_gpu_ms=increased,
        ),
    )
    with pytest.raises(ValueError, match="budget differs from the scheduler authority"):
        forged.validate()


def test_executor_rejects_every_shared_session_mode_before_mutation(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan

    class CallerLifecycle:
        def claim_startup_interval_ns(self, *, execution_plan_sha256: str):
            raise AssertionError("caller timing callback must not execute")

        async def prepare_trace(self, *, execution_plan_sha256: str):
            raise AssertionError("caller reset callback must not execute")

        async def complete_trace(
            self,
            *,
            execution_plan_sha256: str,
            terminal_receipt_sha256: str,
            run_id: str,
        ):
            raise AssertionError("caller close callback must not execute")

    session_handle = _FakeHandle()
    session_modes: tuple[dict[str, object], ...] = (
        {"existing_handle": session_handle},
        {"transport_already_open": True},
        {"keep_session_open": True},
        {"session_lifecycle": CallerLifecycle()},
        {
            "session_lifecycle": session_module._SessionExecutionLifecycle(
                SimpleNamespace()
            )
        },
    )
    launch_count = 0

    async def forbidden_launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launch_count
        launch_count += 1
        return _FakeHandle()

    for session_mode in session_modes:
        transport = _FakeTransport()
        with pytest.raises(
            SharedSessionUnavailableError,
            match=SHARED_SESSION_UNAVAILABLE_REASON,
        ):
            asyncio.run(
                execute_industrial_plan(
                    plan,
                    output_root=plan.runtime_plan.cell.resources.evidence_root,
                    run_nonce_sha256="3" * 64,
                    launch_server=forbidden_launch,
                    transport=transport,
                    **session_mode,  # type: ignore[arg-type]
                )
            )
        assert transport.opened == 0 and transport.closed == 0
        assert not Path(plan.runtime_plan.cell.resources.evidence_root).exists()
    assert launch_count == 0
    assert session_handle.ready == 0 and session_handle.terminated == 0


def test_server_session_rejects_forged_static_before_any_side_effect(
    tmp_path: Path,
) -> None:
    target = _execution_fixture(tmp_path, request_count=1).plan
    session_plan = IndustrialServerSessionPlan.create(
        (target,),
        capability_receipt_sha256="a" * 64,
        compile_cache_receipt_sha256="b" * 64,
        dtype="bfloat16",
        precision="bf16",
        graph_buckets=(1,),
        hbm_reservation_bytes=0,
    )
    forged_static = SimpleNamespace(
        runtime_plan=SimpleNamespace(
            rank_configs=(SimpleNamespace(method="static"),),
        ),
        server_launch=replace(target.server_launch, method="static"),
        sha256=target.sha256,
        validate=lambda: None,
    )
    side_effects: list[str] = []

    async def launch(server: ServerLaunch) -> _FakeHandle:
        side_effects.append("launch")
        return _FakeHandle()

    class BoundaryRuntime:
        async def attest_open(self, *, session_plan, handle):
            side_effects.append("attest_open")
            raise AssertionError("forged session reached native attestation")

    transport = _FakeTransport()
    with pytest.raises(
        SharedSessionUnavailableError,
        match=SHARED_SESSION_UNAVAILABLE_REASON,
    ):
        asyncio.run(
            session_module.open_server_session(
                session_plan,
                (forged_static,),  # type: ignore[arg-type]
                output_roots=(target.runtime_plan.cell.resources.evidence_root,),
                run_nonce_sha256s=("7" * 64,),
                launch_server=launch,
                transport=transport,
                boundary_runtime=BoundaryRuntime(),  # type: ignore[arg-type]
            )
        )
    assert side_effects == []
    assert transport.opened == 0 and transport.closed == 0
    assert not Path(target.runtime_plan.cell.resources.evidence_root).exists()


def test_server_session_blocks_before_foreign_boundary_launch(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    session_plan = IndustrialServerSessionPlan.create(
        (plan,),
        capability_receipt_sha256="a" * 64,
        compile_cache_receipt_sha256="b" * 64,
        dtype="bfloat16",
        precision="bf16",
        graph_buckets=(1,),
        hbm_reservation_bytes=0,
    )
    handle = _FakeHandle()
    launch_count = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launch_count
        launch_count += 1
        return handle

    class ForeignCapabilityBoundary:
        async def attest_open(self, *, session_plan, handle):
            return IndustrialSessionOpenReceipt(
                session_plan_sha256=session_plan.sha256,
                process_identity="foreign-capability-process",
                process_started_ns=1,
                session_epoch=1,
                clean_state_sha256="c" * 64,
                native_capability_receipt_sha256="f" * 64,
            )

    transport = _FakeTransport()
    startup_authorities = dict(session_module._STARTUP_TIMING_AUTHORITIES)
    with pytest.raises(
        SharedSessionUnavailableError,
        match=SHARED_SESSION_UNAVAILABLE_REASON,
    ):
        asyncio.run(
            session_module.open_server_session(
                session_plan,
                (plan,),
                output_roots=(output,),
                run_nonce_sha256s=("8" * 64,),
                launch_server=launch,
                transport=transport,
                boundary_runtime=ForeignCapabilityBoundary(),  # type: ignore[arg-type]
            )
        )
    assert launch_count == 0 and handle.ready == 0 and handle.terminated == 0
    assert transport.opened == 0 and transport.closed == 0
    assert session_module._STARTUP_TIMING_AUTHORITIES == startup_authorities


def test_shared_session_rejects_completed_trace_before_any_side_effect(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    nonce = "6" * 64
    initial_handle = _FakeHandle()

    async def initial_launch(server: ServerLaunch) -> _FakeHandle:
        assert server == plan.server_launch
        return initial_handle

    completed = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=output,
            run_nonce_sha256=nonce,
            launch_server=initial_launch,
            transport=_FakeTransport(),
        )
    )
    assert Path(completed.terminal_receipt).is_file()
    assert initial_handle.ready == 1 and initial_handle.terminated == 1

    session_plan = IndustrialServerSessionPlan.create(
        (plan,),
        capability_receipt_sha256="a" * 64,
        compile_cache_receipt_sha256="b" * 64,
        dtype="bfloat16",
        precision="bf16",
        graph_buckets=(1,),
        hbm_reservation_bytes=0,
    )
    side_effects: list[str] = []

    async def forbidden_launch(server: ServerLaunch) -> _FakeHandle:
        side_effects.append("launch")
        return _FakeHandle()

    class ForbiddenTransport(_FakeTransport):
        async def open(
            self,
            *,
            request_timeout_s: float,
            abort_timeout_s: float,
        ) -> None:
            side_effects.append("transport_open")
            await super().open(
                request_timeout_s=request_timeout_s,
                abort_timeout_s=abort_timeout_s,
            )

    class ForbiddenBoundaryRuntime:
        async def attest_open(self, *, session_plan, handle):
            side_effects.append("attest_open")
            raise AssertionError("preexisting evidence reached native attestation")

    transport = ForbiddenTransport()
    with pytest.raises(
        SharedSessionUnavailableError,
        match=SHARED_SESSION_UNAVAILABLE_REASON,
    ):
        asyncio.run(
            execute_industrial_server_session(
                session_plan,
                (plan,),
                output_roots=(output,),
                run_nonce_sha256s=(nonce,),
                launch_server=forbidden_launch,
                transport=transport,
                boundary_runtime=ForbiddenBoundaryRuntime(),  # type: ignore[arg-type]
            )
        )
    assert side_effects == []
    assert transport.opened == 0 and transport.closed == 0


def test_block_scoped_session_is_blocked_without_any_claimable_evidence(
    tmp_path: Path,
) -> None:
    first = _execution_fixture(tmp_path, request_count=2).plan
    second = replace(first, abort_grace_s=2.0)
    session_plan = IndustrialServerSessionPlan.create(
        (first, second),
        capability_receipt_sha256="a" * 64,
        compile_cache_receipt_sha256="b" * 64,
        dtype="bfloat16",
        precision="bf16",
        graph_buckets=(1,),
        hbm_reservation_bytes=0,
    )
    root = Path(first.runtime_plan.cell.resources.evidence_root)
    side_effects: list[str] = []

    async def forbidden_launch(server: ServerLaunch) -> _FakeHandle:
        side_effects.append("launch")
        return _FakeHandle()

    class ForbiddenBoundaryRuntime:
        async def attest_open(self, *, session_plan, handle):
            side_effects.append("attest_open")
            raise AssertionError("blocked session reached native boundary")

    transport = _FakeTransport()
    with pytest.raises(
        SharedSessionUnavailableError,
        match=SHARED_SESSION_UNAVAILABLE_REASON,
    ):
        asyncio.run(
            execute_industrial_server_session(
                session_plan,
                (first, second),
                output_roots=(root, root),
                run_nonce_sha256s=("4" * 64, "5" * 64),
                launch_server=forbidden_launch,
                transport=transport,
                boundary_runtime=ForbiddenBoundaryRuntime(),  # type: ignore[arg-type]
            )
        )
    assert side_effects == []
    assert transport.opened == 0 and transport.closed == 0
    assert not root.exists()


def test_terminal_bench_failure_is_durable_unfinished_evidence(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=2).plan

    class FailedTransport(_FakeTransport):
        async def submit(
            self,
            request: BoundServingRequest,
            *,
            base_url: str,
            served_model: str,
        ) -> BenchServingResult:
            if not self.requests:
                return await super().submit(
                    request,
                    base_url=base_url,
                    served_model=served_model,
                )
            self.requests.append(request.request_id)
            return BenchServingResult(
                request_id=request.request_id,
                success=False,
                generated_text="",
                output_tokens=0,
                latency_us=100,
                stop_reason=None,
                error_code="server_overloaded",
                chunks=(),
                generated_token_ids=(),
            )

    async def launch(server: ServerLaunch) -> _FakeHandle:
        assert server == plan.server_launch
        return _FakeHandle()

    output = Path(plan.runtime_plan.cell.resources.evidence_root)
    with pytest.raises(RuntimeError, match="evidence is nonclaimable"):
        asyncio.run(
            execute_industrial_plan(
                plan,
                output_root=output,
                run_nonce_sha256=hashlib.sha256(b"terminal-failure").hexdigest(),
                launch_server=launch,
                transport=FailedTransport(),
            )
        )
    assert not tuple(output.glob("*.complete.json"))
    request_paths = sorted(output.glob("*.request.wal.*.parquet"))
    performance_path = next(output.glob("*.performance.wal.*.parquet"))
    run_path = next(output.glob("*.run.wal.*.parquet"))
    requests = pq.read_table([str(path) for path in request_paths]).to_pylist()
    performance = pq.read_table(performance_path).to_pylist()[0]
    run = pq.read_table(run_path).to_pylist()[0]
    assert [request["outcome_status"] for request in requests] == [
        "completed",
        "unfinished",
    ]
    assert requests[1]["error_code"] == "official_bench_error:server_overloaded"
    assert requests[1]["finished"] is False
    assert performance["offered_requests"] == 2
    assert performance["completed_requests"] == 1
    assert performance["unfinished_requests"] == 1
    assert run["status"] == "aborted"


def test_closed_loop_request_pool_exhaustion_is_nonclaimable(tmp_path: Path) -> None:
    plan = _execution_fixture(tmp_path, request_count=1).plan
    observed: list[executor_module.RequestExecution] = []

    async def observe(execution: executor_module.RequestExecution) -> None:
        observed.append(execution)

    with pytest.raises(RuntimeError, match="request pool exhausted"):
        asyncio.run(
            executor_module._execute_closed_loop_corpus(
                plan.scored_requests,
                concurrency=1,
                arrival_duration_us=100_000,
                request_deadline_us=100_000,
                scored_global_end_us=200_000,
                transport=_FakeTransport(),
                base_url=plan.server_launch.base_url,
                served_model=plan.runtime_plan.rank_configs[0].model.target,
                abort_grace_s=1.0,
                clock=executor_module.ExecutionClock(),
                on_terminal=observe,
            )
        )
    assert len(observed) == 1
    assert observed[0].outcome.status == "completed"


def test_async_evidence_saturation_persists_triggering_terminal_row(
    tmp_path: Path,
) -> None:
    run_id = "bounded-evidence-saturation"
    writer = EvidenceWriter(
        tmp_path,
        run_id=run_id,
        rank=0,
        checkpoint_interval_s=None,
    )

    def record(index: int) -> RequestRecord:
        token_ids = json.dumps([index], separators=(",", ":"))
        digest = hashlib.sha256(token_ids.encode()).hexdigest()
        return RequestRecord(
            run_id=run_id,
            request_id=f"request-{index}",
            prompt_id=f"prompt-{index}",
            method="target_only",
            repetition_block=0,
            concurrency=1,
            input_tokens=1,
            output_tokens=1,
            output_hash_format=OUTPUT_HASH_FORMAT,
            output_sha256=digest,
            ttft_ms=1.0,
            finished=True,
            stop_reason="length",
            output_token_ids=token_ids,
            output_token_ids_sha256=digest,
            outcome_status="completed",
            arrival_ns=index * 10,
            queue_enter_ns=index * 10,
            admitted_ns=index * 10,
            first_token_ns=index * 10 + 1,
            completed_ns=index * 10 + 2,
        )

    async def exercise() -> None:
        sink = executor_module._AsyncEvidenceSink(writer, max_queued_rows=1)
        started = threading.Event()
        release = threading.Event()
        original = sink._write_one
        calls = 0

        def blocking_write(
            item: executor_module.EvidenceItem, *, flush_after: bool
        ) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test evidence writer was not released")
            return original(item, flush_after=flush_after)

        sink._write_one = blocking_write
        await sink.write(record(0))
        assert await asyncio.to_thread(started.wait, 1.0)
        await sink.write(record(1))
        saturated = (
            asyncio.create_task(sink.write(record(2))),
            asyncio.create_task(sink.write(record(3))),
        )
        while sink.backpressure_events < 2:
            await asyncio.sleep(0)
        assert not any(task.done() for task in saturated)
        release.set()
        results = await asyncio.gather(*saturated, return_exceptions=True)
        assert all(
            isinstance(result, RuntimeError) and "queue saturated" in str(result)
            for result in results
        )
        await sink.close()

    asyncio.run(exercise())
    writer.register_external_backpressure_events(2)
    writer.abort(reason="expected saturation")
    rows = pq.read_table(
        [str(path) for path in sorted(tmp_path.glob("*.request.wal.*.parquet"))]
    ).to_pylist()
    assert {row["request_id"] for row in rows} == {
        "request-0",
        "request-1",
        "request-2",
        "request-3",
    }
    checkpoint = json.loads(next(tmp_path.glob("*.aborted.json")).read_text())
    assert checkpoint["counters"]["backpressure_events"] == 2


def test_async_evidence_sink_batches_without_empty_queue_fsync(
    tmp_path: Path,
) -> None:
    run_id = "batched-evidence"
    writer = EvidenceWriter(
        tmp_path,
        run_id=run_id,
        rank=0,
        row_group_rows=100,
        checkpoint_interval_s=None,
    )

    def record(index: int) -> RequestRecord:
        token_ids = json.dumps([index], separators=(",", ":"))
        digest = hashlib.sha256(token_ids.encode()).hexdigest()
        return RequestRecord(
            run_id=run_id,
            request_id=f"request-{index}",
            prompt_id=f"prompt-{index}",
            method="target_only",
            repetition_block=0,
            concurrency=1,
            input_tokens=1,
            output_tokens=1,
            output_hash_format=OUTPUT_HASH_FORMAT,
            output_sha256=digest,
            ttft_ms=1.0,
            finished=True,
            stop_reason="length",
            output_token_ids=token_ids,
            output_token_ids_sha256=digest,
            outcome_status="completed",
        )

    observed_batches: list[int] = []

    async def exercise() -> None:
        sink = executor_module._AsyncEvidenceSink(
            writer,
            max_queued_rows=32,
            max_batch_rows=8,
        )
        original = sink._write_batch

        def observe(rows: tuple[executor_module.EvidenceItem, ...]) -> bool:
            observed_batches.append(len(rows))
            return original(rows)

        sink._write_batch = observe
        await asyncio.gather(*(sink.write(record(index)) for index in range(20)))
        await sink.flush()
        await sink.close()

    asyncio.run(exercise())
    assert sum(observed_batches) == 20
    assert any(size > 1 for size in observed_batches)
    assert writer.counters["flushes"] == 1
    assert writer.counters["fsync_time_ns"] > 0
    writer.abort("batch test")


def test_static_execution_is_blocked_without_a_trusted_terminal_provider(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="no exact scheduler-issued physical assignment",
    ):
        _execution_fixture(tmp_path, method="static", request_count=1)


def test_adapted_native_evidence_preflight_remains_fail_closed() -> None:
    plan = SimpleNamespace(
        runtime_plan=SimpleNamespace(
            rank_configs=(SimpleNamespace(method="l0"),),
        ),
        patched_sglang_tree=executor_module.PINNED_SGLANG_TREE,
    )
    preflight = native_evidence_preflight(plan, None)
    assert preflight.status == "BLOCKED"
    assert (
        preflight.reason_code
        == executor_module.TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON
    )
    assert preflight.missing_hook is None
    with pytest.raises(NativeEvidenceUnavailableError):
        raise NativeEvidenceUnavailableError(preflight)

    forged_provider = SimpleNamespace(
        native_evidence_hook=NATIVE_TERMINAL_EVIDENCE_HOOK,
        patched_sglang_tree=executor_module.PINNED_SGLANG_TREE,
        supported_methods=frozenset({"l0"}),
    )
    assert native_evidence_preflight(plan, forged_provider).status == "BLOCKED"


def test_execution_plan_rejects_trace_semantics_from_another_registry_cell(
    tmp_path: Path,
) -> None:
    plan = _execution_fixture(tmp_path, request_count=2).plan
    templates = tuple(
        RequestTemplate(
            input_token_ids=request.input_token_ids,
            requested_output_tokens=request.requested_output_tokens,
            sampling=request.sampling,
            cancellation_offset_us=request.cancellation_offset_us,
        )
        for request in plan.load_plan.scored.requests
    )
    poisson = controlled_poisson_corpus(
        templates,
        namespace="wrong-arrival",
        split="tuning",
        rate_per_second=2.0,
        arrival_seed=17,
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=7,
    )
    wrong_arrival = replace(
        plan,
        load_plan=replace(
            plan.load_plan,
            scored=poisson,
            window=replace(
                plan.load_plan.window,
                arrival_duration_us=max(
                    request.arrival_us for request in poisson.requests
                )
                + 1,
            ),
        ),
    )
    with pytest.raises(
        ValueError, match="ExperimentBudget differs|source kind differs"
    ):
        wrong_arrival.validate()

    wrong_split = closed_loop_corpus(
        templates,
        namespace="wrong-split",
        split="confirmation",
        concurrency=1,
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=7,
    )
    with pytest.raises(ValueError, match="split differs"):
        replace(
            plan,
            load_plan=replace(plan.load_plan, scored=wrong_split),
        ).validate()


def test_resume_rejects_a_receipt_from_another_content_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _execution_fixture(tmp_path, request_count=1)
    handle = _FakeHandle()
    launched = 0

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launched
        launched += 1
        return handle

    output = Path(fixture.plan.runtime_plan.cell.resources.evidence_root)
    nonce = "8" * 64
    first = asyncio.run(
        execute_industrial_plan(
            fixture.plan,
            output_root=output,
            run_nonce_sha256=nonce,
            launch_server=launch,
            transport=_FakeTransport(),
        )
    )
    other_split = _write_artifact(
        tmp_path,
        "other-split",
        b'{"split":"other"}\n',
    )
    other_plan = replace(fixture.plan, split_artifact=other_split)
    monkeypatch.setattr(
        executor_module,
        "industrial_run_id",
        lambda plan, run_nonce_sha256: first.run_id,
    )
    with pytest.raises(ValueError, match="split artifact differs"):
        asyncio.run(
            execute_industrial_plan(
                other_plan,
                output_root=output,
                run_nonce_sha256=nonce,
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    assert launched == 1


def test_artifact_mutation_fails_before_server_launch(tmp_path: Path) -> None:
    fixture = _execution_fixture(tmp_path, request_count=1)
    changed = fixture.dependency_artifacts[0]
    Path(changed.path).write_bytes(b"mutated")
    launched = False

    async def launch(server: ServerLaunch) -> _FakeHandle:
        nonlocal launched
        launched = True
        return _FakeHandle()

    with pytest.raises(RuntimeError, match="bound artifact changed"):
        asyncio.run(
            execute_industrial_plan(
                fixture.plan,
                output_root=fixture.plan.runtime_plan.cell.resources.evidence_root,
                run_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
                launch_server=launch,
                transport=_FakeTransport(),
            )
        )
    assert not launched


def test_server_argv_is_part_of_the_validated_immutable_plan(tmp_path: Path) -> None:
    fixture = _execution_fixture(tmp_path, request_count=1)
    launch = fixture.plan.server_launch
    tampered = replace(
        fixture.plan,
        server_launch=replace(
            launch,
            argv=tuple(
                "2" if value == "1" and index == 13 else value
                for index, value in enumerate(launch.argv)
            ),
        ),
    )
    with pytest.raises(ValueError, match="base argv differs"):
        tampered.validate()


def test_opt_in_subprocess_launcher_binds_the_exact_gpu_uuid_without_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _execution_fixture(tmp_path, request_count=1)
    observed: dict[str, object] = {}

    class Process:
        pid = 12345
        returncode = None

    async def fake_create(*argv: str, **kwargs):
        observed["argv"] = argv
        observed["environment"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    handle = asyncio.run(launch_server_subprocess(fixture.plan.server_launch))
    assert handle is not None
    assert observed["argv"] == fixture.plan.server_launch.argv
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-physical-executor"


@pytest.mark.parametrize(
    (
        "cancelled",
        "deadline_us",
        "expected_status",
        "counter",
        "accounting_field",
    ),
    (
        (True, 100_000, "scheduled_cancellation", "cancellations", "cancelled"),
        (False, 5_000, "request_deadline", "timeouts", "timed_out"),
    ),
)
def test_cancellation_and_timeout_keep_partial_output_without_timing_imputation(
    tmp_path: Path,
    cancelled: bool,
    deadline_us: int,
    expected_status: str,
    counter: str,
    accounting_field: str,
) -> None:
    fixture = _execution_fixture(
        tmp_path,
        request_count=1,
        cancelled=cancelled,
        request_deadline_us=deadline_us,
    )
    handle = _FakeHandle()

    async def launch(server: ServerLaunch) -> _FakeHandle:
        return handle

    transport = _FakeTransport(delay_s=0.02)
    output = Path(fixture.plan.runtime_plan.cell.resources.evidence_root)
    result = asyncio.run(
        execute_industrial_plan(
            fixture.plan,
            output_root=output,
            run_nonce_sha256=hashlib.sha256(counter.encode()).hexdigest(),
            launch_server=launch,
            transport=transport,
        )
    )
    assert result.accounting is not None
    assert getattr(result.accounting, accounting_field) == 1
    assert transport.aborts == [fixture.plan.scored_requests[0].request_id]
    completed = load_completed_evidence(output, run_id=result.run_id, rank=0)
    assert completed is not None
    request = pq.read_table(completed["request"]).to_pylist()[0]
    performance = pq.read_table(completed["performance"]).to_pylist()[0]
    assert request["finished"] is False
    assert request["outcome_status"] == ("cancelled" if cancelled else "timed_out")
    assert request["stop_reason"] == expected_status
    assert request["output_tokens"] == 2
    assert request["output_sha256"] != hashlib.sha256(b"").hexdigest()
    assert performance[counter] == 1
    assert performance["offered_requests"] == 1
    assert performance["admitted_requests"] == 1
    assert performance["completed_requests"] == 0
    assert performance["unfinished_requests"] == 0
    assert performance["output_tokens"] == 0
    assert performance["itl_p99_ms"] is None
