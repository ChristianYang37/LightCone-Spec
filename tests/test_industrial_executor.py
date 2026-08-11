from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

import lightcone_spec.orchestration.executor as executor_module
from lightcone_spec.config.schema import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ProductionLoadPlan,
    ProductionWindow,
    RequestTemplate,
    TokenChunkTiming,
    closed_loop_corpus,
    controlled_poisson_corpus,
    immediate_burst_corpus,
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
    MISSING_NATIVE_EVIDENCE_REASON,
    NATIVE_TERMINAL_EVIDENCE_FIELDS,
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
    render_industrial_cell_runtime_plan,
)
from lightcone_spec.orchestration.runtime import ServerLaunch
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


def _execution_fixture(
    tmp_path: Path,
    *,
    method: str = "target_only",
    request_count: int = 2,
    cancelled: bool = False,
    request_deadline_us: int = 100_000,
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
                device_identity=cell.identity.gpu_uuids[0],
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
            arrival_duration_us=max(1, request_count * 2_500),
            request_deadline_us=request_deadline_us,
            drain_duration_us=100_000,
        ),
    )
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
    runtime = render_industrial_cell_runtime_plan(
        registry=registry,
        cell_id=cell.cell_id,
        rank_configs=(config,),
        topology_receipts=_topology(cell.identity.gpu_uuids[0]),
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
        str(cell.resources.ports[0]),
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
        )
    launch = ServerLaunch(
        method=method,
        base_url=f"http://127.0.0.1:{cell.resources.ports[0]}",
        exclusive_device=True,
        run_config=str(config_path),
        adaptation_config=None,
        telemetry_path=None,
        argv=server_argv,
    )
    plan = build_industrial_execution_plan(
        runtime_plan=runtime,
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

    async def request_callable(*, request_func_input, pbar):
        assert pbar is None
        assert request_func_input is not None
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
    result = asyncio.run(
        transport.submit(
            request,
            base_url="http://127.0.0.1:30000",
            served_model="model",
        )
    )

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
    assert plan.to_dict()["topology_receipt_sha256"] == (
        plan.runtime_plan.topology_receipt_sha256
    )
    assert Path(first.terminal_receipt).is_file()
    assert handle.ready == 1 and handle.terminated == 1
    assert launch_count == 1
    assert transport.aborts == []

    completed = load_completed_evidence(output, run_id=first.run_id, rank=0)
    assert completed is not None
    run = pq.read_table(completed["run"]).to_pylist()[0]
    requests = pq.read_table(completed["request"]).to_pylist()
    performance = pq.read_table(completed["performance"]).to_pylist()[0]
    assert run["runtime_sha256"] == plan.sha256
    assert run["industrial_cell_id"] == plan.runtime_plan.cell_id
    assert run["rank_config_sha256"] == plan.rank_config_sha256
    assert run["topology_sha256"] == plan.topology_sha256
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
    assert launch_count == 1


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


def test_static_execution_is_blocked_without_a_trusted_terminal_provider(
    tmp_path: Path,
) -> None:
    target_plan = _execution_fixture(tmp_path, request_count=1).plan
    plan = SimpleNamespace(
        runtime_plan=SimpleNamespace(
            rank_configs=(SimpleNamespace(method="static"),),
        ),
        patched_sglang_tree=target_plan.patched_sglang_tree,
    )
    preflight = native_evidence_preflight(plan, None)
    assert preflight.status == "BLOCKED"
    assert preflight.reason_code == MISSING_NATIVE_EVIDENCE_REASON
    assert preflight.missing_hook == NATIVE_TERMINAL_EVIDENCE_HOOK
    assert preflight.required_fields == NATIVE_TERMINAL_EVIDENCE_FIELDS
    forged_provider = SimpleNamespace(
        native_evidence_hook=NATIVE_TERMINAL_EVIDENCE_HOOK,
        patched_sglang_tree=target_plan.patched_sglang_tree,
        supported_methods=frozenset({"static"}),
    )
    assert native_evidence_preflight(plan, forged_provider).status == "BLOCKED"


def test_adapted_native_evidence_preflight_remains_fail_closed() -> None:
    plan = SimpleNamespace(
        runtime_plan=SimpleNamespace(
            rank_configs=(SimpleNamespace(method="l0"),),
        ),
        patched_sglang_tree=executor_module.PINNED_SGLANG_TREE,
    )
    preflight = native_evidence_preflight(plan, None)
    assert preflight.status == "BLOCKED"
    assert preflight.reason_code == MISSING_NATIVE_EVIDENCE_REASON
    assert preflight.missing_hook == NATIVE_TERMINAL_EVIDENCE_HOOK
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
    with pytest.raises(ValueError, match="source kind differs"):
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
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-executor-a"


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
