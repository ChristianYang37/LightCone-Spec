from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from lightcone_spec.orchestration import gang_execution
from lightcone_spec.orchestration.gang_execution import (
    DiagnosticGangCompletion,
    FreshProcessServingGangExecutor,
    ServingGangAttemptFailed,
    ServingGangAttemptReceipt,
    ServingGangLaunch,
    bind_diagnostic_gang_run_record,
    build_diagnostic_serving_gang_launch,
    execute_formal_serving_gang,
    inject_diagnostic_replica_route,
)
from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalRunBinding,
    canonical_sha256,
)
from lightcone_spec.orchestration.native_terminal_gang import (
    NativeTerminalGangAuthorityBlocked,
    NativeTerminalGangBinding,
    NativeTerminalRankBinding,
    build_replica_route_plan,
)
from lightcone_spec.runtime.distributed import (
    CohortRouteIdentity,
    RankTopologyReceipt,
    TopologyIdentity,
    TopologyReceiptSet,
)
from lightcone_spec.telemetry.records import RunRecord

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _topology(*, tp: int, dp: int) -> TopologyReceiptSet:
    world = tp * dp
    return TopologyReceiptSet(
        receipts=tuple(
            RankTopologyReceipt(
                topology=TopologyIdentity(
                    tensor_parallel_size=tp,
                    data_parallel_size=dp,
                    node_count=1,
                    node_id="host-0",
                    node_rank=0,
                    global_rank=rank,
                    local_rank=rank,
                    tensor_parallel_rank=rank % tp,
                    data_parallel_rank=rank // tp,
                    device_id=f"GPU-{rank}",
                    rendezvous_id="rdzv-0",
                    router_id="router-0",
                    clock_id="clock-0",
                ),
                process_id=f"scheduler-{rank}",
                observed_world_size=world,
            )
            for rank in range(world)
        )
    )


def _rank_binding(
    topology: TopologyReceiptSet,
    *,
    rank: int,
    request_ids: tuple[str, ...],
) -> NativeTerminalRankBinding:
    identity = topology.receipt_for_rank(rank).topology
    run = NativeTerminalRunBinding(
        run_id="run-0",
        run_nonce_sha256=SHA_A,
        execution_plan_sha256=SHA_B,
        rank_config_sha256=canonical_sha256({"rank": rank}),
        attempt_id="attempt-0",
        session_id="session-0",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=SHA_C,
        method="static",
        warmup_request_ids=(),
        scored_request_ids=request_ids,
    )
    return NativeTerminalRankBinding(
        run=run,
        topology_sha256=topology.topology_sha256,
        topology_receipt_sha256=topology.receipt_for_rank(rank).sha256,
        global_rank=rank,
        tensor_parallel_rank=identity.tensor_parallel_rank,
        data_parallel_rank=identity.data_parallel_rank,
        tensor_parallel_size=topology.tensor_parallel_size,
        data_parallel_size=topology.data_parallel_size,
        world_size=topology.world_size,
        node_count=1,
        node_rank=0,
    )


def _gang_binding(*, mode: str) -> NativeTerminalGangBinding:
    if mode == "tp2":
        topology = _topology(tp=2, dp=1)
        return NativeTerminalGangBinding(
            ranks=tuple(
                _rank_binding(topology, rank=rank, request_ids=("request-0",))
                for rank in range(2)
            )
        )
    topology = _topology(tp=1, dp=2)
    identities = tuple(
        CohortRouteIdentity(
            tenant_id=f"tenant-{index}",
            cohort_sha256=character * 64,
            router_id="router-0",
            topology_sha256=topology.topology_sha256,
        )
        for index, character in enumerate(("a", "b", "c", "d"))
    )
    route_plan = build_replica_route_plan(
        topology=topology,
        request_cohorts=tuple(
            (f"request-{index}", identity) for index, identity in enumerate(identities)
        ),
    )
    return NativeTerminalGangBinding(
        ranks=tuple(
            _rank_binding(
                topology,
                rank=rank,
                request_ids=tuple(
                    route.request_id
                    for route in route_plan.routes
                    if route.data_parallel_rank == rank
                ),
            )
            for rank in range(2)
        ),
        route_plan=route_plan,
    )


def _assignment(*, tp: int, dp: int) -> IndustrialPhysicalAssignment:
    gpu_uuids = ("GPU-0", "GPU-1")
    rank_groups = (
        (gpu_uuids,) if (tp, dp) == (2, 1) else ((gpu_uuids[0],), (gpu_uuids[1],))
    )
    if (tp, dp) == (2, 1):
        rank_groups = (gpu_uuids,)
    return IndustrialPhysicalAssignment(
        inventory_sha256=SHA_A,
        inventory_source_receipt_sha256=SHA_B,
        dispatch_plan_sha256=SHA_C,
        experiment_budget_sha256=SHA_D,
        budget_plan_sha256=SHA_E,
        capacity_authority_sha256=canonical_sha256("capacity"),
        budget_materialization_authority_sha256=canonical_sha256("materialize"),
        assignment_sha256=canonical_sha256("assignment"),
        work_item_sha256=canonical_sha256("work-item"),
        gpu_uuids=gpu_uuids,
        rank_groups=rank_groups,
        ports=(30_000, 30_001, 30_002),
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        fixed_instance_gpu_count=2,
        host_id="host-0",
        topology_group_ids=((("nvlink-0",),) if tp == 2 else ((), ())),
    )


def _argv(*, tp: int, dp: int) -> tuple[str, ...]:
    values = [
        "/venv/bin/python",
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--checkout",
        "/pinned/sglang",
        "--",
        "--model-path",
        "/models/target",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--nccl-port",
        "30001",
        "--tp-size",
        str(tp),
        "--dp-size",
        str(dp),
    ]
    if dp == 2:
        values.extend(("--load-balance-method", "round_robin"))
    return tuple(values)


def _launch(*, mode: str) -> tuple[ServingGangLaunch, NativeTerminalGangBinding]:
    binding = _gang_binding(mode=mode)
    tp, dp = (2, 1) if mode == "tp2" else (1, 2)
    launch = build_diagnostic_serving_gang_launch(
        gang_binding=binding,
        assignment=_assignment(tp=tp, dp=dp),
        supervisor_argv=_argv(tp=tp, dp=dp),
    )
    return launch, binding


def test_tp2_launch_is_one_supervisor_with_exact_ports_ranks_and_codec() -> None:
    launch, _ = _launch(mode="tp2")
    assert launch.mode == "tp2_dp1"
    assert launch.tensor_parallel_size == 2
    assert launch.data_parallel_size == 1
    assert launch.ports.ports == (30_000, 30_001, 30_002)
    assert launch.supervisor_environment[-1] == (
        "CUDA_VISIBLE_DEVICES",
        "GPU-0,GPU-1",
    )
    assert launch.supervisor_argv.count("--") == 1
    assert ServingGangLaunch.from_dict(launch.to_dict()) == launch

    unknown = launch.to_dict()
    unknown["second_server"] = True
    with pytest.raises(ValueError, match="fields differ"):
        ServingGangLaunch.from_dict(unknown)
    tampered = launch.to_dict()
    tampered["rank_config_set_sha256"] = SHA_A
    with pytest.raises(ValueError, match="rank-config set differs"):
        ServingGangLaunch.from_dict(tampered)


def test_launch_rejects_multinode_duplicate_flags_and_live_reserved_port() -> None:
    binding = _gang_binding(mode="tp2")
    assignment = _assignment(tp=2, dp=1)
    with pytest.raises(ValueError, match="multi-node"):
        build_diagnostic_serving_gang_launch(
            gang_binding=binding,
            assignment=assignment,
            supervisor_argv=(*_argv(tp=2, dp=1), "--nnodes", "2"),
        )
    with pytest.raises(ValueError, match="exactly one --tp-size"):
        build_diagnostic_serving_gang_launch(
            gang_binding=binding,
            assignment=assignment,
            supervisor_argv=(*_argv(tp=2, dp=1), "--tp-size", "2"),
        )
    with pytest.raises(ValueError, match="reserved control"):
        build_diagnostic_serving_gang_launch(
            gang_binding=binding,
            assignment=assignment,
            supervisor_argv=(*_argv(tp=2, dp=1), "--unused-port", "30002"),
        )
    misplaced = list(_argv(tp=2, dp=1))
    separator = misplaced.index("--")
    host = misplaced.index("--host")
    host_pair = misplaced[host : host + 2]
    del misplaced[host : host + 2]
    misplaced[separator:separator] = host_pair
    with pytest.raises(ValueError, match="differs from the gang topology"):
        build_diagnostic_serving_gang_launch(
            gang_binding=binding,
            assignment=assignment,
            supervisor_argv=tuple(misplaced),
        )


def test_dp2_route_is_release_injected_sticky_and_caller_cannot_select() -> None:
    launch, binding = _launch(mode="dp2")
    request_id = binding.route_plan.routes[0].request_id
    expected_rank = binding.route_plan.routes[0].data_parallel_rank
    first = inject_diagnostic_replica_route(
        launch=launch,
        gang_binding=binding,
        request_id=request_id,
        request_body={"rid": request_id, "sampling_params": {"max_new_tokens": 4}},
    )
    retry = inject_diagnostic_replica_route(
        launch=launch,
        gang_binding=binding,
        request_id=request_id,
        request_body={"rid": request_id, "sampling_params": {"max_new_tokens": 4}},
    )
    assert first.sha256 == retry.sha256
    assert first.to_request_body()["routed_dp_rank"] == expected_rank
    with pytest.raises(ValueError, match="caller-authored DP rank"):
        inject_diagnostic_replica_route(
            launch=launch,
            gang_binding=binding,
            request_id=request_id,
            request_body={"rid": request_id, "routed_dp_rank": 1 - expected_rank},
        )
    with pytest.raises(ValueError, match="absent or duplicated"):
        inject_diagnostic_replica_route(
            launch=launch,
            gang_binding=binding,
            request_id="foreign-request",
            request_body={"rid": "foreign-request"},
        )


def test_tp2_request_injection_rejects_request_outside_terminal_contract() -> None:
    launch, binding = _launch(mode="tp2")
    request = inject_diagnostic_replica_route(
        launch=launch,
        gang_binding=binding,
        request_id="request-0",
        request_body={"rid": "request-0"},
    )
    assert request.data_parallel_rank is None
    with pytest.raises(ValueError, match="gang request contract"):
        inject_diagnostic_replica_route(
            launch=launch,
            gang_binding=binding,
            request_id="foreign-request",
            request_body={"rid": "foreign-request"},
        )


def _run_record() -> RunRecord:
    return RunRecord(
        run_id="run-0",
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
        method="static",
        model_pair="target+drafter",
        repetition_block=0,
        started_ns=1,
        completed_ns=2,
        status="COMPLETED",
    )


def test_run_record_is_one_truthful_logical_observation_not_rank_samples() -> None:
    launch, _ = _launch(mode="dp2")
    summary = bind_diagnostic_gang_run_record(record=_run_record(), launch=launch)
    record = summary.run_record
    assert (record.tensor_parallel_size, record.data_parallel_size) == (1, 2)
    assert (record.world_size, record.rank) == (2, 0)
    assert record.rank_config_sha256 == launch.ranks[0].rank_config_sha256
    assert summary.rank_config_set_sha256 == launch.rank_config_set_sha256
    assert summary.rank_semantics.endswith("ranks_are_not_samples")
    with pytest.raises(ValueError, match="world_size conflicts"):
        bind_diagnostic_gang_run_record(
            record=replace(_run_record(), world_size=1), launch=launch
        )
    with pytest.raises(ValueError, match="method conflicts"):
        bind_diagnostic_gang_run_record(
            record=replace(_run_record(), method="l0"), launch=launch
        )


def test_completed_attempt_receipt_requires_supervisor_identity() -> None:
    launch, _ = _launch(mode="tp2")
    with pytest.raises(ValueError, match="requires a supervisor"):
        ServingGangAttemptReceipt(
            attempt_id="attempt-1",
            serving_gang_launch_sha256=launch.sha256,
            previous_failed_attempt_sha256=None,
            process_identity=None,
            status="DIAGNOSTIC_COMPLETE",
            completion_sha256=SHA_A,
            error_code=None,
            restart_required=False,
        )


class _Handle:
    def __init__(
        self,
        process_identity: str,
        *,
        ready_error: Exception | None = None,
        terminate_error: Exception | None = None,
    ) -> None:
        self._process_identity = process_identity
        self.ready_error = ready_error
        self.terminate_error = terminate_error
        self.ready_calls = 0
        self.terminate_calls = 0

    @property
    def process_identity(self) -> str:
        return self._process_identity

    async def wait_ready(self, timeout_s: float) -> None:
        assert timeout_s == 1.0
        self.ready_calls += 1
        if self.ready_error is not None:
            raise self.ready_error

    async def terminate(self, timeout_s: float) -> None:
        assert timeout_s == 2.0
        self.terminate_calls += 1
        if self.terminate_error is not None:
            raise self.terminate_error


class _Launcher:
    def __init__(self, handles: tuple[_Handle, ...]) -> None:
        self.handles = list(handles)
        self.launches: list[ServingGangLaunch] = []

    async def __call__(self, launch: ServingGangLaunch) -> _Handle:
        self.launches.append(launch)
        return self.handles.pop(0)


def _completion(launch: ServingGangLaunch) -> DiagnosticGangCompletion:
    return DiagnosticGangCompletion(
        serving_gang_launch_sha256=launch.sha256,
        terminal_aggregate_sha256=SHA_D,
        run_record_topology_summary_sha256=SHA_E,
    )


def test_fresh_executor_success_uses_and_terminates_one_supervisor() -> None:
    launch, _ = _launch(mode="tp2")
    handle = _Handle("pid-1")
    launcher = _Launcher((handle,))
    executor = FreshProcessServingGangExecutor(
        launcher, startup_timeout_s=1.0, shutdown_timeout_s=2.0
    )

    async def workload(actual: _Handle) -> DiagnosticGangCompletion:
        assert actual is handle
        return _completion(launch)

    receipt = asyncio.run(
        executor.execute_fresh(
            launch=launch,
            attempt_id="attempt-1",
            previous_failed_attempt_sha256=None,
            workload=workload,
        )
    )
    assert receipt.status == "DIAGNOSTIC_COMPLETE"
    assert receipt.restart_required is False
    assert len(launcher.launches) == 1
    assert (handle.ready_calls, handle.terminate_calls) == (1, 1)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        asyncio.run(
            executor.execute_fresh(
                launch=launch,
                attempt_id="attempt-2",
                previous_failed_attempt_sha256=None,
                workload=workload,
            )
        )


def test_failure_poison_requires_exact_lineage_and_new_supervisor() -> None:
    launch, _ = _launch(mode="tp2")
    failed_handle = _Handle("pid-1")
    recovered_handle = _Handle("pid-2")
    launcher = _Launcher((failed_handle, recovered_handle))
    executor = FreshProcessServingGangExecutor(
        launcher, startup_timeout_s=1.0, shutdown_timeout_s=2.0
    )

    async def fail(_handle: _Handle) -> DiagnosticGangCompletion:
        raise RuntimeError("one rank poisoned")

    with pytest.raises(ServingGangAttemptFailed) as error:
        asyncio.run(
            executor.execute_fresh(
                launch=launch,
                attempt_id="attempt-1",
                previous_failed_attempt_sha256=None,
                workload=fail,
            )
        )
    failed = error.value.receipt
    assert failed.status == "POISONED" and failed.restart_required
    assert failed_handle.terminate_calls == 1

    async def succeed(_handle: _Handle) -> DiagnosticGangCompletion:
        return _completion(launch)

    with pytest.raises(ValueError, match="exact failed-attempt lineage"):
        asyncio.run(
            executor.execute_fresh(
                launch=launch,
                attempt_id="attempt-2",
                previous_failed_attempt_sha256=SHA_A,
                workload=succeed,
            )
        )
    foreign_launch = replace(launch, physical_assignment_sha256=SHA_A)
    with pytest.raises(ValueError, match="changed the serving-gang launch"):
        asyncio.run(
            executor.execute_fresh(
                launch=foreign_launch,
                attempt_id="attempt-foreign",
                previous_failed_attempt_sha256=failed.sha256,
                workload=succeed,
            )
        )
    receipt = asyncio.run(
        executor.execute_fresh(
            launch=launch,
            attempt_id="attempt-3",
            previous_failed_attempt_sha256=failed.sha256,
            workload=succeed,
        )
    )
    assert receipt.status == "DIAGNOSTIC_COMPLETE"
    assert receipt.process_identity == "pid-2"
    assert recovered_handle.terminate_calls == 1


def test_readiness_or_termination_failure_poison_never_returns_completion() -> None:
    launch, _ = _launch(mode="dp2")
    for handle, expected_code in (
        (
            _Handle("pid-ready", ready_error=RuntimeError("rank missing")),
            "gang_execution_failed",
        ),
        (
            _Handle("pid-stop", terminate_error=ValueError("tree alive")),
            "gang_supervisor_termination_failed",
        ),
    ):
        executor = FreshProcessServingGangExecutor(
            _Launcher((handle,)), startup_timeout_s=1.0, shutdown_timeout_s=2.0
        )

        async def complete(_handle: _Handle) -> DiagnosticGangCompletion:
            return _completion(launch)

        with pytest.raises(ServingGangAttemptFailed) as error:
            asyncio.run(
                executor.execute_fresh(
                    launch=launch,
                    attempt_id=f"attempt-{handle.process_identity}",
                    previous_failed_attempt_sha256=None,
                    workload=complete,
                )
            )
        assert error.value.receipt.completion_sha256 is None
        assert error.value.receipt.error_code == expected_code


def test_formal_gate_blocks_before_launcher_is_called() -> None:
    launch, _ = _launch(mode="tp2")
    launcher = _Launcher((_Handle("must-not-launch"),))
    with pytest.raises(NativeTerminalGangAuthorityBlocked) as error:
        asyncio.run(
            execute_formal_serving_gang(
                launch=launch,
                claimed_capability_sha256=SHA_A,
                launcher=launcher,
            )
        )
    assert error.value.code == "native_terminal_gang_release_capability_unavailable"
    assert launcher.launches == []


def test_formal_gate_still_blocks_native_pointer_and_actual_dp_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gang_execution,
        "require_native_terminal_gang_release_capability",
        lambda **_kwargs: SHA_A,
    )
    for mode, expected_code in (
        ("tp2", "native_terminal_gang_first_party_result_pointer_unavailable"),
        ("dp2", "native_terminal_dp_actual_route_producer_unavailable"),
    ):
        launch, _ = _launch(mode=mode)
        launcher = _Launcher((_Handle("must-not-launch"),))
        with pytest.raises(NativeTerminalGangAuthorityBlocked) as error:
            asyncio.run(
                execute_formal_serving_gang(
                    launch=launch,
                    claimed_capability_sha256=SHA_A,
                    launcher=launcher,
                )
            )
        assert error.value.code == expected_code
        assert launcher.launches == []
