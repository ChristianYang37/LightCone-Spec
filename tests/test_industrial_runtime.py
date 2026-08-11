from __future__ import annotations

import multiprocessing
from dataclasses import replace
from datetime import timedelta

import pytest

from lightcone_spec.adaptation.governor import (
    BoundedCohortStateManager,
    CohortAdmissionReason,
    CohortOffloadMode,
    CohortStateKey,
    HBMAdmissionReason,
    HBMAdmissionRequest,
    HBMGovernor,
    HBMLedger,
    MemoryPressureAction,
    RankMemoryState,
)
from lightcone_spec.runtime.distributed import (
    AllRankPublicationCoordinator,
    CohortRouteIdentity,
    GlooPublicationTransport,
    InferenceParameterOwnership,
    ParameterOwnership,
    PrepareDisposition,
    PublicationCandidate,
    PublicationOutcome,
    RankDecisionReceipt,
    RankPrepare,
    RankTopologyReceipt,
    ReplicaLocalRouter,
    TopologyIdentity,
    TopologyReceiptSet,
    UpdateIdentity,
    validate_decision_receipts,
)


def _gloo_publication_worker(
    init_file: str,
    rank: int,
    queue: multiprocessing.Queue,
    nonfinite_rank: int | None,
) -> None:
    from torch import distributed

    try:
        distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=20),
        )
        topology = topology_receipts(tp=2, dp=1)
        update = candidate()
        vote = prepare_votes(topology, update)[rank]
        if rank == nonfinite_rank:
            vote = replace(vote, finite=False)
        transport = GlooPublicationTransport(topology, local_rank=rank)
        decision = transport.prepare_and_decide(update, vote)
        receipts = transport.finalize(
            decision,
            applied=decision.outcome is PublicationOutcome.COMMIT,
        )
        queue.put(
            (
                "ok",
                rank,
                decision.sha256,
                decision.outcome.value,
                tuple(receipt.rank for receipt in receipts),
            )
        )
    except Exception as error:  # noqa: BLE001  # pragma: no cover - process boundary
        queue.put(("error", rank, type(error).__name__, str(error)))
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def topology_receipts(
    *,
    tp: int = 2,
    dp: int = 2,
    process_prefix: str = "process",
) -> TopologyReceiptSet:
    world = tp * dp
    receipts = []
    for rank in range(world):
        node_rank = rank // tp
        topology = TopologyIdentity(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            node_count=dp,
            node_id=f"node-{node_rank}",
            node_rank=node_rank,
            global_rank=rank,
            local_rank=rank % tp,
            tensor_parallel_rank=rank % tp,
            data_parallel_rank=rank // tp,
            device_id=f"gpu-{rank}",
            rendezvous_id="rdzv-locked",
            router_id="router-locked",
            clock_id="clock-ptp-locked",
        )
        receipts.append(
            RankTopologyReceipt(
                topology=topology,
                process_id=f"{process_prefix}-{rank}",
                observed_world_size=world,
            )
        )
    return TopologyReceiptSet(tuple(receipts))


def candidate(*, sequence: int = 7) -> PublicationCandidate:
    return PublicationCandidate(
        update=UpdateIdentity(
            cohort_sha256="a" * 64,
            source_version=3,
            cohort_epoch=2,
            sequence_number=sequence,
            source_rows_sha256="b" * 64,
        ),
        buffer_generation=5,
        optimizer_generation=11,
    )


def prepare_votes(
    topology: TopologyReceiptSet,
    value: PublicationCandidate,
) -> tuple[RankPrepare, ...]:
    return tuple(
        RankPrepare(
            rank=rank,
            topology_receipt_sha256=topology.receipt_for_rank(rank).sha256,
            candidate_sha256=value.sha256,
            source_version=value.update.source_version,
            cohort_epoch=value.update.cohort_epoch,
            buffer_generation=value.buffer_generation,
            optimizer_generation=value.optimizer_generation,
            ready=True,
            finite=True,
            memory_reserved=True,
            safe_boundary=True,
        )
        for rank in range(topology.world_size)
    )


def decision_receipts(
    topology: TopologyReceiptSet,
    decision_sha256: str,
    *,
    applied: bool,
) -> tuple[RankDecisionReceipt, ...]:
    return tuple(
        RankDecisionReceipt(
            rank=rank,
            topology_receipt_sha256=topology.receipt_for_rank(rank).sha256,
            decision_sha256=decision_sha256,
            applied=applied,
        )
        for rank in range(topology.world_size)
    )


def test_topology_identity_is_complete_stable_and_fail_closed() -> None:
    receipts = topology_receipts()
    rebuilt = topology_receipts()
    assert receipts.topology_sha256 == rebuilt.topology_sha256
    assert receipts.receipt_sha256 == rebuilt.receipt_sha256
    assert receipts.tensor_parallel_group(0) == (0, 1)
    assert receipts.tensor_parallel_group(1) == (2, 3)
    with pytest.raises(ValueError, match="global rank"):
        replace(
            receipts.receipt_for_rank(0).topology,
            global_rank=1,
        )
    with pytest.raises(ValueError, match="cover every declared rank"):
        TopologyReceiptSet(receipts.receipts[:-1])


def test_inference_ownership_never_averages_across_dp_replicas() -> None:
    topology = topology_receipts()
    sharded = InferenceParameterOwnership(
        "layers.0.q_proj.weight",
        ParameterOwnership.SHARDED,
        (0, 1, 2, 3),
        shard_axis=0,
    )
    replicated = InferenceParameterOwnership(
        "acceptance_projection",
        ParameterOwnership.REPLICATED,
        (0, 1, 2, 3),
    )
    assert sharded.gradient_reduction_ranks(2, topology) == (2,)
    assert replicated.gradient_reduction_ranks(2, topology) == (2, 3)
    with pytest.raises(ValueError, match="partially covers"):
        InferenceParameterOwnership(
            "partial",
            ParameterOwnership.REPLICATED,
            (0, 2),
        ).validate(topology)


def test_two_phase_publication_commits_one_all_rank_decision() -> None:
    topology = topology_receipts()
    update = candidate()
    coordinator = AllRankPublicationCoordinator(topology)
    prepared = coordinator.prepare(update, prepare_votes(topology, update))
    assert prepared.disposition is PrepareDisposition.COMMIT_READY
    decision = coordinator.decide(prepared)
    assert decision.outcome is PublicationOutcome.COMMIT
    assert not decision.service_ready
    assert not decision.admission_allowed
    assert not coordinator.service_ready
    assert not coordinator.admission_allowed
    receipts = decision_receipts(topology, decision.sha256, applied=True)
    coordinator.finalize(decision, receipts)
    assert coordinator.service_ready
    assert coordinator.admission_allowed
    with pytest.raises(ValueError, match="partial model"):
        validate_decision_receipts(
            decision,
            (replace(receipts[0], applied=False), *receipts[1:]),
            topology,
        )


def test_partial_rank_copy_never_reopens_service() -> None:
    topology = topology_receipts()
    update = candidate()
    coordinator = AllRankPublicationCoordinator(topology)
    decision = coordinator.decide(
        coordinator.prepare(update, prepare_votes(topology, update))
    )
    receipts = decision_receipts(topology, decision.sha256, applied=True)
    with pytest.raises(ValueError, match="partial model"):
        coordinator.finalize(
            decision,
            (replace(receipts[0], applied=False), *receipts[1:]),
        )
    assert not coordinator.service_ready
    assert not coordinator.admission_allowed
    assert coordinator.restart_required


@pytest.mark.parametrize(
    ("nonfinite_rank", "expected"),
    [(None, "commit"), (1, "abort_static")],
)
def test_real_gloo_processes_reach_one_two_phase_decision(
    tmp_path, nonfinite_rank: int | None, expected: str
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / f"gloo-{expected}")
    processes = [
        context.Process(
            target=_gloo_publication_worker,
            args=(init_file, rank, queue, nonfinite_rank),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive(), "gloo publication worker hung"
        assert process.exitcode == 0
    rows = [queue.get(timeout=5) for _ in processes]
    assert {row[0] for row in rows} == {"ok"}
    assert {row[2] for row in rows} == {rows[0][2]}
    assert {row[3] for row in rows} == {expected}
    assert {row[4] for row in rows} == {(0, 1)}


def test_tp_candidate_failure_collectively_aborts_to_static() -> None:
    topology = topology_receipts()
    update = candidate()
    votes = list(prepare_votes(topology, update))
    votes[1] = replace(votes[1], finite=False)
    coordinator = AllRankPublicationCoordinator(topology)
    prepared = coordinator.prepare(update, tuple(votes))
    assert prepared.disposition is PrepareDisposition.ABORT_STATIC
    assert prepared.reasons == ("rank_1:finiteness",)
    decision = coordinator.decide(prepared)
    assert decision.outcome is PublicationOutcome.ABORT_STATIC
    assert decision.service_ready
    assert decision.admission_allowed
    assert not decision.restart_required
    validate_decision_receipts(
        decision,
        decision_receipts(topology, decision.sha256, applied=False),
        topology,
    )
    coordinator.finalize(
        decision,
        decision_receipts(topology, decision.sha256, applied=False),
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("source_version", 4, "source_version"),
        ("cohort_epoch", 3, "cohort_epoch"),
        ("buffer_generation", 6, "buffer_generation"),
        ("optimizer_generation", 12, "optimizer_generation"),
        ("ready", False, "readiness"),
        ("finite", False, "finiteness"),
        ("memory_reserved", False, "memory_reservation"),
        ("safe_boundary", False, "safe_boundary"),
    ],
)
def test_prepare_checks_every_candidate_generation_and_gate(
    field: str,
    value: int | bool,
    reason: str,
) -> None:
    topology = topology_receipts()
    update = candidate()
    votes = list(prepare_votes(topology, update))
    votes[0] = replace(votes[0], **{field: value})
    prepared = AllRankPublicationCoordinator(topology).prepare(update, tuple(votes))
    assert prepared.disposition is PrepareDisposition.ABORT_STATIC
    assert prepared.reasons == (f"rank_0:{reason}",)


def test_process_group_failure_stops_admission_until_clean_restart() -> None:
    topology = topology_receipts()
    update = candidate()
    votes = list(prepare_votes(topology, update))
    votes[3] = replace(votes[3], process_group_healthy=False)
    coordinator = AllRankPublicationCoordinator(topology)
    decision = coordinator.decide(coordinator.prepare(update, tuple(votes)))
    assert decision.outcome is PublicationOutcome.PROCESS_GROUP_FAILURE
    assert not coordinator.service_ready
    assert not coordinator.admission_allowed
    assert coordinator.restart_required
    with pytest.raises(RuntimeError, match="restart"):
        coordinator.prepare(
            candidate(sequence=8), prepare_votes(topology, candidate(sequence=8))
        )
    restarted = topology_receipts(process_prefix="restarted")
    assert restarted.topology_sha256 == topology.topology_sha256
    assert restarted.receipt_sha256 != topology.receipt_sha256
    coordinator.mark_process_group_restarted(restarted)
    assert coordinator.service_ready
    assert coordinator.admission_allowed
    assert not coordinator.restart_required


def test_missing_rank_is_process_failure_and_retry_identity_is_deduplicated() -> None:
    topology = topology_receipts()
    first = candidate()
    coordinator = AllRankPublicationCoordinator(topology)
    first_decision = coordinator.decide(
        coordinator.prepare(first, prepare_votes(topology, first))
    )
    coordinator.finalize(
        first_decision,
        decision_receipts(topology, first_decision.sha256, applied=True),
    )
    duplicate = coordinator.prepare(first, prepare_votes(topology, first))
    assert duplicate.disposition is PrepareDisposition.ABORT_STATIC
    assert duplicate.reasons == ("duplicate_update_identity",)
    duplicate_decision = coordinator.decide(duplicate)
    coordinator.finalize(
        duplicate_decision,
        decision_receipts(
            topology,
            duplicate_decision.sha256,
            applied=False,
        ),
    )

    second = candidate(sequence=8)
    missing = coordinator.prepare(second, prepare_votes(topology, second)[:-1])
    assert missing.disposition is PrepareDisposition.PROCESS_GROUP_FAILURE
    assert missing.reasons == ("missing_ranks:3",)


def test_replica_local_routing_is_sticky_and_topology_bound() -> None:
    topology = topology_receipts()
    router = ReplicaLocalRouter(topology)
    identity = CohortRouteIdentity(
        tenant_id="tenant-a",
        cohort_sha256="c" * 64,
        router_id="router-locked",
        topology_sha256=topology.topology_sha256,
    )
    replica = router.route(identity)
    assert router.route(identity) == replica
    assert router.ranks_for(identity) == topology.tensor_parallel_group(replica)
    assert not router.data_parallel_gradient_averaging
    with pytest.raises(ValueError, match="another router"):
        router.route(replace(identity, router_id="other-router"))


def full_ledger(*, global_peak: int) -> HBMLedger:
    return HBMLedger(
        target_weights_bytes=10,
        drafter_weights_bytes=10,
        target_kv_bytes=10,
        drafter_kv_bytes=10,
        active_merged_parameters_bytes=10,
        fp32_masters_bytes=10,
        gradients_bytes=10,
        optimizer_tensors_bytes=10,
        candidate_bytes=10,
        staging_bytes=10,
        merge_scratch_bytes=10,
        differentiable_activations_bytes=10,
        graph_private_pools_bytes=10,
        library_workspace_bytes=10,
        nccl_buffers_bytes=10,
        kv_gather_scratch_bytes=10,
        backend_scratch_bytes=10,
        telemetry_staging_bytes=10,
        fragmentation_margin_bytes=20,
        allocator_allocated_peak_bytes=250,
        allocator_reserved_peak_bytes=300,
        nvml_process_peak_bytes=280,
        nvml_global_peak_bytes=global_peak,
    )


def test_hbm_ledger_keeps_predictions_and_observations_separate() -> None:
    ledger = full_ledger(global_peak=320)
    assert ledger.predicted_resident_bytes == 130
    assert ledger.predicted_peak_bytes == 200
    assert ledger.observed_process_peak_bytes == 300
    assert ledger.prediction_error_bytes == 100
    assert RankMemoryState(0, 1000, ledger).charged_peak_bytes == 320
    with pytest.raises(ValueError, match="reserved peak"):
        replace(ledger, allocator_reserved_peak_bytes=200)


def test_hbm_admission_uses_least_feasible_rank_and_reserves_before_kv() -> None:
    governor = HBMGovernor(
        (
            RankMemoryState(0, 1000, full_ledger(global_peak=320)),
            RankMemoryState(1, 800, full_ledger(global_peak=400)),
        ),
        expected_ranks=2,
    )
    admitted = governor.assess(
        HBMAdmissionRequest(
            adaptation_reserve_bytes=100,
            target_kv_bytes=100,
            drafter_kv_bytes=50,
            safety_margin_bytes=50,
        )
    )
    assert admitted.admitted
    assert admitted.limiting_rank == 1
    assert admitted.ranks[1].headroom_bytes == 100

    kv_blocked = governor.assess(
        HBMAdmissionRequest(
            adaptation_reserve_bytes=100,
            target_kv_bytes=200,
            drafter_kv_bytes=100,
            safety_margin_bytes=50,
        )
    )
    assert not kv_blocked.admitted
    assert kv_blocked.reason is HBMAdmissionReason.KV_ADMISSION_EXCEEDS_LEAST_RANK
    adaptation_blocked = governor.assess(
        HBMAdmissionRequest(
            adaptation_reserve_bytes=400,
            target_kv_bytes=1,
            drafter_kv_bytes=1,
            safety_margin_bytes=50,
        )
    )
    assert adaptation_blocked.reason is (
        HBMAdmissionReason.ADAPTATION_RESERVE_EXCEEDS_LEAST_RANK
    )


def test_memory_pressure_order_never_sacrifices_active_correctness() -> None:
    base = HBMGovernor.pressure_plan()
    assert [step.action for step in base] == list(MemoryPressureAction)[:5]
    assert all(
        step.action is not MemoryPressureAction.OFFLOAD_COLD_INACTIVE_COHORT
        for step in base
    )
    with_offload = HBMGovernor.pressure_plan(allow_cold_offload=True)
    assert with_offload[-1].action is (
        MemoryPressureAction.OFFLOAD_COLD_INACTIVE_COHORT
    )


def cohort_key(
    suffix: str,
    *,
    tenant: str = "tenant-a",
    replica: int = 0,
) -> CohortStateKey:
    return CohortStateKey(
        tenant_id=tenant,
        cohort_sha256=suffix * 64,
        replica_id=replica,
    )


def test_cohort_manager_enforces_quota_ttl_and_privacy_isolation() -> None:
    manager = BoundedCohortStateManager(
        capacity=2,
        slab_bytes=4096,
        tenant_quotas={"tenant-a": 1, "tenant-b": 1},
        ttl_seconds=10,
    )
    a = cohort_key("a")
    a_other_tenant = cohort_key("a", tenant="tenant-b")
    assert manager.admit(a, now=0).reason is CohortAdmissionReason.ADMITTED
    assert manager.admit(cohort_key("b"), now=0).reason is (
        CohortAdmissionReason.TENANT_QUOTA
    )
    assert manager.admit(a_other_tenant, now=0).admitted
    assert manager.snapshot(a) != manager.snapshot(a_other_tenant)

    manager.acquire(a, now=5)
    expired = manager.reclaim_expired(now=11)
    assert [receipt.key for receipt in expired] == [a_other_tenant]
    assert manager.snapshot(a) is not None
    manager.release(a, now=12)
    assert manager.reclaim_expired(now=21) == ()
    final = manager.reclaim_expired(now=22)
    assert [receipt.key for receipt in final] == [a]
    assert manager.state_count == 0


def test_cohort_manager_requires_explicit_lru_reclamation() -> None:
    manager = BoundedCohortStateManager(
        capacity=2,
        slab_bytes=1024,
        tenant_quotas={"tenant-a": 2},
        ttl_seconds=100,
    )
    old = cohort_key("a")
    recent = cohort_key("b")
    assert manager.admit(old, now=0).admitted
    assert manager.admit(recent, now=1).admitted
    manager.acquire(old, now=5)
    manager.release(old, now=5)
    blocked = manager.admit(cohort_key("c"), now=6)
    assert blocked.reason is CohortAdmissionReason.TENANT_QUOTA
    reclaimed = manager.reclaim_lru(count=1, now=6)
    assert [receipt.key for receipt in reclaimed] == [recent]
    assert manager.admit(cohort_key("c"), now=6).admitted

    capacity_limited = BoundedCohortStateManager(
        capacity=1,
        slab_bytes=1024,
        tenant_quotas={"tenant-a": 1, "tenant-b": 1},
        ttl_seconds=100,
    )
    assert capacity_limited.admit(cohort_key("d"), now=0).admitted
    assert (
        capacity_limited.admit(cohort_key("e", tenant="tenant-b"), now=0).reason
        is CohortAdmissionReason.CAPACITY
    )


def test_cold_offload_is_separately_enabled_timed_and_inactive_only() -> None:
    key = cohort_key("d")
    disabled = BoundedCohortStateManager(
        capacity=1,
        slab_bytes=2048,
        tenant_quotas={"tenant-a": 1},
        ttl_seconds=100,
    )
    disabled.admit(key, now=0)
    with pytest.raises(RuntimeError, match="not explicitly enabled"):
        disabled.offload_cold(key, started_at=1, completed_at=2)

    manager = BoundedCohortStateManager(
        capacity=1,
        slab_bytes=2048,
        tenant_quotas={"tenant-a": 1},
        ttl_seconds=100,
        offload_mode=CohortOffloadMode.COLD_INACTIVE_TIMED,
        offloaded_capacity=1,
    )
    manager.admit(key, now=0)
    manager.acquire(key, now=1)
    with pytest.raises(RuntimeError, match="active"):
        manager.offload_cold(key, started_at=1, completed_at=2)
    manager.release(key, now=2)
    offload = manager.offload_cold(key, started_at=3, completed_at=4)
    assert offload.operation == "cold_offload"
    assert offload.bytes_transferred == 2048
    assert manager.resident_count == 0
    assert manager.admit(key, now=5).reason is (
        CohortAdmissionReason.OFFLOADED_RESTORE_REQUIRED
    )
    replacement = cohort_key("e")
    assert manager.admit(replacement, now=5).admitted
    assert manager.resident_count == 1
    assert manager.offloaded_count == 1
    manager.acquire(replacement, now=5)
    manager.release(replacement, now=5)
    with pytest.raises(MemoryError, match="host cohort tier is full"):
        manager.offload_cold(replacement, started_at=5, completed_at=6)
    with pytest.raises(MemoryError, match="no fixed cohort slab"):
        manager.restore_cold(key, started_at=6, completed_at=7)
    manager.reclaim(replacement, now=7, reason_code="test_reclaim")
    restore = manager.restore_cold(key, started_at=6, completed_at=7)
    assert restore.operation == "cold_restore"
    assert manager.resident_count == 1


def test_cold_offload_requires_an_explicit_bounded_host_tier() -> None:
    with pytest.raises(ValueError, match="bounded host tier"):
        BoundedCohortStateManager(
            capacity=1,
            slab_bytes=2048,
            tenant_quotas={"tenant-a": 1},
            ttl_seconds=100,
            offload_mode=CohortOffloadMode.COLD_INACTIVE_TIMED,
        )
