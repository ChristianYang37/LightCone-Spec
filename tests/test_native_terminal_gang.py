from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalProvider,
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    canonical_json_bytes,
    canonical_sha256,
)
from lightcone_spec.orchestration.native_terminal_gang import (
    NATIVE_TERMINAL_GANG_HOOK,
    NATIVE_TERMINAL_GANG_PROTOCOL_SHA256,
    NativeTerminalGangAuthorityBlocked,
    NativeTerminalGangBinding,
    NativeTerminalGangProvider,
    NativeTerminalGangTransition,
    NativeTerminalRankBinding,
    ReplicaRoutePlan,
    SingleRankNativeTerminalGangTransport,
    build_diagnostic_native_terminal_gang_transition,
    build_replica_route_plan,
    publish_native_terminal_gang_aggregate,
    publish_single_rank_legacy_aggregate,
    reopen_native_terminal_gang_aggregate_diagnostic,
    require_native_terminal_gang_aggregate,
)
from lightcone_spec.runtime.attestation import NO_TRUSTED_ATTESTERS
from lightcone_spec.runtime.distributed import (
    CohortRouteIdentity,
    RankTopologyReceipt,
    TopologyIdentity,
    TopologyReceiptSet,
)

_PROVIDER_TEST_PATH = Path(__file__).with_name("test_native_terminal_provider.py")
_PROVIDER_TEST_SPEC = importlib.util.spec_from_file_location(
    "_lightcone_test_native_terminal_provider", _PROVIDER_TEST_PATH
)
assert _PROVIDER_TEST_SPEC is not None and _PROVIDER_TEST_SPEC.loader is not None
_PROVIDER_TEST_MODULE = importlib.util.module_from_spec(_PROVIDER_TEST_SPEC)
_PROVIDER_TEST_SPEC.loader.exec_module(_PROVIDER_TEST_MODULE)
FakeAdminTransport = _PROVIDER_TEST_MODULE.FakeAdminTransport


class _RankAdminTransport(FakeAdminTransport):
    """Give the imported v1 fixture one distinct scheduler process identity."""

    def __init__(self, *args, process_id: int, process_started_ns: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_id = process_id
        self.process_started_ns = process_started_ns

    def _begin(self, payload):
        value = super()._begin(payload)
        value["server_process_id"] = self.process_id
        value["server_process_started_ns"] = self.process_started_ns
        value.pop("begin_sha256")
        value["begin_sha256"] = canonical_sha256(value)
        return value

    def _reset(self, payload):
        value = super()._reset(payload)
        value["server_process_id"] = self.process_id
        value["server_process_started_ns"] = self.process_started_ns
        value.pop("reset_sha256")
        value["reset_sha256"] = canonical_sha256(value)
        return value

    def _terminal(self, payload):
        value = super()._terminal(payload)
        value["server_process_id"] = self.process_id
        value["server_process_started_ns"] = self.process_started_ns
        value.pop("terminal_sha256")
        attestation = value.pop("attestation")
        value["terminal_sha256"] = canonical_sha256(value)
        message = {
            "schema_version": 1,
            "kind": "lightcone_terminal_attestation_challenge",
            "hook": value["hook"],
            "challenge_nonce_sha256": value["challenge_nonce_sha256"],
            "terminal_sha256": value["terminal_sha256"],
            "run_id": value["run_id"],
            "run_nonce_sha256": value["run_nonce_sha256"],
            "server_process_id": self.process_id,
            "server_process_started_ns": self.process_started_ns,
            "session_id": value["session_id"],
            "session_epoch": value["session_epoch"],
            "attempt_id": value["attempt_id"],
        }
        attestation["message_sha256"] = canonical_sha256(message)
        value["attestation"] = attestation
        return value


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _request(request_id: str) -> TerminalRequestExpectation:
    return TerminalRequestExpectation(
        request_id=request_id,
        input_token_ids=(1, 2),
        output_token_ids=(3,),
        terminal_status="completed",
        terminal_reason="FINISH_LENGTH",
        submitted_to_server=True,
    )


def _topology(*, tp: int, dp: int) -> TopologyReceiptSet:
    world = tp * dp
    receipts = []
    for rank in range(world):
        receipts.append(
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
                    device_id=f"gpu-{rank}",
                    rendezvous_id="rdzv-0",
                    router_id="router-0",
                    clock_id="clock-0",
                ),
                process_id=f"scheduler-{rank}",
                observed_world_size=world,
            )
        )
    return TopologyReceiptSet(receipts=tuple(receipts))


def _rank_binding(
    topology: TopologyReceiptSet,
    *,
    rank: int,
    warmup: tuple[str, ...],
    scored: tuple[str, ...],
) -> NativeTerminalRankBinding:
    topo = topology.receipt_for_rank(rank).topology
    run = NativeTerminalRunBinding(
        run_id="run-0",
        run_nonce_sha256=SHA_A,
        execution_plan_sha256=SHA_B,
        rank_config_sha256=canonical_sha256({"rank": rank}),
        attempt_id="attempt-0",
        session_id="session-0",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=SHA_D,
        method="static",
        warmup_request_ids=warmup,
        scored_request_ids=scored,
    )
    return NativeTerminalRankBinding(
        run=run,
        topology_sha256=topology.topology_sha256,
        topology_receipt_sha256=topology.receipt_for_rank(rank).sha256,
        global_rank=rank,
        tensor_parallel_rank=topo.tensor_parallel_rank,
        data_parallel_rank=topo.data_parallel_rank,
        tensor_parallel_size=topology.tensor_parallel_size,
        data_parallel_size=topology.data_parallel_size,
        world_size=topology.world_size,
        node_count=1,
        node_rank=0,
    )


def _tp2_binding() -> NativeTerminalGangBinding:
    topology = _topology(tp=2, dp=1)
    return NativeTerminalGangBinding(
        ranks=tuple(
            _rank_binding(
                topology,
                rank=rank,
                warmup=("warm-0",),
                scored=("score-0",),
            )
            for rank in range(2)
        )
    )


def _dp2_binding() -> NativeTerminalGangBinding:
    topology = _topology(tp=1, dp=2)
    identities = (
        CohortRouteIdentity("tenant-0", SHA_A, "router-0", topology.topology_sha256),
        CohortRouteIdentity("tenant-1", SHA_B, "router-0", topology.topology_sha256),
        CohortRouteIdentity("tenant-2", SHA_C, "router-0", topology.topology_sha256),
        CohortRouteIdentity("tenant-3", SHA_D, "router-0", topology.topology_sha256),
    )
    request_cohorts = tuple(
        (f"score-{index}", identity) for index, identity in enumerate(identities)
    )
    plan = build_replica_route_plan(
        topology=topology,
        request_cohorts=request_cohorts,
    )
    ranks = tuple(
        _rank_binding(
            topology,
            rank=rank,
            warmup=(),
            scored=tuple(
                route.request_id
                for route in plan.routes
                if route.data_parallel_rank == rank
            ),
        )
        for rank in range(2)
    )
    return NativeTerminalGangBinding(ranks=ranks, route_plan=plan)


def _transition(
    binding: NativeTerminalGangBinding,
    phase: str,
    *,
    digests: tuple[str, ...] | None = None,
) -> NativeTerminalGangTransition:
    return build_diagnostic_native_terminal_gang_transition(
        binding=binding,
        phase=phase,
        process_identities=tuple((1234 + rank, 1_000_000 + rank) for rank in range(2)),
        phase_receipt_sha256s=digests or (SHA_E, SHA_F),
    )


def test_rank_gang_and_route_codecs_are_strict_and_sticky() -> None:
    binding = _dp2_binding()
    assert binding == NativeTerminalGangBinding.from_dict(binding.to_dict())
    assert binding.route_plan == ReplicaRoutePlan.from_dict(
        binding.route_plan.to_dict()
    )
    assert binding.route_plan.routes
    by_cohort: dict[str, set[int]] = {}
    for route in binding.route_plan.routes:
        by_cohort.setdefault(route.cohort_identity_sha256, set()).add(
            route.data_parallel_rank
        )
        assert route.rank_group == (route.data_parallel_rank,)
    assert all(len(ranks) == 1 for ranks in by_cohort.values())

    unknown = binding.to_dict()
    unknown["caller_rank"] = 0
    with pytest.raises(ValueError, match="fields differ"):
        NativeTerminalGangBinding.from_dict(unknown)

    wrong_rank = binding.to_dict()
    wrong_rank["ranks"] = list(reversed(wrong_rank["ranks"]))
    with pytest.raises(ValueError, match="global-rank sorted"):
        NativeTerminalGangBinding.from_dict(wrong_rank)

    changed = binding.route_plan.to_dict()
    changed["routes"][0]["data_parallel_rank"] ^= 1
    with pytest.raises(ValueError, match="rank group|route plan"):
        NativeTerminalGangBinding(
            ranks=binding.ranks,
            route_plan=ReplicaRoutePlan.from_dict(changed),
        )
    with pytest.raises(ValueError, match="one local node"):
        replace(binding.ranks[0], node_count=2)


def test_dp_transition_requires_exact_native_actual_route_proof() -> None:
    binding = _dp2_binding()
    transition = _transition(binding, "finalize")
    transition.validate(binding)
    rank = replace(
        transition.ranks[0],
        native_route_proof_sha256=SHA_A,
    )
    rank = replace(rank, phase_sha256=rank.expected_phase_sha256)
    forged = replace(transition, ranks=(rank, transition.ranks[1]))
    with pytest.raises(ValueError, match="actual-route proof"):
        forged.validate(binding)


class _FakeGangTransport:
    def __init__(
        self,
        binding: NativeTerminalGangBinding,
        *,
        fail_phase: str | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.binding = binding
        self.fail_phase = fail_phase
        self.delay_s = delay_s
        self.evidence = ()

    async def capability(self, binding: NativeTerminalGangBinding) -> object:
        await asyncio.sleep(self.delay_s)
        if self.fail_phase == "capability":
            raise RuntimeError("capability failed")
        return {
            "schema_version": 1,
            "hook": NATIVE_TERMINAL_GANG_HOOK,
            "protocol_sha256": NATIVE_TERMINAL_GANG_PROTOCOL_SHA256,
            "topology_sha256": binding.topology_sha256,
            "world_size": binding.world_size,
            "native_all_rank_receipts": True,
            "native_actual_route_proof": binding.route_plan is not None,
        }

    async def begin_all(
        self, binding: NativeTerminalGangBinding
    ) -> NativeTerminalGangTransition:
        if self.fail_phase == "begin":
            raise RuntimeError("one rank begin failed")
        return _transition(binding, "begin")

    async def reset_all(self, binding, warmup_requests):
        if self.fail_phase == "reset":
            raise RuntimeError("one rank reset failed")
        return _transition(binding, "reset")

    async def finalize_all(self, binding, scored_requests):
        if self.fail_phase == "finalize":
            raise RuntimeError("one rank finalize failed")
        digests = tuple(item.terminal_sha256 for item in self.evidence)
        return _transition(binding, "finalize", digests=digests), self.evidence


def test_all_rank_provider_poison_is_permanent_on_partial_failure_and_timeout() -> None:
    binding = _tp2_binding()
    requests = {rank: (_request("score-0"),) for rank in range(binding.world_size)}
    warmup = {rank: (_request("warm-0"),) for rank in range(binding.world_size)}

    async def fail() -> None:
        provider = NativeTerminalGangProvider(
            _FakeGangTransport(binding, fail_phase="reset"), timeout_s=1.0
        )
        await provider.begin(binding)
        with pytest.raises(RuntimeError, match="one rank reset failed"):
            await provider.reset(warmup_requests=warmup)
        assert provider.phase == "FAILED"
        with pytest.raises(RuntimeError, match="requires phase SCORED"):
            await provider.finalize(scored_requests=requests)
        assert provider.phase == "FAILED"

    async def timeout() -> None:
        provider = NativeTerminalGangProvider(
            _FakeGangTransport(binding, delay_s=0.05), timeout_s=0.001
        )
        with pytest.raises(TimeoutError):
            await provider.begin(binding)
        assert provider.phase == "FAILED"

    asyncio.run(fail())
    asyncio.run(timeout())


def test_world1_provider_adapter_runs_unchanged_v1_lifecycle() -> None:
    topology = _topology(tp=1, dp=1)
    rank = _rank_binding(
        topology,
        rank=0,
        warmup=("warm-0",),
        scored=("score-0",),
    )
    binding = NativeTerminalGangBinding(ranks=(rank,))
    warmup = {0: (_request("warm-0"),)}
    scored = {0: (_request("score-0"),)}
    admin = FakeAdminTransport(
        binding=rank.run,
        warmup=warmup[0],
        scored=scored[0],
    )
    provider = NativeTerminalGangProvider(
        SingleRankNativeTerminalGangTransport(NativeTerminalProvider(admin)),
        timeout_s=1.0,
    )

    async def run():
        begin = await provider.begin(binding)
        reset = await provider.reset(warmup_requests=warmup)
        final, evidence = await provider.finalize(scored_requests=scored)
        return begin, reset, final, evidence

    begin, reset, final, evidence = asyncio.run(run())
    assert provider.phase == "FINALIZED"
    assert (begin.phase, reset.phase, final.phase) == ("begin", "reset", "finalize")
    assert evidence[0].binding == rank.run
    # The adapter only wraps the v1 wire; begin still performs its mandatory
    # live capability GET in addition to the gang preflight.
    assert len(admin.get_paths) == 2


async def _world1_evidence():
    topology = _topology(tp=1, dp=1)
    rank = _rank_binding(
        topology,
        rank=0,
        warmup=("warm-0",),
        scored=("score-0",),
    )
    warmup = (_request("warm-0"),)
    scored = (_request("score-0"),)
    transport = FakeAdminTransport(
        binding=rank.run,
        warmup=warmup,
        scored=scored,
    )
    provider = NativeTerminalProvider(transport)
    await provider.begin(rank.run)
    await provider.reset(warmup_requests=warmup)
    evidence = await provider.finalize(requests=scored)
    return rank, warmup, scored, evidence


def test_world1_adapter_keeps_v1_artifact_and_resume_exact(tmp_path: Path) -> None:
    rank, warmup, scored, evidence = asyncio.run(_world1_evidence())
    path = (tmp_path / "aggregate.json").resolve()
    receipt = publish_single_rank_legacy_aggregate(
        output_path=path,
        rank_binding=rank,
        evidence=evidence,
        warmup_requests=warmup,
    )
    assert receipt.gang_binding.world_size == 1
    rank_value = json.loads(Path(receipt.rank_artifacts[0].path).read_text())
    assert rank_value["rank"] == 0
    assert rank_value["artifact_kind"] == "native_terminal_evidence_bundle_v1"

    reopened = reopen_native_terminal_gang_aggregate_diagnostic(
        aggregate_path=path,
        expected_aggregate_sha256=receipt.sha256,
        expected_binding=receipt.gang_binding,
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        expected_warmup_requests={0: warmup},
        expected_scored_requests={0: scored},
    )
    assert reopened == receipt


def _tp2_artifacts():
    binding = _tp2_binding()
    warmup = {rank: (_request("warm-0"),) for rank in range(2)}
    scored = {rank: (_request("score-0"),) for rank in range(2)}

    async def build():
        evidence = []
        for rank in range(2):
            transport = _RankAdminTransport(
                binding=binding.ranks[rank].run,
                warmup=warmup[rank],
                scored=scored[rank],
                process_id=1234 + rank,
                process_started_ns=1_000_000 + rank,
            )
            provider = NativeTerminalProvider(transport)
            await provider.begin(binding.ranks[rank].run)
            await provider.reset(warmup_requests=warmup[rank])
            evidence.append(await provider.finalize(requests=scored[rank]))
        return tuple(evidence)

    evidence = asyncio.run(build())
    processes = ((1234, 1_000_000), (1235, 1_000_001))
    transitions = tuple(
        build_diagnostic_native_terminal_gang_transition(
            binding=binding,
            phase=phase,
            process_identities=processes,
            phase_receipt_sha256s=tuple(digest(item) for item in evidence),
        )
        for phase, digest in (
            ("begin", lambda item: item.begin_receipt.begin_sha256),
            ("reset", lambda item: item.reset_receipt.reset_sha256),
            ("finalize", lambda item: item.terminal_sha256),
        )
    )
    return binding, warmup, scored, evidence, transitions


def test_publish_last_and_resume_reject_rank_tamper_and_coordinated_rehash(
    tmp_path: Path,
) -> None:
    binding, warmup, scored, evidence, transitions = _tp2_artifacts()
    path = (tmp_path / "aggregate.json").resolve()
    receipt = publish_native_terminal_gang_aggregate(
        output_path=path,
        binding=binding,
        transitions=transitions,
        rank_artifacts=evidence,
        warmup_requests=warmup,
    )
    assert path.exists() and Path(f"{path}.sha256").exists()
    assert all(Path(row.path).exists() for row in receipt.rank_artifacts)
    assert (
        reopen_native_terminal_gang_aggregate_diagnostic(
            aggregate_path=path,
            expected_aggregate_sha256=receipt.sha256,
            expected_binding=binding,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            expected_warmup_requests=warmup,
            expected_scored_requests=scored,
        )
        == receipt
    )

    rank_path = Path(receipt.rank_artifacts[1].path)
    value = json.loads(rank_path.read_text())
    value["terminal_sha256"] = SHA_A
    body = canonical_json_bytes(value)
    rank_path.write_bytes(body)
    Path(f"{rank_path}.sha256").write_text(
        hashlib.sha256(body).hexdigest() + "\n", encoding="ascii"
    )
    aggregate = receipt.to_dict()
    aggregate["rank_artifacts"][1]["raw_sha256"] = hashlib.sha256(body).hexdigest()
    aggregate["rank_artifacts"][1]["sidecar_file_sha256"] = hashlib.sha256(
        Path(f"{rank_path}.sha256").read_bytes()
    ).hexdigest()
    aggregate_body = canonical_json_bytes(aggregate)
    path.write_bytes(aggregate_body)
    Path(f"{path}.sha256").write_text(
        canonical_sha256(aggregate) + "\n", encoding="ascii"
    )
    with pytest.raises((ValueError, RuntimeError)):
        reopen_native_terminal_gang_aggregate_diagnostic(
            aggregate_path=path,
            expected_aggregate_sha256=receipt.sha256,
            expected_binding=binding,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            expected_warmup_requests=warmup,
            expected_scored_requests=scored,
        )


def test_resume_rejects_missing_rank_symlink_duplicate_json_and_formal_blocks_prepath(
    tmp_path: Path,
) -> None:
    rank, warmup, scored, evidence = asyncio.run(_world1_evidence())
    path = (tmp_path / "aggregate.json").resolve()
    receipt = publish_single_rank_legacy_aggregate(
        output_path=path,
        rank_binding=rank,
        evidence=evidence,
        warmup_requests=warmup,
    )
    rank_path = Path(receipt.rank_artifacts[0].path)
    backup = rank_path.with_suffix(".backup")
    rank_path.rename(backup)
    rank_path.symlink_to(backup)
    with pytest.raises((ValueError, RuntimeError), match="symlink-free|readable"):
        reopen_native_terminal_gang_aggregate_diagnostic(
            aggregate_path=path,
            expected_aggregate_sha256=receipt.sha256,
            expected_binding=receipt.gang_binding,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            expected_warmup_requests={0: warmup},
            expected_scored_requests={0: scored},
        )

    missing = (tmp_path / "does-not-exist.json").resolve()
    with pytest.raises(NativeTerminalGangAuthorityBlocked) as error:
        require_native_terminal_gang_aggregate(
            aggregate_path=missing,
            expected_aggregate_sha256=receipt.sha256,
            expected_binding=receipt.gang_binding,
            claimed_capability_sha256=SHA_A,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            expected_warmup_requests={0: warmup},
            expected_scored_requests={0: scored},
        )
    assert error.value.code == "native_terminal_gang_release_capability_unavailable"
    assert not missing.exists()

    duplicate = (tmp_path / "duplicate.json").resolve()
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    Path(f"{duplicate}.sha256").write_text(SHA_A + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        reopen_native_terminal_gang_aggregate_diagnostic(
            aggregate_path=duplicate,
            expected_aggregate_sha256=SHA_A,
            expected_binding=receipt.gang_binding,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            expected_warmup_requests={0: warmup},
            expected_scored_requests={0: scored},
        )


def test_publication_preflight_writes_nothing_on_missing_or_wrong_rank(
    tmp_path: Path,
) -> None:
    binding, warmup, _, evidence, transitions = _tp2_artifacts()
    path = (tmp_path / "aggregate.json").resolve()
    with pytest.raises(ValueError, match="exact world coverage"):
        publish_native_terminal_gang_aggregate(
            output_path=path,
            binding=binding,
            transitions=transitions,
            rank_artifacts=evidence[:1],
            warmup_requests=warmup,
        )
    assert tuple(tmp_path.iterdir()) == ()

    wrong = (evidence[1], evidence[0])
    with pytest.raises(ValueError, match="binding/transition"):
        publish_native_terminal_gang_aggregate(
            output_path=path,
            binding=binding,
            transitions=transitions,
            rank_artifacts=wrong,
            warmup_requests=warmup,
        )
    assert tuple(tmp_path.iterdir()) == ()


def test_gang_files_are_single_link_and_immutable(tmp_path: Path) -> None:
    rank, warmup, _, evidence = asyncio.run(_world1_evidence())
    path = (tmp_path / "aggregate.json").resolve()
    receipt = publish_single_rank_legacy_aggregate(
        output_path=path,
        rank_binding=rank,
        evidence=evidence,
        warmup_requests=warmup,
    )
    for item in (
        path,
        Path(f"{path}.sha256"),
        Path(receipt.rank_artifacts[0].path),
        Path(receipt.rank_artifacts[0].sidecar_path),
    ):
        assert os.stat(item, follow_symlinks=False).st_nlink == 1
    with pytest.raises(FileExistsError):
        publish_single_rank_legacy_aggregate(
            output_path=path,
            rank_binding=rank,
            evidence=evidence,
            warmup_requests=warmup,
        )


@pytest.mark.parametrize("mutation", ["missing", "hardlink"])
def test_resume_rejects_missing_or_hardlinked_rank_artifact(
    tmp_path: Path, mutation: str
) -> None:
    rank, warmup, scored, evidence = asyncio.run(_world1_evidence())
    path = (tmp_path / "aggregate.json").resolve()
    receipt = publish_single_rank_legacy_aggregate(
        output_path=path,
        rank_binding=rank,
        evidence=evidence,
        warmup_requests=warmup,
    )
    rank_path = Path(receipt.rank_artifacts[0].path)
    if mutation == "missing":
        rank_path.rename(rank_path.with_suffix(".retained-attempt"))
    else:
        os.link(rank_path, rank_path.with_suffix(".hardlink"))
    with pytest.raises(RuntimeError, match="readable|single-link"):
        reopen_native_terminal_gang_aggregate_diagnostic(
            aggregate_path=path,
            expected_aggregate_sha256=receipt.sha256,
            expected_binding=receipt.gang_binding,
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            expected_warmup_requests={0: warmup},
            expected_scored_requests={0: scored},
        )
