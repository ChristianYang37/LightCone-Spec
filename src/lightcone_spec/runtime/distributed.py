"""Distributed adaptation identities and two-phase publication contracts.

The deterministic coordinator is transport independent. ``GlooPublicationTransport``
exercises the real all-rank protocol on CPU process groups; the pinned SGLang patch
owns the NCCL/CUDA transport and must provide a separately attested capability receipt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


def _sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _require_hash(name: str, value: str, length: int = 64) -> None:
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase {length * 4}-bit hash")


def _require_nonempty(name: str, value: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical identifier")


def _require_counter(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class TopologyIdentity:
    """Complete rank identity used by manifests and rank receipts."""

    tensor_parallel_size: int
    data_parallel_size: int
    node_count: int
    node_id: str
    node_rank: int
    global_rank: int
    local_rank: int
    tensor_parallel_rank: int
    data_parallel_rank: int
    device_id: str
    rendezvous_id: str
    router_id: str
    clock_id: str

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "node_count",
        ):
            _require_counter(name, getattr(self, name), minimum=1)
        for name in (
            "node_rank",
            "global_rank",
            "local_rank",
            "tensor_parallel_rank",
            "data_parallel_rank",
        ):
            _require_counter(name, getattr(self, name))
        for name in (
            "node_id",
            "device_id",
            "rendezvous_id",
            "router_id",
            "clock_id",
        ):
            _require_nonempty(name, getattr(self, name))
        if self.node_rank >= self.node_count:
            raise ValueError("node_rank is outside the declared node topology")
        if self.global_rank >= self.world_size:
            raise ValueError("global_rank is outside the declared world")
        if self.tensor_parallel_rank >= self.tensor_parallel_size:
            raise ValueError("tensor_parallel_rank is outside its TP group")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise ValueError("data_parallel_rank is outside its DP topology")
        expected_rank = (
            self.data_parallel_rank * self.tensor_parallel_size
            + self.tensor_parallel_rank
        )
        if self.global_rank != expected_rank:
            raise ValueError("global rank does not match the declared TP/DP ranks")

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))

    @property
    def common_identity(self) -> dict[str, int | str]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "node_count": self.node_count,
            "rendezvous_id": self.rendezvous_id,
            "router_id": self.router_id,
            "clock_id": self.clock_id,
        }


@dataclass(frozen=True)
class RankTopologyReceipt:
    """A process-bound observation of one declared rank identity."""

    topology: TopologyIdentity
    process_id: str
    observed_world_size: int

    def __post_init__(self) -> None:
        _require_nonempty("process_id", self.process_id)
        _require_counter("observed_world_size", self.observed_world_size, minimum=1)
        if self.observed_world_size != self.topology.world_size:
            raise ValueError("rank observed a different process-group world size")

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "topology": asdict(self.topology),
                "process_id": self.process_id,
                "observed_world_size": self.observed_world_size,
            }
        )


@dataclass(frozen=True)
class TopologyReceiptSet:
    """Exact all-rank topology coverage with a stable topology digest."""

    receipts: tuple[RankTopologyReceipt, ...]

    def __post_init__(self) -> None:
        if not self.receipts:
            raise ValueError("topology receipts cannot be empty")
        first = self.receipts[0].topology
        expected_ranks = set(range(first.world_size))
        ranks = [receipt.topology.global_rank for receipt in self.receipts]
        if len(ranks) != len(set(ranks)):
            raise ValueError("duplicate topology rank receipt")
        if set(ranks) != expected_ranks:
            raise ValueError("topology receipts do not cover every declared rank")
        if len({receipt.process_id for receipt in self.receipts}) != len(self.receipts):
            raise ValueError("topology process identities must be unique")
        if any(
            receipt.topology.common_identity != first.common_identity
            for receipt in self.receipts
        ):
            raise ValueError("rank receipts disagree on the common topology identity")
        devices = [receipt.topology.device_id for receipt in self.receipts]
        if len(devices) != len(set(devices)):
            raise ValueError("each rank must bind a distinct device identity")
        node_pairs = {
            (receipt.topology.node_rank, receipt.topology.node_id)
            for receipt in self.receipts
        }
        if len({rank for rank, _ in node_pairs}) != first.node_count:
            raise ValueError("topology receipts do not cover every declared node")
        if len({node_id for _, node_id in node_pairs}) != first.node_count:
            raise ValueError("node ranks and node identities are not one-to-one")
        local_ranks = [
            (receipt.topology.node_rank, receipt.topology.local_rank)
            for receipt in self.receipts
        ]
        if len(local_ranks) != len(set(local_ranks)):
            raise ValueError("local rank is duplicated within a node")

    @property
    def world_size(self) -> int:
        return self.receipts[0].topology.world_size

    @property
    def tensor_parallel_size(self) -> int:
        return self.receipts[0].topology.tensor_parallel_size

    @property
    def data_parallel_size(self) -> int:
        return self.receipts[0].topology.data_parallel_size

    @property
    def topology_sha256(self) -> str:
        identities = [
            asdict(receipt.topology)
            for receipt in sorted(
                self.receipts, key=lambda item: item.topology.global_rank
            )
        ]
        return _sha256(identities)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(
            [
                receipt.sha256
                for receipt in sorted(
                    self.receipts, key=lambda item: item.topology.global_rank
                )
            ]
        )

    def receipt_for_rank(self, rank: int) -> RankTopologyReceipt:
        _require_counter("rank", rank)
        for receipt in self.receipts:
            if receipt.topology.global_rank == rank:
                return receipt
        raise KeyError(f"rank {rank} is outside the topology")

    def tensor_parallel_group(self, data_parallel_rank: int) -> tuple[int, ...]:
        _require_counter("data_parallel_rank", data_parallel_rank)
        if data_parallel_rank >= self.data_parallel_size:
            raise ValueError("data_parallel_rank is outside the topology")
        start = data_parallel_rank * self.tensor_parallel_size
        return tuple(range(start, start + self.tensor_parallel_size))


class ParameterOwnership(str, Enum):
    SHARDED = "sharded"
    REPLICATED = "replicated"


@dataclass(frozen=True)
class InferenceParameterOwnership:
    """Inference-aligned state ownership without cross-DP convenience gathers."""

    parameter_name: str
    ownership: ParameterOwnership
    owner_ranks: tuple[int, ...]
    shard_axis: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty("parameter_name", self.parameter_name)
        if not self.owner_ranks:
            raise ValueError("parameter ownership requires at least one rank")
        if len(set(self.owner_ranks)) != len(self.owner_ranks):
            raise ValueError("parameter owner ranks must be unique")
        for rank in self.owner_ranks:
            _require_counter("owner rank", rank)
        if self.ownership is ParameterOwnership.SHARDED:
            if self.shard_axis is None:
                raise ValueError("sharded parameters require a shard axis")
            _require_counter("shard_axis", self.shard_axis)
        elif self.shard_axis is not None:
            raise ValueError("replicated parameters cannot declare a shard axis")

    def validate(self, topology: TopologyReceiptSet) -> None:
        if any(rank >= topology.world_size for rank in self.owner_ranks):
            raise ValueError("parameter owner rank is outside the topology")
        owner_set = set(self.owner_ranks)
        for replica in range(topology.data_parallel_size):
            group = set(topology.tensor_parallel_group(replica))
            overlap = group & owner_set
            if overlap and overlap != group:
                raise ValueError("parameter ownership partially covers a TP replica")

    def gradient_reduction_ranks(
        self,
        rank: int,
        topology: TopologyReceiptSet,
    ) -> tuple[int, ...]:
        self.validate(topology)
        if rank not in self.owner_ranks:
            raise ValueError("rank does not own this parameter")
        if self.ownership is ParameterOwnership.SHARDED:
            return (rank,)
        replica = rank // topology.tensor_parallel_size
        return tuple(
            member
            for member in topology.tensor_parallel_group(replica)
            if member in self.owner_ranks
        )


@dataclass(frozen=True)
class CohortRouteIdentity:
    tenant_id: str
    cohort_sha256: str
    router_id: str
    topology_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty("tenant_id", self.tenant_id)
        _require_nonempty("router_id", self.router_id)
        _require_hash("cohort_sha256", self.cohort_sha256)
        _require_hash("topology_sha256", self.topology_sha256)

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


class ReplicaLocalRouter:
    """Sticky cohort routing; DP replicas never average adaptation gradients."""

    data_parallel_gradient_averaging = False

    def __init__(self, topology: TopologyReceiptSet) -> None:
        self.topology = topology
        self._routes: dict[str, int] = {}

    def route(self, identity: CohortRouteIdentity) -> int:
        reference = self.topology.receipts[0].topology
        if identity.topology_sha256 != self.topology.topology_sha256:
            raise ValueError("cohort route belongs to another topology")
        if identity.router_id != reference.router_id:
            raise ValueError("cohort route belongs to another router")
        existing = self._routes.get(identity.sha256)
        if existing is not None:
            return existing
        replica = int(identity.sha256[:16], 16) % self.topology.data_parallel_size
        self._routes[identity.sha256] = replica
        return replica

    def ranks_for(self, identity: CohortRouteIdentity) -> tuple[int, ...]:
        return self.topology.tensor_parallel_group(self.route(identity))


@dataclass(frozen=True)
class UpdateIdentity:
    """Retry-stable identity for exactly one source update."""

    cohort_sha256: str
    source_version: int
    cohort_epoch: int
    sequence_number: int
    source_rows_sha256: str

    def __post_init__(self) -> None:
        _require_hash("cohort_sha256", self.cohort_sha256)
        _require_hash("source_rows_sha256", self.source_rows_sha256)
        for name in ("source_version", "cohort_epoch", "sequence_number"):
            _require_counter(name, getattr(self, name))

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class PublicationCandidate:
    update: UpdateIdentity
    buffer_generation: int
    optimizer_generation: int

    def __post_init__(self) -> None:
        _require_counter("buffer_generation", self.buffer_generation)
        _require_counter("optimizer_generation", self.optimizer_generation)

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "update_sha256": self.update.sha256,
                "buffer_generation": self.buffer_generation,
                "optimizer_generation": self.optimizer_generation,
            }
        )


@dataclass(frozen=True)
class RankPrepare:
    rank: int
    topology_receipt_sha256: str
    candidate_sha256: str
    source_version: int
    cohort_epoch: int
    buffer_generation: int
    optimizer_generation: int
    ready: bool
    finite: bool
    memory_reserved: bool
    safe_boundary: bool
    process_group_healthy: bool = True

    def __post_init__(self) -> None:
        _require_counter("rank", self.rank)
        _require_hash("topology_receipt_sha256", self.topology_receipt_sha256)
        _require_hash("candidate_sha256", self.candidate_sha256)
        for name in (
            "source_version",
            "cohort_epoch",
            "buffer_generation",
            "optimizer_generation",
        ):
            _require_counter(name, getattr(self, name))
        for name in (
            "ready",
            "finite",
            "memory_reserved",
            "safe_boundary",
            "process_group_healthy",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


class PrepareDisposition(str, Enum):
    COMMIT_READY = "commit_ready"
    ABORT_STATIC = "abort_static"
    PROCESS_GROUP_FAILURE = "process_group_failure"


@dataclass(frozen=True)
class PreparedPublication:
    update_sha256: str
    candidate_sha256: str
    topology_sha256: str
    disposition: PrepareDisposition
    reasons: tuple[str, ...]
    ranks: tuple[int, ...]

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "update_sha256": self.update_sha256,
                "candidate_sha256": self.candidate_sha256,
                "topology_sha256": self.topology_sha256,
                "disposition": self.disposition.value,
                "reasons": self.reasons,
                "ranks": self.ranks,
            }
        )


class PublicationOutcome(str, Enum):
    COMMIT = "commit"
    ABORT_STATIC = "abort_static"
    PROCESS_GROUP_FAILURE = "process_group_failure"


@dataclass(frozen=True)
class PublicationDecision:
    update_sha256: str
    candidate_sha256: str
    topology_sha256: str
    outcome: PublicationOutcome
    reasons: tuple[str, ...]
    ranks: tuple[int, ...]
    service_ready: bool
    admission_allowed: bool
    restart_required: bool

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "update_sha256": self.update_sha256,
                "candidate_sha256": self.candidate_sha256,
                "topology_sha256": self.topology_sha256,
                "outcome": self.outcome.value,
                "reasons": self.reasons,
                "ranks": self.ranks,
                "service_ready": self.service_ready,
                "admission_allowed": self.admission_allowed,
                "restart_required": self.restart_required,
            }
        )


@dataclass(frozen=True)
class RankDecisionReceipt:
    rank: int
    topology_receipt_sha256: str
    decision_sha256: str
    applied: bool

    def __post_init__(self) -> None:
        _require_counter("rank", self.rank)
        _require_hash("topology_receipt_sha256", self.topology_receipt_sha256)
        _require_hash("decision_sha256", self.decision_sha256)
        if type(self.applied) is not bool:
            raise ValueError("applied must be a boolean")


def validate_decision_receipts(
    decision: PublicationDecision,
    receipts: tuple[RankDecisionReceipt, ...],
    topology: TopologyReceiptSet,
) -> None:
    """Reject missing, duplicate, mixed, or foreign all-rank outcomes."""

    ranks = [receipt.rank for receipt in receipts]
    expected = set(range(topology.world_size))
    if len(ranks) != len(set(ranks)):
        raise ValueError("duplicate publication decision receipt")
    if set(ranks) != expected:
        raise ValueError("publication decision receipts lack all-rank coverage")
    should_apply = decision.outcome is PublicationOutcome.COMMIT
    for receipt in receipts:
        topology_receipt = topology.receipt_for_rank(receipt.rank)
        if receipt.topology_receipt_sha256 != topology_receipt.sha256:
            raise ValueError("publication receipt binds another rank topology")
        if receipt.decision_sha256 != decision.sha256:
            raise ValueError("ranks did not observe one publication decision")
        if receipt.applied is not should_apply:
            raise ValueError("publication receipt would create a partial model")


class AllRankPublicationCoordinator:
    """Collective prepare/decide protocol with fail-closed service state."""

    def __init__(self, topology: TopologyReceiptSet) -> None:
        self.topology = topology
        self.service_ready = True
        self.admission_allowed = True
        self.restart_required = False
        self._pending: PreparedPublication | None = None
        self._pending_decision: PublicationDecision | None = None
        self._consumed_updates: set[str] = set()

    def prepare(
        self,
        candidate: PublicationCandidate,
        votes: tuple[RankPrepare, ...],
    ) -> PreparedPublication:
        if self.restart_required:
            raise RuntimeError("process group restart is required before preparation")
        if self._pending is not None or self._pending_decision is not None:
            raise RuntimeError("a publication decision is already pending")
        expected_ranks = tuple(range(self.topology.world_size))
        vote_ranks = [vote.rank for vote in votes]
        if len(vote_ranks) != len(set(vote_ranks)):
            raise ValueError("duplicate rank prepare vote")

        process_failures: list[str] = []
        missing = sorted(set(expected_ranks) - set(vote_ranks))
        unexpected = sorted(set(vote_ranks) - set(expected_ranks))
        if missing:
            process_failures.append(f"missing_ranks:{','.join(map(str, missing))}")
        if unexpected:
            process_failures.append(
                f"unexpected_ranks:{','.join(map(str, unexpected))}"
            )

        invalid: list[str] = []
        if candidate.update.sha256 in self._consumed_updates:
            invalid.append("duplicate_update_identity")
        for vote in sorted(votes, key=lambda item: item.rank):
            if vote.rank not in expected_ranks:
                continue
            topology_receipt = self.topology.receipt_for_rank(vote.rank)
            if vote.topology_receipt_sha256 != topology_receipt.sha256:
                process_failures.append(f"rank_{vote.rank}:topology_receipt_mismatch")
            if not vote.process_group_healthy:
                process_failures.append(f"rank_{vote.rank}:process_group_failed")
            checks = (
                (vote.candidate_sha256 == candidate.sha256, "candidate_identity"),
                (
                    vote.source_version == candidate.update.source_version,
                    "source_version",
                ),
                (vote.cohort_epoch == candidate.update.cohort_epoch, "cohort_epoch"),
                (
                    vote.buffer_generation == candidate.buffer_generation,
                    "buffer_generation",
                ),
                (
                    vote.optimizer_generation == candidate.optimizer_generation,
                    "optimizer_generation",
                ),
                (vote.ready, "readiness"),
                (vote.finite, "finiteness"),
                (vote.memory_reserved, "memory_reservation"),
                (vote.safe_boundary, "safe_boundary"),
            )
            invalid.extend(
                f"rank_{vote.rank}:{reason}" for passed, reason in checks if not passed
            )

        if process_failures:
            disposition = PrepareDisposition.PROCESS_GROUP_FAILURE
            reasons = tuple(sorted(set(process_failures)))
        elif invalid:
            disposition = PrepareDisposition.ABORT_STATIC
            reasons = tuple(sorted(set(invalid)))
        else:
            disposition = PrepareDisposition.COMMIT_READY
            reasons = ("all_ranks_prepared",)
        prepared = PreparedPublication(
            update_sha256=candidate.update.sha256,
            candidate_sha256=candidate.sha256,
            topology_sha256=self.topology.topology_sha256,
            disposition=disposition,
            reasons=reasons,
            ranks=expected_ranks,
        )
        self._pending = prepared
        return prepared

    def decide(self, prepared: PreparedPublication) -> PublicationDecision:
        if self._pending is None or prepared.sha256 != self._pending.sha256:
            raise RuntimeError("prepared publication is not the active collective")
        outcome = {
            PrepareDisposition.COMMIT_READY: PublicationOutcome.COMMIT,
            PrepareDisposition.ABORT_STATIC: PublicationOutcome.ABORT_STATIC,
            PrepareDisposition.PROCESS_GROUP_FAILURE: (
                PublicationOutcome.PROCESS_GROUP_FAILURE
            ),
        }[prepared.disposition]
        process_failed = outcome is PublicationOutcome.PROCESS_GROUP_FAILURE
        commit_in_progress = outcome is PublicationOutcome.COMMIT
        decision = PublicationDecision(
            update_sha256=prepared.update_sha256,
            candidate_sha256=prepared.candidate_sha256,
            topology_sha256=prepared.topology_sha256,
            outcome=outcome,
            reasons=prepared.reasons,
            ranks=prepared.ranks,
            service_ready=not process_failed and not commit_in_progress,
            admission_allowed=not process_failed and not commit_in_progress,
            restart_required=process_failed,
        )
        self.service_ready = decision.service_ready
        self.admission_allowed = decision.admission_allowed
        self.restart_required = decision.restart_required
        self._pending = None
        if process_failed:
            self._pending_decision = None
            self._consumed_updates.add(prepared.update_sha256)
        else:
            self._pending_decision = decision
        return decision

    def finalize(
        self,
        decision: PublicationDecision,
        receipts: tuple[RankDecisionReceipt, ...],
    ) -> None:
        """Open admission only after every rank receipts the same copy outcome."""

        if (
            self._pending_decision is None
            or decision.sha256 != self._pending_decision.sha256
        ):
            raise RuntimeError("publication decision is not awaiting finalization")
        try:
            validate_decision_receipts(decision, receipts, self.topology)
        except ValueError:
            self.mark_collective_failed()
            raise
        self._consumed_updates.add(decision.update_sha256)
        self._pending_decision = None
        self.service_ready = True
        self.admission_allowed = True
        self.restart_required = False

    def mark_process_group_restarted(self, topology: TopologyReceiptSet) -> None:
        if not self.restart_required:
            raise RuntimeError("no process-group restart is pending")
        if topology.topology_sha256 != self.topology.topology_sha256:
            raise ValueError("restart topology differs from the failed topology")
        self.topology = topology
        self.service_ready = True
        self.admission_allowed = True
        self.restart_required = False
        self._pending_decision = None

    def mark_collective_failed(self) -> None:
        """Fail service readiness after a transport exception or split decision."""
        self._pending = None
        self._pending_decision = None
        self.service_ready = False
        self.admission_allowed = False
        self.restart_required = True


class GlooPublicationTransport:
    """Real CPU collective harness for the all-rank publication state machine.

    This class intentionally rejects NCCL. It validates process-group behavior without
    pretending to test CUDA streams, graph boundaries, fixed-address copies, or NCCL.
    """

    def __init__(
        self,
        topology: TopologyReceiptSet,
        *,
        local_rank: int,
        process_group: object | None = None,
    ) -> None:
        _require_counter("local_rank", local_rank)
        if local_rank >= topology.world_size:
            raise ValueError("local rank is outside the topology")
        self.topology = topology
        self.local_rank = local_rank
        self.process_group = process_group
        self.coordinator = AllRankPublicationCoordinator(topology)

    def _distributed(self) -> object:
        from torch import distributed

        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeError("a live gloo process group is required")
        if distributed.get_backend(self.process_group) != "gloo":
            raise RuntimeError("CPU publication harness requires the gloo backend")
        if distributed.get_world_size(self.process_group) != self.topology.world_size:
            raise RuntimeError("process-group world size differs from topology receipts")
        if distributed.get_rank(self.process_group) != self.local_rank:
            raise RuntimeError("process-group rank differs from the local topology rank")
        return distributed

    def prepare_and_decide(
        self,
        candidate: PublicationCandidate,
        local_vote: RankPrepare,
    ) -> PublicationDecision:
        if local_vote.rank != self.local_rank:
            raise ValueError("local prepare vote belongs to another rank")
        distributed = self._distributed()
        gathered: list[RankPrepare | None] = [None] * self.topology.world_size
        try:
            distributed.all_gather_object(
                gathered,
                local_vote,
                group=self.process_group,
            )
            if not all(isinstance(vote, RankPrepare) for vote in gathered):
                raise RuntimeError("collective returned an invalid prepare vote")
            prepared = self.coordinator.prepare(candidate, tuple(gathered))
            decision = self.coordinator.decide(prepared)
            decisions: list[str | None] = [None] * self.topology.world_size
            distributed.all_gather_object(
                decisions,
                decision.sha256,
                group=self.process_group,
            )
            if decisions != [decision.sha256] * self.topology.world_size:
                raise RuntimeError("ranks derived different publication decisions")
            return decision
        except Exception:
            self.coordinator.mark_collective_failed()
            raise

    def finalize(
        self,
        decision: PublicationDecision,
        *,
        applied: bool,
    ) -> tuple[RankDecisionReceipt, ...]:
        """Gather post-copy receipts; partial application makes service unready."""
        distributed = self._distributed()
        local = RankDecisionReceipt(
            rank=self.local_rank,
            topology_receipt_sha256=(
                self.topology.receipt_for_rank(self.local_rank).sha256
            ),
            decision_sha256=decision.sha256,
            applied=applied,
        )
        gathered: list[RankDecisionReceipt | None] = [None] * self.topology.world_size
        try:
            distributed.all_gather_object(
                gathered,
                local,
                group=self.process_group,
            )
            if not all(isinstance(row, RankDecisionReceipt) for row in gathered):
                raise RuntimeError("collective returned an invalid decision receipt")
            receipts = tuple(gathered)
            self.coordinator.finalize(decision, receipts)
            return receipts
        except Exception:
            self.coordinator.mark_collective_failed()
            raise
