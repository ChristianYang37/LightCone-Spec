"""Pure HBM admission and bounded cohort-state governance contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum


def _require_bytes(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer byte count")


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_time(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative time")


def _require_hash(name: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class HBMLedger:
    """Logical prediction plus separately reported allocator/NVML observations."""

    target_weights_bytes: int = 0
    drafter_weights_bytes: int = 0
    target_kv_bytes: int = 0
    drafter_kv_bytes: int = 0
    active_merged_parameters_bytes: int = 0
    fp32_masters_bytes: int = 0
    gradients_bytes: int = 0
    optimizer_tensors_bytes: int = 0
    candidate_bytes: int = 0
    staging_bytes: int = 0
    merge_scratch_bytes: int = 0
    differentiable_activations_bytes: int = 0
    graph_private_pools_bytes: int = 0
    library_workspace_bytes: int = 0
    nccl_buffers_bytes: int = 0
    kv_gather_scratch_bytes: int = 0
    backend_scratch_bytes: int = 0
    telemetry_staging_bytes: int = 0
    fragmentation_margin_bytes: int = 0
    allocator_allocated_peak_bytes: int = 0
    allocator_reserved_peak_bytes: int = 0
    nvml_process_peak_bytes: int = 0
    nvml_global_peak_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _require_bytes(name, value)
        if self.allocator_reserved_peak_bytes < self.allocator_allocated_peak_bytes:
            raise ValueError("allocator reserved peak cannot be below allocated peak")
        if self.nvml_global_peak_bytes < self.nvml_process_peak_bytes:
            raise ValueError("NVML global use cannot be below this process use")

    @property
    def predicted_resident_bytes(self) -> int:
        return sum(
            (
                self.target_weights_bytes,
                self.drafter_weights_bytes,
                self.target_kv_bytes,
                self.drafter_kv_bytes,
                self.active_merged_parameters_bytes,
                self.fp32_masters_bytes,
                self.optimizer_tensors_bytes,
                self.candidate_bytes,
                self.staging_bytes,
                self.graph_private_pools_bytes,
                self.library_workspace_bytes,
                self.nccl_buffers_bytes,
                self.telemetry_staging_bytes,
            )
        )

    @property
    def predicted_peak_bytes(self) -> int:
        return self.predicted_resident_bytes + sum(
            (
                self.gradients_bytes,
                self.merge_scratch_bytes,
                self.differentiable_activations_bytes,
                self.kv_gather_scratch_bytes,
                self.backend_scratch_bytes,
                self.fragmentation_margin_bytes,
            )
        )

    @property
    def observed_process_peak_bytes(self) -> int:
        return max(
            self.allocator_allocated_peak_bytes,
            self.allocator_reserved_peak_bytes,
            self.nvml_process_peak_bytes,
        )

    @property
    def prediction_error_bytes(self) -> int:
        return self.observed_process_peak_bytes - self.predicted_peak_bytes

    @property
    def adaptation_resident_bytes(self) -> int:
        return sum(
            (
                self.active_merged_parameters_bytes,
                self.fp32_masters_bytes,
                self.optimizer_tensors_bytes,
                self.candidate_bytes,
                self.staging_bytes,
                self.telemetry_staging_bytes,
            )
        )


@dataclass(frozen=True)
class RankMemoryState:
    rank: int
    capacity_bytes: int
    ledger: HBMLedger

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        _require_bytes("capacity_bytes", self.capacity_bytes)
        if self.capacity_bytes == 0:
            raise ValueError("rank HBM capacity must be positive")
        if self.ledger.nvml_global_peak_bytes > self.capacity_bytes:
            raise ValueError("observed global HBM use exceeds device capacity")

    @property
    def charged_peak_bytes(self) -> int:
        return max(
            self.ledger.predicted_peak_bytes,
            self.ledger.observed_process_peak_bytes,
            self.ledger.nvml_global_peak_bytes,
        )


@dataclass(frozen=True)
class HBMAdmissionRequest:
    adaptation_reserve_bytes: int
    target_kv_bytes: int
    drafter_kv_bytes: int
    safety_margin_bytes: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _require_bytes(name, value)

    @property
    def kv_bytes(self) -> int:
        return self.target_kv_bytes + self.drafter_kv_bytes


class HBMAdmissionReason(str, Enum):
    ADMITTED = "admitted"
    ADAPTATION_RESERVE_EXCEEDS_LEAST_RANK = "adaptation_reserve_exceeds_least_rank"
    KV_ADMISSION_EXCEEDS_LEAST_RANK = "kv_admission_exceeds_least_rank"


@dataclass(frozen=True)
class RankHeadroom:
    rank: int
    charged_peak_bytes: int
    after_adaptation_bytes: int
    projected_peak_bytes: int
    usable_limit_bytes: int
    headroom_bytes: int


@dataclass(frozen=True)
class HBMAdmission:
    admitted: bool
    reason: HBMAdmissionReason
    limiting_rank: int
    ranks: tuple[RankHeadroom, ...]


class MemoryPressureAction(IntEnum):
    PRESERVE_IMMUTABLE_AND_CORRECTNESS = 1
    PRESERVE_ACTIVE_KV_AND_PUBLISHED_STATE = 2
    EVICT_NATIVE_INACTIVE_PREFIX = 3
    ABORT_PENDING_ADAPTATION = 4
    QUEUE_OR_REJECT_NEW_WORK = 5
    OFFLOAD_COLD_INACTIVE_COHORT = 6


@dataclass(frozen=True)
class MemoryPressureStep:
    action: MemoryPressureAction
    reason_code: str


class HBMGovernor:
    """All-rank HBM admission governed by the least feasible rank."""

    def __init__(
        self,
        states: tuple[RankMemoryState, ...],
        *,
        expected_ranks: int,
    ) -> None:
        _require_positive_int("expected_ranks", expected_ranks)
        ranks = [state.rank for state in states]
        if len(ranks) != len(set(ranks)):
            raise ValueError("duplicate rank memory state")
        if set(ranks) != set(range(expected_ranks)):
            raise ValueError("memory state lacks exact all-rank coverage")
        self.states = tuple(sorted(states, key=lambda state: state.rank))

    def assess(self, request: HBMAdmissionRequest) -> HBMAdmission:
        rows: list[RankHeadroom] = []
        adaptation_fits = True
        for state in self.states:
            usable = state.capacity_bytes - request.safety_margin_bytes
            after_adaptation = (
                state.charged_peak_bytes + request.adaptation_reserve_bytes
            )
            projected = after_adaptation + request.kv_bytes
            adaptation_fits = adaptation_fits and after_adaptation <= usable
            rows.append(
                RankHeadroom(
                    rank=state.rank,
                    charged_peak_bytes=state.charged_peak_bytes,
                    after_adaptation_bytes=after_adaptation,
                    projected_peak_bytes=projected,
                    usable_limit_bytes=usable,
                    headroom_bytes=usable - projected,
                )
            )
        limiting = min(rows, key=lambda row: (row.headroom_bytes, row.rank))
        admitted = limiting.headroom_bytes >= 0
        if admitted:
            reason = HBMAdmissionReason.ADMITTED
        elif not adaptation_fits:
            reason = HBMAdmissionReason.ADAPTATION_RESERVE_EXCEEDS_LEAST_RANK
        else:
            reason = HBMAdmissionReason.KV_ADMISSION_EXCEEDS_LEAST_RANK
        return HBMAdmission(
            admitted=admitted,
            reason=reason,
            limiting_rank=limiting.rank,
            ranks=tuple(rows),
        )

    @staticmethod
    def pressure_plan(
        *, allow_cold_offload: bool = False
    ) -> tuple[MemoryPressureStep, ...]:
        steps = [
            MemoryPressureStep(
                MemoryPressureAction.PRESERVE_IMMUTABLE_AND_CORRECTNESS,
                "preserve_immutable_runtime_and_active_correctness",
            ),
            MemoryPressureStep(
                MemoryPressureAction.PRESERVE_ACTIVE_KV_AND_PUBLISHED_STATE,
                "preserve_active_kv_and_published_cohort",
            ),
            MemoryPressureStep(
                MemoryPressureAction.EVICT_NATIVE_INACTIVE_PREFIX,
                "evict_native_inactive_prefix_only",
            ),
            MemoryPressureStep(
                MemoryPressureAction.ABORT_PENDING_ADAPTATION,
                "abort_pending_adaptation_release_temporary_state",
            ),
            MemoryPressureStep(
                MemoryPressureAction.QUEUE_OR_REJECT_NEW_WORK,
                "queue_or_reject_new_work",
            ),
        ]
        if allow_cold_offload:
            steps.append(
                MemoryPressureStep(
                    MemoryPressureAction.OFFLOAD_COLD_INACTIVE_COHORT,
                    "cold_inactive_timed_offload",
                )
            )
        return tuple(steps)


class CohortOffloadMode(str, Enum):
    DISABLED = "disabled"
    COLD_INACTIVE_TIMED = "cold_inactive_timed"


@dataclass(frozen=True)
class CohortStateKey:
    tenant_id: str
    cohort_sha256: str
    replica_id: int

    def __post_init__(self) -> None:
        if not self.tenant_id or self.tenant_id.strip() != self.tenant_id:
            raise ValueError("tenant_id must be a non-empty canonical identifier")
        _require_hash("cohort_sha256", self.cohort_sha256)
        if type(self.replica_id) is not int or self.replica_id < 0:
            raise ValueError("replica_id must be a non-negative integer")


@dataclass(frozen=True)
class CohortStateSnapshot:
    key: CohortStateKey
    slab_id: int | None
    slab_bytes: int
    version: int
    created_at: float
    last_used_at: float
    expires_at: float
    active_requests: int
    offloaded: bool


@dataclass
class _CohortState:
    key: CohortStateKey
    slab_id: int | None
    slab_bytes: int
    version: int
    created_at: float
    last_used_at: float
    active_requests: int = 0
    offloaded: bool = False


class CohortAdmissionReason(str, Enum):
    ADMITTED = "admitted"
    ALREADY_RESIDENT = "already_resident"
    UNKNOWN_TENANT = "unknown_tenant"
    TENANT_QUOTA = "tenant_quota"
    CAPACITY = "capacity"
    OFFLOADED_RESTORE_REQUIRED = "offloaded_restore_required"


@dataclass(frozen=True)
class CohortAdmission:
    admitted: bool
    reason: CohortAdmissionReason
    state: CohortStateSnapshot | None


@dataclass(frozen=True)
class CohortReclamationReceipt:
    key: CohortStateKey
    slab_id: int | None
    reason_code: str
    reclaimed_at: float


@dataclass(frozen=True)
class CohortTransferReceipt:
    key: CohortStateKey
    operation: str
    slab_id: int
    bytes_transferred: int
    started_at: float
    completed_at: float
    mode: CohortOffloadMode


class BoundedCohortStateManager:
    """Fixed-slab, quota-bound, privacy-isolated cohort metadata manager."""

    def __init__(
        self,
        *,
        capacity: int,
        slab_bytes: int,
        tenant_quotas: Mapping[str, int],
        ttl_seconds: float,
        offload_mode: CohortOffloadMode = CohortOffloadMode.DISABLED,
        offloaded_capacity: int = 0,
    ) -> None:
        _require_positive_int("capacity", capacity)
        _require_bytes("slab_bytes", slab_bytes)
        if slab_bytes == 0:
            raise ValueError("slab_bytes must be positive")
        _require_time("ttl_seconds", ttl_seconds)
        if ttl_seconds == 0:
            raise ValueError("ttl_seconds must be positive")
        if not tenant_quotas:
            raise ValueError("at least one tenant quota is required")
        _require_bytes("offloaded_capacity", offloaded_capacity)
        if offload_mode is CohortOffloadMode.DISABLED and offloaded_capacity != 0:
            raise ValueError("disabled offload mode cannot reserve a host tier")
        if (
            offload_mode is CohortOffloadMode.COLD_INACTIVE_TIMED
            and offloaded_capacity == 0
        ):
            raise ValueError("cold offload requires an explicit bounded host tier")
        quotas: dict[str, int] = {}
        for tenant, quota in tenant_quotas.items():
            if not tenant or tenant.strip() != tenant:
                raise ValueError("tenant quota identity is not canonical")
            _require_positive_int("tenant quota", quota)
            if quota > capacity:
                raise ValueError("tenant quota cannot exceed bounded K capacity")
            quotas[tenant] = quota
        self.capacity = capacity
        self.slab_bytes = slab_bytes
        self.tenant_quotas = quotas
        self.ttl_seconds = ttl_seconds
        self.offload_mode = offload_mode
        self.offloaded_capacity = offloaded_capacity
        self._free_slabs = set(range(capacity))
        self._states: dict[CohortStateKey, _CohortState] = {}

    @property
    def resident_count(self) -> int:
        return sum(state.slab_id is not None for state in self._states.values())

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def offloaded_count(self) -> int:
        return sum(state.offloaded for state in self._states.values())

    def _snapshot(self, state: _CohortState) -> CohortStateSnapshot:
        return CohortStateSnapshot(
            key=state.key,
            slab_id=state.slab_id,
            slab_bytes=state.slab_bytes,
            version=state.version,
            created_at=state.created_at,
            last_used_at=state.last_used_at,
            expires_at=state.last_used_at + self.ttl_seconds,
            active_requests=state.active_requests,
            offloaded=state.offloaded,
        )

    def snapshot(self, key: CohortStateKey) -> CohortStateSnapshot | None:
        state = self._states.get(key)
        return None if state is None else self._snapshot(state)

    def admit(
        self,
        key: CohortStateKey,
        *,
        now: float,
        version: int = 0,
    ) -> CohortAdmission:
        _require_time("now", now)
        if type(version) is not int or version < 0:
            raise ValueError("version must be a non-negative integer")
        existing = self._states.get(key)
        if existing is not None:
            if existing.offloaded:
                return CohortAdmission(
                    False,
                    CohortAdmissionReason.OFFLOADED_RESTORE_REQUIRED,
                    self._snapshot(existing),
                )
            existing.last_used_at = max(existing.last_used_at, now)
            return CohortAdmission(
                True,
                CohortAdmissionReason.ALREADY_RESIDENT,
                self._snapshot(existing),
            )
        quota = self.tenant_quotas.get(key.tenant_id)
        if quota is None:
            return CohortAdmission(False, CohortAdmissionReason.UNKNOWN_TENANT, None)
        tenant_resident_count = sum(
            state.key.tenant_id == key.tenant_id and not state.offloaded
            for state in self._states.values()
        )
        if tenant_resident_count >= quota:
            return CohortAdmission(False, CohortAdmissionReason.TENANT_QUOTA, None)
        if (
            len(self._states) >= self.capacity + self.offloaded_capacity
            or self.resident_count >= self.capacity
            or not self._free_slabs
        ):
            return CohortAdmission(False, CohortAdmissionReason.CAPACITY, None)
        slab = min(self._free_slabs)
        self._free_slabs.remove(slab)
        state = _CohortState(
            key=key,
            slab_id=slab,
            slab_bytes=self.slab_bytes,
            version=version,
            created_at=now,
            last_used_at=now,
        )
        self._states[key] = state
        return CohortAdmission(
            True,
            CohortAdmissionReason.ADMITTED,
            self._snapshot(state),
        )

    def acquire(self, key: CohortStateKey, *, now: float) -> CohortStateSnapshot:
        _require_time("now", now)
        state = self._states.get(key)
        if state is None:
            raise KeyError("cohort state does not exist for this tenant identity")
        if state.offloaded or state.slab_id is None:
            raise RuntimeError("cold cohort state must be restored before acquisition")
        state.active_requests += 1
        state.last_used_at = max(state.last_used_at, now)
        return self._snapshot(state)

    def release(self, key: CohortStateKey, *, now: float) -> CohortStateSnapshot:
        _require_time("now", now)
        state = self._states.get(key)
        if state is None:
            raise KeyError("cohort state does not exist for this tenant identity")
        if state.active_requests == 0:
            raise RuntimeError("cohort state has no active request to release")
        state.active_requests -= 1
        state.last_used_at = max(state.last_used_at, now)
        return self._snapshot(state)

    def publish_version(
        self,
        key: CohortStateKey,
        *,
        source_version: int,
        new_version: int,
        now: float,
    ) -> CohortStateSnapshot:
        _require_time("now", now)
        state = self._states.get(key)
        if state is None or state.offloaded:
            raise RuntimeError("only resident cohort state can publish")
        if state.version != source_version or new_version != source_version + 1:
            raise ValueError("cohort publication version conflict")
        state.version = new_version
        state.last_used_at = max(state.last_used_at, now)
        return self._snapshot(state)

    def reclaim(
        self,
        key: CohortStateKey,
        *,
        now: float,
        reason_code: str,
    ) -> CohortReclamationReceipt:
        _require_time("now", now)
        if not reason_code or reason_code.strip() != reason_code:
            raise ValueError("reclamation requires a canonical reason code")
        state = self._states.get(key)
        if state is None:
            raise KeyError("cohort state does not exist for this tenant identity")
        if state.active_requests:
            raise RuntimeError("active cohort state cannot be reclaimed")
        if state.slab_id is not None:
            self._free_slabs.add(state.slab_id)
        del self._states[key]
        return CohortReclamationReceipt(
            key=key,
            slab_id=state.slab_id,
            reason_code=reason_code,
            reclaimed_at=now,
        )

    def reclaim_expired(
        self,
        *,
        now: float,
    ) -> tuple[CohortReclamationReceipt, ...]:
        _require_time("now", now)
        expired = sorted(
            (
                state
                for state in self._states.values()
                if state.active_requests == 0
                and now >= state.last_used_at + self.ttl_seconds
            ),
            key=lambda state: (
                state.last_used_at,
                state.key.tenant_id,
                state.key.cohort_sha256,
                state.key.replica_id,
            ),
        )
        return tuple(
            self.reclaim(state.key, now=now, reason_code="ttl_expired")
            for state in expired
        )

    def reclaim_lru(
        self,
        *,
        count: int,
        now: float,
    ) -> tuple[CohortReclamationReceipt, ...]:
        _require_positive_int("count", count)
        _require_time("now", now)
        candidates = sorted(
            (state for state in self._states.values() if state.active_requests == 0),
            key=lambda state: (
                state.last_used_at,
                state.created_at,
                state.key.tenant_id,
                state.key.cohort_sha256,
                state.key.replica_id,
            ),
        )
        return tuple(
            self.reclaim(state.key, now=now, reason_code="lru_reclaimed")
            for state in candidates[:count]
        )

    def offload_cold(
        self,
        key: CohortStateKey,
        *,
        started_at: float,
        completed_at: float,
    ) -> CohortTransferReceipt:
        if self.offload_mode is not CohortOffloadMode.COLD_INACTIVE_TIMED:
            raise RuntimeError("cold offload mode is not explicitly enabled")
        _require_time("started_at", started_at)
        _require_time("completed_at", completed_at)
        if completed_at < started_at:
            raise ValueError("offload completion precedes its start")
        state = self._states.get(key)
        if state is None:
            raise KeyError("cohort state does not exist for this tenant identity")
        if state.active_requests:
            raise RuntimeError("active cohort state cannot be offloaded")
        if state.slab_id is None or state.offloaded:
            raise RuntimeError("cohort state is already offloaded")
        if self.offloaded_count >= self.offloaded_capacity:
            raise MemoryError("bounded cold-host cohort tier is full")
        slab = state.slab_id
        self._free_slabs.add(slab)
        state.slab_id = None
        state.offloaded = True
        state.last_used_at = max(state.last_used_at, completed_at)
        return CohortTransferReceipt(
            key=key,
            operation="cold_offload",
            slab_id=slab,
            bytes_transferred=state.slab_bytes,
            started_at=started_at,
            completed_at=completed_at,
            mode=self.offload_mode,
        )

    def restore_cold(
        self,
        key: CohortStateKey,
        *,
        started_at: float,
        completed_at: float,
    ) -> CohortTransferReceipt:
        if self.offload_mode is not CohortOffloadMode.COLD_INACTIVE_TIMED:
            raise RuntimeError("cold offload mode is not explicitly enabled")
        _require_time("started_at", started_at)
        _require_time("completed_at", completed_at)
        if completed_at < started_at:
            raise ValueError("restore completion precedes its start")
        state = self._states.get(key)
        if state is None:
            raise KeyError("cohort state does not exist for this tenant identity")
        if not state.offloaded or state.slab_id is not None:
            raise RuntimeError("cohort state is already resident")
        if not self._free_slabs:
            raise MemoryError("no fixed cohort slab is available for restore")
        slab = min(self._free_slabs)
        self._free_slabs.remove(slab)
        state.slab_id = slab
        state.offloaded = False
        state.last_used_at = max(state.last_used_at, completed_at)
        return CohortTransferReceipt(
            key=key,
            operation="cold_restore",
            slab_id=slab,
            bytes_transferred=state.slab_bytes,
            started_at=started_at,
            completed_at=completed_at,
            mode=self.offload_mode,
        )
