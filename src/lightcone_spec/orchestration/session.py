"""Clean, block-scoped industrial server-session contracts.

Session reuse is an execution optimization only.  Every member keeps its own
execution-plan identity, run nonce, writer, terminal evidence, and statistical
unit.  A reset receipt must prove that server state returned to the exact clean
state attested when the process opened; otherwise the process cannot be reused.

The current release exposes the immutable planning and receipt data contracts
for audit and now has source-owned HTTP connection accounting for reset-state
receipts on the supported single-tokenizer HTTP/1.1 uvicorn paths.  Other HTTP
server topologies fail closed before producing that capability, and every live
shared-session mutation remains blocked.  GPU reset semantics, a durable
session-receipt bundle, and exact continuous whole-inventory authority are not
yet available.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from lightcone_spec.experiments.serving import BenchServingTransport
    from lightcone_spec.orchestration.executor import (
        IndustrialExecutionPlan,
        IndustrialExecutionResult,
        ServerLauncher,
    )
    from lightcone_spec.orchestration.native_terminal import NativeTerminalProvider

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")

SHARED_SESSION_UNAVAILABLE_REASON = (
    "shared_session_gpu_reset_and_durable_terminal_authority_unavailable"
)
SHARED_SESSION_FALLBACK_MODE = "fresh_process_per_trace"


class SharedSessionUnavailableError(RuntimeError):
    """Current release cannot safely mutate or claim a reused server session."""


def _raise_shared_session_unavailable() -> None:
    raise SharedSessionUnavailableError(SHARED_SESSION_UNAVAILABLE_REASON)


def _content_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or "\n" in value:
        raise ValueError(f"{name} must be non-empty single-line text")


class _ExecutionPlan(Protocol):
    patched_sglang_tree: str
    runtime_plan: object
    server_launch: object

    def validate(self) -> None: ...

    @property
    def sha256(self) -> str: ...

    @property
    def rank_config_sha256(self) -> str: ...

    @property
    def topology_sha256(self) -> str: ...


@dataclass(frozen=True)
class IndustrialServerSessionKey:
    """Complete process-affecting identity for a reusable server."""

    patched_sglang_tree: str
    capability_receipt_sha256: str
    rank_config_sha256: str
    model_lock_sha256: str
    parameter_plan_sha256: str | None
    topology_sha256: str
    gpu_uuids: tuple[str, ...]
    method: str
    backend: str
    dtype: str
    precision: str
    context_limit: int
    graph_buckets: tuple[int, ...]
    max_running_requests: int
    memory_fraction: str
    hbm_reservation_bytes: int
    telemetry_mode: str
    compile_cache_receipt_sha256: str
    port_router_sha256: str
    server_argv_sha256: str

    def validate(self) -> None:
        if not _GIT_OBJECT.fullmatch(self.patched_sglang_tree):
            raise ValueError("patched SGLang tree must be a lowercase Git object")
        for name in (
            "capability_receipt_sha256",
            "rank_config_sha256",
            "model_lock_sha256",
            "topology_sha256",
            "compile_cache_receipt_sha256",
            "port_router_sha256",
            "server_argv_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.parameter_plan_sha256 is not None:
            _require_sha256("parameter_plan_sha256", self.parameter_plan_sha256)
        if not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("session key requires unique GPU UUIDs")
        for index, gpu_uuid in enumerate(self.gpu_uuids):
            _require_text(f"gpu_uuids[{index}]", gpu_uuid)
        for name in (
            "method",
            "backend",
            "dtype",
            "precision",
            "memory_fraction",
            "telemetry_mode",
        ):
            _require_text(name, getattr(self, name))
        for name in ("context_limit", "max_running_requests"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.hbm_reservation_bytes, bool)
            or self.hbm_reservation_bytes < 0
        ):
            raise ValueError("HBM reservation must be a non-negative integer")
        if tuple(sorted(set(self.graph_buckets))) != self.graph_buckets or any(
            isinstance(bucket, bool) or not isinstance(bucket, int) or bucket < 1
            for bucket in self.graph_buckets
        ):
            raise ValueError("graph buckets must be unique increasing integers")

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256({"schema_version": 1, **asdict(self)})

    @classmethod
    def from_execution_plan(
        cls,
        plan: _ExecutionPlan,
        *,
        capability_receipt_sha256: str,
        compile_cache_receipt_sha256: str,
        dtype: str,
        precision: str,
        graph_buckets: tuple[int, ...],
        hbm_reservation_bytes: int,
    ) -> Self:
        plan.validate()
        runtime_plan = plan.runtime_plan
        rank_configs = runtime_plan.rank_configs
        if len(rank_configs) != 1:
            raise ValueError("current session executor requires one rank config")
        config = rank_configs[0]
        if not runtime_plan.physical_dispatch_ready:
            raise ValueError(
                "logical runtime plans cannot create a live server-session key"
            )
        physical_gpu_uuids = tuple(runtime_plan.physical_gpu_uuids)
        physical_rank_groups = tuple(runtime_plan.physical_rank_groups)
        physical_ports = tuple(runtime_plan.physical_ports)
        launch = plan.server_launch
        argv = tuple(launch.argv)
        try:
            memory_index = argv.index("--mem-fraction-static") + 1
            memory_fraction = argv[memory_index]
        except (ValueError, IndexError) as error:
            raise ValueError(
                "server argv lacks the memory-fraction identity"
            ) from error
        value = cls(
            patched_sglang_tree=plan.patched_sglang_tree,
            capability_receipt_sha256=capability_receipt_sha256,
            rank_config_sha256=plan.rank_config_sha256,
            model_lock_sha256=plan.model_lock_artifact.content_sha256,
            parameter_plan_sha256=runtime_plan.parameter_plan_sha256,
            topology_sha256=plan.topology_sha256,
            gpu_uuids=physical_gpu_uuids,
            method=config.method,
            backend=config.model.algorithm,
            dtype=dtype,
            precision=precision,
            context_limit=config.model.max_context_length,
            graph_buckets=graph_buckets,
            max_running_requests=config.runtime.max_running_requests,
            memory_fraction=memory_fraction,
            hbm_reservation_bytes=hbm_reservation_bytes,
            telemetry_mode=config.runtime.telemetry_detail,
            compile_cache_receipt_sha256=compile_cache_receipt_sha256,
            port_router_sha256=_content_sha256(
                {
                    "physical_ports": list(physical_ports),
                    "physical_rank_groups": [
                        list(group) for group in physical_rank_groups
                    ],
                    "physical_assignment_sha256": (
                        runtime_plan.physical_assignment.assignment_sha256
                    ),
                    "router": config.runtime.router_identity,
                    "base_url": launch.base_url,
                }
            ),
            server_argv_sha256=_content_sha256(list(argv)),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class IndustrialServerSessionPlan:
    """Ordered logical traces allowed to reuse exactly one server process."""

    session_key: IndustrialServerSessionKey
    execution_plan_sha256s: tuple[str, ...]
    method: str
    block: int
    fault_injection: bool

    def validate(self) -> None:
        self.session_key.validate()
        if not self.execution_plan_sha256s or len(
            set(self.execution_plan_sha256s)
        ) != len(self.execution_plan_sha256s):
            raise ValueError("session members must be non-empty and unique")
        for value in self.execution_plan_sha256s:
            _require_sha256("execution plan", value)
        if self.method != self.session_key.method:
            raise ValueError("session method differs from its process key")
        if isinstance(self.block, bool) or self.block < 0:
            raise ValueError("session block must be non-negative")
        if self.fault_injection and len(self.execution_plan_sha256s) != 1:
            raise ValueError("fault injection requires a fresh one-trace process")

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256(
            {
                "schema_version": 1,
                "session_key_sha256": self.session_key.sha256,
                "execution_plan_sha256s": list(self.execution_plan_sha256s),
                "method": self.method,
                "block": self.block,
                "fault_injection": self.fault_injection,
            }
        )

    @classmethod
    def create(
        cls,
        plans: tuple[_ExecutionPlan, ...],
        *,
        capability_receipt_sha256: str,
        compile_cache_receipt_sha256: str,
        dtype: str,
        precision: str,
        graph_buckets: tuple[int, ...],
        hbm_reservation_bytes: int,
    ) -> Self:
        if not plans:
            raise ValueError("a server session requires at least one execution plan")
        keys = tuple(
            IndustrialServerSessionKey.from_execution_plan(
                plan,
                capability_receipt_sha256=capability_receipt_sha256,
                compile_cache_receipt_sha256=compile_cache_receipt_sha256,
                dtype=dtype,
                precision=precision,
                graph_buckets=graph_buckets,
                hbm_reservation_bytes=hbm_reservation_bytes,
            )
            for plan in plans
        )
        if any(key != keys[0] for key in keys[1:]):
            raise ValueError("adjacent execution plans do not share one session key")
        cells = tuple(plan.runtime_plan.cell for plan in plans)
        methods = {cell.identity.method for cell in cells}
        blocks = {cell.identity.block for cell in cells}
        if len(methods) != 1 or len(blocks) != 1:
            raise ValueError("a session cannot cross method or scientific block")
        fault_injection = any(
            cell.identity.task == "failure_injection" for cell in cells
        )
        value = cls(
            session_key=keys[0],
            execution_plan_sha256s=tuple(plan.sha256 for plan in plans),
            method=next(iter(methods)),
            block=next(iter(blocks)),
            fault_injection=fault_injection,
        )
        value.validate()
        return value


@dataclass(frozen=True)
class SessionBoundaryState:
    """Native state required to prove a clean logical-trace boundary."""

    process_identity: str
    session_epoch: int
    reset_generation: int
    active_requests: int
    queued_requests: int
    rng_sha256: str
    inference_weights_sha256: str
    fp32_master_sha256: str | None
    optimizer_state_sha256: str | None
    candidate_buffers_sha256: str | None
    scheduler_state_sha256: str
    kv_state_sha256: str
    telemetry_state_sha256: str
    adapter_version: int
    optimizer_generation: int
    allocator_allocated_bytes: int
    allocator_reserved_bytes: int
    hbm_state_sha256: str
    completion_event_sha256: str

    def validate(self) -> None:
        _require_text("process_identity", self.process_identity)
        for name in (
            "session_epoch",
            "reset_generation",
            "active_requests",
            "queued_requests",
            "adapter_version",
            "optimizer_generation",
            "allocator_allocated_bytes",
            "allocator_reserved_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "rng_sha256",
            "inference_weights_sha256",
            "scheduler_state_sha256",
            "kv_state_sha256",
            "telemetry_state_sha256",
            "hbm_state_sha256",
            "completion_event_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "fp32_master_sha256",
            "optimizer_state_sha256",
            "candidate_buffers_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(name, value)

    @property
    def clean_state_sha256(self) -> str:
        self.validate()
        return _content_sha256(
            {
                "active_requests": self.active_requests,
                "queued_requests": self.queued_requests,
                "rng_sha256": self.rng_sha256,
                "inference_weights_sha256": self.inference_weights_sha256,
                "fp32_master_sha256": self.fp32_master_sha256,
                "optimizer_state_sha256": self.optimizer_state_sha256,
                "candidate_buffers_sha256": self.candidate_buffers_sha256,
                "scheduler_state_sha256": self.scheduler_state_sha256,
                "kv_state_sha256": self.kv_state_sha256,
                "telemetry_state_sha256": self.telemetry_state_sha256,
                "adapter_version": self.adapter_version,
                "optimizer_generation": self.optimizer_generation,
                "allocator_allocated_bytes": self.allocator_allocated_bytes,
                "allocator_reserved_bytes": self.allocator_reserved_bytes,
                "hbm_state_sha256": self.hbm_state_sha256,
            }
        )


@dataclass(frozen=True)
class IndustrialSessionOpenReceipt:
    session_plan_sha256: str
    process_identity: str
    process_started_ns: int
    session_epoch: int
    clean_state_sha256: str
    native_capability_receipt_sha256: str

    def validate(self) -> None:
        for name in (
            "session_plan_sha256",
            "clean_state_sha256",
            "native_capability_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("process_identity", self.process_identity)
        if (
            isinstance(self.process_started_ns, bool)
            or self.process_started_ns < 0
            or isinstance(self.session_epoch, bool)
            or self.session_epoch < 0
        ):
            raise ValueError("session open counters must be non-negative")

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256({"schema_version": 1, **asdict(self)})


@dataclass(frozen=True)
class IndustrialResetReceipt:
    session_plan_sha256: str
    session_open_receipt_sha256: str
    process_identity: str
    session_epoch: int
    reset_generation: int
    prior_execution_plan_sha256: str | None
    next_execution_plan_sha256: str
    before_state_sha256: str
    after_state_sha256: str
    allocator_allocated_bytes: int
    allocator_reserved_bytes: int
    adapter_version: int
    optimizer_generation: int
    active_requests: int
    queued_requests: int
    completion_event_sha256: str
    reset_duration_ms: float

    def validate(
        self,
        *,
        session_plan: IndustrialServerSessionPlan,
        open_receipt: IndustrialSessionOpenReceipt,
    ) -> None:
        session_plan.validate()
        open_receipt.validate()
        if (
            self.session_plan_sha256 != session_plan.sha256
            or self.session_open_receipt_sha256 != open_receipt.sha256
            or self.process_identity != open_receipt.process_identity
            or self.session_epoch != open_receipt.session_epoch
        ):
            raise ValueError("reset receipt belongs to another server session")
        if self.next_execution_plan_sha256 not in session_plan.execution_plan_sha256s:
            raise ValueError("reset receipt names an unregistered next trace")
        next_index = session_plan.execution_plan_sha256s.index(
            self.next_execution_plan_sha256
        )
        expected_prior = (
            None
            if next_index == 0
            else session_plan.execution_plan_sha256s[next_index - 1]
        )
        if self.prior_execution_plan_sha256 != expected_prior:
            raise ValueError("reset receipt breaks the ordered trace chain")
        for name in (
            "session_plan_sha256",
            "session_open_receipt_sha256",
            "next_execution_plan_sha256",
            "before_state_sha256",
            "after_state_sha256",
            "completion_event_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.prior_execution_plan_sha256 is not None:
            _require_sha256(
                "prior_execution_plan_sha256", self.prior_execution_plan_sha256
            )
        for name in (
            "reset_generation",
            "allocator_allocated_bytes",
            "allocator_reserved_bytes",
            "adapter_version",
            "optimizer_generation",
            "active_requests",
            "queued_requests",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.active_requests or self.queued_requests:
            raise ValueError("reset receipt requires zero live and queued requests")
        if self.after_state_sha256 != open_receipt.clean_state_sha256:
            raise ValueError("reset did not restore the attested clean state")
        if not math.isfinite(self.reset_duration_ms) or self.reset_duration_ms < 0:
            raise ValueError("reset duration must be finite and non-negative")

    @property
    def sha256(self) -> str:
        return _content_sha256({"schema_version": 1, **asdict(self)})

    @classmethod
    def create(
        cls,
        *,
        session_plan: IndustrialServerSessionPlan,
        open_receipt: IndustrialSessionOpenReceipt,
        prior_execution_plan_sha256: str | None,
        next_execution_plan_sha256: str,
        before: SessionBoundaryState,
        after: SessionBoundaryState,
        reset_duration_ms: float,
    ) -> Self:
        before.validate()
        after.validate()
        if (
            before.process_identity != after.process_identity
            or after.process_identity != open_receipt.process_identity
            or before.session_epoch != after.session_epoch
            or after.session_epoch != open_receipt.session_epoch
            or after.reset_generation != before.reset_generation + 1
        ):
            raise ValueError("reset boundary process/generation identity is invalid")
        value = cls(
            session_plan_sha256=session_plan.sha256,
            session_open_receipt_sha256=open_receipt.sha256,
            process_identity=after.process_identity,
            session_epoch=after.session_epoch,
            reset_generation=after.reset_generation,
            prior_execution_plan_sha256=prior_execution_plan_sha256,
            next_execution_plan_sha256=next_execution_plan_sha256,
            before_state_sha256=before.clean_state_sha256,
            after_state_sha256=after.clean_state_sha256,
            allocator_allocated_bytes=after.allocator_allocated_bytes,
            allocator_reserved_bytes=after.allocator_reserved_bytes,
            adapter_version=after.adapter_version,
            optimizer_generation=after.optimizer_generation,
            active_requests=after.active_requests,
            queued_requests=after.queued_requests,
            completion_event_sha256=after.completion_event_sha256,
            reset_duration_ms=reset_duration_ms,
        )
        value.validate(session_plan=session_plan, open_receipt=open_receipt)
        return value


@dataclass(frozen=True)
class SessionExecutionBinding:
    session_plan_sha256: str
    session_open_receipt_sha256: str
    reset_receipt_sha256: str
    execution_plan_sha256: str
    session_epoch: int
    native_session_id: str
    native_trace_epoch: int
    native_previous_run_id: str | None

    def validate(self) -> None:
        for name in (
            "session_plan_sha256",
            "session_open_receipt_sha256",
            "reset_receipt_sha256",
            "execution_plan_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if isinstance(self.session_epoch, bool) or self.session_epoch < 0:
            raise ValueError("session epoch must be a non-negative integer")
        _require_text("native_session_id", self.native_session_id)
        if (
            isinstance(self.native_trace_epoch, bool)
            or not isinstance(self.native_trace_epoch, int)
            or self.native_trace_epoch < 1
        ):
            raise ValueError("native trace epoch must be a positive integer")
        if self.native_previous_run_id is not None:
            _require_text("native_previous_run_id", self.native_previous_run_id)


class SessionServerHandle(Protocol):
    async def wait_ready(self, timeout_s: float) -> None: ...

    async def terminate(self, timeout_s: float) -> None: ...


class SessionTransport(Protocol):
    async def open(
        self,
        *,
        request_timeout_s: float,
        abort_timeout_s: float,
    ) -> None: ...

    async def close(self) -> None: ...


class SessionBoundaryRuntime(Protocol):
    """Native begin/reset/finalize/close boundary for one live process."""

    async def attest_open(
        self,
        *,
        session_plan: IndustrialServerSessionPlan,
        handle: SessionServerHandle,
    ) -> IndustrialSessionOpenReceipt: ...

    async def reset(
        self,
        *,
        session_plan: IndustrialServerSessionPlan,
        open_receipt: IndustrialSessionOpenReceipt,
        prior_execution_plan_sha256: str | None,
        next_execution_plan_sha256: str,
    ) -> IndustrialResetReceipt: ...

    async def finalize(
        self,
        *,
        session_plan: IndustrialServerSessionPlan,
        open_receipt: IndustrialSessionOpenReceipt,
        execution_plan_sha256: str,
        terminal_receipt_sha256: str,
    ) -> str: ...

    async def close(
        self,
        *,
        session_plan: IndustrialServerSessionPlan,
        open_receipt: IndustrialSessionOpenReceipt,
    ) -> None: ...


class SessionExecutionLifecycle(Protocol):
    """Content-bound callbacks whose elapsed time is measured by the executor."""

    def claim_startup_interval_ns(
        self,
        *,
        execution_plan_sha256: str,
    ) -> tuple[int, int] | None: ...

    async def prepare_trace(
        self,
        *,
        execution_plan_sha256: str,
    ) -> SessionExecutionBinding: ...

    async def complete_trace(
        self,
        *,
        execution_plan_sha256: str,
        terminal_receipt_sha256: str,
        run_id: str,
    ) -> IndustrialSessionTraceReceipt: ...


@dataclass(eq=False, slots=True)
class OpenIndustrialServerSession:
    """One live process; intentionally mutable only for ordered trace progress."""

    plan: IndustrialServerSessionPlan
    execution_plans: tuple[IndustrialExecutionPlan, ...]
    output_roots: tuple[Path, ...]
    run_nonce_sha256s: tuple[str, ...]
    handle: SessionServerHandle
    transport: SessionTransport
    boundary_runtime: SessionBoundaryRuntime
    open_receipt: IndustrialSessionOpenReceipt
    next_trace_index: int = 0
    previous_native_run_id: str | None = None
    pending_reset_receipt: IndustrialResetReceipt | None = None
    last_trace_receipt: IndustrialSessionTraceReceipt | None = None
    closed: bool = False
    _authority_sealed: bool = field(default=False, init=False, repr=False)

    _IMMUTABLE_AUTHORITY_FIELDS = frozenset(
        {
            "plan",
            "execution_plans",
            "output_roots",
            "run_nonce_sha256s",
            "handle",
            "transport",
            "boundary_runtime",
            "open_receipt",
            "_authority_sealed",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_authority_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._IMMUTABLE_AUTHORITY_FIELDS and getattr(
            self, "_authority_sealed", False
        ):
            raise AttributeError(f"live session authority field is immutable: {name}")
        object.__setattr__(self, name, value)

    def validate(self) -> None:
        # Local import avoids the executor/session type cycle while rejecting
        # proxy plans that merely self-report an already-authorized digest.
        from lightcone_spec.orchestration.executor import IndustrialExecutionPlan

        if type(self.plan) is not IndustrialServerSessionPlan or any(
            type(plan) is not IndustrialExecutionPlan for plan in self.execution_plans
        ):
            raise TypeError("live session requires exact execution-plan authorities")
        for plan in self.execution_plans:
            plan.validate()
        self.plan.validate()
        self.open_receipt.validate()
        if self.open_receipt.session_plan_sha256 != self.plan.sha256:
            raise ValueError("session-open receipt belongs to another session plan")
        if (
            self.open_receipt.native_capability_receipt_sha256
            != self.plan.session_key.capability_receipt_sha256
        ):
            raise ValueError(
                "session-open receipt has a foreign native capability binding"
            )
        if tuple(plan.sha256 for plan in self.execution_plans) != (
            self.plan.execution_plan_sha256s
        ):
            raise ValueError("live session members differ from the immutable plan")
        if not (
            len(self.execution_plans)
            == len(self.output_roots)
            == len(self.run_nonce_sha256s)
        ):
            raise ValueError("live session trace authorities have unequal coverage")
        for plan, root, nonce in zip(
            self.execution_plans,
            self.output_roots,
            self.run_nonce_sha256s,
            strict=True,
        ):
            _require_sha256("run nonce", nonce)
            expected_root = Path(
                plan.runtime_plan.cell.resources.evidence_root
            ).resolve()
            if root != expected_root:
                raise ValueError("live session output root differs from its plan")
        if not 0 <= self.next_trace_index <= len(self.execution_plans):
            raise ValueError("live session trace index is outside its plan")
        if self.previous_native_run_id is not None:
            _require_text("previous_native_run_id", self.previous_native_run_id)
        if (self.next_trace_index == 0) != (self.previous_native_run_id is None):
            raise ValueError("live session native trace lineage is incomplete")
        if self.pending_reset_receipt is not None:
            self.pending_reset_receipt.validate(
                session_plan=self.plan,
                open_receipt=self.open_receipt,
            )
        if self.last_trace_receipt is not None:
            self.last_trace_receipt.validate()
        if self.closed and self.next_trace_index < len(self.execution_plans):
            raise ValueError("server session closed before all registered traces")


@dataclass(frozen=True)
class IndustrialSessionTraceReceipt:
    session_plan_sha256: str
    session_open_receipt_sha256: str
    execution_plan_sha256: str
    reset_receipt_sha256: str
    terminal_receipt_sha256: str
    native_finalize_receipt_sha256: str

    def validate(self) -> None:
        for name in asdict(self):
            _require_sha256(name, getattr(self, name))

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256({"schema_version": 1, **asdict(self)})


@dataclass(frozen=True)
class IndustrialServerBlockResult:
    """A block executed as independent cold traces, never as session reuse.

    This is the release-safe fallback while the live shared-session boundary is
    unavailable.  Every member retains its own terminal receipt and budget
    observation; this aggregate binds their order without claiming a reset or
    startup saving.
    """

    session_plan_sha256: str
    execution_mode: str
    fallback_reason: str
    executions: tuple[IndustrialExecutionResult, ...]

    def validate(
        self,
        *,
        session_plan: IndustrialServerSessionPlan | None = None,
    ) -> None:
        from lightcone_spec.orchestration.executor import IndustrialExecutionResult

        _require_sha256("session_plan_sha256", self.session_plan_sha256)
        if self.execution_mode != SHARED_SESSION_FALLBACK_MODE:
            raise ValueError("server block execution mode is not release-supported")
        if self.fallback_reason != SHARED_SESSION_UNAVAILABLE_REASON:
            raise ValueError("server block fallback reason is not canonical")
        if not self.executions:
            raise ValueError("server block result requires at least one execution")
        if any(type(row) is not IndustrialExecutionResult for row in self.executions):
            raise TypeError("server block result requires exact execution results")
        if any(row.resumed for row in self.executions):
            raise ValueError("fresh-process block execution cannot contain a resume")
        if len({row.execution_plan_sha256 for row in self.executions}) != len(
            self.executions
        ):
            raise ValueError("server block execution-plan results must be unique")
        for row in self.executions:
            for name in (
                "execution_plan_sha256",
                "experiment_budget_sha256",
                "rank_config_sha256",
                "topology_sha256",
                "terminal_receipt_sha256",
                "budget_observation_sha256",
            ):
                _require_sha256(name, getattr(row, name))
            _require_text("run_id", row.run_id)
        if session_plan is not None:
            if type(session_plan) is not IndustrialServerSessionPlan:
                raise TypeError(
                    "server block validation requires an exact session plan"
                )
            session_plan.validate()
            if (
                self.session_plan_sha256 != session_plan.sha256
                or tuple(row.execution_plan_sha256 for row in self.executions)
                != session_plan.execution_plan_sha256s
            ):
                raise ValueError(
                    "server block results differ from the ordered session plan"
                )

    @property
    def sha256(self) -> str:
        self.validate()
        return _content_sha256(
            {
                "schema_version": 1,
                "session_plan_sha256": self.session_plan_sha256,
                "execution_mode": self.execution_mode,
                "fallback_reason": self.fallback_reason,
                "executions": [
                    {
                        "run_id": row.run_id,
                        "execution_plan_sha256": row.execution_plan_sha256,
                        "experiment_budget_sha256": row.experiment_budget_sha256,
                        "rank_config_sha256": row.rank_config_sha256,
                        "topology_sha256": row.topology_sha256,
                        "terminal_receipt_sha256": row.terminal_receipt_sha256,
                        "budget_observation_sha256": row.budget_observation_sha256,
                    }
                    for row in self.executions
                ],
            }
        )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _StartupTimingAuthority:
    """Immutable, module-owned startup observation for one live session."""

    session: OpenIndustrialServerSession
    interval_ns: tuple[int, int]


_STARTUP_TIMING_AUTHORITIES: dict[int, _StartupTimingAuthority] = {}


def _register_startup_timing(
    session: OpenIndustrialServerSession,
    *,
    started_ns: int,
    completed_ns: int,
) -> None:
    if (
        isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or isinstance(completed_ns, bool)
        or not isinstance(completed_ns, int)
        or completed_ns < started_ns
    ):
        raise RuntimeError("session startup monotonic interval is invalid")
    key = id(session)
    if key in _STARTUP_TIMING_AUTHORITIES:
        raise RuntimeError("session startup timing authority already exists")
    _STARTUP_TIMING_AUTHORITIES[key] = _StartupTimingAuthority(
        session=session,
        interval_ns=(started_ns, completed_ns),
    )


def _claim_startup_timing(
    session: OpenIndustrialServerSession,
) -> tuple[int, int] | None:
    key = id(session)
    authority = _STARTUP_TIMING_AUTHORITIES.get(key)
    if session.next_trace_index == 0:
        if authority is None or authority.session is not session:
            raise RuntimeError("session startup observation was already assigned")
        del _STARTUP_TIMING_AUTHORITIES[key]
        return authority.interval_ns
    if authority is not None:
        if authority.session is not session:
            raise RuntimeError("session startup timing authority identity collided")
        raise RuntimeError("session startup observation was never assigned")
    return None


def _discard_startup_timing(session: OpenIndustrialServerSession) -> None:
    authority = _STARTUP_TIMING_AUTHORITIES.get(id(session))
    if authority is not None and authority.session is session:
        del _STARTUP_TIMING_AUTHORITIES[id(session)]


def _validate_session_launch_authority(
    session_plan: IndustrialServerSessionPlan,
    execution_plans: tuple[IndustrialExecutionPlan, ...],
    *,
    native_evidence: object,
) -> None:
    """Replay exact plan, dispatch, session-key, and release capability authority."""

    # Local import avoids the executor/session type cycle while making the
    # mutation boundary require the concrete release-owned plan type.
    from lightcone_spec.orchestration.executor import (
        IndustrialExecutionPlan,
        NativeEvidenceUnavailableError,
        native_evidence_preflight,
    )

    if not execution_plans or any(
        type(plan) is not IndustrialExecutionPlan for plan in execution_plans
    ):
        raise TypeError(
            "server session launch requires exact IndustrialExecutionPlan instances"
        )
    for plan in execution_plans:
        plan.validate()
    session_plan.validate()
    expected = IndustrialServerSessionPlan.create(
        execution_plans,
        capability_receipt_sha256=(session_plan.session_key.capability_receipt_sha256),
        compile_cache_receipt_sha256=(
            session_plan.session_key.compile_cache_receipt_sha256
        ),
        dtype=session_plan.session_key.dtype,
        precision=session_plan.session_key.precision,
        graph_buckets=session_plan.session_key.graph_buckets,
        hbm_reservation_bytes=session_plan.session_key.hbm_reservation_bytes,
    )
    if session_plan != expected:
        raise ValueError(
            "server-session plan differs from the exact execution-plan replay"
        )
    for plan in execution_plans:
        preflight = native_evidence_preflight(plan, native_evidence)
        if preflight.status == "BLOCKED":
            raise NativeEvidenceUnavailableError(preflight)


async def open_server_session(
    session_plan: IndustrialServerSessionPlan,
    execution_plans: tuple[IndustrialExecutionPlan, ...],
    *,
    output_roots: tuple[str | Path, ...],
    run_nonce_sha256s: tuple[str, ...],
    launch_server: Any,
    transport: SessionTransport,
    boundary_runtime: SessionBoundaryRuntime,
    native_evidence: Any = None,
) -> OpenIndustrialServerSession:
    """Fail closed before launching an unsupported shared server session."""

    _raise_shared_session_unavailable()

    _validate_session_launch_authority(
        session_plan,
        execution_plans,
        native_evidence=native_evidence,
    )
    if not (len(execution_plans) == len(output_roots) == len(run_nonce_sha256s)):
        raise ValueError(
            "session trace plans, roots, and nonces must have equal length"
        )
    from lightcone_spec.orchestration.executor import (
        _preflight_industrial_session_trace_evidence,
    )

    normalized_roots = tuple(Path(root).resolve() for root in output_roots)
    for plan, output_root, nonce in zip(
        execution_plans,
        output_roots,
        run_nonce_sha256s,
        strict=True,
    ):
        _preflight_industrial_session_trace_evidence(
            plan,
            output_root=output_root,
            run_nonce_sha256=nonce,
        )
    first = execution_plans[0]
    startup_started_ns = time.perf_counter_ns()
    handle = await launch_server(first.server_launch)
    try:
        await handle.wait_ready(max(plan.startup_timeout_s for plan in execution_plans))
        await transport.open(
            request_timeout_s=max(
                plan.load_plan.window.request_deadline_us / 1_000_000
                for plan in execution_plans
            ),
            abort_timeout_s=max(plan.abort_grace_s for plan in execution_plans),
        )
        try:
            receipt = await boundary_runtime.attest_open(
                session_plan=session_plan,
                handle=handle,
            )
            startup_completed_ns = time.perf_counter_ns()
            if startup_completed_ns < startup_started_ns:
                raise RuntimeError("session startup monotonic clock moved backwards")
            value = OpenIndustrialServerSession(
                plan=session_plan,
                execution_plans=execution_plans,
                output_roots=normalized_roots,
                run_nonce_sha256s=run_nonce_sha256s,
                handle=handle,
                transport=transport,
                boundary_runtime=boundary_runtime,
                open_receipt=receipt,
            )
            value.validate()
            _register_startup_timing(
                value,
                started_ns=startup_started_ns,
                completed_ns=startup_completed_ns,
            )
            return value
        except BaseException:
            await transport.close()
            raise
    except BaseException:
        await handle.terminate(max(plan.shutdown_timeout_s for plan in execution_plans))
        raise


async def reset_and_attest_trace_boundary(
    session: OpenIndustrialServerSession,
) -> IndustrialResetReceipt:
    """Fail closed before mutating an unsupported shared server session."""

    _raise_shared_session_unavailable()

    session.validate()
    if session.closed or session.next_trace_index >= len(session.execution_plans):
        raise RuntimeError("server session has no pending logical trace")
    if session.pending_reset_receipt is not None:
        raise RuntimeError("server session already has an unfinalized reset boundary")
    next_plan = session.execution_plans[session.next_trace_index]
    prior = (
        None
        if session.next_trace_index == 0
        else session.execution_plans[session.next_trace_index - 1].sha256
    )
    receipt = await session.boundary_runtime.reset(
        session_plan=session.plan,
        open_receipt=session.open_receipt,
        prior_execution_plan_sha256=prior,
        next_execution_plan_sha256=next_plan.sha256,
    )
    receipt.validate(session_plan=session.plan, open_receipt=session.open_receipt)
    session.pending_reset_receipt = receipt
    return receipt


async def execute_trace_in_session(
    session: OpenIndustrialServerSession,
    *,
    output_root: str | Path,
    run_nonce_sha256: str,
    native_evidence: Any = None,
) -> tuple[Any, IndustrialSessionTraceReceipt]:
    """Fail closed before executing an unsupported shared-session trace."""

    _raise_shared_session_unavailable()

    session.validate()
    if session.closed or session.next_trace_index >= len(session.execution_plans):
        raise RuntimeError("server session has no pending logical trace")
    plan = session.execution_plans[session.next_trace_index]
    if (
        Path(output_root).resolve() != session.output_roots[session.next_trace_index]
        or run_nonce_sha256 != session.run_nonce_sha256s[session.next_trace_index]
    ):
        raise ValueError("session trace differs from its preflighted output authority")
    from lightcone_spec.orchestration.executor import execute_industrial_plan

    lifecycle = _issue_session_execution_lifecycle(session)
    try:
        result = await execute_industrial_plan(
            plan,
            output_root=output_root,
            run_nonce_sha256=run_nonce_sha256,
            launch_server=None,
            transport=session.transport,
            native_evidence=native_evidence,
            existing_handle=session.handle,
            transport_already_open=True,
            keep_session_open=True,
            session_lifecycle=lifecycle,
        )
    finally:
        _revoke_session_execution_lifecycle(lifecycle)
    receipt = session.last_trace_receipt
    if receipt is None or receipt.execution_plan_sha256 != plan.sha256:
        raise RuntimeError("executor did not finalize the validated session trace")
    return result, receipt


async def finalize_trace(
    session: OpenIndustrialServerSession,
    *,
    result: Any,
    reset_receipt: IndustrialResetReceipt,
) -> IndustrialSessionTraceReceipt:
    """Fail closed before finalizing an unsupported shared-session trace."""

    _raise_shared_session_unavailable()

    return await _finalize_trace_from_terminal(
        session,
        execution_plan_sha256=result.execution_plan_sha256,
        terminal_receipt_sha256=_file_sha256(result.terminal_receipt),
        run_id=result.run_id,
        reset_receipt=reset_receipt,
    )


async def _finalize_trace_from_terminal(
    session: OpenIndustrialServerSession,
    *,
    execution_plan_sha256: str,
    terminal_receipt_sha256: str,
    run_id: str,
    reset_receipt: IndustrialResetReceipt,
) -> IndustrialSessionTraceReceipt:
    """Fail closed even if an internal lifecycle object is caller-forged."""

    _raise_shared_session_unavailable()

    session.validate()
    if session.closed or session.next_trace_index >= len(session.execution_plans):
        raise RuntimeError("server session has no trace awaiting finalization")
    plan = session.execution_plans[session.next_trace_index]
    if execution_plan_sha256 != plan.sha256:
        raise ValueError("trace result belongs to another execution plan")
    if session.pending_reset_receipt != reset_receipt:
        raise ValueError("trace finalization differs from its prepared reset boundary")
    _require_sha256("terminal receipt", terminal_receipt_sha256)
    _require_text("run_id", run_id)
    native_finalize_sha256 = await session.boundary_runtime.finalize(
        session_plan=session.plan,
        open_receipt=session.open_receipt,
        execution_plan_sha256=plan.sha256,
        terminal_receipt_sha256=terminal_receipt_sha256,
    )
    _require_sha256("native finalize receipt", native_finalize_sha256)
    receipt = IndustrialSessionTraceReceipt(
        session_plan_sha256=session.plan.sha256,
        session_open_receipt_sha256=session.open_receipt.sha256,
        execution_plan_sha256=plan.sha256,
        reset_receipt_sha256=reset_receipt.sha256,
        terminal_receipt_sha256=terminal_receipt_sha256,
        native_finalize_receipt_sha256=native_finalize_sha256,
    )
    receipt.validate()
    session.previous_native_run_id = run_id
    session.pending_reset_receipt = None
    session.next_trace_index += 1
    session.last_trace_receipt = receipt
    return receipt


async def close_server_session(session: OpenIndustrialServerSession) -> None:
    """Fail closed: callers cannot acquire a supported live shared session."""

    _raise_shared_session_unavailable()

    if session.closed:
        raise RuntimeError("server session is already closed")
    error: BaseException | None = None
    try:
        await session.boundary_runtime.close(
            session_plan=session.plan,
            open_receipt=session.open_receipt,
        )
    except BaseException as caught:  # noqa: BLE001 - lifecycle cleanup boundary
        error = caught
    try:
        await session.transport.close()
    except BaseException as caught:  # noqa: BLE001 - lifecycle cleanup boundary
        if error is None:
            error = caught
        else:
            error.add_note(f"HTTP pool close also failed: {caught}")
    try:
        await session.handle.terminate(
            max(plan.shutdown_timeout_s for plan in session.execution_plans)
        )
    except BaseException as caught:  # noqa: BLE001 - lifecycle cleanup boundary
        if error is None:
            error = caught
        else:
            error.add_note(f"server termination also failed: {caught}")
    session.closed = True
    _discard_startup_timing(session)
    if error is not None:
        raise error


@dataclass(frozen=True)
class _SessionExecutionLifecycle:
    """Executor callback surface for exactly one live, ordered session."""

    session: OpenIndustrialServerSession

    def _current_plan(self, execution_plan_sha256: str) -> _ExecutionPlan:
        self.session.validate()
        if self.session.closed or self.session.next_trace_index >= len(
            self.session.execution_plans
        ):
            raise RuntimeError("server session has no pending logical trace")
        plan = self.session.execution_plans[self.session.next_trace_index]
        if plan.sha256 != execution_plan_sha256:
            raise ValueError("session lifecycle callback names the wrong trace")
        return plan

    def claim_startup_interval_ns(
        self,
        *,
        execution_plan_sha256: str,
    ) -> tuple[int, int] | None:
        _raise_shared_session_unavailable()
        self._current_plan(execution_plan_sha256)
        return _claim_startup_timing(self.session)

    async def prepare_trace(
        self,
        *,
        execution_plan_sha256: str,
    ) -> SessionExecutionBinding:
        _raise_shared_session_unavailable()
        plan = self._current_plan(execution_plan_sha256)
        reset_receipt = await reset_and_attest_trace_boundary(self.session)
        binding = SessionExecutionBinding(
            session_plan_sha256=self.session.plan.sha256,
            session_open_receipt_sha256=self.session.open_receipt.sha256,
            reset_receipt_sha256=reset_receipt.sha256,
            execution_plan_sha256=plan.sha256,
            session_epoch=self.session.open_receipt.session_epoch,
            native_session_id=self.session.plan.sha256,
            native_trace_epoch=self.session.next_trace_index + 1,
            native_previous_run_id=self.session.previous_native_run_id,
        )
        binding.validate()
        return binding

    async def complete_trace(
        self,
        *,
        execution_plan_sha256: str,
        terminal_receipt_sha256: str,
        run_id: str,
    ) -> IndustrialSessionTraceReceipt:
        _raise_shared_session_unavailable()
        self._current_plan(execution_plan_sha256)
        reset_receipt = self.session.pending_reset_receipt
        if reset_receipt is None:
            raise RuntimeError("session trace completed without a reset boundary")
        receipt = await _finalize_trace_from_terminal(
            self.session,
            execution_plan_sha256=execution_plan_sha256,
            terminal_receipt_sha256=terminal_receipt_sha256,
            run_id=run_id,
            reset_receipt=reset_receipt,
        )
        if self.session.next_trace_index == len(self.session.execution_plans):
            await close_server_session(self.session)
        return receipt


_ISSUED_SESSION_EXECUTION_LIFECYCLES: dict[int, _SessionExecutionLifecycle] = {}


def _issue_session_execution_lifecycle(
    session: OpenIndustrialServerSession,
) -> _SessionExecutionLifecycle:
    _raise_shared_session_unavailable()
    value = _SessionExecutionLifecycle(session)
    key = id(value)
    if key in _ISSUED_SESSION_EXECUTION_LIFECYCLES:
        raise RuntimeError("session execution lifecycle identity collided")
    _ISSUED_SESSION_EXECUTION_LIFECYCLES[key] = value
    return value


def _revoke_session_execution_lifecycle(
    value: _SessionExecutionLifecycle,
) -> None:
    issued = _ISSUED_SESSION_EXECUTION_LIFECYCLES.get(id(value))
    if issued is value:
        del _ISSUED_SESSION_EXECUTION_LIFECYCLES[id(value)]


def _is_internal_session_execution_lifecycle(value: object) -> bool:
    """Accept only a consume-once lifecycle minted for the active session call."""

    return (
        type(value) is _SessionExecutionLifecycle
        and _ISSUED_SESSION_EXECUTION_LIFECYCLES.get(id(value)) is value
    )


def _validate_fresh_process_resource_pair(
    transport: BenchServingTransport,
    provider: NativeTerminalProvider,
) -> None:
    """Reject any HTTP/native state that could refer to an older process."""

    from lightcone_spec.experiments.serving import PinnedBenchServingTransport
    from lightcone_spec.orchestration.native_terminal import NativeTerminalProvider

    if not isinstance(transport, PinnedBenchServingTransport):
        raise TypeError("server block requires the pinned official bench transport")
    if type(provider) is not NativeTerminalProvider:
        raise TypeError("server block requires exact native terminal providers")
    if provider._transport is not transport:
        raise ValueError(
            "server block native and serving traffic use different HTTP pools"
        )
    metrics = transport.metrics()
    if (
        type(metrics) is not dict
        or set(metrics)
        != {"connections_created", "submitted_requests", "reused_requests"}
        or any(type(value) is not int or value != 0 for value in metrics.values())
        or getattr(transport, "_native_admin_base_url", None) is not None
    ):
        raise ValueError(
            "clean-process fallback rejects a stale or previously used HTTP pool"
        )
    if type(transport) is PinnedBenchServingTransport and any(
        getattr(transport, name) is not None
        for name in ("_session", "_request_timeout_s", "_abort_timeout_s")
    ):
        raise ValueError("clean-process fallback rejects a live HTTP pool")
    if (
        provider.phase != "IDLE"
        or provider._binding is not None
        or provider._begin is not None
        or provider._reset is not None
        or provider._process is not None
        or provider._reset_generation != 0
        or provider._next_session_epoch != 1
        or provider._last_finalized_run_id is not None
        or provider._session_id is not None
        or provider._seen_runs
        or provider._seen_attempts
    ):
        raise ValueError("clean-process fallback rejects stale native terminal state")


def _validate_fresh_execution_result(
    plan: IndustrialExecutionPlan,
    result: IndustrialExecutionResult,
) -> None:
    """Bind one standalone result before another process may be launched."""

    from lightcone_spec.orchestration.executor import IndustrialExecutionResult

    if type(result) is not IndustrialExecutionResult:
        raise TypeError("standalone executor returned an inexact execution result")
    if (
        result.resumed
        or result.execution_plan_sha256 != plan.sha256
        or result.experiment_budget_sha256 != plan.budget.sha256
        or result.rank_config_sha256 != plan.rank_config_sha256
        or result.topology_sha256 != plan.topology_sha256
    ):
        raise ValueError(
            "fresh-process result differs from its exact standalone execution plan"
        )
    for name in (
        "execution_plan_sha256",
        "experiment_budget_sha256",
        "rank_config_sha256",
        "topology_sha256",
        "terminal_receipt_sha256",
        "budget_observation_sha256",
    ):
        _require_sha256(name, getattr(result, name))
    _require_text("run_id", result.run_id)


async def execute_industrial_fresh_process_fallback(
    session_plan: IndustrialServerSessionPlan,
    execution_plans: tuple[IndustrialExecutionPlan, ...],
    *,
    output_roots: tuple[str | Path, ...],
    run_nonce_sha256s: tuple[str, ...],
    launch_server: ServerLauncher,
    resources_for_plan: Callable[
        [IndustrialExecutionPlan],
        tuple[BenchServingTransport, NativeTerminalProvider],
    ],
) -> IndustrialServerBlockResult:
    """Execute a registered block via one clean process per logical trace.

    This explicit fallback has no old-handle input.  It validates the entire
    block and pristine per-trace HTTP/native resources before the first launch,
    then awaits the ordinary standalone executor for each plan in order.  A
    failed cleanup therefore stops the block before the next process starts.
    The result never claims shared-session reset or reuse.
    """

    from lightcone_spec.orchestration.executor import (
        IndustrialExecutionPlan,
        NativeEvidenceUnavailableError,
        _preflight_industrial_session_trace_evidence,
        execute_industrial_plan,
        native_evidence_preflight,
    )

    if type(session_plan) is not IndustrialServerSessionPlan:
        raise TypeError("server block requires an exact session-plan authority")
    if not execution_plans or any(
        type(plan) is not IndustrialExecutionPlan for plan in execution_plans
    ):
        raise TypeError("server block requires exact execution-plan authorities")
    if not (len(execution_plans) == len(output_roots) == len(run_nonce_sha256s)):
        raise ValueError(
            "session trace plans, roots, and nonces must have equal length"
        )
    for plan in execution_plans:
        plan.validate()
    session_plan.validate()
    expected = IndustrialServerSessionPlan.create(
        execution_plans,
        capability_receipt_sha256=(session_plan.session_key.capability_receipt_sha256),
        compile_cache_receipt_sha256=(
            session_plan.session_key.compile_cache_receipt_sha256
        ),
        dtype=session_plan.session_key.dtype,
        precision=session_plan.session_key.precision,
        graph_buckets=session_plan.session_key.graph_buckets,
        hbm_reservation_bytes=session_plan.session_key.hbm_reservation_bytes,
    )
    if session_plan != expected:
        raise ValueError(
            "server block plan differs from the exact execution-plan replay"
        )
    normalized_roots = tuple(Path(root).resolve() for root in output_roots)
    for plan, output_root, nonce in zip(
        execution_plans,
        normalized_roots,
        run_nonce_sha256s,
        strict=True,
    ):
        _preflight_industrial_session_trace_evidence(
            plan,
            output_root=output_root,
            run_nonce_sha256=nonce,
        )

    # Materialize and validate every resource pair before the first process or
    # network mutation.  Distinct identities make the clean-process guarantee
    # explicit and prevent accidental HTTP/native lifecycle reuse.
    resources = tuple(resources_for_plan(plan) for plan in execution_plans)
    if any(type(row) is not tuple or len(row) != 2 for row in resources):
        raise TypeError("resource factory must return (transport, native provider)")
    transports = tuple(row[0] for row in resources)
    providers = tuple(row[1] for row in resources)
    if len({id(value) for value in transports}) != len(transports) or len(
        {id(value) for value in providers}
    ) != len(providers):
        raise ValueError("clean-process fallback requires unique per-trace resources")
    for plan, transport, provider in zip(
        execution_plans,
        transports,
        providers,
        strict=True,
    ):
        _validate_fresh_process_resource_pair(transport, provider)
        preflight = native_evidence_preflight(plan, provider)
        if preflight.status == "BLOCKED":
            raise NativeEvidenceUnavailableError(preflight)

    results = []
    for plan, output_root, nonce, transport, provider in zip(
        execution_plans,
        normalized_roots,
        run_nonce_sha256s,
        transports,
        providers,
        strict=True,
    ):
        result = await execute_industrial_plan(
            plan,
            output_root=output_root,
            run_nonce_sha256=nonce,
            launch_server=launch_server,
            transport=transport,
            native_evidence=provider,
        )
        _validate_fresh_execution_result(plan, result)
        results.append(result)
    value = IndustrialServerBlockResult(
        session_plan_sha256=session_plan.sha256,
        execution_mode=SHARED_SESSION_FALLBACK_MODE,
        fallback_reason=SHARED_SESSION_UNAVAILABLE_REASON,
        executions=tuple(results),
    )
    value.validate(session_plan=session_plan)
    return value


async def execute_industrial_server_session(
    session_plan: IndustrialServerSessionPlan,
    execution_plans: tuple[IndustrialExecutionPlan, ...],
    *,
    output_roots: tuple[str | Path, ...],
    run_nonce_sha256s: tuple[str, ...],
    launch_server: ServerLauncher,
    resources_for_plan: Callable[
        [IndustrialExecutionPlan],
        tuple[BenchServingTransport, NativeTerminalProvider],
    ],
) -> IndustrialServerBlockResult:
    """Route the release-blocked shared session to the explicit safe fallback.

    Shared-session open/reset remains blocked before mutation.  Consequently no
    shared handle exists to reuse or transfer: every trace is delegated to the
    standalone executor with a pristine resource pair and no session arguments.
    """

    return await execute_industrial_fresh_process_fallback(
        session_plan,
        execution_plans,
        output_roots=output_roots,
        run_nonce_sha256s=run_nonce_sha256s,
        launch_server=launch_server,
        resources_for_plan=resources_for_plan,
    )
