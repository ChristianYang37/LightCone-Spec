"""Concrete pinned-SGLang process owner for one resident TP1 group.

This module is the physical producer missing from the schema-only resident
path.  It launches one process group from the first member's deeply replayed
prepared launch, opens one official bench HTTP pool, and drives every logical
trace through the source-owned session lifecycle.  Per-cell terminal evidence
is published at its registered path, but the caller still delays cell-manifest
publication until the shared close receipt proves the process group empty.

The implementation deliberately passes no ``VerifiedNativeRuntimeGpuProof``.
Reuse eligibility comes from the separately qualified, path-only trusted
empirical authority consumed by the group plan; this producer remains unsigned
and ``formal_measured=False`` throughout.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lightcone_spec.config import RunConfig, load_run_config
from lightcone_spec.orchestration.formal_serving_session_group import (
    FormalServingSessionGroupSpec,
)
from lightcone_spec.orchestration.formal_serving_session_group_launch import (
    RevalidatedFormalServingResidentGroupLaunch,
    publish_formal_serving_resident_group_launch_authority,
)
from lightcone_spec.orchestration.formal_serving_session_group_physical import (
    FormalServingResidentCloseEvidence,
    FormalServingResidentProcessDriver,
    FormalServingResidentProcessFactory,
    FormalServingResidentResetEvidence,
    FormalServingResidentTraceEvidence,
)
from lightcone_spec.orchestration.formal_serving_session_group_worker import (
    FormalServingSessionMemberPhysicalResult,
    RevalidatedFormalServingSessionGroupExecution,
)
from lightcone_spec.orchestration.formal_serving_session_source_chain import (
    FormalServingResidentSourceChainPublisher,
)
from lightcone_spec.orchestration.formal_terminal_shards import (
    publish_scalable_client_request_lifecycle,
    publish_scalable_native_terminal_artifact,
    publish_scalable_unsigned_native_itl_bundle,
)
from lightcone_spec.orchestration.live_sglang import (
    _ABORT_TIMEOUT_SECONDS,
    _SERVER_READY_TIMEOUT_SECONDS,
    PinnedNvidiaSmiTool,
    _capture_gpu_process_snapshot,
    _execute_source_owned_phase,
    _observe_live_server_execution_policy,
    _reopen_native_scored_interval,
    _require_port_unused,
    _terminate_process_group,
    _wait_server_ready,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalProvider,
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    UnsignedNativeServingPhaseResult,
    ValidatedNativeTerminalEvidence,
    canonical_sha256,
    validate_native_terminal_artifact,
    validate_unsigned_native_itl_pointer_bundle,
)
from lightcone_spec.orchestration.session_live_runtime import (
    SessionLiveContractResult,
    SessionLivePinnedBenchTransport,
    SessionLiveStepBinding,
    SessionLiveTerminalObserver,
    SessionLiveTraceDriver,
    SessionLiveTraceInput,
    run_session_live_contract,
)
from lightcone_spec.orchestration.session_reuse_authority import (
    ConnectionAccounting,
    SourceOwnedCloseReceipt,
    SourceOwnedInitialStateReceipt,
    SourceOwnedResetReceipt,
    SourceOwnedScoredClockReceipt,
    SourceOwnedSessionCapability,
    SourceOwnedTraceReceipt,
    SourceOwnedWarmupReceipt,
)
from lightcone_spec.runtime.attestation import NO_TRUSTED_ATTESTERS
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

if TYPE_CHECKING:
    from lightcone_spec.experiments.serving import BoundServingRequest
    from lightcone_spec.orchestration.executor import RegisteredServingExecutionPolicy
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRequestScheduleReceipt,
        FormalServingRunPlan,
    )


@dataclass(frozen=True)
class _PreparedResidentMember:
    spec: FormalServingSessionGroupSpec
    plan: FormalServingRunPlan
    launch: CompileLaunchManifest
    config: RunConfig
    schedule: FormalServingRequestScheduleReceipt
    warmup: tuple[BoundServingRequest, ...]
    scored: tuple[BoundServingRequest, ...]
    execution_policy: RegisteredServingExecutionPolicy
    timeout_seconds: float


def _prepare_resident_member(
    member: FormalServingSessionGroupSpec,
) -> _PreparedResidentMember:
    """Deep-replay the exact current prepared TP1 input used by fresh execution."""

    from lightcone_spec.orchestration.executor import RegisteredServingExecutionPolicy
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        _load_formal_single_operator_trusted_run_plan,
        formal_serving_request_schedule_rows,
    )

    plan, launch, schedule = _load_formal_single_operator_trusted_run_plan(
        member.run_plan.absolute_path
    )
    config = load_run_config(launch.run_config_path)
    source = plan.single_operator_execution_rebuild_source
    if (
        plan.schema_version != 4
        or plan.topology_mode != "tp1_dp1"
        or launch.schema_version != 2
        or source is None
        or source.reopen().get("kind")
        != "formal_single_operator_prepared_downstream_run_plan_inputs"
        or plan.nextn_mtp_mode is not None
        or type(plan.serving_execution_policy) is not RegisteredServingExecutionPolicy
        or type(plan.process_hard_timeout_ns) is not int
        or member.physical_kind != "serving"
        or member.topology_mode != "tp1_dp1"
        or member.run_plan
        != CanonicalJsonProofBinding.bind(member.run_plan.absolute_path)
        or member.materialized_cell_id != plan.materialized_cell_id
        or member.compile_launch_manifest_sha256 != launch.sha256
        or member.run_config_sha256 != launch.run_config_semantic_sha256
        or member.request_schedule_sha256 != schedule.sha256
        or member.assigned_gpu_uuids != launch.gpu_uuids
        or member.inventory_sha256 != launch.inventory_sha256
        or member.method != plan.method
        or member.backend != config.model.algorithm
        or member.output_directory != plan.private_output_root
    ):
        raise ValueError("resident SGLang member is not current prepared ordinary TP1")
    rows = tuple(formal_serving_request_schedule_rows(schedule))
    warmup = tuple(row.request for row in rows if row.phase == "warmup")
    scored = tuple(row.request for row in rows if row.phase == "scored")
    if not warmup or not scored:
        raise ValueError("resident SGLang member lacks request phase coverage")
    return _PreparedResidentMember(
        spec=member,
        plan=plan,
        launch=launch,
        config=config,
        schedule=schedule,
        warmup=warmup,
        scored=scored,
        execution_policy=plan.serving_execution_policy,
        timeout_seconds=plan.process_hard_timeout_ns / 1_000_000_000,
    )


def _open_evidence_stream(path: Path, *, header: str):
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    stream = os.fdopen(descriptor, "wb", buffering=0)
    stream.write((header.rstrip("\n") + "\n").encode("utf-8"))
    return stream


def _resident_transport_from_launch(
    launch: CompileLaunchManifest,
) -> SessionLivePinnedBenchTransport:
    transport = SessionLivePinnedBenchTransport.from_checkout(
        launch.patched_sglang_checkout
    )
    if type(transport) is not SessionLivePinnedBenchTransport:
        raise TypeError(
            "resident SGLang launch did not create the exact live transport"
        )
    return transport


def _spawn_resident_server(
    *,
    actual_server_argv: tuple[str, ...],
    patched_sglang_checkout: str,
    actual_child_environment: tuple[tuple[str, str], ...],
    stdout_file,
    stderr_file,
) -> subprocess.Popen[bytes]:
    """Spawn the server with Linux parent-death cleanup, then exec in-place."""

    if not Path("/proc/self/stat").is_file():
        raise RuntimeError(
            "resident server parent-death ownership requires Linux /proc"
        )
    bootstrap = (
        "import ctypes,os,signal,sys;"
        "parent=os.getppid();"
        "libc=ctypes.CDLL(None,use_errno=True);"
        "rc=libc.prctl(1,signal.SIGTERM,0,0,0);"
        "rc and (_ for _ in ()).throw(OSError(ctypes.get_errno(),"
        "'prctl(PR_SET_PDEATHSIG)'));"
        "os.getppid()!=parent and os._exit(70);"
        "os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
    )
    return subprocess.Popen(
        [sys.executable, "-c", bootstrap, *actual_server_argv],
        cwd=patched_sglang_checkout,
        env=dict(actual_child_environment),
        stdin=subprocess.DEVNULL,
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,
        close_fds=True,
    )


def _step_value(step: SessionLiveStepBinding) -> dict[str, object]:
    value = json.loads(step.raw_json)
    if type(value) is not dict:
        raise TypeError("resident source step must be one JSON object")
    return value


class _ResidentTraceDriver(SessionLiveTraceDriver):
    def __init__(
        self,
        *,
        owner: PinnedSglangResidentProcessDriver,
        index: int,
        prepared: _PreparedResidentMember,
        binding: NativeTerminalRunBinding,
    ) -> None:
        self.owner = owner
        self.index = index
        self.prepared = prepared
        self.binding = binding
        self.execute_permit = asyncio.Event()
        self.reset_ready = asyncio.Event()
        self.terminal_ready = asyncio.Event()
        self.reset_ready_ns: int | None = None
        self.trace_started_ns: int | None = None
        self.scored_call_started_ns: int | None = None
        self.trace_finished_ns: int | None = None
        self.warmup_result: UnsignedNativeServingPhaseResult | None = None
        self.scored_result: UnsignedNativeServingPhaseResult | None = None
        self.terminal: ValidatedNativeTerminalEvidence | None = None

    async def run_warmup(
        self,
        *,
        binding: NativeTerminalRunBinding,
        transport: SessionLivePinnedBenchTransport,
    ) -> tuple[TerminalRequestExpectation, ...]:
        if binding != self.binding or transport is not self.owner.transport:
            raise ValueError("resident warmup left its process/trace binding")
        await self.owner._observe_server_policy_once()
        self.reset_ready_ns = time.monotonic_ns()
        self.reset_ready.set()
        await self.execute_permit.wait()
        self.trace_started_ns = time.monotonic_ns()
        result = await _execute_source_owned_phase(
            "warmup",
            self.prepared.warmup,
            concurrency=self.prepared.config.runtime.max_running_requests,
            transport=transport,
            base_url=self.owner.base_url,
            served_model=self.prepared.config.model.target,
            execution_policy=self.prepared.execution_policy,
        )
        self.warmup_result = result
        return result.requests

    async def run_scored(
        self,
        *,
        binding: NativeTerminalRunBinding,
        transport: SessionLivePinnedBenchTransport,
    ) -> tuple[TerminalRequestExpectation, ...]:
        if (
            binding != self.binding
            or transport is not self.owner.transport
            or self.warmup_result is None
        ):
            raise ValueError("resident scored phase left its warmup/process binding")
        self.scored_call_started_ns = time.monotonic_ns()
        result = await _execute_source_owned_phase(
            "scored",
            self.prepared.scored,
            concurrency=self.prepared.config.runtime.max_running_requests,
            transport=transport,
            base_url=self.owner.base_url,
            served_model=self.prepared.config.model.target,
            execution_policy=self.prepared.execution_policy,
        )
        self.scored_result = result
        return result.requests

    def accept_terminal(self, terminal: ValidatedNativeTerminalEvidence) -> None:
        if terminal.binding != self.binding or self.terminal is not None:
            raise ValueError("resident terminal observer changed/replayed the trace")
        self.terminal = terminal
        self.trace_finished_ns = time.monotonic_ns()
        self.terminal_ready.set()


class PinnedSglangResidentProcessDriver(
    FormalServingResidentProcessDriver, SessionLiveTerminalObserver
):
    """One concrete server process, pool, native provider, and ordered trace set."""

    def __init__(
        self,
        *,
        execution: RevalidatedFormalServingSessionGroupExecution,
        prepared: tuple[_PreparedResidentMember, ...],
        evidence_root: Path,
        nvidia_smi_tool: PinnedNvidiaSmiTool,
        process: subprocess.Popen[bytes],
        group_launch: RevalidatedFormalServingResidentGroupLaunch,
        transport: SessionLivePinnedBenchTransport,
        before_gpu_snapshot: CanonicalJsonProofBinding,
        ready_gpu_snapshot: CanonicalJsonProofBinding,
        server_ready_ns: int,
        server_log,
        server_stdout,
        server_stderr,
    ) -> None:
        self.execution = execution
        self.prepared = prepared
        self.evidence_root = evidence_root
        self.nvidia_smi_tool = nvidia_smi_tool
        self.process = process
        self.group_launch = group_launch
        self.transport = transport
        self.provider = NativeTerminalProvider(
            transport, trusted_attester_policy=NO_TRUSTED_ATTESTERS
        )
        self._before_gpu_snapshot = before_gpu_snapshot
        self._ready_gpu_snapshot = ready_gpu_snapshot
        self._server_ready_ns = server_ready_ns
        self._server_log = server_log
        self._server_stdout = server_stdout
        self._server_stderr = server_stderr
        self._server_log_path = str((evidence_root / "server.log").resolve())
        self._server_stdout_path = str((evidence_root / "server.stdout").resolve())
        self._server_stderr_path = str((evidence_root / "server.stderr").resolve())
        self._native_process_started_ns: int | None = None
        self._policy_observed = False
        self._policy_lock = asyncio.Lock()
        self._termination_lock = asyncio.Lock()
        self._termination: tuple[int, str, int, int] | None = None
        self._contract_result: SessionLiveContractResult | None = None
        self._parsed_capability: SourceOwnedSessionCapability | None = None
        self._parsed_initial: SourceOwnedInitialStateReceipt | None = None
        self._accounting: ConnectionAccounting | None = None
        self._reset_generation: int | None = None
        self._clock_generation = 0
        self._parsed_resets: list[SourceOwnedResetReceipt] = []
        self._parsed_warmups: list[SourceOwnedWarmupReceipt] = []
        self._parsed_clocks: list[SourceOwnedScoredClockReceipt] = []
        self._parsed_traces: list[SourceOwnedTraceReceipt] = []
        self._source_bindings: dict[
            tuple[str, int | None], CanonicalJsonProofBinding
        ] = {}
        source_chain_root = evidence_root / "source-chain"
        source_chain_root.mkdir(mode=0o700)
        self._source_chain = FormalServingResidentSourceChainPublisher(
            output_dir=source_chain_root,
            session_plan_sha256=execution.plan.session_plan_sha256,
            execution_plan_sha256s=tuple(
                self._effective_binding(index).execution_plan_sha256
                for index in range(len(prepared))
            ),
        )
        self._source_reset_bindings: list[CanonicalJsonProofBinding] = []
        self._trace_drivers = tuple(
            _ResidentTraceDriver(
                owner=self,
                index=index,
                prepared=item,
                binding=self._effective_binding(index),
            )
            for index, item in enumerate(prepared)
        )
        self._contract_task = asyncio.create_task(
            self._run_contract(),
            name=f"resident-session-{execution.plan.group_id}",
        )

    def _effective_binding(self, index: int) -> NativeTerminalRunBinding:
        from lightcone_spec.orchestration.formal_serving_session_group_physical import (
            effective_formal_serving_resident_terminal_binding,
        )

        return effective_formal_serving_resident_terminal_binding(
            plan=self.execution.plan, member_index=index
        )

    @property
    def process_id(self) -> int:
        return self.process.pid

    @property
    def process_group_id(self) -> int:
        return (
            os.getpgid(self.process.pid)
            if self.process.poll() is None
            else self.process.pid
        )

    @property
    def process_started_ns(self) -> int:
        if self._native_process_started_ns is None:
            raise RuntimeError("resident native process start identity is not ready")
        return self._native_process_started_ns

    @property
    def ready_ns(self) -> int:
        return self._server_ready_ns

    @property
    def actual_server_argv(self) -> tuple[str, ...]:
        return self.group_launch.authority.actual_server_argv

    @property
    def group_launch_authority(self) -> CanonicalJsonProofBinding:
        return self.group_launch.binding

    @property
    def base_url(self) -> str:
        return (
            f"http://{self.group_launch.authority.host}:"
            f"{self.group_launch.authority.port}"
        )

    @property
    def before_gpu_snapshot(self) -> CanonicalJsonProofBinding:
        return self._before_gpu_snapshot

    @property
    def ready_gpu_snapshot(self) -> CanonicalJsonProofBinding:
        return self._ready_gpu_snapshot

    @property
    def server_log_path(self) -> str:
        return self._server_log_path

    @property
    def server_stdout_path(self) -> str:
        return self._server_stdout_path

    @property
    def server_stderr_path(self) -> str:
        return self._server_stderr_path

    async def _run_contract(self) -> None:
        traces = tuple(
            SessionLiveTraceInput(binding=row.binding, driver=row)
            for row in self._trace_drivers
        )
        timeout = sum(item.timeout_seconds for item in self.prepared) + 300.0
        async with asyncio.timeout(timeout):
            self._contract_result = await run_session_live_contract(
                session_plan_sha256=self.execution.plan.session_plan_sha256,
                traces=traces,
                base_url=self.base_url,
                request_timeout_s=max(item.timeout_seconds for item in self.prepared),
                abort_timeout_s=_ABORT_TIMEOUT_SECONDS,
                transport=self.transport,
                provider=self.provider,
                process_owner=self,
                verified_gpu_proof=None,
                terminal_observer=self,
            )

    async def _observe_server_policy_once(self) -> None:
        async with self._policy_lock:
            if self._policy_observed:
                return
            observed = await _observe_live_server_execution_policy(
                transport=self.transport,
                config=self.group_launch.run_config,
            )
            self._policy_observed = True
            publish_canonical_json_no_replace(
                self.evidence_root / "server-execution-policy.json",
                {"fields_json": observed[0], "fields_sha256": observed[1]},
            )

    async def terminal_finalized(
        self,
        *,
        trace_index: int,
        binding: NativeTerminalRunBinding,
        terminal: ValidatedNativeTerminalEvidence,
    ) -> None:
        if not 0 <= trace_index < len(self._trace_drivers):
            raise ValueError("resident terminal observer index differs")
        driver = self._trace_drivers[trace_index]
        if binding != driver.binding:
            raise ValueError("resident terminal observer binding differs")
        driver.accept_terminal(terminal)

    def _find_step(self, step: str, index: int | None = None) -> SessionLiveStepBinding:
        plan_sha = (
            None
            if index is None
            else self._trace_drivers[index].binding.execution_plan_sha256
        )
        rows = tuple(
            row
            for row in self.transport.live_steps
            if row.step == step and row.execution_plan_sha256 == plan_sha
        )
        if len(rows) != 1:
            raise RuntimeError(f"resident source step coverage differs: {step}:{index}")
        return rows[0]

    def _publish_source_step(
        self, step: str, index: int | None = None
    ) -> CanonicalJsonProofBinding:
        key = (step, index)
        cached = self._source_bindings.get(key)
        if cached is not None:
            return cached
        suffix = "session" if index is None else f"{index + 1:04d}"
        destination = self.evidence_root / "source" / f"{step}-{suffix}.json"
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        value = _step_value(self._find_step(step, index))
        publish_canonical_json_no_replace(destination, value)
        binding = CanonicalJsonProofBinding.bind(destination)
        self._source_bindings[key] = binding
        return binding

    async def wait_initial_reset_ready(self) -> None:
        await self._wait_event_or_contract(self._trace_drivers[0].reset_ready)
        capability_value = _step_value(self._find_step("session_capability"))
        initial_value = _step_value(self._find_step("session_initial_state"))
        capability = SourceOwnedSessionCapability.parse(
            capability_value,
            session_plan_sha256=self.execution.plan.session_plan_sha256,
            execution_plan_sha256s=tuple(
                row.binding.execution_plan_sha256 for row in self._trace_drivers
            ),
        )
        initial = SourceOwnedInitialStateReceipt.parse(
            initial_value, capability=capability
        )
        if capability.process_identity != f"scheduler:{self.process.pid}":
            raise RuntimeError("resident source capability names another process")
        self._source_chain.record_capability(capability_value)
        self._source_chain.record_initial_state(initial_value)
        self._parsed_capability = capability
        self._parsed_initial = initial
        self._accounting = initial.state.connection_accounting
        self._reset_generation = initial.state.reset_generation
        self._native_process_started_ns = capability.process_started_ns

    async def _wait_event_or_contract(self, event: asyncio.Event) -> None:
        if event.is_set():
            return
        waiter = asyncio.create_task(event.wait())
        try:
            done, _pending = await asyncio.wait(
                {waiter, self._contract_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if waiter in done and event.is_set():
                return
            await self._contract_task
            if not event.is_set():
                detail = (
                    "without a contract result"
                    if self._contract_result is None
                    else (
                        f"with {self._contract_result.audit.status}: "
                        f"{self._contract_result.audit.reason}"
                    )
                )
                raise RuntimeError(
                    "resident session ended before the requested boundary " + detail
                )
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    def _parse_reset(self, index: int) -> SourceOwnedResetReceipt:
        if (
            self._parsed_capability is None
            or self._parsed_initial is None
            or self._accounting is None
            or self._reset_generation is None
            or index != len(self._parsed_resets)
        ):
            raise RuntimeError("resident reset parser lineage is incomplete")
        reset = SourceOwnedResetReceipt.parse(
            _step_value(self._find_step("session_reset_boundary", index)),
            capability=self._parsed_capability,
            prior_execution_plan_sha256=(
                None
                if index == 0
                else self._trace_drivers[index - 1].binding.execution_plan_sha256
            ),
            next_execution_plan_sha256=(
                self._trace_drivers[index].binding.execution_plan_sha256
            ),
            initial_state_receipt_sha256=(
                self._parsed_initial.initial_state_receipt_sha256
            ),
            clean_state_sha256=self._parsed_initial.state.clean_state_sha256,
            expected_reset_generation=self._reset_generation,
            prior_accounting=self._accounting,
        )
        self._parsed_resets.append(reset)
        self._reset_generation = reset.after.reset_generation
        self._accounting = reset.after.connection_accounting
        return reset

    def _ensure_reset_recorded(
        self, index: int
    ) -> tuple[SourceOwnedResetReceipt, CanonicalJsonProofBinding]:
        if index < len(self._parsed_resets):
            return self._parsed_resets[index], self._source_reset_bindings[index]
        if index != len(self._parsed_resets):
            raise RuntimeError("resident reset recording skipped an epoch")
        value = _step_value(self._find_step("session_reset_boundary", index))
        reset = self._parse_reset(index)
        binding = self._source_chain.record_reset(value)
        self._source_reset_bindings.append(binding)
        return reset, binding

    def _combined_source(self, step: str, index: int) -> dict[str, object]:
        value = _step_value(self._find_step(step, index))
        source = value.get("source_owned_session")
        if type(source) is not dict:
            raise TypeError("resident atomic source response is incomplete")
        return source

    def _parse_trace_source(self, index: int) -> SourceOwnedTraceReceipt:
        if (
            self._parsed_capability is None
            or self._accounting is None
            or index != len(self._parsed_traces)
            or len(self._parsed_resets) != index + 1
        ):
            raise RuntimeError("resident source trace parser lineage is incomplete")
        ready = self._combined_source("atomic_trace_reset", index)
        warmup = SourceOwnedWarmupReceipt.parse(
            ready.get("warmup_receipt"),
            execution_plan_sha256=self._trace_drivers[
                index
            ].binding.execution_plan_sha256,
            prior_accounting=self._accounting,
        )
        clock = SourceOwnedScoredClockReceipt.parse(
            ready.get("clock_receipt"),
            execution_plan_sha256=self._trace_drivers[
                index
            ].binding.execution_plan_sha256,
            warmup=warmup,
            prior_clock_generation=self._clock_generation,
        )
        trace = SourceOwnedTraceReceipt.parse(
            self._combined_source("atomic_trace_finalize", index),
            execution_plan_sha256=self._trace_drivers[
                index
            ].binding.execution_plan_sha256,
            clock=clock,
            prior_accounting=warmup.connection_accounting,
        )
        terminal = self._trace_drivers[index].terminal
        if (
            terminal is None
            or trace.terminal_receipt_sha256 != terminal.terminal_sha256
        ):
            raise RuntimeError("resident source trace is not bound to its terminal")
        self._parsed_warmups.append(warmup)
        self._parsed_clocks.append(clock)
        self._parsed_traces.append(trace)
        self._clock_generation = clock.clock_generation
        self._accounting = trace.connection_accounting
        for step in (
            "native_terminal_capability",
            "atomic_trace_begin",
            "atomic_trace_reset",
            "atomic_trace_finalize",
        ):
            self._publish_source_step(step, index)
        return trace

    async def reset_member(
        self, *, member: FormalServingSessionGroupSpec, member_index: int
    ) -> FormalServingResidentResetEvidence:
        if (
            not 0 <= member_index < len(self.prepared)
            or self.prepared[member_index].spec != member
        ):
            raise ValueError("resident reset member leaves prepared group")
        driver = self._trace_drivers[member_index]
        await self._wait_event_or_contract(driver.reset_ready)
        reset, binding = self._ensure_reset_recorded(member_index)
        prior_finished = (
            self._server_ready_ns
            if member_index == 0
            else self._trace_drivers[member_index - 1].trace_finished_ns
        )
        if prior_finished is None or driver.reset_ready_ns is None:
            raise RuntimeError("resident reset host timing is incomplete")
        return FormalServingResidentResetEvidence(
            source_reset_receipt=binding,
            reset_started_ns=prior_finished,
            reset_finished_ns=driver.reset_ready_ns,
            hbm_allocated_bytes=reset.after.allocator_allocated_bytes,
            request_queue_empty=(
                reset.after.active_requests == 0 and reset.after.queued_requests == 0
            ),
            optimizer_state_reset=True,
            adaptation_state_reset=True,
            candidate_state_reset=True,
            cache_policy_restored=(
                reset.after.clean_state_sha256
                == self._parsed_initial.state.clean_state_sha256
                if self._parsed_initial is not None
                else False
            ),
            terminal_writer_flushed=reset.after.completion_event_complete,
            previous_requests_fully_terminal=(
                member_index == 0 or len(self._parsed_traces) == member_index
            ),
        )

    async def execute_trace(
        self,
        *,
        member: FormalServingSessionGroupSpec,
        member_index: int,
        effective_terminal_binding: NativeTerminalRunBinding,
    ) -> FormalServingResidentTraceEvidence:
        if (
            not 0 <= member_index < len(self.prepared)
            or self.prepared[member_index].spec != member
            or self._trace_drivers[member_index].binding != effective_terminal_binding
            or len(self._parsed_resets) != member_index + 1
        ):
            raise ValueError("resident trace leaves prepared/reset group")
        driver = self._trace_drivers[member_index]
        driver.execute_permit.set()
        await self._wait_event_or_contract(driver.terminal_ready)
        if member_index + 1 < len(self._trace_drivers):
            try:
                await self._wait_event_or_contract(
                    self._trace_drivers[member_index + 1].reset_ready
                )
            except RuntimeError:
                # A failed following reset is compatible with a terminally
                # complete prefix.  Its reset call will select fresh fallback.
                result = self._contract_result
                if (
                    result is None
                    or len(result.audit.trace_receipt_sha256s) <= member_index
                ):
                    raise
        else:
            await self._contract_task
        self._parse_trace_source(member_index)
        evidence = self._publish_trace_evidence(member_index)
        if (
            member_index + 1 < len(self._trace_drivers)
            and self._trace_drivers[member_index + 1].reset_ready.is_set()
        ):
            # The background source contract reaches the following reset before
            # the group worker asks for it.  Seal that reset now, after the
            # current terminal, so a crash cannot lose the actual boundary.
            self._ensure_reset_recorded(member_index + 1)
        return evidence

    def _publish_trace_evidence(self, index: int) -> FormalServingResidentTraceEvidence:
        prepared = self.prepared[index]
        driver = self._trace_drivers[index]
        terminal = driver.terminal
        warmup = driver.warmup_result
        scored = driver.scored_result
        if (
            terminal is None
            or warmup is None
            or scored is None
            or driver.trace_started_ns is None
            or driver.trace_finished_ns is None
        ):
            raise RuntimeError("resident trace scientific evidence is incomplete")
        terminal_binding = publish_scalable_native_terminal_artifact(
            output_path=prepared.plan.terminal_output_path,
            legacy_artifact=terminal.to_artifact(warmup_requests=warmup.requests),
        )
        ready_source = self._combined_source("atomic_trace_reset", index)
        self._source_chain.record_warmup(ready_source.get("warmup_receipt"))
        self._source_chain.record_scored_clock(ready_source.get("clock_receipt"))
        self._source_chain.record_trace(
            self._combined_source("atomic_trace_finalize", index),
            terminal_artifact=terminal_binding,
        )
        pointers = scored.validate(
            expected_phase="scored", bound_requests=prepared.scored
        )
        itl_binding = publish_scalable_unsigned_native_itl_bundle(
            output_path=prepared.plan.native_itl_pointer_output_path,
            legacy_bundle={
                "schema_version": 1,
                "kind": "unsigned_native_itl_result_pointer_bundle",
                "run_binding_sha256": canonical_sha256(driver.binding.begin_payload()),
                "terminal_artifact_raw_sha256": terminal_binding.raw_sha256,
                "terminal_artifact_semantic_sha256": (terminal_binding.semantic_sha256),
                "scored_request_inputs_sha256": canonical_sha256(
                    [request.sha256 for request in prepared.scored]
                ),
                "native_result_pointers": [dict(value) for value in pointers],
            },
        )
        validated_terminal = validate_native_terminal_artifact(
            terminal_binding.reopen(),
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            expected_binding=driver.binding,
        )
        terminal_outputs = {
            row.request_id: row.output_token_ids
            for row in validated_terminal.requests
            if row.submitted_to_server
            and row.terminal_status == "completed"
            and row.output_token_ids is not None
        }
        validate_unsigned_native_itl_pointer_bundle(
            itl_binding,
            expected_binding=driver.binding,
            expected_terminal_artifact=terminal_binding,
            expected_scored_request_inputs_sha256=canonical_sha256(
                [request.sha256 for request in prepared.scored]
            ),
            expected_terminal_output_tokens=terminal_outputs,
        )
        scored_started_ns, scored_finished_ns = _reopen_native_scored_interval(
            pointer_artifact=itl_binding,
            terminal_artifact=terminal_binding,
            binding=driver.binding,
            terminal_evidence=validated_terminal,
            scored_request_inputs_sha256=canonical_sha256(
                [request.sha256 for request in prepared.scored]
            ),
        )
        lifecycle_path = (
            Path(prepared.plan.private_output_root) / "client-request-lifecycle.json"
        )
        lifecycle_binding = publish_scalable_client_request_lifecycle(
            output_path=lifecycle_path,
            run_binding_sha256=canonical_sha256(driver.binding.begin_payload()),
            execution_policy_sha256=prepared.execution_policy.sha256,
            rows=[*warmup.client_lifecycle_rows, *scored.client_lifecycle_rows],
        )
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            _publish_formal_serving_junit,
        )

        junit = _publish_formal_serving_junit(
            output_path=prepared.plan.junit_output_path,
            topology_mode=prepared.plan.topology_mode,
            request_ids=tuple(row.request_id for row in prepared.scored),
        )
        lifecycle_value = {
            "schema_version": 1,
            "kind": "formal_serving_resident_trace_lifecycle",
            "group_id": self.execution.plan.group_id,
            "materialized_cell_id": prepared.plan.materialized_cell_id,
            "member_index": index,
            "session_epoch": index + 1,
            "run_binding_sha256": canonical_sha256(driver.binding.begin_payload()),
            "trace_started_ns": driver.trace_started_ns,
            "scored_started_ns": scored_started_ns,
            "scored_finished_ns": scored_finished_ns,
            "trace_finished_ns": max(driver.trace_finished_ns, scored_finished_ns),
            "terminal_artifact": terminal_binding.to_dict(),
            "native_itl": itl_binding.to_dict(),
            "client_lifecycle": lifecycle_binding.to_dict(),
            "formal_measured": False,
        }
        publish_canonical_json_no_replace(
            prepared.plan.lifecycle_timing_output_path, lifecycle_value
        )
        trace_lifecycle = CanonicalJsonProofBinding.bind(
            prepared.plan.lifecycle_timing_output_path
        )
        return FormalServingResidentTraceEvidence(
            effective_terminal_binding=driver.binding,
            raw_terminal=terminal_binding,
            native_itl=itl_binding,
            client_lifecycle=lifecycle_binding,
            junit=junit,
            trace_lifecycle=trace_lifecycle,
            trace_started_ns=driver.trace_started_ns,
            scored_started_ns=scored_started_ns,
            trace_finished_ns=max(driver.trace_finished_ns, scored_finished_ns),
        )

    async def close(self) -> None:
        await self._terminate(force=False)

    async def force_close(self) -> None:
        await self._terminate(force=True)

    async def _terminate(self, *, force: bool) -> None:
        async with self._termination_lock:
            if self._termination is not None:
                return
            if force and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
            exit_code, cleanup, exited_ns = await asyncio.to_thread(
                _terminate_process_group, self.process
            )
            checked_ns = time.monotonic_ns()
            self._termination = (exit_code, cleanup, exited_ns, checked_ns)

    def _parse_close(self) -> SourceOwnedCloseReceipt:
        if (
            self._parsed_capability is None
            or self._parsed_initial is None
            or self._accounting is None
            or len(self._parsed_traces) != len(self.prepared)
        ):
            raise RuntimeError("resident close source lineage is incomplete")
        close = SourceOwnedCloseReceipt.parse(
            _step_value(self._find_step("session_close_terminal")),
            capability=self._parsed_capability,
            prior_accounting=self._accounting,
            initial_state_receipt_sha256=(
                self._parsed_initial.initial_state_receipt_sha256
            ),
            execution_plan_sha256s=tuple(
                row.binding.execution_plan_sha256 for row in self._trace_drivers
            ),
            reset_receipt_sha256s=tuple(
                row.reset_receipt_sha256 for row in self._parsed_resets
            ),
            warmup_receipt_sha256s=tuple(
                row.warmup_receipt_sha256 for row in self._parsed_warmups
            ),
            clock_receipt_sha256s=tuple(
                row.clock_receipt_sha256 for row in self._parsed_clocks
            ),
            trace_receipt_sha256s=tuple(
                row.trace_receipt_sha256 for row in self._parsed_traces
            ),
            terminal_receipt_sha256s=tuple(
                row.terminal_receipt_sha256 for row in self._parsed_traces
            ),
        )
        return close

    async def close_session(self, *, force: bool) -> FormalServingResidentCloseEvidence:
        close_started_ns = time.monotonic_ns()
        if force and not self._contract_task.done():
            self._contract_task.cancel()
        await asyncio.gather(self._contract_task, return_exceptions=True)
        await self._terminate(force=force)
        assert self._termination is not None
        source_close: CanonicalJsonProofBinding | None = None
        if not force and self._contract_result is not None:
            close_value = _step_value(self._find_step("session_close_terminal"))
            self._parse_close()
            source_chain = self._source_chain.record_close(close_value)
            source_close = source_chain.manifest
        for stream in (self._server_log, self._server_stdout, self._server_stderr):
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
        after = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=self.nvidia_smi_tool,
            gpu_uuids=self.group_launch.source_launch.gpu_uuids,
            inventory_sha256=self.group_launch.source_launch.inventory_sha256,
            phase="after",
            output_path=self.evidence_root / "after-gpu-snapshot.json",
        )
        evidence_flush_completed_ns = time.monotonic_ns()
        exit_code, cleanup, exited_ns, checked_ns = self._termination
        return FormalServingResidentCloseEvidence(
            source_close_receipt=source_close,
            server_process_id=self.process.pid,
            server_process_group_id=self.process.pid,
            close_started_ns=min(close_started_ns, exited_ns),
            process_exited_ns=exited_ns,
            process_exit_code=exit_code,
            process_group_empty=True,
            process_group_empty_checked_ns=checked_ns,
            evidence_flush_completed_ns=evidence_flush_completed_ns,
            cleanup_kind=(
                "forced_sigterm"
                if force
                else "already_exited_clean"
                if cleanup == "already_exited_clean"
                else "source_close_sigterm"
            ),
            after_gpu_snapshot=after,
            server_log=EvidenceFileBinding.bind(
                Path(self._server_log_path), label="resident server log"
            ),
            server_stdout=EvidenceFileBinding.bind(
                Path(self._server_stdout_path), label="resident server stdout"
            ),
            server_stderr=EvidenceFileBinding.bind(
                Path(self._server_stderr_path), label="resident server stderr"
            ),
        )


class PinnedSglangResidentProcessFactory(FormalServingResidentProcessFactory):
    """Production factory using the exact launch and fresh runner primitives."""

    def __init__(
        self,
        *,
        nvidia_smi_tool: PinnedNvidiaSmiTool,
        repository_root: str | Path,
    ) -> None:
        if type(nvidia_smi_tool) is not PinnedNvidiaSmiTool:
            raise TypeError("resident SGLang factory requires pinned nvidia-smi")
        nvidia_smi_tool.revalidate()
        self._nvidia_smi_tool = nvidia_smi_tool
        repository = Path(repository_root).resolve()
        if not (repository / ".git").exists():
            raise ValueError("resident SGLang factory repository is not a Git checkout")
        self._repository_root = repository

    async def launch(
        self,
        *,
        execution: RevalidatedFormalServingSessionGroupExecution,
        evidence_root: Path,
    ) -> PinnedSglangResidentProcessDriver:
        if not execution.plan.members:
            raise ValueError("resident SGLang group is empty")
        prepared = tuple(
            _prepare_resident_member(row) for row in execution.plan.members
        )
        root = evidence_root.resolve()
        if root.exists():
            if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
                raise FileExistsError("resident SGLang evidence root is not empty")
        else:
            root.mkdir(parents=True, mode=0o700)
        group_launch = publish_formal_serving_resident_group_launch_authority(
            execution=execution,
            output_root=root / "group-launch",
        )
        launch = group_launch.source_launch
        _require_port_unused(group_launch.authority.port)
        before = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=self._nvidia_smi_tool,
            gpu_uuids=launch.gpu_uuids,
            inventory_sha256=launch.inventory_sha256,
            phase="before",
            output_path=root / "before-gpu-snapshot.json",
        )
        log = _open_evidence_stream(
            root / "server.log", header="resident pinned SGLang shared session"
        )
        stdout = _open_evidence_stream(
            root / "server.stdout", header="resident pinned SGLang stdout"
        )
        stderr = _open_evidence_stream(
            root / "server.stderr", header="resident pinned SGLang stderr"
        )
        process: subprocess.Popen[bytes] | None = None
        try:
            process = await asyncio.to_thread(
                _spawn_resident_server,
                actual_server_argv=group_launch.authority.actual_server_argv,
                patched_sglang_checkout=(
                    group_launch.authority.patched_sglang_checkout
                ),
                actual_child_environment=(
                    group_launch.authority.actual_child_environment
                ),
                stdout_file=stdout,
                stderr_file=stderr,
            )
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError("resident SGLang server lacks an independent PGID")
            await asyncio.to_thread(
                _wait_server_ready,
                process,
                port=group_launch.authority.port,
                timeout_seconds=min(
                    max(item.timeout_seconds for item in prepared),
                    _SERVER_READY_TIMEOUT_SECONDS,
                ),
            )
            ready_ns = time.monotonic_ns()
            ready = await asyncio.to_thread(
                _capture_gpu_process_snapshot,
                tool=self._nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="ready",
                output_path=root / "ready-gpu-snapshot.json",
                expected_server_process_group_ids=(process.pid,),
            )
            transport = _resident_transport_from_launch(launch)
            driver = PinnedSglangResidentProcessDriver(
                execution=execution,
                prepared=prepared,
                evidence_root=root,
                nvidia_smi_tool=self._nvidia_smi_tool,
                process=process,
                group_launch=group_launch,
                transport=transport,
                before_gpu_snapshot=before,
                ready_gpu_snapshot=ready,
                server_ready_ns=ready_ns,
                server_log=log,
                server_stdout=stdout,
                server_stderr=stderr,
            )
            await driver.wait_initial_reset_ready()
            return driver
        except BaseException as error:
            if process is not None:
                try:
                    await asyncio.to_thread(_terminate_process_group, process)
                except BaseException as cleanup_error:  # noqa: BLE001
                    error.add_note(
                        "resident launch cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            for stream in (log, stdout, stderr):
                if not stream.closed:
                    stream.close()
            raise

    async def execute_fresh_member(
        self, *, member: FormalServingSessionGroupSpec, fallback_reason: str
    ) -> FormalServingSessionMemberPhysicalResult:
        """Execute an unstarted remainder member through the unchanged TP1 runner."""

        if not fallback_reason:
            raise ValueError("resident fresh fallback requires its failure reason")
        prepared = _prepare_resident_member(member)
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            execute_formal_single_operator_serving_run_plan,
        )
        from lightcone_spec.orchestration.live_sglang import (
            ValidatedUnsignedPinnedSglangServingRun,
        )
        from lightcone_spec.runtime.formal_single_operator import (
            finalize_formal_single_operator_run,
        )

        result = await execute_formal_single_operator_serving_run_plan(
            plan_path=member.run_plan.absolute_path,
            nvidia_smi_tool=self._nvidia_smi_tool,
        )
        if type(result) is not ValidatedUnsignedPinnedSglangServingRun:
            raise TypeError("resident fresh fallback left ordinary TP1")
        manifest = finalize_formal_single_operator_run(
            repository_root=self._repository_root,
            run_plan_path=member.run_plan.absolute_path,
        )
        pointer = CanonicalJsonProofBinding.bind(
            Path(prepared.plan.private_output_root)
            / "formal-single-operator-manifest.json"
        )
        return FormalServingSessionMemberPhysicalResult(
            status="COMPLETE",
            process_id=result.receipt.server_process_id,
            started_ns=manifest.started_ns,
            finished_ns=manifest.finished_ns,
            exit_code=0,
            result_pointer=pointer,
            failure_code=None,
        )


__all__ = (
    "PinnedSglangResidentProcessDriver",
    "PinnedSglangResidentProcessFactory",
)
