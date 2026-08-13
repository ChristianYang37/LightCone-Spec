"""Content-bound execution of one materialised industrial serving cell.

Planning is separate from execution.  The planner validates every local
artifact and emits immutable server/bench argv without starting a process.
The async runner performs network and process mutation only through injected
interfaces, accounts every offered request, and publishes evidence only via
the repository's durable :class:`EvidenceWriter` terminal-receipt protocol.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from itertools import pairwise
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

from lightcone_spec import PINNED_SGLANG_PATCH_COUNT, PINNED_SGLANG_TREE
from lightcone_spec.adaptation.plan_authority import (
    TrainablePlanAuthorityBinding,
    TrainablePlanRawJsonBinding,
    require_trainable_plan_authority_for_method,
    trainable_plan_authority_binding_to_dict,
)
from lightcone_spec.config import run_config_sha256
from lightcone_spec.config.schema import RunConfig
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.failure_authority import (
    FAILURE_INJECTION_RAW_PLAN_AUTHORITY_REQUIRED_REASON,
    FailureExecutionAuthorityToken,
    FailureInjectionAuthorityBlocked,
    require_failure_execution_lifecycle,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuDispatchExecutionContext,
    GpuDispatchPlan,
    GpuInventory,
    validate_dispatch_plan_for_execution,
)
from lightcone_spec.experiments.itl_authority import (
    E2ItlTimestampPlan,
    release_e2_itl_timestamp_plan,
    require_e2_itl_timestamp_prelaunch,
)
from lightcone_spec.experiments.load import (
    LoadAccounting,
    ProductionLoadPlan,
    RequestCorpus,
    RequestOutcome,
    TimingCoverage,
    TokenChunkTiming,
    account_scored_requests,
    evaluate_token_timing,
)
from lightcone_spec.experiments.planning import (
    BudgetJobKind,
    BudgetObservationReceipt,
    BudgetPlan,
    ExperimentBudget,
    P99AnchorStatus,
)
from lightcone_spec.experiments.registry import (
    CellStatus,
    ExperimentCell,
    ExperimentReceipt,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.serving import (
    BenchServingResult,
    BenchServingTransport,
    BoundServingRequest,
    PinnedBenchServingTransport,
    official_bench_argv,
)
from lightcone_spec.experiments.stage_activation import (
    is_serving_interference_calibration_cell,
)
from lightcone_spec.locking.prepared_models import (
    PreparedModelContentAuthorityBlocked,
    has_prepared_model_content_release_manifest_sha256,
)
from lightcone_spec.orchestration.industrial import (
    IndustrialPhysicalAssignment,
    IndustrialRuntimePlan,
    _require_registered_e1_execution_recipe,
    validate_industrial_execution_semantics_authority,
)
from lightcone_spec.orchestration.native_terminal import (
    NATIVE_TERMINAL_EVIDENCE_FIELDS,
    NATIVE_TERMINAL_EVIDENCE_HOOK,
    NativeTerminalProvider,
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    ValidatedNativeTerminalEvidence,
    validate_native_terminal_artifact,
)
from lightcone_spec.orchestration.runtime import (
    ServerLaunch,
    _execution_argv,
    _execution_role,
    _immutable_json,
    _render_server,
    _runtime_execution_policy,
)
from lightcone_spec.orchestration.session import (
    SHARED_SESSION_UNAVAILABLE_REASON,
    SessionExecutionBinding,
    SessionExecutionLifecycle,
    SharedSessionUnavailableError,
)
from lightcone_spec.runtime.attestation import (
    NO_TRUSTED_ATTESTERS,
    TrustedAttesterPolicy,
    require_release_trusted_attester_policy,
)
from lightcone_spec.runtime.compile_cache import (
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    CompileCacheLaunchPlan,
    preflight_compile_cache_launch,
)
from lightcone_spec.sglang_bridge.checkout import verify_patched_checkout
from lightcone_spec.sglang_bridge.config import sglang_adaptation_payload
from lightcone_spec.telemetry.records import (
    OUTPUT_HASH_FORMAT,
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
    UpdateRecord,
)
from lightcone_spec.telemetry.writer import (
    DEFAULT_EVIDENCE_WRITER_POLICY,
    EvidenceWriter,
    EvidenceWriterPolicy,
    evidence_writer_policy_from_receipt,
    load_completed_evidence,
    publish_prepared_evidence_completion,
)

MISSING_NATIVE_EVIDENCE_REASON = "missing_content_bound_native_speculative_evidence"
TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON = (
    "trusted_native_terminal_attester_unavailable"
)
PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON = (
    "prepared_model_content_release_manifest_pin_unavailable"
)
TRAINABLE_PLAN_RAW_AUTHORITY_UNAVAILABLE_REASON = (
    "trainable_plan_raw_authority_unavailable"
)
MAX_IN_MEMORY_REQUEST_EXECUTIONS = 100_000

_GPU_INVENTORY_QUERY = (
    "index,uuid,name,memory.total,driver_version,compute_cap,pci.bus_id,"
    "power.limit,temperature.gpu.tlimit,clocks.max.sm,persistence_mode"
)
_GPU_INVENTORY_ARGV = (
    "nvidia-smi",
    f"--query-gpu={_GPU_INVENTORY_QUERY}",
    "--format=csv,noheader,nounits",
)
_GPU_PROCESS_ARGV = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name",
    "--format=csv,noheader,nounits",
)
_GPU_TOPOLOGY_ARGV = ("nvidia-smi", "topo", "-m")
_NVCC_RELEASE = re.compile(r"\brelease\s+(\d+\.\d+)\b")

_BUDGET_OBSERVATION_COMPONENTS = (
    "startup_model_load",
    "compile_jit_graph_prewarm",
    "excluded_warmup",
    "scored_arrival",
    "drain",
    "reset_finalization",
    "evidence_flush_shutdown",
    "soak",
    "failure_injection",
    "retry",
    "profiler",
    "download_compile_reservation",
)
_STRUCTURALLY_ABSENT_SERVING_COMPONENTS = (
    "compile_jit_graph_prewarm",
    "download_compile_reservation",
)
_SCORING_COMPONENT_BY_JOB = {
    BudgetJobKind.STANDARD: "scored_arrival",
    BudgetJobKind.SHORT: "scored_arrival",
    BudgetJobKind.P99_ANCHOR: "scored_arrival",
    BudgetJobKind.SOAK: "soak",
    BudgetJobKind.FAILURE: "failure_injection",
    BudgetJobKind.PROFILER: "profiler",
}
_SCORING_COMPONENTS = (
    "scored_arrival",
    "soak",
    "failure_injection",
    "profiler",
)
_BUDGET_OBSERVATION_KIND = "industrial_budget_observation_receipt_v1"
_RESERVED_GANG_MEASUREMENT = "exclusive_reserved_gang_wall_ms_x_gpu_count"
_WHOLE_INSTANCE_BILLING = "whole_inventory_wall_clock_v1"

type EvidenceItem = (
    PerformanceRecord | RequestRecord | RoundRecord | RunRecord | UpdateRecord
)


class _AsyncEvidenceSink:
    """Bounded single-writer bridge from the event loop to durable WAL."""

    def __init__(
        self,
        writer: EvidenceWriter,
        *,
        max_queued_rows: int = 1024,
        max_batch_rows: int | None = None,
    ) -> None:
        if max_batch_rows is None:
            max_batch_rows = min(128, max_queued_rows)
        if (
            isinstance(max_batch_rows, bool)
            or not isinstance(max_batch_rows, int)
            or max_batch_rows < 1
            or max_batch_rows > max_queued_rows
        ):
            raise ValueError("evidence batch rows must lie within the queue bound")
        self._writer = writer
        self._queue: asyncio.Queue[EvidenceItem | object] = asyncio.Queue(
            maxsize=max_queued_rows
        )
        self._stop = object()
        self._error: BaseException | None = None
        self._closed = False
        self._backpressure_events = 0
        self._max_batch_rows = max_batch_rows
        self._overflow_lock = asyncio.Lock()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            batch = [await self._queue.get()]
            while len(batch) < self._max_batch_rows:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            terminal = self._stop in batch
            if terminal and batch[-1] is not self._stop:
                self._error = RuntimeError("evidence stop marker was not terminal")
            records = tuple(item for item in batch if item is not self._stop)
            try:
                if self._error is not None:
                    raise self._error
                if records and not await asyncio.to_thread(self._write_batch, records):
                    raise RuntimeError("bounded evidence writer dropped a row")
            except BaseException as error:  # noqa: BLE001 - background boundary
                self._error = error
                while not self._queue.empty():
                    self._queue.get_nowait()
                    self._queue.task_done()
                return
            finally:
                for _ in batch:
                    self._queue.task_done()
            if terminal:
                return

    def _write_batch(self, records: tuple[EvidenceItem, ...]) -> bool:
        return all(self._write_one(record, flush_after=False) for record in records)

    def _write_one(self, record: EvidenceItem, *, flush_after: bool) -> bool:
        written = self._writer.write(record)
        if written and flush_after:
            self._writer.flush()
        return written

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("background evidence writer failed") from self._error

    @property
    def backpressure_events(self) -> int:
        return self._backpressure_events

    async def write(self, record: EvidenceItem) -> None:
        if self._closed:
            raise RuntimeError("background evidence sink is closed")
        self._raise_error()
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull as error:
            self._backpressure_events += 1
            # Saturation makes the run nonclaimable, so measured-path latency no
            # longer matters.  Drain the bounded queue and durably persist the
            # triggering terminal row before aborting; a negative row is never
            # silently dropped to preserve zero-think timing.
            async with self._overflow_lock:
                await self.flush()
                await self._queue.put(record)
                await self.flush()
            raise RuntimeError(
                "background evidence queue saturated; run is nonclaimable"
            ) from error
        self._raise_error()

    async def flush(self) -> None:
        await self._queue.join()
        self._raise_error()
        await asyncio.to_thread(self._writer.flush)
        self._raise_error()

    async def close(self) -> None:
        if self._closed:
            self._raise_error()
            return
        if not self._task.done():
            await self._queue.put(self._stop)
        await self._task
        self._closed = True
        self._raise_error()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _elapsed_milliseconds(start_ns: int, end_ns: int) -> int:
    """Quantize one internally observed monotonic interval without undercounting."""

    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (start_ns, end_ns)
    ):
        raise RuntimeError("monotonic budget boundaries must be integer nanoseconds")
    if end_ns < start_ns:
        raise RuntimeError("monotonic budget boundary moved backwards")
    elapsed_ns = end_ns - start_ns
    return (elapsed_ns + 999_999) // 1_000_000


def _scenario_is_explicit_zero(value: object) -> bool:
    """Require zero in every planning scenario, not only the registered row."""

    return all(
        not isinstance(component, bool)
        and isinstance(component, int)
        and component == 0
        for component in (
            getattr(value, "optimistic", None),
            getattr(value, "registered", None),
            getattr(value, "quota_envelope", None),
        )
    )


def _initial_budget_observations(plan: IndustrialExecutionPlan) -> dict[str, int]:
    """Bind inactive phases and this first attempt's explicit zero retry."""

    observations: dict[str, int] = {}
    active_scoring = _SCORING_COMPONENT_BY_JOB.get(plan.budget.job_kind)
    if active_scoring is None:
        raise ValueError("compile/download budgets cannot enter the serving executor")
    for name in _STRUCTURALLY_ABSENT_SERVING_COMPONENTS:
        if not _scenario_is_explicit_zero(getattr(plan.budget, name)):
            raise ValueError(
                f"serving executor cannot observe registered budget component {name}"
            )
        observations[name] = 0
    for name in _SCORING_COMPONENTS:
        if name == active_scoring:
            continue
        if not _scenario_is_explicit_zero(getattr(plan.budget, name)):
            raise ValueError(
                f"inactive {name} budget must be registered explicitly as zero"
            )
        observations[name] = 0
    # Retry is a multi-attempt quota envelope.  This executor is one immutable
    # first attempt; scheduler-level aggregation must add a separately bound
    # prior-attempt observation before a retry can be claimed.
    observations["retry"] = 0
    return observations


def _budget_observation_artifact(
    observation: BudgetObservationReceipt,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": _BUDGET_OBSERVATION_KIND,
        "experiment_budget_sha256": observation.budget.sha256,
        "budget_observation_sha256": observation.sha256,
        "budget": asdict(observation.budget),
        "observed_component_ms": [
            [name, value] for name, value in observation.observed_component_ms
        ],
        "measured_gpu_ms": observation.measured_gpu_ms,
        "fixed_instance_billed_gpu_ms": observation.fixed_instance_billed_gpu_ms,
        "terminal_evidence_sha256": observation.terminal_evidence_sha256,
        "observed_wall_ms": observation.observed_wall_ms,
        "registered_wall_delta_ms": observation.registered_wall_delta_ms,
        "registered_gpu_delta_ms": observation.registered_gpu_delta_ms,
        "registered_billed_delta_ms": observation.registered_billed_delta_ms,
        "gpu_measurement_semantics": _RESERVED_GANG_MEASUREMENT,
        "fixed_instance_billing_semantics": _WHOLE_INSTANCE_BILLING,
    }


def _budget_observation_directory(root: Path, run_id: str) -> Path:
    return root / f"{run_id}.rank0.budget-observation"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_budget_observation(
    *,
    root: Path,
    run_id: str,
    observation: BudgetObservationReceipt,
) -> tuple[Path, Path]:
    """Atomically publish one JSON receipt and semantic-digest sidecar directory."""

    final = _budget_observation_directory(root, run_id)
    if os.path.lexists(final):
        raise RuntimeError(f"budget observation already exists for run {run_id}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.budget-observation.", dir=root)
    )
    receipt = temporary / "observation.json"
    sidecar = temporary / "observation.json.sha256"
    body = (
        json.dumps(
            _budget_observation_artifact(observation),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    with receipt.open("x", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(observation.sha256 + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(temporary)
    if os.path.lexists(final):
        raise RuntimeError(f"budget observation already exists for run {run_id}")
    os.rename(temporary, final)
    _fsync_directory(root)
    return final / receipt.name, final / sidecar.name


def _load_budget_observation(
    *,
    root: Path,
    run_id: str,
    plan: IndustrialExecutionPlan,
    terminal_receipt: Path,
) -> tuple[BudgetObservationReceipt, Path, Path]:
    directory = _budget_observation_directory(root, run_id)
    receipt_path = directory / "observation.json"
    sidecar_path = directory / "observation.json.sha256"
    if (
        terminal_receipt.is_symlink()
        or not terminal_receipt.is_file()
        or directory.is_symlink()
        or not directory.is_dir()
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or sidecar_path.is_symlink()
        or not sidecar_path.is_file()
    ):
        raise RuntimeError("completed industrial evidence lacks a budget observation")
    try:
        terminal_value = json.loads(terminal_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("industrial terminal evidence is not valid JSON") from error
    if type(terminal_value) is not dict:
        raise RuntimeError("industrial terminal evidence is malformed")
    prepared_binding = terminal_value.get("prepared_receipt_sha256")
    post_binding_fields = {
        "prepared_receipt_name",
        "prepared_receipt_sha256",
        "prepared_receipt_size",
        "budget_observation",
    }
    present_post_bindings = post_binding_fields.intersection(terminal_value)
    is_prepared_receipt = terminal_receipt.name == f"{run_id}.rank0.prepared.json"
    is_canonical_receipt = terminal_receipt.name == f"{run_id}.rank0.complete.json"
    if (
        not (is_prepared_receipt or is_canonical_receipt)
        or (is_prepared_receipt and present_post_bindings)
        or (is_canonical_receipt and present_post_bindings != post_binding_fields)
        or (
            is_canonical_receipt
            and (type(prepared_binding) is not str or not _is_sha256(prepared_binding))
        )
    ):
        raise RuntimeError("industrial terminal evidence has an invalid post-binding")
    observed_evidence_sha256 = (
        prepared_binding
        if present_post_bindings == post_binding_fields
        else _file_sha256(terminal_receipt)
    )
    try:
        artifact = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("budget observation is not valid JSON") from error
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "experiment_budget_sha256",
        "budget_observation_sha256",
        "budget",
        "observed_component_ms",
        "measured_gpu_ms",
        "fixed_instance_billed_gpu_ms",
        "terminal_evidence_sha256",
        "observed_wall_ms",
        "registered_wall_delta_ms",
        "registered_gpu_delta_ms",
        "registered_billed_delta_ms",
        "gpu_measurement_semantics",
        "fixed_instance_billing_semantics",
    }
    if type(artifact) is not dict or set(artifact) != expected_fields:
        raise RuntimeError("budget observation fields differ from the exact contract")
    if (
        type(artifact["schema_version"]) is not int
        or artifact["schema_version"] != 1
        or type(artifact["artifact_kind"]) is not str
        or artifact["artifact_kind"] != _BUDGET_OBSERVATION_KIND
        or type(artifact["gpu_measurement_semantics"]) is not str
        or artifact["gpu_measurement_semantics"] != _RESERVED_GANG_MEASUREMENT
        or type(artifact["fixed_instance_billing_semantics"]) is not str
        or artifact["fixed_instance_billing_semantics"] != _WHOLE_INSTANCE_BILLING
        or type(artifact["experiment_budget_sha256"]) is not str
        or type(artifact["budget_observation_sha256"]) is not str
        or type(artifact["terminal_evidence_sha256"]) is not str
    ):
        raise RuntimeError("budget observation identity fields are malformed")
    observed_rows = artifact["observed_component_ms"]
    if type(observed_rows) is not list or any(
        type(row) is not list
        or len(row) != 2
        or type(row[0]) is not str
        or type(row[1]) is not int
        for row in observed_rows
    ):
        raise RuntimeError("budget observation component rows are malformed")
    if type(artifact["budget"]) is not dict or artifact["budget"] != asdict(
        plan.budget
    ):
        raise RuntimeError("budget observation belongs to another ExperimentBudget")
    integer_fields = (
        "measured_gpu_ms",
        "fixed_instance_billed_gpu_ms",
        "observed_wall_ms",
        "registered_wall_delta_ms",
        "registered_gpu_delta_ms",
        "registered_billed_delta_ms",
    )
    if any(type(artifact[name]) is not int for name in integer_fields):
        raise RuntimeError(
            "budget observation accounting must use integral milliseconds"
        )
    try:
        observation = BudgetObservationReceipt(
            schema_version=artifact["schema_version"],
            budget=plan.budget,
            observed_component_ms=tuple(
                (str(row[0]), int(row[1])) for row in observed_rows
            ),
            measured_gpu_ms=artifact["measured_gpu_ms"],
            fixed_instance_billed_gpu_ms=artifact["fixed_instance_billed_gpu_ms"],
            terminal_evidence_sha256=artifact["terminal_evidence_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "budget observation violates its planning contract"
        ) from error
    expected = _budget_observation_artifact(observation)
    if artifact != expected:
        raise RuntimeError("budget observation deltas or identities are inconsistent")
    if (
        observation.measured_gpu_ms
        != observation.observed_wall_ms * plan.budget.gpu_count
        or observation.fixed_instance_billed_gpu_ms
        != (
            observation.observed_wall_ms
            * plan.runtime_plan.physical_fixed_instance_gpu_count
        )
    ):
        raise RuntimeError(
            "budget observation violates its declared GPU accounting semantics"
        )
    try:
        sidecar_body = sidecar_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError("budget observation sidecar is unreadable") from error
    if (
        sidecar_body != observation.sha256 + "\n"
        or observation.budget.sha256 != plan.budget.sha256
        or observation.terminal_evidence_sha256 != observed_evidence_sha256
    ):
        raise RuntimeError("budget observation content binding is invalid")
    return observation, receipt_path, sidecar_path


def _recover_prepared_completion(
    *,
    root: Path,
    run_id: str,
    run_nonce_sha256: str,
    plan: IndustrialExecutionPlan,
) -> dict[str, Path] | None:
    """Promote a prepared receipt only after its observation is durable."""

    prepared = root / f"{run_id}.rank0.prepared.json"
    observation_directory = _budget_observation_directory(root, run_id)
    prepared_exists = os.path.lexists(prepared)
    observation_exists = os.path.lexists(observation_directory)
    if not prepared_exists and not observation_exists:
        return None
    if not prepared_exists or not observation_exists:
        raise RuntimeError(
            "industrial completion preparation is incomplete and non-resumable"
        )
    observation, _, _ = _load_budget_observation(
        root=root,
        run_id=run_id,
        plan=plan,
        terminal_receipt=prepared,
    )

    def validate_prepared(completed: dict[str, Path]) -> None:
        _validate_resume(
            completed=completed,
            run_id=run_id,
            plan=plan,
            run_nonce_sha256=run_nonce_sha256,
        )

    def validate_post_binding() -> None:
        recovered, _, _ = _load_budget_observation(
            root=root,
            run_id=run_id,
            plan=plan,
            terminal_receipt=prepared,
        )
        if recovered != observation:
            raise RuntimeError("budget observation changed during recovery")

    return publish_prepared_evidence_completion(
        root,
        run_id=run_id,
        rank=0,
        expected_receipt_sha256=observation.terminal_evidence_sha256,
        validate=validate_prepared,
        validate_post_binding=validate_post_binding,
    )


@dataclass(frozen=True)
class ArtifactBinding:
    """One file bound by raw bytes plus its optional semantic identity."""

    name: str
    path: str
    content_sha256: str
    file_sha256: str
    size: int
    experiment: str | None = None

    @classmethod
    def from_path(
        cls,
        *,
        name: str,
        path: str | Path,
        expected_sha256: str | None = None,
        semantic_sha256: str | None = None,
        experiment: str | None = None,
    ) -> ArtifactBinding:
        unresolved = Path(path)
        if unresolved.is_symlink():
            raise ValueError("artifact must be a regular non-symlinked file")
        resolved = unresolved.resolve()
        if (
            not name
            or "\n" in name
            or (experiment is not None and (not experiment or "\n" in experiment))
        ):
            raise ValueError("artifact identities must be non-empty single-line text")
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError("artifact must be a regular non-symlinked file")
        file_digest = _file_sha256(resolved)
        content_digest = semantic_sha256 or file_digest
        if not _is_sha256(content_digest):
            raise ValueError("artifact semantic identity must be a lowercase SHA-256")
        if expected_sha256 is not None and content_digest != expected_sha256:
            raise ValueError(f"artifact {name!r} differs from its locked digest")
        value = cls(
            name=name,
            path=str(resolved),
            content_sha256=content_digest,
            file_sha256=file_digest,
            size=resolved.stat().st_size,
            experiment=experiment,
        )
        value.assert_unchanged()
        return value

    def assert_unchanged(self) -> None:
        if (
            not self.name
            or not _is_sha256(self.content_sha256)
            or not _is_sha256(self.file_sha256)
            or self.size < 0
        ):
            raise ValueError("artifact binding is malformed")
        path = Path(self.path)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != self.size
            or _file_sha256(path) != self.file_sha256
        ):
            raise RuntimeError(f"bound artifact changed: {self.name}")

    def identity_dict(self) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "name": self.name,
            "content_sha256": self.content_sha256,
            "file_sha256": self.file_sha256,
            "size": self.size,
        }


def _bound_json_object(binding: ArtifactBinding, *, label: str) -> dict[str, object]:
    binding.assert_unchanged()
    try:
        value = json.loads(Path(binding.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    return value


def _required_object(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    return value


def _required_command(
    commands: Mapping[str, object],
    *,
    name: str,
    expected_argv: tuple[str, ...],
) -> str:
    command = _required_object(commands.get(name), label=f"inventory {name} command")
    if set(command) != {"argv", "stdout"}:
        raise ValueError(f"inventory {name} command fields are incomplete")
    argv = command["argv"]
    stdout = command["stdout"]
    if argv != list(expected_argv) or not isinstance(stdout, str):
        raise ValueError(f"inventory {name} command authority differs")
    return stdout


def _csv_rows(value: str, *, columns: int, label: str) -> tuple[tuple[str, ...], ...]:
    rows = tuple(
        tuple(field.strip() for field in row)
        for row in csv.reader(io.StringIO(value))
        if any(field.strip() for field in row)
    )
    if any(len(row) != columns or any(not field for field in row) for row in rows):
        raise ValueError(f"{label} has an ambiguous CSV schema")
    return rows


def _validate_inventory_source_artifact(
    binding: ArtifactBinding,
    *,
    inventory: GpuInventory,
) -> str:
    """Reparse the exact first-party inventory receipt and return its driver."""

    if binding.name != "gpu_inventory_source_receipt":
        raise ValueError("GPU inventory source artifact has the wrong identity")
    receipt = _bound_json_object(binding, label="GPU inventory source receipt")
    expected_fields = {
        "schema_version",
        "kind",
        "challenge_nonce_sha256",
        "host_id",
        "hostname",
        "machine_id_sha256",
        "commands",
        "parsed_topology",
        "pci_locality",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("GPU inventory source receipt fields are incomplete")
    receipt_sha256 = receipt.get("receipt_sha256")
    receipt_content = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "gpu_inventory_probe_receipt"
        or not _is_sha256(receipt.get("challenge_nonce_sha256"))
        or not _is_sha256(receipt.get("machine_id_sha256"))
        or not isinstance(receipt.get("host_id"), str)
        or not isinstance(receipt.get("hostname"), str)
        or not str(receipt.get("hostname")).strip()
        or "\n" in str(receipt.get("hostname"))
        or not _is_sha256(receipt_sha256)
        or content_sha256(receipt_content) != receipt_sha256
        or binding.content_sha256 != receipt_sha256
        or inventory.source_receipt_sha256 != receipt_sha256
    ):
        raise ValueError("GPU inventory source receipt identity differs")
    host_id = str(receipt["host_id"])
    if not host_id or {device.host_id for device in inventory.devices} != {host_id}:
        raise ValueError("GPU inventory source host differs from the scheduler pool")

    commands = _required_object(receipt["commands"], label="inventory commands")
    if set(commands) != {"gpu", "processes", "topology"}:
        raise ValueError("GPU inventory source command set is incomplete")
    gpu_rows = _csv_rows(
        _required_command(commands, name="gpu", expected_argv=_GPU_INVENTORY_ARGV),
        columns=11,
        label="GPU inventory probe",
    )
    process_rows = _csv_rows(
        _required_command(commands, name="processes", expected_argv=_GPU_PROCESS_ARGV),
        columns=3,
        label="GPU process probe",
    )
    _required_command(commands, name="topology", expected_argv=_GPU_TOPOLOGY_ARGV)
    if len(gpu_rows) != len(inventory.devices):
        raise ValueError("GPU inventory source changed full-pool cardinality")
    try:
        indexed = tuple(sorted(gpu_rows, key=lambda row: int(row[0])))
    except ValueError as error:
        raise ValueError("GPU inventory source contains a malformed index") from error
    if tuple(int(row[0]) for row in indexed) != tuple(range(len(indexed))):
        raise ValueError("GPU inventory source indices are not contiguous")
    devices = {device.uuid: device for device in inventory.devices}
    processes: dict[str, list[str]] = {uuid: [] for uuid in devices}
    for uuid, pid, process_name in process_rows:
        if uuid not in devices or not pid.isdigit() or int(pid) < 1:
            raise ValueError("GPU process source references an invalid device/process")
        processes[uuid].append(f"{pid}:{process_name}")

    drivers: set[str] = set()
    for row in indexed:
        (
            _,
            uuid,
            model,
            memory_mib,
            driver,
            compute_capability,
            pci_bus_id,
            power_limit,
            thermal_limit,
            max_sm_clock,
            persistence_mode,
        ) = row
        device = devices.get(uuid)
        if device is None:
            raise ValueError("GPU inventory source references a foreign UUID")
        try:
            compute = tuple(int(value) for value in compute_capability.split("."))
            memory_bytes = int(memory_mib) * 1024 * 1024
            power_watts = float(power_limit)
            thermal_celsius = float(thermal_limit)
            max_sm_mhz = int(max_sm_clock)
        except ValueError as error:
            raise ValueError(
                "GPU inventory source contains malformed hardware"
            ) from error
        if (
            len(compute) != 2
            or device.model != model
            or device.memory_bytes != memory_bytes
            or device.compute_capability != compute
            or device.pci_bus_id.lower() != pci_bus_id.lower()
            or device.power_limit_watts != power_watts
            or device.thermal_limit_celsius != thermal_celsius
            or device.clock_policy
            != f"persistence={persistence_mode};max_sm_mhz={max_sm_mhz}"
            or device.reserved_processes != tuple(sorted(processes[uuid]))
        ):
            raise ValueError("GPU inventory source differs from scheduler hardware")
        drivers.add(driver)
    if len(drivers) != 1:
        raise ValueError("GPU inventory source lacks one exact host driver authority")

    locality = receipt["pci_locality"]
    if not isinstance(locality, list) or len(locality) != len(indexed):
        raise ValueError("GPU inventory source PCI locality is incomplete")
    index_by_uuid = {row[1]: int(row[0]) for row in indexed}
    for row in locality:
        if type(row) is not dict or set(row) != {
            "index",
            "uuid",
            "pci_bus_id",
            "pci_root",
            "numa_node",
        }:
            raise ValueError("GPU inventory source PCI locality is malformed")
        device = devices.get(row["uuid"])
        if device is None or (
            row["index"] != index_by_uuid.get(device.uuid)
            or row["pci_bus_id"] != device.pci_bus_id.lower()
            or row["pci_root"] != device.pci_root
            or row["numa_node"] != device.numa_node
        ):
            raise ValueError("GPU inventory source PCI locality differs")
    topology = _required_object(
        receipt["parsed_topology"], label="inventory parsed topology"
    )
    if topology.get("parse_error") is not None:
        raise ValueError("GPU inventory source topology is incomplete")
    return next(iter(drivers))


def _validate_compile_key_for_run_config(
    plan: CompileCacheLaunchPlan,
    *,
    config: RunConfig,
) -> None:
    plan.validate()
    key = plan.key
    expected_drafter = (
        None if config.method == "target_only" else config.model.drafter_revision
    )
    if (
        key.source_sha256 != PINNED_SGLANG_COMPILE_SOURCE_SHA256
        or key.target_revision != config.model.target_revision
        or key.drafter_revision != expected_drafter
        or key.tensor_parallel_size != config.runtime.tensor_parallel_size
        or key.context_limit != config.runtime.context_length
        or key.max_running_requests != config.runtime.max_running_requests
    ):
        raise ValueError("compile-cache key differs from the exact RunConfig")


def _validate_runtime_envelope_artifact(
    binding: ArtifactBinding,
    *,
    inventory: GpuInventory,
    checkout: Path,
    compile_plan: CompileCacheLaunchPlan,
    config: RunConfig,
    inventory_driver: str,
    assigned_gpu_uuid: str,
) -> None:
    """Bind compile inputs to the complete PASS doctor envelope."""

    if binding.name != "runtime_envelope":
        raise ValueError("runtime authority must be the locked runtime_envelope")
    report = _bound_json_object(binding, label="runtime envelope")
    readiness = _required_object(report.get("readiness"), label="runtime readiness")
    checks = _required_object(report.get("checks"), label="runtime checks")
    manifest = _required_object(
        report.get("runtime_manifest"), label="runtime compatibility manifest"
    )
    if (
        report.get("schema_version") != 1
        or report.get("status") != "PASS"
        or readiness.get("status") != "PASS"
        or readiness.get("fail_count") != 0
        or readiness.get("unknown_count") != 0
        or not checks
        or readiness.get("pass_count") != len(checks)
        or any(
            type(value) is not dict or value.get("status") != "PASS"
            for value in checks.values()
        )
        or manifest.get("valid") is not True
        or not _is_sha256(manifest.get("sha256"))
        or manifest.get("sidecar_sha256") != manifest.get("sha256")
    ):
        raise ValueError("runtime envelope is not a complete PASS authority")
    roots = _required_object(report.get("roots"), label="runtime roots")
    source = _required_object(report.get("source_tree"), label="SGLang source tree")
    if (
        roots.get("patched_sglang") != str(checkout)
        or roots.get("distinct") is not True
        or source.get("path") != str(checkout)
        or source.get("is_git_checkout") is not True
        or source.get("root_matches_toplevel") is not True
        or source.get("dirty") is not False
        or source.get("pinned_ancestor") is not True
        or source.get("patch_commits") != PINNED_SGLANG_PATCH_COUNT
        or source.get("tree") != PINNED_SGLANG_TREE
    ):
        raise ValueError("runtime envelope differs from the verified SGLang source")

    gpu = _required_object(report.get("gpu"), label="runtime GPU envelope")
    parsed = _required_object(
        gpu.get("parsed_inventory"), label="runtime parsed GPU inventory"
    )
    raw_devices = parsed.get("devices")
    if parsed.get("parse_error") is not None or not isinstance(raw_devices, list):
        raise ValueError("runtime envelope GPU inventory is incomplete")
    devices = {device.uuid: device for device in inventory.devices}
    if len(raw_devices) != len(devices):
        raise ValueError("runtime envelope changed full-pool cardinality")
    doctor_drivers: set[str] = set()
    seen: set[str] = set()
    for raw in raw_devices:
        if type(raw) is not dict:
            raise TypeError("runtime envelope GPU row must be an object")
        uuid = raw.get("uuid")
        device = devices.get(uuid) if isinstance(uuid, str) else None
        if device is None or uuid in seen:
            raise ValueError("runtime envelope contains a foreign/duplicate GPU")
        seen.add(uuid)
        expected_memory_mib, remainder = divmod(device.memory_bytes, 1024 * 1024)
        if (
            remainder
            or raw.get("name") != device.model
            or raw.get("memory_total_mib") != expected_memory_mib
            or raw.get("compute_capability")
            != ".".join(str(value) for value in device.compute_capability)
            or str(raw.get("pci_bus_id", "")).lower() != device.pci_bus_id.lower()
            or not isinstance(raw.get("driver_version"), str)
        ):
            raise ValueError("runtime envelope GPU row differs from scheduler hardware")
        doctor_drivers.add(str(raw["driver_version"]))
    if seen != set(devices) or doctor_drivers != {inventory_driver}:
        raise ValueError("runtime envelope driver authority differs from inventory")

    torch_runtime = _required_object(gpu.get("torch"), label="runtime Torch envelope")
    python = _required_object(report.get("python"), label="runtime Python envelope")
    packages = _required_object(report.get("packages"), label="runtime packages")
    commands = _required_object(report.get("commands"), label="runtime commands")
    nvcc = commands.get("nvcc")
    nvcc_match = _NVCC_RELEASE.search(nvcc) if isinstance(nvcc, str) else None
    key = compile_plan.key
    if (
        torch_runtime.get("importable") is not True
        or torch_runtime.get("cuda_available") is not True
        or torch_runtime.get("device_count") != len(devices)
        or key.python_version != python.get("version")
        or key.torch_version != torch_runtime.get("version")
        or packages.get("torch")
        != str(torch_runtime.get("version", "")).partition("+")[0]
        or key.triton_version != packages.get("triton")
        or key.cuda_version != torch_runtime.get("cuda_build")
        or nvcc_match is None
        or key.cuda_version != nvcc_match.group(1)
        or key.driver_version != inventory_driver
    ):
        raise ValueError("compile-cache key differs from exact runtime toolchain")
    assigned = devices.get(assigned_gpu_uuid)
    if assigned is None:
        raise ValueError("compile-cache launch lacks its assigned GPU authority")
    expected_sm = "sm_" + "".join(str(value) for value in assigned.compute_capability)
    if key.gpu_model != assigned.model or key.sm_architecture != expected_sm:
        raise ValueError("compile-cache key differs from assigned GPU model/SM")
    _validate_compile_key_for_run_config(compile_plan, config=config)


def _load_server_compile_plan(launch: ServerLaunch) -> CompileCacheLaunchPlan:
    if (
        not isinstance(launch.compile_cache_plan, str)
        or not isinstance(launch.compile_cache_plan_sha256, str)
        or not isinstance(launch.compile_cache_key_sha256, str)
    ):
        raise TypeError("server launch lacks an exact compile-cache plan binding")
    path = Path(launch.compile_cache_plan)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path.is_symlink()
    ):
        raise ValueError("server compile-cache plan path is not immutable/absolute")
    plan = CompileCacheLaunchPlan.load(path)
    if (
        plan.sha256 != launch.compile_cache_plan_sha256
        or plan.key.sha256 != launch.compile_cache_key_sha256
    ):
        raise ValueError("server compile-cache plan identity differs")
    return plan


def _bound_sampling_profile(binding: ArtifactBinding) -> SamplingProfile:
    """Load the exact schema-v2 sampling semantics from a byte-bound file."""

    binding.assert_unchanged()
    try:
        value = json.loads(Path(binding.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("sampling artifact is not valid JSON") from error
    expected_fields = {
        "schema_version",
        "purpose",
        "temperature",
        "top_p",
        "ignore_eos",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("sampling artifact fields do not match schema-v2")
    try:
        profile = SamplingProfile(**value)
        profile.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("sampling artifact has invalid schema-v2 semantics") from error
    if profile.sha256 != binding.content_sha256:
        raise ValueError("sampling artifact differs from its semantic digest")
    return profile


def _validate_request_sampling(
    request: BoundServingRequest,
    *,
    profile: SamplingProfile,
) -> None:
    parameters = dict(request.sampling.items)
    expected = {
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "ignore_eos": profile.ignore_eos,
        "max_new_tokens": request.requested_output_tokens,
    }
    for name, expected_value in expected.items():
        actual = parameters.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"request sampling {name} differs from the bound sampling profile"
            )
    sampling_seed = parameters.get("sampling_seed")
    if (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or sampling_seed < 0
    ):
        raise ValueError("request sampling requires a non-negative sampling_seed")


def _expected_scored_split(experiment: str, block: int) -> str:
    if experiment in {"E3b", "E5"}:
        return "pilot" if block in {0, 1, 2, 3} else "confirmation"
    if experiment == "E0":
        return "broad_replication"
    if experiment in {"E3a", "E1", "E2", "E4", "E1a"}:
        return "tuning"
    raise ValueError("serving cell has no registered scored split")


def _validate_cell_load_contract(
    cell: ExperimentCell,
    load_plan: ProductionLoadPlan,
) -> None:
    """Bind registry workload semantics to one immutable production trace."""

    identity = cell.identity
    corpus = load_plan.scored
    expected_split = _expected_scored_split(identity.experiment, identity.block)
    if corpus.split != expected_split:
        raise ValueError("load split differs from the registry stage/block")

    arrival = identity.arrival
    if arrival.startswith("closed_loop"):
        expected_source = "closed_loop"
    elif arrival in {"immediate_burst", "deterministic_stratified_requests"}:
        expected_source = "immediate_burst"
    elif arrival in {
        "poisson",
        "moderate_soak",
        "saturation_soak",
        "overload_soak",
    }:
        expected_source = "poisson"
    elif arrival == "burstgpt_shape":
        expected_source = "external_shape"
    else:
        raise ValueError("cell arrival policy is not materialized for execution")
    if corpus.source_kind != expected_source:
        raise ValueError("load source kind differs from the registry arrival policy")
    arrivals = tuple(request.arrival_us for request in corpus.requests)
    if expected_source in {"closed_loop", "immediate_burst"} and any(
        value != 0 for value in arrivals
    ):
        raise ValueError(
            "closed-loop/immediate-burst requests must start at offset zero"
        )
    if expected_source == "poisson" and (
        not arrivals
        or arrivals[0] <= 0
        or any(right <= left for left, right in pairwise(arrivals))
    ):
        raise ValueError("registered Poisson arrivals require strictly positive gaps")

    source = dict(corpus.source_parameters)
    cohort_text = identity.cohort
    try:
        count_text, popularity = cohort_text.removeprefix("K=").split(":", 1)
        cohort_count = int(count_text)
    except (AttributeError, ValueError) as error:
        raise ValueError("registry cohort policy is malformed") from error
    if (
        cohort_count != identity.cohort_count
        or source.get("cohort_count") != cohort_count
        or source.get("cohort_popularity") != popularity
    ):
        raise ValueError("load cohort generator differs from the registry cell")
    if (
        expected_source == "closed_loop"
        and source.get("concurrency") != identity.concurrency
    ):
        raise ValueError("closed-loop population cap differs from the registry cell")
    if identity.load_factor is not None and (
        expected_source != "poisson"
        or source.get("registered_load_factor") != identity.load_factor
    ):
        raise ValueError("Poisson load factor differs from the registry cell")

    lengths = tuple(
        (len(request.input_token_ids), request.requested_output_tokens)
        for request in corpus.requests
    )
    context = identity.context
    if context is None:
        raise ValueError("registered serving cell lacks a context identity")
    if identity.regime == "long_input_short_output" and any(
        input_tokens < math.ceil(0.75 * context)
        or output_tokens > math.floor(0.25 * context)
        for input_tokens, output_tokens in lengths
    ):
        raise ValueError("corpus violates the long-input/short-output regime")
    if identity.regime == "short_input_long_generation" and any(
        input_tokens > math.floor(0.25 * context)
        or output_tokens < math.ceil(0.75 * context)
        for input_tokens, output_tokens in lengths
    ):
        raise ValueError("corpus violates the short-input/long-generation regime")
    if identity.regime == "multi_turn_shared_prefix":
        inputs = tuple(request.input_token_ids for request in corpus.requests)
        shared = min((len(row) for row in inputs), default=0) // 2
        if (
            len(inputs) < 2
            or shared < 1
            or not all(row[:shared] == inputs[0][:shared] for row in inputs)
        ):
            raise ValueError("multi-turn corpus lacks a shared token prefix")
    if identity.regime == "production_mix" and not (
        any(left > right for left, right in lengths)
        and any(left < right for left, right in lengths)
    ):
        raise ValueError("production-mix corpus lacks both request regimes")


def industrial_execution_split_contract(
    *,
    registry_sha256: str,
    cell: ExperimentCell,
    load_plan: ProductionLoadPlan,
    sampling_profile_sha256: str,
    model_lock_sha256: str,
) -> dict[str, object]:
    """Return the exact local split/load identity required by the executor."""

    if not _is_sha256(registry_sha256):
        raise ValueError("split contract requires the registry SHA-256")
    if not _is_sha256(sampling_profile_sha256) or not _is_sha256(model_lock_sha256):
        raise ValueError("split contract requires sampling/model-lock SHA-256 values")
    load_plan.validate()
    _validate_cell_load_contract(cell, load_plan)
    hashes = load_plan.scored.hashes
    return {
        "schema_version": 1,
        "kind": "industrial_execution_split",
        "registry_sha256": registry_sha256,
        "cell_id": cell.cell_id,
        "experiment": cell.identity.experiment,
        "block": cell.identity.block,
        "task": cell.identity.task,
        "variant": cell.identity.variant,
        "scored_split": load_plan.scored.split,
        "paired_replay_sha256": load_plan.paired_replay_sha256,
        "warmup_corpus_sha256": (
            None if load_plan.warmup is None else load_plan.warmup.hashes.corpus_sha256
        ),
        "corpus_sha256": hashes.corpus_sha256,
        "request_ids_sha256": hashes.request_ids_sha256,
        "arrivals_sha256": hashes.arrivals_sha256,
        "cohorts_sha256": hashes.cohorts_sha256,
        "cancellations_sha256": hashes.cancellations_sha256,
        "source_kind": load_plan.scored.source_kind,
        "source_identity_sha256": load_plan.scored.source_identity_sha256,
        "source_parameters": dict(load_plan.scored.source_parameters),
        "window_sha256": load_plan.window.sha256,
        "sampling_profile_sha256": sampling_profile_sha256,
        "model_lock_sha256": model_lock_sha256,
    }


def _request_routes(
    corpus: RequestCorpus | None,
    *,
    route_id: str,
) -> tuple[BoundServingRequest, ...]:
    if corpus is None:
        return ()
    corpus.validate()
    return tuple(
        BoundServingRequest.create(request, route_id=route_id)
        for request in corpus.requests
    )


def _validate_runtime_dispatch_authority(
    *,
    runtime_plan: IndustrialRuntimePlan,
    dispatch_plan: GpuDispatchPlan,
    dispatch_context: GpuDispatchExecutionContext,
    budget_plan: BudgetPlan,
    budget: ExperimentBudget,
) -> None:
    """Replay the scheduler and compare the complete launch-time binding."""

    if not isinstance(dispatch_plan, GpuDispatchPlan):
        raise TypeError("execution dispatch_plan must be a GpuDispatchPlan")
    if not isinstance(dispatch_context, GpuDispatchExecutionContext):
        raise TypeError(
            "execution dispatch_context must be a GpuDispatchExecutionContext"
        )
    if type(budget_plan) is not BudgetPlan:
        raise TypeError("execution requires one exact BudgetPlan")
    if dispatch_context.budget_plan != budget_plan:
        raise ValueError("launch BudgetPlan differs from the dispatch authority")
    ready_budgets = dispatch_context.require_ready_budget_authority()
    capacity_authority = budget_plan.capacity_authority
    if capacity_authority is None:
        raise ValueError("launch BudgetPlan lacks raw capacity authority")
    validate_dispatch_plan_for_execution(
        dispatch_plan,
        execution_context=dispatch_context,
    )
    registry = dispatch_context.registry
    inventory = dispatch_context.inventory
    if (
        runtime_plan.registry_sha256 != registry.sha256
        or dispatch_plan.registry_sha256 != registry.sha256
        or dispatch_plan.inventory_sha256 != inventory.sha256
    ):
        raise ValueError(
            "launch dispatch authority belongs to another registry or pool"
        )
    canonical_cells = tuple(
        cell for cell in registry.cells if cell.cell_id == runtime_plan.cell_id
    )
    if canonical_cells != (runtime_plan.cell,) or (
        runtime_plan.cell_declaration_sha256 != runtime_plan.cell.sha256
    ):
        raise ValueError("launch runtime cell differs from the canonical registry cell")
    if dispatch_context.budgets_by_cell_id.get(runtime_plan.cell_id) != budget:
        raise ValueError("launch budget differs from the scheduler authority")
    if {row.cell_id: row for row in ready_budgets}.get(runtime_plan.cell_id) != budget:
        raise ValueError("launch budget differs from the READY BudgetPlan")
    assignments = tuple(
        assignment
        for wave in dispatch_plan.waves
        for assignment in wave.assignments
        if assignment.work_item.item_id == runtime_plan.cell_id
    )
    if len(assignments) != 1:
        raise ValueError("launch cell lacks one exact scheduler-issued assignment")
    assignment = assignments[0]
    physical = runtime_plan.physical_assignment
    if physical is None:
        raise ValueError("launch runtime lacks a physical scheduler assignment")
    claim = assignment.work_item.claim
    devices = tuple(inventory.device(uuid) for uuid in assignment.gpu_uuids)
    host_ids = {device.host_id for device in devices}
    if len(host_ids) != 1:
        raise ValueError("scheduler-issued launch assignment crosses hosts")
    host_id = next(iter(host_ids))
    topology_group_ids: list[tuple[str, ...]] = []
    for rank_group in assignment.rank_groups:
        if claim.gang_shape.tensor_parallel_size == 1:
            topology_group_ids.append(())
            continue
        rank_set = set(rank_group)
        eligible = tuple(
            group.group_id
            for group in inventory.topology_groups
            if group.host_id == host_id
            and rank_set <= set(group.gpu_uuids)
            and (
                not claim.allowed_topology_groups
                or group.group_id in claim.allowed_topology_groups
            )
            and (not claim.allowed_fabrics or group.fabric in claim.allowed_fabrics)
        )
        if not eligible:
            raise ValueError("scheduler-issued launch TP gang lacks topology authority")
        topology_group_ids.append(eligible)
    expected = IndustrialPhysicalAssignment(
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        dispatch_plan_sha256=dispatch_plan.sha256,
        experiment_budget_sha256=budget.sha256,
        budget_plan_sha256=budget_plan.sha256,
        capacity_authority_sha256=capacity_authority.sha256,
        budget_materialization_authority_sha256=(
            dispatch_context.budget_materialization_authority.sha256
        ),
        assignment_sha256=assignment.sha256,
        work_item_sha256=assignment.work_item.sha256,
        gpu_uuids=assignment.gpu_uuids,
        rank_groups=assignment.rank_groups,
        ports=assignment.ports,
        tensor_parallel_size=claim.gang_shape.tensor_parallel_size,
        data_parallel_size=claim.gang_shape.data_parallel_size,
        fixed_instance_gpu_count=len(inventory.devices),
        host_id=host_id,
        topology_group_ids=tuple(topology_group_ids),
    )
    if physical != expected:
        raise ValueError(
            "launch physical assignment differs from the exact scheduler replay"
        )


class TrainablePlanExecutionBlockedError(RuntimeError):
    """A named pre-mutation failure of adapted execution authority."""

    def __init__(self, reason_code: str) -> None:
        if (
            type(reason_code) is not str
            or not reason_code
            or reason_code.strip() != reason_code
        ):
            raise ValueError("trainable-plan BLOCKED reason must be canonical")
        self.reason_code = reason_code
        super().__init__(
            f"industrial trainable-plan execution is BLOCKED: {reason_code}"
        )


def _require_trainable_raw_artifact_match(
    source: TrainablePlanRawJsonBinding,
    artifact: ArtifactBinding,
    *,
    label: str,
) -> None:
    artifact.assert_unchanged()
    if (
        source.path != artifact.path
        or source.semantic_sha256 != artifact.content_sha256
        or source.file_sha256 != artifact.file_sha256
        or source.size != artifact.size
    ):
        raise ValueError(f"trainable-plan {label} differs from execution artifact")


def _launch_model_root(launch: ServerLaunch, option: str) -> str:
    try:
        index = launch.argv.index(option)
        root = launch.argv[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"server launch lacks exact {option} authority") from error
    path = Path(root)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"server launch {option} root is not absolute and resolved")
    return root


def _require_adapted_execution_semantics_sha256(
    runtime_plan: IndustrialRuntimePlan,
) -> str:
    """Return the exact E1 overlay digest or a stable scientific BLOCK."""

    from lightcone_spec.experiments.execution_semantics import (
        EXECUTION_SEMANTICS_RAW_ACTIVATION_UNAVAILABLE_REASON,
        EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON,
        CellExecutionSemantics,
        CellExecutionSemanticsBlockedError,
    )

    if type(runtime_plan) is not IndustrialRuntimePlan:
        raise TypeError("adapted execution semantics require an exact runtime plan")
    semantics = runtime_plan.execution_semantics
    if type(semantics) is not CellExecutionSemantics:
        raise CellExecutionSemanticsBlockedError(
            EXECUTION_SEMANTICS_RAW_ACTIVATION_UNAVAILABLE_REASON
            if runtime_plan.cell.identity.experiment == "E1"
            else EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON
        )
    return semantics.sha256


def _require_execution_trainable_plan_authority(
    plan: IndustrialExecutionPlan,
) -> None:
    """Replay adapted parameter authority before any executor-side mutation."""

    runtime_plan = plan.runtime_plan
    if type(runtime_plan) is not IndustrialRuntimePlan:
        _require_adapted_execution_semantics_sha256(runtime_plan)
    if runtime_plan.cell.identity.experiment == "E1":
        _require_registered_e1_execution_recipe(
            registry=plan.dispatch_context.registry,
            cell=runtime_plan.cell,
            execution_semantics=runtime_plan.execution_semantics,
        )
    config = runtime_plan.rank_configs[0]
    method = config.method
    authority = plan.trainable_plan_authority
    release_pin = plan.prepared_model_content_release_manifest_sha256
    if method in {"target_only", "static"}:
        if authority is not None or release_pin is not None:
            raise ValueError(
                "Target-only/Static execution must not carry trainable-plan authority"
            )
        if runtime_plan.parameter_plan_sha256 is not None:
            raise ValueError("Target-only/Static runtime carries a parameter-plan SHA")
        return
    if method not in {"tts", "l0"}:
        raise ValueError("execution trainable-plan gate supports only core methods")
    execution_semantics_sha256 = _require_adapted_execution_semantics_sha256(
        runtime_plan
    )
    if release_pin is None:
        raise TrainablePlanExecutionBlockedError(
            PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON
        )
    if type(release_pin) is not str or not _is_sha256(release_pin):
        raise ValueError("prepared model content release pin must be SHA-256")
    if type(authority) is not TrainablePlanAuthorityBinding:
        raise TrainablePlanExecutionBlockedError(
            TRAINABLE_PLAN_RAW_AUTHORITY_UNAVAILABLE_REASON
        )
    if not has_prepared_model_content_release_manifest_sha256(
        model_lock_sha256=authority.model_lock_sha256,
        prepared=authority.prepared_model_content_authority.prepared_model_set,
        claimed_manifest_sha256=release_pin,
    ):
        raise TrainablePlanExecutionBlockedError(
            PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON
        )
    _require_trainable_raw_artifact_match(
        authority.model_lock,
        plan.model_lock_artifact,
        label="model lock",
    )
    _require_trainable_raw_artifact_match(
        authority.split,
        plan.split_artifact,
        label="execution split",
    )
    if authority.run_config.semantic_sha256 != run_config_sha256(
        config
    ) or authority.run_config.load() != config.model_dump(mode="json"):
        raise ValueError("trainable-plan RunConfig differs from execution plan")
    prepared_roots = {
        snapshot.model_id: snapshot.root
        for snapshot in authority.prepared_model_content_authority.prepared_model_set.snapshots
    }
    if prepared_roots.get(config.model.target) != _launch_model_root(
        plan.server_launch, "--model-path"
    ) or prepared_roots.get(config.model.drafter) != _launch_model_root(
        plan.server_launch, "--speculative-draft-model-path"
    ):
        raise ValueError("trainable-plan prepared roots differ from server launch")
    adaptation = config.adaptation
    if adaptation is None:  # pragma: no cover - RunConfig/cell invariant
        raise RuntimeError("adapted execution lost its adaptation configuration")
    try:
        parameter_plan = require_trainable_plan_authority_for_method(
            method,
            authority,
            expected_model_lock_sha256=plan.model_lock_artifact.content_sha256,
            expected_prepared_model_content_manifest_sha256=release_pin,
            expected_run_config_sha256=run_config_sha256(config),
            expected_split_sha256=plan.split_artifact.content_sha256,
            expected_cell_id=runtime_plan.cell_id,
            expected_cell_declaration_sha256=(
                plan.runtime_plan.cell_declaration_sha256
            ),
            expected_execution_semantics_sha256=execution_semantics_sha256,
            expected_target_model_id=config.model.target,
            expected_target_revision=config.model.target_revision,
            expected_drafter_model_id=config.model.drafter,
            expected_prepared_drafter_revision=config.model.drafter_revision,
            expected_backend=config.model.algorithm,
            expected_mode=adaptation.weight_update_mode,
            expected_scope=adaptation.parameter_scope,
            expected_optimizer=adaptation.optimizer.name,
            expected_rank=adaptation.rank,
            expected_lora_alpha=adaptation.lora_alpha,
        )
    except PreparedModelContentAuthorityBlocked as error:
        raise TrainablePlanExecutionBlockedError(error.code) from error
    if (
        parameter_plan is None  # pragma: no cover - adapted method postcondition
        or parameter_plan.sha256 != runtime_plan.parameter_plan_sha256
        or parameter_plan.sha256 != authority.trainable_plan_sha256
    ):
        raise ValueError("runtime parameter plan differs from raw execution authority")


def _require_render_trainable_plan_authority(
    *,
    runtime_plan: IndustrialRuntimePlan,
    model_lock_artifact: ArtifactBinding,
    split_artifact: ArtifactBinding,
    model_roots: Mapping[str, str],
    authority: TrainablePlanAuthorityBinding | None,
    release_pin: str | None,
) -> None:
    """Replay adapted authority before rendering any runtime artifact."""

    if len(runtime_plan.rank_configs) != 1:
        raise ValueError("server rendering supports the released one-rank topology")
    config = runtime_plan.rank_configs[0]
    method = config.method
    if method in {"target_only", "static"}:
        if authority is not None or release_pin is not None:
            raise ValueError(
                "Target-only/Static rendering must not carry trainable-plan authority"
            )
        if runtime_plan.parameter_plan_sha256 is not None:
            raise ValueError("Target-only/Static runtime carries a parameter-plan SHA")
        return
    if method not in {"tts", "l0"}:
        raise ValueError("render trainable-plan gate supports only core methods")
    execution_semantics_sha256 = _require_adapted_execution_semantics_sha256(
        runtime_plan
    )
    if release_pin is None:
        raise TrainablePlanExecutionBlockedError(
            PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON
        )
    if type(release_pin) is not str or not _is_sha256(release_pin):
        raise ValueError("prepared model content release pin must be SHA-256")
    if type(authority) is not TrainablePlanAuthorityBinding:
        raise TrainablePlanExecutionBlockedError(
            TRAINABLE_PLAN_RAW_AUTHORITY_UNAVAILABLE_REASON
        )
    if not has_prepared_model_content_release_manifest_sha256(
        model_lock_sha256=authority.model_lock_sha256,
        prepared=authority.prepared_model_content_authority.prepared_model_set,
        claimed_manifest_sha256=release_pin,
    ):
        raise TrainablePlanExecutionBlockedError(
            PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON
        )

    _require_trainable_raw_artifact_match(
        authority.model_lock,
        model_lock_artifact,
        label="model lock",
    )
    _require_trainable_raw_artifact_match(
        authority.split,
        split_artifact,
        label="execution split",
    )
    if authority.run_config.semantic_sha256 != run_config_sha256(
        config
    ) or authority.run_config.load() != config.model_dump(mode="json"):
        raise ValueError("trainable-plan RunConfig differs from rendered execution")
    prepared_roots = {
        snapshot.model_id: snapshot.root
        for snapshot in authority.prepared_model_content_authority.prepared_model_set.snapshots
    }
    expected_roots: dict[str, str] = {}
    for model_id in (config.model.target, config.model.drafter):
        root = model_roots.get(model_id)
        if type(root) is not str:
            raise ValueError(f"verified local model root is missing: {model_id}")
        expected_roots[model_id] = str(Path(root).resolve())
    if any(
        prepared_roots.get(model_id) != root
        for model_id, root in expected_roots.items()
    ):
        raise ValueError("trainable-plan prepared roots differ from rendered roots")
    adaptation = config.adaptation
    if adaptation is None:  # pragma: no cover - RunConfig/cell invariant
        raise RuntimeError("adapted rendering lost its adaptation configuration")
    try:
        parameter_plan = require_trainable_plan_authority_for_method(
            method,
            authority,
            expected_model_lock_sha256=model_lock_artifact.content_sha256,
            expected_prepared_model_content_manifest_sha256=release_pin,
            expected_run_config_sha256=run_config_sha256(config),
            expected_split_sha256=split_artifact.content_sha256,
            expected_cell_id=runtime_plan.cell_id,
            expected_cell_declaration_sha256=(runtime_plan.cell_declaration_sha256),
            expected_execution_semantics_sha256=execution_semantics_sha256,
            expected_target_model_id=config.model.target,
            expected_target_revision=config.model.target_revision,
            expected_drafter_model_id=config.model.drafter,
            expected_prepared_drafter_revision=config.model.drafter_revision,
            expected_backend=config.model.algorithm,
            expected_mode=adaptation.weight_update_mode,
            expected_scope=adaptation.parameter_scope,
            expected_optimizer=adaptation.optimizer.name,
            expected_rank=adaptation.rank,
            expected_lora_alpha=adaptation.lora_alpha,
        )
    except PreparedModelContentAuthorityBlocked as error:
        raise TrainablePlanExecutionBlockedError(error.code) from error
    if (
        parameter_plan is None  # pragma: no cover - adapted method postcondition
        or parameter_plan.sha256 != runtime_plan.parameter_plan_sha256
        or parameter_plan.sha256 != authority.trainable_plan_sha256
    ):
        raise ValueError("runtime parameter plan differs from raw render authority")


def _require_execution_itl_timestamp_authority(
    *,
    runtime_plan: IndustrialRuntimePlan,
    dispatch_context: GpuDispatchExecutionContext,
) -> E2ItlTimestampPlan | None:
    """Replay the release-owned E2 timing gate before execution mutation."""

    if type(runtime_plan) is not IndustrialRuntimePlan:
        raise TypeError("ITL execution gate requires an exact runtime plan")
    if runtime_plan.cell.identity.experiment != "E2":
        return None
    if type(dispatch_context) is not GpuDispatchExecutionContext:
        raise TypeError("ITL execution gate requires an exact execution context")
    plan = release_e2_itl_timestamp_plan(
        dispatch_context.registry,
        runtime_plan.cell,
    )
    require_e2_itl_timestamp_prelaunch(plan)
    return plan


@dataclass(frozen=True)
class IndustrialExecutionPlan:
    """Immutable local plan for exactly one rank and one serving cell."""

    runtime_plan: IndustrialRuntimePlan
    dispatch_plan: GpuDispatchPlan
    dispatch_context: GpuDispatchExecutionContext
    budget_plan: BudgetPlan
    budget: ExperimentBudget
    load_plan: ProductionLoadPlan
    server_launch: ServerLaunch
    dependency_receipt_sha256s: tuple[str, ...]
    expected_dependency_outputs: tuple[tuple[str, str, str], ...]
    dependency_artifacts: tuple[ArtifactBinding, ...]
    split_artifact: ArtifactBinding
    sampling_artifact: ArtifactBinding
    model_lock_artifact: ArtifactBinding
    compile_cache_plan: CompileCacheLaunchPlan
    inventory_source_artifact: ArtifactBinding
    runtime_envelope_artifact: ArtifactBinding
    warmup_requests: tuple[BoundServingRequest, ...]
    scored_requests: tuple[BoundServingRequest, ...]
    bench_argv: tuple[str, ...]
    trainable_plan_authority: TrainablePlanAuthorityBinding | None = None
    failure_execution_authority: FailureExecutionAuthorityToken | None = None
    prepared_model_content_release_manifest_sha256: str | None = None
    evidence_writer_policy: EvidenceWriterPolicy = DEFAULT_EVIDENCE_WRITER_POLICY
    trusted_attester_policy: TrustedAttesterPolicy = NO_TRUSTED_ATTESTERS
    patched_sglang_tree: str = PINNED_SGLANG_TREE
    startup_timeout_s: float = 300.0
    shutdown_timeout_s: float = 30.0
    abort_grace_s: float = 30.0

    def validate(self) -> None:
        # Scientific activation/load/config identity is allocation-free and
        # must fail before any release trust or execution gate.  Rechecking
        # here prevents a caller-authored/replaced runtime-plan overlay from
        # bypassing the bundle's raw-authority replay.
        validate_industrial_execution_semantics_authority(
            runtime_plan=self.runtime_plan,
            dispatch_context=self.dispatch_context,
            registered_load=self.load_plan,
        )
        _require_execution_itl_timestamp_authority(
            runtime_plan=self.runtime_plan,
            dispatch_context=self.dispatch_context,
        )
        if type(self.evidence_writer_policy) is not EvidenceWriterPolicy:
            raise TypeError("execution writer policy must be an exact policy")
        self.evidence_writer_policy.validate()
        if type(self.trusted_attester_policy) is not TrustedAttesterPolicy:
            raise TypeError("execution trust requires an exact release policy")
        require_release_trusted_attester_policy(self.trusted_attester_policy)
        _validate_runtime_dispatch_authority(
            runtime_plan=self.runtime_plan,
            dispatch_plan=self.dispatch_plan,
            dispatch_context=self.dispatch_context,
            budget_plan=self.budget_plan,
            budget=self.budget,
        )
        if not self.runtime_plan.physical_dispatch_ready:
            raise ValueError(
                "logical runtime plan cannot be launched; a physical assignment is required"
            )
        cell = self.runtime_plan.cell
        failure_cell = cell.identity.task == "failure_injection"
        failure_budget = (
            type(self.budget) is ExperimentBudget
            and self.budget.job_kind is BudgetJobKind.FAILURE
        )
        if failure_cell != failure_budget:
            raise ValueError(
                "failure-injection cell and FAILURE budget job kind must match exactly"
            )
        if failure_cell:
            authority = self.failure_execution_authority
            if authority is None:
                raise FailureInjectionAuthorityBlocked(
                    FAILURE_INJECTION_RAW_PLAN_AUTHORITY_REQUIRED_REASON
                )
            expected_scenario = cell.identity.arrival.removeprefix("failure:")
            if (
                type(authority) is not FailureExecutionAuthorityToken
                or authority.registry_sha256 != self.runtime_plan.registry_sha256
                or authority.cell_id != cell.cell_id
                or authority.scenario != expected_scenario
            ):
                raise ValueError(
                    "failure execution authority differs from its runtime cell"
                )
            # A signed plan/token cannot silently degrade into an ordinary
            # serving run.  This release has no source-owned arm/trigger/
            # recover/terminal implementation, so validation remains blocked
            # even if a callable capability entry is added in isolation.
            require_failure_execution_lifecycle(
                authority,
                cell=cell,
                expected_registry_sha256=self.runtime_plan.registry_sha256,
            )
        elif self.failure_execution_authority is not None:
            raise ValueError("non-failure execution cannot carry failure authority")
        if not cell.runnable or cell.status is not CellStatus.UNMEASURED:
            raise ValueError("execution plan requires one runnable UNMEASURED cell")
        calibration = is_serving_interference_calibration_cell(cell)
        if (cell.identity.experiment == "preflight" and not calibration) or (
            cell.resources.workload_class
            in {
                WorkloadClass.DOWNLOAD,
                WorkloadClass.COMPILE,
            }
        ):
            raise ValueError("non-serving/preflight cells cannot enter this executor")
        if len(self.runtime_plan.rank_configs) != 1:
            raise ValueError(
                "the current strict RunConfig exposes only one-rank serving execution"
            )
        self.load_plan.validate()
        budget = self.budget
        if (
            not isinstance(budget, ExperimentBudget)
            or budget.cell_id != cell.cell_id
            or budget.experiment != cell.identity.experiment
            or budget.method != cell.identity.method
            or budget.workload_class is not cell.resources.workload_class
            or budget.gpu_count != cell.resources.gpu_count
            or budget.topology != cell.identity.topology
            or budget.measured_gpu_ms is not None
        ):
            raise ValueError("ExperimentBudget differs from the execution cell")
        physical_assignment = self.runtime_plan.physical_assignment
        if (
            physical_assignment is None
            or physical_assignment.experiment_budget_sha256 != budget.sha256
            or physical_assignment.budget_plan_sha256 != self.budget_plan.sha256
            or physical_assignment.capacity_authority_sha256
            != self.budget_plan.capacity_authority.sha256
            or physical_assignment.budget_materialization_authority_sha256
            != self.dispatch_context.budget_materialization_authority.sha256
        ):
            raise ValueError(
                "ExperimentBudget differs from the physical dispatch-plan binding"
            )
        expected_fixed_instance_bill = budget.wall_time.scale(
            self.runtime_plan.physical_fixed_instance_gpu_count
        )
        if budget.fixed_instance_billed_gpu_ms != expected_fixed_instance_bill:
            raise ValueError(
                "ExperimentBudget fixed-instance billing differs from the physical "
                "inventory"
            )
        scoring_component = _SCORING_COMPONENT_BY_JOB.get(budget.job_kind)
        if scoring_component is None:
            raise ValueError("compile/download budgets cannot enter serving execution")
        if (budget.job_kind is BudgetJobKind.PROFILER) != (
            budget.workload_class is WorkloadClass.PROFILE
        ):
            raise ValueError("profiler budget and PROFILE isolation must match exactly")
        window = self.load_plan.window
        expected_window_ms = {
            "excluded warm-up": (
                budget.excluded_warmup.registered,
                window.warmup_duration_us,
            ),
            "active scored clock": (
                getattr(budget, scoring_component).registered,
                window.arrival_duration_us,
            ),
            "request deadline": (
                budget.request_deadline.registered,
                window.request_deadline_us,
            ),
            "drain": (budget.drain.registered, window.drain_duration_us),
        }
        if any(
            registered_ms * 1000 != observed_us
            for registered_ms, observed_us in expected_window_ms.values()
        ):
            raise ValueError("ExperimentBudget differs from the fixed load window")
        inactive_scoring = set(_SCORING_COMPONENTS) - {scoring_component}
        if any(
            not _scenario_is_explicit_zero(getattr(budget, name))
            for name in inactive_scoring
        ):
            raise ValueError(
                "non-job scored duration components must be registered as zero"
            )
        if budget.excluded_warmup_requests.maximum != len(self.warmup_requests):
            raise ValueError("ExperimentBudget differs from the warm-up request pool")
        maximum_output_tokens = sum(
            request.requested_output_tokens for request in self.scored_requests
        )
        if budget.output_tokens.maximum != maximum_output_tokens:
            raise ValueError("ExperimentBudget differs from the scored token pool")
        if budget.minimum_completed_requests > len(self.scored_requests):
            raise ValueError(
                "ExperimentBudget completion floor exceeds its request pool"
            )
        if (
            budget.p99_anchor_status is P99AnchorStatus.LOCKED
            and budget.minimum_completed_requests < 10_000
        ):
            raise ValueError("a locked p99 anchor requires at least 10,000 completions")
        config = self.runtime_plan.rank_configs[0]
        _require_execution_trainable_plan_authority(self)
        if config.runtime.max_running_requests != cell.identity.concurrency:
            raise ValueError("execution concurrency differs from the registry cell")
        if (
            config.runtime.sampling_profile_sha256
            != self.sampling_artifact.content_sha256
        ):
            raise ValueError("sampling artifact differs from the RunConfig")
        sampling_profile = _bound_sampling_profile(self.sampling_artifact)
        expected_split = industrial_execution_split_contract(
            registry_sha256=self.runtime_plan.registry_sha256,
            cell=cell,
            load_plan=self.load_plan,
            sampling_profile_sha256=self.sampling_artifact.content_sha256,
            model_lock_sha256=self.model_lock_artifact.content_sha256,
        )
        try:
            actual_split = json.loads(
                Path(self.split_artifact.path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("split artifact is not valid JSON") from error
        if actual_split != expected_split:
            raise ValueError("split artifact differs from the exact cell/load contract")
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("execution plan has the wrong patched SGLang tree")
        for value, name in (
            (self.startup_timeout_s, "startup timeout"),
            (self.shutdown_timeout_s, "shutdown timeout"),
            (self.abort_grace_s, "abort grace"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for artifact in (
            *self.dependency_artifacts,
            self.split_artifact,
            self.sampling_artifact,
            self.model_lock_artifact,
            self.inventory_source_artifact,
            self.runtime_envelope_artifact,
        ):
            artifact.assert_unchanged()
        expected = tuple(sorted(self.expected_dependency_outputs))
        actual = tuple(
            sorted(
                (
                    str(artifact.experiment),
                    artifact.name,
                    artifact.content_sha256,
                )
                for artifact in self.dependency_artifacts
            )
        )
        if actual != expected:
            raise ValueError(
                "dependency artifacts do not cover the locked outputs exactly"
            )
        locked_runtime_envelopes = tuple(
            artifact
            for artifact in self.dependency_artifacts
            if artifact.name == "runtime_envelope"
        )
        if locked_runtime_envelopes != (self.runtime_envelope_artifact,):
            raise ValueError(
                "runtime envelope authority is not the exact locked dependency"
            )
        if self.dependency_receipt_sha256s != (
            self.runtime_plan.dependency_receipt_sha256s
        ):
            raise ValueError("execution receipt chain differs from the runtime plan")
        route_id = config.runtime.router_identity
        expected_warmup = _request_routes(self.load_plan.warmup, route_id=route_id)
        expected_scored = _request_routes(self.load_plan.scored, route_id=route_id)
        if (
            self.warmup_requests != expected_warmup
            or self.scored_requests != expected_scored
        ):
            raise ValueError("execution requests differ from the immutable load plan")
        for request in (*self.warmup_requests, *self.scored_requests):
            _validate_request_sampling(request, profile=sampling_profile)
        cohorts = {request.cohort_id for request in self.scored_requests}
        if len(cohorts) != cell.identity.cohort_count:
            raise ValueError(
                "observed cohort cardinality differs from the registry cell"
            )
        context = cell.identity.context
        if context is None or any(
            len(request.input_token_ids) + request.requested_output_tokens > context
            for request in (*self.warmup_requests, *self.scored_requests)
        ):
            raise ValueError("request corpus exceeds the cell context bound")
        loaded_compile_plan = _validate_server_launch(
            self.runtime_plan, self.server_launch
        )
        if (
            type(self.compile_cache_plan) is not CompileCacheLaunchPlan
            or loaded_compile_plan != self.compile_cache_plan
            or loaded_compile_plan.sha256 != self.compile_cache_plan.sha256
        ):
            raise ValueError("execution compile-cache plan changed after binding")
        preflight_compile_cache_launch(loaded_compile_plan)
        inventory_driver = _validate_inventory_source_artifact(
            self.inventory_source_artifact,
            inventory=self.dispatch_context.inventory,
        )
        if len(self.runtime_plan.physical_gpu_uuids) != 1:
            raise ValueError("compile-cache launch requires one assigned physical GPU")
        _validate_runtime_envelope_artifact(
            self.runtime_envelope_artifact,
            inventory=self.dispatch_context.inventory,
            checkout=Path(self.server_launch.argv[4]),
            compile_plan=loaded_compile_plan,
            config=config,
            inventory_driver=inventory_driver,
            assigned_gpu_uuid=self.runtime_plan.physical_gpu_uuids[0],
        )
        expected_bench = official_bench_argv(
            base_url=self.server_launch.base_url,
            served_model=config.model.target,
            request_count=len(self.scored_requests),
            concurrency=config.runtime.max_running_requests,
            arrival_kind=self.load_plan.scored.source_kind,
        )
        if self.bench_argv != expected_bench:
            raise ValueError("bench argv differs from the official adapter surface")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        capacity_authority = self.budget_plan.capacity_authority
        if capacity_authority is None:  # pragma: no cover - validation invariant
            raise RuntimeError("execution capacity authority disappeared")
        hashes = self.load_plan.scored.hashes
        itl_plan = _require_execution_itl_timestamp_authority(
            runtime_plan=self.runtime_plan,
            dispatch_context=self.dispatch_context,
        )
        return {
            "schema_version": 5,
            "runtime_plan_sha256": self.runtime_plan.sha256,
            "dispatch_plan_sha256": self.dispatch_plan.sha256,
            "dispatch_context_sha256": self.dispatch_context.sha256,
            "dispatch_plan": self.dispatch_plan.to_dict(),
            "dispatch_authority": self.dispatch_context.authority_dict(),
            "budget_plan_sha256": self.budget_plan.sha256,
            "capacity_authority_sha256": capacity_authority.sha256,
            "budget_materialization_authority_sha256": (
                self.dispatch_context.budget_materialization_authority.sha256
            ),
            "experiment_budget_sha256": self.budget.sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "topology_sha256": self.topology_sha256,
            "topology_receipt_sha256": self.runtime_plan.topology_receipt_sha256,
            "runtime_plan": self.runtime_plan.to_dict(),
            "itl_timestamp_authority": (
                None
                if itl_plan is None
                else {
                    "plan_sha256": itl_plan.sha256,
                    "producer_sha256": (
                        None if itl_plan.producer is None else itl_plan.producer.sha256
                    ),
                    "protocol_sha256": itl_plan.protocol_sha256,
                }
            ),
            "load": {
                "paired_replay_sha256": self.load_plan.paired_replay_sha256,
                "warmup_corpus_sha256": (
                    self.load_plan.warmup.hashes.corpus_sha256
                    if self.load_plan.warmup is not None
                    else None
                ),
                "scored_corpus_sha256": hashes.corpus_sha256,
                "request_ids_sha256": hashes.request_ids_sha256,
                "arrivals_sha256": hashes.arrivals_sha256,
                "cohorts_sha256": hashes.cohorts_sha256,
                "cancellations_sha256": hashes.cancellations_sha256,
                "window_sha256": self.load_plan.window.sha256,
            },
            "server_launch": asdict(self.server_launch),
            "compile_cache": {
                "plan_path": self.server_launch.compile_cache_plan,
                "plan_sha256": self.compile_cache_plan.sha256,
                "key_sha256": self.compile_cache_plan.key.sha256,
                "mode": self.compile_cache_plan.cache_mode,
                "builder_id": self.compile_cache_plan.builder_id,
            },
            "bench_adapter": ("sglang.benchmark.serving.async_request_sglang_generate"),
            "bench_argv": list(self.bench_argv),
            "evidence_writer_policy": self.evidence_writer_policy.to_dict(),
            "evidence_writer_policy_sha256": self.evidence_writer_policy.sha256,
            "dependency_receipt_sha256s": list(self.dependency_receipt_sha256s),
            "dependency_artifacts": [
                artifact.identity_dict()
                for artifact in sorted(
                    self.dependency_artifacts,
                    key=lambda value: (str(value.experiment), value.name),
                )
            ],
            "split_artifact": self.split_artifact.identity_dict(),
            "sampling_artifact": self.sampling_artifact.identity_dict(),
            "controlled_execution_policy_sha256": (
                self.runtime_plan.rank_configs[0].runtime.execution_policy_sha256
            ),
            "model_lock_artifact": self.model_lock_artifact.identity_dict(),
            "trainable_plan_authority": (
                None
                if self.trainable_plan_authority is None
                else trainable_plan_authority_binding_to_dict(
                    self.trainable_plan_authority
                )
            ),
            "failure_execution_authority": (
                None
                if self.failure_execution_authority is None
                else self.failure_execution_authority.to_dict()
            ),
            "prepared_model_content_release_manifest_sha256": (
                self.prepared_model_content_release_manifest_sha256
            ),
            "inventory_source_artifact": (
                self.inventory_source_artifact.identity_dict()
            ),
            "runtime_envelope_artifact": (
                self.runtime_envelope_artifact.identity_dict()
            ),
            "warmup_request_bindings": [
                request.sha256 for request in self.warmup_requests
            ],
            "scored_request_bindings": [
                request.sha256 for request in self.scored_requests
            ],
            "trusted_attester_policy": self.trusted_attester_policy.to_dict(),
            "trusted_attester_policy_sha256": self.trusted_attester_policy.sha256,
            "patched_sglang_tree": self.patched_sglang_tree,
            "startup_timeout_s": self.startup_timeout_s,
            "shutdown_timeout_s": self.shutdown_timeout_s,
            "abort_grace_s": self.abort_grace_s,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def rank_config_sha256(self) -> str:
        values = self.runtime_plan.to_dict()["rank_config_sha256s"]
        if (
            not isinstance(values, list)
            or len(values) != 1
            or not isinstance(values[0], str)
            or not _is_sha256(values[0])
        ):
            raise ValueError("runtime plan lacks exactly one rank-config digest")
        return values[0]

    @property
    def topology_sha256(self) -> str:
        """Canonical assigned topology identity consumed by execution/reduction."""

        config = self.runtime_plan.rank_configs[0]
        cell = self.runtime_plan.cell
        assignment = self.runtime_plan.physical_assignment
        if assignment is None:
            raise ValueError("topology identity requires a physical assignment")
        return content_sha256(
            {
                "schema_version": 1,
                "cell_id": cell.cell_id,
                "topology": cell.identity.topology,
                "topology_receipt_sha256": (self.runtime_plan.topology_receipt_sha256),
                "physical_assignment_sha256": assignment.assignment_sha256,
                "physical_binding_sha256": assignment.sha256,
                "physical_host_id": assignment.host_id,
                "physical_gpu_uuids": list(self.runtime_plan.physical_gpu_uuids),
                "physical_rank_groups": [
                    list(group) for group in self.runtime_plan.physical_rank_groups
                ],
                "physical_ports": list(self.runtime_plan.physical_ports),
                "topology_group_ids": [
                    list(group_ids) for group_ids in assignment.topology_group_ids
                ],
                "tensor_parallel_size": config.runtime.tensor_parallel_size,
                "data_parallel_size": config.runtime.data_parallel_size,
                "world_size": len(self.runtime_plan.rank_configs),
            }
        )


def _validate_server_launch(
    runtime_plan: IndustrialRuntimePlan,
    launch: ServerLaunch,
) -> CompileCacheLaunchPlan:
    if not runtime_plan.physical_dispatch_ready:
        raise ValueError("logical runtime plan cannot validate a server launch")
    config = runtime_plan.rank_configs[0]
    physical_gpu_uuids = runtime_plan.physical_gpu_uuids
    physical_rank_groups = runtime_plan.physical_rank_groups
    physical_ports = runtime_plan.physical_ports
    if (
        tuple(config.runtime.device_identity for config in runtime_plan.rank_configs)
        != physical_gpu_uuids
    ):
        raise ValueError("server RunConfigs do not bind the physical GPU rank order")
    if physical_rank_groups != (physical_gpu_uuids,):
        raise ValueError("released one-rank server requires one physical rank group")
    if launch.method != config.method or not launch.exclusive_device:
        raise ValueError("server launch method/isolation differs from the runtime plan")
    parsed = urlsplit(launch.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != physical_ports[0]
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("industrial server launch must use its reserved loopback port")
    config_path = Path(launch.run_config)
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("server launch lacks a regular run-config artifact")
    try:
        rendered = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("server run-config artifact is invalid") from error
    if rendered != config.model_dump(mode="json"):
        raise ValueError("server run-config artifact differs from the runtime plan")
    adapted = config.method not in {"target_only", "static"}
    if adapted != (
        launch.adaptation_config is not None and launch.telemetry_path is not None
    ):
        raise ValueError("server adaptation artifacts differ from the method contract")
    compile_plan = _load_server_compile_plan(launch)
    _validate_compile_key_for_run_config(compile_plan, config=config)
    argv = launch.argv
    if len(argv) < 20 or argv[:4] != (
        sys.executable,
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--checkout",
    ):
        raise ValueError("server launch argv is not the registered SGLang launcher")
    checkout = Path(argv[4])
    if (
        not checkout.is_dir()
        or argv[5] != "--compile-cache-plan"
        or argv[6] != launch.compile_cache_plan
        or argv[7] != "--"
    ):
        raise ValueError(
            "server argv lacks its checkout/compile-cache launch authority"
        )
    base = argv[8:20]
    expected_keys = (
        "--model-path",
        "--max-running-requests",
        "--mem-fraction-static",
        "--tp-size",
        "--host",
        "--port",
    )
    if base[::2] != expected_keys:
        raise ValueError("server base argv fields are incomplete or reordered")
    model_root = Path(base[1])
    if model_root.is_symlink() or not model_root.is_dir():
        raise ValueError("server target model root is not a regular directory")
    try:
        max_running = int(base[3])
        mem_fraction = float(base[5])
        tp_size = int(base[7])
        port = int(base[11])
    except ValueError as error:
        raise ValueError("server base argv values are malformed") from error
    if (
        max_running != config.runtime.max_running_requests
        or not math.isfinite(mem_fraction)
        or not 0 < mem_fraction < 1
        or tp_size != config.runtime.tensor_parallel_size
        or base[9] != parsed.hostname
        or port != parsed.port
    ):
        raise ValueError("server base argv differs from the RunConfig/base URL")
    remainder = argv[20:]
    role = _execution_role(config.method)
    execution_argv = tuple(_execution_argv(config.runtime, role=role))
    if remainder[: len(execution_argv)] != execution_argv:
        raise ValueError("server execution-policy argv differs from the RunConfig role")
    remainder = remainder[len(execution_argv) :]
    if config.method == "target_only":
        if remainder != ("--speculative-speed-study-metrics",):
            raise ValueError(
                "target-only argv requires only native terminal accounting after "
                "its execution policy"
            )
        return compile_plan
    if len(remainder) < 13:
        raise ValueError("speculative server argv is incomplete")
    expected_speculative_keys = (
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--speculative-num-draft-tokens",
        "--speculative-draft-window-size",
        "--speculative-accept-threshold-single",
        "--speculative-accept-threshold-acc",
    )
    if remainder[:12:2] != expected_speculative_keys:
        raise ValueError("speculative server argv fields are incomplete or reordered")
    drafter_root = Path(remainder[3])
    try:
        draft_tokens = int(remainder[5])
        draft_window = int(remainder[7])
        accept_single = float(remainder[9])
        accept_acc = float(remainder[11])
    except ValueError as error:
        raise ValueError("speculative server argv values are malformed") from error
    if (
        remainder[1] != config.model.algorithm
        or drafter_root.is_symlink()
        or not drafter_root.is_dir()
        or draft_tokens != config.runtime.speculative_num_draft_tokens
        or draft_window != config.runtime.speculative_num_draft_tokens
        or accept_single != 1.0
        or accept_acc != 1.0
        or remainder[12] != "--speculative-use-rejection-sampling"
    ):
        raise ValueError("speculative server argv differs from the RunConfig")
    adaptation_argv = remainder[13:]
    if not adapted:
        if adaptation_argv != ("--speculative-speed-study-metrics",):
            raise ValueError(
                "Static argv requires only the native terminal-evidence hook"
            )
        return compile_plan
    if not adaptation_argv or adaptation_argv[0] != "--speculative-speed-study-metrics":
        raise ValueError("adapted server argv lacks native speed-study evidence")
    adaptation_argv = adaptation_argv[1:]
    if len(adaptation_argv) != 6 or adaptation_argv[::2] != (
        "--speculative-adaptation-config",
        "--speculative-adaptation-reserve-mb",
        "--speculative-adaptation-telemetry-path",
    ):
        raise ValueError("adapted server argv fields are incomplete or reordered")
    if (
        adaptation_argv[1] != launch.adaptation_config
        or adaptation_argv[5] != launch.telemetry_path
    ):
        raise ValueError("adaptation argv paths differ from the launch contract")
    try:
        reserve_mb = int(adaptation_argv[3])
    except ValueError as error:
        raise ValueError("adaptation reserve is malformed") from error
    adaptation_path = Path(adaptation_argv[1])
    if reserve_mb < 1 or adaptation_path.is_symlink() or not adaptation_path.is_file():
        raise ValueError("adaptation reserve/config artifact is invalid")
    try:
        adaptation_payload = json.loads(adaptation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("adaptation config artifact is invalid") from error
    if adaptation_payload != sglang_adaptation_payload(config):
        raise ValueError("adaptation config artifact differs from the RunConfig")
    return compile_plan


def build_industrial_execution_plan(
    *,
    runtime_plan: IndustrialRuntimePlan,
    dispatch_plan: GpuDispatchPlan,
    dispatch_context: GpuDispatchExecutionContext,
    budget_plan: BudgetPlan,
    budget: ExperimentBudget,
    load_plan: ProductionLoadPlan,
    server_launch: ServerLaunch,
    dependency_receipts: tuple[ExperimentReceipt, ...],
    dependency_artifacts: tuple[ArtifactBinding, ...],
    split_artifact: ArtifactBinding,
    sampling_artifact: ArtifactBinding,
    model_lock_artifact: ArtifactBinding,
    compile_cache_plan: CompileCacheLaunchPlan,
    inventory_source_artifact: ArtifactBinding,
    runtime_envelope_artifact: ArtifactBinding,
    trainable_plan_authority: TrainablePlanAuthorityBinding | None = None,
    failure_execution_authority: FailureExecutionAuthorityToken | None = None,
    prepared_model_content_release_manifest_sha256: str | None = None,
    evidence_writer_policy: EvidenceWriterPolicy = DEFAULT_EVIDENCE_WRITER_POLICY,
    trusted_attester_policy: TrustedAttesterPolicy = NO_TRUSTED_ATTESTERS,
    startup_timeout_s: float = 300.0,
    shutdown_timeout_s: float = 30.0,
    abort_grace_s: float = 30.0,
) -> IndustrialExecutionPlan:
    """Bind a rendered runtime, exact trace, and content-locked artifacts."""

    if runtime_plan.cell.identity.experiment == "E1":
        _require_registered_e1_execution_recipe(
            registry=dispatch_context.registry,
            cell=runtime_plan.cell,
            execution_semantics=runtime_plan.execution_semantics,
        )
    _require_execution_itl_timestamp_authority(
        runtime_plan=runtime_plan,
        dispatch_context=dispatch_context,
    )
    receipt_sha256s = tuple(receipt.sha256 for receipt in dependency_receipts)
    if receipt_sha256s != runtime_plan.dependency_receipt_sha256s:
        raise ValueError("dependency receipt order differs from the runtime plan")
    expected = tuple(
        sorted(
            (
                (receipt.experiment, output.name, output.content_sha256)
                for receipt in dependency_receipts
                for output in receipt.outputs
            ),
            key=lambda row: (row[0], row[1]),
        )
    )
    route_id = runtime_plan.rank_configs[0].runtime.router_identity
    plan = IndustrialExecutionPlan(
        runtime_plan=runtime_plan,
        dispatch_plan=dispatch_plan,
        dispatch_context=dispatch_context,
        budget_plan=budget_plan,
        budget=budget,
        load_plan=load_plan,
        server_launch=server_launch,
        dependency_receipt_sha256s=receipt_sha256s,
        expected_dependency_outputs=expected,
        dependency_artifacts=dependency_artifacts,
        split_artifact=split_artifact,
        sampling_artifact=sampling_artifact,
        model_lock_artifact=model_lock_artifact,
        compile_cache_plan=compile_cache_plan,
        inventory_source_artifact=inventory_source_artifact,
        runtime_envelope_artifact=runtime_envelope_artifact,
        warmup_requests=_request_routes(load_plan.warmup, route_id=route_id),
        scored_requests=_request_routes(load_plan.scored, route_id=route_id),
        bench_argv=official_bench_argv(
            base_url=server_launch.base_url,
            served_model=runtime_plan.rank_configs[0].model.target,
            request_count=len(load_plan.scored.requests),
            concurrency=runtime_plan.rank_configs[0].runtime.max_running_requests,
            arrival_kind=load_plan.scored.source_kind,
        ),
        trainable_plan_authority=trainable_plan_authority,
        failure_execution_authority=failure_execution_authority,
        prepared_model_content_release_manifest_sha256=(
            prepared_model_content_release_manifest_sha256
        ),
        evidence_writer_policy=evidence_writer_policy,
        trusted_attester_policy=trusted_attester_policy,
        startup_timeout_s=startup_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
        abort_grace_s=abort_grace_s,
    )
    plan.validate()
    _ = plan.sha256
    return plan


def render_industrial_execution_plan(
    *,
    output_root: str | Path,
    runtime_plan: IndustrialRuntimePlan,
    dispatch_plan: GpuDispatchPlan,
    dispatch_context: GpuDispatchExecutionContext,
    budget_plan: BudgetPlan,
    budget: ExperimentBudget,
    load_plan: ProductionLoadPlan,
    dependency_receipts: tuple[ExperimentReceipt, ...],
    dependency_artifacts: tuple[ArtifactBinding, ...],
    split_artifact: ArtifactBinding,
    sampling_artifact: ArtifactBinding,
    model_lock_artifact: ArtifactBinding,
    sglang_checkout: str | Path,
    compile_cache_plan_path: str | Path,
    inventory_source_artifact: ArtifactBinding,
    runtime_envelope_artifact: ArtifactBinding,
    model_roots: Mapping[str, str],
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    trainable_plan_authority: TrainablePlanAuthorityBinding | None = None,
    prepared_model_content_release_manifest_sha256: str | None = None,
    evidence_writer_policy: EvidenceWriterPolicy = DEFAULT_EVIDENCE_WRITER_POLICY,
    trusted_attester_policy: TrustedAttesterPolicy = NO_TRUSTED_ATTESTERS,
    host: str = "127.0.0.1",
) -> IndustrialExecutionPlan:
    """Render one argv-only launch, then bind it to the execution plan."""

    if type(runtime_plan) is not IndustrialRuntimePlan:
        _require_adapted_execution_semantics_sha256(runtime_plan)
    # Replay scientific identity before the trainable-plan gate or renderer can
    # inspect/write any launch artifact.  ``IndustrialRuntimePlan`` is a public
    # value object, so its caller may have replaced a recipe and recomputed all
    # local digests after construction.
    validate_industrial_execution_semantics_authority(
        runtime_plan=runtime_plan,
        dispatch_context=dispatch_context,
        registered_load=load_plan,
    )
    _require_execution_itl_timestamp_authority(
        runtime_plan=runtime_plan,
        dispatch_context=dispatch_context,
    )
    _require_render_trainable_plan_authority(
        runtime_plan=runtime_plan,
        model_lock_artifact=model_lock_artifact,
        split_artifact=split_artifact,
        model_roots=model_roots,
        authority=trainable_plan_authority,
        release_pin=prepared_model_content_release_manifest_sha256,
    )
    _validate_runtime_dispatch_authority(
        runtime_plan=runtime_plan,
        dispatch_plan=dispatch_plan,
        dispatch_context=dispatch_context,
        budget_plan=budget_plan,
        budget=budget,
    )
    if not runtime_plan.physical_dispatch_ready:
        raise ValueError(
            "logical runtime plan cannot render a server; physical assignment required"
        )
    config = runtime_plan.rank_configs[0]
    target_id = config.model.target
    drafter_id = config.model.drafter
    required_roots = (
        (target_id,) if config.method == "target_only" else (target_id, drafter_id)
    )
    roots = dict(model_roots)
    for model_id in required_roots:
        root = roots.get(model_id)
        if not isinstance(root, str) or not Path(root).is_dir():
            raise ValueError(f"verified local model root is missing: {model_id}")
    adapted = config.method not in {"target_only", "static"}
    if adapted != (adaptation_reserve_mb > 0):
        raise ValueError(
            "adaptation reserve must be positive exactly for adapted methods"
        )
    if not 0 < mem_fraction_static < 1:
        raise ValueError("mem_fraction_static must lie in (0, 1)")
    verified = verify_patched_checkout(sglang_checkout)
    compile_cache_plan = CompileCacheLaunchPlan.load(compile_cache_plan_path)
    launch = _render_server(
        output=Path(output_root).resolve(),
        method=config.method,
        config=config,
        verified_checkout=verified,
        roots=roots,
        target_id=target_id,
        drafter_id=drafter_id,
        adaptation_reserve_mb=adaptation_reserve_mb,
        mem_fraction_static=mem_fraction_static,
        host=host,
        port=runtime_plan.physical_ports[0],
        compile_cache_plan_path=compile_cache_plan_path,
    )
    if config.method == "static":
        launch = replace(
            launch,
            argv=(*launch.argv, "--speculative-speed-study-metrics"),
        )
    plan = build_industrial_execution_plan(
        runtime_plan=runtime_plan,
        dispatch_plan=dispatch_plan,
        dispatch_context=dispatch_context,
        budget_plan=budget_plan,
        budget=budget,
        load_plan=load_plan,
        server_launch=launch,
        dependency_receipts=dependency_receipts,
        dependency_artifacts=dependency_artifacts,
        split_artifact=split_artifact,
        sampling_artifact=sampling_artifact,
        model_lock_artifact=model_lock_artifact,
        compile_cache_plan=compile_cache_plan,
        inventory_source_artifact=inventory_source_artifact,
        runtime_envelope_artifact=runtime_envelope_artifact,
        trainable_plan_authority=trainable_plan_authority,
        prepared_model_content_release_manifest_sha256=(
            prepared_model_content_release_manifest_sha256
        ),
        evidence_writer_policy=evidence_writer_policy,
        trusted_attester_policy=trusted_attester_policy,
    )
    _immutable_json(
        Path(output_root).resolve() / "industrial-execution-plan.json",
        plan.to_dict(),
    )
    return plan


@dataclass(frozen=True)
class RequestExecution:
    request: BoundServingRequest
    outcome: RequestOutcome
    result: BenchServingResult | None

    def timing(self) -> TimingCoverage:
        output_tokens = self.result.output_tokens if self.result is not None else 0
        admitted = self.outcome.admitted_at_us
        offered = (
            self.outcome.offered_at_us
            if self.outcome.offered_at_us is not None
            else self.request.arrival_us
        )
        if admitted is None or self.result is None or not self.result.chunks:
            return TimingCoverage(
                request_id=self.request.request_id,
                output_tokens=output_tokens,
                expected_itl_intervals=max(0, output_tokens - 1),
                supported_itl_intervals=0,
                coalesced_tokens=output_tokens,
                itl_coverage=1.0 if output_tokens <= 1 else 0.0,
                ttft_us=None,
                supported_itls_us=(),
            )
        terminal = self.outcome.terminal_at_us
        absolute = tuple(
            TokenChunkTiming(
                request_id=chunk.request_id,
                first_token_index=chunk.first_token_index,
                token_count=chunk.token_count,
                chunk_observed_at_us=admitted + chunk.chunk_observed_at_us,
                per_token_observed_at_us=(
                    tuple(admitted + value for value in chunk.per_token_observed_at_us)
                    if chunk.per_token_observed_at_us is not None
                    else None
                ),
            )
            for chunk in self.result.chunks
        )
        if terminal is not None and any(
            chunk.chunk_observed_at_us > terminal for chunk in absolute
        ):
            return TimingCoverage(
                request_id=self.request.request_id,
                output_tokens=output_tokens,
                expected_itl_intervals=max(0, output_tokens - 1),
                supported_itl_intervals=0,
                coalesced_tokens=output_tokens,
                itl_coverage=1.0 if output_tokens <= 1 else 0.0,
                ttft_us=None,
                supported_itls_us=(),
            )
        timing = evaluate_token_timing(
            request_id=self.request.request_id,
            request_started_us=offered,
            expected_output_tokens=output_tokens,
            chunks=absolute,
        )
        if self.result.ttft_us is not None:
            observed_ttft = admitted + self.result.ttft_us - offered
            if timing.ttft_us is not None and timing.ttft_us != observed_ttft:
                raise ValueError("bench chunk timing disagrees with its exact TTFT")
            timing = replace(timing, ttft_us=observed_ttft)
        return timing


@dataclass(frozen=True)
class NativeEvidenceBatch:
    """Server-native rows and aggregate fields unavailable to the HTTP client."""

    rounds: tuple[RoundRecord, ...] = ()
    updates: tuple[UpdateRecord, ...] = ()
    performance_overrides: tuple[tuple[str, object], ...] = ()

    def validate(self, *, run_id: str, method: str) -> None:
        if any(row.run_id != run_id for row in (*self.rounds, *self.updates)):
            raise ValueError("native evidence contains a cross-run row")
        if len({(row.request_id, row.round_index) for row in self.rounds}) != len(
            self.rounds
        ):
            raise ValueError("native round evidence contains duplicate identities")
        if len({(row.cohort_sha256, row.update_index) for row in self.updates}) != len(
            self.updates
        ):
            raise ValueError("native update evidence contains duplicate identities")
        keys = tuple(key for key, _ in self.performance_overrides)
        if len(keys) != len(set(keys)):
            raise ValueError("performance override fields must be unique")
        protected = {
            "run_id",
            "prompt_id",
            "method",
            "repetition_block",
            "region",
            "concurrency",
            "generated_bucket_start",
            "generated_bucket_end",
            "at_risk_requests",
            "output_tokens",
            "elapsed_s",
            "decode_goodput_tps",
            "itl_p50_ms",
            "itl_p95_ms",
            "itl_p99_ms",
            "admission_rejections",
            "timeouts",
            "cancellations",
            "offered_requests",
            "admitted_requests",
            "completed_requests",
            "unfinished_requests",
            "evidence_backpressure_events",
            "evidence_dropped_rows",
        }
        available = {field.name for field in fields(PerformanceRecord)} - protected
        if any(key not in available for key in keys):
            raise ValueError(
                "native performance override targets a protected/unknown field"
            )
        for _, value in self.performance_overrides:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("native performance values must be finite")
        if method in {"target_only", "static"} and (self.rounds or self.updates):
            raise ValueError(
                "allocation-free execution cannot contain round or update rows"
            )
        overrides = dict(self.performance_overrides)
        adaptation_fields = {
            "optimizer_bytes",
            "adaptation_memory_ledger",
            "trainable_parameters",
            "training_cuda_ms",
            "optimizer_cuda_ms",
            "merge_cuda_ms",
            "publish_cuda_ms",
            "barrier_cuda_ms",
            "exposed_update_ms",
            "main_side_overlap_ratio",
            "updates_launched",
            "updates_published",
        }
        if method in {"target_only", "static"} and any(
            overrides.get(name) not in {None, 0} for name in adaptation_fields
        ):
            raise ValueError("allocation-free methods cannot report adaptation state")
        if method not in {"target_only", "static"} and (
            not self.rounds or not self.updates
        ):
            raise ValueError("adapted execution requires native round/update evidence")
        if method not in {"target_only", "static"} and (
            not isinstance(overrides.get("updates_launched"), int)
            or not isinstance(overrides.get("updates_published"), int)
            or int(overrides["updates_launched"]) < 1
            or int(overrides["updates_published"]) < 1
        ):
            raise ValueError("adapted execution requires positive update counters")


@dataclass(frozen=True)
class NativeEvidencePreflight:
    """Machine-readable gate for the native speculative evidence boundary."""

    status: str
    reason_code: str | None
    missing_hook: str | None
    required_fields: tuple[str, ...]

    def validate(self) -> None:
        if self.status == "READY":
            if self.reason_code is not None or self.missing_hook is not None:
                raise ValueError("READY native evidence preflight cannot be blocked")
        elif self.status == "BLOCKED":
            missing_hook = (
                self.reason_code == MISSING_NATIVE_EVIDENCE_REASON
                and self.missing_hook == NATIVE_TERMINAL_EVIDENCE_HOOK
            )
            unavailable_trust = (
                self.reason_code == TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON
                and self.missing_hook is None
            )
            if not (missing_hook or unavailable_trust):
                raise ValueError("BLOCKED native evidence preflight is ambiguous")
        else:
            raise ValueError("native evidence preflight status is invalid")
        if self.required_fields != NATIVE_TERMINAL_EVIDENCE_FIELDS:
            raise ValueError("native evidence preflight fields differ from the hook")


class NativeEvidenceUnavailableError(RuntimeError):
    """Raised before mutation when the pinned server lacks exact evidence."""

    def __init__(self, preflight: NativeEvidencePreflight) -> None:
        preflight.validate()
        if preflight.status != "BLOCKED":
            raise ValueError("only a BLOCKED preflight can be unavailable")
        self.preflight = preflight
        if preflight.missing_hook is None:
            super().__init__(str(preflight.reason_code))
        else:
            fields = ",".join(preflight.required_fields)
            super().__init__(
                f"{preflight.reason_code}: missing hook {preflight.missing_hook}; "
                f"required fields={fields}"
            )


# Compatibility spelling for callers while the implementation is now one
# concrete, wire-validating begin/reset/finalize provider rather than a
# caller-authored ``collect`` protocol.
NativeEvidenceProvider = NativeTerminalProvider


def _is_exact_native_terminal_provider(value: object) -> bool:
    if type(value) is not NativeTerminalProvider:
        return False
    instance_fields = vars(value)
    return not any(
        name in instance_fields for name in ("capability", "begin", "reset", "finalize")
    )


def native_evidence_preflight(
    plan: IndustrialExecutionPlan,
    provider: NativeTerminalProvider | None,
) -> NativeEvidencePreflight:
    """Bind the sole wire provider and release policy before process mutation."""

    method = plan.runtime_plan.rank_configs[0].method
    if not _is_exact_native_terminal_provider(provider):
        status = "BLOCKED"
        reason_code = MISSING_NATIVE_EVIDENCE_REASON
        missing_hook = NATIVE_TERMINAL_EVIDENCE_HOOK
    else:
        try:
            release_policy = require_release_trusted_attester_policy(
                plan.trusted_attester_policy
            )
        except (AttributeError, TypeError, ValueError):
            release_policy = None
        if (
            release_policy is None
            or provider.trusted_attester_policy_sha256 != release_policy.sha256
            or (method != "target_only" and not release_policy.release_ready)
        ):
            status = "BLOCKED"
            reason_code = TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON
            missing_hook = None
        else:
            status = "READY"
            reason_code = None
            missing_hook = None
    value = NativeEvidencePreflight(
        status=status,
        reason_code=reason_code,
        missing_hook=missing_hook,
        required_fields=NATIVE_TERMINAL_EVIDENCE_FIELDS,
    )
    value.validate()
    return value


class ServerHandle(Protocol):
    async def wait_ready(self, timeout_s: float) -> None: ...

    async def terminate(self, timeout_s: float) -> None: ...


ServerLauncher = Callable[[ServerLaunch], Awaitable[ServerHandle]]


class AsyncSubprocessServerHandle:
    """Opt-in local process handle for the validated launcher argv."""

    def __init__(
        self,
        *,
        process: asyncio.subprocess.Process,
        base_url: str,
        poll_interval_s: float = 0.1,
    ) -> None:
        if process.pid is None or process.pid < 1:
            raise RuntimeError("server subprocess has no process identity")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("server poll interval must be finite and positive")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
        ):
            raise ValueError("server readiness checks require a loopback HTTP URL")
        self._process = process
        self._base_url = base_url.rstrip("/")
        self._poll_interval_s = poll_interval_s

    def _ready(self) -> bool:
        request = Request(self._base_url + "/v1/models", method="GET")
        try:
            with urlopen(request, timeout=2.0) as response:
                return int(response.status) == 200
        except (HTTPError, URLError, TimeoutError, OSError):
            return False

    async def wait_ready(self, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("startup timeout must be finite and positive")
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            if self._process.returncode is not None:
                raise RuntimeError(
                    f"SGLang server exited during startup ({self._process.returncode})"
                )
            if await asyncio.to_thread(self._ready):
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("SGLang server did not become ready")
            await asyncio.sleep(min(self._poll_interval_s, remaining))

    async def terminate(self, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("shutdown timeout must be finite and positive")
        if self._process.returncode is not None:
            return
        try:
            self._process.terminate()
        except ProcessLookupError:
            await self._process.wait()
            return
        try:
            await asyncio.wait_for(self._process.wait(), timeout=timeout_s)
        except TimeoutError:
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
            await self._process.wait()


async def launch_server_subprocess(launch: ServerLaunch) -> ServerHandle:
    """Explicitly opt into launching the already-validated local server argv."""

    parsed = urlsplit(launch.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or not launch.argv
        or launch.argv[:3]
        != (sys.executable, "-m", "lightcone_spec.sglang_bridge.launch")
    ):
        raise ValueError("subprocess launch requires the registered loopback launcher")
    if (
        len(launch.argv) < 8
        or launch.argv[3] != "--checkout"
        or launch.argv[5] != "--compile-cache-plan"
        or launch.argv[6] != launch.compile_cache_plan
        or launch.argv[7] != "--"
    ):
        raise ValueError("subprocess launch lacks its exact compile-cache argv")
    compile_plan = _load_server_compile_plan(launch)
    preflight_compile_cache_launch(compile_plan)
    config_path = Path(launch.run_config)
    try:
        config = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("subprocess launch lacks a device-bound RunConfig") from error
    if launch.method != config.method:
        raise ValueError("subprocess launch method differs from its RunConfig")
    _validate_compile_key_for_run_config(compile_plan, config=config)
    verify_patched_checkout(launch.argv[4])
    device_identity = config.runtime.device_identity
    if (
        not isinstance(device_identity, str)
        or not device_identity.startswith("GPU-")
        or "\n" in device_identity
        or "," in device_identity
    ):
        raise ValueError("subprocess launch requires exactly one GPU UUID")
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = device_identity
    process = await asyncio.create_subprocess_exec(*launch.argv, env=environment)
    return AsyncSubprocessServerHandle(process=process, base_url=launch.base_url)


@dataclass(frozen=True)
class ExecutionClock:
    monotonic_ns: Callable[[], int] = time.perf_counter_ns
    wall_ns: Callable[[], int] = time.time_ns
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass(frozen=True)
class IndustrialExecutionResult:
    run_id: str
    execution_plan_sha256: str
    experiment_budget_sha256: str
    rank_config_sha256: str
    topology_sha256: str
    resumed: bool
    terminal_receipt: str
    terminal_receipt_sha256: str
    budget_observation: str
    budget_observation_sidecar: str
    budget_observation_sha256: str
    evidence_files: tuple[str, ...]
    accounting: LoadAccounting | None


@dataclass(frozen=True)
class IndustrialExecutionTerminalBinding:
    """Freshly revalidated raw-file authority for one serving assignment."""

    cell_id: str
    run_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    dispatch_plan_sha256: str
    assignment_sha256: str
    experiment_budget_sha256: str
    inventory_sha256: str
    physical_gpu_uuids: tuple[str, ...]
    terminal_receipt_path: str
    terminal_receipt_sha256: str
    budget_observation_path: str
    budget_observation_sha256: str
    budget_observation_sidecar_path: str
    budget_observation_sidecar_sha256: str
    native_terminal_artifact_path: str
    native_terminal_raw_sha256: str
    native_terminal_sha256: str
    trusted_attester_policy_sha256: str
    trusted_attestation: bool
    evidence_file_paths: tuple[str, ...]
    evidence_file_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "run_nonce_sha256",
            "execution_plan_sha256",
            "dispatch_plan_sha256",
            "assignment_sha256",
            "experiment_budget_sha256",
            "inventory_sha256",
            "terminal_receipt_sha256",
            "budget_observation_sha256",
            "budget_observation_sidecar_sha256",
            "native_terminal_raw_sha256",
            "native_terminal_sha256",
            "trusted_attester_policy_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not self.run_id or "\n" in self.run_id:
            raise ValueError("terminal binding run_id is invalid")
        if type(self.trusted_attestation) is not bool:
            raise TypeError("terminal binding trust status must be a boolean")
        if not self.physical_gpu_uuids or len(set(self.physical_gpu_uuids)) != len(
            self.physical_gpu_uuids
        ):
            raise ValueError("terminal binding GPU coverage is invalid")
        paths = (
            self.terminal_receipt_path,
            self.budget_observation_path,
            self.budget_observation_sidecar_path,
            self.native_terminal_artifact_path,
            *self.evidence_file_paths,
        )
        if any(
            not Path(value).is_absolute() or Path(value).resolve() != Path(value)
            for value in paths
        ):
            raise ValueError("terminal binding paths must be absolute and resolved")
        if (
            not self.evidence_file_paths
            or len(self.evidence_file_paths) != len(self.evidence_file_sha256s)
            or len(set(self.evidence_file_paths)) != len(self.evidence_file_paths)
            or tuple(sorted(self.evidence_file_paths)) != self.evidence_file_paths
            or any(not _is_sha256(value) for value in self.evidence_file_sha256s)
        ):
            raise ValueError("terminal binding evidence-file coverage is invalid")

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "industrial_execution_terminal_binding",
                **asdict(self),
            }
        )


def industrial_run_id(plan: IndustrialExecutionPlan, run_nonce_sha256: str) -> str:
    if not _is_sha256(run_nonce_sha256):
        raise ValueError("run nonce must be a lowercase SHA-256")
    return (
        "industrial-"
        + content_sha256(
            {
                "schema_version": 1,
                "execution_plan_sha256": plan.sha256,
                "run_nonce_sha256": run_nonce_sha256,
                "rank": 0,
            }
        )[:48]
    )


def _preflight_industrial_session_trace_evidence(
    plan: IndustrialExecutionPlan,
    *,
    output_root: str | Path,
    run_nonce_sha256: str,
) -> None:
    """Reject any pre-existing trace state before a shared server is launched.

    Shared sessions deliberately do not resume an earlier logical trace: the
    native process boundary and its startup/reset accounting would otherwise
    differ from the evidence already on disk.  This check is read-only and is
    called for every planned trace before ``open_server_session`` performs any
    process or network mutation.
    """

    plan.validate()
    run_id = industrial_run_id(plan, run_nonce_sha256)
    requested_root = Path(output_root)
    root = requested_root.resolve()
    registered_root = Path(plan.runtime_plan.cell.resources.evidence_root).resolve()
    if root != registered_root:
        raise ValueError("output_root differs from the registry evidence reservation")
    if os.path.lexists(requested_root) and requested_root.is_symlink():
        raise RuntimeError("shared-session evidence root cannot be a symlink")
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("shared-session evidence root is not a regular directory")
    public_prefix = f"{run_id}.rank0."
    private_prefix = f".{run_id}."
    try:
        preexisting = any(
            entry.name.startswith(public_prefix)
            or entry.name.startswith(private_prefix)
            for entry in root.iterdir()
        )
    except OSError as error:
        raise RuntimeError("shared-session evidence root is unreadable") from error
    if preexisting:
        raise RuntimeError(
            "shared-session execution cannot resume preexisting trace evidence"
        )


def _native_terminal_binding(
    *,
    plan: IndustrialExecutionPlan,
    run_id: str,
    run_nonce_sha256: str,
    session_binding: SessionExecutionBinding | None,
) -> NativeTerminalRunBinding:
    if session_binding is None:
        session_id = content_sha256(
            {
                "schema_version": 1,
                "kind": "standalone_native_terminal_session",
                "execution_plan_sha256": plan.sha256,
                "run_nonce_sha256": run_nonce_sha256,
            }
        )
        session_epoch = 1
        previous_run_id = None
    else:
        session_id = session_binding.native_session_id
        session_epoch = session_binding.native_trace_epoch
        previous_run_id = session_binding.native_previous_run_id
    binding = NativeTerminalRunBinding(
        run_id=run_id,
        run_nonce_sha256=run_nonce_sha256,
        execution_plan_sha256=plan.sha256,
        rank_config_sha256=plan.rank_config_sha256,
        attempt_id=f"{run_id}.attempt.{os.urandom(8).hex()}",
        session_id=session_id,
        session_epoch=session_epoch,
        previous_run_id=previous_run_id,
        challenge_nonce_sha256=hashlib.sha256(os.urandom(32)).hexdigest(),
        method=plan.runtime_plan.rank_configs[0].method,
        warmup_request_ids=tuple(
            request.request_id for request in plan.warmup_requests
        ),
        scored_request_ids=tuple(
            request.request_id for request in plan.scored_requests
        ),
    )
    binding.validate()
    return binding


def _terminal_request_expectation(
    execution: RequestExecution,
) -> TerminalRequestExpectation:
    outcome = execution.outcome
    result = execution.result
    submitted = outcome.admitted_at_us is not None
    if submitted:
        if outcome.status == "completed":
            if (
                result is None
                or result.output_tokens != execution.request.requested_output_tokens
            ):
                raise RuntimeError(
                    "submitted completion lacks an exact FINISH_LENGTH terminal"
                )
            terminal_status = "completed"
            terminal_reason = "FINISH_LENGTH"
        elif outcome.status in {"cancelled", "timed_out"}:
            if result is None:
                raise RuntimeError("aborted native request lacks its terminal result")
            terminal_status = "aborted"
            terminal_reason = "FINISH_ABORT"
        else:
            raise RuntimeError(
                "submitted request lacks a reconciliable terminal outcome"
            )
        output_token_ids: tuple[int, ...] | None = result.generated_token_ids
    else:
        if outcome.status not in {"rejected", "cancelled", "timed_out"}:
            raise RuntimeError("non-submitted request lacks a client terminal outcome")
        terminal_status = outcome.status
        terminal_reason = outcome.code
        output_token_ids = None
    expectation = TerminalRequestExpectation(
        request_id=execution.request.request_id,
        input_token_ids=execution.request.input_token_ids,
        output_token_ids=output_token_ids,
        terminal_status=terminal_status,
        terminal_reason=terminal_reason,
        submitted_to_server=submitted,
    )
    expectation.validate()
    return expectation


def _bind_native_terminal_transport(
    *,
    provider: NativeTerminalProvider,
    transport: BenchServingTransport,
    base_url: str,
) -> PinnedBenchServingTransport:
    """Prove native admin and serving requests share one official live pool."""

    if not _is_exact_native_terminal_provider(provider):
        raise TypeError("native evidence requires the concrete pinned wire provider")
    if not isinstance(transport, PinnedBenchServingTransport):
        raise TypeError("native evidence requires the pinned official bench transport")
    # NativeTerminalProvider intentionally keeps the admin boundary private;
    # identity comparison is used only to prove pool sharing, never capability.
    if provider._transport is not transport:
        raise ValueError("native admin and serving traffic use different HTTP pools")
    transport.bind_native_admin_base_url(base_url)
    return transport


async def _require_live_controlled_execution_policy(
    *,
    transport: PinnedBenchServingTransport,
    config: RunConfig,
) -> str:
    """Reopen the exact live server policy through the bound serving pool."""

    if type(config) is not RunConfig:
        raise TypeError("live execution-policy validation requires an exact RunConfig")
    policy = _runtime_execution_policy(config.runtime)
    if type(policy) is not ControlledExecutionPolicy:  # pragma: no cover
        raise TypeError("runtime did not resolve an exact controlled execution policy")
    server_info = await transport.get_json("/server_info")
    try:
        policy.validate_server_info(
            server_info,
            role=_execution_role(config.method),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "live server differs from the registered controlled execution policy"
        ) from error
    return policy.sha256


async def _sleep_until(
    target_us: int,
    *,
    origin_ns: int,
    clock: ExecutionClock,
) -> None:
    remaining = target_us - (clock.monotonic_ns() - origin_ns) // 1000
    if remaining > 0:
        await clock.sleep(remaining / 1_000_000)


def _logical_now_us(*, origin_ns: int, clock: ExecutionClock) -> int:
    return max(0, (clock.monotonic_ns() - origin_ns) // 1000)


async def _execute_request(
    request: BoundServingRequest,
    *,
    deadline_us: int,
    origin_ns: int,
    semaphore: asyncio.Semaphore,
    transport: BenchServingTransport,
    base_url: str,
    served_model: str,
    abort_grace_s: float,
    clock: ExecutionClock,
    offered_at_us: int | None = None,
) -> RequestExecution:
    arrival_us = request.arrival_us if offered_at_us is None else offered_at_us
    await _sleep_until(arrival_us, origin_ns=origin_ns, clock=clock)
    cancel_us = (
        arrival_us + request.cancellation_offset_us
        if request.cancellation_offset_us is not None
        else None
    )
    boundary = min(deadline_us, cancel_us) if cancel_us is not None else deadline_us
    if cancel_us is not None and cancel_us <= arrival_us:
        return RequestExecution(
            request=request,
            outcome=RequestOutcome(
                request_id=request.request_id,
                status="cancelled",
                admitted_at_us=None,
                terminal_at_us=cancel_us,
                code="scheduled_cancellation_before_admission",
                offered_at_us=arrival_us,
            ),
            result=None,
        )

    acquired = False
    acquire_task = asyncio.create_task(semaphore.acquire())
    try:
        remaining = boundary - _logical_now_us(origin_ns=origin_ns, clock=clock)
        if remaining <= 0:
            acquire_task.cancel()
        else:
            done, _ = await asyncio.wait({acquire_task}, timeout=remaining / 1_000_000)
            acquired = acquire_task in done and not acquire_task.cancelled()
        if not acquired:
            acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
            cancelled = cancel_us is not None and cancel_us <= deadline_us
            return RequestExecution(
                request=request,
                outcome=RequestOutcome(
                    request_id=request.request_id,
                    status="cancelled" if cancelled else "rejected",
                    admitted_at_us=None,
                    terminal_at_us=cancel_us if cancelled else deadline_us,
                    code=(
                        "scheduled_cancellation_before_admission"
                        if cancelled
                        else "admission_deadline"
                    ),
                    offered_at_us=arrival_us,
                ),
                result=None,
            )
        admitted_us = max(
            arrival_us,
            _logical_now_us(origin_ns=origin_ns, clock=clock),
        )
        if admitted_us > boundary:
            cancelled = cancel_us is not None and cancel_us <= deadline_us
            return RequestExecution(
                request=request,
                outcome=RequestOutcome(
                    request_id=request.request_id,
                    status="cancelled" if cancelled else "rejected",
                    admitted_at_us=None,
                    terminal_at_us=cancel_us if cancelled else deadline_us,
                    code=(
                        "scheduled_cancellation_before_admission"
                        if cancelled
                        else "admission_deadline"
                    ),
                    offered_at_us=arrival_us,
                ),
                result=None,
            )
        submit_task = asyncio.create_task(
            transport.submit(
                request,
                base_url=base_url,
                served_model=served_model,
            )
        )
        try:
            remaining = max(
                0,
                boundary - _logical_now_us(origin_ns=origin_ns, clock=clock),
            )
            done, _ = await asyncio.wait({submit_task}, timeout=remaining / 1_000_000)
            if submit_task in done:
                try:
                    result = submit_task.result()
                    result.validate(request)
                except Exception as error:  # noqa: BLE001 - terminal request boundary
                    return RequestExecution(
                        request=request,
                        outcome=RequestOutcome(
                            request_id=request.request_id,
                            status="unfinished",
                            admitted_at_us=admitted_us,
                            terminal_at_us=None,
                            code=f"bench_transport_or_result_error:{type(error).__name__}",
                            offered_at_us=arrival_us,
                        ),
                        result=None,
                    )
                terminal = _logical_now_us(origin_ns=origin_ns, clock=clock)
                if terminal <= boundary:
                    if not result.success:
                        return RequestExecution(
                            request=request,
                            outcome=RequestOutcome(
                                request_id=request.request_id,
                                status="unfinished",
                                admitted_at_us=admitted_us,
                                terminal_at_us=None,
                                code=f"official_bench_error:{result.error_code}",
                                offered_at_us=arrival_us,
                            ),
                            result=result,
                        )
                    return RequestExecution(
                        request=request,
                        outcome=RequestOutcome(
                            request_id=request.request_id,
                            status="completed",
                            admitted_at_us=admitted_us,
                            terminal_at_us=max(admitted_us, terminal),
                            code="completed",
                            offered_at_us=arrival_us,
                        ),
                        result=result,
                    )
            try:
                await transport.abort(request.request_id, base_url=base_url)
            except Exception as error:  # noqa: BLE001 - transport boundary
                submit_task.cancel()
                await asyncio.gather(submit_task, return_exceptions=True)
                return RequestExecution(
                    request=request,
                    outcome=RequestOutcome(
                        request_id=request.request_id,
                        status="unfinished",
                        admitted_at_us=admitted_us,
                        terminal_at_us=None,
                        code=f"abort_transport_error:{type(error).__name__}",
                        offered_at_us=arrival_us,
                    ),
                    result=None,
                )
            try:
                result = await asyncio.wait_for(submit_task, timeout=abort_grace_s)
            except TimeoutError:
                submit_task.cancel()
                await asyncio.gather(submit_task, return_exceptions=True)
                return RequestExecution(
                    request=request,
                    outcome=RequestOutcome(
                        request_id=request.request_id,
                        status="unfinished",
                        admitted_at_us=admitted_us,
                        terminal_at_us=None,
                        code="abort_terminal_evidence_timeout",
                        offered_at_us=arrival_us,
                    ),
                    result=None,
                )
            try:
                result.validate(request)
            except Exception as error:  # noqa: BLE001 - terminal request boundary
                return RequestExecution(
                    request=request,
                    outcome=RequestOutcome(
                        request_id=request.request_id,
                        status="unfinished",
                        admitted_at_us=admitted_us,
                        terminal_at_us=None,
                        code=f"aborted_result_invalid:{type(error).__name__}",
                        offered_at_us=arrival_us,
                    ),
                    result=None,
                )
            cancelled = cancel_us is not None and cancel_us <= deadline_us
            return RequestExecution(
                request=request,
                outcome=RequestOutcome(
                    request_id=request.request_id,
                    status="cancelled" if cancelled else "timed_out",
                    admitted_at_us=admitted_us,
                    terminal_at_us=cancel_us if cancelled else deadline_us,
                    code="scheduled_cancellation" if cancelled else "request_deadline",
                    offered_at_us=arrival_us,
                ),
                result=result,
            )
        except BaseException:
            if not submit_task.done():
                submit_task.cancel()
                await asyncio.gather(submit_task, return_exceptions=True)
            raise
    finally:
        if not acquire_task.done():
            acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
        elif (
            not acquired
            and not acquire_task.cancelled()
            and acquire_task.exception() is None
            and bool(acquire_task.result())
        ):
            semaphore.release()
        if acquired:
            semaphore.release()


async def _execute_corpus(
    requests: tuple[BoundServingRequest, ...],
    *,
    deadline_for: Callable[[BoundServingRequest], int],
    concurrency: int,
    transport: BenchServingTransport,
    base_url: str,
    served_model: str,
    abort_grace_s: float,
    clock: ExecutionClock,
    on_terminal: Callable[[RequestExecution], Awaitable[None]] | None = None,
) -> tuple[RequestExecution, ...]:
    if not requests:
        return ()
    if len(requests) > MAX_IN_MEMORY_REQUEST_EXECUTIONS:
        raise ValueError("request corpus exceeds the bounded execution capacity")
    origin_ns = clock.monotonic_ns()
    semaphore = asyncio.Semaphore(concurrency)
    completed: dict[str, RequestExecution] = {}
    errors: list[Exception] = []

    async def run_one(request: BoundServingRequest) -> None:
        try:
            execution = await _execute_request(
                request,
                deadline_us=deadline_for(request),
                origin_ns=origin_ns,
                semaphore=semaphore,
                transport=transport,
                base_url=base_url,
                served_model=served_model,
                abort_grace_s=abort_grace_s,
                clock=clock,
            )
            completed[request.request_id] = execution
            if on_terminal is not None:
                await on_terminal(execution)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(error)

    async with asyncio.TaskGroup() as group:
        for request in requests:
            group.create_task(run_one(request))
    if errors:
        for sibling in errors[1:]:
            errors[0].add_note(f"sibling request also failed: {sibling}")
        raise errors[0]
    if set(completed) != {request.request_id for request in requests}:
        raise RuntimeError("request execution coverage is incomplete")
    return tuple(completed[request.request_id] for request in requests)


async def _execute_closed_loop_corpus(
    requests: tuple[BoundServingRequest, ...],
    *,
    concurrency: int,
    arrival_duration_us: int,
    request_deadline_us: int,
    scored_global_end_us: int,
    transport: BenchServingTransport,
    base_url: str,
    served_model: str,
    abort_grace_s: float,
    clock: ExecutionClock,
    on_terminal: Callable[[RequestExecution], Awaitable[None]],
) -> tuple[RequestExecution, ...]:
    """Issue one zero-think sequential request stream per client lane."""

    if not requests:
        return ()
    if len(requests) > MAX_IN_MEMORY_REQUEST_EXECUTIONS:
        raise ValueError("request corpus exceeds the bounded execution capacity")
    origin_ns = clock.monotonic_ns()
    semaphore = asyncio.Semaphore(concurrency)
    completed: dict[str, RequestExecution] = {}
    exhausted_lanes: set[int] = set()
    errors: list[Exception] = []
    stop_issuing = asyncio.Event()

    async def run_lane(lane: int) -> None:
        next_offer_us = 0
        for request in requests[lane::concurrency]:
            if stop_issuing.is_set() or next_offer_us >= arrival_duration_us:
                return
            try:
                execution = await _execute_request(
                    request,
                    deadline_us=min(
                        next_offer_us + request_deadline_us,
                        scored_global_end_us,
                    ),
                    origin_ns=origin_ns,
                    semaphore=semaphore,
                    transport=transport,
                    base_url=base_url,
                    served_model=served_model,
                    abort_grace_s=abort_grace_s,
                    clock=clock,
                    offered_at_us=next_offer_us,
                )
                completed[request.request_id] = execution
                await on_terminal(execution)
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(error)
                stop_issuing.set()
                return
            if execution.outcome.terminal_at_us is None:
                return
            next_offer_us = execution.outcome.terminal_at_us
        if not stop_issuing.is_set() and next_offer_us < arrival_duration_us:
            exhausted_lanes.add(lane)

    async with asyncio.TaskGroup() as group:
        for lane in range(concurrency):
            group.create_task(run_lane(lane))
    if errors:
        for sibling in errors[1:]:
            errors[0].add_note(f"sibling client lane also failed: {sibling}")
        raise errors[0]
    if exhausted_lanes:
        raise RuntimeError(
            "closed-loop request pool exhausted before the arrival window ended"
        )
    ordered = tuple(
        completed[request.request_id]
        for request in requests
        if request.request_id in completed
    )
    if not ordered:
        raise RuntimeError("closed-loop arrival window offered no requests")
    return ordered


def _timing_timestamps(
    execution: RequestExecution,
    *,
    score_started_ns: int,
) -> tuple[int, ...] | None:
    result = execution.result
    admitted = execution.outcome.admitted_at_us
    if result is None or admitted is None:
        return None
    values: list[int] = []
    for chunk in result.chunks:
        if chunk.per_token_observed_at_us is not None:
            relative = chunk.per_token_observed_at_us
        elif chunk.token_count == 1:
            relative = (chunk.chunk_observed_at_us,)
        else:
            return None
        values.extend(
            score_started_ns + (admitted + value) * 1000 for value in relative
        )
    return tuple(values) if len(values) == result.output_tokens else None


def _request_record(
    execution: RequestExecution,
    *,
    run_id: str,
    method: str,
    block: int,
    concurrency: int,
    score_started_ns: int,
) -> RequestRecord:
    timing = execution.timing()
    outcome = execution.outcome
    offered_at_us = (
        outcome.offered_at_us
        if outcome.offered_at_us is not None
        else execution.request.arrival_us
    )
    result = execution.result
    output_tokens = result.output_tokens if result is not None else 0
    token_ids = (
        result.generated_token_ids
        if result is not None
        else ()
        if execution.outcome.status in {"rejected", "cancelled", "timed_out"}
        else None
    )
    token_ids_body = (
        json.dumps(token_ids, separators=(",", ":")) if token_ids is not None else None
    )
    output_sha256 = (
        hashlib.sha256(token_ids_body.encode("utf-8")).hexdigest()
        if token_ids_body is not None
        else None
    )
    output_hash_format = OUTPUT_HASH_FORMAT if output_sha256 is not None else None
    timestamps = _timing_timestamps(execution, score_started_ns=score_started_ns)
    return RequestRecord(
        run_id=run_id,
        request_id=execution.request.request_id,
        prompt_id=f"{execution.request.namespace}:{execution.request.ordinal}",
        method=method,
        repetition_block=block,
        concurrency=concurrency,
        input_tokens=len(execution.request.input_token_ids),
        output_tokens=output_tokens,
        output_hash_format=output_hash_format,
        output_sha256=output_sha256,
        ttft_ms=(timing.ttft_us / 1000 if timing.ttft_us is not None else None),
        finished=outcome.status == "completed",
        stop_reason=(
            result.stop_reason
            if outcome.status == "completed" and result
            else outcome.code
        ),
        output_token_ids=token_ids_body,
        output_token_ids_sha256=(output_sha256 if token_ids_body is not None else None),
        outcome_status=outcome.status,
        arrival_ns=score_started_ns + offered_at_us * 1000,
        queue_enter_ns=score_started_ns + offered_at_us * 1000,
        admitted_ns=(
            score_started_ns + outcome.admitted_at_us * 1000
            if outcome.admitted_at_us is not None
            else None
        ),
        first_token_ns=(
            score_started_ns + offered_at_us * 1000 + timing.ttft_us * 1000
            if timing.ttft_us is not None
            else None
        ),
        completed_ns=(
            score_started_ns + outcome.terminal_at_us * 1000
            if outcome.terminal_at_us is not None
            else None
        ),
        token_timestamps_ns=(
            json.dumps(timestamps, separators=(",", ":")) if timestamps else None
        ),
        inter_token_ms=(
            json.dumps(
                [value / 1000 for value in timing.supported_itls_us],
                separators=(",", ":"),
            )
            if timing.full_itl_coverage and timing.supported_itls_us
            else None
        ),
        token_timing_coverage=timing.itl_coverage,
        coalesced_intervals=(
            timing.expected_itl_intervals - timing.supported_itl_intervals
        ),
        admission_code="admitted"
        if outcome.admitted_at_us is not None
        else outcome.code,
        cancellation_code=outcome.code if outcome.status == "cancelled" else None,
        error_code=(
            outcome.code
            if outcome.status in {"rejected", "timed_out", "unfinished"}
            else None
        ),
        cohort_sha256=execution.request.cohort_sha256,
        route_id=execution.request.route_id,
    )


def _percentile(values: Sequence[int], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _performance_record(
    *,
    run_id: str,
    plan: IndustrialExecutionPlan,
    requests: tuple[RequestExecution, ...],
    accounting: LoadAccounting,
    native: NativeEvidenceBatch,
    writer: EvidenceWriter,
    transport_metrics: Mapping[str, int] | None = None,
    asynchronous_backpressure_events: int = 0,
) -> PerformanceRecord:
    if accounting.elapsed_us <= 0:
        raise RuntimeError("scored load interval must have positive measured duration")
    completed = tuple(row for row in requests if row.outcome.status == "completed")
    output_tokens = sum(
        row.result.output_tokens for row in completed if row.result is not None
    )
    eligible_timings = tuple(
        row.timing()
        for row in completed
        if row.result is not None and row.result.output_tokens > 1
    )
    claimable = bool(eligible_timings) and all(
        timing.full_itl_coverage for timing in eligible_timings
    )
    itls = tuple(
        value for timing in eligible_timings for value in timing.supported_itls_us
    )
    if claimable and not itls:
        claimable = False
    p50 = _percentile(itls, 0.50) / 1000 if claimable else None
    p95 = _percentile(itls, 0.95) / 1000 if claimable else None
    p99 = _percentile(itls, 0.99) / 1000 if claimable else None
    identity = plan.runtime_plan.cell.identity
    allocation_free = identity.method in {"target_only", "static"}
    structurally_non_speculative = identity.method == "target_only"
    transport_metrics = {} if transport_metrics is None else transport_metrics
    base = PerformanceRecord(
        run_id=run_id,
        prompt_id=plan.load_plan.scored.hashes.corpus_sha256,
        method=identity.method,
        repetition_block=identity.block,
        region="scored_window",
        concurrency=int(identity.concurrency),
        generated_bucket_start=0,
        generated_bucket_end=max(
            request.requested_output_tokens for request in plan.scored_requests
        ),
        at_risk_requests=accounting.admitted,
        output_tokens=output_tokens,
        elapsed_s=accounting.elapsed_us / 1_000_000,
        decode_goodput_tps=output_tokens / (accounting.elapsed_us / 1_000_000),
        itl_p50_ms=p50,
        itl_p95_ms=p95,
        itl_p99_ms=p99,
        survival_weighted_accepted_prefix=None,
        accepted_drafts_per_verify=None,
        committed_tokens_per_verify=None,
        verified_drafts_per_verify=None,
        verification_waste=None,
        target_calls_per_output_token=None,
        batch_fill=None,
        queue_occupancy=None,
        gpu_busy=None,
        sm_utilization=None,
        dram_utilization=None,
        target_estimated_mfu=None,
        peak_hbm_bytes=None,
        kv_bytes=None,
        optimizer_bytes=0 if allocation_free else None,
        adaptation_memory_ledger=None,
        trainable_parameters=0 if allocation_free else None,
        training_cuda_ms=None,
        optimizer_cuda_ms=None,
        merge_cuda_ms=None,
        publish_cuda_ms=None,
        barrier_cuda_ms=None,
        exposed_update_ms=None,
        main_side_overlap_ratio=None,
        graph_replay_hit_rate=None,
        updates_launched=0 if allocation_free else None,
        updates_published=0 if allocation_free else None,
        exactness_violations=0 if structurally_non_speculative else None,
        version_mismatches=0 if structurally_non_speculative else None,
        fallbacks=0 if structurally_non_speculative else None,
        nonfinite_updates=0 if structurally_non_speculative else None,
        oom_events=0 if structurally_non_speculative else None,
        retractions=0 if structurally_non_speculative else None,
        communicator_failures=0 if structurally_non_speculative else None,
        admission_rejections=accounting.rejected,
        timeouts=accounting.timed_out,
        cancellations=accounting.cancelled,
        offered_requests=accounting.offered,
        admitted_requests=accounting.admitted,
        completed_requests=accounting.completed,
        unfinished_requests=accounting.unfinished,
        evidence_backpressure_events=(
            writer.backpressure_events + asynchronous_backpressure_events
        ),
        evidence_dropped_rows=writer.dropped_rows,
        writer_flush_count=int(writer.counters["flushes"]),
        writer_fsync_ms=writer.fsync_time_ns / 1_000_000,
        http_connections_created=transport_metrics.get("connections_created"),
        http_reused_requests=transport_metrics.get("reused_requests"),
    )
    return replace(base, **dict(native.performance_overrides))


def _workload_contract(method: str) -> str:
    if method == "target_only":
        return "industrial_target_only"
    if method == "static":
        return "industrial_static"
    return "industrial_adapted"


def _persisted_terminal_expectations(
    request_rows: Sequence[Mapping[str, object]],
    *,
    plan: IndustrialExecutionPlan,
) -> tuple[TerminalRequestExpectation, ...]:
    """Reconstruct client/server terminal identities from durable request rows."""

    by_id = {str(row.get("request_id")): row for row in request_rows}
    if len(by_id) != len(request_rows) or set(by_id) != {
        request.request_id for request in plan.scored_requests
    }:
        raise RuntimeError(
            "native terminal resume requires exact durable request coverage"
        )
    expectations: list[TerminalRequestExpectation] = []
    for request in plan.scored_requests:
        row = by_id[request.request_id]
        status = row.get("outcome_status")
        submitted = row.get("admitted_ns") is not None
        serialized = row.get("output_token_ids")
        try:
            output_values = (
                json.loads(serialized) if isinstance(serialized, str) else None
            )
        except json.JSONDecodeError as error:
            raise RuntimeError("durable request output IDs are malformed") from error
        if output_values is not None and (
            not isinstance(output_values, list)
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in output_values
            )
        ):
            raise RuntimeError("durable request output IDs are invalid")
        if submitted:
            if status == "completed":
                terminal_status = "completed"
                terminal_reason = "FINISH_LENGTH"
                if (
                    output_values is None
                    or len(output_values) != request.requested_output_tokens
                ):
                    raise RuntimeError(
                        "durable completion lacks its exact FINISH_LENGTH output"
                    )
            elif status in {"cancelled", "timed_out"}:
                terminal_status = "aborted"
                terminal_reason = "FINISH_ABORT"
                if output_values is None:
                    raise RuntimeError(
                        "durable aborted request lacks ordered output IDs"
                    )
            else:
                raise RuntimeError(
                    "durable submitted request has no native terminal status"
                )
            output_token_ids: tuple[int, ...] | None = tuple(output_values)
        else:
            if status not in {"rejected", "cancelled", "timed_out"}:
                raise RuntimeError(
                    "durable non-submitted request has no client terminal status"
                )
            terminal_status = str(status)
            terminal_reason_value = row.get("admission_code")
            if not isinstance(terminal_reason_value, str):
                raise RuntimeError(
                    "durable non-submitted request lacks its terminal reason"
                )
            terminal_reason = terminal_reason_value
            output_token_ids = None
        expectation = TerminalRequestExpectation(
            request_id=request.request_id,
            input_token_ids=request.input_token_ids,
            output_token_ids=output_token_ids,
            terminal_status=terminal_status,
            terminal_reason=terminal_reason,
            submitted_to_server=submitted,
        )
        expectation.validate()
        expectations.append(expectation)
    return tuple(expectations)


def _artifact_warmup_expectations(
    artifact: Mapping[str, object],
    *,
    plan: IndustrialExecutionPlan,
) -> tuple[TerminalRequestExpectation, ...]:
    """Recover warm-up rows while independently checking immutable inputs."""

    rows = artifact.get("warmup_requests")
    if not isinstance(rows, list) or len(rows) != len(plan.warmup_requests):
        raise RuntimeError("native terminal artifact changed warmup coverage")
    fields = {
        "request_id",
        "input_token_ids",
        "output_token_ids",
        "terminal_status",
        "terminal_reason",
        "submitted_to_server",
    }
    expectations: list[TerminalRequestExpectation] = []
    for raw, request in zip(rows, plan.warmup_requests, strict=True):
        if type(raw) is not dict or set(raw) != fields:
            raise RuntimeError("native terminal artifact warmup row is malformed")
        input_values = raw["input_token_ids"]
        output_values = raw["output_token_ids"]
        if (
            not isinstance(input_values, list)
            or not isinstance(output_values, list)
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in (*input_values, *output_values)
            )
        ):
            raise RuntimeError("native terminal warmup token identity is invalid")
        expectation = TerminalRequestExpectation(
            request_id=str(raw["request_id"]),
            input_token_ids=tuple(input_values),
            output_token_ids=tuple(output_values),
            terminal_status=str(raw["terminal_status"]),
            terminal_reason=str(raw["terminal_reason"]),
            submitted_to_server=raw["submitted_to_server"],
        )
        expectation.validate()
        if (
            expectation.request_id != request.request_id
            or expectation.input_token_ids != request.input_token_ids
            or expectation.terminal_status != "completed"
            or expectation.terminal_reason != "FINISH_LENGTH"
            or expectation.submitted_to_server is not True
            or expectation.output_token_ids is None
            or len(expectation.output_token_ids) != request.requested_output_tokens
        ):
            raise RuntimeError("native terminal artifact changed its warmup identity")
        expectations.append(expectation)
    return tuple(expectations)


def _validate_resumed_native_terminal(
    *,
    completed: Mapping[str, Path],
    run_row: Mapping[str, object],
    request_rows: Sequence[Mapping[str, object]],
    performance_row: Mapping[str, object],
    run_id: str,
    plan: IndustrialExecutionPlan,
    run_nonce_sha256: str,
    session_binding: SessionExecutionBinding | None,
) -> Path:
    """Re-read, reverify, and reconvert the durable signed terminal bundle."""

    name = run_row.get("native_terminal_artifact_path")
    size = run_row.get("native_terminal_artifact_size")
    raw_sha256 = run_row.get("native_terminal_raw_sha256")
    terminal_sha256 = run_row.get("native_terminal_sha256")
    policy_sha256 = run_row.get("trusted_attester_policy_sha256")
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
        or not isinstance(raw_sha256, str)
        or not _is_sha256(raw_sha256)
        or not isinstance(terminal_sha256, str)
        or not _is_sha256(terminal_sha256)
        or policy_sha256 != plan.trusted_attester_policy.sha256
    ):
        raise RuntimeError("completed run lacks its release terminal binding")
    artifact_path = completed["run"].parent / name
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise RuntimeError("completed native terminal artifact is not a regular file")
    try:
        body = artifact_path.read_bytes()
        artifact = json.loads(body.decode("utf-8"))
        canonical = (
            json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("completed native terminal artifact is invalid") from error
    if (
        type(artifact) is not dict
        or len(body) != size
        or hashlib.sha256(body).hexdigest() != raw_sha256
        or body != canonical
    ):
        raise RuntimeError("completed native terminal artifact changed on disk")
    scored_expectations = _persisted_terminal_expectations(
        request_rows,
        plan=plan,
    )
    warmup_expectations = _artifact_warmup_expectations(artifact, plan=plan)
    terminal = validate_native_terminal_artifact(
        artifact,
        trusted_attester_policy=plan.trusted_attester_policy,
        expected_warmup_requests=warmup_expectations,
        expected_scored_requests=scored_expectations,
    )
    if terminal.terminal_sha256 != terminal_sha256:
        raise RuntimeError("completed native terminal semantic digest changed")
    binding = terminal.binding
    if session_binding is None:
        expected_session_id = content_sha256(
            {
                "schema_version": 1,
                "kind": "standalone_native_terminal_session",
                "execution_plan_sha256": plan.sha256,
                "run_nonce_sha256": run_nonce_sha256,
            }
        )
        expected_session_epoch = 1
        expected_previous_run_id = None
    else:
        expected_session_id = session_binding.native_session_id
        expected_session_epoch = session_binding.native_trace_epoch
        expected_previous_run_id = session_binding.native_previous_run_id
    if (
        binding.run_id != run_id
        or binding.run_nonce_sha256 != run_nonce_sha256
        or binding.execution_plan_sha256 != plan.sha256
        or binding.rank_config_sha256 != plan.rank_config_sha256
        or binding.session_id != expected_session_id
        or binding.session_epoch != expected_session_epoch
        or binding.previous_run_id != expected_previous_run_id
        or binding.method != plan.runtime_plan.rank_configs[0].method
        or binding.warmup_request_ids
        != tuple(request.request_id for request in plan.warmup_requests)
        or binding.scored_request_ids
        != tuple(request.request_id for request in plan.scored_requests)
        or (session_binding is None and terminal.begin_receipt.reset_generation != 1)
        or (session_binding is None and terminal.reset_receipt.reset_generation != 2)
    ):
        raise RuntimeError("completed native terminal artifact changed run identity")
    if binding.method != "target_only" and not terminal.trusted_attestation:
        raise RuntimeError(TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON)
    native = terminal.to_native_evidence_batch()
    round_rows = (
        pq.read_table(completed["round"]).to_pylist() if "round" in completed else []
    )
    update_rows = (
        pq.read_table(completed["update"]).to_pylist() if "update" in completed else []
    )
    if (
        [asdict(row) for row in native.rounds] != round_rows
        or [asdict(row) for row in native.updates] != update_rows
        or any(
            performance_row.get(key) != value
            for key, value in native.performance_overrides
        )
    ):
        raise RuntimeError(
            "completed native terminal artifact no longer converts to durable rows"
        )
    return artifact_path


def _validate_resume(
    *,
    completed: dict[str, Path],
    run_id: str,
    plan: IndustrialExecutionPlan,
    run_nonce_sha256: str,
    session_binding: SessionExecutionBinding | None = None,
) -> Path:
    root = completed["run"].parent
    terminal_receipt = root / f"{run_id}.rank0.complete.json"
    if not terminal_receipt.exists():
        terminal_receipt = root / f"{run_id}.rank0.prepared.json"
    registered_writer_policy = evidence_writer_policy_from_receipt(terminal_receipt)
    if registered_writer_policy != plan.evidence_writer_policy:
        raise RuntimeError(
            "completed evidence differs from the registered writer policy"
        )
    run_rows = pq.read_table(completed["run"]).to_pylist()
    request_rows = pq.read_table(completed["request"]).to_pylist()
    performance_rows = pq.read_table(completed["performance"]).to_pylist()
    config = plan.runtime_plan.rank_configs[0]
    expected = {
        "run_id": run_id,
        "manifest_sha256": plan.runtime_plan.registry_sha256,
        "config_sha256": plan.runtime_plan.cell_id,
        "rank_config_sha256": plan.rank_config_sha256,
        "method": config.method,
        "industrial_cell_id": plan.runtime_plan.cell_id,
        "runtime_sha256": plan.sha256,
        "split_sha256": plan.split_artifact.content_sha256,
        "corpus_sha256": plan.load_plan.scored.hashes.corpus_sha256,
        "arrival_trace_sha256": plan.load_plan.scored.hashes.arrivals_sha256,
        "request_ids_sha256": plan.load_plan.scored.hashes.request_ids_sha256,
        "sampling_profile_sha256": plan.sampling_artifact.content_sha256,
        "model_lock_sha256": plan.model_lock_artifact.content_sha256,
        "patched_sglang_tree": plan.patched_sglang_tree,
        "run_nonce_sha256": run_nonce_sha256,
        "topology_sha256": plan.topology_sha256,
        "tensor_parallel_size": config.runtime.tensor_parallel_size,
        "data_parallel_size": config.runtime.data_parallel_size,
        "world_size": 1,
        "rank": 0,
        "expected_request_rows": len(request_rows),
        "expected_performance_rows": 1,
        "workload_contract": _workload_contract(config.method),
        "experiment_budget_sha256": plan.budget.sha256,
        "status": "complete",
        "session_plan_sha256": (
            None if session_binding is None else session_binding.session_plan_sha256
        ),
        "session_open_receipt_sha256": (
            None
            if session_binding is None
            else session_binding.session_open_receipt_sha256
        ),
        "reset_receipt_sha256": (
            None if session_binding is None else session_binding.reset_receipt_sha256
        ),
        "session_epoch": (
            None if session_binding is None else session_binding.session_epoch
        ),
    }
    if len(run_rows) != 1 or any(
        run_rows[0].get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("completed industrial receipt has a mismatched run identity")
    for table in ("request", "round", "update", "performance"):
        actual_rows = (
            pq.ParquetFile(completed[table]).metadata.num_rows
            if table in completed
            else 0
        )
        if run_rows[0].get(f"expected_{table}_rows") != actual_rows:
            raise RuntimeError(
                "completed industrial receipt has mismatched table coverage"
            )
    expected_pool = {request.request_id: request for request in plan.scored_requests}
    actual_ids = {str(row.get("request_id")) for row in request_rows}
    if plan.load_plan.scored.source_kind == "closed_loop":
        concurrency = config.runtime.max_running_requests
        if not actual_ids or actual_ids - set(expected_pool):
            raise RuntimeError(
                "completed closed-loop receipt has foreign request identities"
            )
        for lane in range(concurrency):
            seen_gap = False
            for request in plan.scored_requests[lane::concurrency]:
                present = request.request_id in actual_ids
                if seen_gap and present:
                    raise RuntimeError(
                        "completed closed-loop receipt is not a per-client prefix"
                    )
                seen_gap = seen_gap or not present
    elif actual_ids != set(expected_pool):
        raise RuntimeError(
            "completed industrial receipt has mismatched request coverage"
        )
    expected_requests = {
        request_id: expected_pool[request_id] for request_id in actual_ids
    }
    expected_ids = set(expected_requests)
    if (
        len(request_rows) != len(expected_ids)
        or actual_ids != expected_ids
        or any(row.get("method") != config.method for row in request_rows)
    ):
        raise RuntimeError(
            "completed industrial receipt has mismatched request coverage"
        )
    statuses = {
        "rejected",
        "completed",
        "timed_out",
        "cancelled",
        "unfinished",
    }
    status_counts = {status: 0 for status in statuses}
    admitted = 0
    completed_output_tokens = 0
    for row in request_rows:
        request = expected_requests[str(row["request_id"])]
        status = row.get("outcome_status")
        if status not in statuses or row.get("finished") is not (status == "completed"):
            raise RuntimeError(
                "completed industrial receipt has an invalid request outcome"
            )
        status_counts[str(status)] += 1
        if row.get("admitted_ns") is not None:
            admitted += 1
        output_tokens = row.get("output_tokens")
        expected_identity = {
            "prompt_id": f"{request.namespace}:{request.ordinal}",
            "repetition_block": plan.runtime_plan.cell.identity.block,
            "concurrency": config.runtime.max_running_requests,
            "input_tokens": len(request.input_token_ids),
            "cohort_sha256": request.cohort_sha256,
            "route_id": request.route_id,
        }
        if (
            any(row.get(key) != value for key, value in expected_identity.items())
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or not 0 <= output_tokens <= request.requested_output_tokens
            or (status == "completed" and output_tokens < 1)
            or (status == "rejected" and row.get("admitted_ns") is not None)
        ):
            raise RuntimeError(
                "completed industrial receipt has a mismatched request binding"
            )
        if status == "completed":
            completed_output_tokens += output_tokens
    if len(performance_rows) != 1:
        raise RuntimeError(
            "completed industrial receipt lacks exact performance coverage"
        )
    performance = performance_rows[0]
    performance_expected = {
        "run_id": run_id,
        "prompt_id": plan.load_plan.scored.hashes.corpus_sha256,
        "method": config.method,
        "repetition_block": plan.runtime_plan.cell.identity.block,
        "region": "scored_window",
        "concurrency": config.runtime.max_running_requests,
        "at_risk_requests": admitted,
        "output_tokens": completed_output_tokens,
        "offered_requests": len(expected_ids),
        "admitted_requests": admitted,
        "completed_requests": status_counts["completed"],
        "unfinished_requests": status_counts["unfinished"],
        "admission_rejections": status_counts["rejected"],
        "timeouts": status_counts["timed_out"],
        "cancellations": status_counts["cancelled"],
    }
    if any(
        performance.get(key) != value for key, value in performance_expected.items()
    ):
        raise RuntimeError(
            "completed industrial receipt has mismatched load accounting"
        )
    return _validate_resumed_native_terminal(
        completed=completed,
        run_row=run_rows[0],
        request_rows=request_rows,
        performance_row=performance,
        run_id=run_id,
        plan=plan,
        run_nonce_sha256=run_nonce_sha256,
        session_binding=session_binding,
    )


def revalidate_industrial_execution_result(
    *,
    plan: IndustrialExecutionPlan,
    result: IndustrialExecutionResult,
    run_nonce_sha256: str,
) -> IndustrialExecutionTerminalBinding:
    """Reopen a serving result and return its exact durable terminal binding.

    This is the first-party assignment boundary used by the pool executor.  It
    trusts neither ``IndustrialExecutionResult`` nor copied digests: the plan,
    final/prepared evidence chain, budget observation sidecar, Parquet rows,
    and native terminal artifact are all replayed from disk first.
    """

    if type(plan) is not IndustrialExecutionPlan:
        raise TypeError("terminal revalidation requires an exact execution plan")
    if type(result) is not IndustrialExecutionResult:
        raise TypeError("terminal revalidation requires an exact execution result")
    if not _is_sha256(run_nonce_sha256):
        raise ValueError("terminal revalidation requires a lowercase run nonce")
    plan.validate()
    physical = plan.runtime_plan.physical_assignment
    if physical is None:
        raise ValueError("terminal revalidation lacks a physical assignment")
    if plan.runtime_plan.cell.resources.workload_class in {
        WorkloadClass.COMPILE,
        WorkloadClass.DOWNLOAD,
    }:
        raise ValueError("non-serving execution has no terminal-result contract")
    run_id = industrial_run_id(plan, run_nonce_sha256)
    if result.run_id != run_id:
        raise ValueError("execution result names another run")
    root = Path(plan.runtime_plan.cell.resources.evidence_root).resolve()
    terminal_receipt = (root / f"{run_id}.rank0.complete.json").resolve()
    if (
        Path(result.terminal_receipt) != terminal_receipt
        or terminal_receipt.is_symlink()
        or not terminal_receipt.is_file()
        or _file_sha256(terminal_receipt) != result.terminal_receipt_sha256
    ):
        raise RuntimeError("execution result terminal receipt changed or is foreign")
    completed = load_completed_evidence(root, run_id=run_id, rank=0)
    if completed is None:
        raise RuntimeError("execution result lacks durable completed evidence")
    native_path = _validate_resume(
        completed=completed,
        run_id=run_id,
        plan=plan,
        run_nonce_sha256=run_nonce_sha256,
    )
    observation, observation_path, observation_sidecar = _load_budget_observation(
        root=root,
        run_id=run_id,
        plan=plan,
        terminal_receipt=terminal_receipt,
    )
    if (
        result.execution_plan_sha256 != plan.sha256
        or result.experiment_budget_sha256 != plan.budget.sha256
        or result.rank_config_sha256 != plan.rank_config_sha256
        or result.topology_sha256 != plan.topology_sha256
        or result.budget_observation != str(observation_path)
        or result.budget_observation_sidecar != str(observation_sidecar)
        or result.budget_observation_sha256 != observation.sha256
    ):
        raise ValueError("execution result summary differs from revalidated raw files")
    expected_evidence = tuple(
        sorted(
            {
                *(str(path.resolve()) for path in completed.values()),
                str(native_path.resolve()),
                str(observation_path.resolve()),
                str(observation_sidecar.resolve()),
            }
        )
    )
    if tuple(sorted(result.evidence_files)) != expected_evidence:
        raise ValueError("execution result evidence-file coverage is not exact")
    evidence_paths = tuple(Path(value) for value in expected_evidence)
    if any(
        path.is_symlink() or not path.is_file() or path.resolve() != path
        for path in evidence_paths
    ):
        raise RuntimeError("execution result evidence contains a changed file")
    run_rows = pq.read_table(completed["run"]).to_pylist()
    if len(run_rows) != 1:
        raise RuntimeError("execution result run evidence is no longer singular")
    run_row = run_rows[0]
    native_raw_sha256 = run_row.get("native_terminal_raw_sha256")
    native_terminal_sha256 = run_row.get("native_terminal_sha256")
    if (
        not isinstance(native_raw_sha256, str)
        or not _is_sha256(native_raw_sha256)
        or not isinstance(native_terminal_sha256, str)
        or not _is_sha256(native_terminal_sha256)
        or _file_sha256(native_path) != native_raw_sha256
    ):
        raise RuntimeError("execution result native terminal binding changed")
    try:
        native_artifact = json.loads(native_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "execution result native terminal artifact is invalid"
        ) from error
    validated_native = validate_native_terminal_artifact(
        native_artifact,
        trusted_attester_policy=plan.trusted_attester_policy,
    )
    return IndustrialExecutionTerminalBinding(
        cell_id=plan.runtime_plan.cell_id,
        run_id=run_id,
        run_nonce_sha256=run_nonce_sha256,
        execution_plan_sha256=plan.sha256,
        dispatch_plan_sha256=plan.dispatch_plan.sha256,
        assignment_sha256=physical.assignment_sha256,
        experiment_budget_sha256=plan.budget.sha256,
        inventory_sha256=plan.dispatch_context.inventory.sha256,
        physical_gpu_uuids=physical.gpu_uuids,
        terminal_receipt_path=str(terminal_receipt),
        terminal_receipt_sha256=result.terminal_receipt_sha256,
        budget_observation_path=str(observation_path),
        budget_observation_sha256=observation.sha256,
        budget_observation_sidecar_path=str(observation_sidecar),
        budget_observation_sidecar_sha256=_file_sha256(observation_sidecar),
        native_terminal_artifact_path=str(native_path),
        native_terminal_raw_sha256=native_raw_sha256,
        native_terminal_sha256=native_terminal_sha256,
        trusted_attester_policy_sha256=plan.trusted_attester_policy.sha256,
        trusted_attestation=validated_native.trusted_attestation,
        evidence_file_paths=expected_evidence,
        evidence_file_sha256s=tuple(_file_sha256(path) for path in evidence_paths),
    )


async def execute_industrial_plan(
    plan: IndustrialExecutionPlan,
    *,
    output_root: str | Path,
    run_nonce_sha256: str,
    launch_server: ServerLauncher | None,
    transport: BenchServingTransport,
    native_evidence: NativeTerminalProvider | None = None,
    clock: ExecutionClock | None = None,
    existing_handle: ServerHandle | None = None,
    transport_already_open: bool = False,
    keep_session_open: bool = False,
    session_lifecycle: SessionExecutionLifecycle | None = None,
) -> IndustrialExecutionResult:
    """Execute one plan; no process or network action occurs without injection."""

    if (
        existing_handle is not None
        or transport_already_open
        or keep_session_open
        or session_lifecycle is not None
    ):
        raise SharedSessionUnavailableError(SHARED_SESSION_UNAVAILABLE_REASON)
    plan.validate()
    _require_execution_trainable_plan_authority(plan)
    observed_component_ms = _initial_budget_observations(plan)
    if launch_server is None:
        raise ValueError("standalone execution requires a server launcher")
    method = plan.runtime_plan.rank_configs[0].method
    evidence_preflight = native_evidence_preflight(plan, native_evidence)
    if evidence_preflight.status == "BLOCKED":
        raise NativeEvidenceUnavailableError(evidence_preflight)
    if native_evidence is None:  # pragma: no cover - preflight invariant
        raise RuntimeError("native terminal preflight lost its exact provider")
    session_binding: SessionExecutionBinding | None = None
    session_startup_interval_ns: tuple[int, int] | None = None
    session_prepare_observed_ms = 0
    if clock is None:
        clock = ExecutionClock()
    run_id = industrial_run_id(plan, run_nonce_sha256)
    root = Path(output_root).resolve()
    registered_root = Path(plan.runtime_plan.cell.resources.evidence_root).resolve()
    if root != registered_root:
        raise ValueError("output_root differs from the registry evidence reservation")
    root.mkdir(parents=True, exist_ok=True)
    completed = load_completed_evidence(root, run_id=run_id, rank=0)
    terminal_receipt = root / f"{run_id}.rank0.complete.json"
    if completed is None:
        completed = _recover_prepared_completion(
            root=root,
            run_id=run_id,
            run_nonce_sha256=run_nonce_sha256,
            plan=plan,
        )
    if completed is not None:
        if session_lifecycle is not None:
            raise RuntimeError(
                "shared-session execution cannot resume preexisting trace evidence"
            )
        native_terminal_artifact_path = _validate_resume(
            completed=completed,
            run_id=run_id,
            plan=plan,
            run_nonce_sha256=run_nonce_sha256,
            session_binding=session_binding,
        )
        observation, observation_path, observation_sidecar = _load_budget_observation(
            root=root,
            run_id=run_id,
            plan=plan,
            terminal_receipt=terminal_receipt,
        )
        return IndustrialExecutionResult(
            run_id=run_id,
            execution_plan_sha256=plan.sha256,
            experiment_budget_sha256=plan.budget.sha256,
            rank_config_sha256=plan.rank_config_sha256,
            topology_sha256=plan.topology_sha256,
            resumed=True,
            terminal_receipt=str(terminal_receipt),
            terminal_receipt_sha256=_file_sha256(terminal_receipt),
            budget_observation=str(observation_path),
            budget_observation_sidecar=str(observation_sidecar),
            budget_observation_sha256=observation.sha256,
            evidence_files=tuple(
                str(path)
                for path in (
                    *completed.values(),
                    native_terminal_artifact_path,
                    observation_path,
                    observation_sidecar,
                )
            ),
            accounting=None,
        )

    if session_lifecycle is not None:
        session_startup_interval_ns = session_lifecycle.claim_startup_interval_ns(
            execution_plan_sha256=plan.sha256,
        )
        session_prepare_started_ns = time.perf_counter_ns()
        session_binding = await session_lifecycle.prepare_trace(
            execution_plan_sha256=plan.sha256,
        )
        session_prepare_completed_ns = time.perf_counter_ns()
        session_binding.validate()
        if session_binding.execution_plan_sha256 != plan.sha256:
            raise ValueError("session lifecycle prepared another execution plan")
        session_prepare_observed_ms = _elapsed_milliseconds(
            session_prepare_started_ns,
            session_prepare_completed_ns,
        )

    writer_policy = plan.evidence_writer_policy
    writer = EvidenceWriter(
        root,
        run_id=run_id,
        rank=0,
        max_queued_rows=writer_policy.writer_queue_rows,
        row_group_rows=writer_policy.parquet_row_group_rows,
        checkpoint_interval_s=writer_policy.checkpoint_interval_ms / 1000,
        overflow_policy=writer_policy.overflow_policy,
        registered_policy=writer_policy,
    )
    sink = _AsyncEvidenceSink(
        writer,
        max_queued_rows=writer_policy.async_queue_rows,
        max_batch_rows=writer_policy.async_batch_rows,
    )
    handle: ServerHandle | None = existing_handle
    transport_open = transport_already_open
    owns_handle = existing_handle is None
    native_binding: NativeTerminalRunBinding | None = None
    try:
        startup_started_ns = time.perf_counter_ns()
        if handle is None:
            if launch_server is None:
                raise RuntimeError("validated standalone execution lost its launcher")
            handle = await launch_server(plan.server_launch)
            await handle.wait_ready(plan.startup_timeout_s)
        if not transport_open:
            await transport.open(
                request_timeout_s=(
                    plan.load_plan.window.request_deadline_us / 1_000_000
                ),
                abort_timeout_s=plan.abort_grace_s,
            )
            transport_open = True
        pinned_transport = _bind_native_terminal_transport(
            provider=native_evidence,
            transport=transport,
            base_url=plan.server_launch.base_url,
        )
        config = plan.runtime_plan.rank_configs[0]
        live_execution_policy_sha256 = await _require_live_controlled_execution_policy(
            transport=pinned_transport,
            config=config,
        )
        capability = await native_evidence.capability(expected_method=method)
        if method != "target_only" and not capability.trusted_attester_configured:
            raise RuntimeError(TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON)
        native_binding = _native_terminal_binding(
            plan=plan,
            run_id=run_id,
            run_nonce_sha256=run_nonce_sha256,
            session_binding=session_binding,
        )
        await native_evidence.begin(native_binding)
        startup_completed_ns = time.perf_counter_ns()
        observed_session_startup_ms = (
            0
            if session_startup_interval_ns is None
            else _elapsed_milliseconds(*session_startup_interval_ns)
        )
        observed_component_ms["startup_model_load"] = (
            _elapsed_milliseconds(startup_started_ns, startup_completed_ns)
            + observed_session_startup_ms
        )
        concurrency = config.runtime.max_running_requests
        warmup_started_ns = startup_completed_ns
        warmup: tuple[RequestExecution, ...] = ()
        if plan.warmup_requests:
            warmup_end = plan.load_plan.window.warmup_duration_us
            warmup = await _execute_corpus(
                plan.warmup_requests,
                deadline_for=lambda request: min(
                    request.arrival_us + plan.load_plan.window.request_deadline_us,
                    warmup_end,
                ),
                concurrency=concurrency,
                transport=transport,
                base_url=plan.server_launch.base_url,
                served_model=config.model.target,
                abort_grace_s=plan.abort_grace_s,
                clock=clock,
            )
            if any(row.outcome.status != "completed" for row in warmup):
                raise RuntimeError("excluded warm-up did not complete exactly")
        warmup_requests_completed_ns = time.perf_counter_ns()
        warmup_terminal_requests = tuple(
            _terminal_request_expectation(row) for row in warmup
        )
        await native_evidence.reset(warmup_requests=warmup_terminal_requests)
        native_reset_completed_ns = time.perf_counter_ns()
        observed_component_ms["excluded_warmup"] = (
            _elapsed_milliseconds(warmup_started_ns, warmup_requests_completed_ns)
            if plan.warmup_requests
            else 0
        )
        pre_score_native_reset_ms = _elapsed_milliseconds(
            warmup_requests_completed_ns,
            native_reset_completed_ns,
        )
        score_started_observation_ns = native_reset_completed_ns
        score_schedule_origin_ns = clock.monotonic_ns()
        score_started_ns = clock.wall_ns()

        async def record_terminal(execution: RequestExecution) -> None:
            await sink.write(
                _request_record(
                    execution,
                    run_id=run_id,
                    method=config.method,
                    block=plan.runtime_plan.cell.identity.block,
                    concurrency=concurrency,
                    score_started_ns=score_started_ns,
                )
            )

        async def observe_arrival_boundary() -> int:
            await _sleep_until(
                plan.load_plan.window.arrival_duration_us,
                origin_ns=score_schedule_origin_ns,
                clock=clock,
            )
            return time.perf_counter_ns()

        arrival_boundary_task = asyncio.create_task(observe_arrival_boundary())
        try:
            if plan.load_plan.scored.source_kind == "closed_loop":
                requests = await _execute_closed_loop_corpus(
                    plan.scored_requests,
                    concurrency=concurrency,
                    arrival_duration_us=plan.load_plan.window.arrival_duration_us,
                    request_deadline_us=plan.load_plan.window.request_deadline_us,
                    scored_global_end_us=plan.load_plan.window.scored_global_end_us,
                    transport=transport,
                    base_url=plan.server_launch.base_url,
                    served_model=config.model.target,
                    abort_grace_s=plan.abort_grace_s,
                    clock=clock,
                    on_terminal=record_terminal,
                )
            else:
                requests = await _execute_corpus(
                    plan.scored_requests,
                    deadline_for=lambda request: (
                        plan.load_plan.window.request_timeout_us(
                            next(
                                value
                                for value in plan.load_plan.scored.requests
                                if value.request_id == request.request_id
                            )
                        )
                    ),
                    concurrency=concurrency,
                    transport=transport,
                    base_url=plan.server_launch.base_url,
                    served_model=config.model.target,
                    abort_grace_s=plan.abort_grace_s,
                    clock=clock,
                    on_terminal=record_terminal,
                )
            arrival_boundary_ns = await arrival_boundary_task
        except BaseException:
            arrival_boundary_task.cancel()
            await asyncio.gather(arrival_boundary_task, return_exceptions=True)
            raise
        score_completed_monotonic_ns = time.perf_counter_ns()
        scoring_component = _SCORING_COMPONENT_BY_JOB[plan.budget.job_kind]
        observed_component_ms[scoring_component] = _elapsed_milliseconds(
            score_started_observation_ns, arrival_boundary_ns
        )
        observed_component_ms["drain"] = _elapsed_milliseconds(
            arrival_boundary_ns, score_completed_monotonic_ns
        )
        finalization_started_ns = score_completed_monotonic_ns
        accounting = account_scored_requests(
            plan.load_plan,
            tuple(row.outcome for row in requests),
        )
        if native_binding is None:
            raise RuntimeError("native terminal binding was not established")
        terminal_requests = tuple(
            _terminal_request_expectation(row) for row in requests
        )
        terminal = await native_evidence.finalize(requests=terminal_requests)
        if (
            await _require_live_controlled_execution_policy(
                transport=pinned_transport,
                config=config,
            )
            != live_execution_policy_sha256
        ):
            raise RuntimeError("live execution-policy identity changed during scoring")
        if not isinstance(terminal, ValidatedNativeTerminalEvidence):
            raise TypeError("native provider returned an untyped terminal envelope")
        if terminal.binding != native_binding:
            raise RuntimeError("native terminal envelope changed its run binding")
        if (
            terminal.trusted_attester_policy_sha256
            != plan.trusted_attester_policy.sha256
        ):
            raise RuntimeError("native terminal envelope changed its release policy")
        if config.method != "target_only" and not terminal.trusted_attestation:
            raise RuntimeError(TRUSTED_NATIVE_ATTESTER_UNAVAILABLE_REASON)
        terminal_artifact = terminal.to_artifact(
            warmup_requests=warmup_terminal_requests
        )
        revalidated_terminal = validate_native_terminal_artifact(
            terminal_artifact,
            trusted_attester_policy=plan.trusted_attester_policy,
            expected_binding=native_binding,
            expected_warmup_requests=warmup_terminal_requests,
            expected_scored_requests=terminal_requests,
        )
        if (
            revalidated_terminal.terminal_sha256 != terminal.terminal_sha256
            or revalidated_terminal.raw_json != terminal.raw_json
        ):
            raise RuntimeError("native terminal envelope changed during revalidation")
        terminal = revalidated_terminal
        native = terminal.to_native_evidence_batch()
        native.validate(run_id=run_id, method=config.method)
        completed_request_ids = {
            row.request.request_id
            for row in requests
            if row.outcome.status == "completed"
        }
        if native.rounds and not completed_request_ids <= {
            row.request_id for row in native.rounds
        }:
            raise RuntimeError("native round evidence misses a completed request")
        finalization_completed_ns = time.perf_counter_ns()
        observed_component_ms["reset_finalization"] = (
            _elapsed_milliseconds(finalization_started_ns, finalization_completed_ns)
            + pre_score_native_reset_ms
            + session_prepare_observed_ms
        )
        evidence_started_ns = finalization_completed_ns
        await sink.flush()
        native_terminal_artifact = await asyncio.to_thread(
            writer.persist_native_terminal_artifact,
            terminal_artifact,
        )
        completed_ns = clock.wall_ns()
        for row in native.rounds:
            await sink.write(row)
        for row in native.updates:
            await sink.write(row)
        await sink.write(
            _performance_record(
                run_id=run_id,
                plan=plan,
                requests=requests,
                accounting=accounting,
                native=native,
                writer=writer,
                transport_metrics=transport.metrics(),
                asynchronous_backpressure_events=sink.backpressure_events,
            )
        )
        await sink.write(
            RunRecord(
                run_id=run_id,
                manifest_sha256=plan.runtime_plan.registry_sha256,
                config_sha256=plan.runtime_plan.cell_id,
                rank_config_sha256=plan.rank_config_sha256,
                method=config.method,
                model_pair=config.model.key,
                repetition_block=plan.runtime_plan.cell.identity.block,
                started_ns=score_started_ns,
                completed_ns=completed_ns,
                status="aborted" if accounting.unfinished else "complete",
                industrial_cell_id=plan.runtime_plan.cell_id,
                runtime_sha256=plan.sha256,
                split_sha256=plan.split_artifact.content_sha256,
                corpus_sha256=plan.load_plan.scored.hashes.corpus_sha256,
                arrival_trace_sha256=plan.load_plan.scored.hashes.arrivals_sha256,
                request_ids_sha256=plan.load_plan.scored.hashes.request_ids_sha256,
                sampling_profile_sha256=plan.sampling_artifact.content_sha256,
                model_lock_sha256=plan.model_lock_artifact.content_sha256,
                patched_sglang_tree=plan.patched_sglang_tree,
                run_nonce_sha256=run_nonce_sha256,
                topology_sha256=plan.topology_sha256,
                tensor_parallel_size=config.runtime.tensor_parallel_size,
                data_parallel_size=config.runtime.data_parallel_size,
                world_size=1,
                rank=0,
                expected_request_rows=len(requests),
                expected_round_rows=len(native.rounds),
                expected_update_rows=len(native.updates),
                expected_performance_rows=1,
                workload_contract=_workload_contract(config.method),
                experiment_budget_sha256=plan.budget.sha256,
                preflight_attestation_sha256=None,
                session_plan_sha256=(
                    None
                    if session_binding is None
                    else session_binding.session_plan_sha256
                ),
                session_open_receipt_sha256=(
                    None
                    if session_binding is None
                    else session_binding.session_open_receipt_sha256
                ),
                reset_receipt_sha256=(
                    None
                    if session_binding is None
                    else session_binding.reset_receipt_sha256
                ),
                session_epoch=(
                    None if session_binding is None else session_binding.session_epoch
                ),
                native_terminal_artifact_path=str(native_terminal_artifact["path"]),
                native_terminal_artifact_size=int(native_terminal_artifact["size"]),
                native_terminal_raw_sha256=str(native_terminal_artifact["raw_sha256"]),
                native_terminal_sha256=str(native_terminal_artifact["terminal_sha256"]),
                trusted_attester_policy_sha256=str(
                    native_terminal_artifact["trusted_attester_policy_sha256"]
                ),
            )
        )
        if not keep_session_open:
            await transport.close()
            transport_open = False
            if owns_handle:
                await handle.terminate(plan.shutdown_timeout_s)
                handle = None
        await sink.close()
        if accounting.unfinished:
            writer.abort(reason="scored requests contain unfinished outcomes")
            raise RuntimeError(
                "scored requests contain unfinished outcomes; evidence is nonclaimable"
            )
        written, prepared_receipt = writer.prepare_close()
        prepared_receipt_sha256 = _file_sha256(prepared_receipt)
        if session_lifecycle is not None:
            await session_lifecycle.complete_trace(
                execution_plan_sha256=plan.sha256,
                terminal_receipt_sha256=prepared_receipt_sha256,
                run_id=run_id,
            )
        evidence_completed_ns = time.perf_counter_ns()
        observed_component_ms["evidence_flush_shutdown"] = _elapsed_milliseconds(
            evidence_started_ns, evidence_completed_ns
        )
        if set(observed_component_ms) != set(_BUDGET_OBSERVATION_COMPONENTS):
            missing = sorted(
                set(_BUDGET_OBSERVATION_COMPONENTS) - set(observed_component_ms)
            )
            raise RuntimeError(
                "cannot publish a partial BudgetObservationReceipt; "
                f"unobserved components={missing}"
            )
        observed_rows = tuple(
            (name, observed_component_ms[name])
            for name in _BUDGET_OBSERVATION_COMPONENTS
        )
        observed_wall_ms = sum(value for _, value in observed_rows)
        measured_reserved_gang_ms = observed_wall_ms * plan.budget.gpu_count
        fixed_instance_billed_gpu_ms = (
            observed_wall_ms * plan.runtime_plan.physical_fixed_instance_gpu_count
        )
        observation = BudgetObservationReceipt(
            schema_version=1,
            budget=plan.budget,
            observed_component_ms=observed_rows,
            measured_gpu_ms=measured_reserved_gang_ms,
            fixed_instance_billed_gpu_ms=fixed_instance_billed_gpu_ms,
            terminal_evidence_sha256=prepared_receipt_sha256,
        )
        observation_path, observation_sidecar = _publish_budget_observation(
            root=root,
            run_id=run_id,
            observation=observation,
        )

        def validate_post_binding() -> None:
            persisted, persisted_path, persisted_sidecar = _load_budget_observation(
                root=root,
                run_id=run_id,
                plan=plan,
                terminal_receipt=prepared_receipt,
            )
            if (
                persisted != observation
                or persisted_path != observation_path
                or persisted_sidecar != observation_sidecar
            ):
                raise RuntimeError(
                    "budget observation changed before terminal publication"
                )

        writer.publish_close(validate_post_binding=validate_post_binding)
        terminal_receipt_sha256 = _file_sha256(terminal_receipt)
        terminal_value = json.loads(terminal_receipt.read_text(encoding="utf-8"))
        if (
            type(terminal_value) is not dict
            or terminal_value.get("prepared_receipt_sha256") != prepared_receipt_sha256
        ):
            raise RuntimeError(
                "published terminal receipt does not bind prepared evidence"
            )
        return IndustrialExecutionResult(
            run_id=run_id,
            execution_plan_sha256=plan.sha256,
            experiment_budget_sha256=plan.budget.sha256,
            rank_config_sha256=plan.rank_config_sha256,
            topology_sha256=plan.topology_sha256,
            resumed=False,
            terminal_receipt=str(terminal_receipt),
            terminal_receipt_sha256=terminal_receipt_sha256,
            budget_observation=str(observation_path),
            budget_observation_sidecar=str(observation_sidecar),
            budget_observation_sha256=observation.sha256,
            evidence_files=tuple(
                str(path)
                for path in (
                    *written.values(),
                    root / str(native_terminal_artifact["path"]),
                    observation_path,
                    observation_sidecar,
                )
            ),
            accounting=accounting,
        )
    except BaseException as error:
        if transport_open and not keep_session_open:
            try:
                await transport.close()
            except (OSError, RuntimeError, TimeoutError):
                pass
        if handle is not None and owns_handle and not keep_session_open:
            try:
                await handle.terminate(plan.shutdown_timeout_s)
            except (OSError, RuntimeError, TimeoutError) as shutdown_error:
                error.add_note(f"server shutdown also failed: {shutdown_error}")
        try:
            await sink.close()
        except (OSError, RuntimeError, ValueError) as sink_error:
            error.add_note(f"background evidence shutdown also failed: {sink_error}")
        if sink.backpressure_events:
            try:
                writer.register_external_backpressure_events(sink.backpressure_events)
            except (OSError, RuntimeError, ValueError):
                pass
        try:
            writer.abort(reason=f"{type(error).__name__}: {error}")
        except (OSError, RuntimeError):
            pass
        raise
