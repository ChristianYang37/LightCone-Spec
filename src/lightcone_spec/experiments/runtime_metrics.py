"""Strict, receipt-derived runtime metric observations.

This module is intentionally downstream of execution.  It never launches a
server, samples a device, or upgrades a locally valid receipt into formal GPU
execution authority.  Instead, it reopens first-party raw artifacts and emits
one of four explicit states:

``OBSERVED``
    A counter is present in a validated first-party receipt.
``MEASURED``
    A duration was measured by the clock named by that receipt's protocol.
``UNRESOLVED``
    The metric is applicable, but this release has no bound raw source for it.
``N/A``
    The metric does not apply to the execution mode being reduced.

In particular, absent NVML, power, energy, and profiler evidence is never
represented by a numeric zero.  Native-terminal observations also retain the
release-attestation bit; consuming this reduction does not bypass the release
signer or completion-authority gates used by formal experiment reducers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq

from lightcone_spec.experiments.planning import BudgetObservationReceipt
from lightcone_spec.experiments.planning_artifacts import (
    experiment_budget_from_dict,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.orchestration.native_terminal import (
    validate_native_terminal_artifact,
)
from lightcone_spec.orchestration.session import (
    SHARED_SESSION_FALLBACK_MODE,
    SHARED_SESSION_UNAVAILABLE_REASON,
    IndustrialServerBlockResult,
)
from lightcone_spec.runtime.attestation import RELEASE_TRUSTED_ATTESTER_POLICY
from lightcone_spec.runtime.compile_cache import (
    SGLANG_FIRST_PARTY_COMPILE_BUILDER,
    CompileCacheAttemptReceipt,
    CompileCacheFile,
    CompileCacheLaunchPlan,
    CompileCacheReceipt,
    ImmutableCompileCache,
    preflight_compile_cache_launch,
)
from lightcone_spec.telemetry.records import PerformanceRecord
from lightcone_spec.telemetry.writer import load_completed_evidence

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_REASON = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_BUDGET_OBSERVATION_KIND = "industrial_budget_observation_receipt_v1"
_BUDGET_OBSERVATION_FIELDS = frozenset(
    {
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
)
_BUDGET_COMPONENTS = (
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
_RESERVED_GANG_MEASUREMENT = "exclusive_reserved_gang_wall_ms_x_gpu_count"
_WHOLE_INSTANCE_BILLING = "whole_inventory_wall_clock_v1"


class RuntimeMetricStatus(str, Enum):
    """Availability and provenance state for one runtime metric."""

    OBSERVED = "OBSERVED"
    MEASURED = "MEASURED"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "N/A"


class RuntimeMetricSourceKind(str, Enum):
    COMPILE_CACHE = "compile_cache"
    FRESH_PROCESS_BUDGET = "fresh_process_budget"
    COMPLETED_PERFORMANCE = "completed_performance"
    NATIVE_TERMINAL = "native_terminal"


class RuntimeMetricUnit(str, Enum):
    COUNT = "count"
    MILLISECOND = "ms"
    RATIO = "ratio"
    BYTE = "byte"
    FLOP = "flop"
    FLOP_PER_TOKEN = "flop/token"
    BYTE_PER_TOKEN = "byte/token"
    WATT = "W"
    JOULE = "J"
    TOKEN_PER_JOULE = "token/J"


class RuntimeMetricName(str, Enum):
    COMPILE_CACHE_HITS = "compile_cache_hits"
    COMPILE_CACHE_MISSES = "compile_cache_misses"
    JIT_DURATION_MS = "jit_duration_ms"
    COLD_START_MS = "cold_start_ms"
    FRESH_PROCESS_RESET_FINALIZATION_MS = "fresh_process_reset_finalization_ms"
    RESET_DURATION_MS = "reset_duration_ms"
    REUSED_SESSION_STARTUP_SAVINGS_MS = "reused_session_startup_savings_ms"
    HTTP_CONNECTIONS_CREATED = "http_connections_created"
    HTTP_REUSED_REQUESTS = "http_reused_requests"
    GRAPH_REPLAY_HIT_RATE = "graph_replay_hit_rate"
    GRAPH_CAPTURE_MS = "graph_capture_ms"
    GRAPH_REPLAY_COUNT = "graph_replay_count"
    NVML_PROCESS_HBM_BYTES = "nvml_process_hbm_bytes"
    NVML_GLOBAL_HBM_BYTES = "nvml_global_hbm_bytes"
    EXECUTED_FLOPS = "executed_flops"
    COMMITTED_USEFUL_FLOPS = "committed_useful_flops"
    PRECISION_NORMALIZED_EXECUTED_MFU = "precision_normalized_executed_mfu"
    TARGET_EQUIVALENT_USEFUL_UTILIZATION = "target_equivalent_useful_utilization"
    EXECUTED_HBM_BYTES = "executed_hbm_bytes"
    EXECUTED_FLOPS_PER_COMMITTED_TOKEN = "executed_flops_per_committed_token"
    HBM_BYTES_PER_COMMITTED_TOKEN = "hbm_bytes_per_committed_token"
    POWER_WATTS = "power_watts"
    ENERGY_JOULES = "energy_joules"
    OUTPUT_TOKENS_PER_JOULE = "output_tokens_per_joule"


@dataclass(frozen=True)
class _MetricSpec:
    unit: RuntimeMetricUnit
    integral: bool = False
    maximum: float | None = None


_METRIC_SPECS: dict[RuntimeMetricName, _MetricSpec] = {
    RuntimeMetricName.COMPILE_CACHE_HITS: _MetricSpec(
        RuntimeMetricUnit.COUNT, integral=True
    ),
    RuntimeMetricName.COMPILE_CACHE_MISSES: _MetricSpec(
        RuntimeMetricUnit.COUNT, integral=True
    ),
    RuntimeMetricName.JIT_DURATION_MS: _MetricSpec(RuntimeMetricUnit.MILLISECOND),
    RuntimeMetricName.COLD_START_MS: _MetricSpec(RuntimeMetricUnit.MILLISECOND),
    RuntimeMetricName.FRESH_PROCESS_RESET_FINALIZATION_MS: _MetricSpec(
        RuntimeMetricUnit.MILLISECOND
    ),
    RuntimeMetricName.RESET_DURATION_MS: _MetricSpec(RuntimeMetricUnit.MILLISECOND),
    RuntimeMetricName.REUSED_SESSION_STARTUP_SAVINGS_MS: _MetricSpec(
        RuntimeMetricUnit.MILLISECOND
    ),
    RuntimeMetricName.HTTP_CONNECTIONS_CREATED: _MetricSpec(
        RuntimeMetricUnit.COUNT, integral=True
    ),
    RuntimeMetricName.HTTP_REUSED_REQUESTS: _MetricSpec(
        RuntimeMetricUnit.COUNT, integral=True
    ),
    RuntimeMetricName.GRAPH_REPLAY_HIT_RATE: _MetricSpec(
        RuntimeMetricUnit.RATIO, maximum=1.0
    ),
    RuntimeMetricName.GRAPH_CAPTURE_MS: _MetricSpec(RuntimeMetricUnit.MILLISECOND),
    RuntimeMetricName.GRAPH_REPLAY_COUNT: _MetricSpec(
        RuntimeMetricUnit.COUNT, integral=True
    ),
    RuntimeMetricName.NVML_PROCESS_HBM_BYTES: _MetricSpec(
        RuntimeMetricUnit.BYTE, integral=True
    ),
    RuntimeMetricName.NVML_GLOBAL_HBM_BYTES: _MetricSpec(
        RuntimeMetricUnit.BYTE, integral=True
    ),
    RuntimeMetricName.EXECUTED_FLOPS: _MetricSpec(RuntimeMetricUnit.FLOP),
    RuntimeMetricName.COMMITTED_USEFUL_FLOPS: _MetricSpec(RuntimeMetricUnit.FLOP),
    RuntimeMetricName.PRECISION_NORMALIZED_EXECUTED_MFU: _MetricSpec(
        RuntimeMetricUnit.RATIO
    ),
    RuntimeMetricName.TARGET_EQUIVALENT_USEFUL_UTILIZATION: _MetricSpec(
        RuntimeMetricUnit.RATIO
    ),
    RuntimeMetricName.EXECUTED_HBM_BYTES: _MetricSpec(
        RuntimeMetricUnit.BYTE, integral=True
    ),
    RuntimeMetricName.EXECUTED_FLOPS_PER_COMMITTED_TOKEN: _MetricSpec(
        RuntimeMetricUnit.FLOP_PER_TOKEN
    ),
    RuntimeMetricName.HBM_BYTES_PER_COMMITTED_TOKEN: _MetricSpec(
        RuntimeMetricUnit.BYTE_PER_TOKEN
    ),
    RuntimeMetricName.POWER_WATTS: _MetricSpec(RuntimeMetricUnit.WATT),
    RuntimeMetricName.ENERGY_JOULES: _MetricSpec(RuntimeMetricUnit.JOULE),
    RuntimeMetricName.OUTPUT_TOKENS_PER_JOULE: _MetricSpec(
        RuntimeMetricUnit.TOKEN_PER_JOULE
    ),
}

_COMPILE_METRICS = frozenset(
    {
        RuntimeMetricName.COMPILE_CACHE_HITS,
        RuntimeMetricName.COMPILE_CACHE_MISSES,
        RuntimeMetricName.JIT_DURATION_MS,
    }
)
_BUDGET_METRICS = frozenset(
    {
        RuntimeMetricName.COLD_START_MS,
        RuntimeMetricName.FRESH_PROCESS_RESET_FINALIZATION_MS,
        RuntimeMetricName.RESET_DURATION_MS,
        RuntimeMetricName.REUSED_SESSION_STARTUP_SAVINGS_MS,
    }
)
_PERFORMANCE_METRICS = frozenset(
    {
        RuntimeMetricName.HTTP_CONNECTIONS_CREATED,
        RuntimeMetricName.HTTP_REUSED_REQUESTS,
    }
)
_NATIVE_METRICS = frozenset(
    {
        RuntimeMetricName.GRAPH_REPLAY_HIT_RATE,
        RuntimeMetricName.GRAPH_CAPTURE_MS,
        RuntimeMetricName.GRAPH_REPLAY_COUNT,
        RuntimeMetricName.NVML_PROCESS_HBM_BYTES,
        RuntimeMetricName.NVML_GLOBAL_HBM_BYTES,
        RuntimeMetricName.EXECUTED_FLOPS,
        RuntimeMetricName.COMMITTED_USEFUL_FLOPS,
        RuntimeMetricName.PRECISION_NORMALIZED_EXECUTED_MFU,
        RuntimeMetricName.TARGET_EQUIVALENT_USEFUL_UTILIZATION,
        RuntimeMetricName.EXECUTED_HBM_BYTES,
        RuntimeMetricName.EXECUTED_FLOPS_PER_COMMITTED_TOKEN,
        RuntimeMetricName.HBM_BYTES_PER_COMMITTED_TOKEN,
        RuntimeMetricName.POWER_WATTS,
        RuntimeMetricName.ENERGY_JOULES,
        RuntimeMetricName.OUTPUT_TOKENS_PER_JOULE,
    }
)
_FORMAL_RUN_METRICS = frozenset(
    _BUDGET_METRICS | _PERFORMANCE_METRICS | _NATIVE_METRICS
)

RUNTIME_METRICS_REDUCER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "runtime_metrics_reducer_protocol",
        "states": [status.value for status in RuntimeMetricStatus],
        "source_kinds": [kind.value for kind in RuntimeMetricSourceKind],
        "compile_metrics": sorted(metric.value for metric in _COMPILE_METRICS),
        "fresh_process_budget_metrics": sorted(
            metric.value for metric in _BUDGET_METRICS
        ),
        "completed_performance_metrics": sorted(
            metric.value for metric in _PERFORMANCE_METRICS
        ),
        "native_terminal_metrics": sorted(metric.value for metric in _NATIVE_METRICS),
        "missing_numeric_values_are_never_zero": True,
        "fresh_process_does_not_claim_shared_reset_or_reuse_savings": True,
        "native_attestation_status_is_retained": True,
    }
)

FORMAL_RUNTIME_METRICS_EXPORT_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_runtime_metrics_export_protocol",
        "run_metrics": sorted(metric.value for metric in _FORMAL_RUN_METRICS),
        "run_identity": "exact_industrial_reducer_run_bindings",
        "source": "freshly_replayed_runtime_metrics_authority",
        "resolved_value_gate": "release_trusted_native_terminal_for_exact_run",
        "missing_authority_or_run_source": "UNRESOLVED_with_null_value",
        "untrusted_resolved_value": "downgrade_to_UNRESOLVED_with_null_value",
        "not_applicable": "preserve_NA_with_null_value",
        "compile_subjects": "authority_identity_only_not_confirmation_run_metrics",
    }
)


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _canonical_json_bytes(value: object, *, ensure_ascii: bool) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json(body: bytes, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")

    def require_finite(row: object) -> None:
        if isinstance(row, float) and not math.isfinite(row):
            raise ValueError(f"{label} contains a non-finite number")
        if isinstance(row, list):
            for item in row:
                require_finite(item)
        elif isinstance(row, dict):
            for item in row.values():
                require_finite(item)

    require_finite(value)
    return value


@dataclass(frozen=True)
class BoundRuntimeMetricsFile:
    """Raw path, byte length, and digest reopened by every reduction."""

    path: str
    size: int
    raw_sha256: str

    def __post_init__(self) -> None:
        path = Path(_require_text("runtime metric file path", self.path))
        if not path.is_absolute() or path.resolve(strict=False) != path:
            raise ValueError("runtime metric file path must be absolute and normalized")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("runtime metric file size must be non-negative")
        _require_sha256("runtime metric file digest", self.raw_sha256)

    @classmethod
    def bind(cls, path: str | Path) -> Self:
        requested = Path(path)
        if requested.is_symlink():
            raise ValueError("runtime metric source cannot be a symlink")
        try:
            resolved = requested.resolve(strict=True)
        except OSError as error:
            raise ValueError("runtime metric source does not exist") from error
        if not resolved.is_file():
            raise ValueError("runtime metric source must be a regular file")
        body = _stable_regular_file_bytes(resolved, label="runtime metric source")
        return cls(
            path=str(resolved),
            size=len(body),
            raw_sha256=hashlib.sha256(body).hexdigest(),
        )

    def read_bytes(self, *, label: str) -> bytes:
        body = _stable_regular_file_bytes(Path(self.path), label=label)
        if (
            len(body) != self.size
            or hashlib.sha256(body).hexdigest() != self.raw_sha256
        ):
            raise RuntimeError(f"{label} differs from its bound raw bytes")
        return body

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _stable_regular_file_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise ValueError(f"{label} path must be absolute and normalized")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or identity(before) != identity(after)
            or identity(after) != identity(current)
            or len(body) != after.st_size
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CompileRuntimeMetricsSource:
    subject_id: str
    plan: BoundRuntimeMetricsFile
    plan_sidecar: BoundRuntimeMetricsFile
    attempt: BoundRuntimeMetricsFile
    attempt_sidecar: BoundRuntimeMetricsFile
    result_receipt: BoundRuntimeMetricsFile
    result_receipt_sidecar: BoundRuntimeMetricsFile
    plan_sha256: str
    attempt_sha256: str
    result_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_text("compile metric subject_id", self.subject_id)
        for name in (
            "plan_sha256",
            "attempt_sha256",
            "result_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        files = (
            self.plan,
            self.plan_sidecar,
            self.attempt,
            self.attempt_sidecar,
            self.result_receipt,
            self.result_receipt_sidecar,
        )
        if any(type(value) is not BoundRuntimeMetricsFile for value in files):
            raise TypeError("compile metric source requires exact raw-file bindings")
        if len({value.path for value in files}) != len(files):
            raise ValueError("compile metric source reuses a raw-file path")

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "plan": self.plan.to_dict(),
            "plan_sidecar": self.plan_sidecar.to_dict(),
            "attempt": self.attempt.to_dict(),
            "attempt_sidecar": self.attempt_sidecar.to_dict(),
            "result_receipt": self.result_receipt.to_dict(),
            "result_receipt_sidecar": self.result_receipt_sidecar.to_dict(),
            "plan_sha256": self.plan_sha256,
            "attempt_sha256": self.attempt_sha256,
            "result_receipt_sha256": self.result_receipt_sha256,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "compile_runtime_metrics_source",
                **self.to_dict(),
            }
        )


@dataclass(frozen=True)
class NativeRuntimeMetricsSource:
    subject_id: str
    artifact: BoundRuntimeMetricsFile
    terminal_sha256: str
    trusted_attester_policy_sha256: str
    release_trusted_attestation: bool

    def __post_init__(self) -> None:
        _require_text("native metric subject_id", self.subject_id)
        if type(self.artifact) is not BoundRuntimeMetricsFile:
            raise TypeError("native metric source requires an exact raw artifact")
        _require_sha256("native terminal digest", self.terminal_sha256)
        _require_sha256(
            "native trusted-attester policy digest",
            self.trusted_attester_policy_sha256,
        )
        if type(self.release_trusted_attestation) is not bool:
            raise TypeError("native release trust state must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "artifact": self.artifact.to_dict(),
            "terminal_sha256": self.terminal_sha256,
            "trusted_attester_policy_sha256": (self.trusted_attester_policy_sha256),
            "release_trusted_attestation": self.release_trusted_attestation,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "native_runtime_metrics_source",
                **self.to_dict(),
            }
        )


@dataclass(frozen=True)
class FreshProcessExecutionMetricsSource:
    run_id: str
    execution_plan_sha256: str
    experiment_budget_sha256: str
    rank_config_sha256: str
    topology_sha256: str
    terminal_receipt_sha256: str
    budget_observation_sha256: str
    terminal_receipt: BoundRuntimeMetricsFile
    budget_observation: BoundRuntimeMetricsFile
    budget_observation_sidecar: BoundRuntimeMetricsFile
    evidence_files: tuple[BoundRuntimeMetricsFile, ...]
    run_evidence: BoundRuntimeMetricsFile
    performance_evidence: BoundRuntimeMetricsFile
    native_terminal: NativeRuntimeMetricsSource

    def __post_init__(self) -> None:
        _require_text("fresh-process run_id", self.run_id)
        for name in (
            "execution_plan_sha256",
            "experiment_budget_sha256",
            "rank_config_sha256",
            "topology_sha256",
            "terminal_receipt_sha256",
            "budget_observation_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for value in (
            self.terminal_receipt,
            self.budget_observation,
            self.budget_observation_sidecar,
            self.run_evidence,
            self.performance_evidence,
        ):
            if type(value) is not BoundRuntimeMetricsFile:
                raise TypeError("fresh-process source requires exact file bindings")
        if (
            not self.evidence_files
            or any(
                type(value) is not BoundRuntimeMetricsFile
                for value in self.evidence_files
            )
            or tuple(value.path for value in self.evidence_files)
            != tuple(sorted({value.path for value in self.evidence_files}))
        ):
            raise ValueError(
                "fresh-process evidence files must be sorted, non-empty, and unique"
            )
        if self.run_evidence.path not in {
            value.path for value in self.evidence_files
        } or self.performance_evidence.path not in {
            value.path for value in self.evidence_files
        }:
            raise ValueError("fresh-process evidence coverage omits a required table")
        if self.native_terminal.subject_id != self.run_id:
            raise ValueError("fresh-process native source names another run")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "execution_plan_sha256": self.execution_plan_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "topology_sha256": self.topology_sha256,
            "terminal_receipt_sha256": self.terminal_receipt_sha256,
            "budget_observation_sha256": self.budget_observation_sha256,
            "terminal_receipt": self.terminal_receipt.to_dict(),
            "budget_observation": self.budget_observation.to_dict(),
            "budget_observation_sidecar": self.budget_observation_sidecar.to_dict(),
            "evidence_files": [value.to_dict() for value in self.evidence_files],
            "run_evidence": self.run_evidence.to_dict(),
            "performance_evidence": self.performance_evidence.to_dict(),
            "native_terminal": self.native_terminal.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "fresh_process_execution_metrics_source",
                **self.to_dict(),
            }
        )


@dataclass(frozen=True)
class FreshProcessRuntimeMetricsSource:
    session_plan_sha256: str
    block_result_sha256: str
    execution_mode: str
    fallback_reason: str
    executions: tuple[FreshProcessExecutionMetricsSource, ...]

    def __post_init__(self) -> None:
        _require_sha256("session plan digest", self.session_plan_sha256)
        _require_sha256("fresh-process block digest", self.block_result_sha256)
        if self.execution_mode != SHARED_SESSION_FALLBACK_MODE:
            raise ValueError("runtime metrics require the release fresh-process mode")
        if self.fallback_reason != SHARED_SESSION_UNAVAILABLE_REASON:
            raise ValueError("runtime metrics require the canonical fallback reason")
        if (
            not self.executions
            or any(
                type(value) is not FreshProcessExecutionMetricsSource
                for value in self.executions
            )
            or len({value.run_id for value in self.executions}) != len(self.executions)
            or len({value.execution_plan_sha256 for value in self.executions})
            != len(self.executions)
        ):
            raise ValueError("fresh-process metric execution coverage is invalid")
        expected = _fresh_process_block_result_sha256(
            session_plan_sha256=self.session_plan_sha256,
            executions=self.executions,
        )
        if self.block_result_sha256 != expected:
            raise ValueError("fresh-process metric source changed its block result")

    def to_dict(self) -> dict[str, object]:
        return {
            "session_plan_sha256": self.session_plan_sha256,
            "block_result_sha256": self.block_result_sha256,
            "execution_mode": self.execution_mode,
            "fallback_reason": self.fallback_reason,
            "executions": [value.to_dict() for value in self.executions],
        }

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "fresh_process_runtime_metrics_source",
                **self.to_dict(),
            }
        )


def _fresh_process_block_result_sha256(
    *,
    session_plan_sha256: str,
    executions: tuple[FreshProcessExecutionMetricsSource, ...],
) -> str:
    """Reconstruct the source block identity without accepting a summary row."""

    _require_sha256("session plan digest", session_plan_sha256)
    if not executions or any(
        type(row) is not FreshProcessExecutionMetricsSource for row in executions
    ):
        raise TypeError("fresh-process block identity requires exact executions")
    return content_sha256(
        {
            "schema_version": 1,
            "session_plan_sha256": session_plan_sha256,
            "execution_mode": SHARED_SESSION_FALLBACK_MODE,
            "fallback_reason": SHARED_SESSION_UNAVAILABLE_REASON,
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
                for row in executions
            ],
        }
    )


@dataclass(frozen=True)
class RuntimeMetricsAuthority:
    schema_version: int
    reducer_protocol_sha256: str
    compile_sources: tuple[CompileRuntimeMetricsSource, ...]
    fresh_process_sources: tuple[FreshProcessRuntimeMetricsSource, ...]
    native_sources: tuple[NativeRuntimeMetricsSource, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("runtime metrics authority schema is unsupported")
        if self.reducer_protocol_sha256 != RUNTIME_METRICS_REDUCER_PROTOCOL_SHA256:
            raise ValueError("runtime metrics authority uses another reducer protocol")
        for values, expected_type, label in (
            (self.compile_sources, CompileRuntimeMetricsSource, "compile"),
            (
                self.fresh_process_sources,
                FreshProcessRuntimeMetricsSource,
                "fresh-process",
            ),
            (self.native_sources, NativeRuntimeMetricsSource, "native"),
        ):
            if any(type(value) is not expected_type for value in values):
                raise TypeError(f"runtime metrics {label} source has a foreign type")
            digests = tuple(value.sha256 for value in values)
            if digests != tuple(sorted(set(digests))):
                raise ValueError(
                    f"runtime metrics {label} sources must be SHA-sorted and unique"
                )
        if not (
            self.compile_sources or self.fresh_process_sources or self.native_sources
        ):
            raise ValueError("runtime metrics authority requires at least one source")
        compile_subjects = [value.subject_id for value in self.compile_sources]
        if len(compile_subjects) != len(set(compile_subjects)):
            raise ValueError("runtime metrics authority duplicates a compile subject")
        fresh_subjects = [
            execution.run_id
            for block in self.fresh_process_sources
            for execution in block.executions
        ]
        if len(fresh_subjects) != len(set(fresh_subjects)):
            raise ValueError("runtime metrics authority duplicates a fresh-process run")
        native_subjects = [value.subject_id for value in self.native_sources]
        if len(native_subjects) != len(set(native_subjects)):
            raise ValueError("runtime metrics authority duplicates a native run")
        if set(fresh_subjects).intersection(native_subjects):
            raise ValueError(
                "standalone native sources cannot duplicate fresh-process native evidence"
            )

    @property
    def source_sha256s(self) -> tuple[str, ...]:
        values = [value.sha256 for value in self.compile_sources]
        for block in self.fresh_process_sources:
            for execution in block.executions:
                values.extend((execution.sha256, execution.native_terminal.sha256))
        values.extend(value.sha256 for value in self.native_sources)
        return tuple(sorted(values))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "runtime_metrics_authority",
            "reducer_protocol_sha256": self.reducer_protocol_sha256,
            "compile_sources": [value.to_dict() for value in self.compile_sources],
            "fresh_process_sources": [
                value.to_dict() for value in self.fresh_process_sources
            ],
            "native_sources": [value.to_dict() for value in self.native_sources],
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class RuntimeMetricObservation:
    subject_id: str
    metric: RuntimeMetricName
    unit: RuntimeMetricUnit
    status: RuntimeMetricStatus
    value: int | float | None
    source_kind: RuntimeMetricSourceKind
    source_sha256: str
    source_field: str
    reason_code: str | None
    release_trusted_attestation: bool | None = None

    def __post_init__(self) -> None:
        _require_text("runtime metric subject_id", self.subject_id)
        if not isinstance(self.metric, RuntimeMetricName):
            raise TypeError("runtime metric name must be a RuntimeMetricName")
        if not isinstance(self.unit, RuntimeMetricUnit):
            raise TypeError("runtime metric unit must be a RuntimeMetricUnit")
        if not isinstance(self.status, RuntimeMetricStatus):
            raise TypeError("runtime metric status must be a RuntimeMetricStatus")
        if not isinstance(self.source_kind, RuntimeMetricSourceKind):
            raise TypeError("runtime metric source kind is invalid")
        _require_sha256("runtime metric source digest", self.source_sha256)
        _require_text("runtime metric source field", self.source_field)
        if (
            self.release_trusted_attestation is not None
            and type(self.release_trusted_attestation) is not bool
        ):
            raise TypeError("runtime metric native trust state must be bool or None")
        if self.source_kind is RuntimeMetricSourceKind.NATIVE_TERMINAL:
            if self.release_trusted_attestation is None:
                raise ValueError("native runtime metrics must retain attestation state")
        elif self.release_trusted_attestation is not None:
            raise ValueError("non-native runtime metrics cannot claim native trust")
        spec = _METRIC_SPECS[self.metric]
        if self.unit is not spec.unit:
            raise ValueError("runtime metric unit differs from its schema")
        resolved = self.status in {
            RuntimeMetricStatus.OBSERVED,
            RuntimeMetricStatus.MEASURED,
        }
        if resolved:
            if self.reason_code is not None or self.value is None:
                raise ValueError("resolved runtime metric has missing/conflicting data")
            if (
                not isinstance(self.value, (int, float))
                or isinstance(self.value, bool)
                or not math.isfinite(float(self.value))
                or float(self.value) < 0
            ):
                raise ValueError(
                    "resolved runtime metric must be finite and non-negative"
                )
            if spec.integral and type(self.value) is not int:
                raise TypeError("integral runtime metric must use an exact integer")
            if spec.maximum is not None and float(self.value) > spec.maximum:
                raise ValueError("runtime metric exceeds its registered maximum")
        else:
            if self.value is not None:
                raise ValueError("unavailable runtime metric cannot carry a value")
            if (
                not isinstance(self.reason_code, str)
                or _SAFE_REASON.fullmatch(self.reason_code) is None
            ):
                raise ValueError("unavailable runtime metric needs a named reason")

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject_id, self.metric.value)

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "metric": self.metric.value,
            "unit": self.unit.value,
            "status": self.status.value,
            "value": self.value,
            "source_kind": self.source_kind.value,
            "source_sha256": self.source_sha256,
            "source_field": self.source_field,
            "reason_code": self.reason_code,
            "release_trusted_attestation": self.release_trusted_attestation,
        }


@dataclass(frozen=True)
class RuntimeMetricsReduction:
    schema_version: int
    authority_sha256: str
    reducer_protocol_sha256: str
    source_sha256s: tuple[str, ...]
    observations: tuple[RuntimeMetricObservation, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("runtime metrics reduction schema is unsupported")
        _require_sha256("runtime metrics authority digest", self.authority_sha256)
        if self.reducer_protocol_sha256 != RUNTIME_METRICS_REDUCER_PROTOCOL_SHA256:
            raise ValueError("runtime metrics reduction uses another protocol")
        if self.source_sha256s != tuple(sorted(set(self.source_sha256s))):
            raise ValueError("runtime metrics source coverage is not canonical")
        if any(_SHA256.fullmatch(value) is None for value in self.source_sha256s):
            raise ValueError(
                "runtime metrics source coverage contains an invalid digest"
            )
        if not self.observations or any(
            type(value) is not RuntimeMetricObservation for value in self.observations
        ):
            raise ValueError("runtime metrics reduction requires typed observations")
        keys = tuple(value.key for value in self.observations)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("runtime metric observations must be sorted and unique")
        if any(
            value.source_sha256 not in self.source_sha256s
            for value in self.observations
        ):
            raise ValueError("runtime metric observation names an unbound source")

    def validate_against(self, authority: RuntimeMetricsAuthority) -> None:
        if type(authority) is not RuntimeMetricsAuthority:
            raise TypeError("runtime metrics validation requires an exact authority")
        authority.__post_init__()
        if (
            self.authority_sha256 != authority.sha256
            or self.source_sha256s != authority.source_sha256s
        ):
            raise ValueError("runtime metrics reduction differs from its authority")

    def observation(
        self, subject_id: str, metric: RuntimeMetricName
    ) -> RuntimeMetricObservation:
        matches = [
            value
            for value in self.observations
            if value.subject_id == subject_id and value.metric is metric
        ]
        if len(matches) != 1:
            raise KeyError(f"runtime metric {metric.value} is not singular")
        return matches[0]

    def resolved_performance_overrides(self, subject_id: str) -> dict[str, int | float]:
        """Return diagnostic receipt-backed values understood by ``PerformanceRecord``.

        UNRESOLVED and N/A rows are deliberately omitted rather than converted
        to zero.  The fresh-process reset/finalization composite is also omitted
        because it is not the shared-session ``reset_duration_ms`` field.
        This mapping is not a formal-claim boundary: industrial reducers must
        use :func:`export_formal_runtime_metrics`, which additionally requires
        exact run coverage and release-trusted native evidence.
        """

        _require_text("runtime metric subject_id", subject_id)
        fields = PerformanceRecord.__dataclass_fields__
        result: dict[str, int | float] = {}
        for observation in self.observations:
            name = observation.metric.value
            if (
                observation.subject_id != subject_id
                or observation.status
                not in {
                    RuntimeMetricStatus.OBSERVED,
                    RuntimeMetricStatus.MEASURED,
                }
                or (
                    observation.source_kind is RuntimeMetricSourceKind.NATIVE_TERMINAL
                    and observation.release_trusted_attestation is not True
                )
                or name not in fields
            ):
                continue
            assert observation.value is not None
            if name in result:
                raise RuntimeError("runtime metric maps twice to PerformanceRecord")
            result[name] = observation.value
        return result

    def apply_to_performance_record(
        self, record: PerformanceRecord
    ) -> PerformanceRecord:
        if type(record) is not PerformanceRecord:
            raise TypeError(
                "runtime metric mapping requires an exact PerformanceRecord"
            )
        overrides = self.resolved_performance_overrides(record.run_id)
        for name, value in overrides.items():
            existing = getattr(record, name)
            if existing is not None and existing != value:
                raise ValueError(
                    f"runtime metric source conflicts with PerformanceRecord.{name}"
                )
        return replace(record, **overrides)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "runtime_metrics_reduction",
            "authority_sha256": self.authority_sha256,
            "reducer_protocol_sha256": self.reducer_protocol_sha256,
            "source_sha256s": list(self.source_sha256s),
            "observations": [value.to_dict() for value in self.observations],
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class FormalRuntimeMetricObservation:
    """One claim-safe metric row exported to industrial analysis.

    This schema intentionally permits absent provenance only for a synthesized
    missing-source row.  A value can survive only when the exact run is covered
    by a release-trusted native terminal source.
    """

    subject_id: str
    metric: RuntimeMetricName
    unit: RuntimeMetricUnit
    status: RuntimeMetricStatus
    value: int | float | None
    source_kind: RuntimeMetricSourceKind | None
    source_sha256: str | None
    reason_code: str | None
    release_trusted: bool

    def __post_init__(self) -> None:
        _require_text("formal runtime metric subject_id", self.subject_id)
        if not isinstance(self.metric, RuntimeMetricName):
            raise TypeError("formal runtime metric name must be typed")
        if self.metric not in _FORMAL_RUN_METRICS:
            raise ValueError("compile-only metrics cannot become formal run fields")
        if self.unit is not _METRIC_SPECS[self.metric].unit:
            raise ValueError("formal runtime metric unit differs from its schema")
        if not isinstance(self.status, RuntimeMetricStatus):
            raise TypeError("formal runtime metric status must be typed")
        if type(self.release_trusted) is not bool:
            raise TypeError("formal runtime trust state must be a boolean")
        if (self.source_kind is None) != (self.source_sha256 is None):
            raise ValueError(
                "formal runtime provenance must be wholly present or absent"
            )
        if self.source_kind is not None:
            if not isinstance(self.source_kind, RuntimeMetricSourceKind):
                raise TypeError("formal runtime source kind must be typed")
            assert self.source_sha256 is not None
            _require_sha256("formal runtime source", self.source_sha256)
        resolved = self.status in {
            RuntimeMetricStatus.OBSERVED,
            RuntimeMetricStatus.MEASURED,
        }
        if resolved:
            if not self.release_trusted:
                raise ValueError(
                    "untrusted runtime source cannot publish a formal value"
                )
            if self.value is None or self.reason_code is not None:
                raise ValueError("resolved formal runtime metric is incomplete")
            spec = _METRIC_SPECS[self.metric]
            if (
                not isinstance(self.value, (int, float))
                or isinstance(self.value, bool)
                or not math.isfinite(float(self.value))
                or float(self.value) < 0
            ):
                raise ValueError("formal runtime metric value must be finite")
            if spec.integral and type(self.value) is not int:
                raise TypeError("integral formal runtime metric must remain exact")
            if spec.maximum is not None and float(self.value) > spec.maximum:
                raise ValueError("formal runtime metric exceeds its maximum")
        elif self.value is not None:
            raise ValueError("unavailable formal runtime metric cannot carry a value")
        elif (
            not isinstance(self.reason_code, str)
            or _SAFE_REASON.fullmatch(self.reason_code) is None
        ):
            raise ValueError("unavailable formal runtime metric needs a named reason")

    @property
    def key(self) -> tuple[str, str]:
        return self.subject_id, self.metric.value

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "metric": self.metric.value,
            "unit": self.unit.value,
            "status": self.status.value,
            "value": self.value,
            "source_kind": (
                None if self.source_kind is None else self.source_kind.value
            ),
            "source_sha256": self.source_sha256,
            "reason_code": self.reason_code,
            "release_trusted": self.release_trusted,
        }


@dataclass(frozen=True)
class FormalRuntimeMetricsExport:
    schema_version: int
    protocol_sha256: str
    status: RuntimeMetricStatus
    authority_sha256: str | None
    reduction_sha256: str | None
    source_sha256s: tuple[str, ...]
    expected_run_ids: tuple[str, ...]
    observations: tuple[FormalRuntimeMetricObservation, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("formal runtime metrics export schema is unsupported")
        if self.protocol_sha256 != FORMAL_RUNTIME_METRICS_EXPORT_PROTOCOL_SHA256:
            raise ValueError("formal runtime metrics export uses another protocol")
        if self.status not in {
            RuntimeMetricStatus.OBSERVED,
            RuntimeMetricStatus.UNRESOLVED,
        }:
            raise ValueError("formal runtime metrics export status is invalid")
        if (self.authority_sha256 is None) != (self.reduction_sha256 is None):
            raise ValueError("formal runtime authority/reduction identity is partial")
        if self.authority_sha256 is not None:
            _require_sha256("formal runtime authority", self.authority_sha256)
            _require_sha256("formal runtime reduction", self.reduction_sha256)
        if self.source_sha256s != tuple(sorted(set(self.source_sha256s))):
            raise ValueError("formal runtime source coverage is not canonical")
        for value in self.source_sha256s:
            _require_sha256("formal runtime source", value)
        if self.expected_run_ids != tuple(sorted(set(self.expected_run_ids))):
            raise ValueError("formal runtime run IDs must be sorted and unique")
        if not self.expected_run_ids:
            raise ValueError("formal runtime export requires expected runs")
        for value in self.expected_run_ids:
            _require_text("formal runtime run ID", value)
        keys = tuple(value.key for value in self.observations)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("formal runtime observations must be sorted and unique")
        expected_keys = tuple(
            sorted(
                (run_id, metric.value)
                for run_id in self.expected_run_ids
                for metric in _FORMAL_RUN_METRICS
            )
        )
        if keys != expected_keys:
            raise ValueError(
                "formal runtime observations lack exact run/metric coverage"
            )
        unresolved = any(
            row.status is RuntimeMetricStatus.UNRESOLVED for row in self.observations
        )
        if (self.status is RuntimeMetricStatus.UNRESOLVED) != unresolved:
            raise ValueError("formal runtime aggregate status differs from its rows")

    def observation(
        self, subject_id: str, metric: RuntimeMetricName
    ) -> FormalRuntimeMetricObservation:
        matches = tuple(
            row
            for row in self.observations
            if row.subject_id == subject_id and row.metric is metric
        )
        if len(matches) != 1:
            raise KeyError(f"formal runtime metric {metric.value} is not singular")
        return matches[0]

    def formal_values(self, subject_id: str) -> dict[str, int | float]:
        _require_text("formal runtime subject", subject_id)
        if subject_id not in self.expected_run_ids:
            raise KeyError("formal runtime subject is outside the export")
        return {
            row.metric.value: row.value
            for row in self.observations
            if row.subject_id == subject_id
            and row.status
            in {RuntimeMetricStatus.OBSERVED, RuntimeMetricStatus.MEASURED}
            and row.release_trusted
            and row.value is not None
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "formal_runtime_metrics_export",
            "protocol_sha256": self.protocol_sha256,
            "status": self.status.value,
            "authority_sha256": self.authority_sha256,
            "reduction_sha256": self.reduction_sha256,
            "source_sha256s": list(self.source_sha256s),
            "expected_run_ids": list(self.expected_run_ids),
            "observations": [row.to_dict() for row in self.observations],
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _sidecar(path: BoundRuntimeMetricsFile) -> BoundRuntimeMetricsFile:
    source = Path(path.path)
    return BoundRuntimeMetricsFile.bind(source.with_name(f"{source.name}.sha256"))


def _parse_compile_plan(
    source: CompileRuntimeMetricsSource,
) -> CompileCacheLaunchPlan:
    body = source.plan.read_bytes(label="compile launch plan")
    raw = _strict_json(body, label="compile launch plan")
    if body != _canonical_json_bytes(raw, ensure_ascii=False):
        raise ValueError("compile launch plan is not canonical JSON")
    plan = CompileCacheLaunchPlan.from_dict(dict(raw))
    if plan.sha256 != source.plan_sha256:
        raise RuntimeError("compile launch plan semantic digest changed")
    sidecar = source.plan_sidecar.read_bytes(label="compile launch plan sidecar")
    if sidecar != f"{plan.sha256}\n".encode("ascii"):
        raise RuntimeError("compile launch plan sidecar differs from content")
    return plan


def _parse_compile_attempt(
    source: CompileRuntimeMetricsSource,
) -> CompileCacheAttemptReceipt:
    body = source.attempt.read_bytes(label="compile attempt receipt")
    raw = _strict_json(body, label="compile attempt receipt")
    if body != _canonical_json_bytes(raw, ensure_ascii=False):
        raise ValueError("compile attempt receipt is not canonical JSON")
    expected = {
        "schema_version",
        "kind",
        "plan_sha256",
        "key_sha256",
        "attempt_id",
        "process_id",
        "state",
        "started_ns",
        "finished_ns",
        "overlay_name",
        "base_receipt_sha256",
        "result_receipt_sha256",
        "failure_code",
        "failure_detail_sha256",
        "environment",
    }
    if set(raw) != expected:
        raise ValueError("compile attempt receipt has unknown or missing fields")
    payload = dict(raw)
    environment = payload.pop("environment")
    if type(environment) is not list or any(
        type(row) is not list or len(row) != 2 for row in environment
    ):
        raise TypeError("compile attempt environment must contain JSON pairs")
    attempt = CompileCacheAttemptReceipt(
        **payload,
        environment=tuple(tuple(row) for row in environment),
    )
    attempt.validate()
    if attempt.sha256 != source.attempt_sha256:
        raise RuntimeError("compile attempt semantic digest changed")
    sidecar = source.attempt_sidecar.read_bytes(label="compile attempt sidecar")
    if sidecar != f"{attempt.sha256}\n".encode("ascii"):
        raise RuntimeError("compile attempt sidecar differs from content")
    return attempt


def _parse_compile_receipt(
    source: CompileRuntimeMetricsSource,
) -> CompileCacheReceipt:
    body = source.result_receipt.read_bytes(label="compile result receipt")
    raw = _strict_json(body, label="compile result receipt")
    if body != _canonical_json_bytes(raw, ensure_ascii=False):
        raise ValueError("compile result receipt is not canonical JSON")
    expected = {
        "schema_version",
        "kind",
        "key_sha256",
        "content_sha256",
        "builder_id",
        "launch_plan_sha256",
        "attempt_id",
        "process_id",
        "jit_duration_ns",
        "files",
    }
    if set(raw) != expected:
        raise ValueError("compile result receipt has unknown or missing fields")
    payload = dict(raw)
    files = payload.pop("files")
    if type(files) is not list or any(type(row) is not dict for row in files):
        raise TypeError("compile result receipt files must be JSON objects")
    receipt = CompileCacheReceipt(
        **payload,
        files=tuple(CompileCacheFile(**row) for row in files),
    )
    receipt.validate()
    if receipt.receipt_sha256 != source.result_receipt_sha256:
        raise RuntimeError("compile result receipt semantic digest changed")
    sidecar = source.result_receipt_sidecar.read_bytes(
        label="compile result receipt sidecar"
    )
    if sidecar != f"{receipt.receipt_sha256}\n".encode("ascii"):
        raise RuntimeError("compile result receipt sidecar differs from content")
    return receipt


def _reopen_compile_source(
    source: CompileRuntimeMetricsSource,
) -> tuple[CompileCacheLaunchPlan, CompileCacheAttemptReceipt, CompileCacheReceipt]:
    if type(source) is not CompileRuntimeMetricsSource:
        raise TypeError("compile runtime metrics require an exact source")
    source.__post_init__()
    plan = _parse_compile_plan(source)
    attempt = _parse_compile_attempt(source)
    receipt = _parse_compile_receipt(source)
    root = Path(plan.cache_root)
    if (
        Path(source.attempt.path).parent != root / "attempts"
        or Path(source.attempt.path).name != f"{attempt.sha256}.json"
        or Path(source.result_receipt.path).parent != root / "receipts"
        or Path(source.result_receipt.path).name != f"{receipt.receipt_sha256}.json"
    ):
        raise ValueError("compile metric receipts are outside their canonical store")
    if (
        plan.builder_id != SGLANG_FIRST_PARTY_COMPILE_BUILDER
        or attempt.state != "complete"
        or attempt.plan_sha256 != plan.sha256
        or attempt.key_sha256 != plan.key.sha256
        or attempt.result_receipt_sha256 != receipt.receipt_sha256
        or attempt.attempt_id != receipt.attempt_id
        or attempt.process_id != receipt.process_id
        or receipt.builder_id != plan.builder_id
        or receipt.launch_plan_sha256 != plan.sha256
        or receipt.key_sha256 != plan.key.sha256
        or attempt.base_receipt_sha256 != plan.base_receipt_sha256
    ):
        raise RuntimeError("compile metric receipts do not describe one exact attempt")
    preflight_compile_cache_launch(plan)
    cache = ImmutableCompileCache._open_existing_read_only(root)
    cache.verify(plan.key, source.result_receipt.path)
    # Close the read/validation race by reopening every bound artifact.
    _parse_compile_plan(source)
    _parse_compile_attempt(source)
    _parse_compile_receipt(source)
    return plan, attempt, receipt


def bind_compile_runtime_metrics(
    *,
    plan_path: str | Path,
    attempt_path: str | Path,
    result_receipt_path: str | Path,
    subject_id: str | None = None,
) -> CompileRuntimeMetricsSource:
    plan_file = BoundRuntimeMetricsFile.bind(plan_path)
    attempt_file = BoundRuntimeMetricsFile.bind(attempt_path)
    receipt_file = BoundRuntimeMetricsFile.bind(result_receipt_path)

    # Parse once to bind semantic identities, then replay the full cross-file
    # contract before returning the source.
    plan_raw = _strict_json(
        plan_file.read_bytes(label="compile launch plan"),
        label="compile launch plan",
    )
    attempt_raw = _strict_json(
        attempt_file.read_bytes(label="compile attempt receipt"),
        label="compile attempt receipt",
    )
    receipt_raw = _strict_json(
        receipt_file.read_bytes(label="compile result receipt"),
        label="compile result receipt",
    )
    plan = CompileCacheLaunchPlan.from_dict(dict(plan_raw))
    attempt_payload = dict(attempt_raw)
    environment = attempt_payload.pop("environment", None)
    if type(environment) is not list:
        raise TypeError("compile attempt environment must be a JSON list")
    attempt = CompileCacheAttemptReceipt(
        **attempt_payload,
        environment=tuple(tuple(row) for row in environment),
    )
    attempt.validate()
    receipt_payload = dict(receipt_raw)
    file_rows = receipt_payload.pop("files", None)
    if type(file_rows) is not list:
        raise TypeError("compile result receipt files must be a JSON list")
    receipt = CompileCacheReceipt(
        **receipt_payload,
        files=tuple(CompileCacheFile(**row) for row in file_rows),
    )
    receipt.validate()
    value = CompileRuntimeMetricsSource(
        subject_id=(
            attempt.attempt_id
            if subject_id is None
            else _require_text("subject_id", subject_id)
        ),
        plan=plan_file,
        plan_sidecar=_sidecar(plan_file),
        attempt=attempt_file,
        attempt_sidecar=_sidecar(attempt_file),
        result_receipt=receipt_file,
        result_receipt_sidecar=_sidecar(receipt_file),
        plan_sha256=plan.sha256,
        attempt_sha256=attempt.sha256,
        result_receipt_sha256=receipt.receipt_sha256,
    )
    _reopen_compile_source(value)
    return value


def _reopen_native_source(
    source: NativeRuntimeMetricsSource,
) -> tuple[dict[str, object], dict[str, object]]:
    if type(source) is not NativeRuntimeMetricsSource:
        raise TypeError("native runtime metrics require an exact source")
    source.__post_init__()
    body = source.artifact.read_bytes(label="native terminal artifact")
    artifact = _strict_json(body, label="native terminal artifact")
    if body != _canonical_json_bytes(artifact, ensure_ascii=True):
        raise ValueError("native terminal artifact is not canonical JSON")
    evidence = validate_native_terminal_artifact(
        artifact,
        trusted_attester_policy=RELEASE_TRUSTED_ATTESTER_POLICY,
    )
    if (
        evidence.binding.run_id != source.subject_id
        or evidence.terminal_sha256 != source.terminal_sha256
        or evidence.trusted_attester_policy_sha256
        != source.trusted_attester_policy_sha256
        or evidence.trusted_attestation != source.release_trusted_attestation
        or source.trusted_attester_policy_sha256
        != RELEASE_TRUSTED_ATTESTER_POLICY.sha256
    ):
        raise RuntimeError("native runtime metric source changed its exact binding")
    terminal = evidence.to_dict()
    performance = terminal.get("performance_counters")
    if type(performance) is not dict:
        raise RuntimeError("native terminal lacks validated performance counters")
    source.artifact.read_bytes(label="native terminal artifact")
    return artifact, performance


def bind_native_runtime_metrics(
    artifact_path: str | Path,
) -> NativeRuntimeMetricsSource:
    artifact_file = BoundRuntimeMetricsFile.bind(artifact_path)
    body = artifact_file.read_bytes(label="native terminal artifact")
    artifact = _strict_json(body, label="native terminal artifact")
    if body != _canonical_json_bytes(artifact, ensure_ascii=True):
        raise ValueError("native terminal artifact is not canonical JSON")
    evidence = validate_native_terminal_artifact(
        artifact,
        trusted_attester_policy=RELEASE_TRUSTED_ATTESTER_POLICY,
    )
    value = NativeRuntimeMetricsSource(
        subject_id=evidence.binding.run_id,
        artifact=artifact_file,
        terminal_sha256=evidence.terminal_sha256,
        trusted_attester_policy_sha256=evidence.trusted_attester_policy_sha256,
        release_trusted_attestation=evidence.trusted_attestation,
    )
    _reopen_native_source(value)
    return value


def _strict_terminal_receipt(
    source: FreshProcessExecutionMetricsSource,
) -> dict[str, object]:
    body = source.terminal_receipt.read_bytes(label="fresh-process terminal receipt")
    value = _strict_json(body, label="fresh-process terminal receipt")
    if body != _canonical_json_bytes(value, ensure_ascii=True):
        raise ValueError("fresh-process terminal receipt is not canonical JSON")
    if hashlib.sha256(body).hexdigest() != source.terminal_receipt_sha256:
        raise RuntimeError("fresh-process terminal receipt digest changed")
    return value


def _strict_budget_observation(
    source: FreshProcessExecutionMetricsSource,
    *,
    terminal: dict[str, object],
) -> tuple[BudgetObservationReceipt, dict[str, int]]:
    body = source.budget_observation.read_bytes(
        label="fresh-process budget observation"
    )
    artifact = _strict_json(body, label="fresh-process budget observation")
    if body != _canonical_json_bytes(artifact, ensure_ascii=True):
        raise ValueError("fresh-process budget observation is not canonical JSON")
    if set(artifact) != _BUDGET_OBSERVATION_FIELDS:
        raise ValueError("fresh-process budget observation schema differs")
    if (
        artifact["schema_version"] != 1
        or artifact["artifact_kind"] != _BUDGET_OBSERVATION_KIND
        or artifact["experiment_budget_sha256"] != source.experiment_budget_sha256
        or artifact["budget_observation_sha256"] != source.budget_observation_sha256
        or artifact["gpu_measurement_semantics"] != _RESERVED_GANG_MEASUREMENT
        or artifact["fixed_instance_billing_semantics"] != _WHOLE_INSTANCE_BILLING
    ):
        raise RuntimeError("fresh-process budget observation identity differs")
    raw_budget = artifact["budget"]
    if type(raw_budget) is not dict:
        raise TypeError("fresh-process budget observation has a malformed budget")
    budget = experiment_budget_from_dict(
        {
            "artifact_kind": "experiment_budget",
            "artifact_sha256": source.experiment_budget_sha256,
            **raw_budget,
        }
    )
    rows = artifact["observed_component_ms"]
    if (
        type(rows) is not list
        or tuple(row[0] for row in rows if type(row) is list and len(row) == 2)
        != _BUDGET_COMPONENTS
        or any(
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not int
            or row[1] < 0
            for row in rows
        )
    ):
        raise ValueError("fresh-process budget components are malformed")
    observation = BudgetObservationReceipt(
        schema_version=1,
        budget=budget,
        observed_component_ms=tuple((row[0], row[1]) for row in rows),
        measured_gpu_ms=artifact["measured_gpu_ms"],
        fixed_instance_billed_gpu_ms=artifact["fixed_instance_billed_gpu_ms"],
        terminal_evidence_sha256=artifact["terminal_evidence_sha256"],
    )
    prepared_sha256 = terminal.get("prepared_receipt_sha256")
    if (
        observation.sha256 != source.budget_observation_sha256
        or observation.terminal_evidence_sha256 != prepared_sha256
        or artifact["observed_wall_ms"] != observation.observed_wall_ms
        or artifact["registered_wall_delta_ms"] != observation.registered_wall_delta_ms
        or artifact["registered_gpu_delta_ms"] != observation.registered_gpu_delta_ms
        or artifact["registered_billed_delta_ms"]
        != observation.registered_billed_delta_ms
    ):
        raise RuntimeError("fresh-process budget observation accounting differs")
    sidecar = source.budget_observation_sidecar.read_bytes(
        label="fresh-process budget observation sidecar"
    )
    if sidecar != f"{observation.sha256}\n".encode("ascii"):
        raise RuntimeError("fresh-process budget observation sidecar differs")
    return observation, dict(observation.observed_component_ms)


def _reopen_fresh_execution(
    source: FreshProcessExecutionMetricsSource,
) -> tuple[dict[str, int], dict[str, object], dict[str, object]]:
    if type(source) is not FreshProcessExecutionMetricsSource:
        raise TypeError("fresh-process runtime metrics require an exact source")
    source.__post_init__()
    root = Path(source.terminal_receipt.path).parent
    if (
        Path(source.terminal_receipt.path).name
        != f"{source.run_id}.rank0.complete.json"
    ):
        raise ValueError("fresh-process terminal receipt path is not canonical")
    expected_observation_directory = root / (
        f"{source.run_id}.rank0.budget-observation"
    )
    if (
        Path(source.budget_observation.path)
        != expected_observation_directory / "observation.json"
        or Path(source.budget_observation_sidecar.path)
        != expected_observation_directory / "observation.json.sha256"
    ):
        raise ValueError("fresh-process budget observation path is not canonical")
    terminal = _strict_terminal_receipt(source)
    completed = load_completed_evidence(root, run_id=source.run_id, rank=0)
    if completed is None:
        raise RuntimeError("fresh-process runtime metric source is not complete")
    if completed.get("run") != Path(source.run_evidence.path) or completed.get(
        "performance"
    ) != Path(source.performance_evidence.path):
        raise RuntimeError("fresh-process table paths changed after binding")
    native_binding = terminal.get("native_terminal_artifact")
    if type(native_binding) is not dict:
        raise RuntimeError("fresh-process terminal lacks native evidence")
    if (
        native_binding.get("path") != Path(source.native_terminal.artifact.path).name
        or Path(source.native_terminal.artifact.path).parent != root
        or native_binding.get("size") != source.native_terminal.artifact.size
        or native_binding.get("raw_sha256")
        != source.native_terminal.artifact.raw_sha256
        or native_binding.get("terminal_sha256")
        != source.native_terminal.terminal_sha256
        or native_binding.get("trusted_attester_policy_sha256")
        != source.native_terminal.trusted_attester_policy_sha256
    ):
        raise RuntimeError("fresh-process native terminal binding changed")
    expected_evidence = tuple(
        sorted(
            {
                *(str(path.resolve()) for path in completed.values()),
                source.native_terminal.artifact.path,
                source.budget_observation.path,
                source.budget_observation_sidecar.path,
            }
        )
    )
    if expected_evidence != tuple(value.path for value in source.evidence_files):
        raise RuntimeError("fresh-process raw evidence coverage changed")
    for binding in source.evidence_files:
        binding.read_bytes(label="fresh-process bound evidence")
    try:
        run_rows = pq.read_table(
            source.run_evidence.path,
            columns=[
                "run_id",
                "runtime_sha256",
                "rank_config_sha256",
                "topology_sha256",
                "experiment_budget_sha256",
            ],
        ).to_pylist()
        performance_rows = pq.read_table(
            source.performance_evidence.path,
            columns=[
                "run_id",
                "http_connections_created",
                "http_reused_requests",
            ],
        ).to_pylist()
    except (KeyError, pa.ArrowException) as error:
        raise RuntimeError(
            "fresh-process runtime metric tables are malformed"
        ) from error
    if len(run_rows) != 1 or len(performance_rows) != 1:
        raise RuntimeError("fresh-process runtime metric tables are not singular")
    run = run_rows[0]
    if (
        run["run_id"] != source.run_id
        or run["runtime_sha256"] != source.execution_plan_sha256
        or run["rank_config_sha256"] != source.rank_config_sha256
        or run["topology_sha256"] != source.topology_sha256
        or run["experiment_budget_sha256"] != source.experiment_budget_sha256
        or performance_rows[0]["run_id"] != source.run_id
    ):
        raise RuntimeError("fresh-process runtime metric table identity differs")
    _, components = _strict_budget_observation(source, terminal=terminal)
    _, native_performance = _reopen_native_source(source.native_terminal)
    source.run_evidence.read_bytes(label="fresh-process run evidence")
    source.performance_evidence.read_bytes(label="fresh-process performance evidence")
    _strict_terminal_receipt(source)
    _strict_budget_observation(source, terminal=terminal)
    return components, performance_rows[0], native_performance


def bind_fresh_process_runtime_metrics(
    block_result: IndustrialServerBlockResult,
) -> FreshProcessRuntimeMetricsSource:
    if type(block_result) is not IndustrialServerBlockResult:
        raise TypeError("fresh-process runtime metrics require an exact block result")
    block_result.validate()
    executions: list[FreshProcessExecutionMetricsSource] = []
    for result in block_result.executions:
        if result.resumed:
            raise ValueError("fresh-process runtime metrics cannot bind a resumed run")
        terminal = BoundRuntimeMetricsFile.bind(result.terminal_receipt)
        if terminal.raw_sha256 != result.terminal_receipt_sha256:
            raise RuntimeError("execution summary has a foreign terminal receipt")
        root = Path(terminal.path).parent
        completed = load_completed_evidence(root, run_id=result.run_id, rank=0)
        if completed is None:
            raise RuntimeError("fresh-process execution lacks completed evidence")
        terminal_value = _strict_json(
            terminal.read_bytes(label="fresh-process terminal receipt"),
            label="fresh-process terminal receipt",
        )
        native_binding = terminal_value.get("native_terminal_artifact")
        if (
            type(native_binding) is not dict
            or type(native_binding.get("path")) is not str
        ):
            raise RuntimeError("fresh-process execution lacks native terminal evidence")
        native = bind_native_runtime_metrics(root / str(native_binding["path"]))
        observation = BoundRuntimeMetricsFile.bind(result.budget_observation)
        observation_sidecar = BoundRuntimeMetricsFile.bind(
            result.budget_observation_sidecar
        )
        evidence = tuple(
            sorted(
                (BoundRuntimeMetricsFile.bind(path) for path in result.evidence_files),
                key=lambda value: value.path,
            )
        )
        value = FreshProcessExecutionMetricsSource(
            run_id=result.run_id,
            execution_plan_sha256=result.execution_plan_sha256,
            experiment_budget_sha256=result.experiment_budget_sha256,
            rank_config_sha256=result.rank_config_sha256,
            topology_sha256=result.topology_sha256,
            terminal_receipt_sha256=result.terminal_receipt_sha256,
            budget_observation_sha256=result.budget_observation_sha256,
            terminal_receipt=terminal,
            budget_observation=observation,
            budget_observation_sidecar=observation_sidecar,
            evidence_files=evidence,
            run_evidence=BoundRuntimeMetricsFile.bind(completed["run"]),
            performance_evidence=BoundRuntimeMetricsFile.bind(completed["performance"]),
            native_terminal=native,
        )
        _reopen_fresh_execution(value)
        executions.append(value)
    source = FreshProcessRuntimeMetricsSource(
        session_plan_sha256=block_result.session_plan_sha256,
        block_result_sha256=block_result.sha256,
        execution_mode=block_result.execution_mode,
        fallback_reason=block_result.fallback_reason,
        executions=tuple(executions),
    )
    return source


def bind_fresh_process_runtime_metrics_from_terminal_receipts(
    *,
    session_plan_sha256: str,
    terminal_receipt_paths: tuple[str | Path, ...],
) -> FreshProcessRuntimeMetricsSource:
    """Bind a fresh-process block solely from first-party raw artifact paths.

    The caller supplies no serialized observations or execution-result object.
    Run, topology, budget, native-terminal, and table identities are derived
    onsite from the terminal receipt and completed Parquet evidence, then the
    ordinary raw-source reopener validates the resulting cross-file contract.
    ``session_plan_sha256`` is the grouping identity only; it cannot contribute
    a metric value.
    """

    _require_sha256("session plan digest", session_plan_sha256)
    if not terminal_receipt_paths:
        raise ValueError("fresh-process path binding requires terminal receipts")
    executions: list[FreshProcessExecutionMetricsSource] = []
    canonical_terminal_paths: list[str] = []
    terminal_suffix = ".rank0.complete.json"
    for requested_path in terminal_receipt_paths:
        terminal = BoundRuntimeMetricsFile.bind(requested_path)
        canonical_terminal_paths.append(terminal.path)
        if not Path(terminal.path).name.endswith(terminal_suffix):
            raise ValueError("fresh-process terminal receipt path is not canonical")
        run_id = Path(terminal.path).name[: -len(terminal_suffix)]
        _require_text("fresh-process run_id", run_id)
        root = Path(terminal.path).parent
        terminal_body = terminal.read_bytes(label="fresh-process terminal receipt")
        terminal_value = _strict_json(
            terminal_body,
            label="fresh-process terminal receipt",
        )
        if terminal_body != _canonical_json_bytes(terminal_value, ensure_ascii=True):
            raise ValueError("fresh-process terminal receipt is not canonical JSON")
        completed = load_completed_evidence(root, run_id=run_id, rank=0)
        if completed is None:
            raise RuntimeError("fresh-process runtime metric source is not complete")
        if "run" not in completed or "performance" not in completed:
            raise RuntimeError("fresh-process evidence omits a required table")
        try:
            run_rows = pq.read_table(
                completed["run"],
                columns=[
                    "run_id",
                    "runtime_sha256",
                    "rank_config_sha256",
                    "topology_sha256",
                    "experiment_budget_sha256",
                ],
            ).to_pylist()
        except (KeyError, pa.ArrowException) as error:
            raise RuntimeError("fresh-process run table is malformed") from error
        if len(run_rows) != 1 or run_rows[0].get("run_id") != run_id:
            raise RuntimeError("fresh-process run table identity differs")
        run_row = run_rows[0]
        execution_plan_sha256 = _require_sha256(
            "fresh-process execution plan",
            run_row.get("runtime_sha256"),
        )
        experiment_budget_sha256 = _require_sha256(
            "fresh-process experiment budget",
            run_row.get("experiment_budget_sha256"),
        )
        rank_config_sha256 = _require_sha256(
            "fresh-process rank config",
            run_row.get("rank_config_sha256"),
        )
        topology_sha256 = _require_sha256(
            "fresh-process topology",
            run_row.get("topology_sha256"),
        )
        native_binding = terminal_value.get("native_terminal_artifact")
        if (
            type(native_binding) is not dict
            or type(native_binding.get("path")) is not str
        ):
            raise RuntimeError("fresh-process terminal lacks native evidence")
        native = bind_native_runtime_metrics(root / str(native_binding["path"]))
        observation_path = (
            root / f"{run_id}.rank0.budget-observation" / "observation.json"
        )
        observation = BoundRuntimeMetricsFile.bind(observation_path)
        observation_sidecar = BoundRuntimeMetricsFile.bind(
            observation_path.with_name("observation.json.sha256")
        )
        observation_body = observation.read_bytes(
            label="fresh-process budget observation"
        )
        observation_value = _strict_json(
            observation_body,
            label="fresh-process budget observation",
        )
        if observation_body != _canonical_json_bytes(
            observation_value,
            ensure_ascii=True,
        ):
            raise ValueError("fresh-process budget observation is not canonical JSON")
        budget_observation_sha256 = _require_sha256(
            "fresh-process budget observation",
            observation_value.get("budget_observation_sha256"),
        )
        if (
            observation_value.get("experiment_budget_sha256")
            != experiment_budget_sha256
        ):
            raise RuntimeError("fresh-process budget observation names another budget")
        evidence_paths = tuple(
            sorted(
                {
                    *(str(Path(path).resolve()) for path in completed.values()),
                    native.artifact.path,
                    observation.path,
                    observation_sidecar.path,
                }
            )
        )
        source = FreshProcessExecutionMetricsSource(
            run_id=run_id,
            execution_plan_sha256=execution_plan_sha256,
            experiment_budget_sha256=experiment_budget_sha256,
            rank_config_sha256=rank_config_sha256,
            topology_sha256=topology_sha256,
            terminal_receipt_sha256=terminal.raw_sha256,
            budget_observation_sha256=budget_observation_sha256,
            terminal_receipt=terminal,
            budget_observation=observation,
            budget_observation_sidecar=observation_sidecar,
            evidence_files=tuple(
                BoundRuntimeMetricsFile.bind(path) for path in evidence_paths
            ),
            run_evidence=BoundRuntimeMetricsFile.bind(completed["run"]),
            performance_evidence=BoundRuntimeMetricsFile.bind(completed["performance"]),
            native_terminal=native,
        )
        _reopen_fresh_execution(source)
        executions.append(source)
    if canonical_terminal_paths != sorted(set(canonical_terminal_paths)):
        raise ValueError(
            "fresh-process terminal paths must be sorted, unique, and canonical"
        )
    execution_tuple = tuple(executions)
    source = FreshProcessRuntimeMetricsSource(
        session_plan_sha256=session_plan_sha256,
        block_result_sha256=_fresh_process_block_result_sha256(
            session_plan_sha256=session_plan_sha256,
            executions=execution_tuple,
        ),
        execution_mode=SHARED_SESSION_FALLBACK_MODE,
        fallback_reason=SHARED_SESSION_UNAVAILABLE_REASON,
        executions=execution_tuple,
    )
    source.__post_init__()
    return source


def build_runtime_metrics_authority(
    *,
    compile_sources: tuple[CompileRuntimeMetricsSource, ...] = (),
    fresh_process_sources: tuple[FreshProcessRuntimeMetricsSource, ...] = (),
    native_sources: tuple[NativeRuntimeMetricsSource, ...] = (),
) -> RuntimeMetricsAuthority:
    return RuntimeMetricsAuthority(
        schema_version=1,
        reducer_protocol_sha256=RUNTIME_METRICS_REDUCER_PROTOCOL_SHA256,
        compile_sources=tuple(sorted(compile_sources, key=lambda value: value.sha256)),
        fresh_process_sources=tuple(
            sorted(fresh_process_sources, key=lambda value: value.sha256)
        ),
        native_sources=tuple(sorted(native_sources, key=lambda value: value.sha256)),
    )


def _observation(
    *,
    subject_id: str,
    metric: RuntimeMetricName,
    status: RuntimeMetricStatus,
    value: float | None,
    source_kind: RuntimeMetricSourceKind,
    source_sha256: str,
    source_field: str,
    reason_code: str | None = None,
    release_trusted_attestation: bool | None = None,
) -> RuntimeMetricObservation:
    return RuntimeMetricObservation(
        subject_id=subject_id,
        metric=metric,
        unit=_METRIC_SPECS[metric].unit,
        status=status,
        value=value,
        source_kind=source_kind,
        source_sha256=source_sha256,
        source_field=source_field,
        reason_code=reason_code,
        release_trusted_attestation=release_trusted_attestation,
    )


def _reduce_compile_source(
    source: CompileRuntimeMetricsSource,
) -> list[RuntimeMetricObservation]:
    plan, _, receipt = _reopen_compile_source(source)
    hit = int(plan.cache_mode == "reuse")
    miss = int(plan.cache_mode == "build")
    return [
        _observation(
            subject_id=source.subject_id,
            metric=RuntimeMetricName.COMPILE_CACHE_HITS,
            status=RuntimeMetricStatus.OBSERVED,
            value=hit,
            source_kind=RuntimeMetricSourceKind.COMPILE_CACHE,
            source_sha256=source.sha256,
            source_field="compile_cache_launch_plan.cache_mode",
        ),
        _observation(
            subject_id=source.subject_id,
            metric=RuntimeMetricName.COMPILE_CACHE_MISSES,
            status=RuntimeMetricStatus.OBSERVED,
            value=miss,
            source_kind=RuntimeMetricSourceKind.COMPILE_CACHE,
            source_sha256=source.sha256,
            source_field="compile_cache_launch_plan.cache_mode",
        ),
        _observation(
            subject_id=source.subject_id,
            metric=RuntimeMetricName.JIT_DURATION_MS,
            status=RuntimeMetricStatus.MEASURED,
            value=receipt.jit_duration_ns / 1_000_000,
            source_kind=RuntimeMetricSourceKind.COMPILE_CACHE,
            source_sha256=source.sha256,
            source_field="compile_cache_receipt.jit_duration_ns",
        ),
    ]


def _unavailable_native_observations(
    source: NativeRuntimeMetricsSource,
) -> list[RuntimeMetricObservation]:
    reasons = {
        RuntimeMetricName.GRAPH_CAPTURE_MS: (
            "native_terminal_does_not_report_graph_capture_duration"
        ),
        RuntimeMetricName.GRAPH_REPLAY_COUNT: (
            "native_terminal_does_not_report_graph_replay_count"
        ),
        RuntimeMetricName.NVML_PROCESS_HBM_BYTES: "nvml_receipt_unavailable",
        RuntimeMetricName.NVML_GLOBAL_HBM_BYTES: "nvml_receipt_unavailable",
        RuntimeMetricName.EXECUTED_FLOPS: "independent_profiler_receipt_unavailable",
        RuntimeMetricName.COMMITTED_USEFUL_FLOPS: (
            "independent_profiler_receipt_unavailable"
        ),
        RuntimeMetricName.PRECISION_NORMALIZED_EXECUTED_MFU: (
            "independent_profiler_receipt_unavailable"
        ),
        RuntimeMetricName.TARGET_EQUIVALENT_USEFUL_UTILIZATION: (
            "independent_profiler_receipt_unavailable"
        ),
        RuntimeMetricName.EXECUTED_HBM_BYTES: (
            "independent_profiler_receipt_unavailable"
        ),
        RuntimeMetricName.EXECUTED_FLOPS_PER_COMMITTED_TOKEN: (
            "independent_profiler_receipt_unavailable"
        ),
        RuntimeMetricName.HBM_BYTES_PER_COMMITTED_TOKEN: (
            "independent_profiler_receipt_unavailable"
        ),
        RuntimeMetricName.POWER_WATTS: "power_sampler_receipt_unavailable",
        RuntimeMetricName.ENERGY_JOULES: "power_sampler_receipt_unavailable",
        RuntimeMetricName.OUTPUT_TOKENS_PER_JOULE: (
            "power_sampler_receipt_unavailable"
        ),
    }
    return [
        _observation(
            subject_id=source.subject_id,
            metric=metric,
            status=RuntimeMetricStatus.UNRESOLVED,
            value=None,
            source_kind=RuntimeMetricSourceKind.NATIVE_TERMINAL,
            source_sha256=source.sha256,
            source_field="missing_first_party_runtime_receipt",
            reason_code=reason,
            release_trusted_attestation=source.release_trusted_attestation,
        )
        for metric, reason in reasons.items()
    ]


def _reduce_native_source(
    source: NativeRuntimeMetricsSource,
    *,
    performance: dict[str, object] | None = None,
) -> list[RuntimeMetricObservation]:
    if performance is None:
        _, performance = _reopen_native_source(source)
    graph_value = performance.get("graph_replay_hit_rate")
    if graph_value is None:
        graph = _observation(
            subject_id=source.subject_id,
            metric=RuntimeMetricName.GRAPH_REPLAY_HIT_RATE,
            status=RuntimeMetricStatus.UNRESOLVED,
            value=None,
            source_kind=RuntimeMetricSourceKind.NATIVE_TERMINAL,
            source_sha256=source.sha256,
            source_field="terminal.performance_counters.graph_replay_hit_rate",
            reason_code="native_graph_replay_hit_rate_not_reported",
            release_trusted_attestation=source.release_trusted_attestation,
        )
    else:
        if (
            not isinstance(graph_value, (int, float))
            or isinstance(graph_value, bool)
            or not math.isfinite(float(graph_value))
            or not 0 <= float(graph_value) <= 1
        ):
            raise ValueError("native graph replay hit rate is outside [0, 1]")
        graph = _observation(
            subject_id=source.subject_id,
            metric=RuntimeMetricName.GRAPH_REPLAY_HIT_RATE,
            status=RuntimeMetricStatus.OBSERVED,
            value=float(graph_value),
            source_kind=RuntimeMetricSourceKind.NATIVE_TERMINAL,
            source_sha256=source.sha256,
            source_field="terminal.performance_counters.graph_replay_hit_rate",
            release_trusted_attestation=source.release_trusted_attestation,
        )
    return [graph, *_unavailable_native_observations(source)]


def _performance_counter_observation(
    *,
    source: FreshProcessExecutionMetricsSource,
    performance: dict[str, object],
    metric: RuntimeMetricName,
) -> RuntimeMetricObservation:
    value = performance[metric.value]
    if value is None:
        return _observation(
            subject_id=source.run_id,
            metric=metric,
            status=RuntimeMetricStatus.UNRESOLVED,
            value=None,
            source_kind=RuntimeMetricSourceKind.COMPLETED_PERFORMANCE,
            source_sha256=source.sha256,
            source_field=f"performance.{metric.value}",
            reason_code=f"performance_receipt_does_not_report_{metric.value}",
        )
    if type(value) is not int or value < 0:
        raise ValueError(f"completed performance {metric.value} must be non-negative")
    return _observation(
        subject_id=source.run_id,
        metric=metric,
        status=RuntimeMetricStatus.OBSERVED,
        value=value,
        source_kind=RuntimeMetricSourceKind.COMPLETED_PERFORMANCE,
        source_sha256=source.sha256,
        source_field=f"performance.{metric.value}",
    )


def _reduce_fresh_execution(
    source: FreshProcessExecutionMetricsSource,
) -> list[RuntimeMetricObservation]:
    components, performance, native_performance = _reopen_fresh_execution(source)
    observations = [
        _observation(
            subject_id=source.run_id,
            metric=RuntimeMetricName.COLD_START_MS,
            status=RuntimeMetricStatus.MEASURED,
            value=float(components["startup_model_load"]),
            source_kind=RuntimeMetricSourceKind.FRESH_PROCESS_BUDGET,
            source_sha256=source.sha256,
            source_field="budget_observation.startup_model_load",
        ),
        _observation(
            subject_id=source.run_id,
            metric=RuntimeMetricName.FRESH_PROCESS_RESET_FINALIZATION_MS,
            status=RuntimeMetricStatus.MEASURED,
            value=float(components["reset_finalization"]),
            source_kind=RuntimeMetricSourceKind.FRESH_PROCESS_BUDGET,
            source_sha256=source.sha256,
            source_field="budget_observation.reset_finalization",
        ),
        _observation(
            subject_id=source.run_id,
            metric=RuntimeMetricName.RESET_DURATION_MS,
            status=RuntimeMetricStatus.NOT_APPLICABLE,
            value=None,
            source_kind=RuntimeMetricSourceKind.FRESH_PROCESS_BUDGET,
            source_sha256=source.sha256,
            source_field="industrial_server_block_result.execution_mode",
            reason_code="fresh_process_has_no_shared_reset",
        ),
        _observation(
            subject_id=source.run_id,
            metric=RuntimeMetricName.REUSED_SESSION_STARTUP_SAVINGS_MS,
            status=RuntimeMetricStatus.NOT_APPLICABLE,
            value=None,
            source_kind=RuntimeMetricSourceKind.FRESH_PROCESS_BUDGET,
            source_sha256=source.sha256,
            source_field="industrial_server_block_result.execution_mode",
            reason_code="fresh_process_has_no_reuse_baseline",
        ),
        _performance_counter_observation(
            source=source,
            performance=performance,
            metric=RuntimeMetricName.HTTP_CONNECTIONS_CREATED,
        ),
        _performance_counter_observation(
            source=source,
            performance=performance,
            metric=RuntimeMetricName.HTTP_REUSED_REQUESTS,
        ),
    ]
    observations.extend(
        _reduce_native_source(source.native_terminal, performance=native_performance)
    )
    return observations


def _validate_exact_metric_coverage(
    authority: RuntimeMetricsAuthority,
    observations: tuple[RuntimeMetricObservation, ...],
) -> None:
    by_subject: dict[str, set[RuntimeMetricName]] = {}
    for row in observations:
        by_subject.setdefault(row.subject_id, set()).add(row.metric)
    for source in authority.compile_sources:
        if not _COMPILE_METRICS <= by_subject.get(source.subject_id, set()):
            raise RuntimeError("compile runtime metric coverage is incomplete")
    for block in authority.fresh_process_sources:
        for source in block.executions:
            expected = _BUDGET_METRICS | _PERFORMANCE_METRICS | _NATIVE_METRICS
            if not expected <= by_subject.get(source.run_id, set()):
                raise RuntimeError(
                    "fresh-process runtime metric coverage is incomplete"
                )
    for source in authority.native_sources:
        if not _NATIVE_METRICS <= by_subject.get(source.subject_id, set()):
            raise RuntimeError("native runtime metric coverage is incomplete")


def reduce_runtime_metrics(
    authority: RuntimeMetricsAuthority,
) -> RuntimeMetricsReduction:
    """Replay every bound raw source and emit canonical typed observations."""

    if type(authority) is not RuntimeMetricsAuthority:
        raise TypeError("runtime metrics reducer requires an exact authority")
    authority.__post_init__()
    rows: list[RuntimeMetricObservation] = []
    for source in authority.compile_sources:
        rows.extend(_reduce_compile_source(source))
    for block in authority.fresh_process_sources:
        for source in block.executions:
            rows.extend(_reduce_fresh_execution(source))
    for source in authority.native_sources:
        rows.extend(_reduce_native_source(source))
    observations = tuple(sorted(rows, key=lambda value: value.key))
    _validate_exact_metric_coverage(authority, observations)
    reduction = RuntimeMetricsReduction(
        schema_version=1,
        authority_sha256=authority.sha256,
        reducer_protocol_sha256=RUNTIME_METRICS_REDUCER_PROTOCOL_SHA256,
        source_sha256s=authority.source_sha256s,
        observations=observations,
    )
    reduction.validate_against(authority)
    return reduction


def export_formal_runtime_metrics(
    authority: RuntimeMetricsAuthority | None,
    *,
    expected_run_ids: tuple[str, ...],
) -> FormalRuntimeMetricsExport:
    """Build the only runtime-metric view eligible for formal analysis.

    Every supplied authority is replayed.  Resolved values are retained only
    for an exact expected run whose native terminal is release trusted.  Other
    resolved values are downgraded to ``UNRESOLVED`` with ``None``; existing
    ``UNRESOLVED`` and ``N/A`` states remain unchanged.
    """

    expected = tuple(sorted(expected_run_ids))
    if not expected or expected != tuple(sorted(set(expected))):
        raise ValueError("formal runtime expected runs must be non-empty and unique")
    for run_id in expected:
        _require_text("formal runtime expected run", run_id)
    reduction: RuntimeMetricsReduction | None = None
    trusted_runs: set[str] = set()
    observations_by_key: dict[
        tuple[str, RuntimeMetricName], RuntimeMetricObservation
    ] = {}
    if authority is not None:
        if type(authority) is not RuntimeMetricsAuthority:
            raise TypeError("formal runtime export requires an exact authority")
        reduction = reduce_runtime_metrics(authority)
        execution_subjects = {
            execution.run_id
            for block in authority.fresh_process_sources
            for execution in block.executions
        } | {source.subject_id for source in authority.native_sources}
        foreign = execution_subjects - set(expected)
        if foreign:
            raise ValueError("runtime metrics authority contains foreign formal runs")
        trusted_runs.update(
            execution.run_id
            for block in authority.fresh_process_sources
            for execution in block.executions
            if execution.native_terminal.release_trusted_attestation
        )
        trusted_runs.update(
            source.subject_id
            for source in authority.native_sources
            if source.release_trusted_attestation
        )
        observations_by_key = {
            (row.subject_id, row.metric): row
            for row in reduction.observations
            if row.subject_id in expected and row.metric in _FORMAL_RUN_METRICS
        }

    formal_rows: list[FormalRuntimeMetricObservation] = []
    for run_id in expected:
        release_trusted = run_id in trusted_runs
        for metric in sorted(_FORMAL_RUN_METRICS, key=lambda value: value.value):
            source = observations_by_key.get((run_id, metric))
            if source is None:
                formal_rows.append(
                    FormalRuntimeMetricObservation(
                        subject_id=run_id,
                        metric=metric,
                        unit=_METRIC_SPECS[metric].unit,
                        status=RuntimeMetricStatus.UNRESOLVED,
                        value=None,
                        source_kind=None,
                        source_sha256=None,
                        reason_code=(
                            "runtime_metrics_authority_unavailable"
                            if authority is None
                            else "runtime_metric_run_source_unavailable"
                        ),
                        release_trusted=False,
                    )
                )
                continue
            resolved = source.status in {
                RuntimeMetricStatus.OBSERVED,
                RuntimeMetricStatus.MEASURED,
            }
            if resolved and not release_trusted:
                formal_rows.append(
                    FormalRuntimeMetricObservation(
                        subject_id=run_id,
                        metric=metric,
                        unit=source.unit,
                        status=RuntimeMetricStatus.UNRESOLVED,
                        value=None,
                        source_kind=source.source_kind,
                        source_sha256=source.source_sha256,
                        reason_code="release_trusted_runtime_source_required",
                        release_trusted=False,
                    )
                )
                continue
            formal_rows.append(
                FormalRuntimeMetricObservation(
                    subject_id=run_id,
                    metric=metric,
                    unit=source.unit,
                    status=source.status,
                    value=source.value,
                    source_kind=source.source_kind,
                    source_sha256=source.source_sha256,
                    reason_code=source.reason_code,
                    release_trusted=release_trusted,
                )
            )
    rows = tuple(sorted(formal_rows, key=lambda value: value.key))
    status = (
        RuntimeMetricStatus.UNRESOLVED
        if any(row.status is RuntimeMetricStatus.UNRESOLVED for row in rows)
        else RuntimeMetricStatus.OBSERVED
    )
    return FormalRuntimeMetricsExport(
        schema_version=1,
        protocol_sha256=FORMAL_RUNTIME_METRICS_EXPORT_PROTOCOL_SHA256,
        status=status,
        authority_sha256=None if authority is None else authority.sha256,
        reduction_sha256=None if reduction is None else reduction.sha256,
        source_sha256s=() if reduction is None else reduction.source_sha256s,
        expected_run_ids=expected,
        observations=rows,
    )


__all__ = [
    "FORMAL_RUNTIME_METRICS_EXPORT_PROTOCOL_SHA256",
    "RUNTIME_METRICS_REDUCER_PROTOCOL_SHA256",
    "BoundRuntimeMetricsFile",
    "CompileRuntimeMetricsSource",
    "FormalRuntimeMetricObservation",
    "FormalRuntimeMetricsExport",
    "FreshProcessExecutionMetricsSource",
    "FreshProcessRuntimeMetricsSource",
    "NativeRuntimeMetricsSource",
    "RuntimeMetricName",
    "RuntimeMetricObservation",
    "RuntimeMetricSourceKind",
    "RuntimeMetricStatus",
    "RuntimeMetricUnit",
    "RuntimeMetricsAuthority",
    "RuntimeMetricsReduction",
    "bind_compile_runtime_metrics",
    "bind_fresh_process_runtime_metrics",
    "bind_fresh_process_runtime_metrics_from_terminal_receipts",
    "bind_native_runtime_metrics",
    "build_runtime_metrics_authority",
    "export_formal_runtime_metrics",
    "reduce_runtime_metrics",
]
