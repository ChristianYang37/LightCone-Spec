from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from test_compile_cache_launch import _build_base, _key
from test_industrial_executor import _execution_fixture, _FakeHandle, _FakeTransport

from lightcone_spec.experiments.runtime_metrics import (
    FormalRuntimeMetricObservation,
    RuntimeMetricName,
    RuntimeMetricObservation,
    RuntimeMetricSourceKind,
    RuntimeMetricStatus,
    RuntimeMetricUnit,
    bind_compile_runtime_metrics,
    bind_fresh_process_runtime_metrics,
    build_runtime_metrics_authority,
    export_formal_runtime_metrics,
    reduce_runtime_metrics,
)
from lightcone_spec.orchestration.executor import execute_industrial_plan
from lightcone_spec.orchestration.native_terminal import NativeTerminalProvider
from lightcone_spec.orchestration.session import (
    SHARED_SESSION_FALLBACK_MODE,
    SHARED_SESSION_UNAVAILABLE_REASON,
    IndustrialServerBlockResult,
)
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    CompileCacheReceipt,
    start_compile_cache_launch,
)


def _compile_source(tmp_path: Path, *, mode: str, label: str):
    root = tmp_path / f"cache-{label}"
    key = _key()
    base_receipt: Path | None = None
    if mode == "reuse":
        key, _, base_receipt = _build_base(root, key=key, payload=b"base-kernel")
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=root,
        cache_mode=mode,
        base_receipt_path=base_receipt,
    )
    plan_path = plan.write(tmp_path / f"{label}-plan.json")
    session = start_compile_cache_launch(
        plan,
        process_id=os.getpid(),
        attempt_id=f"{label}-attempt",
    )
    environment = session.environment({})
    if mode == "build":
        (Path(environment["TRITON_CACHE_DIR"]) / "kernel.bin").write_bytes(
            b"compiled-kernel"
        )
    _, receipt_path, attempt_path = session.complete()
    return bind_compile_runtime_metrics(
        plan_path=plan_path,
        attempt_path=attempt_path,
        result_receipt_path=receipt_path,
        subject_id=f"{label}-subject",
    )


@pytest.mark.parametrize(
    ("mode", "hits", "misses"),
    (("build", 0, 1), ("reuse", 1, 0)),
)
def test_compile_metrics_replay_first_party_plan_attempt_and_receipt(
    tmp_path: Path,
    mode: str,
    hits: int,
    misses: int,
) -> None:
    source = _compile_source(tmp_path, mode=mode, label=mode)
    reduction = reduce_runtime_metrics(
        build_runtime_metrics_authority(compile_sources=(source,))
    )
    hit = reduction.observation(source.subject_id, RuntimeMetricName.COMPILE_CACHE_HITS)
    miss = reduction.observation(
        source.subject_id, RuntimeMetricName.COMPILE_CACHE_MISSES
    )
    jit = reduction.observation(source.subject_id, RuntimeMetricName.JIT_DURATION_MS)
    receipt = CompileCacheReceipt.load(source.result_receipt.path)

    assert (hit.status, hit.value) == (RuntimeMetricStatus.OBSERVED, hits)
    assert (miss.status, miss.value) == (RuntimeMetricStatus.OBSERVED, misses)
    assert jit.status is RuntimeMetricStatus.MEASURED
    assert jit.value == receipt.jit_duration_ns / 1_000_000
    assert reduction.resolved_performance_overrides(source.subject_id) == {
        "compile_cache_hits": hits,
        "compile_cache_misses": misses,
        "jit_duration_ms": receipt.jit_duration_ns / 1_000_000,
    }


def _fresh_process_source(tmp_path: Path):
    fixture = _execution_fixture(tmp_path, request_count=2)
    plan = fixture.plan
    transport = _FakeTransport(plan=plan)
    provider = NativeTerminalProvider(transport)

    async def launch(_server):
        return _FakeHandle()

    result = asyncio.run(
        execute_industrial_plan(
            plan,
            output_root=Path(plan.runtime_plan.cell.resources.evidence_root),
            run_nonce_sha256="7" * 64,
            launch_server=launch,
            transport=transport,
            native_evidence=provider,
        )
    )
    block = IndustrialServerBlockResult(
        session_plan_sha256="8" * 64,
        execution_mode=SHARED_SESSION_FALLBACK_MODE,
        fallback_reason=SHARED_SESSION_UNAVAILABLE_REASON,
        executions=(result,),
    )
    return bind_fresh_process_runtime_metrics(block), result


def test_fresh_process_metrics_keep_measured_na_and_unresolved_distinct(
    tmp_path: Path,
) -> None:
    source, result = _fresh_process_source(tmp_path)
    reduction = reduce_runtime_metrics(
        build_runtime_metrics_authority(fresh_process_sources=(source,))
    )
    artifact = json.loads(Path(result.budget_observation).read_text(encoding="utf-8"))
    components = dict(artifact["observed_component_ms"])
    performance_path = source.executions[0].performance_evidence.path
    performance = pq.read_table(performance_path).to_pylist()[0]

    cold = reduction.observation(result.run_id, RuntimeMetricName.COLD_START_MS)
    reset_finalization = reduction.observation(
        result.run_id,
        RuntimeMetricName.FRESH_PROCESS_RESET_FINALIZATION_MS,
    )
    reset = reduction.observation(result.run_id, RuntimeMetricName.RESET_DURATION_MS)
    savings = reduction.observation(
        result.run_id,
        RuntimeMetricName.REUSED_SESSION_STARTUP_SAVINGS_MS,
    )
    graph = reduction.observation(
        result.run_id, RuntimeMetricName.GRAPH_REPLAY_HIT_RATE
    )
    capture = reduction.observation(result.run_id, RuntimeMetricName.GRAPH_CAPTURE_MS)
    nvml = reduction.observation(
        result.run_id, RuntimeMetricName.NVML_PROCESS_HBM_BYTES
    )
    energy = reduction.observation(result.run_id, RuntimeMetricName.ENERGY_JOULES)

    assert (cold.status, cold.value) == (
        RuntimeMetricStatus.MEASURED,
        float(components["startup_model_load"]),
    )
    assert (reset_finalization.status, reset_finalization.value) == (
        RuntimeMetricStatus.MEASURED,
        float(components["reset_finalization"]),
    )
    assert (reset.status, reset.value, reset.reason_code) == (
        RuntimeMetricStatus.NOT_APPLICABLE,
        None,
        "fresh_process_has_no_shared_reset",
    )
    assert (savings.status, savings.value, savings.reason_code) == (
        RuntimeMetricStatus.NOT_APPLICABLE,
        None,
        "fresh_process_has_no_reuse_baseline",
    )
    assert (graph.status, graph.value) == (RuntimeMetricStatus.OBSERVED, 1.0)
    assert graph.release_trusted_attestation is False
    assert capture.status is RuntimeMetricStatus.UNRESOLVED
    assert capture.value is None
    assert (nvml.status, nvml.value, nvml.reason_code) == (
        RuntimeMetricStatus.UNRESOLVED,
        None,
        "nvml_receipt_unavailable",
    )
    assert (energy.status, energy.value, energy.reason_code) == (
        RuntimeMetricStatus.UNRESOLVED,
        None,
        "power_sampler_receipt_unavailable",
    )
    overrides = reduction.resolved_performance_overrides(result.run_id)
    assert overrides["cold_start_ms"] == float(components["startup_model_load"])
    assert "graph_replay_hit_rate" not in overrides
    assert (
        overrides["http_connections_created"] == performance["http_connections_created"]
    )
    assert overrides["http_reused_requests"] == performance["http_reused_requests"]
    assert "reset_duration_ms" not in overrides
    assert "reused_session_startup_savings_ms" not in overrides
    assert "energy_joules" not in overrides
    assert "fresh_process_reset_finalization_ms" not in overrides


def test_native_only_reduction_retains_untrusted_release_state(
    tmp_path: Path,
) -> None:
    source, result = _fresh_process_source(tmp_path)
    native = source.executions[0].native_terminal
    reduction = reduce_runtime_metrics(
        build_runtime_metrics_authority(native_sources=(native,))
    )

    assert all(row.subject_id == result.run_id for row in reduction.observations)
    assert {row.release_trusted_attestation for row in reduction.observations} == {
        False
    }
    assert (
        reduction.observation(
            result.run_id, RuntimeMetricName.GRAPH_REPLAY_HIT_RATE
        ).value
        == 1.0
    )
    assert (
        reduction.observation(
            result.run_id, RuntimeMetricName.EXECUTED_FLOPS
        ).reason_code
        == "independent_profiler_receipt_unavailable"
    )


def test_reducer_reopens_bound_raw_files_instead_of_trusting_summary(
    tmp_path: Path,
) -> None:
    source = _compile_source(tmp_path, mode="build", label="tamper")
    authority = build_runtime_metrics_authority(compile_sources=(source,))
    plan_path = Path(source.plan.path)
    plan_path.write_bytes(plan_path.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="bound raw bytes"):
        reduce_runtime_metrics(authority)


def test_unavailable_metric_cannot_carry_a_fabricated_zero() -> None:
    with pytest.raises(ValueError, match="cannot carry a value"):
        RuntimeMetricObservation(
            subject_id="run",
            metric=RuntimeMetricName.ENERGY_JOULES,
            unit=RuntimeMetricUnit.JOULE,
            status=RuntimeMetricStatus.UNRESOLVED,
            value=0,
            source_kind=RuntimeMetricSourceKind.NATIVE_TERMINAL,
            source_sha256="a" * 64,
            source_field="missing_first_party_runtime_receipt",
            reason_code="power_sampler_receipt_unavailable",
            release_trusted_attestation=False,
        )


def test_formal_export_without_authority_is_exact_null_coverage() -> None:
    export = export_formal_runtime_metrics(None, expected_run_ids=("run-b", "run-a"))

    assert export.expected_run_ids == ("run-a", "run-b")
    assert export.status is RuntimeMetricStatus.UNRESOLVED
    assert export.authority_sha256 is None
    assert export.reduction_sha256 is None
    assert export.source_sha256s == ()
    assert export.observations
    assert all(
        row.status is RuntimeMetricStatus.UNRESOLVED for row in export.observations
    )
    assert all(row.value is None for row in export.observations)
    assert all(
        row.reason_code == "runtime_metrics_authority_unavailable"
        for row in export.observations
    )
    assert export.formal_values("run-a") == {}


def test_formal_export_downgrades_untrusted_resolved_values(tmp_path: Path) -> None:
    source, result = _fresh_process_source(tmp_path)
    export = export_formal_runtime_metrics(
        build_runtime_metrics_authority(fresh_process_sources=(source,)),
        expected_run_ids=(result.run_id,),
    )

    cold = export.observation(result.run_id, RuntimeMetricName.COLD_START_MS)
    graph = export.observation(result.run_id, RuntimeMetricName.GRAPH_REPLAY_HIT_RATE)
    reset = export.observation(result.run_id, RuntimeMetricName.RESET_DURATION_MS)
    energy = export.observation(result.run_id, RuntimeMetricName.ENERGY_JOULES)
    assert (cold.status, cold.value, cold.reason_code) == (
        RuntimeMetricStatus.UNRESOLVED,
        None,
        "release_trusted_runtime_source_required",
    )
    assert (graph.status, graph.value, graph.reason_code) == (
        RuntimeMetricStatus.UNRESOLVED,
        None,
        "release_trusted_runtime_source_required",
    )
    assert (reset.status, reset.value, reset.reason_code) == (
        RuntimeMetricStatus.NOT_APPLICABLE,
        None,
        "fresh_process_has_no_shared_reset",
    )
    assert (energy.status, energy.value) == (RuntimeMetricStatus.UNRESOLVED, None)
    assert export.formal_values(result.run_id) == {}

    with pytest.raises(ValueError, match="foreign formal runs"):
        export_formal_runtime_metrics(
            build_runtime_metrics_authority(fresh_process_sources=(source,)),
            expected_run_ids=("another-run",),
        )


def test_formal_observation_rejects_untrusted_resolved_value() -> None:
    with pytest.raises(ValueError, match="untrusted runtime source"):
        FormalRuntimeMetricObservation(
            subject_id="run",
            metric=RuntimeMetricName.ENERGY_JOULES,
            unit=RuntimeMetricUnit.JOULE,
            status=RuntimeMetricStatus.MEASURED,
            value=0.0,
            source_kind=RuntimeMetricSourceKind.NATIVE_TERMINAL,
            source_sha256="a" * 64,
            reason_code=None,
            release_trusted=False,
        )


def test_authority_rejects_same_subject_source_collision(tmp_path: Path) -> None:
    first = _compile_source(tmp_path, mode="build", label="collision-first")
    second = _compile_source(tmp_path, mode="build", label="collision-second")

    with pytest.raises(ValueError, match="duplicates a compile subject"):
        build_runtime_metrics_authority(
            compile_sources=(
                first,
                dataclasses.replace(second, subject_id=first.subject_id),
            )
        )
