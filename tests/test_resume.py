import json
import signal
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.protocol import E0_ONLINESPEC_RECIPES, Job, materialize
from lightcone_spec.runner import (
    ScientificFailure,
    _cleanup_interrupted_servers,
    _complete_blocked_profiler,
    _e1a_source_transfer_jobs,
    _e2_keep_count,
    _e2_missing_dependency_jobs,
    _e5_reference,
    _eta_resource_count,
    _exclude_redundant_e2_dependency_jobs,
    _execution_allocations,
    _ncu_permission_block_reason,
    _records_scientific_rejection,
    _reduce_node,
    _reopen_soft_gate_e3b,
    _repair_completed_s10_downstream_resume,
    _repair_e0_e6_partial_resume_v1,
    _repair_e3b_scientific_rejections,
    _repair_metric_dedup_e5_resume_v1,
    _requeue_dspark_dynamic_batch_budget_failures,
    _restore_soft_gate_width_selection,
    _resume_materialization,
    _run_e1a_latency_protocol_repair_v3,
    _run_node_jobs,
    _run_pending_jobs,
    _run_priority_window_v2,
    _run_tp1_interference_v2,
    _save_or_validate_run_config,
    _seed_e3a_static_deployment_width,
    _segment_jobs,
    _select_e0_recipes,
    _selection_for_job,
    _session_order,
    _set_e2_expected_evidence,
    _skip_satisfied_e2_dependency_jobs,
    _soft_gate_width_replacements,
    _tp1_interference_v2_jobs,
    _upgrade_legacy_e0_materialization,
)
from lightcone_spec.state import StateStore


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="paper-v2-test",
        sglang_root=tmp_path / "sglang",
        results_root=tmp_path,
        models={},
        drafts={},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python"),
        protocol=ProtocolConfig(),
    )


def test_interrupt_retry_skip_and_resume(tmp_path: Path):
    state = StateStore(tmp_path)
    jobs = materialize("preflight")[:3]
    state.add_jobs("preflight", jobs)
    first = jobs[0]
    attempt = state.start(first, (0, 1), tmp_path / "attempt-1")
    assert attempt == 1
    assert state.recover_interrupted() == 1
    assert state.next_attempt(first.job_id) == 2
    attempt = state.start(first, (0, 1), tmp_path / "attempt-2")
    state.fail(first.job_id, attempt, "network", retry=True)
    attempt = state.start(first, (0, 1), tmp_path / "attempt-3")
    state.complete(first.job_id, attempt)
    assert state.status_counts("preflight") == {"completed": 1, "pending": 2}


def test_e1a_source_transfer_never_defaults_native_sts(tmp_path: Path):
    state = StateStore(tmp_path)
    state.set_selection(
        "lightcone_recipe",
        {
            "scope": "last3",
            "parameterization": "lora",
            "rank": 8,
            "optimizer": "adamw",
            "learning_rate": 3e-5,
            "schedule": "constant",
            "stride": 10,
        },
    )
    capture = _segment_jobs(materialize("E1a")[0])[0]
    selected = _selection_for_job(state, capture)
    assert selected is not None
    assert selected["confidence_loss_weight"] == 1.0
    assert selected["confidence_temperatures"] == [1.0] * 7
    assert selected["scope"] == "last3"

    native = _segment_jobs(materialize("E1a")[2])[0]
    with pytest.raises(ScientificFailure, match="lacks fitted STS recipe"):
        _selection_for_job(state, native)
    temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    state.set_selection(
        "dspark_recipe",
        {**selected, "confidence_temperatures": temperatures, "verification": "native_scheduler"},
    )
    assert _selection_for_job(state, native)["confidence_temperatures"] == temperatures


def test_e1a_scientific_sts_unavailability_keeps_independent_latency_work(
    monkeypatch, tmp_path: Path
):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    parents = tuple(
        replace(
            job,
            job_id=f"source-{job.job_id}",
            node="E1a-source-transfer-v2",
            parameters={**job.parameters, "source_node": "E1a"},
        )
        for job in materialize("E1a")
    )
    state.add_internal_jobs(parents, storage_node="E1a-source-transfer-v2")
    executed = []

    def fake_run(config, state, node, stop_event, jobs):
        executed.extend(job.parameters["workload"] for job in jobs)
        for job in jobs:
            state.skip_job(job.job_id, "synthetic completed work")

    monkeypatch.setattr("lightcone_spec.runner._run_pending_jobs", fake_run)
    monkeypatch.setattr(
        "lightcone_spec.runner._fit_dspark_recipe",
        lambda state: (_ for _ in ()).throw(ScientificFailure("empty position 7")),
    )
    _run_node_jobs(config, state, "E1a-source-transfer-v2", threading.Event())
    assert executed == ["dspark_confidence_capture", "dspark_source_latency_panel"]
    assert "empty position 7" in state.selection("E1a_scientific_unavailable")
    assert state.status_counts("E1a-source-transfer-v2") == {"skipped": 3}


def test_dspark_dynamic_batch_budget_repair_requeues_only_latency_cells(tmp_path: Path):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    parents = _e1a_source_transfer_jobs()
    latency_parent = next(
        job
        for job in parents
        if job.parameters.get("workload") == "dspark_source_latency_panel"
    )
    latency = _segment_jobs(latency_parent)
    state.add_internal_jobs(latency, storage_node="E1a-source-transfer-v2-segments")
    for job in latency:
        attempt_dir = config.run_dir / "failed" / job.job_id
        attempt_dir.mkdir(parents=True)
        attempt = state.start(job, (0,), attempt_dir)
        state.fail(job.job_id, attempt, "fixed verification budget lies outside the exact batch", retry=False)

    audit = _requeue_dspark_dynamic_batch_budget_failures(state)
    assert audit["expected_cells"] == 16
    assert audit["reopened_failed_cells"] == 16
    assert state.status_counts("E1a-source-transfer-v2-segments") == {"pending": 16}
    assert all(state.failed_attempts(job.job_id) == 1 for job in latency)

    assert _requeue_dspark_dynamic_batch_budget_failures(state) == audit
    assert state.status_counts("E1a-source-transfer-v2-segments") == {"pending": 16}


def test_e1a_latency_protocol_repair_replaces_legacy_regime(
    monkeypatch, tmp_path: Path
):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    corrected = next(
        job
        for job in _e1a_source_transfer_jobs()
        if job.parameters.get("workload") == "dspark_source_latency_panel"
    )
    legacy_segments = [
        {
            key: value
            for key, value in segment.items()
            if key not in {"regime", "generation_tokens"}
        }
        for segment in corrected.parameters["segments"]
    ]
    legacy = replace(
        corrected,
        parameters={
            **corrected.parameters,
            "regime": "short_input_long_generation",
            "generation_tokens": 8192,
            "segments": legacy_segments,
        },
    )
    children = _segment_jobs(legacy)
    state.add_internal_jobs((legacy,), storage_node="E1a-source-transfer-v2")
    state.add_internal_jobs(children, storage_node="E1a-source-transfer-v2-segments")
    completed_dir = config.run_dir / "legacy-completed"
    completed_dir.mkdir()
    attempt = state.start(children[0], (0,), completed_dir)
    state.complete(children[0].job_id, attempt)
    failed_dir = config.run_dir / "legacy-failed"
    failed_dir.mkdir()
    attempt = state.start(children[1], (1,), failed_dir)
    state.fail(children[1].job_id, attempt, "legacy timeout", retry=False)

    def complete_replacement(config, state, node, stop_event):
        del stop_event
        for job in state.pending_jobs(node):
            output = config.run_dir / "replacement" / job.job_id
            output.mkdir(parents=True)
            attempt = state.start(job, (0, 1), output)
            state.complete(job.job_id, attempt)

    monkeypatch.setattr("lightcone_spec.runner._run_node_jobs", complete_replacement)
    _run_e1a_latency_protocol_repair_v3(
        config, state, threading.Event(), legacy
    )

    replacement = state.jobs("E1a-source-latency-v3")[0]
    assert replacement.parameters["regime"] == "short_input_long_generation"
    assert all(
        segment["regime"] == "long_input_short_output"
        and segment["generation_tokens"] == 256
        for segment in replacement.parameters["segments"]
    )
    excluded = set(state.selection("formal_evidence_exclusions"))
    assert {child.job_id for child in children} <= excluded
    assert state.job_status(legacy.job_id) == "skipped"
    assert state.job_status(children[0].job_id) == "completed"
    assert state.job_status(children[1].job_id) == "skipped"
    assert state.selection("formal_e1a_latency_protocol_repair_v3")["status"] == "completed"


def test_bundled_pool_stops_claiming_after_child_runtime_failure(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    parent = _e1a_source_transfer_jobs()[1]
    children = _segment_jobs(parent)
    state.add_internal_jobs((parent,), storage_node="E1a-source-transfer-v2")

    class FakeServer:
        def __init__(self, *args, **kwargs):
            self.session_key = ("same",)

        def stop(self):
            pass

    def fail_cell(config, state, job, *, gpus, **kwargs):
        attempt_dir = config.run_dir / "failed-pool" / job.job_id
        attempt_dir.mkdir(parents=True)
        attempt = state.start(job, gpus, attempt_dir)
        state.fail(job.job_id, attempt, "synthetic server crash", retry=False)

    monkeypatch.setattr("lightcone_spec.runner.ServerProcess", FakeServer)
    monkeypatch.setattr("lightcone_spec.runner._runtime_job", lambda config, state, job: job)
    monkeypatch.setattr("lightcone_spec.runner._selection_for_job", lambda *_: None)
    monkeypatch.setattr("lightcone_spec.runner._execute_cell", fail_cell)

    _run_pending_jobs(
        config,
        state,
        "E1a-source-transfer-v2",
        threading.Event(),
        (parent,),
    )
    counts = state.status_counts("E1a-source-transfer-v2-segments")
    assert 1 <= counts["failed"] <= 2
    assert counts["pending"] >= len(children) - 2
    assert state.job_status(parent.job_id) == "pending"


def test_preflight_resume_rederives_parallelism_without_new_attempts(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    jobs = materialize("preflight")
    state.add_jobs("preflight", jobs)
    evidence = tmp_path / "old-attempt"
    evidence.mkdir()
    raw = evidence / "metrics.json"
    raw.write_text('{"original": true}\n')
    attempt = state.start(jobs[0], (0, 1), evidence)
    state.complete(jobs[0].job_id, attempt)
    state.set_selection("headline_parallel", {"enabled": False})
    rows = [
        (job.to_dict(), {"goodput": 100.0, "itl_p99_ms": 10.0})
        for job in jobs if job.parameters.get("mode") in {"isolated", "concurrent"}
    ]
    monkeypatch.setattr("lightcone_spec.runner._metric_rows", lambda *_: rows)
    monkeypatch.setattr("lightcone_spec.runner.summarize_attempts", lambda *_: None)
    monkeypatch.setattr("lightcone_spec.runner._check_greedy_trajectories", lambda *_: None)
    monkeypatch.setattr(
        "lightcone_spec.runner.paired_relative_bca_interval",
        lambda candidate, baseline: (
            (-0.0021634, -0.0062713, -0.0005183)
            if baseline[0] == 100.0 else (0.0019563, 0.0000536, 0.0043664)
        ),
    )
    for _ in range(2):
        _reduce_node(config, state, "preflight")
        selection = state.selection("headline_parallel")
        assert selection["enabled"] is True
        assert selection["criterion"] == "paired_relative_bca_within_1pct_v2"
    assert raw.read_text() == '{"original": true}\n'
    assert state.next_attempt(jobs[0].job_id) == 2
    assert state.status_counts("preflight") == {"completed": 1, "pending": 9}
    rows.pop()  # An incomplete pair must still keep headline execution serial.
    _reduce_node(config, state, "preflight")
    assert state.selection("headline_parallel")["enabled"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_headline_workers_obey_admission_and_keep_whole_blocks(monkeypatch, tmp_path: Path, enabled):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    state.set_selection("headline_parallel", {"enabled": enabled})
    jobs = tuple(
        Job(
            job_id=f"final-{block}-{index}", node="E3b-final", ordinal=block * 2 + index,
            method="static", model="Qwen/Qwen3-8B", backend="DFLASH", task="controlled_baseline",
            context=4096, load="c1", width=16, block=block,
        )
        for block in (0, 1) for index in (0, 1)
    )
    state.add_jobs("E3b-final", jobs)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    seen = {}
    active = peak = 0

    class FakeServer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def execute(config, state, job, *, gpus, **kwargs):
        nonlocal active, peak
        attempt = state.start(job, gpus, tmp_path / job.job_id)
        with lock:
            assert job.job_id not in seen
            seen[job.job_id] = gpus
            active += 1
            peak = max(peak, active)
        if enabled:
            barrier.wait(timeout=5)
        time.sleep(0.01)
        state.complete(job.job_id, attempt)
        with lock:
            active -= 1

    monkeypatch.setattr("lightcone_spec.runner.ServerProcess", FakeServer)
    monkeypatch.setattr("lightcone_spec.runner._runtime_job", lambda config, state, job: job)
    monkeypatch.setattr("lightcone_spec.runner._selection_for_job", lambda *_: None)
    monkeypatch.setattr("lightcone_spec.runner._execute_cell", execute)
    _run_pending_jobs(config, state, "E3b-final", threading.Event(), jobs)
    assert peak == (2 if enabled else 1)
    assert seen == {job.job_id: (job.block,) for job in jobs}
    assert state.status_counts("E3b-final") == {"completed": 4}


@pytest.mark.parametrize("node", ["E0-tune", "E0-final", "E5-final"])
@pytest.mark.parametrize("enabled", [False, True])
def test_tp1_resource_workers_preserve_declaration_and_claims(monkeypatch, tmp_path, node, enabled):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    state.set_selection("headline_parallel", {"enabled": True})
    state.set_selection("tp1_resource_parallel_v1", {"enabled": enabled})
    jobs = tuple(Job(
        job_id=f"tp1-{block}-{i}", node=node, ordinal=block * 2 + i,
        method="static", model="Qwen/Qwen3-8B", backend="DFLASH",
        task="controlled_baseline", context=4096, load="c16", width=16,
        block=block if node.endswith("final") else None, gpu_count=2,
    ) for block in (0, 1) for i in (0, 1))
    state.add_jobs(node, jobs)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = peak = 0
    seen = {}
    ports = {}
    expected_parallel = enabled or node == "E0-tune"

    class FakeServer:
        def __init__(self, *args, **kwargs):
            ports[kwargs["gpus"]] = kwargs["port"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def execute(config, state, job, *, gpus, **kwargs):
        nonlocal active, peak
        attempt = state.start(job, gpus, tmp_path / job.job_id)
        with lock:
            assert job.job_id not in seen
            seen[job.job_id] = gpus
            active += 1
            peak = max(peak, active)
        if expected_parallel:
            barrier.wait(timeout=5)
        state.complete(job.job_id, attempt)
        with lock:
            active -= 1

    monkeypatch.setattr("lightcone_spec.runner.ServerProcess", FakeServer)
    monkeypatch.setattr("lightcone_spec.runner._runtime_job", lambda config, state, job: job)
    monkeypatch.setattr("lightcone_spec.runner._selection_for_job", lambda *_: None)
    monkeypatch.setattr("lightcone_spec.runner._execute_cell", execute)
    _run_pending_jobs(config, state, node, threading.Event(), jobs)
    assert peak == (2 if expected_parallel else 1)
    assert len(seen) == 4
    assert all(gpus == ((job.block if job.block is not None else job.ordinal % 2,)
                        if expected_parallel else (0, 1))
               for job in jobs for gpus in [seen[job.job_id]])
    assert len(set(ports.values())) == (2 if expected_parallel else 1)
    assert all(job.gpu_count == 2 for job in state.jobs(node))
    _run_pending_jobs(config, state, node, threading.Event(), state.pending_jobs(node))
    assert all(state.next_attempt(job.job_id) == 2 for job in jobs)


def test_tp1_worker_startup_failure_stops_sibling_after_active_cell(monkeypatch, tmp_path):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    jobs = tuple(
        Job(
            job_id=f"e0-{ordinal}",
            node="E0-tune",
            ordinal=ordinal,
            method="static",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            gpu_count=2,
        )
        for ordinal in range(4)
    )
    state.add_jobs("E0-tune", jobs)
    gpu1_started = threading.Event()
    executed: list[str] = []

    class FakeServer:
        def __init__(self, *args, **kwargs):
            self.gpus = kwargs["gpus"]

        def __enter__(self):
            if self.gpus == (0,):
                assert gpu1_started.wait(timeout=5)
                raise RuntimeError("deterministic startup failure")
            gpu1_started.set()
            return self

        def __exit__(self, *args):
            return False

    def execute(config, state, job, *, gpus, **kwargs):
        attempt = state.start(job, gpus, tmp_path / job.job_id)
        time.sleep(0.02)
        state.complete(job.job_id, attempt)
        executed.append(job.job_id)

    monkeypatch.setattr("lightcone_spec.runner.ServerProcess", FakeServer)
    monkeypatch.setattr("lightcone_spec.runner._runtime_job", lambda config, state, job: job)
    monkeypatch.setattr("lightcone_spec.runner._selection_for_job", lambda *_: None)
    monkeypatch.setattr(
        "lightcone_spec.runner.server_session_key",
        lambda job, selection: (job.job_id,),
    )
    monkeypatch.setattr("lightcone_spec.runner._execute_cell", execute)
    with pytest.raises(RuntimeError, match="deterministic startup failure"):
        _run_pending_jobs(
            config,
            state,
            "E0-tune",
            threading.Event(),
            state.pending_jobs("E0-tune"),
        )
    assert len(executed) == 1
    assert int(executed[0].split("-")[-1]) % 2 == 1
    assert state.status_counts("E0-tune") == {"completed": 1, "pending": 3}


def test_e0_probe_unsupported_architecture_is_terminal_compatibility_outcome(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    job = Job(
        job_id="e0-eagle3-probe",
        node="E0-tune",
        ordinal=0,
        method="static",
        model="Qwen/Qwen3-4B",
        backend="EAGLE3",
        task="CalibrationMix",
        gpu_count=2,
        parameters={"probe": True, "adaptive_probe": True},
    )
    state.add_jobs("E0-tune", (job,))

    class UnsupportedServer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError(
                "Cannot find model module. 'Qwen3Eagle3Model' is not a registered "
                "model and 'AutoModel' is not present in auto_map"
            )

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("lightcone_spec.runner.ServerProcess", UnsupportedServer)
    monkeypatch.setattr("lightcone_spec.runner._runtime_job", lambda config, state, job: job)
    monkeypatch.setattr("lightcone_spec.runner._selection_for_job", lambda *_: None)
    _run_pending_jobs(
        config,
        state,
        "E0-tune",
        threading.Event(),
        state.pending_jobs("E0-tune"),
    )
    assert state.status_counts("E0-tune") == {"completed": 1}
    metrics = json.loads(
        (state.completed_attempt_dir(job.job_id) / "metrics.json").read_text()
    )
    assert metrics["scientific_outcome"] == "infeasible"
    assert metrics["compatible"] is False
    assert metrics["capacity_feasible"] == "N/A"
    assert metrics["compatibility_reason"] == "unsupported_model_architecture"


def test_tp1_started_blocks_and_isolation_are_not_remapped(tmp_path):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    state.set_selection("tp1_resource_parallel_v1", {"enabled": True})
    base = Job(job_id="base", node="E5-final", ordinal=0, method="static",
               gpu_count=2, block=0, model="Qwen/Qwen3-8B", backend="DFLASH", task="MATH-500")
    jobs = tuple(replace(base, job_id=f"j-{i}", ordinal=i, block=i // 2)
                 for i in range(6))
    state.add_jobs(base.node, jobs)
    for job, gpus in ((jobs[0], (0, 1)), (jobs[2], (1,))):
        directory = tmp_path / job.job_id
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({
            "parameters": {"execution_gpu_ids": list(gpus)},
        }))
        attempt = state.start(job, gpus, directory)
        if job == jobs[2]:
            state.interrupt(job.job_id, attempt, "interruption")
        else:
            state.complete(job.job_id, attempt)
    allocations = _execution_allocations(config, state, base.node, state.pending_jobs(base.node))
    assert allocations[jobs[1].job_id] == (0, 1)  # Old block remains isolated.
    assert allocations[jobs[2].job_id] == allocations[jobs[3].job_id] == (1,)
    assert allocations[jobs[4].job_id] == allocations[jobs[5].job_id] == (0,)
    for params in ({"topology": "tp2_dp1"}, {"topology": "two_replica_tp1_dp2"},
                   {"profiler": "nsys"}, {"failure": "kill"},
                   {"controlled_pair_baseline": True}, {"requires_isolation": True}):
        special = replace(base, block=None, parameters=params)
        assert _execution_allocations(config, state, base.node, (special,))["base"] == (0, 1)
    special = replace(base, node="E6-final", block=None)
    assert _execution_allocations(config, state, special.node, (special,))["base"] == (0, 1)


def test_tp1_v2_gate_uses_exact_excluded_interference_matrix(tmp_path):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    state.set_selection("tp1_resource_parallel_v2", {"enabled": True})
    job = Job(
        job_id="ordinary-e5",
        node="E5-pilot",
        ordinal=1,
        method="static",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="CalibrationMix",
        context=40928,
        load="c16",
        width=16,
        gpu_count=2,
    )
    assert _execution_allocations(config, state, job.node, (job,))[job.job_id] == (1,)

    validation = _tp1_interference_v2_jobs()
    assert len(validation) == 8
    assert {(row.parameters["mode"], row.parameters["repetition"], row.parameters["gpu_index"])
            for row in validation} == {
        (mode, repetition, gpu)
        for mode in ("isolated", "concurrent")
        for repetition in range(2)
        for gpu in range(2)
    }
    assert all(row.load == "c16" and row.context == 40928 for row in validation)
    assert all(row.parameters["execution_request_count"] == 16 for row in validation)
    assert all(row.parameters["generation_tokens"] == 256 for row in validation)
    assert all(row.parameters["requires_isolation"] is True for row in validation)


def test_e0_tune_is_the_only_unconditional_single_gpu_exemption(tmp_path):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    base = Job(
        job_id="e0-resource",
        node="E0-tune",
        ordinal=1,
        method="static",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="controlled_baseline",
        gpu_count=2,
    )
    assert _execution_allocations(config, state, "E0-tune", (base,))[base.job_id] == (1,)
    confirmation = replace(base, node="E0-final")
    assert _execution_allocations(
        config, state, "E0-final", (confirmation,)
    )[base.job_id] == (0, 1)


def test_priority_window_v2_migrates_without_overwriting_v1(monkeypatch, tmp_path):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    state.set_selection("formal_soft_gate_resume_version", 1)
    state.set_selection(
        "formal_priority_window_v1",
        {"version": 1, "status": "completed", "order": ["E1a", "E5-pilot", "E0-tune"]},
    )
    calls = []
    monkeypatch.setattr(
        "lightcone_spec.runner._run_e1a_source_transfer_v2",
        lambda config, state, stop: calls.append("E1a"),
    )
    monkeypatch.setattr(
        "lightcone_spec.runner._run_priority_paper_node",
        lambda config, state, node, stop: calls.append(node),
    )
    _run_priority_window_v2(config, state, threading.Event())
    _run_priority_window_v2(config, state, threading.Event())
    assert calls == ["E1a", "E0-tune"]
    assert state.selection("formal_priority_window_v1")["order"][1] == "E5-pilot"
    migrated = state.selection("formal_priority_window_v2")
    assert migrated["status"] == "completed"
    assert migrated["order"] == [
        "E1a", "E0-tune", "all_non_E5", "E5-pilot", "E5-final"
    ]
    assert migrated["preserved_v1_audit"]["version"] == 1


def test_completed_e5_prefix_reopens_only_for_appended_topology_parents(tmp_path):
    state = StateStore(tmp_path)
    expanded = materialize("E5-pilot")
    legacy = expanded[:11]
    state.add_jobs("E5-pilot", legacy)
    state.set_stage_status("E5-pilot", "completed", row_count=len(legacy))
    resumed = _resume_materialization(state, "E5-pilot", expanded)
    assert resumed[:11] == legacy
    assert resumed[11:] == expanded[11:]
    state.add_jobs("E5-pilot", resumed)
    assert state.stage_status("E5-pilot") == "pending"
    assert len(state.jobs("E5-pilot")) == 19


def test_eta_resource_model_includes_e0_exemption_and_dual_gpu_topologies(tmp_path):
    state = StateStore(tmp_path)
    e0 = Job(
        job_id="e0",
        node="E0-tune",
        ordinal=0,
        method="static",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="LiveCodeBench",
        gpu_count=2,
    )
    tp2 = replace(
        e0,
        job_id="tp2",
        node="E5-final",
        parameters={"topology": "tp2_dp1"},
    )
    dp2 = replace(
        e0,
        job_id="dp2",
        node="E5-final",
        parameters={"topology": "two_replica_tp1_dp2"},
    )
    assert _eta_resource_count(state, e0) == 1
    assert _eta_resource_count(state, tp2) == 2
    assert _eta_resource_count(state, dp2) == 2


def test_tp1_v2_gate_stays_disabled_when_numa_binding_is_unavailable(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    state = StateStore(config.run_dir)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "numa-affinity.json").write_text(
        json.dumps({"enabled": False, "reason": "platform unavailable"})
    )
    monkeypatch.setenv("LIGHTCONE_NUMA_ISOLATION", "1")
    _run_tp1_interference_v2(config, state, threading.Event())
    selection = state.selection("tp1_resource_parallel_v2")
    assert selection["enabled"] is False
    assert selection["criterion"] == "numa_affinity_unavailable"
    assert state.jobs("TP1-interference-v2") == ()


def test_session_order_filter_uses_original_pair_seed(tmp_path):
    config = _config(tmp_path)
    keys = [(block, None, method) for block in (0, 1)
            for method in ("static", "lightcone", "tts")]
    original = _session_order(config, "E0-final", (0, 1), keys)
    assert len(original) == len(set(original)) == 6
    for block in (0, 1):
        assert [key for key in original if key[0] == block] == _session_order(
            config, "E0-final", (0, 1), [key for key in keys if key[0] == block],
        )


def test_remapped_sessions_keep_relative_order_and_stop_before_new_claim(monkeypatch, tmp_path):
    config = _config(tmp_path)
    jobs = tuple(Job(
        job_id=f"session-{block}-{i}", node="E0-final", ordinal=block * 10 + i,
        method=method, model="Qwen/Qwen3-8B", backend="DFLASH", task="MATH-500",
        context=4096, load="c1", width=16, block=block, gpu_count=2,
    ) for block in (0, 1) for i, method in enumerate(("static", "lightcone", "tts")))
    observed = []

    class FakeServer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("lightcone_spec.runner.ServerProcess", FakeServer)
    monkeypatch.setattr("lightcone_spec.runner._runtime_job", lambda config, state, job: job)
    monkeypatch.setattr("lightcone_spec.runner._selection_for_job", lambda *_: None)
    for enabled in (False, True):
        state = StateStore(tmp_path / str(enabled))
        state.add_jobs("E0-final", jobs)
        state.set_selection("tp1_resource_parallel_v1", {"enabled": enabled})
        # Serialize workers so the assertion compares session order, not timing.
        state.set_selection("headline_parallel", {"enabled": False})
        stop = threading.Event()
        seen = []

        def execute(config, state, job, *, gpus, **kwargs):
            attempt = state.start(job, gpus, tmp_path / job.job_id)
            state.complete(job.job_id, attempt)
            seen.append(job.job_id)
            if len(seen) == 2:
                stop.set()

        monkeypatch.setattr("lightcone_spec.runner._execute_cell", execute)
        _run_pending_jobs(config, state, "E0-final", stop, state.pending_jobs("E0-final"))
        assert state.status_counts("E0-final") == {"completed": 2, "pending": 4}
        observed.append(seen[:])
    assert observed[0] == observed[1]


def test_interrupted_server_cleanup_reads_proc_without_spawning_ps(
    monkeypatch, tmp_path: Path
):
    run_dir = tmp_path / "run"
    proc_root = tmp_path / "proc"
    active = run_dir / "jobs" / "job-a" / "attempt-01"
    unrelated = run_dir / "jobs" / "job-b" / "attempt-01"
    stopped = run_dir / "sessions" / "session-c"
    for path, pid in ((active, 101), (unrelated, 102), (stopped, 103)):
        path.mkdir(parents=True)
        (path / "server.pid").write_text(str(pid), encoding="utf-8")
        (proc_root / str(pid)).mkdir(parents=True)
    (proc_root / "101" / "cmdline").write_bytes(
        b"python\0-m\0sglang.launch_server\0"
    )
    (proc_root / "102" / "cmdline").write_bytes(b"python\0worker.py\0")
    (proc_root / "103" / "cmdline").write_bytes(
        b"python\0-m\0sglang.launch_server\0"
    )
    (stopped / "server.stopped").touch()
    killed = []
    monkeypatch.setattr(
        "lightcone_spec.runner.os.killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )

    _cleanup_interrupted_servers(run_dir, proc_root=proc_root)

    assert killed == [(101, signal.SIGTERM)]


def test_e2_halving_floor_never_invents_infeasible_finalists():
    assert _e2_keep_count(53, 53, 1) == 21
    assert _e2_keep_count(21, 20, 2) == 20
    assert _e2_keep_count(20, 20, 2) == 20
    assert _e2_keep_count(20, 20, 3) == 1
    assert _e2_keep_count(21, 0, 2) == 0
    with pytest.raises(ValueError, match="invalid E2 candidate cardinality"):
        _e2_keep_count(20, 21, 2)


def test_explicit_interruption_returns_job_to_pending(tmp_path: Path):
    state = StateStore(tmp_path)
    job = materialize("preflight")[0]
    state.add_jobs("preflight", (job,))
    attempt = state.start(job, (0, 1), tmp_path / "attempt")
    state.interrupt(job.job_id, attempt, "runner cancellation")
    assert state.status_counts("preflight") == {"pending": 1}
    assert state.failed_attempts(job.job_id) == 0
    with state.connect() as connection:
        row = connection.execute(
            "SELECT status,error FROM attempts WHERE job_id=?", (job.job_id,)
        ).fetchone()
    assert (row["status"], row["error"]) == ("interrupted", "runner cancellation")


def test_segment_jobs_resume_without_expanding_paper_stage(tmp_path: Path):
    state = StateStore(tmp_path)
    parent = materialize("E3a")[0]
    state.add_jobs("E3a", (parent,))
    children = _segment_jobs(parent)
    state.add_internal_jobs(children, storage_node="E3a-segments")
    assert len(children) == 2
    assert state.status_counts("E3a") == {"pending": 1}
    assert state.status_counts("E3a-segments") == {"pending": 2}
    attempt = state.start(children[0], (0,), tmp_path / "child")
    state.complete(children[0].job_id, attempt)
    assert state.completed_attempt_dir(children[0].job_id) == tmp_path / "child"


def test_e2_dependency_identity_survives_candidate_reordering(tmp_path: Path):
    state = StateStore(tmp_path)
    first = {
        "parameterization": "lora",
        "rank": 1,
        "scope": "last1",
        "optimizer": "nag",
        "learning_rate": 3e-5,
        "schedule": "constant",
    }
    changed = {
        **first,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
    }

    first_jobs = _e2_missing_dependency_jobs(state, "E2-r2", [first])
    changed_jobs = _e2_missing_dependency_jobs(state, "E2-r2", [changed])
    assert len(first_jobs) == len(changed_jobs) == 1
    assert first_jobs[0].ordinal == changed_jobs[0].ordinal == 0
    assert first_jobs[0].job_id != changed_jobs[0].job_id
    assert "nag__lr-3em05" in first_jobs[0].job_id
    assert "adamw__lr-0p001" in changed_jobs[0].job_id

    state.add_internal_jobs(first_jobs, storage_node="S10-e2-dependency-repair")
    state.add_internal_jobs(changed_jobs, storage_node="S10-e2-dependency-repair")
    state.add_internal_jobs(changed_jobs, storage_node="S10-e2-dependency-repair")
    assert state.status_counts("S10-e2-dependency-repair") == {"pending": 2}


def test_e2_dependency_reuses_equivalent_completed_evidence(tmp_path: Path):
    state = StateStore(tmp_path)
    selection = {
        "parameterization": "lora",
        "rank": 8,
        "scope": "last1",
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "schedule": "constant",
        "registered_request_count": 16,
        "stride": 10,
    }
    source = materialize("E2-r1", e2_rows=[selection])[0]
    completed = replace(
        source,
        job_id="legacy-dependency",
        node="S10-e2-dependency-repair",
        parameters={
            **source.parameters,
            "source_node": "E2-r1",
            "reconciliation_kind": "e2_dependency_closure",
        },
    )
    pending = replace(completed, job_id="recipe-identity-dependency")
    duplicate = replace(
        completed,
        job_id=(
            "s10-e2-dependency-v2__E2-r1__"
            "lora-r8__last1__adamw__lr-0p001__constant"
        ),
    )
    state.add_internal_jobs(
        (completed, pending, duplicate),
        storage_node="S10-e2-dependency-repair",
    )
    attempt_dir = tmp_path / "legacy" / "attempt-01"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "config.json").write_text(json.dumps(completed.to_dict()))
    (attempt_dir / "metrics.json").write_text(json.dumps({"finite": True}))
    attempt = state.start(completed, (0,), attempt_dir)
    state.complete(completed.job_id, attempt)
    duplicate_dir = tmp_path / "duplicate" / "attempt-01"
    duplicate_dir.mkdir(parents=True)
    (duplicate_dir / "config.json").write_text(json.dumps(duplicate.to_dict()))
    (duplicate_dir / "metrics.json").write_text(json.dumps({"finite": True}))
    duplicate_attempt = state.start(duplicate, (1,), duplicate_dir)
    state.complete(duplicate.job_id, duplicate_attempt)

    assert _set_e2_expected_evidence(state, "E2-r1", [selection]) == 0
    assert _exclude_redundant_e2_dependency_jobs(state, "E2-r1") == 1
    assert _skip_satisfied_e2_dependency_jobs(state, "E2-r1") == 1
    assert state.status_counts("S10-e2-dependency-repair") == {
        "completed": 2,
        "skipped": 1,
    }
    assert completed.job_id not in state.selection("formal_evidence_exclusions", [])
    assert duplicate.job_id in state.selection("formal_evidence_exclusions", [])


def test_plain_config_resume_rejects_different_values(tmp_path: Path):
    config = _config(tmp_path)
    config.run_dir.mkdir()
    _save_or_validate_run_config(config)
    _save_or_validate_run_config(config)
    changed = replace(config, server=replace(config.server, max_new_tokens=128))
    with pytest.raises(RuntimeError, match="different experiment config"):
        _save_or_validate_run_config(changed)
    saved = yaml.safe_load((config.run_dir / "paper.yaml").read_text())
    assert saved["protocol"]["preset"] == "paper-v2"
    assert "final_blocks" not in saved["protocol"]


def test_plain_config_resume_allows_new_sglang_path_and_dataset_key(tmp_path: Path):
    config = _config(tmp_path)
    config.run_dir.mkdir()
    _save_or_validate_run_config(config)
    changed = replace(
        config,
        sglang_root=tmp_path / "sglang-v14",
        datasets={"AIME-2024": tmp_path / "aime-2024.jsonl"},
    )
    _save_or_validate_run_config(changed)
    saved = yaml.safe_load((config.run_dir / "paper.yaml").read_text())
    assert saved["paths"]["sglang_root"].endswith("sglang-v14")
    assert set(saved["paths"]["datasets"]) == {"AIME-2024"}


def test_sqlite_records_actual_gpu_pair(tmp_path: Path):
    state = StateStore(tmp_path)
    job = materialize("E6-final")[0]
    state.add_jobs("E6-final", (job,))
    state.start(job, (4, 5), tmp_path / "attempt")
    with state.connect() as connection:
        row = connection.execute(
            "SELECT assigned_gpus FROM jobs WHERE job_id=?", (job.job_id,)
        ).fetchone()
    assert row["assigned_gpus"] == "4,5"


def test_protocol_repair_requeues_without_deleting_attempt_history(tmp_path: Path):
    state = StateStore(tmp_path)
    failed = materialize("E4-profile")[0]
    skipped = materialize("E3b-pilot")[0]
    state.add_jobs("E4-profile", (failed,))
    state.add_jobs("E3b-pilot", (skipped,))
    attempt = state.start(failed, (0, 1), tmp_path / "failed-attempt")
    state.fail(failed.job_id, attempt, "diagnostic", retry=False)
    state.skip_job(skipped.job_id, "upstream failure")

    assert state.retry_failed("E4-profile") == 1
    assert state.reopen_skipped(("E3b-pilot",)) == 1
    assert state.status_counts("E4-profile") == {"pending": 1}
    assert state.status_counts("E3b-pilot") == {"pending": 1}
    with state.connect() as connection:
        attempts = connection.execute(
            "SELECT status,error FROM attempts WHERE job_id=?", (failed.job_id,)
        ).fetchall()
    assert [(row["status"], row["error"]) for row in attempts] == [
        ("failed", "diagnostic")
    ]


def test_targeted_registered_load_retry_preserves_other_failures(tmp_path: Path):
    state = StateStore(tmp_path)
    first, second = materialize("E4-profile")[:2]
    state.add_jobs("E4-profile", (first, second))
    first_attempt = state.start(first, (0, 1), tmp_path / "first-attempt")
    state.fail(
        first.job_id,
        first_attempt,
        "RuntimeError: 9 requests did not complete in a measured cell",
        retry=False,
    )
    second_attempt = state.start(second, (0, 1), tmp_path / "second-attempt")
    state.fail(second.job_id, second_attempt, "RuntimeError: connection refused", retry=False)

    assert (
        state.retry_failed_errors(
            "E4-profile",
            "requests did not complete in a measured cell",
            reason="registered-load timeout classification repair",
        )
        == 1
    )
    assert state.status_counts("E4-profile") == {"failed": 1, "pending": 1}
    with state.connect() as connection:
        attempts = connection.execute(
            "SELECT job_id,status,error FROM attempts ORDER BY job_id"
        ).fetchall()
    assert [(row["job_id"], row["status"], row["error"]) for row in attempts] == [
        (first.job_id, "failed", "RuntimeError: 9 requests did not complete in a measured cell"),
        (second.job_id, "failed", "RuntimeError: connection refused"),
    ]


def test_completed_stage_resume_preserves_immutable_materialization(tmp_path: Path):
    state = StateStore(tmp_path)
    original = materialize("E2-r2")[0]
    state.add_jobs("E2-r2", (original,))
    attempt = state.start(original, (0,), tmp_path / "attempt")
    state.complete(original.job_id, attempt)
    assert state.finish_stage("E2-r2") == "completed"

    changed = replace(original, parameters={**original.parameters, "lr": 9e-4})
    resumed = _resume_materialization(state, "E2-r2", (changed,))
    assert resumed == (original,)
    state.add_jobs("E2-r2", resumed)


def test_pending_stage_resume_uses_current_materialization(tmp_path: Path):
    state = StateStore(tmp_path)
    original = materialize("E2-r2")[0]
    state.add_jobs("E2-r2", (original,))
    changed = replace(original, parameters={**original.parameters, "lr": 9e-4})
    assert _resume_materialization(state, "E2-r2", (changed,)) == (changed,)


def test_reopened_stage_preserves_completed_rows_but_checks_pending_rows(
    tmp_path: Path,
):
    state = StateStore(tmp_path)
    completed, pending = materialize("E1a")[:2]
    state.add_jobs("E1a", (completed, pending))
    attempt = state.start(completed, (0,), tmp_path / "completed-attempt")
    state.complete(completed.job_id, attempt)

    changed_completed = replace(
        completed,
        parameters={**completed.parameters, "generation_tokens": 16_384},
    )
    changed_pending = replace(
        pending,
        parameters={**pending.parameters, "generation_tokens": 16_384},
    )
    resumed = _resume_materialization(
        state,
        "E1a",
        (changed_completed, changed_pending),
    )

    assert resumed[0].job_id == completed.job_id
    assert resumed[0].parameters["generation_tokens"] == 8192
    assert resumed[1] == changed_pending
    with pytest.raises(RuntimeError, match="row 1 changed after materialization"):
        state.add_jobs("E1a", resumed)


def test_e0_e6_partial_resume_is_scoped_and_idempotent(tmp_path: Path):
    state = StateStore(tmp_path)
    e0 = Job("e0-unsupported", "E0-tune", 0, "static", "m", "DFLASH", "t")
    e0_sibling = Job("e0-sibling", "E0-tune", 1, "static", "m", "DFLASH", "t")
    state.add_jobs("E0-tune", (e0, e0_sibling))
    attempt = state.start(e0, (0, 1), tmp_path / "e0-attempt")
    state.fail(
        e0.job_id,
        attempt,
        "RuntimeError: specialized variants fail closed",
        retry=False,
    )
    state.skip_job(e0_sibling.job_id, "stopped after sibling failure")
    e6_rows = tuple(
        Job(
            f"e6-{index}",
            "E6-common-load-segments",
            index,
            "target_only",
            "m",
            "NONE",
            "t",
        )
        for index in range(2)
    )
    state.add_internal_jobs(e6_rows, storage_node="E6-common-load-segments")
    for index, job in enumerate(e6_rows):
        attempt = state.start(job, (0, 1), tmp_path / f"e6-attempt-{index}")
        state.fail(
            job.job_id,
            attempt,
            "ValueError: dataset supplied 175 prompts; 256 required",
            retry=False,
        )
    downstream = ("E5-pilot", "E5-final", "E6-pilot", "E6-final", "E0-pilot", "E0-final")
    for index, node in enumerate(downstream):
        job = Job(f"downstream-{index}", node, 0, "static", "m", "DFLASH", "t")
        state.add_jobs(node, (job,))
        state.skip_pending(node, "old dependency failure")

    _repair_e0_e6_partial_resume_v1(state)

    assert state.status_counts("E0-tune") == {"pending": 2}
    assert state.status_counts("E6-common-load-segments") == {"pending": 2}
    assert all(state.status_counts(node) == {"pending": 1} for node in downstream)
    audit = state.selection("formal_e0_e6_partial_resume_version")
    assert audit == {"version": 1, "e0_retried": 1, "e6_retried": 2, "reopened": 7}
    _repair_e0_e6_partial_resume_v1(state)
    assert state.selection("formal_e0_e6_partial_resume_version") == audit


def test_metric_dedup_e5_resume_reopens_only_diagnosed_rows(tmp_path: Path):
    state = StateStore(tmp_path)
    e1a = Job("e1a-bug", "E1a", 0, "static", "m", "DFLASH", "t")
    e1a_science = replace(e1a, job_id="e1a-science", ordinal=1)
    e5 = Job("e5-bug", "E5-pilot", 0, "static", "m", "DSPARK", "t")
    e5_segment = replace(e5, job_id="e5-segment", node="E5-pilot-segments")
    state.add_jobs("E1a", (e1a, e1a_science))
    state.skip_pending("E1a", "DSpark confidence calibration is incomplete")
    state.add_jobs("E5-pilot", (e5,))
    state.skip_pending(
        "E5-pilot", "DSpark confidence calibration was scientifically infeasible"
    )
    state.add_internal_jobs((e5_segment,), storage_node="E5-pilot-segments")
    attempt = state.start(e5_segment, (0, 1), tmp_path / "e5-attempt")
    state.fail(
        e5_segment.job_id,
        attempt,
        "RuntimeError: 8 requests did not complete in a measured cell",
        retry=False,
    )
    # One unrelated skip must remain untouched by the exact migration.
    with state.connect() as connection:
        connection.execute(
            "UPDATE jobs SET error='independent scientific rejection' WHERE job_id=?",
            (e1a_science.job_id,),
        )
    for name in ("E1a_failed", "dspark_confidence_weight", "e1a_finalists"):
        state.set_selection(name, True)

    _repair_metric_dedup_e5_resume_v1(state)

    assert state.pending_jobs("E1a") == (e1a,)
    assert state.status_counts("E1a") == {"pending": 1, "skipped": 1}
    assert state.status_counts("E5-pilot") == {"pending": 1}
    assert state.status_counts("E5-pilot-segments") == {"pending": 1}
    assert state.selection("E1a_failed", None) is None
    audit = state.selection("formal_metric_dedup_e5_resume_version")
    assert audit["e1a_jobs_reopened"] == 1
    assert audit["e5_dependency_jobs_reopened"] == 1
    assert audit["e5_registered_load_cells_retried"] == 1
    _repair_metric_dedup_e5_resume_v1(state)
    assert state.selection("formal_metric_dedup_e5_resume_version") == audit


def test_e0_source_transfer_upgrade_is_idempotent_and_preserves_old_evidence(
    tmp_path: Path,
):
    state = StateStore(tmp_path)
    planned = materialize("E0-tune")
    legacy = Job(
        job_id="E0-tune__legacy-grid-row",
        node="E0-tune",
        ordinal=12,
        method="onlinespec_ogd",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="CalibrationMix",
        parameters={"stride": 20, "learning_rate": 1e-4},
    )
    state.add_jobs("E0-tune", (*planned[:12], legacy))
    attempt_dir = tmp_path / "legacy-attempt"
    attempt_dir.mkdir()
    (attempt_dir / "config.json").write_text(json.dumps(legacy.to_dict()))
    (attempt_dir / "metrics.json").write_text(json.dumps({"goodput": 1.0}))
    attempt = state.start(legacy, (0, 1), attempt_dir)
    state.complete(legacy.job_id, attempt)

    upgraded = _upgrade_legacy_e0_materialization(state, planned)
    assert upgraded is not None
    assert state.selection("formal_e0_source_transfer_upgrade_version") == 1
    assert legacy.job_id in state.selection("formal_evidence_exclusions")
    assert state.completed_attempt_dir(legacy.job_id) == attempt_dir
    assert len([job for job in state.jobs("E0-tune") if job.job_id in {row.job_id for row in planned}]) == 54

    _upgrade_legacy_e0_materialization(state, planned)
    assert len([job for job in state.jobs("E0-tune") if job.job_id in {row.job_id for row in planned}]) == 54


def test_e0_recipe_selection_injects_only_feasible_validated_methods(tmp_path: Path):
    state = StateStore(tmp_path)
    validations = tuple(
        job
        for job in materialize("E0-tune")
        if job.parameters.get("recipe_validation")
    )
    state.add_jobs("E0-tune", validations)
    for job in validations:
        attempt_dir = tmp_path / job.method / "attempt-01"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "config.json").write_text(json.dumps(job.to_dict()))
        feasible = job.method != "onlinespec_opt"
        (attempt_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "feasible": feasible,
                    "slo_pass": feasible,
                    "scientific_outcome": "completed" if feasible else "rejected",
                }
            )
        )
        attempt = state.start(job, (0, 1), attempt_dir)
        state.complete(job.job_id, attempt)

    recipes = _select_e0_recipes(state)
    assert set(recipes) == {
        "Qwen/Qwen3-8B|DFLASH|onlinespec_ogd",
        "Qwen/Qwen3-8B|DFLASH|onlinespec_ens",
    }
    state.set_selection("e0_recipes", recipes)
    downstream = materialize("E0-pilot", e0_recipes=recipes)
    assert {job.method for job in downstream if job.method.startswith("onlinespec")} == {
        "onlinespec_ogd",
        "onlinespec_ens",
    }
    job = next(job for job in downstream if job.method == "onlinespec_ens")
    assert json.dumps(_selection_for_job(state, job), sort_keys=True) == json.dumps(
        E0_ONLINESPEC_RECIPES["onlinespec_ens"], sort_keys=True
    )


def test_ncu_permission_probe_reports_provider_block(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 1, "", "==ERROR== ERR_NVGPUCTRPERM")

    monkeypatch.setattr("lightcone_spec.runner.subprocess.run", fake_run)
    reason = _ncu_permission_block_reason(tmp_path / "ncu", tmp_path / "python", 1)
    assert reason == "Nsight Compute counters blocked by provider (ERR_NVGPUCTRPERM)"
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["kwargs"]["timeout"] == 120


def test_blocked_profiler_is_auditable_completed_outcome(tmp_path: Path):
    state = StateStore(tmp_path)
    job = materialize("E4-profile")[2]
    state.add_jobs("E4-profile", (job,))
    _complete_blocked_profiler(
        state,
        job,
        tmp_path,
        (0, 1),
        "Nsight Compute counters blocked by provider (ERR_NVGPUCTRPERM)",
    )
    assert state.status_counts("E4-profile") == {"completed": 1}
    metrics_path = state.completed_attempt_dir(job.job_id) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["scientific_outcome"] == "blocked"
    assert metrics["feasible"] is False
    assert metrics["profiler"] == "ncu"


def test_completed_s10_repair_requeues_bundled_segments_and_downstream(
    tmp_path: Path,
):
    state = StateStore(tmp_path)
    width = materialize("E3b-pilot")[0]
    failed_segment = replace(
        width,
        job_id="e3-width-test__segment-000",
        node="E3-width-calibration-segments",
    )
    skipped_pilot = materialize("E3b-pilot")[1]
    state.add_internal_jobs((failed_segment,))
    state.add_jobs("E3b-pilot", (skipped_pilot,))
    attempt = state.start(failed_segment, (0,), tmp_path / "failed-segment")
    state.fail(failed_segment.job_id, attempt, "scientific rejection", retry=False)
    state.skip_job(skipped_pilot.job_id, "width calibration incomplete")
    state.mark_stage_failed("E3b-pilot")
    state.set_selection("formal_s10_reconciliation_complete", True)

    _repair_completed_s10_downstream_resume(state)

    assert state.status_counts("E3-width-calibration-segments") == {"pending": 1}
    assert state.status_counts("E3b-pilot") == {"pending": 1}
    assert state.stage_status("E3b-pilot") == "pending"
    assert state.selection("formal_s10_downstream_resume_version") == 2
    audit = json.loads(
        (
            tmp_path
            / "stages"
            / "S10-reconciliation"
            / "downstream-resume-v2.json"
        ).read_text()
    )
    assert audit["width_calibration_segment_retries"] == 1
    assert audit["future_jobs_reopened"] == 1

    _repair_completed_s10_downstream_resume(state)
    assert state.status_counts("E3-width-calibration-segments") == {"pending": 1}


def test_soft_gate_width_reconciliation_materializes_exact_seven_replacements(
    tmp_path: Path,
):
    state = StateStore(tmp_path)
    methods = ("tts", "lightcone", "l0_naive")
    counts = {"tts": 3, "lightcone": 3, "l0_naive": 1}
    rows = []
    ordinal = 0
    for method in methods:
        for index in range(counts[method]):
            job = Job(
                job_id=f"width4-{method}-{index}",
                node="E3-width-calibration-segments",
                ordinal=ordinal,
                method=method,
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="CalibrationMix",
                width=4,
            )
            ordinal += 1
            rows.append(job)
    unaffected = replace(rows[0], job_id="width8-tts", ordinal=ordinal, width=8)
    state.add_internal_jobs(tuple((*rows, unaffected)))
    for index, job in enumerate((*rows, unaffected)):
        attempt_dir = tmp_path / f"attempt-{index}"
        attempt_dir.mkdir()
        (attempt_dir / "config.json").write_text(json.dumps(job.to_dict()))
        (attempt_dir / "metrics.json").write_text(
            json.dumps({"fallbacks": 1, "hard_feasible": False})
        )
        attempt = state.start(job, (index % 2,), attempt_dir)
        state.complete(job.job_id, attempt)

    replacements = _soft_gate_width_replacements(state)

    assert len(replacements) == 7
    assert {job.method for job in replacements} == set(methods)
    assert all(job.width == 4 for job in replacements)
    assert all(job.parameters["reconciliation_kind"] == "dflash_reconstruction_kl64" for job in replacements)
    assert {job.parameters["replaces_job_id"] for job in replacements} == {
        job.job_id for job in rows
    }


def test_soft_gate_resume_reopens_exact_e3b_rows_once(tmp_path: Path):
    state = StateStore(tmp_path)
    pilot = tuple(
        Job(f"pilot-{index}", "E3b-pilot", index, "static", "m", "DFLASH", "t")
        for index in range(20)
    )
    final = tuple(
        Job(f"final-{index}", "E3b-final", index, "static", "m", "DFLASH", "t")
        for index in range(132)
    )
    intentional = Job("intentional", "E3b-final", 132, "static", "m", "DFLASH", "t")
    state.add_jobs("E3b-pilot", pilot)
    state.skip_pending("E3b-pilot", "deployment width tuning failed for tts")
    state.add_jobs("E3b-final", (*final, intentional))
    state.skip_pending("E3b-final", "E3b-pilot did not complete")
    with state.connect() as connection:
        connection.execute(
            "UPDATE jobs SET error='intentional exploratory exclusion' WHERE job_id=?",
            (intentional.job_id,),
        )

    assert _reopen_soft_gate_e3b(state) == (20, 132)
    assert state.status_counts("E3b-pilot") == {"pending": 20}
    assert state.status_counts("E3b-final") == {"pending": 132, "skipped": 1}
    assert _reopen_soft_gate_e3b(state) == (20, 132)
    assert state.status_counts("E3b-pilot") == {"pending": 20}
    assert state.status_counts("E3b-final") == {"pending": 132, "skipped": 1}


def test_frozen_common_width_survives_late_e3a_reduction(tmp_path: Path):
    state = StateStore(tmp_path / "run")
    state.set_selection("deployment_widths", {"static": 4})
    _seed_e3a_static_deployment_width(state, 8)
    assert state.selection("deployment_widths", None) == {"static": 8}

    audit = {
        "version": 1,
        "common_deployment_width": 16,
        "deployment_widths": {
            "static": 16,
            "tts": 16,
            "l0_naive": 16,
            "lightcone": 16,
        },
    }
    _restore_soft_gate_width_selection(state, audit)
    _seed_e3a_static_deployment_width(state, 4)
    assert state.selection("deployment_widths", None) == audit["deployment_widths"]
    assert state.selection("deployment_widths_tuned", None) is True


def test_e3b_safety_rejection_retries_once_then_records_terminal(tmp_path: Path):
    state = StateStore(tmp_path / "run")
    state.set_selection("formal_soft_gate_resume_version", {"version": 1})
    rejected = Job(
        "e3b-rejected",
        "E3b-pilot",
        0,
        "tts",
        "m",
        "DFLASH",
        "t",
        parameters={"segment_index": 2},
    )
    runtime = Job(
        "e3b-runtime",
        "E3b-pilot",
        1,
        "tts",
        "m",
        "DFLASH",
        "t",
        parameters={"segment_index": 2},
    )
    parent = Job("e3b-parent", "E3b-pilot", 2, "tts", "m", "DFLASH", "t")
    state.add_internal_jobs((rejected, runtime), storage_node="E3b-pilot-segments")
    for job, error in (
        (rejected, "scientific safety failure: fallbacks=1"),
        (runtime, "connection refused"),
    ):
        output = tmp_path / job.job_id
        output.mkdir()
        attempt = state.start(job, (0,), output)
        state.fail(job.job_id, attempt, error, retry=False)

    assert _records_scientific_rejection(rejected) is True
    assert _records_scientific_rejection(parent) is False
    assert _repair_e3b_scientific_rejections(state) == 1
    assert {job.job_id for job in state.pending_jobs("E3b-pilot-segments")} == {
        rejected.job_id
    }
    assert state.failed_attempts(rejected.job_id) == 1
    assert _repair_e3b_scientific_rejections(state) == 0


@pytest.mark.parametrize("backend,expected", [("NONE", (4.0, 4)), ("DFLASH", (4.0, 4)), ("DSPARK", (3.0, 8))])
def test_e5_trace_reference_shares_target_only_and_matches_static_panel(monkeypatch, backend, expected):
    parents = materialize("E5-pilot")
    job = next(j for j in parents if j.backend == backend
               and j.method in {"target_only", "static"})
    target = next(j for j in parents if j.method == "target_only")
    dflash = next(j for j in parents if j.method == "static" and j.backend == "DFLASH")
    dspark = next(j for j in parents if j.method == "static" and j.backend == "DSPARK")
    def row(j, concurrency, rate, capacity=True):
        return (replace(j, load=f"closed_loop_c{concurrency}").to_dict(),
                {"capacity_feasible": capacity, "hard_feasible": capacity,
                 "request_rate": rate, "slo_pass": False})
    rows = [row(target, 2, 1.0), row(dflash, 4, 4.0), row(dspark, 8, 3.0),
            row(dflash, 256, 999.0, False), row(dspark, 256, 999.0, False)]
    for change in ({"block": 99}, {"model": "other"}, {"task": "other"}, {"context": 8192}):
        rows.append(row(replace(target, **change), 128, 999.0))
    rows.append(row(replace(target, parameters={**target.parameters, "topology": "tp2_dp1"}), 128, 999.0))
    monkeypatch.setattr("lightcone_spec.runner._metric_rows",
                        lambda state, node: rows if node.endswith("-segments") else [])
    assert _e5_reference(None, job) == expected
    # Sharing Target-only does not allow a missing/rejected Static panel.
    rows[:] = [row(target, 2, 1.0)]
    with pytest.raises(ScientificFailure, match="capacity-feasible Target-only/Static"):
        _e5_reference(None, job)
