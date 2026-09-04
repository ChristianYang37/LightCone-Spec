import gzip
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

import lightcone_spec.runner as runner
from lightcone_spec.metrics import (
    SAFETY_COUNTERS,
    benjamini_hochberg,
    block_bootstrap_interval,
    committed_goodput,
    derive_feasibility_semantics,
    hierarchical_request_interval,
    holm_decisions,
    normalize_attempt_semantics,
    paired_bca_interval,
    paired_block_statistics,
    paired_relative_bca_interval,
    per_user_generation_speed,
    summarize_attempts,
    validate_scientific_metrics,
)
from lightcone_spec.protocol import Job, materialize
from lightcone_spec.runner import _confirmatory_holm, _natural_spline_fit
from lightcone_spec.state import StateStore


def test_goodput_bootstrap_and_holm():
    assert committed_goodput(200, 4.0) == 50.0
    estimate, low, high = paired_bca_interval([2, 3, 4, 5], [1, 1, 2, 3], resamples=500, seed=0)
    assert low <= estimate <= high
    relative = paired_relative_bca_interval(
        [101, 102, 103, 104], [100, 100, 100, 100], resamples=500
    )
    assert relative[1] <= relative[0] <= relative[2]
    assert holm_decisions([0.001, 0.02, 0.8]) == (True, True, False)


def test_native_per_user_speed_and_historical_tts_concurrency(tmp_path):
    requests = [
        {"native_token_timestamps_ns": [1_000_000_000, 1_010_000_000, 1_020_000_000]},
        {"native_token_timestamps_ns": [2_000_000_000, 2_020_000_000]},
    ]
    assert per_user_generation_speed(requests) == pytest.approx(75.0)
    with gzip.open(tmp_path / "requests.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in requests:
            stream.write(json.dumps(row) + "\n")
    config, metrics = normalize_attempt_semantics(
        {"method": "tts", "load": "c2", "parameters": {}},
        {"goodput": 200.0, "per_user_generation_speed": 100.0},
        tmp_path,
    )
    assert config["parameters"]["declared_concurrency"] == 2
    assert config["parameters"]["dispatcher_concurrency"] == 1
    assert metrics["effective_load"] == "c1"
    assert metrics["per_user_generation_speed"] == pytest.approx(75.0)
    raw_metrics = {"goodput": 200.0, "per_user_generation_speed": 100.0}
    (tmp_path / "config.json").write_text(
        json.dumps({"method": "tts", "load": "c2", "parameters": {}})
    )
    (tmp_path / "metrics.json").write_text(json.dumps(raw_metrics))
    summary = summarize_attempts([tmp_path], tmp_path / "summary")
    assert summary.iloc[0]["per_user_generation_speed"] == pytest.approx(75.0)
    assert json.loads((tmp_path / "metrics.json").read_text()) == raw_metrics


def test_tts_source_policy_kl_has_zero_one_step_gradient():
    source_logits = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
    teacher = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)
    gradients = []
    for coefficient in (0.0, 0.1, 1.0, 10.0):
        logits = source_logits.clone().requires_grad_(True)
        source = torch.softmax(source_logits, dim=-1)
        target = torch.softmax(teacher, dim=-1)
        log_q = torch.log_softmax(logits, dim=-1)
        distillation = torch.sum(target * (torch.log(target) - log_q))
        proximal = torch.sum(source * (torch.log(source) - log_q))
        (distillation + coefficient * proximal).backward()
        gradients.append(logits.grad.clone())
    assert all(
        torch.allclose(gradients[0], gradient, atol=1e-14, rtol=0) for gradient in gradients[1:]
    )


def test_dspark_sequential_temperature_scaling_is_positionwise():
    sequences = []
    for offset in range(24):
        probabilities = np.asarray([0.55 + 0.01 * ((offset + pos) % 4) for pos in range(7)])
        outcomes = np.asarray([float((offset + pos) % 3 != 0) for pos in range(7)])
        sequences.append((probabilities, outcomes))
    temperatures = runner._fit_sequential_confidence_temperatures(sequences)
    assert len(temperatures) == 7
    assert all(0.25 <= value <= 4.0 for value in temperatures)
    diagnostics = runner._threshold_replay(sequences, temperatures)
    assert [row["threshold"] for row in diagnostics] == [value / 10 for value in range(10)]
    assert diagnostics[0]["acceptance_rate"] == 1.0


def test_tts_recipe_groups_confirmation_stimuli(monkeypatch):
    rows = []
    for block in range(4):
        rows.append(
            (
                {
                    "parameters": {
                        "learning_rate": 1e-4,
                        "stride": 50,
                        "workload": "tts_calibration_confirmation",
                        "confirmation_block": block,
                        "stimulus_id": f"confirmation-block-{block}",
                    }
                },
                {
                    "goodput": 100.0 + block,
                    "peak_hbm_bytes": 10,
                    "itl_p99_ms": 5.0,
                },
            )
        )

    monkeypatch.setattr(
        runner,
        "_metric_rows",
        lambda state, node: rows if node == "TTS-Cal-confirmation" else [],
    )
    recipe = runner._select_tts_recipe(None)
    assert recipe == {"learning_rate": 1e-4, "stride": 50}


def test_tts_s10_confirmation_uses_registered_tie_break(tmp_path, monkeypatch):
    rows = []
    for learning_rate, goodput, accepted in (
        (3e-5, 100.0, 300.0),
        (1e-4, 100.5, 200.0),
    ):
        for block in range(4):
            rows.append(
                (
                    {
                        "block": block,
                        "parameters": {"learning_rate": learning_rate, "stride": 10},
                    },
                    {
                        "goodput": goodput,
                        "accepted_drafts": accepted,
                        "target_calls": 100,
                        "itl_p99_ms": 10.0,
                        "updates_published": 1,
                        **{counter: 0 for counter in SAFETY_COUNTERS},
                    },
                )
            )
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)
    state = StateStore(tmp_path)
    recipe = runner._select_tts_s10_recipe(state)
    assert recipe["stride"] == 10
    assert recipe["learning_rate"] == 3e-5
    audit = json.loads(
        (tmp_path / "stages/TTS-S10-confirmation/selection_audit.json").read_text()
    )
    assert audit["selected_learning_rate"] == 3e-5


def test_s10_reconciliation_has_exact_registered_replacement_budget(tmp_path):
    state = StateStore(tmp_path)
    for node in ("E1", "E2-r0", "E2-r1", "E2-r2", "E2-r3", "E4-screen"):
        state.add_jobs(node, materialize(node))
    dynamic_source = replace(
        materialize("E2-r1")[0],
        job_id="E2-r1__dynamic-nag-rank1-last1-constant",
        parameters={
            **materialize("E2-r1")[0].parameters,
            "parameterization": "lora",
            "rank": 1,
            "scope": "last1",
            "optimizer": "nag",
            "learning_rate": 3e-5,
            "schedule": "constant",
        },
    )
    state.add_internal_jobs((dynamic_source,), storage_node="E2-r1")
    repairs = runner._s10_reconciliation_jobs(state)
    assert len(repairs) == 19
    assert (
        sum(job.parameters["reconciliation_kind"] == "formal_stride" for job in repairs)
        == 14
    )
    assert (
        sum(
            job.parameters["reconciliation_kind"] == "masked_logit_reconstruction"
            for job in repairs
        )
        == 5
    )
    assert len({job.parameters["replaces_job_id"] for job in repairs}) == 19


def test_bugfix_reconciliation_has_exact_152_cell_budget(tmp_path):
    state = StateStore(tmp_path)
    geometries = [
        {"parameterization": "lora", "rank": rank, "scope": "last1"}
        for rank in (1, 8)
    ]
    state.add_jobs(
        "E2-r0",
        materialize("E2-r0", e2_rows=runner.e2_candidates(geometries)),
    )
    legacy_e1a = tuple(
        Job(
            job_id=f"legacy-e1a-{index}",
            node="E1a",
            ordinal=index,
            method="lightcone_candidate",
            model="Qwen/Qwen3-8B",
            backend="DSPARK",
            task="CalibrationMix",
            parameters={
                "workload": "confidence_calibration",
                "confidence_loss_weight": weight,
                "segments": [
                    {"confidence_threshold": threshold / 10}
                    for threshold in range(10)
                ],
            },
        )
        for index, weight in enumerate((0.05, 0.1, 0.25, 0.5, 1.0))
    )
    state.add_jobs("E1a", legacy_e1a)
    state.add_jobs("TTS-Cal", materialize("TTS-Cal"))
    e3a_parent = materialize("E3a")[111]
    state.add_internal_jobs(
        runner._segment_jobs(e3a_parent),
        storage_node="E3a-segments",
    )
    repairs = runner._bugfix_reconciliation_jobs(state)
    assert len(repairs) == 100
    assert sum(len(job.parameters.get("segments", [])) or 1 for job in repairs) == 145
    reasons = [job.parameters["reconciliation_kind"] for job in repairs]
    assert reasons.count("e2_optimizer_or_cosine_horizon") == 90
    assert reasons.count("e1a_native_confidence_calibration") == 5
    assert reasons.count("screening_runtime_error_classification") == 2
    assert reasons.count("pre_reconstruction_stride1") == 3
    assert 145 + 7 == 152


def test_formal_replacement_excludes_old_attempt_without_overwriting_it(tmp_path):
    state = StateStore(tmp_path)
    source = next(job for job in materialize("E1") if job.method == "tts")
    state.add_jobs("E1", (source,))
    old_dir = tmp_path / "attempt-01"
    old_dir.mkdir()
    old_attempt = state.start(source, (0,), old_dir)
    (old_dir / "config.json").write_text(json.dumps(source.to_dict()))
    (old_dir / "metrics.json").write_text(json.dumps({"goodput": 1.0}))
    state.complete(source.job_id, old_attempt)

    replacement = replace(
        source,
        job_id=f"s10-repair__{source.job_id}",
        node="S10-reconciliation",
        parameters={
            **source.parameters,
            "source_node": "E1",
            "replaces_job_id": source.job_id,
            "reconciliation_kind": "formal_stride",
        },
    )
    state.add_internal_jobs((replacement,))
    new_dir = tmp_path / "attempt-02"
    new_dir.mkdir()
    new_attempt = state.start(replacement, (0,), new_dir)
    (new_dir / "config.json").write_text(json.dumps(replacement.to_dict()))
    (new_dir / "metrics.json").write_text(json.dumps({"goodput": 2.0}))
    state.complete(replacement.job_id, new_attempt)
    state.set_selection("formal_evidence_exclusions", [source.job_id])

    rows = runner._metric_rows(state, "E1")
    assert len(rows) == 1
    assert rows[0][0]["node"] == "E1"
    assert rows[0][1]["goodput"] == 2.0
    assert json.loads((old_dir / "metrics.json").read_text()) == {"goodput": 1.0}


def test_metric_rows_deduplicates_bundled_parent_and_child_storage(tmp_path):
    state = StateStore(tmp_path)
    source = materialize("E1a")[0]
    parent = replace(
        source,
        job_id="bugfix-parent",
        node="bugfix-reconciliation-v1",
        parameters={
            **source.parameters,
            "source_node": "E1a",
            "replaces_job_id": source.job_id,
            "segments": [{"confidence_threshold": 0.0}],
        },
    )
    child = runner._segment_jobs(parent)[0]
    state.add_internal_jobs((parent,), storage_node="bugfix-reconciliation-v1")
    state.add_internal_jobs((child,), storage_node="bugfix-reconciliation-v1-segments")

    child_dir = tmp_path / "child" / "attempt-01"
    child_dir.mkdir(parents=True)
    child_attempt = state.start(child, (0,), child_dir)
    (child_dir / "config.json").write_text(json.dumps(child.to_dict()))
    (child_dir / "metrics.json").write_text(json.dumps({"goodput": 2.0}))
    state.complete(child.job_id, child_attempt)

    parent_dir = tmp_path / "parent" / "attempt-01"
    parent_dir.mkdir(parents=True)
    parent_attempt = state.start(parent, (0,), parent_dir)
    (parent_dir / "config.json").write_text(json.dumps(parent.to_dict()))
    (parent_dir / "metrics.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "config": child.to_dict(),
                        "metrics": {"goodput": 2.0},
                        "attempt_dir": str(child_dir),
                    }
                ]
            }
        )
    )
    state.complete(parent.job_id, parent_attempt)

    rows = runner._metric_rows(state, "E1a")
    assert len(rows) == 1
    assert rows[0][0]["job_id"] == child.job_id
    assert rows[0][1]["goodput"] == 2.0


def test_deployment_width_selector_requires_hard_feasible_common_width(monkeypatch):
    rows = []
    regimes = (
        "long_input_short_output",
        "short_input_long_generation",
        "multi_turn_shared_prefix",
    )
    for method in ("static", "tts", "l0_naive", "lightcone"):
        for regime in regimes:
            rows.append(
                (
                    {"method": method, "width": 4, "parameters": {"regime": regime}},
                    {
                        "slo_pass": False,
                        "hard_feasible": method != "tts",
                        "goodput": 1.0,
                        "peak_hbm_bytes": 1,
                    },
                )
            )
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)
    with pytest.raises(runner.ScientificFailure, match="no hard-feasible common width"):
        runner._select_deployment_widths(object())


def test_deployment_width_selector_uses_report_only_slo_and_common_goodput(monkeypatch):
    rows = []
    regimes = (
        "long_input_short_output",
        "short_input_long_generation",
        "multi_turn_shared_prefix",
    )
    for method in ("static", "tts", "l0_naive", "lightcone"):
        for regime in regimes:
            for width, goodput in ((4, 100.0), (8, 120.0), (16, 110.0)):
                rows.append(
                    (
                        {
                            "method": method,
                            "width": width,
                            "parameters": {"regime": regime},
                        },
                        {
                            "slo_pass": width != 8,
                            "hard_feasible": True,
                            "goodput": goodput,
                            "peak_hbm_bytes": width,
                        },
                    )
                )
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)
    assert runner._select_deployment_widths(object()) == {
        method: 8 for method in ("static", "tts", "l0_naive", "lightcone")
    }


def test_feasibility_semantics_separate_slo_and_capacity():
    complete = {
        # Legacy rows sometimes copied the SLO decision into ``feasible``.
        # The v2 loader derives hard feasibility from requests and safety.
        "feasible": False,
        "slo_pass": False,
        "request_outcomes": {"offered": 2, "completed": 2},
        **{counter: 0 for counter in SAFETY_COUNTERS},
    }
    ordinary = derive_feasibility_semantics(
        {"node": "E3b-pilot", "parameters": {}}, complete
    )
    assert ordinary == {
        "hard_feasible": True,
        "capacity_feasible": "N/A",
        "slo_semantics": "report_only_v2",
    }
    incomplete = derive_feasibility_semantics(
        {"node": "E3a", "parameters": {}},
        {
            **complete,
            "request_outcomes": {"offered": 2, "completed": 1, "timed_out": 1},
        },
    )
    assert incomplete["hard_feasible"] is False
    assert incomplete["capacity_feasible"] is False


def test_activity_trace_proxy_summarizes_kernel_overlap(tmp_path):
    trace = {
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "k1", "ts": 0, "dur": 10},
            {"ph": "X", "cat": "kernel", "name": "k2", "ts": 5, "dur": 10},
            {"ph": "X", "cat": "gpu_memcpy", "name": "memcpy", "ts": 20, "dur": 2},
            {"ph": "X", "cat": "cpu_op", "name": "op", "ts": 0, "dur": 4},
        ]
    }
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    summary = runner._activity_trace_summary(tmp_path)
    assert summary["kernel_count"] == 2
    assert summary["kernel_time_us"] == 20
    assert summary["gpu_busy_time_us"] == 15
    assert summary["stream_overlap_ratio"] == pytest.approx(0.25)
    assert summary["memcpy_count"] == 1


def test_nsys_activity_csv_parser_preserves_timing_totals():
    summary = runner._nsys_csv_summary(
        'Time (%),Total Time (ns),Instances,Avg (ns),Name\n'
        '60.0,1200,2,600,"kernel_a"\n'
        '40.0,800,4,200,"kernel_b"\n'
    )
    assert summary["row_count"] == 2
    assert summary["numeric_totals"]["Total Time (ns)"] == 2000
    assert summary["numeric_totals"]["Instances"] == 6


def test_nsys_activity_csv_parser_rejects_empty_payload():
    assert runner._nsys_csv_summary("NOTICE: no report rows") == {
        "row_count": 0,
        "numeric_totals": {},
    }


def test_e2_ranking_uses_static_goodput_and_tts_native_user_speed(monkeypatch):
    rows = [
        ({"method": "static", "parameters": {}}, {"goodput": 100.0}),
        (
            {"method": "tts", "parameters": {}},
            {"goodput": 10.0, "per_user_generation_speed": 100.0},
        ),
        (
            {"method": "lightcone_candidate", "parameters": {"name": "aggregate"}},
            {
                "goodput": 120.0,
                "per_user_generation_speed": 80.0,
                "peak_hbm_bytes": 1,
                "itl_p99_ms": 1.0,
                "exposed_update_ms": 1.0,
            },
        ),
        (
            {"method": "lightcone_candidate", "parameters": {"name": "balanced"}},
            {
                "goodput": 90.0,
                "per_user_generation_speed": 110.0,
                "peak_hbm_bytes": 1,
                "itl_p99_ms": 1.0,
                "exposed_update_ms": 1.0,
            },
        ),
    ]
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)
    assert runner._rank_e2_candidates(object(), "E2-r0", 1) == [{"name": "balanced"}]


def test_e2_audit_preserves_existing_order_when_scientific_set_matches(tmp_path):
    state = StateStore(tmp_path)
    state.set_selection("e2_round_0", [{"rank": 1}, {"rank": 8}])
    corrected = [{"rank": 8, "metric_semantics": "per_request_native_v2"}, {"rank": 1}]
    assert runner._preserve_or_audit_e2_selection(
        state, "E2-r0", "e2_round_0", corrected
    ) == [{"rank": 1}, {"rank": 8}]
    audit = json.loads((tmp_path / "stages/E2-r0/concurrency_metric_audit.json").read_text())
    assert audit["same_scientific_set"] is True
    assert audit["action"] == "preserved_existing_order"


def test_e5_frontier_rejects_nominal_concurrency_speed_fallback(monkeypatch):
    class State:
        def selection(self, name, default=None):
            return "static" if name == "e5_operational_baseline" else default

    monkeypatch.setattr(
        runner,
        "_metric_rows",
        lambda state, node: [
            (
                {
                    "method": "static",
                    "backend": "DFLASH",
                    "block": 0,
                    "load": "closed_loop_c2",
                    "parameters": {
                        "workload": "primary_serving_frontier",
                        "topology": "tp1_dp1",
                    },
                },
                {"goodput": 200.0},
            )
        ],
    )
    with pytest.raises(runner.ScientificFailure, match="native per-request"):
        runner._e5_frontier_statistic(State())


def test_e5_frontier_ignores_multigpu_transfer_rows(monkeypatch):
    class State:
        def selection(self, name, default=None):
            return "static" if name == "e5_operational_baseline" else default

    monkeypatch.setattr(
        runner,
        "_metric_rows",
        lambda state, node: [
            (
                {
                    "method": "static",
                    "backend": "DFLASH",
                    "block": 0,
                    "load": "closed_loop_c32",
                    "parameters": {
                        "workload": "multigpu_serving_transfer",
                        "topology": "tp2_dp1",
                    },
                },
                {"goodput": 100.0},
            )
        ],
    )
    assert runner._e5_frontier_statistic(State()) is None


def test_e5_topology_transfer_is_paired_within_backend_topology_and_load(monkeypatch):
    rows = []
    for backend in ("DFLASH", "DSPARK"):
        for topology in ("tp2_dp1", "two_replica_tp1_dp2"):
            for block in range(6):
                for method, scale in (("static", 1.0), ("lightcone", 1.1)):
                    rows.append(
                        (
                            {
                                "job_id": f"{backend}-{topology}-{block}-{method}",
                                "method": method,
                                "backend": backend,
                                "block": block,
                                "load": "closed_loop_c32",
                                "parameters": {
                                    "workload": "multigpu_serving_transfer",
                                    "topology": topology,
                                    "registered_load": "closed_loop_c32",
                                    "registered_concurrency_scope": "system",
                                },
                            },
                            {
                                "hard_feasible": True,
                                "goodput": 100.0 * scale,
                                "per_user_generation_speed": 10.0 * scale,
                                "accepted_drafts": 50.0 * scale,
                                "verified_drafts": 100.0,
                                "verification_waste": 5.0 / scale,
                                "ttft_p50_ms": 20.0 / scale,
                                "itl_p99_ms": 30.0 / scale,
                                "peak_hbm_bytes": 1000.0,
                                "kv_capacity": 100.0,
                                "execution_gpu_count": 2,
                            },
                        )
                    )
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)
    reduced = runner._e5_topology_transfer(object(), "E5-final")
    assert len(reduced["rows"]) == 48
    assert {
        (row["backend"], row["topology"], row["load"])
        for row in reduced["paired_statistics"]
    } == {
        (backend, topology, "closed_loop_c32")
        for backend in ("DFLASH", "DSPARK")
        for topology in ("tp2_dp1", "two_replica_tp1_dp2")
    }
    assert all(row["blocks"] == list(range(6)) for row in reduced["paired_statistics"])


def test_final_stage_requires_completed_pilot():
    class State:
        def stage_status(self, node):
            return {"preflight": "completed", "E5-pilot": "skipped"}.get(node)

        def selection(self, name, default=None):
            return default

    assert runner._dependency_reason(None, State(), "E5-final") == "E5-pilot did not complete"


def test_e5_pilot_can_run_without_dspark_recipe():
    class State:
        def stage_status(self, node):
            return "completed" if node == "preflight" else None

        def selection(self, name, default=None):
            return {"lightcone_recipe": {"rank": 8}}.get(name, default)

    assert runner._dependency_reason(None, State(), "E5-pilot") is None


def test_explicit_adaptive_support_boundary_is_compatibility_infeasible(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "ValueError: DFlash updates currently require the base DFlashDraftModel; "
        "specialized variants fail closed\n"
    )
    assert runner._adaptive_probe_incompatible(RuntimeError("startup failed"), log)
    assert runner._adaptive_probe_incompatible(
        RuntimeError(
            "Cannot find model module. 'Qwen3Eagle3Model' is not a registered model "
            "and 'AutoModel' is not present in the model config's 'auto_map'"
        )
    )
    assert runner._adaptive_probe_incompatible(
        RuntimeError(
            "Cannot find model module. 'Gemma4DSparkModel' is not a registered model "
            "and 'AutoModel' is not present in the model config's 'auto_map'"
        )
    )
    assert not runner._adaptive_probe_incompatible(ConnectionError("connection refused"))


def test_only_serving_load_protocols_allow_transparent_prompt_replay():
    e5 = materialize("E5-pilot")[0]
    e6 = runner._e6_load_jobs()[0]
    ordinary = materialize("E3a")[0]
    assert runner._allow_prompt_repeat(e5)
    assert runner._allow_prompt_repeat(e6)
    assert not runner._allow_prompt_repeat(ordinary)


def test_safety_metrics_reject_numerical_failures():
    metrics = {
        "committed_tokens": 100,
        "duration_seconds": 1.0,
        "goodput": 100.0,
        "peak_hbm_bytes": 1,
        "kv_capacity": 1,
        "itl_p99_ms": 1.0,
        **{name: int(name == "nonfinite_updates") for name in SAFETY_COUNTERS},
    }
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_scientific_metrics(metrics)
    with pytest.raises(ValueError):
        committed_goodput(1, math.nan)


def test_fdr_and_time_block_bootstrap():
    assert benjamini_hochberg([0.001, 0.02, 0.9]) == (True, True, False)
    point, low, high = block_bootstrap_interval([10, 11, 12, 13], resamples=500)
    assert low <= point <= high
    point, low, high = hierarchical_request_interval(
        {block: [(10 + block, 1.0), (20 + block, 2.0)] for block in range(4)},
        resamples=500,
    )
    assert low <= point <= high


def test_final_block_statistics_are_paired():
    rows = []
    for block in range(4, 16):
        for method, goodput in (("static", 100.0), ("lightcone", 104.0 + block / 100)):
            rows.append(
                (
                    {
                        "method": method,
                        "model": "m",
                        "backend": "DFLASH",
                        "task": "t",
                        "context": 40928,
                        "load": "c4",
                        "block": block,
                        "parameters": {"regime": "long"},
                    },
                    {"goodput": goodput, "request_count": 10},
                )
            )
    result = paired_block_statistics(rows)
    focal = next(row for row in result if row["baseline"] == "static")
    assert focal["candidate"] == "lightcone"
    assert focal["blocks"] == list(range(4, 16))
    assert focal["ci95_relative_low"] > 0
    assert focal["reducer"] == "paired_log_goodput_bca"
    assert focal["holm_reject"] is None


def test_pairing_separates_effective_concurrency():
    rows = []
    for block in range(4):
        for method, effective_load in (("static", "c2"), ("lightcone", "c1")):
            rows.append(
                (
                    {
                        "method": method,
                        "model": "m",
                        "backend": "DFLASH",
                        "task": "t",
                        "context": 40928,
                        "load": "c2",
                        "block": block,
                        "parameters": {
                            "regime": "long",
                            "effective_load": effective_load,
                        },
                    },
                    {"goodput": 100.0, "request_count": 1},
                )
            )
    assert paired_block_statistics(rows) == []


def test_holm_combines_only_the_three_preregistered_hypotheses(tmp_path):
    state = StateStore(tmp_path)
    path = tmp_path / "stages" / "E3b-final"
    path.mkdir(parents=True)
    rows = [
        {
            "candidate": "lightcone",
            "baseline": baseline,
            "workload": "primary_long_history",
            "context": 32768,
            "p_value": p_value,
        }
        for baseline, p_value in (("tts", 0.001), ("operational_baseline", 0.02))
    ]
    (path / "statistics.json").write_text(json.dumps(rows))
    combined = _confirmatory_holm(
        state,
        {
            "hypothesis": "H3",
            "candidate": "lightcone",
            "baseline": "operational_baseline",
            "metric": "maximum_slo_feasible_rate",
            "p_value": 0.8,
        },
    )
    assert [row["hypothesis"] for row in combined] == ["H1", "H2", "H3"]
    assert [row["holm_reject"] for row in combined] == [True, True, False]


def test_pairing_uses_block_stimulus_but_ignores_runtime_backend_and_width():
    rows = []
    for block in range(4):
        stimulus = f"shared-block-{block}"
        for method, backend, width, goodput in (
            ("target_only", "NONE", None, 100.0),
            ("lightcone", "DFLASH", 16, 104.0 + block / 100),
        ):
            rows.append(
                (
                    {
                        "method": method,
                        "model": "m",
                        "backend": backend,
                        "task": "t",
                        "context": 40928,
                        "load": "c4",
                        "width": width,
                        "block": block,
                        "parameters": {
                            "comparison_backend": "DFLASH",
                            "width_panel": "deployment_optimal",
                            "stimulus_id": stimulus,
                        },
                    },
                    {"goodput": goodput, "request_count": 10},
                )
            )
    result = paired_block_statistics(rows)
    assert [(row["candidate"], row["baseline"]) for row in result] == [("lightcone", "target_only")]


def test_natural_spline_uses_fixed_interior_knots_and_natural_boundaries():
    contexts = pytest.importorskip("numpy").array(
        [1024, 2048, 4096, 8192, 16384, 24576, 32768, 40928], dtype=float
    )
    x = pytest.importorskip("numpy").log(contexts)
    y = 0.2 * x + 0.01 * x**2
    fitted, elasticity, curvature = _natural_spline_fit(x, y, x)
    assert len(fitted) == len(contexts)
    assert len(elasticity) == len(contexts)
    assert abs(curvature[0]) < 1e-8
    assert abs(curvature[-1]) < 1e-8


def test_context_spline_ignores_infeasible_rows_without_goodput(monkeypatch):
    config = {
        "method": "static",
        "context": 40928,
        "load": "c64",
        "parameters": {"regime": "long_input_short_output"},
    }
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: [(config, {"feasible": False})])

    class EmptyState:
        def completed_attempt_dirs(self, node):
            return ()

    assert runner._context_splines(EmptyState(), "E3a") == []


def test_context_spline_skips_unbracketed_fixed_knots(monkeypatch):
    rows = [
        (
            {
                "method": "static",
                "context": context,
                "load": "c1",
                "parameters": {"regime": "long_input_short_output"},
            },
            {"goodput": 10.0},
        )
        for context in (4096, 16384, 32768, 40928)
    ]
    monkeypatch.setattr(runner, "_metric_rows", lambda state, node: rows)

    class EmptyState:
        def completed_attempt_dirs(self, node):
            return ()

    assert runner._context_splines(EmptyState(), "E3a") == []


def test_attempt_summary_serializes_mixed_nested_parquet_columns(tmp_path):
    pytest.importorskip("pyarrow")
    attempts = []
    for index, layout in enumerate(([["weight", [4, 4]]], "N/A")):
        directory = tmp_path / f"attempt-{index:02d}"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"method": "static"}))
        (directory / "metrics.json").write_text(json.dumps({"parameter_layout": layout}))
        attempts.append(directory)
    output = tmp_path / "summary"
    summarize_attempts(attempts, output)
    assert (output / "summary.csv").is_file()
    assert (output / "summary.parquet").is_file()
