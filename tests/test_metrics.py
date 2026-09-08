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
    four_block_log_ratio_statistics,
    four_block_panel_statistics,
    hierarchical_request_interval,
    holm_decisions,
    mechanism_position_summary,
    normalize_attempt_semantics,
    paired_bca_interval,
    paired_block_statistics,
    paired_relative_bca_interval,
    per_user_generation_speed,
    position_acceptance_metrics,
    summarize_attempts,
    validate_scientific_metrics,
)
from lightcone_spec.protocol import Job, materialize
from lightcone_spec.runner import _confirmatory_holm, _natural_spline_fit
from lightcone_spec.state import StateStore


def test_quick_window_counts_cutoff_native_and_event_integrity():
    from lightcone_spec.quick_tuning import WindowEvidence

    now = [10.]
    evidence = WindowEvidence(30, clock=lambda: now[0])

    def event(sequence, ids, new, stamps=None):
        meta = {"id": "r", "completion_tokens": len(ids)}
        if stamps is not None:
            meta["native_token_timestamp_events"] = [
                {"token_index": i, "token_id": token, "committed_ns": stamp}
                for i, (token, stamp) in enumerate(zip(ids, stamps, strict=True))]
        return {"request_id": "r", "sequence": sequence, "token_ids": new,
                "chunk": {"output_ids": ids, "meta_info": meta}}

    now[0] = 12
    evidence.observe(event(1, [3, 4], [3, 4], [100, 1_000_000_100]))
    now[0] = 41
    evidence.observe(event(2, [3, 4, 5], [5], [100, 1_000_000_100, 2_000_000_100]))
    now[0] = 43
    evidence.observe(event(3, [3, 4, 5, 6], [6]))
    row = evidence.report()
    assert row["observed_committed_tokens"] == 3
    assert row["window_goodput"] == 3 / 32
    assert row["decode_window_speed"] == .1
    assert row["per_user_generation_speed"] == 1
    assert row["events"][-1]["inside_window"] is False
    with pytest.raises(ValueError, match="duplicate or missing"):
        evidence.observe(event(3, [3, 4, 5, 6], []))
    with pytest.raises(ValueError, match="identity/count"):
        evidence.observe(event(4, [3, 9], [9]))
    evidence.requests["r"]["native"] = None
    assert evidence.report()["per_user_generation_speed"] is None


def test_quick_pair_decision_is_bounded_and_not_request_pseudoreplication():
    from lightcone_spec.quick_tuning import paired_decision

    def rows(gains, repeats):
        return [{"domain": domain, "repeat": repeat, "variant": variant,
                 "comparison_key": {"domain": domain, "repeat": repeat, "tp": 2},
                 "status": "budget_end", "window_goodput": 100 * (1 + gains[domain]) if variant == "new" else 100}
                for repeat in range(repeats) for domain in ("Code", "Math") for variant in ("old", "new")]

    assert paired_decision(rows({"Code": .05, "Math": .04}, 1))["decision"] == "repeat_promising"
    assert paired_decision(rows({"Code": -.05, "Math": -.04}, 1))["decision"] == "reject"
    assert paired_decision(rows({"Code": .02, "Math": .02}, 3))["decision"] == "candidate_for_long_validation"
    assert paired_decision(rows({"Code": -.001, "Math": .05}, 3))["decision"] == "keep_old"
    assert paired_decision(rows({"Code": .001, "Math": .001}, 3))["decision"] == "keep_old"
    assert paired_decision(rows({"Code": .02, "Math": .02}, 1)[:-1])["decision"] == "await_pairs"
    bad = rows({"Code": .02, "Math": .02}, 1)
    bad[1]["comparison_key"] = {"tp": 1}
    with pytest.raises(ValueError, match="mismatched"):
        paired_decision(bad)


def test_timing_union_does_not_sum_nested_streams_or_ranks(tmp_path):
    from lightcone_spec.timing_audit import overlap_ms, summarize_timing, write_timing_report

    assert overlap_ms([(0, 10), (2, 8)], [(5, 15)]) == 5
    snapshots = []
    for rank in (0, 1):
        snapshots.append({"rank": rank, "valid": True, "dropped_records": 0, "pending_events": 0,
            "records": [{"name": name, "rank": rank, "clock": "cuda", "lane": lane,
                         "start_ms": start, "end_ms": end}
                        for name, lane, start, end in (("forward", "main", 0, 10),
                            ("nested", "main", 2, 8), ("train<script>", "side", 5, 15))]})
    report = summarize_timing(snapshots, expected_ranks=(0, 1), wall_seconds=.02, tokens=100)
    assert [r["main_interval_union_ms"] for r in report["rank_reports"]] == [10, 10]
    assert [r["event_interval_overlap_ms"] for r in report["rank_reports"]] == [5, 5]
    assert report["main_blocked_ms"] is None and report["unattributed_wall_ms"] is None
    write_timing_report(report, tmp_path)
    assert "train&lt;script&gt;" in (tmp_path / "index.html").read_text()
    with pytest.raises(ValueError, match="exactly once"):
        summarize_timing(snapshots[:1], expected_ranks=(0, 1))
    snapshots[0]["dropped_records"] = 1
    with pytest.raises(ValueError, match="incomplete"):
        summarize_timing(snapshots, expected_ranks=(0, 1))


def test_timing_queue_never_waits_inside_round_and_nested_overflow_is_visible():
    from types import SimpleNamespace

    from lightcone_spec.timing_audit import TimingRecorder

    class Event:
        counter = 0
        waits = 0

        def __init__(self, **kwargs):
            self.ready = False

        def record(self):
            Event.counter += 1
            self.timestamp = Event.counter

        def query(self):
            return self.ready

        def synchronize(self):
            Event.waits += 1
            self.ready = True

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    cuda = SimpleNamespace(Event=Event, current_stream=lambda: SimpleNamespace(cuda_stream=1))
    recorder = TimingRecorder(cuda=cuda, max_pending=1)
    with recorder.span("outer", gpu=True):
        with recorder.span("inner", gpu=True):
            assert Event.waits == 0
    assert Event.waits == 0 and recorder.dropped == 1
    with recorder.span("request_reset"):
        pass
    snapshot = recorder.snapshot()
    assert Event.waits == 1 and not snapshot["valid"]
    assert snapshot["pending_high_water"] == 1
    assert {r["name"] for r in snapshot["records"]} == {"outer", "inner", "request_reset"}


def test_timing_boundary_retains_rank_evidence_without_profiling(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import lightcone_spec.timing_diagnostic as diagnostic

    monkeypatch.setattr(diagnostic, "_settings", {"mode": "off", "output_directory": str(tmp_path)})
    monkeypatch.setattr(diagnostic, "_recorder", None)
    monkeypatch.setattr(diagnostic, "_window", None)
    monkeypatch.setattr(diagnostic, "_installed", set())
    monkeypatch.setenv("LIGHTCONE_TIMING_AUDIT", json.dumps({"output_directory": str(tmp_path)}))

    class Scheduler:
        ps = SimpleNamespace(tp_rank=1, tp_size=2)

        def get_internal_state(self):
            return SimpleNamespace(internal_state={"speed_study_metrics": {"fallbacks": 0}})

    diagnostic._instrument(SimpleNamespace(__name__="sglang.srt.managers.scheduler", Scheduler=Scheduler))
    scheduler = Scheduler()
    diagnostic.checkpoint("begin", "cell-1")
    scheduler.get_internal_state()
    first = diagnostic._recorder
    scheduler.get_internal_state()
    assert first is diagnostic._recorder  # repeated observation cannot reset the cell
    with first.span("request_reset"):
        pass
    diagnostic.checkpoint("end", "cell-1")
    scheduler.get_internal_state()
    snapshot = json.loads((tmp_path / "rank-1-timing.json").read_text())
    assert snapshot["job_id"] == "cell-1" and snapshot["rank"] == 1
    assert snapshot["records"][0]["name"] == "request_reset"
    rows = [json.loads(line) for line in (tmp_path / "rank-1-metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["tp_rank"] == 1 and r["tp_size"] == 2 for r in rows)
    assert diagnostic._recorder is None


def test_target_verify_is_not_mislabeled_as_prefill():
    from types import SimpleNamespace

    from lightcone_spec.timing_diagnostic import _model_phase

    mode = SimpleNamespace(is_target_verify=lambda: True, is_extend=lambda: True)
    batch = SimpleNamespace(forward_mode=mode)
    assert _model_phase(SimpleNamespace(is_draft_worker=False), (batch,), {}) == "target_verification"
    assert _model_phase(SimpleNamespace(is_draft_worker=True), (batch,), {}) == "draft_forward"


def test_video_native_final_accounting_rejects_loss_duplicates_and_wrong_denominator():
    from copy import deepcopy

    from lightcone_spec.recording import validate_recording
    rows = [{"request_id": "r", "completion_tokens": 2, "output_ids": [8, 9],
             "stop_reason": "stop", "native_token_timestamps_ns": [100, 200]}]
    events = [{"request_id": "r", "sequence": 1, "elapsed_seconds": .5, "token_ids": [8, 9],
               "chunk": {"output_ids": [8, 9], "meta_info": {"finish_reason": {"type": "stop"}}}}]
    value = validate_recording(events, rows, 2., expected_requests=1)
    assert value["aggregate_tok_s"] == 1. and value["committed_tokens"] == 2
    for broken in (events * 2, [], [{**events[0], "sequence": 2}], [{**events[0], "token_ids": [8]}]):
        with pytest.raises(RuntimeError):
            validate_recording(broken, rows, 2., expected_requests=1)
    bad = deepcopy(rows)
    bad[0]["native_token_timestamps_ns"] = [200, 100]
    with pytest.raises(RuntimeError, match="native"):
        validate_recording(events, bad, 2., expected_requests=1)
    with pytest.raises(RuntimeError, match="duration"):
        validate_recording(events, rows, float("nan"), expected_requests=1)


def test_remaining_preview_qa_capacity_and_runtime_are_distinct(monkeypatch):
    import runpy
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    review = runpy.run_path(str(scripts / "validate_remaining_preview.py"))["review_case"]
    job = Job("qa", "E5-preview-v3", 0, "lightcone", "Qwen/Qwen3-8B", "DSPARK", "LiveCodeBench")
    metrics = {"hard_feasible": True, "rank_local_after": [{k: 0 for k in SAFETY_COUNTERS}],
               "resolved_stride": 1, "updates_published": 2}
    rows = [{"request_id": "r", "stop_reason": "length", "completion_tokens": 1,
             "output_ids": [1], "native_token_timestamps_ns": [100]}]
    assert review(metrics, rows, job) == "passed"
    capacity = {**metrics, "hard_feasible": False, "capacity_feasible": False,
                "request_outcomes": {"timed_out": 1}}
    assert review(capacity, rows, job) == "capacity_infeasible"
    capacity["request_outcomes"]["error"] = 1
    with pytest.raises(RuntimeError, match="runtime failure"):
        review(capacity, rows, job)
    metrics["rank_local_after"][0]["fallbacks"] = 1
    with pytest.raises(RuntimeError, match="safety"):
        review(metrics, rows, job)


def test_video_nvml_window_keeps_rank_peaks_separate(tmp_path):
    from lightcone_spec.recording import recording_nvml_peaks
    path = tmp_path / "measurement-gpu.csv"
    path.write_text("timestamp,index,memory_used_mb\nnow,0,10\nnow,1,20\nnext,0,30\nnext,1,15\n")
    metrics = recording_nvml_peaks(path, (0, 1))
    assert metrics["sum_nvml_rank_peak_bytes"] == 50 * 1024**2
    assert metrics["nvml_peak_hbm_bytes"] == 30 * 1024**2
    with pytest.raises(RuntimeError, match="unassigned"):
        recording_nvml_peaks(path, (0,))


def test_preview_four_block_effects_keep_missing_and_exclude_private_data(tmp_path):
    from lightcone_spec.preview import preview_summary
    from lightcone_spec.protocol import Job

    jobs, evidence = [], []
    for block in range(4):
        for method in ("static", "lightcone"):
            job = Job(job_id=f"preview-{block}-{method}", node="E5-preview-v1", ordinal=len(jobs),
                      model="Qwen/Qwen3-8B", backend="DSPARK", task="LiveCodeBench", method=method,
                      load="closed_loop_c8", block=block,
                      parameters={"panel": "preview_v1", "preview_panel": "serving",
                                  "execution_policy": "automatic_units_v3", "prompt": "PRIVATE"})
            jobs.append(job)
            evidence.append((job.to_dict(), {"hard_feasible": True, "goodput": 100 if method == "static" else 120,
                                             "target_calls": 10, "accepted_drafts": 30,
                                             "effective_load": "c8", "source_attempt_dir": "/PRIVATE"}))
    result = preview_summary(evidence, jobs, tmp_path)
    effect = next(e for e in result["effects"] if e["metric"] == "goodput")
    assert effect["ratio"] == pytest.approx(1.2)
    assert effect["ratio_ci95"] == pytest.approx([1.2, 1.2])
    assert all(row["metrics"]["accepted_drafts_per_target_call"] == 3 for row in result["rows"])
    assert "PRIVATE" not in (tmp_path / "preview.json").read_text()
    partial = preview_summary(evidence[:-1], jobs, tmp_path)
    assert partial["counts"]["UNMEASURED"] == 1
    assert all(e["status"] == "UNMEASURED" for e in partial["effects"])
    with pytest.raises(ValueError, match="duplicate"):
        preview_summary(evidence + evidence[:1], jobs, tmp_path)
    from lightcone_spec.preview import preview_eta

    for _, metrics in evidence:
        metrics.update(duration_seconds=60., session_startup_seconds=10.)
    eta = preview_eta(evidence, jobs, repetitions=20)
    assert eta["status"] == "estimated" and eta["remaining_leaves"] == 8
    # Both methods of each block consume the same device, never divided individually.
    assert eta["p50_seconds"] == 280.
    missing = preview_eta(evidence[:1], jobs, repetitions=20)
    assert missing["status"] == "UNMEASURED" and missing["p50_seconds"] is None


def test_preview_topology_mismatch_is_not_paired(tmp_path):
    from lightcone_spec.preview import preview_summary

    job = Job(job_id="wrong-tp", node="E5-preview-v1", ordinal=0, method="static",
              model="Qwen/Qwen3-8B", backend="DSPARK", task="LiveCodeBench", load="c1", block=0,
              parameters={"panel": "preview_v1", "preview_panel": "serving", "topology": "tp2_dp1"})
    wrong = replace(job, parameters={**job.parameters, "topology": "tp1_dp1"})
    with pytest.raises(ValueError, match="topology"):
        preview_summary([(wrong.to_dict(), {"hard_feasible": True})], [job], tmp_path)


def test_preview_eta_tp2_reserves_both_devices():
    from lightcone_spec.preview import preview_eta

    jobs = tuple(Job(job_id=f"tp2-{i}", node="Qwen38-preview-v1", ordinal=i,
                     method="static", model="Qwen/Qwen3.8-27B", backend="DFLASH",
                     task="LiveCodeBench", load="c1", block=i, gpu_count=2,
                     parameters={"panel": "preview_v1", "preview_panel": "qwen38_transfer",
                                 "topology": "tp2_dp1", "execution_policy": "automatic_units_v3"})
                 for i in range(4))
    evidence = [(j.to_dict(), {"hard_feasible": True, "duration_seconds": 60., "session_startup_seconds": 10.}) for j in jobs]
    assert preview_eta(evidence, jobs, repetitions=10)["p50_seconds"] == 280.


def test_video_composition_requires_submission_clock_and_matched_topology(tmp_path):
    import hashlib
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("compose_preview", Path(__file__).parents[1] / "scripts/compose_preview.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = []
    record = {"status": "completed", "errors": [], "video_sha256": hashlib.sha256(b"EXCLUDED SYNTHETIC TEST").hexdigest(), "alignment": {
        "status": "verified_clock_bounded", "submission_offset_seconds": 2., "clock_uncertainty_seconds": .02,
    }, "measurement": {"model": "Qwen/Qwen3.8-27B", "tp": 2, "dispatcher_concurrency": 8,
                       "input_tokens_per_request": 16384, "max_output_tokens": 1024}}
    for i in range(6):
        directory = tmp_path / str(i)
        directory.mkdir()
        record["measurement"]["label"] = str(i)
        (directory / "capture.json").write_text(json.dumps(record))
        paths.append(directory / "original.mp4")
        paths[-1].write_bytes(b"EXCLUDED SYNTHETIC TEST")
    offsets, bounds = module.submission_alignment(paths)
    assert offsets == [1.98] * 6 and bounds == [.02] * 6
    record["measurement"]["tp"] = 1
    paths[-1].with_name("capture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="share model, TP"):
        module.submission_alignment(paths)
    record["alignment"]["status"] = "UNMEASURED"
    paths[-1].with_name("capture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="submission-clock"):
        module.submission_alignment(paths)


def test_coverage_eta_scales_request_budget_without_reusing_unknown_output_costs():
    from lightcone_spec.coverage import request_budget_eta
    from lightcone_spec.protocol import source_coverage_jobs

    job = source_coverage_jobs()[0]
    evidence = [(job, {"hard_feasible": True, "duration_seconds": 100.,
                       "request_count": 10, "session_startup_seconds": 5.,
                       "memory_budget_policy": "method_peak_v1"})]
    args = dict(resources={job.job_id: 2}, repetitions=20)
    small = request_budget_eta((job,), evidence, request_counts={job.job_id: 10}, **args)
    large = request_budget_eta((job,), evidence, request_counts={job.job_id: 500}, **args)
    assert small["p50_seconds"] == 105 and large["p50_seconds"] == 5005
    legacy = {k: v for k, v in evidence[0][1].items() if k != "memory_budget_policy"}
    assert request_budget_eta((job,), [(job, legacy)],
                              request_counts={job.job_id: 10}, **args)["priced_leaves"] == 0
    other = replace(job, parameters={**job.parameters, "generation_tokens": 32768})
    unknown = request_budget_eta((other,), evidence, request_counts={job.job_id: 500}, **args)
    assert unknown["status"] == "UNMEASURED" and unknown["p50_seconds"] is None
    assert unknown["priced_leaves"] == 0
    excluded = replace(job, parameters={**job.parameters, "excluded_from_analysis": True})
    assert request_budget_eta((job,), [(excluded, evidence[0][1])],
                              request_counts={job.job_id: 500}, **args)["priced_leaves"] == 0
    new_execution = replace(job, parameters={**job.parameters, "execution_policy": "automatic_units_v3"})
    assert request_budget_eta((new_execution,), evidence,
                              request_counts={job.job_id: 500}, **args)["priced_leaves"] == 0


def test_execution_modes_do_not_form_a_paired_effect():
    base = {"block": 0, "model": "qwen", "backend": "DFLASH", "task": "math",
            "context": 100, "load": "c1", "parameters": {}}
    rows = [({**base, "method": "static"}, {"goodput": 100, "hard_feasible": True}),
            ({**base, "method": "lightcone", "parameters": {
                "execution_policy": "automatic_units_v3"}}, {"goodput": 110, "hard_feasible": True})]
    assert paired_block_statistics(rows) == []


def test_legacy_bonus_counters_are_recomputed_only_when_layout_is_identified(tmp_path):
    from lightcone_spec.metrics import historical_position_rounds

    raw = [{"verify_len": [8], "accepted_drafts": [7]}]
    converted = historical_position_rounds({"backend": "DSPARK"}, raw)
    assert converted[0]["verify_len"] == [7]
    assert raw[0]["verify_len"] == [8]
    assert historical_position_rounds({"backend": "EAGLE3"}, raw)[0]["verify_len"] == [7]
    ambiguous = [{"verify_len": [3], "accepted_drafts": [2]}]
    _, derived = normalize_attempt_semantics({"backend": "DSPARK"}, {"rounds": ambiguous}, tmp_path)
    assert derived["position_conditional_acceptance"] == "N/A"
    assert derived["position_metric_semantics"] == "unidentified_legacy_verify_layout"
    current = [{**raw[0], "verify_len_semantics": "draft_positions_v2"}]
    assert historical_position_rounds({"backend": "DSPARK"}, current) == current


def test_mechanism_bins_keep_censoring_timestamps_and_publication_units():
    from lightcone_spec.mechanism import MechanismRecorder

    recorder = MechanismRecorder(bin_tokens=2)
    teacher = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [float("nan"), 0.0]]])
    recorder.record(
        request_ids=["r"],
        generated_offsets=[1],
        teacher_logits=teacher,
        draft_logits=teacher.clone(),
        valid_mask=torch.tensor([[True, True, False]]),
        accepted_drafts=[1],
        committed_tokens=[2],
        prompt_lengths=[100],
    )
    recorder.finish("r", natural_stop=True)
    snapshots = recorder.snapshot(
        [
            {
                "status": "published",
                "request_ids": ["r"],
                "prefix_len_before": [101],
            }
        ]
    )
    assert len(snapshots["mechanism_bins"]) == 2
    assert sum(row["exposed_draft_positions"] for row in snapshots["mechanism_bins"]) == 2
    assert sum(row["updates_published"] for row in snapshots["mechanism_bins"]) == 1
    assert "logits" not in json.dumps(snapshots)
    result = mechanism_position_summary(
        snapshots["mechanism_bins"],
        [
            {
                "request_id": "r",
                "native_token_timestamps_ns": [0, 1_000_000, 3_000_000],
            }
        ],
    )
    assert result[0]["native_p50_itl_ms"] == 1.0
    assert result[1]["native_p50_itl_ms"] == 2.0
    assert result[1]["natural_stops"] == 1
    assert result[0]["effective_requests"] == 1
    assert result[0]["positions"]["1"]["prefix_survival"] == 1
    assert result[1]["positions"]["2"]["conditional_acceptance"] == 0
    assert result[0]["target_entropy"] > result[0]["draft_top1_ce"]
    with pytest.raises(ValueError, match="unknown or duplicate"):
        mechanism_position_summary(
            snapshots["mechanism_bins"] * 2,
            [
                {
                    "request_id": "r",
                    "native_token_timestamps_ns": [0, 1, 2],
                }
            ],
        )
    recorder.reset()
    assert recorder.snapshot()["mechanism_bins"] == []


def test_four_independent_block_log_ratios_and_exact_resolution():
    result = four_block_log_ratio_statistics(
        {0: 110, 1: 120, 2: 115, 3: 118},
        dict.fromkeys(range(4), 100),
    )
    assert result["exact_sign_flip_p"] == 0.125
    assert result["degrees_of_freedom"] == 3
    assert result["ratio_ci95"][0] < result["geometric_mean_ratio"] < result["ratio_ci95"][1]
    assert result["geometric_mean_ratio"] == pytest.approx((1.1 * 1.2 * 1.15 * 1.18) ** 0.25)
    with pytest.raises(ValueError, match="exactly blocks"):
        four_block_log_ratio_statistics({0: 1, 1: 2, 2: 3}, {0: 1, 1: 2, 2: 3})
    with pytest.raises(ValueError, match="positive finite"):
        four_block_log_ratio_statistics(dict.fromkeys(range(4), 0), dict.fromkeys(range(4), 1))


def test_prefix_and_conditional_acceptance_have_distinct_denominators(tmp_path):
    rounds = [{"accepted_drafts": [0, 1, 2, 3], "verify_len": [3, 3, 3, 3]}]
    metrics = position_acceptance_metrics(rounds)
    assert metrics["position_prefix_survival"] == [0.75, 0.5, 0.25]
    assert metrics["position_conditional_survival"] == [0.75, 2 / 3, 0.5]
    censored = position_acceptance_metrics([{"accepted_drafts": [1, 0], "verify_len": [1, 3]}])
    assert censored["position_conditional_survival"] == [0.5, None, None]
    assert censored["position_exposure_counts"] == [2, 1, 1]
    assert (
        position_acceptance_metrics([{"accepted_drafts": [1], "verify_len": []}])[
            "position_prefix_survival"
        ]
        == "N/A"
    )
    original = {"rounds": rounds, "position_conditional_survival": [0.75, 0.5, 0.25]}
    _, corrected = normalize_attempt_semantics({"method": "static"}, original, tmp_path)
    assert corrected["position_conditional_survival"] == [0.75, 2 / 3, 0.5]
    assert original["position_conditional_survival"] == [0.75, 0.5, 0.25]


def test_four_block_reducer_preserves_negative_rows_and_excludes_legacy_bca():
    rows = []
    for method in ("static", "lightcone"):
        for block in range(4):
            rows.append(
                (
                    {
                        "method": method,
                        "block": block,
                        "parameters": {
                            "statistical_unit": "independent_clean_server_paired_block",
                            "sampling_seed": 980406 + block,
                        },
                    },
                    {
                        "goodput": 100 + (10 + block if method == "lightcone" else 0),
                        "hard_feasible": True,
                    },
                )
            )
    assert paired_block_statistics(rows) == []
    result = four_block_panel_statistics(rows)
    comparison = next(
        row
        for row in result
        if row["candidate_method"] == "lightcone" and row["baseline_method"] == "static"
    )
    assert comparison["status"] == "measured"
    assert len(comparison["raw_blocks"]["lightcone"]) == 4
    mixed = [(c, {**m, "memory_budget_policy": "method_peak_v1"}
              if c["method"] == "lightcone" else m) for c, m in rows]
    assert not any(r["status"] == "measured" for r in four_block_panel_statistics(mixed))
    rows[-1][1]["hard_feasible"] = False
    result = four_block_panel_statistics(rows)
    assert all("ratio_ci95" not in row for row in result)
    with pytest.raises(ValueError, match="duplicate four-block"):
        four_block_panel_statistics([*rows, rows[0]])
    rows[-1][0]["parameters"]["sampling_seed"] = 0
    with pytest.raises(ValueError, match="different seeds"):
        four_block_panel_statistics(rows)


def test_mechanism_four_block_ci_uses_run_points_and_keeps_censored_zero(tmp_path):
    from lightcone_spec.metrics import four_block_mechanism_statistics

    rows = []
    for method in ("static", "tts", "lightcone"):
        for block in range(4):
            config = {
                "method": method,
                "block": block,
                "parameters": {
                    "statistical_unit": "independent_clean_server_paired_block",
                    "sampling_seed": block,
                },
            }
            bucket = {
                "position_start": 0,
                "position_end": 2048,
                "effective_requests": 30,
                "natural_stops": 2,
                "target_entropy": 1.0 + 0.1 * block,
                "positions": {"1": {"prefix_survival": 0.0, "conditional_acceptance": None}},
            }
            rows.append((config, {"hard_feasible": True, "mechanism_position_summary": [bucket]}))
    result = four_block_mechanism_statistics(rows)
    measured = [row for row in result if row["metric"] == "target_entropy"]
    assert len(measured) == 3 and all(row["status"] == "measured" for row in measured)
    mixed = [(c, {**m, "memory_budget_policy": "method_peak_v1"}
              if c["method"] == "lightcone" else m) for c, m in rows]
    assert not any(r["status"] == "measured" and r["candidate_method"] == "lightcone"
                   for r in four_block_mechanism_statistics(mixed))
    assert all(
        len(row["raw_blocks"]["lightcone"]) == 4
        for row in measured
        if "lightcone" in row["raw_blocks"]
    )
    zero = [row for row in result if row["metric"] == "position_1_prefix_survival"]
    assert all(row["status"] == "undefined_log_ratio" and "ratio_ci95" not in row for row in zero)


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
