import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.protocol import (
    DSPARK_CONFIDENCE_LOSS_WEIGHT,
    E0_ONLINESPEC_RECIPES,
    FORMAL_ADAPTATION_STRIDE,
    PAPER_NODES,
    TTS_STRIDES,
    Job,
    default_row_counts,
    materialize,
    mechanism_jobs,
    paper_plan,
    segment_count,
    source_coverage_jobs,
    uses_formal_adaptation_stride,
)
from lightcone_spec.runner import (
    ScientificFailure,
    _all_jobs_completed,
    _assigned_gpu,
    _assigned_pair,
    _capacity_infeasible,
    _cell_inputs,
    _gpu_pairs,
    _incomplete_scientific_outcome,
    _interference_within_tolerance,
    _job_from_metric_config,
    _schedule_exhausted_updates,
    _scientific_rejection,
    _screening_incomplete_classification,
    _screening_job,
    _segment_jobs,
    _session_pool_eligible,
    _SessionCellPool,
    _single_gpu_queues,
    _validate_measured_metrics,
)
from lightcone_spec.server import (
    _parse_lscpu_rows,
    _plan_cpu_affinity,
    _sysfs_pci_bdf,
    adaptation_payload,
    apply_runner_affinity,
    server_session_key,
)
from lightcone_spec.state import StateStore


def test_quick_tuning_real_config_and_cpu_entrypoint(tmp_path, monkeypatch):
    import importlib
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    quick = importlib.import_module("quick_tune_lightcone")
    config = replace(ExperimentConfig.load(Path(__file__).resolve().parents[1] / "examples/paper.yaml"),
                     results_root=tmp_path / "formal", sglang_root=tmp_path / "runtime")
    config.sglang_root.mkdir()
    (config.sglang_root / ".lightcone-spec-patched").write_text("verified-test-marker")
    StateStore(config.run_dir)
    monkeypatch.setattr(quick.ExperimentConfig, "load", lambda _: config)
    monkeypatch.setattr(quick, "qa_manifest", lambda _: ({"prompts": {}}, "test"))
    template = Job("t", "E3b", 0, "lightcone", "Qwen/Qwen3-8B", "DFLASH", "MATH-500", block=0)
    monkeypatch.setattr(quick, "preview_jobs", lambda _: [template])
    monkeypatch.setattr(quick, "load_prompt_pool", lambda _: [])
    monkeypatch.setattr(quick, "calibration_split", lambda *_: {"Code": {}, "Math": {}})
    monkeypatch.setattr(sys, "argv", ["quick", "--config", str(config.source), "--output", str(tmp_path / "out"), "--plan-only"])
    quick.main()
    record = json.loads((tmp_path / "out/registration.json").read_text())
    assert record["runtimes"]["old"]["path"] == str(config.sglang_root)
    assert record["window_seconds"] == 30 and record["warmup_seconds"] == 10

    # A/B configs may have different YAML locations but only the runtime may differ.
    candidate = replace(config, source=tmp_path / "candidate.yaml", sglang_root=tmp_path / "candidate")
    candidate.sglang_root.mkdir()
    (candidate.sglang_root / ".lightcone-spec-patched").write_text("candidate-marker")
    monkeypatch.setattr(quick.ExperimentConfig, "load", lambda p: candidate if p == candidate.source else config)
    monkeypatch.setattr(sys, "argv", ["quick", "--config", str(config.source), "--candidate-config", str(candidate.source),
                                     "--phase", "compare", "--output", str(tmp_path / "compare"), "--plan-only"])
    quick.main()
    assert json.loads((tmp_path / "compare/registration.json").read_text())["runtimes"]["new"]["marker"] == "candidate-marker"
    candidate = replace(candidate, server=replace(candidate.server, requests_per_cell=999))
    with pytest.raises(ValueError, match="scientific configuration"):
        quick.main()

    # Real v48 reporter retains committed_tokens across successful flush.
    # Request counters and adaptive reset state must still clear; no gate removed.
    from lightcone_spec.metrics import SAFETY_COUNTERS

    class Client:
        def server_info(self):
            return {}

    ranks = [{**dict.fromkeys(SAFETY_COUNTERS, 0), "committed_tokens": 2048,
              "target_calls": 0, "accepted_drafts": 0, "active_version": 0,
              "round": 0, "updates_published": 0, "disabled_reason": None} for _ in range(2)]
    monkeypatch.setattr(quick, "_speed_metrics", lambda *_: {"rank_local": ranks})
    assert quick.safe_metrics(Client(), adaptive=True, reset=True)["rank_local"] == ranks
    ranks[0]["target_calls"] = 1
    with pytest.raises(RuntimeError, match="request/cache counters"):
        quick.safe_metrics(Client(), adaptive=True, reset=True)
    ranks[0]["target_calls"] = 0
    ranks[0]["active_version"] = 1
    with pytest.raises(RuntimeError, match="TP ranks disagree"):
        quick.safe_metrics(Client(), adaptive=True, reset=True)


def test_stride_audit_budget_split_identity_and_formal_isolation():
    from lightcone_spec.stride_audit import (
        audit_job,
        calibration_split,
        refinement_candidates,
        stage_jobs,
    )

    records = [{"problem_id": f"{source}-{i}", "source": source, "prompt": f"{source} question {i}"}
               for source in ("APPS", "OpenR1-Math") for i in range(15)]
    split = calibration_split(records, records[:1])
    assert split == calibration_split(list(reversed(records)), records[:1])
    for domain in split.values():
        assert len(domain["search"]) == 4 and len(domain["confirmation"]) == 8
        assert not {r["problem_id"] for r in domain["search"]} & {r["problem_id"] for r in domain["confirmation"]}
    template = Job("t", "E3b", 0, "lightcone", "Qwen/Qwen3-8B", "DFLASH", "MATH-500",
                   parameters={"panel": "preview_v1", "preview_revision": 3, "frozen_recipe": {"stride": 1},
                               "generation_tokens": 32768, "respect_eos": True})
    coarse = stage_jobs(template, split, phase="coarse", implementation="test")
    assert len(coarse) == 20 and len({j.job_id for j in coarse}) == 20
    assert len(stage_jobs(template, split, phase="refine", implementation="test", strides=(2, 8))) == 8
    assert len(stage_jobs(template, split, phase="confirmation", implementation="test", strides=(10,))) == 16
    assert refinement_candidates(32) == (16,)
    assert all(j.gpu_count == 2 and j.load == "c1" and j.parameters["generation_tokens"] == 32768 for j in coarse)
    audit = audit_job(template, split, phase="baseline", domain="Code", method="lightcone",
                      stride=32, block=0, implementation="test")
    assert not uses_formal_adaptation_stride(audit)
    assert uses_formal_adaptation_stride(replace(audit, parameters={**audit.parameters, "excluded_from_analysis": False}))
    assert template.parameters["frozen_recipe"]["stride"] == 1


def test_preview_exact_56_frozen_pairs_and_no_public_node_change():
    from collections import Counter

    from lightcone_spec.preview import VIDEO_METHODS, preview_jobs
    from lightcone_spec.scheduling import logical_unit_key

    manifest = {
        "tts_recipe": {"lr": 1e-4, "stride": 10}, "lightcone_recipe": {"stride": 10},
        "dspark_recipe": {"stride": 10, "confidence_temperatures": [1.] * 7},
        "dflash_width": 8, "dspark_width": 16, "trace_request_count": 16,
        "serving_output_tokens": 256, "trace_offset": 17,
        "trace_anchor": {"source_job_ids": ["anchor"]},
        "trace": {"arrivals": list(range(16)), "lengths": [[128, 64]] * 16},
        "prompts": {task: [{"problem_id": str(i), "prompt": str(i)} for i in range(32)]
                    for task in ("MATH-500", "LiveCodeBench")},
    }
    jobs = preview_jobs(manifest)
    assert len(jobs) == 56 and len(PAPER_NODES) == 21
    assert Counter(j.parameters["preview_panel"] for j in jobs) == {
        "long_generation": 24, "serving": 24, "burstgpt": 8,
    }
    assert jobs == preview_jobs(json.loads(json.dumps(manifest)))
    from lightcone_spec.preview import QWEN38_CHECKPOINTS
    extended = {**manifest, "qwen38": {"tp": 2, "checkpoints": QWEN38_CHECKPOINTS,
                                       "prompts": manifest["prompts"]["LiveCodeBench"][:8]}}
    assert len(preview_jobs(extended)) == 80
    assert preview_jobs(extended)[:56] == jobs
    tp2 = {**manifest, "comparison_topologies": {"dspark_serving": 2}}
    with pytest.raises(ValueError, match="anchor"):
        preview_jobs(tp2)
    tp2["tp2_trace_anchor"] = {"source_job_ids": ["tp2-anchor"], "topology": "tp2_dp1"}
    migrated = preview_jobs(tp2)
    assert len(migrated) == 56
    assert all(j == jobs[i] for i, j in enumerate(migrated) if j.backend == "DFLASH")
    assert all(j.gpu_count == 2 and j.parameters["replaces_job_id"] == jobs[i].job_id
               for i, j in enumerate(migrated) if j.backend == "DSPARK")
    units = {}
    for job in jobs:
        units.setdefault(logical_unit_key(job), []).append(job)
        assert job.parameters["sampling_seed"] == job.block
        assert job.parameters["stride"] == 10
        if job.method == "tts":
            assert job.load == "c1" and job.parameters["execution_request_count"] == 8
    assert len(units) == 24
    assert sorted(len(rows) for rows in units.values()) == [2] * 16 + [3] * 8
    for rows in units.values():
        assert len({json.dumps(j.parameters["preview_prompt_records"]) for j in rows}) == 1
    serving = [j for j in jobs if j.parameters["preview_panel"] == "serving"]
    assert all(j.parameters["execution_request_count"] == 32 for j in serving)
    assert len({json.dumps(j.parameters["preview_prompt_records"]) for j in serving}) == 1
    assert len(VIDEO_METHODS) == 6 and all(method != "tts" for _, method, _ in VIDEO_METHODS)
    import runpy
    qa = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/validate_preview_updates.py"))
    assert qa["update_qa_plan"]("s1-long", pressure_only=True) == [("pressure", 32768, 0.)]
    assert len(qa["update_qa_plan"]("s1-long")) == 3
    assert len(qa["update_qa_plan"]("s1-long", reset_diagnostic=True)) == 2
    with pytest.raises(ValueError, match="separate S1"):
        qa["update_qa_plan"]("ensemble", pressure_only=True)
    with pytest.raises(ValueError, match="separate S1"):
        qa["update_qa_plan"]("s1-long", pressure_only=True, reset_diagnostic=True)
    source = next(j for j in jobs if j.method == "lightcone" and j.backend == "DFLASH")
    before = json.dumps(source.to_dict(), sort_keys=True)
    s1, recipe = qa["update_qa_job"](source, "s1-long")
    assert s1.parameters["generation_tokens"] == 32768 and s1.parameters["stride"] == 1
    assert s1.parameters["execution_request_count"] == 1 and s1.parameters["excluded_from_analysis"]
    assert s1.parameters["preview_prompt_records"] == source.parameters["preview_prompt_records"]
    assert s1.block is None and s1.gpu_count == 1 and recipe["rank"] == 8
    ens, _ = qa["update_qa_job"](source, "ensemble")
    assert ens.parameters["stride"] == 10 and ens.parameters["generation_tokens"] == 512
    assert ens.parameters["frozen_recipe"]["ensemble_optimizer"] == "adam_preview_v3"
    ens_tp2, _ = qa["update_qa_job"](source, "ensemble", 2)
    assert ens_tp2.gpu_count == 2 and ens_tp2.parameters["topology"] == "tp2_dp1"
    assert ens_tp2.parameters["execution_request_count"] == ens.parameters["execution_request_count"]
    assert ens_tp2.parameters["frozen_recipe"] == ens.parameters["frozen_recipe"]
    with pytest.raises(ValueError, match="TP1/TP2"):
        qa["update_qa_job"](source, "ensemble", 4)
    assert json.dumps(source.to_dict(), sort_keys=True) == before


def test_qwen38_preview_exact_24_native_mtp_and_common_tp():
    from lightcone_spec.preview import QWEN38_MODEL, QWEN38_VIDEO_METHODS, qwen38_jobs
    from lightcone_spec.scheduling import logical_unit_key

    checkpoints = {key: {"repo": repo, "revision": "a" * 40} for key, repo in {
        "target": QWEN38_MODEL, "NEXTN": QWEN38_MODEL,
        "DSPARK": "RadixArk/Qwen3.8-27B-DSpark", "DFLASH": "incoai/Qwen3.8-27B-DFlash2",
    }.items()}
    manifest = {"tts_recipe": {"lr": 1e-4, "stride": 10}, "lightcone_recipe": {"lr": .001, "stride": 10},
                "qwen38": {"tp": 2, "checkpoints": checkpoints,
                           "prompts": [{"prompt": str(i)} for i in range(8)]}}
    jobs = qwen38_jobs(manifest)
    assert len(jobs) == len({j.job_id for j in jobs}) == 24
    assert len({logical_unit_key(j) for j in jobs}) == 4
    assert {j.block for j in jobs} == {0, 1, 2, 3}
    assert all(j.gpu_count == 2 and j.parameters["topology"] == "tp2_dp1" for j in jobs)
    assert all(j.load == "c1" and j.context == 17408 and j.parameters["generation_tokens"] == 1024 for j in jobs)
    assert all(j.parameters["execution_request_count"] == 8 and j.parameters["memory_budget_policy"] == "method_peak_v1" for j in jobs)
    assert len({(j.backend, j.method) for j in jobs}) == 6
    assert ("NEXTN", "static", "Native MTP") in QWEN38_VIDEO_METHODS
    assert any(m == "onlinespec_ens" for _, m, _ in QWEN38_VIDEO_METHODS)
    assert not any(m == "tts" for _, m, _ in QWEN38_VIDEO_METHODS)
    checkpoints["NEXTN"]["revision"] = "b" * 40
    with pytest.raises(ValueError, match="same target"):
        qwen38_jobs(manifest)


def test_preview_v3_exact_96_isolates_recipes_and_all_baselines(monkeypatch):
    from collections import Counter
    from copy import deepcopy

    from lightcone_spec.preview import QWEN38_CHECKPOINTS, preview_jobs
    from lightcone_spec.scheduling import logical_unit_key

    records = [{"prompt": str(i), "problem_id": str(i)} for i in range(32)]
    manifest = {
        "version": 3, "lightcone_recipe": {"stride": 10, "optimizer": "chronobelief"},
        "dspark_recipe": {"stride": 10, "confidence_temperatures": [1.] * 7},
        "dflash_width": 16, "dspark_width": 16, "trace_request_count": 16,
        "serving_output_tokens": 256, "trace_offset": 17,
        "trace_anchor": {"source_job_ids": ["anchor"]},
        "trace": {"arrivals": list(range(16)), "lengths": [[128, 64]] * 16},
        "prompts": {task: records for task in ("MATH-500", "LiveCodeBench")},
        "qwen38": {"tp": 2, "checkpoints": QWEN38_CHECKPOINTS, "prompts": records[:8]},
    }
    original = deepcopy(manifest)
    jobs = preview_jobs(manifest)
    assert manifest == original
    assert len(jobs) == len({j.job_id for j in jobs}) == 96
    assert Counter(j.parameters["preview_panel"] for j in jobs) == {
        "long_generation": 40, "serving": 24, "burstgpt": 8, "qwen38_transfer": 24}
    assert not any(j.method.startswith("tts") for j in jobs)
    units = {}
    for job in jobs:
        units.setdefault(logical_unit_key(job), []).append(job)
        payload = adaptation_payload(job, job.parameters["frozen_recipe"])
        if job.method == "lightcone":
            assert payload["stride"] == 1
            assert payload["optimizer"]["learning_rate"] == .001
            assert not uses_formal_adaptation_stride(job)
        if job.method == "onlinespec_ens":
            assert payload["stride"] == 10
            assert payload["online_spec"]["ensemble_optimizer"] == "adam_preview_v3"
            assert payload["online_spec"]["hedge_learning_rate"] == 10
    assert all(len({j.gpu_count for j in unit}) == 1 for unit in units.values())
    assert all(len({json.dumps(j.parameters["preview_prompt_records"]) for j in unit}) == 1 for unit in units.values())
    assert jobs == preview_jobs(json.loads(json.dumps(manifest)))
    import runpy
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    qa = runpy.run_path(str(scripts / "validate_preview_group.py"))
    candidate, source_name = qa["qa_manifest"]({"formal_preview_manifest_v1": {**manifest, "version": 1}})
    from lightcone_spec.preview_revision import s10_manifest
    assert candidate == s10_manifest(manifest) and source_name.startswith("excluded_candidate")
    assert qa["qa_manifest"]({"formal_preview_manifest_v3": manifest})[0] == s10_manifest(manifest)
    assert qa["qa_manifest"]({"formal_preview_manifest_v3": candidate}) == (candidate, "formal_preview_manifest_v3")
    with pytest.raises(ValueError, match="no legacy fallback"):
        qa["qa_manifest"]({"formal_preview_manifest_v3": {"version": 2},
                           "formal_preview_manifest_v1": {**manifest, "version": 1}})
    with pytest.raises(ValueError, match="missing frozen"):
        qa["qa_manifest"]({})
    for tp in (1, 2):
        for task in ("MATH-500", "LiveCodeBench"):
            rows = [qa["full_condition_job"](manifest, task, case, tp) for case in qa["CASES"]]
            assert len({row.job_id for row in rows}) == 5
            for row in rows:
                assert row.gpu_count == tp and row.block == 0 and row.load == "c1"
                assert row.parameters["execution_request_count"] == 8
                assert row.parameters["generation_tokens"] == 32768
                assert row.parameters["respect_eos"] is True
                assert row.parameters["excluded_from_analysis"] is True
                assert row.parameters["preview_prompt_records"] == records[:8]
                source = next(j for j in jobs if j.task == task and j.block == 0
                              and j.parameters["preview_panel"] == "long_generation"
                              and (j.backend, j.method) == (row.backend, row.method))
                assert row.parameters["frozen_recipe"] == source.parameters["frozen_recipe"]
                assert row.parameters["sampling_seed"] == source.parameters["sampling_seed"]
    assert manifest == original
    with pytest.raises(ValueError):
        qa["full_condition_job"](manifest, "MATH-500", "tts", 2)
    legacy = replace(next(j for j in jobs if j.method == "lightcone"),
                     parameters={"stride": 1})
    assert adaptation_payload(legacy)["stride"] == 10

    restored = preview_jobs(s10_manifest(manifest))
    assert len(restored) == 96 and manifest == original
    for old, new in zip(jobs, restored, strict=True):
        if old.method != "lightcone":
            assert new == old  # baseline identity, recipe, prompts and pairing remain reusable
            continue
        assert new.job_id == old.job_id + "__s10"
        assert new.parameters["replaces_job_id"] == old.job_id
        assert new.parameters["pairing_key"] == old.parameters["pairing_key"]
        assert new.parameters["preview_prompt_records"] == old.parameters["preview_prompt_records"]
        assert new.parameters["stride"] == 10 and "S=10" in new.parameters["method_label"]
        payload = adaptation_payload(new, new.parameters["frozen_recipe"])
        assert payload["stride"] == 10
        assert payload["optimizer"]["learning_rate"] == .001


def test_full_condition_qa_requires_complete_safe_rank_evidence(tmp_path, monkeypatch):
    import gzip
    import runpy
    from types import SimpleNamespace

    from lightcone_spec.metrics import SAFETY_COUNTERS

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    qa = runpy.run_path(str(scripts / "validate_preview_group.py"))
    job = SimpleNamespace(job_id="excluded", method="onlinespec_ens", backend="DFLASH",
                          task="MATH-500", gpu_count=2, parameters={"qa_source_job_id": "formal"})
    state = SimpleNamespace(completed_attempt_dir=lambda _: tmp_path)
    metrics = {"hard_feasible": True, "updates_published": 2,
               "rank_local_after": [{key: 0 for key in SAFETY_COUNTERS} for _ in range(2)]}
    with gzip.open(tmp_path / "requests.jsonl.gz", "wt") as stream:
        for _ in range(8):
            stream.write(json.dumps({"completion_tokens": 771}) + "\n")
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    result = qa["full_condition_result"](state, job)
    assert result["formal_acceptance"] is False and result["normal_eos"] is True
    assert result["completion_tokens"] == [771] * 8
    metrics["rank_local_after"][1]["fallbacks"] = 1
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="rank-local"):
        qa["full_condition_result"](state, job)
    metrics["rank_local_after"] = metrics["rank_local_after"][:1]
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="rank-local"):
        qa["full_condition_result"](state, job)
    metrics["rank_local_after"] *= 2
    metrics["updates_published"] = 0
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="no updates"):
        qa["full_condition_result"](state, job)


def test_common_tp_requires_all_full_workload_cases_and_correctness():
    from lightcone_spec.preview import select_common_tp

    cases = {"static-c1", "lightcone-c1", "static-c32", "lightcone-c32"}
    rows = [{"case": c, "tp": tp, "correct": True, "full_workload": True,
             "reset_verified": True, "gpu_binding_verified": True}
            for tp in (1, 2) for c in sorted(cases)]
    assert select_common_tp(rows, cases) == 1
    rows[0]["correct"] = False
    assert select_common_tp(rows, cases) == 2
    rows[-1]["full_workload"] = False
    with pytest.raises(ValueError, match="no fully validated"):
        select_common_tp(rows, cases)


def test_preview_heldout_deterministic_and_no_duplicates():
    from lightcone_spec.preview import held_out_pool

    records = [{"problem_id": str(i), "prompt": f"prompt-{i}"} for i in range(12)]
    calibration = [records[0], {"problem_id": "other", "prompt": "prompt-1"}]
    chosen = held_out_pool(records, calibration, 8)
    assert chosen == held_out_pool(list(reversed(records)), calibration, 8)
    assert len({r["prompt"] for r in chosen}) == 8
    assert not {"prompt-0", "prompt-1"}.intersection(r["prompt"] for r in chosen)
    with pytest.raises(ValueError, match="distinct held-out"):
        held_out_pool(records, calibration, 11)


def test_preview_ci_workflow_parses_and_has_read_only_pinned_actions():
    import re

    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/quality.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    assert {"cpu", "patched-compile", "live-ui"}.issubset(workflow["jobs"])
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                assert isinstance(step["run"], str)
            if "uses" in step:
                assert re.fullmatch(r"[\w/-]+@[a-f0-9]{40}", step["uses"])

def test_official_jsonl_keeps_unicode_line_separators_inside_prompts(tmp_path):
    from lightcone_spec.data import load_source_prompt_records

    path = tmp_path / "official.jsonl"
    path.write_text(
        json.dumps({"turns": ["first\u2028second\u2029third", "unused"]}, ensure_ascii=False) + "\n"
    )
    rows = load_source_prompt_records(path, max_samples=1, seed=980406)
    assert rows[0]["prompt"] == "first\u2028second\u2029third"
    assert rows[0]["turns"] == [rows[0]["prompt"]]


def test_coverage_gpu_acceptance_is_excluded_and_does_not_resize_formal_cells():
    from lightcone_spec.coverage import gpu_acceptance_jobs

    original = source_coverage_jobs(), mechanism_jobs()
    jobs = gpu_acceptance_jobs()
    assert len(jobs) == len({job.job_id for job in jobs}) == 41
    assert {job.parameters["qa_phase"] for job in jobs} == {"gemma", "qwen", "dense14", "panels"}
    assert all(job.parameters["excluded_from_analysis"] for job in jobs)
    assert all(job.parameters["execution_request_count"] == 2 for job in jobs)
    assert all(job.block is None and job.parameters["generation_tokens"] == 512 for job in jobs)
    assert all(job.gpu_count == 2 for job in jobs if job.model == "Qwen/Qwen3-14B")
    assert original == (source_coverage_jobs(), mechanism_jobs())


def test_four_block_source_and_mechanism_matrix_preserves_public_dag(tmp_path):
    source, mechanism = source_coverage_jobs(), mechanism_jobs()
    assert len(source) == 1296
    assert len(mechanism) == 48
    assert len(PAPER_NODES) == 21
    assert sum(job.parameters["execution_request_count"] for job in source) == 436320
    assert sum(job.model == "Qwen/Qwen3-14B" for job in source) == 324
    pairs = {}
    for job in source:
        assert job.load == "c1" and job.width == 8
        assert job.parameters["respect_eos"] and job.parameters["verification"] == "fixed_budget"
        assert job.parameters["generation_tokens"] == 2048
        assert job.parameters["sampling_seed"] == 980406 + job.block
        assert job.parameters["topology"] == (
            "tp2_dp1" if job.model == "Qwen/Qwen3-14B" else "tp1_dp1"
        )
        pairs.setdefault((job.parameters["pairing_key"], job.block), []).append(job)
    assert len(pairs) == 432
    for jobs in pairs.values():
        assert {job.method for job in jobs} == {"static", "tts", "lightcone"}
        assert len({job.parameters["source_checkpoint"] for job in jobs}) == 1
        assert len({job.parameters["sampling_seed"] for job in jobs}) == 1
    assert {j.parameters["execution_request_count"] for j in mechanism} == {30, 32}
    assert all(j.parameters["exclude_from_headline_performance"] for j in mechanism)
    assert source == source_coverage_jobs() and mechanism == mechanism_jobs()
    state = StateStore(tmp_path / "run")
    state.add_internal_jobs(source + mechanism)
    state.add_internal_jobs(source + mechanism)
    assert len(state.jobs(source[0].node)) == 1296
    assert len(state.jobs(mechanism[0].node)) == 48


def test_official_source_loader_preserves_first_turn_and_sampling(tmp_path):
    import random

    from lightcone_spec.data import load_source_prompt_records

    path = tmp_path / "official.jsonl"
    path.write_text("\n".join(json.dumps({"turns": [str(i), "ignored"]}) for i in range(6)))
    complete = load_source_prompt_records(path, max_samples=6, seed=980406)
    assert [row["prompt"] for row in complete] == list(map(str, range(6)))
    expected = list(range(6))
    random.Random(980406).shuffle(expected)
    selected = load_source_prompt_records(path, max_samples=3, seed=980406)
    assert [row["source_row_index"] for row in selected] == expected[:3]
    assert all(len(row["turns"]) == 1 for row in selected)
    with pytest.raises(ValueError, match="incomplete official"):
        load_source_prompt_records(path, max_samples=7, seed=980406)


EXPECTED = {
    "preflight": 10,
    "E3a": 140,
    "TTS-Cal": 72,
    "E1": 68,
    "E2-r0": 424,
    "E2-r1": 109,
    "E2-r2": 31,
    "E2-r3": 25,
    "E4-screen": 52,
    "E4-local": 168,
    "E4-profile": 3,
    "E3b-pilot": 20,
    "E3b-final": 132,
    "E1a": 3,
    "E5-pilot": 19,
    "E5-final": 114,
    "E6-pilot": 22,
    "E6-final": 60,
    "E0-tune": 54,
    "E0-pilot": 88,
    "E0-final": 264,
}


def test_paper_v2_node_order_counts_and_plan():
    assert len(PAPER_NODES) == 21
    assert default_row_counts() == EXPECTED
    assert sum(EXPECTED.values()) == 1878
    assert len(paper_plan()) == 21
    assert [row.rows for row in paper_plan() if row.name == "TTS-Cal"] == ["<=108"]


def test_bundled_jobs_preserve_registered_conditions():
    e3a = materialize("E3a")
    assert sum(segment_count(job) for job in e3a) == 252
    assert {job.context for job in e3a} == {4096, 16384, 32768, 40928}
    assert {job.load for job in e3a} == {f"c{x}" for x in (1, 2, 4, 8, 16, 32, 64)}
    assert {segment["regime"] for job in e3a for segment in job.parameters["segments"]} == {
        "long_input_short_output",
        "short_input_long_generation",
        "multi_turn_shared_prefix",
    }
    e0 = materialize("E0-final")
    assert all(len(job.parameters["segments"]) == 9 for job in e0)
    assert {segment["task"] for segment in e0[0].parameters["segments"]} == {
        "GSM8K",
        "MATH-500",
        "AIME-2025",
        "MBPP",
        "HumanEval",
        "LiveCodeBench",
        "MT-Bench",
        "AlpacaEval",
        "Arena-Hard",
    }


def test_primary_and_secondary_block_semantics():
    e3b = materialize("E3b-final")
    assert {job.load for job in materialize("E3b-pilot")} == {"c1"}
    assert {job.load for job in e3b} == {"c1"}
    primary = [job for job in e3b if job.parameters["workload"] == "primary_long_history"]
    secondary = [job for job in e3b if job.parameters["workload"].startswith("secondary_")]
    assert len(primary) == 60
    assert {job.block for job in primary} == set(range(12))
    assert len(secondary) == 72
    assert {job.block for job in secondary} == set(range(6))
    assert len(materialize("E6-final")) == 2 * 5 * 6
    assert len(materialize("E0-final")) == 44 * 6


def test_tts_and_dspark_registered_fidelity():
    tts = materialize("TTS-Cal")
    assert len(tts) == 9 * 8
    assert {job.parameters["stride"] for job in tts} == set(TTS_STRIDES)
    payload = adaptation_payload(tts[0])
    assert payload["optimizer"]["name"] == "adam"
    assert payload["optimizer"]["weight_decay"] == 0
    assert payload["optimizer"]["grad_clip"] == 0
    sgdm = next(
        job
        for job in materialize("E1")
        if job.parameters.get("optimizer") == "sgdm"
    )
    assert adaptation_payload(sgdm)["optimizer"]["momentum"] == 0.9
    assert payload["teacher_row_policy"] == "latest_update_round_only"
    assert payload["loss_position_decay"] == pytest.approx(math.exp(-1 / 7))
    e1a = materialize("E1a")
    assert len(e1a) == 3
    assert sum(segment_count(job) for job in e1a) == 22
    assert {job.parameters["confidence_loss_weight"] for job in e1a} == {
        DSPARK_CONFIDENCE_LOSS_WEIGHT
    }
    assert {job.parameters["source_transfer_recipe"] for job in e1a} == {
        "dflash_lightcone_recipe"
    }
    assert {job.parameters["temperature"] for job in e1a} == {1.0}
    capture_payload = adaptation_payload(
        e1a[0],
        {
            "scope": "last3",
            "parameterization": "lora",
            "rank": 8,
            "confidence_temperatures": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        },
    )
    assert capture_payload["confidence_temperatures"] == [
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
    ]
    downstream_dspark = next(
        job
        for job in materialize("E5-pilot")
        if job.backend == "DSPARK"
        and job.method == "lightcone"
        and job.parameters.get("topology") == "tp2_dp1"
    )
    downstream_payload = adaptation_payload(
        downstream_dspark,
        {
            "scope": "last1",
            "parameterization": "lora",
            "rank": 8,
            "confidence_loss_weight": 1.0,
            "confidence_temperatures": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        },
    )
    assert downstream_payload["confidence_loss_weight"] == 1.0
    assert downstream_payload["confidence_temperatures"] == [
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
    ]
    assert {
        job.parameters["update_steps"]
        for job in materialize("E4-screen")
        if job.parameters.get("workload") == "tts_update_steps"
    } == {1, 2, 4, 8}


def test_formal_adaptive_jobs_resolve_s10_without_erasing_exploratory_sweeps():
    tts_screen = next(job for job in materialize("TTS-Cal") if job.parameters["stride"] == 50)
    assert not uses_formal_adaptation_stride(tts_screen)
    assert adaptation_payload(tts_screen)["stride"] == 50

    e4_screen = next(
        job
        for job in materialize("E4-screen")
        if job.method == "lightcone" and job.parameters["stride"] == 50
    )
    assert not uses_formal_adaptation_stride(e4_screen)
    assert adaptation_payload(e4_screen)["stride"] == 50

    formal = [
        next(job for job in materialize("E1") if job.method == "tts"),
        next(job for job in materialize("E1") if job.method == "l0_naive"),
        next(job for job in materialize("E2-r0") if job.method == "lightcone_candidate"),
        next(job for job in materialize("E3b-final") if job.method == "lightcone"),
        next(job for job in materialize("E5-final") if job.method == "tts_lora_batched"),
    ]
    for job in formal:
        assert uses_formal_adaptation_stride(job)
        assert adaptation_payload(job, {"stride": 50})["stride"] == FORMAL_ADAPTATION_STRIDE

    for onlinespec in (
        job for job in materialize("E0-tune") if job.method.startswith("onlinespec")
    ):
        assert uses_formal_adaptation_stride(onlinespec)
        assert adaptation_payload(onlinespec)["stride"] == FORMAL_ADAPTATION_STRIDE


def test_e2_uses_selected_recipe_without_inheriting_e1_runtime_fields():
    selected = {
        "parameterization": "lora",
        "rank": 8,
        "scope": "last1",
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "schedule": "constant",
        "generation_tokens": 8192,
        "regime": "short_input_long_generation",
        "registered_load": "reference_load",
        "stimulus_id": "E1-selection-stimulus",
    }
    row = materialize("E2-r0", e2_rows=(selected,))[0]
    assert row.parameters["generation_tokens"] == 2048
    assert row.parameters["regime"] == "short_input_long_generation"
    assert row.load == "c2"
    assert "registered_load" not in row.parameters
    assert "stimulus_id" not in row.parameters


def test_e1a_domain_fit_validation_split_is_deterministic_and_disjoint(tmp_path: Path):
    dataset = tmp_path / "calibration.jsonl"
    rows = []
    for source in ("APPS", "OpenR1-Math", "UltraChat"):
        rows.extend(
            {
                "problem_id": f"{source}-{index}",
                "prompt": f"{source} prompt {index}",
                "source": source,
            }
            for index in range(24)
        )
    rows.extend(
        {
            "problem_id": f"synthetic-{index}",
            "prompt": f"synthetic prompt {index}",
            "source": "controlled_synthetic",
        }
        for index in range(4)
    )
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="split-test",
        sglang_root=tmp_path / "sglang",
        results_root=tmp_path,
        models={},
        drafts={},
        datasets={"CalibrationMix": dataset},
        gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python"),
        protocol=ProtocolConfig(),
    )

    class Client:
        @staticmethod
        def tokenize(prompt):
            return tuple(range(max(1, len(prompt.split()))))

    fit = _segment_jobs(materialize("E1a")[0])[0]
    validation = _segment_jobs(materialize("E1a")[2])[0]
    state = StateStore(config.run_dir)
    _, _, fit_meta = _cell_inputs(config, state, Client(), fit)
    _, _, validation_meta = _cell_inputs(config, state, Client(), validation)
    fit_ids = {row["problem_id"] for row in fit_meta["examples"]}
    validation_ids = {row["problem_id"] for row in validation_meta["examples"]}
    assert len(fit_ids) == len(validation_ids) == 12
    assert fit_ids.isdisjoint(validation_ids)
    assert fit_ids | validation_ids == {f"OpenR1-Math-{index}" for index in range(24)}


def test_e5_source_aligned_methods_and_curves():
    pilot = materialize("E5-pilot")
    final = materialize("E5-final")
    assert len(pilot) == 19
    assert len(final) == 114
    assert sum(segment_count(job) for job in pilot) == 98
    assert sum(segment_count(job) for job in final) == 708
    assert sum(job.parameters["workload"] == "topology_compatibility" for job in pilot) == 4
    assert {job.method for job in final if job.block in range(12)} >= {
        "target_only",
        "static",
        "tts",
        "lightcone",
    }
    batched = [job for job in final if job.method == "tts_lora_batched"]
    assert len(batched) == 6
    assert {
        segment["load"] for segment in batched[0].parameters["segments"]
    } >= {"closed_loop_c1", "closed_loop_c256", "burstgpt_shape"}
    full_tts = next(job for job in final if job.method == "tts")
    assert {segment["load"] for segment in full_tts.parameters["segments"]} == {
        "closed_loop_c1",
        "burstgpt_shape",
    }
    transfer_pilot = [
        job for job in pilot if job.parameters["workload"] == "multigpu_serving_transfer"
    ]
    transfer_final = [
        job for job in final if job.parameters["workload"] == "multigpu_serving_transfer"
    ]
    assert len(transfer_pilot) == 8
    assert len(transfer_final) == 48
    assert {
        (job.backend, job.method, job.parameters["topology"])
        for job in transfer_pilot
    } == {
        (backend, method, topology)
        for backend in ("DFLASH", "DSPARK")
        for method in ("static", "lightcone")
        for topology in ("tp2_dp1", "two_replica_tp1_dp2")
    }
    assert {job.block for job in transfer_final} == set(range(6))
    assert all(
        segment["registered_concurrency_scope"] == "system"
        for job in (*transfer_pilot, *transfer_final)
        for segment in job.parameters["segments"]
    )
    assert {
        segment["load"] for job in transfer_pilot for segment in job.parameters["segments"]
    } == {"closed_loop_c1", "closed_loop_c32", "closed_loop_c128", "burstgpt_shape"}
    assert {
        segment["load"] for job in transfer_final for segment in job.parameters["segments"]
    } == {"closed_loop_c32", "closed_loop_c128", "burstgpt_shape"}


def test_e5_extension_is_append_only_for_registered_parent_identity():
    pilot = materialize("E5-pilot")
    final = materialize("E5-final")
    assert [job.ordinal for job in pilot[:11]] == list(range(11))
    assert [job.ordinal for job in final[:66]] == list(range(66))
    assert all(
        job.parameters["workload"] != "multigpu_serving_transfer"
        for job in (*pilot[:11], *final[:66])
    )
    assert all(
        job.parameters["workload"] == "multigpu_serving_transfer"
        for job in (*pilot[11:], *final[66:])
    )


def test_e0_method_scope_is_deliberately_sparse():
    rows = materialize("E0-final")
    assert sum(job.method == "target_only" for job in rows) == 4 * 6
    assert sum(job.method == "l0_naive" for job in rows) == 6
    assert sum(job.method.startswith("onlinespec") for job in rows) == 18
    assert {(job.model, job.backend) for job in rows if job.method.startswith("onlinespec")} == {
        ("Qwen/Qwen3-8B", "DFLASH")
    }


def test_e0_uses_frozen_source_transfer_recipes_without_mapping_chunk_to_stride():
    validations = [
        job
        for job in materialize("E0-tune")
        if job.parameters.get("recipe_validation")
    ]
    assert len(validations) == 3
    assert {job.method for job in validations} == set(E0_ONLINESPEC_RECIPES)
    for job in validations:
        recipe = E0_ONLINESPEC_RECIPES[job.method]
        assert {
            name: job.parameters[name] for name in recipe
        } == recipe
        assert job.parameters["source_chunk_size"] in {40, 80}
        assert job.parameters["source_epochs"] in {3, 5}
        assert job.parameters["stride"] == 10
        assert job.parameters["source_chunk_size"] != job.parameters["stride"]
    assert len({server_session_key(job) for job in validations}) == 3


def test_e5_burstgpt_segment_count_includes_topology_transfer():
    assert sum(
        segment.get("load") == "burstgpt_shape"
        for node in ("E5-pilot", "E5-final")
        for job in materialize(node)
        for segment in job.parameters.get("segments", ())
    ) == 129


def test_server_reuse_and_eight_gpu_block_affinity(tmp_path: Path):
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="run",
        sglang_root=tmp_path / "sglang",
        results_root=tmp_path,
        models={},
        drafts={},
        datasets={},
        gpu_ids=tuple(range(8)),
        server=ServerConfig(python=tmp_path / "python", base_port=30000),
        protocol=ProtocolConfig(),
    )
    assert _gpu_pairs(config) == ((0, 1), (2, 3), (4, 5), (6, 7))
    rows = [
        next(job for job in materialize("E6-final") if job.block == block) for block in range(4)
    ]
    assert [_assigned_pair(config, job) for job in rows] == list(_gpu_pairs(config))
    singles = [job for job in materialize("E3b-final") if job.block == 3]
    assert {_assigned_gpu(config, job) for job in singles} == {3}
    first, second = materialize("TTS-Cal")[:2]
    assert server_session_key(first) == server_session_key(second)


def test_bundled_segments_stay_together_and_parents_balance(tmp_path: Path):
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="run",
        sglang_root=tmp_path / "sglang",
        results_root=tmp_path,
        models={},
        drafts={},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python", base_port=30000),
        protocol=ProtocolConfig(),
    )
    parents = materialize("E3a")[:2]
    children = [_segment_jobs(parent) for parent in parents]
    assert all(len({_assigned_gpu(config, child) for child in rows}) == 1 for rows in children)
    assert {_assigned_gpu(config, rows[0]) for rows in children} == {0, 1}
    assert _screening_job(children[0][0])
    replacement = replace(
        children[0][0],
        node="bugfix-reconciliation-v1",
        parameters={
            **children[0][0].parameters,
            "source_node": "E3a-segments",
        },
    )
    assert _screening_job(replacement)

    tts_screen = materialize("TTS-Cal")
    assert {_assigned_gpu(config, job) for job in tts_screen} == {0, 1}
    queues = _single_gpu_queues(config, tts_screen)
    estimated = {
        gpu: sum(
            job.parameters["generation_tokens"] / job.parameters["stride"] for job in jobs
        )
        for gpu, jobs in queues.items()
    }
    assert set(queues) == {0, 1}
    assert max(estimated.values()) / min(estimated.values()) < 1.01
    assert _screening_job(tts_screen[0])

    e0_pair_calibration = next(
        job
        for job in materialize("E0-tune")
        if job.parameters.get("pair_calibration")
    )
    assert _screening_job(e0_pair_calibration)

    remaining_e1 = tuple(
        job for job in materialize("E1") if job.ordinal in {51, 61}
    )
    e1_queues = _single_gpu_queues(config, remaining_e1)
    assert {gpu: len(rows) for gpu, rows in e1_queues.items()} == {0: 1, 1: 1}

    paired = tuple(
        job
        for job in materialize("E3b-final")
        if job.block in {0, 1} and job.parameters["workload"] == "primary_long_history"
    )
    paired_queues = _single_gpu_queues(config, paired)
    block_gpus = {
        block: {
            gpu
            for gpu, rows in paired_queues.items()
            if any(job.block == block for job in rows)
        }
        for block in (0, 1)
    }
    assert block_gpus == {0: {0}, 1: {1}}

    measured = {
        "goodput": 12.0,
        "fallbacks": 1,
        "request_outcomes": {"offered": 19, "completed": 19, "unfinished": 0},
    }
    rejected = _scientific_rejection(measured, 19, RuntimeError("unsafe recipe"))
    assert rejected["scientific_outcome"] == "rejected"
    assert rejected["feasible"] is False
    assert rejected["request_outcomes"] == measured["request_outcomes"]

    runtime_config = {**tts_screen[0].to_dict(), "adaptation": {"method": "tts"}}
    assert _job_from_metric_config(runtime_config) == tts_screen[0]

    assert _all_jobs_completed({"completed": 72})
    assert not _all_jobs_completed({"completed": 42, "pending": 29, "failed": 1})
    unsafe = {
        "committed_tokens": 1,
        "duration_seconds": 1.0,
        "goodput": 1.0,
        "peak_hbm_bytes": 1,
        "kv_capacity": 1,
        "itl_p99_ms": 1.0,
        "version_mismatches": 0,
        "fallbacks": 1,
        "nonfinite_updates": 0,
        "oom_events": 0,
        "retractions": 0,
        "stale_publications": 0,
    }
    with pytest.raises(ScientificFailure, match="fallbacks=1"):
        _validate_measured_metrics(unsafe)


@pytest.mark.parametrize(
    ("interval", "accepted"),
    [
        ((-0.0021634, -0.0062713, -0.0005183), True),
        ((0.0019563, 0.0000536, 0.0043664), True),
        ((0.0, -0.01, 0.01), True),
        ((0.0, -0.011, 0.001), False),
        ((0.0, -0.001, 0.011), False),
        ((math.nan, -0.001, 0.001), False),
        ((0.0, math.nan, 0.001), False),
        ((0.0, -0.001, math.inf), False),
        ((0.0, 0.001, -0.001), False),
    ],
)
def test_headline_parallel_requires_both_intervals_inside_tolerance(interval, accepted):
    for metric in ("goodput", "itl"):
        intervals = {"goodput": (0.0, 0.0, 0.0), "itl": (0.0, 0.0, 0.0)}
        intervals[metric] = interval
        assert _interference_within_tolerance(intervals) is accepted
    assert not _interference_within_tolerance({})
    assert not _interference_within_tolerance({"goodput": interval})


def test_final_bundled_blocks_keep_gpu_affinity_on_partial_resume(tmp_path: Path):
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml", run_name="run", sglang_root=tmp_path,
        results_root=tmp_path, models={}, drafts={}, datasets={}, gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python"), protocol=ProtocolConfig(),
    )
    parents = materialize("E3b-final")
    cells = tuple(child for parent in parents for child in (_segment_jobs(parent) or (parent,)))
    queues = _single_gpu_queues(config, cells)
    assert set(queues) == {0, 1}
    original = {job.job_id: gpu for gpu, rows in queues.items() for job in rows}
    assert len(original) == len(cells)
    assert all(original[job.job_id] == config.gpu_ids[job.block % 2] for job in cells)
    # Remove an uneven subset, including only part of a bundled block.
    pending = cells[1::3]
    resumed = _single_gpu_queues(config, pending)
    assert {
        job.job_id: gpu for gpu, rows in resumed.items() for job in rows
    } == {job.job_id: original[job.job_id] for job in pending}
    for node in ("E5-final", "E6-final", "E0-final"):
        assert all(
            job.gpu_count == 2 and _assigned_pair(config, job) == (0, 1)
            for job in materialize(node)
        )


def test_session_cell_pool_splits_only_independent_segments():
    parents = materialize("E1a")
    children = tuple(child for parent in parents for child in _segment_jobs(parent))
    assert len(parents) == 3
    assert len(children) == 22
    assert all(_session_pool_eligible(job) for job in children)

    pool = _SessionCellPool((job, ("e1a",), 8192.0) for job in children)
    assignments = {0: [], 1: []}
    while len(pool):
        for gpu in assignments:
            job = pool.claim(("e1a",))
            if job is not None:
                assignments[gpu].append(job.job_id)
    assert sum(len(rows) for rows in assignments.values()) == 22
    assert abs(len(assignments[0]) - len(assignments[1])) <= 1

    remaining = _segment_jobs(parents[1])[8:]
    pool = _SessionCellPool((job, ("e1a",), 8192.0) for job in remaining)
    resumed = {0: [], 1: []}
    while len(pool):
        for gpu in resumed:
            job = pool.claim(("e1a",))
            if job is not None:
                resumed[gpu].append(job.job_id)
    assert {gpu: len(rows) for gpu, rows in resumed.items()} == {0: 4, 1: 4}

    atomic = _SessionCellPool((job, ("e1a",), 8192.0) for job in children)

    def drain() -> list[str]:
        claimed = []
        while True:
            job = atomic.claim(("e1a",))
            if job is None:
                return claimed
            claimed.append(job.job_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(drain) for _ in range(2)]
        claimed = sum((future.result() for future in futures), [])
    assert len(claimed) == len(set(claimed)) == 22

    blocked = replace(children[0], block=0)
    tp2 = replace(
        children[0],
        gpu_count=2,
        parameters={**children[0].parameters, "topology": "tp2_dp1"},
    )
    assert not _session_pool_eligible(blocked)
    assert not _session_pool_eligible(tp2)

    width = replace(
        children[0],
        node="E3-width-calibration",
        parameters={
            **children[0].parameters,
            "workload": "excluded_deployment_width_tuning",
        },
    )
    runtime_repair = replace(
        children[0],
        node="bugfix-reconciliation-v1",
        parameters={
            **children[0].parameters,
            "workload": "runtime_repair",
            "reconciliation_kind": "screening_runtime_error_classification",
        },
    )
    assert _session_pool_eligible(width)
    assert _session_pool_eligible(runtime_repair)


def test_numa_plan_reserves_os_cores_and_assigns_disjoint_gpu_siblings():
    rows = _parse_lscpu_rows(
        "# CPU,Core,Socket,Node\n"
        + "\n".join(
            f"{cpu},{cpu % 8},0,{0 if cpu % 8 < 4 else 1}"
            for cpu in range(16)
        )
    )
    plan = _plan_cpu_affinity({0: 0, 1: 1}, rows)
    runner = set(plan["runner_cpus"])
    gpu0 = set(plan["gpus"]["0"]["cpus"])
    gpu1 = set(plan["gpus"]["1"]["cpus"])
    assert runner and gpu0 and gpu1
    assert runner.isdisjoint(gpu0 | gpu1)
    assert gpu0.isdisjoint(gpu1)
    assert plan["gpus"]["0"]["numa_node"] == 0
    assert plan["gpus"]["1"]["numa_node"] == 1


def test_numa_bdf_normalization_and_safe_discovery_fallback(monkeypatch, tmp_path):
    assert _sysfs_pci_bdf("00000000:3B:00.0") == "0000:3b:00.0"
    assert _sysfs_pci_bdf("0000:af:00.0") == "0000:af:00.0"
    monkeypatch.setenv("LIGHTCONE_NUMA_ISOLATION", "1")
    monkeypatch.setattr(
        "lightcone_spec.server.discover_numa_affinity",
        lambda gpu_ids: (_ for _ in ()).throw(RuntimeError("topology unavailable")),
    )
    output = tmp_path / "numa-affinity.json"
    plan = apply_runner_affinity((0, 1), output)
    assert plan["enabled"] is False
    assert plan["fallback"] == "original_process_affinity_and_isolated_gpu_execution"
    assert json.loads(output.read_text())["enabled"] is False


def test_screening_capacity_detects_zero_kv_after_adaptation_headroom():
    error = RuntimeError(
        "SGLang exited during startup with -9: Loaded weights and the 31.735 GiB "
        "unallocated adaptation headroom leave no GPU memory for the KV cache "
        "under --mem-fraction-static=0.88"
    )
    assert _capacity_infeasible(error)


def test_screening_outcome_classification_separates_runtime_and_capacity():
    assert (
        _screening_incomplete_classification([{"status": "timed_out"}])
        == "scientific_infeasible"
    )
    assert (
        _screening_incomplete_classification([{"status": "unfinished"}])
        == "scientific_infeasible"
    )
    assert (
        _screening_incomplete_classification(
            [{"status": "error", "error": "connection refused"}]
        )
        == "runtime_failure"
    )
    assert (
        _screening_incomplete_classification([{"status": "cancelled"}])
        == "interrupted"
    )

    candidate = Job(
        job_id="candidate",
        node="S10-e2-dependency-repair",
        ordinal=0,
        method="lightcone_candidate",
        model="Qwen3-8B",
        backend="DFLASH",
        task="CalibrationMix",
    )
    assert _incomplete_scientific_outcome(candidate, [{"status": "timed_out"}]) == (
        "rejected"
    )
    screening = replace(
        candidate,
        job_id="screening",
        method="static",
        parameters={"source_node": "E3a"},
    )
    assert _incomplete_scientific_outcome(screening, [{"status": "unfinished"}]) == (
        "infeasible"
    )
    ordinary = replace(candidate, job_id="ordinary", method="static")
    assert _incomplete_scientific_outcome(ordinary, [{"status": "timed_out"}]) is None
    e5 = replace(
        ordinary,
        job_id="e5-serving",
        node="E5-pilot",
        parameters={"registered_load": "closed_loop_c256"},
    )
    assert _incomplete_scientific_outcome(e5, [{"status": "unfinished"}]) == (
        "infeasible"
    )
    assert _incomplete_scientific_outcome(e5, [{"status": "error"}]) is None


def test_cosine_schedule_endpoint_and_registered_overrun():
    adaptation = {"optimizer": {"schedule_total_published_updates": 100}}
    assert _schedule_exhausted_updates({"updates_published": 100}, adaptation) == 0
    assert _schedule_exhausted_updates({"updates_published": 101}, adaptation) == 1
    assert _schedule_exhausted_updates({"updates_published": 101}, None) is None
def test_excluded_trajectory_logprobs_preserve_supported_paths():
    import runpy
    from pathlib import Path
    diagnostic = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/diagnose_preview_trajectory.py"))
    assert diagnostic["diagnostic_logprobs"]("target") == 2
    for variant in ("static", "frozen", "active"):
        assert diagnostic["diagnostic_logprobs"](variant) == 0
    assert diagnostic["first_divergence"]([1, 2, 3], [1, 4, 3])["position"] == 1
    assert diagnostic["first_divergence"]([1, 2], [1, 2]) is None
    assert diagnostic["first_divergence"]([1], [1, 2])["target_token"] is None
    params = {"stride": 10, "panel": "preview_v1", "frozen_recipe": {"stride": 10}}
    frozen = diagnostic["frozen_control"](params, 4096)
    assert params["stride"] == params["frozen_recipe"]["stride"] == 10
    control = Job(job_id="excluded", node="excluded", ordinal=0, method="lightcone",
                  model="Qwen/Qwen3-8B", backend="DFLASH", task="MATH-500", parameters=frozen)
    from lightcone_spec.server import adaptation_payload
    assert adaptation_payload(control, frozen["frozen_recipe"])["stride"] == 32769


@pytest.mark.parametrize("mapping_output", [False, True])
def test_context_benchmark_exact_inputs_and_disjoint_source_splits(mapping_output):
    from lightcone_spec.preview_benchmark import DOMAINS, cases, construct_inputs, sample_sets
    pools = {name: [{"problem_id": str(i), "prompt": f"{name}/{i}:" + "x" * 20000}
                    for i in range(10)] for datasets in DOMAINS.values() for name in datasets}
    splits = sample_sets(pools, [])
    assert len(splits["calibration"]) == len(splits["evaluation"]) == 12
    identities = [{(r["dataset"], r["problem_id"]) for r in splits[s]} for s in splits]
    assert all(not a & b for n, a in enumerate(identities) for b in identities[n + 1:])
    # Compact task text, independent long background. The tokenizer models full template IDs.
    for split in ("calibration", "evaluation"):
        for row in splits[split]:
            row["prompt"] = row["prompt"][:40]

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(map(ord, text))

        def apply_chat_template(self, messages, tokenize, **kwargs):
            text = "<user>" + messages[0]["content"] + "</user><assistant>"
            if not tokenize:
                return text
            assert kwargs["return_dict"] is False
            tokens = self.encode(text)
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)} if mapping_output else tokens

    inputs = construct_inputs(Tokenizer(), splits)
    assert len(inputs) == 240
    assert all(r["input_tokens"] == r["bucket"] * 4096 for r in inputs if r["bucket"])
    manifest = {"inputs": inputs}
    assert len(cases(manifest, "calibrate")) == 120
    assert len({c["id"] for c in cases(manifest, "run")}) == 360
    assert sum(c["output_tokens"] for p in ("calibrate", "run") for c in cases(manifest, p)) == 1966080
    assert all(type(token) is int for row in inputs for token in row["input_ids"])
    inputs[0]["input_ids"] = ["input_ids", "attention_mask"]
    inputs[0]["input_tokens"] = 2
    with pytest.raises(ValueError, match="invalid token IDs"):
        cases(manifest, "calibrate")


def test_context_gate_committed_boundary_reset_retraction_and_scope():
    from lightcone_spec.context_gate import ContextGate, validate_gate
    gate = ContextGate(4096)
    assert not gate.observe("request", 4090)
    assert gate.observe("request", 4096)
    assert gate.observe("request", 4080)  # Retraction does not clear activation.
    assert gate.activation_context == 4096
    with pytest.raises(RuntimeError):
        gate.observe("different-request", 5000)
    gate.reset()
    assert gate.observe("different-request", 8192)
    assert not ContextGate(None).observe("r", 40000)
    with pytest.raises(ValueError):
        validate_gate({"threshold": 4096, "max_context": 40960}, algorithm="DFLASH",
                      method="l0", max_in_flight=2, reset_scope="request", dp_size=1)


def test_fixed20k_benchmark_480_evaluation_identities_no_recalibration():
    from lightcone_spec.preview_benchmark import FIXED_GATE, FIXED_MODES, FIXED_VERSION, cases
    inputs = [{"split": split, "sample": i, "bucket": j, "domain": ("Chat", "Code", "Math")[i//4],
               "seed": i, "input_tokens": 10, "input_ids": [1]*10, "output_tokens": 4096}
              for split in ("calibration", "evaluation") for i in range(12) for j in range(10)]
    old = {"inputs": inputs}
    manifest = {**old, "version": FIXED_VERSION, "fixed_gate": FIXED_GATE}
    rows = cases(manifest, "run")
    assert len(rows) == len({r["id"] for r in rows}) == 480
    assert {r["mode"] for r in rows} == set(FIXED_MODES)
    assert all(sum(r["mode"] == m for r in rows) == 120 for m in FIXED_MODES)
    assert {r["split"] for r in rows} == {"evaluation"}
    assert not {r["id"] for r in rows} & {r["id"] for r in cases(old, "run")}
    assert sum(r["output_tokens"] for r in rows) == 1966080
    for i in range(12):
        assert {r["seed"] for r in rows if r["sample"] == i} == {i}
    with pytest.raises(ValueError, match="do not repeat"):
        cases(manifest, "calibrate")
    with pytest.raises(ValueError, match="20480"):
        cases({**manifest, "fixed_gate": {**FIXED_GATE, "threshold": 20000}}, "run")


def test_adaptive_context_v2_preserves_inputs_excludes_static_and_old_identities():
    from lightcone_spec.preview_benchmark import (
        ADAPTIVE_MODES,
        ADAPTIVE_VERSION,
        FIXED_GATE,
        FIXED_VERSION,
        cases,
    )
    inputs = [{"split": "evaluation", "sample": i, "bucket": j, "domain": ("Chat", "Code", "Math")[i//4],
               "seed": i, "input_tokens": 10, "input_ids": [1]*10, "output_tokens": 4096}
              for i in range(12) for j in range(10)]
    old = {"version": FIXED_VERSION, "fixed_gate": FIXED_GATE, "inputs": inputs}
    manifest = {**old, "version": ADAPTIVE_VERSION}
    rows = cases(manifest, "run")
    assert len(rows) == len({r["id"] for r in rows}) == 480
    assert {m: sum(r["mode"] == m for r in rows) for m in ADAPTIVE_MODES} == dict.fromkeys(ADAPTIVE_MODES, 120)
    assert not {r["id"] for r in rows} & {r["id"] for r in cases(old, "run")}
    assert all(r["mode"] != "static" and r["input_ids"] == [1]*10 and r["seed"] == r["sample"] for r in rows)
    with pytest.raises(ValueError, match="do not repeat"):
        cases(manifest, "calibrate")
