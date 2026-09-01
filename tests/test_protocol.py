import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.protocol import (
    CONFIDENCE_WEIGHTS,
    E0_ONLINESPEC_RECIPES,
    FORMAL_ADAPTATION_STRIDE,
    PAPER_NODES,
    TTS_STRIDES,
    default_row_counts,
    materialize,
    paper_plan,
    segment_count,
    uses_formal_adaptation_stride,
)
from lightcone_spec.runner import (
    ScientificFailure,
    _all_jobs_completed,
    _assigned_gpu,
    _assigned_pair,
    _capacity_infeasible,
    _gpu_pairs,
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
from lightcone_spec.server import adaptation_payload, server_session_key

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
    "E1a": 141,
    "E5-pilot": 11,
    "E5-final": 66,
    "E6-pilot": 22,
    "E6-final": 60,
    "E0-tune": 54,
    "E0-pilot": 88,
    "E0-final": 264,
}


def test_paper_v2_node_order_counts_and_plan():
    assert len(PAPER_NODES) == 21
    assert default_row_counts() == EXPECTED
    assert sum(EXPECTED.values()) == 1960
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
    confidence = [
        job.parameters["confidence_loss_weight"]
        for job in materialize("E1a")
        if job.parameters.get("workload") == "confidence_calibration"
    ]
    assert tuple(confidence) == CONFIDENCE_WEIGHTS
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


def test_e5_source_aligned_methods_and_curves():
    pilot = materialize("E5-pilot")
    final = materialize("E5-final")
    assert len(pilot) == 11
    assert len(final) == 66
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


def test_e5_burstgpt_segment_count_is_unchanged():
    assert sum(
        segment.get("load") == "burstgpt_shape"
        for node in ("E5-pilot", "E5-final")
        for job in materialize(node)
        for segment in job.parameters.get("segments", ())
    ) == 73


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


def test_session_cell_pool_splits_only_independent_segments():
    parents = tuple(
        job
        for job in materialize("E1a")
        if job.parameters.get("workload") == "confidence_calibration"
    )
    children = tuple(child for parent in parents for child in _segment_jobs(parent))
    assert len(parents) == len(CONFIDENCE_WEIGHTS)
    assert len(children) == 50
    assert all(_session_pool_eligible(job) for job in children)

    pool = _SessionCellPool((job, ("e1a",), 8192.0) for job in children)
    assignments = {0: [], 1: []}
    while len(pool):
        for gpu in assignments:
            job = pool.claim(("e1a",))
            if job is not None:
                assignments[gpu].append(job.job_id)
    assert {gpu: len(rows) for gpu, rows in assignments.items()} == {0: 25, 1: 25}

    remaining = _segment_jobs(parents[-1])[2:]
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
    assert len(claimed) == len(set(claimed)) == 50

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


def test_cosine_schedule_endpoint_and_registered_overrun():
    adaptation = {"optimizer": {"schedule_total_published_updates": 100}}
    assert _schedule_exhausted_updates({"updates_published": 100}, adaptation) == 0
    assert _schedule_exhausted_updates({"updates_published": 101}, adaptation) == 1
    assert _schedule_exhausted_updates({"updates_published": 101}, None) is None
