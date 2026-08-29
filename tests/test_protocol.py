import math
from pathlib import Path

import pytest

from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.protocol import (
    CONFIDENCE_WEIGHTS,
    PAPER_NODES,
    TTS_STRIDES,
    default_row_counts,
    materialize,
    paper_plan,
    segment_count,
)
from lightcone_spec.runner import (
    ScientificFailure,
    _all_jobs_completed,
    _assigned_gpu,
    _assigned_pair,
    _gpu_pairs,
    _job_from_metric_config,
    _scientific_rejection,
    _screening_job,
    _segment_jobs,
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
    "E0-tune": 287,
    "E0-pilot": 86,
    "E0-final": 258,
}


def test_paper_v2_node_order_counts_and_plan():
    assert len(PAPER_NODES) == 21
    assert default_row_counts() == EXPECTED
    assert sum(EXPECTED.values()) == 2185
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
    primary = [job for job in e3b if job.parameters["workload"] == "primary_long_history"]
    secondary = [job for job in e3b if job.parameters["workload"].startswith("secondary_")]
    assert len(primary) == 60
    assert {job.block for job in primary} == set(range(12))
    assert len(secondary) == 72
    assert {job.block for job in secondary} == set(range(6))
    assert len(materialize("E6-final")) == 2 * 5 * 6
    assert len(materialize("E0-final")) == 43 * 6


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
    assert sum(job.method.startswith("onlinespec") for job in rows) == 12
    assert {(job.model, job.backend) for job in rows if job.method.startswith("onlinespec")} == {
        ("Qwen/Qwen3-8B", "DFLASH")
    }


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
