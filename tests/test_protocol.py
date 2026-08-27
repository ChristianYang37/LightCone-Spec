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
    _assigned_gpu,
    _assigned_pair,
    _gpu_pairs,
    _scientific_rejection,
    _screening_job,
    _segment_jobs,
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
    "E4-screen": 48,
    "E4-local": 168,
    "E4-profile": 3,
    "E3b-pilot": 20,
    "E3b-final": 132,
    "E1a": 141,
    "E5-pilot": 53,
    "E5-final": 160,
    "E6-pilot": 22,
    "E6-final": 60,
    "E0-tune": 287,
    "E0-pilot": 86,
    "E0-final": 258,
}


def test_paper_v2_node_order_counts_and_plan():
    assert len(PAPER_NODES) == 21
    assert default_row_counts() == EXPECTED
    assert sum(EXPECTED.values()) == 2317
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
    assert payload["teacher_row_policy"] == "latest_update_round_only"
    assert payload["loss_position_decay"] == pytest.approx(math.exp(-1 / 7))
    confidence = [
        job.parameters["confidence_loss_weight"]
        for job in materialize("E1a")
        if job.parameters.get("workload") == "confidence_calibration"
    ]
    assert tuple(confidence) == CONFIDENCE_WEIGHTS


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
    assert _screening_job(tts_screen[0])

    measured = {
        "goodput": 12.0,
        "fallbacks": 1,
        "request_outcomes": {"offered": 19, "completed": 19, "unfinished": 0},
    }
    rejected = _scientific_rejection(measured, 19, RuntimeError("unsafe recipe"))
    assert rejected["scientific_outcome"] == "rejected"
    assert rejected["feasible"] is False
    assert rejected["request_outcomes"] == measured["request_outcomes"]
