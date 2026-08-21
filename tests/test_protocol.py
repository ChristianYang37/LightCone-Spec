import json
import math

import pytest

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.data import load_prompt_records, load_tts_calibration
from lightcone_spec.protocol import (
    PAPER_NODES,
    default_row_counts,
    e2_candidates,
    materialize,
    paper_plan,
)
from lightcone_spec.runner import _e5_execution_phases, _e5_reference, _runtime_job
from lightcone_spec.server import adaptation_payload, server_session_key


def test_registered_node_order_and_counts():
    assert PAPER_NODES == (
        "preflight", "E3a", "TTS-Cal", "E1", "E2-r0", "E2-r1", "E2-r2", "E2-r3",
        "E4-screen", "E4-local", "E4-profile", "E3b-pilot", "E3b-final", "E1a",
        "E5-pilot", "E5-final", "E6-pilot", "E6-final", "E0-tune", "E0-pilot", "E0-final",
    )
    assert default_row_counts() == {
        "preflight": 10, "E3a": 360, "TTS-Cal": 288, "E1": 68,
        "E2-r0": 3364, "E2-r1": 844, "E2-r2": 214, "E2-r3": 57,
        "E4-screen": 48, "E4-local": 96, "E4-profile": 3,
        "E3b-pilot": 1920, "E3b-final": 5760, "E1a": 116,
        "E5-pilot": 2064, "E5-final": 5400, "E6-pilot": 242, "E6-final": 720,
        "E0-tune": 25920, "E0-pilot": 6912, "E0-final": 20736,
    }
    assert len(paper_plan()) == 21


def test_readable_ids_and_dynamic_e0_grid():
    jobs = materialize("E0-tune", valid_e0=[("m", "DFLASH", "task")])
    assert len(jobs) == 108 + 239
    assert jobs[0].job_id.startswith("E0-tune__000000__")
    assert all(len(job.job_id) < 180 for job in jobs)
    candidates = jobs[108 : 108 + 236]
    assert {method: sum(job.method == method for job in candidates) for method in {
        "onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"
    }} == {"onlinespec_ogd": 64, "onlinespec_opt": 64, "onlinespec_ens": 108}
    assert len({repr(sorted(job.parameters.items())) for job in candidates}) == 236


def test_two_gpu_exclusivity_and_real_interface_probes():
    assert {job.gpu_count for job in materialize("E5-pilot")} == {2}
    interfaces = materialize("E6-pilot")[:2]
    assert {job.method for job in interfaces} == {"lightcone"}
    assert all(job.parameters["interface_fit"] for job in interfaces)
    assert all(job.parameters["minimum_updates"] == 1 for job in interfaces)
    probes = materialize("E0-tune", valid_e0=[])[:108]
    assert {job.method for job in probes} == {"static"}
    assert all(job.parameters["adaptive_probe"] for job in probes)


def test_registered_axes_reach_runtime_parameters():
    tts = materialize("TTS-Cal")[0]
    payload = adaptation_payload(tts)
    assert payload["weight_update_mode"] == "full"
    assert payload["optimizer"]["name"] == "adam"
    assert payload["optimizer"]["weight_decay"] == 0
    assert payload["optimizer"]["grad_clip"] == 0
    assert payload["teacher_row_policy"] == "latest_update_round_only"
    assert payload["loss_position_decay"] == math.exp(-1 / 7)

    rounds = [materialize(f"E2-r{index}")[0] for index in range(4)]
    assert [(job.load, job.context) for job in rounds] == [
        ("c2", 4096), ("c4", 8192), ("c8", 16384), ("c16", 40928)
    ]
    assert {job.parameters["regime"] for job in rounds} == {
        "short_input_long_generation"
    }
    state = type(
        "State",
        (),
        {"selection": lambda self, name, default: {"e1_common_load": "c4"}.get(name, default)},
    )()
    capped = _runtime_job(state, rounds[-1])
    assert capped.load == "c4"
    assert capped.parameters["registered_load"] == "c16"
    assert capped.parameters["effective_load"] == "c4"
    screen = materialize("E4-screen")
    assert len(screen) == 48
    assert {job.parameters["stride"] for job in screen} == {1, 50}
    assert {job.parameters["traffic"] for job in screen} == {
        "pure_decode", "mixed_prefill_decode"
    }

    geometries = ({"scope": "all", "parameterization": "full", "rank": None},) * 2
    candidates = e2_candidates(geometries)
    assert len(candidates) == 210
    assert len(materialize("E2-r0", e2_rows=candidates)) == 214


def test_session_key_reuses_scalar_recipe_but_not_layout():
    first, second = materialize("TTS-Cal")[:2]
    assert server_session_key(first) == server_session_key(second)
    lora = materialize("E1")[4]
    full = materialize("E1")[5]
    assert server_session_key(lora) != server_session_key(full)
    context = materialize("E3a")[-1]
    assert server_session_key(context) == server_session_key(
        context.__class__(**{**context.to_dict(), "context": 4096, "load": "c1"})
    )
    high_priority = first.__class__(
        **{
            **first.to_dict(),
            "parameters": {**first.parameters, "stream_priority": "high"},
        }
    )
    assert server_session_key(first) == server_session_key(high_priority)


def test_e5_anchors_execute_before_dependent_rows():
    anchors, dependent = _e5_execution_phases(materialize("E5-pilot"))
    assert len(anchors) == 4 * 2 * 9 * 2
    assert all(job.method in {"target_only", "static"} for job in anchors)
    assert all(job.job_id not in {anchor.job_id for anchor in anchors} for job in dependent)


def test_e5_reference_uses_faster_slo_baseline(monkeypatch):
    anchor = materialize("E5-pilot")[0]
    rows = []
    for method, rate, concurrency in (
        ("target_only", 10.0, 8),
        ("static", 12.0, 16),
    ):
        rows.append(
            (
                {
                    "block": anchor.block,
                    "backend": anchor.backend,
                    "method": method,
                    "load": f"closed_loop_c{concurrency}",
                },
                {"slo_pass": True, "request_rate": rate},
            )
        )
    monkeypatch.setattr("lightcone_spec.runner._metric_rows", lambda state, node: rows)
    assert _e5_reference(object(), anchor) == (12.0, 16)


def test_exact_draft_mapping_and_explicit_tts_split(tmp_path):
    example = ExperimentConfig.load("examples/paper.yaml")
    assert len(example.drafts) == 12
    assert all("|" in key for key in example.drafts)
    rows = []
    for index in range(76):
        rows.append(
            {
                "problem_id": f"t-{index}",
                "split": "tuning",
                "prompt": "p",
                "reference": "r",
            }
        )
    for index in range(4):
        rows.append(
            {
                "problem_id": f"h-{index}",
                "split": "holdout",
                "prompt": "p",
                "reference": "r",
            }
        )
    path = tmp_path / "tts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    tuning, holdout = load_tts_calibration(path)
    assert len(tuning) == 76
    assert holdout == ("h-0", "h-1", "h-2", "h-3")


def test_dataset_split_and_template_are_explicit(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "split": "pilot",
                "question": "2+2",
                "template": "Question: {question}",
                "reference": "4",
            }
        )
        + "\n"
    )
    rows = load_prompt_records(path, limit=1, split="pilot")
    assert rows[0]["prompt"] == "Question: 2+2"
    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps({"problem_id": "p2", "prompt": "x"}) + "\n")
    with pytest.raises(ValueError, match="explicit split"):
        load_prompt_records(missing, limit=1, split="pilot")
    missing_metadata = tmp_path / "missing-metadata.jsonl"
    missing_metadata.write_text(
        json.dumps({"problem_id": "p3", "split": "pilot", "prompt": "x"}) + "\n"
    )
    with pytest.raises(ValueError, match="reference or test metadata"):
        load_prompt_records(missing_metadata, limit=1, split="pilot")

    repeated = load_prompt_records(
        path, limit=3, split="pilot", selection_seed=0, allow_repeat=True
    )
    assert [row["problem_id"] for row in repeated] == ["p1", "p1", "p1"]
    assert [row["repeat_index"] for row in repeated] == [0, 1, 2]

    native = tmp_path / "native.jsonl"
    native.write_text(
        json.dumps(
            {
                "unique_id": "math-1",
                "split": "final",
                "question_content": "Solve it",
                "answer": "42",
            }
        )
        + "\n"
    )
    assert load_prompt_records(native, limit=1, split="final")[0]["problem_id"] == "math-1"


def test_e3_uses_three_explicit_strata_without_filler():
    assert {job.task for job in materialize("E3a")} == {
        "controlled_baseline",
        "LiveCodeBench",
        "MATH-500",
    }
    state = type(
        "State",
        (),
        {
            "selection": lambda self, name, default: {
                "e3a": {"width": 8, "load": "c4"},
                "deployment_widths": {
                    "static": 8,
                    "tts": 16,
                    "l0_naive": 16,
                    "lightcone": 16,
                },
            }.get(name, default)
        },
    )()
    rows = [
        job
        for job in materialize("E3b-pilot")
        if job.parameters["width_panel"] == "deployment_optimal"
    ]
    assert _runtime_job(state, next(job for job in rows if job.method == "static")).width == 8
    assert _runtime_job(state, next(job for job in rows if job.method == "lightcone")).width == 16


def test_final_tail_gate_uses_separate_extension_without_changing_rows():
    from lightcone_spec.protocol import E5_P99_EXTENSION_REQUESTS, E5_P99_MIN_COMPLETED

    final_rows = materialize("E5-final", final_blocks=12)
    saturation = [
        job for job in final_rows if job.load == "saturation_soak"
    ]
    assert len(saturation) == 5 * 2 * 12
    assert E5_P99_EXTENSION_REQUESTS == 11_000
    assert E5_P99_EXTENSION_REQUESTS > E5_P99_MIN_COMPLETED
