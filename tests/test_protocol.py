import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.data import load_calibration_mix, load_prompt_records
from lightcone_spec.protocol import (
    PAPER_NODES,
    default_row_counts,
    e2_candidates,
    materialize,
    paper_plan,
)
from lightcone_spec.runner import (
    _assigned_gpu,
    _assigned_pair,
    _capacity_infeasible,
    _cell_inputs,
    _e5_execution_phases,
    _e5_reference,
    _e6_interface_jobs,
    _e6_role_supported,
    _fit_prompt,
    _gpu_pairs,
    _pair_interference_jobs,
    _resource_port,
    _runtime_job,
    _screening_job,
    _select_tts_recipe,
    _selection_for_job,
)
from lightcone_spec.server import adaptation_payload, server_session_key


def test_registered_node_order_and_counts():
    assert PAPER_NODES == (
        "preflight", "E3a", "TTS-Cal", "E1", "E2-r0", "E2-r1", "E2-r2", "E2-r3",
        "E4-screen", "E4-local", "E4-profile", "E3b-pilot", "E3b-final", "E1a",
        "E5-pilot", "E5-final", "E6-pilot", "E6-final", "E0-tune", "E0-pilot", "E0-final",
    )
    assert default_row_counts() == {
        "preflight": 10, "E3a": 268, "TTS-Cal": 288, "E1": 68,
        "E2-r0": 424, "E2-r1": 109, "E2-r2": 31, "E2-r3": 25,
        "E4-screen": 48, "E4-local": 96, "E4-profile": 3,
        "E3b-pilot": 1360, "E3b-final": 4080, "E1a": 116,
        "E5-pilot": 2064, "E5-final": 5400, "E6-pilot": 282, "E6-final": 840,
        "E0-tune": 2976, "E0-pilot": 3456, "E0-final": 10368,
    }
    assert len(paper_plan()) == 21
    assert {job.task for job in materialize("TTS-Cal")} == {"CalibrationMix"}
    assert {job.task for job in materialize("E1")} == {"CalibrationMix"}
    assert {job.task for job in materialize("E2-r0")} == {"CalibrationMix"}
    assert {job.task for job in materialize("E1a")} == {"CalibrationMix"}
    e0_tasks = {job.task for job in materialize("E0-tune", valid_e0=[])[:108]}
    assert "AlpacaEval" in e0_tasks
    assert "Alpaca" not in e0_tasks


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
    assert {job.task for job in jobs[108:]} == {"CalibrationMix"}


def test_e0_tunes_once_per_model_backend_and_confirms_common_load():
    valid = [("m", "DFLASH", "task-a"), ("m", "DFLASH", "task-b")]
    assert len(materialize("E0-tune", valid_e0=valid)) == 108 + 239
    pilot = materialize("E0-pilot", valid_e0=valid)
    assert len(pilot) == 2 * 4 * 8
    assert {job.load for job in pilot} == {"common_slo_load"}
    recipes = {"m|DFLASH|onlinespec_ogd": {"stride": 40}}
    final = materialize(
        "E0-final", valid_e0=valid[:1], final_blocks=12, e0_recipes=recipes
    )
    ogd = next(job for job in final if job.method == "onlinespec_ogd")
    assert ogd.parameters["stride"] == 40


def test_two_gpu_exclusivity_and_real_interface_probes():
    assert {job.gpu_count for job in materialize("E5-pilot")} == {2}
    interfaces = materialize("E6-pilot")[:2]
    assert {job.method for job in interfaces} == {"lightcone"}
    assert all(job.parameters["interface_fit"] for job in interfaces)
    assert all(job.parameters["minimum_updates"] == 1 for job in interfaces)
    interface_components = _e6_interface_jobs(interfaces)
    assert len(interface_components) == 4
    assert {
        job.parameters["parameterization"] for job in interface_components
    } == {"lora", "full"}
    assert _e6_role_supported("target_only", set())
    assert _e6_role_supported("static", set())
    assert _e6_role_supported("lightcone", {"lora"})
    assert not _e6_role_supported("tts", {"lora"})
    assert _e6_role_supported("l0_naive", {"full"})
    assert all(_screening_job(job) for job in interface_components)
    assert _capacity_infeasible(
        RuntimeError("adaptation peak 9 exceeds pre-KV reserve 8")
    )
    assert not _capacity_infeasible(RuntimeError("NCCL communicator failed"))
    probes = materialize("E0-tune", valid_e0=[])[:108]
    assert {job.method for job in probes} == {"static"}
    assert all(job.parameters["adaptive_probe"] for job in probes)


def test_capacity_screen_reads_only_the_current_server_log(tmp_path):
    server_log = tmp_path / "server.log"
    server_log.write_text("old CUDA out of memory\n")
    offset = server_log.stat().st_size
    server_log.write_text(server_log.read_text() + "scheduler exited\n")
    assert not _capacity_infeasible(RuntimeError("short response"), server_log, offset)
    with server_log.open("a") as stream:
        stream.write("torch.OutOfMemoryError: CUDA out of memory\n")
    assert _capacity_infeasible(RuntimeError("short response"), server_log, offset)


def test_dataset_context_does_not_repeat_its_prompt_pool():
    with pytest.raises(RuntimeError, match="dataset-native context"):
        _fit_prompt((9, 10), (1, 2, 3), 8)


def test_eight_gpu_pair_pool_preserves_block_affinity(tmp_path):
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
    blocks = [
        next(job for job in materialize("E6-final", final_blocks=12) if job.block == block)
        for block in range(4, 8)
    ]
    assert [_assigned_pair(config, job) for job in blocks] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    same_block = [job for job in materialize("E6-final", final_blocks=12) if job.block == 6]
    assert {_assigned_pair(config, job) for job in same_block} == {(4, 5)}
    assert _resource_port(config, (0,)) == 30000
    assert _resource_port(config, (0, 1)) == 30008
    assert _resource_port(config, (2, 3)) == 30010

    singles = [job for job in materialize("E3b-final", final_blocks=12) if job.block == 7]
    assert {_assigned_gpu(config, job) for job in singles} == {7}


def test_even_gpu_config_and_pair_interference_rows(tmp_path):
    source = Path("examples/paper.yaml").read_text()
    eight = tmp_path / "eight.yaml"
    eight.write_text(source.replace("gpu_ids: [0, 1]", "gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]"))
    config = ExperimentConfig.load(eight)
    rows = _pair_interference_jobs(config)
    assert len(rows) == 8
    assert {job.parameters["mode"] for job in rows} == {"isolated", "concurrent"}
    assert {_assigned_pair(config, job) for job in rows} == set(_gpu_pairs(config))

    odd = tmp_path / "odd.yaml"
    odd.write_text(source.replace("gpu_ids: [0, 1]", "gpu_ids: [0, 1, 2]"))
    with pytest.raises(ValueError, match="even number"):
        ExperimentConfig.load(odd)


def test_e6_lightcone_uses_the_fitted_lora_layout():
    class Selections:
        def selection(self, _name, _default):
            return {"parameterization": "full", "rank": None, "scope": "last5"}

    job = next(
        row
        for row in materialize("E6-final", final_blocks=12)
        if row.method == "lightcone"
    )
    selected = _selection_for_job(Selections(), job)
    assert selected is not None
    assert selected["parameterization"] == "lora"
    assert selected["rank"] == 8
    assert selected["scope"] == "all"


def test_registered_axes_reach_runtime_parameters():
    tts = materialize("TTS-Cal")[0]
    payload = adaptation_payload(tts)
    assert payload["weight_update_mode"] == "full"
    assert payload["optimizer"]["name"] == "adam"
    assert payload["optimizer"]["weight_decay"] == 0
    assert payload["optimizer"]["grad_clip"] == 0
    assert payload["teacher_row_policy"] == "latest_update_round_only"
    assert payload["loss_position_decay"] == math.exp(-1 / 7)
    for node in ("TTS-Cal", "E1", "E1a"):
        jobs = materialize(node)
        assert {job.parameters["regime"] for job in jobs} == {
            "short_input_long_generation"
        }
    assert {job.parameters["generation_tokens"] for job in materialize("TTS-Cal")} == {4096}
    assert {job.parameters["generation_tokens"] for job in materialize("E1")} == {8192}
    assert {job.parameters["generation_tokens"] for job in materialize("E1a")} == {8192}

    rounds = [materialize(f"E2-r{index}")[0] for index in range(4)]
    assert [(job.load, job.context) for job in rounds] == [
        ("c2", 4096), ("c4", 8192), ("c8", 16384), ("c16", 40928)
    ]
    assert {job.parameters["regime"] for job in rounds} == {
        "short_input_long_generation"
    }
    assert [job.parameters["generation_tokens"] for job in rounds] == [
        2048, 4096, 8192, 16384
    ]
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


def test_long_generation_uses_one_checkpointed_trajectory_per_condition():
    e3a = materialize("E3a")
    trajectories = [
        job for job in e3a
        if job.parameters.get("regime") == "short_input_long_generation"
    ]
    assert len(trajectories) == 28
    assert {job.context for job in trajectories} == {40928}
    assert {job.parameters["generation_tokens"] for job in trajectories} == {40800}
    assert trajectories[0].parameters["generation_checkpoints"] == (
        1024, 2048, 4096, 8192, 16384, 24576, 32768, 40800
    )

    e3b = materialize("E3b-pilot")
    assert sum(
        job.parameters.get("regime") == "short_input_long_generation"
        for job in e3b
    ) == 80
    assert sum(
        job.parameters.get("regime") == "long_input_short_output"
        for job in e3b
    ) == 640


def test_tts_recipe_uses_all_four_disjoint_windows(monkeypatch):
    rows = []
    for recipe, values in (("stable", (3, 3, 3, 3)), ("lucky", (10, 0, 0, 0))):
        for value in values:
            rows.append(
                (
                    {"parameters": {"recipe": recipe}},
                    {
                        "goodput": value,
                        "peak_hbm_bytes": 1,
                        "itl_p99_ms": 1.0,
                    },
                )
            )
    monkeypatch.setattr("lightcone_spec.runner._metric_rows", lambda *_: rows)
    assert _select_tts_recipe(object())["recipe"] == "stable"


def test_session_key_reuses_scalar_recipe_but_not_layout():
    first, second = materialize("TTS-Cal")[:2]
    assert server_session_key(first) == server_session_key(second)
    lora = materialize("E1")[4]
    full = materialize("E1")[5]
    assert server_session_key(lora) != server_session_key(full)
    context = materialize("E3a")[-1]
    assert server_session_key(context) == server_session_key(
        context.__class__(**{**context.to_dict(), "context": 4096})
    )
    assert server_session_key(context) != server_session_key(
        context.__class__(**{**context.to_dict(), "load": "c1"})
    )
    high_priority = first.__class__(
        **{
            **first.to_dict(),
            "parameters": {**first.parameters, "stream_priority": "high"},
        }
    )
    assert server_session_key(first) == server_session_key(high_priority)
    e6 = next(job for job in materialize("E6-pilot") if job.backend == "NEXTN")
    e6_higher_load = e6.__class__(**{**e6.to_dict(), "load": "c2"})
    assert server_session_key(e6) != server_session_key(e6_higher_load)


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


def test_exact_draft_mapping_and_calibration_mix(tmp_path):
    example = ExperimentConfig.load("examples/paper.yaml")
    assert example.server.cuda_home == Path("/usr/local/cuda-12.9")
    assert len(example.drafts) == 12
    assert all("|" in key for key in example.drafts)
    rows = [
        {"problem_id": f"{source}-{index}", "prompt": "p", "source": source}
        for source, count in {
            "APPS": 24,
            "OpenR1-Math": 24,
            "UltraChat": 24,
            "controlled_synthetic": 4,
        }.items()
        for index in range(count)
    ]
    path = tmp_path / "tts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert len(load_calibration_mix(path)) == 76

    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="run",
        sglang_root=tmp_path / "sglang",
        results_root=tmp_path,
        models={},
        drafts={},
        datasets={"CalibrationMix": path},
        gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python"),
        protocol=ProtocolConfig(),
    )

    class Tokenizer:
        @staticmethod
        def tokenize(prompt):
            return tuple(range(1, len(prompt.split()) + 2))

    job = materialize("TTS-Cal")[0]
    prompts, max_new_tokens, metadata = _cell_inputs(config, object(), Tokenizer(), job)
    assert len(prompts) == 19
    assert max_new_tokens == 4096
    assert max(map(len, prompts)) <= 128
    assert metadata["regime"] == "short_input_long_generation"


def test_preflight_covers_short_and_long_context_without_changing_count():
    jobs = materialize("preflight")
    assert len(jobs) == 10
    interference = jobs[2:]
    assert {job.context for job in interference if job.block == 0} == {4096}
    assert {job.context for job in interference if job.block == 1} == {40928}
    assert {job.task for job in interference if job.block == 1} == {"MATH-500"}
    assert all(job.load == "c1" for job in interference)


def test_prompt_pool_needs_only_identity_and_renderable_prompt(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "question": "2+2",
                "template": "Question: {question}",
            }
        )
        + "\n"
    )
    rows = load_prompt_records(path, limit=1)
    assert rows[0]["prompt"] == "Question: 2+2"
    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps({"prompt": "x"}) + "\n")
    with pytest.raises(ValueError, match="requires problem_id"):
        load_prompt_records(missing, limit=1)

    repeated = load_prompt_records(
        path, limit=3, selection_seed=0, allow_repeat=True
    )
    assert [row["problem_id"] for row in repeated] == ["p1", "p1", "p1"]
    assert [row["repeat_index"] for row in repeated] == [0, 1, 2]

    native = tmp_path / "native.jsonl"
    native.write_text(
        json.dumps(
            {
                "unique_id": "math-1",
                "question_content": "Solve it",
            }
        )
        + "\n"
    )
    assert load_prompt_records(native, limit=1)[0]["problem_id"] == "math-1"

    chat = tmp_path / "chat.jsonl"
    chat.write_text(
        json.dumps({"problem_id": "chat-1", "turns": ["hello", "continue"]})
        + "\n"
    )
    assert load_prompt_records(chat, limit=1)[0]["prompt"] == "hello\ncontinue"

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        json.dumps({"problem_id": "p", "prompt": "a"})
        + "\n"
        + json.dumps({"problem_id": "p", "prompt": "b"})
        + "\n"
    )
    with pytest.raises(ValueError, match="repeats problem_id"):
        load_prompt_records(duplicate, limit=1)


def test_dataset_conversion_keeps_only_workload_inputs(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"problem_id": "p1", "prompt": "hello"}) + "\n")
    output = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_datasets.py",
            "--task",
            f"controlled_baseline={source}",
            "--output-root",
            str(output),
        ],
        check=True,
    )
    row = json.loads((output / "controlled_baseline.jsonl").read_text())
    assert row == {"problem_id": "p1", "prompt": "hello", "turns": None}


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
