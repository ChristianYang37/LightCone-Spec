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
