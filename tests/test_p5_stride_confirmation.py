from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from lightcone_spec.artifacts.rundir import RunDirectory
from lightcone_spec.artifacts.schemas import TABLES
from lightcone_spec.locking.hashing import canonical_json, sha256_file
from lightcone_spec.orchestration.catalog import (
    P5_PRIORITY_CONFIRMATION_LOADS,
    P5_PRIORITY_CONFIRMATION_MIN_PROMPT_CLUSTERS,
    P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF,
    p5_priority_dflash_stride_confirmation_manifest,
    p5_priority_dflash_stride_screen_manifest,
)
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.statistics.tables import p5_prompt_acceptance_table


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/p5_stride_confirmation.py"
SELECTOR_SCRIPT = ROOT / "scripts/experiments/select_p5_stride_screen.py"
MATCHED_FIXTURE = (
    ROOT / "tests/test_build_p5_matched_controller_manifests.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("p5_stride_confirmation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selector_module():
    spec = importlib.util.spec_from_file_location(
        "select_p5_stride_screen_for_confirmation", SELECTOR_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _matched_fixture_module():
    spec = importlib.util.spec_from_file_location(
        "matched_controller_fixture_for_confirmation", MATCHED_FIXTURE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sidecar(path: Path, sidecar: Path | None = None) -> Path:
    sidecar = sidecar or Path(str(path) + ".sha256")
    sidecar.write_text(sha256_file(path) + "\n", encoding="utf-8")
    return sidecar


def _selector(
    tmp_path: Path, *, tts_stride: int = 16, l0_stride: int = 8
) -> tuple[Path, Path]:
    screen_path = tmp_path / "screen.json"
    screen = p5_priority_dflash_stride_screen_manifest()
    screen.write(screen_path)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "cells": {
                    unit.unit_id: {
                        "unit_id": unit.unit_id,
                        "method": unit.method,
                        "stride": unit.stride,
                        "status": "complete_valid",
                    }
                    for unit in screen.units
                },
                "summary": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _sidecar(coverage)

    rows = []
    for context in (4096, 16384):
        static_acceptance = 3.0 if context == 16384 else 4.0
        rows.append(
            {
                "method": "static",
                "update_stride": 1,
                "context_length": context,
                "survival_weighted_accepted_prefix": static_acceptance,
                "acceptance_gain_vs_baseline": 0.0,
                "target_calls_per_output_token": 0.25,
                "decode_goodput_tps": 100.0,
                "exactness_violations": 0,
                "version_mismatch_count": 0,
            }
        )
        for stride in (1, 4, 8, 16):
            tts_gain = 1.0 if stride == tts_stride else 0.5
            l0_gain = 1.3 if stride == l0_stride else 0.6
            for method, gain, target_calls in (
                ("tts", tts_gain, 0.18 if stride == tts_stride else 0.21),
                (
                    "naive_async",
                    l0_gain,
                    0.15 if stride == l0_stride else 0.20,
                ),
            ):
                rows.append(
                    {
                        "method": method,
                        "update_stride": stride,
                        "context_length": context,
                        "survival_weighted_accepted_prefix": (
                            static_acceptance + gain
                        ),
                        "acceptance_gain_vs_baseline": gain,
                        "target_calls_per_output_token": target_calls,
                        "decode_goodput_tps": 120.0 + stride,
                        "exactness_violations": 0,
                        "version_mismatch_count": 0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["adaptation_fallback_count"] = 0

    def write_analysis(root: Path, baseline: str) -> None:
        root.mkdir()
        table = frame.copy()
        if baseline == "tts":
            for context in (4096, 16384):
                for stride in (1, 4, 8, 16):
                    tts_acceptance = float(
                        table[
                            (table.method == "tts")
                            & (table.update_stride == stride)
                            & (table.context_length == context)
                        ].iloc[0].survival_weighted_accepted_prefix
                    )
                    mask = (
                        (table.method == "naive_async")
                        & (table.update_stride == stride)
                        & (table.context_length == context)
                    )
                    table.loc[mask, "acceptance_gain_vs_baseline"] = (
                        table.loc[mask, "survival_weighted_accepted_prefix"]
                        - tts_acceptance
                    )
        table_path = root / "p5_long_context_acceptance.parquet"
        table.to_parquet(table_path, index=False)
        analysis_path = root / "analysis-manifest.json"
        analysis_path.write_text(
            canonical_json(
                {
                    "schema_version": 1,
                    "analysis": {
                        "baseline": baseline,
                        "expected_manifest_sha256": screen.content_sha256(),
                    },
                    "input_runs": [
                        {"run_id": unit.unit_id, "unit_id": unit.unit_id}
                        for unit in screen.units
                    ],
                    "derived_outputs": {},
                }
            ),
            encoding="utf-8",
        )
        analysis_sidecar = _sidecar(
            analysis_path, root / "analysis-manifest.sha256"
        )
        ledger = {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (table_path, analysis_path, analysis_sidecar)
        }
        (root / "analysis-hashes.json").write_text(
            json.dumps(ledger, sort_keys=True), encoding="utf-8"
        )

    static_root = tmp_path / "vs-static"
    tts_root = tmp_path / "vs-tts"
    write_analysis(static_root, "static")
    write_analysis(tts_root, "tts")
    selector_path = tmp_path / "stride-selection.json"
    _selector_module().select(
        manifest_path=screen_path,
        coverage_path=coverage,
        vs_static_root=static_root,
        vs_tts_root=tts_root,
        output_path=selector_path,
    )
    return screen_path, selector_path


def _build(
    module, tmp_path: Path, *, tts_stride: int = 16, l0_stride: int = 8
):
    screen, selector = _selector(
        tmp_path, tts_stride=tts_stride, l0_stride=l0_stride
    )
    manifest = tmp_path / "confirmation.json"
    receipt = tmp_path / "generation.json"
    payload = module.build_confirmation(
        selector_path=selector,
        screen_manifest_path=screen,
        output_manifest_path=manifest,
        output_receipt_path=receipt,
    )
    return screen, selector, manifest, receipt, payload


def _prompt_frame(
    *,
    tts_strides: tuple[int, ...],
    missing_baseline_prompt: bool = False,
    num_prompts: int = 40,
) -> pd.DataFrame:
    rows = []
    for context, concurrency in (
        (4096, 8),
        (16384, 8),
        (4096, 48),
        (16384, 20),
    ):
        for prompt in range(num_prompts):
            common = {
                "model_pair": "qwen3_4b_dflash16",
                "weight_update_mode": "lora",
                "dataset": "livecodebench",
                "lifecycle": "stream",
                "offered_concurrency": concurrency,
                "context_length": context,
                "prompt_cluster": f"prompt-{prompt}",
                "seed": 0,
                "round_count": 10,
                "benchmark_repetitions": 5,
            }
            baseline = 2.0 + context / 100000.0 + prompt / 100.0
            rows.append(
                {
                    **common,
                    "method": "static",
                    "update_stride": 1,
                    "accepted_sum": (baseline - 0.20) * 10,
                    "acceptance": baseline - 0.20,
                }
            )
            offsets = {1: -0.15, 4: -0.12, 8: -0.10, 16: 0.0}
            for stride in tts_strides:
                offset = offsets[stride]
                if (
                    missing_baseline_prompt
                    and stride == 16
                    and context == 16384
                    and concurrency == 20
                    and prompt == 5
                ):
                    continue
                rows.append(
                    {
                        **common,
                        "method": "tts",
                        "update_stride": stride,
                        "accepted_sum": (baseline + offset) * 10,
                        "acceptance": baseline + offset,
                    }
                )
            gain = 0.10 + prompt / 100.0
            rows.append(
                {
                    **common,
                    "method": "naive_async",
                    "update_stride": 8,
                    "accepted_sum": (baseline + gain) * 10,
                    "acceptance": baseline + gain,
                }
            )
    return pd.DataFrame(rows)


def _analysis(
    root: Path,
    manifest_path: Path,
    *,
    missing_baseline_prompt: bool = False,
    num_prompts: int = 40,
    input_runs: list[dict] | None = None,
) -> None:
    root.mkdir(parents=True)
    manifest = ExperimentManifest.load(manifest_path)
    prompt_path = root / "p5_prompt_acceptance.parquet"
    prompt_frame = _prompt_frame(
        tts_strides=tuple(
            sorted({unit.stride for unit in manifest.units if unit.method == "tts"})
        ),
        missing_baseline_prompt=missing_baseline_prompt,
        num_prompts=num_prompts,
    )
    prompt_frame.to_parquet(prompt_path, index=False)
    safety_path = root / "p5_long_context_acceptance.parquet"
    safety_identity = [
        "method",
        "model_pair",
        "weight_update_mode",
        "update_stride",
        "dataset",
        "lifecycle",
        "offered_concurrency",
        "context_length",
    ]
    safety = prompt_frame[safety_identity].drop_duplicates().copy()
    safety["adaptation_fallback_count"] = 0
    safety["exactness_violations"] = 0
    safety["version_mismatch_count"] = 0
    safety.to_parquet(safety_path, index=False)
    analysis_path = root / "analysis-manifest.json"
    body = {
        "schema_version": 1,
        "analysis": {
            "baseline": "static",
            "expected_manifest_sha256": manifest.content_sha256(),
            "weight_update_mode_overlay": "lora",
            "methods_overlay": ["static", "tts", "naive_async"],
            "lifecycles_overlay": None,
            "learning_rate_overlay": None,
        },
        "input_runs": input_runs or [
            {
                "run_id": unit.unit_id,
                "unit_id": unit.unit_id,
                "manifest_sha256": f"{index + 1:064x}",
                "hashes_sha256": f"{index + 101:064x}",
            }
            for index, unit in enumerate(manifest.units)
        ],
        "derived_outputs": {
            prompt_path.name: {
                "sha256": sha256_file(prompt_path),
                "bytes": prompt_path.stat().st_size,
            },
            safety_path.name: {
                "sha256": sha256_file(safety_path),
                "bytes": safety_path.stat().st_size,
            },
        },
    }
    analysis_path.write_text(canonical_json(body), encoding="utf-8")
    analysis_sidecar = _sidecar(
        analysis_path, root / "analysis-manifest.sha256"
    )
    ledger = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (prompt_path, safety_path, analysis_path, analysis_sidecar)
    }
    (root / "analysis-hashes.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _confirmation_artifacts(
    root: Path,
    manifest_path: Path,
    generation: dict,
    *,
    lockfile: Path,
    model_roots: Path,
) -> list[dict]:
    root.mkdir(parents=True)
    manifest = ExperimentManifest.load(manifest_path)
    bindings = generation["generation_identity"]["execution_bindings"]
    dataset = root / "dataset-preflight.json"
    dataset.write_text(
        canonical_json(
            {
                "schema_version": 1,
                "lockfile_sha256": bindings["lockfile_sha256"],
                "limit": manifest.engine_params["prompt_limit"],
                "offset": manifest.engine_params["prompt_offset"],
                "datasets": [
                    {
                        "adapter_key": "livecodebench",
                        "revision": "dataset-revision",
                        "selected_count": manifest.engine_params["prompt_limit"],
                        "selected_sample_ids_sha256": "d" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _sidecar(dataset)
    dataset_sha256 = sha256_file(dataset)
    rows = []
    for index, unit in enumerate(manifest.units):
        run_id = f"confirmation-{index:02d}"
        rd = RunDirectory(root, run_id)
        rd.path.mkdir(parents=True)
        config = rd.path / "adaptation.runtime.yaml"
        config.write_text("schema_version: 1\n", encoding="utf-8")
        engine = {
            **manifest.engine_params,
            "lockfile_path": str(lockfile.resolve()),
            "model_roots_path": str(model_roots.resolve()),
            "locked_target_revision": bindings["model_revisions"]["target"],
            "locked_drafter_revision": bindings["model_revisions"]["drafter"],
            "weight_update_mode_override": "lora",
            "dataset_preflight_sha256": dataset_sha256,
            "adaptation_config_path": str(config.resolve()),
            "runtime_config_sha256": sha256_file(config),
        }
        run_manifest = unit.to_manifest_dict()
        run_manifest.update(
            {
                "run_id": run_id,
                "engine_params": engine,
                "experiment_manifest_sha256": "e" * 64,
                "unit_execution_sha256": f"{index + 1000:064x}",
            }
        )
        rd.create(run_manifest)
        runtime = rd.path / "runtime"
        runtime.mkdir()
        (runtime / "adaptation-telemetry-rank0.jsonl").write_text(
            '{"event":"round"}\n', encoding="utf-8"
        )
        (rd.path / "prefix-checkpoints.json").write_text(
            json.dumps({"checkpoints": [{"prefix_len": 4096}]}),
            encoding="utf-8",
        )
        for table in TABLES:
            rd.write_table(table, [])
        rd.finalize(exit_code=0, status="complete_valid")
        rows.append(
            {
                "run_id": run_id,
                "unit_id": unit.unit_id,
                "manifest_sha256": sha256_file(rd.path / "manifest.json"),
                "hashes_sha256": sha256_file(rd.path / "hashes.json"),
            }
        )
    return rows


def test_confirmation_catalog_covers_three_loads_and_deduplicates_equal_stride():
    different = p5_priority_dflash_stride_confirmation_manifest(
        tts_acceptance_stride=16,
        tts_engineering_stride=4,
        l0_stride=8,
    )
    assert len(different.units) == 15
    assert len({unit.unit_id for unit in different.units}) == 15
    assert {
        (unit.prompt_subset, unit.concurrency) for unit in different.units
    } == set(P5_PRIORITY_CONFIRMATION_LOADS)
    assert different.engine_params == {
        "prompt_limit": 48,
        "prompt_offset": 40,
        "benchmark_repetitions": 5,
        "max_new_tokens": 512,
        "ignore_eos": True,
        "max_running_requests": 48,
        "max_total_tokens": 400000,
        "p5_context_lengths": [4096, 16384],
        "p5_context_timing_contract": "independent_exact_context_group_v1",
        "pytorch_cuda_alloc_conf": P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF,
        "peak_tflops_per_gpu": 500.0,
        "peak_tflops_basis": (
            "nvidia_official_1pflops_bf16_sparse_dense_inferred_half_v1"
        ),
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_prompts": 20,
        "trace_level": "light",
        "claim_scope": "paired_stride_confirmation",
        "p5_confirmation_min_paired_prompt_clusters": (
            P5_PRIORITY_CONFIRMATION_MIN_PROMPT_CLUSTERS
        ),
        "request_timeout_s": 1800,
    }

    equal = p5_priority_dflash_stride_confirmation_manifest(
        tts_acceptance_stride=8,
        tts_engineering_stride=8,
        l0_stride=8,
    )
    assert len(equal.units) == 9
    assert len({unit.unit_id for unit in equal.units}) == 9
    assert [unit.method for unit in equal.units].count("tts") == 3
    with pytest.raises(ValueError, match="tts_acceptance_stride"):
        p5_priority_dflash_stride_confirmation_manifest(
            tts_acceptance_stride=2,
            tts_engineering_stride=8,
            l0_stride=8,
        )


def test_prompt_artifact_preserves_physical_static_stride_and_repetitions():
    rows = []
    for method, stride, accepted in (("static", 1, 2), ("tts", 16, 3)):
        rows.append(
            {
                "method": method,
                "model_pair": "qwen3_4b_dflash16",
                "weight_update_mode": "lora",
                "update_stride": stride,
                "dataset": "livecodebench",
                "lifecycle": "stream",
                "offered_concurrency": 8,
                "context_length": 4096,
                "request_id": "prompt-0",
                "prompt_cluster": "prompt-0",
                "seed": 0,
                "draft_tokens": 16,
                "verify_len": 17,
                "accepted_drafts": accepted,
                "committed_per_verify": accepted + 1,
                "target_calls": 1,
                "draft_cuda_us": 1.0,
                "verify_cuda_us": 2.0,
                "accept_cuda_us": 1.0,
                "batch_size": 8,
                "version_canary_ok": True,
                "prefix_len_before": 4096,
                "benchmark_repetitions": 5,
            }
        )
    prompt = p5_prompt_acceptance_table(pd.DataFrame(rows))
    static = prompt[prompt.method == "static"]
    assert static.update_stride.tolist() == [1]
    assert set(prompt.benchmark_repetitions) == {5}


def test_p5_fallback_counter_deduplicates_requests_then_sums_runs():
    from lightcone_spec.cli.main import _aggregate_p5_run_counter

    frame = pd.DataFrame(
        [
            {
                "method": "tts",
                "context_length": 4096,
                "analysis_run_id": run,
                "adaptation_fallback_count": count,
            }
            for run, count, copies in (("run-a", 2, 3), ("run-b", 3, 2))
            for _ in range(copies)
        ]
    )
    table = _aggregate_p5_run_counter(
        frame,
        ["method", "context_length"],
        counter="adaptation_fallback_count",
    )
    assert table.adaptation_fallback_count.tolist() == [5]
    frame.loc[0, "adaptation_fallback_count"] = -1
    with pytest.raises(ValueError, match="nonnegative integers"):
        _aggregate_p5_run_counter(
            frame,
            ["method", "context_length"],
            counter="adaptation_fallback_count",
        )


@pytest.mark.parametrize(
    ("tts_stride", "l0_stride", "expected_units"),
    [(16, 8, 12), (8, 8, 12)],
)
def test_generation_receipt_binds_selector_roles_and_manifest(
    tmp_path, tts_stride, l0_stride, expected_units
):
    module = _module()
    _, selector, manifest_path, receipt, payload = _build(
        module, tmp_path, tts_stride=tts_stride, l0_stride=l0_stride
    )
    manifest = ExperimentManifest.load(manifest_path)
    assert len(manifest.units) == expected_units
    assert payload["status"] == "ready_for_execution"
    assert payload["generation_identity"]["selector_sha256"] == sha256_file(
        selector
    )
    assert payload["generation_identity"][
        "confirmation_manifest_sha256"
    ] == manifest.content_sha256()
    assert all(len(rows) == 3 for rows in payload["roles"].values())
    assert payload["analysis_contract"][
        "min_paired_prompt_clusters_per_cell"
    ] == 32
    assert payload["generation_identity"][
        "pytorch_cuda_alloc_conf"
    ] == P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF
    if tts_stride == l0_stride:
        assert payload["roles"]["tts_acceptance_best"] == payload["roles"][
            "same_stride_tts_for_l0"
        ]
    assert Path(str(receipt) + ".sha256").read_text().strip() == sha256_file(
        receipt
    )


def test_generation_rejects_selector_unit_identity_drift(tmp_path):
    module = _module()
    screen, selector = _selector(tmp_path)
    payload = json.loads(selector.read_text())
    payload["winners"]["l0_best"]["unit_id"] = "0" * 64
    selector.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    _sidecar(selector)
    with pytest.raises(module.ConfirmationError, match="semantic validation failed"):
        module.build_confirmation(
            selector_path=selector,
            screen_manifest_path=screen,
            output_manifest_path=tmp_path / "confirmation.json",
            output_receipt_path=tmp_path / "generation.json",
        )


def test_cross_stride_receipt_has_prompt_paired_bca_and_hash_identity(tmp_path):
    module = _module()
    screen, selector, manifest, generation, _ = _build(module, tmp_path)
    analysis_root = tmp_path / "analysis"
    _analysis(analysis_root, manifest)
    output = tmp_path / "cross-stride.json"
    payload = module.compare_confirmation(
        selector_path=selector,
        screen_manifest_path=screen,
        generation_receipt_path=generation,
        confirmation_manifest_path=manifest,
        analysis_root=analysis_root,
        output_path=output,
        bootstrap_replicates=200,
    )
    assert payload["status"] == "comparison_complete"
    assert payload["scientific_sample_pass"] is True
    assert payload["all_cells_ci_low_positive"] is True
    assert payload["raw_provenance_pass"] is False
    assert payload["formal_acceptance_claim_pass"] is False
    assert set(payload["results"]) == {
        "tts_acceptance_best_vs_static",
        "tts_engineering_best_vs_static",
        "same_stride_tts_for_l0_vs_static",
        "l0_best_vs_tts_acceptance_best",
        "l0_best_vs_tts_engineering_best",
        "l0_best_vs_same_stride_tts_for_l0",
    }
    assert all(len(rows) == 4 for rows in payload["results"].values())
    l0_rows = payload["results"]["l0_best_vs_tts_acceptance_best"]
    tts_rows = payload["results"]["tts_acceptance_best_vs_static"]
    assert {row["paired_prompt_clusters"] for row in l0_rows} == {40}
    assert {row["candidate_update_stride"] for row in l0_rows} == {8}
    assert {row["baseline_update_stride"] for row in l0_rows} == {16}
    assert all(row["acceptance_gain"] == pytest.approx(0.295) for row in l0_rows)
    assert {row["candidate_update_stride"] for row in tts_rows} == {16}
    assert {row["baseline_update_stride"] for row in tts_rows} == {1}
    assert all(row["acceptance_gain"] == pytest.approx(0.20) for row in tts_rows)
    identity = payload["analysis_identity"]
    assert identity["selector_sha256"] == sha256_file(selector)
    assert identity["generation_receipt_sha256"] == sha256_file(generation)
    assert identity["confirmation_manifest_sha256"] == ExperimentManifest.load(
        manifest
    ).content_sha256()
    assert Path(str(output) + ".sha256").read_text().strip() == sha256_file(output)


def test_cross_stride_comparison_fails_on_unpaired_prompt(tmp_path):
    module = _module()
    screen, selector, manifest, generation, _ = _build(module, tmp_path)
    analysis_root = tmp_path / "analysis"
    _analysis(analysis_root, manifest, missing_baseline_prompt=True)
    with pytest.raises(module.ConfirmationError, match="prompt/seed coverage differs"):
        module.compare_confirmation(
            selector_path=selector,
            screen_manifest_path=screen,
            generation_receipt_path=generation,
            confirmation_manifest_path=manifest,
            analysis_root=analysis_root,
            output_path=tmp_path / "cross-stride.json",
            bootstrap_replicates=40,
        )


def test_cross_stride_formal_gate_requires_32_prompt_clusters(tmp_path):
    module = _module()
    screen, selector, manifest, generation, _ = _build(module, tmp_path)
    analysis_root = tmp_path / "analysis"
    _analysis(analysis_root, manifest, num_prompts=8)
    payload = module.compare_confirmation(
        selector_path=selector,
        screen_manifest_path=screen,
        generation_receipt_path=generation,
        confirmation_manifest_path=manifest,
        analysis_root=analysis_root,
        output_path=tmp_path / "cross-stride.json",
        bootstrap_replicates=40,
    )
    assert payload["scientific_sample_pass"] is False
    assert payload["all_cells_ci_low_positive"] is True
    assert payload["formal_acceptance_claim_pass"] is False


def test_terminal_build_binds_lock_roots_runtime_and_supports_compare(tmp_path):
    module = _module()
    chain = _matched_fixture_module()._chain(tmp_path)
    artifact_root = tmp_path / "confirmation-runs"
    manifest_path = tmp_path / "confirmation-bound.json"
    generation_path = tmp_path / "confirmation-generation.json"
    generation = module.build_confirmation_from_terminal(
        selected_terminal_path=chain["terminal"],
        lockfile_path=chain["lockfile"],
        model_roots_path=chain["model_roots"],
        artifact_root_path=artifact_root,
        output_manifest_path=manifest_path,
        output_receipt_path=generation_path,
    )
    manifest = ExperimentManifest.load(manifest_path)
    bindings = generation["generation_identity"]["execution_bindings"]
    assert manifest.lockfile_sha256 == bindings["lockfile_sha256"]
    assert manifest.engine_params["model_roots_sha256"] == bindings[
        "model_roots_sha256"
    ]
    assert manifest.engine_params["runtime_implementation_fingerprint"] == bindings[
        "runtime_implementation_fingerprint"
    ]
    assert bindings["terminal_receipt_sha256"] == sha256_file(chain["terminal"])
    assert generation["generation_identity"]["artifact_root"] == str(
        artifact_root.resolve()
    )

    input_runs = _confirmation_artifacts(
        artifact_root,
        manifest_path,
        generation,
        lockfile=chain["lockfile"],
        model_roots=chain["model_roots"],
    )
    analysis_root = tmp_path / "confirmation-analysis"
    _analysis(analysis_root, manifest_path, input_runs=input_runs)
    comparison = module.compare_confirmation_from_terminal(
        selected_terminal_path=chain["terminal"],
        lockfile_path=chain["lockfile"],
        model_roots_path=chain["model_roots"],
        artifact_root_path=artifact_root,
        generation_receipt_path=generation_path,
        confirmation_manifest_path=manifest_path,
        analysis_root=analysis_root,
        output_path=tmp_path / "confirmation-comparison.json",
        bootstrap_replicates=40,
    )
    assert comparison["formal_acceptance_claim_pass"] is True
    assert comparison["raw_provenance_pass"] is True
    assert comparison["analysis_identity"][
        "execution_bindings_sha256"
    ] == generation["generation_identity"]["execution_bindings_sha256"]
    raw = comparison["analysis_identity"]["raw_run_provenance"]
    assert raw["artifact_root"] == str(artifact_root.resolve())
    assert len(raw["input_runs"]) == len(ExperimentManifest.load(manifest_path).units)
    evidence = {row["path"] for row in comparison["evidence"]}
    assert str((artifact_root / "dataset-preflight.json").resolve()) in evidence
    assert any(path.endswith("rounds.parquet") for path in evidence)
    assert any(path.endswith("adaptation-telemetry-rank0.jsonl") for path in evidence)


def _bound_confirmation_case(tmp_path: Path):
    module = _module()
    chain = _matched_fixture_module()._chain(tmp_path)
    artifact_root = tmp_path / "confirmation-runs"
    manifest_path = tmp_path / "confirmation.json"
    generation_path = tmp_path / "generation.json"
    generation = module.build_confirmation_from_terminal(
        selected_terminal_path=chain["terminal"],
        lockfile_path=chain["lockfile"],
        model_roots_path=chain["model_roots"],
        artifact_root_path=artifact_root,
        output_manifest_path=manifest_path,
        output_receipt_path=generation_path,
    )
    input_runs = _confirmation_artifacts(
        artifact_root,
        manifest_path,
        generation,
        lockfile=chain["lockfile"],
        model_roots=chain["model_roots"],
    )
    analysis_root = tmp_path / "analysis-confirmation"
    _analysis(analysis_root, manifest_path, input_runs=input_runs)
    return {
        "module": module,
        "chain": chain,
        "artifact_root": artifact_root,
        "manifest": manifest_path,
        "generation": generation_path,
        "analysis": analysis_root,
        "input_runs": input_runs,
    }


def _compare_bound_case(case: dict, output: Path):
    return case["module"].compare_confirmation_from_terminal(
        selected_terminal_path=case["chain"]["terminal"],
        lockfile_path=case["chain"]["lockfile"],
        model_roots_path=case["chain"]["model_roots"],
        artifact_root_path=case["artifact_root"],
        generation_receipt_path=case["generation"],
        confirmation_manifest_path=case["manifest"],
        analysis_root=case["analysis"],
        output_path=output,
        bootstrap_replicates=40,
    )


def test_terminal_compare_rejects_raw_telemetry_hash_mutation(tmp_path):
    case = _bound_confirmation_case(tmp_path)
    run_id = case["input_runs"][0]["run_id"]
    telemetry = (
        case["artifact_root"]
        / run_id
        / "runtime"
        / "adaptation-telemetry-rank0.jsonl"
    )
    telemetry.write_text(
        telemetry.read_text(encoding="utf-8") + '{"event":"mutated"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        case["module"].ConfirmationError,
        match="raw-run provenance failed.*ledger evidence drift",
    ):
        _compare_bound_case(case, tmp_path / "comparison.json")


def test_terminal_compare_rejects_self_consistent_allocator_mutation(tmp_path):
    case = _bound_confirmation_case(tmp_path)
    row = case["input_runs"][0]
    run_root = case["artifact_root"] / row["run_id"]
    run_manifest = run_root / "manifest.json"
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload["engine_params"]["pytorch_cuda_alloc_conf"] = "drift"
    run_manifest.write_text(canonical_json(payload), encoding="utf-8")
    manifest_sidecar = _sidecar(run_manifest, run_root / "manifest.sha256")
    hashes_path = run_root / "hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    for path in (run_manifest, manifest_sidecar):
        hashes[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    hashes_path.write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )

    analysis_manifest = case["analysis"] / "analysis-manifest.json"
    analysis = json.loads(analysis_manifest.read_text(encoding="utf-8"))
    analysis_row = next(
        item for item in analysis["input_runs"] if item["run_id"] == row["run_id"]
    )
    analysis_row["manifest_sha256"] = sha256_file(run_manifest)
    analysis_row["hashes_sha256"] = sha256_file(hashes_path)
    analysis_manifest.write_text(canonical_json(analysis), encoding="utf-8")
    analysis_sidecar = _sidecar(
        analysis_manifest, case["analysis"] / "analysis-manifest.sha256"
    )
    analysis_ledger_path = case["analysis"] / "analysis-hashes.json"
    analysis_ledger = json.loads(
        analysis_ledger_path.read_text(encoding="utf-8")
    )
    for path in (analysis_manifest, analysis_sidecar):
        analysis_ledger[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    analysis_ledger_path.write_text(
        json.dumps(analysis_ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        case["module"].ConfirmationError,
        match="raw-run provenance failed.*pytorch_cuda_alloc_conf",
    ):
        _compare_bound_case(case, tmp_path / "comparison.json")
