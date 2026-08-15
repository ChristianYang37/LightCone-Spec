from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_schema import config_value

from lightcone_spec import PINNED_SGLANG_PATCH_COUNT, PINNED_SGLANG_TREE
from lightcone_spec.cli.main import (
    _advance_tuning,
    _concat_evidence_tables,
    _load_patched_gpu_doctor,
    _parser,
    _static_load_rows,
    _write_json,
)
from lightcone_spec.config import run_config_sha256
from lightcone_spec.config.schema import RunConfig
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.data import (
    DFLASH_MODEL_CONTEXT_LIMIT,
    DFLASH_SAFE_CONTEXT_LIMIT,
    DFLASH_SPECULATIVE_HEADROOM,
    LongContinuationAdapter,
    load_natural_prompts,
    sample_set_sha256,
)
from lightcone_spec.experiments.evidence import (
    GpuEvidenceAttestation,
    GreedyTargetReference,
    TargetOutput,
)
from lightcone_spec.experiments.onlinespec import OnlineSpecManifest
from lightcone_spec.experiments.protocol import (
    DFLASH_LOSS_POSITION_DECAY,
    assert_historical_matched_recipe_diagnostic_configs,
    confirmation_blocks,
    select_static_load,
    successive_halving,
    tuning_candidates,
    tuning_stage,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.selection import (
    CandidateMeasurement,
    LossPoint,
    SelectionArtifact,
    SliceMeasurement,
    reduce_tuning_stage,
    select_heldout_anchor,
    select_shared_config,
)
from lightcone_spec.experiments.statistics import (
    bca_mean_interval,
    evaluate_speed_gate,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration import (
    PreliminarySpeedStudyManifest,
    launch_server_subprocess,
)
from lightcone_spec.orchestration.runtime import (
    build_target_only_run_config,
    derive_diagnostic_compile_cache_key,
    render_runtime_plan,
    render_static_load_runtime_plan,
    render_target_only_runtime_plan,
    render_tuning_runtime_plan,
)
from lightcone_spec.runtime.compile_cache import (
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    CompileCacheKey,
    CompileCacheLaunchPlan,
)
from lightcone_spec.sglang_bridge.launch import main as launch_sglang
from lightcone_spec.telemetry.records import OUTPUT_HASH_FORMAT


@pytest.mark.parametrize(
    "command",
    (
        "render-preliminary-runtime",
        "render-preliminary-static-load-runtime",
        "render-preliminary-target-only-runtime",
        "render-preliminary-tuning-runtime",
        "render-preliminary-replication-runtime",
        "render-onlinespec-runtime",
        "render-onlinespec-tuning-runtime",
    ),
)
def test_runtime_render_cli_requires_compile_cache_plan(command: str) -> None:
    commands = _parser()._subparsers._group_actions[0].choices
    action = next(
        action
        for action in commands[command]._actions
        if action.dest == "compile_cache_plan"
    )
    assert action.required
    assert "--compile-cache-plan" in commands[command].format_help()


def test_target_only_runtime_cli_requires_one_gpu_uuid() -> None:
    command = (
        _parser()
        ._subparsers._group_actions[0]
        .choices["render-preliminary-target-only-runtime"]
    )
    action = next(action for action in command._actions if action.dest == "gpu_uuid")
    assert action.required
    assert "--gpu-uuid" in command.format_help()


def _compile_cache_plan(
    tmp_path: Path,
    lock: ModelLock,
    *,
    max_running_requests: int,
    target_only: bool = False,
) -> tuple[CompileCacheLaunchPlan, Path]:
    revisions = {model.model_id: model.revision for model in lock.models}
    key = CompileCacheKey(
        patched_sglang_tree=PINNED_SGLANG_TREE,
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        patch_sha256=PINNED_SGLANG_PATCH_SHA256,
        source_sha256=PINNED_SGLANG_COMPILE_SOURCE_SHA256,
        python_version="3.12.11",
        torch_version="2.11.0+cu130",
        triton_version="3.6.0",
        cuda_version="13.0",
        driver_version="580.65.06",
        sm_architecture="sm_120",
        gpu_model="RTX PRO 6000 Blackwell Server Edition",
        dtype="bfloat16",
        target_revision=revisions["Qwen/Qwen3-8B"],
        drafter_revision=(
            None if target_only else revisions["z-lab/Qwen3-8B-DFlash-b16"]
        ),
        tensor_parallel_size=1,
        context_limit=40960,
        max_running_requests=max_running_requests,
        graph_buckets=(1,),
        allocator="cuda_malloc_async",
        build_flags=(),
    )
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=tmp_path / "compile-cache",
        cache_mode="build",
    )
    path = plan.write(tmp_path / "compile-cache-plan.json")
    return plan, path


def _passing_compile_doctor() -> dict[str, object]:
    checks = {"runtime": {"status": "PASS"}}
    return {
        "schema_version": 2,
        "status": "PASS",
        "readiness": {
            "status": "PASS",
            "pass_count": len(checks),
            "fail_count": 0,
            "unknown_count": 0,
        },
        "checks": checks,
        "python": {"version": "3.12.11"},
        "packages": {"triton": "3.6.0"},
        "commands": {"nvcc": "Cuda compilation tools, release 13.0, V13.0.0"},
        "gpu": {
            "torch": {
                "importable": True,
                "version": "2.11.0+cu130",
                "cuda_build": "13.0",
                "cuda_available": True,
                "device_count": 1,
            },
            "parsed_inventory": {
                "parse_error": None,
                "devices": [
                    {
                        "uuid": "GPU-test",
                        "name": "RTX PRO 6000 Blackwell Server Edition",
                        "driver_version": "580.65.06",
                        "compute_capability": "12.0",
                    }
                ],
            },
        },
    }


def test_diagnostic_compile_key_is_derived_from_doctor_lock_and_run_config() -> None:
    lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "a" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    raw_config = config_value("static")
    raw_config["runtime"]["device_identity"] = "GPU-test"
    config = RunConfig.model_validate(raw_config)

    key = derive_diagnostic_compile_cache_key(
        doctor_report=_passing_compile_doctor(),
        model_lock=lock,
        config=config,
    )

    assert key.python_version == "3.12.11"
    assert key.torch_version == "2.11.0+cu130"
    assert key.triton_version == "3.6.0"
    assert key.cuda_version == "13.0"
    assert key.gpu_model == "RTX PRO 6000 Blackwell Server Edition"
    assert key.sm_architecture == "sm_120"
    assert key.target_revision == "a" * 40
    assert key.drafter_revision == "b" * 40
    assert key.graph_buckets == (1,)
    assert key.allocator == "cuda_malloc_async"
    assert key.build_flags == ()

    incomplete = _passing_compile_doctor()
    incomplete["status"] = "UNKNOWN"
    with pytest.raises(ValueError, match="complete PASS doctor"):
        derive_diagnostic_compile_cache_key(
            doctor_report=incomplete,
            model_lock=lock,
            config=config,
        )
    foreign_lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "c" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    with pytest.raises(ValueError, match="exact model lock"):
        derive_diagnostic_compile_cache_key(
            doctor_report=_passing_compile_doctor(),
            model_lock=foreign_lock,
            config=config,
        )


def test_controlled_windows_are_sized_unique_and_content_disjoint() -> None:
    adapter = LongContinuationAdapter()
    adapter.assert_disjoint()
    assert [len(adapter.window(name)) for name in ("load", "tune", "confirm")] == [
        8,
        16,
        32,
    ]
    all_samples = [
        sample
        for name in ("load", "tune", "confirm")
        for sample in adapter.window(name)
    ]
    assert len({sample.sample_id for sample in all_samples}) == 56
    assert len({sample.prompt for sample in all_samples}) == 56


def test_dflash_safe_limit_reserves_two_verification_blocks() -> None:
    assert DFLASH_SPECULATIVE_HEADROOM == 32
    assert (
        DFLASH_SAFE_CONTEXT_LIMIT + DFLASH_SPECULATIVE_HEADROOM
        == DFLASH_MODEL_CONTEXT_LIMIT
    )


def test_controlled_path_does_not_import_optional_dataset_package() -> None:
    import sys

    before = sys.modules.get("datasets")
    LongContinuationAdapter().window("confirm")
    assert sys.modules.get("datasets") is before


def test_natural_prompt_loader_streams_only_the_locked_window(monkeypatch) -> None:
    import sys
    import types

    calls: list[tuple[str, dict]] = []
    module = types.ModuleType("datasets")

    def load_dataset(repository: str, **kwargs):
        calls.append((repository, kwargs))
        return ({"problem": f"problem-{index}"} for index in range(64))

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    revision = "a" * 40
    samples = load_natural_prompts("math500", revision=revision, split="test", limit=32)
    assert len(samples) == 32
    assert len({sample.sample_id for sample in samples}) == 32
    assert [sample.prompt for sample in samples] == [
        f"problem-{index}" for index in range(32)
    ]
    assert calls == [
        (
            "HuggingFaceH4/MATH-500",
            {"split": "test", "revision": revision, "streaming": True},
        )
    ]


def test_sampling_profiles_separate_controlled_and_natural_eos() -> None:
    controlled = SamplingProfile()
    natural = SamplingProfile(purpose="natural", ignore_eos=False)
    controlled.validate()
    natural.validate()
    assert controlled.temperature == 0.0
    assert controlled.sha256 != natural.sha256
    with pytest.raises(ValueError, match="natural replication"):
        SamplingProfile(purpose="natural", ignore_eos=True).validate()
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature/top_p"):
            SamplingProfile(temperature=value).validate()


def test_tracked_controlled_profile_matches_registered_manifests() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = SamplingProfile.load(
        root / "manifests/speed-study/sampling_profile_v2.json"
    )
    assert profile == SamplingProfile()
    assert (
        PreliminarySpeedStudyManifest.load(
            root / "manifests/speed-study/static_tts_l0_v2.json"
        ).sampling_profile_sha256
        == profile.sha256
    )
    assert (
        OnlineSpecManifest.load(
            root / "manifests/speed-study/onlinespec_baseline_v2.json"
        ).sampling_profile_sha256
        == profile.sha256
    )


def test_speed_manifest_is_immutable_and_hash_bound(tmp_path) -> None:
    manifest = PreliminarySpeedStudyManifest.default()
    path = tmp_path / "speed-study.json"
    manifest.write(path)
    assert PreliminarySpeedStudyManifest.load(path) == manifest
    assert manifest.confirmation_schedule_seed == 20260809
    path.write_text("{}", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        PreliminarySpeedStudyManifest.load(path)


def test_tuning_grid_is_complete_and_unique() -> None:
    candidates = tuning_candidates()
    assert len(candidates) == 64
    assert len({candidate.candidate_id for candidate in candidates}) == 64
    assert {candidate.optimizer for candidate in candidates} == {"adamw", "sgdm"}
    assert {candidate.parameter_scope for candidate in candidates} == {
        "last1",
        "last3",
        "last5",
        "all",
    }
    assert {candidate.stride for candidate in candidates} == {10}
    assert {
        candidate.rank
        for candidate in candidates
        if candidate.weight_update_mode == "lora"
    } == {
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    }
    assert all(
        candidate.rank is None
        for candidate in candidates
        if candidate.weight_update_mode == "full"
    )
    assert all(
        candidate.lora_alpha == candidate.rank
        for candidate in candidates
        if candidate.weight_update_mode == "lora"
    )
    assert all(
        candidate.momentum is not None
        for candidate in candidates
        if candidate.optimizer in {"sgdm", "nag", "muon"}
    )
    assert all(
        candidate.muon_ns_steps is not None
        for candidate in candidates
        if candidate.optimizer == "muon"
    )
    assert [tuning_stage(index) for index in range(4)] == [
        (2, 4096),
        (4, 8192),
        (8, 16384),
        (16, DFLASH_SAFE_CONTEXT_LIMIT),
    ]


def test_successive_halving_is_deterministic_and_tuning_only() -> None:
    identifiers = tuple("abcd")
    scores = {"a": 1.0, "b": 3.0, "c": 2.0, "d": 0.0}
    assert successive_halving(identifiers, scores, keep_fraction=0.5) == ("b", "c")
    with pytest.raises(ValueError, match="every candidate"):
        successive_halving(identifiers, {"a": 1.0})
    with pytest.raises(ValueError, match="finite"):
        successive_halving(("a",), {"a": float("nan")})


def test_confirmation_schedule_has_independent_randomized_blocks() -> None:
    blocks = confirmation_blocks(123)
    assert len(blocks) == 8
    assert blocks == confirmation_blocks(123)
    assert all(
        set(block.method_order) == {"static", "tts", "l0"}
        and block.reset_cohort_before_each
        for block in blocks
    )
    assert len({block.method_order for block in blocks}) > 1


def load_screen_rows() -> list[dict]:
    return [
        {
            "concurrency": concurrency,
            "decode_goodput_tps": float(concurrency * 10),
            "itl_p99_ms": 10.0 + concurrency,
            "oom_events": 0,
            "retractions": 0,
            "kv_token_capacity": 48 * DFLASH_SAFE_CONTEXT_LIMIT,
        }
        for concurrency in (1, 2, 4, 8, 16, 32, 48)
    ]


def test_static_load_selection_respects_latency_and_safety() -> None:
    rows = load_screen_rows()
    # c16 and above exceed twice the c1 p99 (22 ms).
    assert (
        select_static_load(rows, required_context_limit=DFLASH_SAFE_CONTEXT_LIMIT) == 8
    )
    rows[-1]["oom_events"] = 1
    assert (
        select_static_load(rows, required_context_limit=DFLASH_SAFE_CONTEXT_LIMIT) == 8
    )
    rows[3]["kv_token_capacity"] = 7 * DFLASH_SAFE_CONTEXT_LIMIT
    assert (
        select_static_load(rows, required_context_limit=DFLASH_SAFE_CONTEXT_LIMIT) == 4
    )
    with pytest.raises(ValueError, match="complete grid"):
        select_static_load(rows[:-1], required_context_limit=DFLASH_SAFE_CONTEXT_LIMIT)
    duplicated = load_screen_rows() + [load_screen_rows()[0]]
    with pytest.raises(ValueError, match="complete grid"):
        select_static_load(duplicated, required_context_limit=DFLASH_SAFE_CONTEXT_LIMIT)
    invalid = load_screen_rows()
    invalid[0]["decode_goodput_tps"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        select_static_load(invalid, required_context_limit=DFLASH_SAFE_CONTEXT_LIMIT)

    saturated = load_screen_rows()
    for row in saturated:
        row["itl_p99_ms"] = 1.0
    assert (
        select_static_load(saturated, required_context_limit=DFLASH_SAFE_CONTEXT_LIMIT)
        == 32
    )


def test_static_load_terminal_is_bound_to_manifest_sampling_and_window() -> None:
    manifest = PreliminarySpeedStudyManifest.default()
    artifact = {
        "schema_version": 2,
        "phase": "static_load_screen",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": "a" * 64,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "window_sha256": manifest.controlled_window_hashes["load"],
        "rows": load_screen_rows(),
    }
    assert _static_load_rows(artifact, manifest=manifest) == load_screen_rows()
    for field in ("manifest_sha256", "sampling_profile_sha256", "window_sha256"):
        tampered = {**artifact, field: "f" * 64}
        with pytest.raises(ValueError, match="identity mismatch"):
            _static_load_rows(tampered, manifest=manifest)


def test_tuning_stage_rejects_a_load_change_and_binds_its_predecessor(
    tmp_path,
) -> None:
    manifest = PreliminarySpeedStudyManifest.default()
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    candidate = tuning_candidates()[0]
    expected_window = sample_set_sha256(LongContinuationAdapter().window("tune")[:4])
    paths = []
    for method, candidate_id, goodput in (
        ("static", None, 100.0),
        ("tts", candidate.candidate_id, 102.0),
        ("l0", candidate.candidate_id, 101.0),
    ):
        row = replace(
            slice_measurement(
                method,
                candidate_id=candidate_id,
                goodput=goodput,
            ),
            stage=1,
            manifest_sha256=manifest.sha256,
            sampling_profile_sha256=manifest.sampling_profile_sha256,
            window_sha256=expected_window,
            prompt_count=4,
            context_limit=8192,
            concurrency=4,
        )
        path = tmp_path / f"{method}.json"
        row.write(path)
        paths.append(str(path))
    prior = {
        "schema_version": 2,
        "phase": "shared_config_tuning",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": "f" * 64,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "execution_policy_sha256": manifest.execution_policy_sha256,
        "window_sha256": manifest.controlled_window_hashes["tune"],
        "tuning_grid_sha256": manifest.tuning_grid_sha256,
        "concurrency": 8,
        "stage": 0,
        "next_stage": 1,
        "prior_stage_sha256": None,
        "active_candidates": [candidate.candidate_id],
        "survivors": [candidate.candidate_id],
        "measurement_sha256": [],
        "candidate_measurements": [],
    }
    prior_path = tmp_path / "prior.json"
    _write_json(prior_path, prior)
    arguments = argparse.Namespace(
        manifest=str(manifest_path),
        stage=1,
        measurements=paths,
        active_set=str(prior_path),
        output=str(tmp_path / "stage-1.json"),
    )
    with pytest.raises(ValueError, match="changes the selected runtime load"):
        _advance_tuning(arguments)

    matching_prior = {**prior, "concurrency": 4}
    matching_path = tmp_path / "matching-prior.json"
    _write_json(matching_path, matching_prior)
    arguments.active_set = str(matching_path)
    assert _advance_tuning(arguments) == 0
    terminal = json.loads(Path(arguments.output).read_text(encoding="utf-8"))
    assert terminal["prior_stage_sha256"] is not None
    assert terminal["concurrency"] == 4
    assert terminal["sampling_profile_sha256"] == manifest.sampling_profile_sha256


def measurements(first: str, second: str) -> list[CandidateMeasurement]:
    return [
        CandidateMeasurement(
            first,
            "tts",
            "tune",
            1.04,
            100,
            5.0,
            1.0,
            updates_launched=1,
            updates_published=1,
        ),
        CandidateMeasurement(
            first,
            "l0",
            "tune",
            1.03,
            100,
            5.0,
            1.0,
            updates_launched=1,
            updates_published=1,
        ),
        CandidateMeasurement(
            second,
            "tts",
            "tune",
            1.06,
            200,
            7.0,
            2.0,
            updates_launched=1,
            updates_published=1,
        ),
        CandidateMeasurement(
            second,
            "l0",
            "tune",
            1.01,
            200,
            7.0,
            2.0,
            updates_launched=1,
            updates_published=1,
        ),
    ]


def slice_measurement(
    method: str,
    *,
    candidate_id: str | None,
    goodput: float,
) -> SliceMeasurement:
    adapted = method != "static"
    return SliceMeasurement(
        schema_version=2,
        phase="shared_config_tuning",
        stage=0,
        method=method,
        candidate_id=candidate_id,
        manifest_sha256="a" * 64,
        config_sha256=("b" if method == "static" else "c") * 64,
        model_lock_sha256="f" * 64,
        sampling_profile_sha256="d" * 64,
        window_sha256="e" * 64,
        output_set_sha256="1" * 64,
        prompt_count=2,
        context_limit=4096,
        concurrency=8,
        decode_goodput_tps=goodput,
        itl_p99_ms=5.0,
        peak_hbm_bytes=100,
        kv_bytes=50,
        kv_token_capacity=409600,
        optimizer_bytes=20 if adapted else 0,
        trainable_parameters=10 if adapted else 0,
        exposed_update_ms=1.0 if adapted else 0.0,
        updates_launched=2 if adapted else 0,
        updates_published=1 if adapted else 0,
        exactness_violations=0,
        version_mismatches=0,
        fallbacks=0,
        nonfinite_updates=0,
        oom_events=0,
        retractions=0,
        loss_points=(LossPoint(100, 110, 105.0, 0.5),) if adapted else (),
    )


def test_tuning_stage_reducer_pairs_methods_and_static() -> None:
    first, second = tuning_candidates()[:2]
    rows = [slice_measurement("static", candidate_id=None, goodput=100.0)]
    for candidate, tts, l0 in (
        (first, 110.0, 108.0),
        (second, 105.0, 104.0),
    ):
        rows.extend(
            (
                slice_measurement(
                    "tts", candidate_id=candidate.candidate_id, goodput=tts
                ),
                slice_measurement(
                    "l0",
                    candidate_id=candidate.candidate_id,
                    goodput=l0,
                ),
            )
        )
    survivors, reduced = reduce_tuning_stage(
        rows,
        candidates={
            first.candidate_id: first,
            second.candidate_id: second,
        },
        active_candidate_ids=(first.candidate_id, second.candidate_id),
        stage=0,
    )
    assert survivors == (first.candidate_id,)
    assert len(reduced) == 4
    assert all(row.safe for row in reduced)


def test_tuning_stage_rejects_a_different_greedy_output_trajectory() -> None:
    candidate = tuning_candidates()[0]
    static = slice_measurement("static", candidate_id=None, goodput=100.0)
    tts = slice_measurement("tts", candidate_id=candidate.candidate_id, goodput=101.0)
    l0 = slice_measurement("l0", candidate_id=candidate.candidate_id, goodput=101.0)
    with pytest.raises(ValueError, match="paired to the Static"):
        reduce_tuning_stage(
            [static, replace(tts, output_set_sha256="2" * 64), l0],
            candidates={candidate.candidate_id: candidate},
            active_candidate_ids=(candidate.candidate_id,),
            stage=0,
        )


def test_tuning_stage_eliminates_unsafe_candidate_before_goodput_ranking() -> None:
    safe, unsafe = tuning_candidates()[:2]
    rows = [slice_measurement("static", candidate_id=None, goodput=100.0)]
    for method in ("tts", "l0"):
        rows.append(
            slice_measurement(
                method,
                candidate_id=safe.candidate_id,
                goodput=101.0,
            )
        )
        invalid = slice_measurement(
            method,
            candidate_id=unsafe.candidate_id,
            goodput=1000.0,
        )
        rows.append(
            SliceMeasurement(
                **{
                    **invalid.__dict__,
                    "exactness_violations": 1,
                }
            )
        )
    survivors, _ = reduce_tuning_stage(
        rows,
        candidates={safe.candidate_id: safe, unsafe.candidate_id: unsafe},
        active_candidate_ids=(safe.candidate_id, unsafe.candidate_id),
        stage=0,
    )
    assert survivors == (safe.candidate_id,)


def test_shared_selection_uses_maximin_then_resource_tiebreak(tmp_path) -> None:
    first, second = tuning_candidates()[:2]
    artifact = select_shared_config(
        measurements(first.candidate_id, second.candidate_id),
        candidates={first.candidate_id: first, second.candidate_id: second},
        selected_concurrency=8,
        manifest_sha256="a" * 64,
        sampling_profile_sha256="b" * 64,
        tuning_grid_sha256="c" * 64,
        load_screen_sha256="e" * 64,
        tuning_window_sha256=LongContinuationAdapter().window_sha256("tune"),
        model_lock_sha256="d" * 64,
    )
    assert artifact.candidate_id == first.candidate_id
    assert artifact.minimum_goodput_ratio == 1.03
    path = tmp_path / "selection.json"
    artifact.write(path)
    assert SelectionArtifact.load(path) == artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["evidence_classification"] == (
        "matched_recipe_publication_policy_diagnostic_not_tts_reproduction"
    )
    assert payload["formal_execution_authorized"] is False

    legacy_payload = dict(payload)
    legacy_payload["schema_version"] = 2
    legacy_payload.pop("evidence_classification")
    legacy_payload.pop("formal_execution_authorized")
    legacy_body = json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"))
    legacy_path = tmp_path / "legacy-selection.json"
    legacy_path.write_text(legacy_body, encoding="utf-8")
    Path(f"{legacy_path}.sha256").write_text(
        hashlib.sha256(legacy_body.encode()).hexdigest() + "\n",
        encoding="utf-8",
    )
    legacy = SelectionArtifact.load(legacy_path)
    assert legacy.schema_version == 2
    assert legacy.evidence_classification == payload["evidence_classification"]
    assert legacy.formal_execution_authorized is False
    with pytest.raises(ValueError, match="read-only"):
        legacy.write(tmp_path / "legacy-rewrite.json")
    with pytest.raises(ValueError, match="invalid concurrency"):
        replace(artifact, selected_concurrency=48).validate()


def test_heldout_anchor_uses_terminal_tuning_only_without_grid_claim() -> None:
    candidate = tuning_candidates()[0]
    prompt_count, context_limit = tuning_stage(3)
    window = sample_set_sha256(LongContinuationAdapter().window("tune")[:prompt_count])
    rows = []
    for method, candidate_id, goodput in (
        ("static", None, 100.0),
        ("tts", candidate.candidate_id, 110.0),
        ("l0", candidate.candidate_id, 112.0),
    ):
        rows.append(
            replace(
                slice_measurement(
                    method,
                    candidate_id=candidate_id,
                    goodput=goodput,
                ),
                stage=3,
                prompt_count=prompt_count,
                context_limit=context_limit,
                window_sha256=window,
            )
        )
    artifact = select_heldout_anchor(
        rows,
        candidate=candidate,
        selected_concurrency=8,
        manifest_sha256="a" * 64,
        sampling_profile_sha256="d" * 64,
        tuning_grid_sha256="c" * 64,
        load_screen_sha256="e" * 64,
        tuning_window_sha256=LongContinuationAdapter().window_sha256("tune"),
        model_lock_sha256="f" * 64,
        tuning_evidence_sha256="9" * 64,
    )
    assert artifact.candidate_id == candidate.candidate_id
    assert artifact.selection_protocol == "heldout_anchor"
    assert artifact.minimum_goodput_ratio == pytest.approx(1.1)

    with pytest.raises(ValueError, match="coverage is incomplete"):
        select_heldout_anchor(
            rows[:-1],
            candidate=candidate,
            selected_concurrency=8,
            manifest_sha256="a" * 64,
            sampling_profile_sha256="d" * 64,
            tuning_grid_sha256="c" * 64,
            load_screen_sha256="e" * 64,
            tuning_window_sha256=window,
            model_lock_sha256="f" * 64,
            tuning_evidence_sha256="9" * 64,
        )

    with pytest.raises(ValueError, match="Static load screen"):
        select_heldout_anchor(
            rows,
            candidate=candidate,
            selected_concurrency=4,
            manifest_sha256="a" * 64,
            sampling_profile_sha256="d" * 64,
            tuning_grid_sha256="c" * 64,
            load_screen_sha256="e" * 64,
            tuning_window_sha256=window,
            model_lock_sha256="f" * 64,
            tuning_evidence_sha256="9" * 64,
        )


def test_selection_rejects_confirmation_or_unsafe_evidence() -> None:
    candidate = tuning_candidates()[0]
    rows = [
        CandidateMeasurement(
            candidate.candidate_id,
            "tts",
            "confirm",
            1.1,
            1,
            1.0,
            1.0,
        )
    ]
    with pytest.raises(ValueError, match="confirmation"):
        select_shared_config(
            rows,
            candidates={candidate.candidate_id: candidate},
            selected_concurrency=1,
            manifest_sha256="c" * 64,
            sampling_profile_sha256="d" * 64,
            tuning_grid_sha256="e" * 64,
            load_screen_sha256="f" * 64,
            tuning_window_sha256="a" * 64,
            model_lock_sha256="b" * 64,
        )
    invalid = measurements(candidate.candidate_id, tuning_candidates()[1].candidate_id)
    invalid[0] = CandidateMeasurement(
        candidate.candidate_id,
        "tts",
        "tune",
        float("nan"),
        1,
        1.0,
        1.0,
    )
    with pytest.raises(ValueError, match="finite"):
        select_shared_config(
            invalid,
            candidates={
                candidate.candidate_id: candidate,
                tuning_candidates()[1].candidate_id: tuning_candidates()[1],
            },
            selected_concurrency=1,
            manifest_sha256="c" * 64,
            sampling_profile_sha256="d" * 64,
            tuning_grid_sha256="e" * 64,
            load_screen_sha256="f" * 64,
            tuning_window_sha256="a" * 64,
            model_lock_sha256="b" * 64,
        )


def test_historical_matched_recipe_diagnostic_binds_old_selected_candidate() -> None:
    selected = tuning_candidates()[0]
    static = RunConfig.model_validate(config_value("static"))
    adapted = {}
    for method in ("tts", "l0"):
        value = config_value(method)
        value["adaptation"].update(
            weight_update_mode=selected.weight_update_mode,
            parameter_scope=selected.parameter_scope,
            rank=selected.rank,
            lora_alpha=selected.lora_alpha,
            stride=selected.stride,
        )
        value["adaptation"]["optimizer"].update(
            name=selected.optimizer,
            learning_rate=selected.learning_rate,
            weight_decay=selected.weight_decay,
            grad_clip=selected.grad_clip,
            schedule=selected.schedule,
        )
        adapted[method] = RunConfig.model_validate(value)
    configs = {"static": static, **adapted}
    assert_historical_matched_recipe_diagnostic_configs(
        configs, selected_candidate=selected, selected_concurrency=8
    )
    changed = config_value("l0")
    changed["adaptation"]["stride"] = selected.stride + 1
    configs["l0"] = RunConfig.model_validate(changed)
    with pytest.raises(ValueError):
        assert_historical_matched_recipe_diagnostic_configs(
            configs, selected_candidate=selected, selected_concurrency=8
        )


def test_bca_interval_is_finite_and_ordered() -> None:
    estimate, lower, upper = bca_mean_interval(
        {f"p{i}": np.asarray([0.03 + i * 0.001]) for i in range(8)},
        repetitions=1000,
        seed=7,
    )
    assert lower <= estimate <= upper
    assert np.isfinite([estimate, lower, upper]).all()


def test_evidence_concat_promotes_only_all_null_inferred_columns(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static.parquet"
    adapted = tmp_path / "adapted.parquet"
    pq.write_table(
        pa.table(
            {
                "method": ["static"],
                "adaptation_memory_ledger": pa.array([None], type=pa.null()),
                "training_cuda_ms": pa.array([None], type=pa.null()),
            }
        ),
        static,
    )
    pq.write_table(
        pa.table(
            {
                "method": ["tts"],
                "adaptation_memory_ledger": ["{}"],
                "training_cuda_ms": [1.25],
            }
        ),
        adapted,
    )

    table = _concat_evidence_tables((static, adapted))

    assert table.schema.field("adaptation_memory_ledger").type == pa.string()
    assert table.schema.field("training_cuda_ms").type == pa.float64()
    assert table.to_pylist() == [
        {
            "method": "static",
            "adaptation_memory_ledger": None,
            "training_cuda_ms": None,
        },
        {
            "method": "tts",
            "adaptation_memory_ledger": "{}",
            "training_cuda_ms": 1.25,
        },
    ]

    incompatible = tmp_path / "incompatible.parquet"
    pq.write_table(
        pa.table(
            {
                "method": ["tts"],
                "adaptation_memory_ledger": [1],
                "training_cuda_ms": [1.25],
            }
        ),
        incompatible,
    )
    with pytest.raises(ValueError, match="incompatible types"):
        _concat_evidence_tables((adapted, incompatible))


def performance_rows(
    *, speed_tts: float = 1.05, speed_l0: float = 1.06, unsafe: bool = False
) -> list[dict]:
    rows: list[dict] = []
    generated_limit = DFLASH_SAFE_CONTEXT_LIMIT - 49
    methods = {"static": 1.0, "tts": speed_tts, "l0": speed_l0}
    for block in range(8):
        for method, ratio in methods.items():
            safety = {
                "updates_launched": 0 if method == "static" else 4,
                "updates_published": 0 if method == "static" else 3,
                "exactness_violations": int(unsafe and method == "tts" and block == 0),
                "version_mismatches": 0,
                "fallbacks": 0,
                "nonfinite_updates": 0,
                "oom_events": 0,
                "retractions": 0,
            }
            for bucket in (16384, 24576, 32768):
                rows.append(
                    {
                        "prompt_id": "batch-confirmation",
                        "method": method,
                        "repetition_block": block,
                        "region": "generated_bucket",
                        "concurrency": 8,
                        "generated_bucket_start": bucket,
                        "at_risk_requests": 32,
                        "output_tokens": 8192 * 32,
                        "decode_goodput_tps": 100.0 * ratio,
                        **{key: None for key in safety},
                    }
                )
            rows.append(
                {
                    "prompt_id": "batch-confirmation",
                    "method": method,
                    "repetition_block": block,
                    "region": "long_region",
                    "concurrency": 8,
                    "generated_bucket_start": 16384,
                    "generated_bucket_end": generated_limit,
                    "at_risk_requests": 32,
                    "output_tokens": (generated_limit - 16384) * 32,
                    "decode_goodput_tps": 100.0 * ratio,
                    **{key: None for key in safety},
                }
            )
            rows.append(
                {
                    "prompt_id": "batch-confirmation",
                    "method": method,
                    "repetition_block": block,
                    "region": "full_trajectory",
                    "concurrency": 8,
                    "generated_bucket_start": 0,
                    "at_risk_requests": 32,
                    "output_tokens": DFLASH_SAFE_CONTEXT_LIMIT * 32,
                    "decode_goodput_tps": 100.0 * ratio,
                    **safety,
                }
            )
    return rows


def test_speed_gate_never_claims_unattested_gpu_success() -> None:
    gate = evaluate_speed_gate(performance_rows(), seed=1)
    assert gate.status == "UNMEASURED"
    assert not gate.passed
    assert gate.tts.acceleration_pass
    assert gate.l0.acceleration_pass
    assert gate.l0_vs_tts.no_worse_pass


def test_speed_gate_distinguishes_generated_position_from_total_context() -> None:
    rows = performance_rows()
    assert evaluate_speed_gate(rows, seed=1).tts.acceleration_pass
    invalid = [
        {
            **row,
            "generated_bucket_end": DFLASH_SAFE_CONTEXT_LIMIT,
        }
        if row["region"] == "long_region"
        else row
        for row in rows
    ]
    with pytest.raises(ValueError, match="generated-token positions"):
        evaluate_speed_gate(invalid, seed=1)


def test_measured_speed_gate_rejects_caller_authored_attestation_state() -> None:
    with pytest.raises(ValueError, match="trusted hardware attestation"):
        evaluate_speed_gate(
            performance_rows(),
            seed=1,
            gpu_evidence="MEASURED",
            evidence_sha256="e" * 64,
        )
    unsafe = evaluate_speed_gate(performance_rows(unsafe=True), seed=1)
    assert unsafe.status == "UNMEASURED"
    assert not unsafe.passed


def test_speed_gate_requires_l0_to_be_noninferior_to_tts() -> None:
    inferior = evaluate_speed_gate(
        performance_rows(speed_tts=1.08, speed_l0=1.05),
        seed=1,
    )
    assert inferior.tts.passed
    assert inferior.l0.passed
    assert not inferior.l0_vs_tts.passed
    assert inferior.status == "UNMEASURED"
    assert not inferior.passed

    tied = evaluate_speed_gate(
        performance_rows(speed_tts=1.05, speed_l0=1.05),
        seed=1,
    )
    assert tied.l0_vs_tts.mean_speedup == pytest.approx(0.0)
    assert tied.l0_vs_tts.ci_lower == pytest.approx(0.0)
    assert tied.l0_vs_tts.passed
    assert tied.status == "UNMEASURED"
    assert not tied.passed


def test_speed_gate_rejects_incomplete_coverage() -> None:
    with pytest.raises(ValueError, match="coverage|paired cells"):
        evaluate_speed_gate(performance_rows()[:-1])
    short = performance_rows()
    next(row for row in short if row["region"] == "long_region")[
        "generated_bucket_end"
    ] = 32768
    with pytest.raises(ValueError, match="bounds|same at-risk sample"):
        evaluate_speed_gate(short)


def test_legacy_gpu_attestation_api_is_categorically_disabled(tmp_path) -> None:
    attestation = GpuEvidenceAttestation(
        schema_version=2,
        status="MEASURED",
        manifest_sha256="a" * 64,
        selection_sha256="b" * 64,
        model_lock_sha256="f" * 64,
        performance_sha256="d" * 64,
        target_reference_sha256="e" * 64,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        target_revision="c" * 40,
        drafter_revision="d" * 40,
        hardware_sha256="e" * 64,
        methods=("static", "tts", "l0"),
        repetitions=8,
        context_start=16384,
        context_limit=DFLASH_SAFE_CONTEXT_LIMIT,
    )
    path = tmp_path / "attestation.json"
    with pytest.raises(RuntimeError, match="legacy_gpu_attestation_api_disabled"):
        attestation.write(path)
    with pytest.raises(RuntimeError, match="legacy_gpu_attestation_api_disabled"):
        _ = attestation.sha256
    with pytest.raises(RuntimeError, match="legacy_gpu_attestation_api_disabled"):
        attestation.verify_performance(())
    with pytest.raises(RuntimeError, match="legacy_gpu_attestation_api_disabled"):
        GpuEvidenceAttestation.load(tmp_path / "missing-attestation.json")
    assert not path.exists()


def test_target_reference_is_bound_and_matches_one_study(tmp_path) -> None:
    reference = GreedyTargetReference(
        schema_version=2,
        status="PRELIMINARY_DIAGNOSTIC_ONLY",
        model_lock_sha256="a" * 64,
        target_model_id="Qwen/Qwen3-8B",
        target_revision="b" * 40,
        sampling_profile_sha256="c" * 64,
        execution_policy_sha256=ControlledExecutionPolicy().sha256,
        window_sha256="d" * 64,
        runtime_config_sha256="e" * 64,
        hardware_sha256="f" * 64,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        concurrency=8,
        context_limit=DFLASH_SAFE_CONTEXT_LIMIT,
        output_hash_format=OUTPUT_HASH_FORMAT,
        outputs=tuple(
            TargetOutput(
                prompt_id=f"prompt-{index:02d}",
                input_tokens=64,
                output_tokens=DFLASH_SAFE_CONTEXT_LIMIT - 64,
                output_sha256=f"{index:064x}",
            )
            for index in range(32)
        ),
    )
    path = tmp_path / "reference.json"
    reference.write(path)
    loaded = GreedyTargetReference.load(path)
    loaded.verify_study(
        model_lock_sha256="a" * 64,
        target_revision="b" * 40,
        sampling_profile_sha256="c" * 64,
        execution_policy_sha256=ControlledExecutionPolicy().sha256,
        window_sha256="d" * 64,
        concurrency=8,
    )
    with pytest.raises(ValueError, match="different preliminary diagnostic study"):
        loaded.verify_study(
            model_lock_sha256="a" * 64,
            target_revision="b" * 40,
            sampling_profile_sha256="c" * 64,
            execution_policy_sha256=ControlledExecutionPolicy().sha256,
            window_sha256="d" * 64,
            concurrency=4,
        )
    with pytest.raises(ValueError, match="output hash format"):
        replace(reference, output_hash_format="decoded-text-sha256").validate()
    with pytest.raises(ValueError, match="PRELIMINARY_DIAGNOSTIC_ONLY"):
        replace(reference, status="MEASURED").validate()
    canonical_body = path.read_text(encoding="utf-8")
    duplicate_body = canonical_body.replace(
        '"schema_version":2',
        '"schema_version":false,"schema_version":2',
        1,
    )
    assert duplicate_body != canonical_body
    duplicate_path = tmp_path / "reference-duplicate.json"
    duplicate_path.write_text(duplicate_body, encoding="utf-8")
    Path(f"{duplicate_path}.sha256").write_text(
        reference.sha256 + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        GreedyTargetReference.load(duplicate_path)

    sidecar_path = tmp_path / "reference-sidecar.json"
    reference.write(sidecar_path)
    Path(f"{sidecar_path}.sha256").write_text(
        f" {reference.sha256}\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="sidecar"):
        GreedyTargetReference.load(sidecar_path)

    value = json.loads(path.read_text())
    value["outputs"][0]["unexpected"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="output fields"):
        GreedyTargetReference.load(path)


def test_target_reference_requires_exact_patched_gpu_doctor(tmp_path) -> None:
    report = {
        "commands": {"nvidia_smi": "GPU, 98304 MiB, driver"},
        "source_tree": {
            "is_git_checkout": True,
            "tree": PINNED_SGLANG_TREE,
            "dirty": False,
            "pinned_ancestor": True,
            "patch_commits": PINNED_SGLANG_PATCH_COUNT,
        },
    }
    path = tmp_path / "doctor.json"
    path.write_text(json.dumps(report))
    assert _load_patched_gpu_doctor(path, purpose="target reference") == report
    report["source_tree"]["dirty"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact clean patched checkout"):
        _load_patched_gpu_doctor(path, purpose="target reference")


def test_runtime_renderer_produces_three_matched_argv_plans(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "lightcone_spec.orchestration.runtime.verify_patched_checkout",
        lambda path: Path(path).resolve(),
    )
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "a" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    candidate = tuning_candidates()[0]
    selection = SelectionArtifact(
        schema_version=3,
        candidate=candidate,
        selected_concurrency=8,
        minimum_goodput_ratio=1.0,
        peak_hbm_bytes=1,
        itl_p99_ms=1.0,
        exposed_update_ms=1.0,
        manifest_sha256="a" * 64,
        sampling_profile_sha256="b" * 64,
        tuning_grid_sha256="c" * 64,
        load_screen_sha256="d" * 64,
        tuning_window_sha256=LongContinuationAdapter().window_sha256("tune"),
        model_lock_sha256=lock.sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        tuning_evidence_sha256="e" * 64,
    )
    _plan, compile_plan_path = _compile_cache_plan(
        tmp_path,
        lock,
        max_running_requests=8,
    )
    launches = render_runtime_plan(
        output_root=tmp_path / "runtime",
        selection=selection,
        model_lock=lock,
        model_roots={
            "schema_version": 2,
            "lock_sha256": lock.sha256,
            "roots": {
                "Qwen/Qwen3-8B": str(target),
                "z-lab/Qwen3-8B-DFlash-b16": str(drafter),
            },
        },
        sampling_profile=SamplingProfile(),
        sglang_checkout=tmp_path / "patched-sglang",
        compile_cache_plan_path=compile_plan_path,
        adaptation_group_id="study-a",
        adaptation_reserve_mb=1024,
        mem_fraction_static=0.7,
    )
    assert [launch.method for launch in launches] == [
        "static",
        "tts",
        "l0",
    ]
    assert launches[0].adaptation_config is None
    assert len({launch.base_url for launch in launches}) == 1
    assert all(launch.exclusive_device for launch in launches)
    assert all(
        "--speculative-adaptation-config" in launch.argv for launch in launches[1:]
    )
    assert all("--disable-cuda-graph" in launch.argv for launch in launches)
    assert all("--disable-radix-cache" in launch.argv for launch in launches)
    assert all(
        "--speculative-use-rejection-sampling" in launch.argv for launch in launches
    )
    assert all(
        "--speculative-speed-study-metrics" in launch.argv for launch in launches
    )
    assert all(
        "lightcone_spec.sglang_bridge.launch" in launch.argv
        and "--checkout" in launch.argv
        for launch in launches
    )
    adapted = [
        RunConfig.model_validate_json(Path(launch.run_config).read_text())
        for launch in launches[1:]
    ]
    assert all(
        config.adaptation is not None
        and config.adaptation.loss_position_decay
        == pytest.approx(DFLASH_LOSS_POSITION_DECAY, abs=1e-15)
        for config in adapted
    )


def test_static_load_renderer_has_no_adaptation_identity_or_allocation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "lightcone_spec.orchestration.runtime.verify_patched_checkout",
        lambda path: Path(path).resolve(),
    )
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "a" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    _plan, compile_plan_path = _compile_cache_plan(
        tmp_path,
        lock,
        max_running_requests=48,
    )
    launches = render_static_load_runtime_plan(
        output_root=tmp_path / "static-c48",
        concurrency=48,
        model_lock=lock,
        model_roots={
            "schema_version": 2,
            "lock_sha256": lock.sha256,
            "roots": {
                "Qwen/Qwen3-8B": str(target),
                "z-lab/Qwen3-8B-DFlash-b16": str(drafter),
            },
        },
        sampling_profile=SamplingProfile(),
        sglang_checkout=tmp_path / "patched-sglang",
        compile_cache_plan_path=compile_plan_path,
        mem_fraction_static=0.7,
    )
    assert len(launches) == 1
    launch = launches[0]
    assert launch.method == "static"
    assert launch.adaptation_config is None
    assert launch.telemetry_path is None
    assert "--speculative-adaptation-config" not in launch.argv
    assert "--speculative-use-rejection-sampling" in launch.argv
    assert "--speculative-speed-study-metrics" in launch.argv
    assert "--disable-cuda-graph" in launch.argv
    assert "--disable-radix-cache" in launch.argv
    config = RunConfig.model_validate_json(
        (tmp_path / "static-c48" / "static" / "run-config.json").read_text()
    )
    assert config.method == "static"
    assert config.adaptation is None
    assert config.runtime.max_running_requests == 48


def test_target_only_renderer_never_resolves_or_launches_a_drafter(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "lightcone_spec.orchestration.runtime.verify_patched_checkout",
        lambda path: Path(path).resolve(),
    )
    target = tmp_path / "target"
    target.mkdir()
    lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "a" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    sampling_profile = SamplingProfile()
    source_config = build_target_only_run_config(
        concurrency=8,
        gpu_uuid="GPU-test",
        model_lock=lock,
        sampling_profile=sampling_profile,
    )
    key = derive_diagnostic_compile_cache_key(
        doctor_report=_passing_compile_doctor(),
        model_lock=lock,
        config=source_config,
    )
    plan = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=tmp_path / "compile-cache",
        cache_mode="build",
    )
    compile_plan_path = plan.write(tmp_path / "compile-cache-plan.json")
    launches = render_target_only_runtime_plan(
        output_root=tmp_path / "target-only",
        concurrency=8,
        gpu_uuid="GPU-test",
        model_lock=lock,
        model_roots={
            "schema_version": 2,
            "lock_sha256": lock.sha256,
            "roots": {"Qwen/Qwen3-8B": str(target)},
        },
        sampling_profile=sampling_profile,
        sglang_checkout=tmp_path / "patched-sglang",
        compile_cache_plan_path=compile_plan_path,
        mem_fraction_static=0.7,
    )
    launch = launches[0]
    assert launch.method == "target_only"
    assert launch.adaptation_config is None
    assert launch.telemetry_path is None
    assert not any("draft" in argument for argument in launch.argv)
    assert "--speculative-algorithm" not in launch.argv
    assert "--speculative-speed-study-metrics" in launch.argv
    assert not any("adaptation" in argument for argument in launch.argv)
    assert "--context-length" in launch.argv
    assert "40960" in launch.argv
    assert "--random-seed" in launch.argv
    assert "--disable-radix-cache" in launch.argv
    assert "--disable-cuda-graph" in launch.argv
    assert "--disable-overlap-schedule" in launch.argv
    rendered_config = RunConfig.model_validate_json(Path(launch.run_config).read_text())
    assert rendered_config == source_config
    assert launch.argv[3:16] == (
        "--checkout",
        str((tmp_path / "patched-sglang").resolve()),
        "--compile-cache-plan",
        str(compile_plan_path.resolve()),
        "--compile-cache-plan-sha256",
        plan.sha256,
        "--compile-cache-key-sha256",
        plan.key.sha256,
        "--run-config",
        launch.run_config,
        "--run-config-sha256",
        run_config_sha256(rendered_config),
        "--",
    )
    assert launch.compile_cache_plan == str(compile_plan_path.resolve())
    assert launch.compile_cache_plan_sha256 == plan.sha256
    assert launch.compile_cache_key_sha256 == plan.key.sha256

    observed: dict[str, object] = {}

    class Process:
        pid = 12345
        returncode = None

    async def fake_create(*argv: str, **kwargs):
        observed["argv"] = argv
        observed["environment"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(
        "lightcone_spec.orchestration.executor.verify_patched_checkout",
        lambda path: Path(path).resolve(),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    asyncio.run(launch_server_subprocess(launch))
    assert observed["argv"] == launch.argv
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test"

    events: list[str] = []

    class _Session:
        def environment(self, environment: dict[str, str]) -> dict[str, str]:
            return dict(environment)

        def complete(self) -> None:
            events.append("complete")

        def fail(self, _error: BaseException, *, reason_code: str) -> None:
            pytest.fail(f"unexpected launch failure: {reason_code}")

    def start(
        loaded: CompileCacheLaunchPlan, *, _release_builder_receipt: bool
    ) -> _Session:
        assert loaded == plan
        assert _release_builder_receipt is False
        events.append("plan-loaded")
        return _Session()

    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout",
        lambda path: Path(path).resolve(),
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.start_compile_cache_launch", start
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch._validate_compile_runtime_environment",
        lambda _plan, _config, _argv: None,
    )
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.runpy.run_module",
        lambda name, *, run_name: events.append(f"{name}:{run_name}"),
    )
    original_path = os.environ.get("PATH")
    assert launch_sglang(list(launch.argv[3:])) == 0
    assert events == ["plan-loaded", "sglang.launch_server:__main__", "complete"]
    assert os.environ.get("PATH") == original_path


def test_runtime_renderer_rejects_foreign_compile_identity_before_materialization(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "lightcone_spec.orchestration.runtime.verify_patched_checkout",
        lambda path: Path(path).resolve(),
    )
    target = tmp_path / "target"
    target.mkdir()
    lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "a" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    plan, _path = _compile_cache_plan(
        tmp_path,
        lock,
        max_running_requests=8,
        target_only=True,
    )
    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir()
    foreign = CompileCacheLaunchPlan.issue(
        key=replace(plan.key, max_running_requests=7),
        cache_root=foreign_root / "cache",
        cache_mode="build",
    )
    foreign_path = foreign.write(foreign_root / "plan.json")
    output = tmp_path / "runtime"

    with pytest.raises(ValueError, match="exact RunConfig"):
        render_target_only_runtime_plan(
            output_root=output,
            concurrency=8,
            gpu_uuid="GPU-target-only",
            model_lock=lock,
            model_roots={
                "schema_version": 2,
                "lock_sha256": lock.sha256,
                "roots": {"Qwen/Qwen3-8B": str(target)},
            },
            sampling_profile=SamplingProfile(),
            sglang_checkout=tmp_path / "patched-sglang",
            compile_cache_plan_path=foreign_path,
            mem_fraction_static=0.7,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "gpu_uuid",
    (
        "local-device-0",
        "GPU-target-a,GPU-target-b",
        "GPU-target-a\nGPU-target-b",
    ),
)
def test_target_only_renderer_rejects_noncanonical_gpu_selector(
    tmp_path: Path, gpu_uuid: str
) -> None:
    lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "a" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    output = tmp_path / "invalid-target-only"

    with pytest.raises(ValueError, match="exactly one canonical GPU UUID"):
        render_target_only_runtime_plan(
            output_root=output,
            concurrency=1,
            gpu_uuid=gpu_uuid,
            model_lock=lock,
            model_roots={},
            sampling_profile=SamplingProfile(),
            sglang_checkout=tmp_path / "patched-sglang",
            compile_cache_plan_path=tmp_path / "compile-cache-plan.json",
            mem_fraction_static=0.7,
        )

    assert not output.exists()


def test_tuning_renderer_cannot_duplicate_static_baseline(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "lightcone_spec.orchestration.runtime.verify_patched_checkout",
        lambda path: Path(path).resolve(),
    )
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "a" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    _plan, compile_plan_path = _compile_cache_plan(
        tmp_path,
        lock,
        max_running_requests=8,
    )
    launches = render_tuning_runtime_plan(
        output_root=tmp_path / "tuning",
        candidate=tuning_candidates()[0],
        concurrency=8,
        model_lock=lock,
        model_roots={
            "schema_version": 2,
            "lock_sha256": lock.sha256,
            "roots": {
                "Qwen/Qwen3-8B": str(target),
                "z-lab/Qwen3-8B-DFlash-b16": str(drafter),
            },
        },
        sampling_profile=SamplingProfile(),
        sglang_checkout=tmp_path / "patched-sglang",
        compile_cache_plan_path=compile_plan_path,
        adaptation_group_id="tuning-a",
        adaptation_reserve_mb=1024,
        mem_fraction_static=0.7,
    )
    assert [launch.method for launch in launches] == ["tts", "l0"]
    assert not (tmp_path / "tuning" / "static").exists()
