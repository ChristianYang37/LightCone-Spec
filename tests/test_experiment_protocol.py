from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_schema import config_value

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.cli.main import (
    _advance_tuning,
    _static_load_rows,
    _write_json,
)
from lightcone_spec.config.schema import RunConfig
from lightcone_spec.experiments.data import (
    LongContinuationAdapter,
    sample_set_sha256,
)
from lightcone_spec.experiments.evidence import (
    GpuEvidenceAttestation,
    evidence_files_sha256,
)
from lightcone_spec.experiments.onlinespec import OnlineSpecManifest
from lightcone_spec.experiments.protocol import (
    DFLASH_LOSS_POSITION_DECAY,
    assert_matched_confirmation_configs,
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
    select_shared_config,
)
from lightcone_spec.experiments.statistics import (
    bca_mean_interval,
    evaluate_speed_gate,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration import SpeedStudyManifest
from lightcone_spec.orchestration.runtime import (
    render_runtime_plan,
    render_static_load_runtime_plan,
    render_tuning_runtime_plan,
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


def test_controlled_path_does_not_import_optional_dataset_package() -> None:
    import sys

    before = sys.modules.get("datasets")
    LongContinuationAdapter().window("confirm")
    assert sys.modules.get("datasets") is before


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
        SpeedStudyManifest.load(
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
    manifest = SpeedStudyManifest.default()
    path = tmp_path / "speed-study.json"
    manifest.write(path)
    assert SpeedStudyManifest.load(path) == manifest
    assert manifest.confirmation_schedule_seed == 20260809
    path.write_text("{}", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        SpeedStudyManifest.load(path)


def test_tuning_grid_is_complete_and_unique() -> None:
    candidates = tuning_candidates()
    assert len(candidates) == 1050
    assert len({candidate.candidate_id for candidate in candidates}) == 1050
    assert {candidate.optimizer for candidate in candidates} == {
        "adam",
        "adamw",
        "sgdm",
        "nag",
        "muon",
        "lion",
    }
    assert {candidate.stride for candidate in candidates} == {1, 5, 10, 20, 40, 80}
    assert {
        candidate.rank
        for candidate in candidates
        if candidate.weight_update_mode == "lora"
    } == {
        4,
        8,
        16,
        32,
    }
    assert all(
        candidate.rank is None
        for candidate in candidates
        if candidate.weight_update_mode == "full"
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
        (16, 40960),
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
        set(block.method_order) == {"static", "tts", "naive_async"}
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
            "kv_token_capacity": 48 * 40960,
        }
        for concurrency in (1, 2, 4, 8, 16, 32, 48)
    ]


def test_static_load_selection_respects_latency_and_safety() -> None:
    rows = load_screen_rows()
    # c16 and above exceed twice the c1 p99 (22 ms).
    assert select_static_load(rows, required_context_limit=40960) == 8
    rows[-1]["oom_events"] = 1
    assert select_static_load(rows, required_context_limit=40960) == 8
    rows[3]["kv_token_capacity"] = 7 * 40960
    assert select_static_load(rows, required_context_limit=40960) == 4
    with pytest.raises(ValueError, match="complete grid"):
        select_static_load(rows[:-1], required_context_limit=40960)
    duplicated = load_screen_rows() + [load_screen_rows()[0]]
    with pytest.raises(ValueError, match="complete grid"):
        select_static_load(duplicated, required_context_limit=40960)
    invalid = load_screen_rows()
    invalid[0]["decode_goodput_tps"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        select_static_load(invalid, required_context_limit=40960)


def test_static_load_terminal_is_bound_to_manifest_sampling_and_window() -> None:
    manifest = SpeedStudyManifest.default()
    artifact = {
        "schema_version": 2,
        "phase": "static_load_screen",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": "a" * 64,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
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
    manifest = SpeedStudyManifest.default()
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    candidate = tuning_candidates()[0]
    expected_window = sample_set_sha256(LongContinuationAdapter().window("tune")[:4])
    paths = []
    for method, candidate_id, goodput in (
        ("static", None, 100.0),
        ("tts", candidate.candidate_id, 102.0),
        ("naive_async", candidate.candidate_id, 101.0),
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
            "naive_async",
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
            "naive_async",
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
                    "naive_async",
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
    l0 = slice_measurement(
        "naive_async", candidate_id=candidate.candidate_id, goodput=101.0
    )
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
    for method in ("tts", "naive_async"):
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


def test_matched_confirmation_configs_bind_selected_candidate() -> None:
    selected = tuning_candidates()[0]
    static = RunConfig.model_validate(config_value("static"))
    adapted = {}
    for method in ("tts", "naive_async"):
        value = config_value(method)
        value["adaptation"].update(
            weight_update_mode=selected.weight_update_mode,
            parameter_scope=selected.parameter_scope,
            rank=selected.rank,
            stride=selected.stride,
        )
        value["adaptation"]["optimizer"].update(
            name=selected.optimizer,
            learning_rate=selected.learning_rate,
            weight_decay=selected.weight_decay,
            grad_clip=selected.grad_clip,
        )
        adapted[method] = RunConfig.model_validate(value)
    configs = {"static": static, **adapted}
    assert_matched_confirmation_configs(
        configs, selected_candidate=selected, selected_concurrency=8
    )
    changed = config_value("naive_async")
    changed["adaptation"]["stride"] = selected.stride + 1
    configs["naive_async"] = RunConfig.model_validate(changed)
    with pytest.raises(ValueError):
        assert_matched_confirmation_configs(
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


def performance_rows(
    *, speed_tts: float = 1.05, speed_l0: float = 1.04, unsafe: bool = False
) -> list[dict]:
    rows: list[dict] = []
    methods = {"static": 1.0, "tts": speed_tts, "naive_async": speed_l0}
    for prompt in range(32):
        for block in range(8):
            for bucket in (16384, 24576, 32768):
                for method, ratio in methods.items():
                    rows.append(
                        {
                            "prompt_id": f"p{prompt}",
                            "method": method,
                            "repetition_block": block,
                            "region": "generated_bucket",
                            "concurrency": 8,
                            "generated_bucket_start": bucket,
                            "at_risk_requests": 8,
                            "output_tokens": 8192 * 8,
                            "decode_goodput_tps": 100.0 * ratio,
                            "updates_launched": 0 if method == "static" else 4,
                            "updates_published": 0 if method == "static" else 3,
                            "exactness_violations": int(
                                unsafe and method == "tts" and prompt == 0
                            ),
                            "version_mismatches": 0,
                            "fallbacks": 0,
                            "nonfinite_updates": 0,
                            "oom_events": 0,
                            "retractions": 0,
                        }
                    )
    return rows


def test_speed_gate_never_claims_unattested_gpu_success() -> None:
    gate = evaluate_speed_gate(performance_rows(), seed=1)
    assert gate.status == "UNMEASURED"
    assert not gate.passed
    assert gate.tts.acceleration_pass
    assert gate.naive_async.acceleration_pass


def test_measured_speed_gate_requires_both_methods_and_safety() -> None:
    gate = evaluate_speed_gate(
        performance_rows(),
        seed=1,
        gpu_evidence="MEASURED",
        evidence_sha256="e" * 64,
    )
    assert gate.status == "PASS"
    assert gate.passed
    unsafe = evaluate_speed_gate(
        performance_rows(unsafe=True),
        seed=1,
        gpu_evidence="MEASURED",
        evidence_sha256="e" * 64,
    )
    assert unsafe.status == "BLOCKED"
    assert not unsafe.passed


def test_speed_gate_rejects_incomplete_coverage() -> None:
    with pytest.raises(ValueError, match="32 independent"):
        evaluate_speed_gate(performance_rows()[:-72])


def test_gpu_attestation_binds_exact_performance_files(tmp_path) -> None:
    evidence = tmp_path / "performance.parquet"
    evidence.write_bytes(b"parquet-evidence")
    attestation = GpuEvidenceAttestation(
        schema_version=2,
        status="MEASURED",
        manifest_sha256="a" * 64,
        selection_sha256="b" * 64,
        model_lock_sha256="f" * 64,
        performance_sha256=evidence_files_sha256((evidence,)),
        patched_sglang_tree=PINNED_SGLANG_TREE,
        target_revision="c" * 40,
        drafter_revision="d" * 40,
        hardware_sha256="e" * 64,
        methods=("static", "tts", "naive_async"),
        repetitions=8,
        context_start=16384,
        context_limit=40960,
    )
    path = tmp_path / "attestation.json"
    attestation.write(path)
    loaded = GpuEvidenceAttestation.load(path)
    loaded.verify_performance((evidence,))
    evidence.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not bind"):
        loaded.verify_performance((evidence,))


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
        schema_version=2,
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
        adaptation_group_id="study-a",
        adaptation_reserve_mb=1024,
        mem_fraction_static=0.7,
    )
    assert [launch.method for launch in launches] == [
        "static",
        "tts",
        "naive_async",
    ]
    assert launches[0].adaptation_config is None
    assert len({launch.base_url for launch in launches}) == 1
    assert all(launch.exclusive_device for launch in launches)
    assert all(
        "--speculative-adaptation-config" in launch.argv for launch in launches[1:]
    )
    assert all("--disable-cuda-graph" not in launch.argv for launch in launches)
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
    config = RunConfig.model_validate_json(
        (tmp_path / "static-c48" / "static" / "run-config.json").read_text()
    )
    assert config.method == "static"
    assert config.adaptation is None
    assert config.runtime.max_running_requests == 48


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
        adaptation_group_id="tuning-a",
        adaptation_reserve_mb=1024,
        mem_fraction_static=0.7,
    )
    assert [launch.method for launch in launches] == ["tts", "naive_async"]
    assert not (tmp_path / "tuning" / "static").exists()
