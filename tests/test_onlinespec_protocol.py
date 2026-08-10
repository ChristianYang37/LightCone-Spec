from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.cli.main import (
    _assert_onlinespec_candidate_config,
    _write_json,
    main,
)
from lightcone_spec.config import RunConfig
from lightcone_spec.experiments.data import DFLASH_SAFE_CONTEXT_LIMIT
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_CLAIM_SCOPE,
    ONLINE_SPEC_COMMIT,
    ONLINE_SPEC_METHODS,
    ONLINE_SPEC_SOURCE_AUDIT_SHA256,
    ONLINE_SPEC_STUDY_METHODS,
    ONLINE_SPEC_TREE,
    OnlineSpecManifest,
    OnlineSpecSelection,
    OnlineSpecTuningMeasurement,
    compare_onlinespec,
    onlinespec_candidates,
    reduce_onlinespec_tuning_stage,
    select_onlinespec,
    select_onlinespec_heldout_anchor,
    verify_onlinespec_source_checkout,
)
from lightcone_spec.experiments.protocol import (
    DFLASH_LOSS_POSITION_DECAY,
    TUNING_STAGES,
    onlinespec_blocks,
    tuning_candidates,
)
from lightcone_spec.experiments.runner import _earlier_slices
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.selection import (
    LossPoint,
    SelectionArtifact,
    SliceMeasurement,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration.manifest import SpeedStudyManifest
from lightcone_spec.orchestration.runtime import (
    render_onlinespec_runtime_plan,
    render_onlinespec_tuning_runtime_plan,
)


def test_onlinespec_manifest_pins_clean_room_provenance(tmp_path) -> None:
    manifest = OnlineSpecManifest.default()
    assert manifest.official_commit == ONLINE_SPEC_COMMIT
    assert manifest.official_tree == ONLINE_SPEC_TREE
    assert manifest.implementation == "clean-room-paper-equations"
    assert manifest.claim_scope == ONLINE_SPEC_CLAIM_SCOPE
    assert manifest.source_audit_sha256 == ONLINE_SPEC_SOURCE_AUDIT_SHA256
    assert manifest.methods == ONLINE_SPEC_STUDY_METHODS
    path = tmp_path / "onlinespec.json"
    manifest.write(path)
    assert OnlineSpecManifest.load(path) == manifest
    with pytest.raises(ValueError, match="registered protocol"):
        replace(manifest, official_commit="f" * 40).validate()


def test_onlinespec_source_checkout_is_content_verified_and_must_be_clean(
    tmp_path,
) -> None:
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    subprocess.run(("git", "init", "-q", str(checkout)), check=True)
    subprocess.run(
        ("git", "-C", str(checkout), "config", "user.name", "Fixture"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(checkout), "config", "user.email", "fixture@example.invalid"),
        check=True,
    )
    source = checkout / "pipeline.py"
    source.write_text("UPDATE = 'predict-feedback-update'\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(checkout), "add", "pipeline.py"), check=True)
    subprocess.run(
        ("git", "-C", str(checkout), "commit", "-q", "-m", "fixture"),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    audit = {
        "schema_version": 2,
        "repository": "https://example.invalid/upstream",
        "commit": commit,
        "tree": tree,
        "key_files": {"pipeline.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "license_files": [],
        "license_status": "no-license-file-present-at-audited-commit",
    }
    canonical = json.dumps(audit, sort_keys=True, separators=(",", ":"))
    audit_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(canonical, encoding="utf-8")
    Path(f"{audit_path}.sha256").write_text(audit_sha256 + "\n", encoding="utf-8")

    receipt = verify_onlinespec_source_checkout(
        checkout,
        audit_path,
        expected_audit_sha256=audit_sha256,
    )
    assert receipt["commit"] == commit
    assert receipt["tree"] == tree
    assert receipt["verified_key_files"] == 1
    assert receipt["clean"] is True

    source.write_text("UPDATE = 'tampered'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be clean"):
        verify_onlinespec_source_checkout(
            checkout,
            audit_path,
            expected_audit_sha256=audit_sha256,
        )


def test_onlinespec_grid_is_complete_unique_and_algorithm_specific() -> None:
    candidates = onlinespec_candidates()
    assert len(candidates) == 236
    assert len({candidate.candidate_id for candidate in candidates}) == 236
    assert {candidate.method for candidate in candidates} == set(ONLINE_SPEC_METHODS)
    ensemble = [
        candidate for candidate in candidates if candidate.method == "onlinespec_ens"
    ]
    assert {candidate.weight_update_mode for candidate in ensemble} == {
        "full",
        "lora",
    }
    assert {candidate.rank for candidate in ensemble} == {None, 8, 16, 32}
    assert all(len(candidate.additional_learning_rates) == 2 for candidate in ensemble)
    assert {
        candidate.weight_update_mode
        for candidate in candidates
        if candidate.method != "onlinespec_ens"
    } == {"full", "lora"}
    assert {
        candidate.stride
        for candidate in candidates
        if candidate.method != "onlinespec_ens"
    } == {20, 40, 80, 160}
    assert {
        candidate.learning_rate
        for candidate in candidates
        if candidate.method != "onlinespec_ens"
    } == {1e-4, 1e-3, 1e-2, 1e-1}
    assert {
        candidate.learning_rate
        for candidate in candidates
        if candidate.method == "onlinespec_ens"
    } == {1e-4, 1e-3, 1e-2}


def test_onlinespec_schedule_is_paired_randomized_and_resumable() -> None:
    blocks = onlinespec_blocks(17)
    assert len(blocks) == 8
    assert all(
        set(block.method_order) == set(ONLINE_SPEC_STUDY_METHODS) for block in blocks
    )
    target = blocks[3].method_order[2]
    prior = _earlier_slices(
        method=target,
        block=3,
        schedule_seed=17,
        study_methods=ONLINE_SPEC_STUDY_METHODS,
    )
    assert prior[-1] == (3, blocks[3].method_order[1])


def measurement(candidate, ratio, *, safe=True) -> OnlineSpecTuningMeasurement:
    return OnlineSpecTuningMeasurement(
        method=candidate.method,
        candidate_id=candidate.candidate_id,
        goodput_ratio_to_static=ratio,
        peak_hbm_bytes=100,
        itl_p99_ms=2.0,
        exposed_update_ms=1.0,
        updates_launched=2,
        updates_published=2,
        safety_violations=0 if safe else 1,
    )


def test_onlinespec_selection_is_per_method_tuning_only_and_fail_closed() -> None:
    by_method = {
        method: [row for row in onlinespec_candidates() if row.method == method][:2]
        for method in ONLINE_SPEC_METHODS
    }
    candidates = {
        candidate.candidate_id: candidate
        for rows in by_method.values()
        for candidate in rows
    }
    evidence = []
    for rows in by_method.values():
        evidence.extend((measurement(rows[0], 1.0), measurement(rows[1], 1.1)))
    selection = select_onlinespec(
        evidence,
        candidates=candidates,
        selected_concurrency=8,
        manifest_sha256="a" * 64,
        model_lock_sha256="b" * 64,
        sampling_profile_sha256="c" * 64,
        reference_core_selection_sha256="e" * 64,
        tuning_evidence_sha256="d" * 64,
    )
    assert all(
        candidate == by_method[candidate.method][1] for candidate in selection.selected
    )
    assert selection.tuning_evidence_sha256 == "d" * 64
    assert selection.selection_protocol == "successive_halving"
    unsafe = [
        replace(row, safety_violations=1)
        for row in evidence
        if row.method == "onlinespec_opt"
    ]
    with pytest.raises(ValueError, match="no safe"):
        select_onlinespec(
            [row for row in evidence if row.method != "onlinespec_opt"] + unsafe,
            candidates=candidates,
            selected_concurrency=8,
            manifest_sha256="a" * 64,
            model_lock_sha256="b" * 64,
            sampling_profile_sha256="c" * 64,
            reference_core_selection_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="selection load"):
        select_onlinespec(
            evidence,
            candidates=candidates,
            selected_concurrency=48,
            manifest_sha256="a" * 64,
            model_lock_sha256="b" * 64,
            sampling_profile_sha256="c" * 64,
            reference_core_selection_sha256="e" * 64,
        )


def test_onlinespec_cli_inherits_and_binds_the_core_static_load(tmp_path) -> None:
    manifest = OnlineSpecManifest.default()
    manifest_path = tmp_path / "onlinespec-manifest.json"
    manifest.write(manifest_path)
    sampling = SamplingProfile()
    sampling_path = tmp_path / "sampling.json"
    sampling.write(sampling_path)
    lock = ModelLock(
        2,
        (
            LockedModel("Qwen/Qwen3-8B", "a" * 40),
            LockedModel("z-lab/Qwen3-8B-DFlash-b16", "b" * 40),
        ),
    )
    lock_path = tmp_path / "models.json"
    lock.write(lock_path)
    core_manifest = SpeedStudyManifest.default()
    core_selection = SelectionArtifact(
        schema_version=2,
        candidate=tuning_candidates()[0],
        selected_concurrency=8,
        minimum_goodput_ratio=1.0,
        peak_hbm_bytes=1,
        itl_p99_ms=1.0,
        exposed_update_ms=1.0,
        manifest_sha256=core_manifest.sha256,
        sampling_profile_sha256=sampling.sha256,
        tuning_grid_sha256=core_manifest.tuning_grid_sha256,
        load_screen_sha256="c" * 64,
        tuning_window_sha256=core_manifest.controlled_window_hashes["tune"],
        model_lock_sha256=lock.sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        tuning_evidence_sha256="d" * 64,
    )
    core_selection_path = tmp_path / "core-selection.json"
    core_selection.write(core_selection_path)
    candidates = {
        method: next(
            candidate
            for candidate in onlinespec_candidates()
            if candidate.method == method
        )
        for method in ONLINE_SPEC_METHODS
    }
    terminal = {
        "schema_version": 2,
        "phase": "onlinespec_tuning",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": lock.sha256,
        "sampling_profile_sha256": sampling.sha256,
        "window_sha256": manifest.tuning_window_sha256,
        "tuning_grid_sha256": manifest.tuning_grid_sha256,
        "stage": len(TUNING_STAGES) - 1,
        "next_stage": None,
        "prior_stage_sha256": "e" * 64,
        "concurrency": 8,
        "measurements": [
            asdict(measurement(candidates[method], 1.0))
            for method in ONLINE_SPEC_METHODS
        ],
    }
    terminal_path = tmp_path / "terminal.json"
    _write_json(terminal_path, terminal)
    output = tmp_path / "onlinespec-selection.json"
    assert (
        main(
            [
                "select-onlinespec-config",
                "--measurements",
                str(terminal_path),
                "--manifest",
                str(manifest_path),
                "--model-lock",
                str(lock_path),
                "--sampling-profile",
                str(sampling_path),
                "--core-selection",
                str(core_selection_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    selected = OnlineSpecSelection.load(output)
    assert selected.selected_concurrency == 8
    assert selected.reference_core_selection_sha256 == core_selection.sha256
    assert selected.selection_protocol == "successive_halving"

    anchor_rows = [tuning_slice(None, 100.0, static=True)]
    anchor_rows.extend(
        tuning_slice(candidates[method], 101.0) for method in ONLINE_SPEC_METHODS
    )
    anchor_paths = []
    for index, row in enumerate(anchor_rows):
        terminal_row = replace(
            row,
            stage=len(TUNING_STAGES) - 1,
            manifest_sha256=manifest.sha256,
            model_lock_sha256=lock.sha256,
            sampling_profile_sha256=sampling.sha256,
            window_sha256=manifest.tuning_window_sha256,
            prompt_count=16,
            context_limit=DFLASH_SAFE_CONTEXT_LIMIT,
            concurrency=8,
        )
        path = tmp_path / f"anchor-{index}.json"
        terminal_row.write(path)
        anchor_paths.append(path)
    anchor_output = tmp_path / "onlinespec-anchor-selection.json"
    assert (
        main(
            [
                "select-onlinespec-anchor-config",
                "--measurements",
                *(str(path) for path in anchor_paths),
                "--candidate-ids",
                *(candidates[method].candidate_id for method in ONLINE_SPEC_METHODS),
                "--manifest",
                str(manifest_path),
                "--model-lock",
                str(lock_path),
                "--sampling-profile",
                str(sampling_path),
                "--core-selection",
                str(core_selection_path),
                "--output",
                str(anchor_output),
            ]
        )
        == 0
    )
    anchor = OnlineSpecSelection.load(anchor_output)
    assert anchor.selection_protocol == "heldout_anchor"
    assert tuple(row.method for row in anchor.selected) == ONLINE_SPEC_METHODS

    mismatched = replace(core_selection, selected_concurrency=4)
    mismatched_path = tmp_path / "mismatched-core-selection.json"
    mismatched.write(mismatched_path)
    with pytest.raises(ValueError, match="tuning artifact identity mismatch"):
        main(
            [
                "select-onlinespec-config",
                "--measurements",
                str(terminal_path),
                "--manifest",
                str(manifest_path),
                "--model-lock",
                str(lock_path),
                "--sampling-profile",
                str(sampling_path),
                "--core-selection",
                str(mismatched_path),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )


def tuning_slice(candidate, goodput, *, static=False, unsafe=False) -> SliceMeasurement:
    return SliceMeasurement(
        schema_version=2,
        phase="onlinespec_tuning",
        stage=0,
        method="static" if static else candidate.method,
        candidate_id=None if static else candidate.candidate_id,
        manifest_sha256="a" * 64,
        config_sha256=("b" if static else "c") * 64,
        model_lock_sha256="d" * 64,
        sampling_profile_sha256="e" * 64,
        window_sha256="f" * 64,
        output_set_sha256="1" * 64,
        prompt_count=2,
        context_limit=4096,
        concurrency=8,
        decode_goodput_tps=goodput,
        itl_p99_ms=2.0,
        peak_hbm_bytes=100,
        kv_bytes=50,
        kv_token_capacity=409600,
        optimizer_bytes=0 if static else 20,
        trainable_parameters=0 if static else 10,
        exposed_update_ms=0.0 if static else 1.0,
        updates_launched=0 if static else 2,
        updates_published=0 if static else 2,
        exactness_violations=int(unsafe),
        version_mismatches=0,
        fallbacks=0,
        nonfinite_updates=0,
        oom_events=0,
        retractions=0,
        loss_points=() if static else (LossPoint(1, 1, 1.0, 0.5),),
    )


def test_onlinespec_anchor_requires_safe_terminal_paired_measurements() -> None:
    selected = {
        method: next(
            candidate
            for candidate in onlinespec_candidates()
            if candidate.method == method
        )
        for method in ONLINE_SPEC_METHODS
    }
    candidates = {candidate.candidate_id: candidate for candidate in selected.values()}
    rows = [tuning_slice(None, 100.0, static=True)]
    rows.extend(tuning_slice(selected[method], 101.0) for method in ONLINE_SPEC_METHODS)
    rows = [
        replace(
            row,
            stage=len(TUNING_STAGES) - 1,
            prompt_count=16,
            context_limit=DFLASH_SAFE_CONTEXT_LIMIT,
        )
        for row in rows
    ]
    selection = select_onlinespec_heldout_anchor(
        rows,
        candidates=candidates,
        selected_concurrency=8,
        manifest_sha256="a" * 64,
        model_lock_sha256="d" * 64,
        sampling_profile_sha256="e" * 64,
        reference_core_selection_sha256="2" * 64,
        tuning_evidence_sha256="3" * 64,
    )
    assert selection.selection_protocol == "heldout_anchor"
    assert tuple(row.method for row in selection.selected) == ONLINE_SPEC_METHODS
    unsafe = [rows[0], replace(rows[1], exactness_violations=1), *rows[2:]]
    with pytest.raises(ValueError, match="no safe candidate|failed"):
        select_onlinespec_heldout_anchor(
            unsafe,
            candidates=candidates,
            selected_concurrency=8,
            manifest_sha256="a" * 64,
            model_lock_sha256="d" * 64,
            sampling_profile_sha256="e" * 64,
            reference_core_selection_sha256="2" * 64,
            tuning_evidence_sha256="3" * 64,
        )


def test_onlinespec_tuning_halves_each_learner_without_confirmation_leakage() -> None:
    chosen = {
        method: [
            candidate
            for candidate in onlinespec_candidates()
            if candidate.method == method
        ][:2]
        for method in ONLINE_SPEC_METHODS
    }
    grid = {
        candidate.candidate_id: candidate
        for values in chosen.values()
        for candidate in values
    }
    slices = [tuning_slice(None, 100.0, static=True)]
    for values in chosen.values():
        slices.extend(
            (
                tuning_slice(values[0], 101.0),
                tuning_slice(values[1], 102.0),
            )
        )
    survivors, reduced = reduce_onlinespec_tuning_stage(
        slices,
        candidates=grid,
        active_candidate_ids=tuple(grid),
        stage=0,
    )
    assert len(survivors) == len(ONLINE_SPEC_METHODS)
    assert {grid[candidate_id].method for candidate_id in survivors} == set(
        ONLINE_SPEC_METHODS
    )
    assert all(row.goodput_ratio_to_static > 1.0 for row in reduced)
    with pytest.raises(ValueError, match="another OnlineSPEC stage"):
        reduce_onlinespec_tuning_stage(
            [replace(slices[0], stage=1), *slices[1:]],
            candidates=grid,
            active_candidate_ids=tuple(grid),
            stage=0,
        )
    with pytest.raises(ValueError, match="Static reference failed"):
        reduce_onlinespec_tuning_stage(
            [replace(slices[0], oom_events=1), *slices[1:]],
            candidates=grid,
            active_candidate_ids=tuple(grid),
            stage=0,
        )
    with pytest.raises(ValueError, match="paired to Static"):
        reduce_onlinespec_tuning_stage(
            [slices[0], replace(slices[1], output_set_sha256="2" * 64), *slices[2:]],
            candidates=grid,
            active_candidate_ids=tuple(grid),
            stage=0,
        )


def test_onlinespec_comparison_is_diagnostic_and_strictly_paired() -> None:
    rows = []
    generated_limit = DFLASH_SAFE_CONTEXT_LIMIT - 49
    ratios = {
        "static": 1.0,
        "onlinespec_ogd": 1.01,
        "onlinespec_opt": 1.04,
        "onlinespec_ens": 0.99,
    }
    for block in range(8):
        for method, ratio in ratios.items():
            safety = {
                "updates_launched": 0 if method == "static" else 1,
                "updates_published": 0 if method == "static" else 1,
                "exactness_violations": 0,
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
                        "generated_bucket_start": bucket,
                        "concurrency": 4,
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
                    "generated_bucket_start": 16384,
                    "generated_bucket_end": generated_limit,
                    "concurrency": 4,
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
                    "generated_bucket_start": 0,
                    "concurrency": 4,
                    "at_risk_requests": 32,
                    "output_tokens": DFLASH_SAFE_CONTEXT_LIMIT * 32,
                    "decode_goodput_tps": 100.0 * ratio,
                    **safety,
                }
            )
    result = compare_onlinespec(rows, seed=7)
    assert [row.method for row in result] == list(ONLINE_SPEC_METHODS)
    assert result[0].mean_speedup == pytest.approx(0.01)
    assert result[2].mean_speedup == pytest.approx(-0.01)
    invalid_bounds = [
        {
            **row,
            "generated_bucket_end": DFLASH_SAFE_CONTEXT_LIMIT,
        }
        if row["region"] == "long_region"
        else row
        for row in rows
    ]
    with pytest.raises(ValueError, match="generated-token positions"):
        compare_onlinespec(invalid_bounds, seed=7)
    assert [row.acceleration_pass for row in result] == [False, True, False]
    assert [row.passed for row in result] == [False, True, False]
    unsafe_rows = [dict(row) for row in rows]
    next(
        row
        for row in unsafe_rows
        if row["method"] == "static" and row["region"] == "full_trajectory"
    )["exactness_violations"] = 1
    assert not any(row.safety_pass for row in compare_onlinespec(unsafe_rows, seed=7))
    with pytest.raises(ValueError, match="minimum speedup"):
        compare_onlinespec(rows, minimum_speedup=-0.01)
    with pytest.raises(ValueError, match="coverage"):
        compare_onlinespec(rows[:-1])
    short = [dict(row) for row in rows]
    next(row for row in short if row["region"] == "long_region")[
        "generated_bucket_end"
    ] = 32768
    with pytest.raises(ValueError, match="bounds|paired work"):
        compare_onlinespec(short)


def test_onlinespec_renderer_creates_four_sequential_servers(
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
    selected = tuple(
        next(
            candidate
            for candidate in onlinespec_candidates()
            if candidate.method == method
        )
        for method in ONLINE_SPEC_METHODS
    )
    selection = OnlineSpecSelection(
        schema_version=2,
        selected=selected,
        selected_concurrency=8,
        manifest_sha256="c" * 64,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=SamplingProfile().sha256,
        tuning_evidence_sha256="d" * 64,
        reference_core_selection_sha256="e" * 64,
        patched_sglang_tree=PINNED_SGLANG_TREE,
    )
    launches = render_onlinespec_runtime_plan(
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
        adaptation_group_id="onlinespec-study",
        adaptation_reserve_mb=4096,
        mem_fraction_static=0.7,
    )
    assert [launch.method for launch in launches] == list(ONLINE_SPEC_STUDY_METHODS)
    assert len({launch.base_url for launch in launches}) == 1
    assert launches[0].adaptation_config is None
    for launch in launches[1:]:
        config = RunConfig.model_validate_json(Path(launch.run_config).read_text())
        assert config.online_spec is not None
        assert config.adaptation is not None
        assert config.adaptation.optimizer.name == "sgd"
        assert config.adaptation.loss_position_decay == pytest.approx(
            DFLASH_LOSS_POSITION_DECAY, abs=1e-15
        )


def test_onlinespec_tuning_renderer_pairs_candidate_with_static(
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
    candidate = onlinespec_candidates()[0]
    launches = render_onlinespec_tuning_runtime_plan(
        output_root=tmp_path / "tune-runtime",
        candidate=candidate,
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
        adaptation_group_id="candidate-isolated",
        adaptation_reserve_mb=4096,
        mem_fraction_static=0.7,
    )
    assert [launch.method for launch in launches] == ["static", candidate.method]
    assert launches[0].adaptation_config is None
    assert launches[1].adaptation_config is not None
    config = RunConfig.model_validate_json(Path(launches[1].run_config).read_text())
    assert config.adaptation is not None
    assert config.adaptation.loss_position_decay == pytest.approx(
        DFLASH_LOSS_POSITION_DECAY, abs=1e-15
    )


def test_onlinespec_protocol_rejects_runtime_and_loss_drift() -> None:
    candidate = onlinespec_candidates()[0]
    base = RunConfig.model_validate(
        {
            "schema_version": 2,
            "method": candidate.method,
            "model": {
                "target_revision": "a" * 40,
                "drafter_revision": "b" * 40,
            },
            "runtime": {
                "sampling_profile_sha256": "c" * 64,
                "speculative_num_draft_tokens": 16,
                "max_running_requests": 8,
                "telemetry_detail": "headline",
            },
            "adaptation": {
                "weight_update_mode": candidate.weight_update_mode,
                "parameter_scope": candidate.parameter_scope,
                "adaptation_group_id": "onlinespec-test",
                "optimizer": {
                    "name": "sgd",
                    "learning_rate": candidate.learning_rate,
                    "grad_clip": candidate.grad_clip,
                },
                "rank": candidate.rank,
                "stride": candidate.stride,
                "canvas_tokens": 16,
                "loss_position_decay": DFLASH_LOSS_POSITION_DECAY,
            },
            "online_spec": {
                "projection_radius": candidate.projection_radius,
                "additional_learning_rates": candidate.additional_learning_rates,
                "hedge_learning_rate": candidate.hedge_learning_rate,
            },
        }
    )
    _assert_onlinespec_candidate_config(
        base,
        method=candidate.method,
        candidate=candidate,
        concurrency=8,
    )
    with pytest.raises(ValueError, match="load mismatch"):
        _assert_onlinespec_candidate_config(
            base.model_copy(
                update={
                    "runtime": base.runtime.model_copy(
                        update={"max_running_requests": 16}
                    )
                }
            ),
            method=candidate.method,
            candidate=candidate,
            concurrency=8,
        )
    with pytest.raises(ValueError, match="tuning selection"):
        _assert_onlinespec_candidate_config(
            base.model_copy(
                update={
                    "adaptation": base.adaptation.model_copy(
                        update={"loss_position_decay": 0.5}
                    )
                }
            ),
            method=candidate.method,
            candidate=candidate,
            concurrency=8,
        )
