from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts" / "experiments"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis_test = _load(
    "_stage2_rank_analysis_test_helpers",
    Path(__file__).with_name("test_analyze_dflash_tts_calibration.py"),
)
builder = _load(
    "build_dflash_tts_lora_rank_candidates",
    SCRIPTS / "build_dflash_tts_lora_rank_candidates.py",
)


def _candidate(
    candidate_id: str,
    mode: str,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    rank: int | None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "mode": mode,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "rank": rank,
        "draft_cache_policy": "stale",
        "diagnostic_kind": "selection",
        "parameter_audit_stride": 0,
    }


def _stage1_candidates(*, omit_mode: str | None = None) -> list[dict]:
    candidates = [
        _candidate("static", "static", "adam", 1e-4, 0.0, None),
        _candidate(
            "drafter-r8-adam-lr1e-4",
            "drafter-lora",
            "adam",
            1e-4,
            0.0,
            8,
        ),
        _candidate(
            "drafter-r8-adam-lr3e-4",
            "drafter-lora",
            "adam",
            3e-4,
            0.0,
            8,
        ),
        _candidate(
            "tail-r16-adamw-lr1e-4",
            "tail-lora",
            "adamw",
            1e-4,
            1e-2,
            16,
        ),
        _candidate(
            "tail-r16-adamw-lr3e-4",
            "tail-lora",
            "adamw",
            3e-4,
            1e-2,
            16,
        ),
    ]
    return [row for row in candidates if row["mode"] != omit_mode]


def _completed_stage1(
    tmp_path: Path,
    *,
    omit_mode: str | None = None,
    unsafe_mode: str | None = None,
) -> tuple[Path, Path, Path]:
    runner_test = analysis_test.runner_test
    argv, generated_candidate_spec = runner_test._base_argv(tmp_path)
    output_root = Path(argv[argv.index("--output-root") + 1])
    output_root.mkdir(parents=True)
    candidate_spec = output_root / "candidates.json"
    argv[argv.index("--candidate-spec") + 1] = str(candidate_spec)
    generated_candidate_spec.unlink()
    payload = runner_test._candidate_spec()
    payload["study_id"] = "mock-stage1-for-rank-builder-v3"
    payload["candidates"] = _stage1_candidates(omit_mode=omit_mode)
    candidate_spec.write_text(json.dumps(payload, sort_keys=True) + "\n")
    args = builder.calibration.build_parser().parse_args(argv)
    plans = builder.calibration.build_run_plans(args)
    builder.calibration.frozen._ensure_artifact_identity_lock(plans[0])
    output_root = Path(args.output_root)

    calls = {
        "static": {0: 1024, 419: 1024},
        "drafter-r8-adam-lr1e-4": {0: 900, 419: 925},
        "drafter-r8-adam-lr3e-4": {0: 700, 419: 750},
        "tail-r16-adamw-lr1e-4": {0: 650, 419: 700},
        "tail-r16-adamw-lr3e-4": {0: 800, 419: 850},
    }
    if unsafe_mode is not None:
        for row in payload["candidates"]:
            if row["mode"] == unsafe_mode or (
                unsafe_mode == "all" and row["mode"] != "static"
            ):
                calls[row["candidate_id"]] = {0: 1100, 419: 1200}
    for plan in plans:
        candidate_id = plan.identity["calibration_candidate"]["candidate_id"]
        sample_index = plan.identity["dataset"]["sample_index"]
        mode = plan.identity["mode"]
        analysis_test._write_artifact(
            plan,
            verification_calls=calls[candidate_id][sample_index],
            peak_hbm={"static": 100, "drafter-lora": 130, "tail-lora": 115}[
                mode
            ],
            trainable_parameters={
                "static": 0,
                "drafter-lora": 256,
                "tail-lora": 128,
            }[mode],
        )

    analysis_path = output_root / "stage1-analysis.json"
    assert (
        builder.stage1_analyzer.main(
            [
                "--candidate-spec",
                str(candidate_spec),
                "--output-root",
                str(output_root),
                "--output",
                str(analysis_path),
            ]
        )
        == 0
    )
    return candidate_spec, output_root, analysis_path


@pytest.mark.parametrize(
    ("omit_mode", "unsafe_mode", "omitted_mode", "reason"),
    [
        ("drafter-lora", None, "drafter-lora", "stage1_mode_absent"),
        (None, "tail-lora", "tail-lora", "stage1_no_safe_selection"),
    ],
)
def test_missing_or_unsafe_stage1_scope_is_machine_readable_omission(
    tmp_path: Path,
    omit_mode: str | None,
    unsafe_mode: str | None,
    omitted_mode: str,
    reason: str,
):
    _spec, _root, analysis_path = _completed_stage1(
        tmp_path,
        omit_mode=omit_mode,
        unsafe_mode=unsafe_mode,
    )
    candidate_spec, provenance = builder.build_bundle(
        stage1_analysis_path=analysis_path,
        candidate_spec_path=tmp_path / "stage2.json",
    )
    assert len(candidate_spec["candidates"]) == 16
    assert provenance["stage2_candidate_specification"]["planned_run_count"] == 32
    [omission] = provenance["derivation"]["omissions"]
    assert omission["mode"] == omitted_mode
    assert omission["reason"] == reason
    assert provenance["derivation"]["active_modes"] == [
        mode for mode in builder.MODES if mode != omitted_mode
    ]


def test_all_unsafe_stage1_scopes_fail_closed(tmp_path: Path):
    _spec, _root, analysis_path = _completed_stage1(
        tmp_path,
        unsafe_mode="all",
    )
    with pytest.raises(ValueError, match="no safe LoRA scope"):
        builder.build_bundle(
            stage1_analysis_path=analysis_path,
            candidate_spec_path=tmp_path / "stage2.json",
        )


def test_stage1_analysis_tamper_is_rebuilt_and_rejected(tmp_path: Path):
    _spec, _root, analysis_path = _completed_stage1(tmp_path)
    payload = json.loads(analysis_path.read_text())
    decision = next(
        row
        for row in payload["selection_decisions"]
        if row["mode"] == "drafter-lora"
    )
    decision["winner"]["learning_rate"] = 9e-4
    unsigned = dict(payload)
    unsigned.pop("analysis_sha256")
    payload["analysis_sha256"] = builder.calibration.frozen._sha256_json(unsigned)
    analysis_path.write_text(builder.stage1_analyzer._render(payload))

    with pytest.raises(ValueError, match="stale or tampered"):
        builder.build_bundle(
            stage1_analysis_path=analysis_path,
            candidate_spec_path=tmp_path / "stage2.json",
        )


def test_exact_rank_lr_list_count_and_preserved_controls(tmp_path: Path):
    stage1_spec, _root, analysis_path = _completed_stage1(tmp_path)
    output = tmp_path / "stage2" / "rank-candidates.json"
    assert (
        builder.main(
            [
                "--stage1-analysis",
                str(analysis_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    sweep = builder.calibration.load_candidate_sweep(output)
    assert len(sweep.candidates) == 31
    assert len(sweep.samples) == 2
    assert len(sweep.candidates) * len(sweep.samples) == 62
    assert [dict(sample) for sample in sweep.samples] == json.loads(
        stage1_spec.read_text()
    )["samples"]
    assert sweep.kind == builder.calibration.SPEC_KIND
    assert sweep.evidence_scope == builder.calibration.EVIDENCE_SCOPE
    assert sum(candidate.mode == "static" for candidate in sweep.candidates) == 1

    expected = {
        "drafter-lora": {
            "optimizer": "adam",
            "weight_decay": 0.0,
            "center": 3e-4,
        },
        "tail-lora": {
            "optimizer": "adamw",
            "weight_decay": 1e-2,
            "center": 1e-4,
        },
    }
    for mode, fixed in expected.items():
        rows = [candidate for candidate in sweep.candidates if candidate.mode == mode]
        assert len(rows) == 15
        assert {candidate.config.rank for candidate in rows} == set(builder.RANKS)
        for rank in builder.RANKS:
            rank_rows = [row for row in rows if row.config.rank == rank]
            assert len(rank_rows) == 3
            assert {row.config.optimizer for row in rank_rows} == {
                fixed["optimizer"]
            }
            assert {row.config.weight_decay for row in rank_rows} == {
                fixed["weight_decay"]
            }
            assert sorted(row.config.learning_rate for row in rank_rows) == pytest.approx(
                [fixed["center"] / 3.0, fixed["center"], fixed["center"] * 3.0]
            )
            assert all(row.draft_cache_policy == "stale" for row in rank_rows)
            assert all(row.selection_eligible for row in rank_rows)
            assert all(row.parameter_audit_stride == 0 for row in rank_rows)

    sidecar = builder.provenance_path(output)
    provenance = json.loads(sidecar.read_text())
    assert provenance["stage2_candidate_specification"]["candidate_count"] == 31
    assert provenance["stage2_candidate_specification"]["planned_run_count"] == 62
    assert provenance["stage1"]["candidate_specification"]["file_sha256"] == (
        builder.calibration.frozen._sha256_file(stage1_spec)
    )
    assert provenance["derivation"]["stage1_binding"][
        "candidate_specification_file_sha256"
    ] == builder.calibration.frozen._sha256_file(stage1_spec)
    assert json.loads(output.read_text())["study_id"].endswith(
        provenance["derivation_sha256"][:24]
    )
    assert provenance["derivation"]["selection_hot_path"]["adapter_seed"] == 0
    assert provenance["stage1"]["analysis"]["locator"]
    assert not Path(provenance["stage1"]["analysis"]["locator"]).is_absolute()
    assert provenance["stage2_candidate_specification"]["locator"] == output.name
    assert provenance["companion_locator"] == sidecar.name
    assert set(provenance["builder"]) == {"file", "sha256"}
    unsigned = dict(provenance)
    observed = unsigned.pop("provenance_sha256")
    assert builder.calibration.frozen._sha256_json(unsigned) == observed


def test_fixed_center_and_boundary_provenance(tmp_path: Path):
    _spec, _root, analysis_path = _completed_stage1(tmp_path)
    output = tmp_path / "rank.json"
    candidate_spec, provenance = builder.build_bundle(
        stage1_analysis_path=analysis_path,
        candidate_spec_path=output,
    )
    candidates = {
        row["candidate_id"]: row for row in candidate_spec["candidates"]
    }
    scopes = {
        row["mode"]: row for row in provenance["derivation"]["scopes"]
    }
    for mode, scope in scopes.items():
        winner = scope["source_winner"]
        control = scope["fixed_center_control"]
        assert control["matches_stage1_winner_config"] is True
        assert winner["adapter_seed"] == 0
        assert control["adapter_seed"] == 0
        candidate = candidates[control["candidate_id"]]
        for key in ("mode", "optimizer", "learning_rate", "weight_decay", "rank"):
            assert candidate[key] == winner[key]
        boundary = scope["boundary_metadata"]
        assert boundary["stage1_learning_rate"] == winner[
            "learning_rate_boundary"
        ]
        assert boundary["stage2_rank_grid"]["minimum"] == 4
        assert boundary["stage2_rank_grid"]["maximum"] == 64
        assert boundary["stage2_rank_grid"]["source_winner_at_rank_boundary"] is False


def test_bundle_is_no_clobber_checkable_and_sidecar_tamper_fails(tmp_path: Path):
    _spec, _root, analysis_path = _completed_stage1(tmp_path)
    output = tmp_path / "rank.json"
    argv = ["--stage1-analysis", str(analysis_path), "--output", str(output)]
    assert builder.main(argv) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.main(argv)
    assert output.read_bytes() == original
    assert (
        builder.main(
            ["--stage1-analysis", str(analysis_path), "--check", str(output)]
        )
        == 0
    )

    sidecar = builder.provenance_path(output)
    sidecar.write_text(sidecar.read_text() + "\n")
    with pytest.raises(ValueError, match="stale or tampered"):
        builder.main(
            ["--stage1-analysis", str(analysis_path), "--check", str(output)]
        )
