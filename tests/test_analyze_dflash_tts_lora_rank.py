from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
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


builder_test = _load(
    "_stage2_rank_builder_test_helpers_for_analysis",
    Path(__file__).with_name("test_build_dflash_tts_lora_rank_candidates.py"),
)
rank_analysis = _load(
    "analyze_dflash_tts_lora_rank",
    SCRIPTS / "analyze_dflash_tts_lora_rank.py",
)


def _stage2_calls(candidate_id: str, sample_index: int) -> int:
    if candidate_id == "static":
        return 1024
    # IDs begin with drafter-lora/tail-lora, so split from the right.
    prefix, slice_name = candidate_id.rsplit("-lr-", 1)
    mode_name, rank_text = prefix.rsplit("-r", 1)
    rank = int(rank_text)
    # Both views prefer rank 64.  The local LR optimum differs by mode so the
    # tuned envelope is demonstrably distinct from the fixed-center control.
    rank_gain = {4: 10, 8: 20, 16: 35, 32: 55, 64: 80}[rank]
    slice_gain = {
        "drafter-lora": {"div3": 0, "center": 8, "times3": 18},
        "tail-lora": {"div3": 12, "center": 18, "times3": 5},
    }[mode_name][slice_name]
    prompt_penalty = 5 if sample_index == 419 else 0
    return 960 - rank_gain - slice_gain + prompt_penalty


def _completed_stage2(
    tmp_path: Path,
    *,
    unsafe_candidate: str | None = None,
    rank_seed_override: tuple[str, int, int] | None = None,
    omit_stage1_mode: str | None = None,
) -> tuple[Path, Path, Path]:
    stage1_root = tmp_path / "stage1"
    stage1_root.mkdir()
    _stage1_spec, _stage1_output, stage1_analysis = builder_test._completed_stage1(
        stage1_root,
        omit_mode=omit_stage1_mode,
    )
    run_fixture = tmp_path / "stage2-run-fixture"
    run_fixture.mkdir()
    argv, generated_spec = builder_test.analysis_test.runner_test._base_argv(
        run_fixture
    )
    output_root = Path(argv[argv.index("--output-root") + 1])
    candidate_spec = output_root / "stage2-rank-candidates.json"
    assert (
        builder_test.builder.main(
            [
                "--stage1-analysis",
                str(stage1_analysis),
                "--output",
                str(candidate_spec),
            ]
        )
        == 0
    )
    candidate_index = argv.index("--candidate-spec") + 1
    argv[candidate_index] = str(candidate_spec)
    generated_spec.unlink()
    args = rank_analysis.calibration.build_parser().parse_args(argv)
    plans = rank_analysis.calibration.build_run_plans(args)
    rank_analysis.calibration.frozen._ensure_artifact_identity_lock(plans[0])
    output_root = Path(args.output_root)
    for plan in plans:
        candidate_id = plan.identity["calibration_candidate"]["candidate_id"]
        sample_index = plan.identity["dataset"]["sample_index"]
        mode = plan.identity["mode"]
        rank = plan.identity["optimization"]["rank"]
        if (
            rank_seed_override is not None
            and mode == rank_seed_override[0]
            and rank == rank_seed_override[1]
        ):
            identity = deepcopy(plan.identity)
            identity["optimization"]["adapter_seed"] = rank_seed_override[2]
            command = list(plan.command)
            seed_index = command.index("--adapter-seed") + 1
            command[seed_index] = str(rank_seed_override[2])
            identity_sha256 = rank_analysis.calibration.frozen._sha256_json(
                identity
            )
            identity_index = command.index("--run-identity-sha256") + 1
            command[identity_index] = identity_sha256
            command_hash_index = command.index("--command-sha256")
            unsigned_harness_argv = [
                *command[1:command_hash_index],
                *command[command_hash_index + 2 :],
            ]
            command[command_hash_index + 1] = (
                rank_analysis.calibration.frozen._sha256_json(
                    unsigned_harness_argv
                )
            )
            plan = replace(
                plan,
                identity=identity,
                identity_sha256=identity_sha256,
                command=tuple(command),
            )
        calls = _stage2_calls(candidate_id, sample_index)
        if candidate_id == unsafe_candidate and sample_index == 419:
            calls = 1100
        trainable_parameters = (
            0
            if mode == "static"
            else int(rank) * (100 if mode == "drafter-lora" else 50)
        )
        builder_test.analysis_test._write_artifact(
            plan,
            verification_calls=calls,
            peak_hbm=1000 + trainable_parameters * 2,
            trainable_parameters=trainable_parameters,
        )
    return candidate_spec, output_root, stage1_analysis


def test_two_rank_views_resource_records_pareto_boundary_and_attestation(
    tmp_path: Path,
):
    candidate_spec, output_root, _stage1_analysis = _completed_stage2(tmp_path)
    payload = rank_analysis.build_analysis(
        candidate_spec=candidate_spec,
        output_root=output_root,
    )

    assert payload["status"] == "complete"
    assert len(payload["candidate_rows"]) == 31
    assert payload["source_attestation"]["portable_evidence_core"][
        "source_run_count"
    ] == 62
    portable_core = payload["source_attestation"]["portable_evidence_core"]
    assert rank_analysis.calibration.frozen._sha256_json(portable_core) == payload[
        "source_attestation"
    ]["portable_evidence_core_sha256"]
    assert "base_analysis_sha256" not in portable_core
    assert "base_analysis_sha256" in payload["source_attestation"][
        "locator_bound_provenance"
    ]
    assert all(
        "run_root" not in sample
        for row in payload["candidate_rows"]
        for sample in row["sample_results"]
    )
    assert rank_analysis.calibration.frozen._sha256_json(
        payload["candidate_rows"]
    ) == payload["candidate_rows_sha256"]

    tuned = {
        decision["mode"]: decision
        for decision in payload["comparisons"]["tuned_envelope"]
    }
    fixed = {
        decision["mode"]: decision
        for decision in payload["comparisons"]["fixed_center_control"]
    }
    assert tuned["drafter-lora"]["winner"]["candidate_id"] == (
        "drafter-lora-r64-lr-times3"
    )
    assert tuned["tail-lora"]["winner"]["candidate_id"] == (
        "tail-lora-r64-lr-center"
    )
    assert fixed["drafter-lora"]["winner"]["candidate_id"] == (
        "drafter-lora-r64-lr-center"
    )
    assert fixed["tail-lora"]["winner"]["candidate_id"] == (
        "tail-lora-r64-lr-center"
    )
    for decision in (*tuned.values(), *fixed.values()):
        assert decision["rank_boundary"][
            "requires_rank_grid_extension_before_optimum_claim"
        ] is True
        assert decision["rank_boundary"]["suggested_extension_ranks"] == [128]
        assert decision["global_optimum_claim"] is False
        assert all(
            prompt["safe_nonnegative"]
            for prompt in decision["winner"]["prompt_safety"]
        )
        assert decision["winner"]["adapter_seed"] == 0
    assert tuned["drafter-lora"][
        "requires_lr_grid_extension_before_optimum_claim"
    ] is True
    assert tuned["drafter-lora"]["winner_learning_rate_boundary"][
        "at_upper_boundary"
    ] is True
    assert tuned["tail-lora"][
        "requires_lr_grid_extension_before_optimum_claim"
    ] is False
    assert len(tuned["drafter-lora"]["per_rank_local_lr_selection"]) == 5
    assert len(fixed["drafter-lora"]["per_rank_points"]) == 5

    rows = {
        row["candidate_id"]: row for row in payload["candidate_rows"]
    }
    rank64 = rows["drafter-lora-r64-lr-times3"]
    assert rank64["aggregate"]["trainable_parameter_count"] == 6400
    assert rank64["aggregate"]["max_whole_process_peak_hbm_bytes"] == 13800
    assert rank64["aggregate"][
        "max_whole_process_peak_hbm_reserved_bytes"
    ] == 13810
    assert rank64["aggregate"]["optimizer_memory_ledger"][
        "optimizer_moment_bytes"
    ] == 51200
    for sample in rank64["sample_results"]:
        assert sample["update"]["prefix_len_min"] is not None
        assert sample["update"]["prefix_len_max"] > sample["update"][
            "prefix_len_min"
        ]
        assert sample["update"]["loss_first"] > sample["update"]["loss_final"]
        assert rank_analysis.calibration.frozen._is_sha256(
            sample["evidence_hashes"]["rounds_sha256"]
        )
        assert sample["memory"]["whole_process_peak_reserved_bytes"] == 13810
        assert sample["loss_context"]["raw_rounds_relative_path"].endswith(
            "/drafter-lora-r64-lr-times3/artifact/rounds.jsonl"
        )
        assert sample["loss_context"]["points"]
        assert rank_analysis.calibration.frozen._sha256_json(
            sample["loss_context"]["points"]
        ) == sample["loss_context"]["points_sha256"]
    assert payload["pareto"]["views"]["tuned_envelope"][
        "by_mode"
    ]["drafter-lora"]["allocated"]["rows"]
    assert payload["pareto"]["views"]["tuned_envelope"][
        "by_mode"
    ]["drafter-lora"]["reserved"]["rows"]
    for mode in ("drafter-lora", "tail-lora"):
        for axis in ("allocated", "reserved"):
            assert all(
                point["mode"] == mode
                for point in payload["pareto"]["raw_safe_candidates"][
                    "by_mode"
                ][mode][axis]["rows"]
            )
    assert all(
        row["safe_for_selection"]
        for row in payload["pareto"]["raw_safe_candidates"][
            "overall_cross_mode"
        ]["allocated"]["rows"]
    )
    unsigned = dict(payload)
    observed = unsigned.pop("analysis_sha256")
    assert rank_analysis.calibration.frozen._sha256_json(unsigned) == observed


def test_per_prompt_negative_delta_is_excluded_before_rank_selection(tmp_path: Path):
    unsafe = "drafter-lora-r64-lr-times3"
    candidate_spec, output_root, _stage1_analysis = _completed_stage2(
        tmp_path,
        unsafe_candidate=unsafe,
    )
    payload = rank_analysis.build_analysis(
        candidate_spec=candidate_spec,
        output_root=output_root,
    )
    row = next(
        item for item in payload["candidate_rows"] if item["candidate_id"] == unsafe
    )
    prompts = rank_analysis._prompt_safety(row, payload["sample_indices"])
    assert [item["safe_nonnegative"] for item in prompts] == [True, False]
    decision = next(
        item
        for item in payload["comparisons"]["tuned_envelope"]
        if item["mode"] == "drafter-lora"
    )
    assert decision["winner"]["candidate_id"] != unsafe
    rank64 = next(
        item
        for item in decision["per_rank_local_lr_selection"]
        if item["rank"] == 64
    )
    assert unsafe not in rank64["ordered_safe_candidate_ids"]
    assert all(
        item["candidate_id"] != unsafe
        for item in payload["pareto"]["raw_safe_candidates"][
            "overall_cross_mode"
        ]["allocated"]["rows"]
    )


def test_rank_only_control_rejects_attested_mixed_adapter_seed(tmp_path: Path):
    candidate_spec, output_root, _stage1_analysis = _completed_stage2(
        tmp_path,
        rank_seed_override=("drafter-lora", 64, 7),
    )
    with pytest.raises(ValueError, match="adapter_seed mismatch"):
        rank_analysis.build_analysis(
            candidate_spec=candidate_spec,
            output_root=output_root,
        )


def test_single_safe_scope_runs_independently_with_machine_readable_omission(
    tmp_path: Path,
):
    candidate_spec, output_root, _stage1_analysis = _completed_stage2(
        tmp_path,
        omit_stage1_mode="tail-lora",
    )
    payload = rank_analysis.build_analysis(
        candidate_spec=candidate_spec,
        output_root=output_root,
    )
    assert len(payload["candidate_rows"]) == 16
    assert payload["source_attestation"]["portable_evidence_core"][
        "source_run_count"
    ] == 32
    [omission] = payload["mode_omissions"]
    assert omission["mode"] == "tail-lora"
    assert omission["reason"] == "stage1_mode_absent"
    for view in rank_analysis.VIEWS:
        assert [row["mode"] for row in payload["comparisons"][view]] == [
            "drafter-lora"
        ]
        assert set(payload["pareto"]["views"][view]["by_mode"]) == {
            "drafter-lora"
        }


def test_publication_no_clobber_check_and_provenance_tamper(tmp_path: Path):
    candidate_spec, output_root, _stage1_analysis = _completed_stage2(tmp_path)
    output = output_root / "rank-stage2-analysis.json"
    argv = [
        "--candidate-spec",
        str(candidate_spec),
        "--output-root",
        str(output_root),
        "--output",
        str(output),
    ]
    assert rank_analysis.main(argv) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        rank_analysis.main(argv)
    assert output.read_bytes() == original
    assert (
        rank_analysis.main(
            [
                "--candidate-spec",
                str(candidate_spec),
                "--output-root",
                str(output_root),
                "--check",
                str(output),
            ]
        )
        == 0
    )

    output.write_bytes(original.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="stale or tampered"):
        rank_analysis.main(
            [
                "--candidate-spec",
                str(candidate_spec),
                "--output-root",
                str(output_root),
                "--check",
                str(output),
            ]
        )
    output.write_bytes(original)

    original_spec = candidate_spec.read_bytes()
    candidate_spec.write_bytes(original_spec.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="stale or tampered"):
        rank_analysis.build_analysis(
            candidate_spec=candidate_spec,
            output_root=output_root,
        )
    candidate_spec.write_bytes(original_spec)

    sidecar = builder_test.builder.provenance_path(candidate_spec)
    original_sidecar = sidecar.read_bytes()
    sidecar.write_bytes(original_sidecar.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="stale or tampered"):
        rank_analysis.build_analysis(
            candidate_spec=candidate_spec,
            output_root=output_root,
        )
    sidecar.write_bytes(original_sidecar)
    provenance = json.loads(original_sidecar)
    provenance["derivation"]["scopes"][0]["source_winner"][
        "optimizer"
    ] = "adamw"
    sidecar.write_text(json.dumps(provenance, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="provenance_sha256"):
        rank_analysis.build_analysis(
            candidate_spec=candidate_spec,
            output_root=output_root,
        )
    sidecar.write_bytes(original_sidecar)
    assert output.read_bytes() == original


def test_complete_stage2_bundle_relocates_without_byte_rewrite(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    candidate_spec, output_root, _stage1_analysis = _completed_stage2(source)
    analysis_path = output_root / "rank-stage2-analysis.json"
    assert (
        rank_analysis.main(
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
    relative_spec = candidate_spec.relative_to(source)
    relative_root = output_root.relative_to(source)
    relative_analysis = analysis_path.relative_to(source)
    source_bytes = {
        "spec": candidate_spec.read_bytes(),
        "provenance": builder_test.builder.provenance_path(
            candidate_spec
        ).read_bytes(),
        "analysis": analysis_path.read_bytes(),
    }

    relocated = tmp_path / "relocated"
    source.rename(relocated)
    relocated_spec = relocated / relative_spec
    relocated_root = relocated / relative_root
    relocated_analysis = relocated / relative_analysis
    assert relocated_spec.read_bytes() == source_bytes["spec"]
    assert builder_test.builder.provenance_path(relocated_spec).read_bytes() == (
        source_bytes["provenance"]
    )
    assert relocated_analysis.read_bytes() == source_bytes["analysis"]
    assert (
        rank_analysis.main(
            [
                "--candidate-spec",
                str(relocated_spec),
                "--output-root",
                str(relocated_root),
                "--check",
                str(relocated_analysis),
            ]
        )
        == 0
    )


def test_rank_boundary_extension_rules_are_symmetric():
    assert rank_analysis._rank_boundary(4)["suggested_extension_ranks"] == [2]
    assert rank_analysis._rank_boundary(64)["suggested_extension_ranks"] == [128]
    assert rank_analysis._rank_boundary(16)[
        "requires_rank_grid_extension_before_optimum_claim"
    ] is False
    assert rank_analysis._rank_boundary(None)["global_optimum_claim"] is False
