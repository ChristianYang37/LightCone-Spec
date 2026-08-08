from __future__ import annotations

import copy
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


runner_test = _load(
    "_diagnostic_spec_runner_test_helpers",
    Path(__file__).with_name("test_run_dflash_tts_calibration_sweep.py"),
)
stage1 = _load(
    "analyze_dflash_tts_calibration",
    SCRIPTS / "analyze_dflash_tts_calibration.py",
)
builder = _load(
    "build_dflash_tts_diagnostic_spec",
    SCRIPTS / "build_dflash_tts_diagnostic_spec.py",
)


def _stage1_candidates() -> list[dict]:
    return [
        {
            "candidate_id": "static",
            "mode": "static",
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "rank": None,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
        {
            "candidate_id": "full-safe",
            "mode": "full-drafter",
            "optimizer": "adamw",
            "learning_rate": 1e-5,
            "weight_decay": 1e-3,
            "rank": None,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
        {
            "candidate_id": "drafter-lora-safe",
            "mode": "drafter-lora",
            "optimizer": "adam",
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "rank": 8,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
        {
            "candidate_id": "full-rank-tail-safe",
            "mode": "full-rank-tail",
            "optimizer": "adam",
            "learning_rate": 3e-6,
            "weight_decay": 0.0,
            "rank": None,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
        {
            "candidate_id": "tail-lora-safe",
            "mode": "tail-lora",
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "rank": 16,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
        {
            "candidate_id": "residual-safe",
            "mode": "output-residual",
            "optimizer": "adam",
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "rank": 16,
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
        },
    ]


def _source_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    unsafe_modes: set[str] | None = None,
) -> tuple[Path, dict, Path]:
    unsafe = unsafe_modes or set()
    output_root = (tmp_path / "stage1-runs").resolve()
    output_root.mkdir()
    spec_payload = runner_test._candidate_spec()
    spec_payload["study_id"] = "mock-stage1-v3"
    spec_payload["candidates"] = _stage1_candidates()
    candidate_spec = output_root / "stage1-candidates.json"
    candidate_spec.write_text(json.dumps(spec_payload, sort_keys=True) + "\n")
    sweep = builder.calibration.load_candidate_sweep(candidate_spec)
    lock_path = output_root / "artifact_identity_lock.json"
    lock_payload = {
        "target": {"identity_sha256": "1" * 64},
        "draft": {"identity_sha256": "2" * 64},
        "tokenizer": {"identity_sha256": "3" * 64},
    }
    lock_path.write_text(json.dumps(lock_payload, sort_keys=True) + "\n")
    lock_record = {
        "path": lock_path.name,
        "file_sha256": builder.aggregation._sha256_file(lock_path),
        "content_sha256": builder.aggregation._sha256_json(lock_payload),
    }

    decisions = []
    candidate_rows = []
    for candidate in sweep.candidates:
        safe = candidate.mode != "static" and candidate.mode not in unsafe
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "aggregate": {"safe_for_selection": safe},
            }
        )
        if candidate.mode == "static":
            continue
        decisions.append(
            {
                "mode": candidate.mode,
                "rank": candidate.config.rank,
                "status": "local_grid_winner" if safe else "no_safe_selection",
                "winner": (
                    {
                        "candidate_id": candidate.candidate_id,
                        "optimizer": candidate.config.optimizer,
                        "learning_rate": candidate.config.learning_rate,
                        "weight_decay": candidate.config.weight_decay,
                        "rank": candidate.config.rank,
                        "aggregate": {"safe_for_selection": True},
                    }
                    if safe
                    else None
                ),
            }
        )
    selection_rule = {"fixture": "verified-stage1-contract-v1"}
    implementation = {
        name: {
            "file": Path(module.__file__).name,
            "sha256": builder.aggregation._sha256_file(Path(module.__file__)),
        }
        for name, module in (
            ("analyzer", stage1),
            ("metric_aggregator", builder.aggregation),
            ("calibration_orchestrator", builder.calibration),
            ("frozen_run_validator", builder.calibration.frozen),
        )
    }
    source_runs = []
    for sample in sweep.samples:
        for candidate in sweep.candidates:
            digest = builder.aggregation._sha256_json(
                {
                    "sample_index": sample["sample_index"],
                    "candidate_id": candidate.candidate_id,
                }
            )
            source_runs.append(
                {
                    "sample_index": sample["sample_index"],
                    "candidate_id": candidate.candidate_id,
                    "run_identity_sha256": digest,
                    "identity_sha256": digest,
                    "completion_sha256": digest,
                    "summary_sha256": digest,
                    "rounds_sha256": digest,
                    "command_sha256": digest,
                }
            )
    pareto_rows: list[dict] = []
    payload = {
        "schema_version": stage1.SCHEMA_VERSION,
        "kind": stage1.KIND,
        "status": "complete",
        "study_id": sweep.study_id,
        "evidence_scope": sweep.evidence_scope,
        "candidate_specification": {
            "path": candidate_spec.name,
            "file_sha256": sweep.file_sha256,
            "content_sha256": sweep.content_sha256,
            "study_id": sweep.study_id,
            "schema_version": builder.calibration.SCHEMA_VERSION,
            "kind": sweep.kind,
            "evidence_scope": sweep.evidence_scope,
        },
        "sample_indices": [sample["sample_index"] for sample in sweep.samples],
        "selection_rule": selection_rule,
        "selection_rule_sha256": builder.aggregation._sha256_json(selection_rule),
        "analysis_implementation": implementation,
        "analysis_implementation_sha256": builder.aggregation._sha256_json(
            implementation
        ),
        "candidate_rows": candidate_rows,
        "candidate_rows_sha256": builder.aggregation._sha256_json(candidate_rows),
        "selection_decisions": decisions,
        "selection_decisions_sha256": builder.aggregation._sha256_json(decisions),
        "pareto": {
            "rows": pareto_rows,
            "rows_sha256": builder.aggregation._sha256_json(pareto_rows),
        },
        "artifact_identity_lock": lock_record,
        "source_run_count": len(source_runs),
        "source_artifact_count": len(source_runs) * 4 + 1,
        "source_artifact_set_sha256": builder.aggregation._sha256_json(
            {"artifact_identity_lock": lock_record, "runs": source_runs}
        ),
        "source_runs": source_runs,
        "analysis_hash_scheme": "canonical_json_without_analysis_sha256_v1",
    }
    payload["analysis_sha256"] = builder.aggregation._sha256_json(payload)
    analysis_path = output_root / "stage1-analysis.json"
    analysis_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    active_stage1 = sys.modules["analyze_dflash_tts_calibration"]
    monkeypatch.setattr(
        active_stage1,
        "build_analysis",
        lambda **_kwargs: copy.deepcopy(payload),
    )
    return analysis_path, payload, output_root


def _write_spec(path: Path, payload: dict) -> None:
    path.write_text(builder._render(payload), encoding="utf-8")


def _write_resigned_analysis(path: Path, payload: dict) -> None:
    resigned = copy.deepcopy(payload)
    resigned.pop("analysis_sha256", None)
    resigned["analysis_sha256"] = builder.aggregation._sha256_json(resigned)
    path.write_text(stage1._render(resigned), encoding="utf-8")


def test_public_stage1_verifier_closes_embedded_and_live_evidence_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    analysis_path, source, _stage1_root = _source_analysis(tmp_path, monkeypatch)
    verified = builder.calibration.verify_stage1_published_analysis(analysis_path)
    assert verified.path == analysis_path.resolve()
    assert verified.file_sha256 == builder.aggregation._sha256_file(analysis_path)
    assert verified.payload["analysis_sha256"] == source["analysis_sha256"]

    # A forger can make all local JSON hashes internally consistent.  The
    # published-analysis verifier must still rebuild from live run artifacts.
    tampered = copy.deepcopy(source)
    tampered["source_runs"][0]["summary_sha256"] = "0" * 64
    tampered["source_artifact_set_sha256"] = builder.aggregation._sha256_json(
        {
            "artifact_identity_lock": tampered["artifact_identity_lock"],
            "runs": tampered["source_runs"],
        }
    )
    _write_resigned_analysis(analysis_path, tampered)
    with pytest.raises(ValueError, match="calibration analysis is stale"):
        builder.calibration.verify_stage1_published_analysis(analysis_path)

    analysis_path.write_text(stage1._render(source), encoding="utf-8")
    lock_path = analysis_path.parent / source["artifact_identity_lock"]["path"]
    lock_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity lock hash mismatch"):
        builder.calibration.verify_stage1_published_analysis(analysis_path)


def test_stage1_candidate_spec_content_binding_is_revalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    analysis_path, source, _stage1_root = _source_analysis(tmp_path, monkeypatch)
    candidate_path = (
        analysis_path.parent / source["candidate_specification"]["path"]
    )
    candidate_payload = json.loads(candidate_path.read_text())
    candidate_payload["study_id"] = "forged-stage1-study"
    candidate_path.write_text(json.dumps(candidate_payload, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="candidate specification binding mismatch"):
        builder.calibration.verify_stage1_published_analysis(analysis_path)


def test_builds_isolated_complete_pairs_audits_and_projection_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    analysis_path, source, stage1_root = _source_analysis(tmp_path, monkeypatch)
    diagnostic_root = (tmp_path / "diagnostic-runs").resolve()
    payload = builder.build_diagnostic_spec(
        stage1_analysis=analysis_path,
        diagnostic_output_root=diagnostic_root,
    )
    spec_path = tmp_path / "diagnostics.json"
    _write_spec(spec_path, payload)
    sweep = builder.calibration.load_candidate_sweep(spec_path)

    assert sweep.kind == builder.calibration.DIAGNOSTIC_SPEC_KIND
    assert sweep.study_id != source["study_id"]
    assert len(sweep.candidates) == 13
    assert [candidate.selection_eligible for candidate in sweep.candidates] == [
        True,
        *([False] * 12),
    ]
    by_mode_kind: dict[tuple[str, str], list] = {}
    for candidate in sweep.candidates:
        by_mode_kind.setdefault(
            (candidate.mode, candidate.diagnostic_kind), []
        ).append(candidate)
    static = [candidate for candidate in sweep.candidates if candidate.mode == "static"]
    assert len(static) == 2
    assert {
        (candidate.draft_cache_policy, candidate.diagnostic_kind)
        for candidate in static
    } == {
        ("stale", "selection"),
        ("rebuild", "cache-policy-diagnostic"),
    }
    for mode in builder.CACHE_MODES:
        pair = by_mode_kind[(mode, "cache-policy-diagnostic")]
        assert {candidate.draft_cache_policy for candidate in pair} == {
            "stale",
            "rebuild",
        }
        assert len({candidate.config for candidate in pair}) == 1
    for mode, stride in builder.AUDIT_STRIDES.items():
        [audit] = by_mode_kind[(mode, "parameter-audit")]
        assert audit.parameter_audit_stride == stride
        assert audit.draft_cache_policy == "stale"

    provenance = sweep.provenance
    assert provenance is not None
    isolation = provenance["selection_isolation"]
    assert Path(isolation["source_output_root"]) == stage1_root
    assert Path(isolation["diagnostic_output_root"]) == diagnostic_root
    residual = provenance["residual_projection_requirement"]
    assert residual == {
        "required": True,
        "rank": 16,
        "candidate_ids": ["audit-output-residual-stride32"],
        "runner_binding": "single_identity_bound_projection_artifact_at_run_plan_v1",
    }
    assert provenance["winner_derivation"]["omitted_diagnostics"] == []


def test_no_safe_scope_is_omitted_with_reason_and_never_half_paired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    analysis_path, _source, _stage1_root = _source_analysis(
        tmp_path,
        monkeypatch,
        unsafe_modes={"drafter-lora", "output-residual"},
    )
    payload = builder.build_diagnostic_spec(
        stage1_analysis=analysis_path,
        diagnostic_output_root=tmp_path / "diagnostic-runs",
    )
    assert not any(
        candidate["mode"] in {"drafter-lora", "output-residual"}
        for candidate in payload["candidates"]
    )
    omissions = payload["provenance"]["winner_derivation"][
        "omitted_diagnostics"
    ]
    assert {
        (item["mode"], item["diagnostic_kind"], item["reason"])
        for item in omissions
    } == {
        (
            "drafter-lora",
            "cache-policy-diagnostic",
            "stage1_no_safe_selection",
        ),
        ("drafter-lora", "parameter-audit", "stage1_no_safe_selection"),
        ("output-residual", "parameter-audit", "stage1_no_safe_selection"),
    }
    assert payload["provenance"]["residual_projection_requirement"][
        "required"
    ] is False
    spec_path = tmp_path / "diagnostics.json"
    _write_spec(spec_path, payload)
    sweep = builder.calibration.load_candidate_sweep(spec_path)
    for mode in ("full-drafter", "tail-lora"):
        pair = [
            candidate
            for candidate in sweep.candidates
            if candidate.mode == mode
            and candidate.diagnostic_kind == "cache-policy-diagnostic"
        ]
        assert {candidate.draft_cache_policy for candidate in pair} == {
            "stale",
            "rebuild",
        }


def test_pair_tamper_and_wrong_output_root_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    analysis_path, _source, _stage1_root = _source_analysis(tmp_path, monkeypatch)
    diagnostic_root = (tmp_path / "diagnostic-runs").resolve()
    payload = builder.build_diagnostic_spec(
        stage1_analysis=analysis_path,
        diagnostic_output_root=diagnostic_root,
    )
    payload["candidates"] = [
        candidate
        for candidate in payload["candidates"]
        if candidate["candidate_id"] != "cache-full-drafter-rebuild"
    ]
    spec_path = tmp_path / "tampered-pair.json"
    _write_spec(spec_path, payload)
    with pytest.raises(ValueError, match="requires explicit stale and rebuild"):
        builder.calibration.load_candidate_sweep(spec_path)

    missing_static_rebuild = builder.build_diagnostic_spec(
        stage1_analysis=analysis_path,
        diagnostic_output_root=diagnostic_root,
    )
    missing_static_rebuild["candidates"] = [
        candidate
        for candidate in missing_static_rebuild["candidates"]
        if candidate["candidate_id"] != "static-rebuild"
    ]
    _write_spec(spec_path, missing_static_rebuild)
    with pytest.raises(
        ValueError,
        match="one selection Static-stale and one non-selection Static-rebuild",
    ):
        builder.calibration.load_candidate_sweep(spec_path)

    clean = builder.build_diagnostic_spec(
        stage1_analysis=analysis_path,
        diagnostic_output_root=diagnostic_root,
    )
    provenance_tamper = copy.deepcopy(clean)
    provenance_tamper["provenance"]["residual_projection_requirement"][
        "rank"
    ] = 8
    _write_spec(spec_path, provenance_tamper)
    with pytest.raises(ValueError, match="residual projection provenance mismatch"):
        builder.calibration.load_candidate_sweep(spec_path)

    _write_spec(spec_path, clean)
    runner_root = tmp_path / "runner"
    runner_root.mkdir()
    argv, _selection_spec = runner_test._base_argv(runner_root)
    spec_index = argv.index("--candidate-spec") + 1
    argv[spec_index] = str(spec_path)
    args = builder.calibration.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="does not match the provenance-bound"):
        builder.calibration.build_run_plans(args)


def test_winner_candidate_and_source_derivation_tamper_fail_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    analysis_path, _source, _stage1_root = _source_analysis(tmp_path, monkeypatch)
    diagnostic_root = (tmp_path / "diagnostic-runs").resolve()
    clean = builder.build_diagnostic_spec(
        stage1_analysis=analysis_path,
        diagnostic_output_root=diagnostic_root,
    )
    spec_path = tmp_path / "tampered-derivation.json"

    relative_source = copy.deepcopy(clean)
    relative_source["provenance"]["source_stage1_analysis"]["path"] = (
        str(analysis_path.relative_to(spec_path.parent))
    )
    _write_spec(spec_path, relative_source)
    builder.calibration.load_candidate_sweep(spec_path)

    winner_tamper = copy.deepcopy(clean)
    winner_tamper["provenance"]["winner_derivation"]["selected_winners"][0][
        "source_candidate_sha256"
    ] = "0" * 64
    _write_spec(spec_path, winner_tamper)
    with pytest.raises(ValueError, match="winner derivation does not match Stage-1"):
        builder.calibration.load_candidate_sweep(spec_path)

    candidate_tamper = copy.deepcopy(clean)
    audit = next(
        candidate
        for candidate in candidate_tamper["candidates"]
        if candidate["candidate_id"] == "audit-full-rank-tail-stride32"
    )
    audit["learning_rate"] = 9e-6
    _write_spec(spec_path, candidate_tamper)
    with pytest.raises(ValueError, match="not the exact Stage-1 winner derivation"):
        builder.calibration.load_candidate_sweep(spec_path)

    source_tamper = copy.deepcopy(clean)
    source_tamper["provenance"]["source_stage1_analysis"][
        "analysis_sha256"
    ] = "0" * 64
    _write_spec(spec_path, source_tamper)
    with pytest.raises(ValueError, match="does not bind the verified Stage-1"):
        builder.calibration.load_candidate_sweep(spec_path)

    # The execution entry point has to reject derivation tamper before any
    # model directory, lock, projection, or artifact identity is loaded.
    _write_spec(spec_path, candidate_tamper)
    runner_root = tmp_path / "runner-before-model"
    runner_root.mkdir()
    argv, _selection_spec = runner_test._base_argv(runner_root)
    argv[argv.index("--candidate-spec") + 1] = str(spec_path)
    argv[argv.index("--output-root") + 1] = str(diagnostic_root)
    touched = False

    def _model_load_forbidden(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("model identity loading must not be reached")

    monkeypatch.setattr(
        builder.calibration, "_common_identities", _model_load_forbidden
    )
    with pytest.raises(ValueError, match="not the exact Stage-1 winner derivation"):
        builder.calibration.build_run_plans(
            builder.calibration.build_parser().parse_args(argv)
        )
    assert touched is False


def test_publication_is_no_clobber_checkable_and_source_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    analysis_path, _source, _stage1_root = _source_analysis(tmp_path, monkeypatch)
    diagnostic_root = tmp_path / "diagnostic-runs"
    output = tmp_path / "published" / "diagnostics.json"
    argv = [
        "--stage1-analysis",
        str(analysis_path),
        "--diagnostic-output-root",
        str(diagnostic_root),
        "--output",
        str(output),
    ]
    assert builder.main(argv) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        builder.main(argv)
    assert output.read_bytes() == original
    assert builder.main(
        [
            "--stage1-analysis",
            str(analysis_path),
            "--diagnostic-output-root",
            str(diagnostic_root),
            "--check",
            str(output),
        ]
    ) == 0

    output.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="diagnostic candidate specification is stale"):
        builder.main(
            [
                "--stage1-analysis",
                str(analysis_path),
                "--diagnostic-output-root",
                str(diagnostic_root),
                "--check",
                str(output),
            ]
        )
    output.write_bytes(original)
    source = json.loads(analysis_path.read_text())
    source["source_artifact_set_sha256"] = "b" * 64
    analysis_path.write_text(json.dumps(source, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        builder.main(
            [
                "--stage1-analysis",
                str(analysis_path),
                "--diagnostic-output-root",
                str(diagnostic_root),
                "--check",
                str(output),
            ]
        )
