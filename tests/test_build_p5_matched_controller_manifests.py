from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from lightcone_spec.artifacts.coverage import build_coverage
from lightcone_spec.exit_codes import ConfigError, LockError
from lightcone_spec.locking.download import write_model_roots
from lightcone_spec.locking.hashing import canonical_json, sha256_json
from lightcone_spec.locking.lockfile import Lockfile
from lightcone_spec.orchestration.catalog import (
    p5_priority_dflash_l3_evaluation_manifest,
    p5_priority_dflash_paired_trace_manifest,
    p5_priority_dflash_stride_screen_manifest,
)
from lightcone_spec.orchestration.controller_manifests import (
    MATCHED_PYTORCH_CUDA_ALLOC_CONF,
    build_matched_controller_manifests,
)
from lightcone_spec.orchestration import controller_manifests as controller_module
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.runtime_config import (
    _verify_bound_model_roots,
    runtime_implementation_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/build_p5_matched_controller_manifests.py"
SELECTOR = ROOT / "scripts/experiments/select_p5_stride_screen.py"
QUEUE = ROOT / "scripts/experiments/run_priority_l0_stride_screen_queue.sh"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(path: Path, sidecar: Path | None = None) -> Path:
    sidecar = sidecar or Path(str(path) + ".sha256")
    sidecar.write_text(_sha(path) + "\n", encoding="utf-8")
    return sidecar


def _write_receipt(
    path: Path,
    *,
    status: str,
    scope: str,
    evidence: list[Path],
) -> Path:
    payload = {
        "schema_version": 1,
        "status": status,
        "scope": scope,
        "evidence": [
            {"path": str(item.resolve()), "sha256": _sha(item)}
            for item in evidence
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(path)
    return path


def _winner(manifest, method: str, stride: int, family: str) -> dict:
    unit = next(
        unit
        for unit in manifest.units
        if unit.method == method and unit.stride == stride
    )
    return {
        "family": family,
        "method": method,
        "stride": stride,
        "unit_id": unit.unit_id,
        "eligible": True,
    }


def _selection(manifest, *, l0_stride: int = 8) -> dict:
    tts_best = _winner(manifest, "tts", 16, "tts")
    l0_best = _winner(manifest, "naive_async", l0_stride, "l0")
    same_tts = _winner(manifest, "tts", l0_stride, "tts")
    static = next(unit for unit in manifest.units if unit.method == "static")
    return {
        "schema_version": 1,
        "status": "winner_selected",
        "scope": "candidate_screen_only_no_claim",
        "objective_screen_pass": True,
        "selection_rule": {"id": "p5_stride_screen_selection_v1"},
        "cardinality": {
            "contexts": [4096, 16384],
            "strides": [1, 4, 8, 16],
            "no_stride_pooling": True,
        },
        "winners": {
            "tts_best": tts_best,
            "l0_best": l0_best,
            "same_stride_tts_for_l0": same_tts,
        },
        "candidates": {
            "tts": [
                _winner(manifest, "tts", stride, "tts")
                for stride in (1, 4, 8, 16)
            ],
            "l0": [
                _winner(manifest, "naive_async", stride, "l0")
                for stride in (1, 4, 8, 16)
            ],
        },
        "confirmation_unit_ids": [
            static.unit_id,
            tts_best["unit_id"],
            l0_best["unit_id"],
            same_tts["unit_id"],
        ],
    }


def _bindings(source, **overrides) -> dict:
    static = next(unit for unit in source.units if unit.method == "static")
    runtime_body = {
        "schema_version": 1,
        "files": {},
        "locked_reference": {},
    }
    runtime = {**runtime_body, "sha256": sha256_json(runtime_body)}
    value = {
        "schema_version": 1,
        "terminal_receipt_sha256": "1" * 64,
        "execution_receipt_sha256": "2" * 64,
        "selection_receipt_sha256": "3" * 64,
        "source_manifest_file_sha256": "4" * 64,
        "source_manifest_sha256": source.content_sha256(),
        "source_static_unit_id": static.unit_id,
        "lockfile_sha256": "5" * 64,
        "model_roots_sha256": "6" * 64,
        "screen_runtime_implementation_sha256": runtime["sha256"],
        "consumer_runtime_implementation_sha256": runtime["sha256"],
        "model_revisions": {
            "target": "7" * 40,
            "drafter": "8" * 40,
            "tokenizer": "7" * 40,
        },
        "pytorch_cuda_alloc_conf": MATCHED_PYTORCH_CUDA_ALLOC_CONF,
        "screen_runtime_implementation_fingerprint": runtime,
        "runtime_implementation_fingerprint": runtime,
        "runtime_transition": {
            "schema_version": 2,
            "screen_sha256": runtime["sha256"],
            "consumer_sha256": runtime["sha256"],
            "equal": True,
            "changed_files": [],
            "added_files": [],
            "removed_files": [],
            "locked_reference_changed": False,
            "authorization_id": "identical_runtime",
            "authorization_basis": "exact_runtime_fingerprint_pair",
            "screen_measurements_reusable": True,
            "selection_reuse_only": False,
            "scientific_equivalence_claim": False,
            "requires_matched_confirmation": False,
        },
        "queue_source_sha256": "a" * 64,
        "selector_source_sha256": "b" * 64,
        "builder_source_sha256": "c" * 64,
        "helper_source_sha256": "d" * 64,
        "contention_mapping": {
            "phase1_naive_async": "realistic_async",
            "phase1_tts": "none",
            "phase2_lc_transport": "realistic_async",
        },
    }
    value.update(overrides)
    return value


def _runtime(files: dict) -> dict:
    body = {
        "schema_version": 1,
        "files": files,
        "locked_reference": {},
    }
    return {**body, "sha256": sha256_json(body)}


def _cell(unit) -> dict:
    value = unit.to_manifest_dict()
    for key in ("phase", "method", "contention_condition", "unit_id"):
        value.pop(key)
    return value


def _frame() -> pd.DataFrame:
    tts = {
        1: (0.80, 0.200, 130.0),
        4: (1.00, 0.180, 110.0),
        8: (0.99, 0.170, 115.0),
        16: (0.99, 0.171, 125.0),
    }
    l0 = {
        1: (0.90, 0.190, 120.0),
        4: (1.10, 0.160, 130.0),
        8: (1.08, 0.155, 140.0),
        16: (1.07, 0.150, 150.0),
    }
    rows = []
    for context in (4096, 16384):
        static_a = 3.0 if context == 16384 else 4.0
        rows.append(
            {
                "method": "static",
                "update_stride": 1,
                "context_length": context,
                "survival_weighted_accepted_prefix": static_a,
                "acceptance_gain_vs_baseline": 0.0,
                "target_calls_per_output_token": 0.25,
                "decode_goodput_tps": 100.0,
                "exactness_violations": 0,
                "version_mismatch_count": 0,
                "adaptation_fallback_count": 0,
            }
        )
        for method, values in (("tts", tts), ("naive_async", l0)):
            for stride, (gain, target_calls, goodput) in values.items():
                rows.append(
                    {
                        "method": method,
                        "update_stride": stride,
                        "context_length": context,
                        "survival_weighted_accepted_prefix": static_a + gain,
                        "acceptance_gain_vs_baseline": gain,
                        "target_calls_per_output_token": target_calls,
                        "decode_goodput_tps": goodput,
                        "exactness_violations": 0,
                        "version_mismatch_count": 0,
                        "adaptation_fallback_count": 0,
                    }
                )
    return pd.DataFrame(rows)


def _write_run(
    root: Path,
    unit,
    *,
    engine: dict,
) -> dict:
    run_id = f"screen-{unit.method}-{unit.stride}"
    run_root = root / run_id
    run_root.mkdir(parents=True)
    manifest = run_root / "manifest.json"
    manifest.write_text(
        canonical_json(
            {
                "run_id": run_id,
                "unit_id": unit.unit_id,
                "engine_params": engine,
            }
        ),
        encoding="utf-8",
    )
    manifest_sidecar = run_root / "manifest.sha256"
    _write_sidecar(manifest, manifest_sidecar)
    exit_path = run_root / "exit.json"
    exit_path.write_text(
        json.dumps({"status": "complete_valid", "exit_code": 0}),
        encoding="utf-8",
    )
    ledger = {}
    for path in (manifest, manifest_sidecar, exit_path):
        ledger[path.name] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    hashes = run_root / "hashes.json"
    hashes.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    return {
        "run_id": run_id,
        "unit_id": unit.unit_id,
        "manifest_sha256": _sha(manifest),
        "hashes_sha256": _sha(hashes),
    }


def _write_analysis(
    root: Path,
    *,
    baseline: str,
    source: ExperimentManifest,
    input_runs: list[dict],
) -> None:
    root.mkdir(parents=True)
    frame = _frame()
    if baseline == "tts":
        frame = frame.copy()
        for context in (4096, 16384):
            for stride in (1, 4, 8, 16):
                tts_a = float(
                    frame[
                        (frame.method == "tts")
                        & (frame.update_stride == stride)
                        & (frame.context_length == context)
                    ].iloc[0].survival_weighted_accepted_prefix
                )
                mask = (
                    (frame.method == "naive_async")
                    & (frame.update_stride == stride)
                    & (frame.context_length == context)
                )
                frame.loc[mask, "acceptance_gain_vs_baseline"] = (
                    frame.loc[mask, "survival_weighted_accepted_prefix"] - tts_a
                )
    table = root / "p5_long_context_acceptance.parquet"
    frame.to_parquet(table, index=False)
    derived = {
        table.name: {"sha256": _sha(table), "bytes": table.stat().st_size}
    }
    manifest = root / "analysis-manifest.json"
    manifest.write_text(
        canonical_json(
            {
                "schema_version": 1,
                "analysis": {
                    "baseline": baseline,
                    "itl_slo_ms": 50.0,
                    "expected_manifest_sha256": source.content_sha256(),
                    "weight_update_mode_overlay": "lora",
                    "methods_overlay": ["static", "tts", "naive_async"],
                    "lifecycles_overlay": None,
                    "learning_rate_overlay": None,
                },
                "input_runs": input_runs,
                "derived_outputs": derived,
            }
        ),
        encoding="utf-8",
    )
    sidecar = root / "analysis-manifest.sha256"
    _write_sidecar(manifest, sidecar)
    ledger = {
        path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in (table, manifest, sidecar)
    }
    (root / "analysis-hashes.json").write_text(
        json.dumps(ledger, sort_keys=True), encoding="utf-8"
    )


def _chain(tmp_path: Path) -> dict[str, Path]:
    builder = _module("matched_builder_fixture", SCRIPT)
    selector = _module("selector_fixture", SELECTOR)
    source = p5_priority_dflash_stride_screen_manifest()
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    source_path = tmp_path / "source-screen.json"
    source.write(source_path)

    target_root = tmp_path / "models" / "target"
    drafter_root = tmp_path / "models" / "drafter"
    target_root.mkdir(parents=True)
    drafter_root.mkdir(parents=True)
    lock = Lockfile.model_validate(
        {
            "schema_version": 1,
            "created_utc": "2026-01-01T00:00:00Z",
            "git_repos": [],
            "hf_snapshots": [
                {
                    "repo_id": "Qwen/Qwen3-4B",
                    "snapshot_sha": "1" * 40,
                    "role": "target",
                    "files": [],
                },
                {
                    "repo_id": "z-lab/Qwen3-4B-DFlash-b16",
                    "snapshot_sha": "2" * 40,
                    "role": "drafter",
                    "files": [],
                },
            ],
            "datasets": [],
            "environment": {
                "python_version": "3.12",
                "cuda_version": "12.9",
                "driver_version": "575",
                "torch_version": "2.8",
                "triton_version": "3.4",
                "sglang_version": "test",
                "compiler_versions": {},
            },
            "gpus": [],
        }
    )
    lock_path = tmp_path / "screen.lock.json"
    lock.write(lock_path)
    roots_path = tmp_path / "screen.model-roots.json"
    write_model_roots(
        {
            "Qwen/Qwen3-4B": str(target_root),
            "z-lab/Qwen3-4B-DFlash-b16": str(drafter_root),
        },
        roots_path,
    )
    runtime = runtime_implementation_fingerprint(locked_reference={})
    engine = {
        "lockfile_path": str(lock_path.resolve()),
        "model_roots_path": str(roots_path.resolve()),
        "locked_target_revision": "1" * 40,
        "locked_drafter_revision": "2" * 40,
        "pytorch_cuda_alloc_conf": MATCHED_PYTORCH_CUDA_ALLOC_CONF,
        "runtime_implementation_fingerprint": runtime,
        "lr": 1e-4,
        "weight_decay": 1e-2,
    }
    input_runs = [
        _write_run(artifact_root, unit, engine=engine) for unit in source.units
    ]
    coverage = build_coverage(
        source.expected_units(),
        {unit.unit_id: "complete_valid" for unit in source.units},
    )
    coverage_path = tmp_path / "analysis" / "coverage.json"
    coverage.write(coverage_path)
    static_root = tmp_path / "analysis" / "vs-static"
    tts_root = tmp_path / "analysis" / "vs-tts"
    _write_analysis(
        static_root,
        baseline="static",
        source=source,
        input_runs=input_runs,
    )
    _write_analysis(
        tts_root,
        baseline="tts",
        source=source,
        input_runs=input_runs,
    )
    dataset = tmp_path / "dataset-preflight.json"
    dataset.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    dataset_sidecar = _write_sidecar(dataset)
    execution = artifact_root / "EXECUTION_COMPLETE.json"
    _write_receipt(
        execution,
        status="execution_complete",
        scope="candidate_stride_screen_no_claim",
        evidence=[
            QUEUE,
            SELECTOR,
            source_path,
            Path(str(source_path) + ".sha256"),
            lock_path,
            Path(str(lock_path) + ".sha256"),
            roots_path,
            Path(str(roots_path) + ".sha256"),
            dataset,
            dataset_sidecar,
            coverage_path,
            Path(str(coverage_path) + ".sha256"),
            static_root / "analysis-hashes.json",
            tts_root / "analysis-hashes.json",
        ],
    )
    selection = tmp_path / "analysis" / "stride-selection.json"
    selector.select(
        manifest_path=source_path,
        coverage_path=coverage_path,
        vs_static_root=static_root,
        vs_tts_root=tts_root,
        output_path=selection,
    )
    terminal = artifact_root / "CANDIDATE_SCREEN_SELECTED.json"
    _write_receipt(
        terminal,
        status="candidate_screen_selected",
        scope="candidate_screen_only_no_claim",
        evidence=[
            execution,
            Path(str(execution) + ".sha256"),
            selection,
            Path(str(selection) + ".sha256"),
        ],
    )
    return {
        "builder": builder,
        "source": source_path,
        "lockfile": lock_path,
        "model_roots": roots_path,
        "selection": selection,
        "terminal": terminal,
    }


def test_pure_builder_binds_allocator_lock_runtime_and_exact_mirror():
    source = p5_priority_dflash_stride_screen_manifest()
    result = build_matched_controller_manifests(
        _selection(source), source, bindings=_bindings(source)
    )

    assert result.update_stride == 8
    assert result.identity["schema_version"] == 2
    assert result.identity["pytorch_cuda_alloc_conf"] == (
        MATCHED_PYTORCH_CUDA_ALLOC_CONF
    )
    assert result.identity["source_unit_ids"]["static"]
    assert result.identity["bindings"]["lockfile_sha256"] == "5" * 64
    assert result.trace.lockfile_sha256 == "5" * 64
    assert result.trace.engine_params["pytorch_cuda_alloc_conf"] == (
        MATCHED_PYTORCH_CUDA_ALLOC_CONF
    )
    assert result.trace.engine_params["runtime_implementation_fingerprint"] == (
        result.identity["bindings"]["runtime_implementation_fingerprint"]
    )
    assert {unit.method for unit in result.trace.units} == {"naive_async", "tts"}
    assert {unit.stride for unit in result.trace.units} == {8}
    assert sorted(
        (_cell(unit) for unit in result.trace.units if unit.method == "tts"),
        key=lambda row: row["concurrency"],
    ) == sorted(
        (_cell(unit) for unit in result.l3_phase2.units),
        key=lambda row: row["concurrency"],
    )
    assert result.trace.name.endswith(f"b{result.identity['sha256'][:12]}_v1")

    trace_template = p5_priority_dflash_paired_trace_manifest()
    l3_template = p5_priority_dflash_l3_evaluation_manifest()
    assert result.trace.content_sha256() != trace_template.content_sha256()
    assert result.l3_phase2.content_sha256() != l3_template.content_sha256()


def _blocked_selection(source) -> dict:
    selection = _selection(source)
    selection["status"] = "scientifically_blocked"
    selection["objective_screen_pass"] = False
    return selection


def test_oracle_scope_admits_screen_where_l0_is_not_superior():
    source = p5_priority_dflash_stride_screen_manifest()
    blocked = _blocked_selection(source)

    with pytest.raises(ConfigError, match="selection has no winner"):
        build_matched_controller_manifests(
            blocked, source, bindings=_bindings(source)
        )

    result = build_matched_controller_manifests(
        blocked,
        source,
        bindings=_bindings(source),
        allow_l0_not_superior=True,
    )
    assert result.update_stride == 8
    assert {unit.method for unit in result.trace.units} == {"naive_async", "tts"}


def test_oracle_scope_still_rejects_a_passing_or_malformed_screen():
    source = p5_priority_dflash_stride_screen_manifest()

    # A screen that did establish the ordering must not also validate under the
    # widened entry condition; the two statuses stay mutually exclusive.
    with pytest.raises(ConfigError, match="selection has no winner"):
        build_matched_controller_manifests(
            _selection(source),
            source,
            bindings=_bindings(source),
            allow_l0_not_superior=True,
        )

    # Widening the L0 ordering must not widen the objective flag itself.
    inconsistent = _blocked_selection(source)
    inconsistent["objective_screen_pass"] = True
    with pytest.raises(ConfigError, match="objective did not pass"):
        build_matched_controller_manifests(
            inconsistent,
            source,
            bindings=_bindings(source),
            allow_l0_not_superior=True,
        )

    # An unresolved candidate role still fails closed under oracle scope.
    missing_role = _blocked_selection(source)
    missing_role["winners"] = dict(missing_role["winners"])
    missing_role["winners"].pop("l0_best")
    with pytest.raises(ConfigError):
        build_matched_controller_manifests(
            missing_role,
            source,
            bindings=_bindings(source),
            allow_l0_not_superior=True,
        )


def test_source_allocator_missing_or_binding_drift_fails_closed():
    source = p5_priority_dflash_stride_screen_manifest()
    engine = dict(source.engine_params)
    engine.pop("pytorch_cuda_alloc_conf")
    missing = dataclasses.replace(source, engine_params=engine)
    with pytest.raises(ConfigError, match="allocator contract mismatch"):
        build_matched_controller_manifests(
            _selection(source), missing, bindings=_bindings(missing)
        )
    with pytest.raises(ConfigError, match="binding CUDA allocator mismatch"):
        build_matched_controller_manifests(
            _selection(source),
            source,
            bindings=_bindings(source, pytorch_cuda_alloc_conf="drift"),
        )


def test_runtime_enforces_generated_model_roots_binding(tmp_path):
    roots = tmp_path / "model-roots.json"
    roots.write_text("{}", encoding="utf-8")
    binding = {"model_roots_sha256": _sha(roots)}
    _verify_bound_model_roots(roots, binding)
    roots.write_text('{"drift":true}', encoding="utf-8")
    with pytest.raises(LockError, match="immutable manifest binding"):
        _verify_bound_model_roots(roots, binding)


def test_runtime_transition_requires_one_reviewed_exact_hash_pair(monkeypatch):
    source = p5_priority_dflash_stride_screen_manifest()
    runtime_path = "lightcone_spec/orchestration/runtime_config.py"
    stable_path = "sglang/srt/speculative/dflash_worker_v2.py"
    screen = _runtime(
        {
            runtime_path: {"sha256": "1" * 64, "bytes": 10},
            stable_path: {"sha256": "2" * 64, "bytes": 20},
        }
    )
    consumer = _runtime(
        {
            runtime_path: {"sha256": "3" * 64, "bytes": 11},
            stable_path: {"sha256": "2" * 64, "bytes": 20},
        }
    )
    builder = _module("matched_builder_transition_test", SCRIPT)
    with pytest.raises(
        builder.BuildError, match="no reviewed exact hash-pair authorization"
    ):
        builder._runtime_transition(screen, consumer)

    authorization = {
        "id": "test-exact-measurement-pair",
        "changed_files": [runtime_path],
        "added_files": [],
        "removed_files": [],
        "locked_reference_changed": False,
    }
    monkeypatch.setattr(
        controller_module,
        "EXACT_RUNTIME_TRANSITION_AUTHORIZATIONS",
        {(screen["sha256"], consumer["sha256"]): authorization},
    )
    transition = builder._runtime_transition(screen, consumer)
    assert transition == {
        "schema_version": 2,
        "screen_sha256": screen["sha256"],
        "consumer_sha256": consumer["sha256"],
        "equal": False,
        "changed_files": [runtime_path],
        "added_files": [],
        "removed_files": [],
        "locked_reference_changed": False,
        "authorization_id": "test-exact-measurement-pair",
        "authorization_basis": "exact_runtime_fingerprint_pair",
        "screen_measurements_reusable": False,
        "selection_reuse_only": True,
        "scientific_equivalence_claim": False,
        "requires_matched_confirmation": True,
    }
    bindings = _bindings(
        source,
        screen_runtime_implementation_sha256=screen["sha256"],
        consumer_runtime_implementation_sha256=consumer["sha256"],
        screen_runtime_implementation_fingerprint=screen,
        runtime_implementation_fingerprint=consumer,
        runtime_transition=transition,
    )
    build_matched_controller_manifests(
        _selection(source), source, bindings=bindings
    )

    # A second edit to the very same pathname creates a new aggregate pair and
    # is not covered by the reviewed authorization.
    unsafe = _runtime(
        {
            runtime_path: {"sha256": "4" * 64, "bytes": 12},
            stable_path: {"sha256": "2" * 64, "bytes": 20},
        }
    )
    with pytest.raises(
        builder.BuildError, match="no reviewed exact hash-pair authorization"
    ):
        builder._runtime_transition(screen, unsafe)


def test_terminal_builder_writes_cas_closed_bound_manifests(tmp_path):
    chain = _chain(tmp_path)
    builder = chain["builder"]
    output = tmp_path / "matched"
    receipt = builder.build(
        selected_receipt=chain["terminal"],
        lockfile=chain["lockfile"],
        model_roots=chain["model_roots"],
        output_dir=output,
    )

    assert receipt["status"] == "matched_controller_manifests_generated"
    assert receipt["selection"]["semantics_recomputed"] is True
    assert receipt["mirror_contract"]["exact"] is True
    assert receipt["controller_identity"]["bindings"][
        "pytorch_cuda_alloc_conf"
    ] == MATCHED_PYTORCH_CUDA_ALLOC_CONF
    assert set(receipt["artifacts"]) == {
        "TRACE_MATCHED",
        "L3_PHASE2_MATCHED",
        "L3_PHASE2_TTS_REFERENCE",
    }
    assert receipt["controller_identity"]["prompt_windows"] == {
        "phase1_trace": {"offset": 88, "limit": 48, "half_open": [88, 136]},
        "phase2_l3": {"offset": 136, "limit": 48, "half_open": [136, 184]},
    }
    phase2_tts = ExperimentManifest.load(
        receipt["artifacts"]["L3_PHASE2_TTS_REFERENCE"]["path"]
    )
    phase2_l3 = ExperimentManifest.load(
        receipt["artifacts"]["L3_PHASE2_MATCHED"]["path"]
    )
    assert phase2_tts.engine_params["prompt_offset"] == 136
    assert phase2_tts.engine_params["phase2_tts_reference_only"] is True
    assert {unit.method for unit in phase2_tts.units} == {"tts"}
    assert phase2_l3.engine_params["prompt_offset"] == 136
    assert phase2_l3.engine_params["l3_evaluation_only"] is True
    for record in receipt["artifacts"].values():
        path = Path(record["path"])
        sidecar = Path(record["sidecar_path"])
        assert _sha(path) == record["sha256"] == sidecar.read_text().strip()
        loaded = ExperimentManifest.load(path)
        assert loaded.lockfile_sha256 == receipt["locked_inputs"][
            "lockfile_sha256"
        ]
    generation = next(output.glob("matched-controller-manifests-b*.json"))
    assert Path(str(generation) + ".sha256").read_text().strip() == _sha(generation)
    assert receipt == builder.build(
        selected_receipt=chain["terminal"],
        lockfile=chain["lockfile"],
        model_roots=chain["model_roots"],
        output_dir=output,
    )


def test_terminal_builder_recomputes_and_rejects_self_signed_winner(tmp_path):
    chain = _chain(tmp_path)
    selection_path = chain["selection"]
    selection = json.loads(selection_path.read_text())
    selection["winners"]["l0_best"] = selection["candidates"]["l0"][0]
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_sidecar(selection_path)
    terminal_path = chain["terminal"]
    terminal = json.loads(terminal_path.read_text())
    for row in terminal["evidence"]:
        if row["path"] == str(selection_path.resolve()):
            row["sha256"] = _sha(selection_path)
        if row["path"] == str(Path(str(selection_path) + ".sha256").resolve()):
            row["sha256"] = _sha(Path(str(selection_path) + ".sha256"))
    terminal_path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_sidecar(terminal_path)

    with pytest.raises(chain["builder"].BuildError, match="semantic recomputation"):
        chain["builder"].build(
            selected_receipt=terminal_path,
            lockfile=chain["lockfile"],
            model_roots=chain["model_roots"],
            output_dir=tmp_path / "out",
        )


def test_terminal_builder_rejects_unattested_lock_and_output_collision(tmp_path):
    chain = _chain(tmp_path)
    builder = chain["builder"]
    copied_lock = tmp_path / "copy.lock.json"
    copied_lock.write_bytes(chain["lockfile"].read_bytes())
    _write_sidecar(copied_lock)
    with pytest.raises(builder.BuildError, match="omits a required frozen input"):
        builder.build(
            selected_receipt=chain["terminal"],
            lockfile=copied_lock,
            model_roots=chain["model_roots"],
            output_dir=tmp_path / "out-unattested",
        )

    output = tmp_path / "matched"
    receipt = builder.build(
        selected_receipt=chain["terminal"],
        lockfile=chain["lockfile"],
        model_roots=chain["model_roots"],
        output_dir=output,
    )
    trace = Path(receipt["artifacts"]["TRACE_MATCHED"]["path"])
    trace.write_text(trace.read_text() + " ", encoding="utf-8")
    with pytest.raises(builder.BuildError, match="immutable output collision"):
        builder.build(
            selected_receipt=chain["terminal"],
            lockfile=chain["lockfile"],
            model_roots=chain["model_roots"],
            output_dir=output,
        )
