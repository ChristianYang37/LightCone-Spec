"""Fail-closed CLI for the Static/TTS/L0 speed study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lightcone_spec import PINNED_SGLANG_PATCH_COUNT, PINNED_SGLANG_TREE
from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.doctor import format_doctor
from lightcone_spec.experiments.data import (
    DFLASH_MODEL_CONTEXT_LIMIT,
    LongContinuationAdapter,
    load_natural_prompts,
    sample_set_sha256,
)
from lightcone_spec.experiments.evidence import (
    GpuEvidenceAttestation,
    GreedyTargetReference,
    evidence_files_sha256,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_METHODS,
    ONLINE_SPEC_STUDY_METHODS,
    ONLINE_SPEC_TUNING_STAGES,
    OnlineSpecCandidate,
    OnlineSpecGpuAttestation,
    OnlineSpecManifest,
    OnlineSpecSelection,
    OnlineSpecTuningMeasurement,
    compare_onlinespec,
    onlinespec_candidates,
    onlinespec_tuning_stage,
    reduce_onlinespec_tuning_stage,
    select_onlinespec,
    select_onlinespec_heldout_anchor,
    verify_onlinespec_source_checkout,
)
from lightcone_spec.experiments.protocol import (
    DFLASH_LOSS_POSITION_DECAY,
    TUNING_STAGES,
    assert_confirmation_slice_config,
    assert_matched_confirmation_configs,
    confirmation_blocks,
    onlinespec_blocks,
    select_static_load,
    tuning_candidates,
    tuning_stage,
)
from lightcone_spec.experiments.runner import (
    collect_confirmation_performance,
    collect_onlinespec_performance,
    measure_controlled_slice,
    run_confirmation_slice,
    run_greedy_target_reference,
    run_natural_replication_slice,
    run_onlinespec_confirmation_slice,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.selection import (
    CandidateMeasurement,
    SelectionArtifact,
    SliceMeasurement,
    reduce_tuning_stage,
    select_heldout_anchor,
    select_shared_config,
)
from lightcone_spec.experiments.statistics import evaluate_speed_gate
from lightcone_spec.locking import ModelLock, prepare_models, resolve_model_lock
from lightcone_spec.orchestration import (
    SpeedStudyManifest,
    render_onlinespec_runtime_plan,
    render_onlinespec_tuning_runtime_plan,
    render_replication_runtime_plan,
    render_runtime_plan,
    render_static_load_runtime_plan,
    render_tuning_runtime_plan,
)
from lightcone_spec.sglang_bridge import (
    SGLangHTTPClient,
    sglang_adaptation_sha256,
    verify_patched_checkout,
)


def _write_json(path: str | Path, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != body:
        raise ValueError(f"refusing to overwrite immutable artifact {output}")
    output.write_text(body, encoding="utf-8")
    digest = _canonical_sha256(value)
    sidecar = Path(f"{output}.sha256")
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() != digest:
        raise ValueError(f"artifact sidecar does not match {output}")
    sidecar.write_text(digest + "\n", encoding="utf-8")


def _canonical_sha256(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_bound_json(path: str | Path) -> object:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    sidecar = Path(f"{source}.sha256")
    if not sidecar.is_file() or sidecar.read_text(
        encoding="utf-8"
    ).strip() != _canonical_sha256(value):
        raise ValueError(f"JSON artifact sidecar is missing or invalid: {source}")
    return value


def _load_bound_run_config(path: str | Path) -> RunConfig:
    source = Path(path)
    config = load_run_config(source)
    sidecar = Path(f"{source}.sha256")
    if not sidecar.is_file() or sidecar.read_text(
        encoding="utf-8"
    ).strip() != run_config_sha256(config):
        raise ValueError(f"run-config sidecar is missing or invalid: {source}")
    return config


def _static_load_rows(
    value: object,
    *,
    manifest: SpeedStudyManifest,
) -> list[dict]:
    if not isinstance(value, dict):
        raise TypeError("Static load screen must be a schema-v2 terminal artifact")
    expected = {
        "schema_version": 2,
        "phase": "static_load_screen",
        "manifest_sha256": manifest.sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "window_sha256": manifest.controlled_window_hashes["load"],
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("Static load-screen artifact identity mismatch")
    model_lock_sha256 = value.get("model_lock_sha256")
    if not _is_lower_sha256(model_lock_sha256):
        raise ValueError("Static load-screen model-lock identity is invalid")
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("Static load-screen artifact lacks measurement rows")
    return rows


def _formal_table_metadata(
    *,
    manifest: SpeedStudyManifest,
    selection: SelectionArtifact,
    model_lock: ModelLock,
    config_sha256: dict[str, str],
    source_evidence_sha256: str,
    target_reference_sha256: str,
) -> dict[bytes, bytes]:
    return {
        b"lightcone_schema_version": b"2",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": model_lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": (
            manifest.sampling_profile_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_config_set_sha256": _canonical_sha256(config_sha256).encode(),
        b"lightcone_source_evidence_sha256": source_evidence_sha256.encode(),
        b"lightcone_target_reference_sha256": target_reference_sha256.encode(),
    }


def _load_formal_table(
    path: str | Path,
    *,
    manifest: SpeedStudyManifest,
    selection: SelectionArtifact,
    model_lock: ModelLock,
    target_reference: GreedyTargetReference,
) -> pa.Table:
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    expected = {
        b"lightcone_schema_version": b"2",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": model_lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": (
            manifest.sampling_profile_sha256.encode()
        ),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("formal speed table identity metadata mismatch")
    for key in (
        b"lightcone_config_set_sha256",
        b"lightcone_source_evidence_sha256",
    ):
        value = metadata.get(key, b"").decode(errors="ignore")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("formal speed table evidence metadata is invalid")
    return table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lightcone-spec")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--path", default=".")

    validate = commands.add_parser("validate-config")
    validate.add_argument("config")

    build = commands.add_parser("build-speed-study")
    build.add_argument("--output", required=True)

    build_online = commands.add_parser("build-onlinespec-study")
    build_online.add_argument("--output", required=True)

    verify_online_source = commands.add_parser("verify-onlinespec-source")
    verify_online_source.add_argument("--checkout", required=True)
    verify_online_source.add_argument("--audit", required=True)
    verify_online_source.add_argument("--output", required=True)

    list_online = commands.add_parser("list-onlinespec-candidates")
    list_online.add_argument("--output", required=True)

    select_online = commands.add_parser("select-onlinespec-config")
    select_online.add_argument("--measurements", required=True)
    select_online.add_argument("--manifest", required=True)
    select_online.add_argument("--model-lock", required=True)
    select_online.add_argument("--sampling-profile", required=True)
    select_online.add_argument("--core-selection", required=True)
    select_online.add_argument("--output", required=True)

    select_online_anchor = commands.add_parser("select-onlinespec-anchor-config")
    select_online_anchor.add_argument("--measurements", nargs=4, required=True)
    select_online_anchor.add_argument("--candidate-ids", nargs=3, required=True)
    select_online_anchor.add_argument("--manifest", required=True)
    select_online_anchor.add_argument("--model-lock", required=True)
    select_online_anchor.add_argument("--sampling-profile", required=True)
    select_online_anchor.add_argument("--core-selection", required=True)
    select_online_anchor.add_argument("--output", required=True)

    lock = commands.add_parser("lock-models")
    lock.add_argument("--output", required=True)
    lock.add_argument("models", nargs="+")

    prepare = commands.add_parser("prepare-models")
    prepare.add_argument("--lockfile", required=True)
    prepare.add_argument("--model-cache", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--offline", action="store_true")

    select = commands.add_parser("select-speed-config")
    select.add_argument("--measurements", required=True)
    select.add_argument("--static-load-screen", required=True)
    select.add_argument("--manifest", required=True)
    select.add_argument("--model-lock", required=True)
    select.add_argument("--sampling-profile", required=True)
    select.add_argument("--output", required=True)

    select_anchor = commands.add_parser("select-anchor-config")
    select_anchor.add_argument("--measurements", nargs=3, required=True)
    select_anchor.add_argument("--candidate-id", required=True)
    select_anchor.add_argument("--static-load-screen", required=True)
    select_anchor.add_argument("--manifest", required=True)
    select_anchor.add_argument("--model-lock", required=True)
    select_anchor.add_argument("--sampling-profile", required=True)
    select_anchor.add_argument("--output", required=True)

    render = commands.add_parser("render-runtime")
    render.add_argument("--selection", required=True)
    render.add_argument("--model-lock", required=True)
    render.add_argument("--model-roots", required=True)
    render.add_argument("--sglang-checkout", required=True)
    render.add_argument("--sampling-profile", required=True)
    render.add_argument("--adaptation-group-id", required=True)
    render.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render.add_argument("--mem-fraction-static", type=float, required=True)
    render.add_argument("--output-root", required=True)
    render.add_argument("--host", default="127.0.0.1")
    render.add_argument("--first-port", type=int, default=30000)

    render_online = commands.add_parser("render-onlinespec-runtime")
    render_online.add_argument("--selection", required=True)
    render_online.add_argument("--model-lock", required=True)
    render_online.add_argument("--model-roots", required=True)
    render_online.add_argument("--sglang-checkout", required=True)
    render_online.add_argument("--sampling-profile", required=True)
    render_online.add_argument("--adaptation-group-id", required=True)
    render_online.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_online.add_argument("--mem-fraction-static", type=float, required=True)
    render_online.add_argument("--output-root", required=True)
    render_online.add_argument("--host", default="127.0.0.1")
    render_online.add_argument("--first-port", type=int, default=30000)

    render_online_tune = commands.add_parser("render-onlinespec-tuning-runtime")
    render_online_tune.add_argument("--candidate-id", required=True)
    render_online_tune.add_argument("--concurrency", type=int, required=True)
    render_online_tune.add_argument("--model-lock", required=True)
    render_online_tune.add_argument("--model-roots", required=True)
    render_online_tune.add_argument("--sglang-checkout", required=True)
    render_online_tune.add_argument("--sampling-profile", required=True)
    render_online_tune.add_argument("--adaptation-group-id", required=True)
    render_online_tune.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_online_tune.add_argument("--mem-fraction-static", type=float, required=True)
    render_online_tune.add_argument("--output-root", required=True)
    render_online_tune.add_argument("--host", default="127.0.0.1")
    render_online_tune.add_argument("--first-port", type=int, default=30000)

    render_static = commands.add_parser("render-static-load-runtime")
    render_static.add_argument("--concurrency", type=int, required=True)
    render_static.add_argument("--model-lock", required=True)
    render_static.add_argument("--model-roots", required=True)
    render_static.add_argument("--sglang-checkout", required=True)
    render_static.add_argument("--sampling-profile", required=True)
    render_static.add_argument("--mem-fraction-static", type=float, required=True)
    render_static.add_argument("--output-root", required=True)
    render_static.add_argument("--host", default="127.0.0.1")
    render_static.add_argument("--first-port", type=int, default=30000)

    render_tuning = commands.add_parser("render-tuning-runtime")
    render_tuning.add_argument("--candidate-id", required=True)
    render_tuning.add_argument("--concurrency", type=int, required=True)
    render_tuning.add_argument("--model-lock", required=True)
    render_tuning.add_argument("--model-roots", required=True)
    render_tuning.add_argument("--sglang-checkout", required=True)
    render_tuning.add_argument("--sampling-profile", required=True)
    render_tuning.add_argument("--adaptation-group-id", required=True)
    render_tuning.add_argument("--adaptation-reserve-mb", type=int, required=True)
    render_tuning.add_argument("--mem-fraction-static", type=float, required=True)
    render_tuning.add_argument("--output-root", required=True)
    render_tuning.add_argument("--host", default="127.0.0.1")
    render_tuning.add_argument("--first-port", type=int, default=30000)

    replication = commands.add_parser("render-replication-runtime")
    replication.add_argument("--phase", choices=("natural", "profile"), required=True)
    replication.add_argument("--selection", required=True)
    replication.add_argument("--model-lock", required=True)
    replication.add_argument("--model-roots", required=True)
    replication.add_argument("--sglang-checkout", required=True)
    replication.add_argument("--sampling-profile", required=True)
    replication.add_argument("--adaptation-group-id", required=True)
    replication.add_argument("--adaptation-reserve-mb", type=int, required=True)
    replication.add_argument("--mem-fraction-static", type=float, required=True)
    replication.add_argument("--output-root", required=True)
    replication.add_argument("--host", default="127.0.0.1")
    replication.add_argument("--first-port", type=int, default=30000)

    candidates = commands.add_parser("list-tuning-candidates")
    candidates.add_argument("--output", required=True)

    controlled = commands.add_parser("run-controlled-slice")
    controlled.add_argument("--manifest", required=True)
    controlled.add_argument("--model-lock", required=True)
    controlled.add_argument("--sampling-profile", required=True)
    controlled.add_argument("--config", required=True)
    controlled.add_argument("--url", required=True)
    controlled.add_argument("--phase", choices=("static-load", "tune"), required=True)
    controlled.add_argument(
        "--method", choices=("static", "tts", "naive_async"), required=True
    )
    controlled.add_argument("--candidate-id")
    controlled.add_argument("--stage", type=int, default=-1)
    controlled.add_argument("--concurrency", type=int, required=True)
    controlled.add_argument("--output", required=True)
    controlled.add_argument("--no-warmup", action="store_true")

    controlled_online = commands.add_parser("run-onlinespec-tuning-slice")
    controlled_online.add_argument("--manifest", required=True)
    controlled_online.add_argument("--model-lock", required=True)
    controlled_online.add_argument("--sampling-profile", required=True)
    controlled_online.add_argument("--config", required=True)
    controlled_online.add_argument("--url", required=True)
    controlled_online.add_argument(
        "--method", choices=ONLINE_SPEC_STUDY_METHODS, required=True
    )
    controlled_online.add_argument("--candidate-id")
    controlled_online.add_argument("--stage", type=int, required=True)
    controlled_online.add_argument("--concurrency", type=int, required=True)
    controlled_online.add_argument("--output", required=True)
    controlled_online.add_argument("--no-warmup", action="store_true")

    natural = commands.add_parser("run-natural-slice")
    natural.add_argument("--manifest", required=True)
    natural.add_argument("--selection", required=True)
    natural.add_argument("--model-lock", required=True)
    natural.add_argument("--sampling-profile", required=True)
    natural.add_argument("--config", required=True)
    natural.add_argument("--url", required=True)
    natural.add_argument(
        "--method", choices=("static", "tts", "naive_async"), required=True
    )
    natural.add_argument(
        "--dataset", choices=("livecodebench", "math500"), required=True
    )
    natural.add_argument("--dataset-revision", required=True)
    natural.add_argument("--split", required=True)
    natural.add_argument("--output-root", required=True)
    natural.add_argument("--no-warmup", action="store_true")

    profiler = commands.add_parser("build-profiler-plan")
    profiler.add_argument("--launch-plan", required=True)
    profiler.add_argument(
        "--method", choices=("static", "tts", "naive_async"), required=True
    )
    profiler.add_argument("--trace-root", required=True)
    profiler.add_argument("--output", required=True)
    profiler.add_argument("workload_argv", nargs=argparse.REMAINDER)

    load_collect = commands.add_parser("collect-static-load-screen")
    load_collect.add_argument("--manifest", required=True)
    load_collect.add_argument("--measurements", nargs="+", required=True)
    load_collect.add_argument("--output", required=True)

    advance = commands.add_parser("advance-tuning-stage")
    advance.add_argument("--manifest", required=True)
    advance.add_argument("--stage", type=int, required=True)
    advance.add_argument("--measurements", nargs="+", required=True)
    advance.add_argument("--active-set")
    advance.add_argument("--output", required=True)

    advance_online = commands.add_parser("advance-onlinespec-tuning-stage")
    advance_online.add_argument("--manifest", required=True)
    advance_online.add_argument("--stage", type=int, required=True)
    advance_online.add_argument("--measurements", nargs="+", required=True)
    advance_online.add_argument("--active-set")
    advance_online.add_argument("--output", required=True)

    run = commands.add_parser("run-confirmation")
    run.add_argument("--manifest", required=True)
    run.add_argument("--selection", required=True)
    run.add_argument("--model-lock", required=True)
    run.add_argument("--sampling-profile", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--url", required=True)
    run.add_argument(
        "--method", choices=("static", "tts", "naive_async"), required=True
    )
    run.add_argument("--block", type=int, required=True)
    run.add_argument("--adaptation-group-id", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--no-warmup", action="store_true")

    run_online = commands.add_parser("run-onlinespec-confirmation")
    run_online.add_argument("--manifest", required=True)
    run_online.add_argument("--selection", required=True)
    run_online.add_argument("--model-lock", required=True)
    run_online.add_argument("--sampling-profile", required=True)
    run_online.add_argument("--config", required=True)
    run_online.add_argument("--url", required=True)
    run_online.add_argument(
        "--method", choices=ONLINE_SPEC_STUDY_METHODS, required=True
    )
    run_online.add_argument("--block", type=int, required=True)
    run_online.add_argument("--adaptation-group-id", required=True)
    run_online.add_argument("--output-root", required=True)
    run_online.add_argument("--no-warmup", action="store_true")

    target_reference = commands.add_parser("run-target-reference")
    target_reference.add_argument("--model-lock", required=True)
    target_reference.add_argument("--sampling-profile", required=True)
    target_reference.add_argument("--url", required=True)
    target_reference.add_argument("--concurrency", type=int, required=True)
    target_reference.add_argument("--doctor-json", required=True)
    target_reference.add_argument("--output", required=True)
    target_reference.add_argument("--no-warmup", action="store_true")

    collect = commands.add_parser("collect-speed-study")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--selection", required=True)
    collect.add_argument("--model-lock", required=True)
    collect.add_argument("--static-config", required=True)
    collect.add_argument("--tts-config", required=True)
    collect.add_argument("--l0-config", required=True)
    collect.add_argument("--evidence-root", required=True)
    collect.add_argument("--target-reference", required=True)
    collect.add_argument("--output", required=True)

    collect_online = commands.add_parser("collect-onlinespec-study")
    collect_online.add_argument("--manifest", required=True)
    collect_online.add_argument("--selection", required=True)
    collect_online.add_argument("--model-lock", required=True)
    collect_online.add_argument("--static-config", required=True)
    collect_online.add_argument("--ogd-config", required=True)
    collect_online.add_argument("--opt-config", required=True)
    collect_online.add_argument("--ens-config", required=True)
    collect_online.add_argument("--evidence-root", required=True)
    collect_online.add_argument("--target-reference", required=True)
    collect_online.add_argument("--output", required=True)

    queue = commands.add_parser("build-confirmation-queue")
    queue.add_argument("--manifest", required=True)
    queue.add_argument("--selection", required=True)
    queue.add_argument("--model-lock", required=True)
    queue.add_argument("--sampling-profile", required=True)
    queue.add_argument("--launch-plan", required=True)
    queue.add_argument("--evidence-root", required=True)
    queue.add_argument("--output", required=True)

    queue_online = commands.add_parser("build-onlinespec-queue")
    queue_online.add_argument("--manifest", required=True)
    queue_online.add_argument("--selection", required=True)
    queue_online.add_argument("--model-lock", required=True)
    queue_online.add_argument("--sampling-profile", required=True)
    queue_online.add_argument("--launch-plan", required=True)
    queue_online.add_argument("--evidence-root", required=True)
    queue_online.add_argument("--output", required=True)

    attest = commands.add_parser("attest-speed-study")
    attest.add_argument("--manifest", required=True)
    attest.add_argument("--selection", required=True)
    attest.add_argument("--model-lock", required=True)
    attest.add_argument("--performance", required=True)
    attest.add_argument("--target-reference", required=True)
    attest.add_argument("--doctor-json", required=True)
    attest.add_argument("--output", required=True)

    attest_online = commands.add_parser("attest-onlinespec-study")
    attest_online.add_argument("--manifest", required=True)
    attest_online.add_argument("--selection", required=True)
    attest_online.add_argument("--model-lock", required=True)
    attest_online.add_argument("--performance", required=True)
    attest_online.add_argument("--target-reference", required=True)
    attest_online.add_argument("--doctor-json", required=True)
    attest_online.add_argument("--output", required=True)

    analyze = commands.add_parser("analyze-speed-study")
    analyze.add_argument("--performance", required=True)
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--selection", required=True)
    analyze.add_argument("--model-lock", required=True)
    analyze.add_argument("--target-reference", required=True)
    analyze.add_argument("--attestation")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--bootstrap-seed", type=int, default=0)

    analyze_online = commands.add_parser("analyze-onlinespec-study")
    analyze_online.add_argument("--performance", required=True)
    analyze_online.add_argument("--manifest", required=True)
    analyze_online.add_argument("--selection", required=True)
    analyze_online.add_argument("--model-lock", required=True)
    analyze_online.add_argument("--target-reference", required=True)
    analyze_online.add_argument("--attestation")
    analyze_online.add_argument("--output", required=True)
    analyze_online.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def _select(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    tuning_artifact = _load_bound_json(args.measurements)
    load_artifact = _load_bound_json(args.static_load_screen)
    load_rows = _static_load_rows(load_artifact, manifest=manifest)
    measurements_value = tuning_artifact
    if isinstance(tuning_artifact, dict):
        if (
            tuning_artifact.get("schema_version") != 2
            or tuning_artifact.get("phase") != "shared_config_tuning"
            or tuning_artifact.get("manifest_sha256") != manifest.sha256
            or tuning_artifact.get("stage") != len(TUNING_STAGES) - 1
            or tuning_artifact.get("next_stage") is not None
            or not _is_lower_sha256(tuning_artifact.get("prior_stage_sha256"))
        ):
            raise ValueError("selection requires the terminal tuning-stage artifact")
        measurements_value = tuning_artifact.get("candidate_measurements")
    else:
        raise TypeError("selection requires a terminal tuning-stage artifact")
    if not isinstance(measurements_value, list):
        raise TypeError("terminal tuning measurements must be a JSON array")
    lock = ModelLock.load(args.model_lock)
    selected_concurrency = select_static_load(
        load_rows,
        required_context_limit=manifest.safe_context_limit,
    )
    if (
        tuning_artifact.get("model_lock_sha256") != lock.sha256
        or load_artifact.get("model_lock_sha256") != lock.sha256
    ):
        raise ValueError("selection inputs belong to a different model lock")
    if (
        tuning_artifact.get("sampling_profile_sha256") != sampling.sha256
        or tuning_artifact.get("window_sha256")
        != manifest.controlled_window_hashes["tune"]
        or tuning_artifact.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
        or tuning_artifact.get("concurrency") != selected_concurrency
    ):
        raise ValueError(
            "terminal tuning artifact is not bound to this study and selected load"
        )
    measurements = [CandidateMeasurement(**row) for row in measurements_value]
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    artifact = select_shared_config(
        measurements,
        candidates=grid,
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest.sha256,
        sampling_profile_sha256=sampling.sha256,
        tuning_grid_sha256=manifest.tuning_grid_sha256,
        load_screen_sha256=_canonical_sha256(load_artifact),
        tuning_window_sha256=LongContinuationAdapter().window_sha256("tune"),
        model_lock_sha256=lock.sha256,
        tuning_evidence_sha256=_canonical_sha256(tuning_artifact),
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _select_anchor(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    lock = ModelLock.load(args.model_lock)
    load_artifact = _load_bound_json(args.static_load_screen)
    load_rows = _static_load_rows(load_artifact, manifest=manifest)
    selected_concurrency = select_static_load(
        load_rows,
        required_context_limit=manifest.safe_context_limit,
    )
    if load_artifact.get("model_lock_sha256") != lock.sha256:
        raise ValueError("Static load screen belongs to a different model lock")
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("anchor candidate is outside the registered tuning grid")
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    expected_count, expected_context = tuning_stage(len(TUNING_STAGES) - 1)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.model_lock_sha256 != lock.sha256
        or row.sampling_profile_sha256 != sampling.sha256
        or row.window_sha256 != expected_window
        or row.prompt_count != expected_count
        or row.context_limit != expected_context
        for row in measurements
    ):
        raise ValueError("anchor measurement identity is not terminal tuning evidence")
    evidence = _canonical_sha256(
        {
            "selection_protocol": "heldout_anchor",
            "candidate_id": candidate.candidate_id,
            "measurement_sha256": sorted(row.sha256 for row in measurements),
        }
    )
    artifact = select_heldout_anchor(
        measurements,
        candidate=candidate,
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest.sha256,
        sampling_profile_sha256=sampling.sha256,
        tuning_grid_sha256=manifest.tuning_grid_sha256,
        load_screen_sha256=_canonical_sha256(load_artifact),
        tuning_window_sha256=manifest.controlled_window_hashes["tune"],
        model_lock_sha256=lock.sha256,
        tuning_evidence_sha256=evidence,
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _select_onlinespec(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    core_selection = SelectionArtifact.load(args.core_selection)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC manifest uses another sampling profile")
    if (
        core_selection.manifest_sha256 != SpeedStudyManifest.default().sha256
        or core_selection.model_lock_sha256 != lock.sha256
        or core_selection.sampling_profile_sha256 != sampling.sha256
    ):
        raise ValueError(
            "OnlineSPEC selection requires the registered core Static load"
        )
    value = _load_bound_json(args.measurements)
    if (
        value.get("schema_version") != 2
        or value.get("phase") != "onlinespec_tuning"
        or value.get("manifest_sha256") != manifest.sha256
        or value.get("model_lock_sha256") != lock.sha256
        or value.get("sampling_profile_sha256") != sampling.sha256
        or value.get("window_sha256") != manifest.tuning_window_sha256
        or value.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
        or value.get("stage") != len(ONLINE_SPEC_TUNING_STAGES) - 1
        or value.get("next_stage") is not None
        or not _is_lower_sha256(value.get("prior_stage_sha256"))
        or value.get("concurrency") != core_selection.selected_concurrency
    ):
        raise ValueError("OnlineSPEC tuning artifact identity mismatch")
    raw_rows = value.get("measurements")
    if not isinstance(raw_rows, list):
        raise TypeError("OnlineSPEC tuning artifact lacks measurements")
    candidates = {
        candidate.candidate_id: candidate for candidate in onlinespec_candidates()
    }
    selection = select_onlinespec(
        [OnlineSpecTuningMeasurement(**row) for row in raw_rows],
        candidates=candidates,
        selected_concurrency=core_selection.selected_concurrency,
        manifest_sha256=manifest.sha256,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        reference_core_selection_sha256=core_selection.sha256,
        tuning_evidence_sha256=_canonical_sha256(value),
    )
    selection.write(args.output)
    print(selection.sha256)
    return 0


def _select_onlinespec_anchor(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    core_selection = SelectionArtifact.load(args.core_selection)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC manifest uses another sampling profile")
    if (
        core_selection.manifest_sha256 != SpeedStudyManifest.default().sha256
        or core_selection.model_lock_sha256 != lock.sha256
        or core_selection.sampling_profile_sha256 != sampling.sha256
    ):
        raise ValueError(
            "OnlineSPEC selection requires the registered core Static load"
        )
    candidate_ids = tuple(args.candidate_ids)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("OnlineSPEC anchor candidate identities must be unique")
    registered = {
        candidate.candidate_id: candidate for candidate in onlinespec_candidates()
    }
    try:
        candidates = {
            candidate_id: registered[candidate_id] for candidate_id in candidate_ids
        }
    except KeyError as exc:
        raise ValueError(
            "OnlineSPEC anchor is outside the registered tuning grid"
        ) from exc
    if {candidate.method for candidate in candidates.values()} != set(
        ONLINE_SPEC_METHODS
    ):
        raise ValueError("OnlineSPEC anchor requires one candidate per learner")
    measurements = tuple(SliceMeasurement.load(path) for path in args.measurements)
    expected_count, expected_context = onlinespec_tuning_stage(
        len(ONLINE_SPEC_TUNING_STAGES) - 1
    )
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.model_lock_sha256 != lock.sha256
        or row.sampling_profile_sha256 != sampling.sha256
        or row.window_sha256 != expected_window
        or row.prompt_count != expected_count
        or row.context_limit != expected_context
        for row in measurements
    ):
        raise ValueError(
            "OnlineSPEC anchor measurement identity is not terminal tuning evidence"
        )
    evidence = _canonical_sha256(
        {
            "selection_protocol": "heldout_anchor",
            "candidate_ids": sorted(candidate_ids),
            "measurement_sha256": sorted(row.sha256 for row in measurements),
        }
    )
    selection = select_onlinespec_heldout_anchor(
        measurements,
        candidates=candidates,
        selected_concurrency=core_selection.selected_concurrency,
        manifest_sha256=manifest.sha256,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        reference_core_selection_sha256=core_selection.sha256,
        tuning_evidence_sha256=evidence,
    )
    selection.write(args.output)
    print(selection.sha256)
    return 0


def _assert_onlinespec_study(
    manifest: OnlineSpecManifest,
    selection: OnlineSpecSelection,
    lock: ModelLock,
    sampling: SamplingProfile | None = None,
) -> None:
    if (
        selection.manifest_sha256 != manifest.sha256
        or selection.model_lock_sha256 != lock.sha256
        or selection.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or (
            sampling is not None
            and selection.sampling_profile_sha256 != sampling.sha256
        )
    ):
        raise ValueError("OnlineSPEC study identities do not match")


def _study_inputs(
    args: argparse.Namespace,
) -> tuple[SpeedStudyManifest, ModelLock, SamplingProfile]:
    manifest = SpeedStudyManifest.load(args.manifest)
    model_lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("sampling profile belongs to a different manifest")
    return manifest, model_lock, sampling


def _run_controlled_slice(args: argparse.Namespace) -> int:
    manifest, model_lock, sampling = _study_inputs(args)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling,
    )
    if config.method != args.method:
        raise ValueError("run config is bound to another method")
    if config.runtime.max_running_requests < args.concurrency:
        raise ValueError("slice concurrency exceeds the runtime admission limit")
    adapter = LongContinuationAdapter()
    if args.phase == "static-load":
        if args.method != "static" or args.candidate_id is not None or args.stage != -1:
            raise ValueError(
                "Static load screen accepts only an unadapted stage -1 slice"
            )
        if args.concurrency not in manifest.concurrency_grid:
            raise ValueError("Static load concurrency is outside the registered grid")
        samples = adapter.window("load")
        phase = "static_load_screen"
        context_limit = manifest.load_screen_context_limit
        candidate_id = None
    else:
        prompt_count, context_limit = tuning_stage(args.stage)
        samples = adapter.window("tune")[:prompt_count]
        phase = "shared_config_tuning"
        candidate_id = args.candidate_id
        if args.method == "static":
            if candidate_id is not None:
                raise ValueError("Static tuning baseline has no candidate ID")
        else:
            grid = {
                candidate.candidate_id: candidate for candidate in tuning_candidates()
            }
            candidate = grid.get(candidate_id or "")
            if candidate is None:
                raise ValueError("adapted tuning slice has an unknown candidate ID")
            assert_confirmation_slice_config(
                config,
                method=args.method,
                selected_candidate=candidate,
                selected_concurrency=args.concurrency,
            )
    if config.model.max_context_length < context_limit:
        raise ValueError("slice context exceeds the locked model configuration")
    group = (
        "static-preselection"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    measurement = measure_controlled_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        samples=samples,
        phase=phase,
        stage=args.stage,
        candidate_id=candidate_id,
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        model_lock_sha256=model_lock.sha256,
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        sampling_profile=sampling,
        context_limit=context_limit,
        concurrency=args.concurrency,
        adaptation_group_id=group,
        warmup=not args.no_warmup,
    )
    measurement.write(args.output)
    print(measurement.sha256)
    return 0


def _collect_static_load(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    expected_window = LongContinuationAdapter().window_sha256("load")
    if len(measurements) != len(manifest.concurrency_grid):
        raise ValueError("Static load screen has incomplete concurrency coverage")
    rows = []
    model_locks = {row.model_lock_sha256 for row in measurements}
    if len(model_locks) != 1:
        raise ValueError("Static load screen mixes model-lock identities")
    for row in measurements:
        if (
            row.phase != "static_load_screen"
            or row.stage != -1
            or row.method != "static"
            or row.manifest_sha256 != manifest.sha256
            or row.window_sha256 != expected_window
            or row.prompt_count != 8
            or row.context_limit != manifest.load_screen_context_limit
            or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        ):
            raise ValueError("Static load slice identity mismatch")
        rows.append(
            {
                "concurrency": row.concurrency,
                "decode_goodput_tps": row.decode_goodput_tps,
                "itl_p99_ms": row.itl_p99_ms,
                "oom_events": row.oom_events,
                "retractions": row.retractions,
                "kv_token_capacity": row.kv_token_capacity,
                "measurement_sha256": row.sha256,
            }
        )
    rows.sort(key=lambda value: int(value["concurrency"]))
    select_static_load(
        rows,
        required_context_limit=manifest.safe_context_limit,
    )
    artifact = {
        "schema_version": 2,
        "phase": "static_load_screen",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": next(iter(model_locks)),
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "window_sha256": expected_window,
        "rows": rows,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _advance_tuning(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    tuning_stage(args.stage)
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    prior = None
    if args.stage == 0:
        if args.active_set:
            raise ValueError("stage zero must start from the complete registered grid")
        active = tuple(sorted(grid))
    else:
        if not args.active_set:
            raise ValueError("later tuning stages require the prior survivor artifact")
        prior = _load_bound_json(args.active_set)
        if (
            not isinstance(prior, dict)
            or prior.get("schema_version") != 2
            or prior.get("phase") != "shared_config_tuning"
            or prior.get("manifest_sha256") != manifest.sha256
            or prior.get("next_stage") != args.stage
            or prior.get("stage") != args.stage - 1
            or not isinstance(prior.get("survivors"), list)
            or prior.get("sampling_profile_sha256") != manifest.sampling_profile_sha256
            or prior.get("window_sha256") != manifest.controlled_window_hashes["tune"]
            or prior.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
            or (args.stage == 1 and prior.get("prior_stage_sha256") is not None)
            or (
                args.stage > 1 and not _is_lower_sha256(prior.get("prior_stage_sha256"))
            )
        ):
            raise ValueError("prior tuning survivor artifact is invalid")
        active = tuple(str(value) for value in prior["survivors"])
    measurements = [SliceMeasurement.load(path) for path in args.measurements]
    model_locks = {row.model_lock_sha256 for row in measurements}
    if len(model_locks) != 1:
        raise ValueError("tuning stage mixes model-lock identities")
    model_lock_sha256 = next(iter(model_locks))
    if prior is not None and prior.get("model_lock_sha256") != model_lock_sha256:
        raise ValueError(
            "tuning stage uses a different model lock than its predecessor"
        )
    expected_count, _ = tuning_stage(args.stage)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:expected_count]
    )
    concurrencies = {row.concurrency for row in measurements}
    if len(concurrencies) != 1:
        raise ValueError("tuning stage mixes runtime concurrency")
    concurrency = next(iter(concurrencies))
    if prior is not None and prior.get("concurrency") != concurrency:
        raise ValueError("tuning stage changes the selected runtime load")
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.window_sha256 != expected_window
        or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        for row in measurements
    ):
        raise ValueError("tuning measurements use another manifest or prompt window")
    survivors, candidate_rows = reduce_tuning_stage(
        measurements,
        candidates=grid,
        active_candidate_ids=active,
        stage=args.stage,
    )
    next_stage = args.stage + 1 if args.stage + 1 < len(TUNING_STAGES) else None
    artifact = {
        "schema_version": 2,
        "phase": "shared_config_tuning",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": model_lock_sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "window_sha256": manifest.controlled_window_hashes["tune"],
        "tuning_grid_sha256": manifest.tuning_grid_sha256,
        "concurrency": concurrency,
        "stage": args.stage,
        "next_stage": next_stage,
        "prior_stage_sha256": (None if prior is None else _canonical_sha256(prior)),
        "active_candidates": list(active),
        "survivors": list(survivors),
        "measurement_sha256": sorted(row.sha256 for row in measurements),
        "candidate_measurements": [asdict(row) for row in candidate_rows],
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _list_tuning_candidates(args: argparse.Namespace) -> int:
    rows = [
        {**asdict(candidate), "candidate_id": candidate.candidate_id}
        for candidate in tuning_candidates()
    ]
    _write_json(args.output, rows)
    print(_canonical_sha256(rows))
    return 0


def _list_onlinespec_candidates(args: argparse.Namespace) -> int:
    rows = [
        {**asdict(candidate), "candidate_id": candidate.candidate_id}
        for candidate in onlinespec_candidates()
    ]
    _write_json(args.output, rows)
    print(_canonical_sha256(rows))
    return 0


def _confirmation_inputs(
    args: argparse.Namespace,
) -> tuple[SpeedStudyManifest, SelectionArtifact, ModelLock, SamplingProfile]:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    if selection.tuning_window_sha256 != manifest.controlled_window_hashes["tune"]:
        raise ValueError("selection artifact belongs to a different tuning window")
    sampling_profile = SamplingProfile.load(args.sampling_profile)
    if (
        sampling_profile.sha256 != manifest.sampling_profile_sha256
        or selection.sampling_profile_sha256 != sampling_profile.sha256
        or selection.tuning_grid_sha256 != manifest.tuning_grid_sha256
    ):
        raise ValueError("sampling profile belongs to a different manifest")
    return manifest, selection, model_lock, sampling_profile


def _assert_selection_study(
    selection: SelectionArtifact, manifest: SpeedStudyManifest
) -> None:
    if (
        selection.manifest_sha256 != manifest.sha256
        or selection.tuning_grid_sha256 != manifest.tuning_grid_sha256
        or selection.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or selection.tuning_window_sha256 != manifest.controlled_window_hashes["tune"]
    ):
        raise ValueError("selection artifact belongs to a different speed study")


def _assert_locked_config(
    config: RunConfig,
    *,
    model_lock: ModelLock,
    sampling_profile: SamplingProfile,
) -> None:
    if config.runtime.sampling_profile_sha256 != sampling_profile.sha256:
        raise ValueError("run config does not match the sampling profile")
    locked = {model.model_id: model.revision for model in model_lock.models}
    pair = config.model
    if (
        locked.get(pair.target) != pair.target_revision
        or locked.get(pair.drafter) != pair.drafter_revision
    ):
        raise ValueError("run config does not match the immutable model lock")


def _load_target_reference(
    path: str | Path,
    *,
    model_lock: ModelLock,
    sampling_profile_sha256: str,
    concurrency: int,
) -> GreedyTargetReference:
    revisions = {model.model_id: model.revision for model in model_lock.models}
    target_revision = revisions.get("Qwen/Qwen3-8B")
    if target_revision is None:
        raise ValueError("model lock lacks the formal Qwen3-8B target")
    reference = GreedyTargetReference.load(path)
    reference.verify_study(
        model_lock_sha256=model_lock.sha256,
        target_revision=target_revision,
        sampling_profile_sha256=sampling_profile_sha256,
        window_sha256=LongContinuationAdapter().window_sha256("confirm"),
        concurrency=concurrency,
    )
    return reference


def _load_patched_gpu_doctor(path: str | Path, *, purpose: str) -> dict:
    hardware = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(hardware, dict):
        raise TypeError(f"{purpose} doctor evidence is not an object")
    commands = hardware.get("commands")
    if not isinstance(commands, dict):
        raise TypeError(f"{purpose} doctor commands are malformed")
    nvidia = commands.get("nvidia_smi")
    source_tree = hardware.get("source_tree")
    if not isinstance(nvidia, str) or not nvidia.strip():
        raise ValueError(f"{purpose} requires a successful nvidia-smi report")
    if (
        not isinstance(source_tree, dict)
        or source_tree.get("is_git_checkout") is not True
        or source_tree.get("tree") != PINNED_SGLANG_TREE
        or source_tree.get("dirty") is not False
        or source_tree.get("pinned_ancestor") is not True
        or source_tree.get("patch_commits") != PINNED_SGLANG_PATCH_COUNT
    ):
        raise ValueError(f"{purpose} requires the exact clean patched checkout")
    return hardware


def _run_confirmation(args: argparse.Namespace) -> int:
    manifest, selection, model_lock, sampling_profile = _confirmation_inputs(args)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling_profile,
    )
    assert_confirmation_slice_config(
        config,
        method=args.method,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    if (
        config.adaptation is not None
        and config.adaptation.adaptation_group_id != args.adaptation_group_id
    ):
        raise ValueError("run argument and config adaptation groups differ")
    written = run_confirmation_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        block=args.block,
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=args.adaptation_group_id,
        schedule_seed=manifest.confirmation_schedule_seed,
        sampling_profile=sampling_profile,
        model_pair=manifest.model_pair,
        warmup=not args.no_warmup,
    )
    if not written:
        raise RuntimeError("confirmation slice produced no completed evidence")
    print(f"completed {args.block}/{args.method}: {len(written)} files")
    return 0


def _assert_onlinespec_candidate_config(
    config: RunConfig,
    *,
    method: str,
    candidate: OnlineSpecCandidate | None,
    concurrency: int,
) -> None:
    model = config.model
    runtime = config.runtime
    if (
        model.key != "qwen3_8b_dflash16"
        or model.target != "Qwen/Qwen3-8B"
        or model.drafter != "z-lab/Qwen3-8B-DFlash-b16"
        or model.algorithm != "DFLASH"
        or model.max_context_length != DFLASH_MODEL_CONTEXT_LIMIT
        or model.draft_depth != 15
        or runtime.speculative_num_draft_tokens != 16
        or runtime.telemetry_detail != "headline"
    ):
        raise ValueError("OnlineSPEC run config is outside the registered DFlash study")
    if config.method != method or runtime.max_running_requests != concurrency:
        raise ValueError("OnlineSPEC run config method or load mismatch")
    if method == "static":
        if candidate is not None:
            raise ValueError("OnlineSPEC Static reference has no candidate")
        if config.adaptation is not None or config.online_spec is not None:
            raise ValueError("OnlineSPEC Static reference allocated adaptation state")
        return
    adaptation = config.adaptation
    learner = config.online_spec
    if (
        candidate is None
        or candidate.method != method
        or adaptation is None
        or learner is None
    ):
        raise ValueError("OnlineSPEC run config is incomplete")
    actual = {
        "weight_update_mode": adaptation.weight_update_mode,
        "parameter_scope": adaptation.parameter_scope,
        "learning_rate": adaptation.optimizer.learning_rate,
        "rank": adaptation.rank,
        "stride": adaptation.stride,
        "projection_radius": learner.projection_radius,
        "additional_learning_rates": learner.additional_learning_rates,
        "hedge_learning_rate": learner.hedge_learning_rate,
        "grad_clip": adaptation.optimizer.grad_clip,
    }
    expected = {
        key: value for key, value in asdict(candidate).items() if key not in {"method"}
    }
    if (
        actual != expected
        or adaptation.optimizer.name != "sgd"
        or not math.isclose(
            adaptation.loss_position_decay,
            DFLASH_LOSS_POSITION_DECAY,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("OnlineSPEC run config does not match its tuning selection")


def _assert_onlinespec_config(
    config: RunConfig,
    *,
    method: str,
    selection: OnlineSpecSelection,
) -> None:
    candidate = next(
        (candidate for candidate in selection.selected if candidate.method == method),
        None,
    )
    _assert_onlinespec_candidate_config(
        config,
        method=method,
        candidate=candidate,
        concurrency=selection.selected_concurrency,
    )


def _run_onlinespec_tuning_slice(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.sha256 != manifest.sampling_profile_sha256:
        raise ValueError("OnlineSPEC tuning uses another sampling profile")
    config = _load_bound_run_config(args.config)
    _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
    prompt_count, context_limit = onlinespec_tuning_stage(args.stage)
    samples = LongContinuationAdapter().window("tune")[:prompt_count]
    candidate = None
    if args.method == "static":
        if args.candidate_id is not None:
            raise ValueError("OnlineSPEC Static tuning has no candidate ID")
    else:
        grid = {row.candidate_id: row for row in onlinespec_candidates()}
        candidate = grid.get(args.candidate_id or "")
        if candidate is None:
            raise ValueError("OnlineSPEC tuning candidate is not registered")
    _assert_onlinespec_candidate_config(
        config,
        method=args.method,
        candidate=candidate,
        concurrency=args.concurrency,
    )
    if config.model.max_context_length < context_limit:
        raise ValueError("OnlineSPEC tuning exceeds the locked model context")
    group = (
        "onlinespec-static-tuning"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    measurement = measure_controlled_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        samples=samples,
        phase="onlinespec_tuning",
        stage=args.stage,
        candidate_id=(None if candidate is None else candidate.candidate_id),
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        model_lock_sha256=lock.sha256,
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        sampling_profile=sampling,
        context_limit=context_limit,
        concurrency=args.concurrency,
        adaptation_group_id=group,
        warmup=not args.no_warmup,
    )
    measurement.write(args.output)
    print(measurement.sha256)
    return 0


def _advance_onlinespec_tuning(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    onlinespec_tuning_stage(args.stage)
    grid = {candidate.candidate_id: candidate for candidate in onlinespec_candidates()}
    prior = None
    if args.stage == 0:
        if args.active_set:
            raise ValueError("stage zero starts from the complete OnlineSPEC grid")
        active = tuple(sorted(grid))
    else:
        if not args.active_set:
            raise ValueError("later OnlineSPEC stages require survivor evidence")
        prior = _load_bound_json(args.active_set)
        if (
            prior.get("schema_version") != 2
            or prior.get("phase") != "onlinespec_tuning"
            or prior.get("manifest_sha256") != manifest.sha256
            or prior.get("tuning_grid_sha256") != manifest.tuning_grid_sha256
            or prior.get("window_sha256") != manifest.tuning_window_sha256
            or prior.get("sampling_profile_sha256") != manifest.sampling_profile_sha256
            or prior.get("stage") != args.stage - 1
            or prior.get("next_stage") != args.stage
            or not isinstance(prior.get("survivors"), list)
            or (args.stage == 1 and prior.get("prior_stage_sha256") is not None)
            or (
                args.stage > 1 and not _is_lower_sha256(prior.get("prior_stage_sha256"))
            )
        ):
            raise ValueError("prior OnlineSPEC tuning artifact is invalid")
        active = tuple(str(value) for value in prior["survivors"])
    measurements = tuple(SliceMeasurement.load(path) for path in args.measurements)
    model_locks = {row.model_lock_sha256 for row in measurements}
    concurrencies = {row.concurrency for row in measurements}
    if len(model_locks) != 1 or len(concurrencies) != 1:
        raise ValueError("OnlineSPEC stage mixes model locks or runtime loads")
    model_lock_sha256 = next(iter(model_locks))
    concurrency = next(iter(concurrencies))
    if prior is not None and (
        prior.get("model_lock_sha256") != model_lock_sha256
        or prior.get("concurrency") != concurrency
    ):
        raise ValueError("OnlineSPEC tuning changed its model lock or load")
    prompt_count, _ = onlinespec_tuning_stage(args.stage)
    expected_window = sample_set_sha256(
        LongContinuationAdapter().window("tune")[:prompt_count]
    )
    if any(
        row.manifest_sha256 != manifest.sha256
        or row.sampling_profile_sha256 != manifest.sampling_profile_sha256
        or row.window_sha256 != expected_window
        for row in measurements
    ):
        raise ValueError("OnlineSPEC measurements use another registered input")
    survivors, reduced = reduce_onlinespec_tuning_stage(
        measurements,
        candidates=grid,
        active_candidate_ids=active,
        stage=args.stage,
    )
    next_stage = (
        args.stage + 1
        if args.stage + 1 < len(ONLINE_SPEC_TUNING_STAGES)
        else None
    )
    artifact = {
        "schema_version": 2,
        "phase": "onlinespec_tuning",
        "manifest_sha256": manifest.sha256,
        "model_lock_sha256": model_lock_sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "tuning_grid_sha256": manifest.tuning_grid_sha256,
        "window_sha256": manifest.tuning_window_sha256,
        "measurement_window_sha256": expected_window,
        "concurrency": concurrency,
        "stage": args.stage,
        "next_stage": next_stage,
        "prior_stage_sha256": (None if prior is None else _canonical_sha256(prior)),
        "active_candidates": list(active),
        "survivors": list(survivors),
        "measurement_sha256": sorted(row.sha256 for row in measurements),
        "measurements": [asdict(row) for row in reduced],
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _run_onlinespec_confirmation(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    _assert_onlinespec_study(manifest, selection, lock, sampling)
    config = _load_bound_run_config(args.config)
    _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
    _assert_onlinespec_config(config, method=args.method, selection=selection)
    if (
        config.adaptation is not None
        and config.adaptation.adaptation_group_id != args.adaptation_group_id
    ):
        raise ValueError("OnlineSPEC adaptation group mismatch")
    written = run_onlinespec_confirmation_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        block=args.block,
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=args.adaptation_group_id,
        schedule_seed=manifest.confirmation_schedule_seed,
        sampling_profile=sampling,
        warmup=not args.no_warmup,
    )
    if not written:
        raise RuntimeError("OnlineSPEC slice produced no completed evidence")
    print(f"completed OnlineSPEC {args.block}/{args.method}: {len(written)} files")
    return 0


def _run_target_reference(args: argparse.Namespace) -> int:
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    hardware = _load_patched_gpu_doctor(
        args.doctor_json,
        purpose="target reference",
    )
    revisions = {model.model_id: model.revision for model in lock.models}
    target_revision = revisions.get("Qwen/Qwen3-8B")
    if target_revision is None:
        raise ValueError("model lock lacks the formal Qwen3-8B target")
    artifact = run_greedy_target_reference(
        client=SGLangHTTPClient(args.url),
        model_lock_sha256=lock.sha256,
        target_revision=target_revision,
        hardware_sha256=_canonical_sha256(hardware),
        concurrency=args.concurrency,
        sampling_profile=sampling,
        warmup=not args.no_warmup,
    )
    artifact.write(args.output)
    print(artifact.sha256)
    return 0


def _confirmation_configs(args: argparse.Namespace) -> dict:
    return {
        "static": _load_bound_run_config(args.static_config),
        "tts": _load_bound_run_config(args.tts_config),
        "naive_async": _load_bound_run_config(args.l0_config),
    }


def _concat_evidence_tables(paths: tuple[Path, ...]) -> pa.Table:
    """Concatenate evidence while promoting only all-null inferred columns."""
    tables = [pq.read_table(path) for path in paths]
    if not tables:
        raise ValueError("evidence table set cannot be empty")
    column_names = tables[0].column_names
    if any(table.column_names != column_names for table in tables[1:]):
        raise ValueError("evidence tables have different columns")
    for index, name in enumerate(column_names):
        concrete_types: list[pa.DataType] = []
        for table in tables:
            data_type = table.schema.field(index).type
            if not pa.types.is_null(data_type) and data_type not in concrete_types:
                concrete_types.append(data_type)
        if len(concrete_types) > 1:
            rendered = ", ".join(str(value) for value in concrete_types)
            raise ValueError(
                f"evidence column {name!r} has incompatible types: {rendered}"
            )
    return pa.concat_tables(tables, promote_options="default")


def _collect_speed_study(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    configs = _confirmation_configs(args)
    for config in configs.values():
        if config.runtime.sampling_profile_sha256 != manifest.sampling_profile_sha256:
            raise ValueError("run config does not match the manifest sampling profile")
        locked = {model.model_id: model.revision for model in model_lock.models}
        if (
            locked.get(config.model.target) != config.model.target_revision
            or locked.get(config.model.drafter) != config.model.drafter_revision
        ):
            raise ValueError("run config does not match the immutable model lock")
    assert_matched_confirmation_configs(
        configs,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=model_lock,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    target_revision = next(
        model.revision
        for model in model_lock.models
        if model.model_id == "Qwen/Qwen3-8B"
    )
    performance, source_evidence_sha256 = collect_confirmation_performance(
        evidence_root=args.evidence_root,
        manifest_sha256=manifest.sha256,
        config_sha256={
            method: run_config_sha256(config) for method, config in configs.items()
        },
        concurrency=selection.selected_concurrency,
        target_reference=target_reference,
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        target_revision=target_revision,
    )
    table = _concat_evidence_tables(performance)
    table = table.replace_schema_metadata(
        _formal_table_metadata(
            manifest=manifest,
            selection=selection,
            model_lock=model_lock,
            config_sha256={
                method: run_config_sha256(config) for method, config in configs.items()
            },
            source_evidence_sha256=source_evidence_sha256,
            target_reference_sha256=target_reference.sha256,
        )
    )
    output = Path(args.output)
    if output.exists():
        existing = _load_formal_table(
            output,
            manifest=manifest,
            selection=selection,
            model_lock=model_lock,
            target_reference=target_reference,
        )
        if existing.schema.metadata == table.schema.metadata and existing.equals(table):
            print(output)
            return 0
        raise RuntimeError("existing speed-study table differs from evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, output)
    print(output)
    return 0


def _collect_onlinespec_study(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    _assert_onlinespec_study(manifest, selection, lock)
    configs = {
        "static": _load_bound_run_config(args.static_config),
        "onlinespec_ogd": _load_bound_run_config(args.ogd_config),
        "onlinespec_opt": _load_bound_run_config(args.opt_config),
        "onlinespec_ens": _load_bound_run_config(args.ens_config),
    }
    locked = {model.model_id: model.revision for model in lock.models}
    for method, config in configs.items():
        if config.runtime.sampling_profile_sha256 != selection.sampling_profile_sha256:
            raise ValueError("OnlineSPEC config sampling identity mismatch")
        if (
            locked.get(config.model.target) != config.model.target_revision
            or locked.get(config.model.drafter) != config.model.drafter_revision
        ):
            raise ValueError("OnlineSPEC config model-lock identity mismatch")
        _assert_onlinespec_config(config, method=method, selection=selection)
    config_hashes = {
        method: run_config_sha256(config) for method, config in configs.items()
    }
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=lock,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    target_revision = next(
        model.revision
        for model in lock.models
        if model.model_id == "Qwen/Qwen3-8B"
    )
    performance, evidence_sha256 = collect_onlinespec_performance(
        evidence_root=args.evidence_root,
        manifest_sha256=manifest.sha256,
        config_sha256=config_hashes,
        concurrency=selection.selected_concurrency,
        target_reference=target_reference,
        model_lock_sha256=lock.sha256,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        target_revision=target_revision,
    )
    table = _concat_evidence_tables(performance)
    metadata = {
        b"lightcone_schema_version": b"2",
        b"lightcone_study": b"onlinespec-clean-room-baseline",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": selection.sampling_profile_sha256.encode(),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_config_set_sha256": _canonical_sha256(config_hashes).encode(),
        b"lightcone_source_evidence_sha256": evidence_sha256.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    table = table.replace_schema_metadata(metadata)
    output = Path(args.output)
    if output.exists():
        existing = pq.read_table(output)
        if existing.schema.metadata == metadata and existing.equals(table):
            print(output)
            return 0
        raise RuntimeError("existing OnlineSPEC table differs from evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, output)
    print(output)
    return 0


def _render_runtime(args: argparse.Namespace) -> int:
    selection = SelectionArtifact.load(args.selection)
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    launches = render_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    print(Path(args.output_root).resolve() / "launch-plan.json")
    if len(launches) != 3:
        raise AssertionError("runtime renderer did not create three method slices")
    if len({launch.base_url for launch in launches}) != 1 or not all(
        launch.exclusive_device for launch in launches
    ):
        raise AssertionError("formal method slices must share one exclusive endpoint")
    return 0


def _render_onlinespec_runtime(args: argparse.Namespace) -> int:
    selection = OnlineSpecSelection.load(args.selection)
    launches = render_onlinespec_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=ModelLock.load(args.model_lock),
        model_roots=_load_bound_json(args.model_roots),
        sampling_profile=SamplingProfile.load(args.sampling_profile),
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != list(ONLINE_SPEC_STUDY_METHODS):
        raise AssertionError("OnlineSPEC runtime plan has incomplete method coverage")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_onlinespec_tuning_runtime(args: argparse.Namespace) -> int:
    grid = {candidate.candidate_id: candidate for candidate in onlinespec_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("OnlineSPEC candidate ID is outside the registered grid")
    launches = render_onlinespec_tuning_runtime_plan(
        output_root=args.output_root,
        candidate=candidate,
        concurrency=args.concurrency,
        model_lock=ModelLock.load(args.model_lock),
        model_roots=_load_bound_json(args.model_roots),
        sampling_profile=SamplingProfile.load(args.sampling_profile),
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != ["static", candidate.method]:
        raise AssertionError("OnlineSPEC tuning plan is not paired to Static")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_static_load_runtime(args: argparse.Namespace) -> int:
    if args.concurrency not in {1, 2, 4, 8, 16, 32, 48}:
        raise ValueError("Static concurrency is outside the registered load grid")
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.purpose != "controlled":
        raise ValueError("Static load screen requires the controlled sampling profile")
    launches = render_static_load_runtime_plan(
        output_root=args.output_root,
        concurrency=args.concurrency,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if (
        len(launches) != 1
        or launches[0].method != "static"
        or launches[0].adaptation_config is not None
        or launches[0].telemetry_path is not None
        or "--speculative-adaptation-config" in launches[0].argv
    ):
        raise AssertionError("Static load renderer allocated an adaptation path")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_tuning_runtime(args: argparse.Namespace) -> int:
    grid = {candidate.candidate_id: candidate for candidate in tuning_candidates()}
    candidate = grid.get(args.candidate_id)
    if candidate is None:
        raise ValueError("candidate ID is outside the registered tuning grid")
    if args.concurrency not in {1, 2, 4, 8, 16, 32, 48}:
        raise ValueError("tuning concurrency is outside the registered load grid")
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    launches = render_tuning_runtime_plan(
        output_root=args.output_root,
        candidate=candidate,
        concurrency=args.concurrency,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        host=args.host,
        first_port=args.first_port,
    )
    if [launch.method for launch in launches] != ["tts", "naive_async"] or len(
        {launch.base_url for launch in launches}
    ) != 1:
        raise AssertionError("tuning runtime must contain two adapted slices")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _render_replication_runtime(args: argparse.Namespace) -> int:
    selection = SelectionArtifact.load(args.selection)
    model_lock = ModelLock.load(args.model_lock)
    roots = _load_bound_json(args.model_roots)
    sampling = SamplingProfile.load(args.sampling_profile)
    phase = (
        "natural_task_replication"
        if args.phase == "natural"
        else "independent_profiler"
    )
    if phase == "natural_task_replication" and sampling.purpose != "natural":
        raise ValueError("natural runtime requires an EOS-enabled sampling profile")
    if phase == "independent_profiler" and sampling.purpose != "controlled":
        raise ValueError("profiler runtime requires the controlled sampling profile")
    launches = render_replication_runtime_plan(
        output_root=args.output_root,
        selection=selection,
        model_lock=model_lock,
        model_roots=roots,
        sampling_profile=sampling,
        sglang_checkout=args.sglang_checkout,
        adaptation_group_id=args.adaptation_group_id,
        adaptation_reserve_mb=args.adaptation_reserve_mb,
        mem_fraction_static=args.mem_fraction_static,
        phase=phase,
        host=args.host,
        first_port=args.first_port,
    )
    if len(launches) != 3 or not all(launch.exclusive_device for launch in launches):
        raise AssertionError("replication runtime slices are not exclusive")
    print(Path(args.output_root).resolve() / "launch-plan.json")
    return 0


def _run_natural_slice(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to another model lock")
    sampling = SamplingProfile.load(args.sampling_profile)
    if sampling.purpose != "natural" or sampling.ignore_eos:
        raise ValueError("natural side table requires the EOS-enabled profile")
    config = _load_bound_run_config(args.config)
    _assert_locked_config(
        config,
        model_lock=model_lock,
        sampling_profile=sampling,
    )
    assert_confirmation_slice_config(
        config,
        method=args.method,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    samples = load_natural_prompts(
        args.dataset,
        revision=args.dataset_revision,
        split=args.split,
        limit=32,
    )
    adaptation_group_id = (
        "natural-static"
        if config.adaptation is None
        else config.adaptation.adaptation_group_id
    )
    paths = run_natural_replication_slice(
        client=SGLangHTTPClient(args.url),
        method=args.method,
        dataset_name=args.dataset,
        samples=samples,
        manifest_sha256=manifest.sha256,
        config_sha256=run_config_sha256(config),
        adaptation_config_sha256=sglang_adaptation_sha256(config),
        output_root=args.output_root,
        concurrency=selection.selected_concurrency,
        safe_context_limit=manifest.safe_context_limit,
        adaptation_group_id=adaptation_group_id,
        sampling_profile=sampling,
        model_pair=manifest.model_pair,
        warmup=not args.no_warmup,
    )
    print(f"completed natural {args.dataset}/{args.method}: {len(paths)} files")
    return 0


def _build_profiler_plan(args: argparse.Namespace) -> int:
    source = _load_bound_json(args.launch_plan)
    if (
        source.get("schema_version") != 2
        or source.get("phase") != "independent_profiler"
        or source.get("execution_mode") != "sequential_exclusive_device"
        or source.get("patched_sglang_tree") != PINNED_SGLANG_TREE
    ):
        raise ValueError("profiler requires an independent-profiler launch plan")
    verify_patched_checkout(str(source.get("sglang_checkout", "")))
    servers = source.get("servers")
    if not isinstance(servers, list):
        raise TypeError("profiler launch plan lacks server slices")
    matching = [row for row in servers if row.get("method") == args.method]
    if len(matching) != 1 or matching[0].get("exclusive_device") is not True:
        raise ValueError("profiler method slice is missing or not exclusive")
    workload = list(args.workload_argv)
    if workload and workload[0] == "--":
        workload = workload[1:]
    if not workload or not all(isinstance(value, str) and value for value in workload):
        raise ValueError("profiler plan requires an explicit workload argv after --")
    trace_root = Path(args.trace_root).resolve()
    server_argv = matching[0].get("argv")
    if not isinstance(server_argv, list) or not server_argv:
        raise ValueError("profiler server argv is missing")
    artifact = {
        "schema_version": 2,
        "phase": "independent_profiler",
        "method": args.method,
        "launch_plan_sha256": _canonical_sha256(source),
        "exclusive_device": True,
        "profile_launch_argv": [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--force-overwrite=false",
            "--output",
            str(trace_root / args.method),
            *server_argv,
        ],
        "workload_argv": workload,
        "device_monitor_argv": [
            "nvidia-smi",
            "dmon",
            "-s",
            "puctm",
            "-o",
            "DT",
        ],
        "headline_evidence_forbidden": True,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _build_confirmation_queue(args: argparse.Namespace) -> int:
    manifest, selection, model_lock, sampling = _confirmation_inputs(args)
    plan_path = Path(args.launch_plan).resolve()
    plan = _load_bound_json(plan_path)
    if (
        plan.get("schema_version") != 2
        or plan.get("execution_mode") != "sequential_exclusive_device"
        or plan.get("selection_sha256") != selection.sha256
        or plan.get("model_lock_sha256") != model_lock.sha256
        or plan.get("sampling_profile_sha256") != sampling.sha256
        or plan.get("patched_sglang_tree") != PINNED_SGLANG_TREE
    ):
        raise ValueError("launch plan identity does not match the speed study")
    verify_patched_checkout(str(plan.get("sglang_checkout", "")))
    rows = plan.get("servers")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("launch plan must contain exactly three method slices")
    servers: dict[str, dict] = {}
    configs: dict[str, RunConfig] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("launch plan server entry is not an object")
        method = row.get("method")
        if method not in {"static", "tts", "naive_async"} or method in servers:
            raise ValueError("launch plan has invalid or duplicate methods")
        if row.get("exclusive_device") is not True:
            raise ValueError("every formal slice must own the GPU exclusively")
        argv = row.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(value, str) and value for value in argv
        ):
            raise ValueError("launch argv must be a non-empty string vector")
        config_path = Path(str(row.get("run_config"))).resolve()
        config = _load_bound_run_config(config_path)
        _assert_locked_config(
            config,
            model_lock=model_lock,
            sampling_profile=sampling,
        )
        servers[method] = {**row, "run_config": str(config_path)}
        configs[method] = config
    if len({str(row["base_url"]) for row in servers.values()}) != 1:
        raise ValueError("sequential method slices must reuse one endpoint")
    assert_matched_confirmation_configs(
        configs,
        selected_candidate=selection.candidate,
        selected_concurrency=selection.selected_concurrency,
    )
    adaptation = configs["tts"].adaptation
    if adaptation is None:
        raise AssertionError("TTS launch lacks adaptation identity")
    common = [
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--selection",
        str(Path(args.selection).resolve()),
        "--model-lock",
        str(Path(args.model_lock).resolve()),
        "--sampling-profile",
        str(Path(args.sampling_profile).resolve()),
        "--adaptation-group-id",
        adaptation.adaptation_group_id,
        "--output-root",
        str(Path(args.evidence_root).resolve()),
    ]
    jobs: list[dict] = []
    ordinal = 0
    for block in confirmation_blocks(manifest.confirmation_schedule_seed):
        for method in block.method_order:
            server = servers[method]
            jobs.append(
                {
                    "ordinal": ordinal,
                    "block": block.block,
                    "method": method,
                    "launch_argv": server["argv"],
                    "run_argv": [
                        "lightcone-spec",
                        "run-confirmation",
                        *common,
                        "--config",
                        server["run_config"],
                        "--url",
                        str(server["base_url"]),
                        "--method",
                        method,
                        "--block",
                        str(block.block),
                    ],
                    "requires_clean_server_start": True,
                    "requires_server_exit_after": True,
                }
            )
            ordinal += 1
    artifact = {
        "schema_version": 2,
        "execution_mode": "sequential_exclusive_device",
        "manifest_sha256": manifest.sha256,
        "selection_sha256": selection.sha256,
        "model_lock_sha256": model_lock.sha256,
        "sampling_profile_sha256": sampling.sha256,
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "launch_plan_sha256": _canonical_sha256(plan),
        "schedule_seed": manifest.confirmation_schedule_seed,
        "jobs": jobs,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _build_onlinespec_queue(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    sampling = SamplingProfile.load(args.sampling_profile)
    _assert_onlinespec_study(manifest, selection, lock, sampling)
    plan = _load_bound_json(args.launch_plan)
    if (
        plan.get("schema_version") != 2
        or plan.get("phase") != "onlinespec_paired_confirmation"
        or plan.get("selection_sha256") != selection.sha256
        or plan.get("model_lock_sha256") != lock.sha256
        or plan.get("sampling_profile_sha256") != sampling.sha256
        or plan.get("patched_sglang_tree") != PINNED_SGLANG_TREE
    ):
        raise ValueError("OnlineSPEC launch plan identity mismatch")
    verify_patched_checkout(str(plan.get("sglang_checkout", "")))
    raw_servers = plan.get("servers")
    if not isinstance(raw_servers, list) or len(raw_servers) != 4:
        raise ValueError("OnlineSPEC launch plan requires four method slices")
    servers = {}
    for row in raw_servers:
        if (
            not isinstance(row, dict)
            or row.get("method") not in ONLINE_SPEC_STUDY_METHODS
        ):
            raise ValueError("OnlineSPEC launch plan contains an invalid server")
        method = str(row["method"])
        if method in servers or row.get("exclusive_device") is not True:
            raise ValueError("OnlineSPEC servers must be unique and exclusive")
        config = _load_bound_run_config(row["run_config"])
        _assert_locked_config(config, model_lock=lock, sampling_profile=sampling)
        _assert_onlinespec_config(config, method=method, selection=selection)
        servers[method] = {**row, "config": config}
    if (
        set(servers) != set(ONLINE_SPEC_STUDY_METHODS)
        or len({str(row["base_url"]) for row in servers.values()}) != 1
    ):
        raise ValueError("OnlineSPEC servers do not share one exclusive endpoint")
    group_ids = {
        row["config"].adaptation.adaptation_group_id
        for method, row in servers.items()
        if method != "static"
    }
    if len(group_ids) != 1:
        raise ValueError("OnlineSPEC configs do not share one cohort group")
    common = [
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--selection",
        str(Path(args.selection).resolve()),
        "--model-lock",
        str(Path(args.model_lock).resolve()),
        "--sampling-profile",
        str(Path(args.sampling_profile).resolve()),
        "--adaptation-group-id",
        next(iter(group_ids)),
        "--output-root",
        str(Path(args.evidence_root).resolve()),
    ]
    jobs = []
    ordinal = 0
    for block in onlinespec_blocks(manifest.confirmation_schedule_seed):
        for method in block.method_order:
            server = servers[method]
            jobs.append(
                {
                    "ordinal": ordinal,
                    "block": block.block,
                    "method": method,
                    "launch_argv": server["argv"],
                    "run_argv": [
                        "lightcone-spec",
                        "run-onlinespec-confirmation",
                        *common,
                        "--config",
                        str(Path(server["run_config"]).resolve()),
                        "--url",
                        str(server["base_url"]),
                        "--method",
                        method,
                        "--block",
                        str(block.block),
                    ],
                    "requires_clean_server_start": True,
                    "requires_server_exit_after": True,
                }
            )
            ordinal += 1
    artifact = {
        "schema_version": 2,
        "execution_mode": "sequential_exclusive_device",
        "study": "onlinespec-clean-room-baseline",
        "manifest_sha256": manifest.sha256,
        "selection_sha256": selection.sha256,
        "model_lock_sha256": lock.sha256,
        "sampling_profile_sha256": sampling.sha256,
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "launch_plan_sha256": _canonical_sha256(plan),
        "schedule_seed": manifest.confirmation_schedule_seed,
        "jobs": jobs,
    }
    _write_json(args.output, artifact)
    print(_canonical_sha256(artifact))
    return 0


def _attest(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    revisions = {model.model_id: model.revision for model in model_lock.models}
    target_revision = revisions.get("Qwen/Qwen3-8B")
    drafter_revision = revisions.get("z-lab/Qwen3-8B-DFlash-b16")
    if target_revision is None or drafter_revision is None:
        raise ValueError("model lock lacks the formal Qwen3-8B/DFlash pair")
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=model_lock,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    _load_formal_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        model_lock=model_lock,
        target_reference=target_reference,
    )
    hardware = _load_patched_gpu_doctor(
        args.doctor_json,
        purpose="GPU attestation",
    )
    if target_reference.hardware_sha256 != _canonical_sha256(hardware):
        raise ValueError("target reference belongs to a different GPU report")
    attestation = GpuEvidenceAttestation(
        schema_version=2,
        status="MEASURED",
        manifest_sha256=manifest.sha256,
        selection_sha256=selection.sha256,
        model_lock_sha256=model_lock.sha256,
        performance_sha256=evidence_files_sha256((args.performance,)),
        target_reference_sha256=target_reference.sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        target_revision=target_revision,
        drafter_revision=drafter_revision,
        hardware_sha256=_canonical_sha256(hardware),
        methods=manifest.methods,
        repetitions=manifest.confirmation_repetitions,
        context_start=manifest.formal_context_start,
        context_limit=manifest.safe_context_limit,
    )
    attestation.write(args.output)
    print(attestation.sha256)
    return 0


def _onlinespec_table(
    path: str | Path,
    *,
    manifest: OnlineSpecManifest,
    selection: OnlineSpecSelection,
    lock: ModelLock,
    target_reference: GreedyTargetReference,
) -> pa.Table:
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    expected = {
        b"lightcone_schema_version": b"2",
        b"lightcone_study": b"onlinespec-clean-room-baseline",
        b"lightcone_manifest_sha256": manifest.sha256.encode(),
        b"lightcone_selection_sha256": selection.sha256.encode(),
        b"lightcone_model_lock_sha256": lock.sha256.encode(),
        b"lightcone_sampling_profile_sha256": selection.sampling_profile_sha256.encode(),
        b"lightcone_patched_sglang_tree": PINNED_SGLANG_TREE.encode(),
        b"lightcone_target_reference_sha256": target_reference.sha256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("OnlineSPEC table identity metadata mismatch")
    return table


def _attest_onlinespec(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    _assert_onlinespec_study(manifest, selection, lock)
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=lock,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    _onlinespec_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        lock=lock,
        target_reference=target_reference,
    )
    hardware = _load_patched_gpu_doctor(
        args.doctor_json,
        purpose="OnlineSPEC attestation",
    )
    if target_reference.hardware_sha256 != _canonical_sha256(hardware):
        raise ValueError("target reference belongs to a different GPU report")
    attestation = OnlineSpecGpuAttestation(
        schema_version=2,
        status="MEASURED",
        manifest_sha256=manifest.sha256,
        selection_sha256=selection.sha256,
        model_lock_sha256=lock.sha256,
        performance_sha256=evidence_files_sha256((args.performance,)),
        target_reference_sha256=target_reference.sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        hardware_sha256=_canonical_sha256(hardware),
        methods=manifest.methods,
        repetitions=manifest.confirmation_repetitions,
    )
    attestation.write(args.output)
    print(attestation.sha256)
    return 0


def _analyze(args: argparse.Namespace) -> int:
    manifest = SpeedStudyManifest.load(args.manifest)
    selection = SelectionArtifact.load(args.selection)
    _assert_selection_study(selection, manifest)
    model_lock = ModelLock.load(args.model_lock)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("selection artifact belongs to a different model lock")
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=model_lock,
        sampling_profile_sha256=manifest.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    table = _load_formal_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        model_lock=model_lock,
        target_reference=target_reference,
    )
    evidence_state = "UNMEASURED"
    evidence_sha256 = None
    if args.attestation:
        attestation = GpuEvidenceAttestation.load(args.attestation)
        attestation.verify_performance((args.performance,))
        attestation.verify_target_reference(target_reference)
        if attestation.manifest_sha256 != manifest.sha256:
            raise ValueError("attestation manifest identity mismatch")
        if attestation.selection_sha256 != selection.sha256:
            raise ValueError("attestation selection identity mismatch")
        if attestation.model_lock_sha256 != model_lock.sha256:
            raise ValueError("attestation model-lock identity mismatch")
        revisions = {model.model_id: model.revision for model in model_lock.models}
        if attestation.target_revision != revisions.get(
            "Qwen/Qwen3-8B"
        ) or attestation.drafter_revision != revisions.get("z-lab/Qwen3-8B-DFlash-b16"):
            raise ValueError("attestation model revisions mismatch")
        evidence_state = "MEASURED"
        evidence_sha256 = attestation.sha256
    gate = evaluate_speed_gate(
        table.to_pylist(),
        seed=args.bootstrap_seed,
        gpu_evidence=evidence_state,
        evidence_sha256=evidence_sha256,
    )
    _write_json(
        args.output,
        {
            **asdict(gate),
            "selection_protocol": selection.selection_protocol,
            "optimized_grid_claim": (
                selection.selection_protocol == "successive_halving"
            ),
        },
    )
    return 0 if gate.passed else 42


def _analyze_onlinespec(args: argparse.Namespace) -> int:
    manifest = OnlineSpecManifest.load(args.manifest)
    selection = OnlineSpecSelection.load(args.selection)
    lock = ModelLock.load(args.model_lock)
    _assert_onlinespec_study(manifest, selection, lock)
    target_reference = _load_target_reference(
        args.target_reference,
        model_lock=lock,
        sampling_profile_sha256=selection.sampling_profile_sha256,
        concurrency=selection.selected_concurrency,
    )
    table = _onlinespec_table(
        args.performance,
        manifest=manifest,
        selection=selection,
        lock=lock,
        target_reference=target_reference,
    )
    evidence = "UNMEASURED"
    attestation_sha256 = None
    if args.attestation:
        attestation = OnlineSpecGpuAttestation.load(args.attestation)
        if (
            attestation.manifest_sha256 != manifest.sha256
            or attestation.selection_sha256 != selection.sha256
            or attestation.model_lock_sha256 != lock.sha256
            or attestation.performance_sha256
            != evidence_files_sha256((args.performance,))
            or attestation.target_reference_sha256 != target_reference.sha256
        ):
            raise ValueError("OnlineSPEC attestation does not bind this table")
        evidence = "MEASURED"
        attestation_sha256 = attestation.sha256
    comparisons = compare_onlinespec(table.to_pylist(), seed=args.bootstrap_seed)
    safety_pass = all(comparison.safety_pass for comparison in comparisons)
    acceleration_pass = any(
        comparison.acceleration_pass for comparison in comparisons
    )
    status = (
        "UNMEASURED"
        if evidence == "UNMEASURED"
        else "PASS" if safety_pass and acceleration_pass else "BLOCKED"
    )
    _write_json(
        args.output,
        {
            "schema_version": 2,
            "study": "onlinespec-clean-room-baseline",
            "gpu_evidence": evidence,
            "status": status,
            "attestation_sha256": attestation_sha256,
            "core_speed_gate_affected": False,
            "safety_pass": safety_pass,
            "at_least_one_acceleration_pass": acceleration_pass,
            "passing_methods": [
                comparison.method
                for comparison in comparisons
                if comparison.passed
            ],
            "selection_protocol": selection.selection_protocol,
            "optimized_grid_claim": (
                selection.selection_protocol == "successive_halving"
            ),
            "comparisons": [asdict(row) for row in comparisons],
        },
    )
    return 0 if status == "PASS" else 42


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(format_doctor(args.path))
        return 0
    if args.command == "validate-config":
        config = load_run_config(args.config)
        print(config.model_dump_json(indent=2))
        return 0
    if args.command == "build-speed-study":
        manifest = SpeedStudyManifest.default()
        manifest.write(args.output)
        print(manifest.sha256)
        return 0
    if args.command == "build-onlinespec-study":
        manifest = OnlineSpecManifest.default()
        manifest.write(args.output)
        print(manifest.sha256)
        return 0
    if args.command == "verify-onlinespec-source":
        receipt = verify_onlinespec_source_checkout(args.checkout, args.audit)
        _write_json(args.output, receipt)
        print(_canonical_sha256(receipt))
        return 0
    if args.command == "list-onlinespec-candidates":
        return _list_onlinespec_candidates(args)
    if args.command == "lock-models":
        lock = resolve_model_lock(tuple(args.models), token=os.environ.get("HF_TOKEN"))
        lock.write(args.output)
        print(lock.sha256)
        return 0
    if args.command == "prepare-models":
        lock = ModelLock.load(args.lockfile)
        roots = prepare_models(
            lock,
            args.model_cache,
            token=os.environ.get("HF_TOKEN"),
            local_files_only=args.offline,
        )
        _write_json(
            args.output,
            {
                "schema_version": 2,
                "lock_sha256": lock.sha256,
                "roots": roots,
            },
        )
        return 0
    if args.command == "select-speed-config":
        return _select(args)
    if args.command == "select-anchor-config":
        return _select_anchor(args)
    if args.command == "select-onlinespec-config":
        return _select_onlinespec(args)
    if args.command == "select-onlinespec-anchor-config":
        return _select_onlinespec_anchor(args)
    if args.command == "render-runtime":
        return _render_runtime(args)
    if args.command == "render-onlinespec-runtime":
        return _render_onlinespec_runtime(args)
    if args.command == "render-onlinespec-tuning-runtime":
        return _render_onlinespec_tuning_runtime(args)
    if args.command == "render-static-load-runtime":
        return _render_static_load_runtime(args)
    if args.command == "render-tuning-runtime":
        return _render_tuning_runtime(args)
    if args.command == "render-replication-runtime":
        return _render_replication_runtime(args)
    if args.command == "list-tuning-candidates":
        return _list_tuning_candidates(args)
    if args.command == "run-controlled-slice":
        return _run_controlled_slice(args)
    if args.command == "run-onlinespec-tuning-slice":
        return _run_onlinespec_tuning_slice(args)
    if args.command == "run-natural-slice":
        return _run_natural_slice(args)
    if args.command == "build-profiler-plan":
        return _build_profiler_plan(args)
    if args.command == "collect-static-load-screen":
        return _collect_static_load(args)
    if args.command == "advance-tuning-stage":
        return _advance_tuning(args)
    if args.command == "advance-onlinespec-tuning-stage":
        return _advance_onlinespec_tuning(args)
    if args.command == "run-confirmation":
        return _run_confirmation(args)
    if args.command == "run-onlinespec-confirmation":
        return _run_onlinespec_confirmation(args)
    if args.command == "run-target-reference":
        return _run_target_reference(args)
    if args.command == "collect-speed-study":
        return _collect_speed_study(args)
    if args.command == "collect-onlinespec-study":
        return _collect_onlinespec_study(args)
    if args.command == "build-confirmation-queue":
        return _build_confirmation_queue(args)
    if args.command == "build-onlinespec-queue":
        return _build_onlinespec_queue(args)
    if args.command == "attest-speed-study":
        return _attest(args)
    if args.command == "attest-onlinespec-study":
        return _attest_onlinespec(args)
    if args.command == "analyze-speed-study":
        return _analyze(args)
    if args.command == "analyze-onlinespec-study":
        return _analyze_onlinespec(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
