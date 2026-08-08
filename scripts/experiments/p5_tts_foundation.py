#!/usr/bin/env python3
"""Build and attest the independent 0--40K TTS-vs-Static foundation gate.

This is deliberately a small consumer of the stride screen.  It does not
select a stride and it does not run an experiment: it turns the three
canonical TTS roles selected by the screen into one immutable, paired
confirmation manifest and later closes the evidence graph of that run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from lightcone_spec.artifacts.coverage import build_coverage
from lightcone_spec.artifacts.rundir import REQUIRED_FILES
from lightcone_spec.config.schema import MODEL_PAIRS
from lightcone_spec.exit_codes import ConfigError, LockError
from lightcone_spec.locking.download import load_model_roots
from lightcone_spec.locking.hashing import canonical_json, sha256_json
from lightcone_spec.locking.lockfile import load_lockfile
from lightcone_spec.orchestration.catalog import (
    P5_PRIORITY_FINAL_CONTEXTS,
    P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF,
    P5_RTX_PRO_6000_SERVER_DENSE_BF16_PEAK_BASIS,
    P5_RTX_PRO_6000_SERVER_DENSE_BF16_TFLOPS_PER_GPU,
)
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.units import RunUnit


FOUNDATION_CONTEXTS = tuple(P5_PRIORITY_FINAL_CONTEXTS)
FOUNDATION_ROLES = (
    "tts_acceptance_best",
    "tts_engineering_best",
    "same_stride_tts_for_l0",
)
FOUNDATION_PHASE = "p5_tts_0_40k_foundation_v1"
MIN_PROMPT_CLUSTERS = 2
MIN_REPETITIONS = 5
PROMPT_WINDOWS = {
    "selection": {"offset": 0, "limit": 40, "half_open": [0, 40]},
    "foundation": {"offset": 40, "limit": 48, "half_open": [40, 88]},
}


class FoundationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise FoundationError(f"duplicate JSON key {key!r}: {path}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FoundationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FoundationError(f"JSON must be an object: {path}")
    return value


def _verify_sidecar(path: Path, sidecar: Path | None = None) -> Path:
    path = path.resolve()
    sidecar = (sidecar or Path(str(path) + ".sha256")).resolve()
    if not path.is_file() or not sidecar.is_file():
        raise FoundationError(f"artifact or SHA-256 sidecar is missing: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != _sha256(path):
        raise FoundationError(f"SHA-256 sidecar mismatch: {path}")
    return sidecar


def _cas_text(path: Path, value: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise FoundationError(f"immutable output collision: {path}")
        return
    _atomic_text(path, value)


def _write_receipt(
    path: Path, payload: Mapping[str, Any], evidence: Iterable[Path]
) -> dict[str, Any]:
    paths = sorted({Path(item).resolve() for item in evidence})
    for item in paths:
        if not item.is_file():
            raise FoundationError(f"receipt evidence is missing: {item}")
    body = {
        **payload,
        "evidence": [
            {"path": str(item), "sha256": _sha256(item)} for item in paths
        ],
    }
    text = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = path.resolve()
    _cas_text(path, text)
    _cas_text(Path(str(path) + ".sha256"), _sha256(path) + "\n")
    return body


def _receipt_graph(root: Path) -> tuple[dict[Path, dict[str, Any]], list[Path]]:
    """Verify a receipt and every receipt reachable through its evidence."""

    nodes: dict[Path, dict[str, Any]] = {}
    evidence: set[Path] = set()
    active: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in active:
            raise FoundationError(f"receipt evidence cycle: {path}")
        if path in nodes:
            return
        active.add(path)
        sidecar = _verify_sidecar(path)
        payload = _load_json(path)
        if not isinstance(payload.get("schema_version"), int):
            raise FoundationError(f"receipt schema is missing: {path}")
        rows = payload.get("evidence")
        if not isinstance(rows, list) or not rows:
            raise FoundationError(f"receipt has no evidence: {path}")
        nodes[path] = payload
        evidence.update((path, sidecar))
        seen: set[Path] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise FoundationError(f"invalid evidence[{index}]: {path}")
            raw = row.get("path")
            item = Path(str(raw)).resolve()
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise FoundationError(f"non-absolute evidence path: {path}")
            if item in seen or not item.is_file() or row.get("sha256") != _sha256(item):
                raise FoundationError(f"receipt evidence mismatch: {item}")
            seen.add(item)
            evidence.add(item)
            if item.suffix == ".json":
                try:
                    nested = _load_json(item)
                except FoundationError:
                    nested = {}
                if {"schema_version", "status", "scope", "evidence"}.issubset(nested):
                    visit(item)
        active.remove(path)

    visit(root)
    return nodes, sorted(evidence)


def _find_selection(
    selected_or_terminal: Path,
    source_manifest_path: Path,
) -> tuple[dict[str, Any], Path, list[Path]]:
    nodes, evidence = _receipt_graph(selected_or_terminal)
    candidates = []
    for path, payload in nodes.items():
        winners = payload.get("winners")
        if (
            payload.get("schema_version") == 2
            and isinstance(winners, dict)
            and set(FOUNDATION_ROLES).issubset(winners)
            and "l0_best" in winners
        ):
            candidates.append((path, payload))
    if len(candidates) != 1:
        old = [
            path
            for path, payload in nodes.items()
            if payload.get("schema_version") == 1
            and isinstance(payload.get("winners"), dict)
            and "tts_best" in payload["winners"]
        ]
        hint = (
            f"; legacy v1 selection found at {old[0]} -- regenerate schema v2"
            if old
            else ""
        )
        raise FoundationError(
            f"expected exactly one canonical schema-v2 selection, found "
            f"{len(candidates)}{hint}"
        )
    selection_path, selection = candidates[0]

    # Recompute the selector result when the trusted selector exposes its
    # validator.  This prevents an attested but hand-edited winner row from
    # becoming a formal confirmation input.
    selector_script = Path(__file__).with_name("select_p5_stride_screen.py")
    spec = importlib.util.spec_from_file_location(
        "_p5_tts_foundation_selector", selector_script
    )
    if spec is None or spec.loader is None:
        raise FoundationError(f"cannot load selector: {selector_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        recomputed = module.validate_selection_receipt(
            selector_path=selection_path,
            manifest_path=source_manifest_path,
        )
    except Exception as exc:
        raise FoundationError(f"selection semantic validation failed: {exc}") from exc
    if recomputed != selection:
        raise FoundationError("selection changed during semantic validation")
    evidence.append(selector_script.resolve())
    return selection, selection_path, sorted(set(evidence))


def _runtime_fingerprint(path: Path) -> tuple[dict[str, Any], list[Path]]:
    sidecar = _verify_sidecar(path)
    value = _load_json(path)
    claimed = value.get("sha256")
    body = dict(value)
    body.pop("sha256", None)
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("files"), dict)
        or not isinstance(value.get("locked_reference"), dict)
        or claimed != sha256_json(body)
    ):
        raise FoundationError("runtime fingerprint identity mismatch")
    return json.loads(canonical_json(value)), [path.resolve(), sidecar]


def _locked_inputs(
    lockfile_path: Path, model_roots_path: Path
) -> tuple[str, str, dict[str, str], list[Path]]:
    try:
        lock = load_lockfile(lockfile_path)
        roots = load_model_roots(model_roots_path)
    except LockError as exc:
        raise FoundationError(str(exc)) from exc
    pair = MODEL_PAIRS["qwen3_4b_dflash16"]
    target = lock.find_snapshot(pair["target"])
    drafter = lock.find_snapshot(pair["drafter"])
    for repo_id in (pair["target"], pair["drafter"]):
        root = roots.get(repo_id)
        if not isinstance(root, str) or not Path(root).is_dir():
            raise FoundationError(f"model-roots lacks existing {repo_id}")
    lock_sidecar = _verify_sidecar(lockfile_path)
    roots_sidecar = _verify_sidecar(model_roots_path)
    return (
        lock.content_sha256(),
        _sha256(model_roots_path),
        {
            "target": target.snapshot_sha,
            "drafter": drafter.snapshot_sha,
            "tokenizer": target.snapshot_sha,
        },
        [
            lockfile_path.resolve(),
            lock_sidecar,
            model_roots_path.resolve(),
            roots_sidecar,
        ],
    )


def _selected_roles(
    selection: Mapping[str, Any],
    source: ExperimentManifest,
    *,
    allow_l0_not_superior: bool = False,
) -> dict[str, dict[str, Any]]:
    """Resolve TTS foundation roles from a schema-v2 screen selection.

    ``allow_l0_not_superior`` admits a screen that resolved every TTS/L0 role but
    did not establish L0 over the acceptance-best TTS.  That ordering is the
    controller-phase hypothesis; requiring it here would block the independent
    TTS-vs-Static foundation as well.  No acceptance, elasticity, or CI gate in
    the foundation compare path is affected.
    """

    if allow_l0_not_superior:
        expected_status = "scientifically_blocked"
        expected_objective = False
    else:
        expected_status = "winner_selected"
        expected_objective = True
    if (
        selection.get("schema_version") != 2
        or selection.get("status") != expected_status
        or selection.get("scope") != "candidate_screen_only_no_claim"
        or selection.get("objective_screen_pass") is not expected_objective
    ):
        raise FoundationError("schema-v2 stride screen did not select a winner")
    winners = selection.get("winners")
    if not isinstance(winners, dict) or set(winners) != {
        *FOUNDATION_ROLES,
        "l0_best",
    }:
        raise FoundationError("selection winners are not canonical schema-v2 roles")
    source_tts = {
        (unit.stride, unit.unit_id): unit
        for unit in source.units
        if unit.method == "tts"
    }
    roles = {}
    for role in FOUNDATION_ROLES:
        row = winners.get(role)
        if not isinstance(row, dict):
            raise FoundationError(f"selection role is missing: {role}")
        stride = row.get("stride")
        if (
            isinstance(stride, bool)
            or not isinstance(stride, int)
            or stride <= 0
            or row.get("method") != "tts"
            or row.get("eligible") is not True
            or (stride, row.get("unit_id")) not in source_tts
        ):
            raise FoundationError(f"selection role identity mismatch: {role}")
        roles[role] = {
            "method": "tts",
            "stride": stride,
            "source_unit_id": str(row["unit_id"]),
        }
    return roles


def _inputs(
    *,
    selected_or_terminal: Path,
    source_manifest_path: Path,
    lockfile_path: Path,
    model_roots_path: Path,
    runtime_fingerprint_path: Path,
    allow_l0_not_superior: bool = False,
) -> dict[str, Any]:
    source_sidecar = _verify_sidecar(source_manifest_path)
    try:
        source = ExperimentManifest.load(source_manifest_path)
    except ConfigError as exc:
        raise FoundationError(f"invalid source manifest: {exc}") from exc
    if not source.units or any(
        unit.model_pair != "qwen3_4b_dflash16" for unit in source.units
    ):
        raise FoundationError("source screen is not Qwen3-4B DFlash")
    if (
        int(source.engine_params.get("prompt_offset", 0)) != 0
        or source.engine_params.get("prompt_limit") != 40
    ):
        raise FoundationError("source screen is not bound to prompt window [0,40)")
    selection, selection_path, selection_evidence = _find_selection(
        selected_or_terminal, source_manifest_path
    )
    roles = _selected_roles(
        selection, source, allow_l0_not_superior=allow_l0_not_superior
    )
    runtime, runtime_evidence = _runtime_fingerprint(runtime_fingerprint_path)
    lock_sha, roots_sha, revisions, locked_evidence = _locked_inputs(
        lockfile_path, model_roots_path
    )
    bindings = {
        "schema_version": 2,
        "selection_sha256": _sha256(selection_path),
        "selected_terminal_sha256": _sha256(selected_or_terminal),
        "source_manifest_file_sha256": _sha256(source_manifest_path),
        "source_manifest_sha256": source.content_sha256(),
        "lockfile_sha256": lock_sha,
        "model_roots_sha256": roots_sha,
        "model_revisions": revisions,
        "runtime_implementation_fingerprint": runtime,
        "role_source_unit_ids": {
            role: value["source_unit_id"] for role, value in roles.items()
        },
        "screen_entry_condition": (
            "l0_not_superior_oracle_scope"
            if allow_l0_not_superior
            else "objective_screen_pass"
        ),
    }
    return {
        "source": source,
        "selection": selection,
        "selection_path": selection_path,
        "roles": roles,
        "bindings": bindings,
        "evidence": sorted(
            {
                *selection_evidence,
                source_manifest_path.resolve(),
                source_sidecar,
                *runtime_evidence,
                *locked_evidence,
                Path(__file__).resolve(),
            }
        ),
    }


def _manifest(context: Mapping[str, Any]) -> tuple[ExperimentManifest, dict[str, Any]]:
    source: ExperimentManifest = context["source"]
    roles: dict[str, dict[str, Any]] = context["roles"]
    static_source = next(
        (unit for unit in source.units if unit.method == "static"), None
    )
    tts_by_id = {unit.unit_id: unit for unit in source.units if unit.method == "tts"}
    if static_source is None:
        raise FoundationError("source screen has no Static unit")

    common = {
        "phase": FOUNDATION_PHASE,
        "prompt_subset": "p5_ctx_512-40000",
        "concurrency": 4,
        "trainable_scope": "tail_lora",
        "adapter_rank": 16,
        "lifecycle": "stream",
        "sampling_profile": "greedy_t0",
        "logical_delay": 0,
        "parameter_scope": "tail",
        "parameter_allowlist": (),
    }
    units = [
        replace(
            static_source,
            **common,
            method="static",
            stride=1,
            contention_condition="none",
        )
    ]
    foundation_by_stride: dict[int, RunUnit] = {}
    for role in FOUNDATION_ROLES:
        selected = roles[role]
        stride = int(selected["stride"])
        if stride not in foundation_by_stride:
            source_unit = tts_by_id[selected["source_unit_id"]]
            unit = replace(
                source_unit,
                **common,
                method="tts",
                stride=stride,
                contention_condition="none",
            )
            foundation_by_stride[stride] = unit
            units.append(unit)

    bindings = context["bindings"]
    engine = {
        "prompt_limit": 48,
        "prompt_offset": 40,
        "benchmark_repetitions": 5,
        "max_new_tokens": 512,
        "ignore_eos": True,
        "max_running_requests": 4,
        "max_total_tokens": 400000,
        "p5_context_lengths": list(FOUNDATION_CONTEXTS),
        "p5_context_timing_contract": "independent_exact_context_group_v1",
        "p5_load_groups": [
            {"prompt_subset": "p5_ctx_512-40000", "concurrency": 4}
        ],
        "checkpoint_max_context_length": 40960,
        "checkpoint_max_new_tokens": 512,
        "pytorch_cuda_alloc_conf": P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF,
        "peak_tflops_per_gpu": P5_RTX_PRO_6000_SERVER_DENSE_BF16_TFLOPS_PER_GPU,
        "peak_tflops_basis": P5_RTX_PRO_6000_SERVER_DENSE_BF16_PEAK_BASIS,
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_prompts": 20,
        "trace_level": "light",
        "claim_scope": "tts_0_40k_foundation",
        "model_roots_sha256": bindings["model_roots_sha256"],
        "locked_model_revisions": bindings["model_revisions"],
        "runtime_implementation_fingerprint": bindings[
            "runtime_implementation_fingerprint"
        ],
        "foundation_input_bindings_sha256": sha256_json(bindings),
        "request_timeout_s": 1800,
    }
    manifest = ExperimentManifest(
        name=FOUNDATION_PHASE,
        phase=FOUNDATION_PHASE,
        description=(
            "Independent exact-context Qwen3-4B DFlash Tail-LoRA/AdamW "
            "TTS-vs-Static foundation at 512/1K/2K/4K/8K/16K/32K/40K."
        ),
        profile="local_1x96gb",
        lockfile_sha256=bindings["lockfile_sha256"],
        engine_params=engine,
        units=units,
    )
    resolved_roles = {
        role: {
            **roles[role],
            "foundation_unit_id": foundation_by_stride[
                int(roles[role]["stride"])
            ].unit_id,
        }
        for role in FOUNDATION_ROLES
    }
    return manifest, resolved_roles


def build(
    *,
    selected_or_terminal: Path,
    source_manifest_path: Path,
    lockfile_path: Path,
    model_roots_path: Path,
    runtime_fingerprint_path: Path,
    artifact_root: Path,
    output_manifest_path: Path,
    output_receipt_path: Path,
    allow_l0_not_superior: bool = False,
) -> dict[str, Any]:
    context = _inputs(
        selected_or_terminal=selected_or_terminal.resolve(),
        source_manifest_path=source_manifest_path.resolve(),
        lockfile_path=lockfile_path.resolve(),
        model_roots_path=model_roots_path.resolve(),
        runtime_fingerprint_path=runtime_fingerprint_path.resolve(),
        allow_l0_not_superior=allow_l0_not_superior,
    )
    manifest, roles = _manifest(context)
    body = canonical_json(manifest.to_dict())
    output_manifest_path = output_manifest_path.resolve()
    _cas_text(output_manifest_path, body)
    _cas_text(
        Path(str(output_manifest_path) + ".sha256"),
        hashlib.sha256(body.encode("utf-8")).hexdigest() + "\n",
    )
    ExperimentManifest.load(output_manifest_path)
    identity = {
        "schema_version": 2,
        "manifest_sha256": manifest.content_sha256(),
        "artifact_root": str(artifact_root.resolve()),
        "contexts": list(FOUNDATION_CONTEXTS),
        "concurrency": 4,
        "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-2},
        "weight_update_mode": "lora",
        "parameter_scope": "tail",
        "prompt_windows": PROMPT_WINDOWS,
        "bindings": context["bindings"],
        "roles": roles,
    }
    return _write_receipt(
        output_receipt_path,
        {
            "schema_version": 2,
            "status": "ready_for_execution",
            "scope": "tts_0_40k_foundation",
            "identity": identity,
            "identity_sha256": sha256_json(identity),
            "unit_ids": [unit.unit_id for unit in manifest.units],
        },
        [
            *context["evidence"],
            output_manifest_path,
            Path(str(output_manifest_path) + ".sha256"),
        ],
    )


def _verify_ledger(root: Path, path: Path) -> list[Path]:
    ledger = _load_json(path)
    evidence = [path.resolve()]
    for relative, row in ledger.items():
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise FoundationError(f"ledger path escapes root: {relative}") from exc
        if (
            not isinstance(row, dict)
            or not candidate.is_file()
            or row.get("sha256") != _sha256(candidate)
            or row.get("bytes") != candidate.stat().st_size
        ):
            raise FoundationError(f"ledger mismatch: {candidate}")
        evidence.append(candidate)
    return evidence


def _validate_runs(
    *,
    artifact_root: Path,
    manifest: ExperimentManifest,
    input_runs: list[dict[str, Any]],
) -> list[Path]:
    by_id = {unit.unit_id: unit for unit in manifest.units}
    if len(input_runs) != len(by_id) or {
        str(row.get("unit_id")) for row in input_runs if isinstance(row, dict)
    } != set(by_id):
        raise FoundationError("analysis run coverage differs from foundation")
    evidence = []
    for row in input_runs:
        if not isinstance(row, dict) or set(row) != {
            "run_id", "unit_id", "manifest_sha256", "hashes_sha256"
        }:
            raise FoundationError("analysis input-run schema mismatch")
        run_id = str(row["run_id"])
        if Path(run_id).name != run_id:
            raise FoundationError(f"run_id is not a basename: {run_id}")
        root = (artifact_root / run_id).resolve()
        manifest_path = root / "manifest.json"
        hashes_path = root / "hashes.json"
        if (
            not manifest_path.is_file()
            or row["manifest_sha256"] != _sha256(manifest_path)
            or not hashes_path.is_file()
            or row["hashes_sha256"] != _sha256(hashes_path)
        ):
            raise FoundationError(f"analysis input-run hash drift: {run_id}")
        payload = _load_json(manifest_path)
        try:
            observed = RunUnit.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise FoundationError(f"invalid run unit {run_id}: {exc}") from exc
        expected = by_id[str(row["unit_id"])]
        if observed.unit_id != expected.unit_id or any(
            payload.get(key) != value
            for key, value in expected.to_manifest_dict().items()
        ):
            raise FoundationError(f"run unit drift: {run_id}")
        if payload.get("experiment_manifest_sha256") != manifest.content_sha256():
            raise FoundationError(f"run experiment-manifest drift: {run_id}")
        engine = payload.get("engine_params")
        if not isinstance(engine, dict) or any(
            engine.get(key) != value
            for key, value in manifest.engine_params.items()
        ):
            raise FoundationError(f"run engine contract drift: {run_id}")
        exit_payload = _load_json(root / "exit.json")
        if exit_payload.get("status") != "complete_valid":
            raise FoundationError(f"run is not complete_valid: {run_id}")
        ledger = _load_json(hashes_path)
        if not (set(REQUIRED_FILES) - {"hashes.json"}).issubset(ledger):
            raise FoundationError(f"run ledger omits normative files: {run_id}")
        evidence.extend([manifest_path, *_verify_ledger(root, hashes_path)])
    return evidence


def _validate_analysis(
    *, analysis_root: Path, artifact_root: Path, manifest: ExperimentManifest
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[Path]]:
    analysis_manifest_path = analysis_root / "analysis-manifest.json"
    analysis_sidecar = _verify_sidecar(
        analysis_manifest_path, analysis_root / "analysis-manifest.sha256"
    )
    hashes_path = analysis_root / "analysis-hashes.json"
    evidence = [analysis_manifest_path, analysis_sidecar]
    evidence.extend(_verify_ledger(analysis_root, hashes_path))
    analysis_manifest = _load_json(analysis_manifest_path)
    analysis = analysis_manifest.get("analysis")
    if (
        not isinstance(analysis, dict)
        or analysis.get("baseline") != "static"
        or analysis.get("expected_manifest_sha256") != manifest.content_sha256()
    ):
        raise FoundationError("analysis identity differs from foundation")
    input_runs = analysis_manifest.get("input_runs")
    if not isinstance(input_runs, list):
        raise FoundationError("analysis input runs are missing")
    evidence.extend(
        _validate_runs(
            artifact_root=artifact_root, manifest=manifest, input_runs=input_runs
        )
    )
    derived = analysis_manifest.get("derived_outputs")
    ledger = _load_json(hashes_path)
    if not isinstance(derived, dict) or any(
        name not in derived
        for name in ("p5_long_context_acceptance.parquet", "p5_claim_gates.json")
    ):
        raise FoundationError("analysis lacks required P5 outputs")
    if set(ledger) != {*derived, "analysis-manifest.json", "analysis-manifest.sha256"}:
        raise FoundationError("analysis hash ledger is not transitively closed")
    for name, row in derived.items():
        if ledger.get(name) != row:
            raise FoundationError(f"derived-output ledger mismatch: {name}")
    curve_path = analysis_root / "p5_long_context_acceptance.parquet"
    try:
        curve = pd.read_parquet(curve_path)
    except Exception as exc:
        raise FoundationError(f"cannot read P5 acceptance curve: {exc}") from exc
    gates = json.loads((analysis_root / "p5_claim_gates.json").read_text())
    if not isinstance(gates, list) or not all(isinstance(row, dict) for row in gates):
        raise FoundationError("p5_claim_gates.json must contain a row list")
    return curve, gates, evidence


def _validate_coverage(path: Path, manifest: ExperimentManifest) -> list[Path]:
    sidecar = _verify_sidecar(path)
    payload = _load_json(path)
    cells = payload.get("cells")
    if not isinstance(cells, dict) or set(cells) != {
        unit.unit_id for unit in manifest.units
    }:
        raise FoundationError("coverage unit set differs from foundation")
    statuses = {
        unit_id: row.get("status")
        for unit_id, row in cells.items()
        if isinstance(row, dict)
    }
    if len(statuses) != len(cells) or set(statuses.values()) != {"complete_valid"}:
        raise FoundationError("foundation coverage is not complete_valid")
    expected = build_coverage(manifest.expected_units(), statuses)
    if payload != {"cells": expected.cells, "summary": expected.summary()}:
        raise FoundationError("coverage differs from deterministic recomputation")
    return [path.resolve(), sidecar]


def _one_gate(
    rows: list[dict[str, Any]], *, stride: int
) -> dict[str, Any]:
    matched = [
        row
        for row in rows
        if row.get("method") == "tts"
        and row.get("baseline_method") == "static"
        and row.get("update_stride") == stride
        and row.get("offered_concurrency") == 4
    ]
    if len(matched) != 1:
        raise FoundationError(
            f"claim-gate role lookup (tts,stride={stride},c4) found {len(matched)} rows"
        )
    return matched[0]


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FoundationError(f"claim-gate value is not numeric: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise FoundationError("claim-gate value is not finite")
    return value


def _one_numeric_value(series: pd.Series, *, field: str) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or values.nunique(dropna=False) != 1:
        raise FoundationError(f"P5 curve has inconsistent {field}")
    return _finite(values.iloc[0])


def _role_result(
    *,
    role: str,
    definition: Mapping[str, Any],
    gates: list[dict[str, Any]],
    curve: pd.DataFrame,
) -> dict[str, Any]:
    stride = int(definition["stride"])
    gate = _one_gate(gates, stride=stride)
    selected = curve[
        (curve["method"] == "tts")
        & (pd.to_numeric(curve["update_stride"], errors="coerce") == stride)
        & (pd.to_numeric(curve["offered_concurrency"], errors="coerce") == 4)
    ].copy()
    if len(selected) != len(FOUNDATION_CONTEXTS) or set(
        pd.to_numeric(selected["context_length"], errors="coerce").astype(int)
    ) != set(FOUNDATION_CONTEXTS):
        raise FoundationError(f"{role} does not cover all eight exact contexts")
    if selected["context_length"].duplicated().any():
        raise FoundationError(f"{role} has duplicate context buckets")
    for name in (
        "version_mismatch_count",
        "exactness_violations",
        "adaptation_fallback_count",
    ):
        values = pd.to_numeric(selected[name], errors="coerce")
        if values.isna().any() or (values != 0).any():
            raise FoundationError(f"{role} safety gate failed: {name}")
    lcag_low = _finite(gate.get("lcag_ci_low"))
    delta_e = _finite(gate.get("mean_delta_acceptance_elasticity"))
    clusters = gate.get("paired_prompt_clusters")
    repetitions = gate.get("benchmark_repetitions")
    long_rows = selected[
        pd.to_numeric(selected["context_length"], errors="coerce") >= 4096
    ]
    curve_lcag = _one_numeric_value(long_rows["lcag"], field="lcag")
    curve_lcag_low = _one_numeric_value(
        long_rows["lcag_ci_low"], field="lcag_ci_low"
    )
    curve_lcag_high = _one_numeric_value(
        long_rows["lcag_ci_high"], field="lcag_ci_high"
    )
    curve_clusters = _one_numeric_value(
        long_rows["lcag_prompt_clusters"], field="lcag_prompt_clusters"
    )
    curve_repetitions = _one_numeric_value(
        selected["benchmark_repetitions"], field="benchmark_repetitions"
    )
    if (
        not math.isclose(curve_lcag_low, lcag_low, rel_tol=1e-12, abs_tol=1e-12)
        or curve_clusters != clusters
        or curve_repetitions != repetitions
        or curve_lcag_high < curve_lcag_low
    ):
        raise FoundationError(f"{role} aggregate CI/sample provenance mismatch")
    sample_pass = (
        isinstance(clusters, int)
        and not isinstance(clusters, bool)
        and clusters >= MIN_PROMPT_CLUSTERS
        and isinstance(repetitions, int)
        and not isinstance(repetitions, bool)
        and repetitions >= MIN_REPETITIONS
    )
    exactness = gate.get("exactness_pass") is True
    algorithmic = bool(
        gate.get("algorithmic_pass") is True
        and exactness
        and sample_pass
        and lcag_low > 0
        and delta_e < 0
    )
    curve_fields = [
        name
        for name in (
            "context_length",
            "survival_weighted_accepted_prefix",
            "acceptance_gain_vs_baseline",
            "acceptance_gain_ci_low",
            "acceptance_gain_ci_high",
            "committed_tokens_per_verify",
            "target_calls_per_output_token",
            "decode_goodput_tps",
            "throughput_speedup_vs_baseline",
            "p50_itl_ms",
            "p95_itl_ms",
            "p99_itl_ms",
            "peak_hbm_bytes",
        )
        if name in selected
    ]
    window_curve = json.loads(
        selected.sort_values("context_length")[curve_fields].to_json(
            orient="records", double_precision=15
        )
    )
    return {
        **definition,
        "gate": {
            "algorithmic_pass": algorithmic,
            "engineering_pass": gate.get("engineering_pass") is True,
            "exactness_pass": exactness,
            "scientific_sample_pass": sample_pass,
            "paired_prompt_clusters": clusters,
            "benchmark_repetitions": repetitions,
            "lcag_ci_low": lcag_low,
            "lcag": curve_lcag,
            "lcag_ci_high": curve_lcag_high,
            "mean_delta_acceptance_elasticity": delta_e,
            "window_dominance_pass": gate.get("window_dominance_pass") is True,
        },
        "window_curve": window_curve,
    }


def _validate_curve_identity(
    curve: pd.DataFrame, manifest: ExperimentManifest
) -> None:
    expected = {
        (unit.method, unit.stride) for unit in manifest.units
    }
    observed = {
        (str(method), int(stride), int(concurrency))
        for method, stride, concurrency in curve[
            ["method", "update_stride", "offered_concurrency"]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    expected_loads = {(method, stride, 4) for method, stride in expected}
    if observed != expected_loads:
        raise FoundationError(
            "P5 curve method/stride/load identity differs: "
            f"{observed} != {expected_loads}"
        )
    for method, stride in expected:
        rows = curve[
            (curve["method"] == method)
            & (pd.to_numeric(curve["update_stride"], errors="coerce") == stride)
            & (
                pd.to_numeric(curve["offered_concurrency"], errors="coerce")
                == 4
            )
        ]
        contexts = pd.to_numeric(rows["context_length"], errors="coerce")
        if (
            len(rows) != len(FOUNDATION_CONTEXTS)
            or contexts.isna().any()
            or contexts.duplicated().any()
            or set(contexts.astype(int)) != set(FOUNDATION_CONTEXTS)
        ):
            raise FoundationError(
                f"P5 curve lacks exact eight-bucket coverage: {method}/s{stride}"
            )
        for name in (
            "version_mismatch_count",
            "exactness_violations",
            "adaptation_fallback_count",
        ):
            values = pd.to_numeric(rows[name], errors="coerce")
            if values.isna().any() or (values != 0).any():
                raise FoundationError(
                    f"P5 curve safety failed for {method}/s{stride}: {name}"
                )


def compare(
    *,
    selected_or_terminal: Path,
    source_manifest_path: Path,
    lockfile_path: Path,
    model_roots_path: Path,
    runtime_fingerprint_path: Path,
    artifact_root: Path,
    generation_receipt_path: Path,
    foundation_manifest_path: Path,
    analysis_root: Path,
    coverage_path: Path,
    output_path: Path,
    allow_l0_not_superior: bool = False,
) -> dict[str, Any]:
    context = _inputs(
        selected_or_terminal=selected_or_terminal.resolve(),
        source_manifest_path=source_manifest_path.resolve(),
        lockfile_path=lockfile_path.resolve(),
        model_roots_path=model_roots_path.resolve(),
        runtime_fingerprint_path=runtime_fingerprint_path.resolve(),
        allow_l0_not_superior=allow_l0_not_superior,
    )
    expected, roles = _manifest(context)
    _verify_sidecar(foundation_manifest_path)
    observed = ExperimentManifest.load(foundation_manifest_path)
    if observed.to_dict() != expected.to_dict():
        raise FoundationError("foundation manifest differs from frozen design")
    receipt_nodes, generation_evidence = _receipt_graph(generation_receipt_path)
    generation = receipt_nodes[generation_receipt_path.resolve()]
    expected_identity = {
        "schema_version": 2,
        "manifest_sha256": expected.content_sha256(),
        "artifact_root": str(artifact_root.resolve()),
        "contexts": list(FOUNDATION_CONTEXTS),
        "concurrency": 4,
        "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-2},
        "weight_update_mode": "lora",
        "parameter_scope": "tail",
        "prompt_windows": PROMPT_WINDOWS,
        "bindings": context["bindings"],
        "roles": roles,
    }
    if (
        generation.get("schema_version") != 2
        or generation.get("status") != "ready_for_execution"
        or generation.get("scope") != "tts_0_40k_foundation"
        or generation.get("identity") != expected_identity
        or generation.get("identity_sha256") != sha256_json(expected_identity)
        or generation.get("unit_ids") != [unit.unit_id for unit in expected.units]
    ):
        raise FoundationError("generation receipt identity mismatch")

    coverage_evidence = _validate_coverage(coverage_path, expected)
    curve, gates, analysis_evidence = _validate_analysis(
        analysis_root=analysis_root.resolve(),
        artifact_root=artifact_root.resolve(),
        manifest=expected,
    )
    missing_columns = sorted(
        {
            "method",
            "update_stride",
            "offered_concurrency",
            "context_length",
            "version_mismatch_count",
            "exactness_violations",
            "adaptation_fallback_count",
            "lcag",
            "lcag_ci_low",
            "lcag_ci_high",
            "lcag_prompt_clusters",
            "benchmark_repetitions",
        }
        - set(curve.columns)
    )
    if missing_columns:
        raise FoundationError(f"P5 curve lacks columns: {missing_columns}")
    _validate_curve_identity(curve, expected)
    results = {
        role: _role_result(
            role=role, definition=roles[role], gates=gates, curve=curve
        )
        for role in FOUNDATION_ROLES
    }
    acceptance_pass = results["tts_acceptance_best"]["gate"][
        "algorithmic_pass"
    ]
    identity = {
        "schema_version": 2,
        "generation_receipt_sha256": _sha256(generation_receipt_path),
        "manifest_sha256": expected.content_sha256(),
        "coverage_sha256": _sha256(coverage_path),
        "analysis_manifest_sha256": _sha256(
            analysis_root / "analysis-manifest.json"
        ),
        "analysis_hashes_sha256": _sha256(analysis_root / "analysis-hashes.json"),
        "contexts": list(FOUNDATION_CONTEXTS),
        "role_mapping": roles,
        "bindings_sha256": sha256_json(context["bindings"]),
    }
    return _write_receipt(
        output_path,
        {
            "schema_version": 2,
            "status": (
                "TTS_0_40K_CONFIRMED" if acceptance_pass else "BLOCKED"
            ),
            "scope": "tts_0_40k_foundation",
            "formal_acceptance_foundation_pass": acceptance_pass,
            "engineering_role_nonblocking": True,
            "identity": identity,
            "identity_sha256": sha256_json(identity),
            "roles": results,
        },
        [
            *context["evidence"],
            *generation_evidence,
            *coverage_evidence,
            *analysis_evidence,
            foundation_manifest_path.resolve(),
            Path(str(foundation_manifest_path) + ".sha256").resolve(),
            generation_receipt_path.resolve(),
        ],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--selected-receipt", type=Path, required=True)
        command.add_argument("--source-screen-manifest", type=Path, required=True)
        command.add_argument("--lockfile", type=Path, required=True)
        command.add_argument("--model-roots", type=Path, required=True)
        command.add_argument("--runtime-fingerprint", type=Path, required=True)
        command.add_argument("--artifact-root", type=Path, required=True)
        command.add_argument("--foundation-manifest", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        command.add_argument(
            "--allow-l0-not-superior-oracle-scope",
            action="store_true",
            help=(
                "accept a candidate screen that resolved every role but did not "
                "establish L0 over the acceptance-best TTS; this only widens the "
                "screen entry condition and never relaxes a foundation CI gate"
            ),
        )
    compare_parser = commands.choices["compare"]
    compare_parser.add_argument("--generation-receipt", type=Path, required=True)
    compare_parser.add_argument("--analysis-root", type=Path, required=True)
    compare_parser.add_argument("--coverage", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "selected_or_terminal": args.selected_receipt,
        "source_manifest_path": args.source_screen_manifest,
        "lockfile_path": args.lockfile,
        "model_roots_path": args.model_roots,
        "runtime_fingerprint_path": args.runtime_fingerprint,
        "artifact_root": args.artifact_root,
        "allow_l0_not_superior": args.allow_l0_not_superior_oracle_scope,
    }
    try:
        if args.command == "build":
            build(
                **kwargs,
                output_manifest_path=args.foundation_manifest,
                output_receipt_path=args.receipt,
            )
        else:
            compare(
                **kwargs,
                generation_receipt_path=args.generation_receipt,
                foundation_manifest_path=args.foundation_manifest,
                analysis_root=args.analysis_root,
                coverage_path=args.coverage,
                output_path=args.receipt,
            )
    except FoundationError as exc:
        raise SystemExit(f"TTS foundation {args.command} failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
