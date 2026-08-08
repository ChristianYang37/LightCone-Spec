#!/usr/bin/env python3
"""Build hash-closed P5 controller manifests from a selected terminal receipt."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from lightcone_spec.artifacts.coverage import build_coverage
from lightcone_spec.config.schema import MODEL_PAIRS
from lightcone_spec.exit_codes import ConfigError, LockError
from lightcone_spec.locking.download import load_model_roots
from lightcone_spec.locking.hashing import canonical_json, sha256_json
from lightcone_spec.locking.lockfile import load_lockfile
from lightcone_spec.orchestration import controller_manifests as controller_builder
from lightcone_spec.orchestration.controller_manifests import (
    MATCHED_PYTORCH_CUDA_ALLOC_CONF,
    SOURCE_MANIFEST_NAME,
    build_matched_controller_manifests,
)
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.runtime_config import (
    runtime_implementation_fingerprint,
)


TRUSTED_SELECTOR = Path(__file__).resolve().with_name("select_p5_stride_screen.py")
TRUSTED_QUEUE = Path(__file__).resolve().with_name(
    "run_priority_l0_stride_screen_queue.sh"
)
HEX = frozenset("0123456789abcdef")


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReceiptNode:
    path: Path
    sidecar: Path
    payload: dict[str, Any]
    evidence: tuple[Path, ...]


@dataclass(frozen=True)
class ValidatedSelectedScreenInputs:
    """Read-only, transitively verified inputs shared by downstream builders."""

    terminal_receipt: Path
    execution_receipt: Path
    selection_receipt: Path
    selection: dict[str, Any]
    source_manifest_path: Path
    source_manifest: ExperimentManifest
    lockfile_path: Path
    lockfile_sha256: str
    model_roots_path: Path
    model_roots_sha256: str
    model_revisions: dict[str, str]
    screen_runtime_implementation_fingerprint: dict[str, Any]
    runtime_implementation_fingerprint: dict[str, Any]
    runtime_transition: dict[str, Any]
    pytorch_cuda_alloc_conf: str
    static_unit_id: str
    binding_hashes: dict[str, str]
    evidence: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in HEX for char in value)
    ):
        raise BuildError(f"{field} is not a lowercase SHA-256")
    return value


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
    def reject_duplicate_keys(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise BuildError(f"duplicate JSON key {key!r} in {path}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"JSON must be an object: {path}")
    return value


def _verify_exact_sidecar(path: Path, sidecar: Path | None = None) -> Path:
    path = path.resolve()
    sidecar = (sidecar or Path(str(path) + ".sha256")).resolve()
    if not path.is_file() or not sidecar.is_file():
        raise BuildError(f"artifact or SHA-256 sidecar is missing: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != _sha256(path):
        raise BuildError(f"SHA-256 sidecar mismatch: {path}")
    return sidecar


def _verify_manifest_sidecar(path: Path) -> Path:
    path = path.resolve()
    sidecar = Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise BuildError(f"source manifest or sidecar is missing: {path}")
    raw = _load_json(path)
    canonical = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
    if sidecar.read_text(encoding="utf-8").strip() not in {
        _sha256(path),
        canonical,
    }:
        raise BuildError(f"source manifest sidecar mismatch: {path}")
    return sidecar.resolve()


def _evidence(payload: Mapping[str, Any], *, owner: Path) -> tuple[Path, ...]:
    rows = payload.get("evidence")
    if not isinstance(rows, list) or not rows:
        raise BuildError(f"receipt has no evidence: {owner}")
    output: list[Path] = []
    seen: set[Path] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BuildError(f"{owner}: evidence[{index}] is not an object")
        raw_path = row.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise BuildError(f"{owner}: evidence[{index}] path is not absolute")
        path = Path(raw_path).resolve()
        if raw_path != str(path):
            raise BuildError(f"{owner}: evidence[{index}] path is not canonical")
        if path in seen:
            raise BuildError(f"{owner}: duplicate evidence path {path}")
        digest = _sha(row.get("sha256"), field=f"{owner}: evidence[{index}]")
        if not path.is_file() or _sha256(path) != digest:
            raise BuildError(f"receipt evidence drift: {path}")
        seen.add(path)
        output.append(path)
    return tuple(output)


def _looks_like_receipt(payload: Mapping[str, Any]) -> bool:
    return {"schema_version", "status", "scope", "evidence"}.issubset(payload)


def _load_receipt(
    path: Path,
    *,
    cache: dict[Path, ReceiptNode],
    active: set[Path],
) -> ReceiptNode:
    path = path.resolve()
    if path in active:
        raise BuildError(f"receipt evidence cycle: {path}")
    if path in cache:
        return cache[path]
    active.add(path)
    try:
        sidecar = _verify_exact_sidecar(path)
        payload = _load_json(path)
        if (
            payload.get("schema_version") not in {1, 2}
            or not isinstance(payload.get("status"), str)
            or not isinstance(payload.get("scope"), str)
        ):
            raise BuildError(f"receipt schema/status/scope mismatch: {path}")
        evidence = _evidence(payload, owner=path)
        node = ReceiptNode(path, sidecar, payload, evidence)
        cache[path] = node
        for item in evidence:
            if item.suffix != ".json":
                continue
            try:
                nested = _load_json(item)
            except BuildError:
                continue
            if _looks_like_receipt(nested):
                _load_receipt(item, cache=cache, active=active)
        return node
    finally:
        active.remove(path)


_CANDIDATE_ROLES = (
    "tts_acceptance_best",
    "tts_engineering_best",
    "l0_best",
    "same_stride_tts_for_l0",
)
_HARD_SAFETY_COLUMNS = (
    "exactness_violations",
    "version_mismatch_count",
    "adaptation_fallback_count",
)


def _require_resolved_candidate_roles(selection: ReceiptNode) -> None:
    """Require every candidate role and a clean hard-safety record.

    Only the L0-versus-TTS ordering may be unresolved.  A screen that failed to
    resolve a role, or that observed an exactness/version/fallback violation,
    still cannot seed a controller phase.
    """

    winners = selection.payload.get("winners")
    if not isinstance(winners, dict):
        raise BuildError("selection receipt has no winners table")
    for role in _CANDIDATE_ROLES:
        row = winners.get(role)
        if not isinstance(row, dict):
            raise BuildError(f"screen did not resolve candidate role: {role}")
        if row.get("eligible") is not True:
            raise BuildError(f"screen candidate role is not eligible: {role}")
        metrics = row.get("metrics_by_context")
        if not isinstance(metrics, dict) or not metrics:
            raise BuildError(f"screen candidate role has no metrics: {role}")
        for context, values in metrics.items():
            if not isinstance(values, dict):
                raise BuildError(f"screen candidate metrics are invalid: {role}")
            for column in _HARD_SAFETY_COLUMNS:
                observed = values.get(column)
                if observed is None:
                    raise BuildError(
                        f"screen candidate {role} lacks {column} at {context}"
                    )
                if observed != 0:
                    raise BuildError(
                        f"screen candidate {role} violates {column} at {context}"
                    )


def _terminal_chain(
    path: Path,
    *,
    allow_l0_not_superior: bool = False,
) -> tuple[ReceiptNode, ReceiptNode, ReceiptNode, dict[Path, ReceiptNode]]:
    """Bind the candidate-screen terminal to its execution and selection.

    ``allow_l0_not_superior`` accepts a screen whose selector resolved every
    candidate role but did not establish that L0 already beats the
    acceptance-best TTS.  That ordering is the downstream hypothesis under test,
    so requiring it here would make the L1/L2 oracle ceiling unmeasurable.  The
    weaker entry condition is recorded in the generation receipt and never
    relaxes an acceptance, utility, or confidence-interval gate.
    """

    path = path.resolve()
    if allow_l0_not_superior:
        terminal_name = "CANDIDATE_SCREEN_BLOCKED.json"
        terminal_status = "candidate_screen_blocked"
        selection_status = "scientifically_blocked"
        conflicts = ("CANDIDATE_SCREEN_SELECTED.json", "PRIORITY_FAILED.json")
    else:
        terminal_name = "CANDIDATE_SCREEN_SELECTED.json"
        terminal_status = "candidate_screen_selected"
        selection_status = "winner_selected"
        conflicts = ("CANDIDATE_SCREEN_BLOCKED.json", "PRIORITY_FAILED.json")
    if path.name != terminal_name:
        raise BuildError(f"builder requires {terminal_name}")
    for name in conflicts:
        conflict = path.with_name(name)
        if conflict.exists() or Path(str(conflict) + ".sha256").exists():
            raise BuildError(f"conflicting candidate-screen terminal: {conflict}")
    cache: dict[Path, ReceiptNode] = {}
    terminal = _load_receipt(path, cache=cache, active=set())
    if (
        terminal.payload.get("schema_version") != 1
        or terminal.payload["status"] != terminal_status
        or terminal.payload["scope"] != "candidate_screen_only_no_claim"
    ):
        raise BuildError(f"terminal is not a {terminal_status} candidate screen")
    children = [cache[item] for item in terminal.evidence if item in cache]
    execution = [node for node in children if node.payload["status"] == "execution_complete"]
    selections = [node for node in children if node.payload["status"] == selection_status]
    if len(execution) != 1 or len(selections) != 1 or len(children) != 2:
        raise BuildError("selected terminal must contain one execution and one selection")
    execution_node, selection_node = execution[0], selections[0]
    expected_objective = not allow_l0_not_superior
    if (
        execution_node.payload.get("schema_version") != 1
        or selection_node.payload.get("schema_version") != 2
        or execution_node.payload["scope"] != "candidate_stride_screen_no_claim"
        or selection_node.payload["scope"] != "candidate_screen_only_no_claim"
        or selection_node.payload.get("objective_screen_pass") is not expected_objective
    ):
        raise BuildError("nested execution/selection status contract mismatch")
    if allow_l0_not_superior:
        _require_resolved_candidate_roles(selection_node)
    expected = {
        execution_node.path,
        execution_node.sidecar,
        selection_node.path,
        selection_node.sidecar,
    }
    if set(terminal.evidence) != expected:
        raise BuildError("terminal direct evidence does not exactly bind its children")
    if execution_node.path.parent != terminal.path.parent:
        raise BuildError("execution receipt is not rooted with the selected terminal")
    if execution_node.path.name != "EXECUTION_COMPLETE.json":
        raise BuildError("nested execution receipt has an unexpected name")
    return terminal, execution_node, selection_node, cache


def _selection_inputs(selection: ReceiptNode) -> tuple[Path, Path, Path, Path]:
    inputs = selection.payload.get("source_inputs")
    keys = {"manifest", "coverage", "vs_static_analysis", "vs_tts_analysis"}
    if not isinstance(inputs, dict) or set(inputs) != keys:
        raise BuildError("selection source_inputs are incomplete")

    def resolved(name: str) -> Path:
        raw = inputs[name]
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise BuildError(f"selection source input {name} is not absolute")
        value = Path(raw).resolve()
        if str(value) != raw:
            raise BuildError(f"selection source input {name} is not canonical")
        return value

    manifest = resolved("manifest")
    coverage = resolved("coverage")
    static_root = resolved("vs_static_analysis")
    tts_root = resolved("vs_tts_analysis")
    expected = {
        TRUSTED_SELECTOR,
        manifest,
        coverage,
        Path(str(coverage) + ".sha256"),
    }
    for root in (static_root, tts_root):
        expected.update(
            {
                root / "p5_long_context_acceptance.parquet",
                root / "analysis-manifest.json",
                root / "analysis-manifest.sha256",
                root / "analysis-hashes.json",
            }
        )
    if set(selection.evidence) != {path.resolve() for path in expected}:
        raise BuildError("selection evidence does not exactly match source_inputs")
    return manifest, coverage, static_root, tts_root


def _load_trusted_selector():
    spec = importlib.util.spec_from_file_location(
        "_trusted_p5_stride_selector", TRUSTED_SELECTOR
    )
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load trusted selector: {TRUSTED_SELECTOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recompute_selection(selection: ReceiptNode, manifest_path: Path) -> dict[str, Any]:
    module = _load_trusted_selector()
    try:
        return module.validate_selection_receipt(
            selector_path=selection.path,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise BuildError(f"selection semantic recomputation failed: {exc}") from exc


def _verify_ledger(root: Path, ledger_path: Path) -> list[Path]:
    ledger = _load_json(ledger_path)
    evidence: list[Path] = [ledger_path.resolve()]
    for relative, row in ledger.items():
        if not isinstance(relative, str) or not relative:
            raise BuildError(f"invalid ledger path in {ledger_path}")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise BuildError(f"ledger path escapes root: {relative}") from exc
        if not isinstance(row, dict):
            raise BuildError(f"invalid ledger row for {candidate}")
        digest = _sha(row.get("sha256"), field=f"ledger {relative}")
        size = row.get("bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not candidate.is_file()
            or candidate.stat().st_size != size
            or _sha256(candidate) != digest
        ):
            raise BuildError(f"ledger evidence drift: {candidate}")
        evidence.append(candidate)
    return evidence


def _verify_analysis(
    root: Path,
    *,
    baseline: str,
    source: ExperimentManifest,
) -> tuple[list[dict[str, Any]], list[Path]]:
    root = root.resolve()
    manifest_path = root / "analysis-manifest.json"
    sidecar = _verify_exact_sidecar(
        manifest_path, root / "analysis-manifest.sha256"
    )
    ledger_path = root / "analysis-hashes.json"
    ledger_evidence = _verify_ledger(root, ledger_path)
    payload = _load_json(manifest_path)
    if payload.get("schema_version") != 1:
        raise BuildError(f"analysis schema mismatch: {root}")
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise BuildError(f"analysis contract is missing: {root}")
    expected_analysis = {
        "baseline": baseline,
        "expected_manifest_sha256": source.content_sha256(),
        "weight_update_mode_overlay": "lora",
        "methods_overlay": ["static", "tts", "naive_async"],
        "lifecycles_overlay": None,
        "learning_rate_overlay": None,
    }
    for key, value in expected_analysis.items():
        if analysis.get(key) != value:
            raise BuildError(f"{root}: analysis {key} mismatch")
    rows = payload.get("input_runs")
    if not isinstance(rows, list) or len(rows) != len(source.units):
        raise BuildError(f"{root}: analysis input run cardinality mismatch")
    unit_ids = [row.get("unit_id") for row in rows if isinstance(row, dict)]
    run_ids = [row.get("run_id") for row in rows if isinstance(row, dict)]
    if (
        len(unit_ids) != len(rows)
        or set(unit_ids) != {unit.unit_id for unit in source.units}
        or len(set(run_ids)) != len(rows)
        or not all(isinstance(value, str) and value for value in run_ids)
    ):
        raise BuildError(f"{root}: analysis input run identity mismatch")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "run_id",
            "unit_id",
            "manifest_sha256",
            "hashes_sha256",
        }:
            raise BuildError(f"{root}: analysis input run schema mismatch")
        _sha(row.get("manifest_sha256"), field="analysis manifest_sha256")
        _sha(row.get("hashes_sha256"), field="analysis hashes_sha256")
    derived = payload.get("derived_outputs")
    if not isinstance(derived, dict) or "p5_long_context_acceptance.parquet" not in derived:
        raise BuildError(f"{root}: required P5 derived table is absent")
    ledger = _load_json(ledger_path)
    expected_keys = {
        *derived,
        "analysis-manifest.json",
        "analysis-manifest.sha256",
    }
    if set(ledger) != expected_keys:
        raise BuildError(f"{root}: analysis ledger is not transitively closed")
    for relative, record in derived.items():
        if ledger.get(relative) != record:
            raise BuildError(f"{root}: derived output ledger mismatch for {relative}")
    return rows, [manifest_path, sidecar, *ledger_evidence]


def _verify_coverage(path: Path, source: ExperimentManifest) -> list[Path]:
    sidecar = _verify_exact_sidecar(path)
    payload = _load_json(path)
    cells = payload.get("cells")
    if not isinstance(cells, dict) or set(cells) != {
        unit.unit_id for unit in source.units
    }:
        raise BuildError("coverage does not contain the exact source unit set")
    if not all(isinstance(cell, dict) for cell in cells.values()):
        raise BuildError("coverage contains a non-object cell")
    statuses = {unit_id: cell.get("status") for unit_id, cell in cells.items()}
    if set(statuses.values()) != {"complete_valid"}:
        raise BuildError("stride screen coverage is not entirely complete_valid")
    expected = build_coverage(source.expected_units(), statuses)
    expected_payload = {"cells": expected.cells, "summary": expected.summary()}
    if payload != expected_payload:
        raise BuildError("coverage dimensions or summary differ from recomputation")
    return [path.resolve(), sidecar]


def _verify_execution_contract(
    execution: ReceiptNode,
    *,
    selection: ReceiptNode,
    manifest_path: Path,
    coverage_path: Path,
    static_root: Path,
    tts_root: Path,
    lockfile: Path,
    model_roots: Path,
) -> tuple[Path, Path]:
    source_sidecar = _verify_manifest_sidecar(manifest_path)
    lock_sidecar = _verify_exact_sidecar(lockfile)
    roots_sidecar = _verify_exact_sidecar(model_roots)
    coverage_sidecar = _verify_exact_sidecar(coverage_path)
    expected = {
        TRUSTED_QUEUE,
        TRUSTED_SELECTOR,
        manifest_path.resolve(),
        source_sidecar,
        lockfile.resolve(),
        lock_sidecar,
        model_roots.resolve(),
        roots_sidecar,
        coverage_path.resolve(),
        coverage_sidecar,
        (static_root / "analysis-hashes.json").resolve(),
        (tts_root / "analysis-hashes.json").resolve(),
    }
    observed = set(execution.evidence)
    if not expected.issubset(observed):
        raise BuildError("execution receipt omits a required frozen input")
    remaining = observed - expected
    if len(remaining) != 2:
        raise BuildError("execution receipt has unexpected evidence roles")
    dataset = next(
        (path for path in remaining if Path(str(path) + ".sha256") in remaining),
        None,
    )
    if dataset is None:
        raise BuildError("execution receipt lacks the dataset receipt pair")
    dataset_sidecar = _verify_exact_sidecar(dataset)
    if remaining != {dataset, dataset_sidecar}:
        raise BuildError("execution dataset evidence is not an exact pair")
    _load_json(dataset)
    if TRUSTED_SELECTOR not in selection.evidence:
        raise BuildError("selection is not bound to the trusted selector source")
    return dataset, dataset_sidecar


def _validate_runtime_fingerprint(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise BuildError(f"{label} runtime fingerprint schema mismatch")
    claimed = _sha(value.get("sha256"), field=f"{label} runtime fingerprint")
    body = dict(value)
    body.pop("sha256", None)
    if sha256_json(body) != claimed:
        raise BuildError(f"{label} runtime fingerprint digest mismatch")
    if not isinstance(value.get("files"), dict) or not isinstance(
        value.get("locked_reference"), dict
    ):
        raise BuildError(f"{label} runtime fingerprint is incomplete")
    return json.loads(canonical_json(value))


def _runtime_transition(
    screen: dict[str, Any], consumer: dict[str, Any]
) -> dict[str, Any]:
    screen_files = screen["files"]
    consumer_files = consumer["files"]
    common = set(screen_files) & set(consumer_files)
    changed = sorted(
        path
        for path in common
        if screen_files[path] != consumer_files[path]
    )
    added = sorted(set(consumer_files) - set(screen_files))
    removed = sorted(set(screen_files) - set(consumer_files))
    locked_reference_changed = (
        screen["locked_reference"] != consumer["locked_reference"]
    )
    equal = screen == consumer
    if equal:
        authorization_id = "identical_runtime"
    else:
        authorization_id = controller_builder.exact_runtime_transition_authorization(
            screen_sha256=screen["sha256"],
            consumer_sha256=consumer["sha256"],
            changed_files=changed,
            added_files=added,
            removed_files=removed,
            locked_reference_changed=locked_reference_changed,
        )
    if authorization_id is None:
        raise BuildError(
            "screen/consumer runtime drift has no reviewed exact hash-pair "
            "authorization; run a fresh screen under the consumer runtime "
            f"(changed={changed}, added={added}, removed={removed}, "
            f"locked_reference_changed={locked_reference_changed})"
        )
    return {
        "schema_version": 2,
        "screen_sha256": screen["sha256"],
        "consumer_sha256": consumer["sha256"],
        "equal": equal,
        "changed_files": changed,
        "added_files": added,
        "removed_files": removed,
        "locked_reference_changed": locked_reference_changed,
        "authorization_id": authorization_id,
        "authorization_basis": "exact_runtime_fingerprint_pair",
        "screen_measurements_reusable": equal,
        "selection_reuse_only": not equal,
        "scientific_equivalence_claim": False,
        "requires_matched_confirmation": not equal,
    }


def _verify_run_provenance(
    artifact_root: Path,
    rows: list[dict[str, Any]],
    *,
    source: ExperimentManifest,
    lockfile_path: Path,
    model_roots_path: Path,
    target_revision: str,
    drafter_revision: str,
 ) -> tuple[list[Path], dict[str, Any]]:
    evidence: list[Path] = []
    screen_runtime: dict[str, Any] | None = None
    by_unit = {unit.unit_id: unit for unit in source.units}
    for row in rows:
        run_id = str(row["run_id"])
        if Path(run_id).name != run_id:
            raise BuildError(f"analysis run_id is not a basename: {run_id}")
        run_root = (artifact_root / run_id).resolve()
        manifest_path = run_root / "manifest.json"
        hashes_path = run_root / "hashes.json"
        if (
            not manifest_path.is_file()
            or _sha256(manifest_path) != row["manifest_sha256"]
            or not hashes_path.is_file()
            or _sha256(hashes_path) != row["hashes_sha256"]
        ):
            raise BuildError(f"analysis input run hash drift: {run_root}")
        run = _load_json(manifest_path)
        unit = by_unit.get(str(row["unit_id"]))
        if unit is None or run.get("unit_id") != unit.unit_id or run.get("run_id") != run_id:
            raise BuildError(f"analysis input run identity drift: {run_root}")
        engine = run.get("engine_params")
        if not isinstance(engine, dict):
            raise BuildError(f"run engine parameters are missing: {run_root}")
        checks = {
            "lockfile_path": str(lockfile_path),
            "model_roots_path": str(model_roots_path),
            "locked_target_revision": target_revision,
            "locked_drafter_revision": drafter_revision,
            "pytorch_cuda_alloc_conf": MATCHED_PYTORCH_CUDA_ALLOC_CONF,
        }
        for key, value in checks.items():
            if engine.get(key) != value:
                raise BuildError(f"run {run_id} has mismatched {key}")
        observed_runtime = _validate_runtime_fingerprint(
            engine.get("runtime_implementation_fingerprint"),
            label=f"run {run_id}",
        )
        if screen_runtime is None:
            screen_runtime = observed_runtime
        elif observed_runtime != screen_runtime:
            raise BuildError("stride-screen runs used different runtime fingerprints")
        if float(engine.get("lr", -1)) != 1e-4 or float(
            engine.get("weight_decay", -1)
        ) != 1e-2:
            raise BuildError(f"run {run_id} optimizer tier mismatch")
        ledger_evidence = _verify_ledger(run_root, hashes_path)
        ledger = _load_json(hashes_path)
        if (
            ledger.get("manifest.json", {}).get("sha256") != _sha256(manifest_path)
            or "manifest.sha256" not in ledger
            or "exit.json" not in ledger
        ):
            raise BuildError(f"run ledger lacks immutable completion files: {run_root}")
        exit_payload = _load_json(run_root / "exit.json")
        if exit_payload.get("status") != "complete_valid":
            raise BuildError(f"analysis input run is not complete_valid: {run_root}")
        evidence.extend([manifest_path, *ledger_evidence])
    if screen_runtime is None:
        raise BuildError("stride-screen analysis contains no run provenance")
    return evidence, screen_runtime


def _locked_inputs(lockfile_path: Path, model_roots_path: Path):
    try:
        lock = load_lockfile(lockfile_path)
        roots = load_model_roots(model_roots_path)
    except LockError as exc:
        raise BuildError(str(exc)) from exc
    pair = MODEL_PAIRS["qwen3_4b_dflash16"]
    target = lock.find_snapshot(pair["target"])
    drafter = lock.find_snapshot(pair["drafter"])
    for repo_id in (pair["target"], pair["drafter"]):
        root = roots.get(repo_id)
        if not isinstance(root, str) or not Path(root).is_dir():
            raise BuildError(f"model-roots lacks an existing {repo_id} snapshot")
    compiler = lock.environment.compiler_versions or {}
    locked_reference = {
        key: compiler[key]
        for key in (
            "lightcone_runtime_source_sha256",
            "sglang_runtime_source_sha256",
            "sglang_fork_commit",
            "sglang_fork_dirty",
        )
        if key in compiler
    }
    try:
        runtime = runtime_implementation_fingerprint(
            locked_reference=locked_reference
        )
    except ConfigError as exc:
        raise BuildError(f"cannot fingerprint current runtime: {exc}") from exc
    revisions = {
        "target": target.snapshot_sha,
        "drafter": drafter.snapshot_sha,
        "tokenizer": target.snapshot_sha,
    }
    return lock, roots, revisions, runtime


def _cas_text(path: Path, text: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise BuildError(f"immutable output collision: {path}")
        return
    _atomic_text(path, text)


def _publish_manifest(manifest: ExperimentManifest, path: Path) -> dict[str, Any]:
    path = path.resolve()
    body = canonical_json(manifest.to_dict())
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    if sidecar.exists() and not path.exists():
        raise BuildError(f"orphan manifest sidecar: {sidecar}")
    _cas_text(path, body)
    _cas_text(sidecar, digest + "\n")
    if _sha256(path) != digest or sidecar.read_text().strip() != digest:
        raise BuildError(f"generated manifest CAS verification failed: {path}")
    loaded = ExperimentManifest.load(path)
    if loaded.to_dict() != manifest.to_dict():
        raise BuildError(f"generated manifest round-trip mismatch: {path}")
    return {
        "path": str(path),
        "sha256": digest,
        "manifest_sha256": manifest.content_sha256(),
        "sidecar_path": str(sidecar),
        "sidecar_sha256": _sha256(sidecar),
        "name": manifest.name,
        "unit_ids": [unit.unit_id for unit in manifest.units],
        "methods": sorted({unit.method for unit in manifest.units}),
    }


def _write_receipt(
    path: Path,
    payload: dict[str, Any],
    *,
    evidence: Iterable[Path],
) -> dict[str, Any]:
    rows = [
        {"path": str(item), "sha256": _sha256(item)}
        for item in sorted({Path(value).resolve() for value in evidence})
    ]
    body = {**payload, "evidence": rows}
    text = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = path.resolve()
    sidecar = Path(str(path) + ".sha256")
    if sidecar.exists() and not path.exists():
        raise BuildError(f"orphan generation receipt sidecar: {sidecar}")
    _cas_text(path, text)
    _cas_text(sidecar, _sha256(path) + "\n")
    if sidecar.read_text().strip() != _sha256(path):
        raise BuildError(f"generation receipt CAS verification failed: {path}")
    return body


@contextmanager
def _output_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".matched-controller-manifests.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BuildError(f"matched-manifest output is already locked: {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalized_cell(unit) -> dict[str, Any]:
    row = unit.to_manifest_dict()
    for key in ("phase", "method", "contention_condition", "unit_id"):
        row.pop(key)
    return row


def validate_selected_screen_inputs(
    *,
    selected_receipt: Path,
    lockfile: Path,
    model_roots: Path,
    allow_l0_not_superior: bool = False,
) -> ValidatedSelectedScreenInputs:
    """Validate the selected screen through receipts, ledgers, and raw runs.

    This function performs no writes and is the sole downstream trust boundary
    for both confirmation and controller-manifest builders.
    """

    selected_receipt = selected_receipt.resolve()
    lockfile = lockfile.resolve()
    model_roots = model_roots.resolve()
    terminal, execution, selection_node, receipt_nodes = _terminal_chain(
        selected_receipt, allow_l0_not_superior=allow_l0_not_superior
    )
    manifest_path, coverage_path, static_root, tts_root = _selection_inputs(
        selection_node
    )
    _verify_execution_contract(
        execution,
        selection=selection_node,
        manifest_path=manifest_path,
        coverage_path=coverage_path,
        static_root=static_root,
        tts_root=tts_root,
        lockfile=lockfile,
        model_roots=model_roots,
    )
    try:
        source = ExperimentManifest.load(manifest_path)
    except ConfigError as exc:
        raise BuildError(f"invalid source manifest: {exc}") from exc
    coverage_evidence = _verify_coverage(coverage_path, source)
    static_rows, static_evidence = _verify_analysis(
        static_root, baseline="static", source=source
    )
    tts_rows, tts_evidence = _verify_analysis(
        tts_root, baseline="tts", source=source
    )
    if static_rows != tts_rows:
        raise BuildError("vs-static/vs-tts analyses do not bind identical input runs")
    selection = _recompute_selection(selection_node, manifest_path)
    locked, _roots, revisions, runtime = _locked_inputs(lockfile, model_roots)
    run_evidence, screen_runtime = _verify_run_provenance(
        execution.path.parent,
        static_rows,
        source=source,
        lockfile_path=lockfile,
        model_roots_path=model_roots,
        target_revision=revisions["target"],
        drafter_revision=revisions["drafter"],
    )
    transition = _runtime_transition(screen_runtime, runtime)
    static_unit = next(unit for unit in source.units if unit.method == "static")
    binding_hashes = {
        "terminal_receipt_sha256": _sha256(terminal.path),
        "execution_receipt_sha256": _sha256(execution.path),
        "selection_receipt_sha256": _sha256(selection_node.path),
        "source_manifest_file_sha256": _sha256(manifest_path),
        "source_manifest_sha256": source.content_sha256(),
        "lockfile_sha256": locked.content_sha256(),
        "model_roots_sha256": _sha256(model_roots),
        "screen_runtime_implementation_sha256": screen_runtime["sha256"],
        "consumer_runtime_implementation_sha256": runtime["sha256"],
        "queue_source_sha256": _sha256(TRUSTED_QUEUE),
        "selector_source_sha256": _sha256(TRUSTED_SELECTOR),
    }
    all_evidence: list[Path] = []
    for node in receipt_nodes.values():
        all_evidence.extend([node.path, node.sidecar, *node.evidence])
    all_evidence.extend(
        [
            *coverage_evidence,
            *static_evidence,
            *tts_evidence,
            *run_evidence,
            lockfile,
            Path(str(lockfile) + ".sha256"),
            model_roots,
            Path(str(model_roots) + ".sha256"),
            TRUSTED_QUEUE,
            TRUSTED_SELECTOR,
        ]
    )
    return ValidatedSelectedScreenInputs(
        terminal_receipt=terminal.path,
        execution_receipt=execution.path,
        selection_receipt=selection_node.path,
        selection=selection,
        source_manifest_path=manifest_path,
        source_manifest=source,
        lockfile_path=lockfile,
        lockfile_sha256=locked.content_sha256(),
        model_roots_path=model_roots,
        model_roots_sha256=_sha256(model_roots),
        model_revisions=revisions,
        screen_runtime_implementation_fingerprint=screen_runtime,
        runtime_implementation_fingerprint=runtime,
        runtime_transition=transition,
        pytorch_cuda_alloc_conf=str(
            source.engine_params.get("pytorch_cuda_alloc_conf", "")
        ),
        static_unit_id=static_unit.unit_id,
        binding_hashes=binding_hashes,
        evidence=tuple(sorted(set(path.resolve() for path in all_evidence))),
    )


def build(
    *,
    selected_receipt: Path,
    lockfile: Path,
    model_roots: Path,
    output_dir: Path,
    generation_receipt: Path | None = None,
    allow_l0_not_superior: bool = False,
) -> dict[str, Any]:
    validated = validate_selected_screen_inputs(
        selected_receipt=selected_receipt,
        lockfile=lockfile,
        model_roots=model_roots,
        allow_l0_not_superior=allow_l0_not_superior,
    )
    source = validated.source_manifest
    selection = validated.selection
    lockfile = validated.lockfile_path
    model_roots = validated.model_roots_path
    revisions = validated.model_revisions
    runtime = validated.runtime_implementation_fingerprint
    helper_path = Path(controller_builder.__file__).resolve()
    bindings = {
        "schema_version": 1,
        **validated.binding_hashes,
        "source_static_unit_id": validated.static_unit_id,
        "model_revisions": revisions,
        "pytorch_cuda_alloc_conf": validated.pytorch_cuda_alloc_conf,
        "screen_runtime_implementation_fingerprint": (
            validated.screen_runtime_implementation_fingerprint
        ),
        "runtime_implementation_fingerprint": runtime,
        "runtime_transition": validated.runtime_transition,
        "builder_source_sha256": _sha256(Path(__file__).resolve()),
        "helper_source_sha256": _sha256(helper_path),
        "screen_entry_condition": (
            "l0_not_superior_oracle_scope"
            if allow_l0_not_superior
            else "objective_screen_pass"
        ),
        "contention_mapping": {
            "phase1_naive_async": "realistic_async",
            "phase1_tts": "none",
            "phase2_lc_transport": "realistic_async",
        },
    }
    try:
        matched = build_matched_controller_manifests(
            selection,
            source,
            bindings=bindings,
            allow_l0_not_superior=allow_l0_not_superior,
        )
    except ConfigError as exc:
        raise BuildError(f"matched manifest identity check failed: {exc}") from exc

    # Prompt windows are part of the scientific identity, not queue-local
    # flags.  Phase 1 trains the gate/damper on [88,136); phase 2 evaluates
    # transport on the disjoint [136,184) holdout.
    prompt_windows = {
        "phase1_trace": {"offset": 88, "limit": 48, "half_open": [88, 136]},
        "phase2_l3": {"offset": 136, "limit": 48, "half_open": [136, 184]},
    }
    identity_body = dict(matched.identity)
    identity_body.pop("sha256", None)
    identity_body["prompt_windows"] = prompt_windows
    identity = {**identity_body, "sha256": sha256_json(identity_body)}

    def bind_window(manifest, *, role: str):
        window = prompt_windows[role]
        old_phase = manifest.phase
        new_phase = f"{old_phase.rsplit('_b', 1)[0]}_b{identity['sha256'][:12]}_v2"
        engine = {
            **manifest.engine_params,
            "prompt_offset": window["offset"],
            "prompt_limit": window["limit"],
            "controller_prompt_windows": prompt_windows,
            "matched_controller_identity_sha256": identity["sha256"],
        }
        return replace(
            manifest,
            name=new_phase,
            phase=new_phase,
            engine_params=engine,
            units=[replace(unit, phase=new_phase) for unit in manifest.units],
        )

    matched = replace(
        matched,
        identity=identity,
        trace=bind_window(matched.trace, role="phase1_trace"),
        l3_phase2=bind_window(matched.l3_phase2, role="phase2_l3"),
    )
    phase2_reference_name = (
        f"p5_priority_dflash_l3_tts_reference_"
        f"b{identity['sha256'][:12]}_v2"
    )
    phase2_reference_engine = {
        **matched.trace.engine_params,
        "prompt_offset": prompt_windows["phase2_l3"]["offset"],
        "prompt_limit": prompt_windows["phase2_l3"]["limit"],
        "trace_producer_methods": ["tts"],
        "phase2_tts_reference_only": True,
        "matched_controller_identity_sha256": identity["sha256"],
    }
    phase2_tts_reference = replace(
        matched.trace,
        name=phase2_reference_name,
        phase=phase2_reference_name,
        description=(
            "Held-out phase-2 TTS reference on [136,184), paired exactly "
            "with L3 transport and excluded from phase-1 controller fitting."
        ),
        engine_params=phase2_reference_engine,
        units=[
            replace(unit, phase=phase2_reference_name)
            for unit in matched.trace.units
            if unit.method == "tts"
        ],
    )

    output_dir = output_dir.resolve()
    with _output_lock(output_dir):
        trace_path = output_dir / f"{matched.trace.name}.json"
        l3_path = output_dir / f"{matched.l3_phase2.name}.json"
        phase2_tts_path = output_dir / f"{phase2_tts_reference.name}.json"
        trace_record = _publish_manifest(matched.trace, trace_path)
        l3_record = _publish_manifest(matched.l3_phase2, l3_path)
        phase2_tts_record = _publish_manifest(
            phase2_tts_reference, phase2_tts_path
        )
        trace_cells = sorted(
            (_normalized_cell(unit) for unit in matched.trace.units if unit.method == "tts"),
            key=lambda row: row["concurrency"],
        )
        l3_cells = sorted(
            (_normalized_cell(unit) for unit in matched.l3_phase2.units),
            key=lambda row: row["concurrency"],
        )
        reference_cells = sorted(
            (_normalized_cell(unit) for unit in phase2_tts_reference.units),
            key=lambda row: row["concurrency"],
        )
        if trace_cells != l3_cells or l3_cells != reference_cells:
            raise BuildError("published L3/TTS cells do not exactly mirror phase-1 TTS")
        mirror = {
            "exact": True,
            "normalized_cells_sha256": sha256_json(trace_cells),
            "prompt_windows": prompt_windows,
            "prompt_windows_disjoint": True,
            "unit_allowed_differences": [
                "phase",
                "method",
                "contention_condition",
                "unit_id",
            ],
            "engine_allowed_differences": [
                "trace_producer_methods",
                "l3_evaluation_only",
                "prompt_offset",
            ],
            "contention_mapping": bindings["contention_mapping"],
        }
        receipt_path = (
            generation_receipt.resolve()
            if generation_receipt is not None
            else output_dir
            / f"matched-controller-manifests-b{matched.identity['sha256'][:12]}.json"
        )
        generated = [
            Path(trace_record["path"]),
            Path(trace_record["sidecar_path"]),
            Path(l3_record["path"]),
            Path(l3_record["sidecar_path"]),
            Path(phase2_tts_record["path"]),
            Path(phase2_tts_record["sidecar_path"]),
        ]
        payload = {
            "schema_version": 2,
            "status": "matched_controller_manifests_generated",
            "terminal": {
                "path": str(validated.terminal_receipt),
                "sha256": bindings["terminal_receipt_sha256"],
            },
            "execution": {
                "path": str(validated.execution_receipt),
                "sha256": bindings["execution_receipt_sha256"],
            },
            "selection": {
                "path": str(validated.selection_receipt),
                "sha256": bindings["selection_receipt_sha256"],
                "semantics_recomputed": True,
            },
            "source_manifest": {
                "path": str(validated.source_manifest_path),
                "file_sha256": bindings["source_manifest_file_sha256"],
                "manifest_sha256": bindings["source_manifest_sha256"],
                "static_unit_id": validated.static_unit_id,
            },
            "locked_inputs": {
                "lockfile": str(lockfile),
                "lockfile_sha256": bindings["lockfile_sha256"],
                "model_roots": str(model_roots),
                "model_roots_sha256": bindings["model_roots_sha256"],
                "model_revisions": revisions,
            },
            "runtime_transition": validated.runtime_transition,
            "controller_identity": matched.identity,
            "controller_identity_sha256": matched.identity["sha256"],
            "mirror_contract": mirror,
            "artifacts": {
                "TRACE_MATCHED": trace_record,
                "L3_PHASE2_MATCHED": l3_record,
                "L3_PHASE2_TTS_REFERENCE": phase2_tts_record,
            },
        }
        result = _write_receipt(
            receipt_path,
            payload,
            evidence=[
                *validated.evidence,
                Path(__file__).resolve(),
                helper_path,
                *generated,
            ],
        )
        # Final re-read under the output lock closes check-then-replace races.
        for record in result["artifacts"].values():
            artifact = Path(record["path"])
            if _sha256(artifact) != record["sha256"]:
                raise BuildError(f"generated artifact changed before commit: {artifact}")
        if _load_json(receipt_path) != result:
            raise BuildError("generation receipt changed before commit")
        _verify_exact_sidecar(receipt_path)
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-receipt", type=Path, required=True)
    parser.add_argument("--lockfile", type=Path, required=True)
    parser.add_argument("--model-roots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation-receipt", type=Path)
    parser.add_argument(
        "--allow-l0-not-superior-oracle-scope",
        action="store_true",
        help=(
            "accept a candidate screen that resolved every role but did not "
            "establish L0 over the acceptance-best TTS; the L0/L1/L2 ordering "
            "is the downstream hypothesis, so this only widens the entry "
            "condition and never relaxes an acceptance or utility gate"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build(
            selected_receipt=args.selected_receipt,
            lockfile=args.lockfile,
            model_roots=args.model_roots,
            output_dir=args.output_dir,
            generation_receipt=args.generation_receipt,
            allow_l0_not_superior=args.allow_l0_not_superior_oracle_scope,
        )
    except BuildError as exc:
        raise SystemExit(f"matched controller manifest build failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
