#!/usr/bin/env python3
"""Build and compare receipt-selected P5 stride confirmation experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from lightcone_spec.artifacts.rundir import REQUIRED_FILES
from lightcone_spec.locking.hashing import canonical_json, sha256_file, sha256_json
from lightcone_spec.orchestration.catalog import (
    P5_PRIORITY_CONFIRMATION_LOADS,
    P5_PRIORITY_CONFIRMATION_MIN_PROMPT_CLUSTERS,
    P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF,
    P5_PRIORITY_STRIDE_CANDIDATES,
    p5_priority_dflash_stride_confirmation_manifest,
    p5_priority_dflash_stride_screen_manifest,
)
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.units import RunUnit
from lightcone_spec.statistics.tables import (
    P5_IDENTITY_COLUMNS,
    paired_cross_stride_acceptance_table,
)


class ConfirmationError(RuntimeError):
    pass


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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ConfirmationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfirmationError(f"JSON must be an object: {path}")
    return value


def _verify_sidecar(path: Path, sidecar: Path | None = None) -> Path:
    sidecar = sidecar or Path(str(path) + ".sha256")
    if not path.is_file():
        raise ConfirmationError(f"missing attested file: {path}")
    if not sidecar.is_file():
        raise ConfirmationError(f"missing SHA-256 sidecar: {sidecar}")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ConfirmationError(f"SHA-256 sidecar mismatch: {path}")
    return sidecar


def _verify_receipt(path: Path) -> tuple[dict[str, Any], list[Path]]:
    sidecar = _verify_sidecar(path)
    payload = _load_json(path)
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ConfirmationError(f"receipt has no evidence: {path}")
    evidence = []
    observed: set[Path] = set()
    for raw in raw_evidence:
        if not isinstance(raw, dict):
            raise ConfirmationError(f"receipt evidence row is invalid: {path}")
        evidence_path = Path(str(raw.get("path", ""))).resolve()
        declared = raw.get("sha256")
        if evidence_path in observed:
            raise ConfirmationError(f"receipt repeats evidence: {evidence_path}")
        if not evidence_path.is_file() or declared != sha256_file(evidence_path):
            raise ConfirmationError(
                f"receipt evidence hash mismatch: {evidence_path}"
            )
        observed.add(evidence_path)
        evidence.append(evidence_path)
    return payload, [path, sidecar, *evidence]


def _selector_module():
    script = Path(__file__).with_name("select_p5_stride_screen.py")
    module_name = "_lightcone_p5_stride_selector"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise ConfirmationError(f"cannot load selector implementation: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _semantically_validate_selector(
    selector_path: Path, manifest_path: Path
) -> dict[str, Any]:
    module = _selector_module()
    try:
        return module.validate_selection_receipt(
            selector_path=selector_path,
            manifest_path=manifest_path,
        )
    except module.SelectionError as exc:
        raise ConfirmationError(f"selector semantic validation failed: {exc}") from exc


def _matched_builder_module():
    script = Path(__file__).with_name("build_p5_matched_controller_manifests.py")
    module_name = "_lightcone_p5_matched_builder"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise ConfirmationError(f"cannot load terminal validator: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _validated_terminal_context(
    *, selected_terminal: Path, lockfile: Path, model_roots: Path
) -> dict[str, Any]:
    """Reuse the controller builder's full terminal/run/lock validation."""

    validator = _matched_builder_module()
    try:
        validated = validator.validate_selected_screen_inputs(
            selected_receipt=selected_terminal,
            lockfile=lockfile,
            model_roots=model_roots,
        )
    except Exception as exc:
        raise ConfirmationError(f"selected-terminal validation failed: {exc}") from exc

    validator_path = Path(validator.__file__).resolve()
    bindings = {
        **validated.binding_hashes,
        "model_revisions": validated.model_revisions,
        "runtime_implementation_fingerprint": (
            validated.runtime_implementation_fingerprint
        ),
        "pytorch_cuda_alloc_conf": validated.pytorch_cuda_alloc_conf,
        "terminal_validator_source_sha256": sha256_file(validator_path),
        "confirmation_builder_source_sha256": sha256_file(Path(__file__)),
    }
    return {
        "selection": validated.selection,
        "selection_path": validated.selection_receipt,
        "source": validated.source_manifest,
        "source_path": validated.source_manifest_path,
        "bindings": bindings,
        "evidence": [
            *validated.evidence,
            validator_path,
            Path(__file__).resolve(),
        ],
    }


def _write_receipt(
    path: Path, payload: dict[str, Any], evidence: Iterable[Path]
) -> dict[str, Any]:
    if "evidence" in payload:
        raise ConfirmationError("receipt payload must not predeclare evidence")
    paths = sorted({Path(item).resolve() for item in evidence})
    for evidence_path in paths:
        if not evidence_path.is_file():
            raise ConfirmationError(f"receipt evidence is missing: {evidence_path}")
    body = {
        **payload,
        "evidence": [
            {"path": str(item), "sha256": sha256_file(item)} for item in paths
        ],
    }
    text = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != text:
        raise ConfirmationError(
            f"receipt {path} exists with different content; choose a new path"
        )
    _atomic_text(path, text)
    _atomic_text(Path(str(path) + ".sha256"), sha256_file(path) + "\n")
    return body


def _screen_unit_index(manifest: ExperimentManifest) -> dict[tuple[str, int], str]:
    return {(unit.method, unit.stride): unit.unit_id for unit in manifest.units}


def _validate_screen_manifest(path: Path) -> tuple[ExperimentManifest, Path]:
    sidecar = _verify_sidecar(path)
    manifest = ExperimentManifest.load(path)
    expected = p5_priority_dflash_stride_screen_manifest()
    expected = replace(expected, lockfile_sha256=manifest.lockfile_sha256)
    observed = json.loads(canonical_json(manifest.to_dict()))
    if observed != expected.to_dict():
        raise ConfirmationError("source manifest is not the frozen P5 stride screen")
    return manifest, sidecar


def _winner(
    selector: dict[str, Any],
    *,
    name: str,
    method: str,
    screen_index: dict[tuple[str, int], str],
) -> dict[str, Any]:
    winners = selector.get("winners")
    row = winners.get(name) if isinstance(winners, dict) else None
    if not isinstance(row, dict):
        raise ConfirmationError(f"selector lacks winner {name}")
    stride = row.get("stride")
    if isinstance(stride, bool) or stride not in P5_PRIORITY_STRIDE_CANDIDATES:
        raise ConfirmationError(f"selector winner {name} has invalid stride")
    if row.get("method") != method or row.get("eligible") is not True:
        raise ConfirmationError(f"selector winner {name} is not eligible {method}")
    expected_id = screen_index.get((method, int(stride)))
    if row.get("unit_id") != expected_id:
        raise ConfirmationError(f"selector winner {name} unit identity mismatch")
    return row


def _validate_selector(
    selector_path: Path, screen_path: Path
) -> tuple[dict[str, Any], dict[str, Any], ExperimentManifest, list[Path]]:
    screen, screen_sidecar = _validate_screen_manifest(screen_path)
    semantic_selector = _semantically_validate_selector(
        selector_path, screen_path
    )
    raw_selector, evidence = _verify_receipt(selector_path)
    module = _selector_module()
    try:
        selector = module.canonicalize_selection(raw_selector)
    except module.SelectionError as exc:
        raise ConfirmationError(f"selector migration failed: {exc}") from exc
    if selector != semantic_selector:
        raise ConfirmationError("selector changed during semantic validation")
    if (
        selector.get("schema_version") != 2
        or selector.get("status") != "winner_selected"
        or selector.get("scope") != "candidate_screen_only_no_claim"
        or selector.get("objective_screen_pass") is not True
    ):
        raise ConfirmationError("selector did not pass the candidate screen")
    evidence_hashes = {
        Path(str(row["path"])).resolve(): row["sha256"]
        for row in selector["evidence"]
    }
    if evidence_hashes.get(screen_path.resolve()) != sha256_file(screen_path):
        raise ConfirmationError("selector is not bound to the source screen manifest")

    index = _screen_unit_index(screen)
    tts_acceptance = _winner(
        selector,
        name="tts_acceptance_best",
        method="tts",
        screen_index=index,
    )
    tts_engineering = _winner(
        selector,
        name="tts_engineering_best",
        method="tts",
        screen_index=index,
    )
    l0 = _winner(
        selector,
        name="l0_best",
        method="naive_async",
        screen_index=index,
    )
    same = _winner(
        selector,
        name="same_stride_tts_for_l0",
        method="tts",
        screen_index=index,
    )
    if int(same["stride"]) != int(l0["stride"]):
        raise ConfirmationError("same-stride TTS does not match the L0 winner")
    expected_sources = {
        index[("static", 1)],
        str(tts_acceptance["unit_id"]),
        str(tts_engineering["unit_id"]),
        str(l0["unit_id"]),
        str(same["unit_id"]),
    }
    declared_sources = selector.get("confirmation_unit_ids")
    if not isinstance(declared_sources, list) or set(declared_sources) != expected_sources:
        raise ConfirmationError("selector confirmation source units are inconsistent")
    selected = {
        "tts_acceptance_stride": int(tts_acceptance["stride"]),
        "tts_engineering_stride": int(tts_engineering["stride"]),
        "tts_engineering_eligible": (
            tts_engineering.get("engineering_eligible") is True
        ),
        "tts_engineering_fallback_reason": tts_engineering.get(
            "engineering_fallback_reason"
        ),
        "l0_stride": int(l0["stride"]),
        "source_unit_ids": sorted(expected_sources),
    }
    return selector, selected, screen, [*evidence, screen_path, screen_sidecar]


def _role_units(
    manifest: ExperimentManifest, *, method: str, stride: int
) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": unit.unit_id,
            "prompt_subset": unit.prompt_subset,
            "concurrency": unit.concurrency,
        }
        for unit in manifest.units
        if unit.method == method and unit.stride == stride
    ]


def _analysis_contract(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparisons": {
            "tts_acceptance_best_vs_static": {
                "candidate_method": "tts",
                "candidate_update_stride": selected["tts_acceptance_stride"],
                "baseline_method": "static",
                "baseline_update_stride": 1,
            },
            "tts_engineering_best_vs_static": {
                "candidate_method": "tts",
                "candidate_update_stride": selected["tts_engineering_stride"],
                "baseline_method": "static",
                "baseline_update_stride": 1,
            },
            "same_stride_tts_for_l0_vs_static": {
                "candidate_method": "tts",
                "candidate_update_stride": selected["l0_stride"],
                "baseline_method": "static",
                "baseline_update_stride": 1,
            },
            "l0_best_vs_tts_acceptance_best": {
                "candidate_method": "naive_async",
                "candidate_update_stride": selected["l0_stride"],
                "baseline_method": "tts",
                "baseline_update_stride": selected["tts_acceptance_stride"],
            },
            "l0_best_vs_tts_engineering_best": {
                "candidate_method": "naive_async",
                "candidate_update_stride": selected["l0_stride"],
                "baseline_method": "tts",
                "baseline_update_stride": selected["tts_engineering_stride"],
            },
            "l0_best_vs_same_stride_tts_for_l0": {
                "candidate_method": "naive_async",
                "candidate_update_stride": selected["l0_stride"],
                "baseline_method": "tts",
                "baseline_update_stride": selected["l0_stride"],
            },
        },
        "prompt_artifact": "p5_prompt_acceptance.parquet",
        "pairing": "exact_prompt_cluster_and_seed",
        "bootstrap": "prompt_cluster_bca",
        "min_paired_prompt_clusters_per_cell": (
            P5_PRIORITY_CONFIRMATION_MIN_PROMPT_CLUSTERS
        ),
        "prompt_limit": 48,
        "prompt_offset": 40,
        "benchmark_repetitions": 5,
    }


def _resolved_confirmation_manifest(
    *,
    selected: dict[str, Any],
    source: ExperimentManifest,
    bindings: dict[str, Any] | None,
) -> ExperimentManifest:
    lock_digest = (
        bindings["lockfile_sha256"] if bindings else source.lockfile_sha256
    )
    manifest = p5_priority_dflash_stride_confirmation_manifest(
        tts_acceptance_stride=selected["tts_acceptance_stride"],
        tts_engineering_stride=selected["tts_engineering_stride"],
        l0_stride=selected["l0_stride"],
        lockfile_sha256=lock_digest,
    )
    if bindings:
        binding_sha256 = sha256_json(bindings)
        manifest = replace(
            manifest,
            engine_params={
                **manifest.engine_params,
                "model_roots_sha256": bindings["model_roots_sha256"],
                "locked_model_revisions": bindings["model_revisions"],
                "runtime_implementation_fingerprint": bindings[
                    "runtime_implementation_fingerprint"
                ],
                "confirmation_input_bindings_sha256": binding_sha256,
            },
        )
    return manifest


def _generation_identity(
    *,
    selector_path: Path,
    selected: dict[str, Any],
    source: ExperimentManifest,
    manifest: ExperimentManifest,
    bindings: dict[str, Any] | None,
    artifact_root_path: Path | None,
) -> dict[str, Any]:
    identity = {
        "selector_sha256": sha256_file(selector_path),
        "source_manifest_sha256": source.content_sha256(),
        "confirmation_manifest_sha256": manifest.content_sha256(),
        "tts_acceptance_stride": selected["tts_acceptance_stride"],
        "tts_engineering_stride": selected["tts_engineering_stride"],
        "tts_engineering_eligible": selected["tts_engineering_eligible"],
        "tts_engineering_fallback_reason": selected[
            "tts_engineering_fallback_reason"
        ],
        "l0_stride": selected["l0_stride"],
        "screen_prompt_offset": int(source.engine_params.get("prompt_offset", 0)),
        "confirmation_prompt_offset": int(
            manifest.engine_params["prompt_offset"]
        ),
        "pytorch_cuda_alloc_conf": source.engine_params.get(
            "pytorch_cuda_alloc_conf"
        ),
    }
    if bindings:
        identity["execution_bindings"] = bindings
        identity["execution_bindings_sha256"] = sha256_json(bindings)
    if artifact_root_path is not None:
        artifact_root = artifact_root_path.expanduser().resolve()
        identity["artifact_root"] = str(artifact_root)
        identity["artifact_root_binding_sha256"] = sha256_json(
            {"schema_version": 1, "artifact_root": str(artifact_root)}
        )
    return identity


def _emit_confirmation(
    *,
    selector_path: Path,
    selected: dict[str, Any],
    source: ExperimentManifest,
    output_manifest_path: Path,
    output_receipt_path: Path,
    evidence: Iterable[Path],
    bindings: dict[str, Any] | None = None,
    artifact_root_path: Path | None = None,
) -> dict[str, Any]:
    if output_manifest_path.resolve() == output_receipt_path.resolve():
        raise ConfirmationError("manifest and receipt outputs must differ")
    manifest = _resolved_confirmation_manifest(
        selected=selected, source=source, bindings=bindings
    )
    source_allocator = source.engine_params.get("pytorch_cuda_alloc_conf")
    if (
        source_allocator != P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF
        or manifest.engine_params.get("pytorch_cuda_alloc_conf")
        != source_allocator
    ):
        raise ConfirmationError("screen/confirmation allocator policy mismatch")
    try:
        manifest.write(output_manifest_path)
    except Exception as exc:
        raise ConfirmationError(f"cannot write confirmation manifest: {exc}") from exc
    manifest_sidecar = _verify_sidecar(output_manifest_path)

    roles = {
        "static": _role_units(manifest, method="static", stride=1),
        "tts_acceptance_best": _role_units(
            manifest, method="tts", stride=selected["tts_acceptance_stride"]
        ),
        "tts_engineering_best": _role_units(
            manifest, method="tts", stride=selected["tts_engineering_stride"]
        ),
        "l0_best": _role_units(
            manifest, method="naive_async", stride=selected["l0_stride"]
        ),
        "same_stride_tts_for_l0": _role_units(
            manifest, method="tts", stride=selected["l0_stride"]
        ),
    }
    if any(len(rows) != len(P5_PRIORITY_CONFIRMATION_LOADS) for rows in roles.values()):
        raise ConfirmationError("generated confirmation role coverage is incomplete")
    identity = _generation_identity(
        selector_path=selector_path,
        selected=selected,
        source=source,
        manifest=manifest,
        bindings=bindings,
        artifact_root_path=artifact_root_path,
    )
    payload = {
        "schema_version": 1,
        "status": "ready_for_execution",
        "scope": "paired_stride_confirmation",
        "generation_identity": identity,
        "generation_identity_sha256": sha256_json(identity),
        "source_selection_unit_ids": selected["source_unit_ids"],
        "load_groups": [
            {"prompt_subset": subset, "concurrency": concurrency}
            for subset, concurrency in P5_PRIORITY_CONFIRMATION_LOADS
        ],
        "roles": roles,
        "unit_ids": sorted(unit.unit_id for unit in manifest.units),
        "analysis_contract": _analysis_contract(selected),
    }
    return _write_receipt(
        output_receipt_path,
        payload,
        [*evidence, output_manifest_path, manifest_sidecar],
    )


def build_confirmation(
    *,
    selector_path: Path,
    screen_manifest_path: Path,
    output_manifest_path: Path,
    output_receipt_path: Path,
) -> dict[str, Any]:
    _, selected, screen, evidence = _validate_selector(
        selector_path, screen_manifest_path
    )
    return _emit_confirmation(
        selector_path=selector_path,
        selected=selected,
        source=screen,
        output_manifest_path=output_manifest_path,
        output_receipt_path=output_receipt_path,
        evidence=[
            *evidence,
            selector_path,
            Path(str(selector_path) + ".sha256"),
            screen_manifest_path,
        ],
    )


def build_confirmation_from_terminal(
    *,
    selected_terminal_path: Path,
    lockfile_path: Path,
    model_roots_path: Path,
    artifact_root_path: Path,
    output_manifest_path: Path,
    output_receipt_path: Path,
) -> dict[str, Any]:
    context = _validated_terminal_context(
        selected_terminal=selected_terminal_path,
        lockfile=lockfile_path,
        model_roots=model_roots_path,
    )
    _, selected, screen, selector_evidence = _validate_selector(
        context["selection_path"], context["source_path"]
    )
    return _emit_confirmation(
        selector_path=context["selection_path"],
        selected=selected,
        source=screen,
        output_manifest_path=output_manifest_path,
        output_receipt_path=output_receipt_path,
        evidence=[*context["evidence"], *selector_evidence],
        bindings=context["bindings"],
        artifact_root_path=artifact_root_path,
    )


def _validate_generation(
    *,
    generation_path: Path,
    selector_path: Path,
    manifest_path: Path,
    selected: dict[str, Any],
    source_manifest: ExperimentManifest,
    bindings: dict[str, Any] | None = None,
    artifact_root_path: Path | None = None,
) -> tuple[dict[str, Any], ExperimentManifest, list[Path]]:
    manifest_sidecar = _verify_sidecar(manifest_path)
    manifest = ExperimentManifest.load(manifest_path)
    expected_manifest = _resolved_confirmation_manifest(
        selected=selected,
        source=source_manifest,
        bindings=bindings,
    )
    if manifest.to_dict() != expected_manifest.to_dict():
        raise ConfirmationError("confirmation manifest differs from frozen design")
    generation, evidence = _verify_receipt(generation_path)
    identity = generation.get("generation_identity")
    expected = _generation_identity(
        selector_path=selector_path,
        selected=selected,
        source=source_manifest,
        manifest=manifest,
        bindings=bindings,
        artifact_root_path=artifact_root_path,
    )
    if (
        generation.get("schema_version") != 1
        or generation.get("status") != "ready_for_execution"
        or generation.get("scope") != "paired_stride_confirmation"
        or not isinstance(identity, dict)
        or identity != expected
        or generation.get("generation_identity_sha256") != sha256_json(identity)
    ):
        raise ConfirmationError("generation receipt identity mismatch")
    if set(generation.get("unit_ids", [])) != {
        unit.unit_id for unit in manifest.units
    }:
        raise ConfirmationError("generation receipt unit coverage mismatch")
    expected_roles = {
        "static": _role_units(manifest, method="static", stride=1),
        "tts_acceptance_best": _role_units(
            manifest, method="tts", stride=selected["tts_acceptance_stride"]
        ),
        "tts_engineering_best": _role_units(
            manifest, method="tts", stride=selected["tts_engineering_stride"]
        ),
        "l0_best": _role_units(
            manifest, method="naive_async", stride=selected["l0_stride"]
        ),
        "same_stride_tts_for_l0": _role_units(
            manifest, method="tts", stride=selected["l0_stride"]
        ),
    }
    expected_contract = _analysis_contract(selected)
    if (
        generation.get("roles") != expected_roles
        or generation.get("analysis_contract") != expected_contract
        or generation.get("source_selection_unit_ids")
        != selected["source_unit_ids"]
    ):
        raise ConfirmationError("generation receipt role/analysis contract mismatch")
    return generation, manifest, [*evidence, manifest_path, manifest_sidecar]


def _load_prompt_analysis(
    analysis_root: Path, manifest: ExperimentManifest
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    prompt_path = analysis_root / "p5_prompt_acceptance.parquet"
    safety_path = analysis_root / "p5_long_context_acceptance.parquet"
    analysis_path = analysis_root / "analysis-manifest.json"
    analysis_sidecar = analysis_root / "analysis-manifest.sha256"
    hashes_path = analysis_root / "analysis-hashes.json"
    for path in (
        prompt_path,
        safety_path,
        analysis_path,
        analysis_sidecar,
        hashes_path,
    ):
        if not path.is_file():
            raise ConfirmationError(f"analysis evidence is missing: {path}")
    _verify_sidecar(analysis_path, analysis_sidecar)
    analysis_manifest = _load_json(analysis_path)
    analysis = analysis_manifest.get("analysis")
    expected_analysis = {
        "baseline": "static",
        "expected_manifest_sha256": manifest.content_sha256(),
        "weight_update_mode_overlay": "lora",
        "methods_overlay": ["static", "tts", "naive_async"],
        "lifecycles_overlay": None,
        "learning_rate_overlay": None,
    }
    if not isinstance(analysis, dict) or any(
        analysis.get(key) != value for key, value in expected_analysis.items()
    ):
        raise ConfirmationError("analysis execution-overlay identity mismatch")

    input_runs = analysis_manifest.get("input_runs")
    expected_units = {unit.unit_id for unit in manifest.units}
    if not isinstance(input_runs, list) or len(input_runs) != len(expected_units):
        raise ConfirmationError("analysis run coverage differs from confirmation")
    for row in input_runs:
        if not isinstance(row, dict) or set(row) != {
            "run_id",
            "unit_id",
            "manifest_sha256",
            "hashes_sha256",
        }:
            raise ConfirmationError("analysis input-run schema mismatch")
        if not all(
            isinstance(row.get(name), str) and row[name]
            for name in ("run_id", "unit_id", "manifest_sha256", "hashes_sha256")
        ):
            raise ConfirmationError("analysis input-run identity is incomplete")
    observed_units = {str(row["unit_id"]) for row in input_runs}
    observed_runs = {str(row["run_id"]) for row in input_runs}
    if (
        observed_units != expected_units
        or len(observed_units) != len(input_runs)
        or len(observed_runs) != len(input_runs)
    ):
        raise ConfirmationError("analysis run coverage differs from confirmation")
    derived = analysis_manifest.get("derived_outputs", {})
    for path in (prompt_path, safety_path):
        row = derived.get(path.name) if isinstance(derived, dict) else None
        if not isinstance(row, dict) or row.get("sha256") != sha256_file(path):
            raise ConfirmationError(
                f"analysis manifest does not attest {path.name}"
            )
    ledger = _load_json(hashes_path)
    expected_ledger = {
        *derived,
        analysis_path.name,
        analysis_sidecar.name,
    }
    if set(ledger) != expected_ledger:
        raise ConfirmationError("analysis hash ledger is not transitively closed")
    analysis_evidence = [hashes_path]
    for relative, row in ledger.items():
        candidate = (analysis_root / relative).resolve()
        try:
            candidate.relative_to(analysis_root.resolve())
        except ValueError as exc:
            raise ConfirmationError(
                f"analysis ledger path escapes its root: {relative}"
            ) from exc
        if (
            not isinstance(row, dict)
            or not candidate.is_file()
            or row.get("sha256") != sha256_file(candidate)
            or row.get("bytes") != candidate.stat().st_size
        ):
            raise ConfirmationError(
                f"analysis hash ledger mismatch for {relative}"
            )
        if relative in derived and derived.get(relative) != row:
            raise ConfirmationError(
                f"analysis derived-output mismatch for {relative}"
            )
        analysis_evidence.append(candidate)
    try:
        prompt = pd.read_parquet(prompt_path)
        safety = pd.read_parquet(safety_path)
    except Exception as exc:
        raise ConfirmationError(f"cannot read P5 analysis evidence: {exc}") from exc
    expected_repetitions = int(manifest.engine_params["benchmark_repetitions"])
    if "benchmark_repetitions" not in prompt:
        raise ConfirmationError("prompt evidence lacks benchmark_repetitions")
    repetitions = pd.to_numeric(
        prompt["benchmark_repetitions"], errors="coerce"
    )
    if repetitions.isna().any() or not (repetitions == expected_repetitions).all():
        raise ConfirmationError(
            "prompt evidence benchmark_repetitions differs from confirmation"
        )
    required_safety = {
        "method",
        *P5_IDENTITY_COLUMNS,
        "context_length",
        "adaptation_fallback_count",
        "exactness_violations",
        "version_mismatch_count",
    }
    missing_safety = sorted(required_safety - set(safety.columns))
    if missing_safety:
        raise ConfirmationError(
            f"P5 safety evidence lacks columns: {missing_safety}"
        )
    for name in (
        "adaptation_fallback_count",
        "exactness_violations",
        "version_mismatch_count",
    ):
        values = pd.to_numeric(safety[name], errors="coerce")
        if values.isna().any() or (values != 0).any():
            raise ConfirmationError(f"P5 safety gate failed: {name}")
    identity = ["method", *P5_IDENTITY_COLUMNS, "context_length"]
    expected_adapted = {
        tuple(row)
        for row in prompt[prompt["method"] != "static"][identity]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    observed_adapted = {
        tuple(row)
        for row in safety[safety["method"] != "static"][identity]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if observed_adapted != expected_adapted:
        raise ConfirmationError("P5 safety/prompt adaptive coverage differs")
    return prompt, analysis_manifest, analysis_evidence


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_dataset_preflight(
    *,
    artifact_root: Path,
    manifest: ExperimentManifest,
    bindings: dict[str, Any],
) -> tuple[str, list[Path]]:
    receipt = artifact_root / "dataset-preflight.json"
    sidecar = _verify_sidecar(receipt)
    payload = _load_json(receipt)
    if (
        payload.get("schema_version") != 1
        or payload.get("lockfile_sha256") != bindings["lockfile_sha256"]
        or payload.get("limit") != int(manifest.engine_params["prompt_limit"])
        or payload.get("offset") != int(manifest.engine_params["prompt_offset"])
    ):
        raise ConfirmationError("confirmation dataset preflight identity mismatch")
    datasets = payload.get("datasets")
    selected = [
        row
        for row in datasets
        if isinstance(row, dict) and row.get("adapter_key") == "livecodebench"
    ] if isinstance(datasets, list) else []
    if len(selected) != 1:
        raise ConfirmationError("confirmation dataset preflight lacks LiveCodeBench")
    dataset = selected[0]
    if (
        dataset.get("selected_count") != int(manifest.engine_params["prompt_limit"])
        or not isinstance(dataset.get("revision"), str)
        or not dataset["revision"]
        or not _is_sha256(dataset.get("selected_sample_ids_sha256"))
    ):
        raise ConfirmationError("confirmation dataset selection is incomplete")
    return sha256_file(receipt), [receipt, sidecar]


def _validate_confirmation_run_provenance(
    *,
    artifact_root: Path,
    source_analysis: dict[str, Any],
    manifest: ExperimentManifest,
    lockfile_path: Path,
    model_roots_path: Path,
    bindings: dict[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    """Re-open every analyzed run and close the raw evidence graph."""

    artifact_root = artifact_root.expanduser().resolve()
    if not artifact_root.is_dir():
        raise ConfirmationError(
            f"confirmation artifact root does not exist: {artifact_root}"
        )
    rows = source_analysis.get("input_runs")
    if not isinstance(rows, list):
        raise ConfirmationError("analysis input runs are missing")
    revisions = bindings.get("model_revisions")
    expected_runtime = bindings.get("runtime_implementation_fingerprint")
    if not isinstance(revisions, dict) or not isinstance(expected_runtime, dict):
        raise ConfirmationError("confirmation execution bindings are incomplete")

    validator = _matched_builder_module()
    try:
        raw_evidence, observed_runtime = validator._verify_run_provenance(
            artifact_root,
            rows,
            source=manifest,
            lockfile_path=lockfile_path.resolve(),
            model_roots_path=model_roots_path.resolve(),
            target_revision=str(revisions["target"]),
            drafter_revision=str(revisions["drafter"]),
        )
    except Exception as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise ConfirmationError(
            f"confirmation raw-run provenance failed: {exc}"
        ) from exc
    if observed_runtime != expected_runtime:
        raise ConfirmationError(
            "confirmation runs used a runtime outside the generation binding"
        )

    dataset_sha256, dataset_evidence = _validate_dataset_preflight(
        artifact_root=artifact_root,
        manifest=manifest,
        bindings=bindings,
    )
    expected_units = {unit.unit_id: unit for unit in manifest.units}
    experiment_hashes: set[str] = set()
    execution_hashes: dict[str, str] = {}
    additional_evidence: list[Path] = []
    required_ledger_files = set(REQUIRED_FILES) - {"hashes.json"}
    for row in rows:
        run_id = str(row["run_id"])
        run_root = (artifact_root / run_id).resolve()
        try:
            run_root.relative_to(artifact_root)
        except ValueError as exc:
            raise ConfirmationError(f"run path escapes artifact root: {run_id}") from exc
        run = _load_json(run_root / "manifest.json")
        expected_unit = expected_units[str(row["unit_id"])]
        try:
            resolved_unit = RunUnit.from_dict(run)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfirmationError(
                f"run {run_id} has invalid unit identity: {exc}"
            ) from exc
        expected_unit_manifest = expected_unit.to_manifest_dict()
        if resolved_unit.unit_id != expected_unit.unit_id or any(
            run.get(key) != value for key, value in expected_unit_manifest.items()
        ):
            raise ConfirmationError(f"run {run_id} unit manifest drift")

        engine = run.get("engine_params")
        if not isinstance(engine, dict) or any(
            engine.get(key) != value
            for key, value in manifest.engine_params.items()
        ):
            raise ConfirmationError(f"run {run_id} source engine contract drift")
        if (
            engine.get("weight_update_mode_override") != "lora"
            or engine.get("dataset_preflight_sha256") != dataset_sha256
        ):
            raise ConfirmationError(f"run {run_id} execution overlay drift")

        experiment_sha = run.get("experiment_manifest_sha256")
        execution_sha = run.get("unit_execution_sha256")
        if not _is_sha256(experiment_sha) or not _is_sha256(execution_sha):
            raise ConfirmationError(f"run {run_id} execution hash is invalid")
        experiment_hashes.add(str(experiment_sha))
        execution_hashes[expected_unit.unit_id] = str(execution_sha)

        hashes_path = run_root / "hashes.json"
        ledger = _load_json(hashes_path)
        if not required_ledger_files.issubset(ledger):
            raise ConfirmationError(f"run {run_id} ledger omits normative files")
        telemetry = sorted((run_root / "runtime").glob("*.jsonl"))
        if not telemetry or not any(path.stat().st_size > 0 for path in telemetry):
            raise ConfirmationError(f"run {run_id} has no non-empty telemetry shard")
        for path in telemetry:
            relative = path.relative_to(run_root).as_posix()
            if relative not in ledger:
                raise ConfirmationError(f"run {run_id} telemetry is not hash-bound")
        checkpoints = run_root / "prefix-checkpoints.json"
        if checkpoints.name not in ledger or not checkpoints.is_file():
            raise ConfirmationError(f"run {run_id} lacks bound prefix checkpoints")
        checkpoint_payload = _load_json(checkpoints)
        if not checkpoint_payload.get("checkpoints"):
            raise ConfirmationError(f"run {run_id} has empty prefix checkpoints")

        config_path = Path(str(engine.get("adaptation_config_path", ""))).resolve()
        try:
            config_path.relative_to(run_root)
        except ValueError as exc:
            raise ConfirmationError(
                f"run {run_id} adaptation config escapes its run root"
            ) from exc
        if (
            not config_path.is_file()
            or sha256_file(config_path) != engine.get("runtime_config_sha256")
        ):
            raise ConfirmationError(f"run {run_id} adaptation config hash drift")
        additional_evidence.append(config_path)

    if (
        len(experiment_hashes) != 1
        or len(execution_hashes) != len(manifest.units)
        or len(set(execution_hashes.values())) != len(execution_hashes)
    ):
        raise ConfirmationError("confirmation run execution hashes are inconsistent")
    raw_identity = {
        "schema_version": 1,
        "artifact_root": str(artifact_root),
        "dataset_preflight_sha256": dataset_sha256,
        "experiment_manifest_sha256": next(iter(experiment_hashes)),
        "unit_execution_sha256": dict(sorted(execution_hashes.items())),
        "runtime_implementation_sha256": observed_runtime["sha256"],
        "input_runs": sorted(
            (
                {
                    "run_id": str(row["run_id"]),
                    "unit_id": str(row["unit_id"]),
                    "manifest_sha256": str(row["manifest_sha256"]),
                    "hashes_sha256": str(row["hashes_sha256"]),
                }
                for row in rows
            ),
            key=lambda row: row["unit_id"],
        ),
    }
    return [
        *raw_evidence,
        *dataset_evidence,
        *additional_evidence,
    ], raw_identity


def compare_confirmation(
    *,
    selector_path: Path,
    screen_manifest_path: Path,
    generation_receipt_path: Path,
    confirmation_manifest_path: Path,
    analysis_root: Path,
    output_path: Path,
    bootstrap_replicates: int = 5000,
    bindings: dict[str, Any] | None = None,
    additional_evidence: Iterable[Path] = (),
    artifact_root_path: Path | None = None,
    lockfile_path: Path | None = None,
    model_roots_path: Path | None = None,
) -> dict[str, Any]:
    _, selected, screen, selector_evidence = _validate_selector(
        selector_path, screen_manifest_path
    )
    _, manifest, generation_evidence = _validate_generation(
        generation_path=generation_receipt_path,
        selector_path=selector_path,
        manifest_path=confirmation_manifest_path,
        selected=selected,
        source_manifest=screen,
        bindings=bindings,
        artifact_root_path=artifact_root_path,
    )
    prompt, source_analysis, analysis_evidence = _load_prompt_analysis(
        analysis_root, manifest
    )
    raw_evidence: list[Path] = []
    raw_identity: dict[str, Any] | None = None
    raw_inputs = (
        artifact_root_path,
        lockfile_path,
        model_roots_path,
        bindings,
    )
    if any(value is not None for value in raw_inputs):
        if any(value is None for value in raw_inputs):
            raise ConfirmationError(
                "raw-run validation requires artifact root, lockfile, "
                "model roots and execution bindings together"
            )
        raw_evidence, raw_identity = _validate_confirmation_run_provenance(
            artifact_root=artifact_root_path,
            source_analysis=source_analysis,
            manifest=manifest,
            lockfile_path=lockfile_path,
            model_roots_path=model_roots_path,
            bindings=bindings,
        )
    contract = _analysis_contract(selected)
    comparisons = {}
    try:
        for name, spec in contract["comparisons"].items():
            comparisons[name] = paired_cross_stride_acceptance_table(
                prompt,
                candidate_method=spec["candidate_method"],
                candidate_stride=spec["candidate_update_stride"],
                baseline_method=spec["baseline_method"],
                baseline_stride=spec["baseline_update_stride"],
                b=bootstrap_replicates,
            )
    except ValueError as exc:
        raise ConfirmationError(f"cross-stride comparison failed: {exc}") from exc

    expected_cells = {(4096, 8), (16384, 8), (4096, 48), (16384, 20)}
    for name, comparison in comparisons.items():
        observed_cells = {
            (int(row.context_length), int(row.offered_concurrency))
            for row in comparison.itertuples()
        }
        if observed_cells != expected_cells or len(comparison) != len(expected_cells):
            raise ConfirmationError(
                f"{name} comparison cell coverage mismatch: {observed_cells}"
            )
    analysis_manifest_path = analysis_root / "analysis-manifest.json"
    prompt_path = analysis_root / "p5_prompt_acceptance.parquet"
    safety_path = analysis_root / "p5_long_context_acceptance.parquet"
    identity = {
        "id": "p5_prompt_paired_cross_stride_v1",
        "comparisons": contract["comparisons"],
        "bootstrap_replicates": int(bootstrap_replicates),
        "min_paired_prompt_clusters_per_cell": (
            P5_PRIORITY_CONFIRMATION_MIN_PROMPT_CLUSTERS
        ),
        "prompt_limit": int(manifest.engine_params["prompt_limit"]),
        "benchmark_repetitions": int(
            manifest.engine_params["benchmark_repetitions"]
        ),
        "pytorch_cuda_alloc_conf": manifest.engine_params[
            "pytorch_cuda_alloc_conf"
        ],
        "selector_sha256": sha256_file(selector_path),
        "generation_receipt_sha256": sha256_file(generation_receipt_path),
        "confirmation_manifest_sha256": manifest.content_sha256(),
        "source_analysis_manifest_sha256": sha256_file(analysis_manifest_path),
        "prompt_acceptance_sha256": sha256_file(prompt_path),
        "safety_acceptance_sha256": sha256_file(safety_path),
    }
    if bindings:
        identity["execution_bindings_sha256"] = sha256_json(bindings)
    if raw_identity is not None:
        identity["raw_run_provenance"] = raw_identity
        identity["raw_run_provenance_sha256"] = sha256_json(raw_identity)
    records = {
        name: json.loads(table.to_json(orient="records", double_precision=15))
        for name, table in comparisons.items()
    }
    sample_pass = all(
        bool(
            (
                table["paired_prompt_clusters"]
                >= P5_PRIORITY_CONFIRMATION_MIN_PROMPT_CLUSTERS
            ).all()
        )
        for table in comparisons.values()
    )
    ci_gates = {
        name: bool((table["acceptance_gain_ci_low"] > 0).all())
        for name, table in comparisons.items()
    }
    raw_provenance_pass = raw_identity is not None
    formal_by_comparison = {
        name: (
            sample_pass
            and passed
            and raw_provenance_pass
            and (
                name != "tts_engineering_best_vs_static"
                or selected["tts_engineering_eligible"]
            )
        )
        for name, passed in ci_gates.items()
    }
    payload = {
        "schema_version": 1,
        "status": "comparison_complete",
        "scope": "paired_stride_confirmation",
        "analysis_identity": identity,
        "analysis_identity_sha256": sha256_json(identity),
        "source_analysis_baseline": source_analysis["analysis"].get("baseline"),
        "scientific_sample_pass": sample_pass,
        "ci_gates": ci_gates,
        "tts_engineering_selection": {
            "eligible": selected["tts_engineering_eligible"],
            "fallback_reason": selected["tts_engineering_fallback_reason"],
        },
        "formal_acceptance_claim_pass_by_comparison": formal_by_comparison,
        "all_cells_ci_low_positive": all(ci_gates.values()),
        "raw_provenance_pass": raw_provenance_pass,
        "formal_acceptance_claim_pass": (
            sample_pass and all(ci_gates.values()) and raw_provenance_pass
        ),
        "results": records,
    }
    return _write_receipt(
        output_path,
        payload,
        [
            *selector_evidence,
            *generation_evidence,
            *analysis_evidence,
            *raw_evidence,
            selector_path,
            generation_receipt_path,
            confirmation_manifest_path,
            *additional_evidence,
        ],
    )


def compare_confirmation_from_terminal(
    *,
    selected_terminal_path: Path,
    lockfile_path: Path,
    model_roots_path: Path,
    artifact_root_path: Path,
    generation_receipt_path: Path,
    confirmation_manifest_path: Path,
    analysis_root: Path,
    output_path: Path,
    bootstrap_replicates: int = 5000,
) -> dict[str, Any]:
    context = _validated_terminal_context(
        selected_terminal=selected_terminal_path,
        lockfile=lockfile_path,
        model_roots=model_roots_path,
    )
    return compare_confirmation(
        selector_path=context["selection_path"],
        screen_manifest_path=context["source_path"],
        generation_receipt_path=generation_receipt_path,
        confirmation_manifest_path=confirmation_manifest_path,
        analysis_root=analysis_root,
        output_path=output_path,
        bootstrap_replicates=bootstrap_replicates,
        bindings=context["bindings"],
        additional_evidence=context["evidence"],
        artifact_root_path=artifact_root_path,
        lockfile_path=lockfile_path,
        model_roots_path=model_roots_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a receipt-bound manifest")
    build.add_argument("--selected-terminal", type=Path, required=True)
    build.add_argument("--lockfile", type=Path, required=True)
    build.add_argument("--model-roots", type=Path, required=True)
    build.add_argument("--artifact-root", type=Path, required=True)
    build.add_argument("--output-manifest", type=Path, required=True)
    build.add_argument("--output-receipt", type=Path, required=True)

    compare = commands.add_parser(
        "compare", help="compare L0-best against TTS-best across strides"
    )
    compare.add_argument("--selected-terminal", type=Path, required=True)
    compare.add_argument("--lockfile", type=Path, required=True)
    compare.add_argument("--model-roots", type=Path, required=True)
    compare.add_argument("--artifact-root", type=Path, required=True)
    compare.add_argument("--generation-receipt", type=Path, required=True)
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--analysis-root", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            build_confirmation_from_terminal(
                selected_terminal_path=args.selected_terminal,
                lockfile_path=args.lockfile,
                model_roots_path=args.model_roots,
                artifact_root_path=args.artifact_root,
                output_manifest_path=args.output_manifest,
                output_receipt_path=args.output_receipt,
            )
        else:
            compare_confirmation_from_terminal(
                selected_terminal_path=args.selected_terminal,
                lockfile_path=args.lockfile,
                model_roots_path=args.model_roots,
                artifact_root_path=args.artifact_root,
                generation_receipt_path=args.generation_receipt,
                confirmation_manifest_path=args.manifest,
                analysis_root=args.analysis_root,
                output_path=args.output,
                bootstrap_replicates=args.bootstrap_replicates,
            )
    except ConfirmationError as exc:
        raise SystemExit(f"P5 stride confirmation failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
