#!/usr/bin/env python3
"""Validate and close the receipt graph for the matched P5 controller queue.

The GPU work stays in the shell queue and the existing ``lightcone-spec``
commands.  This helper owns only the small, security-sensitive part: gate
interpretation, recursive receipt validation, resumable failure archival, and
terminal publication.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


class QueueEvidenceError(RuntimeError):
    pass


SELECTED = "CONTROLLER_SELECTED.json"
BLOCKED = "CONTROLLER_BLOCKED.json"
FAILED = "CONTROLLER_FAILED.json"
PHASE1_GATE = "PHASE1_GATE.json"
HEADLINE_GENERATION = "FINAL_0_40K_MANIFESTS.json"
_HEX = frozenset("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise QueueEvidenceError(f"duplicate JSON key {key!r}: {path}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueueEvidenceError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QueueEvidenceError(f"JSON is not an object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value)
    )


def _verify_sidecar(path: Path) -> Path:
    path = path.resolve()
    sidecar = Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise QueueEvidenceError(f"missing artifact or sidecar: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != _sha256(path):
        raise QueueEvidenceError(f"SHA-256 sidecar mismatch: {path}")
    return sidecar


def _receipt_evidence(
    path: Path,
    *,
    recursive: bool = True,
    visited: set[Path] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    path = path.resolve()
    visited = set() if visited is None else visited
    if path in visited:
        return _load_json(path), [path, _verify_sidecar(path)]
    visited.add(path)
    sidecar = _verify_sidecar(path)
    payload = _load_json(path)
    rows = payload.get("evidence")
    if not isinstance(rows, list) or not rows:
        raise QueueEvidenceError(f"receipt has no evidence: {path}")
    evidence: list[Path] = [path, sidecar]
    seen: set[Path] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise QueueEvidenceError(f"invalid evidence row {index}: {path}")
        raw = row.get("path")
        digest = row.get("sha256")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise QueueEvidenceError(f"non-absolute evidence path {index}: {path}")
        item = Path(raw).resolve()
        if raw != str(item) or item in seen:
            raise QueueEvidenceError(f"non-canonical/duplicate evidence: {item}")
        if not _is_sha256(digest) or not item.is_file() or _sha256(item) != digest:
            raise QueueEvidenceError(f"receipt evidence drift: {item}")
        seen.add(item)
        evidence.append(item)
        if recursive and item.suffix == ".json":
            try:
                nested = _load_json(item)
            except QueueEvidenceError:
                continue
            if isinstance(nested.get("evidence"), list):
                _, nested_evidence = _receipt_evidence(
                    item, recursive=True, visited=visited
                )
                evidence.extend(nested_evidence)
    return payload, sorted(set(evidence))


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


def _publish_receipt(
    path: Path,
    payload: Mapping[str, Any],
    evidence: Iterable[Path],
) -> dict[str, Any]:
    path = path.resolve()
    if "evidence" in payload:
        raise QueueEvidenceError("receipt body must not predeclare evidence")
    rows = []
    for item in sorted({Path(value).resolve() for value in evidence}):
        if not item.is_file():
            raise QueueEvidenceError(f"terminal evidence is missing: {item}")
        rows.append({"path": str(item), "sha256": _sha256(item)})
    if not rows:
        raise QueueEvidenceError("terminal receipt cannot have empty evidence")
    result = {**payload, "evidence": rows}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise QueueEvidenceError(f"refusing to overwrite changed receipt: {path}")
    else:
        _atomic_text(path, text)
    digest = _sha256(path)
    sidecar = Path(str(path) + ".sha256")
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() != digest:
        raise QueueEvidenceError(f"refusing to replace changed sidecar: {sidecar}")
    if not sidecar.exists():
        _atomic_text(sidecar, digest + "\n")
    return result


def _verify_comparison(path: Path) -> tuple[dict[str, Any], list[Path]]:
    payload, evidence = _receipt_evidence(path)
    required = {
        "status": "comparison_complete",
        "scope": "paired_stride_confirmation",
        "scientific_sample_pass": True,
        "all_cells_ci_low_positive": True,
        "raw_provenance_pass": True,
        "formal_acceptance_claim_pass": True,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise QueueEvidenceError(
                f"formal confirmation is not eligible: {field}={payload.get(field)!r}"
            )
    gates = payload.get("ci_gates")
    if not isinstance(gates, Mapping) or set(gates) != {
        "tts_best_vs_static",
        "l0_best_vs_tts_best",
    } or not all(value is True for value in gates.values()):
        raise QueueEvidenceError("formal confirmation CI gate set is incomplete")
    return payload, evidence


def block_confirmation(
    *, root: Path, comparison: Path, queue_source: Path
) -> dict[str, Any]:
    payload, evidence = _receipt_evidence(comparison)
    if (
        payload.get("status") != "comparison_complete"
        or payload.get("scope") != "paired_stride_confirmation"
        or payload.get("formal_acceptance_claim_pass") is not False
    ):
        raise QueueEvidenceError(
            "confirmation block requires a valid negative comparison receipt"
        )
    root = root.resolve()
    selected = root / SELECTED
    if selected.exists() or Path(str(selected) + ".sha256").exists():
        raise QueueEvidenceError(f"conflicting selected terminal exists: {selected}")
    return _publish_receipt(
        root / BLOCKED,
        {
            "schema_version": 1,
            "status": "matched_controller_blocked",
            "scope": "matched_dflash_l1_l2_l3_evidence",
            "controller_identity_sha256": None,
            "eligible": {"l1": False, "l2": False, "l3": False},
            "blocked_reasons": ["formal_confirmation_gate_failed"],
            "phase2_executed": False,
            "helper_source_sha256": _sha256(Path(__file__).resolve()),
            "queue_source_sha256": _sha256(queue_source.resolve()),
        },
        [*evidence, Path(__file__).resolve(), queue_source.resolve()],
    )


def _verify_generation(path: Path) -> tuple[dict[str, Any], list[Path]]:
    payload, evidence = _receipt_evidence(path)
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "matched_controller_manifests_generated"
    ):
        raise QueueEvidenceError("matched-controller generation receipt mismatch")
    identity = payload.get("controller_identity")
    if (
        not isinstance(identity, Mapping)
        or not _is_sha256(identity.get("sha256"))
        or payload.get("controller_identity_sha256") != identity.get("sha256")
    ):
        raise QueueEvidenceError("controller identity is absent or inconsistent")
    identity_body = dict(identity)
    identity_sha = identity_body.pop("sha256")
    if _sha256_json(identity_body) != identity_sha:
        raise QueueEvidenceError("controller identity digest mismatch")
    mirror = payload.get("mirror_contract")
    if not isinstance(mirror, Mapping) or mirror.get("exact") is not True:
        raise QueueEvidenceError("phase-1/phase-2 mirror contract is not exact")
    expected_windows = {
        "phase1_trace": {"offset": 88, "limit": 48, "half_open": [88, 136]},
        "phase2_l3": {"offset": 136, "limit": 48, "half_open": [136, 184]},
    }
    if (
        mirror.get("prompt_windows") != expected_windows
        or mirror.get("prompt_windows_disjoint") is not True
        or identity.get("prompt_windows") != expected_windows
    ):
        raise QueueEvidenceError("controller prompt-window identity mismatch")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "TRACE_MATCHED",
        "L3_PHASE2_MATCHED",
        "L3_PHASE2_TTS_REFERENCE",
    }:
        raise QueueEvidenceError("matched manifest artifact set is incomplete")
    for name, row in artifacts.items():
        if not isinstance(row, Mapping):
            raise QueueEvidenceError(f"invalid generated artifact row: {name}")
        artifact = Path(str(row.get("path", ""))).resolve()
        sidecar = Path(str(row.get("sidecar_path", ""))).resolve()
        if (
            not artifact.is_file()
            or not sidecar.is_file()
            or row.get("sha256") != _sha256(artifact)
            or row.get("sidecar_sha256") != _sha256(sidecar)
        ):
            raise QueueEvidenceError(f"generated manifest drift: {name}")
        manifest = _load_json(artifact)
        engine = manifest.get("engine_params")
        units = manifest.get("units")
        if not isinstance(engine, Mapping) or not isinstance(units, list):
            raise QueueEvidenceError(f"generated manifest contract missing: {name}")
        methods = {
            str(unit.get("method")) for unit in units if isinstance(unit, Mapping)
        }
        if name == "TRACE_MATCHED":
            valid = engine.get("prompt_offset") == 88 and methods == {
                "naive_async",
                "tts",
            }
        elif name == "L3_PHASE2_MATCHED":
            valid = (
                engine.get("prompt_offset") == 136
                and engine.get("l3_evaluation_only") is True
                and methods == {"lc_transport"}
            )
        else:
            valid = (
                engine.get("prompt_offset") == 136
                and engine.get("phase2_tts_reference_only") is True
                and methods == {"tts"}
            )
        if not valid:
            raise QueueEvidenceError(f"generated manifest phase ownership mismatch: {name}")
    return payload, evidence


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fraction(value: Any) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _verify_controller_report(
    path: Path,
    *,
    controller_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    path = path.resolve()
    report = _load_json(path)
    artifact_raw = report.get("artifact_path")
    if not isinstance(artifact_raw, str) or not Path(artifact_raw).is_absolute():
        raise QueueEvidenceError(f"controller report has invalid artifact path: {path}")
    artifact = Path(artifact_raw).resolve()
    sidecar = _verify_sidecar(artifact)
    if report.get("artifact_sha256") != _sha256(artifact):
        raise QueueEvidenceError(f"controller report artifact digest mismatch: {path}")
    artifact_payload = _load_json(artifact)
    expected_pair = (
        controller_identity.get("model_pair")
        if isinstance(controller_identity, Mapping)
        else artifact_payload.get("model_pair_id")
    )
    if (
        artifact_payload.get("schema_version") != 1
        or not isinstance(expected_pair, str)
        or not expected_pair
        or artifact_payload.get("model_pair_id") != expected_pair
    ):
        raise QueueEvidenceError("controller payload schema/model-pair mismatch")
    extra = artifact_payload.get("extra")
    if not isinstance(extra, Mapping):
        raise QueueEvidenceError("controller payload has no frozen extra metadata")
    calibration_fastpath = extra.get("constant_fast_path_calibration_coverage")
    if (
        extra.get("constant_fast_path_source") != "calibration_only_v1"
        or not isinstance(calibration_fastpath, Mapping)
        or not _integer_at_least(calibration_fastpath.get("records"), 1)
        or any(
            not _fraction(calibration_fastpath.get(field))
            for field in (
                "l1_constant_apply_fraction",
                "l1_constant_discard_fraction",
                "l2_constant_profile_fraction",
            )
        )
    ):
        raise QueueEvidenceError(
            "controller constant fast paths are not calibration-bound"
        )
    runtime = extra.get("controller_runtime_identity")
    runtime_sha = extra.get("controller_runtime_identity_sha256")
    layout_sha = extra.get("parameter_layout_sha256")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("schema_version") != 3
        or not _is_sha256(runtime_sha)
        or _sha256_json(runtime) != runtime_sha
        or not _is_sha256(layout_sha)
    ):
        raise QueueEvidenceError("controller runtime/layout identity is invalid")
    transport_map = artifact_payload.get("transport_map")
    transport_map_sha = extra.get("transport_map_sha256")
    if (
        not isinstance(transport_map, Mapping)
        or not _is_sha256(transport_map_sha)
        or _sha256_json(transport_map) != transport_map_sha
    ):
        raise QueueEvidenceError("controller transport-map identity is invalid")
    expected_name = (
        f"{expected_pair}."
        f"{runtime.get('candidate', {}).get('weight_update_mode')}."
        f"{layout_sha}.controller.json"
    )
    if artifact.name != expected_name:
        raise QueueEvidenceError(
            f"controller filename does not bind pair/mode/layout: {artifact.name}"
        )
    for gate in (
        "trace_exactness",
        "oracle_replay_gate",
        "tts_paired_gate",
        "learned_policy_gate",
        "l3_gate",
    ):
        if report.get(gate) != extra.get(gate):
            raise QueueEvidenceError(f"controller report/payload {gate} mismatch")
    utility_gate = extra.get("l3_gate", {}).get(
        "heldout_transported_utility_gate", {}
    )
    if utility_gate.get("complete") is True and (
        utility_gate.get("transport_map_sha256") != transport_map_sha
    ):
        raise QueueEvidenceError("L3 utility gate uses another transport map")
    if controller_identity is not None:
        bindings = controller_identity.get("bindings")
        revisions = bindings.get("model_revisions") if isinstance(bindings, Mapping) else None
        model = runtime.get("model")
        candidate = runtime.get("candidate")
        sampling = runtime.get("sampling")
        expected_candidate = {
            "weight_update_mode": controller_identity.get("tail_layout_mode"),
            "adapter_rank": controller_identity.get("adapter_rank"),
            "optimizer": controller_identity.get("optimizer"),
            "lr": controller_identity.get("lr"),
            "weight_decay": controller_identity.get("weight_decay"),
            "update_stride": controller_identity.get("update_stride"),
            "lifecycle": controller_identity.get("lifecycle"),
        }
        if (
            not isinstance(revisions, Mapping)
            or model
            != {
                "pair_id": controller_identity.get("model_pair"),
                "target_revision": revisions.get("target"),
                "drafter_revision": revisions.get("drafter"),
                "tokenizer_revision": revisions.get("tokenizer"),
            }
            or not isinstance(candidate, Mapping)
            or any(candidate.get(key) != value for key, value in expected_candidate.items())
            or not isinstance(sampling, Mapping)
            or sampling.get("temperature") != 0.0
            or sampling.get("top_p") != 1.0
        ):
            raise QueueEvidenceError(
                "controller artifact does not match the selected optimizer, "
                "stride, model revisions, lifecycle or sampling identity"
            )
    return report, [path, artifact, sidecar]


def _finite_positive_ci(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (float, int)) for item in value)
        and float(value[0]) > 0.0
        and float(value[1]) >= float(value[0])
    )


def _integer_at_least(value: Any, minimum: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
    )


def _policy_gate(
    value: Any,
    *,
    method: str,
    reference: str,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(
        value.get("complete") is True
        and value.get("reference") == reference
        and _integer_at_least(value.get("n_test_groups"), 8)
        and _integer_at_least(value.get("incomplete_pairs", 0), 0)
        and value.get("incomplete_pairs", 0) == 0
        and value.get(f"{method}_eligible") is True
        and _finite_positive_ci(value.get(f"{method}_ci95"))
    )


def _phase1_flags(report: Mapping[str, Any]) -> dict[str, bool]:
    oracle = report.get("oracle_replay_gate", {})
    paired = report.get("tts_paired_gate", {})
    learned = report.get("learned_policy_gate", {})
    exact = report.get("trace_exactness", {})
    l3 = report.get("l3_gate", {})
    exact_ok = bool(
        isinstance(exact, Mapping)
        and exact.get("verified") is True
        and exact.get("violation_count") == 0
        and _integer_at_least(exact.get("rounds_checked"), 1)
    )
    return {
        "trace_exactness": exact_ok,
        "oracle_l1": _policy_gate(
            oracle,
            method="l1",
            reference="same_arrival_full_candidate_l0",
        ),
        "oracle_l2": _policy_gate(
            oracle,
            method="l2",
            reference="same_arrival_full_candidate_l0",
        ),
        "paired_tts_l1": _policy_gate(
            paired,
            method="l1",
            reference="same_candidate_actual_tts_barrier",
        ),
        "paired_tts_l2": _policy_gate(
            paired,
            method="l2",
            reference="same_candidate_actual_tts_barrier",
        ),
        "learned_l1": _policy_gate(
            learned,
            method="l1",
            reference="learned_policy_same_candidate_actual_tts_barrier",
        ),
        "learned_l2": _policy_gate(
            learned,
            method="l2",
            reference="learned_policy_same_candidate_actual_tts_barrier",
        ),
        "l3_evaluation_ready": bool(
            isinstance(l3, Mapping) and l3.get("evaluation_ready") is True
        ),
    }


def _fast_path_coverage(report: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _load_json(Path(str(report["artifact_path"])).resolve())
    extra = artifact["extra"]
    calibration = extra["constant_fast_path_calibration_coverage"]
    learned = report.get("learned_policy_gate", {})
    heldout_fields = (
        "l1_zero_delay_fastpath_fraction",
        "l1_constant_apply_fastpath_fraction",
        "l1_constant_discard_fastpath_fraction",
        "l1_predictor_path_fraction",
        "l2_zero_delay_fastpath_fraction",
        "l2_constant_profile_fastpath_fraction",
        "l2_unit_kappa_fastpath_fraction",
        "l2_predictor_path_fraction",
    )
    if not isinstance(learned, Mapping) or any(
        not _fraction(learned.get(field)) for field in heldout_fields
    ):
        return {
            "source": extra["constant_fast_path_source"],
            "calibration": dict(calibration),
            "heldout": {
                "available": False,
                "disabled_reason": (
                    learned.get("disabled_reason")
                    if isinstance(learned, Mapping)
                    else "learned policy gate is absent"
                ),
            },
            "enablement_contract": (
                "calibration_constructs_fast_path_heldout_policy_ci_enables_v1"
            ),
        }
    heldout = {
        "available": True,
        **{field: float(learned[field]) for field in heldout_fields},
    }
    if abs(
        sum(
            heldout[field]
            for field in (
                "l1_zero_delay_fastpath_fraction",
                "l1_constant_apply_fastpath_fraction",
                "l1_constant_discard_fastpath_fraction",
                "l1_predictor_path_fraction",
            )
        )
        - 1.0
    ) > 1e-9:
        raise QueueEvidenceError("L1 fast-path coverage is not exhaustive")
    if abs(
        sum(
            heldout[field]
            for field in (
                "l2_zero_delay_fastpath_fraction",
                "l2_constant_profile_fastpath_fraction",
                "l2_predictor_path_fraction",
            )
        )
        - 1.0
    ) > 1e-9:
        raise QueueEvidenceError("L2 fast-path coverage is not exhaustive")
    return {
        "source": extra["constant_fast_path_source"],
        "calibration": dict(calibration),
        "heldout": heldout,
        "enablement_contract": (
            "calibration_constructs_fast_path_heldout_policy_ci_enables_v1"
        ),
    }


def phase1_gate(
    *,
    comparison: Path | None,
    tts_foundation_terminal: Path | None = None,
    generation: Path,
    report: Path,
    output: Path,
    queue_source: Path | None = None,
) -> dict[str, Any]:
    if tts_foundation_terminal is not None:
        _, prerequisite_evidence = _headline_foundation(tts_foundation_terminal)
    elif comparison is not None:
        _, prerequisite_evidence = _verify_comparison(comparison)
    else:
        raise QueueEvidenceError("phase-1 gate lacks a scientific prerequisite")
    generation_payload, generation_evidence = _verify_generation(generation)
    report_payload, report_evidence = _verify_controller_report(
        report,
        controller_identity=generation_payload["controller_identity"],
    )
    flags = _phase1_flags(report_payload)
    fast_path_coverage = _fast_path_coverage(report_payload)
    l1_eligible = all(
        flags[name] for name in ("trace_exactness", "oracle_l1", "paired_tts_l1", "learned_l1")
    )
    l2_eligible = all(
        flags[name] for name in ("trace_exactness", "oracle_l2", "paired_tts_l2", "learned_l2")
    )
    payload = {
        "schema_version": 1,
        "status": "phase1_controller_gated",
        "scope": "matched_dflash_controller_phase1",
        "controller_identity_sha256": generation_payload[
            "controller_identity_sha256"
        ],
        "helper_source_sha256": _sha256(Path(__file__).resolve()),
        "queue_source_sha256": (
            _sha256(queue_source.resolve()) if queue_source is not None else None
        ),
        "flags": flags,
        "fast_path_coverage": fast_path_coverage,
        "l1_eligible": l1_eligible,
        "l2_eligible": l2_eligible,
        # This is deliberately a readiness gate, not an L3 performance claim.
        "l3_phase2_allowed": bool(
            flags["trace_exactness"] and flags["l3_evaluation_ready"]
        ),
    }
    return _publish_receipt(
        output,
        payload,
        [
            *prerequisite_evidence,
            *generation_evidence,
            *report_evidence,
            Path(__file__).resolve(),
            *([queue_source.resolve()] if queue_source is not None else []),
        ],
    )


def _verify_phase1_gate(path: Path) -> tuple[dict[str, Any], list[Path]]:
    payload, evidence = _receipt_evidence(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "phase1_controller_gated"
        or payload.get("scope") != "matched_dflash_controller_phase1"
    ):
        raise QueueEvidenceError("phase-1 gate receipt mismatch")
    flags = payload.get("flags")
    if not isinstance(flags, Mapping) or set(flags) != set(
        _phase1_flags({
            "oracle_replay_gate": {},
            "tts_paired_gate": {},
            "learned_policy_gate": {},
            "trace_exactness": {},
            "l3_gate": {},
        })
    ) or not all(isinstance(value, bool) for value in flags.values()):
        raise QueueEvidenceError("phase-1 gate flags are malformed")
    coverage = payload.get("fast_path_coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("source") != "calibration_only_v1"
        or coverage.get("enablement_contract")
        != "calibration_constructs_fast_path_heldout_policy_ci_enables_v1"
        or not isinstance(coverage.get("calibration"), Mapping)
        or not isinstance(coverage.get("heldout"), Mapping)
    ):
        raise QueueEvidenceError("phase-1 fast-path coverage contract mismatch")
    if (
        payload.get("helper_source_sha256") != _sha256(Path(__file__).resolve())
        or not _is_sha256(payload.get("queue_source_sha256"))
        or not any(
            _sha256(item) == payload["queue_source_sha256"]
            for item in evidence
            if item.is_file()
        )
    ):
        raise QueueEvidenceError("phase-1 queue/helper source binding mismatch")
    return payload, evidence


def _verify_ledger(path: Path) -> list[Path]:
    path = path.resolve()
    payload = _load_json(path)
    evidence = [path]
    root = path.parent
    for relative, row in payload.items():
        if not isinstance(relative, str) or not isinstance(row, Mapping):
            raise QueueEvidenceError(f"invalid ledger row: {path}")
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise QueueEvidenceError(f"ledger path escapes root: {artifact}") from exc
        if (
            not artifact.is_file()
            or row.get("sha256") != _sha256(artifact)
            or row.get("bytes") != artifact.stat().st_size
        ):
            raise QueueEvidenceError(f"ledger evidence drift: {artifact}")
        evidence.append(artifact)
    return evidence


def _tree_evidence(trace_root: Path) -> list[Path]:
    trace_root = trace_root.resolve()
    if not trace_root.is_dir():
        raise QueueEvidenceError(f"trace root is missing: {trace_root}")
    ledgers = sorted(trace_root.rglob("hashes.json"))
    if not ledgers:
        raise QueueEvidenceError(f"trace root has no run ledgers: {trace_root}")
    evidence: list[Path] = []
    for ledger in ledgers:
        evidence.extend(_verify_ledger(ledger))
    return sorted(set(evidence))


def _l3_flags(report: Mapping[str, Any]) -> dict[str, bool]:
    l3 = report.get("l3_gate", {})
    utility = l3.get("heldout_transported_utility_gate", {})
    exactness = l3.get("exactness", {})
    return {
        "enabled": l3.get("enabled") is True,
        "heldout_utility": bool(
            utility.get("complete") is True
            and utility.get("eligible") is True
            and utility.get("utility_metric")
            == "survival_weighted_accepted_prefix_v1"
            and utility.get("l3_contract")
            == "joint_fisher_transport_adamw_damping_v1"
            and _integer_at_least(utility.get("n_test_groups"), 8)
            and _finite_positive_ci(utility.get("ci95_vs_tts"))
            and _finite_positive_ci(utility.get("ci95_vs_l2"))
        ),
        "phase2_exactness": bool(
            exactness.get("verified") is True
            and exactness.get("violation_count") == 0
            and _integer_at_least(exactness.get("rounds_checked"), 1)
        ),
        "pairing": utility.get("pairing_contract")
        == "exact_request_seed_concurrency_trace_stage_v1",
    }


def finalize(
    *,
    root: Path,
    comparison: Path | None,
    tts_foundation_terminal: Path | None = None,
    generation: Path,
    phase1_gate_path: Path,
    trace_root: Path,
    final_report: Path | None,
    queue_source: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if tts_foundation_terminal is not None:
        _, prerequisite_evidence = _headline_foundation(tts_foundation_terminal)
    elif comparison is not None:
        _, prerequisite_evidence = _verify_comparison(comparison)
    else:
        raise QueueEvidenceError("controller finalization lacks a scientific prerequisite")
    generation_payload, generation_evidence = _verify_generation(generation)
    phase1, phase1_evidence = _verify_phase1_gate(phase1_gate_path)
    if phase1.get("controller_identity_sha256") != generation_payload.get(
        "controller_identity_sha256"
    ):
        raise QueueEvidenceError("phase-1/controller generation identities differ")
    if (
        queue_source is None
        or phase1.get("queue_source_sha256") != _sha256(queue_source.resolve())
        or phase1.get("helper_source_sha256") != _sha256(Path(__file__).resolve())
    ):
        raise QueueEvidenceError("queue/helper source changed between phases")
    evidence = [
        *prerequisite_evidence,
        *generation_evidence,
        *phase1_evidence,
        *_tree_evidence(trace_root),
        Path(__file__).resolve(),
        *([queue_source.resolve()] if queue_source is not None else []),
    ]
    l3_flags = {
        "enabled": False,
        "heldout_utility": False,
        "phase2_exactness": False,
        "pairing": False,
    }
    if final_report is not None:
        report, report_evidence = _verify_controller_report(
            final_report,
            controller_identity=generation_payload["controller_identity"],
        )
        evidence.extend(report_evidence)
        if (
            _phase1_flags(report) != phase1["flags"]
            or _fast_path_coverage(report) != phase1["fast_path_coverage"]
        ):
            raise QueueEvidenceError(
                "phase-2 replay changed the frozen phase-1 L1/L2 policy"
            )
        l3_flags = _l3_flags(report)
    eligible = {
        "l1": phase1.get("l1_eligible") is True,
        "l2": phase1.get("l2_eligible") is True,
        "l3": all(l3_flags.values()),
    }
    reasons = [f"{method}_evidence_gate_failed" for method, ok in eligible.items() if not ok]
    status = (
        "matched_controller_selected"
        if all(eligible.values())
        else "matched_controller_blocked"
    )
    terminal_name = SELECTED if all(eligible.values()) else BLOCKED
    conflicting = root / (BLOCKED if all(eligible.values()) else SELECTED)
    if conflicting.exists() or Path(str(conflicting) + ".sha256").exists():
        raise QueueEvidenceError(f"conflicting controller terminal exists: {conflicting}")
    payload = {
        "schema_version": 1,
        "status": status,
        "scope": "matched_dflash_l1_l2_l3_evidence",
        "controller_identity_sha256": generation_payload[
            "controller_identity_sha256"
        ],
        "helper_source_sha256": _sha256(Path(__file__).resolve()),
        "queue_source_sha256": (
            _sha256(queue_source.resolve()) if queue_source is not None else None
        ),
        "eligible": eligible,
        "blocked_reasons": reasons,
        "phase1_flags": phase1["flags"],
        "fast_path_coverage": phase1["fast_path_coverage"],
        "l3_flags": l3_flags,
        "phase2_executed": final_report is not None,
    }
    return _publish_receipt(root / terminal_name, payload, evidence)


def _headline_terminal(path: Path) -> tuple[dict[str, Any], list[Path]]:
    payload, evidence = _receipt_evidence(path)
    expected = {
        SELECTED: "matched_controller_selected",
        BLOCKED: "matched_controller_blocked",
    }
    if path.name not in expected or (
        payload.get("schema_version") != 1
        or payload.get("status") != expected[path.name]
        or payload.get("scope") != "matched_dflash_l1_l2_l3_evidence"
    ):
        raise QueueEvidenceError("final P5 builder requires a controller gate terminal")
    conflicting = path.with_name(BLOCKED if path.name == SELECTED else SELECTED)
    if conflicting.exists() or Path(str(conflicting) + ".sha256").exists():
        raise QueueEvidenceError(f"conflicting controller terminal exists: {conflicting}")
    eligible = payload.get("eligible")
    if (
        not isinstance(eligible, Mapping)
        or set(eligible) != {"l1", "l2", "l3"}
        or not all(isinstance(value, bool) for value in eligible.values())
    ):
        raise QueueEvidenceError("controller terminal eligibility map is malformed")
    if path.name == SELECTED and not all(eligible.values()):
        raise QueueEvidenceError("selected controller terminal has an ineligible method")
    return payload, evidence


def _headline_foundation(path: Path) -> tuple[dict[str, int], list[Path]]:
    payload, evidence = _receipt_evidence(path)
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "TTS_0_40K_CONFIRMED"
        or payload.get("scope") != "tts_0_40k_foundation"
        or payload.get("formal_acceptance_foundation_pass") is not True
    ):
        raise QueueEvidenceError("final P5 requires a confirmed TTS foundation")
    identity = payload.get("identity")
    if (
        not isinstance(identity, Mapping)
        or payload.get("identity_sha256") != _sha256_json(identity)
    ):
        raise QueueEvidenceError("TTS foundation identity digest mismatch")
    roles = payload.get("roles")
    aliases = {
        "acceptance_best": "tts_acceptance_best",
        "engineering_best": "tts_engineering_best",
        "same_stride": "same_stride_tts_for_l0",
    }
    if not isinstance(roles, Mapping) or not set(aliases.values()).issubset(roles):
        raise QueueEvidenceError("TTS foundation role mapping is incomplete")
    resolved: dict[str, int] = {}
    for canonical, source in aliases.items():
        row = roles[source]
        stride = row.get("stride") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("method") != "tts"
            or isinstance(stride, bool)
            or not isinstance(stride, int)
            or stride <= 0
        ):
            raise QueueEvidenceError(f"invalid TTS foundation role: {source}")
        resolved[canonical] = stride
    return resolved, evidence


def _headline_strides(comparison: Mapping[str, Any]) -> tuple[int, int]:
    identity = comparison.get("analysis_identity")
    digest = comparison.get("analysis_identity_sha256")
    if (
        not isinstance(identity, Mapping)
        or not _is_sha256(digest)
        or _sha256_json(identity) != digest
    ):
        raise QueueEvidenceError("formal comparison analysis identity is invalid")
    comparisons = identity.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise QueueEvidenceError("formal comparison lacks stride bindings")
    tts = comparisons.get("tts_best_vs_static")
    l0 = comparisons.get("l0_best_vs_tts_best")
    if (
        not isinstance(tts, Mapping)
        or not isinstance(l0, Mapping)
        or tts.get("candidate_method") != "tts"
        or tts.get("baseline_method") != "static"
        or tts.get("baseline_update_stride") != 1
        or l0.get("candidate_method") != "naive_async"
        or l0.get("baseline_method") != "tts"
        or l0.get("baseline_update_stride")
        != tts.get("candidate_update_stride")
    ):
        raise QueueEvidenceError("formal comparison role/stride contract is malformed")
    tts_stride = tts.get("candidate_update_stride")
    l0_stride = l0.get("candidate_update_stride")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (tts_stride, l0_stride)
    ):
        raise QueueEvidenceError("formal comparison selected an invalid stride")
    return int(tts_stride), int(l0_stride)


def _publish_headline_manifest(manifest, path: Path) -> dict[str, str]:
    path = path.resolve()
    try:
        manifest.write(path)
    except Exception as exc:
        raise QueueEvidenceError(f"cannot publish final P5 manifest: {exc}") from exc
    sidecar = _verify_sidecar(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "manifest_sha256": manifest.content_sha256(),
        "sidecar_path": str(sidecar),
        "sidecar_sha256": _sha256(sidecar),
    }


def build_headline_manifests(
    *,
    comparison: Path | None,
    generation: Path,
    terminal: Path,
    tts_foundation_terminal: Path,
    output_dir: Path,
    output: Path,
    controller_report: Path | None = None,
) -> dict[str, Any]:
    """Generate the final algorithmic and MFU manifests from frozen evidence.

    Static/TTS/L0 are unconditional.  Controller methods enter only when the
    terminal's independent L1/L2/L3 evidence flag is true.  The two load
    profiles are separate manifests, preventing a mixed-load wall clock from
    becoming a context-specific speed claim.
    """

    from lightcone_spec.config.schema import (
        MODEL_PAIRS,
        canonical_tail_layout_mode,
        canonical_weight_update_mode,
    )
    from lightcone_spec.orchestration import catalog as catalog_module
    from lightcone_spec.orchestration.catalog import (
        P5_PRIORITY_FINAL_CONTEXTS,
        p5_priority_dflash_final_manifest,
    )

    comparison_evidence: list[Path] = []
    if comparison is not None:
        _, comparison_evidence = _verify_comparison(comparison)
    generation_payload, generation_evidence = _verify_generation(generation)
    terminal_payload, terminal_evidence = _headline_terminal(terminal.resolve())
    tts_role_strides, foundation_evidence = _headline_foundation(
        tts_foundation_terminal.resolve()
    )
    identity = generation_payload["controller_identity"]
    identity_sha = generation_payload["controller_identity_sha256"]
    if terminal_payload.get("controller_identity_sha256") != identity_sha:
        raise QueueEvidenceError("controller terminal/generation identity mismatch")

    adaptation_stride = identity.get("update_stride")
    if (
        isinstance(adaptation_stride, bool)
        or not isinstance(adaptation_stride, int)
        or adaptation_stride <= 0
    ):
        raise QueueEvidenceError("controller identity has an invalid update stride")
    if identity.get("update_stride") != adaptation_stride:
        raise QueueEvidenceError(
            "controller identity does not use the formally confirmed L0 stride"
        )
    model_pair = identity.get("model_pair")
    if model_pair not in MODEL_PAIRS:
        raise QueueEvidenceError(f"unknown selected model pair: {model_pair!r}")
    pair = MODEL_PAIRS[model_pair]
    if pair.get("speculative_algorithm") != "DFLASH":
        raise QueueEvidenceError("final DFlash builder received another backend")

    try:
        mode = canonical_weight_update_mode(str(identity.get("weight_update_mode")))
        tail_layout = canonical_tail_layout_mode(mode)
    except ValueError as exc:
        raise QueueEvidenceError(f"invalid selected tail mode: {exc}") from exc
    if identity.get("tail_layout_mode") != tail_layout:
        raise QueueEvidenceError("controller tail mode aliases do not canonicalize")
    rank = identity.get("adapter_rank")
    if mode == "full":
        if rank is not None:
            raise QueueEvidenceError("full-rank tail identity must have null rank")
        unit_rank = 16
    else:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise QueueEvidenceError("ranked tail identity lacks a positive rank")
        unit_rank = rank
    if str(identity.get("optimizer", "")).lower() != "adamw":
        raise QueueEvidenceError("final P5 requires the confirmed AdamW optimizer")
    try:
        lr = float(identity["lr"])
        weight_decay = float(identity["weight_decay"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise QueueEvidenceError("controller optimizer identity is incomplete") from exc
    if not math.isfinite(lr) or lr <= 0.0 or not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise QueueEvidenceError("controller optimizer values are outside safe bounds")
    if identity.get("lifecycle") != "stream":
        raise QueueEvidenceError("final P5 requires stream lifecycle")

    bindings = identity.get("bindings")
    locked = generation_payload.get("locked_inputs")
    if not isinstance(bindings, Mapping) or not isinstance(locked, Mapping):
        raise QueueEvidenceError("controller identity lacks locked inputs")
    lock_sha = bindings.get("lockfile_sha256")
    if not _is_sha256(lock_sha) or locked.get("lockfile_sha256") != lock_sha:
        raise QueueEvidenceError("final P5 lockfile binding is inconsistent")
    model_roots_sha = bindings.get("model_roots_sha256")
    if (
        not _is_sha256(model_roots_sha)
        or locked.get("model_roots_sha256") != model_roots_sha
    ):
        raise QueueEvidenceError("final P5 model-root binding is inconsistent")
    revisions = bindings.get("model_revisions")
    if (
        not isinstance(revisions, Mapping)
        or set(revisions) != {"target", "drafter", "tokenizer"}
        or locked.get("model_revisions") != revisions
    ):
        raise QueueEvidenceError("final P5 model revisions are inconsistent")
    runtime_fingerprint = bindings.get("runtime_implementation_fingerprint")
    if not isinstance(runtime_fingerprint, Mapping):
        raise QueueEvidenceError("final P5 lacks a runtime implementation binding")
    runtime_body = dict(runtime_fingerprint)
    runtime_sha = runtime_body.pop("sha256", None)
    if (
        runtime_body.get("schema_version") != 1
        or not _is_sha256(runtime_sha)
        or _sha256_json(runtime_body) != runtime_sha
    ):
        raise QueueEvidenceError("final P5 runtime implementation binding is invalid")

    contexts = tuple(P5_PRIORITY_FINAL_CONTEXTS)
    if contexts != (512, 1024, 2048, 4096, 8192, 16384, 32768, 40000):
        raise QueueEvidenceError("final P5 context grid drifted")
    max_new_tokens = 512
    checkpoint_limit = pair.get("max_context_length")
    if (
        isinstance(checkpoint_limit, bool)
        or not isinstance(checkpoint_limit, int)
        or max(contexts) + max_new_tokens > checkpoint_limit
    ):
        raise QueueEvidenceError(
            f"{model_pair} cannot run prefix=40000 plus {max_new_tokens} output tokens"
        )

    eligible = dict(terminal_payload["eligible"])
    methods = ["static", "tts", "naive_async"]
    for gate, method in (
        ("l1", "lc_gate"),
        ("l2", "lc_damp"),
        ("l3", "lc_transport"),
    ):
        if eligible[gate]:
            methods.append(method)

    report_evidence: list[Path] = []
    controller_artifact = None
    if len(methods) > 3:
        if controller_report is None:
            raise QueueEvidenceError(
                "eligible controller methods require their frozen replay report"
            )
        report, report_evidence = _verify_controller_report(
            controller_report,
            controller_identity=identity,
        )
        report_flags = {
            "l1": all(
                _phase1_flags(report)[name]
                for name in ("trace_exactness", "oracle_l1", "paired_tts_l1", "learned_l1")
            ),
            "l2": all(
                _phase1_flags(report)[name]
                for name in ("trace_exactness", "oracle_l2", "paired_tts_l2", "learned_l2")
            ),
            "l3": all(_l3_flags(report).values()),
        }
        if report_flags != eligible:
            raise QueueEvidenceError("controller artifact eligibility differs from terminal")
        controller_artifact = Path(report["artifact_path"]).resolve()

    terminal_sha = _sha256(terminal.resolve())
    slug = f"b{identity_sha[:12]}_g{terminal_sha[:12]}_v1"
    output_dir = output_dir.resolve()
    algorithmic = p5_priority_dflash_final_manifest(
        name=f"p5_priority_dflash_0_40k_algorithmic_{slug}",
        model_pair=str(model_pair),
        trainable_scope=tail_layout,
        adapter_rank=unit_rank,
        methods=tuple(methods),
        load_groups=(("p5_ctx_512-40000", 4),),
        tts_stride=None,
        tts_role_strides=tts_role_strides,
        adaptation_stride=adaptation_stride,
        lr=lr,
        weight_decay=weight_decay,
        lockfile_sha256=str(lock_sha),
        model_roots_sha256=str(model_roots_sha),
        locked_model_revisions=dict(revisions),
        runtime_implementation_fingerprint=dict(runtime_fingerprint),
        controller_identity_sha256=str(identity_sha),
        claim_scope="paired_context_algorithmic_c4",
    )
    mfu = p5_priority_dflash_final_manifest(
        name=f"p5_priority_dflash_0_40k_mfu_{slug}",
        model_pair=str(model_pair),
        trainable_scope=tail_layout,
        adapter_rank=unit_rank,
        methods=tuple(methods),
        load_groups=(
            ("p5_ctx_512-4096", 48),
            ("p5_ctx_8192-16384", 20),
            ("p5_ctx_32768-40000", 8),
        ),
        tts_stride=None,
        tts_role_strides=tts_role_strides,
        adaptation_stride=adaptation_stride,
        lr=lr,
        weight_decay=weight_decay,
        lockfile_sha256=str(lock_sha),
        model_roots_sha256=str(model_roots_sha),
        locked_model_revisions=dict(revisions),
        runtime_implementation_fingerprint=dict(runtime_fingerprint),
        controller_identity_sha256=str(identity_sha),
        claim_scope="context_specific_mfu_load",
    )

    max_total_tokens = 400000
    draft_tokens = int(pair["default_num_draft_tokens"])
    load_contracts = {
        "ALGORITHMIC_C4": {context: 4 for context in contexts},
        "MFU_CONTEXT_LOAD": {
            **{context: 48 for context in contexts if context <= 4096},
            **{context: 20 for context in contexts if 8192 <= context <= 16384},
            **{context: 8 for context in contexts if context >= 32768},
        },
    }
    capacity_rows = []
    for profile, loads in load_contracts.items():
        for context, concurrency in loads.items():
            required = concurrency * (context + max_new_tokens + draft_tokens)
            if required > max_total_tokens:
                raise QueueEvidenceError(
                    f"{profile} context={context}/c={concurrency} requires "
                    f"{required} KV slots, above {max_total_tokens}"
                )
            capacity_rows.append(
                {
                    "profile": profile,
                    "context_length": context,
                    "concurrency": concurrency,
                    "required_kv_token_slots": required,
                }
            )

    algorithmic_record = _publish_headline_manifest(
        algorithmic, output_dir / f"{algorithmic.name}.json"
    )
    mfu_record = _publish_headline_manifest(mfu, output_dir / f"{mfu.name}.json")
    payload = {
        "schema_version": 1,
        "status": "final_0_40k_manifests_generated",
        "scope": "evidence_bound_context_specific_p5",
        "controller_identity_sha256": identity_sha,
        "controller_terminal_sha256": terminal_sha,
        "tts_foundation_terminal_sha256": _sha256(
            tts_foundation_terminal.resolve()
        ),
        "tts_role_strides": tts_role_strides,
        "optimizer_identity": {
            "model_pair": model_pair,
            "weight_update_mode": mode,
            "tail_layout_mode": tail_layout,
            "adapter_rank": rank,
            "optimizer": "adamw",
            "lr": lr,
            "weight_decay": weight_decay,
            "tts_role_strides": tts_role_strides,
            "adaptation_stride": adaptation_stride,
        },
        "eligible": eligible,
        "methods": methods,
        "controller_artifact_path": (
            str(controller_artifact) if controller_artifact is not None else None
        ),
        "controller_root": (
            str(controller_artifact.parent)
            if controller_artifact is not None
            else None
        ),
        "context_contract": {
            "contexts": list(contexts),
            "same_dataset_prompt_seed": True,
            "dataset": "livecodebench",
            "seed": 0,
            "trajectory": "independent_exact_prefix_checkpoint",
            "timing": "independent_exact_context_group_v1",
            "checkpoint_limit": checkpoint_limit,
            "max_new_tokens": max_new_tokens,
            "max_total_tokens": max_total_tokens,
            "load_profiles_are_not_pooled": True,
            "prompt_offset": 184,
            "prompt_limit": 48,
            "capacity": capacity_rows,
        },
        "artifacts": {
            "ALGORITHMIC_C4": algorithmic_record,
            "MFU_CONTEXT_LOAD": mfu_record,
        },
    }
    return _publish_receipt(
        output,
        payload,
        [
            *comparison_evidence,
            *generation_evidence,
            *terminal_evidence,
            *foundation_evidence,
            *report_evidence,
            Path(__file__).resolve(),
            Path(catalog_module.__file__).resolve(),
            Path(algorithmic_record["path"]),
            Path(algorithmic_record["sidecar_path"]),
            Path(mfu_record["path"]),
            Path(mfu_record["sidecar_path"]),
        ],
    )


def archive_failure(root: Path) -> None:
    root = root.resolve()
    failed = root / FAILED
    sidecar = Path(str(failed) + ".sha256")
    if not failed.exists() and not sidecar.exists():
        return
    _verify_sidecar(failed)
    digest = _sha256(failed)
    archive = root / "attempts"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"CONTROLLER_FAILED.{digest[:16]}.json"
    target_sidecar = Path(str(target) + ".sha256")
    if target.exists():
        if _sha256(target) != digest or not target_sidecar.is_file():
            raise QueueEvidenceError(f"failed-attempt archive collision: {target}")
        failed.unlink()
        sidecar.unlink()
    else:
        os.replace(failed, target)
        os.replace(sidecar, target_sidecar)


def write_failure(
    *, root: Path, phase: str, return_code: int, evidence: Iterable[Path]
) -> dict[str, Any]:
    existing = [Path(item).resolve() for item in evidence if Path(item).is_file()]
    if not existing:
        marker = root.resolve() / "queue-state.jsonl"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
        existing = [marker]
    return _publish_receipt(
        root.resolve() / FAILED,
        {
            "schema_version": 1,
            "status": "matched_controller_failed_resumable",
            "scope": "matched_dflash_controller_queue_attempt",
            "phase": phase,
            "return_code": int(return_code),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        [*existing, Path(__file__).resolve()],
    )


def terminal_status(root: Path) -> str:
    root = root.resolve()
    paths = {name: root / name for name in (SELECTED, BLOCKED, FAILED)}
    present = [
        name
        for name, path in paths.items()
        if path.exists() or Path(str(path) + ".sha256").exists()
    ]
    if len(present) > 1:
        raise QueueEvidenceError(f"conflicting controller terminals: {present}")
    if not present:
        return "none"
    name = present[0]
    payload, _ = _receipt_evidence(paths[name])
    expected = {
        SELECTED: "matched_controller_selected",
        BLOCKED: "matched_controller_blocked",
        FAILED: "matched_controller_failed_resumable",
    }[name]
    if payload.get("status") != expected:
        raise QueueEvidenceError(f"controller terminal status mismatch: {name}")
    return {SELECTED: "selected", BLOCKED: "blocked", FAILED: "failed"}[name]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("terminal-status")
    status.add_argument("--root", type=Path, required=True)
    archive = commands.add_parser("archive-failure")
    archive.add_argument("--root", type=Path, required=True)
    preblock = commands.add_parser("block-confirmation")
    preblock.add_argument("--root", type=Path, required=True)
    preblock.add_argument("--comparison", type=Path, required=True)
    preblock.add_argument("--queue-source", type=Path, required=True)
    gate = commands.add_parser("phase1-gate")
    gate.add_argument("--comparison", type=Path)
    gate.add_argument("--tts-foundation-terminal", type=Path)
    gate.add_argument("--generation", type=Path, required=True)
    gate.add_argument("--report", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--queue-source", type=Path, required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--root", type=Path, required=True)
    finish.add_argument("--comparison", type=Path)
    finish.add_argument("--tts-foundation-terminal", type=Path)
    finish.add_argument("--generation", type=Path, required=True)
    finish.add_argument("--phase1-gate", type=Path, required=True)
    finish.add_argument("--trace-root", type=Path, required=True)
    finish.add_argument("--final-report", type=Path)
    finish.add_argument("--queue-source", type=Path, required=True)
    headline = commands.add_parser("build-headline")
    headline.add_argument("--comparison", type=Path)
    headline.add_argument("--generation", type=Path, required=True)
    headline.add_argument("--terminal", type=Path, required=True)
    headline.add_argument("--tts-foundation-terminal", type=Path, required=True)
    headline.add_argument("--controller-report", type=Path)
    headline.add_argument("--output-dir", type=Path, required=True)
    headline.add_argument("--output", type=Path, required=True)
    failure = commands.add_parser("write-failure")
    failure.add_argument("--root", type=Path, required=True)
    failure.add_argument("--phase", required=True)
    failure.add_argument("--return-code", type=int, required=True)
    failure.add_argument("--evidence", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "terminal-status":
            print(terminal_status(args.root))
        elif args.command == "archive-failure":
            archive_failure(args.root)
        elif args.command == "block-confirmation":
            block_confirmation(
                root=args.root,
                comparison=args.comparison,
                queue_source=args.queue_source,
            )
        elif args.command == "phase1-gate":
            result = phase1_gate(
                comparison=args.comparison,
                tts_foundation_terminal=args.tts_foundation_terminal,
                generation=args.generation,
                report=args.report,
                output=args.output,
                queue_source=args.queue_source,
            )
            print("1" if result["l3_phase2_allowed"] else "0")
        elif args.command == "finalize":
            result = finalize(
                root=args.root,
                comparison=args.comparison,
                tts_foundation_terminal=args.tts_foundation_terminal,
                generation=args.generation,
                phase1_gate_path=args.phase1_gate,
                trace_root=args.trace_root,
                final_report=args.final_report,
                queue_source=args.queue_source,
            )
            print("selected" if result["status"] == "matched_controller_selected" else "blocked")
        elif args.command == "build-headline":
            build_headline_manifests(
                comparison=args.comparison,
                generation=args.generation,
                terminal=args.terminal,
                tts_foundation_terminal=args.tts_foundation_terminal,
                controller_report=args.controller_report,
                output_dir=args.output_dir,
                output=args.output,
            )
        else:
            write_failure(
                root=args.root,
                phase=args.phase,
                return_code=args.return_code,
                evidence=args.evidence,
            )
    except QueueEvidenceError as exc:
        raise SystemExit(f"matched controller queue evidence failure: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
