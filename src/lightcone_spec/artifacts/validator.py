"""Artifact validator (`lightcone-spec validate-artifacts`, spec 9.2).

Checks per run directory: required files, hash integrity, Parquet
readability, schema fields, event monotonicity, version consistency,
decision enum, seed/pair/sampling consistency against the manifest, and
duplicate/missing run units against an expected-unit list. Returns a
structured report; any error means a non-zero exit (code 8), and a
failed-exactness unit invalidates every performance summary it touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from lightcone_spec.artifacts.rundir import REQUIRED_FILES, RunDirectory
from lightcone_spec.artifacts.schemas import (
    DECISION_ENUM,
    SCHEMA_COMPAT_OPTIONAL_FIELDS,
    TABLES,
)
from lightcone_spec.locking.hashing import canonical_json, sha256_bytes, sha256_file
from lightcone_spec.orchestration.units import RunUnit


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_runs: list[str] = field(default_factory=list)
    run_status: dict[str, str] = field(default_factory=dict)
    unit_status: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_runs": self.checked_runs,
            "run_status": self.run_status,
            "unit_status": self.unit_status,
        }


def _err(report: ValidationReport, run_id: str, msg: str) -> None:
    report.errors.append(f"[{run_id}] {msg}")


def validate_run_dir(path: str | Path, report: ValidationReport) -> str | None:
    """Validate a single run directory; returns the unit_id if readable."""
    path = Path(path)
    run_id = path.name
    report.checked_runs.append(run_id)
    run_error_start = len(report.errors)

    missing_files = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    for name in missing_files:
        _err(report, run_id, f"missing required file {name}")
    if missing_files:
        return None

    # Manifest hash.
    try:
        manifest_text = (path / "manifest.json").read_text()
        manifest = json.loads(manifest_text)
        expected = (path / "manifest.sha256").read_text().strip()
        actual = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _err(report, run_id, f"unreadable manifest: {exc}")
        return None
    if not isinstance(manifest, dict):
        _err(report, run_id, "manifest must be a JSON object")
        return None
    if actual != expected:
        _err(report, run_id, "manifest.sha256 mismatch")
    claimed_unit_id = manifest.get("unit_id")
    if not isinstance(claimed_unit_id, str) or not claimed_unit_id:
        _err(report, run_id, "manifest unit_id must be a non-empty string")
        unit_id = run_id
    else:
        unit_id = claimed_unit_id
        try:
            # A scoped validator must not trust the field that decides whether a
            # run is in scope.  Reconstructing the canonical RunUnit validates
            # the claim against every identity field (including supported
            # schema-v1 aliases) before it can satisfy expected-unit coverage.
            RunUnit.from_dict(manifest)
        except (KeyError, TypeError, ValueError) as exc:
            _err(report, run_id, f"manifest unit identity is invalid: {exc}")

    try:
        exit_info = json.loads((path / "exit.json").read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _err(report, run_id, f"unreadable exit.json: {exc}")
        report.run_status[run_id] = "invalid_artifact"
        report.unit_status[unit_id] = "invalid_artifact"
        return unit_id
    declared_status = exit_info.get("status")
    expected_exit_codes = {
        "complete_valid": 0,
        "failed_exactness": 5,
        "failed_runtime": 7,
        "resource_skip": 4,
    }
    if declared_status not in expected_exit_codes:
        _err(report, run_id, f"invalid exit status {declared_status!r}")
    elif exit_info.get("exit_code") != expected_exit_codes[declared_status]:
        _err(
            report,
            run_id,
            f"exit code/status mismatch: {exit_info.get('exit_code')} for "
            f"{declared_status}",
        )

    # File hashes.
    try:
        hashes = json.loads((path / "hashes.json").read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _err(report, run_id, f"unreadable hashes.json: {exc}")
        report.run_status[run_id] = "invalid_artifact"
        report.unit_status[unit_id] = "invalid_artifact"
        return unit_id
    if not isinstance(hashes, dict):
        _err(report, run_id, "hashes.json must be an object")
        hashes = {}
    for name in REQUIRED_FILES:
        if name != "hashes.json" and name not in hashes:
            _err(report, run_id, f"required file is not hash-bound: {name}")
    for name, entry in hashes.items():
        p = path / name
        if not p.is_file():
            _err(report, run_id, f"hashed file missing: {name}")
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            _err(report, run_id, f"malformed hash entry: {name}")
            continue
        if entry.get("bytes") != p.stat().st_size:
            _err(report, run_id, f"size drift: {name}")
        if sha256_file(p) != entry["sha256"]:
            _err(report, run_id, f"hash drift: {name}")

    # Conditional raw evidence uses the same hash ledger as the normative
    # Parquet artifacts.  A file merely existing beside a completed run is not
    # provenance: every JSONL/checkpoint must be bound, and successful real
    # runs must retain at least one non-empty telemetry shard.
    runtime_paths = sorted((path / "runtime").glob("*.jsonl"))
    checkpoint_path = path / "prefix-checkpoints.json"
    auxiliary_paths = list(runtime_paths)
    if checkpoint_path.is_file():
        auxiliary_paths.append(checkpoint_path)
    for auxiliary in auxiliary_paths:
        relative = auxiliary.relative_to(path).as_posix()
        if relative not in hashes:
            _err(report, run_id, f"unhashed provenance file: {relative}")
    complete_valid = declared_status == "complete_valid"
    is_real_model = not str(manifest.get("model_pair", "")).startswith("toy_")
    if complete_valid and is_real_model and not any(
        telemetry.stat().st_size > 0 for telemetry in runtime_paths
    ):
        _err(report, run_id, "complete real-model run has no raw runtime telemetry")
    if complete_valid and str(manifest.get("phase", "")).startswith("p5"):
        if not checkpoint_path.is_file():
            _err(report, run_id, "complete P5 run has no prefix-checkpoints.json")
        elif checkpoint_path.name not in hashes:
            _err(report, run_id, "prefix-checkpoints.json is not hash-bound")
        else:
            try:
                checkpoints = json.loads(checkpoint_path.read_text()).get(
                    "checkpoints", []
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                _err(report, run_id, f"invalid prefix-checkpoints.json: {exc}")
            else:
                if not checkpoints:
                    _err(report, run_id, "prefix-checkpoints.json has no checkpoints")

    # Parquet readability + schemas.
    tables = {}
    for name, schema in TABLES.items():
        p = path / f"{name}.parquet"
        try:
            table = pq.read_table(p)
        except Exception as exc:
            _err(report, run_id, f"corrupt parquet {name}: {exc}")
            continue
        missing = (
            set(schema.names)
            - set(table.schema.names)
            - set(SCHEMA_COMPAT_OPTIONAL_FIELDS.get(name, ()))
        )
        if missing:
            _err(report, run_id, f"{name}: missing fields {sorted(missing)}")
        tables[name] = table

    # Consistency checks against the manifest.
    expect_keys = {
        "seed": manifest.get("seed"),
        "model_pair_id": manifest.get("model_pair"),
        "method": manifest.get("method"),
        "lifecycle": manifest.get("lifecycle"),
    }
    for name, table in tables.items():
        if table.num_rows == 0:
            continue
        cols = table.to_pydict()
        for key, expected_val in expect_keys.items():
            if expected_val is None or key not in cols:
                continue
            vals = set(cols[key])
            if vals != {expected_val}:
                _err(
                    report,
                    run_id,
                    f"{name}: column {key} = {sorted(map(str, vals))} but manifest "
                    f"declares {expected_val}",
                )

    # Event monotonicity and version consistency in updates.parquet.
    upd = tables.get("updates")
    if upd is not None and upd.num_rows > 0:
        cols = upd.to_pydict()
        for i in range(upd.num_rows):
            uid = cols["update_id"][i]
            snap, teach = cols["snapshot_ts_us"][i], cols["teacher_ts_us"][i]
            launch, done = cols["launch_ts_us"][i], cols["done_ts_us"][i]
            commit = cols["commit_ts_us"][i]
            reason = cols["failure_reason"][i]
            if reason == "max_in_flight":
                if any(value is not None for value in (teach, launch, done, commit)):
                    _err(
                        report,
                        run_id,
                        f"update {uid}: rejected admission has execution events",
                    )
                continue
            if (
                any(value is None for value in (teach, launch, done))
                or not (snap < teach <= launch <= done)
            ):
                _err(report, run_id, f"update {uid}: event chain not monotone")
            if commit is not None and not (done <= commit):
                _err(report, run_id, f"update {uid}: commit before done")
            pub = cols["published_version"][i]
            src = cols["source_version"][i]
            if pub is not None and pub <= src:
                _err(
                    report,
                    run_id,
                    f"update {uid}: published_version {pub} <= source_version {src}",
                )

    dec = tables.get("decisions")
    if dec is not None and dec.num_rows > 0:
        for i, val in enumerate(dec.to_pydict()["decision"]):
            if val not in DECISION_ENUM:
                _err(report, run_id, f"decisions row {i}: invalid decision {val!r}")

    # Exit status is authoritative; summary rows must agree with it.  Empty
    # normative telemetry is permitted only for an immutable resource/runtime
    # failure record, never for complete_valid.
    summ = tables.get("request_summary")
    status = (
        declared_status
        if declared_status in ("complete_valid", "failed_exactness", "failed_runtime", "resource_skip")
        else "invalid_artifact"
    )
    if declared_status == "complete_valid":
        for name in ("rounds", "system_samples", "request_summary"):
            table = tables.get(name)
            if table is None or table.num_rows == 0:
                _err(report, run_id, f"complete_valid has empty normative table {name}")
    if summ is not None and summ.num_rows > 0:
        cols = summ.to_pydict()
        statuses = set(cols["status"])
        if declared_status == "complete_valid" and statuses != {"complete_valid"}:
            _err(
                report,
                run_id,
                f"complete_valid exit disagrees with request statuses {sorted(statuses)}",
            )
        if declared_status == "failed_exactness" and "failed_exactness" not in statuses:
            _err(report, run_id, "failed_exactness exit has no failed request row")
        if declared_status == "failed_runtime" and "failed_runtime" not in statuses:
            _err(report, run_id, "failed_runtime exit disagrees with request rows")
        for i, request_status in enumerate(cols["status"]):
            if request_status == "failed_exactness" and cols["decode_tps"][i] not in (
                0.0,
                None,
            ):
                _err(
                    report,
                    run_id,
                    "failed_exactness request has a nonzero throughput summary",
                )
    if len(report.errors) > run_error_start:
        status = "invalid_artifact"
    report.run_status[run_id] = status
    report.unit_status[unit_id] = status
    return unit_id


def validate_artifact_root(
    root: str | Path,
    expected_units: list[dict] | None = None,
) -> ValidationReport:
    """Validate all completed run dirs under root; when expected_units is
    provided (from the experiment manifests), check duplicates, missing
    units and required-unit coverage."""
    root = Path(root)
    report = ValidationReport()
    seen_units: dict[str, list[str]] = {}
    expected_unit_ids = (
        {str(unit["unit_id"]) for unit in expected_units}
        if expected_units is not None
        else None
    )
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rd = RunDirectory(root, run_dir.name)
        if not rd.is_complete:
            report.warnings.append(f"[{run_dir.name}] incomplete run dir (ignored)")
            continue
        # Shared artifact roots intentionally contain immutable attempts from
        # several effective manifests (for example one Static run reused by
        # every update-mode sweep).  When a caller supplies expected units,
        # first bind a run through its readable manifest and validate only the
        # requested unit set.  An unbindable/corrupt manifest is merely an
        # unrelated warning here; a required expected unit still fails below
        # as missing.  With no expected set, retain strict whole-root behavior.
        if expected_unit_ids is not None:
            try:
                candidate_manifest = json.loads(
                    (run_dir / "manifest.json").read_text()
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                report.warnings.append(
                    f"[{run_dir.name}] cannot bind run to expected units "
                    f"(ignored): {exc}"
                )
                continue
            if not isinstance(candidate_manifest, dict):
                report.warnings.append(
                    f"[{run_dir.name}] manifest is not an object and cannot "
                    "bind to expected units (ignored)"
                )
                continue
            candidate_unit_id = candidate_manifest.get("unit_id")
            if not isinstance(candidate_unit_id, str) or not candidate_unit_id:
                report.warnings.append(
                    f"[{run_dir.name}] has no bindable unit_id for this "
                    "expected-unit scope (ignored)"
                )
                continue
            if candidate_unit_id not in expected_unit_ids:
                report.warnings.append(
                    f"[{run_dir.name}] unrelated unit {candidate_unit_id} "
                    "(ignored)"
                )
                continue
        unit_id = validate_run_dir(run_dir, report)
        if unit_id is not None:
            seen_units.setdefault(unit_id, []).append(run_dir.name)
    # Aggregate attempts only after every run has an individual status.  A
    # structurally valid failed attempt remains immutable evidence but must not
    # override a later successful retry of the same unit.
    report.unit_status = {}
    for unit_id, runs in seen_units.items():
        statuses = {
            run_id: report.run_status.get(run_id, "invalid_artifact")
            for run_id in runs
        }
        successful = sorted(
            run_id for run_id, status in statuses.items()
            if status == "complete_valid"
        )
        if len(successful) > 1:
            report.errors.append(
                f"unit {unit_id} has duplicate complete_valid runs: {successful}"
            )
            report.unit_status[unit_id] = "invalid_artifact"
        elif successful:
            report.unit_status[unit_id] = "complete_valid"
        elif "invalid_artifact" in statuses.values():
            report.unit_status[unit_id] = "invalid_artifact"
        elif "failed_exactness" in statuses.values():
            report.unit_status[unit_id] = "failed_exactness"
        elif "failed_runtime" in statuses.values():
            report.unit_status[unit_id] = "failed_runtime"
        elif "resource_skip" in statuses.values():
            report.unit_status[unit_id] = "resource_skip"
        else:
            report.unit_status[unit_id] = "invalid_artifact"
    if expected_units is not None:
        for unit in expected_units:
            uid = unit["unit_id"]
            required = unit.get("required", True)
            allow_skip = unit.get("allow_resource_skip", False)
            if uid not in seen_units:
                if required:
                    report.errors.append(f"required unit missing: {uid}")
                    report.unit_status[uid] = "missing"
                continue
            status = report.unit_status.get(uid, "complete_valid")
            if required and status == "resource_skip" and not allow_skip:
                report.errors.append(
                    f"required unit {uid} is resource_skip but skip is not allowed"
                )
            elif required and status in (
                "failed_exactness",
                "failed_runtime",
                "invalid_artifact",
            ):
                report.errors.append(f"required unit {uid} ended with {status}")
    return report
