"""Manifest executor with unit-level resume (spec 9.2 run-manifest).

- Toy/synthetic units run on the CPU reference engine and produce full
  artifact rows locally.
- Real model-pair units require CUDA plus the pinned SGLang fork
  (`sglang_bridge`); on hardware without them execution fails closed
  with the runtime/GPU exit code -- never a silent skip.
- Completed units whose run dir carries the same unit-execution hash are skipped;
  incomplete run dirs are ignored (a fresh run_id restarts the unit from
  the last complete unit boundary).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from lightcone_spec.artifacts.rundir import RunDirectory
from lightcone_spec.config.schema import SYNTHETIC_DATASET_KEYS
from lightcone_spec.exit_codes import (
    ConfigError,
    ExactnessViolation,
    LightconeError,
    NumericalFailure,
    ResourceSkip,
    RuntimeGpuFailure,
)
from lightcone_spec.locking.hashing import sha256_file, sha256_json
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.units import RunUnit


@dataclass
class ExecutionOutcome:
    unit_id: str
    run_id: Optional[str]
    status: str  # complete_valid | failed_* | resource_skip | skipped_existing
    detail: str = ""


@dataclass
class ExecutionReport:
    outcomes: list[ExecutionOutcome] = field(default_factory=list)

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for o in self.outcomes:
            out[o.status] = out.get(o.status, 0) + 1
        return out

    @property
    def ok(self) -> bool:
        return all(
            o.status in ("complete_valid", "skipped_existing", "resource_skip")
            for o in self.outcomes
        )


def _existing_complete_unit(
    artifact_root: Path,
    unit_id: str,
    unit_execution_sha256: str,
) -> Optional[str]:
    if not artifact_root.is_dir():
        return None
    for run_dir in artifact_root.iterdir():
        if not run_dir.is_dir():
            continue
        rd = RunDirectory(artifact_root, run_dir.name)
        if not rd.is_complete:
            continue
        try:
            manifest = rd.read_manifest()
        except Exception:
            continue
        if (
            manifest.get("unit_id") == unit_id
            and manifest.get("unit_execution_sha256") == unit_execution_sha256
            and _is_resumable_complete(rd, manifest)
        ):
            return run_dir.name
    return None


def _hashed_file_is_current(rd: RunDirectory, hashes: dict, relative: str) -> bool:
    entry = hashes.get(relative)
    path = rd.path / relative
    return bool(
        isinstance(entry, dict)
        and isinstance(entry.get("sha256"), str)
        and path.is_file()
        and int(entry.get("bytes", -1)) == path.stat().st_size
        and entry["sha256"] == sha256_file(path)
    )


def _is_resumable_complete(rd: RunDirectory, manifest: dict) -> bool:
    """Accept only a successful, provenance-bound measurement as complete.

    ``hashes.json`` is a finalization marker, not a success marker.  Runtime
    and exactness failures intentionally carry it as immutable failure
    evidence, so resume must also inspect the declared status and the minimum
    normative telemetry needed by every analysis.
    """
    try:
        exit_info = rd.read_exit()
        hashes = json.loads((rd.path / "hashes.json").read_text())
        if exit_info.get("status") != "complete_valid" or int(
            exit_info.get("exit_code", -1)
        ) != 0:
            return False
        for relative in (
            "manifest.json",
            "manifest.sha256",
            "exit.json",
            "rounds.parquet",
            "system_samples.parquet",
            "request_summary.parquet",
        ):
            if not _hashed_file_is_current(rd, hashes, relative):
                return False
        for table_name in ("rounds", "system_samples", "request_summary"):
            if pq.read_metadata(rd.path / f"{table_name}.parquet").num_rows <= 0:
                return False

        runtime_paths = sorted((rd.path / "runtime").glob("*.jsonl"))
        is_real_model = not str(manifest.get("model_pair", "")).startswith("toy_")
        if is_real_model and not any(path.stat().st_size > 0 for path in runtime_paths):
            return False
        for path in runtime_paths:
            relative = path.relative_to(rd.path).as_posix()
            if not _hashed_file_is_current(rd, hashes, relative):
                return False

        checkpoint = rd.path / "prefix-checkpoints.json"
        if str(manifest.get("phase", "")).startswith("p5"):
            if not _hashed_file_is_current(rd, hashes, checkpoint.name):
                return False
            payload = json.loads(checkpoint.read_text())
            if not payload.get("checkpoints"):
                return False
        elif checkpoint.is_file() and not _hashed_file_is_current(
            rd, hashes, checkpoint.name
        ):
            return False
    except Exception:
        return False
    return True


def _finalize_run(
    rd: RunDirectory,
    exit_code: int,
    status: str,
    extra: dict | None = None,
) -> None:
    rd.finalize(exit_code, status, extra)


def _finalize_empty_failure(
    rd: RunDirectory,
    unit: RunUnit,
    run_id: str,
    *,
    status: str,
    exit_code: int,
    error: LightconeError,
) -> ExecutionOutcome:
    rd.append_log(
        "stderr",
        f"{status}: {type(error).__name__}: {error.message}\n",
    )
    for name in ("rounds", "updates", "decisions", "system_samples", "request_summary"):
        rd.write_table(name, [])
    _finalize_run(
        rd,
        exit_code,
        status,
        {
            "reason": error.message,
            "error_type": type(error).__name__,
            "source_exit_code": error.exit_code,
        },
    )
    return ExecutionOutcome(unit.unit_id, run_id, status, error.message)


def _unit_execution_sha256(
    unit: RunUnit,
    engine_params: dict,
    lockfile_sha256: str | None,
) -> str:
    """Hash every input that can affect one unit's execution.

    Static inference has no update tier and never loads a controller, so the
    CLI-only mode marker and controller directory are its only safe exclusions.
    All other engine, lock, model and runtime inputs remain bound.  Adapted
    units bind the complete engine parameter map.
    """
    effective_engine_params = dict(engine_params)
    if unit.method == "static":
        effective_engine_params.pop("weight_update_mode_override", None)
        effective_engine_params.pop("controller_root", None)
    return sha256_json(
        {
            "schema_version": 1,
            "unit": unit.identity_dict(),
            "engine_params": effective_engine_params,
            "lockfile_sha256": lockfile_sha256,
        }
    )


def _is_cpu_unit(unit: RunUnit) -> bool:
    return unit.model_pair.startswith("toy_") or (
        unit.dataset in SYNTHETIC_DATASET_KEYS and unit.model_pair.startswith("toy_")
    )


def execute_unit(
    unit: RunUnit,
    engine_params: dict,
    artifact_root: Path,
    lockfile_sha256: str | None,
    experiment_manifest_sha256: str | None = None,
    unit_execution_sha256: str | None = None,
) -> ExecutionOutcome:
    run_id = f"{unit.phase}-{unit.method}-{uuid.uuid4().hex[:10]}"
    rd = RunDirectory(artifact_root, run_id)
    runtime_engine_params = dict(engine_params)
    if not _is_cpu_unit(unit):
        from lightcone_spec.orchestration.runtime_config import materialize_gpu_runtime

        runtime_engine_params = materialize_gpu_runtime(
            unit, runtime_engine_params, rd.path
        )
    manifest_dict = unit.to_manifest_dict()
    manifest_dict["run_id"] = run_id
    manifest_dict["engine_params"] = runtime_engine_params
    manifest_dict["experiment_manifest_sha256"] = experiment_manifest_sha256
    manifest_dict["unit_execution_sha256"] = unit_execution_sha256
    environment = None
    if not _is_cpu_unit(unit):
        from lightcone_spec.doctor import collect_doctor_report

        environment = collect_doctor_report(
            runtime_engine_params.get("runtime_root", "~/lightcone-tts-runtime"),
            min_free_gib=0,
            check_network=False,
        )
        environment["runtime_config_sha256"] = runtime_engine_params.get(
            "runtime_config_sha256"
        )
        environment["locked_target_revision"] = runtime_engine_params.get(
            "locked_target_revision"
        )
        environment["locked_drafter_revision"] = runtime_engine_params.get(
            "locked_drafter_revision"
        )
    rd.create(
        manifest_dict,
        lock_reference={"lockfile_sha256": lockfile_sha256},
        environment=environment,
    )
    try:
        if _is_cpu_unit(unit):
            rows = _run_cpu_unit(unit, runtime_engine_params, run_id)
        else:
            _require_gpu_stack()
            rows = _run_gpu_unit(unit, runtime_engine_params, run_id)
    except ResourceSkip as exc:
        return _finalize_empty_failure(
            rd, unit, run_id,
            status="resource_skip", exit_code=exc.exit_code, error=exc,
        )
    except ExactnessViolation as exc:
        return _finalize_empty_failure(
            rd, unit, run_id,
            status="failed_exactness", exit_code=exc.exit_code, error=exc,
        )
    except RuntimeGpuFailure as exc:
        return _finalize_empty_failure(
            rd, unit, run_id,
            status="failed_runtime", exit_code=exc.exit_code, error=exc,
        )
    except (NumericalFailure, LightconeError) as exc:
        # The artifact schema has no separate numerical/config terminal state.
        # Once a run directory exists, retain the original type/code as evidence
        # while publishing the schema-compatible failed_runtime envelope.
        return _finalize_empty_failure(
            rd, unit, run_id,
            status="failed_runtime",
            exit_code=RuntimeGpuFailure.exit_code,
            error=exc,
        )

    status = rows.pop("_status")
    for name, table_rows in rows.items():
        rd.write_table(name, table_rows)
    exit_code = {
        "complete_valid": 0,
        "failed_exactness": 5,
        "failed_runtime": 7,
    }.get(status, 7)
    _finalize_run(rd, exit_code, status)
    return ExecutionOutcome(unit.unit_id, run_id, status)


def _require_gpu_stack() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeGpuFailure(
            "real model-pair units require CUDA GPUs and the pinned SGLang "
            "fork; refusing to fake results on this host"
        )
    try:
        import sglang  # noqa: F401
    except ImportError as exc:
        raise RuntimeGpuFailure(
            "pinned SGLang fork not importable; install the fork from "
            "patches/sglang before running GPU units"
        ) from exc


def _run_gpu_unit(unit: RunUnit, engine_params: dict, run_id: str) -> dict:
    from lightcone_spec.sglang_bridge.client import run_unit_via_sglang

    return run_unit_via_sglang(unit, engine_params, run_id)


def _run_cpu_unit(unit: RunUnit, engine_params: dict, run_id: str) -> dict:
    import torch

    from lightcone_spec.benchmarks.synthetic import (
        make_twin_profiles,
        synthetic_world,
    )
    from lightcone_spec.methods.base import CandidateGeneratorConfig
    from lightcone_spec.methods.registry import build_method
    from lightcone_spec.config.loader import validate_adaptation_config_dict
    from lightcone_spec.runtime.engine import EngineConfig, ReferenceEngine
    from lightcone_spec.runtime.toy_model import make_toy_pair

    torch.manual_seed(unit.seed)
    world = synthetic_world(unit.dataset, seed=unit.seed)
    pair = make_toy_pair(world, seed=unit.seed)
    profile = make_twin_profiles().get(unit.dataset)

    controller_methods = (
        "lc_gate",
        "lc_damp",
        "lc_transport",
        "round_discard",
        "wall_damp",
        "endpoint_gate",
        "parameter_only",
        "random_transport",
    )
    artifact_path = (
        engine_params.get("controller_artifact_path")
        if unit.method in controller_methods
        else None
    )
    method_cfg = {
        "schema_version": 1,
        "method": unit.method,
        "lifecycle": unit.lifecycle,
        "trainable_scope": unit.trainable_scope,
        "weight_update_mode": unit.weight_update_mode,
        "parameter_scope": unit.parameter_scope,
        "parameter_allowlist": list(unit.parameter_allowlist),
        "update_stride": unit.stride,
        "optimizer": _optimizer_for(unit.method),
        "lr": engine_params.get("lr", 1e-3),
        "lambda_prox": engine_params.get("lambda_prox", 0.1),
        "async": {
            "enabled": unit.method not in ("static", "sync_fresh"),
            "logical_delay_rounds": unit.logical_delay,
            "max_in_flight": 1,
        },
        "controller": {"artifact_path": artifact_path},
        "transport": {
            "rank": unit.adapter_rank if unit.adapter_rank in (8, 16, 32) else 16,
            "basis_path": artifact_path if unit.method == "lc_transport" else None,
        },
        "trace": {"artifact_root": engine_params.get("artifact_root", "/tmp")},
        "model": {"pair_id": unit.model_pair},
        "dataset": {"adapter": unit.dataset},
        "sampling": {
            "temperature": 0.0 if unit.sampling_profile == "greedy_t0" else 1.0,
            "top_p": 1.0,
            "max_new_tokens": engine_params.get("max_new_tokens", 48),
        },
        "runtime": {"seed": unit.seed, "concurrency": unit.concurrency},
    }
    config = validate_adaptation_config_dict(method_cfg)
    method = build_method(
        config, pair.shapes, pair.basis,
        transport_variant=unit.transport_variant or "joint",
    )
    eng_cfg = EngineConfig(
        method_key=unit.method,
        seed=unit.seed,
        update_stride=unit.stride,
        logical_delay_rounds=unit.logical_delay,
        max_rounds=engine_params.get("max_rounds", 24),
        max_new_tokens=engine_params.get("max_new_tokens", 48),
        draft_depth=engine_params.get("draft_depth", 3),
        temperature=config.sampling.temperature,
        lifecycle=unit.lifecycle,
        idle_dilation=(profile.dilation if profile and profile.kind == "idle_insertion" else 0),
        extra_wall_us=(profile.extra_wall_us if profile and profile.kind == "wall_only" else 0.0),
    )
    engine = ReferenceEngine(
        pair,
        method,
        eng_cfg,
        stream_id=(f"stream-{unit.seed}" if unit.lifecycle == "stream" else None),
    )
    meta = {
        "run_id": run_id,
        "unit_id": unit.unit_id,
        "model_pair_id": unit.model_pair,
        "dataset": unit.dataset,
        "task_type": "synthetic",
        "offered_concurrency": unit.concurrency,
    }
    n_requests = engine_params.get("num_requests", 4)
    all_rows = {
        "rounds": [],
        "updates": [],
        "decisions": [],
        "system_samples": [],
        "request_summary": [],
    }
    status = "complete_valid"
    for i in range(n_requests):
        res = engine.run_request(f"{run_id}-req{i}", run_meta=meta)
        all_rows["rounds"].extend(res.rounds_rows)
        all_rows["updates"].extend(res.updates_rows)
        all_rows["decisions"].extend(res.decisions_rows)
        all_rows["request_summary"].append(res.summary_row)
        all_rows["system_samples"].append(
            {
                **res.summary_row
                | {
                    k: res.summary_row[k]
                    for k in (
                        "schema_version",
                        "run_id",
                        "unit_id",
                        "request_id",
                        "stream_id",
                        "tenant_id_hash",
                        "model_pair_id",
                        "method",
                        "dataset",
                        "seed",
                        "lifecycle",
                    )
                },
                "timestamp_us": time.time() * 1e6,
                "gpu_index": -1,
                "hbm_used_bytes": 0,
                "sm_occupancy": None,
                "gpu_utilization": 0.0,
                "power_watts": 0.0,
                "energy_joules_delta": 0.0,
                "main_stream_active": True,
                "side_stream_active": False,
                "stream_contention_class": unit.contention_condition,
                "sync_us_delta": 0.0,
                "sample_source": "simulation",
                "activity_provenance": "simulated",
                "contention_provenance": "simulated",
                "sync_provenance": "simulated",
            }
        )
        if res.status != "complete_valid":
            status = res.status
            break
    all_rows["_status"] = status
    return all_rows


def _optimizer_for(method: str) -> str:
    if method == "static":
        return "none"
    if method in ("onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"):
        return "sgd"
    return "adamw"


def execute_manifest(
    manifest: ExperimentManifest,
    artifact_root: str | Path,
    resume: bool = True,
) -> ExecutionReport:
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    report = ExecutionReport()
    experiment_manifest_sha256 = manifest.content_sha256()
    runtime_failure_limit = _failure_circuit_breaker_limit(
        manifest.engine_params,
        "max_consecutive_runtime_failures",
        default=2,
    )
    exactness_failure_limit = _failure_circuit_breaker_limit(
        manifest.engine_params,
        "max_consecutive_exactness_failures",
        default=1,
    )
    # Failures are consecutive within one execution domain.  P5 deliberately
    # interleaves Static and adapted units, so a healthy (or resumed) Static
    # baseline must not erase repeated failures from the adapted backend.
    failure_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for index, unit in enumerate(manifest.units):
        failure_domain = (
            unit.model_pair,
            unit.method,
            unit.weight_update_mode,
        )
        execution_sha256 = _unit_execution_sha256(
            unit,
            manifest.engine_params,
            manifest.lockfile_sha256,
        )
        if resume:
            existing = _existing_complete_unit(
                artifact_root,
                unit.unit_id,
                execution_sha256,
            )
            if existing is not None:
                report.outcomes.append(
                    ExecutionOutcome(
                        unit.unit_id, existing, "skipped_existing",
                        "completed run with matching unit execution hash",
                    )
                )
                failure_counts.pop(failure_domain, None)
                continue
        outcome = execute_unit(
            unit,
            manifest.engine_params,
            artifact_root,
            manifest.lockfile_sha256,
            experiment_manifest_sha256,
            execution_sha256,
        )
        report.outcomes.append(outcome)
        counts = failure_counts.setdefault(
            failure_domain, {"runtime": 0, "exactness": 0}
        )
        counts["runtime"] = (
            counts["runtime"] + 1
            if outcome.status == "failed_runtime"
            else 0
        )
        counts["exactness"] = (
            counts["exactness"] + 1
            if outcome.status == "failed_exactness"
            else 0
        )
        breaker = None
        if (
            runtime_failure_limit
            and counts["runtime"] >= runtime_failure_limit
        ):
            breaker = (
                "runtime",
                runtime_failure_limit,
                outcome.detail,
            )
        elif (
            exactness_failure_limit
            and counts["exactness"] >= exactness_failure_limit
        ):
            breaker = (
                "exactness",
                exactness_failure_limit,
                outcome.detail,
            )
        if breaker is not None:
            kind, limit, detail = breaker
            message = (
                f"{kind} circuit breaker opened after {limit} consecutive "
                f"failure(s); last failure: {detail}"
            )
            report.outcomes.extend(
                ExecutionOutcome(
                    pending.unit_id,
                    None,
                    "not_run_circuit_breaker",
                    message,
                )
                for pending in manifest.units[index + 1 :]
            )
            break
    return report


def _failure_circuit_breaker_limit(
    engine_params: dict,
    key: str,
    *,
    default: int,
) -> int:
    value = engine_params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"engine_params.{key} must be a non-negative integer")
    return value
