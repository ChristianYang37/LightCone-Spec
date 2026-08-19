#!/usr/bin/env python3
"""Durable formal experiment operator and resident non-LLM scheduler CLI.

Administrative subcommands mutate only SQLite state. ``scheduler-run`` holds a
singleton flock, reconciles durable attempts every 30 seconds, and launches only
already-materialized argv through the production no-shell process boundary.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lightcone_spec.orchestration.experiment_operator import (
    ArchiveRequest,
    ArchiveStepReceipt,
    CellAttemptSpec,
    ExperimentOperatorError,
    ExperimentOperatorStore,
    FormalExperimentSchedulerDaemon,
    InterferenceEnvelope,
    LegacyStaleAttempt,
    MetricRecord,
    ProviderRuntimeSample,
    QueuedCommandSpec,
    SingletonOperatorLock,
    StagePlanEntry,
    WatchdogPolicy,
    default_formal_stage_plan,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    ProductionArchiveRuntime,
    ProductionSchedulerRuntime,
)

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MUTATING_COMMANDS = frozenset(
    {
        "init",
        "materialize",
        "claim-running",
        "attach-process",
        "heartbeat",
        "finish",
        "mark-stale",
        "watchdog-once",
        "record-selection",
        "record-metric",
        "update-stage",
        "import-stale",
        "disk-gate",
        "configure-interference",
        "enqueue-command",
        "scheduler-resume",
        "scheduler-stop",
        "record-provider-sample",
        "archive-run",
        "archive-register",
        "archive-record-step",
        "archive-authorize",
        "controller-block",
        "controller-resume",
    }
)


def _json_file(path: str | Path) -> Any:
    if str(path) == "-":
        body = sys.stdin.buffer.read(_MAX_JSON_BYTES + 1)
        if not body or len(body) > _MAX_JSON_BYTES:
            raise ValueError("stdin JSON is empty or too large")
        return json.loads(body)
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"JSON input cannot be a symlink: {source}")
    if not source.is_file() or source.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"JSON input is missing or too large: {source}")
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _object_file(path: str | Path) -> dict[str, Any]:
    value = _json_file(path)
    if not isinstance(value, dict):
        raise TypeError("JSON input must be an object")
    return value


def _optional_object_file(path: str | None) -> dict[str, Any]:
    return {} if path is None else _object_file(path)


def _stage_entries(path: str | None) -> tuple[StagePlanEntry, ...]:
    if path is None:
        return default_formal_stage_plan()
    value = _json_file(path)
    if not isinstance(value, list):
        raise TypeError("stage plan JSON must be a list")
    return tuple(StagePlanEntry(**row) for row in value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="authoritative SQLite path")
    parser.add_argument(
        "--lock",
        help="singleton lock path (defaults to <db>.operator.lock)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init")
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--stage-plan")

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--spec", required=True)

    interference = commands.add_parser("configure-interference")
    interference.add_argument("--envelope", required=True)

    enqueue = commands.add_parser("enqueue-command")
    enqueue.add_argument("--command-spec", required=True)
    enqueue.add_argument("--enqueued-at-ns", type=int)

    scheduler_run = commands.add_parser("scheduler-run")
    scheduler_run.add_argument(
        "--scheduler-lock",
        help="resident scheduler flock path (defaults to <db>.scheduler.lock)",
    )
    scheduler_run.add_argument("--max-cycles", type=int)
    scheduler_run.add_argument("--nvidia-smi", default="nvidia-smi")
    scheduler_run.add_argument("--python-executable", default=sys.executable)
    scheduler_run.add_argument("--process-attach-grace-seconds", type=float, default=30)
    scheduler_run.add_argument("--heartbeat-timeout-seconds", type=float, default=120)
    scheduler_run.add_argument("--log-stall-timeout-seconds", type=float, default=300)
    scheduler_run.add_argument("--event-repeat-seconds", type=float, default=300)
    scheduler_run.add_argument(
        "--minimum-free-disk-bytes", type=int, default=15 * 1024**3
    )

    scheduler_stop = commands.add_parser("scheduler-stop")
    scheduler_stop.add_argument("--reason", required=True)
    scheduler_stop.add_argument("--stopped-at-ns", type=int)

    scheduler_resume = commands.add_parser("scheduler-resume")
    scheduler_resume.add_argument("--reason", required=True)
    scheduler_resume.add_argument("--cleared-at-ns", type=int)

    stale_import = commands.add_parser("import-stale")
    stale_import.add_argument("--manifest", required=True)

    claim = commands.add_parser("claim-running")
    claim.add_argument("--cell-id", required=True)
    claim.add_argument("--attempt", required=True, type=int)
    claim.add_argument("--gpu-uuid", action="append", default=[])
    claim.add_argument("--started-at-ns", type=int)

    attach = commands.add_parser("attach-process")
    attach.add_argument("--cell-id", required=True)
    attach.add_argument("--attempt", required=True, type=int)
    attach.add_argument("--pid", required=True, type=int)
    attach.add_argument("--pgid", required=True, type=int)

    heartbeat = commands.add_parser("heartbeat")
    heartbeat.add_argument("--cell-id", required=True)
    heartbeat.add_argument("--attempt", required=True, type=int)
    heartbeat.add_argument("--pid", required=True, type=int)
    heartbeat.add_argument("--pgid", required=True, type=int)
    heartbeat.add_argument("--log-size-bytes", required=True, type=int)
    heartbeat.add_argument("--gpu-observation", required=True)
    heartbeat.add_argument("--observed-at-ns", type=int)

    finish = commands.add_parser("finish")
    finish.add_argument("--cell-id", required=True)
    finish.add_argument("--attempt", required=True, type=int)
    finish.add_argument(
        "--status",
        required=True,
        choices=("COMPLETE", "FAILED", "BLOCKED", "STALE_IDENTITY"),
    )
    finish.add_argument("--exit-code", type=int)
    finish.add_argument("--terminal-sha256")
    finish.add_argument("--junit-sha256")
    finish.add_argument("--raw-log-sha256")
    finish.add_argument("--evidence-files")
    finish.add_argument("--failure-code")
    finish.add_argument("--retry-decision")
    finish.add_argument("--included-in-analysis", action="store_true")
    finish.add_argument("--exclusion-reason")
    finish.add_argument("--compute-gpu-seconds", type=float, default=0.0)
    finish.add_argument("--reserved-gpu-seconds", type=float, default=0.0)
    finish.add_argument("--billed-gpu-seconds", type=float, default=0.0)
    finish.add_argument("--finished-at-ns", type=int)

    stale = commands.add_parser("mark-stale")
    stale.add_argument("--cell-id", required=True)
    stale.add_argument("--attempt", required=True, type=int)
    stale.add_argument("--reason", required=True)
    stale.add_argument("--retry-decision", default="RERUN_UNDER_FROZEN_IDENTITY")
    stale.add_argument("--marked-at-ns", type=int)

    watchdog = commands.add_parser("watchdog-once")
    watchdog.add_argument("--monitored-path")
    watchdog.add_argument("--now-ns", type=int)
    watchdog.add_argument("--process-attach-grace-seconds", type=float, default=30)
    watchdog.add_argument("--heartbeat-timeout-seconds", type=float, default=120)
    watchdog.add_argument("--log-stall-timeout-seconds", type=float, default=300)
    watchdog.add_argument("--event-repeat-seconds", type=float, default=300)
    watchdog.add_argument("--minimum-free-disk-bytes", type=int, default=15 * 1024**3)

    selection = commands.add_parser("record-selection")
    selection.add_argument("--decision-id", required=True)
    selection.add_argument("--stage", required=True)
    selection.add_argument("--phase", required=True)
    selection.add_argument("--decision-kind", required=True)
    selection.add_argument("--source-sha256", required=True)
    selection.add_argument("--decision", required=True)
    selection.add_argument("--occurred-at-ns", type=int)

    metric = commands.add_parser("record-metric")
    metric.add_argument("--metric", required=True)
    metric.add_argument("--recorded-at-ns", type=int)

    provider = commands.add_parser("record-provider-sample")
    provider.add_argument(
        "--sample",
        required=True,
        help="credential-free provider runtime sample JSON",
    )

    archive = commands.add_parser("archive-run")
    archive.add_argument("--request", required=True)
    archive.add_argument("--rsync-executable", default="rsync")
    archive.add_argument("--minimum-local-free-bytes", type=int, default=100 * 1024**3)

    archive_register = commands.add_parser("archive-register")
    archive_register.add_argument("--request", required=True)

    archive_step = commands.add_parser("archive-record-step")
    archive_step.add_argument("--archive-id", required=True)
    archive_step.add_argument("--receipt", required=True)
    archive_step.add_argument("--recorded-at-ns", type=int)

    archive_authorize = commands.add_parser("archive-authorize")
    archive_authorize.add_argument("--archive-id", required=True)
    archive_authorize.add_argument("--authorized-at-ns", type=int)

    archive_status = commands.add_parser("archive-status")
    archive_status.add_argument("--archive-id", required=True)

    controller_block = commands.add_parser("controller-block")
    controller_block.add_argument("--node", required=True)
    controller_block.add_argument("--reason", required=True)
    controller_block.add_argument("--recorded-at-ns", type=int)

    controller_resume = commands.add_parser("controller-resume")
    controller_resume.add_argument("--node", required=True)
    controller_resume.add_argument("--reason", required=True)
    controller_resume.add_argument("--recorded-at-ns", type=int)

    update = commands.add_parser("update-stage")
    update.add_argument("--node", required=True)
    update.add_argument("--expected-formula", required=True)
    update.add_argument("--known-expected-cells", required=True, type=int)
    update.add_argument("--estimated-remaining-gpu-hours", type=float)

    disk_gate = commands.add_parser("disk-gate")
    disk_gate.add_argument("--monitored-path", required=True)
    disk_gate.add_argument(
        "--predicted-next-wave-high-water-bytes", required=True, type=int
    )
    disk_gate.add_argument("--safety-reserve-bytes", type=int, default=15 * 1024**3)
    disk_gate.add_argument("--observed-at-ns", type=int)

    export = commands.add_parser("export")
    export.add_argument("--output-root", required=True)
    export.add_argument("--exported-at-ns", type=int)

    commands.add_parser("status")
    return parser


def _lock_path(arguments: argparse.Namespace) -> Path:
    if arguments.lock:
        return Path(arguments.lock)
    database = Path(arguments.db)
    return database.with_name(f"{database.name}.operator.lock")


def _scheduler_lock_path(arguments: argparse.Namespace) -> Path:
    if arguments.scheduler_lock:
        return Path(arguments.scheduler_lock)
    database = Path(arguments.db)
    return database.with_name(f"{database.name}.scheduler.lock")


def _queued_command(path: str | Path) -> QueuedCommandSpec:
    value = _object_file(path)
    value["argv"] = tuple(value["argv"])
    environment = value.get("environment", {})
    if isinstance(environment, dict):
        value["environment"] = tuple(sorted(environment.items()))
    elif isinstance(environment, list):
        value["environment"] = tuple(tuple(row) for row in environment)
    else:
        raise TypeError("command environment must be an object or pair list")
    return QueuedCommandSpec(**value)


def _execute(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command == "init":
        with ExperimentOperatorStore(arguments.db, run_id=arguments.run_id) as store:
            store.initialize_stage_plan(_stage_entries(arguments.stage_plan))
            return store.snapshot()

    with ExperimentOperatorStore(arguments.db) as store:
        if arguments.command == "materialize":
            spec = CellAttemptSpec(**_object_file(arguments.spec))
            store.materialize_attempt(spec)
            return store.attempt(spec.cell_id, spec.attempt)
        if arguments.command == "configure-interference":
            value = _object_file(arguments.envelope)
            value["gpu_uuids"] = tuple(value["gpu_uuids"])
            store.configure_interference_envelope(InterferenceEnvelope(**value))
            return vars(store.interference_envelope())
        if arguments.command == "enqueue-command":
            command = _queued_command(arguments.command_spec)
            store.enqueue_command(
                command,
                enqueued_at_ns=arguments.enqueued_at_ns,
            )
            return asdict(command)
        if arguments.command == "scheduler-run":
            stop = False

            def request_stop(_signum: int, _frame: object) -> None:
                nonlocal stop
                stop = True

            prior_handlers = {
                signum: signal.signal(signum, request_stop)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                runtime = ProductionSchedulerRuntime(
                    nvidia_smi_executable=arguments.nvidia_smi,
                    python_executable=arguments.python_executable,
                )
                daemon = FormalExperimentSchedulerDaemon(
                    store,
                    lock_path=_scheduler_lock_path(arguments),
                    callbacks=runtime.callbacks(),
                    watchdog_policy=WatchdogPolicy(
                        process_attach_grace_seconds=(
                            arguments.process_attach_grace_seconds
                        ),
                        heartbeat_timeout_seconds=(arguments.heartbeat_timeout_seconds),
                        log_stall_timeout_seconds=(arguments.log_stall_timeout_seconds),
                        event_repeat_seconds=arguments.event_repeat_seconds,
                        minimum_free_disk_bytes=arguments.minimum_free_disk_bytes,
                    ),
                )
                cycles = daemon.run_forever(
                    stop_requested=lambda: stop,
                    max_cycles=arguments.max_cycles,
                )
            finally:
                for signum, prior in prior_handlers.items():
                    signal.signal(signum, prior)
            return {
                "cycles": [asdict(cycle) for cycle in cycles],
                "stop_requested": stop,
            }
        if arguments.command == "scheduler-stop":
            store.set_dispatch_stop(
                arguments.reason,
                stopped_at_ns=arguments.stopped_at_ns,
            )
            state, reason = store.dispatch_control()
            return {"dispatch_state": state, "dispatch_stop_reason": reason}
        if arguments.command == "scheduler-resume":
            store.clear_dispatch_stop(
                reason=arguments.reason,
                cleared_at_ns=arguments.cleared_at_ns,
            )
            state, reason = store.dispatch_control()
            return {"dispatch_state": state, "dispatch_stop_reason": reason}
        if arguments.command == "import-stale":
            manifest = _json_file(arguments.manifest)
            if not isinstance(manifest, list):
                raise TypeError("legacy stale manifest must be a list")
            imported = []
            for value in manifest:
                if not isinstance(value, dict) or not isinstance(
                    value.get("spec"), dict
                ):
                    raise TypeError("legacy stale rows require an object spec")
                row = dict(value)
                row["spec"] = CellAttemptSpec(**row["spec"])
                imported.append(LegacyStaleAttempt(**row))
            return {"imported": store.import_legacy_stale_attempts(imported)}
        if arguments.command == "claim-running":
            store.mark_running_before_spawn(
                arguments.cell_id,
                arguments.attempt,
                assigned_gpu_uuids=arguments.gpu_uuid,
                started_at_ns=arguments.started_at_ns,
            )
            return store.attempt(arguments.cell_id, arguments.attempt)
        if arguments.command == "attach-process":
            store.attach_process(
                arguments.cell_id,
                arguments.attempt,
                pid=arguments.pid,
                pgid=arguments.pgid,
            )
            return store.attempt(arguments.cell_id, arguments.attempt)
        if arguments.command == "heartbeat":
            store.record_heartbeat(
                arguments.cell_id,
                arguments.attempt,
                pid=arguments.pid,
                pgid=arguments.pgid,
                log_size_bytes=arguments.log_size_bytes,
                gpu_observation=_object_file(arguments.gpu_observation),
                observed_at_ns=arguments.observed_at_ns,
            )
            return store.attempt(arguments.cell_id, arguments.attempt)
        if arguments.command == "finish":
            store.finish_attempt(
                arguments.cell_id,
                arguments.attempt,
                status=arguments.status,
                exit_code=arguments.exit_code,
                terminal_sha256=arguments.terminal_sha256,
                junit_sha256=arguments.junit_sha256,
                raw_log_sha256=arguments.raw_log_sha256,
                evidence_files=_optional_object_file(arguments.evidence_files),
                failure_code=arguments.failure_code,
                retry_decision=arguments.retry_decision,
                included_in_analysis=arguments.included_in_analysis,
                exclusion_reason=arguments.exclusion_reason,
                compute_gpu_seconds=arguments.compute_gpu_seconds,
                reserved_gpu_seconds=arguments.reserved_gpu_seconds,
                billed_gpu_seconds=arguments.billed_gpu_seconds,
                finished_at_ns=arguments.finished_at_ns,
            )
            return store.attempt(arguments.cell_id, arguments.attempt)
        if arguments.command == "mark-stale":
            store.mark_stale_identity(
                arguments.cell_id,
                arguments.attempt,
                reason=arguments.reason,
                retry_decision=arguments.retry_decision,
                marked_at_ns=arguments.marked_at_ns,
            )
            return store.attempt(arguments.cell_id, arguments.attempt)
        if arguments.command == "watchdog-once":
            findings = store.watchdog_once(
                policy=WatchdogPolicy(
                    process_attach_grace_seconds=(
                        arguments.process_attach_grace_seconds
                    ),
                    heartbeat_timeout_seconds=arguments.heartbeat_timeout_seconds,
                    log_stall_timeout_seconds=arguments.log_stall_timeout_seconds,
                    event_repeat_seconds=arguments.event_repeat_seconds,
                    minimum_free_disk_bytes=arguments.minimum_free_disk_bytes,
                ),
                monitored_path=arguments.monitored_path,
                now_ns=arguments.now_ns,
            )
            return {"findings": [vars(row) for row in findings]}
        if arguments.command == "record-selection":
            store.record_selection_decision(
                decision_id=arguments.decision_id,
                stage=arguments.stage,
                phase=arguments.phase,
                decision_kind=arguments.decision_kind,
                source_sha256=arguments.source_sha256,
                decision=_object_file(arguments.decision),
                occurred_at_ns=arguments.occurred_at_ns,
            )
            return {"decision_id": arguments.decision_id}
        if arguments.command == "record-metric":
            store.record_metric(
                MetricRecord(**_object_file(arguments.metric)),
                recorded_at_ns=arguments.recorded_at_ns,
            )
            return {"recorded": True}
        if arguments.command == "record-provider-sample":
            sample = ProviderRuntimeSample(**_object_file(arguments.sample))
            store.record_provider_runtime_sample(sample)
            return {
                "sample_id": sample.sample_id,
                "whole_instance_billed_gpu_seconds": (
                    store.whole_instance_billed_gpu_seconds()
                ),
                "credential_stored": False,
            }
        if arguments.command == "archive-run":
            request = ArchiveRequest(**_object_file(arguments.request))
            authorization = store.run_archive_callbacks(
                request,
                ProductionArchiveRuntime(
                    rsync_executable=arguments.rsync_executable,
                    full_rehydrate=True,
                    minimum_local_free_bytes=arguments.minimum_local_free_bytes,
                ).callbacks(),
            )
            return {
                **asdict(authorization),
                "remote_deletion_performed": False,
            }
        if arguments.command == "archive-register":
            request = ArchiveRequest(**_object_file(arguments.request))
            store.register_archive_safe_boundary(request)
            return store.archive_checkpoint(request.archive_id)
        if arguments.command == "archive-record-step":
            receipt = ArchiveStepReceipt(**_object_file(arguments.receipt))
            store.record_archive_step(
                arguments.archive_id,
                receipt,
                recorded_at_ns=arguments.recorded_at_ns,
            )
            return store.archive_checkpoint(arguments.archive_id)
        if arguments.command == "archive-authorize":
            authorization = store.authorize_remote_eviction(
                arguments.archive_id,
                authorized_at_ns=arguments.authorized_at_ns,
            )
            return {
                **asdict(authorization),
                "remote_deletion_performed": False,
            }
        if arguments.command == "archive-status":
            return store.archive_checkpoint(arguments.archive_id)
        if arguments.command == "controller-block":
            store.mark_controller_blocked(
                node=arguments.node,
                reason=arguments.reason,
                recorded_at_ns=arguments.recorded_at_ns,
            )
            return store.controller_node(arguments.node)
        if arguments.command == "controller-resume":
            store.resume_controller_node(
                node=arguments.node,
                reason=arguments.reason,
                recorded_at_ns=arguments.recorded_at_ns,
            )
            return store.controller_node(arguments.node)
        if arguments.command == "update-stage":
            store.update_stage_expectation(
                node=arguments.node,
                expected_formula=arguments.expected_formula,
                known_expected_cells=arguments.known_expected_cells,
                estimated_remaining_gpu_hours=(arguments.estimated_remaining_gpu_hours),
            )
            return store.snapshot()
        if arguments.command == "disk-gate":
            return vars(
                store.check_dispatch_disk_capacity(
                    monitored_path=arguments.monitored_path,
                    predicted_next_wave_high_water_bytes=(
                        arguments.predicted_next_wave_high_water_bytes
                    ),
                    safety_reserve_bytes=arguments.safety_reserve_bytes,
                    observed_at_ns=arguments.observed_at_ns,
                )
            )
        if arguments.command == "export":
            return vars(
                store.export_progress(
                    arguments.output_root,
                    exported_at_ns=arguments.exported_at_ns,
                )
            )
        if arguments.command == "status":
            return store.snapshot()
    raise AssertionError(f"unhandled operator command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command in _MUTATING_COMMANDS:
            with SingletonOperatorLock(_lock_path(arguments)):
                output = _execute(arguments)
        else:
            output = _execute(arguments)
    except (ExperimentOperatorError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"formal experiment operator: {error}", file=sys.stderr)
        return 2
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
