#!/usr/bin/env python3
"""Keep an AutoDL token only in process memory and power off via documented API."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import math
import os
import resource
import signal
import sys
import time
import urllib.request
from pathlib import Path


def _attempt_payload(run_id: str, attempt_generation: int) -> str:
    return f"{run_id}\t{attempt_generation}"


def marker_matches(path: Path, run_id: str, attempt_generation: int) -> bool:
    """Return true only for a marker owned by this exact queue attempt."""
    try:
        return path.read_text(encoding="utf-8").strip() == _attempt_payload(
            run_id, attempt_generation
        )
    except (FileNotFoundError, OSError, UnicodeError):
        return False


def current_run_matches(queue_root: Path, run_id: str) -> bool:
    """A superseded watchdog must never shut down a newer experiment."""
    try:
        return (
            queue_root.joinpath("CURRENT_RUN")
            .read_text(encoding="utf-8")
            .strip()
            == run_id
        )
    except (FileNotFoundError, OSError, UnicodeError):
        return False


def current_attempt_matches(
    queue_root: Path, run_id: str, attempt_generation: int
) -> bool:
    try:
        return (
            queue_root.joinpath("CURRENT_ATTEMPT")
            .read_text(encoding="utf-8")
            .strip()
            == _attempt_payload(run_id, attempt_generation)
        )
    except (FileNotFoundError, OSError, UnicodeError):
        return False


def attempt_binding_state(
    queue_root: Path, run_id: str, attempt_generation: int
) -> str:
    """Classify launcher publication without mistaking startup for supersession."""

    current_run = queue_root / "CURRENT_RUN"
    try:
        observed_run = current_run.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "pending"
    except (OSError, UnicodeError):
        return "invalid"
    if observed_run != run_id:
        return "superseded"
    return (
        "match"
        if current_attempt_matches(queue_root, run_id, attempt_generation)
        else "pending"
    )


def wait_for_attempt_binding(
    queue_root: Path,
    run_id: str,
    attempt_generation: int,
    *,
    timeout_seconds: float,
    poll_seconds: float = 1.0,
) -> str:
    """Wait a bounded interval for CURRENT_RUN/CURRENT_ATTEMPT publication."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        state = attempt_binding_state(queue_root, run_id, attempt_generation)
        if state in {"match", "superseded", "invalid"}:
            return state
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0.0)))


def _attempt_root(
    queue_root: Path, run_id: str, attempt_generation: int
) -> Path:
    return queue_root / "sessions" / run_id / "attempts" / str(attempt_generation)


def _bound_fields(path: Path, expected_fields: int) -> list[str] | None:
    try:
        fields = path.read_text(encoding="utf-8").strip().split("\t")
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    return fields if len(fields) == expected_fields else None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def launcher_health(
    queue_root: Path,
    run_id: str,
    attempt_generation: int,
    *,
    heartbeat_timeout_seconds: float,
    now_epoch: float | None = None,
) -> tuple[bool, str]:
    """Validate the queue PID and a run/attempt-bound heartbeat record."""

    root = _attempt_root(queue_root, run_id, attempt_generation)
    pid_fields = _bound_fields(root / "queue.pid", 3)
    expected_prefix = [run_id, str(attempt_generation)]
    if pid_fields is None or pid_fields[:2] != expected_prefix:
        return False, "queue_pid_missing_or_unbound"
    try:
        queue_pid = int(pid_fields[2])
    except ValueError:
        return False, "queue_pid_invalid"
    if queue_pid <= 1:
        return False, "queue_pid_invalid"
    if not _pid_is_alive(queue_pid):
        return False, "queue_pid_not_alive"

    heartbeat_fields = _bound_fields(root / "heartbeat", 4)
    if heartbeat_fields is None or heartbeat_fields[:3] != pid_fields:
        return False, "heartbeat_missing_or_unbound"
    try:
        heartbeat_epoch = float(heartbeat_fields[3])
    except ValueError:
        return False, "heartbeat_timestamp_invalid"
    now = time.time() if now_epoch is None else now_epoch
    age = max(now - heartbeat_epoch, 0.0)
    if not math.isfinite(heartbeat_epoch) or age > heartbeat_timeout_seconds:
        return False, "heartbeat_stale"
    return True, "healthy"


def marker_seen_since(previous: float | None, present: bool, now: float) -> float | None:
    """Track a continuous marker interval; archival on resume resets grace."""

    return (previous if previous is not None else now) if present else None


def power_off(token: str, instance_uuid: str, log) -> None:
    request = urllib.request.Request(
        "https://api.autodl.com/api/v1/dev/instance/pro/power_off",
        data=json.dumps({"instance_uuid": instance_uuid}).encode("utf-8"),
        headers={"Authorization": token, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("AutoDL power_off returned a non-object response")
    safe = {key: payload.get(key) for key in ("code", "msg", "request_id")}
    print(json.dumps({"time": time.time(), "power_off_response": safe}), file=log, flush=True)
    if payload.get("code") != "Success":
        raise RuntimeError("AutoDL power_off did not return code=Success")


def power_off_if_attempt_current(
    token: str,
    instance_uuid: str,
    log,
    *,
    queue_root: Path,
    run_id: str,
    attempt_generation: int,
) -> bool:
    """Serialize final binding validation with compatible attempt publishers.

    A plain check followed by an API call has a race: a new attempt can publish
    between the two.  ``autodl_attempt_binder.py`` uses this same lock and must
    finish publication before the replacement workload is launched.
    """

    lock_path = queue_root / ".attempt-binder.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if attempt_binding_state(queue_root, run_id, attempt_generation) != "match":
            return False
        power_off(token, instance_uuid, log)
        return True


def _validate_timings(args: argparse.Namespace) -> None:
    values = {
        "--hard-deadline-hours": (args.hard_deadline_hours, False),
        "--success-grace-minutes": (args.success_grace_minutes, True),
        "--failure-grace-hours": (args.failure_grace_hours, True),
        "--arm-wait-seconds": (args.arm_wait_seconds, True),
        "--heartbeat-timeout-minutes": (
            args.heartbeat_timeout_minutes,
            False,
        ),
    }
    for label, (value, zero_allowed) in values.items():
        if not math.isfinite(value) or value < 0 or (not zero_allowed and value == 0):
            qualifier = "non-negative" if zero_allowed else "positive"
            raise SystemExit(f"{label} must be finite and {qualifier}")


def _acquire_singleton_lock(
    queue_root: Path, run_id: str, attempt_generation: int
):
    """Hold one watchdog per attempt without putting secrets in argv/env."""

    queue_root.mkdir(parents=True, exist_ok=True)
    lock = queue_root.joinpath(
        f"poweroff-watchdog.{run_id}.{attempt_generation}.lock"
    ).open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise SystemExit("another power-off watchdog is already active") from exc
    return lock


def _atomic_pid_file(path: Path, pid: int) -> None:
    temporary = path.with_name(f".{path.name}.{pid}.tmp")
    temporary.write_text(f"{pid}\n", encoding="utf-8")
    os.replace(temporary, path)


def _remove_pid_file_if_owned(path: Path, pid: int) -> None:
    try:
        if path.read_text(encoding="utf-8").strip() == str(pid):
            path.unlink()
    except (FileNotFoundError, OSError, UnicodeError):
        pass


def _detach_standard_streams() -> None:
    """Do not keep an SSH/launcher pipe open for the watchdog lifetime."""

    descriptor = os.open(os.devnull, os.O_RDWR)
    try:
        for target in (0, 1, 2):
            os.dup2(descriptor, target)
    finally:
        if descriptor > 2:
            os.close(descriptor)


def _record_power_off_failure(log, error: Exception, *, reason: str) -> None:
    # Exception messages can contain arbitrary server text.  Keep the log
    # useful without ever risking credential-bearing request material.
    payload = {
        "time": time.time(),
        "power_off_failed": {
            "reason": reason,
            "error_type": type(error).__name__,
        },
    }
    status = getattr(error, "code", None)
    if isinstance(status, int):
        payload["power_off_failed"]["http_status"] = status
    print(json.dumps(payload), file=log, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-generation", required=True, type=int)
    parser.add_argument("--instance-uuid", required=True)
    parser.add_argument("--hard-deadline-hours", type=float, required=True)
    parser.add_argument("--success-grace-minutes", type=float, default=10.0)
    parser.add_argument("--failure-grace-hours", type=float, default=3.0)
    parser.add_argument("--arm-wait-seconds", type=float, default=120.0)
    parser.add_argument("--heartbeat-timeout-minutes", type=float, default=10.0)
    args = parser.parse_args()
    if not args.run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_id
    ):
        raise SystemExit("invalid --run-id")
    if args.attempt_generation <= 0:
        raise SystemExit("--attempt-generation must be positive")
    _validate_timings(args)
    # Acquire before reading the token.  A duplicate invocation fails without
    # consuming or retaining another credential.
    _watchdog_lock = _acquire_singleton_lock(
        args.queue_root, args.run_id, args.attempt_generation
    )
    token = sys.stdin.readline().strip()
    if not token:
        raise SystemExit("missing token on stdin")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    pid = os.fork()
    if pid:
        print(json.dumps({"watchdog_pid": pid, "token_persisted": False}))
        return 0
    os.setsid()
    _detach_standard_streams()
    log_path = args.queue_root / "poweroff-watchdog.log"
    child_pid = os.getpid()

    def terminate(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with log_path.open("a", encoding="utf-8") as log:
        arm_state = wait_for_attempt_binding(
            args.queue_root,
            args.run_id,
            args.attempt_generation,
            timeout_seconds=args.arm_wait_seconds,
        )
        if arm_state != "match":
            print(
                json.dumps(
                    {
                        "time": time.time(),
                        "run_id": args.run_id,
                        "attempt_generation": args.attempt_generation,
                        "disarmed": f"attempt_arm_{arm_state}",
                    }
                ),
                file=log,
                flush=True,
            )
            return 0

        attempt_root = _attempt_root(
            args.queue_root, args.run_id, args.attempt_generation
        )
        if not attempt_root.is_dir():
            print(
                json.dumps(
                    {
                        "time": time.time(),
                        "run_id": args.run_id,
                        "attempt_generation": args.attempt_generation,
                        "disarmed": "attempt_runtime_root_missing",
                    }
                ),
                file=log,
                flush=True,
            )
            return 0
        pid_path = attempt_root / "poweroff-watchdog.pid"
        _atomic_pid_file(pid_path, child_pid)
        atexit.register(_remove_pid_file_if_owned, pid_path, child_pid)

        deadline_epoch = time.time() + args.hard_deadline_hours * 3600
        deadline_monotonic = time.monotonic() + args.hard_deadline_hours * 3600
        success_seen = None
        failure_seen = None
        terminal_conflict_logged = False
        previous_health_reason = None
        print(
            json.dumps(
                {
                    "time": time.time(),
                    "pid": os.getpid(),
                    "run_id": args.run_id,
                    "attempt_generation": args.attempt_generation,
                    "hard_deadline_epoch": deadline_epoch,
                    "hard_deadline_hours": args.hard_deadline_hours,
                    "heartbeat_timeout_minutes": args.heartbeat_timeout_minutes,
                }
            ),
            file=log,
            flush=True,
        )
        while True:
            now_epoch = time.time()
            now = time.monotonic()
            if attempt_binding_state(
                args.queue_root, args.run_id, args.attempt_generation
            ) != "match":
                print(
                    json.dumps(
                        {
                            "time": now_epoch,
                            "run_id": args.run_id,
                            "attempt_generation": args.attempt_generation,
                            "disarmed": "queue_attempt_superseded_or_missing",
                        }
                    ),
                    file=log,
                    flush=True,
                )
                return 0
            complete = marker_matches(
                args.queue_root / "QUEUE_COMPLETE",
                args.run_id,
                args.attempt_generation,
            )
            failed = marker_matches(
                args.queue_root / "QUEUE_FAILED",
                args.run_id,
                args.attempt_generation,
            )
            healthy, health_reason = launcher_health(
                args.queue_root,
                args.run_id,
                args.attempt_generation,
                heartbeat_timeout_seconds=args.heartbeat_timeout_minutes * 60,
                now_epoch=now_epoch,
            )
            if health_reason != previous_health_reason:
                print(
                    json.dumps(
                        {
                            "time": now_epoch,
                            "run_id": args.run_id,
                            "attempt_generation": args.attempt_generation,
                            "launcher_health": health_reason,
                        }
                    ),
                    file=log,
                    flush=True,
                )
                previous_health_reason = health_reason
            success_seen = marker_seen_since(success_seen, complete, now)
            failure_condition = not complete and (failed or not healthy)
            failure_seen = marker_seen_since(
                failure_seen, failure_condition, now
            )

            reason = None
            if complete and failed:
                if not terminal_conflict_logged:
                    print(
                        json.dumps(
                            {
                                "time": now_epoch,
                                "run_id": args.run_id,
                                "attempt_generation": args.attempt_generation,
                                "terminal_marker_conflict": True,
                                "action": "wait_for_hard_deadline",
                            }
                        ),
                        file=log,
                        flush=True,
                    )
                    terminal_conflict_logged = True
            else:
                terminal_conflict_logged = False
                if complete and now - success_seen >= args.success_grace_minutes * 60:
                    reason = "queue_complete_grace_elapsed"
                elif failure_condition and (
                    now - failure_seen >= args.failure_grace_hours * 3600
                ):
                    reason = (
                        "queue_failed_grace_elapsed"
                        if failed
                        else f"queue_launcher_{health_reason}_grace_elapsed"
                    )
            if now >= deadline_monotonic:
                reason = "hard_deadline_elapsed"
            if reason is not None:
                try:
                    powered_off = power_off_if_attempt_current(
                        token,
                        args.instance_uuid,
                        log,
                        queue_root=args.queue_root,
                        run_id=args.run_id,
                        attempt_generation=args.attempt_generation,
                    )
                except Exception as exc:  # keep retrying transient API failures
                    _record_power_off_failure(log, exc, reason=reason)
                else:
                    if powered_off:
                        return 0
                    print(
                        json.dumps(
                            {
                                "time": time.time(),
                                "run_id": args.run_id,
                                "attempt_generation": args.attempt_generation,
                                "disarmed": "queue_attempt_superseded_before_power_off",
                            }
                        ),
                        file=log,
                        flush=True,
                    )
                    return 0
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
