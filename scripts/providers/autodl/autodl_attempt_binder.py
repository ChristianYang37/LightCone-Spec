#!/usr/bin/env python3
"""Bind a legacy experiment launcher to the power-off watchdog protocol.

This process never receives an AutoDL credential.  It only publishes the
attempt-bound PID, heartbeat, and terminal markers consumed by
``autodl_poweroff_watchdog.py``.  PID reuse is rejected with Linux boot-id and
``/proc/<pid>/stat`` start-time identity.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, NamedTuple


class ProcessIdentity(NamedTuple):
    pid: int
    boot_id: str
    start_ticks: int


def _attempt_payload(run_id: str, attempt_generation: int) -> str:
    return f"{run_id}\t{attempt_generation}"


def _attempt_root(
    queue_root: Path, run_id: str, attempt_generation: int
) -> Path:
    return queue_root / "sessions" / run_id / "attempts" / str(attempt_generation)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None


def capture_process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessIdentity | None:
    """Capture an alive, non-zombie Linux process identity."""

    if pid <= 1:
        return None
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        # comm may contain spaces and parentheses; the final ')' precedes state.
        remainder = stat.rsplit(")", 1)[1].strip().split()
        state = remainder[0]
        start_ticks = int(remainder[19])  # proc(5) field 22
        boot_id = (
            proc_root / "sys" / "kernel" / "random" / "boot_id"
        ).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError):
        return None
    if state == "Z" or not boot_id or start_ticks < 0:
        return None
    return ProcessIdentity(pid=pid, boot_id=boot_id, start_ticks=start_ticks)


def process_identity_matches(
    expected: ProcessIdentity,
    *,
    proc_root: Path = Path("/proc"),
) -> bool:
    return capture_process_identity(expected.pid, proc_root=proc_root) == expected


def process_start_epoch(
    identity: ProcessIdentity,
    *,
    proc_root: Path = Path("/proc"),
    now_epoch: float | None = None,
) -> float | None:
    """Convert proc start ticks to wall time for fresh-terminal binding."""

    try:
        uptime_seconds = float(
            (proc_root / "uptime").read_text(encoding="utf-8").split()[0]
        )
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError):
        return None
    if (
        not math.isfinite(uptime_seconds)
        or uptime_seconds < 0
        or ticks_per_second <= 0
    ):
        return None
    now = time.time() if now_epoch is None else now_epoch
    return now - uptime_seconds + identity.start_ticks / ticks_per_second


def _binding_matches(
    queue_root: Path, run_id: str, attempt_generation: int
) -> bool:
    return (
        _read_text(queue_root / "CURRENT_RUN") == run_id
        and _read_text(queue_root / "CURRENT_ATTEMPT")
        == _attempt_payload(run_id, attempt_generation)
    )


def publish_attempt_binding(
    queue_root: Path,
    run_id: str,
    attempt_generation: int,
    identity: ProcessIdentity,
    *,
    launcher_start_epoch: float | None = None,
    supersede_current: bool = False,
) -> Path:
    """CAS-publish one attempt and its immutable launcher identity.

    Replacing a different attempt requires an explicit flag.  Publishing a new
    attempt makes every older watchdog disarm before it can power off.
    """

    queue_root.mkdir(parents=True, exist_ok=True)
    lock_path = queue_root / ".attempt-binder.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current_run = _read_text(queue_root / "CURRENT_RUN")
        current_attempt = _read_text(queue_root / "CURRENT_ATTEMPT")
        payload = _attempt_payload(run_id, attempt_generation)
        exact = current_run == run_id and current_attempt == payload
        occupied = current_run is not None or current_attempt is not None
        if occupied and not exact and not supersede_current:
            raise RuntimeError(
                "another attempt is published; pass --supersede-current only "
                "when intentionally starting its replacement"
            )

        attempt_root = _attempt_root(queue_root, run_id, attempt_generation)
        attempt_root.mkdir(parents=True, exist_ok=True)
        existing_identity_path = attempt_root / "launcher.identity.json"
        preserve_existing_identity = False
        if exact and existing_identity_path.exists():
            try:
                existing_identity = json.loads(
                    existing_identity_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("existing launcher identity is invalid") from exc
            expected_identity = {
                "pid": identity.pid,
                "boot_id": identity.boot_id,
                "start_ticks": identity.start_ticks,
            }
            if any(
                existing_identity.get(key) != value
                for key, value in expected_identity.items()
            ):
                raise RuntimeError(
                    "the published attempt is already bound to another launcher"
                )
            preserve_existing_identity = True
        if launcher_start_epoch is not None and (
            not math.isfinite(launcher_start_epoch) or launcher_start_epoch <= 0
        ):
            raise ValueError("launcher_start_epoch must be finite and positive")
        identity_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_generation": attempt_generation,
            "pid": identity.pid,
            "boot_id": identity.boot_id,
            "start_ticks": identity.start_ticks,
            "launcher_start_epoch": launcher_start_epoch,
        }
        if not preserve_existing_identity:
            _atomic_write(
                existing_identity_path,
                json.dumps(identity_payload, sort_keys=True) + "\n",
            )
        _atomic_write(
            attempt_root / "queue.pid",
            f"{run_id}\t{attempt_generation}\t{identity.pid}\n",
        )
        # Publish a fail-closed transition value first.  This disarms the old
        # watchdog even when a new generation reuses the same run-id; the new
        # watchdog must be launched only after this function returns.
        transition = f".transition-{os.getpid()}-{time.monotonic_ns()}"
        _atomic_write(queue_root / "CURRENT_RUN", transition + "\n")
        _atomic_write(queue_root / "CURRENT_ATTEMPT", payload + "\n")
        _atomic_write(queue_root / "CURRENT_RUN", run_id + "\n")
        return attempt_root


def publish_heartbeat(
    queue_root: Path,
    run_id: str,
    attempt_generation: int,
    identity: ProcessIdentity,
    *,
    now_epoch: float | None = None,
) -> bool:
    """Publish only while this attempt is still the global current attempt."""

    lock_path = queue_root / ".attempt-binder.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not _binding_matches(queue_root, run_id, attempt_generation):
            return False
        epoch = time.time() if now_epoch is None else now_epoch
        _atomic_write(
            _attempt_root(queue_root, run_id, attempt_generation) / "heartbeat",
            f"{run_id}\t{attempt_generation}\t{identity.pid}\t{epoch:.6f}\n",
        )
        return True


def _verified_json(
    path: Path,
    *,
    not_before_epoch: float,
    sidecar_required: bool,
) -> dict[str, Any] | None:
    try:
        if path.stat().st_mtime + 1e-6 < not_before_epoch:
            return None
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        sidecar = Path(str(path) + ".sha256")
        if sidecar_required or sidecar.exists():
            expected = sidecar.read_text(encoding="utf-8").strip()
            if expected != hashlib.sha256(raw).hexdigest():
                return None
        return payload
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None


def _verified_receipt_tree(
    path: Path,
    *,
    not_before_epoch: float,
) -> dict[str, Any] | None:
    """Verify one fresh receipt and every receipt nested in its evidence.

    Only the terminal itself must be newer than the launcher.  Its immutable
    inputs normally predate the attempt, so nested evidence is hash-checked but
    is not subject to the freshness bound.  Receipt sidecars are mandatory;
    arbitrary evidence files are bound by the SHA-256 recorded by their owner.
    """

    active: set[Path] = set()
    verified: set[Path] = set()

    def visit(receipt: Path, *, require_fresh: bool) -> dict[str, Any] | None:
        try:
            receipt = receipt.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return None
        if receipt in verified:
            return _verified_json(
                receipt,
                not_before_epoch=0.0,
                sidecar_required=True,
            )
        if receipt in active:
            return None
        active.add(receipt)
        try:
            freshness = not_before_epoch if require_fresh else 0.0
            payload = _verified_json(
                receipt,
                not_before_epoch=freshness,
                sidecar_required=True,
            )
            if payload is None:
                return None
            sidecar = Path(str(receipt) + ".sha256")
            if require_fresh and sidecar.stat().st_mtime + 1e-6 < not_before_epoch:
                return None
            if not {
                "schema_version",
                "status",
                "scope",
                "evidence",
            }.issubset(payload):
                return None
            rows = payload.get("evidence")
            if not isinstance(rows, list) or not rows:
                return None
            for row in rows:
                if not isinstance(row, dict):
                    return None
                raw_path = row.get("path")
                expected_sha256 = row.get("sha256")
                if (
                    not isinstance(raw_path, str)
                    or not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                ):
                    return None
                evidence = Path(raw_path)
                if not evidence.is_absolute():
                    return None
                try:
                    raw = evidence.read_bytes()
                except (FileNotFoundError, OSError):
                    return None
                if hashlib.sha256(raw).hexdigest() != expected_sha256:
                    return None
                try:
                    nested = json.loads(raw)
                except (UnicodeError, json.JSONDecodeError):
                    nested = None
                if isinstance(nested, dict) and {
                    "schema_version",
                    "status",
                    "scope",
                    "evidence",
                }.issubset(nested):
                    if visit(evidence, require_fresh=False) is None:
                        return None
            verified.add(receipt)
            return payload
        except (FileNotFoundError, OSError):
            return None
        finally:
            active.remove(receipt)

    return visit(path, require_fresh=True)


def _fresh_headline_completion(
    controller_root: Path,
    headline_root: Path,
    *,
    not_before_epoch: float,
) -> bool:
    state = controller_root / "queue-state.jsonl"
    try:
        if state.stat().st_mtime + 1e-6 < not_before_epoch:
            return False
        rows = [line for line in state.read_text(encoding="utf-8").splitlines() if line]
        latest = json.loads(rows[-1])
    except (
        FileNotFoundError,
        IndexError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        return False
    if not isinstance(latest, dict) or not (
        latest.get("phase") == "headline" and latest.get("status") == "complete"
    ):
        return False
    # These are written only after each corresponding validate-artifacts call.
    return all(
        _verified_json(
            path,
            not_before_epoch=not_before_epoch,
            sidecar_required=False,
        )
        is not None
        for path in (
            headline_root / "algorithmic-coverage.json",
            headline_root / "mfu-coverage.json",
        )
    )


def proof_chain_terminal_state(
    *,
    screen_root: Path,
    confirmation_root: Path,
    controller_root: Path,
    headline_root: Path,
    not_before_epoch: float,
) -> tuple[str, str]:
    """Classify the v8-v11 proof chain after its launcher has exited.

    A scientific gate block is successful queue completion.  Anything without
    a fresh, hash-consistent terminal is an engineering failure, which keeps
    the instance available for the longer failure grace.
    """

    controller_terminals = (
        (controller_root / "CONTROLLER_SELECTED.json", "matched_controller_selected"),
        (controller_root / "CONTROLLER_BLOCKED.json", "matched_controller_blocked"),
    )
    for path, expected_status in controller_terminals:
        payload = _verified_json(
            path, not_before_epoch=not_before_epoch, sidecar_required=True
        )
        if (
            payload is not None
            and payload.get("status") == expected_status
            and _fresh_headline_completion(
                controller_root,
                headline_root,
                not_before_epoch=not_before_epoch,
            )
        ):
            return "complete", "controller_terminal_and_headline_complete"

    foundation_path = (
        confirmation_root
        / "tts-0-40k-foundation"
        / "TTS_0_40K_FOUNDATION.json"
    )
    foundation_fresh = False
    try:
        foundation_fresh = (
            foundation_path.stat().st_mtime + 1e-6 >= not_before_epoch
        )
    except (FileNotFoundError, OSError):
        pass
    if foundation_fresh:
        foundation = _verified_receipt_tree(
            foundation_path,
            not_before_epoch=not_before_epoch,
        )
        if foundation is None:
            return "failed", "tts_foundation_terminal_invalid"
        if (
            foundation.get("schema_version") != 2
            or foundation.get("scope") != "tts_0_40k_foundation"
        ):
            return "failed", "tts_foundation_terminal_contract_mismatch"
        if (
            foundation.get("status") == "BLOCKED"
            and foundation.get("formal_acceptance_foundation_pass") is False
        ):
            return "complete", "tts_0_40k_foundation_scientifically_blocked"
        if not (
            foundation.get("status") == "TTS_0_40K_CONFIRMED"
            and foundation.get("formal_acceptance_foundation_pass") is True
        ):
            return "failed", "tts_foundation_terminal_contract_mismatch"
        # A confirmed foundation is only a prerequisite.  It must never turn
        # an attempt into success before controller/headline completion, nor
        # may a stale legacy comparison override it below.
    else:
        comparison = _verified_json(
            confirmation_root / "formal-acceptance-comparison.json",
            not_before_epoch=not_before_epoch,
            sidecar_required=True,
        )
        if (
            comparison is not None
            and comparison.get("formal_acceptance_claim_pass") is False
        ):
            return "complete", "formal_confirmation_scientifically_blocked"

    screen_blocked = _verified_json(
        screen_root / "runs" / "CANDIDATE_SCREEN_BLOCKED.json",
        not_before_epoch=not_before_epoch,
        sidecar_required=True,
    )
    if screen_blocked is not None and screen_blocked.get("status") == "candidate_screen_blocked":
        return "complete", "candidate_screen_scientifically_blocked"

    return "failed", "launcher_exited_without_fresh_proof_terminal"


def publish_terminal(
    queue_root: Path,
    run_id: str,
    attempt_generation: int,
    state: str,
    *,
    reason: str = "unspecified",
) -> bool:
    if state not in {"complete", "failed"}:
        raise ValueError(f"invalid terminal state: {state}")
    lock_path = queue_root / ".attempt-binder.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not _binding_matches(queue_root, run_id, attempt_generation):
            return False
        marker = "QUEUE_COMPLETE" if state == "complete" else "QUEUE_FAILED"
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_generation": attempt_generation,
            "state": state,
            "reason": reason,
            "timestamp_epoch": time.time(),
        }
        _atomic_write(
            _attempt_root(queue_root, run_id, attempt_generation)
            / "binder-terminal.json",
            json.dumps(receipt, sort_keys=True) + "\n",
        )
        _atomic_write(
            queue_root / marker,
            _attempt_payload(run_id, attempt_generation) + "\n",
        )
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-generation", required=True, type=int)
    parser.add_argument("--launcher-pid", required=True, type=int)
    parser.add_argument("--screen-root", required=True, type=Path)
    parser.add_argument("--confirmation-root", required=True, type=Path)
    parser.add_argument("--controller-root", required=True, type=Path)
    parser.add_argument("--headline-root", required=True, type=Path)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--supersede-current", action="store_true")
    args = parser.parse_args()
    if not args.run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_id
    ):
        raise SystemExit("invalid --run-id")
    if args.attempt_generation <= 0:
        raise SystemExit("--attempt-generation must be positive")
    if not math.isfinite(args.heartbeat_seconds) or not (
        0 < args.heartbeat_seconds <= 60
    ):
        raise SystemExit("--heartbeat-seconds must be finite and in (0, 60]")

    identity = capture_process_identity(args.launcher_pid)
    if identity is None:
        raise SystemExit("launcher PID is absent, invalid, or already a zombie")
    not_before_epoch = process_start_epoch(identity)
    if not_before_epoch is None:
        raise SystemExit("cannot bind launcher start time to terminal evidence")
    publish_attempt_binding(
        args.queue_root,
        args.run_id,
        args.attempt_generation,
        identity,
        launcher_start_epoch=not_before_epoch,
        supersede_current=args.supersede_current,
    )
    if not publish_heartbeat(
        args.queue_root,
        args.run_id,
        args.attempt_generation,
        identity,
    ):
        return 0

    while process_identity_matches(identity):
        time.sleep(args.heartbeat_seconds)
        if not publish_heartbeat(
            args.queue_root,
            args.run_id,
            args.attempt_generation,
            identity,
        ):
            return 0

    state, reason = proof_chain_terminal_state(
        screen_root=args.screen_root,
        confirmation_root=args.confirmation_root,
        controller_root=args.controller_root,
        headline_root=args.headline_root,
        not_before_epoch=not_before_epoch,
    )
    publish_terminal(
        args.queue_root,
        args.run_id,
        args.attempt_generation,
        state,
        reason=reason,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
