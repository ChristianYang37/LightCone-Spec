#!/usr/bin/env python3
"""Start a long-running queue launcher that can still be interrupted.

A shell backgrounds an asynchronous list by setting SIGINT and SIGQUIT to
SIG_IGN in the child, and that disposition survives exec.  A non-interactive
bash cannot trap or reset a signal that was already ignored when it started,
so ``nohup bash -c '...' &`` produces a launcher whose ``trap ... INT`` handler
is silently dead: INT never drains the managed child and never publishes a
``failed_resumable`` receipt.

This helper becomes a session leader, restores default dispositions for the
signals the queue relies on, redirects stdio to a log, and then execs the
launcher.  Signalling the printed PGID therefore reaches the queue traps.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

RESET_SIGNALS = ("SIGHUP", "SIGINT", "SIGQUIT", "SIGTERM", "SIGPIPE")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="combined stdout/stderr log")
    parser.add_argument(
        "--pid-file", help="optional file receiving the launcher pid/pgid JSON"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = [value for value in args.command if value != "--"]
    if not command:
        raise SystemExit("a command is required")

    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child > 0:
        os.close(write_fd)
        with os.fdopen(read_fd, "r") as handle:
            payload = handle.read().strip()
        os.waitpid(child, 0)
        if not payload:
            raise SystemExit("launcher failed to report its identity")
        if args.pid_file:
            with open(args.pid_file, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        print(payload)
        return 0

    os.close(read_fd)
    # Detach from the caller's session so the queue owns its own process group
    # and an SSH disconnect cannot deliver SIGHUP to it.
    os.setsid()

    grandchild = os.fork()
    if grandchild > 0:
        with os.fdopen(write_fd, "w") as handle:
            handle.write(
                json.dumps({"launcher_pid": grandchild, "launcher_pgid": grandchild})
            )
            handle.flush()
        os._exit(0)

    os.close(write_fd)
    os.setpgid(0, 0)

    for name in RESET_SIGNALS:
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, set())

    log_fd = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    if log_fd > 2:
        os.close(log_fd)
    if null_fd > 2:
        os.close(null_fd)

    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    sys.exit(main())
