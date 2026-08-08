#!/usr/bin/env bash

# Signal-safe child ownership for long-running experiment queues.  Each
# managed command gets a new session so an INT/TERM sent only to the queue can
# be forwarded to that command and all of its descendants without touching
# the SSH session or unrelated jobs.  The caller owns the graceful timeout;
# this helper keeps waiting after forwarding the signal.

queue_process_control_init() {
  if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "queue_process_control_init requires a Python executable" >&2
    return 64
  fi
  QUEUE_PROCESS_PYTHON=$1
  QUEUE_MANAGED_PID=
  QUEUE_MANAGED_PGID=
  QUEUE_MANAGED_STARTING=0
  QUEUE_STOP_SIGNAL=
  QUEUE_STOP_EXIT_CODE=0
  trap 'queue_forward_signal INT' INT
  trap 'queue_forward_signal TERM' TERM
}

queue_forward_signal() {
  local signal=$1
  case "$signal" in
    INT) QUEUE_STOP_EXIT_CODE=130 ;;
    TERM) QUEUE_STOP_EXIT_CODE=143 ;;
    *) return 64 ;;
  esac
  QUEUE_STOP_SIGNAL=$signal
  if [ -n "${QUEUE_MANAGED_PGID:-}" ]; then
    # The PGID is created by the Python launcher below and can only name the
    # child we started.  The PID fallback closes the small pre-setsid race.
    kill -s "$signal" -- "-$QUEUE_MANAGED_PGID" 2>/dev/null \
      || kill -s "$signal" -- "$QUEUE_MANAGED_PID" 2>/dev/null \
      || true
  elif [ "${QUEUE_MANAGED_STARTING:-0}" -eq 0 ]; then
    # Outside a managed command there is no process subtree to drain.  Exit
    # immediately so a signal cannot race into publication of a success
    # terminal during a short shell/Python bookkeeping step.
    trap - INT TERM
    exit "$QUEUE_STOP_EXIT_CODE"
  fi
}

queue_run_managed() {
  if [ "$#" -eq 0 ]; then
    echo "queue_run_managed requires a command" >&2
    return 64
  fi
  if [ -n "${QUEUE_STOP_SIGNAL:-}" ]; then
    return "$QUEUE_STOP_EXIT_CODE"
  fi

  QUEUE_MANAGED_STARTING=1
  "$QUEUE_PROCESS_PYTHON" - "$@" <<'PY' &
import os
import signal
import sys

argv = sys.argv[1:]
if not argv:
    raise SystemExit(64)
os.setsid()
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
os.execvpe(argv[0], argv, os.environ)
PY
  QUEUE_MANAGED_PID=$!
  QUEUE_MANAGED_PGID=$QUEUE_MANAGED_PID
  QUEUE_MANAGED_STARTING=0
  if [ -n "${QUEUE_STOP_SIGNAL:-}" ]; then
    queue_forward_signal "$QUEUE_STOP_SIGNAL"
  fi

  local child_rc=0
  while :; do
    if wait "$QUEUE_MANAGED_PID"; then
      child_rc=0
    else
      child_rc=$?
    fi
    if ! kill -0 "$QUEUE_MANAGED_PID" 2>/dev/null; then
      break
    fi
    # wait(1) is interrupted when Bash runs the signal trap.  Continue
    # waiting for the owned child to flush and close its artifacts.
  done
  if [ -n "${QUEUE_STOP_SIGNAL:-}" ]; then
    # The session leader may exit before a worker finishes its own shutdown.
    # A process-group existence check lets those descendants flush without an
    # internal timeout; the supervising wrapper decides when to escalate.
    while kill -0 -- "-$QUEUE_MANAGED_PGID" 2>/dev/null; do
      "$QUEUE_PROCESS_PYTHON" -c 'import time; time.sleep(0.05)'
    done
  fi
  QUEUE_MANAGED_PID=
  QUEUE_MANAGED_PGID=

  if [ -n "${QUEUE_STOP_SIGNAL:-}" ]; then
    return "$QUEUE_STOP_EXIT_CODE"
  fi
  return "$child_rc"
}
