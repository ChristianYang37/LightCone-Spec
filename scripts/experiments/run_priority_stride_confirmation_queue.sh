#!/usr/bin/env bash
: "${BASH_VERSION:?this queue requires bash}"
set -euo pipefail

# Independent 0--40K TTS-vs-Static foundation selected by the non-claim
# stride screen.  Its attested terminal, not the process exit code, is
# authoritative.
QUEUE_SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=${LIGHTCONE_WORKSPACE:-$(cd "$QUEUE_SOURCE_DIR/../.." && pwd)}
RUNTIME_ROOT=${LIGHTCONE_RUNTIME_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lightcone-spec}
LC=${LIGHTCONE_CLI:-$RUNTIME_ROOT/venv/bin/lightcone-spec}
PY=${LIGHTCONE_PYTHON:-$RUNTIME_ROOT/venv/bin/python}
PY_BIN_DIR=$(dirname -- "$PY")
FOUNDATION_TOOL=${LIGHTCONE_TTS_FOUNDATION_TOOL:-$WORKSPACE/scripts/experiments/p5_tts_foundation.py}
SCREEN_QUEUE=${LIGHTCONE_SCREEN_QUEUE:-$WORKSPACE/scripts/experiments/run_priority_l0_stride_screen_queue.sh}
SCREEN_ROOT=${LIGHTCONE_SCREEN_ROOT:-$RUNTIME_ROOT/priority/stride-screen}
SCREEN_ARTIFACT_ROOT=${LIGHTCONE_SCREEN_ARTIFACT_ROOT:-$SCREEN_ROOT/runs}
SCREEN_ANALYSIS_ROOT=${LIGHTCONE_SCREEN_ANALYSIS_ROOT:-$SCREEN_ROOT/analysis}
SCREEN_QUEUE_LOCK=${LIGHTCONE_SCREEN_QUEUE_LOCK_PATH:-$SCREEN_ROOT/.priority-l0-stride-screen.lock}
SCREEN_MANIFEST=${LIGHTCONE_SCREEN_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_stride_screen_v1.json}
SCREEN_SELECTOR=${LIGHTCONE_SCREEN_STRIDE_SELECTOR:-$WORKSPACE/scripts/experiments/select_p5_stride_screen.py}
SCREEN_INPUT_RECEIPT=${LIGHTCONE_SCREEN_INPUT_RECEIPT:-$SCREEN_ROOT/dataset-preflight.json}
SCREEN_STATE=${LIGHTCONE_SCREEN_STATE_PATH:-$SCREEN_ARTIFACT_ROOT/priority-state.jsonl}
SCREEN_EXECUTION=${LIGHTCONE_SCREEN_EXECUTION_RECEIPT:-$SCREEN_ARTIFACT_ROOT/EXECUTION_COMPLETE.json}
SCREEN_SELECTION=${LIGHTCONE_SCREEN_SELECTION_RECEIPT:-$SCREEN_ANALYSIS_ROOT/stride-selection.json}
SCREEN_FAILED=${LIGHTCONE_SCREEN_FAILED_RECEIPT:-$SCREEN_ARTIFACT_ROOT/PRIORITY_FAILED.json}
SELECTED=${LIGHTCONE_SELECTED_RECEIPT:-$SCREEN_ROOT/runs/CANDIDATE_SCREEN_SELECTED.json}
BLOCKED=${LIGHTCONE_BLOCKED_RECEIPT:-$SCREEN_ROOT/runs/CANDIDATE_SCREEN_BLOCKED.json}
LOCKFILE=${LIGHTCONE_LOCKFILE:-$RUNTIME_ROOT/priority/stride-screen.lock.json}
MODEL_ROOTS=${LIGHTCONE_MODEL_ROOTS:-$RUNTIME_ROOT/priority/stride-screen.model-roots.json}
ROOT=${LIGHTCONE_CONFIRMATION_ROOT:-$RUNTIME_ROOT/priority/tts-foundation}
RUNTIME_FINGERPRINT=${LIGHTCONE_RUNTIME_FINGERPRINT:-$ROOT/runtime-fingerprint.json}
FOUNDATION_ROOT=${LIGHTCONE_TTS_FOUNDATION_ROOT:-$ROOT/tts-0-40k-foundation}
FOUNDATION_MANIFEST=${LIGHTCONE_TTS_FOUNDATION_MANIFEST:-$FOUNDATION_ROOT/manifest.json}
FOUNDATION_GENERATION=${LIGHTCONE_TTS_FOUNDATION_GENERATION:-$FOUNDATION_ROOT/generation.json}
FOUNDATION_ARTIFACT_ROOT=${LIGHTCONE_TTS_FOUNDATION_ARTIFACT_ROOT:-$FOUNDATION_ROOT/runs}
FOUNDATION_ANALYSIS_ROOT=${LIGHTCONE_TTS_FOUNDATION_ANALYSIS_ROOT:-$FOUNDATION_ROOT/analysis-vs-static}
FOUNDATION_COVERAGE=${LIGHTCONE_TTS_FOUNDATION_COVERAGE:-$FOUNDATION_ROOT/coverage.json}
FOUNDATION_TERMINAL=${LIGHTCONE_TTS_FOUNDATION_TERMINAL:-$FOUNDATION_ROOT/TTS_0_40K_FOUNDATION.json}
# Reuse the existing alias-safety validator for the actual foundation paths.
MANIFEST=$FOUNDATION_MANIFEST
GENERATION=$FOUNDATION_GENERATION
ARTIFACT_ROOT=$FOUNDATION_ARTIFACT_ROOT
ANALYSIS_ROOT=$FOUNDATION_ANALYSIS_ROOT
COMPARISON=$FOUNDATION_TERMINAL
FAILED=${LIGHTCONE_FAILED_RECEIPT:-$ROOT/PRIORITY_FAILED.json}
CUDA_TOOLKIT=${LIGHTCONE_CUDA_TOOLKIT_ROOT:-$RUNTIME_ROOT/cuda-12.9}
QUEUE_LOCK=${LIGHTCONE_QUEUE_LOCK_PATH:-$ROOT/.confirmation.lock}
STATE=${LIGHTCONE_STATE_PATH:-$ROOT/priority-state.jsonl}
QUEUE_SOURCE=${BASH_SOURCE[0]}
QUEUE_SOURCE_DIR=$(cd -- "$(dirname -- "$QUEUE_SOURCE")" && pwd)
PROCESS_CONTROL=${LIGHTCONE_QUEUE_PROCESS_CONTROL:-$QUEUE_SOURCE_DIR/priority_queue_process.sh}

# shellcheck source=priority_queue_process.sh
source "$PROCESS_CONTROL"

# Fail before opening a lock or clearing a resumable receipt when an override
# aliases scientific terminals, mutable queue files, or screen/confirmation
# artifact roots. These files have intentionally different mutability rules.
"$PY" - \
  "$ARTIFACT_ROOT" "$ANALYSIS_ROOT" \
  "$SCREEN_ARTIFACT_ROOT" "$SCREEN_ANALYSIS_ROOT" \
  "$MANIFEST" "$GENERATION" "$COMPARISON" "$FAILED" "$STATE" "$QUEUE_LOCK" \
  "$SELECTED" "$BLOCKED" "$LOCKFILE" "$MODEL_ROOTS" "$SCREEN_QUEUE_LOCK" \
  "$SCREEN_MANIFEST" "$SCREEN_SELECTOR" "$SCREEN_INPUT_RECEIPT" \
  "$SCREEN_STATE" "$SCREEN_EXECUTION" "$SCREEN_SELECTION" "$SCREEN_FAILED" \
  "$FOUNDATION_COVERAGE" "$RUNTIME_FINGERPRINT" "$FOUNDATION_TOOL" <<'PY'
import sys
from pathlib import Path

(
    artifact_root,
    analysis_root,
    screen_artifact_root,
    screen_analysis_root,
    manifest,
    generation,
    comparison,
    failed,
    state,
    queue_lock,
    selected,
    blocked,
    lockfile,
    model_roots,
    screen_queue_lock,
    screen_manifest,
    screen_selector,
    screen_input,
    screen_state,
    screen_execution,
    screen_selection,
    screen_failed,
    foundation_coverage,
    runtime_fingerprint,
    foundation_tool,
) = (Path(value).expanduser().resolve() for value in sys.argv[1:])

confirmation_files = {
    "manifest": manifest,
    "generation": generation,
    "comparison": comparison,
    "failed": failed,
    "state": state,
    "queue_lock": queue_lock,
    "foundation_coverage": foundation_coverage,
    "runtime_fingerprint": runtime_fingerprint,
}
protected_inputs = {
    "selected": selected,
    "blocked": blocked,
    "lockfile": lockfile,
    "model_roots": model_roots,
    "foundation_tool": foundation_tool,
}
screen_files = {
    "screen_manifest": screen_manifest,
    "screen_selector": screen_selector,
    "screen_input": screen_input,
    "screen_state": screen_state,
    "screen_execution": screen_execution,
    "screen_selection": screen_selection,
    "screen_failed": screen_failed,
}
all_files = {**confirmation_files, **protected_inputs, **screen_files}
by_path = {}
for name, path in all_files.items():
    by_path.setdefault(path, []).append(name)
collisions = [names for names in by_path.values() if len(names) > 1]
if collisions:
    raise SystemExit(f"confirmation path aliases are forbidden: {collisions}")
if queue_lock == screen_queue_lock:
    raise SystemExit("screen and confirmation queue locks must differ")

def overlaps(left, right):
    return left == right or left in right.parents or right in left.parents

for confirmation in (artifact_root, analysis_root):
    for screen in (screen_artifact_root, screen_analysis_root):
        if overlaps(confirmation, screen):
            raise SystemExit(
                "screen and confirmation artifact/analysis roots must not overlap"
            )
if overlaps(artifact_root, analysis_root):
    raise SystemExit("confirmation artifact and analysis roots must not overlap")
for name, path in confirmation_files.items():
    if any(
        screen == path or screen in path.parents
        for screen in (screen_artifact_root, screen_analysis_root)
    ):
        raise SystemExit(f"confirmation output {name} is inside a screen root")
PY

mkdir -p "$ROOT" "$ARTIFACT_ROOT" "$ANALYSIS_ROOT" "$(dirname "$QUEUE_LOCK")"
exec 9>>"$QUEUE_LOCK"
if ! flock -n 9; then
  echo "stride confirmation queue is already running: $QUEUE_LOCK" >&2
  exit 75
fi

record() {
  "$PY" - "$STATE" "$1" "$2" "${3:-}" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

path, phase, status, detail = sys.argv[1:]
row = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "phase": phase,
    "status": status,
    "detail": detail,
}
target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

write_failure_receipt() {
  "$PY" - "$FAILED" "$QUEUE_SOURCE" "$phase" "$1" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

destination, queue_source, phase, exit_code = sys.argv[1:]
destination = Path(destination)
source = Path(queue_source).resolve()
body = {
    "schema_version": 1,
    "status": "failed_resumable",
    "scope": "tts_0_40k_foundation_queue",
    "phase": phase,
    "exit_code": int(exit_code),
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "evidence": [{
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }],
}
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
with temporary.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
sidecar = Path(str(destination) + ".sha256")
sidecar_tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
with sidecar_tmp.open("w", encoding="utf-8") as handle:
    handle.write(digest + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(sidecar_tmp, sidecar)
PY
}

clear_failed_receipt() {
  rm -f "$FAILED" "$FAILED.sha256"
}

phase=queue
on_exit() {
  rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 42 ]; then
    set +e
    record "$phase" failed_resumable \
      "exit_code=$rc signal=${QUEUE_STOP_SIGNAL:-none}"
    write_failure_receipt "$rc"
  fi
}
trap on_exit EXIT
queue_process_control_init "$PY"

# This is also the authoritative recursive validation of either screen
# terminal. Keep its roots and lock namespace isolated from confirmation
# overrides: both queues intentionally use similarly named LIGHTCONE_* knobs.
# A valid scientific block is distinct from successful confirmation.
phase=screen_terminal
FOUNDATION_ALLOW_FLAG=
SCREEN_RECEIPT=""
SKIP_SCREEN_QUEUE=0
# Oracle-scope resume: if a blocked terminal already exists with intact
# sidecar/evidence hashes, do not re-enter the screen queue.  That queue's
# resume validator still rejects nested schema-v2 selection receipts, which
# would trap an otherwise complete blocked screen.
if [ -f "$BLOCKED" ] && [ -f "$BLOCKED.sha256" ]; then
  if "$PY" "$QUEUE_SOURCE_DIR/verify_screen_terminal_hashes.py" --terminal "$BLOCKED"; then
    SKIP_SCREEN_QUEUE=1
    SCREEN_RECEIPT=$BLOCKED
    FOUNDATION_ALLOW_FLAG=--allow-l0-not-superior-oracle-scope
    clear_failed_receipt
    record "$phase" oracle_scope_resume_skip_screen_queue "$BLOCKED"
  fi
fi
if [ "$SKIP_SCREEN_QUEUE" -eq 0 ]; then
  queue_run_managed env \
    LIGHTCONE_WORKSPACE="$WORKSPACE" \
    LIGHTCONE_RUNTIME_ROOT="$RUNTIME_ROOT" \
    LIGHTCONE_CLI="$LC" \
    LIGHTCONE_PYTHON="$PY" \
    LIGHTCONE_SCREEN_ROOT="$SCREEN_ROOT" \
    LIGHTCONE_MANIFEST="$SCREEN_MANIFEST" \
    LIGHTCONE_STRIDE_SELECTOR="$SCREEN_SELECTOR" \
    LIGHTCONE_ARTIFACT_ROOT="$SCREEN_ARTIFACT_ROOT" \
    LIGHTCONE_ANALYSIS_ROOT="$SCREEN_ANALYSIS_ROOT" \
    LIGHTCONE_INPUT_RECEIPT="$SCREEN_INPUT_RECEIPT" \
    LIGHTCONE_STATE_PATH="$SCREEN_STATE" \
    LIGHTCONE_EXECUTION_RECEIPT="$SCREEN_EXECUTION" \
    LIGHTCONE_SELECTION_RECEIPT="$SCREEN_SELECTION" \
    LIGHTCONE_SELECTED_RECEIPT="$SELECTED" \
    LIGHTCONE_BLOCKED_RECEIPT="$BLOCKED" \
    LIGHTCONE_FAILED_RECEIPT="$SCREEN_FAILED" \
    LIGHTCONE_LOCKFILE="$LOCKFILE" \
    LIGHTCONE_MODEL_ROOTS="$MODEL_ROOTS" \
    LIGHTCONE_CUDA_TOOLKIT_ROOT="$CUDA_TOOLKIT" \
    LIGHTCONE_QUEUE_LOCK_PATH="$SCREEN_QUEUE_LOCK" \
    "$SCREEN_QUEUE"
  if [ -f "$SELECTED" ] && [ -f "$BLOCKED" ]; then
    echo "conflicting selected and blocked screen terminals" >&2
    exit 8
  fi
  # When L0 does not beat acceptance-best TTS, the screen emits BLOCKED.  That
  # ordering is the controller-phase hypothesis; for oracle-scope runs we keep
  # the blocked terminal as attested evidence and continue into the independent
  # TTS foundation.  Final CONFIRMED still requires every L0/L1/L2/L3 gate.
  if [ -f "$BLOCKED" ]; then
    "$PY" "$QUEUE_SOURCE_DIR/verify_screen_terminal_hashes.py" --terminal "$BLOCKED"
    SCREEN_RECEIPT=$BLOCKED
    FOUNDATION_ALLOW_FLAG=--allow-l0-not-superior-oracle-scope
    clear_failed_receipt
    record "$phase" diagnostic_l0_not_superior_continue_controller "$BLOCKED"
  elif [ -f "$SELECTED" ]; then
    SCREEN_RECEIPT=$SELECTED
    clear_failed_receipt
  else
    echo "selected stride-screen receipt is missing after validation: $SELECTED" >&2
    exit 8
  fi
fi

# Attest the exact consumer runtime against the compiler/source references in
# the lockfile.  The file is immutable and its digest is embedded in every
# downstream foundation/controller identity.
phase=runtime_fingerprint
record "$phase" running "$RUNTIME_FINGERPRINT"
# Must not use `python - <<'PY'` through queue_run_managed: that helper already
# consumes stdin for its session-leader wrapper, so a nested heredoc becomes an
# empty program that exits 0 without writing the fingerprint.
queue_run_managed "$PY" "$QUEUE_SOURCE_DIR/write_runtime_fingerprint.py" \
  --lockfile "$LOCKFILE" --output "$RUNTIME_FINGERPRINT"
record "$phase" complete "$RUNTIME_FINGERPRINT"

# Independent 0--40K TTS-vs-Static foundation.  This is the first scientific
# terminal after the candidate screen; the narrower stride confirmation is
# unreachable unless this receipt recursively validates as CONFIRMED.
phase=tts_foundation_generation
queue_run_managed "$PY" "$FOUNDATION_TOOL" build \
  --selected-receipt "$SCREEN_RECEIPT" \
  --source-screen-manifest "$SCREEN_MANIFEST" \
  --lockfile "$LOCKFILE" --model-roots "$MODEL_ROOTS" \
  --runtime-fingerprint "$RUNTIME_FINGERPRINT" \
  --artifact-root "$FOUNDATION_ARTIFACT_ROOT" \
  --foundation-manifest "$FOUNDATION_MANIFEST" \
  --receipt "$FOUNDATION_GENERATION" \
  ${FOUNDATION_ALLOW_FLAG:+"$FOUNDATION_ALLOW_FLAG"}

FOUNDATION_ALLOCATOR=$($PY - "$FOUNDATION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["engine_params"]["pytorch_cuda_alloc_conf"])
PY
)

if [ ! -e "$FOUNDATION_TERMINAL" ] && [ ! -e "$FOUNDATION_TERMINAL.sha256" ]; then
  phase=tts_foundation_inference
  queue_run_managed env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING -u PYTORCH_ALLOC_CONF \
    PYTORCH_CUDA_ALLOC_CONF="$FOUNDATION_ALLOCATOR" CUDA_HOME="$CUDA_TOOLKIT" \
    PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH" \
    LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LC" run-manifest --manifest "$FOUNDATION_MANIFEST" \
      --artifact-root "$FOUNDATION_ARTIFACT_ROOT" --lockfile "$LOCKFILE" \
      --runtime-root "$RUNTIME_ROOT" --model-roots "$MODEL_ROOTS" \
      --methods static tts --weight-update-mode lora
  phase=tts_foundation_validation
  queue_run_managed "$LC" validate-artifacts \
    --artifact-root "$FOUNDATION_ARTIFACT_ROOT" --manifest "$FOUNDATION_MANIFEST" \
    --methods static tts --weight-update-mode lora \
    --coverage-output "$FOUNDATION_COVERAGE"
  phase=tts_foundation_analysis
  queue_run_managed "$LC" analyze \
    --artifact-root "$FOUNDATION_ARTIFACT_ROOT" --output-dir "$FOUNDATION_ANALYSIS_ROOT" \
    --manifest "$FOUNDATION_MANIFEST" --methods static tts \
    --weight-update-mode lora --baseline static
fi
phase=tts_foundation_terminal
queue_run_managed "$PY" "$FOUNDATION_TOOL" compare \
  --selected-receipt "$SCREEN_RECEIPT" --source-screen-manifest "$SCREEN_MANIFEST" \
  --lockfile "$LOCKFILE" --model-roots "$MODEL_ROOTS" \
  --runtime-fingerprint "$RUNTIME_FINGERPRINT" \
  --artifact-root "$FOUNDATION_ARTIFACT_ROOT" \
  --foundation-manifest "$FOUNDATION_MANIFEST" \
  --generation-receipt "$FOUNDATION_GENERATION" \
  --analysis-root "$FOUNDATION_ANALYSIS_ROOT" --coverage "$FOUNDATION_COVERAGE" \
  --receipt "$FOUNDATION_TERMINAL" \
  ${FOUNDATION_ALLOW_FLAG:+"$FOUNDATION_ALLOW_FLAG"}
FOUNDATION_STATUS=$($PY - "$FOUNDATION_TERMINAL" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["status"])
PY
)
if [ "$FOUNDATION_STATUS" != TTS_0_40K_CONFIRMED ]; then
  clear_failed_receipt
  record "$phase" scientifically_blocked "$FOUNDATION_TERMINAL"
  echo TTS_0_40K_BLOCKED
  exit 42
fi
record "$phase" confirmed "$FOUNDATION_TERMINAL"
clear_failed_receipt
echo TTS_0_40K_CONFIRMED
exit 0
