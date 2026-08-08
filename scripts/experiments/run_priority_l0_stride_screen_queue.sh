#!/usr/bin/env bash
: "${BASH_VERSION:?this queue requires bash}"
set -euo pipefail

# Candidate-only DFlash stride screen.  This queue selects confirmation inputs;
# neither terminal receipt is evidence for a superiority claim.
QUEUE_SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=${LIGHTCONE_WORKSPACE:-$(cd "$QUEUE_SOURCE_DIR/../.." && pwd)}
RUNTIME_ROOT=${LIGHTCONE_RUNTIME_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lightcone-spec}
LC=${LIGHTCONE_CLI:-$RUNTIME_ROOT/venv/bin/lightcone-spec}
PY=${LIGHTCONE_PYTHON:-$RUNTIME_ROOT/venv/bin/python}
PY_BIN_DIR=$(dirname -- "$PY")
MANIFEST=${LIGHTCONE_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_stride_screen_v1.json}
SELECTOR=${LIGHTCONE_STRIDE_SELECTOR:-$WORKSPACE/scripts/experiments/select_p5_stride_screen.py}
LOCKFILE=${LIGHTCONE_LOCKFILE:-$RUNTIME_ROOT/priority/stride-screen.lock.json}
MODEL_ROOTS=${LIGHTCONE_MODEL_ROOTS:-$RUNTIME_ROOT/priority/stride-screen.model-roots.json}
SCREEN_ROOT=${LIGHTCONE_SCREEN_ROOT:-$RUNTIME_ROOT/priority/stride-screen}
ARTIFACT_ROOT=${LIGHTCONE_ARTIFACT_ROOT:-$SCREEN_ROOT/runs}
ANALYSIS_ROOT=${LIGHTCONE_ANALYSIS_ROOT:-$SCREEN_ROOT/analysis}
INPUT_RECEIPT=${LIGHTCONE_INPUT_RECEIPT:-$SCREEN_ROOT/dataset-preflight.json}
CUDA_TOOLKIT=${LIGHTCONE_CUDA_TOOLKIT_ROOT:-$RUNTIME_ROOT/cuda-12.9}
STATE=${LIGHTCONE_STATE_PATH:-$ARTIFACT_ROOT/priority-state.jsonl}
EXECUTION_COMPLETE=${LIGHTCONE_EXECUTION_RECEIPT:-$ARTIFACT_ROOT/EXECUTION_COMPLETE.json}
SELECTION_RECEIPT=${LIGHTCONE_SELECTION_RECEIPT:-$ANALYSIS_ROOT/stride-selection.json}
SELECTED=${LIGHTCONE_SELECTED_RECEIPT:-$ARTIFACT_ROOT/CANDIDATE_SCREEN_SELECTED.json}
BLOCKED=${LIGHTCONE_BLOCKED_RECEIPT:-$ARTIFACT_ROOT/CANDIDATE_SCREEN_BLOCKED.json}
FAILED=${LIGHTCONE_FAILED_RECEIPT:-$ARTIFACT_ROOT/PRIORITY_FAILED.json}
QUEUE_SOURCE=${BASH_SOURCE[0]}
QUEUE_LOCK=${LIGHTCONE_QUEUE_LOCK_PATH:-$SCREEN_ROOT/.priority-l0-stride-screen.lock}
PROCESS_CONTROL=${LIGHTCONE_QUEUE_PROCESS_CONTROL:-$QUEUE_SOURCE_DIR/priority_queue_process.sh}

# shellcheck source=priority_queue_process.sh
source "$PROCESS_CONTROL"

mkdir -p "$ARTIFACT_ROOT" "$ANALYSIS_ROOT" "$(dirname "$INPUT_RECEIPT")"
exec 9>>"$QUEUE_LOCK"
if ! flock -n 9; then
  echo "priority stride-screen queue is already running: $QUEUE_LOCK" >&2
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

headline() {
  queue_run_managed env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING \
    -u PYTORCH_ALLOC_CONF \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_LOCKED" \
    CUDA_HOME="$CUDA_TOOLKIT" \
    PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH" \
    LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LC" "$@"
}

write_receipt() {
  "$PY" - "$@" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

destination = Path(sys.argv[1])
status = sys.argv[2]
scope = sys.argv[3]
evidence = [Path(value) for value in sys.argv[4:]]
rows = []
for path in evidence:
    if not path.is_file():
        raise SystemExit(f"receipt evidence is missing: {path}")
    rows.append({
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
payload = {
    "schema_version": 1,
    "status": status,
    "scope": scope,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "evidence": rows,
}
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
destination.parent.mkdir(parents=True, exist_ok=True)
with temporary.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
sidecar = Path(str(destination) + ".sha256")
sidecar_tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
sidecar_tmp.write_text(digest + "\n", encoding="utf-8")
os.replace(sidecar_tmp, sidecar)
PY
}

receipt_valid() {
  "$PY" - "$1" "$2" "$3" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
expected_status = sys.argv[2]
expected_scope = sys.argv[3]
active = set()
verified = set()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def load_receipt(receipt):
    receipt = receipt.resolve()
    if receipt in verified:
        return True
    if receipt in active or not receipt.is_file():
        return False
    active.add(receipt)
    try:
        body = receipt.read_bytes()
        sidecar = Path(str(receipt) + ".sha256")
        if (
            not sidecar.is_file()
            or sidecar.read_text(encoding="utf-8").strip() != sha256(body)
        ):
            return False
        payload = json.loads(body)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("status"), str)
            or not isinstance(payload.get("scope"), str)
            or not isinstance(payload.get("evidence"), list)
        ):
            return False
        for row in payload["evidence"]:
            if not isinstance(row, dict):
                return False
            raw_path = row.get("path")
            expected_sha = row.get("sha256")
            if (
                not isinstance(raw_path, str)
                or not isinstance(expected_sha, str)
                or len(expected_sha) != 64
            ):
                return False
            evidence = Path(raw_path)
            if not evidence.is_absolute() or not evidence.is_file():
                return False
            evidence_body = evidence.read_bytes()
            if sha256(evidence_body) != expected_sha:
                return False
            try:
                nested = json.loads(evidence_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                nested = None
            if isinstance(nested, dict) and {
                "schema_version",
                "status",
                "scope",
                "evidence",
            }.issubset(nested):
                if not load_receipt(evidence):
                    return False
        verified.add(receipt)
        return True
    finally:
        active.remove(receipt)


try:
    payload = json.loads(path.read_bytes()) if path.is_file() else {}
    valid = load_receipt(path)
    valid = (
        valid
        and payload.get("status") == expected_status
        and payload.get("scope") == expected_scope
    )
except (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    json.JSONDecodeError,
):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

finished=false
on_exit() {
  rc=$?
  if [ "$finished" = false ] && [ "$rc" -ne 0 ]; then
    record queue failed_resumable \
      "exit_code=$rc signal=${QUEUE_STOP_SIGNAL:-none}"
    write_receipt "$FAILED" failed_resumable candidate_stride_screen
  fi
}
trap on_exit EXIT
queue_process_control_init "$PY"

PYTORCH_CUDA_ALLOC_CONF_LOCKED=$("$PY" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("engine_params", {}).get("pytorch_cuda_alloc_conf")
if not isinstance(value, str) or not value.strip():
    raise SystemExit(
        "stride-screen engine_params.pytorch_cuda_alloc_conf must be non-empty"
    )
print(value)
PY
)

selected_valid=false
blocked_valid=false
if receipt_valid \
  "$SELECTED" candidate_screen_selected candidate_screen_only_no_claim; then
  selected_valid=true
elif [ -e "$SELECTED" ] || [ -e "$SELECTED.sha256" ]; then
  echo "invalid selected terminal receipt: $SELECTED" >&2
  exit 8
fi
if receipt_valid \
  "$BLOCKED" candidate_screen_blocked candidate_screen_only_no_claim; then
  blocked_valid=true
elif [ -e "$BLOCKED" ] || [ -e "$BLOCKED.sha256" ]; then
  echo "invalid blocked terminal receipt: $BLOCKED" >&2
  exit 8
fi
if [ "$selected_valid" = true ] && [ "$blocked_valid" = true ]; then
  echo "conflicting candidate-screen terminal receipts" >&2
  exit 8
fi
if [ "$selected_valid" = true ] || [ "$blocked_valid" = true ]; then
  rm -f "$FAILED" "$FAILED.sha256"
  record queue skipped_terminal "selected=$selected_valid blocked=$blocked_valid"
  finished=true
  exit 0
fi
rm -f "$FAILED" "$FAILED.sha256"
record queue started \
  "DFlash Static/TTS/L0 stride candidate screen; allocator=$PYTORCH_CUDA_ALLOC_CONF_LOCKED"

PROMPT_LIMIT=$("$PY" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("engine_params", {}).get("prompt_limit")
if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise SystemExit("stride-screen prompt_limit must be a positive integer")
print(value)
PY
)
record inputs running "locked LiveCodeBench limit=$PROMPT_LIMIT"
queue_run_managed "$LC" prepare-datasets \
  --lockfile "$LOCKFILE" \
  --datasets livecodebench \
  --limit "$PROMPT_LIMIT" \
  --output "$INPUT_RECEIPT"
record inputs complete "$INPUT_RECEIPT"

record inference running "methods=static,tts,naive_async mode=lora strides=1,4,8,16"
headline run-manifest \
  --manifest "$MANIFEST" \
  --artifact-root "$ARTIFACT_ROOT" \
  --lockfile "$LOCKFILE" \
  --runtime-root "$RUNTIME_ROOT" \
  --model-roots "$MODEL_ROOTS" \
  --methods static tts naive_async \
  --weight-update-mode lora
record inference complete "$ARTIFACT_ROOT"

record validation running "immutable coverage and artifact integrity"
queue_run_managed "$LC" validate-artifacts \
  --artifact-root "$ARTIFACT_ROOT" \
  --manifest "$MANIFEST" \
  --methods static tts naive_async \
  --weight-update-mode lora \
  --coverage-output "$ANALYSIS_ROOT/coverage.json"

record analysis running "whole-root stride-aware baselines"
queue_run_managed "$LC" analyze \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-dir "$ANALYSIS_ROOT/vs-static" \
  --manifest "$MANIFEST" \
  --methods static tts naive_async \
  --weight-update-mode lora \
  --baseline static
queue_run_managed "$LC" analyze \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-dir "$ANALYSIS_ROOT/vs-tts" \
  --manifest "$MANIFEST" \
  --methods static tts naive_async \
  --weight-update-mode lora \
  --baseline tts
record analysis complete "$ANALYSIS_ROOT"

write_receipt \
  "$EXECUTION_COMPLETE" execution_complete candidate_stride_screen_no_claim \
  "$QUEUE_SOURCE" "$SELECTOR" \
  "$MANIFEST" "$MANIFEST.sha256" \
  "$LOCKFILE" "$LOCKFILE.sha256" \
  "$MODEL_ROOTS" "$MODEL_ROOTS.sha256" \
  "$INPUT_RECEIPT" "$INPUT_RECEIPT.sha256" \
  "$ANALYSIS_ROOT/coverage.json" "$ANALYSIS_ROOT/coverage.json.sha256" \
  "$ANALYSIS_ROOT/vs-static/analysis-hashes.json" \
  "$ANALYSIS_ROOT/vs-tts/analysis-hashes.json"
record execution complete "$EXECUTION_COMPLETE"

record selection running "deterministic confirmation-candidate selection"
queue_run_managed "$PY" "$SELECTOR" \
  --manifest "$MANIFEST" \
  --coverage "$ANALYSIS_ROOT/coverage.json" \
  --vs-static-analysis "$ANALYSIS_ROOT/vs-static" \
  --vs-tts-analysis "$ANALYSIS_ROOT/vs-tts" \
  --output "$SELECTION_RECEIPT"

SELECTION_STATUS=$("$PY" - "$SELECTION_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
objective = payload.get("objective_screen_pass")
if status == "winner_selected" and objective is True:
    print("selected")
elif status == "scientifically_blocked" and objective is False:
    print("blocked")
else:
    raise SystemExit("selector returned an inconsistent candidate-screen status")
PY
)

if [ "$SELECTION_STATUS" = selected ]; then
  write_receipt \
    "$SELECTED" candidate_screen_selected candidate_screen_only_no_claim \
    "$EXECUTION_COMPLETE" "$EXECUTION_COMPLETE.sha256" \
    "$SELECTION_RECEIPT" "$SELECTION_RECEIPT.sha256"
  record selection selected "$SELECTED"
else
  write_receipt \
    "$BLOCKED" candidate_screen_blocked candidate_screen_only_no_claim \
    "$EXECUTION_COMPLETE" "$EXECUTION_COMPLETE.sha256" \
    "$SELECTION_RECEIPT" "$SELECTION_RECEIPT.sha256"
  record selection blocked "$BLOCKED"
fi

record queue complete "candidate screen terminal=$SELECTION_STATUS"
finished=true
