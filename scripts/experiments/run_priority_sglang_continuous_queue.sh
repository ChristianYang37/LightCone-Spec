#!/usr/bin/env bash
: "${BASH_VERSION:?this queue requires bash}"
set -euo pipefail

# Production-SGLang counterpart to the canonical reference pilot.  Static,
# synchronous TTS and L0 share one immutable continuous-prefix manifest and
# differ only in their method scheduler.
QUEUE_SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=${LIGHTCONE_WORKSPACE:-$(cd "$QUEUE_SOURCE_DIR/../.." && pwd)}
RUNTIME_ROOT=${LIGHTCONE_RUNTIME_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lightcone-spec}
LC=${LIGHTCONE_CLI:-$RUNTIME_ROOT/venv/bin/lightcone-spec}
PY=${LIGHTCONE_PYTHON:-$RUNTIME_ROOT/venv/bin/python}
PY_BIN_DIR=$(dirname -- "$PY")
MANIFEST=${LIGHTCONE_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_continuous40k_calibration_v3.json}
LOCKFILE=${LIGHTCONE_LOCKFILE:-$RUNTIME_ROOT/priority/continuous-prefix.lock.json}
MODEL_ROOTS=${LIGHTCONE_MODEL_ROOTS:-$RUNTIME_ROOT/priority/continuous-prefix.model-roots.json}
ARTIFACT_ROOT=${LIGHTCONE_ARTIFACT_ROOT:-$RUNTIME_ROOT/priority/continuous-prefix/runs}
ANALYSIS_ROOT=${LIGHTCONE_ANALYSIS_ROOT:-$RUNTIME_ROOT/priority/continuous-prefix/analysis}
INPUT_RECEIPT=${LIGHTCONE_INPUT_RECEIPT:-$RUNTIME_ROOT/priority/continuous-prefix/dataset-preflight.json}
CUDA_TOOLKIT=${LIGHTCONE_CUDA_TOOLKIT_ROOT:-$RUNTIME_ROOT/cuda-12.9}
STATE="$ARTIFACT_ROOT/priority-state.jsonl"
EXECUTION_COMPLETE="$ARTIFACT_ROOT/EXECUTION_COMPLETE.json"
COMPLETE="$ARTIFACT_ROOT/PRIORITY_COMPLETE.json"
OBJECTIVE_BLOCKED="$ARTIFACT_ROOT/OBJECTIVE_BLOCKED.json"
FAILED="$ARTIFACT_ROOT/PRIORITY_FAILED.json"

mkdir -p "$ARTIFACT_ROOT" "$ANALYSIS_ROOT" "$(dirname "$INPUT_RECEIPT")"

record() {
  "$PY" - "$STATE" "$1" "$2" "${3:-}" <<'PY'
import datetime as dt
import json
import os
import sys

path, phase, status, detail = sys.argv[1:]
row = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "phase": phase,
    "status": status,
    "detail": detail,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

headline() {
  env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING \
    CUDA_HOME="$CUDA_TOOLKIT" \
    PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH" \
    LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LC" "$@"
}

finished=false
on_exit() {
  rc=$?
  if [ "$finished" = false ] && [ "$rc" -ne 0 ]; then
    record queue failed "exit_code=$rc"
    "$PY" - "$FAILED" "$rc" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps({
    "status": "failed_resumable",
    "exit_code": int(sys.argv[2]),
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
  fi
}
trap on_exit EXIT

if [ -s "$COMPLETE" ]; then
  record queue skipped_complete "$COMPLETE"
  finished=true
  exit 0
fi
rm -f "$FAILED" "$OBJECTIVE_BLOCKED"
record queue started "continuous true-prefix Static/TTS/L0 tail-LoRA"

record inputs running "offline locked dataset preflight"
PROMPT_LIMIT=$("$PY" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("engine_params", {}).get("prompt_limit")
if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise SystemExit(
        "priority manifest engine_params.prompt_limit must be a positive integer"
    )
print(value)
PY
)
"$LC" prepare-datasets \
  --lockfile "$LOCKFILE" \
  --datasets math500 \
  --limit "$PROMPT_LIMIT" \
  --output "$INPUT_RECEIPT"
record inputs complete "$INPUT_RECEIPT"

record inference running "methods=static,tts,naive_async mode=lora stride=1"
headline run-manifest \
  --manifest "$MANIFEST" \
  --artifact-root "$ARTIFACT_ROOT" \
  --lockfile "$LOCKFILE" \
  --runtime-root "$RUNTIME_ROOT" \
  --model-roots "$MODEL_ROOTS" \
  --methods static tts naive_async \
  --weight-update-mode lora
record inference complete "$ARTIFACT_ROOT"

record validation running "hash/coverage/exactness"
"$LC" validate-artifacts \
  --artifact-root "$ARTIFACT_ROOT" \
  --manifest "$MANIFEST" \
  --methods static tts naive_async \
  --weight-update-mode lora \
  --coverage-output "$ANALYSIS_ROOT/coverage.json"

"$LC" analyze \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-dir "$ANALYSIS_ROOT/vs-static" \
  --manifest "$MANIFEST" \
  --methods static tts naive_async \
  --weight-update-mode lora \
  --baseline static

"$LC" analyze \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-dir "$ANALYSIS_ROOT/vs-tts" \
  --manifest "$MANIFEST" \
  --methods static tts naive_async \
  --weight-update-mode lora \
  --baseline tts
record validation complete "$ANALYSIS_ROOT"

"$PY" - "$EXECUTION_COMPLETE" "$MANIFEST" "$LOCKFILE" "$MODEL_ROOTS" "$ANALYSIS_ROOT" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

path, manifest, lockfile, roots, analysis = map(Path, sys.argv[1:])
payload = {
    "status": "execution_complete",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "manifest": str(manifest),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "lockfile": str(lockfile),
    "lockfile_sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest(),
    "model_roots": str(roots),
    "model_roots_sha256": hashlib.sha256(roots.read_bytes()).hexdigest(),
    "analysis_root": str(analysis),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
record execution complete "$EXECUTION_COMPLETE"

record objective running "require TTS>Static and L0>TTS algorithmic gates"
if "$PY" - \
  "$COMPLETE" \
  "$OBJECTIVE_BLOCKED" \
  "$ANALYSIS_ROOT/vs-static/p5_claim_gates.json" \
  "$ANALYSIS_ROOT/vs-tts/p5_claim_gates.json" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

passed_path, blocked_path, static_gate_path, tts_gate_path = map(
    Path, sys.argv[1:]
)


def one_gate(path: Path, *, method: str, baseline: str) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("baseline_method") == baseline
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one {method}>{baseline} gate in {path}, got {len(selected)}"
        )
    gate = selected[0]
    required = (
        gate.get("scientific_sample_pass") is True
        and gate.get("exactness_pass") is True
        and gate.get("algorithmic_pass") is True
        and gate.get("window_dominance_pass") is True
    )
    return {"passed": required, "gate": gate}


checks = {
    "tts_over_static": one_gate(
        static_gate_path, method="tts", baseline="static"
    ),
    "l0_over_tts": one_gate(
        tts_gate_path, method="naive_async", baseline="tts"
    ),
}
passed = all(check["passed"] for check in checks.values())
payload = {
    "status": "objective_pass" if passed else "objective_blocked",
    "scope": "algorithmic_continuous_prefix",
    "engineering_pass_evaluated": False,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "checks": checks,
    "gate_files": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (static_gate_path, tts_gate_path)
    },
}
destination = passed_path if passed else blocked_path
tmp = destination.with_suffix(destination.suffix + ".tmp")
tmp.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(tmp, destination)
if passed:
    blocked_path.unlink(missing_ok=True)
raise SystemExit(0 if passed else 8)
PY
then
  record objective pass "$COMPLETE"
else
  rc=$?
  record objective blocked "$OBJECTIVE_BLOCKED"
  # Execution evidence is complete and immutable; this is a scientific gate,
  # not a resumable runtime failure.  A new method/LR must use a new artifact
  # identity instead of relabeling these results as success.
  finished=true
  exit "$rc"
fi
record queue complete "$COMPLETE"
finished=true
