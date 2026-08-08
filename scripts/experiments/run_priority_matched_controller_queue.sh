#!/usr/bin/env bash
: "${BASH_VERSION:?this queue requires bash}"
set -euo pipefail

# Production two-pass controller queue.  The independent TTS foundation is
# recursively revalidated before any controller trace is collected.
# Exit codes: 0 = all L1/L2/L3 gates selected; 42 = valid scientific block;
# 75 = another queue owns the lock; every other non-zero code is resumable
# engineering failure with CONTROLLER_FAILED.json evidence.
QUEUE_SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=${LIGHTCONE_WORKSPACE:-$(cd "$QUEUE_SOURCE_DIR/../.." && pwd)}
RUNTIME_ROOT=${LIGHTCONE_RUNTIME_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lightcone-spec}
LC=${LIGHTCONE_CLI:-$RUNTIME_ROOT/venv/bin/lightcone-spec}
PY=${LIGHTCONE_PYTHON:-$RUNTIME_ROOT/venv/bin/python}
PY_BIN_DIR=$(dirname -- "$PY")
FOUNDATION_TOOL=${LIGHTCONE_TTS_FOUNDATION_TOOL:-$WORKSPACE/scripts/experiments/p5_tts_foundation.py}
FINAL_GATE_TOOL=${LIGHTCONE_FINAL_GATE_TOOL:-$WORKSPACE/scripts/experiments/p5_final_headline_gate.py}
OLD_ABLATION_QUEUE=${LIGHTCONE_OLD_ABLATION_QUEUE:-$WORKSPACE/scripts/experiments/run_remote_experiment_queue.sh}
MANIFEST_BUILDER=${LIGHTCONE_CONTROLLER_MANIFEST_BUILDER:-$WORKSPACE/scripts/experiments/build_p5_matched_controller_manifests.py}
QUEUE_HELPER=${LIGHTCONE_CONTROLLER_QUEUE_HELPER:-$WORKSPACE/scripts/experiments/p5_controller_queue.py}

CONFIRMATION_ROOT=${LIGHTCONE_CONFIRMATION_ROOT:-$RUNTIME_ROOT/priority/tts-foundation}
SCREEN_TERMINAL_ROOT=${LIGHTCONE_SCREEN_TERMINAL_ROOT:-$RUNTIME_ROOT/priority/stride-screen/runs}
if [ -n "${LIGHTCONE_SELECTED_RECEIPT:-}" ]; then
  SELECTED_SCREEN=$LIGHTCONE_SELECTED_RECEIPT
elif [ -f "$SCREEN_TERMINAL_ROOT/CANDIDATE_SCREEN_SELECTED.json" ]; then
  SELECTED_SCREEN=$SCREEN_TERMINAL_ROOT/CANDIDATE_SCREEN_SELECTED.json
elif [ -f "$SCREEN_TERMINAL_ROOT/CANDIDATE_SCREEN_BLOCKED.json" ]; then
  SELECTED_SCREEN=$SCREEN_TERMINAL_ROOT/CANDIDATE_SCREEN_BLOCKED.json
else
  SELECTED_SCREEN=$SCREEN_TERMINAL_ROOT/CANDIDATE_SCREEN_SELECTED.json
fi
SCREEN_MANIFEST=${LIGHTCONE_SCREEN_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_stride_screen_v1.json}
LOCKFILE=${LIGHTCONE_LOCKFILE:-$RUNTIME_ROOT/priority/stride-screen.lock.json}
MODEL_ROOTS=${LIGHTCONE_MODEL_ROOTS:-$RUNTIME_ROOT/priority/stride-screen.model-roots.json}
RUNTIME_FINGERPRINT=${LIGHTCONE_RUNTIME_FINGERPRINT:-$CONFIRMATION_ROOT/runtime-fingerprint.json}
FOUNDATION_ROOT=${LIGHTCONE_TTS_FOUNDATION_ROOT:-$CONFIRMATION_ROOT/tts-0-40k-foundation}
FOUNDATION_MANIFEST=${LIGHTCONE_TTS_FOUNDATION_MANIFEST:-$FOUNDATION_ROOT/manifest.json}
FOUNDATION_GENERATION=${LIGHTCONE_TTS_FOUNDATION_GENERATION:-$FOUNDATION_ROOT/generation.json}
FOUNDATION_ARTIFACT_ROOT=${LIGHTCONE_TTS_FOUNDATION_ARTIFACT_ROOT:-$FOUNDATION_ROOT/runs}
FOUNDATION_ANALYSIS_ROOT=${LIGHTCONE_TTS_FOUNDATION_ANALYSIS_ROOT:-$FOUNDATION_ROOT/analysis-vs-static}
FOUNDATION_COVERAGE=${LIGHTCONE_TTS_FOUNDATION_COVERAGE:-$FOUNDATION_ROOT/coverage.json}
FOUNDATION_TERMINAL=${LIGHTCONE_TTS_FOUNDATION_TERMINAL:-$FOUNDATION_ROOT/TTS_0_40K_FOUNDATION.json}

ROOT=${LIGHTCONE_CONTROLLER_QUEUE_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-matched-controller-v9}
MANIFEST_ROOT=${LIGHTCONE_CONTROLLER_MANIFEST_ROOT:-$ROOT/manifests}
GENERATION=${LIGHTCONE_CONTROLLER_GENERATION_RECEIPT:-$MANIFEST_ROOT/matched-generation.json}
TRACE_ROOT=${LIGHTCONE_CONTROLLER_TRACE_ROOT:-$ROOT/traces}
PHASE1_TRACE_ROOT=${LIGHTCONE_PHASE1_TRACE_ROOT:-$TRACE_ROOT/phase1}
PHASE2_TRACE_ROOT=${LIGHTCONE_PHASE2_TRACE_ROOT:-$TRACE_ROOT/phase2}
PHASE1_CONTROLLER_ROOT=${LIGHTCONE_PHASE1_CONTROLLER_ROOT:-$ROOT/controller-phase1}
FINAL_CONTROLLER_ROOT=${LIGHTCONE_FINAL_CONTROLLER_ROOT:-$ROOT/controller-final}
PHASE1_GATE=${LIGHTCONE_PHASE1_GATE:-$ROOT/PHASE1_GATE.json}
STATE=${LIGHTCONE_CONTROLLER_STATE:-$ROOT/queue-state.jsonl}
QUEUE_LOCK=${LIGHTCONE_CONTROLLER_QUEUE_LOCK:-$ROOT/.queue.lock}
CUDA_TOOLKIT=${LIGHTCONE_CUDA_TOOLKIT_ROOT:-$RUNTIME_ROOT/cuda-12.9}
PAIR=${LIGHTCONE_CONTROLLER_PAIR:-qwen3_4b_dflash16}
PEAK_TFLOPS=${LIGHTCONE_PEAK_TFLOPS_PER_GPU:-}
HEADLINE_ROOT=${LIGHTCONE_HEADLINE_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-final-0-40k-v10}
HEADLINE_MANIFEST_ROOT=${LIGHTCONE_HEADLINE_MANIFEST_ROOT:-$HEADLINE_ROOT/manifests}
HEADLINE_GENERATION=${LIGHTCONE_HEADLINE_GENERATION:-$HEADLINE_ROOT/FINAL_0_40K_MANIFESTS.json}
HEADLINE_ALGORITHMIC_ROOT=${LIGHTCONE_HEADLINE_ALGORITHMIC_ROOT:-$HEADLINE_ROOT/algorithmic-c4}
HEADLINE_MFU_ROOT=${LIGHTCONE_HEADLINE_MFU_ROOT:-$HEADLINE_ROOT/mfu-context-load}
HEADLINE_ALGORITHMIC_STATIC_ANALYSIS=${LIGHTCONE_HEADLINE_ALGORITHMIC_STATIC_ANALYSIS:-$HEADLINE_ROOT/analysis/algorithmic-vs-static}
HEADLINE_ALGORITHMIC_TTS_ANALYSIS=${LIGHTCONE_HEADLINE_ALGORITHMIC_TTS_ANALYSIS:-$HEADLINE_ROOT/analysis/algorithmic-vs-tts}
HEADLINE_MFU_STATIC_ANALYSIS=${LIGHTCONE_HEADLINE_MFU_STATIC_ANALYSIS:-$HEADLINE_ROOT/analysis/mfu-vs-static}
HEADLINE_MFU_TTS_ANALYSIS=${LIGHTCONE_HEADLINE_MFU_TTS_ANALYSIS:-$HEADLINE_ROOT/analysis/mfu-vs-tts}
HEADLINE_DECISION=${LIGHTCONE_HEADLINE_DECISION:-$HEADLINE_ROOT/FINAL_0_40K_DECISION.json}
HEADLINE_CONFIRMED=${LIGHTCONE_HEADLINE_CONFIRMED:-$HEADLINE_ROOT/FINAL_0_40K_CONFIRMED.json}
HEADLINE_BLOCKED=${LIGHTCONE_HEADLINE_BLOCKED:-$HEADLINE_ROOT/FINAL_0_40K_BLOCKED.json}
HEADLINE_ATTEMPT_ROOT=${LIGHTCONE_HEADLINE_ATTEMPT_ROOT:-$HEADLINE_ROOT/queue-attempt}
QUEUE_SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROCESS_CONTROL=${LIGHTCONE_QUEUE_PROCESS_CONTROL:-$QUEUE_SOURCE_DIR/priority_queue_process.sh}

# shellcheck source=priority_queue_process.sh
source "$PROCESS_CONTROL"

mkdir -p "$ROOT" "$MANIFEST_ROOT" "$TRACE_ROOT" "$HEADLINE_MANIFEST_ROOT"
exec 9>>"$QUEUE_LOCK"
if ! flock -n 9; then
  echo "matched controller queue already owns $QUEUE_LOCK" >&2
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
target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
row = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "phase": phase,
    "status": status,
    "detail": detail,
}
with target.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

run_final_headline() {
  local terminal=$1
  local attempt_status
  attempt_status=$($PY "$QUEUE_HELPER" terminal-status --root "$HEADLINE_ATTEMPT_ROOT")
  case "$attempt_status" in
    failed)
      "$PY" "$QUEUE_HELPER" archive-failure --root "$HEADLINE_ATTEMPT_ROOT"
      record headline_generation resumed_archived_failure \
        "$HEADLINE_ATTEMPT_ROOT/attempts"
      ;;
    none) ;;
    *)
      echo "unexpected headline-attempt status: $attempt_status" >&2
      return 8
      ;;
  esac
  local report_args=()
  if [ -f "$FINAL_CONTROLLER_ROOT/replay_report.json" ]; then
    report_args=(--controller-report "$FINAL_CONTROLLER_ROOT/replay_report.json")
  elif [ -f "$PHASE1_CONTROLLER_ROOT/replay_report.json" ]; then
    report_args=(--controller-report "$PHASE1_CONTROLLER_ROOT/replay_report.json")
  fi
  record headline_generation started "$terminal"
  queue_run_managed "$PY" "$QUEUE_HELPER" build-headline \
    --generation "$GENERATION" \
    --terminal "$terminal" --tts-foundation-terminal "$FOUNDATION_TERMINAL" \
    "${report_args[@]}" \
    --output-dir "$HEADLINE_MANIFEST_ROOT" --output "$HEADLINE_GENERATION"

  local resolved=()
  readarray -t resolved < <("$PY" - "$HEADLINE_GENERATION" "$GENERATION" <<'PY'
import json
import sys
from pathlib import Path

headline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
print(headline["artifacts"]["ALGORITHMIC_C4"]["path"])
print(headline["artifacts"]["MFU_CONTEXT_LOAD"]["path"])
print(headline["optimizer_identity"]["weight_update_mode"])
print(headline.get("controller_root") or "")
print(json.dumps(headline["methods"], separators=(",", ":")))
print(generation["controller_identity"]["pytorch_cuda_alloc_conf"])
PY
  )
  local algorithmic_manifest=${resolved[0]}
  local mfu_manifest=${resolved[1]}
  local mode=${resolved[2]}
  local controller_root=${resolved[3]}
  local methods_json=${resolved[4]}
  local allocator=${resolved[5]}
  local methods=()
  readarray -t methods < <("$PY" - "$methods_json" <<'PY'
import json
import sys
for method in json.loads(sys.argv[1]):
    print(method)
PY
  )
  local peak_args=()
  if [ -n "$PEAK_TFLOPS" ]; then
    peak_args=(--peak-tflops-per-gpu "$PEAK_TFLOPS")
  fi

  local manifest artifact_root coverage static_analysis tts_analysis
  for manifest in "$algorithmic_manifest" "$mfu_manifest"; do
    if [ "$manifest" = "$algorithmic_manifest" ]; then
      artifact_root=$HEADLINE_ALGORITHMIC_ROOT
      coverage=$HEADLINE_ROOT/algorithmic-coverage.json
      static_analysis=$HEADLINE_ALGORITHMIC_STATIC_ANALYSIS
      tts_analysis=$HEADLINE_ALGORITHMIC_TTS_ANALYSIS
    else
      artifact_root=$HEADLINE_MFU_ROOT
      coverage=$HEADLINE_ROOT/mfu-coverage.json
      static_analysis=$HEADLINE_MFU_STATIC_ANALYSIS
      tts_analysis=$HEADLINE_MFU_TTS_ANALYSIS
    fi
    local args=(
      run-manifest --manifest "$manifest" --artifact-root "$artifact_root"
      --lockfile "$LOCKFILE" --runtime-root "$RUNTIME_ROOT"
      --model-roots "$MODEL_ROOTS" --weight-update-mode "$mode"
      "${peak_args[@]}"
    )
    if [ -n "$controller_root" ]; then
      args+=(--controller-root "$controller_root")
    fi
    record headline_execution started "$manifest"
    queue_run_managed env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING -u PYTORCH_ALLOC_CONF \
      PYTORCH_CUDA_ALLOC_CONF="$allocator" CUDA_HOME="$CUDA_TOOLKIT" \
      PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH" \
      LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$LC" "${args[@]}"
    queue_run_managed "$LC" validate-artifacts \
      --artifact-root "$artifact_root" --manifest "$manifest" \
      --methods "${methods[@]}" --weight-update-mode "$mode" \
      --coverage-output "$coverage"
    queue_run_managed "$LC" analyze \
      --artifact-root "$artifact_root" --output-dir "$static_analysis" \
      --manifest "$manifest" --methods "${methods[@]}" \
      --weight-update-mode "$mode" --baseline static
    queue_run_managed "$LC" analyze \
      --artifact-root "$artifact_root" --output-dir "$tts_analysis" \
      --manifest "$manifest" --methods "${methods[@]}" \
      --weight-update-mode "$mode" --baseline tts
    record headline_execution complete "$artifact_root"
  done

  phase=final_0_40k_gate
  local gate_rc
  if queue_run_managed "$PY" "$FINAL_GATE_TOOL" \
      --headline-generation "$HEADLINE_GENERATION" \
      --controller-terminal "$terminal" \
      --tts-foundation-terminal "$FOUNDATION_TERMINAL" \
      --algorithmic-artifact-root "$HEADLINE_ALGORITHMIC_ROOT" \
      --algorithmic-manifest "$algorithmic_manifest" \
      --algorithmic-analysis-root "$HEADLINE_ALGORITHMIC_TTS_ANALYSIS" \
      --mfu-artifact-root "$HEADLINE_MFU_ROOT" \
      --mfu-manifest "$mfu_manifest" \
      --mfu-analysis-root "$HEADLINE_MFU_TTS_ANALYSIS" \
      --output "$HEADLINE_DECISION" --bootstrap-replicates 5000; then
    gate_rc=0
  else
    gate_rc=$?
  fi
  if [ "$gate_rc" -ne 0 ] && [ "$gate_rc" -ne 3 ]; then
    return "$gate_rc"
  fi
  local final_status final_terminal
  final_status=$($PY - "$HEADLINE_DECISION" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["status"])
PY
  )
  if [ "$final_status" = CONFIRMED ]; then
    final_terminal=$HEADLINE_CONFIRMED
  else
    final_terminal=$HEADLINE_BLOCKED
  fi
  # Avoid `python - <<'PY'` under queue_run_managed: the helper already uses
  # stdin for its session-leader wrapper, which would make this a no-op.
  queue_run_managed "$PY" "$QUEUE_SOURCE_DIR/cas_copy_text.py" \
    --source "$HEADLINE_DECISION" --target "$final_terminal"
  record headline complete "$final_terminal"
  if [ "$final_status" != CONFIRMED ]; then
    return 42
  fi
}

phase=bootstrap
failure_armed=1
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 42 ] && [ "$rc" -ne 75 ]; then
    record "$phase" failed_resumable \
      "exit_code=$rc signal=${QUEUE_STOP_SIGNAL:-none}" || true
    if [ "$failure_armed" -eq 1 ]; then
      "$PY" "$QUEUE_HELPER" write-failure \
        --root "$ROOT" --phase "$phase" --return-code "$rc" \
        --evidence "$STATE" --evidence "$FOUNDATION_TERMINAL" \
        --evidence "$GENERATION" --evidence "$PHASE1_GATE" \
        --evidence "$0" || true
    else
      # A controller terminal is immutable and cannot coexist with a failure
      # receipt.  Headline execution therefore owns a separate resumable
      # attempt namespace while reusing the same attested receipt machinery.
      "$PY" "$QUEUE_HELPER" write-failure \
        --root "$HEADLINE_ATTEMPT_ROOT" --phase "$phase" \
        --return-code "$rc" --evidence "$STATE" \
        --evidence "$HEADLINE_GENERATION" --evidence "$0" || true
    fi
  fi
}
trap on_exit EXIT
queue_process_control_init "$PY"

# The TTS-vs-Static foundation is an independent, recursively validated
# scientific prerequisite.  A controller terminal can never bypass it on a
# resumed queue.
ORACLE_SCOPE_FLAG=
if [ "$(basename -- "$SELECTED_SCREEN")" = CANDIDATE_SCREEN_BLOCKED.json ]; then
  ORACLE_SCOPE_FLAG=--allow-l0-not-superior-oracle-scope
fi
phase=tts_foundation_terminal
queue_run_managed "$PY" "$FOUNDATION_TOOL" compare \
  --selected-receipt "$SELECTED_SCREEN" \
  --source-screen-manifest "$SCREEN_MANIFEST" \
  --lockfile "$LOCKFILE" --model-roots "$MODEL_ROOTS" \
  --runtime-fingerprint "$RUNTIME_FINGERPRINT" \
  --artifact-root "$FOUNDATION_ARTIFACT_ROOT" \
  --foundation-manifest "$FOUNDATION_MANIFEST" \
  --generation-receipt "$FOUNDATION_GENERATION" \
  --analysis-root "$FOUNDATION_ANALYSIS_ROOT" \
  --coverage "$FOUNDATION_COVERAGE" --receipt "$FOUNDATION_TERMINAL" \
  ${ORACLE_SCOPE_FLAG:+"$ORACLE_SCOPE_FLAG"}
FOUNDATION_STATUS=$($PY - "$FOUNDATION_TERMINAL" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["status"])
PY
)
if [ "$FOUNDATION_STATUS" != TTS_0_40K_CONFIRMED ]; then
  record "$phase" scientifically_blocked "$FOUNDATION_TERMINAL"
  failure_armed=0
  exit 42
fi
record "$phase" verified "$FOUNDATION_TERMINAL"

status=$($PY "$QUEUE_HELPER" terminal-status --root "$ROOT")
case "$status" in
  selected)
    record terminal skipped_verified_selected "$ROOT/CONTROLLER_SELECTED.json"
    failure_armed=0
    run_final_headline "$ROOT/CONTROLLER_SELECTED.json"
    exit 0
    ;;
  blocked)
    record terminal skipped_verified_blocked "$ROOT/CONTROLLER_BLOCKED.json"
    failure_armed=0
    exit 42
    ;;
  failed)
    "$PY" "$QUEUE_HELPER" archive-failure --root "$ROOT"
    record bootstrap resumed_archived_failure "$ROOT/attempts"
    ;;
  none) ;;
  *) echo "unexpected controller terminal status: $status" >&2; exit 8 ;;
esac

phase=manifest_generation
queue_run_managed "$PY" "$MANIFEST_BUILDER" \
  --selected-receipt "$SELECTED_SCREEN" \
  --lockfile "$LOCKFILE" \
  --model-roots "$MODEL_ROOTS" \
  --output-dir "$MANIFEST_ROOT" \
  --generation-receipt "$GENERATION" \
  ${ORACLE_SCOPE_FLAG:+"$ORACLE_SCOPE_FLAG"}

readarray -t MATCHED < <("$PY" - "$GENERATION" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
artifacts = payload["artifacts"]
print(artifacts["TRACE_MATCHED"]["path"])
print(artifacts["L3_PHASE2_MATCHED"]["path"])
print(artifacts["L3_PHASE2_TTS_REFERENCE"]["path"])
print(payload["controller_identity"]["pytorch_cuda_alloc_conf"])
PY
)
TRACE_MANIFEST=${MATCHED[0]}
L3_MANIFEST=${MATCHED[1]}
L3_TTS_MANIFEST=${MATCHED[2]}
ALLOCATOR=${MATCHED[3]}
record "$phase" generated "$GENERATION"

PEAK_TFLOPS_ARGS=()
if [ -n "$PEAK_TFLOPS" ]; then
  PEAK_TFLOPS_ARGS=(--peak-tflops-per-gpu "$PEAK_TFLOPS")
fi

run_manifest() {
  local manifest=$1 artifact_root=$2 controller_root=${3:-}
  local args=(
    run-manifest --manifest "$manifest" --artifact-root "$artifact_root"
    --lockfile "$LOCKFILE" --runtime-root "$RUNTIME_ROOT"
    --model-roots "$MODEL_ROOTS" --weight-update-mode lora
    "${PEAK_TFLOPS_ARGS[@]}"
  )
  if [ -n "$controller_root" ]; then
    args+=(--controller-root "$controller_root")
  fi
  queue_run_managed env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING -u PYTORCH_ALLOC_CONF \
    PYTORCH_CUDA_ALLOC_CONF="$ALLOCATOR" CUDA_HOME="$CUDA_TOOLKIT" \
    PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH" \
    LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LC" "${args[@]}"
}

phase=phase1_trace
run_manifest "$TRACE_MANIFEST" "$PHASE1_TRACE_ROOT"
queue_run_managed "$LC" validate-artifacts \
  --artifact-root "$PHASE1_TRACE_ROOT" --manifest "$TRACE_MANIFEST" \
  --methods tts naive_async --weight-update-mode lora \
  --coverage-output "$ROOT/phase1-coverage.json"
record "$phase" complete "$PHASE1_TRACE_ROOT"

phase=phase1_controller
mkdir -p "$PHASE1_CONTROLLER_ROOT"
queue_run_managed "$LC" replay --trace-root "$PHASE1_TRACE_ROOT" --pair "$PAIR" \
  --transport-rank 16 --seed 0 --output-dir "$PHASE1_CONTROLLER_ROOT"
PHASE2_ALLOWED=$($PY "$QUEUE_HELPER" phase1-gate \
  --tts-foundation-terminal "$FOUNDATION_TERMINAL" --generation "$GENERATION" \
  --report "$PHASE1_CONTROLLER_ROOT/replay_report.json" \
  --output "$PHASE1_GATE" --queue-source "$0")
record "$phase" gated "phase2_allowed=$PHASE2_ALLOWED"

FINAL_REPORT_ARGS=()
if [ "$PHASE2_ALLOWED" = 1 ]; then
  phase=phase2_l3_evaluation
  run_manifest "$L3_TTS_MANIFEST" "$PHASE2_TRACE_ROOT"
  run_manifest "$L3_MANIFEST" "$PHASE2_TRACE_ROOT" "$PHASE1_CONTROLLER_ROOT"
  queue_run_managed "$LC" validate-artifacts \
    --artifact-root "$PHASE2_TRACE_ROOT" --manifest "$L3_TTS_MANIFEST" \
    --methods tts --weight-update-mode lora \
    --coverage-output "$ROOT/phase2-tts-coverage.json"
  queue_run_managed "$LC" validate-artifacts \
    --artifact-root "$PHASE2_TRACE_ROOT" --manifest "$L3_MANIFEST" \
    --methods lc_transport --weight-update-mode lora \
    --coverage-output "$ROOT/phase2-coverage.json"
  record "$phase" complete "$PHASE2_TRACE_ROOT"

  phase=final_controller
  mkdir -p "$FINAL_CONTROLLER_ROOT"
  queue_run_managed "$LC" replay --trace-root "$TRACE_ROOT" --pair "$PAIR" \
    --transport-rank 16 --seed 0 --output-dir "$FINAL_CONTROLLER_ROOT"
  FINAL_REPORT_ARGS=(--final-report "$FINAL_CONTROLLER_ROOT/replay_report.json")
  record "$phase" complete "$FINAL_CONTROLLER_ROOT/replay_report.json"
else
  record phase2_l3_evaluation scientifically_blocked phase1_readiness_gate
fi

phase=terminal
FINAL_STATUS=$($PY "$QUEUE_HELPER" finalize \
  --root "$ROOT" --tts-foundation-terminal "$FOUNDATION_TERMINAL" \
  --generation "$GENERATION" \
  --phase1-gate "$PHASE1_GATE" --trace-root "$TRACE_ROOT" \
  --queue-source "$0" \
  "${FINAL_REPORT_ARGS[@]}")
if [ "$FINAL_STATUS" = selected ]; then
  record "$phase" selected "$ROOT/CONTROLLER_SELECTED.json"
  failure_armed=0
  run_final_headline "$ROOT/CONTROLLER_SELECTED.json"
  phase=resume_old_ablations
  queue_run_managed env \
    LIGHTCONE_SKIP_LEGACY_PRIORITY=1 \
    LIGHTCONE_RESUME_RECEIPT="$HEADLINE_CONFIRMED" \
    LIGHTCONE_FINAL_GATE_TOOL="$FINAL_GATE_TOOL" \
    "$OLD_ABLATION_QUEUE"
  record "$phase" complete "$HEADLINE_CONFIRMED"
  echo MATCHED_CONTROLLER_SELECTED
  exit 0
fi
record "$phase" scientifically_blocked "$ROOT/CONTROLLER_BLOCKED.json"
failure_armed=0
echo MATCHED_CONTROLLER_BLOCKED
exit 42
