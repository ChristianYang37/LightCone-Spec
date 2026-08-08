#!/usr/bin/env bash
# Resume-safe, credential-free legacy ablation queue.

set -u -o pipefail

QUEUE_SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=${LIGHTCONE_WORKSPACE:-$(cd "$QUEUE_SOURCE_DIR/../.." && pwd)}
RUNTIME_ROOT=${LIGHTCONE_RUNTIME_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lightcone-spec}

QUEUE_ROOT=${LIGHTCONE_QUEUE_ROOT:-$RUNTIME_ROOT/queue/legacy-ablations}
SCRIPTS="$QUEUE_SOURCE_DIR"
STAGE1=${LIGHTCONE_STAGE1_ROOT:-$RUNTIME_ROOT/experiments/optimizer-screen}
STAGE1_SPEC=${LIGHTCONE_STAGE1_SPEC:-$WORKSPACE/scripts/experiments/dflash_tts_schema_v3_optimizer_lr_candidates.json}
STAGE1_BUNDLE_SPEC="$STAGE1/candidate-specification.json"
STAGE1_ANALYSIS="$STAGE1/selection-analysis.json"
STAGE2=${LIGHTCONE_STAGE2_ROOT:-$RUNTIME_ROOT/experiments/rank-screen}
STAGE2_SPEC="$STAGE2/candidate-specification.json"
STAGE2_ANALYSIS="$STAGE2/rank-analysis.json"
DIAGNOSTIC=${LIGHTCONE_DIAGNOSTIC_ROOT:-$RUNTIME_ROOT/experiments/diagnostics}
DIAGNOSTIC_SPEC="$DIAGNOSTIC/candidate-specification.json"
LONG_CONTEXT=${LIGHTCONE_LONG_CONTEXT_ROOT:-$RUNTIME_ROOT/experiments/long-context}
LONG_CONTEXT_SAMPLE=${LIGHTCONE_LONG_CONTEXT_SAMPLE:-1}
LONG_CONTEXTS=${LIGHTCONE_LONG_CONTEXTS:-"8192 16384 32768 40960"}
LOAD_TUNE_MANIFEST=${LIGHTCONE_LOAD_TUNE_MANIFEST:-$WORKSPACE/manifests/load_tune/load_tune_gpu_qwen3_4b.json}
LOAD_TUNE_LOCKFILE=${LIGHTCONE_LOAD_TUNE_LOCKFILE:-$RUNTIME_ROOT/lightcone.lock.json}
LOAD_TUNE_ROOT=${LIGHTCONE_LOAD_TUNE_ROOT:-$RUNTIME_ROOT/runs/load-tune}
LOAD_TUNE_ANALYSIS=${LIGHTCONE_LOAD_TUNE_ANALYSIS:-$RUNTIME_ROOT/analysis/load-tune}
PEAK_TFLOPS_PER_GPU=${LIGHTCONE_PEAK_TFLOPS_PER_GPU:-}
P5_MANIFEST=${LIGHTCONE_P5_MANIFEST:-$WORKSPACE/manifests/p5/p5_long_context_acceptance_engine_reuse.json}
P5_LOCKFILE=${LIGHTCONE_P5_LOCKFILE:-$RUNTIME_ROOT/p5-formal.lock.json}
P5_ROOT=${LIGHTCONE_P5_ROOT:-$RUNTIME_ROOT/runs/p5-residual}
P5_ANALYSIS=${LIGHTCONE_P5_ANALYSIS:-$RUNTIME_ROOT/analysis/p5-residual}
P5_TTS_ANALYSIS_BASE=${LIGHTCONE_P5_TTS_ANALYSIS_BASE:-$RUNTIME_ROOT/analysis/p5-tts-mode}
# A compact, evidence-first DFlash lane runs before the historical queue.  Its
# roots are deliberately disjoint from q00--q07 so a failed priority attempt
# cannot make old artifacts appear complete (or vice versa).
PRIORITY_MANIFEST=${LIGHTCONE_PRIORITY_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_0_40k_v1.json}
PRIORITY_CALIBRATION_MANIFEST=${LIGHTCONE_PRIORITY_CALIBRATION_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_calibration_v1.json}
PRIORITY_TRACE_MANIFEST=${LIGHTCONE_PRIORITY_TRACE_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_paired_trace_v1.json}
PRIORITY_L3_EVALUATION_MANIFEST=${LIGHTCONE_PRIORITY_L3_EVALUATION_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_l3_evaluation_v1.json}
PRIORITY_SMOKE_MANIFEST=${LIGHTCONE_PRIORITY_SMOKE_MANIFEST:-$WORKSPACE/manifests/p5/p5_priority_dflash_smoke_v1.json}
PRIORITY_LOCKFILE=${LIGHTCONE_PRIORITY_LOCKFILE:-$RUNTIME_ROOT/priority/p5-dflash4b-v1.lock.json}
PRIORITY_MODEL_ROOTS=${LIGHTCONE_PRIORITY_MODEL_ROOTS:-$RUNTIME_ROOT/priority/p5-dflash4b-v1.model-roots.json}
PRIORITY_MODEL_CACHE=${LIGHTCONE_PRIORITY_MODEL_CACHE:-$RUNTIME_ROOT/priority/model-cache}
PRIORITY_SMOKE_ROOT=${LIGHTCONE_PRIORITY_SMOKE_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-v1/smoke}
PRIORITY_CALIBRATION_ROOT=${LIGHTCONE_PRIORITY_CALIBRATION_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-calibration-v1}
PRIORITY_CALIBRATION_SCRIPT=${LIGHTCONE_PRIORITY_CALIBRATION_SCRIPT:-$WORKSPACE/scripts/experiments/run_priority_dflash_calibration_queue.py}
PRIORITY_CALIBRATION_SPEC=${LIGHTCONE_PRIORITY_CALIBRATION_SPEC:-$WORKSPACE/scripts/experiments/priority_calibration_candidates_v1.json}
PRIORITY_CALIBRATION_READY=${LIGHTCONE_PRIORITY_CALIBRATION_READY:-$PRIORITY_CALIBRATION_ROOT/DOWNSTREAM_READY.json}
PRIORITY_EVAL_ROOT=${LIGHTCONE_PRIORITY_EVAL_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-v1/eval}
PRIORITY_TRACE_ROOT=${LIGHTCONE_PRIORITY_TRACE_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-v1/trace}
PRIORITY_PHASE1_TRACE_ROOT=${LIGHTCONE_PRIORITY_PHASE1_TRACE_ROOT:-$PRIORITY_TRACE_ROOT/phase1}
PRIORITY_L3_EVALUATION_TRACE_ROOT=${LIGHTCONE_PRIORITY_L3_EVALUATION_TRACE_ROOT:-$PRIORITY_TRACE_ROOT/l3-evaluation}
PRIORITY_PHASE1_CONTROLLER_ROOT=${LIGHTCONE_PRIORITY_PHASE1_CONTROLLER_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-v1/controller-phase1}
PRIORITY_CONTROLLER_ROOT=${LIGHTCONE_PRIORITY_CONTROLLER_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-v1/controller}
PRIORITY_ANALYSIS_ROOT=${LIGHTCONE_PRIORITY_ANALYSIS_ROOT:-$RUNTIME_ROOT/priority/p5-dflash4b-v1/analysis}
PRIORITY_DATASET_RECEIPT=${LIGHTCONE_PRIORITY_DATASET_RECEIPT:-$RUNTIME_ROOT/priority/p5-dflash4b-v1.inputs/dataset-preflight.json}
PRIORITY_ARTIFACT_CONTRACT=${LIGHTCONE_PRIORITY_ARTIFACT_CONTRACT:-$RUNTIME_ROOT/priority/p5-dflash4b-v1.artifact-contract.json}
PRIORITY_WORKSPACE_ROOT=${LIGHTCONE_PRIORITY_WORKSPACE_ROOT:-$WORKSPACE}
PRIORITY_PAIR=qwen3_4b_dflash16
PRIORITY_MODE=${LIGHTCONE_PRIORITY_MODE:-lora}
PRIORITY_LR=${LIGHTCONE_PRIORITY_LR:-0.00003}
PRIORITY_SCIENTIFIC_BLOCKED_RC=20
# Keep the reviewed critical deployment surface explicit.  The full runtime
# trees are Merkle-bound below as a backstop because the generic runtime
# fingerprint intentionally covers a smaller stable API surface.
PRIORITY_PRODUCTION_FILES=(
  src/lightcone_spec/adapters/adapter_params.py
  src/lightcone_spec/adapters/losses.py
  src/lightcone_spec/methods/base.py
  src/lightcone_spec/methods/registry.py
  src/lightcone_spec/replay/real.py
  src/lightcone_spec/config/schema.py
  src/lightcone_spec/orchestration/runtime_config.py
  src/lightcone_spec/orchestration/catalog.py
  src/lightcone_spec/sglang_bridge/client.py
  src/lightcone_spec/sglang_bridge/hooks.py
  src/lightcone_spec/sglang_bridge/runtime.py
  src/lightcone_spec/sglang_bridge/static_observer.py
  src/lightcone_spec/sglang_bridge/telemetry.py
  src/lightcone_spec/artifacts/schemas.py
  src/lightcone_spec/statistics/tables.py
  src/lightcone_spec/cli/main.py
  src/lightcone_spec/runtime/engine.py
  sglang/python/sglang/srt/speculative/dflash_info_v2.py
  sglang/python/sglang/srt/speculative/dflash_worker_v2.py
  sglang/python/sglang/srt/speculative/dspark_components/dspark_adaptation.py
  sglang/python/sglang/srt/speculative/eagle_worker_v2.py
  sglang/python/sglang/srt/speculative/tail_adaptation.py
)
PY=${LIGHTCONE_PYTHON:-$RUNTIME_ROOT/venv/bin/python}
LC=${LIGHTCONE_CLI:-$RUNTIME_ROOT/venv/bin/lightcone-spec}
FINAL_GATE_TOOL=${LIGHTCONE_FINAL_GATE_TOOL:-$WORKSPACE/scripts/experiments/p5_final_headline_gate.py}
SKIP_LEGACY_PRIORITY=${LIGHTCONE_SKIP_LEGACY_PRIORITY:-0}
RESUME_RECEIPT=${LIGHTCONE_RESUME_RECEIPT:-}
CUDA_TOOLKIT_ROOT=${LIGHTCONE_CUDA_TOOLKIT_ROOT:-$RUNTIME_ROOT/cuda-12.9}
PY_BIN_DIR=$(dirname -- "$PY")
PEAK_TFLOPS_ARGS=()
if [ -n "$PEAK_TFLOPS_PER_GPU" ]; then
  PEAK_TFLOPS_ARGS=(--peak-tflops-per-gpu "$PEAK_TFLOPS_PER_GPU")
fi

# q00--q04 intentionally inherit deterministic CUDA settings for reference
# calibration.  Headline SGLang runs must not inherit debug/determinism knobs
# that constrain GEMM selection or serialize kernel launches.
run_sglang_headline() {
  env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING \
    CUDA_HOME="$CUDA_HOME" PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    "$LC" "$@"
}
HARNESS="$SCRIPTS/dflash_tts_reference.py"
REFERENCE_ROOT=${LIGHTCONE_REFERENCE_ROOT:-$RUNTIME_ROOT/reference/dflash}
TARGET=${LIGHTCONE_TARGET_MODEL:-$RUNTIME_ROOT/models/target}
DRAFT=${LIGHTCONE_DRAFT_MODEL:-$RUNTIME_ROOT/models/drafter}
DATASET=${LIGHTCONE_DATASET:-$RUNTIME_ROOT/data/evaluation.jsonl}
PROJECTION=${LIGHTCONE_PROJECTION:-$RUNTIME_ROOT/projections/output-residual.npz}

# The remote host already contains schema-v3 runs produced before canonical
# q_len=1 target verification was added.  Those artifacts are valuable failure
# evidence, but they are not resumable inputs for the corrected harness.  Fail
# before publishing a queue run (and before using any GPU) if an operator points
# this queue back at one of the known legacy roots.
reject_legacy_root() {
  local candidate=${1%/}
  local label=$2
  while [ "${candidate%/}" != "$candidate" ]; do
    candidate=${candidate%/}
  done
  case "$candidate" in
    $RUNTIME_ROOT/reference/tts-dflash/runs/calibration-schema-v4-canonical-exact-stage1)
      echo "$label points at historical runner-unbound evidence: $candidate" >&2
      echo "select the runner-bound Stage-1 root; do not migrate old artifacts" >&2
      exit 2
      ;;
    $RUNTIME_ROOT/reference/tts-dflash/runs/calibration-schema-v3-optlr-stage1|\
    $RUNTIME_ROOT/reference/tts-dflash/runs/calibration-schema-v3-lora-rank-stage2|\
    $RUNTIME_ROOT/reference/tts-dflash/runs/calibration-schema-v3-diagnostics)
      echo "$label points at pre-canonical schema-v3 evidence: $candidate" >&2
      echo "select a fresh schema-v4-canonical-exact root" >&2
      exit 2
      ;;
  esac
}

reject_legacy_root "$STAGE1" LIGHTCONE_STAGE1_ROOT
reject_legacy_root "$STAGE2" LIGHTCONE_STAGE2_ROOT
reject_legacy_root "$DIAGNOSTIC" LIGHTCONE_DIAGNOSTIC_ROOT

[ -x "$CUDA_TOOLKIT_ROOT/bin/nvcc" ] || {
  echo "locked CUDA toolkit is missing nvcc: $CUDA_TOOLKIT_ROOT" >&2
  exit 2
}
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export PATH="$PY_BIN_DIR:$CUDA_TOOLKIT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

command -v ninja >/dev/null 2>&1 || {
  echo "ninja is required for FlashInfer JIT; install it in $PY_BIN_DIR" >&2
  exit 2
}

mkdir -p "$QUEUE_ROOT" || exit 2

usage() {
  echo "usage: $0 [--run-id ID | --resume ID]" >&2
  exit 2
}

resume=0
run_id=${LIGHTCONE_QUEUE_RUN_ID:-}
mode_seen=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-id)
      [ "$#" -ge 2 ] || usage
      [ "$mode_seen" -eq 0 ] || usage
      run_id=$2
      mode_seen=1
      shift 2
      ;;
    --resume)
      [ "$#" -ge 2 ] || usage
      [ "$mode_seen" -eq 0 ] || usage
      run_id=$2
      resume=1
      mode_seen=1
      shift 2
      ;;
    *) usage ;;
  esac
done
if [ -z "$run_id" ]; then
  run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
case "$run_id" in
  *[!A-Za-z0-9._-]*|'')
    echo "invalid queue run id: $run_id" >&2
    exit 2
    ;;
esac

# A queue is singleton per queue root.  Keep the lock in the outer ``flock``
# process and close its descriptor in the queue child: model servers or other
# grandchildren must not accidentally keep the queue locked after this script
# exits.  Normalize the generated run id across the re-exec boundary.
lock_identity="$QUEUE_ROOT:$run_id"
if [ "${LIGHTCONE_QUEUE_LOCK_HELD:-}" != "$lock_identity" ]; then
  if [ "$resume" -eq 1 ]; then
    lock_args=(--resume "$run_id")
  else
    lock_args=(--run-id "$run_id")
  fi
  env LIGHTCONE_QUEUE_LOCK_HELD="$lock_identity" \
    flock -n -E 10 -o "$QUEUE_ROOT/.queue.lock" \
    bash "$0" "${lock_args[@]}"
  rc=$?
  if [ "$rc" -eq 10 ]; then
    echo "another queue owns $QUEUE_ROOT/.queue.lock" >&2
  fi
  exit "$rc"
fi
unset LIGHTCONE_QUEUE_LOCK_HELD

path_exists() {
  [ -e "$1" ] || [ -L "$1" ]
}

new_archive_dir() {
  local archive_parent=$1
  local archive_prefix=$2
  mkdir -p "$archive_parent" || return
  mktemp -d "$archive_parent/${archive_prefix}.XXXXXX"
}

attempt_payload_matches() {
  local path=$1
  local generation=$2
  [ -f "$path" ] && [ "$(cat "$path")" = "$run_id"$'\t'"$generation" ]
}

marker_owned_by_attempt() {
  local marker_path=$1
  local generation=$2
  attempt_payload_matches "$marker_path" "$generation"
}

SESSION_ROOT="$QUEUE_ROOT/sessions/$run_id"
CURRENT_RUN="$QUEUE_ROOT/CURRENT_RUN"
CURRENT_ATTEMPT="$QUEUE_ROOT/CURRENT_ATTEMPT"
ATTEMPT_GENERATION_FILE="$SESSION_ROOT/attempt-generation"
if [ "$resume" -eq 1 ]; then
  [ -d "$SESSION_ROOT" ] || {
    echo "cannot resume unknown queue run: $run_id" >&2
    exit 2
  }
  [ -f "$CURRENT_RUN" ] && [ "$(cat "$CURRENT_RUN")" = "$run_id" ] || {
    echo "cannot resume superseded queue run: $run_id" >&2
    exit 2
  }
  previous_attempt_run=
  previous_attempt_generation=
  previous_attempt_extra=
  if [ -f "$CURRENT_ATTEMPT" ]; then
    IFS=$'\t' read -r previous_attempt_run previous_attempt_generation \
      previous_attempt_extra < "$CURRENT_ATTEMPT"
  fi
  case "$previous_attempt_generation" in
    *[!0-9]*|''|0)
      echo "invalid previous attempt generation: $previous_attempt_generation" >&2
      exit 2
      ;;
  esac
  if [ "$previous_attempt_run" != "$run_id" ] || \
      [ -n "$previous_attempt_extra" ]; then
    echo "cannot resume superseded queue attempt: $run_id/$previous_attempt_generation" >&2
    exit 2
  fi
  if path_exists "$QUEUE_ROOT/QUEUE_COMPLETE"; then
    marker_owned_by_attempt \
      "$QUEUE_ROOT/QUEUE_COMPLETE" "$previous_attempt_generation" || {
      echo "QUEUE_COMPLETE is owned by a different or invalid run" >&2
      exit 2
    }
    echo "queue run $run_id is already complete; start a new run id" >&2
    exit 2
  fi
  attempt_history=
  for stale in QUEUE_FINISHED QUEUE_FAILED; do
    if path_exists "$QUEUE_ROOT/$stale"; then
      marker_owned_by_attempt \
        "$QUEUE_ROOT/$stale" "$previous_attempt_generation" || {
        echo "$stale is owned by a different or invalid run" >&2
        exit 2
      }
    fi
  done
  for stale in QUEUE_FINISHED QUEUE_FAILED; do
    if path_exists "$QUEUE_ROOT/$stale"; then
      if [ -z "$attempt_history" ]; then
        attempt_history=$(new_archive_dir \
          "$SESSION_ROOT/attempt-history" "$(date -u +%Y%m%dT%H%M%SZ)") || exit 2
      fi
      mv "$QUEUE_ROOT/$stale" "$attempt_history/$stale" || exit 2
    fi
  done
  attempt_generation=$((previous_attempt_generation + 1))
else
  ! path_exists "$SESSION_ROOT" || {
    echo "queue run id already exists; use --resume $run_id" >&2
    exit 2
  }
  mkdir -p "$SESSION_ROOT" || exit 2
  prior_run=unknown
  [ ! -f "$CURRENT_RUN" ] || prior_run=$(cat "$CURRENT_RUN")
  case "$prior_run" in
    *[!A-Za-z0-9._-]*|'') prior_run=invalid ;;
  esac
  if path_exists "$CURRENT_RUN" || path_exists "$QUEUE_ROOT/QUEUE_FINISHED" || \
      path_exists "$CURRENT_ATTEMPT" || \
      path_exists "$QUEUE_ROOT/QUEUE_COMPLETE" || \
      path_exists "$QUEUE_ROOT/QUEUE_FAILED" || \
      path_exists "$QUEUE_ROOT/queue-state.tsv"; then
    history=$(new_archive_dir "$QUEUE_ROOT/history" \
      "$(date -u +%Y%m%dT%H%M%SZ)-$prior_run") || exit 2
    for stale in CURRENT_RUN CURRENT_ATTEMPT QUEUE_FINISHED QUEUE_COMPLETE QUEUE_FAILED queue-state.tsv; do
      path_exists "$QUEUE_ROOT/$stale" || continue
      mv "$QUEUE_ROOT/$stale" "$history/$stale" || exit 2
    done
  fi
  current_tmp="$QUEUE_ROOT/.CURRENT_RUN.$$"
  printf '%s\n' "$run_id" > "$current_tmp" || exit 2
  mv "$current_tmp" "$CURRENT_RUN" || exit 2
  attempt_generation=1
fi

ATTEMPT_ROOT="$SESSION_ROOT/attempts/$attempt_generation"
mkdir -p "$ATTEMPT_ROOT" || exit 2
current_attempt_tmp="$QUEUE_ROOT/.CURRENT_ATTEMPT.$$"
printf '%s\t%s\n' "$run_id" "$attempt_generation" > "$current_attempt_tmp" || exit 2
mv "$current_attempt_tmp" "$CURRENT_ATTEMPT" || exit 2
attempt_generation_tmp="$SESSION_ROOT/.attempt-generation.$$"
printf '%s\n' "$attempt_generation" > "$attempt_generation_tmp" || exit 2
mv "$attempt_generation_tmp" "$ATTEMPT_GENERATION_FILE" || exit 2

STATE="$SESSION_ROOT/queue-state.tsv"
touch "$STATE" || exit 2

# Keep the historical root-level path as an atomic pointer to the active
# session.  Existing regular files or stale links are preserved on fresh-run
# archival above; a resume fails closed instead of overwriting an unexpected
# object.
state_link="$QUEUE_ROOT/queue-state.tsv"
state_target="$STATE"
if path_exists "$state_link"; then
  if [ ! -L "$state_link" ] || [ "$(readlink "$state_link")" != "$state_target" ]; then
    echo "active queue-state.tsv does not point at run $run_id" >&2
    exit 2
  fi
else
  state_link_tmp="$QUEUE_ROOT/.queue-state.$$"
  ln -s "$state_target" "$state_link_tmp" || exit 2
  mv "$state_link_tmp" "$state_link" || exit 2
fi

write_marker() {
  local marker=$1
  local marker_tmp
  [ -f "$CURRENT_RUN" ] && [ "$(cat "$CURRENT_RUN")" = "$run_id" ] || {
    echo "refusing to publish $marker for a superseded queue run" >&2
    return 2
  }
  attempt_payload_matches "$CURRENT_ATTEMPT" "$attempt_generation" || {
    echo "refusing to publish $marker for a superseded queue attempt" >&2
    return 2
  }
  if path_exists "$QUEUE_ROOT/$marker"; then
    echo "refusing to overwrite unexpected terminal marker $marker" >&2
    return 2
  fi
  marker_tmp="$QUEUE_ROOT/.${marker}.$$"
  printf '%s\t%s\n' "$run_id" "$attempt_generation" > "$marker_tmp" || return
  mv "$marker_tmp" "$QUEUE_ROOT/$marker"
}

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" >> "$STATE" || {
    echo "cannot append queue state: $STATE" >&2
    exit 2
  }
}

run_task() {
  local task_id=$1
  local rc
  shift
  record "$task_id" started
  "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    record "$task_id" complete
  else
    record "$task_id" failed "exit=$rc"
  fi
  return "$rc"
}

heartbeat_interval=${LIGHTCONE_QUEUE_HEARTBEAT_SECONDS:-30}
case "$heartbeat_interval" in
  *[!0-9]*|''|0)
    echo "LIGHTCONE_QUEUE_HEARTBEAT_SECONDS must be a positive integer" >&2
    exit 2
    ;;
esac
queue_pid=$$
queue_pid_tmp="$ATTEMPT_ROOT/.queue.pid.$$"
printf '%s\t%s\t%s\n' \
  "$run_id" "$attempt_generation" "$queue_pid" > "$queue_pid_tmp" || exit 2
mv "$queue_pid_tmp" "$ATTEMPT_ROOT/queue.pid" || exit 2

heartbeat_once() {
  local heartbeat_tmp="$ATTEMPT_ROOT/.heartbeat.$$"
  printf '%s\t%s\t%s\t%s\n' \
    "$run_id" "$attempt_generation" "$queue_pid" "$(date +%s)" \
    > "$heartbeat_tmp" || return
  mv "$heartbeat_tmp" "$ATTEMPT_ROOT/heartbeat"
}

heartbeat_worker() {
  trap 'exit 0' HUP INT TERM
  while kill -0 "$queue_pid" 2>/dev/null; do
    heartbeat_once || exit 1
    sleep "$heartbeat_interval" &
    wait $!
  done
}

heartbeat_once || exit 2
heartbeat_worker &
heartbeat_pid=$!

queue_exit_trap() {
  local rc=$1
  trap - EXIT HUP INT TERM
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  if attempt_payload_matches "$CURRENT_ATTEMPT" "$attempt_generation"; then
    if ! marker_owned_by_attempt "$QUEUE_ROOT/QUEUE_COMPLETE" "$attempt_generation" && \
        ! marker_owned_by_attempt "$QUEUE_ROOT/QUEUE_FAILED" "$attempt_generation"; then
      printf '%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" queue failed \
        "unexpected_exit=$rc;attempt=$attempt_generation" >> "$STATE" || true
      write_marker QUEUE_FAILED || true
    fi
    if ! marker_owned_by_attempt "$QUEUE_ROOT/QUEUE_FINISHED" "$attempt_generation"; then
      write_marker QUEUE_FINISHED || true
    fi
  fi
  exit "$rc"
}

trap 'queue_exit_trap "$?"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
record queue attempt_started "attempt=$attempt_generation;pid=$queue_pid"

run_stage1() {
  local -a args=()
  mkdir -p "$STAGE1"
  if [ -e "$STAGE1_BUNDLE_SPEC" ]; then
    cmp --silent "$STAGE1_SPEC" "$STAGE1_BUNDLE_SPEC" || return 4
  else
    cp --no-clobber "$STAGE1_SPEC" "$STAGE1_BUNDLE_SPEC" || return 5
  fi
  mapfile -t args < <(common_calibration_args)
  append_projection_arg_if_required "$STAGE1_BUNDLE_SPEC" || return
  "$PY" "$SCRIPTS/run_dflash_tts_calibration_sweep.py" \
    --candidate-spec "$STAGE1_BUNDLE_SPEC" --output-root "$STAGE1" "${args[@]}"
}

publish_stage1_analysis() {
  if [ -e "$STAGE1_BUNDLE_SPEC" ]; then
    cmp --silent "$STAGE1_SPEC" "$STAGE1_BUNDLE_SPEC" || return 4
  else
    cp --no-clobber "$STAGE1_SPEC" "$STAGE1_BUNDLE_SPEC" || return 5
  fi
  if [ -e "$STAGE1_ANALYSIS" ]; then
    "$PY" "$SCRIPTS/analyze_dflash_tts_calibration.py" \
      --candidate-spec "$STAGE1_BUNDLE_SPEC" --output-root "$STAGE1" \
      --check "$STAGE1_ANALYSIS"
  else
    "$PY" "$SCRIPTS/analyze_dflash_tts_calibration.py" \
      --candidate-spec "$STAGE1_BUNDLE_SPEC" --output-root "$STAGE1" \
      --output "$STAGE1_ANALYSIS"
  fi
}

common_calibration_args() {
  printf '%s\n' \
    --python "$PY" \
    --harness "$HARNESS" \
    --reference-root "$REFERENCE_ROOT" \
    --reference-module dflash.model \
    --reference-revision 94e4abc \
    --target-model "$TARGET" \
    --target-revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --draft-model "$DRAFT" \
    --draft-revision b74e3a329c4d963783143b1e970d95b002be72bd \
    --dataset "$DATASET" \
    --dataset-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
    --pythonpath "$QUEUE_ROOT" \
    --pythonpath $RUNTIME_ROOT/reference/tts-dflash/site-packages \
    --pythonpath "$REFERENCE_ROOT" \
    --mask-token-id 151669 \
    --deterministic \
    --parity-max-new-tokens 128
}

spec_uses_output_residual() {
  "$PY" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
candidates = payload.get("candidates")
if not isinstance(candidates, list):
    raise ValueError("candidate specification lacks a candidates list")
raise SystemExit(0 if any(
    isinstance(row, dict) and row.get("mode") == "output-residual"
    for row in candidates
) else 1)
' "$1"
}

append_projection_arg_if_required() {
  local spec=$1
  local status
  if spec_uses_output_residual "$spec"; then
    args+=(--projection-artifact "$PROJECTION")
    return 0
  else
    status=$?
  fi
  # Exit 1 means the valid spec has no residual candidate.  Parse/schema
  # failures remain fatal instead of silently dropping a required artifact.
  [ "$status" -eq 1 ] || return "$status"
}

run_stage2() {
  local -a args=()
  mkdir -p "$STAGE2"
  if [ -e "$STAGE2_SPEC" ]; then
    "$PY" "$SCRIPTS/build_dflash_tts_lora_rank_candidates.py" \
      --stage1-analysis "$STAGE1_ANALYSIS" --check "$STAGE2_SPEC" || return
  else
    "$PY" "$SCRIPTS/build_dflash_tts_lora_rank_candidates.py" \
      --stage1-analysis "$STAGE1_ANALYSIS" --output "$STAGE2_SPEC" || return
  fi
  mapfile -t args < <(common_calibration_args)
  "$PY" "$SCRIPTS/run_dflash_tts_calibration_sweep.py" \
    --candidate-spec "$STAGE2_SPEC" --output-root "$STAGE2" "${args[@]}" || return
  if [ -e "$STAGE2_ANALYSIS" ]; then
    "$PY" "$SCRIPTS/analyze_dflash_tts_lora_rank.py" \
      --candidate-spec "$STAGE2_SPEC" --output-root "$STAGE2" \
      --check "$STAGE2_ANALYSIS"
  else
    "$PY" "$SCRIPTS/analyze_dflash_tts_lora_rank.py" \
      --candidate-spec "$STAGE2_SPEC" --output-root "$STAGE2" \
      --output "$STAGE2_ANALYSIS"
  fi
}

run_diagnostics() {
  local -a args=()
  mkdir -p "$DIAGNOSTIC"
  if [ -e "$DIAGNOSTIC_SPEC" ]; then
    "$PY" "$SCRIPTS/build_dflash_tts_diagnostic_spec.py" \
      --stage1-analysis "$STAGE1_ANALYSIS" \
      --diagnostic-output-root "$DIAGNOSTIC" --check "$DIAGNOSTIC_SPEC" || return
  else
    "$PY" "$SCRIPTS/build_dflash_tts_diagnostic_spec.py" \
      --stage1-analysis "$STAGE1_ANALYSIS" \
      --diagnostic-output-root "$DIAGNOSTIC" --output "$DIAGNOSTIC_SPEC" || return
  fi
  mapfile -t args < <(common_calibration_args)
  append_projection_arg_if_required "$DIAGNOSTIC_SPEC" || return
  "$PY" "$SCRIPTS/run_dflash_tts_calibration_sweep.py" \
    --candidate-spec "$DIAGNOSTIC_SPEC" --output-root "$DIAGNOSTIC" \
    --keep-going "${args[@]}" || return

  # Enumerate the exact paths declared by the immutable spec.  Counting any N
  # artifact directories could otherwise substitute a failed/quarantined run
  # for a missing required candidate.
  expected_list="$SESSION_ROOT/.diagnostic-artifacts.$$"
  if ! "$PY" -c \
    'import json,pathlib,sys
p=json.load(open(sys.argv[1], encoding="utf-8")); root=pathlib.Path(sys.argv[2])
for sample in p["samples"]:
    for candidate in p["candidates"]:
        print(root / ("sample-%04d" % int(sample["sample_index"])) / candidate["candidate_id"] / "artifact")' \
      "$DIAGNOSTIC_SPEC" "$DIAGNOSTIC" > "$expected_list"; then
    rm -f "$expected_list"
    return 6
  fi
  mapfile -t expected_paths < "$expected_list"
  rm -f "$expected_list"
  artifacts=()
  while IFS= read -r -d '' artifact; do
    artifacts+=("$artifact")
  done < <(find "$DIAGNOSTIC" -mindepth 3 -maxdepth 3 -type d -name artifact -print0)
  expected_artifacts=${#expected_paths[@]}
  if [ "${#artifacts[@]}" -ne "$expected_artifacts" ]; then
    record q03_cache_and_drift_diagnostics incomplete \
      "artifacts=${#artifacts[@]}/$expected_artifacts"
    return 6
  fi
  for artifact in "${expected_paths[@]}"; do
    if [ ! -s "$artifact/summary.json" ] || \
        [ ! -s "$artifact/rounds.jsonl" ] || \
        [ ! -s "${artifact%/artifact}/completion.json" ] || \
        [ ! -s "${artifact%/artifact}/run_identity.json" ]; then
      record q03_cache_and_drift_diagnostics incomplete \
        "missing_completion_evidence=$artifact"
      return 6
    fi
  done
  artifacts=("${expected_paths[@]}")
  "$PY" "$SCRIPTS/aggregate_dflash_tts_ablations.py" \
    --output-dir "$DIAGNOSTIC/aggregate" --bucket-size 256 --parquet \
    "${artifacts[@]}"
}

dflash_preflight_complete() {
  local summary=$1
  local rounds=$2
  [ -s "$summary" ] && [ -s "$rounds" ] || return 1
  "$PY" - "$SCRIPTS" "$summary" "$rounds" "$HARNESS" \
    "$REFERENCE_ROOT" "$TARGET" "$DRAFT" "$DATASET" \
    "$LONG_CONTEXT_SAMPLE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

scripts, summary_name, rounds_name, harness_name, reference_name, \
    target_name, draft_name, dataset_name, sample_text = sys.argv[1:]
sys.path.insert(0, scripts)
import run_dflash_tts_frozen_sweep as frozen


def validate() -> None:
    summary_path = Path(summary_name).resolve()
    rounds_path = Path(rounds_name).resolve()
    harness = Path(harness_name).resolve()
    reference_root = Path(reference_name).resolve()
    target = Path(target_name).resolve()
    draft = Path(draft_name).resolve()
    dataset = Path(dataset_name).resolve()
    sample_index = int(sample_text)
    summary = frozen._read_json(summary_path)
    reference_source = reference_root / "dflash" / "model.py"
    harness_sha256 = frozen._sha256_file(harness)
    reference_sha256 = frozen._sha256_file(reference_source)
    dataset_sha256 = frozen._sha256_file(dataset)
    frozen._check_fields(
        summary,
        (
            ("schema_version", frozen.HARNESS_ARTIFACT_SCHEMA_VERSION),
            ("status", "complete_reference_run"),
            ("method", "static_dflash"),
            ("mode", "static"),
            ("trainable_scope", "none_static"),
            (
                "run_attestation",
                {
                    "status": "direct_unbound",
                    "scheme": None,
                    "run_identity_sha256": None,
                    "command_sha256": None,
                },
            ),
            (
                "artifact_identity",
                {
                    "verification_status": "fully_verified_content_sha256_v1",
                    "lock": {
                        "path": None,
                        "sha256": None,
                        "verification": "direct_content_sha256_v1",
                    },
                },
            ),
            ("harness.source_sha256", harness_sha256),
            ("reference.declared_revision", "94e4abc"),
            ("reference.source_path", str(reference_source)),
            ("reference.source_sha256", reference_sha256),
            ("reference.official_static_parity.status", "passed"),
            (
                "reference.official_static_parity.classification",
                "official_stale_cache_block_verifier_reconstruction",
            ),
            ("reference.official_static_parity.official_policy", "stale"),
            ("reference.official_static_parity.max_new_tokens", 1),
            ("reference.official_static_parity.policies.stale.output_ids_match", True),
            (
                "reference.official_static_parity.policies.stale.acceptance_lengths_match",
                True,
            ),
            ("models.target.declared_revision", "1cfa9a7208912126459214e8b04321603b3df60c"),
            ("models.draft.declared_revision", "b74e3a329c4d963783143b1e970d95b002be72bd"),
            ("dataset.path", str(dataset)),
            ("dataset.sha256", dataset_sha256),
            ("dataset.declared_revision", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
            ("dataset.sample_index", sample_index),
            ("parameters.seed", 0),
            ("parameters.deterministic", True),
            ("parameters.canonical_greedy_verifier", True),
            ("parameters.max_new_tokens", 1),
            ("parameters.block_size", 16),
            ("parameters.mask_token_id", 151669),
            ("parameters.lr", 0.0),
            ("parameters.proximal_lambda", 0.0),
            ("parameters.update_stride", 1),
            ("parameters.position_weighting", "uniform"),
            ("parameters.loss_reduction", "sum"),
            ("parameters.draft_cache_policy", "stale"),
            ("parameters.enable_thinking", True),
            ("parameters.parity_max_new_tokens", 1),
            ("generation.mode", "static"),
            ("generation.num_output_tokens", 1),
            ("generation.optimizer_steps", 0),
            ("generation.final_parameter_version", 0),
            ("generation.exactness.selection_eligible", True),
            ("output.rounds_jsonl", "rounds.jsonl"),
        ),
        label="DFlash direct preflight",
    )

    for role, root, revision in (
        ("target", target, "1cfa9a7208912126459214e8b04321603b3df60c"),
        ("draft", draft, "b74e3a329c4d963783143b1e970d95b002be72bd"),
    ):
        expected = frozen._model_identity(root, revision)
        observed = frozen._get(summary, f"models.{role}")
        if not isinstance(observed, dict):
            raise ValueError(f"preflight models.{role} must be an object")
        for key, value in expected.items():
            if key != "revision":
                frozen._expect(observed.get(key), value, f"preflight {role} {key}")
    frozen._expect(
        frozen._get(summary, "tokenizer"),
        frozen._tokenizer_identity(target),
        "preflight tokenizer identity",
    )
    input_tokens = frozen._get(summary, "generation.num_input_tokens")
    if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens <= 0:
        raise ValueError("preflight input token count must be a positive integer")
    output_ids = frozen._get(summary, "output.token_ids")
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != input_tokens + 1
        or any(isinstance(value, bool) or not isinstance(value, int) for value in output_ids)
    ):
        raise ValueError("preflight output token sequence is invalid")
    rendered = frozen._get(summary, "dataset.rendered_input_token_ids")
    frozen._expect(rendered.get("serialization"), "int64_le_c_order_v1", "rendered serialization")
    frozen._expect(rendered.get("shape"), [1, input_tokens], "rendered shape")
    packed = struct.pack(f"<{input_tokens}q", *output_ids[:input_tokens])
    rendered_sha256 = hashlib.sha256(packed).hexdigest()
    frozen._expect(rendered.get("sha256"), rendered_sha256, "rendered token sha256")

    rows = []
    with rounds_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank rounds.jsonl line {line_number}")
            row = json.loads(line, object_pairs_hook=frozen._reject_duplicate_keys)
            if not isinstance(row, dict):
                raise ValueError(f"rounds.jsonl line {line_number} is not an object")
            rows.append(row)
    frozen._expect(len(rows), 1, "preflight round count")
    frozen._expect(
        frozen._get(summary, "generation.rounds"), len(rows), "summary round count"
    )
    frozen._expect(
        frozen._get(summary, "output.rounds_sha256"),
        frozen._sha256_file(rounds_path),
        "preflight rounds sha256",
    )
    sample_id = frozen._get(summary, "dataset.sample_id")
    provenance = {
        "reference_revision": "94e4abc",
        "reference_source_sha256": reference_sha256,
        "target_declared_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "draft_declared_revision": "b74e3a329c4d963783143b1e970d95b002be72bd",
        "dataset_declared_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "dataset_sha256": dataset_sha256,
        "harness_source_sha256": harness_sha256,
    }
    frozen._check_fields(
        rows[0],
        (
            ("schema_version", frozen.HARNESS_ARTIFACT_SCHEMA_VERSION),
            ("round_index", 0),
            ("sample_id", sample_id),
            ("mode", "static"),
            ("trainable_scope", "none_static"),
            ("draft_cache_policy", "stale"),
            ("proposal_parameter_version", 0),
            ("parameter_version_after_update", 0),
        ),
        label="DFlash direct preflight round",
    )
    for key, value in provenance.items():
        frozen._expect(rows[0]["provenance"].get(key), value, f"round provenance {key}")
    frozen._expect(
        rows[0]["provenance"].get("tokenizer_content_identity_sha256"),
        frozen._get(summary, "tokenizer.content_identity_sha256"),
        "round tokenizer identity",
    )
    frozen._expect(
        rows[0]["provenance"].get("rendered_input_token_ids_sha256"),
        rendered_sha256,
        "round rendered input identity",
    )


try:
    validate()
except Exception as exc:
    print(f"DFlash direct preflight validation failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

run_dflash_long_context() {
  local preflight_root
  preflight_root="$LONG_CONTEXT/preflight/sample-$(printf '%04d' "$LONG_CONTEXT_SAMPLE")"
  local preflight_artifact="$preflight_root/artifact"
  local preflight_summary="$preflight_artifact/summary.json"
  local preflight_rounds="$preflight_artifact/rounds.jsonl"
  local preflight_history
  local -a modes=(
    static full-drafter drafter-lora full-rank-tail tail-lora output-residual
  )
  local -a contexts=()
  local -a artifacts=()
  read -r -a contexts <<< "$LONG_CONTEXTS"
  [ "${#contexts[@]}" -gt 0 ] || {
    echo "LIGHTCONE_LONG_CONTEXTS resolved to an empty list" >&2
    return 2
  }

  # A one-token held-out Static run binds the exact tokenizer rendering and
  # input-token count used to derive every requested total context.  Reuse a
  # completed preflight on resume.  A killed, malformed, or identity-mismatched
  # one-token run is immutable failure evidence, but must not poison the fixed
  # path: archive it under this queue session before retrying from a clean path.
  if ! dflash_preflight_complete "$preflight_summary" "$preflight_rounds"; then
    if path_exists "$preflight_root"; then
      preflight_history=$(new_archive_dir \
        "$SESSION_ROOT/attempt-history" \
        "$(date -u +%Y%m%dT%H%M%SZ)-attempt-$attempt_generation-q04-preflight") \
        || return
      mv "$preflight_root" "$preflight_history/preflight-partial" || return
      record q04_dflash_8k_16k_32k_40k archived_partial_preflight \
        "destination=$preflight_history/preflight-partial"
    fi
    mkdir -p "$preflight_root" || return
    env \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONPATH="$QUEUE_ROOT:$RUNTIME_ROOT/reference/tts-dflash/site-packages:$REFERENCE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" "$HARNESS" \
      --mode static \
      --reference-root "$REFERENCE_ROOT" \
      --reference-module dflash.model \
      --reference-revision 94e4abc \
      --target-model "$TARGET" \
      --target-revision 1cfa9a7208912126459214e8b04321603b3df60c \
      --draft-model "$DRAFT" \
      --draft-revision b74e3a329c4d963783143b1e970d95b002be72bd \
      --dataset "$DATASET" \
      --dataset-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
      --sample-index "$LONG_CONTEXT_SAMPLE" \
      --output-dir "$preflight_artifact" \
      --max-new-tokens 1 \
      --block-size 16 \
      --mask-token-id 151669 \
      --seed 0 \
      --lr 0.0001 \
      --proximal-lambda 0 \
      --update-stride 1 \
      --position-weighting uniform \
      --loss-reduction sum \
      --draft-cache-policy stale \
      --enable-thinking \
      --deterministic \
      --parity-max-new-tokens 1 || return
    if ! dflash_preflight_complete "$preflight_summary" "$preflight_rounds"; then
      echo "DFlash preflight exited without valid identity-bound completion evidence" >&2
      return 3
    fi
  fi

  "$PY" "$SCRIPTS/run_dflash_tts_frozen_sweep.py" \
    --python "$PY" \
    --harness "$HARNESS" \
    --stage1-analysis "$STAGE1_ANALYSIS" \
    --stage2-analysis "$STAGE2_ANALYSIS" \
    --reference-root "$REFERENCE_ROOT" \
    --reference-module dflash.model \
    --reference-revision 94e4abc \
    --target-model "$TARGET" \
    --target-revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --draft-model "$DRAFT" \
    --draft-revision b74e3a329c4d963783143b1e970d95b002be72bd \
    --dataset "$DATASET" \
    --dataset-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
    --sample-index "$LONG_CONTEXT_SAMPLE" \
    --preflight-summary "$preflight_summary" \
    --output-root "$LONG_CONTEXT" \
    --total-contexts "${contexts[@]}" \
    --modes "${modes[@]}" \
    --mask-token-id 151669 \
    --projection-artifact "$PROJECTION" \
    --pythonpath "$QUEUE_ROOT" \
    --pythonpath $RUNTIME_ROOT/reference/tts-dflash/site-packages \
    --pythonpath "$REFERENCE_ROOT" \
    --deterministic \
    --audit-cuda-timing \
    --parity-max-new-tokens 32 \
    --keep-going || return

  for context in "${contexts[@]}"; do
    for mode in "${modes[@]}"; do
      artifact="$LONG_CONTEXT/sample-$(printf '%04d' "$LONG_CONTEXT_SAMPLE")/context-$context/$mode/artifact"
      run_root=${artifact%/artifact}
      if [ ! -s "$artifact/summary.json" ] || \
          [ ! -s "$artifact/rounds.jsonl" ] || \
          [ ! -s "$run_root/completion.json" ] || \
          [ ! -s "$run_root/run_identity.json" ]; then
        echo "long-context run lacks immutable completion evidence: $run_root" >&2
        return 6
      fi
      artifacts+=("$artifact")
    done
  done
  "$PY" "$SCRIPTS/aggregate_dflash_tts_ablations.py" \
    --output-dir "$LONG_CONTEXT/aggregate" --bucket-size 1024 --parquet \
    "${artifacts[@]}"
}

run_load_saturation() {
  # Keep saturation evidence independent from P5.  The source manifest also
  # contains lc_gate, but this controller-free preflight deliberately selects
  # only Static, TTS and L0 so it can run before any controller artifact
  # exists.  One engine unit receives all 128 prepared prompts and the
  # manifest sweeps offered concurrency 1..48.
  mkdir -p "$LOAD_TUNE_ROOT" "$LOAD_TUNE_ANALYSIS" || return
  "$LC" prepare-datasets \
    --lockfile "$LOAD_TUNE_LOCKFILE" \
    --datasets alpaca \
    --limit 128 \
    --output "$LOAD_TUNE_ANALYSIS/dataset-preflight.json" || return
  run_sglang_headline run-manifest \
    --manifest "$LOAD_TUNE_MANIFEST" \
    --artifact-root "$LOAD_TUNE_ROOT" \
    --lockfile "$LOAD_TUNE_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots $RUNTIME_ROOT/model-roots.json \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods static tts naive_async \
    --weight-update-mode residual || return
  "$LC" validate-artifacts \
    --artifact-root "$LOAD_TUNE_ROOT" \
    --manifest "$LOAD_TUNE_MANIFEST" \
    --methods static tts naive_async \
    --weight-update-mode residual \
    --coverage-output "$LOAD_TUNE_ANALYSIS/coverage.json" || return
  "$LC" analyze \
    --artifact-root "$LOAD_TUNE_ROOT" \
    --output-dir "$LOAD_TUNE_ANALYSIS" \
    --manifest "$LOAD_TUNE_MANIFEST" \
    --methods static tts naive_async \
    --weight-update-mode residual
}

require_p5_engine_reuse_transition_safe() {
  # The engine-reuse manifest has different unit identities from the old
  # per-context P5 manifest.  Switching is safe while the shared root is empty,
  # and resume is safe only when every existing run belongs to one of the four
  # exact execution overlays below.  Reject mixed/legacy evidence before any
  # model load instead of letting an unrelated run hide behind scoped
  # validation warnings.
  "$PY" - "$P5_MANIFEST" "$P5_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.units import RunUnit

manifest_path = Path(sys.argv[1])
artifact_root = Path(sys.argv[2])
source = ExperimentManifest.load(manifest_path)
if source.name != "p5_long_context_acceptance_engine_reuse":
    raise SystemExit(
        "q05 requires the validated p5_long_context_acceptance_engine_reuse "
        f"manifest, got {source.name!r}"
    )

plans = (
    (("static", "tts"), "residual"),
    (("static", "tts"), "lora"),
    (("static", "tts"), "full"),
    (("naive_async",), "residual"),
)
allowed = {}
for methods, mode in plans:
    effective = source.with_methods(methods).with_weight_update_mode(mode)
    digest = effective.content_sha256()
    for unit in effective.units:
        allowed.setdefault(unit.unit_id, set()).add(digest)

if artifact_root.is_dir():
    for run_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        run_manifest_path = run_dir / "manifest.json"
        if not run_manifest_path.is_file():
            raise SystemExit(
                f"q05 artifact root contains an unbound directory: {run_dir}"
            )
        try:
            payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            unit = RunUnit.from_dict(payload)
        except Exception as exc:
            raise SystemExit(
                f"q05 artifact root contains an unreadable identity at "
                f"{run_manifest_path}: {exc}"
            ) from exc
        claimed = payload.get("unit_id")
        if claimed != unit.unit_id or claimed not in allowed:
            raise SystemExit(
                f"q05 artifact root contains a legacy or foreign unit: {run_dir}"
            )
        experiment_digest = payload.get("experiment_manifest_sha256")
        if experiment_digest not in allowed[claimed]:
            raise SystemExit(
                f"q05 artifact root contains a cross-overlay manifest identity: "
                f"{run_dir}"
            )
PY
}

run_p5_residual_l0() {
  mkdir -p "$P5_ROOT" "$P5_ANALYSIS"
  "$LC" prepare-datasets \
    --lockfile "$P5_LOCKFILE" \
    --datasets livecodebench math500 mt_bench \
    --limit 32 \
    --output "$P5_ANALYSIS/dataset-preflight.json" || return
  run_sglang_headline run-manifest \
    --manifest "$P5_MANIFEST" \
    --artifact-root "$P5_ROOT" \
    --lockfile "$P5_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots $RUNTIME_ROOT/model-roots.json \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods naive_async \
    --weight-update-mode residual || return
  # q05a populated the same artifact root with residual Static+TTS.  Validate
  # and analyze the controller-free three-method matrix after L0 is added; no
  # GPU unit is executed twice.
  "$LC" validate-artifacts \
    --artifact-root "$P5_ROOT" --manifest "$P5_MANIFEST" \
    --methods static tts naive_async \
    --weight-update-mode residual \
    --coverage-output "$P5_ANALYSIS/coverage.json" || return
  "$LC" analyze --artifact-root "$P5_ROOT" --output-dir "$P5_ANALYSIS" \
    --manifest "$P5_MANIFEST" \
    --methods static tts naive_async \
    --weight-update-mode residual
}

run_p5_tts_mode_screen() {
  local mode=$1
  shift
  local -a methods=("$@")
  [ "${#methods[@]}" -gt 0 ] || {
    echo "run_p5_tts_mode_screen requires at least one method" >&2
    return 2
  }
  # Static has no update parameterization: its unit and execution hashes are
  # identical for all three overlays.  Keep one immutable evidence set and let
  # later mode screens resume/skip the 15 Static units.  Analysis remains
  # mode-scoped and expected-unit-bound.
  local root="$P5_ROOT"
  local analysis="$P5_TTS_ANALYSIS_BASE-$mode"
  mkdir -p "$root" "$analysis"
  "$LC" prepare-datasets \
    --lockfile "$P5_LOCKFILE" \
    --datasets livecodebench math500 mt_bench \
    --limit 32 \
    --output "$analysis/dataset-preflight.json" || return
  run_sglang_headline run-manifest \
    --manifest "$P5_MANIFEST" \
    --artifact-root "$root" \
    --lockfile "$P5_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots $RUNTIME_ROOT/model-roots.json \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods "${methods[@]}" \
    --weight-update-mode "$mode" || return
  "$LC" validate-artifacts \
    --artifact-root "$root" --manifest "$P5_MANIFEST" \
    --methods "${methods[@]}" \
    --weight-update-mode "$mode" \
    --coverage-output "$analysis/coverage.json" || return
  "$LC" analyze --artifact-root "$root" --output-dir "$analysis" \
    --manifest "$P5_MANIFEST" \
    --methods "${methods[@]}" \
    --weight-update-mode "$mode"
}

priority_terminal_closed() {
  local terminal="$SESSION_ROOT/priority-terminal.json"
  [ -s "$terminal" ] || return 1
  "$PY" - "$terminal" "$run_id" "$PRIORITY_PAIR" \
    "$PRIORITY_MODE" "$PRIORITY_LR" \
    "$PRIORITY_ARTIFACT_CONTRACT" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

terminal_name, run_id, pair, mode, lr_text, epoch_name = sys.argv[1:]
try:
    epoch_path = pathlib.Path(epoch_name)
    epoch_sidecar = pathlib.Path(str(epoch_path) + ".sha256")
    epoch_sha = hashlib.sha256(epoch_path.read_bytes()).hexdigest()
    if epoch_sidecar.read_text(encoding="utf-8").strip() != epoch_sha:
        raise ValueError("artifact_epoch_sidecar")
    payload = json.loads(pathlib.Path(terminal_name).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("schema")
    if payload.get("queue_run_id") != run_id:
        raise ValueError("run")
    if payload.get("status") not in {"complete", "scientifically_blocked"}:
        raise ValueError("status")
    reasons = payload.get("scientific_block_reasons")
    if not isinstance(reasons, list):
        raise ValueError("scientific_block_reasons")
    if (payload["status"] == "complete") == bool(reasons):
        raise ValueError("status/reasons")
    contract = payload.get("contract", {})
    if contract.get("model_pair") != pair or contract.get("weight_update_mode") != mode:
        raise ValueError("contract")
    if contract.get("artifact_epoch_sha256") != epoch_sha:
        raise ValueError("artifact_epoch")
    if not math.isclose(
        float(contract.get("learning_rate")), float(lr_text), rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("learning_rate")
    actual = payload.get("actual_methods")
    if not isinstance(actual, list) or actual[:3] != ["static", "tts", "naive_async"]:
        raise ValueError("methods")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence")
    for row in evidence:
        path = pathlib.Path(row["path"])
        if not path.is_file():
            raise ValueError(f"missing {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.get("sha256"):
            raise ValueError(f"hash drift {path}")
    claims = payload.get("claim_gate_summary")
    if not isinstance(claims, dict) or set(claims) != {"static_vs_tts", "final_vs_tts"}:
        raise ValueError("claim_gate_summary")
    coverage = payload.get("coverage_contract")
    if not isinstance(coverage, dict) or coverage.get("contexts") != [
        512, 4096, 16384, 40000
    ]:
        raise ValueError("coverage_contract")
    if set(claims["final_vs_tts"]) != {
        "naive_async", "lc_gate", "lc_damp", "lc_transport"
    }:
        raise ValueError("final method coverage")
except Exception as exc:
    print(f"priority terminal is not closed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

priority_terminal_status() {
  "$PY" -c \
    'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])' \
    "$SESSION_ROOT/priority-terminal.json"
}

priority_archive_partial_pair() {
  local path=$1
  local sidecar="${path}.sha256"
  local label=$2
  local history
  if { [ -e "$path" ] && [ ! -e "$sidecar" ]; } || \
      { [ ! -e "$path" ] && [ -e "$sidecar" ]; }; then
    history=$(new_archive_dir "$SESSION_ROOT/attempt-history" \
      "$(date -u +%Y%m%dT%H%M%SZ)-attempt-$attempt_generation-$label") \
      || return
    [ ! -e "$path" ] || mv "$path" "$history/" || return
    [ ! -e "$sidecar" ] || mv "$sidecar" "$history/" || return
    record p00_priority_inputs archived_partial "$label=$history"
  fi
}

run_priority_calibration() {
  PYTHONPATH="$PRIORITY_WORKSPACE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$PRIORITY_CALIBRATION_SCRIPT" \
      --artifact-root "$PRIORITY_CALIBRATION_ROOT" \
      --calibration-manifest "$PRIORITY_CALIBRATION_MANIFEST" \
      --evaluation-manifest "$PRIORITY_MANIFEST" \
      --candidate-spec "$PRIORITY_CALIBRATION_SPEC" \
      --lockfile "$PRIORITY_LOCKFILE" \
      --model-roots "$PRIORITY_MODEL_ROOTS" \
      --runtime-root $RUNTIME_ROOT \
      --lightcone-cli "$LC"
}

load_priority_calibration_winner() {
  local winner="$SESSION_ROOT/.priority-calibration-winner.$$"
  "$PY" - "$PRIORITY_CALIBRATION_READY" > "$winner" <<'PY'
import hashlib
import json
import math
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_attested(path, seen=None):
    path = pathlib.Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return json.loads(path.read_text(encoding="utf-8"))
    seen.add(path)
    sidecar = pathlib.Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"incomplete attested calibration receipt: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != digest(path):
        raise ValueError(f"calibration receipt hash drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"calibration receipt is not an object: {path}")
    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError(f"calibration receipt evidence is not a list: {path}")
    for row in evidence:
        evidence_path = pathlib.Path(row["path"]).resolve()
        if (
            not evidence_path.is_file()
            or not isinstance(row.get("sha256"), str)
            or digest(evidence_path) != row["sha256"]
        ):
            raise ValueError(f"calibration receipt evidence drift: {evidence_path}")
        evidence_sidecar = pathlib.Path(str(evidence_path) + ".sha256")
        if evidence_sidecar.is_file():
            try:
                nested = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                nested = None
            if isinstance(nested, dict) and "evidence" in nested:
                verify_attested(evidence_path, seen)
    return payload


ready_path = pathlib.Path(sys.argv[1]).resolve()
ready = verify_attested(ready_path)
if ready.get("schema_version") != 1 or ready.get("status") != "ready":
    raise ValueError("calibration DOWNSTREAM_READY is not ready/schema-v1")
winner = ready.get("winner")
if not isinstance(winner, dict):
    raise ValueError("calibration DOWNSTREAM_READY has no winner")
mode = winner.get("weight_update_mode")
lr = winner.get("learning_rate")
if mode not in {"residual", "lora", "full"}:
    raise ValueError(f"invalid calibrated weight-update mode: {mode!r}")
if isinstance(lr, bool) or not isinstance(lr, (int, float)):
    raise ValueError("invalid calibrated learning rate")
lr = float(lr)
if not math.isfinite(lr) or lr <= 0.0:
    raise ValueError("invalid calibrated learning rate")
evidence_by_name = {
    pathlib.Path(row["path"]).name: pathlib.Path(row["path"]).resolve()
    for row in ready["evidence"]
}
gate_path = evidence_by_name.get("heldout-gate.json")
selection_path = evidence_by_name.get("calibration-selection.json")
if gate_path is None or selection_path is None:
    raise ValueError("DOWNSTREAM_READY lacks gate or selection evidence")
gate = verify_attested(gate_path)
selection = verify_attested(selection_path)
if digest(gate_path) != ready.get("heldout_gate_sha256"):
    raise ValueError("DOWNSTREAM_READY heldout gate hash mismatch")
if gate.get("winner") != winner or selection.get("winner") != winner:
    raise ValueError("calibrated winner differs across frozen receipts")
if gate.get("verdict", {}).get("downstream_ready") is not True:
    raise ValueError("held-out calibration gate is not downstream-ready")
print(f"{mode}\t{lr:.17g}")
PY
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$winner"
    return "$rc"
  fi
  IFS=$'\t' read -r PRIORITY_MODE PRIORITY_LR < "$winner"
  rm -f "$winner"
  [ -n "$PRIORITY_MODE" ] && [ -n "$PRIORITY_LR" ]
}

archive_stale_priority_evidence() {
  # Artifact roots are reusable only inside one explicit deployment epoch.
  # Unit IDs do not bind every runtime source file; without this fence a new
  # implementation can publish a second complete run for the same unit beside
  # an older one and make validation permanently ambiguous.  Archive all six
  # result roots and the terminal as one recoverable transaction on a contract
  # change (or when introducing the contract over legacy evidence).
  "$PY" - "$PRIORITY_ARTIFACT_CONTRACT" \
    "$PRIORITY_SMOKE_ROOT" "$PRIORITY_EVAL_ROOT" \
    "$PRIORITY_TRACE_ROOT" "$PRIORITY_PHASE1_CONTROLLER_ROOT" \
    "$PRIORITY_CONTROLLER_ROOT" \
    "$PRIORITY_ANALYSIS_ROOT" "$SESSION_ROOT/priority-terminal.json" \
    "$SESSION_ROOT/attempt-history" "$attempt_generation" \
    "$PRIORITY_PAIR" "$PRIORITY_MODE" "$PRIORITY_LR" \
    "$PRIORITY_LOCKFILE" "$PRIORITY_MODEL_ROOTS" \
    "$PRIORITY_DATASET_RECEIPT" "$PRIORITY_WORKSPACE_ROOT" "$0" \
    "$PRIORITY_CALIBRATION_SCRIPT" \
    "$PRIORITY_MANIFEST" "$PRIORITY_TRACE_MANIFEST" \
    "$PRIORITY_L3_EVALUATION_MANIFEST" "$PRIORITY_SMOKE_MANIFEST" \
    "$PRIORITY_CALIBRATION_MANIFEST" \
    "$PRIORITY_CALIBRATION_SPEC" "$PRIORITY_CALIBRATION_READY" \
    "${PRIORITY_PRODUCTION_FILES[@]}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import time
import sys

(
    contract_name,
    smoke_name,
    eval_name,
    trace_name,
    phase1_controller_name,
    controller_name,
    analysis_name,
    terminal_name,
    archive_parent_name,
    generation,
    pair,
    mode,
    lr_text,
    lock_name,
    model_roots_name,
    dataset_receipt_name,
    workspace_name,
    queue_script_name,
    calibration_script_name,
    manifest_name,
    trace_manifest_name,
    l3_evaluation_manifest_name,
    smoke_manifest_name,
    calibration_manifest_name,
    calibration_spec_name,
    calibration_ready_name,
    *production_names,
) = sys.argv[1:]
if len(production_names) != 22:
    raise SystemExit(
        f"priority deployment contract requires 22 production files; "
        f"received {len(production_names)}"
    )

contract_path = pathlib.Path(contract_name).expanduser().absolute()
contract_sidecar = pathlib.Path(str(contract_path) + ".sha256")
terminal_path = pathlib.Path(terminal_name).expanduser().absolute()
archive_parent = pathlib.Path(archive_parent_name).expanduser().absolute()
roots = [
    ("smoke", pathlib.Path(smoke_name).expanduser().absolute()),
    ("eval", pathlib.Path(eval_name).expanduser().absolute()),
    ("trace", pathlib.Path(trace_name).expanduser().absolute()),
    (
        "phase1-controller",
        pathlib.Path(phase1_controller_name).expanduser().absolute(),
    ),
    ("controller", pathlib.Path(controller_name).expanduser().absolute()),
    ("analysis", pathlib.Path(analysis_name).expanduser().absolute()),
]


def exists(path):
    return os.path.lexists(path)


def within(path, parent):
    return path == parent or parent in path.parents


root_paths = [root for _, root in roots]
if len(set(root_paths)) != len(root_paths):
    raise ValueError("priority artifact roots must be distinct")
for index, root in enumerate(root_paths):
    for other in root_paths[index + 1:]:
        if within(root, other) or within(other, root):
            raise ValueError("priority artifact roots must not be nested")
    for protected in (
        archive_parent,
        contract_path,
        contract_sidecar,
        terminal_path,
        pathlib.Path(dataset_receipt_name).expanduser().absolute(),
        pathlib.Path(calibration_ready_name).expanduser().absolute(),
    ):
        if within(protected, root):
            raise ValueError(
                f"protected priority path {protected} is inside live root {root}"
            )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path, logical_path=None):
    path = pathlib.Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"contract input is not a file: {path}")
    return {
        "path": str(path) if logical_path is None else logical_path,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def paired_record(path, label):
    path = pathlib.Path(path).resolve()
    sidecar = pathlib.Path(str(path) + ".sha256")
    primary = file_record(path)
    side = file_record(sidecar)
    declared = sidecar.read_text(encoding="utf-8").strip().split()
    if not declared or declared[0] != primary["sha256"]:
        raise ValueError(f"{label} hash sidecar mismatch: {path}")
    return {"artifact": primary, "sidecar": side}


workspace = pathlib.Path(workspace_name).resolve()
production = [
    file_record(workspace / relative, logical_path=relative)
    for relative in production_names
]
manifests = [
    paired_record(path, "manifest")
    for path in (
        manifest_name,
        trace_manifest_name,
        l3_evaluation_manifest_name,
        smoke_manifest_name,
        calibration_manifest_name,
    )
]


def command_output(command):
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return "\n".join(sorted(line.strip() for line in result.stdout.splitlines()))


def tree_fingerprint(root):
    root = pathlib.Path(root).resolve()
    records = []
    total_bytes = 0
    source_suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
    for path in sorted(root.rglob("*")):
        if path.suffix not in source_suffixes:
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": digest(path),
            }
        )
    if not records:
        raise ValueError(f"runtime tree has no source files: {root}")
    body = json.dumps(records, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return {
        "root": str(root),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "merkle_sha256": hashlib.sha256(body).hexdigest(),
    }


packages = {}
for package in (
    "flashinfer-python",
    "huggingface-hub",
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "safetensors",
    "scikit-learn",
    "scipy",
    "sglang",
    "tokenizers",
    "torch",
    "transformers",
    "triton",
):
    try:
        packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        packages[package] = None
cuda_home = os.environ.get("CUDA_HOME")
try:
    import torch

    torch_cuda_version = torch.version.cuda
except Exception:
    torch_cuda_version = None
runtime_environment = {
    "python_executable": str(pathlib.Path(sys.executable).resolve()),
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "packages": packages,
    "torch_cuda_version": torch_cuda_version,
    "cuda_home": cuda_home,
    "nvcc_version": command_output(
        [str(pathlib.Path(cuda_home) / "bin" / "nvcc"), "--version"]
    ) if cuda_home else None,
    "gpu_identity": command_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ),
}
expected = {
    "schema_version": 1,
    "experiment": {
        "model_pair": pair,
        "weight_update_mode": mode,
        "learning_rate": float(lr_text),
    },
    "deployment": {
        "production_files": production,
        "manifests": manifests,
        "queue_script": file_record(queue_script_name),
        "calibration_queue_script": file_record(calibration_script_name),
        "runtime_trees": [
            tree_fingerprint(workspace / "src/lightcone_spec"),
            tree_fingerprint(workspace / "sglang/python/sglang/srt"),
        ],
        "runtime_environment": runtime_environment,
    },
    "inputs": {
        "lockfile": paired_record(lock_name, "lockfile"),
        "model_roots": paired_record(model_roots_name, "model-roots"),
        "dataset_receipt": paired_record(
            dataset_receipt_name, "dataset receipt"
        ),
        "calibration_candidate_spec": paired_record(
            calibration_spec_name, "calibration candidate spec"
        ),
        "calibration_ready": paired_record(
            calibration_ready_name, "calibration downstream-ready receipt"
        ),
    },
}
contract_body = (
    json.dumps(expected, indent=2, sort_keys=True) + "\n"
).encode("utf-8")
expected_sha = hashlib.sha256(contract_body).hexdigest()


def hash_ledger_is_current(base, ledger_path):
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict) or not ledger:
            return False
        base = base.resolve()
        for relative, entry in ledger.items():
            candidate = (base / relative).resolve()
            if base not in candidate.parents:
                return False
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("sha256"), str)
                or not isinstance(entry.get("bytes"), int)
                or not candidate.is_file()
                or candidate.stat().st_size != entry["bytes"]
                or digest(candidate) != entry["sha256"]
            ):
                return False
        return True
    except Exception:
        return False


def completed_run_root_is_current(root):
    normative = {
        "manifest.json",
        "manifest.sha256",
        "environment.json",
        "lock-reference.json",
        "stdout.log",
        "stderr.log",
        "exit.json",
        "rounds.parquet",
        "updates.parquet",
        "decisions.parquet",
        "system_samples.parquet",
        "request_summary.parquet",
    }
    run_dirs = sorted({
        path.parent
        for path in root.rglob("manifest.json")
        if path.parent.is_dir() and not path.parent.name.startswith(".")
    })
    if not run_dirs:
        return False
    unit_ids = []
    for run_dir in run_dirs:
        try:
            manifest_path = run_dir / "manifest.json"
            exit_path = run_dir / "exit.json"
            hashes_path = run_dir / "hashes.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            completion = json.loads(exit_path.read_text(encoding="utf-8"))
            ledger = json.loads(hashes_path.read_text(encoding="utf-8"))
            if (
                completion.get("status") != "complete_valid"
                or completion.get("exit_code") != 0
                or not normative.issubset(ledger)
                or not hash_ledger_is_current(run_dir, hashes_path)
                or digest(manifest_path)
                != (run_dir / "manifest.sha256").read_text(
                    encoding="utf-8"
                ).strip()
            ):
                return False
            unit_ids.append(manifest["unit_id"])
        except Exception:
            return False
    return len(unit_ids) == len(set(unit_ids))


def completed_analysis_is_current(root):
    for relative in ("static-tts", "l0-vs-tts", "final-vs-tts"):
        output = root / relative
        ledger = output / "analysis-hashes.json"
        if not hash_ledger_is_current(output, ledger):
            return False
        try:
            manifest = output / "analysis-manifest.json"
            if digest(manifest) != (output / "analysis-manifest.sha256").read_text(
                encoding="utf-8"
            ).strip():
                return False
        except Exception:
            return False
    return True


def terminal_is_bound():
    if not exists(terminal_path):
        return True
    try:
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("contract", {}).get("artifact_epoch_sha256") != expected_sha:
            return False
        for row in terminal.get("evidence", []):
            evidence_path = pathlib.Path(row["path"])
            if not evidence_path.is_file() or digest(evidence_path) != row["sha256"]:
                return False
        live = dict(roots)
        live_ledgers = {
            str(path.resolve())
            for label in ("smoke", "eval", "trace")
            for path in live[label].rglob("hashes.json")
        }
        live_ledgers.update(
            str(path.resolve())
            for path in live["analysis"].rglob("analysis-hashes.json")
        )
        if set(terminal.get("artifact_ledgers", [])) != live_ledgers:
            return False
        return bool(terminal.get("evidence")) and all(
            completed_run_root_is_current(live[label])
            for label in ("smoke", "eval", "trace")
        ) and completed_analysis_is_current(live["analysis"])
    except Exception:
        return False


def read_current_contract():
    if not contract_path.is_file() or not contract_sidecar.is_file():
        return None
    try:
        body = contract_path.read_bytes()
        declared = contract_sidecar.read_text(encoding="utf-8").strip().split()
        if not declared or declared[0] != hashlib.sha256(body).hexdigest():
            return None
        return json.loads(body)
    except Exception:
        return None


current = read_current_contract()
reasons = []
if current is None:
    if exists(contract_path) or exists(contract_sidecar):
        reasons.append("artifact contract is partial or invalid")
    legacy = exists(terminal_path) or any(
        exists(root) and (not root.is_dir() or any(root.iterdir()))
        for _, root in roots
    )
    if legacy:
        reasons.append("legacy priority evidence has no artifact contract")
elif current != expected:
    reasons.append("priority artifact contract changed")
elif not terminal_is_bound():
    reasons.append("priority terminal is invalid or belongs to another epoch")
for label, root in roots:
    if exists(root) and (not root.is_dir() or root.is_symlink()):
        reasons.append(f"{label} root is not a directory: {root}")


def write_contract():
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_tmp = contract_path.with_name(f".{contract_path.name}.{os.getpid()}")
    sidecar_tmp = contract_sidecar.with_name(
        f".{contract_sidecar.name}.{os.getpid()}"
    )
    try:
        contract_tmp.write_bytes(contract_body)
        sidecar_tmp.write_text(expected_sha + "\n", encoding="utf-8")
        os.replace(contract_tmp, contract_path)
        os.replace(sidecar_tmp, contract_sidecar)
    finally:
        for temporary in (contract_tmp, sidecar_tmp):
            if exists(temporary):
                temporary.unlink()


if not reasons:
    for _, root in roots:
        root.mkdir(parents=True, exist_ok=True)
    if current is None:
        write_contract()
    print(f"priority artifact contract is current: {expected_sha}")
    raise SystemExit(0)

archive_parent.mkdir(parents=True, exist_ok=True)
archive = pathlib.Path(tempfile.mkdtemp(
    prefix=f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-attempt-"
    f"{generation}-priority-artifact-epoch.",
    dir=archive_parent,
))
reason_path = archive / "archive-reason.json"
reason_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "reasons": reasons,
            "expected_artifact_contract_sha256": expected_sha,
            "expected_artifact_contract": expected,
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
moved = []
created = []
write_started = False
try:
    for label, root in roots:
        if exists(root):
            destination = archive / label
            os.replace(root, destination)
            moved.append((destination, root))
    for label, path in (
        ("priority-terminal.json", terminal_path),
        ("artifact-contract.json", contract_path),
        ("artifact-contract.json.sha256", contract_sidecar),
    ):
        if exists(path):
            destination = archive / label
            os.replace(path, destination)
            moved.append((destination, path))
    for _, root in roots:
        root.mkdir(parents=True, exist_ok=False)
        created.append(root)
    write_started = True
    write_contract()
except Exception:
    if write_started:
        for path in (contract_path, contract_sidecar):
            if exists(path):
                path.unlink()
    for root in reversed(created):
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()
    for destination, original in reversed(moved):
        if exists(destination) and not exists(original):
            original.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, original)
    if reason_path.exists():
        reason_path.unlink()
    if archive.is_dir() and not any(archive.iterdir()):
        archive.rmdir()
    raise
print(f"archived stale priority artifact epoch at {archive}: {'; '.join(reasons)}")
PY
}

quarantine_invalid_priority_runs() {
  local root=$1
  local label=$2
  "$PY" - "$root" "$SESSION_ROOT/attempt-history" \
    "$attempt_generation" "$label" <<'PY'
import json
import os
import pathlib
import tempfile
import time
import sys

from lightcone_spec.artifacts.rundir import REQUIRED_FILES
from lightcone_spec.locking.hashing import sha256_file

root_name, archive_parent_name, generation, label = sys.argv[1:]
root = pathlib.Path(root_name).resolve()
archive_parent = pathlib.Path(archive_parent_name).resolve()
if not root.is_dir():
    raise SystemExit(f"priority {label} root is not a directory: {root}")
if archive_parent == root or root in archive_parent.parents:
    raise SystemExit("priority quarantine parent must be outside the live root")


def current(run_dir):
    try:
        exit_payload = json.loads(
            (run_dir / "exit.json").read_text(encoding="utf-8")
        )
        if exit_payload.get("status") != "complete_valid" or exit_payload.get(
            "exit_code"
        ) != 0:
            return False
        hashes = json.loads((run_dir / "hashes.json").read_text(encoding="utf-8"))
        if not isinstance(hashes, dict) or not (
            set(REQUIRED_FILES) - {"hashes.json"}
        ).issubset(hashes):
            return False
        base = run_dir.resolve()
        for relative, entry in hashes.items():
            path = (run_dir / relative).resolve()
            if base not in path.parents or (
                not isinstance(entry, dict)
                or not isinstance(entry.get("sha256"), str)
                or not isinstance(entry.get("bytes"), int)
                or not path.is_file()
                or path.stat().st_size != entry["bytes"]
                or sha256_file(path) != entry["sha256"]
            ):
                return False
        manifest = run_dir / "manifest.json"
        return sha256_file(manifest) == (run_dir / "manifest.sha256").read_text(
            encoding="utf-8"
        ).strip()
    except Exception:
        return False


invalid = sorted(
    path
    for path in root.iterdir()
    if path.is_dir() and not path.name.startswith(".") and not current(path)
)
if not invalid:
    raise SystemExit(0)
archive_parent.mkdir(parents=True, exist_ok=True)
archive = pathlib.Path(tempfile.mkdtemp(
    prefix=f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-attempt-"
    f"{generation}-priority-{label}-quarantine.",
    dir=archive_parent,
))
reason = archive / "archive-reason.json"
reason.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "reason": "incomplete, failed, or hash-invalid run",
            "entries": [path.name for path in invalid],
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
moved = []
try:
    for path in invalid:
        destination = archive / path.name
        os.replace(path, destination)
        moved.append((destination, path))
except Exception:
    for destination, original in reversed(moved):
        if destination.exists() and not original.exists():
            os.replace(destination, original)
    reason.unlink(missing_ok=True)
    if archive.is_dir() and not any(archive.iterdir()):
        archive.rmdir()
    raise
print(f"quarantined invalid priority {label} runs at {archive}")
PY
}

validate_priority_trace_contract() {
  local stage=${1:-phase1}
  local require_controller=${2:-0}
  "$PY" - "$PRIORITY_TRACE_MANIFEST" \
    "$PRIORITY_L3_EVALUATION_MANIFEST" "$PRIORITY_LOCKFILE" \
    "$PRIORITY_PHASE1_TRACE_ROOT" \
    "$PRIORITY_L3_EVALUATION_TRACE_ROOT" \
    "$PRIORITY_PHASE1_CONTROLLER_ROOT" "$PRIORITY_CONTROLLER_ROOT" \
    "$SESSION_ROOT/attempt-history" "$attempt_generation" \
    "$PRIORITY_MODE" "$PRIORITY_LR" "$stage" \
    "$require_controller" <<'PY'
import json
import os
import pathlib
import tempfile
import time
import sys
from collections import Counter
from dataclasses import replace

from lightcone_spec.artifacts.rundir import REQUIRED_FILES
from lightcone_spec.locking.hashing import sha256_file, sha256_json
from lightcone_spec.locking.lockfile import load_lockfile
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.runtime_config import (
    runtime_implementation_fingerprint,
)

(
    manifest_name,
    l3_evaluation_manifest_name,
    lock_name,
    phase1_trace_name,
    l3_evaluation_trace_name,
    phase1_controller_name,
    controller_name,
    archive_parent_name,
    generation,
    mode,
    lr_text,
    stage,
    require_controller_text,
) = sys.argv[1:]
if stage == "phase1":
    source = ExperimentManifest.load(manifest_name)
    effective = source.with_methods(("tts", "naive_async"))
    trace_root = pathlib.Path(phase1_trace_name).resolve()
    controller_root = pathlib.Path(phase1_controller_name).resolve()
elif stage == "l3-evaluation":
    source = ExperimentManifest.load(l3_evaluation_manifest_name)
    effective = source.with_methods(("lc_transport",))
    trace_root = pathlib.Path(l3_evaluation_trace_name).resolve()
    controller_root = pathlib.Path(phase1_controller_name).resolve()
elif stage == "final":
    source = ExperimentManifest.load(manifest_name)
    effective = source.with_methods(("tts", "naive_async"))
    trace_root = pathlib.Path(phase1_trace_name).resolve()
    controller_root = pathlib.Path(controller_name).resolve()
else:
    raise SystemExit(f"unknown priority trace-contract stage: {stage}")
archive_parent = pathlib.Path(archive_parent_name).resolve()
require_controller = require_controller_text == "1"

lock = load_lockfile(lock_name)
effective = (
    effective.with_weight_update_mode(mode)
    .with_learning_rate(float(lr_text))
)
effective = replace(effective, lockfile_sha256=lock.content_sha256())
expected_units = {unit.unit_id: unit for unit in effective.units}
compiler = getattr(lock.environment, "compiler_versions", {}) or {}
locked_reference = {
    key: compiler[key]
    for key in (
        "lightcone_runtime_source_sha256",
        "sglang_runtime_source_sha256",
        "sglang_fork_commit",
        "sglang_fork_dirty",
    )
    if key in compiler
}
expected_runtime_sha = runtime_implementation_fingerprint(
    locked_reference=locked_reference
)["sha256"]
materialized_engine_keys = {
    "adaptation_config_path",
    "model_roots",
    "telemetry_glob",
    "runtime_config_sha256",
    "locked_target_revision",
    "locked_drafter_revision",
    "speculative_algorithm",
    "speculative_capabilities",
    "weight_update_mode",
    "parameter_scope",
    "tail_layout_mode",
    "effective_adapter_rank",
    "memory_calibration_identity",
    "memory_calibration_sha256",
    "memory_calibration_path",
    "calibrated_reserve_mb",
    "preflight_adaptation_reserve_mb",
    "forward_dtype",
    "profile_output_dir",
}


def inside(path, parent):
    path = path.resolve()
    parent = parent.resolve()
    return parent in path.parents


def validate_run(run_dir):
    errors = []
    unit_id = None
    layouts = set()
    manifest_path = run_dir / "manifest.json"
    exit_path = run_dir / "exit.json"
    hashes_path = run_dir / "hashes.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        unit_id = payload.get("unit_id")
        if unit_id not in expected_units:
            errors.append("foreign trace unit")
        else:
            expected_unit = expected_units[unit_id].to_manifest_dict()
            if any(payload.get(key) != value for key, value in expected_unit.items()):
                errors.append("trace unit contract mismatch")

        completion = json.loads(exit_path.read_text(encoding="utf-8"))
        if completion.get("status") != "complete_valid" or completion.get(
            "exit_code"
        ) != 0:
            errors.append("trace run is not complete_valid")

        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        if not isinstance(hashes, dict):
            raise ValueError("hashes.json is not an object")
        missing = sorted((set(REQUIRED_FILES) - {"hashes.json"}) - set(hashes))
        if missing:
            errors.append(f"normative hashes missing: {missing}")
        for relative, entry in hashes.items():
            candidate = (run_dir / relative).resolve()
            if not inside(candidate, run_dir):
                errors.append(f"hash path escapes run: {relative}")
                continue
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64
                or not isinstance(entry.get("bytes"), int)
                or not candidate.is_file()
                or candidate.stat().st_size != entry["bytes"]
                or sha256_file(candidate) != entry["sha256"]
            ):
                errors.append(f"trace hash drift: {relative}")
        sidecar = run_dir / "manifest.sha256"
        if (
            not sidecar.is_file()
            or sidecar.read_text(encoding="utf-8").strip()
            != sha256_file(manifest_path)
        ):
            errors.append("trace manifest sidecar mismatch")

        indexes = sorted((run_dir / "runtime").rglob("index*.jsonl"))
        records = 0
        if not indexes:
            errors.append("trace run has no replay index")
        for index_path in indexes:
            for line_number, line in enumerate(
                index_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                records += 1
                item = json.loads(line)
                layouts.add(item.get("parameter_layout_sha256"))
                shard = (index_path.parent / item["path"]).resolve()
                if not inside(shard, index_path.parent):
                    errors.append(
                        f"trace index path escapes shard root: "
                        f"{index_path.name}:{line_number}"
                    )
                elif (
                    not shard.is_file()
                    or not isinstance(item.get("sha256"), str)
                    or len(item["sha256"]) != 64
                    or not isinstance(item.get("bytes"), int)
                    or shard.stat().st_size != item["bytes"]
                    or sha256_file(shard) != item["sha256"]
                ):
                    errors.append(
                        f"trace replay shard drift: {index_path.name}:{line_number}"
                    )
        if indexes and records == 0:
            errors.append("trace replay indexes are empty")

        runtime_engine = payload.get("engine_params", {})
        outer_engine = {
            key: value
            for key, value in runtime_engine.items()
            if key not in materialized_engine_keys
        }
        reconstructed = replace(
            effective,
            engine_params=outer_engine,
            lockfile_sha256=lock.content_sha256(),
        )
        if reconstructed.content_sha256() != payload.get(
            "experiment_manifest_sha256"
        ):
            errors.append("experiment manifest hash mismatch")
        if any(
            outer_engine.get(key) != value
            for key, value in effective.engine_params.items()
        ):
            errors.append("source manifest engine mismatch")
        runtime_sha = (
            runtime_engine.get("runtime_implementation_fingerprint", {}).get(
                "sha256"
            )
        )
        if runtime_sha != expected_runtime_sha:
            errors.append("runtime hash mismatch")
    except Exception as exc:
        errors.append(f"unreadable trace identity: {exc}")
    return unit_id, layouts, errors


if not trace_root.is_dir():
    raise SystemExit(f"priority trace root is not a directory: {trace_root}")
invalid = []
valid = []
allowed_files = {
    "dataset-preflight.json",
    "dataset-preflight.json.sha256",
}
for entry in sorted(trace_root.iterdir()):
    if not entry.is_dir():
        if entry.name == ".run-manifest.lock":
            continue
        if entry.name not in allowed_files:
            invalid.append((entry, ["unbound trace-root entry"]))
        continue
    unit_id, layouts, errors = validate_run(entry)
    if errors:
        invalid.append((entry, errors))
    else:
        valid.append((entry, unit_id, layouts))

if invalid:
    archive_parent.mkdir(parents=True, exist_ok=True)
    archive = pathlib.Path(tempfile.mkdtemp(
        prefix=f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-attempt-"
        f"{generation}-priority-trace-quarantine.",
        dir=archive_parent,
    ))
    reason = archive / "archive-reason.json"
    reason.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    entry.name: errors for entry, errors in invalid
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    moved = []
    try:
        for entry, _ in invalid:
            destination = archive / entry.name
            os.replace(entry, destination)
            moved.append((destination, entry))
    except Exception:
        for destination, original in reversed(moved):
            if destination.exists() and not original.exists():
                os.replace(destination, original)
        reason.unlink(missing_ok=True)
        if archive.is_dir() and not any(archive.iterdir()):
            archive.rmdir()
        raise
    print(f"quarantined invalid priority trace entries at {archive}")

counts = Counter(unit_id for _, unit_id, _ in valid)
missing_or_duplicate = {
    unit_id: counts.get(unit_id, 0)
    for unit_id in expected_units
    if counts.get(unit_id, 0) != 1
}
foreign_valid = sorted(set(counts) - set(expected_units))
if missing_or_duplicate or foreign_valid:
    raise SystemExit(
        "priority trace allowlist is incomplete or ambiguous: "
        f"counts={dict(sorted(missing_or_duplicate.items()))}, "
        f"foreign={foreign_valid}"
    )
layouts = set().union(*(item[2] for item in valid)) if valid else set()
if len(layouts) != 1 or None in layouts:
    raise SystemExit(
        f"priority trace parameter-layout count is {len(layouts)}, expected one"
    )

controller_root.mkdir(parents=True, exist_ok=True)
controller_files = sorted(controller_root.glob("*.controller.json"))
controller_errors = []
controller_payloads = []
if len(controller_files) != 1:
    controller_errors.append(
        f"priority controller artifact count is {len(controller_files)}"
    )
for artifact_path in controller_files:
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        controller_payloads.append(payload)
        sidecar = pathlib.Path(str(artifact_path) + ".sha256")
        if (
            not sidecar.is_file()
            or sidecar.read_text(encoding="utf-8").strip()
            != sha256_json(payload)
        ):
            controller_errors.append(
                f"controller hash mismatch: {artifact_path.name}"
            )
        if payload.get("extra", {}).get("parameter_layout_sha256") not in layouts:
            controller_errors.append(
                f"controller/trace layout mismatch: {artifact_path.name}"
            )
    except Exception as exc:
        controller_errors.append(
            f"unreadable controller artifact {artifact_path.name}: {exc}"
        )
if stage == "l3-evaluation" and not controller_errors:
    try:
        from lightcone_spec.replay.real import load_real_replay_records

        expected_map_sha = controller_payloads[0].get("extra", {}).get(
            "transport_map_sha256"
        )
        if not isinstance(expected_map_sha, str) or len(expected_map_sha) != 64:
            raise ValueError("phase-1 controller lacks a bound transport map hash")
        records = load_real_replay_records(
            trace_root, model_pair_id="qwen3_4b_dflash16"
        )
        if not records:
            raise ValueError("L3 evaluation emitted no schema-v3 replay records")
        keys = []
        for record in records:
            key = (
                record.evaluation_pair_id,
                record.evaluation_concurrency,
                record.trace_stage_index,
            )
            keys.append(key)
            if (
                record.provenance_method != "lc_transport"
                or not isinstance(record.evaluation_pair_id, str)
                or not record.evaluation_pair_id
                or record.evaluation_concurrency not in (1, 4)
                or record.trace_capture_sampling != "staged"
                or record.trace_stage_count != 3
                or record.trace_stage_index not in (0, 1, 2)
                or record.transport_map_sha256 != expected_map_sha
                or record.transport_evaluation_contract
                != "joint_fisher_transport_adamw_damping_v1"
                or record.transport_variant != "joint"
                or record.transported_candidate_utility is None
                or record.paired_l2_utility is None
                or record.actual_published_utility is None
            ):
                raise ValueError(
                    "L3 evaluation record violates the frozen joint-utility "
                    "contract"
                )
        if len(keys) != len(set(keys)):
            raise ValueError(
                "L3 evaluation has duplicate request/concurrency/stage pairs"
            )
    except Exception as exc:
        controller_errors.append(f"invalid L3 evaluation evidence: {exc}")
if not (controller_root / "replay_report.json").is_file():
    controller_errors.append("priority controller replay report is missing")

if controller_errors and not require_controller and any(controller_root.iterdir()):
    archive_parent.mkdir(parents=True, exist_ok=True)
    archive = pathlib.Path(tempfile.mkdtemp(
        prefix=f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-attempt-"
        f"{generation}-priority-controller-quarantine.",
        dir=archive_parent,
    ))
    reason = archive / "archive-reason.json"
    reason.write_text(
        json.dumps(
            {"schema_version": 1, "reasons": controller_errors},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    destination = archive / "controller"
    try:
        os.replace(controller_root, destination)
        controller_root.mkdir(parents=True, exist_ok=False)
    except Exception:
        if controller_root.is_dir() and not any(controller_root.iterdir()):
            controller_root.rmdir()
        if destination.exists() and not controller_root.exists():
            os.replace(destination, controller_root)
        reason.unlink(missing_ok=True)
        if archive.is_dir() and not any(archive.iterdir()):
            archive.rmdir()
        raise
    controller_errors = []
    print(f"quarantined partial priority controller at {archive}")
if require_controller and controller_errors:
    raise SystemExit("; ".join(controller_errors))
print("priority trace/controller evidence contract is current")
PY
}

prepare_priority_inputs() {
  local receipt="$PRIORITY_DATASET_RECEIPT"
  mkdir -p \
    "$(dirname -- "$PRIORITY_LOCKFILE")" "$PRIORITY_MODEL_CACHE" \
    "$(dirname -- "$receipt")" || return

  priority_archive_partial_pair "$PRIORITY_LOCKFILE" priority-lock || return
  if [ -s "$PRIORITY_LOCKFILE" ] && [ -s "$PRIORITY_LOCKFILE.sha256" ]; then
    "$PY" -c \
      'from lightcone_spec.locking.lockfile import load_lockfile; import sys; load_lockfile(sys.argv[1])' \
      "$PRIORITY_LOCKFILE" || return
  else
    "$LC" lock \
      --output "$PRIORITY_LOCKFILE" \
      --reuse-inputs-from "$P5_LOCKFILE" \
      --pairs "$PRIORITY_PAIR" \
      --datasets livecodebench math500 mt_bench || return
  fi

  # snapshot_download resumes partial files in the explicit cache.  The roots
  # receipt is content-hashed after every successful verification.
  priority_archive_partial_pair "$PRIORITY_MODEL_ROOTS" priority-model-roots \
    || return
  "$LC" prepare-models \
    --lockfile "$PRIORITY_LOCKFILE" \
    --model-cache "$PRIORITY_MODEL_CACHE" \
    --pairs "$PRIORITY_PAIR" \
    --output "$PRIORITY_MODEL_ROOTS" || return
  "$LC" prepare-datasets \
    --lockfile "$PRIORITY_LOCKFILE" \
    --datasets livecodebench math500 mt_bench \
    --limit 96 \
    --output "$receipt"
}

run_priority_smoke() {
  local analysis="$PRIORITY_ANALYSIS_ROOT/smoke"
  mkdir -p "$analysis" || return
  run_sglang_headline run-manifest \
    --manifest "$PRIORITY_SMOKE_MANIFEST" \
    --artifact-root "$PRIORITY_SMOKE_ROOT" \
    --lockfile "$PRIORITY_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots "$PRIORITY_MODEL_ROOTS" \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods static tts naive_async \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" || return
  quarantine_invalid_priority_runs "$PRIORITY_SMOKE_ROOT" smoke || return
  "$LC" validate-artifacts \
    --artifact-root "$PRIORITY_SMOKE_ROOT" \
    --manifest "$PRIORITY_SMOKE_MANIFEST" \
    --methods static tts naive_async \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --coverage-output "$analysis/coverage.json"
}

run_priority_static_tts() {
  local analysis="$PRIORITY_ANALYSIS_ROOT/static-tts"
  mkdir -p "$analysis" || return
  run_sglang_headline run-manifest \
    --manifest "$PRIORITY_MANIFEST" \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --lockfile "$PRIORITY_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots "$PRIORITY_MODEL_ROOTS" \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods static tts \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" || return
  quarantine_invalid_priority_runs "$PRIORITY_EVAL_ROOT" eval || return
  "$LC" validate-artifacts \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --manifest "$PRIORITY_MANIFEST" \
    --methods static tts \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --coverage-output "$analysis/coverage.json" || return
  "$LC" analyze \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --output-dir "$analysis" \
    --manifest "$PRIORITY_MANIFEST" \
    --methods static tts \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --baseline static
}

run_priority_l0() {
  local analysis="$PRIORITY_ANALYSIS_ROOT/l0-vs-tts"
  mkdir -p "$analysis" || return
  run_sglang_headline run-manifest \
    --manifest "$PRIORITY_MANIFEST" \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --lockfile "$PRIORITY_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots "$PRIORITY_MODEL_ROOTS" \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods naive_async \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" || return
  quarantine_invalid_priority_runs "$PRIORITY_EVAL_ROOT" eval || return
  "$LC" validate-artifacts \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --manifest "$PRIORITY_MANIFEST" \
    --methods static tts naive_async \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --coverage-output "$analysis/coverage.json" || return
  "$LC" analyze \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --output-dir "$analysis" \
    --manifest "$PRIORITY_MANIFEST" \
    --methods static tts naive_async \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --baseline tts
}

run_priority_paired_trace() {
  local analysis="$PRIORITY_ANALYSIS_ROOT/paired-trace"
  mkdir -p "$analysis" || return
  # Do not filter logical delay: every delay cell in the immutable trace
  # manifest is needed for the controller's staleness and paired-TTS gates.
  run_sglang_headline run-manifest \
    --manifest "$PRIORITY_TRACE_MANIFEST" \
    --artifact-root "$PRIORITY_PHASE1_TRACE_ROOT" \
    --lockfile "$PRIORITY_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots "$PRIORITY_MODEL_ROOTS" \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods tts naive_async \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" || return
  "$LC" validate-artifacts \
    --artifact-root "$PRIORITY_PHASE1_TRACE_ROOT" \
    --manifest "$PRIORITY_TRACE_MANIFEST" \
    --methods tts naive_async \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --coverage-output "$analysis/coverage.json"
}

fit_priority_phase1_controller() {
  "$LC" replay \
    --trace-root "$PRIORITY_PHASE1_TRACE_ROOT" \
    --pair "$PRIORITY_PAIR" \
    --transport-rank 16 \
    --seed 0 \
    --output-dir "$PRIORITY_PHASE1_CONTROLLER_ROOT"
}

read_priority_l3_evaluation_ready() {
  local report="$PRIORITY_PHASE1_CONTROLLER_ROOT/replay_report.json"
  [ -s "$report" ] || return 2
  PRIORITY_L3_EVALUATION_READY=$("$PY" - "$report" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
l3 = report.get("l3_gate", {})
exactness = report.get("trace_exactness", {})
artifact = pathlib.Path(report.get("artifact_path", ""))
try:
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
except (OSError, ValueError):
    artifact_payload = {}
map_sha = artifact_payload.get("extra", {}).get("transport_map_sha256")
ready = (
    l3.get("evaluation_ready") is True
    and exactness.get("verified") is True
    and artifact.is_file()
    and isinstance(map_sha, str)
    and len(map_sha) == 64
)
print("1" if ready else "0")
PY
  ) || return
  [ "$PRIORITY_L3_EVALUATION_READY" = 0 ] || \
    [ "$PRIORITY_L3_EVALUATION_READY" = 1 ]
}

run_priority_l3_evaluation() {
  local analysis="$PRIORITY_ANALYSIS_ROOT/l3-evaluation"
  mkdir -p "$analysis" || return
  run_sglang_headline run-manifest \
    --manifest "$PRIORITY_L3_EVALUATION_MANIFEST" \
    --artifact-root "$PRIORITY_L3_EVALUATION_TRACE_ROOT" \
    --lockfile "$PRIORITY_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots "$PRIORITY_MODEL_ROOTS" \
    --controller-root "$PRIORITY_PHASE1_CONTROLLER_ROOT" \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods lc_transport \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" || return
  "$LC" validate-artifacts \
    --artifact-root "$PRIORITY_L3_EVALUATION_TRACE_ROOT" \
    --manifest "$PRIORITY_L3_EVALUATION_MANIFEST" \
    --methods lc_transport \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --coverage-output "$analysis/coverage.json"
}

fit_priority_controller() {
  "$LC" replay \
    --trace-root "$PRIORITY_TRACE_ROOT" \
    --pair "$PRIORITY_PAIR" \
    --transport-rank 16 \
    --seed 0 \
    --output-dir "$PRIORITY_CONTROLLER_ROOT"
}

read_priority_gates() {
  local report="$PRIORITY_CONTROLLER_ROOT/replay_report.json"
  local gates="$SESSION_ROOT/.priority-gates.$$"
  [ -s "$report" ] || return 2
  "$PY" - "$report" > "$gates" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
oracle = report.get("oracle_replay_gate", {})
paired = report.get("tts_paired_gate", {})
learned = report.get("learned_policy_gate", {})
l3 = report.get("l3_gate", {})
exactness = report.get("trace_exactness", {})
flags = (
    oracle.get("l1_eligible") is True,
    paired.get("l1_eligible") is True,
    learned.get("l1_eligible") is True,
    oracle.get("l2_eligible") is True,
    paired.get("l2_eligible") is True,
    learned.get("l2_eligible") is True,
    exactness.get("verified") is True,
    l3.get("enabled") is True,
    l3.get("heldout_transported_utility_gate", {}).get("eligible") is True,
    l3.get("exactness", {}).get("verified") is True,
    l3.get("heldout_transported_utility_gate", {}).get("pairing_contract")
    == "exact_request_seed_concurrency_trace_stage_v1",
)
print("\t".join("1" if flag else "0" for flag in flags))
PY
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$gates"
    return "$rc"
  fi
  IFS=$'\t' read -r PRIORITY_ORACLE_L1 PRIORITY_TTS_L1 \
    PRIORITY_LEARNED_L1 PRIORITY_ORACLE_L2 PRIORITY_TTS_L2 \
    PRIORITY_LEARNED_L2 PRIORITY_TRACE_EXACT \
    PRIORITY_L3 PRIORITY_L3_UTILITY PRIORITY_L3_EXACT \
    PRIORITY_L3_PAIRING < "$gates"
  rm -f "$gates"
  [ -n "$PRIORITY_ORACLE_L1" ] && [ -n "$PRIORITY_TTS_L1" ] && \
    [ -n "$PRIORITY_LEARNED_L1" ] && \
    [ -n "$PRIORITY_ORACLE_L2" ] && [ -n "$PRIORITY_TTS_L2" ] && \
    [ -n "$PRIORITY_LEARNED_L2" ] && \
    [ -n "$PRIORITY_TRACE_EXACT" ] && [ -n "$PRIORITY_L3" ] && \
    [ -n "$PRIORITY_L3_UTILITY" ] && [ -n "$PRIORITY_L3_EXACT" ] && \
    [ -n "$PRIORITY_L3_PAIRING" ]
}

select_priority_methods() {
  PRIORITY_ACTUAL_METHODS=(static tts naive_async)
  PRIORITY_BLOCKED_METHODS=()
  if [ "$PRIORITY_ORACLE_L1" -eq 1 ] && [ "$PRIORITY_TTS_L1" -eq 1 ] && \
      [ "$PRIORITY_LEARNED_L1" -eq 1 ] && \
      [ "$PRIORITY_TRACE_EXACT" -eq 1 ]; then
    PRIORITY_ACTUAL_METHODS+=(lc_gate)
    record p08a_priority_l1 eligible \
      "oracle=$PRIORITY_ORACLE_L1;tts=$PRIORITY_TTS_L1;learned=$PRIORITY_LEARNED_L1;exact=$PRIORITY_TRACE_EXACT"
  else
    PRIORITY_BLOCKED_METHODS+=(lc_gate)
    record p08a_priority_l1 blocked \
      "oracle=$PRIORITY_ORACLE_L1;tts=$PRIORITY_TTS_L1;learned=$PRIORITY_LEARNED_L1;exact=$PRIORITY_TRACE_EXACT"
  fi
  if [ "$PRIORITY_ORACLE_L2" -eq 1 ] && [ "$PRIORITY_TTS_L2" -eq 1 ] && \
      [ "$PRIORITY_LEARNED_L2" -eq 1 ] && \
      [ "$PRIORITY_TRACE_EXACT" -eq 1 ]; then
    PRIORITY_ACTUAL_METHODS+=(lc_damp)
    record p08b_priority_l2 eligible \
      "oracle=$PRIORITY_ORACLE_L2;tts=$PRIORITY_TTS_L2;learned=$PRIORITY_LEARNED_L2;exact=$PRIORITY_TRACE_EXACT"
  else
    PRIORITY_BLOCKED_METHODS+=(lc_damp)
    record p08b_priority_l2 blocked \
      "oracle=$PRIORITY_ORACLE_L2;tts=$PRIORITY_TTS_L2;learned=$PRIORITY_LEARNED_L2;exact=$PRIORITY_TRACE_EXACT"
  fi
  if [ "$PRIORITY_L3" -eq 1 ] && [ "$PRIORITY_L3_UTILITY" -eq 1 ] && \
      [ "$PRIORITY_TRACE_EXACT" -eq 1 ] && \
      [ "$PRIORITY_L3_EXACT" -eq 1 ] && \
      [ "$PRIORITY_L3_PAIRING" -eq 1 ]; then
    PRIORITY_ACTUAL_METHODS+=(lc_transport)
    record p08c_priority_l3 eligible \
      "l3_gate=1;heldout_transported_utility=1;phase1_exact=$PRIORITY_TRACE_EXACT;phase2_exact=$PRIORITY_L3_EXACT;pairing=$PRIORITY_L3_PAIRING"
  else
    PRIORITY_BLOCKED_METHODS+=(lc_transport)
    record p08c_priority_l3 blocked \
      "evidence_insufficient;transport_gate=$PRIORITY_L3;heldout_transported_utility=$PRIORITY_L3_UTILITY;phase1_exact=$PRIORITY_TRACE_EXACT;phase2_exact=$PRIORITY_L3_EXACT;pairing=$PRIORITY_L3_PAIRING"
  fi
}

run_priority_eligible_methods() {
  local -a eligible=()
  local method
  for method in "${PRIORITY_ACTUAL_METHODS[@]:3}"; do
    eligible+=("$method")
  done
  [ "${#eligible[@]}" -gt 0 ] || return 0
  run_sglang_headline run-manifest \
    --manifest "$PRIORITY_MANIFEST" \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --lockfile "$PRIORITY_LOCKFILE" \
    --runtime-root $RUNTIME_ROOT \
    --model-roots "$PRIORITY_MODEL_ROOTS" \
    --controller-root "$PRIORITY_CONTROLLER_ROOT" \
    "${PEAK_TFLOPS_ARGS[@]}" \
    --methods "${eligible[@]}" \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" || return
  quarantine_invalid_priority_runs "$PRIORITY_EVAL_ROOT" eval
}

finalize_priority_analysis() {
  local analysis="$PRIORITY_ANALYSIS_ROOT/final-vs-tts"
  mkdir -p "$analysis" || return
  "$LC" validate-artifacts \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --manifest "$PRIORITY_MANIFEST" \
    --methods "${PRIORITY_ACTUAL_METHODS[@]}" \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --coverage-output "$analysis/coverage.json" || return
  "$LC" analyze \
    --artifact-root "$PRIORITY_EVAL_ROOT" \
    --output-dir "$analysis" \
    --manifest "$PRIORITY_MANIFEST" \
    --methods "${PRIORITY_ACTUAL_METHODS[@]}" \
    --learning-rate "$PRIORITY_LR" \
    --weight-update-mode "$PRIORITY_MODE" \
    --baseline tts
}

write_priority_terminal() {
  local status=complete
  local actual_csv blocked_csv controller_artifact phase1_controller_artifact
  local report report_evidence
  local static_claims final_claims static_acceptance final_acceptance
  local terminal="$SESSION_ROOT/priority-terminal.json"
  report="$PRIORITY_CONTROLLER_ROOT/replay_report.json"
  # The report is both parsed into the terminal gate summary and independently
  # content-hashed as closure evidence.  Keep the two argument roles explicit.
  report_evidence=$report
  static_claims="$PRIORITY_ANALYSIS_ROOT/static-tts/p5_claim_gates.json"
  final_claims="$PRIORITY_ANALYSIS_ROOT/final-vs-tts/p5_claim_gates.json"
  static_acceptance="$PRIORITY_ANALYSIS_ROOT/static-tts/p5_long_context_acceptance.csv"
  final_acceptance="$PRIORITY_ANALYSIS_ROOT/final-vs-tts/p5_long_context_acceptance.csv"
  actual_csv=$(IFS=,; echo "${PRIORITY_ACTUAL_METHODS[*]}")
  blocked_csv=$(IFS=,; echo "${PRIORITY_BLOCKED_METHODS[*]}")
  [ "${#PRIORITY_BLOCKED_METHODS[@]}" -eq 0 ] || status=scientifically_blocked
  controller_artifact=$("$PY" - "$report" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["artifact_path"])
PY
  ) || return
  phase1_controller_artifact=$("$PY" - \
    "$PRIORITY_PHASE1_CONTROLLER_ROOT/replay_report.json" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["artifact_path"])
PY
  ) || return
  "$PY" - "$terminal" "$run_id" "$status" "$PRIORITY_PAIR" \
    "$PRIORITY_MODE" "$PRIORITY_LR" "$actual_csv" "$blocked_csv" \
    "$report" "$PRIORITY_ARTIFACT_CONTRACT" \
    "$PRIORITY_SMOKE_ROOT" "$PRIORITY_EVAL_ROOT" \
    "$PRIORITY_TRACE_ROOT" "$PRIORITY_ANALYSIS_ROOT" \
    "$static_claims" "$final_claims" \
    "$PRIORITY_MANIFEST" "$static_acceptance" "$final_acceptance" \
    "$report_evidence" \
    "$PRIORITY_MANIFEST" "$PRIORITY_TRACE_MANIFEST" \
    "$PRIORITY_L3_EVALUATION_MANIFEST" "$PRIORITY_SMOKE_MANIFEST" \
    "$PRIORITY_CALIBRATION_MANIFEST" \
    "$PRIORITY_CALIBRATION_MANIFEST.sha256" \
    "$PRIORITY_CALIBRATION_SCRIPT" \
    "$PRIORITY_CALIBRATION_SPEC" "$PRIORITY_CALIBRATION_SPEC.sha256" \
    "$PRIORITY_CALIBRATION_READY" "$PRIORITY_CALIBRATION_READY.sha256" \
    "$PRIORITY_LOCKFILE" "$PRIORITY_LOCKFILE.sha256" \
    "$PRIORITY_MODEL_ROOTS" "$PRIORITY_MODEL_ROOTS.sha256" \
    "$PRIORITY_DATASET_RECEIPT" "$PRIORITY_DATASET_RECEIPT.sha256" \
    "$PRIORITY_ARTIFACT_CONTRACT" \
    "$PRIORITY_ARTIFACT_CONTRACT.sha256" \
    "$PRIORITY_ANALYSIS_ROOT/smoke/coverage.json" \
    "$PRIORITY_ANALYSIS_ROOT/paired-trace/coverage.json" \
    "$PRIORITY_ANALYSIS_ROOT/final-vs-tts/coverage.json" \
    "$PRIORITY_PHASE1_CONTROLLER_ROOT/replay_report.json" \
    "$phase1_controller_artifact" "$phase1_controller_artifact.sha256" \
    "$controller_artifact" "$controller_artifact.sha256" <<'PY'
import hashlib
import csv
import json
import math
import pathlib
import sys

(
    terminal_name, run_id, status, pair, mode, lr_text, actual_csv, blocked_csv,
    report_name, epoch_name, smoke_root, eval_root, trace_root, analysis_root,
    static_claims_name, final_claims_name,
    manifest_name, static_acceptance_name, final_acceptance_name,
    *evidence_names
) = sys.argv[1:]
report = json.loads(pathlib.Path(report_name).read_text(encoding="utf-8"))
static_claims = json.loads(
    pathlib.Path(static_claims_name).read_text(encoding="utf-8")
)
final_claims = json.loads(
    pathlib.Path(final_claims_name).read_text(encoding="utf-8")
)
if not isinstance(static_claims, list) or not isinstance(final_claims, list):
    raise SystemExit("priority claim gates must be JSON lists")
manifest = json.loads(pathlib.Path(manifest_name).read_text(encoding="utf-8"))
with open(static_acceptance_name, newline="", encoding="utf-8") as handle:
    static_acceptance = list(csv.DictReader(handle))
with open(final_acceptance_name, newline="", encoding="utf-8") as handle:
    final_acceptance = list(csv.DictReader(handle))
actual_methods = [value for value in actual_csv.split(",") if value]
blocked_methods = [value for value in blocked_csv.split(",") if value]
scientific_block_reasons = [
    f"controller_gate_blocked:{method}" for method in blocked_methods
]
required_contexts = [512, 4096, 16384, 40000]
manifest_contexts = sorted(
    int(value)
    for value in manifest.get("engine_params", {}).get(
        "p5_context_lengths", []
    )
)
if manifest_contexts != required_contexts:
    scientific_block_reasons.append(
        "manifest_context_contract_mismatch:"
        f"expected={required_contexts}:observed={manifest_contexts}"
    )
units = manifest.get("units")
if not isinstance(units, list):
    raise SystemExit("priority manifest units must be a JSON list")
concurrency_by_method = {}
for unit in units:
    method = str(unit.get("method"))
    concurrency_by_method.setdefault(method, set()).add(
        int(unit["concurrency"])
    )
required_adaptation_methods = [
    "naive_async", "lc_gate", "lc_damp", "lc_transport"
]
for method in required_adaptation_methods:
    if method not in actual_methods:
        scientific_block_reasons.append(f"required_method_not_run:{method}")

def integer_field(row, name):
    value = float(row[name])
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"non-integer {name}={row[name]!r}")
    return int(value)

def finite_field(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}={row[name]!r}")
    return value

def validate_method(method, baseline, claim_rows, curve_rows):
    expected_concurrency = sorted(concurrency_by_method.get(method, ()))
    summary = {
        "baseline_method": baseline,
        "expected_concurrencies": expected_concurrency,
        "expected_contexts": required_contexts,
        "aggregate": {},
        "buckets": {},
    }
    reasons = []
    if not expected_concurrency:
        reasons.append(f"manifest_missing_method:{method}")
        return summary, reasons
    for concurrency in expected_concurrency:
        claim_matches = [
            row for row in claim_rows
            if row.get("method") == method
            and row.get("baseline_method") == baseline
            and integer_field(row, "offered_concurrency") == concurrency
        ]
        aggregate_key = f"c{concurrency}"
        if len(claim_matches) != 1:
            reasons.append(
                f"aggregate_coverage:{method}:c{concurrency}:"
                f"found={len(claim_matches)}"
            )
            summary["aggregate"][aggregate_key] = {
                "rows": len(claim_matches), "pass": False
            }
        else:
            claim = claim_matches[0]
            lcag_low = claim.get("lcag_ci_low")
            delta_e = claim.get("mean_delta_acceptance_elasticity")
            aggregate_pass = (
                claim.get("algorithmic_pass") is True
                and claim.get("exactness_pass") is True
                and lcag_low is not None
                and delta_e is not None
                and finite_field(claim, "lcag_ci_low") > 0.0
                and finite_field(
                    claim, "mean_delta_acceptance_elasticity"
                ) < 0.0
            )
            summary["aggregate"][aggregate_key] = {
                "rows": 1,
                "pass": bool(aggregate_pass),
                "algorithmic_pass": claim.get("algorithmic_pass") is True,
                "exactness_pass": claim.get("exactness_pass") is True,
                "lcag_ci_low": lcag_low,
                "mean_delta_acceptance_elasticity": delta_e,
            }
            if not aggregate_pass:
                reasons.append(
                    f"aggregate_gate_failed:{method}:c{concurrency}"
                )
        for context in required_contexts:
            matches = [
                row for row in curve_rows
                if row.get("method") == method
                and row.get("baseline_method") == baseline
                and integer_field(row, "offered_concurrency") == concurrency
                and integer_field(row, "context_length") == context
            ]
            bucket_key = f"c{concurrency}:L{context}"
            if len(matches) != 1:
                reasons.append(
                    f"bucket_coverage:{method}:{bucket_key}:found={len(matches)}"
                )
                summary["buckets"][bucket_key] = {
                    "rows": len(matches), "pass": False
                }
                continue
            row = matches[0]
            try:
                gain = finite_field(row, "acceptance_gain_vs_baseline")
                paired_clusters = integer_field(row, "gain_prompt_clusters")
                version_mismatch = integer_field(row, "version_mismatch_count")
                exactness_violations = integer_field(row, "exactness_violations")
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(
                    f"bucket_invalid:{method}:{bucket_key}:{exc}"
                )
                summary["buckets"][bucket_key] = {"rows": 1, "pass": False}
                continue
            bucket_pass = (
                gain >= -1e-12
                and paired_clusters > 0
                and version_mismatch == 0
                and exactness_violations == 0
            )
            summary["buckets"][bucket_key] = {
                "rows": 1,
                "pass": bool(bucket_pass),
                "acceptance_gain": gain,
                "paired_prompt_clusters": paired_clusters,
                "version_mismatch_count": version_mismatch,
                "exactness_violations": exactness_violations,
            }
            if not bucket_pass:
                reasons.append(
                    f"bucket_gate_failed:{method}:{bucket_key}"
                )
    return summary, reasons

static_tts_summary, reasons = validate_method(
    "tts", "static", static_claims, static_acceptance
)
scientific_block_reasons.extend(reasons)
final_summary = {}
for method in required_adaptation_methods:
    final_summary[method], reasons = validate_method(
        method, "tts", final_claims, final_acceptance
    )
    scientific_block_reasons.extend(reasons)
scientific_block_reasons = list(dict.fromkeys(scientific_block_reasons))
status = "complete" if not scientific_block_reasons else "scientifically_blocked"
epoch_path = pathlib.Path(epoch_name).resolve()
epoch_sha = hashlib.sha256(epoch_path.read_bytes()).hexdigest()
epoch_sidecar = pathlib.Path(str(epoch_path) + ".sha256")
if epoch_sidecar.read_text(encoding="utf-8").strip() != epoch_sha:
    raise SystemExit("priority artifact contract sidecar mismatch")
ledger_names = []
for root_name in (smoke_root, eval_root, trace_root):
    ledger_names.extend(
        str(path) for path in pathlib.Path(root_name).rglob("hashes.json")
    )
ledger_names.extend(
    str(path)
    for path in pathlib.Path(analysis_root).rglob("analysis-hashes.json")
)
evidence = []
for name in dict.fromkeys([
    static_claims_name, final_claims_name, manifest_name,
    static_acceptance_name, final_acceptance_name,
    *evidence_names, *ledger_names
]):
    path = pathlib.Path(name).resolve()
    if not path.is_file():
        raise SystemExit(f"priority closure evidence missing: {path}")
    evidence.append({
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
payload = {
    "schema_version": 1,
    "queue_run_id": run_id,
    "status": status,
    "contract": {
        "model_pair": pair,
        "weight_update_mode": mode,
        "learning_rate": float(lr_text),
        "artifact_epoch_sha256": epoch_sha,
    },
    "actual_methods": actual_methods,
    "blocked_methods": blocked_methods,
    "scientific_block_reasons": scientific_block_reasons,
    "claim_gate_summary": {
        "static_vs_tts": {"tts": static_tts_summary},
        "final_vs_tts": final_summary,
    },
    "coverage_contract": {
        "contexts": required_contexts,
        "concurrency_by_method": {
            method: sorted(values)
            for method, values in sorted(concurrency_by_method.items())
        },
        "per_bucket_rule": (
            "paired acceptance point gain >= 0; paired clusters > 0; "
            "version mismatch = exactness violations = 0"
        ),
        "long_context_rule": (
            "4K-40K algorithmic pass with LCAG CI low > 0 and "
            "acceptance elasticity improvement"
        ),
    },
    "gates": {
        "trace_exactness": report.get("trace_exactness"),
        "oracle_replay_gate": report.get("oracle_replay_gate"),
        "tts_paired_gate": report.get("tts_paired_gate"),
        "learned_policy_gate": report.get("learned_policy_gate"),
        "l3_gate": report.get("l3_gate"),
    },
    "artifact_ledgers": sorted(
        str(pathlib.Path(name).resolve()) for name in ledger_names
    ),
    "evidence": evidence,
}
target = pathlib.Path(terminal_name)
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
temporary.replace(target)
PY
  status=$("$PY" -c \
    'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])' \
    "$terminal") || return
  record priority terminal "$status;methods=$actual_csv;blocked=$blocked_csv"
}

run_priority_chain() {
  local calibration_rc=0
  run_task p00_priority_inputs prepare_priority_inputs || return
  run_task p00a_priority_calibration run_priority_calibration \
    || calibration_rc=$?
  if [ "$calibration_rc" -eq 3 ]; then
    record p00a_priority_calibration scientifically_blocked \
      no_heldout_tts_winner
    return "$PRIORITY_SCIENTIFIC_BLOCKED_RC"
  elif [ "$calibration_rc" -ne 0 ]; then
    return "$calibration_rc"
  fi
  run_task p00b_priority_calibration_winner \
    load_priority_calibration_winner || return
  record p00b_priority_calibration_winner selected \
    "mode=$PRIORITY_MODE;lr=$PRIORITY_LR"
  run_task p00c_priority_evidence_contract \
    archive_stale_priority_evidence || return
  if priority_terminal_closed; then
    record priority skipped terminal_hash_closed
    [ "$(priority_terminal_status)" = complete ] && return 0
    return "$PRIORITY_SCIENTIFIC_BLOCKED_RC"
  fi
  run_task p00d_priority_smoke run_priority_smoke || return
  run_task p01_priority_static_tts run_priority_static_tts || return
  run_task p02_priority_l0 run_priority_l0 || return
  run_task p03_priority_paired_trace run_priority_paired_trace || return
  run_task p03a_priority_trace_contract \
    validate_priority_trace_contract phase1 0 || return
  run_task p04_priority_phase1_replay \
    fit_priority_phase1_controller || return
  run_task p04a_priority_phase1_controller_contract \
    validate_priority_trace_contract phase1 1 || return
  run_task p05_priority_l3_evaluation_readiness \
    read_priority_l3_evaluation_ready || return
  if [ "$PRIORITY_L3_EVALUATION_READY" -eq 1 ]; then
    run_task p06_priority_l3_evaluation \
      run_priority_l3_evaluation || return
    run_task p06a_priority_l3_evaluation_contract \
      validate_priority_trace_contract l3-evaluation 1 || return
  else
    record p06_priority_l3_evaluation blocked \
      phase1_map_or_exactness_not_evaluation_ready
  fi
  run_task p07_priority_final_replay fit_priority_controller || return
  run_task p07a_priority_final_controller_contract \
    validate_priority_trace_contract final 1 || return
  run_task p08_priority_gate_read read_priority_gates || return
  select_priority_methods || return
  run_task p09_priority_eligible_methods run_priority_eligible_methods || return
  run_task p10_priority_final_analysis finalize_priority_analysis || return
  write_priority_terminal || return
  [ "$(priority_terminal_status)" = complete ] && return 0
  return "$PRIORITY_SCIENTIFIC_BLOCKED_RC"
}

overall=0
dflash_ready=1
rank_ready=1
# A scientifically blocked priority terminal is resumable evidence, but it is
# not queue success: stop before q00 and publish owned failure/finished markers
# so the watchdog can power down.  Engineering failures use a separate reason.
priority_rc=0
if [ "$SKIP_LEGACY_PRIORITY" = 1 ]; then
  if [ -z "$RESUME_RECEIPT" ]; then
    echo "LIGHTCONE_RESUME_RECEIPT is required when skipping legacy priority" >&2
    exit 2
  fi
  "$PY" "$FINAL_GATE_TOOL" verify-resume --receipt "$RESUME_RECEIPT" \
    || exit 2
  record priority skipped_verified_final_gate "$RESUME_RECEIPT"
else
  run_priority_chain || priority_rc=$?
fi
if [ "$priority_rc" -eq "$PRIORITY_SCIENTIFIC_BLOCKED_RC" ]; then
  record priority scientifically_blocked stop_before_q00
  record q00_stage1_sweep skipped priority_scientifically_blocked
  record queue failed priority_scientifically_blocked
  write_marker QUEUE_FAILED || exit 2
  write_marker QUEUE_FINISHED || exit 2
  exit 1
elif [ "$priority_rc" -ne 0 ]; then
  record priority engineering_failed fail_closed_before_q00
  record q00_stage1_sweep skipped priority_engineering_failure
  record queue failed priority_engineering_failure
  write_marker QUEUE_FAILED || exit 2
  write_marker QUEUE_FINISHED || exit 2
  exit 1
fi
run_task q00_stage1_sweep run_stage1 || { overall=1; dflash_ready=0; }
if [ "$dflash_ready" -eq 1 ]; then
  run_task q01_stage1_analysis publish_stage1_analysis || { overall=1; dflash_ready=0; }
fi
if [ "$dflash_ready" -eq 1 ]; then
  run_task q02_lora_rank_stage2 run_stage2 || { overall=1; rank_ready=0; }
  run_task q03_cache_and_drift_diagnostics run_diagnostics || overall=1
else
  record q01_stage1_analysis skipped dependency_failed
  record q02_lora_rank_stage2 skipped dependency_failed
  record q03_cache_and_drift_diagnostics skipped dependency_failed
  rank_ready=0
fi

if [ "$dflash_ready" -eq 1 ] && [ "$rank_ready" -eq 1 ]; then
  run_task q04_dflash_8k_16k_32k_40k run_dflash_long_context || overall=1
else
  record q04_dflash_8k_16k_32k_40k skipped dependency_failed
fi

# Measure the real saturation knee independently before the long-context P5
# matrix.  Filtering lc_gate makes this a controller-free 21-unit sweep
# (Static/TTS/L0 x concurrency 1,2,4,8,16,32,48) in its own artifact root.
run_task q04b_sglang_load_saturation run_load_saturation || overall=1

if require_p5_engine_reuse_transition_safe; then
  # First isolate the DSpark TTS-vs-Static question for all three public update
  # representations.  The engine-reuse manifest collapses compatible context
  # buckets into one 15-unit method lane instead of reloading SGLang for 69
  # per-context units.  These runs need no controller.  ``full`` remains the
  # certified cache-safe tail scope; all-drafter DSpark stays fail-closed.
  run_task q05a_dspark_tts_residual \
    run_p5_tts_mode_screen residual static tts || overall=1
  run_task q05b_dspark_tts_lora \
    run_p5_tts_mode_screen lora static tts || overall=1
  run_task q05c_dspark_tts_full_tail \
    run_p5_tts_mode_screen full static tts || overall=1

  # Add residual L0 (15 engine-reuse units) to q05a's shared Static+TTS root.
  # L1/L2 cannot share this invocation: the available controller is
  # identity-bound to stream/lr=1e-2, while P5 resolves to lr=1e-4.
  run_task q05d_dspark_p5_residual_l0 run_p5_residual_l0 || overall=1
else
  record q05a_dspark_tts_residual blocked unsafe_p5_manifest_transition
  record q05b_dspark_tts_lora blocked unsafe_p5_manifest_transition
  record q05c_dspark_tts_full_tail blocked unsafe_p5_manifest_transition
  record q05d_dspark_p5_residual_l0 blocked unsafe_p5_manifest_transition
  overall=1
fi

# Every remote controller artifact predates the canonical mode/layout and
# oracle/paired-TTS gate contract.  Record L1/L2 as blocked instead of trying
# to rename or reuse an incompatible file.  A future canonical stream
# controller must be fitted from real trace/replay evidence before execution.
record q05e_dspark_l1_l2 blocked canonical_stream_controller_artifact_not_available
overall=1

# These remain queued but intentionally non-executable until their recorded
# code/model gates are closed by the next agent.
record q06_dspark_l3 blocked heldout_l3_enable_gate_not_passed
record q07_dflash_eagle_l0_l3 blocked remote_backend_sync_and_checkpoint_gate
overall=1

if [ "$overall" -eq 0 ]; then
  record queue complete
  write_marker QUEUE_COMPLETE || exit 2
else
  record queue failed executable_failure_or_blocked_coverage
  write_marker QUEUE_FAILED || exit 2
fi
write_marker QUEUE_FINISHED || exit 2
exit "$overall"
