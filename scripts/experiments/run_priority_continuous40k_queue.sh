#!/usr/bin/env bash
: "${BASH_VERSION:?this queue requires bash}"
set -euo pipefail

QUEUE_SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=${LIGHTCONE_WORKSPACE:-$(cd "$QUEUE_SOURCE_DIR/../.." && pwd)}
RUNTIME_ROOT=${LIGHTCONE_RUNTIME_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lightcone-spec}

# Resume-safe priority lane: one continuous DFlash request, bucketed by the
# real prefix observed before every proposal.  The frozen runner owns logical
# run quarantine/resume; this wrapper only adds phase receipts and analysis.
QUEUE_ROOT=${LIGHTCONE_QUEUE_ROOT:-$RUNTIME_ROOT/queue/continuous-prefix}
PY=${LIGHTCONE_PYTHON:-$RUNTIME_ROOT/venv/bin/python}
REFERENCE_ROOT=${LIGHTCONE_REFERENCE_ROOT:-$RUNTIME_ROOT/reference/dflash}
TARGET=${LIGHTCONE_TARGET_MODEL:-$RUNTIME_ROOT/models/target}
DRAFT=${LIGHTCONE_DRAFT_MODEL:-$RUNTIME_ROOT/models/drafter}
DATASET=${LIGHTCONE_DATASET:-$RUNTIME_ROOT/data/evaluation.jsonl}
SELECTION=${LIGHTCONE_SELECTION:-$RUNTIME_ROOT/selection/optimizer.json}
OUTPUT_ROOT=${LIGHTCONE_OUTPUT_ROOT:-$RUNTIME_ROOT/runs/continuous-prefix}
SAMPLE_INDEX=${LIGHTCONE_SAMPLE_INDEX:-1}
TOTAL_CONTEXT=${LIGHTCONE_TOTAL_CONTEXT:-40960}

HARNESS="$QUEUE_SOURCE_DIR/dflash_tts_reference.py"
SWEEP="$QUEUE_SOURCE_DIR/run_dflash_tts_frozen_sweep.py"
AGGREGATE="$QUEUE_SOURCE_DIR/aggregate_dflash_tts_ablations.py"
SITE_PACKAGES=${LIGHTCONE_REFERENCE_SITE_PACKAGES:-$RUNTIME_ROOT/site-packages}
PYTHONPATH_VALUE="$QUEUE_ROOT:$SITE_PACKAGES:$REFERENCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PREFLIGHT_ROOT="$OUTPUT_ROOT/preflight/sample-$(printf '%04d' "$SAMPLE_INDEX")"
PREFLIGHT_ARTIFACT="$PREFLIGHT_ROOT/artifact"
STATE="$OUTPUT_ROOT/priority-state.jsonl"
COMPLETE="$OUTPUT_ROOT/PRIORITY_COMPLETE.json"
FAILED="$OUTPUT_ROOT/PRIORITY_FAILED.json"

mkdir -p "$OUTPUT_ROOT"

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

preflight_complete() {
  "$PY" - "$PREFLIGHT_ARTIFACT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = root / "summary.json"
rounds_path = root / "rounds.jsonl"
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed = hashlib.sha256(rounds_path.read_bytes()).hexdigest()
    valid = (
        summary.get("status") == "complete_reference_run"
        and summary.get("mode") == "static"
        and summary.get("output", {}).get("rounds_sha256") == observed
        and summary.get("generation", {}).get("num_output_tokens") == 1
    )
except (OSError, ValueError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
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
rm -f "$FAILED"
record queue started "sample=$SAMPLE_INDEX total_context=$TOTAL_CONTEXT"

if ! preflight_complete; then
  if [ -e "$PREFLIGHT_ROOT" ]; then
    archive="$OUTPUT_ROOT/attempt-history/$(date -u +%Y%m%dT%H%M%SZ)-partial-preflight"
    mkdir -p "$(dirname "$archive")"
    mv "$PREFLIGHT_ROOT" "$archive"
    record preflight archived_partial "$archive"
  fi
  mkdir -p "$PREFLIGHT_ARTIFACT"
  record preflight running "static one-token identity lock"
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PYTHONPATH_VALUE" \
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
    --sample-index "$SAMPLE_INDEX" \
    --output-dir "$PREFLIGHT_ARTIFACT" \
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
    --parity-max-new-tokens 1
  preflight_complete
  record preflight complete "$PREFLIGHT_ARTIFACT/summary.json"
fi

record sweep running "static,drafter-lora; stride=1; locked Adam r8 lr=3e-4"
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PYTHONPATH_VALUE" \
  "$PY" "$SWEEP" \
  --python "$PY" \
  --harness "$HARNESS" \
  --selected-optimizer-config "$SELECTION" \
  --reference-root "$REFERENCE_ROOT" \
  --reference-module dflash.model \
  --reference-revision 94e4abc \
  --target-model "$TARGET" \
  --target-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --draft-model "$DRAFT" \
  --draft-revision b74e3a329c4d963783143b1e970d95b002be72bd \
  --dataset "$DATASET" \
  --dataset-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --sample-index "$SAMPLE_INDEX" \
  --preflight-summary "$PREFLIGHT_ARTIFACT/summary.json" \
  --output-root "$OUTPUT_ROOT" \
  --total-contexts "$TOTAL_CONTEXT" \
  --modes static drafter-lora \
  --draft-block-size 16 \
  --mask-token-id 151669 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --temperature 0 \
  --seed 0 \
  --adapter-seed 0 \
  --proximal-lambda 0 \
  --update-stride 1 \
  --position-weighting exponential \
  --position-decay-gamma 7 \
  --loss-reduction weighted-mean \
  --adam-beta1 .9 \
  --adam-beta2 .999 \
  --adam-eps 1e-8 \
  --draft-cache-policy stale \
  --pythonpath "$QUEUE_ROOT" \
  --pythonpath "$SITE_PACKAGES" \
  --pythonpath "$REFERENCE_ROOT" \
  --deterministic \
  --audit-cuda-timing \
  --parity-max-new-tokens 32 \
  --keep-going
record sweep complete "$OUTPUT_ROOT/sample-$(printf '%04d' "$SAMPLE_INDEX")/context-$TOTAL_CONTEXT"

STATIC="$OUTPUT_ROOT/sample-$(printf '%04d' "$SAMPLE_INDEX")/context-$TOTAL_CONTEXT/static/artifact"
LORA="$OUTPUT_ROOT/sample-$(printf '%04d' "$SAMPLE_INDEX")/context-$TOTAL_CONTEXT/drafter-lora/artifact"
ANALYSIS="$OUTPUT_ROOT/continuous-prefix-analysis"
record analysis running "true-prefix 4Ki-token buckets"
env PYTHONPATH="$PYTHONPATH_VALUE" "$PY" "$AGGREGATE" \
  --output-dir "$ANALYSIS" \
  --bucket-size 4096 \
  --parquet \
  "$STATIC" "$LORA"

"$PY" - "$COMPLETE" "$OUTPUT_ROOT" "$ANALYSIS" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

path, root, analysis = map(Path, sys.argv[1:])
manifest = analysis / "dflash_tts_ablation_manifest.json"
payload = {
    "status": "execution_complete_exploratory",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "output_root": str(root),
    "analysis_manifest": str(manifest),
    "analysis_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "trajectory": "one_request_continuous_true_prefix",
    "claim_scope": "single_held_out_prompt_candidate_screen_no_ci",
    "prefix_windows": [
        [start, start + 4096] for start in range(0, 40960, 4096)
    ],
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
record queue complete "$COMPLETE"
finished=true
