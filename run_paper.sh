#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != /* ]]; then
  echo "usage: ./run_paper.sh /absolute/paper.yaml" >&2
  exit 2
fi

project_root="$(cd "$(dirname "$0")" && pwd)"
config_path="$1"
if ! runtime_paths="$(awk '
function value_of(line, value, first, last) {
  value = line
  sub(/^[^:]*:[[:space:]]*/, "", value)
  sub(/[[:space:]]+$/, "", value)
  first = substr(value, 1, 1)
  last = substr(value, length(value), 1)
  if ((first == "\"" && last == "\"") || (first == sprintf("%c", 39) && last == first))
    value = substr(value, 2, length(value) - 2)
  return value
}
/^[^[:space:]#][^:]*:[[:space:]]*$/ {
  section = $0
  sub(/:.*/, "", section)
  next
}
/^  [^[:space:]#][^:]*:/ {
  line = $0
  sub(/^  /, "", line)
  key = line
  sub(/:.*/, "", key)
  if (section == "paths" && key == "sglang_root") sglang = value_of(line)
  if (section == "server" && key == "python") python = value_of(line)
  if (section == "server" && key == "cuda_home") cuda = value_of(line)
}
END {
  if (sglang == "" || python == "" || cuda == "") {
    print "paper.yaml is missing paths.sglang_root, server.python, or server.cuda_home" > "/dev/stderr"
    exit 1
  }
  print sglang
  print python
  print cuda
}
' "$config_path")"; then
  exit 1
fi
sglang_root="${runtime_paths%%$'\n'*}"
remaining="${runtime_paths#*$'\n'}"
python_bin="${remaining%%$'\n'*}"
cuda_home="${remaining#*$'\n'}"
export CUDA_HOME="$cuda_home"
export CUDA_PATH="$cuda_home"
export PATH="$cuda_home/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
marker="$sglang_root/.lightcone-spec-patched"
patch_version="paper-v1-nextn-shadow-v31"

if [[ ! -e "$marker" ]]; then
  patches=()
  while IFS= read -r patch; do
    patches+=("$patch")
  done < <(find "$project_root/patches/sglang" -maxdepth 1 -name '*.diff' -print | sort)
  if [[ ${#patches[@]} -eq 0 ]]; then
    echo "no SGLang patches found" >&2
    exit 1
  fi
  for patch in "${patches[@]}"; do
    sed -n '1,$p' "$patch"
  done | git -C "$sglang_root" apply --recount --check -
  for patch in "${patches[@]}"; do
    sed -n '1,$p' "$patch"
  done | git -C "$sglang_root" apply --recount -
  printf '%s\n' "$patch_version" > "$marker"
else
  if [[ "$(<"$marker")" != "$patch_version" ]]; then
    echo "patched SGLang version changed; restore SGLang and reapply patches" >&2
    exit 1
  fi
  for module in \
    python/sglang/srt/speculative/online_adaptation_runtime.py \
    python/sglang/srt/speculative/dflash_online_adaptation.py \
    python/sglang/srt/speculative/dspark_online_adaptation.py \
    python/sglang/srt/speculative/online_adaptation_config.py \
    python/sglang/srt/models/gemma4_draft.py \
    python/sglang/srt/models/qwen3_eagle3.py \
    python/sglang/srt/models/qwen3_draft_replay.py \
    python/sglang/srt/managers/native_token_timestamps.py; do
    if [[ ! -f "$sglang_root/$module" ]]; then
      echo "patched SGLang is missing $module; remove $marker and restore a clean checkout" >&2
      exit 1
    fi
  done
  grep -q 'request_batched' \
    "$sglang_root/python/sglang/srt/speculative/online_adaptation_config.py"
  grep -q 'native_token_timestamp_events' \
    "$sglang_root/python/sglang/srt/managers/native_token_timestamps.py"
fi

PYTHONPATH="$project_root/src:$sglang_root/python" "$python_bin" - <<'PY'
from lightcone_spec.nextn import MergedPublicationBank, RequestLedger
from sglang.srt.managers.native_token_timestamps import record_committed_output_tokens
from sglang.srt.speculative.dspark_online_adaptation import dspark_composite_loss
from sglang.srt.speculative.dflash_online_adaptation import RequestLoRASlotBank
from sglang.srt.speculative.native_backend_online_adaptation import NativeBackendOnlineAdapter
from sglang.srt.speculative.online_adaptation_config import OnlineAdaptationConfig

assert callable(record_committed_output_tokens)
assert callable(dspark_composite_loss)
assert RequestLoRASlotBank is not None
assert NativeBackendOnlineAdapter is not None
assert OnlineAdaptationConfig is not None
assert MergedPublicationBank is not None
assert RequestLedger is not None
PY

exec env PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m lightcone_spec.cli run --config "$config_path"
