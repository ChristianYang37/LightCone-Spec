#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != /* ]]; then
  echo "usage: ./run_paper.sh /absolute/paper.yaml" >&2
  exit 2
fi

project_root="$(cd "$(dirname "$0")" && pwd)"
config_path="$1"

runtime_paths="$(python3 - "$config_path" <<'PY'
import ast
import sys

wanted = {("paths", "sglang_root"), ("server", "python")}
found = {}
section = None
for raw in open(sys.argv[1], encoding="utf-8"):
    if not raw.strip() or raw.lstrip().startswith("#"):
        continue
    indent = len(raw) - len(raw.lstrip(" "))
    text = raw.strip()
    if indent == 0 and text.endswith(":"):
        section = text[:-1]
        continue
    if indent != 2 or ":" not in text:
        continue
    key, value = (part.strip() for part in text.split(":", 1))
    if (section, key) not in wanted:
        continue
    if value[:1] in {'"', "'"}:
        value = ast.literal_eval(value)
    found[(section, key)] = value
missing = wanted - found.keys()
if missing:
    raise SystemExit(f"paper.yaml is missing launcher paths: {sorted(missing)}")
print(found[("paths", "sglang_root")])
print(found[("server", "python")])
PY
)"
sglang_root="${runtime_paths%%$'\n'*}"
python_bin="${runtime_paths#*$'\n'}"
marker="$sglang_root/.lightcone-spec-patched"

if [[ ! -e "$marker" ]]; then
  patches=()
  while IFS= read -r patch; do
    patches+=("$patch")
  done < <(find "$project_root/patches/sglang" -maxdepth 1 -name '*.diff' -print | sort)
  if [[ ${#patches[@]} -eq 0 ]]; then
    echo "no SGLang patches found" >&2
    exit 1
  fi
  git -C "$sglang_root" apply --check "${patches[@]}"
  git -C "$sglang_root" apply "${patches[@]}"
  touch "$marker"
else
  for module in \
    python/sglang/srt/speculative/online_adaptation_runtime.py \
    python/sglang/srt/speculative/dflash_online_adaptation.py \
    python/sglang/srt/managers/native_token_timestamps.py; do
    if [[ ! -f "$sglang_root/$module" ]]; then
      echo "patched SGLang is missing $module; remove $marker and restore a clean checkout" >&2
      exit 1
    fi
  done
  grep -q 'reset_scope' \
    "$sglang_root/python/sglang/srt/speculative/online_adaptation_config.py"
  grep -q 'native_token_timestamp_events' \
    "$sglang_root/python/sglang/srt/managers/native_token_timestamps.py"
fi

exec env PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m lightcone_spec.cli run --config "$config_path"
