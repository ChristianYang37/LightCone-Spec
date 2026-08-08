#!/usr/bin/env bash
set -euo pipefail

readonly PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly UPSTREAM_COMMIT="3312645a307453893a00778592f105581e3d1c3d"
source_repo="https://github.com/sgl-project/sglang.git"
run_tests=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) source_repo="${2:?missing value for --source}"; shift 2 ;;
        --skip-tests) run_tests=0; shift ;;
        *) echo "usage: verify.sh [--source REPOSITORY] [--skip-tests]" >&2; exit 2 ;;
    esac
done

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/lightcone-sglang-verify.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
git clone --no-hardlinks "$source_repo" "$work_dir/sglang" >/dev/null
git -C "$work_dir/sglang" checkout --detach "$UPSTREAM_COMMIT" >/dev/null
base_tree="$(git -C "$work_dir/sglang" rev-parse 'HEAD^{tree}')"
"$PATCH_DIR/apply.sh" "$work_dir/sglang"

mapfile_cmd=()
while IFS= read -r file_name; do
    [[ "$file_name" == *.py ]] && mapfile_cmd+=("$work_dir/sglang/$file_name")
done < <(git -C "$work_dir/sglang" diff --name-only "$UPSTREAM_COMMIT" HEAD)
python3 -m py_compile "${mapfile_cmd[@]}"

if [[ "$run_tests" -eq 1 ]]; then
    PYTHONPATH="$work_dir/sglang/python" python3 -m pytest -q \
        "$work_dir/sglang/test/registered/unit/spec/test_speculative_adaptation_preflight.py" \
        "$work_dir/sglang/test/registered/unit/spec/test_tail_adaptation_manager.py" \
        "$work_dir/sglang/test/registered/unit/spec/test_dflash_tail_adaptation.py" \
        "$work_dir/sglang/test/registered/unit/spec/test_eagle_adaptation_protocol.py" \
        "$work_dir/sglang/test/registered/unit/spec/test_eagle_draft_cuda_graph_runner.py" \
        "$work_dir/sglang/test/registered/unit/observability/test_metrics_lifecycle.py"
fi

git -C "$work_dir/sglang" checkout --detach "$UPSTREAM_COMMIT" >/dev/null
[[ "$(git -C "$work_dir/sglang" rev-parse 'HEAD^{tree}')" == "$base_tree" ]]
[[ -z "$(git -C "$work_dir/sglang" status --porcelain --untracked-files=all)" ]]
if [[ "$run_tests" -eq 1 ]]; then
    echo "Patch application, compilation, tests, and reverse checkout passed."
else
    echo "Patch application, compilation, and reverse checkout passed (tests skipped)."
fi
