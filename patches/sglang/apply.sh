#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CHECKOUT=${1:-.}
UPSTREAM=3312645a307453893a00778592f105581e3d1c3d
EXPECTED_TREE=22bd0d1d16aab33addbdacdbf75ad5bfe21164a8

git -C "$CHECKOUT" rev-parse --is-inside-work-tree >/dev/null
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$UPSTREAM" || {
  echo "SGLang HEAD must equal $UPSTREAM" >&2
  exit 2
}
test -z "$(git -C "$CHECKOUT" status --porcelain=v1)" || {
  echo "SGLang checkout must be clean, including untracked files" >&2
  exit 2
}

while IFS= read -r patch; do
  test -n "$patch" || continue
  GIT_COMMITTER_NAME="Christian Yang" \
  GIT_COMMITTER_EMAIL="ChristianYang37@users.noreply.github.com" \
    git -C "$CHECKOUT" am "$PATCH_ROOT/$patch"
done < "$PATCH_ROOT/series"

actual_tree=$(git -C "$CHECKOUT" rev-parse HEAD^{tree})
test "$actual_tree" = "$EXPECTED_TREE" || {
  echo "patched tree mismatch: expected $EXPECTED_TREE, got $actual_tree" >&2
  exit 3
}
