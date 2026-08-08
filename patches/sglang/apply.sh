#!/usr/bin/env bash
set -euo pipefail

readonly UPSTREAM_COMMIT="3312645a307453893a00778592f105581e3d1c3d"
readonly EXPECTED_TREE="9e32eb1872a963cb3981710079101fe1f4f8fe0a"
readonly CHECKOUT="${1:?usage: apply.sh <clean-sglang-checkout>}"
readonly PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

cd "$CHECKOUT"
[[ "$(git rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]] || {
    echo "error: SGLang HEAD must be exactly $UPSTREAM_COMMIT" >&2
    exit 1
}
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
    echo "error: SGLang checkout must be clean" >&2
    exit 1
}

while IFS=$'\t' read -r expected patch_name; do
    [[ -n "$expected" && -n "$patch_name" ]] || continue
    actual="$(sha256_file "$PATCH_DIR/$patch_name")"
    [[ "$actual" == "$expected" ]] || {
        echo "error: SHA-256 mismatch for $patch_name" >&2
        exit 1
    }
done < "$PATCH_DIR/SHA256SUMS"

patches=()
while IFS= read -r patch_name; do
    [[ -n "$patch_name" ]] && patches+=("$PATCH_DIR/$patch_name")
done < "$PATCH_DIR/series"
[[ "${#patches[@]}" -eq 7 ]] || {
    echo "error: expected seven patches in series" >&2
    exit 1
}

# Fresh CI and disposable runtime checkouts intentionally have no global Git
# identity.  Preserve each mail patch's author while supplying only the local
# committer identity required by ``git am``; do not mutate user Git config.
git -c user.name="LightCone patch applicator" \
    -c user.email="ChristianYang37@users.noreply.github.com" \
    am "${patches[@]}"
actual_tree="$(git rev-parse 'HEAD^{tree}')"
[[ "$actual_tree" == "$EXPECTED_TREE" ]] || {
    echo "error: patched tree $actual_tree != $EXPECTED_TREE" >&2
    exit 1
}
printf 'Applied %s patches; tree %s\n' "${#patches[@]}" "$actual_tree"
