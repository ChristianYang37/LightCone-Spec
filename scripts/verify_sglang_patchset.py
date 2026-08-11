#!/usr/bin/env python3
"""Verify the pinned SGLang series in a disposable clone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="skip pytest when the host lacks SGLang test dependencies",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    patch_root = repository / "patches" / "sglang"
    manifest = json.loads((patch_root / "manifest.json").read_text(encoding="utf-8"))
    upstream = manifest["upstream"]["commit"]
    registered_files = [entry["file"] for entry in manifest["patches"]]
    series_files = [
        line.strip()
        for line in (patch_root / "series").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if series_files != registered_files:
        raise SystemExit("patch series does not match manifest order")
    artifact_files = sorted(path.name for path in patch_root.glob("*.patch"))
    if artifact_files != sorted(registered_files):
        raise SystemExit("patch directory contains unregistered mail patches")
    source = args.upstream_checkout.resolve()
    if _run("git", "rev-parse", "HEAD", cwd=source) != upstream:
        raise SystemExit("upstream checkout is not at the pinned commit")
    if _run("git", "status", "--porcelain=v1", cwd=source):
        raise SystemExit("upstream checkout is not clean")
    for entry in manifest["patches"]:
        patch = patch_root / entry["file"]
        if _sha256(patch) != entry["sha256"]:
            raise SystemExit(f"patch digest mismatch: {patch.name}")

    with tempfile.TemporaryDirectory(prefix="lightcone-sglang-verify-") as tmp:
        checkout = Path(tmp) / "sglang"
        _run("git", "clone", "--quiet", str(source), str(checkout))
        _run("git", "checkout", "--quiet", upstream, cwd=checkout)
        _run(str(patch_root / "apply.sh"), str(checkout))
        if (
            _run("git", "rev-parse", "HEAD^{tree}", cwd=checkout)
            != manifest["expected_tree"]
        ):
            raise SystemExit("patched tree does not match manifest")
        commits = _run(
            "git", "rev-list", "--reverse", f"{upstream}..HEAD", cwd=checkout
        ).splitlines()
        if len(commits) != len(manifest["patches"]):
            raise SystemExit("applied commit count does not match manifest")
        for entry, commit in zip(manifest["patches"], commits, strict=True):
            changed = sorted(
                _run(
                    "git",
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    commit,
                    cwd=checkout,
                ).splitlines()
            )
            if changed != entry["files"]:
                raise SystemExit(f"patch file list mismatch: {entry['file']}")

        changed_python = sorted(
            {
                file
                for entry in manifest["patches"]
                for file in entry["files"]
                if file.endswith(".py")
            }
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(checkout / "python")
        subprocess.run(
            [os.fspath(Path(os.sys.executable)), "-m", "compileall", "-q"]
            + changed_python,
            cwd=checkout,
            env=env,
            check=True,
        )
        if not args.compile_only:
            subprocess.run(
                [
                    os.fspath(Path(os.sys.executable)),
                    "-m",
                    "pytest",
                    "-q",
                    "test/registered/unit/benchmark/test_serving_output_token_ids.py",
                    "test/registered/unit/spec/test_online_adaptation_protocol.py",
                ],
                cwd=checkout,
                env=env,
                check=True,
            )

        for entry in reversed(manifest["patches"]):
            _run(
                "git",
                "apply",
                "--reverse",
                str(patch_root / entry["file"]),
                cwd=checkout,
            )
        _run("git", "diff", "--exit-code", upstream, cwd=checkout)

    if _run("git", "rev-parse", "HEAD", cwd=source) != upstream:
        raise SystemExit("verification changed the upstream checkout")
    if _run("git", "status", "--porcelain=v1", cwd=source):
        raise SystemExit("verification dirtied the upstream checkout")
    print("SGLang patchset verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
