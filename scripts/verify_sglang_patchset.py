#!/usr/bin/env python3
"""Verify the pinned SGLang series in a disposable clone."""

from __future__ import annotations

import argparse
import ast
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


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise SystemExit(f"patched SGLang contract is missing callable: {name}")


def _verify_native_terminal_contract(checkout: Path, changed_python: list[str]) -> None:
    hook = "sglang.schema_v3.content_bound_terminal_speculative_evidence.v1"
    terminal = (
        checkout / "python/sglang/srt/speculative/terminal_speculative_evidence.py"
    ).read_text(encoding="utf-8")
    required_terminal_symbols = (
        "class TerminalSpeculativeEvidenceLifecycle",
        "def begin(",
        "def reset(",
        "def finalize(",
        "def fail_closed(",
        "def register_terminal_attestation_signer_provider(",
        "def verify_terminal_attestation(",
        'frozenset({"rejected", "cancelled", "timed_out"})',
        "client_terminal_rows: object",
        '"completion_marker": "TERMINAL_COMPLETE"',
        (
            "SUPPORTED_TERMINAL_METHODS = "
            'frozenset({"target_only", "static", "tts", "l0"})'
        ),
        'attester_id.lower().startswith(("test", "fixture", "cpu"))',
    )
    if hook not in terminal or any(
        symbol not in terminal for symbol in required_terminal_symbols
    ):
        raise SystemExit("native terminal evidence lifecycle contract is incomplete")

    static_worker = (
        checkout / "python/sglang/srt/speculative/dflash_worker_v2.py"
    ).read_text(encoding="utf-8")
    required_static_symbols = (
        "def terminal_static_safety_counters(",
        'self._terminal_static_safety["exactness_violations"] += 1',
        'self._terminal_static_safety["fallbacks"] += 1',
    )
    if any(symbol not in static_worker for symbol in required_static_symbols):
        raise SystemExit("native Static safety instrumentation is incomplete")

    scheduler_source = (checkout / "python/sglang/srt/managers/scheduler.py").read_text(
        encoding="utf-8"
    )
    if (
        '"client_terminal_rows"' not in scheduler_source
        or "native Static safety counters are unavailable" not in scheduler_source
    ):
        raise SystemExit("terminal scheduler reconciliation contract is incomplete")

    server_source = (
        checkout / "python/sglang/srt/entrypoints/http_server.py"
    ).read_text(encoding="utf-8")
    for endpoint in (
        '"/v1/lightcone-spec/terminal-evidence/capability"',
        '"/v1/lightcone-spec/terminal-evidence"',
    ):
        if endpoint not in server_source:
            raise SystemExit("native terminal evidence endpoint is missing")

    serving_path = checkout / "python/sglang/benchmark/serving.py"
    serving_tree = ast.parse(serving_path.read_text(encoding="utf-8"))
    signatures = {
        "async_request_sglang_generate": (
            ["request_func_input", "pbar"],
            ["client_session", "timeout_s"],
        ),
        "async_request_sglang_abort": (
            ["request_id", "base_url"],
            ["client_session", "timeout_s"],
        ),
    }
    for name, (positional, keyword_only) in signatures.items():
        function = _function(serving_tree, name)
        if not isinstance(function, ast.AsyncFunctionDef):
            raise SystemExit(f"official SGLang callable is not async: {name}")
        if [argument.arg for argument in function.args.args] != positional:
            raise SystemExit(f"official SGLang positional signature changed: {name}")
        if [argument.arg for argument in function.args.kwonlyargs] != keyword_only:
            raise SystemExit(
                f"official SGLang client_session signature changed: {name}"
            )
        if any(
            not isinstance(default, ast.Constant) or default.value is not None
            for default in function.args.kw_defaults
        ):
            raise SystemExit(
                f"official SGLang optional session defaults changed: {name}"
            )
    for name in ("open_bench_client_session", "close_bench_client_session"):
        if not isinstance(_function(serving_tree, name), ast.AsyncFunctionDef):
            raise SystemExit(f"official SGLang session lifecycle is not async: {name}")

    for relative in changed_python:
        if relative.startswith("test/"):
            continue
        source = (checkout / relative).read_text(encoding="utf-8", errors="ignore")
        if "-----BEGIN " + "PRIVATE KEY-----" in source:
            raise SystemExit(
                f"patched runtime contains private-key material: {relative}"
            )


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
        _verify_native_terminal_contract(checkout, changed_python)
        if not args.compile_only:
            subprocess.run(
                [
                    os.fspath(Path(os.sys.executable)),
                    "-m",
                    "pytest",
                    "-q",
                    "test/registered/unit/benchmark/test_serving_output_token_ids.py",
                    "test/registered/unit/spec/test_dspark_online_adaptation_contract.py",
                    "test/registered/unit/spec/test_online_adaptation_protocol.py",
                    "test/registered/unit/spec/test_terminal_speculative_evidence.py",
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
