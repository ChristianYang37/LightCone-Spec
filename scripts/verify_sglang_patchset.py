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


def _verify_source_owned_session_reset_contract(checkout: Path) -> None:
    hook = "sglang.schema_v3.source_owned_all_reset_session.v1"
    reset_source = (
        checkout / "python/sglang/srt/speculative/session_reset_evidence.py"
    ).read_text(encoding="utf-8")
    required_reset_symbols = (
        hook,
        "SOURCE_OWNED_SESSION_ACTIONS = frozenset(",
        '"session_capability"',
        '"session_initial_state"',
        '"session_reset_prepare"',
        '"session_reset_finalize"',
        '"session_trace_begin"',
        '"session_trace_ready"',
        '"session_trace_finalized"',
        '"session_close_terminal"',
        'GPU_RESET_SEMANTICS = "PENDING"',
        "CONTINUOUS_CONNECTION_ACCOUNTING_AVAILABLE = True",
        "class SourceOwnedAllResetSessionProducer",
        "def open(",
        "def validate_session_plan_sha256(",
        "def validate_reset_request(",
        "def validate_before_state(",
        "def initial_state_receipt(",
        'finalized_state["connection_accounting"] = accounting',
        "self.initial_state_receipt_value is not None",
        "self.initial_state_receipt_value = initial_receipt",
        "def prepare_reset(",
        "def finalize_reset(",
        "def reset_receipt(",
        "def bind_terminal_begin(",
        "def bind_terminal_reset(",
        "def bind_terminal_finalize(",
        "def close_terminal(",
        '"execution_plan_sha256s"',
        '"native_reset_sha256"',
        '"terminal_receipt_sha256"',
        "source_reset_receipt_sha256 = self.last_reset_receipt_sha256",
        "self.last_reset_receipt_sha256 = None",
        'raise RuntimeError("prior registered trace lifecycle is incomplete")',
        '"lifecycle_closed": True',
        '"transport_close_pending": True',
        '"initial_state_receipt_sha256"',
        '"reset request breaks the source-owned trace chain"',
        '"pre-reset generation breaks the source-owned chain"',
        '"reset transition mismatch"',
        'post["connection_accounting"] = validate_connection_accounting(',
        '"fresh_process_per_trace"',
        '"disabled method allocated adaptation state"',
        '"all-reset did not restore the process baseline"',
    )
    if any(symbol not in reset_source for symbol in required_reset_symbols):
        raise SystemExit("source-owned all-reset producer contract is incomplete")
    open_branch = reset_source.split("    def open(", maxsplit=1)[1].split(
        "    def initial_state_receipt(", maxsplit=1
    )[0]
    if "initial_state_receipt_sha256" in open_branch:
        raise SystemExit("capability snapshot prematurely mints the initial receipt")

    scheduler = (checkout / "python/sglang/srt/managers/scheduler.py").read_text(
        encoding="utf-8"
    )
    required_scheduler_symbols = (
        "whose DFlash\n        # implementation restores the adapter",
        "def _source_owned_session_state(",
        'recv_req.action == "session_capability"',
        'recv_req.action == "session_initial_state"',
        'recv_req.action == "session_reset_prepare"',
        'recv_req.action == "session_reset_finalize"',
        'recv_req.action == "session_trace_begin"',
        'recv_req.action == "session_trace_ready"',
        'recv_req.action == "session_trace_finalized"',
        'recv_req.action == "session_close_terminal"',
        "prior_plan, next_plan = (",
        "before = self.source_owned_session_reset.validate_before_state(",
        '"connection_accounting": connection_accounting',
        'native_reset_sha256=lifecycle.reset_receipt["reset_sha256"]',
        "terminal_receipt_sha256=self._session_last_terminal[",
        'f"fresh_process_required:',
    )
    if any(symbol not in scheduler for symbol in required_scheduler_symbols):
        raise SystemExit("native scheduler all-reset integration is incomplete")
    reset_prepare_branch = scheduler.split(
        'elif recv_req.action == "session_reset_prepare":', maxsplit=1
    )[1].split('elif recv_req.action == "session_reset_finalize":', maxsplit=1)[0]
    prevalidation_index = reset_prepare_branch.index("validate_before_state(")
    mutation_index = reset_prepare_branch.index("_reset_terminal_server_state(")
    if prevalidation_index > mutation_index:
        raise SystemExit("native reset mutates state before structural prevalidation")
    if "_session_reset_request_count" in scheduler:
        raise SystemExit("scheduler counters impersonate HTTP connection accounting")
    close_branch = reset_source.split("    def close_terminal(", maxsplit=1)[1].split(
        "\n\n\n__all__", maxsplit=1
    )[0]
    if (
        '"lifecycle_closed": True' not in close_branch
        or '"transport_close_pending": True' not in close_branch
        or '"connections_current": 0' in close_branch
        or 'accounting["connections_current"] < 1' not in close_branch
    ):
        raise SystemExit("session close receipt fabricates transport termination")

    dflash = (checkout / "python/sglang/srt/speculative/dflash_worker_v2.py").read_text(
        encoding="utf-8"
    )
    if (
        "adapter.reset()" not in dflash
        or "_terminal_static_safety[field] = 0" not in dflash
    ):
        raise SystemExit("native DFlash reset does not restore adapted/Static state")
    adaptation = (
        checkout / "python/sglang/srt/speculative/dflash_online_adaptation.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "self.runtime.reset()",
        "self.optimizer.reset(self.initial_trainable)",
        "self.inference.stage(",
        "self.inference.publish()",
        "def reset_receipt_state(",
        '"master_reset": master_reset',
        '"optimizer_reset": optimizer_reset',
        '"inference_reset": inference_reset',
        '"runtime_reset": runtime_reset',
        '"captured_state_empty": captured_state_empty',
    ):
        if symbol not in adaptation:
            raise SystemExit("native DFlash all-reset coverage is incomplete")

    server = (checkout / "python/sglang/srt/entrypoints/http_server.py").read_text(
        encoding="utf-8"
    )
    generic_handler = server.split(
        "async def terminal_speculative_evidence(", maxsplit=1
    )[1].split("async def _send_source_owned_session_transition(", maxsplit=1)[0]
    generic_guard = "if obj.action in SOURCE_OWNED_SESSION_ACTIONS:"
    generic_dispatch = "tokenizer_manager.terminal_speculative_evidence(obj)"
    if (
        generic_guard not in generic_handler
        or generic_dispatch not in generic_handler
        or generic_handler.index(generic_guard)
        > generic_handler.index(generic_dispatch)
    ):
        raise SystemExit("generic terminal route can inject reserved session actions")
    reset_endpoint = server.split("async def source_owned_session_reset(", maxsplit=1)[
        1
    ].split('@app.get("/get_load")', maxsplit=1)[0]
    for symbol in (
        'action="session_reset_prepare"',
        'action="session_reset_finalize"',
        '"reset_transition_sha256"',
    ):
        if symbol not in reset_endpoint:
            raise SystemExit("HTTP reset lacks two-phase accounting finalization")
    if reset_endpoint.count("_source_owned_http_snapshot()") != 2:
        raise SystemExit("HTTP reset must sample distinct pre/post lifecycle states")
    for endpoint in (
        '"/v1/lightcone-spec/session-reset/capability"',
        '"/v1/lightcone-spec/session-reset/initial-state"',
        '"/v1/lightcone-spec/session-reset"',
        '"/v1/lightcone-spec/session-reset/trace/begin"',
        '"/v1/lightcone-spec/session-reset/trace/reset"',
        '"/v1/lightcone-spec/session-reset/trace/finalize"',
        '"/v1/lightcone-spec/session-reset/close-terminal"',
    ):
        if endpoint not in server:
            raise SystemExit("source-owned session reset endpoint is missing")
    trace_helper = server.split(
        "async def _source_owned_session_trace_transition(", maxsplit=1
    )[1].split('@app.post("/v1/lightcone-spec/session-reset/trace/begin")', maxsplit=1)[
        0
    ]
    terminal_dispatch = "await _send_terminal_speculative_evidence_transition("
    source_bridge = "await _source_owned_session_transition("
    if (
        terminal_dispatch not in trace_helper
        or source_bridge not in trace_helper
        or trace_helper.index(terminal_dispatch) > trace_helper.index(source_bridge)
        or 'payload={"capability_sha256": obj["capability_sha256"]}' not in trace_helper
        or "terminal_sha256" in trace_helper
        or "reset_sha256" in trace_helper
    ):
        raise SystemExit("HTTP trace bridge accepts caller-authored native evidence")
    setup = server.split("def _setup_and_run_http_server(", maxsplit=1)[1].split(
        "def _start_native_grpc_server_for_runtime(", maxsplit=1
    )[0]
    initializer = "initialize_source_owned_http_connection_accounting()"
    if setup.count(initializer) != 1:
        raise SystemExit("HTTP connection accounting lacks one startup initializer")
    if setup.index(initializer) > setup.index("set_global_state("):
        raise SystemExit("HTTP connection accounting initializes after server state")
    if server.count("http=source_owned_uvicorn_http_protocol()") != 2:
        raise SystemExit("single-process uvicorn paths lack lifecycle instrumentation")

    accounting = (
        checkout / "python/sglang/srt/entrypoints/source_owned_http_accounting.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "class SourceOwnedHttpConnectionAccounting",
        "def connection_made(",
        "def connection_lost(",
        "def initialize_source_owned_http_connection_accounting(",
        "def _require_initialized_accounting(",
        '"process_id"',
        '"generation"',
        '"connections_created"',
        '"connections_closed"',
        '"connections_current"',
        "source_owned_uvicorn_http_protocol",
    ):
        if symbol not in accounting:
            raise SystemExit("source-owned HTTP lifecycle accounting is incomplete")
    accounting_tree = ast.parse(accounting)
    global_initializers = [
        node
        for node in accounting_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SOURCE_OWNED_HTTP_CONNECTION_ACCOUNTING"
    ]
    if (
        len(global_initializers) != 1
        or not isinstance(global_initializers[0].value, ast.Constant)
        or global_initializers[0].value.value is not None
    ):
        raise SystemExit("HTTP accounting must not capture a PID at module import")
    for forbidden in ("request.headers", "submitted_requests", "request_count"):
        if forbidden in accounting:
            raise SystemExit("HTTP lifecycle authority depends on request metadata")


def _verify_cpu_native_token_observation_contract(checkout: Path) -> None:
    hook = checkout / "python/sglang/srt/managers/native_token_timestamps.py"
    source = hook.read_text(encoding="utf-8")
    for symbol in (
        "sglang.schema_v3.native_per_token_timestamp.v1",
        "cpu_committed_token_observed_at_streamer_v1",
        'NATIVE_TOKEN_TIMESTAMP_RELEASE_STATUS = "CPU_CONTRACT_ONLY"',
        "not a CUDA event",
        "req.output_ids_through_stop",
        "time.monotonic_ns",
    ):
        if symbol not in source:
            raise SystemExit("CPU-only native token observation contract is incomplete")
    serving = (checkout / "python/sglang/benchmark/serving.py").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "_merge_sglang_native_token_timestamp_events",
        "_summarize_sglang_client_itl_coverage",
        '"supported_values": None if unsupported_reason is not None else values',
        "incomplete_client_interval_coverage",
        "itl_expected_intervals",
        "itl_observed_intervals",
        "itl_coalesced_intervals",
        "itl_missing_intervals",
        "itl_coverage",
        "itl_unsupported_reason",
        "mean_itl_ms=mean_itl_ms",
        "p99_itl_ms=p99_itl_ms",
    ):
        if symbol not in serving:
            raise SystemExit("serving benchmark lacks exact ITL coverage semantics")
    for forbidden in (
        "adjust_itl = chunk_gap / num_new_tokens",
        "mean_itl_ms=np.mean(itls or 0)",
        "p99_itl_ms=np.percentile(itls or 0, 99)",
    ):
        if forbidden in serving:
            raise SystemExit("serving benchmark can fabricate or sparsely reduce ITL")


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
        _verify_source_owned_session_reset_contract(checkout)
        _verify_cpu_native_token_observation_contract(checkout)
        subprocess.run(
            [
                os.fspath(Path(os.sys.executable)),
                "-m",
                "pytest",
                "-q",
                "test/registered/unit/spec/test_session_reset_evidence.py",
                "test/registered/unit/spec/test_source_owned_http_accounting.py",
                "test/registered/unit/spec/test_terminal_speculative_evidence.py",
            ],
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
                    "test/registered/unit/benchmark/test_native_token_timestamps.py",
                    "test/registered/unit/spec/test_dspark_online_adaptation_contract.py",
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
