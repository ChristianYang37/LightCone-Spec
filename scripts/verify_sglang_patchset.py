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

_ADAPTIVE_METHODS = frozenset(
    {
        "tts",
        "l0",
        "onlinespec_ogd",
        "onlinespec_opt",
        "onlinespec_ens",
    }
)
_SUPPORTED_METHODS = frozenset({"target_only", "static", *_ADAPTIVE_METHODS})


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


def _literal_string_frozenset(tree: ast.Module, name: str) -> frozenset[str]:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
            and len(node.value.args) == 1
            and not node.value.keywords
            and isinstance(node.value.args[0], ast.Set)
        ):
            values = node.value.args[0].elts
            if not all(
                isinstance(value, ast.Constant) and isinstance(value.value, str)
                for value in values
            ):
                break
            return frozenset(str(value.value) for value in values)
    raise SystemExit(f"patched SGLang contract lacks literal frozenset: {name}")


def _verify_native_terminal_contract(checkout: Path, changed_python: list[str]) -> None:
    hook = "sglang.schema_v3.content_bound_terminal_speculative_evidence.v2"
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
        "ADAPTIVE_TERMINAL_METHODS = frozenset(",
        "SUPPORTED_TERMINAL_METHODS = frozenset(",
        "REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256 = hashlib.sha256(",
        '"request_source_point_resets"',
        '"serialized_native_scheduler_v1"',
        '"state_untouched"',
        'attester_id.lower().startswith(("test", "fixture", "cpu"))',
    )
    if hook not in terminal or any(
        symbol not in terminal for symbol in required_terminal_symbols
    ):
        raise SystemExit("native terminal evidence lifecycle contract is incomplete")
    terminal_tree = ast.parse(terminal)
    if (
        _literal_string_frozenset(terminal_tree, "ADAPTIVE_TERMINAL_METHODS")
        != _ADAPTIVE_METHODS
        or _literal_string_frozenset(terminal_tree, "SUPPORTED_TERMINAL_METHODS")
        != _SUPPORTED_METHODS
    ):
        raise SystemExit("native terminal method closure is not exact")

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
        or "def _terminal_verified_drafts(" not in scheduler_source
        or "speculative_num_draft_tokens is required for speculative metrics"
        not in scheduler_source
        or "verified_drafts = self._terminal_verified_drafts(target_calls)"
        not in scheduler_source
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
    for symbol in (
        '"supported_methods": sorted(SUPPORTED_TERMINAL_METHODS)',
        "active_method in SUPPORTED_TERMINAL_METHODS",
        "else adaptation.method",
    ):
        if symbol not in server_source:
            raise SystemExit(
                "native terminal live endpoint method closure is incomplete"
            )

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
        'GPU_RESET_SEMANTICS = "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"',
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
    reset_tree = ast.parse(reset_source)
    if (
        _literal_string_frozenset(reset_tree, "ADAPTIVE_SESSION_METHODS")
        != _ADAPTIVE_METHODS
        or _literal_string_frozenset(reset_tree, "SUPPORTED_SESSION_METHODS")
        != _SUPPORTED_METHODS
    ):
        raise SystemExit("source-owned all-reset method closure is not exact")
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
        "sglang.schema_v3.native_per_token_timestamp.v2",
        "scheduler_committed_token_at_result_processor_v1",
        (
            "NATIVE_TOKEN_TIMESTAMP_RELEASE_STATUS = "
            '"IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"'
        ),
        "def record_scheduler_committed_output_tokens(",
        "def require_scheduler_committed_output_tokens(",
        "req.output_ids_through_stop",
        "time.monotonic_ns",
    ):
        if symbol not in source:
            raise SystemExit("native token commit/diagnostic contract is incomplete")
    tree = ast.parse(source)
    recorder = _function(tree, "record_scheduler_committed_output_tokens")
    recorder_source = ast.get_source_segment(source, recorder) or ""
    for forbidden in (".synchronize(", ".cpu(", ".item(", ".tolist("):
        if forbidden in recorder_source:
            raise SystemExit("native scheduler commit recorder blocks the hot path")
    processor = (
        checkout
        / "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
    ).read_text(encoding="utf-8")
    streamer = (
        checkout / "python/sglang/srt/managers/scheduler_components/output_streamer.py"
    ).read_text(encoding="utf-8")
    if processor.count("record_scheduler_committed_output_tokens(") != 2:
        raise SystemExit("native scheduler commit recorder is not exact")
    if (
        "require_scheduler_committed_output_tokens(req)" not in streamer
        or "observe_committed_output_tokens(req)" in streamer
    ):
        raise SystemExit("formal output path falls back to streamer timestamps")
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


def _verify_source_owned_compile_contract(checkout: Path) -> None:
    source = (
        checkout / "python/sglang/srt/speculative/compile_cache_evidence.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "sglang.schema_v3.source_owned_compile_cache_lifecycle.v1",
        (
            "SOURCE_OWNED_COMPILE_RELEASE_STATUS = "
            '"IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"'
        ),
        'GPU_COMPILE_SEMANTICS = "MEASURED_PENDING_PROOF"',
        "GPU_COMPILE_REASON = None",
        "class SourceOwnedCompileCacheProducer",
        "def begin(",
        "def note_request_terminal(",
        "def finalize(",
        '"patched_sglang_tree"',
        '"assignment_plan_sha256"',
        '"compile_key_sha256"',
        '"model_lock_sha256"',
        '"sampling_profile_sha256"',
        '"prewarm_manifest_sha256"',
        '"physical_assignment_sha256"',
        '"experiment_budget_sha256"',
        '"inventory_sha256"',
        '"ordered_terminals"',
        '"begin_state"',
        '"final_state"',
        '"gpu_measurements": measurements',
        '"completion_marker": "GPU_COMPILE_COMPLETE"',
    ):
        if symbol not in source:
            raise SystemExit("source-owned compile producer contract is incomplete")

    server = (checkout / "python/sglang/srt/entrypoints/http_server.py").read_text(
        encoding="utf-8"
    )
    for endpoint in (
        '"/v1/lightcone-spec/compile-cache-evidence/begin"',
        '"/v1/lightcone-spec/compile-cache-evidence/finalize"',
    ):
        if endpoint not in server:
            raise SystemExit("source-owned compile endpoint is missing")
    generic = server.split("async def terminal_speculative_evidence(", maxsplit=1)[
        1
    ].split("async def _send_terminal_speculative_evidence_transition(", maxsplit=1)[0]
    if "SOURCE_OWNED_COMPILE_ACTIONS" not in generic:
        raise SystemExit("generic terminal route accepts reserved compile actions")
    begin = server.split("async def source_owned_compile_begin(", maxsplit=1)[1].split(
        '@app.post("/v1/lightcone-spec/compile-cache-evidence/finalize")',
        maxsplit=1,
    )[0]
    for forbidden in (
        "cache_hits",
        "cache_misses",
        "jit_time_ns",
        "graph_capture_count",
        "graph_replay_count",
        "cache_write_count",
        "active_requests",
        "queued_requests",
        "completed",
    ):
        if forbidden in begin:
            raise SystemExit("compile begin accepts caller-authored source state")

    scheduler = (checkout / "python/sglang/srt/managers/scheduler.py").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "def _source_owned_compile_scheduler_state(",
        "compile_producer.note_request_terminal(",
        "requested_output_tokens=req.sampling_params.max_new_tokens",
        "sampling_seed=req.sampling_params.sampling_seed",
        '"compile_terminal",',
        "SourceOwnedCompileCacheProducer(",
        "self._source_owned_compile_scheduler_state()",
    ):
        if symbol not in scheduler:
            raise SystemExit("native scheduler compile integration is incomplete")


def _verify_source_owned_failure_actuator_contract(checkout: Path) -> None:
    """Freeze the E5 child endpoint, exact scenarios, and rank evidence hooks."""

    runtime_path = checkout / (
        "python/sglang/srt/speculative/failure_actuator_runtime.py"
    )
    test_path = checkout / (
        "test/registered/unit/spec/test_failure_actuator_runtime.py"
    )
    if not runtime_path.is_file() or not test_path.is_file():
        raise SystemExit("source-owned failure actuator module/tests are missing")
    runtime = runtime_path.read_text(encoding="utf-8")
    test_source = test_path.read_text(encoding="utf-8")
    scenarios = {
        "queue_saturation",
        "cancellation",
        "duplicate_retry",
        "nonfinite_candidate",
        "oom_candidate",
        "evidence_backpressure",
        "disk_quota",
        "slow_rank",
        "communicator_failure",
        "replica_drain",
        "replica_restart",
    }
    tree = ast.parse(runtime)
    scenario_keys: set[str] | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_SCENARIOS"
            and isinstance(node.value, ast.Dict)
        ):
            scenario_keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            break
    if scenario_keys != scenarios:
        raise SystemExit("failure actuator scenarios are not exactly the registered 11")
    for symbol in (
        "class SourceOwnedFailureActuatorRuntime",
        "def capability(",
        "def fail_closed(",
        "def _revalidate_live_identity(",
        'Path(f"/proc/{process_id}/stat")',
        "os.getpgid(0)",
        "CUDA_VISIBLE_DEVICES",
        "LIGHTCONE_FAILURE_ACTUATOR_QUOTA_ROOT_BASE",
        "torch.cuda.OutOfMemoryError",
        "socket.socketpair()",
        "os.fsync(descriptor)",
        "start_new_session=True",
        '"tp1_dp1": 1',
        '"tp2_dp1": 2',
        '"tp1_dp2": 2',
    ):
        if symbol not in runtime:
            raise SystemExit(f"failure actuator runtime lacks {symbol}")
    if "pytest.skip" in test_source or "pytest.mark.skip" in test_source:
        raise SystemExit("failure actuator CPU contract may not skip")

    io_source = (checkout / "python/sglang/srt/managers/io_struct.py").read_text(
        encoding="utf-8"
    )
    scheduler = (checkout / "python/sglang/srt/managers/scheduler.py").read_text(
        encoding="utf-8"
    )
    server = (checkout / "python/sglang/srt/entrypoints/http_server.py").read_text(
        encoding="utf-8"
    )
    for action in ('"failure_actuator_capability"', '"failure_actuator"'):
        if action not in io_source or action not in scheduler:
            raise SystemExit(f"failure actuator dispatch lacks {action}")
    for symbol in (
        "SourceOwnedFailureActuatorRuntime.from_environment",
        "torch.distributed.all_gather_object",
        "group=self.tp_cpu_group",
        "runtime.fail_closed()",
    ):
        if symbol not in scheduler:
            raise SystemExit(f"failure actuator scheduler hook lacks {symbol}")
    for symbol in (
        '@app.get("/v1/lightcone-spec/failure-actuator/capability")',
        '@app.post("/v1/lightcone-spec/failure-actuator")',
        "failure actuator target rank differs",
        "rank_bindings.sort",
    ):
        if symbol not in server:
            raise SystemExit(f"failure actuator HTTP hook lacks {symbol}")


def _verify_native_release_readiness_contract(checkout: Path) -> None:
    serving = (checkout / "python/sglang/benchmark/serving.py").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "native_token_timestamp_result_pointer: Optional[Dict[str, Any]]",
        "def _merge_sglang_native_token_timestamp_result_pointer(",
        '!= "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"',
        "output.native_token_timestamp_result_pointer = (",
    ):
        if symbol not in serving:
            raise SystemExit("native ITL result pointer transport is incomplete")

    formal_gang = (
        checkout / "python/sglang/srt/speculative/formal_gang_serving.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "class SourceOwnedFormalGangRuntime",
        "def aggregate_formal_gang_rank_responses(",
        '"tp2": "all_rank_prepare_then_atomic_commit_or_zero_rank_abort"',
        '"dp2": "sticky_cohort_disjoint_request_partition_no_gradient_collective"',
        '"native_itl_semantics": native_token_timestamp_semantics',
        '"native_itl_event_count": len(native_itl_events)',
        '"native_itl_events_sha256": content_sha256(native_itl_events)',
        '"published_ranks": [0, 1] if complete else []',
        '"cross_replica_gradient_collective": False',
    ):
        if symbol not in formal_gang:
            raise SystemExit("formal distributed serving producer is incomplete")
    scheduler = (checkout / "python/sglang/srt/managers/scheduler.py").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "native_token_timestamp_semantics=getattr(",
        "native_token_timestamp_events=tuple(",
        "req.native_token_timestamp_events[",
    ):
        if symbol not in scheduler:
            raise SystemExit("formal gang terminal lacks native ITL commit lineage")
    server = (checkout / "python/sglang/srt/entrypoints/http_server.py").read_text(
        encoding="utf-8"
    )
    for endpoint in (
        '"/v1/lightcone-spec/formal-gang/capability"',
        '"/v1/lightcone-spec/formal-gang/begin"',
        '"/v1/lightcone-spec/formal-gang/reset"',
        '"/v1/lightcone-spec/formal-gang/finalize"',
    ):
        if endpoint not in server:
            raise SystemExit("formal distributed serving endpoint is missing")

    release = (
        checkout / "python/sglang/srt/speculative/native_runtime_release.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "class NativeRuntimeQualificationBootstrap",
        "class Tp2AllRankPublisher",
        "class Dp2StickyReplicaRuntime",
        "class DSparkSelectorAuthority",
        "class DSparkNativeBatch",
        "class Eagle3OfficialSelectorBinding",
        "class Eagle3NativeBatch",
        "class NextNNativeBatch",
        "class NativeHotPathResultPointer",
        "class NativeQualificationRankLifecycle",
        "def publish_preflight_rank_terminals(",
        "def qualification_rank_publication_hook_provider(",
        "LIGHTCONE_PREFLIGHT_ASSIGNMENT_SHA256",
        "LIGHTCONE_PREFLIGHT_RUNNER_PROTOCOL_SHA256",
        "LIGHTCONE_PREFLIGHT_RANK{rank}_TERMINAL_PATH",
        '"formal_exactness_raw_rank_terminal"',
    ):
        if symbol not in release:
            raise SystemExit("native release readiness implementation is incomplete")

    qualification = checkout / (
        "test/registered/unit/spec/test_formal_preflight_gpu_qualification.py"
    )
    tree = ast.parse(qualification.read_text(encoding="utf-8"))
    observed = tuple(
        sorted(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    )
    expected = (
        "test_dp2_sticky_replica_isolation",
        "test_dspark_real_serving_native_heads",
        "test_graph_fixed_address_no_host_sync",
        "test_native_itl_full_token_coverage",
        "test_nextn_tp1_real_serving",
        "test_nextn_tp2_real_serving",
        "test_tp2_one_rank_abort_zero_partial_publication",
        "test_tp2_shard_reference_parity",
    )
    if observed != expected:
        raise SystemExit("native release qualification test set is not exact")
    qualification_source = qualification.read_text(encoding="utf-8")
    if (
        "pytest.skip" in qualification_source
        or "pytest.mark.skip" in qualification_source
    ):
        raise SystemExit("native release qualification may not skip GPU coverage")
    for symbol in (
        'dist.init_process_group(\n        "nccl"',
        "publish_preflight_rank_terminals(_lifecycles())",
        'raise AssertionError("formal qualification requires exactly two visible GPUs")',
    ):
        if symbol not in qualification_source:
            raise SystemExit("native release GPU qualification is incomplete")

    live_helper = checkout / (
        "test/registered/unit/spec/lightcone_live_qualification.py"
    )
    live_helper_source = live_helper.read_text(encoding="utf-8")
    for symbol in (
        "class LiveQualificationHarness",
        "subprocess.Popen(",
        '"/generate"',
        '"/server_info"',
        "native_token_timestamp_result_pointer",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "os.getpgid(pid) != self.process.pid",
        "assigned GPU has a foreign compute process",
        '"source_owned_native_runtime_live_observation"',
        '"actual_sglang_server": True',
        '"component_only": False',
    ):
        if symbol not in live_helper_source:
            raise SystemExit("native live-server qualification harness is incomplete")
    live_suites = {
        "test_chronobelief_gpu_parity_qualification.py": (
            "test_chronobelief_fp16_gpu_parity",
            "test_chronobelief_bf16_gpu_parity",
            "test_chronobelief_fp32_gpu_parity",
            "test_chronobelief_safe_boundary_age_exactness",
            "test_chronobelief_abort_no_state_advance",
            "test_chronobelief_skip_no_state_advance",
            "test_chronobelief_commit_once_decoupled_decay",
            "test_chronobelief_nonfinite_overflow_fail_closed",
        ),
        "test_dspark_live_gpu_qualification.py": (
            "test_dspark_real_predecessor",
            "test_dspark_markov_w1_w2",
            "test_dspark_confidence_head",
            "test_dspark_56_scope_selector",
            "test_dspark_fixed_budget",
            "test_dspark_native_scheduler",
            "test_native_itl_pointer",
            "test_graph_fixed_address_no_host_sync",
        ),
        "test_dspark_tp2_live_gpu_qualification.py": (
            "test_dspark_tp2_real_predecessor",
            "test_dspark_tp2_native_heads",
            "test_dspark_tp2_selector",
            "test_dspark_tp2_all_rank_prepare",
            "test_dspark_tp2_two_phase_commit",
            "test_dspark_tp2_one_rank_abort_zero_partial",
            "test_native_itl_pointer",
            "test_graph_fixed_address_no_host_sync",
        ),
        "test_dspark_dp2_live_gpu_qualification.py": (
            "test_dspark_dp2_real_predecessor",
            "test_dspark_dp2_native_heads",
            "test_dspark_dp2_selector",
            "test_dspark_dp2_sticky_routing",
            "test_dspark_dp2_replica_state_isolation",
            "test_dspark_dp2_zero_cross_replica_gradient",
            "test_native_itl_pointer",
            "test_graph_fixed_address_no_host_sync",
        ),
        "test_eagle3_live_gpu_qualification.py": (
            "test_eagle3_official_selector_binding",
            "test_eagle3_target_revision_binding",
            "test_eagle3_drafter_revision_binding",
            "test_eagle3_interface_binding",
            "test_eagle3_source_commit_binding",
            "test_eagle3_live_scheduler_boundary",
            "test_native_itl_pointer",
            "test_graph_fixed_address_no_host_sync",
        ),
        "test_native_hot_path_live_gpu_qualification.py": (
            "test_native_itl_full_token_coverage",
            "test_native_itl_monotonic_clock",
            "test_native_itl_pointer_stability",
            "test_cuda_stream_event_dependency",
            "test_graph_input_pointer_stability",
            "test_graph_candidate_pointer_stability",
            "test_no_blocking_d2h",
            "test_no_host_synchronize",
        ),
        "test_nextn_tp1_live_gpu_qualification.py": (
            "test_nextn_mtp_hidden_interface",
            "test_nextn_teacher_rows",
            "test_nextn_valid_mask",
            "test_nextn_source_adapter_version",
            "test_nextn_target_model_authority",
            "test_nextn_drafter_model_authority",
            "test_native_itl_pointer",
            "test_graph_fixed_address_no_host_sync",
        ),
        "test_nextn_tp2_live_gpu_qualification.py": (
            "test_nextn_mtp_hidden_interface_tp2",
            "test_nextn_teacher_mask_tp2",
            "test_nextn_source_adapter_version_tp2",
            "test_nextn_target_two_shard_authority",
            "test_nextn_drafter_two_shard_authority",
            "test_nextn_tp2_sharded_candidate_parity",
            "test_native_itl_pointer",
            "test_graph_fixed_address_no_host_sync",
        ),
        "test_tp1_dp2_live_gpu_qualification.py": (
            "test_dp2_two_worker_launch",
            "test_dp2_sticky_routing",
            "test_dp2_replica_state_isolation",
            "test_dp2_zero_cross_replica_gradient",
            "test_dp2_failure_isolation",
            "test_dp2_rank_terminal_evidence",
            "test_native_itl_pointer",
            "test_graph_fixed_address_no_host_sync",
        ),
        "test_tp2_dp1_live_gpu_qualification.py": (
            "test_tp2_nccl_rank_coverage",
            "test_tp2_sharded_candidate_parity",
            "test_tp2_all_rank_prepare",
            "test_tp2_two_phase_commit",
            "test_tp2_one_rank_abort_zero_partial",
            "test_tp2_rank_terminal_evidence",
            "test_native_itl_pointer",
            "test_graph_fixed_address_no_host_sync",
        ),
    }
    for filename, expected_live_tests in live_suites.items():
        source = (checkout / "test/registered/unit/spec" / filename).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        observed_live_tests = tuple(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
        if observed_live_tests != expected_live_tests:
            raise SystemExit(f"{filename} live qualification test set is not exact")
        if "pytest.skip" in source or "pytest.mark.skip" in source:
            raise SystemExit(f"{filename} may not skip live GPU coverage")
        if "LiveQualificationHarness" not in source:
            raise SystemExit(f"{filename} does not use the source-owned live harness")

    session_qualification = checkout / (
        "test/registered/unit/spec/test_session_reset_gpu_qualification.py"
    )
    session_tree = ast.parse(session_qualification.read_text(encoding="utf-8"))
    observed_session = tuple(
        node.name
        for node in session_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )
    expected_session = (
        "test_session_reuse_token_trajectory",
        "test_session_reset_state_receipt",
        "test_session_hbm_drift",
        "test_session_graph_pointer_stability",
        "test_session_startup_latency_comparison",
        "test_session_http_connection_accounting",
        "test_session_fault_fallback",
        "test_session_terminal_lifecycle",
    )
    if observed_session != expected_session:
        raise SystemExit("session reset qualification test set is not exact")
    session_source = session_qualification.read_text(encoding="utf-8")
    for forbidden in ("pytest.skip", "pytest.mark.skip"):
        if forbidden in session_source:
            raise SystemExit("session reset qualification may not skip GPU coverage")
    for symbol in (
        "torch.cuda.device_count() != 1",
        '"/v1/lightcone-spec/session-reset/capability"',
        '"/v1/lightcone-spec/session-reset/trace/finalize"',
        '"/v1/lightcone-spec/session-reset/close-terminal"',
        '"formal_session_reset_raw_rank_terminal"',
        "LIGHTCONE_SESSION_RANK0_TERMINAL_PATH",
    ):
        if symbol not in session_source:
            raise SystemExit("session reset live GPU qualification is incomplete")

    online = (
        checkout / "python/sglang/srt/speculative/native_backend_online_adaptation.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "source_version=source_version",
        "self.optimizer.propose(",
        '"feedback_source_version": feedback_source_version',
        '"safe_boundary_version": source_version',
        'if self.config.optimizer.name == "chronobelief"',
    ):
        if symbol not in online:
            raise SystemExit("native ChronoBelief live wiring is incomplete")
    optimizer_runtime = (
        checkout / "python/sglang/srt/speculative/online_adaptation_runtime.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "class SafeBoundaryAgeAuthority",
        "def derive(",
        "safe_boundary_age=safe_boundary_version - source_version",
        "safe_boundary_age=age_authority.safe_boundary_age",
    ):
        if symbol not in optimizer_runtime:
            raise SystemExit("native ChronoBelief age derivation is incomplete")

    for relative, symbols in (
        (
            "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py",
            ("maybe_build_native_backend_adapter", "DSparkNativeBatch"),
        ),
        (
            "python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py",
            ("maybe_build_native_backend_adapter", "NextNNativeBatch"),
        ),
        (
            "python/sglang/srt/speculative/eagle_worker_v2.py",
            (
                "maybe_build_native_backend_adapter",
                "Eagle3NativeBatch",
                "eagle3_selector_binding",
                "torch.enable_grad() if online_update",
            ),
        ),
    ):
        worker = (checkout / relative).read_text(encoding="utf-8")
        if any(symbol not in worker for symbol in symbols):
            raise SystemExit("native adaptive worker release hook is incomplete")


def _verify_gpu_qualification_collection(
    checkout: Path, *, environment: dict[str, str]
) -> None:
    """Import and collect the exact non-skippable GPU qualification files.

    AST checks above freeze the node names and reject explicit skip calls.  A
    real pytest collection additionally proves that the patched modules and
    their source-owned harness import together before an expensive GPU host is
    allocated.  Collection is deliberately limited to the eight-test formal
    exactness suite plus the eleven required eight-test runtime suites.
    """

    files = (
        "test/registered/unit/spec/test_formal_preflight_gpu_qualification.py",
        "test/registered/unit/spec/test_chronobelief_gpu_parity_qualification.py",
        "test/registered/unit/spec/test_dspark_live_gpu_qualification.py",
        "test/registered/unit/spec/test_dspark_tp2_live_gpu_qualification.py",
        "test/registered/unit/spec/test_dspark_dp2_live_gpu_qualification.py",
        "test/registered/unit/spec/test_eagle3_live_gpu_qualification.py",
        "test/registered/unit/spec/test_native_hot_path_live_gpu_qualification.py",
        "test/registered/unit/spec/test_nextn_tp1_live_gpu_qualification.py",
        "test/registered/unit/spec/test_nextn_tp2_live_gpu_qualification.py",
        "test/registered/unit/spec/test_session_reset_gpu_qualification.py",
        "test/registered/unit/spec/test_tp1_dp2_live_gpu_qualification.py",
        "test/registered/unit/spec/test_tp2_dp1_live_gpu_qualification.py",
    )
    completed = subprocess.run(
        [
            os.fspath(Path(os.sys.executable)),
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "--collect-only",
            "-q",
            *files,
        ],
        cwd=checkout,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    node_ids = tuple(
        line.strip() for line in completed.stdout.splitlines() if "::test_" in line
    )
    if len(node_ids) != 96 or len(set(node_ids)) != 96:
        raise SystemExit("GPU qualification collection is not exactly 96 nodes")
    counts: dict[str, int] = {}
    for node_id in node_ids:
        filename = Path(node_id.split("::", maxsplit=1)[0]).name
        counts[filename] = counts.get(filename, 0) + 1
    if set(counts) != {Path(value).name for value in files} or any(
        count != 8 for count in counts.values()
    ):
        raise SystemExit("GPU qualification files are not exactly eight nodes each")
    if " skipped" in completed.stdout or " deselected" in completed.stdout:
        raise SystemExit("GPU qualification collection skipped or deselected coverage")


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
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    sidecar = (patch_root / "manifest.json.sha256").read_text(encoding="utf-8")
    if sidecar != f"{manifest_sha256}\n":
        raise SystemExit("patch manifest sidecar does not match canonical JSON")
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
        _verify_source_owned_compile_contract(checkout)
        _verify_source_owned_failure_actuator_contract(checkout)
        _verify_native_release_readiness_contract(checkout)
        if not args.compile_only:
            _verify_gpu_qualification_collection(checkout, environment=env)
        subprocess.run(
            [
                os.fspath(Path(os.sys.executable)),
                "-m",
                "pytest",
                "-q",
                "test/registered/unit/managers/test_scheduler_request_scoped_admission.py",
                "test/registered/unit/spec/test_session_reset_evidence.py",
                "test/registered/unit/spec/test_source_owned_http_accounting.py",
                "test/registered/unit/spec/test_terminal_speculative_evidence.py",
                "test/registered/unit/spec/test_compile_cache_evidence.py",
                "test/registered/unit/spec/test_failure_actuator_runtime.py",
                "test/registered/unit/spec/test_formal_gang_serving.py",
                "test/registered/unit/spec/test_dspark_online_adaptation_contract.py",
                "test/registered/unit/spec/test_eagle3_online_adaptation_contract.py",
                "test/registered/unit/spec/test_native_runtime_release.py",
                "test/registered/unit/spec/test_online_adaptation_protocol.py",
                "test/registered/unit/spec/test_terminal_target_metrics.py",
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
        request_reset_test = (
            checkout
            / "test/registered/unit/managers/test_scheduler_request_scoped_admission.py"
        )
        if request_reset_test.exists():
            raise SystemExit("reverse removal retained the patch-0008 contract")
        if (
            checkout / "test/registered/unit/spec/test_terminal_target_metrics.py"
        ).exists():
            raise SystemExit("reverse removal retained the patch-0006 focused test")
        distributed_contract = (
            checkout / "python/sglang/srt/speculative/online_adaptation_config.py"
        )
        if distributed_contract.exists() and "DISTRIBUTED_RUNTIME_CONTRACTS" in (
            distributed_contract.read_text(encoding="utf-8")
        ):
            raise SystemExit("reverse removal retained the patch-0007 contract")
        _run("git", "diff", "--exit-code", upstream, cwd=checkout)

    if _run("git", "rev-parse", "HEAD", cwd=source) != upstream:
        raise SystemExit("verification changed the upstream checkout")
    if _run("git", "status", "--porcelain=v1", cwd=source):
        raise SystemExit("verification dirtied the upstream checkout")
    print("SGLang patchset verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
