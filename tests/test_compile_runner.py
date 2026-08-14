from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.serving import PinnedBenchServingTransport
from lightcone_spec.runtime.compile_cache import (
    COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
    COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256,
    COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    SGLANG_FIRST_PARTY_COMPILE_BUILDER,
    CompileCacheKey,
    CompileCacheLaunchPlan,
    CompileOnlyAssignmentContract,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
    _content_sha256,
)
from lightcone_spec.runtime.compile_runner import (
    COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
    RELEASE_COMPILE_GPU_SOURCE_REGISTRY_EMPTY,
    RELEASE_COMPILE_RUNNER_UNAVAILABLE,
    RELEASE_COMPILE_SUBPROCESSES,
    RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S,
    CompileAssignmentPlan,
    CompilePrewarmObservation,
    CompileResultPointer,
    CompileRunnerBlocked,
    CompileShutdownObservation,
    CompileSubprocessLifecycleReceipt,
    CompileWorkerSourceDescriptor,
    ReleaseCompileSubprocess,
    execute_compile_assignment_for_cpu_test,
    execute_compile_assignment_subprocess_for_cpu_test,
    execute_release_compile_assignment_plan,
    require_release_compile_assignment_plan,
    write_compile_prewarm_manifest,
)
from lightcone_spec.sglang_bridge.compile_worker import (
    GPU_COMPILE_REASON,
    NATIVE_COMPILE_BEGIN_PATH,
    NATIVE_COMPILE_FINALIZE_PATH,
    SOURCE_OWNED_COMPILE_HOOK,
    SOURCE_OWNED_COMPILE_PROTOCOL_SHA256,
    PinnedCompileLifecycleWorker,
)
from lightcone_spec.sglang_bridge.launch import main as launch_main


def _sha(label: str) -> str:
    return hashlib.sha256(f"compile-runner:{label}".encode()).hexdigest()


def _key(*, target_revision: str = "a" * 40) -> CompileCacheKey:
    return CompileCacheKey(
        patched_sglang_tree=PINNED_SGLANG_TREE,
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        patch_sha256=PINNED_SGLANG_PATCH_SHA256,
        source_sha256=PINNED_SGLANG_COMPILE_SOURCE_SHA256,
        python_version="3.12.11",
        torch_version="2.11.0+cu130",
        triton_version="3.6.0",
        cuda_version="13.0",
        driver_version="580.65.06",
        sm_architecture="sm_120",
        gpu_model="RTX PRO 6000 Blackwell Server Edition",
        dtype="bfloat16",
        target_revision=target_revision,
        drafter_revision="b" * 40,
        tensor_parallel_size=2,
        context_limit=4096,
        max_running_requests=2,
        graph_buckets=(1, 2),
        allocator="cuda_malloc_async",
        build_flags=("CUDA_ARCH=120",),
    )


def _inputs(tmp_path: Path) -> tuple[CompileAssignmentPlan, Path]:
    cache_root = (tmp_path / "cache").resolve()
    cache_plan = CompileCacheLaunchPlan(
        schema_version=1,
        kind="compile_cache_launch_plan",
        key=_key(),
        cache_root=str(cache_root),
        cache_mode="build",
        builder_id=SGLANG_FIRST_PARTY_COMPILE_BUILDER,
        base_receipt_path=None,
        base_receipt_sha256=None,
    )
    manifest = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=_sha("model-lock"),
        sampling_profile_sha256=_sha("sampling"),
        payloads=(
            CompileOnlyPrewarmPayload("bucket-1", 1, (1, 2), 1, 11),
            CompileOnlyPrewarmPayload("bucket-2", 2, (3, 4), 1, 22),
        ),
    )
    result_path = (tmp_path / "compile-result.json").resolve()
    assignment = CompileOnlyAssignmentContract(
        schema_version=1,
        kind="compile_only_assignment_contract",
        assignment_protocol_sha256=COMPILE_ONLY_ASSIGNMENT_PROTOCOL_SHA256,
        cell_id=_sha("cell"),
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        physical_assignment_sha256=_sha("physical"),
        experiment_budget_sha256=_sha("budget"),
        budget_materialization_authority_sha256=_sha("budget-authority"),
        inventory_sha256=_sha("inventory"),
        inventory_source_receipt_sha256=_sha("inventory-source"),
        gpu_uuids=("GPU-0", "GPU-1"),
        host_id="host-0",
        fixed_instance_gpu_count=8,
        compile_cache_plan=cache_plan,
        prewarm_manifest=manifest,
        graceful_shutdown_protocol_sha256=(
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ),
        result_pointer_protocol_sha256=COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
        result_pointer_path=str(result_path),
    )
    assignment_path = assignment.write((tmp_path / "assignment.json").resolve())
    cache_plan_path = cache_plan.write((tmp_path / "cache-plan.json").resolve())
    manifest_path = write_compile_prewarm_manifest(
        manifest,
        (tmp_path / "prewarm.json").resolve(),
    )
    return (
        CompileAssignmentPlan.issue(
            assignment_manifest_path=assignment_path,
            compile_cache_plan_path=cache_plan_path,
            prewarm_manifest_path=manifest_path,
            result_pointer_path=result_path,
            attempt_id="attempt-001",
        ),
        result_path,
    )


class _FakeDriver:
    process_id = 4242

    def __init__(self, *, active_requests: int = 0, omit_second: bool = False) -> None:
        self.active_requests = active_requests
        self.omit_second = omit_second
        self.started = False

    def start(self, environment: object) -> None:
        assert environment
        self.started = True

    def prewarm(self, payload: CompileOnlyPrewarmPayload) -> CompilePrewarmObservation:
        request_id = payload.request_id
        if self.omit_second and payload.graph_bucket == 2:
            request_id = "wrong-request"
        return CompilePrewarmObservation(
            request_id=request_id,
            graph_bucket=payload.graph_bucket,
            completed=True,
            provider_receipt_sha256=_sha(f"provider:{payload.request_id}"),
        )

    def graceful_shutdown(self) -> CompileShutdownObservation:
        return CompileShutdownObservation(
            process_id=self.process_id,
            shutdown_requested_ns=100,
            process_exited_ns=200,
            exit_code=0,
            active_requests=self.active_requests,
            queued_requests=0,
            provider_ack_sha256=_sha("shutdown-ack"),
        )


def _materialize(overlay: Path) -> None:
    target = overlay / "triton" / "kernel.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"compiled-kernel")


_CHILD = r"""
import hashlib
import json
import os
import sys

protocol = os.environ["TEST_COMPILE_PROTOCOL"]
plan = os.environ["TEST_COMPILE_PLAN"]
bad_request = os.environ.get("TEST_COMPILE_BAD_REQUEST")
stubborn_pidfile = os.environ.get("TEST_COMPILE_STUBBORN_PIDFILE")
assert "TEST_COMPILE_SECRET_SENTINEL" not in os.environ


def send(kind, **values):
    row = {
        "kind": kind,
        "protocol_sha256": protocol,
        "assignment_plan_sha256": plan,
        **values,
    }
    sys.stdout.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def receive():
    row = json.loads(sys.stdin.readline())
    assert row["protocol_sha256"] == protocol
    assert row["assignment_plan_sha256"] == plan
    return row


send("compile_subprocess_ready", process_id=os.getpid())
start = receive()
assert start["kind"] == "compile_subprocess_start"
for path in start["cache_environment"].values():
    assert os.path.isabs(path)
    assert os.path.isdir(path)
send("compile_subprocess_started", process_id=os.getpid())
while True:
    row = receive()
    if row["kind"] == "compile_subprocess_shutdown":
        if stubborn_pidfile:
            import signal
            import subprocess
            import time

            code = (
                "import os,signal,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                f"open({stubborn_pidfile!r},'w').write(str(os.getpid()));"
                "time.sleep(300)"
            )
            subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            for _ in range(100):
                if os.path.exists(stubborn_pidfile):
                    break
                time.sleep(0.01)
        send(
            "compile_subprocess_drained",
            active_requests=0,
            queued_requests=0,
            provider_ack_sha256=hashlib.sha256(b"drained").hexdigest(),
        )
        break
    assert row["kind"] == "compile_subprocess_prewarm"
    cache = start["cache_environment"]["TRITON_CACHE_DIR"]
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, f"bucket-{row['graph_bucket']}.bin"), "wb") as handle:
        handle.write(str(row["graph_bucket"]).encode())
    request_id = row["request_id"]
    if bad_request and row["graph_bucket"] == 2:
        request_id = "wrong-request"
    send(
        "compile_subprocess_prewarm_complete",
        request_id=request_id,
        graph_bucket=row["graph_bucket"],
        completed=True,
        provider_receipt_sha256=hashlib.sha256(
            f"prewarm:{row['request_id']}".encode()
        ).hexdigest(),
    )
"""


def _subprocess_argv(
    plan: CompileAssignmentPlan,
    *,
    bad_request: bool = False,
    stubborn_pidfile: Path | None = None,
) -> tuple[str, ...]:
    executable = str(Path(os.sys.executable).resolve())
    environment_prefix = (
        "import os;"
        f"os.environ['TEST_COMPILE_PROTOCOL']={COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256!r};"
        f"os.environ['TEST_COMPILE_PLAN']={plan.sha256!r};"
    )
    if bad_request:
        environment_prefix += "os.environ['TEST_COMPILE_BAD_REQUEST']='1';"
    if stubborn_pidfile is not None:
        environment_prefix += (
            f"os.environ['TEST_COMPILE_STUBBORN_PIDFILE']={str(stubborn_pidfile)!r};"
        )
    return (executable, "-c", environment_prefix + _CHILD)


def _rewrite_canonical_json(path: Path, value: object) -> bytes:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    body = canonical + b"\n"
    path.write_bytes(body)
    Path(f"{path}.sha256").write_text(
        f"{hashlib.sha256(canonical).hexdigest()}\n",
        encoding="ascii",
    )
    return body


def test_cpu_fake_lifecycle_requires_exact_prewarm_shutdown_and_terminal_pointer(
    tmp_path: Path,
) -> None:
    plan, result_path = _inputs(tmp_path)
    plan_path = plan.write((tmp_path / "compile-assignment-plan.json").resolve())
    assert CompileAssignmentPlan.load(plan_path) == plan
    pointer = execute_compile_assignment_for_cpu_test(
        plan,
        _FakeDriver(),
        materialize_cache_files=_materialize,
    )

    assert result_path.is_file()
    assert pointer.formal_execution_authorized is False
    assert Path(f"{result_path}.sha256").read_text().strip() == pointer.sha256
    assert pointer.assignment_plan_sha256 == plan.sha256
    assert set(pointer.bindings()) == {
        "assignment_manifest",
        "compile_cache_plan",
        "prewarm_manifest",
        "attempt_receipt",
        "graceful_shutdown_receipt",
        "final_cache_receipt",
        "immutable_cache_object_manifest",
    }
    pointer.reopen()
    assert CompileResultPointer.load(result_path) == pointer


def test_real_subprocess_lifecycle_publishes_raw_receipt_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_COMPILE_SECRET_SENTINEL", "must-not-reach-child")
    plan, result_path = _inputs(tmp_path)
    plan_path = plan.write((tmp_path / "compile-assignment-plan.json").resolve())

    pointer = execute_compile_assignment_subprocess_for_cpu_test(
        plan_path,
        _subprocess_argv(plan),
    )

    assert pointer.schema_version == 2
    assert pointer.formal_execution_authorized is False
    assert pointer.assignment_plan_source is not None
    assert pointer.subprocess_lifecycle_receipt is not None
    receipt = CompileSubprocessLifecycleReceipt.load(
        pointer.subprocess_lifecycle_receipt.absolute_path
    )
    assert receipt.assignment_plan_sha256 == plan.sha256
    assert receipt.formal_execution_authorized is False
    rows = [json.loads(event.canonical_json) for event in receipt.events]
    assert [row["kind"] for row in rows] == [
        "compile_subprocess_ready",
        "compile_subprocess_start",
        "compile_subprocess_started",
        "compile_subprocess_prewarm",
        "compile_subprocess_prewarm_complete",
        "compile_subprocess_prewarm",
        "compile_subprocess_prewarm_complete",
        "compile_subprocess_shutdown",
        "compile_subprocess_drained",
    ]
    assert CompileResultPointer.load(result_path) == pointer

    # A valid terminal pointer is the sole resume authority.  No second cache
    # attempt or child transcript is created on exact replay.
    attempts_before = tuple(sorted((tmp_path / "cache" / "attempts").glob("*.json")))
    assert (
        execute_compile_assignment_subprocess_for_cpu_test(
            plan_path,
            _subprocess_argv(plan),
        )
        == pointer
    )
    assert (
        tuple(sorted((tmp_path / "cache" / "attempts").glob("*.json")))
        == attempts_before
    )


def test_real_subprocess_protocol_failure_retains_attempt_without_result_pointer(
    tmp_path: Path,
) -> None:
    plan, result_path = _inputs(tmp_path)
    plan_path = plan.write((tmp_path / "compile-assignment-plan.json").resolve())

    with pytest.raises(ValueError, match="differs|cover"):
        execute_compile_assignment_subprocess_for_cpu_test(
            plan_path,
            _subprocess_argv(plan, bad_request=True),
        )

    assert not result_path.exists()
    assert not Path(f"{result_path}.sha256").exists()
    attempts = tuple((tmp_path / "cache" / "attempts").glob("*.json"))
    assert attempts
    assert any(json.loads(path.read_text())["state"] == "failed" for path in attempts)


def test_real_subprocess_failure_kills_stubborn_process_group(
    tmp_path: Path,
) -> None:
    plan, result_path = _inputs(tmp_path)
    plan_path = plan.write((tmp_path / "compile-assignment-plan.json").resolve())
    pidfile = (tmp_path / "stubborn-grandchild.pid").resolve()

    with pytest.raises(ValueError, match="left a live child process group"):
        execute_compile_assignment_subprocess_for_cpu_test(
            plan_path,
            _subprocess_argv(plan, stubborn_pidfile=pidfile),
            timeout_seconds=3.0,
        )

    grandchild_pid = int(pidfile.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)
    assert not result_path.exists()
    assert not Path(f"{result_path}.sha256").exists()


def test_coordinated_rehash_cannot_promote_diagnostic_receipt_or_pointer(
    tmp_path: Path,
) -> None:
    plan, result_path = _inputs(tmp_path)
    plan_path = plan.write((tmp_path / "compile-assignment-plan.json").resolve())
    pointer = execute_compile_assignment_subprocess_for_cpu_test(
        plan_path,
        _subprocess_argv(plan),
    )
    assert pointer.subprocess_lifecycle_receipt is not None
    receipt_path = Path(pointer.subprocess_lifecycle_receipt.absolute_path)

    receipt_row = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_row["formal_execution_authorized"] = True
    receipt_row["source_authority_sha256"] = "f" * 64
    receipt_body = _rewrite_canonical_json(receipt_path, receipt_row)
    with pytest.raises(CompileRunnerBlocked) as receipt_blocked:
        CompileSubprocessLifecycleReceipt.load(receipt_path)
    assert receipt_blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE

    pointer_row = json.loads(result_path.read_text(encoding="utf-8"))
    pointer_row["formal_execution_authorized"] = True
    pointer_row["subprocess_lifecycle_receipt"]["raw_sha256"] = hashlib.sha256(
        receipt_body
    ).hexdigest()
    pointer_row["subprocess_lifecycle_receipt"]["size"] = len(receipt_body)
    _rewrite_canonical_json(result_path, pointer_row)
    forged_pointer = CompileResultPointer.from_dict(pointer_row)
    with pytest.raises(CompileRunnerBlocked) as reopen_blocked:
        forged_pointer.reopen()
    assert reopen_blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE
    with pytest.raises(CompileRunnerBlocked) as pointer_blocked:
        CompileResultPointer.load(result_path)
    assert pointer_blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE


def test_formal_subprocess_gate_blocks_before_path_or_process_access(
    tmp_path: Path,
) -> None:
    assert RELEASE_COMPILE_SUBPROCESSES == ()
    assert RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S == ()
    missing = (tmp_path / "missing-plan.json").resolve()
    with pytest.raises(CompileRunnerBlocked) as blocked:
        execute_release_compile_assignment_plan(missing)
    assert blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE
    assert not missing.exists()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "driver, message",
    [
        (_FakeDriver(omit_second=True), "exactly cover"),
        (_FakeDriver(active_requests=1), "not drained"),
    ],
)
def test_incomplete_prewarm_or_shutdown_never_publishes_result(
    tmp_path: Path,
    driver: _FakeDriver,
    message: str,
) -> None:
    plan, result_path = _inputs(tmp_path)
    with pytest.raises(ValueError, match=message):
        execute_compile_assignment_for_cpu_test(
            plan,
            driver,
            materialize_cache_files=_materialize,
        )
    assert not result_path.exists()


def test_assignment_plan_rejects_caller_key_and_revalidation_tamper(
    tmp_path: Path,
) -> None:
    plan, _ = _inputs(tmp_path)
    foreign = replace(
        CompileCacheLaunchPlan.load(plan.compile_cache_plan_path),
        key=_key(target_revision="c" * 40),
    )
    foreign_path = foreign.write((tmp_path / "foreign-plan.json").resolve())
    with pytest.raises(ValueError, match="differs from assignment"):
        CompileAssignmentPlan.issue(
            assignment_manifest_path=plan.assignment_manifest_path,
            compile_cache_plan_path=foreign_path,
            prewarm_manifest_path=plan.prewarm_manifest_path,
            result_pointer_path=plan.result_pointer_path,
            attempt_id="attempt-foreign",
        )

    Path(plan.prewarm_manifest_path).write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="sidecar differs|fields differ"):
        plan.revalidate()


def test_resume_reopens_every_pointer_binding_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    plan, result_path = _inputs(tmp_path)
    pointer = execute_compile_assignment_for_cpu_test(
        plan,
        _FakeDriver(),
        materialize_cache_files=_materialize,
    )
    victim = Path(pointer.graceful_shutdown_receipt.absolute_path)
    victim.write_bytes(victim.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after terminal publication"):
        CompileResultPointer.load(result_path)


def test_release_entrypoint_is_named_block_before_any_path_read(tmp_path: Path) -> None:
    assert RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S == ()
    missing = CompileAssignmentPlan(
        schema_version=1,
        kind="first_party_compile_assignment_plan",
        protocol_sha256=(
            "6a2b33824bb1b45b1cf207220fd7762f254182331520467b38718a81b7b90487"
        ),
        assignment_manifest_path=str((tmp_path / "missing-assignment").resolve()),
        assignment_sha256="0" * 64,
        compile_cache_plan_path=str((tmp_path / "missing-plan").resolve()),
        compile_cache_plan_sha256="1" * 64,
        prewarm_manifest_path=str((tmp_path / "missing-prewarm").resolve()),
        prewarm_manifest_sha256="2" * 64,
        compile_key_sha256="3" * 64,
        model_lock_sha256="4" * 64,
        target_revision="target",
        drafter_revision=None,
        physical_assignment_sha256="5" * 64,
        experiment_budget_sha256="6" * 64,
        inventory_sha256="7" * 64,
        gpu_uuids=("GPU-0",),
        host_id="host-0",
        tensor_parallel_size=1,
        context_limit=1,
        max_running_requests=1,
        graph_buckets=(1,),
        graceful_shutdown_protocol_sha256=(
            COMPILE_ONLY_GRACEFUL_SHUTDOWN_PROTOCOL_SHA256
        ),
        result_pointer_protocol_sha256=COMPILE_ONLY_RESULT_POINTER_PROTOCOL_SHA256,
        attempt_id="blocked",
        result_pointer_path=str((tmp_path / "must-not-exist").resolve()),
    )
    # Use the module constant so this test cannot silently bless a stale digest.
    from lightcone_spec.runtime.compile_runner import (
        COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256,
    )

    missing = replace(
        missing,
        protocol_sha256=COMPILE_ASSIGNMENT_PLAN_PROTOCOL_SHA256,
    )
    with pytest.raises(CompileRunnerBlocked) as blocked:
        require_release_compile_assignment_plan(missing)
    assert blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE
    assert not Path(missing.result_pointer_path).exists()


def test_launcher_accepts_exact_compile_flags_but_blocks_before_path_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.launch.verify_patched_checkout",
        lambda _path: pytest.fail("compile BLOCK must precede checkout verification"),
    )
    result_path = (tmp_path / "must-not-exist.json").resolve()
    with pytest.raises(CompileRunnerBlocked) as blocked:
        launch_main(
            [
                "--checkout",
                str((tmp_path / "missing-checkout").resolve()),
                "--compile-cache-plan",
                str((tmp_path / "missing-cache-plan").resolve()),
                "--compile-cache-plan-sha256",
                "0" * 64,
                "--compile-cache-key-sha256",
                "0" * 64,
                "--run-config",
                str((tmp_path / "missing-run-config").resolve()),
                "--run-config-sha256",
                "0" * 64,
                "--compile-only-assignment",
                str((tmp_path / "missing-assignment").resolve()),
                "--compile-only-manifest",
                str((tmp_path / "missing-manifest").resolve()),
                "--compile-result-pointer",
                str(result_path),
                "--",
            ]
        )
    assert blocked.value.reason_code == RELEASE_COMPILE_RUNNER_UNAVAILABLE
    assert not result_path.exists()


class _NativeCompileTransport(PinnedBenchServingTransport):
    def __init__(self) -> None:
        self.begin: dict[str, object] | None = None
        self.submitted: list[CompileOnlyPrewarmPayload] = []

    async def post_json(self, path: str, body: dict[str, object], /) -> object:
        if path == NATIVE_COMPILE_BEGIN_PATH:
            assert self.begin is None
            identity = {key: value for key, value in body.items() if key != "prewarm"}
            receipt = {
                "schema_version": 1,
                "hook": SOURCE_OWNED_COMPILE_HOOK,
                "protocol_sha256": SOURCE_OWNED_COMPILE_PROTOCOL_SHA256,
                "release_status": "CPU_CONTRACT_ONLY",
                "gpu_compile_semantics": "PENDING",
                "gpu_compile_reason": GPU_COMPILE_REASON,
                "patched_sglang_tree": PINNED_SGLANG_TREE,
                "process_id": 4242,
                "process_started_ns": 9,
                **identity,
                "prewarm_sha256": _content_sha256(body["prewarm"]),
                "ordered_prewarm": body["prewarm"],
                "begin_state": {"active_requests": 0, "queued_requests": 0},
            }
            receipt["begin_sha256"] = _content_sha256(receipt)
            self.begin = receipt
            return receipt
        if path != NATIVE_COMPILE_FINALIZE_PATH or self.begin is None:
            raise AssertionError(path)
        assert body == {"begin_sha256": self.begin["begin_sha256"]}
        terminals = []
        for index, payload in enumerate(self.submitted):
            row = {
                "sequence": index,
                "graph_bucket": payload.graph_bucket,
                "request_id": payload.request_id,
                "input_token_count": len(payload.input_token_ids),
                "input_token_ids_sha256": _content_sha256(
                    list(payload.input_token_ids)
                ),
                "requested_output_tokens": payload.requested_output_tokens,
                "sampling_seed": payload.sampling_seed,
                "output_token_count": 1,
                "output_token_ids_sha256": _content_sha256([100 + index]),
                "terminal_status": "completed",
                "terminal_reason": "FINISH_LENGTH",
            }
            row["terminal_sha256"] = _content_sha256(row)
            terminals.append(row)
        final = {
            key: value
            for key, value in self.begin.items()
            if key not in {"ordered_prewarm", "begin_sha256"}
        }
        final.update(
            {
                "begin_sha256": self.begin["begin_sha256"],
                "begin_state": self.begin["begin_state"],
                "ordered_terminals": terminals,
                "final_state": {"active_requests": 0, "queued_requests": 0},
                "gpu_measurements": {
                    name: {"value": None, "reason": GPU_COMPILE_REASON}
                    for name in (
                        "cache_hits",
                        "cache_misses",
                        "jit_time_ns",
                        "graph_capture_count",
                        "graph_replay_count",
                        "cache_write_count",
                    )
                },
                "completion_marker": "COMPILE_CPU_CONTRACT_COMPLETE",
            }
        )
        final["compile_receipt_sha256"] = _content_sha256(final)
        return final


def test_pinned_compile_worker_consumes_only_native_source_receipts(
    tmp_path: Path,
) -> None:
    plan, _result_path = _inputs(tmp_path)
    transport = _NativeCompileTransport()
    worker = PinnedCompileLifecycleWorker(transport)

    async def submit(payload: CompileOnlyPrewarmPayload) -> object:
        transport.submitted.append(payload)
        return {"caller_summary": "ignored"}

    result = asyncio.run(worker.execute(plan, submit_prewarm=submit))
    assert result.release_status == "CPU_CONTRACT_ONLY"
    assert result.gpu_compile_semantics == "PENDING"
    assert result.formal_execution_authorized is False
    assert len(result.ordered_terminal_sha256s) == len(
        CompileOnlyPrewarmManifest.from_dict(
            json.loads(Path(plan.prewarm_manifest_path).read_text())
        ).payloads
    )
    with pytest.raises(RuntimeError, match="one-shot"):
        asyncio.run(worker.execute(plan, submit_prewarm=submit))


def test_monkeypatched_compile_allowlists_still_hit_gpu_source_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightcone_spec.runtime.compile_runner as runner

    plan, _result_path = _inputs(tmp_path)
    checkout = (tmp_path / "patched-sglang").resolve()
    checkout.mkdir()
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.checkout.verify_patched_checkout",
        lambda path: Path(path).resolve(),
    )
    descriptor = CompileWorkerSourceDescriptor.issue(patched_sglang_checkout=checkout)
    source = ReleaseCompileSubprocess(
        argv=(descriptor.interpreter_path, descriptor.helper_path),
        worker=descriptor,
        protocol_sha256=COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
        gpu_qualification_sha256=_sha("gpu-qualification"),
    )
    monkeypatch.setattr(runner, "RELEASE_COMPILE_SUBPROCESSES", (source,))
    monkeypatch.setattr(
        runner, "RELEASE_TRUSTED_COMPILE_ASSIGNMENT_PLAN_SHA256S", (plan.sha256,)
    )
    assert runner.RELEASE_GPU_VETTED_COMPILE_SOURCE_SHA256S == ()
    with pytest.raises(CompileRunnerBlocked) as blocked:
        require_release_compile_assignment_plan(plan)
    assert blocked.value.reason_code == RELEASE_COMPILE_GPU_SOURCE_REGISTRY_EMPTY
