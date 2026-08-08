from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/run_priority_l0_stride_screen_queue.sh"
PROCESS_CONTROL = ROOT / "scripts/experiments/priority_queue_process.sh"
CONFIRMATION_QUEUE = (
    ROOT / "scripts/experiments/run_priority_stride_confirmation_queue.sh"
)
CONTROLLER_QUEUE = (
    ROOT / "scripts/experiments/run_priority_matched_controller_queue.sh"
)


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(
    path: Path, *, status: str, scope: str, evidence: list[Path]
) -> Path:
    payload = {
        "schema_version": 1,
        "status": status,
        "scope": scope,
        "evidence": [
            {"path": str(item.resolve()), "sha256": _sha(item)}
            for item in evidence
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    Path(str(path) + ".sha256").write_text(_sha(path) + "\n", encoding="utf-8")
    return path


def _receipt_validator_source() -> str:
    source = _source()
    function_start = source.index("receipt_valid() {")
    marker = "<<'PY'\n"
    python_start = source.index(marker, function_start) + len(marker)
    python_end = source.index("\nPY\n}", python_start)
    return source[python_start:python_end]


def _validate_receipt(path: Path, *, status: str, scope: str):
    return subprocess.run(
        [sys.executable, "-", str(path), status, scope],
        input=_receipt_validator_source(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_stride_screen_queue_shell_is_syntax_valid():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_stride_screen_queue_binds_inputs_and_source_overrides():
    source = _source()
    assert "stride-screen.lock.json" in source
    assert "stride-screen.model-roots.json" in source
    assert "p5_priority_dflash_stride_screen_v1.json" in source
    for override in (
        "LIGHTCONE_WORKSPACE",
        "LIGHTCONE_RUNTIME_ROOT",
        "LIGHTCONE_CLI",
        "LIGHTCONE_PYTHON",
        "LIGHTCONE_MANIFEST",
        "LIGHTCONE_STRIDE_SELECTOR",
        "LIGHTCONE_LOCKFILE",
        "LIGHTCONE_MODEL_ROOTS",
        "LIGHTCONE_ARTIFACT_ROOT",
        "LIGHTCONE_ANALYSIS_ROOT",
        "LIGHTCONE_CUDA_TOOLKIT_ROOT",
        "LIGHTCONE_QUEUE_LOCK_PATH",
    ):
        assert override in source


def test_stride_screen_queue_has_locked_order_and_no_sync_profiling():
    source = _source()
    ordered = (
        '"$LC" prepare-datasets',
        "headline run-manifest",
        '"$LC" validate-artifacts',
        '--baseline static',
        '--baseline tts',
        '"$PY" "$SELECTOR"',
    )
    positions = [source.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert "--datasets livecodebench" in source
    assert '--limit "$PROMPT_LIMIT"' in source
    assert 'manifest.get("engine_params", {}).get("prompt_limit")' in source
    assert "--methods static tts naive_async" in source
    assert "--weight-update-mode lora" in source
    assert "env -u CUBLAS_WORKSPACE_CONFIG -u CUDA_LAUNCH_BLOCKING" in source
    assert "-u PYTORCH_ALLOC_CONF" in source
    assert 'PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_LOCKED"' in source
    # FlashInfer may JIT through the Python environment's bundled ninja.
    # Invoking the CLI by absolute path does not activate that environment,
    # so the headline subprocess must carry its bin directory explicitly.
    assert 'PY_BIN_DIR=$(dirname -- "$PY")' in source
    assert 'PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH"' in source
    assert 'get("pytorch_cuda_alloc_conf")' in source
    assert "CUDA_LAUNCH_BLOCKING=1" not in source
    assert "cuda synchronize" not in source.lower()


def test_stride_screen_queue_separates_execution_and_candidate_terminals():
    source = _source()
    assert "EXECUTION_COMPLETE.json" in source
    assert "CANDIDATE_SCREEN_SELECTED.json" in source
    assert "CANDIDATE_SCREEN_BLOCKED.json" in source
    assert "candidate_screen_only_no_claim" in source
    assert "candidate_screen_selected" in source
    assert "candidate_screen_blocked" in source
    assert "candidate_screen_selected candidate_screen_only_no_claim" in source
    assert "candidate_screen_blocked candidate_screen_only_no_claim" in source
    assert 'for row in payload["evidence"]' in source
    assert "load_receipt(evidence)" in source
    assert "conflicting candidate-screen terminal receipts" in source
    assert "objective_pass" not in source
    assert "algorithmic_pass" not in source
    assert "engineering_pass" not in source


def test_stride_screen_queue_holds_one_root_lock_for_the_whole_process():
    source = _source()
    assert 'QUEUE_LOCK=${LIGHTCONE_QUEUE_LOCK_PATH:-$SCREEN_ROOT/' in source
    assert 'exec 9>>"$QUEUE_LOCK"' in source
    assert "flock -n 9" in source
    assert source.index("flock -n 9") < source.index("trap on_exit EXIT")


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


@pytest.mark.parametrize(
    ("stop_signal", "expected_code", "expected_name"),
    ((signal.SIGINT, 130, "INT"), (signal.SIGTERM, 143, "TERM")),
)
def test_managed_process_group_interrupt_is_scoped_and_preserves_partial(
    tmp_path: Path, stop_signal: int, expected_code: int, expected_name: str
):
    subprocess.run(["bash", "-n", str(PROCESS_CONTROL)], check=True)
    for queue in (SCRIPT, CONFIRMATION_QUEUE, CONTROLLER_QUEUE):
        source = queue.read_text(encoding="utf-8")
        assert "queue_process_control_init" in source
        assert "queue_run_managed" in source
        assert "signal=${QUEUE_STOP_SIGNAL:-none}" in source

    pid_record = tmp_path / "managed-pids.json"
    partial = tmp_path / "partial-evidence.jsonl"
    receipt = tmp_path / "failed-resumable.txt"
    worker = tmp_path / "worker.py"
    worker.write_text(
        """import json
import os
import subprocess
import sys
import time
from pathlib import Path

pid_record, partial = map(Path, sys.argv[1:])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pid_record.write_text(
    json.dumps({"leader": os.getpid(), "descendant": child.pid}) + "\\n",
    encoding="utf-8",
)
partial.write_text('{"status":"partial"}\\n', encoding="utf-8")
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    driver = tmp_path / "driver.sh"
    driver.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
source {PROCESS_CONTROL!s}
receipt=$1
on_exit() {{
  rc=$?
  printf 'failed_resumable exit_code=%s signal=%s\\n' \
    "$rc" "${{QUEUE_STOP_SIGNAL:-none}}" >"$receipt"
}}
trap on_exit EXIT
queue_process_control_init {sys.executable!s}
queue_run_managed {sys.executable!s} "$2" "$3" "$4"
""",
        encoding="utf-8",
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    queue = subprocess.Popen(
        [
            "bash",
            str(driver),
            str(receipt),
            str(worker),
            str(pid_record),
            str(partial),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not pid_record.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_record.is_file(), "managed worker did not start"
        pids = json.loads(pid_record.read_text(encoding="utf-8"))
        queue.send_signal(stop_signal)  # Only the queue receives the signal.
        stdout, stderr = queue.communicate(timeout=10)
        assert queue.returncode == expected_code, (stdout, stderr)
        assert receipt.read_text(encoding="utf-8").strip() == (
            f"failed_resumable exit_code={expected_code} signal={expected_name}"
        )
        assert partial.read_text(encoding="utf-8") == '{"status":"partial"}\n'

        deadline = time.monotonic() + 5
        while any(_process_is_running(int(pid)) for pid in pids.values()):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        assert not _process_is_running(int(pids["leader"]))
        assert not _process_is_running(int(pids["descendant"]))
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_terminal_receipt_recursively_binds_leaf_evidence(tmp_path):
    leaf = tmp_path / "leaf.txt"
    leaf.write_text("original\n", encoding="utf-8")
    nested = _write_receipt(
        tmp_path / "execution.json",
        status="execution_complete",
        scope="candidate_stride_screen_no_claim",
        evidence=[leaf],
    )
    terminal = _write_receipt(
        tmp_path / "terminal.json",
        status="candidate_screen_selected",
        scope="candidate_screen_only_no_claim",
        evidence=[nested, Path(str(nested) + ".sha256")],
    )

    valid = _validate_receipt(
        terminal,
        status="candidate_screen_selected",
        scope="candidate_screen_only_no_claim",
    )
    assert valid.returncode == 0, valid.stderr
    wrong_scope = _validate_receipt(
        terminal,
        status="candidate_screen_selected",
        scope="wrong_scope",
    )
    assert wrong_scope.returncode != 0

    leaf.write_text("mutated\n", encoding="utf-8")
    stale = _validate_receipt(
        terminal,
        status="candidate_screen_selected",
        scope="candidate_screen_only_no_claim",
    )
    assert stale.returncode != 0
