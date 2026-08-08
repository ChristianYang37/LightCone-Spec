from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/run_priority_stride_confirmation_queue.sh"


def _write_executable(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _mock_queue(tmp_path: Path, *, comparison_pass: bool = True) -> dict[str, object]:
    fake_bin = tmp_path / "bin"
    log = tmp_path / "commands.jsonl"
    screen_root = tmp_path / "screen"
    confirmation_root = tmp_path / "confirmation"
    runtime_root = tmp_path / "runtime"
    workspace = tmp_path / "workspace"
    lockfile = tmp_path / "screen.lock.json"
    model_roots = tmp_path / "screen.model-roots.json"
    lock_body = {
        "schema_version": 1,
        "created_utc": "2026-08-08T00:00:00Z",
        "git_repos": [],
        "hf_snapshots": [],
        "datasets": [],
        "environment": {
            "docker_image_digest": None,
            "python_version": "3.12",
            "cuda_version": None,
            "driver_version": None,
            "torch_version": "test",
            "triton_version": None,
            "sglang_version": None,
            "compiler_versions": {},
        },
        "gpus": [],
    }
    lock_text = json.dumps(
        lock_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    lockfile.write_text(lock_text, encoding="utf-8")
    Path(str(lockfile) + ".sha256").write_text(
        hashlib.sha256(lock_text.encode("utf-8")).hexdigest() + "\n",
        encoding="utf-8",
    )
    model_roots.write_text("{}\n", encoding="utf-8")

    _write_executable(
        fake_bin / "flock",
        "#!/bin/sh\nexit \"${MOCK_FLOCK_RC:-0}\"\n",
    )
    screen = _write_executable(
        tmp_path / "mock_screen.py",
        f"""#!{sys.executable}
import json
import os
from pathlib import Path

log = Path(os.environ["MOCK_COMMAND_LOG"])
observed = {{
    "command": "screen",
    "artifact_root": os.environ.get("LIGHTCONE_ARTIFACT_ROOT"),
    "analysis_root": os.environ.get("LIGHTCONE_ANALYSIS_ROOT"),
    "queue_lock": os.environ.get("LIGHTCONE_QUEUE_LOCK_PATH"),
    "state": os.environ.get("LIGHTCONE_STATE_PATH"),
    "failed": os.environ.get("LIGHTCONE_FAILED_RECEIPT"),
}}
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(observed, sort_keys=True) + "\\n")
expected = {{
    "artifact_root": os.environ["MOCK_EXPECT_SCREEN_ARTIFACT_ROOT"],
    "analysis_root": os.environ["MOCK_EXPECT_SCREEN_ANALYSIS_ROOT"],
    "queue_lock": os.environ["MOCK_EXPECT_SCREEN_QUEUE_LOCK"],
    "state": os.environ["MOCK_EXPECT_SCREEN_STATE"],
    "failed": os.environ["MOCK_EXPECT_SCREEN_FAILED"],
}}
if any(observed[key] != value for key, value in expected.items()):
    raise SystemExit(91)
selected = Path(os.environ["LIGHTCONE_SELECTED_RECEIPT"])
blocked = Path(os.environ["LIGHTCONE_BLOCKED_RECEIPT"])
selected.parent.mkdir(parents=True, exist_ok=True)
status = os.environ.get("MOCK_SCREEN_STATUS", "selected")
if status == "selected":
    if not selected.exists():
        selected.write_text('{{"status":"selected"}}\\n', encoding="utf-8")
elif status == "blocked":
    if not blocked.exists():
        blocked.write_text('{{"status":"blocked"}}\\n', encoding="utf-8")
elif status == "conflict":
    selected.write_text('{{"status":"selected"}}\\n', encoding="utf-8")
    blocked.write_text('{{"status":"blocked"}}\\n', encoding="utf-8")
else:
    raise SystemExit(92)
""",
    )
    lightcone = _write_executable(
        tmp_path / "mock_lightcone.py",
        f"""#!{sys.executable}
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

command, args = sys.argv[1], sys.argv[2:]
def option(name):
    index = args.index(name)
    return args[index + 1]
with Path(os.environ["MOCK_COMMAND_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"command": "lc:" + command, "args": args}}, sort_keys=True) + "\\n")
fail = os.environ.get("MOCK_FAIL_COMMAND")
marker = Path(os.environ["MOCK_FAIL_ONCE_MARKER"])
if fail == command and not marker.exists():
    marker.write_text(command, encoding="utf-8")
    raise SystemExit(23)
artifact_root = Path(option("--artifact-root"))
if artifact_root.resolve() != Path(os.environ["MOCK_EXPECT_CONFIRM_ARTIFACT_ROOT"]).resolve():
    raise SystemExit(93)
artifact_root.mkdir(parents=True, exist_ok=True)
run_marker = artifact_root / "mock-run-complete.json"
if command == "run-manifest":
    if os.environ.get("MOCK_BLOCK_RUN_MANIFEST") == "1":
        descendant = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        Path(os.environ["MOCK_MANAGED_PIDS"]).write_text(
            json.dumps({{"leader": os.getpid(), "descendant": descendant.pid}}) + "\\n",
            encoding="utf-8",
        )
        Path(os.environ["MOCK_PARTIAL_EVIDENCE"]).write_text(
            '{{"status":"partial"}}\\n', encoding="utf-8"
        )
        while True:
            time.sleep(1)
    status = "resume-skip" if run_marker.exists() else "executed"
    run_marker.write_text('{{"status":"complete_valid"}}\\n', encoding="utf-8")
    with Path(os.environ["MOCK_COMMAND_LOG"]).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({{"command": "lc:run-manifest:" + status}}) + "\\n")
elif command == "validate-artifacts":
    if not run_marker.is_file():
        raise SystemExit(94)
    coverage = Path(option("--coverage-output"))
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text('{{"summary":{{"final_validation_ok":true}}}}\\n', encoding="utf-8")
    Path(str(coverage) + ".sha256").write_text(
        hashlib.sha256(coverage.read_bytes()).hexdigest() + "\\n", encoding="utf-8"
    )
elif command == "analyze":
    if not run_marker.is_file():
        raise SystemExit(95)
    output = Path(option("--output-dir"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "mock-analysis-ready.json").write_text('{{"ready":true}}\\n', encoding="utf-8")
else:
    raise SystemExit(96)
""",
    )
    builder = _write_executable(
        tmp_path / "mock_builder.py",
        """import hashlib
import json
import os
import sys
from pathlib import Path

command, args = sys.argv[1], sys.argv[2:]
def option(name):
    index = args.index(name)
    return args[index + 1]
def write_attested(path, payload):
    path = Path(path)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\\n"
    if path.exists():
        sidecar = Path(str(path) + ".sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != hashlib.sha256(path.read_bytes()).hexdigest():
            raise SystemExit(81)
        if path.read_text() != body:
            raise SystemExit(82)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    Path(str(path) + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\\n", encoding="utf-8"
    )
row = {"command": "builder:" + command, "args": args}
with Path(os.environ["MOCK_COMMAND_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\\n")
for name in ("--selected-receipt", "--lockfile", "--model-roots", "--artifact-root"):
    if name not in args:
        raise SystemExit(83)
if Path(option("--artifact-root")).resolve() != Path(os.environ["MOCK_EXPECT_CONFIRM_ARTIFACT_ROOT"]).resolve():
    raise SystemExit(84)
for name in ("--selected-receipt", "--lockfile", "--model-roots"):
    if not Path(option(name)).is_file():
        raise SystemExit(85)
if command == "build":
    write_attested(option("--foundation-manifest"), {
        "engine_params": {
            "prompt_limit": 48,
            "pytorch_cuda_alloc_conf": "backend:native,expandable_segments:True",
        }
    })
    write_attested(option("--receipt"), {"status": "ready_for_execution"})
elif command == "compare":
    required = (
        "--generation-receipt", "--foundation-manifest", "--analysis-root",
        "--coverage", "--receipt",
    )
    if any(name not in args for name in required):
        raise SystemExit(86)
    if not (Path(option("--analysis-root")) / "mock-analysis-ready.json").is_file():
        raise SystemExit(87)
    passed = os.environ.get("MOCK_COMPARISON_PASS", "1") == "1"
    write_attested(option("--receipt"), {
        "status": "TTS_0_40K_CONFIRMED" if passed else "BLOCKED",
        "formal_acceptance_foundation_pass": passed,
    })
else:
    raise SystemExit(88)
""",
    )

    screen_artifact = screen_root / "runs"
    screen_analysis = screen_root / "analysis"
    screen_lock = screen_root / ".priority-l0-stride-screen.lock"
    screen_state = screen_artifact / "priority-state.jsonl"
    screen_failed = screen_artifact / "PRIORITY_FAILED.json"
    confirmation_artifact = confirmation_root / "runs"
    confirmation_analysis = confirmation_root / "analysis"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "LIGHTCONE_WORKSPACE": str(workspace),
        "LIGHTCONE_RUNTIME_ROOT": str(runtime_root),
        "LIGHTCONE_PYTHON": sys.executable,
        "LIGHTCONE_CLI": str(lightcone),
        "LIGHTCONE_CONFIRMATION_BUILDER": str(builder),
        "LIGHTCONE_TTS_FOUNDATION_TOOL": str(builder),
        "LIGHTCONE_SCREEN_QUEUE": str(screen),
        "LIGHTCONE_SCREEN_ROOT": str(screen_root),
        "LIGHTCONE_CONFIRMATION_ROOT": str(confirmation_root),
        # Deliberately customize the colliding names. The screen mock asserts
        # that the confirmation values do not leak into its process.
        "LIGHTCONE_ARTIFACT_ROOT": str(confirmation_artifact),
        "LIGHTCONE_ANALYSIS_ROOT": str(confirmation_analysis),
        "LIGHTCONE_TTS_FOUNDATION_ARTIFACT_ROOT": str(confirmation_artifact),
        "LIGHTCONE_TTS_FOUNDATION_ANALYSIS_ROOT": str(confirmation_analysis),
        "LIGHTCONE_TTS_FOUNDATION_COVERAGE": str(confirmation_analysis / "coverage.json"),
        "LIGHTCONE_TTS_FOUNDATION_MANIFEST": str(confirmation_root / "manifest.json"),
        "LIGHTCONE_TTS_FOUNDATION_GENERATION": str(confirmation_root / "manifest-generation.json"),
        "LIGHTCONE_TTS_FOUNDATION_TERMINAL": str(confirmation_root / "formal-acceptance-comparison.json"),
        "LIGHTCONE_QUEUE_LOCK_PATH": str(confirmation_root / ".confirmation.lock"),
        "LIGHTCONE_STATE_PATH": str(confirmation_root / "priority-state.jsonl"),
        "LIGHTCONE_FAILED_RECEIPT": str(confirmation_root / "PRIORITY_FAILED.json"),
        "LIGHTCONE_LOCKFILE": str(lockfile),
        "LIGHTCONE_MODEL_ROOTS": str(model_roots),
        "LIGHTCONE_CUDA_TOOLKIT_ROOT": str(tmp_path / "cuda"),
        "MOCK_COMMAND_LOG": str(log),
        "MOCK_EXPECT_SCREEN_ARTIFACT_ROOT": str(screen_artifact),
        "MOCK_EXPECT_SCREEN_ANALYSIS_ROOT": str(screen_analysis),
        "MOCK_EXPECT_SCREEN_QUEUE_LOCK": str(screen_lock),
        "MOCK_EXPECT_SCREEN_STATE": str(screen_state),
        "MOCK_EXPECT_SCREEN_FAILED": str(screen_failed),
        "MOCK_EXPECT_CONFIRM_ARTIFACT_ROOT": str(confirmation_artifact),
        "MOCK_FAIL_ONCE_MARKER": str(tmp_path / "failed-once.marker"),
        "MOCK_COMPARISON_PASS": "1" if comparison_pass else "0",
        "MOCK_MANAGED_PIDS": str(tmp_path / "managed-pids.json"),
        "MOCK_PARTIAL_EVIDENCE": str(tmp_path / "partial-evidence.jsonl"),
    }
    return {
        "environment": environment,
        "log": log,
        "screen_root": screen_root,
        "confirmation_root": confirmation_root,
        "artifact_root": confirmation_artifact,
    }


def _run_queue(mock: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=mock["environment"],
        text=True,
        capture_output=True,
        check=False,
    )


def _commands(mock: dict[str, object]) -> list[dict[str, object]]:
    log = Path(mock["log"])
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def _assert_attested(path: Path) -> dict[str, object]:
    sidecar = Path(str(path) + ".sha256")
    assert sidecar.read_text(encoding="utf-8").strip() == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def test_confirmation_queue_is_valid_bash_and_hash_bound():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    for value in (
        "CANDIDATE_SCREEN_SELECTED.json",
        "stride-screen.lock.json",
        "stride-screen.model-roots.json",
        "--selected-receipt",
        "--lockfile",
        "--model-roots",
        "--artifact-root",
        "--generation-receipt",
        "--runtime-fingerprint",
        "TTS_0_40K_CONFIRMED",
    ):
        assert value in source


def test_confirmation_queue_preserves_the_formal_load_contract():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "FOUNDATION_STATUS" in source
    assert "--methods static tts" in source.replace("\\\n", "")
    assert "--methods static tts naive_async" not in source.replace("\\\n", "")
    assert "--weight-update-mode lora" in source
    assert "--baseline static" in source
    assert 'PYTORCH_CUDA_ALLOC_CONF="$FOUNDATION_ALLOCATOR"' in source
    assert 'PY_BIN_DIR=$(dirname -- "$PY")' in source
    assert 'PATH="$PY_BIN_DIR:$CUDA_TOOLKIT/bin:$PATH"' in source
    assert "flock -n 9" in source
    assert "exit 42" in source
    assert 'LIGHTCONE_ARTIFACT_ROOT="$SCREEN_ARTIFACT_ROOT"' in source
    assert 'LIGHTCONE_QUEUE_LOCK_PATH="$SCREEN_QUEUE_LOCK"' in source
    assert 'if [ -f "$SELECTED" ] && [ -f "$BLOCKED" ]' in source
    assert "prepare-datasets" not in source


def test_mock_end_to_end_pass_revalidates_terminal_without_rerunning_gpu(
    tmp_path: Path,
):
    mock = _mock_queue(tmp_path)
    first = _run_queue(mock)
    assert first.returncode == 0, first.stderr
    assert "TTS_0_40K_CONFIRMED" in first.stdout
    commands = _commands(mock)
    assert [row["command"] for row in commands].count("lc:run-manifest") == 1
    assert [row["command"] for row in commands].count("lc:validate-artifacts") == 1
    assert [row["command"] for row in commands].count("lc:analyze") == 1
    compares = [row for row in commands if row["command"] == "builder:compare"]
    assert len(compares) == 1
    for name in (
        "--selected-receipt",
        "--lockfile",
        "--model-roots",
        "--artifact-root",
        "--generation-receipt",
        "--foundation-manifest",
        "--analysis-root",
        "--coverage",
    ):
        assert name in compares[0]["args"]

    second = _run_queue(mock)
    assert second.returncode == 0, second.stderr
    commands = _commands(mock)
    assert [row["command"] for row in commands].count("lc:run-manifest") == 1
    assert [row["command"] for row in commands].count("lc:validate-artifacts") == 1
    assert [row["command"] for row in commands].count("lc:analyze") == 1
    assert [row["command"] for row in commands].count("builder:compare") == 2
    state = Path(mock["confirmation_root"]) / "priority-state.jsonl"
    assert any(
        row["status"] == "confirmed"
        for row in map(json.loads, state.read_text(encoding="utf-8").splitlines())
    )


def test_mock_failure_receipt_is_attested_and_rerun_uses_unit_resume(
    tmp_path: Path,
):
    mock = _mock_queue(tmp_path)
    environment = mock["environment"]
    assert isinstance(environment, dict)
    environment["MOCK_FAIL_COMMAND"] = "validate-artifacts"
    failed = _run_queue(mock)
    assert failed.returncode == 23
    failure_path = Path(mock["confirmation_root"]) / "PRIORITY_FAILED.json"
    payload = _assert_attested(failure_path)
    assert payload["status"] == "failed_resumable"
    assert payload["scope"] == "tts_0_40k_foundation_queue"
    assert payload["phase"] == "tts_foundation_validation"
    assert payload["exit_code"] == 23

    resumed = _run_queue(mock)
    assert resumed.returncode == 0, resumed.stderr
    assert not failure_path.exists()
    assert not Path(str(failure_path) + ".sha256").exists()
    command_names = [row["command"] for row in _commands(mock)]
    assert command_names.count("lc:run-manifest") == 2
    assert "lc:run-manifest:executed" in command_names
    assert "lc:run-manifest:resume-skip" in command_names


def test_term_interrupts_only_managed_inference_and_reruns_partial(
    tmp_path: Path,
):
    mock = _mock_queue(tmp_path)
    environment = mock["environment"]
    assert isinstance(environment, dict)
    environment["MOCK_BLOCK_RUN_MANIFEST"] = "1"
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    queue = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid_record = Path(environment["MOCK_MANAGED_PIDS"])
    partial = Path(environment["MOCK_PARTIAL_EVIDENCE"])
    try:
        deadline = time.monotonic() + 10
        while not pid_record.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_record.is_file(), "mock run-manifest did not start"
        managed = json.loads(pid_record.read_text(encoding="utf-8"))
        queue.terminate()
        stdout, stderr = queue.communicate(timeout=10)
        assert queue.returncode == 143, (stdout, stderr)
        assert unrelated.poll() is None
        for pid in managed.values():
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid), 0)

        failure = Path(mock["confirmation_root"]) / "PRIORITY_FAILED.json"
        payload = _assert_attested(failure)
        assert payload["status"] == "failed_resumable"
        assert payload["phase"] == "tts_foundation_inference"
        assert payload["exit_code"] == 143
        state = Path(mock["confirmation_root"]) / "priority-state.jsonl"
        assert any(
            row["status"] == "failed_resumable"
            and "signal=TERM" in row["detail"]
            for row in map(json.loads, state.read_text(encoding="utf-8").splitlines())
        )
        assert partial.read_text(encoding="utf-8") == '{"status":"partial"}\n'
        assert not (Path(mock["artifact_root"]) / "mock-run-complete.json").exists()

        del environment["MOCK_BLOCK_RUN_MANIFEST"]
        resumed = _run_queue(mock)
        assert resumed.returncode == 0, resumed.stderr
        assert (Path(mock["artifact_root"]) / "mock-run-complete.json").is_file()
        assert partial.is_file()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("screen_status", ["blocked", "conflict"])
def test_unattested_or_conflicting_screen_terminal_fails_before_confirmation(
    tmp_path: Path, screen_status: str
):
    mock = _mock_queue(tmp_path)
    environment = mock["environment"]
    assert isinstance(environment, dict)
    environment["MOCK_SCREEN_STATUS"] = screen_status
    result = _run_queue(mock)
    assert result.returncode == (1 if screen_status == "blocked" else 8)
    command_names = [row["command"] for row in _commands(mock)]
    assert not any(name.startswith("builder:") for name in command_names)
    assert not any(name.startswith("lc:") for name in command_names)
    failure = Path(mock["confirmation_root"]) / "PRIORITY_FAILED.json"
    assert _assert_attested(failure)["phase"] == "screen_terminal"


def test_mock_formal_block_is_terminal_but_not_process_success(tmp_path: Path):
    mock = _mock_queue(tmp_path, comparison_pass=False)
    first = _run_queue(mock)
    assert first.returncode == 42
    assert "TTS_0_40K_BLOCKED" in first.stdout
    failure = Path(mock["confirmation_root"]) / "PRIORITY_FAILED.json"
    assert not failure.exists()
    before = [row["command"] for row in _commands(mock)].count("lc:run-manifest")
    second = _run_queue(mock)
    assert second.returncode == 42
    after = [row["command"] for row in _commands(mock)].count("lc:run-manifest")
    assert before == after == 1


def test_mock_lock_conflict_exits_without_claim_or_failure_terminal(tmp_path: Path):
    mock = _mock_queue(tmp_path)
    environment = mock["environment"]
    assert isinstance(environment, dict)
    environment["MOCK_FLOCK_RC"] = "1"
    result = _run_queue(mock)
    assert result.returncode == 75
    assert "already running" in result.stderr
    assert _commands(mock) == []
    assert not (Path(mock["confirmation_root"]) / "PRIORITY_FAILED.json").exists()


def test_orphaned_comparison_sidecar_fails_closed_before_gpu(tmp_path: Path):
    mock = _mock_queue(tmp_path)
    comparison = Path(mock["confirmation_root"]) / "formal-acceptance-comparison.json"
    comparison.parent.mkdir(parents=True, exist_ok=True)
    Path(str(comparison) + ".sha256").write_text("0" * 64 + "\n")
    result = _run_queue(mock)
    assert result.returncode != 0
    assert not any(
        row["command"] == "lc:run-manifest" for row in _commands(mock)
    )
    failure = Path(mock["confirmation_root"]) / "PRIORITY_FAILED.json"
    assert _assert_attested(failure)["phase"] == "tts_foundation_terminal"


def test_tampered_completed_terminal_is_not_trusted_as_a_resume_checkpoint(
    tmp_path: Path,
):
    mock = _mock_queue(tmp_path)
    assert _run_queue(mock).returncode == 0
    comparison = Path(mock["confirmation_root"]) / "formal-acceptance-comparison.json"
    comparison.write_text(
        comparison.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    before = [row["command"] for row in _commands(mock)].count("lc:run-manifest")
    result = _run_queue(mock)
    assert result.returncode == 81
    after = [row["command"] for row in _commands(mock)].count("lc:run-manifest")
    assert before == after == 1
    failure = Path(mock["confirmation_root"]) / "PRIORITY_FAILED.json"
    assert _assert_attested(failure)["phase"] == "tts_foundation_terminal"


def test_path_aliases_fail_before_lock_or_terminal_mutation(tmp_path: Path):
    mock = _mock_queue(tmp_path)
    environment = mock["environment"]
    assert isinstance(environment, dict)
    environment["LIGHTCONE_TTS_FOUNDATION_TERMINAL"] = environment[
        "LIGHTCONE_FAILED_RECEIPT"
    ]
    result = _run_queue(mock)
    assert result.returncode != 0
    assert "path aliases are forbidden" in result.stderr
    assert _commands(mock) == []
    aliased = Path(environment["LIGHTCONE_FAILED_RECEIPT"])
    assert not aliased.exists()
    assert not Path(str(aliased) + ".sha256").exists()
