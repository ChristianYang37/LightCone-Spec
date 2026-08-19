from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.nonserving_authority import (
    RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON,
    DownloadExecutionBlocked,
    DownloadModelRevision,
    DownloadOutputExpectation,
    DownloadPlan,
    bind_download_plan_authority,
    issue_download_plan,
)
from lightcone_spec.experiments.planning import (
    ZERO_COUNT,
    ZERO_MILLISECONDS,
    BudgetJobKind,
    ExperimentBudget,
    P99AnchorStatus,
    ScenarioMilliseconds,
)
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)
from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment
from lightcone_spec.runtime import download_runner
from lightcone_spec.runtime.download_runner import (
    CPU_DIAGNOSTIC_SELF_REPORTED,
    DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
    RELEASE_DOWNLOAD_PLAN_ALLOWLIST_EMPTY,
    RELEASE_DOWNLOAD_SUBPROCESSES,
    RELEASE_TRUSTED_DOWNLOAD_PLAN_SHA256S,
    DownloadDiagnosticResultPointer,
    DownloadSubprocessLifecycleReceipt,
    execute_download_subprocess_for_cpu_test,
    execute_release_download_plan,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_semantic_json(path: Path, value: object, semantic_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))
    Path(f"{path}.sha256").write_text(f"{semantic_sha256}\n", encoding="ascii")


def _inventory() -> GpuInventory:
    uuids = ("GPU-000", "GPU-001")
    devices = tuple(
        GpuDevice(
            uuid=uuid,
            host_id="host-a",
            model="H100-SXM",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("NVLink4", "PCIe5"),
            peer_access_class="NVSwitch",
            clock_policy="locked-1980MHz",
            power_limit_watts=700.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=("pair-00",),
        )
        for index, uuid in enumerate(uuids)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="pair-00",
                host_id="host-a",
                gpu_uuids=uuids,
                fabric="NVLink",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=content_sha256("runner-inventory-receipt"),
    )


def _download_budget(cell: ExperimentCell) -> ExperimentBudget:
    duration = ScenarioMilliseconds(100, 100, 100)
    gpu_duration = duration.scale(2)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=WorkloadClass.DOWNLOAD,
        job_kind=BudgetJobKind.DOWNLOAD,
        startup_model_load=ZERO_MILLISECONDS,
        compile_jit_graph_prewarm=ZERO_MILLISECONDS,
        excluded_warmup=ZERO_MILLISECONDS,
        excluded_warmup_requests=ZERO_COUNT,
        scored_arrival=ZERO_MILLISECONDS,
        request_deadline=ZERO_MILLISECONDS,
        drain=ZERO_MILLISECONDS,
        reset_finalization=ZERO_MILLISECONDS,
        evidence_flush_shutdown=ZERO_MILLISECONDS,
        output_tokens=ZERO_COUNT,
        minimum_completed_requests=0,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=ZERO_MILLISECONDS,
        failure_injection=ZERO_MILLISECONDS,
        retry=ZERO_MILLISECONDS,
        retry_allowance=0,
        profiler=ZERO_MILLISECONDS,
        download_compile_reservation=duration,
        gpu_count=2,
        topology=cell.identity.topology,
        reserved_gpu_ms=gpu_duration,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=gpu_duration,
    )


def _assignment(
    inventory: GpuInventory,
    budget: ExperimentBudget,
) -> IndustrialPhysicalAssignment:
    return IndustrialPhysicalAssignment(
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        dispatch_plan_sha256=content_sha256("runner-dispatch"),
        experiment_budget_sha256=budget.sha256,
        budget_plan_sha256=content_sha256("runner-budget-plan"),
        capacity_authority_sha256=content_sha256("runner-capacity"),
        budget_materialization_authority_sha256=content_sha256(
            "runner-budget-authority"
        ),
        assignment_sha256=content_sha256("runner-scheduler-assignment"),
        work_item_sha256=content_sha256("runner-work-item"),
        gpu_uuids=("GPU-000", "GPU-001"),
        rank_groups=(("GPU-000", "GPU-001"),),
        ports=(31_000,),
        tensor_parallel_size=2,
        data_parallel_size=1,
        fixed_instance_gpu_count=2,
        host_id="host-a",
        topology_group_ids=(("pair-00",),),
    )


@dataclass(frozen=True)
class _Case:
    registry: ExperimentRegistry
    cell: ExperimentCell
    plan: DownloadPlan
    plan_path: Path


def _case(tmp_path: Path) -> _Case:
    cache_root = (tmp_path / "cache").resolve()
    evidence_root = (tmp_path / "evidence").resolve()
    cache_root.mkdir()
    evidence_root.mkdir()
    registry = build_industrial_registry(
        cache_root=str(cache_root),
        evidence_root=str(evidence_root),
    )
    cell = next(
        value
        for value in registry.cells_for("E6")
        if value.resources.workload_class is WorkloadClass.DOWNLOAD
    )
    inventory = _inventory()
    budget = _download_budget(cell)
    assignment = _assignment(inventory, budget)
    payloads = (
        ("snapshots/config.json", b'{"architectures":["Qwen"]}\n'),
        ("snapshots/model.safetensors", b"locked-model-payload"),
    )
    expectations = tuple(
        DownloadOutputExpectation(
            relative_path=relative_path,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for relative_path, payload in payloads
    )
    plan = issue_download_plan(
        registry=registry,
        cell=cell,
        model_revisions=(
            DownloadModelRevision(
                role="target",
                repository=cell.identity.model,
                revision="1" * 40,
                source_manifest_sha256=content_sha256("runner-model-source"),
            ),
        ),
        inventory=inventory,
        assignment=assignment,
        budget=budget,
        expected_outputs=expectations,
    )
    plan_path = Path(plan.plan_path)
    _write_semantic_json(plan_path, plan.to_dict(), plan.sha256)
    bind_download_plan_authority(plan_path, expected_plan=plan)
    return _Case(registry=registry, cell=cell, plan=plan, plan_path=plan_path)


_CHILD = r"""
import json
import os
import signal
import socket
import subprocess
import sys
import time

for forbidden in (
    "TEST_DOWNLOAD_SECRET_SENTINEL",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "MODELSCOPE_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "PIP_CACHE_DIR",
    "HOME",
    "AWS_SECRET_ACCESS_KEY",
    "OPENAI_API_KEY",
    "PYTHONPATH",
):
    assert forbidden not in os.environ


def send(kind, **values):
    row = {"kind": kind, "protocol_sha256": PROTOCOL, **values}
    sys.stdout.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def receive():
    row = json.loads(sys.stdin.readline())
    assert row["protocol_sha256"] == PROTOCOL
    return row


send("download_subprocess_ready", process_id=os.getpid())
start = receive()
assert start == {
    "kind": "download_subprocess_start",
    "protocol_sha256": PROTOCOL,
    "download_plan_sha256": PLAN,
    "plan_authority_sha256": AUTHORITY,
    "diagnostic_only": True,
    "network_activity_requested": False,
    "payload_materialization_requested": False,
    "activity_observation_available": False,
    "model_revision_manifest_sha256": MODEL_MANIFEST,
    "output_manifest_sha256": OUTPUT_MANIFEST,
}
assert "cache_root" not in start
if MODE == "untrusted_activity":
    with socket.create_connection(("127.0.0.1", NETWORK_PORT), timeout=2.0) as probe:
        probe.sendall(b"actual-network-byte")
    with open(PAYLOADFILE, "wb") as payload_handle:
        payload_handle.write(b"actual-payload-byte")
send(
    "download_subprocess_started",
    download_plan_sha256=PLAN,
    plan_authority_sha256=AUTHORITY,
    process_id=os.getpid(),
    model_revision_manifest_sha256=MODEL_MANIFEST,
    output_manifest_sha256=OUTPUT_MANIFEST,
)
while True:
    row = receive()
    common = {
        "download_plan_sha256": PLAN,
        "plan_authority_sha256": AUTHORITY,
    }
    if row["kind"] == "download_subprocess_model_revision":
        model = row["model_revision"]
        send(
            "download_subprocess_model_revision_accepted",
            **common,
            index=row["index"],
            role=model["role"],
            revision=model["revision"],
            model_revision_sha256=row["model_revision_sha256"],
        )
    elif row["kind"] == "download_subprocess_output_expectation":
        output = row["expectation"]
        size = output["size"]
        if MODE == "bad_bool_size":
            size = False
        send(
            "download_subprocess_output_expectation_accepted",
            **common,
            index=row["index"],
            relative_path=output["relative_path"],
            size=size,
            sha256=output["sha256"],
        )
    else:
        assert row["kind"] == "download_subprocess_drain"
        if MODE == "stubborn_group":
            code = (
                "import os,signal,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                f"open({PIDFILE!r},'w').write(str(os.getpid()));"
                "time.sleep(300)"
            )
            subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
            for _ in range(100):
                if os.path.exists(PIDFILE):
                    break
                time.sleep(0.01)
        send(
            "download_subprocess_drained",
            **common,
            active_transfers=0,
            queued_transfers=0,
            network_requests=0,
            bytes_written=0,
            activity_counter_authority="UNTRUSTED_CHILD_SELF_REPORT",
        )
        break
"""


def _argv(
    case: _Case,
    *,
    mode: str = "success",
    pidfile: Path | None = None,
    network_port: int | None = None,
    payload_file: Path | None = None,
) -> tuple[str, ...]:
    authority = bind_download_plan_authority(
        case.plan_path,
        expected_plan=case.plan,
    )
    executable = str(Path(sys.executable).resolve())
    prefix = (
        f"PROTOCOL={DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256!r};"
        f"PLAN={case.plan.sha256!r};"
        f"AUTHORITY={authority.sha256!r};"
        f"MODEL_MANIFEST={case.plan.inputs.model_revision_manifest_sha256!r};"
        f"OUTPUT_MANIFEST={case.plan.output_manifest_sha256!r};"
        f"MODE={mode!r};"
        f"PIDFILE={'' if pidfile is None else str(pidfile)!r};"
        f"NETWORK_PORT={-1 if network_port is None else network_port!r};"
        f"PAYLOADFILE={'' if payload_file is None else str(payload_file)!r};"
    )
    return (executable, "-c", prefix + _CHILD)


def _rewrite_semantic(path: Path, value: object) -> bytes:
    body = _canonical_bytes(value)
    os.chmod(path, 0o600)
    os.chmod(Path(f"{path}.sha256"), 0o600)
    path.write_bytes(body)
    Path(f"{path}.sha256").write_text(
        f"{content_sha256(value)}\n",
        encoding="ascii",
    )
    return body


def test_formal_entry_blocks_before_plan_path_process_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = (tmp_path / "missing-plan.json").resolve()
    output = (tmp_path / "must-not-exist").resolve()
    assert RELEASE_DOWNLOAD_SUBPROCESSES == ()
    assert RELEASE_TRUSTED_DOWNLOAD_PLAN_SHA256S == ()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("formal DOWNLOAD gate performed a side effect")

    monkeypatch.setattr(download_runner, "_stable_file_bytes", forbidden)
    monkeypatch.setattr(download_runner.subprocess, "Popen", forbidden)
    with pytest.raises(DownloadExecutionBlocked) as blocked:
        execute_release_download_plan(missing, expected_plan=object())  # type: ignore[arg-type]
    assert blocked.value.reason_code == RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON
    assert not missing.exists()
    assert not output.exists()


def test_formal_plan_allowlist_gate_precedes_source_validation_and_plan_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = (tmp_path / "missing-plan.json").resolve()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("formal DOWNLOAD gate performed a side effect")

    monkeypatch.setattr(download_runner, "RELEASE_DOWNLOAD_SUBPROCESSES", (object(),))
    monkeypatch.setattr(download_runner, "_stable_file_bytes", forbidden)
    monkeypatch.setattr(download_runner.subprocess, "Popen", forbidden)
    with pytest.raises(DownloadExecutionBlocked) as blocked:
        execute_release_download_plan(missing, expected_plan=object())  # type: ignore[arg-type]
    assert blocked.value.reason_code == RELEASE_DOWNLOAD_PLAN_ALLOWLIST_EMPTY
    assert not missing.exists()


def test_real_diagnostic_subprocess_covers_plan_publishes_last_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DOWNLOAD_SECRET_SENTINEL", "must-not-reach-child")
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-child")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "must-not-reach-cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "must-not-reach-home"))
    case = _case(tmp_path)
    argv = _argv(case)

    pointer = execute_download_subprocess_for_cpu_test(
        case.plan_path,
        expected_plan=case.plan,
        argv=argv,
    )

    assert pointer.diagnostic_status == CPU_DIAGNOSTIC_SELF_REPORTED
    assert pointer.formal_execution_authorized is False
    assert pointer.source_authority_sha256 is None
    assert not any(Path(case.plan.inputs.cache_root).rglob("*"))
    terminal_path = Path(case.plan.terminal_receipt_path)
    pointer_path = Path(case.plan.result_pointer_path)
    assert stat.S_IMODE(terminal_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(pointer_path.stat().st_mode) == 0o400
    assert Path(f"{terminal_path}.sha256").is_file()
    assert Path(f"{pointer_path}.sha256").is_file()
    assert (
        DownloadDiagnosticResultPointer.load(
            pointer_path,
            expected_plan=case.plan,
        )
        == pointer
    )
    terminal = DownloadSubprocessLifecycleReceipt.from_dict(
        pointer.subprocess_terminal.reopen(
            label="download diagnostic subprocess terminal"
        )
    )
    rows = [json.loads(event.canonical_json) for event in terminal.events]
    assert [row["kind"] for row in rows] == [
        "download_subprocess_ready",
        "download_subprocess_start",
        "download_subprocess_started",
        "download_subprocess_model_revision",
        "download_subprocess_model_revision_accepted",
        "download_subprocess_output_expectation",
        "download_subprocess_output_expectation_accepted",
        "download_subprocess_output_expectation",
        "download_subprocess_output_expectation_accepted",
        "download_subprocess_drain",
        "download_subprocess_drained",
    ]
    assert rows[-1]["activity_counter_authority"] == ("UNTRUSTED_CHILD_SELF_REPORT")

    def forbidden_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact resume spawned a second process")

    monkeypatch.setattr(download_runner.subprocess, "Popen", forbidden_spawn)
    assert (
        execute_download_subprocess_for_cpu_test(
            case.plan_path,
            expected_plan=case.plan,
            argv=argv,
        )
        == pointer
    )


def test_untrusted_child_can_act_but_only_mints_self_reported_cpu_diagnostic(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    payload_file = (tmp_path / "untrusted-child-payload.bin").resolve()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3.0)
        port = listener.getsockname()[1]
        pointer = execute_download_subprocess_for_cpu_test(
            case.plan_path,
            expected_plan=case.plan,
            argv=_argv(
                case,
                mode="untrusted_activity",
                network_port=port,
                payload_file=payload_file,
            ),
            timeout_seconds=3.0,
        )
        connection, _address = listener.accept()
        with connection:
            assert connection.recv(1024) == b"actual-network-byte"

    assert payload_file.read_bytes() == b"actual-payload-byte"
    assert pointer.diagnostic_status == CPU_DIAGNOSTIC_SELF_REPORTED
    assert pointer.formal_execution_authorized is False
    assert pointer.source_authority_sha256 is None
    receipt = DownloadSubprocessLifecycleReceipt.from_dict(
        pointer.subprocess_terminal.reopen(
            label="download diagnostic subprocess terminal"
        )
    )
    drained = json.loads(receipt.events[-1].canonical_json)
    assert drained["network_requests"] == 0
    assert drained["bytes_written"] == 0
    assert drained["activity_counter_authority"] == ("UNTRUSTED_CHILD_SELF_REPORT")
    assert receipt.diagnostic_status == CPU_DIAGNOSTIC_SELF_REPORTED
    assert receipt.formal_execution_authorized is False


def test_bool_typed_protocol_failure_retains_attempt_without_terminal_pointer(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    with pytest.raises(ValueError, match="acknowledgement differs"):
        execute_download_subprocess_for_cpu_test(
            case.plan_path,
            expected_plan=case.plan,
            argv=_argv(case, mode="bad_bool_size"),
        )

    terminal = Path(case.plan.terminal_receipt_path)
    pointer = Path(case.plan.result_pointer_path)
    assert not terminal.exists()
    assert not Path(f"{terminal}.sha256").exists()
    assert not pointer.exists()
    assert not Path(f"{pointer}.sha256").exists()
    attempts = tuple(Path(case.plan.inputs.evidence_root).glob("*.attempt-*.json"))
    assert len(attempts) == 1
    attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert attempt["state"] == "FAILED"
    assert attempt["formal_execution_authorized"] is False


def test_process_group_survivor_is_killed_and_cannot_publish_pointer(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    pidfile = (tmp_path / "stubborn-grandchild.pid").resolve()

    with pytest.raises(ValueError, match="live child process group"):
        execute_download_subprocess_for_cpu_test(
            case.plan_path,
            expected_plan=case.plan,
            argv=_argv(case, mode="stubborn_group", pidfile=pidfile),
            timeout_seconds=3.0,
        )

    grandchild_pid = int(pidfile.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)
    assert not Path(case.plan.result_pointer_path).exists()
    assert tuple(Path(case.plan.inputs.evidence_root).glob("*.attempt-*.json"))


def test_symlink_or_incomplete_pointer_blocks_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    pointer = Path(case.plan.result_pointer_path)
    outside = (tmp_path / "outside.json").resolve()
    outside.write_text("{}\n", encoding="utf-8")
    pointer.symlink_to(outside)
    Path(f"{pointer}.sha256").write_text(f"{'0' * 64}\n", encoding="ascii")

    def forbidden_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("incomplete pointer must block before spawn")

    monkeypatch.setattr(download_runner.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="normalized|non-symlink"):
        execute_download_subprocess_for_cpu_test(
            case.plan_path,
            expected_plan=case.plan,
            argv=_argv(case),
        )


@pytest.mark.parametrize(
    "artifact",
    ("pointer_body", "pointer_sidecar", "terminal_body", "terminal_sidecar"),
)
def test_partial_publication_blocks_resume_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    case = _case(tmp_path)
    pointer = Path(case.plan.result_pointer_path)
    terminal = Path(case.plan.terminal_receipt_path)
    paths = {
        "pointer_body": pointer,
        "pointer_sidecar": Path(f"{pointer}.sha256"),
        "terminal_body": terminal,
        "terminal_sidecar": Path(f"{terminal}.sha256"),
    }
    paths[artifact].write_text("{}\n", encoding="ascii")

    def forbidden_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("partial publication must block before spawn")

    monkeypatch.setattr(download_runner.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="commit marker is incomplete|uncommitted"):
        execute_download_subprocess_for_cpu_test(
            case.plan_path,
            expected_plan=case.plan,
            argv=_argv(case),
        )


@pytest.mark.parametrize(
    "artifact",
    ("pointer", "pointer_sidecar", "terminal", "terminal_sidecar"),
)
def test_hardlinked_evidence_is_not_replay_authority(
    tmp_path: Path,
    artifact: str,
) -> None:
    case = _case(tmp_path)
    pointer = execute_download_subprocess_for_cpu_test(
        case.plan_path,
        expected_plan=case.plan,
        argv=_argv(case),
    )
    targets = {
        "pointer": Path(case.plan.result_pointer_path),
        "pointer_sidecar": Path(f"{case.plan.result_pointer_path}.sha256"),
        "terminal": Path(pointer.subprocess_terminal.path),
        "terminal_sidecar": Path(f"{pointer.subprocess_terminal.path}.sha256"),
    }
    target = targets[artifact]
    alias = (tmp_path / f"{artifact}-hardlink.json").resolve()
    os.link(target, alias)

    with pytest.raises(ValueError, match="regular file"):
        DownloadDiagnosticResultPointer.load(
            case.plan.result_pointer_path,
            expected_plan=case.plan,
        )


@pytest.mark.parametrize("sidecar", (False, True))
def test_hardlinked_plan_is_rejected_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar: bool,
) -> None:
    case = _case(tmp_path)
    target = Path(f"{case.plan_path}.sha256") if sidecar else case.plan_path
    os.link(target, (tmp_path / "plan-hardlink").resolve())

    def forbidden_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("hardlinked plan must block before spawn")

    monkeypatch.setattr(download_runner.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="regular file"):
        execute_download_subprocess_for_cpu_test(
            case.plan_path,
            expected_plan=case.plan,
            argv=_argv(case),
        )


def test_coordinated_rehash_cannot_promote_diagnostic_terminal_or_pointer(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    pointer = execute_download_subprocess_for_cpu_test(
        case.plan_path,
        expected_plan=case.plan,
        argv=_argv(case),
    )
    terminal_path = Path(pointer.subprocess_terminal.path)
    pointer_path = Path(case.plan.result_pointer_path)
    terminal_row = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_row["formal_execution_authorized"] = True
    terminal_row["source_authority_sha256"] = "f" * 64
    terminal_body = _rewrite_semantic(terminal_path, terminal_row)

    pointer_row = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer_row["formal_execution_authorized"] = True
    pointer_row["source_authority_sha256"] = "f" * 64
    pointer_row["subprocess_lifecycle_sha256"] = content_sha256(terminal_row)
    terminal_binding = pointer_row["subprocess_terminal"]
    assert isinstance(terminal_binding, dict)
    terminal_binding["raw_sha256"] = hashlib.sha256(terminal_body).hexdigest()
    terminal_binding["semantic_sha256"] = content_sha256(terminal_row)
    terminal_binding["sidecar_raw_sha256"] = hashlib.sha256(
        f"{content_sha256(terminal_row)}\n".encode("ascii")
    ).hexdigest()
    _rewrite_semantic(pointer_path, pointer_row)

    with pytest.raises(DownloadExecutionBlocked) as blocked:
        DownloadDiagnosticResultPointer.load(
            pointer_path,
            expected_plan=case.plan,
        )
    assert blocked.value.reason_code == RELEASE_DOWNLOADER_TERMINAL_UNAVAILABLE_REASON


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", True, "schema is unsupported"),
        ("formal_execution_authorized", 0, "formal flag must be boolean"),
    ),
)
def test_pointer_schema_rejects_bool_integer_aliases(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    case = _case(tmp_path)
    pointer = execute_download_subprocess_for_cpu_test(
        case.plan_path,
        expected_plan=case.plan,
        argv=_argv(case),
    )
    raw = pointer.to_dict()
    raw[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        DownloadDiagnosticResultPointer.from_dict(raw)


def test_terminal_schema_rejects_bool_integer_aliases_and_status_promotion(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    pointer = execute_download_subprocess_for_cpu_test(
        case.plan_path,
        expected_plan=case.plan,
        argv=_argv(case),
    )
    raw = pointer.subprocess_terminal.reopen(
        label="download diagnostic subprocess terminal"
    )
    for field, value, message in (
        ("schema_version", True, "schema is unsupported"),
        ("process_id", True, "process ID is invalid"),
        ("exit_code", False, "requires zero exit"),
        ("formal_execution_authorized", 0, "formal flag must be boolean"),
        ("diagnostic_status", "FORMAL", "diagnostic status is unsupported"),
    ):
        tampered = dict(raw)
        tampered[field] = value
        with pytest.raises((TypeError, ValueError), match=message):
            DownloadSubprocessLifecycleReceipt.from_dict(tampered)


def test_joint_plan_and_sidecar_rehash_is_not_expected_plan_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    argv = _argv(case)
    raw = case.plan.to_dict()
    inputs = raw["inputs"]
    assert isinstance(inputs, dict)
    inputs["assignment_sha256"] = "f" * 64
    forged = DownloadPlan.from_dict(raw)
    _write_semantic_json(case.plan_path, raw, forged.sha256)

    def forbidden_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tampered plan must block before spawn")

    monkeypatch.setattr(download_runner.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="release-derived inputs"):
        execute_download_subprocess_for_cpu_test(
            case.plan_path,
            expected_plan=case.plan,
            argv=argv,
        )
