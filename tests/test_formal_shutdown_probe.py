from __future__ import annotations

import subprocess
from pathlib import Path

from lightcone_spec.orchestration.autodl_provider_runtime import (
    AutoDlPowerOffSafetyProbe,
)
from lightcone_spec.orchestration.experiment_operator import (
    ExperimentOperatorStore,
)
from lightcone_spec.orchestration.formal_shutdown_probe import (
    collect_formal_shutdown_probe,
    shutdown_probe_is_safe,
)

_TEST_INSTANCE_UUID = "pro-test-instance-0001"


def _proc_fixture(root: Path, *, listen_port: int | None = None) -> None:
    (root / "net").mkdir(parents=True)
    row = ""
    if listen_port is not None:
        row = f"  0: 00000000:{listen_port:04X} 00000000:0000 0A 00000000:00000000\n"
    header = "sl local_address rem_address st tx_queue\n"
    (root / "net" / "tcp").write_text(header + row, encoding="ascii")
    (root / "net" / "tcp6").write_text(header, encoding="ascii")


def _empty_nvidia(argv, *, check, capture_output, text, shell):
    assert argv[0] == "nvidia-smi"
    assert check is False and capture_output is True and text is True
    assert shell is False
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def test_shutdown_probe_accepts_only_stable_zero_activity(tmp_path: Path) -> None:
    database = (tmp_path / "operator.sqlite").resolve()
    run_root = (tmp_path / "run").resolve()
    proc_root = (tmp_path / "proc").resolve()
    run_root.mkdir()
    _proc_fixture(proc_root)
    with ExperimentOperatorStore(database, run_id="run-v03") as store:
        store.set_dispatch_stop("final_shutdown", stopped_at_ns=10)
    probe = collect_formal_shutdown_probe(
        database_path=database,
        instance_uuid=_TEST_INSTANCE_UUID,
        run_root=run_root,
        measurement_ports=(31900, 31910),
        proc_root=proc_root,
        command_runner=_empty_nvidia,
        sleeper=lambda _seconds: None,
        clock_ns=lambda: 20,
        self_pid=999,
    )
    assert shutdown_probe_is_safe(probe)
    assert AutoDlPowerOffSafetyProbe(**probe).run_id == "run-v03"


def test_shutdown_probe_readonly_mode_does_not_reopen_write_capable_store(
    tmp_path: Path,
) -> None:
    database = (tmp_path / "operator.sqlite").resolve()
    run_root = (tmp_path / "run").resolve()
    proc_root = (tmp_path / "proc").resolve()
    run_root.mkdir()
    _proc_fixture(proc_root)
    with ExperimentOperatorStore(database, run_id="run-v03") as store:
        store.set_dispatch_stop("scientific_closure", stopped_at_ns=10)
    sqlite_paths = tuple(
        path
        for path in (
            database,
            database.with_name(database.name + "-wal"),
            database.with_name(database.name + "-shm"),
        )
        if path.exists()
    )
    before = {path: path.read_bytes() for path in sqlite_paths}
    probe = collect_formal_shutdown_probe(
        database_path=database,
        instance_uuid=_TEST_INSTANCE_UUID,
        run_root=run_root,
        measurement_ports=(31900, 31910),
        proc_root=proc_root,
        command_runner=_empty_nvidia,
        sleeper=lambda _seconds: None,
        clock_ns=lambda: 20,
        self_pid=999,
        readonly_database=True,
    )
    assert shutdown_probe_is_safe(probe)
    assert {path: path.read_bytes() for path in sqlite_paths} == before


def test_shutdown_probe_preserves_growth_and_open_port_as_unsafe(
    tmp_path: Path,
) -> None:
    database = (tmp_path / "operator.sqlite").resolve()
    run_root = (tmp_path / "run").resolve()
    proc_root = (tmp_path / "proc").resolve()
    run_root.mkdir()
    log = run_root / "writer.log"
    log.write_bytes(b"before")
    _proc_fixture(proc_root, listen_port=31900)
    with ExperimentOperatorStore(database, run_id="run-v03") as store:
        store.set_dispatch_stop("bug_shutdown", stopped_at_ns=10)

    def grow(_seconds: float) -> None:
        with log.open("ab") as handle:
            handle.write(b"-after")

    probe = collect_formal_shutdown_probe(
        database_path=database,
        instance_uuid=_TEST_INSTANCE_UUID,
        run_root=run_root,
        measurement_ports=(31900, 31910),
        proc_root=proc_root,
        command_runner=_empty_nvidia,
        sleeper=grow,
        clock_ns=lambda: 20,
        self_pid=999,
    )
    assert probe["open_measurement_port_count"] == 1
    assert probe["log_growth_bytes"] == len(b"-after")
    assert not shutdown_probe_is_safe(probe)
