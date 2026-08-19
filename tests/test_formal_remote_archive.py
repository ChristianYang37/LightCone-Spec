from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lightcone_spec.orchestration.experiment_operator import ArchiveRequest
from lightcone_spec.orchestration.experiment_operator_production import (
    canonical_json_bytes,
    file_sha256,
)
from lightcone_spec.orchestration.formal_remote_archive import (
    FormalRemoteArchiveError,
    SshRsyncArchiveEndpoint,
    load_remote_archive_result,
    run_remote_archive,
)


def test_remote_pull_archive_records_each_step_and_resumes_without_delete(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote-source"
    remote.mkdir()
    payloads = {"raw/terminal.json": b"terminal\n", "logs/stdout.log": b"log\n"}
    rows = []
    for relative, payload in sorted(payloads.items()):
        path = remote / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest = remote / "sha256_manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "kind": "formal_archive_sha256_manifest",
                "files": rows,
            }
        )
    )
    request = ArchiveRequest(
        archive_id="wave-0001",
        safe_boundary="E3a:sealed",
        remote_payload_root="/srv/lightcone-v03/spool/wave-0001",
        local_partial_root=str((tmp_path / "local.partial").resolve()),
        local_final_root=str((tmp_path / "local.final").resolve()),
        remote_manifest_sha256=file_sha256(manifest),
        predicted_payload_bytes=sum(len(value) for value in payloads.values()),
    )
    endpoint = SshRsyncArchiveEndpoint(
        ssh_target="root@example.test",
        ssh_port=40371,
        remote_python="/srv/lightcone-v03/venv/bin/python",
        remote_operator_script="/srv/lightcone-v03/scripts/operator.py",
        remote_operator_database="/srv/lightcone-v03/operator.sqlite3",
        remote_operator_lock="/srv/lightcone-v03/operator.lock",
    )
    checkpoint: dict[str, object] = {}
    ssh_operations: list[str] = []

    def ssh_runner(argv, *, input, check, capture_output, shell):
        assert check is False and capture_output is True and shell is False
        command = shlex.split(argv[-1])
        operation = command[command.index(endpoint.remote_operator_database) + 3]
        ssh_operations.append(operation)
        if operation == "archive-register":
            supplied = json.loads(input)
            checkpoint.update(
                {
                    **supplied,
                    "state": "REGISTERED",
                    "transfer_receipt": None,
                    "local_sha_receipt": None,
                    "rehydrate_receipt": None,
                }
            )
            output = checkpoint
        elif operation == "archive-record-step":
            receipt = json.loads(input)
            if receipt["step"] == "TRANSFER":
                checkpoint["transfer_receipt"] = receipt
                checkpoint["state"] = "TRANSFERRED"
            elif receipt["step"] == "LOCAL_SHA_VERIFY":
                checkpoint["local_sha_receipt"] = receipt
                checkpoint["state"] = "LOCAL_SHA_VERIFIED"
            else:
                checkpoint["rehydrate_receipt"] = receipt
                checkpoint["state"] = "REHYDRATE_VERIFIED"
            output = checkpoint
        elif operation == "archive-authorize":
            checkpoint["state"] = "EVICTION_AUTHORIZED"
            output = {
                "archive_id": request.archive_id,
                "remote_payload_root": request.remote_payload_root,
                "manifest_sha256": request.remote_manifest_sha256,
                "local_final_root": request.local_final_root,
                "local_sha_evidence_sha256": checkpoint["local_sha_receipt"][
                    "evidence_sha256"
                ],
                "rehydrate_evidence_sha256": checkpoint["rehydrate_receipt"][
                    "evidence_sha256"
                ],
                "rehydrated_content_tree_sha256": checkpoint["rehydrate_receipt"][
                    "content_tree_sha256"
                ],
                "authorized_at_ns": 123,
                "remote_deletion_performed": False,
            }
        else:
            assert operation == "archive-status"
            output = checkpoint
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=canonical_json_bytes(output),
            stderr=b"",
        )

    def rsync_runner(argv, *, check, shell):
        assert check is True and shell is False
        assert argv[-2] == ("root@example.test:/srv/lightcone-v03/spool/wave-0001/")
        destination = Path(argv[-1].removesuffix("/"))
        shutil.copytree(remote, destination, dirs_exist_ok=True)
        return subprocess.CompletedProcess(argv, 0)

    output = (tmp_path / "archive-result.json").resolve()
    result = run_remote_archive(
        endpoint=endpoint,
        request=request,
        result_output_path=output,
        local_lock_path=(tmp_path / "archive.lock").resolve(),
        minimum_local_free_bytes=0,
        ssh_runner=ssh_runner,
        rsync_runner=rsync_runner,
    )
    assert result.remote_deletion_performed is False
    assert remote.is_dir() and all((remote / name).is_file() for name in payloads)
    assert Path(request.local_final_root, "raw", "terminal.json").is_file()
    assert load_remote_archive_result(output) == result
    assert ssh_operations == [
        "archive-register",
        "archive-record-step",
        "archive-record-step",
        "archive-record-step",
        "archive-authorize",
        "archive-status",
    ]

    before = tuple(ssh_operations)
    resumed = run_remote_archive(
        endpoint=endpoint,
        request=request,
        result_output_path=output,
        local_lock_path=(tmp_path / "archive.lock").resolve(),
        minimum_local_free_bytes=0,
        ssh_runner=ssh_runner,
        rsync_runner=rsync_runner,
    )
    assert resumed == result
    assert tuple(ssh_operations) == before


def test_endpoint_requires_batch_safe_identity_file(tmp_path: Path) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("fixture", encoding="utf-8")
    identity.chmod(0o600)
    endpoint = SshRsyncArchiveEndpoint(
        ssh_target="root@example.test",
        ssh_port=40371,
        remote_python="/srv/v03/python",
        remote_operator_script="/srv/v03/operator.py",
        remote_operator_database="/srv/v03/operator.sqlite3",
        remote_operator_lock="/srv/v03/operator.lock",
        ssh_identity_file=str(identity.resolve()),
    )
    assert "BatchMode=yes" in endpoint.ssh_transport_argv
    assert endpoint.rsync_remote_shell.startswith("ssh -p 40371")
    assert sys.executable


def test_archive_capacity_failure_stops_remote_scheduler(tmp_path: Path) -> None:
    request = ArchiveRequest(
        archive_id="wave-capacity-stop",
        safe_boundary="E3a:sealed",
        remote_payload_root="/srv/lightcone-v03/spool/wave-capacity-stop",
        local_partial_root=str((tmp_path / "local.partial").resolve()),
        local_final_root=str((tmp_path / "local.final").resolve()),
        remote_manifest_sha256="a" * 64,
        predicted_payload_bytes=1 << 80,
    )
    endpoint = SshRsyncArchiveEndpoint(
        ssh_target="root@example.test",
        ssh_port=40371,
        remote_python="/srv/v03/python",
        remote_operator_script="/srv/v03/operator.py",
        remote_operator_database="/srv/v03/operator.sqlite3",
        remote_operator_lock="/srv/v03/operator.lock",
    )
    operations: list[str] = []

    def ssh_runner(argv, *, input, check, capture_output, shell):
        del check, capture_output, shell
        command = shlex.split(argv[-1])
        operation = command[command.index(endpoint.remote_operator_database) + 3]
        operations.append(operation)
        if operation == "archive-register":
            output = {
                **json.loads(input),
                "state": "REGISTERED",
                "transfer_receipt": None,
                "local_sha_receipt": None,
                "rehydrate_receipt": None,
            }
        else:
            assert operation == "scheduler-stop"
            output = {"control_state": "STOP"}
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=canonical_json_bytes(output),
            stderr=b"",
        )

    with pytest.raises(
        FormalRemoteArchiveError,
        match="remote dispatch stop was requested",
    ):
        run_remote_archive(
            endpoint=endpoint,
            request=request,
            result_output_path=(tmp_path / "archive-result.json").resolve(),
            local_lock_path=(tmp_path / "archive.lock").resolve(),
            minimum_local_free_bytes=0,
            ssh_runner=ssh_runner,
        )
    assert operations == ["archive-register", "scheduler-stop"]
