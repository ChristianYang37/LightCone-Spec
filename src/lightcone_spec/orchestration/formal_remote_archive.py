"""Local pull-archiver for a remote formal experiment spool.

The remote SQLite operator remains authoritative.  This local process registers
the safe boundary over SSH, pulls with checksum-enabled rsync into ``.partial``,
deep-verifies and atomically publishes the local copy, performs a rehydrate
check, and records each receipt back into the remote WAL.  It never deletes the
remote payload; the final response is only an eviction authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lightcone_spec.orchestration.experiment_operator import (
    ArchiveRequest,
    ArchiveStepReceipt,
    RemoteEvictionAuthorization,
    SingletonOperatorLock,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    MINIMUM_LOCAL_ARCHIVE_FREE_BYTES,
    ProductionArchiveRuntime,
    ProductionCallbackError,
    canonical_json_bytes,
)

_REMOTE_IDENTITY = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$")
_SAFE_REMOTE_ARGUMENT = re.compile(r"^[A-Za-z0-9_./:@+-]+$")


class FormalRemoteArchiveError(RuntimeError):
    """A remote archive transport or receipt failed closed."""


@dataclass(frozen=True)
class SshRsyncArchiveEndpoint:
    ssh_target: str
    ssh_port: int
    remote_python: str
    remote_operator_script: str
    remote_operator_database: str
    remote_operator_lock: str
    ssh_executable: str = "ssh"
    rsync_executable: str = "rsync"
    ssh_identity_file: str | None = None

    def __post_init__(self) -> None:
        if _REMOTE_IDENTITY.fullmatch(self.ssh_target) is None:
            raise ValueError("archive SSH target must be canonical user@host")
        if (
            isinstance(self.ssh_port, bool)
            or not isinstance(self.ssh_port, int)
            or not 1 <= self.ssh_port <= 65535
        ):
            raise ValueError("archive SSH port is invalid")
        for label, value in (
            ("remote Python", self.remote_python),
            ("remote operator script", self.remote_operator_script),
            ("remote operator database", self.remote_operator_database),
            ("remote operator lock", self.remote_operator_lock),
        ):
            path = Path(value)
            if (
                not path.is_absolute()
                or path != path.resolve(strict=False)
                or _SAFE_REMOTE_ARGUMENT.fullmatch(value) is None
            ):
                raise ValueError(f"{label} must be a canonical absolute path")
        for label, value in (
            ("SSH executable", self.ssh_executable),
            ("rsync executable", self.rsync_executable),
        ):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{label} is invalid")
        if self.ssh_identity_file is not None:
            identity = Path(self.ssh_identity_file)
            if (
                not identity.is_absolute()
                or identity != identity.resolve(strict=False)
                or identity.is_symlink()
                or not identity.is_file()
                or stat.S_IMODE(identity.stat().st_mode) & 0o077
            ):
                raise ValueError(
                    "archive SSH identity must be a normalized private 0600 file"
                )

    @property
    def ssh_transport_argv(self) -> tuple[str, ...]:
        values = [
            self.ssh_executable,
            "-p",
            str(self.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=20",
        ]
        if self.ssh_identity_file is not None:
            values.extend(("-i", self.ssh_identity_file))
        return tuple(values)

    @property
    def rsync_remote_shell(self) -> str:
        return shlex.join(self.ssh_transport_argv)

    def remote_archive_source(self, request: ArchiveRequest) -> str:
        return f"{self.ssh_target}:{request.remote_payload_root}"


@dataclass(frozen=True)
class RemoteArchiveResult:
    schema_version: int
    kind: str
    archive_id: str
    ssh_target: str
    ssh_port: int
    remote_payload_root: str
    local_final_root: str
    manifest_sha256: str
    checked_file_count: int
    checked_bytes: int
    rehydrated_content_tree_sha256: str
    authorized_at_ns: int
    remote_deletion_performed: bool

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_remote_archive_result"
            or self.remote_deletion_performed is not False
        ):
            raise ValueError("remote archive result schema differs")
        if (
            not self.archive_id
            or not 1 <= self.ssh_port <= 65535
            or _REMOTE_IDENTITY.fullmatch(self.ssh_target) is None
            or Path(self.remote_payload_root)
            != Path(self.remote_payload_root).resolve(strict=False)
            or Path(self.local_final_root)
            != Path(self.local_final_root).resolve(strict=False)
            or not _is_sha256(self.manifest_sha256)
            or not _is_sha256(self.rehydrated_content_tree_sha256)
            or self.checked_file_count < 0
            or self.checked_bytes < 0
            or self.authorized_at_ns <= 0
        ):
            raise ValueError("remote archive result identity differs")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()


class SshRemoteOperatorClient:
    def __init__(
        self,
        endpoint: SshRsyncArchiveEndpoint,
        *,
        runner: Any = subprocess.run,
    ) -> None:
        self.endpoint = endpoint
        self.runner = runner

    def call(
        self,
        operation: str,
        arguments: tuple[str, ...],
        *,
        stdin_object: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if _SAFE_REMOTE_ARGUMENT.fullmatch(operation) is None or any(
            _SAFE_REMOTE_ARGUMENT.fullmatch(value) is None for value in arguments
        ):
            raise ValueError("remote operator arguments are not canonical")
        remote_argv = (
            self.endpoint.remote_python,
            self.endpoint.remote_operator_script,
            "--db",
            self.endpoint.remote_operator_database,
            "--lock",
            self.endpoint.remote_operator_lock,
            operation,
            *arguments,
        )
        remote_command = shlex.join(remote_argv)
        completed = self.runner(
            [
                *self.endpoint.ssh_transport_argv,
                self.endpoint.ssh_target,
                remote_command,
            ],
            input=(
                None if stdin_object is None else canonical_json_bytes(stdin_object)
            ),
            check=False,
            capture_output=True,
            shell=False,
        )
        if completed.returncode != 0:
            raise FormalRemoteArchiveError(
                f"remote operator command failed with exit {completed.returncode}"
            )
        try:
            value = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FormalRemoteArchiveError(
                "remote operator returned invalid JSON"
            ) from error
        if type(value) is not dict:
            raise FormalRemoteArchiveError("remote operator response is not an object")
        return value


def run_remote_archive(
    *,
    endpoint: SshRsyncArchiveEndpoint,
    request: ArchiveRequest,
    result_output_path: str | Path,
    local_lock_path: str | Path,
    minimum_local_free_bytes: int = MINIMUM_LOCAL_ARCHIVE_FREE_BYTES,
    ssh_runner: Any = subprocess.run,
    rsync_runner: Any = subprocess.run,
) -> RemoteArchiveResult:
    """Resume one exact remote-safe-boundary archive to authorization."""

    client = SshRemoteOperatorClient(endpoint, runner=ssh_runner)
    output = Path(result_output_path)
    if not output.is_absolute() or output != output.resolve(strict=False):
        raise ValueError("remote archive result path must be absolute and normalized")
    runtime = ProductionArchiveRuntime(
        rsync_executable=endpoint.rsync_executable,
        runner=rsync_runner,
        full_rehydrate=True,
        minimum_local_free_bytes=minimum_local_free_bytes,
        rsync_source=endpoint.remote_archive_source,
        rsync_remote_shell=endpoint.rsync_remote_shell,
    )
    with SingletonOperatorLock(local_lock_path):
        if output.exists() or output.is_symlink():
            return load_remote_archive_result(output)
        checkpoint = client.call(
            "archive-register",
            ("--request", "-"),
            stdin_object=asdict(request),
        )
        while True:
            _validate_checkpoint(checkpoint, request=request)
            state = checkpoint["state"]
            if state == "REGISTERED":
                receipt = _archive_step_or_stop(
                    client,
                    operation=lambda: runtime.transfer(request, None),
                )
                checkpoint = _record_remote_step(client, request, receipt)
                continue
            if state == "TRANSFERRED":
                prior = ArchiveStepReceipt(**checkpoint["transfer_receipt"])
                receipt = _archive_step_or_stop(
                    client,
                    operation=lambda prior=prior: runtime.verify_local_sha(
                        request, prior
                    ),
                )
                checkpoint = _record_remote_step(client, request, receipt)
                continue
            if state == "LOCAL_SHA_VERIFIED":
                prior = ArchiveStepReceipt(**checkpoint["local_sha_receipt"])
                receipt = _archive_step_or_stop(
                    client,
                    operation=lambda prior=prior: runtime.rehydrate(request, prior),
                )
                checkpoint = _record_remote_step(client, request, receipt)
                continue
            if state in {"REHYDRATE_VERIFIED", "EVICTION_AUTHORIZED"}:
                authorization_value = client.call(
                    "archive-authorize",
                    ("--archive-id", request.archive_id),
                )
                authorization_value.pop("remote_deletion_performed", None)
                authorization = RemoteEvictionAuthorization(**authorization_value)
                if (
                    authorization.archive_id != request.archive_id
                    or authorization.remote_payload_root != request.remote_payload_root
                    or authorization.manifest_sha256 != request.remote_manifest_sha256
                    or authorization.local_final_root != request.local_final_root
                ):
                    raise FormalRemoteArchiveError(
                        "remote eviction authorization differs from the archive"
                    )
                final_checkpoint = client.call(
                    "archive-status",
                    ("--archive-id", request.archive_id),
                )
                _validate_checkpoint(final_checkpoint, request=request)
                receipt = ArchiveStepReceipt(**final_checkpoint["rehydrate_receipt"])
                if receipt.content_tree_sha256 is None:
                    raise AssertionError("rehydrate receipt lacks content-tree SHA")
                result = RemoteArchiveResult(
                    schema_version=1,
                    kind="formal_remote_archive_result",
                    archive_id=request.archive_id,
                    ssh_target=endpoint.ssh_target,
                    ssh_port=endpoint.ssh_port,
                    remote_payload_root=request.remote_payload_root,
                    local_final_root=request.local_final_root,
                    manifest_sha256=request.remote_manifest_sha256,
                    checked_file_count=receipt.checked_file_count,
                    checked_bytes=receipt.checked_bytes,
                    rehydrated_content_tree_sha256=receipt.content_tree_sha256,
                    authorized_at_ns=authorization.authorized_at_ns,
                    remote_deletion_performed=False,
                )
                _publish_result(output, result)
                return result
            raise FormalRemoteArchiveError("remote archive checkpoint state differs")


def _record_remote_step(
    client: SshRemoteOperatorClient,
    request: ArchiveRequest,
    receipt: ArchiveStepReceipt,
) -> dict[str, Any]:
    return client.call(
        "archive-record-step",
        ("--archive-id", request.archive_id, "--receipt", "-"),
        stdin_object=asdict(receipt),
    )


def _archive_step_or_stop(
    client: SshRemoteOperatorClient,
    *,
    operation: Any,
) -> ArchiveStepReceipt:
    try:
        receipt = operation()
        if type(receipt) is not ArchiveStepReceipt:
            raise TypeError("archive operation returned another receipt type")
        return receipt
    except (OSError, TypeError, ValueError, ProductionCallbackError) as error:
        try:
            client.call(
                "scheduler-stop",
                ("--reason", "archive_pull_or_verification_failed"),
            )
        except BaseException as stop_error:  # noqa: BLE001
            error.add_note(
                f"remote scheduler-stop also failed: {type(stop_error).__name__}"
            )
        raise FormalRemoteArchiveError(
            "archive step failed; remote dispatch stop was requested"
        ) from error


def _validate_checkpoint(value: dict[str, Any], *, request: ArchiveRequest) -> None:
    if (
        value.get("archive_id") != request.archive_id
        or value.get("remote_payload_root") != request.remote_payload_root
        or value.get("local_partial_root") != request.local_partial_root
        or value.get("local_final_root") != request.local_final_root
        or value.get("remote_manifest_sha256") != request.remote_manifest_sha256
        or value.get("predicted_payload_bytes") != request.predicted_payload_bytes
        or value.get("state")
        not in {
            "REGISTERED",
            "TRANSFERRED",
            "LOCAL_SHA_VERIFIED",
            "REHYDRATE_VERIFIED",
            "EVICTION_AUTHORIZED",
        }
    ):
        raise FormalRemoteArchiveError("remote archive checkpoint identity differs")


def _publish_result(path: Path, result: RemoteArchiveResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes({**asdict(result), "result_sha256": result.sha256})
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_remote_archive_result(path: str | Path) -> RemoteArchiveResult:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FormalRemoteArchiveError("remote archive result is not a regular file")
    payload = source.read_bytes()
    value = json.loads(payload)
    if type(value) is not dict or payload != canonical_json_bytes(value):
        raise FormalRemoteArchiveError("remote archive result is not canonical")
    expected = value.pop("result_sha256", None)
    result = RemoteArchiveResult(**value)
    if expected != result.sha256:
        raise FormalRemoteArchiveError("remote archive result digest differs")
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument(
        "--minimum-local-free-bytes",
        type=int,
        default=MINIMUM_LOCAL_ARCHIVE_FREE_BYTES,
    )
    return parser


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_bytes())
    if type(value) is not dict:
        raise ValueError("remote archive input must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_remote_archive(
            endpoint=SshRsyncArchiveEndpoint(**_load_object(arguments.endpoint)),
            request=ArchiveRequest(**_load_object(arguments.request)),
            result_output_path=arguments.result,
            local_lock_path=arguments.lock,
            minimum_local_free_bytes=arguments.minimum_local_free_bytes,
        )
    except (OSError, TypeError, ValueError, FormalRemoteArchiveError) as error:
        print(f"formal remote archive: {error}", file=os.sys.stderr)
        return 2
    print(
        json.dumps({**asdict(result), "result_sha256": result.sha256}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FormalRemoteArchiveError",
    "RemoteArchiveResult",
    "SshRemoteOperatorClient",
    "SshRsyncArchiveEndpoint",
    "load_remote_archive_result",
    "run_remote_archive",
]
