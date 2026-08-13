from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

import lightcone_spec.orchestration.remote_dispatch as remote_dispatch_module
from lightcone_spec.experiments.gpu_pool import (
    AssignmentExecutionReceipt,
    AssignmentExecutionStatus,
    DispatchExecutionPhase,
    DispatchScheduleReceipt,
    DispatchWaveExecutionReceipt,
)
from lightcone_spec.orchestration.remote_dispatch import (
    CrossHostCollectivesUnvalidated,
    FleetWaveOutcome,
    HostAssignmentBinding,
    RemoteFleetWaveReceipt,
    RemoteHostExecutionBinding,
    RemoteHostWaveRequest,
    RemoteHostWaveResponse,
    RemoteHostWaveResult,
    RemoteTransportOutcome,
    RemoteWorkerStatus,
    SshHostRoute,
    SshProcessResult,
    build_ssh_argv,
    canonical_json_bytes,
    canonical_sha256,
    decode_remote_host_wave_request,
    execute_fleet_wave,
    execute_host_local_wave_request,
    execute_remote_host_wave,
)


def _digest(label: str) -> str:
    return canonical_sha256({"test": label})


class _AgentSocket:
    def __init__(self, host_id: str) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="lightcone-remote-")).resolve()
        self.path = self.root / f"{host_id}.sock"
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.bind(str(self.path))

    def close(self) -> None:
        self.socket.close()
        self.path.unlink(missing_ok=True)
        self.root.rmdir()


def _route(
    tmp_path: Path,
    *,
    host_id: str = "host-a",
    destination: str = "runner@node-a.example",
) -> tuple[SshHostRoute, _AgentSocket]:
    known_hosts = (tmp_path / f"{host_id}.known_hosts").resolve()
    known_hosts.write_text(
        "node-a.example ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest\n",
        encoding="ascii",
    )
    known_hosts.chmod(0o600)
    agent = _AgentSocket(host_id)
    return (
        SshHostRoute(
            host_id=host_id,
            destination=destination,
            known_hosts_path=str(known_hosts),
            agent_socket_path=str(agent.path),
            port=22022,
        ),
        agent,
    )


def _binding(host_id: str, *, port: int = 31_000) -> RemoteHostExecutionBinding:
    assignment = HostAssignmentBinding(
        assignment_sha256=_digest(f"assignment-{host_id}"),
        gpu_host_bindings=((f"GPU-{host_id}", host_id),),
        ports=(port,),
        cache_namespace=f"runtime-cache/{host_id}/cell",
        evidence_namespace=f"evidence/{host_id}/cell",
    )
    return RemoteHostExecutionBinding(
        schema_version=1,
        host_id=host_id,
        fleet_inventory_sha256=_digest("fleet"),
        host_inventory_sha256=_digest(f"inventory-{host_id}"),
        dispatch_plan_sha256=_digest("plan"),
        wave_index=3,
        wave_sha256=_digest("wave-3"),
        execution_bundle_manifest_path=(
            f"/srv/lightcone/{host_id}/dispatch-execution-bundle-manifest.json"
        ),
        execution_bundle_manifest_sha256=_digest(f"manifest-{host_id}"),
        receipt_output_path=f"/srv/lightcone/{host_id}/wave-3-receipt.json",
        resume_receipt_path=None,
        assignments=(assignment,),
    )


def _request(host_id: str, *, port: int = 31_000) -> RemoteHostWaveRequest:
    return RemoteHostWaveRequest(
        schema_version=1,
        challenge_nonce_sha256=_digest(f"nonce-{host_id}"),
        binding=_binding(host_id, port=port),
    )


def _remote_response(
    request: RemoteHostWaveRequest,
    *,
    status: str = "SUCCEEDED",
    reason_code: str | None = None,
) -> bytes:
    succeeded = status == "SUCCEEDED"
    value = {
        "schema_version": 1,
        "kind": "lightcone_remote_host_wave_response",
        "host_id": request.binding.host_id,
        "request_sha256": request.sha256,
        "binding_sha256": request.binding.sha256,
        "status": status,
        "reason_code": reason_code,
        "dispatch_schedule_receipt_sha256": (
            _digest(f"schedule-{request.binding.host_id}") if succeeded else None
        ),
        "completed_assignment_sha256": (
            list(request.binding.assignment_sha256) if succeeded else []
        ),
        "failed_assignment_sha256": [],
    }
    return canonical_json_bytes(value) + b"\n"


class _CapturingTransport:
    def __init__(self, result: SshProcessResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def run(self, **kwargs: object) -> SshProcessResult:
        self.calls.append(kwargs)
        return self.result


def test_route_enforces_agent_known_hosts_and_fixed_ssh_policy(tmp_path: Path) -> None:
    route, agent = _route(tmp_path)
    try:
        argv = build_ssh_argv(route)
        assert argv[:4] == ("ssh", "-F", "/dev/null", "-T")
        assert "BatchMode=yes" in argv
        assert "PasswordAuthentication=no" in argv
        assert "KbdInteractiveAuthentication=no" in argv
        assert "StrictHostKeyChecking=yes" in argv
        assert "ForwardAgent=no" in argv
        assert argv[-4:] == (
            route.destination,
            "lightcone-spec",
            "execute-dispatch-wave",
            "--host-request-stdin",
        )
        assert route.destination not in repr(route)
        assert route.known_hosts_path not in repr(route)
        assert route.agent_socket_path not in repr(route)

        symlink = tmp_path / "known-hosts-link"
        symlink.symlink_to(route.known_hosts_path)
        with pytest.raises(ValueError, match="known_hosts_path"):
            replace(route, known_hosts_path=str(symlink))
        with pytest.raises(ValueError, match="safe user/host"):
            replace(route, destination="-oProxyCommand=private-marker")
    finally:
        agent.close()


def test_binding_is_canonical_host_local_and_rejects_cross_host_gang() -> None:
    binding = _binding("host-a")
    assert RemoteHostExecutionBinding.from_dict(binding.to_dict()) == binding
    assert binding.execution_bundle_manifest_path.startswith("/")
    encoded = canonical_json_bytes(binding.to_dict())
    assert b"runner@" not in encoded
    assert b'"execution_bundle"' not in encoded

    with pytest.raises(ValueError, match="absolute canonical POSIX"):
        replace(binding, execution_bundle_manifest_path="bundles/manifest.json")
    with pytest.raises(CrossHostCollectivesUnvalidated) as error:
        HostAssignmentBinding(
            assignment_sha256=_digest("cross-host"),
            gpu_host_bindings=(("GPU-a", "host-a"), ("GPU-b", "host-b")),
            ports=(31_000,),
            cache_namespace="cache/cross-host",
            evidence_namespace="evidence/cross-host",
        )
    assert error.value.reason_code == "cross_host_collectives_unvalidated"


def test_request_stdin_is_exact_canonical_control_data_only() -> None:
    request = _request("host-a")
    stdin = request.canonical_stdin()
    assert stdin == canonical_json_bytes(request.to_dict()) + b"\n"
    assert RemoteHostWaveRequest.from_dict(json.loads(stdin)) == request
    assert b"dispatch-execution-bundle-manifest.json" in stdin
    assert b'"binding_sha256"' in stdin
    assert b'"bundle_bytes"' not in stdin
    assert decode_remote_host_wave_request(stdin) == request
    with pytest.raises(ValueError, match="not canonical"):
        decode_remote_host_wave_request(
            json.dumps(request.to_dict(), indent=2).encode() + b"\n"
        )


def test_host_worker_verifies_local_manifest_before_executor(tmp_path: Path) -> None:
    manifest = (tmp_path / "dispatch-execution-bundle-manifest.json").resolve()
    manifest_value = {"kind": "test-host-manifest", "schema_version": 1}
    manifest.write_bytes(canonical_json_bytes(manifest_value) + b"\n")
    request = _request("host-a")
    request = replace(
        request,
        binding=replace(
            request.binding,
            execution_bundle_manifest_path=str(manifest),
            execution_bundle_manifest_sha256=canonical_sha256(manifest_value),
            receipt_output_path=str((tmp_path / "receipt.json").resolve()),
        ),
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def fake_execute(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return object()

    exit_code, stdout = asyncio.run(
        execute_host_local_wave_request(
            request.canonical_stdin(), execute_wave=fake_execute
        )
    )
    response = RemoteHostWaveResponse.from_dict(json.loads(stdout))
    assert exit_code == 42
    assert response.status is RemoteWorkerStatus.FAILED
    assert response.reason_code == "remote_host_receipt_invalid"
    assert calls == [
        (
            (str(manifest),),
            {
                "wave_index": request.binding.wave_index,
                "receipt_output": request.binding.receipt_output_path,
                "resume_receipt_path": None,
            },
        )
    ]

    tampered = replace(
        request,
        binding=replace(
            request.binding,
            execution_bundle_manifest_sha256=_digest("wrong-manifest"),
        ),
    )
    calls.clear()
    exit_code, stdout = asyncio.run(
        execute_host_local_wave_request(
            tampered.canonical_stdin(), execute_wave=fake_execute
        )
    )
    response = RemoteHostWaveResponse.from_dict(json.loads(stdout))
    assert exit_code == 42
    assert response.status is RemoteWorkerStatus.BLOCKED
    assert response.reason_code == "remote_host_manifest_invalid"
    assert calls == []


def test_host_worker_rejects_hard_linked_manifest(tmp_path: Path) -> None:
    manifest = (tmp_path / "dispatch-execution-bundle-manifest.json").resolve()
    manifest_value = {"kind": "test-host-manifest", "schema_version": 1}
    manifest.write_bytes(canonical_json_bytes(manifest_value) + b"\n")
    os.link(manifest, tmp_path / "second-name.json")
    request = _request("host-a")
    request = replace(
        request,
        binding=replace(
            request.binding,
            execution_bundle_manifest_path=str(manifest),
            execution_bundle_manifest_sha256=canonical_sha256(manifest_value),
            receipt_output_path=str((tmp_path / "receipt.json").resolve()),
        ),
    )
    called = False

    async def fake_execute(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    exit_code, stdout = asyncio.run(
        execute_host_local_wave_request(
            request.canonical_stdin(), execute_wave=fake_execute
        )
    )
    response = RemoteHostWaveResponse.from_dict(json.loads(stdout))
    assert exit_code == 42
    assert response.status is RemoteWorkerStatus.BLOCKED
    assert response.reason_code == "remote_host_manifest_invalid"
    assert not called


def test_host_worker_rejects_leaf_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = (tmp_path / "dispatch-execution-bundle-manifest.json").resolve()
    manifest_value = {"kind": "test-host-manifest", "schema_version": 1}
    manifest_body = canonical_json_bytes(manifest_value) + b"\n"
    manifest.write_bytes(manifest_body)
    replacement = (tmp_path / "replacement.json").resolve()
    replacement.write_bytes(manifest_body)
    request = _request("host-a")
    request = replace(
        request,
        binding=replace(
            request.binding,
            execution_bundle_manifest_path=str(manifest),
            execution_bundle_manifest_sha256=canonical_sha256(manifest_value),
            receipt_output_path=str((tmp_path / "receipt.json").resolve()),
        ),
    )
    original_read = remote_dispatch_module.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if chunk and not replaced:
            replaced = True
            replacement.replace(manifest)
        return chunk

    monkeypatch.setattr(remote_dispatch_module.os, "read", replacing_read)
    called = False

    async def fake_execute(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    exit_code, stdout = asyncio.run(
        execute_host_local_wave_request(
            request.canonical_stdin(), execute_wave=fake_execute
        )
    )
    response = RemoteHostWaveResponse.from_dict(json.loads(stdout))
    assert exit_code == 42
    assert response.status is RemoteWorkerStatus.BLOCKED
    assert response.reason_code == "remote_host_manifest_invalid"
    assert replaced
    assert not called


def test_host_worker_preserves_failed_schedule_receipt(tmp_path: Path) -> None:
    manifest = (tmp_path / "dispatch-execution-bundle-manifest.json").resolve()
    manifest_value = {"kind": "test-host-manifest", "schema_version": 1}
    manifest.write_bytes(canonical_json_bytes(manifest_value) + b"\n")
    request = _request("host-a")
    binding = replace(
        request.binding,
        wave_index=0,
        wave_sha256=_digest("wave-0"),
        execution_bundle_manifest_path=str(manifest),
        execution_bundle_manifest_sha256=canonical_sha256(manifest_value),
        receipt_output_path=str((tmp_path / "receipt.json").resolve()),
    )
    request = replace(request, binding=binding)
    assignment_receipt = AssignmentExecutionReceipt(
        plan_sha256=binding.dispatch_plan_sha256,
        wave_sha256=binding.wave_sha256,
        assignment_sha256=binding.assignment_sha256[0],
        budget_sha256=_digest("budget"),
        attempt=1,
        status=AssignmentExecutionStatus.FAILED,
        terminal_receipt_sha256=None,
        terminal_binding=None,
        failure_sha256=_digest("failure"),
        prior_attempt_receipt_sha256=None,
        gpu_count=1,
        fixed_instance_gpu_count=1,
        attempt_intervals_monotonic_ns=((10, 20),),
        attributed_gpu_ns=10,
        attributed_fixed_instance_gpu_ns=10,
    )
    wave_receipt = DispatchWaveExecutionReceipt(
        plan_sha256=binding.dispatch_plan_sha256,
        wave_index=0,
        wave_sha256=binding.wave_sha256,
        assignment_receipts=(assignment_receipt,),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((10, 20),),
        fixed_instance_actual_billed_gpu_ns=10,
        per_assignment_attributed_gpu_ns=10,
        per_assignment_attributed_fixed_instance_gpu_ns=10,
    )
    schedule_receipt = DispatchScheduleReceipt(
        plan_sha256=binding.dispatch_plan_sha256,
        phase=DispatchExecutionPhase.FAILED,
        wave_receipts=(wave_receipt,),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((10, 20),),
        fixed_instance_actual_billed_gpu_ns=10,
        per_assignment_attributed_gpu_ns=10,
        per_assignment_attributed_fixed_instance_gpu_ns=10,
    )

    async def fake_execute(*args: object, **kwargs: object) -> object:
        return schedule_receipt

    exit_code, stdout = asyncio.run(
        execute_host_local_wave_request(
            request.canonical_stdin(), execute_wave=fake_execute
        )
    )
    response = RemoteHostWaveResponse.from_dict(json.loads(stdout))
    assert exit_code == 42
    assert response.status is RemoteWorkerStatus.FAILED
    assert response.dispatch_schedule_receipt_sha256 == schedule_receipt.sha256
    assert response.completed_assignment_sha256 == ()
    assert response.failed_assignment_sha256 == binding.assignment_sha256


def test_execute_remote_wave_uses_canonical_stdin_and_safe_receipt(
    tmp_path: Path,
) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a")
    private_stderr = b"sensitive-marker-do-not-retain"
    transport = _CapturingTransport(
        SshProcessResult(0, _remote_response(request), private_stderr)
    )
    try:
        result = asyncio.run(
            execute_remote_host_wave(route, request, transport=transport)
        )
    finally:
        agent.close()
    assert result.transport_outcome is RemoteTransportOutcome.REMOTE_SUCCEEDED
    assert result.completed_assignment_sha256 == request.binding.assignment_sha256
    assert result.dispatch_schedule_receipt_sha256 == _digest("schedule-host-a")
    assert RemoteHostWaveResult.from_dict(result.to_dict()) == result
    call = transport.calls[0]
    assert call["stdin"] == request.canonical_stdin()
    assert call["environment"] == {
        "LC_ALL": "C",
        "SSH_AUTH_SOCK": route.agent_socket_path,
    }
    receipt_body = canonical_json_bytes(result.to_dict())
    assert route.destination.encode() not in receipt_body
    assert route.known_hosts_path.encode() not in receipt_body
    assert route.agent_socket_path.encode() not in receipt_body
    assert private_stderr not in receipt_body
    assert result.stderr_sha256 == _digest_bytes(private_stderr)


def _digest_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def test_negative_remote_reason_is_hashed_not_copied(tmp_path: Path) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a")
    remote_reason = "remote_access_revoked"
    transport = _CapturingTransport(
        SshProcessResult(
            42,
            _remote_response(
                request,
                status="BLOCKED",
                reason_code=remote_reason,
            ),
            b"private diagnostic",
        )
    )
    try:
        result = asyncio.run(
            execute_remote_host_wave(route, request, transport=transport)
        )
    finally:
        agent.close()
    assert result.transport_outcome is RemoteTransportOutcome.REMOTE_BLOCKED
    assert result.reason_code == "remote_wave_blocked"
    assert result.remote_reason_sha256 == canonical_sha256(
        {"reason_code": remote_reason}
    )
    assert remote_reason.encode() not in canonical_json_bytes(result.to_dict())


@pytest.mark.parametrize(
    ("process", "outcome"),
    [
        (
            SshProcessResult(255, b"", b"ssh: private transport details"),
            RemoteTransportOutcome.SSH_FAILED,
        ),
        (
            SshProcessResult(0, b'{"not":"canonical"}\n', b"private failure"),
            RemoteTransportOutcome.INVALID_RESPONSE,
        ),
        (
            SshProcessResult(0, b"x" * 65, b""),
            RemoteTransportOutcome.OUTPUT_LIMIT_EXCEEDED,
        ),
    ],
)
def test_transport_failures_are_bounded_and_do_not_leak(
    tmp_path: Path,
    process: SshProcessResult,
    outcome: RemoteTransportOutcome,
) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a")
    transport = _CapturingTransport(process)
    try:
        result = asyncio.run(
            execute_remote_host_wave(
                route,
                request,
                transport=transport,
                stdout_limit_bytes=64,
                stderr_limit_bytes=64,
            )
        )
    finally:
        agent.close()
    assert result.transport_outcome is outcome
    assert result.completed_assignment_sha256 == ()
    assert result.dispatch_schedule_receipt_sha256 is None
    body = canonical_json_bytes(result.to_dict())
    assert route.destination.encode() not in body
    assert b"private transport details" not in body
    assert b"private failure" not in body


class _FleetTransport:
    async def run(self, **kwargs: object) -> SshProcessResult:
        request = RemoteHostWaveRequest.from_dict(json.loads(kwargs["stdin"]))
        if request.binding.host_id == "host-b":
            raise OSError("sensitive-marker host=node-b.example")
        return SshProcessResult(0, _remote_response(request), b"")


def test_fleet_node_failure_preserves_completed_host_and_is_round_trippable(
    tmp_path: Path,
) -> None:
    route_a, agent_a = _route(tmp_path, host_id="host-a")
    route_b, agent_b = _route(
        tmp_path,
        host_id="host-b",
        destination="runner@node-b.example",
    )
    request_a = _request("host-a", port=31_000)
    request_b = _request("host-b", port=31_000)
    try:
        receipt = asyncio.run(
            execute_fleet_wave(
                (route_b, route_a),
                (request_b, request_a),
                transport=_FleetTransport(),
            )
        )
    finally:
        agent_a.close()
        agent_b.close()
    assert receipt.outcome is FleetWaveOutcome.PARTIAL
    assert tuple(result.host_id for result in receipt.host_results) == (
        "host-a",
        "host-b",
    )
    assert receipt.host_results[0].succeeded
    assert receipt.host_results[0].dispatch_schedule_receipt_sha256 == _digest(
        "schedule-host-a"
    )
    assert (
        receipt.host_results[1].transport_outcome is RemoteTransportOutcome.SSH_FAILED
    )
    assert receipt.host_results[1].completed_assignment_sha256 == ()
    assert RemoteFleetWaveReceipt.from_dict(receipt.to_dict()) == receipt
    body = canonical_json_bytes(receipt.to_dict())
    assert b"node-a.example" not in body
    assert b"node-b.example" not in body
    assert b"sensitive-marker" not in body


def test_fleet_refuses_mixed_wave_authority(tmp_path: Path) -> None:
    route_a, agent_a = _route(tmp_path, host_id="host-a")
    route_b, agent_b = _route(
        tmp_path,
        host_id="host-b",
        destination="runner@node-b.example",
    )
    request_a = _request("host-a")
    request_b = replace(
        _request("host-b"),
        binding=replace(_binding("host-b"), wave_sha256=_digest("other-wave")),
    )
    try:
        with pytest.raises(ValueError, match="mix execution authorities"):
            asyncio.run(
                execute_fleet_wave(
                    (route_a, route_b),
                    (request_a, request_b),
                    transport=_FleetTransport(),
                )
            )
    finally:
        agent_a.close()
        agent_b.close()
