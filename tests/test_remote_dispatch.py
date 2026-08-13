from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightcone_spec.orchestration.remote_dispatch as remote_dispatch_module
from lightcone_spec.experiments.completion_authority import AssignmentTerminalBinding
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
    SshOutputLimitExceeded,
    SshProcessResult,
    SshTransportTimedOut,
    build_ssh_argv,
    canonical_json_bytes,
    canonical_sha256,
    decode_remote_host_wave_request,
    execute_fleet_wave,
    execute_host_local_wave_request,
    execute_remote_host_wave,
    reconcile_fleet_wave_receipt,
    reconcile_remote_host_wave,
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


def _binding(
    host_id: str,
    *,
    port: int = 31_000,
    host_wave_index: int | None = None,
) -> RemoteHostExecutionBinding:
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
        fleet_dispatch_plan_sha256=_digest("fleet-plan"),
        host_dispatch_plan_sha256=_digest(f"host-plan-{host_id}"),
        fleet_wave_index=3,
        fleet_wave_sha256=_digest("fleet-wave-3"),
        host_wave_index=(0 if host_id == "host-a" else 2)
        if host_wave_index is None
        else host_wave_index,
        host_wave_sha256=_digest(f"host-wave-{host_id}"),
        execution_bundle_manifest_path=(
            f"/srv/lightcone/{host_id}/dispatch-execution-bundle-manifest.json"
        ),
        execution_bundle_manifest_sha256=_digest(f"manifest-{host_id}"),
        receipt_output_path=f"/srv/lightcone/{host_id}/wave-3-receipt.json",
        resume_receipt_path=None,
        resume_receipt_sha256=None,
        resume_receipt_envelope_sha256=None,
        assignments=(assignment,),
    )


def _request(
    host_id: str,
    *,
    port: int = 31_000,
    route: SshHostRoute | None = None,
    ssh_route_authority_sha256: str | None = None,
) -> RemoteHostWaveRequest:
    return RemoteHostWaveRequest(
        schema_version=1,
        challenge_nonce_sha256=_digest(f"nonce-{host_id}"),
        ssh_route_authority_sha256=(
            (
                route.authority_sha256
                if route is not None
                else _digest(f"route-{host_id}")
            )
            if ssh_route_authority_sha256 is None
            else ssh_route_authority_sha256
        ),
        binding=_binding(host_id, port=port),
    )


def _remote_response(
    request: RemoteHostWaveRequest,
    *,
    status: str = "SUCCEEDED",
    reason_code: str | None = None,
) -> bytes:
    succeeded = status == "SUCCEEDED"
    failed = status == "FAILED"
    value = {
        "schema_version": 1,
        "kind": "lightcone_remote_host_wave_response",
        "host_id": request.binding.host_id,
        "request_sha256": request.sha256,
        "binding_sha256": request.binding.sha256,
        "status": status,
        "reason_code": reason_code,
        "dispatch_schedule_receipt_sha256": (
            _digest(f"schedule-{request.binding.host_id}")
            if succeeded or failed
            else None
        ),
        "dispatch_schedule_receipt_envelope_sha256": (
            _digest(f"schedule-envelope-{request.binding.host_id}")
            if succeeded or failed
            else None
        ),
        "completed_assignment_sha256": (
            list(request.binding.assignment_sha256) if succeeded else []
        ),
        "failed_assignment_sha256": (
            list(request.binding.assignment_sha256) if failed else []
        ),
    }
    return canonical_json_bytes(value) + b"\n"


def _failed_schedule_receipt(
    binding: RemoteHostExecutionBinding,
) -> DispatchScheduleReceipt:
    if binding.host_wave_index != 0:
        raise ValueError("test failed receipt helper requires host-local wave zero")
    assignment_receipt = AssignmentExecutionReceipt(
        plan_sha256=binding.host_dispatch_plan_sha256,
        wave_sha256=binding.host_wave_sha256,
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
        plan_sha256=binding.host_dispatch_plan_sha256,
        wave_index=0,
        wave_sha256=binding.host_wave_sha256,
        assignment_receipts=(assignment_receipt,),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((10, 20),),
        fixed_instance_actual_billed_gpu_ns=10,
        per_assignment_attributed_gpu_ns=10,
        per_assignment_attributed_fixed_instance_gpu_ns=10,
    )
    return DispatchScheduleReceipt(
        plan_sha256=binding.host_dispatch_plan_sha256,
        phase=DispatchExecutionPhase.FAILED,
        wave_receipts=(wave_receipt,),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((10, 20),),
        fixed_instance_actual_billed_gpu_ns=10,
        per_assignment_attributed_gpu_ns=10,
        per_assignment_attributed_fixed_instance_gpu_ns=10,
    )


def _successful_schedule_receipt(
    binding: RemoteHostExecutionBinding,
    *,
    tmp_path: Path,
    evidence_file_sha256s: tuple[str, ...],
) -> DispatchScheduleReceipt:
    if binding.host_wave_index != 0:
        raise ValueError("test success receipt helper requires host-local wave zero")
    evidence_paths = tuple(
        str((tmp_path / f"evidence-{index}.json").resolve())
        for index in range(len(evidence_file_sha256s))
    )
    budget_sha256 = _digest("success-budget")
    terminal = AssignmentTerminalBinding(
        authority_sha256=_digest("success-authority"),
        cell_id=_digest("success-cell"),
        assignment_sha256=binding.assignment_sha256[0],
        budget_sha256=budget_sha256,
        inventory_sha256=binding.host_inventory_sha256,
        physical_gpu_uuids=binding.assignments[0].gpu_uuids,
        execution_plan_sha256=_digest("success-execution-plan"),
        dispatch_plan_sha256=binding.host_dispatch_plan_sha256,
        run_id="remote-evidence-success",
        run_nonce_sha256=_digest("success-run-nonce"),
        terminal_receipt_path=str((tmp_path / "terminal.json").resolve()),
        terminal_receipt_sha256=_digest("success-terminal-receipt"),
        budget_observation_path=str((tmp_path / "budget.json").resolve()),
        budget_observation_sha256=_digest("success-budget-observation"),
        budget_observation_sidecar_path=str(
            (tmp_path / "budget.json.sha256").resolve()
        ),
        budget_observation_sidecar_sha256=_digest("success-budget-sidecar"),
        native_terminal_artifact_path=str((tmp_path / "native.json").resolve()),
        native_terminal_raw_sha256=_digest("success-native-raw"),
        native_terminal_sha256=_digest("success-native"),
        trusted_attester_policy_sha256=_digest("success-attester-policy"),
        evidence_file_paths=evidence_paths,
        evidence_file_sha256s=evidence_file_sha256s,
    )
    assignment_receipt = AssignmentExecutionReceipt(
        plan_sha256=binding.host_dispatch_plan_sha256,
        wave_sha256=binding.host_wave_sha256,
        assignment_sha256=binding.assignment_sha256[0],
        budget_sha256=budget_sha256,
        attempt=1,
        status=AssignmentExecutionStatus.SUCCEEDED,
        terminal_receipt_sha256=terminal.sha256,
        terminal_binding=terminal,
        failure_sha256=None,
        prior_attempt_receipt_sha256=None,
        gpu_count=1,
        fixed_instance_gpu_count=1,
        attempt_intervals_monotonic_ns=((10, 20),),
        attributed_gpu_ns=10,
        attributed_fixed_instance_gpu_ns=10,
    )
    wave_receipt = DispatchWaveExecutionReceipt(
        plan_sha256=binding.host_dispatch_plan_sha256,
        wave_index=0,
        wave_sha256=binding.host_wave_sha256,
        assignment_receipts=(assignment_receipt,),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((10, 20),),
        fixed_instance_actual_billed_gpu_ns=10,
        per_assignment_attributed_gpu_ns=10,
        per_assignment_attributed_fixed_instance_gpu_ns=10,
    )
    return DispatchScheduleReceipt(
        plan_sha256=binding.host_dispatch_plan_sha256,
        phase=DispatchExecutionPhase.COMPLETE,
        wave_receipts=(wave_receipt,),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((10, 20),),
        fixed_instance_actual_billed_gpu_ns=10,
        per_assignment_attributed_gpu_ns=10,
        per_assignment_attributed_fixed_instance_gpu_ns=10,
    )


def _raw_schedule_envelope(
    receipt: DispatchScheduleReceipt,
    *,
    journal_path: str = "/srv/lightcone/private/receipt.attempt-journal",
) -> bytes:
    value = {
        "schema_version": 2,
        "kind": "industrial_dispatch_schedule_receipt_envelope",
        "receipt": receipt.to_dict(),
        "sidecar": receipt.sidecar().to_dict(),
        "attempt_journal": {
            "journal_path": journal_path,
            "manifest_sha256": _digest("raw-journal-manifest"),
            "head_event_sha256": _digest("raw-journal-head"),
            "event_count": 3,
        },
    }
    return canonical_json_bytes(value) + b"\n"


def _raw_evidence_bundle(
    request: RemoteHostWaveRequest,
    unknown_sha256: str,
    receipt: DispatchScheduleReceipt,
    *,
    evidence_bodies: tuple[tuple[str, int, bytes], ...] = (),
) -> object:
    files = tuple(
        remote_dispatch_module._RemoteRawEvidenceFile(
            assignment_sha256=assignment_sha256,
            evidence_index=index,
            blob=remote_dispatch_module._RemoteRawBlob.from_body(body),
        )
        for assignment_sha256, index, body in evidence_bodies
    )
    return remote_dispatch_module._RemoteRawEvidenceBundle(
        schema_version=1,
        host_id=request.binding.host_id,
        source_request_sha256=request.sha256,
        source_binding_sha256=request.binding.sha256,
        unknown_result_sha256=unknown_sha256,
        schedule_envelope=remote_dispatch_module._RemoteRawBlob.from_body(
            _raw_schedule_envelope(receipt)
        ),
        evidence_files=files,
    )


class _CapturingTransport:
    def __init__(self, result: SshProcessResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def run(self, **kwargs: object) -> SshProcessResult:
        self.calls.append(kwargs)
        return self.result


class _SnapshotCapturingTransport(_CapturingTransport):
    def __init__(self, result: SshProcessResult) -> None:
        super().__init__(result)
        self.snapshot_path: Path | None = None
        self.snapshot_body: bytes | None = None
        self.snapshot_mode: int | None = None

    async def run(self, **kwargs: object) -> SshProcessResult:
        argv = kwargs["argv"]
        assert isinstance(argv, tuple)
        option = next(
            item
            for item in argv
            if isinstance(item, str) and item.startswith("UserKnownHostsFile=")
        )
        self.snapshot_path = Path(option.split("=", 1)[1])
        self.snapshot_body = self.snapshot_path.read_bytes()
        self.snapshot_mode = self.snapshot_path.stat().st_mode & 0o777
        return await super().run(**kwargs)


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


def test_route_authority_binds_endpoint_and_host_key_bytes_only(
    tmp_path: Path,
) -> None:
    route, agent = _route(tmp_path)
    second_agent = _AgentSocket("host-a-second")
    copied_known_hosts = (tmp_path / "copied.known_hosts").resolve()
    copied_known_hosts.write_bytes(Path(route.known_hosts_path).read_bytes())
    copied_known_hosts.chmod(0o600)
    copied_route = SshHostRoute(
        host_id=route.host_id,
        destination=route.destination,
        known_hosts_path=str(copied_known_hosts),
        agent_socket_path=str(second_agent.path),
        port=route.port,
        connect_timeout_seconds=91,
    )
    changed_key_path = (tmp_path / "changed.known_hosts").resolve()
    changed_key_path.write_text(
        "node-a.example ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIChanged\n",
        encoding="ascii",
    )
    changed_key_path.chmod(0o600)
    changed_key_route = replace(copied_route, known_hosts_path=str(changed_key_path))
    try:
        authority = route.authority_sha256
        assert copied_route.authority_sha256 == authority
        assert (
            replace(copied_route, destination="runner@node-b.example").authority_sha256
            != authority
        )
        assert replace(copied_route, port=22023).authority_sha256 != authority
        assert changed_key_route.authority_sha256 != authority
    finally:
        agent.close()
        second_agent.close()


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

    with pytest.raises(ValueError, match="path and identities must be complete"):
        replace(binding, resume_receipt_path="/srv/lightcone/resume.json")
    with pytest.raises(ValueError, match="path and identities must be complete"):
        replace(binding, resume_receipt_sha256=_digest("resume-only"))
    with pytest.raises(ValueError, match="path and identities must be complete"):
        replace(
            binding,
            resume_receipt_envelope_sha256=_digest("resume-envelope-only"),
        )


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
    assert response.status is RemoteWorkerStatus.BLOCKED
    assert response.reason_code == "remote_host_manifest_invalid"
    assert calls == []

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


def test_host_worker_private_kind_returns_only_bounded_raw_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("host-a")
    unknown_sha256 = _digest("unknown-host-attempt")
    raw = _raw_evidence_bundle(
        request,
        unknown_sha256,
        _failed_schedule_receipt(request.binding),
    )
    evidence_request = remote_dispatch_module._RemoteEvidenceRequest(
        schema_version=1,
        source_request=request,
        unknown_result_sha256=unknown_sha256,
        limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES,
    )
    monkeypatch.setattr(
        remote_dispatch_module,
        "_collect_remote_raw_evidence_bundle",
        lambda value: raw,
    )

    exit_code, stdout = asyncio.run(
        execute_host_local_wave_request(evidence_request.canonical_stdin())
    )

    assert exit_code == 0
    assert stdout == raw.canonical_payload(
        limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES
    )
    assert b"attempt_journal" in raw.schedule_envelope.body
    assert request.binding.receipt_output_path.encode() not in canonical_json_bytes(
        remote_dispatch_module._decode_remote_raw_evidence_bundle(
            stdout,
            limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES,
        ).to_dict()
    ).replace(raw.schedule_envelope.body_base64.encode(), b"")

    private_marker = "/srv/private/operator/evidence"

    def fail_collection(value: object) -> object:
        raise ValueError(private_marker)

    monkeypatch.setattr(
        remote_dispatch_module,
        "_collect_remote_raw_evidence_bundle",
        fail_collection,
    )
    exit_code, stdout = asyncio.run(
        execute_host_local_wave_request(evidence_request.canonical_stdin())
    )
    assert exit_code == 42
    assert private_marker.encode() not in stdout


def test_host_worker_preserves_failed_schedule_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = (tmp_path / "dispatch-execution-bundle-manifest.json").resolve()
    manifest_value = {"kind": "test-host-manifest", "schema_version": 1}
    manifest.write_bytes(canonical_json_bytes(manifest_value) + b"\n")
    request = _request("host-a")
    binding = replace(
        request.binding,
        host_wave_index=0,
        host_wave_sha256=_digest("host-wave-0"),
        execution_bundle_manifest_path=str(manifest),
        execution_bundle_manifest_sha256=canonical_sha256(manifest_value),
        receipt_output_path=str((tmp_path / "receipt.json").resolve()),
    )
    request = replace(request, binding=binding)
    schedule_receipt = _failed_schedule_receipt(binding)
    publication = object()
    verified_plans = (object(),)
    monkeypatch.setattr(
        remote_dispatch_module,
        "_load_verified_host_local_publication",
        lambda _binding: (publication, verified_plans),
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    receipt_envelope = _raw_schedule_envelope(schedule_receipt)

    async def fake_execute(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        Path(binding.receipt_output_path).write_bytes(receipt_envelope)
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
    assert response.dispatch_schedule_receipt_envelope_sha256 == canonical_sha256(
        json.loads(receipt_envelope)
    )
    assert response.completed_assignment_sha256 == ()
    assert response.failed_assignment_sha256 == binding.assignment_sha256
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (binding.execution_bundle_manifest_path,)
    assert kwargs["_verified_publication"] is publication
    assert kwargs["_verified_plans"] is verified_plans
    assert kwargs["expected_resume_receipt_envelope_sha256"] is None


@pytest.mark.parametrize("mode", ["executor_exception", "invalid_receipt"])
def test_host_worker_post_executor_failure_has_no_retry_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    request = _request("host-a")
    request = replace(
        request,
        binding=replace(
            request.binding,
            receipt_output_path=str((tmp_path / "receipt.json").resolve()),
        ),
    )
    monkeypatch.setattr(
        remote_dispatch_module,
        "_load_verified_host_local_publication",
        lambda _binding: (object(), (object(),)),
    )

    async def fake_execute(*args: object, **kwargs: object) -> object:
        if mode == "executor_exception":
            raise RuntimeError("private post-executor detail")
        receipt = _failed_schedule_receipt(request.binding)
        Path(request.binding.receipt_output_path).write_bytes(
            _raw_schedule_envelope(receipt)
        )
        return object()

    exit_code, stdout = asyncio.run(
        execute_host_local_wave_request(
            request.canonical_stdin(), execute_wave=fake_execute
        )
    )

    assert exit_code == 43
    assert stdout == b""


@pytest.mark.parametrize("identity_field", ["plan_sha256", "wave_sha256"])
def test_host_receipt_rejects_foreign_child_plan_or_wave(
    identity_field: str,
) -> None:
    binding = replace(
        _binding("host-a"),
        host_wave_index=0,
        host_wave_sha256=_digest("host-wave-0"),
    )
    request = replace(_request("host-a"), binding=binding)
    receipt = _failed_schedule_receipt(binding)
    wave = receipt.wave_receipts[0]
    child = replace(
        wave.assignment_receipts[0],
        **{identity_field: _digest(f"foreign-{identity_field}")},
    )
    foreign = replace(
        receipt,
        wave_receipts=(replace(wave, assignment_receipts=(child,)),),
    )

    with pytest.raises(ValueError, match="child receipt differs"):
        remote_dispatch_module._worker_response_from_receipt(
            request,
            foreign,
            receipt_envelope_sha256=_digest("foreign-child-envelope"),
        )


def test_host_receipt_rejects_waves_beyond_requested_target() -> None:
    binding = replace(
        _binding("host-a"),
        host_wave_index=0,
        host_wave_sha256=_digest("host-wave-0"),
    )
    request = replace(_request("host-a"), binding=binding)
    receipt = _failed_schedule_receipt(binding)
    later_assignment = replace(
        receipt.wave_receipts[0].assignment_receipts[0],
        wave_sha256=_digest("host-wave-1"),
        assignment_sha256=_digest("later-assignment"),
        failure_sha256=_digest("later-failure"),
        attempt_intervals_monotonic_ns=((30, 40),),
    )
    later_wave = DispatchWaveExecutionReceipt(
        plan_sha256=binding.host_dispatch_plan_sha256,
        wave_index=1,
        wave_sha256=_digest("host-wave-1"),
        assignment_receipts=(later_assignment,),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((30, 40),),
        fixed_instance_actual_billed_gpu_ns=10,
        per_assignment_attributed_gpu_ns=10,
        per_assignment_attributed_fixed_instance_gpu_ns=10,
    )
    overrun = DispatchScheduleReceipt(
        plan_sha256=binding.host_dispatch_plan_sha256,
        phase=DispatchExecutionPhase.FAILED,
        wave_receipts=(*receipt.wave_receipts, later_wave),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((10, 20), (30, 40)),
        fixed_instance_actual_billed_gpu_ns=20,
        per_assignment_attributed_gpu_ns=20,
        per_assignment_attributed_fixed_instance_gpu_ns=20,
    )

    with pytest.raises(ValueError, match="differs from request authority"):
        remote_dispatch_module._worker_response_from_receipt(
            request,
            overrun,
            receipt_envelope_sha256=_digest("overrun-envelope"),
        )


def test_host_receipt_accepts_exact_successful_prefix_before_target_retry(
    tmp_path: Path,
) -> None:
    binding = replace(
        _binding("host-a"),
        host_wave_index=1,
        host_wave_sha256=_digest("host-wave-1"),
    )
    request = replace(_request("host-a"), binding=binding)
    prefix_assignment = replace(
        binding.assignments[0],
        assignment_sha256=_digest("prefix-assignment"),
    )
    prefix_binding = replace(
        binding,
        host_wave_index=0,
        host_wave_sha256=_digest("host-wave-0"),
        assignments=(prefix_assignment,),
    )
    prefix = _successful_schedule_receipt(
        prefix_binding,
        tmp_path=tmp_path,
        evidence_file_sha256s=(_digest("prefix-evidence"),),
    ).wave_receipts[0]
    target_binding = replace(binding, host_wave_index=0)
    target = _failed_schedule_receipt(target_binding).wave_receipts[0]
    target_assignment = replace(
        target.assignment_receipts[0],
        attempt_intervals_monotonic_ns=((30, 40),),
    )
    target = replace(
        target,
        wave_index=1,
        assignment_receipts=(target_assignment,),
        active_intervals_monotonic_ns=((30, 40),),
    )
    receipt = DispatchScheduleReceipt(
        plan_sha256=binding.host_dispatch_plan_sha256,
        phase=DispatchExecutionPhase.FAILED,
        wave_receipts=(prefix, target),
        inventory_sha256=binding.host_inventory_sha256,
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=((10, 20), (30, 40)),
        fixed_instance_actual_billed_gpu_ns=20,
        per_assignment_attributed_gpu_ns=20,
        per_assignment_attributed_fixed_instance_gpu_ns=20,
    )

    envelope_sha256 = _digest("prefix-retry-envelope")
    response = remote_dispatch_module._worker_response_from_receipt(
        request,
        receipt,
        receipt_envelope_sha256=envelope_sha256,
    )

    assert response.status is RemoteWorkerStatus.FAILED
    assert response.failed_assignment_sha256 == binding.assignment_sha256
    assert response.dispatch_schedule_receipt_envelope_sha256 == envelope_sha256


def test_execute_remote_wave_uses_canonical_stdin_and_safe_receipt(
    tmp_path: Path,
) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a", route=route)
    private_stderr = b"sensitive-marker-do-not-retain"
    transport = _SnapshotCapturingTransport(
        SshProcessResult(0, _remote_response(request), private_stderr)
    )
    known_hosts_body = Path(route.known_hosts_path).read_bytes()
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
    assert result.ssh_route_authority_sha256 == route.authority_sha256
    call = transport.calls[0]
    assert call["stdin"] == request.canonical_stdin()
    assert call["environment"] == {
        "LC_ALL": "C",
        "SSH_AUTH_SOCK": route.agent_socket_path,
    }
    assert transport.snapshot_path is not None
    assert transport.snapshot_path != Path(route.known_hosts_path)
    assert transport.snapshot_body == known_hosts_body
    assert transport.snapshot_mode == 0o400
    assert not transport.snapshot_path.exists()
    argv = call["argv"]
    assert isinstance(argv, tuple)
    assert f"UserKnownHostsFile={transport.snapshot_path}" in argv
    assert f"UserKnownHostsFile={route.known_hosts_path}" not in argv
    request_body = request.canonical_stdin()
    receipt_body = canonical_json_bytes(result.to_dict())
    for body in (request_body, receipt_body):
        assert route.destination.encode() not in body
        assert route.known_hosts_path.encode() not in body
        assert route.agent_socket_path.encode() not in body
        assert known_hosts_body.rstrip(b"\n") not in body
        assert request.ssh_route_authority_sha256.encode() in body
    assert private_stderr not in receipt_body
    assert result.stderr_sha256 == _digest_bytes(private_stderr)


def test_route_authority_mismatch_refuses_before_transport(tmp_path: Path) -> None:
    route, agent = _route(tmp_path)
    request = _request(
        "host-a",
        ssh_route_authority_sha256=_digest("foreign-route-authority"),
    )
    transport = _CapturingTransport(SshProcessResult(0, _remote_response(request), b""))
    try:
        result = asyncio.run(
            execute_remote_host_wave(route, request, transport=transport)
        )
    finally:
        agent.close()
    assert transport.calls == []
    assert result.transport_outcome is RemoteTransportOutcome.SSH_FAILED
    assert result.ssh_route_authority_sha256 == request.ssh_route_authority_sha256


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_negative_remote_reason_is_hashed_not_copied(tmp_path: Path) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a", route=route)
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
    "process",
    [
        SshProcessResult(255, b"", b"ssh: private transport details"),
        SshProcessResult(0, b'{"not":"canonical"}\n', b"private failure"),
        SshProcessResult(0, b"x" * 65, b""),
    ],
)
def test_non_authoritative_process_results_stay_unknown_and_do_not_leak(
    tmp_path: Path,
    process: SshProcessResult,
) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a", route=route)
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
    assert result.transport_outcome is RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN
    assert result.outcome_unknown
    assert not result.retryable
    assert result.completed_assignment_sha256 == ()
    assert result.dispatch_schedule_receipt_sha256 is None
    body = canonical_json_bytes(result.to_dict())
    assert route.destination.encode() not in body
    assert b"private transport details" not in body
    assert b"private failure" not in body


class _RaisingTransport:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.called = False

    async def run(self, **kwargs: object) -> SshProcessResult:
        self.called = True
        raise self.error


class _NonExactTransport:
    async def run(self, **kwargs: object) -> object:
        return object()


@pytest.mark.parametrize(
    "error",
    [
        SshTransportTimedOut("ambiguous timeout"),
        SshOutputLimitExceeded("ambiguous output truncation"),
        OSError("ambiguous transport failure"),
        RuntimeError("ambiguous process failure"),
        TypeError("ambiguous transport contract failure"),
        ValueError("ambiguous transport value failure"),
    ],
)
def test_post_dispatch_transport_exceptions_stay_unknown(
    tmp_path: Path,
    error: BaseException,
) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a", route=route)
    transport = _RaisingTransport(error)
    try:
        result = asyncio.run(
            execute_remote_host_wave(route, request, transport=transport)
        )
    finally:
        agent.close()
    assert transport.called
    assert result.transport_outcome is RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN
    assert result.outcome_unknown
    assert not result.retryable


def test_post_dispatch_non_exact_process_stays_unknown(tmp_path: Path) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a", route=route)
    try:
        result = asyncio.run(
            execute_remote_host_wave(
                route,
                request,
                transport=_NonExactTransport(),  # type: ignore[arg-type]
            )
        )
    finally:
        agent.close()
    assert result.transport_outcome is RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN
    assert result.outcome_unknown
    assert not result.retryable


def test_pre_dispatch_local_validation_failure_is_retryable_ssh_failure(
    tmp_path: Path,
) -> None:
    route, agent = _route(tmp_path)
    request = _request("host-a", route=route)
    transport = _CapturingTransport(SshProcessResult(0, _remote_response(request), b""))
    Path(route.known_hosts_path).unlink()
    try:
        result = asyncio.run(
            execute_remote_host_wave(route, request, transport=transport)
        )
    finally:
        agent.close()
    assert transport.calls == []
    assert result.transport_outcome is RemoteTransportOutcome.SSH_FAILED
    assert result.retryable
    assert not result.outcome_unknown


class _FleetTransport:
    def __init__(self, failed_hosts: tuple[str, ...] = ("host-b",)) -> None:
        self.failed_hosts = frozenset(failed_hosts)
        self.hosts: list[str] = []

    async def run(self, **kwargs: object) -> SshProcessResult:
        request = RemoteHostWaveRequest.from_dict(json.loads(kwargs["stdin"]))
        self.hosts.append(request.binding.host_id)
        if request.binding.host_id in self.failed_hosts:
            return SshProcessResult(
                42,
                _remote_response(
                    request,
                    status="FAILED",
                    reason_code="remote_host_wave_failed",
                ),
                b"sensitive-marker host=node-b.example",
            )
        return SshProcessResult(0, _remote_response(request), b"")


class _TimeoutTransport:
    async def run(self, **kwargs: object) -> SshProcessResult:
        raise SshTransportTimedOut("remote outcome is ambiguous")


def _install_raw_evidence_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes | None,
) -> list[object]:
    calls: list[object] = []

    async def run(_transport: object, **kwargs: object) -> SshProcessResult:
        value = json.loads(kwargs["stdin"])
        calls.append(value)
        assert value["kind"] == "lightcone_remote_raw_evidence_request"
        if payload is None:
            raise OSError("raw evidence is temporarily unavailable")
        return SshProcessResult(0, payload, b"")

    monkeypatch.setattr(remote_dispatch_module.AsyncioSshTransport, "run", run)
    return calls


class _BoundedFleetTransport:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def run(self, **kwargs: object) -> SshProcessResult:
        request = RemoteHostWaveRequest.from_dict(json.loads(kwargs["stdin"]))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            return SshProcessResult(0, _remote_response(request), b"")
        finally:
            self.active -= 1


def test_fleet_node_failure_preserves_completed_host_and_is_round_trippable(
    tmp_path: Path,
) -> None:
    route_a, agent_a = _route(tmp_path, host_id="host-a")
    route_b, agent_b = _route(
        tmp_path,
        host_id="host-b",
        destination="runner@node-b.example",
    )
    request_a = _request("host-a", port=31_000, route=route_a)
    request_b = _request("host-b", port=31_000, route=route_b)
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
        receipt.host_results[0].host_dispatch_plan_sha256
        != receipt.host_results[1].host_dispatch_plan_sha256
    )
    assert (
        receipt.host_results[0].host_wave_index
        != receipt.host_results[1].host_wave_index
    )
    assert (
        receipt.host_results[0].host_wave_sha256
        != receipt.host_results[1].host_wave_sha256
    )
    assert (
        receipt.host_results[1].transport_outcome
        is RemoteTransportOutcome.REMOTE_FAILED
    )
    assert receipt.host_results[1].completed_assignment_sha256 == ()
    assert RemoteFleetWaveReceipt.from_dict(receipt.to_dict()) == receipt
    body = canonical_json_bytes(receipt.to_dict())
    assert b"node-a.example" not in body
    assert b"node-b.example" not in body
    assert b"sensitive-marker" not in body


def test_partial_retry_preserves_success_and_contacts_only_failed_host(
    tmp_path: Path,
) -> None:
    route_a, agent_a = _route(tmp_path, host_id="host-a")
    route_b, agent_b = _route(
        tmp_path,
        host_id="host-b",
        destination="runner@node-b.example",
    )
    request_a = _request("host-a", port=31_000, route=route_a)
    request_b = _request("host-b", port=31_000, route=route_b)
    initial_transport = _FleetTransport()
    retry_transport = _FleetTransport(failed_hosts=())
    try:
        initial = asyncio.run(
            execute_fleet_wave(
                (route_a, route_b),
                (request_a, request_b),
                transport=initial_transport,
            )
        )
        failed_result = next(
            result for result in initial.host_results if result.host_id == "host-b"
        )
        assert failed_result.dispatch_schedule_receipt_sha256 is not None
        assert failed_result.dispatch_schedule_receipt_envelope_sha256 is not None
        retry_binding = replace(
            request_b.binding,
            receipt_output_path="/srv/lightcone/host-b/wave-3-retry-1.json",
            resume_receipt_path=request_b.binding.receipt_output_path,
            resume_receipt_sha256=failed_result.dispatch_schedule_receipt_sha256,
            resume_receipt_envelope_sha256=(
                failed_result.dispatch_schedule_receipt_envelope_sha256
            ),
        )
        retry_request = RemoteHostWaveRequest(
            schema_version=1,
            challenge_nonce_sha256=_digest("nonce-host-b-retry-1"),
            ssh_route_authority_sha256=request_b.ssh_route_authority_sha256,
            binding=retry_binding,
            prior_fleet_wave_receipt_sha256=initial.sha256,
        )
        wrong_envelope_request = replace(
            retry_request,
            binding=replace(
                retry_request.binding,
                resume_receipt_envelope_sha256=_digest("wrong-failed-envelope"),
            ),
        )
        wrong_transport = _FleetTransport(failed_hosts=())
        with pytest.raises(ValueError, match="resume envelope differs"):
            asyncio.run(
                execute_fleet_wave(
                    (route_b,),
                    (wrong_envelope_request,),
                    transport=wrong_transport,
                    prior_fleet_wave_receipt=initial,
                )
            )
        assert wrong_transport.hosts == []
        retried = asyncio.run(
            execute_fleet_wave(
                (route_b,),
                (retry_request,),
                transport=retry_transport,
                prior_fleet_wave_receipt=initial,
            )
        )
    finally:
        agent_a.close()
        agent_b.close()

    assert initial.outcome is FleetWaveOutcome.PARTIAL
    assert retried.outcome is FleetWaveOutcome.COMPLETE
    assert retried.prior_fleet_wave_receipt_sha256 == initial.sha256
    assert retried.host_results[0] is initial.host_results[0]
    assert retried.host_results[1].succeeded
    assert initial_transport.hosts == ["host-a", "host-b"]
    assert retry_transport.hosts == ["host-b"]
    assert RemoteFleetWaveReceipt.from_dict(retried.to_dict()) == retried


def test_timeout_stays_unknown_until_exact_remote_evidence_reconciles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, agent = _route(tmp_path, host_id="host-a")
    request = _request("host-a", route=route)
    try:
        unknown_receipt = asyncio.run(
            execute_fleet_wave(
                (route,),
                (request,),
                transport=_TimeoutTransport(),
            )
        )
        unknown = unknown_receipt.host_results[0]
        assert (
            unknown.transport_outcome is RemoteTransportOutcome.REMOTE_OUTCOME_UNKNOWN
        )
        assert unknown_receipt.outcome is FleetWaveOutcome.UNKNOWN
        with pytest.raises(ValueError, match="exact prior/evidence authority"):
            replace(
                unknown,
                reconciliation_evidence_sha256=_digest("partial-reconciliation"),
            )
        retry_without_evidence = RemoteHostWaveRequest(
            schema_version=1,
            challenge_nonce_sha256=_digest("unknown-direct-retry"),
            ssh_route_authority_sha256=request.ssh_route_authority_sha256,
            binding=replace(
                request.binding,
                receipt_output_path="/srv/lightcone/host-a/unknown-retry.json",
            ),
            prior_fleet_wave_receipt_sha256=unknown_receipt.sha256,
        )
        with pytest.raises(ValueError, match="requires exact evidence reconciliation"):
            asyncio.run(
                execute_fleet_wave(
                    (route,),
                    (retry_without_evidence,),
                    transport=_FleetTransport(failed_hosts=()),
                    prior_fleet_wave_receipt=unknown_receipt,
                )
            )

        schedule_receipt = _failed_schedule_receipt(request.binding)
        raw = _raw_evidence_bundle(
            request,
            unknown.sha256,
            schedule_receipt,
        )
        payload = raw.canonical_payload(
            limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES
        )
        evidence_calls = _install_raw_evidence_response(monkeypatch, payload)
        reconciled = asyncio.run(
            reconcile_remote_host_wave(
                route,
                request,
                unknown,
            )
        )
        repeated = asyncio.run(
            reconcile_remote_host_wave(
                route,
                request,
                unknown,
            )
        )
        assert repeated == reconciled
        assert reconciled.transport_outcome is RemoteTransportOutcome.RECONCILED_FAILED
        assert reconciled.reconciles_unknown_result_sha256 == unknown.sha256
        assert reconciled.reconciliation_evidence_sha256 == raw.sha256
        durable = canonical_json_bytes(reconciled.to_dict())
        assert request.binding.receipt_output_path.encode() not in durable
        assert b"attempt_journal" not in durable
        assert payload not in durable
        reconciled_receipt = reconcile_fleet_wave_receipt(
            unknown_receipt,
            (reconciled,),
        )
        assert reconciled_receipt.outcome is FleetWaveOutcome.FAILED
        assert reconciled_receipt.prior_fleet_wave_receipt_sha256 == (
            unknown_receipt.sha256
        )

        retry_after_evidence = RemoteHostWaveRequest(
            schema_version=1,
            challenge_nonce_sha256=_digest("reconciled-retry"),
            ssh_route_authority_sha256=request.ssh_route_authority_sha256,
            binding=replace(
                request.binding,
                receipt_output_path="/srv/lightcone/host-a/reconciled-retry.json",
                resume_receipt_path=request.binding.receipt_output_path,
                resume_receipt_sha256=(reconciled.dispatch_schedule_receipt_sha256),
                resume_receipt_envelope_sha256=(
                    reconciled.dispatch_schedule_receipt_envelope_sha256
                ),
            ),
            prior_fleet_wave_receipt_sha256=reconciled_receipt.sha256,
        )
        wrong_resume = replace(
            retry_after_evidence,
            binding=replace(
                retry_after_evidence.binding,
                resume_receipt_sha256=_digest("wrong-resume-receipt"),
            ),
        )
        wrong_transport = _FleetTransport(failed_hosts=())
        with pytest.raises(ValueError, match="resume content differs"):
            asyncio.run(
                execute_fleet_wave(
                    (route,),
                    (wrong_resume,),
                    transport=wrong_transport,
                    prior_fleet_wave_receipt=reconciled_receipt,
                )
            )
        assert wrong_transport.hosts == []
        wrong_envelope = replace(
            retry_after_evidence,
            binding=replace(
                retry_after_evidence.binding,
                resume_receipt_envelope_sha256=_digest("wrong-reconciled-envelope"),
            ),
        )
        with pytest.raises(ValueError, match="resume envelope differs"):
            asyncio.run(
                execute_fleet_wave(
                    (route,),
                    (wrong_envelope,),
                    transport=wrong_transport,
                    prior_fleet_wave_receipt=reconciled_receipt,
                )
            )
        assert wrong_transport.hosts == []
        retried = asyncio.run(
            execute_fleet_wave(
                (route,),
                (retry_after_evidence,),
                transport=_FleetTransport(failed_hosts=()),
                prior_fleet_wave_receipt=reconciled_receipt,
            )
        )
    finally:
        agent.close()

    assert retried.outcome is FleetWaveOutcome.COMPLETE
    assert len(evidence_calls) == 2


def test_retry_unknown_reconciliation_reuses_resume_bound_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import gpu_pool as gpu_pool_module
    from lightcone_spec.orchestration import (
        execution_bundle as execution_bundle_module,
    )

    original_receipt_path = "/srv/lightcone/host-a/original.json"
    retry_receipt_path = "/srv/lightcone/host-a/retry.json"
    resume_journal_path = "/srv/lightcone/private/receipt.attempt-journal"
    base_binding = replace(
        _binding("host-a", host_wave_index=0),
        host_wave_sha256=_digest("host-wave-0"),
        receipt_output_path=retry_receipt_path,
    )
    receipt = _failed_schedule_receipt(base_binding)
    envelope = _raw_schedule_envelope(receipt)
    binding = replace(
        base_binding,
        resume_receipt_path=original_receipt_path,
        resume_receipt_sha256=receipt.sha256,
        resume_receipt_envelope_sha256=canonical_sha256(json.loads(envelope)),
    )
    request = replace(
        _request("host-a"),
        binding=binding,
        prior_fleet_wave_receipt_sha256=_digest("reconciled-prior-wave"),
    )
    evidence_request = remote_dispatch_module._RemoteEvidenceRequest(
        schema_version=1,
        source_request=request,
        unknown_result_sha256=_digest("retry-remote-outcome-unknown"),
        limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES,
    )
    decoded_receipt, journal_binding = (
        remote_dispatch_module._decode_raw_schedule_envelope(envelope)
    )
    assert decoded_receipt == receipt
    assert journal_binding.journal_path == resume_journal_path

    @dataclass(frozen=True)
    class _Context:
        inventory: object
        resume_terminal_authorities: tuple[object, ...] = ()

    assignment_sha256 = binding.assignment_sha256[0]
    wave = SimpleNamespace(
        wave_index=0,
        sha256=binding.host_wave_sha256,
        assignments=(
            SimpleNamespace(
                assignment_id=assignment_sha256,
                gpu_uuids=binding.assignments[0].gpu_uuids,
            ),
        ),
    )
    dispatch_plan = SimpleNamespace(
        sha256=binding.host_dispatch_plan_sha256,
        waves=(wave,),
    )
    context = _Context(inventory=SimpleNamespace(sha256=binding.host_inventory_sha256))
    plan = SimpleNamespace(
        runtime_plan=SimpleNamespace(
            physical_assignment=SimpleNamespace(assignment_sha256=assignment_sha256)
        ),
        dispatch_plan=dispatch_plan,
        dispatch_context=context,
    )
    publication = SimpleNamespace(
        bundles=(SimpleNamespace(assignment_sha256=assignment_sha256),),
        manifest=SimpleNamespace(sha256=binding.execution_bundle_manifest_sha256),
    )
    monkeypatch.setattr(
        remote_dispatch_module,
        "_load_verified_host_local_publication",
        lambda _binding: (publication, (plan,)),
    )
    evidence_reads: list[str] = []

    def read_evidence(path: str, **_kwargs: object) -> bytes:
        evidence_reads.append(path)
        assert path in {retry_receipt_path, original_receipt_path}
        return envelope

    monkeypatch.setattr(
        remote_dispatch_module,
        "_read_stable_evidence_leaf",
        read_evidence,
    )

    class _Snapshot:
        binding = journal_binding
        receipt = decoded_receipt
        replay_authority = object()
        terminal_bindings: tuple[object, ...] = ()

        @staticmethod
        def require_complete_cost_authority() -> None:
            return None

    replay_counts: list[int | None] = []

    class _Journal:
        @staticmethod
        def replay(*, event_count: int | None = None) -> _Snapshot:
            replay_counts.append(event_count)
            return _Snapshot()

    opened: list[tuple[object, dict[str, object]]] = []

    def open_existing(root: object, **kwargs: object) -> _Journal:
        opened.append((root, kwargs))
        return _Journal()

    monkeypatch.setattr(
        execution_bundle_module.DispatchAttemptJournal,
        "open_existing",
        staticmethod(open_existing),
    )
    monkeypatch.setattr(
        gpu_pool_module,
        "validate_dispatch_resume",
        lambda *_args, **_kwargs: None,
    )

    raw = remote_dispatch_module._collect_remote_raw_evidence_bundle(evidence_request)

    assert raw.schedule_envelope.body == envelope
    assert evidence_reads == [retry_receipt_path, original_receipt_path]
    assert len(opened) == 1
    opened_path, opened_kwargs = opened[0]
    assert opened_path == resume_journal_path
    assert opened_path != f"{retry_receipt_path}.attempt-journal"
    assert opened_kwargs["expected_prefix"] == journal_binding
    assert replay_counts == [None, journal_binding.event_count]

    swapped_envelope = _raw_schedule_envelope(
        receipt,
        journal_path="/srv/lightcone/private/swapped.attempt-journal",
    )
    swapped_receipt, swapped_journal = (
        remote_dispatch_module._decode_raw_schedule_envelope(swapped_envelope)
    )
    assert swapped_receipt == decoded_receipt
    assert swapped_journal != journal_binding

    def read_swapped_resume(path: str, **_kwargs: object) -> bytes:
        return swapped_envelope if path == original_receipt_path else envelope

    monkeypatch.setattr(
        remote_dispatch_module,
        "_read_stable_evidence_leaf",
        read_swapped_resume,
    )
    with pytest.raises(ValueError, match="resume envelope differs"):
        remote_dispatch_module._collect_remote_raw_evidence_bundle(evidence_request)
    assert len(opened) == 1


def test_reconciliation_rejects_foreign_content_addressed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, agent = _route(tmp_path, host_id="host-a")
    request = _request("host-a", route=route)
    try:
        unknown = asyncio.run(
            execute_remote_host_wave(
                route,
                request,
                transport=_TimeoutTransport(),
            )
        )
        receipt = _failed_schedule_receipt(request.binding)
        raw = _raw_evidence_bundle(
            request,
            unknown.sha256,
            receipt,
        )
        foreign = replace(raw, host_id="host-b")
        with pytest.raises(ValueError, match="differs from UNKNOWN authority"):
            remote_dispatch_module._projection_from_raw_bundle(
                foreign,
                request=request,
                unknown_result_sha256=unknown.sha256,
            )
        evidence_calls = _install_raw_evidence_response(monkeypatch, None)
        with pytest.raises(ValueError, match="route authority differs"):
            asyncio.run(
                reconcile_remote_host_wave(
                    replace(route, destination="runner@node-b.example"),
                    request,
                    unknown,
                )
            )
        assert evidence_calls == []
        unavailable = asyncio.run(
            reconcile_remote_host_wave(
                route,
                request,
                unknown,
            )
        )
    finally:
        agent.close()
    assert unavailable is unknown
    assert len(evidence_calls) == 1


def test_legacy_summary_payload_cannot_authorize_unknown_remote_success() -> None:
    summary = (
        canonical_json_bytes(
            {
                "schema_version": 1,
                "kind": "lightcone_remote_host_evidence_fetch",
            }
        )
        + b"\n"
    )
    with pytest.raises(ValueError, match="raw-evidence bundle fields"):
        remote_dispatch_module._decode_remote_raw_evidence_bundle(
            summary,
            limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES,
        )


def test_success_reconciliation_requires_exact_raw_evidence_coverage(
    tmp_path: Path,
) -> None:
    request = replace(
        _request("host-a"),
        binding=replace(
            _binding("host-a"),
            host_wave_index=0,
            host_wave_sha256=_digest("host-wave-0"),
        ),
    )
    bodies = (b'{"terminal":"one"}\n', b"PAR1-test-evidence")
    receipt = _successful_schedule_receipt(
        request.binding,
        tmp_path=tmp_path,
        evidence_file_sha256s=tuple(
            hashlib.sha256(body).hexdigest() for body in bodies
        ),
    )
    unknown_sha256 = _digest("unknown-success-attempt")
    raw = _raw_evidence_bundle(
        request,
        unknown_sha256,
        receipt,
        evidence_bodies=tuple(
            (request.binding.assignment_sha256[0], index, body)
            for index, body in enumerate(bodies)
        ),
    )
    payload = raw.canonical_payload(
        limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES
    )
    decoded = remote_dispatch_module._decode_remote_raw_evidence_bundle(
        payload,
        limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES,
    )
    projection = remote_dispatch_module._projection_from_raw_bundle(
        decoded,
        request=request,
        unknown_result_sha256=unknown_sha256,
    )
    assert projection.succeeded
    assert projection.completed_assignment_sha256 == (request.binding.assignment_sha256)
    assert projection.failed_assignment_sha256 == ()

    variants = (
        replace(raw, evidence_files=raw.evidence_files[:-1]),
        replace(
            raw,
            evidence_files=raw.evidence_files
            + (
                remote_dispatch_module._RemoteRawEvidenceFile(
                    assignment_sha256=request.binding.assignment_sha256[0],
                    evidence_index=len(bodies),
                    blob=remote_dispatch_module._RemoteRawBlob.from_body(b"extra"),
                ),
            ),
        ),
        replace(
            raw,
            evidence_files=(
                replace(
                    raw.evidence_files[0],
                    blob=remote_dispatch_module._RemoteRawBlob.from_body(b"tampered"),
                ),
                raw.evidence_files[1],
            ),
        ),
    )
    for variant in variants:
        with pytest.raises(ValueError, match="evidence (file coverage|content)"):
            remote_dispatch_module._projection_from_raw_bundle(
                variant,
                request=request,
                unknown_result_sha256=unknown_sha256,
            )


def test_raw_reconciliation_rejects_receipt_tamper_and_protocol_bounds() -> None:
    request = _request("host-a")
    unknown_sha256 = _digest("unknown-bounded-attempt")
    receipt = _failed_schedule_receipt(request.binding)
    raw = _raw_evidence_bundle(request, unknown_sha256, receipt)
    envelope = json.loads(raw.schedule_envelope.body)
    envelope["sidecar"]["artifact_sha256"] = _digest("forged-sidecar")
    tampered = replace(
        raw,
        schedule_envelope=remote_dispatch_module._RemoteRawBlob.from_body(
            canonical_json_bytes(envelope) + b"\n"
        ),
    )
    with pytest.raises(ValueError, match="sidecar identity"):
        remote_dispatch_module._projection_from_raw_bundle(
            tampered,
            request=request,
            unknown_result_sha256=unknown_sha256,
        )
    with pytest.raises(ValueError, match="file exceeds"):
        remote_dispatch_module._RemoteRawEvidenceFile(
            assignment_sha256=request.binding.assignment_sha256[0],
            evidence_index=0,
            blob=remote_dispatch_module._RemoteRawBlob.from_body(
                b"x" * (remote_dispatch_module.MAX_RECONCILE_FILE_BYTES + 1)
            ),
        )
    too_many = tuple(
        remote_dispatch_module._RemoteRawEvidenceFile(
            assignment_sha256=request.binding.assignment_sha256[0],
            evidence_index=index,
            blob=remote_dispatch_module._RemoteRawBlob.from_body(b"x"),
        )
        for index in range(remote_dispatch_module.MAX_RECONCILE_FILE_COUNT + 1)
    )
    with pytest.raises(ValueError, match="file count"):
        replace(raw, evidence_files=too_many)
    payload = raw.canonical_payload(
        limit_bytes=remote_dispatch_module.MAX_RECONCILE_EVIDENCE_BYTES
    )
    with pytest.raises(ValueError, match="byte limit"):
        remote_dispatch_module._decode_remote_raw_evidence_bundle(
            payload,
            limit_bytes=128,
        )


def test_fleet_execution_bounds_concurrent_host_transports(tmp_path: Path) -> None:
    routes: list[SshHostRoute] = []
    agents: list[_AgentSocket] = []
    requests: list[RemoteHostWaveRequest] = []
    for index in range(5):
        host_id = f"host-{index}"
        route, agent = _route(
            tmp_path,
            host_id=host_id,
            destination=f"runner@node-{index}.example",
        )
        routes.append(route)
        agents.append(agent)
        requests.append(_request(host_id, route=route))
    transport = _BoundedFleetTransport()
    try:
        receipt = asyncio.run(
            execute_fleet_wave(
                routes,
                requests,
                transport=transport,
                max_concurrency=2,
            )
        )
        with pytest.raises(ValueError, match="max_concurrency"):
            asyncio.run(
                execute_fleet_wave(
                    routes,
                    requests,
                    transport=transport,
                    max_concurrency=0,
                )
            )
    finally:
        for agent in agents:
            agent.close()
    assert receipt.outcome is FleetWaveOutcome.COMPLETE
    assert transport.peak == 2


def test_retry_rejects_successful_host_wrong_route_and_assignment_migration(
    tmp_path: Path,
) -> None:
    route_a, agent_a = _route(tmp_path, host_id="host-a")
    route_b, agent_b = _route(
        tmp_path,
        host_id="host-b",
        destination="runner@node-b.example",
    )
    request_a = _request("host-a", port=31_000, route=route_a)
    request_b = _request("host-b", port=31_000, route=route_b)
    try:
        initial = asyncio.run(
            execute_fleet_wave(
                (route_a, route_b),
                (request_a, request_b),
                transport=_FleetTransport(),
            )
        )
        retry_a = RemoteHostWaveRequest(
            schema_version=1,
            challenge_nonce_sha256=_digest("nonce-host-a-illegal-retry"),
            ssh_route_authority_sha256=request_a.ssh_route_authority_sha256,
            binding=replace(
                request_a.binding,
                receipt_output_path="/srv/lightcone/host-a/illegal-retry.json",
            ),
            prior_fleet_wave_receipt_sha256=initial.sha256,
        )
        with pytest.raises(ValueError, match="cannot re-execute a successful host"):
            asyncio.run(
                execute_fleet_wave(
                    (route_a,),
                    (retry_a,),
                    transport=_FleetTransport(failed_hosts=()),
                    prior_fleet_wave_receipt=initial,
                )
            )

        retry_b = RemoteHostWaveRequest(
            schema_version=1,
            challenge_nonce_sha256=_digest("nonce-host-b-illegal-migration"),
            ssh_route_authority_sha256=request_b.ssh_route_authority_sha256,
            binding=replace(
                request_b.binding,
                receipt_output_path="/srv/lightcone/host-b/illegal-migration.json",
                assignments=(
                    replace(
                        request_b.binding.assignments[0],
                        assignment_sha256=(
                            request_a.binding.assignments[0].assignment_sha256
                        ),
                    ),
                ),
            ),
            prior_fleet_wave_receipt_sha256=initial.sha256,
        )
        with pytest.raises(ValueError, match="host-local execution identity"):
            asyncio.run(
                execute_fleet_wave(
                    (route_b,),
                    (retry_b,),
                    transport=_FleetTransport(failed_hosts=()),
                    prior_fleet_wave_receipt=initial,
                )
            )
        with pytest.raises(ValueError, match="routes do not exactly cover"):
            asyncio.run(
                execute_fleet_wave(
                    (route_a,),
                    (retry_b,),
                    transport=_FleetTransport(failed_hosts=()),
                    prior_fleet_wave_receipt=initial,
                )
            )
    finally:
        agent_a.close()
        agent_b.close()


def test_fleet_refuses_mixed_wave_authority(tmp_path: Path) -> None:
    route_a, agent_a = _route(tmp_path, host_id="host-a")
    route_b, agent_b = _route(
        tmp_path,
        host_id="host-b",
        destination="runner@node-b.example",
    )
    request_a = _request("host-a", route=route_a)
    request_b = replace(
        _request("host-b", route=route_b),
        binding=replace(
            _binding("host-b"), fleet_wave_sha256=_digest("other-fleet-wave")
        ),
    )
    try:
        with pytest.raises(ValueError, match="mix fleet execution authorities"):
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
