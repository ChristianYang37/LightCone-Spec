from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from lightcone_spec.orchestration.experiment_operator import ArchiveRequest
from lightcone_spec.orchestration.experiment_operator_production import (
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_remote_archive import (
    RemoteArchiveResult,
    SshRsyncArchiveEndpoint,
)
from lightcone_spec.orchestration.formal_rolling_archive import (
    ArchiveManifestFile,
    EvictedFileReceipt,
    EvictionFileBinding,
    FormalArchiveSha256Manifest,
    RemoteEvictionPlan,
    RemoteEvictionReceipt,
)
from lightcone_spec.orchestration.formal_rolling_archive_companion import (
    CompanionNodeIndex,
    FormalRollingArchiveCompanionError,
    RollingArchiveCompanion,
    RollingArchiveCompanionConfig,
    SshRollingArchiveTransport,
    load_companion_node_index,
)


class FakeTransport:
    def __init__(
        self,
        config: RollingArchiveCompanionConfig,
        *,
        available: set[str],
        fail_evict: bool = False,
        fail_restore: bool = False,
    ) -> None:
        self.config = config
        self.available = available
        self.fail_evict = fail_evict
        self.fail_restore = fail_restore
        self.probes: list[str] = []
        self.prepared: list[str] = []
        self.archived: list[str] = []
        self.evicted: list[str] = []
        self.restored: list[str] = []
        self.stops: list[str] = []

    def probe(self, *, node: str, ordinal: int) -> dict[str, Any]:
        self.probes.append(node)
        return {
            "run_id": self.config.run_id,
            "node": node,
            "ordinal": ordinal,
            "status": "AVAILABLE" if node in self.available else "ABSENT",
        }

    def prepare(self, *, node: str, ordinal: int) -> ArchiveRequest:
        self.prepared.append(node)
        remote = Path(self.config.remote_run_root) / "candidates" / node
        local = Path(self.config.local_results_root) / self.config.run_id / node
        return ArchiveRequest(
            archive_id=f"{self.config.run_id}.{node}.wave-{ordinal:02d}",
            safe_boundary=f"{node}:reduced",
            remote_payload_root=str(remote),
            local_partial_root=str(local / f"wave-{ordinal:02d}.partial"),
            local_final_root=str(local / f"wave-{ordinal:02d}.final"),
            remote_manifest_sha256=f"{ordinal + 1:064x}",
            predicted_payload_bytes=7,
        )

    def archive(
        self,
        *,
        node: str,
        ordinal: int,
        request: ArchiveRequest,
        result_path: Path,
        lock_path: Path,
    ) -> RemoteArchiveResult:
        del lock_path
        self.archived.append(node)
        result = RemoteArchiveResult(
            schema_version=1,
            kind="formal_remote_archive_result",
            archive_id=request.archive_id,
            ssh_target="root@example.test",
            ssh_port=22,
            remote_payload_root=request.remote_payload_root,
            local_final_root=request.local_final_root,
            manifest_sha256=request.remote_manifest_sha256,
            checked_file_count=1,
            checked_bytes=7,
            rehydrated_content_tree_sha256=f"{ordinal + 101:064x}",
            authorized_at_ns=ordinal + 1,
            remote_deletion_performed=False,
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(
            canonical_json_bytes({**asdict(result), "result_sha256": result.sha256})
        )
        return result

    def evict(
        self,
        *,
        node: str,
        ordinal: int,
        request: ArchiveRequest,
        result: RemoteArchiveResult,
    ) -> dict[str, Any]:
        del result
        self.evicted.append(node)
        if self.fail_evict:
            raise ValueError("injected eviction failure")
        candidate = Path(request.remote_payload_root)
        binding = EvictionFileBinding(
            absolute_path=str(candidate / "payload.bin"),
            archive_relative_path="payload.bin",
            device=1,
            inode=ordinal + 1,
            size_bytes=7,
            sha256=f"{ordinal + 201:064x}",
        )
        plan = RemoteEvictionPlan(
            schema_version=1,
            kind="formal_remote_eviction_plan",
            run_id=self.config.run_id,
            run_root=self.config.remote_run_root,
            node=node,
            archive_id=request.archive_id,
            archive_candidate_root=str(candidate),
            archive_candidate_root_device=1,
            archive_candidate_root_inode=ordinal + 1,
            archive_manifest_path=str(candidate / "sha256_manifest.json"),
            archive_manifest_sha256=request.remote_manifest_sha256,
            archive_request_sha256=f"{ordinal + 301:064x}",
            remote_archive_result_sha256=f"{ordinal + 302:064x}",
            archive_authorization_sha256=f"{ordinal + 303:064x}",
            retained_dependency_manifest_path=str(
                Path(self.config.remote_run_root)
                / "formal-dag-nodes"
                / f"{ordinal:02d}-{node}"
                / "reduction"
                / "retained-future-dependency-manifest.json"
            ),
            retained_dependency_manifest_sha256=f"{ordinal + 304:064x}",
            operator_checkpoint_sha256=f"{ordinal + 305:064x}",
            operator_snapshot_sha256=f"{ordinal + 306:064x}",
            files=(binding,),
            planned_bytes=7,
            created_at_ns=ordinal + 1,
            remote_eviction_authorized=True,
        )
        deleted = EvictedFileReceipt(
            absolute_path=binding.absolute_path,
            archive_relative_path=binding.archive_relative_path,
            size_bytes=binding.size_bytes,
            sha256=binding.sha256,
            deleted_at_ns=ordinal + 1,
        )
        receipt = RemoteEvictionReceipt(
            schema_version=1,
            kind="formal_remote_eviction_receipt",
            plan_sha256=plan.sha256,
            archive_id=plan.archive_id,
            archive_authorization_sha256=plan.archive_authorization_sha256,
            status="COMPLETE",
            deleted_files=(deleted,),
            deleted_bytes=7,
            failure_code=None,
            failure_path=None,
            missing_unrecorded_files=(),
            scheduler_stop_requested=False,
            scheduler_stop_succeeded=False,
            finished_at_ns=ordinal + 1,
        )
        control = (
            Path(self.config.remote_run_root)
            / "rolling-archive-control"
            / f"{ordinal:02d}-{node}"
        )
        return {
            "plan": {**plan.to_dict(), "plan_sha256": plan.sha256},
            "receipt": {
                **receipt.to_dict(),
                "receipt_sha256": receipt.sha256,
            },
            "plan_path": str(control / "remote-eviction-plan.json"),
            "receipt_path": str(control / "remote-eviction-receipt.json"),
        }

    def restore(self, index: object) -> dict[str, str]:
        self.restored.append(index.node)
        if self.fail_restore:
            raise FormalRollingArchiveCompanionError("injected restore failure")
        return {"receipt_sha256": f"{index.ordinal + 401:064x}"}

    def stop(self, reason: str) -> None:
        self.stops.append(reason)


def _config(tmp_path: Path) -> RollingArchiveCompanionConfig:
    results = (tmp_path / "local" / "results").resolve()
    state = (tmp_path / "state").resolve()
    results.mkdir(parents=True)
    state.mkdir()
    return RollingArchiveCompanionConfig(
        endpoint=SshRsyncArchiveEndpoint(
            ssh_target="root@example.test",
            ssh_port=22,
            remote_python="/srv/lightcone-v03/venv/bin/python",
            remote_operator_script="/srv/lightcone-v03/operator.py",
            remote_operator_database="/srv/lightcone-v03/run-v03/operator.sqlite3",
            remote_operator_lock="/srv/lightcone-v03/run-v03/operator.lock",
        ),
        remote_run_root="/srv/lightcone-v03/run-v03",
        local_results_root=str(results),
        state_root=str(state),
        lock_path=str((tmp_path / "companion.lock").resolve()),
        minimum_local_free_bytes=0,
    )


def test_run_once_archives_only_earliest_available_node_and_restarts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    transport = FakeTransport(config, available={"preflight", "e1"})
    companion = RollingArchiveCompanion(
        config,
        transport=transport,
        clock_ns=lambda: 100,
    )
    first = companion.run_once()
    assert first.action == "ARCHIVED"
    assert first.node == "preflight"
    assert transport.evicted == ["preflight"]
    index = load_companion_node_index(
        Path(config.state_root) / "00-preflight" / "rolling-archive-index.json"
    )
    assert index.node == "preflight"
    assert Path(index.local_archive_result_path).is_file()
    assert Path(index.local_plan_path).is_file()
    assert Path(index.local_eviction_receipt_path).is_file()

    restarted = RollingArchiveCompanion(
        config,
        transport=transport,
        clock_ns=lambda: 101,
    )
    second = restarted.run_once()
    assert second.action == "ARCHIVED"
    assert second.node == "e1"
    assert transport.evicted == ["preflight", "e1"]


def test_run_emits_only_state_changes_and_polls_every_thirty_seconds(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    transport = FakeTransport(
        config,
        available={"preflight"},
        fail_evict=True,
    )
    sleeps: list[float] = []
    companion = RollingArchiveCompanion(
        config,
        transport=transport,
        clock_ns=lambda: 200,
        sleeper=sleeps.append,
    )
    events = companion.run(max_cycles=3)
    assert len(events) == 1
    assert events[0].action == "FAILED"
    assert sleeps == [30.0, 30.0]
    assert transport.stops == [
        "rolling_archive_companion_failure",
        "rolling_archive_companion_failure",
        "rolling_archive_companion_failure",
    ]


def test_failure_requests_remote_scheduler_stop_before_retry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    transport = FakeTransport(
        config,
        available={"preflight"},
        fail_evict=True,
    )
    event = RollingArchiveCompanion(
        config,
        transport=transport,
        clock_ns=lambda: 300,
    ).run_once()
    assert event.action == "FAILED"
    assert transport.stops == ["rolling_archive_companion_failure"]
    assert not (
        Path(config.state_root) / "00-preflight" / "rolling-archive-index.json"
    ).exists()


def test_restore_all_obeys_explicit_forward_and_reverse_order(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    transport = FakeTransport(config, available={"preflight", "e3a"})
    companion = RollingArchiveCompanion(
        config,
        transport=transport,
        clock_ns=lambda: 400,
    )
    assert companion.run_once().node == "preflight"
    assert companion.run_once().node == "e3a"
    restored = companion.restore_all(order="reverse")
    assert [event.node for event in restored] == ["e3a", "preflight"]
    assert transport.restored == ["e3a", "preflight"]

    second_config = _config(tmp_path / "forward")
    second_transport = FakeTransport(
        second_config,
        available={"preflight", "e3a"},
    )
    second = RollingArchiveCompanion(
        second_config,
        transport=second_transport,
        clock_ns=lambda: 500,
    )
    second.run_once()
    second.run_once()
    second.restore_all(order="forward")
    assert second_transport.restored == ["preflight", "e3a"]


def test_restore_failure_requests_remote_scheduler_stop(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive_transport = FakeTransport(config, available={"preflight"})
    companion = RollingArchiveCompanion(config, transport=archive_transport)
    assert companion.run_once().action == "ARCHIVED"

    restore_transport = FakeTransport(
        config,
        available=set(),
        fail_restore=True,
    )
    events = RollingArchiveCompanion(
        config,
        transport=restore_transport,
    ).restore_all()
    assert [event.action for event in events] == ["FAILED"]
    assert restore_transport.stops == ["rolling_archive_companion_failure"]


def test_restore_rejects_tampered_index_path_before_transport(tmp_path: Path) -> None:
    config = _config(tmp_path)
    archive_transport = FakeTransport(config, available={"preflight"})
    companion = RollingArchiveCompanion(config, transport=archive_transport)
    assert companion.run_once().action == "ARCHIVED"
    index_path = Path(config.state_root) / "00-preflight" / "rolling-archive-index.json"
    value = json.loads(index_path.read_bytes())
    value["node"] = "../../escape"
    index_path.write_bytes(canonical_json_bytes(value))

    restore_transport = FakeTransport(config, available=set())
    events = RollingArchiveCompanion(
        config,
        transport=restore_transport,
    ).restore_all()
    assert [event.action for event in events] == ["FAILED"]
    assert restore_transport.restored == []
    assert restore_transport.stops == ["rolling_archive_companion_failure"]
    assert not (tmp_path / "escape").exists()


def test_default_ssh_restore_streams_only_deep_bound_plan_members(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    config = _config(tmp_path)
    final = (
        Path(config.local_results_root) / config.run_id / "preflight" / "wave-00.final"
    )
    final.mkdir(parents=True)
    payload = b"exact-remote-restore"
    payload_path = final / "payload.bin"
    payload_path.write_bytes(payload)
    payload_sha = hashlib.sha256(payload).hexdigest()
    manifest = FormalArchiveSha256Manifest(
        schema_version=1,
        kind="formal_archive_sha256_manifest",
        files=(ArchiveManifestFile("payload.bin", payload_sha, len(payload)),),
    )
    (final / "sha256_manifest.json").write_bytes(
        canonical_json_bytes(manifest.to_dict())
    )
    remote_root = Path(config.remote_run_root)
    candidate = remote_root / "candidate-preflight"
    result = RemoteArchiveResult(
        schema_version=1,
        kind="formal_remote_archive_result",
        archive_id="run-v03.preflight.wave-00",
        ssh_target="root@example.test",
        ssh_port=22,
        remote_payload_root=str(candidate),
        local_final_root=str(final),
        manifest_sha256=manifest.sha256,
        checked_file_count=1,
        checked_bytes=len(payload),
        rehydrated_content_tree_sha256="4" * 64,
        authorized_at_ns=1,
        remote_deletion_performed=False,
    )
    binding = EvictionFileBinding(
        absolute_path=str(candidate / "payload.bin"),
        archive_relative_path="payload.bin",
        device=1,
        inode=2,
        size_bytes=len(payload),
        sha256=payload_sha,
    )
    plan = RemoteEvictionPlan(
        schema_version=1,
        kind="formal_remote_eviction_plan",
        run_id=config.run_id,
        run_root=config.remote_run_root,
        node="preflight",
        archive_id=result.archive_id,
        archive_candidate_root=str(candidate),
        archive_candidate_root_device=1,
        archive_candidate_root_inode=1,
        archive_manifest_path=str(candidate / "sha256_manifest.json"),
        archive_manifest_sha256=manifest.sha256,
        archive_request_sha256="5" * 64,
        remote_archive_result_sha256=result.sha256,
        archive_authorization_sha256="6" * 64,
        retained_dependency_manifest_path=str(
            remote_root / "control" / "retained.json"
        ),
        retained_dependency_manifest_sha256="7" * 64,
        operator_checkpoint_sha256="8" * 64,
        operator_snapshot_sha256="9" * 64,
        files=(binding,),
        planned_bytes=len(payload),
        created_at_ns=1,
        remote_eviction_authorized=True,
    )
    deleted = EvictedFileReceipt(
        absolute_path=binding.absolute_path,
        archive_relative_path=binding.archive_relative_path,
        size_bytes=binding.size_bytes,
        sha256=binding.sha256,
        deleted_at_ns=1,
    )
    eviction = RemoteEvictionReceipt(
        schema_version=1,
        kind="formal_remote_eviction_receipt",
        plan_sha256=plan.sha256,
        archive_id=plan.archive_id,
        archive_authorization_sha256=plan.archive_authorization_sha256,
        status="COMPLETE",
        deleted_files=(deleted,),
        deleted_bytes=len(payload),
        failure_code=None,
        failure_path=None,
        missing_unrecorded_files=(),
        scheduler_stop_requested=False,
        scheduler_stop_succeeded=False,
        finished_at_ns=2,
    )
    state = Path(config.state_root) / "00-preflight"
    state.mkdir(parents=True)
    result_path = state / "remote-archive-result.json"
    plan_path = state / "remote-eviction-plan.json"
    eviction_path = state / "remote-eviction-receipt.json"
    result_path.write_bytes(
        canonical_json_bytes({**asdict(result), "result_sha256": result.sha256})
    )
    plan_path.write_bytes(
        canonical_json_bytes({**plan.to_dict(), "plan_sha256": plan.sha256})
    )
    eviction_path.write_bytes(
        canonical_json_bytes({**eviction.to_dict(), "receipt_sha256": eviction.sha256})
    )
    remote_control = remote_root / "rolling-archive-control" / "00-preflight"
    index = CompanionNodeIndex(
        schema_version=1,
        kind="formal_rolling_archive_companion_index",
        run_id=config.run_id,
        node="preflight",
        ordinal=0,
        request_sha256="a" * 64,
        local_archive_result_path=str(result_path),
        local_archive_result_sha256=result.sha256,
        local_plan_path=str(plan_path),
        plan_sha256=plan.sha256,
        local_eviction_receipt_path=str(eviction_path),
        eviction_receipt_sha256=eviction.sha256,
        remote_plan_path=str(remote_control / "remote-eviction-plan.json"),
        remote_eviction_receipt_path=str(
            remote_control / "remote-eviction-receipt.json"
        ),
        archived_at_ns=3,
    )
    transport = SshRollingArchiveTransport(config)
    streamed: list[tuple[str, tuple[str, ...], bytes]] = []

    def fake_stream(
        operation: str,
        arguments: tuple[str, ...],
        *,
        source: Path,
    ) -> dict[str, object]:
        streamed.append((operation, arguments, source.read_bytes()))
        return {"progress_sha256": "b" * 64}

    def fake_call(
        operation: str,
        arguments: tuple[str, ...],
        *,
        stdin_object: object = None,
    ) -> dict[str, object]:
        del arguments, stdin_object
        assert operation == "finalize-stream-restore"
        return {"receipt_sha256": "c" * 64}

    monkeypatch.setattr(transport, "_call_stream", fake_stream)
    monkeypatch.setattr(transport, "_call", fake_call)
    response = transport.restore(index)
    assert response["receipt_sha256"] == "c" * 64
    assert len(streamed) == 1
    operation, arguments, body = streamed[0]
    assert operation == "restore-member"
    assert body == payload
    assert arguments[arguments.index("--relative-path") + 1] == "payload.bin"
    assert not any("rm" in argument or "*" in argument for argument in arguments)

    tampered_deleted = replace(deleted, sha256="d" * 64)
    tampered_eviction = replace(eviction, deleted_files=(tampered_deleted,))
    eviction_path.write_bytes(
        canonical_json_bytes(
            {
                **tampered_eviction.to_dict(),
                "receipt_sha256": tampered_eviction.sha256,
            }
        )
    )
    tampered_index = replace(
        index,
        eviction_receipt_sha256=tampered_eviction.sha256,
    )
    with pytest.raises(
        FormalRollingArchiveCompanionError,
        match="authorities differ",
    ):
        transport.restore(tampered_index)
    assert len(streamed) == 1
