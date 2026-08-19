from __future__ import annotations

import hashlib
import io
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from lightcone_spec.orchestration.experiment_operator import (
    ArchiveRequest,
    ArchiveStepReceipt,
    ControllerArtifactBinding,
    RemoteEvictionAuthorization,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_remote_archive import RemoteArchiveResult
from lightcone_spec.orchestration.formal_rolling_archive import (
    ArchiveManifestFile,
    FormalArchiveSha256Manifest,
    FormalRollingArchiveError,
    RemoteEvictionPlan,
    SimulatedEvictionCrash,
    build_archive_request,
    build_remote_eviction_plan,
    evaluate_remote_eviction,
    execute_remote_eviction_plan,
    finalize_remote_stream_restore,
    load_archive_restore_receipt,
    load_formal_archive_sha256_manifest,
    load_remote_eviction_receipt,
    publish_formal_archive_sha256_manifest,
    publish_remote_eviction_plan,
    restore_evicted_files,
    restore_remote_member_from_stream,
)
from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    DriverFileBinding,
    RetainedFutureDependencyManifest,
)


@dataclass(frozen=True)
class ArchiveCase:
    root: Path
    candidate: Path
    retained_file: Path
    retained_tree_file: Path
    retained_manifest_path: Path
    manifest_path: Path
    request: ArchiveRequest
    result: RemoteArchiveResult
    authorization: RemoteEvictionAuthorization
    checkpoint: dict[str, object]
    snapshot: dict[str, object]
    result_path: Path


def _write(path: Path, body: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path.resolve()


def _semantic_sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _publish_retained_manifest(
    *,
    root: Path,
    candidate: Path,
    retained_file: Path,
    retained_tree: Path,
) -> Path:
    control = root / "control"
    completion_path = _write(control / "completion.json", b'{"complete":true}\n')
    decision_path = _write(control / "decision.json", b'{"allow":true}\n')
    retained = RetainedFutureDependencyManifest(
        schema_version=1,
        kind="formal_single_operator_retained_future_dependency_manifest",
        run_id=root.name,
        run_root=str(root),
        node="e1",
        completion=ControllerArtifactBinding.bind(completion_path),
        decision=ControllerArtifactBinding.bind(decision_path),
        retained_files=tuple(
            sorted(
                (
                    DriverFileBinding.bind(completion_path),
                    DriverFileBinding.bind(decision_path),
                    DriverFileBinding.bind(retained_file),
                ),
                key=lambda row: row.absolute_path,
            )
        ),
        retained_transitive_roots=(str(retained_tree),),
        archive_candidate_roots=(str(candidate),),
        archive_safe_after_reduction=True,
        remote_eviction_authorized_for_nonretained_files=True,
        remote_eviction_scope=(
            "archive_candidate_roots_excluding_retained_files_and_transitive_roots"
        ),
        eviction_preconditions=(
            "local_sha_manifest_verified",
            "local_rehydrate_test_passed",
        ),
        transitive_evidence_must_rehydrate_at_original_paths=True,
    )
    path = control / "retained-e1.json"
    path.write_bytes(canonical_json_bytes(retained.to_dict()))
    return path.resolve()


def _base_tree(
    tmp_path: Path,
    *,
    version: str = "v03",
    candidate_component: str = "work",
) -> tuple[Path, Path, Path, Path, Path]:
    root = (tmp_path / version / "run-v03").resolve()
    candidate = root / "formal-dag-nodes" / "e1" / "execution" / candidate_component
    retained_file = _write(candidate / "keep.json", b'{"keep":true}\n')
    retained_tree = candidate / "retained-auxiliary"
    retained_tree_file = _write(retained_tree / "proof.bin", b"transitive-proof")
    _write(candidate / "delete-a.bin", b"A" * 97)
    _write(candidate / "nested" / "delete-b.txt", b"delete-me")
    retained_manifest = _publish_retained_manifest(
        root=root,
        candidate=candidate,
        retained_file=retained_file,
        retained_tree=retained_tree,
    )
    return (
        root,
        candidate.resolve(),
        retained_file,
        retained_tree_file,
        retained_manifest,
    )


def _authorized_case(
    tmp_path: Path,
    *,
    add_hardlink: bool = False,
) -> ArchiveCase:
    root, candidate, retained_file, retained_tree_file, retained_path = _base_tree(
        tmp_path
    )
    if add_hardlink:
        os.link(candidate / "delete-a.bin", candidate / "delete-hardlink.bin")
    published = publish_formal_archive_sha256_manifest(
        run_root=root,
        candidate_root=candidate,
        retained_dependency_manifest_path=retained_path,
        lock_path=root / "control" / "manifest.lock",
    )
    manifest_path = Path(published.path)
    manifest = load_formal_archive_sha256_manifest(manifest_path)
    results = (tmp_path / "local" / "results").resolve()
    results.mkdir(parents=True)
    request = build_archive_request(
        manifest_path=manifest_path,
        retained_dependency_manifest_path=retained_path,
        local_results_root=results,
        wave="wave-0001",
    )
    final = Path(request.local_final_root)
    final.parent.mkdir(parents=True)
    shutil.copytree(candidate, final)
    content_tree_sha = _semantic_sha(
        {
            "manifest_sha256": manifest.sha256,
            "files": [row.to_dict() for row in manifest.files],
        }
    )
    transfer = ArchiveStepReceipt(
        step="TRANSFER",
        manifest_sha256=manifest.sha256,
        evidence_sha256="1" * 64,
        checked_file_count=len(manifest.files),
        checked_bytes=manifest.payload_bytes,
    )
    local = ArchiveStepReceipt(
        step="LOCAL_SHA_VERIFY",
        manifest_sha256=manifest.sha256,
        evidence_sha256="2" * 64,
        checked_file_count=len(manifest.files),
        checked_bytes=manifest.payload_bytes,
    )
    rehydrate = ArchiveStepReceipt(
        step="REHYDRATE_VERIFY",
        manifest_sha256=manifest.sha256,
        evidence_sha256="3" * 64,
        checked_file_count=len(manifest.files),
        checked_bytes=manifest.payload_bytes,
        content_tree_sha256=content_tree_sha,
    )
    authorization = RemoteEvictionAuthorization(
        archive_id=request.archive_id,
        remote_payload_root=request.remote_payload_root,
        manifest_sha256=manifest.sha256,
        local_final_root=request.local_final_root,
        local_sha_evidence_sha256=local.evidence_sha256,
        rehydrate_evidence_sha256=rehydrate.evidence_sha256,
        rehydrated_content_tree_sha256=content_tree_sha,
        authorized_at_ns=50,
    )
    result = RemoteArchiveResult(
        schema_version=1,
        kind="formal_remote_archive_result",
        archive_id=request.archive_id,
        ssh_target="root@example.test",
        ssh_port=22,
        remote_payload_root=request.remote_payload_root,
        local_final_root=request.local_final_root,
        manifest_sha256=manifest.sha256,
        checked_file_count=len(manifest.files),
        checked_bytes=manifest.payload_bytes,
        rehydrated_content_tree_sha256=content_tree_sha,
        authorized_at_ns=50,
        remote_deletion_performed=False,
    )
    checkpoint: dict[str, object] = {
        "archive_id": request.archive_id,
        "safe_boundary": request.safe_boundary,
        "cell_id": request.cell_id,
        "attempt": request.attempt,
        "remote_payload_root": request.remote_payload_root,
        "local_partial_root": request.local_partial_root,
        "local_final_root": request.local_final_root,
        "remote_manifest_sha256": request.remote_manifest_sha256,
        "predicted_payload_bytes": request.predicted_payload_bytes,
        "state": "EVICTION_AUTHORIZED",
        "transfer_receipt": asdict(transfer),
        "local_sha_receipt": asdict(local),
        "rehydrate_receipt": asdict(rehydrate),
        "eviction_authorized_at_ns": authorization.authorized_at_ns,
    }
    snapshot: dict[str, object] = {
        "run_id": root.name,
        "dispatch_state": "STOP",
        "dispatch_stop_reason": "rolling_archive_boundary",
        "attempts": [],
        "archives": [checkpoint],
    }
    result_path = results / root.name / "archive-result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(
        canonical_json_bytes({**asdict(result), "result_sha256": result.sha256})
    )
    return ArchiveCase(
        root=root,
        candidate=candidate,
        retained_file=retained_file,
        retained_tree_file=retained_tree_file,
        retained_manifest_path=retained_path,
        manifest_path=manifest_path,
        request=request,
        result=result,
        authorization=authorization,
        checkpoint=checkpoint,
        snapshot=snapshot,
        result_path=result_path,
    )


def _plan(case: ArchiveCase, *, created_at_ns: int = 100) -> RemoteEvictionPlan:
    return build_remote_eviction_plan(
        request=case.request,
        remote_archive_result=case.result,
        authorization=case.authorization,
        retained_dependency_manifest_path=case.retained_manifest_path,
        operator_checkpoint=case.checkpoint,
        operator_snapshot=case.snapshot,
        active_writer_probe=lambda _path: False,
        process_probe=lambda _pid, _pgid: False,
        created_at_ns=created_at_ns,
    )


def _publish_plan(case: ArchiveCase, plan: RemoteEvictionPlan) -> Path:
    path = case.root / "control" / "eviction-plan.json"
    publish_remote_eviction_plan(
        path,
        plan,
        lock_path=case.root / "control" / "plan.lock",
    )
    return path


def test_manifest_and_request_are_exact_canonical_and_resumable(
    tmp_path: Path,
) -> None:
    root, candidate, _retained, _tree_file, retained_path = _base_tree(tmp_path)
    published = publish_formal_archive_sha256_manifest(
        run_root=root,
        candidate_root=candidate,
        retained_dependency_manifest_path=retained_path,
        lock_path=root / "control" / "manifest.lock",
    )
    manifest = load_formal_archive_sha256_manifest(published.path)
    assert [row.path for row in manifest.files] == sorted(
        row.path for row in manifest.files
    )
    assert all(row.path != "sha256_manifest.json" for row in manifest.files)
    assert published.predicted_payload_bytes == sum(
        path.stat().st_size
        for path in candidate.rglob("*")
        if path.is_file() and path.name != "sha256_manifest.json"
    )
    assert (
        publish_formal_archive_sha256_manifest(
            run_root=root,
            candidate_root=candidate,
            retained_dependency_manifest_path=retained_path,
            lock_path=root / "control" / "manifest.lock",
        )
        == published
    )
    results = (tmp_path / "local" / "results").resolve()
    request = build_archive_request(
        manifest_path=published.path,
        retained_dependency_manifest_path=retained_path,
        local_results_root=results,
        wave="wave-0009",
    )
    assert request.predicted_payload_bytes == published.predicted_payload_bytes
    assert Path(request.local_partial_root) == (
        results / root.name / "e1" / "wave-0009.partial"
    )
    assert Path(request.local_final_root) == (
        results / root.name / "e1" / "wave-0009.final"
    )


@pytest.mark.parametrize("member_kind", ["symlink", "fifo"])
def test_manifest_rejects_links_and_special_files(
    tmp_path: Path,
    member_kind: str,
) -> None:
    root, candidate, _retained, _tree_file, retained_path = _base_tree(tmp_path)
    unsafe = candidate / f"unsafe-{member_kind}"
    if member_kind == "symlink":
        unsafe.symlink_to(candidate / "delete-a.bin")
    else:
        os.mkfifo(unsafe)
    with pytest.raises(FormalRollingArchiveError):
        publish_formal_archive_sha256_manifest(
            run_root=root,
            candidate_root=candidate,
            retained_dependency_manifest_path=retained_path,
            lock_path=root / "control" / "manifest.lock",
        )


@pytest.mark.parametrize(
    ("version", "candidate_component"),
    [("v02", "work"), ("v03", "cache")],
)
def test_manifest_rejects_v02_and_cache_scopes(
    tmp_path: Path,
    version: str,
    candidate_component: str,
) -> None:
    root, candidate, _retained, _tree_file, retained_path = _base_tree(
        tmp_path,
        version=version,
        candidate_component=candidate_component,
    )
    with pytest.raises(FormalRollingArchiveError):
        publish_formal_archive_sha256_manifest(
            run_root=root,
            candidate_root=candidate,
            retained_dependency_manifest_path=retained_path,
            lock_path=root / "control" / "manifest.lock",
        )


def test_manifest_schema_rejects_duplicate_and_self_entries() -> None:
    row = ArchiveManifestFile("a.bin", "a" * 64, 1)
    with pytest.raises(ValueError):
        FormalArchiveSha256Manifest(
            schema_version=1,
            kind="formal_archive_sha256_manifest",
            files=(row, row),
        )
    with pytest.raises(ValueError):
        ArchiveManifestFile("sha256_manifest.json", "a" * 64, 1)


def test_eviction_gate_stays_false_without_full_durable_authority(
    tmp_path: Path,
) -> None:
    case = _authorized_case(tmp_path)
    base = {
        "request": case.request,
        "remote_archive_result": case.result,
        "authorization": case.authorization,
        "retained_dependency_manifest_path": case.retained_manifest_path,
        "operator_checkpoint": case.checkpoint,
        "operator_snapshot": case.snapshot,
        "active_writer_probe": lambda _path: False,
        "process_probe": lambda _pid, _pgid: False,
        "created_at_ns": 100,
    }
    assert not evaluate_remote_eviction(
        **{**base, "authorization": None}
    ).remote_eviction_authorized
    no_rehydrate = {
        **case.checkpoint,
        "state": "LOCAL_SHA_VERIFIED",
        "rehydrate_receipt": None,
    }
    assert not evaluate_remote_eviction(
        **{**base, "operator_checkpoint": no_rehydrate}
    ).remote_eviction_authorized
    foreign = {**case.snapshot, "run_id": "foreign-v03"}
    assert not evaluate_remote_eviction(
        **{**base, "operator_snapshot": foreign}
    ).remote_eviction_authorized
    running = {
        **case.snapshot,
        "attempts": [{"status": "RUNNING", "pid": None, "pgid": None}],
    }
    assert not evaluate_remote_eviction(
        **{**base, "operator_snapshot": running}
    ).remote_eviction_authorized


def test_plan_excludes_exact_and_transitive_retained_files(tmp_path: Path) -> None:
    case = _authorized_case(tmp_path)
    plan = _plan(case)
    paths = {Path(row.absolute_path) for row in plan.files}
    assert case.retained_file not in paths
    assert case.retained_tree_file not in paths
    assert case.candidate / "delete-a.bin" in paths
    assert case.candidate / "nested" / "delete-b.txt" in paths
    assert plan.planned_bytes == sum(path.stat().st_size for path in paths)
    assert plan.remote_eviction_authorized is True


def test_plan_rejects_hardlink_writer_and_live_pid(tmp_path: Path) -> None:
    case = _authorized_case(tmp_path, add_hardlink=True)
    with pytest.raises(FormalRollingArchiveError):
        _plan(case)

    fresh = _authorized_case(tmp_path / "writer")
    writer_gate = evaluate_remote_eviction(
        request=fresh.request,
        remote_archive_result=fresh.result,
        authorization=fresh.authorization,
        retained_dependency_manifest_path=fresh.retained_manifest_path,
        operator_checkpoint=fresh.checkpoint,
        operator_snapshot=fresh.snapshot,
        active_writer_probe=lambda path: path.name == "delete-a.bin",
        process_probe=lambda _pid, _pgid: False,
        created_at_ns=100,
    )
    assert not writer_gate.remote_eviction_authorized
    live_snapshot = {
        **fresh.snapshot,
        "attempts": [{"status": "COMPLETE", "pid": 123, "pgid": 123}],
    }
    live_gate = evaluate_remote_eviction(
        request=fresh.request,
        remote_archive_result=fresh.result,
        authorization=fresh.authorization,
        retained_dependency_manifest_path=fresh.retained_manifest_path,
        operator_checkpoint=fresh.checkpoint,
        operator_snapshot=live_snapshot,
        active_writer_probe=lambda _path: False,
        process_probe=lambda _pid, _pgid: True,
        created_at_ns=100,
    )
    assert not live_gate.remote_eviction_authorized


def test_executor_stops_with_zero_progress_on_path_mutation(tmp_path: Path) -> None:
    case = _authorized_case(tmp_path)
    plan = _plan(case)
    plan_path = _publish_plan(case, plan)
    first = Path(plan.files[0].absolute_path)
    first.write_bytes(b"mutated")
    stops: list[str] = []
    receipt_path = case.root / "control" / "eviction-receipt.json"
    receipt = execute_remote_eviction_plan(
        plan_path=plan_path,
        receipt_path=receipt_path,
        lock_path=case.root / "control" / "execute.lock",
        operator_snapshot=case.snapshot,
        scheduler_stop=stops.append,
        active_writer_probe=lambda _path: False,
        process_probe=lambda _pid, _pgid: False,
        clock_ns=lambda: 200,
    )
    assert receipt.status == "FAILED_ZERO"
    assert receipt.deleted_files == ()
    assert first.exists()
    assert stops == ["remote_eviction_identity_or_progress_mismatch"]
    assert load_remote_eviction_receipt(receipt_path) == receipt


def test_executor_crash_window_fails_partial_on_restart(tmp_path: Path) -> None:
    case = _authorized_case(tmp_path)
    plan = _plan(case)
    plan_path = _publish_plan(case, plan)
    receipt_path = case.root / "control" / "eviction-receipt.json"
    crashed = False

    def crash_once(_binding: object) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise SimulatedEvictionCrash

    with pytest.raises(SimulatedEvictionCrash):
        execute_remote_eviction_plan(
            plan_path=plan_path,
            receipt_path=receipt_path,
            lock_path=case.root / "control" / "execute.lock",
            operator_snapshot=case.snapshot,
            scheduler_stop=lambda _reason: None,
            active_writer_probe=lambda _path: False,
            process_probe=lambda _pid, _pgid: False,
            after_unlink=crash_once,
            clock_ns=lambda: 300,
        )
    stops: list[str] = []
    receipt = execute_remote_eviction_plan(
        plan_path=plan_path,
        receipt_path=receipt_path,
        lock_path=case.root / "control" / "execute.lock",
        operator_snapshot=case.snapshot,
        scheduler_stop=stops.append,
        active_writer_probe=lambda _path: False,
        process_probe=lambda _pid, _pgid: False,
        clock_ns=lambda: 301,
    )
    assert receipt.status == "FAILED_PARTIAL"
    assert receipt.missing_unrecorded_files
    assert stops


def test_complete_eviction_and_original_path_no_replace_restore(
    tmp_path: Path,
) -> None:
    case = _authorized_case(tmp_path)
    plan = _plan(case)
    original = {
        row.absolute_path: Path(row.absolute_path).read_bytes() for row in plan.files
    }
    plan_path = _publish_plan(case, plan)
    receipt_path = case.root / "control" / "eviction-receipt.json"
    receipt = execute_remote_eviction_plan(
        plan_path=plan_path,
        receipt_path=receipt_path,
        lock_path=case.root / "control" / "execute.lock",
        operator_snapshot=case.snapshot,
        scheduler_stop=lambda _reason: pytest.fail("unexpected scheduler STOP"),
        active_writer_probe=lambda _path: False,
        process_probe=lambda _pid, _pgid: False,
        clock_ns=lambda: 400,
    )
    assert receipt.status == "COMPLETE"
    assert receipt.deleted_bytes == plan.planned_bytes
    assert all(not Path(row.absolute_path).exists() for row in plan.files)
    assert case.retained_file.exists()
    assert case.retained_tree_file.exists()
    assert case.manifest_path.exists()
    # Recreate one exact member to prove restoration never overwrites it.
    existing = Path(plan.files[0].absolute_path)
    existing.write_bytes(original[str(existing)])
    restore_path = case.root / "control" / "restore-receipt.json"
    restored = restore_evicted_files(
        plan_path=plan_path,
        remote_archive_result_path=case.result_path,
        receipt_path=restore_path,
        lock_path=case.root / "control" / "restore.lock",
        clock_ns=lambda: 500,
    )
    assert len(restored.already_present_files) == 1
    assert restored.existing_files_overwritten is False
    assert {
        row.absolute_path
        for row in (*restored.restored_files, *restored.already_present_files)
    } == set(original)
    assert all(Path(path).read_bytes() == body for path, body in original.items())
    assert load_archive_restore_receipt(restore_path) == restored


def test_restore_refuses_to_overwrite_existing_different_file(tmp_path: Path) -> None:
    case = _authorized_case(tmp_path)
    plan = _plan(case)
    plan_path = _publish_plan(case, plan)
    receipt = execute_remote_eviction_plan(
        plan_path=plan_path,
        receipt_path=case.root / "control" / "eviction-receipt.json",
        lock_path=case.root / "control" / "execute.lock",
        operator_snapshot=case.snapshot,
        scheduler_stop=lambda _reason: None,
        active_writer_probe=lambda _path: False,
        process_probe=lambda _pid, _pgid: False,
        clock_ns=lambda: 600,
    )
    assert receipt.status == "COMPLETE"
    occupied = Path(plan.files[0].absolute_path)
    occupied.write_bytes(b"foreign")
    with pytest.raises(FormalRollingArchiveError):
        restore_evicted_files(
            plan_path=plan_path,
            remote_archive_result_path=case.result_path,
            receipt_path=case.root / "control" / "restore-receipt.json",
            lock_path=case.root / "control" / "restore.lock",
            clock_ns=lambda: 700,
        )
    assert occupied.read_bytes() == b"foreign"


def test_plan_and_receipts_must_be_outside_candidate(tmp_path: Path) -> None:
    case = _authorized_case(tmp_path)
    plan = _plan(case)
    with pytest.raises(ValueError):
        publish_remote_eviction_plan(
            case.candidate / "unsafe-plan.json",
            plan,
            lock_path=case.root / "control" / "plan.lock",
        )


def test_authorization_result_mutation_is_rejected(tmp_path: Path) -> None:
    case = _authorized_case(tmp_path)
    changed = replace(case.result, checked_bytes=case.result.checked_bytes + 1)
    gate = evaluate_remote_eviction(
        request=case.request,
        remote_archive_result=changed,
        authorization=case.authorization,
        retained_dependency_manifest_path=case.retained_manifest_path,
        operator_checkpoint=case.checkpoint,
        operator_snapshot=case.snapshot,
        active_writer_probe=lambda _path: False,
        process_probe=lambda _pid, _pgid: False,
        created_at_ns=100,
    )
    assert not gate.remote_eviction_authorized


def test_stream_restore_is_exact_restartable_and_finalizable(
    tmp_path: Path,
) -> None:
    case = _authorized_case(tmp_path)
    plan = _plan(case)
    plan_path = _publish_plan(case, plan)
    source_bytes = {
        row.archive_relative_path: (
            Path(case.request.local_final_root) / row.archive_relative_path
        ).read_bytes()
        for row in plan.files
    }
    eviction_path = case.root / "control" / "eviction-receipt.json"
    eviction = execute_remote_eviction_plan(
        plan_path=plan_path,
        receipt_path=eviction_path,
        lock_path=case.root / "control" / "execute.lock",
        operator_snapshot=case.snapshot,
        scheduler_stop=lambda _reason: None,
        active_writer_probe=lambda _path: False,
        process_probe=lambda _pid, _pgid: False,
        clock_ns=lambda: 800,
    )
    assert eviction.status == "COMPLETE"
    progress_root = case.root / "control" / "stream-progress"
    first = plan.files[0]
    first_high_water = sum(row.size_bytes for row in plan.files)
    crashed = False

    def crash_after_link(_binding: object) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise SimulatedEvictionCrash

    with pytest.raises(SimulatedEvictionCrash):
        restore_remote_member_from_stream(
            plan_path=plan_path,
            eviction_receipt_path=eviction_path,
            remote_archive_result_path=case.result_path,
            archive_relative_path=first.archive_relative_path,
            progress_output_path=progress_root / "00000000.json",
            lock_path=case.root / "control" / "stream.lock",
            stream=io.BytesIO(source_bytes[first.archive_relative_path]),
            minimum_free_bytes=0,
            free_bytes_probe=lambda _path: first_high_water,
            after_target_link=crash_after_link,
            clock_ns=lambda: 801,
        )
    assert (
        Path(first.absolute_path).read_bytes()
        == source_bytes[first.archive_relative_path]
    )
    recovered = restore_remote_member_from_stream(
        plan_path=plan_path,
        eviction_receipt_path=eviction_path,
        remote_archive_result_path=case.result_path,
        archive_relative_path=first.archive_relative_path,
        progress_output_path=progress_root / "00000000.json",
        lock_path=case.root / "control" / "stream.lock",
        stream=io.BytesIO(source_bytes[first.archive_relative_path]),
        minimum_free_bytes=0,
        free_bytes_probe=lambda _path: first_high_water - first.size_bytes,
        clock_ns=lambda: 802,
    )
    assert recovered.disposition == "ALREADY_PRESENT"
    for index, binding in enumerate(plan.files[1:], start=1):
        remaining_high_water = sum(row.size_bytes for row in plan.files[index:])
        progress = restore_remote_member_from_stream(
            plan_path=plan_path,
            eviction_receipt_path=eviction_path,
            remote_archive_result_path=case.result_path,
            archive_relative_path=binding.archive_relative_path,
            progress_output_path=progress_root / f"{index:08d}.json",
            lock_path=case.root / "control" / "stream.lock",
            stream=io.BytesIO(source_bytes[binding.archive_relative_path]),
            minimum_free_bytes=0,
            free_bytes_probe=lambda _path, size=remaining_high_water: size,
            clock_ns=lambda: 803,
        )
        assert progress.disposition == "RESTORED"
    receipt = finalize_remote_stream_restore(
        plan_path=plan_path,
        eviction_receipt_path=eviction_path,
        remote_archive_result_path=case.result_path,
        progress_root=progress_root,
        receipt_output_path=case.root / "control" / "stream-restore-receipt.json",
        lock_path=case.root / "control" / "stream-finalize.lock",
        clock_ns=lambda: 804,
    )
    assert {
        row.absolute_path
        for row in (*receipt.restored_files, *receipt.already_present_files)
    } == {row.absolute_path for row in plan.files}
    assert len(receipt.already_present_files) == 1
    # A completed member replays without replacing the target.
    assert (
        restore_remote_member_from_stream(
            plan_path=plan_path,
            eviction_receipt_path=eviction_path,
            remote_archive_result_path=case.result_path,
            archive_relative_path=first.archive_relative_path,
            progress_output_path=progress_root / "00000000.json",
            lock_path=case.root / "control" / "stream.lock",
            stream=io.BytesIO(source_bytes[first.archive_relative_path]),
            minimum_free_bytes=0,
            clock_ns=lambda: 805,
        )
        == recovered
    )
    assert (
        finalize_remote_stream_restore(
            plan_path=plan_path,
            eviction_receipt_path=eviction_path,
            remote_archive_result_path=case.result_path,
            progress_root=progress_root,
            receipt_output_path=case.root / "control" / "stream-restore-receipt.json",
            lock_path=case.root / "control" / "stream-finalize.lock",
            clock_ns=lambda: 806,
        )
        == receipt
    )
    Path(first.absolute_path).write_bytes(b"x" * first.size_bytes)
    with pytest.raises(FormalRollingArchiveError, match="target identity"):
        finalize_remote_stream_restore(
            plan_path=plan_path,
            eviction_receipt_path=eviction_path,
            remote_archive_result_path=case.result_path,
            progress_root=progress_root,
            receipt_output_path=case.root / "control" / "stream-restore-receipt.json",
            lock_path=case.root / "control" / "stream-finalize.lock",
            clock_ns=lambda: 807,
        )


def test_stream_restore_rejects_tamper_escape_and_low_capacity(
    tmp_path: Path,
) -> None:
    case = _authorized_case(tmp_path)
    plan = _plan(case)
    plan_path = _publish_plan(case, plan)
    eviction_path = case.root / "control" / "eviction-receipt.json"
    assert (
        execute_remote_eviction_plan(
            plan_path=plan_path,
            receipt_path=eviction_path,
            lock_path=case.root / "control" / "execute.lock",
            operator_snapshot=case.snapshot,
            scheduler_stop=lambda _reason: None,
            active_writer_probe=lambda _path: False,
            process_probe=lambda _pid, _pgid: False,
            clock_ns=lambda: 900,
        ).status
        == "COMPLETE"
    )
    binding = plan.files[0]
    restore_high_water = sum(row.size_bytes for row in plan.files)
    arguments = {
        "plan_path": plan_path,
        "eviction_receipt_path": eviction_path,
        "remote_archive_result_path": case.result_path,
        "progress_output_path": case.root
        / "control"
        / "stream-progress"
        / "00000000.json",
        "lock_path": case.root / "control" / "stream.lock",
        "minimum_free_bytes": 10,
        "clock_ns": lambda: 901,
    }
    with pytest.raises(FormalRollingArchiveError, match="not one exact"):
        restore_remote_member_from_stream(
            **arguments,
            archive_relative_path="../escape",
            stream=io.BytesIO(b""),
            free_bytes_probe=lambda _path: 1_000,
        )
    with pytest.raises(FormalRollingArchiveError, match="capacity"):
        restore_remote_member_from_stream(
            **arguments,
            archive_relative_path=binding.archive_relative_path,
            stream=io.BytesIO(b"x" * binding.size_bytes),
            free_bytes_probe=lambda _path: restore_high_water + 9,
        )
    assert not Path(binding.absolute_path).exists()
    with pytest.raises(FormalRollingArchiveError, match="identity differs"):
        restore_remote_member_from_stream(
            **arguments,
            archive_relative_path=binding.archive_relative_path,
            stream=io.BytesIO(b"x" * binding.size_bytes),
            free_bytes_probe=lambda _path: restore_high_water + 10,
        )
    assert not Path(binding.absolute_path).exists()
