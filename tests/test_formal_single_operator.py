from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.config import ModelPair, RunConfig, RuntimeConfig, run_config_sha256
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.orchestration.live_sglang import (
    PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
)
from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding
from lightcone_spec.runtime import formal_single_operator as single
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _run_config() -> RunConfig:
    return RunConfig(
        method="target_only",
        model=ModelPair(
            target_revision="1" * 40,
            drafter_revision="2" * 40,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256="3" * 64,
            speculation_enabled=False,
        ),
    )


def _artifacts(run_root: Path) -> tuple[single.FormalSingleOperatorArtifact, ...]:
    rows = []
    for name in (
        "after_gpu_snapshot",
        "before_gpu_snapshot",
        "junit",
        "raw_terminal",
        "native_itl",
        "lifecycle",
        "live_run_receipt",
        "ready_gpu_snapshot",
        "request_schedule",
        "run_plan",
        "server_log",
        "stdout",
        "stderr",
    ):
        path = run_root / f"{name}.json"
        path.write_text(f"{name}\n", encoding="utf-8")
        rows.append(
            single.FormalSingleOperatorArtifact.observe(
                name=name,
                run_root=run_root,
                path=path,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.name))


def _manifest(
    *,
    run_root: Path,
    git_head: str = "4" * 40,
    git_tree: str = "5" * 40,
    patch_manifest_sha256: str = "7" * 64,
) -> single.FormalSingleOperatorRunManifest:
    config = _run_config()
    launch_argv = ("python", "-m", "sglang.launch_server", "--port", "31001")
    request_schedule = {
        "kind": "formal_serving_request_schedule_receipt",
        "rows": [{"request_id": "request-0", "phase": "scored"}],
    }
    return single.FormalSingleOperatorRunManifest(
        schema="formal_single_operator_v1",
        protocol_sha256=single.FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256,
        trust_assumptions=single.FORMAL_SINGLE_OPERATOR_TRUST_ASSUMPTIONS,
        git_head=git_head,
        git_tree=git_tree,
        sglang_upstream_commit="6" * 40,
        patch_manifest_sha256=patch_manifest_sha256,
        patched_sglang_tree="8" * 40,
        registry_sha256=build_industrial_registry().sha256,
        physical_dispatch_protocol_sha256="9" * 64,
        run_plan_sha256="8" * 64,
        launch_manifest_sha256="7" * 64,
        execution_binding_sha256="6" * 64,
        execution_subject_sha256="5" * 64,
        materialization_protocol_lock_sha256="a" * 64,
        materialization_sha256="a" * 64,
        inventory_sha256="9" * 64,
        run_config_sha256=run_config_sha256(config),
        run_config=config.model_dump(mode="json"),
        launch_argv_sha256=single._content_sha256({"argv": list(launch_argv)}),
        launch_argv=launch_argv,
        localhost_port=31_001,
        request_schedule_sha256=single._content_sha256(request_schedule),
        request_schedule=request_schedule,
        target_model_id="target/model",
        target_revision="b" * 40,
        target_content_sha256="c" * 64,
        drafter_model_id=None,
        drafter_revision=None,
        drafter_content_sha256=None,
        tokenizer_model_id="tokenizer/model",
        tokenizer_revision="d" * 40,
        tokenizer_content_sha256="e" * 64,
        workload_artifact_id="workload:member",
        workload_authority_sha256="f" * 64,
        workload_member_sha256s=("f" * 64,),
        workload_raw_sha256="0" * 64,
        workload_semantic_sha256="1" * 64,
        stage="E3a",
        cell_id="cell-001",
        role="Target-only",
        backend="DFLASH",
        topology="tp1_dp1",
        block=0,
        attempt="attempt-0",
        run_directory=str(run_root.resolve()),
        gpu_environment=(
            single.FormalSingleOperatorGpu(
                uuid="GPU-0001",
                model="NVIDIA A100-SXM4-80GB",
                driver_version="550.54.15",
                cuda_version="12.4",
            ),
        ),
        started_ns=10,
        finished_ns=20,
        exit_code=0,
        completion_status="COMPLETE",
        failure_reason=None,
        artifacts=_artifacts(run_root),
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str, str, str]:
    repository = tmp_path / "source"
    patch_root = repository / "patches" / "sglang"
    patch_root.mkdir(parents=True)
    patch = patch_root / "0001.patch"
    patch.write_text("test patch\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "upstream": {
            "repository": "https://example.invalid/sglang",
            "commit": "6" * 40,
        },
        "expected_tree": "8" * 40,
        "patches": [
            {
                "file": patch.name,
                "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                "files": ["python/example.py"],
            }
        ],
    }
    (patch_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test Operator")
    _git(repository, "config", "user.email", "operator@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "source identity")
    return (
        repository,
        _git(repository, "rev-parse", "HEAD"),
        _git(repository, "rev-parse", "HEAD^{tree}"),
        single._content_sha256(manifest),
    )


def _publish_manifest(
    path: Path,
    manifest: single.FormalSingleOperatorRunManifest,
) -> None:
    raw_sha256, size = publish_canonical_json_no_replace(path, manifest.to_dict())
    publish_canonical_json_no_replace(
        path.with_name("formal-single-operator-manifest.sha256.json"),
        {
            "schema": "formal_single_operator_manifest_pointer_v1",
            "manifest_raw_sha256": raw_sha256,
            "manifest_semantic_sha256": manifest.sha256,
            "manifest_size": size,
        },
    )


def test_single_operator_manifest_round_trip_is_stable_after_cached_sha(
    tmp_path: Path,
) -> None:
    manifest = _manifest(run_root=tmp_path)
    expected = manifest.to_dict()
    assert manifest.sha256 == expected["manifest_sha256"]
    assert manifest.to_dict() == expected
    assert single.FormalSingleOperatorRunManifest.from_dict(expected) == manifest


def test_trusted_content_manifest_is_tagged_without_offline_authority_claims(
    tmp_path: Path,
) -> None:
    legacy = _manifest(run_root=tmp_path)
    trusted = replace(
        legacy,
        schema=single.TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_MODE,
        protocol_sha256=(single.TRUSTED_CONTENT_FORMAL_SINGLE_OPERATOR_PROTOCOL_SHA256),
        content_source_mode="trusted_single_operator",
        target_content_sha256=None,
        tokenizer_content_sha256=None,
        workload_authority_sha256=None,
        trusted_content_source_binding_sha256="2" * 64,
        trusted_content_bundle_sha256="3" * 64,
        target_content_member_sha256="4" * 64,
        target_tree_sha256="5" * 64,
        target_snapshot_content_sha256="6" * 64,
        tokenizer_content_member_sha256="7" * 64,
        tokenizer_tree_sha256="8" * 64,
        tokenizer_snapshot_content_sha256="9" * 64,
        trusted_workload_member_sha256="a" * 64,
    )
    value = trusted.to_dict()

    assert value["target_content_sha256"] is None
    assert value["tokenizer_content_sha256"] is None
    assert value["workload_authority_sha256"] is None
    assert value["trusted_content_bundle_sha256"] == "3" * 64
    assert single.FormalSingleOperatorRunManifest.from_dict(value) == trusted

    with pytest.raises(ValueError, match="trusted single-operator content"):
        replace(trusted, target_content_sha256="c" * 64)


def test_single_operator_run_directory_and_manifest_are_no_replace(
    tmp_path: Path,
) -> None:
    repository, _head, _tree, _patch_sha256 = _source_repository(tmp_path)
    run = single.create_formal_single_operator_run_directory(
        repository_root=repository,
        base_output_root=tmp_path,
        stage="E3a",
        cell_id="cell-001",
        attempt="attempt-0",
        started_ns=10,
    )
    with pytest.raises(FileExistsError):
        single.create_formal_single_operator_run_directory(
            repository_root=repository,
            base_output_root=tmp_path,
            stage="E3a",
            cell_id="cell-001",
            attempt="attempt-0",
            started_ns=10,
        )
    manifest = _manifest(run_root=run)
    destination = run / "formal-single-operator-manifest.json"
    _publish_manifest(destination, manifest)
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_canonical_json_no_replace(destination, manifest.to_dict())


def test_single_operator_run_directory_rejects_output_inside_checkout(
    tmp_path: Path,
) -> None:
    repository, _head, _tree, _patch_sha256 = _source_repository(tmp_path)
    output_root = repository / "artifacts"
    output_root.mkdir()
    with pytest.raises(ValueError, match="outside the Git checkout"):
        single.create_formal_single_operator_run_directory(
            repository_root=repository,
            base_output_root=output_root,
            stage="E3a",
            cell_id="cell-001",
            attempt="attempt-0",
            started_ns=10,
        )


def test_single_operator_revalidation_detects_output_change(tmp_path: Path) -> None:
    repository, head, tree, patch_sha256 = _source_repository(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest = _manifest(
        run_root=run_root,
        git_head=head,
        git_tree=tree,
        patch_manifest_sha256=patch_sha256,
    )
    path = run_root / "formal-single-operator-manifest.json"
    _publish_manifest(path, manifest)
    assert (
        single.revalidate_formal_single_operator_run_manifest(
            repository_root=repository,
            manifest_path=path,
        )
        == manifest
    )
    (run_root / "stdout.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output changed"):
        single.revalidate_formal_single_operator_run_manifest(
            repository_root=repository,
            manifest_path=path,
        )


def test_single_operator_rejects_git_tree_as_sha256(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Git tree"):
        _manifest(run_root=tmp_path, git_tree="5" * 64)


def test_single_operator_finalizer_has_no_caller_outcome_or_runtime_knobs() -> None:
    assert tuple(
        inspect.signature(single.finalize_formal_single_operator_run).parameters
    ) == (
        "repository_root",
        "run_plan_path",
        "execution_source_path",
        "inventory_path",
    )


def test_distributed_complete_manifest_requires_gang_terminal(tmp_path: Path) -> None:
    manifest = _manifest(run_root=tmp_path)
    with pytest.raises(ValueError, match="required outputs"):
        replace(manifest, topology="tp2_dp1")


def test_tp1_source_owned_fatal_pointer_projects_failed_provenance(
    tmp_path: Path,
) -> None:
    launch_path = tmp_path / "launch.json"
    publish_canonical_json_no_replace(launch_path, {"kind": "launch"})
    launch = CanonicalJsonProofBinding.bind(launch_path)
    binding = NativeTerminalRunBinding(
        run_id="single-failed-run",
        run_nonce_sha256="1" * 64,
        execution_plan_sha256="2" * 64,
        rank_config_sha256="3" * 64,
        attempt_id="attempt-0000",
        session_id="single-failed-session",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256="4" * 64,
        method="static",
        warmup_request_ids=(),
        scored_request_ids=("scored-0",),
    )
    fatal_path = tmp_path / "fatal.json"
    publish_canonical_json_no_replace(
        fatal_path,
        {
            "schema_version": 1,
            "kind": "unsigned_pinned_sglang_serving_fatal_pointer",
            "protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
            "status": "ERROR",
            "formal_execution_authorized": False,
            "reason_code": "server_failed",
            "error_type": "RuntimeError",
            "cleanup_error_type": None,
            "emitted_ns": 200,
            "execution_started_ns": 120,
            "run_binding_sha256": single._content_sha256(binding.begin_payload()),
            "requested_launch_manifest_path": str(launch_path),
            "launch_manifest": launch.to_dict(),
            "terminal_artifact": None,
            "native_itl_pointer_artifact": None,
            "live_run_receipt": None,
            "server_log": None,
            "server_process_id": None,
            "server_process_exit_code": None,
            "process_exited_ns": None,
            "process_group_empty": None,
            "process_group_empty_checked_ns": None,
            "before_gpu_snapshot": None,
            "ready_gpu_snapshot": None,
            "after_gpu_snapshot": None,
        },
    )
    plan = SimpleNamespace(
        fatal_output_path=str(fatal_path),
        native_terminal_binding=binding,
        launch_manifest=launch,
        terminal_output_path=str(tmp_path / "terminal.json"),
        native_itl_pointer_output_path=str(tmp_path / "itl.json"),
        live_run_receipt_output_path=str(tmp_path / "live.json"),
        before_gpu_snapshot_output_path=str(tmp_path / "before.json"),
        ready_gpu_snapshot_output_path=str(tmp_path / "ready.json"),
        after_gpu_snapshot_output_path=str(tmp_path / "after.json"),
    )
    assert single._failed_tp1_outcome(plan=plan) == (
        120,
        200,
        None,
        "server_failed:RuntimeError",
    )
