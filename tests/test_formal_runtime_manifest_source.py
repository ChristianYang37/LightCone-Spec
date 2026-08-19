from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lightcone_spec.cli.main import main
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS,
)
from lightcone_spec.experiments.formal_registry import (
    formal_runtime_authority_manifest_from_dict,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    FORMAL_RUNTIME_AUTHORITY_ID,
    FORMAL_RUNTIME_SOURCE_LAYOUT,
    build_source_formal_runtime_authority_manifest,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
    FORMAL_SERVING_EXECUTION_RUNNER_SHA256,
    FORMAL_SERVING_EXECUTION_TEST_SET_SHA256,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_source_runtime_manifest_has_exact_closed_members_and_is_deterministic() -> (
    None
):
    first = build_source_formal_runtime_authority_manifest(_repository_root())
    second = build_source_formal_runtime_authority_manifest(_repository_root())
    assert first == second
    assert first.authority_id == FORMAL_RUNTIME_AUTHORITY_ID
    assert tuple(row.member_id for row in first.members) == (
        FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
    )
    assert tuple(row.member_id for row in FORMAL_RUNTIME_SOURCE_LAYOUT) == (
        FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
    )
    assert all(
        len(
            {
                row.protocol_sha256,
                row.runner_sha256,
                row.test_set_sha256,
                row.source_sha256,
            }
        )
        == 4
        for row in first.members
    )
    execution = first.member("all_stage_execution_mapper")
    assert (
        execution.protocol_sha256,
        execution.runner_sha256,
        execution.test_set_sha256,
    ) == (
        FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
        FORMAL_SERVING_EXECUTION_RUNNER_SHA256,
        FORMAL_SERVING_EXECUTION_TEST_SET_SHA256,
    )
    gpu_hour = next(
        row
        for row in FORMAL_RUNTIME_SOURCE_LAYOUT
        if row.member_id == "gpu_hour_budget_reducer"
    )
    assert gpu_hour.runner_sources == (
        "src/lightcone_spec/experiments/gpu_hour_authority.py",
        "src/lightcone_spec/experiments/formal_gpu_hour_registry.py",
        "src/lightcone_spec/experiments/formal_gpu_hour_proof.py",
        "src/lightcone_spec/runtime/scientific_source_validation.py",
    )
    assert gpu_hour.test_nodes == (
        "tests/test_gpu_hour_authority.py",
        "tests/test_formal_gpu_hour_registry.py",
        "tests/test_formal_gpu_hour_proof.py",
        "tests/test_gpu_hour_operator_cli.py",
        "tests/test_offline_scientific_signing.py",
    )
    coverage = next(
        row
        for row in FORMAL_RUNTIME_SOURCE_LAYOUT
        if row.member_id == "stage_coverage_reducer"
    )
    assert coverage.runner_sources == (
        "src/lightcone_spec/experiments/formal_downstream_prefix.py",
        "src/lightcone_spec/experiments/formal_stage_coverage.py",
        "src/lightcone_spec/experiments/formal_stage_coverage_portable.py",
        "src/lightcone_spec/experiments/formal_materialization_shards.py",
        "src/lightcone_spec/runtime/scientific_source_validation.py",
    )
    assert coverage.test_nodes == (
        "tests/test_formal_stage_coverage.py",
        "tests/test_formal_stage_coverage_portable.py",
        "tests/test_offline_scientific_signing.py",
        "tests/test_tts_calibration_authority.py",
    )
    e1 = next(
        row
        for row in FORMAL_RUNTIME_SOURCE_LAYOUT
        if row.member_id == "e1_pareto_reducer"
    )
    assert e1.runner_sources == (
        "src/lightcone_spec/experiments/e1_stage_authority.py",
        "src/lightcone_spec/orchestration/formal_serving_lift.py",
    )
    assert e1.test_nodes == (
        "tests/test_formal_stage_execution.py",
        "tests/test_stage_itl_proof.py",
    )


def test_gpu_hour_proof_source_edit_changes_runtime_source_identity_only(
    tmp_path: Path,
) -> None:
    source_root = _repository_root()
    checkout = (tmp_path / "checkout").resolve()
    checkout.mkdir()
    shutil.copy2(source_root / "pyproject.toml", checkout / "pyproject.toml")
    relative_paths = {
        "src/lightcone_spec/experiments/formal_protocol.py",
        *(
            path
            for layout in FORMAL_RUNTIME_SOURCE_LAYOUT
            for path in layout.runner_sources
        ),
        *(
            node.partition("::")[0]
            for layout in FORMAL_RUNTIME_SOURCE_LAYOUT
            for node in layout.test_nodes
        ),
    }
    for relative_path in sorted(relative_paths):
        destination = checkout / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative_path, destination)

    before = build_source_formal_runtime_authority_manifest(checkout)
    proof_source = checkout / (
        "src/lightcone_spec/experiments/formal_gpu_hour_proof.py"
    )
    proof_source.write_bytes(proof_source.read_bytes() + b"\n")
    after = build_source_formal_runtime_authority_manifest(checkout)

    changed = tuple(
        before_row.member_id
        for before_row, after_row in zip(before.members, after.members, strict=True)
        if before_row.source_sha256 != after_row.source_sha256
    )
    assert changed == ("gpu_hour_budget_reducer",)
    assert (
        before.member("gpu_hour_budget_reducer").source_sha256
        != after.member("gpu_hour_budget_reducer").source_sha256
    )


def test_formal_itl_wrapper_edit_changes_e1_reducer_source_identity_only(
    tmp_path: Path,
) -> None:
    source_root = _repository_root()
    checkout = (tmp_path / "checkout").resolve()
    checkout.mkdir()
    shutil.copy2(source_root / "pyproject.toml", checkout / "pyproject.toml")
    relative_paths = {
        "src/lightcone_spec/experiments/formal_protocol.py",
        *(
            path
            for layout in FORMAL_RUNTIME_SOURCE_LAYOUT
            for path in layout.runner_sources
        ),
        *(
            node.partition("::")[0]
            for layout in FORMAL_RUNTIME_SOURCE_LAYOUT
            for node in layout.test_nodes
        ),
    }
    for relative_path in sorted(relative_paths):
        destination = checkout / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative_path, destination)

    before = build_source_formal_runtime_authority_manifest(checkout)
    source = checkout / "src/lightcone_spec/orchestration/formal_serving_lift.py"
    source.write_bytes(source.read_bytes() + b"\n")
    after = build_source_formal_runtime_authority_manifest(checkout)

    changed = tuple(
        before_row.member_id
        for before_row, after_row in zip(before.members, after.members, strict=True)
        if before_row.source_sha256 != after_row.source_sha256
    )
    assert changed == ("e1_pareto_reducer",)


def test_runtime_manifest_cli_publishes_no_replace_without_digest_arguments(
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal-runtime-authority.json"
    argv = [
        "publish-formal-runtime-authority-manifest",
        "--repository-root",
        str(_repository_root()),
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    manifest = formal_runtime_authority_manifest_from_dict(
        CanonicalJsonProofBinding.bind(str(output.resolve())).reopen()
    )
    assert manifest == build_source_formal_runtime_authority_manifest(
        _repository_root()
    )
    with pytest.raises(RuntimeError, match="target already exists"):
        main(argv)
    with pytest.raises(SystemExit):
        main(
            [
                *argv,
                "--protocol-sha256",
                "0" * 64,
            ]
        )


def test_runtime_manifest_rejects_symlinked_repository_root(tmp_path: Path) -> None:
    link = tmp_path / "checkout-link"
    link.symlink_to(_repository_root(), target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        build_source_formal_runtime_authority_manifest(link)
    with pytest.raises(ValueError, match="real directory"):
        main(
            [
                "publish-formal-runtime-authority-manifest",
                "--repository-root",
                str(link),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )
