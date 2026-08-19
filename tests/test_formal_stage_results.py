from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_registry_power_sources import _signature_parts

import lightcone_spec.cli.main as cli_module
from lightcone_spec.cli.main import (
    _formal_stage_operation,
    _write_json,
)
from lightcone_spec.experiments.e4_stage_authority import (
    E4ConfigurationEvaluation,
    E4StageSelectionReceipt,
    SignedE4StageSelectionReceipt,
    reduce_e4_profiler_completion_from_registry,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
)
from lightcone_spec.experiments.formal_stage_prefix import (
    FORMAL_STAGE_PREFIX_ARTIFACT_KIND,
    FormalStagePrefixArtifact,
    publish_formal_stage_prefix_artifact,
)
from lightcone_spec.experiments.profiler_authority import ProfilerAuthorityBlocked
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    SignedStageCoverageReceipt,
    SignedStageMaterializationReceipt,
    StageCellDisposition,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return content_sha256({"formal-stage-result-test": label})


def _signed_local_selection(*, lock_sha256: str, registry_sha256: str):
    winner = (
        ("update_stride", 1),
        ("microbatch", 1),
        ("coalescing", 1),
        ("stream_priority", "default"),
    )
    evaluation = E4ConfigurationEvaluation(
        configuration=winner,
        cell_ids=tuple(sorted(_sha(f"stratum-{index}") for index in range(6))),
        minimum_request_rate_numerator=2,
        minimum_request_rate_denominator=1,
        peak_hbm_bytes=1,
        p99_itl_us=1,
        exposed_update_us=1,
    )
    payload = E4StageSelectionReceipt(
        schema_version=1,
        phase="local",
        protocol_lock_sha256=lock_sha256,
        registry_sha256=registry_sha256,
        materialization_receipt_sha256=_sha("local-materialization"),
        coverage_receipt_sha256=_sha("local-coverage"),
        upstream_signed_authority_sha256=_sha("screen-selection"),
        evidence_manifest_sha256=_sha("local-evidence"),
        inventory_sha256=_sha("inventory"),
        model="Qwen/Qwen3-8B",
        lightcone_recipe_sha256=_sha("recipe"),
        evaluations=(evaluation,),
        winner_configuration=winner,
        factor_neighborhoods=None,
    )
    return SignedE4StageSelectionReceipt(
        payload,
        *_signature_parts(payload, key_id="e4-local-release-signer"),
    )


def _profiler_registry(monkeypatch):
    lock_sha256 = _sha("lock")
    registry_sha256 = _sha("registry")
    selection = _signed_local_selection(
        lock_sha256=lock_sha256,
        registry_sha256=registry_sha256,
    )
    cells = tuple(
        sorted(
            (
                MaterializedCell(
                    stage="E4",
                    method_role="LightCone",
                    model="Qwen/Qwen3-8B",
                    backend="DFLASH",
                    task="mechanism_profile_only",
                    publication_policy="diagnostic_only",
                    recipe_sha256=_sha("recipe"),
                    dimensions=(("profiler", profiler),),
                )
                for profiler in ("nvtx", "nsight_systems", "nsight_compute")
            ),
            key=lambda row: row.cell_id,
        )
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E4",
        protocol_lock_sha256=lock_sha256,
        upstream_receipt_sha256s=(_sha("local-materialization"),),
        source_decision_sha256=selection.sha256,
        materialization_rule="three_profiler_only_rows_separate_from_headline",
        expected_cell_count=3,
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="E4",
        protocol_lock_sha256=lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage="E4",
                cell_id=cell.cell_id,
                status="COMPLETE",
                reason_code="terminal_receipt_verified",
                terminal_receipt_sha256=_sha(f"terminal:{cell.cell_id}"),
            )
            for cell in cells
        ),
    )
    signed_materialization = SignedStageMaterializationReceipt(
        materialization,
        *_signature_parts(materialization, key_id="e4-profiler-materialization"),
    )
    signed_coverage = SignedStageCoverageReceipt(
        coverage,
        *_signature_parts(coverage, key_id="e4-profiler-coverage"),
    )
    receipt = object.__new__(FormalRegistryVerificationReceipt)
    object.__setattr__(
        receipt,
        "signed_protocol_lock",
        SimpleNamespace(
            payload=SimpleNamespace(
                sha256=lock_sha256,
                registry_sha256=registry_sha256,
            )
        ),
    )
    object.__setattr__(receipt, "prior_receipt", None)
    object.__setattr__(
        receipt, "appended_signed_materializations", (signed_materialization,)
    )
    object.__setattr__(receipt, "appended_signed_coverage", (signed_coverage,))
    object.__setattr__(receipt, "appended_signed_e4_stage_selections", (selection,))
    object.__setattr__(receipt, "sha256", _sha("registry-receipt"))
    monkeypatch.setattr(
        FormalRegistryVerificationReceipt,
        "revalidate",
        lambda self, *, current_ns: SimpleNamespace(
            receipt_sha256=self.sha256,
            verified_ns=current_ns,
        ),
    )
    return receipt, materialization, coverage


def test_e4_profiler_completion_rejects_generic_registry_coverage(
    monkeypatch,
) -> None:
    registry, materialization, _coverage = _profiler_registry(monkeypatch)
    with pytest.raises(ProfilerAuthorityBlocked, match="allowlist_empty"):
        reduce_e4_profiler_completion_from_registry(
            registry_verification_receipt=registry,
            materialization=materialization,
            now_ns=10,
        )


def test_e4_profiler_cli_is_blocked_without_raw_profiler_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry, _materialization, _coverage = _profiler_registry(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_load_formal_registry_receipt_path",
        lambda _path, *, now_ns: registry,
    )
    registry_path = tmp_path / "registry.json"
    output = tmp_path / "e4-profiler-result.json"
    _write_json(registry_path, {"kind": "typed-test-registry"})
    with pytest.raises(ProfilerAuthorityBlocked, match="allowlist_empty"):
        _formal_stage_operation(
            argparse.Namespace(
                operation="reduce",
                stage="E4",
                phase="profiler",
                registry_verification_receipt=str(registry_path),
                e0_authority_bundle=None,
                e0_materialization=None,
                result_rebuild_artifact=None,
                signed_stage_result=None,
                now_ns=10,
                output=str(output),
            )
        )
    assert not output.exists()


def test_e4_profiler_prefix_cannot_materialize_e3b_pilot_without_raw_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry, _materialization, _coverage = _profiler_registry(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_load_formal_registry_receipt_path",
        lambda _path, *, now_ns: registry,
    )
    registry_path = tmp_path / "registry.json"
    coverage_proof_path = tmp_path / "coverage-proof.json"
    prior_prefix_path = tmp_path / "e4-local-prefix.json"
    prefix_path = tmp_path / "e4-profiler-prefix.json"
    output = tmp_path / "e3b-pilot-materialization.json"
    publish_canonical_json_no_replace(registry_path, {"kind": "typed-test-registry"})
    publish_canonical_json_no_replace(
        coverage_proof_path, {"kind": "typed-test-coverage-proof"}
    )
    publish_canonical_json_no_replace(
        prior_prefix_path, {"kind": "typed-test-prior-prefix"}
    )
    publish_formal_stage_prefix_artifact(
        FormalStagePrefixArtifact(
            schema_version=2,
            kind=FORMAL_STAGE_PREFIX_ARTIFACT_KIND,
            phase="e4_profiler",
            registry_verification_receipt_source=CanonicalJsonProofBinding.bind(
                registry_path
            ),
            coverage_proof_source=CanonicalJsonProofBinding.bind(coverage_proof_path),
            e1_recipe_anchor_authority_source=None,
            prior_prefix_source=CanonicalJsonProofBinding.bind(prior_prefix_path),
        ),
        prefix_path,
    )

    with pytest.raises(ProfilerAuthorityBlocked, match="allowlist_empty"):
        _formal_stage_operation(
            argparse.Namespace(
                operation="materialize",
                stage="E3b",
                phase="pilot",
                registry_verification_receipt=str(registry_path),
                stage_prefix_artifact=str(prefix_path),
                e0_authority_bundle=None,
                e0_materialization=None,
                result_rebuild_artifact=None,
                signed_stage_result=None,
                now_ns=10,
                output=str(output),
            )
        )
    assert not output.exists()
