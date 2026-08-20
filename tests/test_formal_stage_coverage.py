from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_registry_power_sources import _signature_parts

from lightcone_spec.experiments import formal_stage_coverage as coverage_authority
from lightcone_spec.experiments.breadth_fdr_authority import (
    SignedE0FormalBreadthFdrReceipt,
    reduce_formal_e0_breadth_fdr_from_projection,
)
from lightcone_spec.experiments.e0_authority_artifact import (
    E0_FINAL_COMPLETION_PROTOCOL_SHA256,
    E0FinalAnalysisProjection,
    E0FinalCompletionReceipt,
    SignedE0FinalCompletionReceipt,
)
from lightcone_spec.experiments.e6_stage_authority import (
    E6_CONFIRMATION_PROTOCOL_SHA256,
    E6ConfirmationReceipt,
    SignedE6ConfirmationReceipt,
)
from lightcone_spec.experiments.formal_materialization_shards import (
    FormalMaterializationShardIndex,
    FormalSignedMaterializationShardWrapper,
    publish_formal_materialization_shard_index,
    publish_formal_signed_materialization_shard_wrapper,
    rebuild_formal_signed_materialization_shard_wrapper,
    revalidate_formal_materialization_shard_index,
)
from lightcone_spec.experiments.formal_protocol import (
    E6_MODELS,
    FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS,
    CandidateStateReplay,
    CandidateStateTerminalPair,
    FormalRuntimeAuthorityManifest,
    FormalRuntimeAuthorityMember,
    ProtocolLock,
    TtsCalibrationAuthority,
    TtsL0CandidateStateCoverage,
)
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
    formal_runtime_authority_manifest_to_dict,
    protocol_lock_to_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_stage_coverage import (
    FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
    FORMAL_STAGE_COVERAGE_RUNNER_SHA256,
    FORMAL_STAGE_COVERAGE_TEST_SET_SHA256,
    FormalSignedStageCoverageProofWrapper,
    FormalStageCoverageEvidenceCell,
    FormalStageCoverageProofArtifact,
    derived_coverage_shards,
    publish_formal_signed_stage_coverage_proof_wrapper,
    publish_formal_stage_coverage_proof_artifact,
    publish_formal_stage_derived_coverage_shard,
    rebuild_formal_stage_bound_materialization,
    reduce_e0_all_na_stage_coverage_from_proofs,
    reduce_formal_serving_stage_coverage_from_proofs,
    reduce_tts_calibration_stage_coverage_from_proofs,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
    FORMAL_SERVING_EXECUTION_RUNNER_SHA256,
    FORMAL_SERVING_EXECUTION_TEST_SET_SHA256,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.industrial_analysis import (
    BoundArtifact,
    IndustrialCellEvidence,
    RawTtsCalibrationEvidenceManifest,
    TtsCalibrationPilotEvidence,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.registry import (
    PILOT_BLOCKS,
    build_industrial_registry,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_ALL_NA_MATERIALIZATION_PROTOCOL_SHA256,
    E0_ALL_NA_MATERIALIZATION_RULE,
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
    E0CompatibilityDecision,
    E0CompatibilityReceipt,
    GpuHourEstimate,
    MaterializedCell,
    SignedE0CompatibilityReceipt,
    SignedStageCoverageReceipt,
    SignedStageMaterializationReceipt,
    StageMaterializationReceipt,
    _materialize_e1_first_slice_from_verified_decisions,
    _materialize_e4_profiler_diagnostic,
    _materialize_e4_strength2_screen_diagnostic,
    _materialize_e6_diagnostic,
    _materialize_tts_calibration_diagnostic,
)
from lightcone_spec.experiments.statistics import (
    PRIMARY_CONTRASTS,
    MultiplicityDecision,
    PairedBcaContrast,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.relocatable_evidence import (
    activate_relocatable_evidence_bundle,
    materialize_relocatable_evidence_bundle,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authority() -> TtsCalibrationAuthority:
    return TtsCalibrationAuthority(
        schema_version=2,
        authority_id="tts-primary-source-reconstruction-v2",
        primary_source_id="arXiv:2605.09329",
        primary_source_version="v2",
        paper_pdf_sha256=_sha("paper-pdf"),
        paper_source_sha256=_sha("paper-source"),
        tuning_window_sha256=_sha("tuning-window"),
        trainable_plan_sha256=_sha("trainable-plan"),
        drafter_native_loss_recipe_sha256=_sha("native-loss"),
    )


def _runtime_manifest() -> FormalRuntimeAuthorityManifest:
    return FormalRuntimeAuthorityManifest(
        schema_version=2,
        authority_id="formal-stage-coverage-test-runtime",
        members=tuple(
            FormalRuntimeAuthorityMember(
                member_id=member_id,
                protocol_sha256=(
                    FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
                    if member_id == "stage_coverage_reducer"
                    else FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256
                    if member_id == "all_stage_execution_mapper"
                    else _sha(f"{member_id}:protocol")
                ),
                runner_sha256=(
                    FORMAL_STAGE_COVERAGE_RUNNER_SHA256
                    if member_id == "stage_coverage_reducer"
                    else FORMAL_SERVING_EXECUTION_RUNNER_SHA256
                    if member_id == "all_stage_execution_mapper"
                    else _sha(f"{member_id}:runner")
                ),
                test_set_sha256=(
                    FORMAL_STAGE_COVERAGE_TEST_SET_SHA256
                    if member_id == "stage_coverage_reducer"
                    else FORMAL_SERVING_EXECUTION_TEST_SET_SHA256
                    if member_id == "all_stage_execution_mapper"
                    else _sha(f"{member_id}:tests")
                ),
                source_sha256=_sha(f"{member_id}:source"),
            )
            for member_id in FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
        ),
    )


def _protocol_lock(
    authority: TtsCalibrationAuthority,
    runtime: FormalRuntimeAuthorityManifest,
) -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="lightcone-formal-stage-coverage-test",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=build_industrial_registry().sha256,
        english_protocol_sha256=_sha("protocol-en"),
        chinese_protocol_sha256=_sha("protocol-zh"),
        tts_calibration_authority_sha256=authority.sha256,
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1-anchor"),
        e2_recipe_grid_authority_sha256=_sha("e2-grid"),
        formal_runtime_authority_manifest_sha256=runtime.sha256,
        offline_release_trust_root_sha256=_sha("release-root"),
        prepared_model_content_authorization_sha256=_sha("prepared-model"),
        formal_workload_e3a_authorization_sha256=_sha("e3a-workload"),
        formal_workload_e0_authorization_sha256=_sha("e0-workload"),
        burstgpt_shape_authorization_sha256=_sha("burstgpt"),
        native_runtime_qualification_protocol_sha256=_sha("native-protocol"),
        native_runtime_qualification_runner_sha256=_sha("native-runner"),
        native_runtime_qualification_test_set_sha256=_sha("native-tests"),
        compile_qualification_protocol_sha256=_sha("compile-protocol"),
        compile_qualification_runner_sha256=_sha("compile-runner"),
        compile_qualification_test_set_sha256=_sha("compile-tests"),
        exactness_qualification_protocol_sha256=_sha("exactness-protocol"),
        exactness_qualification_runner_sha256=_sha("exactness-runner"),
        exactness_qualification_test_set_sha256=_sha("exactness-tests"),
    )


def _inventory() -> GpuInventory:
    device = GpuDevice(
        uuid="GPU-formal-stage-coverage-0",
        host_id="formal-stage-coverage-host",
        model="RTX PRO 6000 Blackwell Server Edition",
        memory_bytes=96 * 1024**3,
        compute_capability=(12, 0),
        pci_bus_id="0000:01:00.0",
        pci_root="root-0",
        numa_node=0,
        interconnects=("PCIe",),
        peer_access_class="P2P",
        clock_policy="locked",
        power_limit_watts=600.0,
        thermal_limit_celsius=83.0,
        availability=GpuAvailability.READY,
        reserved_processes=(),
        allowed_topology_groups=("single",),
    )
    return GpuInventory(
        schema_version=1,
        devices=(device,),
        topology_groups=(
            GpuTopologyGroup(
                group_id="single",
                host_id=device.host_id,
                gpu_uuids=(device.uuid,),
                fabric="PCIe",
                bandwidth_class="local",
            ),
        ),
        source_receipt_sha256=_sha("inventory-source"),
    )


def _unmeasured() -> GpuHourEstimate:
    return GpuHourEstimate.unmeasured()


def _publish(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(str(path))


def _large_e2_materialization(cell_count: int = 105 * 64 + 4):
    cells = tuple(
        sorted(
            (
                MaterializedCell(
                    stage="E2",
                    method_role="LightCone-candidate",
                    model="Qwen/Qwen3.6-35B-A3B",
                    backend="DFLASH",
                    task="successive_halving_round0",
                    publication_policy="first_ready",
                    recipe_sha256=_sha(f"large-e2-recipe-{index}"),
                    dimensions=(("candidate_index", index),),
                )
                for index in range(cell_count)
            ),
            key=lambda row: row.cell_id,
        )
    )
    return StageMaterializationReceipt(
        schema_version=1,
        stage="E2",
        protocol_lock_sha256=_sha("large-e2-protocol"),
        upstream_receipt_sha256s=(_sha("large-e2-upstream"),),
        source_decision_sha256=_sha("large-e2-decision"),
        materialization_rule="registered_105_per_geometry_plus_four_anchors",
        expected_cell_count=len(cells),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _e0_cardinality_materialization(
    decision_ids: tuple[str, ...],
    *,
    phase: str,
    rows_per_decision: int,
) -> StageMaterializationReceipt:
    rules = {
        "onlinespec_tuning": (
            "e0_full_registered_onlinespec_grid_per_valid_combination_tuning_only"
        ),
        "excluded_pilot": (
            "e0_exact_16_rows_per_valid_combination_x_4_excluded_pilot_blocks"
        ),
        "final": (
            "valid_compatibilities_x_8_roles_x_2_loads_x_final_only_powered_prefix"
        ),
    }
    cells = tuple(
        sorted(
            (
                MaterializedCell(
                    stage="E0",
                    method_role="OnlineSPEC-OGD-candidate",
                    model="Qwen/Qwen3.6-35B-A3B",
                    backend="DFLASH",
                    task="e0-cardinality-test",
                    publication_policy="first_ready",
                    recipe_sha256=_sha(f"e0-cardinality-recipe:{decision_id}:{index}"),
                    dimensions=(
                        ("compatibility_decision_id", decision_id),
                        ("row", index),
                    ),
                )
                for decision_id in decision_ids
                for index in range(rows_per_decision)
            ),
            key=lambda row: row.cell_id,
        )
    )
    return StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=_sha("e0-cardinality-protocol"),
        upstream_receipt_sha256s=(_sha("e0-cardinality-upstream"),),
        source_decision_sha256=_sha("e0-cardinality-source"),
        materialization_rule=rules[phase],
        expected_cell_count=len(cells),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def test_e0_positive_cardinality_is_derived_from_exact_valid_decision_ids() -> None:
    valid_ids = tuple(sorted((_sha("e0-valid-0"), _sha("e0-valid-1"))))
    for phase, rows in (
        ("onlinespec_tuning", 239),
        ("excluded_pilot", 64),
        ("final", 16 * 12),
    ):
        materialization = _e0_cardinality_materialization(
            valid_ids,
            phase=phase,
            rows_per_decision=rows,
        )
        coverage_authority._validate_materialization_cardinality(
            materialization,
            phase=phase,
            e0_valid_compatibility_decision_ids=valid_ids,
        )
        with pytest.raises(ValueError, match="signed compatibility|match every"):
            coverage_authority._validate_materialization_cardinality(
                materialization,
                phase=phase,
                e0_valid_compatibility_decision_ids=(valid_ids[0],),
            )
    with pytest.raises(ValueError, match="deep-rebuilt signed compatibility"):
        coverage_authority._validate_materialization_cardinality(
            _e0_cardinality_materialization(
                valid_ids,
                phase="onlinespec_tuning",
                rows_per_decision=239,
            ),
            phase="onlinespec_tuning",
        )
    with pytest.raises(ValueError, match="cardinality differs"):
        coverage_authority._validate_materialization_cardinality(
            _e0_cardinality_materialization(
                valid_ids,
                phase="final",
                rows_per_decision=16 * 11,
            ),
            phase="final",
            e0_valid_compatibility_decision_ids=valid_ids,
        )


def test_e0_all_na_coverage_requires_signed_completion_and_756_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    runtime = _runtime_manifest()
    lock = _protocol_lock(authority, runtime)
    inventory = _inventory()
    e6_blocks = tuple(f"E6:final:{index:02d}" for index in range(12))
    contrasts = tuple(
        PairedBcaContrast(
            name=name,
            block_ids=e6_blocks,
            mean_log_ratio=0.1,
            mean_relative_gain=0.105,
            ci_lower_relative_gain=0.01,
            ci_upper_relative_gain=0.2,
            raw_p_value=0.001,
            confidence=0.95,
        )
        for name in PRIMARY_CONTRASTS
    )
    decisions = tuple(
        MultiplicityDecision(
            name=name,
            raw_p_value=0.001,
            adjusted_p_value=0.002,
            rejected=True,
            procedure="holm",
        )
        for name in PRIMARY_CONTRASTS
    )
    e6_payload = E6ConfirmationReceipt(
        schema_version=1,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        materialization_receipt_sha256=_sha("all-na-e6-materialization"),
        coverage_receipt_sha256=_sha("all-na-e6-coverage"),
        evidence_manifest_sha256=_sha("all-na-e6-evidence"),
        inventory_sha256=inventory.sha256,
        protocol_sha256=E6_CONFIRMATION_PROTOCOL_SHA256,
        upstream_e5_confirmation_sha256=_sha("all-na-e5-confirmation"),
        signed_model_compatibility_sha256=_sha("all-na-e6-compatibility"),
        frozen_tts_recipe_sha256=_sha("all-na-frozen-tts"),
        lightcone_recipe_sha256=_sha("all-na-lightcone"),
        models=E6_MODELS,
        final_block_ids=e6_blocks,
        primary_contrasts=contrasts,
        holm_decisions=decisions,
        status="CONFIRMED",
    )
    signed_e6 = SignedE6ConfirmationReceipt(
        e6_payload,
        *_signature_parts(e6_payload, key_id="all-na-e6-confirmation"),
    )
    compatibility_payload = E0CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=lock.sha256,
        upstream_e6_receipt_sha256=e6_payload.materialization_receipt_sha256,
        decisions=tuple(
            sorted(
                (
                    E0CompatibilityDecision(
                        model=model,
                        backend=backend,
                        task=task,
                        disposition="N/A",
                        reason_code="proof_backed_not_applicable",
                        interface_sha256=_sha(f"all-na-interface:{model}:{backend}"),
                        task_native_workload_sha256=_sha(f"all-na-workload:{task}"),
                    )
                    for model in E0_MODELS
                    for backend in E0_BACKENDS
                    for task in E0_TASKS
                ),
                key=lambda row: row.decision_id,
            )
        ),
    )
    signed_compatibility = SignedE0CompatibilityReceipt(
        compatibility_payload,
        *_signature_parts(compatibility_payload, key_id="all-na-compatibility"),
    )
    source = coverage_authority.content_sha256(
        {
            "protocol_sha256": E0_ALL_NA_MATERIALIZATION_PROTOCOL_SHA256,
            "signed_e6_confirmation_sha256": signed_e6.sha256,
            "signed_e0_compatibility_sha256": signed_compatibility.sha256,
        }
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(e6_payload.materialization_receipt_sha256,),
        source_decision_sha256=source,
        materialization_rule=E0_ALL_NA_MATERIALIZATION_RULE,
        expected_cell_count=0,
        cells=(),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    expected_coverage = coverage_authority.StageCoverageReceipt(
        schema_version=2,
        stage="E0",
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=(),
    )
    registry_receipt = object.__new__(FormalRegistryVerificationReceipt)
    object.__setattr__(registry_receipt, "prior_receipt", None)
    object.__setattr__(
        registry_receipt,
        "appended_signed_e0_compatibilities",
        (signed_compatibility,),
    )
    object.__setattr__(
        registry_receipt,
        "signed_protocol_lock",
        SimpleNamespace(payload=lock),
    )
    object.__setattr__(registry_receipt, "inventory_sha256", inventory.sha256)
    object.__setattr__(registry_receipt, "sha256", _sha("all-na-registry-receipt"))
    manifest = SimpleNamespace(
        materializations=(
            SimpleNamespace(
                materialization_receipt_sha256=(
                    e6_payload.materialization_receipt_sha256
                )
            ),
        ),
        coverage=(
            SimpleNamespace(coverage_receipt_sha256=e6_payload.coverage_receipt_sha256),
        ),
    )
    policy = SimpleNamespace(sha256=_sha("all-na-policy"))
    monkeypatch.setattr(
        FormalRegistryVerificationReceipt,
        "revalidate",
        lambda self, *, current_ns: manifest,
    )
    monkeypatch.setattr(
        FormalRegistryVerificationReceipt,
        "trusted_release_policy",
        lambda self, *, current_ns: policy,
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_protocol.verify_signed_payload",
        lambda *_args, **_kwargs: None,
    )
    lineage = coverage_authority.content_sha256(
        {
            "protocol_sha256": (
                coverage_authority.FORMAL_E0_ALL_NA_COVERAGE_PROTOCOL_SHA256
            ),
            "registry_verification_receipt_sha256": registry_receipt.sha256,
            "signed_e6_confirmation_sha256": signed_e6.sha256,
            "signed_compatibility_sha256": signed_compatibility.sha256,
            "materialization_receipt_sha256": materialization.sha256,
        }
    )
    completion = E0FinalCompletionReceipt(
        schema_version=1,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        prior_registry_verification_receipt_sha256=registry_receipt.sha256,
        current_registry_verification_receipt_sha256=registry_receipt.sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=expected_coverage.sha256,
        stage_source_binding_sha256=lineage,
        evidence_manifest_sha256=coverage_authority.content_sha256(
            {"all_na_lineage_sha256": lineage, "terminal_count": 0}
        ),
        inventory_sha256=inventory.sha256,
        rebuild_artifact_sha256=coverage_authority.content_sha256(
            {"all_na_lineage_sha256": lineage, "rebuild_kind": "zero_cell"}
        ),
        selected_final_prefix=(),
        valid_compatibility_count=0,
        cells=(),
        protocol_sha256=E0_FINAL_COMPLETION_PROTOCOL_SHA256,
    )
    signed_completion = SignedE0FinalCompletionReceipt(
        completion,
        *_signature_parts(completion, key_id="all-na-completion"),
    )
    projection = E0FinalAnalysisProjection(
        schema_version=1,
        completion_receipt=completion,
        compatibility_decisions=compatibility_payload.decisions,
        cells=(),
    )
    fdr = reduce_formal_e0_breadth_fdr_from_projection(
        build_industrial_registry(), projection
    )
    signed_fdr = SignedE0FormalBreadthFdrReceipt(
        fdr,
        *_signature_parts(fdr, key_id="all-na-fdr"),
    )
    kwargs = {
        "protocol_lock": lock,
        "formal_runtime_authority_manifest": runtime,
        "materialization": materialization,
        "inventory": inventory,
        "registry_verification_receipt": registry_receipt,
        "signed_e6_confirmation": signed_e6,
        "signed_compatibility": signed_compatibility,
        "signed_final_completion": signed_completion,
        "signed_formal_fdr": signed_fdr,
        "now_ns": 40_000_000_000,
    }
    assert reduce_e0_all_na_stage_coverage_from_proofs(**kwargs) == expected_coverage
    assert len(fdr.hypotheses) == 756
    assert {row.status for row in fdr.hypotheses} == {"EXCLUDED_NA"}
    assert fdr.decisions == ()
    foreign_completion = replace(
        completion,
        rebuild_artifact_sha256=_sha("foreign-all-na-rebuild"),
    )
    with pytest.raises(ValueError, match="signed completion"):
        reduce_e0_all_na_stage_coverage_from_proofs(
            **{
                **kwargs,
                "signed_final_completion": SignedE0FinalCompletionReceipt(
                    foreign_completion,
                    *_signature_parts(
                        foreign_completion, key_id="foreign-all-na-completion"
                    ),
                ),
            }
        )


def test_materialization_shards_roundtrip_max_e2_and_reject_aliases(
    tmp_path: Path,
) -> None:
    materialization = _large_e2_materialization()
    shard_count = (len(materialization.cells) + 255) // 256
    shard_paths = tuple(
        (tmp_path / f"e2-materialization-{index}.json").resolve()
        for index in range(shard_count)
    )
    index_path = (tmp_path / "e2-materialization-index.json").resolve()
    binding = publish_formal_materialization_shard_index(
        materialization,
        cell_shard_output_paths=shard_paths,
        index_output_path=index_path,
    )
    assert binding.size < 2 * 1024 * 1024
    index = FormalMaterializationShardIndex.from_dict(binding.reopen())
    assert len(index.cell_shard_sources) == shard_count
    assert all(row.size < 2 * 1024 * 1024 for row in index.cell_shard_sources)
    rebuilt = revalidate_formal_materialization_shard_index(
        index_path,
        expected_materialization_receipt_sha256=materialization.sha256,
    )
    assert rebuilt == materialization
    assert rebuilt.sha256 == materialization.sha256
    signed = SignedStageMaterializationReceipt(
        materialization,
        *_signature_parts(materialization, key_id="materialization-shard-wrapper"),
    )
    wrapper_binding = publish_formal_signed_materialization_shard_wrapper(
        signed,
        materialization_index_source=binding,
        output_path=(tmp_path / "signed-materialization-wrapper.json").resolve(),
    )
    wrapper = FormalSignedMaterializationShardWrapper.from_dict(
        wrapper_binding.reopen()
    )
    assert wrapper_binding.size < 2 * 1024 * 1024
    assert wrapper.signed_materialization_receipt_sha256 == signed.sha256
    assert (
        rebuild_formal_signed_materialization_shard_wrapper(
            wrapper_binding.absolute_path
        )
        == signed
    )

    with pytest.raises(ValueError, match="absolute and normalized"):
        publish_formal_materialization_shard_index(
            _large_e2_materialization(1),
            cell_shard_output_paths=(Path("relative-shard.json"),),
            index_output_path=(tmp_path / "relative-index.json").resolve(),
        )
    with pytest.raises(ValueError, match="alias"):
        publish_formal_materialization_shard_index(
            _large_e2_materialization(1),
            cell_shard_output_paths=((tmp_path / "alias.json").resolve(),),
            index_output_path=(tmp_path / "alias.json").resolve(),
        )


def test_materialization_shards_reject_nested_cell_tamper(tmp_path: Path) -> None:
    materialization = _large_e2_materialization(2)
    shard_path = (tmp_path / "materialization-cells.json").resolve()
    index_path = (tmp_path / "materialization-index.json").resolve()
    binding = publish_formal_materialization_shard_index(
        materialization,
        cell_shard_output_paths=(shard_path,),
        index_output_path=index_path,
    )
    index = FormalMaterializationShardIndex.from_dict(binding.reopen())
    tampered = index.cell_shard_sources[0].reopen()
    tampered["cells"][0]["task"] = "tampered-task"
    replacement = (tmp_path / "tampered-cells.json").resolve()
    tampered_source = _publish(replacement, tampered)
    tampered_index = FormalMaterializationShardIndex(
        schema_version=index.schema_version,
        kind=index.kind,
        protocol_sha256=index.protocol_sha256,
        materialization_receipt_sha256=index.materialization_receipt_sha256,
        header=index.header,
        cell_shard_sources=(tampered_source,),
    )
    tampered_index_path = (tmp_path / "tampered-index.json").resolve()
    _publish(tampered_index_path, tampered_index.to_dict())
    with pytest.raises(ValueError, match="digest differs"):
        revalidate_formal_materialization_shard_index(tampered_index_path)


def test_materialization_shards_rebuild_from_distinct_stable_pull_root(
    tmp_path: Path,
) -> None:
    remote_root = tmp_path / "coverage-remote-root"
    remote_root.mkdir(mode=0o700)
    materialization = _large_e2_materialization(2)
    shard_path = (remote_root / "cells.json").resolve()
    index_path = (remote_root / "index.json").resolve()
    remote_index = publish_formal_materialization_shard_index(
        materialization,
        cell_shard_output_paths=(shard_path,),
        index_output_path=index_path,
    )
    local_root = tmp_path / "coverage-local-root"
    local_root.mkdir(mode=0o700)
    bundle = materialize_relocatable_evidence_bundle(
        remote_root=remote_root.resolve(),
        entry_paths=(index_path,),
        local_root=local_root.resolve(),
    )
    remote_root.rename(tmp_path / "coverage-remote-root-offline")
    with activate_relocatable_evidence_bundle(bundle.absolute_path):
        assert (
            rebuild_formal_stage_bound_materialization(
                remote_index,
                expected_receipt_sha256=materialization.sha256,
            )
            == materialization
        )


def _e4_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authority = _authority()
    runtime = _runtime_manifest()
    lock = _protocol_lock(authority, runtime)
    inventory = _inventory()
    materialization = _materialize_e4_strength2_screen_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_e2_receipt_sha256=_sha("e2-materialization"),
        source_decision_sha256=_sha("e2-final-selection"),
        model="Qwen/Qwen3.6-35B-A3B",
        lightcone_recipe_sha256=_sha("lightcone-recipe"),
        gpu_hours=_unmeasured(),
    )
    evidences = []
    bindings = []
    result_by_path: dict[str, str] = {}
    timing_by_path: dict[str, tuple[str, str, str, StageItlExecutionIdentity]] = {}
    for index, cell in enumerate(materialization.cells):
        result = _publish(
            (tmp_path / f"result-{index}.json").resolve(),
            {"kind": "test-result", "row": index},
        )
        timing = _publish(
            (tmp_path / f"timing-{index}.json").resolve(),
            {"kind": "test-timing", "row": index},
        )
        identity = StageItlExecutionIdentity(
            schema_version=1,
            kind="stage_itl_execution_identity",
            materialized_cell_id=cell.cell_id,
            inventory_sha256=inventory.sha256,
            registry_sha256=lock.registry_sha256,
            execution_plan_sha256=_sha(f"plan-{index}"),
            rank_config_sha256=_sha(f"rank-{index}"),
            run_id=f"e4-screen-{index}",
            run_nonce_sha256=_sha(f"nonce-{index}"),
            attempt_id=f"attempt-{index}",
            method="l0",
            runtime_trust_mode=None,
            formal_measurement=None,
        )
        binding_sha256 = _sha(f"execution-binding-{index}")
        subject = SimpleNamespace(
            stage="E4",
            protocol_lock_sha256=lock.sha256,
            materialization_receipt_sha256=materialization.sha256,
            materialized_cell_id=cell.cell_id,
            inventory_sha256=inventory.sha256,
            execution_identity=identity,
            topology_mode="tp1_dp1",
        )
        bindings.append(SimpleNamespace(sha256=binding_sha256, subject=subject))
        evidences.append(
            FormalStageCoverageEvidenceCell(
                schema_version=1,
                materialized_cell_id=cell.cell_id,
                execution_binding_sha256=binding_sha256,
                execution_identity=identity,
                native_result_proof=result,
                stage_itl_proof=timing,
            )
        )
        terminal = _sha(f"terminal-{index}")
        result_by_path[result.absolute_path] = terminal
        timing_by_path[timing.absolute_path] = (
            result.absolute_path,
            result.raw_sha256,
            result.semantic_sha256,
            identity,
        )

    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage."
        "require_verified_formal_serving_execution_binding",
        lambda value: value,
    )

    def validate_result(path: str, **_kwargs):
        return SimpleNamespace(terminal_sha256=result_by_path[path])

    def validate_timing(path: str, **_kwargs):
        result_path, raw, semantic, identity = timing_by_path[path]
        return SimpleNamespace(
            execution_identity=identity,
            native_result_proof_path=result_path,
            native_result_proof_raw_sha256=raw,
            native_result_proof_semantic_sha256=semantic,
        )

    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage."
        "validate_formal_terminal_result_proof_artifact",
        validate_result,
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage."
        "validate_formal_serving_itl_proof",
        validate_timing,
    )
    return lock, runtime, inventory, materialization, tuple(evidences), tuple(bindings)


def test_serving_coverage_is_exact_rerunnable_and_rejects_missing_or_foreign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, runtime, inventory, materialization, evidence, bindings = _e4_bundle(
        tmp_path, monkeypatch
    )
    kwargs = {
        "protocol_lock": lock,
        "formal_runtime_authority_manifest": runtime,
        "materialization": materialization,
        "inventory": inventory,
        "stage": "E4",
        "phase": "screen",
        "evidence_cells": evidence,
        "execution_bindings": bindings,
        "now_ns": 10,
    }
    first = reduce_formal_serving_stage_coverage_from_proofs(**kwargs)
    second = reduce_formal_serving_stage_coverage_from_proofs(**kwargs)
    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.dispositions) == 48
    assert {row.status for row in first.dispositions} == {"COMPLETE"}

    with pytest.raises(ValueError, match="every and only"):
        reduce_formal_serving_stage_coverage_from_proofs(
            **{**kwargs, "evidence_cells": evidence[:-1]}
        )
    foreign = replace(evidence[0], execution_binding_sha256=_sha("foreign-binding"))
    with pytest.raises(ValueError, match="sealed execution binding"):
        reduce_formal_serving_stage_coverage_from_proofs(
            **{**kwargs, "evidence_cells": (foreign, *evidence[1:])}
        )


def test_serving_coverage_rejects_duplicate_proof_and_caller_status_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, runtime, inventory, materialization, evidence, bindings = _e4_bundle(
        tmp_path, monkeypatch
    )
    duplicate = replace(
        evidence[1],
        native_result_proof=evidence[0].native_result_proof,
    )
    with pytest.raises(ValueError, match="reuses a result"):
        reduce_formal_serving_stage_coverage_from_proofs(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            materialization=materialization,
            inventory=inventory,
            stage="E4",
            phase="screen",
            evidence_cells=(evidence[0], duplicate, *evidence[2:]),
            execution_bindings=bindings,
            now_ns=10,
        )
    row = evidence[0].to_dict()
    row["status"] = "COMPLETE"
    with pytest.raises(ValueError, match="fields differ"):
        FormalStageCoverageEvidenceCell.from_dict(row)


def test_e4_profiler_rejects_ordinary_serving_terminal_and_itl_union() -> None:
    authority = _authority()
    runtime = _runtime_manifest()
    lock = _protocol_lock(authority, runtime)
    materialization = _materialize_e4_profiler_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_local_receipt_sha256=_sha("e4-local-materialization"),
        source_decision_sha256=_sha("e4-local-selection"),
        selected_configuration_sha256=_sha("e4-profiler-configuration"),
        model="Qwen/Qwen3.6-35B-A3B",
        lightcone_recipe_sha256=_sha("e4-profiler-lightcone-recipe"),
        gpu_hours=_unmeasured(),
    )
    with pytest.raises(ValueError, match="dedicated profiler-proof reducer"):
        reduce_formal_serving_stage_coverage_from_proofs(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            materialization=materialization,
            inventory=_inventory(),
            stage="E4",
            phase="profiler",
            evidence_cells=(),
            execution_bindings=(),
            now_ns=10,
        )


def test_e5_final_rejects_ordinary_serving_terminal_and_itl_union() -> None:
    authority = _authority()
    runtime = _runtime_manifest()
    lock = _protocol_lock(authority, runtime)
    cells = tuple(
        sorted(
            (
                MaterializedCell(
                    stage="E5",
                    method_role=("LightCone" if index < 264 else "Static"),
                    model="Qwen/Qwen3.6-35B-A3B",
                    backend="DFlash",
                    task=(
                        "deterministic_failure_injection"
                        if index < 264
                        else "production_slo_power_prefix"
                    ),
                    publication_policy=("diagnostic_only" if index < 264 else "none"),
                    recipe_sha256=(
                        _sha("e5-lightcone-recipe") if index < 264 else None
                    ),
                    dimensions=(("test_row_index", index),),
                )
                for index in range(264 + 450 * 12)
            ),
            key=lambda row: row.cell_id,
        )
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E5",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("e1a-materialization"),),
        source_decision_sha256=_sha("e5-final-source"),
        materialization_rule=(
            "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
        ),
        expected_cell_count=len(cells),
        cells=cells,
        gpu_hours=_unmeasured(),
    )
    with pytest.raises(ValueError, match="closed failure-proof reducer"):
        reduce_formal_serving_stage_coverage_from_proofs(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            materialization=materialization,
            inventory=_inventory(),
            stage="E5",
            phase="final_and_one_shot_failure",
            evidence_cells=(),
            execution_bindings=(),
            now_ns=10,
        )


def test_e6_final_rejects_repeated_model_preflights() -> None:
    authority = _authority()
    runtime = _runtime_manifest()
    lock = _protocol_lock(authority, runtime)
    materialization = _materialize_e6_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256=_sha("e5-materialization"),
        source_decision_sha256=_sha("e6-source"),
        frozen_tts_recipe_sha256=_sha("e6-frozen-tts"),
        lightcone_recipe_sha256=_sha("e6-lightcone"),
        final_blocks=12,
        gpu_hours=_unmeasured(),
    )
    materialization = replace(
        materialization,
        materialization_rule=(
            "60_final_rows_per_block_reusing_global_model_preflights"
        ),
    )
    with pytest.raises(ValueError, match="cardinality"):
        reduce_formal_serving_stage_coverage_from_proofs(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            materialization=materialization,
            inventory=_inventory(),
            stage="E6",
            phase="final",
            evidence_cells=(),
            execution_bindings=(),
            now_ns=10,
        )


def test_e1_candidate_pair_is_replayed_and_terminal_joined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    runtime = _runtime_manifest()
    lock = _protocol_lock(authority, runtime)
    inventory = _inventory()
    materialization = _materialize_e1_first_slice_from_verified_decisions(
        protocol_lock_sha256=lock.sha256,
        tts_calibration_receipt_sha256=_sha("tts-calibration"),
        signed_tts_calibration_seal_sha256=_sha("tts-seal"),
        e3a_selection_sha256=_sha("e3a-selection"),
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        e1_recipe_anchor_authority_sha256=lock.e1_recipe_anchor_authority_sha256,
        model="Qwen/Qwen3.6-35B-A3B",
        matched_width=4,
        common_load=2,
        gpu_hours=_unmeasured(),
    )
    tts = next(row for row in materialization.cells if row.method_role == "TTS")
    l0 = next(row for row in materialization.cells if row.method_role == "L0-naive")
    pair_id = dict(tts.dimensions)["tts_l0_pair_id"]
    trainable = _sha("candidate-trainable")
    shared = {
        "source_round": 1,
        "source_version": 0,
        "source_state_sha256": _sha("candidate-source"),
        "trainable_plan_sha256": trainable,
        "candidate_bytes_sha256": _sha("candidate-bytes"),
        "optimizer_state_bytes_sha256": _sha("candidate-optimizer"),
        "proposal_evidence_sha256": _sha("candidate-proposal"),
    }
    tts_pointer = _sha("tts-pointer")
    l0_pointer = _sha("l0-pointer")
    tts_terminal = _sha("tts-terminal")
    l0_terminal = _sha("l0-terminal")
    candidate = TtsL0CandidateStateCoverage(
        schema_version=1,
        stage="E1",
        scope="materialized_pair",
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        pair_id=pair_id,
        tts_cell_id=tts.cell_id,
        l0_naive_cell_id=l0.cell_id,
        tts_native_replay_pointer_sha256=tts_pointer,
        l0_naive_native_replay_pointer_sha256=l0_pointer,
        qualification_cell_id=None,
        source_round_plan_sha256=_sha("candidate-round-plan"),
        trainable_plan_sha256=trainable,
        expected_source_rounds=(1,),
        tts_observations=(
            CandidateStateReplay(
                method_role="TTS",
                cell_id=tts.cell_id,
                native_replay_pointer_sha256=tts_pointer,
                run_id="e1-tts-run",
                publication_policy="fixed_barrier",
                **shared,
            ),
        ),
        l0_naive_observations=(
            CandidateStateReplay(
                method_role="L0-naive",
                cell_id=l0.cell_id,
                native_replay_pointer_sha256=l0_pointer,
                run_id="e1-l0-run",
                publication_policy="first_ready",
                **shared,
            ),
        ),
        terminal_pairs=(
            CandidateStateTerminalPair(
                source_round=1,
                tts_cell_id=tts.cell_id,
                l0_naive_cell_id=l0.cell_id,
                tts_run_id="e1-tts-run",
                l0_naive_run_id="e1-l0-run",
                tts_native_replay_pointer_sha256=tts_pointer,
                l0_naive_native_replay_pointer_sha256=l0_pointer,
                proposal_evidence_sha256=shared["proposal_evidence_sha256"],
                tts_terminal_receipt_sha256=tts_terminal,
                l0_naive_terminal_receipt_sha256=l0_terminal,
            ),
        ),
    )
    pointers = {
        "/proof/tts.json": SimpleNamespace(semantic_commitment_sha256=tts_pointer),
        "/proof/l0.json": SimpleNamespace(semantic_commitment_sha256=l0_pointer),
    }
    replayed: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        coverage_authority,
        "validate_candidate_state_replay_proof_artifact",
        lambda path, **_kwargs: pointers[path],
    )
    monkeypatch.setattr(
        TtsL0CandidateStateCoverage,
        "validate_native_replay_pointers",
        lambda self, rows: replayed.append(rows),
    )
    kwargs = {
        "candidate_coverages": (candidate,),
        "protocol_lock": lock,
        "materialization": materialization,
        "inventory": inventory,
        "candidate_replay_proof_artifact_paths": (
            "/proof/tts.json",
            "/proof/l0.json",
        ),
        "terminal_by_cell": {
            tts.cell_id: tts_terminal,
            l0.cell_id: l0_terminal,
        },
        "now_ns": 10,
    }
    coverage_authority._validate_candidate_coverages(**kwargs)
    assert len(replayed) == 1
    with pytest.raises(ValueError, match="proof set is not exact"):
        coverage_authority._validate_candidate_coverages(
            **{
                **kwargs,
                "candidate_replay_proof_artifact_paths": ("/proof/tts.json",),
            }
        )
    with pytest.raises(ValueError, match="terminal differs"):
        coverage_authority._validate_candidate_coverages(
            **{
                **kwargs,
                "terminal_by_cell": {
                    tts.cell_id: _sha("wrong-terminal"),
                    l0.cell_id: l0_terminal,
                },
            }
        )


def test_durable_coverage_artifact_points_to_derived_shards_and_is_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, runtime, inventory, materialization, evidence, bindings = _e4_bundle(
        tmp_path, monkeypatch
    )
    coverage = reduce_formal_serving_stage_coverage_from_proofs(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime,
        materialization=materialization,
        inventory=inventory,
        stage="E4",
        phase="screen",
        evidence_cells=evidence,
        execution_bindings=bindings,
        now_ns=10,
    )
    derived = derived_coverage_shards(
        coverage, phase="screen", maximum_dispositions_per_shard=24
    )
    derived_sources = tuple(
        publish_formal_stage_derived_coverage_shard(
            shard, (tmp_path / f"derived-{index}.json").resolve()
        )
        for index, shard in enumerate(derived)
    )
    protocol_source = _publish(
        (tmp_path / "protocol.json").resolve(), protocol_lock_to_dict(lock)
    )
    runtime_source = _publish(
        (tmp_path / "runtime.json").resolve(),
        formal_runtime_authority_manifest_to_dict(runtime),
    )
    materialization_source = _publish(
        (tmp_path / "materialization.json").resolve(),
        stage_materialization_receipt_to_dict(materialization),
    )
    inventory_source = _publish(
        (tmp_path / "inventory.json").resolve(), inventory.to_dict()
    )
    evidence_source = _publish(
        (tmp_path / "evidence-placeholder.json").resolve(), {"evidence": True}
    )
    rebuild_source = _publish(
        (tmp_path / "rebuild-placeholder.json").resolve(), {"rebuild": True}
    )
    artifact = FormalStageCoverageProofArtifact(
        schema_version=1,
        kind="formal_stage_coverage_proof_artifact",
        protocol_sha256=FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
        stage="E4",
        phase="screen",
        protocol_lock_sha256=lock.sha256,
        formal_runtime_authority_manifest_sha256=runtime.sha256,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        coverage_receipt_sha256=coverage.sha256,
        protocol_lock_source=protocol_source,
        runtime_authority_source=runtime_source,
        materialization_source=materialization_source,
        inventory_source=inventory_source,
        tts_authority_source=None,
        raw_tts_evidence_source=None,
        stage_source_rebuild_input_source=None,
        evidence_shard_sources=(evidence_source,),
        execution_rebuild_shard_sources=(rebuild_source,),
        candidate_replay_proof_sources=(),
        derived_coverage_shard_sources=derived_sources,
    )
    output = (tmp_path / "coverage-proof.json").resolve()
    binding = publish_formal_stage_coverage_proof_artifact(artifact, output)
    assert FormalStageCoverageProofArtifact.from_dict(binding.reopen()) == artifact
    signed = SignedStageCoverageReceipt(
        coverage,
        *_signature_parts(coverage, key_id="coverage-proof-wrapper"),
    )
    wrapper_binding = publish_formal_signed_stage_coverage_proof_wrapper(
        signed,
        coverage_proof_source=binding,
        output_path=(tmp_path / "signed-coverage-proof-wrapper.json").resolve(),
    )
    wrapper = FormalSignedStageCoverageProofWrapper.from_dict(wrapper_binding.reopen())
    assert wrapper.coverage_receipt_sha256 == coverage.sha256
    assert wrapper.signed_coverage_receipt_sha256 == signed.sha256
    assert wrapper_binding.size < 2 * 1024 * 1024
    with pytest.raises(RuntimeError, match="already exists"):
        publish_formal_stage_coverage_proof_artifact(artifact, output)


def _tts_raw_manifest(tmp_path: Path, authority: TtsCalibrationAuthority):
    registry_cells = build_industrial_registry().cells_for("TTS-Cal")
    pilots = []
    for block in PILOT_BLOCKS:
        rows = tuple(row for row in registry_cells if row.identity.block == block)
        cells = tuple(
            IndustrialCellEvidence(
                cell_id=row.cell_id,
                terminal_receipts=(
                    BoundArtifact(
                        (tmp_path / f"terminal-{row.cell_id}.json").resolve(),
                        _sha(f"terminal-{row.cell_id}"),
                    ),
                ),
                hardware_receipt=BoundArtifact(
                    (tmp_path / f"hardware-{row.cell_id}.json").resolve(),
                    _sha(f"hardware-{row.cell_id}"),
                ),
                budget_observation=BoundArtifact(
                    (tmp_path / f"budget-{row.cell_id}.json").resolve(),
                    _sha(f"budget-{row.cell_id}"),
                ),
            )
            for row in rows
        )
        controls = tuple(
            BoundArtifact(
                (tmp_path / f"control-{row.cell_id}.json").resolve(),
                _sha(f"control-{row.cell_id}"),
            )
            for row in rows
        )
        pilots.append(
            TtsCalibrationPilotEvidence(
                block=block,
                qualification_lock=BoundArtifact(
                    (tmp_path / f"qualification-{block}.json").resolve(),
                    _sha(f"qualification-{block}"),
                ),
                cells=cells,
                terminal_control_attestations=controls,
            )
        )
    return RawTtsCalibrationEvidenceManifest(
        schema_version=2,
        tuning_window=BoundArtifact(
            (tmp_path / "tuning-window.json").resolve(),
            authority.tuning_window_sha256,
        ),
        pilots=tuple(pilots),
    )


def test_tts_coverage_requires_exact_current_request_reset_receipts() -> None:
    protocol = coverage_authority.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
    request = SimpleNamespace(request_id="scored-0", submitted_to_server=True)
    second_request = SimpleNamespace(request_id="scored-1", submitted_to_server=True)
    client_only = SimpleNamespace(request_id="scored-client", submitted_to_server=False)
    warmup_resets = SimpleNamespace(
        reset_scope="request",
        request_admission_policy="serialized_native_scheduler_v1",
        protocol_sha256=protocol,
        receipts=(),
    )
    scored_resets = SimpleNamespace(
        reset_scope="request",
        request_admission_policy="serialized_native_scheduler_v1",
        protocol_sha256=protocol,
        # Native epoch/terminal order is authoritative and may differ from
        # caller submission order.
        receipts=(
            SimpleNamespace(request_id=second_request.request_id),
            SimpleNamespace(request_id=request.request_id),
        ),
    )
    evidence = SimpleNamespace(
        terminal_schema_version=2,
        binding=SimpleNamespace(
            method="tts",
            reset_scope="request",
            request_admission_policy="serialized_native_scheduler_v1",
        ),
        reset_receipt=SimpleNamespace(
            request_source_point_reset_protocol_sha256=protocol,
            warmup_request_source_point_resets=warmup_resets,
            warmup_requests=(),
        ),
        request_source_point_resets=scored_resets,
        requests=(request, second_request, client_only),
    )
    coverage_authority._require_tts_calibration_request_reset_evidence(evidence)

    missing = SimpleNamespace(
        **{
            **evidence.__dict__,
            "request_source_point_resets": SimpleNamespace(
                **{**scored_resets.__dict__, "receipts": ()}
            ),
        }
    )
    with pytest.raises(ValueError, match="scored reset coverage"):
        coverage_authority._require_tts_calibration_request_reset_evidence(missing)


def test_tts_coverage_derives_288_terminals_from_nonconsuming_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    runtime = _runtime_manifest()
    lock = _protocol_lock(authority, runtime)
    inventory = _inventory()
    materialization = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_e3a_receipt_sha256=_sha("e3a-selection"),
        calibration_authority_sha256=authority.sha256,
        gpu_hours=_unmeasured(),
    )
    manifest = _tts_raw_manifest(tmp_path, authority)
    qualification_blocks: list[int] = []
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage."
        "validate_raw_evidence_manifest_sidecars",
        lambda value: None,
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage."
        "_validate_tts_qualification_lock",
        lambda _reference, *, block, **_kwargs: qualification_blocks.append(block),
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage._load_bound_json",
        lambda reference, *, label: {
            "path": str(reference.path),
            "sha256": reference.sha256,
            "label": label,
        },
    )

    class FakeControl:
        def __init__(self, cell_id: str) -> None:
            self.cell_id = cell_id
            self.deployment_policy_authorization = SimpleNamespace(
                root_manifest_sha256=lock.offline_release_trust_root_sha256
            )

    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage.ControlArtifactAttestation",
        SimpleNamespace(
            from_dict=lambda value: FakeControl(
                Path(value["path"]).stem.removeprefix("control-")
            )
        ),
    )

    def prepare(value, **_kwargs):
        cell_id = Path(value["path"]).stem.removeprefix("terminal-")
        request_id = f"scored-{cell_id}"
        warmup_resets = SimpleNamespace(
            reset_scope="request",
            request_admission_policy="serialized_native_scheduler_v1",
            protocol_sha256=(
                coverage_authority.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
            ),
            receipts=(),
        )
        scored_resets = SimpleNamespace(
            reset_scope="request",
            request_admission_policy="serialized_native_scheduler_v1",
            protocol_sha256=(
                coverage_authority.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
            ),
            receipts=(SimpleNamespace(request_id=request_id),),
        )
        return SimpleNamespace(
            binding=SimpleNamespace(
                canonical_raw_sha256=value["sha256"],
                sha256=_sha(f"control-binding-{cell_id}"),
            ),
            evidence=SimpleNamespace(
                terminal_schema_version=2,
                binding=SimpleNamespace(
                    method="tts",
                    reset_scope="request",
                    request_admission_policy="serialized_native_scheduler_v1",
                    run_id=f"tts-{cell_id}",
                    run_nonce_sha256=_sha(f"run-nonce-{cell_id}"),
                ),
                reset_receipt=SimpleNamespace(
                    request_source_point_reset_protocol_sha256=(
                        coverage_authority.REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
                    ),
                    warmup_request_source_point_resets=warmup_resets,
                    warmup_requests=(),
                ),
                request_source_point_resets=scored_resets,
                requests=(
                    SimpleNamespace(
                        request_id=request_id,
                        submitted_to_server=True,
                    ),
                ),
            ),
        )

    def verify(control, **_kwargs):
        return SimpleNamespace(
            artifact_sha256=_sha(f"control-binding-{control.cell_id}"),
            challenge_sha256=_sha(f"challenge-{control.cell_id}"),
            deployment_policy_challenge_sha256=_sha(f"deployment-{control.cell_id}"),
        )

    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage."
        "prepare_native_terminal_external_control",
        prepare,
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_stage_coverage."
        "verify_release_control_artifact_attestation",
        verify,
    )
    receipt = reduce_tts_calibration_stage_coverage_from_proofs(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime,
        materialization=materialization,
        inventory=inventory,
        authority=authority,
        manifest=manifest,
        now_ns=10,
    )
    assert len(receipt.dispositions) == 288
    assert qualification_blocks == [0, 1, 2, 3]
    qualification_blocks.clear()
    assert (
        receipt.sha256
        == reduce_tts_calibration_stage_coverage_from_proofs(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            materialization=materialization,
            inventory=inventory,
            authority=authority,
            manifest=manifest,
            now_ns=10,
        ).sha256
    )
