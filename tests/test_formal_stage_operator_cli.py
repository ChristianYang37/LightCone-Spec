from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import INVENTORY_SHA256, _bundle, _root_binding
from test_e0_authority_artifact import (
    _binding,
    _node,
    _signed_confirmation,
    _signed_e1a_verification,
    _signed_e3b_confirmation,
    _signed_e5_confirmation,
    _signed_model_compatibility,
)
from test_e0_stage_authority import (
    _compatibility,
    _sign_power_prefix,
    _signed_tuning_seals,
)
from test_e6_stage_authority import _source
from test_formal_cli_replay import _control as _registry_control
from test_formal_cli_replay import _deployment_authorization
from test_formal_cli_replay import _lock as _registry_lock
from test_formal_cli_replay import _sign as _registry_sign
from test_formal_registry_power_sources import _power_sizing, _signature_parts

import lightcone_spec.cli.main as cli_module
from lightcone_spec.cli.main import (
    _extend_formal_registry_verification,
    _formal_stage_operation,
    _load_bound_json,
    _parser,
    _write_json,
    main,
)
from lightcone_spec.experiments import e0_stage_authority
from lightcone_spec.experiments.e0_authority_artifact import (
    E0FormalRegistryAuthorityArtifact,
    E6RecursiveSourceDagArtifact,
)
from lightcone_spec.experiments.e0_stage_authority import (
    E0OnlineSpecSourceAuthority,
    E0PowerPrefixReceipt,
)
from lightcone_spec.experiments.e2_stage_authority import (
    E2StagedCandidateEvaluation,
    E2StagedRoundSelectionReceipt,
    SignedE2StagedRoundSelectionReceipt,
)
from lightcone_spec.experiments.formal_protocol import (
    E6_MODELS,
    SignedProtocolLock,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    reserve_formal_registry_verification_receipt,
    signed_stage_materialization_to_dict,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FormalStageSourceRebuildInput,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_COMMIT,
    ONLINE_SPEC_SOURCE_AUDIT_SHA256,
    ONLINE_SPEC_TREE,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    SignedE0CompatibilityReceipt,
    SignedStageMaterializationReceipt,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
    e1_geometries,
    e2_candidate_recipes,
)
from lightcone_spec.runtime import release_trust_root as root_module
from lightcone_spec.runtime.control_attestation import ChallengeReplayStore
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return content_sha256({"formal-stage-operator-test": label})


def _real_registry_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(signer_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    lock = _registry_lock(root_binding.semantic_sha256)
    signed_lock = SignedProtocolLock(
        lock,
        *_registry_sign(signer_private, lock, nonce_byte=b"a"),
    )
    lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": ((signed_lock.sha256, "dispatch"),),
        }
    )
    authorization = _deployment_authorization(
        root_private,
        binding=root_binding,
        bundle=bundle,
        nonce_byte=b"b",
    )
    control = _registry_control(
        signer_private,
        binding=root_binding,
        bundle=bundle,
        authorization=authorization,
        artifact_type="dispatch",
        artifact_sha256=signed_lock.sha256,
        protocol_sha256=lock.sha256,
        lineage_sha256=lineage,
        nonce_byte=b"c",
    )
    replay = tmp_path / "root-replay"
    replay.mkdir()
    receipt = reserve_formal_registry_verification_receipt(
        signed_lock,
        control_attestation=control,
        expected_inventory_sha256=INVENTORY_SHA256,
        replay_store=ChallengeReplayStore(str(replay.resolve())),
        now_ns=2_000_000_000,
    )
    return receipt, tmp_path / "unused-prior-root-layer.json"


def _stage_source_descriptor() -> FormalStageSourceRebuildInput:
    return FormalStageSourceRebuildInput(
        schema_version=1,
        kind="formal_stage_source_rebuild_input",
        stage="E4",
        phase="screen",
        materialization_receipt_sha256=_sha("materialization"),
        source_decision_sha256=_sha("source-decision"),
        registry_verification_receipt_sha256=_sha("registry-receipt"),
        source_input_commitment_sha256=_sha("source-input"),
        expected_stage_source_sha256=_sha("expected-source"),
    )


def test_closed_stage_source_publisher_is_typed_path_bound_and_no_replace(
    tmp_path: Path,
) -> None:
    descriptor = _stage_source_descriptor()
    source = tmp_path / "stage-source-input.json"
    output = tmp_path / "stage-source.json"
    _write_json(source, descriptor.to_dict())

    assert (
        main(
            [
                "publish-formal-rebuild-artifact",
                "--artifact-kind",
                "stage-source",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    binding = CanonicalJsonProofBinding.bind(str(output.resolve()))
    assert FormalStageSourceRebuildInput.from_dict(binding.reopen()) == descriptor

    with pytest.raises(RuntimeError, match="target already exists"):
        main(
            [
                "publish-formal-rebuild-artifact",
                "--artifact-kind",
                "stage-source",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )


def test_closed_publisher_rejects_digest_only_or_legacy_shape(tmp_path: Path) -> None:
    value = _stage_source_descriptor().to_dict()
    value["digest_only_fallback"] = _sha("digest-only")
    source = tmp_path / "legacy-stage-source.json"
    _write_json(source, value)

    with pytest.raises(ValueError, match="fields differ"):
        main(
            [
                "publish-formal-rebuild-artifact",
                "--artifact-kind",
                "stage-source",
                "--input",
                str(source),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )


def test_stage_dispatch_calls_source_owned_e3a_materializer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preflight_cell = MaterializedCell(
        stage="preflight",
        method_role="Target-only",
        model="Qwen/Qwen3.6-35B-A3B",
        backend="dflash",
        task="compile_preflight",
        publication_policy="fixed_barrier",
        recipe_sha256=None,
        dimensions=(),
    )
    preflight = StageMaterializationReceipt(
        schema_version=1,
        stage="preflight",
        protocol_lock_sha256=_sha("operator-lock"),
        upstream_receipt_sha256s=(),
        source_decision_sha256=_sha("preflight-source"),
        materialization_rule="typed-test-preflight",
        expected_cell_count=1,
        cells=(preflight_cell,),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    coverage = SimpleNamespace(
        stage="preflight",
        materialization_receipt_sha256=preflight.sha256,
        sha256=_sha("preflight-coverage"),
    )
    signed_preflight = SignedStageMaterializationReceipt(
        preflight,
        *_signature_parts(preflight, key_id="preflight-materialization"),
    )
    signed_coverage = SimpleNamespace(
        payload=coverage,
        sha256=_sha("signed-preflight-coverage"),
    )
    receipt = SimpleNamespace(
        sha256=_sha("preflight-registry"),
        signed_protocol_lock=SimpleNamespace(payload=SimpleNamespace()),
        cumulative_signed_materializations=(signed_preflight,),
        cumulative_signed_coverage=(signed_coverage,),
        revalidate=lambda **_kwargs: SimpleNamespace(status="COVERED"),
    )
    monkeypatch.setattr(
        cli_module,
        "_load_formal_registry_receipt_path",
        lambda _path, *, now_ns: receipt,
    )
    created = StageMaterializationReceipt(
        schema_version=1,
        stage="E3a",
        protocol_lock_sha256=preflight.protocol_lock_sha256,
        upstream_receipt_sha256s=(signed_coverage.sha256,),
        source_decision_sha256=_sha("e3a-source"),
        materialization_rule="source-owned-test-e3a",
        expected_cell_count=1,
        cells=(
            MaterializedCell(
                stage="E3a",
                method_role="Target-only",
                model="Qwen/Qwen3.6-35B-A3B",
                backend="dflash",
                task="capacity_probe",
                publication_policy="fixed_barrier",
                recipe_sha256=None,
                dimensions=(),
            ),
        ),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    captured = {}

    def materialize(**kwargs):
        captured.update(kwargs)
        return created

    monkeypatch.setattr(cli_module, "materialize_e3a", materialize)
    registry_path = tmp_path / "registry.json"
    output = tmp_path / "e3a-materialization.json"
    _write_json(registry_path, {"kind": "typed-test-registry"})
    assert (
        _formal_stage_operation(
            argparse.Namespace(
                operation="materialize",
                stage="E3a",
                phase="selection",
                registry_verification_receipt=str(registry_path),
                e0_authority_bundle=None,
                e0_materialization=None,
                result_rebuild_artifact=None,
                signed_stage_result=None,
                tts_calibration_authority=None,
                now_ns=2_000_000_001,
                output=str(output),
            )
        )
        == 0
    )
    assert captured["preflight_materialization"] == preflight
    assert captured["preflight_coverage"] == coverage
    result = _load_bound_json(output)
    assert result["status"] == "PROOF_MATERIALIZED"
    assert result["artifacts"][0]["artifact_sha256"] == created.sha256


def _e2_fixture():
    recipe = e2_candidate_recipes(
        (e1_geometries()[0],), grid=default_e2_recipe_grid_authority()
    )[0]
    cell = MaterializedCell(
        stage="E2",
        method_role="LightCone-candidate",
        model="meta-llama/Llama-3.1-8B-Instruct",
        backend="dflash",
        task="staged_successive_halving",
        publication_policy="first_ready",
        recipe_sha256=recipe.sha256,
        dimensions=(("round", 3),),
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E2",
        protocol_lock_sha256=_sha("protocol-lock"),
        upstream_receipt_sha256s=(_sha("e2-round-two"),),
        source_decision_sha256=_sha("round-two-selection"),
        materialization_rule="e2_quarter_retention_floor_21_plus_four_anchors",
        expected_cell_count=1,
        cells=(cell,),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    payload_sha256, challenge, attestation = _signature_parts(
        materialization, key_id="operator-e2-materialization"
    )
    signed_materialization = SignedStageMaterializationReceipt(
        payload=materialization,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )
    evaluation = E2StagedCandidateEvaluation(
        recipe=recipe,
        cell_id=cell.cell_id,
        confidence_lower_request_rate_ratio=1.1,
        peak_hbm_bytes=1,
        p99_itl_us=1,
        exposed_update_us=1,
        launched_updates=1,
        published_updates=1,
    )
    selection = E2StagedRoundSelectionReceipt(
        schema_version=1,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        registry_sha256=_sha("registry"),
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=_sha("coverage"),
        source_selection_sha256=materialization.source_decision_sha256,
        evidence_manifest_sha256=_sha("evidence"),
        inventory_sha256=_sha("inventory"),
        round_index=3,
        source_candidate_count=1,
        evaluations=(evaluation,),
        survivor_recipes=(recipe,),
        final_recipe=recipe,
    )
    payload_sha256, challenge, attestation = _signature_parts(
        selection, key_id="operator-e2-selection"
    )
    signed_selection = SignedE2StagedRoundSelectionReceipt(
        payload=selection,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )
    return signed_materialization, signed_selection


@pytest.mark.parametrize("operation", ("materialize", "reduce", "sign"))
def test_stage_dispatch_rejects_registry_only_e2_replay_as_non_operator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
) -> None:
    signed_materialization, signed_selection = _e2_fixture()
    receipt = SimpleNamespace(
        sha256=_sha("registry-receipt"),
        cumulative_signed_materializations=(signed_materialization,),
        cumulative_signed_e2_staged_selections=(signed_selection,),
        revalidate=lambda **_kwargs: SimpleNamespace(status="COVERED"),
    )
    monkeypatch.setattr(
        cli_module,
        "_load_formal_registry_receipt_path",
        lambda _path, *, now_ns: receipt,
    )
    registry_source = tmp_path / "registry.json"
    output = tmp_path / f"e2-{operation}.json"
    _write_json(registry_source, {"schema_version": 1, "kind": "typed-fixture"})

    with pytest.raises(ValueError, match="NON_OPERATOR_BLOCKED"):
        _formal_stage_operation(
            argparse.Namespace(
                operation=operation,
                stage="E2",
                phase="round3",
                registry_verification_receipt=str(registry_source),
                e0_authority_bundle=None,
                e0_materialization=None,
                now_ns=30_000_000_001,
                output=str(output),
            )
        )
    assert not output.exists()


def _recursive_dag(tmp_path: Path) -> E6RecursiveSourceDagArtifact:
    source_root = tmp_path / "dag-sources"
    source_root.mkdir()
    nodes = tuple(
        _node(source_root, node_id)
        for node_id in (
            "e2_final",
            "e4_screen",
            "e4_local",
            "e4_profiler",
            "e3b_pilot",
            "e3b_final",
            "e1a",
            "e5_pilot",
            "e5_final",
            "e6_pilot",
        )
    )
    return E6RecursiveSourceDagArtifact(
        schema_version=1,
        kind="lightcone_e6_recursive_source_dag_artifact",
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_verification_receipt_sha256=_sha("prior-registry"),
        signed_e3b_confirmation=_signed_e3b_confirmation(),
        signed_e1a_verification=_signed_e1a_verification(),
        signed_e5_confirmation=_signed_e5_confirmation(),
        nodes=nodes,
    )


def _source_authority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkout = (tmp_path / "onlinespec").resolve()
    checkout.mkdir()
    audit = (tmp_path / "onlinespec-audit.json").resolve()
    audit.write_text("{}\n", encoding="utf-8")
    verified = {
        "commit": ONLINE_SPEC_COMMIT,
        "tree": ONLINE_SPEC_TREE,
        "key_files": {"optimizer.py": _sha("optimizer")},
    }
    monkeypatch.setattr(
        e0_stage_authority,
        "verify_onlinespec_source_checkout",
        lambda *_args, **_kwargs: verified,
    )
    return E0OnlineSpecSourceAuthority(
        schema_version=1,
        checkout_path=str(checkout),
        audit_path=str(audit),
        source_audit_sha256=ONLINE_SPEC_SOURCE_AUDIT_SHA256,
        commit=ONLINE_SPEC_COMMIT,
        tree=ONLINE_SPEC_TREE,
        verification_sha256=content_sha256(verified),
    )


def _typed_e2_to_e0_aggregate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    main_materialization_sha256: str,
) -> E0FormalRegistryAuthorityArtifact:
    dag = _recursive_dag(tmp_path)
    dag_input = tmp_path / "e6-dag-input.json"
    dag_output = tmp_path / "e6-dag.json"
    _write_json(dag_input, dag.to_dict())
    assert (
        main(
            [
                "publish-formal-rebuild-artifact",
                "--artifact-kind",
                "e6-recursive-dag",
                "--input",
                str(dag_input),
                "--output",
                str(dag_output),
            ]
        )
        == 0
    )

    compatibility = _compatibility(valid_count=1)
    payload_sha256, challenge, attestation = _signature_parts(
        compatibility, key_id="operator-e0-compatibility"
    )
    signed_compatibility = SignedE0CompatibilityReceipt(
        payload=compatibility,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )
    tuning_seals = _signed_tuning_seals(compatibility)
    power = _sign_power_prefix(
        E0PowerPrefixReceipt(
            schema_version=1,
            protocol_lock_sha256=compatibility.protocol_lock_sha256,
            registry_sha256=_sha("registry"),
            upstream_e6_confirmation_sha256=_sha("e6-confirmation"),
            signed_compatibility_sha256=signed_compatibility.sha256,
            signed_tuning_seal_sha256s=tuple(
                sorted(row.sha256 for row in tuning_seals)
            ),
            pilot_materialization_receipt_sha256=_sha("e0-pilot-materialization"),
            pilot_coverage_receipt_sha256=_sha("e0-pilot-coverage"),
            evidence_manifest_sha256=_sha("e0-pilot-evidence"),
            inventory_sha256=_sha("inventory"),
            power_sizing=_power_sizing(),
            selected_final_blocks=12,
            selected_final_prefix=tuple(range(4, 16)),
        )
    )
    compatibility_dir = tmp_path / "e6-compatibility"
    compatibility_dir.mkdir()
    signed_e6_compatibility = _signed_model_compatibility(compatibility_dir)
    e6_sources_dir = tmp_path / "e6-source-inputs"
    e6_sources_dir.mkdir()
    e6_sources = tuple(
        _source(e6_sources_dir, model, index) for index, model in enumerate(E6_MODELS)
    )
    binding_root = tmp_path / "aggregate-sources"
    binding_root.mkdir()

    def bound(label: str) -> CanonicalJsonProofBinding:
        return _binding(binding_root, label)

    return E0FormalRegistryAuthorityArtifact(
        schema_version=1,
        kind="lightcone_e0_formal_registry_authority_bundle_artifact",
        protocol_lock_sha256=dag.protocol_lock_sha256,
        prior_registry_verification_receipt_sha256=(
            dag.registry_verification_receipt_sha256
        ),
        main_materialization_receipt_sha256=main_materialization_sha256,
        formal_runtime_authority_manifest_source=bound("runtime-manifest"),
        inventory_source=bound("inventory"),
        signed_e6_confirmation=_signed_confirmation(signed_e6_compatibility),
        signed_e6_model_compatibility=signed_e6_compatibility,
        e6_compatibility_sources=e6_sources,
        signed_e0_compatibility=signed_compatibility,
        onlinespec_source_authority=_source_authority(
            monkeypatch, tmp_path / "source-authority"
        ),
        signed_e0_tuning_seals=tuning_seals,
        signed_e0_power_prefix=power,
        e6_recursive_source_dag_source=CanonicalJsonProofBinding.bind(
            str(dag_output.resolve())
        ),
        e6_materialization_source=bound("e6-materialization"),
        e6_coverage_source=bound("e6-coverage"),
        e6_evidence_manifest_source=bound("e6-evidence"),
        e6_stage_source_rebuild_source=bound("e6-stage-source"),
        e6_execution_rebuild_shards=(bound("e6-shard"),),
        e0_tuning_materialization_source=bound("e0-tuning-materialization"),
        e0_tuning_coverage_source=bound("e0-tuning-coverage"),
        e0_tuning_evidence_manifest_source=bound("e0-tuning-evidence"),
        e0_tuning_stage_source_rebuild_source=bound("e0-tuning-stage-source"),
        e0_tuning_execution_rebuild_shards=(bound("e0-tuning-shard"),),
        e0_pilot_materialization_source=bound("e0-pilot-materialization"),
        e0_pilot_coverage_source=bound("e0-pilot-coverage"),
        e0_pilot_evidence_manifest_source=bound("e0-pilot-evidence"),
        e0_pilot_stage_source_rebuild_source=bound("e0-pilot-stage-source"),
        e0_pilot_execution_rebuild_shards=(bound("e0-pilot-shard"),),
    )


def test_future_e0_aggregate_cannot_bypass_typed_materialization_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E0",
        protocol_lock_sha256=_sha("protocol-lock"),
        upstream_receipt_sha256s=(_sha("e6-final"),),
        source_decision_sha256=_sha("e0-power"),
        materialization_rule="all_proof_backed_combinations_are_na",
        expected_cell_count=0,
        cells=(),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    payload_sha256, challenge, attestation = _signature_parts(
        materialization, key_id="operator-e0-materialization"
    )
    signed_materialization = SignedStageMaterializationReceipt(
        payload=materialization,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=attestation,
    )
    aggregate = _typed_e2_to_e0_aggregate(
        monkeypatch,
        tmp_path,
        main_materialization_sha256=materialization.sha256,
    )
    aggregate_input = tmp_path / "aggregate-input.json"
    aggregate_output = tmp_path / "aggregate.json"
    _write_json(aggregate_input, aggregate.to_dict())
    assert (
        main(
            [
                "publish-formal-rebuild-artifact",
                "--artifact-kind",
                "e0-aggregate",
                "--input",
                str(aggregate_input),
                "--output",
                str(aggregate_output),
            ]
        )
        == 0
    )

    prior, prior_path = _real_registry_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_load_formal_registry_receipt_path",
        lambda *_args, **_kwargs: prior,
    )
    bundle_marker = object()
    captured: dict[str, object] = {}

    def load_bundle(path, *, registry_verification_receipt, materialization, now_ns):
        del now_ns
        typed = E0FormalRegistryAuthorityArtifact.from_dict(
            CanonicalJsonProofBinding.bind(str(Path(path).resolve())).reopen()
        )
        assert typed.sha256 == aggregate.sha256
        assert registry_verification_receipt == prior
        assert materialization.sha256 == typed.main_materialization_receipt_sha256
        return bundle_marker

    result_receipt = prior

    def extend(prior_receipt, **kwargs):
        captured.update(kwargs)
        assert prior_receipt == prior
        return result_receipt

    monkeypatch.setattr(
        cli_module, "load_e0_formal_registry_authority_bundle", load_bundle
    )
    monkeypatch.setattr(
        cli_module, "extend_formal_registry_verification_receipt", extend
    )
    layer_marker = object()
    monkeypatch.setattr(
        cli_module,
        "bind_formal_registry_layer_artifact",
        lambda receipt, **_kwargs: layer_marker,
    )
    monkeypatch.setattr(
        cli_module,
        "publish_formal_registry_layer_artifact",
        lambda artifact, path: _write_json(
            path,
            {
                "schema_version": 1,
                "kind": "typed_registry_extend_fixture",
                "receipt_sha256": result_receipt.sha256,
            },
        ),
    )
    materialization_path = tmp_path / "signed-e0-materialization.json"
    publish_canonical_json_no_replace(
        materialization_path,
        signed_stage_materialization_to_dict(signed_materialization),
    )
    output = tmp_path / "extended.json"
    with pytest.raises(ValueError, match="typed predecessor reducer proof"):
        _extend_formal_registry_verification(
            argparse.Namespace(
                prior_receipt=str(prior_path),
                signed_materialization=[str(materialization_path)],
                signed_coverage=[],
                tts_calibration_authority=[],
                signed_tts_calibration_seal=[],
                signed_e3b_power_prefix=[],
                signed_e5_power_and_anchor_prefix=[],
                signed_e6_power_prefix=[],
                e0_authority_bundle=[str(aggregate_output)],
                control_attestation=[],
                candidate_state_replay_proof_artifact=[],
                control_replay_store=str(tmp_path / "replay.sqlite"),
                now_ns=2_000_000_001,
                output=str(output),
            )
        )
    assert not captured
    assert not output.exists()


def test_stage_dispatch_rejects_unregistered_phase_before_io() -> None:
    args = _parser().parse_args(
        [
            "formal-stage-operation",
            "--operation",
            "reduce",
            "--stage",
            "E0",
            "--phase",
            "legacy_digest",
            "--registry-verification-receipt",
            "missing.json",
            "--now-ns",
            "1",
            "--output",
            "unused.json",
        ]
    )
    with pytest.raises(ValueError, match="phase is unsupported"):
        _formal_stage_operation(args)


def test_initial_materialization_proof_cli_maps_the_immediate_prefix_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = SimpleNamespace(sha256=_sha("initial-materialization-proof"))
    captured: list[tuple[str, object]] = []

    def bind(**kwargs: object) -> object:
        captured.append(("bind", dict(kwargs)))
        return artifact

    def publish(value: object, path: str) -> None:
        captured.append(("publish", (value, path)))

    monkeypatch.setattr(
        cli_module,
        "bind_formal_initial_stage_materialization_proof_artifact",
        bind,
    )
    monkeypatch.setattr(
        cli_module,
        "publish_formal_initial_stage_materialization_proof_artifact",
        publish,
    )
    registry = str((tmp_path / "root-layer.json").resolve())
    authority = str((tmp_path / "tts-authority.json").resolve())
    output = str((tmp_path / "initial-proof.json").resolve())
    argv = [
        "publish-formal-initial-stage-materialization-proof",
        "--phase",
        "tts_calibration",
        "--registry-layer",
        registry,
        "--tts-calibration-authority",
        authority,
        "--now-ns",
        "2000000000",
        "--output",
        output,
    ]
    assert main(argv) == 0
    assert captured == [
        (
            "bind",
            {
                "phase": "tts_calibration",
                "registry_layer_path": registry,
                "tts_calibration_authority_path": authority,
                "now_ns": 2_000_000_000,
            },
        ),
        ("publish", (artifact, output)),
    ]
    with pytest.raises(SystemExit):
        main([*argv, "--materialization", "caller.json"])


def test_protocol_lock_source_proof_cli_has_no_caller_digest_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_binding = SimpleNamespace(semantic_sha256=_sha("git-snapshot"))
    source_artifact = SimpleNamespace(verified_ns=2_000_000_000)
    source_binding = SimpleNamespace(
        absolute_path=str((tmp_path / "lock-proof.json").resolve())
    )
    lock = SimpleNamespace(sha256=_sha("protocol-lock"))
    captured: list[tuple[str, object]] = []

    monkeypatch.setattr(
        cli_module,
        "publish_formal_protocol_lock_git_snapshot",
        lambda **kwargs: captured.append(("snapshot", kwargs)) or snapshot_binding,
    )
    monkeypatch.setattr(
        cli_module,
        "bind_formal_protocol_lock_source_proof_artifact",
        lambda **kwargs: captured.append(("bind", kwargs)) or source_artifact,
    )
    monkeypatch.setattr(
        cli_module,
        "publish_formal_protocol_lock_source_proof_artifact",
        lambda artifact, path: (
            captured.append(("publish", (artifact, path))) or source_binding
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "revalidate_formal_protocol_lock_source_proof_artifact",
        lambda path, *, now_ns: captured.append(("revalidate", (path, now_ns))) or lock,
    )

    repository = str((tmp_path / "repo").resolve())
    chunks = str((tmp_path / "chunks").resolve())
    snapshot = str((tmp_path / "git-snapshot.json").resolve())
    assert (
        main(
            [
                "publish-formal-protocol-lock-git-snapshot",
                "--project-root",
                repository,
                "--chunk-output-directory",
                chunks,
                "--output",
                snapshot,
            ]
        )
        == 0
    )
    assert captured.pop(0) == (
        "snapshot",
        {
            "project_root": Path(repository),
            "chunk_output_directory": Path(chunks),
            "index_output_path": Path(snapshot),
        },
    )

    sources = {
        "git_snapshot": snapshot,
        "runtime_authority": str((tmp_path / "runtime.json").resolve()),
        "tts_calibration_authority": str((tmp_path / "tts.json").resolve()),
        "chronobelief_authority": str((tmp_path / "chrono.json").resolve()),
        "e1_recipe_anchor_authority": str((tmp_path / "e1.json").resolve()),
        "content_verification_receipt": str((tmp_path / "content.json").resolve()),
        "burstgpt_shape_authority": str((tmp_path / "burst.json").resolve()),
    }
    output = str((tmp_path / "lock-proof.json").resolve())
    argv = [
        "publish-formal-protocol-lock-source-proof",
        "--protocol-id",
        "formal-test-v1",
        "--git-snapshot",
        sources["git_snapshot"],
        "--patch-manifest-relative-path",
        "patch.json",
        "--english-protocol-relative-path",
        "protocol-en.md",
        "--chinese-protocol-relative-path",
        "protocol-zh.md",
        "--formal-runtime-authority-manifest",
        sources["runtime_authority"],
        "--tts-calibration-authority",
        sources["tts_calibration_authority"],
        "--chronobelief-authority",
        sources["chronobelief_authority"],
        "--e1-recipe-anchor-authority",
        sources["e1_recipe_anchor_authority"],
        "--content-verification-receipt",
        sources["content_verification_receipt"],
        "--burstgpt-shape-authority",
        sources["burstgpt_shape_authority"],
        "--now-ns",
        "2000000000",
        "--output",
        output,
    ]
    assert main(argv) == 0
    assert captured[0] == (
        "bind",
        {
            "protocol_id": "formal-test-v1",
            "git_snapshot_path": Path(sources["git_snapshot"]),
            "patch_manifest_relative_path": "patch.json",
            "english_protocol_relative_path": "protocol-en.md",
            "chinese_protocol_relative_path": "protocol-zh.md",
            "runtime_authority_path": Path(sources["runtime_authority"]),
            "tts_calibration_authority_path": Path(
                sources["tts_calibration_authority"]
            ),
            "chronobelief_authority_path": Path(sources["chronobelief_authority"]),
            "e1_recipe_anchor_authority_path": Path(
                sources["e1_recipe_anchor_authority"]
            ),
            "content_verification_receipt_path": Path(
                sources["content_verification_receipt"]
            ),
            "burstgpt_shape_authority_path": Path(sources["burstgpt_shape_authority"]),
            "now_ns": 2_000_000_000,
        },
    )
    assert captured[1] == ("publish", (source_artifact, Path(output)))
    assert captured[2] == (
        "revalidate",
        (source_binding.absolute_path, 2_000_000_000),
    )
    with pytest.raises(SystemExit):
        main([*argv, "--code-git-head", "0" * 40])
