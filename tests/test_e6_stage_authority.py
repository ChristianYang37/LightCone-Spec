from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_dispatch import _protocol_lock

from lightcone_spec.experiments import e6_stage_authority
from lightcone_spec.experiments.e6_stage_authority import (
    E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
    E6ModelCompatibilityReceipt,
    E6NextnModelAuthorityInput,
    E6NextnModelCompatibility,
    reduce_e6_model_compatibility_from_proofs,
)
from lightcone_spec.experiments.formal_protocol import E6_MODELS, content_sha256
from lightcone_spec.experiments.formal_registry import (
    _validate_downstream_main_materialization,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    _e6_cells_from_verified_sources,
    _receipt,
)


def _sha(label: str) -> str:
    return content_sha256({"label": label})


def _artifact(tmp_path: Path, label: str) -> str:
    path = (tmp_path / f"{label}.json").resolve()
    path.write_text(
        json.dumps(
            {"schema_version": 1, "label": label},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _source(tmp_path: Path, model: str, index: int) -> E6NextnModelAuthorityInput:
    return E6NextnModelAuthorityInput.bind(
        model=model,
        target_member_id=f"e6-target-{index}",
        drafter_member_id=f"e6-nextn-drafter-{index}",
        artifact_path=_artifact(tmp_path, f"nextn-{index}"),
        expected_interface_sha256=_sha(f"interface-{index}"),
        expected_topology_sha256=_sha("tp2-dp1-topology"),
        expected_source_adapter_version=1,
    )


def _compatibility(
    source: E6NextnModelAuthorityInput,
    index: int,
) -> E6NextnModelCompatibility:
    return E6NextnModelCompatibility(
        model=source.model,
        source_input_sha256=source.sha256,
        dynamic_artifact_sha256=_sha(f"dynamic-artifact-{index}"),
        verified_authority_sha256=_sha(f"verified-authority-{index}"),
        interface_sha256=source.expected_interface_sha256,
        target_member_id=source.target_member_id,
        drafter_member_id=source.drafter_member_id,
        target_model_id=source.model,
        drafter_model_id=f"nextn-drafter-model-{index}",
        target_revision=f"target-revision-{index}",
        drafter_revision=f"drafter-revision-{index}",
        target_shard_manifest_sha256=_sha(f"target-shards-{index}"),
        drafter_shard_manifest_sha256=_sha(f"drafter-shards-{index}"),
        topology_sha256=source.expected_topology_sha256,
        source_adapter_version=source.expected_source_adapter_version,
        native_gpu_proof_sha256=_sha(f"native-gpu-{index}"),
        distributed_gpu_proof_sha256=_sha(f"distributed-gpu-{index}"),
        content_verification_receipt_sha256=_sha("prepared-content-receipt"),
        inventory_sha256=_sha("inventory"),
        gpu_uuids=("GPU-0", "GPU-1"),
    )


def test_e6_authority_inputs_are_path_bound_and_exact_two_model_receipt(
    tmp_path: Path,
) -> None:
    sources = tuple(
        _source(tmp_path, model, index) for index, model in enumerate(E6_MODELS)
    )
    rows = tuple(_compatibility(source, index) for index, source in enumerate(sources))
    receipt = E6ModelCompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_sha256=_sha("registry"),
        release_root_manifest_sha256=_sha("release-root"),
        protocol_sha256=E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        models=rows,
    )

    assert tuple(row.model for row in receipt.models) == E6_MODELS
    assert len({row.dynamic_artifact_sha256 for row in receipt.models}) == 2
    assert len({row.source_input_sha256 for row in receipt.models}) == 2

    with pytest.raises(ValueError, match="both exact models once"):
        replace(receipt, models=receipt.models[:1])
    with pytest.raises(ValueError, match="both exact models once"):
        replace(receipt, models=tuple(reversed(receipt.models)))
    with pytest.raises(ValueError, match="both exact models once"):
        replace(receipt, models=(receipt.models[0], receipt.models[0]))


def test_e6_source_rejects_banned_or_changed_artifact(tmp_path: Path) -> None:
    source = _source(tmp_path, E6_MODELS[0], 0)
    Path(source.artifact_path).write_text(
        json.dumps(
            {"schema_version": 1, "label": "tampered"},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed after binding"):
        source.__post_init__()

    with pytest.raises(ValueError, match="unsupported"):
        E6NextnModelAuthorityInput.bind(
            model="Qwen/Qwen3.5-35B-A3B",
            target_member_id="banned-target",
            drafter_member_id="banned-drafter",
            artifact_path=_artifact(tmp_path, "banned"),
            expected_interface_sha256=_sha("interface"),
            expected_topology_sha256=_sha("topology"),
            expected_source_adapter_version=1,
        )


def test_e6_compatibility_requires_shared_inventory_and_content_receipt(
    tmp_path: Path,
) -> None:
    sources = tuple(
        _source(tmp_path, model, index) for index, model in enumerate(E6_MODELS)
    )
    rows = tuple(_compatibility(source, index) for index, source in enumerate(sources))
    base = E6ModelCompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_sha256=_sha("registry"),
        release_root_manifest_sha256=_sha("release-root"),
        protocol_sha256=E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        models=rows,
    )

    with pytest.raises(ValueError, match="both exact models once"):
        replace(
            base,
            models=(
                rows[0],
                replace(rows[1], inventory_sha256=_sha("foreign-inventory")),
            ),
        )
    with pytest.raises(ValueError, match="both exact models once"):
        replace(
            base,
            models=(
                rows[0],
                replace(
                    rows[1],
                    content_verification_receipt_sha256=_sha("foreign-content"),
                ),
            ),
        )


def test_e6_tuning_and_main_cells_have_nonoverlapping_block_prefixes(
    tmp_path: Path,
) -> None:
    sources = tuple(
        _source(tmp_path, model, index) for index, model in enumerate(E6_MODELS)
    )
    compatibility = E6ModelCompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol-lock"),
        registry_sha256=_sha("registry"),
        release_root_manifest_sha256=_sha("release-root"),
        protocol_sha256=E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        models=tuple(
            _compatibility(source, index) for index, source in enumerate(sources)
        ),
    )
    common = {
        "signed_e5_confirmation_sha256": _sha("e5-confirmation"),
        "signed_model_compatibility_sha256": _sha("e6-compatibility"),
        "model_compatibility": compatibility,
        "frozen_tts_recipe_sha256": _sha("frozen-tts"),
        "lightcone_recipe_sha256": _sha("lightcone"),
    }
    pilots = _e6_cells_from_verified_sources(
        **common,
        block_indices=tuple(range(4)),
    )
    signed_power_prefix_sha256 = _sha("e6-power-prefix")
    final = _e6_cells_from_verified_sources(
        **common,
        block_indices=tuple(range(4, 16)),
        power_prefix_source_sha256=signed_power_prefix_sha256,
        pilot_materialization_receipt_sha256=_sha("e6-pilots"),
        pilot_coverage_receipt_sha256=_sha("e6-pilot-coverage"),
    )

    pilot_blocks = {
        dict(cell.dimensions).get("block")
        for cell in pilots
        if "block" in dict(cell.dimensions)
    }
    final_blocks = {
        dict(cell.dimensions).get("block")
        for cell in final
        if "block" in dict(cell.dimensions)
    }
    assert len(pilots) == 2 + 60 * 4
    assert len(final) == 60 * 12
    assert pilot_blocks == set(range(4))
    assert final_blocks == set(range(4, 16))
    assert pilot_blocks.isdisjoint(final_blocks)
    materialization = _receipt(
        stage="E6",
        protocol_lock_sha256=_sha("protocol-lock"),
        upstream_receipt_sha256s=(_sha("e5-materialization"),),
        source_decision_sha256=signed_power_prefix_sha256,
        materialization_rule=(
            "60_final_rows_per_block_reusing_global_model_preflights"
        ),
        cells=final,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    _validate_downstream_main_materialization(materialization)

    with pytest.raises(ValueError, match="exact pilot or final"):
        _e6_cells_from_verified_sources(
            **common,
            block_indices=tuple(range(3, 15)),
        )
    with pytest.raises(ValueError, match="lineage must be complete"):
        _e6_cells_from_verified_sources(
            **common,
            block_indices=tuple(range(4, 16)),
            power_prefix_source_sha256=signed_power_prefix_sha256,
        )


def test_e6_reducer_rejects_valid_proof_relabelled_as_another_target_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = tuple(
        _source(tmp_path, model, index) for index, model in enumerate(E6_MODELS)
    )
    source = sources[0]
    monkeypatch.setattr(
        e6_stage_authority,
        "validate_nextn_tp2_dynamic_authority_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(
            artifact_sha256=_sha("artifact"),
            sha256=_sha("verified"),
            interface_sha256=source.expected_interface_sha256,
            target_model_id=E6_MODELS[1],
            drafter_model_id="Qwen/Qwen3.5-NEXTN-drafter",
            target_revision="target-revision",
            drafter_revision="drafter-revision",
            target_shard_manifest_sha256=_sha("target-shards"),
            drafter_shard_manifest_sha256=_sha("drafter-shards"),
            topology_sha256=source.expected_topology_sha256,
            source_adapter_version=1,
            native_gpu_proof_sha256=_sha("native"),
            distributed_gpu_proof_sha256=_sha("distributed"),
            content_verification_receipt_sha256=_sha("content"),
            inventory_sha256=_sha("inventory"),
            gpu_uuids=("GPU-0", "GPU-1"),
        ),
    )

    with pytest.raises(ValueError, match="target model differs"):
        reduce_e6_model_compatibility_from_proofs(
            protocol_lock=_protocol_lock(),
            sources=sources,
            expected_inventory_sha256=_sha("inventory"),
            now_ns=2_000_000_000,
        )
