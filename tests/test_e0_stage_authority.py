from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.experiments import e0_stage_authority
from lightcone_spec.experiments.e0_stage_authority import (
    E0_EXCLUDED_PILOT_RULE,
    E0_FINAL_MATERIALIZATION_RULE,
    E0_ONLINESPEC_ROLES,
    E0_ONLINESPEC_TUNING_RULE,
    E0OnlineSpecSelectedRecipe,
    E0OnlineSpecSourceAuthority,
    E0OnlineSpecTuningSeal,
    E0PowerPrefixReceipt,
    SignedE0OnlineSpecTuningSeal,
    SignedE0PowerPrefixReceipt,
)
from lightcone_spec.experiments.formal_protocol import (
    BANNED_MODEL,
    E0_METHOD_ROLES,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    _ordered_materializations,
    _validate_downstream_main_materialization,
    e0_onlinespec_source_authority_from_dict,
    e0_onlinespec_source_authority_to_dict,
    signed_e0_onlinespec_tuning_seal_from_dict,
    signed_e0_onlinespec_tuning_seal_to_dict,
    signed_e0_power_prefix_from_dict,
    signed_e0_power_prefix_to_dict,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_COMMIT,
    ONLINE_SPEC_SOURCE_AUDIT_SHA256,
    ONLINE_SPEC_TREE,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_BACKENDS,
    E0_LOADS,
    E0_MODELS,
    E0_TASKS,
    E0CompatibilityDecision,
    E0CompatibilityReceipt,
    GpuHourEstimate,
    _e0_cells_from_verified_sources,
    _e0_tuning_cells_from_verified_sources,
    _receipt,
)
from lightcone_spec.experiments.statistics import PILOT_BLOCK_COUNT, PowerSizingPlan
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
)


def _sha(label: str) -> str:
    return content_sha256({"label": label})


def _compatibility(*, valid_count: int = 2) -> E0CompatibilityReceipt:
    decisions = []
    index = 0
    for model in E0_MODELS:
        for backend in E0_BACKENDS:
            for task in E0_TASKS:
                decisions.append(
                    E0CompatibilityDecision(
                        model=model,
                        backend=backend,
                        task=task,
                        disposition="VALID" if index < valid_count else "N/A",
                        reason_code=(
                            "proof_backed_compatible"
                            if index < valid_count
                            else "proof_backed_not_applicable"
                        ),
                        interface_sha256=_sha(f"interface:{model}:{backend}:{task}"),
                        task_native_workload_sha256=_sha(f"workload:{task}"),
                    )
                )
                index += 1
    return E0CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol-lock"),
        upstream_e6_receipt_sha256=_sha("e6-materialization"),
        decisions=tuple(sorted(decisions, key=lambda row: row.decision_id)),
    )


def _sign_tuning_seal(
    payload: E0OnlineSpecTuningSeal,
    *,
    index: int,
) -> SignedE0OnlineSpecTuningSeal:
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload_sha256 = payload.sha256
    challenge = AttestationChallenge.issue(
        challenge_id=f"e0-tuning-validation-{index}",
        subject_sha256=payload_sha256,
        lifetime_s=60,
        now_ns=20_000_000_000,
    )
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=payload_sha256)
    )
    return SignedE0OnlineSpecTuningSeal(
        payload=payload,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="validation-signer",
            key_id=f"e0-tuning-key-{index}",
            environment="release",
            public_key_base64=base64.b64encode(public_bytes).decode(),
            challenge_sha256=challenge.sha256,
            payload_sha256=payload_sha256,
            signature_base64=base64.b64encode(signature).decode(),
        ),
    )


def _sign_power_prefix(payload: E0PowerPrefixReceipt) -> SignedE0PowerPrefixReceipt:
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload_sha256 = payload.sha256
    challenge = AttestationChallenge.issue(
        challenge_id="e0-power-validation",
        subject_sha256=payload_sha256,
        lifetime_s=60,
        now_ns=20_000_000_000,
    )
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=payload_sha256)
    )
    return SignedE0PowerPrefixReceipt(
        payload=payload,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="validation-signer",
            key_id="e0-power-key",
            environment="release",
            public_key_base64=base64.b64encode(public_bytes).decode(),
            challenge_sha256=challenge.sha256,
            payload_sha256=payload_sha256,
            signature_base64=base64.b64encode(signature).decode(),
        ),
    )


def _signed_tuning_seals(
    compatibility: E0CompatibilityReceipt,
) -> tuple[SignedE0OnlineSpecTuningSeal, ...]:
    valid = tuple(row for row in compatibility.decisions if row.disposition == "VALID")
    rows = []
    for index, decision in enumerate(valid):
        selected = tuple(
            E0OnlineSpecSelectedRecipe(
                method_role=role,
                candidate_id=_sha(f"candidate:{decision.decision_id}:{role}"),
                selected_cell_id=_sha(f"tuning-cell:{decision.decision_id}:{role}"),
            )
            for role in E0_ONLINESPEC_ROLES
        )
        rows.append(
            _sign_tuning_seal(
                E0OnlineSpecTuningSeal(
                    schema_version=1,
                    protocol_lock_sha256=compatibility.protocol_lock_sha256,
                    registry_sha256=_sha("registry"),
                    upstream_e6_confirmation_sha256=_sha("e6-confirmation"),
                    signed_compatibility_sha256=_sha("signed-compatibility"),
                    onlinespec_source_authority_sha256=_sha("onlinespec-source"),
                    tuning_materialization_receipt_sha256=_sha(
                        "tuning-materialization"
                    ),
                    tuning_coverage_receipt_sha256=_sha("tuning-coverage"),
                    evidence_manifest_sha256=_sha("tuning-evidence"),
                    inventory_sha256=_sha("inventory"),
                    decision_id=decision.decision_id,
                    model=decision.model,
                    backend=decision.backend,
                    task=decision.task,
                    interface_sha256=decision.interface_sha256,
                    task_native_workload_sha256=(decision.task_native_workload_sha256),
                    selected_recipes=selected,
                ),
                index=index,
            )
        )
    return tuple(rows)


def _cells_inputs() -> tuple[
    E0CompatibilityReceipt,
    tuple[SignedE0OnlineSpecTuningSeal, ...],
    dict[str, object],
]:
    compatibility = _compatibility()
    signed_seals = _signed_tuning_seals(compatibility)
    return (
        compatibility,
        signed_seals,
        {
            "compatibility": compatibility,
            "signed_compatibility_sha256": _sha("signed-compatibility"),
            "signed_e6_confirmation_sha256": _sha("e6-confirmation"),
            "signed_tuning_seals": signed_seals,
            "frozen_tts_recipe_sha256": _sha("frozen-tts"),
            "lightcone_recipe_sha256": _sha("lightcone"),
        },
    )


def test_e0_tuning_pilots_and_final_prefix_are_disjoint_and_exact() -> None:
    compatibility, _signed_seals, common = _cells_inputs()
    tuning = _e0_tuning_cells_from_verified_sources(
        compatibility=compatibility,
        signed_compatibility_sha256=_sha("signed-compatibility"),
        signed_e6_confirmation_sha256=_sha("e6-confirmation"),
        onlinespec_source_authority_sha256=_sha("onlinespec-source"),
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
    )
    pilots = _e0_cells_from_verified_sources(
        **common,
        block_indices=tuple(range(PILOT_BLOCK_COUNT)),
    )
    power_sha256 = _sha("e0-power-prefix")
    final = _e0_cells_from_verified_sources(
        **common,
        block_indices=tuple(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + 12)),
        power_prefix_source_sha256=power_sha256,
        pilot_materialization_receipt_sha256=_sha("e0-pilot-materialization"),
        pilot_coverage_receipt_sha256=_sha("e0-pilot-coverage"),
    )

    assert compatibility.valid_count == 2
    assert len(tuning) > 3 * compatibility.valid_count
    assert len(pilots) == 16 * compatibility.valid_count * PILOT_BLOCK_COUNT
    assert len(final) == 16 * compatibility.valid_count * 12
    assert E0_ONLINESPEC_ROLES == (
        "OnlineSPEC-OGD",
        "OnlineSPEC-OPT",
        "OnlineSPEC-ENS",
    )
    assert {row.method_role for row in final} == set(E0_METHOD_ROLES)
    assert {dict(row.dimensions)["load"] for row in final} == set(E0_LOADS)
    assert {dict(row.dimensions)["block"] for row in pilots} == set(range(4))
    assert {dict(row.dimensions)["block"] for row in final} == set(range(4, 16))
    assert {row.cell_id for row in pilots}.isdisjoint(row.cell_id for row in final)
    assert all(dict(row.dimensions)["block_phase"] == "final" for row in final)

    main = _receipt(
        stage="E0",
        protocol_lock_sha256=compatibility.protocol_lock_sha256,
        upstream_receipt_sha256s=(compatibility.upstream_e6_receipt_sha256,),
        source_decision_sha256=power_sha256,
        materialization_rule=E0_FINAL_MATERIALIZATION_RULE,
        cells=final,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    _validate_downstream_main_materialization(main)


def test_e0_main_registry_rejects_tuning_and_excluded_pilot_receipts() -> None:
    compatibility, _signed_seals, common = _cells_inputs()
    pilots = _e0_cells_from_verified_sources(
        **common,
        block_indices=tuple(range(PILOT_BLOCK_COUNT)),
    )
    for rule, cells in (
        (
            E0_ONLINESPEC_TUNING_RULE,
            _e0_tuning_cells_from_verified_sources(
                compatibility=compatibility,
                signed_compatibility_sha256=_sha("signed-compatibility"),
                signed_e6_confirmation_sha256=_sha("e6-confirmation"),
                onlinespec_source_authority_sha256=_sha("onlinespec-source"),
                frozen_tts_recipe_sha256=_sha("frozen-tts"),
            ),
        ),
        (E0_EXCLUDED_PILOT_RULE, pilots),
    ):
        receipt = _receipt(
            stage="E0",
            protocol_lock_sha256=compatibility.protocol_lock_sha256,
            upstream_receipt_sha256s=(compatibility.upstream_e6_receipt_sha256,),
            source_decision_sha256=_sha(f"source:{rule}"),
            materialization_rule=rule,
            cells=cells,
            gpu_hours=GpuHourEstimate.unmeasured(),
        )
        with pytest.raises(ValueError, match="excluded-pilot tuning materialization"):
            _ordered_materializations((receipt,))


def test_e0_cell_materializer_rejects_incomplete_seals_and_bad_prefix() -> None:
    _compatibility_receipt, signed_seals, common = _cells_inputs()
    with pytest.raises(ValueError, match="cover every VALID decision"):
        _e0_cells_from_verified_sources(
            **{**common, "signed_tuning_seals": signed_seals[:-1]},
            block_indices=tuple(range(4, 16)),
            power_prefix_source_sha256=_sha("e0-power-prefix"),
            pilot_materialization_receipt_sha256=_sha("e0-pilots"),
            pilot_coverage_receipt_sha256=_sha("e0-pilot-coverage"),
        )
    with pytest.raises(ValueError, match="exact pilot or powered final prefix"):
        _e0_cells_from_verified_sources(
            **common,
            block_indices=tuple(range(3, 15)),
            power_prefix_source_sha256=_sha("e0-power-prefix"),
            pilot_materialization_receipt_sha256=_sha("e0-pilots"),
            pilot_coverage_receipt_sha256=_sha("e0-pilot-coverage"),
        )


def test_e0_typed_authorities_reject_role_drift_and_banned_model() -> None:
    compatibility = _compatibility(valid_count=1)
    decision = next(
        row for row in compatibility.decisions if row.disposition == "VALID"
    )
    selected = tuple(
        E0OnlineSpecSelectedRecipe(
            method_role=role,
            candidate_id=_sha(f"candidate:{role}"),
            selected_cell_id=_sha(f"cell:{role}"),
        )
        for role in E0_ONLINESPEC_ROLES
    )
    base = E0OnlineSpecTuningSeal(
        schema_version=1,
        protocol_lock_sha256=compatibility.protocol_lock_sha256,
        registry_sha256=_sha("registry"),
        upstream_e6_confirmation_sha256=_sha("e6-confirmation"),
        signed_compatibility_sha256=_sha("signed-compatibility"),
        onlinespec_source_authority_sha256=_sha("onlinespec-source"),
        tuning_materialization_receipt_sha256=_sha("tuning-materialization"),
        tuning_coverage_receipt_sha256=_sha("tuning-coverage"),
        evidence_manifest_sha256=_sha("evidence"),
        inventory_sha256=_sha("inventory"),
        decision_id=decision.decision_id,
        model=decision.model,
        backend=decision.backend,
        task=decision.task,
        interface_sha256=decision.interface_sha256,
        task_native_workload_sha256=decision.task_native_workload_sha256,
        selected_recipes=selected,
    )
    with pytest.raises(ValueError, match="recipe/model panel"):
        replace(base, selected_recipes=tuple(reversed(selected)))
    with pytest.raises(ValueError, match="108-cell universe"):
        replace(decision, model=BANNED_MODEL)


def test_e0_signed_tuning_and_power_codecs_are_strict() -> None:
    compatibility = _compatibility(valid_count=2)
    signed_tuning = _signed_tuning_seals(compatibility)
    tuning_encoded = json.loads(
        json.dumps(signed_e0_onlinespec_tuning_seal_to_dict(signed_tuning[0]))
    )
    assert (
        signed_e0_onlinespec_tuning_seal_from_dict(tuning_encoded) == signed_tuning[0]
    )

    sizing = PowerSizingPlan(
        status="READY",
        pilot_block_ids=tuple(f"E0:excluded_pilot:{index}" for index in range(4)),
        selected_final_blocks=12,
        minimum_final_blocks=12,
        maximum_final_blocks=20,
        target_power=0.8,
        family_alpha=0.05,
        adjusted_alpha=0.025,
        minimum_relative_effect=0.03,
        minimum_log_effect=0.029558802241544398,
        pilot_log_standard_deviations=(
            ("LightCone-Static", 0.01),
            ("LightCone-TTS", 0.01),
        ),
        power_grid=(),
    )
    signed_power = _sign_power_prefix(
        E0PowerPrefixReceipt(
            schema_version=1,
            protocol_lock_sha256=compatibility.protocol_lock_sha256,
            registry_sha256=_sha("registry"),
            upstream_e6_confirmation_sha256=_sha("e6-confirmation"),
            signed_compatibility_sha256=_sha("signed-compatibility"),
            signed_tuning_seal_sha256s=tuple(
                sorted(row.sha256 for row in signed_tuning)
            ),
            pilot_materialization_receipt_sha256=_sha("pilot-materialization"),
            pilot_coverage_receipt_sha256=_sha("pilot-coverage"),
            evidence_manifest_sha256=_sha("pilot-evidence"),
            inventory_sha256=_sha("inventory"),
            power_sizing=sizing,
            selected_final_blocks=12,
            selected_final_prefix=tuple(range(4, 16)),
        )
    )
    power_encoded = json.loads(json.dumps(signed_e0_power_prefix_to_dict(signed_power)))
    assert signed_e0_power_prefix_from_dict(power_encoded) == signed_power
    power_encoded["payload"]["selected_final_prefix"][0] = 3
    with pytest.raises(ValueError, match="power prefix"):
        signed_e0_power_prefix_from_dict(power_encoded)


def test_e0_source_authority_codec_reopens_exact_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = (tmp_path / "onlinespec").resolve()
    checkout.mkdir()
    audit = (tmp_path / "onlinespec-audit.json").resolve()
    audit.write_text("{}\n", encoding="utf-8")
    verified = {
        "commit": ONLINE_SPEC_COMMIT,
        "tree": ONLINE_SPEC_TREE,
        "key_files": {"optimizer.py": _sha("optimizer")},
    }

    def fake_verify(*_args: object, **_kwargs: object) -> dict[str, object]:
        return verified

    monkeypatch.setattr(
        e0_stage_authority,
        "verify_onlinespec_source_checkout",
        fake_verify,
    )
    authority = E0OnlineSpecSourceAuthority(
        schema_version=1,
        checkout_path=str(checkout),
        audit_path=str(audit),
        source_audit_sha256=ONLINE_SPEC_SOURCE_AUDIT_SHA256,
        commit=ONLINE_SPEC_COMMIT,
        tree=ONLINE_SPEC_TREE,
        verification_sha256=content_sha256(verified),
    )
    encoded = json.loads(json.dumps(e0_onlinespec_source_authority_to_dict(authority)))
    assert e0_onlinespec_source_authority_from_dict(encoded) == authority

    encoded["verification_sha256"] = _sha("foreign-verification")
    with pytest.raises(ValueError, match="source verification changed"):
        e0_onlinespec_source_authority_from_dict(encoded)
