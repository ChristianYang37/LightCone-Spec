from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.experiments.downstream_stage_authority import (
    E3B_POWER_PREFIX_PROTOCOL_SHA256,
    E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
    E3bPowerPrefixReceipt,
    E5PowerAndAnchorReceipt,
    FormalFamilyPowerCommitment,
    SignedE3bPowerPrefixReceipt,
    SignedE5PowerAndAnchorReceipt,
)
from lightcone_spec.experiments.e6_stage_authority import (
    E6_POWER_PREFIX_PROTOCOL_SHA256,
    E6PowerPrefixReceipt,
    SignedE6PowerPrefixReceipt,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
    e3b_power_prefix_receipt_from_dict,
    e3b_power_prefix_receipt_to_dict,
    e5_power_and_anchor_receipt_from_dict,
    e5_power_and_anchor_receipt_to_dict,
    e6_power_prefix_receipt_from_dict,
    e6_power_prefix_receipt_to_dict,
    signed_e3b_power_prefix_from_dict,
    signed_e3b_power_prefix_to_dict,
    signed_e5_power_and_anchor_from_dict,
    signed_e5_power_and_anchor_to_dict,
    signed_e6_power_prefix_from_dict,
    signed_e6_power_prefix_to_dict,
)
from lightcone_spec.experiments.formal_slo_metrics import (
    FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
)
from lightcone_spec.experiments.stage_materialization import E5SelectedP99Anchor
from lightcone_spec.experiments.statistics import PowerSizingPlan
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
)


def _sha(label: str) -> str:
    return content_sha256({"label": label})


def _power_sizing() -> PowerSizingPlan:
    return PowerSizingPlan(
        status="READY",
        pilot_block_ids=tuple(f"excluded-pilot-{index}" for index in range(4)),
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


def _e3b_family_commitments() -> tuple[FormalFamilyPowerCommitment, ...]:
    power = _power_sizing()
    rows = []
    for index in range(96):
        dimensions = (("family", f"family-{index:03d}"),)
        family_sha256 = content_sha256(
            {
                "stage": "E3b",
                "model": "model",
                "task": "task",
                "dimensions": list(dimensions),
            }
        )
        rows.append(
            FormalFamilyPowerCommitment(
                schema_version=1,
                stage="E3b",
                model="model",
                task="task",
                family_dimensions=dimensions,
                family_sha256=family_sha256,
                slo_goodput_protocol_sha256=FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
                pilot_goodput_observation_sha256s=tuple(
                    sorted(
                        (
                            block,
                            role,
                            _sha(f"family-{index}-block-{block}-{role}"),
                        )
                        for block in range(4)
                        for role in ("Static", "TTS", "LightCone")
                    )
                ),
                power_sizing=power,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.family_sha256))


def _e5_family_commitments() -> tuple[FormalFamilyPowerCommitment, ...]:
    power = _power_sizing()
    rows = []
    for index in range(90):
        dimensions = (("family", f"family-{index:03d}"),)
        family_sha256 = content_sha256(
            {
                "stage": "E5",
                "model": "Qwen/Qwen3-8B",
                "task": "production_slo_power_prefix",
                "dimensions": list(dimensions),
            }
        )
        rows.append(
            FormalFamilyPowerCommitment(
                schema_version=1,
                stage="E5",
                model="Qwen/Qwen3-8B",
                task="production_slo_power_prefix",
                family_dimensions=dimensions,
                family_sha256=family_sha256,
                slo_goodput_protocol_sha256=FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
                pilot_goodput_observation_sha256s=tuple(
                    sorted(
                        (block, role, _sha(f"e5-{index}-{block}-{role}"))
                        for block in range(4)
                        for role in ("Static", "TTS", "LightCone")
                    )
                ),
                power_sizing=power,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.family_sha256))


def _signature_parts(payload: object, *, key_id: str):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload_sha256 = content_sha256(payload)
    challenge = AttestationChallenge.issue(
        challenge_id=f"{key_id}-challenge",
        subject_sha256=payload_sha256,
        lifetime_s=60,
        now_ns=30_000_000_000,
    )
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=payload_sha256)
    )
    return (
        payload_sha256,
        challenge,
        SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="validation-signer",
            key_id=key_id,
            environment="release",
            public_key_base64=base64.b64encode(public_bytes).decode(),
            challenge_sha256=challenge.sha256,
            payload_sha256=payload_sha256,
            signature_base64=base64.b64encode(signature).decode(),
        ),
    )


def _e3b() -> SignedE3bPowerPrefixReceipt:
    payload = E3bPowerPrefixReceipt(
        schema_version=2,
        protocol_lock_sha256=_sha("lock"),
        registry_sha256=_sha("registry"),
        pilot_materialization_receipt_sha256=_sha("e3b-pilots"),
        pilot_coverage_receipt_sha256=_sha("e3b-pilot-coverage"),
        evidence_manifest_sha256=_sha("e3b-pilot-evidence"),
        inventory_sha256=_sha("inventory"),
        protocol_sha256=E3B_POWER_PREFIX_PROTOCOL_SHA256,
        family_power_commitments=_e3b_family_commitments(),
        selected_final_blocks=12,
        selected_final_prefix=tuple(range(4, 16)),
    )
    return SignedE3bPowerPrefixReceipt(
        payload=payload,
        **dict(
            zip(
                ("payload_sha256", "challenge", "attestation"),
                _signature_parts(payload, key_id="e3b-power-key"),
                strict=True,
            )
        ),
    )


def _e5() -> SignedE5PowerAndAnchorReceipt:
    anchors = tuple(
        sorted(
            (
                E5SelectedP99Anchor(
                    backend=backend,
                    topology=topology,
                    family_id=f"family-{backend.lower()}-{topology}",
                    minimum_completions=10_000,
                )
                for backend in ("DFLASH", "DSPARK")
                for topology in ("tp1_dp1", "tp2_dp1", "tp1_dp2")
            ),
            key=lambda row: row.anchor_id,
        )
    )
    payload = E5PowerAndAnchorReceipt(
        schema_version=2,
        protocol_lock_sha256=_sha("lock"),
        registry_sha256=_sha("registry"),
        upstream_e1a_verification_sha256=_sha("e1a-verification"),
        pilot_materialization_receipt_sha256=_sha("e5-pilots"),
        pilot_coverage_receipt_sha256=_sha("e5-pilot-coverage"),
        evidence_manifest_sha256=_sha("e5-pilot-evidence"),
        inventory_sha256=_sha("inventory"),
        protocol_sha256=E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
        model="Qwen/Qwen3-8B",
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        dflash_lightcone_recipe_sha256=_sha("dflash-lightcone"),
        dspark_lightcone_recipe_sha256=_sha("dspark-lightcone"),
        family_power_commitments=_e5_family_commitments(),
        selected_final_blocks=12,
        selected_final_prefix=tuple(range(4, 16)),
        p99_anchors=anchors,
    )
    return SignedE5PowerAndAnchorReceipt(
        payload=payload,
        **dict(
            zip(
                ("payload_sha256", "challenge", "attestation"),
                _signature_parts(payload, key_id="e5-power-key"),
                strict=True,
            )
        ),
    )


def _e6() -> SignedE6PowerPrefixReceipt:
    payload = E6PowerPrefixReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("lock"),
        registry_sha256=_sha("registry"),
        upstream_e5_confirmation_sha256=_sha("e5-confirmation"),
        signed_model_compatibility_sha256=_sha("e6-compatibility"),
        pilot_materialization_receipt_sha256=_sha("e6-pilots"),
        pilot_coverage_receipt_sha256=_sha("e6-pilot-coverage"),
        evidence_manifest_sha256=_sha("e6-pilot-evidence"),
        inventory_sha256=_sha("inventory"),
        protocol_sha256=E6_POWER_PREFIX_PROTOCOL_SHA256,
        frozen_tts_recipe_sha256=_sha("frozen-tts"),
        lightcone_recipe_sha256=_sha("lightcone"),
        power_sizing=_power_sizing(),
        selected_final_blocks=12,
        selected_final_prefix=tuple(range(4, 16)),
    )
    return SignedE6PowerPrefixReceipt(
        payload=payload,
        **dict(
            zip(
                ("payload_sha256", "challenge", "attestation"),
                _signature_parts(payload, key_id="e6-power-key"),
                strict=True,
            )
        ),
    )


@pytest.mark.parametrize(
    (
        "signed",
        "signed_encoder",
        "signed_decoder",
        "payload_encoder",
        "payload_decoder",
    ),
    (
        (
            _e3b(),
            signed_e3b_power_prefix_to_dict,
            signed_e3b_power_prefix_from_dict,
            e3b_power_prefix_receipt_to_dict,
            e3b_power_prefix_receipt_from_dict,
        ),
        (
            _e5(),
            signed_e5_power_and_anchor_to_dict,
            signed_e5_power_and_anchor_from_dict,
            e5_power_and_anchor_receipt_to_dict,
            e5_power_and_anchor_receipt_from_dict,
        ),
        (
            _e6(),
            signed_e6_power_prefix_to_dict,
            signed_e6_power_prefix_from_dict,
            e6_power_prefix_receipt_to_dict,
            e6_power_prefix_receipt_from_dict,
        ),
    ),
)
def test_signed_power_sources_round_trip_and_reject_prefix_tamper(
    signed: object,
    signed_encoder: object,
    signed_decoder: object,
    payload_encoder: object,
    payload_decoder: object,
) -> None:
    encoded = json.loads(json.dumps(signed_encoder(signed)))
    assert signed_decoder(encoded) == signed
    payload = signed.payload
    assert payload_decoder(json.loads(json.dumps(payload_encoder(payload)))) == payload

    encoded["payload"]["selected_final_prefix"][0] = 3
    with pytest.raises(ValueError, match="prefix"):
        signed_decoder(encoded)


def test_durable_registry_receipt_declares_all_typed_power_sources() -> None:
    fields = FormalRegistryVerificationReceipt.__dataclass_fields__
    assert {
        "appended_signed_e3b_power_prefixes",
        "appended_signed_e5_power_and_anchor_prefixes",
        "appended_signed_e6_power_prefixes",
    } <= set(fields)
