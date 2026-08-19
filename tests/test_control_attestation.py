from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import lightcone_spec.runtime.release_trust_root as root_module
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
)
from lightcone_spec.runtime.attester_bundle import (
    AttestationNoncePolicy,
    TrustedAttesterPolicyBundle,
)
from lightcone_spec.runtime.control_attestation import (
    CONTROL_ARTIFACT_TYPES,
    MAXIMUM_FORMAL_ATOMIC_CHALLENGES,
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
    VerifiedControlArtifact,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    SourceReleaseEd25519Root,
    SourceReleaseRootBinding,
    deployment_policy_subject_sha256,
)

HARDWARE_SHA256 = "a" * 64
ARTIFACT_SHA256 = "b" * 64
INVENTORY_SHA256 = "f" * 64
NOW_NS = 2_000_000_000


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _root_binding(private_key: Ed25519PrivateKey) -> SourceReleaseRootBinding:
    public_key = _public_bytes(private_key)
    spki = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    root = SourceReleaseEd25519Root(
        schema_version=1,
        kind="lightcone_source_release_ed25519_root",
        root_id="lightcone-release-root-2026q3",
        key_id="lightcone-release-root-key-2026q3",
        algorithm="Ed25519",
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        public_key_sha256=hashlib.sha256(public_key).hexdigest(),
        spki_sha256=hashlib.sha256(spki).hexdigest(),
    )
    path = str(Path("/validation/release-root.json"))
    return SourceReleaseRootBinding(
        root=root,
        path=path,
        sidecar_path=f"{path}.sha256",
        semantic_sha256=root.sha256,
        file_sha256="1" * 64,
        sidecar_file_sha256="2" * 64,
    )


def _bundle(private_key: Ed25519PrivateKey) -> TrustedAttesterPolicyBundle:
    public_key = _public_bytes(private_key)
    fingerprint = hashlib.sha256(public_key).hexdigest()
    return TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="validation-control-bundle-v1",
        valid_from_ns=1,
        expires_ns=10_000_000_000,
        nonce_policy=AttestationNoncePolicy(
            schema_version=1,
            kind="lightcone_attestation_nonce_policy",
            nonce_bytes=32,
            minimum_lifetime_ns=100_000_000,
            maximum_lifetime_ns=2_000_000_000,
            maximum_clock_skew_ns=100_000_000,
            replay_policy="external_single_use_store",
            subject_binding_required=True,
        ),
        hardware_envelope_sha256_allowlist=(HARDWARE_SHA256,),
        trusted_attester_policy=TrustedAttesterPolicy(
            policy_id="validation-control-policy-v1",
            trusted_attesters=(
                ("validation-signer", "validation-signer-key", fingerprint),
            ),
            public_keys=((fingerprint, base64.b64encode(public_key).decode("ascii")),),
        ),
    )


def _deployment_authorization(
    *,
    root_private_key: Ed25519PrivateKey,
    root_binding: SourceReleaseRootBinding,
    bundle: TrustedAttesterPolicyBundle,
) -> DeploymentPolicyAuthorization:
    subject_sha256 = deployment_policy_subject_sha256(
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=INVENTORY_SHA256,
        bundle_sha256=bundle.sha256,
    )
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="validation-deployment-policy-1",
        nonce_base64=base64.b64encode(b"d" * 32).decode("ascii"),
        subject_sha256=subject_sha256,
        issued_ns=1_500_000_000,
        expires_ns=3_000_000_000,
    )
    signature = root_private_key.sign(
        attestation_message(challenge, payload_sha256=bundle.sha256)
    )
    return DeploymentPolicyAuthorization(
        schema_version=1,
        kind="lightcone_deployment_policy_authorization",
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=INVENTORY_SHA256,
        bundle=bundle,
        challenge=challenge,
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )


def _envelope(
    *,
    artifact_type: str,
    artifact_private_key: Ed25519PrivateKey,
    root_binding: SourceReleaseRootBinding,
    bundle: TrustedAttesterPolicyBundle,
    authorization: DeploymentPolicyAuthorization,
    nonce_byte: bytes = b"n",
) -> ControlArtifactAttestation:
    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type=artifact_type,
        artifact_sha256=ARTIFACT_SHA256,
        protocol_sha256="c" * 64,
        registry_sha256="d" * 64,
        lineage_sha256="e" * 64,
    )
    nonce = nonce_byte if len(nonce_byte) == 32 else nonce_byte * 32
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"validation-{artifact_type.replace('_', '-')}-1",
        nonce_base64=base64.b64encode(nonce).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    public_key = _public_bytes(artifact_private_key)
    signature = artifact_private_key.sign(
        attestation_message(challenge, payload_sha256=subject.artifact_sha256)
    )
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="validation-signer",
        key_id="validation-signer-key",
        environment="release",
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        challenge_sha256=challenge.sha256,
        payload_sha256=subject.artifact_sha256,
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )
    return ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=HARDWARE_SHA256,
        trust_anchor_sha256=root_binding.sha256,
        trust_bundle_sha256=bundle.sha256,
        trusted_attester_policy_sha256=bundle.trusted_attester_policy.sha256,
        deployment_policy_authorization=authorization,
        challenge=challenge,
        attestation=attestation,
    )


def _authority(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Ed25519PrivateKey,
    SourceReleaseRootBinding,
    TrustedAttesterPolicyBundle,
    DeploymentPolicyAuthorization,
]:
    root_private = Ed25519PrivateKey.generate()
    artifact_private = Ed25519PrivateKey.generate()
    binding = _root_binding(root_private)
    bundle = _bundle(artifact_private)
    authorization = _deployment_authorization(
        root_private_key=root_private,
        root_binding=binding,
        bundle=bundle,
    )
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: binding,
    )
    return artifact_private, binding, bundle, authorization


@pytest.mark.parametrize("artifact_type", CONTROL_ARTIFACT_TYPES)
def test_all_control_types_consume_dynamic_root_signed_policy(
    monkeypatch: pytest.MonkeyPatch,
    artifact_type: str,
) -> None:
    private, binding, bundle, authorization = _authority(monkeypatch)
    envelope = _envelope(
        artifact_type=artifact_type,
        artifact_private_key=private,
        root_binding=binding,
        bundle=bundle,
        authorization=authorization,
    )
    result = verify_release_control_artifact_attestation(
        envelope,
        expected_inventory_sha256=INVENTORY_SHA256,
        now_ns=NOW_NS,
        consumed_challenge_sha256s=(),
    )
    assert result.artifact_type == artifact_type
    assert result.challenge_sha256 == envelope.challenge.sha256
    assert result.deployment_policy_challenge_sha256 == authorization.challenge.sha256
    assert ControlArtifactAttestation.from_dict(envelope.to_dict()) == envelope


def test_control_rejects_inventory_hardware_signature_expiry_and_both_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, binding, bundle, authorization = _authority(monkeypatch)
    envelope = _envelope(
        artifact_type="capacity",
        artifact_private_key=private,
        root_binding=binding,
        bundle=bundle,
        authorization=authorization,
    )
    verify = lambda value, **kwargs: verify_release_control_artifact_attestation(
        value,
        expected_inventory_sha256=kwargs.pop(
            "expected_inventory_sha256", INVENTORY_SHA256
        ),
        now_ns=kwargs.pop("now_ns", NOW_NS),
        consumed_challenge_sha256s=kwargs.pop("consumed", ()),
        **kwargs,
    )
    with pytest.raises(ValueError, match="inventory"):
        verify(envelope, expected_inventory_sha256="0" * 64)
    with pytest.raises(ValueError, match="not allowlisted"):
        verify(replace(envelope, hardware_envelope_sha256="0" * 64))
    with pytest.raises(ValueError, match="expired"):
        verify(envelope, now_ns=3_000_000_001)
    with pytest.raises(ValueError, match="already consumed"):
        verify(envelope, consumed=(authorization.challenge.sha256,))
    with pytest.raises(ValueError, match="already consumed"):
        verify(envelope, consumed=(envelope.challenge.sha256,))

    forged = bytearray(base64.b64decode(envelope.attestation.signature_base64))
    forged[0] ^= 1
    with pytest.raises(ValueError, match="signature is invalid"):
        verify(
            replace(
                envelope,
                attestation=replace(
                    envelope.attestation,
                    signature_base64=base64.b64encode(forged).decode("ascii"),
                ),
            )
        )


def test_batch_reserves_deployment_and_artifact_challenges_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private, binding, bundle, authorization = _authority(monkeypatch)
    envelopes = (
        _envelope(
            artifact_type="capacity",
            artifact_private_key=private,
            root_binding=binding,
            bundle=bundle,
            authorization=authorization,
            nonce_byte=b"1",
        ),
        _envelope(
            artifact_type="rank_aggregate",
            artifact_private_key=private,
            root_binding=binding,
            bundle=bundle,
            authorization=authorization,
            nonce_byte=b"2",
        ),
    )
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    store = ChallengeReplayStore(str(replay_root.resolve()))
    inner_challenge = "9" * 64
    results = verify_and_reserve_release_control_artifact_attestations(
        envelopes,
        expected_inventory_sha256=INVENTORY_SHA256,
        now_ns=NOW_NS,
        replay_store=store,
        additional_challenge_sha256s=(inner_challenge,),
    )
    assert len(results) == 2
    reservation_sha256 = control_challenge_reservation_sha256(
        results,
        reserved_ns=NOW_NS,
        additional_challenge_sha256s=(inner_challenge,),
    )
    reservation = replay_root / f"reservation-{reservation_sha256}.json"
    assert reservation.is_file()
    assert inner_challenge in reservation.read_text(encoding="utf-8")
    bound_reservation = store.bind_reservation(reservation_sha256)
    assert isinstance(bound_reservation, ChallengeReplayReservationBinding)
    assert inner_challenge in bound_reservation.revalidate()
    assert (
        ChallengeReplayReservationBinding.from_dict(
            bound_reservation.to_dict()
        ).revalidate()
        == bound_reservation.challenge_sha256s
    )
    assert len(tuple(replay_root.glob("reservation-*.json"))) == 1
    with pytest.raises(ValueError, match="already consumed"):
        verify_and_reserve_release_control_artifact_attestations(
            envelopes,
            expected_inventory_sha256=INVENTORY_SHA256,
            now_ns=NOW_NS,
            replay_store=store,
            additional_challenge_sha256s=(inner_challenge,),
        )

    fresh_private, fresh_binding, fresh_bundle, fresh_authorization = _authority(
        monkeypatch
    )
    fresh = _envelope(
        artifact_type="compile",
        artifact_private_key=fresh_private,
        root_binding=fresh_binding,
        bundle=fresh_bundle,
        authorization=fresh_authorization,
        nonce_byte=b"3",
    )
    with pytest.raises(ValueError, match="additional challenge is already consumed"):
        verify_and_reserve_release_control_artifact_attestations(
            (fresh,),
            expected_inventory_sha256=INVENTORY_SHA256,
            now_ns=NOW_NS,
            replay_store=store,
            additional_challenge_sha256s=(inner_challenge,),
        )


def test_tts_sized_288_control_batch_is_durable_and_max_plus_one_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private, binding, bundle, authorization = _authority(monkeypatch)
    envelopes = tuple(
        _envelope(
            artifact_type="non_serving_terminal",
            artifact_private_key=private,
            root_binding=binding,
            bundle=bundle,
            authorization=authorization,
            nonce_byte=hashlib.sha256(f"tts-cal-{index}".encode()).digest(),
        )
        for index in range(288)
    )
    replay_root = tmp_path / "large-replay"
    replay_root.mkdir(mode=0o700)
    store = ChallengeReplayStore(str(replay_root.resolve()))
    results = verify_and_reserve_release_control_artifact_attestations(
        envelopes,
        expected_inventory_sha256=INVENTORY_SHA256,
        now_ns=NOW_NS,
        replay_store=store,
    )
    assert len(results) == 288
    reservation_sha256 = control_challenge_reservation_sha256(
        results, reserved_ns=NOW_NS
    )
    binding_row = store.bind_reservation(reservation_sha256)
    assert len(binding_row.revalidate()) == 289
    assert binding_row.size > 4_096

    result = VerifiedControlArtifact(
        artifact_type="rank_aggregate",
        artifact_sha256="1" * 64,
        envelope_sha256="2" * 64,
        challenge_sha256="3" * 64,
        deployment_policy_challenge_sha256="4" * 64,
        deployment_policy_authorization_sha256="5" * 64,
        trust_bundle_sha256="6" * 64,
        trusted_attester_policy_sha256="7" * 64,
    )
    additional = tuple(
        hashlib.sha256(f"extra-{index}".encode()).hexdigest()
        for index in range(MAXIMUM_FORMAL_ATOMIC_CHALLENGES - 1)
    )
    assert len(
        {
            result.challenge_sha256,
            result.deployment_policy_challenge_sha256,
            *additional,
        }
    ) == (MAXIMUM_FORMAL_ATOMIC_CHALLENGES + 1)
    with pytest.raises(ValueError, match="exceeds source bound"):
        control_challenge_reservation_sha256(
            (result,),
            reserved_ns=NOW_NS,
            additional_challenge_sha256s=additional,
        )


def test_batch_rejects_duplicate_or_overlapping_additional_challenges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private, binding, bundle, authorization = _authority(monkeypatch)
    envelope = _envelope(
        artifact_type="compile",
        artifact_private_key=private,
        root_binding=binding,
        bundle=bundle,
        authorization=authorization,
    )
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    store = ChallengeReplayStore(str(replay_root.resolve()))
    with pytest.raises(ValueError, match="must be unique"):
        verify_and_reserve_release_control_artifact_attestations(
            (envelope,),
            expected_inventory_sha256=INVENTORY_SHA256,
            now_ns=NOW_NS,
            replay_store=store,
            additional_challenge_sha256s=("9" * 64, "9" * 64),
        )
    with pytest.raises(ValueError, match="additional and control challenges overlap"):
        verify_and_reserve_release_control_artifact_attestations(
            (envelope,),
            expected_inventory_sha256=INVENTORY_SHA256,
            now_ns=NOW_NS,
            replay_store=store,
            additional_challenge_sha256s=(envelope.challenge.sha256,),
        )
    assert not tuple(replay_root.glob("reservation-*.json"))


def test_replay_store_rejects_tampered_or_unknown_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private, binding, bundle, authorization = _authority(monkeypatch)
    envelope = _envelope(
        artifact_type="compile",
        artifact_private_key=private,
        root_binding=binding,
        bundle=bundle,
        authorization=authorization,
    )
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    (replay_root / "foreign.txt").write_text("not authority\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown entry"):
        verify_and_reserve_release_control_artifact_attestations(
            (envelope,),
            expected_inventory_sha256=INVENTORY_SHA256,
            now_ns=NOW_NS,
            replay_store=ChallengeReplayStore(str(replay_root.resolve())),
        )


@pytest.mark.parametrize(
    "hazard",
    ("unsafe_root", "fifo", "hardlink", "wrong_mode", "symlink"),
)
def test_replay_store_rejects_unsafe_directory_or_lock_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hazard: str,
) -> None:
    private, binding, bundle, authorization = _authority(monkeypatch)
    envelope = _envelope(
        artifact_type="compile",
        artifact_private_key=private,
        root_binding=binding,
        bundle=bundle,
        authorization=authorization,
    )
    replay_root = tmp_path / hazard
    replay_root.mkdir(mode=0o700)
    lock = replay_root / ".lock"
    if hazard == "unsafe_root":
        replay_root.chmod(0o777)
    elif hazard == "fifo":
        os.mkfifo(lock, mode=0o600)
    elif hazard == "hardlink":
        external = tmp_path / "foreign-lock"
        external.touch(mode=0o600)
        os.link(external, lock)
    elif hazard == "wrong_mode":
        lock.touch(mode=0o644)
    elif hazard == "symlink":
        target = tmp_path / "symlink-target"
        target.touch(mode=0o600)
        lock.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        verify_and_reserve_release_control_artifact_attestations(
            (envelope,),
            expected_inventory_sha256=INVENTORY_SHA256,
            now_ns=NOW_NS,
            replay_store=ChallengeReplayStore(str(replay_root.resolve())),
        )
    assert not tuple(replay_root.glob("reservation-*.json"))
