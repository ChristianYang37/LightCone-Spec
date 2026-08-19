from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.runtime import offline_signer
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.attester_bundle import (
    AttestationNoncePolicy,
    TrustedAttesterPolicyBundle,
)
from lightcone_spec.runtime.control_attestation import (
    ControlArtifactAttestation,
    ControlArtifactSubject,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    SourceReleaseEd25519Root,
    SourceReleaseRootBinding,
    deployment_policy_subject_sha256,
    verify_source_signed_deployment_policy,
)

NOW_NS = 2_000_000_000


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _public(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _root_binding(private_key: Ed25519PrivateKey) -> SourceReleaseRootBinding:
    public = _public(private_key)
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
        public_key_base64=base64.b64encode(public).decode("ascii"),
        public_key_sha256=hashlib.sha256(public).hexdigest(),
        spki_sha256=hashlib.sha256(spki).hexdigest(),
    )
    return SourceReleaseRootBinding(
        root=root,
        path="/validation/release-root.json",
        sidecar_path="/validation/release-root.json.sha256",
        semantic_sha256=root.sha256,
        file_sha256=_sha("root-file"),
        sidecar_file_sha256=_sha("root-sidecar"),
    )


def _bundle(
    private_key: Ed25519PrivateKey,
    *,
    hardware_sha256: str,
) -> TrustedAttesterPolicyBundle:
    public = _public(private_key)
    fingerprint = hashlib.sha256(public).hexdigest()
    return TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="offline-signer-release-bundle-v1",
        valid_from_ns=1,
        expires_ns=20_000_000_000,
        nonce_policy=AttestationNoncePolicy(
            schema_version=1,
            kind="lightcone_attestation_nonce_policy",
            nonce_bytes=32,
            minimum_lifetime_ns=1_000_000_000,
            maximum_lifetime_ns=600_000_000_000,
            maximum_clock_skew_ns=30_000_000_000,
            replay_policy="external_single_use_store",
            subject_binding_required=True,
        ),
        hardware_envelope_sha256_allowlist=(hardware_sha256,),
        trusted_attester_policy=TrustedAttesterPolicy(
            policy_id="offline-signer-release-policy-v1",
            trusted_attesters=(
                (
                    "release-signer",
                    "release-signer-key",
                    fingerprint,
                ),
            ),
            public_keys=((fingerprint, base64.b64encode(public).decode("ascii")),),
        ),
    )


def _key_file(path: Path, private_key: Ed25519PrivateKey) -> int:
    path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return os.open(path, os.O_RDONLY)


def _install_root(
    monkeypatch: pytest.MonkeyPatch,
    binding: SourceReleaseRootBinding,
) -> None:
    from lightcone_spec.runtime import release_trust_root

    monkeypatch.setattr(
        offline_signer,
        "load_source_release_ed25519_root",
        lambda: binding,
    )
    monkeypatch.setattr(
        release_trust_root,
        "load_source_release_ed25519_root",
        lambda: binding,
    )


def test_local_cli_signs_deployment_and_control_via_fd_without_secret_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    _install_root(monkeypatch, root_binding)
    hardware = _sha("hardware")
    inventory = _sha("inventory")
    bundle = _bundle(signer_private, hardware_sha256=hardware)
    bundle_path = (tmp_path / "bundle.json").resolve()
    publish_canonical_json_no_replace(bundle_path, bundle.to_dict())
    root_key_path = (tmp_path / "root-private-key").resolve()
    root_fd = _key_file(root_key_path, root_private)
    deployment_path = (tmp_path / "deployment.json").resolve()
    deployment_argv = [
        "sign-deployment",
        "--bundle",
        str(bundle_path),
        "--inventory-sha256",
        inventory,
        "--challenge-id",
        "release-deployment-001",
        "--now-ns",
        str(NOW_NS),
        "--key-fd",
        str(root_fd),
        "--output",
        str(deployment_path),
    ]
    assert str(root_key_path) not in deployment_argv
    try:
        assert offline_signer.main(deployment_argv) == 0
    finally:
        os.close(root_fd)
    authorization = DeploymentPolicyAuthorization.from_dict(
        offline_signer._load_public_json(str(deployment_path))
    )
    verify_source_signed_deployment_policy(
        authorization,
        expected_inventory_sha256=inventory,
        now_ns=NOW_NS,
        consumed_challenge_sha256s=(),
    )

    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="interference",
        artifact_sha256=_sha("artifact"),
        protocol_sha256=_sha("protocol"),
        registry_sha256=_sha("registry"),
        lineage_sha256=_sha("lineage"),
    )
    subject_path = (tmp_path / "subject.json").resolve()
    publish_canonical_json_no_replace(subject_path, subject.to_dict())
    signer_key_path = (tmp_path / "signer-private-key").resolve()
    signer_fd = _key_file(signer_key_path, signer_private)
    control_path = (tmp_path / "control.json").resolve()
    control_argv = [
        "sign-control",
        "--subject",
        str(subject_path),
        "--deployment-authorization",
        str(deployment_path),
        "--hardware-envelope-sha256",
        hardware,
        "--attester-id",
        "release-signer",
        "--key-id",
        "release-signer-key",
        "--challenge-id",
        "release-control-001",
        "--now-ns",
        str(NOW_NS),
        "--key-fd",
        str(signer_fd),
        "--output",
        str(control_path),
    ]
    assert str(signer_key_path) not in control_argv
    try:
        assert offline_signer.main(control_argv) == 0
    finally:
        os.close(signer_fd)
    envelope = ControlArtifactAttestation.from_dict(
        offline_signer._load_public_json(str(control_path))
    )
    verified = verify_release_control_artifact_attestation(
        envelope,
        expected_inventory_sha256=inventory,
        now_ns=NOW_NS,
        consumed_challenge_sha256s=(),
    )
    assert verified.artifact_sha256 == subject.artifact_sha256
    output = capsys.readouterr()
    assert str(root_key_path) not in output.out + output.err
    assert str(signer_key_path) not in output.out + output.err


def test_signer_rejects_wrong_key_expiry_unsafe_mode_and_non_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_private = Ed25519PrivateKey.generate()
    foreign_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    _install_root(monkeypatch, root_binding)
    bundle = _bundle(foreign_private, hardware_sha256=_sha("hardware"))
    inventory = _sha("inventory")
    subject = deployment_policy_subject_sha256(
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=inventory,
        bundle_sha256=bundle.sha256,
    )
    expired = AttestationChallenge.issue(
        challenge_id="expired-deployment",
        subject_sha256=subject,
        lifetime_s=1.0,
        now_ns=1,
    )
    with pytest.raises(ValueError, match="expired"):
        offline_signer.sign_deployment_policy_authorization(
            bundle=bundle,
            inventory_sha256=inventory,
            challenge=expired,
            private_key=root_private,
            now_ns=NOW_NS,
        )
    valid = AttestationChallenge.issue(
        challenge_id="wrong-root",
        subject_sha256=subject,
        lifetime_s=1.0,
        now_ns=NOW_NS,
    )
    with pytest.raises(ValueError, match="not the source root"):
        offline_signer.sign_deployment_policy_authorization(
            bundle=bundle,
            inventory_sha256=inventory,
            challenge=valid,
            private_key=foreign_private,
            now_ns=NOW_NS,
        )

    unsafe_path = (tmp_path / "unsafe-key").resolve()
    unsafe_fd = _key_file(unsafe_path, root_private)
    os.close(unsafe_fd)
    unsafe_path.chmod(0o644)
    descriptor = os.open(unsafe_path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="0600"):
            offline_signer.load_ed25519_private_key_from_fd(
                descriptor,
                expected_public_key_sha256=root_binding.root.public_key_sha256,
            )
    finally:
        os.close(descriptor)

    monkeypatch.setattr(
        offline_signer.sys,
        "stdin",
        SimpleNamespace(isatty=lambda: False),
    )
    monkeypatch.setattr(
        offline_signer.sys,
        "stderr",
        SimpleNamespace(isatty=lambda: False),
    )
    with pytest.raises(RuntimeError, match="requires a TTY"):
        offline_signer.load_ed25519_private_key_from_tty(
            expected_public_key_sha256=root_binding.root.public_key_sha256
        )


def test_cli_failure_is_generic_and_no_replace(tmp_path: Path) -> None:
    output = (tmp_path / "must-not-overwrite.json").resolve()
    output.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit) as blocked:
        offline_signer.main(
            [
                "sign-deployment",
                "--bundle",
                str((tmp_path / "missing-bundle.json").resolve()),
                "--inventory-sha256",
                _sha("inventory"),
                "--challenge-id",
                "no-secret-in-error",
                "--key-fd",
                "0",
                "--output",
                str(output),
            ]
        )
    assert str(blocked.value) == (
        "offline signing failed closed; no private material was logged"
    )
    assert output.read_text(encoding="utf-8") == "sentinel\n"
