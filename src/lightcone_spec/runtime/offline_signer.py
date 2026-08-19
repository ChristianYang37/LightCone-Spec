"""Local-only Ed25519 signing ceremony for release control artifacts.

This module is intentionally separate from the verification-only runtime
modules.  It never accepts private bytes, a private path, or a passphrase on
the command line or through the environment.  A key is read either from an
already-open file descriptor or from an absolute path entered through an
unechoed TTY prompt.  Only public, canonical, no-replace artifacts are written.

The module must run on the trusted local signing host.  It is not installed or
copied to the GPU host as a source of secret material.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import stat
import sys
import time
from pathlib import Path
from typing import NoReturn

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
)
from lightcone_spec.runtime.attester_bundle import TrustedAttesterPolicyBundle
from lightcone_spec.runtime.content_authorization import (
    CONTENT_AUTHORIZATION_SOURCE_TYPES,
    DatasetContentReleaseAuthorization,
    PreparedModelContentReleaseAuthorization,
    ReleaseWorkloadSourceAuthorization,
    build_content_authorization_from_source,
    content_authorization_source_from_dict,
    verify_dataset_content_release_authorization,
    verify_prepared_model_content_release_authorization,
    verify_release_workload_source_authorization,
)
from lightcone_spec.runtime.control_attestation import (
    ControlArtifactAttestation,
    ControlArtifactSubject,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.release_trust_root import (
    DEPLOYMENT_POLICY_MAXIMUM_LIFETIME_NS,
    DeploymentPolicyAuthorization,
    deployment_policy_subject_sha256,
    load_source_release_ed25519_root,
    verify_source_signed_deployment_policy,
)
from lightcone_spec.runtime.scientific_signing import (
    SCIENTIFIC_ARTIFACT_TYPES,
    finalize_scientific_candidate,
    scientific_payload_json_from_source,
    scientific_payload_sha256,
    sign_scientific_candidate,
)

_MAXIMUM_PRIVATE_KEY_BYTES = 16 * 1024


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _decode_private_key(body: bytearray) -> Ed25519PrivateKey:
    try:
        if len(body) == 32:
            return Ed25519PrivateKey.from_private_bytes(bytes(body))
        loaded = serialization.load_pem_private_key(bytes(body), password=None)
        if not isinstance(loaded, Ed25519PrivateKey):
            raise TypeError("offline signer key is not Ed25519")
        return loaded
    except (TypeError, ValueError) as error:
        raise ValueError("offline signer key encoding is invalid") from error
    finally:
        for index in range(len(body)):
            body[index] = 0


def load_ed25519_private_key_from_fd(
    descriptor: int,
    *,
    expected_public_key_sha256: str,
) -> Ed25519PrivateKey:
    """Read one bounded key from an inherited FD without discovering its path."""

    expected = _require_sha256(
        "offline signer expected public key", expected_public_key_sha256
    )
    if type(descriptor) is not int or descriptor < 0 or descriptor in {1, 2}:
        raise ValueError("offline signer key descriptor is invalid")
    duplicate = os.dup(descriptor)
    try:
        before = os.fstat(duplicate)
        regular = stat.S_ISREG(before.st_mode)
        fifo = stat.S_ISFIFO(before.st_mode)
        if not regular and not fifo:
            raise ValueError("offline signer key FD must be a regular file or pipe")
        if before.st_uid != os.geteuid():
            raise ValueError("offline signer key FD has another owner")
        if regular and (
            before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 1 <= before.st_size <= _MAXIMUM_PRIVATE_KEY_BYTES
        ):
            raise ValueError(
                "offline signer key file must be one current-user 0600 inode"
            )
        body = bytearray()
        while len(body) <= _MAXIMUM_PRIVATE_KEY_BYTES:
            chunk = os.read(
                duplicate,
                min(4096, _MAXIMUM_PRIVATE_KEY_BYTES + 1 - len(body)),
            )
            if not chunk:
                break
            body.extend(chunk)
        after = os.fstat(duplicate)
        if (
            not body
            or len(body) > _MAXIMUM_PRIVATE_KEY_BYTES
            or (
                regular
                and (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_nlink,
                    before.st_uid,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                != (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_nlink,
                    after.st_uid,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
            )
        ):
            raise RuntimeError("offline signer key changed while it was read")
        private_key = _decode_private_key(body)
        if hashlib.sha256(_public_key_bytes(private_key)).hexdigest() != expected:
            raise ValueError("offline signer key does not match the public policy")
        return private_key
    finally:
        os.close(duplicate)


def _require_safe_private_path(path: Path) -> None:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError("offline signer key path must be absolute and normalized")
    current = Path(path.anchor)
    for component in path.parts[1:-1]:
        current /= component
        status = current.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise ValueError("offline signer key path contains an unsafe ancestor")
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or stat.S_IMODE(parent.st_mode) & 0o022
    ):
        raise ValueError(
            "offline signer key parent is not private and current-user owned"
        )


def load_ed25519_private_key_from_tty(
    *,
    expected_public_key_sha256: str,
    prompt: str = "Offline Ed25519 key path: ",
) -> Ed25519PrivateKey:
    """Prompt for a key path without echoing or placing it in argv/environment."""

    if not sys.stdin.isatty() or not sys.stderr.isatty():
        raise RuntimeError(
            "offline signer requires a TTY prompt or an inherited key descriptor"
        )
    path_text = getpass.getpass(prompt=prompt, stream=sys.stderr)
    try:
        path = Path(path_text)
        _require_safe_private_path(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            return load_ed25519_private_key_from_fd(
                descriptor,
                expected_public_key_sha256=expected_public_key_sha256,
            )
        finally:
            os.close(descriptor)
    finally:
        path_text = ""


def sign_deployment_policy_authorization(
    *,
    bundle: TrustedAttesterPolicyBundle,
    inventory_sha256: str,
    challenge: AttestationChallenge,
    private_key: Ed25519PrivateKey,
    now_ns: int,
) -> DeploymentPolicyAuthorization:
    """Sign and immediately verify one source-root deployment policy."""

    if type(bundle) is not TrustedAttesterPolicyBundle:
        raise TypeError("offline deployment signer requires an exact bundle")
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("offline deployment signer requires an Ed25519 key")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("offline deployment signing time is invalid")
    inventory = _require_sha256("offline deployment inventory", inventory_sha256)
    bundle.validate(now_ns=now_ns)
    root = load_source_release_ed25519_root()
    if hashlib.sha256(_public_key_bytes(private_key)).hexdigest() != (
        root.root.public_key_sha256
    ):
        raise ValueError("offline deployment signer is not the source root")
    subject_sha256 = deployment_policy_subject_sha256(
        root_manifest_sha256=root.semantic_sha256,
        inventory_sha256=inventory,
        bundle_sha256=bundle.sha256,
    )
    challenge.validate(now_ns=now_ns)
    if (
        challenge.subject_sha256 != subject_sha256
        or challenge.expires_ns - challenge.issued_ns
        > DEPLOYMENT_POLICY_MAXIMUM_LIFETIME_NS
    ):
        raise ValueError("offline deployment challenge is not exact")
    authorization = DeploymentPolicyAuthorization(
        schema_version=1,
        kind="lightcone_deployment_policy_authorization",
        root_manifest_sha256=root.semantic_sha256,
        inventory_sha256=inventory,
        bundle=bundle,
        challenge=challenge,
        signature_base64=base64.b64encode(
            private_key.sign(
                attestation_message(challenge, payload_sha256=bundle.sha256)
            )
        ).decode("ascii"),
    )
    verify_source_signed_deployment_policy(
        authorization,
        expected_inventory_sha256=inventory,
        now_ns=now_ns,
        consumed_challenge_sha256s=(),
    )
    return authorization


def sign_control_artifact_attestation(
    *,
    subject: ControlArtifactSubject,
    hardware_envelope_sha256: str,
    deployment_policy_authorization: DeploymentPolicyAuthorization,
    challenge: AttestationChallenge,
    private_key: Ed25519PrivateKey,
    attester_id: str,
    key_id: str,
    now_ns: int,
) -> ControlArtifactAttestation:
    """Sign and immediately verify one typed local control artifact."""

    if type(subject) is not ControlArtifactSubject:
        raise TypeError("offline control signer requires an exact subject")
    if type(deployment_policy_authorization) is not DeploymentPolicyAuthorization:
        raise TypeError("offline control signer requires a deployment authorization")
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("offline control signer requires an Ed25519 key")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("offline control signing time is invalid")
    inventory = deployment_policy_authorization.inventory_sha256
    verified_deployment = verify_source_signed_deployment_policy(
        deployment_policy_authorization,
        expected_inventory_sha256=inventory,
        now_ns=now_ns,
        consumed_challenge_sha256s=(),
    )
    public_key = _public_key_bytes(private_key)
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    policy = verified_deployment.bundle.trusted_attester_policy
    expected_identity = (attester_id, key_id, public_key_sha256)
    if expected_identity not in policy.trusted_attesters:
        raise ValueError("offline control signer is not in the deployment policy")
    challenge.validate(now_ns=now_ns)
    if challenge.subject_sha256 != subject.sha256:
        raise ValueError("offline control challenge is not exact")
    if challenge.sha256 == deployment_policy_authorization.challenge.sha256:
        raise ValueError("deployment and control challenges must be distinct")
    signed = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id=attester_id,
        key_id=key_id,
        environment="release",
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        challenge_sha256=challenge.sha256,
        payload_sha256=subject.artifact_sha256,
        signature_base64=base64.b64encode(
            private_key.sign(
                attestation_message(
                    challenge,
                    payload_sha256=subject.artifact_sha256,
                )
            )
        ).decode("ascii"),
    )
    envelope = ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=_require_sha256(
            "offline control hardware envelope", hardware_envelope_sha256
        ),
        trust_anchor_sha256=verified_deployment.root_binding.sha256,
        trust_bundle_sha256=verified_deployment.bundle.sha256,
        trusted_attester_policy_sha256=policy.sha256,
        deployment_policy_authorization=deployment_policy_authorization,
        challenge=challenge,
        attestation=signed,
    )
    verify_release_control_artifact_attestation(
        envelope,
        expected_inventory_sha256=inventory,
        now_ns=now_ns,
        consumed_challenge_sha256s=(),
    )
    return envelope


def _load_public_json(path: str) -> dict:
    return CanonicalJsonProofBinding.bind(path).reopen()


def _private_key(
    args: argparse.Namespace, *, expected_sha256: str
) -> Ed25519PrivateKey:
    if args.key_fd is None:
        return load_ed25519_private_key_from_tty(
            expected_public_key_sha256=expected_sha256
        )
    return load_ed25519_private_key_from_fd(
        args.key_fd,
        expected_public_key_sha256=expected_sha256,
    )


def _now_ns(value: int | None) -> int:
    current = time.time_ns() if value is None else value
    if type(current) is not int or current < 0:
        raise ValueError("offline signer current time is invalid")
    return current


def _lifetime_seconds(value: float) -> float:
    if not 1.0 <= value <= DEPLOYMENT_POLICY_MAXIMUM_LIFETIME_NS / 1_000_000_000:
        raise ValueError("offline signer lifetime is outside release bounds")
    return value


def _sign_deployment_cli(args: argparse.Namespace) -> int:
    now_ns = _now_ns(args.now_ns)
    bundle = TrustedAttesterPolicyBundle.from_dict(_load_public_json(args.bundle))
    root = load_source_release_ed25519_root()
    subject = deployment_policy_subject_sha256(
        root_manifest_sha256=root.semantic_sha256,
        inventory_sha256=args.inventory_sha256,
        bundle_sha256=bundle.sha256,
    )
    challenge = AttestationChallenge.issue(
        challenge_id=args.challenge_id,
        subject_sha256=subject,
        lifetime_s=_lifetime_seconds(args.lifetime_seconds),
        now_ns=now_ns,
    )
    authorization = sign_deployment_policy_authorization(
        bundle=bundle,
        inventory_sha256=args.inventory_sha256,
        challenge=challenge,
        private_key=_private_key(args, expected_sha256=root.root.public_key_sha256),
        now_ns=now_ns,
    )
    publish_canonical_json_no_replace(args.output, authorization.to_dict())
    print(authorization.sha256)
    return 0


def _sign_control_cli(args: argparse.Namespace) -> int:
    now_ns = _now_ns(args.now_ns)
    subject = ControlArtifactSubject.from_dict(_load_public_json(args.subject))
    authorization = DeploymentPolicyAuthorization.from_dict(
        _load_public_json(args.deployment_authorization)
    )
    policy = authorization.bundle.trusted_attester_policy
    matches = tuple(
        row
        for row in policy.trusted_attesters
        if row[0] == args.attester_id and row[1] == args.key_id
    )
    if len(matches) != 1:
        raise ValueError("offline control attester identity is not uniquely authorized")
    challenge = AttestationChallenge.issue(
        challenge_id=args.challenge_id,
        subject_sha256=subject.sha256,
        lifetime_s=_lifetime_seconds(args.lifetime_seconds),
        now_ns=now_ns,
    )
    envelope = sign_control_artifact_attestation(
        subject=subject,
        hardware_envelope_sha256=args.hardware_envelope_sha256,
        deployment_policy_authorization=authorization,
        challenge=challenge,
        private_key=_private_key(args, expected_sha256=matches[0][2]),
        attester_id=args.attester_id,
        key_id=args.key_id,
        now_ns=now_ns,
    )
    publish_canonical_json_no_replace(args.output, envelope.to_dict())
    print(envelope.sha256)
    return 0


def _sign_content_authorization_cli(args: argparse.Namespace) -> int:
    """Sign one member of the closed typed content-authorization allowlist."""

    now_ns = _now_ns(args.now_ns)
    source = content_authorization_source_from_dict(
        args.artifact_type,
        _load_public_json(args.source),
    )
    root = load_source_release_ed25519_root()
    if source.root_manifest_sha256 != root.semantic_sha256:
        raise ValueError("content authorization source names another release root")
    challenge = AttestationChallenge.issue(
        challenge_id=args.challenge_id,
        subject_sha256=source.subject_sha256,
        lifetime_s=_lifetime_seconds(args.lifetime_seconds),
        now_ns=now_ns,
    )
    private_key = _private_key(
        args,
        expected_sha256=root.root.public_key_sha256,
    )
    authorization = build_content_authorization_from_source(
        source,
        challenge=challenge,
        signature_base64=base64.b64encode(
            private_key.sign(
                attestation_message(
                    challenge,
                    payload_sha256=source.subject_sha256,
                )
            )
        ).decode("ascii"),
    )
    if type(authorization) is ReleaseWorkloadSourceAuthorization:
        verified = verify_release_workload_source_authorization(
            authorization,
            now_ns=now_ns,
        )
    elif type(authorization) is PreparedModelContentReleaseAuthorization:
        verified = verify_prepared_model_content_release_authorization(
            authorization,
            now_ns=now_ns,
        )
    elif type(authorization) is DatasetContentReleaseAuthorization:
        verified = verify_dataset_content_release_authorization(
            authorization,
            expected_authority_domain=authorization.authority_domain,
            now_ns=now_ns,
        )
    else:  # pragma: no cover - closed typed decoder makes this unreachable
        raise TypeError("content authorization ceremony produced another type")
    if verified.authorization_sha256 != authorization.sha256:
        raise RuntimeError("content authorization verification changed identity")
    publish_canonical_json_no_replace(args.output, authorization.to_dict())
    print(authorization.sha256)
    return 0


def _authorized_signer_identity(
    authorization: DeploymentPolicyAuthorization,
    *,
    attester_id: str,
    key_id: str,
) -> tuple[str, str, str]:
    policy = authorization.bundle.trusted_attester_policy
    matches = tuple(
        row
        for row in policy.trusted_attesters
        if row[0] == attester_id and row[1] == key_id
    )
    if len(matches) != 1:
        raise ValueError(
            "offline scientific signer identity is not uniquely authorized"
        )
    return matches[0]


def _sign_scientific_cli(args: argparse.Namespace) -> int:
    now_ns = _now_ns(args.now_ns)
    if args.payload is None:
        if args.source_validation_artifact is None:
            raise ValueError("scientific signing requires a payload or proof source")
        payload_json = scientific_payload_json_from_source(
            artifact_type=args.artifact_type,
            source_validation_artifact_path=args.source_validation_artifact,
            now_ns=now_ns,
        )
    else:
        payload_json = _load_public_json(args.payload)
    authorization = DeploymentPolicyAuthorization.from_dict(
        _load_public_json(args.deployment_authorization)
    )
    identity = _authorized_signer_identity(
        authorization,
        attester_id=args.attester_id,
        key_id=args.key_id,
    )
    payload_sha256 = scientific_payload_sha256(
        artifact_type=args.artifact_type,
        payload_json=payload_json,
        source_validation_artifact_path=args.source_validation_artifact,
        now_ns=now_ns,
    )
    challenge = AttestationChallenge.issue(
        challenge_id=args.challenge_id,
        subject_sha256=payload_sha256,
        lifetime_s=_lifetime_seconds(args.lifetime_seconds),
        now_ns=now_ns,
    )
    candidate = sign_scientific_candidate(
        artifact_type=args.artifact_type,
        payload_json=payload_json,
        deployment_policy_authorization=authorization,
        challenge=challenge,
        private_key=_private_key(args, expected_sha256=identity[2]),
        attester_id=args.attester_id,
        key_id=args.key_id,
        now_ns=now_ns,
        source_validation_artifact_path=args.source_validation_artifact,
    )
    publish_canonical_json_no_replace(args.output, candidate)
    print(candidate["candidate_sha256"])
    return 0


def _finalize_scientific_cli(args: argparse.Namespace) -> int:
    now_ns = _now_ns(args.now_ns)
    candidate = _load_public_json(args.candidate)
    authorization = DeploymentPolicyAuthorization.from_dict(
        _load_public_json(args.deployment_authorization)
    )
    signed_artifact = finalize_scientific_candidate(
        artifact_type=args.artifact_type,
        candidate_json=candidate,
        deployment_policy_authorization=authorization,
        challenge_ledger=args.challenge_ledger,
        now_ns=now_ns,
    )
    publish_canonical_json_no_replace(args.output, signed_artifact)
    print(candidate["signed_artifact_sha256"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lightcone_spec.runtime.offline_signer",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    deployment = commands.add_parser("sign-deployment", allow_abbrev=False)
    deployment.add_argument("--bundle", required=True)
    deployment.add_argument("--inventory-sha256", required=True)
    deployment.add_argument("--challenge-id", required=True)
    deployment.add_argument("--lifetime-seconds", type=float, default=300.0)
    deployment.add_argument("--now-ns", type=int)
    deployment.add_argument("--key-fd", type=int)
    deployment.add_argument("--output", required=True)
    deployment.set_defaults(handler=_sign_deployment_cli)

    control = commands.add_parser("sign-control", allow_abbrev=False)
    control.add_argument("--subject", required=True)
    control.add_argument("--deployment-authorization", required=True)
    control.add_argument("--hardware-envelope-sha256", required=True)
    control.add_argument("--attester-id", required=True)
    control.add_argument("--key-id", required=True)
    control.add_argument("--challenge-id", required=True)
    control.add_argument("--lifetime-seconds", type=float, default=300.0)
    control.add_argument("--now-ns", type=int)
    control.add_argument("--key-fd", type=int)
    control.add_argument("--output", required=True)
    control.set_defaults(handler=_sign_control_cli)

    content = commands.add_parser("sign-content-authorization", allow_abbrev=False)
    content.add_argument(
        "--artifact-type",
        required=True,
        choices=CONTENT_AUTHORIZATION_SOURCE_TYPES,
    )
    content.add_argument("--source", required=True)
    content.add_argument("--challenge-id", required=True)
    content.add_argument("--lifetime-seconds", type=float, default=300.0)
    content.add_argument("--now-ns", type=int)
    content.add_argument("--key-fd", type=int)
    content.add_argument("--output", required=True)
    content.set_defaults(handler=_sign_content_authorization_cli)

    scientific = commands.add_parser("sign-scientific", allow_abbrev=False)
    scientific.add_argument(
        "--artifact-type", required=True, choices=SCIENTIFIC_ARTIFACT_TYPES
    )
    scientific.add_argument("--payload")
    scientific.add_argument(
        "--source-validation-artifact",
        help=(
            "mandatory path-bound reducer validation for proof-derived stage "
            "authorities"
        ),
    )
    scientific.add_argument("--deployment-authorization", required=True)
    scientific.add_argument("--attester-id", required=True)
    scientific.add_argument("--key-id", required=True)
    scientific.add_argument("--challenge-id", required=True)
    scientific.add_argument("--lifetime-seconds", type=float, default=300.0)
    scientific.add_argument("--now-ns", type=int)
    scientific.add_argument("--key-fd", type=int)
    scientific.add_argument("--output", required=True)
    scientific.set_defaults(handler=_sign_scientific_cli)

    finalize = commands.add_parser("finalize-scientific", allow_abbrev=False)
    finalize.add_argument(
        "--artifact-type", required=True, choices=SCIENTIFIC_ARTIFACT_TYPES
    )
    finalize.add_argument("--candidate", required=True)
    finalize.add_argument("--deployment-authorization", required=True)
    finalize.add_argument("--challenge-ledger", required=True)
    finalize.add_argument("--now-ns", type=int)
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(handler=_finalize_scientific_cli)
    return parser


def _abort(message: str) -> NoReturn:
    # Never include an exception representation here: a platform exception may
    # contain the private path entered at the hidden prompt.
    raise SystemExit(message)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, RuntimeError, TypeError, ValueError):
        _abort("offline signing failed closed; no private material was logged")


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main())


__all__ = [
    "finalize_scientific_candidate",
    "load_ed25519_private_key_from_fd",
    "load_ed25519_private_key_from_tty",
    "main",
    "sign_control_artifact_attestation",
    "sign_deployment_policy_authorization",
    "sign_scientific_candidate",
]
