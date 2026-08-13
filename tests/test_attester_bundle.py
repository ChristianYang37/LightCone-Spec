from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import lightcone_spec.runtime.attester_bundle as bundle_module
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.attester_bundle import (
    SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR,
    AttestationNoncePolicy,
    LoadedTrustedAttesterPolicyBundle,
    TrustedAttesterAnchorDescriptor,
    TrustedAttesterPolicyBundle,
    TrustedAttesterPolicyBundleBinding,
    load_source_release_trusted_attester_policy_bundle,
    load_trusted_attester_policy_bundle,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _semantic_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _policy() -> TrustedAttesterPolicy:
    public_key = bytes(range(32))
    public_key_base64 = base64.b64encode(public_key).decode("ascii")
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    return TrustedAttesterPolicy(
        policy_id="release-hardware-attesters-v1",
        trusted_attesters=(("prod-hsm-1", "release-key-1", public_key_sha256),),
        public_keys=((public_key_sha256, public_key_base64),),
    )


def _bundle(
    *,
    valid_from_ns: int = 100,
    expires_ns: int = 10_000,
) -> TrustedAttesterPolicyBundle:
    return TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="release-attester-bundle-v1",
        valid_from_ns=valid_from_ns,
        expires_ns=expires_ns,
        nonce_policy=AttestationNoncePolicy(
            schema_version=1,
            kind="lightcone_attestation_nonce_policy",
            nonce_bytes=32,
            minimum_lifetime_ns=100,
            maximum_lifetime_ns=1_000,
            maximum_clock_skew_ns=10,
            replay_policy="external_single_use_store",
            subject_binding_required=True,
        ),
        hardware_envelope_sha256_allowlist=(SHA_A, SHA_B),
        trusted_attester_policy=_policy(),
    )


def _write_value(path: Path, value: object, *, canonical: bool = True) -> str:
    if canonical:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    else:
        encoded = json.dumps(value, indent=2, ensure_ascii=False)
    path.write_text(encoded + "\n", encoding="utf-8")
    digest = _semantic_sha256(value)
    Path(f"{path}.sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _write_bundle(path: Path, bundle: TrustedAttesterPolicyBundle) -> None:
    path.write_bytes(bundle.canonical_bytes)
    Path(f"{path}.sha256").write_text(bundle.sha256 + "\n", encoding="ascii")


def _anchor(
    path: Path,
    bundle_sha256: str,
    *,
    authority: str = "operator_configuration",
    valid_from_ns: int = 0,
    expires_ns: int = 20_000,
) -> TrustedAttesterAnchorDescriptor:
    return TrustedAttesterAnchorDescriptor(
        schema_version=1,
        kind="lightcone_trusted_attester_anchor",
        anchor_id="operator-release-anchor-v1",
        authority=authority,
        bundle_path=str(path),
        bundle_sha256=bundle_sha256,
        valid_from_ns=valid_from_ns,
        expires_ns=expires_ns,
    )


def test_operator_anchor_loads_canonical_public_bundle_and_reopens(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trusted-attesters.json"
    bundle = _bundle()
    _write_bundle(path, bundle)
    anchor = _anchor(path, bundle.sha256)

    loaded = load_trusted_attester_policy_bundle(anchor, now_ns=500)

    assert type(loaded) is LoadedTrustedAttesterPolicyBundle
    assert type(loaded.binding) is TrustedAttesterPolicyBundleBinding
    assert loaded.bundle == bundle
    assert loaded.policy == bundle.trusted_attester_policy
    assert loaded.binding.anchor_sha256 == anchor.sha256
    assert loaded.binding.bundle_sha256 == bundle.sha256
    assert loaded.binding.path == str(path)
    assert loaded.binding.sidecar_path == f"{path}.sha256"
    assert loaded.reopen(now_ns=501) == loaded
    bundle.require_hardware_envelope(SHA_A)
    with pytest.raises(ValueError, match="not allowlisted"):
        bundle.require_hardware_envelope("c" * 64)


def test_release_loader_has_no_anchor_and_fails_closed() -> None:
    assert SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR is None
    with pytest.raises(RuntimeError, match="anchor is unavailable"):
        load_source_release_trusted_attester_policy_bundle(now_ns=500)


def test_source_loader_rejects_an_operator_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "trusted-attesters.json"
    bundle = _bundle()
    _write_bundle(path, bundle)
    monkeypatch.setattr(
        bundle_module,
        "SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR",
        _anchor(path, bundle.sha256),
    )
    with pytest.raises(RuntimeError, match="anchor is malformed"):
        load_source_release_trusted_attester_policy_bundle(now_ns=500)


def test_caller_cannot_pair_a_bundle_path_with_an_arbitrary_source_anchor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trusted-attesters.json"
    bundle = _bundle()
    _write_bundle(path, bundle)
    forged_source_anchor = _anchor(
        path,
        bundle.sha256,
        authority="source_release",
    )
    with pytest.raises(RuntimeError, match="anchor is unavailable"):
        load_trusted_attester_policy_bundle(forged_source_anchor, now_ns=500)

    parameters = inspect.signature(load_trusted_attester_policy_bundle).parameters
    assert tuple(parameters) == ("anchor", "now_ns")
    with pytest.raises(TypeError, match="unexpected keyword"):
        load_trusted_attester_policy_bundle(
            forged_source_anchor,
            now_ns=500,
            bundle_path=str(path),  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("private_key_base64", "do-not-load", "private material"),
        ("signing_seed", "do-not-load", "private material"),
        ("test_identity", "fixture", "test identities"),
    ),
)
def test_private_and_test_material_is_categorically_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = tmp_path / f"forbidden-{field}.json"
    raw = _bundle().to_dict()
    raw["trusted_attester_policy"][field] = value
    digest = _write_value(path, raw)
    with pytest.raises(ValueError, match=message):
        load_trusted_attester_policy_bundle(_anchor(path, digest), now_ns=500)


@pytest.mark.parametrize(
    ("location", "identity"),
    (
        ("bundle", "test-release-bundle"),
        ("policy", "fixture-release-policy"),
        ("attester", "cpu-attester"),
        ("key", "test-key"),
    ),
)
def test_test_fixture_and_cpu_identities_cannot_enter_a_bundle(
    tmp_path: Path,
    location: str,
    identity: str,
) -> None:
    path = tmp_path / f"forbidden-{location}.json"
    raw = _bundle().to_dict()
    if location == "bundle":
        raw["bundle_id"] = identity
    elif location == "policy":
        raw["trusted_attester_policy"]["policy_id"] = identity
    else:
        index = 0 if location == "attester" else 1
        raw["trusted_attester_policy"]["trusted_attesters"][0][index] = identity
    digest = _write_value(path, raw)
    with pytest.raises(ValueError, match="release identity|policy entries"):
        load_trusted_attester_policy_bundle(_anchor(path, digest), now_ns=500)


def test_nested_policy_schema_version_rejects_boolean_alias(tmp_path: Path) -> None:
    path = tmp_path / "boolean-schema.json"
    raw = _bundle().to_dict()
    raw["trusted_attester_policy"]["schema_version"] = True
    digest = _write_value(path, raw)
    with pytest.raises(ValueError, match="policy schema is unsupported"):
        load_trusted_attester_policy_bundle(_anchor(path, digest), now_ns=500)


def test_bundle_and_anchor_validity_are_both_enforced(tmp_path: Path) -> None:
    path = tmp_path / "trusted-attesters.json"
    bundle = _bundle(valid_from_ns=100, expires_ns=1_000)
    _write_bundle(path, bundle)
    anchor = _anchor(path, bundle.sha256, valid_from_ns=0, expires_ns=2_000)

    with pytest.raises(ValueError, match="not yet valid"):
        load_trusted_attester_policy_bundle(anchor, now_ns=99)
    with pytest.raises(ValueError, match="bundle expired"):
        load_trusted_attester_policy_bundle(anchor, now_ns=1_001)
    with pytest.raises(ValueError, match="validity exceeds"):
        load_trusted_attester_policy_bundle(
            replace(anchor, valid_from_ns=200),
            now_ns=500,
        )
    with pytest.raises(ValueError, match="anchor expired"):
        load_trusted_attester_policy_bundle(
            replace(anchor, expires_ns=400),
            now_ns=500,
        )


@pytest.mark.parametrize(
    "allowlist",
    (
        (),
        (SHA_A, SHA_A),
        (SHA_A.upper(),),
        (SHA_B, SHA_A),
    ),
)
def test_hardware_envelope_allowlist_is_exact_sorted_and_unique(
    tmp_path: Path,
    allowlist: tuple[str, ...],
) -> None:
    path = tmp_path / f"bad-envelope-{len(allowlist)}.json"
    raw = _bundle().to_dict()
    raw["hardware_envelope_sha256_allowlist"] = list(allowlist)
    digest = _write_value(path, raw)
    with pytest.raises(ValueError, match="allowlist|lowercase SHA-256"):
        load_trusted_attester_policy_bundle(_anchor(path, digest), now_ns=500)


def test_nonce_policy_enforces_lifetime_clock_skew_and_replay() -> None:
    bundle = _bundle()
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="release-challenge-1",
        nonce_base64=base64.b64encode(b"n" * 32).decode("ascii"),
        subject_sha256=SHA_A,
        issued_ns=500,
        expires_ns=800,
    )
    challenge_sha256 = bundle.validate_challenge(
        challenge,
        now_ns=600,
        consumed_challenge_sha256s=(),
    )
    assert challenge_sha256 == challenge.sha256

    with pytest.raises(ValueError, match="already consumed"):
        bundle.validate_challenge(
            challenge,
            now_ns=600,
            consumed_challenge_sha256s=(challenge.sha256,),
        )
    with pytest.raises(ValueError, match="lifetime violates"):
        bundle.validate_challenge(
            replace(challenge, expires_ns=501),
            now_ns=500,
            consumed_challenge_sha256s=(),
        )
    with pytest.raises(ValueError, match="too far in the future"):
        bundle.validate_challenge(
            replace(challenge, issued_ns=620, expires_ns=920),
            now_ns=600,
            consumed_challenge_sha256s=(),
        )
    with pytest.raises(TypeError, match="digest collection"):
        bundle.validate_challenge(
            challenge,
            now_ns=600,
            consumed_challenge_sha256s=SHA_B,
        )


@pytest.mark.parametrize(
    "change",
    (
        {"nonce_bytes": 16},
        {"minimum_lifetime_ns": 0},
        {"maximum_lifetime_ns": 99},
        {"maximum_clock_skew_ns": 1_001},
        {"replay_policy": "none"},
        {"subject_binding_required": False},
    ),
)
def test_weak_nonce_policies_are_rejected(change: dict[str, object]) -> None:
    policy = _bundle().nonce_policy
    with pytest.raises(ValueError, match="nonce|replay|subject"):
        replace(policy, **change).validate()


def test_noncanonical_bytes_and_sidecar_mismatch_are_rejected(
    tmp_path: Path,
) -> None:
    pretty_path = tmp_path / "pretty.json"
    raw = _bundle().to_dict()
    digest = _write_value(pretty_path, raw, canonical=False)
    with pytest.raises(ValueError, match="bytes are not canonical"):
        load_trusted_attester_policy_bundle(
            _anchor(pretty_path, digest),
            now_ns=500,
        )

    mismatch_path = tmp_path / "mismatch.json"
    bundle = _bundle()
    _write_bundle(mismatch_path, bundle)
    Path(f"{mismatch_path}.sha256").write_text(SHA_A + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="sidecar differs"):
        load_trusted_attester_policy_bundle(
            _anchor(mismatch_path, bundle.sha256),
            now_ns=500,
        )


def test_bundle_and_sidecar_reads_reject_symlinks_and_hardlinks(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    target = tmp_path / "target.json"
    _write_bundle(target, bundle)

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="readable regular file"):
        load_trusted_attester_policy_bundle(
            _anchor(symlink, bundle.sha256),
            now_ns=500,
        )

    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    Path(f"{hardlink}.sha256").write_text(bundle.sha256 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="single-link regular file"):
        load_trusted_attester_policy_bundle(
            _anchor(hardlink, bundle.sha256),
            now_ns=500,
        )


def test_binding_reopen_detects_replaced_bundle_bytes(tmp_path: Path) -> None:
    path = tmp_path / "trusted-attesters.json"
    bundle = _bundle()
    _write_bundle(path, bundle)
    loaded = load_trusted_attester_policy_bundle(
        _anchor(path, bundle.sha256),
        now_ns=500,
    )

    replacement = replace(bundle, bundle_id="release-attester-bundle-v2")
    _write_bundle(path, replacement)
    with pytest.raises(RuntimeError, match="changed after load"):
        loaded.reopen(now_ns=501)


def test_anchor_descriptor_rejects_relative_paths_and_test_identity(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    path = tmp_path / "trusted-attesters.json"
    _write_bundle(path, bundle)
    anchor = _anchor(path, bundle.sha256)

    with pytest.raises(ValueError, match="absolute and normalized"):
        replace(anchor, bundle_path="trusted-attesters.json").validate()
    with pytest.raises(ValueError, match="non-test release identity"):
        replace(anchor, anchor_id="test-anchor").validate()
    with pytest.raises(TypeError, match="exact anchor descriptor"):
        load_trusted_attester_policy_bundle(None, now_ns=500)  # type: ignore[arg-type]
