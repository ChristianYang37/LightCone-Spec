from __future__ import annotations

import os
from pathlib import Path

import pytest

from lightcone_spec.experiments.capacity_authority import (
    bind_capacity_raw_json,
    load_capacity_raw_json,
)
from lightcone_spec.runtime.content_authorization import ContentJsonArtifactBinding
from lightcone_spec.runtime.control_attestation import ChallengeReplayStore
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
    relocated_evidence_path,
)
from lightcone_spec.runtime.relocatable_evidence import (
    RelocatableEvidenceMember,
    activate_relocatable_evidence_bundle,
    materialize_relocatable_evidence_bundle,
    validate_relocatable_evidence_bundle,
)


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700)
    return path.resolve()


def _remote_fixture(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    root = _private(tmp_path / "remote-root-A")
    nested = root / "members"
    nested.mkdir(mode=0o700)
    leaf = nested / "native-itl.json"
    publish_canonical_json_no_replace(
        leaf,
        {"schema_version": 1, "kind": "native-itl", "events": [11, 12]},
    )
    leaf_binding = CanonicalJsonProofBinding.bind(leaf)
    entries = []
    for mode in ("tp1", "tp2-dp2", "e5"):
        path = root / f"{mode}-raw.json"
        publish_canonical_json_no_replace(
            path,
            {
                "schema_version": 1,
                "kind": f"formal-{mode}-raw",
                "nested_native_itl": leaf_binding.to_dict(),
            },
        )
        entries.append(path)
    return root, tuple(entries)


def test_remote_root_a_reopens_from_distinct_local_root_b(tmp_path: Path) -> None:
    remote_root, entries = _remote_fixture(tmp_path)
    remote_bindings = tuple(CanonicalJsonProofBinding.bind(path) for path in entries)
    local_root = _private(tmp_path / "stable-pull-root-B")
    manifest = materialize_relocatable_evidence_bundle(
        remote_root=remote_root,
        entry_paths=entries,
        local_root=local_root,
    )
    moved_root = tmp_path / "remote-root-A-offline"
    remote_root.rename(moved_root)
    assert all(not Path(binding.absolute_path).exists() for binding in remote_bindings)

    with activate_relocatable_evidence_bundle(manifest.absolute_path) as bundle:
        assert Path(bundle.local_root) == local_root
        assert len(bundle.artifact.members) == 4
        for binding, mode in zip(
            remote_bindings, ("tp1", "tp2-dp2", "e5"), strict=True
        ):
            assert binding.reopen()["kind"] == f"formal-{mode}-raw"
            rebound = CanonicalJsonProofBinding.bind(binding.absolute_path)
            assert rebound == binding
            nested = CanonicalJsonProofBinding.from_dict(
                rebound.reopen()["nested_native_itl"]
            )
            assert nested.reopen()["events"] == [11, 12]


@pytest.mark.parametrize("mutation", ("missing", "extra", "symlink", "alias"))
def test_bundle_rejects_missing_extra_symlink_and_alias(
    tmp_path: Path, mutation: str
) -> None:
    remote_root, entries = _remote_fixture(tmp_path)
    local_root = _private(tmp_path / "local-root")
    manifest = materialize_relocatable_evidence_bundle(
        remote_root=remote_root,
        entry_paths=entries,
        local_root=local_root,
    )
    validated = validate_relocatable_evidence_bundle(manifest.absolute_path)
    member_paths = tuple(
        local_root / row.relative_path for row in validated.artifact.members
    )
    if mutation == "missing":
        member_paths[0].unlink()
    elif mutation == "extra":
        publish_canonical_json_no_replace(
            local_root / "unexpected.json", {"kind": "unexpected"}
        )
    elif mutation == "symlink":
        member_paths[0].unlink()
        member_paths[0].symlink_to(member_paths[1])
    else:
        member_paths[1].unlink()
        os.link(member_paths[0], member_paths[1])
    with pytest.raises(
        ValueError, match="missing|extra|symlink|aliases|identity|unsafe"
    ):
        validate_relocatable_evidence_bundle(manifest.absolute_path)


def test_bundle_rejects_traversal_and_secret_members(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="traverses"):
        RelocatableEvidenceMember(
            remote_absolute_path="/remote/escape",
            relative_path="../escape",
            raw_sha256="1" * 64,
            semantic_sha256="2" * 64,
            size=2,
        )
    remote_root = _private(tmp_path / "remote-root")
    secret = remote_root / "receipt.json"
    publish_canonical_json_no_replace(
        secret,
        {"kind": "bad-receipt", "private_key": "must-not-move"},
    )
    local_root = _private(tmp_path / "local-root")
    with pytest.raises(ValueError, match="secret"):
        materialize_relocatable_evidence_bundle(
            remote_root=remote_root,
            entry_paths=(secret,),
            local_root=local_root,
        )


@pytest.mark.parametrize(
    "secret_value",
    (
        "-----BEGIN " + "PRIVATE" + " KEY-----\nnot-a-real-key",
        "h" + "f_" + "abcdefghijklmnopqrstuvwxyz012345",
        "s" + "k-" + "abcdefghijklmnopqrstuvwxyz012345",
        "eyJabcdefghijk.abcdefghijkl.abcdefghijkl",
    ),
)
def test_bundle_rejects_secret_shaped_values(tmp_path: Path, secret_value: str) -> None:
    remote_root = _private(tmp_path / "remote")
    receipt = remote_root / "receipt.json"
    publish_canonical_json_no_replace(
        receipt, {"kind": "bad-receipt", "opaque_value": secret_value}
    )
    with pytest.raises(ValueError, match="secret-shaped"):
        materialize_relocatable_evidence_bundle(
            remote_root=remote_root,
            entry_paths=(receipt,),
            local_root=_private(tmp_path / "local"),
        )


def test_bundle_validation_is_read_only_when_ancestor_is_missing(
    tmp_path: Path,
) -> None:
    remote_root, entries = _remote_fixture(tmp_path)
    local_root = _private(tmp_path / "local")
    manifest = materialize_relocatable_evidence_bundle(
        remote_root=remote_root,
        entry_paths=entries,
        local_root=local_root,
    )
    nested = local_root / "members"
    (nested / "native-itl.json").unlink()
    nested.rmdir()
    with pytest.raises(ValueError, match="missing"):
        validate_relocatable_evidence_bundle(manifest.absolute_path)
    assert not nested.exists()


def test_closed_raw_binding_union_reopens_after_remote_root_disappears(
    tmp_path: Path,
) -> None:
    remote = _private(tmp_path / "remote-A")
    raw = remote / "qualification.junit.xml"
    raw.write_bytes(b"<testsuite tests='1' failures='0'/>\n")
    raw_binding = EvidenceFileBinding.bind(raw, label="qualification JUnit")

    content_path = remote / "content.json"
    publish_canonical_json_no_replace(content_path, {"kind": "content", "rows": [1]})
    content_binding = ContentJsonArtifactBinding.from_path("content:test", content_path)

    capacity_path = remote / "capacity.json"
    capacity_value = {"kind": "capacity", "free_bytes": 123}
    publish_canonical_json_no_replace(capacity_path, capacity_value)
    capacity_semantic = CanonicalJsonProofBinding.bind(capacity_path).semantic_sha256
    Path(f"{capacity_path}.sha256").write_text(
        f"{capacity_semantic}\n", encoding="ascii"
    )
    capacity_binding = bind_capacity_raw_json(capacity_path)

    replay_root = remote / "replay"
    replay_root.mkdir(mode=0o700)
    replay = ChallengeReplayStore(str(replay_root))
    reservation = replay.reserve_verified_content_challenges(
        ("1" * 64,), reserved_ns=11
    )

    entry = remote / "terminal.json"
    publish_canonical_json_no_replace(
        entry,
        {
            "kind": "formal-terminal",
            "raw_junit": raw_binding.to_dict(),
            "content": content_binding.to_dict(),
            "capacity": capacity_binding.to_dict(),
            "reservation": reservation.to_dict(),
        },
    )
    local = _private(tmp_path / "local-B")
    manifest = materialize_relocatable_evidence_bundle(
        remote_root=remote,
        entry_paths=(entry,),
        local_root=local,
    )
    remote.rename(tmp_path / "remote-A-offline")

    with activate_relocatable_evidence_bundle(manifest.absolute_path):
        raw_binding.reopen(label="qualification JUnit")
        assert content_binding.load() == {"kind": "content", "rows": [1]}
        assert load_capacity_raw_json(capacity_binding) == capacity_value
        assert reservation.revalidate() == ("1" * 64,)


def test_unknown_absolute_path_binding_fails_closed(tmp_path: Path) -> None:
    remote = _private(tmp_path / "remote")
    payload = remote / "payload.bin"
    payload.write_bytes(b"payload")
    entry = remote / "entry.json"
    publish_canonical_json_no_replace(
        entry,
        {"kind": "unknown-binding", "mystery_path": str(payload)},
    )
    with pytest.raises(ValueError, match="unknown path-bearing"):
        materialize_relocatable_evidence_bundle(
            remote_root=remote,
            entry_paths=(entry,),
            local_root=_private(tmp_path / "local"),
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "tamper", "symlink"))
def test_rehydrated_snapshot_rebind_is_exact_and_revalidated(
    tmp_path: Path,
    mutation: str,
) -> None:
    remote = _private(tmp_path / "remote")
    remote_snapshot = (remote / "models--fixture" / "snapshots" / ("1" * 40)).resolve()
    entry = remote / "entry.json"
    publish_canonical_json_no_replace(
        entry,
        {
            "kind": "prepared-snapshot-source",
            "snapshot": {
                "model_id": "fixture/model",
                "revision": "1" * 40,
                "root": str(remote_snapshot),
            },
        },
    )
    hydrated_root = _private(tmp_path / "hydrated")
    (hydrated_root / "snapshots").mkdir(mode=0o700)
    hydrated = _private(hydrated_root / "snapshots" / ("1" * 40))
    model_file = hydrated / "config.json"
    model_file.write_bytes(b'{"model":"fixture"}\n')
    local = _private(tmp_path / "local")
    manifest = materialize_relocatable_evidence_bundle(
        remote_root=remote,
        entry_paths=(entry,),
        local_root=local,
        directory_rebindings={remote_snapshot: hydrated},
    )
    with activate_relocatable_evidence_bundle(manifest.absolute_path):
        assert relocated_evidence_path(remote_snapshot) == hydrated
    if mutation == "missing":
        model_file.unlink()
    elif mutation == "extra":
        (hydrated / "foreign.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "tamper":
        model_file.write_bytes(b'{"model":"altered"}\n')
    else:
        model_file.unlink()
        model_file.symlink_to(hydrated / "missing-target")
    with pytest.raises(ValueError, match="unavailable|changed|unsafe|readable|empty"):
        validate_relocatable_evidence_bundle(manifest.absolute_path)


def test_rehydrated_snapshot_rebind_requires_exact_remote_universe(
    tmp_path: Path,
) -> None:
    remote = _private(tmp_path / "remote")
    expected = (remote / "models" / "snapshots" / ("1" * 40)).resolve()
    entry = remote / "entry.json"
    publish_canonical_json_no_replace(
        entry,
        {
            "snapshot": {
                "model_id": "fixture/model",
                "revision": "1" * 40,
                "root": str(expected),
            }
        },
    )
    hydrated = _private(tmp_path / "hydrated")
    (hydrated / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ from prepared snapshots"):
        materialize_relocatable_evidence_bundle(
            remote_root=remote,
            entry_paths=(entry,),
            local_root=_private(tmp_path / "local"),
            directory_rebindings={Path(f"{expected}-foreign"): hydrated},
        )


def test_member_cap_rejects_oversize_and_accepts_bounded_chunk_index(
    tmp_path: Path,
) -> None:
    remote = _private(tmp_path / "remote")
    oversized = remote / "oversized.log"
    oversized.write_bytes(b"x" * ((2 * 1024 * 1024) + 1))
    oversized_binding = EvidenceFileBinding.bind(oversized, label="oversized log")
    bad_entry = remote / "bad-entry.json"
    publish_canonical_json_no_replace(
        bad_entry, {"kind": "raw-log-index", "chunks": [oversized_binding.to_dict()]}
    )
    with pytest.raises(ValueError, match="bounded|cap"):
        materialize_relocatable_evidence_bundle(
            remote_root=remote,
            entry_paths=(bad_entry,),
            local_root=_private(tmp_path / "bad-local"),
        )

    chunks = []
    for index in range(2):
        path = remote / f"chunk-{index}.log"
        path.write_bytes((f"chunk-{index}\n").encode("ascii"))
        chunks.append(EvidenceFileBinding.bind(path, label=f"log chunk {index}"))
    index_path = remote / "chunk-index.json"
    publish_canonical_json_no_replace(
        index_path,
        {
            "kind": "bounded-log-chunk-index",
            "chunks": [row.to_dict() for row in chunks],
        },
    )
    manifest = materialize_relocatable_evidence_bundle(
        remote_root=remote,
        entry_paths=(index_path,),
        local_root=_private(tmp_path / "good-local"),
    )
    assert (
        len(
            validate_relocatable_evidence_bundle(
                manifest.absolute_path
            ).artifact.members
        )
        == 3
    )


def test_rehydrated_directory_accepts_an_immutable_zero_byte_member(
    tmp_path: Path,
) -> None:
    remote = _private(tmp_path / "remote-evidence")
    remote_snapshot = (tmp_path / "remote-snapshot-A").resolve()
    entry = remote / "prepared-pointer.json"
    publish_canonical_json_no_replace(
        entry,
        {
            "kind": "prepared-snapshot-pointer",
            "snapshot": {
                "model_id": "example/model",
                "revision": "a" * 40,
                "root": str(remote_snapshot),
            },
        },
    )
    rehydrated = _private(tmp_path / "rehydrated-snapshot-B")
    (rehydrated / "empty-tokenizer-sidecar").write_bytes(b"")
    manifest = materialize_relocatable_evidence_bundle(
        remote_root=remote,
        entry_paths=(entry,),
        local_root=_private(tmp_path / "local-evidence"),
        directory_rebindings={remote_snapshot: rehydrated},
    )

    validated = validate_relocatable_evidence_bundle(manifest.absolute_path)
    assert validated.artifact.directory_rebindings[0].members[0].size == 0
