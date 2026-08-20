"""Bounded append-only disk layers for the formal registry.

``FormalRegistryVerificationReceipt`` is the in-memory semantic receipt used by
the reducers.  Serializing it recursively embeds every prior receipt and every
large stage payload.  That representation cannot be a durable proof file once
E2 round zero reaches thousands of cells.  This module stores only the current
delta, binds the prior layer by path/raw/semantic identity, and replaces large
signed materialization/coverage payloads with compact proof-graph wrappers.
Deep loading reconstructs the original semantic receipt and reruns its normal
registry verifier; the bounded layer is transport, never a second authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.e2_stage_authority import (
    SignedE2StagedRoundSelectionReceipt,
)
from lightcone_spec.experiments.e3a_stage_authority import (
    E3aStagedSelectionArtifact,
    SignedE3aStagedSelectionReceipt,
)
from lightcone_spec.experiments.e4_stage_authority import (
    SignedE4StageSelectionReceipt,
)
from lightcone_spec.experiments.formal_protocol import (
    SignedProtocolLock,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    FormalCandidateReplayProofBinding,
    FormalRegistryManifest,
    FormalRegistryVerificationReceipt,
    e2_staged_evidence_manifest_to_dict,
    e3a_staged_selection_artifact_to_dict,
    e4_staged_evidence_manifest_to_dict,
    formal_registry_manifest_from_dict,
    formal_registry_verification_receipt_from_dict,
    signed_e1_survivor_selection_to_dict,
    signed_e2_staged_selection_to_dict,
    signed_e3a_staged_selection_to_dict,
    signed_e4_stage_selection_to_dict,
    signed_protocol_lock_to_dict,
    signed_stage_coverage_to_dict,
    signed_stage_materialization_to_dict,
)
from lightcone_spec.experiments.stage_decisions import (
    SignedE1SurvivorSelectionReceipt,
)
from lightcone_spec.experiments.stage_materialization import (
    TTS_CAL_MATERIALIZATION_RULE,
    SignedStageCoverageReceipt,
    SignedStageMaterializationReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.attestation import AttestationChallenge, SignedAttestation
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_REGISTRY_LAYER_ARTIFACT_KIND = "lightcone_formal_registry_layer_artifact"

_REPLACED_RECEIPT_FIELDS = frozenset(
    {
        "prior_receipt",
        "signed_protocol_lock",
        "appended_signed_materializations",
        "appended_signed_coverage",
        "appended_e3a_staged_selection_artifacts",
        "appended_signed_e3a_staged_selections",
        "appended_e2_staged_evidence_manifests",
        "appended_signed_e1_survivor_selections",
        "appended_signed_e2_staged_selections",
        "appended_e4_staged_evidence_manifests",
        "appended_signed_e4_stage_selections",
        "appended_formal_stage_prefix_artifacts",
        "manifest",
        "receipt_sha256",
    }
)
_SMALL_RECEIPT_FIELDS = (
    frozenset(
        {
            *FormalRegistryVerificationReceipt.__dataclass_fields__,
            "receipt_sha256",
        }
    )
    - _REPLACED_RECEIPT_FIELDS
)
_MANIFEST_FIELDS = frozenset(
    {*FormalRegistryManifest.__dataclass_fields__, "manifest_sha256"}
) - {"candidate_replay_proofs"}
_UNPROVED_SMALL_APPEND_FIELDS = frozenset(
    {
        "appended_signed_e3b_power_prefixes",
        "appended_signed_e5_power_and_anchor_prefixes",
        "appended_signed_e6_power_prefixes",
        "appended_e0_onlinespec_source_authorities",
        "appended_signed_e0_compatibilities",
        "appended_signed_e0_onlinespec_tuning_seals",
        "appended_signed_e0_power_prefixes",
    }
)

FORMAL_REGISTRY_REPLAY_PROOF_SHARD_KIND = "lightcone_formal_registry_replay_proof_shard"

FORMAL_SIGNED_PREFIX_RESULT_HEADER_KIND = "lightcone_formal_signed_prefix_result_header"


@dataclass(frozen=True)
class FormalSignedPrefixResultHeader:
    """Compact signature fields for one prefix-derived selection payload."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_signed_prefix_result_header"]
    phase: str
    payload_sha256: str
    signed_receipt_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_SIGNED_PREFIX_RESULT_HEADER_KIND
        ):
            raise ValueError("formal signed prefix-result header identity differs")
        if self.phase not in {
            "e1_selection",
            "e2_round0",
            "e2_round1",
            "e2_round2",
            "e2_round3",
            "e4_screen",
            "e4_local",
        }:
            raise ValueError("formal signed prefix-result phase is unsupported")
        _sha256("formal prefix-result payload", self.payload_sha256)
        _sha256("formal prefix-result signed receipt", self.signed_receipt_sha256)
        if (
            type(self.challenge) is not AttestationChallenge
            or type(self.attestation) is not SignedAttestation
        ):
            raise TypeError("formal prefix-result signature fields are not exact")
        self.challenge.validate()
        self.attestation.validate()
        if (
            self.challenge.subject_sha256 != self.payload_sha256
            or self.attestation.payload_sha256 != self.payload_sha256
            or self.attestation.challenge_sha256 != self.challenge.sha256
        ):
            raise ValueError("formal prefix-result signature lineage differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "phase": self.phase,
            "payload_sha256": self.payload_sha256,
            "signed_receipt_sha256": self.signed_receipt_sha256,
            "challenge": asdict(self.challenge),
            "attestation": asdict(self.attestation),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != cls.__dataclass_fields__:
            raise ValueError("formal signed prefix-result header fields differ")
        row = dict(value)
        challenge = row["challenge"]
        attestation = row["attestation"]
        if type(challenge) is not dict or type(attestation) is not dict:
            raise TypeError("formal prefix-result signature fields must be objects")
        row["challenge"] = AttestationChallenge(**challenge)
        row["attestation"] = SignedAttestation(**attestation)
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalRegistryReplayProofShard:
    """One bounded current-layer slice of candidate replay bindings."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_registry_replay_proof_shard"]
    semantic_receipt_sha256: str
    shard_index: int
    shard_count: int
    proofs: tuple[FormalCandidateReplayProofBinding, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_REGISTRY_REPLAY_PROOF_SHARD_KIND
        ):
            raise ValueError("formal registry replay shard identity differs")
        _sha256("formal registry replay shard receipt", self.semantic_receipt_sha256)
        if (
            type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or self.shard_index not in range(self.shard_count)
        ):
            raise ValueError("formal registry replay shard position is invalid")
        if (
            type(self.proofs) is not tuple
            or not self.proofs
            or any(
                type(row) is not FormalCandidateReplayProofBinding
                for row in self.proofs
            )
        ):
            raise TypeError("formal registry replay shard proofs are not exact")
        keys = tuple(row.pointer_commitment_sha256 for row in self.proofs)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("formal registry replay shard proofs are not canonical")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "semantic_receipt_sha256": self.semantic_receipt_sha256,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "proofs": [
                {
                    "pointer_commitment_sha256": row.pointer_commitment_sha256,
                    "proof_artifact": row.proof_artifact.to_dict(),
                }
                for row in self.proofs
            ],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "shard_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal registry replay shard fields differ")
        row = dict(value)
        declared = _sha256("formal registry replay shard", row.pop("shard_sha256"))
        raw = row["proofs"]
        if type(raw) is not list:
            raise TypeError("formal registry replay shard proofs must be an array")
        proofs = []
        for item in raw:
            if type(item) is not dict or set(item) != {
                "pointer_commitment_sha256",
                "proof_artifact",
            }:
                raise ValueError("formal registry replay proof fields differ")
            proofs.append(
                FormalCandidateReplayProofBinding(
                    pointer_commitment_sha256=item["pointer_commitment_sha256"],
                    proof_artifact=CanonicalJsonProofBinding.from_dict(
                        item["proof_artifact"]
                    ),
                )
            )
        row["proofs"] = tuple(proofs)
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("formal registry replay shard digest differs")
        return shard


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _open_binding(binding: CanonicalJsonProofBinding, *, label: str) -> dict:
    observed = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if observed != binding:
        raise ValueError(f"{label} path identity changed")
    value = observed.reopen()
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != observed:
        raise RuntimeError(f"{label} changed while reopened")
    return value


def _signed_protocol_lock_from_source(
    binding: CanonicalJsonProofBinding,
    *,
    now_ns: int,
) -> SignedProtocolLock:
    value = _open_binding(binding, label="formal signed ProtocolLock source")
    if value.get("kind") != "lightcone_scientific_signed_proof_wrapper":
        raise ValueError(
            "formal registry root requires a ProtocolLock proof-replay wrapper"
        )
    from lightcone_spec.runtime.scientific_signing import (
        rebuild_scientific_signed_proof_wrapper,
    )

    signed = rebuild_scientific_signed_proof_wrapper(
        binding.absolute_path,
        now_ns=now_ns,
    )
    if type(signed) is not SignedProtocolLock:
        raise TypeError("formal ProtocolLock proof wrapper has another type")
    return signed


def _signed_e3a_selection_from_source(
    binding: CanonicalJsonProofBinding,
    *,
    now_ns: int,
) -> SignedE3aStagedSelectionReceipt:
    value = _open_binding(binding, label="formal signed E3a selection source")
    if value.get("kind") != "lightcone_scientific_signed_proof_wrapper":
        raise ValueError(
            "formal E3a selection requires a proof-replay signature wrapper"
        )
    from lightcone_spec.runtime.scientific_signing import (
        rebuild_scientific_signed_proof_wrapper,
    )

    signed = rebuild_scientific_signed_proof_wrapper(
        binding.absolute_path,
        now_ns=now_ns,
    )
    if type(signed) is not SignedE3aStagedSelectionReceipt:
        raise TypeError("formal E3a selection proof wrapper has another type")
    return signed


def _rebuild_e3a_selection_delta(
    proof_sources: tuple[CanonicalJsonProofBinding, ...],
    signed_sources: tuple[CanonicalJsonProofBinding, ...],
    *,
    now_ns: int,
) -> tuple[
    tuple[E3aStagedSelectionArtifact, ...],
    tuple[SignedE3aStagedSelectionReceipt, ...],
]:
    if len(proof_sources) != len(signed_sources):
        raise ValueError("formal E3a proof/signature coverage differs")
    from lightcone_spec.experiments.e3a_staged_selection_proof import (
        revalidate_formal_e3a_staged_selection_proof_artifact,
    )

    artifacts: list[E3aStagedSelectionArtifact] = []
    signed_rows: list[SignedE3aStagedSelectionReceipt] = []
    for proof_source, signed_source in zip(
        proof_sources,
        signed_sources,
        strict=True,
    ):
        artifact, receipt = revalidate_formal_e3a_staged_selection_proof_artifact(
            proof_source.absolute_path,
            now_ns=now_ns,
        )
        signed = _signed_e3a_selection_from_source(
            signed_source,
            now_ns=now_ns,
        )
        if signed.payload != receipt or signed.payload_sha256 != receipt.sha256:
            raise ValueError("formal signed E3a selection differs from reducer proof")
        artifacts.append(artifact)
        signed_rows.append(signed)
    return tuple(artifacts), tuple(signed_rows)


@dataclass(frozen=True)
class FormalRegistryLayerArtifact:
    """One bounded current-only durable registry delta."""

    schema_version: Literal[5]
    kind: Literal["lightcone_formal_registry_layer_artifact"]
    semantic_receipt_sha256: str
    prior_layer_source: CanonicalJsonProofBinding | None
    signed_protocol_lock_source: CanonicalJsonProofBinding | None
    signed_materialization_sources: tuple[CanonicalJsonProofBinding, ...]
    signed_coverage_sources: tuple[CanonicalJsonProofBinding, ...]
    formal_stage_prefix_sources: tuple[CanonicalJsonProofBinding, ...]
    signed_prefix_result_headers: tuple[FormalSignedPrefixResultHeader, ...]
    candidate_replay_proof_shard_sources: tuple[CanonicalJsonProofBinding, ...]
    tts_calibration_reduction_proof_sources: tuple[CanonicalJsonProofBinding, ...]
    e3a_staged_selection_proof_sources: tuple[CanonicalJsonProofBinding, ...]
    signed_e3a_staged_selection_sources: tuple[CanonicalJsonProofBinding, ...]
    manifest_fields: dict[str, object]
    small_receipt_fields: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 5 or self.kind != FORMAL_REGISTRY_LAYER_ARTIFACT_KIND:
            raise ValueError("formal registry layer identity is unsupported")
        _sha256("formal registry layer semantic receipt", self.semantic_receipt_sha256)
        if (
            self.prior_layer_source is not None
            and type(self.prior_layer_source) is not CanonicalJsonProofBinding
        ):
            raise TypeError("formal registry prior layer must be path-bound")
        if (
            self.signed_protocol_lock_source is not None
            and type(self.signed_protocol_lock_source) is not CanonicalJsonProofBinding
        ):
            raise TypeError("formal registry ProtocolLock source must be path-bound")
        if (self.prior_layer_source is None) != (
            self.signed_protocol_lock_source is not None
        ):
            raise ValueError(
                "formal registry root must bind exactly one ProtocolLock proof"
            )
        for label, rows in (
            ("materialization", self.signed_materialization_sources),
            ("coverage", self.signed_coverage_sources),
            ("stage prefix", self.formal_stage_prefix_sources),
            ("candidate replay shard", self.candidate_replay_proof_shard_sources),
            (
                "TTS calibration reduction proof",
                self.tts_calibration_reduction_proof_sources,
            ),
            (
                "E3a staged selection proof",
                self.e3a_staged_selection_proof_sources,
            ),
            (
                "signed E3a staged selection",
                self.signed_e3a_staged_selection_sources,
            ),
        ):
            if type(rows) is not tuple or any(
                type(row) is not CanonicalJsonProofBinding for row in rows
            ):
                raise TypeError(f"formal registry {label} sources are not exact")
        if type(self.signed_prefix_result_headers) is not tuple or any(
            type(row) is not FormalSignedPrefixResultHeader
            for row in self.signed_prefix_result_headers
        ):
            raise TypeError("formal registry prefix-result headers are not exact")
        phases = tuple(row.phase for row in self.signed_prefix_result_headers)
        if phases != tuple(dict.fromkeys(phases)):
            raise ValueError("formal registry prefix-result phases repeat")
        if (
            type(self.small_receipt_fields) is not dict
            or set(self.small_receipt_fields) != _SMALL_RECEIPT_FIELDS
        ):
            raise ValueError("formal registry small receipt fields differ")
        if (
            type(self.manifest_fields) is not dict
            or set(self.manifest_fields) != _MANIFEST_FIELDS
        ):
            raise ValueError("formal registry compact manifest fields differ")
        nonempty_unproved = tuple(
            name
            for name in sorted(_UNPROVED_SMALL_APPEND_FIELDS)
            if self.small_receipt_fields[name]
        )
        if nonempty_unproved:
            raise ValueError(
                "formal registry layer lacks reducer-proof sources for "
                + ", ".join(nonempty_unproved)
            )
        if "prior_receipt" in self.small_receipt_fields:  # pragma: no cover
            raise AssertionError("formal registry layer embeds a prior receipt")
        paths = tuple(
            row.absolute_path
            for row in (
                *(
                    ()
                    if self.prior_layer_source is None
                    else (self.prior_layer_source,)
                ),
                *(
                    ()
                    if self.signed_protocol_lock_source is None
                    else (self.signed_protocol_lock_source,)
                ),
                *self.signed_materialization_sources,
                *self.signed_coverage_sources,
                *self.formal_stage_prefix_sources,
                *self.candidate_replay_proof_shard_sources,
                *self.tts_calibration_reduction_proof_sources,
                *self.e3a_staged_selection_proof_sources,
                *self.signed_e3a_staged_selection_sources,
            )
        )
        if len(paths) != len(set(paths)):
            raise ValueError("formal registry layer aliases a proof path")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "semantic_receipt_sha256": self.semantic_receipt_sha256,
            "prior_layer_source": (
                None
                if self.prior_layer_source is None
                else self.prior_layer_source.to_dict()
            ),
            "signed_protocol_lock_source": (
                None
                if self.signed_protocol_lock_source is None
                else self.signed_protocol_lock_source.to_dict()
            ),
            "signed_materialization_sources": [
                row.to_dict() for row in self.signed_materialization_sources
            ],
            "signed_coverage_sources": [
                row.to_dict() for row in self.signed_coverage_sources
            ],
            "formal_stage_prefix_sources": [
                row.to_dict() for row in self.formal_stage_prefix_sources
            ],
            "signed_prefix_result_headers": [
                row.to_dict() for row in self.signed_prefix_result_headers
            ],
            "candidate_replay_proof_shard_sources": [
                row.to_dict() for row in self.candidate_replay_proof_shard_sources
            ],
            "tts_calibration_reduction_proof_sources": [
                row.to_dict() for row in self.tts_calibration_reduction_proof_sources
            ],
            "e3a_staged_selection_proof_sources": [
                row.to_dict() for row in self.e3a_staged_selection_proof_sources
            ],
            "signed_e3a_staged_selection_sources": [
                row.to_dict() for row in self.signed_e3a_staged_selection_sources
            ],
            "manifest_fields": self.manifest_fields,
            "small_receipt_fields": self.small_receipt_fields,
        }
        if include_sha256:
            value["layer_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "layer_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal registry layer fields differ")
        row = dict(value)
        declared = _sha256("formal registry layer", row.pop("layer_sha256"))
        row["prior_layer_source"] = (
            None
            if row["prior_layer_source"] is None
            else CanonicalJsonProofBinding.from_dict(row["prior_layer_source"])
        )
        row["signed_protocol_lock_source"] = (
            None
            if row["signed_protocol_lock_source"] is None
            else CanonicalJsonProofBinding.from_dict(row["signed_protocol_lock_source"])
        )
        for name in (
            "signed_materialization_sources",
            "signed_coverage_sources",
            "formal_stage_prefix_sources",
            "candidate_replay_proof_shard_sources",
            "tts_calibration_reduction_proof_sources",
            "e3a_staged_selection_proof_sources",
            "signed_e3a_staged_selection_sources",
        ):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(f"formal registry layer {name} must be an array")
            row[name] = tuple(CanonicalJsonProofBinding.from_dict(item) for item in raw)
        raw_headers = row["signed_prefix_result_headers"]
        if type(raw_headers) is not list:
            raise TypeError("formal prefix-result headers must be an array")
        row["signed_prefix_result_headers"] = tuple(
            FormalSignedPrefixResultHeader.from_dict(item) for item in raw_headers
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal registry layer digest differs")
        return artifact


def _signed_materialization_from_source(
    binding: CanonicalJsonProofBinding,
    *,
    now_ns: int,
) -> SignedStageMaterializationReceipt:
    value = _open_binding(binding, label="formal signed materialization source")
    if value.get("kind") == "lightcone_scientific_signed_proof_wrapper":
        from lightcone_spec.runtime.scientific_signing import (
            rebuild_scientific_signed_proof_wrapper,
        )

        signed = rebuild_scientific_signed_proof_wrapper(
            binding.absolute_path,
            now_ns=now_ns,
        )
        if type(signed) is not SignedStageMaterializationReceipt:
            raise TypeError("formal materialization proof wrapper has another type")
        return signed
    raise ValueError(
        "formal registry rejects raw or transport-only signed materialization; "
        "a typed predecessor reducer proof wrapper is required"
    )


def _rebuilt_prefixes(
    bindings: tuple[CanonicalJsonProofBinding, ...],
    *,
    now_ns: int,
) -> tuple[object, ...]:
    from lightcone_spec.experiments.formal_stage_prefix import (
        load_and_rebuild_formal_stage_prefix,
    )

    return tuple(
        load_and_rebuild_formal_stage_prefix(binding.absolute_path, now_ns=now_ns)
        for binding in bindings
    )


def _signed_coverage_from_source(
    binding: CanonicalJsonProofBinding,
    *,
    now_ns: int,
) -> SignedStageCoverageReceipt:
    value = _open_binding(binding, label="formal signed coverage source")
    if value.get("kind") == "lightcone_scientific_signed_proof_wrapper":
        from lightcone_spec.runtime.scientific_signing import (
            rebuild_scientific_signed_proof_wrapper,
        )

        signed = rebuild_scientific_signed_proof_wrapper(
            binding.absolute_path,
            now_ns=now_ns,
        )
        if type(signed) is not SignedStageCoverageReceipt:
            raise TypeError("formal coverage proof wrapper has another type")
        return signed
    raise ValueError(
        "formal registry rejects raw or low-level signed coverage; a portable "
        "pre-coverage registry proof wrapper is required"
    )


def _prefix_evidence(
    rebuilt_prefixes: tuple[object, ...],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    e2 = tuple(
        prefix.evidence_manifest
        for prefix in rebuilt_prefixes
        if prefix.artifact.phase.startswith("e2_round")
    )
    e4 = tuple(
        prefix.evidence_manifest
        for prefix in rebuilt_prefixes
        if prefix.artifact.phase in {"e4_screen", "e4_local"}
    )
    if any(row is None for row in (*e2, *e4)):
        raise ValueError("formal registry prefix lacks proof-derived evidence")
    return e2, e4


def _rebuild_prefix_signed_results(
    rebuilt_prefixes: tuple[object, ...],
    headers: tuple[FormalSignedPrefixResultHeader, ...],
    *,
    now_ns: int,
) -> tuple[
    tuple[SignedE1SurvivorSelectionReceipt, ...],
    tuple[SignedE2StagedRoundSelectionReceipt, ...],
    tuple[SignedE4StageSelectionReceipt, ...],
]:
    from lightcone_spec.experiments.formal_stage_prefix import (
        reduce_formal_stage_prefix,
    )

    header_by_phase = {row.phase: row for row in headers}
    selection_prefixes = tuple(
        row for row in rebuilt_prefixes if row.artifact.phase != "e4_profiler"
    )
    if set(header_by_phase) != {row.artifact.phase for row in selection_prefixes}:
        raise ValueError("formal registry prefix-result header coverage differs")
    e1 = []
    e2 = []
    e4 = []
    for prefix in selection_prefixes:
        phase = prefix.artifact.phase
        header = header_by_phase[phase]
        payload = reduce_formal_stage_prefix(prefix, now_ns=now_ns)
        if content_sha256(payload) != header.payload_sha256:
            raise ValueError("formal registry prefix-result payload differs")
        if phase == "e1_selection":
            signed = SignedE1SurvivorSelectionReceipt(
                payload=payload,
                payload_sha256=header.payload_sha256,
                challenge=header.challenge,
                attestation=header.attestation,
            )
            e1.append(signed)
        elif phase.startswith("e2_round"):
            signed = SignedE2StagedRoundSelectionReceipt(
                payload=payload,
                payload_sha256=header.payload_sha256,
                challenge=header.challenge,
                attestation=header.attestation,
            )
            e2.append(signed)
        else:
            signed = SignedE4StageSelectionReceipt(
                payload=payload,
                payload_sha256=header.payload_sha256,
                challenge=header.challenge,
                attestation=header.attestation,
            )
            e4.append(signed)
        if signed.sha256 != header.signed_receipt_sha256:
            raise ValueError("formal registry prefix-result signature differs")
    return tuple(e1), tuple(e2), tuple(e4)


def _bind_prefix_result_headers(
    receipt: FormalRegistryVerificationReceipt,
    rebuilt_prefixes: tuple[object, ...],
    *,
    now_ns: int,
) -> tuple[FormalSignedPrefixResultHeader, ...]:
    from lightcone_spec.experiments.formal_stage_prefix import (
        reduce_formal_stage_prefix,
    )

    sources = {
        "e1_selection": receipt.appended_signed_e1_survivor_selections,
        **{
            f"e2_round{index}": receipt.appended_signed_e2_staged_selections
            for index in range(4)
        },
        "e4_screen": receipt.appended_signed_e4_stage_selections,
        "e4_local": receipt.appended_signed_e4_stage_selections,
    }
    headers = []
    for prefix in rebuilt_prefixes:
        phase = prefix.artifact.phase
        if phase == "e4_profiler":
            continue
        payload = reduce_formal_stage_prefix(prefix, now_ns=now_ns)
        matches = tuple(row for row in sources[phase] if row.payload == payload)
        if len(matches) != 1:
            raise ValueError("formal registry prefix lacks one exact signed result")
        signed = matches[0]
        headers.append(
            FormalSignedPrefixResultHeader(
                schema_version=1,
                kind=FORMAL_SIGNED_PREFIX_RESULT_HEADER_KIND,
                phase=phase,
                payload_sha256=signed.payload_sha256,
                signed_receipt_sha256=signed.sha256,
                challenge=signed.challenge,
                attestation=signed.attestation,
            )
        )
    result = tuple(headers)
    rebuilt = _rebuild_prefix_signed_results(
        rebuilt_prefixes,
        result,
        now_ns=now_ns,
    )
    if rebuilt != (
        receipt.appended_signed_e1_survivor_selections,
        receipt.appended_signed_e2_staged_selections,
        receipt.appended_signed_e4_stage_selections,
    ):
        raise ValueError("formal registry prefix-result delta differs")
    return result


def _replay_proofs_from_shards(
    bindings: tuple[CanonicalJsonProofBinding, ...],
    *,
    semantic_receipt_sha256: str,
) -> tuple[FormalCandidateReplayProofBinding, ...]:
    shards = tuple(
        FormalRegistryReplayProofShard.from_dict(
            _open_binding(binding, label="formal registry replay proof shard")
        )
        for binding in bindings
    )
    if not shards:
        return ()
    if (
        tuple(row.shard_index for row in shards) != tuple(range(len(shards)))
        or any(row.shard_count != len(shards) for row in shards)
        or any(row.semantic_receipt_sha256 != semantic_receipt_sha256 for row in shards)
    ):
        raise ValueError("formal registry replay shard sequence differs")
    proofs = tuple(row for shard in shards for row in shard.proofs)
    keys = tuple(row.pointer_commitment_sha256 for row in proofs)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("formal registry replay shard union is not canonical")
    for proof in proofs:
        _open_binding(
            proof.proof_artifact,
            label="formal registry candidate replay proof",
        )
    return proofs


def _current_replay_proofs(
    receipt: FormalRegistryVerificationReceipt,
    prior: FormalRegistryVerificationReceipt | None,
) -> tuple[FormalCandidateReplayProofBinding, ...]:
    prior_rows = () if prior is None else prior.manifest.candidate_replay_proofs
    prior_keys = {row.pointer_commitment_sha256 for row in prior_rows}
    if len(prior_keys) != len(prior_rows):  # pragma: no cover - manifest invariant
        raise AssertionError("formal registry prior replay proofs repeat")
    current = tuple(
        row
        for row in receipt.manifest.candidate_replay_proofs
        if row.pointer_commitment_sha256 not in prior_keys
    )
    if (
        tuple(
            sorted(
                (*prior_rows, *current), key=lambda row: row.pointer_commitment_sha256
            )
        )
        != receipt.manifest.candidate_replay_proofs
    ):
        raise ValueError("formal registry replay proof delta differs")
    return current


def validate_formal_precoverage_registry_state(
    receipt: FormalRegistryVerificationReceipt,
    *,
    stage: str,
    phase: str,
    materialization: StageMaterializationReceipt,
    immediate_predecessor_prefix_sha256: str | None = None,
) -> None:
    """Require the exact atomic predecessor-to-materialization transition.

    A formal registry layer closes its immediate predecessor and appends the
    successor materialization atomically.  For example, the TTS-Cal launch
    layer contains the E3a coverage and signed staged selection together with
    the TTS-Cal materialization.  Treating every predecessor result as
    forbidden would split that scientific transition into an invented layer;
    accepting arbitrary results would permit retrospective proof replay.  The
    closed union below therefore permits exactly the predecessor rows for the
    requested phase and rejects the current result and every future source.
    """

    if type(receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("formal pre-coverage state requires an exact registry")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal pre-coverage state requires exact materialization")
    if type(stage) is not str or not stage or type(phase) is not str or not phase:
        raise ValueError("formal pre-coverage stage/phase is invalid")
    if materialization.stage != stage:
        raise ValueError("formal pre-coverage materialization stage differs")
    receipt.revalidate(current_ns=receipt.verified_ns)
    if (
        receipt.prior_receipt is None
        or len(receipt.appended_signed_materializations) != 1
        or receipt.appended_signed_materializations[0].payload != materialization
        or receipt.signed_protocol_lock.payload.sha256
        != materialization.protocol_lock_sha256
    ):
        raise ValueError(
            "formal pre-coverage registry is not the current materialization layer"
        )

    phase_key = (stage, phase)
    allowed_by_phase = {
        ("E3a", "capacity"): frozenset(),
        ("TTS-Cal", "calibration"): frozenset(
            {
                "appended_signed_coverage",
                "appended_e3a_staged_selection_artifacts",
                "appended_signed_e3a_staged_selections",
            }
        ),
        ("E1", "selection"): frozenset(
            {
                "appended_signed_coverage",
                "appended_tts_calibration_authorities",
                "appended_signed_tts_calibration_seals",
            }
        ),
        ("E2", "round0"): frozenset(
            {
                "appended_signed_coverage",
                "appended_signed_e1_survivor_selections",
                "appended_formal_stage_prefix_artifacts",
            }
        ),
        **{
            ("E2", f"round{round_index}"): frozenset(
                {
                    "appended_signed_coverage",
                    "appended_e2_staged_evidence_manifests",
                    "appended_signed_e2_staged_selections",
                    "appended_formal_stage_prefix_artifacts",
                }
            )
            for round_index in range(1, 4)
        },
        ("E4", "screen"): frozenset(
            {
                "appended_signed_coverage",
                "appended_e2_staged_evidence_manifests",
                "appended_signed_e2_staged_selections",
                "appended_formal_stage_prefix_artifacts",
            }
        ),
        **{
            ("E4", e4_phase): frozenset(
                {
                    "appended_signed_coverage",
                    "appended_e4_staged_evidence_manifests",
                    "appended_signed_e4_stage_selections",
                    "appended_formal_stage_prefix_artifacts",
                }
            )
            for e4_phase in ("local", "profiler")
        },
    }
    try:
        allowed_fields = allowed_by_phase[phase_key]
    except KeyError as error:
        raise ValueError("formal pre-coverage stage/phase is unsupported") from error

    phase_rules = {
        ("E3a", "capacity"): "exact_360_row_capacity_width_and_drift_grid",
        ("TTS-Cal", "calibration"): TTS_CAL_MATERIALIZATION_RULE,
        ("E1", "selection"): ("four_fixed_anchors_plus_32_geometries_x_2_optimizers"),
        ("E2", "round0"): "e2_round_0_105_per_geometry_plus_four_anchors",
        ("E2", "round1"): "e2_quarter_retention_floor_21_plus_four_anchors",
        ("E2", "round2"): "e2_quarter_retention_floor_21_plus_four_anchors",
        ("E2", "round3"): "e2_quarter_retention_floor_21_plus_four_anchors",
        ("E4", "screen"): "strength2_8_rows_x_3_loads_x_2_traffic",
        ("E4", "local"): ("winner_neighborhood_2pow4_x_3_loads_x_2_traffic"),
        ("E4", "profiler"): "three_profiler_only_rows_separate_from_headline",
    }
    if materialization.materialization_rule != phase_rules[phase_key]:
        raise ValueError("formal pre-coverage materialization phase differs")
    if stage == "E2":
        rounds = {dict(cell.dimensions).get("round") for cell in materialization.cells}
        if rounds != {int(phase.removeprefix("round"))}:
            raise ValueError("formal pre-coverage E2 round differs")

    scientific_fields = (
        "appended_signed_coverage",
        "appended_e3a_staged_selection_artifacts",
        "appended_signed_e3a_staged_selections",
        "appended_tts_calibration_authorities",
        "appended_signed_tts_calibration_seals",
        "appended_e2_staged_evidence_manifests",
        "appended_signed_e2_staged_selections",
        "appended_signed_e1_survivor_selections",
        "appended_e4_staged_evidence_manifests",
        "appended_signed_e4_stage_selections",
        "appended_signed_e3b_power_prefixes",
        "appended_signed_e5_power_and_anchor_prefixes",
        "appended_signed_e6_power_prefixes",
        "appended_e0_onlinespec_source_authorities",
        "appended_signed_e0_compatibilities",
        "appended_signed_e0_onlinespec_tuning_seals",
        "appended_signed_e0_power_prefixes",
        "appended_formal_stage_prefix_artifacts",
    )
    if any(
        bool(getattr(receipt, name)) != (name in allowed_fields)
        or (name in allowed_fields and len(getattr(receipt, name)) != 1)
        for name in scientific_fields
    ):
        raise ValueError(
            "formal pre-coverage current layer differs from its exact transition"
        )

    if receipt.appended_signed_coverage:
        predecessor_coverage = receipt.appended_signed_coverage[0].payload
        expected_predecessor_stage = {
            ("TTS-Cal", "calibration"): "E3a",
            ("E1", "selection"): "TTS-Cal",
            ("E2", "round0"): "E1",
            ("E2", "round1"): "E2",
            ("E2", "round2"): "E2",
            ("E2", "round3"): "E2",
            ("E4", "screen"): "E2",
            ("E4", "local"): "E4",
            ("E4", "profiler"): "E4",
        }[phase_key]
        if predecessor_coverage.stage != expected_predecessor_stage:
            raise ValueError("formal pre-coverage predecessor stage differs")
        if phase_key == ("TTS-Cal", "calibration"):
            artifact = receipt.appended_e3a_staged_selection_artifacts[0]
            signed_selection = receipt.appended_signed_e3a_staged_selections[0]
            if (
                artifact.coverage_receipt_sha256 != predecessor_coverage.sha256
                or artifact.materialization_receipt_sha256
                != predecessor_coverage.materialization_receipt_sha256
                or signed_selection.payload.selection_artifact_sha256 != artifact.sha256
                or materialization.upstream_receipt_sha256s
                != (signed_selection.sha256,)
            ):
                raise ValueError("formal TTS-Cal predecessor transition differs")
        else:
            if (
                not materialization.upstream_receipt_sha256s
                or materialization.upstream_receipt_sha256s[0]
                != predecessor_coverage.materialization_receipt_sha256
            ):
                raise ValueError("formal pre-coverage predecessor receipt differs")
            if phase_key == ("E1", "selection"):
                signed_seal = receipt.appended_signed_tts_calibration_seals[0]
                if (
                    signed_seal.payload.coverage_receipt_sha256
                    != predecessor_coverage.sha256
                    or materialization.upstream_receipt_sha256s[1] != signed_seal.sha256
                ):
                    raise ValueError("formal E1 TTS-Cal transition differs")
            elif phase_key == ("E2", "round0"):
                if (
                    materialization.source_decision_sha256
                    != receipt.appended_signed_e1_survivor_selections[0].sha256
                ):
                    raise ValueError("formal E2 round-zero decision differs")
            elif stage == "E2" or phase_key == ("E4", "screen"):
                if (
                    materialization.source_decision_sha256
                    != receipt.appended_signed_e2_staged_selections[0].sha256
                ):
                    raise ValueError("formal E2 predecessor decision differs")
            else:
                if (
                    materialization.source_decision_sha256
                    != receipt.appended_signed_e4_stage_selections[0].sha256
                ):
                    raise ValueError("formal E4 predecessor decision differs")
    if any(
        row.payload.materialization_receipt_sha256 == materialization.sha256
        for row in receipt.cumulative_signed_coverage
    ):
        raise ValueError("formal pre-coverage registry already contains coverage")
    prefixes = receipt.cumulative_formal_stage_prefix_artifacts
    if immediate_predecessor_prefix_sha256 is None:
        if stage in {"E3a", "TTS-Cal", "E1"} and prefixes:
            raise ValueError("formal initial pre-coverage state has a future prefix")
        if stage in {"E2", "E4"}:
            raise ValueError("formal pre-coverage predecessor prefix is missing")
    else:
        _sha256(
            "formal pre-coverage predecessor prefix",
            immediate_predecessor_prefix_sha256,
        )
        if (
            not prefixes
            or prefixes[-1].semantic_sha256 != immediate_predecessor_prefix_sha256
            or len(receipt.appended_formal_stage_prefix_artifacts) != 1
            or receipt.appended_formal_stage_prefix_artifacts[0].semantic_sha256
            != immediate_predecessor_prefix_sha256
        ):
            raise ValueError("formal pre-coverage predecessor prefix differs")


def _load_layer_binding(
    binding: CanonicalJsonProofBinding,
    *,
    now_ns: int,
    seen_paths: frozenset[str],
) -> FormalRegistryVerificationReceipt:
    if binding.absolute_path in seen_paths:
        raise ValueError("formal registry layer chain contains a cycle")
    value = _open_binding(binding, label="formal registry layer")
    if value.get("kind") != FORMAL_REGISTRY_LAYER_ARTIFACT_KIND:
        raise ValueError(
            "formal registry consumers require a bounded proof-replay layer"
        )
    artifact = FormalRegistryLayerArtifact.from_dict(value)
    replay_ns = artifact.small_receipt_fields.get("verified_ns")
    if type(replay_ns) is not int or replay_ns < 1 or replay_ns > now_ns:
        raise ValueError("formal registry layer verification time is invalid")
    prior = (
        None
        if artifact.prior_layer_source is None
        else _load_layer_binding(
            artifact.prior_layer_source,
            now_ns=now_ns,
            seen_paths=seen_paths | {binding.absolute_path},
        )
    )
    rebuilt_prefixes = _rebuilt_prefixes(
        artifact.formal_stage_prefix_sources,
        now_ns=replay_ns,
    )
    materializations = tuple(
        _signed_materialization_from_source(row, now_ns=replay_ns)
        for row in artifact.signed_materialization_sources
    )
    coverages = tuple(
        _signed_coverage_from_source(
            row,
            now_ns=replay_ns,
        )
        for row in artifact.signed_coverage_sources
    )
    e2_evidence, e4_evidence = _prefix_evidence(rebuilt_prefixes)
    e1_results, e2_results, e4_results = _rebuild_prefix_signed_results(
        rebuilt_prefixes,
        artifact.signed_prefix_result_headers,
        now_ns=replay_ns,
    )
    e3a_artifacts, e3a_results = _rebuild_e3a_selection_delta(
        artifact.e3a_staged_selection_proof_sources,
        artifact.signed_e3a_staged_selection_sources,
        now_ns=replay_ns,
    )
    signed_protocol_lock = (
        prior.signed_protocol_lock
        if prior is not None
        else _signed_protocol_lock_from_source(
            artifact.signed_protocol_lock_source,
            now_ns=replay_ns,
        )
    )
    current_replay_proofs = _replay_proofs_from_shards(
        artifact.candidate_replay_proof_shard_sources,
        semantic_receipt_sha256=artifact.semantic_receipt_sha256,
    )
    prior_replay_proofs = (
        () if prior is None else prior.manifest.candidate_replay_proofs
    )
    manifest_row = dict(artifact.manifest_fields)
    manifest_row["candidate_replay_proofs"] = [
        {
            "pointer_commitment_sha256": row.pointer_commitment_sha256,
            "proof_artifact": row.proof_artifact.to_dict(),
        }
        for row in sorted(
            (*prior_replay_proofs, *current_replay_proofs),
            key=lambda row: row.pointer_commitment_sha256,
        )
    ]
    formal_registry_manifest_from_dict(manifest_row)
    row = dict(artifact.small_receipt_fields)
    row.update(
        {
            "prior_receipt": None if prior is None else prior.to_dict(),
            "signed_protocol_lock": signed_protocol_lock_to_dict(signed_protocol_lock),
            "appended_signed_materializations": [
                signed_stage_materialization_to_dict(item) for item in materializations
            ],
            "appended_signed_coverage": [
                signed_stage_coverage_to_dict(item) for item in coverages
            ],
            "appended_e3a_staged_selection_artifacts": [
                e3a_staged_selection_artifact_to_dict(item) for item in e3a_artifacts
            ],
            "appended_signed_e3a_staged_selections": [
                signed_e3a_staged_selection_to_dict(item) for item in e3a_results
            ],
            "appended_e2_staged_evidence_manifests": [
                e2_staged_evidence_manifest_to_dict(item) for item in e2_evidence
            ],
            "appended_signed_e1_survivor_selections": [
                signed_e1_survivor_selection_to_dict(item) for item in e1_results
            ],
            "appended_signed_e2_staged_selections": [
                signed_e2_staged_selection_to_dict(item) for item in e2_results
            ],
            "appended_e4_staged_evidence_manifests": [
                e4_staged_evidence_manifest_to_dict(item) for item in e4_evidence
            ],
            "appended_signed_e4_stage_selections": [
                signed_e4_stage_selection_to_dict(item) for item in e4_results
            ],
            "appended_formal_stage_prefix_artifacts": [
                item.to_dict() for item in artifact.formal_stage_prefix_sources
            ],
            "manifest": manifest_row,
            "receipt_sha256": artifact.semantic_receipt_sha256,
        }
    )
    receipt = formal_registry_verification_receipt_from_dict(row)
    if receipt.sha256 != artifact.semantic_receipt_sha256:
        raise ValueError("formal registry layer reconstructs another receipt")
    receipt.revalidate(current_ns=replay_ns)
    if len(artifact.tts_calibration_reduction_proof_sources) != len(
        receipt.appended_signed_tts_calibration_seals
    ):
        raise ValueError("formal registry TTS reduction proof coverage differs")
    if artifact.tts_calibration_reduction_proof_sources:
        from lightcone_spec.experiments.tts_calibration_authority import (
            revalidate_signed_tts_calibration_seal_from_reduction_proof,
        )

        policy = receipt.trusted_release_policy(current_ns=replay_ns)
        for source, signed in zip(
            artifact.tts_calibration_reduction_proof_sources,
            receipt.appended_signed_tts_calibration_seals,
            strict=True,
        ):
            revalidate_signed_tts_calibration_seal_from_reduction_proof(
                source.absolute_path,
                signed_seal=signed,
                policy=policy,
                expected_policy_sha256=policy.sha256,
                now_ns=replay_ns,
            )
    return receipt


def load_formal_registry_verification_receipt_path(
    path: str | Path,
    *,
    now_ns: int,
) -> FormalRegistryVerificationReceipt:
    """Deep-open a bounded schema-5 layer chain and every reducer proof."""

    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("formal registry layer verification time is invalid")
    return _load_layer_binding(
        CanonicalJsonProofBinding.bind(path),
        now_ns=now_ns,
        seen_paths=frozenset(),
    )


def load_formal_signed_materialization_path(
    path: str | Path,
    *,
    now_ns: int,
) -> SignedStageMaterializationReceipt:
    """Deep-open a reducer-proved compact signed materialization source."""

    return _signed_materialization_from_source(
        CanonicalJsonProofBinding.bind(path),
        now_ns=now_ns,
    )


def load_formal_signed_coverage_path(
    path: str | Path,
    *,
    formal_stage_prefix_paths: tuple[str | Path, ...],
    now_ns: int,
) -> SignedStageCoverageReceipt:
    """Deep-open a reducer-proved compact signed coverage source."""

    if type(formal_stage_prefix_paths) is not tuple:
        raise TypeError("formal stage prefix paths must be an exact tuple")
    return _signed_coverage_from_source(
        CanonicalJsonProofBinding.bind(path),
        now_ns=now_ns,
    )


def bind_formal_registry_layer_artifact(
    receipt: FormalRegistryVerificationReceipt,
    *,
    prior_layer_path: str | Path | None,
    signed_protocol_lock_path: str | Path | None = None,
    signed_materialization_paths: tuple[str | Path, ...],
    signed_coverage_paths: tuple[str | Path, ...],
    formal_stage_prefix_paths: tuple[str | Path, ...],
    candidate_replay_proof_shard_paths: tuple[str | Path, ...] = (),
    tts_calibration_reduction_proof_paths: tuple[str | Path, ...] = (),
    e3a_staged_selection_proof_paths: tuple[str | Path, ...] = (),
    signed_e3a_staged_selection_paths: tuple[str | Path, ...] = (),
) -> FormalRegistryLayerArtifact:
    """Bind current operator outputs and exact-compare the semantic receipt."""

    if type(receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("formal registry layer requires an exact receipt")
    prior_source = (
        None
        if prior_layer_path is None
        else CanonicalJsonProofBinding.bind(prior_layer_path)
    )
    prior = (
        None
        if prior_source is None
        else load_formal_registry_verification_receipt_path(
            prior_source.absolute_path,
            now_ns=receipt.verified_ns,
        )
    )
    if prior != receipt.prior_receipt:
        raise ValueError("formal registry layer prior semantic receipt differs")
    protocol_lock_source = (
        None
        if signed_protocol_lock_path is None
        else CanonicalJsonProofBinding.bind(signed_protocol_lock_path)
    )
    if prior is None:
        if protocol_lock_source is None:
            raise ValueError(
                "formal registry root requires a ProtocolLock proof-replay wrapper"
            )
        rebuilt_lock = _signed_protocol_lock_from_source(
            protocol_lock_source,
            now_ns=receipt.verified_ns,
        )
        if rebuilt_lock != receipt.signed_protocol_lock:
            raise ValueError("formal registry root ProtocolLock proof differs")
    elif protocol_lock_source is not None:
        raise ValueError("formal registry child must inherit the root ProtocolLock")
    elif prior.signed_protocol_lock != receipt.signed_protocol_lock:
        raise ValueError("formal registry child ProtocolLock differs from its root")
    materialization_sources = tuple(
        CanonicalJsonProofBinding.bind(path) for path in signed_materialization_paths
    )
    coverage_sources = tuple(
        CanonicalJsonProofBinding.bind(path) for path in signed_coverage_paths
    )
    prefix_sources = tuple(
        CanonicalJsonProofBinding.bind(path) for path in formal_stage_prefix_paths
    )
    rebuilt_prefixes = _rebuilt_prefixes(
        prefix_sources,
        now_ns=receipt.verified_ns,
    )
    materializations = tuple(
        _signed_materialization_from_source(
            row,
            now_ns=receipt.verified_ns,
        )
        for row in materialization_sources
    )
    coverages = tuple(
        _signed_coverage_from_source(
            row,
            now_ns=receipt.verified_ns,
        )
        for row in coverage_sources
    )
    e2_evidence, e4_evidence = _prefix_evidence(rebuilt_prefixes)
    signed_prefix_result_headers = _bind_prefix_result_headers(
        receipt,
        rebuilt_prefixes,
        now_ns=receipt.verified_ns,
    )
    replay_shard_sources = tuple(
        CanonicalJsonProofBinding.bind(path)
        for path in candidate_replay_proof_shard_paths
    )
    current_replay_proofs = _replay_proofs_from_shards(
        replay_shard_sources,
        semantic_receipt_sha256=receipt.sha256,
    )
    if current_replay_proofs != _current_replay_proofs(receipt, prior):
        raise ValueError("formal registry replay proof shard delta differs")
    tts_reduction_sources = tuple(
        CanonicalJsonProofBinding.bind(path)
        for path in tts_calibration_reduction_proof_paths
    )
    if len(tts_reduction_sources) != len(receipt.appended_signed_tts_calibration_seals):
        raise ValueError("formal registry TTS reduction proof coverage differs")
    if tts_reduction_sources:
        from lightcone_spec.experiments.tts_calibration_authority import (
            revalidate_signed_tts_calibration_seal_from_reduction_proof,
        )

        policy = receipt.trusted_release_policy(current_ns=receipt.verified_ns)
        for source, signed in zip(
            tts_reduction_sources,
            receipt.appended_signed_tts_calibration_seals,
            strict=True,
        ):
            revalidate_signed_tts_calibration_seal_from_reduction_proof(
                source.absolute_path,
                signed_seal=signed,
                policy=policy,
                expected_policy_sha256=policy.sha256,
                now_ns=receipt.verified_ns,
            )
    e3a_proof_sources = tuple(
        CanonicalJsonProofBinding.bind(path)
        for path in e3a_staged_selection_proof_paths
    )
    signed_e3a_sources = tuple(
        CanonicalJsonProofBinding.bind(path)
        for path in signed_e3a_staged_selection_paths
    )
    e3a_artifacts, e3a_results = _rebuild_e3a_selection_delta(
        e3a_proof_sources,
        signed_e3a_sources,
        now_ns=receipt.verified_ns,
    )
    if (
        materializations != receipt.appended_signed_materializations
        or coverages != receipt.appended_signed_coverage
        or e3a_artifacts != receipt.appended_e3a_staged_selection_artifacts
        or e3a_results != receipt.appended_signed_e3a_staged_selections
        or prefix_sources != receipt.appended_formal_stage_prefix_artifacts
        or e2_evidence != receipt.appended_e2_staged_evidence_manifests
        or e4_evidence != receipt.appended_e4_staged_evidence_manifests
    ):
        raise ValueError("formal registry layer current proof delta differs")
    encoded = receipt.to_dict()
    small = {
        name: value
        for name, value in encoded.items()
        if name not in _REPLACED_RECEIPT_FIELDS
    }
    raw_manifest_fields = encoded["manifest"]
    if type(raw_manifest_fields) is not dict:  # pragma: no cover - receipt codec
        raise AssertionError("formal registry manifest codec is not an object")
    manifest_fields = dict(raw_manifest_fields)
    manifest_fields.pop("candidate_replay_proofs")
    artifact = FormalRegistryLayerArtifact(
        schema_version=5,
        kind=FORMAL_REGISTRY_LAYER_ARTIFACT_KIND,
        semantic_receipt_sha256=receipt.sha256,
        prior_layer_source=prior_source,
        signed_protocol_lock_source=protocol_lock_source,
        signed_materialization_sources=materialization_sources,
        signed_coverage_sources=coverage_sources,
        formal_stage_prefix_sources=prefix_sources,
        signed_prefix_result_headers=signed_prefix_result_headers,
        candidate_replay_proof_shard_sources=replay_shard_sources,
        tts_calibration_reduction_proof_sources=tts_reduction_sources,
        e3a_staged_selection_proof_sources=e3a_proof_sources,
        signed_e3a_staged_selection_sources=signed_e3a_sources,
        manifest_fields=manifest_fields,
        small_receipt_fields=small,
    )
    artifact.__post_init__()
    return artifact


def publish_formal_registry_replay_proof_shards(
    receipt: FormalRegistryVerificationReceipt,
    *,
    prior_receipt: FormalRegistryVerificationReceipt | None,
    candidate_replay_proof_paths: tuple[str | Path, ...],
    shard_output_paths: tuple[str | Path, ...],
    maximum_proofs_per_shard: int = 256,
) -> tuple[CanonicalJsonProofBinding, ...]:
    """Publish bounded current-delta replay bindings for one registry layer."""

    if type(receipt) is not FormalRegistryVerificationReceipt or (
        prior_receipt is not None
        and type(prior_receipt) is not FormalRegistryVerificationReceipt
    ):
        raise TypeError("formal registry replay shard receipts are not exact")
    if type(
        maximum_proofs_per_shard
    ) is not int or maximum_proofs_per_shard not in range(1, 513):
        raise ValueError("formal registry replay shard bound is outside [1,512]")
    current = _current_replay_proofs(receipt, prior_receipt)
    supplied = tuple(
        CanonicalJsonProofBinding.bind(path) for path in candidate_replay_proof_paths
    )
    if {row.proof_artifact for row in current} != set(supplied) or len(supplied) != len(
        current
    ):
        raise ValueError("formal registry replay proof source set differs")
    chunks = tuple(
        current[index : index + maximum_proofs_per_shard]
        for index in range(0, len(current), maximum_proofs_per_shard)
    )
    if len(shard_output_paths) != len(chunks):
        raise ValueError("formal registry replay shard output count differs")
    result = []
    for index, (chunk, output_path) in enumerate(
        zip(chunks, shard_output_paths, strict=True)
    ):
        shard = FormalRegistryReplayProofShard(
            schema_version=1,
            kind=FORMAL_REGISTRY_REPLAY_PROOF_SHARD_KIND,
            semantic_receipt_sha256=receipt.sha256,
            shard_index=index,
            shard_count=len(chunks),
            proofs=chunk,
        )
        publish_canonical_json_no_replace(output_path, shard.to_dict())
        result.append(CanonicalJsonProofBinding.bind(output_path))
    return tuple(result)


def publish_formal_registry_layer_artifact(
    artifact: FormalRegistryLayerArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not FormalRegistryLayerArtifact:
        raise TypeError("formal registry layer publisher requires exact input")
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


__all__ = (
    "FORMAL_REGISTRY_LAYER_ARTIFACT_KIND",
    "FORMAL_REGISTRY_REPLAY_PROOF_SHARD_KIND",
    "FormalRegistryLayerArtifact",
    "FormalRegistryReplayProofShard",
    "bind_formal_registry_layer_artifact",
    "load_formal_registry_verification_receipt_path",
    "load_formal_signed_coverage_path",
    "load_formal_signed_materialization_path",
    "publish_formal_registry_layer_artifact",
    "publish_formal_registry_replay_proof_shards",
    "validate_formal_precoverage_registry_state",
)
