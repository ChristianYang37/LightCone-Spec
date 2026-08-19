"""Small backend contract for exact, versioned online adaptation.

The contract deliberately describes only values shared by speculative backends.
Backend-specific tensors remain in an opaque payload whose validator and
reconstructor are owned by that backend. Static and target-only execution do
not construct :class:`ProposalEvidence`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from torch import Tensor

from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

BackendName = Literal["DFLASH", "DSPARK", "EAGLE", "EAGLE3", "NEXTN"]


def _hash_body(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


EAGLE3_COMPATIBILITY_PROTOCOL_SHA256 = _hash_body(
    {
        "schema_version": 1,
        "kind": "lightcone_eagle3_formal_compatibility_authority",
        "authority": (
            "dynamic_release_root_control_binds_target_drafter_interface_"
            "source_and_model_selector"
        ),
        "diagnostic_public_key_maps": "never_formal",
    }
)

EAGLE3_E0_EXECUTION_PROTOCOL_SHA256 = _hash_body(
    {
        "schema_version": 1,
        "kind": "lightcone_eagle3_e0_execution_authority",
        "stage": "E0",
        "runtime_methods": [
            "target_only",
            "static",
            "tts",
            "l0",
            "onlinespec_ogd",
            "onlinespec_opt",
            "onlinespec_ens",
        ],
        "compatibility": "verified_official_selector_authority",
        "qualification": "suite_specific_eagle3_tp1_gpu_proof",
        "scope": "exact_model_interface_inventory_and_gpu",
    }
)

EAGLE3_OFFICIAL_SELECTOR_CONTENT_PROTOCOL_SHA256 = _hash_body(
    {
        "schema_version": 1,
        "kind": "lightcone_eagle3_official_selector_content_authority",
        "source": (
            "path_bound_selector_manifest_listed_in_the_offline_root_authorized_"
            "prepared_drafter_snapshot"
        ),
        "scope": "exact_E0_target_drafter_backend_task_row",
        "dispositions": ["COMPATIBLE", "N/A"],
        "selector_relative_path": "eagle3_official_selector_manifest.json",
    }
)

EAGLE3_OFFICIAL_SELECTOR_RELATIVE_PATH = "eagle3_official_selector_manifest.json"

_EAGLE3_E0_RUNTIME_METHODS = frozenset(
    {
        "target_only",
        "static",
        "tts",
        "l0",
        "onlinespec_ogd",
        "onlinespec_opt",
        "onlinespec_ens",
    }
)

_EAGLE3_E0_TASKS = (
    "AIME-2025",
    "Alpaca",
    "Arena-Hard",
    "GSM8K",
    "HumanEval",
    "LiveCodeBench",
    "MATH-500",
    "MBPP",
    "MT-Bench",
)

DSPARK_SELECTOR_SCOPES = (
    "all",
    "last1",
    "last1_native_heads",
    "last3",
    "last3_native_heads",
    "last5",
    "last5_native_heads",
)
DSPARK_SELECTOR_LORA_RANKS = (1, 2, 4, 8, 16, 32, 64)
DSPARK_NATIVE_HEAD_NAMES = (
    "markov.w1",
    "markov.w2",
    "acceptance.confidence",
)
DSPARK_SELECTOR_PROTOCOL_SHA256 = _hash_body(
    {
        "schema_version": 1,
        "kind": "lightcone_dspark_56_cell_selector",
        "scopes": DSPARK_SELECTOR_SCOPES,
        "modes": {"full": None, "lora_ranks": DSPARK_SELECTOR_LORA_RANKS},
        "native_head_names": DSPARK_NATIVE_HEAD_NAMES,
        "binding": "exact_model_interface_and_parameter_inventory",
    }
)


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_revision(name: str, value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase 40-hex revision")


def _require_text(name: str, value: str) -> None:
    if not value or value.strip() != value or len(value) > 192:
        raise ValueError(f"{name} must be a canonical non-empty identifier")


class BackendNotApplicable(RuntimeError):
    """Stable fail-closed outcome for a backend lacking native authority."""

    def __init__(self, reason_code: str) -> None:
        _require_text("backend N/A reason", reason_code)
        super().__init__(f"backend is N/A: {reason_code}")
        self.reason_code = reason_code


def dspark_selector_candidate_ids(
    *,
    model_interface_sha256: str,
    parameter_inventory_sha256: str,
) -> tuple[str, ...]:
    """Derive the registered 7 × (Full + seven LoRA ranks) selector."""

    _require_sha256("DSpark model interface", model_interface_sha256)
    _require_sha256("DSpark parameter inventory", parameter_inventory_sha256)
    common = {
        "schema_version": 1,
        "model_interface_sha256": model_interface_sha256,
        "parameter_inventory_sha256": parameter_inventory_sha256,
        "native_head_names": DSPARK_NATIVE_HEAD_NAMES,
    }
    candidates = [
        _hash_body({**common, "scope": scope, "mode": "full", "rank": None})
        for scope in DSPARK_SELECTOR_SCOPES
    ]
    candidates.extend(
        _hash_body({**common, "scope": scope, "mode": "lora", "rank": rank})
        for scope in DSPARK_SELECTOR_SCOPES
        for rank in DSPARK_SELECTOR_LORA_RANKS
    )
    return tuple(sorted(candidates))


_DSPARK_SELECTOR_AUTHORITY_SENTINEL = object()


@dataclass(frozen=True, init=False)
class DSparkSelectorAuthority:
    """Exact 56-candidate DSpark selector and fixed native-head identity."""

    schema_version: int
    selector_protocol_sha256: str
    model_interface_sha256: str
    parameter_inventory_sha256: str
    candidate_ids: tuple[str, ...]
    native_head_names: tuple[str, str, str]

    def __init__(
        self,
        *,
        model_interface_sha256: str,
        parameter_names: Sequence[str],
        supplied_candidate_ids: tuple[str, ...],
        native_head_names: tuple[str, str, str] = DSPARK_NATIVE_HEAD_NAMES,
        _verification_tag: object = _DSPARK_SELECTOR_AUTHORITY_SENTINEL,
    ) -> None:
        if _verification_tag is not _DSPARK_SELECTOR_AUTHORITY_SENTINEL:
            raise TypeError("DSpark selector authority requires the source reducer")
        _require_sha256("DSpark model interface", model_interface_sha256)
        names = tuple(parameter_names)
        if (
            not names
            or names != tuple(sorted(set(names)))
            or any(type(name) is not str or not name.strip() for name in names)
        ):
            raise ValueError("DSpark parameter inventory must be sorted and unique")
        if any(name not in names for name in DSPARK_NATIVE_HEAD_NAMES):
            raise ValueError("DSpark parameter inventory lacks exact native heads")
        inventory_sha256 = _hash_body(
            {
                "schema_version": 1,
                "model_interface_sha256": model_interface_sha256,
                "parameter_names": names,
            }
        )
        expected_candidates = dspark_selector_candidate_ids(
            model_interface_sha256=model_interface_sha256,
            parameter_inventory_sha256=inventory_sha256,
        )
        if supplied_candidate_ids != expected_candidates:
            raise ValueError(
                "DSpark supplied selector differs from the exact 56-cell grid"
            )
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(
            self, "selector_protocol_sha256", DSPARK_SELECTOR_PROTOCOL_SHA256
        )
        object.__setattr__(self, "model_interface_sha256", model_interface_sha256)
        object.__setattr__(self, "parameter_inventory_sha256", inventory_sha256)
        object.__setattr__(self, "candidate_ids", supplied_candidate_ids)
        object.__setattr__(self, "native_head_names", native_head_names)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("DSpark selector authority schema is unsupported")
        if self.selector_protocol_sha256 != DSPARK_SELECTOR_PROTOCOL_SHA256:
            raise ValueError("DSpark selector protocol differs from source policy")
        _require_sha256("DSpark model interface", self.model_interface_sha256)
        _require_sha256("DSpark parameter inventory", self.parameter_inventory_sha256)
        expected_candidates = dspark_selector_candidate_ids(
            model_interface_sha256=self.model_interface_sha256,
            parameter_inventory_sha256=self.parameter_inventory_sha256,
        )
        if (
            type(self.candidate_ids) is not tuple
            or len(self.candidate_ids) != 56
            or self.candidate_ids != tuple(sorted(set(self.candidate_ids)))
            or self.candidate_ids != expected_candidates
        ):
            raise ValueError(
                "DSpark selector authority requires 56 sorted unique candidates"
            )
        for candidate_id in self.candidate_ids:
            _require_sha256("DSpark selector candidate", candidate_id)
        if self.native_head_names != DSPARK_NATIVE_HEAD_NAMES:
            raise ValueError("DSpark authority requires exact W1/W2/confidence heads")
        for name in self.native_head_names:
            _require_text("DSpark native head", name)

    @property
    def sha256(self) -> str:
        return _hash_body(
            {
                "schema_version": self.schema_version,
                "selector_protocol_sha256": self.selector_protocol_sha256,
                "model_interface_sha256": self.model_interface_sha256,
                "parameter_inventory_sha256": self.parameter_inventory_sha256,
                "candidate_ids": self.candidate_ids,
                "native_head_names": self.native_head_names,
            }
        )


@dataclass(frozen=True)
class NextNTwoModelTp2Authority:
    """Structural target+MTP sharding authority; never self-enables release use."""

    schema_version: int
    interface_sha256: str
    target_revision: str
    drafter_revision: str
    target_shard_manifest_sha256: str
    drafter_shard_manifest_sha256: str
    topology_sha256: str
    source_adapter_version: int
    status: Literal["CPU_CONTRACT_ONLY", "GPU_VERIFIED"]
    gpu_proof_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("NEXTN TP2 authority schema is unsupported")
        for name, value in (
            ("NEXTN interface", self.interface_sha256),
            ("NEXTN target shard manifest", self.target_shard_manifest_sha256),
            ("NEXTN drafter shard manifest", self.drafter_shard_manifest_sha256),
            ("NEXTN topology", self.topology_sha256),
        ):
            _require_sha256(name, value)
        _require_revision("NEXTN target revision", self.target_revision)
        _require_revision("NEXTN drafter revision", self.drafter_revision)
        if (
            type(self.source_adapter_version) is not int
            or self.source_adapter_version < 0
        ):
            raise ValueError("NEXTN source adapter version must be non-negative")
        if self.status == "CPU_CONTRACT_ONLY":
            if self.gpu_proof_sha256 is not None:
                raise ValueError("CPU-only NEXTN authority cannot carry GPU proof")
        elif self.status == "GPU_VERIFIED":
            if self.gpu_proof_sha256 is None:
                raise ValueError("GPU-verified NEXTN authority requires proof")
            _require_sha256("NEXTN GPU proof", self.gpu_proof_sha256)
        else:
            raise ValueError("NEXTN TP2 authority status is unsupported")

    @property
    def sha256(self) -> str:
        return _hash_body(
            {
                "schema_version": self.schema_version,
                "interface_sha256": self.interface_sha256,
                "target_revision": self.target_revision,
                "drafter_revision": self.drafter_revision,
                "target_shard_manifest_sha256": self.target_shard_manifest_sha256,
                "drafter_shard_manifest_sha256": self.drafter_shard_manifest_sha256,
                "topology_sha256": self.topology_sha256,
                "source_adapter_version": self.source_adapter_version,
                "status": self.status,
                "gpu_proof_sha256": self.gpu_proof_sha256,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "interface_sha256": self.interface_sha256,
            "target_revision": self.target_revision,
            "drafter_revision": self.drafter_revision,
            "target_shard_manifest_sha256": self.target_shard_manifest_sha256,
            "drafter_shard_manifest_sha256": self.drafter_shard_manifest_sha256,
            "topology_sha256": self.topology_sha256,
            "source_adapter_version": self.source_adapter_version,
            "status": self.status,
            "gpu_proof_sha256": self.gpu_proof_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> NextNTwoModelTp2Authority:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "interface_sha256",
            "target_revision",
            "drafter_revision",
            "target_shard_manifest_sha256",
            "drafter_shard_manifest_sha256",
            "topology_sha256",
            "source_adapter_version",
            "status",
            "gpu_proof_sha256",
        }:
            raise ValueError("NEXTN TP2 authority fields differ")
        return cls(**value)


# A semantic patch and GPU proof must add an exact source-owned digest here.
# A caller-authored authority, even one claiming GPU_VERIFIED, cannot do so.
RELEASE_NEXTN_TP2_AUTHORITY_SHA256S: tuple[str, ...] = ()


def require_release_nextn_tp2_authority(
    authority: NextNTwoModelTp2Authority,
) -> None:
    if type(authority) is not NextNTwoModelTp2Authority:
        raise TypeError("NEXTN TP2 requires an exact two-model authority")
    authority.__post_init__()
    if (
        authority.status != "GPU_VERIFIED"
        or authority.sha256 not in RELEASE_NEXTN_TP2_AUTHORITY_SHA256S
    ):
        raise BackendNotApplicable("nextn_tp2_native_gpu_authority_unavailable")


NEXTN_TP2_DYNAMIC_AUTHORITY_PROTOCOL_SHA256 = _hash_body(
    {
        "schema_version": 1,
        "kind": "lightcone_nextn_tp2_dynamic_authority",
        "native_suite": "nextn_tp2",
        "distributed_mode": "tp2_dp1",
        "models": "root_authorized_target_and_drafter_snapshot_shards",
        "trust": "path_bound_external_control_and_atomic_replay",
        "legacy_static_allowlist": "diagnostic_only",
    }
)


def nextn_snapshot_shard_manifest_sha256(snapshot: object) -> str:
    """Derive the exact model-shard identity from a prepared snapshot.

    The local import keeps the backend contract independent from model-lock
    construction while ensuring this digest can only be derived from the
    validated prepared-content representation.
    """

    from lightcone_spec.locking.prepared_models import PreparedModelSnapshotContent

    if type(snapshot) is not PreparedModelSnapshotContent:
        raise TypeError("NEXTN shard identity requires prepared snapshot content")
    snapshot.__post_init__()
    return _hash_body(
        {
            "schema_version": 1,
            "kind": "lightcone_nextn_prepared_shard_manifest",
            "model_id": snapshot.model_id,
            "revision": snapshot.revision,
            "weight_kind": snapshot.weight_kind,
            "tensor_metadata_sha256": snapshot.tensor_metadata_sha256,
            "shards": [
                {
                    "relative_path": row.relative_path,
                    "file_size": row.file_size,
                    "header_size": row.header_size,
                    "header_sha256": row.header_sha256,
                    "raw_sha256": row.raw_sha256,
                }
                for row in snapshot.weight_headers
            ],
        }
    )


@dataclass(frozen=True)
class NextNTp2DynamicAuthorityArtifact:
    """Unsigned composition of already trusted, durable release proofs.

    This wrapper grants no authority by itself.  Its validator reopens every
    bound artifact, rechecks signatures and replay reservations at their
    recorded acceptance times, and only then returns a private runtime token.
    """

    schema_version: int
    kind: Literal["lightcone_nextn_tp2_dynamic_authority_artifact"]
    protocol_sha256: str
    authority: NextNTwoModelTp2Authority
    native_gpu_proof: CanonicalJsonProofBinding
    distributed_gpu_proof: CanonicalJsonProofBinding
    content_verification_receipt: CanonicalJsonProofBinding
    target_member_id: str
    drafter_member_id: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "lightcone_nextn_tp2_dynamic_authority_artifact"
        ):
            raise ValueError("NEXTN dynamic authority artifact schema is unsupported")
        if self.protocol_sha256 != NEXTN_TP2_DYNAMIC_AUTHORITY_PROTOCOL_SHA256:
            raise ValueError("NEXTN dynamic authority artifact protocol differs")
        if type(self.authority) is not NextNTwoModelTp2Authority:
            raise TypeError("NEXTN dynamic artifact requires structural authority")
        self.authority.__post_init__()
        for binding in (
            self.native_gpu_proof,
            self.distributed_gpu_proof,
            self.content_verification_receipt,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("NEXTN dynamic artifact requires path-bound inputs")
            binding.__post_init__()
        _require_text("NEXTN target member", self.target_member_id)
        _require_text("NEXTN drafter member", self.drafter_member_id)
        if self.target_member_id == self.drafter_member_id:
            raise ValueError("NEXTN target and drafter members must differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "authority": self.authority.to_dict(),
            "native_gpu_proof": self.native_gpu_proof.to_dict(),
            "distributed_gpu_proof": self.distributed_gpu_proof.to_dict(),
            "content_verification_receipt": (
                self.content_verification_receipt.to_dict()
            ),
            "target_member_id": self.target_member_id,
            "drafter_member_id": self.drafter_member_id,
        }

    @property
    def sha256(self) -> str:
        return _hash_body(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> NextNTp2DynamicAuthorityArtifact:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "protocol_sha256",
            "authority",
            "native_gpu_proof",
            "distributed_gpu_proof",
            "content_verification_receipt",
            "target_member_id",
            "drafter_member_id",
        }:
            raise ValueError("NEXTN dynamic authority artifact fields differ")
        row = dict(value)
        return cls(
            authority=NextNTwoModelTp2Authority.from_dict(row.pop("authority")),
            native_gpu_proof=CanonicalJsonProofBinding.from_dict(
                row.pop("native_gpu_proof")
            ),
            distributed_gpu_proof=CanonicalJsonProofBinding.from_dict(
                row.pop("distributed_gpu_proof")
            ),
            content_verification_receipt=CanonicalJsonProofBinding.from_dict(
                row.pop("content_verification_receipt")
            ),
            **row,
        )


_VERIFIED_NEXTN_TP2_AUTHORITY_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedNextNTp2Authority:
    """Verifier-owned authorization consumed by the live NEXTN backend."""

    artifact_sha256: str
    authority_sha256: str
    interface_sha256: str
    target_model_id: str
    drafter_model_id: str
    target_revision: str
    drafter_revision: str
    target_shard_manifest_sha256: str
    drafter_shard_manifest_sha256: str
    topology_sha256: str
    source_adapter_version: int
    native_gpu_proof_sha256: str
    distributed_gpu_proof_sha256: str
    content_verification_receipt_sha256: str
    inventory_sha256: str
    registry_sha256: str
    root_manifest_sha256: str
    gpu_uuids: tuple[str, str]

    def __init__(
        self,
        *,
        artifact_sha256: str,
        authority: NextNTwoModelTp2Authority,
        target_model_id: str,
        drafter_model_id: str,
        native_gpu_proof_sha256: str,
        distributed_gpu_proof_sha256: str,
        content_verification_receipt_sha256: str,
        inventory_sha256: str,
        registry_sha256: str,
        root_manifest_sha256: str,
        gpu_uuids: tuple[str, str],
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_NEXTN_TP2_AUTHORITY_SENTINEL:
            raise TypeError("verified NEXTN TP2 authority is verifier-owned")
        values = {
            "artifact_sha256": artifact_sha256,
            "authority_sha256": authority.sha256,
            "interface_sha256": authority.interface_sha256,
            "target_model_id": target_model_id,
            "drafter_model_id": drafter_model_id,
            "target_revision": authority.target_revision,
            "drafter_revision": authority.drafter_revision,
            "target_shard_manifest_sha256": (authority.target_shard_manifest_sha256),
            "drafter_shard_manifest_sha256": (authority.drafter_shard_manifest_sha256),
            "topology_sha256": authority.topology_sha256,
            "source_adapter_version": authority.source_adapter_version,
            "native_gpu_proof_sha256": native_gpu_proof_sha256,
            "distributed_gpu_proof_sha256": distributed_gpu_proof_sha256,
            "content_verification_receipt_sha256": (
                content_verification_receipt_sha256
            ),
            "inventory_sha256": inventory_sha256,
            "registry_sha256": registry_sha256,
            "root_manifest_sha256": root_manifest_sha256,
            "gpu_uuids": gpu_uuids,
        }
        for name, item in values.items():
            object.__setattr__(self, name, item)
        for label, digest in (
            ("NEXTN artifact", self.artifact_sha256),
            ("NEXTN authority", self.authority_sha256),
            ("NEXTN native proof", self.native_gpu_proof_sha256),
            ("NEXTN distributed proof", self.distributed_gpu_proof_sha256),
            ("NEXTN content receipt", self.content_verification_receipt_sha256),
            ("NEXTN inventory", self.inventory_sha256),
            ("NEXTN registry", self.registry_sha256),
            ("NEXTN release root", self.root_manifest_sha256),
        ):
            _require_sha256(label, digest)
        _require_text("NEXTN target model", self.target_model_id)
        _require_text("NEXTN drafter model", self.drafter_model_id)
        if self.target_model_id == self.drafter_model_id:
            raise ValueError("verified NEXTN target and drafter models must differ")
        if len(self.gpu_uuids) != 2 or len(set(self.gpu_uuids)) != 2:
            raise ValueError("verified NEXTN TP2 authority requires two GPUs")

    @property
    def sha256(self) -> str:
        return _hash_body(self.__dict__)


def nextn_tp2_dynamic_proof_sha256(
    *,
    native_gpu_proof: CanonicalJsonProofBinding,
    distributed_gpu_proof: CanonicalJsonProofBinding,
    content_verification_receipt: CanonicalJsonProofBinding,
    target_member_id: str,
    drafter_member_id: str,
) -> str:
    """Commit the complete upstream proof DAG without an identity cycle."""

    for binding in (
        native_gpu_proof,
        distributed_gpu_proof,
        content_verification_receipt,
    ):
        if type(binding) is not CanonicalJsonProofBinding:
            raise TypeError("NEXTN proof commitment requires exact path bindings")
        binding.__post_init__()
    _require_text("NEXTN target member", target_member_id)
    _require_text("NEXTN drafter member", drafter_member_id)
    return _hash_body(
        {
            "schema_version": 1,
            "protocol_sha256": NEXTN_TP2_DYNAMIC_AUTHORITY_PROTOCOL_SHA256,
            "native_gpu_proof_sha256": native_gpu_proof.semantic_sha256,
            "distributed_gpu_proof_sha256": (distributed_gpu_proof.semantic_sha256),
            "content_verification_receipt_sha256": (
                content_verification_receipt.semantic_sha256
            ),
            "target_member_id": target_member_id,
            "drafter_member_id": drafter_member_id,
        }
    )


def publish_nextn_tp2_dynamic_authority_artifact(
    output_path: str,
    *,
    authority: NextNTwoModelTp2Authority,
    native_gpu_proof_artifact_path: str,
    distributed_gpu_proof_artifact_path: str,
    content_verification_receipt_path: str,
    target_member_id: str,
    drafter_member_id: str,
) -> CanonicalJsonProofBinding:
    """Publish a no-replace wrapper; validation still performs the trust lift."""

    native = CanonicalJsonProofBinding.bind(native_gpu_proof_artifact_path)
    distributed = CanonicalJsonProofBinding.bind(distributed_gpu_proof_artifact_path)
    content = CanonicalJsonProofBinding.bind(content_verification_receipt_path)
    expected_proof = nextn_tp2_dynamic_proof_sha256(
        native_gpu_proof=native,
        distributed_gpu_proof=distributed,
        content_verification_receipt=content,
        target_member_id=target_member_id,
        drafter_member_id=drafter_member_id,
    )
    if (
        type(authority) is not NextNTwoModelTp2Authority
        or authority.status != "GPU_VERIFIED"
        or authority.gpu_proof_sha256 != expected_proof
    ):
        raise ValueError(
            "NEXTN structural authority differs from its dynamic proof DAG"
        )
    artifact = NextNTp2DynamicAuthorityArtifact(
        schema_version=1,
        kind="lightcone_nextn_tp2_dynamic_authority_artifact",
        protocol_sha256=NEXTN_TP2_DYNAMIC_AUTHORITY_PROTOCOL_SHA256,
        authority=authority,
        native_gpu_proof=native,
        distributed_gpu_proof=distributed,
        content_verification_receipt=content,
        target_member_id=target_member_id,
        drafter_member_id=drafter_member_id,
    )
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(
        output_path,
        semantic_sha256=artifact.sha256,
    )


def _nextn_prepared_snapshot(
    *,
    content_receipt: object,
    prepared_release: object,
    member: object,
) -> object:
    """Reopen the authorized manifest and return one exact snapshot row."""

    from lightcone_spec.locking.prepared_models import PreparedModelSnapshotContent
    from lightcone_spec.runtime.content_authorization import AuthorizedPreparedModel

    if type(member) is not AuthorizedPreparedModel:
        raise TypeError("NEXTN content member is not a prepared-model authority")
    release = prepared_release.authorization
    matches = tuple(
        artifact
        for artifact in content_receipt.content_artifacts
        if artifact.raw_sha256 == release.content_manifest_raw_sha256
        and artifact.semantic_sha256 == release.content_manifest_semantic_sha256
        and artifact.size == release.content_manifest_size
    )
    if len(matches) != 1:
        raise ValueError("NEXTN content receipt lacks one exact model manifest")
    manifest = matches[0].load()
    if type(manifest) is not dict or type(manifest.get("snapshots")) is not list:
        raise TypeError("NEXTN prepared model manifest is malformed")
    snapshots = tuple(
        PreparedModelSnapshotContent.from_dict(row) for row in manifest["snapshots"]
    )
    selected = tuple(
        row
        for row in snapshots
        if row.model_id == member.model_id and row.revision == member.revision
    )
    if len(selected) != 1:
        raise ValueError("NEXTN member lacks one exact prepared snapshot")
    snapshot = selected[0]
    snapshot_sha256 = _hash_body(snapshot.to_dict())
    if (
        member.snapshot_manifest_raw_sha256 != snapshot_sha256
        or member.snapshot_manifest_semantic_sha256 != snapshot_sha256
    ):
        raise ValueError("NEXTN member snapshot identity differs from prepared bytes")
    return snapshot


def _prepared_release_from_receipt(content_receipt: object, *, now_ns: int) -> object:
    from lightcone_spec.runtime.content_authorization import (
        ContentVerificationReceipt,
        VerifiedPreparedModelContentRelease,
    )

    if type(content_receipt) is not ContentVerificationReceipt:
        raise TypeError("NEXTN requires a durable content verification receipt")
    verified = content_receipt.revalidate(current_ns=now_ns)
    prepared = tuple(
        row for row in verified if type(row) is VerifiedPreparedModelContentRelease
    )
    if len(prepared) != 1:
        raise ValueError("NEXTN requires one exact prepared-model authorization")
    return prepared[0]


def validate_nextn_tp2_dynamic_authority_artifact(
    artifact_path: str,
    *,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_interface_sha256: str,
    expected_topology_sha256: str,
    expected_source_adapter_version: int,
    expected_target_member_id: str,
    expected_drafter_member_id: str,
    now_ns: int,
) -> VerifiedNextNTp2Authority:
    """Deep-reopen a two-model authority and return its unforgeable token."""

    from lightcone_spec.runtime.content_authorization import (
        ContentVerificationReceipt,
    )
    from lightcone_spec.runtime.distributed import (
        DistributedRuntimeGpuProofArtifact,
    )
    from lightcone_spec.runtime.readiness import NativeRuntimeGpuProofArtifact

    for label, digest in (
        ("NEXTN expected inventory", expected_inventory_sha256),
        ("NEXTN expected registry", expected_registry_sha256),
        ("NEXTN expected release root", expected_root_manifest_sha256),
        ("NEXTN expected interface", expected_interface_sha256),
        ("NEXTN expected topology", expected_topology_sha256),
    ):
        _require_sha256(label, digest)
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("NEXTN authority verification time is invalid")
    if type(expected_source_adapter_version) is not int or (
        expected_source_adapter_version < 0
    ):
        raise ValueError("NEXTN expected source adapter version is invalid")
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = NextNTp2DynamicAuthorityArtifact.from_dict(binding.reopen())
    if artifact.sha256 != binding.semantic_sha256:
        raise ValueError("NEXTN authority artifact semantic identity changed")
    if (
        artifact.target_member_id != expected_target_member_id
        or artifact.drafter_member_id != expected_drafter_member_id
    ):
        raise ValueError("NEXTN authority model members differ from assignment")
    expected_dynamic_proof = nextn_tp2_dynamic_proof_sha256(
        native_gpu_proof=artifact.native_gpu_proof,
        distributed_gpu_proof=artifact.distributed_gpu_proof,
        content_verification_receipt=artifact.content_verification_receipt,
        target_member_id=artifact.target_member_id,
        drafter_member_id=artifact.drafter_member_id,
    )
    authority = artifact.authority
    if (
        authority.status != "GPU_VERIFIED"
        or authority.gpu_proof_sha256 != expected_dynamic_proof
        or authority.interface_sha256 != expected_interface_sha256
        or authority.topology_sha256 != expected_topology_sha256
        or authority.source_adapter_version != expected_source_adapter_version
    ):
        raise ValueError("NEXTN structural authority differs from assignment/proofs")

    native_artifact = NativeRuntimeGpuProofArtifact.from_dict(
        artifact.native_gpu_proof.reopen()
    )
    distributed_artifact = DistributedRuntimeGpuProofArtifact.from_dict(
        artifact.distributed_gpu_proof.reopen()
    )
    content_receipt = ContentVerificationReceipt.from_dict(
        artifact.content_verification_receipt.reopen()
    )
    native = native_artifact.revalidate(now_ns=now_ns)
    distributed = distributed_artifact.revalidate(now_ns=now_ns)
    prepared = _prepared_release_from_receipt(content_receipt, now_ns=now_ns)
    if (
        native.suite_id != "nextn_tp2"
        or native.topology_mode != "tp2_dp1"
        or "nextn" not in native.backend_capabilities
        or distributed.topology_mode != "tp2_dp1"
        or native.inventory_sha256 != expected_inventory_sha256
        or distributed.inventory_sha256 != expected_inventory_sha256
        or native.source_identity_sha256 != expected_registry_sha256
        or distributed.source_identity_sha256 != expected_registry_sha256
        or native.topology_sha256 != expected_topology_sha256
        or distributed.topology_sha256 != expected_topology_sha256
        or native.gpu_uuids != distributed.gpu_uuids
    ):
        raise ValueError("NEXTN native/distributed proof identity is foreign")
    roots = {
        native_artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256,
        distributed_artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256,
        prepared.authorization.root_manifest_sha256,
    }
    if roots != {expected_root_manifest_sha256}:
        raise ValueError("NEXTN proof DAG uses a foreign release root")
    replay_roots = {
        str(Path(native_artifact.replay_reservation.path).parent),
        str(Path(distributed_artifact.replay_reservation.path).parent),
        str(Path(content_receipt.reservation.path).parent),
    }
    if len(replay_roots) != 1:
        raise ValueError("NEXTN proof DAG does not share one replay ledger")

    target = prepared.member(artifact.target_member_id)
    drafter = prepared.member(artifact.drafter_member_id)
    if target.role != "target" or drafter.role != "drafter":
        raise ValueError("NEXTN prepared members have the wrong model roles")
    if drafter.backend != "NEXTN":
        raise ValueError("NEXTN drafter is not authorized for the NEXTN backend")
    target_snapshot = _nextn_prepared_snapshot(
        content_receipt=content_receipt,
        prepared_release=prepared,
        member=target,
    )
    drafter_snapshot = _nextn_prepared_snapshot(
        content_receipt=content_receipt,
        prepared_release=prepared,
        member=drafter,
    )
    if (
        authority.target_revision != target.revision
        or authority.drafter_revision != drafter.revision
        or authority.target_shard_manifest_sha256
        != nextn_snapshot_shard_manifest_sha256(target_snapshot)
        or authority.drafter_shard_manifest_sha256
        != nextn_snapshot_shard_manifest_sha256(drafter_snapshot)
    ):
        raise ValueError("NEXTN authority differs from prepared model shards")
    return VerifiedNextNTp2Authority(
        artifact_sha256=binding.semantic_sha256,
        authority=authority,
        target_model_id=target.model_id,
        drafter_model_id=drafter.model_id,
        native_gpu_proof_sha256=native.sha256,
        distributed_gpu_proof_sha256=distributed.sha256,
        content_verification_receipt_sha256=content_receipt.sha256,
        inventory_sha256=expected_inventory_sha256,
        registry_sha256=expected_registry_sha256,
        root_manifest_sha256=expected_root_manifest_sha256,
        gpu_uuids=native.gpu_uuids,
        _verification_tag=_VERIFIED_NEXTN_TP2_AUTHORITY_SENTINEL,
    )


@dataclass(frozen=True)
class Eagle3CompatibilityAuthority:
    """Signed compatible/N/A decision for one exact EAGLE3 model interface."""

    schema_version: int
    status: Literal["COMPATIBLE", "N/A"]
    target_revision: str
    drafter_revision: str
    interface_sha256: str
    source_commit: str
    model_selector_sha256: str
    reason_code: str
    signer_key_id: str
    signature_hex: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("EAGLE3 compatibility authority schema is unsupported")
        if self.status not in {"COMPATIBLE", "N/A"}:
            raise ValueError("EAGLE3 compatibility status is unsupported")
        _require_revision("EAGLE3 target revision", self.target_revision)
        _require_revision("EAGLE3 drafter revision", self.drafter_revision)
        _require_sha256("EAGLE3 interface", self.interface_sha256)
        _require_revision("EAGLE3 source commit", self.source_commit)
        _require_sha256("EAGLE3 model selector", self.model_selector_sha256)
        _require_text("EAGLE3 compatibility reason", self.reason_code)
        _require_text("EAGLE3 signer key", self.signer_key_id)
        if len(self.signature_hex) != 128 or any(
            character not in "0123456789abcdef" for character in self.signature_hex
        ):
            raise ValueError("EAGLE3 signature must be lowercase Ed25519 hex")

    @property
    def message(self) -> bytes:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "status": self.status,
                "target_revision": self.target_revision,
                "drafter_revision": self.drafter_revision,
                "interface_sha256": self.interface_sha256,
                "source_commit": self.source_commit,
                "model_selector_sha256": self.model_selector_sha256,
                "reason_code": self.reason_code,
                "signer_key_id": self.signer_key_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            self.message + bytes.fromhex(self.signature_hex)
        ).hexdigest()


@dataclass(frozen=True)
class Eagle3OfficialSelectorRow:
    """One root-reviewed official-selector disposition for an E0 task."""

    task: str
    status: Literal["COMPATIBLE", "N/A"]
    interface_sha256: str
    reason_code: str

    def __post_init__(self) -> None:
        _require_text("official EAGLE3 task", self.task)
        if self.status not in {"COMPATIBLE", "N/A"}:
            raise ValueError("official EAGLE3 selector disposition is unsupported")
        _require_sha256("official EAGLE3 row interface", self.interface_sha256)
        _require_text("official EAGLE3 row reason", self.reason_code)

    def to_dict(self) -> dict[str, str]:
        return {
            "task": self.task,
            "status": self.status,
            "interface_sha256": self.interface_sha256,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> Eagle3OfficialSelectorRow:
        if type(value) is not dict or set(value) != {
            "task",
            "status",
            "interface_sha256",
            "reason_code",
        }:
            raise ValueError("official EAGLE3 selector row fields differ")
        return cls(**value)


@dataclass(frozen=True)
class Eagle3OfficialSelectorManifest:
    """Canonical selector asset carried inside an authorized model snapshot.

    This is data discovered after the model is prepared, not a compile-time
    allowlist.  The offline-root prepared-content authorization reviews the
    snapshot digest, and the reducer below deep-opens both that snapshot
    manifest and this exact file before it can construct a verified token.
    """

    schema_version: Literal[1]
    kind: Literal["lightcone_eagle3_official_selector_manifest"]
    protocol_sha256: str
    backend: Literal["EAGLE3"]
    target_member_id: str
    drafter_member_id: str
    target_model_id: str
    drafter_model_id: str
    target_revision: str
    drafter_revision: str
    source_repository: str
    source_commit: str
    rows: tuple[Eagle3OfficialSelectorRow, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_eagle3_official_selector_manifest"
            or self.protocol_sha256 != EAGLE3_OFFICIAL_SELECTOR_CONTENT_PROTOCOL_SHA256
            or self.backend != "EAGLE3"
        ):
            raise ValueError("official EAGLE3 selector manifest schema is unsupported")
        for label, value in (
            ("target member", self.target_member_id),
            ("drafter member", self.drafter_member_id),
            ("target model", self.target_model_id),
            ("drafter model", self.drafter_model_id),
            ("source repository", self.source_repository),
        ):
            _require_text(f"official EAGLE3 {label}", value)
        _require_revision("official EAGLE3 target revision", self.target_revision)
        _require_revision("official EAGLE3 drafter revision", self.drafter_revision)
        _require_revision("official EAGLE3 source commit", self.source_commit)
        if (
            type(self.rows) is not tuple
            or not self.rows
            or any(type(row) is not Eagle3OfficialSelectorRow for row in self.rows)
            or tuple(row.task for row in self.rows) != _EAGLE3_E0_TASKS
        ):
            raise ValueError(
                "official EAGLE3 selector rows must cover the exact E0 task universe"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "backend": self.backend,
            "target_member_id": self.target_member_id,
            "drafter_member_id": self.drafter_member_id,
            "target_model_id": self.target_model_id,
            "drafter_model_id": self.drafter_model_id,
            "target_revision": self.target_revision,
            "drafter_revision": self.drafter_revision,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "rows": [row.to_dict() for row in self.rows],
        }

    @property
    def sha256(self) -> str:
        return _hash_body(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Eagle3OfficialSelectorManifest:
        fields = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "backend",
            "target_member_id",
            "drafter_member_id",
            "target_model_id",
            "drafter_model_id",
            "target_revision",
            "drafter_revision",
            "source_repository",
            "source_commit",
            "rows",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("official EAGLE3 selector manifest fields differ")
        payload = dict(value)
        rows = payload.pop("rows")
        if type(rows) is not list:
            raise TypeError("official EAGLE3 selector rows must be an array")
        return cls(
            **payload,
            rows=tuple(Eagle3OfficialSelectorRow.from_dict(row) for row in rows),
        )


_VERIFIED_EAGLE3_OFFICIAL_SELECTOR_CONTENT_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedEagle3OfficialSelectorContentAuthority:
    """Verifier-owned projection of one prepared, root-authorized selector row."""

    protocol_sha256: str
    stage: Literal["E0"]
    backend: Literal["EAGLE3"]
    task: str
    status: Literal["COMPATIBLE", "N/A"]
    reason_code: str
    target_member_id: str
    drafter_member_id: str
    target_model_id: str
    drafter_model_id: str
    target_revision: str
    drafter_revision: str
    interface_sha256: str
    source_repository: str
    source_commit: str
    model_selector_sha256: str
    selector_asset_path: str
    selector_asset_raw_sha256: str
    selector_asset_semantic_sha256: str
    target_snapshot_raw_sha256: str
    target_snapshot_semantic_sha256: str
    drafter_snapshot_raw_sha256: str
    drafter_snapshot_semantic_sha256: str
    prepared_content_receipt_path: str
    prepared_content_receipt_raw_sha256: str
    prepared_content_receipt_sha256: str
    prepared_content_authorization_sha256: str
    root_manifest_sha256: str

    def __init__(
        self,
        *,
        manifest: Eagle3OfficialSelectorManifest,
        row: Eagle3OfficialSelectorRow,
        selector_asset: CanonicalJsonProofBinding,
        target_snapshot_raw_sha256: str,
        target_snapshot_semantic_sha256: str,
        drafter_snapshot_raw_sha256: str,
        drafter_snapshot_semantic_sha256: str,
        prepared_content_receipt: CanonicalJsonProofBinding,
        prepared_content_authorization_sha256: str,
        root_manifest_sha256: str,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_EAGLE3_OFFICIAL_SELECTOR_CONTENT_SENTINEL:
            raise TypeError(
                "verified official EAGLE3 selector requires prepared-content replay"
            )
        model_selector_sha256 = _hash_body(
            {
                "schema_version": 1,
                "protocol_sha256": EAGLE3_OFFICIAL_SELECTOR_CONTENT_PROTOCOL_SHA256,
                "selector_asset_semantic_sha256": selector_asset.semantic_sha256,
                "target_member_id": manifest.target_member_id,
                "drafter_member_id": manifest.drafter_member_id,
                "target_revision": manifest.target_revision,
                "drafter_revision": manifest.drafter_revision,
                "source_commit": manifest.source_commit,
                "row": row.to_dict(),
            }
        )
        values = {
            "protocol_sha256": EAGLE3_OFFICIAL_SELECTOR_CONTENT_PROTOCOL_SHA256,
            "stage": "E0",
            "backend": "EAGLE3",
            "task": row.task,
            "status": row.status,
            "reason_code": row.reason_code,
            "target_member_id": manifest.target_member_id,
            "drafter_member_id": manifest.drafter_member_id,
            "target_model_id": manifest.target_model_id,
            "drafter_model_id": manifest.drafter_model_id,
            "target_revision": manifest.target_revision,
            "drafter_revision": manifest.drafter_revision,
            "interface_sha256": row.interface_sha256,
            "source_repository": manifest.source_repository,
            "source_commit": manifest.source_commit,
            "model_selector_sha256": model_selector_sha256,
            "selector_asset_path": selector_asset.absolute_path,
            "selector_asset_raw_sha256": selector_asset.raw_sha256,
            "selector_asset_semantic_sha256": selector_asset.semantic_sha256,
            "target_snapshot_raw_sha256": target_snapshot_raw_sha256,
            "target_snapshot_semantic_sha256": target_snapshot_semantic_sha256,
            "drafter_snapshot_raw_sha256": drafter_snapshot_raw_sha256,
            "drafter_snapshot_semantic_sha256": drafter_snapshot_semantic_sha256,
            "prepared_content_receipt_path": prepared_content_receipt.absolute_path,
            "prepared_content_receipt_raw_sha256": prepared_content_receipt.raw_sha256,
            "prepared_content_receipt_sha256": prepared_content_receipt.semantic_sha256,
            "prepared_content_authorization_sha256": (
                prepared_content_authorization_sha256
            ),
            "root_manifest_sha256": root_manifest_sha256,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            self.protocol_sha256 != EAGLE3_OFFICIAL_SELECTOR_CONTENT_PROTOCOL_SHA256
            or self.stage != "E0"
            or self.backend != "EAGLE3"
            or self.status not in {"COMPATIBLE", "N/A"}
        ):
            raise ValueError("verified official EAGLE3 selector scope is unsupported")
        for label, value in (
            ("task", self.task),
            ("reason", self.reason_code),
            ("target member", self.target_member_id),
            ("drafter member", self.drafter_member_id),
            ("target model", self.target_model_id),
            ("drafter model", self.drafter_model_id),
            ("source repository", self.source_repository),
        ):
            _require_text(f"verified official EAGLE3 {label}", value)
        _require_revision("verified official EAGLE3 target", self.target_revision)
        _require_revision("verified official EAGLE3 drafter", self.drafter_revision)
        _require_revision("verified official EAGLE3 source", self.source_commit)
        for label, value in (
            ("protocol", self.protocol_sha256),
            ("interface", self.interface_sha256),
            ("model selector", self.model_selector_sha256),
            ("selector raw", self.selector_asset_raw_sha256),
            ("selector semantic", self.selector_asset_semantic_sha256),
            ("target snapshot raw", self.target_snapshot_raw_sha256),
            ("target snapshot semantic", self.target_snapshot_semantic_sha256),
            ("drafter snapshot raw", self.drafter_snapshot_raw_sha256),
            ("drafter snapshot semantic", self.drafter_snapshot_semantic_sha256),
            ("prepared receipt raw", self.prepared_content_receipt_raw_sha256),
            ("prepared receipt", self.prepared_content_receipt_sha256),
            ("prepared authorization", self.prepared_content_authorization_sha256),
            ("release root", self.root_manifest_sha256),
        ):
            _require_sha256(f"verified official EAGLE3 {label}", value)
        for label, value in (
            ("selector asset", self.selector_asset_path),
            ("prepared receipt", self.prepared_content_receipt_path),
        ):
            path = Path(value)
            if not path.is_absolute() or path.resolve(strict=False) != path:
                raise ValueError(f"verified official EAGLE3 {label} path is invalid")

    @property
    def sha256(self) -> str:
        return _hash_body(self.__dict__)


def resolve_eagle3_official_selector_content_authority(
    prepared_content_receipt: CanonicalJsonProofBinding,
    *,
    expected_prepared_content_authorization_sha256: str,
    expected_root_manifest_sha256: str,
    expected_target_member_id: str,
    expected_drafter_member_id: str,
    expected_task: str,
    now_ns: int,
) -> VerifiedEagle3OfficialSelectorContentAuthority:
    """Deep-open one official selector row from root-authorized E0 content."""

    from lightcone_spec.locking.prepared_models import PreparedModelSnapshotContent
    from lightcone_spec.runtime.content_authorization import (
        ContentVerificationReceipt,
        VerifiedPreparedModelContentRelease,
    )

    if type(prepared_content_receipt) is not CanonicalJsonProofBinding:
        raise TypeError("EAGLE3 selector requires a path-bound content receipt")
    _require_sha256(
        "expected EAGLE3 prepared authorization",
        expected_prepared_content_authorization_sha256,
    )
    _require_sha256("expected EAGLE3 release root", expected_root_manifest_sha256)
    _require_text("expected EAGLE3 target member", expected_target_member_id)
    _require_text("expected EAGLE3 drafter member", expected_drafter_member_id)
    _require_text("expected EAGLE3 task", expected_task)
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("EAGLE3 selector replay time is invalid")

    receipt = ContentVerificationReceipt.from_dict(prepared_content_receipt.reopen())
    if receipt.sha256 != prepared_content_receipt.semantic_sha256:
        raise ValueError("EAGLE3 content receipt semantic identity differs")
    verified_rows = receipt.revalidate(current_ns=now_ns)
    prepared_rows = tuple(
        row for row in verified_rows if type(row) is VerifiedPreparedModelContentRelease
    )
    if len(prepared_rows) != 1 or len(verified_rows) != 1:
        raise ValueError("EAGLE3 selector receipt must contain one prepared authority")
    prepared = prepared_rows[0]
    if (
        prepared.authorization_sha256 != expected_prepared_content_authorization_sha256
        or prepared.authorization.root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("EAGLE3 selector prepared authority differs from ProtocolLock")
    stage_members = prepared.require_stage("E0")
    target_rows = tuple(
        row for row in stage_members if row.member_id == expected_target_member_id
    )
    drafter_rows = tuple(
        row for row in stage_members if row.member_id == expected_drafter_member_id
    )
    if (
        len(target_rows) != 1
        or len(drafter_rows) != 1
        or target_rows[0].role != "target"
        or drafter_rows[0].role != "drafter"
        or target_rows[0].backend != "EAGLE3"
        or drafter_rows[0].backend != "EAGLE3"
    ):
        raise ValueError("EAGLE3 selector members differ from the E0 model pair")
    target, drafter = target_rows[0], drafter_rows[0]

    def snapshot_binding(member: Any) -> CanonicalJsonProofBinding:
        raw_sha256 = member.snapshot_manifest_raw_sha256
        semantic_sha256 = member.snapshot_manifest_semantic_sha256
        matches = tuple(
            artifact
            for artifact in receipt.content_artifacts
            if artifact.raw_sha256 == raw_sha256
            and artifact.semantic_sha256 == semantic_sha256
        )
        if len(matches) != 1:
            raise ValueError("EAGLE3 prepared snapshot content is not exact")
        artifact = matches[0]
        return CanonicalJsonProofBinding(
            absolute_path=artifact.path,
            raw_sha256=artifact.raw_sha256,
            semantic_sha256=artifact.semantic_sha256,
            size=artifact.size,
        )

    target_snapshot_binding = snapshot_binding(target)
    drafter_snapshot_binding = snapshot_binding(drafter)
    target_snapshot = PreparedModelSnapshotContent.from_dict(
        target_snapshot_binding.reopen()
    )
    drafter_snapshot = PreparedModelSnapshotContent.from_dict(
        drafter_snapshot_binding.reopen()
    )
    if (
        target_snapshot.model_id != target.model_id
        or target_snapshot.revision != target.revision
        or drafter_snapshot.model_id != drafter.model_id
        or drafter_snapshot.revision != drafter.revision
    ):
        raise ValueError("EAGLE3 prepared snapshot identity differs from members")
    selector_files = tuple(
        row
        for row in drafter_snapshot.critical_files
        if row.relative_path == EAGLE3_OFFICIAL_SELECTOR_RELATIVE_PATH
    )
    if len(selector_files) != 1:
        raise BackendNotApplicable("eagle3_official_selector_manifest_unavailable")
    selector_file = selector_files[0]
    selector_artifact_id = f"eagle3_official_selector:{drafter.member_id}"
    selector_artifacts = tuple(
        artifact
        for artifact in receipt.content_artifacts
        if artifact.artifact_id == selector_artifact_id
        and artifact.raw_sha256 == selector_file.raw_sha256
        and artifact.size == selector_file.size
    )
    if len(selector_artifacts) != 1:
        raise ValueError("EAGLE3 selector asset is not path-bound exactly once")
    selector_artifact = selector_artifacts[0]
    expected_selector_path = (
        Path(drafter_snapshot.root) / EAGLE3_OFFICIAL_SELECTOR_RELATIVE_PATH
    )
    if expected_selector_path.resolve(
        strict=False
    ) != expected_selector_path or selector_artifact.path != str(
        expected_selector_path
    ):
        raise ValueError("EAGLE3 selector asset is outside the drafter snapshot")
    selector_binding = CanonicalJsonProofBinding(
        absolute_path=selector_artifact.path,
        raw_sha256=selector_artifact.raw_sha256,
        semantic_sha256=selector_artifact.semantic_sha256,
        size=selector_artifact.size,
    )
    manifest = Eagle3OfficialSelectorManifest.from_dict(selector_binding.reopen())
    if (
        manifest.sha256 != selector_binding.semantic_sha256
        or manifest.target_member_id != target.member_id
        or manifest.drafter_member_id != drafter.member_id
        or manifest.target_model_id != target.model_id
        or manifest.drafter_model_id != drafter.model_id
        or manifest.target_revision != target.revision
        or manifest.drafter_revision != drafter.revision
    ):
        raise ValueError("EAGLE3 selector manifest differs from prepared members")
    matching_rows = tuple(row for row in manifest.rows if row.task == expected_task)
    if len(matching_rows) != 1:
        raise ValueError("EAGLE3 selector manifest does not cover the E0 task exactly")
    return VerifiedEagle3OfficialSelectorContentAuthority(
        manifest=manifest,
        row=matching_rows[0],
        selector_asset=selector_binding,
        target_snapshot_raw_sha256=target_snapshot_binding.raw_sha256,
        target_snapshot_semantic_sha256=target_snapshot_binding.semantic_sha256,
        drafter_snapshot_raw_sha256=drafter_snapshot_binding.raw_sha256,
        drafter_snapshot_semantic_sha256=drafter_snapshot_binding.semantic_sha256,
        prepared_content_receipt=prepared_content_receipt,
        prepared_content_authorization_sha256=prepared.authorization_sha256,
        root_manifest_sha256=prepared.authorization.root_manifest_sha256,
        _verification_tag=_VERIFIED_EAGLE3_OFFICIAL_SELECTOR_CONTENT_SENTINEL,
    )


def verify_eagle3_compatibility_authority(
    authority: Eagle3CompatibilityAuthority,
    *,
    trusted_public_keys: Mapping[str, bytes],
) -> None:
    if type(authority) is not Eagle3CompatibilityAuthority:
        raise TypeError("EAGLE3 requires an exact signed compatibility authority")
    authority.__post_init__()
    public_key = trusted_public_keys.get(authority.signer_key_id)
    if type(public_key) is not bytes or len(public_key) != 32:
        raise BackendNotApplicable("eagle3_trusted_compatibility_signer_unavailable")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(authority.signature_hex),
            authority.message,
        )
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("EAGLE3 compatibility signature is invalid") from exc


_VERIFIED_EAGLE3_COMPATIBILITY_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedEagle3CompatibilityAuthority:
    """Unforgeable release-root decision for one exact EAGLE3 interface."""

    authority_sha256: str
    status: Literal["COMPATIBLE", "N/A"]
    task: str
    target_member_id: str
    drafter_member_id: str
    target_revision: str
    drafter_revision: str
    interface_sha256: str
    source_commit: str
    model_selector_sha256: str
    official_selector_content_authority_sha256: str
    prepared_content_receipt_sha256: str
    trusted_policy_sha256: str
    control_envelope_sha256: str
    challenge_reservation_sha256: str

    def __init__(
        self,
        *,
        authority: Eagle3CompatibilityAuthority,
        official_selector_content: VerifiedEagle3OfficialSelectorContentAuthority,
        trusted_policy_sha256: str,
        control_envelope_sha256: str,
        challenge_reservation_sha256: str,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_EAGLE3_COMPATIBILITY_SENTINEL:
            raise TypeError(
                "verified EAGLE3 compatibility can only come from release verification"
            )
        if type(official_selector_content) is not (
            VerifiedEagle3OfficialSelectorContentAuthority
        ):
            raise TypeError(
                "verified EAGLE3 compatibility requires official selector content"
            )
        for name, value in (
            ("authority_sha256", authority.sha256),
            ("status", authority.status),
            ("task", official_selector_content.task),
            ("target_member_id", official_selector_content.target_member_id),
            ("drafter_member_id", official_selector_content.drafter_member_id),
            ("target_revision", authority.target_revision),
            ("drafter_revision", authority.drafter_revision),
            ("interface_sha256", authority.interface_sha256),
            ("source_commit", authority.source_commit),
            ("model_selector_sha256", authority.model_selector_sha256),
            (
                "official_selector_content_authority_sha256",
                official_selector_content.sha256,
            ),
            (
                "prepared_content_receipt_sha256",
                official_selector_content.prepared_content_receipt_sha256,
            ),
            ("trusted_policy_sha256", trusted_policy_sha256),
            ("control_envelope_sha256", control_envelope_sha256),
            ("challenge_reservation_sha256", challenge_reservation_sha256),
        ):
            object.__setattr__(self, name, value)
        for label, value in (
            ("authority", self.authority_sha256),
            ("interface", self.interface_sha256),
            ("model selector", self.model_selector_sha256),
            (
                "official selector content",
                self.official_selector_content_authority_sha256,
            ),
            ("prepared content receipt", self.prepared_content_receipt_sha256),
            ("trusted policy", self.trusted_policy_sha256),
            ("control envelope", self.control_envelope_sha256),
            ("challenge reservation", self.challenge_reservation_sha256),
        ):
            _require_sha256(f"verified EAGLE3 {label}", value)
        for label, value in (
            ("task", self.task),
            ("target member", self.target_member_id),
            ("drafter member", self.drafter_member_id),
        ):
            _require_text(f"verified EAGLE3 {label}", value)

    @property
    def sha256(self) -> str:
        return _hash_body(self.__dict__)


def verify_formal_eagle3_compatibility_authority(
    authority: Eagle3CompatibilityAuthority,
    *,
    verified_official_selector_content: (
        VerifiedEagle3OfficialSelectorContentAuthority | None
    ) = None,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_inventory_sha256: str,
    expected_hardware_envelope_sha256: str,
    expected_target_revision: str,
    expected_drafter_revision: str,
    expected_interface_sha256: str,
    expected_source_commit: str,
    expected_model_selector_sha256: str,
    now_ns: int,
) -> VerifiedEagle3CompatibilityAuthority:
    """Verify dynamic release control and reserve every nonce atomically."""

    if type(authority) is not Eagle3CompatibilityAuthority:
        raise TypeError("formal EAGLE3 requires an exact compatibility authority")
    authority.__post_init__()
    if type(control_attestation) is not ControlArtifactAttestation:
        raise TypeError("formal EAGLE3 requires an exact control envelope")
    if type(replay_store) is not ChallengeReplayStore:
        raise TypeError("formal EAGLE3 requires the release replay store")
    if (
        authority.target_revision != expected_target_revision
        or authority.drafter_revision != expected_drafter_revision
        or authority.interface_sha256 != expected_interface_sha256
        or authority.source_commit != expected_source_commit
        or authority.model_selector_sha256 != expected_model_selector_sha256
    ):
        raise ValueError("EAGLE3 compatibility differs from the expected model binding")
    policy = control_attestation.deployment_policy_authorization.bundle.trusted_attester_policy
    signer_rows = tuple(
        row for row in policy.trusted_attesters if row[1] == authority.signer_key_id
    )
    if len(signer_rows) != 1:
        raise BackendNotApplicable("eagle3_trusted_compatibility_signer_unavailable")
    public_key_sha256 = signer_rows[0][2]
    try:
        public_key = base64.b64decode(
            dict(policy.public_keys)[public_key_sha256],
            validate=True,
        )
    except (KeyError, ValueError) as error:
        raise BackendNotApplicable(
            "eagle3_trusted_compatibility_signer_unavailable"
        ) from error
    verify_eagle3_compatibility_authority(
        authority,
        trusted_public_keys={authority.signer_key_id: public_key},
    )
    if type(verified_official_selector_content) is not (
        VerifiedEagle3OfficialSelectorContentAuthority
    ):
        raise TypeError("formal EAGLE3 requires verified official selector content")
    if (
        verified_official_selector_content.stage != "E0"
        or verified_official_selector_content.backend != "EAGLE3"
        or authority.status != verified_official_selector_content.status
        or authority.reason_code != verified_official_selector_content.reason_code
        or authority.target_revision
        != verified_official_selector_content.target_revision
        or authority.drafter_revision
        != verified_official_selector_content.drafter_revision
        or authority.interface_sha256
        != verified_official_selector_content.interface_sha256
        or authority.source_commit != verified_official_selector_content.source_commit
        or authority.model_selector_sha256
        != verified_official_selector_content.model_selector_sha256
    ):
        raise ValueError("EAGLE3 compatibility differs from official selector content")
    subject = control_attestation.subject
    lineage_sha256 = _hash_body(
        {
            "target_revision": authority.target_revision,
            "drafter_revision": authority.drafter_revision,
            "interface_sha256": authority.interface_sha256,
            "source_commit": authority.source_commit,
            "model_selector_sha256": authority.model_selector_sha256,
            "official_selector_content_authority_sha256": (
                verified_official_selector_content.sha256
            ),
            "prepared_content_receipt_sha256": (
                verified_official_selector_content.prepared_content_receipt_sha256
            ),
        }
    )
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != authority.sha256
        or subject.protocol_sha256 != EAGLE3_COMPATIBILITY_PROTOCOL_SHA256
        or subject.registry_sha256 != authority.model_selector_sha256
        or subject.lineage_sha256 != lineage_sha256
        or control_attestation.hardware_envelope_sha256
        != expected_hardware_envelope_sha256
    ):
        raise ValueError("EAGLE3 compatibility control subject is not exact")
    verified_controls = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    verified = verified_controls[0]
    return VerifiedEagle3CompatibilityAuthority(
        authority=authority,
        official_selector_content=verified_official_selector_content,
        trusted_policy_sha256=verified.trusted_attester_policy_sha256,
        control_envelope_sha256=verified.envelope_sha256,
        challenge_reservation_sha256=control_challenge_reservation_sha256(
            verified_controls,
            reserved_ns=now_ns,
        ),
        _verification_tag=_VERIFIED_EAGLE3_COMPATIBILITY_SENTINEL,
    )


_VERIFIED_EAGLE3_E0_EXECUTION_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedEagle3E0ExecutionAuthority:
    """Verifier-owned E0 bridge from an official selector to live GPU code.

    The token is deliberately stage- and model-bound.  It cannot be used to
    turn a syntactically valid RunConfig, an operator key, or the generic
    exactness aggregate into EAGLE3 execution authority.
    """

    protocol_sha256: str
    stage: Literal["E0"]
    method: str
    compatibility_authority_sha256: str
    official_selector_content_authority_sha256: str
    prepared_content_receipt_sha256: str
    task: str
    target_member_id: str
    drafter_member_id: str
    target_revision: str
    drafter_revision: str
    interface_sha256: str
    source_commit: str
    model_selector_sha256: str
    native_gpu_proof_sha256: str
    native_gpu_receipt_sha256: str
    native_source_identity_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    gpu_uuids: tuple[str, ...]

    def __init__(
        self,
        *,
        method: str,
        compatibility: VerifiedEagle3CompatibilityAuthority,
        native_gpu_proof: object,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_EAGLE3_E0_EXECUTION_SENTINEL:
            raise TypeError("verified EAGLE3 E0 execution authority is verifier-owned")
        from lightcone_spec.runtime.readiness import VerifiedNativeRuntimeGpuProof

        if type(compatibility) is not VerifiedEagle3CompatibilityAuthority:
            raise TypeError("EAGLE3 E0 requires verified compatibility authority")
        if type(native_gpu_proof) is not VerifiedNativeRuntimeGpuProof:
            raise TypeError("EAGLE3 E0 requires a suite-specific native GPU proof")
        rows = {
            "protocol_sha256": EAGLE3_E0_EXECUTION_PROTOCOL_SHA256,
            "stage": "E0",
            "method": method,
            "compatibility_authority_sha256": compatibility.authority_sha256,
            "official_selector_content_authority_sha256": (
                compatibility.official_selector_content_authority_sha256
            ),
            "prepared_content_receipt_sha256": (
                compatibility.prepared_content_receipt_sha256
            ),
            "task": compatibility.task,
            "target_member_id": compatibility.target_member_id,
            "drafter_member_id": compatibility.drafter_member_id,
            "target_revision": compatibility.target_revision,
            "drafter_revision": compatibility.drafter_revision,
            "interface_sha256": compatibility.interface_sha256,
            "source_commit": compatibility.source_commit,
            "model_selector_sha256": compatibility.model_selector_sha256,
            "native_gpu_proof_sha256": native_gpu_proof.sha256,
            "native_gpu_receipt_sha256": native_gpu_proof.receipt_sha256,
            "native_source_identity_sha256": (native_gpu_proof.source_identity_sha256),
            "inventory_sha256": native_gpu_proof.inventory_sha256,
            "hardware_envelope_sha256": native_gpu_proof.hardware_envelope_sha256,
            "gpu_uuids": native_gpu_proof.gpu_uuids,
        }
        for name, value in rows.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            self.protocol_sha256 != EAGLE3_E0_EXECUTION_PROTOCOL_SHA256
            or self.stage != "E0"
            or self.method not in _EAGLE3_E0_RUNTIME_METHODS
        ):
            raise ValueError("EAGLE3 E0 execution authority scope is unsupported")
        _require_revision("EAGLE3 E0 target revision", self.target_revision)
        _require_revision("EAGLE3 E0 drafter revision", self.drafter_revision)
        _require_revision("EAGLE3 E0 source commit", self.source_commit)
        for label, value in (
            ("protocol", self.protocol_sha256),
            ("compatibility", self.compatibility_authority_sha256),
            (
                "official selector content",
                self.official_selector_content_authority_sha256,
            ),
            ("prepared content receipt", self.prepared_content_receipt_sha256),
            ("interface", self.interface_sha256),
            ("model selector", self.model_selector_sha256),
            ("native GPU proof", self.native_gpu_proof_sha256),
            ("native GPU receipt", self.native_gpu_receipt_sha256),
            ("native source identity", self.native_source_identity_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
        ):
            _require_sha256(f"EAGLE3 E0 {label}", value)
        for label, value in (
            ("task", self.task),
            ("target member", self.target_member_id),
            ("drafter member", self.drafter_member_id),
        ):
            _require_text(f"EAGLE3 E0 {label}", value)
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 1
            or len(set(self.gpu_uuids)) != 1
            or not self.gpu_uuids[0].startswith("GPU-")
        ):
            raise ValueError("EAGLE3 E0 requires one exact GPU UUID")

    @property
    def sha256(self) -> str:
        return _hash_body(self.__dict__)


def bind_eagle3_e0_execution_authority(
    *,
    method: str,
    verified_compatibility_authority: VerifiedEagle3CompatibilityAuthority,
    verified_native_gpu_proof: object,
    expected_target_revision: str,
    expected_drafter_revision: str,
    expected_task: str,
    expected_target_member_id: str,
    expected_drafter_member_id: str,
    expected_source_identity_sha256: str,
    expected_inventory_sha256: str,
    expected_gpu_uuids: tuple[str, ...],
) -> VerifiedEagle3E0ExecutionAuthority:
    """Join the independent compatibility and live qualification proofs."""

    from lightcone_spec.runtime.readiness import VerifiedNativeRuntimeGpuProof

    if type(verified_compatibility_authority) is not (
        VerifiedEagle3CompatibilityAuthority
    ):
        raise TypeError("EAGLE3 E0 compatibility token has the wrong type")
    if type(verified_native_gpu_proof) is not VerifiedNativeRuntimeGpuProof:
        raise TypeError("EAGLE3 E0 native qualification token has the wrong type")
    if verified_compatibility_authority.status != "COMPATIBLE":
        raise BackendNotApplicable("eagle3_official_selector_not_compatible")
    if method not in _EAGLE3_E0_RUNTIME_METHODS:
        raise BackendNotApplicable("eagle3_execution_outside_e0_role_set")
    for label, value in (
        ("task", expected_task),
        ("target member", expected_target_member_id),
        ("drafter member", expected_drafter_member_id),
    ):
        _require_text(f"expected EAGLE3 E0 {label}", value)
    if (
        verified_compatibility_authority.target_revision != expected_target_revision
        or verified_compatibility_authority.drafter_revision
        != expected_drafter_revision
        or verified_compatibility_authority.task != expected_task
        or verified_compatibility_authority.target_member_id
        != expected_target_member_id
        or verified_compatibility_authority.drafter_member_id
        != expected_drafter_member_id
        or verified_native_gpu_proof.suite_id != "eagle3_tp1"
        or verified_native_gpu_proof.topology_mode != "tp1_dp1"
        or verified_native_gpu_proof.backend_capabilities
        != ("eagle3", "graph_hot_path", "native_itl")
        or verified_native_gpu_proof.source_identity_sha256
        != expected_source_identity_sha256
        or verified_native_gpu_proof.inventory_sha256 != expected_inventory_sha256
        or verified_native_gpu_proof.gpu_uuids != expected_gpu_uuids
    ):
        raise BackendNotApplicable("eagle3_e0_exact_gpu_authority_unavailable")
    return VerifiedEagle3E0ExecutionAuthority(
        method=method,
        compatibility=verified_compatibility_authority,
        native_gpu_proof=verified_native_gpu_proof,
        _verification_tag=_VERIFIED_EAGLE3_E0_EXECUTION_SENTINEL,
    )


def require_eagle3_e0_execution_authority(
    *,
    claimed_execution_authority_sha256: str | None,
    claimed_compatibility_authority_sha256: str | None,
    claimed_model_selector_sha256: str | None,
    claimed_native_gpu_proof_sha256: str | None,
    verified_execution_authority: VerifiedEagle3E0ExecutionAuthority | None,
    expected_method: str,
    expected_target_revision: str,
    expected_drafter_revision: str,
    expected_source_identity_sha256: str | None,
    expected_inventory_sha256: str,
    expected_gpu_uuids: tuple[str, ...],
) -> VerifiedEagle3E0ExecutionAuthority:
    """Fail before allocation unless every E0 bridge identity is exact."""

    claims = (
        claimed_execution_authority_sha256,
        claimed_compatibility_authority_sha256,
        claimed_model_selector_sha256,
        claimed_native_gpu_proof_sha256,
        expected_source_identity_sha256,
    )
    if any(type(value) is not str for value in claims):
        raise BackendNotApplicable("eagle3_e0_execution_authority_unavailable")
    for label, value in zip(
        (
            "execution authority",
            "compatibility authority",
            "model selector",
            "native GPU proof",
            "native source identity",
        ),
        claims,
        strict=True,
    ):
        assert isinstance(value, str)
        _require_sha256(f"EAGLE3 E0 claimed {label}", value)
    if (
        type(verified_execution_authority) is not VerifiedEagle3E0ExecutionAuthority
        or verified_execution_authority.stage != "E0"
        or verified_execution_authority.method != expected_method
        or verified_execution_authority.target_revision != expected_target_revision
        or verified_execution_authority.drafter_revision != expected_drafter_revision
        or verified_execution_authority.sha256 != claimed_execution_authority_sha256
        or verified_execution_authority.compatibility_authority_sha256
        != claimed_compatibility_authority_sha256
        or verified_execution_authority.model_selector_sha256
        != claimed_model_selector_sha256
        or verified_execution_authority.native_gpu_receipt_sha256
        != claimed_native_gpu_proof_sha256
        or verified_execution_authority.native_source_identity_sha256
        != expected_source_identity_sha256
        or verified_execution_authority.inventory_sha256 != expected_inventory_sha256
        or verified_execution_authority.gpu_uuids != expected_gpu_uuids
    ):
        raise BackendNotApplicable("eagle3_e0_execution_authority_unavailable")
    return verified_execution_authority


def _payload_value_identity(value: object) -> dict[str, object]:
    if isinstance(value, Tensor):
        return {
            "type": "tensor",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if value is None or type(value) in {str, int, bool}:
        return {"type": type(value).__qualname__, "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("backend payload scalar must be finite")
        return {"type": "float", "value": value}
    digest = getattr(value, "sha256", None)
    if type(digest) is str:
        _require_sha256("backend payload authority", digest)
        return {"type": type(value).__qualname__, "sha256": digest}
    raise TypeError(
        f"unsupported backend payload identity value {type(value).__qualname__}"
    )


@dataclass(frozen=True)
class BackendPayload:
    """Backend-owned values plus an immutable schema identity."""

    schema: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.schema or self.schema.strip() != self.schema:
            raise ValueError("backend payload schema must be non-empty")
        if type(self.values) is not dict or not self.values:
            raise ValueError("backend payload must not be empty")
        if any(type(name) is not str or not name for name in self.values):
            raise ValueError("backend payload field names must be non-empty strings")
        frozen = MappingProxyType(dict(self.values))
        for value in frozen.values():
            _payload_value_identity(value)
        object.__setattr__(self, "values", frozen)

    @property
    def sha256(self) -> str:
        identity = [
            {"name": name, **_payload_value_identity(value)}
            for name, value in sorted(self.values.items())
        ]
        return _hash_body({"schema": self.schema, "identity": identity})


@dataclass(frozen=True)
class ProposalEvidence:
    """Source-bound evidence for one exact speculative proposal.

    Construction performs structural checks only. Device-side numerical
    predicates are returned as tensors so a headline path never needs
    ``Tensor.item()``, a DtoH copy, or an implicit synchronization.
    """

    backend: BackendName
    adapter_free_logits: Tensor
    proposal_logits: Tensor
    corrected_distribution: Tensor
    valid_mask: Tensor
    teacher_rows: Tensor
    predecessor_token_ids: Tensor
    predecessor_embeddings: Tensor
    confidence: Tensor | None
    request_ids: tuple[str, ...]
    cohort_sha256: str
    source_adapter_version: int
    payload: BackendPayload

    def __post_init__(self) -> None:
        _require_sha256("cohort_sha256", self.cohort_sha256)
        if self.source_adapter_version < 0:
            raise ValueError("source_adapter_version must be non-negative")
        if not self.request_ids or len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("proposal request IDs must be non-empty and unique")
        devices = {tensor.device for tensor in self.common_tensors}
        if len(devices) != 1:
            raise ValueError("proposal evidence tensors must share one device")
        if self.adapter_free_logits.shape != self.proposal_logits.shape:
            raise ValueError("adapter-free and deployed proposal logits must align")
        if self.corrected_distribution.shape != self.proposal_logits.shape:
            raise ValueError("corrected proposal distribution must align with logits")
        if self.teacher_rows.shape != self.proposal_logits.shape:
            raise ValueError("teacher rows must align with proposal logits")
        if self.valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be boolean")
        if self.valid_mask.shape != self.proposal_logits.shape[:-1]:
            raise ValueError("valid_mask must cover every non-vocabulary row")
        if self.predecessor_token_ids.shape != self.valid_mask.shape:
            raise ValueError("predecessor-token identity must cover proposal rows")
        if self.predecessor_embeddings.shape[:-1] != self.valid_mask.shape:
            raise ValueError("predecessor embeddings must cover proposal rows")
        if (
            self.confidence is not None
            and self.confidence.shape != self.valid_mask.shape
        ):
            raise ValueError("confidence must cover proposal rows")
        if self.proposal_logits.shape[0] != len(self.request_ids):
            raise ValueError("request identity count must match the proposal batch")

    @property
    def common_tensors(self) -> tuple[Tensor, ...]:
        values = (
            self.adapter_free_logits,
            self.proposal_logits,
            self.corrected_distribution,
            self.valid_mask,
            self.teacher_rows,
            self.predecessor_token_ids,
            self.predecessor_embeddings,
        )
        return values if self.confidence is None else (*values, self.confidence)

    def numerical_predicate(self) -> Tensor:
        """Return a scalar device predicate without reading it on the host."""
        floating = tuple(
            tensor for tensor in self.common_tensors if tensor.is_floating_point()
        )
        finite = torch.stack(
            tuple(torch.isfinite(tensor).all() for tensor in floating)
        ).all()
        nonnegative = (self.corrected_distribution >= 0).all()
        row_sums = self.corrected_distribution.sum(dim=-1)
        normalised = torch.isclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-5,
            atol=1e-6,
        ).all()
        return finite & nonnegative & normalised

    @property
    def identity_sha256(self) -> str:
        return _hash_body(
            {
                "backend": self.backend,
                "requests": self.request_ids,
                "cohort": self.cohort_sha256,
                "source_adapter_version": self.source_adapter_version,
                "payload": self.payload.sha256,
                "shapes": [tuple(tensor.shape) for tensor in self.common_tensors],
                "dtypes": [str(tensor.dtype) for tensor in self.common_tensors],
            }
        )


@dataclass(frozen=True)
class Reconstruction:
    proposal_logits: Tensor
    corrected_distribution: Tensor
    confidence: Tensor | None

    def numerical_predicate(self) -> Tensor:
        tensors = (self.proposal_logits, self.corrected_distribution)
        if self.confidence is not None:
            tensors = (*tensors, self.confidence)
        return torch.stack(
            tuple(torch.isfinite(tensor).all() for tensor in tensors)
        ).all()


@runtime_checkable
class BackendContract(Protocol):
    """One registered backend validator and differentiable reconstructor."""

    name: BackendName

    def validate_payload(self, evidence: ProposalEvidence) -> None: ...

    def reconstruct(
        self,
        evidence: ProposalEvidence,
        *,
        adapter_delta: Mapping[str, Tensor],
        adapter_already_applied: bool,
    ) -> Reconstruction: ...


class BackendRegistry:
    """Process-local backend contracts with duplicate registration rejected."""

    def __init__(self, contracts: Sequence[BackendContract] = ()) -> None:
        self._contracts: dict[BackendName, BackendContract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: BackendContract) -> None:
        if contract.name in self._contracts:
            raise ValueError(f"duplicate backend contract {contract.name}")
        self._contracts[contract.name] = contract

    def validate(self, evidence: ProposalEvidence) -> None:
        try:
            contract = self._contracts[evidence.backend]
        except KeyError as exc:
            raise ValueError(f"unregistered backend {evidence.backend}") from exc
        contract.validate_payload(evidence)

    def reconstruct(
        self,
        evidence: ProposalEvidence,
        *,
        adapter_delta: Mapping[str, Tensor],
        adapter_already_applied: bool = False,
    ) -> Reconstruction:
        if adapter_already_applied and adapter_delta:
            raise ValueError("refusing to double-count an already-applied adapter")
        self.validate(evidence)
        return self._contracts[evidence.backend].reconstruct(
            evidence,
            adapter_delta=adapter_delta,
            adapter_already_applied=adapter_already_applied,
        )


@dataclass(frozen=True)
class FunctionalBackendContract:
    """Thin registered adapter around backend-owned validation/reconstruction."""

    name: BackendName
    payload_schema: str
    required_payload_fields: frozenset[str]
    reconstruct_fn: Callable[
        [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
    ]

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        if (
            evidence.backend != self.name
            or evidence.payload.schema != self.payload_schema
        ):
            raise ValueError("proposal evidence is bound to a different backend schema")
        missing = self.required_payload_fields - evidence.payload.values.keys()
        if missing:
            raise ValueError(f"backend payload is incomplete: {sorted(missing)}")
        unknown = evidence.payload.values.keys() - self.required_payload_fields
        if unknown:
            raise ValueError(f"backend payload has unknown fields: {sorted(unknown)}")

    def reconstruct(
        self,
        evidence: ProposalEvidence,
        *,
        adapter_delta: Mapping[str, Tensor],
        adapter_already_applied: bool,
    ) -> Reconstruction:
        # Contracts are public Python objects; callers must not be able to
        # bypass validation by invoking the reconstruction hook directly.
        self.validate_payload(evidence)
        result = self.reconstruct_fn(
            evidence,
            adapter_delta,
            adapter_already_applied,
        )
        if result.proposal_logits.shape != evidence.proposal_logits.shape:
            raise ValueError("backend reconstruction changed proposal-logit shape")
        if result.corrected_distribution.shape != evidence.corrected_distribution.shape:
            raise ValueError(
                "backend reconstruction changed proposal-distribution shape"
            )
        if (result.confidence is None) != (evidence.confidence is None):
            raise ValueError("backend reconstruction changed confidence availability")
        if (
            result.confidence is not None
            and result.confidence.shape != evidence.valid_mask.shape
        ):
            raise ValueError("backend reconstruction changed confidence shape")
        return result


class DFlashBackendContract(FunctionalBackendContract):
    """DFlash contract bound to the deployed differentiable-canvas state."""

    def __init__(
        self,
        reconstruct_fn: Callable[
            [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
        ],
    ) -> None:
        super().__init__(
            name="DFLASH",
            payload_schema="dflash-native-v1",
            required_payload_fields=frozenset({"canvas_state", "proposal_correction"}),
            reconstruct_fn=reconstruct_fn,
        )

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        super().validate_payload(evidence)
        if evidence.payload.values["proposal_correction"] != "frozen_at_sampling":
            raise ValueError("DFlash proposal correction must remain sampling-bound")


class EagleBackendContract(FunctionalBackendContract):
    """EAGLE-family topk=1 tree-state reconstruction contract."""

    def __init__(
        self,
        name: Literal["EAGLE", "EAGLE3"],
        reconstruct_fn: Callable[
            [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
        ],
        *,
        verified_compatibility_authority: (
            VerifiedEagle3CompatibilityAuthority | None
        ) = None,
        diagnostic_trusted_public_keys: Mapping[str, bytes] | None = None,
    ) -> None:
        schema = "eagle-native-v1" if name == "EAGLE" else "eagle3-native-v2"
        fields = {"tree_state", "topk", "proposal_correction"}
        if name == "EAGLE3":
            fields.update(
                {
                    "compatibility_authority",
                    "target_revision",
                    "drafter_revision",
                }
            )
        elif diagnostic_trusted_public_keys:
            raise ValueError("EAGLE cannot carry EAGLE3 compatibility trust")
        elif verified_compatibility_authority is not None:
            raise ValueError("EAGLE cannot carry EAGLE3 formal compatibility")
        if (
            verified_compatibility_authority is not None
            and diagnostic_trusted_public_keys is not None
        ):
            raise ValueError("EAGLE3 formal and diagnostic trust cannot be mixed")
        if (
            verified_compatibility_authority is not None
            and type(verified_compatibility_authority)
            is not VerifiedEagle3CompatibilityAuthority
        ):
            raise TypeError("EAGLE3 requires an exact verified compatibility token")
        super().__init__(
            name=name,
            payload_schema=schema,
            required_payload_fields=frozenset(fields),
            reconstruct_fn=reconstruct_fn,
        )
        trusted = (
            {}
            if diagnostic_trusted_public_keys is None
            else dict(diagnostic_trusted_public_keys)
        )
        object.__setattr__(self, "_diagnostic_trusted_public_keys", trusted)
        object.__setattr__(
            self,
            "_verified_compatibility_authority",
            verified_compatibility_authority,
        )

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        super().validate_payload(evidence)
        values = evidence.payload.values
        if values["topk"] != 1:
            raise ValueError("adapted EAGLE reconstruction requires topk=1")
        if values["proposal_correction"] != "frozen_at_sampling":
            raise ValueError("EAGLE proposal correction must remain sampling-bound")
        if self.name == "EAGLE3":
            authority = values["compatibility_authority"]
            verified = self._verified_compatibility_authority
            if verified is None:
                if self._diagnostic_trusted_public_keys:
                    verify_eagle3_compatibility_authority(
                        authority,
                        trusted_public_keys=self._diagnostic_trusted_public_keys,
                    )
                raise BackendNotApplicable(
                    "eagle3_formal_compatibility_authority_unavailable"
                )
            if (
                type(authority) is not Eagle3CompatibilityAuthority
                or verified.authority_sha256 != authority.sha256
                or verified.target_revision != authority.target_revision
                or verified.drafter_revision != authority.drafter_revision
                or verified.interface_sha256 != authority.interface_sha256
                or verified.source_commit != authority.source_commit
                or verified.model_selector_sha256 != authority.model_selector_sha256
            ):
                raise ValueError(
                    "EAGLE3 payload differs from verified compatibility authority"
                )
            if verified.status == "N/A":
                raise BackendNotApplicable(authority.reason_code)
            if (
                values["target_revision"] != authority.target_revision
                or values["drafter_revision"] != authority.drafter_revision
            ):
                raise ValueError(
                    "EAGLE3 payload revisions differ from signed compatibility"
                )


class NextNBackendContract(FunctionalBackendContract):
    """Native NEXTN interface with an immutable upstream interface digest."""

    def __init__(
        self,
        reconstruct_fn: Callable[
            [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
        ],
        *,
        verified_tp2_authority: VerifiedNextNTp2Authority | None = None,
    ) -> None:
        super().__init__(
            name="NEXTN",
            payload_schema="nextn-native-v2",
            required_payload_fields=frozenset(
                {
                    "mtp_hidden_state",
                    "mtp_teacher_rows",
                    "mtp_valid_mask",
                    "interface_sha256",
                    "source_adapter_version",
                    "target_revision",
                    "drafter_revision",
                    "topology_mode",
                    "tp2_model_authority",
                    "proposal_correction",
                }
            ),
            reconstruct_fn=reconstruct_fn,
        )
        if verified_tp2_authority is not None and (
            type(verified_tp2_authority) is not VerifiedNextNTp2Authority
        ):
            raise TypeError("NEXTN verified TP2 authority has the wrong type")
        object.__setattr__(self, "_verified_tp2_authority", verified_tp2_authority)

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        super().validate_payload(evidence)
        values = evidence.payload.values
        _require_sha256("NEXTN interface_sha256", values["interface_sha256"])
        _require_revision("NEXTN target revision", values["target_revision"])
        _require_revision("NEXTN drafter revision", values["drafter_revision"])
        hidden = values["mtp_hidden_state"]
        if not isinstance(hidden, Tensor):
            raise TypeError("NEXTN mtp_hidden_state must be a tensor")
        if hidden.device != evidence.proposal_logits.device:
            raise ValueError("NEXTN hidden state must remain device-resident")
        if hidden.shape[:-1] != evidence.valid_mask.shape:
            raise ValueError("NEXTN hidden state must cover proposal rows")
        if values["mtp_teacher_rows"] is not evidence.teacher_rows:
            raise ValueError("NEXTN teacher rows must be the exact MTP source tensor")
        if values["mtp_valid_mask"] is not evidence.valid_mask:
            raise ValueError("NEXTN valid mask must be the exact MTP source mask")
        if values["source_adapter_version"] != evidence.source_adapter_version:
            raise ValueError("NEXTN payload source version differs from evidence")
        if values["topology_mode"] == "tp1_dp1":
            if values["tp2_model_authority"] is not None:
                raise ValueError("TP1 NEXTN cannot carry a TP2 model authority")
            if self._verified_tp2_authority is not None:
                raise ValueError("TP1 NEXTN cannot consume a TP2 release authority")
        elif values["topology_mode"] == "tp2_dp1":
            authority = values["tp2_model_authority"]
            if type(authority) is not NextNTwoModelTp2Authority:
                raise TypeError("TP2 NEXTN requires an exact two-model authority")
            if (
                authority.interface_sha256 != values["interface_sha256"]
                or authority.target_revision != values["target_revision"]
                or authority.drafter_revision != values["drafter_revision"]
                or authority.source_adapter_version != evidence.source_adapter_version
            ):
                raise ValueError("NEXTN TP2 authority differs from proposal evidence")
            verified = self._verified_tp2_authority
            if type(verified) is not VerifiedNextNTp2Authority:
                raise BackendNotApplicable("nextn_tp2_native_gpu_authority_unavailable")
            if (
                verified.authority_sha256 != authority.sha256
                or verified.interface_sha256 != authority.interface_sha256
                or verified.target_revision != authority.target_revision
                or verified.drafter_revision != authority.drafter_revision
                or verified.target_shard_manifest_sha256
                != authority.target_shard_manifest_sha256
                or verified.drafter_shard_manifest_sha256
                != authority.drafter_shard_manifest_sha256
                or verified.topology_sha256 != authority.topology_sha256
                or verified.source_adapter_version != authority.source_adapter_version
            ):
                raise ValueError(
                    "NEXTN payload differs from verified two-model authority"
                )
        else:
            raise ValueError("NEXTN supports only tp1_dp1 or tp2_dp1")
        if values["proposal_correction"] != "frozen_at_sampling":
            raise ValueError("NEXTN proposal correction must remain sampling-bound")


class DSparkBackendContract(FunctionalBackendContract):
    """DSpark envelope requiring sampled-predecessor and native Markov evidence."""

    def __init__(
        self,
        reconstruct_fn: Callable[
            [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
        ],
    ) -> None:
        super().__init__(
            name="DSPARK",
            payload_schema="dspark-native-v2",
            required_payload_fields=frozenset(
                {
                    "markov_w1_feature",
                    "markov_w2_feature",
                    "markov_w1_source",
                    "markov_w2_source",
                    "predecessor_source",
                    "predecessor_embedding_source",
                    "confidence_head_source",
                    "selector_authority",
                    "selector_candidate_id",
                    "model_interface_sha256",
                    "parameter_inventory_sha256",
                    "native_head_mode",
                    "scheduler_mode",
                    "fixed_verification_budget",
                    "proposal_correction",
                }
            ),
            reconstruct_fn=reconstruct_fn,
        )

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        super().validate_payload(evidence)
        values = evidence.payload.values
        if (
            values["markov_w1_source"] != "inference_native"
            or values["markov_w2_source"] != "inference_native"
        ):
            raise ValueError("DSpark requires real inference Markov W1/W2 features")
        if values["predecessor_source"] != "sampled_token":
            raise ValueError("DSpark requires the actual sampled predecessor")
        if values["predecessor_embedding_source"] != (
            "embedding_lookup_of_sampled_token"
        ):
            raise ValueError(
                "DSpark predecessor embedding must come from the sampled token"
            )
        if evidence.confidence is None:
            raise ValueError("DSpark requires the native confidence head")
        if values["confidence_head_source"] != "inference_native":
            raise ValueError("DSpark confidence must come from the native head")
        authority = values["selector_authority"]
        if type(authority) is not DSparkSelectorAuthority:
            raise TypeError("DSpark requires an exact 56-candidate selector authority")
        authority.__post_init__()
        if (
            values["model_interface_sha256"] != authority.model_interface_sha256
            or values["parameter_inventory_sha256"]
            != authority.parameter_inventory_sha256
        ):
            raise ValueError(
                "DSpark selector differs from the live model parameter inventory"
            )
        if values["selector_candidate_id"] not in authority.candidate_ids:
            raise ValueError("DSpark candidate is outside the 56-candidate selector")
        if values["native_head_mode"] not in {
            "frozen",
            "full_w1_w2_confidence",
        }:
            raise ValueError("DSpark native-head mode is not registered")
        if values["scheduler_mode"] not in {"fixed_budget", "native_scheduler"}:
            raise ValueError("DSpark scheduler mode is not registered")
        fixed_budget = values["fixed_verification_budget"]
        if values["scheduler_mode"] == "fixed_budget":
            if type(fixed_budget) is not int or fixed_budget < 1:
                raise ValueError(
                    "fixed-budget DSpark requires a positive verification budget"
                )
        elif fixed_budget is not None:
            raise ValueError("native-scheduler DSpark cannot carry a fixed budget")
        if values["proposal_correction"] != "frozen_at_sampling":
            raise ValueError("DSpark proposal correction must remain sampling-bound")
        for name in ("markov_w1_feature", "markov_w2_feature"):
            tensor = values[name]
            if not isinstance(tensor, Tensor):
                raise TypeError(f"DSpark {name} must be a tensor")
            if tensor.device != evidence.proposal_logits.device:
                raise ValueError("DSpark Markov features must remain device-resident")
            if tensor.shape[:-1] != evidence.valid_mask.shape:
                raise ValueError("DSpark Markov features must cover proposal rows")


def dspark_conditional_survival_target(
    teacher_distribution: Tensor,
    proposal_distribution: Tensor,
) -> Tensor:
    """Stop-gradient target ``1 - TV(target, proposal)`` for confidence."""
    if teacher_distribution.shape != proposal_distribution.shape:
        raise ValueError("target and proposal distributions must align")
    total_variation = 0.5 * (
        teacher_distribution.detach() - proposal_distribution.detach()
    ).abs().sum(dim=-1)
    return (1.0 - total_variation).clamp(0.0, 1.0).detach()


def dspark_composite_loss(
    *,
    teacher_distribution: Tensor,
    proposal_distribution: Tensor,
    confidence_logits: Tensor,
    valid_mask: Tensor,
    confidence_weight: float,
) -> Tensor:
    """Proposal cross-entropy plus a tuning-locked proper confidence loss."""
    if not 0.0 <= confidence_weight < float("inf"):
        raise ValueError("confidence loss weight must be finite and non-negative")
    if teacher_distribution.shape != proposal_distribution.shape:
        raise ValueError("target and proposal distributions must align")
    if (
        valid_mask.dtype is not torch.bool
        or valid_mask.shape != confidence_logits.shape
    ):
        raise ValueError("confidence mask and logits must align")
    if teacher_distribution.shape[:-1] != valid_mask.shape:
        raise ValueError("proposal rows and confidence mask must align")
    tiny = torch.finfo(proposal_distribution.dtype).tiny
    proposal_loss = -(
        teacher_distribution.detach() * proposal_distribution.clamp_min(tiny).log()
    ).sum(dim=-1)
    survival = dspark_conditional_survival_target(
        teacher_distribution,
        proposal_distribution,
    )
    confidence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence_logits,
        survival,
        reduction="none",
    )
    weights = valid_mask.to(proposal_loss.dtype)
    # An empty mask intentionally produces a non-finite candidate. The
    # device-side candidate predicate then discards it without a host read.
    return ((proposal_loss + confidence_weight * confidence_loss) * weights).sum() / (
        weights.sum()
    )
