"""Distributed adaptation identities and two-phase publication contracts.

The deterministic coordinator is transport independent. ``GlooPublicationTransport``
exercises the real all-rank protocol on CPU process groups; the pinned SGLang patch
owns the NCCL/CUDA transport and must provide a separately attested capability receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from .control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from .proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from .qualification_spec import (
    DISTRIBUTED_RUNTIME_GPU_PROOF_PROTOCOL_SHA256,
    DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS,
    DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
)
from .readiness import NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256

RuntimeTopologyMode = Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
DistributedControlMode = Literal[
    "single_rank",
    "tp_all_rank_two_phase",
    "dp_sticky_replica_local",
]
AdaptationCollectiveMode = Literal["none", "tp_shard_all_rank"]

_REGISTERED_RUNTIME_TOPOLOGIES: dict[tuple[int, int], RuntimeTopologyMode] = {
    (1, 1): "tp1_dp1",
    (2, 1): "tp2_dp1",
    (1, 2): "tp1_dp2",
}


def _sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _require_hash(name: str, value: str, length: int = 64) -> None:
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase {length * 4}-bit hash")


def _require_nonempty(name: str, value: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical identifier")


def _require_counter(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def registered_runtime_topology_mode(
    tensor_parallel_size: int,
    data_parallel_size: int,
    node_count: int,
) -> RuntimeTopologyMode:
    """Return the only registered single-host runtime topology identity.

    The fleet control plane may dispatch independent cells to multiple hosts,
    but one serving/adaptation gang is deliberately limited to at most two
    ranks on one host.  In particular, TP2+DP2 is not a convenient shorthand
    for a supported four-rank topology.
    """

    for name, value in (
        ("tensor_parallel_size", tensor_parallel_size),
        ("data_parallel_size", data_parallel_size),
        ("node_count", node_count),
    ):
        _require_counter(name, value, minimum=1)
    if node_count != 1:
        raise ValueError("multi-node runtime collectives are restricted to one host")
    try:
        return _REGISTERED_RUNTIME_TOPOLOGIES[
            (tensor_parallel_size, data_parallel_size)
        ]
    except KeyError as exc:
        raise ValueError(
            "runtime topology must be tp1_dp1, tp2_dp1, or tp1_dp2"
        ) from exc


def distributed_control_mode(mode: RuntimeTopologyMode) -> DistributedControlMode:
    """Return the control plane required by one registered topology."""

    return {
        "tp1_dp1": "single_rank",
        "tp2_dp1": "tp_all_rank_two_phase",
        "tp1_dp2": "dp_sticky_replica_local",
    }[mode]


def adaptation_collective_mode(
    mode: RuntimeTopologyMode,
) -> AdaptationCollectiveMode:
    """Return the adaptation collective, deliberately excluding cross-DP sync."""

    return "tp_shard_all_rank" if mode == "tp2_dp1" else "none"


class DistributedRuntimeAuthorityBlocked(RuntimeError):
    """Stable fail-closed result for an unavailable audited release capability."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class DistributedRuntimeReleaseCapability:
    """Source-owned CPU-audited identity for one host-local runtime mode.

    A digest supplied in a run configuration is only a claim.  Authority comes
    from an exact object in ``DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES``.  That
    source table pins the semantic patch before GPU smoke.  It never represents
    GPU proof; a signed ``DistributedRuntimeGpuProofReceipt`` supplies that
    dynamic authority without changing the source tree after smoke.
    """

    schema_version: Literal[1]
    topology_mode: Literal["tp2_dp1", "tp1_dp2"]
    pinned_sglang_commit: str
    patched_sglang_tree: str
    semantic_patch_sha256: str
    native_terminal_protocol_sha256: str
    control_mode: Literal[
        "tp_all_rank_two_phase",
        "dp_sticky_replica_local",
    ]
    process_group_backend: Literal["nccl", "none"]
    adaptation_collective: AdaptationCollectiveMode
    evidence_status: Literal["CPU_CONTRACT_ONLY", "GPU_VERIFIED"]
    gpu_proof_sha256: str | None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("distributed release capability schema is unsupported")
        for label, value in (
            ("pinned SGLang commit", self.pinned_sglang_commit),
            ("patched SGLang tree", self.patched_sglang_tree),
        ):
            _require_hash(label, value, length=40)
        for label, value in (
            ("semantic patch", self.semantic_patch_sha256),
            ("native terminal protocol", self.native_terminal_protocol_sha256),
        ):
            _require_hash(label, value)
        expected_control = distributed_control_mode(self.topology_mode)
        expected_collective = adaptation_collective_mode(self.topology_mode)
        expected_backend = "nccl" if self.topology_mode == "tp2_dp1" else "none"
        if self.control_mode != expected_control:
            raise ValueError("distributed release control mode differs from topology")
        if self.adaptation_collective != expected_collective:
            raise ValueError(
                "distributed release adaptation collective differs from topology"
            )
        if self.process_group_backend != expected_backend:
            raise ValueError(
                "distributed release process-group backend differs from topology"
            )
        if self.evidence_status == "GPU_VERIFIED":
            if self.gpu_proof_sha256 is None:
                raise ValueError("GPU-verified distributed release requires proof")
            _require_hash("distributed GPU proof", self.gpu_proof_sha256)
        elif self.gpu_proof_sha256 is not None:
            raise ValueError("CPU-only distributed capability cannot carry GPU proof")

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


# A caller-authored digest never mutates this table, and a source entry alone
# authorizes only diagnostic smoke, never formal execution.  Both entries bind
# semantic patch 0007 and its resulting complete SGLang tree.
DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES: Mapping[
    RuntimeTopologyMode, DistributedRuntimeReleaseCapability
] = MappingProxyType(
    {
        mode: DistributedRuntimeReleaseCapability(
            schema_version=1,
            topology_mode=mode,
            pinned_sglang_commit="3312645a307453893a00778592f105581e3d1c3d",
            patched_sglang_tree="c6571336b70cd5f0e0f609d731a65fa98fd7e0b2",
            semantic_patch_sha256=(
                "38b5ec81b9d75950558f8c72c1297bab47badf89d855b3e13dc1ad1c639f7d95"
            ),
            native_terminal_protocol_sha256=(
                "5c3113405e0646e0fa61bbd054e690d588996982e2f57ded94d77b6e0c072e02"
            ),
            control_mode=distributed_control_mode(mode),
            process_group_backend="nccl" if mode == "tp2_dp1" else "none",
            adaptation_collective=adaptation_collective_mode(mode),
            evidence_status="CPU_CONTRACT_ONLY",
            gpu_proof_sha256=None,
        )
        for mode in ("tp2_dp1", "tp1_dp2")
    }
)


def require_distributed_runtime_release_capability(
    *,
    topology_mode: RuntimeTopologyMode,
    claimed_capability_sha256: str,
) -> DistributedRuntimeReleaseCapability:
    """Resolve a CPU-audited source capability; never trust the run claim."""

    if topology_mode == "tp1_dp1":
        raise ValueError("single-rank topology does not use a distributed capability")
    _require_hash("claimed distributed release capability", claimed_capability_sha256)
    capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES.get(topology_mode)
    if (
        type(capability) is not DistributedRuntimeReleaseCapability
        or capability.topology_mode != topology_mode
        or capability.sha256 != claimed_capability_sha256
        or capability.evidence_status != "CPU_CONTRACT_ONLY"
    ):
        raise DistributedRuntimeAuthorityBlocked(
            "distributed_runtime_release_capability_unavailable",
            "the release has no CPU-audited semantic patch identity for this mode",
        )
    return capability


@dataclass(frozen=True)
class DistributedRuntimeGpuProofReceipt:
    """Unsigned remote base-smoke plus exact distributed qualification."""

    schema_version: Literal[1]
    kind: Literal["lightcone_distributed_runtime_gpu_proof"]
    topology_mode: Literal["tp2_dp1", "tp1_dp2"]
    topology_sha256: str
    runner_protocol_sha256: str
    assignment_sha256: str
    qualification_observation_sha256: str
    base_exactness_result_pointer_sha256: str
    source_capability_sha256: str
    pinned_sglang_commit: str
    patched_sglang_tree: str
    semantic_patch_sha256: str
    run_nonce_sha256: str
    qualification_authority_sha256: str
    source_identity_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, str]
    hardware_envelope_sha256: str
    junit_xml_sha256: str
    tests_collected: Literal[8]
    tests_passed: Literal[8]
    tests_failed: Literal[0]
    tests_errored: Literal[0]
    tests_skipped: Literal[0]
    qualification_junit_xml_sha256: str
    qualification_test_names: tuple[str, ...]
    qualification_tests_collected: int
    qualification_tests_passed: int
    qualification_tests_failed: Literal[0]
    qualification_tests_errored: Literal[0]
    qualification_tests_skipped: Literal[0]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_distributed_runtime_gpu_proof"
        ):
            raise ValueError("distributed GPU proof schema is unsupported")
        for label, value, length in (
            ("proof source capability", self.source_capability_sha256, 64),
            ("proof topology", self.topology_sha256, 64),
            ("proof runner protocol", self.runner_protocol_sha256, 64),
            ("proof assignment", self.assignment_sha256, 64),
            (
                "proof qualification observation",
                self.qualification_observation_sha256,
                64,
            ),
            (
                "proof base exactness result pointer",
                self.base_exactness_result_pointer_sha256,
                64,
            ),
            ("proof pinned SGLang commit", self.pinned_sglang_commit, 40),
            ("proof patched SGLang tree", self.patched_sglang_tree, 40),
            ("proof semantic patch", self.semantic_patch_sha256, 64),
            ("proof run nonce", self.run_nonce_sha256, 64),
            (
                "proof qualification authority",
                self.qualification_authority_sha256,
                64,
            ),
            ("proof source identity", self.source_identity_sha256, 64),
            ("proof inventory", self.inventory_sha256, 64),
            ("proof hardware envelope", self.hardware_envelope_sha256, 64),
            ("proof JUnit XML", self.junit_xml_sha256, 64),
            (
                "proof qualification JUnit XML",
                self.qualification_junit_xml_sha256,
                64,
            ),
        ):
            _require_hash(label, value, length=length)
        if (
            self.qualification_authority_sha256
            != NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256
        ):
            raise ValueError("distributed qualification authority differs")
        if (
            self.runner_protocol_sha256
            != (DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[self.topology_mode])
        ):
            raise ValueError("distributed GPU proof runner protocol differs")
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
        ):
            raise ValueError("distributed GPU proof requires two unique GPU UUIDs")
        for gpu_uuid in self.gpu_uuids:
            _require_nonempty("proof GPU UUID", gpu_uuid)
        counts = (
            self.tests_collected,
            self.tests_passed,
            self.tests_failed,
            self.tests_errored,
            self.tests_skipped,
        )
        if counts != (8, 8, 0, 0, 0):
            raise ValueError(
                "distributed GPU proof requires the base 8-test smoke with zero "
                "skip/failure"
            )
        expected_qualification = DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS[
            self.topology_mode
        ]
        if self.qualification_test_names != expected_qualification:
            raise ValueError(
                "distributed GPU proof lacks the exact named qualification suite"
            )
        qualification_counts = (
            self.qualification_tests_collected,
            self.qualification_tests_passed,
            self.qualification_tests_failed,
            self.qualification_tests_errored,
            self.qualification_tests_skipped,
        )
        if qualification_counts != (
            len(expected_qualification),
            len(expected_qualification),
            0,
            0,
            0,
        ):
            raise ValueError(
                "distributed GPU proof requires every named qualification with "
                "zero skip/failure"
            )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "topology_mode": self.topology_mode,
            "topology_sha256": self.topology_sha256,
            "runner_protocol_sha256": self.runner_protocol_sha256,
            "assignment_sha256": self.assignment_sha256,
            "qualification_observation_sha256": (self.qualification_observation_sha256),
            "base_exactness_result_pointer_sha256": (
                self.base_exactness_result_pointer_sha256
            ),
            "source_capability_sha256": self.source_capability_sha256,
            "pinned_sglang_commit": self.pinned_sglang_commit,
            "patched_sglang_tree": self.patched_sglang_tree,
            "semantic_patch_sha256": self.semantic_patch_sha256,
            "run_nonce_sha256": self.run_nonce_sha256,
            "qualification_authority_sha256": self.qualification_authority_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "inventory_sha256": self.inventory_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "junit_xml_sha256": self.junit_xml_sha256,
            "tests_collected": self.tests_collected,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_errored": self.tests_errored,
            "tests_skipped": self.tests_skipped,
            "qualification_junit_xml_sha256": (self.qualification_junit_xml_sha256),
            "qualification_test_names": list(self.qualification_test_names),
            "qualification_tests_collected": self.qualification_tests_collected,
            "qualification_tests_passed": self.qualification_tests_passed,
            "qualification_tests_failed": self.qualification_tests_failed,
            "qualification_tests_errored": self.qualification_tests_errored,
            "qualification_tests_skipped": self.qualification_tests_skipped,
        }

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload)

    @property
    def sha256(self) -> str:
        return self.payload_sha256

    def to_dict(self) -> dict[str, object]:
        return self.payload

    def write_unsigned(self, path: str) -> CanonicalJsonProofBinding:
        """Publish remote observations; this operation grants no trust."""

        publish_canonical_json_no_replace(path, self.to_dict())
        return CanonicalJsonProofBinding.bind(
            path,
            semantic_sha256=self.sha256,
        )

    @classmethod
    def from_dict(cls, value: object) -> DistributedRuntimeGpuProofReceipt:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "topology_mode",
            "topology_sha256",
            "runner_protocol_sha256",
            "assignment_sha256",
            "qualification_observation_sha256",
            "base_exactness_result_pointer_sha256",
            "source_capability_sha256",
            "pinned_sglang_commit",
            "patched_sglang_tree",
            "semantic_patch_sha256",
            "run_nonce_sha256",
            "qualification_authority_sha256",
            "source_identity_sha256",
            "inventory_sha256",
            "gpu_uuids",
            "hardware_envelope_sha256",
            "junit_xml_sha256",
            "tests_collected",
            "tests_passed",
            "tests_failed",
            "tests_errored",
            "tests_skipped",
            "qualification_junit_xml_sha256",
            "qualification_test_names",
            "qualification_tests_collected",
            "qualification_tests_passed",
            "qualification_tests_failed",
            "qualification_tests_errored",
            "qualification_tests_skipped",
        }:
            raise ValueError("distributed GPU proof receipt fields differ")
        row = dict(value)
        gpu_uuids = row.pop("gpu_uuids")
        test_names = row.pop("qualification_test_names")
        if type(gpu_uuids) is not list or type(test_names) is not list:
            raise TypeError("distributed GPU proof receipt arrays are malformed")
        return cls(
            gpu_uuids=tuple(gpu_uuids),
            qualification_test_names=tuple(test_names),
            **row,
        )

    @property
    def control_lineage_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_distributed_gpu_proof_control_lineage",
                "receipt_sha256": self.sha256,
                "run_nonce_sha256": self.run_nonce_sha256,
                "source_capability_sha256": self.source_capability_sha256,
                "qualification_authority_sha256": (self.qualification_authority_sha256),
                "source_identity_sha256": self.source_identity_sha256,
                "runner_protocol_sha256": self.runner_protocol_sha256,
                "assignment_sha256": self.assignment_sha256,
                "qualification_observation_sha256": (
                    self.qualification_observation_sha256
                ),
                "base_exactness_result_pointer_sha256": (
                    self.base_exactness_result_pointer_sha256
                ),
                "inventory_sha256": self.inventory_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "topology_mode": self.topology_mode,
                "topology_sha256": self.topology_sha256,
                "gpu_uuids": self.gpu_uuids,
                "base_junit_xml_sha256": self.junit_xml_sha256,
                "qualification_junit_xml_sha256": (self.qualification_junit_xml_sha256),
            }
        )


_VERIFIED_GPU_PROOF_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedDistributedRuntimeGpuProof:
    """Formal, replay-consumed authorization for one exact runtime identity."""

    receipt_sha256: str
    receipt_raw_sha256: str
    runner_protocol_sha256: str
    assignment_sha256: str
    qualification_observation_sha256: str
    base_exactness_result_pointer_sha256: str
    source_capability_sha256: str
    qualification_authority_sha256: str
    trusted_policy_sha256: str
    challenge_sha256: str
    source_identity_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    topology_mode: Literal["tp2_dp1", "tp1_dp2"]
    topology_sha256: str
    gpu_uuids: tuple[str, str]
    control_envelope_sha256: str
    challenge_reservation_sha256: str

    def __init__(
        self,
        *,
        receipt_sha256: str,
        receipt_raw_sha256: str,
        runner_protocol_sha256: str,
        assignment_sha256: str,
        qualification_observation_sha256: str,
        base_exactness_result_pointer_sha256: str,
        source_capability_sha256: str,
        qualification_authority_sha256: str,
        trusted_policy_sha256: str,
        challenge_sha256: str,
        source_identity_sha256: str,
        inventory_sha256: str,
        hardware_envelope_sha256: str,
        topology_mode: Literal["tp2_dp1", "tp1_dp2"],
        topology_sha256: str,
        gpu_uuids: tuple[str, str],
        control_envelope_sha256: str,
        challenge_reservation_sha256: str,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_GPU_PROOF_SENTINEL:
            raise TypeError(
                "verified distributed GPU proof can only come from signature verification"
            )
        for name, value in (
            ("receipt_sha256", receipt_sha256),
            ("receipt_raw_sha256", receipt_raw_sha256),
            ("runner_protocol_sha256", runner_protocol_sha256),
            ("assignment_sha256", assignment_sha256),
            (
                "qualification_observation_sha256",
                qualification_observation_sha256,
            ),
            (
                "base_exactness_result_pointer_sha256",
                base_exactness_result_pointer_sha256,
            ),
            ("source_capability_sha256", source_capability_sha256),
            ("qualification_authority_sha256", qualification_authority_sha256),
            ("trusted_policy_sha256", trusted_policy_sha256),
            ("challenge_sha256", challenge_sha256),
            ("source_identity_sha256", source_identity_sha256),
            ("inventory_sha256", inventory_sha256),
            ("hardware_envelope_sha256", hardware_envelope_sha256),
            ("topology_mode", topology_mode),
            ("topology_sha256", topology_sha256),
            ("gpu_uuids", gpu_uuids),
            ("control_envelope_sha256", control_envelope_sha256),
            ("challenge_reservation_sha256", challenge_reservation_sha256),
        ):
            object.__setattr__(self, name, value)
        for label, value in (
            ("verified proof receipt", self.receipt_sha256),
            ("verified proof receipt raw", self.receipt_raw_sha256),
            ("verified proof runner protocol", self.runner_protocol_sha256),
            ("verified proof assignment", self.assignment_sha256),
            (
                "verified proof qualification observation",
                self.qualification_observation_sha256,
            ),
            (
                "verified proof base exactness result pointer",
                self.base_exactness_result_pointer_sha256,
            ),
            ("verified proof capability", self.source_capability_sha256),
            (
                "verified proof qualification authority",
                self.qualification_authority_sha256,
            ),
            ("verified proof trust policy", self.trusted_policy_sha256),
            ("verified proof challenge", self.challenge_sha256),
            ("verified proof source identity", self.source_identity_sha256),
            ("verified proof inventory", self.inventory_sha256),
            ("verified proof hardware envelope", self.hardware_envelope_sha256),
            ("verified proof topology", self.topology_sha256),
            ("verified proof control envelope", self.control_envelope_sha256),
            (
                "verified proof challenge reservation",
                self.challenge_reservation_sha256,
            ),
        ):
            _require_hash(label, value)
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
        ):
            raise ValueError("verified distributed proof requires two GPU UUIDs")

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


def verify_distributed_runtime_gpu_proof(
    receipt_path: str,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_topology_mode: Literal["tp2_dp1", "tp1_dp2"],
    expected_topology_sha256: str,
    expected_source_capability_sha256: str,
    expected_source_identity_sha256: str,
    expected_inventory_sha256: str,
    expected_gpu_uuids: tuple[str, str],
    expected_hardware_envelope_sha256: str,
    expected_run_nonce_sha256: str,
    now_ns: int,
) -> VerifiedDistributedRuntimeGpuProof:
    """Trust-lift one unsigned remote proof and reserve every nonce atomically."""

    raw = CanonicalJsonProofBinding.bind(
        receipt_path,
    )
    receipt = DistributedRuntimeGpuProofReceipt.from_dict(raw.reopen())
    if receipt.sha256 != raw.semantic_sha256:
        raise ValueError("distributed proof semantic identity differs from content")
    binding = raw
    if type(control_attestation) is not ControlArtifactAttestation:
        raise TypeError("distributed GPU proof requires an exact control envelope")
    if type(replay_store) is not ChallengeReplayStore:
        raise TypeError("distributed GPU proof requires the release replay store")
    capability = require_distributed_runtime_release_capability(
        topology_mode=expected_topology_mode,
        claimed_capability_sha256=expected_source_capability_sha256,
    )
    if (
        receipt.topology_mode != expected_topology_mode
        or receipt.topology_sha256 != expected_topology_sha256
        or receipt.source_capability_sha256 != capability.sha256
        or receipt.pinned_sglang_commit != capability.pinned_sglang_commit
        or receipt.patched_sglang_tree != capability.patched_sglang_tree
        or receipt.semantic_patch_sha256 != capability.semantic_patch_sha256
        or receipt.run_nonce_sha256 != expected_run_nonce_sha256
        or receipt.source_identity_sha256 != expected_source_identity_sha256
        or receipt.inventory_sha256 != expected_inventory_sha256
        or receipt.gpu_uuids != expected_gpu_uuids
        or receipt.hardware_envelope_sha256 != expected_hardware_envelope_sha256
    ):
        raise ValueError("distributed GPU proof differs from the expected identity")
    subject = control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != binding.raw_sha256
        or subject.protocol_sha256 != DISTRIBUTED_RUNTIME_GPU_PROOF_PROTOCOL_SHA256
        or subject.registry_sha256 != receipt.source_identity_sha256
        or subject.lineage_sha256 != receipt.control_lineage_sha256
    ):
        raise ValueError("distributed GPU proof control subject is not exact")
    bundle = control_attestation.deployment_policy_authorization.bundle
    bundle.require_hardware_envelope(receipt.hardware_envelope_sha256)
    verified_controls = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=receipt.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=(receipt.run_nonce_sha256,),
    )
    verified_control = verified_controls[0]
    reservation_sha256 = control_challenge_reservation_sha256(
        verified_controls,
        reserved_ns=now_ns,
        additional_challenge_sha256s=(receipt.run_nonce_sha256,),
    )
    return VerifiedDistributedRuntimeGpuProof(
        receipt_sha256=receipt.sha256,
        receipt_raw_sha256=binding.raw_sha256,
        runner_protocol_sha256=receipt.runner_protocol_sha256,
        assignment_sha256=receipt.assignment_sha256,
        qualification_observation_sha256=receipt.qualification_observation_sha256,
        base_exactness_result_pointer_sha256=(
            receipt.base_exactness_result_pointer_sha256
        ),
        source_capability_sha256=capability.sha256,
        qualification_authority_sha256=receipt.qualification_authority_sha256,
        trusted_policy_sha256=verified_control.trusted_attester_policy_sha256,
        challenge_sha256=verified_control.challenge_sha256,
        source_identity_sha256=receipt.source_identity_sha256,
        inventory_sha256=receipt.inventory_sha256,
        hardware_envelope_sha256=receipt.hardware_envelope_sha256,
        topology_mode=receipt.topology_mode,
        topology_sha256=receipt.topology_sha256,
        gpu_uuids=receipt.gpu_uuids,
        control_envelope_sha256=verified_control.envelope_sha256,
        challenge_reservation_sha256=reservation_sha256,
        _verification_tag=_VERIFIED_GPU_PROOF_SENTINEL,
    )


@dataclass(frozen=True)
class DistributedRuntimeGpuProofArtifact:
    """Durable external-control lift for one unsigned distributed receipt."""

    schema_version: Literal[1]
    kind: Literal["lightcone_distributed_runtime_gpu_proof_artifact"]
    receipt: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    verified_proof_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "lightcone_distributed_runtime_gpu_proof_artifact"
        ):
            raise ValueError("distributed proof artifact schema is unsupported")
        if type(self.receipt) is not CanonicalJsonProofBinding:
            raise TypeError("distributed proof artifact requires a raw receipt")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("distributed proof artifact requires external control")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("distributed proof artifact requires replay reservation")
        self.receipt.__post_init__()
        self.control_attestation.__post_init__()
        self.replay_reservation.__post_init__()
        _require_hash("distributed verified proof", self.verified_proof_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "receipt": self.receipt.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "verified_proof_sha256": self.verified_proof_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> DistributedRuntimeGpuProofArtifact:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "receipt",
            "control_attestation",
            "replay_reservation",
            "verified_proof_sha256",
        }:
            raise ValueError("distributed proof artifact fields differ")
        row = dict(value)
        return cls(
            receipt=CanonicalJsonProofBinding.from_dict(row.pop("receipt")),
            control_attestation=ControlArtifactAttestation.from_dict(
                row.pop("control_attestation")
            ),
            replay_reservation=ChallengeReplayReservationBinding.from_dict(
                row.pop("replay_reservation")
            ),
            **row,
        )

    def revalidate(self, *, now_ns: int) -> VerifiedDistributedRuntimeGpuProof:
        self.__post_init__()
        reserved = self.replay_reservation.revalidate()
        if type(now_ns) is not int or now_ns < self.replay_reservation.reserved_ns:
            raise ValueError("distributed proof current time precedes reservation")
        verified_at_ns = self.replay_reservation.reserved_ns
        receipt = DistributedRuntimeGpuProofReceipt.from_dict(self.receipt.reopen())
        if receipt.sha256 != self.receipt.semantic_sha256:
            raise ValueError("distributed proof semantic receipt changed")
        capability = require_distributed_runtime_release_capability(
            topology_mode=receipt.topology_mode,
            claimed_capability_sha256=receipt.source_capability_sha256,
        )
        if (
            receipt.pinned_sglang_commit != capability.pinned_sglang_commit
            or receipt.patched_sglang_tree != capability.patched_sglang_tree
            or receipt.semantic_patch_sha256 != capability.semantic_patch_sha256
        ):
            raise ValueError("distributed proof source capability changed")
        subject = self.control_attestation.subject
        if (
            subject.artifact_type != "non_serving_terminal"
            or subject.artifact_sha256 != self.receipt.raw_sha256
            or subject.protocol_sha256 != DISTRIBUTED_RUNTIME_GPU_PROOF_PROTOCOL_SHA256
            or subject.registry_sha256 != receipt.source_identity_sha256
            or subject.lineage_sha256 != receipt.control_lineage_sha256
        ):
            raise ValueError("distributed proof external control differs")
        verified_control = verify_release_control_artifact_attestation(
            self.control_attestation,
            expected_inventory_sha256=receipt.inventory_sha256,
            now_ns=verified_at_ns,
            consumed_challenge_sha256s=(),
        )
        self.control_attestation.deployment_policy_authorization.bundle.require_hardware_envelope(
            receipt.hardware_envelope_sha256
        )
        expected_challenges = tuple(
            sorted(
                {
                    receipt.run_nonce_sha256,
                    verified_control.challenge_sha256,
                    verified_control.deployment_policy_challenge_sha256,
                }
            )
        )
        expected_reservation = control_challenge_reservation_sha256(
            (verified_control,),
            reserved_ns=self.replay_reservation.reserved_ns,
            additional_challenge_sha256s=(receipt.run_nonce_sha256,),
        )
        if (
            reserved != expected_challenges
            or self.replay_reservation.reservation_sha256 != expected_reservation
        ):
            raise ValueError("distributed proof replay reservation differs")
        verified = VerifiedDistributedRuntimeGpuProof(
            receipt_sha256=receipt.sha256,
            receipt_raw_sha256=self.receipt.raw_sha256,
            runner_protocol_sha256=receipt.runner_protocol_sha256,
            assignment_sha256=receipt.assignment_sha256,
            qualification_observation_sha256=(receipt.qualification_observation_sha256),
            base_exactness_result_pointer_sha256=(
                receipt.base_exactness_result_pointer_sha256
            ),
            source_capability_sha256=capability.sha256,
            qualification_authority_sha256=receipt.qualification_authority_sha256,
            trusted_policy_sha256=verified_control.trusted_attester_policy_sha256,
            challenge_sha256=verified_control.challenge_sha256,
            source_identity_sha256=receipt.source_identity_sha256,
            inventory_sha256=receipt.inventory_sha256,
            hardware_envelope_sha256=receipt.hardware_envelope_sha256,
            topology_mode=receipt.topology_mode,
            topology_sha256=receipt.topology_sha256,
            gpu_uuids=receipt.gpu_uuids,
            control_envelope_sha256=verified_control.envelope_sha256,
            challenge_reservation_sha256=expected_reservation,
            _verification_tag=_VERIFIED_GPU_PROOF_SENTINEL,
        )
        if verified.sha256 != self.verified_proof_sha256:
            raise ValueError("distributed verified proof identity changed")
        return verified


def build_distributed_runtime_gpu_proof_artifact(
    *,
    receipt_path: str,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    verified_proof: VerifiedDistributedRuntimeGpuProof,
) -> DistributedRuntimeGpuProofArtifact:
    if type(verified_proof) is not VerifiedDistributedRuntimeGpuProof:
        raise TypeError("distributed proof artifact requires an exact verified proof")
    binding = CanonicalJsonProofBinding.bind(
        receipt_path,
        semantic_sha256=verified_proof.receipt_sha256,
    )
    receipt = DistributedRuntimeGpuProofReceipt.from_dict(binding.reopen())
    if (
        receipt.sha256 != verified_proof.receipt_sha256
        or binding.raw_sha256 != verified_proof.receipt_raw_sha256
        or control_attestation.sha256 != verified_proof.control_envelope_sha256
        or control_attestation.challenge.sha256 != verified_proof.challenge_sha256
    ):
        raise ValueError("distributed proof artifact inputs differ")
    return DistributedRuntimeGpuProofArtifact(
        schema_version=1,
        kind="lightcone_distributed_runtime_gpu_proof_artifact",
        receipt=binding,
        control_attestation=control_attestation,
        replay_reservation=replay_store.bind_reservation(
            verified_proof.challenge_reservation_sha256
        ),
        verified_proof_sha256=verified_proof.sha256,
    )


def validate_distributed_runtime_gpu_proof_artifact(
    artifact_path: str,
    *,
    expected_topology_mode: Literal["tp2_dp1", "tp1_dp2"],
    expected_topology_sha256: str,
    expected_source_identity_sha256: str,
    expected_inventory_sha256: str,
    expected_gpu_uuids: tuple[str, str],
    expected_hardware_envelope_sha256: str,
    expected_assignment_sha256: str,
    expected_qualification_observation_sha256: str,
    expected_base_exactness_result_pointer_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> VerifiedDistributedRuntimeGpuProof:
    """Deep-open one topology-specific durable distributed GPU proof."""

    for label, value in (
        ("distributed expected topology", expected_topology_sha256),
        ("distributed expected source identity", expected_source_identity_sha256),
        ("distributed expected inventory", expected_inventory_sha256),
        ("distributed expected hardware envelope", expected_hardware_envelope_sha256),
        ("distributed expected assignment", expected_assignment_sha256),
        (
            "distributed expected qualification observation",
            expected_qualification_observation_sha256,
        ),
        (
            "distributed expected base exactness result pointer",
            expected_base_exactness_result_pointer_sha256,
        ),
        ("distributed expected root manifest", expected_root_manifest_sha256),
    ):
        _require_hash(label, value)
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = DistributedRuntimeGpuProofArtifact.from_dict(binding.reopen())
    if artifact.sha256 != binding.semantic_sha256:
        raise ValueError("distributed proof artifact semantic identity changed")
    verified = artifact.revalidate(now_ns=now_ns)
    receipt = DistributedRuntimeGpuProofReceipt.from_dict(artifact.receipt.reopen())
    if (
        verified.topology_mode != expected_topology_mode
        or verified.topology_sha256 != expected_topology_sha256
        or verified.source_identity_sha256 != expected_source_identity_sha256
        or verified.inventory_sha256 != expected_inventory_sha256
        or verified.gpu_uuids != expected_gpu_uuids
        or verified.hardware_envelope_sha256 != expected_hardware_envelope_sha256
        or verified.runner_protocol_sha256
        != DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[expected_topology_mode]
        or verified.assignment_sha256 != expected_assignment_sha256
        or verified.qualification_observation_sha256
        != expected_qualification_observation_sha256
        or verified.base_exactness_result_pointer_sha256
        != expected_base_exactness_result_pointer_sha256
        or receipt.sha256 != verified.receipt_sha256
        or artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("distributed proof artifact differs from expected identity")
    return verified


@dataclass(frozen=True)
class TopologyIdentity:
    """Complete rank identity used by manifests and rank receipts."""

    tensor_parallel_size: int
    data_parallel_size: int
    node_count: int
    node_id: str
    node_rank: int
    global_rank: int
    local_rank: int
    tensor_parallel_rank: int
    data_parallel_rank: int
    device_id: str
    rendezvous_id: str
    router_id: str
    clock_id: str

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "node_count",
        ):
            _require_counter(name, getattr(self, name), minimum=1)
        for name in (
            "node_rank",
            "global_rank",
            "local_rank",
            "tensor_parallel_rank",
            "data_parallel_rank",
        ):
            _require_counter(name, getattr(self, name))
        for name in (
            "node_id",
            "device_id",
            "rendezvous_id",
            "router_id",
            "clock_id",
        ):
            _require_nonempty(name, getattr(self, name))
        if self.node_rank >= self.node_count:
            raise ValueError("node_rank is outside the declared node topology")
        if self.global_rank >= self.world_size:
            raise ValueError("global_rank is outside the declared world")
        if self.tensor_parallel_rank >= self.tensor_parallel_size:
            raise ValueError("tensor_parallel_rank is outside its TP group")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise ValueError("data_parallel_rank is outside its DP topology")
        registered_runtime_topology_mode(
            self.tensor_parallel_size,
            self.data_parallel_size,
            self.node_count,
        )
        expected_rank = (
            self.data_parallel_rank * self.tensor_parallel_size
            + self.tensor_parallel_rank
        )
        if self.global_rank != expected_rank:
            raise ValueError("global rank does not match the declared TP/DP ranks")
        if self.node_rank != 0 or self.local_rank != self.global_rank:
            raise ValueError(
                "registered runtime ranks must be host-local with local_rank=global_rank"
            )

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size

    @property
    def mode(self) -> RuntimeTopologyMode:
        return registered_runtime_topology_mode(
            self.tensor_parallel_size,
            self.data_parallel_size,
            self.node_count,
        )

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))

    @property
    def common_identity(self) -> dict[str, int | str]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "node_count": self.node_count,
            "rendezvous_id": self.rendezvous_id,
            "router_id": self.router_id,
            "clock_id": self.clock_id,
        }


@dataclass(frozen=True)
class RankTopologyReceipt:
    """A process-bound observation of one declared rank identity."""

    topology: TopologyIdentity
    process_id: str
    observed_world_size: int

    def __post_init__(self) -> None:
        _require_nonempty("process_id", self.process_id)
        _require_counter("observed_world_size", self.observed_world_size, minimum=1)
        if self.observed_world_size != self.topology.world_size:
            raise ValueError("rank observed a different process-group world size")

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "topology": asdict(self.topology),
                "process_id": self.process_id,
                "observed_world_size": self.observed_world_size,
            }
        )


@dataclass(frozen=True)
class TopologyReceiptSet:
    """Exact all-rank topology coverage with a stable topology digest."""

    receipts: tuple[RankTopologyReceipt, ...]

    def __post_init__(self) -> None:
        if not self.receipts:
            raise ValueError("topology receipts cannot be empty")
        first = self.receipts[0].topology
        expected_ranks = set(range(first.world_size))
        ranks = [receipt.topology.global_rank for receipt in self.receipts]
        if len(ranks) != len(set(ranks)):
            raise ValueError("duplicate topology rank receipt")
        if set(ranks) != expected_ranks:
            raise ValueError("topology receipts do not cover every declared rank")
        if len({receipt.process_id for receipt in self.receipts}) != len(self.receipts):
            raise ValueError("topology process identities must be unique")
        if any(
            receipt.topology.common_identity != first.common_identity
            for receipt in self.receipts
        ):
            raise ValueError("rank receipts disagree on the common topology identity")
        devices = [receipt.topology.device_id for receipt in self.receipts]
        if len(devices) != len(set(devices)):
            raise ValueError("each rank must bind a distinct device identity")
        node_pairs = {
            (receipt.topology.node_rank, receipt.topology.node_id)
            for receipt in self.receipts
        }
        if len({rank for rank, _ in node_pairs}) != first.node_count:
            raise ValueError("topology receipts do not cover every declared node")
        if len({node_id for _, node_id in node_pairs}) != first.node_count:
            raise ValueError("node ranks and node identities are not one-to-one")
        local_ranks = [
            (receipt.topology.node_rank, receipt.topology.local_rank)
            for receipt in self.receipts
        ]
        if len(local_ranks) != len(set(local_ranks)):
            raise ValueError("local rank is duplicated within a node")

    @property
    def world_size(self) -> int:
        return self.receipts[0].topology.world_size

    @property
    def tensor_parallel_size(self) -> int:
        return self.receipts[0].topology.tensor_parallel_size

    @property
    def data_parallel_size(self) -> int:
        return self.receipts[0].topology.data_parallel_size

    @property
    def mode(self) -> RuntimeTopologyMode:
        return self.receipts[0].topology.mode

    @property
    def topology_sha256(self) -> str:
        identities = [
            asdict(receipt.topology)
            for receipt in sorted(
                self.receipts, key=lambda item: item.topology.global_rank
            )
        ]
        return _sha256(identities)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(
            [
                receipt.sha256
                for receipt in sorted(
                    self.receipts, key=lambda item: item.topology.global_rank
                )
            ]
        )

    def receipt_for_rank(self, rank: int) -> RankTopologyReceipt:
        _require_counter("rank", rank)
        for receipt in self.receipts:
            if receipt.topology.global_rank == rank:
                return receipt
        raise KeyError(f"rank {rank} is outside the topology")

    def tensor_parallel_group(self, data_parallel_rank: int) -> tuple[int, ...]:
        _require_counter("data_parallel_rank", data_parallel_rank)
        if data_parallel_rank >= self.data_parallel_size:
            raise ValueError("data_parallel_rank is outside the topology")
        start = data_parallel_rank * self.tensor_parallel_size
        return tuple(range(start, start + self.tensor_parallel_size))


class ParameterOwnership(str, Enum):
    SHARDED = "sharded"
    REPLICATED = "replicated"


@dataclass(frozen=True)
class InferenceParameterOwnership:
    """Inference-aligned state ownership without cross-DP convenience gathers."""

    parameter_name: str
    ownership: ParameterOwnership
    owner_ranks: tuple[int, ...]
    shard_axis: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty("parameter_name", self.parameter_name)
        if not self.owner_ranks:
            raise ValueError("parameter ownership requires at least one rank")
        if len(set(self.owner_ranks)) != len(self.owner_ranks):
            raise ValueError("parameter owner ranks must be unique")
        for rank in self.owner_ranks:
            _require_counter("owner rank", rank)
        if self.ownership is ParameterOwnership.SHARDED:
            if self.shard_axis is None:
                raise ValueError("sharded parameters require a shard axis")
            _require_counter("shard_axis", self.shard_axis)
        elif self.shard_axis is not None:
            raise ValueError("replicated parameters cannot declare a shard axis")

    def validate(self, topology: TopologyReceiptSet) -> None:
        if any(rank >= topology.world_size for rank in self.owner_ranks):
            raise ValueError("parameter owner rank is outside the topology")
        owner_set = set(self.owner_ranks)
        for replica in range(topology.data_parallel_size):
            group = set(topology.tensor_parallel_group(replica))
            overlap = group & owner_set
            if overlap and overlap != group:
                raise ValueError("parameter ownership partially covers a TP replica")

    def gradient_reduction_ranks(
        self,
        rank: int,
        topology: TopologyReceiptSet,
    ) -> tuple[int, ...]:
        self.validate(topology)
        if rank not in self.owner_ranks:
            raise ValueError("rank does not own this parameter")
        if self.ownership is ParameterOwnership.SHARDED:
            return (rank,)
        replica = rank // topology.tensor_parallel_size
        return tuple(
            member
            for member in topology.tensor_parallel_group(replica)
            if member in self.owner_ranks
        )


@dataclass(frozen=True)
class CohortRouteIdentity:
    tenant_id: str
    cohort_sha256: str
    router_id: str
    topology_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty("tenant_id", self.tenant_id)
        _require_nonempty("router_id", self.router_id)
        _require_hash("cohort_sha256", self.cohort_sha256)
        _require_hash("topology_sha256", self.topology_sha256)

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


class ReplicaLocalRouter:
    """Sticky cohort routing; DP replicas never average adaptation gradients."""

    data_parallel_gradient_averaging = False

    def __init__(self, topology: TopologyReceiptSet) -> None:
        self.topology = topology
        self._routes: dict[str, int] = {}

    def route(self, identity: CohortRouteIdentity) -> int:
        reference = self.topology.receipts[0].topology
        if identity.topology_sha256 != self.topology.topology_sha256:
            raise ValueError("cohort route belongs to another topology")
        if identity.router_id != reference.router_id:
            raise ValueError("cohort route belongs to another router")
        existing = self._routes.get(identity.sha256)
        if existing is not None:
            return existing
        replica = int(identity.sha256[:16], 16) % self.topology.data_parallel_size
        self._routes[identity.sha256] = replica
        return replica

    def ranks_for(self, identity: CohortRouteIdentity) -> tuple[int, ...]:
        return self.topology.tensor_parallel_group(self.route(identity))


@dataclass(frozen=True)
class UpdateIdentity:
    """Retry-stable identity for exactly one source update."""

    cohort_sha256: str
    source_version: int
    cohort_epoch: int
    sequence_number: int
    source_rows_sha256: str

    def __post_init__(self) -> None:
        _require_hash("cohort_sha256", self.cohort_sha256)
        _require_hash("source_rows_sha256", self.source_rows_sha256)
        for name in ("source_version", "cohort_epoch", "sequence_number"):
            _require_counter(name, getattr(self, name))

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class PublicationCandidate:
    update: UpdateIdentity
    buffer_generation: int
    optimizer_generation: int

    def __post_init__(self) -> None:
        _require_counter("buffer_generation", self.buffer_generation)
        _require_counter("optimizer_generation", self.optimizer_generation)

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "update_sha256": self.update.sha256,
                "buffer_generation": self.buffer_generation,
                "optimizer_generation": self.optimizer_generation,
            }
        )


@dataclass(frozen=True)
class RankPrepare:
    rank: int
    topology_receipt_sha256: str
    candidate_sha256: str
    source_version: int
    cohort_epoch: int
    buffer_generation: int
    optimizer_generation: int
    ready: bool
    finite: bool
    memory_reserved: bool
    safe_boundary: bool
    process_group_healthy: bool = True

    def __post_init__(self) -> None:
        _require_counter("rank", self.rank)
        _require_hash("topology_receipt_sha256", self.topology_receipt_sha256)
        _require_hash("candidate_sha256", self.candidate_sha256)
        for name in (
            "source_version",
            "cohort_epoch",
            "buffer_generation",
            "optimizer_generation",
        ):
            _require_counter(name, getattr(self, name))
        for name in (
            "ready",
            "finite",
            "memory_reserved",
            "safe_boundary",
            "process_group_healthy",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


class PrepareDisposition(str, Enum):
    COMMIT_READY = "commit_ready"
    ABORT_STATIC = "abort_static"
    PROCESS_GROUP_FAILURE = "process_group_failure"


@dataclass(frozen=True)
class PreparedPublication:
    update_sha256: str
    candidate_sha256: str
    topology_sha256: str
    disposition: PrepareDisposition
    reasons: tuple[str, ...]
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("prepared update", self.update_sha256),
            ("prepared candidate", self.candidate_sha256),
            ("prepared topology", self.topology_sha256),
        ):
            _require_hash(name, value)
        if type(self.disposition) is not PrepareDisposition:
            raise TypeError("prepared publication disposition must be exact")
        if (
            type(self.reasons) is not tuple
            or not self.reasons
            or self.reasons != tuple(sorted(set(self.reasons)))
        ):
            raise ValueError("prepared publication reasons must be canonical")
        if self.ranks != tuple(range(len(self.ranks))) or not self.ranks:
            raise ValueError("prepared publication requires exact all-rank coverage")

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "update_sha256": self.update_sha256,
                "candidate_sha256": self.candidate_sha256,
                "topology_sha256": self.topology_sha256,
                "disposition": self.disposition.value,
                "reasons": self.reasons,
                "ranks": self.ranks,
            }
        )


class PublicationOutcome(str, Enum):
    COMMIT = "commit"
    ABORT_STATIC = "abort_static"
    PROCESS_GROUP_FAILURE = "process_group_failure"


@dataclass(frozen=True)
class PublicationDecision:
    update_sha256: str
    candidate_sha256: str
    topology_sha256: str
    outcome: PublicationOutcome
    reasons: tuple[str, ...]
    ranks: tuple[int, ...]
    service_ready: bool
    admission_allowed: bool
    restart_required: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("decision update", self.update_sha256),
            ("decision candidate", self.candidate_sha256),
            ("decision topology", self.topology_sha256),
        ):
            _require_hash(name, value)
        if type(self.outcome) is not PublicationOutcome:
            raise TypeError("publication outcome must be exact")
        if (
            type(self.reasons) is not tuple
            or not self.reasons
            or self.reasons != tuple(sorted(set(self.reasons)))
        ):
            raise ValueError("publication decision reasons must be canonical")
        if self.ranks != tuple(range(len(self.ranks))) or not self.ranks:
            raise ValueError("publication decision requires exact all-rank coverage")
        state = (
            self.service_ready,
            self.admission_allowed,
            self.restart_required,
        )
        expected = (
            (False, False, True)
            if self.outcome is PublicationOutcome.PROCESS_GROUP_FAILURE
            else (False, False, False)
        )
        if state != expected:
            raise ValueError("publication decision service state is not fail-closed")

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "update_sha256": self.update_sha256,
                "candidate_sha256": self.candidate_sha256,
                "topology_sha256": self.topology_sha256,
                "outcome": self.outcome.value,
                "reasons": self.reasons,
                "ranks": self.ranks,
                "service_ready": self.service_ready,
                "admission_allowed": self.admission_allowed,
                "restart_required": self.restart_required,
            }
        )


@dataclass(frozen=True)
class RankDecisionReceipt:
    rank: int
    topology_receipt_sha256: str
    decision_sha256: str
    applied: bool

    def __post_init__(self) -> None:
        _require_counter("rank", self.rank)
        _require_hash("topology_receipt_sha256", self.topology_receipt_sha256)
        _require_hash("decision_sha256", self.decision_sha256)
        if type(self.applied) is not bool:
            raise ValueError("applied must be a boolean")


def validate_decision_receipts(
    decision: PublicationDecision,
    receipts: tuple[RankDecisionReceipt, ...],
    topology: TopologyReceiptSet,
) -> None:
    """Reject missing, duplicate, mixed, or foreign all-rank outcomes."""

    if decision.topology_sha256 != topology.topology_sha256:
        raise ValueError("publication decision belongs to another topology")
    if decision.ranks != tuple(range(topology.world_size)):
        raise ValueError("publication decision lacks exact all-rank coverage")
    ranks = [receipt.rank for receipt in receipts]
    expected = set(range(topology.world_size))
    if len(ranks) != len(set(ranks)):
        raise ValueError("duplicate publication decision receipt")
    if set(ranks) != expected:
        raise ValueError("publication decision receipts lack all-rank coverage")
    should_apply = decision.outcome is PublicationOutcome.COMMIT
    for receipt in receipts:
        topology_receipt = topology.receipt_for_rank(receipt.rank)
        if receipt.topology_receipt_sha256 != topology_receipt.sha256:
            raise ValueError("publication receipt binds another rank topology")
        if receipt.decision_sha256 != decision.sha256:
            raise ValueError("ranks did not observe one publication decision")
        if receipt.applied is not should_apply:
            raise ValueError("publication receipt would create a partial model")


class AllRankPublicationCoordinator:
    """Collective prepare/decide protocol with fail-closed service state."""

    def __init__(self, topology: TopologyReceiptSet) -> None:
        self.topology = topology
        self.service_ready = True
        self.admission_allowed = True
        self.restart_required = False
        self._pending: PreparedPublication | None = None
        self._pending_decision: PublicationDecision | None = None
        self._consumed_updates: set[str] = set()

    def prepare(
        self,
        candidate: PublicationCandidate,
        votes: tuple[RankPrepare, ...],
    ) -> PreparedPublication:
        if self.restart_required:
            raise RuntimeError("process group restart is required before preparation")
        if self._pending is not None or self._pending_decision is not None:
            raise RuntimeError("a publication decision is already pending")
        expected_ranks = tuple(range(self.topology.world_size))
        vote_ranks = [vote.rank for vote in votes]
        if len(vote_ranks) != len(set(vote_ranks)):
            raise ValueError("duplicate rank prepare vote")

        process_failures: list[str] = []
        missing = sorted(set(expected_ranks) - set(vote_ranks))
        unexpected = sorted(set(vote_ranks) - set(expected_ranks))
        if missing:
            process_failures.append(f"missing_ranks:{','.join(map(str, missing))}")
        if unexpected:
            process_failures.append(
                f"unexpected_ranks:{','.join(map(str, unexpected))}"
            )

        invalid: list[str] = []
        if candidate.update.sha256 in self._consumed_updates:
            invalid.append("duplicate_update_identity")
        for vote in sorted(votes, key=lambda item: item.rank):
            if vote.rank not in expected_ranks:
                continue
            topology_receipt = self.topology.receipt_for_rank(vote.rank)
            if vote.topology_receipt_sha256 != topology_receipt.sha256:
                process_failures.append(f"rank_{vote.rank}:topology_receipt_mismatch")
            if not vote.process_group_healthy:
                process_failures.append(f"rank_{vote.rank}:process_group_failed")
            checks = (
                (vote.candidate_sha256 == candidate.sha256, "candidate_identity"),
                (
                    vote.source_version == candidate.update.source_version,
                    "source_version",
                ),
                (vote.cohort_epoch == candidate.update.cohort_epoch, "cohort_epoch"),
                (
                    vote.buffer_generation == candidate.buffer_generation,
                    "buffer_generation",
                ),
                (
                    vote.optimizer_generation == candidate.optimizer_generation,
                    "optimizer_generation",
                ),
                (vote.ready, "readiness"),
                (vote.finite, "finiteness"),
                (vote.memory_reserved, "memory_reservation"),
                (vote.safe_boundary, "safe_boundary"),
            )
            invalid.extend(
                f"rank_{vote.rank}:{reason}" for passed, reason in checks if not passed
            )

        if process_failures:
            disposition = PrepareDisposition.PROCESS_GROUP_FAILURE
            reasons = tuple(sorted(set(process_failures)))
        elif invalid:
            disposition = PrepareDisposition.ABORT_STATIC
            reasons = tuple(sorted(set(invalid)))
        else:
            disposition = PrepareDisposition.COMMIT_READY
            reasons = ("all_ranks_prepared",)
        prepared = PreparedPublication(
            update_sha256=candidate.update.sha256,
            candidate_sha256=candidate.sha256,
            topology_sha256=self.topology.topology_sha256,
            disposition=disposition,
            reasons=reasons,
            ranks=expected_ranks,
        )
        self._pending = prepared
        return prepared

    def decide(self, prepared: PreparedPublication) -> PublicationDecision:
        if self._pending is None or prepared.sha256 != self._pending.sha256:
            raise RuntimeError("prepared publication is not the active collective")
        outcome = {
            PrepareDisposition.COMMIT_READY: PublicationOutcome.COMMIT,
            PrepareDisposition.ABORT_STATIC: PublicationOutcome.ABORT_STATIC,
            PrepareDisposition.PROCESS_GROUP_FAILURE: (
                PublicationOutcome.PROCESS_GROUP_FAILURE
            ),
        }[prepared.disposition]
        process_failed = outcome is PublicationOutcome.PROCESS_GROUP_FAILURE
        decision = PublicationDecision(
            update_sha256=prepared.update_sha256,
            candidate_sha256=prepared.candidate_sha256,
            topology_sha256=prepared.topology_sha256,
            outcome=outcome,
            reasons=prepared.reasons,
            ranks=prepared.ranks,
            service_ready=False,
            admission_allowed=False,
            restart_required=process_failed,
        )
        self.service_ready = decision.service_ready
        self.admission_allowed = decision.admission_allowed
        self.restart_required = decision.restart_required
        self._pending = None
        if process_failed:
            self._pending_decision = None
            self._consumed_updates.add(prepared.update_sha256)
        else:
            self._pending_decision = decision
        return decision

    def finalize(
        self,
        decision: PublicationDecision,
        receipts: tuple[RankDecisionReceipt, ...],
    ) -> None:
        """Open admission only after every rank receipts the same copy outcome."""

        if (
            self._pending_decision is None
            or decision.sha256 != self._pending_decision.sha256
        ):
            raise RuntimeError("publication decision is not awaiting finalization")
        try:
            validate_decision_receipts(decision, receipts, self.topology)
        except ValueError:
            self.mark_collective_failed()
            raise
        self._consumed_updates.add(decision.update_sha256)
        self._pending_decision = None
        self.service_ready = True
        self.admission_allowed = True
        self.restart_required = False

    def mark_process_group_restarted(self, topology: TopologyReceiptSet) -> None:
        if not self.restart_required:
            raise RuntimeError("no process-group restart is pending")
        if topology.topology_sha256 != self.topology.topology_sha256:
            raise ValueError("restart topology differs from the failed topology")
        self.topology = topology
        self.service_ready = True
        self.admission_allowed = True
        self.restart_required = False
        self._pending_decision = None

    def mark_collective_failed(self) -> None:
        """Fail service readiness after a transport exception or split decision."""
        self._pending = None
        self._pending_decision = None
        self.service_ready = False
        self.admission_allowed = False
        self.restart_required = True


class GlooPublicationTransport:
    """Real CPU collective harness for the all-rank publication state machine.

    This class intentionally rejects NCCL. It validates process-group behavior without
    pretending to test CUDA streams, graph boundaries, fixed-address copies, or NCCL.
    """

    def __init__(
        self,
        topology: TopologyReceiptSet,
        *,
        local_rank: int,
        process_group: object | None = None,
    ) -> None:
        _require_counter("local_rank", local_rank)
        if local_rank >= topology.world_size:
            raise ValueError("local rank is outside the topology")
        self.topology = topology
        self.local_rank = local_rank
        self.process_group = process_group
        self.coordinator = AllRankPublicationCoordinator(topology)

    def _distributed(self) -> object:
        from torch import distributed

        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeError("a live gloo process group is required")
        if distributed.get_backend(self.process_group) != "gloo":
            raise RuntimeError("CPU publication harness requires the gloo backend")
        if distributed.get_world_size(self.process_group) != self.topology.world_size:
            raise RuntimeError(
                "process-group world size differs from topology receipts"
            )
        if distributed.get_rank(self.process_group) != self.local_rank:
            raise RuntimeError(
                "process-group rank differs from the local topology rank"
            )
        return distributed

    def prepare_and_decide(
        self,
        candidate: PublicationCandidate,
        local_vote: RankPrepare,
    ) -> PublicationDecision:
        if local_vote.rank != self.local_rank:
            raise ValueError("local prepare vote belongs to another rank")
        distributed = self._distributed()
        gathered: list[RankPrepare | None] = [None] * self.topology.world_size
        try:
            distributed.all_gather_object(
                gathered,
                local_vote,
                group=self.process_group,
            )
            if not all(isinstance(vote, RankPrepare) for vote in gathered):
                raise RuntimeError("collective returned an invalid prepare vote")
            prepared = self.coordinator.prepare(candidate, tuple(gathered))
            decision = self.coordinator.decide(prepared)
            decisions: list[str | None] = [None] * self.topology.world_size
            distributed.all_gather_object(
                decisions,
                decision.sha256,
                group=self.process_group,
            )
            if decisions != [decision.sha256] * self.topology.world_size:
                raise RuntimeError("ranks derived different publication decisions")
            return decision
        except Exception:
            self.coordinator.mark_collective_failed()
            raise

    def finalize(
        self,
        decision: PublicationDecision,
        *,
        applied: bool,
    ) -> tuple[RankDecisionReceipt, ...]:
        """Gather post-copy receipts; partial application makes service unready."""
        distributed = self._distributed()
        local = RankDecisionReceipt(
            rank=self.local_rank,
            topology_receipt_sha256=(
                self.topology.receipt_for_rank(self.local_rank).sha256
            ),
            decision_sha256=decision.sha256,
            applied=applied,
        )
        gathered: list[RankDecisionReceipt | None] = [None] * self.topology.world_size
        try:
            distributed.all_gather_object(
                gathered,
                local,
                group=self.process_group,
            )
            if not all(isinstance(row, RankDecisionReceipt) for row in gathered):
                raise RuntimeError("collective returned an invalid decision receipt")
            receipts = tuple(gathered)
            self.coordinator.finalize(decision, receipts)
            return receipts
        except Exception:
            self.coordinator.mark_collective_failed()
            raise
