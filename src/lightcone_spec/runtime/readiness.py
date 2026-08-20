"""CPU-testable native hot-path contracts that remain GPU fail-closed.

These state machines validate identities and ordering only.  They do not claim
that the pinned SGLang patch emits the observations, that CUDA graphs preserve
the addresses, or that a device hot path avoided synchronization.  A future
semantic patch and GPU proof must be source-allowlisted before either receipt
can become formal evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal

from torch import Tensor

from .compile_cache import PINNED_SGLANG_PATCH_MANIFEST_SHA256
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
    FORMAL_RUNTIME_QUALIFICATION_CORE_SUITES,
    FORMAL_RUNTIME_QUALIFICATION_EAGLE_RESOLUTION,
    FORMAL_RUNTIME_QUALIFICATION_OPTIONAL_SUITES,
    build_formal_runtime_qualification_authority_sha256,
)


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


NativeQualificationSuite = Literal[
    "chronobelief_gpu_parity",
    "dspark_tp1",
    "dspark_tp2",
    "dspark_dp2",
    "eagle3_tp1",
    "nextn_tp1",
    "nextn_tp2",
    "native_hot_path_tp1",
    "session_reset_tp1",
]

NATIVE_RUNTIME_QUALIFICATION_TESTS = MappingProxyType(
    {
        "chronobelief_gpu_parity": (
            "chronobelief_fp16_gpu_parity",
            "chronobelief_bf16_gpu_parity",
            "chronobelief_fp32_gpu_parity",
            "chronobelief_safe_boundary_age_exactness",
            "chronobelief_abort_no_state_advance",
            "chronobelief_skip_no_state_advance",
            "chronobelief_commit_once_decoupled_decay",
            "chronobelief_nonfinite_overflow_fail_closed",
        ),
        "dspark_tp1": (
            "dspark_real_predecessor",
            "dspark_markov_w1_w2",
            "dspark_confidence_head",
            "dspark_56_scope_selector",
            "dspark_fixed_budget",
            "dspark_native_scheduler",
            "native_itl_pointer",
            "graph_fixed_address_no_host_sync",
        ),
        "dspark_tp2": (
            "dspark_tp2_real_predecessor",
            "dspark_tp2_native_heads",
            "dspark_tp2_selector",
            "dspark_tp2_all_rank_prepare",
            "dspark_tp2_two_phase_commit",
            "dspark_tp2_one_rank_abort_zero_partial",
            "native_itl_pointer",
            "graph_fixed_address_no_host_sync",
        ),
        "dspark_dp2": (
            "dspark_dp2_real_predecessor",
            "dspark_dp2_native_heads",
            "dspark_dp2_selector",
            "dspark_dp2_sticky_routing",
            "dspark_dp2_replica_state_isolation",
            "dspark_dp2_zero_cross_replica_gradient",
            "native_itl_pointer",
            "graph_fixed_address_no_host_sync",
        ),
        "eagle3_tp1": (
            "eagle3_official_selector_binding",
            "eagle3_target_revision_binding",
            "eagle3_drafter_revision_binding",
            "eagle3_interface_binding",
            "eagle3_source_commit_binding",
            "eagle3_live_scheduler_boundary",
            "native_itl_pointer",
            "graph_fixed_address_no_host_sync",
        ),
        "nextn_tp1": (
            "nextn_mtp_hidden_interface",
            "nextn_teacher_rows",
            "nextn_valid_mask",
            "nextn_source_adapter_version",
            "nextn_target_model_authority",
            "nextn_drafter_model_authority",
            "native_itl_pointer",
            "graph_fixed_address_no_host_sync",
        ),
        "nextn_tp2": (
            "nextn_mtp_hidden_interface_tp2",
            "nextn_teacher_mask_tp2",
            "nextn_source_adapter_version_tp2",
            "nextn_target_two_shard_authority",
            "nextn_drafter_two_shard_authority",
            "nextn_tp2_sharded_candidate_parity",
            "native_itl_pointer",
            "graph_fixed_address_no_host_sync",
        ),
        "native_hot_path_tp1": (
            "native_itl_full_token_coverage",
            "native_itl_monotonic_clock",
            "native_itl_pointer_stability",
            "cuda_stream_event_dependency",
            "graph_input_pointer_stability",
            "graph_candidate_pointer_stability",
            "no_blocking_d2h",
            "no_host_synchronize",
        ),
        "session_reset_tp1": (
            "session_reuse_token_trajectory",
            "session_reset_state_receipt",
            "session_hbm_drift",
            "session_graph_pointer_stability",
            "session_startup_latency_comparison",
            "session_http_connection_accounting",
            "session_fault_fallback",
            "session_terminal_lifecycle",
        ),
    }
)

NATIVE_RUNTIME_SUITE_CAPABILITIES = MappingProxyType(
    {
        "chronobelief_gpu_parity": ("chronobelief_gpu_parity",),
        "dspark_tp1": ("dspark", "graph_hot_path", "native_itl"),
        "dspark_tp2": (
            "dspark",
            "graph_hot_path",
            "native_itl",
            "tp2_all_rank_publication",
        ),
        "dspark_dp2": (
            "dspark",
            "graph_hot_path",
            "native_itl",
            "dp2_sticky_isolation",
        ),
        "eagle3_tp1": ("eagle3", "graph_hot_path", "native_itl"),
        "nextn_tp1": ("graph_hot_path", "native_itl", "nextn"),
        "nextn_tp2": ("graph_hot_path", "native_itl", "nextn"),
        "native_hot_path_tp1": ("graph_hot_path", "native_itl"),
        "session_reset_tp1": ("session_reset",),
    }
)

NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S = MappingProxyType(
    {
        suite_id: _sha256(
            {
                "schema_version": 1,
                "kind": "source_owned_native_runtime_gpu_qualification_runner",
                "suite_id": suite_id,
                "test_names": test_names,
                "remote_private_key": False,
                "gpu_process_gate": "pre_and_post_empty_exact_uuid",
                "live_observation": "path_bound_first_party_server_worker_terminal",
                "junit": "exact_all_pass_zero_skip",
                "trust_lift": "local_external_control_atomic_replay",
            }
        )
        for suite_id, test_names in NATIVE_RUNTIME_QUALIFICATION_TESTS.items()
    }
)


@dataclass(frozen=True)
class NativeRuntimeReleaseCapability:
    schema_version: Literal[1]
    pinned_sglang_commit: str
    patched_sglang_tree: str
    semantic_patch_sha256: str
    suite_protocol_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("native runtime release capability schema is unsupported")
        for label, value, length in (
            ("native pinned SGLang commit", self.pinned_sglang_commit, 40),
            ("native patched SGLang tree", self.patched_sglang_tree, 40),
            ("native semantic patch", self.semantic_patch_sha256, 64),
            ("native suite protocol", self.suite_protocol_sha256, 64),
        ):
            if len(value) != length or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{label} must be a lower-case digest")

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


NATIVE_RUNTIME_RELEASE_CAPABILITY = NativeRuntimeReleaseCapability(
    schema_version=1,
    pinned_sglang_commit="3312645a307453893a00778592f105581e3d1c3d",
    patched_sglang_tree="bb6371242e82592d1b8a2f5f4ba6d0630d8365cb",
    semantic_patch_sha256=(
        "0c4db4f8798645c0ba65e97031030fb5e891d15f63cd75105fc1e1656c1a2874"
    ),
    suite_protocol_sha256=_sha256(
        {
            "schema_version": 1,
            "suites": dict(NATIVE_RUNTIME_QUALIFICATION_TESTS),
            "capabilities": dict(NATIVE_RUNTIME_SUITE_CAPABILITIES),
            "required": "all_named_tests_pass_zero_skip",
        }
    ),
)

FORMAL_RUNTIME_QUALIFICATION_TESTS = MappingProxyType(
    {
        **dict(NATIVE_RUNTIME_QUALIFICATION_TESTS),
        **dict(DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS),
    }
)
FORMAL_RUNTIME_QUALIFICATION_SUITE_RUNNER_PROTOCOL_SHA256S = MappingProxyType(
    {
        **dict(NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S),
        **dict(DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S),
    }
)
NATIVE_RUNTIME_QUALIFICATION_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 2,
        "kind": "formal_runtime_qualification_closed_protocol",
        "native_release_capability_sha256": (NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256),
        "distributed_gpu_proof_protocol_sha256": (
            DISTRIBUTED_RUNTIME_GPU_PROOF_PROTOCOL_SHA256
        ),
        "required_core_suites": FORMAL_RUNTIME_QUALIFICATION_CORE_SUITES,
        "optional_suites": FORMAL_RUNTIME_QUALIFICATION_OPTIONAL_SUITES,
        "eagle3_resolution": FORMAL_RUNTIME_QUALIFICATION_EAGLE_RESOLUTION,
        "test_count_per_executed_suite": 8,
        "caller_resolved_suite_set": "forbidden",
    }
)
NATIVE_RUNTIME_QUALIFICATION_RUNNER_SHA256 = _sha256(
    dict(FORMAL_RUNTIME_QUALIFICATION_SUITE_RUNNER_PROTOCOL_SHA256S)
)
NATIVE_RUNTIME_QUALIFICATION_TEST_SET_SHA256 = _sha256(
    {
        "tests": dict(FORMAL_RUNTIME_QUALIFICATION_TESTS),
        "required_core_suites": FORMAL_RUNTIME_QUALIFICATION_CORE_SUITES,
        "optional_suites": FORMAL_RUNTIME_QUALIFICATION_OPTIONAL_SUITES,
        "eagle3_resolution": FORMAL_RUNTIME_QUALIFICATION_EAGLE_RESOLUTION,
    }
)
NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256 = (
    build_formal_runtime_qualification_authority_sha256(
        native_runtime_release_capability_sha256=(
            NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
        ),
        qualification_protocol_sha256=(NATIVE_RUNTIME_QUALIFICATION_PROTOCOL_SHA256),
        qualification_runner_sha256=NATIVE_RUNTIME_QUALIFICATION_RUNNER_SHA256,
        qualification_test_set_sha256=(NATIVE_RUNTIME_QUALIFICATION_TEST_SET_SHA256),
        patched_sglang_tree=NATIVE_RUNTIME_RELEASE_CAPABILITY.patched_sglang_tree,
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    )
)

HotPathOperation = Literal[
    "cuda_event_record",
    "cuda_stream_wait_event",
    "device_to_device_copy",
    "graph_replay",
    "blocking_d2h",
    "host_synchronize",
    "tensor_item",
]

_ALLOWED_HOT_PATH_OPERATIONS: tuple[HotPathOperation, ...] = (
    "cuda_event_record",
    "cuda_stream_wait_event",
    "device_to_device_copy",
    "graph_replay",
)


def _require_sha256(label: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_text(label: str, value: str) -> None:
    if not value or value.strip() != value or len(value) > 192:
        raise ValueError(f"{label} must be a canonical non-empty identifier")


def _require_nonnegative_int(label: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


class NativeReadinessBlocked(RuntimeError):
    """A structural contract exists, but source-owned GPU proof does not."""

    def __init__(self, reason_code: str) -> None:
        _require_text("native readiness reason", reason_code)
        super().__init__(f"native readiness is BLOCKED: {reason_code}")
        self.reason_code = reason_code


@dataclass(frozen=True)
class NativeRuntimeGpuProofReceipt:
    """One unsigned remote exact-suite receipt awaiting local trust lift."""

    schema_version: Literal[1]
    kind: Literal["lightcone_native_runtime_gpu_proof"]
    suite_id: NativeQualificationSuite
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    topology_sha256: str
    runner_protocol_sha256: str
    assignment_sha256: str
    qualification_observation_sha256: str
    source_capability_sha256: str
    pinned_sglang_commit: str
    patched_sglang_tree: str
    semantic_patch_sha256: str
    run_nonce_sha256: str
    qualification_authority_sha256: str
    source_identity_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    hardware_envelope_sha256: str
    junit_xml_sha256: str
    test_names: tuple[str, ...]
    tests_collected: int
    tests_passed: int
    tests_failed: Literal[0]
    tests_errored: Literal[0]
    tests_skipped: Literal[0]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_native_runtime_gpu_proof"
        ):
            raise ValueError("native runtime GPU proof schema is unsupported")
        if self.suite_id not in NATIVE_RUNTIME_QUALIFICATION_TESTS:
            raise ValueError("native runtime GPU proof suite is unregistered")
        capability = NATIVE_RUNTIME_RELEASE_CAPABILITY
        if (
            self.source_capability_sha256 != capability.sha256
            or self.pinned_sglang_commit != capability.pinned_sglang_commit
            or self.patched_sglang_tree != capability.patched_sglang_tree
            or self.semantic_patch_sha256 != capability.semantic_patch_sha256
        ):
            raise ValueError("native runtime GPU proof differs from source capability")
        if (
            self.runner_protocol_sha256
            != NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[self.suite_id]
        ):
            raise ValueError("native runtime GPU proof runner protocol differs")
        for label, value in (
            ("native proof topology", self.topology_sha256),
            ("native proof runner protocol", self.runner_protocol_sha256),
            ("native proof assignment", self.assignment_sha256),
            (
                "native proof qualification observation",
                self.qualification_observation_sha256,
            ),
            ("native proof source capability", self.source_capability_sha256),
            ("native proof run nonce", self.run_nonce_sha256),
            (
                "native proof qualification authority",
                self.qualification_authority_sha256,
            ),
            ("native proof source identity", self.source_identity_sha256),
            ("native proof inventory", self.inventory_sha256),
            ("native proof hardware envelope", self.hardware_envelope_sha256),
            ("native proof JUnit XML", self.junit_xml_sha256),
        ):
            _require_sha256(label, value)
        if (
            self.qualification_authority_sha256
            != NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256
        ):
            raise ValueError("native proof qualification authority differs")
        expected_topology = {
            "nextn_tp2": "tp2_dp1",
            "dspark_tp2": "tp2_dp1",
            "dspark_dp2": "tp1_dp2",
        }.get(self.suite_id, "tp1_dp1")
        if self.topology_mode != expected_topology:
            raise ValueError("native qualification suite differs from topology")
        expected_gpus = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpus
            or len(set(self.gpu_uuids)) != expected_gpus
            or any(not value or value.strip() != value for value in self.gpu_uuids)
        ):
            raise ValueError("native proof has the wrong unique GPU coverage")
        expected_tests = NATIVE_RUNTIME_QUALIFICATION_TESTS[self.suite_id]
        if self.test_names != expected_tests:
            raise ValueError("native proof lacks the exact named qualification suite")
        if (
            self.tests_collected,
            self.tests_passed,
            self.tests_failed,
            self.tests_errored,
            self.tests_skipped,
        ) != (len(expected_tests), len(expected_tests), 0, 0, 0):
            raise ValueError(
                "native proof requires every named test with zero skip/failure"
            )

    @property
    def backend_capabilities(self) -> tuple[str, ...]:
        return NATIVE_RUNTIME_SUITE_CAPABILITIES[self.suite_id]

    @property
    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "suite_id": self.suite_id,
            "topology_mode": self.topology_mode,
            "topology_sha256": self.topology_sha256,
            "runner_protocol_sha256": self.runner_protocol_sha256,
            "assignment_sha256": self.assignment_sha256,
            "qualification_observation_sha256": (self.qualification_observation_sha256),
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
            "test_names": list(self.test_names),
            "tests_collected": self.tests_collected,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_errored": self.tests_errored,
            "tests_skipped": self.tests_skipped,
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
    def from_dict(cls, value: object) -> NativeRuntimeGpuProofReceipt:
        if type(value) is not dict:
            raise TypeError("native runtime GPU proof receipt must be an object")
        expected = {
            "schema_version",
            "kind",
            "suite_id",
            "topology_mode",
            "topology_sha256",
            "runner_protocol_sha256",
            "assignment_sha256",
            "qualification_observation_sha256",
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
            "test_names",
            "tests_collected",
            "tests_passed",
            "tests_failed",
            "tests_errored",
            "tests_skipped",
        }
        if set(value) != expected:
            raise ValueError("native runtime GPU proof receipt fields differ")
        row = dict(value)
        gpu_uuids = row.pop("gpu_uuids")
        test_names = row.pop("test_names")
        if type(gpu_uuids) is not list or type(test_names) is not list:
            raise TypeError("native runtime GPU proof receipt arrays are malformed")
        return cls(
            gpu_uuids=tuple(gpu_uuids),
            test_names=tuple(test_names),
            **row,
        )

    @property
    def control_lineage_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_native_runtime_gpu_proof_control_lineage",
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
                "inventory_sha256": self.inventory_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "topology_sha256": self.topology_sha256,
                "gpu_uuids": self.gpu_uuids,
                "junit_xml_sha256": self.junit_xml_sha256,
            }
        )


_VERIFIED_NATIVE_GPU_PROOF_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedNativeRuntimeGpuProof:
    suite_id: NativeQualificationSuite
    receipt_sha256: str
    receipt_raw_sha256: str
    runner_protocol_sha256: str
    assignment_sha256: str
    qualification_observation_sha256: str
    junit_xml_sha256: str
    source_capability_sha256: str
    qualification_authority_sha256: str
    source_identity_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    topology_sha256: str
    gpu_uuids: tuple[str, ...]
    backend_capabilities: tuple[str, ...]
    trusted_policy_sha256: str
    challenge_sha256: str
    control_envelope_sha256: str
    challenge_reservation_sha256: str

    def __init__(
        self,
        *,
        receipt: NativeRuntimeGpuProofReceipt,
        receipt_raw_sha256: str,
        trusted_policy_sha256: str,
        challenge_sha256: str,
        control_envelope_sha256: str,
        challenge_reservation_sha256: str,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_NATIVE_GPU_PROOF_SENTINEL:
            raise TypeError(
                "verified native proof can only come from signature verification"
            )
        for name, value in (
            ("suite_id", receipt.suite_id),
            ("receipt_sha256", receipt.sha256),
            ("receipt_raw_sha256", receipt_raw_sha256),
            ("runner_protocol_sha256", receipt.runner_protocol_sha256),
            ("assignment_sha256", receipt.assignment_sha256),
            (
                "qualification_observation_sha256",
                receipt.qualification_observation_sha256,
            ),
            ("junit_xml_sha256", receipt.junit_xml_sha256),
            ("source_capability_sha256", receipt.source_capability_sha256),
            (
                "qualification_authority_sha256",
                receipt.qualification_authority_sha256,
            ),
            ("source_identity_sha256", receipt.source_identity_sha256),
            ("inventory_sha256", receipt.inventory_sha256),
            ("hardware_envelope_sha256", receipt.hardware_envelope_sha256),
            ("topology_mode", receipt.topology_mode),
            ("topology_sha256", receipt.topology_sha256),
            ("gpu_uuids", receipt.gpu_uuids),
            ("backend_capabilities", receipt.backend_capabilities),
            ("trusted_policy_sha256", trusted_policy_sha256),
            ("challenge_sha256", challenge_sha256),
            ("control_envelope_sha256", control_envelope_sha256),
            ("challenge_reservation_sha256", challenge_reservation_sha256),
        ):
            object.__setattr__(self, name, value)
        for label, value in (
            ("native verified policy", self.trusted_policy_sha256),
            ("native verified receipt raw", self.receipt_raw_sha256),
            ("native verified runner protocol", self.runner_protocol_sha256),
            ("native verified assignment", self.assignment_sha256),
            (
                "native verified qualification observation",
                self.qualification_observation_sha256,
            ),
            ("native verified JUnit XML", self.junit_xml_sha256),
            (
                "native verified qualification authority",
                self.qualification_authority_sha256,
            ),
            ("native verified challenge", self.challenge_sha256),
            ("native verified control envelope", self.control_envelope_sha256),
            (
                "native verified challenge reservation",
                self.challenge_reservation_sha256,
            ),
        ):
            _require_sha256(label, value)

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


def require_fixed_address_graph_gpu_proof(
    *,
    claimed_source_capability_sha256: str,
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None,
    expected_source_identity_sha256: str | None,
    expected_inventory_sha256: str,
    expected_gpu_uuids: tuple[str, ...],
) -> VerifiedNativeRuntimeGpuProof:
    """Authorize graph allocation only from the exact sealed GPU proof."""

    if claimed_source_capability_sha256 != NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256:
        raise NativeReadinessBlocked("graph_source_capability_unregistered")
    if expected_source_identity_sha256 is None:
        raise NativeReadinessBlocked("graph_source_identity_unbound")
    _require_sha256("graph expected source identity", expected_source_identity_sha256)
    _require_sha256("graph expected inventory", expected_inventory_sha256)
    if (
        type(expected_gpu_uuids) is not tuple
        or len(expected_gpu_uuids) != 1
        or len(set(expected_gpu_uuids)) != 1
    ):
        raise NativeReadinessBlocked("graph_gpu_assignment_unregistered")
    if (
        type(verified_gpu_proof) is not VerifiedNativeRuntimeGpuProof
        or verified_gpu_proof.suite_id != "native_hot_path_tp1"
        or verified_gpu_proof.topology_mode != "tp1_dp1"
        or verified_gpu_proof.source_capability_sha256
        != NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
        or verified_gpu_proof.source_identity_sha256 != expected_source_identity_sha256
        or verified_gpu_proof.inventory_sha256 != expected_inventory_sha256
        or verified_gpu_proof.gpu_uuids != expected_gpu_uuids
        or "graph_hot_path" not in verified_gpu_proof.backend_capabilities
    ):
        raise NativeReadinessBlocked("graph_hot_path_gpu_proof_unavailable")
    return verified_gpu_proof


def require_chronobelief_gpu_proof(
    *,
    claimed_source_capability_sha256: str,
    claimed_gpu_proof_sha256: str,
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None,
    expected_source_identity_sha256: str | None,
    expected_inventory_sha256: str,
    expected_gpu_uuids: tuple[str, ...],
) -> VerifiedNativeRuntimeGpuProof:
    """Authorize ChronoBelief allocation only from its exact live GPU suite."""

    if claimed_source_capability_sha256 != NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256:
        raise NativeReadinessBlocked("chronobelief_source_capability_unregistered")
    _require_sha256("ChronoBelief claimed GPU proof", claimed_gpu_proof_sha256)
    if expected_source_identity_sha256 is None:
        raise NativeReadinessBlocked("chronobelief_source_identity_unbound")
    _require_sha256(
        "ChronoBelief expected source identity", expected_source_identity_sha256
    )
    _require_sha256("ChronoBelief expected inventory", expected_inventory_sha256)
    if (
        type(expected_gpu_uuids) is not tuple
        or len(expected_gpu_uuids) != 1
        or len(set(expected_gpu_uuids)) != 1
    ):
        raise NativeReadinessBlocked("chronobelief_gpu_assignment_unregistered")
    if (
        type(verified_gpu_proof) is not VerifiedNativeRuntimeGpuProof
        or verified_gpu_proof.suite_id != "chronobelief_gpu_parity"
        or verified_gpu_proof.topology_mode != "tp1_dp1"
        or verified_gpu_proof.source_capability_sha256
        != NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
        or verified_gpu_proof.source_identity_sha256 != expected_source_identity_sha256
        or verified_gpu_proof.inventory_sha256 != expected_inventory_sha256
        or verified_gpu_proof.gpu_uuids != expected_gpu_uuids
        or verified_gpu_proof.receipt_sha256 != claimed_gpu_proof_sha256
        or verified_gpu_proof.backend_capabilities != ("chronobelief_gpu_parity",)
    ):
        raise NativeReadinessBlocked("chronobelief_gpu_parity_proof_unavailable")
    return verified_gpu_proof


def verify_native_runtime_gpu_proof(
    receipt_path: str,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_suite_id: NativeQualificationSuite,
    expected_topology_sha256: str,
    expected_source_identity_sha256: str,
    expected_inventory_sha256: str,
    expected_gpu_uuids: tuple[str, ...],
    expected_hardware_envelope_sha256: str,
    expected_run_nonce_sha256: str,
    now_ns: int,
) -> VerifiedNativeRuntimeGpuProof:
    raw = CanonicalJsonProofBinding.bind(
        receipt_path,
    )
    receipt = NativeRuntimeGpuProofReceipt.from_dict(raw.reopen())
    if receipt.sha256 != raw.semantic_sha256:
        raise ValueError("native proof semantic identity differs from content")
    binding = raw
    if type(control_attestation) is not ControlArtifactAttestation:
        raise TypeError("native runtime proof requires an exact control envelope")
    if type(replay_store) is not ChallengeReplayStore:
        raise TypeError("native runtime proof requires the release replay store")
    if (
        receipt.suite_id != expected_suite_id
        or receipt.topology_sha256 != expected_topology_sha256
        or receipt.run_nonce_sha256 != expected_run_nonce_sha256
        or receipt.source_identity_sha256 != expected_source_identity_sha256
        or receipt.inventory_sha256 != expected_inventory_sha256
        or receipt.gpu_uuids != expected_gpu_uuids
        or receipt.hardware_envelope_sha256 != expected_hardware_envelope_sha256
    ):
        raise ValueError("native runtime GPU proof differs from expected identity")
    subject = control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != binding.raw_sha256
        or subject.protocol_sha256
        != NATIVE_RUNTIME_RELEASE_CAPABILITY.suite_protocol_sha256
        or subject.registry_sha256 != receipt.source_identity_sha256
        or subject.lineage_sha256 != receipt.control_lineage_sha256
    ):
        raise ValueError("native runtime proof control subject is not exact")
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
    return VerifiedNativeRuntimeGpuProof(
        receipt=receipt,
        receipt_raw_sha256=binding.raw_sha256,
        trusted_policy_sha256=verified_control.trusted_attester_policy_sha256,
        challenge_sha256=verified_control.challenge_sha256,
        control_envelope_sha256=verified_control.envelope_sha256,
        challenge_reservation_sha256=reservation_sha256,
        _verification_tag=_VERIFIED_NATIVE_GPU_PROOF_SENTINEL,
    )


@dataclass(frozen=True)
class NativeRuntimeGpuProofArtifact:
    """Durable proof whose signatures and atomic replay record can be reopened."""

    schema_version: Literal[1]
    kind: Literal["lightcone_native_runtime_gpu_proof_artifact"]
    receipt: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    verified_proof_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_native_runtime_gpu_proof_artifact"
        ):
            raise ValueError("native runtime proof artifact schema is unsupported")
        if type(self.receipt) is not CanonicalJsonProofBinding:
            raise TypeError("native proof artifact requires a path-bound raw receipt")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("native proof artifact requires its control envelope")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("native proof artifact requires its replay reservation")
        self.receipt.__post_init__()
        self.control_attestation.__post_init__()
        self.replay_reservation.__post_init__()
        _require_sha256("native artifact verified proof", self.verified_proof_sha256)

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
    def from_dict(cls, value: object) -> NativeRuntimeGpuProofArtifact:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "receipt",
            "control_attestation",
            "replay_reservation",
            "verified_proof_sha256",
        }:
            raise ValueError("native runtime proof artifact fields differ")
        row = dict(value)
        receipt = CanonicalJsonProofBinding.from_dict(row.pop("receipt"))
        control = ControlArtifactAttestation.from_dict(row.pop("control_attestation"))
        reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("replay_reservation")
        )
        return cls(
            receipt=receipt,
            control_attestation=control,
            replay_reservation=reservation,
            **row,
        )

    def revalidate(self, *, now_ns: int) -> VerifiedNativeRuntimeGpuProof:
        """Reopen signatures and the already-committed replay transaction."""

        self.__post_init__()
        reserved = self.replay_reservation.revalidate()
        if type(now_ns) is not int or now_ns < self.replay_reservation.reserved_ns:
            raise ValueError("native proof artifact current time precedes reservation")
        verified_at_ns = self.replay_reservation.reserved_ns
        receipt = NativeRuntimeGpuProofReceipt.from_dict(self.receipt.reopen())
        if receipt.sha256 != self.receipt.semantic_sha256:
            raise ValueError("native proof artifact semantic receipt changed")
        subject = self.control_attestation.subject
        if (
            subject.artifact_type != "non_serving_terminal"
            or subject.artifact_sha256 != self.receipt.raw_sha256
            or subject.protocol_sha256
            != NATIVE_RUNTIME_RELEASE_CAPABILITY.suite_protocol_sha256
            or subject.registry_sha256 != receipt.source_identity_sha256
            or subject.lineage_sha256 != receipt.control_lineage_sha256
        ):
            raise ValueError("native proof artifact control subject is not exact")
        verified_control = verify_release_control_artifact_attestation(
            self.control_attestation,
            expected_inventory_sha256=receipt.inventory_sha256,
            now_ns=verified_at_ns,
            consumed_challenge_sha256s=(),
        )
        bundle = self.control_attestation.deployment_policy_authorization.bundle
        bundle.require_hardware_envelope(receipt.hardware_envelope_sha256)
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
            raise ValueError("native proof artifact replay reservation differs")
        verified = VerifiedNativeRuntimeGpuProof(
            receipt=receipt,
            receipt_raw_sha256=self.receipt.raw_sha256,
            trusted_policy_sha256=(verified_control.trusted_attester_policy_sha256),
            challenge_sha256=verified_control.challenge_sha256,
            control_envelope_sha256=verified_control.envelope_sha256,
            challenge_reservation_sha256=expected_reservation,
            _verification_tag=_VERIFIED_NATIVE_GPU_PROOF_SENTINEL,
        )
        if verified.sha256 != self.verified_proof_sha256:
            raise ValueError("native proof artifact verified identity changed")
        return verified


def build_native_runtime_gpu_proof_artifact(
    *,
    receipt_path: str,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    verified_proof: VerifiedNativeRuntimeGpuProof,
) -> NativeRuntimeGpuProofArtifact:
    """Bind an already-verified proof to its immutable replay transaction."""

    if type(verified_proof) is not VerifiedNativeRuntimeGpuProof:
        raise TypeError("native proof artifact requires an exact verified proof")
    provisional = CanonicalJsonProofBinding.bind(
        receipt_path,
        semantic_sha256=verified_proof.receipt_sha256,
    )
    receipt = NativeRuntimeGpuProofReceipt.from_dict(provisional.reopen())
    if (
        verified_proof.receipt_sha256 != receipt.sha256
        or verified_proof.receipt_raw_sha256 != provisional.raw_sha256
        or verified_proof.challenge_sha256 != control_attestation.challenge.sha256
        or verified_proof.control_envelope_sha256 != control_attestation.sha256
    ):
        raise ValueError("native proof artifact inputs differ from verified proof")
    reservation = replay_store.bind_reservation(
        verified_proof.challenge_reservation_sha256
    )
    return NativeRuntimeGpuProofArtifact(
        schema_version=1,
        kind="lightcone_native_runtime_gpu_proof_artifact",
        receipt=provisional,
        control_attestation=control_attestation,
        replay_reservation=reservation,
        verified_proof_sha256=verified_proof.sha256,
    )


def validate_native_runtime_gpu_proof_artifact(
    artifact_path: str,
    *,
    expected_suite_id: NativeQualificationSuite,
    expected_topology_sha256: str,
    expected_source_identity_sha256: str,
    expected_inventory_sha256: str,
    expected_gpu_uuids: tuple[str, ...],
    expected_hardware_envelope_sha256: str,
    expected_assignment_sha256: str,
    expected_qualification_observation_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> VerifiedNativeRuntimeGpuProof:
    """Deep-open one durable, suite-specific native GPU proof artifact."""

    for label, value in (
        ("native proof expected topology", expected_topology_sha256),
        ("native proof expected source identity", expected_source_identity_sha256),
        ("native proof expected inventory", expected_inventory_sha256),
        ("native proof expected hardware envelope", expected_hardware_envelope_sha256),
        ("native proof expected assignment", expected_assignment_sha256),
        (
            "native proof expected qualification observation",
            expected_qualification_observation_sha256,
        ),
        ("native proof expected root manifest", expected_root_manifest_sha256),
    ):
        _require_sha256(label, value)
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = NativeRuntimeGpuProofArtifact.from_dict(binding.reopen())
    if artifact.sha256 != binding.semantic_sha256:
        raise ValueError("native proof artifact semantic identity changed")
    verified = artifact.revalidate(now_ns=now_ns)
    receipt = NativeRuntimeGpuProofReceipt.from_dict(artifact.receipt.reopen())
    if (
        verified.suite_id != expected_suite_id
        or verified.topology_sha256 != expected_topology_sha256
        or verified.source_identity_sha256 != expected_source_identity_sha256
        or verified.inventory_sha256 != expected_inventory_sha256
        or verified.gpu_uuids != expected_gpu_uuids
        or verified.hardware_envelope_sha256 != expected_hardware_envelope_sha256
        or verified.assignment_sha256 != expected_assignment_sha256
        or verified.qualification_observation_sha256
        != expected_qualification_observation_sha256
        or verified.runner_protocol_sha256
        != NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[expected_suite_id]
        or receipt.sha256 != verified.receipt_sha256
        or artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("native proof artifact differs from expected suite identity")
    return verified


@dataclass(frozen=True)
class NativeItlBufferBinding:
    """One source-owned per-token timestamp buffer and its live pointer."""

    schema_version: int
    run_sha256: str
    producer_sha256: str
    clock_identity: Literal["monotonic_ns"]
    producer_stream_id: str
    buffer_pointer: int
    buffer_capacity_tokens: int
    buffer_generation: int
    native_patch_capability_sha256: str | None = None
    gpu_proof_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("native ITL buffer schema is unsupported")
        _require_sha256("native ITL run", self.run_sha256)
        _require_sha256("native ITL producer", self.producer_sha256)
        if self.clock_identity != "monotonic_ns":
            raise ValueError("native ITL timestamps require monotonic_ns")
        _require_text("native ITL producer stream", self.producer_stream_id)
        if type(self.buffer_pointer) is not int or self.buffer_pointer <= 0:
            raise ValueError("native ITL buffer pointer must be positive")
        if (
            type(self.buffer_capacity_tokens) is not int
            or self.buffer_capacity_tokens < 2
        ):
            raise ValueError("native ITL buffer must hold at least two tokens")
        _require_nonnegative_int("native ITL buffer generation", self.buffer_generation)
        proof_fields = (
            self.native_patch_capability_sha256,
            self.gpu_proof_sha256,
        )
        if any(value is None for value in proof_fields) and any(
            value is not None for value in proof_fields
        ):
            raise ValueError("native ITL patch capability and GPU proof are atomic")
        for label, value in (
            ("native ITL patch capability", self.native_patch_capability_sha256),
            ("native ITL GPU proof", self.gpu_proof_sha256),
        ):
            if value is not None:
                _require_sha256(label, value)

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "schema_version": self.schema_version,
                "run_sha256": self.run_sha256,
                "producer_sha256": self.producer_sha256,
                "clock_identity": self.clock_identity,
                "producer_stream_id": self.producer_stream_id,
                "buffer_pointer": self.buffer_pointer,
                "buffer_capacity_tokens": self.buffer_capacity_tokens,
                "buffer_generation": self.buffer_generation,
                "native_patch_capability_sha256": (self.native_patch_capability_sha256),
                "gpu_proof_sha256": self.gpu_proof_sha256,
            }
        )


@dataclass(frozen=True)
class NativeTokenTimestampEvent:
    binding_sha256: str
    request_id: str
    token_index: int
    token_id: int
    observed_ns: int
    buffer_pointer: int
    buffer_generation: int

    def __post_init__(self) -> None:
        _require_sha256("native ITL event binding", self.binding_sha256)
        _require_text("native ITL event request", self.request_id)
        for label, value in (
            ("native ITL token index", self.token_index),
            ("native ITL token ID", self.token_id),
            ("native ITL observation", self.observed_ns),
            ("native ITL buffer generation", self.buffer_generation),
        ):
            _require_nonnegative_int(label, value)
        if type(self.buffer_pointer) is not int or self.buffer_pointer <= 0:
            raise ValueError("native ITL event buffer pointer must be positive")

    @property
    def sha256(self) -> str:
        return _sha256(self.__dict__)


@dataclass(frozen=True)
class NativeItlResultPointer:
    """Content-bound diagnostic result; never promoted by self-described proof."""

    schema_version: int
    binding_sha256: str
    run_sha256: str
    request_id: str
    output_token_ids: tuple[int, ...]
    token_observed_ns: tuple[int, ...]
    events_sha256: str
    evidence_level: Literal["CPU_CONTRACT_ONLY"] = "CPU_CONTRACT_ONLY"
    formal_authorized: Literal[False] = False
    reason_code: Literal["native_itl_gpu_pointer_proof_unavailable"] = (
        "native_itl_gpu_pointer_proof_unavailable"
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("native ITL result pointer schema is unsupported")
        for label, value in (
            ("native ITL result binding", self.binding_sha256),
            ("native ITL result run", self.run_sha256),
            ("native ITL result events", self.events_sha256),
        ):
            _require_sha256(label, value)
        _require_text("native ITL result request", self.request_id)
        if len(self.output_token_ids) < 2 or len(self.output_token_ids) != len(
            self.token_observed_ns
        ):
            raise ValueError("native ITL result lacks full per-token coverage")
        if any(
            current <= previous
            for previous, current in zip(
                self.token_observed_ns,
                self.token_observed_ns[1:],
                strict=False,
            )
        ):
            raise ValueError("native ITL result timestamps must increase")
        if self.evidence_level != "CPU_CONTRACT_ONLY" or self.formal_authorized:
            raise ValueError("native ITL result cannot self-authorize formal evidence")

    @property
    def sha256(self) -> str:
        return _sha256(self.__dict__)


class NativeItlTimestampStateMachine:
    """Validate ordered native token writes against one fixed buffer pointer."""

    def __init__(
        self,
        binding: NativeItlBufferBinding,
        *,
        request_id: str,
        output_token_ids: Sequence[int],
        request_started_ns: int,
    ) -> None:
        if type(binding) is not NativeItlBufferBinding:
            raise TypeError("native ITL recorder requires an exact buffer binding")
        binding.__post_init__()
        _require_text("native ITL request", request_id)
        tokens = tuple(output_token_ids)
        if (
            len(tokens) < 2
            or len(tokens) > binding.buffer_capacity_tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("native ITL output-token contract is invalid")
        _require_nonnegative_int("native ITL request start", request_started_ns)
        self.binding = binding
        self.request_id = request_id
        self.output_token_ids = tokens
        self.request_started_ns = request_started_ns
        self._events: list[NativeTokenTimestampEvent] = []
        self._finalized = False

    def record(self, event: NativeTokenTimestampEvent) -> None:
        if self._finalized:
            raise RuntimeError("native ITL result is already finalized")
        if type(event) is not NativeTokenTimestampEvent:
            raise TypeError("native ITL recorder requires exact token events")
        expected_index = len(self._events)
        if expected_index >= len(self.output_token_ids):
            raise ValueError("native ITL event exceeds expected token coverage")
        if (
            event.binding_sha256 != self.binding.sha256
            or event.request_id != self.request_id
            or event.token_index != expected_index
            or event.token_id != self.output_token_ids[expected_index]
        ):
            raise ValueError("native ITL event differs from ordered token identity")
        if (
            event.buffer_pointer != self.binding.buffer_pointer
            or event.buffer_generation != self.binding.buffer_generation
        ):
            raise ValueError("native ITL timestamp buffer pointer changed")
        previous_ns = (
            self.request_started_ns
            if not self._events
            else self._events[-1].observed_ns
        )
        if event.observed_ns <= previous_ns:
            raise ValueError("native ITL event timestamps must strictly increase")
        self._events.append(event)

    def finalize(self, *, request_terminal_ns: int) -> NativeItlResultPointer:
        if self._finalized:
            raise RuntimeError("native ITL result is already finalized")
        _require_nonnegative_int("native ITL request terminal", request_terminal_ns)
        if len(self._events) != len(self.output_token_ids):
            raise ValueError("native ITL result lacks full per-token coverage")
        if request_terminal_ns < self._events[-1].observed_ns:
            raise ValueError("native ITL terminal precedes the last token")
        self._finalized = True
        events_sha256 = _sha256(tuple(event.sha256 for event in self._events))
        return NativeItlResultPointer(
            schema_version=1,
            binding_sha256=self.binding.sha256,
            run_sha256=self.binding.run_sha256,
            request_id=self.request_id,
            output_token_ids=self.output_token_ids,
            token_observed_ns=tuple(event.observed_ns for event in self._events),
            events_sha256=events_sha256,
        )

    def require_formal_authority(
        self,
        verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    ) -> None:
        if (
            type(verified_gpu_proof) is not VerifiedNativeRuntimeGpuProof
            or "native_itl" not in verified_gpu_proof.backend_capabilities
            or self.binding.gpu_proof_sha256 != verified_gpu_proof.receipt_sha256
            or self.binding.native_patch_capability_sha256
            != verified_gpu_proof.source_capability_sha256
        ):
            raise NativeReadinessBlocked("native_itl_gpu_pointer_proof_unavailable")


@dataclass(frozen=True)
class GraphTensorBinding:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    data_pointer: int

    def __post_init__(self) -> None:
        _require_text("graph tensor name", self.name)
        if not self.shape or any(
            type(value) is not int or value < 1 for value in self.shape
        ):
            raise ValueError("graph tensor shape must be non-empty and positive")
        _require_text("graph tensor dtype", self.dtype)
        _require_text("graph tensor device", self.device)
        if type(self.data_pointer) is not int or self.data_pointer <= 0:
            raise ValueError("graph tensor data pointer must be positive")

    @classmethod
    def from_tensor(cls, name: str, tensor: Tensor) -> GraphTensorBinding:
        if not isinstance(tensor, Tensor) or tensor.numel() < 1:
            raise ValueError("graph binding requires one non-empty tensor")
        return cls(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            data_pointer=tensor.data_ptr(),
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.__dict__)


@dataclass(frozen=True)
class GraphFixedAddressContract:
    schema_version: int
    graph_identity_sha256: str
    capture_generation: int
    tensors: tuple[GraphTensorBinding, ...]
    allowed_operations: tuple[HotPathOperation, ...] = _ALLOWED_HOT_PATH_OPERATIONS
    fixed_addresses_required: Literal[True] = True
    blocking_d2h_forbidden: Literal[True] = True
    host_synchronization_forbidden: Literal[True] = True
    gpu_proof_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("graph fixed-address contract schema is unsupported")
        _require_sha256("graph identity", self.graph_identity_sha256)
        _require_nonnegative_int("graph capture generation", self.capture_generation)
        if (
            type(self.tensors) is not tuple
            or not self.tensors
            or any(type(value) is not GraphTensorBinding for value in self.tensors)
            or tuple(value.name for value in self.tensors)
            != tuple(sorted({value.name for value in self.tensors}))
        ):
            raise ValueError("graph tensor bindings must be sorted, unique, and exact")
        if self.allowed_operations != _ALLOWED_HOT_PATH_OPERATIONS:
            raise ValueError("graph hot-path operation allowlist is not registered")
        if not (
            self.fixed_addresses_required
            and self.blocking_d2h_forbidden
            and self.host_synchronization_forbidden
        ):
            raise ValueError("graph contract cannot weaken hot-path invariants")
        if self.gpu_proof_sha256 is not None:
            _require_sha256("graph hot-path GPU proof", self.gpu_proof_sha256)

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "schema_version": self.schema_version,
                "graph_identity_sha256": self.graph_identity_sha256,
                "capture_generation": self.capture_generation,
                "tensors": tuple(value.sha256 for value in self.tensors),
                "allowed_operations": self.allowed_operations,
                "fixed_addresses_required": self.fixed_addresses_required,
                "blocking_d2h_forbidden": self.blocking_d2h_forbidden,
                "host_synchronization_forbidden": (self.host_synchronization_forbidden),
                "gpu_proof_sha256": self.gpu_proof_sha256,
            }
        )

    def validate_tensors(self, tensors: Sequence[tuple[str, Tensor]]) -> None:
        observed = tuple(
            GraphTensorBinding.from_tensor(name, tensor) for name, tensor in tensors
        )
        if observed != self.tensors:
            raise RuntimeError("graph fixed-address tensor identity changed")


@dataclass(frozen=True)
class GraphHotPathReceipt:
    contract_sha256: str
    operations: tuple[HotPathOperation, ...]
    evidence_level: Literal["CPU_CONTRACT_ONLY"] = "CPU_CONTRACT_ONLY"
    formal_authorized: Literal[False] = False
    reason_code: Literal["graph_hot_path_gpu_proof_unavailable"] = (
        "graph_hot_path_gpu_proof_unavailable"
    )

    def __post_init__(self) -> None:
        _require_sha256("graph hot-path contract", self.contract_sha256)
        if (
            "device_to_device_copy" not in self.operations
            or "graph_replay" not in self.operations
            or any(
                value not in _ALLOWED_HOT_PATH_OPERATIONS for value in self.operations
            )
        ):
            raise ValueError("graph hot-path receipt lacks required safe operations")
        if self.evidence_level != "CPU_CONTRACT_ONLY" or self.formal_authorized:
            raise ValueError("graph hot-path receipt cannot self-authorize GPU proof")

    @property
    def sha256(self) -> str:
        return _sha256(self.__dict__)


class GraphHotPathStateMachine:
    """Reject host synchronization/D2H and pointer drift in one graph epoch."""

    def __init__(self, contract: GraphFixedAddressContract) -> None:
        if type(contract) is not GraphFixedAddressContract:
            raise TypeError("graph state machine requires an exact contract")
        contract.__post_init__()
        self.contract = contract
        self._operations: list[HotPathOperation] = []
        self._finalized = False

    def observe(
        self,
        operation: HotPathOperation,
        *,
        tensors: Sequence[tuple[str, Tensor]],
    ) -> None:
        if self._finalized:
            raise RuntimeError("graph hot-path receipt is already finalized")
        if operation not in self.contract.allowed_operations:
            raise NativeReadinessBlocked(f"forbidden_hot_path_operation:{operation}")
        self.contract.validate_tensors(tensors)
        self._operations.append(operation)

    def finalize(self) -> GraphHotPathReceipt:
        if self._finalized:
            raise RuntimeError("graph hot-path receipt is already finalized")
        self._finalized = True
        return GraphHotPathReceipt(
            contract_sha256=self.contract.sha256,
            operations=tuple(self._operations),
        )

    def require_formal_authority(
        self,
        verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    ) -> None:
        if (
            type(verified_gpu_proof) is not VerifiedNativeRuntimeGpuProof
            or "graph_hot_path" not in verified_gpu_proof.backend_capabilities
            or self.contract.gpu_proof_sha256 != verified_gpu_proof.receipt_sha256
        ):
            raise NativeReadinessBlocked("graph_hot_path_gpu_proof_unavailable")


__all__ = [
    "NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256",
    "NATIVE_RUNTIME_QUALIFICATION_PROTOCOL_SHA256",
    "NATIVE_RUNTIME_QUALIFICATION_RUNNER_SHA256",
    "NATIVE_RUNTIME_QUALIFICATION_TESTS",
    "NATIVE_RUNTIME_QUALIFICATION_TEST_SET_SHA256",
    "NATIVE_RUNTIME_RELEASE_CAPABILITY",
    "NATIVE_RUNTIME_SUITE_CAPABILITIES",
    "NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S",
    "GraphFixedAddressContract",
    "GraphHotPathReceipt",
    "GraphHotPathStateMachine",
    "GraphTensorBinding",
    "HotPathOperation",
    "NativeItlBufferBinding",
    "NativeItlResultPointer",
    "NativeItlTimestampStateMachine",
    "NativeReadinessBlocked",
    "NativeRuntimeGpuProofArtifact",
    "NativeRuntimeGpuProofReceipt",
    "NativeRuntimeReleaseCapability",
    "NativeTokenTimestampEvent",
    "VerifiedNativeRuntimeGpuProof",
    "build_native_runtime_gpu_proof_artifact",
    "require_chronobelief_gpu_proof",
    "require_fixed_address_graph_gpu_proof",
    "validate_native_runtime_gpu_proof_artifact",
    "verify_native_runtime_gpu_proof",
]
