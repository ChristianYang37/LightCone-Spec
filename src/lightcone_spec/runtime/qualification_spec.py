"""Code-owned cross-module runtime qualification universe.

This module is deliberately dependency-light.  ``readiness`` and
``distributed`` both consume it, so neither module can silently define a
different TP2/DP2 suite identity.  EAGLE3 is present in the immutable source
universe, but its execution is conditional on the root-verified, exact
108-decision E0 selector reduction; a caller boolean is never an authority.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_formal_runtime_qualification_authority_sha256(
    *,
    native_runtime_release_capability_sha256: str,
    qualification_protocol_sha256: str,
    qualification_runner_sha256: str,
    qualification_test_set_sha256: str,
    patched_sglang_tree: str,
    patch_manifest_sha256: str,
) -> str:
    """Build the one stable authority shared by lock and proof validators."""

    for label, value, length in (
        (
            "native runtime release capability",
            native_runtime_release_capability_sha256,
            64,
        ),
        ("qualification protocol", qualification_protocol_sha256, 64),
        ("qualification runner", qualification_runner_sha256, 64),
        ("qualification test set", qualification_test_set_sha256, 64),
        ("patched SGLang tree", patched_sglang_tree, 40),
        ("patch manifest", patch_manifest_sha256, 64),
    ):
        if (
            type(value) is not str
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} is not an exact lower-case digest")
    return _sha256(
        {
            "schema_version": 2,
            "kind": "formal_runtime_qualification_stable_authority",
            "native_runtime_release_capability_sha256": (
                native_runtime_release_capability_sha256
            ),
            "qualification_protocol_sha256": qualification_protocol_sha256,
            "qualification_runner_sha256": qualification_runner_sha256,
            "qualification_test_set_sha256": qualification_test_set_sha256,
            "patched_sglang_tree": patched_sglang_tree,
            "patch_manifest_sha256": patch_manifest_sha256,
            "required_core_suites": FORMAL_RUNTIME_QUALIFICATION_CORE_SUITES,
            "optional_suites": FORMAL_RUNTIME_QUALIFICATION_OPTIONAL_SUITES,
            "eagle3_resolution": FORMAL_RUNTIME_QUALIFICATION_EAGLE_RESOLUTION,
        }
    )


FORMAL_RUNTIME_QUALIFICATION_CORE_SUITES = (
    "chronobelief_gpu_parity",
    "dspark_tp1",
    "dspark_tp2",
    "dspark_dp2",
    "native_hot_path_tp1",
    "nextn_tp1",
    "nextn_tp2",
    "session_reset_tp1",
    "tp1_dp2",
    "tp2_dp1",
)
FORMAL_RUNTIME_QUALIFICATION_OPTIONAL_SUITES = ("eagle3_tp1",)
FORMAL_RUNTIME_QUALIFICATION_EAGLE_RESOLUTION = (
    "root_verified_prepared_eagle3_selector_exact_108_e0_decisions_any_"
    "compatible_requires_suite_all_na_forbids_launch"
)

DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS = MappingProxyType(
    {
        "tp2_dp1": (
            "tp2_nccl_rank_coverage",
            "tp2_sharded_candidate_parity",
            "tp2_all_rank_prepare",
            "tp2_two_phase_commit",
            "tp2_one_rank_abort_zero_partial",
            "tp2_rank_terminal_evidence",
            "native_itl_pointer",
            "graph_fixed_address_no_host_sync",
        ),
        "tp1_dp2": (
            "dp2_two_worker_launch",
            "dp2_sticky_routing",
            "dp2_replica_state_isolation",
            "dp2_zero_cross_replica_gradient",
            "dp2_failure_isolation",
            "dp2_rank_terminal_evidence",
            "native_itl_pointer",
            "graph_fixed_address_no_host_sync",
        ),
    }
)

DISTRIBUTED_RUNTIME_GPU_PROOF_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 1,
        "kind": "distributed_runtime_gpu_proof",
        "base_smoke": "exact_8_pass_zero_skip",
        "qualification_tests": dict(DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS),
        "trust": "root_authorized_dynamic_policy_plus_atomic_replay_store",
    }
)

DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S = MappingProxyType(
    {
        topology_mode: _sha256(
            {
                "schema_version": 1,
                "kind": ("source_owned_distributed_runtime_gpu_qualification_runner"),
                "topology_mode": topology_mode,
                "test_names": test_names,
                "base_smoke": "path_bound_exactness_8_pass_zero_skip",
                "gpu_process_gate": "pre_and_post_empty_exact_two_uuid",
                "live_observation": "path_bound_all_rank_worker_terminal",
                "remote_private_key": False,
                "trust_lift": "local_external_control_atomic_replay",
            }
        )
        for topology_mode, test_names in (
            DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS.items()
        )
    }
)


__all__ = [
    "DISTRIBUTED_RUNTIME_GPU_PROOF_PROTOCOL_SHA256",
    "DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS",
    "DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S",
    "FORMAL_RUNTIME_QUALIFICATION_CORE_SUITES",
    "FORMAL_RUNTIME_QUALIFICATION_EAGLE_RESOLUTION",
    "FORMAL_RUNTIME_QUALIFICATION_OPTIONAL_SUITES",
    "build_formal_runtime_qualification_authority_sha256",
]
