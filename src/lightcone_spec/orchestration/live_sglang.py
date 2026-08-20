"""Source-owned pinned-SGLang serving runner for unsigned GPU evidence.

The generic native-terminal collector deliberately remains useful for CPU
protocol tests, so it accepts a request callback.  Formal serving must never
accept that boundary.  This module owns the concrete process, verified
``CompileLaunchManifest``, official patched SGLang HTTP pool, request
scheduling, cleanup, and an independently reopenable live-run receipt.

Nothing emitted here is formal authority.  The GPU host has no release key;
the raw terminal, native timestamp pointers, and live-run receipt must all be
pulled and covered by a later local external-control attestation.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import http.client
import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config import RunConfig, load_run_config
from lightcone_spec.orchestration.formal_terminal_shards import (
    publish_scalable_client_request_lifecycle,
    reopen_scalable_client_request_lifecycle,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalResultProjection,
    NativeTerminalResultProofArtifact,
    NativeTerminalRunBinding,
    TerminalRequestExpectation,
    UnsignedNativeLifecycleEvents,
    UnsignedNativeServingPhaseResult,
    UnsignedNativeTerminalCollection,
    ValidatedNativeTerminalEvidence,
    canonical_sha256,
    collect_unsigned_native_terminal_artifact,
    validate_native_terminal_artifact,
    validate_native_terminal_result_proof_artifact,
    validate_unsigned_native_itl_pointer_bundle,
)
from lightcone_spec.runtime.attestation import NO_TRUSTED_ATTESTERS
from lightcone_spec.runtime.backend import (
    VerifiedEagle3E0ExecutionAuthority,
    require_eagle3_e0_execution_authority,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    VerifiedNativeRuntimeGpuProof,
    require_chronobelief_gpu_proof,
    require_fixed_address_graph_gpu_proof,
)

if TYPE_CHECKING:
    from lightcone_spec.experiments.formal_single_operator_chronobelief import (
        TrustedSingleOperatorChronoBeliefGpuParityProof,
    )
    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        TrustedSingleOperatorEagle3ExecutionAuthority,
    )
    from lightcone_spec.experiments.serving import (
        BenchServingResult,
        BoundServingRequest,
        PinnedBenchServingTransport,
    )
    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
    )
else:
    # Importing ``lightcone_spec.experiments.serving`` executes the experiments
    # package initializer, whose staged reducers reach the physical dispatcher
    # and then this module.  Resolve these concrete runtime classes only after
    # module initialization; tests may still replace the public globals.
    BenchServingResult = None
    BoundServingRequest = None
    PinnedBenchServingTransport = None


def _serving_runtime_types() -> tuple[type, type]:
    global BoundServingRequest, PinnedBenchServingTransport
    if BoundServingRequest is None or PinnedBenchServingTransport is None:
        from lightcone_spec.experiments.serving import (
            BoundServingRequest as ServingRequest,
        )
        from lightcone_spec.experiments.serving import (
            PinnedBenchServingTransport as ServingTransport,
        )

        if BoundServingRequest is None:
            BoundServingRequest = ServingRequest
        if PinnedBenchServingTransport is None:
            PinnedBenchServingTransport = ServingTransport
    return BoundServingRequest, PinnedBenchServingTransport


PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256 = canonical_sha256(
    {
        "schema_version": 1,
        "kind": "source_owned_pinned_sglang_unsigned_serving",
        "launch": (
            "deep_reopened_compile_launch_manifest_exact_argv_checkout_tree_"
            "environment_gpu_assignment_budget_inventory"
        ),
        "transport": (
            "exact_PinnedBenchServingTransport.from_checkout_same_http_pool_for_"
            "generate_abort_and_native_admin"
        ),
        "request_executor": (
            "internal_only_arrival_ordered_bounded_concurrency_exact_observed_tokens"
        ),
        "outputs": [
            "unsigned_native_terminal",
            "unsigned_native_itl_pointer_bundle",
            "unsigned_pinned_sglang_serving_run_receipt",
            "unsigned_pinned_sglang_concurrent_group_receipt",
            "path_bound_gpu_process_snapshots",
            "raw_server_log",
        ],
        "concurrent_group": (
            "exact_two_servers_ready_before_one_shared_monotonic_scored_barrier_"
            "distinct_gpu_uuid_and_port_actual_interval_overlap"
        ),
        "cleanup": "bounded_process_group_sigterm_then_verify_empty",
        "chronobelief_authority": (
            "legacy_verified_native_GPU_proof_or_distinct_path_bound_trusted_"
            "single_operator_empirical_proof_with_qualified_GPU_membership"
        ),
        "remote_private_key": False,
        "formal_execution_authorized": False,
        "callback_or_transport_injection": False,
    }
)

PINNED_SGLANG_REGISTERED_SERVING_PROTOCOL_SHA256 = canonical_sha256(
    {
        "schema_version": 2,
        "kind": "source_owned_pinned_sglang_unsigned_serving",
        "base_protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
        "admission": (
            "source_owned_arrival_deadline_drain_and_max_concurrency_with_"
            "semaphore_wait_inside_the_offered_request_deadline"
        ),
        "client_terminal_partition": [
            "completed",
            "rejected",
            "timed_out",
            "cancelled",
            "unfinished",
        ],
        "native_terminal_partition": ["completed", "aborted", "not_submitted"],
        "closed_loop": (
            "one_zero_think_stream_per_lane_with_registered_window_or_exact_"
            "p99_extension_pool_completion"
        ),
        "warmup": "separate_excluded_phase_requires_exact_completion",
        "negative_outcomes": "scientific_data_not_phase_cancellation",
        "client_lifecycle": "bounded_sharded_path_bound_artifact",
        "formal_execution_authorized": False,
    }
)

PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256 = canonical_sha256(
    {
        "schema_version": 1,
        "kind": "unsigned_pinned_sglang_lifecycle_timing",
        "clock": "time.monotonic_ns",
        "phases": [
            "startup",
            "warmup",
            "adaptation_reset",
            "scored_arrival_and_window",
            "drain",
            "process_cleanup",
            "evidence_flush",
        ],
        "attempt": "exact_run_binding_attempt_id",
        "profile": "derived_from_RunConfig_telemetry_detail",
        "durations": "derived_integer_ns_only",
        "formal_execution_authorized": False,
    }
)

_MAX_REQUESTS_PER_PHASE = 100_000
_MAX_RUN_TIMEOUT_SECONDS = 3_600.0
_MAX_REGISTERED_RUN_TIMEOUT_SECONDS = 60.0 * 24.0 * 60.0 * 60.0
_SERVER_READY_TIMEOUT_SECONDS = 600.0
_ABORT_TIMEOUT_SECONDS = 30.0
_PROCESS_GROUP_CLEANUP_SECONDS = 120.0
_GPU_SNAPSHOT_TIMEOUT_SECONDS = 30.0


class PinnedSglangServingRunError(RuntimeError):
    """Stable fail-closed reason from the source-owned live serving boundary."""

    def __init__(
        self,
        reason_code: str,
        *,
        fatal_pointer: CanonicalJsonProofBinding | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.fatal_pointer = fatal_pointer
        super().__init__(f"pinned SGLang unsigned serving failed: {reason_code}")


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _adaptation_mechanism_server_fields(config: RunConfig) -> dict[str, object]:
    return {
        "lightcone_adaptation_mechanism_enabled": config.adaptation is not None,
        "lightcone_adaptation_microbatch_size": (
            config.runtime.adaptation_microbatch_size
        ),
        "lightcone_adaptation_publication_coalescing": (
            config.runtime.adaptation_publication_coalescing
        ),
        "lightcone_adaptation_stream_priority": (
            config.runtime.adaptation_stream_priority
        ),
    }


def _mechanism_execution_authority(
    *, config: RunConfig, server_argv: tuple[str, ...]
) -> dict[str, object]:
    """Bind adapted worker knobs to RunConfig before process allocation."""

    expected = {
        "--lightcone-adaptation-microbatch-size": str(
            config.runtime.adaptation_microbatch_size
        ),
        "--lightcone-adaptation-publication-coalescing": str(
            config.runtime.adaptation_publication_coalescing
        ),
        "--lightcone-adaptation-stream-priority": (
            config.runtime.adaptation_stream_priority
        ),
    }
    present = {
        flag: tuple(
            index for index, argument in enumerate(server_argv) if argument == flag
        )
        for flag in expected
    }
    if config.adaptation is None:
        if any(present.values()) or any(
            argument.startswith(f"{flag}=")
            for flag in expected
            for argument in server_argv
        ):
            raise ValueError(
                "allocation-free live serving forbids adaptation mechanism argv"
            )
    else:
        for flag, value in expected.items():
            positions = present[flag]
            if (
                len(positions) != 1
                or positions[0] + 1 >= len(server_argv)
                or server_argv[positions[0] + 1] != value
                or any(argument.startswith(f"{flag}=") for argument in server_argv)
            ):
                raise ValueError(
                    "live serving adaptation mechanism argv differs from RunConfig"
                )
    return _adaptation_mechanism_server_fields(config)


def _graph_execution_authority(
    *,
    runtime: object,
    server_argv: tuple[str, ...],
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None,
    expected_source_identity_sha256: str | None,
    inventory_sha256: str,
    gpu_uuids: tuple[str, ...],
) -> tuple[str, str, str | None, str | None]:
    """Validate config/argv/proof before any server or GPU allocation."""

    mode = getattr(runtime, "cuda_graph_mode", None)
    execution_policy_sha256 = getattr(runtime, "execution_policy_sha256", None)
    _require_sha256("live serving execution policy", execution_policy_sha256)
    graph_switch = "--lightcone-fixed-address-publication-graph"
    graph_batches = "--lightcone-graph-batch-sizes"
    graph_no_fallback = "--lightcone-disable-graph-eager-fallback"
    graph_flags = {graph_switch, graph_batches, graph_no_fallback}
    if mode == "disabled":
        if (
            verified_gpu_proof is not None
            or expected_source_identity_sha256 is not None
        ):
            raise ValueError("disabled graph execution cannot consume graph authority")
        if server_argv.count("--disable-cuda-graph") != 1 or any(
            argument in graph_flags for argument in server_argv
        ):
            raise ValueError("disabled graph execution argv differs")
        return execution_policy_sha256, mode, None, None
    if mode != "fixed_address_publication_v1":
        raise ValueError("live serving CUDA graph mode is unsupported")
    try:
        batches_index = server_argv.index(graph_batches)
        batches_value = server_argv[batches_index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("fixed-address graph batch argv is incomplete") from error
    if (
        "--disable-cuda-graph" in server_argv
        or server_argv.count(graph_switch) != 1
        or server_argv.count(graph_batches) != 1
        or batches_value != "1"
        or server_argv.count(graph_no_fallback) != 1
    ):
        raise ValueError("fixed-address graph execution argv differs")
    source_capability_sha256 = getattr(
        runtime, "native_graph_release_capability_sha256", None
    )
    if type(source_capability_sha256) is not str:
        raise ValueError("fixed-address graph runtime lost source capability")
    proof = require_fixed_address_graph_gpu_proof(
        claimed_source_capability_sha256=source_capability_sha256,
        verified_gpu_proof=verified_gpu_proof,
        expected_source_identity_sha256=expected_source_identity_sha256,
        expected_inventory_sha256=inventory_sha256,
        expected_gpu_uuids=gpu_uuids,
    )
    return (
        execution_policy_sha256,
        mode,
        proof.sha256,
        expected_source_identity_sha256,
    )


def _eagle3_execution_authority(
    *,
    config: RunConfig,
    verified_authority: VerifiedEagle3E0ExecutionAuthority | None,
    expected_source_identity_sha256: str | None,
    inventory_sha256: str,
    gpu_uuids: tuple[str, ...],
    trusted_single_operator_authority: (
        TrustedSingleOperatorEagle3ExecutionAuthority | None
    ) = None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Bind adaptive EAGLE3 to its exact E0 compatibility and GPU proofs."""

    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        TrustedSingleOperatorEagle3ExecutionAuthority,
    )

    adaptation = getattr(config, "adaptation", None)
    is_adaptive_eagle3 = adaptation is not None and (
        getattr(config.model, "algorithm", None) == "EAGLE3"
    )
    if not is_adaptive_eagle3:
        if (
            verified_authority is not None
            or expected_source_identity_sha256 is not None
            or trusted_single_operator_authority is not None
        ):
            raise ValueError("non-adaptive EAGLE3 path cannot consume E0 authority")
        if adaptation is not None and any(
            value is not None
            for value in (
                getattr(adaptation, "eagle3_e0_execution_authority_sha256", None),
                getattr(adaptation, "eagle3_compatibility_authority_sha256", None),
                getattr(adaptation, "eagle3_model_selector_sha256", None),
                getattr(adaptation, "eagle3_native_gpu_proof_sha256", None),
            )
        ):
            raise ValueError("non-EAGLE3 adaptation cannot carry E0 authority")
        return None, None, None, None, None
    if trusted_single_operator_authority is not None:
        if (
            verified_authority is not None
            or expected_source_identity_sha256 is not None
            or type(trusted_single_operator_authority)
            is not TrustedSingleOperatorEagle3ExecutionAuthority
        ):
            raise ValueError("trusted and legacy EAGLE3 authorities cannot be mixed")
        # Re-run every path join immediately before the server process is
        # allocated; a previously constructed Python object is not a bearer
        # capability if any source path has changed.
        trusted_single_operator_authority.__post_init__()
        trusted = trusted_single_operator_authority
        if (
            trusted.trust_mode != "trusted_single_operator_empirical_no_signature"
            or trusted.formal_measured_authorization is not False
            or trusted.method != config.method
            or trusted.target_revision != config.model.target_revision
            or trusted.drafter_revision != config.model.drafter_revision
            or trusted.inventory_sha256 != inventory_sha256
            or trusted.gpu_uuids != gpu_uuids
            or adaptation.eagle3_e0_execution_authority_sha256
            != trusted.eagle3_e0_execution_authority_sha256
            or adaptation.eagle3_compatibility_authority_sha256
            != trusted.eagle3_compatibility_authority_sha256
            or adaptation.eagle3_model_selector_sha256
            != trusted.eagle3_model_selector_sha256
            or adaptation.eagle3_native_gpu_proof_sha256
            != trusted.eagle3_native_gpu_proof_sha256
        ):
            raise ValueError("trusted EAGLE3 authority differs from live launch")
        return (
            trusted.eagle3_e0_execution_authority_sha256,
            trusted.eagle3_compatibility_authority_sha256,
            trusted.eagle3_model_selector_sha256,
            trusted.eagle3_native_gpu_proof_sha256,
            trusted.native_source_identity_sha256,
        )
    authority = require_eagle3_e0_execution_authority(
        claimed_execution_authority_sha256=(
            adaptation.eagle3_e0_execution_authority_sha256
        ),
        claimed_compatibility_authority_sha256=(
            adaptation.eagle3_compatibility_authority_sha256
        ),
        claimed_model_selector_sha256=adaptation.eagle3_model_selector_sha256,
        claimed_native_gpu_proof_sha256=adaptation.eagle3_native_gpu_proof_sha256,
        verified_execution_authority=verified_authority,
        expected_method=config.method,
        expected_target_revision=config.model.target_revision,
        expected_drafter_revision=config.model.drafter_revision,
        expected_source_identity_sha256=expected_source_identity_sha256,
        expected_inventory_sha256=inventory_sha256,
        expected_gpu_uuids=gpu_uuids,
    )
    return (
        authority.sha256,
        authority.compatibility_authority_sha256,
        authority.model_selector_sha256,
        authority.native_gpu_receipt_sha256,
        authority.native_source_identity_sha256,
    )


def _chronobelief_execution_authority(
    *,
    config: RunConfig,
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None,
    expected_source_identity_sha256: str | None,
    inventory_sha256: str,
    gpu_uuids: tuple[str, ...],
    trusted_single_operator_proof: (
        TrustedSingleOperatorChronoBeliefGpuParityProof | None
    ) = None,
) -> tuple[str | None, str | None]:
    """Require the independent mixed-precision ChronoBelief GPU suite."""

    adaptation = getattr(config, "adaptation", None)
    optimizer = None if adaptation is None else getattr(adaptation, "optimizer", None)
    is_chronobelief = optimizer is not None and (
        getattr(optimizer, "name", None) == "chronobelief"
    )
    if not is_chronobelief:
        if (
            verified_gpu_proof is not None
            or expected_source_identity_sha256 is not None
            or trusted_single_operator_proof is not None
        ):
            raise ValueError(
                "non-ChronoBelief path cannot consume Chrono GPU authority"
            )
        return None, None
    if trusted_single_operator_proof is not None:
        from lightcone_spec.experiments.formal_single_operator_chronobelief import (
            TrustedSingleOperatorChronoBeliefGpuParityProof,
        )

        if (
            verified_gpu_proof is not None
            or expected_source_identity_sha256 is not None
            or type(trusted_single_operator_proof)
            is not TrustedSingleOperatorChronoBeliefGpuParityProof
            or adaptation.chronobelief_gpu_proof_sha256
            != trusted_single_operator_proof.sha256
            or trusted_single_operator_proof.inventory_sha256 != inventory_sha256
            or len(gpu_uuids) != 1
            or gpu_uuids[0] not in trusted_single_operator_proof.qualified_gpu_uuids
            or trusted_single_operator_proof.trust_mode
            != "trusted_single_operator_empirical_no_signature"
            or trusted_single_operator_proof.formal_execution_authorized is not False
        ):
            raise ValueError(
                "trusted ChronoBelief empirical authority differs from launch"
            )
        return (
            trusted_single_operator_proof.sha256,
            trusted_single_operator_proof.source_identity_sha256,
        )
    proof = require_chronobelief_gpu_proof(
        claimed_source_capability_sha256=(
            adaptation.chronobelief_release_capability_sha256
        ),
        claimed_gpu_proof_sha256=adaptation.chronobelief_gpu_proof_sha256,
        verified_gpu_proof=verified_gpu_proof,
        expected_source_identity_sha256=expected_source_identity_sha256,
        expected_inventory_sha256=inventory_sha256,
        expected_gpu_uuids=gpu_uuids,
    )
    return proof.sha256, proof.source_identity_sha256


def _expected_server_execution_policy_fields(config: RunConfig) -> dict[str, object]:
    from lightcone_spec.orchestration.runtime import (
        _execution_role,
        _runtime_execution_policy,
    )

    policy = _runtime_execution_policy(config.runtime)
    role = _execution_role(config.method)
    return {
        **policy.server_info_fields(role=role),
        **_adaptation_mechanism_server_fields(config),
    }


async def _observe_live_server_execution_policy(
    *,
    transport: PinnedBenchServingTransport,
    config: RunConfig,
) -> tuple[str, str]:
    """Read the actual patched server policy before native evidence begins."""

    from lightcone_spec.orchestration.runtime import (
        _execution_role,
        _runtime_execution_policy,
    )

    policy = _runtime_execution_policy(config.runtime)
    role = _execution_role(config.method)
    observed = await transport.get_json("/server_info")
    policy.validate_server_info(observed, role=role)
    exact_fields = _expected_server_execution_policy_fields(config)
    if any(
        type(observed.get(name)) is not type(expected) or observed.get(name) != expected
        for name, expected in exact_fields.items()
    ):
        raise ValueError("server adaptation mechanism policy differs from RunConfig")
    canonical = json.dumps(
        exact_fields,
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _absolute_output_path(label: str, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    if path == Path(path.anchor) or not path.name:
        raise ValueError(f"{label} must name one file")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"{label} parent must be a regular directory")
    parent_status = parent.stat(follow_symlinks=False)
    if (
        parent_status.st_uid != os.geteuid()
        or stat.S_IMODE(parent_status.st_mode) & 0o022
    ):
        raise ValueError(f"{label} parent must be current-user-owned and non-writable")
    if os.path.lexists(path):
        raise FileExistsError(f"{label} already exists")
    return path


def _optional_json_binding(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file() or path.is_symlink():
        return None
    try:
        return CanonicalJsonProofBinding.bind(path).to_dict()
    except (OSError, TypeError, ValueError):
        return None


def _optional_file_binding(
    path: Path | None, *, label: str
) -> dict[str, object] | None:
    if path is None or not path.is_file() or path.is_symlink():
        return None
    try:
        return EvidenceFileBinding.bind(path, label=label).to_dict()
    except (OSError, TypeError, ValueError):
        return None


def _publish_fatal_pointer(
    path: Path,
    *,
    reason_code: str,
    error: BaseException,
    binding: NativeTerminalRunBinding | None,
    requested_launch_manifest_path: str,
    launch_manifest: CanonicalJsonProofBinding | None,
    terminal_path: Path | None,
    pointer_path: Path | None,
    receipt_path: Path | None,
    log_path: Path | None,
    execution_started_ns: int | None,
    process: subprocess.Popen[bytes] | None,
    process_exited_ns: int | None,
    process_group_empty_checked_ns: int | None,
    cleanup_error: BaseException | None,
    before_gpu_snapshot: CanonicalJsonProofBinding | None = None,
    ready_gpu_snapshot: CanonicalJsonProofBinding | None = None,
    after_gpu_snapshot: CanonicalJsonProofBinding | None = None,
) -> CanonicalJsonProofBinding:
    run_binding_sha256: str | None = None
    if type(binding) is NativeTerminalRunBinding:
        try:
            binding.validate()
            run_binding_sha256 = canonical_sha256(binding.begin_payload())
        except (TypeError, ValueError):
            pass
    process_id = None if process is None else process.pid
    exit_code = None if process is None else process.poll()
    process_group_empty = (
        None
        if process is None or process_group_empty_checked_ns is None
        else not _process_group_exists(process.pid)
    )
    payload = {
        "schema_version": 1,
        "kind": "unsigned_pinned_sglang_serving_fatal_pointer",
        "protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
        "status": "ERROR",
        "formal_execution_authorized": False,
        "reason_code": reason_code,
        "error_type": type(error).__name__,
        "cleanup_error_type": (
            None if cleanup_error is None else type(cleanup_error).__name__
        ),
        "emitted_ns": time.monotonic_ns(),
        "execution_started_ns": execution_started_ns,
        "run_binding_sha256": run_binding_sha256,
        "requested_launch_manifest_path": requested_launch_manifest_path,
        "launch_manifest": (
            None if launch_manifest is None else launch_manifest.to_dict()
        ),
        "terminal_artifact": _optional_json_binding(terminal_path),
        "native_itl_pointer_artifact": _optional_json_binding(pointer_path),
        "live_run_receipt": _optional_json_binding(receipt_path),
        "server_log": _optional_file_binding(
            log_path, label="failed live serving server log"
        ),
        "server_process_id": process_id,
        "server_process_exit_code": exit_code,
        "process_exited_ns": process_exited_ns,
        "process_group_empty": process_group_empty,
        "process_group_empty_checked_ns": process_group_empty_checked_ns,
        "before_gpu_snapshot": (
            None if before_gpu_snapshot is None else before_gpu_snapshot.to_dict()
        ),
        "ready_gpu_snapshot": (
            None if ready_gpu_snapshot is None else ready_gpu_snapshot.to_dict()
        ),
        "after_gpu_snapshot": (
            None if after_gpu_snapshot is None else after_gpu_snapshot.to_dict()
        ),
    }
    publish_canonical_json_no_replace(path, payload)
    return CanonicalJsonProofBinding.bind(path)


@dataclass(frozen=True)
class PinnedNvidiaSmiTool:
    """Path/content identity supplied by the sealed host environment lock."""

    executable_path: str
    executable_raw_sha256: str
    executable_size: int

    @classmethod
    def bind(cls, executable_path: str | Path) -> PinnedNvidiaSmiTool:
        path = Path(executable_path).resolve()
        binding = EvidenceFileBinding.bind(path, label="live serving nvidia-smi")
        value = cls(
            executable_path=binding.absolute_path,
            executable_raw_sha256=binding.raw_sha256,
            executable_size=binding.size,
        )
        value.revalidate()
        return value

    def revalidate(self) -> None:
        EvidenceFileBinding(
            absolute_path=self.executable_path,
            raw_sha256=self.executable_raw_sha256,
            size=self.executable_size,
        ).reopen(label="live serving nvidia-smi")

    def to_dict(self) -> dict[str, object]:
        return {
            "executable_path": self.executable_path,
            "executable_raw_sha256": self.executable_raw_sha256,
            "executable_size": self.executable_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> PinnedNvidiaSmiTool:
        if type(value) is not dict or set(value) != {
            "executable_path",
            "executable_raw_sha256",
            "executable_size",
        }:
            raise ValueError("live serving nvidia-smi fields differ")
        result = cls(**value)
        _require_sha256("live serving nvidia-smi", result.executable_raw_sha256)
        if type(result.executable_size) is not int or result.executable_size < 1:
            raise ValueError("live serving nvidia-smi size is invalid")
        return result


def _capture_tool_output(command: tuple[str, ...]) -> str:
    process = subprocess.Popen(
        command,
        env={"LANG": "C", "LC_ALL": "C"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_GPU_SNAPSHOT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=2.0)
        raise TimeoutError("live serving nvidia-smi timed out") from error
    if process.returncode != 0 or stderr or len(stdout) > 1024 * 1024:
        raise RuntimeError("live serving nvidia-smi query failed")
    try:
        return stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("live serving nvidia-smi output is not UTF-8") from error


def _csv_rows(value: str, *, columns: int, label: str) -> tuple[tuple[str, ...], ...]:
    if not value.strip():
        return ()
    rows = tuple(
        tuple(field.strip() for field in row) for row in csv.reader(io.StringIO(value))
    )
    if any(len(row) != columns or any(not field for field in row) for row in rows):
        raise ValueError(f"{label} CSV shape differs")
    return rows


def _capture_gpu_process_snapshot(
    *,
    tool: PinnedNvidiaSmiTool,
    gpu_uuids: tuple[str, ...],
    inventory_sha256: str,
    phase: Literal["before", "ready", "after"],
    output_path: Path,
    expected_server_process_group_ids: tuple[int, ...] | None = None,
    shared_server_process_group_id: int | None = None,
) -> CanonicalJsonProofBinding:
    tool.revalidate()
    _require_sha256("live serving snapshot inventory", inventory_sha256)
    if (
        type(gpu_uuids) is not tuple
        or not gpu_uuids
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(not value.startswith("GPU-") for value in gpu_uuids)
    ):
        raise ValueError("live serving snapshot GPU UUIDs are invalid")
    if phase == "ready":
        separate_groups = (
            type(expected_server_process_group_ids) is tuple
            and len(expected_server_process_group_ids) == len(gpu_uuids)
            and len(set(expected_server_process_group_ids))
            == len(expected_server_process_group_ids)
            and all(
                type(value) is int and value > 0
                for value in expected_server_process_group_ids
            )
            and shared_server_process_group_id is None
        )
        shared_group = (
            expected_server_process_group_ids is None
            and type(shared_server_process_group_id) is int
            and shared_server_process_group_id > 0
            and len(gpu_uuids) > 1
        )
        if not (separate_groups or shared_group):
            raise ValueError("ready snapshot requires exact server process groups")
    elif (
        expected_server_process_group_ids is not None
        or shared_server_process_group_id is not None
    ):
        raise ValueError("clean snapshots cannot claim server process groups")
    selector = ",".join(gpu_uuids)
    gpu_output = _capture_tool_output(
        (
            tool.executable_path,
            f"--id={selector}",
            "--query-gpu=uuid,name,memory.used",
            "--format=csv,noheader,nounits",
        )
    )
    process_output = _capture_tool_output(
        (
            tool.executable_path,
            f"--id={selector}",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    gpu_rows = _csv_rows(gpu_output, columns=3, label="live serving GPU")
    process_rows = _csv_rows(
        process_output, columns=3, label="live serving compute process"
    )
    parsed_gpu_by_uuid: dict[str, dict[str, object]] = {}
    for uuid, name, memory in gpu_rows:
        try:
            used_mib = int(memory)
        except ValueError as error:
            raise ValueError("live serving GPU memory is not integral") from error
        if used_mib < 0:
            raise ValueError("live serving GPU memory is negative")
        if uuid in parsed_gpu_by_uuid:
            raise ValueError("live serving GPU snapshot repeats a UUID")
        parsed_gpu_by_uuid[uuid] = {
            "uuid": uuid,
            "name": name,
            "memory_used_mib": used_mib,
        }
    if set(parsed_gpu_by_uuid) != set(gpu_uuids):
        raise ValueError("live serving GPU snapshot UUID coverage differs")
    parsed_gpus = [parsed_gpu_by_uuid[uuid] for uuid in gpu_uuids]
    parsed_processes: list[dict[str, object]] = []
    seen_pids: set[tuple[str, int]] = set()
    for uuid, pid_value, memory in process_rows:
        try:
            pid = int(pid_value)
            used_mib = int(memory)
        except ValueError as error:
            raise ValueError(
                "live serving compute-process row is not integral"
            ) from error
        if uuid not in gpu_uuids or pid < 1 or used_mib < 0 or (uuid, pid) in seen_pids:
            raise ValueError("live serving compute-process row is invalid")
        try:
            process_group_id = os.getpgid(pid)
        except ProcessLookupError as error:
            raise RuntimeError("live serving GPU process disappeared") from error
        if phase != "ready":
            raise RuntimeError(f"live serving {phase} GPU process gate is not empty")
        expected_group = (
            shared_server_process_group_id
            if shared_server_process_group_id is not None
            else expected_server_process_group_ids[gpu_uuids.index(uuid)]
        )
        if process_group_id != expected_group:
            raise RuntimeError("live serving ready snapshot contains a foreign process")
        seen_pids.add((uuid, pid))
        parsed_processes.append(
            {
                "gpu_uuid": uuid,
                "pid": pid,
                "process_group_id": process_group_id,
                "used_gpu_memory_mib": used_mib,
            }
        )
    uuid_order = {uuid: index for index, uuid in enumerate(gpu_uuids)}
    parsed_processes.sort(
        key=lambda row: (uuid_order[str(row["gpu_uuid"])], row["pid"])
    )
    by_uuid = {
        uuid: sum(row["gpu_uuid"] == uuid for row in parsed_processes)
        for uuid in gpu_uuids
    }
    if phase == "ready" and any(count < 1 for count in by_uuid.values()):
        raise RuntimeError("live serving ready snapshot lacks a bound GPU process")
    payload = {
        "schema_version": 1,
        "kind": "unsigned_pinned_sglang_gpu_process_snapshot",
        "protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
        "phase": phase,
        "captured_ns": time.monotonic_ns(),
        "inventory_sha256": inventory_sha256,
        "gpu_uuids": list(gpu_uuids),
        "server_process_group_ids": (
            [shared_server_process_group_id] * len(gpu_uuids)
            if shared_server_process_group_id is not None
            else (
                None
                if expected_server_process_group_ids is None
                else list(expected_server_process_group_ids)
            )
        ),
        "nvidia_smi": tool.to_dict(),
        "gpu_rows": parsed_gpus,
        "compute_process_rows": parsed_processes,
    }
    publish_canonical_json_no_replace(output_path, payload)
    return CanonicalJsonProofBinding.bind(output_path)


def _publish_gpu_snapshot_error(
    *,
    tool: PinnedNvidiaSmiTool,
    gpu_uuids: tuple[str, ...],
    inventory_sha256: str,
    phase: Literal["before", "ready", "after"],
    output_path: Path,
    error: BaseException,
    expected_server_process_group_ids: tuple[int, ...] | None = None,
) -> CanonicalJsonProofBinding:
    payload = {
        "schema_version": 1,
        "kind": "unsigned_pinned_sglang_gpu_process_snapshot_error",
        "protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
        "phase": phase,
        "captured_ns": time.monotonic_ns(),
        "inventory_sha256": inventory_sha256,
        "gpu_uuids": list(gpu_uuids),
        "server_process_group_ids": (
            None
            if expected_server_process_group_ids is None
            else list(expected_server_process_group_ids)
        ),
        "nvidia_smi": tool.to_dict(),
        "error_type": type(error).__name__,
    }
    publish_canonical_json_no_replace(output_path, payload)
    return CanonicalJsonProofBinding.bind(output_path)


def validate_pinned_sglang_gpu_process_snapshot(
    snapshot: CanonicalJsonProofBinding,
    *,
    expected_tool: PinnedNvidiaSmiTool,
    expected_gpu_uuids: tuple[str, ...],
    expected_inventory_sha256: str,
    expected_phase: Literal["before", "ready", "after"],
    expected_server_process_group_ids: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if type(snapshot) is not CanonicalJsonProofBinding:
        raise TypeError("live serving snapshot requires an exact binding")
    value = snapshot.reopen()
    fields = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "phase",
        "captured_ns",
        "inventory_sha256",
        "gpu_uuids",
        "server_process_group_ids",
        "nvidia_smi",
        "gpu_rows",
        "compute_process_rows",
    }
    if set(value) != fields:
        raise ValueError("live serving GPU snapshot fields differ")
    tool = PinnedNvidiaSmiTool.from_dict(value["nvidia_smi"])
    if (
        value["schema_version"] != 1
        or value["kind"] != "unsigned_pinned_sglang_gpu_process_snapshot"
        or value["protocol_sha256"] != PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256
        or value["phase"] != expected_phase
        or value["inventory_sha256"] != expected_inventory_sha256
        or value["gpu_uuids"] != list(expected_gpu_uuids)
        or value["server_process_group_ids"]
        != (
            None
            if expected_server_process_group_ids is None
            else list(expected_server_process_group_ids)
        )
        or tool != expected_tool
        or type(value["captured_ns"]) is not int
        or value["captured_ns"] < 1
    ):
        raise ValueError("live serving GPU snapshot identity differs")
    gpu_rows = value["gpu_rows"]
    if (
        type(gpu_rows) is not list
        or tuple(row.get("uuid") if type(row) is dict else None for row in gpu_rows)
        != expected_gpu_uuids
        or any(
            type(row) is not dict
            or set(row) != {"uuid", "name", "memory_used_mib"}
            or type(row["name"]) is not str
            or not row["name"]
            or type(row["memory_used_mib"]) is not int
            or row["memory_used_mib"] < 0
            for row in gpu_rows
        )
    ):
        raise ValueError("live serving GPU snapshot coverage differs")
    processes = value["compute_process_rows"]
    if type(processes) is not list:
        raise TypeError("live serving compute-process rows must be an array")
    if any(
        type(row) is not dict
        or set(row) != {"gpu_uuid", "pid", "process_group_id", "used_gpu_memory_mib"}
        or row["gpu_uuid"] not in expected_gpu_uuids
        or type(row["pid"]) is not int
        or row["pid"] < 1
        or type(row["process_group_id"]) is not int
        or row["process_group_id"] < 1
        or type(row["used_gpu_memory_mib"]) is not int
        or row["used_gpu_memory_mib"] < 0
        for row in processes
    ):
        raise ValueError("live serving compute-process row is malformed")
    if expected_phase in {"before", "after"} and processes:
        raise ValueError("live serving clean GPU snapshot contains a process")
    if expected_phase == "ready" and any(
        not any(type(row) is dict and row.get("gpu_uuid") == uuid for row in processes)
        for uuid in expected_gpu_uuids
    ):
        raise ValueError("live serving ready GPU snapshot lacks process coverage")
    if expected_phase == "ready":
        if (
            type(expected_server_process_group_ids) is not tuple
            or len(expected_server_process_group_ids) != len(expected_gpu_uuids)
            or any(
                row["process_group_id"]
                != expected_server_process_group_ids[
                    expected_gpu_uuids.index(row["gpu_uuid"])
                ]
                for row in processes
            )
        ):
            raise ValueError(
                "live serving ready snapshot has foreign process ownership"
            )
    elif expected_server_process_group_ids is not None:
        raise ValueError("clean snapshot unexpectedly binds a process group")
    return value


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_gone(process_group_id: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.02)
    return not _process_group_exists(process_group_id)


def _terminate_process_group(
    process: subprocess.Popen[bytes],
) -> tuple[int, Literal["already_exited_clean", "sigterm_clean"], int]:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=_PROCESS_GROUP_CLEANUP_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30.0)
            if not _wait_process_group_gone(process.pid, timeout_seconds=5.0):
                raise RuntimeError("pinned SGLang process group survived SIGKILL")
            raise RuntimeError("pinned SGLang process group required SIGKILL")
        cleanup: Literal["already_exited_clean", "sigterm_clean"] = "sigterm_clean"
    else:
        cleanup = "already_exited_clean"
    if not _wait_process_group_gone(process.pid, timeout_seconds=5.0):
        raise RuntimeError("pinned SGLang process group is not empty")
    exit_code = process.returncode
    if type(exit_code) is not int or exit_code not in {0, -signal.SIGTERM}:
        raise RuntimeError("pinned SGLang server exited unexpectedly")
    return exit_code, cleanup, time.monotonic_ns()


def _require_port_unused(port: int) -> None:
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
    except OSError:
        return
    connection.close()
    raise RuntimeError("compile launch localhost port is already serving")


def _wait_server_ready(
    process: subprocess.Popen[bytes], *, port: int, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("pinned SGLang server exited before readiness")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        try:
            connection.request("GET", "/health_generate")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(0.1)
    raise TimeoutError("pinned SGLang server readiness timed out")


def _spawn_server(
    launch: CompileLaunchManifest,
    *,
    child_environment_overlay: Mapping[str, str] | None,
    stdout_file,
    stderr_file,
) -> subprocess.Popen[bytes]:
    environment = launch.child_environment()
    if child_environment_overlay is not None:
        if any(
            type(name) is not str or not name or type(value) is not str or not value
            for name, value in child_environment_overlay.items()
        ):
            raise ValueError("live serving child environment overlay differs")
        environment.update(child_environment_overlay)
    return subprocess.Popen(
        launch.server_argv,
        cwd=launch.patched_sglang_checkout,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,
        close_fds=True,
    )


def _require_source_owned_server_executable(value: str) -> Path:
    """Accept the current interpreter, including its exact venv launcher link."""

    executable = Path(value)
    if (
        not executable.is_absolute()
        or executable.parent != executable.parent.resolve(strict=True)
        or not executable.is_file()
    ):
        raise ValueError("live serving argv requires an absolute executable")
    resolved = executable.resolve(strict=True)
    current_launcher = Path(sys.executable)
    current = current_launcher.resolve(strict=True)
    if resolved != current or resolved.is_symlink():
        raise ValueError("live serving argv uses another Python interpreter")
    if executable.is_symlink():
        if not current_launcher.is_symlink() or executable != current_launcher:
            raise ValueError("live serving venv launcher differs from this process")
        virtual_environment = executable.parent.parent / "pyvenv.cfg"
        if not virtual_environment.is_file() or virtual_environment.is_symlink():
            raise ValueError("live serving venv interpreter identity is unavailable")
    return executable


def _validate_phase_inputs(
    phase: str,
    requests: Sequence[BoundServingRequest],
    *,
    expected_ids: tuple[str, ...],
    timeout_seconds: float,
    allow_registered_outcomes: bool = False,
) -> tuple[BoundServingRequest, ...]:
    bound_request_type, _transport_type = _serving_runtime_types()
    values = tuple(requests)
    if phase not in {"warmup", "scored"}:
        raise ValueError("live serving phase is unsupported")
    if len(values) > _MAX_REQUESTS_PER_PHASE:
        raise ValueError("live serving phase exceeds bounded request coverage")
    if tuple(request.request_id for request in values) != expected_ids:
        raise ValueError("live serving request coverage differs from run binding")
    for request in values:
        if type(request) is not bound_request_type:
            raise TypeError("live serving requires exact bound requests")
        request.validate()
        if (
            type(request.arrival_us) is not int
            or request.arrival_us < 0
            or request.arrival_us > int(timeout_seconds * 1_000_000)
            or type(request.requested_output_tokens) is not int
            or request.requested_output_tokens < 1
            or not request.input_token_ids
            or any(
                type(token_id) is not int or token_id < 0
                for token_id in request.input_token_ids
            )
            or (
                request.cancellation_offset_us is not None
                and not allow_registered_outcomes
            )
        ):
            raise ValueError("formal live serving request timing contract is invalid")
    return values


def _completed_expectation(
    request: BoundServingRequest, result: BenchServingResult
) -> TerminalRequestExpectation:
    if (
        not result.success
        or result.output_tokens != request.requested_output_tokens
        or result.native_result_pointer_json is None
    ):
        raise RuntimeError("live serving request lacks exact native completion")
    expectation = TerminalRequestExpectation(
        request_id=request.request_id,
        input_token_ids=request.input_token_ids,
        output_token_ids=result.generated_token_ids,
        terminal_status="completed",
        terminal_reason="FINISH_LENGTH",
        submitted_to_server=True,
    )
    expectation.validate()
    return expectation


def _reopen_native_scored_interval(
    *,
    pointer_artifact: CanonicalJsonProofBinding,
    terminal_artifact: CanonicalJsonProofBinding,
    binding: NativeTerminalRunBinding,
    terminal_evidence: ValidatedNativeTerminalEvidence,
    scored_request_inputs_sha256: str,
) -> tuple[int, int]:
    """Derive the actual scored interval from first-party native pointers."""

    terminal_output_tokens = {
        request.request_id: request.output_token_ids
        for request in terminal_evidence.requests
        if request.submitted_to_server
        and request.terminal_status == "completed"
        and request.output_token_ids is not None
    }
    ordered_outputs = {
        request_id: terminal_output_tokens[request_id]
        for request_id in binding.scored_request_ids
        if request_id in terminal_output_tokens
    }
    bundle = validate_unsigned_native_itl_pointer_bundle(
        pointer_artifact,
        expected_binding=binding,
        expected_terminal_artifact=terminal_artifact,
        expected_scored_request_inputs_sha256=scored_request_inputs_sha256,
        expected_terminal_output_tokens=ordered_outputs,
    )
    if not bundle.pointers:
        raise ValueError("live serving scored interval has no native pointers")
    started_ns = min(pointer.request_started_ns for pointer in bundle.pointers)
    finished_ns = max(pointer.request_terminal_ns for pointer in bundle.pointers)
    if finished_ns <= started_ns:
        raise ValueError("live serving native scored interval is empty")
    return started_ns, finished_ns


def _derive_actual_group_overlap(
    *,
    shared_origin_ns: int,
    scored_intervals: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    """Intersect actual native request intervals, never scheduling envelopes."""

    if type(shared_origin_ns) is not int or shared_origin_ns < 1:
        raise ValueError("live serving group shared origin is invalid")
    if (
        type(scored_intervals) is not tuple
        or len(scored_intervals) != 2
        or any(
            type(interval) is not tuple
            or len(interval) != 2
            or any(type(value) is not int for value in interval)
            or interval[0] < shared_origin_ns
            or interval[1] <= interval[0]
            for interval in scored_intervals
        )
    ):
        raise ValueError("live serving group native intervals are invalid")
    overlap_started_ns = max(interval[0] for interval in scored_intervals)
    overlap_finished_ns = min(interval[1] for interval in scored_intervals)
    if overlap_finished_ns <= overlap_started_ns:
        raise RuntimeError(
            "concurrent serving native request intervals did not overlap"
        )
    return overlap_started_ns, overlap_finished_ns


async def _execute_one_request(
    request: BoundServingRequest,
    *,
    origin_ns: int,
    semaphore: asyncio.Semaphore,
    transport: PinnedBenchServingTransport,
    base_url: str,
    served_model: str,
) -> tuple[TerminalRequestExpectation, str | None]:
    remaining_us = request.arrival_us - (time.monotonic_ns() - origin_ns) // 1_000
    if remaining_us > 0:
        await asyncio.sleep(remaining_us / 1_000_000)
    async with semaphore:
        submit_task = asyncio.create_task(
            transport.submit(request, base_url=base_url, served_model=served_model)
        )
        try:
            result = await submit_task
            result.validate(request)
            expectation = _completed_expectation(request, result)
        except BaseException:
            if not submit_task.done():
                submit_task.cancel()
                await asyncio.gather(submit_task, return_exceptions=True)
            raise
    pointer = (
        result.native_result_pointer_json
        if expectation.terminal_status == "completed"
        else None
    )
    return expectation, pointer


async def _execute_source_owned_phase(
    phase: str,
    requests: tuple[BoundServingRequest, ...],
    *,
    concurrency: int,
    transport: PinnedBenchServingTransport,
    base_url: str,
    served_model: str,
    shared_origin_ns: int | None = None,
    execution_policy: RegisteredServingExecutionPolicy | None = None,
) -> UnsignedNativeServingPhaseResult:
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("live serving concurrency is invalid")
    origin_ns = time.monotonic_ns() if shared_origin_ns is None else shared_origin_ns
    if type(origin_ns) is not int or origin_ns < 1 or origin_ns > time.monotonic_ns():
        raise ValueError("live serving phase origin is invalid")
    if execution_policy is not None:
        from lightcone_spec.orchestration.executor import (
            RegisteredServingExecutionPolicy,
            execute_registered_serving_phase,
            terminal_request_expectation_for_execution,
        )

        if type(execution_policy) is not RegisteredServingExecutionPolicy:
            raise TypeError("live serving requires an exact execution policy")
        execution_policy.__post_init__()
        phase_execution = await execute_registered_serving_phase(
            phase,
            requests,
            source_kind=(
                "scheduled" if phase == "warmup" else execution_policy.source_kind
            ),
            arrival_duration_us=(
                execution_policy.warmup_duration_us
                if phase == "warmup"
                else execution_policy.arrival_duration_us
            ),
            request_deadline_us=execution_policy.request_deadline_us,
            drain_duration_us=execution_policy.drain_duration_us,
            concurrency=execution_policy.max_concurrency,
            transport=transport,
            base_url=base_url,
            served_model=served_model,
            abort_grace_s=_ABORT_TIMEOUT_SECONDS,
            complete_closed_loop_pool=(
                phase == "scored" and execution_policy.complete_closed_loop_pool
            ),
            shared_origin_ns=origin_ns,
        )
        if phase == "warmup" and any(
            not row.offered
            or not row.submitted_to_server
            or row.outcome_status != "completed"
            or row.native_terminal_status != "completed"
            for row in phase_execution.lifecycles
        ):
            raise RuntimeError(
                "registered serving warmup requires strict native completion"
            )
        by_id = {
            execution.request.request_id: execution
            for execution in phase_execution.executions
        }
        expectations: list[TerminalRequestExpectation] = []
        pointers: list[str] = []
        for request in requests:
            execution = by_id.get(request.request_id)
            if execution is None:
                expectation = TerminalRequestExpectation(
                    request_id=request.request_id,
                    input_token_ids=request.input_token_ids,
                    output_token_ids=None,
                    terminal_status="rejected",
                    terminal_reason="NOT_OFFERED_WINDOW_CLOSED",
                    submitted_to_server=False,
                )
            else:
                expectation = terminal_request_expectation_for_execution(execution)
                if (
                    expectation.terminal_status == "completed"
                    and execution.result is not None
                    and execution.result.native_result_pointer_json is not None
                ):
                    pointers.append(execution.result.native_result_pointer_json)
            expectations.append(expectation)
        result = UnsignedNativeServingPhaseResult(
            phase=phase,
            requests=tuple(expectations),
            native_result_pointer_json=tuple(pointers),
            client_lifecycle_rows=phase_execution.accounting_rows,
        )
        result.validate(expected_phase=phase, bound_requests=requests)
        return result
    semaphore = asyncio.Semaphore(concurrency)
    tasks = tuple(
        asyncio.create_task(
            _execute_one_request(
                request,
                origin_ns=origin_ns,
                semaphore=semaphore,
                transport=transport,
                base_url=base_url,
                served_model=served_model,
            )
        )
        for request in requests
    )
    try:
        rows = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    result = UnsignedNativeServingPhaseResult(
        phase=phase,
        requests=tuple(row[0] for row in rows),
        native_result_pointer_json=tuple(
            pointer for _expectation, pointer in rows if pointer is not None
        ),
    )
    result.validate(expected_phase=phase, bound_requests=requests)
    return result


@dataclass(frozen=True)
class UnsignedPinnedSglangServingRunReceipt:
    """Durable source/run/process binding; still not a trust authority."""

    schema_version: Literal[1, 2]
    kind: Literal["unsigned_pinned_sglang_serving_run_receipt"]
    protocol_sha256: str
    formal_execution_authorized: Literal[False]
    runner_source_raw_sha256: str
    runner_source_size: int
    launch_manifest: CanonicalJsonProofBinding
    formal_launch_admission: CanonicalJsonProofBinding | None
    formal_launch_consumption: CanonicalJsonProofBinding | None
    budget_consumption: CanonicalJsonProofBinding | None
    run_binding_sha256: str
    terminal_artifact: CanonicalJsonProofBinding
    native_itl_pointer_artifact: CanonicalJsonProofBinding
    terminal_sha256: str
    patched_sglang_commit: str
    patched_sglang_tree: str
    server_argv_sha256: str
    execution_policy_sha256: str
    cuda_graph_mode: Literal["disabled", "fixed_address_publication_v1"]
    native_graph_gpu_proof_sha256: str | None
    native_graph_source_identity_sha256: str | None
    eagle3_e0_execution_authority_sha256: str | None
    eagle3_compatibility_authority_sha256: str | None
    eagle3_model_selector_sha256: str | None
    eagle3_native_gpu_proof_sha256: str | None
    eagle3_native_source_identity_sha256: str | None
    chronobelief_gpu_proof_sha256: str | None
    chronobelief_source_identity_sha256: str | None
    native_lifecycle_events_json: str
    native_lifecycle_events_sha256: str
    server_execution_policy_fields_json: str
    server_execution_policy_fields_sha256: str
    physical_assignment_sha256: str
    experiment_budget_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    server_process_id: int
    server_process_started_ns: int
    execution_started_ns: int
    scored_started_ns: int
    scored_finished_ns: int
    native_terminal_finished_ns: int
    process_exited_ns: int
    process_exit_code: int
    cleanup_kind: Literal["already_exited_clean", "sigterm_clean"]
    process_group_empty: Literal[True]
    process_group_empty_checked_ns: int
    server_log: EvidenceFileBinding
    snapshot_gpu_uuids: tuple[str, ...]
    server_process_group_ids: tuple[int, ...]
    ready_compute_process_rows_sha256: str
    before_gpu_snapshot: CanonicalJsonProofBinding
    ready_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    execution_policy: RegisteredServingExecutionPolicy | None = None
    client_request_lifecycle: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        from lightcone_spec.orchestration.executor import (
            RegisteredServingExecutionPolicy,
        )

        if (
            self.schema_version not in {1, 2}
            or self.kind != "unsigned_pinned_sglang_serving_run_receipt"
            or self.protocol_sha256
            != (
                PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256
                if self.schema_version == 1
                else PINNED_SGLANG_REGISTERED_SERVING_PROTOCOL_SHA256
            )
            or self.formal_execution_authorized is not False
            or self.patched_sglang_commit != PINNED_SGLANG_COMMIT
            or self.patched_sglang_tree != PINNED_SGLANG_TREE
            or self.process_group_empty is not True
        ):
            raise ValueError("live serving receipt schema/source differs")
        if self.schema_version == 1:
            if (
                self.execution_policy is not None
                or self.client_request_lifecycle is not None
            ):
                raise ValueError("legacy live receipt carries registered lifecycle")
        else:
            if (
                type(self.execution_policy) is not RegisteredServingExecutionPolicy
                or type(self.client_request_lifecycle) is not CanonicalJsonProofBinding
            ):
                raise ValueError("registered live receipt lacks client lifecycle")
            self.execution_policy.__post_init__()
            rows = reopen_scalable_client_request_lifecycle(
                self.client_request_lifecycle,
                expected_run_binding_sha256=self.run_binding_sha256,
                expected_execution_policy_sha256=self.execution_policy.sha256,
            )
            if not rows:
                raise ValueError("registered live receipt lifecycle is empty")
        launch_lineage = (
            self.formal_launch_admission,
            self.formal_launch_consumption,
            self.budget_consumption,
        )
        if len({value is None for value in launch_lineage}) != 1 or any(
            value is not None and type(value) is not CanonicalJsonProofBinding
            for value in launch_lineage
        ):
            raise TypeError("live serving formal launch lineage must be atomic")
        for label, digest in (
            ("run binding", self.run_binding_sha256),
            ("runner source", self.runner_source_raw_sha256),
            ("terminal", self.terminal_sha256),
            ("server argv", self.server_argv_sha256),
            ("execution policy", self.execution_policy_sha256),
            (
                "server execution policy fields",
                self.server_execution_policy_fields_sha256,
            ),
            ("physical assignment", self.physical_assignment_sha256),
            ("experiment budget", self.experiment_budget_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"live serving {label}", digest)
        graph_authority = (
            self.native_graph_gpu_proof_sha256,
            self.native_graph_source_identity_sha256,
        )
        if self.cuda_graph_mode == "disabled":
            if any(value is not None for value in graph_authority):
                raise ValueError("disabled live serving receipt claims graph authority")
        elif self.cuda_graph_mode == "fixed_address_publication_v1":
            if any(value is None for value in graph_authority):
                raise ValueError("graph live serving receipt lacks dynamic authority")
            for label, value in zip(
                ("graph GPU proof", "graph source identity"),
                graph_authority,
                strict=True,
            ):
                _require_sha256(f"live serving {label}", value)
        else:
            raise ValueError("live serving receipt graph mode is unsupported")
        eagle3_authority = (
            self.eagle3_e0_execution_authority_sha256,
            self.eagle3_compatibility_authority_sha256,
            self.eagle3_model_selector_sha256,
            self.eagle3_native_gpu_proof_sha256,
            self.eagle3_native_source_identity_sha256,
        )
        if any(value is None for value in eagle3_authority) and any(
            value is not None for value in eagle3_authority
        ):
            raise ValueError("live serving EAGLE3 authority must be atomic")
        for label, value in zip(
            (
                "EAGLE3 E0 execution authority",
                "EAGLE3 compatibility authority",
                "EAGLE3 model selector",
                "EAGLE3 native GPU proof",
                "EAGLE3 native source identity",
            ),
            eagle3_authority,
            strict=True,
        ):
            if value is not None:
                _require_sha256(f"live serving {label}", value)
        try:
            lifecycle_value = json.loads(self.native_lifecycle_events_json)
            lifecycle = UnsignedNativeLifecycleEvents(**lifecycle_value)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                "live serving native lifecycle events are invalid"
            ) from error
        if (
            json.dumps(lifecycle_value, sort_keys=True, separators=(",", ":"))
            != self.native_lifecycle_events_json
            or lifecycle.sha256 != self.native_lifecycle_events_sha256
        ):
            raise ValueError("live serving native lifecycle event identity changed")
        chronobelief_authority = (
            self.chronobelief_gpu_proof_sha256,
            self.chronobelief_source_identity_sha256,
        )
        if any(value is None for value in chronobelief_authority) and any(
            value is not None for value in chronobelief_authority
        ):
            raise ValueError("live serving ChronoBelief authority must be atomic")
        for label, value in zip(
            ("ChronoBelief GPU proof", "ChronoBelief source identity"),
            chronobelief_authority,
            strict=True,
        ):
            if value is not None:
                _require_sha256(f"live serving {label}", value)
        try:
            observed_policy_fields = json.loads(
                self.server_execution_policy_fields_json
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "live serving server policy fields are not JSON"
            ) from error
        canonical_policy_fields = json.dumps(
            observed_policy_fields,
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            type(observed_policy_fields) is not dict
            or canonical_policy_fields != self.server_execution_policy_fields_json
            or hashlib.sha256(canonical_policy_fields.encode("utf-8")).hexdigest()
            != self.server_execution_policy_fields_sha256
        ):
            raise ValueError("live serving server policy fields are not canonical")
        if (
            type(self.launch_manifest) is not CanonicalJsonProofBinding
            or type(self.terminal_artifact) is not CanonicalJsonProofBinding
            or type(self.native_itl_pointer_artifact) is not CanonicalJsonProofBinding
            or type(self.server_log) is not EvidenceFileBinding
            or type(self.before_gpu_snapshot) is not CanonicalJsonProofBinding
            or type(self.ready_gpu_snapshot) is not CanonicalJsonProofBinding
            or type(self.after_gpu_snapshot) is not CanonicalJsonProofBinding
        ):
            raise TypeError("live serving receipt lost a path-bound input")
        if type(self.runner_source_size) is not int or self.runner_source_size < 1:
            raise ValueError("live serving runner source size is invalid")
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 1
            or not self.gpu_uuids[0].startswith("GPU-")
        ):
            raise ValueError("live serving receipt GPU binding is invalid")
        if (
            type(self.snapshot_gpu_uuids) is not tuple
            or len(self.snapshot_gpu_uuids) not in {1, 2}
            or len(set(self.snapshot_gpu_uuids)) != len(self.snapshot_gpu_uuids)
            or self.gpu_uuids[0] not in self.snapshot_gpu_uuids
            or type(self.server_process_group_ids) is not tuple
            or len(self.server_process_group_ids) != len(self.snapshot_gpu_uuids)
            or len(set(self.server_process_group_ids))
            != len(self.server_process_group_ids)
            or self.server_process_id not in self.server_process_group_ids
            or any(
                type(value) is not int or value < 1
                for value in self.server_process_group_ids
            )
        ):
            raise ValueError("live serving snapshot/process-group binding is invalid")
        _require_sha256(
            "live serving ready process rows", self.ready_compute_process_rows_sha256
        )
        if (
            type(self.server_process_id) is not int
            or self.server_process_id < 1
            or type(self.server_process_started_ns) is not int
            or self.server_process_started_ns < 1
            or type(self.process_exit_code) is not int
            or self.process_exit_code not in {0, -signal.SIGTERM}
            or self.cleanup_kind not in {"already_exited_clean", "sigterm_clean"}
        ):
            raise ValueError("live serving receipt process outcome is invalid")
        times = (
            self.execution_started_ns,
            self.scored_started_ns,
            self.scored_finished_ns,
            self.native_terminal_finished_ns,
            self.process_exited_ns,
            self.process_group_empty_checked_ns,
        )
        if any(
            type(value) is not int or value < 1 for value in times
        ) or times != tuple(sorted(times)):
            raise ValueError("live serving receipt timestamps are not ordered")
        if (
            lifecycle.begin_started_ns < self.execution_started_ns
            or lifecycle.scored_started_ns > self.scored_started_ns
            or lifecycle.scored_finished_ns < self.scored_finished_ns
            or lifecycle.itl_pointer_published_ns > self.native_terminal_finished_ns
            or self.native_terminal_finished_ns > self.process_exited_ns
        ):
            raise ValueError("live serving lifecycle does not enclose native execution")

    def to_dict(self) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "formal_execution_authorized": self.formal_execution_authorized,
            "runner_source_raw_sha256": self.runner_source_raw_sha256,
            "runner_source_size": self.runner_source_size,
            "launch_manifest": self.launch_manifest.to_dict(),
            "formal_launch_admission": (
                None
                if self.formal_launch_admission is None
                else self.formal_launch_admission.to_dict()
            ),
            "formal_launch_consumption": (
                None
                if self.formal_launch_consumption is None
                else self.formal_launch_consumption.to_dict()
            ),
            "budget_consumption": (
                None
                if self.budget_consumption is None
                else self.budget_consumption.to_dict()
            ),
            "run_binding_sha256": self.run_binding_sha256,
            "terminal_artifact": self.terminal_artifact.to_dict(),
            "native_itl_pointer_artifact": (self.native_itl_pointer_artifact.to_dict()),
            "terminal_sha256": self.terminal_sha256,
            "patched_sglang_commit": self.patched_sglang_commit,
            "patched_sglang_tree": self.patched_sglang_tree,
            "server_argv_sha256": self.server_argv_sha256,
            "execution_policy_sha256": self.execution_policy_sha256,
            "cuda_graph_mode": self.cuda_graph_mode,
            "native_graph_gpu_proof_sha256": self.native_graph_gpu_proof_sha256,
            "native_graph_source_identity_sha256": (
                self.native_graph_source_identity_sha256
            ),
            "eagle3_e0_execution_authority_sha256": (
                self.eagle3_e0_execution_authority_sha256
            ),
            "eagle3_compatibility_authority_sha256": (
                self.eagle3_compatibility_authority_sha256
            ),
            "eagle3_model_selector_sha256": self.eagle3_model_selector_sha256,
            "eagle3_native_gpu_proof_sha256": self.eagle3_native_gpu_proof_sha256,
            "eagle3_native_source_identity_sha256": (
                self.eagle3_native_source_identity_sha256
            ),
            "chronobelief_gpu_proof_sha256": self.chronobelief_gpu_proof_sha256,
            "chronobelief_source_identity_sha256": (
                self.chronobelief_source_identity_sha256
            ),
            "native_lifecycle_events_json": self.native_lifecycle_events_json,
            "native_lifecycle_events_sha256": self.native_lifecycle_events_sha256,
            "server_execution_policy_fields_json": (
                self.server_execution_policy_fields_json
            ),
            "server_execution_policy_fields_sha256": (
                self.server_execution_policy_fields_sha256
            ),
            "physical_assignment_sha256": self.physical_assignment_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "inventory_sha256": self.inventory_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "server_process_id": self.server_process_id,
            "server_process_started_ns": self.server_process_started_ns,
            "execution_started_ns": self.execution_started_ns,
            "scored_started_ns": self.scored_started_ns,
            "scored_finished_ns": self.scored_finished_ns,
            "native_terminal_finished_ns": self.native_terminal_finished_ns,
            "process_exited_ns": self.process_exited_ns,
            "process_exit_code": self.process_exit_code,
            "cleanup_kind": self.cleanup_kind,
            "process_group_empty": self.process_group_empty,
            "process_group_empty_checked_ns": self.process_group_empty_checked_ns,
            "server_log": self.server_log.to_dict(),
            "snapshot_gpu_uuids": list(self.snapshot_gpu_uuids),
            "server_process_group_ids": list(self.server_process_group_ids),
            "ready_compute_process_rows_sha256": (
                self.ready_compute_process_rows_sha256
            ),
            "before_gpu_snapshot": self.before_gpu_snapshot.to_dict(),
            "ready_gpu_snapshot": self.ready_gpu_snapshot.to_dict(),
            "after_gpu_snapshot": self.after_gpu_snapshot.to_dict(),
        }
        if self.schema_version == 2:
            assert self.execution_policy is not None
            assert self.client_request_lifecycle is not None
            value["execution_policy"] = self.execution_policy.to_dict()
            value["client_request_lifecycle"] = self.client_request_lifecycle.to_dict()
        return value

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def native_lifecycle_events(self) -> UnsignedNativeLifecycleEvents:
        value = json.loads(self.native_lifecycle_events_json)
        if type(value) is not dict:  # pragma: no cover - __post_init__ invariant
            raise TypeError("live serving native lifecycle events changed type")
        return UnsignedNativeLifecycleEvents(**value)

    @classmethod
    def from_dict(cls, value: object) -> UnsignedPinnedSglangServingRunReceipt:
        from lightcone_spec.orchestration.executor import (
            RegisteredServingExecutionPolicy,
        )

        fields = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "runner_source_raw_sha256",
            "runner_source_size",
            "launch_manifest",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "run_binding_sha256",
            "terminal_artifact",
            "native_itl_pointer_artifact",
            "terminal_sha256",
            "patched_sglang_commit",
            "patched_sglang_tree",
            "server_argv_sha256",
            "execution_policy_sha256",
            "cuda_graph_mode",
            "native_graph_gpu_proof_sha256",
            "native_graph_source_identity_sha256",
            "eagle3_e0_execution_authority_sha256",
            "eagle3_compatibility_authority_sha256",
            "eagle3_model_selector_sha256",
            "eagle3_native_gpu_proof_sha256",
            "eagle3_native_source_identity_sha256",
            "chronobelief_gpu_proof_sha256",
            "chronobelief_source_identity_sha256",
            "native_lifecycle_events_json",
            "native_lifecycle_events_sha256",
            "server_execution_policy_fields_json",
            "server_execution_policy_fields_sha256",
            "physical_assignment_sha256",
            "experiment_budget_sha256",
            "inventory_sha256",
            "gpu_uuids",
            "server_process_id",
            "server_process_started_ns",
            "execution_started_ns",
            "scored_started_ns",
            "scored_finished_ns",
            "native_terminal_finished_ns",
            "process_exited_ns",
            "process_exit_code",
            "cleanup_kind",
            "process_group_empty",
            "process_group_empty_checked_ns",
            "server_log",
            "snapshot_gpu_uuids",
            "server_process_group_ids",
            "ready_compute_process_rows_sha256",
            "before_gpu_snapshot",
            "ready_gpu_snapshot",
            "after_gpu_snapshot",
        }
        if type(value) is dict and value.get("schema_version") == 2:
            fields.update({"execution_policy", "client_request_lifecycle"})
        if type(value) is not dict or set(value) != fields:
            raise ValueError("live serving receipt fields differ from schema")
        row = dict(value)
        gpu_uuids = row.pop("gpu_uuids")
        snapshot_gpu_uuids = row.pop("snapshot_gpu_uuids")
        process_group_ids = row.pop("server_process_group_ids")
        if any(
            type(value) is not list
            for value in (gpu_uuids, snapshot_gpu_uuids, process_group_ids)
        ):
            raise TypeError("live serving receipt GPU/process rows must be arrays")
        launch_manifest = CanonicalJsonProofBinding.from_dict(
            row.pop("launch_manifest")
        )
        formal_launch_admission = row.pop("formal_launch_admission")
        formal_launch_consumption = row.pop("formal_launch_consumption")
        budget_consumption = row.pop("budget_consumption")
        terminal_artifact = CanonicalJsonProofBinding.from_dict(
            row.pop("terminal_artifact")
        )
        native_itl_pointer_artifact = CanonicalJsonProofBinding.from_dict(
            row.pop("native_itl_pointer_artifact")
        )
        server_log = EvidenceFileBinding.from_dict(
            row.pop("server_log"), label="live serving server log"
        )
        before_gpu_snapshot = CanonicalJsonProofBinding.from_dict(
            row.pop("before_gpu_snapshot")
        )
        ready_gpu_snapshot = CanonicalJsonProofBinding.from_dict(
            row.pop("ready_gpu_snapshot")
        )
        after_gpu_snapshot = CanonicalJsonProofBinding.from_dict(
            row.pop("after_gpu_snapshot")
        )
        execution_policy_value = row.pop("execution_policy", None)
        client_lifecycle_value = row.pop("client_request_lifecycle", None)
        return cls(
            **row,
            gpu_uuids=tuple(gpu_uuids),
            launch_manifest=launch_manifest,
            formal_launch_admission=(
                None
                if formal_launch_admission is None
                else CanonicalJsonProofBinding.from_dict(formal_launch_admission)
            ),
            formal_launch_consumption=(
                None
                if formal_launch_consumption is None
                else CanonicalJsonProofBinding.from_dict(formal_launch_consumption)
            ),
            budget_consumption=(
                None
                if budget_consumption is None
                else CanonicalJsonProofBinding.from_dict(budget_consumption)
            ),
            terminal_artifact=terminal_artifact,
            native_itl_pointer_artifact=native_itl_pointer_artifact,
            server_log=server_log,
            snapshot_gpu_uuids=tuple(snapshot_gpu_uuids),
            server_process_group_ids=tuple(process_group_ids),
            before_gpu_snapshot=before_gpu_snapshot,
            ready_gpu_snapshot=ready_gpu_snapshot,
            after_gpu_snapshot=after_gpu_snapshot,
            execution_policy=(
                None
                if execution_policy_value is None
                else RegisteredServingExecutionPolicy.from_dict(execution_policy_value)
            ),
            client_request_lifecycle=(
                None
                if client_lifecycle_value is None
                else CanonicalJsonProofBinding.from_dict(client_lifecycle_value)
            ),
        )


_LIFECYCLE_EDGE_NAMES = (
    "execution_started_ns",
    "server_ready_ns",
    "begin_started_ns",
    "begin_finished_ns",
    "warmup_started_ns",
    "warmup_finished_ns",
    "reset_started_ns",
    "reset_finished_ns",
    "scored_executor_started_ns",
    "scored_request_started_ns",
    "scored_request_finished_ns",
    "scored_executor_finished_ns",
    "finalize_started_ns",
    "finalize_finished_ns",
    "terminal_published_ns",
    "itl_pointer_published_ns",
    "native_terminal_finished_ns",
    "process_exited_ns",
    "process_group_empty_checked_ns",
    "evidence_flush_started_ns",
    "evidence_flush_finished_ns",
)
_LIFECYCLE_DURATION_NAMES = (
    "startup_ns",
    "warmup_ns",
    "adaptation_reset_ns",
    "scored_request_window_ns",
    "drain_ns",
    "process_cleanup_ns",
    "evidence_flush_ns",
    "reserved_wall_ns",
    "profile_reserved_ns",
)


def _lifecycle_durations(
    edges: dict[str, int], *, telemetry_detail: str
) -> dict[str, int]:
    values = {
        "startup_ns": edges["server_ready_ns"] - edges["execution_started_ns"],
        "warmup_ns": edges["warmup_finished_ns"] - edges["warmup_started_ns"],
        "adaptation_reset_ns": (edges["reset_finished_ns"] - edges["reset_started_ns"]),
        "scored_request_window_ns": (
            edges["scored_request_finished_ns"] - edges["scored_request_started_ns"]
        ),
        "drain_ns": (
            edges["native_terminal_finished_ns"] - edges["scored_request_finished_ns"]
        ),
        "process_cleanup_ns": (
            edges["process_group_empty_checked_ns"]
            - edges["native_terminal_finished_ns"]
        ),
        "evidence_flush_ns": (
            edges["evidence_flush_finished_ns"] - edges["evidence_flush_started_ns"]
        ),
        "reserved_wall_ns": (
            edges["evidence_flush_finished_ns"] - edges["execution_started_ns"]
        ),
    }
    values["profile_reserved_ns"] = (
        values["reserved_wall_ns"] if telemetry_detail == "profile" else 0
    )
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError("live lifecycle duration is negative or non-integral")
    return values


@dataclass(frozen=True)
class UnsignedPinnedSglangLifecycleTimingReceipt:
    """Path-bound integer-ns phase evidence; never self-authorizing."""

    schema_version: Literal[1]
    kind: Literal["unsigned_pinned_sglang_lifecycle_timing_receipt"]
    protocol_sha256: str
    formal_execution_authorized: Literal[False]
    live_run_receipt: CanonicalJsonProofBinding
    formal_launch_admission: CanonicalJsonProofBinding | None
    formal_launch_consumption: CanonicalJsonProofBinding | None
    budget_consumption: CanonicalJsonProofBinding | None
    run_binding_sha256: str
    run_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    method: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    telemetry_detail: Literal["headline", "profile"]
    phase_edges_ns: dict[str, int]
    phase_durations_ns: dict[str, int]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "unsigned_pinned_sglang_lifecycle_timing_receipt"
            or self.protocol_sha256 != PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256
            or self.formal_execution_authorized is not False
            or type(self.live_run_receipt) is not CanonicalJsonProofBinding
        ):
            raise ValueError("live lifecycle timing schema is unsupported")
        launch_lineage = (
            self.formal_launch_admission,
            self.formal_launch_consumption,
            self.budget_consumption,
        )
        if len({value is None for value in launch_lineage}) != 1 or any(
            value is not None and type(value) is not CanonicalJsonProofBinding
            for value in launch_lineage
        ):
            raise TypeError("live lifecycle formal launch lineage must be atomic")
        for label, value in (
            ("run binding", self.run_binding_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("rank config", self.rank_config_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"live lifecycle {label}", value)
        for label, value in (
            ("run ID", self.run_id),
            ("attempt ID", self.attempt_id),
            ("method", self.method),
        ):
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"live lifecycle {label} is invalid")
        if (
            type(self.gpu_uuids) is not tuple
            or not self.gpu_uuids
            or len(set(self.gpu_uuids)) != len(self.gpu_uuids)
            or any(not value.startswith("GPU-") for value in self.gpu_uuids)
        ):
            raise ValueError("live lifecycle GPU coverage is invalid")
        if (
            type(self.phase_edges_ns) is not dict
            or tuple(self.phase_edges_ns) != _LIFECYCLE_EDGE_NAMES
            or any(
                type(value) is not int or value < 1
                for value in self.phase_edges_ns.values()
            )
            or type(self.phase_durations_ns) is not dict
            or tuple(self.phase_durations_ns) != _LIFECYCLE_DURATION_NAMES
            or self.phase_durations_ns
            != _lifecycle_durations(
                self.phase_edges_ns, telemetry_detail=self.telemetry_detail
            )
        ):
            raise ValueError("live lifecycle phase timing is not canonical")
        edges = self.phase_edges_ns
        if tuple(edges.values()) != tuple(sorted(edges.values())):
            raise ValueError("live lifecycle phase edges are not monotonic")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "formal_execution_authorized": self.formal_execution_authorized,
            "live_run_receipt": self.live_run_receipt.to_dict(),
            "formal_launch_admission": (
                None
                if self.formal_launch_admission is None
                else self.formal_launch_admission.to_dict()
            ),
            "formal_launch_consumption": (
                None
                if self.formal_launch_consumption is None
                else self.formal_launch_consumption.to_dict()
            ),
            "budget_consumption": (
                None
                if self.budget_consumption is None
                else self.budget_consumption.to_dict()
            ),
            "run_binding_sha256": self.run_binding_sha256,
            "run_id": self.run_id,
            "run_nonce_sha256": self.run_nonce_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "attempt_id": self.attempt_id,
            "method": self.method,
            "inventory_sha256": self.inventory_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "telemetry_detail": self.telemetry_detail,
            "phase_edges_ns": self.phase_edges_ns,
            "phase_durations_ns": self.phase_durations_ns,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> UnsignedPinnedSglangLifecycleTimingReceipt:
        fields = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "live_run_receipt",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "run_binding_sha256",
            "run_id",
            "run_nonce_sha256",
            "execution_plan_sha256",
            "rank_config_sha256",
            "attempt_id",
            "method",
            "inventory_sha256",
            "gpu_uuids",
            "telemetry_detail",
            "phase_edges_ns",
            "phase_durations_ns",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("live lifecycle receipt fields differ")
        row = dict(value)
        gpu_uuids = row.pop("gpu_uuids")
        if type(gpu_uuids) is not list:
            raise TypeError("live lifecycle GPU UUIDs must be an array")
        edges = row.get("phase_edges_ns")
        durations = row.get("phase_durations_ns")
        if type(edges) is not dict or set(edges) != set(_LIFECYCLE_EDGE_NAMES):
            raise ValueError("live lifecycle phase edge coverage differs")
        if type(durations) is not dict or set(durations) != set(
            _LIFECYCLE_DURATION_NAMES
        ):
            raise ValueError("live lifecycle phase duration coverage differs")
        # Canonical JSON sorts object keys.  Restore the protocol order before
        # constructing the typed receipt so durable reopen and the in-memory
        # producer validate against the same exact phase sequence.
        row["phase_edges_ns"] = {name: edges[name] for name in _LIFECYCLE_EDGE_NAMES}
        row["phase_durations_ns"] = {
            name: durations[name] for name in _LIFECYCLE_DURATION_NAMES
        }
        live_run_receipt = CanonicalJsonProofBinding.from_dict(
            row.pop("live_run_receipt")
        )
        launch_admission = row.pop("formal_launch_admission")
        launch_consumption = row.pop("formal_launch_consumption")
        budget_consumption = row.pop("budget_consumption")
        return cls(
            **row,
            live_run_receipt=live_run_receipt,
            formal_launch_admission=(
                None
                if launch_admission is None
                else CanonicalJsonProofBinding.from_dict(launch_admission)
            ),
            formal_launch_consumption=(
                None
                if launch_consumption is None
                else CanonicalJsonProofBinding.from_dict(launch_consumption)
            ),
            budget_consumption=(
                None
                if budget_consumption is None
                else CanonicalJsonProofBinding.from_dict(budget_consumption)
            ),
            gpu_uuids=tuple(gpu_uuids),
        )


def _publish_unsigned_lifecycle_timing_receipt(
    *,
    output_path: str | Path,
    live_run_receipt: CanonicalJsonProofBinding,
    binding: NativeTerminalRunBinding,
    config: RunConfig,
    evidence_flush_started_ns: int,
    evidence_flush_finished_ns: int,
) -> CanonicalJsonProofBinding:
    run_receipt = UnsignedPinnedSglangServingRunReceipt.from_dict(
        live_run_receipt.reopen()
    )
    ready = run_receipt.ready_gpu_snapshot.reopen()
    events = run_receipt.native_lifecycle_events
    edges = {
        "execution_started_ns": run_receipt.execution_started_ns,
        "server_ready_ns": int(ready["captured_ns"]),
        "begin_started_ns": events.begin_started_ns,
        "begin_finished_ns": events.begin_finished_ns,
        "warmup_started_ns": events.warmup_started_ns,
        "warmup_finished_ns": events.warmup_finished_ns,
        "reset_started_ns": events.reset_started_ns,
        "reset_finished_ns": events.reset_finished_ns,
        "scored_executor_started_ns": events.scored_started_ns,
        "scored_request_started_ns": run_receipt.scored_started_ns,
        "scored_request_finished_ns": run_receipt.scored_finished_ns,
        "scored_executor_finished_ns": events.scored_finished_ns,
        "finalize_started_ns": events.finalize_started_ns,
        "finalize_finished_ns": events.finalize_finished_ns,
        "terminal_published_ns": events.terminal_published_ns,
        "itl_pointer_published_ns": events.itl_pointer_published_ns,
        "native_terminal_finished_ns": run_receipt.native_terminal_finished_ns,
        "process_exited_ns": run_receipt.process_exited_ns,
        "process_group_empty_checked_ns": (run_receipt.process_group_empty_checked_ns),
        "evidence_flush_started_ns": evidence_flush_started_ns,
        "evidence_flush_finished_ns": evidence_flush_finished_ns,
    }
    receipt = UnsignedPinnedSglangLifecycleTimingReceipt(
        schema_version=1,
        kind="unsigned_pinned_sglang_lifecycle_timing_receipt",
        protocol_sha256=PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        live_run_receipt=live_run_receipt,
        formal_launch_admission=run_receipt.formal_launch_admission,
        formal_launch_consumption=run_receipt.formal_launch_consumption,
        budget_consumption=run_receipt.budget_consumption,
        run_binding_sha256=canonical_sha256(binding.begin_payload()),
        run_id=binding.run_id,
        run_nonce_sha256=binding.run_nonce_sha256,
        execution_plan_sha256=binding.execution_plan_sha256,
        rank_config_sha256=binding.rank_config_sha256,
        attempt_id=binding.attempt_id,
        method=binding.method,
        inventory_sha256=run_receipt.inventory_sha256,
        gpu_uuids=run_receipt.gpu_uuids,
        telemetry_detail=config.runtime.telemetry_detail,
        phase_edges_ns=edges,
        phase_durations_ns=_lifecycle_durations(
            edges, telemetry_detail=config.runtime.telemetry_detail
        ),
    )
    destination = _absolute_output_path("live lifecycle timing", output_path)
    publish_canonical_json_no_replace(destination, receipt.to_dict())
    return CanonicalJsonProofBinding.bind(destination, semantic_sha256=receipt.sha256)


def validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
    receipt_path: str | Path | CanonicalJsonProofBinding,
    *,
    expected_live_run_receipt: CanonicalJsonProofBinding,
    expected_binding: NativeTerminalRunBinding,
    expected_telemetry_detail: Literal["headline", "profile"],
) -> UnsignedPinnedSglangLifecycleTimingReceipt:
    """Deep-reopen raw integer timing without promoting it to authority."""

    timing_binding = (
        receipt_path
        if type(receipt_path) is CanonicalJsonProofBinding
        else CanonicalJsonProofBinding.bind(receipt_path)
    )
    assert isinstance(timing_binding, CanonicalJsonProofBinding)
    timing = UnsignedPinnedSglangLifecycleTimingReceipt.from_dict(
        timing_binding.reopen()
    )
    run_receipt = UnsignedPinnedSglangServingRunReceipt.from_dict(
        expected_live_run_receipt.reopen()
    )
    events = run_receipt.native_lifecycle_events
    ready = run_receipt.ready_gpu_snapshot.reopen()
    expected_binding.validate()
    # Rename the two source-collector scored edges and add first-party outer
    # process/evidence edges.  No duration is accepted from the caller.
    expected_edges = {
        "execution_started_ns": run_receipt.execution_started_ns,
        "server_ready_ns": int(ready["captured_ns"]),
        "begin_started_ns": events.begin_started_ns,
        "begin_finished_ns": events.begin_finished_ns,
        "warmup_started_ns": events.warmup_started_ns,
        "warmup_finished_ns": events.warmup_finished_ns,
        "reset_started_ns": events.reset_started_ns,
        "reset_finished_ns": events.reset_finished_ns,
        "scored_executor_started_ns": events.scored_started_ns,
        "scored_request_started_ns": run_receipt.scored_started_ns,
        "scored_request_finished_ns": run_receipt.scored_finished_ns,
        "scored_executor_finished_ns": events.scored_finished_ns,
        "finalize_started_ns": events.finalize_started_ns,
        "finalize_finished_ns": events.finalize_finished_ns,
        "terminal_published_ns": events.terminal_published_ns,
        "itl_pointer_published_ns": events.itl_pointer_published_ns,
        "native_terminal_finished_ns": run_receipt.native_terminal_finished_ns,
        "process_exited_ns": run_receipt.process_exited_ns,
        "process_group_empty_checked_ns": (run_receipt.process_group_empty_checked_ns),
        "evidence_flush_started_ns": timing.phase_edges_ns["evidence_flush_started_ns"],
        "evidence_flush_finished_ns": timing.phase_edges_ns[
            "evidence_flush_finished_ns"
        ],
    }
    if (
        timing_binding.semantic_sha256 != timing.sha256
        or timing.live_run_receipt != expected_live_run_receipt
        or timing.formal_launch_admission != run_receipt.formal_launch_admission
        or timing.formal_launch_consumption != run_receipt.formal_launch_consumption
        or timing.budget_consumption != run_receipt.budget_consumption
        or run_receipt.sha256 != expected_live_run_receipt.semantic_sha256
        or timing.run_binding_sha256
        != canonical_sha256(expected_binding.begin_payload())
        or timing.run_id != expected_binding.run_id
        or timing.run_nonce_sha256 != expected_binding.run_nonce_sha256
        or timing.execution_plan_sha256 != expected_binding.execution_plan_sha256
        or timing.rank_config_sha256 != expected_binding.rank_config_sha256
        or timing.attempt_id != expected_binding.attempt_id
        or timing.method != expected_binding.method
        or timing.inventory_sha256 != run_receipt.inventory_sha256
        or timing.gpu_uuids != run_receipt.gpu_uuids
        or timing.telemetry_detail != expected_telemetry_detail
        or timing.phase_edges_ns != expected_edges
        or timing.phase_durations_ns
        != _lifecycle_durations(
            expected_edges, telemetry_detail=expected_telemetry_detail
        )
        or timing.phase_edges_ns["evidence_flush_started_ns"]
        < run_receipt.process_group_empty_checked_ns
        or timing.phase_edges_ns["evidence_flush_finished_ns"]
        < timing.phase_edges_ns["evidence_flush_started_ns"]
    ):
        raise ValueError("live lifecycle timing differs from first-party evidence")
    run_receipt.terminal_artifact.reopen()
    run_receipt.native_itl_pointer_artifact.reopen()
    run_receipt.before_gpu_snapshot.reopen()
    run_receipt.ready_gpu_snapshot.reopen()
    run_receipt.after_gpu_snapshot.reopen()
    if run_receipt.formal_launch_admission is not None:
        run_receipt.formal_launch_admission.reopen()
        assert run_receipt.formal_launch_consumption is not None
        run_receipt.formal_launch_consumption.reopen()
        assert run_receipt.budget_consumption is not None
        run_receipt.budget_consumption.reopen()
    run_receipt.server_log.reopen(label="live lifecycle server log")
    return timing


def _lifecycle_control_lineage_sha256(
    *,
    timing_binding: CanonicalJsonProofBinding,
    timing: UnsignedPinnedSglangLifecycleTimingReceipt,
    live_run_receipt: CanonicalJsonProofBinding,
    native_result_proof: CanonicalJsonProofBinding,
    registry_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "pinned_sglang_lifecycle_timing_control_lineage",
            "timing_raw_sha256": timing_binding.raw_sha256,
            "timing_semantic_sha256": timing_binding.semantic_sha256,
            "live_run_receipt_sha256": live_run_receipt.semantic_sha256,
            "native_result_proof_sha256": native_result_proof.semantic_sha256,
            "formal_launch_admission": (
                None
                if timing.formal_launch_admission is None
                else timing.formal_launch_admission.to_dict()
            ),
            "formal_launch_consumption": (
                None
                if timing.formal_launch_consumption is None
                else timing.formal_launch_consumption.to_dict()
            ),
            "budget_consumption": (
                None
                if timing.budget_consumption is None
                else timing.budget_consumption.to_dict()
            ),
            "run_binding_sha256": timing.run_binding_sha256,
            "run_nonce_sha256": timing.run_nonce_sha256,
            "execution_plan_sha256": timing.execution_plan_sha256,
            "rank_config_sha256": timing.rank_config_sha256,
            "attempt_id": timing.attempt_id,
            "inventory_sha256": timing.inventory_sha256,
            "registry_sha256": registry_sha256,
            "phase_edges_ns": timing.phase_edges_ns,
            "phase_durations_ns": timing.phase_durations_ns,
        }
    )


_VERIFIED_LIFECYCLE_TIMING_PROOF_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedPinnedSglangLifecycleTimingProof:
    """Verifier-owned phase durations from raw live and terminal evidence."""

    raw_timing_sha256: str
    live_run_receipt_sha256: str
    native_result_proof_sha256: str
    run_binding_sha256: str
    run_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    attempt_id: str
    method: str
    inventory_sha256: str
    registry_sha256: str
    root_manifest_sha256: str
    hardware_envelope_sha256: str
    gpu_uuids: tuple[str, ...]
    telemetry_detail: Literal["headline", "profile"]
    phase_edges_ns: tuple[tuple[str, int], ...]
    phase_durations_ns: tuple[tuple[str, int], ...]
    control_envelope_sha256: str
    replay_reservation_sha256: str

    def __init__(
        self,
        *,
        timing_binding: CanonicalJsonProofBinding,
        timing: UnsignedPinnedSglangLifecycleTimingReceipt,
        native_result_proof: CanonicalJsonProofBinding,
        registry_sha256: str,
        root_manifest_sha256: str,
        hardware_envelope_sha256: str,
        control_envelope_sha256: str,
        replay_reservation_sha256: str,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_LIFECYCLE_TIMING_PROOF_SENTINEL:
            raise TypeError("verified lifecycle timing proof is verifier-owned")
        values = {
            "raw_timing_sha256": timing.sha256,
            "live_run_receipt_sha256": timing.live_run_receipt.semantic_sha256,
            "native_result_proof_sha256": native_result_proof.semantic_sha256,
            "run_binding_sha256": timing.run_binding_sha256,
            "run_id": timing.run_id,
            "run_nonce_sha256": timing.run_nonce_sha256,
            "execution_plan_sha256": timing.execution_plan_sha256,
            "rank_config_sha256": timing.rank_config_sha256,
            "attempt_id": timing.attempt_id,
            "method": timing.method,
            "inventory_sha256": timing.inventory_sha256,
            "registry_sha256": registry_sha256,
            "root_manifest_sha256": root_manifest_sha256,
            "hardware_envelope_sha256": hardware_envelope_sha256,
            "gpu_uuids": timing.gpu_uuids,
            "telemetry_detail": timing.telemetry_detail,
            "phase_edges_ns": tuple(timing.phase_edges_ns.items()),
            "phase_durations_ns": tuple(timing.phase_durations_ns.items()),
            "control_envelope_sha256": control_envelope_sha256,
            "replay_reservation_sha256": replay_reservation_sha256,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if timing_binding.semantic_sha256 != self.raw_timing_sha256:
            raise ValueError("verified lifecycle timing semantic identity differs")
        self.__post_init__()

    def __post_init__(self) -> None:
        for label, value in (
            ("raw timing", self.raw_timing_sha256),
            ("live run", self.live_run_receipt_sha256),
            ("native result", self.native_result_proof_sha256),
            ("run binding", self.run_binding_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("rank config", self.rank_config_sha256),
            ("inventory", self.inventory_sha256),
            ("registry", self.registry_sha256),
            ("release root", self.root_manifest_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("control envelope", self.control_envelope_sha256),
            ("replay reservation", self.replay_reservation_sha256),
        ):
            _require_sha256(f"verified lifecycle {label}", value)
        for label, value in (
            ("run ID", self.run_id),
            ("attempt ID", self.attempt_id),
            ("method", self.method),
        ):
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"verified lifecycle {label} is invalid")
        if (
            tuple(name for name, _ in self.phase_edges_ns) != _LIFECYCLE_EDGE_NAMES
            or tuple(name for name, _ in self.phase_durations_ns)
            != _LIFECYCLE_DURATION_NAMES
            or any(
                type(value) is not int or value < 0
                for _, value in (*self.phase_edges_ns, *self.phase_durations_ns)
            )
        ):
            raise ValueError("verified lifecycle phase coverage is invalid")

    @property
    def phase_durations(self) -> dict[str, int]:
        return dict(self.phase_durations_ns)

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "raw_timing_sha256": self.raw_timing_sha256,
                "live_run_receipt_sha256": self.live_run_receipt_sha256,
                "native_result_proof_sha256": self.native_result_proof_sha256,
                "run_binding_sha256": self.run_binding_sha256,
                "run_id": self.run_id,
                "run_nonce_sha256": self.run_nonce_sha256,
                "execution_plan_sha256": self.execution_plan_sha256,
                "rank_config_sha256": self.rank_config_sha256,
                "attempt_id": self.attempt_id,
                "method": self.method,
                "inventory_sha256": self.inventory_sha256,
                "registry_sha256": self.registry_sha256,
                "root_manifest_sha256": self.root_manifest_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "gpu_uuids": list(self.gpu_uuids),
                "telemetry_detail": self.telemetry_detail,
                "phase_edges_ns": dict(self.phase_edges_ns),
                "phase_durations_ns": dict(self.phase_durations_ns),
                "control_envelope_sha256": self.control_envelope_sha256,
                "replay_reservation_sha256": self.replay_reservation_sha256,
            }
        )


@dataclass(frozen=True)
class PinnedSglangLifecycleTimingProofArtifact:
    """Durable local trust lift for one unsigned lifecycle receipt."""

    schema_version: Literal[1]
    kind: Literal["pinned_sglang_lifecycle_timing_proof_artifact"]
    raw_timing: CanonicalJsonProofBinding
    live_run_receipt: CanonicalJsonProofBinding
    native_result_proof: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    expected_inventory_sha256: str
    expected_registry_sha256: str
    expected_root_manifest_sha256: str
    verified_proof_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "pinned_sglang_lifecycle_timing_proof_artifact"
        ):
            raise ValueError("lifecycle timing proof artifact schema is unsupported")
        for label, value in (
            ("raw timing", self.raw_timing),
            ("live run", self.live_run_receipt),
            ("native result", self.native_result_proof),
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError(f"lifecycle proof {label} binding is invalid")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("lifecycle proof requires an exact control envelope")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("lifecycle proof requires an exact replay reservation")
        for label, value in (
            ("inventory", self.expected_inventory_sha256),
            ("registry", self.expected_registry_sha256),
            ("release root", self.expected_root_manifest_sha256),
            ("verified proof", self.verified_proof_sha256),
        ):
            _require_sha256(f"lifecycle proof {label}", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_timing": self.raw_timing.to_dict(),
            "live_run_receipt": self.live_run_receipt.to_dict(),
            "native_result_proof": self.native_result_proof.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "expected_inventory_sha256": self.expected_inventory_sha256,
            "expected_registry_sha256": self.expected_registry_sha256,
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
            "verified_proof_sha256": self.verified_proof_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> PinnedSglangLifecycleTimingProofArtifact:
        fields = {
            "schema_version",
            "kind",
            "raw_timing",
            "live_run_receipt",
            "native_result_proof",
            "control_attestation",
            "replay_reservation",
            "expected_inventory_sha256",
            "expected_registry_sha256",
            "expected_root_manifest_sha256",
            "verified_proof_sha256",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("lifecycle timing proof artifact fields differ")
        payload = dict(value)
        raw_timing = CanonicalJsonProofBinding.from_dict(payload.pop("raw_timing"))
        live_run_receipt = CanonicalJsonProofBinding.from_dict(
            payload.pop("live_run_receipt")
        )
        native_result_proof = CanonicalJsonProofBinding.from_dict(
            payload.pop("native_result_proof")
        )
        control_attestation = ControlArtifactAttestation.from_dict(
            payload.pop("control_attestation")
        )
        replay_reservation = ChallengeReplayReservationBinding.from_dict(
            payload.pop("replay_reservation")
        )
        return cls(
            **payload,
            raw_timing=raw_timing,
            live_run_receipt=live_run_receipt,
            native_result_proof=native_result_proof,
            control_attestation=control_attestation,
            replay_reservation=replay_reservation,
        )


def _validate_lifecycle_proof_sources(
    *,
    raw_timing: CanonicalJsonProofBinding,
    live_run_receipt: CanonicalJsonProofBinding,
    native_result_proof: CanonicalJsonProofBinding,
    expected_binding: NativeTerminalRunBinding,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_telemetry_detail: Literal["headline", "profile"],
    now_ns: int,
) -> tuple[
    UnsignedPinnedSglangLifecycleTimingReceipt,
    NativeTerminalResultProjection,
    NativeTerminalResultProofArtifact,
]:
    timing = validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
        raw_timing,
        expected_live_run_receipt=live_run_receipt,
        expected_binding=expected_binding,
        expected_telemetry_detail=expected_telemetry_detail,
    )
    native_binding = CanonicalJsonProofBinding.bind(
        native_result_proof.absolute_path,
        semantic_sha256=native_result_proof.semantic_sha256,
    )
    if native_binding != native_result_proof:
        raise ValueError("lifecycle proof native result binding changed")
    native_artifact = NativeTerminalResultProofArtifact.from_dict(
        native_result_proof.reopen()
    )
    if native_artifact.sha256 != native_result_proof.semantic_sha256:
        raise ValueError("lifecycle proof native result semantic identity changed")
    native_result = validate_native_terminal_result_proof_artifact(
        native_result_proof.absolute_path,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_execution_plan_sha256=expected_binding.execution_plan_sha256,
        expected_rank_config_sha256=expected_binding.rank_config_sha256,
        expected_run_id=expected_binding.run_id,
        expected_run_nonce_sha256=expected_binding.run_nonce_sha256,
        expected_attempt_id=expected_binding.attempt_id,
        expected_method=expected_binding.method,
        now_ns=now_ns,
    )
    live = UnsignedPinnedSglangServingRunReceipt.from_dict(live_run_receipt.reopen())
    if (
        live.sha256 != live_run_receipt.semantic_sha256
        or native_artifact.raw_terminal != live.terminal_artifact
        or native_result.terminal_sha256 != live.terminal_sha256
        or timing.inventory_sha256 != expected_inventory_sha256
        or native_artifact.control_attestation.deployment_policy_authorization.bundle.hardware_envelope_sha256_allowlist
        != (native_artifact.control_attestation.hardware_envelope_sha256,)
    ):
        raise ValueError("lifecycle proof source DAG differs")
    return timing, native_result, native_artifact


def _verified_lifecycle_timing_proof(
    *,
    timing_binding: CanonicalJsonProofBinding,
    timing: UnsignedPinnedSglangLifecycleTimingReceipt,
    native_result_proof: CanonicalJsonProofBinding,
    registry_sha256: str,
    root_manifest_sha256: str,
    hardware_envelope_sha256: str,
    control_envelope_sha256: str,
    replay_reservation_sha256: str,
) -> VerifiedPinnedSglangLifecycleTimingProof:
    return VerifiedPinnedSglangLifecycleTimingProof(
        timing_binding=timing_binding,
        timing=timing,
        native_result_proof=native_result_proof,
        registry_sha256=registry_sha256,
        root_manifest_sha256=root_manifest_sha256,
        hardware_envelope_sha256=hardware_envelope_sha256,
        control_envelope_sha256=control_envelope_sha256,
        replay_reservation_sha256=replay_reservation_sha256,
        _verification_tag=_VERIFIED_LIFECYCLE_TIMING_PROOF_SENTINEL,
    )


def build_pinned_sglang_lifecycle_timing_control_subject(
    raw_timing_receipt_path: str,
    *,
    live_run_receipt_path: str,
    native_result_proof_artifact_path: str,
    expected_binding: NativeTerminalRunBinding,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_telemetry_detail: Literal["headline", "profile"],
    now_ns: int,
) -> ControlArtifactSubject:
    """Build the local signing subject from the exact pulled lifecycle DAG."""

    timing_binding = CanonicalJsonProofBinding.bind(raw_timing_receipt_path)
    live_binding = CanonicalJsonProofBinding.bind(live_run_receipt_path)
    native_binding = CanonicalJsonProofBinding.bind(native_result_proof_artifact_path)
    timing, _native_result, _native_artifact = _validate_lifecycle_proof_sources(
        raw_timing=timing_binding,
        live_run_receipt=live_binding,
        native_result_proof=native_binding,
        expected_binding=expected_binding,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_telemetry_detail=expected_telemetry_detail,
        now_ns=now_ns,
    )
    lineage = _lifecycle_control_lineage_sha256(
        timing_binding=timing_binding,
        timing=timing,
        live_run_receipt=live_binding,
        native_result_proof=native_binding,
        registry_sha256=expected_registry_sha256,
    )
    return ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=timing_binding.raw_sha256,
        protocol_sha256=PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256,
        registry_sha256=expected_registry_sha256,
        lineage_sha256=lineage,
    )


def publish_pinned_sglang_lifecycle_timing_proof_artifact(
    raw_timing_receipt_path: str,
    *,
    live_run_receipt_path: str,
    native_result_proof_artifact_path: str,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_binding: NativeTerminalRunBinding,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_telemetry_detail: Literal["headline", "profile"],
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    """Locally trust-lift one raw timing receipt after terminal proof exists."""

    for label, value in (
        ("lifecycle expected inventory", expected_inventory_sha256),
        ("lifecycle expected registry", expected_registry_sha256),
        ("lifecycle expected root", expected_root_manifest_sha256),
    ):
        _require_sha256(label, value)
    if type(control_attestation) is not ControlArtifactAttestation:
        raise TypeError("lifecycle proof requires an exact control envelope")
    if type(replay_store) is not ChallengeReplayStore:
        raise TypeError("lifecycle proof requires an exact replay store")
    output = _absolute_output_path("lifecycle proof artifact", proof_artifact_path)
    if output.exists():
        raise ValueError("lifecycle proof artifact output already exists")
    timing_binding = CanonicalJsonProofBinding.bind(raw_timing_receipt_path)
    live_binding = CanonicalJsonProofBinding.bind(live_run_receipt_path)
    native_binding = CanonicalJsonProofBinding.bind(native_result_proof_artifact_path)
    timing, _native_result, native_artifact = _validate_lifecycle_proof_sources(
        raw_timing=timing_binding,
        live_run_receipt=live_binding,
        native_result_proof=native_binding,
        expected_binding=expected_binding,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_telemetry_detail=expected_telemetry_detail,
        now_ns=now_ns,
    )
    subject = control_attestation.subject
    expected_lineage = _lifecycle_control_lineage_sha256(
        timing_binding=timing_binding,
        timing=timing,
        live_run_receipt=live_binding,
        native_result_proof=native_binding,
        registry_sha256=expected_registry_sha256,
    )
    hardware_envelope_sha256 = (
        native_artifact.control_attestation.hardware_envelope_sha256
    )
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != timing_binding.raw_sha256
        or subject.protocol_sha256 != PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_registry_sha256
        or subject.lineage_sha256 != expected_lineage
        or control_attestation.hardware_envelope_sha256 != hardware_envelope_sha256
        or control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("lifecycle proof control subject is not exact")
    verified_controls = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified_controls,
        reserved_ns=now_ns,
    )
    reservation = replay_store.bind_reservation(reservation_sha256)
    verified = _verified_lifecycle_timing_proof(
        timing_binding=timing_binding,
        timing=timing,
        native_result_proof=native_binding,
        registry_sha256=expected_registry_sha256,
        root_manifest_sha256=expected_root_manifest_sha256,
        hardware_envelope_sha256=hardware_envelope_sha256,
        control_envelope_sha256=verified_controls[0].envelope_sha256,
        replay_reservation_sha256=reservation_sha256,
    )
    artifact = PinnedSglangLifecycleTimingProofArtifact(
        schema_version=1,
        kind="pinned_sglang_lifecycle_timing_proof_artifact",
        raw_timing=timing_binding,
        live_run_receipt=live_binding,
        native_result_proof=native_binding,
        control_attestation=control_attestation,
        replay_reservation=reservation,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        verified_proof_sha256=verified.sha256,
    )
    publish_canonical_json_no_replace(output, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output, semantic_sha256=artifact.sha256)


def validate_pinned_sglang_lifecycle_timing_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_binding: NativeTerminalRunBinding,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    expected_gpu_uuids: tuple[str, ...],
    expected_telemetry_detail: Literal["headline", "profile"],
    now_ns: int,
) -> VerifiedPinnedSglangLifecycleTimingProof:
    """Deep-open a durable lifecycle proof without consuming replay again."""

    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = PinnedSglangLifecycleTimingProofArtifact.from_dict(binding.reopen())
    if (
        artifact.sha256 != binding.semantic_sha256
        or artifact.expected_inventory_sha256 != expected_inventory_sha256
        or artifact.expected_registry_sha256 != expected_registry_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("lifecycle proof artifact identity differs")
    timing, _native_result, native_artifact = _validate_lifecycle_proof_sources(
        raw_timing=artifact.raw_timing,
        live_run_receipt=artifact.live_run_receipt,
        native_result_proof=artifact.native_result_proof,
        expected_binding=expected_binding,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_telemetry_detail=expected_telemetry_detail,
        now_ns=now_ns,
    )
    if timing.gpu_uuids != expected_gpu_uuids:
        raise ValueError("lifecycle proof GPU assignment differs")
    reserved = artifact.replay_reservation.revalidate()
    if type(now_ns) is not int or now_ns < artifact.replay_reservation.reserved_ns:
        raise ValueError("lifecycle proof validation predates reservation")
    verified_control = verify_release_control_artifact_attestation(
        artifact.control_attestation,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=artifact.replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    expected_reservation_sha256 = control_challenge_reservation_sha256(
        (verified_control,),
        reserved_ns=artifact.replay_reservation.reserved_ns,
    )
    expected_challenges = tuple(
        sorted(
            {
                verified_control.challenge_sha256,
                verified_control.deployment_policy_challenge_sha256,
            }
        )
    )
    expected_lineage = _lifecycle_control_lineage_sha256(
        timing_binding=artifact.raw_timing,
        timing=timing,
        live_run_receipt=artifact.live_run_receipt,
        native_result_proof=artifact.native_result_proof,
        registry_sha256=expected_registry_sha256,
    )
    native_control = native_artifact.control_attestation
    subject = artifact.control_attestation.subject
    if (
        reserved != expected_challenges
        or artifact.replay_reservation.reservation_sha256 != expected_reservation_sha256
        or subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != artifact.raw_timing.raw_sha256
        or subject.protocol_sha256 != PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_registry_sha256
        or subject.lineage_sha256 != expected_lineage
        or artifact.control_attestation.hardware_envelope_sha256
        != native_control.hardware_envelope_sha256
        or artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("lifecycle proof control or replay identity differs")
    verified = _verified_lifecycle_timing_proof(
        timing_binding=artifact.raw_timing,
        timing=timing,
        native_result_proof=artifact.native_result_proof,
        registry_sha256=expected_registry_sha256,
        root_manifest_sha256=expected_root_manifest_sha256,
        hardware_envelope_sha256=native_control.hardware_envelope_sha256,
        control_envelope_sha256=verified_control.envelope_sha256,
        replay_reservation_sha256=expected_reservation_sha256,
    )
    if verified.sha256 != artifact.verified_proof_sha256:
        raise ValueError("lifecycle proof derived identity changed")
    return verified


_VALIDATED_LIVE_RUN_SENTINEL = object()


@dataclass(frozen=True, init=False)
class ValidatedUnsignedPinnedSglangServingRun:
    """Verifier-created projection of all three unsigned serving artifacts."""

    receipt_binding: CanonicalJsonProofBinding
    receipt: UnsignedPinnedSglangServingRunReceipt
    terminal_evidence: ValidatedNativeTerminalEvidence

    def __init__(
        self,
        *,
        receipt_binding: CanonicalJsonProofBinding,
        receipt: UnsignedPinnedSglangServingRunReceipt,
        terminal_evidence: ValidatedNativeTerminalEvidence,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VALIDATED_LIVE_RUN_SENTINEL:
            raise TypeError("validated live serving run requires first-party replay")
        object.__setattr__(self, "receipt_binding", receipt_binding)
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "terminal_evidence", terminal_evidence)

    @property
    def collection(self) -> UnsignedNativeTerminalCollection:
        return UnsignedNativeTerminalCollection(
            terminal_artifact=self.receipt.terminal_artifact,
            native_itl_pointer_artifact=self.receipt.native_itl_pointer_artifact,
        )


@dataclass(frozen=True)
class PinnedSglangServingRunSpec:
    """One sealed-dispatch row consumed by the two-server group runner."""

    launch_manifest: CanonicalJsonProofBinding
    binding: NativeTerminalRunBinding
    warmup_requests: tuple[BoundServingRequest, ...]
    scored_requests: tuple[BoundServingRequest, ...]
    terminal_output_path: str
    native_itl_pointer_output_path: str
    live_run_receipt_output_path: str
    server_log_output_path: str
    verified_native_graph_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None
    expected_graph_source_identity_sha256: str | None = None
    verified_eagle3_e0_execution_authority: (
        VerifiedEagle3E0ExecutionAuthority | None
    ) = None
    expected_eagle3_source_identity_sha256: str | None = None
    verified_chronobelief_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None
    expected_chronobelief_source_identity_sha256: str | None = None
    lifecycle_timing_output_path: str | None = None

    def validate(self, *, timeout_seconds: float) -> CompileLaunchManifest:
        if type(self.launch_manifest) is not CanonicalJsonProofBinding:
            raise TypeError("serving group requires a path-bound launch manifest")
        launch = CompileLaunchManifest.load(self.launch_manifest.absolute_path)
        if launch.sha256 != self.launch_manifest.semantic_sha256:
            raise ValueError("serving group launch manifest identity changed")
        self.binding.validate()
        config = load_run_config(launch.run_config_path)
        if (
            config.method != self.binding.method
            or config.model.target != launch.target_model_id
            or config.runtime.tensor_parallel_size != 1
            or config.runtime.data_parallel_size != 1
            or config.runtime.node_count != 1
            or len(launch.gpu_uuids) != 1
        ):
            raise ValueError("serving group requires exact TP1/DP1 rows")
        _graph_execution_authority(
            runtime=config.runtime,
            server_argv=launch.server_argv,
            verified_gpu_proof=self.verified_native_graph_gpu_proof,
            expected_source_identity_sha256=(
                self.expected_graph_source_identity_sha256
            ),
            inventory_sha256=launch.inventory_sha256,
            gpu_uuids=launch.gpu_uuids,
        )
        _mechanism_execution_authority(
            config=config,
            server_argv=launch.server_argv,
        )
        _eagle3_execution_authority(
            config=config,
            verified_authority=self.verified_eagle3_e0_execution_authority,
            expected_source_identity_sha256=(
                self.expected_eagle3_source_identity_sha256
            ),
            inventory_sha256=launch.inventory_sha256,
            gpu_uuids=launch.gpu_uuids,
        )
        _chronobelief_execution_authority(
            config=config,
            verified_gpu_proof=self.verified_chronobelief_gpu_proof,
            expected_source_identity_sha256=(
                self.expected_chronobelief_source_identity_sha256
            ),
            inventory_sha256=launch.inventory_sha256,
            gpu_uuids=launch.gpu_uuids,
        )
        if self.lifecycle_timing_output_path is not None:
            _absolute_output_path(
                "serving group lifecycle timing", self.lifecycle_timing_output_path
            )
        _validate_phase_inputs(
            "warmup",
            self.warmup_requests,
            expected_ids=self.binding.warmup_request_ids,
            timeout_seconds=timeout_seconds,
        )
        _validate_phase_inputs(
            "scored",
            self.scored_requests,
            expected_ids=self.binding.scored_request_ids,
            timeout_seconds=timeout_seconds,
        )
        return launch


@dataclass(frozen=True)
class UnsignedPinnedSglangServingGroupReceipt:
    """Exact two-server overlap and cleanup receipt; never formal authority."""

    schema_version: Literal[1]
    kind: Literal["unsigned_pinned_sglang_concurrent_group_receipt"]
    protocol_sha256: str
    formal_execution_authorized: Literal[False]
    inventory_sha256: str
    gpu_uuids: tuple[str, str]
    localhost_ports: tuple[int, int]
    server_process_group_ids: tuple[int, int]
    ready_compute_process_rows_sha256: str
    launch_manifests: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]
    run_binding_sha256s: tuple[str, str]
    live_run_receipts: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]
    before_gpu_snapshot: CanonicalJsonProofBinding
    ready_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    shared_scored_origin_ns: int
    overlap_started_ns: int
    overlap_finished_ns: int
    overlap_duration_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "unsigned_pinned_sglang_concurrent_group_receipt"
            or self.protocol_sha256 != PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256
            or self.formal_execution_authorized is not False
        ):
            raise ValueError("live serving group receipt schema differs")
        _require_sha256("live serving group inventory", self.inventory_sha256)
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
            or any(not value.startswith("GPU-") for value in self.gpu_uuids)
            or type(self.localhost_ports) is not tuple
            or len(self.localhost_ports) != 2
            or len(set(self.localhost_ports)) != 2
            or any(
                type(value) is not int or not 1024 <= value <= 65535
                for value in self.localhost_ports
            )
            or type(self.server_process_group_ids) is not tuple
            or len(self.server_process_group_ids) != 2
            or len(set(self.server_process_group_ids)) != 2
            or any(
                type(value) is not int or value < 1
                for value in self.server_process_group_ids
            )
            or type(self.launch_manifests) is not tuple
            or len(self.launch_manifests) != 2
            or type(self.live_run_receipts) is not tuple
            or len(self.live_run_receipts) != 2
            or type(self.run_binding_sha256s) is not tuple
            or len(self.run_binding_sha256s) != 2
        ):
            raise ValueError("live serving group cardinality/binding is invalid")
        _require_sha256(
            "live serving group ready process rows",
            self.ready_compute_process_rows_sha256,
        )
        for digest in self.run_binding_sha256s:
            _require_sha256("live serving group run binding", digest)
        for value in (
            *self.launch_manifests,
            *self.live_run_receipts,
            self.before_gpu_snapshot,
            self.ready_gpu_snapshot,
            self.after_gpu_snapshot,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("live serving group lost a path-bound artifact")
        if (
            type(self.shared_scored_origin_ns) is not int
            or type(self.overlap_started_ns) is not int
            or type(self.overlap_finished_ns) is not int
            or type(self.overlap_duration_ns) is not int
            or self.shared_scored_origin_ns < 1
            or self.overlap_started_ns < self.shared_scored_origin_ns
            or self.overlap_finished_ns <= self.overlap_started_ns
            or self.overlap_duration_ns
            != self.overlap_finished_ns - self.overlap_started_ns
        ):
            raise ValueError("live serving group overlap interval is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "formal_execution_authorized": self.formal_execution_authorized,
            "inventory_sha256": self.inventory_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "localhost_ports": list(self.localhost_ports),
            "server_process_group_ids": list(self.server_process_group_ids),
            "ready_compute_process_rows_sha256": (
                self.ready_compute_process_rows_sha256
            ),
            "launch_manifests": [value.to_dict() for value in self.launch_manifests],
            "run_binding_sha256s": list(self.run_binding_sha256s),
            "live_run_receipts": [value.to_dict() for value in self.live_run_receipts],
            "before_gpu_snapshot": self.before_gpu_snapshot.to_dict(),
            "ready_gpu_snapshot": self.ready_gpu_snapshot.to_dict(),
            "after_gpu_snapshot": self.after_gpu_snapshot.to_dict(),
            "shared_scored_origin_ns": self.shared_scored_origin_ns,
            "overlap_started_ns": self.overlap_started_ns,
            "overlap_finished_ns": self.overlap_finished_ns,
            "overlap_duration_ns": self.overlap_duration_ns,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> UnsignedPinnedSglangServingGroupReceipt:
        fields = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "inventory_sha256",
            "gpu_uuids",
            "localhost_ports",
            "server_process_group_ids",
            "ready_compute_process_rows_sha256",
            "launch_manifests",
            "run_binding_sha256s",
            "live_run_receipts",
            "before_gpu_snapshot",
            "ready_gpu_snapshot",
            "after_gpu_snapshot",
            "shared_scored_origin_ns",
            "overlap_started_ns",
            "overlap_finished_ns",
            "overlap_duration_ns",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("live serving group receipt fields differ")
        row = dict(value)
        arrays = {
            name: row.pop(name)
            for name in (
                "gpu_uuids",
                "localhost_ports",
                "server_process_group_ids",
                "launch_manifests",
                "run_binding_sha256s",
                "live_run_receipts",
            )
        }
        if any(type(value) is not list for value in arrays.values()):
            raise TypeError("live serving group receipt arrays are malformed")
        before_gpu_snapshot = CanonicalJsonProofBinding.from_dict(
            row.pop("before_gpu_snapshot")
        )
        ready_gpu_snapshot = CanonicalJsonProofBinding.from_dict(
            row.pop("ready_gpu_snapshot")
        )
        after_gpu_snapshot = CanonicalJsonProofBinding.from_dict(
            row.pop("after_gpu_snapshot")
        )
        return cls(
            **row,
            gpu_uuids=tuple(arrays["gpu_uuids"]),
            localhost_ports=tuple(arrays["localhost_ports"]),
            server_process_group_ids=tuple(arrays["server_process_group_ids"]),
            launch_manifests=tuple(
                CanonicalJsonProofBinding.from_dict(item)
                for item in arrays["launch_manifests"]
            ),
            run_binding_sha256s=tuple(arrays["run_binding_sha256s"]),
            live_run_receipts=tuple(
                CanonicalJsonProofBinding.from_dict(item)
                for item in arrays["live_run_receipts"]
            ),
            before_gpu_snapshot=before_gpu_snapshot,
            ready_gpu_snapshot=ready_gpu_snapshot,
            after_gpu_snapshot=after_gpu_snapshot,
        )


_VALIDATED_LIVE_GROUP_SENTINEL = object()


@dataclass(frozen=True, init=False)
class ValidatedUnsignedPinnedSglangServingGroup:
    receipt_binding: CanonicalJsonProofBinding
    receipt: UnsignedPinnedSglangServingGroupReceipt
    runs: tuple[
        ValidatedUnsignedPinnedSglangServingRun,
        ValidatedUnsignedPinnedSglangServingRun,
    ]

    def __init__(
        self,
        *,
        receipt_binding: CanonicalJsonProofBinding,
        receipt: UnsignedPinnedSglangServingGroupReceipt,
        runs: tuple[
            ValidatedUnsignedPinnedSglangServingRun,
            ValidatedUnsignedPinnedSglangServingRun,
        ],
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VALIDATED_LIVE_GROUP_SENTINEL:
            raise TypeError("validated live serving group requires first-party replay")
        object.__setattr__(self, "receipt_binding", receipt_binding)
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "runs", runs)


class _SharedScoredBarrier:
    def __init__(self) -> None:
        self._arrivals = 0
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()
        self.origin_ns: int | None = None

    async def arrive(self) -> int:
        async with self._lock:
            self._arrivals += 1
            if self._arrivals > 2:
                raise RuntimeError("live serving scored barrier was reused")
            if self._arrivals == 2:
                self.origin_ns = time.monotonic_ns()
                self._event.set()
        await self._event.wait()
        if self.origin_ns is None:  # pragma: no cover - event invariant
            raise RuntimeError("live serving scored barrier lacks an origin")
        return self.origin_ns


def validate_unsigned_pinned_sglang_serving_run_receipt(
    receipt_path: str | Path | CanonicalJsonProofBinding,
    *,
    expected_launch_manifest: CanonicalJsonProofBinding,
    expected_binding: NativeTerminalRunBinding,
    expected_terminal_artifact: CanonicalJsonProofBinding,
    expected_native_itl_pointer_artifact: CanonicalJsonProofBinding,
    expected_scored_request_inputs_sha256: str,
    expected_gpu_uuids: tuple[str, ...],
    expected_inventory_sha256: str,
    expected_physical_assignment_sha256: str,
    expected_experiment_budget_sha256: str,
    expected_tool: PinnedNvidiaSmiTool,
    expected_snapshot_gpu_uuids: tuple[str, ...],
    expected_server_process_group_ids: tuple[int, ...],
    expected_verified_native_graph_gpu_proof: (
        VerifiedNativeRuntimeGpuProof | None
    ) = None,
    expected_graph_source_identity_sha256: str | None = None,
    expected_verified_eagle3_e0_execution_authority: (
        VerifiedEagle3E0ExecutionAuthority | None
    ) = None,
    expected_trusted_single_operator_eagle3_execution_authority: (
        TrustedSingleOperatorEagle3ExecutionAuthority | None
    ) = None,
    expected_eagle3_source_identity_sha256: str | None = None,
    expected_verified_chronobelief_gpu_proof: (
        VerifiedNativeRuntimeGpuProof | None
    ) = None,
    expected_trusted_single_operator_chronobelief_gpu_parity_proof: (
        TrustedSingleOperatorChronoBeliefGpuParityProof | None
    ) = None,
    expected_chronobelief_source_identity_sha256: str | None = None,
    expected_execution_policy: RegisteredServingExecutionPolicy | None = None,
) -> ValidatedUnsignedPinnedSglangServingRun:
    """Deep-reopen one unsigned live run without promoting it to authority."""

    for label, digest in (
        ("scored request inputs", expected_scored_request_inputs_sha256),
        ("inventory", expected_inventory_sha256),
        ("physical assignment", expected_physical_assignment_sha256),
        ("experiment budget", expected_experiment_budget_sha256),
    ):
        _require_sha256(f"expected live serving {label}", digest)
    if (
        type(expected_launch_manifest) is not CanonicalJsonProofBinding
        or type(expected_terminal_artifact) is not CanonicalJsonProofBinding
        or type(expected_native_itl_pointer_artifact) is not CanonicalJsonProofBinding
        or type(expected_binding) is not NativeTerminalRunBinding
        or type(expected_tool) is not PinnedNvidiaSmiTool
        or type(expected_snapshot_gpu_uuids) is not tuple
        or type(expected_server_process_group_ids) is not tuple
    ):
        raise TypeError("live serving replay requires exact path/run bindings")
    expected_tool.revalidate()
    expected_binding.validate()
    for binding in (
        expected_launch_manifest,
        expected_terminal_artifact,
        expected_native_itl_pointer_artifact,
    ):
        binding.reopen()
    receipt_binding = (
        receipt_path
        if type(receipt_path) is CanonicalJsonProofBinding
        else CanonicalJsonProofBinding.bind(receipt_path)
    )
    assert isinstance(receipt_binding, CanonicalJsonProofBinding)
    receipt = UnsignedPinnedSglangServingRunReceipt.from_dict(receipt_binding.reopen())
    if receipt.execution_policy != expected_execution_policy:
        raise ValueError("live serving registered execution policy differs")
    launch = CompileLaunchManifest.load(expected_launch_manifest.absolute_path)
    config = load_run_config(launch.run_config_path)
    (
        execution_policy_sha256,
        cuda_graph_mode,
        native_graph_gpu_proof_sha256,
        native_graph_source_identity_sha256,
    ) = _graph_execution_authority(
        runtime=config.runtime,
        server_argv=launch.server_argv,
        verified_gpu_proof=expected_verified_native_graph_gpu_proof,
        expected_source_identity_sha256=expected_graph_source_identity_sha256,
        inventory_sha256=expected_inventory_sha256,
        gpu_uuids=expected_gpu_uuids,
    )
    _mechanism_execution_authority(
        config=config,
        server_argv=launch.server_argv,
    )
    (
        eagle3_e0_execution_authority_sha256,
        eagle3_compatibility_authority_sha256,
        eagle3_model_selector_sha256,
        eagle3_native_gpu_proof_sha256,
        eagle3_native_source_identity_sha256,
    ) = _eagle3_execution_authority(
        config=config,
        verified_authority=expected_verified_eagle3_e0_execution_authority,
        expected_source_identity_sha256=(expected_eagle3_source_identity_sha256),
        inventory_sha256=expected_inventory_sha256,
        gpu_uuids=expected_gpu_uuids,
        trusted_single_operator_authority=(
            expected_trusted_single_operator_eagle3_execution_authority
        ),
    )
    (
        chronobelief_gpu_proof_sha256,
        chronobelief_source_identity_sha256,
    ) = _chronobelief_execution_authority(
        config=config,
        verified_gpu_proof=expected_verified_chronobelief_gpu_proof,
        expected_source_identity_sha256=(expected_chronobelief_source_identity_sha256),
        inventory_sha256=expected_inventory_sha256,
        gpu_uuids=expected_gpu_uuids,
        trusted_single_operator_proof=(
            expected_trusted_single_operator_chronobelief_gpu_parity_proof
        ),
    )
    expected_server_policy_fields_json = json.dumps(
        _expected_server_execution_policy_fields(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_server_policy_fields_sha256 = hashlib.sha256(
        expected_server_policy_fields_json.encode("utf-8")
    ).hexdigest()
    run_binding_sha256 = canonical_sha256(expected_binding.begin_payload())
    if (
        receipt_binding.semantic_sha256 != receipt.sha256
        or receipt.launch_manifest != expected_launch_manifest
        or receipt.terminal_artifact != expected_terminal_artifact
        or receipt.native_itl_pointer_artifact != expected_native_itl_pointer_artifact
        or receipt.run_binding_sha256 != run_binding_sha256
        or launch.sha256 != expected_launch_manifest.semantic_sha256
        or receipt.patched_sglang_commit != launch.patched_sglang_commit
        or receipt.patched_sglang_tree != launch.patched_sglang_tree
        or receipt.server_argv_sha256 != launch.server_argv_sha256
        or receipt.execution_policy_sha256 != execution_policy_sha256
        or receipt.cuda_graph_mode != cuda_graph_mode
        or receipt.native_graph_gpu_proof_sha256 != native_graph_gpu_proof_sha256
        or receipt.native_graph_source_identity_sha256
        != native_graph_source_identity_sha256
        or receipt.eagle3_e0_execution_authority_sha256
        != eagle3_e0_execution_authority_sha256
        or receipt.eagle3_compatibility_authority_sha256
        != eagle3_compatibility_authority_sha256
        or receipt.eagle3_model_selector_sha256 != eagle3_model_selector_sha256
        or receipt.eagle3_native_gpu_proof_sha256 != eagle3_native_gpu_proof_sha256
        or receipt.eagle3_native_source_identity_sha256
        != eagle3_native_source_identity_sha256
        or receipt.chronobelief_gpu_proof_sha256 != chronobelief_gpu_proof_sha256
        or receipt.chronobelief_source_identity_sha256
        != chronobelief_source_identity_sha256
        or receipt.server_execution_policy_fields_json
        != expected_server_policy_fields_json
        or receipt.server_execution_policy_fields_sha256
        != expected_server_policy_fields_sha256
        or receipt.physical_assignment_sha256 != expected_physical_assignment_sha256
        or receipt.experiment_budget_sha256 != expected_experiment_budget_sha256
        or receipt.inventory_sha256 != expected_inventory_sha256
        or receipt.gpu_uuids != expected_gpu_uuids
        or launch.physical_assignment_sha256 != expected_physical_assignment_sha256
        or launch.experiment_budget_sha256 != expected_experiment_budget_sha256
        or launch.inventory_sha256 != expected_inventory_sha256
        or launch.gpu_uuids != expected_gpu_uuids
        or receipt.snapshot_gpu_uuids != expected_snapshot_gpu_uuids
        or receipt.server_process_group_ids != expected_server_process_group_ids
    ):
        raise ValueError("live serving receipt differs from sealed execution inputs")
    runner_source_body = Path(__file__).resolve().read_bytes()
    if receipt.runner_source_raw_sha256 != hashlib.sha256(
        runner_source_body
    ).hexdigest() or receipt.runner_source_size != len(runner_source_body):
        raise ValueError("live serving runner source identity changed")
    receipt.server_log.reopen(label="live serving server log")
    validate_pinned_sglang_gpu_process_snapshot(
        receipt.before_gpu_snapshot,
        expected_tool=expected_tool,
        expected_gpu_uuids=expected_snapshot_gpu_uuids,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_phase="before",
    )
    ready_snapshot = validate_pinned_sglang_gpu_process_snapshot(
        receipt.ready_gpu_snapshot,
        expected_tool=expected_tool,
        expected_gpu_uuids=expected_snapshot_gpu_uuids,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_phase="ready",
        expected_server_process_group_ids=expected_server_process_group_ids,
    )
    validate_pinned_sglang_gpu_process_snapshot(
        receipt.after_gpu_snapshot,
        expected_tool=expected_tool,
        expected_gpu_uuids=expected_snapshot_gpu_uuids,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_phase="after",
    )
    if (
        canonical_sha256(ready_snapshot["compute_process_rows"])
        != receipt.ready_compute_process_rows_sha256
    ):
        raise ValueError("live serving ready process row identity changed")
    evidence = validate_native_terminal_artifact(
        expected_terminal_artifact.reopen(),
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        expected_binding=expected_binding,
    )
    if (
        evidence.authority_kind != "untrusted_raw_terminal"
        or evidence.terminal_sha256 != receipt.terminal_sha256
        or evidence.begin_receipt.server_process_id != receipt.server_process_id
        or evidence.begin_receipt.server_process_started_ns
        != receipt.server_process_started_ns
    ):
        raise ValueError("live serving terminal/process identity differs")
    completed_scored = tuple(
        row
        for row in evidence.requests
        if row.request_id in expected_binding.scored_request_ids
        and row.terminal_status == "completed"
    )
    if completed_scored:
        actual_started_ns, actual_finished_ns = _reopen_native_scored_interval(
            pointer_artifact=expected_native_itl_pointer_artifact,
            terminal_artifact=expected_terminal_artifact,
            binding=expected_binding,
            terminal_evidence=evidence,
            scored_request_inputs_sha256=expected_scored_request_inputs_sha256,
        )
    elif receipt.schema_version == 2:
        lifecycle = receipt.native_lifecycle_events
        actual_started_ns = lifecycle.scored_started_ns
        actual_finished_ns = lifecycle.scored_finished_ns
    else:
        raise ValueError("legacy live serving has no completed scored requests")
    if (
        receipt.scored_started_ns != actual_started_ns
        or receipt.scored_finished_ns != actual_finished_ns
    ):
        raise ValueError("live serving receipt does not bind native scored interval")
    return ValidatedUnsignedPinnedSglangServingRun(
        receipt_binding=receipt_binding,
        receipt=receipt,
        terminal_evidence=evidence,
        _verification_tag=_VALIDATED_LIVE_RUN_SENTINEL,
    )


def validate_unsigned_pinned_sglang_serving_group_receipt_by_identity(
    receipt_path: str | Path | CanonicalJsonProofBinding,
    *,
    expected_launch_manifests: tuple[
        CanonicalJsonProofBinding, CanonicalJsonProofBinding
    ],
    expected_run_bindings: tuple[NativeTerminalRunBinding, NativeTerminalRunBinding],
    expected_terminal_artifacts: tuple[
        CanonicalJsonProofBinding, CanonicalJsonProofBinding
    ],
    expected_native_itl_pointer_artifacts: tuple[
        CanonicalJsonProofBinding, CanonicalJsonProofBinding
    ],
    expected_live_run_receipts: tuple[
        CanonicalJsonProofBinding, CanonicalJsonProofBinding
    ],
    expected_scored_request_inputs_sha256s: tuple[str, str],
    expected_gpu_uuids: tuple[str, str],
    expected_localhost_ports: tuple[int, int],
    expected_physical_assignment_sha256s: tuple[str, str],
    expected_experiment_budget_sha256s: tuple[str, str],
    expected_tool: PinnedNvidiaSmiTool,
    expected_inventory_sha256: str,
    expected_verified_native_graph_gpu_proofs: tuple[
        VerifiedNativeRuntimeGpuProof | None,
        VerifiedNativeRuntimeGpuProof | None,
    ] = (None, None),
    expected_graph_source_identity_sha256s: tuple[str | None, str | None] = (
        None,
        None,
    ),
    expected_verified_eagle3_e0_execution_authorities: tuple[
        VerifiedEagle3E0ExecutionAuthority | None,
        VerifiedEagle3E0ExecutionAuthority | None,
    ] = (None, None),
    expected_eagle3_source_identity_sha256s: tuple[str | None, str | None] = (
        None,
        None,
    ),
    expected_verified_chronobelief_gpu_proofs: tuple[
        VerifiedNativeRuntimeGpuProof | None,
        VerifiedNativeRuntimeGpuProof | None,
    ] = (None, None),
    expected_chronobelief_source_identity_sha256s: tuple[str | None, str | None] = (
        None,
        None,
    ),
) -> ValidatedUnsignedPinnedSglangServingGroup:
    """Deep-reopen a pulled group using identities, not request payloads."""

    pairs: tuple[tuple[str, object], ...] = (
        ("launch manifests", expected_launch_manifests),
        ("run bindings", expected_run_bindings),
        ("terminal artifacts", expected_terminal_artifacts),
        ("native ITL pointer artifacts", expected_native_itl_pointer_artifacts),
        ("live run receipts", expected_live_run_receipts),
        ("scored request input digests", expected_scored_request_inputs_sha256s),
        ("GPU UUIDs", expected_gpu_uuids),
        ("localhost ports", expected_localhost_ports),
        ("physical assignment digests", expected_physical_assignment_sha256s),
        ("experiment budget digests", expected_experiment_budget_sha256s),
        ("graph GPU proofs", expected_verified_native_graph_gpu_proofs),
        ("graph source identities", expected_graph_source_identity_sha256s),
        (
            "EAGLE3 E0 execution authorities",
            expected_verified_eagle3_e0_execution_authorities,
        ),
        ("EAGLE3 source identities", expected_eagle3_source_identity_sha256s),
        ("ChronoBelief GPU proofs", expected_verified_chronobelief_gpu_proofs),
        (
            "ChronoBelief source identities",
            expected_chronobelief_source_identity_sha256s,
        ),
    )
    if any(type(value) is not tuple or len(value) != 2 for _label, value in pairs):
        raise TypeError("live serving group identity replay requires exact pairs")
    if (
        any(
            type(value) is not CanonicalJsonProofBinding
            for pair in (
                expected_launch_manifests,
                expected_terminal_artifacts,
                expected_native_itl_pointer_artifacts,
                expected_live_run_receipts,
            )
            for value in pair
        )
        or any(
            type(value) is not NativeTerminalRunBinding
            for value in expected_run_bindings
        )
        or type(expected_tool) is not PinnedNvidiaSmiTool
    ):
        raise TypeError("live serving group replay bindings are not source types")
    expected_tool.revalidate()
    _require_sha256("live serving group inventory", expected_inventory_sha256)
    for label, values in (
        ("scored request inputs", expected_scored_request_inputs_sha256s),
        ("physical assignment", expected_physical_assignment_sha256s),
        ("experiment budget", expected_experiment_budget_sha256s),
    ):
        for digest in values:
            _require_sha256(f"live serving group {label}", digest)
    for binding in expected_run_bindings:
        binding.validate()
    launches = tuple(
        CompileLaunchManifest.load(binding.absolute_path)
        for binding in expected_launch_manifests
    )
    for binding, launch in zip(expected_launch_manifests, launches, strict=True):
        if binding.semantic_sha256 != launch.sha256:
            raise ValueError("live serving group launch manifest identity changed")
    receipt_binding = (
        receipt_path
        if type(receipt_path) is CanonicalJsonProofBinding
        else CanonicalJsonProofBinding.bind(receipt_path)
    )
    assert isinstance(receipt_binding, CanonicalJsonProofBinding)
    receipt = UnsignedPinnedSglangServingGroupReceipt.from_dict(
        receipt_binding.reopen()
    )
    expected_run_binding_sha256s = tuple(
        canonical_sha256(binding.begin_payload()) for binding in expected_run_bindings
    )
    if (
        receipt_binding.semantic_sha256 != receipt.sha256
        or receipt.inventory_sha256 != expected_inventory_sha256
        or receipt.gpu_uuids != expected_gpu_uuids
        or receipt.localhost_ports != expected_localhost_ports
        or receipt.launch_manifests != expected_launch_manifests
        or receipt.run_binding_sha256s != expected_run_binding_sha256s
        or receipt.live_run_receipts != expected_live_run_receipts
        or len(set(expected_gpu_uuids)) != 2
        or len(set(expected_localhost_ports)) != 2
        or len(set(expected_physical_assignment_sha256s)) != 2
        or any(
            launch.inventory_sha256 != expected_inventory_sha256
            or launch.gpu_uuids != (gpu_uuid,)
            or launch.localhost_port != port
            or launch.physical_assignment_sha256 != assignment_sha256
            or launch.experiment_budget_sha256 != budget_sha256
            for launch, gpu_uuid, port, assignment_sha256, budget_sha256 in zip(
                launches,
                expected_gpu_uuids,
                expected_localhost_ports,
                expected_physical_assignment_sha256s,
                expected_experiment_budget_sha256s,
                strict=True,
            )
        )
    ):
        raise ValueError("live serving group differs from sealed inputs")
    validate_pinned_sglang_gpu_process_snapshot(
        receipt.before_gpu_snapshot,
        expected_tool=expected_tool,
        expected_gpu_uuids=expected_gpu_uuids,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_phase="before",
    )
    ready_snapshot = validate_pinned_sglang_gpu_process_snapshot(
        receipt.ready_gpu_snapshot,
        expected_tool=expected_tool,
        expected_gpu_uuids=expected_gpu_uuids,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_phase="ready",
        expected_server_process_group_ids=receipt.server_process_group_ids,
    )
    validate_pinned_sglang_gpu_process_snapshot(
        receipt.after_gpu_snapshot,
        expected_tool=expected_tool,
        expected_gpu_uuids=expected_gpu_uuids,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_phase="after",
    )
    runs = tuple(
        validate_unsigned_pinned_sglang_serving_run_receipt(
            run_receipt,
            expected_launch_manifest=launch_binding,
            expected_binding=run_binding,
            expected_terminal_artifact=terminal_artifact,
            expected_native_itl_pointer_artifact=pointer_artifact,
            expected_scored_request_inputs_sha256=scored_inputs_sha256,
            expected_gpu_uuids=launch.gpu_uuids,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_physical_assignment_sha256=assignment_sha256,
            expected_experiment_budget_sha256=budget_sha256,
            expected_tool=expected_tool,
            expected_snapshot_gpu_uuids=expected_gpu_uuids,
            expected_server_process_group_ids=receipt.server_process_group_ids,
            expected_verified_native_graph_gpu_proof=graph_gpu_proof,
            expected_graph_source_identity_sha256=graph_source_identity_sha256,
            expected_verified_eagle3_e0_execution_authority=eagle3_authority,
            expected_eagle3_source_identity_sha256=eagle3_source_identity_sha256,
            expected_verified_chronobelief_gpu_proof=chronobelief_gpu_proof,
            expected_chronobelief_source_identity_sha256=(
                chronobelief_source_identity_sha256
            ),
        )
        for (
            launch_binding,
            run_binding,
            terminal_artifact,
            pointer_artifact,
            run_receipt,
            scored_inputs_sha256,
            launch,
            assignment_sha256,
            budget_sha256,
            graph_gpu_proof,
            graph_source_identity_sha256,
            eagle3_authority,
            eagle3_source_identity_sha256,
            chronobelief_gpu_proof,
            chronobelief_source_identity_sha256,
        ) in zip(
            expected_launch_manifests,
            expected_run_bindings,
            expected_terminal_artifacts,
            expected_native_itl_pointer_artifacts,
            expected_live_run_receipts,
            expected_scored_request_inputs_sha256s,
            launches,
            expected_physical_assignment_sha256s,
            expected_experiment_budget_sha256s,
            expected_verified_native_graph_gpu_proofs,
            expected_graph_source_identity_sha256s,
            expected_verified_eagle3_e0_execution_authorities,
            expected_eagle3_source_identity_sha256s,
            expected_verified_chronobelief_gpu_proofs,
            expected_chronobelief_source_identity_sha256s,
            strict=True,
        )
    )
    if len(runs) != 2:  # pragma: no cover - zip invariant
        raise RuntimeError("live serving group run coverage changed")
    if (
        tuple(run.receipt.server_process_id for run in runs)
        != receipt.server_process_group_ids
        or canonical_sha256(ready_snapshot["compute_process_rows"])
        != receipt.ready_compute_process_rows_sha256
    ):
        raise ValueError("live serving group ready process ownership differs")
    scored_intervals = tuple(
        (run.receipt.scored_started_ns, run.receipt.scored_finished_ns) for run in runs
    )
    if len(scored_intervals) != 2:  # pragma: no cover - tuple invariant
        raise RuntimeError("live serving group interval cardinality changed")
    overlap_started, overlap_finished = _derive_actual_group_overlap(
        shared_origin_ns=receipt.shared_scored_origin_ns,
        scored_intervals=(scored_intervals[0], scored_intervals[1]),
    )
    if (
        overlap_started != receipt.overlap_started_ns
        or overlap_finished != receipt.overlap_finished_ns
        or receipt.overlap_duration_ns != overlap_finished - overlap_started
    ):
        raise ValueError("live serving group has no exact simultaneous overlap")
    return ValidatedUnsignedPinnedSglangServingGroup(
        receipt_binding=receipt_binding,
        receipt=receipt,
        runs=(runs[0], runs[1]),
        _verification_tag=_VALIDATED_LIVE_GROUP_SENTINEL,
    )


def validate_unsigned_pinned_sglang_serving_group_receipt(
    receipt_path: str | Path | CanonicalJsonProofBinding,
    *,
    expected_specs: tuple[PinnedSglangServingRunSpec, PinnedSglangServingRunSpec],
    expected_tool: PinnedNvidiaSmiTool,
    expected_inventory_sha256: str,
) -> ValidatedUnsignedPinnedSglangServingGroup:
    """Deep-reopen both live runs from the source-owned execution specs."""

    if (
        type(expected_specs) is not tuple
        or len(expected_specs) != 2
        or any(type(spec) is not PinnedSglangServingRunSpec for spec in expected_specs)
    ):
        raise TypeError("live serving group replay requires exactly two specs")
    launches = tuple(
        replace(spec, lifecycle_timing_output_path=None).validate(
            timeout_seconds=_MAX_RUN_TIMEOUT_SECONDS
        )
        for spec in expected_specs
    )
    for spec in expected_specs:
        if spec.lifecycle_timing_output_path is not None:
            CanonicalJsonProofBinding.bind(spec.lifecycle_timing_output_path)
    return validate_unsigned_pinned_sglang_serving_group_receipt_by_identity(
        receipt_path,
        expected_launch_manifests=(
            expected_specs[0].launch_manifest,
            expected_specs[1].launch_manifest,
        ),
        expected_run_bindings=(expected_specs[0].binding, expected_specs[1].binding),
        expected_terminal_artifacts=tuple(
            CanonicalJsonProofBinding.bind(spec.terminal_output_path)
            for spec in expected_specs
        ),
        expected_native_itl_pointer_artifacts=tuple(
            CanonicalJsonProofBinding.bind(spec.native_itl_pointer_output_path)
            for spec in expected_specs
        ),
        expected_live_run_receipts=tuple(
            CanonicalJsonProofBinding.bind(spec.live_run_receipt_output_path)
            for spec in expected_specs
        ),
        expected_scored_request_inputs_sha256s=tuple(
            canonical_sha256([request.sha256 for request in spec.scored_requests])
            for spec in expected_specs
        ),
        expected_gpu_uuids=(launches[0].gpu_uuids[0], launches[1].gpu_uuids[0]),
        expected_localhost_ports=(
            launches[0].localhost_port,
            launches[1].localhost_port,
        ),
        expected_physical_assignment_sha256s=(
            launches[0].physical_assignment_sha256,
            launches[1].physical_assignment_sha256,
        ),
        expected_experiment_budget_sha256s=(
            launches[0].experiment_budget_sha256,
            launches[1].experiment_budget_sha256,
        ),
        expected_tool=expected_tool,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_verified_native_graph_gpu_proofs=tuple(
            spec.verified_native_graph_gpu_proof for spec in expected_specs
        ),
        expected_graph_source_identity_sha256s=tuple(
            spec.expected_graph_source_identity_sha256 for spec in expected_specs
        ),
        expected_verified_eagle3_e0_execution_authorities=tuple(
            spec.verified_eagle3_e0_execution_authority for spec in expected_specs
        ),
        expected_eagle3_source_identity_sha256s=tuple(
            spec.expected_eagle3_source_identity_sha256 for spec in expected_specs
        ),
        expected_verified_chronobelief_gpu_proofs=tuple(
            spec.verified_chronobelief_gpu_proof for spec in expected_specs
        ),
        expected_chronobelief_source_identity_sha256s=tuple(
            spec.expected_chronobelief_source_identity_sha256 for spec in expected_specs
        ),
    )


@dataclass
class _GroupLiveState:
    spec: PinnedSglangServingRunSpec
    launch: CompileLaunchManifest
    config: RunConfig
    terminal_path: Path
    pointer_path: Path
    receipt_path: Path
    log_path: Path
    server_execution_policy_fields_json: str = ""
    server_execution_policy_fields_sha256: str = ""
    execution_started_ns: int = 0
    scored_started_ns: int = 0
    scored_finished_ns: int = 0
    native_terminal_finished_ns: int = 0
    process_exited_ns: int = 0
    process_group_empty_checked_ns: int = 0
    process_exit_code: int = 0
    cleanup_kind: Literal["already_exited_clean", "sigterm_clean"] = (
        "already_exited_clean"
    )
    process: subprocess.Popen[bytes] | None = None
    transport: PinnedBenchServingTransport | None = None
    log_file: object | None = None
    collection: UnsignedNativeTerminalCollection | None = None
    terminal_evidence: ValidatedNativeTerminalEvidence | None = None


def _publish_group_fatal_pointer(
    path: Path,
    *,
    reason_code: str,
    error: BaseException,
    specs: tuple[PinnedSglangServingRunSpec, PinnedSglangServingRunSpec],
    inventory_sha256: str,
    before_snapshot: CanonicalJsonProofBinding | None,
    ready_snapshot: CanonicalJsonProofBinding | None,
    after_snapshot: CanonicalJsonProofBinding | None,
    states: Sequence[_GroupLiveState],
) -> CanonicalJsonProofBinding:
    payload = {
        "schema_version": 1,
        "kind": "unsigned_pinned_sglang_concurrent_group_fatal_pointer",
        "protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
        "status": "ERROR",
        "formal_execution_authorized": False,
        "reason_code": reason_code,
        "error_type": type(error).__name__,
        "emitted_ns": time.monotonic_ns(),
        "inventory_sha256": inventory_sha256,
        "launch_manifests": [spec.launch_manifest.to_dict() for spec in specs],
        "run_binding_sha256s": [
            canonical_sha256(spec.binding.begin_payload()) for spec in specs
        ],
        "before_gpu_snapshot": (
            None if before_snapshot is None else before_snapshot.to_dict()
        ),
        "ready_gpu_snapshot": (
            None if ready_snapshot is None else ready_snapshot.to_dict()
        ),
        "after_gpu_snapshot": (
            None if after_snapshot is None else after_snapshot.to_dict()
        ),
        "runs": [
            {
                "terminal_artifact": _optional_json_binding(state.terminal_path),
                "native_itl_pointer_artifact": _optional_json_binding(
                    state.pointer_path
                ),
                "live_run_receipt": _optional_json_binding(state.receipt_path),
                "server_log": _optional_file_binding(
                    state.log_path, label="failed concurrent server log"
                ),
                "server_process_id": (
                    None if state.process is None else state.process.pid
                ),
                "server_process_exit_code": (
                    None if state.process is None else state.process.poll()
                ),
                "process_group_empty": (
                    None
                    if state.process is None or state.process_group_empty_checked_ns < 1
                    else not _process_group_exists(state.process.pid)
                ),
                "process_group_empty_checked_ns": (
                    state.process_group_empty_checked_ns or None
                ),
            }
            for state in states
        ],
    }
    publish_canonical_json_no_replace(path, payload)
    return CanonicalJsonProofBinding.bind(path)


def _publish_concurrent_run_receipt(
    state: _GroupLiveState,
    *,
    states: tuple[_GroupLiveState, _GroupLiveState],
    gpu_uuids: tuple[str, str],
    before_snapshot: CanonicalJsonProofBinding,
    ready_snapshot: CanonicalJsonProofBinding,
    after_snapshot: CanonicalJsonProofBinding,
    runner_source_body: bytes,
) -> CanonicalJsonProofBinding:
    if state.collection is None or state.terminal_evidence is None:
        raise RuntimeError("concurrent live run lacks terminal evidence")
    lifecycle_events = state.collection.lifecycle_events
    if type(lifecycle_events) is not UnsignedNativeLifecycleEvents:
        raise RuntimeError("concurrent live run lacks source lifecycle events")
    lifecycle_events_json = json.dumps(
        lifecycle_events.to_dict(), sort_keys=True, separators=(",", ":")
    )
    if states[0].process is None or states[1].process is None:
        raise RuntimeError("concurrent live run lacks process identity")
    log_binding = EvidenceFileBinding.bind(
        state.log_path, label="concurrent serving server log"
    )
    ready_process_rows = ready_snapshot.reopen()["compute_process_rows"]
    (
        execution_policy_sha256,
        cuda_graph_mode,
        native_graph_gpu_proof_sha256,
        native_graph_source_identity_sha256,
    ) = _graph_execution_authority(
        runtime=state.config.runtime,
        server_argv=state.launch.server_argv,
        verified_gpu_proof=state.spec.verified_native_graph_gpu_proof,
        expected_source_identity_sha256=(
            state.spec.expected_graph_source_identity_sha256
        ),
        inventory_sha256=state.launch.inventory_sha256,
        gpu_uuids=state.launch.gpu_uuids,
    )
    _mechanism_execution_authority(
        config=state.config,
        server_argv=state.launch.server_argv,
    )
    (
        eagle3_e0_execution_authority_sha256,
        eagle3_compatibility_authority_sha256,
        eagle3_model_selector_sha256,
        eagle3_native_gpu_proof_sha256,
        eagle3_native_source_identity_sha256,
    ) = _eagle3_execution_authority(
        config=state.config,
        verified_authority=state.spec.verified_eagle3_e0_execution_authority,
        expected_source_identity_sha256=(
            state.spec.expected_eagle3_source_identity_sha256
        ),
        inventory_sha256=state.launch.inventory_sha256,
        gpu_uuids=state.launch.gpu_uuids,
    )
    (
        chronobelief_gpu_proof_sha256,
        chronobelief_source_identity_sha256,
    ) = _chronobelief_execution_authority(
        config=state.config,
        verified_gpu_proof=state.spec.verified_chronobelief_gpu_proof,
        expected_source_identity_sha256=(
            state.spec.expected_chronobelief_source_identity_sha256
        ),
        inventory_sha256=state.launch.inventory_sha256,
        gpu_uuids=state.launch.gpu_uuids,
    )
    if (
        not state.server_execution_policy_fields_json
        or not state.server_execution_policy_fields_sha256
    ):
        raise RuntimeError("concurrent live run lacks server policy observation")
    receipt = UnsignedPinnedSglangServingRunReceipt(
        schema_version=1,
        kind="unsigned_pinned_sglang_serving_run_receipt",
        protocol_sha256=PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        runner_source_raw_sha256=hashlib.sha256(runner_source_body).hexdigest(),
        runner_source_size=len(runner_source_body),
        launch_manifest=state.spec.launch_manifest,
        formal_launch_admission=None,
        formal_launch_consumption=None,
        budget_consumption=None,
        run_binding_sha256=canonical_sha256(state.spec.binding.begin_payload()),
        terminal_artifact=state.collection.terminal_artifact,
        native_itl_pointer_artifact=state.collection.native_itl_pointer_artifact,
        terminal_sha256=state.terminal_evidence.terminal_sha256,
        patched_sglang_commit=state.launch.patched_sglang_commit,
        patched_sglang_tree=state.launch.patched_sglang_tree,
        server_argv_sha256=state.launch.server_argv_sha256,
        execution_policy_sha256=execution_policy_sha256,
        cuda_graph_mode=cuda_graph_mode,
        native_graph_gpu_proof_sha256=native_graph_gpu_proof_sha256,
        native_graph_source_identity_sha256=(native_graph_source_identity_sha256),
        eagle3_e0_execution_authority_sha256=(eagle3_e0_execution_authority_sha256),
        eagle3_compatibility_authority_sha256=(eagle3_compatibility_authority_sha256),
        eagle3_model_selector_sha256=eagle3_model_selector_sha256,
        eagle3_native_gpu_proof_sha256=eagle3_native_gpu_proof_sha256,
        eagle3_native_source_identity_sha256=(eagle3_native_source_identity_sha256),
        chronobelief_gpu_proof_sha256=chronobelief_gpu_proof_sha256,
        chronobelief_source_identity_sha256=(chronobelief_source_identity_sha256),
        native_lifecycle_events_json=lifecycle_events_json,
        native_lifecycle_events_sha256=lifecycle_events.sha256,
        server_execution_policy_fields_json=(state.server_execution_policy_fields_json),
        server_execution_policy_fields_sha256=(
            state.server_execution_policy_fields_sha256
        ),
        physical_assignment_sha256=state.launch.physical_assignment_sha256,
        experiment_budget_sha256=state.launch.experiment_budget_sha256,
        inventory_sha256=state.launch.inventory_sha256,
        gpu_uuids=state.launch.gpu_uuids,
        server_process_id=state.terminal_evidence.begin_receipt.server_process_id,
        server_process_started_ns=(
            state.terminal_evidence.begin_receipt.server_process_started_ns
        ),
        execution_started_ns=state.execution_started_ns,
        scored_started_ns=state.scored_started_ns,
        scored_finished_ns=state.scored_finished_ns,
        native_terminal_finished_ns=state.native_terminal_finished_ns,
        process_exited_ns=state.process_exited_ns,
        process_exit_code=state.process_exit_code,
        cleanup_kind=state.cleanup_kind,
        process_group_empty=True,
        process_group_empty_checked_ns=state.process_group_empty_checked_ns,
        server_log=log_binding,
        snapshot_gpu_uuids=gpu_uuids,
        server_process_group_ids=(states[0].process.pid, states[1].process.pid),
        ready_compute_process_rows_sha256=canonical_sha256(ready_process_rows),
        before_gpu_snapshot=before_snapshot,
        ready_gpu_snapshot=ready_snapshot,
        after_gpu_snapshot=after_snapshot,
    )
    evidence_flush_started_ns = time.monotonic_ns()
    publish_canonical_json_no_replace(state.receipt_path, receipt.to_dict())
    receipt_binding = CanonicalJsonProofBinding.bind(
        state.receipt_path, semantic_sha256=receipt.sha256
    )
    evidence_flush_finished_ns = time.monotonic_ns()
    if state.spec.lifecycle_timing_output_path is not None:
        lifecycle_binding = _publish_unsigned_lifecycle_timing_receipt(
            output_path=state.spec.lifecycle_timing_output_path,
            live_run_receipt=receipt_binding,
            binding=state.spec.binding,
            config=state.config,
            evidence_flush_started_ns=evidence_flush_started_ns,
            evidence_flush_finished_ns=evidence_flush_finished_ns,
        )
        validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
            lifecycle_binding,
            expected_live_run_receipt=receipt_binding,
            expected_binding=state.spec.binding,
            expected_telemetry_detail=state.config.runtime.telemetry_detail,
        )
    return receipt_binding


async def execute_unsigned_native_serving_group(
    *,
    specs: tuple[PinnedSglangServingRunSpec, PinnedSglangServingRunSpec],
    nvidia_smi_tool: PinnedNvidiaSmiTool,
    inventory_sha256: str,
    before_gpu_snapshot_output_path: str | Path,
    ready_gpu_snapshot_output_path: str | Path,
    after_gpu_snapshot_output_path: str | Path,
    group_receipt_output_path: str | Path,
    fatal_output_path: str | Path,
    timeout_seconds: float,
) -> ValidatedUnsignedPinnedSglangServingGroup:
    """Run exactly two GPU-isolated servers behind one scored start barrier."""

    if (
        type(specs) is not tuple
        or len(specs) != 2
        or any(type(spec) is not PinnedSglangServingRunSpec for spec in specs)
    ):
        raise TypeError("live serving concurrent group requires exactly two specs")
    try:
        fatal_path = _absolute_output_path(
            "live serving group fatal output", fatal_output_path
        )
    except BaseException as error:
        raise PinnedSglangServingRunError("group_fatal_output_invalid") from error
    states: list[_GroupLiveState] = []
    before_snapshot: CanonicalJsonProofBinding | None = None
    ready_snapshot: CanonicalJsonProofBinding | None = None
    after_snapshot: CanonicalJsonProofBinding | None = None
    try:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 1.0 <= float(timeout_seconds) <= _MAX_RUN_TIMEOUT_SECONDS
        ):
            raise ValueError("group timeout is outside the source-owned bound")
        timeout = float(timeout_seconds)
        _require_sha256("live serving group inventory", inventory_sha256)
        if type(nvidia_smi_tool) is not PinnedNvidiaSmiTool:
            raise TypeError("live serving group requires a pinned nvidia-smi tool")
        nvidia_smi_tool.revalidate()
        launches = tuple(spec.validate(timeout_seconds=timeout) for spec in specs)
        if (
            launches[0].patched_sglang_checkout != launches[1].patched_sglang_checkout
            or any(launch.inventory_sha256 != inventory_sha256 for launch in launches)
            or launches[0].gpu_uuids[0] == launches[1].gpu_uuids[0]
            or launches[0].localhost_port == launches[1].localhost_port
            or launches[0].physical_assignment_sha256
            == launches[1].physical_assignment_sha256
        ):
            raise ValueError("live serving group launch isolation differs")
        before_path = _absolute_output_path(
            "live serving group before snapshot", before_gpu_snapshot_output_path
        )
        ready_path = _absolute_output_path(
            "live serving group ready snapshot", ready_gpu_snapshot_output_path
        )
        after_path = _absolute_output_path(
            "live serving group after snapshot", after_gpu_snapshot_output_path
        )
        group_receipt_path = _absolute_output_path(
            "live serving group receipt", group_receipt_output_path
        )
        all_paths: set[Path] = {
            before_path,
            ready_path,
            after_path,
            group_receipt_path,
            fatal_path,
        }
        for spec, launch in zip(specs, launches, strict=True):
            terminal_path = _absolute_output_path(
                "concurrent terminal output", spec.terminal_output_path
            )
            pointer_path = _absolute_output_path(
                "concurrent ITL output", spec.native_itl_pointer_output_path
            )
            receipt_path = _absolute_output_path(
                "concurrent live receipt", spec.live_run_receipt_output_path
            )
            log_path = _absolute_output_path(
                "concurrent server log", spec.server_log_output_path
            )
            row_paths = {terminal_path, pointer_path, receipt_path, log_path}
            if spec.lifecycle_timing_output_path is not None:
                row_paths.add(
                    _absolute_output_path(
                        "concurrent lifecycle timing output",
                        spec.lifecycle_timing_output_path,
                    )
                )
            expected_row_path_count = (
                5 if spec.lifecycle_timing_output_path is not None else 4
            )
            if len(row_paths) != expected_row_path_count or all_paths & row_paths:
                raise ValueError("live serving group output paths overlap")
            all_paths.update(row_paths)
            _require_source_owned_server_executable(launch.server_argv[0])
            _require_port_unused(launch.localhost_port)
            states.append(
                _GroupLiveState(
                    spec=spec,
                    launch=launch,
                    config=load_run_config(launch.run_config_path),
                    terminal_path=terminal_path,
                    pointer_path=pointer_path,
                    receipt_path=receipt_path,
                    log_path=log_path,
                )
            )
    except BaseException as error:
        fatal = _publish_group_fatal_pointer(
            fatal_path,
            reason_code="group_prelaunch_validation_failed",
            error=error,
            specs=specs,
            inventory_sha256=inventory_sha256,
            before_snapshot=None,
            ready_snapshot=None,
            after_snapshot=None,
            states=states,
        )
        raise PinnedSglangServingRunError(
            "group_prelaunch_validation_failed", fatal_pointer=fatal
        ) from error

    group_error: BaseException | None = None
    gpu_uuids = tuple(state.launch.gpu_uuids[0] for state in states)
    assert len(gpu_uuids) == 2
    try:
        before_snapshot = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=nvidia_smi_tool,
            gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
            inventory_sha256=inventory_sha256,
            phase="before",
            output_path=before_path,
        )
        for state in states:
            descriptor = os.open(
                state.log_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            state.log_file = os.fdopen(descriptor, "wb", buffering=0)
            state.log_file.write(
                b"source-owned pinned SGLang concurrent serving started\n"
            )
            state.execution_started_ns = time.monotonic_ns()
            state.process = await asyncio.to_thread(
                _spawn_server,
                state.launch,
                child_environment_overlay=None,
                stdout_file=state.log_file,
                stderr_file=state.log_file,
            )
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    _wait_server_ready,
                    state.process,
                    port=state.launch.localhost_port,
                    timeout_seconds=min(timeout, _SERVER_READY_TIMEOUT_SECONDS),
                )
                for state in states
            )
        )
        ready_snapshot = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=nvidia_smi_tool,
            gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
            inventory_sha256=inventory_sha256,
            phase="ready",
            output_path=ready_path,
            expected_server_process_group_ids=(
                states[0].process.pid,
                states[1].process.pid,
            ),
        )
        for state in states:
            _bound_request_type, transport_type = _serving_runtime_types()
            state.transport = transport_type.from_checkout(
                state.launch.patched_sglang_checkout
            )
            if type(state.transport) is not transport_type:
                raise TypeError("concurrent serving requires exact pinned transports")
        await asyncio.gather(
            *(
                state.transport.open(
                    request_timeout_s=min(timeout, _SERVER_READY_TIMEOUT_SECONDS),
                    abort_timeout_s=_ABORT_TIMEOUT_SECONDS,
                )
                for state in states
                if state.transport is not None
            )
        )
        for state in states:
            assert state.transport is not None
            state.transport.bind_native_admin_base_url(
                f"http://127.0.0.1:{state.launch.localhost_port}"
            )
        policy_observations = await asyncio.gather(
            *(
                _observe_live_server_execution_policy(
                    transport=state.transport,
                    config=state.config,
                )
                for state in states
                if state.transport is not None
            )
        )
        if len(policy_observations) != 2:
            raise RuntimeError("concurrent server policy coverage is incomplete")
        for state, observation in zip(states, policy_observations, strict=True):
            (
                state.server_execution_policy_fields_json,
                state.server_execution_policy_fields_sha256,
            ) = observation
        barrier = _SharedScoredBarrier()

        async def collect_state(state: _GroupLiveState) -> None:
            assert state.transport is not None
            assert state.process is not None

            async def source_owned_executor(
                phase: str, requests: tuple[BoundServingRequest, ...]
            ) -> UnsignedNativeServingPhaseResult:
                shared_origin: int | None = None
                if phase == "scored":
                    shared_origin = await barrier.arrive()
                result = await _execute_source_owned_phase(
                    phase,
                    requests,
                    concurrency=state.config.runtime.max_running_requests,
                    transport=state.transport,
                    base_url=f"http://127.0.0.1:{state.launch.localhost_port}",
                    served_model=state.config.model.target,
                    shared_origin_ns=shared_origin,
                )
                return result

            state.collection = await collect_unsigned_native_terminal_artifact(
                state.transport,
                binding=state.spec.binding,
                warmup_requests=state.spec.warmup_requests,
                scored_requests=state.spec.scored_requests,
                execute_requests=source_owned_executor,
                output_path=str(state.terminal_path),
                native_itl_pointer_output_path=str(state.pointer_path),
                expected_server_process_id=state.process.pid,
            )
            state.native_terminal_finished_ns = time.monotonic_ns()
            state.terminal_evidence = validate_native_terminal_artifact(
                state.collection.terminal_artifact.reopen(),
                trusted_attester_policy=NO_TRUSTED_ATTESTERS,
                expected_binding=state.spec.binding,
            )
            if (
                state.terminal_evidence.authority_kind != "untrusted_raw_terminal"
                or state.terminal_evidence.begin_receipt.server_process_id
                != state.process.pid
            ):
                raise RuntimeError("concurrent terminal belongs to another process")
            (
                state.scored_started_ns,
                state.scored_finished_ns,
            ) = _reopen_native_scored_interval(
                pointer_artifact=state.collection.native_itl_pointer_artifact,
                terminal_artifact=state.collection.terminal_artifact,
                binding=state.spec.binding,
                terminal_evidence=state.terminal_evidence,
                scored_request_inputs_sha256=canonical_sha256(
                    [request.sha256 for request in state.spec.scored_requests]
                ),
            )

        async with asyncio.timeout(timeout):
            await asyncio.gather(*(collect_state(state) for state in states))
    except BaseException as error:  # noqa: BLE001 - preserve group evidence
        group_error = error
    finally:
        close_results = await asyncio.gather(
            *(
                state.transport.close()
                for state in states
                if state.transport is not None
            ),
            return_exceptions=True,
        )
        for result in close_results:
            if isinstance(result, BaseException) and group_error is None:
                group_error = result
        for state in states:
            if state.process is not None:
                try:
                    (
                        state.process_exit_code,
                        state.cleanup_kind,
                        state.process_exited_ns,
                    ) = await asyncio.to_thread(_terminate_process_group, state.process)
                except BaseException as error:  # noqa: BLE001 - cleanup evidence
                    if state.process.poll() is not None:
                        state.process_exited_ns = time.monotonic_ns()
                    if group_error is None:
                        group_error = error
                finally:
                    state.process_group_empty_checked_ns = time.monotonic_ns()
            if state.log_file is not None:
                try:
                    state.log_file.flush()
                    os.fsync(state.log_file.fileno())
                    state.log_file.close()
                except BaseException as error:  # noqa: BLE001 - retain evidence
                    if group_error is None:
                        group_error = error
        try:
            after_snapshot = await asyncio.to_thread(
                _capture_gpu_process_snapshot,
                tool=nvidia_smi_tool,
                gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
                inventory_sha256=inventory_sha256,
                phase="after",
                output_path=after_path,
            )
        except BaseException as error:  # noqa: BLE001 - retain failed snapshot
            after_snapshot = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
                inventory_sha256=inventory_sha256,
                phase="after",
                output_path=after_path,
                error=error,
            )
            if group_error is None:
                group_error = error
    if group_error is not None:
        if before_snapshot is None and not before_path.exists():
            before_snapshot = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
                inventory_sha256=inventory_sha256,
                phase="before",
                output_path=before_path,
                error=group_error,
            )
        if ready_snapshot is None and not ready_path.exists():
            ready_snapshot = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
                inventory_sha256=inventory_sha256,
                phase="ready",
                output_path=ready_path,
                error=group_error,
                expected_server_process_group_ids=tuple(
                    state.process.pid for state in states if state.process is not None
                )
                if all(state.process is not None for state in states)
                else None,
            )
        fatal = _publish_group_fatal_pointer(
            fatal_path,
            reason_code="concurrent_group_execution_failed",
            error=group_error,
            specs=specs,
            inventory_sha256=inventory_sha256,
            before_snapshot=before_snapshot,
            ready_snapshot=ready_snapshot,
            after_snapshot=after_snapshot,
            states=states,
        )
        raise PinnedSglangServingRunError(
            "concurrent_group_execution_failed", fatal_pointer=fatal
        ) from group_error

    assert before_snapshot is not None
    assert ready_snapshot is not None
    assert after_snapshot is not None
    assert barrier.origin_ns is not None
    runner_source_body = Path(__file__).resolve().read_bytes()
    run_receipts: list[CanonicalJsonProofBinding] = []
    for state in states:
        if (
            state.collection is None
            or state.terminal_evidence is None
            or state.scored_started_ns < barrier.origin_ns
            or state.scored_finished_ns <= state.scored_started_ns
            or state.process_exited_ns < state.native_terminal_finished_ns
            or state.process_group_empty_checked_ns < state.process_exited_ns
        ):
            error = RuntimeError("concurrent live run terminal is incomplete")
            fatal = _publish_group_fatal_pointer(
                fatal_path,
                reason_code="concurrent_group_postprocessing_failed",
                error=error,
                specs=specs,
                inventory_sha256=inventory_sha256,
                before_snapshot=before_snapshot,
                ready_snapshot=ready_snapshot,
                after_snapshot=after_snapshot,
                states=states,
            )
            raise PinnedSglangServingRunError(
                "concurrent_group_postprocessing_failed", fatal_pointer=fatal
            ) from error
        try:
            run_receipts.append(
                _publish_concurrent_run_receipt(
                    state,
                    states=(states[0], states[1]),
                    gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
                    before_snapshot=before_snapshot,
                    ready_snapshot=ready_snapshot,
                    after_snapshot=after_snapshot,
                    runner_source_body=runner_source_body,
                )
            )
        except BaseException as error:
            fatal = _publish_group_fatal_pointer(
                fatal_path,
                reason_code="concurrent_group_postprocessing_failed",
                error=error,
                specs=specs,
                inventory_sha256=inventory_sha256,
                before_snapshot=before_snapshot,
                ready_snapshot=ready_snapshot,
                after_snapshot=after_snapshot,
                states=states,
            )
            raise PinnedSglangServingRunError(
                "concurrent_group_postprocessing_failed", fatal_pointer=fatal
            ) from error
    try:
        overlap_started_ns, overlap_finished_ns = _derive_actual_group_overlap(
            shared_origin_ns=barrier.origin_ns,
            scored_intervals=(
                (states[0].scored_started_ns, states[0].scored_finished_ns),
                (states[1].scored_started_ns, states[1].scored_finished_ns),
            ),
        )
    except (TypeError, ValueError, RuntimeError) as overlap_error:
        error = RuntimeError("concurrent serving intervals did not overlap")
        fatal = _publish_group_fatal_pointer(
            fatal_path,
            reason_code="concurrent_group_no_overlap",
            error=error,
            specs=specs,
            inventory_sha256=inventory_sha256,
            before_snapshot=before_snapshot,
            ready_snapshot=ready_snapshot,
            after_snapshot=after_snapshot,
            states=states,
        )
        raise PinnedSglangServingRunError(
            "concurrent_group_no_overlap", fatal_pointer=fatal
        ) from overlap_error
    try:
        if states[0].process is None or states[1].process is None:
            raise RuntimeError("concurrent serving lost process identities")
        ready_snapshot_value = ready_snapshot.reopen()
        group_receipt = UnsignedPinnedSglangServingGroupReceipt(
            schema_version=1,
            kind="unsigned_pinned_sglang_concurrent_group_receipt",
            protocol_sha256=PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
            formal_execution_authorized=False,
            inventory_sha256=inventory_sha256,
            gpu_uuids=(gpu_uuids[0], gpu_uuids[1]),
            localhost_ports=(
                states[0].launch.localhost_port,
                states[1].launch.localhost_port,
            ),
            server_process_group_ids=(
                states[0].process.pid,
                states[1].process.pid,
            ),
            ready_compute_process_rows_sha256=canonical_sha256(
                ready_snapshot_value["compute_process_rows"]
            ),
            launch_manifests=(specs[0].launch_manifest, specs[1].launch_manifest),
            run_binding_sha256s=(
                canonical_sha256(specs[0].binding.begin_payload()),
                canonical_sha256(specs[1].binding.begin_payload()),
            ),
            live_run_receipts=(run_receipts[0], run_receipts[1]),
            before_gpu_snapshot=before_snapshot,
            ready_gpu_snapshot=ready_snapshot,
            after_gpu_snapshot=after_snapshot,
            shared_scored_origin_ns=barrier.origin_ns,
            overlap_started_ns=overlap_started_ns,
            overlap_finished_ns=overlap_finished_ns,
            overlap_duration_ns=overlap_finished_ns - overlap_started_ns,
        )
        publish_canonical_json_no_replace(group_receipt_path, group_receipt.to_dict())
        return validate_unsigned_pinned_sglang_serving_group_receipt(
            group_receipt_path,
            expected_specs=specs,
            expected_tool=nvidia_smi_tool,
            expected_inventory_sha256=inventory_sha256,
        )
    except BaseException as error:
        fatal = _publish_group_fatal_pointer(
            fatal_path,
            reason_code="concurrent_group_receipt_failed",
            error=error,
            specs=specs,
            inventory_sha256=inventory_sha256,
            before_snapshot=before_snapshot,
            ready_snapshot=ready_snapshot,
            after_snapshot=after_snapshot,
            states=states,
        )
        raise PinnedSglangServingRunError(
            "concurrent_group_receipt_failed", fatal_pointer=fatal
        ) from error


async def execute_unsigned_native_serving_run(
    *,
    launch_manifest_path: str | Path,
    binding: NativeTerminalRunBinding,
    warmup_requests: Sequence[BoundServingRequest],
    scored_requests: Sequence[BoundServingRequest],
    terminal_output_path: str | Path,
    native_itl_pointer_output_path: str | Path,
    live_run_receipt_output_path: str | Path,
    server_log_output_path: str | Path,
    nvidia_smi_tool: PinnedNvidiaSmiTool,
    before_gpu_snapshot_output_path: str | Path,
    ready_gpu_snapshot_output_path: str | Path,
    after_gpu_snapshot_output_path: str | Path,
    fatal_output_path: str | Path,
    timeout_seconds: float,
    formal_launch_admission: CanonicalJsonProofBinding | None = None,
    formal_launch_consumption: CanonicalJsonProofBinding | None = None,
    budget_consumption: CanonicalJsonProofBinding | None = None,
    verified_native_graph_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    expected_graph_source_identity_sha256: str | None = None,
    verified_eagle3_e0_execution_authority: (
        VerifiedEagle3E0ExecutionAuthority | None
    ) = None,
    trusted_single_operator_eagle3_execution_authority: (
        TrustedSingleOperatorEagle3ExecutionAuthority | None
    ) = None,
    expected_eagle3_source_identity_sha256: str | None = None,
    verified_chronobelief_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
    trusted_single_operator_chronobelief_gpu_parity_proof: (
        TrustedSingleOperatorChronoBeliefGpuParityProof | None
    ) = None,
    expected_chronobelief_source_identity_sha256: str | None = None,
    lifecycle_timing_output_path: str | Path | None = None,
    server_stdout_output_path: str | Path | None = None,
    server_stderr_output_path: str | Path | None = None,
    execution_policy: RegisteredServingExecutionPolicy | None = None,
    child_environment_overlay: Mapping[str, str] | None = None,
) -> ValidatedUnsignedPinnedSglangServingRun:
    """Execute one exact TP1/DP1 run with no caller-injected live boundary."""

    try:
        fatal_path = _absolute_output_path(
            "live serving fatal output", fatal_output_path
        )
    except BaseException as error:
        raise PinnedSglangServingRunError("fatal_output_invalid") from error
    requested_launch_manifest_path = str(launch_manifest_path)
    process: subprocess.Popen[bytes] | None = None
    transport: PinnedBenchServingTransport | None = None
    log_file = None
    stdout_file = None
    stderr_file = None
    execution_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    execution_started_ns: int | None = None
    process_exited_ns: int | None = None
    process_group_empty_checked_ns: int | None = None
    launch_binding: CanonicalJsonProofBinding | None = None
    terminal_path: Path | None = None
    pointer_path: Path | None = None
    receipt_path: Path | None = None
    log_path: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    before_path: Path | None = None
    ready_path: Path | None = None
    after_path: Path | None = None
    lifecycle_path: Path | None = None
    before_snapshot: CanonicalJsonProofBinding | None = None
    ready_snapshot: CanonicalJsonProofBinding | None = None
    after_snapshot: CanonicalJsonProofBinding | None = None
    execution_policy_sha256: str | None = None
    cuda_graph_mode: str | None = None
    native_graph_gpu_proof_sha256: str | None = None
    native_graph_source_identity_sha256: str | None = None
    eagle3_e0_execution_authority_sha256: str | None = None
    eagle3_compatibility_authority_sha256: str | None = None
    eagle3_model_selector_sha256: str | None = None
    eagle3_native_gpu_proof_sha256: str | None = None
    eagle3_native_source_identity_sha256: str | None = None
    chronobelief_gpu_proof_sha256: str | None = None
    chronobelief_source_identity_sha256: str | None = None
    server_execution_policy_fields_json: str | None = None
    server_execution_policy_fields_sha256: str | None = None
    try:
        maximum_timeout_seconds = (
            _MAX_REGISTERED_RUN_TIMEOUT_SECONDS
            if execution_policy is not None
            else _MAX_RUN_TIMEOUT_SECONDS
        )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 1.0 <= float(timeout_seconds) <= maximum_timeout_seconds
        ):
            raise ValueError("timeout is outside the source-owned bound")
        timeout = float(timeout_seconds)
        if type(nvidia_smi_tool) is not PinnedNvidiaSmiTool:
            raise TypeError("live serving requires a pinned nvidia-smi tool")
        nvidia_smi_tool.revalidate()
        launch_path = Path(launch_manifest_path)
        launch = CompileLaunchManifest.load(launch_path)
        launch_binding = CanonicalJsonProofBinding.bind(
            launch_path, semantic_sha256=launch.sha256
        )
        config = load_run_config(launch.run_config_path)
        binding.validate()
        if execution_policy is not None:
            from lightcone_spec.orchestration.executor import (
                RegisteredServingExecutionPolicy,
            )

            if type(execution_policy) is not RegisteredServingExecutionPolicy:
                raise TypeError("live serving execution policy must be exact")
            execution_policy.__post_init__()
            if (
                execution_policy.max_concurrency != config.runtime.max_running_requests
                or timeout * 1_000_000 < execution_policy.minimum_process_timeout_us
            ):
                raise ValueError(
                    "live serving timeout/concurrency is below its registered policy"
                )
        if (
            config.method != binding.method
            or config.model.target != launch.target_model_id
            or config.runtime.tensor_parallel_size != 1
            or config.runtime.data_parallel_size != 1
            or config.runtime.node_count != 1
            or len(launch.gpu_uuids) != 1
        ):
            raise ValueError("live serving requires exact TP1/DP1 launch identity")
        (
            execution_policy_sha256,
            cuda_graph_mode,
            native_graph_gpu_proof_sha256,
            native_graph_source_identity_sha256,
        ) = _graph_execution_authority(
            runtime=config.runtime,
            server_argv=launch.server_argv,
            verified_gpu_proof=verified_native_graph_gpu_proof,
            expected_source_identity_sha256=(expected_graph_source_identity_sha256),
            inventory_sha256=launch.inventory_sha256,
            gpu_uuids=launch.gpu_uuids,
        )
        _mechanism_execution_authority(
            config=config,
            server_argv=launch.server_argv,
        )
        (
            eagle3_e0_execution_authority_sha256,
            eagle3_compatibility_authority_sha256,
            eagle3_model_selector_sha256,
            eagle3_native_gpu_proof_sha256,
            eagle3_native_source_identity_sha256,
        ) = _eagle3_execution_authority(
            config=config,
            verified_authority=verified_eagle3_e0_execution_authority,
            expected_source_identity_sha256=(expected_eagle3_source_identity_sha256),
            inventory_sha256=launch.inventory_sha256,
            gpu_uuids=launch.gpu_uuids,
            trusted_single_operator_authority=(
                trusted_single_operator_eagle3_execution_authority
            ),
        )
        (
            chronobelief_gpu_proof_sha256,
            chronobelief_source_identity_sha256,
        ) = _chronobelief_execution_authority(
            config=config,
            verified_gpu_proof=verified_chronobelief_gpu_proof,
            expected_source_identity_sha256=(
                expected_chronobelief_source_identity_sha256
            ),
            inventory_sha256=launch.inventory_sha256,
            gpu_uuids=launch.gpu_uuids,
            trusted_single_operator_proof=(
                trusted_single_operator_chronobelief_gpu_parity_proof
            ),
        )
        warmup = _validate_phase_inputs(
            "warmup",
            warmup_requests,
            expected_ids=binding.warmup_request_ids,
            timeout_seconds=timeout,
            allow_registered_outcomes=execution_policy is not None,
        )
        scored = _validate_phase_inputs(
            "scored",
            scored_requests,
            expected_ids=binding.scored_request_ids,
            timeout_seconds=timeout,
            allow_registered_outcomes=execution_policy is not None,
        )
        terminal_path = _absolute_output_path(
            "live serving terminal output", terminal_output_path
        )
        pointer_path = _absolute_output_path(
            "live serving ITL pointer output", native_itl_pointer_output_path
        )
        receipt_path = _absolute_output_path(
            "live serving receipt output", live_run_receipt_output_path
        )
        client_lifecycle_path = (
            None
            if execution_policy is None
            else receipt_path.with_name("client-request-lifecycle.json")
        )
        log_path = _absolute_output_path(
            "live serving server log", server_log_output_path
        )
        if (server_stdout_output_path is None) != (server_stderr_output_path is None):
            raise ValueError("live serving stdout/stderr paths must be paired")
        if server_stdout_output_path is not None:
            stdout_path = _absolute_output_path(
                "live serving server stdout", server_stdout_output_path
            )
            stderr_path = _absolute_output_path(
                "live serving server stderr", server_stderr_output_path
            )
        before_path = _absolute_output_path(
            "live serving before snapshot", before_gpu_snapshot_output_path
        )
        ready_path = _absolute_output_path(
            "live serving ready snapshot", ready_gpu_snapshot_output_path
        )
        after_path = _absolute_output_path(
            "live serving after snapshot", after_gpu_snapshot_output_path
        )
        lifecycle_path = (
            None
            if lifecycle_timing_output_path is None
            else _absolute_output_path(
                "live serving lifecycle timing", lifecycle_timing_output_path
            )
        )
        outputs = {
            terminal_path,
            pointer_path,
            receipt_path,
            log_path,
            before_path,
            ready_path,
            after_path,
            fatal_path,
        }
        if client_lifecycle_path is not None:
            outputs.add(client_lifecycle_path)
        if lifecycle_path is not None:
            outputs.add(lifecycle_path)
        if stdout_path is not None:
            assert stderr_path is not None
            outputs.update((stdout_path, stderr_path))
        expected_output_count = (
            8
            + (1 if lifecycle_path is not None else 0)
            + (1 if client_lifecycle_path is not None else 0)
        )
        if stdout_path is not None:
            expected_output_count += 2
        if len(outputs) != expected_output_count:
            raise ValueError("live serving output paths must be distinct")
        _require_source_owned_server_executable(launch.server_argv[0])
        _require_port_unused(launch.localhost_port)
    except BaseException as error:
        fatal = _publish_fatal_pointer(
            fatal_path,
            reason_code="prelaunch_validation_failed",
            error=error,
            binding=binding,
            requested_launch_manifest_path=requested_launch_manifest_path,
            launch_manifest=launch_binding,
            terminal_path=terminal_path,
            pointer_path=pointer_path,
            receipt_path=receipt_path,
            log_path=log_path,
            execution_started_ns=execution_started_ns,
            process=process,
            process_exited_ns=process_exited_ns,
            process_group_empty_checked_ns=process_group_empty_checked_ns,
            cleanup_error=None,
        )
        raise PinnedSglangServingRunError(
            "prelaunch_validation_failed", fatal_pointer=fatal
        ) from error

    runner_source_body = Path(__file__).resolve().read_bytes()
    execution_started_ns = time.monotonic_ns()
    execution_task = asyncio.current_task()
    if execution_task is None:  # pragma: no cover - asyncio always owns this call
        raise RuntimeError("live serving lacks an owning asyncio task")
    hard_timeout_state = {"fired": False}

    def trigger_hard_timeout() -> None:
        hard_timeout_state["fired"] = True
        execution_task.cancel()

    hard_timeout_handle = asyncio.get_running_loop().call_later(
        timeout,
        trigger_hard_timeout,
    )
    scored_started_ns = 0
    scored_finished_ns = 0
    native_terminal_finished_ns = 0
    process_exit_code = 0
    process_exited_ns = None
    process_group_empty_checked_ns = None
    cleanup_kind: Literal["already_exited_clean", "sigterm_clean"] = (
        "already_exited_clean"
    )
    collection: UnsignedNativeTerminalCollection | None = None
    terminal_evidence: ValidatedNativeTerminalEvidence | None = None
    client_lifecycle_binding: CanonicalJsonProofBinding | None = None
    phase_results: dict[str, UnsignedNativeServingPhaseResult] = {}
    try:
        assert before_path is not None
        assert ready_path is not None
        assert after_path is not None
        before_snapshot = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=nvidia_smi_tool,
            gpu_uuids=launch.gpu_uuids,
            inventory_sha256=launch.inventory_sha256,
            phase="before",
            output_path=before_path,
        )
        log_descriptor = os.open(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        log_file = os.fdopen(log_descriptor, "wb", buffering=0)
        log_file.write(b"source-owned pinned SGLang unsigned serving started\n")
        if stdout_path is None:
            stdout_file = log_file
            stderr_file = subprocess.STDOUT
        else:
            assert stderr_path is not None
            stdout_descriptor = os.open(
                stdout_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            stderr_descriptor = os.open(
                stderr_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            stdout_file = os.fdopen(stdout_descriptor, "wb", buffering=0)
            stderr_file = os.fdopen(stderr_descriptor, "wb", buffering=0)
        process = await asyncio.to_thread(
            _spawn_server,
            launch,
            child_environment_overlay=child_environment_overlay,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )
        await asyncio.to_thread(
            _wait_server_ready,
            process,
            port=launch.localhost_port,
            timeout_seconds=min(timeout, _SERVER_READY_TIMEOUT_SECONDS),
        )
        ready_snapshot = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=nvidia_smi_tool,
            gpu_uuids=launch.gpu_uuids,
            inventory_sha256=launch.inventory_sha256,
            phase="ready",
            output_path=ready_path,
            expected_server_process_group_ids=(process.pid,),
        )
        _bound_request_type, transport_type = _serving_runtime_types()
        transport = transport_type.from_checkout(launch.patched_sglang_checkout)
        if type(transport) is not transport_type:
            raise TypeError("formal live serving requires the exact pinned transport")
        await transport.open(
            request_timeout_s=min(timeout, _SERVER_READY_TIMEOUT_SECONDS),
            abort_timeout_s=_ABORT_TIMEOUT_SECONDS,
        )
        base_url = f"http://127.0.0.1:{launch.localhost_port}"
        transport.bind_native_admin_base_url(base_url)
        (
            server_execution_policy_fields_json,
            server_execution_policy_fields_sha256,
        ) = await _observe_live_server_execution_policy(
            transport=transport,
            config=config,
        )

        async def source_owned_executor(
            phase: str, requests: tuple[BoundServingRequest, ...]
        ) -> UnsignedNativeServingPhaseResult:
            phase_result = await _execute_source_owned_phase(
                phase,
                requests,
                concurrency=config.runtime.max_running_requests,
                transport=transport,
                base_url=base_url,
                served_model=config.model.target,
                execution_policy=execution_policy,
            )
            phase_results[phase] = phase_result
            return phase_result

        async with asyncio.timeout(timeout):
            collection = await collect_unsigned_native_terminal_artifact(
                transport,
                binding=binding,
                warmup_requests=warmup,
                scored_requests=scored,
                execute_requests=source_owned_executor,
                output_path=str(terminal_path),
                native_itl_pointer_output_path=str(pointer_path),
                expected_server_process_id=process.pid,
            )
        native_terminal_finished_ns = time.monotonic_ns()
        if execution_policy is not None:
            if set(phase_results) != {"warmup", "scored"}:
                raise RuntimeError(
                    "registered live serving phase coverage is incomplete"
                )
            assert client_lifecycle_path is not None
            client_lifecycle_binding = publish_scalable_client_request_lifecycle(
                output_path=client_lifecycle_path,
                run_binding_sha256=canonical_sha256(binding.begin_payload()),
                execution_policy_sha256=execution_policy.sha256,
                rows=[
                    row
                    for phase in ("warmup", "scored")
                    for row in phase_results[phase].client_lifecycle_rows
                ],
            )
        terminal_evidence = validate_native_terminal_artifact(
            collection.terminal_artifact.reopen(),
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            expected_binding=binding,
        )
        if (
            terminal_evidence.authority_kind != "untrusted_raw_terminal"
            or terminal_evidence.begin_receipt.server_process_id != process.pid
        ):
            raise RuntimeError("live serving terminal belongs to another process")
        completed_scored = tuple(
            row
            for row in terminal_evidence.requests
            if row.request_id in binding.scored_request_ids
            and row.terminal_status == "completed"
        )
        if completed_scored:
            scored_started_ns, scored_finished_ns = _reopen_native_scored_interval(
                pointer_artifact=collection.native_itl_pointer_artifact,
                terminal_artifact=collection.terminal_artifact,
                binding=binding,
                terminal_evidence=terminal_evidence,
                scored_request_inputs_sha256=canonical_sha256(
                    [request.sha256 for request in scored]
                ),
            )
        elif execution_policy is not None:
            scored_phase = phase_results["scored"]
            offered_rows = tuple(
                row
                for row in scored_phase.client_lifecycle_rows
                if row.get("offered") is True
            )
            if not offered_rows:
                raise RuntimeError("registered serving offered no scored requests")
            if collection.lifecycle_events is None:
                raise RuntimeError("registered serving lacks native lifecycle edges")
            scored_started_ns = collection.lifecycle_events.scored_started_ns
            scored_finished_ns = collection.lifecycle_events.scored_finished_ns
        else:
            raise RuntimeError("legacy live serving has no completed scored requests")
    except BaseException as error:  # noqa: BLE001 - preserve cleanup on cancellation
        execution_error = (
            TimeoutError("source-owned process hard timeout expired")
            if hard_timeout_state["fired"] and isinstance(error, asyncio.CancelledError)
            else error
        )
    finally:
        if transport is not None:
            try:
                await transport.close()
            except BaseException as error:  # noqa: BLE001 - cleanup evidence
                process_exited_ns = (
                    time.monotonic_ns()
                    if process.poll() is not None
                    else process_exited_ns
                )
                process_group_empty_checked_ns = time.monotonic_ns()
                cleanup_error = cleanup_error or error
        if process is not None:
            try:
                (
                    process_exit_code,
                    cleanup_kind,
                    process_exited_ns,
                ) = await asyncio.to_thread(_terminate_process_group, process)
                process_group_empty_checked_ns = time.monotonic_ns()
            except BaseException as error:  # noqa: BLE001 - cleanup evidence
                cleanup_error = cleanup_error or error
        if log_file is not None:
            try:
                log_file.flush()
                os.fsync(log_file.fileno())
                log_file.close()
            except BaseException as error:  # noqa: BLE001 - retain evidence
                cleanup_error = cleanup_error or error
        for stream in (stdout_file, stderr_file):
            if stream is None or stream is log_file or stream is subprocess.STDOUT:
                continue
            try:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            except BaseException as error:  # noqa: BLE001 - retain evidence
                cleanup_error = cleanup_error or error
        if after_path is not None:
            try:
                after_snapshot = await asyncio.to_thread(
                    _capture_gpu_process_snapshot,
                    tool=nvidia_smi_tool,
                    gpu_uuids=launch.gpu_uuids,
                    inventory_sha256=launch.inventory_sha256,
                    phase="after",
                    output_path=after_path,
                )
            except BaseException as error:  # noqa: BLE001 - retain failed snapshot
                after_snapshot = _publish_gpu_snapshot_error(
                    tool=nvidia_smi_tool,
                    gpu_uuids=launch.gpu_uuids,
                    inventory_sha256=launch.inventory_sha256,
                    phase="after",
                    output_path=after_path,
                    error=error,
                )
                cleanup_error = cleanup_error or error
    hard_timeout_handle.cancel()
    failure = cleanup_error or execution_error
    if failure is not None:
        assert before_path is not None
        assert ready_path is not None
        if before_snapshot is None and not before_path.exists():
            before_snapshot = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="before",
                output_path=before_path,
                error=failure,
            )
        if ready_snapshot is None and not ready_path.exists():
            ready_snapshot = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="ready",
                output_path=ready_path,
                error=failure,
                expected_server_process_group_ids=(process.pid,)
                if process is not None
                else None,
            )
        reason_code = (
            "server_cleanup_failed"
            if cleanup_error is not None
            else "live_execution_failed"
        )
        fatal = _publish_fatal_pointer(
            fatal_path,
            reason_code=reason_code,
            error=failure,
            binding=binding,
            requested_launch_manifest_path=requested_launch_manifest_path,
            launch_manifest=launch_binding,
            terminal_path=terminal_path,
            pointer_path=pointer_path,
            receipt_path=receipt_path,
            log_path=log_path,
            execution_started_ns=execution_started_ns,
            process=process,
            process_exited_ns=process_exited_ns,
            process_group_empty_checked_ns=process_group_empty_checked_ns,
            cleanup_error=cleanup_error,
            before_gpu_snapshot=before_snapshot,
            ready_gpu_snapshot=ready_snapshot,
            after_gpu_snapshot=after_snapshot,
        )
        raise PinnedSglangServingRunError(reason_code, fatal_pointer=fatal) from failure
    if (
        collection is None
        or terminal_evidence is None
        or scored_started_ns < 1
        or scored_finished_ns < scored_started_ns
        or native_terminal_finished_ns < 1
        or process_exited_ns is None
        or process_exited_ns < 1
        or process_group_empty_checked_ns is None
        or process_group_empty_checked_ns < process_exited_ns
        or before_snapshot is None
        or ready_snapshot is None
        or after_snapshot is None
        or (execution_policy is not None and client_lifecycle_binding is None)
    ):
        error = RuntimeError("live serving terminal state is incomplete")
        fatal = _publish_fatal_pointer(
            fatal_path,
            reason_code="live_terminal_incomplete",
            error=error,
            binding=binding,
            requested_launch_manifest_path=requested_launch_manifest_path,
            launch_manifest=launch_binding,
            terminal_path=terminal_path,
            pointer_path=pointer_path,
            receipt_path=receipt_path,
            log_path=log_path,
            execution_started_ns=execution_started_ns,
            process=process,
            process_exited_ns=process_exited_ns,
            process_group_empty_checked_ns=process_group_empty_checked_ns,
            cleanup_error=None,
            before_gpu_snapshot=before_snapshot,
            ready_gpu_snapshot=ready_snapshot,
            after_gpu_snapshot=after_snapshot,
        )
        raise PinnedSglangServingRunError(
            "live_terminal_incomplete", fatal_pointer=fatal
        ) from error
    try:
        if (
            execution_policy_sha256 is None
            or cuda_graph_mode not in {"disabled", "fixed_address_publication_v1"}
            or server_execution_policy_fields_json is None
            or server_execution_policy_fields_sha256 is None
        ):
            raise RuntimeError("live serving graph execution identity is incomplete")
        log_binding = EvidenceFileBinding.bind(
            log_path, label="live serving server log"
        )
        lifecycle_events = collection.lifecycle_events
        if type(lifecycle_events) is not UnsignedNativeLifecycleEvents:
            raise RuntimeError("live serving lacks source lifecycle events")
        lifecycle_events_json = json.dumps(
            lifecycle_events.to_dict(), sort_keys=True, separators=(",", ":")
        )
        receipt = UnsignedPinnedSglangServingRunReceipt(
            schema_version=1 if execution_policy is None else 2,
            kind="unsigned_pinned_sglang_serving_run_receipt",
            protocol_sha256=(
                PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256
                if execution_policy is None
                else PINNED_SGLANG_REGISTERED_SERVING_PROTOCOL_SHA256
            ),
            formal_execution_authorized=False,
            runner_source_raw_sha256=hashlib.sha256(runner_source_body).hexdigest(),
            runner_source_size=len(runner_source_body),
            launch_manifest=launch_binding,
            formal_launch_admission=formal_launch_admission,
            formal_launch_consumption=formal_launch_consumption,
            budget_consumption=budget_consumption,
            run_binding_sha256=canonical_sha256(binding.begin_payload()),
            terminal_artifact=collection.terminal_artifact,
            native_itl_pointer_artifact=collection.native_itl_pointer_artifact,
            terminal_sha256=terminal_evidence.terminal_sha256,
            patched_sglang_commit=launch.patched_sglang_commit,
            patched_sglang_tree=launch.patched_sglang_tree,
            server_argv_sha256=launch.server_argv_sha256,
            execution_policy_sha256=execution_policy_sha256,
            cuda_graph_mode=cuda_graph_mode,
            native_graph_gpu_proof_sha256=native_graph_gpu_proof_sha256,
            native_graph_source_identity_sha256=(native_graph_source_identity_sha256),
            eagle3_e0_execution_authority_sha256=(eagle3_e0_execution_authority_sha256),
            eagle3_compatibility_authority_sha256=(
                eagle3_compatibility_authority_sha256
            ),
            eagle3_model_selector_sha256=eagle3_model_selector_sha256,
            eagle3_native_gpu_proof_sha256=eagle3_native_gpu_proof_sha256,
            eagle3_native_source_identity_sha256=(eagle3_native_source_identity_sha256),
            chronobelief_gpu_proof_sha256=chronobelief_gpu_proof_sha256,
            chronobelief_source_identity_sha256=(chronobelief_source_identity_sha256),
            native_lifecycle_events_json=lifecycle_events_json,
            native_lifecycle_events_sha256=lifecycle_events.sha256,
            server_execution_policy_fields_json=(server_execution_policy_fields_json),
            server_execution_policy_fields_sha256=(
                server_execution_policy_fields_sha256
            ),
            physical_assignment_sha256=launch.physical_assignment_sha256,
            experiment_budget_sha256=launch.experiment_budget_sha256,
            inventory_sha256=launch.inventory_sha256,
            gpu_uuids=launch.gpu_uuids,
            server_process_id=terminal_evidence.begin_receipt.server_process_id,
            server_process_started_ns=(
                terminal_evidence.begin_receipt.server_process_started_ns
            ),
            execution_started_ns=execution_started_ns,
            scored_started_ns=scored_started_ns,
            scored_finished_ns=scored_finished_ns,
            native_terminal_finished_ns=native_terminal_finished_ns,
            process_exited_ns=process_exited_ns,
            process_exit_code=process_exit_code,
            cleanup_kind=cleanup_kind,
            process_group_empty=True,
            process_group_empty_checked_ns=process_group_empty_checked_ns,
            server_log=log_binding,
            snapshot_gpu_uuids=launch.gpu_uuids,
            server_process_group_ids=(process.pid,),
            ready_compute_process_rows_sha256=canonical_sha256(
                ready_snapshot.reopen()["compute_process_rows"]
            ),
            before_gpu_snapshot=before_snapshot,
            ready_gpu_snapshot=ready_snapshot,
            after_gpu_snapshot=after_snapshot,
            execution_policy=execution_policy,
            client_request_lifecycle=client_lifecycle_binding,
        )
        evidence_flush_started_ns = time.monotonic_ns()
        publish_canonical_json_no_replace(receipt_path, receipt.to_dict())
        live_run_receipt = CanonicalJsonProofBinding.bind(
            receipt_path, semantic_sha256=receipt.sha256
        )
        validated_run = validate_unsigned_pinned_sglang_serving_run_receipt(
            receipt_path,
            expected_launch_manifest=launch_binding,
            expected_binding=binding,
            expected_terminal_artifact=collection.terminal_artifact,
            expected_native_itl_pointer_artifact=(
                collection.native_itl_pointer_artifact
            ),
            expected_scored_request_inputs_sha256=canonical_sha256(
                [request.sha256 for request in scored]
            ),
            expected_gpu_uuids=launch.gpu_uuids,
            expected_inventory_sha256=launch.inventory_sha256,
            expected_physical_assignment_sha256=launch.physical_assignment_sha256,
            expected_experiment_budget_sha256=launch.experiment_budget_sha256,
            expected_tool=nvidia_smi_tool,
            expected_snapshot_gpu_uuids=launch.gpu_uuids,
            expected_server_process_group_ids=(process.pid,),
            expected_verified_native_graph_gpu_proof=(verified_native_graph_gpu_proof),
            expected_graph_source_identity_sha256=(
                expected_graph_source_identity_sha256
            ),
            expected_verified_eagle3_e0_execution_authority=(
                verified_eagle3_e0_execution_authority
            ),
            expected_trusted_single_operator_eagle3_execution_authority=(
                trusted_single_operator_eagle3_execution_authority
            ),
            expected_eagle3_source_identity_sha256=(
                expected_eagle3_source_identity_sha256
            ),
            expected_verified_chronobelief_gpu_proof=(verified_chronobelief_gpu_proof),
            expected_trusted_single_operator_chronobelief_gpu_parity_proof=(
                trusted_single_operator_chronobelief_gpu_parity_proof
            ),
            expected_chronobelief_source_identity_sha256=(
                expected_chronobelief_source_identity_sha256
            ),
            expected_execution_policy=execution_policy,
        )
        evidence_flush_finished_ns = time.monotonic_ns()
        if lifecycle_path is not None:
            lifecycle_binding = _publish_unsigned_lifecycle_timing_receipt(
                output_path=lifecycle_path,
                live_run_receipt=live_run_receipt,
                binding=binding,
                config=config,
                evidence_flush_started_ns=evidence_flush_started_ns,
                evidence_flush_finished_ns=evidence_flush_finished_ns,
            )
            validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
                lifecycle_binding,
                expected_live_run_receipt=live_run_receipt,
                expected_binding=binding,
                expected_telemetry_detail=config.runtime.telemetry_detail,
            )
        return validated_run
    except BaseException as error:
        fatal = _publish_fatal_pointer(
            fatal_path,
            reason_code="live_receipt_validation_failed",
            error=error,
            binding=binding,
            requested_launch_manifest_path=requested_launch_manifest_path,
            launch_manifest=launch_binding,
            terminal_path=terminal_path,
            pointer_path=pointer_path,
            receipt_path=receipt_path,
            log_path=log_path,
            execution_started_ns=execution_started_ns,
            process=process,
            process_exited_ns=process_exited_ns,
            process_group_empty_checked_ns=process_group_empty_checked_ns,
            cleanup_error=None,
            before_gpu_snapshot=before_snapshot,
            ready_gpu_snapshot=ready_snapshot,
            after_gpu_snapshot=after_snapshot,
        )
        raise PinnedSglangServingRunError(
            "live_receipt_validation_failed", fatal_pointer=fatal
        ) from error


__all__ = [
    "PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256",
    "PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256",
    "PinnedNvidiaSmiTool",
    "PinnedSglangLifecycleTimingProofArtifact",
    "PinnedSglangServingRunError",
    "PinnedSglangServingRunSpec",
    "UnsignedPinnedSglangLifecycleTimingReceipt",
    "UnsignedPinnedSglangServingGroupReceipt",
    "UnsignedPinnedSglangServingRunReceipt",
    "ValidatedUnsignedPinnedSglangServingGroup",
    "ValidatedUnsignedPinnedSglangServingRun",
    "VerifiedPinnedSglangLifecycleTimingProof",
    "build_pinned_sglang_lifecycle_timing_control_subject",
    "execute_unsigned_native_serving_group",
    "execute_unsigned_native_serving_run",
    "publish_pinned_sglang_lifecycle_timing_proof_artifact",
    "validate_pinned_sglang_gpu_process_snapshot",
    "validate_pinned_sglang_lifecycle_timing_proof_artifact",
    "validate_unsigned_pinned_sglang_lifecycle_timing_receipt",
    "validate_unsigned_pinned_sglang_serving_group_receipt",
    "validate_unsigned_pinned_sglang_serving_group_receipt_by_identity",
    "validate_unsigned_pinned_sglang_serving_run_receipt",
]
