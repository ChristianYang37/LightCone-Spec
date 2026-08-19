"""Formal raw authority for within-request inter-token latency timestamps.

The official serving adapter may receive several generated token IDs in one
SSE chunk.  A chunk arrival time is not a timestamp for every token in that
chunk, and dividing its gap by the token count would fabricate a latency
distribution.  Formal ITL evidence therefore has exactly two future source
modes: a native per-token timestamp hook, or raw SSE frame observations that
prove every frame contributes exactly one new token.

The pinned runtime exposes a first-party committed-token observation hook and
a content-bound result pointer.  Source presence alone never authorizes a
formal p99 claim: E2 remains ``BLOCKED`` until the exact native-hot-path GPU
qualification receipt has been verified through the release trust root.  The
same verified token is required again when the raw pointer is opened, so a
caller-authored producer row or receipt digest cannot unlock the reducer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Literal

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.load import TokenChunkTiming
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.distributed import (
    DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS,
    DistributedRuntimeGpuProofArtifact,
    VerifiedDistributedRuntimeGpuProof,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_RELEASE_CAPABILITY,
    NativeRuntimeGpuProofArtifact,
    VerifiedNativeRuntimeGpuProof,
)

ItlTimestampSourceMode = Literal[
    "native_per_token_timestamp_hook",
    "sse_one_token_per_frame",
]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_MAX_RAW_RECEIPT_BYTES = 64 * 1024 * 1024

ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON = (
    "release_per_token_timestamp_producer_unavailable"
)
ITL_FIRST_PARTY_RESULT_POINTER_UNAVAILABLE_REASON = (
    "first_party_itl_result_pointer_unavailable"
)
ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON = "native_itl_dynamic_gpu_proof_unavailable"
ITL_CPU_CONTRACT_ONLY_REASON = "cpu_contract_only_not_formal_itl_authority"
SGLANG_CPU_ITL_CONTRACT_HOOK = "sglang.schema_v3.native_per_token_timestamp.v1"
SGLANG_CPU_ITL_CONTRACT_SEMANTICS = "cpu_committed_token_observed_at_streamer_v1"
SGLANG_CPU_ITL_CONTRACT_RELEASE_STATUS = "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"
SGLANG_NATIVE_ITL_CONTRACT_HOOK = "sglang.schema_v3.native_per_token_timestamp.v2"
SGLANG_NATIVE_ITL_CONTRACT_SEMANTICS = (
    "scheduler_committed_token_at_result_processor_v1"
)
SGLANG_NATIVE_ITL_CONTRACT_RELEASE_STATUS = "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"
SGLANG_NATIVE_ITL_RESULT_POINTER_KIND = "sglang_native_itl_result_pointer"
ITL_COALESCED_CHUNK_UNPROVEN_REASON = "coalesced_sse_chunk_has_no_token_timestamps"
ITL_RAW_RECEIPT_MISSING_REASON = "per_token_timestamp_raw_receipt_missing"
ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON = (
    "per_token_timestamp_request_coverage_incomplete"
)

ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_e2_per_token_itl_timestamp_authority",
        "sources": [
            "native_per_token_timestamp_hook",
            "sse_one_token_per_frame",
        ],
        "native_contract": (
            "ordered_token_id_and_monotonic_ns_for_every_generated_token"
        ),
        "sse_contract": "raw_frame_exactly_one_new_token_and_monotonic_ns",
        "request_binding": "request_id_ordered_output_token_ids_and_completed_status",
        "clock_domains": (
            "terminal_identity_has_no_cross_clock_timestamp_comparison;"
            "producer_start_tokens_terminal_share_one_monotonic_clock"
        ),
        "result_authority": "source_owned_first_party_result_pointer_required",
        "coalesced_without_native_timestamps": "BLOCKED",
        "chunk_gap_interpolation": "forbidden",
        "missing": "BLOCKED_and_None_never_zero",
    }
)

STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_stage_neutral_itl_timestamp_proof_protocol",
        "execution_identity": [
            "materialized_cell_id",
            "inventory_sha256",
            "registry_sha256",
            "execution_plan_sha256",
            "rank_config_sha256",
            "run_id",
            "run_nonce_sha256",
            "attempt_id",
            "method",
        ],
        "terminal_join": (
            "durable_external_control_native_result_proof_exact_request_tokens"
        ),
        "producer_join": (
            "durable_native_hot_path_gpu_proof_source_capability_and_identity"
        ),
        "timing": (
            "source_native_result_pointer_request_start_every_token_terminal_ns"
        ),
        "trust": (
            "local_release_controls_then_one_atomic_batch_replay_reservation;"
            "durable_artifact_embeds_the_exact_verified_control_batch"
        ),
        "throughput": "derive_only_from_raw_request_start_and_terminal_timestamps",
        "slo_goodput": (
            "derive_only_after_joining_the_sealed_materialized_cell_slo_and_"
            "request_level_qualification"
        ),
        "interpolation": "forbidden",
    }
)


class ItlTimestampAuthorityBlocked(RuntimeError):
    """Raised before launch or promotion when formal ITL timing is unavailable."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"formal ITL timestamp authority is BLOCKED: {reason}")
        self.reason = reason


def reject_cpu_contract_only_itl_metadata(metadata: Mapping[str, object]) -> None:
    """Reject CPU observation metadata before any formal raw receipt is opened."""

    value = _strict_mapping("CPU ITL metadata", metadata)
    if any(
        value.get(field) == expected
        for field, expected in (
            ("native_token_timestamp_hook", SGLANG_CPU_ITL_CONTRACT_HOOK),
            (
                "native_token_timestamp_semantics",
                SGLANG_CPU_ITL_CONTRACT_SEMANTICS,
            ),
            (
                "native_token_timestamp_release_status",
                SGLANG_CPU_ITL_CONTRACT_RELEASE_STATUS,
            ),
        )
    ):
        raise ItlTimestampAuthorityBlocked(ITL_CPU_CONTRACT_ONLY_REASON)


def _require_sha256(label: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_safe_id(label: str, value: object) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _require_exact_text(label: str, value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be an exact non-empty string")
    return value


def _strict_mapping(label: str, value: object) -> Mapping[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise TypeError(f"{label} must be an exact string-keyed object")
    return value


def _strict_sequence(label: str, value: object) -> Sequence[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an exact array")
    return value


def _strict_int(label: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_keys(
    label: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


@dataclass(frozen=True)
class ReleaseItlTimestampProducer:
    producer_id: str
    source_mode: ItlTimestampSourceMode
    hook_id: str
    producer_version_sha256: str
    patched_sglang_tree: str
    clock: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        _require_safe_id("ITL producer", self.producer_id)
        _require_safe_id("ITL producer hook", self.hook_id)
        _require_sha256("ITL producer version", self.producer_version_sha256)
        if self.source_mode not in {
            "native_per_token_timestamp_hook",
            "sse_one_token_per_frame",
        }:
            raise ValueError("ITL timestamp source mode is unsupported")
        expected_hook = {
            "native_per_token_timestamp_hook": (SGLANG_NATIVE_ITL_CONTRACT_HOOK),
            "sse_one_token_per_frame": (
                "sglang.benchmark.serving.raw_sse_frame_observation.v1"
            ),
        }[self.source_mode]
        if self.hook_id != expected_hook:
            raise ValueError("ITL timestamp hook differs from its source mode")
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("ITL producer uses another patched SGLang tree")
        if self.clock != "monotonic_ns":
            raise ValueError("ITL producer must use the monotonic nanosecond clock")
        if self.protocol_sha256 != ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256:
            raise ValueError("ITL producer uses another authority protocol")

    def to_dict(self) -> dict[str, str]:
        return {
            "producer_id": self.producer_id,
            "source_mode": self.source_mode,
            "hook_id": self.hook_id,
            "producer_version_sha256": self.producer_version_sha256,
            "patched_sglang_tree": self.patched_sglang_tree,
            "clock": self.clock,
            "protocol_sha256": self.protocol_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReleaseItlTimestampProducer:
        row = _strict_mapping("release ITL timestamp producer", value)
        _strict_keys(
            "release ITL timestamp producer",
            row,
            {
                "producer_id",
                "source_mode",
                "hook_id",
                "producer_version_sha256",
                "patched_sglang_tree",
                "clock",
                "protocol_sha256",
            },
        )
        return cls(
            producer_id=_require_safe_id("ITL producer", row["producer_id"]),
            source_mode=row["source_mode"],  # type: ignore[arg-type]
            hook_id=_require_safe_id("ITL producer hook", row["hook_id"]),
            producer_version_sha256=_require_sha256(
                "ITL producer version", row["producer_version_sha256"]
            ),
            patched_sglang_tree=_require_safe_id(
                "ITL producer patched tree", row["patched_sglang_tree"]
            ),
            clock=_require_safe_id("ITL producer clock", row["clock"]),
            protocol_sha256=_require_sha256(
                "ITL producer protocol", row["protocol_sha256"]
            ),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


# Source-owned implementation identity.  It remains unavailable until an exact
# root-authorized ``native_hot_path_tp1`` proof is supplied at both activation
# and raw-pointer replay.
RELEASE_ITL_TIMESTAMP_PRODUCERS: tuple[ReleaseItlTimestampProducer, ...] = (
    ReleaseItlTimestampProducer(
        producer_id="sglang-native-scheduler-committed-token-itl-v2",
        source_mode="native_per_token_timestamp_hook",
        hook_id=SGLANG_NATIVE_ITL_CONTRACT_HOOK,
        producer_version_sha256=content_sha256(
            {
                "hook": SGLANG_NATIVE_ITL_CONTRACT_HOOK,
                "semantics": SGLANG_NATIVE_ITL_CONTRACT_SEMANTICS,
                "release_status": SGLANG_NATIVE_ITL_CONTRACT_RELEASE_STATUS,
                "result_pointer_kind": SGLANG_NATIVE_ITL_RESULT_POINTER_KIND,
                "full_ordered_token_coverage": True,
            }
        ),
        patched_sglang_tree=PINNED_SGLANG_TREE,
        clock="monotonic_ns",
        protocol_sha256=ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256,
    ),
)


@dataclass(frozen=True)
class E2ItlTimestampPlan:
    schema_version: int
    kind: str
    registry_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    patched_sglang_tree: str
    producer: ReleaseItlTimestampProducer | None
    interpolation_forbidden: bool
    full_request_coverage_required: bool
    protocol_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("E2 ITL timestamp plan schema is unsupported")
        if self.kind != "e2_itl_timestamp_plan":
            raise ValueError("E2 ITL timestamp plan schema is unsupported")
        for label, value in (
            ("ITL registry", self.registry_sha256),
            ("ITL cell", self.cell_id),
            ("ITL cell declaration", self.cell_declaration_sha256),
            ("ITL protocol", self.protocol_sha256),
        ):
            _require_sha256(label, value)
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("ITL plan uses another patched SGLang tree")
        if self.producer is not None:
            if type(self.producer) is not ReleaseItlTimestampProducer:
                raise TypeError("ITL plan requires an exact release producer")
            self.producer.__post_init__()
        if self.interpolation_forbidden is not True:
            raise ValueError("ITL plan cannot authorize chunk-gap interpolation")
        if self.full_request_coverage_required is not True:
            raise ValueError("ITL plan must require full request coverage")
        if self.protocol_sha256 != ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256:
            raise ValueError("ITL plan uses another authority protocol")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "cell_declaration_sha256": self.cell_declaration_sha256,
            "patched_sglang_tree": self.patched_sglang_tree,
            "producer": None if self.producer is None else self.producer.to_dict(),
            "interpolation_forbidden": self.interpolation_forbidden,
            "full_request_coverage_required": self.full_request_coverage_required,
            "protocol_sha256": self.protocol_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> E2ItlTimestampPlan:
        row = _strict_mapping("E2 ITL timestamp plan", value)
        _strict_keys(
            "E2 ITL timestamp plan",
            row,
            {
                "schema_version",
                "kind",
                "registry_sha256",
                "cell_id",
                "cell_declaration_sha256",
                "patched_sglang_tree",
                "producer",
                "interpolation_forbidden",
                "full_request_coverage_required",
                "protocol_sha256",
            },
        )
        producer_value = row["producer"]
        return cls(
            schema_version=_strict_int("E2 ITL plan schema", row["schema_version"]),
            kind=row["kind"],  # type: ignore[arg-type]
            registry_sha256=_require_sha256("ITL registry", row["registry_sha256"]),
            cell_id=_require_sha256("ITL cell", row["cell_id"]),
            cell_declaration_sha256=_require_sha256(
                "ITL cell declaration", row["cell_declaration_sha256"]
            ),
            patched_sglang_tree=_require_safe_id(
                "ITL plan patched tree", row["patched_sglang_tree"]
            ),
            producer=(
                None
                if producer_value is None
                else ReleaseItlTimestampProducer.from_dict(producer_value)
            ),
            interpolation_forbidden=row["interpolation_forbidden"],  # type: ignore[arg-type]
            full_request_coverage_required=row["full_request_coverage_required"],  # type: ignore[arg-type]
            protocol_sha256=_require_sha256("ITL protocol", row["protocol_sha256"]),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class ItlTimestampActivation:
    status: Literal["READY", "BLOCKED"]
    reason_code: str | None
    plan_sha256: str
    producer_sha256: str | None

    def __post_init__(self) -> None:
        if self.status not in {"READY", "BLOCKED"}:
            raise ValueError("ITL activation status is unsupported")
        _require_sha256("ITL activation plan", self.plan_sha256)
        if self.status == "BLOCKED":
            if not self.reason_code or self.producer_sha256 is not None:
                raise ValueError("blocked ITL activation must retain only its reason")
        else:
            if self.reason_code is not None or self.producer_sha256 is None:
                raise ValueError("ready ITL activation requires one producer")
            _require_sha256("ITL activation producer", self.producer_sha256)


def release_e2_itl_timestamp_plan(
    registry: ExperimentRegistry,
    cell: ExperimentCell,
) -> E2ItlTimestampPlan:
    """Derive an exact E2 timing plan without launching a serving process."""

    if type(registry) is not ExperimentRegistry or type(cell) is not ExperimentCell:
        raise TypeError("ITL planning requires exact registry and cell objects")
    matches = tuple(
        row for row in registry.cells_for("E2") if row.cell_id == cell.cell_id
    )
    if len(matches) != 1 or matches[0] != cell:
        raise ValueError("ITL cell is foreign to the E2 registry")
    if (
        cell.identity.experiment != "E2"
        or cell.identity.task != "LiveCodeBench_tuning"
        or cell.resources.workload_class is not WorkloadClass.TUNING
    ):
        raise ValueError("ITL plan requires an exact E2 tuning cell")
    if len(RELEASE_ITL_TIMESTAMP_PRODUCERS) > 1:
        raise RuntimeError("release ITL timestamp producer allowlist is ambiguous")
    producer = (
        RELEASE_ITL_TIMESTAMP_PRODUCERS[0] if RELEASE_ITL_TIMESTAMP_PRODUCERS else None
    )
    if producer is not None:
        producer.__post_init__()
    return E2ItlTimestampPlan(
        schema_version=1,
        kind="e2_itl_timestamp_plan",
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        cell_declaration_sha256=cell.sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        producer=producer,
        interpolation_forbidden=True,
        full_request_coverage_required=True,
        protocol_sha256=ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256,
    )


def _require_native_itl_gpu_proof(
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None,
) -> VerifiedNativeRuntimeGpuProof:
    if (
        type(verified_gpu_proof) is not VerifiedNativeRuntimeGpuProof
        or verified_gpu_proof.suite_id != "native_hot_path_tp1"
        or verified_gpu_proof.topology_mode != "tp1_dp1"
        or "native_itl" not in verified_gpu_proof.backend_capabilities
        or verified_gpu_proof.source_capability_sha256
        != NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
    ):
        raise ItlTimestampAuthorityBlocked(ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON)
    return verified_gpu_proof


def _require_stage_itl_gpu_proof(
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof
    | VerifiedDistributedRuntimeGpuProof,
    *,
    expected_topology: str,
) -> VerifiedNativeRuntimeGpuProof | VerifiedDistributedRuntimeGpuProof:
    """Accept only a topology-matched proof whose registered suite covers ITL."""

    if expected_topology == "tp1_dp1":
        return _require_native_itl_gpu_proof(
            verified_gpu_proof
            if type(verified_gpu_proof) is VerifiedNativeRuntimeGpuProof
            else None
        )
    if (
        type(verified_gpu_proof) is not VerifiedDistributedRuntimeGpuProof
        or verified_gpu_proof.topology_mode != expected_topology
        or "native_itl_pointer"
        not in DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS[verified_gpu_proof.topology_mode]
    ):
        raise ItlTimestampAuthorityBlocked(ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON)
    return verified_gpu_proof


def evaluate_e2_itl_timestamp_activation(
    plan: E2ItlTimestampPlan,
    *,
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
) -> ItlTimestampActivation:
    if type(plan) is not E2ItlTimestampPlan:
        raise TypeError("ITL activation requires an exact E2 plan")
    plan.__post_init__()
    if len(RELEASE_ITL_TIMESTAMP_PRODUCERS) > 1:
        raise RuntimeError("release ITL timestamp producer allowlist is ambiguous")
    release_producer = (
        RELEASE_ITL_TIMESTAMP_PRODUCERS[0] if RELEASE_ITL_TIMESTAMP_PRODUCERS else None
    )
    if plan.producer != release_producer:
        raise ValueError("ITL plan producer differs from source-owned release policy")
    if plan.producer is None:
        return ItlTimestampActivation(
            status="BLOCKED",
            reason_code=ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON,
            plan_sha256=plan.sha256,
            producer_sha256=None,
        )
    try:
        _require_native_itl_gpu_proof(verified_gpu_proof)
    except ItlTimestampAuthorityBlocked as error:
        return ItlTimestampActivation(
            status="BLOCKED",
            reason_code=error.reason,
            plan_sha256=plan.sha256,
            producer_sha256=None,
        )
    return ItlTimestampActivation(
        status="READY",
        reason_code=None,
        plan_sha256=plan.sha256,
        producer_sha256=plan.producer.sha256,
    )


def require_e2_itl_timestamp_prelaunch(
    plan: E2ItlTimestampPlan,
    *,
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
) -> ReleaseItlTimestampProducer:
    activation = evaluate_e2_itl_timestamp_activation(
        plan,
        verified_gpu_proof=verified_gpu_proof,
    )
    if activation.status != "READY" or plan.producer is None:
        raise ItlTimestampAuthorityBlocked(
            activation.reason_code or ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON
        )
    return plan.producer


@dataclass(frozen=True)
class ItlRequestTimestamps:
    request_id: str
    request_started_ns: int
    request_terminal_ns: int
    output_token_ids: tuple[int, ...]
    token_observed_ns: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_safe_id("ITL request", self.request_id)
        _strict_int("ITL request start", self.request_started_ns)
        _strict_int("ITL request terminal", self.request_terminal_ns)
        if self.request_terminal_ns < self.request_started_ns:
            raise ValueError("ITL request terminal precedes its start")
        if len(self.output_token_ids) < 2:
            raise ValueError("formal ITL requires at least two generated tokens")
        if len(self.token_observed_ns) != len(self.output_token_ids):
            raise ValueError("ITL token timestamp coverage is incomplete")
        for token_id in self.output_token_ids:
            _strict_int("ITL output token ID", token_id)
        for observed_ns in self.token_observed_ns:
            _strict_int("ITL token timestamp", observed_ns)
        if any(
            current <= previous
            for previous, current in pairwise(self.token_observed_ns)
        ):
            raise ValueError("ITL token timestamps must be strictly increasing")
        if (
            self.token_observed_ns[0] < self.request_started_ns
            or self.token_observed_ns[-1] > self.request_terminal_ns
        ):
            raise ValueError("ITL token timestamps are outside the request lifetime")

    @property
    def inter_token_ns(self) -> tuple[int, ...]:
        return tuple(
            current - previous for previous, current in pairwise(self.token_observed_ns)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "request_started_ns": self.request_started_ns,
            "request_terminal_ns": self.request_terminal_ns,
            "output_token_ids": list(self.output_token_ids),
            "token_observed_ns": list(self.token_observed_ns),
        }

    @classmethod
    def from_dict(cls, value: object) -> ItlRequestTimestamps:
        row = _strict_mapping("bound ITL request timestamps", value)
        _strict_keys(
            "bound ITL request timestamps",
            row,
            {
                "request_id",
                "request_started_ns",
                "request_terminal_ns",
                "output_token_ids",
                "token_observed_ns",
            },
        )
        return cls(
            request_id=_require_safe_id("ITL request", row["request_id"]),
            request_started_ns=_strict_int(
                "ITL request start", row["request_started_ns"]
            ),
            request_terminal_ns=_strict_int(
                "ITL request terminal", row["request_terminal_ns"]
            ),
            output_token_ids=tuple(
                _strict_int("ITL output token ID", token_id)
                for token_id in _strict_sequence(
                    "ITL output token IDs", row["output_token_ids"]
                )
            ),
            token_observed_ns=tuple(
                _strict_int("ITL token timestamp", timestamp)
                for timestamp in _strict_sequence(
                    "ITL token timestamps", row["token_observed_ns"]
                )
            ),
        )


@dataclass(frozen=True)
class ItlRequestExpectation:
    """Terminal request identity derived outside the timestamp producer."""

    request_id: str
    output_token_ids: tuple[int, ...]
    terminal_status: Literal["completed"]

    def __post_init__(self) -> None:
        _require_safe_id("ITL expected request", self.request_id)
        if len(self.output_token_ids) < 2:
            raise ValueError("formal ITL expectation requires at least two tokens")
        for token_id in self.output_token_ids:
            _strict_int("ITL expected output token ID", token_id)
        if self.terminal_status != "completed":
            raise ValueError("formal ITL expectation requires completed status")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "output_token_ids": list(self.output_token_ids),
            "terminal_status": self.terminal_status,
        }

    @classmethod
    def from_dict(cls, value: object) -> ItlRequestExpectation:
        row = _strict_mapping("ITL request expectation", value)
        _strict_keys(
            "ITL request expectation",
            row,
            {
                "request_id",
                "output_token_ids",
                "terminal_status",
            },
        )
        return cls(
            request_id=_require_safe_id("ITL expected request", row["request_id"]),
            output_token_ids=tuple(
                _strict_int("ITL expected output token ID", token_id)
                for token_id in _strict_sequence(
                    "ITL expected output token IDs", row["output_token_ids"]
                )
            ),
            terminal_status=row["terminal_status"],  # type: ignore[arg-type]
        )


def itl_request_expectations_sha256(
    expectations: Sequence[ItlRequestExpectation],
) -> str:
    values = tuple(expectations)
    if (
        not values
        or any(type(value) is not ItlRequestExpectation for value in values)
        or tuple(value.request_id for value in values)
        != tuple(sorted({value.request_id for value in values}))
    ):
        raise ValueError("ITL expectations must be sorted, non-empty, and unique")
    for value in values:
        value.__post_init__()
    return content_sha256([value.to_dict() for value in values])


_BOUND_ITL_AUTHORITY_SENTINEL = object()


@dataclass(frozen=True, init=False)
class BoundItlTimestampAuthority:
    schema_version: int
    kind: str
    plan: E2ItlTimestampPlan
    raw_receipt_path: str
    raw_receipt_sha256: str
    producer_sha256: str
    gpu_proof_receipt_sha256: str
    native_result_pointer_sha256s: tuple[str, ...]
    expectations_sha256: str
    expectations: tuple[ItlRequestExpectation, ...]
    requests: tuple[ItlRequestTimestamps, ...]

    def __init__(
        self,
        *,
        schema_version: int,
        kind: str,
        plan: E2ItlTimestampPlan,
        raw_receipt_path: str,
        raw_receipt_sha256: str,
        producer_sha256: str,
        gpu_proof_receipt_sha256: str,
        native_result_pointer_sha256s: tuple[str, ...],
        expectations_sha256: str,
        expectations: tuple[ItlRequestExpectation, ...],
        requests: tuple[ItlRequestTimestamps, ...],
        verified_gpu_proof: VerifiedNativeRuntimeGpuProof,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _BOUND_ITL_AUTHORITY_SENTINEL:
            raise TypeError(
                "bound ITL authority requires first-party pointer validation"
            )
        proof = _require_native_itl_gpu_proof(verified_gpu_proof)
        if gpu_proof_receipt_sha256 != proof.receipt_sha256:
            raise ValueError("ITL authority uses another GPU qualification proof")
        for name, value in (
            ("schema_version", schema_version),
            ("kind", kind),
            ("plan", plan),
            ("raw_receipt_path", raw_receipt_path),
            ("raw_receipt_sha256", raw_receipt_sha256),
            ("producer_sha256", producer_sha256),
            ("gpu_proof_receipt_sha256", gpu_proof_receipt_sha256),
            ("native_result_pointer_sha256s", native_result_pointer_sha256s),
            ("expectations_sha256", expectations_sha256),
            ("expectations", expectations),
            ("requests", requests),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 3:
            raise ValueError("bound ITL timestamp authority schema is unsupported")
        if self.kind != "bound_itl_timestamp_authority":
            raise ValueError("bound ITL timestamp authority schema is unsupported")
        if type(self.plan) is not E2ItlTimestampPlan:
            raise TypeError("bound ITL authority requires an exact E2 plan")
        self.plan.__post_init__()
        if len(RELEASE_ITL_TIMESTAMP_PRODUCERS) != 1:
            raise RuntimeError("release ITL timestamp producer allowlist is ambiguous")
        producer = RELEASE_ITL_TIMESTAMP_PRODUCERS[0]
        if self.plan.producer != producer:
            raise ValueError("ITL bound producer differs from release policy")
        path = Path(self.raw_receipt_path)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError("ITL raw receipt path must be absolute and resolved")
        _require_sha256("ITL raw receipt", self.raw_receipt_sha256)
        _require_sha256("ITL bound producer", self.producer_sha256)
        _require_sha256("ITL GPU proof receipt", self.gpu_proof_receipt_sha256)
        _require_sha256("ITL expectations", self.expectations_sha256)
        if self.producer_sha256 != producer.sha256:
            raise ValueError("ITL bound producer differs from the release plan")
        if self.expectations_sha256 != itl_request_expectations_sha256(
            self.expectations
        ):
            raise ValueError("ITL expected request binding changed")
        if (
            type(self.native_result_pointer_sha256s) is not tuple
            or len(self.native_result_pointer_sha256s) != len(self.expectations)
            or len(set(self.native_result_pointer_sha256s))
            != len(self.native_result_pointer_sha256s)
        ):
            raise ValueError("ITL first-party result pointer coverage is incomplete")
        for digest in self.native_result_pointer_sha256s:
            _require_sha256("ITL native result pointer", digest)
        if not self.requests or tuple(row.request_id for row in self.requests) != tuple(
            sorted({row.request_id for row in self.requests})
        ):
            raise ValueError(
                "ITL request receipts must be sorted, non-empty, and unique"
            )
        for row in self.requests:
            row.__post_init__()
        if tuple(row.request_id for row in self.requests) != tuple(
            row.request_id for row in self.expectations
        ):
            raise ValueError("ITL raw receipt omits an expected request")
        for expected, observed in zip(self.expectations, self.requests, strict=True):
            if (
                expected.output_token_ids != observed.output_token_ids
                or expected.terminal_status != "completed"
            ):
                raise ValueError("ITL raw request differs from terminal expectations")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "plan": self.plan.to_dict(),
            "raw_receipt_path": self.raw_receipt_path,
            "raw_receipt_sha256": self.raw_receipt_sha256,
            "producer_sha256": self.producer_sha256,
            "gpu_proof_receipt_sha256": self.gpu_proof_receipt_sha256,
            "native_result_pointer_sha256s": list(self.native_result_pointer_sha256s),
            "expectations_sha256": self.expectations_sha256,
            "expectations": [value.to_dict() for value in self.expectations],
            "requests": [row.to_dict() for row in self.requests],
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        verified_gpu_proof: VerifiedNativeRuntimeGpuProof,
    ) -> BoundItlTimestampAuthority:
        row = _strict_mapping("bound ITL timestamp authority", value)
        _strict_keys(
            "bound ITL timestamp authority",
            row,
            {
                "schema_version",
                "kind",
                "plan",
                "raw_receipt_path",
                "raw_receipt_sha256",
                "producer_sha256",
                "gpu_proof_receipt_sha256",
                "native_result_pointer_sha256s",
                "expectations_sha256",
                "expectations",
                "requests",
            },
        )
        return cls(
            schema_version=_strict_int(
                "bound ITL authority schema", row["schema_version"]
            ),
            kind=row["kind"],  # type: ignore[arg-type]
            plan=E2ItlTimestampPlan.from_dict(row["plan"]),
            raw_receipt_path=_require_exact_text(
                "ITL raw receipt path", row["raw_receipt_path"]
            ),
            raw_receipt_sha256=_require_sha256(
                "ITL raw receipt", row["raw_receipt_sha256"]
            ),
            producer_sha256=_require_sha256(
                "ITL bound producer", row["producer_sha256"]
            ),
            gpu_proof_receipt_sha256=_require_sha256(
                "ITL GPU proof receipt", row["gpu_proof_receipt_sha256"]
            ),
            native_result_pointer_sha256s=tuple(
                _require_sha256("ITL native result pointer", value)
                for value in _strict_sequence(
                    "ITL native result pointers",
                    row["native_result_pointer_sha256s"],
                )
            ),
            expectations_sha256=_require_sha256(
                "ITL expectations", row["expectations_sha256"]
            ),
            expectations=tuple(
                ItlRequestExpectation.from_dict(item)
                for item in _strict_sequence(
                    "ITL authority expectations", row["expectations"]
                )
            ),
            requests=tuple(
                ItlRequestTimestamps.from_dict(item)
                for item in _strict_sequence("ITL authority requests", row["requests"])
            ),
            verified_gpu_proof=verified_gpu_proof,
            _verification_tag=_BOUND_ITL_AUTHORITY_SENTINEL,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def token_timestamps_for(self, request_id: str) -> tuple[int, ...]:
        matches = tuple(
            row.token_observed_ns
            for row in self.requests
            if row.request_id == request_id
        )
        if len(matches) != 1:
            raise KeyError(request_id)
        return matches[0]


def _read_stable_receipt(path_value: str | Path) -> tuple[Path, bytes]:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("ITL raw receipt path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ItlTimestampAuthorityBlocked(ITL_RAW_RECEIPT_MISSING_REASON) from error
    if resolved != path:
        raise ValueError("ITL raw receipt path must be resolved and non-symlink")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        directory_descriptor = os.open(path.parent, directory_flags)
    except OSError as error:
        raise ValueError("ITL raw receipt parent is not a stable directory") from error
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        directory_opened = os.fstat(directory_descriptor)
        directory_current = path.parent.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(directory_opened.st_mode)
            or directory_opened.st_dev != directory_current.st_dev
            or directory_opened.st_ino != directory_current.st_ino
        ):
            raise ValueError("ITL raw receipt parent changed before read")
        descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        os.close(directory_descriptor)
        raise ValueError("ITL raw receipt cannot be opened safely") from error
    except BaseException:
        os.close(directory_descriptor)
        raise
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_RAW_RECEIPT_BYTES
        ):
            raise ValueError("ITL raw receipt must be a bounded regular file")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("ITL raw receipt changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("ITL raw receipt grew while being read")
        after = os.fstat(descriptor)
        current = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        directory_after = os.fstat(directory_descriptor)
        directory_current = path.parent.stat(follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or after.st_nlink != 1
            or identity(before) != identity(after)
            or identity(after) != identity(current)
            or directory_after.st_dev != directory_opened.st_dev
            or directory_after.st_ino != directory_opened.st_ino
            or directory_current.st_dev != directory_opened.st_dev
            or directory_current.st_ino != directory_opened.st_ino
        ):
            raise ValueError("ITL raw receipt changed during coordinated read")
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)
    return path, b"".join(chunks)


def _load_strict_json(raw: bytes) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"ITL raw receipt contains duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"ITL raw receipt contains non-finite {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ITL raw receipt is not strict UTF-8 JSON") from error
    return _strict_mapping("ITL raw receipt", value)


def _native_request_timing(row: Mapping[str, object]) -> ItlRequestTimestamps:
    _strict_keys(
        "native ITL request",
        row,
        {
            "request_id",
            "request_started_ns",
            "request_terminal_ns",
            "output_token_ids",
            "token_events",
        },
    )
    output_values = _strict_sequence("ITL output token IDs", row["output_token_ids"])
    output_ids = tuple(
        _strict_int("ITL output token ID", value) for value in output_values
    )
    events = _strict_sequence("native ITL token events", row["token_events"])
    observed: list[int] = []
    for index, event_value in enumerate(events):
        event = _strict_mapping("native ITL token event", event_value)
        _strict_keys(
            "native ITL token event",
            event,
            {"token_index", "token_id", "observed_ns"},
        )
        if (
            _strict_int("native ITL token index", event["token_index"]) != index
            or index >= len(output_ids)
            or _strict_int("native ITL token ID", event["token_id"])
            != output_ids[index]
        ):
            raise ValueError("native ITL events differ from ordered output tokens")
        observed.append(_strict_int("native ITL observation", event["observed_ns"]))
    return ItlRequestTimestamps(
        request_id=_require_safe_id("ITL request", row["request_id"]),
        request_started_ns=_strict_int("ITL request start", row["request_started_ns"]),
        request_terminal_ns=_strict_int(
            "ITL request terminal", row["request_terminal_ns"]
        ),
        output_token_ids=output_ids,
        token_observed_ns=tuple(observed),
    )


def _sse_request_timing(row: Mapping[str, object]) -> ItlRequestTimestamps:
    _strict_keys(
        "SSE ITL request",
        row,
        {
            "request_id",
            "request_started_ns",
            "request_terminal_ns",
            "output_token_ids",
            "raw_sse_frames",
        },
    )
    output_values = _strict_sequence("ITL output token IDs", row["output_token_ids"])
    output_ids = tuple(
        _strict_int("ITL output token ID", value) for value in output_values
    )
    frames = _strict_sequence("raw SSE frames", row["raw_sse_frames"])
    observed: list[int] = []
    for index, frame_value in enumerate(frames):
        frame = _strict_mapping("raw SSE frame", frame_value)
        _strict_keys(
            "raw SSE frame",
            frame,
            {"frame_index", "new_token_ids", "observed_ns"},
        )
        new_tokens = _strict_sequence("raw SSE new token IDs", frame["new_token_ids"])
        if len(new_tokens) != 1:
            raise ItlTimestampAuthorityBlocked(ITL_COALESCED_CHUNK_UNPROVEN_REASON)
        token_id = _strict_int("raw SSE token ID", new_tokens[0])
        if (
            _strict_int("raw SSE frame index", frame["frame_index"]) != index
            or index >= len(output_ids)
            or token_id != output_ids[index]
        ):
            raise ValueError("raw SSE frames differ from ordered output tokens")
        observed.append(_strict_int("raw SSE observation", frame["observed_ns"]))
    return ItlRequestTimestamps(
        request_id=_require_safe_id("ITL request", row["request_id"]),
        request_started_ns=_strict_int("ITL request start", row["request_started_ns"]),
        request_terminal_ns=_strict_int(
            "ITL request terminal", row["request_terminal_ns"]
        ),
        output_token_ids=output_ids,
        token_observed_ns=tuple(observed),
    )


def _native_result_pointer_timing(
    value: object,
) -> tuple[ItlRequestTimestamps, str]:
    pointer = _strict_mapping("native ITL result pointer", value)
    _strict_keys(
        "native ITL result pointer",
        pointer,
        {
            "schema_version",
            "kind",
            "hook",
            "semantics",
            "release_status",
            "request_id",
            "request_started_ns",
            "request_terminal_ns",
            "terminal_status",
            "terminal_reason",
            "events",
            "result_pointer_sha256",
        },
    )
    pointer_sha256 = _require_sha256(
        "native ITL result pointer", pointer["result_pointer_sha256"]
    )
    unsigned = dict(pointer)
    unsigned.pop("result_pointer_sha256")
    if content_sha256(unsigned) != pointer_sha256:
        raise ValueError("native ITL result pointer content digest mismatch")
    if (
        type(pointer["schema_version"]) is not int
        or pointer["schema_version"] != 1
        or pointer["kind"] != SGLANG_NATIVE_ITL_RESULT_POINTER_KIND
        or pointer["hook"] != SGLANG_NATIVE_ITL_CONTRACT_HOOK
        or pointer["semantics"] != SGLANG_NATIVE_ITL_CONTRACT_SEMANTICS
        or pointer["release_status"] != SGLANG_NATIVE_ITL_CONTRACT_RELEASE_STATUS
        or pointer["terminal_status"] != "completed"
    ):
        raise ValueError("native ITL result pointer differs from release semantics")
    _require_exact_text("native ITL terminal reason", pointer["terminal_reason"])
    raw_events = _strict_sequence("native ITL result events", pointer["events"])
    output_ids: list[int] = []
    for event_value in raw_events:
        event = _strict_mapping("native ITL result event", event_value)
        _strict_keys(
            "native ITL result event",
            event,
            {"token_index", "token_id", "observed_ns"},
        )
        output_ids.append(_strict_int("native ITL token ID", event["token_id"]))
    timing = _native_request_timing(
        {
            "request_id": pointer["request_id"],
            "request_started_ns": pointer["request_started_ns"],
            "request_terminal_ns": pointer["request_terminal_ns"],
            "output_token_ids": output_ids,
            "token_events": list(raw_events),
        }
    )
    return timing, pointer_sha256


StageItlMethod = Literal[
    "target_only",
    "static",
    "tts",
    "l0",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
]


@dataclass(frozen=True)
class StageItlExecutionIdentity:
    """Scientific-stage-neutral identity joined to native and timing proofs."""

    schema_version: int
    kind: str
    materialized_cell_id: str
    inventory_sha256: str
    registry_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    run_id: str
    run_nonce_sha256: str
    attempt_id: str
    method: StageItlMethod

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "stage_itl_execution_identity":
            raise ValueError("stage ITL execution identity schema is unsupported")
        for label, value in (
            ("stage ITL materialized cell", self.materialized_cell_id),
            ("stage ITL inventory", self.inventory_sha256),
            ("stage ITL registry", self.registry_sha256),
            ("stage ITL execution plan", self.execution_plan_sha256),
            ("stage ITL rank config", self.rank_config_sha256),
            ("stage ITL run nonce", self.run_nonce_sha256),
        ):
            _require_sha256(label, value)
        _require_safe_id("stage ITL run", self.run_id)
        _require_safe_id("stage ITL attempt", self.attempt_id)
        if self.method not in {
            "target_only",
            "static",
            "tts",
            "l0",
            "onlinespec_ogd",
            "onlinespec_opt",
            "onlinespec_ens",
        }:
            raise ValueError("stage ITL method is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "materialized_cell_id": self.materialized_cell_id,
            "inventory_sha256": self.inventory_sha256,
            "registry_sha256": self.registry_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "run_id": self.run_id,
            "run_nonce_sha256": self.run_nonce_sha256,
            "attempt_id": self.attempt_id,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, value: object) -> StageItlExecutionIdentity:
        row = _strict_mapping("stage ITL execution identity", value)
        _strict_keys(
            "stage ITL execution identity",
            row,
            {
                "schema_version",
                "kind",
                "materialized_cell_id",
                "inventory_sha256",
                "registry_sha256",
                "execution_plan_sha256",
                "rank_config_sha256",
                "run_id",
                "run_nonce_sha256",
                "attempt_id",
                "method",
            },
        )
        return cls(**row)  # type: ignore[arg-type]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _stage_itl_expectations_from_native_result(
    result: object,
) -> tuple[ItlRequestExpectation, ...]:
    # Local import avoids the experiments -> orchestration package-init ->
    # executor -> experiments cycle during registry/planning collection.
    from lightcone_spec.orchestration.formal_terminal_result import (
        FormalDistributedTerminalRequestResult,
        FormalDistributedTerminalResultProjection,
    )
    from lightcone_spec.orchestration.native_terminal import (
        NativeTerminalRequestResult,
        NativeTerminalResultProjection,
    )

    if type(result) not in {
        NativeTerminalResultProjection,
        FormalDistributedTerminalResultProjection,
    }:
        raise TypeError("stage ITL requires one exact formal terminal projection")
    requests = tuple(sorted(result.requests, key=lambda row: row.request_id))
    if (
        not requests
        or any(
            type(row)
            not in {
                NativeTerminalRequestResult,
                FormalDistributedTerminalRequestResult,
            }
            for row in requests
        )
        or len({row.request_id for row in requests}) != len(requests)
    ):
        raise ValueError("stage ITL formal request coverage is incomplete")
    expectations: list[ItlRequestExpectation] = []
    for row in requests:
        if (
            not row.submitted_to_server
            or row.terminal_status != "completed"
            or row.output_token_ids is None
        ):
            raise ItlTimestampAuthorityBlocked(
                ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON
            )
        expectations.append(
            ItlRequestExpectation(
                request_id=row.request_id,
                output_token_ids=row.output_token_ids,
                terminal_status="completed",
            )
        )
    values = tuple(expectations)
    itl_request_expectations_sha256(values)
    return values


@dataclass(frozen=True)
class StageItlExternalControlBinding:
    """Exact timing/result/proof lineage signed by the local release control."""

    schema_version: int
    kind: str
    execution_identity_sha256: str
    raw_receipt_raw_sha256: str
    raw_receipt_semantic_sha256: str
    native_result_proof_raw_sha256: str
    native_result_proof_semantic_sha256: str
    native_gpu_proof_raw_sha256: str
    native_gpu_proof_semantic_sha256: str
    native_gpu_verified_proof_sha256: str
    producer_sha256: str
    expectations_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "stage_itl_external_control_binding"
        ):
            raise ValueError("stage ITL control binding schema is unsupported")
        for label, value in (
            ("stage ITL execution identity", self.execution_identity_sha256),
            ("stage ITL raw receipt raw", self.raw_receipt_raw_sha256),
            ("stage ITL raw receipt semantic", self.raw_receipt_semantic_sha256),
            ("stage ITL native result raw", self.native_result_proof_raw_sha256),
            (
                "stage ITL native result semantic",
                self.native_result_proof_semantic_sha256,
            ),
            ("stage ITL native GPU proof raw", self.native_gpu_proof_raw_sha256),
            (
                "stage ITL native GPU proof semantic",
                self.native_gpu_proof_semantic_sha256,
            ),
            (
                "stage ITL verified GPU proof",
                self.native_gpu_verified_proof_sha256,
            ),
            ("stage ITL producer", self.producer_sha256),
            ("stage ITL expectations", self.expectations_sha256),
        ):
            _require_sha256(label, value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "execution_identity_sha256": self.execution_identity_sha256,
            "raw_receipt_raw_sha256": self.raw_receipt_raw_sha256,
            "raw_receipt_semantic_sha256": self.raw_receipt_semantic_sha256,
            "native_result_proof_raw_sha256": self.native_result_proof_raw_sha256,
            "native_result_proof_semantic_sha256": (
                self.native_result_proof_semantic_sha256
            ),
            "native_gpu_proof_raw_sha256": self.native_gpu_proof_raw_sha256,
            "native_gpu_proof_semantic_sha256": (self.native_gpu_proof_semantic_sha256),
            "native_gpu_verified_proof_sha256": (self.native_gpu_verified_proof_sha256),
            "producer_sha256": self.producer_sha256,
            "expectations_sha256": self.expectations_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @cached_property
    def lineage_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "stage_itl_external_control_lineage",
                "binding_sha256": self.sha256,
                "execution_identity_sha256": self.execution_identity_sha256,
                "native_result_proof_semantic_sha256": (
                    self.native_result_proof_semantic_sha256
                ),
                "native_gpu_verified_proof_sha256": (
                    self.native_gpu_verified_proof_sha256
                ),
                "producer_sha256": self.producer_sha256,
                "expectations_sha256": self.expectations_sha256,
            }
        )


_STAGE_ITL_AUTHORITY_SENTINEL = object()


@dataclass(frozen=True, init=False)
class StageItlTimestampAuthority:
    """Sealed request timing authority joined to one formal execution."""

    schema_version: int
    kind: str
    execution_identity: StageItlExecutionIdentity
    raw_receipt_path: str
    raw_receipt_raw_sha256: str
    raw_receipt_semantic_sha256: str
    native_result_proof_path: str
    native_result_proof_raw_sha256: str
    native_result_proof_semantic_sha256: str
    native_gpu_proof_path: str
    native_gpu_proof_raw_sha256: str
    native_gpu_proof_semantic_sha256: str
    native_gpu_verified_proof_sha256: str
    producer_sha256: str
    control_binding_sha256: str
    control_envelope_sha256: str
    replay_reservation_sha256: str
    expectations_sha256: str
    native_result_pointer_sha256s: tuple[str, ...]
    requests: tuple[ItlRequestTimestamps, ...]

    def __init__(
        self,
        *,
        execution_identity: StageItlExecutionIdentity,
        raw_receipt: CanonicalJsonProofBinding,
        native_result_proof: CanonicalJsonProofBinding,
        native_gpu_proof: CanonicalJsonProofBinding,
        native_gpu_verified_proof_sha256: str,
        producer_sha256: str,
        control_binding_sha256: str,
        control_envelope_sha256: str,
        replay_reservation_sha256: str,
        expectations_sha256: str,
        native_result_pointer_sha256s: tuple[str, ...],
        requests: tuple[ItlRequestTimestamps, ...],
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _STAGE_ITL_AUTHORITY_SENTINEL:
            raise TypeError("stage ITL authority requires first-party validation")
        values: dict[str, object] = {
            "schema_version": 1,
            "kind": "stage_itl_timestamp_authority",
            "execution_identity": execution_identity,
            "raw_receipt_path": raw_receipt.absolute_path,
            "raw_receipt_raw_sha256": raw_receipt.raw_sha256,
            "raw_receipt_semantic_sha256": raw_receipt.semantic_sha256,
            "native_result_proof_path": native_result_proof.absolute_path,
            "native_result_proof_raw_sha256": native_result_proof.raw_sha256,
            "native_result_proof_semantic_sha256": (
                native_result_proof.semantic_sha256
            ),
            "native_gpu_proof_path": native_gpu_proof.absolute_path,
            "native_gpu_proof_raw_sha256": native_gpu_proof.raw_sha256,
            "native_gpu_proof_semantic_sha256": native_gpu_proof.semantic_sha256,
            "native_gpu_verified_proof_sha256": native_gpu_verified_proof_sha256,
            "producer_sha256": producer_sha256,
            "control_binding_sha256": control_binding_sha256,
            "control_envelope_sha256": control_envelope_sha256,
            "replay_reservation_sha256": replay_reservation_sha256,
            "expectations_sha256": expectations_sha256,
            "native_result_pointer_sha256s": native_result_pointer_sha256s,
            "requests": requests,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "stage_itl_timestamp_authority":
            raise ValueError("stage ITL authority schema is unsupported")
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("stage ITL authority execution identity is not exact")
        self.execution_identity.__post_init__()
        for label, path_value in (
            ("stage ITL raw receipt", self.raw_receipt_path),
            ("stage ITL native result proof", self.native_result_proof_path),
            ("stage ITL native GPU proof", self.native_gpu_proof_path),
        ):
            path = Path(path_value)
            if not path.is_absolute() or Path(os.path.abspath(path)) != path:
                raise ValueError(f"{label} path must be absolute and normalized")
        for label, value in (
            ("stage ITL raw receipt raw", self.raw_receipt_raw_sha256),
            ("stage ITL raw receipt semantic", self.raw_receipt_semantic_sha256),
            ("stage ITL native result raw", self.native_result_proof_raw_sha256),
            (
                "stage ITL native result semantic",
                self.native_result_proof_semantic_sha256,
            ),
            ("stage ITL native GPU raw", self.native_gpu_proof_raw_sha256),
            ("stage ITL native GPU semantic", self.native_gpu_proof_semantic_sha256),
            (
                "stage ITL native GPU verified",
                self.native_gpu_verified_proof_sha256,
            ),
            ("stage ITL producer", self.producer_sha256),
            ("stage ITL control binding", self.control_binding_sha256),
            ("stage ITL control envelope", self.control_envelope_sha256),
            ("stage ITL replay reservation", self.replay_reservation_sha256),
            ("stage ITL expectations", self.expectations_sha256),
        ):
            _require_sha256(label, value)
        if (
            not self.native_result_pointer_sha256s
            or len(self.native_result_pointer_sha256s) != len(self.requests)
            or len(set(self.native_result_pointer_sha256s))
            != len(self.native_result_pointer_sha256s)
        ):
            raise ValueError("stage ITL pointer coverage is incomplete")
        for value in self.native_result_pointer_sha256s:
            _require_sha256("stage ITL native result pointer", value)
        if tuple(row.request_id for row in self.requests) != tuple(
            sorted({row.request_id for row in self.requests})
        ):
            raise ValueError("stage ITL request timing coverage is not exact")
        for row in self.requests:
            row.__post_init__()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "execution_identity": self.execution_identity.to_dict(),
            "raw_receipt_path": self.raw_receipt_path,
            "raw_receipt_raw_sha256": self.raw_receipt_raw_sha256,
            "raw_receipt_semantic_sha256": self.raw_receipt_semantic_sha256,
            "native_result_proof_path": self.native_result_proof_path,
            "native_result_proof_raw_sha256": self.native_result_proof_raw_sha256,
            "native_result_proof_semantic_sha256": (
                self.native_result_proof_semantic_sha256
            ),
            "native_gpu_proof_path": self.native_gpu_proof_path,
            "native_gpu_proof_raw_sha256": self.native_gpu_proof_raw_sha256,
            "native_gpu_proof_semantic_sha256": (self.native_gpu_proof_semantic_sha256),
            "native_gpu_verified_proof_sha256": (self.native_gpu_verified_proof_sha256),
            "producer_sha256": self.producer_sha256,
            "control_binding_sha256": self.control_binding_sha256,
            "control_envelope_sha256": self.control_envelope_sha256,
            "replay_reservation_sha256": self.replay_reservation_sha256,
            "expectations_sha256": self.expectations_sha256,
            "native_result_pointer_sha256s": list(self.native_result_pointer_sha256s),
            "requests": [row.to_dict() for row in self.requests],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def client_timing_inputs(self) -> tuple[dict[str, object], ...]:
        """Return integer-only raw inputs; never accept caller-derived floats."""

        return tuple(
            {
                "request_id": row.request_id,
                "arrival_ns": row.request_started_ns,
                "first_token_ns": row.token_observed_ns[0],
                "completion_ns": row.request_terminal_ns,
                "output_token_ids": list(row.output_token_ids),
                "native_per_token_observed_ns": list(row.token_observed_ns),
                "inter_token_ns": list(row.inter_token_ns),
            }
            for row in self.requests
        )

    @property
    def throughput_numerator_tokens(self) -> int:
        return sum(len(row.output_token_ids) for row in self.requests)

    @property
    def throughput_window_ns(self) -> int:
        window = max(row.request_terminal_ns for row in self.requests) - min(
            row.request_started_ns for row in self.requests
        )
        if window <= 0:
            raise ValueError("stage ITL throughput window is not positive")
        return window

    @property
    def p99_itl_input_ns(self) -> tuple[int, ...]:
        values = tuple(value for row in self.requests for value in row.inter_token_ns)
        if not values:
            raise ValueError("stage ITL p99 input is empty")
        return values


@dataclass(frozen=True)
class StageItlTimestampProofArtifact:
    """Durable path-bound stage-neutral timing authority."""

    schema_version: int
    kind: str
    raw_receipt: CanonicalJsonProofBinding
    native_result_proof: CanonicalJsonProofBinding
    native_gpu_proof: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    batch_control_attestations: tuple[ControlArtifactAttestation, ...]
    replay_reservation: ChallengeReplayReservationBinding
    expected_root_manifest_sha256: str
    execution_identity: StageItlExecutionIdentity
    authority: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.kind != (
            "stage_itl_timestamp_proof_artifact"
        ):
            raise ValueError("stage ITL proof artifact schema is unsupported")
        for value, label in (
            (self.raw_receipt, "stage ITL raw receipt"),
            (self.native_result_proof, "stage ITL native result proof"),
            (self.native_gpu_proof, "stage ITL native GPU proof"),
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError(f"{label} binding is not exact")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("stage ITL proof control envelope is not exact")
        if (
            type(self.batch_control_attestations) is not tuple
            or not self.batch_control_attestations
            or any(
                type(value) is not ControlArtifactAttestation
                for value in self.batch_control_attestations
            )
        ):
            raise TypeError("stage ITL proof control batch is not exact")
        if tuple(
            sorted(self.batch_control_attestations, key=lambda value: value.sha256)
        ) != self.batch_control_attestations or len(
            {value.sha256 for value in self.batch_control_attestations}
        ) != len(self.batch_control_attestations):
            raise ValueError("stage ITL proof control batch is not canonical")
        if self.control_attestation not in self.batch_control_attestations:
            raise ValueError("stage ITL proof control is absent from its batch")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("stage ITL proof replay reservation is not exact")
        _require_sha256(
            "stage ITL proof release root", self.expected_root_manifest_sha256
        )
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("stage ITL proof execution identity is not exact")
        self.execution_identity.__post_init__()
        if (
            type(self.authority) is not dict
            or self.authority.get("kind") != "stage_itl_timestamp_authority"
        ):
            raise TypeError("stage ITL proof authority projection is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_receipt": self.raw_receipt.to_dict(),
            "native_result_proof": self.native_result_proof.to_dict(),
            "native_gpu_proof": self.native_gpu_proof.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "batch_control_attestations": [
                value.to_dict() for value in self.batch_control_attestations
            ],
            "replay_reservation": self.replay_reservation.to_dict(),
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
            "execution_identity": self.execution_identity.to_dict(),
            "authority": self.authority,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> StageItlTimestampProofArtifact:
        row = dict(_strict_mapping("stage ITL proof artifact", value))
        _strict_keys(
            "stage ITL proof artifact",
            row,
            {
                "schema_version",
                "kind",
                "raw_receipt",
                "native_result_proof",
                "native_gpu_proof",
                "control_attestation",
                "batch_control_attestations",
                "replay_reservation",
                "expected_root_manifest_sha256",
                "execution_identity",
                "authority",
            },
        )
        raw_receipt = CanonicalJsonProofBinding.from_dict(row.pop("raw_receipt"))
        native_result = CanonicalJsonProofBinding.from_dict(
            row.pop("native_result_proof")
        )
        native_gpu = CanonicalJsonProofBinding.from_dict(row.pop("native_gpu_proof"))
        control = ControlArtifactAttestation.from_dict(row.pop("control_attestation"))
        batch_control_value = row.pop("batch_control_attestations")
        if type(batch_control_value) is not list:
            raise TypeError("stage ITL proof control batch must be a list")
        batch_controls = tuple(
            ControlArtifactAttestation.from_dict(value) for value in batch_control_value
        )
        reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("replay_reservation")
        )
        identity = StageItlExecutionIdentity.from_dict(row.pop("execution_identity"))
        authority = row.pop("authority")
        if type(authority) is not dict:
            raise TypeError("stage ITL proof authority must be an object")
        return cls(
            **row,
            raw_receipt=raw_receipt,
            native_result_proof=native_result,
            native_gpu_proof=native_gpu,
            control_attestation=control,
            batch_control_attestations=batch_controls,
            replay_reservation=reservation,
            execution_identity=identity,
            authority=authority,
        )


@dataclass(frozen=True)
class StageItlTimestampProofRequest:
    """One fully path-bound member of an atomic stage-ITL proof batch."""

    raw_receipt_path: str
    native_result_proof_path: str
    native_gpu_proof_path: str
    execution_identity: StageItlExecutionIdentity
    control_attestation: ControlArtifactAttestation
    proof_artifact_path: str

    def __post_init__(self) -> None:
        for label, value in (
            ("stage ITL raw receipt path", self.raw_receipt_path),
            ("stage ITL native result proof path", self.native_result_proof_path),
            ("stage ITL native GPU proof path", self.native_gpu_proof_path),
            ("stage ITL proof output path", self.proof_artifact_path),
        ):
            if type(value) is not str or not value:
                raise TypeError(f"{label} must be a non-empty string")
            path = Path(value)
            if not path.is_absolute() or Path(os.path.abspath(path)) != path:
                raise ValueError(f"{label} must be absolute and normalized")
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("stage ITL batch execution identity is not exact")
        self.execution_identity.__post_init__()
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("stage ITL batch control envelope is not exact")


def _stage_itl_upstream_replay_root(
    result_binding: CanonicalJsonProofBinding,
    gpu_binding: CanonicalJsonProofBinding,
) -> Path:
    from lightcone_spec.orchestration.formal_terminal_result import (
        FormalCurrentPreflightTp1TerminalResultProofArtifact,
        FormalDistributedTerminalResultProofArtifact,
        FormalPreflightTp1TerminalResultProofArtifact,
        FormalTp1TerminalResultProofArtifact,
    )
    from lightcone_spec.orchestration.native_terminal import (
        NativeTerminalResultProofArtifact,
    )

    result_value = result_binding.reopen()
    kind = result_value.get("kind")
    if kind == "native_terminal_result_proof_artifact":
        result_artifact = NativeTerminalResultProofArtifact.from_dict(result_value)
        result_sha256 = result_artifact.sha256
        result_replay_path = result_artifact.replay_reservation.path
    elif kind == "formal_tp1_terminal_result_proof_artifact":
        formal_tp1 = FormalTp1TerminalResultProofArtifact.from_dict(result_value)
        result_sha256 = formal_tp1.sha256
        result_replay_path = formal_tp1.replay_reservation.path
    elif kind == "formal_preflight_tp1_terminal_result_proof_artifact":
        preflight_tp1 = FormalPreflightTp1TerminalResultProofArtifact.from_dict(
            result_value
        )
        native = NativeTerminalResultProofArtifact.from_dict(
            preflight_tp1.native_result_proof.reopen()
        )
        result_sha256 = preflight_tp1.sha256
        result_replay_path = native.replay_reservation.path
    elif kind == "formal_current_preflight_tp1_terminal_result_proof_artifact":
        current_preflight = (
            FormalCurrentPreflightTp1TerminalResultProofArtifact.from_dict(result_value)
        )
        native = NativeTerminalResultProofArtifact.from_dict(
            current_preflight.native_result_proof.reopen()
        )
        result_sha256 = current_preflight.sha256
        result_replay_path = native.replay_reservation.path
    elif kind == "formal_distributed_terminal_result_proof_artifact":
        distributed = FormalDistributedTerminalResultProofArtifact.from_dict(
            result_value
        )
        result_sha256 = distributed.sha256
        result_replay_path = distributed.replay_reservation.path
    else:
        raise ValueError("stage ITL terminal proof kind is unsupported")
    gpu_value = gpu_binding.reopen()
    if type(gpu_value) is not dict:
        raise TypeError("stage ITL GPU proof artifact must be an object")
    if gpu_value.get("kind") == "lightcone_native_runtime_gpu_proof_artifact":
        gpu_artifact = NativeRuntimeGpuProofArtifact.from_dict(gpu_value)
    elif gpu_value.get("kind") == "lightcone_distributed_runtime_gpu_proof_artifact":
        gpu_artifact = DistributedRuntimeGpuProofArtifact.from_dict(gpu_value)
    else:
        raise ValueError("stage ITL GPU proof kind is unsupported")
    if (
        result_binding.semantic_sha256 != result_sha256
        or gpu_binding.semantic_sha256 != gpu_artifact.sha256
    ):
        raise ValueError("stage ITL upstream artifact identity differs")
    roots = {
        Path(result_replay_path).parent,
        Path(gpu_artifact.replay_reservation.path).parent,
    }
    if len(roots) != 1:
        raise ValueError("stage ITL upstream proofs use different replay ledgers")
    return next(iter(roots))


def _stage_itl_inputs(
    *,
    execution_identity: StageItlExecutionIdentity,
    raw_receipt_path: str,
    native_result_proof_path: str,
    native_gpu_proof_path: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> tuple[
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    VerifiedNativeRuntimeGpuProof | VerifiedDistributedRuntimeGpuProof,
    ReleaseItlTimestampProducer,
    tuple[ItlRequestExpectation, ...],
    tuple[ItlRequestTimestamps, ...],
    tuple[str, ...],
]:
    """Deep-open every upstream proof and parse exact native timing pointers."""

    from lightcone_spec.orchestration.formal_terminal_result import (
        validate_formal_terminal_result_proof_artifact,
    )

    if type(execution_identity) is not StageItlExecutionIdentity:
        raise TypeError("stage ITL inputs require an exact execution identity")
    execution_identity.__post_init__()
    _require_sha256("stage ITL release root", expected_root_manifest_sha256)
    result_binding = CanonicalJsonProofBinding.bind(native_result_proof_path)
    result = validate_formal_terminal_result_proof_artifact(
        native_result_proof_path,
        expected_inventory_sha256=execution_identity.inventory_sha256,
        expected_registry_sha256=execution_identity.registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_execution_plan_sha256=execution_identity.execution_plan_sha256,
        expected_rank_config_sha256=execution_identity.rank_config_sha256,
        expected_run_id=execution_identity.run_id,
        expected_run_nonce_sha256=execution_identity.run_nonce_sha256,
        expected_attempt_id=execution_identity.attempt_id,
        expected_method=execution_identity.method,
        now_ns=now_ns,
    )
    expectations = _stage_itl_expectations_from_native_result(result)
    expected_topology = getattr(result, "topology_mode", "tp1_dp1")
    gpu_binding = CanonicalJsonProofBinding.bind(native_gpu_proof_path)
    gpu_value = gpu_binding.reopen()
    if type(gpu_value) is not dict:
        raise TypeError("stage ITL GPU proof artifact must be an object")
    if gpu_value.get("kind") == "lightcone_native_runtime_gpu_proof_artifact":
        gpu_artifact = NativeRuntimeGpuProofArtifact.from_dict(gpu_value)
    elif gpu_value.get("kind") == "lightcone_distributed_runtime_gpu_proof_artifact":
        gpu_artifact = DistributedRuntimeGpuProofArtifact.from_dict(gpu_value)
    else:
        raise ValueError("stage ITL GPU proof kind is unsupported")
    if gpu_binding.semantic_sha256 != gpu_artifact.sha256:
        raise ValueError("stage ITL GPU proof file identity differs")
    if (
        gpu_artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("stage ITL GPU proof uses another release root")
    _stage_itl_upstream_replay_root(result_binding, gpu_binding)
    proof = _require_stage_itl_gpu_proof(
        gpu_artifact.revalidate(now_ns=now_ns),
        expected_topology=expected_topology,
    )
    if proof.inventory_sha256 != execution_identity.inventory_sha256:
        raise ValueError("stage ITL GPU proof uses another inventory")
    if len(RELEASE_ITL_TIMESTAMP_PRODUCERS) != 1:
        raise ItlTimestampAuthorityBlocked(ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON)
    producer = RELEASE_ITL_TIMESTAMP_PRODUCERS[0]
    producer.__post_init__()
    raw_binding = CanonicalJsonProofBinding.bind(raw_receipt_path)
    receipt = _strict_mapping("stage ITL raw receipt", raw_binding.reopen())
    _strict_keys(
        "stage ITL raw receipt",
        receipt,
        {
            "schema_version",
            "kind",
            "execution_identity_sha256",
            "native_result_proof_semantic_sha256",
            "native_gpu_proof_semantic_sha256",
            "native_gpu_verified_proof_sha256",
            "producer_id",
            "producer_version_sha256",
            "source_mode",
            "hook_id",
            "clock",
            "complete",
            "native_result_pointers",
        },
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "formal_stage_itl_timestamp_raw_receipt"
        or receipt["execution_identity_sha256"] != execution_identity.sha256
        or receipt["native_result_proof_semantic_sha256"]
        != result_binding.semantic_sha256
        or receipt["native_gpu_proof_semantic_sha256"] != gpu_binding.semantic_sha256
        or receipt["native_gpu_verified_proof_sha256"] != proof.sha256
        or receipt["producer_id"] != producer.producer_id
        or receipt["producer_version_sha256"] != producer.producer_version_sha256
        or receipt["source_mode"] != producer.source_mode
        or receipt["hook_id"] != producer.hook_id
        or receipt["clock"] != producer.clock
        or receipt["complete"] is not True
    ):
        raise ValueError("stage ITL receipt differs from its proof lineage")
    raw_pointers = _strict_sequence(
        "stage ITL native result pointers", receipt["native_result_pointers"]
    )
    parsed = tuple(
        sorted(
            (_native_result_pointer_timing(value) for value in raw_pointers),
            key=lambda value: value[0].request_id,
        )
    )
    requests = tuple(value[0] for value in parsed)
    pointer_sha256s = tuple(value[1] for value in parsed)
    if tuple(row.request_id for row in requests) != tuple(
        row.request_id for row in expectations
    ):
        raise ItlTimestampAuthorityBlocked(ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON)
    for expected, observed in zip(expectations, requests, strict=True):
        if expected.output_token_ids != observed.output_token_ids:
            raise ValueError("stage ITL result differs from native terminal tokens")
    return (
        raw_binding,
        result_binding,
        gpu_binding,
        proof,
        producer,
        expectations,
        requests,
        pointer_sha256s,
    )


def publish_stage_itl_timestamp_raw_receipt(
    raw_receipt_path: str,
    *,
    native_result_proof_path: str,
    native_gpu_proof_path: str,
    execution_identity: StageItlExecutionIdentity,
    expected_root_manifest_sha256: str,
    native_result_pointers: tuple[Mapping[str, object], ...],
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Publish an unsigned canonical bundle of first-party native pointers.

    This producer has no signing key and grants no authority.  The returned
    raw binding must be pulled to the local release signer and authorized
    by :func:`publish_stage_itl_timestamp_proof_artifact`.
    """

    from lightcone_spec.orchestration.formal_terminal_result import (
        validate_formal_terminal_result_proof_artifact,
    )

    if type(execution_identity) is not StageItlExecutionIdentity:
        raise TypeError("stage ITL raw receipt requires exact execution identity")
    execution_identity.__post_init__()
    if type(native_result_pointers) is not tuple or not native_result_pointers:
        raise TypeError("stage ITL raw receipt requires exact native pointers")
    result_binding = CanonicalJsonProofBinding.bind(native_result_proof_path)
    result = validate_formal_terminal_result_proof_artifact(
        native_result_proof_path,
        expected_inventory_sha256=execution_identity.inventory_sha256,
        expected_registry_sha256=execution_identity.registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_execution_plan_sha256=execution_identity.execution_plan_sha256,
        expected_rank_config_sha256=execution_identity.rank_config_sha256,
        expected_run_id=execution_identity.run_id,
        expected_run_nonce_sha256=execution_identity.run_nonce_sha256,
        expected_attempt_id=execution_identity.attempt_id,
        expected_method=execution_identity.method,
        now_ns=now_ns,
    )
    expectations = _stage_itl_expectations_from_native_result(result)
    expected_topology = getattr(result, "topology_mode", "tp1_dp1")
    gpu_binding = CanonicalJsonProofBinding.bind(native_gpu_proof_path)
    gpu_value = gpu_binding.reopen()
    if type(gpu_value) is not dict:
        raise TypeError("stage ITL GPU proof artifact must be an object")
    if gpu_value.get("kind") == "lightcone_native_runtime_gpu_proof_artifact":
        gpu_artifact = NativeRuntimeGpuProofArtifact.from_dict(gpu_value)
    elif gpu_value.get("kind") == "lightcone_distributed_runtime_gpu_proof_artifact":
        gpu_artifact = DistributedRuntimeGpuProofArtifact.from_dict(gpu_value)
    else:
        raise ValueError("stage ITL GPU proof kind is unsupported")
    if gpu_binding.semantic_sha256 != gpu_artifact.sha256:
        raise ValueError("stage ITL raw receipt GPU proof identity differs")
    if (
        gpu_artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("stage ITL raw receipt GPU proof uses another release root")
    _stage_itl_upstream_replay_root(result_binding, gpu_binding)
    proof = _require_stage_itl_gpu_proof(
        gpu_artifact.revalidate(now_ns=now_ns),
        expected_topology=expected_topology,
    )
    if proof.inventory_sha256 != execution_identity.inventory_sha256:
        raise ValueError("stage ITL raw receipt GPU inventory differs")
    if len(RELEASE_ITL_TIMESTAMP_PRODUCERS) != 1:
        raise ItlTimestampAuthorityBlocked(ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON)
    producer = RELEASE_ITL_TIMESTAMP_PRODUCERS[0]
    parsed = tuple(
        sorted(
            (_native_result_pointer_timing(value) for value in native_result_pointers),
            key=lambda value: value[0].request_id,
        )
    )
    if tuple(value[0].request_id for value in parsed) != tuple(
        row.request_id for row in expectations
    ) or any(
        expected.output_token_ids != observed.output_token_ids
        for expected, (observed, _) in zip(expectations, parsed, strict=True)
    ):
        raise ValueError("stage ITL raw pointers differ from terminal requests")
    receipt = {
        "schema_version": 1,
        "kind": "formal_stage_itl_timestamp_raw_receipt",
        "execution_identity_sha256": execution_identity.sha256,
        "native_result_proof_semantic_sha256": result_binding.semantic_sha256,
        "native_gpu_proof_semantic_sha256": gpu_binding.semantic_sha256,
        "native_gpu_verified_proof_sha256": proof.sha256,
        "producer_id": producer.producer_id,
        "producer_version_sha256": producer.producer_version_sha256,
        "source_mode": producer.source_mode,
        "hook_id": producer.hook_id,
        "clock": producer.clock,
        "complete": True,
        "native_result_pointers": [dict(value) for value in native_result_pointers],
    }
    publish_canonical_json_no_replace(raw_receipt_path, receipt)
    return CanonicalJsonProofBinding.bind(raw_receipt_path)


def build_stage_itl_external_control_binding(
    raw_receipt_path: str,
    *,
    native_result_proof_path: str,
    native_gpu_proof_path: str,
    execution_identity: StageItlExecutionIdentity,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> StageItlExternalControlBinding:
    """Build the exact subject binding that the offline control authorizes."""

    (
        raw_binding,
        result_binding,
        gpu_binding,
        proof,
        producer,
        expectations,
        _,
        _,
    ) = _stage_itl_inputs(
        execution_identity=execution_identity,
        raw_receipt_path=raw_receipt_path,
        native_result_proof_path=native_result_proof_path,
        native_gpu_proof_path=native_gpu_proof_path,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
    )
    return StageItlExternalControlBinding(
        schema_version=1,
        kind="stage_itl_external_control_binding",
        execution_identity_sha256=execution_identity.sha256,
        raw_receipt_raw_sha256=raw_binding.raw_sha256,
        raw_receipt_semantic_sha256=raw_binding.semantic_sha256,
        native_result_proof_raw_sha256=result_binding.raw_sha256,
        native_result_proof_semantic_sha256=result_binding.semantic_sha256,
        native_gpu_proof_raw_sha256=gpu_binding.raw_sha256,
        native_gpu_proof_semantic_sha256=gpu_binding.semantic_sha256,
        native_gpu_verified_proof_sha256=proof.sha256,
        producer_sha256=producer.sha256,
        expectations_sha256=itl_request_expectations_sha256(expectations),
    )


def build_stage_itl_control_subject(
    raw_receipt_path: str,
    *,
    native_result_proof_path: str,
    native_gpu_proof_path: str,
    execution_identity: StageItlExecutionIdentity,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> ControlArtifactSubject:
    """Derive the exact offline-signing subject from reopened ITL inputs."""

    binding = build_stage_itl_external_control_binding(
        raw_receipt_path,
        native_result_proof_path=native_result_proof_path,
        native_gpu_proof_path=native_gpu_proof_path,
        execution_identity=execution_identity,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
    )
    return ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=binding.sha256,
        protocol_sha256=STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256,
        registry_sha256=execution_identity.registry_sha256,
        lineage_sha256=binding.lineage_sha256,
    )


def _stage_itl_authority(
    *,
    execution_identity: StageItlExecutionIdentity,
    raw_binding: CanonicalJsonProofBinding,
    result_binding: CanonicalJsonProofBinding,
    gpu_binding: CanonicalJsonProofBinding,
    proof: VerifiedNativeRuntimeGpuProof | VerifiedDistributedRuntimeGpuProof,
    producer: ReleaseItlTimestampProducer,
    expectations: tuple[ItlRequestExpectation, ...],
    requests: tuple[ItlRequestTimestamps, ...],
    pointer_sha256s: tuple[str, ...],
    control_binding: StageItlExternalControlBinding,
    control: ControlArtifactAttestation,
    reservation: ChallengeReplayReservationBinding,
) -> StageItlTimestampAuthority:
    return StageItlTimestampAuthority(
        execution_identity=execution_identity,
        raw_receipt=raw_binding,
        native_result_proof=result_binding,
        native_gpu_proof=gpu_binding,
        native_gpu_verified_proof_sha256=proof.sha256,
        producer_sha256=producer.sha256,
        control_binding_sha256=control_binding.sha256,
        control_envelope_sha256=control.sha256,
        replay_reservation_sha256=reservation.reservation_sha256,
        expectations_sha256=itl_request_expectations_sha256(expectations),
        native_result_pointer_sha256s=pointer_sha256s,
        requests=requests,
        _verification_tag=_STAGE_ITL_AUTHORITY_SENTINEL,
    )


def publish_stage_itl_timestamp_proof_artifact(
    raw_receipt_path: str,
    *,
    native_result_proof_path: str,
    native_gpu_proof_path: str,
    execution_identity: StageItlExecutionIdentity,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str,
) -> CanonicalJsonProofBinding:
    """Trust-lift one native timing receipt and publish its durable proof."""

    return publish_stage_itl_timestamp_proof_artifacts(
        (
            StageItlTimestampProofRequest(
                raw_receipt_path=raw_receipt_path,
                native_result_proof_path=native_result_proof_path,
                native_gpu_proof_path=native_gpu_proof_path,
                execution_identity=execution_identity,
                control_attestation=control_attestation,
                proof_artifact_path=proof_artifact_path,
            ),
        ),
        replay_store=replay_store,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
    )[0]


def publish_stage_itl_timestamp_proof_artifacts(
    requests: tuple[StageItlTimestampProofRequest, ...],
    *,
    replay_store: ChallengeReplayStore,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> tuple[CanonicalJsonProofBinding, ...]:
    """Trust-lift a complete timing batch under one replay reservation.

    Every raw file, upstream durable proof, execution identity, control subject,
    and destination is reopened before the replay ledger is mutated.  This is
    the formal eight-row preflight path: a bad row cannot consume any challenge
    or publish any proof from the batch.
    """

    if type(requests) is not tuple or not requests:
        raise TypeError("stage ITL proof batch must be a non-empty exact tuple")
    if any(type(request) is not StageItlTimestampProofRequest for request in requests):
        raise TypeError("stage ITL proof batch request is not exact")
    _require_sha256("stage ITL expected release root", expected_root_manifest_sha256)
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("stage ITL proof verification time is invalid")

    output_paths = tuple(Path(request.proof_artifact_path) for request in requests)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("stage ITL proof output paths must be unique")
    for output_path in output_paths:
        if output_path.exists():
            raise ValueError("stage ITL proof output must be a new absolute path")

    inventory_sha256s = {
        request.execution_identity.inventory_sha256 for request in requests
    }
    if len(inventory_sha256s) != 1:
        raise ValueError("stage ITL proof batch spans multiple GPU inventories")
    expected_inventory_sha256 = next(iter(inventory_sha256s))

    prepared: list[
        tuple[
            StageItlTimestampProofRequest,
            CanonicalJsonProofBinding,
            CanonicalJsonProofBinding,
            CanonicalJsonProofBinding,
            VerifiedNativeRuntimeGpuProof | VerifiedDistributedRuntimeGpuProof,
            ReleaseItlTimestampProducer,
            tuple[ItlRequestExpectation, ...],
            tuple[ItlRequestTimestamps, ...],
            tuple[str, ...],
            StageItlExternalControlBinding,
        ]
    ] = []
    upstream_roots: set[Path] = set()
    for request in requests:
        request.__post_init__()
        (
            raw_binding,
            result_binding,
            gpu_binding,
            proof,
            producer,
            expectations,
            request_timestamps,
            pointer_sha256s,
        ) = _stage_itl_inputs(
            execution_identity=request.execution_identity,
            raw_receipt_path=request.raw_receipt_path,
            native_result_proof_path=request.native_result_proof_path,
            native_gpu_proof_path=request.native_gpu_proof_path,
            expected_root_manifest_sha256=expected_root_manifest_sha256,
            now_ns=now_ns,
        )
        upstream_roots.add(_stage_itl_upstream_replay_root(result_binding, gpu_binding))
        control_binding = StageItlExternalControlBinding(
            schema_version=1,
            kind="stage_itl_external_control_binding",
            execution_identity_sha256=request.execution_identity.sha256,
            raw_receipt_raw_sha256=raw_binding.raw_sha256,
            raw_receipt_semantic_sha256=raw_binding.semantic_sha256,
            native_result_proof_raw_sha256=result_binding.raw_sha256,
            native_result_proof_semantic_sha256=result_binding.semantic_sha256,
            native_gpu_proof_raw_sha256=gpu_binding.raw_sha256,
            native_gpu_proof_semantic_sha256=gpu_binding.semantic_sha256,
            native_gpu_verified_proof_sha256=proof.sha256,
            producer_sha256=producer.sha256,
            expectations_sha256=itl_request_expectations_sha256(expectations),
        )
        control = request.control_attestation
        subject = control.subject
        if (
            control.deployment_policy_authorization.root_manifest_sha256
            != expected_root_manifest_sha256
            or control.hardware_envelope_sha256 != proof.hardware_envelope_sha256
            or subject.artifact_type != "non_serving_terminal"
            or subject.artifact_sha256 != control_binding.sha256
            or subject.protocol_sha256 != STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256
            or subject.registry_sha256 != request.execution_identity.registry_sha256
            or subject.lineage_sha256 != control_binding.lineage_sha256
        ):
            raise ValueError("stage ITL control subject is not exact")
        prepared.append(
            (
                request,
                raw_binding,
                result_binding,
                gpu_binding,
                proof,
                producer,
                expectations,
                request_timestamps,
                pointer_sha256s,
                control_binding,
            )
        )

    if upstream_roots != {Path(replay_store.root)}:
        raise ValueError("stage ITL proof batch must use the upstream replay ledger")

    controls = tuple(request.control_attestation for request in requests)
    verified_rows = verify_and_reserve_release_control_artifact_attestations(
        controls,
        expected_inventory_sha256=expected_inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified_rows,
        reserved_ns=now_ns,
    )
    reservation = replay_store.bind_reservation(reservation_sha256)
    canonical_controls = tuple(sorted(controls, key=lambda value: value.sha256))

    artifacts: list[tuple[str, StageItlTimestampProofArtifact]] = []
    for (
        request,
        raw_binding,
        result_binding,
        gpu_binding,
        proof,
        producer,
        expectations,
        request_timestamps,
        pointer_sha256s,
        control_binding,
    ) in prepared:
        authority = _stage_itl_authority(
            execution_identity=request.execution_identity,
            raw_binding=raw_binding,
            result_binding=result_binding,
            gpu_binding=gpu_binding,
            proof=proof,
            producer=producer,
            expectations=expectations,
            requests=request_timestamps,
            pointer_sha256s=pointer_sha256s,
            control_binding=control_binding,
            control=request.control_attestation,
            reservation=reservation,
        )
        artifacts.append(
            (
                request.proof_artifact_path,
                StageItlTimestampProofArtifact(
                    schema_version=2,
                    kind="stage_itl_timestamp_proof_artifact",
                    raw_receipt=raw_binding,
                    native_result_proof=result_binding,
                    native_gpu_proof=gpu_binding,
                    control_attestation=request.control_attestation,
                    batch_control_attestations=canonical_controls,
                    replay_reservation=reservation,
                    expected_root_manifest_sha256=expected_root_manifest_sha256,
                    execution_identity=request.execution_identity,
                    authority=authority.to_dict(),
                ),
            )
        )

    bindings: list[CanonicalJsonProofBinding] = []
    try:
        for output_path, artifact in artifacts:
            publish_canonical_json_no_replace(output_path, artifact.to_dict())
            bindings.append(
                CanonicalJsonProofBinding.bind(
                    output_path,
                    semantic_sha256=artifact.sha256,
                )
            )
    except Exception as error:
        raise RuntimeError(
            "stage ITL proof batch publication failed after reservation; "
            "discard every partial output and issue new controls"
        ) from error
    return tuple(bindings)


def validate_stage_itl_timestamp_proof_artifact(
    proof_artifact_path: str,
    *,
    expected_execution_identity: StageItlExecutionIdentity,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> StageItlTimestampAuthority:
    """Deep-reopen a stage ITL proof without consuming challenges again."""

    if type(expected_execution_identity) is not StageItlExecutionIdentity:
        raise TypeError("stage ITL validation requires exact execution identity")
    expected_execution_identity.__post_init__()
    _require_sha256("stage ITL expected release root", expected_root_manifest_sha256)
    proof_binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = StageItlTimestampProofArtifact.from_dict(proof_binding.reopen())
    if (
        proof_binding.semantic_sha256 != artifact.sha256
        or artifact.execution_identity != expected_execution_identity
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
    ):
        raise ValueError("stage ITL proof file identity differs")
    inputs = _stage_itl_inputs(
        execution_identity=expected_execution_identity,
        raw_receipt_path=artifact.raw_receipt.absolute_path,
        native_result_proof_path=artifact.native_result_proof.absolute_path,
        native_gpu_proof_path=artifact.native_gpu_proof.absolute_path,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
    )
    (
        raw_binding,
        result_binding,
        gpu_binding,
        proof,
        producer,
        expectations,
        requests,
        pointer_sha256s,
    ) = inputs
    if (
        raw_binding != artifact.raw_receipt
        or result_binding != artifact.native_result_proof
        or gpu_binding != artifact.native_gpu_proof
    ):
        raise ValueError("stage ITL upstream proof binding changed")
    if Path(artifact.replay_reservation.path).parent != (
        _stage_itl_upstream_replay_root(result_binding, gpu_binding)
    ):
        raise ValueError("stage ITL proof replay ledger differs from upstream")
    control_binding = StageItlExternalControlBinding(
        schema_version=1,
        kind="stage_itl_external_control_binding",
        execution_identity_sha256=expected_execution_identity.sha256,
        raw_receipt_raw_sha256=raw_binding.raw_sha256,
        raw_receipt_semantic_sha256=raw_binding.semantic_sha256,
        native_result_proof_raw_sha256=result_binding.raw_sha256,
        native_result_proof_semantic_sha256=result_binding.semantic_sha256,
        native_gpu_proof_raw_sha256=gpu_binding.raw_sha256,
        native_gpu_proof_semantic_sha256=gpu_binding.semantic_sha256,
        native_gpu_verified_proof_sha256=proof.sha256,
        producer_sha256=producer.sha256,
        expectations_sha256=itl_request_expectations_sha256(expectations),
    )
    control = artifact.control_attestation
    subject = control.subject
    if (
        control.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
        or control.hardware_envelope_sha256 != proof.hardware_envelope_sha256
        or subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != control_binding.sha256
        or subject.protocol_sha256 != STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256
        or subject.registry_sha256 != expected_execution_identity.registry_sha256
        or subject.lineage_sha256 != control_binding.lineage_sha256
    ):
        raise ValueError("stage ITL proof control subject differs")
    reserved = artifact.replay_reservation.revalidate()
    if type(now_ns) is not int or now_ns < artifact.replay_reservation.reserved_ns:
        raise ValueError("stage ITL proof time precedes reservation")
    verified_batch = tuple(
        verify_release_control_artifact_attestation(
            batch_control,
            expected_inventory_sha256=expected_execution_identity.inventory_sha256,
            now_ns=artifact.replay_reservation.reserved_ns,
            consumed_challenge_sha256s=(),
        )
        for batch_control in artifact.batch_control_attestations
    )
    expected_challenges = tuple(
        sorted(
            {
                *(value.challenge_sha256 for value in verified_batch),
                *(value.deployment_policy_challenge_sha256 for value in verified_batch),
            }
        )
    )
    expected_reservation = control_challenge_reservation_sha256(
        verified_batch,
        reserved_ns=artifact.replay_reservation.reserved_ns,
    )
    if (
        reserved != expected_challenges
        or artifact.replay_reservation.reservation_sha256 != expected_reservation
    ):
        raise ValueError("stage ITL proof replay reservation differs")
    authority = _stage_itl_authority(
        execution_identity=expected_execution_identity,
        raw_binding=raw_binding,
        result_binding=result_binding,
        gpu_binding=gpu_binding,
        proof=proof,
        producer=producer,
        expectations=expectations,
        requests=requests,
        pointer_sha256s=pointer_sha256s,
        control_binding=control_binding,
        control=control,
        reservation=artifact.replay_reservation,
    )
    if authority.to_dict() != artifact.authority:
        raise ValueError("stage ITL proof derived authority changed")
    return authority


def bind_itl_timestamp_authority(
    plan: E2ItlTimestampPlan,
    raw_receipt_path: str | Path,
    *,
    expected_requests: tuple[ItlRequestExpectation, ...],
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
) -> BoundItlTimestampAuthority:
    """Deep-open native result pointers under an exact verified GPU proof."""

    producer = require_e2_itl_timestamp_prelaunch(
        plan,
        verified_gpu_proof=verified_gpu_proof,
    )
    proof = _require_native_itl_gpu_proof(verified_gpu_proof)
    expectations_sha256 = itl_request_expectations_sha256(expected_requests)
    path, raw = _read_stable_receipt(raw_receipt_path)
    receipt = _load_strict_json(raw)
    _strict_keys(
        "formal ITL raw receipt",
        receipt,
        {
            "schema_version",
            "kind",
            "plan_sha256",
            "producer_id",
            "producer_version_sha256",
            "source_mode",
            "hook_id",
            "clock",
            "gpu_proof_receipt_sha256",
            "complete",
            "native_result_pointers",
        },
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["kind"] != "formal_e2_itl_timestamp_raw_receipt"
        or receipt["plan_sha256"] != plan.sha256
        or receipt["producer_id"] != producer.producer_id
        or receipt["producer_version_sha256"] != producer.producer_version_sha256
        or receipt["source_mode"] != producer.source_mode
        or receipt["hook_id"] != producer.hook_id
        or receipt["clock"] != producer.clock
        or receipt["gpu_proof_receipt_sha256"] != proof.receipt_sha256
        or receipt["complete"] is not True
    ):
        raise ValueError("formal ITL receipt differs from its release authority")
    raw_pointers = _strict_sequence(
        "formal ITL native result pointers", receipt["native_result_pointers"]
    )
    parsed = tuple(
        sorted(
            (_native_result_pointer_timing(value) for value in raw_pointers),
            key=lambda value: value[0].request_id,
        )
    )
    requests = tuple(value[0] for value in parsed)
    pointer_sha256s = tuple(value[1] for value in parsed)
    if tuple(row.request_id for row in requests) != tuple(
        row.request_id for row in expected_requests
    ):
        raise ItlTimestampAuthorityBlocked(ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON)
    for expected, observed in zip(expected_requests, requests, strict=True):
        if expected.output_token_ids != observed.output_token_ids:
            raise ValueError("native ITL result differs from terminal expectations")
    return BoundItlTimestampAuthority(
        schema_version=3,
        kind="bound_itl_timestamp_authority",
        plan=plan,
        raw_receipt_path=str(path),
        raw_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        producer_sha256=producer.sha256,
        gpu_proof_receipt_sha256=proof.receipt_sha256,
        native_result_pointer_sha256s=pointer_sha256s,
        expectations_sha256=expectations_sha256,
        expectations=expected_requests,
        requests=requests,
        verified_gpu_proof=proof,
        _verification_tag=_BOUND_ITL_AUTHORITY_SENTINEL,
    )


def _parse_itl_timestamp_receipt_for_cpu_test(
    plan: E2ItlTimestampPlan,
    raw_receipt_path: str | Path,
    *,
    expected_requests: tuple[ItlRequestExpectation, ...],
) -> tuple[ItlRequestTimestamps, ...]:
    """Exercise the strict timestamp parser without minting formal authority."""

    if type(plan) is not E2ItlTimestampPlan:
        raise TypeError("CPU ITL parser requires an exact E2 plan")
    plan.__post_init__()
    if len(RELEASE_ITL_TIMESTAMP_PRODUCERS) != 1:
        raise ValueError("CPU ITL parser requires one explicit test producer")
    producer = RELEASE_ITL_TIMESTAMP_PRODUCERS[0]
    if plan.producer != producer:
        raise ValueError("CPU ITL parser plan differs from its test producer")
    itl_request_expectations_sha256(expected_requests)
    path, raw = _read_stable_receipt(raw_receipt_path)
    receipt = _load_strict_json(raw)
    _strict_keys(
        "ITL raw receipt",
        receipt,
        {
            "schema_version",
            "kind",
            "plan_sha256",
            "producer_id",
            "producer_version_sha256",
            "source_mode",
            "hook_id",
            "clock",
            "complete",
            "requests",
        },
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["kind"] != "cpu_test_itl_timestamp_raw_receipt"
        or receipt["plan_sha256"] != plan.sha256
        or receipt["producer_id"] != producer.producer_id
        or receipt["producer_version_sha256"] != producer.producer_version_sha256
        or receipt["source_mode"] != producer.source_mode
        or receipt["hook_id"] != producer.hook_id
        or receipt["clock"] != producer.clock
        or receipt["complete"] is not True
    ):
        raise ValueError("ITL raw receipt differs from its release plan")
    request_values = _strict_sequence("ITL raw requests", receipt["requests"])
    if not request_values:
        raise ItlTimestampAuthorityBlocked(ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON)
    reducer = (
        _native_request_timing
        if producer.source_mode == "native_per_token_timestamp_hook"
        else _sse_request_timing
    )
    requests = tuple(
        sorted(
            (
                reducer(_strict_mapping("ITL raw request", value))
                for value in request_values
            ),
            key=lambda value: value.request_id,
        )
    )
    if tuple(row.request_id for row in requests) != tuple(
        row.request_id for row in expected_requests
    ):
        raise ValueError("ITL raw receipt omits an expected request")
    for expected, observed in zip(expected_requests, requests, strict=True):
        if expected.output_token_ids != observed.output_token_ids:
            raise ValueError("ITL raw request differs from terminal expectations")
    # ``path`` and ``raw`` are deliberately consumed only for strict parsing;
    # no digest or object returned here is formal authority.
    _ = path, raw
    return requests


def revalidate_itl_timestamp_authority(
    authority: BoundItlTimestampAuthority,
    *,
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
) -> BoundItlTimestampAuthority:
    if type(authority) is not BoundItlTimestampAuthority:
        raise TypeError("ITL revalidation requires an exact bound authority")
    authority.__post_init__()
    rebound = bind_itl_timestamp_authority(
        authority.plan,
        authority.raw_receipt_path,
        expected_requests=authority.expectations,
        verified_gpu_proof=verified_gpu_proof,
    )
    if rebound != authority or rebound.sha256 != authority.sha256:
        raise ValueError("ITL timestamp authority changed during revalidation")
    return rebound


def replay_e2_itl_timestamp_plan(
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    value: object,
) -> E2ItlTimestampPlan:
    """Rebuild the source-owned prelaunch plan and reject caller substitutions."""

    declared = E2ItlTimestampPlan.from_dict(value)
    expected = release_e2_itl_timestamp_plan(registry, cell)
    if declared != expected or declared.sha256 != expected.sha256:
        raise ValueError("E2 ITL timestamp plan differs from source-owned replay")
    return expected


@dataclass(frozen=True)
class PathBoundItlTimestampAuthority:
    """Stable outer authority file plus its recursively replayed raw receipt."""

    authority_path: str
    authority_file_sha256: str
    authority: BoundItlTimestampAuthority

    def __post_init__(self) -> None:
        path = Path(self.authority_path)
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError("ITL authority path must be absolute and resolved")
        _require_sha256("ITL authority file", self.authority_file_sha256)
        if type(self.authority) is not BoundItlTimestampAuthority:
            raise TypeError("path-bound ITL authority requires an exact authority")
        self.authority.__post_init__()

    def to_dict(self) -> dict[str, object]:
        return {
            "authority_path": self.authority_path,
            "authority_file_sha256": self.authority_file_sha256,
            "authority_sha256": self.authority.sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def load_path_bound_itl_timestamp_authority(
    authority_path: str | Path,
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    expected_requests: tuple[ItlRequestExpectation, ...],
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
) -> PathBoundItlTimestampAuthority:
    """Open one path-only authority and replay plan, coverage, and raw bytes."""

    expected_plan = release_e2_itl_timestamp_plan(registry, cell)
    # The source-owned release policy is evaluated before even resolving the
    # caller-supplied path.  An unavailable/CPU-only producer therefore cannot
    # use filesystem observations as a substitute capability probe.
    require_e2_itl_timestamp_prelaunch(
        expected_plan,
        verified_gpu_proof=verified_gpu_proof,
    )
    path, raw = _read_stable_receipt(authority_path)
    value = _load_strict_json(raw)
    proof = _require_native_itl_gpu_proof(verified_gpu_proof)
    authority = BoundItlTimestampAuthority.from_dict(
        value,
        verified_gpu_proof=proof,
    )
    if authority.plan != expected_plan or authority.plan.sha256 != expected_plan.sha256:
        raise ValueError("bound ITL authority plan differs from source-owned replay")
    if authority.expectations != expected_requests:
        raise ValueError("bound ITL authority differs from terminal expectations")
    rebound = revalidate_itl_timestamp_authority(
        authority,
        verified_gpu_proof=proof,
    )
    if rebound != authority or rebound.sha256 != authority.sha256:
        raise ValueError("bound ITL authority changed during raw replay")
    return PathBoundItlTimestampAuthority(
        authority_path=str(path),
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority=rebound,
    )


def revalidate_path_bound_itl_timestamp_authority(
    binding: PathBoundItlTimestampAuthority,
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    expected_requests: tuple[ItlRequestExpectation, ...],
    verified_gpu_proof: VerifiedNativeRuntimeGpuProof | None = None,
) -> PathBoundItlTimestampAuthority:
    """Reopen both the outer authority and its raw producer receipt."""

    if type(binding) is not PathBoundItlTimestampAuthority:
        raise TypeError("ITL path revalidation requires an exact binding")
    binding.__post_init__()
    rebound = load_path_bound_itl_timestamp_authority(
        binding.authority_path,
        registry=registry,
        cell=cell,
        expected_requests=expected_requests,
        verified_gpu_proof=verified_gpu_proof,
    )
    if rebound != binding or rebound.sha256 != binding.sha256:
        raise ValueError("path-bound ITL timestamp authority changed")
    return rebound


def assess_serving_chunks_for_formal_itl(
    *,
    request_id: str,
    output_tokens: int,
    chunks: Sequence[TokenChunkTiming],
) -> str | None:
    """Return the named blocker for aggregate serving chunks, never an ITL."""

    _require_safe_id("ITL request", request_id)
    _strict_int("ITL output tokens", output_tokens, minimum=1)
    covered = 0
    for chunk in chunks:
        chunk.validate()
        if chunk.request_id != request_id or chunk.first_token_index != covered:
            raise ValueError("serving chunks do not exactly cover the ITL request")
        if chunk.per_token_observed_at_us is None and chunk.token_count > 1:
            return ITL_COALESCED_CHUNK_UNPROVEN_REASON
        covered += chunk.token_count
    if covered != output_tokens:
        raise ValueError("serving chunks do not cover every generated token")
    # Exact-looking client chunks still need one allowlisted raw producer receipt.
    return ITL_RAW_RECEIPT_MISSING_REASON


__all__ = [
    "ITL_COALESCED_CHUNK_UNPROVEN_REASON",
    "ITL_CPU_CONTRACT_ONLY_REASON",
    "ITL_DYNAMIC_GPU_PROOF_UNAVAILABLE_REASON",
    "ITL_FIRST_PARTY_RESULT_POINTER_UNAVAILABLE_REASON",
    "ITL_RAW_RECEIPT_MISSING_REASON",
    "ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON",
    "ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256",
    "ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON",
    "RELEASE_ITL_TIMESTAMP_PRODUCERS",
    "SGLANG_CPU_ITL_CONTRACT_HOOK",
    "SGLANG_CPU_ITL_CONTRACT_RELEASE_STATUS",
    "SGLANG_CPU_ITL_CONTRACT_SEMANTICS",
    "SGLANG_NATIVE_ITL_CONTRACT_HOOK",
    "SGLANG_NATIVE_ITL_CONTRACT_RELEASE_STATUS",
    "SGLANG_NATIVE_ITL_CONTRACT_SEMANTICS",
    "SGLANG_NATIVE_ITL_RESULT_POINTER_KIND",
    "STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256",
    "BoundItlTimestampAuthority",
    "E2ItlTimestampPlan",
    "ItlRequestExpectation",
    "ItlRequestTimestamps",
    "ItlTimestampActivation",
    "ItlTimestampAuthorityBlocked",
    "PathBoundItlTimestampAuthority",
    "ReleaseItlTimestampProducer",
    "StageItlExecutionIdentity",
    "StageItlExternalControlBinding",
    "StageItlMethod",
    "StageItlTimestampAuthority",
    "StageItlTimestampProofArtifact",
    "StageItlTimestampProofRequest",
    "assess_serving_chunks_for_formal_itl",
    "bind_itl_timestamp_authority",
    "build_stage_itl_control_subject",
    "build_stage_itl_external_control_binding",
    "evaluate_e2_itl_timestamp_activation",
    "itl_request_expectations_sha256",
    "load_path_bound_itl_timestamp_authority",
    "publish_stage_itl_timestamp_proof_artifact",
    "publish_stage_itl_timestamp_proof_artifacts",
    "publish_stage_itl_timestamp_raw_receipt",
    "reject_cpu_contract_only_itl_metadata",
    "release_e2_itl_timestamp_plan",
    "replay_e2_itl_timestamp_plan",
    "require_e2_itl_timestamp_prelaunch",
    "revalidate_itl_timestamp_authority",
    "revalidate_path_bound_itl_timestamp_authority",
    "validate_stage_itl_timestamp_proof_artifact",
]
