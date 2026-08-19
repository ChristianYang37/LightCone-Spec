"""Dynamic external-control authority for the eight preflight interference rows.

The GPU host never owns an offline signing key.  It emits one canonical raw
native terminal (and first-party per-token result pointers) per registered
cell.  A pulled batch therefore has the explicit non-authorizing state
``WAITING_FOR_LOCAL_CONTROL``.  Local verification first creates the durable
native-result and native-ITL proofs, then signs one exact aggregate binding and
atomically reserves its fresh deployment/control challenges.  Reopening the
aggregate replays every upstream path and reducer input without consuming a
challenge again.

This module intentionally does not accept the historical
``InterferenceCalibrationExecutionAuthority``.  That object is coupled to the
legacy static attester policy and cannot authorize the signed staged protocol.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from lightcone_spec.experiments.formal_dispatch import (
    FormalPreflightExecutionBinding,
    VerifiedFormalPreflightDispatch,
    require_verified_formal_preflight_dispatch,
)
from lightcone_spec.experiments.gpu_pool import registry_pool_work_item
from lightcone_spec.experiments.interference_authority import (
    INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256,
    InterferenceCalibrationGroup,
    InterferenceCalibrationGroupDiagnostic,
    InterferenceCalibrationProtocol,
    InterferenceCalibrationRun,
    InterferenceRawObservation,
    diagnose_interference_calibration,
)
from lightcone_spec.experiments.itl_authority import (
    StageItlExecutionIdentity,
    StageItlTimestampAuthority,
    StageItlTimestampProofArtifact,
    validate_stage_itl_timestamp_proof_artifact,
)
from lightcone_spec.experiments.registry import (
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.experiments.serving import BoundServingRequest
from lightcone_spec.experiments.statistics import (
    TTFT_LIMIT_MS,
    WITHIN_REQUEST_P99_ITL_LIMIT_MS,
    SloRequest,
    account_slo,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalResultProjection,
    NativeTerminalResultProofArtifact,
    NativeTerminalRunBinding,
    validate_native_terminal_artifact,
    validate_unsigned_native_itl_pointer_bundle,
)
from lightcone_spec.runtime.attestation import NO_TRUSTED_ATTESTERS
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.release_trust_root import (
    load_source_release_ed25519_root,
)

if TYPE_CHECKING:
    from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool

FORMAL_PREFLIGHT_INTERFERENCE_RAW_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "formal_preflight_interference_unsigned_raw_batch",
        "coverage": "exact_eight_registered_static_cells",
        "remote_authority": "none",
        "successful_intermediate_status": "WAITING_FOR_LOCAL_CONTROL",
        "failure_intermediate_status": "ERROR_with_path_bound_fatal_terminal",
        "terminal_source": "first_party_patched_sglang_native_lifecycle",
        "timing_source": "first_party_native_result_pointers_integer_ns",
        "live_run_source": (
            "deep_reopened_compile_launch_manifest_plus_unsigned_pinned_sglang_"
            "serving_run_receipt_process_tree_log_cleanup_and_output_bindings"
        ),
        "gpu_process_source": (
            "path_content_bound_nvidia_smi_plus_before_ready_after_snapshots"
        ),
        "concurrent_source": (
            "one_shared_barrier_group_receipt_per_repetition_binding_distinct_"
            "gpu_port_process_groups_and_positive_measured_overlap"
        ),
        "private_key_on_gpu_host": False,
    }
)

FORMAL_PREFLIGHT_INTERFERENCE_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_preflight_interference_dynamic_proof",
        "upstream": (
            "eight_unsigned_native_terminals",
            "eight_dynamic_native_result_proofs",
            "eight_dynamic_native_itl_proofs",
        ),
        "control": "local_root_authorized_interference_control",
        "reservation": "fresh_deployment_and_control_challenges_atomic",
        "reducer": INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256,
        "acceptance": (
            "paired_slo_qualified_goodput_and_native_p99_itl_absolute_effect_le_1pct_"
            "and_95pct_intervals_include_zero"
        ),
        "goodput": (
            "request_qualification_lock_plus_native_first_token_itl_terminal_"
            "timestamps_and_only_individually_slo_qualified_output_tokens"
        ),
        "terminal_coverage": "exact_2_modes_x_2_repetitions_x_2_slots",
        "legacy_static_attester_authority": "forbidden",
    }
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SAFETY_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
)

RawInterferenceStatus = Literal["WAITING_FOR_LOCAL_CONTROL", "ERROR"]
QualifiedInterferenceStatus = Literal["PASSED", "FAILED"]

_SLO_POLICY_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_preflight_interference_request_slo_policy",
        "ttft_limits_ms": sorted(TTFT_LIMIT_MS.items()),
        "within_request_p99_itl_limit_ms": WITHIN_REQUEST_P99_ITL_LIMIT_MS,
        "qualification_rate_minimum": 0.99,
        "error_rate_maximum": 0.001,
        "completion_rate_minimum": 0.999,
        "goodput_numerator": "output_tokens_from_individually_slo_qualified_requests",
        "goodput_denominator": "max_terminal_ns_minus_min_request_started_ns",
    }
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _safe_id(label: str, value: object) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe identifier")
    return value


def _strict_object(
    label: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be one JSON object")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return value


def _run_binding_dict(binding: NativeTerminalRunBinding) -> dict[str, object]:
    binding.validate()
    return {
        "run_id": binding.run_id,
        "run_nonce_sha256": binding.run_nonce_sha256,
        "execution_plan_sha256": binding.execution_plan_sha256,
        "rank_config_sha256": binding.rank_config_sha256,
        "attempt_id": binding.attempt_id,
        "session_id": binding.session_id,
        "session_epoch": binding.session_epoch,
        "previous_run_id": binding.previous_run_id,
        "challenge_nonce_sha256": binding.challenge_nonce_sha256,
        "method": binding.method,
        "warmup_request_ids": list(binding.warmup_request_ids),
        "scored_request_ids": list(binding.scored_request_ids),
    }


def _run_binding_from_dict(value: object) -> NativeTerminalRunBinding:
    row = _strict_object(
        "preflight interference native run binding",
        value,
        frozenset(
            {
                "run_id",
                "run_nonce_sha256",
                "execution_plan_sha256",
                "rank_config_sha256",
                "attempt_id",
                "session_id",
                "session_epoch",
                "previous_run_id",
                "challenge_nonce_sha256",
                "method",
                "warmup_request_ids",
                "scored_request_ids",
            }
        ),
    )
    warmup = row.pop("warmup_request_ids")
    scored = row.pop("scored_request_ids")
    if type(warmup) is not list or type(scored) is not list:
        raise TypeError("preflight interference request IDs must be arrays")
    binding = NativeTerminalRunBinding(
        **row,
        warmup_request_ids=tuple(warmup),
        scored_request_ids=tuple(scored),
    )
    binding.validate()
    return binding


def _mode_repetition_slot(
    binding: FormalPreflightExecutionBinding,
) -> tuple[Literal["isolated", "concurrent"], int, int]:
    cell = binding.cell
    variant = str(cell.identity.variant)
    match = re.fullmatch(r"(isolated|concurrent)_slot_([01])", variant)
    if (
        match is None
        or cell.identity.experiment != "preflight"
        or cell.identity.task != "simultaneous_single_gpu_interference"
        or cell.identity.method != "static"
        or cell.identity.backend != "DFLASH"
        or cell.resources.workload_class is not WorkloadClass.CORRECTNESS
        or cell.identity.block not in {0, 1}
        or len(binding.gpu_uuids) != 1
    ):
        raise ValueError("formal preflight interference cell identity is not exact")
    mode = match.group(1)
    slot = int(match.group(2))
    if cell.identity.gpu_uuids != (f"logical-rank-slot-{slot}",):
        raise ValueError("formal preflight interference slot/GPU index differs")
    return mode, int(cell.identity.block), slot  # type: ignore[return-value]


def _interference_bindings(
    dispatch: object,
) -> tuple[FormalPreflightExecutionBinding, ...]:
    token = require_verified_formal_preflight_dispatch(dispatch)
    rows = tuple(
        sorted(
            (
                row
                for row in token.subject.execution_bindings
                if row.runner_kind == "first_party_interference"
            ),
            key=lambda row: row.registry_cell_id,
        )
    )
    if len(rows) != 8 or len({row.registry_cell_id for row in rows}) != 8:
        raise ValueError("formal preflight requires exact eight interference rows")
    for row in rows:
        _mode_repetition_slot(row)
    return rows


@dataclass(frozen=True)
class FormalPreflightInterferenceQualificationRow:
    """One request's preregistered prompt bucket and eligibility decision."""

    request_id: str
    prompt_bucket: Literal["short", "medium", "long"]
    eligible: bool

    def __post_init__(self) -> None:
        _safe_id("interference qualification request", self.request_id)
        if self.prompt_bucket not in TTFT_LIMIT_MS:
            raise ValueError("interference qualification prompt bucket is invalid")
        if type(self.eligible) is not bool:
            raise TypeError("interference qualification eligibility must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "prompt_bucket": self.prompt_bucket,
            "eligible": self.eligible,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "interference qualification row",
                value,
                frozenset({"request_id", "prompt_bucket", "eligible"}),
            )
        )


@dataclass(frozen=True)
class FormalPreflightInterferenceQualificationLock:
    """Path-bound request qualification materialized before GPU execution.

    This lock does not authorize execution by itself.  It binds the exact
    workload authorization already carried by the sealed dispatch, the exact
    scored request IDs, and the registered production-SLO policy.  The local
    aggregate control subsequently signs its path/raw/semantic identity.
    """

    schema_version: int
    kind: str
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    materialized_cell_id: str
    registry_cell_id: str
    workload_authorization_sha256: str
    scored_request_inputs_sha256: str
    slo_policy_sha256: str
    rows: tuple[FormalPreflightInterferenceQualificationRow, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_preflight_interference_request_qualification_lock"
        ):
            raise ValueError("interference qualification lock schema is unsupported")
        for label, value in (
            ("qualification registry", self.registry_sha256),
            ("qualification runtime", self.runtime_sha256),
            ("qualification split", self.split_sha256),
            ("qualification materialized cell", self.materialized_cell_id),
            ("qualification registry cell", self.registry_cell_id),
            (
                "qualification workload authorization",
                self.workload_authorization_sha256,
            ),
            ("qualification scored request inputs", self.scored_request_inputs_sha256),
        ):
            _sha256(label, value)
        if self.slo_policy_sha256 != _SLO_POLICY_SHA256:
            raise ValueError("interference qualification SLO policy differs")
        if not self.rows or any(
            type(row) is not FormalPreflightInterferenceQualificationRow
            for row in self.rows
        ):
            raise TypeError("interference qualification rows are incomplete")
        request_ids = tuple(row.request_id for row in self.rows)
        if request_ids != tuple(sorted(set(request_ids))):
            raise ValueError("interference qualification rows must be request-sorted")
        if not any(row.eligible for row in self.rows):
            raise ValueError("interference qualification requires an eligible request")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "materialized_cell_id": self.materialized_cell_id,
            "registry_cell_id": self.registry_cell_id,
            "workload_authorization_sha256": self.workload_authorization_sha256,
            "scored_request_inputs_sha256": self.scored_request_inputs_sha256,
            "slo_policy_sha256": self.slo_policy_sha256,
            "rows": [row.to_dict() for row in self.rows],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        raw = _strict_object(
            "interference qualification lock",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "registry_sha256",
                    "runtime_sha256",
                    "split_sha256",
                    "materialized_cell_id",
                    "registry_cell_id",
                    "workload_authorization_sha256",
                    "scored_request_inputs_sha256",
                    "slo_policy_sha256",
                    "rows",
                }
            ),
        )
        rows = raw.pop("rows")
        if type(rows) is not list:
            raise TypeError("interference qualification rows must be an array")
        return cls(
            **raw,
            rows=tuple(
                FormalPreflightInterferenceQualificationRow.from_dict(row)
                for row in rows
            ),
        )


def publish_formal_preflight_interference_qualification_lock(
    dispatch: object,
    *,
    binding: FormalPreflightExecutionBinding,
    scored_requests: Sequence[BoundServingRequest],
    rows: Sequence[FormalPreflightInterferenceQualificationRow],
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Materialize the exact request/SLO input lock before GPU execution.

    Publication is intentionally non-authorizing.  The rows must come from the
    workload materializer governed by the sealed workload authorization; the
    later local aggregate control binds this exact file.  No bare request or
    digest is accepted by the remote-result reducer.
    """

    token = require_verified_formal_preflight_dispatch(dispatch)
    expected = {row.registry_cell_id: row for row in _interference_bindings(token)}
    if type(binding) is not FormalPreflightExecutionBinding or (
        expected.get(binding.registry_cell_id) != binding
    ):
        raise ValueError("interference qualification row is not sealed")
    requests = tuple(scored_requests)
    for request in requests:
        if type(request) is not BoundServingRequest:
            raise TypeError("interference qualification requires bound requests")
        request.validate()
    request_ids = tuple(request.request_id for request in requests)
    if not request_ids or len(request_ids) != len(set(request_ids)):
        raise ValueError("interference qualification request coverage is invalid")
    qualification_rows = tuple(sorted(rows, key=lambda row: row.request_id))
    if any(
        type(row) is not FormalPreflightInterferenceQualificationRow
        for row in qualification_rows
    ) or {row.request_id for row in qualification_rows} != set(request_ids):
        raise ValueError("interference qualification rows differ from requests")
    activation = token.dispatch_context.activation_artifact
    source = dict(binding.source_authority_bindings)
    value = FormalPreflightInterferenceQualificationLock(
        schema_version=1,
        kind="formal_preflight_interference_request_qualification_lock",
        registry_sha256=token.dispatch_context.registry.sha256,
        runtime_sha256=activation.runtime_sha256,
        split_sha256=activation.split_sha256,
        materialized_cell_id=binding.materialized_cell_id,
        registry_cell_id=binding.registry_cell_id,
        workload_authorization_sha256=source["formal_workload_e3a"],
        scored_request_inputs_sha256=content_sha256(
            [request.sha256 for request in requests]
        ),
        slo_policy_sha256=_SLO_POLICY_SHA256,
        rows=qualification_rows,
    )
    publish_canonical_json_no_replace(output_path, value.to_dict())
    return CanonicalJsonProofBinding.bind(
        output_path,
        semantic_sha256=value.sha256,
    )


def _publish_single_operator_preflight_interference_qualification_lock(
    *,
    binding: FormalPreflightExecutionBinding,
    registry_sha256: str,
    runtime_sha256: str,
    split_sha256: str,
    scored_requests: Sequence[BoundServingRequest],
    rows: Sequence[FormalPreflightInterferenceQualificationRow],
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Trusted-current equivalent built only by the exact-ten operator adapter."""

    if type(binding) is not FormalPreflightExecutionBinding or (
        binding.runner_kind != "first_party_interference"
    ):
        raise TypeError("single-operator qualification requires one exact binding")
    for label, digest in (
        ("registry", registry_sha256),
        ("runtime", runtime_sha256),
        ("split", split_sha256),
    ):
        _sha256(f"single-operator qualification {label}", digest)
    requests = tuple(scored_requests)
    for request in requests:
        if type(request) is not BoundServingRequest:
            raise TypeError("single-operator qualification requires bound requests")
        request.validate()
    request_ids = tuple(request.request_id for request in requests)
    qualification_rows = tuple(sorted(rows, key=lambda row: row.request_id))
    if (
        not request_ids
        or len(request_ids) != len(set(request_ids))
        or any(
            type(row) is not FormalPreflightInterferenceQualificationRow
            for row in qualification_rows
        )
        or {row.request_id for row in qualification_rows} != set(request_ids)
    ):
        raise ValueError("single-operator qualification coverage differs")
    source = dict(binding.source_authority_bindings)
    value = FormalPreflightInterferenceQualificationLock(
        schema_version=1,
        kind="formal_preflight_interference_request_qualification_lock",
        registry_sha256=registry_sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        materialized_cell_id=binding.materialized_cell_id,
        registry_cell_id=binding.registry_cell_id,
        workload_authorization_sha256=source["formal_workload_e3a"],
        scored_request_inputs_sha256=content_sha256(
            [request.sha256 for request in requests]
        ),
        slo_policy_sha256=_SLO_POLICY_SHA256,
        rows=qualification_rows,
    )
    publish_canonical_json_no_replace(output_path, value.to_dict())
    return CanonicalJsonProofBinding.bind(output_path, semantic_sha256=value.sha256)


@dataclass(frozen=True)
class FormalPreflightInterferenceFatalTerminal:
    """Unsigned but path-bound failure terminal from the remote runner."""

    schema_version: int
    kind: str
    registry_cell_id: str
    assignment_sha256: str
    experiment_budget_sha256: str
    inventory_sha256: str
    run_binding_sha256: str
    error_code: str
    source_fatal_terminal: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.kind != (
            "formal_preflight_interference_fatal_terminal"
        ):
            raise ValueError("interference fatal terminal schema is unsupported")
        for label, value in (
            ("fatal cell", self.registry_cell_id),
            ("fatal assignment", self.assignment_sha256),
            ("fatal budget", self.experiment_budget_sha256),
            ("fatal inventory", self.inventory_sha256),
            ("fatal run binding", self.run_binding_sha256),
        ):
            _sha256(label, value)
        _safe_id("fatal error code", self.error_code)
        if type(self.source_fatal_terminal) is not CanonicalJsonProofBinding:
            raise TypeError("interference fatal terminal lost its source pointer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_cell_id": self.registry_cell_id,
            "assignment_sha256": self.assignment_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "inventory_sha256": self.inventory_sha256,
            "run_binding_sha256": self.run_binding_sha256,
            "error_code": self.error_code,
            "source_fatal_terminal": self.source_fatal_terminal.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "interference fatal terminal",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "registry_cell_id",
                    "assignment_sha256",
                    "experiment_budget_sha256",
                    "inventory_sha256",
                    "run_binding_sha256",
                    "error_code",
                    "source_fatal_terminal",
                }
            ),
        )
        source = CanonicalJsonProofBinding.from_dict(row.pop("source_fatal_terminal"))
        return cls(**row, source_fatal_terminal=source)


def publish_formal_preflight_interference_fatal_terminal(
    path: str | Path,
    *,
    binding: FormalPreflightExecutionBinding,
    inventory_sha256: str,
    run_binding: NativeTerminalRunBinding,
    error_code: str,
    source_fatal_terminal_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Always-publish error path for a failed unsigned remote row."""

    if type(binding) is not FormalPreflightExecutionBinding:
        raise TypeError("interference fatal terminal requires one sealed row binding")
    run_binding.validate()
    source_fatal = CanonicalJsonProofBinding.bind(source_fatal_terminal_path)
    terminal = FormalPreflightInterferenceFatalTerminal(
        schema_version=2,
        kind="formal_preflight_interference_fatal_terminal",
        registry_cell_id=binding.registry_cell_id,
        assignment_sha256=binding.assignment_sha256,
        experiment_budget_sha256=binding.experiment_budget_sha256,
        inventory_sha256=_sha256("fatal inventory", inventory_sha256),
        run_binding_sha256=content_sha256(run_binding.begin_payload()),
        error_code=_safe_id("fatal error code", error_code),
        source_fatal_terminal=source_fatal,
    )
    _revalidate_interference_fatal_source(terminal)
    publish_canonical_json_no_replace(path, terminal.to_dict())
    return CanonicalJsonProofBinding.bind(path, semantic_sha256=terminal.sha256)


def _revalidate_interference_fatal_source(
    terminal: FormalPreflightInterferenceFatalTerminal,
) -> None:
    from lightcone_spec.orchestration.live_sglang import (
        PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
    )

    source = terminal.source_fatal_terminal.reopen()
    if type(source) is not dict:
        raise TypeError("interference source fatal terminal must be an object")
    run_digest = source.get("run_binding_sha256")
    run_digests = source.get("run_binding_sha256s")
    if (
        source.get("schema_version") != 1
        or source.get("kind")
        not in {
            "unsigned_pinned_sglang_serving_fatal_pointer",
            "unsigned_pinned_sglang_concurrent_group_fatal_pointer",
        }
        or source.get("protocol_sha256") != PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256
        or source.get("status") != "ERROR"
        or source.get("formal_execution_authorized") is not False
        or source.get("reason_code") != terminal.error_code
        or (
            run_digest != terminal.run_binding_sha256
            and not (
                type(run_digests) is list and terminal.run_binding_sha256 in run_digests
            )
        )
        or (
            source.get("kind")
            == "unsigned_pinned_sglang_concurrent_group_fatal_pointer"
            and source.get("inventory_sha256") != terminal.inventory_sha256
        )
    ):
        raise ValueError("interference source fatal terminal lineage differs")


@dataclass(frozen=True)
class FormalPreflightInterferenceRawRow:
    """One unsigned remote row; it cannot claim formal completion."""

    materialized_cell_id: str
    registry_cell_id: str
    assignment_sha256: str
    experiment_budget_sha256: str
    inventory_sha256: str
    gpu_uuid: str
    mode: Literal["isolated", "concurrent"]
    repetition: int
    slot: int
    run_binding: NativeTerminalRunBinding
    status: RawInterferenceStatus
    launch_manifest: CanonicalJsonProofBinding | None
    live_run_receipt: CanonicalJsonProofBinding | None
    raw_terminal: CanonicalJsonProofBinding | None
    native_itl_pointer_artifact: CanonicalJsonProofBinding | None
    qualification_lock: CanonicalJsonProofBinding | None
    concurrent_group_receipt: CanonicalJsonProofBinding | None
    fatal_terminal: CanonicalJsonProofBinding | None

    def __post_init__(self) -> None:
        for label, value in (
            ("raw materialized cell", self.materialized_cell_id),
            ("raw registry cell", self.registry_cell_id),
            ("raw assignment", self.assignment_sha256),
            ("raw budget", self.experiment_budget_sha256),
            ("raw inventory", self.inventory_sha256),
        ):
            _sha256(label, value)
        _safe_id("raw GPU UUID", self.gpu_uuid)
        if self.mode not in {"isolated", "concurrent"}:
            raise ValueError("raw interference mode is unsupported")
        if self.repetition not in {0, 1} or self.slot not in {0, 1}:
            raise ValueError("raw interference repetition/slot is invalid")
        if type(self.run_binding) is not NativeTerminalRunBinding:
            raise TypeError("raw interference run binding is not exact")
        self.run_binding.validate()
        if self.run_binding.method != "static":
            raise ValueError("preflight interference must run Static")
        if self.status == "WAITING_FOR_LOCAL_CONTROL":
            if (
                type(self.launch_manifest) is not CanonicalJsonProofBinding
                or type(self.live_run_receipt) is not CanonicalJsonProofBinding
                or type(self.raw_terminal) is not CanonicalJsonProofBinding
                or type(self.native_itl_pointer_artifact)
                is not CanonicalJsonProofBinding
                or type(self.qualification_lock) is not CanonicalJsonProofBinding
                or (
                    self.mode == "concurrent"
                    and type(self.concurrent_group_receipt)
                    is not CanonicalJsonProofBinding
                )
                or (
                    self.mode == "isolated"
                    and self.concurrent_group_receipt is not None
                )
                or self.fatal_terminal is not None
            ):
                raise ValueError("waiting interference row lacks exact raw evidence")
        elif self.status == "ERROR":
            if (
                self.launch_manifest is not None
                or self.live_run_receipt is not None
                or self.raw_terminal is not None
                or self.native_itl_pointer_artifact is not None
                or self.qualification_lock is not None
                or self.concurrent_group_receipt is not None
                or type(self.fatal_terminal) is not CanonicalJsonProofBinding
            ):
                raise ValueError("errored interference row lacks one fatal terminal")
        else:
            raise ValueError("raw interference status is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "materialized_cell_id": self.materialized_cell_id,
            "registry_cell_id": self.registry_cell_id,
            "assignment_sha256": self.assignment_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "inventory_sha256": self.inventory_sha256,
            "gpu_uuid": self.gpu_uuid,
            "mode": self.mode,
            "repetition": self.repetition,
            "slot": self.slot,
            "run_binding": _run_binding_dict(self.run_binding),
            "status": self.status,
            "launch_manifest": (
                None if self.launch_manifest is None else self.launch_manifest.to_dict()
            ),
            "live_run_receipt": (
                None
                if self.live_run_receipt is None
                else self.live_run_receipt.to_dict()
            ),
            "raw_terminal": (
                None if self.raw_terminal is None else self.raw_terminal.to_dict()
            ),
            "native_itl_pointer_artifact": (
                None
                if self.native_itl_pointer_artifact is None
                else self.native_itl_pointer_artifact.to_dict()
            ),
            "qualification_lock": (
                None
                if self.qualification_lock is None
                else self.qualification_lock.to_dict()
            ),
            "concurrent_group_receipt": (
                None
                if self.concurrent_group_receipt is None
                else self.concurrent_group_receipt.to_dict()
            ),
            "fatal_terminal": (
                None if self.fatal_terminal is None else self.fatal_terminal.to_dict()
            ),
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "preflight interference raw row",
            value,
            frozenset(
                {
                    "materialized_cell_id",
                    "registry_cell_id",
                    "assignment_sha256",
                    "experiment_budget_sha256",
                    "inventory_sha256",
                    "gpu_uuid",
                    "mode",
                    "repetition",
                    "slot",
                    "run_binding",
                    "status",
                    "launch_manifest",
                    "live_run_receipt",
                    "raw_terminal",
                    "native_itl_pointer_artifact",
                    "qualification_lock",
                    "concurrent_group_receipt",
                    "fatal_terminal",
                }
            ),
        )
        launch_manifest = row.pop("launch_manifest")
        live_run_receipt = row.pop("live_run_receipt")
        raw_terminal = row.pop("raw_terminal")
        pointer_artifact = row.pop("native_itl_pointer_artifact")
        qualification_lock = row.pop("qualification_lock")
        concurrent_group_receipt = row.pop("concurrent_group_receipt")
        fatal_terminal = row.pop("fatal_terminal")
        run_binding = row.pop("run_binding")
        return cls(
            **row,
            run_binding=_run_binding_from_dict(run_binding),
            launch_manifest=(
                None
                if launch_manifest is None
                else CanonicalJsonProofBinding.from_dict(launch_manifest)
            ),
            live_run_receipt=(
                None
                if live_run_receipt is None
                else CanonicalJsonProofBinding.from_dict(live_run_receipt)
            ),
            raw_terminal=(
                None
                if raw_terminal is None
                else CanonicalJsonProofBinding.from_dict(raw_terminal)
            ),
            native_itl_pointer_artifact=(
                None
                if pointer_artifact is None
                else CanonicalJsonProofBinding.from_dict(pointer_artifact)
            ),
            qualification_lock=(
                None
                if qualification_lock is None
                else CanonicalJsonProofBinding.from_dict(qualification_lock)
            ),
            concurrent_group_receipt=(
                None
                if concurrent_group_receipt is None
                else CanonicalJsonProofBinding.from_dict(concurrent_group_receipt)
            ),
            fatal_terminal=(
                None
                if fatal_terminal is None
                else CanonicalJsonProofBinding.from_dict(fatal_terminal)
            ),
        )

    def deep_revalidate_unsigned(
        self,
        *,
        nvidia_smi_tool: PinnedNvidiaSmiTool | None = None,
    ) -> None:
        if self.status == "WAITING_FOR_LOCAL_CONTROL":
            assert self.launch_manifest is not None
            assert self.live_run_receipt is not None
            assert self.raw_terminal is not None
            assert self.native_itl_pointer_artifact is not None
            assert self.qualification_lock is not None
            if nvidia_smi_tool is None:
                raise ValueError("raw interference row lacks pinned nvidia-smi")
            evidence = validate_native_terminal_artifact(
                self.raw_terminal.reopen(),
                trusted_attester_policy=NO_TRUSTED_ATTESTERS,
                expected_binding=self.run_binding,
            )
            if evidence.authority_kind != "untrusted_raw_terminal":
                raise ValueError(
                    "raw interference terminal gained unexpected authority"
                )
            qualification = FormalPreflightInterferenceQualificationLock.from_dict(
                self.qualification_lock.reopen()
            )
            terminal_output_tokens = {
                request.request_id: request.output_token_ids
                for request in evidence.requests
                if request.submitted_to_server
                and request.terminal_status == "completed"
                and request.output_token_ids is not None
            }
            ordered_terminal_outputs = {
                request_id: terminal_output_tokens[request_id]
                for request_id in self.run_binding.scored_request_ids
                if request_id in terminal_output_tokens
            }
            pointer_bundle = validate_unsigned_native_itl_pointer_bundle(
                self.native_itl_pointer_artifact,
                expected_binding=self.run_binding,
                expected_terminal_artifact=self.raw_terminal,
                expected_scored_request_inputs_sha256=(
                    qualification.scored_request_inputs_sha256
                ),
                expected_terminal_output_tokens=ordered_terminal_outputs,
            )
            from lightcone_spec.orchestration.live_sglang import (
                UnsignedPinnedSglangServingRunReceipt,
                validate_unsigned_pinned_sglang_serving_run_receipt,
            )

            receipt = UnsignedPinnedSglangServingRunReceipt.from_dict(
                self.live_run_receipt.reopen()
            )
            if self.mode == "isolated" and (
                receipt.snapshot_gpu_uuids != (self.gpu_uuid,)
                or receipt.server_process_group_ids != (receipt.server_process_id,)
            ):
                raise ValueError(
                    "isolated interference run contains a foreign GPU process"
                )
            live_run = validate_unsigned_pinned_sglang_serving_run_receipt(
                self.live_run_receipt,
                expected_launch_manifest=self.launch_manifest,
                expected_binding=self.run_binding,
                expected_terminal_artifact=self.raw_terminal,
                expected_native_itl_pointer_artifact=(self.native_itl_pointer_artifact),
                expected_scored_request_inputs_sha256=(
                    qualification.scored_request_inputs_sha256
                ),
                expected_gpu_uuids=(self.gpu_uuid,),
                expected_inventory_sha256=self.inventory_sha256,
                expected_physical_assignment_sha256=self.assignment_sha256,
                expected_experiment_budget_sha256=self.experiment_budget_sha256,
                expected_tool=nvidia_smi_tool,
                expected_snapshot_gpu_uuids=receipt.snapshot_gpu_uuids,
                expected_server_process_group_ids=receipt.server_process_group_ids,
            )
            if (
                self.native_itl_pointer_artifact.semantic_sha256
                != pointer_bundle.artifact_semantic_sha256
                or live_run.receipt_binding != self.live_run_receipt
                or self.qualification_lock.semantic_sha256 != qualification.sha256
                or tuple(row.request_id for row in qualification.rows)
                != tuple(sorted(self.run_binding.scored_request_ids))
            ):
                raise ValueError(
                    "interference raw pointer/qualification lineage differs"
                )
            return
        assert self.fatal_terminal is not None
        terminal = FormalPreflightInterferenceFatalTerminal.from_dict(
            self.fatal_terminal.reopen()
        )
        _revalidate_interference_fatal_source(terminal)
        if (
            terminal.sha256 != self.fatal_terminal.semantic_sha256
            or terminal.registry_cell_id != self.registry_cell_id
            or terminal.assignment_sha256 != self.assignment_sha256
            or terminal.experiment_budget_sha256 != self.experiment_budget_sha256
            or terminal.inventory_sha256 != self.inventory_sha256
            or terminal.run_binding_sha256
            != content_sha256(self.run_binding.begin_payload())
        ):
            raise ValueError("interference fatal terminal lineage differs")


def build_formal_preflight_interference_raw_row(
    binding: FormalPreflightExecutionBinding,
    *,
    inventory_sha256: str,
    run_binding: NativeTerminalRunBinding,
    nvidia_smi_tool: PinnedNvidiaSmiTool | None = None,
    launch_manifest_path: str | Path | None = None,
    live_run_receipt_path: str | Path | None = None,
    raw_terminal_path: str | Path | None = None,
    native_itl_pointer_artifact_path: str | Path | None = None,
    qualification_lock_path: str | Path | None = None,
    concurrent_group_receipt_path: str | Path | None = None,
    fatal_terminal_path: str | Path | None = None,
) -> FormalPreflightInterferenceRawRow:
    """Bind one pulled collector result to its exact sealed scheduler row."""

    if type(binding) is not FormalPreflightExecutionBinding:
        raise TypeError("raw interference row requires an exact dispatch binding")
    mode, repetition, slot = _mode_repetition_slot(binding)
    run_binding.validate()
    common_success_paths = (
        launch_manifest_path,
        live_run_receipt_path,
        raw_terminal_path,
        native_itl_pointer_artifact_path,
        qualification_lock_path,
    )
    if (
        any(path is not None for path in common_success_paths)
        or concurrent_group_receipt_path is not None
    ) and fatal_terminal_path is not None:
        raise ValueError("raw interference row cannot be success and error")
    if fatal_terminal_path is None:
        if not all(path is not None for path in common_success_paths):
            raise ValueError("raw interference row requires a terminal path")
        if (mode == "concurrent") != (concurrent_group_receipt_path is not None):
            raise ValueError(
                "raw interference group receipt differs from execution mode"
            )
    if run_binding.method != "static":
        raise ValueError("raw interference row method differs from registry")
    launch_manifest = (
        None
        if launch_manifest_path is None
        else CanonicalJsonProofBinding.bind(launch_manifest_path)
    )
    live_run_receipt = (
        None
        if live_run_receipt_path is None
        else CanonicalJsonProofBinding.bind(live_run_receipt_path)
    )
    raw_terminal = (
        None
        if raw_terminal_path is None
        else CanonicalJsonProofBinding.bind(raw_terminal_path)
    )
    native_itl_pointer_artifact = (
        None
        if native_itl_pointer_artifact_path is None
        else CanonicalJsonProofBinding.bind(native_itl_pointer_artifact_path)
    )
    qualification_lock = (
        None
        if qualification_lock_path is None
        else CanonicalJsonProofBinding.bind(qualification_lock_path)
    )
    concurrent_group_receipt = (
        None
        if concurrent_group_receipt_path is None
        else CanonicalJsonProofBinding.bind(concurrent_group_receipt_path)
    )
    fatal_terminal = (
        None
        if fatal_terminal_path is None
        else CanonicalJsonProofBinding.bind(fatal_terminal_path)
    )
    row = FormalPreflightInterferenceRawRow(
        materialized_cell_id=binding.materialized_cell_id,
        registry_cell_id=binding.registry_cell_id,
        assignment_sha256=binding.assignment_sha256,
        experiment_budget_sha256=binding.experiment_budget_sha256,
        inventory_sha256=_sha256("raw inventory", inventory_sha256),
        gpu_uuid=binding.gpu_uuids[0],
        mode=mode,
        repetition=repetition,
        slot=slot,
        run_binding=run_binding,
        status=("WAITING_FOR_LOCAL_CONTROL" if raw_terminal is not None else "ERROR"),
        launch_manifest=launch_manifest,
        live_run_receipt=live_run_receipt,
        raw_terminal=raw_terminal,
        native_itl_pointer_artifact=native_itl_pointer_artifact,
        qualification_lock=qualification_lock,
        concurrent_group_receipt=concurrent_group_receipt,
        fatal_terminal=fatal_terminal,
    )
    row.deep_revalidate_unsigned(nvidia_smi_tool=nvidia_smi_tool)
    return row


@dataclass(frozen=True)
class FormalPreflightInterferenceRawBatch:
    """Exact eight-row intermediate terminal; never a completion authority."""

    schema_version: int
    kind: str
    protocol_sha256: str
    dispatch_sha256: str
    registry_sha256: str
    activation_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    nvidia_smi_tool: PinnedNvidiaSmiTool
    status: RawInterferenceStatus
    rows: tuple[FormalPreflightInterferenceRawRow, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 3 or self.kind != (
            "formal_preflight_interference_raw_batch"
        ):
            raise ValueError("preflight interference raw batch schema is unsupported")
        if self.protocol_sha256 != FORMAL_PREFLIGHT_INTERFERENCE_RAW_PROTOCOL_SHA256:
            raise ValueError("preflight interference raw batch protocol differs")
        for label, value in (
            ("raw dispatch", self.dispatch_sha256),
            ("raw registry", self.registry_sha256),
            ("raw activation", self.activation_sha256),
            ("raw runtime", self.runtime_sha256),
            ("raw split", self.split_sha256),
            ("raw inventory", self.inventory_sha256),
        ):
            _sha256(label, value)
        from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool

        if type(self.nvidia_smi_tool) is not PinnedNvidiaSmiTool:
            raise TypeError("preflight interference lacks pinned nvidia-smi")
        self.nvidia_smi_tool.revalidate()
        cell_ids = tuple(row.registry_cell_id for row in self.rows)
        if len(self.rows) != 8 or cell_ids != tuple(sorted(set(cell_ids))):
            raise ValueError("preflight interference raw batch coverage is not eight")
        schedules = {(row.mode, row.repetition, row.slot) for row in self.rows}
        expected = {
            (mode, repetition, slot)
            for mode in ("isolated", "concurrent")
            for repetition in range(2)
            for slot in range(2)
        }
        if schedules != expected:
            raise ValueError("preflight interference raw schedule is incomplete")
        run_identities = tuple(
            (
                row.run_binding.run_id,
                row.run_binding.run_nonce_sha256,
                row.run_binding.attempt_id,
                row.run_binding.challenge_nonce_sha256,
            )
            for row in self.rows
        )
        if (
            len(set(run_identities)) != 8
            or len({row.run_binding.run_nonce_sha256 for row in self.rows}) != 8
            or len({row.run_binding.challenge_nonce_sha256 for row in self.rows}) != 8
        ):
            raise ValueError("preflight interference run/replay identities repeat")
        successful_group_receipts: list[CanonicalJsonProofBinding] = []
        for repetition in range(2):
            pair = tuple(
                sorted(
                    (
                        row
                        for row in self.rows
                        if row.mode == "concurrent" and row.repetition == repetition
                    ),
                    key=lambda row: row.slot,
                )
            )
            if len(pair) != 2 or pair[0].status != pair[1].status:
                raise ValueError("preflight concurrent group terminal coverage differs")
            if pair[0].status == "WAITING_FOR_LOCAL_CONTROL" and (
                pair[0].concurrent_group_receipt != pair[1].concurrent_group_receipt
            ):
                raise ValueError(
                    "preflight concurrent rows do not share one group receipt"
                )
            if pair[0].status == "WAITING_FOR_LOCAL_CONTROL":
                assert pair[0].concurrent_group_receipt is not None
                successful_group_receipts.append(pair[0].concurrent_group_receipt)
        if len(set(successful_group_receipts)) != len(successful_group_receipts):
            raise ValueError("preflight concurrent repetitions reused a group receipt")
        expected_status = (
            "ERROR"
            if any(row.status == "ERROR" for row in self.rows)
            else "WAITING_FOR_LOCAL_CONTROL"
        )
        if self.status != expected_status:
            raise ValueError("preflight interference raw batch status differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "dispatch_sha256": self.dispatch_sha256,
            "registry_sha256": self.registry_sha256,
            "activation_sha256": self.activation_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "inventory_sha256": self.inventory_sha256,
            "nvidia_smi_tool": self.nvidia_smi_tool.to_dict(),
            "status": self.status,
            "rows": [row.to_dict() for row in self.rows],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "preflight interference raw batch",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "protocol_sha256",
                    "dispatch_sha256",
                    "registry_sha256",
                    "activation_sha256",
                    "runtime_sha256",
                    "split_sha256",
                    "inventory_sha256",
                    "nvidia_smi_tool",
                    "status",
                    "rows",
                }
            ),
        )
        values = row.pop("rows")
        tool = row.pop("nvidia_smi_tool")
        if type(values) is not list:
            raise TypeError("preflight interference raw rows must be an array")
        from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool

        return cls(
            **row,
            nvidia_smi_tool=PinnedNvidiaSmiTool.from_dict(tool),
            rows=tuple(
                FormalPreflightInterferenceRawRow.from_dict(value) for value in values
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_dict(CanonicalJsonProofBinding.bind(path).reopen())

    def revalidate(self) -> None:
        self.nvidia_smi_tool.revalidate()
        for row in self.rows:
            if row.inventory_sha256 != self.inventory_sha256:
                raise ValueError("preflight interference row inventory differs")
            row.deep_revalidate_unsigned(nvidia_smi_tool=self.nvidia_smi_tool)
        successful_group_receipts: list[CanonicalJsonProofBinding] = []
        from lightcone_spec.orchestration.live_sglang import (
            validate_unsigned_pinned_sglang_serving_group_receipt_by_identity,
        )
        from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

        for repetition in range(2):
            pair = tuple(
                sorted(
                    (
                        row
                        for row in self.rows
                        if row.mode == "concurrent" and row.repetition == repetition
                    ),
                    key=lambda row: row.slot,
                )
            )
            if pair[0].status == "ERROR":
                continue
            for row in pair:
                assert row.launch_manifest is not None
                assert row.live_run_receipt is not None
                assert row.raw_terminal is not None
                assert row.native_itl_pointer_artifact is not None
                assert row.qualification_lock is not None
                assert row.concurrent_group_receipt is not None
            group_receipt = pair[0].concurrent_group_receipt
            assert group_receipt is not None
            successful_group_receipts.append(group_receipt)
            qualifications = tuple(
                FormalPreflightInterferenceQualificationLock.from_dict(
                    row.qualification_lock.reopen()  # type: ignore[union-attr]
                )
                for row in pair
            )
            launches = tuple(
                CompileLaunchManifest.load(
                    row.launch_manifest.absolute_path  # type: ignore[union-attr]
                )
                for row in pair
            )
            validate_unsigned_pinned_sglang_serving_group_receipt_by_identity(
                group_receipt,
                expected_launch_manifests=(
                    pair[0].launch_manifest,
                    pair[1].launch_manifest,
                ),
                expected_run_bindings=(
                    pair[0].run_binding,
                    pair[1].run_binding,
                ),
                expected_terminal_artifacts=(
                    pair[0].raw_terminal,
                    pair[1].raw_terminal,
                ),
                expected_native_itl_pointer_artifacts=(
                    pair[0].native_itl_pointer_artifact,
                    pair[1].native_itl_pointer_artifact,
                ),
                expected_live_run_receipts=(
                    pair[0].live_run_receipt,
                    pair[1].live_run_receipt,
                ),
                expected_scored_request_inputs_sha256s=(
                    qualifications[0].scored_request_inputs_sha256,
                    qualifications[1].scored_request_inputs_sha256,
                ),
                expected_gpu_uuids=(pair[0].gpu_uuid, pair[1].gpu_uuid),
                expected_localhost_ports=(
                    launches[0].localhost_port,
                    launches[1].localhost_port,
                ),
                expected_physical_assignment_sha256s=(
                    pair[0].assignment_sha256,
                    pair[1].assignment_sha256,
                ),
                expected_experiment_budget_sha256s=(
                    pair[0].experiment_budget_sha256,
                    pair[1].experiment_budget_sha256,
                ),
                expected_tool=self.nvidia_smi_tool,
                expected_inventory_sha256=self.inventory_sha256,
            )
        if len(set(successful_group_receipts)) != len(successful_group_receipts):
            raise ValueError("preflight concurrent repetitions reused a group receipt")


def publish_formal_preflight_interference_raw_batch(
    dispatch: object,
    *,
    rows: tuple[FormalPreflightInterferenceRawRow, ...],
    nvidia_smi_tool: PinnedNvidiaSmiTool,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish the exact eight-row non-authorizing post-pull checkpoint."""

    token = require_verified_formal_preflight_dispatch(dispatch)
    expected = {row.registry_cell_id: row for row in _interference_bindings(token)}
    actual = {row.registry_cell_id: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("raw interference rows differ from sealed dispatch")
    for cell_id, row in actual.items():
        binding = expected[cell_id]
        mode, repetition, slot = _mode_repetition_slot(binding)
        if (
            row.materialized_cell_id != binding.materialized_cell_id
            or row.assignment_sha256 != binding.assignment_sha256
            or row.experiment_budget_sha256 != binding.experiment_budget_sha256
            or row.inventory_sha256 != token.subject.inventory_sha256
            or row.gpu_uuid != binding.gpu_uuids[0]
            or (row.mode, row.repetition, row.slot) != (mode, repetition, slot)
        ):
            raise ValueError("raw interference row differs from sealed assignment")
        row.deep_revalidate_unsigned(nvidia_smi_tool=nvidia_smi_tool)
        if row.status == "WAITING_FOR_LOCAL_CONTROL":
            assert row.qualification_lock is not None
            qualification = FormalPreflightInterferenceQualificationLock.from_dict(
                row.qualification_lock.reopen()
            )
            source = dict(binding.source_authority_bindings)
            if (
                qualification.registry_sha256 != token.dispatch_context.registry.sha256
                or qualification.runtime_sha256
                != token.dispatch_context.activation_artifact.runtime_sha256
                or qualification.split_sha256
                != token.dispatch_context.activation_artifact.split_sha256
                or qualification.materialized_cell_id != binding.materialized_cell_id
                or qualification.registry_cell_id != binding.registry_cell_id
                or qualification.workload_authorization_sha256
                != source["formal_workload_e3a"]
            ):
                raise ValueError(
                    "interference qualification lock differs from sealed dispatch"
                )
    activation = token.dispatch_context.activation_artifact
    batch = FormalPreflightInterferenceRawBatch(
        schema_version=3,
        kind="formal_preflight_interference_raw_batch",
        protocol_sha256=FORMAL_PREFLIGHT_INTERFERENCE_RAW_PROTOCOL_SHA256,
        dispatch_sha256=token.sha256,
        registry_sha256=token.dispatch_context.registry.sha256,
        activation_sha256=activation.sha256,
        runtime_sha256=activation.runtime_sha256,
        split_sha256=activation.split_sha256,
        inventory_sha256=token.subject.inventory_sha256,
        nvidia_smi_tool=nvidia_smi_tool,
        status=(
            "ERROR"
            if any(row.status == "ERROR" for row in rows)
            else "WAITING_FOR_LOCAL_CONTROL"
        ),
        rows=tuple(sorted(rows, key=lambda row: row.registry_cell_id)),
    )
    batch.revalidate()
    publish_canonical_json_no_replace(output_path, batch.to_dict())
    return CanonicalJsonProofBinding.bind(output_path, semantic_sha256=batch.sha256)


def _publish_single_operator_preflight_interference_raw_batch(
    *,
    execution_authority_sha256: str,
    registry_sha256: str,
    activation_sha256: str,
    runtime_sha256: str,
    split_sha256: str,
    inventory_sha256: str,
    expected_bindings: tuple[FormalPreflightExecutionBinding, ...],
    rows: tuple[FormalPreflightInterferenceRawRow, ...],
    nvidia_smi_tool: PinnedNvidiaSmiTool,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish the trusted-current exact-eight non-authorizing raw batch."""

    for label, digest in (
        ("execution authority", execution_authority_sha256),
        ("registry", registry_sha256),
        ("activation", activation_sha256),
        ("runtime", runtime_sha256),
        ("split", split_sha256),
        ("inventory", inventory_sha256),
    ):
        _sha256(f"single-operator raw {label}", digest)
    expected = {binding.registry_cell_id: binding for binding in expected_bindings}
    actual = {row.registry_cell_id: row for row in rows}
    if (
        len(expected) != 8
        or any(
            binding.runner_kind != "first_party_interference"
            for binding in expected.values()
        )
        or len(actual) != len(rows)
        or set(actual) != set(expected)
    ):
        raise ValueError("single-operator raw interference coverage differs")
    for cell_id, row in actual.items():
        binding = expected[cell_id]
        mode, repetition, slot = _mode_repetition_slot(binding)
        if (
            row.materialized_cell_id != binding.materialized_cell_id
            or row.assignment_sha256 != binding.assignment_sha256
            or row.experiment_budget_sha256 != binding.experiment_budget_sha256
            or row.inventory_sha256 != inventory_sha256
            or row.gpu_uuid != binding.gpu_uuids[0]
            or (row.mode, row.repetition, row.slot) != (mode, repetition, slot)
        ):
            raise ValueError("single-operator raw row differs from exact binding")
        row.deep_revalidate_unsigned(nvidia_smi_tool=nvidia_smi_tool)
        if row.status == "WAITING_FOR_LOCAL_CONTROL":
            assert row.qualification_lock is not None
            qualification = FormalPreflightInterferenceQualificationLock.from_dict(
                row.qualification_lock.reopen()
            )
            source = dict(binding.source_authority_bindings)
            if (
                qualification.registry_sha256 != registry_sha256
                or qualification.runtime_sha256 != runtime_sha256
                or qualification.split_sha256 != split_sha256
                or qualification.materialized_cell_id != binding.materialized_cell_id
                or qualification.registry_cell_id != binding.registry_cell_id
                or qualification.workload_authorization_sha256
                != source["formal_workload_e3a"]
            ):
                raise ValueError("single-operator qualification lock differs")
    batch = FormalPreflightInterferenceRawBatch(
        schema_version=3,
        kind="formal_preflight_interference_raw_batch",
        protocol_sha256=FORMAL_PREFLIGHT_INTERFERENCE_RAW_PROTOCOL_SHA256,
        dispatch_sha256=execution_authority_sha256,
        registry_sha256=registry_sha256,
        activation_sha256=activation_sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        inventory_sha256=inventory_sha256,
        nvidia_smi_tool=nvidia_smi_tool,
        status=(
            "ERROR"
            if any(row.status == "ERROR" for row in rows)
            else "WAITING_FOR_LOCAL_CONTROL"
        ),
        rows=tuple(sorted(rows, key=lambda row: row.registry_cell_id)),
    )
    batch.revalidate()
    publish_canonical_json_no_replace(output_path, batch.to_dict())
    return CanonicalJsonProofBinding.bind(output_path, semantic_sha256=batch.sha256)


def _observation_dict(value: InterferenceRawObservation) -> dict[str, object]:
    return {
        "observation_id": value.observation_id,
        "terminal_authority_sha256": value.terminal_authority_sha256,
        "mode": value.mode,
        "repetition": value.repetition,
        "slot": value.slot,
        "started_ns": value.started_ns,
        "finished_ns": value.finished_ns,
        "request_ids": list(value.request_ids),
        "token_trajectory_sha256": value.token_trajectory_sha256,
        "completed_requests": value.completed_requests,
        "output_tokens": value.output_tokens,
        "goodput_tps": value.goodput_tps,
        "p99_itl_ms": value.p99_itl_ms,
        "safety_counters": [list(row) for row in value.safety_counters],
        "hardware_valid": value.hardware_valid,
    }


def _observation_from_dict(value: object) -> InterferenceRawObservation:
    row = _strict_object(
        "preflight interference observation",
        value,
        frozenset(
            {
                "observation_id",
                "terminal_authority_sha256",
                "mode",
                "repetition",
                "slot",
                "started_ns",
                "finished_ns",
                "request_ids",
                "token_trajectory_sha256",
                "completed_requests",
                "output_tokens",
                "goodput_tps",
                "p99_itl_ms",
                "safety_counters",
                "hardware_valid",
            }
        ),
    )
    request_ids = row.pop("request_ids")
    counters = row.pop("safety_counters")
    if type(request_ids) is not list or type(counters) is not list:
        raise TypeError("preflight interference observation arrays are malformed")
    return InterferenceRawObservation(
        **row,
        request_ids=tuple(request_ids),
        safety_counters=tuple(tuple(value) for value in counters),
    )


def _percentile_99_ns(values: tuple[int, ...]) -> float:
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("preflight interference ITL samples are incomplete")
    ordered = tuple(sorted(values))
    return ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)] / 1_000_000.0


def _derive_observation(
    *,
    row: FormalPreflightInterferenceRawRow,
    terminal_authority_sha256: str,
    result: NativeTerminalResultProjection,
    itl: StageItlTimestampAuthority,
) -> tuple[
    InterferenceRawObservation,
    str,
    Literal["PASS", "FAIL"],
    tuple[str, ...],
]:
    result_requests = tuple(sorted(result.requests, key=lambda item: item.request_id))
    timing_requests = tuple(sorted(itl.requests, key=lambda item: item.request_id))
    if (
        not result_requests
        or tuple(item.request_id for item in result_requests)
        != tuple(item.request_id for item in timing_requests)
        or any(
            result_row.output_token_ids != timing_row.output_token_ids
            or result_row.terminal_status != "completed"
            or not result_row.submitted_to_server
            for result_row, timing_row in zip(
                result_requests, timing_requests, strict=True
            )
        )
    ):
        raise ValueError("preflight interference result/timing coverage differs")
    started_ns = min(item.request_started_ns for item in timing_requests)
    finished_ns = max(item.request_terminal_ns for item in timing_requests)
    elapsed_ns = finished_ns - started_ns
    output_tokens = sum(len(item.output_token_ids) for item in timing_requests)
    if elapsed_ns <= 0 or output_tokens < 1:
        raise ValueError("preflight interference scored timing window is empty")
    if row.qualification_lock is None:
        raise ValueError("preflight interference lacks request qualification")
    qualification = FormalPreflightInterferenceQualificationLock.from_dict(
        row.qualification_lock.reopen()
    )
    qualification_by_id = {item.request_id: item for item in qualification.rows}
    if set(qualification_by_id) != {item.request_id for item in timing_requests}:
        raise ValueError("preflight interference SLO request coverage differs")
    result_by_id = {item.request_id: item for item in result_requests}
    slo_rows: list[SloRequest] = []
    qualified_request_ids: list[str] = []
    for timing in timing_requests:
        result_row = result_by_id[timing.request_id]
        qualification_row = qualification_by_id[timing.request_id]
        ttft_ms = (
            timing.token_observed_ns[0] - timing.request_started_ns
        ) / 1_000_000.0
        request_itl_ms = _percentile_99_ns(timing.inter_token_ns)
        completed = result_row.terminal_status == "completed"
        error = not completed
        slo_row = SloRequest(
            request_id=timing.request_id,
            prompt_bucket=qualification_row.prompt_bucket,
            eligible=qualification_row.eligible,
            completed=completed,
            error=error,
            ttft_ms=ttft_ms,
            within_request_p99_itl_ms=request_itl_ms,
        )
        slo_rows.append(slo_row)
        if (
            slo_row.eligible
            and slo_row.completed
            and not slo_row.error
            and ttft_ms <= TTFT_LIMIT_MS[slo_row.prompt_bucket]
            and request_itl_ms <= WITHIN_REQUEST_P99_ITL_LIMIT_MS
        ):
            qualified_request_ids.append(timing.request_id)
    slo = account_slo(tuple(slo_rows))
    qualified_ids = tuple(sorted(qualified_request_ids))
    slo_output_tokens = sum(
        len(item.output_token_ids)
        for item in timing_requests
        if item.request_id in set(qualified_ids)
    )
    if slo_output_tokens < 1:
        raise ValueError("preflight interference has no positive SLO-qualified goodput")
    performance = result.performance_counters
    counters: list[tuple[str, int]] = []
    for name in _SAFETY_COUNTERS:
        value = performance.get(name)
        if type(value) is not int or value < 0:
            raise ValueError("preflight interference safety coverage is incomplete")
        counters.append((name, value))
    observation = InterferenceRawObservation(
        observation_id=f"preflight-{row.mode}-{row.repetition}-{row.slot}",
        terminal_authority_sha256=terminal_authority_sha256,
        mode=row.mode,
        repetition=row.repetition,
        slot=row.slot,
        started_ns=started_ns,
        finished_ns=finished_ns,
        request_ids=tuple(item.request_id for item in timing_requests),
        token_trajectory_sha256=content_sha256(
            [[item.request_id, list(item.output_token_ids)] for item in timing_requests]
        ),
        completed_requests=len(timing_requests),
        output_tokens=output_tokens,
        goodput_tps=slo_output_tokens / (elapsed_ns / 1_000_000_000.0),
        p99_itl_ms=_percentile_99_ns(itl.p99_itl_input_ns),
        safety_counters=tuple(counters),
        hardware_valid=True,
    )
    return observation, content_sha256(slo), slo.status, qualified_ids


@dataclass(frozen=True)
class FormalPreflightInterferenceProofRow:
    """One locally controlled result/timing proof joined to a sealed row."""

    materialized_cell_id: str
    registry_cell_id: str
    assignment_sha256: str
    experiment_budget_sha256: str
    gpu_uuid: str
    mode: Literal["isolated", "concurrent"]
    repetition: int
    slot: int
    run_binding: NativeTerminalRunBinding
    load_plan_sha256: str
    topology_sha256: str
    hardware_envelope_sha256: str
    native_result_proof: CanonicalJsonProofBinding
    native_itl_proof: CanonicalJsonProofBinding
    slo_accounting_sha256: str
    slo_status: Literal["PASS", "FAIL"]
    qualified_request_ids: tuple[str, ...]
    observation: InterferenceRawObservation

    def __post_init__(self) -> None:
        for label, value in (
            ("proof materialized cell", self.materialized_cell_id),
            ("proof registry cell", self.registry_cell_id),
            ("proof assignment", self.assignment_sha256),
            ("proof budget", self.experiment_budget_sha256),
            ("proof load plan", self.load_plan_sha256),
            ("proof topology", self.topology_sha256),
            ("proof hardware", self.hardware_envelope_sha256),
            ("proof SLO accounting", self.slo_accounting_sha256),
        ):
            _sha256(label, value)
        _safe_id("proof GPU UUID", self.gpu_uuid)
        if self.mode not in {"isolated", "concurrent"}:
            raise ValueError("proof interference mode is unsupported")
        if self.repetition not in {0, 1} or self.slot not in {0, 1}:
            raise ValueError("proof interference repetition/slot is invalid")
        if type(self.run_binding) is not NativeTerminalRunBinding:
            raise TypeError("proof interference run binding is not exact")
        self.run_binding.validate()
        if (
            type(self.native_result_proof) is not CanonicalJsonProofBinding
            or type(self.native_itl_proof) is not CanonicalJsonProofBinding
        ):
            raise TypeError("proof interference upstream binding is not exact")
        if type(self.observation) is not InterferenceRawObservation:
            raise TypeError("proof interference observation is not exact")
        if self.slo_status not in {"PASS", "FAIL"}:
            raise ValueError("proof interference SLO status is unsupported")
        if self.qualified_request_ids != tuple(sorted(set(self.qualified_request_ids))):
            raise ValueError("proof interference qualified requests are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "materialized_cell_id": self.materialized_cell_id,
            "registry_cell_id": self.registry_cell_id,
            "assignment_sha256": self.assignment_sha256,
            "experiment_budget_sha256": self.experiment_budget_sha256,
            "gpu_uuid": self.gpu_uuid,
            "mode": self.mode,
            "repetition": self.repetition,
            "slot": self.slot,
            "run_binding": _run_binding_dict(self.run_binding),
            "load_plan_sha256": self.load_plan_sha256,
            "topology_sha256": self.topology_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "native_result_proof": self.native_result_proof.to_dict(),
            "native_itl_proof": self.native_itl_proof.to_dict(),
            "slo_accounting_sha256": self.slo_accounting_sha256,
            "slo_status": self.slo_status,
            "qualified_request_ids": list(self.qualified_request_ids),
            "observation": _observation_dict(self.observation),
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "preflight interference proof row",
            value,
            frozenset(
                {
                    "materialized_cell_id",
                    "registry_cell_id",
                    "assignment_sha256",
                    "experiment_budget_sha256",
                    "gpu_uuid",
                    "mode",
                    "repetition",
                    "slot",
                    "run_binding",
                    "load_plan_sha256",
                    "topology_sha256",
                    "hardware_envelope_sha256",
                    "native_result_proof",
                    "native_itl_proof",
                    "slo_accounting_sha256",
                    "slo_status",
                    "qualified_request_ids",
                    "observation",
                }
            ),
        )
        run_binding = _run_binding_from_dict(row.pop("run_binding"))
        native_result_proof = CanonicalJsonProofBinding.from_dict(
            row.pop("native_result_proof")
        )
        native_itl_proof = CanonicalJsonProofBinding.from_dict(
            row.pop("native_itl_proof")
        )
        qualified_request_ids = row.pop("qualified_request_ids")
        if type(qualified_request_ids) is not list:
            raise TypeError("proof interference qualified requests must be an array")
        observation = _observation_from_dict(row.pop("observation"))
        return cls(
            **row,
            run_binding=run_binding,
            native_result_proof=native_result_proof,
            native_itl_proof=native_itl_proof,
            qualified_request_ids=tuple(qualified_request_ids),
            observation=observation,
        )


@dataclass(frozen=True)
class FormalPreflightInterferenceAggregateBinding:
    """Exact aggregate subject signed by the local interference signer."""

    schema_version: int
    kind: str
    raw_batch_sha256: str
    dispatch_sha256: str
    registry_sha256: str
    activation_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    expected_root_manifest_sha256: str
    status: QualifiedInterferenceStatus
    diagnostic_sha256: str
    row_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_preflight_interference_aggregate_binding"
        ):
            raise ValueError("interference aggregate binding schema is unsupported")
        for label, value in (
            ("aggregate raw batch", self.raw_batch_sha256),
            ("aggregate dispatch", self.dispatch_sha256),
            ("aggregate registry", self.registry_sha256),
            ("aggregate activation", self.activation_sha256),
            ("aggregate runtime", self.runtime_sha256),
            ("aggregate split", self.split_sha256),
            ("aggregate inventory", self.inventory_sha256),
            ("aggregate hardware", self.hardware_envelope_sha256),
            ("aggregate root", self.expected_root_manifest_sha256),
            ("aggregate diagnostic", self.diagnostic_sha256),
        ):
            _sha256(label, value)
        if self.status not in {"PASSED", "FAILED"}:
            raise ValueError("interference aggregate status is unsupported")
        if len(self.row_sha256s) != 8 or self.row_sha256s != tuple(
            sorted(set(self.row_sha256s))
        ):
            raise ValueError("interference aggregate row identities are incomplete")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_batch_sha256": self.raw_batch_sha256,
            "dispatch_sha256": self.dispatch_sha256,
            "registry_sha256": self.registry_sha256,
            "activation_sha256": self.activation_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
            "status": self.status,
            "diagnostic_sha256": self.diagnostic_sha256,
            "row_sha256s": list(self.row_sha256s),
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @cached_property
    def lineage_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_preflight_interference_aggregate_lineage",
                "binding_sha256": self.sha256,
                "dispatch_sha256": self.dispatch_sha256,
                "activation_sha256": self.activation_sha256,
                "inventory_sha256": self.inventory_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "row_sha256s": list(self.row_sha256s),
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "preflight interference aggregate binding",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "raw_batch_sha256",
                    "dispatch_sha256",
                    "registry_sha256",
                    "activation_sha256",
                    "runtime_sha256",
                    "split_sha256",
                    "inventory_sha256",
                    "hardware_envelope_sha256",
                    "expected_root_manifest_sha256",
                    "status",
                    "diagnostic_sha256",
                    "row_sha256s",
                }
            ),
        )
        values = row.pop("row_sha256s")
        if type(values) is not list:
            raise TypeError("interference aggregate row identities must be an array")
        return cls(**row, row_sha256s=tuple(values))


@dataclass(frozen=True)
class FormalPreflightInterferenceProofArtifact:
    """Durable public aggregate; reopening never mutates replay state."""

    schema_version: int
    kind: str
    raw_batch: CanonicalJsonProofBinding
    binding: FormalPreflightInterferenceAggregateBinding
    rows: tuple[FormalPreflightInterferenceProofRow, ...]
    diagnostic: dict[str, object]
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_preflight_interference_proof_artifact"
        ):
            raise ValueError("interference proof artifact schema is unsupported")
        if type(self.raw_batch) is not CanonicalJsonProofBinding:
            raise TypeError("interference proof raw batch binding is not exact")
        if type(self.binding) is not FormalPreflightInterferenceAggregateBinding:
            raise TypeError("interference proof aggregate binding is not exact")
        if (
            len(self.rows) != 8
            or tuple(row.sha256 for row in self.rows) != self.binding.row_sha256s
        ):
            raise ValueError("interference proof row identities differ")
        if (
            type(self.diagnostic) is not dict
            or content_sha256(self.diagnostic) != self.binding.diagnostic_sha256
        ):
            raise ValueError("interference proof diagnostic identity differs")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("interference proof control envelope is not exact")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("interference proof reservation is not exact")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_batch": self.raw_batch.to_dict(),
            "binding": self.binding.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "diagnostic": self.diagnostic,
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "preflight interference proof artifact",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "raw_batch",
                    "binding",
                    "rows",
                    "diagnostic",
                    "control_attestation",
                    "replay_reservation",
                }
            ),
        )
        rows = row.pop("rows")
        diagnostic = row.pop("diagnostic")
        if type(rows) is not list or type(diagnostic) is not dict:
            raise TypeError("interference proof rows/diagnostic are malformed")
        raw_batch = CanonicalJsonProofBinding.from_dict(row.pop("raw_batch"))
        binding = FormalPreflightInterferenceAggregateBinding.from_dict(
            row.pop("binding")
        )
        control_attestation = ControlArtifactAttestation.from_dict(
            row.pop("control_attestation")
        )
        replay_reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("replay_reservation")
        )
        return cls(
            **row,
            raw_batch=raw_batch,
            binding=binding,
            rows=tuple(
                FormalPreflightInterferenceProofRow.from_dict(value) for value in rows
            ),
            diagnostic=dict(diagnostic),
            control_attestation=control_attestation,
            replay_reservation=replay_reservation,
        )


_VERIFIED_INTERFERENCE_SENTINEL = object()


@dataclass(frozen=True, init=False)
class VerifiedFormalPreflightInterferenceProof:
    """Verifier-owned durable result consumed by the coverage reducer."""

    artifact_sha256: str
    binding: FormalPreflightInterferenceAggregateBinding
    rows: tuple[FormalPreflightInterferenceProofRow, ...]
    diagnostic: InterferenceCalibrationGroupDiagnostic
    _verification_tag: object

    def __init__(
        self,
        *,
        artifact_sha256: str,
        binding: FormalPreflightInterferenceAggregateBinding,
        rows: tuple[FormalPreflightInterferenceProofRow, ...],
        diagnostic: InterferenceCalibrationGroupDiagnostic,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_INTERFERENCE_SENTINEL:
            raise TypeError("verified interference proof is verifier-constructed only")
        _sha256("verified interference artifact", artifact_sha256)
        for name, value in (
            ("artifact_sha256", artifact_sha256),
            ("binding", binding),
            ("rows", rows),
            ("diagnostic", diagnostic),
            ("_verification_tag", _verification_tag),
        ):
            object.__setattr__(self, name, value)

    @property
    def status(self) -> QualifiedInterferenceStatus:
        return self.binding.status


def _request_contract_sha256(result: NativeTerminalResultProjection) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_preflight_interference_observed_request_contract",
            "requests": [
                {
                    "request_id": row.request_id,
                    "input_token_ids": list(row.input_token_ids),
                }
                for row in sorted(result.requests, key=lambda row: row.request_id)
            ],
        }
    )


def _topology_sha256(
    binding: FormalPreflightExecutionBinding,
    token: VerifiedFormalPreflightDispatch,
) -> tuple[str, str]:
    inventory = token.dispatch_context.inventory
    device = inventory.device(binding.gpu_uuids[0])
    devices = tuple(sorted(inventory.devices, key=lambda row: row.uuid))
    hardware_envelopes = {row.hardware_envelope_sha256 for row in devices}
    if (
        len(devices) != 2
        or len(hardware_envelopes) != 1
        or any(row.host_id != device.host_id for row in devices)
    ):
        raise ValueError(
            "preflight interference requires one homogeneous dual-card inventory"
        )
    return (
        content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_preflight_interference_dual_card_topology",
                "inventory_sha256": inventory.sha256,
                "host_id": device.host_id,
                "gpu_uuids": [row.uuid for row in devices],
                "pci_bus_ids": [row.pci_bus_id for row in devices],
                "pci_roots": [row.pci_root for row in devices],
                "numa_nodes": [row.numa_node for row in devices],
                "peer_access_classes": [row.peer_access_class for row in devices],
                "topology_groups": [
                    group.to_dict() for group in inventory.topology_groups
                ],
            }
        ),
        device.hardware_envelope_sha256,
    )


def _revalidate_proof_row(
    row: FormalPreflightInterferenceProofRow,
    *,
    raw_row: FormalPreflightInterferenceRawRow,
    expected_registry_sha256: str,
    expected_inventory_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> tuple[NativeTerminalResultProjection, StageItlTimestampAuthority]:
    from lightcone_spec.orchestration.formal_terminal_result import (
        validate_formal_terminal_result_proof_artifact,
    )

    result = validate_formal_terminal_result_proof_artifact(
        row.native_result_proof.absolute_path,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        expected_execution_plan_sha256=row.run_binding.execution_plan_sha256,
        expected_rank_config_sha256=row.run_binding.rank_config_sha256,
        expected_run_id=row.run_binding.run_id,
        expected_run_nonce_sha256=row.run_binding.run_nonce_sha256,
        expected_attempt_id=row.run_binding.attempt_id,
        expected_method="static",
        expected_stage="preflight",
        expected_topology="tp1_dp1",
        now_ns=now_ns,
    )
    identity = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id=row.materialized_cell_id,
        inventory_sha256=expected_inventory_sha256,
        registry_sha256=expected_registry_sha256,
        execution_plan_sha256=row.run_binding.execution_plan_sha256,
        rank_config_sha256=row.run_binding.rank_config_sha256,
        run_id=row.run_binding.run_id,
        run_nonce_sha256=row.run_binding.run_nonce_sha256,
        attempt_id=row.run_binding.attempt_id,
        method="static",
    )
    itl = validate_stage_itl_timestamp_proof_artifact(
        row.native_itl_proof.absolute_path,
        expected_execution_identity=identity,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
    )
    result_artifact = _preflight_native_terminal_source(row.native_result_proof)
    itl_artifact = StageItlTimestampProofArtifact.from_dict(
        row.native_itl_proof.reopen()
    )
    if (
        raw_row.raw_terminal is None
        or result_artifact.raw_terminal != raw_row.raw_terminal
        or itl_artifact.native_result_proof != row.native_result_proof
        or result_artifact.control_attestation.hardware_envelope_sha256
        != row.hardware_envelope_sha256
        or itl_artifact.control_attestation.hardware_envelope_sha256
        != row.hardware_envelope_sha256
        or _request_contract_sha256(result) != row.load_plan_sha256
    ):
        raise ValueError("preflight interference upstream proof lineage differs")
    observation, slo_sha256, slo_status, qualified_request_ids = _derive_observation(
        row=raw_row,
        terminal_authority_sha256=row.native_result_proof.semantic_sha256,
        result=result,
        itl=itl,
    )
    if (
        _observation_dict(observation) != _observation_dict(row.observation)
        or slo_sha256 != row.slo_accounting_sha256
        or slo_status != row.slo_status
        or qualified_request_ids != row.qualified_request_ids
    ):
        raise ValueError("preflight interference derived observation changed")
    return result, itl


def _preflight_native_terminal_source(
    binding: CanonicalJsonProofBinding,
) -> NativeTerminalResultProofArtifact:
    """Deep-open the native member of the closed preflight wrapper union."""

    from lightcone_spec.orchestration.formal_terminal_result import (
        FormalCurrentPreflightTp1TerminalResultProofArtifact,
        FormalPreflightTp1TerminalResultProofArtifact,
    )

    value = binding.reopen()
    if type(value) is not dict:
        raise TypeError("preflight terminal result proof is not an object")
    kind = value.get("kind")
    if kind == "formal_preflight_tp1_terminal_result_proof_artifact":
        wrapper = FormalPreflightTp1TerminalResultProofArtifact.from_dict(value)
    elif kind == "formal_current_preflight_tp1_terminal_result_proof_artifact":
        wrapper = FormalCurrentPreflightTp1TerminalResultProofArtifact.from_dict(value)
    else:
        raise ValueError("preflight terminal result proof kind is unsupported")
    return NativeTerminalResultProofArtifact.from_dict(
        wrapper.native_result_proof.reopen()
    )


def _calibration_run(
    row: FormalPreflightInterferenceProofRow,
    *,
    registry: ExperimentRegistry,
) -> InterferenceCalibrationRun:
    cells = tuple(
        cell for cell in registry.cells if cell.cell_id == row.registry_cell_id
    )
    if len(cells) != 1:
        raise ValueError("preflight interference proof cell is not unique")
    cell = cells[0]
    claim = registry_pool_work_item(cell, estimated_duration_seconds=1.0).claim
    if (
        cell.identity.experiment != "preflight"
        or cell.identity.task != "simultaneous_single_gpu_interference"
        or cell.identity.method != "static"
        or cell.identity.block != row.repetition
        or cell.identity.gpu_uuids != (f"logical-rank-slot-{row.slot}",)
        or str(cell.identity.variant) != f"{row.mode}_slot_{row.slot}"
    ):
        raise ValueError("preflight interference proof schedule differs from registry")
    calibration_class = content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_preflight_interference_calibration_class",
            "model": cell.identity.model,
            "backend": cell.identity.backend,
            "task": cell.identity.task,
            "method": cell.identity.method,
            "context": cell.identity.context,
            "regime": cell.identity.regime,
            "arrival": cell.identity.arrival,
            "concurrency": cell.identity.concurrency,
            "gang_shape": claim.gang_shape.signature,
            "load_thermal_power_envelope": claim.load_thermal_power_envelope,
            "contention_class": claim.contention_class,
        }
    )
    return InterferenceCalibrationRun(
        observation_id=row.observation.observation_id,
        mode=row.mode,
        repetition=row.repetition,
        slot=row.slot,
        terminal_authority_sha256=row.native_result_proof.semantic_sha256,
        assignment_sha256=row.assignment_sha256,
        cell_id=row.registry_cell_id,
        execution_plan_sha256=row.run_binding.execution_plan_sha256,
        budget_sha256=row.experiment_budget_sha256,
        load_plan_sha256=row.load_plan_sha256,
        run_nonce_sha256=row.run_binding.run_nonce_sha256,
        gpu_uuids=(row.gpu_uuid,),
        rank_groups=((row.gpu_uuid,),),
        topology_sha256=row.topology_sha256,
        hardware_envelope_sha256=row.hardware_envelope_sha256,
        workload_class=claim.workload_class,
        co_run_signature=calibration_class,
        gang_shape=claim.gang_shape.signature,
        load_thermal_power_envelope=claim.load_thermal_power_envelope,
        cpu_cores=claim.cpu_cores,
        numa_nodes=claim.numa_nodes,
        ram_bytes=claim.ram_bytes,
        disk_io_class=claim.disk_io_class,
        network_class=claim.network_class,
        contention_class=claim.contention_class,
        data_partition="interference_calibration_only",
    )


def _diagnose(
    rows: tuple[FormalPreflightInterferenceProofRow, ...],
    *,
    registry: ExperimentRegistry,
    inventory_sha256: str,
    hardware_envelope_sha256: str,
) -> InterferenceCalibrationGroupDiagnostic:
    runs = tuple(_calibration_run(row, registry=registry) for row in rows)
    group = InterferenceCalibrationGroup(
        group_id="formal-preflight-static-two-way",
        simultaneous_jobs=2,
        isolated=tuple(
            sorted(
                (run for run in runs if run.mode == "isolated"), key=lambda run: run.key
            )
        ),
        concurrent=tuple(
            sorted(
                (run for run in runs if run.mode == "concurrent"),
                key=lambda run: run.key,
            )
        ),
    )
    protocol = InterferenceCalibrationProtocol(
        schema_version=2,
        kind="interference_calibration_protocol",
        reducer_protocol_sha256=INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256,
        inventory_sha256=inventory_sha256,
        hardware_envelope_sha256=hardware_envelope_sha256,
        data_partition="interference_calibration_only",
        confirmation_data_visible=False,
        acceptance_status="REGISTERED",
        minimum_isolated_repetitions=2,
        minimum_concurrent_repetitions=2,
        maximum_absolute_relative_difference=0.01,
        confidence=0.95,
        interval_method="paired_bca_mean_log_ratio_v1",
        bootstrap_repetitions=10_000,
        bootstrap_seed=0,
    )
    observations = tuple(row.observation for row in rows)
    diagnostic = diagnose_interference_calibration(
        group,
        observations,
        protocol=protocol,
    )
    if all(row.slo_status == "PASS" for row in rows):
        return diagnostic
    return replace(
        diagnostic,
        status="FAIL",
        reason_codes=tuple(
            sorted({*diagnostic.reason_codes, "request_slo_qualification_failed"})
        ),
    )


def _aggregate_control_exact(
    *,
    control: ControlArtifactAttestation,
    binding: FormalPreflightInterferenceAggregateBinding,
) -> None:
    subject = control.subject
    if (
        control.deployment_policy_authorization.root_manifest_sha256
        != binding.expected_root_manifest_sha256
        or control.hardware_envelope_sha256 != binding.hardware_envelope_sha256
        or subject.artifact_type != "interference"
        or subject.artifact_sha256 != binding.sha256
        or subject.protocol_sha256
        != FORMAL_PREFLIGHT_INTERFERENCE_PROOF_PROTOCOL_SHA256
        or subject.registry_sha256 != binding.registry_sha256
        or subject.lineage_sha256 != binding.lineage_sha256
    ):
        raise ValueError("preflight interference aggregate control is not exact")


def _derive_formal_preflight_interference_aggregate(
    dispatch: object,
    *,
    raw_batch_path: str | Path,
    native_result_proof_paths: Mapping[str, str | Path],
    native_itl_proof_paths: Mapping[str, str | Path],
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> tuple[
    CanonicalJsonProofBinding,
    FormalPreflightInterferenceRawBatch,
    tuple[FormalPreflightInterferenceProofRow, ...],
    dict[str, object],
    FormalPreflightInterferenceAggregateBinding,
]:
    """Deep-open every upstream path and derive the unsigned control subject."""

    token = require_verified_formal_preflight_dispatch(dispatch)
    source_root = load_source_release_ed25519_root()
    if expected_root_manifest_sha256 != source_root.semantic_sha256:
        raise ValueError("interference proof uses another source release root")
    raw_binding = CanonicalJsonProofBinding.bind(raw_batch_path)
    raw_batch = FormalPreflightInterferenceRawBatch.from_dict(raw_binding.reopen())
    raw_batch.revalidate()
    activation = token.dispatch_context.activation_artifact
    if (
        raw_batch.status != "WAITING_FOR_LOCAL_CONTROL"
        or raw_binding.semantic_sha256 != raw_batch.sha256
        or raw_batch.dispatch_sha256 != token.sha256
        or raw_batch.registry_sha256 != token.dispatch_context.registry.sha256
        or raw_batch.activation_sha256 != activation.sha256
        or raw_batch.runtime_sha256 != activation.runtime_sha256
        or raw_batch.split_sha256 != activation.split_sha256
        or raw_batch.inventory_sha256 != token.subject.inventory_sha256
    ):
        raise ValueError("interference proof requires the exact waiting raw batch")
    expected = {row.registry_cell_id: row for row in _interference_bindings(token)}
    if set(native_result_proof_paths) != set(expected) or set(
        native_itl_proof_paths
    ) != set(expected):
        raise ValueError("interference proof upstream path coverage differs")
    proof_rows: list[FormalPreflightInterferenceProofRow] = []
    for raw_row in raw_batch.rows:
        execution_binding = expected[raw_row.registry_cell_id]
        result_binding = CanonicalJsonProofBinding.bind(
            native_result_proof_paths[raw_row.registry_cell_id]
        )
        from lightcone_spec.orchestration.formal_terminal_result import (
            validate_formal_terminal_result_proof_artifact,
        )

        result = validate_formal_terminal_result_proof_artifact(
            result_binding.absolute_path,
            expected_inventory_sha256=raw_batch.inventory_sha256,
            expected_registry_sha256=raw_batch.registry_sha256,
            expected_root_manifest_sha256=expected_root_manifest_sha256,
            expected_execution_plan_sha256=(raw_row.run_binding.execution_plan_sha256),
            expected_rank_config_sha256=raw_row.run_binding.rank_config_sha256,
            expected_run_id=raw_row.run_binding.run_id,
            expected_run_nonce_sha256=raw_row.run_binding.run_nonce_sha256,
            expected_attempt_id=raw_row.run_binding.attempt_id,
            expected_method="static",
            expected_stage="preflight",
            expected_topology="tp1_dp1",
            now_ns=now_ns,
        )
        itl_binding = CanonicalJsonProofBinding.bind(
            native_itl_proof_paths[raw_row.registry_cell_id]
        )
        topology_sha256, hardware_sha256 = _topology_sha256(execution_binding, token)
        identity = StageItlExecutionIdentity(
            schema_version=1,
            kind="stage_itl_execution_identity",
            materialized_cell_id=raw_row.materialized_cell_id,
            inventory_sha256=raw_batch.inventory_sha256,
            registry_sha256=raw_batch.registry_sha256,
            execution_plan_sha256=raw_row.run_binding.execution_plan_sha256,
            rank_config_sha256=raw_row.run_binding.rank_config_sha256,
            run_id=raw_row.run_binding.run_id,
            run_nonce_sha256=raw_row.run_binding.run_nonce_sha256,
            attempt_id=raw_row.run_binding.attempt_id,
            method="static",
        )
        itl = validate_stage_itl_timestamp_proof_artifact(
            itl_binding.absolute_path,
            expected_execution_identity=identity,
            expected_root_manifest_sha256=expected_root_manifest_sha256,
            now_ns=now_ns,
        )
        observation, slo_sha256, slo_status, qualified_request_ids = (
            _derive_observation(
                row=raw_row,
                terminal_authority_sha256=result_binding.semantic_sha256,
                result=result,
                itl=itl,
            )
        )
        proof_row = FormalPreflightInterferenceProofRow(
            materialized_cell_id=raw_row.materialized_cell_id,
            registry_cell_id=raw_row.registry_cell_id,
            assignment_sha256=raw_row.assignment_sha256,
            experiment_budget_sha256=raw_row.experiment_budget_sha256,
            gpu_uuid=raw_row.gpu_uuid,
            mode=raw_row.mode,
            repetition=raw_row.repetition,
            slot=raw_row.slot,
            run_binding=raw_row.run_binding,
            load_plan_sha256=_request_contract_sha256(result),
            topology_sha256=topology_sha256,
            hardware_envelope_sha256=hardware_sha256,
            native_result_proof=result_binding,
            native_itl_proof=itl_binding,
            slo_accounting_sha256=slo_sha256,
            slo_status=slo_status,
            qualified_request_ids=qualified_request_ids,
            observation=observation,
        )
        _revalidate_proof_row(
            proof_row,
            raw_row=raw_row,
            expected_registry_sha256=raw_batch.registry_sha256,
            expected_inventory_sha256=raw_batch.inventory_sha256,
            expected_root_manifest_sha256=expected_root_manifest_sha256,
            now_ns=now_ns,
        )
        proof_rows.append(proof_row)
    rows = tuple(sorted(proof_rows, key=lambda row: row.sha256))
    hardware_ids = {row.hardware_envelope_sha256 for row in rows}
    if len(hardware_ids) != 1:
        raise ValueError("interference proof spans heterogeneous hardware")
    hardware_sha256 = next(iter(hardware_ids))
    diagnostic = _diagnose(
        rows,
        registry=token.dispatch_context.registry,
        inventory_sha256=raw_batch.inventory_sha256,
        hardware_envelope_sha256=hardware_sha256,
    )
    diagnostic_dict = diagnostic.to_dict()
    binding = FormalPreflightInterferenceAggregateBinding(
        schema_version=1,
        kind="formal_preflight_interference_aggregate_binding",
        raw_batch_sha256=raw_batch.sha256,
        dispatch_sha256=token.sha256,
        registry_sha256=raw_batch.registry_sha256,
        activation_sha256=raw_batch.activation_sha256,
        runtime_sha256=raw_batch.runtime_sha256,
        split_sha256=raw_batch.split_sha256,
        inventory_sha256=raw_batch.inventory_sha256,
        hardware_envelope_sha256=hardware_sha256,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        status="PASSED" if diagnostic.status == "PASS" else "FAILED",
        diagnostic_sha256=content_sha256(diagnostic_dict),
        row_sha256s=tuple(row.sha256 for row in rows),
    )
    return raw_binding, raw_batch, rows, diagnostic_dict, binding


def build_formal_preflight_interference_aggregate_binding(
    dispatch: object,
    *,
    raw_batch_path: str | Path,
    native_result_proof_paths: Mapping[str, str | Path],
    native_itl_proof_paths: Mapping[str, str | Path],
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> FormalPreflightInterferenceAggregateBinding:
    """Derive the exact local-control subject without granting authority."""

    return _derive_formal_preflight_interference_aggregate(
        dispatch,
        raw_batch_path=raw_batch_path,
        native_result_proof_paths=native_result_proof_paths,
        native_itl_proof_paths=native_itl_proof_paths,
        expected_root_manifest_sha256=expected_root_manifest_sha256,
        now_ns=now_ns,
    )[-1]


def publish_formal_preflight_interference_proof_artifact(
    dispatch: object,
    *,
    raw_batch_path: str | Path,
    native_result_proof_paths: Mapping[str, str | Path],
    native_itl_proof_paths: Mapping[str, str | Path],
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    proof_artifact_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Locally authorize the exact pulled eight-row batch and publish it."""

    root_sha256 = (
        control_attestation.deployment_policy_authorization.root_manifest_sha256
    )
    raw_binding, raw_batch, rows, diagnostic_dict, binding = (
        _derive_formal_preflight_interference_aggregate(
            dispatch,
            raw_batch_path=raw_batch_path,
            native_result_proof_paths=native_result_proof_paths,
            native_itl_proof_paths=native_itl_proof_paths,
            expected_root_manifest_sha256=root_sha256,
            now_ns=now_ns,
        )
    )
    _aggregate_control_exact(control=control_attestation, binding=binding)
    upstream_reservation_roots: set[Path] = set()
    for row in rows:
        result_artifact = _preflight_native_terminal_source(row.native_result_proof)
        itl_artifact = StageItlTimestampProofArtifact.from_dict(
            row.native_itl_proof.reopen()
        )
        upstream_reservation_roots.add(
            Path(result_artifact.replay_reservation.path).parent
        )
        upstream_reservation_roots.add(
            Path(itl_artifact.replay_reservation.path).parent
        )
    if upstream_reservation_roots != {Path(replay_store.root)}:
        raise ValueError("interference aggregate uses another replay ledger")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=raw_batch.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified,
        reserved_ns=now_ns,
    )
    reservation = replay_store.bind_reservation(reservation_sha256)
    artifact = FormalPreflightInterferenceProofArtifact(
        schema_version=1,
        kind="formal_preflight_interference_proof_artifact",
        raw_batch=raw_binding,
        binding=binding,
        rows=rows,
        diagnostic=diagnostic_dict,
        control_attestation=control_attestation,
        replay_reservation=reservation,
    )
    try:
        publish_canonical_json_no_replace(proof_artifact_path, artifact.to_dict())
    except Exception as error:
        raise RuntimeError(
            "interference proof publication failed after reservation; issue fresh control"
        ) from error
    return CanonicalJsonProofBinding.bind(
        proof_artifact_path,
        semantic_sha256=artifact.sha256,
    )


def validate_formal_preflight_interference_proof_artifact(
    proof_artifact_path: str | Path,
    *,
    registry: ExperimentRegistry,
    expected_activation_sha256: str,
    expected_runtime_sha256: str,
    expected_split_sha256: str,
    expected_inventory_sha256: str,
    now_ns: int,
) -> VerifiedFormalPreflightInterferenceProof:
    """Deep-reopen a durable aggregate without consuming replay state."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("interference proof validation requires exact registry")
    source_root = load_source_release_ed25519_root()
    proof_binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalPreflightInterferenceProofArtifact.from_dict(
        proof_binding.reopen()
    )
    binding = artifact.binding
    if (
        proof_binding.semantic_sha256 != artifact.sha256
        or binding.registry_sha256 != registry.sha256
        or binding.activation_sha256 != expected_activation_sha256
        or binding.runtime_sha256 != expected_runtime_sha256
        or binding.split_sha256 != expected_split_sha256
        or binding.inventory_sha256 != expected_inventory_sha256
        or binding.expected_root_manifest_sha256 != source_root.semantic_sha256
        or artifact.raw_batch.semantic_sha256 != binding.raw_batch_sha256
    ):
        raise ValueError("interference proof expected identity differs")
    raw_batch = FormalPreflightInterferenceRawBatch.from_dict(
        artifact.raw_batch.reopen()
    )
    raw_batch.revalidate()
    if (
        raw_batch.status != "WAITING_FOR_LOCAL_CONTROL"
        or raw_batch.sha256 != binding.raw_batch_sha256
        or raw_batch.registry_sha256 != binding.registry_sha256
        or raw_batch.activation_sha256 != binding.activation_sha256
        or raw_batch.runtime_sha256 != binding.runtime_sha256
        or raw_batch.split_sha256 != binding.split_sha256
        or raw_batch.inventory_sha256 != binding.inventory_sha256
    ):
        raise ValueError("interference proof raw batch changed")
    by_cell = {row.registry_cell_id: row for row in raw_batch.rows}
    proof_by_cell = {row.registry_cell_id: row for row in artifact.rows}
    if len(proof_by_cell) != len(artifact.rows) or set(proof_by_cell) != set(by_cell):
        raise ValueError("interference proof row coverage differs from raw batch")
    for row in artifact.rows:
        raw_row = by_cell.get(row.registry_cell_id)
        if raw_row is None:
            raise ValueError("interference proof has a foreign row")
        if (
            row.materialized_cell_id != raw_row.materialized_cell_id
            or row.assignment_sha256 != raw_row.assignment_sha256
            or row.experiment_budget_sha256 != raw_row.experiment_budget_sha256
            or row.gpu_uuid != raw_row.gpu_uuid
            or (row.mode, row.repetition, row.slot)
            != (raw_row.mode, raw_row.repetition, raw_row.slot)
            or row.run_binding != raw_row.run_binding
        ):
            raise ValueError("interference proof row differs from raw assignment")
        _revalidate_proof_row(
            row,
            raw_row=raw_row,
            expected_registry_sha256=binding.registry_sha256,
            expected_inventory_sha256=binding.inventory_sha256,
            expected_root_manifest_sha256=binding.expected_root_manifest_sha256,
            now_ns=now_ns,
        )
    diagnostic = _diagnose(
        artifact.rows,
        registry=registry,
        inventory_sha256=binding.inventory_sha256,
        hardware_envelope_sha256=binding.hardware_envelope_sha256,
    )
    if (
        diagnostic.to_dict() != artifact.diagnostic
        or content_sha256(diagnostic.to_dict()) != binding.diagnostic_sha256
        or binding.status != ("PASSED" if diagnostic.status == "PASS" else "FAILED")
    ):
        raise ValueError("interference proof reduction changed")
    _aggregate_control_exact(control=artifact.control_attestation, binding=binding)
    reserved = artifact.replay_reservation.revalidate()
    if type(now_ns) is not int or now_ns < artifact.replay_reservation.reserved_ns:
        raise ValueError("interference proof validation time precedes reservation")
    verified = verify_release_control_artifact_attestation(
        artifact.control_attestation,
        expected_inventory_sha256=binding.inventory_sha256,
        now_ns=artifact.replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    expected_challenges = tuple(
        sorted(
            {
                verified.challenge_sha256,
                verified.deployment_policy_challenge_sha256,
            }
        )
    )
    if tuple(reserved) != expected_challenges:
        raise ValueError("interference proof reservation challenge set differs")
    expected_reservation_sha256 = control_challenge_reservation_sha256(
        (verified,), reserved_ns=artifact.replay_reservation.reserved_ns
    )
    if artifact.replay_reservation.reservation_sha256 != expected_reservation_sha256:
        raise ValueError("interference proof reservation identity differs")
    return VerifiedFormalPreflightInterferenceProof(
        artifact_sha256=artifact.sha256,
        binding=binding,
        rows=artifact.rows,
        diagnostic=diagnostic,
        _verification_tag=_VERIFIED_INTERFERENCE_SENTINEL,
    )


__all__ = [
    "FORMAL_PREFLIGHT_INTERFERENCE_PROOF_PROTOCOL_SHA256",
    "FORMAL_PREFLIGHT_INTERFERENCE_RAW_PROTOCOL_SHA256",
    "FormalPreflightInterferenceAggregateBinding",
    "FormalPreflightInterferenceFatalTerminal",
    "FormalPreflightInterferenceProofArtifact",
    "FormalPreflightInterferenceProofRow",
    "FormalPreflightInterferenceQualificationLock",
    "FormalPreflightInterferenceQualificationRow",
    "FormalPreflightInterferenceRawBatch",
    "FormalPreflightInterferenceRawRow",
    "VerifiedFormalPreflightInterferenceProof",
    "build_formal_preflight_interference_aggregate_binding",
    "build_formal_preflight_interference_raw_row",
    "publish_formal_preflight_interference_fatal_terminal",
    "publish_formal_preflight_interference_proof_artifact",
    "publish_formal_preflight_interference_qualification_lock",
    "publish_formal_preflight_interference_raw_batch",
    "validate_formal_preflight_interference_proof_artifact",
]
