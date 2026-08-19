"""Physical trusted-operator TP1 session-reset qualification.

The worker in this module is intentionally an unsigned empirical producer.  It
opens a path-only plan, exercises exactly two compatible TP1 traces through one
``SessionLivePinnedBenchTransport`` and one server process, publishes the raw
evidence needed by the trusted empirical authority, and invokes that authority's
path-only publisher only after all eight registered assertions pass.

No object in this module can construct ``VerifiedNativeRuntimeGpuProof`` or make
a repository-level ``MEASURED`` claim.  A failure keeps its partial evidence and
never publishes a reset authority.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Protocol, Self
from urllib.parse import urlsplit

from lightcone_spec.config import RunConfig, load_run_config
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundleBinding,
)
from lightcone_spec.experiments.formal_single_operator_session_reset import (
    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_QUALIFICATION_SPEC_PROTOCOL_SHA256,
    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE,
    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS,
    TrustedEmpiricalTp1SessionResetQualificationSpec,
    publish_trusted_empirical_tp1_session_reset_authority,
    revalidate_trusted_empirical_tp1_session_reset_authority,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.orchestration.formal_serving_session_group import (
    FormalServingNormalizedProcessKey,
    FormalServingSessionGroupSpec,
    normalized_formal_serving_process_key,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalProvider,
    TerminalRequestExpectation,
    canonical_json_bytes,
)
from lightcone_spec.orchestration.session_live_runtime import (
    SessionLiveContractResult,
    SessionLiveEvidenceSink,
    SessionLivePinnedBenchTransport,
    SessionLiveProcessOwner,
    SessionLiveStepBinding,
    SessionLiveTraceInput,
    run_session_live_contract,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_PLAN_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "trusted_empirical_tp1_session_reset_physical_plan",
        "input": (
            "paths_only_protocol_lock_content_inventory_two_group_specs_and_"
            "two_compile_launch_manifests"
        ),
        "trace_count": 2,
        "launch": "one_tp1_server_one_http_pool_no_verified_gpu_token",
        "evidence": (
            "two_native_terminal_chains_exact_completed_outputs_native_token_"
            "timestamps_initial_plus_two_reset_hbm_and_eight_assertion_junit"
        ),
        "success": "path_only_trusted_empirical_authority_publisher",
        "failure": "retain_raw_prefix_and_never_publish_authority",
        "formal_measured": False,
    }
)
TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_RESULT_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "trusted_empirical_tp1_session_reset_physical_result",
        "status": ("PASS", "FAIL"),
        "authority": "present_only_after_exact_8_of_8_pass",
        "claim": "trusted_single_operator_empirical_no_signature",
        "formal_measured": False,
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_path(label: str, value: object) -> Path:
    if type(value) is not str:
        raise TypeError(f"{label} must be a path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path == Path(path.anchor)
    ):
        raise ValueError(f"{label} must be an absolute normalized non-root path")
    return path


@dataclass(frozen=True)
class TrustedEmpiricalTp1SessionResetPhysicalPlan:
    """Path-only physical qualification input.

    The worker accepts only a canonical path containing this object.  All
    digests, scope, process compatibility, and qualification IDs are derived by
    reopening the named artifacts.
    """

    schema_version: Literal[1]
    kind: Literal["trusted_empirical_tp1_session_reset_physical_plan"]
    protocol_sha256: str
    protocol_lock_path: str
    content_bundle_path: str
    inventory_path: str
    trace_spec_paths: tuple[str, str]
    compile_launch_manifest_paths: tuple[str, str]
    output_directory: str
    request_timeout_seconds: float
    abort_timeout_seconds: float
    hbm_allowed_growth_bytes: int
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_empirical_tp1_session_reset_physical_plan"
            or self.protocol_sha256
            != TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_PLAN_PROTOCOL_SHA256
            or self.formal_measured is not False
        ):
            raise ValueError("trusted empirical reset physical plan schema differs")
        input_paths = (
            self.protocol_lock_path,
            self.content_bundle_path,
            self.inventory_path,
            *self.trace_spec_paths,
            *self.compile_launch_manifest_paths,
        )
        for index, value in enumerate(input_paths):
            _absolute_path(f"trusted empirical reset physical input {index}", value)
        if len(set(input_paths)) != len(input_paths):
            raise ValueError("trusted empirical reset physical inputs alias")
        _absolute_path("trusted empirical reset physical output", self.output_directory)
        if (
            type(self.trace_spec_paths) is not tuple
            or len(self.trace_spec_paths) != 2
            or type(self.compile_launch_manifest_paths) is not tuple
            or len(self.compile_launch_manifest_paths) != 2
        ):
            raise ValueError("trusted empirical reset requires exactly two traces")
        for label, value in (
            ("request timeout", self.request_timeout_seconds),
            ("abort timeout", self.abort_timeout_seconds),
        ):
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0
                or value > 3_600
            ):
                raise ValueError(f"trusted empirical reset {label} is invalid")
        if (
            type(self.hbm_allowed_growth_bytes) is not int
            or self.hbm_allowed_growth_bytes < 0
        ):
            raise ValueError("trusted empirical reset HBM tolerance is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["trace_spec_paths"] = list(self.trace_spec_paths)
        value["compile_launch_manifest_paths"] = list(
            self.compile_launch_manifest_paths
        )
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("trusted empirical reset physical plan fields differ")
        row = dict(value)
        traces = row.pop("trace_spec_paths")
        launches = row.pop("compile_launch_manifest_paths")
        if type(traces) is not list or type(launches) is not list:
            raise TypeError("trusted empirical reset physical paths must be arrays")
        return cls(
            **row,
            trace_spec_paths=tuple(traces),
            compile_launch_manifest_paths=tuple(launches),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class RevalidatedTrustedEmpiricalTp1SessionResetPhysicalPlan:
    binding: CanonicalJsonProofBinding
    plan: TrustedEmpiricalTp1SessionResetPhysicalPlan
    trace_bindings: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]
    traces: tuple[FormalServingSessionGroupSpec, FormalServingSessionGroupSpec]
    launches: tuple[CompileLaunchManifest, CompileLaunchManifest]
    configs: tuple[RunConfig, RunConfig]
    process_key: FormalServingNormalizedProcessKey
    protocol_lock: FormalSingleOperatorJsonBinding
    content_bundle: TrustedSingleOperatorContentBundleBinding
    inventory: CanonicalJsonProofBinding
    qualification_run_id: str

    @property
    def execution_plan_sha256s(self) -> tuple[str, str]:
        return (
            self.traces[0].run_plan.semantic_sha256,
            self.traces[1].run_plan.semantic_sha256,
        )

    @property
    def gpu_uuid(self) -> str:
        return self.traces[0].assigned_gpu_uuids[0]

    @property
    def backend(self) -> str:
        return self.traces[0].backend

    @property
    def method_family(self) -> str:
        return self.traces[0].method_family


def publish_trusted_empirical_tp1_session_reset_physical_plan(
    *,
    plan: TrustedEmpiricalTp1SessionResetPhysicalPlan,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish a path-only plan without accepting caller-authored digests."""

    if type(plan) is not TrustedEmpiricalTp1SessionResetPhysicalPlan:
        raise TypeError("trusted empirical reset physical plan type differs")
    publish_canonical_json_no_replace(output_path, plan.to_dict())
    reopened = revalidate_trusted_empirical_tp1_session_reset_physical_plan(output_path)
    binding = reopened.binding
    if reopened.plan != plan or binding.semantic_sha256 != plan.sha256:
        raise RuntimeError("trusted empirical reset physical plan changed")
    return binding


def revalidate_trusted_empirical_tp1_session_reset_physical_plan(
    path: str | Path,
) -> RevalidatedTrustedEmpiricalTp1SessionResetPhysicalPlan:
    """Deep-reopen the path-only plan and its exact two compatible launches."""

    binding = CanonicalJsonProofBinding.bind(path)
    plan = TrustedEmpiricalTp1SessionResetPhysicalPlan.from_dict(binding.reopen())
    if binding.semantic_sha256 != plan.sha256:
        raise ValueError("trusted empirical reset physical plan identity differs")
    protocol_lock = FormalSingleOperatorJsonBinding.bind(
        plan.protocol_lock_path,
        label="trusted empirical reset physical ProtocolLock",
    )
    content_bundle = TrustedSingleOperatorContentBundleBinding.bind(
        plan.content_bundle_path
    )
    content = content_bundle.reopen()
    inventory_binding = CanonicalJsonProofBinding.bind(plan.inventory_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    trace_bindings = tuple(
        CanonicalJsonProofBinding.bind(item) for item in plan.trace_spec_paths
    )
    traces = tuple(
        FormalServingSessionGroupSpec.from_dict(item.reopen())
        for item in trace_bindings
    )
    launches = tuple(
        CompileLaunchManifest.load(item) for item in plan.compile_launch_manifest_paths
    )
    configs = tuple(load_run_config(item.run_config_path) for item in launches)
    keys = tuple(
        normalized_formal_serving_process_key(launch=launch, config=config)
        for launch, config in zip(launches, configs, strict=True)
    )
    first, second = traces
    if (
        len({(row.materialized_cell_id, row.attempt) for row in traces}) != 2
        or len({row.run_plan.semantic_sha256 for row in traces}) != 2
        or len({row.output_directory for row in traces}) != 2
        or first.request_schedule_sha256 is None
        or first.request_schedule_sha256 != second.request_schedule_sha256
        or first.protocol_lock_sha256 != protocol_lock.semantic_sha256
        or second.protocol_lock_sha256 != protocol_lock.semantic_sha256
        or first.inventory_sha256 != inventory_binding.semantic_sha256
        or second.inventory_sha256 != inventory_binding.semantic_sha256
        or first.source_snapshot_sha256
        != content.source_snapshot.source_snapshot_sha256
        or second.source_snapshot_sha256
        != content.source_snapshot.source_snapshot_sha256
        or first.assigned_gpu_uuids != second.assigned_gpu_uuids
        or len(first.assigned_gpu_uuids) != 1
        or first.backend != second.backend
        or first.method_family != second.method_family
        or first.method != second.method
        or first.topology_mode != "tp1_dp1"
        or second.topology_mode != "tp1_dp1"
        or first.reuse_exclusion_reason is not None
        or second.reuse_exclusion_reason is not None
        or keys[0] != keys[1]
    ):
        raise ValueError("trusted empirical reset traces are not exactly compatible")
    for trace, launch in zip(traces, launches, strict=True):
        if (
            trace.compile_launch_manifest_sha256 != launch.sha256
            or trace.run_config_sha256 != launch.run_config_semantic_sha256
            or trace.assigned_gpu_uuids != launch.gpu_uuids
        ):
            raise ValueError("trusted empirical reset trace launch identity differs")
    inventory.device(first.assigned_gpu_uuids[0])
    qualification_run_id = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_empirical_tp1_session_reset_qualification_run",
            "physical_plan_sha256": plan.sha256,
            "trace_spec_sha256s": [row.sha256 for row in traces],
            "launch_manifest_sha256s": [row.sha256 for row in launches],
            "normalized_process_key_sha256": keys[0].sha256,
        }
    )
    return RevalidatedTrustedEmpiricalTp1SessionResetPhysicalPlan(
        binding=binding,
        plan=plan,
        trace_bindings=(trace_bindings[0], trace_bindings[1]),
        traces=(traces[0], traces[1]),
        launches=(launches[0], launches[1]),
        configs=(configs[0], configs[1]),
        process_key=keys[0],
        protocol_lock=protocol_lock,
        content_bundle=content_bundle,
        inventory=inventory_binding,
        qualification_run_id=qualification_run_id,
    )


@dataclass(frozen=True)
class TrustedEmpiricalTp1SessionResetLiveResources:
    """One already-launched physical process and its exact two live traces."""

    server_pid: int
    base_url: str
    transport: SessionLivePinnedBenchTransport
    provider: NativeTerminalProvider
    process_owner: SessionLiveProcessOwner
    traces: tuple[SessionLiveTraceInput, SessionLiveTraceInput]
    native_timestamp_evidence_paths: tuple[str, str]

    def validate(
        self,
        *,
        plan: RevalidatedTrustedEmpiricalTp1SessionResetPhysicalPlan,
        expected_timestamp_paths: tuple[str, str],
    ) -> None:
        if type(self.server_pid) is not int or self.server_pid < 1:
            raise ValueError("trusted empirical reset server PID is invalid")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("trusted empirical reset base URL is not loopback HTTP")
        if type(self.transport) is not SessionLivePinnedBenchTransport:
            raise TypeError("trusted empirical reset requires the pinned HTTP pool")
        if (
            type(self.provider) is not NativeTerminalProvider
            or self.provider._transport is not self.transport
        ):
            raise TypeError("trusted empirical reset terminal provider changed pool")
        if not callable(getattr(self.process_owner, "close", None)) or not callable(
            getattr(self.process_owner, "force_close", None)
        ):
            raise TypeError("trusted empirical reset process owner is incomplete")
        if type(self.traces) is not tuple or len(self.traces) != 2:
            raise ValueError("trusted empirical reset runtime requires two traces")
        expected_plans = plan.execution_plan_sha256s
        for index, trace in enumerate(self.traces):
            if type(trace) is not SessionLiveTraceInput:
                raise TypeError("trusted empirical reset live trace type differs")
            trace.validate()
            binding = trace.binding
            if (
                binding.execution_plan_sha256 != expected_plans[index]
                or binding.session_id != plan.qualification_run_id
                or binding.session_epoch != index + 1
                or binding.previous_run_id
                != (None if index == 0 else self.traces[index - 1].binding.run_id)
                or binding.method != plan.traces[index].method
            ):
                raise ValueError("trusted empirical reset live trace lineage differs")
        if self.native_timestamp_evidence_paths != expected_timestamp_paths:
            raise ValueError("trusted empirical reset timestamp evidence paths differ")


class TrustedEmpiricalTp1SessionResetPhysicalRuntime(Protocol):
    """Adapter implemented by the physical serving runner integration."""

    async def launch(
        self,
        *,
        plan: RevalidatedTrustedEmpiricalTp1SessionResetPhysicalPlan,
        evidence_sink: SessionLiveEvidenceSink,
        native_timestamp_evidence_paths: tuple[str, str],
    ) -> TrustedEmpiricalTp1SessionResetLiveResources: ...


class _RawEvidenceSink(SessionLiveEvidenceSink):
    def __init__(self, path: Path, *, qualification_run_id: str) -> None:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        self._file = os.fdopen(descriptor, "wb", buffering=0)
        self._closed = False
        self.writer_flushed = False
        self._write(
            {
                "kind": "trusted_empirical_tp1_session_reset_raw_begin",
                "qualification_run_id": qualification_run_id,
            }
        )

    def _write(self, value: object) -> None:
        if self._closed:
            raise RuntimeError("trusted empirical reset raw writer is closed")
        self._file.write(canonical_json_bytes(value) + b"\n")

    def record_step(self, step: SessionLiveStepBinding) -> None:
        self._write(
            {
                "kind": "trusted_empirical_tp1_session_reset_raw_step",
                "step": step.step,
                "execution_plan_sha256": step.execution_plan_sha256,
                "content_sha256": step.content_sha256,
                "raw_json": step.raw_json,
            }
        )
        self.flush()

    def finalize(self, result: SessionLiveContractResult) -> None:
        self._write(
            {
                "kind": "trusted_empirical_tp1_session_reset_raw_live_result",
                "audit_sha256": result.audit.sha256,
                "terminal_sha256s": [
                    item.terminal_sha256 for item in result.native_terminals
                ],
                "status": result.audit.status,
            }
        )
        self.flush()

    def close_partial(self) -> None:
        if self._closed:
            return
        self._write({"kind": "trusted_empirical_tp1_session_reset_raw_partial_close"})
        self.flush()

    def append_bound_evidence(
        self, *, label: str, binding: CanonicalJsonProofBinding
    ) -> None:
        self._write(
            {
                "kind": "trusted_empirical_tp1_session_reset_raw_member",
                "label": label,
                "binding": binding.to_dict(),
                "value": binding.reopen(),
            }
        )
        self.flush()

    def flush(self) -> None:
        if self._closed:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self.writer_flushed = True

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._file.close()
        self._closed = True


@dataclass(frozen=True)
class TrustedEmpiricalTp1SessionResetPhysicalResult:
    schema_version: Literal[1]
    kind: Literal["trusted_empirical_tp1_session_reset_physical_result"]
    protocol_sha256: str
    status: Literal["PASS", "FAIL"]
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]
    qualification_run_id: str
    plan: CanonicalJsonProofBinding
    raw_terminal: EvidenceFileBinding
    junit_xml: EvidenceFileBinding
    native_lifecycle: CanonicalJsonProofBinding | None
    reset_state_evidence: CanonicalJsonProofBinding | None
    hbm_evidence: CanonicalJsonProofBinding | None
    qualification_spec: CanonicalJsonProofBinding | None
    authority: CanonicalJsonProofBinding | None
    failure_terminal: CanonicalJsonProofBinding | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_empirical_tp1_session_reset_physical_result"
            or self.protocol_sha256
            != TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_RESULT_PROTOCOL_SHA256
            or self.evidence_level != "trusted_single_operator_empirical_no_signature"
            or self.formal_measured is not False
            or self.status not in {"PASS", "FAIL"}
        ):
            raise ValueError("trusted empirical reset physical result differs")
        _require_sha256(
            "trusted empirical reset physical qualification run",
            self.qualification_run_id,
        )
        if type(self.plan) is not CanonicalJsonProofBinding:
            raise TypeError("trusted empirical reset physical result lacks its plan")
        if (
            type(self.raw_terminal) is not EvidenceFileBinding
            or type(self.junit_xml) is not EvidenceFileBinding
        ):
            raise TypeError("trusted empirical reset physical raw evidence differs")
        successful = (
            self.native_lifecycle,
            self.reset_state_evidence,
            self.hbm_evidence,
            self.qualification_spec,
            self.authority,
        )
        if self.status == "PASS":
            if any(type(item) is not CanonicalJsonProofBinding for item in successful):
                raise TypeError("trusted empirical reset PASS lacks bound evidence")
            if self.failure_terminal is not None:
                raise ValueError("trusted empirical reset PASS carries a failure")
        elif (
            self.authority is not None
            or self.qualification_spec is not None
            or type(self.failure_terminal) is not CanonicalJsonProofBinding
        ):
            raise ValueError("trusted empirical reset FAIL could authorize reuse")


def _strict_timestamp_evidence(
    binding: CanonicalJsonProofBinding,
    *,
    qualification_run_id: str,
    execution_plan_sha256: str,
    requests: tuple[TerminalRequestExpectation, ...],
) -> None:
    value = binding.reopen()
    if type(value) is not dict or set(value) != {
        "schema_version",
        "kind",
        "qualification_run_id",
        "execution_plan_sha256",
        "requests",
    }:
        raise ValueError("trusted empirical reset native timestamp fields differ")
    rows = value["requests"]
    if (
        value["schema_version"] != 1
        or value["kind"] != "trusted_empirical_tp1_session_reset_native_timestamps"
        or value["qualification_run_id"] != qualification_run_id
        or value["execution_plan_sha256"] != execution_plan_sha256
        or type(rows) is not list
        or len(rows) != len(requests)
    ):
        raise ValueError("trusted empirical reset timestamp scope differs")
    for row, request in zip(rows, requests, strict=True):
        if type(row) is not dict or set(row) != {
            "request_id",
            "output_token_ids",
            "native_token_timestamps_ns",
            "request_started_ns",
            "request_terminal_ns",
        }:
            raise ValueError("trusted empirical reset timestamp row differs")
        timestamps = row["native_token_timestamps_ns"]
        outputs = row["output_token_ids"]
        if (
            request.output_token_ids is None
            or row["request_id"] != request.request_id
            or outputs != list(request.output_token_ids)
            or type(timestamps) is not list
            or len(timestamps) != len(request.output_token_ids)
            or any(type(item) is not int or item < 0 for item in timestamps)
            or timestamps != sorted(timestamps)
            or type(row["request_started_ns"]) is not int
            or type(row["request_terminal_ns"]) is not int
            or row["request_started_ns"] < 0
            or row["request_terminal_ns"] < row["request_started_ns"]
            or (
                timestamps
                and (
                    timestamps[0] < row["request_started_ns"]
                    or timestamps[-1] > row["request_terminal_ns"]
                )
            )
        ):
            raise ValueError("trusted empirical reset native timestamps are incomplete")


def _step_object(result: SessionLiveContractResult, name: str) -> dict[str, object]:
    matches = [step for step in result.steps if step.step == name]
    if len(matches) != 1:
        raise ValueError(f"trusted empirical reset requires one {name}")
    value = json.loads(matches[0].raw_json)
    if type(value) is not dict:
        raise TypeError(f"trusted empirical reset {name} is not an object")
    return value


def _reset_step_objects(
    result: SessionLiveContractResult,
) -> tuple[dict[str, object], dict[str, object]]:
    values = []
    for step in result.steps:
        if step.step == "session_reset_boundary":
            value = json.loads(step.raw_json)
            if type(value) is not dict:
                raise TypeError("trusted empirical reset boundary is not an object")
            values.append(value)
    if len(values) != 2:
        raise ValueError("trusted empirical reset requires two reset boundaries")
    return values[0], values[1]


def _state_hbm_bytes(value: object) -> int:
    if type(value) is not dict:
        raise TypeError("trusted empirical reset state is not an object")
    allocated = value.get("allocator_allocated_bytes")
    reserved = value.get("allocator_reserved_bytes")
    if (
        type(allocated) is not int
        or allocated < 0
        or type(reserved) is not int
        or reserved < allocated
    ):
        raise ValueError("trusted empirical reset HBM state differs")
    return reserved


def _trajectory(
    requests: tuple[TerminalRequestExpectation, ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    rows = []
    for request in requests:
        request.validate()
        if (
            not request.submitted_to_server
            or request.terminal_status != "completed"
            or request.output_token_ids is None
        ):
            raise ValueError("trusted empirical reset requires completed outputs")
        rows.append((request.input_token_ids, request.output_token_ids))
    if not rows:
        raise ValueError("trusted empirical reset trace has no completed request")
    return tuple(rows)


def _write_junit(path: Path, assertions: dict[str, bool], *, error: str | None) -> None:
    root = ET.Element(
        "testsuite",
        {
            "tests": "8",
            "failures": str(sum(not value for value in assertions.values())),
            "errors": "0",
            "skipped": "0",
        },
    )
    for name in TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS:
        case = ET.SubElement(root, "testcase", {"name": name})
        if not assertions[name]:
            failure = ET.SubElement(case, "failure", {"message": error or name})
            failure.text = error or f"{name} failed"
    body = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", buffering=0) as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _common_scope(
    plan: RevalidatedTrustedEmpiricalTp1SessionResetPhysicalPlan,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "suite_id": TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE,
        "topology_mode": "tp1_dp1",
        "gpu_uuid": plan.gpu_uuid,
        "backend": plan.backend,
        "method_family": plan.method_family,
        "qualification_run_id": plan.qualification_run_id,
    }


def _publish_evidence(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


async def execute_trusted_empirical_tp1_session_reset_qualification(
    *,
    physical_plan_path: str | Path,
    runtime: TrustedEmpiricalTp1SessionResetPhysicalRuntime,
) -> TrustedEmpiricalTp1SessionResetPhysicalResult:
    """Execute exactly two live traces and publish authority only on 8/8 PASS."""

    plan = revalidate_trusted_empirical_tp1_session_reset_physical_plan(
        physical_plan_path
    )
    output = Path(plan.plan.output_directory)
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(
                "trusted empirical reset output must be absent or empty"
            )
    else:
        output.mkdir(parents=False, mode=0o700)
    paths = {
        "raw": output / "raw-terminal.jsonl",
        "junit": output / "junit.xml",
        "lifecycle": output / "native-lifecycle.json",
        "reset": output / "reset-state.json",
        "hbm": output / "hbm.json",
        "spec": output / "qualification-spec.json",
        "authority": output / "authority.json",
        "failure": output / "failure.json",
    }
    timestamp_paths = (
        str((output / "trace-1-native-timestamps.json").resolve()),
        str((output / "trace-2-native-timestamps.json").resolve()),
    )
    sink = _RawEvidenceSink(
        paths["raw"], qualification_run_id=plan.qualification_run_id
    )
    assertions = {name: False for name in TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS}
    lifecycle_binding: CanonicalJsonProofBinding | None = None
    reset_binding: CanonicalJsonProofBinding | None = None
    hbm_binding: CanonicalJsonProofBinding | None = None
    execution_error: BaseException | None = None
    result: SessionLiveContractResult | None = None
    try:
        resources = await runtime.launch(
            plan=plan,
            evidence_sink=sink,
            native_timestamp_evidence_paths=timestamp_paths,
        )
        if type(resources) is not TrustedEmpiricalTp1SessionResetLiveResources:
            raise TypeError("trusted empirical reset runtime resources differ")
        resources.validate(plan=plan, expected_timestamp_paths=timestamp_paths)
        result = await run_session_live_contract(
            session_plan_sha256=plan.qualification_run_id,
            traces=resources.traces,
            base_url=resources.base_url,
            request_timeout_s=plan.plan.request_timeout_seconds,
            abort_timeout_s=plan.plan.abort_timeout_seconds,
            transport=resources.transport,
            provider=resources.provider,
            process_owner=resources.process_owner,
            verified_gpu_proof=None,
        )
        if result.audit.status != "CPU_CONTRACT_ONLY" or result.reuse_authorized:
            raise ValueError("unsigned empirical reset unexpectedly authorized reuse")
        if len(result.native_terminals) != 2:
            raise ValueError("trusted empirical reset lacks two native terminals")
        timestamp_bindings = tuple(
            CanonicalJsonProofBinding.bind(item) for item in timestamp_paths
        )
        for index, (binding, terminal) in enumerate(
            zip(timestamp_bindings, result.native_terminals, strict=True)
        ):
            _strict_timestamp_evidence(
                binding,
                qualification_run_id=plan.qualification_run_id,
                execution_plan_sha256=plan.execution_plan_sha256s[index],
                requests=terminal.requests,
            )
            sink.append_bound_evidence(
                label=f"trace_{index + 1}_native_timestamps", binding=binding
            )
        sink.close()

        terminals = result.native_terminals
        pids = {
            resources.server_pid,
            *(item.begin_receipt.server_process_id for item in terminals),
            *(item.reset_receipt.server_process_id for item in terminals),
        }
        starts = {
            item.begin_receipt.server_process_started_ns for item in terminals
        } | {item.reset_receipt.server_process_started_ns for item in terminals}
        assertions["same_server_process_identity"] = len(pids) == len(starts) == 1
        assertions["native_session_epoch_lineage"] = (
            tuple(item.binding.session_epoch for item in terminals) == (1, 2)
            and terminals[0].binding.previous_run_id is None
            and terminals[1].binding.previous_run_id == terminals[0].binding.run_id
        )
        trajectories = tuple(_trajectory(item.requests) for item in terminals)
        assertions["exact_output_token_trajectory"] = trajectories[0] == trajectories[1]
        initial = _step_object(result, "session_initial_state")
        reset_rows = _reset_step_objects(result)
        initial_state = initial.get("state")
        reset_after = tuple(row.get("after") for row in reset_rows)
        clean_states = (initial_state, *reset_after)
        assertions["request_queue_empty_after_trace"] = all(
            type(state) is dict
            and state.get("active_requests") == 0
            and state.get("queued_requests") == 0
            and state.get("request_kv_entries") == 0
            for state in clean_states
        )
        assertions["optimizer_candidate_and_adaptation_state_reset"] = all(
            type(state) is dict
            and state.get("adapter_version") == 0
            and state.get("optimizer_generation") == 0
            and state.get("update_counter") == 0
            for state in clean_states
        )
        clean_projection = tuple(
            {
                key: value
                for key, value in state.items()
                if key
                not in {
                    "reset_generation",
                    "completion_event_generation",
                    "completion_event_complete",
                    "completion_event_sha256",
                    "connection_accounting",
                }
            }
            for state in clean_states
            if type(state) is dict
        )
        assertions["registered_cache_policy_restored"] = (
            len(clean_projection) == 3
            and clean_projection[0] == clean_projection[1] == clean_projection[2]
        )
        assertions["terminal_writer_fully_flushed"] = (
            sink.writer_flushed
            and result.transport_closed
            and result.process_closed
            and not result.process_force_closed
        )
        hbm_values = tuple(_state_hbm_bytes(state) for state in clean_states)
        monotonic_growth = hbm_values[0] < hbm_values[1] < hbm_values[2]
        assertions["hbm_returns_without_monotonic_growth"] = (
            not monotonic_growth
            and all(
                value <= hbm_values[0] + plan.plan.hbm_allowed_growth_bytes
                for value in hbm_values[1:]
            )
        )
        common = _common_scope(plan)
        lifecycle = {
            **common,
            "kind": "trusted_empirical_tp1_session_reset_native_lifecycle",
            "server_pid": resources.server_pid,
            "session_epochs": [1, 2],
            "execution_plan_sha256s": list(plan.execution_plan_sha256s),
            "exact_output_token_trajectory": assertions[
                "exact_output_token_trajectory"
            ],
            "native_timestamp_coverage": True,
        }
        reset_evidence = {
            **common,
            "kind": "trusted_empirical_tp1_session_reset_state_evidence",
            "reset_boundary_count": len(result.audit.reset_receipt_sha256s),
            "request_queue_empty": assertions["request_queue_empty_after_trace"],
            "optimizer_state_reset": assertions[
                "optimizer_candidate_and_adaptation_state_reset"
            ],
            "candidate_state_reset": assertions[
                "optimizer_candidate_and_adaptation_state_reset"
            ],
            "adaptation_state_reset": assertions[
                "optimizer_candidate_and_adaptation_state_reset"
            ],
            "registered_cache_policy_restored": assertions[
                "registered_cache_policy_restored"
            ],
            "terminal_writer_flushed": assertions["terminal_writer_fully_flushed"],
            "previous_requests_fully_terminal": all(
                request.terminal_status == "completed"
                for terminal in terminals
                for request in terminal.requests
            ),
        }
        hbm = {
            **common,
            "kind": "trusted_empirical_tp1_session_reset_hbm_evidence",
            "initial_memory_bytes": hbm_values[0],
            "memory_after_reset_bytes": list(hbm_values[1:]),
            "allowed_growth_bytes": plan.plan.hbm_allowed_growth_bytes,
            "monotonic_growth_detected": monotonic_growth,
        }
        lifecycle_binding = _publish_evidence(paths["lifecycle"], lifecycle)
        reset_binding = _publish_evidence(paths["reset"], reset_evidence)
        hbm_binding = _publish_evidence(paths["hbm"], hbm)
        if not all(assertions.values()):
            raise RuntimeError("trusted empirical reset registered assertion failed")
    except BaseException as error:  # noqa: BLE001 - retain all physical evidence
        execution_error = error
        sink.close_partial()
        sink.close()

    _write_junit(
        paths["junit"],
        assertions,
        error=(
            None
            if execution_error is None
            else f"{type(execution_error).__name__}: {execution_error}"
        ),
    )
    raw_binding = EvidenceFileBinding.bind(
        paths["raw"], label="trusted empirical reset raw terminal"
    )
    junit_binding = EvidenceFileBinding.bind(
        paths["junit"], label="trusted empirical reset JUnit"
    )
    if execution_error is not None:
        failure_binding = _publish_evidence(
            paths["failure"],
            {
                "schema_version": 1,
                "kind": "trusted_empirical_tp1_session_reset_physical_failure",
                "protocol_sha256": (
                    TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_RESULT_PROTOCOL_SHA256
                ),
                "qualification_run_id": plan.qualification_run_id,
                "physical_plan_sha256": plan.plan.sha256,
                "status": "FAIL",
                "error_type": type(execution_error).__name__,
                "error_message": str(execution_error),
                "passed_assertions": [
                    name for name, passed in assertions.items() if passed
                ],
                "failed_assertions": [
                    name for name, passed in assertions.items() if not passed
                ],
                "formal_measured": False,
            },
        )
        if paths["authority"].exists():
            raise RuntimeError(
                "failed trusted empirical qualification published an authority"
            )
        return TrustedEmpiricalTp1SessionResetPhysicalResult(
            schema_version=1,
            kind="trusted_empirical_tp1_session_reset_physical_result",
            protocol_sha256=(
                TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_RESULT_PROTOCOL_SHA256
            ),
            status="FAIL",
            evidence_level="trusted_single_operator_empirical_no_signature",
            formal_measured=False,
            qualification_run_id=plan.qualification_run_id,
            plan=plan.binding,
            raw_terminal=raw_binding,
            junit_xml=junit_binding,
            native_lifecycle=lifecycle_binding,
            reset_state_evidence=reset_binding,
            hbm_evidence=hbm_binding,
            qualification_spec=None,
            authority=None,
            failure_terminal=failure_binding,
        )

    qualification = TrustedEmpiricalTp1SessionResetQualificationSpec(
        schema_version=1,
        kind="trusted_empirical_tp1_session_reset_qualification_spec",
        protocol_sha256=(
            TRUSTED_EMPIRICAL_TP1_SESSION_RESET_QUALIFICATION_SPEC_PROTOCOL_SHA256
        ),
        topology_mode="tp1_dp1",
        suite_id=TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE,
        gpu_uuid=plan.gpu_uuid,
        backend=plan.backend,  # type: ignore[arg-type]
        method_family=plan.method_family,  # type: ignore[arg-type]
        qualification_run_id=plan.qualification_run_id,
        protocol_lock_path=plan.plan.protocol_lock_path,
        content_bundle_path=plan.plan.content_bundle_path,
        inventory_path=plan.plan.inventory_path,
        junit_xml_path=str(paths["junit"].resolve()),
        raw_terminal_path=str(paths["raw"].resolve()),
        native_lifecycle_path=str(paths["lifecycle"].resolve()),
        reset_state_evidence_path=str(paths["reset"].resolve()),
        hbm_evidence_path=str(paths["hbm"].resolve()),
    )
    qualification_binding = _publish_evidence(paths["spec"], qualification.to_dict())
    authority_binding = publish_trusted_empirical_tp1_session_reset_authority(
        qualification_spec_path=qualification_binding.absolute_path,
        output_path=paths["authority"],
    )
    rebound, _authority = revalidate_trusted_empirical_tp1_session_reset_authority(
        authority_binding.absolute_path
    )
    if rebound != authority_binding:
        raise RuntimeError("trusted empirical reset authority changed after publish")
    return TrustedEmpiricalTp1SessionResetPhysicalResult(
        schema_version=1,
        kind="trusted_empirical_tp1_session_reset_physical_result",
        protocol_sha256=(
            TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_RESULT_PROTOCOL_SHA256
        ),
        status="PASS",
        evidence_level="trusted_single_operator_empirical_no_signature",
        formal_measured=False,
        qualification_run_id=plan.qualification_run_id,
        plan=plan.binding,
        raw_terminal=raw_binding,
        junit_xml=junit_binding,
        native_lifecycle=lifecycle_binding,
        reset_state_evidence=reset_binding,
        hbm_evidence=hbm_binding,
        qualification_spec=qualification_binding,
        authority=authority_binding,
        failure_terminal=None,
    )


__all__ = (
    "TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_PLAN_PROTOCOL_SHA256",
    "TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PHYSICAL_RESULT_PROTOCOL_SHA256",
    "RevalidatedTrustedEmpiricalTp1SessionResetPhysicalPlan",
    "TrustedEmpiricalTp1SessionResetLiveResources",
    "TrustedEmpiricalTp1SessionResetPhysicalPlan",
    "TrustedEmpiricalTp1SessionResetPhysicalResult",
    "TrustedEmpiricalTp1SessionResetPhysicalRuntime",
    "execute_trusted_empirical_tp1_session_reset_qualification",
    "publish_trusted_empirical_tp1_session_reset_physical_plan",
    "revalidate_trusted_empirical_tp1_session_reset_physical_plan",
)
