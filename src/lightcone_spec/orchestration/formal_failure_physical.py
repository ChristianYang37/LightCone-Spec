"""Integrated, callback-free physical runner for formal E5 failure cells.

The patched SGLang child is spawned once, its live rank capability is converted
to an exact actuator launch binding, all five failure phases execute while that
same process group remains alive, and cleanup/lifecycle evidence is published
only after the process group is empty.  The later local trust lift controls the
whole lifecycle receipt rather than reusing an unrelated serving timing proof.
"""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import time
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self
from xml.sax.saxutils import quoteattr

from lightcone_spec.config import load_run_config
from lightcone_spec.experiments.failure_actuator import (
    FAILURE_ACTUATOR_PROTOCOL_SHA256,
    FailureRankLaunchBinding,
    SglangHttpFailureNativeControlTransport,
    release_failure_actuator_capability,
)
from lightcone_spec.experiments.formal_failure_actuator import (
    FormalFailureActuationReceipt,
    FormalFailureActuatorLaunchBinding,
    _reopen_raw,
    execute_formal_failure_actuator_unsigned,
)
from lightcone_spec.experiments.formal_failure_execution import (
    FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
    FormalFailureExecutionSubject,
    FormalSingleOperatorE5FailureExecutionDescriptor,
    VerifiedFormalFailureExecutionBinding,
    require_verified_formal_failure_execution_binding,
    revalidate_formal_single_operator_e5_failure_execution_descriptor,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FormalServingExecutionBinding,
    FormalSingleOperatorExecutionBinding,
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.experiments.serving import PinnedBenchServingTransport
from lightcone_spec.orchestration.formal_physical_dispatch import (
    FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
    FormalPhysicalDispatchError,
    FormalServingRunPlan,
    _capture_gpu_process_snapshot,
    _load_formal_single_operator_trusted_run_plan,
    _observe_live_server_execution_policy,
    _process_group_exists_for_formal_dispatch,
    _publish_gpu_snapshot_error,
    _require_port_unused,
    _terminate_process_group,
    _wait_server_ready,
    load_formal_serving_run_plan,
    rebuild_formal_single_operator_execution_binding_from_plan,
    revalidate_formal_serving_run_plan,
)
from lightcone_spec.orchestration.live_sglang import (
    PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
    PinnedNvidiaSmiTool,
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

FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "formal_e5_integrated_failure_physical_protocol",
        "lifecycle": (
            "spawn_ready_snapshot_live_capability_five_failure_phases_"
            "cleanup_empty_process_group_after_snapshot_flush"
        ),
        "scope": "one_sealed_failure_binding_one_serving_plan_one_run_nonce",
        "caller_injection": False,
        "publication": "unsigned_raw_then_local_external_control",
        "operator_artifacts": (
            "separate_stdout_stderr_source_event_log_deterministic_junit_exit_code"
        ),
    }
)
FORMAL_E5_FAILURE_LIFECYCLE_CONTROL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_e5_integrated_failure_lifecycle_external_control",
        "physical_protocol_sha256": FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256,
        "raw_authority": (
            "plan_failure_binding_rank_launch_raw_recovery_gpu_snapshots_"
            "process_group_cleanup_monotonic_lifecycle"
        ),
        "replay": "deployment_control_and_failure_run_nonce",
    }
)
_FAILURE_ACTUATOR_HOOK = "sglang.lightcone_e5_failure_actuator.v1"

FormalFailurePhysicalExecutionBinding = (
    VerifiedFormalFailureExecutionBinding
    | FormalSingleOperatorE5FailureExecutionDescriptor
)


def _physical_failure_binding_sha256(
    failure: FormalFailurePhysicalExecutionBinding,
) -> str:
    if type(failure) is FormalSingleOperatorE5FailureExecutionDescriptor:
        return failure.expected_failure_execution_binding_sha256
    return failure.sha256


def _physical_failure_gpu_uuids(
    failure: FormalFailurePhysicalExecutionBinding,
) -> tuple[str, ...]:
    if type(failure) is FormalSingleOperatorE5FailureExecutionDescriptor:
        return failure.gpu_uuids
    return failure.serving_execution.subject.gpu_uuids


def _strict_object(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _sha(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _private_new_directory(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path.exists()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise ValueError("formal failure quota root must be one new resolved directory")
    path.mkdir(mode=0o700)
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("formal failure quota root is not private")
    return path


@dataclass(frozen=True)
class FormalE5FailureLifecycleRawReceipt:
    schema_version: Literal[3]
    kind: Literal["formal_e5_failure_lifecycle_raw_receipt"]
    protocol_sha256: str
    formal_execution_authorized: Literal[False]
    plan: CanonicalJsonProofBinding
    formal_launch_admission: CanonicalJsonProofBinding
    formal_launch_consumption: CanonicalJsonProofBinding
    budget_consumption: CanonicalJsonProofBinding
    formal_failure_execution_binding_sha256: str
    raw_failure_terminal: CanonicalJsonProofBinding
    launch_binding_sha256: str
    inventory_sha256: str
    registry_sha256: str
    run_nonce_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    gpu_uuids: tuple[str, ...]
    before_gpu_snapshot: CanonicalJsonProofBinding
    ready_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    server_log: EvidenceFileBinding
    server_stdout: EvidenceFileBinding
    server_stderr: EvidenceFileBinding
    junit: EvidenceFileBinding
    server_process_group_id: int
    process_exit_code: int
    process_group_empty: Literal[True]
    phase_edges_ns: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 3
            or self.kind != "formal_e5_failure_lifecycle_raw_receipt"
            or self.protocol_sha256 != FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256
            or self.formal_execution_authorized is not False
            or self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}
            or self.process_group_empty is not True
        ):
            raise ValueError("formal failure lifecycle receipt schema/status differs")
        for label, value in (
            ("failure binding", self.formal_failure_execution_binding_sha256),
            ("launch", self.launch_binding_sha256),
            ("inventory", self.inventory_sha256),
            ("registry", self.registry_sha256),
            ("run nonce", self.run_nonce_sha256),
        ):
            _sha(f"formal failure lifecycle {label}", value)
        if (
            type(self.plan) is not CanonicalJsonProofBinding
            or type(self.formal_launch_admission) is not CanonicalJsonProofBinding
            or type(self.formal_launch_consumption) is not CanonicalJsonProofBinding
            or type(self.budget_consumption) is not CanonicalJsonProofBinding
            or type(self.raw_failure_terminal) is not CanonicalJsonProofBinding
            or type(self.before_gpu_snapshot) is not CanonicalJsonProofBinding
            or type(self.ready_gpu_snapshot) is not CanonicalJsonProofBinding
            or type(self.after_gpu_snapshot) is not CanonicalJsonProofBinding
            or type(self.server_log) is not EvidenceFileBinding
            or type(self.server_stdout) is not EvidenceFileBinding
            or type(self.server_stderr) is not EvidenceFileBinding
            or type(self.junit) is not EvidenceFileBinding
            or type(self.server_process_group_id) is not int
            or self.server_process_group_id < 1
            or type(self.process_exit_code) is not int
        ):
            raise TypeError("formal failure lifecycle path/process binding differs")
        expected_gpus = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpus
            or len(set(self.gpu_uuids)) != expected_gpus
        ):
            raise ValueError("formal failure lifecycle GPU coverage differs")
        expected_edges = (
            "execution_started_ns",
            "server_spawned_ns",
            "server_ready_ns",
            "failure_started_ns",
            "failure_finished_ns",
            "process_exited_ns",
            "after_snapshot_ns",
            "process_group_empty_checked_ns",
            "gpu_release_ns",
            "evidence_flush_started_ns",
            "evidence_flush_finished_ns",
        )
        if (
            type(self.phase_edges_ns) is not tuple
            or tuple(name for name, _value in self.phase_edges_ns) != expected_edges
            or any(
                type(value) is not int or value < 1
                for _name, value in self.phase_edges_ns
            )
        ):
            raise ValueError("formal failure lifecycle phase coverage differs")
        values = tuple(value for _name, value in self.phase_edges_ns)
        if any(right < left for left, right in pairwise(values)):
            raise ValueError("formal failure lifecycle phases are not monotonic")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "formal_execution_authorized": self.formal_execution_authorized,
            "plan": self.plan.to_dict(),
            "formal_launch_admission": self.formal_launch_admission.to_dict(),
            "formal_launch_consumption": self.formal_launch_consumption.to_dict(),
            "budget_consumption": self.budget_consumption.to_dict(),
            "formal_failure_execution_binding_sha256": (
                self.formal_failure_execution_binding_sha256
            ),
            "raw_failure_terminal": self.raw_failure_terminal.to_dict(),
            "launch_binding_sha256": self.launch_binding_sha256,
            "inventory_sha256": self.inventory_sha256,
            "registry_sha256": self.registry_sha256,
            "run_nonce_sha256": self.run_nonce_sha256,
            "topology_mode": self.topology_mode,
            "gpu_uuids": list(self.gpu_uuids),
            "before_gpu_snapshot": self.before_gpu_snapshot.to_dict(),
            "ready_gpu_snapshot": self.ready_gpu_snapshot.to_dict(),
            "after_gpu_snapshot": self.after_gpu_snapshot.to_dict(),
            "server_log": self.server_log.to_dict(),
            "server_stdout": self.server_stdout.to_dict(),
            "server_stderr": self.server_stderr.to_dict(),
            "junit": self.junit.to_dict(),
            "server_process_group_id": self.server_process_group_id,
            "process_exit_code": self.process_exit_code,
            "process_group_empty": self.process_group_empty,
            "phase_edges_ns": [list(row) for row in self.phase_edges_ns],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "formal failure lifecycle receipt",
            value,
            set(cls.__dataclass_fields__),
        )
        gpu_uuids = row.pop("gpu_uuids")
        phase_edges = row.pop("phase_edges_ns")
        if (
            type(gpu_uuids) is not list
            or type(phase_edges) is not list
            or any(type(item) is not list or len(item) != 2 for item in phase_edges)
        ):
            raise TypeError("formal failure lifecycle collections differ")
        plan = CanonicalJsonProofBinding.from_dict(row.pop("plan"))
        launch_admission = CanonicalJsonProofBinding.from_dict(
            row.pop("formal_launch_admission")
        )
        launch_consumption = CanonicalJsonProofBinding.from_dict(
            row.pop("formal_launch_consumption")
        )
        budget_consumption = CanonicalJsonProofBinding.from_dict(
            row.pop("budget_consumption")
        )
        terminal = CanonicalJsonProofBinding.from_dict(row.pop("raw_failure_terminal"))
        before = CanonicalJsonProofBinding.from_dict(row.pop("before_gpu_snapshot"))
        ready = CanonicalJsonProofBinding.from_dict(row.pop("ready_gpu_snapshot"))
        after = CanonicalJsonProofBinding.from_dict(row.pop("after_gpu_snapshot"))
        log = EvidenceFileBinding.from_dict(
            row.pop("server_log"), label="formal failure server log"
        )
        stdout = EvidenceFileBinding.from_dict(
            row.pop("server_stdout"), label="formal failure server stdout"
        )
        stderr = EvidenceFileBinding.from_dict(
            row.pop("server_stderr"), label="formal failure server stderr"
        )
        junit = EvidenceFileBinding.from_dict(
            row.pop("junit"), label="formal failure JUnit"
        )
        return cls(
            **row,  # type: ignore[arg-type]
            plan=plan,
            formal_launch_admission=launch_admission,
            formal_launch_consumption=launch_consumption,
            budget_consumption=budget_consumption,
            raw_failure_terminal=terminal,
            gpu_uuids=tuple(gpu_uuids),
            before_gpu_snapshot=before,
            ready_gpu_snapshot=ready,
            after_gpu_snapshot=after,
            server_log=log,
            server_stdout=stdout,
            server_stderr=stderr,
            junit=junit,
            phase_edges_ns=tuple((item[0], item[1]) for item in phase_edges),
        )


@dataclass(frozen=True)
class ValidatedUnsignedFormalE5FailureRun:
    lifecycle_receipt: CanonicalJsonProofBinding
    raw_failure_terminal: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            type(self.lifecycle_receipt) is not CanonicalJsonProofBinding
            or type(self.raw_failure_terminal) is not CanonicalJsonProofBinding
        ):
            raise TypeError("formal failure physical result is not path-bound")


@dataclass(frozen=True)
class FormalSingleOperatorE5PhysicalOutcome:
    """Public COMPLETE projection for trusted single-operator provenance."""

    status: Literal["COMPLETE"]
    plan: CanonicalJsonProofBinding
    run_receipt: CanonicalJsonProofBinding
    lifecycle_receipt: CanonicalJsonProofBinding
    raw_failure_terminal: CanonicalJsonProofBinding
    formal_launch_admission: CanonicalJsonProofBinding
    formal_launch_consumption: CanonicalJsonProofBinding
    before_gpu_snapshot: CanonicalJsonProofBinding
    ready_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    server_log: EvidenceFileBinding
    server_stdout: EvidenceFileBinding
    server_stderr: EvidenceFileBinding
    junit: EvidenceFileBinding
    started_ns: int
    finished_ns: int
    process_exit_code: int
    process_group_empty: Literal[True]

    def __post_init__(self) -> None:
        if self.status != "COMPLETE" or self.process_group_empty is not True:
            raise ValueError("single-operator E5 physical outcome is not complete")
        for value in (
            self.plan,
            self.run_receipt,
            self.lifecycle_receipt,
            self.raw_failure_terminal,
            self.formal_launch_admission,
            self.formal_launch_consumption,
            self.before_gpu_snapshot,
            self.ready_gpu_snapshot,
            self.after_gpu_snapshot,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("single-operator E5 outcome is not path-bound")
        for value in (
            self.server_log,
            self.server_stdout,
            self.server_stderr,
            self.junit,
        ):
            if type(value) is not EvidenceFileBinding:
                raise TypeError("single-operator E5 file outcome is not path-bound")
        if (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 1
            or self.finished_ns < self.started_ns
            or type(self.process_exit_code) is not int
            or self.process_exit_code not in {0, -15}
        ):
            raise ValueError("single-operator E5 outcome timing/exit differs")


def validate_formal_single_operator_e5_physical_outcome(
    *,
    plan_path: str | Path,
    run_receipt_path: str | Path,
    lifecycle_receipt_path: str | Path,
    execution_binding: FormalServingExecutionBinding | None = None,
) -> FormalSingleOperatorE5PhysicalOutcome:
    """Deep-open a successful E5 run without a private failure token.

    This projection is intentionally limited to the trusted single-operator
    admission branch.  It validates the source-owned plan, admission and
    one-shot consumption, raw failure receipt, process/GPU lifecycle, separate
    stdout/stderr, and deterministic JUnit.  It never accepts caller timing,
    exit status, failure scenario, or artifact digests.
    """

    from lightcone_spec.experiments.formal_failure_actuator import (
        FormalFailureActuationReceipt,
        FormalFailureActuatorLaunchBinding,
    )
    from lightcone_spec.orchestration.formal_single_operator_admission import (
        validate_formal_single_operator_admission,
        validate_formal_single_operator_admission_consumption,
    )

    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    preliminary = FormalServingRunPlan.from_dict(plan_binding.reopen())
    source_kind = (
        None
        if preliminary.single_operator_execution_rebuild_source is None
        else preliminary.single_operator_execution_rebuild_source.reopen().get("kind")
    )
    current_failure = None
    if source_kind == "formal_single_operator_e5_failure_execution_descriptor":
        if execution_binding is not None:
            raise ValueError("current E5 outcome cannot carry a private token")
        assert preliminary.single_operator_execution_rebuild_source is not None
        current_failure = (
            revalidate_formal_single_operator_e5_failure_execution_descriptor(
                preliminary.single_operator_execution_rebuild_source.absolute_path,
                current_ns=time.time_ns(),
            )
        )
        plan, _launch_manifest, _schedule = (
            _load_formal_single_operator_trusted_run_plan(plan_path)
        )
        serving = None
        expected_registry_sha256 = current_failure.failure_subject.registry_sha256
        expected_execution_binding_sha256 = plan.execution_binding_sha256
    else:
        serving = (
            rebuild_formal_single_operator_execution_binding_from_plan(plan_path)
            if execution_binding is None
            else require_verified_formal_serving_execution_binding(execution_binding)
        )
        plan = load_formal_serving_run_plan(
            plan_binding.absolute_path,
            execution_binding=serving,
            verified_nextn_tp2_authority=serving.verified_nextn_tp2_authority,
        )
        expected_registry_sha256 = serving.subject.execution_identity.registry_sha256
        expected_execution_binding_sha256 = serving.sha256
    if plan.stage != "E5" or plan_binding.semantic_sha256 != plan.sha256:
        raise ValueError("single-operator E5 outcome plan differs")

    lifecycle_binding = CanonicalJsonProofBinding.bind(lifecycle_receipt_path)
    lifecycle = FormalE5FailureLifecycleRawReceipt.from_dict(lifecycle_binding.reopen())
    if lifecycle_binding.semantic_sha256 != lifecycle.sha256:
        raise ValueError("single-operator E5 lifecycle identity differs")

    run_binding = CanonicalJsonProofBinding.bind(run_receipt_path)
    run = _strict_object(
        "single-operator E5 run receipt",
        run_binding.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "plan_sha256",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "lifecycle_receipt",
            "raw_failure_terminal",
            "server_log",
            "server_stdout",
            "server_stderr",
            "junit",
            "process_exit_code",
            "process_group_empty",
        },
    )
    run_admission = CanonicalJsonProofBinding.from_dict(run["formal_launch_admission"])
    run_consumption = CanonicalJsonProofBinding.from_dict(
        run["formal_launch_consumption"]
    )
    run_budget = CanonicalJsonProofBinding.from_dict(run["budget_consumption"])
    run_lifecycle = CanonicalJsonProofBinding.from_dict(run["lifecycle_receipt"])
    run_terminal = CanonicalJsonProofBinding.from_dict(run["raw_failure_terminal"])
    run_log = EvidenceFileBinding.from_dict(
        run["server_log"], label="single-operator E5 run log"
    )
    run_stdout = EvidenceFileBinding.from_dict(
        run["server_stdout"], label="single-operator E5 run stdout"
    )
    run_stderr = EvidenceFileBinding.from_dict(
        run["server_stderr"], label="single-operator E5 run stderr"
    )
    run_junit = EvidenceFileBinding.from_dict(
        run["junit"], label="single-operator E5 run JUnit"
    )
    if (
        run["schema_version"] != 1
        or run["kind"] != "unsigned_formal_e5_failure_physical_run_receipt"
        or run["protocol_sha256"] != FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        or run["formal_execution_authorized"] is not False
        or run["plan_sha256"] != plan.sha256
        or run["process_group_empty"] is not True
        or run["process_exit_code"] != lifecycle.process_exit_code
        or run_lifecycle != lifecycle_binding
        or run_admission != lifecycle.formal_launch_admission
        or run_consumption != lifecycle.formal_launch_consumption
        or run_budget != lifecycle.budget_consumption
        or run_terminal != lifecycle.raw_failure_terminal
        or run_log != lifecycle.server_log
        or run_stdout != lifecycle.server_stdout
        or run_stderr != lifecycle.server_stderr
        or run_junit != lifecycle.junit
    ):
        raise ValueError("single-operator E5 run receipt differs from lifecycle")

    admission_value = lifecycle.formal_launch_admission.reopen()
    if admission_value.get("kind") != "formal_single_operator_admission":
        raise ValueError("public E5 outcome requires single-operator admission")
    admission = validate_formal_single_operator_admission(
        lifecycle.formal_launch_admission.absolute_path,
        plan_path=plan_binding.absolute_path,
    )
    validate_formal_single_operator_admission_consumption(
        lifecycle.formal_launch_consumption.absolute_path,
        admission_path=lifecycle.formal_launch_admission.absolute_path,
        plan_path=plan_binding.absolute_path,
    )
    if (
        lifecycle.budget_consumption != lifecycle.formal_launch_consumption
        or lifecycle.plan != plan_binding
        or admission.stage != "E5"
        or admission.execution_binding_sha256 != expected_execution_binding_sha256
        or admission.materialized_cell_id != plan.materialized_cell_id
        or admission.inventory_sha256 != plan.inventory_sha256
        or admission.registry_sha256 != expected_registry_sha256
        or admission.topology_mode != plan.topology_mode
        or admission.gpu_uuids != plan.gpu_uuids
        or lifecycle.inventory_sha256 != plan.inventory_sha256
        or lifecycle.registry_sha256 != admission.registry_sha256
        or lifecycle.run_nonce_sha256 != plan.native_terminal_binding.run_nonce_sha256
        or lifecycle.topology_mode != plan.topology_mode
        or lifecycle.gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("single-operator E5 admission/lifecycle differs from plan")

    raw = _strict_object(
        "single-operator E5 raw failure terminal",
        lifecycle.raw_failure_terminal.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "source_capability_sha256",
            "formal_failure_execution_binding_sha256",
            "launch_binding",
            "launch_binding_sha256",
            "recovery_receipt",
            "recovery_receipt_sha256",
            "formal_execution_authorized",
        },
    )
    launch = FormalFailureActuatorLaunchBinding.from_dict(raw["launch_binding"])
    recovery = FormalFailureActuationReceipt.from_dict(raw["recovery_receipt"])
    if (
        raw["schema_version"] != 1
        or raw["kind"] != "e5_unsigned_formal_failure_recovery_terminal"
        or raw["protocol_sha256"] != FAILURE_ACTUATOR_PROTOCOL_SHA256
        or raw["source_capability_sha256"]
        != release_failure_actuator_capability().sha256
        or raw["formal_execution_authorized"] is not False
        or raw["formal_failure_execution_binding_sha256"]
        != lifecycle.formal_failure_execution_binding_sha256
        or raw["launch_binding_sha256"] != launch.sha256
        or raw["launch_binding_sha256"] != lifecycle.launch_binding_sha256
        or raw["recovery_receipt_sha256"] != recovery.sha256
        or launch.formal_failure_execution_binding_sha256
        != lifecycle.formal_failure_execution_binding_sha256
        or recovery.formal_failure_execution_binding_sha256
        != lifecycle.formal_failure_execution_binding_sha256
        or recovery.materialized_cell_id != plan.materialized_cell_id
        or recovery.inventory_sha256 != plan.inventory_sha256
        or recovery.registry_sha256 != admission.registry_sha256
        or recovery.serving_execution_plan_sha256
        != plan.native_terminal_binding.execution_plan_sha256
        or recovery.run_nonce_sha256 != plan.native_terminal_binding.run_nonce_sha256
        or recovery.topology != plan.topology_mode
        or recovery.launch_binding_sha256 != launch.sha256
        or tuple(row.gpu_uuid for row in launch.ranks) != plan.gpu_uuids
        or any(
            row.process_group_id != lifecycle.server_process_group_id
            for row in launch.ranks
        )
        or recovery.correctness_only is not True
    ):
        raise ValueError("single-operator E5 raw recovery differs from plan")
    if current_failure is not None:
        _reopen_failure_raw_against_subject(
            lifecycle.raw_failure_terminal,
            subject=current_failure.failure_subject,
            failure_execution_binding_sha256=(
                current_failure.expected_failure_execution_binding_sha256
            ),
            gpu_uuids=current_failure.gpu_uuids,
            process_group_id=lifecycle.server_process_group_id,
        )

    for snapshot, phase in (
        (lifecycle.before_gpu_snapshot, "before"),
        (lifecycle.ready_gpu_snapshot, "ready"),
        (lifecycle.after_gpu_snapshot, "after"),
    ):
        _validate_snapshot(
            snapshot,
            phase=phase,
            inventory_sha256=plan.inventory_sha256,
            gpu_uuids=plan.gpu_uuids,
        )
    for file_binding, label in (
        (lifecycle.server_log, "single-operator E5 server log"),
        (lifecycle.server_stdout, "single-operator E5 server stdout"),
        (lifecycle.server_stderr, "single-operator E5 server stderr"),
        (lifecycle.junit, "single-operator E5 JUnit"),
    ):
        file_binding.reopen(label=label)
    edges = dict(lifecycle.phase_edges_ns)
    return FormalSingleOperatorE5PhysicalOutcome(
        status="COMPLETE",
        plan=plan_binding,
        run_receipt=run_binding,
        lifecycle_receipt=lifecycle_binding,
        raw_failure_terminal=lifecycle.raw_failure_terminal,
        formal_launch_admission=lifecycle.formal_launch_admission,
        formal_launch_consumption=lifecycle.formal_launch_consumption,
        before_gpu_snapshot=lifecycle.before_gpu_snapshot,
        ready_gpu_snapshot=lifecycle.ready_gpu_snapshot,
        after_gpu_snapshot=lifecycle.after_gpu_snapshot,
        server_log=lifecycle.server_log,
        server_stdout=lifecycle.server_stdout,
        server_stderr=lifecycle.server_stderr,
        junit=lifecycle.junit,
        started_ns=edges["execution_started_ns"],
        finished_ns=edges["evidence_flush_finished_ns"],
        process_exit_code=lifecycle.process_exit_code,
        process_group_empty=True,
    )


def _spawn_failure_server(
    launch: CompileLaunchManifest,
    *,
    failure: FormalFailurePhysicalExecutionBinding,
    quota_root: Path,
    stdout_file,
    stderr_file,
) -> subprocess.Popen[bytes]:
    subject = failure.subject
    environment = launch.child_environment()
    environment.update(
        {
            "LIGHTCONE_FAILURE_ACTUATOR_ENABLE": "1",
            "LIGHTCONE_FAILURE_ACTUATOR_ASSIGNMENT_SHA256": (subject.assignment_sha256),
            "LIGHTCONE_FAILURE_ACTUATOR_INVENTORY_SHA256": (subject.inventory_sha256),
            "LIGHTCONE_FAILURE_ACTUATOR_PLAN_SHA256": (
                subject.serving_execution_plan_sha256
            ),
            "LIGHTCONE_FAILURE_ACTUATOR_RUN_NONCE_SHA256": (subject.run_nonce_sha256),
            "LIGHTCONE_FAILURE_ACTUATOR_QUOTA_ROOT_BASE": str(quota_root),
        }
    )
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


def _publish_failure_junit(
    *,
    output_path: str | Path,
    materialized_cell_id: str,
    scenario: str,
) -> EvidenceFileBinding:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<testsuite name="lightcone-formal-e5-failure" tests="1" '
        'failures="0" errors="0" skipped="0">'
        f"<testcase classname={quoteattr('lightcone.E5.failure')} "
        f"name={quoteattr(materialized_cell_id + ':' + scenario)}/>"
        "</testsuite>\n"
    ).encode()
    path = Path(output_path)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return EvidenceFileBinding.bind(path, label="formal failure JUnit")


def _launch_from_capability(
    value: object,
    *,
    failure: FormalFailurePhysicalExecutionBinding,
    port: int,
    process_group_id: int,
) -> FormalFailureActuatorLaunchBinding:
    subject = failure.subject
    row = _strict_object(
        "formal failure capability",
        value,
        {"schema_version", "hook", "topology", "world_size", "rank_bindings"},
    )
    ranks = row["rank_bindings"]
    world_size = 1 if subject.topology == "tp1_dp1" else 2
    if (
        row["schema_version"] != 1
        or row["hook"] != _FAILURE_ACTUATOR_HOOK
        or row["topology"] != subject.topology
        or row["world_size"] != world_size
        or type(ranks) is not list
        or len(ranks) != world_size
    ):
        raise ValueError("formal failure capability identity differs")
    expected_fields = {
        "schema_version",
        "hook",
        "protocol_sha256",
        "assignment_sha256",
        "inventory_sha256",
        "plan_sha256",
        "run_nonce_sha256",
        "topology",
        "rank",
        "world_size",
        "process_id",
        "process_group_id",
        "process_start_monotonic_ns",
        "gpu_uuid",
        "temp_quota_root",
    }
    rows: list[FailureRankLaunchBinding] = []
    expected_gpus = _physical_failure_gpu_uuids(failure)
    for rank, item in enumerate(ranks):
        capability = _strict_object(
            "formal failure rank capability", item, expected_fields
        )
        if (
            capability["schema_version"] != 1
            or capability["hook"] != _FAILURE_ACTUATOR_HOOK
            or capability["protocol_sha256"] != FAILURE_ACTUATOR_PROTOCOL_SHA256
            or capability["assignment_sha256"] != subject.assignment_sha256
            or capability["inventory_sha256"] != subject.inventory_sha256
            or capability["plan_sha256"] != subject.serving_execution_plan_sha256
            or capability["run_nonce_sha256"] != subject.run_nonce_sha256
            or capability["topology"] != subject.topology
            or capability["rank"] != rank
            or capability["world_size"] != world_size
            or capability["gpu_uuid"] != expected_gpus[rank]
            or capability["process_group_id"] != process_group_id
        ):
            raise ValueError("formal failure rank capability differs")
        rows.append(
            FailureRankLaunchBinding(
                topology=subject.topology,  # type: ignore[arg-type]
                rank=rank,
                process_id=capability["process_id"],  # type: ignore[arg-type]
                process_group_id=capability["process_group_id"],  # type: ignore[arg-type]
                process_start_monotonic_ns=capability["process_start_monotonic_ns"],  # type: ignore[arg-type]
                gpu_uuid=capability["gpu_uuid"],  # type: ignore[arg-type]
                control_url=(
                    f"http://127.0.0.1:{port}/v1/lightcone-spec/failure-actuator"
                ),
                temp_quota_root=capability["temp_quota_root"],  # type: ignore[arg-type]
            )
        )
    return FormalFailureActuatorLaunchBinding(
        schema_version=1,
        kind="formal_e5_failure_actuator_launch_binding",
        formal_failure_execution_binding_sha256=(
            _physical_failure_binding_sha256(failure)
        ),
        assignment_sha256=subject.assignment_sha256,
        inventory_sha256=subject.inventory_sha256,
        registry_sha256=subject.registry_sha256,
        serving_execution_plan_sha256=subject.serving_execution_plan_sha256,
        run_nonce_sha256=subject.run_nonce_sha256,
        topology=subject.topology,  # type: ignore[arg-type]
        ranks=tuple(rows),
    )


def _validate_snapshot(
    binding: CanonicalJsonProofBinding,
    *,
    phase: str,
    inventory_sha256: str,
    gpu_uuids: tuple[str, ...],
) -> None:
    value = binding.reopen()
    if (
        type(value) is not dict
        or value.get("phase") != phase
        or value.get("inventory_sha256") != inventory_sha256
        or tuple(value.get("gpu_uuids", ())) != gpu_uuids
    ):
        raise ValueError("formal failure GPU snapshot differs")


def _failure_execution_binding_sha256(
    subject: FormalFailureExecutionSubject,
) -> str:
    subject.__post_init__()
    return content_sha256(
        {
            "protocol_sha256": FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
            "subject_sha256": subject.sha256,
            "serving_execution_binding_sha256": (
                subject.serving_execution_binding_sha256
            ),
        }
    )


def _reopen_failure_raw_against_subject(
    raw: CanonicalJsonProofBinding,
    *,
    subject: FormalFailureExecutionSubject,
    failure_execution_binding_sha256: str,
    gpu_uuids: tuple[str, ...],
    process_group_id: int,
) -> tuple[FormalFailureActuatorLaunchBinding, FormalFailureActuationReceipt]:
    value = _strict_object(
        "formal failure raw terminal",
        raw.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "source_capability_sha256",
            "formal_failure_execution_binding_sha256",
            "launch_binding",
            "launch_binding_sha256",
            "recovery_receipt",
            "recovery_receipt_sha256",
            "formal_execution_authorized",
        },
    )
    launch = FormalFailureActuatorLaunchBinding.from_dict(value["launch_binding"])
    receipt = FormalFailureActuationReceipt.from_dict(value["recovery_receipt"])
    if (
        value["schema_version"] != 1
        or value["kind"] != "e5_unsigned_formal_failure_recovery_terminal"
        or value["protocol_sha256"] != FAILURE_ACTUATOR_PROTOCOL_SHA256
        or value["source_capability_sha256"]
        != release_failure_actuator_capability().sha256
        or value["formal_failure_execution_binding_sha256"]
        != failure_execution_binding_sha256
        or value["launch_binding_sha256"] != launch.sha256
        or value["recovery_receipt_sha256"] != receipt.sha256
        or value["formal_execution_authorized"] is not False
        or launch.formal_failure_execution_binding_sha256
        != failure_execution_binding_sha256
        or launch.assignment_sha256 != subject.assignment_sha256
        or launch.inventory_sha256 != subject.inventory_sha256
        or launch.registry_sha256 != subject.registry_sha256
        or launch.serving_execution_plan_sha256 != subject.serving_execution_plan_sha256
        or launch.run_nonce_sha256 != subject.run_nonce_sha256
        or launch.topology != subject.topology
        or tuple(row.gpu_uuid for row in launch.ranks) != gpu_uuids
        or any(row.process_group_id != process_group_id for row in launch.ranks)
        or receipt.formal_failure_execution_binding_sha256
        != failure_execution_binding_sha256
        or receipt.assignment_sha256 != subject.assignment_sha256
        or receipt.inventory_sha256 != subject.inventory_sha256
        or receipt.registry_sha256 != subject.registry_sha256
        or receipt.serving_execution_plan_sha256
        != subject.serving_execution_plan_sha256
        or receipt.materialized_cell_id != subject.materialized_cell_id
        or receipt.scenario != subject.scenario
        or receipt.run_nonce_sha256 != subject.run_nonce_sha256
        or receipt.topology != subject.topology
        or receipt.launch_binding_sha256 != launch.sha256
        or receipt.correctness_only is not True
        or tuple(row.process_id for row in receipt.rank_receipts)
        != tuple(row.process_id for row in launch.ranks)
        or tuple(row.process_start_monotonic_ns for row in receipt.rank_receipts)
        != tuple(row.process_start_monotonic_ns for row in launch.ranks)
    ):
        raise ValueError("formal failure raw terminal differs from durable subject")
    return launch, receipt


def _validate_cost_snapshot(
    binding: CanonicalJsonProofBinding,
    *,
    phase: Literal["before", "ready", "after"],
    inventory_sha256: str,
    gpu_uuids: tuple[str, ...],
    process_group_id: int,
) -> int:
    value = _strict_object(
        f"formal failure {phase} GPU snapshot",
        binding.reopen(),
        {
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
        },
    )
    processes = value["compute_process_rows"]
    expected_groups = [process_group_id] * len(gpu_uuids) if phase == "ready" else None
    if (
        value["schema_version"] != 1
        or value["kind"] != "unsigned_pinned_sglang_gpu_process_snapshot"
        or value["protocol_sha256"] != PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256
        or value["phase"] != phase
        or value["inventory_sha256"] != inventory_sha256
        or value["gpu_uuids"] != list(gpu_uuids)
        or value["server_process_group_ids"] != expected_groups
        or type(value["captured_ns"]) is not int
        or value["captured_ns"] < 1
        or type(processes) is not list
        or any(type(row) is not dict for row in processes)
    ):
        raise ValueError("formal failure GPU snapshot identity differs")
    if phase in {"before", "after"}:
        if processes:
            raise ValueError("formal failure clean GPU snapshot contains a process")
    elif {row.get("gpu_uuid") for row in processes} != set(gpu_uuids) or any(
        row.get("process_group_id") != process_group_id for row in processes
    ):
        raise ValueError("formal failure ready GPU process ownership differs")
    return value["captured_ns"]


def validate_formal_e5_failure_lifecycle_raw_receipt(
    lifecycle_receipt_path: str | Path,
    *,
    plan_path: str | Path,
    execution_binding: VerifiedFormalServingExecutionBinding,
    failure_execution_binding: VerifiedFormalFailureExecutionBinding,
) -> FormalE5FailureLifecycleRawReceipt:
    serving = require_verified_formal_serving_execution_binding(execution_binding)
    failure = require_verified_formal_failure_execution_binding(
        failure_execution_binding
    )
    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = load_formal_serving_run_plan(
        plan_path,
        execution_binding=serving,
        verified_nextn_tp2_authority=serving.verified_nextn_tp2_authority,
    )
    binding = CanonicalJsonProofBinding.bind(lifecycle_receipt_path)
    receipt = FormalE5FailureLifecycleRawReceipt.from_dict(binding.reopen())
    if binding.semantic_sha256 != receipt.sha256:
        raise ValueError("formal failure lifecycle semantic identity differs")
    reopened = tuple(
        CanonicalJsonProofBinding.bind(item.absolute_path)
        for item in (
            receipt.plan,
            receipt.formal_launch_admission,
            receipt.formal_launch_consumption,
            receipt.budget_consumption,
            receipt.raw_failure_terminal,
            receipt.before_gpu_snapshot,
            receipt.ready_gpu_snapshot,
            receipt.after_gpu_snapshot,
        )
    )
    if reopened != (
        receipt.plan,
        receipt.formal_launch_admission,
        receipt.formal_launch_consumption,
        receipt.budget_consumption,
        receipt.raw_failure_terminal,
        receipt.before_gpu_snapshot,
        receipt.ready_gpu_snapshot,
        receipt.after_gpu_snapshot,
    ):
        raise ValueError("formal failure lifecycle path identity changed")
    receipt.server_log.reopen(label="formal failure server log")
    receipt.server_stdout.reopen(label="formal failure server stdout")
    receipt.server_stderr.reopen(label="formal failure server stderr")
    receipt.junit.reopen(label="formal failure JUnit")
    admission_value = receipt.formal_launch_admission.reopen()
    if admission_value.get("kind") == "formal_single_operator_admission":
        from lightcone_spec.orchestration.formal_single_operator_admission import (
            validate_formal_single_operator_admission,
            validate_formal_single_operator_admission_consumption,
        )

        single_admission = validate_formal_single_operator_admission(
            receipt.formal_launch_admission.absolute_path,
            plan_path=plan_path,
        )
        validate_formal_single_operator_admission_consumption(
            receipt.formal_launch_consumption.absolute_path,
            admission_path=receipt.formal_launch_admission.absolute_path,
            plan_path=plan_path,
        )
        if (
            receipt.budget_consumption != receipt.formal_launch_consumption
            or single_admission.stage != "E5"
            or single_admission.materialized_cell_id
            != failure.subject.materialized_cell_id
            or single_admission.inventory_sha256 != failure.subject.inventory_sha256
            or single_admission.registry_sha256 != failure.subject.registry_sha256
            or single_admission.topology_mode != failure.subject.topology
            or single_admission.gpu_uuids != serving.subject.gpu_uuids
        ):
            raise ValueError("single-operator failure launch evidence differs")
    else:
        from lightcone_spec.orchestration.formal_launch_admission import (
            validate_formal_stage_launch_evidence_lineage,
        )

        formal_admission = validate_formal_stage_launch_evidence_lineage(
            admission=receipt.formal_launch_admission,
            launch_consumption=receipt.formal_launch_consumption,
            budget_consumption=receipt.budget_consumption,
            run_plan_path=plan_path,
            current_ns=max(dict(receipt.phase_edges_ns).values()),
        )
        if (
            formal_admission.budget_mode != "registered_e5_one_shot"
            or formal_admission.failure_execution_binding_sha256 != failure.sha256
        ):
            raise ValueError("formal failure launch evidence is not one-shot authority")
    launch_binding, failure_receipt = _reopen_raw(
        receipt.raw_failure_terminal,
        binding=failure,
    )
    if type(failure_receipt) is not FormalFailureActuationReceipt:
        raise TypeError("formal failure lifecycle lost its recovery receipt")
    if (
        receipt.plan != plan_binding
        or plan.sha256 != plan_binding.semantic_sha256
        or receipt.formal_failure_execution_binding_sha256 != failure.sha256
        or failure.serving_execution.sha256 != serving.sha256
        or failure.subject.materialized_cell_id != plan.materialized_cell_id
        or failure.subject.serving_execution_plan_sha256
        != plan.native_terminal_binding.execution_plan_sha256
        or receipt.raw_failure_terminal.absolute_path != plan.terminal_output_path
        or receipt.launch_binding_sha256 != launch_binding.sha256
        or receipt.inventory_sha256 != failure.subject.inventory_sha256
        or receipt.registry_sha256 != failure.subject.registry_sha256
        or receipt.run_nonce_sha256 != failure.subject.run_nonce_sha256
        or receipt.topology_mode != failure.subject.topology
        or receipt.gpu_uuids != failure.serving_execution.subject.gpu_uuids
        or tuple(row.gpu_uuid for row in launch_binding.ranks) != receipt.gpu_uuids
        or any(
            row.process_group_id != receipt.server_process_group_id
            for row in launch_binding.ranks
        )
        or failure_receipt.correctness_only is not True
        or receipt.process_exit_code not in {0, -15}
    ):
        raise ValueError("formal failure lifecycle authority differs")
    _validate_snapshot(
        receipt.before_gpu_snapshot,
        phase="before",
        inventory_sha256=receipt.inventory_sha256,
        gpu_uuids=receipt.gpu_uuids,
    )
    _validate_snapshot(
        receipt.ready_gpu_snapshot,
        phase="ready",
        inventory_sha256=receipt.inventory_sha256,
        gpu_uuids=receipt.gpu_uuids,
    )
    _validate_snapshot(
        receipt.after_gpu_snapshot,
        phase="after",
        inventory_sha256=receipt.inventory_sha256,
        gpu_uuids=receipt.gpu_uuids,
    )
    return receipt


def _validate_formal_e5_failure_lifecycle_raw_cost_receipt(
    lifecycle_receipt_path: str | Path,
    *,
    failure_subject: FormalFailureExecutionSubject,
    expected_failure_execution_binding_sha256: str,
    current_ns: int,
) -> tuple[
    FormalE5FailureLifecycleRawReceipt,
    FormalServingRunPlan,
    FormalFailureActuationReceipt,
    object,
]:
    """Reopen the E5 raw DAG without reconstructing a private execution seal."""

    failure_subject.__post_init__()
    expected_binding = _failure_execution_binding_sha256(failure_subject)
    if expected_binding != expected_failure_execution_binding_sha256:
        raise ValueError("formal failure durable binding identity differs")
    lifecycle_binding = CanonicalJsonProofBinding.bind(lifecycle_receipt_path)
    receipt = FormalE5FailureLifecycleRawReceipt.from_dict(lifecycle_binding.reopen())
    if receipt.sha256 != lifecycle_binding.semantic_sha256:
        raise ValueError("formal failure lifecycle semantic identity differs")
    for item in (
        receipt.plan,
        receipt.formal_launch_admission,
        receipt.formal_launch_consumption,
        receipt.budget_consumption,
        receipt.raw_failure_terminal,
        receipt.before_gpu_snapshot,
        receipt.ready_gpu_snapshot,
        receipt.after_gpu_snapshot,
    ):
        if CanonicalJsonProofBinding.bind(item.absolute_path) != item:
            raise ValueError("formal failure lifecycle path identity changed")
    receipt.server_log.reopen(label="formal failure server log")
    plan = FormalServingRunPlan.from_dict(receipt.plan.reopen())
    if (
        plan.sha256 != receipt.plan.semantic_sha256
        or plan.stage != "E5"
        or plan.method != "l0"
        or plan.execution_binding_sha256
        != failure_subject.serving_execution_binding_sha256
        or plan.materialized_cell_id != failure_subject.materialized_cell_id
        or plan.inventory_sha256 != failure_subject.inventory_sha256
        or plan.topology_mode != failure_subject.topology
        or plan.native_terminal_binding.execution_plan_sha256
        != failure_subject.serving_execution_plan_sha256
        or plan.native_terminal_binding.rank_config_sha256
        != failure_subject.serving_rank_config_sha256
        or plan.native_terminal_binding.run_nonce_sha256
        != failure_subject.run_nonce_sha256
        or plan.lifecycle_timing_output_path != lifecycle_binding.absolute_path
        or receipt.formal_failure_execution_binding_sha256 != expected_binding
        or receipt.inventory_sha256 != failure_subject.inventory_sha256
        or receipt.registry_sha256 != failure_subject.registry_sha256
        or receipt.run_nonce_sha256 != failure_subject.run_nonce_sha256
        or receipt.topology_mode != failure_subject.topology
        or receipt.gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("formal failure lifecycle differs from durable subject")
    launch_manifest = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    if (
        launch_manifest.sha256 != plan.launch_manifest.semantic_sha256
        or launch_manifest.inventory_sha256 != plan.inventory_sha256
        or launch_manifest.gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("formal failure lifecycle launch manifest differs")
    from lightcone_spec.orchestration.formal_launch_admission import (
        validate_formal_stage_launch_evidence_lineage,
    )

    admission = validate_formal_stage_launch_evidence_lineage(
        admission=receipt.formal_launch_admission,
        launch_consumption=receipt.formal_launch_consumption,
        budget_consumption=receipt.budget_consumption,
        run_plan_path=receipt.plan.absolute_path,
        current_ns=current_ns,
    )
    if (
        admission.budget_mode != "registered_e5_one_shot"
        or admission.failure_execution_binding_sha256 != expected_binding
        or admission.execution_binding_sha256
        != failure_subject.serving_execution_binding_sha256
        or admission.execution_plan_sha256
        != failure_subject.serving_execution_plan_sha256
        or admission.materialization_receipt_sha256
        != failure_subject.materialization_receipt_sha256
        or admission.materialized_cell_id != failure_subject.materialized_cell_id
        or admission.protocol_lock_sha256 != failure_subject.protocol_lock_sha256
        or admission.runtime_authority_manifest_sha256
        != failure_subject.formal_runtime_authority_manifest_sha256
        or admission.registry_sha256 != failure_subject.registry_sha256
        or admission.inventory_sha256 != failure_subject.inventory_sha256
        or admission.topology_mode != failure_subject.topology
        or admission.gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("formal failure launch admission differs from subject")
    _launch, failure_receipt = _reopen_failure_raw_against_subject(
        receipt.raw_failure_terminal,
        subject=failure_subject,
        failure_execution_binding_sha256=expected_binding,
        gpu_uuids=receipt.gpu_uuids,
        process_group_id=receipt.server_process_group_id,
    )
    edges = dict(receipt.phase_edges_ns)
    before_ns = _validate_cost_snapshot(
        receipt.before_gpu_snapshot,
        phase="before",
        inventory_sha256=receipt.inventory_sha256,
        gpu_uuids=receipt.gpu_uuids,
        process_group_id=receipt.server_process_group_id,
    )
    ready_ns = _validate_cost_snapshot(
        receipt.ready_gpu_snapshot,
        phase="ready",
        inventory_sha256=receipt.inventory_sha256,
        gpu_uuids=receipt.gpu_uuids,
        process_group_id=receipt.server_process_group_id,
    )
    after_ns = _validate_cost_snapshot(
        receipt.after_gpu_snapshot,
        phase="after",
        inventory_sha256=receipt.inventory_sha256,
        gpu_uuids=receipt.gpu_uuids,
        process_group_id=receipt.server_process_group_id,
    )
    if (
        not edges["execution_started_ns"] <= before_ns <= edges["server_spawned_ns"]
        or not edges["server_ready_ns"] <= ready_ns <= edges["failure_started_ns"]
        or not edges["process_exited_ns"] <= after_ns <= edges["after_snapshot_ns"]
        or edges["process_group_empty_checked_ns"] > edges["gpu_release_ns"]
        or edges["gpu_release_ns"] > edges["evidence_flush_started_ns"]
        or edges["process_exited_ns"] - edges["execution_started_ns"]
        > admission.hard_timeout_ns
        or edges["gpu_release_ns"] - edges["execution_started_ns"]
        > admission.provider_wave_hard_timeout_ns
    ):
        raise ValueError("formal failure lifecycle timing exceeds admitted boundaries")
    return receipt, plan, failure_receipt, admission


@dataclass(frozen=True)
class FormalE5FailureLifecycleCostProjection:
    """Public immutable E5 timing projection accepted by GPU-hour authority."""

    schema_version: Literal[1]
    kind: Literal["formal_e5_failure_lifecycle_cost_projection"]
    proof_artifact_sha256: str
    raw_lifecycle_receipt_sha256: str
    formal_failure_execution_binding_sha256: str
    failure_subject_sha256: str
    materialized_cell_id: str
    serving_execution_binding_sha256: str
    serving_execution_plan_sha256: str
    assignment_sha256: str
    inventory_sha256: str
    registry_sha256: str
    root_manifest_sha256: str
    run_nonce_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    gpu_uuids: tuple[str, ...]
    server_process_group_id: int
    formal_launch_admission_sha256: str
    formal_launch_consumption_sha256: str
    budget_consumption_sha256: str
    raw_failure_terminal_sha256: str
    recovery_receipt_sha256: str
    execution_started_ns: int
    process_exited_ns: int
    process_group_empty_checked_ns: int
    gpu_release_ns: int
    evidence_flush_finished_ns: int
    compute_gpu_ns: int
    provider_reserved_gpu_ns: int
    evidence_gpu_ns: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_e5_failure_lifecycle_cost_projection"
        ):
            raise ValueError("formal failure cost projection schema differs")
        for name in (
            "proof_artifact_sha256",
            "raw_lifecycle_receipt_sha256",
            "formal_failure_execution_binding_sha256",
            "failure_subject_sha256",
            "materialized_cell_id",
            "serving_execution_binding_sha256",
            "serving_execution_plan_sha256",
            "assignment_sha256",
            "inventory_sha256",
            "registry_sha256",
            "root_manifest_sha256",
            "run_nonce_sha256",
            "formal_launch_admission_sha256",
            "formal_launch_consumption_sha256",
            "budget_consumption_sha256",
            "raw_failure_terminal_sha256",
            "recovery_receipt_sha256",
        ):
            _sha(f"formal failure cost {name}", getattr(self, name))
        if (
            self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}
            or type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != (1 if self.topology_mode == "tp1_dp1" else 2)
            or type(self.server_process_group_id) is not int
            or self.server_process_group_id < 1
        ):
            raise ValueError("formal failure cost execution scope differs")
        edges = (
            self.execution_started_ns,
            self.process_exited_ns,
            self.process_group_empty_checked_ns,
            self.gpu_release_ns,
            self.evidence_flush_finished_ns,
        )
        if any(
            type(value) is not int or value < 1 for value in edges
        ) or edges != tuple(sorted(edges)):
            raise ValueError("formal failure cost timing is not monotonic")
        compute = (self.process_exited_ns - self.execution_started_ns) * len(
            self.gpu_uuids
        )
        if (
            type(self.compute_gpu_ns) is not int
            or self.compute_gpu_ns != compute
            or type(self.provider_reserved_gpu_ns) is not int
            or self.provider_reserved_gpu_ns < compute
            or type(self.evidence_gpu_ns) is not int
            or self.evidence_gpu_ns < 0
        ):
            raise ValueError("formal failure cost arithmetic differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "gpu_uuids"
            },
            "gpu_uuids": list(self.gpu_uuids),
        }


async def execute_formal_e5_failure_run_plan(
    *,
    plan_path: str | Path,
    launch_admission_path: str | Path,
    execution_binding: FormalServingExecutionBinding | None = None,
    failure_execution_binding: VerifiedFormalFailureExecutionBinding | None = None,
    failure_execution_descriptor_path: str | Path | None = None,
    nvidia_smi_tool: PinnedNvidiaSmiTool,
) -> ValidatedUnsignedFormalE5FailureRun:
    """Spawn, actuate, flush, and clean one exact E5 failure cell."""

    current = failure_execution_descriptor_path is not None
    if current:
        if execution_binding is not None or failure_execution_binding is not None:
            raise ValueError(
                "current E5 failure execution cannot carry a private token"
            )
        failure = revalidate_formal_single_operator_e5_failure_execution_descriptor(
            failure_execution_descriptor_path,
            current_ns=time.time_ns(),
        )
        plan, launch, _schedule = _load_formal_single_operator_trusted_run_plan(
            plan_path
        )
        if (
            plan.single_operator_execution_rebuild_source
            != CanonicalJsonProofBinding.bind(failure_execution_descriptor_path)
        ):
            raise ValueError("current E5 failure plan uses another descriptor")
        serving = None
        run_config = load_run_config(launch.run_config_path)
    else:
        if failure_execution_binding is None:
            raise TypeError("legacy E5 failure execution requires its sealed token")
        serving = (
            rebuild_formal_single_operator_execution_binding_from_plan(plan_path)
            if execution_binding is None
            else require_verified_formal_serving_execution_binding(execution_binding)
        )
        private_serving = (
            serving.verified_binding
            if type(serving) is FormalSingleOperatorExecutionBinding
            else serving
        )
        failure = require_verified_formal_failure_execution_binding(
            failure_execution_binding
        )
        if failure.serving_execution.sha256 != private_serving.sha256:
            raise ValueError(
                "formal failure physical runner received another serving token"
            )
        plan = load_formal_serving_run_plan(
            plan_path,
            execution_binding=serving,
            verified_nextn_tp2_authority=serving.verified_nextn_tp2_authority,
        )
        launch, _schedule = revalidate_formal_serving_run_plan(
            plan,
            execution_binding=serving,
            verified_nextn_tp2_authority=serving.verified_nextn_tp2_authority,
        )
        run_config = serving.run_config
    if (
        plan.stage != "E5"
        or plan.materialized_cell_id != failure.subject.materialized_cell_id
        or plan.topology_mode != failure.subject.topology
        or plan.inventory_sha256 != failure.subject.inventory_sha256
        or plan.native_terminal_binding.execution_plan_sha256
        != failure.subject.serving_execution_plan_sha256
        or plan.native_terminal_binding.run_nonce_sha256
        != failure.subject.run_nonce_sha256
    ):
        raise ValueError("formal failure physical plan differs from sealed assignment")
    output_paths = (
        plan.terminal_output_path,
        plan.live_run_receipt_output_path,
        plan.lifecycle_timing_output_path,
        plan.server_log_output_path,
        plan.server_stdout_output_path,
        plan.server_stderr_output_path,
        plan.junit_output_path,
        plan.before_gpu_snapshot_output_path,
        plan.ready_gpu_snapshot_output_path,
        plan.after_gpu_snapshot_output_path,
        plan.fatal_output_path,
    )
    if any(os.path.lexists(path) for path in output_paths):
        raise FileExistsError("formal failure physical output already exists")
    executable = Path(launch.server_argv[0])
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or executable.is_symlink()
    ):
        raise ValueError("formal failure server executable is invalid")
    admission_binding = CanonicalJsonProofBinding.bind(launch_admission_path)
    admission_value = admission_binding.reopen()
    if admission_value.get("kind") == "formal_single_operator_admission":
        from lightcone_spec.orchestration.formal_single_operator_admission import (
            consume_formal_single_operator_admission,
            validate_formal_single_operator_admission,
        )

        single_admission = validate_formal_single_operator_admission(
            launch_admission_path,
            plan_path=plan_path,
        )
        if (
            single_admission.stage != "E5"
            or single_admission.materialized_cell_id
            != failure.subject.materialized_cell_id
            or single_admission.inventory_sha256 != failure.subject.inventory_sha256
            or single_admission.topology_mode != failure.subject.topology
            or single_admission.gpu_uuids != _physical_failure_gpu_uuids(failure)
            or single_admission.registry_sha256 != failure.subject.registry_sha256
        ):
            raise ValueError(
                "single-operator failure admission differs from assignment"
            )
        timeout_ns = single_admission.process_hard_timeout_ns
        launch_consumption = consume_formal_single_operator_admission(
            single_admission,
            consumed_ns=time.time_ns(),
        )
        launch_admission = admission_binding
        budget_consumption = launch_consumption
    else:
        from lightcone_spec.orchestration.formal_launch_admission import (
            consume_formal_stage_launch_admission,
            validate_formal_stage_launch_admission,
        )

        admission = validate_formal_stage_launch_admission(
            launch_admission_path,
            execution_binding=serving,
            run_plan_path=plan_path,
            current_ns=time.time_ns(),
        )
        if (
            admission.artifact.budget_mode != "registered_e5_one_shot"
            or admission.artifact.failure_execution_binding_sha256
            != _physical_failure_binding_sha256(failure)
        ):
            raise ValueError(
                "formal failure runner requires its dedicated one-shot admission"
            )
        timeout_ns = admission.artifact.hard_timeout_ns
        launch_consumption = consume_formal_stage_launch_admission(
            admission,
            consumed_ns=time.time_ns(),
        )
        launch_admission = CanonicalJsonProofBinding.bind(
            launch_admission_path,
            semantic_sha256=admission.artifact.sha256,
        )
        budget_consumption = admission.artifact.budget_consumption
    timeout = timeout_ns / 1_000_000_000
    if not 1.0 <= timeout <= 3_600.0:
        raise ValueError("formal failure admission timeout is outside runner bounds")
    _require_port_unused(launch.localhost_port)
    quota_root = _private_new_directory(
        Path(plan.private_output_root) / "failure-quota"
    )
    before_path = Path(plan.before_gpu_snapshot_output_path)
    ready_path = Path(plan.ready_gpu_snapshot_output_path)
    after_path = Path(plan.after_gpu_snapshot_output_path)
    process: subprocess.Popen[bytes] | None = None
    admin: PinnedBenchServingTransport | None = None
    log_file = None
    stdout_file = None
    stderr_file = None
    before: CanonicalJsonProofBinding | None = None
    ready: CanonicalJsonProofBinding | None = None
    after: CanonicalJsonProofBinding | None = None
    raw_terminal: CanonicalJsonProofBinding | None = None
    launch_binding: FormalFailureActuatorLaunchBinding | None = None
    failure_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    phase: dict[str, int] = {"execution_started_ns": time.monotonic_ns()}
    try:
        before = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=nvidia_smi_tool,
            gpu_uuids=launch.gpu_uuids,
            inventory_sha256=launch.inventory_sha256,
            phase="before",
            output_path=before_path,
        )
        descriptor = os.open(
            plan.server_log_output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        log_file = os.fdopen(descriptor, "wb", buffering=0)
        log_file.write(b"source-owned formal E5 failure run started\n")
        stdout_descriptor = os.open(
            plan.server_stdout_output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        stderr_descriptor = os.open(
            plan.server_stderr_output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        stdout_file = os.fdopen(stdout_descriptor, "wb", buffering=0)
        stderr_file = os.fdopen(stderr_descriptor, "wb", buffering=0)
        stdout_file.write(b"source-owned formal E5 server stdout opened\n")
        stderr_file.write(b"source-owned formal E5 server stderr opened\n")
        process = await asyncio.to_thread(
            _spawn_failure_server,
            launch,
            failure=failure,
            quota_root=quota_root,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )
        phase["server_spawned_ns"] = time.monotonic_ns()
        await asyncio.to_thread(
            _wait_server_ready,
            process,
            port=launch.localhost_port,
            timeout_seconds=min(timeout, 600.0),
        )
        phase["server_ready_ns"] = time.monotonic_ns()
        ready = await asyncio.to_thread(
            _capture_gpu_process_snapshot,
            tool=nvidia_smi_tool,
            gpu_uuids=launch.gpu_uuids,
            inventory_sha256=launch.inventory_sha256,
            phase="ready",
            output_path=ready_path,
            shared_server_process_group_id=process.pid,
        )
        admin = PinnedBenchServingTransport.from_checkout(
            launch.patched_sglang_checkout
        )
        if type(admin) is not PinnedBenchServingTransport:
            raise TypeError("formal failure run requires exact pinned admin transport")
        await admin.open(
            request_timeout_s=min(timeout, 600.0),
            abort_timeout_s=30.0,
        )
        base_url = f"http://127.0.0.1:{launch.localhost_port}"
        admin.bind_native_admin_base_url(base_url)
        await _observe_live_server_execution_policy(
            transport=admin,
            config=run_config,
        )
        capability = await admin.get_json(
            "/v1/lightcone-spec/failure-actuator/capability"
        )
        launch_binding = _launch_from_capability(
            capability,
            failure=failure,
            port=launch.localhost_port,
            process_group_id=process.pid,
        )
        await admin.close()
        admin = None
        phase["failure_started_ns"] = time.monotonic_ns()
        raw_terminal = await asyncio.to_thread(
            execute_formal_failure_actuator_unsigned,
            failure,
            launch=launch_binding,
            transport=SglangHttpFailureNativeControlTransport(),
            raw_terminal_path=plan.terminal_output_path,
        )
        phase["failure_finished_ns"] = time.monotonic_ns()
    except BaseException as error:  # noqa: BLE001 - cleanup is mandatory
        failure_error = error
    finally:
        if admin is not None:
            try:
                await admin.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_error = cleanup_error or error
        if process is not None:
            try:
                await asyncio.to_thread(_terminate_process_group, process)
                phase["process_exited_ns"] = time.monotonic_ns()
            except BaseException as error:  # noqa: BLE001
                cleanup_error = cleanup_error or error
        for stream in (log_file, stdout_file, stderr_file):
            if stream is None:
                continue
            try:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_error = cleanup_error or error
        try:
            after = await asyncio.to_thread(
                _capture_gpu_process_snapshot,
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="after",
                output_path=after_path,
            )
            phase["after_snapshot_ns"] = time.monotonic_ns()
        except BaseException as error:  # noqa: BLE001
            after = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="after",
                output_path=after_path,
                error=error,
            )
            cleanup_error = cleanup_error or error
    error = cleanup_error or failure_error
    if error is not None:
        if before is None and not before_path.exists():
            before = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="before",
                output_path=before_path,
                error=error,
            )
        if ready is None and not ready_path.exists():
            ready = _publish_gpu_snapshot_error(
                tool=nvidia_smi_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="ready",
                output_path=ready_path,
                error=error,
                expected_server_process_group_ids=(
                    None
                    if process is None
                    else tuple(process.pid for _gpu in launch.gpu_uuids)
                ),
            )
        publish_canonical_json_no_replace(
            plan.fatal_output_path,
            {
                "schema_version": 1,
                "kind": "unsigned_formal_e5_failure_physical_fatal",
                "protocol_sha256": FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256,
                "formal_execution_authorized": False,
                "plan_sha256": plan.sha256,
                "formal_failure_execution_binding_sha256": (
                    _physical_failure_binding_sha256(failure)
                ),
                "reason_code": "formal_e5_failure_physical_execution_failed",
                "error_type": type(error).__name__,
                "process_group_empty": (
                    None
                    if process is None
                    else not _process_group_exists_for_formal_dispatch(process.pid)
                ),
            },
        )
        raise FormalPhysicalDispatchError(
            "formal_e5_failure_physical_execution_failed",
            CanonicalJsonProofBinding.bind(plan.fatal_output_path),
        ) from error
    assert process is not None
    assert raw_terminal is not None
    assert launch_binding is not None
    assert before is not None and ready is not None and after is not None
    if _process_group_exists_for_formal_dispatch(process.pid):
        raise RuntimeError("formal failure process group survived cleanup")
    phase["process_group_empty_checked_ns"] = time.monotonic_ns()
    phase["gpu_release_ns"] = phase["process_group_empty_checked_ns"]
    phase["evidence_flush_started_ns"] = time.monotonic_ns()
    if type(process.returncode) is not int:
        raise RuntimeError("formal failure child lacks a terminal exit code")
    log_binding = EvidenceFileBinding.bind(
        Path(plan.server_log_output_path), label="formal failure server log"
    )
    stdout_binding = EvidenceFileBinding.bind(
        Path(plan.server_stdout_output_path), label="formal failure server stdout"
    )
    stderr_binding = EvidenceFileBinding.bind(
        Path(plan.server_stderr_output_path), label="formal failure server stderr"
    )
    junit_binding = _publish_failure_junit(
        output_path=plan.junit_output_path,
        materialized_cell_id=plan.materialized_cell_id,
        scenario=failure.subject.scenario,
    )
    for binding in (raw_terminal, before, ready, after):
        binding.reopen()
    phase["evidence_flush_finished_ns"] = time.monotonic_ns()
    ordered_names = (
        "execution_started_ns",
        "server_spawned_ns",
        "server_ready_ns",
        "failure_started_ns",
        "failure_finished_ns",
        "process_exited_ns",
        "after_snapshot_ns",
        "process_group_empty_checked_ns",
        "gpu_release_ns",
        "evidence_flush_started_ns",
        "evidence_flush_finished_ns",
    )
    receipt = FormalE5FailureLifecycleRawReceipt(
        schema_version=3,
        kind="formal_e5_failure_lifecycle_raw_receipt",
        protocol_sha256=FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        plan=CanonicalJsonProofBinding.bind(plan_path),
        formal_launch_admission=launch_admission,
        formal_launch_consumption=launch_consumption,
        budget_consumption=budget_consumption,
        formal_failure_execution_binding_sha256=(
            _physical_failure_binding_sha256(failure)
        ),
        raw_failure_terminal=raw_terminal,
        launch_binding_sha256=launch_binding.sha256,
        inventory_sha256=failure.subject.inventory_sha256,
        registry_sha256=failure.subject.registry_sha256,
        run_nonce_sha256=failure.subject.run_nonce_sha256,
        topology_mode=failure.subject.topology,  # type: ignore[arg-type]
        gpu_uuids=_physical_failure_gpu_uuids(failure),
        before_gpu_snapshot=before,
        ready_gpu_snapshot=ready,
        after_gpu_snapshot=after,
        server_log=log_binding,
        server_stdout=stdout_binding,
        server_stderr=stderr_binding,
        junit=junit_binding,
        server_process_group_id=process.pid,
        process_exit_code=process.returncode,
        process_group_empty=True,
        phase_edges_ns=tuple((name, phase[name]) for name in ordered_names),
    )
    publish_canonical_json_no_replace(
        plan.lifecycle_timing_output_path,
        receipt.to_dict(),
    )
    lifecycle_binding = CanonicalJsonProofBinding.bind(
        plan.lifecycle_timing_output_path,
        semantic_sha256=receipt.sha256,
    )
    publish_canonical_json_no_replace(
        plan.live_run_receipt_output_path,
        {
            "schema_version": 1,
            "kind": "unsigned_formal_e5_failure_physical_run_receipt",
            "protocol_sha256": FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
            "formal_execution_authorized": False,
            "plan_sha256": plan.sha256,
            "formal_launch_admission": launch_admission.to_dict(),
            "formal_launch_consumption": launch_consumption.to_dict(),
            "budget_consumption": budget_consumption.to_dict(),
            "lifecycle_receipt": lifecycle_binding.to_dict(),
            "raw_failure_terminal": raw_terminal.to_dict(),
            "server_log": log_binding.to_dict(),
            "server_stdout": stdout_binding.to_dict(),
            "server_stderr": stderr_binding.to_dict(),
            "junit": junit_binding.to_dict(),
            "process_exit_code": process.returncode,
            "process_group_empty": True,
        },
    )
    if current:
        validate_formal_single_operator_e5_physical_outcome(
            plan_path=plan_path,
            run_receipt_path=plan.live_run_receipt_output_path,
            lifecycle_receipt_path=lifecycle_binding.absolute_path,
        )
    else:
        assert serving is not None
        assert type(failure) is VerifiedFormalFailureExecutionBinding
        validate_formal_e5_failure_lifecycle_raw_receipt(
            lifecycle_binding.absolute_path,
            plan_path=plan_path,
            execution_binding=serving,
            failure_execution_binding=failure,
        )
    return ValidatedUnsignedFormalE5FailureRun(
        lifecycle_receipt=lifecycle_binding,
        raw_failure_terminal=raw_terminal,
    )


def _lifecycle_subject(
    receipt: FormalE5FailureLifecycleRawReceipt,
) -> ControlArtifactSubject:
    lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_e5_failure_lifecycle_control_lineage",
            "plan_sha256": receipt.plan.semantic_sha256,
            "formal_failure_execution_binding_sha256": (
                receipt.formal_failure_execution_binding_sha256
            ),
            "raw_failure_terminal_sha256": (
                receipt.raw_failure_terminal.semantic_sha256
            ),
            "formal_launch_admission_sha256": (
                receipt.formal_launch_admission.semantic_sha256
            ),
            "formal_launch_consumption_sha256": (
                receipt.formal_launch_consumption.semantic_sha256
            ),
            "budget_consumption_sha256": (receipt.budget_consumption.semantic_sha256),
            "launch_binding_sha256": receipt.launch_binding_sha256,
            "run_nonce_sha256": receipt.run_nonce_sha256,
            "gpu_uuids": list(receipt.gpu_uuids),
            "server_process_group_id": receipt.server_process_group_id,
            "process_exit_code": receipt.process_exit_code,
            "server_log_raw_sha256": receipt.server_log.raw_sha256,
            "server_stdout_raw_sha256": receipt.server_stdout.raw_sha256,
            "server_stderr_raw_sha256": receipt.server_stderr.raw_sha256,
            "junit_raw_sha256": receipt.junit.raw_sha256,
            "phase_edges_ns": [list(row) for row in receipt.phase_edges_ns],
        }
    )
    return ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=receipt.sha256,
        protocol_sha256=FORMAL_E5_FAILURE_LIFECYCLE_CONTROL_PROTOCOL_SHA256,
        registry_sha256=receipt.registry_sha256,
        lineage_sha256=lineage,
    )


def build_formal_e5_failure_lifecycle_control_subject(
    lifecycle_receipt_path: str | Path,
    *,
    plan_path: str | Path,
    execution_binding: VerifiedFormalServingExecutionBinding,
    failure_execution_binding: VerifiedFormalFailureExecutionBinding,
) -> ControlArtifactSubject:
    receipt = validate_formal_e5_failure_lifecycle_raw_receipt(
        lifecycle_receipt_path,
        plan_path=plan_path,
        execution_binding=execution_binding,
        failure_execution_binding=failure_execution_binding,
    )
    return _lifecycle_subject(receipt)


@dataclass(frozen=True)
class FormalE5FailureLifecycleProofArtifact:
    schema_version: Literal[1]
    kind: Literal["formal_e5_failure_lifecycle_proof_artifact"]
    raw_lifecycle_receipt: CanonicalJsonProofBinding
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    expected_root_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_e5_failure_lifecycle_proof_artifact"
        ):
            raise ValueError("formal failure lifecycle proof schema differs")
        _sha("formal failure lifecycle proof root", self.expected_root_manifest_sha256)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_lifecycle_receipt": self.raw_lifecycle_receipt.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "expected_root_manifest_sha256": self.expected_root_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "formal failure lifecycle proof",
            value,
            set(cls.__dataclass_fields__),
        )
        raw = CanonicalJsonProofBinding.from_dict(row.pop("raw_lifecycle_receipt"))
        control = ControlArtifactAttestation.from_dict(row.pop("control_attestation"))
        reservation = ChallengeReplayReservationBinding.from_dict(
            row.pop("replay_reservation")
        )
        return cls(
            **row,  # type: ignore[arg-type]
            raw_lifecycle_receipt=raw,
            control_attestation=control,
            replay_reservation=reservation,
        )


def publish_formal_e5_failure_lifecycle_proof_artifact(
    lifecycle_receipt_path: str | Path,
    *,
    plan_path: str | Path,
    execution_binding: VerifiedFormalServingExecutionBinding,
    failure_execution_binding: VerifiedFormalFailureExecutionBinding,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    expected_root_manifest_sha256: str,
    now_ns: int,
    proof_artifact_path: str | Path,
) -> CanonicalJsonProofBinding:
    receipt = validate_formal_e5_failure_lifecycle_raw_receipt(
        lifecycle_receipt_path,
        plan_path=plan_path,
        execution_binding=execution_binding,
        failure_execution_binding=failure_execution_binding,
    )
    subject = _lifecycle_subject(receipt)
    if control_attestation.subject != subject or (
        control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("formal failure lifecycle external control differs")
    controls = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=receipt.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=(receipt.run_nonce_sha256,),
    )
    reservation_sha = control_challenge_reservation_sha256(
        controls,
        reserved_ns=now_ns,
        additional_challenge_sha256s=(receipt.run_nonce_sha256,),
    )
    artifact = FormalE5FailureLifecycleProofArtifact(
        schema_version=1,
        kind="formal_e5_failure_lifecycle_proof_artifact",
        raw_lifecycle_receipt=CanonicalJsonProofBinding.bind(lifecycle_receipt_path),
        control_attestation=control_attestation,
        replay_reservation=replay_store.bind_reservation(reservation_sha),
        expected_root_manifest_sha256=expected_root_manifest_sha256,
    )
    publish_canonical_json_no_replace(proof_artifact_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(
        proof_artifact_path,
        semantic_sha256=artifact.sha256,
    )


def validate_formal_e5_failure_lifecycle_proof_artifact(
    proof_artifact_path: str | Path,
    *,
    plan_path: str | Path,
    execution_binding: VerifiedFormalServingExecutionBinding,
    failure_execution_binding: VerifiedFormalFailureExecutionBinding,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> FormalE5FailureLifecycleRawReceipt:
    binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalE5FailureLifecycleProofArtifact.from_dict(binding.reopen())
    if (
        artifact.sha256 != binding.semantic_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
        or CanonicalJsonProofBinding.bind(artifact.raw_lifecycle_receipt.absolute_path)
        != artifact.raw_lifecycle_receipt
    ):
        raise ValueError("formal failure lifecycle proof identity/root differs")
    receipt = validate_formal_e5_failure_lifecycle_raw_receipt(
        artifact.raw_lifecycle_receipt.absolute_path,
        plan_path=plan_path,
        execution_binding=execution_binding,
        failure_execution_binding=failure_execution_binding,
    )
    subject = _lifecycle_subject(receipt)
    if artifact.control_attestation.subject != subject or (
        artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("formal failure lifecycle proof control differs")
    reserved = artifact.replay_reservation.revalidate()
    if type(now_ns) is not int or now_ns < artifact.replay_reservation.reserved_ns:
        raise ValueError("formal failure lifecycle proof time precedes reservation")
    control = verify_release_control_artifact_attestation(
        artifact.control_attestation,
        expected_inventory_sha256=receipt.inventory_sha256,
        now_ns=artifact.replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    expected = tuple(
        sorted(
            {
                receipt.run_nonce_sha256,
                control.challenge_sha256,
                control.deployment_policy_challenge_sha256,
            }
        )
    )
    reservation_sha = control_challenge_reservation_sha256(
        (control,),
        reserved_ns=artifact.replay_reservation.reserved_ns,
        additional_challenge_sha256s=(receipt.run_nonce_sha256,),
    )
    if (
        reserved != expected
        or artifact.replay_reservation.reservation_sha256 != reservation_sha
    ):
        raise ValueError("formal failure lifecycle proof replay differs")
    return receipt


def validate_formal_e5_failure_lifecycle_cost_proof_artifact(
    proof_artifact_path: str | Path,
    *,
    failure_subject: FormalFailureExecutionSubject,
    expected_protocol_lock_sha256: str,
    expected_runtime_authority_manifest_sha256: str,
    expected_materialization_receipt_sha256: str,
    expected_materialized_cell_id: str,
    expected_inventory_sha256: str,
    expected_registry_sha256: str,
    expected_root_manifest_sha256: str,
    now_ns: int,
) -> FormalE5FailureLifecycleCostProjection:
    """Deep-reopen one controlled E5 lifecycle into a public cost projection.

    The verifier consumes only durable public values.  In particular it does
    not accept either verifier-private execution token, nor any caller timing,
    GPU count, process ID, or cost scalar.
    """

    if type(failure_subject) is not FormalFailureExecutionSubject:
        raise TypeError("formal failure cost requires an exact durable subject")
    failure_subject.__post_init__()
    for label, expected, observed in (
        (
            "protocol lock",
            expected_protocol_lock_sha256,
            failure_subject.protocol_lock_sha256,
        ),
        (
            "runtime authority",
            expected_runtime_authority_manifest_sha256,
            failure_subject.formal_runtime_authority_manifest_sha256,
        ),
        (
            "materialization",
            expected_materialization_receipt_sha256,
            failure_subject.materialization_receipt_sha256,
        ),
        (
            "materialized cell",
            expected_materialized_cell_id,
            failure_subject.materialized_cell_id,
        ),
        ("inventory", expected_inventory_sha256, failure_subject.inventory_sha256),
        ("registry", expected_registry_sha256, failure_subject.registry_sha256),
    ):
        _sha(f"formal failure expected {label}", expected)
        if expected != observed:
            raise ValueError(f"formal failure cost {label} differs")
    _sha("formal failure expected root", expected_root_manifest_sha256)
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("formal failure cost validation time is invalid")
    proof_binding = CanonicalJsonProofBinding.bind(proof_artifact_path)
    artifact = FormalE5FailureLifecycleProofArtifact.from_dict(proof_binding.reopen())
    if (
        artifact.sha256 != proof_binding.semantic_sha256
        or artifact.expected_root_manifest_sha256 != expected_root_manifest_sha256
        or CanonicalJsonProofBinding.bind(artifact.raw_lifecycle_receipt.absolute_path)
        != artifact.raw_lifecycle_receipt
    ):
        raise ValueError("formal failure cost proof identity/root differs")
    expected_failure_binding = _failure_execution_binding_sha256(failure_subject)
    receipt, _plan, recovery, admission = (
        _validate_formal_e5_failure_lifecycle_raw_cost_receipt(
            artifact.raw_lifecycle_receipt.absolute_path,
            failure_subject=failure_subject,
            expected_failure_execution_binding_sha256=(expected_failure_binding),
            current_ns=now_ns,
        )
    )
    subject = _lifecycle_subject(receipt)
    if artifact.control_attestation.subject != subject or (
        artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
        != expected_root_manifest_sha256
    ):
        raise ValueError("formal failure cost proof control differs")
    reserved = artifact.replay_reservation.revalidate()
    if now_ns < artifact.replay_reservation.reserved_ns:
        raise ValueError("formal failure cost proof time precedes reservation")
    control = verify_release_control_artifact_attestation(
        artifact.control_attestation,
        expected_inventory_sha256=receipt.inventory_sha256,
        now_ns=artifact.replay_reservation.reserved_ns,
        consumed_challenge_sha256s=(),
    )
    expected_challenges = tuple(
        sorted(
            {
                receipt.run_nonce_sha256,
                control.challenge_sha256,
                control.deployment_policy_challenge_sha256,
            }
        )
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        (control,),
        reserved_ns=artifact.replay_reservation.reserved_ns,
        additional_challenge_sha256s=(receipt.run_nonce_sha256,),
    )
    if (
        reserved != expected_challenges
        or artifact.replay_reservation.reservation_sha256 != reservation_sha256
    ):
        raise ValueError("formal failure cost proof replay differs")
    from lightcone_spec.orchestration.formal_launch_admission import (
        _load_capacity_schedule,
    )

    schedule = _load_capacity_schedule(admission.capacity_schedule.absolute_path)
    if (
        schedule.sha256 != admission.capacity_schedule.semantic_sha256
        or schedule.materialized_cell_id != failure_subject.materialized_cell_id
        or schedule.inventory_sha256 != failure_subject.inventory_sha256
    ):
        raise ValueError("formal failure cost capacity schedule differs")
    edges = dict(receipt.phase_edges_ns)
    compute_gpu_ns = (edges["process_exited_ns"] - edges["execution_started_ns"]) * len(
        receipt.gpu_uuids
    )
    provider_reserved_gpu_ns = (
        edges["gpu_release_ns"] - edges["execution_started_ns"]
    ) * schedule.provider_reserved_gpu_count
    evidence_gpu_ns = (
        edges["evidence_flush_finished_ns"] - edges["gpu_release_ns"]
    ) * schedule.provider_inventory_gpu_count
    projection = FormalE5FailureLifecycleCostProjection(
        schema_version=1,
        kind="formal_e5_failure_lifecycle_cost_projection",
        proof_artifact_sha256=artifact.sha256,
        raw_lifecycle_receipt_sha256=receipt.sha256,
        formal_failure_execution_binding_sha256=expected_failure_binding,
        failure_subject_sha256=failure_subject.sha256,
        materialized_cell_id=failure_subject.materialized_cell_id,
        serving_execution_binding_sha256=(
            failure_subject.serving_execution_binding_sha256
        ),
        serving_execution_plan_sha256=(failure_subject.serving_execution_plan_sha256),
        assignment_sha256=failure_subject.assignment_sha256,
        inventory_sha256=receipt.inventory_sha256,
        registry_sha256=receipt.registry_sha256,
        root_manifest_sha256=expected_root_manifest_sha256,
        run_nonce_sha256=receipt.run_nonce_sha256,
        topology_mode=receipt.topology_mode,
        gpu_uuids=receipt.gpu_uuids,
        server_process_group_id=receipt.server_process_group_id,
        formal_launch_admission_sha256=(
            receipt.formal_launch_admission.semantic_sha256
        ),
        formal_launch_consumption_sha256=(
            receipt.formal_launch_consumption.semantic_sha256
        ),
        budget_consumption_sha256=receipt.budget_consumption.semantic_sha256,
        raw_failure_terminal_sha256=receipt.raw_failure_terminal.semantic_sha256,
        recovery_receipt_sha256=recovery.sha256,
        execution_started_ns=edges["execution_started_ns"],
        process_exited_ns=edges["process_exited_ns"],
        process_group_empty_checked_ns=edges["process_group_empty_checked_ns"],
        gpu_release_ns=edges["gpu_release_ns"],
        evidence_flush_finished_ns=edges["evidence_flush_finished_ns"],
        compute_gpu_ns=compute_gpu_ns,
        provider_reserved_gpu_ns=provider_reserved_gpu_ns,
        evidence_gpu_ns=evidence_gpu_ns,
    )
    projection.__post_init__()
    return projection


__all__ = [
    "FORMAL_E5_FAILURE_LIFECYCLE_CONTROL_PROTOCOL_SHA256",
    "FORMAL_E5_FAILURE_PHYSICAL_PROTOCOL_SHA256",
    "FormalE5FailureLifecycleCostProjection",
    "FormalE5FailureLifecycleProofArtifact",
    "FormalE5FailureLifecycleRawReceipt",
    "FormalSingleOperatorE5PhysicalOutcome",
    "ValidatedUnsignedFormalE5FailureRun",
    "build_formal_e5_failure_lifecycle_control_subject",
    "execute_formal_e5_failure_run_plan",
    "publish_formal_e5_failure_lifecycle_proof_artifact",
    "validate_formal_e5_failure_lifecycle_cost_proof_artifact",
    "validate_formal_e5_failure_lifecycle_proof_artifact",
    "validate_formal_e5_failure_lifecycle_raw_receipt",
    "validate_formal_single_operator_e5_physical_outcome",
]
