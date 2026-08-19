"""Trusted exact-ten native qualification plans and empirical terminals.

The signed release qualification machinery remains the authority for formal
``MEASURED`` claims.  This module is deliberately narrower: the trusted
single-operator exactness cell owns six source-defined live suites and runs
them sequentially inside that one logical cell.  Plans are derived only from
the frozen ProtocolLock, BOUND content, PASS runtime observations, the exact
two-GPU assignment, and existing source-owned launch manifests.  Results are
explicitly unsigned empirical evidence and cannot be parsed as signed release
proofs.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration.runtime import _render_server
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
)
from lightcone_spec.runtime.compile_runner import (
    CompileLaunchManifest,
    write_compile_prewarm_manifest,
)
from lightcone_spec.runtime.distributed import (
    DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS,
    DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
    DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
)
from lightcone_spec.runtime.native_qualification_runner import (
    NATIVE_RUNTIME_GPU_TEST_FILES,
    NATIVE_RUNTIME_GPU_TEST_NAMES,
    NativeRuntimeQualificationAssignment,
    NativeRuntimeQualificationObservation,
    _junit_summary,
    _process_group_exists,
    _publish_snapshot,
    _source_capability_sha256,
    _terminate_process_group,
    _validate_gpu_snapshot,
)
from lightcone_spec.runtime.preflight_runner import (
    EvidenceFileBinding,
    ExactnessPreflightAssignment,
    ExactnessPreflightResultPointer,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
)

TrustedPreflightQualificationSuite = Literal[
    "chronobelief_gpu_parity",
    "dspark_tp1",
    "dspark_tp2",
    "dspark_dp2",
    "tp2_dp1",
    "tp1_dp2",
]

TRUSTED_PREFLIGHT_QUALIFICATION_SUITES: tuple[
    TrustedPreflightQualificationSuite, ...
] = (
    "chronobelief_gpu_parity",
    "dspark_dp2",
    "dspark_tp1",
    "dspark_tp2",
    "tp1_dp2",
    "tp2_dp1",
)

_SUITE_ALGORITHM: dict[TrustedPreflightQualificationSuite, str] = {
    "chronobelief_gpu_parity": "DFLASH",
    "dspark_tp1": "DSPARK",
    "dspark_tp2": "DSPARK",
    "dspark_dp2": "DSPARK",
    "tp2_dp1": "DFLASH",
    "tp1_dp2": "DFLASH",
}
_SUITE_TOPOLOGY: dict[TrustedPreflightQualificationSuite, str] = {
    "chronobelief_gpu_parity": "tp1_dp1",
    "dspark_tp1": "tp1_dp1",
    "dspark_tp2": "tp2_dp1",
    "dspark_dp2": "tp1_dp2",
    "tp2_dp1": "tp2_dp1",
    "tp1_dp2": "tp1_dp2",
}
_SUITE_PORTS = {
    suite_id: 34_200 + index
    for index, suite_id in enumerate(TRUSTED_PREFLIGHT_QUALIFICATION_SUITES)
}
_QUALIFICATION_TIMEOUT_SECONDS = 1_800.0

FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_preflight_qualification",
        "trust_mode": "trusted_single_operator_no_signature",
        "formal_measurement": False,
        "logical_cell": "exactness_memory_telemetry_preflight",
        "physical_suites": list(TRUSTED_PREFLIGHT_QUALIFICATION_SUITES),
        "suite_coverage": "source_owned_exact_8_of_8_zero_skip_each",
        "lineage": (
            "ProtocolLock_BOUND_content_inventory_doctor_exactness_assignment_"
            "launch_argv_GPU_UUID_stdout_stderr_JUnit_terminal_timestamps"
        ),
        "execution": "sequential_under_exactness_two_GPU_gang_lock",
        "claim": "empirical_only_not_signed_not_formal_MEASURED",
    }
)


def _strict(value: object, expected: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _binding(value: object) -> CanonicalJsonProofBinding:
    return CanonicalJsonProofBinding.from_dict(value)


def _absolute_new_directory(path: str | Path) -> Path:
    root = Path(path)
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or os.path.lexists(root)
        or not root.parent.is_dir()
        or root.parent.is_symlink()
    ):
        raise ValueError("qualification output root must be one new directory")
    root.mkdir(mode=0o700)
    return root


def _raw_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runner_protocol(suite_id: TrustedPreflightQualificationSuite) -> str:
    if suite_id in DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S:
        return DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[suite_id]
    return NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[suite_id]


def _topology_shape(topology: str) -> tuple[int, int]:
    return {
        "tp1_dp1": (1, 1),
        "tp2_dp1": (2, 1),
        "tp1_dp2": (1, 2),
    }[topology]


@dataclass(frozen=True)
class TrustedQualificationDispatchAuthority:
    """Unsigned, launch-independent authority for one exact physical suite."""

    schema_version: Literal[1]
    kind: Literal["trusted_single_operator_preflight_qualification_dispatch"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_no_signature"]
    formal_measurement: Literal[False]
    signature_status: Literal["NOT_APPLICABLE"]
    protocol_lock: CanonicalJsonProofBinding
    content_source: FormalContentSourceBinding
    inventory: CanonicalJsonProofBinding
    doctor: CanonicalJsonProofBinding
    exactness_assignment: CanonicalJsonProofBinding
    suite_id: TrustedPreflightQualificationSuite
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    gpu_uuids: tuple[str, ...]
    source_capability_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_single_operator_preflight_qualification_dispatch"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_no_signature"
            or self.formal_measurement is not False
            or self.signature_status != "NOT_APPLICABLE"
            or self.suite_id not in TRUSTED_PREFLIGHT_QUALIFICATION_SUITES
            or self.topology_mode != _SUITE_TOPOLOGY[self.suite_id]
            or self.source_capability_sha256 != _source_capability_sha256(self.suite_id)
        ):
            raise ValueError("trusted qualification dispatch schema differs")
        lock = protocol_lock_from_dict(self.protocol_lock.reopen())
        bundle = self.content_source.reopen()
        inventory = GpuInventory.from_dict(self.inventory.reopen())
        exactness = ExactnessPreflightAssignment.load(
            self.exactness_assignment.absolute_path
        )
        if (
            lock.schema_version != 5
            or lock.content_source_mode != "trusted_single_operator"
            or any(
                getattr(lock, name) is not None
                for name in (
                    "offline_release_trust_root_sha256",
                    "prepared_model_content_authorization_sha256",
                    "formal_workload_e3a_authorization_sha256",
                    "formal_workload_e0_authorization_sha256",
                    "burstgpt_shape_authorization_sha256",
                )
            )
            or lock.trusted_single_operator_content_bundle_sha256
            != self.content_source.content_sha256
            or bundle.runtime_binding_status != "BOUND"
            or bundle.runtime_observations is None
            or type(self.doctor.reopen()) is not dict
            or inventory.sha256 != self.inventory.semantic_sha256
            or exactness.inventory_sha256 != inventory.sha256
            or exactness.gpu_uuids[: len(self.gpu_uuids)] != self.gpu_uuids
        ):
            raise ValueError("trusted qualification dispatch lineage differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "protocol_lock": self.protocol_lock.to_dict(),
            "content_source": self.content_source.to_dict(),
            "inventory": self.inventory.to_dict(),
            "doctor": self.doctor.to_dict(),
            "exactness_assignment": self.exactness_assignment.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            set(cls.__dataclass_fields__),
            label="trusted qualification dispatch",
        )
        for name in (
            "protocol_lock",
            "inventory",
            "doctor",
            "exactness_assignment",
        ):
            row[name] = _binding(row[name])
        row["content_source"] = FormalContentSourceBinding.from_dict(
            row["content_source"]
        )
        raw_gpu_uuids = row.pop("gpu_uuids")
        if type(raw_gpu_uuids) is not list:
            raise TypeError("trusted qualification GPUs must be an array")
        return cls(**row, gpu_uuids=tuple(raw_gpu_uuids))  # type: ignore[arg-type]


def load_trusted_qualification_dispatch_authority(
    path: str | Path,
) -> TrustedQualificationDispatchAuthority:
    binding = CanonicalJsonProofBinding.bind(path)
    authority = TrustedQualificationDispatchAuthority.from_dict(binding.reopen())
    if authority.sha256 != binding.semantic_sha256:
        raise ValueError("trusted qualification dispatch binding differs")
    return authority


@dataclass(frozen=True)
class TrustedQualificationLaunchEntry:
    """One backend/topology launch bound before plan materialization."""

    suite_id: TrustedPreflightQualificationSuite
    backend: Literal["DFLASH", "DSPARK"]
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    gpu_uuids: tuple[str, ...]
    dispatch_authority: CanonicalJsonProofBinding
    launch_manifest: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            self.suite_id not in TRUSTED_PREFLIGHT_QUALIFICATION_SUITES
            or self.backend != _SUITE_ALGORITHM[self.suite_id]
            or self.topology_mode != _SUITE_TOPOLOGY[self.suite_id]
            or not self.gpu_uuids
            or len(set(self.gpu_uuids)) != len(self.gpu_uuids)
            or type(self.dispatch_authority) is not CanonicalJsonProofBinding
            or type(self.launch_manifest) is not CanonicalJsonProofBinding
        ):
            raise ValueError("trusted qualification launch entry differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "backend": self.backend,
            "topology_mode": self.topology_mode,
            "gpu_uuids": list(self.gpu_uuids),
            "dispatch_authority": self.dispatch_authority.to_dict(),
            "launch_manifest": self.launch_manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            set(cls.__dataclass_fields__),
            label="trusted qualification launch entry",
        )
        raw_gpu_uuids = row.pop("gpu_uuids")
        if type(raw_gpu_uuids) is not list:
            raise TypeError("trusted qualification launch GPUs must be an array")
        dispatch = _binding(row.pop("dispatch_authority"))
        launch = _binding(row.pop("launch_manifest"))
        return cls(
            **row,
            gpu_uuids=tuple(raw_gpu_uuids),
            dispatch_authority=dispatch,
            launch_manifest=launch,
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedQualificationLaunchIndex:
    """Exact source-owned DFlash/DSpark topology launch set."""

    schema_version: Literal[1]
    kind: Literal["trusted_single_operator_preflight_qualification_launch_index"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_no_signature"]
    formal_measurement: Literal[False]
    protocol_lock: CanonicalJsonProofBinding
    content_source: FormalContentSourceBinding
    inventory: CanonicalJsonProofBinding
    doctor: CanonicalJsonProofBinding
    exactness_assignment: CanonicalJsonProofBinding
    entries: tuple[TrustedQualificationLaunchEntry, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind
            != "trusted_single_operator_preflight_qualification_launch_index"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_no_signature"
            or self.formal_measurement is not False
            or tuple(row.suite_id for row in self.entries)
            != TRUSTED_PREFLIGHT_QUALIFICATION_SUITES
        ):
            raise ValueError("trusted qualification launch index differs")
        lock = protocol_lock_from_dict(self.protocol_lock.reopen())
        bundle = self.content_source.reopen()
        inventory = GpuInventory.from_dict(self.inventory.reopen())
        exactness = ExactnessPreflightAssignment.load(
            self.exactness_assignment.absolute_path
        )
        if (
            lock.schema_version != 5
            or lock.sha256 != self.protocol_lock.semantic_sha256
            or lock.trusted_single_operator_content_bundle_sha256
            != self.content_source.content_sha256
            or bundle.runtime_binding_status != "BOUND"
            or bundle.runtime_observations is None
            or type(self.doctor.reopen()) is not dict
            or inventory.sha256 != self.inventory.semantic_sha256
            or exactness.inventory_sha256 != inventory.sha256
            or len(exactness.gpu_uuids) != 2
        ):
            raise ValueError("trusted qualification launch sources differ")
        for row in self.entries:
            dispatch = load_trusted_qualification_dispatch_authority(
                row.dispatch_authority.absolute_path
            )
            launch = CompileLaunchManifest.load(row.launch_manifest.absolute_path)
            config = load_run_config(launch.run_config_path)
            tp, dp = _topology_shape(row.topology_mode)
            drafter_matches = tuple(
                member
                for member in bundle.model_members
                if member.sha256 == launch.drafter_content_member_id
                and member.role == "drafter"
                and member.model_id == launch.drafter_model_id
                and member.revision == launch.drafter_revision
                and member.local_snapshot_path == launch.drafter_snapshot_path
                and any(
                    binding.stage == "preflight"
                    and binding.target_model_id == launch.target_model_id
                    and binding.backend == row.backend
                    and binding.draft_depth == config.model.draft_depth
                    for binding in member.runtime_bindings
                )
            )
            if (
                dispatch.protocol_lock != self.protocol_lock
                or dispatch.content_source != self.content_source
                or dispatch.inventory != self.inventory
                or dispatch.doctor != self.doctor
                or dispatch.exactness_assignment != self.exactness_assignment
                or dispatch.suite_id != row.suite_id
                or dispatch.topology_mode != row.topology_mode
                or dispatch.gpu_uuids != row.gpu_uuids
                or launch.schema_version != 2
                or launch.formal_stage != "preflight"
                or launch.content_source_binding != self.content_source
                or launch.inventory_sha256 != inventory.sha256
                or launch.gpu_uuids != row.gpu_uuids
                or config.model.algorithm != row.backend
                or config.runtime.tensor_parallel_size != tp
                or config.runtime.data_parallel_size != dp
                or len(drafter_matches) != 1
                or (
                    row.topology_mode == "tp1_dp1"
                    and config.runtime.distributed_capability_receipt_sha256 is not None
                )
                or (
                    row.topology_mode != "tp1_dp1"
                    and config.runtime.distributed_capability_receipt_sha256
                    != dispatch.sha256
                )
            ):
                raise ValueError(
                    "trusted qualification backend/topology launch differs"
                )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "trust_mode": self.trust_mode,
            "formal_measurement": self.formal_measurement,
            "protocol_lock": self.protocol_lock.to_dict(),
            "content_source": self.content_source.to_dict(),
            "inventory": self.inventory.to_dict(),
            "doctor": self.doctor.to_dict(),
            "exactness_assignment": self.exactness_assignment.to_dict(),
            "entries": [row.to_dict() for row in self.entries],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            set(cls.__dataclass_fields__),
            label="trusted qualification launch index",
        )
        for name in ("protocol_lock", "inventory", "doctor", "exactness_assignment"):
            row[name] = _binding(row[name])
        row["content_source"] = FormalContentSourceBinding.from_dict(
            row["content_source"]
        )
        raw_entries = row.pop("entries")
        if type(raw_entries) is not list:
            raise TypeError("trusted qualification launch entries must be an array")
        return cls(
            **row,
            entries=tuple(
                TrustedQualificationLaunchEntry.from_dict(item) for item in raw_entries
            ),
        )  # type: ignore[arg-type]


def load_trusted_qualification_launch_index(
    path: str | Path,
) -> TrustedQualificationLaunchIndex:
    binding = CanonicalJsonProofBinding.bind(path)
    index = TrustedQualificationLaunchIndex.from_dict(binding.reopen())
    if index.sha256 != binding.semantic_sha256:
        raise ValueError("trusted qualification launch index binding differs")
    return index


@dataclass(frozen=True)
class FormalSingleOperatorPreflightQualificationPlan:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_preflight_qualification_plan"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_no_signature"]
    formal_measurement: Literal[False]
    protocol_lock: CanonicalJsonProofBinding
    content_source: FormalContentSourceBinding
    inventory: CanonicalJsonProofBinding
    doctor: CanonicalJsonProofBinding
    exactness_assignment: CanonicalJsonProofBinding
    dispatch_authority: CanonicalJsonProofBinding
    launch_manifest: CanonicalJsonProofBinding
    suite_id: TrustedPreflightQualificationSuite
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    expected_test_names: tuple[str, ...]
    gpu_uuids: tuple[str, ...]
    topology_sha256: str
    run_nonce_sha256: str
    evidence_directory: str
    assignment_path: str
    stdout_path: str
    stderr_path: str
    junit_path: str
    observation_path: str
    result_path: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_preflight_qualification_plan"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_no_signature"
            or self.formal_measurement is not False
            or self.suite_id not in TRUSTED_PREFLIGHT_QUALIFICATION_SUITES
            or self.topology_mode != _SUITE_TOPOLOGY[self.suite_id]
            or self.expected_test_names != NATIVE_RUNTIME_GPU_TEST_NAMES[self.suite_id]
        ):
            raise ValueError("trusted preflight qualification plan schema differs")
        lock = protocol_lock_from_dict(self.protocol_lock.reopen())
        if (
            lock.schema_version != 5
            or lock.sha256 != self.protocol_lock.semantic_sha256
            or lock.trusted_single_operator_content_bundle_sha256
            != self.content_source.content_sha256
        ):
            raise ValueError("qualification ProtocolLock/content differs")
        bundle = self.content_source.reopen()
        inventory = GpuInventory.from_dict(self.inventory.reopen())
        doctor = self.doctor.reopen()
        exactness = ExactnessPreflightAssignment.load(
            self.exactness_assignment.absolute_path
        )
        dispatch = load_trusted_qualification_dispatch_authority(
            self.dispatch_authority.absolute_path
        )
        launch = CompileLaunchManifest.load(self.launch_manifest.absolute_path)
        config = load_run_config(launch.run_config_path)
        tp, dp = _topology_shape(self.topology_mode)
        if (
            bundle.runtime_binding_status != "BOUND"
            or bundle.runtime_observations is None
            or type(doctor) is not dict
            or inventory.sha256 != self.inventory.semantic_sha256
            or exactness.inventory_sha256 != inventory.sha256
            or exactness.gpu_uuids[: len(self.gpu_uuids)] != self.gpu_uuids
            or dispatch.protocol_lock != self.protocol_lock
            or dispatch.content_source != self.content_source
            or dispatch.inventory != self.inventory
            or dispatch.doctor != self.doctor
            or dispatch.exactness_assignment != self.exactness_assignment
            or dispatch.suite_id != self.suite_id
            or dispatch.topology_mode != self.topology_mode
            or dispatch.gpu_uuids != self.gpu_uuids
            or launch.sha256 != self.launch_manifest.semantic_sha256
            or launch.schema_version != 2
            or launch.formal_stage != "preflight"
            or launch.content_source_binding != self.content_source
            or launch.inventory_sha256 != inventory.sha256
            or launch.gpu_uuids != self.gpu_uuids
            or config.model.algorithm != _SUITE_ALGORITHM[self.suite_id]
            or config.runtime.tensor_parallel_size != tp
            or config.runtime.data_parallel_size != dp
            or (
                self.topology_mode != "tp1_dp1"
                and config.runtime.distributed_capability_receipt_sha256
                != dispatch.sha256
            )
            or (
                self.topology_mode == "tp1_dp1"
                and config.runtime.distributed_capability_receipt_sha256 is not None
            )
        ):
            raise ValueError("qualification plan runtime identity differs")
        for digest in (self.topology_sha256, self.run_nonce_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("qualification plan digest is invalid")
        evidence = Path(self.evidence_directory)
        if (
            not evidence.is_absolute()
            or evidence != evidence.resolve(strict=False)
            or not evidence.is_dir()
            or evidence.is_symlink()
        ):
            raise ValueError("qualification evidence directory is unavailable")
        for value in (
            self.assignment_path,
            self.stdout_path,
            self.stderr_path,
            self.junit_path,
            self.observation_path,
            self.result_path,
        ):
            path = Path(value)
            if path.parent != evidence or path != path.resolve(strict=False):
                raise ValueError("qualification output escaped its evidence directory")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "protocol_lock": self.protocol_lock.to_dict(),
            "content_source": self.content_source.to_dict(),
            "inventory": self.inventory.to_dict(),
            "doctor": self.doctor.to_dict(),
            "exactness_assignment": self.exactness_assignment.to_dict(),
            "dispatch_authority": self.dispatch_authority.to_dict(),
            "launch_manifest": self.launch_manifest.to_dict(),
            "expected_test_names": list(self.expected_test_names),
            "gpu_uuids": list(self.gpu_uuids),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(value, set(cls.__dataclass_fields__), label="qualification plan")
        for name in (
            "protocol_lock",
            "inventory",
            "doctor",
            "exactness_assignment",
            "dispatch_authority",
            "launch_manifest",
        ):
            row[name] = _binding(row[name])
        row["content_source"] = FormalContentSourceBinding.from_dict(
            row["content_source"]
        )
        for name in ("expected_test_names", "gpu_uuids"):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(f"qualification plan {name} must be an array")
            row[name] = tuple(raw)
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightQualificationPlanIndex:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_preflight_qualification_plan_index"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_no_signature"]
    formal_measurement: Literal[False]
    protocol_lock_sha256: str
    exactness_cell_id: str
    plans: tuple[CanonicalJsonProofBinding, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_preflight_qualification_plan_index"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_no_signature"
            or self.formal_measurement is not False
            or len(self.plans) != len(TRUSTED_PREFLIGHT_QUALIFICATION_SUITES)
        ):
            raise ValueError("qualification plan index schema differs")
        values = tuple(
            load_formal_single_operator_preflight_qualification_plan(row.absolute_path)
            for row in self.plans
        )
        if (
            tuple(row.suite_id for row in values)
            != TRUSTED_PREFLIGHT_QUALIFICATION_SUITES
            or any(
                row.protocol_lock.semantic_sha256 != self.protocol_lock_sha256
                for row in values
            )
            or any(
                ExactnessPreflightAssignment.load(
                    row.exactness_assignment.absolute_path
                ).cell_id
                != self.exactness_cell_id
                for row in values
            )
        ):
            raise ValueError("qualification plan index coverage differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "plans": [row.to_dict() for row in self.plans],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(value, set(cls.__dataclass_fields__), label="qualification index")
        raw = row.pop("plans")
        if type(raw) is not list:
            raise TypeError("qualification plans must be an array")
        return cls(
            **row,
            plans=tuple(CanonicalJsonProofBinding.from_dict(item) for item in raw),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorPreflightQualificationResult:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_preflight_qualification_result"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_no_signature"]
    formal_measurement: Literal[False]
    plan: CanonicalJsonProofBinding
    assignment: CanonicalJsonProofBinding
    before_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    stdout: EvidenceFileBinding
    stderr: EvidenceFileBinding
    junit_xml: EvidenceFileBinding
    live_observation: CanonicalJsonProofBinding
    live_native_terminal: CanonicalJsonProofBinding
    live_native_itl: CanonicalJsonProofBinding
    live_graph: CanonicalJsonProofBinding
    live_worker_hook: CanonicalJsonProofBinding
    live_rank_terminals: tuple[CanonicalJsonProofBinding, ...]
    live_server_receipt: CanonicalJsonProofBinding
    live_server_log: EvidenceFileBinding
    empirical_proof: CanonicalJsonProofBinding
    runner_terminal: CanonicalJsonProofBinding
    started_ns: int
    finished_ns: int
    status: Literal["COMPLETE"]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "plan",
            "assignment",
            "before_gpu_snapshot",
            "after_gpu_snapshot",
            "live_observation",
            "live_native_terminal",
            "live_native_itl",
            "live_graph",
            "live_worker_hook",
            "live_server_receipt",
            "empirical_proof",
            "runner_terminal",
        ):
            value[name] = getattr(self, name).to_dict()
        for name in ("stdout", "stderr", "junit_xml", "live_server_log"):
            value[name] = getattr(self, name).to_dict()
        value["live_rank_terminals"] = [
            row.to_dict() for row in self.live_rank_terminals
        ]
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value, set(cls.__dataclass_fields__), label="qualification result"
        )
        for name in (
            "plan",
            "assignment",
            "before_gpu_snapshot",
            "after_gpu_snapshot",
            "live_observation",
            "live_native_terminal",
            "live_native_itl",
            "live_graph",
            "live_worker_hook",
            "live_server_receipt",
            "empirical_proof",
            "runner_terminal",
        ):
            row[name] = _binding(row[name])
        for name in ("stdout", "stderr", "junit_xml", "live_server_log"):
            row[name] = EvidenceFileBinding.from_dict(
                row[name], label=f"qualification {name}"
            )
        ranks = row.pop("live_rank_terminals")
        if type(ranks) is not list:
            raise TypeError("qualification rank terminals must be an array")
        return cls(
            **row,
            live_rank_terminals=tuple(_binding(item) for item in ranks),
        )  # type: ignore[arg-type]


class FormalSingleOperatorPreflightQualificationBlocked(RuntimeError):
    pass


def load_formal_single_operator_preflight_qualification_plan(
    path: str | Path,
) -> FormalSingleOperatorPreflightQualificationPlan:
    binding = CanonicalJsonProofBinding.bind(path)
    plan = FormalSingleOperatorPreflightQualificationPlan.from_dict(binding.reopen())
    if plan.sha256 != binding.semantic_sha256:
        raise ValueError("qualification plan binding differs")
    return plan


def load_formal_single_operator_preflight_qualification_plan_index(
    path: str | Path,
) -> FormalSingleOperatorPreflightQualificationPlanIndex:
    binding = CanonicalJsonProofBinding.bind(path)
    index = FormalSingleOperatorPreflightQualificationPlanIndex.from_dict(
        binding.reopen()
    )
    if index.sha256 != binding.semantic_sha256:
        raise ValueError("qualification plan index binding differs")
    return index


def _qualification_config(
    *,
    base: RunConfig,
    suite_id: TrustedPreflightQualificationSuite,
    gpu_uuids: tuple[str, ...],
    distributed_receipt_sha256: str,
    drafter_model_id: str,
    drafter_revision: str,
    draft_depth: int,
) -> RunConfig:
    value = base.model_dump(mode="python")
    value["model"]["algorithm"] = _SUITE_ALGORITHM[suite_id]
    value["model"]["drafter"] = drafter_model_id
    value["model"]["drafter_revision"] = drafter_revision
    value["model"]["draft_depth"] = draft_depth
    value["runtime"]["speculative_num_draft_tokens"] = draft_depth + 1
    tp, dp = _topology_shape(_SUITE_TOPOLOGY[suite_id])
    runtime = value["runtime"]
    runtime.update(
        {
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "tp_rank": 0,
            "dp_rank": 0,
            "device_identity": ",".join(gpu_uuids),
            "router_identity": (
                "preflight-qualified-sticky-router-v1" if dp == 2 else "single-replica"
            ),
        }
    )
    topology = _SUITE_TOPOLOGY[suite_id]
    if topology == "tp1_dp1":
        runtime.update(
            {
                "distributed_runtime_capability": "single_rank",
                "distributed_release_capability_sha256": None,
                "distributed_capability_receipt_sha256": None,
                "process_group_backend": "nccl",
            }
        )
    else:
        capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[topology]
        runtime.update(
            {
                "distributed_runtime_capability": "patched_two_gpu_v1",
                "distributed_release_capability_sha256": capability.sha256,
                "distributed_capability_receipt_sha256": (distributed_receipt_sha256),
                "process_group_backend": capability.process_group_backend,
            }
        )
    return RunConfig.model_validate(value)


def _derive_launch(
    *,
    root: Path,
    suite_id: TrustedPreflightQualificationSuite,
    environment_base: CompileLaunchManifest,
    doctor: dict[str, object],
    gpu_uuids: tuple[str, ...],
    shared_cache_root: Path,
    protocol_lock_sha256: str,
    distributed_receipt_sha256: str,
    drafter: object,
) -> CanonicalJsonProofBinding:
    # Importing here avoids making formal_preflight_inputs depend on this
    # module during import-time protocol construction.
    from lightcone_spec.experiments.formal_preflight_inputs import _compile_key

    row_root = root / suite_id
    row_root.mkdir(mode=0o700, exist_ok=True)
    base_config = load_run_config(environment_base.run_config_path)
    config = _qualification_config(
        base=base_config,
        suite_id=suite_id,
        gpu_uuids=gpu_uuids,
        distributed_receipt_sha256=distributed_receipt_sha256,
        drafter_model_id=drafter.model_id,
        drafter_revision=drafter.revision,
        draft_depth=next(
            row.draft_depth
            for row in drafter.runtime_bindings
            if row.stage == "preflight"
            and row.target_model_id == environment_base.target_model_id
            and row.backend == _SUITE_ALGORITHM[suite_id]
        ),
    )
    key = _compile_key(doctor=doctor, config=config, gpu_uuid=gpu_uuids[0])
    cache = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=shared_cache_root,
        cache_mode="build",
    )
    cache_path = row_root / "compile-cache-plan.json"
    cache.write(cache_path)
    roots = {
        environment_base.target_model_id: environment_base.target_snapshot_path,
        drafter.model_id: drafter.local_snapshot_path,
    }
    rendered = _render_server(
        output=row_root,
        method="static",
        config=config,
        verified_checkout=Path(environment_base.patched_sglang_checkout),
        roots={
            key: value
            for key, value in roots.items()
            if key is not None and value is not None
        },
        target_id=environment_base.target_model_id,
        drafter_id=drafter.model_id,
        adaptation_reserve_mb=0,
        mem_fraction_static=0.75,
        host="127.0.0.1",
        port=_SUITE_PORTS[suite_id],
        compile_cache_plan_path=cache_path,
    )
    model_lock = ModelLock(
        schema_version=2,
        models=tuple(
            sorted(
                (
                    LockedModel(
                        environment_base.target_model_id,
                        environment_base.target_revision,
                    ),
                    LockedModel(drafter.model_id, drafter.revision),
                ),
                key=lambda row: row.model_id,
            )
        ),
    )
    model_lock.write(row_root / "model-lock.json")
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=environment_base.sampling_profile_sha256,
        payloads=(CompileOnlyPrewarmPayload("graph-bucket-1", 1, (1, 2), 1, 1),),
    )
    prewarm_path = write_compile_prewarm_manifest(prewarm, row_root / "prewarm.json")
    config_path = Path(rendered.run_config).resolve()
    launch = CompileLaunchManifest(
        schema_version=2,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=environment_base.protocol_sha256,
        patched_sglang_checkout=environment_base.patched_sglang_checkout,
        patched_sglang_commit=environment_base.patched_sglang_commit,
        patched_sglang_tree=environment_base.patched_sglang_tree,
        run_config_path=str(config_path),
        run_config_raw_sha256=_raw_sha256(config_path),
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=_raw_sha256(cache_path),
        compile_cache_plan_sha256=cache.sha256,
        prewarm_manifest_path=str(prewarm_path),
        prewarm_manifest_raw_sha256=_raw_sha256(prewarm_path),
        prewarm_manifest_sha256=prewarm.sha256,
        sampling_profile_path=environment_base.sampling_profile_path,
        sampling_profile_raw_sha256=(environment_base.sampling_profile_raw_sha256),
        sampling_profile_sha256=environment_base.sampling_profile_sha256,
        prepared_model_content_manifest_path=(
            environment_base.prepared_model_content_manifest_path
        ),
        prepared_model_content_manifest_raw_sha256=(
            environment_base.prepared_model_content_manifest_raw_sha256
        ),
        prepared_model_content_manifest_sha256=(
            environment_base.prepared_model_content_manifest_sha256
        ),
        prepared_model_content_manifest_size=(
            environment_base.prepared_model_content_manifest_size
        ),
        target_content_member_id=environment_base.target_content_member_id,
        target_model_id=environment_base.target_model_id,
        target_snapshot_path=environment_base.target_snapshot_path,
        target_revision=environment_base.target_revision,
        target_content_authority_sha256=None,
        drafter_content_member_id=drafter.sha256,
        drafter_model_id=drafter.model_id,
        drafter_snapshot_path=drafter.local_snapshot_path,
        drafter_revision=drafter.revision,
        drafter_content_authority_sha256=None,
        tokenizer_content_member_id=environment_base.tokenizer_content_member_id,
        tokenizer_model_id=environment_base.tokenizer_model_id,
        tokenizer_snapshot_path=environment_base.tokenizer_snapshot_path,
        tokenizer_revision=environment_base.tokenizer_revision,
        tokenizer_content_authority_sha256=None,
        model_lock_sha256=model_lock.sha256,
        server_argv=rendered.argv,
        server_argv_sha256=content_sha256({"argv": list(rendered.argv)}),
        localhost_port=_SUITE_PORTS[suite_id],
        physical_assignment_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_preflight_qualification_physical_assignment",
                "protocol_lock_sha256": protocol_lock_sha256,
                "suite_id": suite_id,
                "gpu_uuids": list(gpu_uuids),
            }
        ),
        experiment_budget_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_preflight_qualification_budget",
                "suite_id": suite_id,
                "timeout_seconds": _QUALIFICATION_TIMEOUT_SECONDS,
                "physical_execution_count": 1,
            }
        ),
        budget_materialization_authority_sha256=protocol_lock_sha256,
        inventory_sha256=environment_base.inventory_sha256,
        gpu_uuids=gpu_uuids,
        path_entries=environment_base.path_entries,
        library_path_entries=environment_base.library_path_entries,
        cuda_home=environment_base.cuda_home,
        formal_stage="preflight",
        content_source_binding=environment_base.content_source_binding,
    )
    launch_path = row_root / "compile-launch.json"
    launch.write(launch_path)
    return CanonicalJsonProofBinding.bind(launch_path, semantic_sha256=launch.sha256)


def publish_formal_single_operator_preflight_qualification_launch_index(
    *,
    protocol_lock_path: str | Path,
    content_source_path: str | Path,
    inventory_path: str | Path,
    doctor_report_path: str | Path,
    exactness_assignment_path: str | Path,
    base_tp1_launch_path: str | Path,
    base_tp2_launch_path: str | Path,
    output_root: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish the exact backend/topology launch set from immutable paths."""

    root = _absolute_new_directory(output_root)
    lock_binding = CanonicalJsonProofBinding.bind(protocol_lock_path)
    lock = protocol_lock_from_dict(lock_binding.reopen())
    content = FormalContentSourceBinding.bind_trusted_single_operator(
        str(content_source_path)
    )
    bundle = content.reopen()
    inventory_binding = CanonicalJsonProofBinding.bind(inventory_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    doctor_binding = CanonicalJsonProofBinding.bind(doctor_report_path)
    doctor = doctor_binding.reopen()
    exact_binding = CanonicalJsonProofBinding.bind(exactness_assignment_path)
    exactness = ExactnessPreflightAssignment.load(exactness_assignment_path)
    tp1 = CompileLaunchManifest.load(base_tp1_launch_path)
    tp2 = CompileLaunchManifest.load(base_tp2_launch_path)
    if (
        lock.schema_version != 5
        or lock.trusted_single_operator_content_bundle_sha256 != content.content_sha256
        or bundle.runtime_binding_status != "BOUND"
        or bundle.runtime_observations is None
        or type(doctor) is not dict
        or inventory.sha256 != inventory_binding.semantic_sha256
        or exactness.inventory_sha256 != inventory.sha256
        or len(exactness.gpu_uuids) != 2
        or tp1.content_source_binding != content
        or tp2.content_source_binding != content
        or tp1.inventory_sha256 != inventory.sha256
        or tp2.inventory_sha256 != inventory.sha256
        or len(tp1.gpu_uuids) != 1
        or tp1.gpu_uuids != exactness.gpu_uuids[:1]
        or tp2.gpu_uuids != exactness.gpu_uuids
        or load_run_config(tp1.run_config_path).runtime.topology_mode != "tp1_dp1"
        or load_run_config(tp2.run_config_path).runtime.topology_mode != "tp2_dp1"
    ):
        raise ValueError("qualification frozen sources do not form one runtime")
    drafter_by_backend = {}
    for backend in ("DFLASH", "DSPARK"):
        matches = tuple(
            member
            for member in bundle.model_members
            if member.role == "drafter"
            and "preflight" in member.stages
            and any(
                binding.stage == "preflight"
                and binding.target_model_id == tp1.target_model_id
                and binding.backend == backend
                for binding in member.runtime_bindings
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"trusted preflight lacks one exact {backend} drafter binding"
            )
        drafter_by_backend[backend] = matches[0]
    dflash = drafter_by_backend["DFLASH"]
    if (
        tp1.drafter_content_member_id != dflash.sha256
        or tp2.drafter_content_member_id != dflash.sha256
        or tp1.drafter_model_id != dflash.model_id
        or tp2.drafter_model_id != dflash.model_id
        or tp1.drafter_revision != dflash.revision
        or tp2.drafter_revision != dflash.revision
    ):
        raise ValueError("base DFlash launches differ from typed content binding")
    shared_cache_root = CompileCacheLaunchPlan.load(
        tp2.compile_cache_plan_path
    ).cache_root
    dispatches: dict[str, CanonicalJsonProofBinding] = {}
    launches: dict[str, CanonicalJsonProofBinding] = {}
    for suite_id in TRUSTED_PREFLIGHT_QUALIFICATION_SUITES:
        topology = _SUITE_TOPOLOGY[suite_id]
        gpu_uuids = (
            exactness.gpu_uuids if topology != "tp1_dp1" else exactness.gpu_uuids[:1]
        )
        dispatch = TrustedQualificationDispatchAuthority(
            schema_version=1,
            kind="trusted_single_operator_preflight_qualification_dispatch",
            protocol_sha256=(
                FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
            ),
            trust_mode="trusted_single_operator_no_signature",
            formal_measurement=False,
            signature_status="NOT_APPLICABLE",
            protocol_lock=lock_binding,
            content_source=content,
            inventory=inventory_binding,
            doctor=doctor_binding,
            exactness_assignment=exact_binding,
            suite_id=suite_id,
            topology_mode=topology,  # type: ignore[arg-type]
            gpu_uuids=gpu_uuids,
            source_capability_sha256=_source_capability_sha256(suite_id),
        )
        dispatch_path = root / suite_id / "dispatch.json"
        dispatch_path.parent.mkdir(mode=0o700, exist_ok=True)
        publish_canonical_json_no_replace(dispatch_path, dispatch.to_dict())
        dispatches[suite_id] = CanonicalJsonProofBinding.bind(
            dispatch_path, semantic_sha256=dispatch.sha256
        )
        launches[suite_id] = _derive_launch(
            root=root,
            suite_id=suite_id,
            environment_base=tp1 if topology == "tp1_dp1" else tp2,
            doctor=doctor,
            gpu_uuids=gpu_uuids,
            shared_cache_root=Path(shared_cache_root),
            protocol_lock_sha256=lock.sha256,
            distributed_receipt_sha256=dispatch.sha256,
            drafter=drafter_by_backend[_SUITE_ALGORITHM[suite_id]],
        )
    launch_index = TrustedQualificationLaunchIndex(
        schema_version=1,
        kind="trusted_single_operator_preflight_qualification_launch_index",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
        ),
        trust_mode="trusted_single_operator_no_signature",
        formal_measurement=False,
        protocol_lock=lock_binding,
        content_source=content,
        inventory=inventory_binding,
        doctor=doctor_binding,
        exactness_assignment=exact_binding,
        entries=tuple(
            TrustedQualificationLaunchEntry(
                suite_id=suite_id,
                backend=_SUITE_ALGORITHM[suite_id],  # type: ignore[arg-type]
                topology_mode=_SUITE_TOPOLOGY[suite_id],  # type: ignore[arg-type]
                gpu_uuids=(
                    exactness.gpu_uuids
                    if _SUITE_TOPOLOGY[suite_id] != "tp1_dp1"
                    else exactness.gpu_uuids[:1]
                ),
                dispatch_authority=dispatches[suite_id],
                launch_manifest=launches[suite_id],
            )
            for suite_id in TRUSTED_PREFLIGHT_QUALIFICATION_SUITES
        ),
    )
    index_path = root / "launch-index.json"
    publish_canonical_json_no_replace(index_path, launch_index.to_dict())
    rebound = load_trusted_qualification_launch_index(index_path)
    if rebound != launch_index:
        raise RuntimeError("trusted qualification launch index changed")
    return CanonicalJsonProofBinding.bind(
        index_path,
        semantic_sha256=launch_index.sha256,
    )


def materialize_formal_single_operator_preflight_qualification_plans(
    *,
    qualification_launch_index_path: str | Path,
    output_root: str | Path,
) -> CanonicalJsonProofBinding:
    """Materialize exact-six plans from one deeply validated launch index."""

    launch_index = load_trusted_qualification_launch_index(
        qualification_launch_index_path
    )
    root = _absolute_new_directory(output_root)
    lock_binding = launch_index.protocol_lock
    lock = protocol_lock_from_dict(lock_binding.reopen())
    content = launch_index.content_source
    inventory_binding = launch_index.inventory
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    doctor_binding = launch_index.doctor
    exact_binding = launch_index.exactness_assignment
    exactness = ExactnessPreflightAssignment.load(exact_binding.absolute_path)
    dispatches = {row.suite_id: row.dispatch_authority for row in launch_index.entries}
    launches = {row.suite_id: row.launch_manifest for row in launch_index.entries}
    plans = []
    for suite_id in TRUSTED_PREFLIGHT_QUALIFICATION_SUITES:
        evidence = root / suite_id / "evidence"
        evidence.mkdir(mode=0o700, parents=True)
        topology = _SUITE_TOPOLOGY[suite_id]
        gpu_uuids = (
            exactness.gpu_uuids if topology != "tp1_dp1" else exactness.gpu_uuids[:1]
        )
        topology_sha256 = content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_preflight_qualification_topology",
                "inventory_sha256": inventory.sha256,
                "topology_mode": topology,
                "gpu_uuids": list(gpu_uuids),
            }
        )
        run_nonce = content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_preflight_qualification_run_nonce",
                "protocol_lock_sha256": lock.sha256,
                "exactness_assignment_sha256": exactness.sha256,
                "suite_id": suite_id,
                "launch_manifest_sha256": launches[suite_id].semantic_sha256,
            }
        )
        plan = FormalSingleOperatorPreflightQualificationPlan(
            schema_version=1,
            kind="formal_single_operator_preflight_qualification_plan",
            protocol_sha256=(
                FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
            ),
            trust_mode="trusted_single_operator_no_signature",
            formal_measurement=False,
            protocol_lock=lock_binding,
            content_source=content,
            inventory=inventory_binding,
            doctor=doctor_binding,
            exactness_assignment=exact_binding,
            dispatch_authority=dispatches[suite_id],
            launch_manifest=launches[suite_id],
            suite_id=suite_id,
            topology_mode=topology,  # type: ignore[arg-type]
            expected_test_names=NATIVE_RUNTIME_GPU_TEST_NAMES[suite_id],
            gpu_uuids=gpu_uuids,
            topology_sha256=topology_sha256,
            run_nonce_sha256=run_nonce,
            evidence_directory=str(evidence),
            assignment_path=str(evidence / "assignment.json"),
            stdout_path=str(evidence / "pytest.stdout.log"),
            stderr_path=str(evidence / "pytest.stderr.log"),
            junit_path=str(evidence / "junit.xml"),
            observation_path=str(evidence / "live-observation.json"),
            result_path=str(evidence / "result.json"),
        )
        plan_path = root / suite_id / "plan.json"
        publish_canonical_json_no_replace(plan_path, plan.to_dict())
        plans.append(
            CanonicalJsonProofBinding.bind(plan_path, semantic_sha256=plan.sha256)
        )
    index = FormalSingleOperatorPreflightQualificationPlanIndex(
        schema_version=1,
        kind="formal_single_operator_preflight_qualification_plan_index",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256,
        trust_mode="trusted_single_operator_no_signature",
        formal_measurement=False,
        protocol_lock_sha256=lock.sha256,
        exactness_cell_id=exactness.cell_id,
        plans=tuple(plans),
    )
    index_path = root / "plan-index.json"
    publish_canonical_json_no_replace(index_path, index.to_dict())
    return CanonicalJsonProofBinding.bind(index_path, semantic_sha256=index.sha256)


def _live_bindings(
    plan: FormalSingleOperatorPreflightQualificationPlan,
    assignment: NativeRuntimeQualificationAssignment,
) -> tuple[
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    tuple[CanonicalJsonProofBinding, ...],
    CanonicalJsonProofBinding,
    EvidenceFileBinding,
]:
    root = Path(plan.evidence_directory)
    suite = plan.suite_id
    return (
        CanonicalJsonProofBinding.bind(root / f"{suite}.live-native-terminal.json"),
        CanonicalJsonProofBinding.bind(root / f"{suite}.live-native-itl.json"),
        CanonicalJsonProofBinding.bind(root / f"{suite}.live-graph.json"),
        CanonicalJsonProofBinding.bind(root / f"{suite}.live-worker-hook.json"),
        tuple(
            CanonicalJsonProofBinding.bind(root / f"{suite}.rank-{rank}.json")
            for rank in range(len(assignment.gpu_uuids))
        ),
        CanonicalJsonProofBinding.bind(root / f"{suite}.live-server-receipt.json"),
        EvidenceFileBinding.bind(
            Path(plan.observation_path).with_suffix(".live-server.log"),
            label="qualification live server log",
        ),
    )


def _build_assignment(
    plan_binding: CanonicalJsonProofBinding,
    plan: FormalSingleOperatorPreflightQualificationPlan,
) -> NativeRuntimeQualificationAssignment:
    lock = protocol_lock_from_dict(plan.protocol_lock.reopen())
    inventory = GpuInventory.from_dict(plan.inventory.reopen())
    exactness = ExactnessPreflightAssignment.load(
        plan.exactness_assignment.absolute_path
    )
    python = EvidenceFileBinding.bind(
        exactness.python_executable, label="qualification Python"
    )
    nvidia = EvidenceFileBinding.bind(
        exactness.nvidia_smi_executable, label="qualification nvidia-smi"
    )
    exact_result = CanonicalJsonProofBinding.bind(exactness.result_pointer_path)
    devices = {row.uuid: row for row in inventory.devices}
    return NativeRuntimeQualificationAssignment(
        schema_version=2,
        kind="formal_native_runtime_gpu_qualification_assignment",
        suite_id=plan.suite_id,
        runner_protocol_sha256=_runner_protocol(plan.suite_id),
        registry_sha256=lock.registry_sha256,
        runtime_sha256=lock.formal_runtime_authority_manifest_sha256,
        topology_sha256=plan.topology_sha256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=exactness.hardware_envelope_sha256,
        run_nonce_sha256=plan.run_nonce_sha256,
        gpu_uuids=plan.gpu_uuids,
        gpu_models=tuple(devices[uuid].model for uuid in plan.gpu_uuids),
        launch_manifest=plan.launch_manifest,
        base_exactness_result_pointer=(
            exact_result
            if plan.suite_id in DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS
            else None
        ),
        eagle3_selector_status=None,
        eagle3_compatibility_authority_sha256=None,
        eagle3_model_selector_sha256=None,
        python_executable=python.absolute_path,
        python_executable_raw_sha256=python.raw_sha256,
        python_executable_size=python.size,
        nvidia_smi_executable=nvidia.absolute_path,
        nvidia_smi_raw_sha256=nvidia.raw_sha256,
        nvidia_smi_size=nvidia.size,
        evidence_directory=plan.evidence_directory,
        trusted_single_operator_authority=plan_binding,
    )


def _publish_failed_terminal(
    *,
    plan: FormalSingleOperatorPreflightQualificationPlan,
    assignment: NativeRuntimeQualificationAssignment,
    started_ns: int,
    error_code: str,
    exit_code: int | None,
) -> None:
    path = Path(plan.evidence_directory) / "failed-terminal.json"
    publish_canonical_json_no_replace(
        path,
        {
            "schema_version": 1,
            "kind": "formal_single_operator_preflight_qualification_failed_terminal",
            "protocol_sha256": (
                FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
            ),
            "trust_mode": "trusted_single_operator_no_signature",
            "formal_measurement": False,
            "suite_id": plan.suite_id,
            "assignment_sha256": assignment.sha256,
            "status": "FAILED",
            "error_code": error_code,
            "exit_code": exit_code,
            "started_ns": started_ns,
            "finished_ns": max(time.monotonic_ns(), started_ns + 1),
        },
    )


def execute_formal_single_operator_preflight_qualification_plan(
    plan_path: str | Path,
) -> FormalSingleOperatorPreflightQualificationResult:
    plan_binding = CanonicalJsonProofBinding.bind(plan_path)
    plan = load_formal_single_operator_preflight_qualification_plan(plan_path)
    dispatch = load_trusted_qualification_dispatch_authority(
        plan.dispatch_authority.absolute_path
    )
    exact = ExactnessPreflightResultPointer.load(
        ExactnessPreflightAssignment.load(
            plan.exactness_assignment.absolute_path
        ).result_pointer_path
    )
    if exact.schema_version != 2 or exact.junit_xml is None:
        raise FormalSingleOperatorPreflightQualificationBlocked(
            "trusted_exactness_result_not_complete"
        )
    assignment = _build_assignment(plan_binding, plan)
    assignment_binding = assignment.write(plan.assignment_path)
    before = _publish_snapshot(assignment, phase="before")
    before_value = _validate_gpu_snapshot(
        before.reopen(), assignment=assignment, phase="before"
    )
    started_ns = time.monotonic_ns()
    if before_value["status"] != "AVAILABLE" or before_value["compute_process_rows"]:
        _publish_failed_terminal(
            plan=plan,
            assignment=assignment,
            started_ns=started_ns,
            error_code="gpu_precondition_not_clean",
            exit_code=None,
        )
        raise FormalSingleOperatorPreflightQualificationBlocked(
            "gpu_precondition_not_clean"
        )
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    test_file = (
        Path(launch.patched_sglang_checkout)
        / NATIVE_RUNTIME_GPU_TEST_FILES[plan.suite_id]
    )
    command = (
        assignment.python_executable,
        "-m",
        "pytest",
        "-q",
        *tuple(f"{test_file}::{name}" for name in plan.expected_test_names),
        f"--junitxml={plan.junit_path}",
    )
    environment = launch.child_environment()
    environment.update(
        {
            "PYTHONPATH": launch.patched_sglang_checkout,
            "LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_PATH": (
                assignment_binding.absolute_path
            ),
            "LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_SHA256": assignment.sha256,
            "LIGHTCONE_NATIVE_QUALIFICATION_RUNNER_PROTOCOL_SHA256": (
                assignment.runner_protocol_sha256
            ),
            "LIGHTCONE_NATIVE_QUALIFICATION_SOURCE_CAPABILITY_SHA256": (
                _source_capability_sha256(plan.suite_id)
            ),
            "LIGHTCONE_NATIVE_QUALIFICATION_TRUSTED_AUTHORITY_PATH": (
                plan_binding.absolute_path
            ),
            "LIGHTCONE_NATIVE_QUALIFICATION_TRUSTED_AUTHORITY_SHA256": plan.sha256,
            "LIGHTCONE_NATIVE_QUALIFICATION_OBSERVATION_PATH": plan.observation_path,
            "LIGHTCONE_COMPILE_LAUNCH_MANIFEST_PATH": (
                plan.launch_manifest.absolute_path
            ),
            "LIGHTCONE_COMPILE_LAUNCH_MANIFEST_SHA256": (
                plan.launch_manifest.semantic_sha256
            ),
        }
    )
    process: subprocess.Popen[bytes] | None = None
    error_code: str | None = None
    with (
        Path(plan.stdout_path).open("xb") as stdout,
        Path(plan.stderr_path).open("xb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=launch.patched_sglang_checkout,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
        try:
            process.wait(timeout=_QUALIFICATION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            error_code = "qualification_timeout"
        if process.poll() is None or _process_group_exists(process.pid):
            empty, required_sigkill = _terminate_process_group(process)
            if error_code is None:
                error_code = (
                    "qualification_cleanup_required_sigkill"
                    if required_sigkill
                    else "qualification_cleanup_required_sigterm"
                )
            if not empty:
                error_code = "qualification_cleanup_incomplete"
        if error_code is None and process.returncode != 0:
            error_code = "qualification_pytest_failed"
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    after = _publish_snapshot(assignment, phase="after")
    after_value = _validate_gpu_snapshot(
        after.reopen(), assignment=assignment, phase="after"
    )
    if after_value["status"] != "AVAILABLE" or after_value["compute_process_rows"]:
        error_code = error_code or "gpu_postcondition_not_clean"
    if error_code is not None:
        _publish_failed_terminal(
            plan=plan,
            assignment=assignment,
            started_ns=started_ns,
            error_code=error_code,
            exit_code=process.returncode,
        )
        raise FormalSingleOperatorPreflightQualificationBlocked(error_code)
    expected = tuple(sorted(plan.expected_test_names))
    if _junit_summary(Path(plan.junit_path)) != (expected, 8, 8, 0, 0, 0):
        raise FormalSingleOperatorPreflightQualificationBlocked(
            "qualification_junit_not_exact_8_of_8"
        )
    stdout = EvidenceFileBinding.bind(plan.stdout_path, label="qualification stdout")
    stderr = EvidenceFileBinding.bind(plan.stderr_path, label="qualification stderr")
    junit = EvidenceFileBinding.bind(plan.junit_path, label="qualification JUnit")
    observation_binding = CanonicalJsonProofBinding.bind(plan.observation_path)
    observation = NativeRuntimeQualificationObservation.from_dict(
        observation_binding.reopen()
    )
    observation.validate_assignment(assignment)
    live_native, live_itl, live_graph, live_worker, ranks, server, server_log = (
        _live_bindings(plan, assignment)
    )
    if (
        observation.native_terminal_sha256 != live_native.semantic_sha256
        or observation.native_itl_pointer_sha256 != live_itl.semantic_sha256
        or observation.graph_observation_sha256 != live_graph.semantic_sha256
        or observation.worker_hook_observation_sha256 != live_worker.semantic_sha256
        or observation.rank_terminal_sha256s
        != tuple(row.semantic_sha256 for row in ranks)
        or observation.live_server_receipt_sha256 != server.semantic_sha256
    ):
        raise ValueError("qualification live observation evidence differs")
    proof_value = {
        "schema_version": 1,
        "kind": "formal_single_operator_preflight_qualification_empirical_proof",
        "protocol_sha256": FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256,
        "trust_mode": "trusted_single_operator_no_signature",
        "formal_measurement": False,
        "signature_status": "NOT_APPLICABLE",
        "suite_id": plan.suite_id,
        "plan_sha256": plan.sha256,
        "assignment_sha256": assignment.sha256,
        "protocol_lock_sha256": plan.protocol_lock.semantic_sha256,
        "content_source_sha256": plan.content_source.sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "doctor_sha256": plan.doctor.semantic_sha256,
        "dispatch_authority_sha256": dispatch.sha256,
        "launch_manifest_sha256": plan.launch_manifest.semantic_sha256,
        "server_argv_sha256": launch.server_argv_sha256,
        "gpu_uuids": list(plan.gpu_uuids),
        "junit_raw_sha256": junit.raw_sha256,
        "stdout_raw_sha256": stdout.raw_sha256,
        "stderr_raw_sha256": stderr.raw_sha256,
        "live_observation_sha256": observation.sha256,
        "live_native_terminal_sha256": live_native.semantic_sha256,
        "live_native_itl_sha256": live_itl.semantic_sha256,
        "live_server_receipt_sha256": server.semantic_sha256,
        "tests_collected": 8,
        "tests_passed": 8,
        "tests_failed": 0,
        "tests_errored": 0,
        "tests_skipped": 0,
    }
    proof_path = Path(plan.evidence_directory) / "empirical-proof.json"
    publish_canonical_json_no_replace(proof_path, proof_value)
    proof = CanonicalJsonProofBinding.bind(proof_path)
    finished_ns = max(time.monotonic_ns(), started_ns + 1)
    terminal_value = {
        "schema_version": 1,
        "kind": "formal_single_operator_preflight_qualification_terminal",
        "protocol_sha256": FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256,
        "trust_mode": "trusted_single_operator_no_signature",
        "formal_measurement": False,
        "suite_id": plan.suite_id,
        "plan_sha256": plan.sha256,
        "assignment_sha256": assignment.sha256,
        "runner_process_id": process.pid,
        "launch_argv_sha256": launch.server_argv_sha256,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "exit_code": process.returncode,
        "process_group_empty": not _process_group_exists(process.pid),
        "stdout_raw_sha256": stdout.raw_sha256,
        "stderr_raw_sha256": stderr.raw_sha256,
        "junit_raw_sha256": junit.raw_sha256,
        "empirical_proof_sha256": proof.semantic_sha256,
        "status": "COMPLETE",
    }
    terminal_path = Path(plan.evidence_directory) / "runner-terminal.json"
    publish_canonical_json_no_replace(terminal_path, terminal_value)
    terminal = CanonicalJsonProofBinding.bind(terminal_path)
    result = FormalSingleOperatorPreflightQualificationResult(
        schema_version=1,
        kind="formal_single_operator_preflight_qualification_result",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256,
        trust_mode="trusted_single_operator_no_signature",
        formal_measurement=False,
        plan=plan_binding,
        assignment=assignment_binding,
        before_gpu_snapshot=before,
        after_gpu_snapshot=after,
        stdout=stdout,
        stderr=stderr,
        junit_xml=junit,
        live_observation=observation_binding,
        live_native_terminal=live_native,
        live_native_itl=live_itl,
        live_graph=live_graph,
        live_worker_hook=live_worker,
        live_rank_terminals=ranks,
        live_server_receipt=server,
        live_server_log=server_log,
        empirical_proof=proof,
        runner_terminal=terminal,
        started_ns=started_ns,
        finished_ns=finished_ns,
        status="COMPLETE",
    )
    publish_canonical_json_no_replace(plan.result_path, result.to_dict())
    return revalidate_formal_single_operator_preflight_qualification_result(
        plan.result_path
    )


def revalidate_formal_single_operator_preflight_qualification_result(
    path: str | Path,
) -> FormalSingleOperatorPreflightQualificationResult:
    binding = CanonicalJsonProofBinding.bind(path)
    result = FormalSingleOperatorPreflightQualificationResult.from_dict(
        binding.reopen()
    )
    if (
        result.schema_version != 1
        or result.protocol_sha256
        != FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256
        or result.trust_mode != "trusted_single_operator_no_signature"
        or result.formal_measurement is not False
        or result.status != "COMPLETE"
        or result.sha256 != binding.semantic_sha256
        or result.finished_ns < result.started_ns
    ):
        raise ValueError("qualification result schema differs")
    plan = load_formal_single_operator_preflight_qualification_plan(
        result.plan.absolute_path
    )
    assignment = NativeRuntimeQualificationAssignment.load(
        result.assignment.absolute_path
    )
    if (
        Path(plan.result_path) != Path(binding.absolute_path)
        or assignment.schema_version != 2
        or assignment.trusted_single_operator_authority != result.plan
        or assignment.launch_manifest != plan.launch_manifest
        or assignment.gpu_uuids != plan.gpu_uuids
    ):
        raise ValueError("qualification result assignment differs")
    before = _validate_gpu_snapshot(
        result.before_gpu_snapshot.reopen(), assignment=assignment, phase="before"
    )
    after = _validate_gpu_snapshot(
        result.after_gpu_snapshot.reopen(), assignment=assignment, phase="after"
    )
    if (
        before["status"] != "AVAILABLE"
        or after["status"] != "AVAILABLE"
        or before["compute_process_rows"]
        or after["compute_process_rows"]
    ):
        raise ValueError("qualification GPU boundary is not clean")
    for name in ("stdout", "stderr", "junit_xml", "live_server_log"):
        getattr(result, name).reopen(label=f"qualification {name}")
    if _junit_summary(Path(result.junit_xml.absolute_path)) != (
        tuple(sorted(plan.expected_test_names)),
        8,
        8,
        0,
        0,
        0,
    ):
        raise ValueError("qualification result JUnit differs")
    observation = NativeRuntimeQualificationObservation.from_dict(
        result.live_observation.reopen()
    )
    observation.validate_assignment(assignment)
    expected_live = _live_bindings(plan, assignment)
    if (
        expected_live
        != (
            result.live_native_terminal,
            result.live_native_itl,
            result.live_graph,
            result.live_worker_hook,
            result.live_rank_terminals,
            result.live_server_receipt,
            result.live_server_log,
        )
        or observation.native_terminal_sha256
        != result.live_native_terminal.semantic_sha256
        or observation.native_itl_pointer_sha256
        != result.live_native_itl.semantic_sha256
        or observation.graph_observation_sha256 != result.live_graph.semantic_sha256
        or observation.worker_hook_observation_sha256
        != result.live_worker_hook.semantic_sha256
        or observation.rank_terminal_sha256s
        != tuple(row.semantic_sha256 for row in result.live_rank_terminals)
        or observation.live_server_receipt_sha256
        != result.live_server_receipt.semantic_sha256
    ):
        raise ValueError("qualification result live evidence differs")
    proof = result.empirical_proof.reopen()
    terminal = result.runner_terminal.reopen()
    if (
        type(proof) is not dict
        or proof.get("kind")
        != "formal_single_operator_preflight_qualification_empirical_proof"
        or proof.get("formal_measurement") is not False
        or proof.get("signature_status") != "NOT_APPLICABLE"
        or proof.get("plan_sha256") != plan.sha256
        or proof.get("assignment_sha256") != assignment.sha256
        or proof.get("junit_raw_sha256") != result.junit_xml.raw_sha256
        or type(terminal) is not dict
        or terminal.get("kind")
        != "formal_single_operator_preflight_qualification_terminal"
        or terminal.get("status") != "COMPLETE"
        or terminal.get("plan_sha256") != plan.sha256
        or terminal.get("assignment_sha256") != assignment.sha256
        or terminal.get("empirical_proof_sha256")
        != result.empirical_proof.semantic_sha256
        or terminal.get("started_ns") != result.started_ns
        or terminal.get("finished_ns") != result.finished_ns
    ):
        raise ValueError("qualification result proof/terminal differs")
    return result


def execute_formal_single_operator_preflight_qualification_plan_index(
    index_path: str | Path,
) -> tuple[FormalSingleOperatorPreflightQualificationResult, ...]:
    """Execute exact six sequentially under one non-blocking gang lock."""

    index = load_formal_single_operator_preflight_qualification_plan_index(index_path)
    root = Path(index_path).parent
    lock_path = root / ".qualification-gang.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FormalSingleOperatorPreflightQualificationBlocked(
                "qualification_gang_busy"
            ) from error
        return tuple(
            execute_formal_single_operator_preflight_qualification_plan(
                binding.absolute_path
            )
            for binding in index.plans
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


__all__ = (
    "FORMAL_SINGLE_OPERATOR_PREFLIGHT_QUALIFICATION_PROTOCOL_SHA256",
    "TRUSTED_PREFLIGHT_QUALIFICATION_SUITES",
    "FormalSingleOperatorPreflightQualificationBlocked",
    "FormalSingleOperatorPreflightQualificationPlan",
    "FormalSingleOperatorPreflightQualificationPlanIndex",
    "FormalSingleOperatorPreflightQualificationResult",
    "TrustedQualificationDispatchAuthority",
    "TrustedQualificationLaunchEntry",
    "TrustedQualificationLaunchIndex",
    "execute_formal_single_operator_preflight_qualification_plan",
    "execute_formal_single_operator_preflight_qualification_plan_index",
    "load_formal_single_operator_preflight_qualification_plan",
    "load_formal_single_operator_preflight_qualification_plan_index",
    "load_trusted_qualification_dispatch_authority",
    "load_trusted_qualification_launch_index",
    "materialize_formal_single_operator_preflight_qualification_plans",
    "publish_formal_single_operator_preflight_qualification_launch_index",
    "revalidate_formal_single_operator_preflight_qualification_result",
)
