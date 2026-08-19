"""Source-owned physical execution for the exact 108 E0 compatibility probes.

The compatibility reducer intentionally does not execute models.  This module
bridges that boundary without introducing a caller-controlled scientific
surface.  It materializes 108 immutable probe plans from the current E6-final
completion, schema-5 trusted ProtocolLock, BOUND content/runtime observations,
12 schema-3 pre-probe model/backend interface receipts, and the nine code-owned task
sources.  A worker command is fixed by this module and groups the nine tasks
sharing one model/backend launch so a compatible server is started at most
once per invocation.

An unsupported interface or task still runs the source-owned validation path,
but it never starts a server and never claims GPU work.  A READY/READY probe
must complete exactly one real request through the pinned SGLang bench
transport.  Transport, server, tokenizer, cleanup, or evidence failures are
recorded as failures and are never converted to ``N/A``.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import load_run_config
from lightcone_spec.experiments.e0_stage_authority import (
    E0OnlineSpecSourceAuthority,
)
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock, content_sha256
from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundle,
)
from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
    E0_COMPATIBILITY_VALID_REASON,
    E0CompatibilityProbeTerminal,
    E0CompatibilityPublication,
    E0Eagle3RuntimeProofRow,
    E0PreparedModelBackendInterfaceReceipt,
    E0TaskNativeWorkloadAuthority,
    e0_eagle3_runtime_authority_for_task,
    e0_eagle3_runtime_proof_row_for_task,
    load_e0_compatibility_probe_terminal,
    load_e0_prepared_model_backend_interface_receipt,
    load_e0_task_native_workload_authority,
    publish_e0_compatibility_probe_terminal,
    publish_e0_task_native_workload_authority,
    publish_trusted_e0_compatibility_probe_sources,
)
from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
    E0TaskNativeSourceAuthority,
    bind_e0_task_native_workload_authority,
    load_e0_task_native_source_authority,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
    RebuiltFormalSingleOperatorStageCompletion,
    rebuild_formal_single_operator_stage_completion,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.load import FrozenSamplingParameters
from lightcone_spec.experiments.serving import (
    BoundServingRequest,
    PinnedBenchServingTransport,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
)
from lightcone_spec.experiments.workload_authority import (
    FormalWorkloadAuthority,
    bind_formal_workload_authority,
    revalidate_formal_workload_authority,
)
from lightcone_spec.orchestration.live_sglang import (
    _PROCESS_GROUP_CLEANUP_SECONDS,
    PinnedNvidiaSmiTool,
    _capture_gpu_process_snapshot,
    _process_group_exists,
    _terminate_process_group,
    _wait_server_ready,
    validate_pinned_sglang_gpu_process_snapshot,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_compatibility_physical_protocol",
        "coverage": "4_models_x_3_backends_x_9_tasks_exact_108",
        "launch_groups": "12_model_backend_groups_reusing_one_exact_server",
        "ready_probe": "one_task_native_request_exactly_once",
        "unsupported_probe": "source_owned_validation_without_gpu_claim",
        "failure": "terminal_failure_never_NA",
        "publication": "canonical_no_replace_path_bound_raw_evidence",
        "eagle3": (
            "task_proof_published_only_after_successful_one_request_core_evidence"
        ),
        "timestamps": "runtime_monotonic_ns_only",
    }
)
FORMAL_E0_COMPATIBILITY_WORKER_MODULE = (
    "lightcone_spec.orchestration.formal_e0_compatibility_physical"
)
_LOCKED_TASK_IDS = {
    "LiveCodeBench": "livecodebench_v6_hard",
    "MATH-500": "math500_level5",
}
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_SERVER_READY_TIMEOUT_SECONDS = 600.0
_REQUEST_TIMEOUT_SECONDS = 600.0
_TRANSPORT_ABORT_TIMEOUT_SECONDS = 30.0
_PROCESS_GROUP_SIGKILL_AND_EMPTY_BOUND_SECONDS = 35.0
_CAMPAIGN_PUBLICATION_GRACE_SECONDS = 15 * 60
_SHA_CHARS = frozenset("0123456789abcdef")


class FormalE0CompatibilityPhysicalBlocked(RuntimeError):
    """A current source, runtime, GPU, or physical probe is unavailable."""


def _sha(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be canonical text")
    return value


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _absolute_path(label: str, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    if path == Path(path.anchor):
        raise ValueError(f"{label} cannot be a filesystem root")
    return path


def _new_private_root(value: str | Path) -> Path:
    root = _absolute_path("E0 compatibility output root", value)
    if os.path.lexists(root):
        raise FileExistsError("E0 compatibility output root already exists")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ValueError("E0 compatibility output parent is unavailable")
    root.mkdir(mode=0o700)
    return root


def _existing_private_root(value: str | Path, *, label: str) -> Path:
    root = _absolute_path(label, value)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} is unavailable")
    mode = stat.S_IMODE(root.stat(follow_symlinks=False).st_mode)
    if mode & 0o077:
        raise ValueError(f"{label} must be private")
    return root


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _publish_raw_no_replace(path: Path, body: bytes) -> EvidenceFileBinding:
    if not body:
        raise ValueError("E0 compatibility raw evidence cannot be empty")
    if len(body) > _MAX_EVIDENCE_BYTES:
        raise ValueError("E0 compatibility raw evidence exceeds its bound")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("E0 compatibility evidence parent is unavailable")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return EvidenceFileBinding.bind(path, label="E0 compatibility raw evidence")


def _bind_json(path: str | Path, *, label: str) -> CanonicalJsonProofBinding:
    try:
        return CanonicalJsonProofBinding.bind(path)
    except (OSError, TypeError, ValueError) as error:
        raise FormalE0CompatibilityPhysicalBlocked(f"{label}_unavailable") from error


def _clock_after(previous: int | None = None) -> int:
    """Read the real monotonic clock; never synthesize a timestamp."""

    for _ in range(1_000):
        value = time.monotonic_ns()
        if type(value) is int and value > 0 and (previous is None or value > previous):
            return value
    raise RuntimeError("monotonic clock did not advance")


def _e6_identity(
    *,
    protocol_lock: ProtocolLock,
    completion: RebuiltFormalSingleOperatorStageCompletion,
) -> tuple[str, str]:
    if completion.artifact.node != "e6_final":
        raise ValueError("E0 physical campaign requires E6-final completion")
    materialization = _sha(
        "E0 upstream E6 materialization", completion.materialization.sha256
    )
    confirmation = _sha(
        "E0 upstream E6 confirmation",
        completion.decision.payload.get("confirmation_sha256"),
    )
    if (
        completion.materialization.protocol_lock_sha256 != protocol_lock.sha256
        or completion.decision.next_materialization_source_decision_sha256
        != confirmation
        or completion.decision.next_materialization_upstream_receipt_sha256s
        != (materialization,)
    ):
        raise ValueError("E0 physical campaign E6 lineage differs")
    return materialization, confirmation


def _interface_key(
    receipt: E0PreparedModelBackendInterfaceReceipt,
) -> tuple[str, str]:
    return receipt.model, receipt.backend


def _eagle3_runtime_proof_row_sha256(
    receipt: E0PreparedModelBackendInterfaceReceipt,
    *,
    task: str,
    terminal: E0CompatibilityProbeTerminal | None = None,
) -> str | None:
    if (
        receipt.backend != "EAGLE3"
        or receipt.support_status != "READY"
        or (terminal is not None and terminal.disposition != "VALID")
    ):
        return None
    if receipt.schema_version == 3 and terminal is None:
        return None
    row = e0_eagle3_runtime_proof_row_for_task(
        receipt,
        task=task,
        terminal=terminal,
    )
    claims = e0_eagle3_runtime_authority_for_task(
        receipt,
        task=task,
        terminal=terminal,
    )
    if claims != {
        "eagle3_e0_execution_authority_sha256": row.execution_authority_sha256,
        "eagle3_compatibility_authority_sha256": (row.compatibility_authority_sha256),
        "eagle3_model_selector_sha256": row.model_selector_sha256,
        "eagle3_native_gpu_proof_sha256": row.native_gpu_proof_sha256,
    }:
        raise ValueError("E0 physical EAGLE3 task authority differs")
    return row.sha256


def _workload_key(authority: E0TaskNativeWorkloadAuthority) -> tuple[str, str]:
    return authority.model, authority.task


def _probe_key(plan: E0CompatibilityProbePlan) -> tuple[str, str, str]:
    return plan.model, plan.backend, plan.task


def _expected_interface_keys() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((model, backend) for model in E0_MODELS for backend in E0_BACKENDS)
    )


def _expected_workload_keys() -> tuple[tuple[str, str], ...]:
    return tuple(sorted((model, task) for model in E0_MODELS for task in E0_TASKS))


def expected_e0_compatibility_probe_keys() -> tuple[tuple[str, str, str], ...]:
    """Return the immutable, canonical exact-108 compatibility universe."""

    keys = tuple(
        sorted(
            (model, backend, task)
            for model in E0_MODELS
            for backend in E0_BACKENDS
            for task in E0_TASKS
        )
    )
    if len(keys) != 108 or len(set(keys)) != 108:
        raise RuntimeError("E0 compatibility universe is not exact 108")
    return keys


@dataclass(frozen=True)
class E0CompatibilityProbePlan:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e0_compatibility_probe_plan"]
    protocol_sha256: str
    protocol_lock: FormalSingleOperatorJsonBinding
    e6_completion: FormalSingleOperatorJsonBinding
    content_source: FormalContentSourceBinding
    inventory: CanonicalJsonProofBinding
    doctor: CanonicalJsonProofBinding
    interface_receipt: CanonicalJsonProofBinding
    workload_authority: CanonicalJsonProofBinding
    workload_source: CanonicalJsonProofBinding
    model: str
    backend: str
    task: str
    source_sample_id: str
    source_prompt_sha256: str
    source_seed: int
    execution_kind: Literal["GPU_ONE_REQUEST", "CODE_VALIDATION_ONLY"]
    gpu_uuid: str | None
    topology_mode: Literal["tp1_dp1"]
    worker_command_sha256: str
    group_plan_path: str
    terminal_output_path: str
    evidence_directory: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e0_compatibility_probe_plan"
            or self.protocol_sha256 != FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256
            or self.model not in E0_MODELS
            or self.backend not in E0_BACKENDS
            or self.task not in E0_TASKS
            or self.topology_mode != "tp1_dp1"
        ):
            raise ValueError("E0 compatibility probe plan identity differs")
        for label, digest in (
            ("prompt", self.source_prompt_sha256),
            ("worker command", self.worker_command_sha256),
        ):
            _sha(f"E0 probe {label}", digest)
        _text("E0 probe source sample", self.source_sample_id)
        if type(self.source_seed) is not int or not 0 <= self.source_seed < 2**63:
            raise ValueError("E0 probe source seed differs")
        if self.execution_kind == "GPU_ONE_REQUEST":
            if type(self.gpu_uuid) is not str or not self.gpu_uuid.startswith("GPU-"):
                raise ValueError("GPU E0 probe lacks one UUID")
        elif self.execution_kind == "CODE_VALIDATION_ONLY":
            if self.gpu_uuid is not None:
                raise ValueError("code-only E0 probe claims a GPU")
        else:
            raise ValueError("E0 probe execution kind differs")
        if type(self.protocol_lock) is not FormalSingleOperatorJsonBinding:
            raise TypeError("E0 probe ProtocolLock is not path-bound")
        if type(self.e6_completion) is not FormalSingleOperatorJsonBinding:
            raise TypeError("E0 probe E6 completion is not path-bound")
        if (
            type(self.content_source) is not FormalContentSourceBinding
            or self.content_source.mode != "trusted_single_operator"
        ):
            raise TypeError("E0 probe content source is not trusted")
        for binding in (
            self.inventory,
            self.doctor,
            self.interface_receipt,
            self.workload_authority,
            self.workload_source,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("E0 probe input is not path-bound")
        root = _absolute_path("E0 probe evidence directory", self.evidence_directory)
        terminal = _absolute_path("E0 probe terminal", self.terminal_output_path)
        group = _absolute_path("E0 probe group plan", self.group_plan_path)
        if terminal.parent != root or group.parent != root.parent.parent:
            raise ValueError("E0 probe output topology differs")

    @cached_property
    def probe_id(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_single_operator_e0_compatibility_probe_id",
                "protocol_lock_sha256": self.protocol_lock.semantic_sha256,
                "e6_completion_sha256": self.e6_completion.semantic_sha256,
                "model": self.model,
                "backend": self.backend,
                "task": self.task,
                "interface_receipt_sha256": self.interface_receipt.semantic_sha256,
                "workload_authority_sha256": self.workload_authority.semantic_sha256,
            }
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "protocol_lock": self.protocol_lock.to_dict(),
            "e6_completion": self.e6_completion.to_dict(),
            "content_source": self.content_source.to_dict(),
            "inventory": self.inventory.to_dict(),
            "doctor": self.doctor.to_dict(),
            "interface_receipt": self.interface_receipt.to_dict(),
            "workload_authority": self.workload_authority.to_dict(),
            "workload_source": self.workload_source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E0 compatibility probe plan", value, set(cls.__dataclass_fields__)
        )
        row["protocol_lock"] = FormalSingleOperatorJsonBinding.from_dict(
            row["protocol_lock"]
        )
        row["e6_completion"] = FormalSingleOperatorJsonBinding.from_dict(
            row["e6_completion"]
        )
        row["content_source"] = FormalContentSourceBinding.from_dict(
            row["content_source"]
        )
        for name in (
            "inventory",
            "doctor",
            "interface_receipt",
            "workload_authority",
            "workload_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class E0CompatibilityProbeGroupPlan:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e0_compatibility_probe_group"]
    protocol_sha256: str
    model: str
    backend: str
    interface_receipt: CanonicalJsonProofBinding
    compile_launch_manifest: CanonicalJsonProofBinding | None
    inventory: CanonicalJsonProofBinding
    doctor: CanonicalJsonProofBinding
    python_executable: EvidenceFileBinding
    worker_launcher: EvidenceFileBinding
    worker_source: EvidenceFileBinding
    argv: tuple[str, ...]
    argv_sha256: str
    gpu_uuid: str | None
    topology_mode: Literal["tp1_dp1"]
    probes: tuple[CanonicalJsonProofBinding, ...]
    evidence_directory: str
    completion_output_path: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e0_compatibility_probe_group"
            or self.protocol_sha256 != FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256
            or self.model not in E0_MODELS
            or self.backend not in E0_BACKENDS
            or self.topology_mode != "tp1_dp1"
            or len(self.probes) != 9
            or len(set(self.probes)) != 9
        ):
            raise ValueError("E0 compatibility group identity differs")
        if type(self.argv) is not tuple or self.argv != (
            self.worker_launcher.absolute_path,
            "execute-group",
            "--group-plan",
            str(Path(self.evidence_directory) / "group-plan.json"),
        ):
            raise ValueError("E0 compatibility group argv differs")
        if self.argv_sha256 != content_sha256({"argv": list(self.argv)}):
            raise ValueError("E0 compatibility group argv digest differs")
        if self.compile_launch_manifest is None:
            if self.gpu_uuid is not None:
                raise ValueError("non-executable E0 group claims a GPU")
        elif type(self.gpu_uuid) is not str or not self.gpu_uuid.startswith("GPU-"):
            raise ValueError("executable E0 group lacks one GPU")
        for binding in (self.interface_receipt, self.inventory, self.doctor):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("E0 compatibility group input is not path-bound")
        if (
            self.compile_launch_manifest is not None
            and type(self.compile_launch_manifest) is not CanonicalJsonProofBinding
        ):
            raise TypeError("E0 compatibility group launch is not path-bound")
        for binding in self.probes:
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("E0 compatibility group probe is not path-bound")
        if (
            type(self.python_executable) is not EvidenceFileBinding
            or type(self.worker_launcher) is not EvidenceFileBinding
            or type(self.worker_source) is not EvidenceFileBinding
        ):
            raise TypeError("E0 compatibility worker source is not path-bound")
        root = _absolute_path("E0 compatibility group root", self.evidence_directory)
        completion = _absolute_path(
            "E0 compatibility group completion", self.completion_output_path
        )
        if completion.parent != root:
            raise ValueError("E0 compatibility group completion path differs")
        launcher = _absolute_path(
            "E0 compatibility worker launcher", self.worker_launcher.absolute_path
        )
        if launcher.parent != root.parent.parent:
            raise ValueError("E0 compatibility launcher topology differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "interface_receipt": self.interface_receipt.to_dict(),
            "compile_launch_manifest": (
                None
                if self.compile_launch_manifest is None
                else self.compile_launch_manifest.to_dict()
            ),
            "inventory": self.inventory.to_dict(),
            "doctor": self.doctor.to_dict(),
            "python_executable": self.python_executable.to_dict(),
            "worker_launcher": self.worker_launcher.to_dict(),
            "worker_source": self.worker_source.to_dict(),
            "argv": list(self.argv),
            "probes": [row.to_dict() for row in self.probes],
        }
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E0 compatibility group plan", value, set(cls.__dataclass_fields__)
        )
        for name in ("interface_receipt", "inventory", "doctor"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        launch = row["compile_launch_manifest"]
        row["compile_launch_manifest"] = (
            None if launch is None else CanonicalJsonProofBinding.from_dict(launch)
        )
        for name in ("python_executable", "worker_launcher", "worker_source"):
            row[name] = EvidenceFileBinding.from_dict(
                row[name], label=f"E0 compatibility {name}"
            )
        raw_argv = row.pop("argv")
        raw_probes = row.pop("probes")
        if type(raw_argv) is not list or type(raw_probes) is not list:
            raise TypeError("E0 compatibility group arrays differ")
        return cls(
            **row,
            argv=tuple(raw_argv),
            probes=tuple(
                CanonicalJsonProofBinding.from_dict(item) for item in raw_probes
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class E0CompatibilityPhysicalCampaign:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e0_compatibility_physical_campaign"]
    protocol_sha256: str
    protocol_lock_sha256: str
    e6_materialization_sha256: str
    e6_confirmation_sha256: str
    trusted_content_sha256: str
    inventory_sha256: str
    interface_receipts: tuple[CanonicalJsonProofBinding, ...]
    workload_authorities: tuple[CanonicalJsonProofBinding, ...]
    probe_plans: tuple[CanonicalJsonProofBinding, ...]
    groups: tuple[CanonicalJsonProofBinding, ...]
    physical_probe_count: Literal[108]
    launch_group_count: Literal[12]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e0_compatibility_physical_campaign"
            or self.protocol_sha256 != FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256
            or self.physical_probe_count != 108
            or self.launch_group_count != 12
            or len(self.interface_receipts) != 12
            or len(set(self.interface_receipts)) != 12
            or len(self.workload_authorities) != 36
            or len(set(self.workload_authorities)) != 36
            or len(self.probe_plans) != 108
            or len(set(self.probe_plans)) != 108
            or len(self.groups) != 12
            or len(set(self.groups)) != 12
        ):
            raise ValueError("E0 compatibility physical campaign coverage differs")
        for label, digest in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("E6 materialization", self.e6_materialization_sha256),
            ("E6 confirmation", self.e6_confirmation_sha256),
            ("trusted content", self.trusted_content_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _sha(f"E0 campaign {label}", digest)
        for binding in (
            *self.interface_receipts,
            *self.workload_authorities,
            *self.probe_plans,
            *self.groups,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("E0 campaign input is not path-bound")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "interface_receipts": [row.to_dict() for row in self.interface_receipts],
            "workload_authorities": [
                row.to_dict() for row in self.workload_authorities
            ],
            "probe_plans": [row.to_dict() for row in self.probe_plans],
            "groups": [row.to_dict() for row in self.groups],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict("E0 physical campaign", value, set(cls.__dataclass_fields__))
        for name in (
            "interface_receipts",
            "workload_authorities",
            "probe_plans",
            "groups",
        ):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(f"E0 campaign {name} must be an array")
            row[name] = tuple(CanonicalJsonProofBinding.from_dict(item) for item in raw)
        return cls(**row)  # type: ignore[arg-type]


def _doctor_tools(
    *,
    binding: CanonicalJsonProofBinding,
    inventory: GpuInventory,
    launch: CompileLaunchManifest,
) -> tuple[str, str]:
    value = binding.reopen()
    if type(value) is not dict:
        raise FormalE0CompatibilityPhysicalBlocked("runtime_doctor_malformed")
    readiness = value.get("readiness")
    checks = value.get("checks")
    roots = value.get("roots")
    python = value.get("python")
    gpu = value.get("gpu")
    commands = value.get("commands")
    parsed = None if type(gpu) is not dict else gpu.get("parsed_inventory")
    devices = None if type(parsed) is not dict else parsed.get("devices")
    if (
        value.get("schema_version") != 2
        or value.get("status") != "PASS"
        or type(readiness) is not dict
        or readiness.get("status") != "PASS"
        or readiness.get("fail_count") != 0
        or readiness.get("unknown_count") != 0
        or type(checks) is not dict
        or not checks
        or any(
            type(row) is not dict or row.get("status") != "PASS"
            for row in checks.values()
        )
        or type(roots) is not dict
        or roots.get("patched_sglang") != launch.patched_sglang_checkout
        or type(python) is not dict
        or type(commands) is not dict
        or type(devices) is not list
    ):
        raise FormalE0CompatibilityPhysicalBlocked(
            "complete_pass_runtime_doctor_unavailable"
        )
    by_uuid = {row.get("uuid"): row for row in devices if type(row) is dict}
    for device in inventory.devices:
        row = by_uuid.get(device.uuid)
        if (
            type(row) is not dict
            or row.get("name") != device.model
            or row.get("compute_capability")
            != f"{device.compute_capability[0]}.{device.compute_capability[1]}"
        ):
            raise FormalE0CompatibilityPhysicalBlocked("doctor_gpu_identity_differs")
    python_path = Path(str(python.get("executable"))).resolve(strict=False)
    nvidia_path = shutil.which("nvidia-smi", path=os.pathsep.join(launch.path_entries))
    if (
        not python_path.is_file()
        or python_path.is_symlink()
        or nvidia_path is None
        or not Path(nvidia_path).resolve(strict=True).is_file()
        or not str(commands.get("nvidia_smi", "")).strip()
    ):
        raise FormalE0CompatibilityPhysicalBlocked("doctor_runtime_tools_unavailable")
    return str(python_path), str(Path(nvidia_path).resolve(strict=True))


def _load_probe_source(
    binding: CanonicalJsonProofBinding,
    *,
    task: str,
) -> tuple[str, str, int, str, str, Literal["READY", "UNSUPPORTED"]]:
    """Return sample id, prompt, seed, source revision/evidence, and status."""

    value = binding.reopen()
    if task in _LOCKED_TASK_IDS:
        authority = FormalWorkloadAuthority.from_dict(value)
        revalidate_formal_workload_authority(authority)
        if authority.workload_id != _LOCKED_TASK_IDS[task]:
            raise ValueError("E0 locked workload task differs")
        sample = authority.samples[0]
        revision_sha256 = content_sha256(
            {
                "repository_revision": authority.repository_revision,
                "source_lock_sha256": authority.source_lock_sha256,
            }
        )
        return (
            sample.sample_id,
            sample.prompt,
            sample.seed,
            revision_sha256,
            authority.sha256,
            "READY",
        )
    authority = load_e0_task_native_source_authority(binding.absolute_path)
    if type(authority) is not E0TaskNativeSourceAuthority or authority.task != task:
        raise ValueError("E0 standalone workload task differs")
    request = authority.request_rows[0]
    prompt = "\n".join(request.turns)
    return (
        request.source_id,
        prompt,
        request.seed,
        authority.source_revision_sha256,
        authority.sha256,
        authority.support_status,
    )


def _locked_workload_sha256(authority: FormalWorkloadAuthority, *, task: str) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_e0_task_native_workload",
            "task": task,
            "source_authority_sha256": authority.sha256,
            "source_lock_sha256": authority.source_lock_sha256,
            "selected_rows_sha256": authority.selected_rows_sha256,
            "request_row_count": authority.selected_row_count,
        }
    )


def _source_bindings(
    *,
    bundle: TrustedSingleOperatorContentBundle,
    root: Path,
) -> dict[str, CanonicalJsonProofBinding]:
    source_root = root / "task-sources"
    source_root.mkdir(mode=0o700)
    result: dict[str, CanonicalJsonProofBinding] = {}
    for task, workload_id in _LOCKED_TASK_IDS.items():
        matches = tuple(
            row for row in bundle.locked_workloads if row.workload_id == workload_id
        )
        if len(matches) != 1:
            raise FormalE0CompatibilityPhysicalBlocked(
                f"source_owned_{workload_id}_missing"
            )
        authority = bind_formal_workload_authority(
            workload_id, matches[0].raw_source_path
        )
        if authority.sha256 != matches[0].authority_sha256:
            raise ValueError("E0 locked workload authority differs from content")
        path = source_root / f"{E0_TASKS.index(task):02d}.json"
        publish_canonical_json_no_replace(path, authority.to_dict())
        result[task] = CanonicalJsonProofBinding.bind(
            path, semantic_sha256=authority.sha256
        )
    by_task = {row.task: row for row in bundle.e0_task_native_descriptors}
    for task in E0_TASKS:
        if task in _LOCKED_TASK_IDS:
            continue
        descriptor = by_task.get(task)
        if descriptor is None:
            raise FormalE0CompatibilityPhysicalBlocked(f"source_owned_{task}_missing")
        authority = load_e0_task_native_source_authority(
            descriptor.source.absolute_path
        )
        if (
            authority.task != task
            or descriptor.source.semantic_sha256
            != CanonicalJsonProofBinding.bind(
                descriptor.source.absolute_path
            ).semantic_sha256
        ):
            raise ValueError("E0 task-native descriptor differs from content")
        result[task] = CanonicalJsonProofBinding.bind(descriptor.source.absolute_path)
    if set(result) != set(E0_TASKS):
        raise RuntimeError("E0 source binding coverage is not nine tasks")
    return result


def _load_current_inputs(
    *,
    protocol_lock_path: str | Path,
    e6_completion_path: str | Path,
    trusted_content_bundle_path: str | Path,
) -> tuple[
    FormalSingleOperatorJsonBinding,
    ProtocolLock,
    FormalSingleOperatorJsonBinding,
    RebuiltFormalSingleOperatorStageCompletion,
    FormalContentSourceBinding,
    TrustedSingleOperatorContentBundle,
    CanonicalJsonProofBinding,
    GpuInventory,
    CanonicalJsonProofBinding,
    str,
    str,
]:
    lock_binding = FormalSingleOperatorJsonBinding.bind(
        protocol_lock_path, label="E0 physical ProtocolLock"
    )
    lock = protocol_lock_from_dict(
        lock_binding.reopen(label="E0 physical ProtocolLock")
    )
    if (
        lock.schema_version != 5
        or lock.content_source_mode != "trusted_single_operator"
        or lock.sha256 != lock_binding.semantic_sha256
    ):
        raise FormalE0CompatibilityPhysicalBlocked(
            "schema5_trusted_protocol_lock_required"
        )
    e6_binding = FormalSingleOperatorJsonBinding.bind(
        e6_completion_path, label="E0 physical E6 completion"
    )
    completion = rebuild_formal_single_operator_stage_completion(
        e6_binding.absolute_path
    )
    materialization, confirmation = _e6_identity(
        protocol_lock=lock, completion=completion
    )
    content = FormalContentSourceBinding.bind_trusted_single_operator(
        str(trusted_content_bundle_path)
    )
    bundle = content.reopen()
    if (
        type(bundle) is not TrustedSingleOperatorContentBundle
        or bundle.runtime_binding_status != "BOUND"
        or bundle.runtime_observations is None
        or content.content_sha256 != lock.trusted_single_operator_content_bundle_sha256
    ):
        raise FormalE0CompatibilityPhysicalBlocked(
            "runtime_bound_trusted_content_required"
        )
    inventory_artifact = bundle.runtime_observations.inventory
    doctor_artifact = bundle.runtime_observations.doctor
    inventory_binding = _bind_json(
        inventory_artifact.absolute_path, label="trusted_runtime_inventory"
    )
    doctor_binding = _bind_json(
        doctor_artifact.absolute_path, label="trusted_runtime_doctor"
    )
    if (
        inventory_binding.raw_sha256 != inventory_artifact.raw_sha256
        or inventory_binding.semantic_sha256 != inventory_artifact.semantic_sha256
        or doctor_binding.raw_sha256 != doctor_artifact.raw_sha256
        or doctor_binding.semantic_sha256 != doctor_artifact.semantic_sha256
    ):
        raise RuntimeError("E0 trusted runtime observations changed")
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    if inventory.sha256 != inventory_binding.semantic_sha256:
        raise ValueError("E0 trusted inventory identity differs")
    return (
        lock_binding,
        lock,
        e6_binding,
        completion,
        content,
        bundle,
        inventory_binding,
        inventory,
        doctor_binding,
        materialization,
        confirmation,
    )


def _load_interface_receipts(
    paths: tuple[str | Path, ...],
    *,
    lock: ProtocolLock,
    e6_confirmation_sha256: str,
    content: FormalContentSourceBinding,
    inventory: GpuInventory,
    doctor: CanonicalJsonProofBinding,
) -> tuple[
    tuple[CanonicalJsonProofBinding, ...],
    dict[tuple[str, str], E0PreparedModelBackendInterfaceReceipt],
]:
    if type(paths) is not tuple or len(paths) != 12:
        raise ValueError("E0 physical interface paths must cover exact 12")
    raw_bindings = tuple(CanonicalJsonProofBinding.bind(path) for path in paths)
    raw_receipts = tuple(
        load_e0_prepared_model_backend_interface_receipt(row.absolute_path)
        for row in raw_bindings
    )
    by_key = {
        _interface_key(receipt): (binding, receipt)
        for binding, receipt in zip(raw_bindings, raw_receipts, strict=True)
    }
    keys = _expected_interface_keys()
    if len(by_key) != 12 or set(by_key) != set(keys):
        raise ValueError("E0 physical interface paths do not cover exact 12")
    bindings = tuple(by_key[key][0] for key in keys)
    receipts = tuple(by_key[key][1] for key in keys)
    for binding, receipt in zip(bindings, receipts, strict=True):
        if (
            binding.semantic_sha256 != receipt.sha256
            or receipt.schema_version != 3
            or receipt.protocol_lock_sha256 != lock.sha256
            or receipt.upstream_e6_confirmation_sha256 != e6_confirmation_sha256
        ):
            raise ValueError("E0 physical interface lineage differs")
        launch_binding = receipt.compile_launch_manifest
        if receipt.support_status == "UNSUPPORTED":
            if launch_binding is not None:
                raise ValueError("unsupported E0 interface carries a launch")
            continue
        if launch_binding is None:
            raise ValueError("READY E0 interface lacks a launch")
        launch = CompileLaunchManifest.load(launch_binding.absolute_path)
        config = load_run_config(launch.run_config_path)
        if (
            launch.sha256 != launch_binding.semantic_sha256
            or launch.schema_version != 2
            or launch.formal_stage != "E0"
            or launch.content_source_binding != content
            or launch.inventory_sha256 != inventory.sha256
            or tuple(launch.gpu_uuids)
            not in {(device.uuid,) for device in inventory.devices}
            or config.runtime.topology_mode != "tp1_dp1"
            or config.runtime.tensor_parallel_size != 1
            or config.runtime.data_parallel_size != 1
            or config.method != "static"
            or config.adaptation is not None
            or config.online_spec is not None
            or config.model.algorithm != receipt.backend
            or config.model.target != receipt.target_model_id
            or config.model.target_revision != receipt.target_revision
            or config.model.drafter != receipt.drafter_model_id
            or config.model.drafter_revision != receipt.drafter_revision
        ):
            raise ValueError("E0 physical compile launch differs from interface")
        if receipt.eagle3_runtime_proof_rows:
            raise ValueError("fresh E0 physical interface carries post-probe proofs")
        _doctor_tools(
            binding=doctor,
            inventory=inventory,
            launch=launch,
        )
    return bindings, {key: row for key, row in zip(keys, receipts, strict=True)}


def materialize_formal_e0_compatibility_physical_campaign(
    *,
    protocol_lock_path: str | Path,
    e6_completion_path: str | Path,
    trusted_content_bundle_path: str | Path,
    interface_receipt_paths: tuple[str | Path, ...],
    output_root: str | Path,
) -> E0CompatibilityPhysicalCampaign:
    """Materialize the exact 108 immutable plans and 12 fixed worker groups."""

    (
        lock_binding,
        lock,
        e6_binding,
        _completion,
        content,
        bundle,
        inventory_binding,
        inventory,
        doctor_binding,
        e6_materialization,
        e6_confirmation,
    ) = _load_current_inputs(
        protocol_lock_path=protocol_lock_path,
        e6_completion_path=e6_completion_path,
        trusted_content_bundle_path=trusted_content_bundle_path,
    )
    interface_bindings, interfaces = _load_interface_receipts(
        interface_receipt_paths,
        lock=lock,
        e6_confirmation_sha256=e6_confirmation,
        content=content,
        inventory=inventory,
        doctor=doctor_binding,
    )
    root = _new_private_root(output_root)
    sources = _source_bindings(bundle=bundle, root=root)
    interface_binding_by_key = {
        _interface_key(receipt): binding
        for binding, receipt in zip(
            interface_bindings, interfaces.values(), strict=True
        )
    }

    workload_root = root / "workload-authorities"
    workload_root.mkdir(mode=0o700)
    workload_bindings: list[CanonicalJsonProofBinding] = []
    workload_by_key: dict[tuple[str, str], E0TaskNativeWorkloadAuthority] = {}
    workload_binding_by_key: dict[tuple[str, str], CanonicalJsonProofBinding] = {}
    tokenizer_by_model: dict[str, str] = {}
    for model in sorted(E0_MODELS):
        model_interfaces = tuple(
            interfaces[(model, backend)] for backend in E0_BACKENDS
        )
        tokenizer_values = {row.tokenizer_sha256 for row in model_interfaces}
        if len(tokenizer_values) != 1:
            raise ValueError("E0 model interfaces disagree on tokenizer identity")
        tokenizer_by_model[model] = next(iter(tokenizer_values))
        for task in sorted(E0_TASKS):
            source = sources[task]
            (
                _sample_id,
                _prompt,
                _seed,
                revision_sha,
                evidence_sha,
                source_status,
            ) = _load_probe_source(source, task=task)
            if task in _LOCKED_TASK_IDS:
                locked = FormalWorkloadAuthority.from_dict(source.reopen())
                workload_sha = _locked_workload_sha256(locked, task=task)
                reason = "TASK_WORKLOAD_READY"
            else:
                standalone = load_e0_task_native_source_authority(source.absolute_path)
                bound = bind_e0_task_native_workload_authority(
                    source=standalone,
                    protocol_lock_sha256=lock.sha256,
                    upstream_e6_confirmation_sha256=e6_confirmation,
                    model=model,
                    tokenizer_sha256=tokenizer_by_model[model],
                )
                workload_sha = bound.task_native_workload_sha256
                revision_sha = bound.source_revision_sha256
                evidence_sha = bound.evidence_sha256
                reason = bound.reason_code
            authority = E0TaskNativeWorkloadAuthority(
                schema_version=1,
                protocol_lock_sha256=lock.sha256,
                upstream_e6_confirmation_sha256=e6_confirmation,
                model=model,
                task=task,
                tokenizer_sha256=tokenizer_by_model[model],
                task_native_workload_sha256=workload_sha,
                source_revision_sha256=revision_sha,
                support_status=source_status,
                reason_code=reason,
                evidence_sha256=evidence_sha,
            )
            key = (model, task)
            path = workload_root / (
                f"{E0_MODELS.index(model):02d}-{E0_TASKS.index(task):02d}.json"
            )
            binding = publish_e0_task_native_workload_authority(
                authority, output_path=path
            )
            workload_bindings.append(binding)
            workload_by_key[key] = authority
            workload_binding_by_key[key] = binding
    if tuple(workload_by_key) != _expected_workload_keys():
        raise RuntimeError("E0 materialized workload coverage is not canonical 36")

    worker_path = Path(__file__).resolve(strict=True)
    worker_source = EvidenceFileBinding.bind(
        worker_path, label="E0 compatibility worker source"
    )
    invoked_python_path = Path(sys.executable)
    if (
        not invoked_python_path.is_absolute()
        or any(character.isspace() for character in str(invoked_python_path))
        or not invoked_python_path.exists()
    ):
        raise ValueError("E0 compatibility Python launcher path is unavailable")
    python_path = invoked_python_path.resolve(strict=True)
    python_evidence = EvidenceFileBinding.bind(
        python_path, label="E0 compatibility Python"
    )
    source_root = worker_path.parent.parent.parent
    launcher_path = root / "e0-compatibility-worker"
    launcher = _publish_raw_no_replace(
        launcher_path,
        (
            f"#!{invoked_python_path}\n"
            "import sys\n"
            f"sys.path.insert(0, {str(source_root)!r})\n"
            f"from {FORMAL_E0_COMPATIBILITY_WORKER_MODULE} import main\n"
            "raise SystemExit(main())\n"
        ).encode(),
    )
    launcher_path.chmod(0o700)
    launcher.reopen(label="E0 compatibility worker launcher")
    groups_root = root / "groups"
    groups_root.mkdir(mode=0o700)
    plan_bindings: list[CanonicalJsonProofBinding] = []
    group_bindings: list[CanonicalJsonProofBinding] = []
    for model, backend in _expected_interface_keys():
        interface = interfaces[(model, backend)]
        interface_binding = interface_binding_by_key[(model, backend)]
        launch_binding = interface.compile_launch_manifest
        group_gpu_uuid = (
            None
            if launch_binding is None
            else CompileLaunchManifest.load(launch_binding.absolute_path).gpu_uuids[0]
        )
        group_index = E0_MODELS.index(model) * len(E0_BACKENDS) + E0_BACKENDS.index(
            backend
        )
        group_root = groups_root / f"group-{group_index:02d}"
        group_root.mkdir(mode=0o700)
        probes_root = group_root / "probes"
        probes_root.mkdir(mode=0o700)
        group_plan_path = group_root / "group-plan.json"
        argv = (
            launcher.absolute_path,
            "execute-group",
            "--group-plan",
            str(group_plan_path),
        )
        argv_sha = content_sha256({"argv": list(argv)})
        group_probe_bindings: list[CanonicalJsonProofBinding] = []
        for task in E0_TASKS:
            workload = workload_by_key[(model, task)]
            source = sources[task]
            sample_id, prompt, seed, *_ = _load_probe_source(source, task=task)
            gpu_execution = (
                interface.support_status == "READY"
                and workload.support_status == "READY"
            )
            probe_root = probes_root / f"task-{E0_TASKS.index(task):02d}"
            probe_root.mkdir(mode=0o700)
            plan = E0CompatibilityProbePlan(
                schema_version=1,
                kind="formal_single_operator_e0_compatibility_probe_plan",
                protocol_sha256=FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
                protocol_lock=lock_binding,
                e6_completion=e6_binding,
                content_source=content,
                inventory=inventory_binding,
                doctor=doctor_binding,
                interface_receipt=interface_binding,
                workload_authority=workload_binding_by_key[(model, task)],
                workload_source=source,
                model=model,
                backend=backend,
                task=task,
                source_sample_id=sample_id,
                source_prompt_sha256=content_sha256(prompt),
                source_seed=seed,
                execution_kind=(
                    "GPU_ONE_REQUEST" if gpu_execution else "CODE_VALIDATION_ONLY"
                ),
                gpu_uuid=group_gpu_uuid if gpu_execution else None,
                topology_mode="tp1_dp1",
                worker_command_sha256=argv_sha,
                group_plan_path=str(group_plan_path),
                terminal_output_path=str(probe_root / "probe-terminal.json"),
                evidence_directory=str(probe_root),
            )
            plan_path = probe_root / "probe-plan.json"
            publish_canonical_json_no_replace(plan_path, plan.to_dict())
            plan_binding = CanonicalJsonProofBinding.bind(
                plan_path, semantic_sha256=plan.sha256
            )
            plan_bindings.append(plan_binding)
            group_probe_bindings.append(plan_binding)
        group = E0CompatibilityProbeGroupPlan(
            schema_version=1,
            kind="formal_single_operator_e0_compatibility_probe_group",
            protocol_sha256=FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
            model=model,
            backend=backend,
            interface_receipt=interface_binding,
            compile_launch_manifest=launch_binding,
            inventory=inventory_binding,
            doctor=doctor_binding,
            python_executable=python_evidence,
            worker_launcher=launcher,
            worker_source=worker_source,
            argv=argv,
            argv_sha256=argv_sha,
            gpu_uuid=group_gpu_uuid,
            topology_mode="tp1_dp1",
            probes=tuple(group_probe_bindings),
            evidence_directory=str(group_root),
            completion_output_path=str(group_root / "group-complete.json"),
        )
        publish_canonical_json_no_replace(group_plan_path, group.to_dict())
        group_bindings.append(
            CanonicalJsonProofBinding.bind(
                group_plan_path, semantic_sha256=group.sha256
            )
        )
    campaign = E0CompatibilityPhysicalCampaign(
        schema_version=1,
        kind="formal_single_operator_e0_compatibility_physical_campaign",
        protocol_sha256=FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
        protocol_lock_sha256=lock.sha256,
        e6_materialization_sha256=e6_materialization,
        e6_confirmation_sha256=e6_confirmation,
        trusted_content_sha256=content.content_sha256,
        inventory_sha256=inventory.sha256,
        interface_receipts=interface_bindings,
        workload_authorities=tuple(workload_bindings),
        probe_plans=tuple(
            sorted(
                plan_bindings,
                key=lambda row: _probe_key(
                    E0CompatibilityProbePlan.from_dict(row.reopen())
                ),
            )
        ),
        groups=tuple(group_bindings),
        physical_probe_count=108,
        launch_group_count=12,
    )
    campaign_path = root / "campaign.json"
    publish_canonical_json_no_replace(campaign_path, campaign.to_dict())
    return revalidate_formal_e0_compatibility_physical_campaign(campaign_path)


def revalidate_formal_e0_compatibility_probe_plan(
    path: str | Path,
) -> E0CompatibilityProbePlan:
    binding = CanonicalJsonProofBinding.bind(path)
    plan = E0CompatibilityProbePlan.from_dict(binding.reopen())
    if plan.sha256 != binding.semantic_sha256:
        raise ValueError("E0 probe plan binding differs")
    lock = protocol_lock_from_dict(
        plan.protocol_lock.reopen(label="E0 probe ProtocolLock")
    )
    completion = rebuild_formal_single_operator_stage_completion(
        plan.e6_completion.absolute_path
    )
    _e6_identity(protocol_lock=lock, completion=completion)
    bundle = plan.content_source.reopen()
    inventory = GpuInventory.from_dict(plan.inventory.reopen())
    interface = load_e0_prepared_model_backend_interface_receipt(
        plan.interface_receipt.absolute_path
    )
    workload = load_e0_task_native_workload_authority(
        plan.workload_authority.absolute_path
    )
    group_binding = CanonicalJsonProofBinding.bind(plan.group_plan_path)
    group = E0CompatibilityProbeGroupPlan.from_dict(group_binding.reopen())
    if group.sha256 != group_binding.semantic_sha256:
        raise ValueError("E0 physical probe group binding differs")
    sample_id, prompt, seed, revision_sha, evidence_sha, source_status = (
        _load_probe_source(plan.workload_source, task=plan.task)
    )
    expected_execution = (
        "GPU_ONE_REQUEST"
        if interface.support_status == "READY" and workload.support_status == "READY"
        else "CODE_VALIDATION_ONLY"
    )
    launch_binding = interface.compile_launch_manifest
    if expected_execution == "GPU_ONE_REQUEST":
        if launch_binding is None:
            raise ValueError("READY E0 probe lacks its compile launch")
        expected_gpu = CompileLaunchManifest.load(
            launch_binding.absolute_path
        ).gpu_uuids[0]
    else:
        expected_gpu = None
    if (
        lock.schema_version != 5
        or lock.trusted_single_operator_content_bundle_sha256
        != plan.content_source.content_sha256
        or type(bundle) is not TrustedSingleOperatorContentBundle
        or bundle.runtime_binding_status != "BOUND"
        or bundle.runtime_observations is None
        or inventory.sha256 != plan.inventory.semantic_sha256
        or interface.sha256 != plan.interface_receipt.semantic_sha256
        or (interface.model, interface.backend) != (plan.model, plan.backend)
        or workload.sha256 != plan.workload_authority.semantic_sha256
        or (workload.model, workload.task) != (plan.model, plan.task)
        or interface.protocol_lock_sha256 != lock.sha256
        or workload.protocol_lock_sha256 != lock.sha256
        or interface.tokenizer_sha256 != workload.tokenizer_sha256
        or workload.source_revision_sha256 != revision_sha
        or workload.evidence_sha256 != evidence_sha
        or workload.support_status != source_status
        or sample_id != plan.source_sample_id
        or content_sha256(prompt) != plan.source_prompt_sha256
        or seed != plan.source_seed
        or plan.execution_kind != expected_execution
        or plan.gpu_uuid != expected_gpu
        or (group.model, group.backend) != (plan.model, plan.backend)
        or binding not in group.probes
        or plan.worker_command_sha256 != group.argv_sha256
        or Path(plan.terminal_output_path).parent != Path(plan.evidence_directory)
        or Path(binding.absolute_path).parent != Path(plan.evidence_directory)
    ):
        raise ValueError("E0 compatibility probe plan replay differs")
    if interface.support_status == "READY":
        if launch_binding is None:
            raise ValueError("READY E0 interface lacks its compile launch")
        launch = CompileLaunchManifest.load(launch_binding.absolute_path)
        _doctor_tools(binding=plan.doctor, inventory=inventory, launch=launch)
    return plan


def revalidate_formal_e0_compatibility_probe_group(
    path: str | Path,
) -> E0CompatibilityProbeGroupPlan:
    binding = CanonicalJsonProofBinding.bind(path)
    group = E0CompatibilityProbeGroupPlan.from_dict(binding.reopen())
    if group.sha256 != binding.semantic_sha256:
        raise ValueError("E0 compatibility group binding differs")
    group.python_executable.reopen(label="E0 compatibility Python")
    group.worker_launcher.reopen(label="E0 compatibility worker launcher")
    group.worker_source.reopen(label="E0 compatibility worker source")
    if Path(sys.executable).resolve(strict=True) != Path(
        group.python_executable.absolute_path
    ):
        raise ValueError("E0 compatibility worker Python identity differs")
    launcher_mode = stat.S_IMODE(
        Path(group.worker_launcher.absolute_path).stat(follow_symlinks=False).st_mode
    )
    if launcher_mode != 0o700:
        raise ValueError("E0 compatibility worker launcher mode differs")
    if Path(group.worker_source.absolute_path).resolve(strict=True) != Path(
        __file__
    ).resolve(strict=True):
        raise ValueError("E0 compatibility worker resolves to another source")
    interface = load_e0_prepared_model_backend_interface_receipt(
        group.interface_receipt.absolute_path
    )
    plans = tuple(
        revalidate_formal_e0_compatibility_probe_plan(row.absolute_path)
        for row in group.probes
    )
    if (
        (interface.model, interface.backend) != (group.model, group.backend)
        or tuple(plan.task for plan in plans) != E0_TASKS
        or any(
            (plan.model, plan.backend) != (group.model, group.backend)
            or plan.worker_command_sha256 != group.argv_sha256
            or plan.group_plan_path != binding.absolute_path
            for plan in plans
        )
    ):
        raise ValueError("E0 compatibility group probe coverage differs")
    inventory = GpuInventory.from_dict(group.inventory.reopen())
    if interface.support_status == "READY":
        if (
            interface.compile_launch_manifest is None
            or group.compile_launch_manifest != interface.compile_launch_manifest
        ):
            raise ValueError("READY E0 group launch differs")
        launch = CompileLaunchManifest.load(group.compile_launch_manifest.absolute_path)
        if launch.inventory_sha256 != inventory.sha256 or launch.gpu_uuids != (
            group.gpu_uuid,
        ):
            raise ValueError("E0 group GPU assignment differs")
        _doctor_tools(binding=group.doctor, inventory=inventory, launch=launch)
    elif group.compile_launch_manifest is not None or group.gpu_uuid is not None:
        raise ValueError("unsupported E0 group carries executable resources")
    return group


def revalidate_formal_e0_compatibility_physical_campaign(
    path: str | Path,
) -> E0CompatibilityPhysicalCampaign:
    binding = CanonicalJsonProofBinding.bind(path)
    campaign = E0CompatibilityPhysicalCampaign.from_dict(binding.reopen())
    if campaign.sha256 != binding.semantic_sha256:
        raise ValueError("E0 physical campaign binding differs")
    interfaces = tuple(
        load_e0_prepared_model_backend_interface_receipt(row.absolute_path)
        for row in campaign.interface_receipts
    )
    workloads = tuple(
        load_e0_task_native_workload_authority(row.absolute_path)
        for row in campaign.workload_authorities
    )
    plans = tuple(
        revalidate_formal_e0_compatibility_probe_plan(row.absolute_path)
        for row in campaign.probe_plans
    )
    groups = tuple(
        revalidate_formal_e0_compatibility_probe_group(row.absolute_path)
        for row in campaign.groups
    )
    if (
        tuple(_interface_key(row) for row in interfaces) != _expected_interface_keys()
        or tuple(_workload_key(row) for row in workloads) != _expected_workload_keys()
        or tuple(_probe_key(row) for row in plans)
        != expected_e0_compatibility_probe_keys()
        or tuple((row.model, row.backend) for row in groups)
        != _expected_interface_keys()
        or {row.protocol_lock_sha256 for row in interfaces}
        != {campaign.protocol_lock_sha256}
        or {row.protocol_lock_sha256 for row in workloads}
        != {campaign.protocol_lock_sha256}
        or {row.upstream_e6_confirmation_sha256 for row in interfaces}
        != {campaign.e6_confirmation_sha256}
        or {row.upstream_e6_confirmation_sha256 for row in workloads}
        != {campaign.e6_confirmation_sha256}
    ):
        raise ValueError("E0 physical campaign deep coverage differs")
    return campaign


def formal_e0_compatibility_process_hard_timeout_ns(
    campaign_path: str | Path,
) -> int:
    """Deep-replay exact-108/exact-12 coverage and return the worker cap."""

    campaign = revalidate_formal_e0_compatibility_physical_campaign(campaign_path)
    groups = tuple(
        revalidate_formal_e0_compatibility_probe_group(row.absolute_path)
        for row in campaign.groups
    )
    if len(groups) != 12 or any(len(group.probes) != 9 for group in groups):
        raise ValueError("E0 compatibility timeout requires exact 12x9 coverage")
    executable_groups = sum(
        group.compile_launch_manifest is not None for group in groups
    )
    per_executable_group_seconds = (
        _SERVER_READY_TIMEOUT_SECONDS
        + 9 * _REQUEST_TIMEOUT_SECONDS
        + _TRANSPORT_ABORT_TIMEOUT_SECONDS
        + _PROCESS_GROUP_CLEANUP_SECONDS
        + _PROCESS_GROUP_SIGKILL_AND_EMPTY_BOUND_SECONDS
    )
    seconds = (
        executable_groups * per_executable_group_seconds
        + _CAMPAIGN_PUBLICATION_GRACE_SECONDS
    )
    if not seconds.is_integer() or seconds < 1:
        raise RuntimeError("E0 compatibility process timeout is invalid")
    return int(seconds) * 1_000_000_000


def _tokenize_prompt(launch: CompileLaunchManifest, prompt: str) -> tuple[int, ...]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        launch.tokenizer_snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded.get("input_ids")
    if (
        type(token_ids) is not list
        or not token_ids
        or any(type(token) is not int or token < 0 for token in token_ids)
    ):
        raise ValueError("E0 compatibility tokenizer emitted invalid token IDs")
    config = load_run_config(launch.run_config_path)
    if len(token_ids) + 1 > config.runtime.context_length:
        raise FormalE0CompatibilityPhysicalBlocked(
            "task_native_probe_exceeds_server_context"
        )
    return tuple(token_ids)


def _bound_probe_request(
    *, plan: E0CompatibilityProbePlan, launch: CompileLaunchManifest, prompt: str
) -> BoundServingRequest:
    token_ids = _tokenize_prompt(launch, prompt)
    sampling = FrozenSamplingParameters.from_mapping(
        {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0}
    )
    cohort = f"e0-compatibility:{plan.model}:{plan.backend}"
    request = BoundServingRequest(
        request_id=f"e0-probe-{plan.probe_id}",
        namespace="formal-e0-compatibility",
        split="broad_replication",
        ordinal=0,
        input_token_ids=token_ids,
        requested_output_tokens=1,
        arrival_us=0,
        cancellation_offset_us=None,
        cohort_id=cohort,
        cohort_sha256=content_sha256(cohort),
        route_id="tp1-rank0",
        sampling=sampling,
    )
    request.validate()
    return request


def _require_port_unused(port: int) -> None:
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
    except OSError:
        return
    connection.close()
    raise FormalE0CompatibilityPhysicalBlocked(
        "e0_compatibility_launch_port_already_in_use"
    )


def _spawn_server(
    launch: CompileLaunchManifest,
    *,
    stdout_handle,
    stderr_handle,
) -> subprocess.Popen[bytes]:
    executable = Path(launch.server_argv[0])
    if not executable.is_absolute() or not executable.resolve(strict=True).is_file():
        raise ValueError("E0 compatibility server executable is unavailable")
    return subprocess.Popen(
        launch.server_argv,
        cwd=launch.patched_sglang_checkout,
        env=launch.child_environment(),
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
        close_fds=True,
    )


def _write_probe_junit(
    path: Path,
    *,
    plan: E0CompatibilityProbePlan,
    passed: bool,
    error_type: str | None,
) -> EvidenceFileBinding:
    suite = ET.Element(
        "testsuite",
        {
            "name": "formal_e0_compatibility_probe",
            "tests": "1",
            "failures": "0" if passed else "1",
            "errors": "0",
            "skipped": "0",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {"classname": f"{plan.model}.{plan.backend}", "name": plan.task},
    )
    if not passed:
        failure = ET.SubElement(case, "failure", {"type": error_type or "Error"})
        failure.text = "source-owned E0 compatibility probe failed"
    body = ET.tostring(suite, encoding="utf-8", xml_declaration=True) + b"\n"
    return _publish_raw_no_replace(path, body)


def _validate_probe_junit(path: Path, *, passed: bool) -> None:
    try:
        root = ET.fromstring(path.read_bytes())
    except (OSError, ET.ParseError) as error:
        raise ValueError("E0 compatibility JUnit is invalid") from error
    if (
        root.tag != "testsuite"
        or root.attrib.get("tests") != "1"
        or root.attrib.get("errors") != "0"
        or root.attrib.get("skipped") != "0"
        or root.attrib.get("failures") != ("0" if passed else "1")
        or len(root.findall("testcase")) != 1
    ):
        raise ValueError("E0 compatibility JUnit coverage differs")


def _attempt_root(group: E0CompatibilityProbeGroupPlan) -> Path:
    root = Path(group.evidence_directory) / "attempts"
    root.mkdir(mode=0o700, exist_ok=True)
    entries = tuple(sorted(root.iterdir(), key=lambda row: row.name))
    expected = tuple(f"attempt-{index:04d}" for index in range(1, len(entries) + 1))
    if tuple(row.name for row in entries) != expected or any(
        not row.is_dir() or row.is_symlink() for row in entries
    ):
        raise ValueError("E0 compatibility attempt history is non-canonical")
    attempt = root / f"attempt-{len(entries) + 1:04d}"
    attempt.mkdir(mode=0o700)
    return attempt


def _completed_terminal(
    plan: E0CompatibilityProbePlan,
) -> E0CompatibilityProbeTerminal | None:
    path = Path(plan.terminal_output_path)
    if not os.path.lexists(path):
        return None
    terminal = revalidate_formal_e0_compatibility_physical_terminal(path)
    if (
        terminal.terminal_status != "COMPLETE"
        or terminal.exit_code != 0
        or terminal.junit_status != "PASS"
        or (terminal.model, terminal.backend, terminal.task)
        != (plan.model, plan.backend, plan.task)
        or terminal.command_sha256 != plan.worker_command_sha256
    ):
        raise ValueError("existing E0 compatibility terminal is not resumable")
    return terminal


def _publish_or_revalidate_group_completion(
    *,
    group_path: str | Path,
    group: E0CompatibilityProbeGroupPlan,
    plans: tuple[E0CompatibilityProbePlan, ...],
    terminals: tuple[E0CompatibilityProbeTerminal, ...],
    physical_server_launch_count_this_attempt: Literal[0, 1],
) -> CanonicalJsonProofBinding:
    if len(plans) != 9 or len(terminals) != 9:
        raise ValueError("E0 group completion requires exact nine probes")
    path = Path(group.completion_output_path)
    if not os.path.lexists(path):
        publish_canonical_json_no_replace(
            path,
            {
                "schema_version": 1,
                "kind": ("formal_single_operator_e0_compatibility_group_completion"),
                "protocol_sha256": (FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256),
                "group_plan": CanonicalJsonProofBinding.bind(
                    group_path, semantic_sha256=group.sha256
                ).to_dict(),
                "command_sha256": group.argv_sha256,
                "terminal_paths": [plan.terminal_output_path for plan in plans],
                "terminal_sha256s": [row.sha256 for row in terminals],
                "completed_probe_count": 9,
                "physical_server_launch_count_this_attempt": (
                    physical_server_launch_count_this_attempt
                ),
                "status": "COMPLETE",
            },
        )
    return _revalidate_group_completion(group)


def _disposition(
    *,
    interface: E0PreparedModelBackendInterfaceReceipt,
    workload: E0TaskNativeWorkloadAuthority,
) -> tuple[
    Literal["VALID", "N/A"],
    str,
    Literal["PASS", "NOT_REQUIRED"],
    int,
]:
    if interface.support_status == "UNSUPPORTED":
        return "N/A", "MODEL_BACKEND_INTERFACE_UNSUPPORTED", "NOT_REQUIRED", 0
    if workload.support_status == "UNSUPPORTED":
        return "N/A", "TOKENIZER_TASK_WORKLOAD_UNSUPPORTED", "NOT_REQUIRED", 0
    return "VALID", E0_COMPATIBILITY_VALID_REASON, "PASS", 1


@dataclass(frozen=True)
class _ProbeExecution:
    plan: E0CompatibilityProbePlan
    started_ns: int
    request_finished_ns: int
    result: dict[str, object] | None
    input_token_ids_sha256: str | None


async def _run_gpu_probes(
    *,
    plans: tuple[E0CompatibilityProbePlan, ...],
    launch: CompileLaunchManifest,
    transport: PinnedBenchServingTransport,
) -> tuple[_ProbeExecution, ...]:
    base_url = f"http://127.0.0.1:{launch.localhost_port}"
    rows: list[_ProbeExecution] = []
    for plan in plans:
        _sample, prompt, _seed, *_ = _load_probe_source(
            plan.workload_source, task=plan.task
        )
        request = _bound_probe_request(plan=plan, launch=launch, prompt=prompt)
        started = _clock_after()
        result = await transport.submit(
            request,
            base_url=base_url,
            served_model=load_run_config(launch.run_config_path).model.target,
        )
        finished = _clock_after(started)
        result.validate(request)
        if (
            not result.success
            or result.output_tokens != 1
            or len(result.generated_token_ids) != 1
        ):
            raise FormalE0CompatibilityPhysicalBlocked(
                "one_request_gpu_smoke_did_not_complete"
            )
        rows.append(
            _ProbeExecution(
                plan=plan,
                started_ns=started,
                request_finished_ns=finished,
                result={
                    "request_id": request.request_id,
                    "request_sha256": request.sha256,
                    "input_token_count": len(request.input_token_ids),
                    "input_token_ids_sha256": content_sha256(
                        list(request.input_token_ids)
                    ),
                    "output_token_count": result.output_tokens,
                    "output_token_ids": list(result.generated_token_ids),
                    "output_token_ids_sha256": content_sha256(
                        list(result.generated_token_ids)
                    ),
                    "latency_us": result.latency_us,
                    "ttft_us": result.ttft_us,
                    "stop_reason": result.stop_reason,
                    "transport_metrics": transport.metrics(),
                },
                input_token_ids_sha256=content_sha256(list(request.input_token_ids)),
            )
        )
    return tuple(rows)


def _code_only_execution(plan: E0CompatibilityProbePlan) -> _ProbeExecution:
    started = _clock_after()
    sample_id, prompt, seed, *_ = _load_probe_source(
        plan.workload_source, task=plan.task
    )
    if (
        sample_id != plan.source_sample_id
        or seed != plan.source_seed
        or content_sha256(prompt) != plan.source_prompt_sha256
    ):
        raise ValueError("code-only E0 probe source changed")
    finished = _clock_after(started)
    return _ProbeExecution(
        plan=plan,
        started_ns=started,
        request_finished_ns=finished,
        result=None,
        input_token_ids_sha256=None,
    )


def _publish_eagle3_postprobe_proof_row(
    *,
    plan: E0CompatibilityProbePlan,
    attempt_root: Path,
    interface: E0PreparedModelBackendInterfaceReceipt,
    group: E0CompatibilityProbeGroupPlan,
    core: CanonicalJsonProofBinding,
    result: CanonicalJsonProofBinding,
    lifecycle: CanonicalJsonProofBinding,
) -> CanonicalJsonProofBinding:
    """Publish one task authority only after its real request evidence exists."""

    if (
        plan.backend != "EAGLE3"
        or plan.execution_kind != "GPU_ONE_REQUEST"
        or interface.schema_version != 3
        or interface.backend != "EAGLE3"
        or interface.support_status != "READY"
        or interface.compile_launch_manifest is None
        or group.compile_launch_manifest != interface.compile_launch_manifest
        or plan.gpu_uuid is None
    ):
        raise ValueError("fresh EAGLE3 post-probe authority scope differs")
    launch = CompileLaunchManifest.load(interface.compile_launch_manifest.absolute_path)
    if launch.gpu_uuids != (plan.gpu_uuid,):
        raise ValueError("fresh EAGLE3 post-probe GPU scope differs")
    common = {
        "task": plan.task,
        "model": plan.model,
        "interface_sha256": interface.interface_sha256,
        "target_revision": interface.target_revision,
        "drafter_revision": interface.drafter_revision,
    }
    selector_value = {
        "schema_version": 1,
        "kind": "trusted_single_operator_e0_eagle3_postprobe_model_selector",
        **common,
        "core_evidence_sha256": core.semantic_sha256,
        "core_evidence": core.to_dict(),
        "result_sha256": result.semantic_sha256,
        "result": result.to_dict(),
    }
    selector_path = attempt_root / f"{plan.task}.eagle3-model-selector.json"
    publish_canonical_json_no_replace(selector_path, selector_value)
    selector = CanonicalJsonProofBinding.bind(selector_path)
    source_identity_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_e0_eagle3_postprobe_source_identity",
            "protocol_sha256": FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
            "plan_sha256": plan.sha256,
            "compile_launch_manifest_sha256": (
                interface.compile_launch_manifest.semantic_sha256
            ),
            "core_evidence_sha256": core.semantic_sha256,
            "result_sha256": result.semantic_sha256,
            "lifecycle_sha256": lifecycle.semantic_sha256,
        }
    )
    native_value = {
        "schema_version": 1,
        "kind": "trusted_single_operator_e0_eagle3_postprobe_native_gpu_proof",
        **common,
        "source_identity_sha256": source_identity_sha256,
        "inventory_sha256": group.inventory.semantic_sha256,
        "gpu_uuids": list(launch.gpu_uuids),
        "compile_launch_manifest_sha256": (
            interface.compile_launch_manifest.semantic_sha256
        ),
        "core_evidence_sha256": core.semantic_sha256,
        "core_evidence": core.to_dict(),
        "result_sha256": result.semantic_sha256,
        "result": result.to_dict(),
        "lifecycle_sha256": lifecycle.semantic_sha256,
        "lifecycle": lifecycle.to_dict(),
    }
    native_path = attempt_root / f"{plan.task}.eagle3-native-proof.json"
    publish_canonical_json_no_replace(native_path, native_value)
    native = CanonicalJsonProofBinding.bind(native_path)
    compatibility_value = {
        "schema_version": 1,
        "kind": ("trusted_single_operator_e0_eagle3_postprobe_compatibility_authority"),
        **common,
        "status": "COMPATIBLE",
        "reason_code": E0_COMPATIBILITY_VALID_REASON,
        "core_evidence_sha256": core.semantic_sha256,
        "core_evidence": core.to_dict(),
        "model_selector_sha256": selector.semantic_sha256,
        "model_selector": selector.to_dict(),
        "native_gpu_proof_sha256": native.semantic_sha256,
        "native_gpu_proof": native.to_dict(),
    }
    compatibility_path = attempt_root / f"{plan.task}.eagle3-compatibility.json"
    publish_canonical_json_no_replace(compatibility_path, compatibility_value)
    compatibility = CanonicalJsonProofBinding.bind(compatibility_path)
    execution_value = {
        "schema_version": 1,
        "kind": "trusted_single_operator_e0_eagle3_postprobe_execution_authority",
        "stage": "E0",
        **common,
        "inventory_sha256": group.inventory.semantic_sha256,
        "gpu_uuids": list(launch.gpu_uuids),
        "compile_launch_manifest_sha256": (
            interface.compile_launch_manifest.semantic_sha256
        ),
        "core_evidence_sha256": core.semantic_sha256,
        "compatibility_authority_sha256": compatibility.semantic_sha256,
        "model_selector_sha256": selector.semantic_sha256,
        "native_gpu_receipt_sha256": native.semantic_sha256,
    }
    execution_path = attempt_root / f"{plan.task}.eagle3-execution.json"
    publish_canonical_json_no_replace(execution_path, execution_value)
    execution = CanonicalJsonProofBinding.bind(execution_path)
    row = E0Eagle3RuntimeProofRow(
        schema_version=2,
        task=plan.task,
        execution_authority_sha256=execution.semantic_sha256,
        compatibility_authority_sha256=compatibility.semantic_sha256,
        model_selector_sha256=selector.semantic_sha256,
        native_gpu_proof_sha256=native.semantic_sha256,
        execution_authority=execution,
        compatibility_authority=compatibility,
        model_selector_authority=selector,
        native_gpu_proof=native,
    )
    row_path = attempt_root / f"{plan.task}.eagle3-proof-row.json"
    publish_canonical_json_no_replace(row_path, row.to_dict())
    return CanonicalJsonProofBinding.bind(row_path, semantic_sha256=row.sha256)


def _publish_probe_success(
    *,
    execution: _ProbeExecution,
    attempt_root: Path,
    group: E0CompatibilityProbeGroupPlan,
    interface: E0PreparedModelBackendInterfaceReceipt,
    workload: E0TaskNativeWorkloadAuthority,
    process_id: int | None,
    process_started_ns: int | None,
    server_ready_ns: int | None,
    process_exited_ns: int | None,
    before_snapshot: CanonicalJsonProofBinding | None,
    ready_snapshot: CanonicalJsonProofBinding | None,
    after_snapshot: CanonicalJsonProofBinding | None,
    server_stdout: EvidenceFileBinding | None,
    server_stderr: EvidenceFileBinding | None,
) -> E0CompatibilityProbeTerminal:
    plan = execution.plan
    destination = Path(plan.evidence_directory)
    disposition, reason, smoke, completed = _disposition(
        interface=interface, workload=workload
    )
    if (plan.execution_kind == "GPU_ONE_REQUEST") != (completed == 1):
        raise RuntimeError("E0 probe execution/disposition differs")
    finished = (
        _clock_after(execution.request_finished_ns)
        if process_exited_ns is None
        else process_exited_ns
    )
    if (
        execution.request_finished_ns <= execution.started_ns
        or finished <= execution.request_finished_ns
    ):
        raise RuntimeError("E0 probe lifecycle timestamps are non-monotonic")
    result_binding: CanonicalJsonProofBinding | None = None
    if execution.result is not None:
        result_value = {
            "schema_version": 1,
            "kind": "formal_single_operator_e0_compatibility_request_result",
            "protocol_sha256": FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
            "plan_sha256": plan.sha256,
            **execution.result,
        }
        result_path = attempt_root / f"{plan.task}.result.json"
        publish_canonical_json_no_replace(result_path, result_value)
        result_binding = CanonicalJsonProofBinding.bind(result_path)
    stdout = _publish_raw_no_replace(
        attempt_root / f"{plan.task}.stdout.log",
        _canonical_bytes(
            {
                "probe_id": plan.probe_id,
                "status": "COMPLETE",
                "completed_request_count": completed,
                "result_sha256": (
                    None if result_binding is None else result_binding.semantic_sha256
                ),
            }
        ),
    )
    stderr = _publish_raw_no_replace(
        attempt_root / f"{plan.task}.stderr.log",
        b"source-owned E0 compatibility probe stderr was empty\n",
    )
    junit = _write_probe_junit(
        attempt_root / f"{plan.task}.junit.xml",
        plan=plan,
        passed=True,
        error_type=None,
    )
    lifecycle_value = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_compatibility_probe_lifecycle",
        "protocol_sha256": FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
        "plan_sha256": plan.sha256,
        "group_sha256": group.sha256,
        "command_sha256": plan.worker_command_sha256,
        "execution_kind": plan.execution_kind,
        "gpu_uuid": plan.gpu_uuid,
        "topology_mode": plan.topology_mode,
        "process_id": process_id,
        "process_group_id": process_id,
        "process_started_ns": process_started_ns,
        "server_ready_ns": server_ready_ns,
        "request_started_ns": execution.started_ns,
        "request_finished_ns": execution.request_finished_ns,
        "process_exited_ns": process_exited_ns,
        "finished_ns": finished,
        "completed_request_count": completed,
        "status": "COMPLETE",
        "before_gpu_snapshot": (
            None if before_snapshot is None else before_snapshot.to_dict()
        ),
        "ready_gpu_snapshot": (
            None if ready_snapshot is None else ready_snapshot.to_dict()
        ),
        "after_gpu_snapshot": (
            None if after_snapshot is None else after_snapshot.to_dict()
        ),
    }
    lifecycle_path = attempt_root / f"{plan.task}.lifecycle.json"
    publish_canonical_json_no_replace(lifecycle_path, lifecycle_value)
    lifecycle = CanonicalJsonProofBinding.bind(lifecycle_path)
    core_value = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_compatibility_probe_evidence",
        "protocol_sha256": FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
        "group_sha256": group.sha256,
        "plan": CanonicalJsonProofBinding.bind(
            destination / "probe-plan.json", semantic_sha256=plan.sha256
        ).to_dict(),
        "interface_receipt": plan.interface_receipt.to_dict(),
        "workload_authority": plan.workload_authority.to_dict(),
        "lifecycle": lifecycle.to_dict(),
        "stdout": stdout.to_dict(),
        "stderr": stderr.to_dict(),
        "junit": junit.to_dict(),
        "result": None if result_binding is None else result_binding.to_dict(),
        "server_stdout": (None if server_stdout is None else server_stdout.to_dict()),
        "server_stderr": (None if server_stderr is None else server_stderr.to_dict()),
        "input_token_ids_sha256": execution.input_token_ids_sha256,
        "command_sha256": plan.worker_command_sha256,
        "started_ns": execution.started_ns,
        "finished_ns": finished,
        "completed_request_count": completed,
    }
    core_path = attempt_root / f"{plan.task}.evidence.json"
    publish_canonical_json_no_replace(core_path, core_value)
    core = CanonicalJsonProofBinding.bind(core_path)
    eagle3_proof: CanonicalJsonProofBinding | None = None
    if plan.backend == "EAGLE3" and disposition == "VALID":
        if result_binding is None:
            raise RuntimeError("VALID EAGLE3 probe lacks one-request result evidence")
        eagle3_proof = _publish_eagle3_postprobe_proof_row(
            plan=plan,
            attempt_root=attempt_root,
            interface=interface,
            group=group,
            core=core,
            result=result_binding,
            lifecycle=lifecycle,
        )
    terminal = E0CompatibilityProbeTerminal(
        schema_version=3,
        protocol_lock_sha256=plan.protocol_lock.semantic_sha256,
        upstream_e6_confirmation_sha256=interface.upstream_e6_confirmation_sha256,
        model=plan.model,
        backend=plan.backend,
        task=plan.task,
        interface_sha256=interface.interface_sha256,
        task_native_workload_sha256=workload.task_native_workload_sha256,
        tokenizer_sha256=interface.tokenizer_sha256,
        command_sha256=plan.worker_command_sha256,
        started_ns=execution.started_ns,
        finished_ns=finished,
        terminal_status="COMPLETE",
        exit_code=0,
        stdout_sha256=stdout.raw_sha256,
        stderr_sha256=stderr.raw_sha256,
        junit_sha256=junit.raw_sha256,
        junit_status="PASS",
        evidence_sha256=core.semantic_sha256,
        smoke_status=smoke,
        completed_request_count=completed,
        disposition=disposition,
        reason_code=reason,
        interface_receipt_sha256=interface.sha256,
        compile_launch_manifest_sha256=(
            None
            if interface.compile_launch_manifest is None
            else interface.compile_launch_manifest.semantic_sha256
        ),
        eagle3_runtime_proof_row_sha256=(
            None if eagle3_proof is None else eagle3_proof.semantic_sha256
        ),
        eagle3_runtime_proof_row=eagle3_proof,
    )
    publish_e0_compatibility_probe_terminal(
        terminal, output_path=plan.terminal_output_path
    )
    return revalidate_formal_e0_compatibility_physical_terminal(
        plan.terminal_output_path
    )


def _publish_group_failure(
    *,
    path: Path,
    group: E0CompatibilityProbeGroupPlan,
    attempt_started_ns: int,
    error: BaseException,
    process_id: int | None,
    process_exit_code: int | None,
    process_started_ns: int | None,
    server_ready_ns: int | None,
    process_exited_ns: int | None,
    completed_request_count: int,
    before_snapshot: CanonicalJsonProofBinding | None,
    ready_snapshot: CanonicalJsonProofBinding | None,
    after_snapshot: CanonicalJsonProofBinding | None,
    server_stdout: EvidenceFileBinding | None,
    server_stderr: EvidenceFileBinding | None,
) -> None:
    if type(completed_request_count) is not int or completed_request_count < 0:
        raise ValueError("E0 group failure completed count is invalid")
    last_observed = max(
        value
        for value in (
            attempt_started_ns,
            process_started_ns,
            server_ready_ns,
            process_exited_ns,
        )
        if value is not None
    )
    finished = _clock_after(last_observed)
    stdout = _publish_raw_no_replace(
        path.with_name("failure.stdout.log"),
        _canonical_bytes(
            {
                "model": group.model,
                "backend": group.backend,
                "status": "FAILED",
                "completed_request_count": completed_request_count,
            }
        ),
    )
    stderr = _publish_raw_no_replace(
        path.with_name("failure.stderr.log"),
        (
            "source-owned E0 compatibility failure; exception text omitted, "
            f"type={type(error).__name__}\n"
        ).encode("ascii"),
    )
    suite = ET.Element(
        "testsuite",
        {
            "name": "formal_e0_compatibility_probe_group",
            "tests": "1",
            "failures": "1",
            "errors": "0",
            "skipped": "0",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {"classname": f"{group.model}.{group.backend}", "name": "group"},
    )
    failure = ET.SubElement(case, "failure", {"type": type(error).__name__})
    failure.text = "source-owned E0 compatibility probe group failed"
    junit = _publish_raw_no_replace(
        path.with_name("failure.junit.xml"),
        ET.tostring(suite, encoding="utf-8", xml_declaration=True) + b"\n",
    )
    lifecycle_path = path.with_name("failure.lifecycle.json")
    publish_canonical_json_no_replace(
        lifecycle_path,
        {
            "schema_version": 1,
            "kind": "formal_single_operator_e0_compatibility_failure_lifecycle",
            "protocol_sha256": FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
            "group_sha256": group.sha256,
            "started_ns": attempt_started_ns,
            "process_started_ns": process_started_ns,
            "server_ready_ns": server_ready_ns,
            "process_exited_ns": process_exited_ns,
            "finished_ns": finished,
            "process_id": process_id,
            "process_group_id": process_id,
            "process_exit_code": process_exit_code,
            "completed_request_count": completed_request_count,
            "before_gpu_snapshot": (
                None if before_snapshot is None else before_snapshot.to_dict()
            ),
            "ready_gpu_snapshot": (
                None if ready_snapshot is None else ready_snapshot.to_dict()
            ),
            "after_gpu_snapshot": (
                None if after_snapshot is None else after_snapshot.to_dict()
            ),
            "status": "FAILED",
        },
    )
    lifecycle = CanonicalJsonProofBinding.bind(lifecycle_path)
    publish_canonical_json_no_replace(
        path,
        {
            "schema_version": 1,
            "kind": "formal_single_operator_e0_compatibility_group_failure",
            "protocol_sha256": FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256,
            "group_sha256": group.sha256,
            "command_sha256": group.argv_sha256,
            "started_ns": attempt_started_ns,
            "finished_ns": finished,
            "error_type": type(error).__name__,
            "reason_code": f"physical_probe_failed:{type(error).__name__}",
            "process_id": process_id,
            "process_exit_code": process_exit_code,
            "completed_request_count": completed_request_count,
            "stdout": stdout.to_dict(),
            "stderr": stderr.to_dict(),
            "junit": junit.to_dict(),
            "lifecycle": lifecycle.to_dict(),
            "server_stdout": (
                None if server_stdout is None else server_stdout.to_dict()
            ),
            "server_stderr": (
                None if server_stderr is None else server_stderr.to_dict()
            ),
            "status": "FAILED",
            "failure_is_na": False,
        },
    )


async def _execute_group_unlocked(
    group_path: str | Path,
) -> tuple[E0CompatibilityProbeTerminal, ...]:
    group = revalidate_formal_e0_compatibility_probe_group(group_path)
    plans = tuple(
        revalidate_formal_e0_compatibility_probe_plan(row.absolute_path)
        for row in group.probes
    )
    existing = tuple(_completed_terminal(plan) for plan in plans)
    if all(row is not None for row in existing):
        completed = tuple(existing)  # type: ignore[arg-type]
        _publish_or_revalidate_group_completion(
            group_path=group_path,
            group=group,
            plans=plans,
            terminals=completed,
            physical_server_launch_count_this_attempt=0,
        )
        return completed
    pending = tuple(
        plan for plan, terminal in zip(plans, existing, strict=True) if terminal is None
    )
    attempt = _attempt_root(group)
    attempt_started = _clock_after()
    interface = load_e0_prepared_model_backend_interface_receipt(
        group.interface_receipt.absolute_path
    )
    workload_by_task = {
        plan.task: load_e0_task_native_workload_authority(
            plan.workload_authority.absolute_path
        )
        for plan in pending
    }
    code_only = tuple(
        plan for plan in pending if plan.execution_kind == "CODE_VALIDATION_ONLY"
    )
    gpu_plans = tuple(
        plan for plan in pending if plan.execution_kind == "GPU_ONE_REQUEST"
    )
    executions: list[_ProbeExecution] = []
    launch: CompileLaunchManifest | None = None
    process: subprocess.Popen[bytes] | None = None
    transport: PinnedBenchServingTransport | None = None
    stdout_handle = None
    stderr_handle = None
    server_stdout: EvidenceFileBinding | None = None
    server_stderr: EvidenceFileBinding | None = None
    before_snapshot: CanonicalJsonProofBinding | None = None
    ready_snapshot: CanonicalJsonProofBinding | None = None
    after_snapshot: CanonicalJsonProofBinding | None = None
    process_started_ns: int | None = None
    server_ready_ns: int | None = None
    process_exited_ns: int | None = None
    process_exit_code: int | None = None
    execution_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    nvidia_tool: PinnedNvidiaSmiTool | None = None
    try:
        executions.extend(_code_only_execution(plan) for plan in code_only)
        if gpu_plans:
            if group.compile_launch_manifest is None or group.gpu_uuid is None:
                raise RuntimeError("GPU E0 probes lack an executable group launch")
            launch = CompileLaunchManifest.load(
                group.compile_launch_manifest.absolute_path
            )
            inventory = GpuInventory.from_dict(group.inventory.reopen())
            _python, nvidia_path = _doctor_tools(
                binding=group.doctor,
                inventory=inventory,
                launch=launch,
            )
            nvidia_tool = PinnedNvidiaSmiTool.bind(nvidia_path)
            before_snapshot = _capture_gpu_process_snapshot(
                tool=nvidia_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="before",
                output_path=attempt / "gpu-before.json",
            )
            validate_pinned_sglang_gpu_process_snapshot(
                before_snapshot,
                expected_tool=nvidia_tool,
                expected_gpu_uuids=launch.gpu_uuids,
                expected_inventory_sha256=launch.inventory_sha256,
                expected_phase="before",
            )
            stdout_path = attempt / "server.stdout.log"
            stderr_path = attempt / "server.stderr.log"
            stdout_handle = stdout_path.open("xb", buffering=0)
            stderr_handle = stderr_path.open("xb", buffering=0)
            stdout_handle.write(b"source-owned E0 compatibility server stdout\n")
            stderr_handle.write(b"source-owned E0 compatibility server stderr\n")
            _require_port_unused(launch.localhost_port)
            process_started_ns = _clock_after(attempt_started)
            process = await asyncio.to_thread(
                _spawn_server,
                launch,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
            )
            await asyncio.to_thread(
                _wait_server_ready,
                process,
                port=launch.localhost_port,
                timeout_seconds=_SERVER_READY_TIMEOUT_SECONDS,
            )
            server_ready_ns = _clock_after(process_started_ns)
            ready_snapshot = _capture_gpu_process_snapshot(
                tool=nvidia_tool,
                gpu_uuids=launch.gpu_uuids,
                inventory_sha256=launch.inventory_sha256,
                phase="ready",
                output_path=attempt / "gpu-ready.json",
                expected_server_process_group_ids=(process.pid,),
            )
            validate_pinned_sglang_gpu_process_snapshot(
                ready_snapshot,
                expected_tool=nvidia_tool,
                expected_gpu_uuids=launch.gpu_uuids,
                expected_inventory_sha256=launch.inventory_sha256,
                expected_phase="ready",
                expected_server_process_group_ids=(process.pid,),
            )
            transport = PinnedBenchServingTransport.from_checkout(
                launch.patched_sglang_checkout
            )
            if type(transport) is not PinnedBenchServingTransport:
                raise TypeError("E0 compatibility requires exact pinned transport")
            await transport.open(
                request_timeout_s=_REQUEST_TIMEOUT_SECONDS,
                abort_timeout_s=_TRANSPORT_ABORT_TIMEOUT_SECONDS,
            )
            for gpu_plan in gpu_plans:
                executions.extend(
                    await _run_gpu_probes(
                        plans=(gpu_plan,),
                        launch=launch,
                        transport=transport,
                    )
                )
    except BaseException as error:  # noqa: BLE001 - preserve every failure
        execution_error = error
    finally:
        if transport is not None:
            try:
                await transport.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_error = cleanup_error or error
        if process is not None:
            try:
                (
                    process_exit_code,
                    _cleanup,
                    process_exited_ns,
                ) = await asyncio.to_thread(_terminate_process_group, process)
            except BaseException as error:  # noqa: BLE001
                cleanup_error = cleanup_error or error
        for handle in (stdout_handle, stderr_handle):
            if handle is None:
                continue
            try:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_error = cleanup_error or error
        if launch is not None and nvidia_tool is not None:
            try:
                after_snapshot = _capture_gpu_process_snapshot(
                    tool=nvidia_tool,
                    gpu_uuids=launch.gpu_uuids,
                    inventory_sha256=launch.inventory_sha256,
                    phase="after",
                    output_path=attempt / "gpu-after.json",
                )
            except BaseException as error:  # noqa: BLE001
                cleanup_error = cleanup_error or error
        if (
            launch is not None
            and stdout_handle is not None
            and stderr_handle is not None
        ):
            try:
                server_stdout = EvidenceFileBinding.bind(
                    attempt / "server.stdout.log", label="E0 server stdout"
                )
                server_stderr = EvidenceFileBinding.bind(
                    attempt / "server.stderr.log", label="E0 server stderr"
                )
            except BaseException as error:  # noqa: BLE001
                cleanup_error = cleanup_error or error
    failure = execution_error or cleanup_error
    if failure is not None:
        _publish_group_failure(
            path=attempt / "failure.json",
            group=group,
            attempt_started_ns=attempt_started,
            error=failure,
            process_id=None if process is None else process.pid,
            process_exit_code=process_exit_code,
            process_started_ns=process_started_ns,
            server_ready_ns=server_ready_ns,
            process_exited_ns=process_exited_ns,
            completed_request_count=sum(row.result is not None for row in executions),
            before_snapshot=before_snapshot,
            ready_snapshot=ready_snapshot,
            after_snapshot=after_snapshot,
            server_stdout=server_stdout,
            server_stderr=server_stderr,
        )
        raise FormalE0CompatibilityPhysicalBlocked(
            f"physical_probe_failed:{type(failure).__name__}"
        ) from failure
    if launch is not None:
        if (
            process is None
            or process_exited_ns is None
            or process_exit_code not in {0, -signal.SIGTERM}
            or _process_group_exists(process.pid)
            or before_snapshot is None
            or ready_snapshot is None
            or after_snapshot is None
            or nvidia_tool is None
        ):
            raise RuntimeError("E0 compatibility server lifecycle is incomplete")
        validate_pinned_sglang_gpu_process_snapshot(
            before_snapshot,
            expected_tool=nvidia_tool,
            expected_gpu_uuids=launch.gpu_uuids,
            expected_inventory_sha256=launch.inventory_sha256,
            expected_phase="before",
        )
        validate_pinned_sglang_gpu_process_snapshot(
            ready_snapshot,
            expected_tool=nvidia_tool,
            expected_gpu_uuids=launch.gpu_uuids,
            expected_inventory_sha256=launch.inventory_sha256,
            expected_phase="ready",
            expected_server_process_group_ids=(process.pid,),
        )
        validate_pinned_sglang_gpu_process_snapshot(
            after_snapshot,
            expected_tool=nvidia_tool,
            expected_gpu_uuids=launch.gpu_uuids,
            expected_inventory_sha256=launch.inventory_sha256,
            expected_phase="after",
        )
    execution_by_task = {row.plan.task: row for row in executions}
    if set(execution_by_task) != {plan.task for plan in pending}:
        raise RuntimeError("E0 compatibility pending probe coverage differs")
    new_terminals = []
    for plan in pending:
        new_terminals.append(
            _publish_probe_success(
                execution=execution_by_task[plan.task],
                attempt_root=attempt,
                group=group,
                interface=interface,
                workload=workload_by_task[plan.task],
                process_id=(
                    process.pid
                    if plan.execution_kind == "GPU_ONE_REQUEST" and process is not None
                    else None
                ),
                process_started_ns=(
                    process_started_ns
                    if plan.execution_kind == "GPU_ONE_REQUEST"
                    else None
                ),
                server_ready_ns=(
                    server_ready_ns
                    if plan.execution_kind == "GPU_ONE_REQUEST"
                    else None
                ),
                process_exited_ns=(
                    process_exited_ns
                    if plan.execution_kind == "GPU_ONE_REQUEST"
                    else None
                ),
                before_snapshot=(
                    before_snapshot
                    if plan.execution_kind == "GPU_ONE_REQUEST"
                    else None
                ),
                ready_snapshot=(
                    ready_snapshot if plan.execution_kind == "GPU_ONE_REQUEST" else None
                ),
                after_snapshot=(
                    after_snapshot if plan.execution_kind == "GPU_ONE_REQUEST" else None
                ),
                server_stdout=(
                    server_stdout if plan.execution_kind == "GPU_ONE_REQUEST" else None
                ),
                server_stderr=(
                    server_stderr if plan.execution_kind == "GPU_ONE_REQUEST" else None
                ),
            )
        )
    all_terminals = tuple(
        revalidate_formal_e0_compatibility_physical_terminal(plan.terminal_output_path)
        for plan in plans
    )
    _publish_or_revalidate_group_completion(
        group_path=group_path,
        group=group,
        plans=plans,
        terminals=all_terminals,
        physical_server_launch_count_this_attempt=1 if launch is not None else 0,
    )
    del new_terminals
    return all_terminals


def execute_formal_e0_compatibility_probe_group(
    group_path: str | Path,
) -> tuple[E0CompatibilityProbeTerminal, ...]:
    """Execute/resume one fixed nine-task group under its one-GPU lock."""

    group = revalidate_formal_e0_compatibility_probe_group(group_path)
    root = Path(group.evidence_directory).parent.parent
    lock_name = content_sha256(
        {
            "gpu_uuid": (
                group.gpu_uuid
                if group.gpu_uuid is not None
                else f"code-only:{group.model}:{group.backend}"
            )
        }
    )
    lock_path = root / f".e0-compatibility-{lock_name}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FormalE0CompatibilityPhysicalBlocked(
                "e0_compatibility_gpu_group_busy"
            ) from error
        try:
            return asyncio.run(_execute_group_unlocked(group_path))
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def revalidate_formal_e0_compatibility_physical_terminal(
    path: str | Path,
) -> E0CompatibilityProbeTerminal:
    terminal = load_e0_compatibility_probe_terminal(path)
    root = Path(path).parent
    plan = revalidate_formal_e0_compatibility_probe_plan(root / "probe-plan.json")
    interface = load_e0_prepared_model_backend_interface_receipt(
        plan.interface_receipt.absolute_path
    )
    workload = load_e0_task_native_workload_authority(
        plan.workload_authority.absolute_path
    )
    group_binding = CanonicalJsonProofBinding.bind(plan.group_plan_path)
    group = E0CompatibilityProbeGroupPlan.from_dict(group_binding.reopen())
    if group.sha256 != group_binding.semantic_sha256:
        raise ValueError("E0 physical terminal group binding differs")
    if (
        terminal.schema_version != 3
        or terminal.terminal_status != "COMPLETE"
        or terminal.exit_code != 0
        or terminal.junit_status != "PASS"
        or terminal.interface_receipt_sha256 != interface.sha256
        or terminal.compile_launch_manifest_sha256
        != (
            None
            if interface.compile_launch_manifest is None
            else interface.compile_launch_manifest.semantic_sha256
        )
        or terminal.eagle3_runtime_proof_row_sha256
        != _eagle3_runtime_proof_row_sha256(
            interface,
            task=plan.task,
            terminal=terminal,
        )
    ):
        raise ValueError("E0 physical compatibility terminal lineage differs")
    attempts = Path(plan.group_plan_path).parent / "attempts"
    matches = []
    for attempt in attempts.iterdir():
        candidate = attempt / f"{plan.task}.evidence.json"
        if candidate.is_file() and not candidate.is_symlink():
            binding = CanonicalJsonProofBinding.bind(candidate)
            if binding.semantic_sha256 == terminal.evidence_sha256:
                matches.append(binding)
    if len(matches) != 1:
        raise ValueError("E0 physical terminal evidence coverage differs")
    evidence = matches[0].reopen()
    expected_fields = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "group_sha256",
        "plan",
        "interface_receipt",
        "workload_authority",
        "lifecycle",
        "stdout",
        "stderr",
        "junit",
        "result",
        "server_stdout",
        "server_stderr",
        "input_token_ids_sha256",
        "command_sha256",
        "started_ns",
        "finished_ns",
        "completed_request_count",
    }
    if type(evidence) is not dict or set(evidence) != expected_fields:
        raise ValueError("E0 physical evidence fields differ")
    lifecycle = CanonicalJsonProofBinding.from_dict(evidence["lifecycle"])
    lifecycle_value = lifecycle.reopen()
    lifecycle_fields = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "plan_sha256",
        "group_sha256",
        "command_sha256",
        "execution_kind",
        "gpu_uuid",
        "topology_mode",
        "process_id",
        "process_group_id",
        "process_started_ns",
        "server_ready_ns",
        "request_started_ns",
        "request_finished_ns",
        "process_exited_ns",
        "finished_ns",
        "completed_request_count",
        "status",
        "before_gpu_snapshot",
        "ready_gpu_snapshot",
        "after_gpu_snapshot",
    }
    stdout = EvidenceFileBinding.from_dict(evidence["stdout"], label="E0 stdout")
    stderr = EvidenceFileBinding.from_dict(evidence["stderr"], label="E0 stderr")
    junit = EvidenceFileBinding.from_dict(evidence["junit"], label="E0 JUnit")
    stdout.reopen(label="E0 stdout")
    stderr.reopen(label="E0 stderr")
    junit.reopen(label="E0 JUnit")
    _validate_probe_junit(Path(junit.absolute_path), passed=True)
    disposition, reason, smoke, completed = _disposition(
        interface=interface, workload=workload
    )
    gpu = plan.execution_kind == "GPU_ONE_REQUEST"
    if (
        evidence["schema_version"] != 1
        or evidence["kind"] != "formal_single_operator_e0_compatibility_probe_evidence"
        or evidence["protocol_sha256"]
        != FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256
        or evidence["group_sha256"] != group.sha256
        or CanonicalJsonProofBinding.from_dict(evidence["plan"]).semantic_sha256
        != plan.sha256
        or CanonicalJsonProofBinding.from_dict(evidence["interface_receipt"])
        != plan.interface_receipt
        or CanonicalJsonProofBinding.from_dict(evidence["workload_authority"])
        != plan.workload_authority
        or evidence["command_sha256"] != plan.worker_command_sha256
        or evidence["started_ns"] != terminal.started_ns
        or evidence["finished_ns"] != terminal.finished_ns
        or evidence["completed_request_count"] != completed
        or stdout.raw_sha256 != terminal.stdout_sha256
        or stderr.raw_sha256 != terminal.stderr_sha256
        or junit.raw_sha256 != terminal.junit_sha256
        or terminal.disposition != disposition
        or terminal.reason_code != reason
        or terminal.smoke_status != smoke
        or terminal.completed_request_count != completed
        or type(lifecycle_value) is not dict
        or set(lifecycle_value) != lifecycle_fields
        or lifecycle_value.get("schema_version") != 1
        or lifecycle_value.get("kind")
        != "formal_single_operator_e0_compatibility_probe_lifecycle"
        or lifecycle_value.get("protocol_sha256")
        != FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256
        or lifecycle_value.get("plan_sha256") != plan.sha256
        or lifecycle_value.get("command_sha256") != plan.worker_command_sha256
        or lifecycle_value.get("status") != "COMPLETE"
        or lifecycle_value.get("group_sha256") != evidence["group_sha256"]
        or lifecycle_value.get("execution_kind") != plan.execution_kind
        or lifecycle_value.get("gpu_uuid") != plan.gpu_uuid
        or lifecycle_value.get("topology_mode") != "tp1_dp1"
        or lifecycle_value.get("request_started_ns") != terminal.started_ns
        or lifecycle_value.get("finished_ns") != terminal.finished_ns
        or lifecycle_value.get("completed_request_count") != completed
        or (evidence["result"] is not None) != gpu
        or (evidence["server_stdout"] is not None) != gpu
        or (evidence["server_stderr"] is not None) != gpu
        or (evidence["input_token_ids_sha256"] is not None) != gpu
    ):
        raise ValueError("E0 physical terminal evidence replay differs")
    if gpu:
        process_id = lifecycle_value["process_id"]
        process_started = lifecycle_value["process_started_ns"]
        server_ready = lifecycle_value["server_ready_ns"]
        request_finished = lifecycle_value["request_finished_ns"]
        process_exited = lifecycle_value["process_exited_ns"]
        if (
            type(process_id) is not int
            or process_id < 1
            or lifecycle_value["process_group_id"] != process_id
            or any(
                type(value) is not int or value < 1
                for value in (
                    process_started,
                    server_ready,
                    request_finished,
                    process_exited,
                )
            )
            or not (
                process_started
                < server_ready
                <= terminal.started_ns
                < request_finished
                < process_exited
            )
            or process_exited != terminal.finished_ns
        ):
            raise ValueError("E0 physical GPU lifecycle timing differs")
        for name in (
            "before_gpu_snapshot",
            "ready_gpu_snapshot",
            "after_gpu_snapshot",
        ):
            CanonicalJsonProofBinding.from_dict(lifecycle_value[name]).reopen()
        result = CanonicalJsonProofBinding.from_dict(evidence["result"])
        result_value = result.reopen()
        server_stdout = EvidenceFileBinding.from_dict(
            evidence["server_stdout"], label="E0 server stdout"
        )
        server_stderr = EvidenceFileBinding.from_dict(
            evidence["server_stderr"], label="E0 server stderr"
        )
        server_stdout.reopen(label="E0 server stdout")
        server_stderr.reopen(label="E0 server stderr")
        if (
            type(result_value) is not dict
            or result_value.get("output_token_count") != 1
            or len(result_value.get("output_token_ids", [])) != 1
            or result_value.get("input_token_ids_sha256")
            != evidence["input_token_ids_sha256"]
        ):
            raise ValueError("E0 physical one-request result differs")
        if plan.backend == "EAGLE3":
            proof = e0_eagle3_runtime_proof_row_for_task(
                interface,
                task=plan.task,
                terminal=terminal,
            )
            native = proof.native_gpu_proof.reopen()
            if (
                type(native) is not dict
                or CanonicalJsonProofBinding.from_dict(native.get("core_evidence"))
                != matches[0]
                or CanonicalJsonProofBinding.from_dict(native.get("result")) != result
                or CanonicalJsonProofBinding.from_dict(native.get("lifecycle"))
                != lifecycle
                or native.get("inventory_sha256") != group.inventory.semantic_sha256
                or native.get("gpu_uuids") != [plan.gpu_uuid]
            ):
                raise ValueError("E0 physical EAGLE3 post-probe evidence differs")
    elif (
        any(
            lifecycle_value[name] is not None
            for name in (
                "process_id",
                "process_group_id",
                "process_started_ns",
                "server_ready_ns",
                "process_exited_ns",
                "before_gpu_snapshot",
                "ready_gpu_snapshot",
                "after_gpu_snapshot",
            )
        )
        or type(lifecycle_value["request_finished_ns"]) is not int
        or not (
            terminal.started_ns
            < lifecycle_value["request_finished_ns"]
            < terminal.finished_ns
        )
    ):
        raise ValueError("E0 physical code-only lifecycle differs")
    return terminal


def _revalidate_group_completion(
    group: E0CompatibilityProbeGroupPlan,
) -> CanonicalJsonProofBinding:
    binding = CanonicalJsonProofBinding.bind(group.completion_output_path)
    value = binding.reopen()
    expected_fields = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "group_plan",
        "command_sha256",
        "terminal_paths",
        "terminal_sha256s",
        "completed_probe_count",
        "physical_server_launch_count_this_attempt",
        "status",
    }
    plans = tuple(
        revalidate_formal_e0_compatibility_probe_plan(row.absolute_path)
        for row in group.probes
    )
    terminals = tuple(
        revalidate_formal_e0_compatibility_physical_terminal(plan.terminal_output_path)
        for plan in plans
    )
    if (
        type(value) is not dict
        or set(value) != expected_fields
        or value.get("schema_version") != 1
        or value.get("kind")
        != "formal_single_operator_e0_compatibility_group_completion"
        or value.get("protocol_sha256")
        != FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256
        or CanonicalJsonProofBinding.from_dict(value.get("group_plan")).semantic_sha256
        != group.sha256
        or value.get("command_sha256") != group.argv_sha256
        or value.get("terminal_paths") != [plan.terminal_output_path for plan in plans]
        or value.get("terminal_sha256s") != [row.sha256 for row in terminals]
        or value.get("completed_probe_count") != 9
        or value.get("physical_server_launch_count_this_attempt") not in {0, 1}
        or value.get("status") != "COMPLETE"
    ):
        raise ValueError("E0 compatibility group completion differs")
    return binding


def completed_e0_compatibility_terminal_paths(
    campaign_path: str | Path,
) -> tuple[str, ...]:
    """Deep-reopen all 12 group completions and return canonical 108 paths."""

    campaign = revalidate_formal_e0_compatibility_physical_campaign(campaign_path)
    plans = tuple(
        revalidate_formal_e0_compatibility_probe_plan(row.absolute_path)
        for row in campaign.probe_plans
    )
    for binding in campaign.groups:
        group = revalidate_formal_e0_compatibility_probe_group(binding.absolute_path)
        _revalidate_group_completion(group)
    paths = tuple(plan.terminal_output_path for plan in plans)
    if len(paths) != 108 or len(set(paths)) != 108:
        raise RuntimeError("E0 completed terminal coverage is not exact 108")
    return paths


def publish_completed_e0_compatibility_physical_campaign(
    *,
    campaign_path: str | Path,
    protocol_lock_path: str | Path,
    e6_completion_path: str | Path,
    onlinespec_source_authority: E0OnlineSpecSourceAuthority | None,
    bundle_output_path: str | Path,
    evidence_manifest_output_path: str | Path,
) -> E0CompatibilityPublication:
    """Reduce exact completed physical sources through the trusted publisher."""

    campaign = revalidate_formal_e0_compatibility_physical_campaign(campaign_path)
    lock_binding = FormalSingleOperatorJsonBinding.bind(
        protocol_lock_path, label="E0 completed ProtocolLock"
    )
    lock = protocol_lock_from_dict(
        lock_binding.reopen(label="E0 completed ProtocolLock")
    )
    completion_binding = FormalSingleOperatorJsonBinding.bind(
        e6_completion_path, label="E0 completed E6 completion"
    )
    completion = rebuild_formal_single_operator_stage_completion(
        completion_binding.absolute_path
    )
    materialization, confirmation = _e6_identity(
        protocol_lock=lock, completion=completion
    )
    if (
        lock.sha256 != campaign.protocol_lock_sha256
        or materialization != campaign.e6_materialization_sha256
        or confirmation != campaign.e6_confirmation_sha256
    ):
        raise ValueError("completed E0 campaign lineage differs")
    terminal_paths = completed_e0_compatibility_terminal_paths(campaign_path)
    return publish_trusted_e0_compatibility_probe_sources(
        protocol_lock=lock,
        e6_completion=completion,
        interface_receipt_paths=tuple(
            row.absolute_path for row in campaign.interface_receipts
        ),
        workload_authority_paths=tuple(
            row.absolute_path for row in campaign.workload_authorities
        ),
        probe_terminal_paths=terminal_paths,
        onlinespec_source_authority=onlinespec_source_authority,
        bundle_output_path=bundle_output_path,
        evidence_manifest_output_path=evidence_manifest_output_path,
    )


def e0_compatibility_group_worker_argv(
    group_plan_path: str | Path,
) -> tuple[str, ...]:
    """Return the exact no-scientific-knob command accepted by the operator."""

    group = revalidate_formal_e0_compatibility_probe_group(group_plan_path)
    return group.argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    execute = subparsers.add_parser("execute-group")
    execute.add_argument("--group-plan", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.operation != "execute-group":  # pragma: no cover - argparse owns it
        raise RuntimeError("unsupported E0 compatibility worker operation")
    execute_formal_e0_compatibility_probe_group(arguments.group_plan)
    return 0


__all__ = [
    "FORMAL_E0_COMPATIBILITY_PHYSICAL_PROTOCOL_SHA256",
    "E0CompatibilityPhysicalCampaign",
    "E0CompatibilityProbeGroupPlan",
    "E0CompatibilityProbePlan",
    "FormalE0CompatibilityPhysicalBlocked",
    "completed_e0_compatibility_terminal_paths",
    "e0_compatibility_group_worker_argv",
    "execute_formal_e0_compatibility_probe_group",
    "expected_e0_compatibility_probe_keys",
    "formal_e0_compatibility_process_hard_timeout_ns",
    "materialize_formal_e0_compatibility_physical_campaign",
    "publish_completed_e0_compatibility_physical_campaign",
    "revalidate_formal_e0_compatibility_physical_campaign",
    "revalidate_formal_e0_compatibility_physical_terminal",
    "revalidate_formal_e0_compatibility_probe_group",
    "revalidate_formal_e0_compatibility_probe_plan",
]


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
