"""Release-owned raw authority for E5 failure-injection correctness cells.

The industrial registry declares E5 failure cells, but a registry row is not an
actuation command and ``topology_failure_surface`` is only a locked-output name.
This module derives an exact, typed fault plan from a registry cell, binds that
plan to a non-symlink raw JSON file, and defines the atomic all-rank receipt
reducer that a future first-party actuator must satisfy.

The current source release deliberately has neither an allowlisted actuator nor
a release trusted signer.  Consequently ``require_failure_injection_authority``
blocks before returning an execution token, and receipt reduction can produce
only a named ``BLOCKED`` result.  Callers cannot inject a trust root, actuator
identity, cell identity, scenario, topology, or summary digest.
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

from lightcone_spec.experiments.registry import (
    E5_FAILURES,
    ExperimentCell,
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.runtime.attestation import (
    RELEASE_TRUSTED_ATTESTER_POLICY,
    require_release_trusted_attester_policy,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_JSON_BYTES = 8 * 1024 * 1024

FAILURE_INJECTION_FIRST_PARTY_ACTUATOR_UNAVAILABLE_REASON = (
    "failure_injection_first_party_actuator_unavailable"
)
FAILURE_INJECTION_TRUSTED_SIGNER_UNAVAILABLE_REASON = (
    "failure_injection_trusted_signer_unavailable"
)
FAILURE_INJECTION_RAW_RECEIPT_INCOMPLETE_REASON = (
    "failure_injection_raw_receipt_incomplete"
)
FAILURE_INJECTION_RAW_PLAN_AUTHORITY_REQUIRED_REASON = (
    "failure_injection_raw_plan_authority_required"
)
FAILURE_INJECTION_EXECUTION_LIFECYCLE_UNAVAILABLE_REASON = (
    "failure_injection_first_party_execution_lifecycle_unavailable"
)

# This is a source-owned execution allowlist, not a caller parameter.  A future
# reviewed release may add an exact ``(actuator_id, version_sha256)`` pair only
# together with its first-party implementation and device-level tests.
RELEASE_FAILURE_ACTUATORS: tuple[tuple[str, str], ...] = ()

FAILURE_INJECTION_REDUCER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e5_failure_injection_raw_reducer_protocol",
        "scope": "correctness_only_not_headline_performance",
        "source": "release_derived_registry_cell_and_path_bound_raw_receipt",
        "topologies": ["tp2_dp1", "two_replica_tp1_dp2"],
        "required_rank_coverage": "every_rank_in_every_topology",
        "lifecycle": ["arm", "trigger", "recover", "terminal"],
        "session_reuse": "forbidden_fresh_process_required",
        "trust": "source_owned_actuator_allowlist_and_release_signer_policy",
        "summary_authority": "forbidden",
    }
)

_SCENARIO_COUNTER = {
    "queue_saturation": "admission_rejections",
    "cancellation": "cancellations",
    "duplicate_retry": "duplicate_retry_suppressions",
    "nonfinite_candidate": "nonfinite_updates",
    "oom_candidate": "oom_events",
    "evidence_backpressure": "evidence_backpressure_events",
    "disk_quota": "disk_quota_events",
    "slow_rank": "slow_rank_events",
    "communicator_failure": "communicator_failures",
    "replica_drain": "replica_drain_events",
    "replica_restart": "replica_restart_events",
}
_UNIVERSAL_ZERO_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "partial_target_continuations",
    "retractions",
)


class FailureInjectionAuthorityBlocked(RuntimeError):
    """Raised before actuation when the source release cannot authorize it."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"failure-injection authority is BLOCKED: {reason}")
        self.reason = reason


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_safe_id(label: str, value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _strict_mapping(label: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _strict_keys(
    label: str,
    row: Mapping[str, object],
    expected: set[str],
) -> None:
    if "topology_failure_surface" in row:
        raise ValueError(
            "caller-supplied topology_failure_surface is not raw authority"
        )
    if set(row) != expected:
        missing = sorted(expected - set(row))
        extra = sorted(set(row) - expected)
        raise ValueError(f"{label} fields differ: missing={missing}, extra={extra}")


def _strict_int(label: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_sequence(label: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _resolved_regular_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} path is unavailable") from error
    if resolved != path:
        raise ValueError(f"{label} path must be resolved and non-symlink")
    return path


def _load_raw_json(path: Path, *, label: str) -> tuple[Mapping[str, object], str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_size <= 0 or before.st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{label} has an invalid size")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or identity(before) != identity(after)
            or identity(after) != identity(current)
        ):
            raise ValueError(f"{label} changed during coordinated read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = item
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _strict_mapping(label, value), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FailureExpectedCounter:
    name: str
    comparison: Literal["eq", "ge"]
    value: int

    def __post_init__(self) -> None:
        _require_safe_id("expected counter", self.name)
        if self.comparison not in {"eq", "ge"}:
            raise ValueError("expected counter comparison must be eq or ge")
        _strict_int("expected counter value", self.value)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "comparison": self.comparison,
            "value": self.value,
        }


@dataclass(frozen=True)
class FailureTopologyTarget:
    topology: Literal["tp2_dp1", "two_replica_tp1_dp2"]
    tensor_parallel_size: int
    data_parallel_size: int
    world_size: int
    target_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = {
            "tp2_dp1": (2, 1),
            "two_replica_tp1_dp2": (1, 2),
        }
        if self.topology not in expected:
            raise ValueError("failure topology is vague or unsupported")
        if (self.tensor_parallel_size, self.data_parallel_size) != expected[
            self.topology
        ]:
            raise ValueError("failure topology dimensions differ from release plan")
        if self.world_size != 2:
            raise ValueError("E5 failure topology must use exactly two ranks")
        if self.target_ranks != tuple(range(self.world_size)):
            raise ValueError("failure actuation must target every rank exactly once")

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "world_size": self.world_size,
            "target_ranks": list(self.target_ranks),
        }


@dataclass(frozen=True)
class FailureLifecycleWindow:
    phases: tuple[str, ...]
    arm_timeout_ms: int
    trigger_timeout_ms: int
    recover_timeout_ms: int
    terminal_timeout_ms: int
    fresh_process_required: bool

    def __post_init__(self) -> None:
        if self.phases != ("arm", "trigger", "recover", "terminal"):
            raise ValueError("failure lifecycle must be arm-trigger-recover-terminal")
        for name in (
            "arm_timeout_ms",
            "trigger_timeout_ms",
            "recover_timeout_ms",
            "terminal_timeout_ms",
        ):
            _strict_int(name, getattr(self, name), minimum=1)
        if self.fresh_process_required is not True:
            raise ValueError("E5 failure cells require a fresh process")

    def to_dict(self) -> dict[str, object]:
        return {
            "phases": list(self.phases),
            "arm_timeout_ms": self.arm_timeout_ms,
            "trigger_timeout_ms": self.trigger_timeout_ms,
            "recover_timeout_ms": self.recover_timeout_ms,
            "terminal_timeout_ms": self.terminal_timeout_ms,
            "fresh_process_required": self.fresh_process_required,
        }


@dataclass(frozen=True)
class ReleaseFailurePlan:
    schema_version: int
    kind: str
    registry_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    scenario: str
    topology_targets: tuple[FailureTopologyTarget, ...]
    lifecycle: FailureLifecycleWindow
    expected_counters: tuple[FailureExpectedCounter, ...]
    reducer_protocol_sha256: str
    correctness_only: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "e5_release_failure_plan":
            raise ValueError("failure plan schema is unsupported")
        for label, value in (
            ("failure registry", self.registry_sha256),
            ("failure cell ID", self.cell_id),
            ("failure cell declaration", self.cell_declaration_sha256),
            ("failure reducer protocol", self.reducer_protocol_sha256),
        ):
            _require_sha256(label, value)
        if self.scenario not in E5_FAILURES or self.scenario not in _SCENARIO_COUNTER:
            raise ValueError("failure scenario is vague or unregistered")
        if tuple(row.topology for row in self.topology_targets) != (
            "tp2_dp1",
            "two_replica_tp1_dp2",
        ):
            raise ValueError("failure plan must cover both registered topologies")
        names = tuple(row.name for row in self.expected_counters)
        expected_names = tuple(
            sorted((*_UNIVERSAL_ZERO_COUNTERS, _SCENARIO_COUNTER[self.scenario]))
        )
        if names != expected_names:
            raise ValueError("failure expected counters differ from release policy")
        if self.reducer_protocol_sha256 != FAILURE_INJECTION_REDUCER_PROTOCOL_SHA256:
            raise ValueError("failure plan uses another reducer protocol")
        if self.correctness_only is not True:
            raise ValueError("failure cells cannot authorize headline performance")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "cell_declaration_sha256": self.cell_declaration_sha256,
            "scenario": self.scenario,
            "topology_targets": [row.to_dict() for row in self.topology_targets],
            "lifecycle": self.lifecycle.to_dict(),
            "expected_counters": [row.to_dict() for row in self.expected_counters],
            "reducer_protocol_sha256": self.reducer_protocol_sha256,
            "correctness_only": self.correctness_only,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def release_failure_plan_for_cell(
    registry: ExperimentRegistry,
    cell: ExperimentCell,
) -> ReleaseFailurePlan:
    """Derive the only release plan for an exact registry-owned E5 cell."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("failure planning requires an exact ExperimentRegistry")
    if type(cell) is not ExperimentCell:
        raise TypeError("failure planning requires an exact ExperimentCell")
    matches = tuple(
        row for row in registry.cells_for("E5") if row.cell_id == cell.cell_id
    )
    if len(matches) != 1 or matches[0] != cell:
        raise ValueError("failure planning rejects caller-selected foreign cell IDs")
    identity = cell.identity
    if (
        identity.task != "failure_injection"
        or identity.method != "l0"
        or identity.topology != "tp2_and_two_replica"
        or not identity.arrival.startswith("failure:")
        or identity.slo != "excluded_from_headline"
        or cell.resources.gpu_count != 2
    ):
        raise ValueError("registry cell is not an exact E5 failure-injection cell")
    scenario = identity.arrival.removeprefix("failure:")
    if scenario not in E5_FAILURES or scenario not in _SCENARIO_COUNTER:
        raise ValueError("registry failure scenario is vague or unregistered")
    counters = [
        FailureExpectedCounter(name=name, comparison="eq", value=0)
        for name in _UNIVERSAL_ZERO_COUNTERS
    ]
    counters.append(
        FailureExpectedCounter(
            name=_SCENARIO_COUNTER[scenario], comparison="ge", value=1
        )
    )
    return ReleaseFailurePlan(
        schema_version=1,
        kind="e5_release_failure_plan",
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        cell_declaration_sha256=cell.sha256,
        scenario=scenario,
        topology_targets=(
            FailureTopologyTarget("tp2_dp1", 2, 1, 2, (0, 1)),
            FailureTopologyTarget("two_replica_tp1_dp2", 1, 2, 2, (0, 1)),
        ),
        lifecycle=FailureLifecycleWindow(
            phases=("arm", "trigger", "recover", "terminal"),
            arm_timeout_ms=30_000,
            trigger_timeout_ms=30_000,
            recover_timeout_ms=120_000,
            terminal_timeout_ms=30_000,
            fresh_process_required=True,
        ),
        expected_counters=tuple(sorted(counters, key=lambda row: row.name)),
        reducer_protocol_sha256=FAILURE_INJECTION_REDUCER_PROTOCOL_SHA256,
        correctness_only=True,
    )


def _plan_from_dict(value: object) -> ReleaseFailurePlan:
    row = _strict_mapping("failure plan", value)
    _strict_keys(
        "failure plan",
        row,
        {
            "schema_version",
            "kind",
            "registry_sha256",
            "cell_id",
            "cell_declaration_sha256",
            "scenario",
            "topology_targets",
            "lifecycle",
            "expected_counters",
            "reducer_protocol_sha256",
            "correctness_only",
        },
    )
    targets = []
    for item in _strict_sequence("failure topology targets", row["topology_targets"]):
        target = _strict_mapping("failure topology target", item)
        _strict_keys(
            "failure topology target",
            target,
            {
                "topology",
                "tensor_parallel_size",
                "data_parallel_size",
                "world_size",
                "target_ranks",
            },
        )
        ranks = tuple(
            _strict_int("failure target rank", rank)
            for rank in _strict_sequence("failure target ranks", target["target_ranks"])
        )
        targets.append(
            FailureTopologyTarget(
                topology=target["topology"],  # type: ignore[arg-type]
                tensor_parallel_size=_strict_int(
                    "failure TP size", target["tensor_parallel_size"], minimum=1
                ),
                data_parallel_size=_strict_int(
                    "failure DP size", target["data_parallel_size"], minimum=1
                ),
                world_size=_strict_int(
                    "failure world size", target["world_size"], minimum=1
                ),
                target_ranks=ranks,
            )
        )
    lifecycle_row = _strict_mapping("failure lifecycle", row["lifecycle"])
    _strict_keys(
        "failure lifecycle",
        lifecycle_row,
        {
            "phases",
            "arm_timeout_ms",
            "trigger_timeout_ms",
            "recover_timeout_ms",
            "terminal_timeout_ms",
            "fresh_process_required",
        },
    )
    counters = []
    for item in _strict_sequence("failure expected counters", row["expected_counters"]):
        counter = _strict_mapping("failure expected counter", item)
        _strict_keys(
            "failure expected counter", counter, {"name", "comparison", "value"}
        )
        counters.append(
            FailureExpectedCounter(
                name=counter["name"],  # type: ignore[arg-type]
                comparison=counter["comparison"],  # type: ignore[arg-type]
                value=_strict_int("expected counter value", counter["value"]),
            )
        )
    return ReleaseFailurePlan(
        schema_version=_strict_int("failure schema version", row["schema_version"]),
        kind=row["kind"],  # type: ignore[arg-type]
        registry_sha256=row["registry_sha256"],  # type: ignore[arg-type]
        cell_id=row["cell_id"],  # type: ignore[arg-type]
        cell_declaration_sha256=row["cell_declaration_sha256"],  # type: ignore[arg-type]
        scenario=row["scenario"],  # type: ignore[arg-type]
        topology_targets=tuple(targets),
        lifecycle=FailureLifecycleWindow(
            phases=tuple(
                str(item)
                for item in _strict_sequence("failure phases", lifecycle_row["phases"])
            ),
            arm_timeout_ms=_strict_int(
                "arm timeout", lifecycle_row["arm_timeout_ms"], minimum=1
            ),
            trigger_timeout_ms=_strict_int(
                "trigger timeout", lifecycle_row["trigger_timeout_ms"], minimum=1
            ),
            recover_timeout_ms=_strict_int(
                "recover timeout", lifecycle_row["recover_timeout_ms"], minimum=1
            ),
            terminal_timeout_ms=_strict_int(
                "terminal timeout", lifecycle_row["terminal_timeout_ms"], minimum=1
            ),
            fresh_process_required=lifecycle_row["fresh_process_required"],  # type: ignore[arg-type]
        ),
        expected_counters=tuple(counters),
        reducer_protocol_sha256=row["reducer_protocol_sha256"],  # type: ignore[arg-type]
        correctness_only=row["correctness_only"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class FailureInjectionAuthorityBinding:
    plan_path: str
    plan_raw_sha256: str
    plan_sha256: str
    registry_sha256: str
    cell_id: str
    scenario: str
    reducer_protocol_sha256: str

    def __post_init__(self) -> None:
        _resolved_regular_path(self.plan_path, label="failure plan")
        for label, value in (
            ("failure plan raw", self.plan_raw_sha256),
            ("failure plan", self.plan_sha256),
            ("failure registry", self.registry_sha256),
            ("failure cell", self.cell_id),
            ("failure reducer protocol", self.reducer_protocol_sha256),
        ):
            _require_sha256(label, value)
        if self.scenario not in E5_FAILURES:
            raise ValueError("failure authority scenario is unregistered")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "failure_injection_authority_binding",
                "plan_path": self.plan_path,
                "plan_raw_sha256": self.plan_raw_sha256,
                "plan_sha256": self.plan_sha256,
                "registry_sha256": self.registry_sha256,
                "cell_id": self.cell_id,
                "scenario": self.scenario,
                "reducer_protocol_sha256": self.reducer_protocol_sha256,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "failure_injection_authority_binding",
            "plan_path": self.plan_path,
            "plan_raw_sha256": self.plan_raw_sha256,
            "plan_sha256": self.plan_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "scenario": self.scenario,
            "reducer_protocol_sha256": self.reducer_protocol_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> FailureInjectionAuthorityBinding:
        row = _strict_mapping("failure authority binding", value)
        _strict_keys(
            "failure authority binding",
            row,
            {
                "schema_version",
                "kind",
                "plan_path",
                "plan_raw_sha256",
                "plan_sha256",
                "registry_sha256",
                "cell_id",
                "scenario",
                "reducer_protocol_sha256",
            },
        )
        if (
            type(row["schema_version"]) is not int
            or row["schema_version"] != 1
            or row["kind"] != ("failure_injection_authority_binding")
        ):
            raise ValueError("failure authority binding schema is unsupported")
        return cls(
            plan_path=row["plan_path"],  # type: ignore[arg-type]
            plan_raw_sha256=row["plan_raw_sha256"],  # type: ignore[arg-type]
            plan_sha256=row["plan_sha256"],  # type: ignore[arg-type]
            registry_sha256=row["registry_sha256"],  # type: ignore[arg-type]
            cell_id=row["cell_id"],  # type: ignore[arg-type]
            scenario=row["scenario"],  # type: ignore[arg-type]
            reducer_protocol_sha256=row["reducer_protocol_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class FailureInjectionAuthorityResult:
    binding: FailureInjectionAuthorityBinding
    plan: ReleaseFailurePlan


def bind_failure_injection_authority(
    plan_path: str | Path,
    *,
    registry: ExperimentRegistry,
) -> FailureInjectionAuthorityBinding:
    """Bind a raw plan only after rederiving it from its exact registry cell."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("failure binding requires an exact ExperimentRegistry")
    path = _resolved_regular_path(plan_path, label="failure plan")
    row, raw_sha256 = _load_raw_json(path, label="failure plan")
    plan = _plan_from_dict(row)
    matches = tuple(
        cell for cell in registry.cells_for("E5") if cell.cell_id == plan.cell_id
    )
    if len(matches) != 1:
        raise ValueError("failure plan names no unique registered E5 cell")
    expected = release_failure_plan_for_cell(registry, matches[0])
    if plan != expected:
        raise ValueError("serialized failure plan differs from release derivation")
    return FailureInjectionAuthorityBinding(
        plan_path=str(path),
        plan_raw_sha256=raw_sha256,
        plan_sha256=plan.sha256,
        registry_sha256=registry.sha256,
        cell_id=plan.cell_id,
        scenario=plan.scenario,
        reducer_protocol_sha256=plan.reducer_protocol_sha256,
    )


def revalidate_failure_injection_authority(
    binding: FailureInjectionAuthorityBinding,
    *,
    registry: ExperimentRegistry,
) -> FailureInjectionAuthorityResult:
    """Reopen the raw file and reject tamper, replacement, or identity drift."""

    if type(binding) is not FailureInjectionAuthorityBinding:
        raise TypeError("failure replay requires an exact authority binding")
    if type(registry) is not ExperimentRegistry:
        raise TypeError("failure replay requires an exact ExperimentRegistry")
    expected = bind_failure_injection_authority(binding.plan_path, registry=registry)
    if expected != binding:
        raise ValueError("failure authority differs from fresh raw replay")
    row, _ = _load_raw_json(Path(binding.plan_path), label="failure plan")
    return FailureInjectionAuthorityResult(binding=binding, plan=_plan_from_dict(row))


@dataclass(frozen=True)
class FailureExecutionAuthorityToken:
    authority_sha256: str
    plan_sha256: str
    registry_sha256: str
    cell_id: str
    scenario: str
    actuator_id: str
    actuator_version_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("failure execution authority", self.authority_sha256),
            ("failure execution plan", self.plan_sha256),
            ("failure execution registry", self.registry_sha256),
            ("failure execution cell", self.cell_id),
            ("failure actuator version", self.actuator_version_sha256),
        ):
            _require_sha256(label, value)
        if self.scenario not in E5_FAILURES:
            raise ValueError("failure execution scenario is unregistered")
        _require_safe_id("failure actuator", self.actuator_id)
        if (self.actuator_id, self.actuator_version_sha256) not in (
            RELEASE_FAILURE_ACTUATORS
        ):
            raise ValueError("failure execution token is not release-authorized")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "failure_execution_authority_token",
            "authority_sha256": self.authority_sha256,
            "plan_sha256": self.plan_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "scenario": self.scenario,
            "actuator_id": self.actuator_id,
            "actuator_version_sha256": self.actuator_version_sha256,
        }


def require_failure_injection_authority(
    binding: FailureInjectionAuthorityBinding,
    *,
    registry: ExperimentRegistry,
) -> FailureExecutionAuthorityToken:
    """Fail closed before actuation unless this release owns actuator and signer."""

    result = revalidate_failure_injection_authority(binding, registry=registry)
    if len(RELEASE_FAILURE_ACTUATORS) != 1:
        raise FailureInjectionAuthorityBlocked(
            FAILURE_INJECTION_FIRST_PARTY_ACTUATOR_UNAVAILABLE_REASON
        )
    policy = require_release_trusted_attester_policy(RELEASE_TRUSTED_ATTESTER_POLICY)
    if not policy.release_ready:
        raise FailureInjectionAuthorityBlocked(
            FAILURE_INJECTION_TRUSTED_SIGNER_UNAVAILABLE_REASON
        )
    actuator_id, actuator_version_sha256 = RELEASE_FAILURE_ACTUATORS[0]
    return FailureExecutionAuthorityToken(
        authority_sha256=result.binding.sha256,
        plan_sha256=result.plan.sha256,
        registry_sha256=result.plan.registry_sha256,
        cell_id=result.plan.cell_id,
        scenario=result.plan.scenario,
        actuator_id=actuator_id,
        actuator_version_sha256=actuator_version_sha256,
    )


def require_failure_execution_lifecycle(
    token: FailureExecutionAuthorityToken,
) -> None:
    """Keep failure jobs blocked until the actuator lifecycle is wired.

    A release allowlist entry and a signed raw plan are necessary but not
    sufficient: the executor must also own and invoke arm, trigger, recovery,
    and terminal-proof operations.  That implementation does not exist in
    this release, so merely adding an allowlist tuple must never turn a
    failure cell into an ordinary serving launch.
    """

    if type(token) is not FailureExecutionAuthorityToken:
        raise TypeError("failure execution lifecycle requires an exact token")
    raise FailureInjectionAuthorityBlocked(
        FAILURE_INJECTION_EXECUTION_LIFECYCLE_UNAVAILABLE_REASON
    )


@dataclass(frozen=True)
class FailureRankReceipt:
    rank: int
    process_id: int
    process_start_monotonic_ns: int
    session_epoch: int
    phases: tuple[tuple[str, int], ...]
    counters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _strict_int("failure rank", self.rank)
        _strict_int("failure process ID", self.process_id, minimum=1)
        _strict_int("failure process start", self.process_start_monotonic_ns, minimum=1)
        if self.session_epoch != 0:
            raise ValueError("fresh failure process must begin at session epoch zero")
        if tuple(name for name, _ in self.phases) != (
            "arm",
            "trigger",
            "recover",
            "terminal",
        ):
            raise ValueError("rank receipt phases are incomplete or unordered")
        times = tuple(value for _, value in self.phases)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in times
        ) or any(right <= left for left, right in pairwise(times)):
            raise ValueError("rank receipt phase times must be strictly increasing")
        if times[0] <= self.process_start_monotonic_ns:
            raise ValueError("failure arm must follow the fresh process start")
        names = tuple(name for name, _ in self.counters)
        if names != tuple(sorted(set(names))):
            raise ValueError("rank receipt counters must be sorted and unique")
        for name, value in self.counters:
            _require_safe_id("rank receipt counter", name)
            _strict_int("rank receipt counter value", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "process_id": self.process_id,
            "process_start_monotonic_ns": self.process_start_monotonic_ns,
            "session_epoch": self.session_epoch,
            "phases": [
                {"phase": phase, "monotonic_ns": value} for phase, value in self.phases
            ],
            "counters": {name: value for name, value in self.counters},
        }


@dataclass(frozen=True)
class FailureTopologyReceipt:
    topology: str
    rank_receipts: tuple[FailureRankReceipt, ...]

    def __post_init__(self) -> None:
        if self.topology not in {"tp2_dp1", "two_replica_tp1_dp2"}:
            raise ValueError("failure topology receipt is vague or unsupported")
        ranks = tuple(row.rank for row in self.rank_receipts)
        if not ranks or ranks != tuple(sorted(set(ranks))):
            raise ValueError("failure rank receipts must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "rank_receipts": [row.to_dict() for row in self.rank_receipts],
        }


@dataclass(frozen=True)
class AtomicFailureActuationReceipt:
    schema_version: int
    kind: str
    plan_sha256: str
    authority_sha256: str
    run_id: str
    run_nonce_sha256: str
    actuator_id: str
    actuator_version_sha256: str
    fresh_process: bool
    topologies: tuple[FailureTopologyReceipt, ...]
    terminal_status: str
    committed: bool
    attester_id: str
    trust_domain: str
    signature_hex: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "e5_atomic_failure_actuation_receipt"
        ):
            raise ValueError("failure actuation receipt schema is unsupported")
        for label, value in (
            ("receipt plan", self.plan_sha256),
            ("receipt authority", self.authority_sha256),
            ("receipt nonce", self.run_nonce_sha256),
            ("receipt actuator version", self.actuator_version_sha256),
        ):
            _require_sha256(label, value)
        for label, value in (
            ("receipt run", self.run_id),
            ("receipt actuator", self.actuator_id),
            ("receipt attester", self.attester_id),
        ):
            _require_safe_id(label, value)
        if self.fresh_process is not True:
            raise ValueError("failure receipt must prove a fresh process")
        if tuple(row.topology for row in self.topologies) != (
            "tp2_dp1",
            "two_replica_tp1_dp2",
        ):
            raise ValueError("failure receipt must cover both topologies")
        if self.terminal_status != "RECOVERED" or self.committed is not True:
            raise ValueError("failure receipt is not atomically terminal")
        if self.trust_domain != "hardware":
            raise ValueError("failure receipt requires hardware trust domain")
        if (
            not isinstance(self.signature_hex, str)
            or re.fullmatch(r"[0-9a-f]{128}", self.signature_hex) is None
        ):
            raise ValueError("failure receipt signature is not canonical Ed25519 hex")

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "plan_sha256": self.plan_sha256,
            "authority_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "run_nonce_sha256": self.run_nonce_sha256,
            "actuator_id": self.actuator_id,
            "actuator_version_sha256": self.actuator_version_sha256,
            "fresh_process": self.fresh_process,
            "topologies": [row.to_dict() for row in self.topologies],
            "terminal_status": self.terminal_status,
            "committed": self.committed,
            "attester_id": self.attester_id,
            "trust_domain": self.trust_domain,
        }

    @cached_property
    def payload_sha256(self) -> str:
        return content_sha256(self.payload_dict())

    @property
    def attestation_message(self) -> bytes:
        return json.dumps(
            {
                "schema_version": 1,
                "domain": "lightcone-e5-failure-actuation-receipt",
                "payload_sha256": self.payload_sha256,
                "run_nonce_sha256": self.run_nonce_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def _receipt_from_dict(value: object) -> AtomicFailureActuationReceipt:
    row = _strict_mapping("failure receipt", value)
    _strict_keys(
        "failure receipt",
        row,
        {
            "schema_version",
            "kind",
            "plan_sha256",
            "authority_sha256",
            "run_id",
            "run_nonce_sha256",
            "actuator_id",
            "actuator_version_sha256",
            "fresh_process",
            "topologies",
            "terminal_status",
            "committed",
            "attester_id",
            "trust_domain",
            "signature_hex",
        },
    )
    topologies = []
    for item in _strict_sequence("failure receipt topologies", row["topologies"]):
        topology = _strict_mapping("failure topology receipt", item)
        _strict_keys(
            "failure topology receipt", topology, {"topology", "rank_receipts"}
        )
        ranks = []
        for rank_item in _strict_sequence(
            "failure rank receipts", topology["rank_receipts"]
        ):
            rank = _strict_mapping("failure rank receipt", rank_item)
            _strict_keys(
                "failure rank receipt",
                rank,
                {
                    "rank",
                    "process_id",
                    "process_start_monotonic_ns",
                    "session_epoch",
                    "phases",
                    "counters",
                },
            )
            phases = []
            for phase_item in _strict_sequence("failure phases", rank["phases"]):
                phase = _strict_mapping("failure phase", phase_item)
                _strict_keys("failure phase", phase, {"phase", "monotonic_ns"})
                phases.append(
                    (
                        phase["phase"],  # type: ignore[arg-type]
                        _strict_int("failure phase time", phase["monotonic_ns"]),
                    )
                )
            counters_row = _strict_mapping("failure counters", rank["counters"])
            ranks.append(
                FailureRankReceipt(
                    rank=_strict_int("failure rank", rank["rank"]),
                    process_id=_strict_int(
                        "failure process ID", rank["process_id"], minimum=1
                    ),
                    process_start_monotonic_ns=_strict_int(
                        "failure process start",
                        rank["process_start_monotonic_ns"],
                        minimum=1,
                    ),
                    session_epoch=_strict_int(
                        "failure session epoch", rank["session_epoch"]
                    ),
                    phases=tuple(phases),
                    counters=tuple(
                        sorted(
                            (
                                (
                                    _require_safe_id("counter", name),
                                    _strict_int("counter", value),
                                )
                                for name, value in counters_row.items()
                            ),
                            key=lambda item: item[0],
                        )
                    ),
                )
            )
        topologies.append(
            FailureTopologyReceipt(
                topology=topology["topology"],  # type: ignore[arg-type]
                rank_receipts=tuple(ranks),
            )
        )
    return AtomicFailureActuationReceipt(
        schema_version=_strict_int("failure receipt schema", row["schema_version"]),
        kind=row["kind"],  # type: ignore[arg-type]
        plan_sha256=row["plan_sha256"],  # type: ignore[arg-type]
        authority_sha256=row["authority_sha256"],  # type: ignore[arg-type]
        run_id=row["run_id"],  # type: ignore[arg-type]
        run_nonce_sha256=row["run_nonce_sha256"],  # type: ignore[arg-type]
        actuator_id=row["actuator_id"],  # type: ignore[arg-type]
        actuator_version_sha256=row["actuator_version_sha256"],  # type: ignore[arg-type]
        fresh_process=row["fresh_process"],  # type: ignore[arg-type]
        topologies=tuple(topologies),
        terminal_status=row["terminal_status"],  # type: ignore[arg-type]
        committed=row["committed"],  # type: ignore[arg-type]
        attester_id=row["attester_id"],  # type: ignore[arg-type]
        trust_domain=row["trust_domain"],  # type: ignore[arg-type]
        signature_hex=row["signature_hex"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class FailureReceiptReduction:
    status: Literal["READY", "BLOCKED"]
    reason: str | None
    authority_sha256: str
    plan_sha256: str
    raw_receipt_sha256: str
    payload_sha256: str
    topology_failure_surface_sha256: str | None

    def __post_init__(self) -> None:
        for label, value in (
            ("failure reduction authority", self.authority_sha256),
            ("failure reduction plan", self.plan_sha256),
            ("failure reduction raw receipt", self.raw_receipt_sha256),
            ("failure reduction payload", self.payload_sha256),
        ):
            _require_sha256(label, value)
        if self.topology_failure_surface_sha256 is not None:
            _require_sha256(
                "failure reduction surface", self.topology_failure_surface_sha256
            )
        if self.status == "READY":
            if self.reason is not None or self.topology_failure_surface_sha256 is None:
                raise ValueError("READY failure reduction is incomplete")
        elif self.status == "BLOCKED":
            if not self.reason or self.topology_failure_surface_sha256 is not None:
                raise ValueError("BLOCKED failure reduction cannot publish a surface")
        else:
            raise ValueError("failure reduction status is unsupported")


def _validate_receipt_against_plan(
    receipt: AtomicFailureActuationReceipt,
    result: FailureInjectionAuthorityResult,
) -> None:
    plan = result.plan
    if (
        receipt.plan_sha256 != plan.sha256
        or receipt.authority_sha256 != result.binding.sha256
    ):
        raise ValueError("failure receipt belongs to another plan or authority")
    expected_counter_by_name = {row.name: row for row in plan.expected_counters}
    for topology, target in zip(receipt.topologies, plan.topology_targets, strict=True):
        ranks = tuple(row.rank for row in topology.rank_receipts)
        if topology.topology != target.topology or ranks != target.target_ranks:
            raise ValueError("failure receipt has partial, duplicate, or foreign ranks")
        process_ids = tuple(row.process_id for row in topology.rank_receipts)
        if len(set(process_ids)) != len(process_ids):
            raise ValueError("failure receipt process identities are not rank-unique")
        aggregate = {name: 0 for name in expected_counter_by_name}
        for rank_receipt in topology.rank_receipts:
            phase_times = dict(rank_receipt.phases)
            windows_ns = (
                (
                    rank_receipt.process_start_monotonic_ns,
                    phase_times["arm"],
                    plan.lifecycle.arm_timeout_ms,
                ),
                (
                    phase_times["arm"],
                    phase_times["trigger"],
                    plan.lifecycle.trigger_timeout_ms,
                ),
                (
                    phase_times["trigger"],
                    phase_times["recover"],
                    plan.lifecycle.recover_timeout_ms,
                ),
                (
                    phase_times["recover"],
                    phase_times["terminal"],
                    plan.lifecycle.terminal_timeout_ms,
                ),
            )
            if any(
                right - left > timeout_ms * 1_000_000
                for left, right, timeout_ms in windows_ns
            ):
                raise ValueError("failure receipt exceeds its release lifecycle window")
            counters = dict(rank_receipt.counters)
            if set(counters) != set(expected_counter_by_name):
                raise ValueError("failure receipt counter coverage is incomplete")
            for name, expectation in expected_counter_by_name.items():
                value = counters[name]
                if expectation.comparison == "eq" and value != expectation.value:
                    raise ValueError(f"failure safety counter {name} differs")
                aggregate[name] += value
        for name, expectation in expected_counter_by_name.items():
            if expectation.comparison == "ge" and aggregate[name] < expectation.value:
                raise ValueError(f"failure trigger counter {name} is incomplete")


def reduce_failure_actuation_receipt(
    binding: FailureInjectionAuthorityBinding,
    receipt_path: str | Path,
    *,
    registry: ExperimentRegistry,
) -> FailureReceiptReduction:
    """Replay raw all-rank evidence; never accept a summary supplied by caller."""

    result = revalidate_failure_injection_authority(binding, registry=registry)
    path = _resolved_regular_path(receipt_path, label="failure receipt")
    row, raw_sha256 = _load_raw_json(path, label="failure receipt")
    receipt = _receipt_from_dict(row)
    _validate_receipt_against_plan(receipt, result)
    if (
        len(RELEASE_FAILURE_ACTUATORS) != 1
        or (receipt.actuator_id, receipt.actuator_version_sha256)
        != RELEASE_FAILURE_ACTUATORS[0]
    ):
        return FailureReceiptReduction(
            status="BLOCKED",
            reason=FAILURE_INJECTION_FIRST_PARTY_ACTUATOR_UNAVAILABLE_REASON,
            authority_sha256=binding.sha256,
            plan_sha256=result.plan.sha256,
            raw_receipt_sha256=raw_sha256,
            payload_sha256=receipt.payload_sha256,
            topology_failure_surface_sha256=None,
        )
    policy = require_release_trusted_attester_policy(RELEASE_TRUSTED_ATTESTER_POLICY)
    if not policy.release_ready:
        return FailureReceiptReduction(
            status="BLOCKED",
            reason=FAILURE_INJECTION_TRUSTED_SIGNER_UNAVAILABLE_REASON,
            authority_sha256=binding.sha256,
            plan_sha256=result.plan.sha256,
            raw_receipt_sha256=raw_sha256,
            payload_sha256=receipt.payload_sha256,
            topology_failure_surface_sha256=None,
        )
    policy.verify_terminal_signature(
        attester_id=receipt.attester_id,
        trust_domain=receipt.trust_domain,
        message=receipt.attestation_message,
        signature_hex=receipt.signature_hex,
    )
    surface_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "e5_topology_failure_surface",
            "registry_sha256": registry.sha256,
            "cell_id": result.plan.cell_id,
            "scenario": result.plan.scenario,
            "plan_sha256": result.plan.sha256,
            "receipt_payload_sha256": receipt.payload_sha256,
            "topologies": [row.to_dict() for row in receipt.topologies],
            "correctness_only": True,
        }
    )
    return FailureReceiptReduction(
        status="READY",
        reason=None,
        authority_sha256=binding.sha256,
        plan_sha256=result.plan.sha256,
        raw_receipt_sha256=raw_sha256,
        payload_sha256=receipt.payload_sha256,
        topology_failure_surface_sha256=surface_sha256,
    )


__all__ = [
    "FAILURE_INJECTION_EXECUTION_LIFECYCLE_UNAVAILABLE_REASON",
    "FAILURE_INJECTION_FIRST_PARTY_ACTUATOR_UNAVAILABLE_REASON",
    "FAILURE_INJECTION_RAW_PLAN_AUTHORITY_REQUIRED_REASON",
    "FAILURE_INJECTION_RAW_RECEIPT_INCOMPLETE_REASON",
    "FAILURE_INJECTION_REDUCER_PROTOCOL_SHA256",
    "FAILURE_INJECTION_TRUSTED_SIGNER_UNAVAILABLE_REASON",
    "AtomicFailureActuationReceipt",
    "FailureExecutionAuthorityToken",
    "FailureExpectedCounter",
    "FailureInjectionAuthorityBinding",
    "FailureInjectionAuthorityBlocked",
    "FailureInjectionAuthorityResult",
    "FailureLifecycleWindow",
    "FailureRankReceipt",
    "FailureReceiptReduction",
    "FailureTopologyReceipt",
    "FailureTopologyTarget",
    "ReleaseFailurePlan",
    "bind_failure_injection_authority",
    "reduce_failure_actuation_receipt",
    "release_failure_plan_for_cell",
    "require_failure_execution_lifecycle",
    "require_failure_injection_authority",
    "revalidate_failure_injection_authority",
]
