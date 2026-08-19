"""Production, non-LLM driver for the current formal single-operator DAG.

The driver is intentionally a thin composition layer.  Scientific values are
owned by the ProtocolLock, stage materializers, and source-owned physical-plan
producers.  This module accepts paths only, journals the concrete launches,
and advances the durable controller by at most one state transition per
``run_once`` call.

Missing launch prerequisites, E5 arrival plans, or GPU auxiliary executors are
explicit blockers.  In particular, this module never synthesizes a launch or
an actual-result path from a directory name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

from lightcone_spec.orchestration.experiment_operator import (
    AuxiliaryCellAdoption,
    AuxiliaryGroupTerminal,
    AuxiliaryJobSpec,
    AuxiliaryPhysicalGroupSpec,
    CellAttemptSpec,
    ControllerArtifactBinding,
    ExperimentOperatorError,
    ExperimentOperatorStore,
    FormalExperimentSchedulerDaemon,
    InterferenceEnvelope,
    MetricRecord,
    PhysicalAttemptGroupMemberSpec,
    ProcessObservation,
    QueuedCommandSpec,
    SchedulerCycleResult,
    SingletonOperatorLock,
    SpawnedProcess,
    WorkerHeartbeat,
    default_formal_stage_plan,
)
from lightcone_spec.orchestration.formal_experiment_controller import (
    DagCellLaunch,
    DagControllerCallbacks,
    DagControllerStep,
    DagExecutionPlan,
    DagMaterialization,
    DagPhysicalAttemptGroup,
    DagReduction,
    FormalExperimentDagBlocked,
    FormalExperimentDagController,
)

_CONFIG_KIND = "formal_single_operator_dag_driver_config"
_CONFIG_SCHEMA_VERSION = 1
_PLAN_JOURNAL_KIND = "formal_single_operator_dag_execution_plan_journal"
_PLAN_JOURNAL_SCHEMA_VERSION = 1
_PREREQUISITE_BINDING_KIND = "formal_single_operator_prerequisite_index_catalog_binding"
_PREREQUISITE_BINDING_SCHEMA_VERSION = 1
_AUXILIARY_INPUT_BINDING_KIND = "formal_single_operator_auxiliary_input_catalog_binding"
_AUXILIARY_INPUT_BINDING_SCHEMA_VERSION = 1
_AUXILIARY_WORKER_KIND = "formal_single_operator_auxiliary_worker_descriptor"
_AUXILIARY_WORKER_TERMINAL_KIND = "formal_single_operator_auxiliary_worker_terminal"
_RETAINED_DEPENDENCY_MANIFEST_KIND = (
    "formal_single_operator_retained_future_dependency_manifest"
)
_FRESH_INTERFERENCE_DIAGNOSTIC_KIND = (
    "formal_single_operator_fresh_preflight_interference_diagnostic"
)
_FRESH_INTERFERENCE_DIAGNOSTIC_SCHEMA_VERSION = 1
_FRESH_INTERFERENCE_DIAGNOSTIC_FILENAME = "fresh-preflight-interference-diagnostic.json"
_CELL_SPOOL_HIGH_WATER_BYTES = 16 * 1024**3
_AUXILIARY_PROCESS_ATTACH_GRACE_NS = 30 * 1_000_000_000
_AUXILIARY_TERMINATION_GRACE_NS = 120 * 1_000_000_000
_AUXILIARY_HEARTBEAT_STALE_NS = 120 * 1_000_000_000
_AUXILIARY_LOG_STALL_WARNING_NS = 10 * 60 * 1_000_000_000
_EARLY_NODES = frozenset({"e3a", "tts_cal", "e1", "e2_r0", "e2_r1", "e2_r2", "e2_r3"})
_E4_DIRECT_NODES = frozenset({"e4_screen", "e4_local"})
_PREPARED_NODES = frozenset(
    {
        "e4_profiler",
        "e3b_pilot",
        "e3b_final",
        "e1a",
        "e5_pilot",
        "e5_final",
        "e6_pilot",
        "e6_final",
        "e0_tuning",
        "e0_pilot",
        "e0_final",
    }
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")

# These are source-validated native terminal counters whose per-cell values are
# useful on their own.  Keeping this list intentionally small prevents the
# descriptive projection from turning a 10^5-cell campaign into millions of
# redundant SQLite rows; every other counter remains in the bound raw terminal.
_SERVING_SCALAR_COUNTER_METRICS = (
    "peak_hbm_bytes",
    "target_calls",
    "committed_tokens",
    "accepted_drafts",
    "verified_drafts",
    "updates_launched",
    "updates_published",
)


@dataclass(frozen=True)
class FormalDagNodeCodeCapability:
    """Import-derived code capability; it never inspects run artifacts."""

    node: str
    materializer: bool
    producer: bool
    mapper: bool
    executor: bool
    finalizer: bool
    blocker: str | None

    @property
    def ready(self) -> bool:
        return all(
            (
                self.materializer,
                self.producer,
                self.mapper,
                self.executor,
                self.finalizer,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "ready": self.ready}


_COMMON_CODE_REQUIREMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "materializer": (
        (
            "lightcone_spec.experiments.formal_single_operator_stages",
            "materialize_formal_single_operator_node",
        ),
    ),
    "finalizer": (
        (
            "lightcone_spec.experiments.formal_single_operator_stages",
            "reduce_formal_single_operator_node",
        ),
    ),
}

_PREFLIGHT_CODE_REQUIREMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "producer": (
        (
            "lightcone_spec.experiments.formal_preflight_inputs",
            "materialize_formal_single_operator_preflight_execution_inputs",
        ),
    ),
    "mapper": (
        (
            "lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker",
            "publish_formal_preflight_exact_ten_group_worker_spec",
        ),
    ),
    "executor": (
        (
            "lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker",
            "run_formal_preflight_exact_ten_group_worker",
        ),
    ),
}

_DIRECT_CODE_REQUIREMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "producer": (
        (
            "lightcone_spec.experiments.formal_single_operator_early_execution",
            "materialize_formal_single_operator_early_run_plan_inputs",
        ),
    ),
    "mapper": (
        (
            "lightcone_spec.orchestration.formal_physical_dispatch",
            "materialize_formal_single_operator_serving_run_plan",
        ),
    ),
    "executor": (
        (
            "lightcone_spec.orchestration.formal_physical_dispatch",
            "execute_formal_single_operator_serving_run_plan",
        ),
    ),
}

_E4_DIRECT_CODE_REQUIREMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "producer": (
        (
            "lightcone_spec.experiments.formal_single_operator_run_dispatch",
            "materialize_formal_single_operator_e4_direct_run_plan_inputs",
        ),
    ),
    "mapper": (
        (
            "lightcone_spec.orchestration.formal_physical_dispatch",
            "materialize_formal_single_operator_downstream_serving_run_plan",
        ),
    ),
    "executor": _DIRECT_CODE_REQUIREMENTS["executor"],
}

_PREPARED_CODE_REQUIREMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "producer": (
        (
            "lightcone_spec.experiments.formal_single_operator_prerequisite_launch_producer",
            "publish_formal_single_operator_prerequisite_launch_index",
        ),
        (
            "lightcone_spec.experiments.formal_single_operator_prepared_launch_producer",
            "prepare_launch_draft",
        ),
        (
            "lightcone_spec.experiments.formal_single_operator_prepared_launch_producer",
            "materialize_prepared_request_schedule",
        ),
        (
            "lightcone_spec.experiments.formal_single_operator_prepared_launch_producer",
            "finalize_prepared_launch_bundle",
        ),
    ),
    "mapper": (
        (
            "lightcone_spec.experiments.formal_single_operator_run_dispatch",
            "materialize_formal_single_operator_prepared_downstream_run_plan_inputs",
        ),
        (
            "lightcone_spec.orchestration.formal_physical_dispatch",
            "materialize_formal_single_operator_prepared_downstream_serving_run_plan",
        ),
    ),
    "executor": _DIRECT_CODE_REQUIREMENTS["executor"],
}


def _merge_code_requirements(
    *values: Mapping[str, tuple[tuple[str, str], ...]],
) -> dict[str, tuple[tuple[str, str], ...]]:
    result = {name: tuple(rows) for name, rows in _COMMON_CODE_REQUIREMENTS.items()}
    for value in values:
        for name, rows in value.items():
            result[name] = result.get(name, ()) + tuple(rows)
    return result


def _node_code_requirements(node: str) -> dict[str, tuple[tuple[str, str], ...]]:
    if node == "preflight":
        return _merge_code_requirements(_PREFLIGHT_CODE_REQUIREMENTS)
    if node in _EARLY_NODES:
        return _merge_code_requirements(_DIRECT_CODE_REQUIREMENTS)
    if node in _E4_DIRECT_NODES:
        return _merge_code_requirements(_E4_DIRECT_CODE_REQUIREMENTS)
    if node not in _PREPARED_NODES:
        return _merge_code_requirements()
    extra: dict[str, tuple[tuple[str, str], ...]] = {}
    if node == "e4_profiler":
        extra = {
            "producer": (
                (
                    "lightcone_spec.experiments.formal_single_operator_profiler_subject_producer",
                    "publish_formal_single_operator_profiler_subject_requirement",
                ),
                (
                    "lightcone_spec.experiments.formal_single_operator_profiler",
                    "materialize_formal_single_operator_profiler_plan",
                ),
            ),
            "executor": (
                (
                    "lightcone_spec.experiments.formal_single_operator_profiler",
                    "run_formal_single_operator_profiler",
                ),
            ),
        }
    elif node in {"e5_pilot", "e5_final"}:
        extra = {
            "producer": (
                (
                    "lightcone_spec.experiments.formal_single_operator_loads",
                    "derive_e5_arrival_plan",
                ),
                (
                    "lightcone_spec.experiments.formal_failure_execution",
                    "materialize_formal_single_operator_e5_failure_execution_descriptor",
                ),
            ),
            "mapper": (
                (
                    "lightcone_spec.orchestration.formal_physical_dispatch",
                    "materialize_formal_single_operator_e5_failure_run_plan",
                ),
            ),
            "executor": (
                (
                    "lightcone_spec.orchestration.formal_failure_physical",
                    "execute_formal_e5_failure_run_plan",
                ),
            ),
        }
    elif node == "e6_pilot":
        extra = {
            "producer": (
                (
                    "lightcone_spec.experiments.formal_single_operator_e6_interface",
                    "materialize_formal_single_operator_e6_interface_fit_campaign",
                ),
                (
                    "lightcone_spec.experiments.formal_single_operator_e6_interface",
                    "finalize_formal_single_operator_e6_interface_fit_bundle",
                ),
            ),
            "executor": (
                (
                    "lightcone_spec.experiments.formal_single_operator_e6_interface",
                    "execute_formal_single_operator_e6_interface_fit_plan",
                ),
            ),
        }
    elif node == "e0_tuning":
        extra = {
            "producer": (
                (
                    "lightcone_spec.orchestration.formal_e0_compatibility_physical",
                    "materialize_formal_e0_compatibility_physical_campaign",
                ),
                (
                    "lightcone_spec.orchestration.formal_e0_compatibility_physical",
                    "publish_completed_e0_compatibility_physical_campaign",
                ),
            ),
            "executor": (
                (
                    "lightcone_spec.orchestration.formal_e0_compatibility_physical",
                    "execute_formal_e0_compatibility_probe_group",
                ),
            ),
        }
    return _merge_code_requirements(_PREPARED_CODE_REQUIREMENTS, extra)


def formal_single_operator_dag_code_capabilities() -> tuple[
    FormalDagNodeCodeCapability, ...
]:
    """Return import/callable readiness for all 21 nodes without reading evidence.

    This intentionally says nothing about whether the run-specific ProtocolLock,
    prerequisite catalog, auxiliary terminals, tools, or model snapshots exist.
    Those remain strict runtime blockers in :class:`ProductionFormalDagCallbackBuilder`.
    """

    from lightcone_spec.experiments.formal_single_operator_stages import (
        FORMAL_SINGLE_OPERATOR_NODE_ORDER,
    )

    module_cache: dict[str, object | BaseException] = {}
    results = []
    for node in FORMAL_SINGLE_OPERATOR_NODE_ORDER:
        state: dict[str, bool] = {}
        blockers = []
        requirements = _node_code_requirements(node)
        for role in ("materializer", "producer", "mapper", "executor", "finalizer"):
            missing = []
            for module_name, attribute in requirements.get(role, ()):
                loaded = module_cache.get(module_name)
                if loaded is None:
                    try:
                        loaded = importlib.import_module(module_name)
                    except BaseException as error:  # noqa: BLE001 - capability probe
                        loaded = error
                    module_cache[module_name] = loaded
                if isinstance(loaded, BaseException) or not callable(
                    getattr(loaded, attribute, None)
                ):
                    missing.append(f"{module_name}.{attribute}")
            state[role] = not missing and bool(requirements.get(role))
            if missing:
                blockers.append(f"{role}:missing:" + ",".join(missing))
            elif not requirements.get(role):
                blockers.append(f"{role}:unregistered")
        results.append(
            FormalDagNodeCodeCapability(
                node=node,
                materializer=state["materializer"],
                producer=state["producer"],
                mapper=state["mapper"],
                executor=state["executor"],
                finalizer=state["finalizer"],
                blocker=";".join(blockers) if blockers else None,
            )
        )
    return tuple(results)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"artifact is not one regular file: {source}")
    before = source.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = source.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"artifact changed while hashing: {source}")
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} is not lower-case SHA-256")
    return value


def _absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    return path


def _existing_file(value: str | Path, label: str) -> Path:
    path = _absolute_path(value, label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular file")
    return path


def _existing_directory(value: str | Path, label: str) -> Path:
    path = _absolute_path(value, label)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be one directory")
    return path


def _read_canonical_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _existing_file(path, label)
    body = source.read_bytes()
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if type(value) is not dict or body != _canonical_bytes(value):
        raise ValueError(f"{label} is not one canonical JSON object")
    return value


def _publish_no_replace(path: str | Path, value: object) -> None:
    destination = _absolute_path(path, "publication path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _preserve_partial_directory(path: str | Path, *, label: str) -> Path:
    """Move uncommitted CPU work aside without deleting or overwriting it.

    Controller bindings are published only after their callback returns.  A
    process death before that point can therefore leave source-owned, no-
    replace files which have never authorized a GPU launch.  Preserving that
    directory and rebuilding in the canonical path is the only automatic
    recovery performed here; durable controller/attempt state is never moved.
    """

    source = _absolute_path(path, label)
    if source.is_symlink() or not source.is_dir():
        raise FormalExperimentDagBlocked(f"{label}: partial path is not a directory")
    abandoned = source.parent / f"{source.name}.abandoned"
    abandoned.mkdir(mode=0o700, exist_ok=True)
    ordinal = 1
    while True:
        destination = abandoned / f"attempt-{ordinal:04d}"
        if not os.path.lexists(destination):
            break
        ordinal += 1
    os.rename(source, destination)
    parent = os.open(source.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return destination


@dataclass(frozen=True)
class DriverFileBinding:
    """Raw binding for a path-only driver input."""

    absolute_path: str
    raw_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = _existing_file(self.absolute_path, "driver input")
        _require_sha256(self.raw_sha256, "driver input raw digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("driver input size is invalid")
        if path.stat(follow_symlinks=False).st_size != self.size_bytes:
            raise ValueError("driver input size changed")
        if _file_sha256(path) != self.raw_sha256:
            raise ValueError("driver input bytes changed")

    @classmethod
    def bind(cls, path: str | Path) -> DriverFileBinding:
        source = _existing_file(path, "driver input")
        return cls(
            absolute_path=str(source),
            raw_sha256=_file_sha256(source),
            size_bytes=source.stat(follow_symlinks=False).st_size,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> DriverFileBinding:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("driver file binding fields differ")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PathBoundFormalDagDriverConfig:
    """Immutable path-only inputs for one run-specific operator."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_dag_driver_config"]
    repository_root: str
    run_root: str
    protocol_lock: DriverFileBinding
    content_source: DriverFileBinding
    runtime_authority_manifest: DriverFileBinding
    inventory: DriverFileBinding
    doctor_report: DriverFileBinding
    preflight_workload_authority: DriverFileBinding
    profiler_tools: tuple[DriverFileBinding, ...]
    prerequisite_index_catalog_directory: str
    session_reset_authority_directory: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _CONFIG_KIND:
            raise ValueError("formal DAG driver config schema differs")
        repository = _existing_directory(self.repository_root, "repository root")
        run_root = _existing_directory(self.run_root, "run root")
        try:
            run_root.relative_to(repository)
        except ValueError:
            pass
        else:
            raise ValueError("formal run root must stay outside the source checkout")
        bindings = (
            self.protocol_lock,
            self.content_source,
            self.runtime_authority_manifest,
            self.inventory,
            self.doctor_report,
            self.preflight_workload_authority,
            *self.profiler_tools,
        )
        if any(type(value) is not DriverFileBinding for value in bindings):
            raise TypeError("formal DAG driver inputs must be exact file bindings")
        tool_paths = tuple(row.absolute_path for row in self.profiler_tools)
        if tool_paths != tuple(sorted(set(tool_paths))):
            raise ValueError("profiler tool bindings must be uniquely sorted")
        _existing_directory(
            self.prerequisite_index_catalog_directory,
            "prerequisite index catalog directory",
        )
        if self.session_reset_authority_directory is not None:
            _existing_directory(
                self.session_reset_authority_directory,
                "session-reset authority directory",
            )

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "repository_root": self.repository_root,
            "run_root": self.run_root,
            "protocol_lock": self.protocol_lock.to_dict(),
            "content_source": self.content_source.to_dict(),
            "runtime_authority_manifest": self.runtime_authority_manifest.to_dict(),
            "inventory": self.inventory.to_dict(),
            "doctor_report": self.doctor_report.to_dict(),
            "preflight_workload_authority": (
                self.preflight_workload_authority.to_dict()
            ),
            "profiler_tools": [row.to_dict() for row in self.profiler_tools],
            "prerequisite_index_catalog_directory": (
                self.prerequisite_index_catalog_directory
            ),
            "session_reset_authority_directory": (
                self.session_reset_authority_directory
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> PathBoundFormalDagDriverConfig:
        if type(value) is not dict or set(value) not in (
            set(cls.__dataclass_fields__),
            set(cls.__dataclass_fields__) - {"session_reset_authority_directory"},
        ):
            raise ValueError("formal DAG driver config fields differ")
        row = dict(value)
        row.setdefault("session_reset_authority_directory", None)
        for name in (
            "protocol_lock",
            "content_source",
            "runtime_authority_manifest",
            "inventory",
            "doctor_report",
            "preflight_workload_authority",
        ):
            row[name] = DriverFileBinding.from_dict(row[name])
        tools = row.pop("profiler_tools")
        if type(tools) is not list:
            raise TypeError("profiler tools must be an array")
        return cls(
            **row,
            profiler_tools=tuple(DriverFileBinding.from_dict(item) for item in tools),
        )  # type: ignore[arg-type]


def publish_path_bound_formal_dag_driver_config(
    *,
    repository_root: str | Path,
    run_root: str | Path,
    protocol_lock_path: str | Path,
    content_source_path: str | Path,
    runtime_authority_manifest_path: str | Path,
    inventory_path: str | Path,
    doctor_report_path: str | Path,
    preflight_workload_authority_path: str | Path,
    profiler_tool_paths: Sequence[str | Path],
    prerequisite_index_catalog_directory: str | Path,
    output_path: str | Path,
    session_reset_authority_directory: str | Path | None = None,
) -> PathBoundFormalDagDriverConfig:
    config = PathBoundFormalDagDriverConfig(
        schema_version=1,
        kind=_CONFIG_KIND,
        repository_root=str(_existing_directory(repository_root, "repository root")),
        run_root=str(_existing_directory(run_root, "run root")),
        protocol_lock=DriverFileBinding.bind(protocol_lock_path),
        content_source=DriverFileBinding.bind(content_source_path),
        runtime_authority_manifest=DriverFileBinding.bind(
            runtime_authority_manifest_path
        ),
        inventory=DriverFileBinding.bind(inventory_path),
        doctor_report=DriverFileBinding.bind(doctor_report_path),
        preflight_workload_authority=DriverFileBinding.bind(
            preflight_workload_authority_path
        ),
        profiler_tools=tuple(
            sorted(
                (DriverFileBinding.bind(path) for path in profiler_tool_paths),
                key=lambda row: row.absolute_path,
            )
        ),
        prerequisite_index_catalog_directory=str(
            _existing_directory(
                prerequisite_index_catalog_directory,
                "prerequisite index catalog directory",
            )
        ),
        session_reset_authority_directory=(
            None
            if session_reset_authority_directory is None
            else str(
                _existing_directory(
                    session_reset_authority_directory,
                    "session-reset authority directory",
                )
            )
        ),
    )
    _publish_no_replace(output_path, config.to_dict())
    if load_path_bound_formal_dag_driver_config(output_path) != config:
        raise RuntimeError("formal DAG driver config changed during publication")
    return config


def load_path_bound_formal_dag_driver_config(
    path: str | Path,
) -> PathBoundFormalDagDriverConfig:
    return PathBoundFormalDagDriverConfig.from_dict(
        _read_canonical_json(path, label="formal DAG driver config")
    )


@dataclass(frozen=True)
class PrerequisiteIndexCatalogBinding:
    """One append-only node/source-specific prerequisite-index binding."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_prerequisite_index_catalog_binding"]
    node: str
    execution_source_sha256: str
    prerequisite_index_path: str
    prerequisite_index_raw_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _PREREQUISITE_BINDING_KIND:
            raise ValueError("prerequisite catalog binding schema differs")
        if type(self.node) is not str or not self.node.strip():
            raise ValueError("prerequisite catalog node is empty")
        _require_sha256(
            self.execution_source_sha256,
            "prerequisite execution-source digest",
        )
        index = _existing_file(self.prerequisite_index_path, "prerequisite index")
        _require_sha256(
            self.prerequisite_index_raw_sha256,
            "prerequisite index raw digest",
        )
        if _file_sha256(index) != self.prerequisite_index_raw_sha256:
            raise ValueError("prerequisite index bytes changed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> PrerequisiteIndexCatalogBinding:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("prerequisite catalog binding fields differ")
        return cls(**value)  # type: ignore[arg-type]


def publish_prerequisite_index_catalog_binding(
    *,
    node: str,
    execution_source_path: str | Path,
    prerequisite_index_path: str | Path,
    output_path: str | Path,
) -> PrerequisiteIndexCatalogBinding:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        load_formal_single_operator_execution_source,
    )

    source = load_formal_single_operator_execution_source(execution_source_path)
    if source.node != node:
        raise ValueError("prerequisite binding node differs from execution source")
    index = _existing_file(prerequisite_index_path, "prerequisite index")
    value = PrerequisiteIndexCatalogBinding(
        schema_version=1,
        kind=_PREREQUISITE_BINDING_KIND,
        node=node,
        execution_source_sha256=source.sha256,
        prerequisite_index_path=str(index),
        prerequisite_index_raw_sha256=_file_sha256(index),
    )
    _publish_no_replace(output_path, value.to_dict())
    return value


@dataclass(frozen=True)
class AuxiliaryInputCatalogBinding:
    """Append-only, predecessor-specific physical-campaign inputs.

    The catalog contains paths and byte identities only.  It deliberately does
    not accept model names, compatibility dispositions, tuning values, or any
    other scientific scalar from the operator CLI.
    """

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_auxiliary_input_catalog_binding"]
    node: Literal["e6_pilot", "e0_tuning"]
    predecessor_completion: ControllerArtifactBinding
    input_files: tuple[DriverFileBinding, ...]
    onlinespec_source_authority: DriverFileBinding | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _AUXILIARY_INPUT_BINDING_KIND
            or self.node not in {"e6_pilot", "e0_tuning"}
            or type(self.predecessor_completion) is not ControllerArtifactBinding
        ):
            raise ValueError("auxiliary input catalog binding schema differs")
        if type(self.input_files) is not tuple or any(
            type(row) is not DriverFileBinding for row in self.input_files
        ):
            raise TypeError("auxiliary input catalog requires exact file bindings")
        paths = tuple(row.absolute_path for row in self.input_files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ValueError("auxiliary input files must be non-empty and sorted")
        if self.node == "e6_pilot":
            if len(paths) != 2 or self.onlinespec_source_authority is not None:
                raise ValueError("E6 auxiliary inputs must contain exact two launches")
        elif len(paths) != 12:
            raise ValueError("E0 auxiliary inputs must contain exact twelve interfaces")
        if (
            self.onlinespec_source_authority is not None
            and type(self.onlinespec_source_authority) is not DriverFileBinding
        ):
            raise TypeError("OnlineSPEC source authority must be path-bound")

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "node": self.node,
            "predecessor_completion": asdict(self.predecessor_completion),
            "input_files": [row.to_dict() for row in self.input_files],
            "onlinespec_source_authority": (
                None
                if self.onlinespec_source_authority is None
                else self.onlinespec_source_authority.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> AuxiliaryInputCatalogBinding:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("auxiliary input catalog binding fields differ")
        row = dict(value)
        predecessor = row.pop("predecessor_completion")
        if type(predecessor) is not dict:
            raise TypeError("auxiliary predecessor binding must be an object")
        raw_inputs = row.pop("input_files")
        if type(raw_inputs) is not list:
            raise TypeError("auxiliary input bindings must be an array")
        raw_authority = row.pop("onlinespec_source_authority")
        return cls(
            **row,
            predecessor_completion=ControllerArtifactBinding(**predecessor),
            input_files=tuple(DriverFileBinding.from_dict(item) for item in raw_inputs),
            onlinespec_source_authority=(
                None
                if raw_authority is None
                else DriverFileBinding.from_dict(raw_authority)
            ),
        )  # type: ignore[arg-type]


def publish_auxiliary_input_catalog_binding(
    *,
    node: Literal["e6_pilot", "e0_tuning"],
    predecessor_completion_path: str | Path,
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    onlinespec_source_authority_path: str | Path | None = None,
) -> AuxiliaryInputCatalogBinding:
    """Publish one exact physical-input shard without scientific CLI values."""

    from lightcone_spec.experiments.formal_single_operator_stages import (
        rebuild_formal_single_operator_stage_completion,
    )

    predecessor = ControllerArtifactBinding.bind(predecessor_completion_path)
    rebuilt = rebuild_formal_single_operator_stage_completion(predecessor.absolute_path)
    expected_predecessor = {"e6_pilot": "e5_final", "e0_tuning": "e6_final"}[node]
    if rebuilt.artifact.node != expected_predecessor:
        raise ValueError("auxiliary input binding predecessor node differs")
    value = AuxiliaryInputCatalogBinding(
        schema_version=1,
        kind=_AUXILIARY_INPUT_BINDING_KIND,
        node=node,
        predecessor_completion=predecessor,
        input_files=tuple(
            sorted(
                (DriverFileBinding.bind(path) for path in input_paths),
                key=lambda row: row.absolute_path,
            )
        ),
        onlinespec_source_authority=(
            None
            if onlinespec_source_authority_path is None
            else DriverFileBinding.bind(onlinespec_source_authority_path)
        ),
    )
    _publish_no_replace(output_path, value.to_dict())
    rebound = AuxiliaryInputCatalogBinding.from_dict(
        _read_canonical_json(output_path, label="auxiliary input catalog binding")
    )
    if rebound != value:
        raise RuntimeError("auxiliary input binding changed during publication")
    return value


@dataclass(frozen=True)
class AuxiliaryWorkerDescriptor:
    """Path-only command contract for one physical auxiliary process."""

    schema_version: Literal[2]
    kind: Literal["formal_single_operator_auxiliary_worker_descriptor"]
    node: Literal["e6_pilot", "e0_tuning"]
    source_kind: Literal["e6_interface_fit", "e0_compatibility"]
    group_id: str
    attempt: int
    campaign: DriverFileBinding
    protocol_lock: DriverFileBinding
    predecessor_completion: ControllerArtifactBinding
    content_source: DriverFileBinding
    onlinespec_source_authority: DriverFileBinding | None
    publication_output_path: str
    evidence_manifest_output_path: str | None
    terminal_output_path: str
    heartbeat_output_path: str
    process_hard_timeout_ns: int

    def __post_init__(self) -> None:
        expected = {
            "e6_pilot": "e6_interface_fit",
            "e0_tuning": "e0_compatibility",
        }
        if (
            self.schema_version != 2
            or self.kind != _AUXILIARY_WORKER_KIND
            or self.node not in expected
            or self.source_kind != expected[self.node]
            or type(self.attempt) is not int
            or self.attempt < 1
            or type(self.process_hard_timeout_ns) is not int
            or self.process_hard_timeout_ns < 1
        ):
            raise ValueError("auxiliary worker descriptor identity differs")
        _require_sha256(self.group_id, "auxiliary worker group ID")
        for binding in (self.campaign, self.protocol_lock, self.content_source):
            if type(binding) is not DriverFileBinding:
                raise TypeError("auxiliary worker input is not path-bound")
        if type(self.predecessor_completion) is not ControllerArtifactBinding:
            raise TypeError("auxiliary worker predecessor is not path-bound")
        if (
            self.onlinespec_source_authority is not None
            and type(self.onlinespec_source_authority) is not DriverFileBinding
        ):
            raise TypeError("auxiliary worker OnlineSPEC authority differs")
        publication = _absolute_path(
            self.publication_output_path, "auxiliary publication output"
        )
        terminal = _absolute_path(
            self.terminal_output_path, "auxiliary terminal output"
        )
        heartbeat = _absolute_path(
            self.heartbeat_output_path, "auxiliary heartbeat output"
        )
        if publication.exists() and (
            publication.is_symlink() or not publication.is_file()
        ):
            raise ValueError("auxiliary publication output is not a regular file")
        if terminal.exists() and (terminal.is_symlink() or not terminal.is_file()):
            raise ValueError("auxiliary terminal output is not a regular file")
        if heartbeat == terminal or heartbeat == publication:
            raise ValueError("auxiliary heartbeat output collides with publication")
        if heartbeat.exists() and (heartbeat.is_symlink() or not heartbeat.is_file()):
            raise ValueError("auxiliary heartbeat output is not a regular file")
        if self.node == "e6_pilot":
            if (
                self.evidence_manifest_output_path is not None
                or self.onlinespec_source_authority is not None
            ):
                raise ValueError("E6 auxiliary worker carries E0-only outputs")
        else:
            if self.evidence_manifest_output_path is None:
                raise ValueError("E0 auxiliary worker lacks evidence-manifest output")
            _absolute_path(
                self.evidence_manifest_output_path,
                "E0 auxiliary evidence-manifest output",
            )

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "node": self.node,
            "source_kind": self.source_kind,
            "group_id": self.group_id,
            "attempt": self.attempt,
            "campaign": self.campaign.to_dict(),
            "protocol_lock": self.protocol_lock.to_dict(),
            "predecessor_completion": asdict(self.predecessor_completion),
            "content_source": self.content_source.to_dict(),
            "onlinespec_source_authority": (
                None
                if self.onlinespec_source_authority is None
                else self.onlinespec_source_authority.to_dict()
            ),
            "publication_output_path": self.publication_output_path,
            "evidence_manifest_output_path": self.evidence_manifest_output_path,
            "terminal_output_path": self.terminal_output_path,
            "heartbeat_output_path": self.heartbeat_output_path,
            "process_hard_timeout_ns": self.process_hard_timeout_ns,
        }

    @classmethod
    def from_dict(cls, value: object) -> AuxiliaryWorkerDescriptor:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("auxiliary worker descriptor fields differ")
        row = dict(value)
        for name in ("campaign", "protocol_lock", "content_source"):
            row[name] = DriverFileBinding.from_dict(row[name])
        predecessor = row["predecessor_completion"]
        if type(predecessor) is not dict:
            raise TypeError("auxiliary worker predecessor must be an object")
        row["predecessor_completion"] = ControllerArtifactBinding(**predecessor)
        authority = row["onlinespec_source_authority"]
        row["onlinespec_source_authority"] = (
            None if authority is None else DriverFileBinding.from_dict(authority)
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RetainedFutureDependencyManifest:
    """Code-owned archive boundary after a node's exact reduction.

    This is not an archive receipt and never authorizes deletion by itself.
    It tells a rolling archiver which small controller/manifests must remain
    addressable for downstream deep replay and which node payload is eligible
    for verified transfer followed by rehydration at its original path.
    """

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_retained_future_dependency_manifest"]
    run_id: str
    run_root: str
    node: str
    completion: ControllerArtifactBinding
    decision: ControllerArtifactBinding
    retained_files: tuple[DriverFileBinding, ...]
    retained_transitive_roots: tuple[str, ...]
    archive_candidate_roots: tuple[str, ...]
    archive_safe_after_reduction: Literal[True]
    remote_eviction_authorized_for_nonretained_files: Literal[True]
    remote_eviction_scope: Literal[
        "archive_candidate_roots_excluding_retained_files_and_transitive_roots"
    ]
    eviction_preconditions: tuple[
        Literal["local_sha_manifest_verified"],
        Literal["local_rehydrate_test_passed"],
    ]
    transitive_evidence_must_rehydrate_at_original_paths: Literal[True]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _RETAINED_DEPENDENCY_MANIFEST_KIND
            or type(self.run_id) is not str
            or not self.run_id
            or Path(self.run_root).name != self.run_id
            or type(self.node) is not str
            or not self.node
            or type(self.completion) is not ControllerArtifactBinding
            or type(self.decision) is not ControllerArtifactBinding
            or self.archive_safe_after_reduction is not True
            or self.remote_eviction_authorized_for_nonretained_files is not True
            or self.remote_eviction_scope
            != "archive_candidate_roots_excluding_retained_files_and_transitive_roots"
            or self.eviction_preconditions
            != (
                "local_sha_manifest_verified",
                "local_rehydrate_test_passed",
            )
            or self.transitive_evidence_must_rehydrate_at_original_paths is not True
        ):
            raise ValueError("retained dependency manifest identity differs")
        _absolute_path(self.run_root, "retained dependency run root")
        paths = tuple(row.absolute_path for row in self.retained_files)
        if (
            type(self.retained_files) is not tuple
            or any(type(row) is not DriverFileBinding for row in self.retained_files)
            or not paths
            or paths != tuple(sorted(set(paths)))
        ):
            raise ValueError("retained dependency files must be exact and sorted")
        roots = tuple(
            str(_absolute_path(path, "archive candidate root"))
            for path in self.archive_candidate_roots
        )
        if roots != tuple(sorted(set(roots))) or not roots:
            raise ValueError("archive candidate roots must be exact and sorted")
        retained_roots = tuple(
            str(_absolute_path(path, "retained transitive root"))
            for path in self.retained_transitive_roots
        )
        if retained_roots != tuple(sorted(set(retained_roots))):
            raise ValueError("retained transitive roots must be exact and sorted")
        if any(
            candidate == retained or candidate.is_relative_to(retained)
            for candidate in map(Path, roots)
            for retained in map(Path, retained_roots)
        ):
            raise ValueError(
                "archive candidate roots cannot equal or be nested within retained roots"
            )

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "run_root": self.run_root,
            "node": self.node,
            "completion": asdict(self.completion),
            "decision": asdict(self.decision),
            "retained_files": [row.to_dict() for row in self.retained_files],
            "retained_transitive_roots": list(self.retained_transitive_roots),
            "archive_candidate_roots": list(self.archive_candidate_roots),
            "archive_safe_after_reduction": self.archive_safe_after_reduction,
            "remote_eviction_authorized_for_nonretained_files": (
                self.remote_eviction_authorized_for_nonretained_files
            ),
            "remote_eviction_scope": self.remote_eviction_scope,
            "eviction_preconditions": list(self.eviction_preconditions),
            "transitive_evidence_must_rehydrate_at_original_paths": (
                self.transitive_evidence_must_rehydrate_at_original_paths
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> RetainedFutureDependencyManifest:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("retained dependency manifest fields differ")
        row = dict(value)
        for name in ("completion", "decision"):
            binding = row[name]
            if type(binding) is not dict:
                raise TypeError("retained controller binding must be an object")
            row[name] = ControllerArtifactBinding(**binding)
        raw_files = row.pop("retained_files")
        raw_roots = row.pop("archive_candidate_roots")
        raw_retained_roots = row.pop("retained_transitive_roots")
        raw_preconditions = row.pop("eviction_preconditions")
        if any(
            type(value) is not list
            for value in (
                raw_files,
                raw_roots,
                raw_retained_roots,
                raw_preconditions,
            )
        ):
            raise TypeError("retained dependency arrays differ")
        return cls(
            **row,
            retained_files=tuple(
                DriverFileBinding.from_dict(item) for item in raw_files
            ),
            retained_transitive_roots=tuple(raw_retained_roots),
            archive_candidate_roots=tuple(raw_roots),
            eviction_preconditions=tuple(raw_preconditions),
        )  # type: ignore[arg-type]


def load_retained_future_dependency_manifest(
    path: str | Path,
) -> RetainedFutureDependencyManifest:
    """Deep-reopen one canonical rolling-archive boundary."""

    manifest = RetainedFutureDependencyManifest.from_dict(
        _read_canonical_json(path, label="retained dependency manifest")
    )
    if not Path(path).resolve(strict=False).is_relative_to(Path(manifest.run_root)):
        raise ValueError("retained dependency manifest lies outside its run root")
    return manifest


def _explicit_headline_metric_payload_rows(
    node: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Extract only registered powered headline intervals for final nodes.

    ``None`` delegates legacy/early reducers to the conservative structural
    extractor.  An empty tuple is meaningful for E0 ``ALL_NA`` and for final
    payloads that have no claimable p99 row; neither case receives a fake CI.
    """

    expected_family_keys = {
        "e3b_final": ("primary_contrasts", "mechanism_contrasts"),
        "e5_final": ("primary_contrasts", "mechanism_contrasts"),
        "e6_final": ("primary_contrasts", "mechanism_contrasts"),
        "e0_final": ("contrasts",),
    }
    if node not in expected_family_keys:
        return None
    families = payload.get("family_results")
    if type(families) is not list:
        raise ValueError(f"{node} headline family results are unavailable")
    rows: list[dict[str, Any]] = []
    fdr_by_hypothesis = {
        decision["name"]: {
            **decision,
            "family_id": family["family_id"],
            "procedure": family["procedure"],
            "false_discovery_rate": family["false_discovery_rate"],
        }
        for family in payload.get("breadth_fdr_families", [])
        if type(family) is dict and type(family.get("decisions")) is list
        for decision in family["decisions"]
        if type(decision) is dict and type(decision.get("name")) is str
    }
    for family in families:
        if type(family) is not dict:
            raise TypeError(f"{node} headline family row is not an object")
        for name in (
            "block_count",
            "request_count",
            "paired",
            "reducer",
            "family_sha256",
            "result_sha256",
        ):
            if name not in family:
                raise ValueError(f"{node} headline family lacks {name}")
        if (
            type(family["block_count"]) is not int
            or type(family["request_count"]) is not int
            or family["paired"] is not True
            or type(family["reducer"]) is not str
        ):
            raise ValueError(f"{node} headline family inference fields differ")
        context = {
            name: family[name]
            for name in (
                "dimensions",
                "compatibility_decision_id",
                "load",
                "family_sha256",
                "result_sha256",
            )
            if name in family
        }
        holm_rows = family.get(
            "holm_primary" if node == "e0_final" else "holm_decisions",
            [],
        )
        holm = {
            row["name"]: row
            for row in holm_rows
            if type(row) is dict and type(row.get("name")) is str
        }

        def add_contrast(
            value: object,
            *,
            family_class: str,
            current_context: Mapping[str, Any] = context,
            current_family: Mapping[str, Any] = family,
            current_holm: Mapping[str, Any] = holm,
        ) -> None:
            if type(value) is not dict:
                raise TypeError(f"{node} headline contrast is not an object")
            status = value.get("status")
            if status in {
                "EXCLUDED_UNSAFE_OR_INACTIVE",
                "UNRESOLVED_ZERO_GOODPUT",
                "UNRESOLVED_ZERO_VARIANCE",
            }:
                reasons = value.get("reason_codes")
                if (
                    type(value.get("name")) is not str
                    or not value["name"]
                    or type(reasons) is not list
                    or not reasons
                    or reasons != sorted(set(reasons))
                    or any(type(reason) is not str or not reason for reason in reasons)
                    or value.get("independent_unit") != "paired_block"
                    or any(
                        key in value
                        for key in (
                            "mean_log_ratio",
                            "mean_relative_gain",
                            "ci_lower_relative_gain",
                            "ci_upper_relative_gain",
                            "raw_p_value",
                            "confidence",
                        )
                    )
                ):
                    raise ValueError(f"{node} unresolved contrast fields differ")
                if status.startswith("UNRESOLVED_ZERO_") and reasons != [status]:
                    raise ValueError(f"{node} unresolved contrast reason/status differ")
                if status == "EXCLUDED_UNSAFE_OR_INACTIVE":
                    if any(
                        type(value.get(key)) is not list or not value[key]
                        for key in ("excluded_roles", "evidence_cell_ids")
                    ) or any(":" not in reason for reason in reasons):
                        raise ValueError(f"{node} excluded contrast evidence differs")
                elif (
                    type(value.get("block_ids")) is not list
                    or len(value["block_ids"]) != current_family["block_count"]
                ):
                    raise ValueError(f"{node} unresolved contrast blocks differ")
                return
            if status not in {None, "RESOLVED"}:
                raise ValueError(f"{node} headline contrast status differs")
            required = {
                "name",
                "block_ids",
                "mean_log_ratio",
                "mean_relative_gain",
                "ci_lower_relative_gain",
                "ci_upper_relative_gain",
                "raw_p_value",
                "confidence",
                "independent_unit",
            }
            if not required.issubset(value) or type(value["block_ids"]) is not list:
                raise ValueError(f"{node} headline contrast fields differ")
            name = str(value["name"])
            multiplicity: dict[str, Any] = {}
            if name in current_holm:
                multiplicity["holm"] = current_holm[name]
            decision_id = current_family.get("compatibility_decision_id")
            if type(decision_id) is str:
                hypothesis = f"{decision_id}:{name}"
                if hypothesis in fdr_by_hypothesis:
                    multiplicity["bh_fdr"] = fdr_by_hypothesis[hypothesis]
            rows.append(
                {
                    "context": current_context,
                    "metric_name": f"goodput_relative_gain/{name}",
                    "point_estimate": value["mean_relative_gain"],
                    "ci_low": value["ci_lower_relative_gain"],
                    "ci_high": value["ci_upper_relative_gain"],
                    "block_count": current_family["block_count"],
                    "request_count": current_family["request_count"],
                    "paired": True,
                    "reducer_method": current_family["reducer"],
                    "attributes": {
                        "family_class": family_class,
                        "mean_log_ratio": value["mean_log_ratio"],
                        "raw_p_value": value["raw_p_value"],
                        "confidence": value["confidence"],
                        "independent_unit": value["independent_unit"],
                        "block_ids": value["block_ids"],
                        "multiplicity": multiplicity,
                    },
                }
            )

        if node == "e0_final":
            contrasts = family.get("contrasts")
            if type(contrasts) is not dict:
                raise ValueError("E0 headline contrasts are unavailable")
            for name in sorted(contrasts):
                value = contrasts[name]
                if type(value) is not dict or value.get("name") != name:
                    raise ValueError("E0 contrast key and payload differ")
                add_contrast(value, family_class="breadth")
        else:
            for key, family_class in (
                ("primary_contrasts", "primary"),
                ("mechanism_contrasts", "mechanism"),
            ):
                values = family.get(key)
                if type(values) is not list:
                    raise ValueError(f"{node} {key} are unavailable")
                for value in values:
                    add_contrast(value, family_class=family_class)
            target = family.get("target_only_gate")
            if type(target) is not dict or "contrast" not in target:
                raise ValueError(f"{node} target-only gate is unavailable")
            add_contrast(target["contrast"], family_class="deployment_gate")
        if node == "e3b_final":
            hierarchical = family.get("hierarchical_intervals")
            if type(hierarchical) is not list:
                raise ValueError("E3b hierarchical intervals are unavailable")
            for interval in hierarchical:
                if type(interval) is not dict:
                    raise TypeError("E3b hierarchical interval is not an object")
                status = interval.get("status")
                if status in {
                    "EXCLUDED_UNSAFE_OR_INACTIVE",
                    "UNRESOLVED_ZERO_GOODPUT",
                    "UNRESOLVED_ZERO_VARIANCE",
                }:
                    reasons = interval.get("reason_codes")
                    if (
                        type(interval.get("name")) is not str
                        or not interval["name"]
                        or type(reasons) is not list
                        or not reasons
                        or reasons != sorted(set(reasons))
                        or any(
                            type(reason) is not str or not reason for reason in reasons
                        )
                        or interval.get("independent_units") != ["block", "request"]
                        or any(
                            key in interval
                            for key in (
                                "mean_log_ratio",
                                "mean_relative_gain",
                                "ci_lower_relative_gain",
                                "ci_upper_relative_gain",
                                "confidence",
                                "repetitions",
                            )
                        )
                    ):
                        raise ValueError(
                            "E3b unresolved hierarchical interval fields differ"
                        )
                    if status.startswith("UNRESOLVED_ZERO_") and reasons != [status]:
                        raise ValueError(
                            "E3b hierarchical interval reason/status differ"
                        )
                    if status == "EXCLUDED_UNSAFE_OR_INACTIVE" and (
                        type(interval.get("evidence_cell_ids")) is not list
                        or not interval["evidence_cell_ids"]
                        or any(":" not in reason for reason in reasons)
                    ):
                        raise ValueError(
                            "E3b excluded hierarchical interval evidence differs"
                        )
                    continue
                if status not in {None, "RESOLVED"}:
                    raise ValueError("E3b hierarchical interval status differs")
                required = {
                    "name",
                    "mean_log_ratio",
                    "mean_relative_gain",
                    "ci_lower_relative_gain",
                    "ci_upper_relative_gain",
                    "confidence",
                    "repetitions",
                    "independent_units",
                }
                if not required.issubset(interval):
                    raise ValueError("E3b hierarchical interval fields differ")
                rows.append(
                    {
                        "context": context,
                        "metric_name": (
                            f"hierarchical_goodput_relative_gain/{interval['name']}"
                        ),
                        "point_estimate": interval["mean_relative_gain"],
                        "ci_low": interval["ci_lower_relative_gain"],
                        "ci_high": interval["ci_upper_relative_gain"],
                        "block_count": family["block_count"],
                        "request_count": family["request_count"],
                        "paired": True,
                        "reducer_method": ("hierarchical_block_request_bootstrap"),
                        "attributes": {
                            "family_class": "hierarchical",
                            "mean_log_ratio": interval["mean_log_ratio"],
                            "confidence": interval["confidence"],
                            "bootstrap_repetitions": interval["repetitions"],
                            "independent_units": interval["independent_units"],
                        },
                    }
                )
    if node == "e5_final":
        anchors = payload.get("p99_anchor_claims")
        if type(anchors) is not list:
            raise ValueError("E5 p99 anchor claims are unavailable")
        for anchor in anchors:
            required = {
                "anchor_id",
                "point_estimate",
                "ci_low",
                "ci_high",
                "confidence",
                "independent_block_count",
                "request_count",
                "paired",
                "reducer_method",
                "metric_name",
            }
            if type(anchor) is not dict:
                raise TypeError("E5 p99 anchor row is not an object")
            status = anchor.get("status")
            if status in {"UNRESOLVED", "EXCLUDED_UNSAFE_OR_INACTIVE"}:
                reasons = anchor.get("reason_codes")
                if (
                    type(anchor.get("anchor_id")) is not str
                    or not anchor["anchor_id"]
                    or type(reasons) is not list
                    or not reasons
                    or reasons != sorted(set(reasons))
                    or any(type(reason) is not str or not reason for reason in reasons)
                    or type(anchor.get("block_evidence")) is not list
                    or not anchor["block_evidence"]
                    or type(anchor.get("independent_block_count")) is not int
                    or anchor["independent_block_count"]
                    != len(anchor["block_evidence"])
                    or type(anchor.get("request_count")) is not int
                    or anchor["request_count"] < 1
                    or any(
                        key in anchor
                        for key in (
                            "point_estimate",
                            "ci_low",
                            "ci_high",
                            "confidence",
                            "reducer_method",
                            "metric_name",
                        )
                    )
                ):
                    raise ValueError("E5 unresolved p99 anchor fields differ")
                if status == "EXCLUDED_UNSAFE_OR_INACTIVE" and (
                    anchor.get("excluded_roles") != ["LightCone"]
                    or type(anchor.get("evidence_cell_ids")) is not list
                    or not anchor["evidence_cell_ids"]
                    or anchor["evidence_cell_ids"]
                    != sorted(set(anchor["evidence_cell_ids"]))
                    or any(
                        type(cell_id) is not str or not cell_id
                        for cell_id in anchor["evidence_cell_ids"]
                    )
                    or any(not reason.startswith("LightCone:") for reason in reasons)
                ):
                    raise ValueError("E5 excluded p99 anchor evidence differs")
                continue
            if status != "CLAIMABLE" or not required.issubset(anchor):
                raise ValueError("E5 p99 anchor interval fields differ")
            rows.append(
                {
                    "context": {"anchor_id": anchor["anchor_id"]},
                    "metric_name": anchor["metric_name"],
                    "point_estimate": anchor["point_estimate"],
                    "ci_low": anchor["ci_low"],
                    "ci_high": anchor["ci_high"],
                    "block_count": anchor["independent_block_count"],
                    "request_count": anchor["request_count"],
                    "paired": anchor["paired"],
                    "reducer_method": anchor["reducer_method"],
                    "attributes": {
                        "family_class": "p99_anchor",
                        "anchor_id": anchor["anchor_id"],
                        "confidence": anchor["confidence"],
                        "completed_request_count": anchor.get(
                            "completed_request_count"
                        ),
                        "offered_request_count": anchor.get("offered_request_count"),
                    },
                }
            )
    return tuple(rows)


class PrerequisiteIndexResolver(Protocol):
    def launch_manifest_paths(
        self,
        *,
        node: str,
        execution_source_path: str,
    ) -> tuple[str, ...]: ...

    def chronobelief_gpu_parity_proof_paths(
        self,
        *,
        node: str,
        execution_source_path: str,
    ) -> tuple[str, ...]: ...


class E5ArrivalPlanResolver(Protocol):
    def arrival_plan_path(
        self,
        *,
        node: str,
        execution_source_path: str,
        materialized_cell_id: str,
    ) -> str | None: ...


class AuxiliaryPhysicalRuntime(Protocol):
    """Injected journal bridge for the two real pre-materialization campaigns."""

    def plan(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
    ) -> AuxiliaryPhysicalGroupSpec | None: ...

    def launch(self, spec: AuxiliaryPhysicalGroupSpec) -> SpawnedProcess: ...

    def terminal(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        durable_group: Mapping[str, Any],
    ) -> AuxiliaryGroupTerminal | None: ...

    def adoptions(
        self,
        node: str,
        node_materialization: ControllerArtifactBinding,
        spec: AuxiliaryPhysicalGroupSpec,
    ) -> tuple[AuxiliaryCellAdoption, ...]: ...

    def actual_result_paths(
        self,
        node: str,
        attempts: tuple[dict[str, Any], ...],
    ) -> Mapping[str, str]: ...


class InterferenceGateResolver(Protocol):
    def resolve(
        self,
        *,
        completion: ControllerArtifactBinding,
        actual_result_paths: Mapping[str, str],
        gpu_uuids: tuple[str, ...],
    ) -> InterferenceEnvelope: ...


class DirectoryPrerequisiteIndexResolver:
    """Resolve exactly one immutable catalog shard for the current source."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = _existing_directory(directory, "prerequisite index catalog")

    def _index(
        self,
        *,
        node: str,
        execution_source_path: str,
    ) -> Any:
        from lightcone_spec.experiments.formal_single_operator_prerequisite_launch_producer import (
            load_formal_single_operator_prerequisite_launch_index,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            load_formal_single_operator_execution_source,
        )

        source = load_formal_single_operator_execution_source(execution_source_path)
        if source.node != node:
            raise ValueError("prerequisite resolver source node differs")
        matches: list[PrerequisiteIndexCatalogBinding] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                row = PrerequisiteIndexCatalogBinding.from_dict(
                    _read_canonical_json(path, label="prerequisite catalog binding")
                )
            except ValueError:
                continue
            if row.node == node and row.execution_source_sha256 == source.sha256:
                matches.append(row)
        if len(matches) != 1:
            raise FormalExperimentDagBlocked(
                f"{node}: exact prerequisite index binding is unavailable"
            )
        binding = matches[0]
        index = load_formal_single_operator_prerequisite_launch_index(
            binding.prerequisite_index_path
        )
        if (
            index.execution_source_sha256 != source.sha256
            or index.execution_source.absolute_path
            != str(_absolute_path(execution_source_path, "execution source"))
        ):
            raise FormalExperimentDagBlocked(
                f"{node}: prerequisite index belongs to another execution source"
            )
        return index

    def launch_manifest_paths(
        self,
        *,
        node: str,
        execution_source_path: str,
    ) -> tuple[str, ...]:
        return self._index(
            node=node,
            execution_source_path=execution_source_path,
        ).launch_manifest_paths

    def chronobelief_gpu_parity_proof_paths(
        self,
        *,
        node: str,
        execution_source_path: str,
    ) -> tuple[str, ...]:
        return self._index(
            node=node,
            execution_source_path=execution_source_path,
        ).chronobelief_gpu_parity_proof_paths


class PathBoundE5ArrivalPlanResolver:
    """Derive E5 arrivals only from the current source and sealed E3a result.

    The resolver deliberately accepts no load scalar.  It walks the immutable
    predecessor chain carried by the current execution source, recovers the
    E3a ``lambda_star_request_rate`` selection, and combines it with the exact
    materialized E5 axes and the BurstGPT paths already bound by the trusted
    content bundle.
    """

    _REGISTERED_E3A_RULE = (
        "completed_requests_per_observed_window_at_static_40928_"
        "short_input_long_generation_matched_width_common_load"
    )
    _E5_RULE = (
        "E3a_Static_context_40928_short_input_long_generation_"
        "matched_width_common_load_completed_requests_per_second"
    )

    def __init__(self, config: PathBoundFormalDagDriverConfig) -> None:
        if type(config) is not PathBoundFormalDagDriverConfig:
            raise TypeError("E5 resolver requires the exact path-bound config")
        self.config = config

    @staticmethod
    def _cell(source: Any, cell_id: str) -> Any:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            rebuild_formal_single_operator_node_materialization,
        )

        rebuilt = rebuild_formal_single_operator_node_materialization(
            source.materialization_source.absolute_path
        )
        matches = tuple(
            row for row in rebuilt.materialization.cells if row.cell_id == cell_id
        )
        if len(matches) != 1:
            raise FormalExperimentDagBlocked(
                f"{source.node}: E5 materialized cell {cell_id} is unavailable"
            )
        return matches[0]

    @classmethod
    def _lambda_star(cls, source: Any) -> Any:
        from lightcone_spec.experiments.formal_single_operator_loads import (
            E3aLambdaStar,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            rebuild_formal_single_operator_stage_completion,
        )

        predecessor_path = (
            None
            if source.predecessor_completion_source is None
            else source.predecessor_completion_source.absolute_path
        )
        if predecessor_path is None:
            raise FormalExperimentDagBlocked(
                f"{source.node}: E3a lambda-star predecessor is unavailable"
            )
        rebuilt = rebuild_formal_single_operator_stage_completion(predecessor_path)
        while rebuilt.artifact.node != "e3a":
            if rebuilt.predecessor is None:
                raise FormalExperimentDagBlocked(
                    f"{source.node}: predecessor chain has no completed E3a"
                )
            rebuilt = rebuilt.predecessor
        payload = rebuilt.decision.payload
        raw = payload.get("lambda_star_request_rate")
        if type(raw) is not dict:
            raise FormalExperimentDagBlocked(
                f"{source.node}: completed E3a lacks lambda-star authority"
            )
        locked = dict(raw)
        if locked.get("rule") != cls._REGISTERED_E3A_RULE:
            raise FormalExperimentDagBlocked(
                f"{source.node}: E3a lambda-star derivation rule differs"
            )
        matched_width = payload.get("matched_width")
        common_load = payload.get("common_load")
        if type(matched_width) is not int or type(common_load) is not int:
            raise FormalExperimentDagBlocked(
                f"{source.node}: E3a load/width authority is incomplete"
            )
        locked.update(
            {
                "common_load": common_load,
                "matched_width": matched_width,
                # The E3a reducer's registered prose and the downstream load
                # authority use different canonical spellings.  Accept only
                # that exact upstream spelling, then normalize it here.
                "rule": cls._E5_RULE,
            }
        )
        return E3aLambdaStar.from_e3a_selection(locked)

    def arrival_plan_path(
        self,
        *,
        node: str,
        execution_source_path: str,
        materialized_cell_id: str,
    ) -> str | None:
        from lightcone_spec.experiments.formal_single_operator_content import (
            load_trusted_single_operator_content_bundle,
        )
        from lightcone_spec.experiments.formal_single_operator_loads import (
            E5ArrivalPlan,
            derive_e5_arrival_plan,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            load_formal_single_operator_execution_source,
        )

        if node not in {"e5_pilot", "e5_final"}:
            raise ValueError("E5 arrival resolver received a non-E5 node")
        source_path = _existing_file(execution_source_path, "E5 execution source")
        source = load_formal_single_operator_execution_source(source_path)
        if source.node != node:
            raise ValueError("E5 arrival resolver source node differs")
        cell = self._cell(source, materialized_cell_id)
        if cell.stage != "E5" or cell.task != "production_slo_power_prefix":
            raise ValueError("E5 arrival resolver received a non-headline cell")
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        family = dimensions.get("family")
        if type(block) is not int or family not in {
            "closed_loop",
            "open_loop",
            "trace_or_soak",
            "topology_cohort",
        }:
            raise ValueError("E5 headline cell load axes differ")
        content = load_trusted_single_operator_content_bundle(
            self.config.content_source.absolute_path
        )
        burst = content.burstgpt_release
        active_paths = tuple(
            row.absolute_path for row in burst.assets if row.name == burst.active_asset
        )
        if len(active_paths) != 1:
            raise FormalExperimentDagBlocked(
                f"{node}: trusted BurstGPT active asset is unavailable"
            )
        plan = derive_e5_arrival_plan(
            cell_id=cell.cell_id,
            block=block,
            family=family,
            dimensions=dimensions,
            lambda_star=self._lambda_star(source),
            burstgpt_verification=burst.release_verification,
            burstgpt_active_asset_path=active_paths[0],
            selected_p99_anchor=("p99_extension_anchor_id" in dimensions),
        )
        output_root = source_path.parent / "work" / "e5-arrival-plans"
        output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        output = output_root / f"{cell.cell_id}.json"
        if output.exists():
            existing = E5ArrivalPlan.from_dict(
                _read_canonical_json(output, label="E5 arrival plan")
            )
            if existing != plan:
                raise FormalExperimentDagBlocked(
                    f"{node}: existing E5 arrival plan differs for {cell.cell_id}"
                )
        else:
            _publish_no_replace(output, plan.to_dict())
        if (
            E5ArrivalPlan.from_dict(
                _read_canonical_json(output, label="E5 arrival plan")
            )
            != plan
        ):
            raise RuntimeError("E5 arrival plan changed during publication")
        return str(output)


class DirectoryAuxiliaryPhysicalRuntime:
    """Run and journal the exact E6/E0 pre-materialization campaigns.

    The physical producers remain the scientific owners.  This class only
    resolves one predecessor-bound path catalog, launches a setsid worker, and
    projects its deeply revalidated terminals into schema-6 auxiliary jobs.
    """

    _SOURCE_KINDS: ClassVar[Mapping[str, str]] = {
        "e6_pilot": "e6_interface_fit",
        "e0_tuning": "e0_compatibility",
    }

    def __init__(
        self,
        *,
        config: PathBoundFormalDagDriverConfig,
        store: ExperimentOperatorStore,
        python_executable: str | Path = sys.executable,
    ) -> None:
        if type(config) is not PathBoundFormalDagDriverConfig:
            raise TypeError("auxiliary runtime requires exact driver config")
        if type(store) is not ExperimentOperatorStore:
            raise TypeError("auxiliary runtime requires exact operator store")
        self.config = config
        self.store = store
        self.python_executable = str(
            _existing_file(
                Path(python_executable).resolve(strict=True),
                "auxiliary Python executable",
            )
        )
        self.catalog = _existing_directory(
            config.prerequisite_index_catalog_directory,
            "auxiliary input catalog directory",
        )
        self.root = Path(config.run_root) / "formal-dag-auxiliary"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _binding(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
    ) -> AuxiliaryInputCatalogBinding:
        if node not in self._SOURCE_KINDS or predecessor is None:
            raise FormalExperimentDagBlocked(
                f"{node}: exact auxiliary predecessor is unavailable"
            )
        matches = []
        for path in sorted(self.catalog.glob("*.json")):
            try:
                raw = _read_canonical_json(path, label="auxiliary input catalog")
            except (OSError, ValueError):
                continue
            if raw.get("kind") != _AUXILIARY_INPUT_BINDING_KIND:
                continue
            row = AuxiliaryInputCatalogBinding.from_dict(raw)
            if row.node == node and row.predecessor_completion == predecessor:
                matches.append(row)
        if len(matches) != 1:
            raise FormalExperimentDagBlocked(
                f"{node}: exact predecessor-bound auxiliary inputs are unavailable"
            )
        return matches[0]

    def _attempt(self, node: str) -> int:
        latest = self.store.latest_controller_auxiliary_group(node)
        if latest is None:
            return 1
        if (
            latest["status"] == "FAILED"
            and latest["failure_class"] == "INFRASTRUCTURE"
            and int(latest["attempt"]) < 3
        ):
            return int(latest["attempt"]) + 1
        return int(latest["attempt"])

    @staticmethod
    def _identity(config: PathBoundFormalDagDriverConfig) -> dict[str, str]:
        from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
        from lightcone_spec.experiments.formal_single_operator_content import (
            load_trusted_single_operator_content_bundle,
        )

        lock = protocol_lock_from_dict(
            _read_canonical_json(
                config.protocol_lock.absolute_path, label="ProtocolLock"
            )
        )
        content = load_trusted_single_operator_content_bundle(
            config.content_source.absolute_path
        )
        return {
            "source_sha256": content.source_snapshot.source_snapshot_sha256,
            "patch_sha256": lock.patch_manifest_sha256,
            "registry_sha256": lock.registry_sha256,
            "protocol_lock_sha256": lock.sha256,
            "content_source_sha256": content.semantic_sha256,
        }

    def _e6_campaign(
        self,
        *,
        binding: AuxiliaryInputCatalogBinding,
        physical_root: Path,
    ) -> tuple[Any, Path]:
        from lightcone_spec.config import load_run_config
        from lightcone_spec.experiments.formal_single_operator_e6_interface import (
            E6_MODELS,
            materialize_formal_single_operator_e6_interface_fit_campaign,
        )
        from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

        launches: dict[str, str] = {}
        for item in binding.input_files:
            launch = CompileLaunchManifest.load(item.absolute_path)
            config = load_run_config(launch.run_config_path)
            if config.model.target in launches:
                raise FormalExperimentDagBlocked(
                    "e6_pilot: auxiliary launches repeat a target model"
                )
            launches[config.model.target] = item.absolute_path
        if set(launches) != set(E6_MODELS):
            raise FormalExperimentDagBlocked(
                "e6_pilot: exact two NEXTN launch models are unavailable"
            )
        physical_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        campaign = materialize_formal_single_operator_e6_interface_fit_campaign(
            protocol_lock_path=self.config.protocol_lock.absolute_path,
            predecessor_completion_path=(binding.predecessor_completion.absolute_path),
            trusted_content_bundle_path=self.config.content_source.absolute_path,
            launch_manifest_paths={model: launches[model] for model in E6_MODELS},
            output_root=physical_root,
        )
        return campaign, physical_root / "e6-interface-fit-campaign.json"

    def _e0_campaign(
        self,
        *,
        binding: AuxiliaryInputCatalogBinding,
        physical_root: Path,
    ) -> tuple[Any, Path]:
        from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
            load_e0_prepared_model_backend_interface_receipt,
        )
        from lightcone_spec.experiments.stage_materialization import (
            E0_BACKENDS,
            E0_MODELS,
        )
        from lightcone_spec.orchestration.formal_e0_compatibility_physical import (
            materialize_formal_e0_compatibility_physical_campaign,
        )

        by_key: dict[tuple[str, str], str] = {}
        for item in binding.input_files:
            receipt = load_e0_prepared_model_backend_interface_receipt(
                item.absolute_path
            )
            key = (receipt.model, receipt.backend)
            if key in by_key:
                raise FormalExperimentDagBlocked(
                    "e0_tuning: auxiliary interfaces repeat model/backend"
                )
            by_key[key] = item.absolute_path
        keys = tuple((model, backend) for model in E0_MODELS for backend in E0_BACKENDS)
        if set(by_key) != set(keys):
            raise FormalExperimentDagBlocked(
                "e0_tuning: exact twelve interface receipts are unavailable"
            )
        campaign = materialize_formal_e0_compatibility_physical_campaign(
            protocol_lock_path=self.config.protocol_lock.absolute_path,
            e6_completion_path=binding.predecessor_completion.absolute_path,
            trusted_content_bundle_path=self.config.content_source.absolute_path,
            interface_receipt_paths=tuple(by_key[key] for key in keys),
            output_root=physical_root,
        )
        return campaign, physical_root / "campaign.json"

    @staticmethod
    def _descriptor_path(attempt_root: Path) -> Path:
        return attempt_root / "auxiliary-worker-descriptor.json"

    @staticmethod
    def _load_descriptor(path: Path) -> AuxiliaryWorkerDescriptor:
        return AuxiliaryWorkerDescriptor.from_dict(
            _read_canonical_json(path, label="auxiliary worker descriptor")
        )

    def _publish_descriptor(
        self,
        descriptor: AuxiliaryWorkerDescriptor,
        path: Path,
    ) -> None:
        if path.exists():
            if self._load_descriptor(path) != descriptor:
                raise FormalExperimentDagBlocked(
                    f"{descriptor.node}: auxiliary worker descriptor changed"
                )
            return
        _publish_no_replace(path, descriptor.to_dict())
        if self._load_descriptor(path) != descriptor:
            raise RuntimeError("auxiliary descriptor changed during publication")

    def _reopen_campaign(self, descriptor: AuxiliaryWorkerDescriptor) -> Any:
        if descriptor.node == "e6_pilot":
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                _load_campaign,
            )

            return _load_campaign(descriptor.campaign.absolute_path)
        from lightcone_spec.orchestration.formal_e0_compatibility_physical import (
            revalidate_formal_e0_compatibility_physical_campaign,
        )

        return revalidate_formal_e0_compatibility_physical_campaign(
            descriptor.campaign.absolute_path
        )

    @staticmethod
    def _source_process_hard_timeout_ns(
        node: str,
        campaign_path: str | Path,
    ) -> int:
        if node == "e6_pilot":
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                formal_single_operator_e6_interface_fit_process_hard_timeout_ns,
            )

            return formal_single_operator_e6_interface_fit_process_hard_timeout_ns(
                campaign_path
            )
        if node == "e0_tuning":
            from lightcone_spec.orchestration.formal_e0_compatibility_physical import (
                formal_e0_compatibility_process_hard_timeout_ns,
            )

            return formal_e0_compatibility_process_hard_timeout_ns(campaign_path)
        raise ValueError("auxiliary process timeout received another node")

    def _jobs(
        self,
        descriptor: AuxiliaryWorkerDescriptor,
        campaign: Any,
    ) -> tuple[AuxiliaryJobSpec, ...]:
        identity_base = {
            **self._identity(self.config),
            "predecessor_completion_sha256": (descriptor.predecessor_completion.sha256),
            "campaign_sha256": descriptor.campaign.raw_sha256,
        }
        jobs = []
        if descriptor.node == "e6_pilot":
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                revalidate_formal_single_operator_e6_interface_fit_plan,
            )

            for plan_binding in campaign.plans:
                plan = revalidate_formal_single_operator_e6_interface_fit_plan(
                    plan_binding.absolute_path
                )
                adoption_key = f"e6:{plan.model}"
                command = _semantic_sha256(
                    {
                        "worker_descriptor_sha256": descriptor.sha256,
                        "physical_plan_sha256": plan.sha256,
                        "operation": "execute_e6_interface_fit_plan",
                    }
                )
                jobs.append(
                    AuxiliaryJobSpec(
                        job_id=_semantic_sha256(
                            {
                                "source_kind": descriptor.source_kind,
                                "adoption_key": adoption_key,
                            }
                        ),
                        attempt=descriptor.attempt,
                        adoption_key=adoption_key,
                        scientific_axes={
                            "backend": "NEXTN",
                            "method_role": "Target-only",
                            "model": plan.model,
                            "publication_policy": "none",
                            "recipe_sha256": None,
                            "task": "immutable_metadata_interface_and_fit_preflight",
                        },
                        identity={**identity_base, "physical_plan_sha256": plan.sha256},
                        command_sha256=command,
                        output_directory=plan.evidence_directory,
                    )
                )
        else:
            from lightcone_spec.orchestration.formal_e0_compatibility_physical import (
                revalidate_formal_e0_compatibility_probe_plan,
            )

            for plan_binding in campaign.probe_plans:
                plan = revalidate_formal_e0_compatibility_probe_plan(
                    plan_binding.absolute_path
                )
                adoption_key = f"e0:{plan.model}:{plan.backend}:{plan.task}"
                command = _semantic_sha256(
                    {
                        "worker_descriptor_sha256": descriptor.sha256,
                        "physical_plan_sha256": plan.sha256,
                        "worker_command_sha256": plan.worker_command_sha256,
                        "operation": "execute_e0_compatibility_probe",
                    }
                )
                jobs.append(
                    AuxiliaryJobSpec(
                        job_id=_semantic_sha256(
                            {
                                "source_kind": descriptor.source_kind,
                                "adoption_key": adoption_key,
                            }
                        ),
                        attempt=descriptor.attempt,
                        adoption_key=adoption_key,
                        scientific_axes={
                            "backend": plan.backend,
                            "deployment_task": plan.task,
                            "method_role": "Compatibility",
                            "model": plan.model,
                            "publication_policy": "none",
                            "recipe_sha256": None,
                            "task": "compatibility_decision",
                        },
                        identity={**identity_base, "physical_plan_sha256": plan.sha256},
                        command_sha256=command,
                        output_directory=plan.evidence_directory,
                    )
                )
        return tuple(sorted(jobs, key=lambda row: (row.job_id, row.attempt)))

    def plan(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
    ) -> AuxiliaryPhysicalGroupSpec | None:
        if node not in self._SOURCE_KINDS:
            return None
        binding = self._binding(node, predecessor)
        attempt = self._attempt(node)
        group_id = _semantic_sha256(
            {
                "node": node,
                "source_kind": self._SOURCE_KINDS[node],
                "predecessor_completion_sha256": (
                    binding.predecessor_completion.sha256
                ),
                "auxiliary_input_binding_sha256": binding.sha256,
            }
        )
        attempt_root = self.root / node / f"attempt-{attempt:04d}"
        descriptor_path = self._descriptor_path(attempt_root)
        if attempt_root.exists() and not descriptor_path.exists():
            _preserve_partial_directory(
                attempt_root,
                label=f"{node} incomplete auxiliary attempt",
            )
        attempt_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if descriptor_path.exists():
            descriptor = self._load_descriptor(descriptor_path)
            if (
                descriptor.node != node
                or descriptor.group_id != group_id
                or descriptor.attempt != attempt
                or descriptor.predecessor_completion != binding.predecessor_completion
            ):
                raise FormalExperimentDagBlocked(
                    f"{node}: retained auxiliary descriptor identity differs"
                )
            campaign = self._reopen_campaign(descriptor)
            expected_timeout_ns = self._source_process_hard_timeout_ns(
                node,
                descriptor.campaign.absolute_path,
            )
            if descriptor.process_hard_timeout_ns != expected_timeout_ns:
                raise FormalExperimentDagBlocked(
                    f"{node}: retained auxiliary process timeout differs"
                )
        else:
            physical_root = attempt_root / "physical"
            if node == "e6_pilot":
                campaign, campaign_path = self._e6_campaign(
                    binding=binding,
                    physical_root=physical_root,
                )
                publication_path = attempt_root / "e6-interface-fit-bundle.json"
                evidence_manifest_path = None
            else:
                campaign, campaign_path = self._e0_campaign(
                    binding=binding,
                    physical_root=physical_root,
                )
                publication_path = attempt_root / "e0-compatibility-bundle.json"
                evidence_manifest_path = (
                    attempt_root / "e0-compatibility-evidence-manifest.json"
                )
            descriptor = AuxiliaryWorkerDescriptor(
                schema_version=2,
                kind=_AUXILIARY_WORKER_KIND,
                node=node,  # type: ignore[arg-type]
                source_kind=self._SOURCE_KINDS[node],  # type: ignore[arg-type]
                group_id=group_id,
                attempt=attempt,
                campaign=DriverFileBinding.bind(campaign_path),
                protocol_lock=self.config.protocol_lock,
                predecessor_completion=binding.predecessor_completion,
                content_source=self.config.content_source,
                onlinespec_source_authority=(binding.onlinespec_source_authority),
                publication_output_path=str(publication_path),
                evidence_manifest_output_path=(
                    None
                    if evidence_manifest_path is None
                    else str(evidence_manifest_path)
                ),
                terminal_output_path=str(
                    attempt_root / "auxiliary-worker-terminal.json"
                ),
                heartbeat_output_path=str(
                    attempt_root / "auxiliary-worker-heartbeat.json"
                ),
                process_hard_timeout_ns=self._source_process_hard_timeout_ns(
                    node,
                    campaign_path,
                ),
            )
            self._publish_descriptor(descriptor, descriptor_path)
        jobs = self._jobs(descriptor, campaign)
        gpu_uuids = self._inventory_gpu_uuids()
        worker_argv = (
            self.python_executable,
            "-m",
            "lightcone_spec.orchestration.formal_single_operator_dag_driver",
            "auxiliary-worker",
            "--descriptor",
            str(descriptor_path),
        )
        return AuxiliaryPhysicalGroupSpec(
            group_id=descriptor.group_id,
            attempt=attempt,
            node=node,
            source_kind=descriptor.source_kind,
            jobs=jobs,
            assigned_gpu_uuids=gpu_uuids,
            launch_command_sha256=_semantic_sha256({"argv": list(worker_argv)}),
            output_directory=str(attempt_root),
            process_hard_timeout_ns=descriptor.process_hard_timeout_ns,
        )

    def _inventory_gpu_uuids(self) -> tuple[str, ...]:
        from lightcone_spec.experiments.gpu_pool import GpuInventory

        inventory = GpuInventory.from_dict(
            _read_canonical_json(
                self.config.inventory.absolute_path, label="GPU inventory"
            )
        )
        result = tuple(sorted(row.uuid for row in inventory.devices if row.ready))
        if len(result) != 2:
            raise FormalExperimentDagBlocked(
                "auxiliary runtime requires exact two ready GPU UUIDs"
            )
        return result

    def launch(self, spec: AuxiliaryPhysicalGroupSpec) -> SpawnedProcess:
        from lightcone_spec.orchestration.experiment_operator_production import (
            revalidate_child_start_receipt,
        )

        descriptor_path = self._descriptor_path(Path(spec.output_directory))
        descriptor = self._load_descriptor(descriptor_path)
        worker_argv = (
            self.python_executable,
            "-m",
            "lightcone_spec.orchestration.formal_single_operator_dag_driver",
            "auxiliary-worker",
            "--descriptor",
            str(descriptor_path),
        )
        if _semantic_sha256({"argv": list(worker_argv)}) != (
            spec.launch_command_sha256
        ):
            raise ValueError("auxiliary worker command changed after journaling")
        log_path = Path(spec.output_directory) / "auxiliary-worker.log"
        start_receipt_path = Path(spec.output_directory) / "auxiliary-worker.start.json"
        exit_receipt_path = Path(spec.output_directory) / "auxiliary-worker.exit.json"
        for path in (start_receipt_path, exit_receipt_path):
            if path.exists() or path.is_symlink():
                raise ValueError("auxiliary wrapper receipt path is occupied")
        descriptor.__post_init__()
        if descriptor.process_hard_timeout_ns != spec.process_hard_timeout_ns:
            raise ValueError("auxiliary worker timeout changed after journaling")
        wrapper_argv = (
            self.python_executable,
            "-m",
            "lightcone_spec.orchestration.experiment_operator_production",
            "child-wrapper",
            "--start-receipt",
            str(start_receipt_path),
            "--exit-receipt",
            str(exit_receipt_path),
            "--command-sha256",
            spec.launch_command_sha256,
            "--",
            *worker_argv,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "LIGHTCONE_AUXILIARY_GROUP_ID": spec.group_id,
                "LIGHTCONE_AUXILIARY_ATTEMPT": str(spec.attempt),
                "LIGHTCONE_AUXILIARY_COMMAND_SHA256": (spec.launch_command_sha256),
                "LIGHTCONE_AUXILIARY_HEARTBEAT_PATH": (
                    descriptor.heartbeat_output_path
                ),
            }
        )
        log_handle = log_path.open("xb")
        try:
            process = subprocess.Popen(
                wrapper_argv,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            log_handle.close()
        receipt_sha256 = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if start_receipt_path.exists() or start_receipt_path.is_symlink():
                recovered = revalidate_child_start_receipt(
                    start_receipt_path,
                    command_sha256=spec.launch_command_sha256,
                )
                if (recovered.pid, recovered.pgid) != (process.pid, process.pid):
                    process.terminate()
                    raise ValueError("auxiliary wrapper receipt process differs")
                receipt_sha256 = recovered.receipt_sha256
                break
            if process.poll() is not None:
                break
            time.sleep(0.01)
        return SpawnedProcess(process.pid, process.pid, receipt_sha256)

    @staticmethod
    def _worker_terminal(
        descriptor: AuxiliaryWorkerDescriptor,
    ) -> dict[str, Any] | None:
        path = Path(descriptor.terminal_output_path)
        if not path.exists():
            return None
        value = _read_canonical_json(path, label="auxiliary worker terminal")
        expected = {
            "schema_version",
            "kind",
            "descriptor_sha256",
            "node",
            "attempt",
            "status",
            "exit_code",
            "started_ns",
            "finished_ns",
            "publication",
            "failure_code",
            "failure_class",
            "failure_detail",
        }
        if (
            set(value) != expected
            or value["schema_version"] != 2
            or value["kind"] != _AUXILIARY_WORKER_TERMINAL_KIND
            or value["descriptor_sha256"] != descriptor.sha256
            or value["node"] != descriptor.node
            or value["attempt"] != descriptor.attempt
            or value["status"] not in {"COMPLETE", "FAILED"}
            or type(value["exit_code"]) is not int
            or type(value["started_ns"]) is not int
            or type(value["finished_ns"]) is not int
            or value["started_ns"] < 1
            or value["finished_ns"] <= value["started_ns"]
            or type(value["publication"]) is not dict
        ):
            raise ValueError("auxiliary worker terminal identity differs")
        publication = ControllerArtifactBinding(**value["publication"])
        if ControllerArtifactBinding.bind(publication.absolute_path) != publication:
            raise ValueError("auxiliary worker publication changed")
        if value["status"] == "COMPLETE":
            if value["exit_code"] != 0 or any(
                value[name] is not None
                for name in ("failure_code", "failure_class", "failure_detail")
            ):
                raise ValueError("complete auxiliary worker carries failure")
        elif (
            value["exit_code"] == 0
            or type(value["failure_code"]) is not str
            or not value["failure_code"]
            or value["failure_class"]
            not in {"INFRASTRUCTURE", "SCIENTIFIC", "EXACTNESS", "UNSAFE"}
            or type(value["failure_detail"]) is not str
            or not value["failure_detail"]
        ):
            raise ValueError("failed auxiliary worker lacks failure detail")
        return value

    @staticmethod
    def _process_alive(pid: int, pgid: int) -> bool:
        if type(pid) is not int or type(pgid) is not int or pid < 1 or pgid < 1:
            return False
        try:
            if os.getpgid(pid) != pgid:
                return False
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    @staticmethod
    def _start_receipt_path(spec: AuxiliaryPhysicalGroupSpec) -> Path:
        return Path(spec.output_directory) / "auxiliary-worker.start.json"

    @staticmethod
    def _log_path(spec: AuxiliaryPhysicalGroupSpec) -> Path:
        return Path(spec.output_directory) / "auxiliary-worker.log"

    @staticmethod
    def _heartbeat(
        spec: AuxiliaryPhysicalGroupSpec,
        descriptor: AuxiliaryWorkerDescriptor,
    ) -> Any | None:
        from lightcone_spec.orchestration.experiment_operator import WorkerHeartbeat

        path = Path(descriptor.heartbeat_output_path)
        if not path.exists() and not path.is_symlink():
            return None
        value = _read_canonical_json(path, label="auxiliary child heartbeat")
        expected = {
            "schema_version",
            "kind",
            "cell_id",
            "attempt",
            "command_sha256",
            "worker_pid",
            "sequence",
            "observed_at_ns",
            "phase",
        }
        if (
            set(value) != expected
            or value["schema_version"] != 1
            or value["kind"] != "formal_experiment_child_heartbeat"
            or value["cell_id"] != spec.group_id
            or value["attempt"] != spec.attempt
            or value["command_sha256"] != spec.launch_command_sha256
        ):
            raise ValueError("auxiliary child heartbeat identity differs")
        return WorkerHeartbeat(
            command_sha256=value["command_sha256"],
            worker_pid=value["worker_pid"],
            sequence=value["sequence"],
            observed_at_ns=value["observed_at_ns"],
            phase=value["phase"],
        )

    @staticmethod
    def _gpu_observation(spec: AuxiliaryPhysicalGroupSpec) -> dict[str, Any]:
        try:
            from lightcone_spec.orchestration.experiment_operator_production import (
                query_nvidia_smi,
            )

            rows = {
                row.uuid: row
                for row in query_nvidia_smi()
                if row.uuid in spec.assigned_gpu_uuids
            }
            if set(rows) != set(spec.assigned_gpu_uuids):
                raise ValueError("assigned auxiliary GPU UUID is unavailable")
            return {
                "status": "AVAILABLE",
                "rows": [
                    {
                        "uuid": gpu_uuid,
                        "utilization_percent": rows[gpu_uuid].utilization_percent,
                        "memory_used_mib": rows[gpu_uuid].memory_used_mib,
                        "memory_total_mib": rows[gpu_uuid].memory_total_mib,
                        "power_draw_watts": rows[gpu_uuid].power_draw_watts,
                    }
                    for gpu_uuid in spec.assigned_gpu_uuids
                ],
            }
        except Exception as error:  # noqa: BLE001 - diagnostic-only observation
            return {"status": "ERROR", "error_type": type(error).__name__}

    @staticmethod
    def _descendant_process_group_ids(root_pid: int) -> tuple[int, ...]:
        proc = Path("/proc")
        if not proc.is_dir():
            return (root_pid,)
        parents: dict[int, int] = {}
        groups: dict[int, int] = {}
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                text = (entry / "stat").read_text(encoding="ascii")
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            closing = text.rfind(")")
            fields = text[closing + 2 :].split() if closing >= 0 else []
            if len(fields) < 3:
                continue
            try:
                pid = int(entry.name)
                parents[pid] = int(fields[1])
                groups[pid] = int(fields[2])
            except ValueError:
                continue
        descendants = {root_pid}
        changed = True
        while changed:
            changed = False
            for pid, parent in parents.items():
                if parent in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        pgids = {groups[pid] for pid in descendants if pid in groups}
        pgids.add(root_pid)
        return tuple(sorted((pgid for pgid in pgids if pgid > 0), reverse=True))

    def _signal_process_tree(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        durable_group: Mapping[str, Any],
        *,
        signal_number: int,
    ) -> None:
        from lightcone_spec.orchestration.experiment_operator_production import (
            revalidate_child_start_receipt,
        )

        receipt = revalidate_child_start_receipt(
            self._start_receipt_path(spec),
            command_sha256=spec.launch_command_sha256,
        )
        if (receipt.pid, receipt.pgid) != (
            int(durable_group["pid"]),
            int(durable_group["pgid"]),
        ):
            raise ValueError("auxiliary signal process identity differs")
        registered = durable_group.get("process_start_receipt_sha256")
        if registered is not None and registered != receipt.receipt_sha256:
            raise ValueError("auxiliary start receipt changed before signal")
        targets_path = (
            Path(spec.output_directory) / "auxiliary-termination-targets.json"
        )
        if targets_path.exists():
            pgids = self._load_termination_targets(
                spec,
                expected_start_receipt_sha256=receipt.receipt_sha256,
            )
        else:
            if signal_number != signal.SIGTERM:
                raise ValueError("auxiliary KILL lacks durable TERM targets")
            pgids = self._descendant_process_group_ids(receipt.pid)
            _publish_no_replace(
                targets_path,
                {
                    "schema_version": 1,
                    "kind": "formal_single_operator_auxiliary_termination_targets",
                    "group_id": spec.group_id,
                    "attempt": spec.attempt,
                    "process_start_receipt_sha256": receipt.receipt_sha256,
                    "pgids": list(pgids),
                },
            )
        for pgid in pgids:
            try:
                os.killpg(pgid, signal_number)
            except ProcessLookupError:
                continue

    @staticmethod
    def _load_termination_targets(
        spec: AuxiliaryPhysicalGroupSpec,
        *,
        expected_start_receipt_sha256: str | None = None,
    ) -> tuple[int, ...]:
        path = Path(spec.output_directory) / "auxiliary-termination-targets.json"
        value = _read_canonical_json(path, label="auxiliary termination targets")
        expected_fields = {
            "schema_version",
            "kind",
            "group_id",
            "attempt",
            "process_start_receipt_sha256",
            "pgids",
        }
        receipt_sha256 = _require_sha256(
            value.get("process_start_receipt_sha256"),
            "auxiliary termination target start receipt",
        )
        pgids = value.get("pgids")
        if (
            set(value) != expected_fields
            or value.get("schema_version") != 1
            or value.get("kind")
            != "formal_single_operator_auxiliary_termination_targets"
            or value.get("group_id") != spec.group_id
            or value.get("attempt") != spec.attempt
            or (
                expected_start_receipt_sha256 is not None
                and receipt_sha256 != expected_start_receipt_sha256
            )
            or type(pgids) is not list
            or not pgids
            or pgids != sorted(set(pgids), reverse=True)
            or any(type(pgid) is not int or pgid < 1 for pgid in pgids)
        ):
            raise ValueError("auxiliary termination targets differ")
        return tuple(pgids)

    @classmethod
    def _termination_targets_alive(cls, spec: AuxiliaryPhysicalGroupSpec) -> bool:
        path = Path(spec.output_directory) / "auxiliary-termination-targets.json"
        if not path.exists() and not path.is_symlink():
            return False
        pgids = cls._load_termination_targets(spec)
        for pgid in pgids:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                return True
            return True
        return False

    def _send_auxiliary_termination_signal(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        durable_group: Mapping[str, Any],
        *,
        signal_number: int,
        sent_at_ns: int,
    ) -> None:
        signal_name = "TERM" if signal_number == signal.SIGTERM else "KILL"
        try:
            self._signal_process_tree(
                spec,
                durable_group,
                signal_number=signal_number,
            )
        except Exception as error:
            self.store.record_watchdog_event(
                event_type="DAG_AUXILIARY_TERMINATION_SIGNAL_FAILED",
                severity="CRITICAL",
                cell_id=None,
                attempt=None,
                payload={
                    "node": spec.node,
                    "group_id": spec.group_id,
                    "group_attempt": spec.attempt,
                    "requested_signal": signal_name,
                    "exception_type": type(error).__name__,
                },
                occurred_at_ns=sent_at_ns,
            )
            self.store.set_dispatch_stop(
                f"auxiliary_process_{signal_name.lower()}_failed",
                stopped_at_ns=sent_at_ns,
            )
            raise FormalExperimentDagBlocked(
                f"{spec.node}: auxiliary {signal_name} failed; dispatch STOP is sealed"
            ) from error
        self.store.record_controller_auxiliary_termination_signal(
            spec,
            signal_name=signal_name,
            sent_at_ns=sent_at_ns,
        )

    def _infrastructure_terminal(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        durable_group: Mapping[str, Any],
        *,
        failure_code: str,
        exclusion_reason: str,
        finished_ns: int,
    ) -> AuxiliaryGroupTerminal:
        from lightcone_spec.orchestration.experiment_operator import TerminalEvidence

        path = Path(spec.output_directory) / "auxiliary-operator-failure.json"
        value = {
            "schema_version": 1,
            "kind": "formal_single_operator_auxiliary_operator_failure",
            "group_id": spec.group_id,
            "attempt": spec.attempt,
            "launch_command_sha256": spec.launch_command_sha256,
            "failure_code": failure_code,
            "exclusion_reason": exclusion_reason,
            "started_ns": int(durable_group["started_at_ns"]),
            "finished_ns": finished_ns,
        }
        if path.exists():
            if _read_canonical_json(path, label="auxiliary operator failure") != value:
                raise ValueError("auxiliary operator failure publication differs")
        else:
            _publish_no_replace(path, value)
        publication = ControllerArtifactBinding.bind(path)
        evidence = {str(path): publication.sha256}
        log_path = self._log_path(spec)
        raw_log_sha256 = publication.sha256
        if log_path.exists() or log_path.is_symlink():
            raw_log_sha256 = _file_sha256(log_path)
            evidence[str(log_path)] = raw_log_sha256
        for candidate in (
            self._start_receipt_path(spec),
            Path(spec.output_directory) / "auxiliary-worker.exit.json",
            Path(spec.output_directory) / "auxiliary-worker-heartbeat.json",
            Path(spec.output_directory) / "auxiliary-termination-targets.json",
        ):
            if candidate.exists() or candidate.is_symlink():
                evidence[str(candidate)] = _file_sha256(candidate)
        terminals = {
            job.job_id: TerminalEvidence(
                status="FAILED",
                exit_code=None,
                atomic_publication_sha256=publication.sha256,
                terminal_sha256=publication.sha256,
                raw_log_sha256=raw_log_sha256,
                evidence_files=evidence,
                failure_class="INFRASTRUCTURE",
                failure_code=failure_code,
                exclusion_reason=exclusion_reason,
                included_in_analysis=False,
                started_ns=int(durable_group["started_at_ns"]),
                finished_ns=finished_ns,
            )
            for job in spec.jobs
        }
        duration = max(
            0.0,
            (finished_ns - int(durable_group["started_at_ns"])) / 1e9,
        )
        reserved = duration * len(spec.assigned_gpu_uuids)
        return AuxiliaryGroupTerminal(
            publication=publication,
            terminals=terminals,
            compute_gpu_seconds=reserved,
            reserved_gpu_seconds=reserved,
        )

    def _deep_publication(self, descriptor: AuxiliaryWorkerDescriptor) -> None:
        if descriptor.node == "e6_pilot":
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                revalidate_formal_single_operator_e6_interface_fit_bundle,
            )

            revalidate_formal_single_operator_e6_interface_fit_bundle(
                descriptor.publication_output_path
            )
            return
        from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
            revalidate_trusted_e0_compatibility_bundle,
        )

        revalidate_trusted_e0_compatibility_bundle(descriptor.publication_output_path)

    def _job_terminal_evidence(
        self,
        *,
        descriptor: AuxiliaryWorkerDescriptor,
        campaign: Any,
        publication: ControllerArtifactBinding,
        worker: Mapping[str, Any],
    ) -> tuple[dict[str, Any], float]:
        from lightcone_spec.orchestration.experiment_operator import TerminalEvidence

        result: dict[str, TerminalEvidence] = {}
        compute_seconds = 0.0
        if worker["status"] == "FAILED":
            failure_digest = publication.sha256
            evidence = {publication.absolute_path: publication.sha256}
            log_path = Path(descriptor.terminal_output_path).with_name(
                "auxiliary-worker.log"
            )
            raw_log_sha256 = failure_digest
            if log_path.exists() or log_path.is_symlink():
                raw_log_sha256 = _file_sha256(log_path)
                evidence[str(log_path)] = raw_log_sha256
            for job in self._jobs(descriptor, campaign):
                result[job.job_id] = TerminalEvidence(
                    status="FAILED",
                    exit_code=int(worker["exit_code"]),
                    atomic_publication_sha256=publication.sha256,
                    terminal_sha256=failure_digest,
                    raw_log_sha256=raw_log_sha256,
                    evidence_files=evidence,
                    failure_class=str(worker["failure_class"]),
                    failure_code=str(worker["failure_code"]),
                    exclusion_reason=str(worker["failure_detail"]),
                    included_in_analysis=False,
                    started_ns=int(worker["started_ns"]),
                    finished_ns=int(worker["finished_ns"]),
                )
            return result, compute_seconds
        self._deep_publication(descriptor)
        jobs = {job.adoption_key: job for job in self._jobs(descriptor, campaign)}
        if descriptor.node == "e6_pilot":
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                revalidate_formal_single_operator_e6_interface_fit_plan,
                revalidate_formal_single_operator_e6_interface_fit_terminal,
            )

            for binding in campaign.plans:
                plan = revalidate_formal_single_operator_e6_interface_fit_plan(
                    binding.absolute_path
                )
                terminal_path = (
                    Path(plan.evidence_directory) / "e6-interface-fit-terminal.json"
                )
                terminal = revalidate_formal_single_operator_e6_interface_fit_terminal(
                    terminal_path
                )
                job = jobs[f"e6:{plan.model}"]
                compute_seconds += (terminal.finished_ns - terminal.started_ns) * 2e-9
                result[job.job_id] = TerminalEvidence(
                    status="COMPLETE",
                    exit_code=0,
                    atomic_publication_sha256=publication.sha256,
                    terminal_sha256=_file_sha256(terminal_path),
                    junit_sha256=terminal.junit_xml.raw_sha256,
                    raw_log_sha256=terminal.runner_log.raw_sha256,
                    evidence_files={
                        str(terminal_path): _file_sha256(terminal_path),
                        terminal.junit_xml.absolute_path: terminal.junit_xml.raw_sha256,
                        terminal.runner_log.absolute_path: terminal.runner_log.raw_sha256,
                    },
                    included_in_analysis=False,
                    exclusion_reason="interface_preflight_non_headline",
                    started_ns=terminal.started_ns,
                    finished_ns=terminal.finished_ns,
                )
        else:
            from lightcone_spec.orchestration.formal_e0_compatibility_physical import (
                revalidate_formal_e0_compatibility_physical_terminal,
                revalidate_formal_e0_compatibility_probe_group,
                revalidate_formal_e0_compatibility_probe_plan,
            )

            group_intervals = []
            for group_binding in campaign.groups:
                group = revalidate_formal_e0_compatibility_probe_group(
                    group_binding.absolute_path
                )
                group_terminals = []
                for plan_binding in group.probes:
                    plan = revalidate_formal_e0_compatibility_probe_plan(
                        plan_binding.absolute_path
                    )
                    terminal_path = Path(plan.terminal_output_path)
                    terminal = revalidate_formal_e0_compatibility_physical_terminal(
                        terminal_path
                    )
                    group_terminals.append(terminal)
                    job = jobs[f"e0:{plan.model}:{plan.backend}:{plan.task}"]
                    result[job.job_id] = TerminalEvidence(
                        status="COMPLETE",
                        exit_code=0,
                        atomic_publication_sha256=publication.sha256,
                        terminal_sha256=_file_sha256(terminal_path),
                        junit_sha256=terminal.junit_sha256,
                        raw_log_sha256=terminal.stdout_sha256,
                        evidence_files={
                            str(terminal_path): _file_sha256(terminal_path),
                        },
                        included_in_analysis=False,
                        exclusion_reason="compatibility_decision_non_headline",
                        started_ns=terminal.started_ns,
                        finished_ns=terminal.finished_ns,
                    )
                if group.compile_launch_manifest is not None:
                    group_intervals.append(
                        (
                            min(row.started_ns for row in group_terminals),
                            max(row.finished_ns for row in group_terminals),
                        )
                    )
            compute_seconds = sum(
                (end - start) * 1e-9 for start, end in group_intervals
            )
        if set(result) != {job.job_id for job in self._jobs(descriptor, campaign)}:
            raise ValueError("auxiliary terminal evidence coverage differs")
        return result, compute_seconds

    def terminal(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        durable_group: Mapping[str, Any],
    ) -> AuxiliaryGroupTerminal | None:
        from lightcone_spec.orchestration.experiment_operator_production import (
            revalidate_child_start_receipt,
        )

        descriptor = self._load_descriptor(
            self._descriptor_path(Path(spec.output_directory))
        )
        if (
            descriptor.group_id != spec.group_id
            or descriptor.process_hard_timeout_ns != spec.process_hard_timeout_ns
        ):
            raise ValueError("auxiliary durable runtime contract differs")
        worker = self._worker_terminal(descriptor)
        if durable_group.get(
            "termination_reason"
        ) is not None and self._termination_targets_alive(spec):
            now_ns = time.time_ns()
            if durable_group.get("term_sent_at_ns") is None:
                self._send_auxiliary_termination_signal(
                    spec,
                    durable_group,
                    signal_number=signal.SIGTERM,
                    sent_at_ns=now_ns,
                )
            elif (
                durable_group.get("kill_sent_at_ns") is None
                and now_ns - int(durable_group["term_sent_at_ns"])
                > _AUXILIARY_TERMINATION_GRACE_NS
            ):
                self._send_auxiliary_termination_signal(
                    spec,
                    durable_group,
                    signal_number=signal.SIGKILL,
                    sent_at_ns=now_ns,
                )
            return None
        if worker is None:
            now_ns = time.time_ns()
            pid = durable_group.get("pid")
            pgid = durable_group.get("pgid")
            if pid is None or pgid is None:
                start_path = self._start_receipt_path(spec)
                if start_path.exists() or start_path.is_symlink():
                    recovered = revalidate_child_start_receipt(
                        start_path,
                        command_sha256=spec.launch_command_sha256,
                    )
                    if recovered.started_ns < int(durable_group["started_at_ns"]):
                        raise ValueError(
                            "auxiliary start receipt predates RUNNING commit"
                        )
                    self.store.attach_controller_auxiliary_group_process(
                        spec,
                        pid=recovered.pid,
                        pgid=recovered.pgid,
                        process_start_receipt_sha256=recovered.receipt_sha256,
                        attached_at_ns=now_ns,
                    )
                    durable_group = (
                        self.store.latest_controller_auxiliary_group(spec.node)
                        or durable_group
                    )
                    pid, pgid = recovered.pid, recovered.pgid
                elif (
                    now_ns - int(durable_group["started_at_ns"])
                    <= _AUXILIARY_PROCESS_ATTACH_GRACE_NS
                ):
                    return None
                else:
                    return self._infrastructure_terminal(
                        spec,
                        durable_group,
                        failure_code="INFRASTRUCTURE:START_RECEIPT_NOT_PUBLISHED",
                        exclusion_reason="auxiliary_job_launch_proven_absent",
                        finished_ns=now_ns,
                    )
            assert type(pid) is int and type(pgid) is int
            if durable_group.get("process_start_receipt_sha256") is None:
                start_path = self._start_receipt_path(spec)
                if start_path.exists() or start_path.is_symlink():
                    recovered = revalidate_child_start_receipt(
                        start_path,
                        command_sha256=spec.launch_command_sha256,
                    )
                    if (recovered.pid, recovered.pgid) != (pid, pgid):
                        raise ValueError("auxiliary late start receipt process differs")
                    self.store.attach_controller_auxiliary_group_process(
                        spec,
                        pid=pid,
                        pgid=pgid,
                        process_start_receipt_sha256=recovered.receipt_sha256,
                        attached_at_ns=now_ns,
                    )
                    durable_group = (
                        self.store.latest_controller_auxiliary_group(spec.node)
                        or durable_group
                    )
                elif (
                    now_ns - int(durable_group["started_at_ns"])
                    > _AUXILIARY_PROCESS_ATTACH_GRACE_NS
                ):
                    return self._infrastructure_terminal(
                        spec,
                        durable_group,
                        failure_code="INFRASTRUCTURE:START_RECEIPT_NOT_PUBLISHED",
                        exclusion_reason="auxiliary_job_launch_proven_absent",
                        finished_ns=now_ns,
                    )
            registered_receipt = durable_group.get("process_start_receipt_sha256")
            if registered_receipt is not None:
                recovered = revalidate_child_start_receipt(
                    self._start_receipt_path(spec),
                    command_sha256=spec.launch_command_sha256,
                )
                if (recovered.pid, recovered.pgid) != (
                    pid,
                    pgid,
                ) or recovered.receipt_sha256 != registered_receipt:
                    raise ValueError("auxiliary RUNNING process start identity differs")
            alive = self._process_alive(pid, pgid)
            if alive:
                heartbeat = self._heartbeat(spec, descriptor)
                log_path = self._log_path(spec)
                log_size = (
                    log_path.stat(follow_symlinks=False).st_size
                    if log_path.exists() and not log_path.is_symlink()
                    else 0
                )
                self.store.record_controller_auxiliary_observation(
                    spec,
                    log_size_bytes=log_size,
                    heartbeat=heartbeat,
                    gpu_observation=self._gpu_observation(spec),
                    observed_at_ns=now_ns,
                )
                refreshed = self.store.latest_controller_auxiliary_group(spec.node)
                assert refreshed is not None
                hard_timeout_ns = spec.process_hard_timeout_ns
                if hard_timeout_ns is None:
                    raise ValueError("auxiliary runtime lacks a source-owned timeout")
                if (
                    refreshed["last_log_growth_ns"] is not None
                    and now_ns - int(refreshed["last_log_growth_ns"])
                    > _AUXILIARY_LOG_STALL_WARNING_NS
                    and int(durable_group["updated_at_ns"])
                    - int(refreshed["last_log_growth_ns"])
                    <= _AUXILIARY_LOG_STALL_WARNING_NS
                ):
                    self.store.record_watchdog_event(
                        event_type="DAG_AUXILIARY_LOG_STALL_WARNING",
                        severity="WARNING",
                        cell_id=None,
                        attempt=None,
                        payload={
                            "node": spec.node,
                            "group_id": spec.group_id,
                            "group_attempt": spec.attempt,
                            "child_heartbeat_present": heartbeat is not None,
                            "hard_termination": False,
                        },
                        occurred_at_ns=now_ns,
                    )
                if refreshed["termination_reason"] is None and (
                    now_ns - int(refreshed["started_at_ns"]) > hard_timeout_ns
                ):
                    self.store.request_controller_auxiliary_termination(
                        spec,
                        reason="SOURCE_PROCESS_HARD_TIMEOUT_EXCEEDED",
                        requested_at_ns=now_ns,
                    )
                    refreshed = self.store.latest_controller_auxiliary_group(spec.node)
                    assert refreshed is not None
                heartbeat_reference_ns = (
                    int(refreshed["started_at_ns"])
                    if refreshed["heartbeat_at_ns"] is None
                    else int(refreshed["heartbeat_at_ns"])
                )
                if (
                    refreshed["termination_reason"] is None
                    and now_ns - heartbeat_reference_ns > _AUXILIARY_HEARTBEAT_STALE_NS
                ):
                    self.store._record_general_finding_once(
                        event_type="DAG_AUXILIARY_CHILD_HEARTBEAT_STALE",
                        severity="CRITICAL",
                        payload={
                            "node": spec.node,
                            "group_id": spec.group_id,
                            "group_attempt": spec.attempt,
                            "heartbeat_age_seconds": (
                                (now_ns - heartbeat_reference_ns) / 1e9
                            ),
                            "heartbeat_sequence": refreshed["heartbeat_sequence"],
                            "automatic_signal": False,
                            "hard_termination": False,
                        },
                        now_ns=now_ns,
                        repeat_seconds=_AUXILIARY_HEARTBEAT_STALE_NS / 1e9,
                    )
                    self.store.set_dispatch_stop(
                        "auxiliary_child_heartbeat_stale",
                        stopped_at_ns=now_ns,
                    )
                    return None
                if refreshed["termination_reason"] is not None:
                    if refreshed["term_sent_at_ns"] is None:
                        self._send_auxiliary_termination_signal(
                            spec,
                            refreshed,
                            signal_number=signal.SIGTERM,
                            sent_at_ns=now_ns,
                        )
                    elif (
                        refreshed["kill_sent_at_ns"] is None
                        and now_ns - int(refreshed["term_sent_at_ns"])
                        > _AUXILIARY_TERMINATION_GRACE_NS
                    ):
                        self._send_auxiliary_termination_signal(
                            spec,
                            refreshed,
                            signal_number=signal.SIGKILL,
                            sent_at_ns=now_ns,
                        )
                return None
            reason = (
                "auxiliary_source_process_hard_timeout"
                if durable_group.get("termination_reason") is not None
                else "auxiliary_process_exited_without_terminal"
            )
            return self._infrastructure_terminal(
                spec,
                durable_group,
                failure_code="INFRASTRUCTURE:AUXILIARY_PROCESS_LOST",
                exclusion_reason=reason,
                finished_ns=now_ns,
            )
        campaign = self._reopen_campaign(descriptor)
        publication = ControllerArtifactBinding(**worker["publication"])
        terminals, compute_seconds = self._job_terminal_evidence(
            descriptor=descriptor,
            campaign=campaign,
            publication=publication,
            worker=worker,
        )
        reserved = (
            (int(worker["finished_ns"]) - int(worker["started_ns"]))
            * len(spec.assigned_gpu_uuids)
            * 1e-9
        )
        return AuxiliaryGroupTerminal(
            publication=publication,
            terminals=terminals,
            compute_gpu_seconds=compute_seconds,
            reserved_gpu_seconds=max(reserved, compute_seconds),
        )

    @staticmethod
    def _materialized_cells(node_materialization: ControllerArtifactBinding) -> Any:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            rebuild_formal_single_operator_node_materialization,
        )

        return rebuild_formal_single_operator_node_materialization(
            node_materialization.absolute_path
        )

    def adoptions(
        self,
        node: str,
        node_materialization: ControllerArtifactBinding,
        spec: AuxiliaryPhysicalGroupSpec,
    ) -> tuple[AuxiliaryCellAdoption, ...]:
        rebuilt = self._materialized_cells(node_materialization)
        jobs = {job.adoption_key: job for job in spec.jobs}
        cells: dict[str, Any] = {}
        if node == "e6_pilot":
            for cell in rebuilt.materialization.cells:
                if cell.task == "immutable_metadata_interface_and_fit_preflight":
                    cells[f"e6:{cell.model}"] = cell
        elif node == "e0_tuning":
            for cell in rebuilt.materialization.cells:
                if cell.task == "compatibility_decision":
                    cells[
                        f"e0:{cell.model}:{cell.backend}:"
                        f"{dict(cell.dimensions).get('deployment_task')}"
                    ] = cell
        else:
            raise ValueError("auxiliary adoption received another node")
        if set(cells) != set(jobs):
            raise ValueError("auxiliary adoption keys differ from materialization")
        stage = self.store.controller_node(node)
        adoptions = []
        for key, job in jobs.items():
            cell = cells[key]
            dimensions = dict(cell.dimensions)
            # The pre-materialization job can bind only input axes.  Output
            # fields such as E0 disposition and E6 verified proof hashes first
            # exist in the later materialization and remain bound there; do not
            # manufacture them before the physical observation.
            adoptions.append(
                AuxiliaryCellAdoption(
                    job_id=job.job_id,
                    job_attempt=job.attempt,
                    adoption_key=job.adoption_key,
                    attempt=CellAttemptSpec(
                        cell_id=cell.cell_id,
                        attempt=job.attempt,
                        stage=str(stage["stage"]),
                        phase=str(stage["phase"]),
                        block=(
                            None
                            if dimensions.get("block") is None
                            else str(dimensions["block"])
                        ),
                        seed=(
                            dimensions["seed"]
                            if type(dimensions.get("seed")) is int
                            else None
                        ),
                        scientific_axes=job.scientific_axes,
                        identity=job.identity,
                        command_sha256=job.command_sha256,
                        output_directory=job.output_directory,
                    ),
                )
            )
        return tuple(sorted(adoptions, key=lambda row: (row.job_id, row.job_attempt)))

    def actual_result_paths(
        self,
        node: str,
        attempts: tuple[dict[str, Any], ...],
    ) -> Mapping[str, str]:
        if node not in self._SOURCE_KINDS:
            return {}
        latest = self.store.latest_controller_auxiliary_group(node)
        if latest is None or latest["status"] != "COMPLETE":
            raise FormalExperimentDagBlocked(
                f"{node}: completed auxiliary publication is unavailable"
            )
        descriptor = self._load_descriptor(
            self._descriptor_path(Path(str(latest["output_directory"])))
        )
        campaign = self._reopen_campaign(descriptor)
        paths: dict[str, str] = {}
        by_key = {
            str(row["adopted_cell_id"]): str(row["adoption_key"])
            for row in latest["jobs"]
            if row.get("adopted_cell_id") is not None
        }
        if descriptor.node == "e6_pilot":
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                revalidate_formal_single_operator_e6_interface_fit_plan,
            )

            terminal_by_key = {}
            for binding in campaign.plans:
                plan = revalidate_formal_single_operator_e6_interface_fit_plan(
                    binding.absolute_path
                )
                terminal_by_key[f"e6:{plan.model}"] = str(
                    Path(plan.evidence_directory) / "e6-interface-fit-terminal.json"
                )
        else:
            from lightcone_spec.orchestration.formal_e0_compatibility_physical import (
                revalidate_formal_e0_compatibility_probe_plan,
            )

            # Every compatibility cell validates and selects its row from the
            # same atomic schema-2 bundle; never expose a probe directory as a
            # result merely because it has the expected name.
            terminal_by_key = {
                f"e0:{plan.model}:{plan.backend}:{plan.task}": (
                    descriptor.publication_output_path
                )
                for binding in campaign.probe_plans
                for plan in (
                    revalidate_formal_e0_compatibility_probe_plan(
                        binding.absolute_path
                    ),
                )
            }
        attempt_ids = {str(row["cell_id"]) for row in attempts}
        for cell_id, key in by_key.items():
            if cell_id not in attempt_ids or key not in terminal_by_key:
                raise ValueError("auxiliary actual-result adoption differs")
            paths[cell_id] = terminal_by_key[key]
        return paths


class IsolatedInterferenceGateResolver:
    """Safe fallback when no paired-BCa gate producer is installed.

    It does not claim that calibration failed; it merely withholds concurrent
    headline authorization and binds that decision to the completed preflight.
    """

    def resolve(
        self,
        *,
        completion: ControllerArtifactBinding,
        actual_result_paths: Mapping[str, str],
        gpu_uuids: tuple[str, ...],
    ) -> InterferenceEnvelope:
        del actual_result_paths
        return InterferenceEnvelope("ISOLATED", gpu_uuids, completion.sha256)


class FreshPreflightInterferenceGateResolver:
    """Apply the registered paired BCa gate to the fresh exact-eight evidence."""

    @staticmethod
    def diagnostic_path(completion: ControllerArtifactBinding) -> Path:
        """Return the source-owned, run-specific diagnostic publication path."""

        if type(completion) is not ControllerArtifactBinding:
            raise TypeError("fresh interference diagnostic requires a completion")
        return Path(completion.absolute_path).with_name(
            _FRESH_INTERFERENCE_DIAGNOSTIC_FILENAME
        )

    @classmethod
    def diagnostic_binding(
        cls,
        completion: ControllerArtifactBinding,
    ) -> ControllerArtifactBinding:
        """Deep-open the retained diagnostic used by the scheduler envelope."""

        path = cls.diagnostic_path(completion)
        value = _read_canonical_json(path, label="fresh interference diagnostic")
        required = {
            "schema_version",
            "kind",
            "stage_completion",
            "exact_ten_completion",
            "gpu_uuids",
            "status",
            "scheduler_mode",
            "diagnostic",
            "proof_row_sha256s",
            "metric_anchor_cell_id",
            "independent_block_count",
            "observation_request_count",
            "completed_request_count",
            "paired_trace_request_count",
            "request_counts_by_observation",
            "confidence_level",
            "reducer_method",
        }
        if (
            set(value) != required
            or value["schema_version"] != _FRESH_INTERFERENCE_DIAGNOSTIC_SCHEMA_VERSION
            or value["kind"] != _FRESH_INTERFERENCE_DIAGNOSTIC_KIND
            or value["stage_completion"] != asdict(completion)
        ):
            raise ValueError("fresh interference diagnostic identity differs")
        diagnostic = value["diagnostic"]
        if type(diagnostic) is not dict:
            raise TypeError("fresh interference diagnostic body differs")
        from lightcone_spec.experiments.interference_authority import (
            InterferenceCalibrationGroupDiagnostic,
        )

        parsed_diagnostic = InterferenceCalibrationGroupDiagnostic(
            **{
                **diagnostic,
                "reason_codes": tuple(diagnostic.get("reason_codes", ())),
                "raw_observation_sha256s": tuple(
                    diagnostic.get("raw_observation_sha256s", ())
                ),
                "goodput_ratios": tuple(
                    tuple(row) for row in diagnostic.get("goodput_ratios", ())
                ),
                "p99_itl_ratios": tuple(
                    tuple(row) for row in diagnostic.get("p99_itl_ratios", ())
                ),
            }
        )
        if (
            parsed_diagnostic.to_dict() != diagnostic
            or parsed_diagnostic.group_id != "formal-preflight-static-two-way"
            or parsed_diagnostic.simultaneous_jobs != 2
            or len(parsed_diagnostic.raw_observation_sha256s) != 8
            or len(parsed_diagnostic.goodput_ratios) != 4
            or len(parsed_diagnostic.p99_itl_ratios) not in {0, 4}
            or value["status"] != parsed_diagnostic.status
            or value["scheduler_mode"]
            != ("DUAL_SINGLE" if value["status"] == "PASS" else "ISOLATED")
            or value["independent_block_count"] != 2
            or value["confidence_level"] != 0.95
            or value["reducer_method"] != "paired_bca_mean_log_ratio_v1"
        ):
            raise ValueError("fresh interference diagnostic contract differs")
        for name in (
            "observation_request_count",
            "completed_request_count",
            "paired_trace_request_count",
        ):
            count = value[name]
            if type(count) is not int or count < 1:
                raise ValueError(f"fresh interference {name} is invalid")
        rows = value["request_counts_by_observation"]
        if type(rows) is not list or len(rows) != 8:
            raise ValueError("fresh interference observation coverage differs")
        row_fields = {
            "materialized_cell_id",
            "mode",
            "repetition",
            "slot",
            "request_count",
            "completed_request_count",
        }
        for row in rows:
            if type(row) is not dict or set(row) != row_fields:
                raise ValueError("fresh interference observation row differs")
            _require_sha256(
                row["materialized_cell_id"],
                "fresh interference materialized cell",
            )
            if (
                row["mode"] not in {"isolated", "concurrent"}
                or row["repetition"] not in {0, 1}
                or row["slot"] not in {0, 1}
                or type(row["request_count"]) is not int
                or row["request_count"] < 1
                or type(row["completed_request_count"]) is not int
                or not 0 <= row["completed_request_count"] <= row["request_count"]
            ):
                raise ValueError("fresh interference observation row is invalid")
        keys = {(row["mode"], row["repetition"], row["slot"]) for row in rows}
        if keys != {
            (mode, repetition, slot)
            for mode in ("isolated", "concurrent")
            for repetition in range(2)
            for slot in range(2)
        }:
            raise ValueError("fresh interference observation keys differ")
        if (
            sum(row["request_count"] for row in rows)
            != value["observation_request_count"]
            or sum(row["completed_request_count"] for row in rows)
            != value["completed_request_count"]
            or sum(row["request_count"] for row in rows if row["mode"] == "isolated")
            != value["paired_trace_request_count"]
        ):
            raise ValueError("fresh interference request accounting differs")
        proof_rows = value["proof_row_sha256s"]
        if (
            type(proof_rows) is not list
            or len(proof_rows) != 8
            or proof_rows != sorted(set(proof_rows))
        ):
            raise ValueError("fresh interference proof identities differ")
        for digest in proof_rows:
            _require_sha256(digest, "fresh interference proof row")
        _require_sha256(
            value["metric_anchor_cell_id"],
            "fresh interference metric anchor",
        )
        if value["metric_anchor_cell_id"] not in {
            row["materialized_cell_id"] for row in rows
        }:
            raise ValueError("fresh interference metric anchor is foreign")
        gpu_uuids = value["gpu_uuids"]
        if (
            type(gpu_uuids) is not list
            or len(gpu_uuids) != 2
            or gpu_uuids != sorted(set(gpu_uuids))
            or any(type(gpu_uuid) is not str or not gpu_uuid for gpu_uuid in gpu_uuids)
        ):
            raise ValueError("fresh interference GPU inventory differs")
        exact_ten = value["exact_ten_completion"]
        if type(exact_ten) is not dict or set(exact_ten) != {
            "absolute_path",
            "sha256",
        }:
            raise ValueError("fresh interference exact-ten binding differs")
        if _file_sha256(exact_ten["absolute_path"]) != _require_sha256(
            exact_ten["sha256"],
            "fresh interference exact-ten completion",
        ):
            raise ValueError("fresh interference exact-ten completion changed")
        return ControllerArtifactBinding.bind(path)

    @classmethod
    def _publish_diagnostic(
        cls,
        *,
        completion: ControllerArtifactBinding,
        exact_ten_completion_path: str,
        gpu_uuids: tuple[str, ...],
        proof_rows: tuple[Any, ...],
        diagnostic: Any,
    ) -> ControllerArtifactBinding:
        """Publish or exactly reopen the complete fresh calibration result."""

        request_rows = [
            {
                "materialized_cell_id": row.materialized_cell_id,
                "mode": row.mode,
                "repetition": row.repetition,
                "slot": row.slot,
                "request_count": len(row.observation.request_ids),
                "completed_request_count": row.observation.completed_requests,
            }
            for row in sorted(
                proof_rows,
                key=lambda item: (
                    item.mode,
                    item.repetition,
                    item.slot,
                    item.materialized_cell_id,
                ),
            )
        ]
        isolated = [row for row in request_rows if row["mode"] == "isolated"]
        if (
            len(request_rows) != 8
            or len(isolated) != 4
            or {row["repetition"] for row in request_rows} != {0, 1}
            or {row["slot"] for row in request_rows} != {0, 1}
        ):
            raise ValueError("fresh interference diagnostic coverage differs")
        mode = "DUAL_SINGLE" if diagnostic.status == "PASS" else "ISOLATED"
        value = {
            "schema_version": _FRESH_INTERFERENCE_DIAGNOSTIC_SCHEMA_VERSION,
            "kind": _FRESH_INTERFERENCE_DIAGNOSTIC_KIND,
            "stage_completion": asdict(completion),
            "exact_ten_completion": {
                "absolute_path": exact_ten_completion_path,
                "sha256": _file_sha256(exact_ten_completion_path),
            },
            "gpu_uuids": list(gpu_uuids),
            "status": diagnostic.status,
            "scheduler_mode": mode,
            "diagnostic": diagnostic.to_dict(),
            "proof_row_sha256s": sorted(row.sha256 for row in proof_rows),
            "metric_anchor_cell_id": min(
                row.materialized_cell_id for row in proof_rows
            ),
            "independent_block_count": 2,
            "observation_request_count": sum(
                int(row["request_count"]) for row in request_rows
            ),
            "completed_request_count": sum(
                int(row["completed_request_count"]) for row in request_rows
            ),
            "paired_trace_request_count": sum(
                int(row["request_count"]) for row in isolated
            ),
            "request_counts_by_observation": request_rows,
            "confidence_level": 0.95,
            "reducer_method": "paired_bca_mean_log_ratio_v1",
        }
        path = cls.diagnostic_path(completion)
        if path.exists():
            if (
                _read_canonical_json(
                    path,
                    label="fresh interference diagnostic",
                )
                != value
            ):
                raise FormalExperimentDagBlocked(
                    "preflight: retained interference diagnostic changed"
                )
        else:
            _publish_no_replace(path, value)
        return cls.diagnostic_binding(completion)

    @staticmethod
    def _native_result(
        wrapper: Any, raw_row: Any, *, current_ns: int
    ) -> tuple[Any, Any]:
        from lightcone_spec.orchestration.formal_terminal_result import (
            FormalCurrentPreflightTp1TerminalResultProofArtifact,
            FormalPreflightTp1TerminalResultProofArtifact,
            FormalSingleOperatorPreflightTp1RawTerminalProofArtifact,
        )
        from lightcone_spec.orchestration.native_terminal import (
            NO_TRUSTED_ATTESTERS,
            NativeTerminalResultProofArtifact,
            validate_native_terminal_artifact,
        )
        from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

        if type(wrapper) in {
            FormalCurrentPreflightTp1TerminalResultProofArtifact,
            FormalPreflightTp1TerminalResultProofArtifact,
        }:
            proof_binding = CanonicalJsonProofBinding.bind(
                wrapper.native_result_proof.absolute_path
            )
            proof = NativeTerminalResultProofArtifact.from_dict(proof_binding.reopen())
            result = proof.revalidate(now_ns=current_ns)
            if result.to_dict() != wrapper.result:
                raise ValueError("preflight interference native projection changed")
            return result, proof_binding
        if type(wrapper) is FormalSingleOperatorPreflightTp1RawTerminalProofArtifact:
            raw_binding = CanonicalJsonProofBinding.bind(
                wrapper.raw_terminal.absolute_path
            )
            result = validate_native_terminal_artifact(
                raw_binding.reopen(),
                trusted_attester_policy=NO_TRUSTED_ATTESTERS,
                expected_binding=raw_row.run_binding,
            )
            if result.authority_kind != "untrusted_raw_terminal":
                raise ValueError("trusted preflight raw terminal authority changed")
            return result, raw_binding
        raise ValueError("preflight interference terminal wrapper kind differs")

    @staticmethod
    def _wrapper(value: object) -> Any:
        from lightcone_spec.orchestration.formal_terminal_result import (
            FormalCurrentPreflightTp1TerminalResultProofArtifact,
            FormalPreflightTp1TerminalResultProofArtifact,
            FormalSingleOperatorPreflightTp1RawTerminalProofArtifact,
        )

        if type(value) is not dict:
            raise TypeError("preflight interference terminal is not an object")
        kind = value.get("kind")
        codecs = {
            "formal_current_preflight_tp1_terminal_result_proof_artifact": (
                FormalCurrentPreflightTp1TerminalResultProofArtifact
            ),
            "formal_preflight_tp1_terminal_result_proof_artifact": (
                FormalPreflightTp1TerminalResultProofArtifact
            ),
            "formal_single_operator_preflight_tp1_raw_terminal_proof_artifact": (
                FormalSingleOperatorPreflightTp1RawTerminalProofArtifact
            ),
        }
        try:
            codec = codecs[kind]
        except KeyError as error:
            raise ValueError(
                "preflight interference terminal wrapper is unsupported"
            ) from error
        return codec.from_dict(value)

    @staticmethod
    def _topology(inventory: Any, gpu_uuid: str) -> tuple[str, str]:
        from lightcone_spec.experiments.formal_protocol import content_sha256

        device = inventory.device(gpu_uuid)
        devices = tuple(sorted(inventory.devices, key=lambda row: row.uuid))
        hardware = {row.hardware_envelope_sha256 for row in devices}
        if (
            len(devices) != 2
            or len(hardware) != 1
            or any(row.host_id != device.host_id for row in devices)
        ):
            raise ValueError(
                "fresh interference gate requires one homogeneous dual-card inventory"
            )
        topology = content_sha256(
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
        )
        return topology, device.hardware_envelope_sha256

    def resolve(
        self,
        *,
        completion: ControllerArtifactBinding,
        actual_result_paths: Mapping[str, str],
        gpu_uuids: tuple[str, ...],
    ) -> InterferenceEnvelope:
        from lightcone_spec.experiments.formal_preflight_inputs import (
            FormalPreflightExecutionInputs,
            FormalSingleOperatorPreflightAuthority,
            revalidate_formal_single_operator_preflight_completion,
        )
        from lightcone_spec.experiments.gpu_pool import GpuInventory
        from lightcone_spec.experiments.interference_authority import (
            InterferenceRawObservation,
        )
        from lightcone_spec.experiments.itl_authority import ItlRequestTimestamps
        from lightcone_spec.experiments.preflight_interference import (
            FormalPreflightInterferenceProofRow,
            FormalPreflightInterferenceQualificationLock,
            FormalPreflightInterferenceRawBatch,
            _derive_observation,
            _diagnose,
            _request_contract_sha256,
        )
        from lightcone_spec.experiments.registry import build_industrial_registry
        from lightcone_spec.orchestration.native_terminal import (
            validate_unsigned_native_itl_pointer_bundle,
        )
        from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

        actual_paths = tuple(sorted(set(actual_result_paths.values())))
        if len(actual_paths) != 1:
            raise ValueError("preflight gate requires one shared exact-ten completion")
        current_ns = time.time_ns()
        exact_ten = revalidate_formal_single_operator_preflight_completion(
            actual_paths[0], current_ns=current_ns
        )
        if exact_ten.status != "COMPLETE" or len(exact_ten.interference_evidence) != 8:
            raise FormalExperimentDagBlocked(
                "preflight: exact-eight interference evidence is incomplete"
            )
        inputs = FormalPreflightExecutionInputs.from_dict(
            exact_ten.execution_inputs.reopen()
        )
        authority = FormalSingleOperatorPreflightAuthority.from_dict(
            inputs.execution_authority.reopen()
        )
        inventory = GpuInventory.from_dict(authority.inventory.reopen())
        observed_uuids = tuple(
            sorted(row.uuid for row in inventory.devices if row.ready)
        )
        if observed_uuids != gpu_uuids:
            raise ValueError("fresh interference inventory differs from scheduler GPUs")
        registry = build_industrial_registry()
        if registry.sha256 != authority.registry_sha256:
            raise ValueError("fresh interference registry identity differs")

        @dataclass(frozen=True)
        class _Timing:
            requests: tuple[Any, ...]

            @property
            def p99_itl_input_ns(self) -> tuple[int, ...]:
                values = tuple(
                    value for row in self.requests for value in row.inter_token_ns
                )
                if not values:
                    raise ValueError("fresh interference ITL input is empty")
                return values

        proof_rows = []
        for evidence in exact_ten.interference_evidence:
            wrapper_binding = CanonicalJsonProofBinding.bind(
                evidence.terminal_result_proof.absolute_path
            )
            wrapper = self._wrapper(wrapper_binding.reopen())
            raw_batch = FormalPreflightInterferenceRawBatch.from_dict(
                wrapper.interference_raw_batch.reopen()
            )
            raw_batch.revalidate()
            matches = tuple(
                row
                for row in raw_batch.rows
                if row.registry_cell_id == evidence.registry_cell_id
                and row.materialized_cell_id == evidence.materialized_cell_id
            )
            if len(matches) != 1:
                raise ValueError("fresh interference raw row coverage differs")
            raw_row = matches[0]
            raw_row.deep_revalidate_unsigned(nvidia_smi_tool=raw_batch.nvidia_smi_tool)
            result, terminal_binding = self._native_result(
                wrapper, raw_row, current_ns=current_ns
            )
            if (
                raw_row.native_itl_pointer_artifact is None
                or raw_row.qualification_lock is None
            ):
                raise FormalExperimentDagBlocked(
                    "preflight: native ITL/qualification evidence is unavailable"
                )
            terminal_outputs = {
                row.request_id: row.output_token_ids
                for row in result.requests
                if row.submitted_to_server
                and row.terminal_status == "completed"
                and row.output_token_ids is not None
            }
            ordered_outputs = {
                request_id: terminal_outputs[request_id]
                for request_id in raw_row.run_binding.scored_request_ids
                if request_id in terminal_outputs
            }
            qualification = FormalPreflightInterferenceQualificationLock.from_dict(
                raw_row.qualification_lock.reopen()
            )
            bundle = validate_unsigned_native_itl_pointer_bundle(
                raw_row.native_itl_pointer_artifact,
                expected_binding=raw_row.run_binding,
                expected_terminal_artifact=raw_row.raw_terminal,
                expected_scored_request_inputs_sha256=(
                    qualification.scored_request_inputs_sha256
                ),
                expected_terminal_output_tokens=ordered_outputs,
            )
            timing = _Timing(
                tuple(
                    ItlRequestTimestamps(
                        request_id=row.request_id,
                        request_started_ns=row.request_started_ns,
                        request_terminal_ns=row.request_terminal_ns,
                        output_token_ids=tuple(event.token_id for event in row.events),
                        token_observed_ns=tuple(
                            event.observed_ns for event in row.events
                        ),
                    )
                    for row in bundle.pointers
                )
            )
            observation, slo_sha, slo_status, qualified = _derive_observation(
                row=raw_row,
                terminal_authority_sha256=terminal_binding.semantic_sha256,
                result=result,
                itl=timing,
            )
            if type(observation) is not InterferenceRawObservation:
                raise TypeError("fresh interference observation type differs")
            topology_sha, hardware_sha = self._topology(inventory, raw_row.gpu_uuid)
            proof_rows.append(
                FormalPreflightInterferenceProofRow(
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
                    topology_sha256=topology_sha,
                    hardware_envelope_sha256=hardware_sha,
                    native_result_proof=terminal_binding,
                    native_itl_proof=raw_row.native_itl_pointer_artifact,
                    slo_accounting_sha256=slo_sha,
                    slo_status=slo_status,
                    qualified_request_ids=qualified,
                    observation=observation,
                )
            )
        rows = tuple(sorted(proof_rows, key=lambda row: row.sha256))
        diagnostic = _diagnose(
            rows,
            registry=registry,
            inventory_sha256=inventory.sha256,
            hardware_envelope_sha256=rows[0].hardware_envelope_sha256,
        )
        binding = self._publish_diagnostic(
            completion=completion,
            exact_ten_completion_path=actual_paths[0],
            gpu_uuids=gpu_uuids,
            proof_rows=rows,
            diagnostic=diagnostic,
        )
        mode = "DUAL_SINGLE" if diagnostic.status == "PASS" else "ISOLATED"
        return InterferenceEnvelope(mode, gpu_uuids, binding.sha256)


@dataclass(frozen=True)
class FormalDagDriverCycle:
    controller: DagControllerStep
    scheduler: SchedulerCycleResult | None

    @property
    def run_state(self) -> str:
        if self.controller.action == "COMPLETE":
            return "DAG_REDUCED_AWAITING_FINAL_AUDIT"
        if self.controller.action == "BLOCKED":
            return "BLOCKED"
        if self.scheduler is not None and self.scheduler.dispatch_state == "STOP":
            return "STOPPED"
        return "DAG_ACTIVE"

    @property
    def changed(self) -> bool:
        # A BLOCKED transition is itself a durable scientific result and must
        # flush the updated stage plan, selection journal, and metrics to the
        # progress exports before run_forever exits.
        if self.controller.action == "BLOCKED":
            return True
        if self.controller.action not in {"WAITING", "COMPLETE"}:
            return True
        return self.scheduler is not None and bool(
            self.scheduler.reconciled or self.scheduler.dispatched
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_state": self.run_state,
            "controller": asdict(self.controller),
            "scheduler": (None if self.scheduler is None else asdict(self.scheduler)),
        }


class FormalSingleOperatorDagDriver:
    """Own one controller/scheduler composition under one shared flock."""

    def __init__(
        self,
        *,
        store: ExperimentOperatorStore,
        callbacks: DagControllerCallbacks,
        scheduler: FormalExperimentSchedulerDaemon | None,
        lock_path: str | Path,
        progress_root: str | Path | None = None,
    ) -> None:
        if type(store) is not ExperimentOperatorStore:
            raise TypeError("formal DAG driver requires an exact operator store")
        if type(callbacks) is not DagControllerCallbacks:
            raise TypeError("formal DAG driver requires exact controller callbacks")
        if (
            scheduler is not None
            and type(scheduler) is not FormalExperimentSchedulerDaemon
        ):
            raise TypeError("formal DAG driver scheduler type differs")
        self.store = store
        self.controller = FormalExperimentDagController(
            store=store,
            callbacks=callbacks,
        )
        self.scheduler = scheduler
        self.lock_path = _absolute_path(lock_path, "formal DAG driver lock")
        self.progress_root = (
            None
            if progress_root is None
            else _absolute_path(progress_root, "formal DAG progress root")
        )

    def close(self) -> None:
        """Close the durable operator store owned by this driver."""

        self.store.close()

    def _cycle_unlocked(self) -> FormalDagDriverCycle:
        step = self.controller.run_once()
        scheduler_result = None
        if step.action == "WAITING" and self.scheduler is not None:
            active = next(
                (
                    row
                    for row in self.store.controller_nodes()
                    if row["state"] != "REDUCED"
                ),
                None,
            )
            if active is not None and active["state"] == "PLANNED":
                scheduler_result = self.scheduler.run_once()
        if scheduler_result is None:
            dispatch_state, stop_reason = self.store.dispatch_control()
            if dispatch_state == "STOP":
                scheduler_result = SchedulerCycleResult(
                    (),
                    (),
                    dispatch_state,
                    stop_reason,
                )
        cycle = FormalDagDriverCycle(step, scheduler_result)
        if cycle.changed and self.progress_root is not None:
            self.store.export_progress(self.progress_root)
        return cycle

    def run_once(self) -> FormalDagDriverCycle:
        with SingletonOperatorLock(self.lock_path):
            return self._cycle_unlocked()

    def resume_node(self, *, node: str, reason: str) -> None:
        """Explicitly resume one durable BLOCKED node under the singleton lock."""

        with SingletonOperatorLock(self.lock_path):
            self.store.resume_controller_node(node=node, reason=reason)

    def _dispatch_resume_evidence(
        self,
        *,
        manual_evidence_path: str | Path | None,
    ) -> Mapping[str, Any] | None:
        running_commands = self.store.queued_commands(status="RUNNING")
        if not running_commands:
            running_auxiliary = tuple(
                row
                for row in self.store.controller_auxiliary_groups()
                if row["status"] == "RUNNING"
            )
            if not running_auxiliary:
                return None
            if len(running_auxiliary) != 1:
                raise ExperimentOperatorError(
                    "dispatch recovery requires exact one RUNNING auxiliary group"
                )
            if manual_evidence_path is not None:
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery requires a fresh child heartbeat"
                )
            if self.scheduler is None:
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery requires the production scheduler"
                )
            state, stop_reason = self.store.dispatch_control()
            if state != "STOP" or stop_reason != "auxiliary_child_heartbeat_stale":
                raise ExperimentOperatorError(
                    "RUNNING auxiliary recovery is limited to child-heartbeat STOP"
                )
            from lightcone_spec.orchestration.experiment_operator_production import (
                revalidate_child_start_receipt,
            )

            group = running_auxiliary[0]
            pid, pgid = group["pid"], group["pgid"]
            start_sha256 = group["process_start_receipt_sha256"]
            if type(pid) is not int or type(pgid) is not int:
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery lacks PID/PGID"
                )
            _require_sha256(start_sha256, "auxiliary dispatch recovery receipt")
            output = Path(str(group["output_directory"]))
            recovered = revalidate_child_start_receipt(
                output / "auxiliary-worker.start.json",
                command_sha256=str(group["launch_command_sha256"]),
            )
            if (
                recovered.pid,
                recovered.pgid,
                recovered.receipt_sha256,
            ) != (pid, pgid, start_sha256):
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery start identity differs"
                )
            observation = self.scheduler.callbacks.process_probe(pid, pgid)
            if (
                type(observation) is not ProcessObservation
                or not observation.alive
                or observation.pid != pid
                or observation.observed_pgid != pgid
            ):
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery process is not live"
                )
            descriptor = DirectoryAuxiliaryPhysicalRuntime._load_descriptor(
                DirectoryAuxiliaryPhysicalRuntime._descriptor_path(output)
            )
            if (
                descriptor.group_id != group["group_id"]
                or descriptor.attempt != group["attempt"]
            ):
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery descriptor identity differs"
                )
            heartbeat_value = _read_canonical_json(
                descriptor.heartbeat_output_path,
                label="auxiliary dispatch recovery heartbeat",
            )
            expected_heartbeat_fields = {
                "schema_version",
                "kind",
                "cell_id",
                "attempt",
                "command_sha256",
                "worker_pid",
                "sequence",
                "observed_at_ns",
                "phase",
            }
            if (
                set(heartbeat_value) != expected_heartbeat_fields
                or heartbeat_value.get("schema_version") != 1
                or heartbeat_value.get("kind") != "formal_experiment_child_heartbeat"
                or heartbeat_value.get("cell_id") != group["group_id"]
                or heartbeat_value.get("attempt") != group["attempt"]
                or heartbeat_value.get("command_sha256")
                != group["launch_command_sha256"]
            ):
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery heartbeat identity differs"
                )
            try:
                heartbeat = WorkerHeartbeat(
                    command_sha256=heartbeat_value["command_sha256"],
                    worker_pid=heartbeat_value["worker_pid"],
                    sequence=heartbeat_value["sequence"],
                    observed_at_ns=heartbeat_value["observed_at_ns"],
                    phase=heartbeat_value["phase"],
                )
            except (TypeError, ValueError) as error:
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery heartbeat values differ"
                ) from error
            now_ns = time.time_ns()
            if (
                heartbeat.sequence < int(group["heartbeat_sequence"])
                or heartbeat.observed_at_ns > now_ns
                or now_ns - heartbeat.observed_at_ns
                > int(self.scheduler.watchdog_policy.heartbeat_timeout_seconds * 1e9)
            ):
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery requires a fresh child heartbeat"
                )
            worker = self.scheduler.callbacks.process_probe(
                heartbeat.worker_pid,
                pgid,
            )
            if (
                type(worker) is not ProcessObservation
                or not worker.alive
                or worker.pid != heartbeat.worker_pid
                or worker.observed_pgid != pgid
            ):
                raise ExperimentOperatorError(
                    "auxiliary dispatch recovery heartbeat worker is not live"
                )
            return {
                "schema_version": 1,
                "kind": "formal_experiment_auxiliary_dispatch_running_recovery",
                "mode": "FRESH_CHILD_HEARTBEAT",
                "stop_reason": stop_reason,
                "verified_at_ns": now_ns,
                "group": {
                    "group_id": group["group_id"],
                    "attempt": group["attempt"],
                    "node": group["node"],
                    "command_sha256": group["launch_command_sha256"],
                    "pid": pid,
                    "pgid": pgid,
                    "process_start_receipt_sha256": start_sha256,
                    "heartbeat": {
                        "command_sha256": heartbeat.command_sha256,
                        "worker_pid": heartbeat.worker_pid,
                        "sequence": heartbeat.sequence,
                        "observed_at_ns": heartbeat.observed_at_ns,
                        "phase": heartbeat.phase,
                    },
                },
                "manual_evidence": None,
            }
        if self.scheduler is None:
            raise ExperimentOperatorError(
                "RUNNING dispatch recovery requires the production scheduler"
            )
        state, stop_reason = self.store.dispatch_control()
        if state != "STOP" or stop_reason != "child_heartbeat_stale":
            raise ExperimentOperatorError(
                "RUNNING dispatch recovery is limited to child-heartbeat STOP"
            )
        callbacks = self.scheduler.callbacks
        if (
            callbacks.recover_started_process is None
            or callbacks.worker_heartbeat is None
            or callbacks.worker_heartbeat_required is None
        ):
            raise ExperimentOperatorError(
                "RUNNING dispatch recovery callbacks are unavailable"
            )
        logical_identities = {
            (command.cell_id, command.attempt) for command in running_commands
        }
        covered_identities: set[tuple[str, int]] = set()
        now_ns = time.time_ns()
        process_rows: list[dict[str, Any]] = []
        heartbeat_rows: list[dict[str, Any]] = []
        fresh_heartbeats = True
        for command in self.store.physical_commands(status="RUNNING"):
            attempt = self.store.attempt(command.cell_id, command.attempt)
            pid, pgid = attempt["pid"], attempt["pgid"]
            start_sha256 = attempt["process_start_receipt_sha256"]
            if type(pid) is not int or type(pgid) is not int:
                raise ExperimentOperatorError(
                    "RUNNING dispatch recovery lacks PID/PGID"
                )
            _require_sha256(start_sha256, "dispatch recovery start receipt")
            recovered = callbacks.recover_started_process(command)
            if recovered is None or (
                recovered.pid,
                recovered.pgid,
                recovered.receipt_sha256,
            ) != (pid, pgid, start_sha256):
                raise ExperimentOperatorError(
                    "RUNNING dispatch recovery start identity differs"
                )
            if recovered.started_ns < int(attempt["started_at_ns"]):
                raise ExperimentOperatorError(
                    "RUNNING dispatch recovery receipt predates ledger start"
                )
            observation = callbacks.process_probe(pid, pgid)
            if (
                type(observation) is not ProcessObservation
                or not observation.alive
                or observation.pid != pid
                or observation.observed_pgid != pgid
            ):
                raise ExperimentOperatorError(
                    "RUNNING dispatch recovery process identity is not live"
                )
            group = self.store.physical_attempt_group_for_attempt(
                command.cell_id,
                command.attempt,
            )
            covered = (
                (command,)
                if group is None
                else self.store.physical_attempt_group_commands(str(group["group_id"]))
            )
            covered_attempts = [
                {"cell_id": row.cell_id, "attempt": row.attempt} for row in covered
            ]
            covered_identities.update((row.cell_id, row.attempt) for row in covered)
            process_rows.append(
                {
                    "cell_id": command.cell_id,
                    "attempt": command.attempt,
                    "command_sha256": command.command_sha256,
                    "pid": pid,
                    "pgid": pgid,
                    "process_start_receipt_sha256": start_sha256,
                    "covered_attempts": covered_attempts,
                }
            )
            if not callbacks.worker_heartbeat_required(command):
                raise ExperimentOperatorError(
                    "heartbeat STOP references a command without child heartbeat"
                )
            heartbeat = callbacks.worker_heartbeat(command)
            if heartbeat is None:
                fresh_heartbeats = False
                continue
            if (
                type(heartbeat) is not WorkerHeartbeat
                or heartbeat.command_sha256 != command.command_sha256
                or heartbeat.observed_at_ns > now_ns
                or heartbeat.observed_at_ns < int(attempt["started_at_ns"])
                or heartbeat.sequence < int(attempt["heartbeat_sequence"])
            ):
                raise ExperimentOperatorError(
                    "RUNNING dispatch recovery heartbeat identity differs"
                )
            worker = callbacks.process_probe(heartbeat.worker_pid, pgid)
            if (
                type(worker) is not ProcessObservation
                or not worker.alive
                or worker.pid != heartbeat.worker_pid
                or worker.observed_pgid != pgid
            ):
                raise ExperimentOperatorError(
                    "RUNNING dispatch recovery heartbeat worker is not live"
                )
            fresh_heartbeats = fresh_heartbeats and (
                now_ns - heartbeat.observed_at_ns
                <= int(self.scheduler.watchdog_policy.heartbeat_timeout_seconds * 1e9)
            )
            heartbeat_rows.append(
                {
                    "cell_id": command.cell_id,
                    "attempt": command.attempt,
                    "worker_pid": heartbeat.worker_pid,
                    "sequence": heartbeat.sequence,
                    "observed_at_ns": heartbeat.observed_at_ns,
                    "phase": heartbeat.phase,
                }
            )
        if covered_identities != logical_identities:
            raise ExperimentOperatorError(
                "RUNNING dispatch recovery physical coverage differs"
            )
        process_rows.sort(key=lambda row: (row["cell_id"], row["attempt"]))
        heartbeat_rows.sort(key=lambda row: (row["cell_id"], row["attempt"]))
        manual_binding = None
        mode = "FRESH_CHILD_HEARTBEAT"
        if manual_evidence_path is None:
            if not fresh_heartbeats or len(heartbeat_rows) != len(process_rows):
                raise ExperimentOperatorError(
                    "RUNNING dispatch recovery requires fresh child heartbeats"
                )
        else:
            manual_path = _existing_file(
                manual_evidence_path,
                "manual dispatch resume evidence",
            )
            manual = _read_canonical_json(
                manual_path,
                label="manual dispatch resume evidence",
            )
            expected_fields = {
                "schema_version",
                "kind",
                "run_id",
                "dispatch_stop_reason",
                "observed_at_ns",
                "operator_observation",
                "running_processes",
            }
            observed_at_ns = manual.get("observed_at_ns")
            operator_observation = manual.get("operator_observation")
            if (
                set(manual) != expected_fields
                or manual.get("schema_version") != 1
                or manual.get("kind")
                != "formal_experiment_manual_dispatch_resume_evidence"
                or manual.get("run_id") != self.store.run_id
                or manual.get("dispatch_stop_reason") != stop_reason
                or type(observed_at_ns) is not int
                or observed_at_ns < 1
                or observed_at_ns > now_ns
                or now_ns - observed_at_ns
                > int(self.scheduler.watchdog_policy.heartbeat_timeout_seconds * 1e9)
                or type(operator_observation) is not str
                or not operator_observation
                or operator_observation.strip() != operator_observation
                or "\x00" in operator_observation
                or manual.get("running_processes") != process_rows
            ):
                raise ExperimentOperatorError(
                    "manual dispatch resume evidence differs from live recovery"
                )
            manual_binding = asdict(ControllerArtifactBinding.bind(manual_path))
            mode = "MANUAL_OPERATOR_EVIDENCE"
        return {
            "schema_version": 1,
            "kind": "formal_experiment_dispatch_running_recovery",
            "mode": mode,
            "stop_reason": stop_reason,
            "verified_at_ns": now_ns,
            "processes": process_rows,
            "heartbeat_observations": heartbeat_rows,
            "manual_evidence": manual_binding,
        }

    def resume_dispatch(
        self,
        *,
        reason: str,
        manual_evidence_path: str | Path | None = None,
    ) -> None:
        """Explicitly clear scheduler STOP without advancing the DAG."""

        with SingletonOperatorLock(self.lock_path):
            state, _ = self.store.dispatch_control()
            if state != "STOP":
                raise ExperimentOperatorError("scheduler dispatch is not STOPPED")
            recovery = self._dispatch_resume_evidence(
                manual_evidence_path=manual_evidence_path,
            )
            self.store.clear_dispatch_stop(
                reason=reason,
                running_recovery_evidence=recovery,
            )

    def run_forever(self) -> None:
        """Run at the fixed watchdog cadence and log only state changes."""

        last: str | None = None
        lock = SingletonOperatorLock(self.lock_path)
        lock.acquire()
        try:
            while True:
                cycle = self._cycle_unlocked()
                encoded = json.dumps(
                    cycle.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if cycle.changed or encoded != last:
                    print(encoded, flush=True)
                    last = encoded
                if cycle.controller.action in {"BLOCKED", "COMPLETE"}:
                    return
                if (
                    cycle.scheduler is not None
                    and cycle.scheduler.dispatch_state == "STOP"
                    and self.store.running_termination_count() == 0
                ):
                    return
                time.sleep(30.0)
        finally:
            lock.release()


class ProductionFormalDagCallbackBuilder:
    """Build current materialize/plan/actual/reduce callbacks from paths."""

    def __init__(
        self,
        *,
        config: PathBoundFormalDagDriverConfig,
        store: ExperimentOperatorStore,
        prerequisite_resolver: PrerequisiteIndexResolver | None = None,
        e5_arrival_resolver: E5ArrivalPlanResolver | None = None,
        auxiliary_runtime: AuxiliaryPhysicalRuntime | None = None,
        interference_gate_resolver: InterferenceGateResolver | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        python_executable: str | Path = sys.executable,
    ) -> None:
        if type(config) is not PathBoundFormalDagDriverConfig:
            raise TypeError("production callback builder requires exact config")
        if type(store) is not ExperimentOperatorStore:
            raise TypeError("production callback builder requires exact store")
        executable = _existing_file(
            Path(python_executable).resolve(strict=True),
            "Python executable",
        )
        self.config = config
        self.store = store
        self.prerequisite_resolver = prerequisite_resolver or (
            DirectoryPrerequisiteIndexResolver(
                config.prerequisite_index_catalog_directory
            )
        )
        self.e5_arrival_resolver = (
            e5_arrival_resolver or PathBoundE5ArrivalPlanResolver(config)
        )
        self.auxiliary_runtime = auxiliary_runtime
        self.interference_gate_resolver = (
            interference_gate_resolver or FreshPreflightInterferenceGateResolver()
        )
        self.clock_ns = clock_ns
        self.python_executable = str(executable)
        self.run_root = Path(config.run_root)
        self.nodes_root = self.run_root / "formal-dag-nodes"
        self.nodes_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._validate_identity_inputs()

    def callbacks(self) -> DagControllerCallbacks:
        return DagControllerCallbacks(
            materialize=self.materialize,
            materialize_with_auxiliary=self.materialize_with_auxiliary,
            plan=self.plan,
            actual_results=self.actual_results,
            reduce=self.reduce,
            auxiliary_plan=self.auxiliary_plan,
            auxiliary_launch=self.auxiliary_launch,
            auxiliary_terminal=self.auxiliary_terminal,
            auxiliary_adoptions=self.auxiliary_adoptions,
        )

    def _validate_identity_inputs(self) -> None:
        from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
        from lightcone_spec.experiments.formal_single_operator_content import (
            load_trusted_single_operator_content_bundle,
        )

        protocol_lock = protocol_lock_from_dict(
            _read_canonical_json(
                self.config.protocol_lock.absolute_path,
                label="ProtocolLock",
            )
        )
        content = load_trusted_single_operator_content_bundle(
            self.config.content_source.absolute_path
        )
        if (
            protocol_lock.content_source_mode != "trusted_single_operator"
            or protocol_lock.trusted_single_operator_content_bundle_sha256
            != content.semantic_sha256
            or content.runtime_binding_status != "BOUND"
            or content.runtime_observations is None
            or content.source_snapshot.repository_root != self.config.repository_root
        ):
            raise ValueError("formal DAG driver identity inputs differ")

    def _node_root(self, node: str) -> Path:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            formal_single_operator_node_spec,
        )

        spec = formal_single_operator_node_spec(node)
        path = self.nodes_root / f"{spec.ordinal:02d}-{spec.node}"
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        return path

    def _retained_manifest_path(self, node: str) -> Path:
        return (
            self._node_root(node)
            / "reduction"
            / "retained-future-dependency-manifest.json"
        )

    def _require_predecessor_archive_boundary(self, node: str) -> None:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            FORMAL_SINGLE_OPERATOR_NODE_ORDER,
        )

        ordinal = FORMAL_SINGLE_OPERATOR_NODE_ORDER.index(node)
        if ordinal == 0:
            return
        predecessor = FORMAL_SINGLE_OPERATOR_NODE_ORDER[ordinal - 1]
        path = self._retained_manifest_path(predecessor)
        if not path.exists():
            raise FormalExperimentDagBlocked(
                f"{node}: predecessor archive-safe reduction boundary is unavailable"
            )
        manifest = load_retained_future_dependency_manifest(path)
        row = self.store.controller_node(predecessor)
        if (
            manifest.node != predecessor
            or manifest.completion.absolute_path != row["completion_path"]
            or manifest.completion.sha256 != row["completion_sha256"]
            or ControllerArtifactBinding.bind(manifest.completion.absolute_path)
            != manifest.completion
        ):
            raise FormalExperimentDagBlocked(
                f"{node}: predecessor retained dependency boundary changed"
            )

    def _clock(self, root: Path, name: str) -> int:
        path = root / name
        if path.exists():
            raw = _read_canonical_json(path, label="driver clock marker")
            if set(raw) != {"current_ns"} or type(raw["current_ns"]) is not int:
                raise FormalExperimentDagBlocked("driver clock marker differs")
            return int(raw["current_ns"])
        value = self.clock_ns()
        if type(value) is not int or value < 1:
            raise ValueError("driver clock must return positive nanoseconds")
        _publish_no_replace(path, {"current_ns": value})
        return value

    def _inherited_auxiliary_paths(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
    ) -> dict[str, str]:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            formal_single_operator_required_auxiliary_source_kinds,
            rebuild_formal_single_operator_stage_completion,
        )

        required = formal_single_operator_required_auxiliary_source_kinds(node)
        if not required:
            return {}
        if predecessor is None:
            raise FormalExperimentDagBlocked(f"{node}: auxiliary predecessor missing")
        rebuilt = rebuild_formal_single_operator_stage_completion(
            predecessor.absolute_path
        )
        values = {
            row.source_kind: row.source.absolute_path
            for row in rebuilt.node_materialization.auxiliary_sources
        }
        if set(values) != set(required):
            raise FormalExperimentDagBlocked(
                f"{node}: inherited auxiliary source is unavailable"
            )
        return values

    def materialize(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
    ) -> DagMaterialization:
        return self._materialize(node, predecessor, {})

    def materialize_with_auxiliary(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
        auxiliary_sources: Mapping[str, ControllerArtifactBinding],
    ) -> DagMaterialization:
        return self._materialize(node, predecessor, auxiliary_sources)

    def _materialize(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
        auxiliary_sources: Mapping[str, ControllerArtifactBinding],
    ) -> DagMaterialization:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            FormalSingleOperatorStageBlocked,
            materialize_formal_single_operator_node,
            rebuild_formal_single_operator_node_materialization,
        )

        root = self._node_root(node) / "materialization"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        materialization_path = root / "materialization.json"
        node_path = root / "node-materialization.json"
        if materialization_path.exists() != node_path.exists():
            _preserve_partial_directory(
                root,
                label=f"{node} partial materialization",
            )
            root.mkdir(mode=0o700)
            materialization_path = root / "materialization.json"
            node_path = root / "node-materialization.json"
        supplied = {
            kind: binding.absolute_path
            for kind, binding in sorted(auxiliary_sources.items())
        }
        if not supplied:
            supplied = self._inherited_auxiliary_paths(node, predecessor)
        if not node_path.exists():
            try:
                materialize_formal_single_operator_node(
                    node=node,
                    predecessor_completion_path=(
                        None if predecessor is None else predecessor.absolute_path
                    ),
                    protocol_lock_path=(
                        self.config.protocol_lock.absolute_path
                        if predecessor is None
                        else None
                    ),
                    content_source_path=(
                        self.config.content_source.absolute_path
                        if predecessor is None
                        else None
                    ),
                    materialization_output_path=materialization_path,
                    node_materialization_output_path=node_path,
                    created_ns=self._clock(root, "materialization-clock.json"),
                    auxiliary_source_paths=supplied,
                )
            except FormalSingleOperatorStageBlocked as error:
                # The predecessor's negative/power decision was already
                # reduced, journaled, and bound before this transition.  Turn
                # its typed scientific stop into the controller's durable
                # BLOCKED state instead of crashing or materializing farther.
                raise FormalExperimentDagBlocked(
                    f"{node}: scientific stage blocked: {error}"
                ) from error
        rebuilt = rebuild_formal_single_operator_node_materialization(node_path)
        expected_auxiliary = {
            row.source_kind: row.source.absolute_path
            for row in rebuilt.artifact.auxiliary_sources
        }
        if rebuilt.artifact.node != node or expected_auxiliary != supplied:
            raise ExperimentOperatorError(
                "rebuilt materialization differs from driver inputs"
            )
        if predecessor is not None and (
            rebuilt.artifact.predecessor_source is None
            or rebuilt.artifact.predecessor_source.absolute_path
            != predecessor.absolute_path
        ):
            raise ExperimentOperatorError(
                "rebuilt materialization differs from predecessor"
            )
        bound_auxiliary = tuple(
            sorted(
                (
                    kind,
                    ControllerArtifactBinding.bind(binding.absolute_path),
                )
                for kind, binding in auxiliary_sources.items()
            )
        )
        return DagMaterialization(
            materialization=ControllerArtifactBinding.bind(materialization_path),
            node_materialization=ControllerArtifactBinding.bind(node_path),
            expected_cell_ids=tuple(
                cell.cell_id for cell in rebuilt.materialization.cells
            ),
            auxiliary_sources=bound_auxiliary,
        )

    def plan(
        self,
        node: str,
        node_materialization: ControllerArtifactBinding,
    ) -> DagExecutionPlan:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            load_formal_single_operator_execution_source,
            publish_formal_single_operator_execution_source,
        )

        self._require_predecessor_archive_boundary(node)
        root = self._node_root(node) / "execution"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        execution_source_path = root / "execution-source.json"
        journal_path = root / "execution-plan-journal.json"
        if journal_path.exists():
            return self._load_plan_journal(
                journal_path,
                expected_node=node,
                expected_node_materialization=node_materialization,
            )
        if not execution_source_path.exists():
            publish_formal_single_operator_execution_source(
                node_materialization_path=node_materialization.absolute_path,
                output_path=execution_source_path,
            )
        source = load_formal_single_operator_execution_source(execution_source_path)
        if (
            source.node != node
            or source.materialization_source.absolute_path
            != self.store.controller_node(node)["materialization_path"]
        ):
            raise ExperimentOperatorError("execution source differs from current node")
        work_root = root / "work"
        if work_root.exists():
            _preserve_partial_directory(
                work_root,
                label=f"{node} incomplete execution-plan work",
            )
        work_root.mkdir(mode=0o700)
        current_ns = self._clock(root, "planning-clock.json")
        if node == "preflight":
            result = self._plan_preflight(
                source_path=str(execution_source_path),
                node_materialization=node_materialization,
                work_root=work_root,
                current_ns=current_ns,
            )
        elif node in _EARLY_NODES or node in _E4_DIRECT_NODES:
            result = self._plan_direct(
                node=node,
                source_path=str(execution_source_path),
                node_materialization=node_materialization,
                work_root=work_root,
            )
        elif node in _PREPARED_NODES:
            result = self._plan_prepared(
                node=node,
                source_path=str(execution_source_path),
                node_materialization=node_materialization,
                work_root=work_root,
                current_ns=current_ns,
            )
        else:
            raise FormalExperimentDagBlocked(
                f"{node}: no source-owned physical plan producer"
            )
        self._publish_plan_journal(journal_path, result)
        return self._load_plan_journal(
            journal_path,
            expected_node=node,
            expected_node_materialization=node_materialization,
        )

    def _preflight_inputs_path(self) -> Path:
        return (
            self._node_root("preflight")
            / "execution"
            / "work"
            / "preflight-inputs"
            / "formal-preflight-execution-inputs.json"
        )

    def _plan_preflight(
        self,
        *,
        source_path: str,
        node_materialization: ControllerArtifactBinding,
        work_root: Path,
        current_ns: int,
    ) -> DagExecutionPlan:
        from lightcone_spec.experiments.formal_preflight_inputs import (
            FormalPreflightExecutionInputs,
            FormalSingleOperatorPreflightAuthority,
            materialize_formal_single_operator_preflight_execution_inputs,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            rebuild_formal_single_operator_node_materialization,
        )
        from lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker import (
            formal_preflight_exact_ten_group_environment,
            publish_formal_preflight_exact_ten_group_worker_spec,
        )

        inputs_root = work_root / "preflight-inputs"
        binding = materialize_formal_single_operator_preflight_execution_inputs(
            execution_source_path=source_path,
            repository_root=self.config.repository_root,
            formal_runtime_authority_manifest_path=(
                self.config.runtime_authority_manifest.absolute_path
            ),
            inventory_path=self.config.inventory.absolute_path,
            content_source_path=self.config.content_source.absolute_path,
            workload_authority_path=(
                self.config.preflight_workload_authority.absolute_path
            ),
            doctor_report_path=self.config.doctor_report.absolute_path,
            private_output_root=inputs_root,
            current_ns=current_ns,
        )
        inputs = FormalPreflightExecutionInputs.from_dict(binding.reopen())
        authority = FormalSingleOperatorPreflightAuthority.from_dict(
            inputs.execution_authority.reopen()
        )
        runner_by_cell = {
            row.materialized_cell_id: row.runner_kind
            for row in authority.execution_bindings
        }
        logical = {
            "first_party_compile": "compile",
            "first_party_exactness": "exactness",
            "first_party_interference": "interference",
        }
        rebuilt = rebuild_formal_single_operator_node_materialization(
            node_materialization.absolute_path
        )
        group_root = work_root / "exact-ten-group"
        group_root.mkdir(mode=0o700)
        group_spec_path = group_root / "group-spec.json"
        parent_argv = (
            self.python_executable,
            "-m",
            "lightcone_spec.cli.main",
            "formal-single-operator",
            "execute-preflight",
            "--execution-inputs",
            binding.absolute_path,
            "--current-ns",
            str(current_ns),
        )
        environment = formal_preflight_exact_ten_group_environment(group_spec_path)
        group_id = _semantic_sha256(
            {
                "kind": "formal_single_operator_preflight_exact_ten_group",
                "execution_inputs_sha256": inputs.sha256,
            }
        )
        members: list[PhysicalAttemptGroupMemberSpec] = []
        for cell in rebuilt.materialization.cells:
            cell_root = work_root / "cells" / cell.cell_id
            cell_root.mkdir(mode=0o700, parents=True)
            command = QueuedCommandSpec(
                cell_id=cell.cell_id,
                attempt=1,
                argv=parent_argv,
                launch_compatibility_key=group_id,
                required_gpu_count=2,
                timing_class="EXCLUSIVE",
                predicted_high_water_bytes=_CELL_SPOOL_HIGH_WATER_BYTES,
                monitored_path=self.config.run_root,
                log_path=str(cell_root / "command.log"),
                expected_terminal_path=str(cell_root / "terminal.json"),
                expected_junit_path=str(cell_root / "junit.xml"),
                expected_raw_log_path=str(cell_root / "raw.json"),
                atomic_pointer_path=str(cell_root / "result-pointer.json"),
                child_exit_receipt_path=str(cell_root / "child-exit.json"),
                environment=environment,
                priority=100,
            )
            attempt = self._attempt_spec(
                node="preflight",
                cell=cell,
                command=command,
                scientific_command_sha256=None,
                output_directory=cell_root,
                execution_source_path=source_path,
            )
            try:
                runner_kind = runner_by_cell[cell.cell_id]
            except KeyError as error:
                raise ValueError(
                    "preflight authority lacks a materialized cell"
                ) from error
            members.append(
                PhysicalAttemptGroupMemberSpec(
                    attempt=attempt,
                    command=command,
                    logical_kind=logical[runner_kind],  # type: ignore[arg-type]
                )
            )
        ordered = tuple(sorted(members, key=lambda row: row.attempt.cell_id))
        publish_formal_preflight_exact_ten_group_worker_spec(
            group_id=group_id,
            members=ordered,
            leader_cell_id=ordered[0].attempt.cell_id,
            output_path=group_spec_path,
        )
        return DagExecutionPlan(
            execution_source=ControllerArtifactBinding.bind(source_path),
            prepared_launch=None,
            launches=(),
            physical_attempt_groups=(
                DagPhysicalAttemptGroup(
                    group_id=group_id,
                    members=ordered,
                    leader_cell_id=ordered[0].attempt.cell_id,
                ),
            ),
        )

    def _session_reset_authority_bindings(self) -> tuple[Any, ...]:
        """Deep-open only empirical authorities from the configured spool."""

        directory = self.config.session_reset_authority_directory
        if directory is None:
            return ()
        from lightcone_spec.experiments.formal_single_operator_session_reset import (
            revalidate_trusted_empirical_tp1_session_reset_authority,
        )

        bindings = []
        for path in sorted(Path(directory).glob("*.json")):
            value = _read_canonical_json(
                path, label="session-reset authority candidate"
            )
            if value.get("kind") != "trusted_empirical_tp1_session_reset_authority":
                continue
            binding, _authority = (
                revalidate_trusted_empirical_tp1_session_reset_authority(path)
            )
            bindings.append(binding)
        return tuple(bindings)

    def _materialize_serving_session_groups(
        self,
        *,
        node: str,
        work_root: Path,
        node_materialization: ControllerArtifactBinding,
        launches: tuple[DagCellLaunch, ...],
        specs: tuple[Any, ...],
    ) -> tuple[tuple[DagCellLaunch, ...], tuple[DagPhysicalAttemptGroup, ...]]:
        """Partition prepared ordinary TP1 rows; no authority means all-fresh."""

        if not specs:
            return launches, ()
        from lightcone_spec.orchestration.formal_serving_session_group import (
            FormalServingSessionGroupSpec,
            partition_formal_serving_session_groups,
        )

        if any(type(row) is not FormalServingSessionGroupSpec for row in specs):
            raise TypeError("prepared session grouping received another spec type")
        plans = partition_formal_serving_session_groups(
            specs,
            reset_authorities=self._session_reset_authority_bindings(),
            max_member_count=32,
            max_estimated_duration_seconds=3600.0,
        )
        shared = tuple(
            row for row in plans if row.execution_mode == "shared_session_tp1"
        )
        if not shared:
            return launches, ()
        by_cell = {row.attempt.cell_id: row for row in launches}
        grouped_cells = {
            member.materialized_cell_id for plan in shared for member in plan.members
        }
        standalone = tuple(
            row for row in launches if row.attempt.cell_id not in grouped_cells
        )
        physical = tuple(
            self._materialize_one_serving_session_group(
                node=node,
                group_root=work_root / "session-groups" / plan.group_id,
                node_materialization=node_materialization,
                plan=plan,
                launches=tuple(
                    by_cell[member.materialized_cell_id] for member in plan.members
                ),
            )
            for plan in shared
        )
        return standalone, physical

    def _materialize_one_serving_session_group(
        self,
        *,
        node: str,
        group_root: Path,
        node_materialization: ControllerArtifactBinding,
        plan: Any,
        launches: tuple[DagCellLaunch, ...],
    ) -> DagPhysicalAttemptGroup:
        """Publish one resident group through its dedicated production bridge."""

        from lightcone_spec.experiments.formal_preflight_inputs import (
            FormalPreflightExecutionInputs,
        )
        from lightcone_spec.experiments.formal_single_operator_session_reset import (
            revalidate_trusted_empirical_tp1_session_reset_authority,
        )
        from lightcone_spec.orchestration.formal_cell_worker import (
            load_formal_cell_worker_spec,
        )
        from lightcone_spec.orchestration.formal_serving_session_group import (
            FormalServingSessionGroupPlan,
        )
        from lightcone_spec.orchestration.formal_serving_session_group_production import (
            build_formal_serving_session_group_production_spec,
            formal_serving_session_group_production_environment,
            formal_serving_session_group_shared_evidence_bound_bytes,
            publish_formal_serving_session_group_production_spec,
        )
        from lightcone_spec.orchestration.formal_serving_session_group_worker import (
            FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
            FormalServingSessionGroupExecutionSpec,
            publish_formal_serving_session_group_execution_spec,
        )
        from lightcone_spec.orchestration.live_sglang import PinnedNvidiaSmiTool
        from lightcone_spec.runtime.preflight_runner import (
            ExactnessPreflightAssignment,
        )
        from lightcone_spec.runtime.proof_artifact import (
            CanonicalJsonProofBinding,
            publish_canonical_json_no_replace,
        )

        if type(plan) is not FormalServingSessionGroupPlan:
            raise TypeError("resident group materializer requires an exact plan")
        if (
            plan.execution_mode != "shared_session_tp1"
            or plan.reset_authority_sha256 is None
            or plan.node != node
            or not 2 <= len(launches) <= 32
            or tuple(row.attempt.cell_id for row in launches)
            != tuple(row.materialized_cell_id for row in plan.members)
            or any(
                (
                    launch.attempt.stage,
                    launch.attempt.phase,
                    launch.attempt.output_directory,
                )
                != (member.stage, member.phase, member.output_directory)
                for launch, member in zip(launches, plan.members, strict=True)
            )
        ):
            raise ValueError("resident group materialization coverage differs")
        if len({row.command.launch_compatibility_key for row in launches}) != 1:
            raise ValueError("resident group launches have different process keys")
        if any(
            row.command.required_gpu_count != 1
            or row.command.timing_class != "HEADLINE"
            or row.attempt.attempt != 1
            for row in launches
        ):
            raise ValueError("resident group received a non-initial TP1 headline")

        authority_matches = []
        for authority_binding in self._session_reset_authority_bindings():
            rebound, authority = (
                revalidate_trusted_empirical_tp1_session_reset_authority(
                    authority_binding.absolute_path
                )
            )
            if rebound != authority_binding:
                raise ValueError("resident reset authority binding changed")
            if authority.sha256 == plan.reset_authority_sha256:
                authority_matches.append(authority_binding)
        if len(authority_matches) != 1:
            raise ValueError("resident group reset authority is not uniquely bound")
        authority_binding = authority_matches[0]

        group_root.mkdir(mode=0o700, parents=True)
        plan_path = group_root / "group-plan.json"
        publish_canonical_json_no_replace(plan_path, plan.to_dict())
        plan_binding = CanonicalJsonProofBinding.bind(plan_path)
        # ``plan.sha256`` excludes its self-declared ``plan_sha256`` field,
        # whereas a canonical file binding hashes the complete JSON object.
        # The deep codec re-computes and verifies that self digest; comparing
        # the two unlike digest domains would reject every valid group.
        if FormalServingSessionGroupPlan.from_dict(plan_binding.reopen()) != plan:
            raise RuntimeError("resident group plan changed after publication")
        execution_spec_path = group_root / "group-execution-spec.json"
        execution_spec = FormalServingSessionGroupExecutionSpec(
            schema_version=1,
            kind="formal_serving_session_group_execution_spec",
            protocol_sha256=(FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256),
            group_plan_path=plan_binding.absolute_path,
            reset_authority_path=authority_binding.absolute_path,
            output_directory=str((group_root / "group-execution").resolve()),
            formal_measured=False,
        )
        publish_formal_serving_session_group_execution_spec(
            spec=execution_spec,
            output_path=execution_spec_path,
        )

        production_spec_path = (group_root / "production-spec.json").resolve()
        environment = formal_serving_session_group_production_environment(
            production_spec_path
        )
        parent_argv = (
            self.python_executable,
            "-m",
            ("lightcone_spec.orchestration.formal_serving_session_group_production"),
            "--spec",
            str(production_spec_path),
        )
        resident_evidence_root = (group_root / "resident-evidence").resolve()
        progress_paths = (
            resident_evidence_root / plan.group_id / "server.log",
            resident_evidence_root / plan.group_id / "server.stdout",
            resident_evidence_root / plan.group_id / "server.stderr",
        )
        environment = tuple(
            sorted(
                (
                    *environment,
                    (
                        "LIGHTCONE_OPERATOR_PROGRESS_LOG_PATHS_JSON",
                        json.dumps(
                            [str(path) for path in progress_paths],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
            )
        )
        inventory_uuids = self._inventory_gpu_uuids()
        if len(plan.assigned_gpu_uuids) != 1:
            raise ValueError("resident group plan is not assigned to one GPU")
        try:
            preferred_gpu_index = inventory_uuids.index(plan.assigned_gpu_uuids[0])
        except ValueError as error:
            raise ValueError(
                "resident group GPU is outside driver inventory"
            ) from error
        shared_runtime_seconds = sum(
            row.command.max_runtime_seconds for row in launches
        )
        shared_log_stall_seconds = min(
            row.command.max_log_stall_seconds for row in launches
        )
        shared_evidence_bound_bytes = (
            formal_serving_session_group_shared_evidence_bound_bytes(len(launches))
        )
        group_high_water_bytes = (
            sum(item.command.predicted_high_water_bytes for item in launches)
            + shared_evidence_bound_bytes
        )
        group_commands = tuple(
            QueuedCommandSpec(
                cell_id=row.command.cell_id,
                attempt=row.command.attempt,
                argv=parent_argv,
                launch_compatibility_key=row.command.launch_compatibility_key,
                required_gpu_count=1,
                timing_class="HEADLINE",
                # Every source cell retains its complete raw/result allowance
                # until shared close.  Add the protocol-bound process-level
                # log/telemetry/receipt allowance; never replace the sum with a
                # single member's bound.
                predicted_high_water_bytes=group_high_water_bytes,
                monitored_path=row.command.monitored_path,
                log_path=row.command.log_path,
                expected_terminal_path=row.command.expected_terminal_path,
                expected_junit_path=row.command.expected_junit_path,
                expected_raw_log_path=row.command.expected_raw_log_path,
                atomic_pointer_path=row.command.atomic_pointer_path,
                child_exit_receipt_path=row.command.child_exit_receipt_path,
                environment=environment,
                paired_gpu_key=f"resident-session:{plan.group_id}",
                preferred_gpu_index=preferred_gpu_index,
                priority=row.command.priority,
                max_runtime_seconds=shared_runtime_seconds,
                max_log_stall_seconds=min(
                    shared_log_stall_seconds,
                    shared_runtime_seconds,
                ),
            )
            for row in launches
        )
        if len({row.command_sha256 for row in group_commands}) != 1:
            raise RuntimeError("resident group physical command identity split")

        worker_spec_paths = []
        for launch in launches:
            argv = launch.command.argv
            if len(argv) < 2 or argv[-2] != "--spec":
                raise ValueError("resident group source command lacks its worker spec")
            worker_path = argv[-1]
            worker_spec, _worker_digest = load_formal_cell_worker_spec(worker_path)
            if (
                worker_spec.cell_id != launch.attempt.cell_id
                or worker_spec.attempt != launch.attempt.attempt
                or worker_spec.node_materialization_path
                != node_materialization.absolute_path
            ):
                raise ValueError(
                    "resident group worker leaves the current node materialization"
                )
            worker_spec_paths.append(worker_path)
        preflight_inputs = FormalPreflightExecutionInputs.from_dict(
            CanonicalJsonProofBinding.bind(self._preflight_inputs_path()).reopen()
        )
        exactness = ExactnessPreflightAssignment.load(
            preflight_inputs.exactness_assignment.absolute_path
        )
        nvidia_smi_tool = PinnedNvidiaSmiTool.bind(exactness.nvidia_smi_executable)
        production_spec = build_formal_serving_session_group_production_spec(
            production_spec_path=production_spec_path,
            group_execution_spec_path=execution_spec_path,
            cell_worker_spec_paths=worker_spec_paths,
            commands=group_commands,
            nvidia_smi_tool=nvidia_smi_tool,
            resident_evidence_root=resident_evidence_root,
            shared_publication_path=(group_root / "shared-publication.json").resolve(),
            server_watch_target_path=(
                group_root / "server-watch-target.json"
            ).resolve(),
        )
        if production_spec.shared_evidence_bound_bytes != shared_evidence_bound_bytes:
            raise RuntimeError("resident group evidence high-water identity changed")
        publish_formal_serving_session_group_production_spec(spec=production_spec)

        members = tuple(
            PhysicalAttemptGroupMemberSpec(
                attempt=CellAttemptSpec(
                    cell_id=launch.attempt.cell_id,
                    attempt=launch.attempt.attempt,
                    stage=launch.attempt.stage,
                    phase=launch.attempt.phase,
                    block=launch.attempt.block,
                    seed=launch.attempt.seed,
                    scientific_axes=launch.attempt.scientific_axes,
                    identity=launch.attempt.identity,
                    command_sha256=command.command_sha256,
                    scientific_command_sha256=(
                        launch.attempt.scientific_command_sha256
                    ),
                    output_directory=launch.attempt.output_directory,
                ),
                command=command,
                logical_kind="serving",
            )
            for launch, command in zip(launches, group_commands, strict=True)
        )
        if any(
            member.attempt.scientific_command_sha256
            != source.attempt.scientific_command_sha256
            for member, source in zip(members, launches, strict=True)
        ):
            raise RuntimeError("resident group changed a scientific command")
        return DagPhysicalAttemptGroup(
            group_id=plan.group_id,
            members=members,
            leader_cell_id=members[0].attempt.cell_id,
            group_kind="tp1_serving_session",
        )

    def _plan_direct(
        self,
        *,
        node: str,
        source_path: str,
        node_materialization: ControllerArtifactBinding,
        work_root: Path,
    ) -> DagExecutionPlan:
        from lightcone_spec.experiments.formal_single_operator_early_execution import (
            materialize_formal_single_operator_early_run_plan_inputs,
        )
        from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
            materialize_formal_single_operator_e4_direct_run_plan_inputs,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            rebuild_formal_single_operator_node_materialization,
        )
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            materialize_formal_single_operator_downstream_serving_run_plan,
            materialize_formal_single_operator_serving_run_plan,
        )

        preflight_inputs = self._preflight_inputs_path()
        if not preflight_inputs.is_file():
            raise FormalExperimentDagBlocked(
                f"{node}: exact preflight execution inputs are unavailable"
            )
        rebuilt = rebuild_formal_single_operator_node_materialization(
            node_materialization.absolute_path
        )
        launches = []
        for cell in rebuilt.materialization.cells:
            cell_root = work_root / "cells" / cell.cell_id
            cell_root.mkdir(mode=0o700, parents=True)
            if node in _EARLY_NODES:
                materialize_formal_single_operator_early_run_plan_inputs(
                    execution_source_path=source_path,
                    materialized_cell_id=cell.cell_id,
                    preflight_inputs_path=preflight_inputs,
                    private_output_root=cell_root,
                )
                plan = materialize_formal_single_operator_serving_run_plan(
                    early_run_plan_inputs_path=(
                        cell_root / "formal-single-operator-early-run-plan-inputs.json"
                    )
                )
            else:
                materialize_formal_single_operator_e4_direct_run_plan_inputs(
                    execution_source_path=source_path,
                    materialized_cell_id=cell.cell_id,
                    repository_root=self.config.repository_root,
                    preflight_inputs_path=preflight_inputs,
                    private_output_root=cell_root,
                )
                plan = materialize_formal_single_operator_downstream_serving_run_plan(
                    downstream_run_plan_inputs_path=(
                        cell_root
                        / "formal-single-operator-downstream-run-plan-inputs.json"
                    )
                )
            launches.append(
                self._cell_launch(
                    node=node,
                    cell=cell,
                    attempt_number=1,
                    node_materialization=node_materialization,
                    execution_source_path=source_path,
                    run_plan=plan,
                    run_plan_path=cell_root / "formal-serving-run-plan.json",
                    run_root=cell_root,
                    physical_kind="serving",
                )
            )
        return DagExecutionPlan(
            execution_source=ControllerArtifactBinding.bind(source_path),
            prepared_launch=None,
            launches=tuple(sorted(launches, key=lambda row: row.attempt.cell_id)),
        )

    def _plan_prepared(
        self,
        *,
        node: str,
        source_path: str,
        node_materialization: ControllerArtifactBinding,
        work_root: Path,
        current_ns: int,
    ) -> DagExecutionPlan:
        from lightcone_spec.experiments.formal_failure_execution import (
            materialize_formal_single_operator_e5_failure_execution_descriptor,
        )
        from lightcone_spec.experiments.formal_single_operator_prepared_launch_producer import (
            finalize_prepared_launch_bundle,
            materialize_prepared_request_schedule,
            prepare_launch_draft,
        )
        from lightcone_spec.experiments.formal_single_operator_profiler import (
            materialize_formal_single_operator_profiler_plan,
        )
        from lightcone_spec.experiments.formal_single_operator_profiler_subject_producer import (
            publish_formal_single_operator_profiler_subject_requirement,
        )
        from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
            materialize_formal_single_operator_prepared_downstream_run_plan_inputs,
            route_formal_single_operator_materialized_cell,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            rebuild_formal_single_operator_node_materialization,
        )
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            materialize_formal_single_operator_e5_failure_run_plan,
            materialize_formal_single_operator_prepared_downstream_serving_run_plan,
        )
        from lightcone_spec.orchestration.formal_single_operator_admission import (
            publish_formal_single_operator_admission,
        )

        prerequisites = self.prerequisite_resolver.launch_manifest_paths(
            node=node,
            execution_source_path=source_path,
        )
        chronobelief_proofs = (
            self.prerequisite_resolver.chronobelief_gpu_parity_proof_paths(
                node=node,
                execution_source_path=source_path,
            )
        )
        if not prerequisites:
            raise FormalExperimentDagBlocked(
                f"{node}: prerequisite producer returned no launch manifests"
            )
        prepared_root = work_root / "prepared"
        prepared_root.mkdir(mode=0o700)
        draft = prepare_launch_draft(
            execution_source_path=source_path,
            content_source_path=self.config.content_source.absolute_path,
            prerequisite_launch_manifest_paths=prerequisites,
            chronobelief_gpu_parity_proof_paths=chronobelief_proofs,
            private_output_root=prepared_root,
        )
        draft_path = prepared_root / "prepared-launch-draft.json"
        rebuilt = rebuild_formal_single_operator_node_materialization(
            node_materialization.absolute_path
        )
        schedule_paths: list[str] = []
        profiler_cells = []
        for cell in rebuilt.materialization.cells:
            route = route_formal_single_operator_materialized_cell(
                node=node,
                phase=rebuilt.artifact.phase,
                cell=cell,
            )
            if route.physical_kind in {
                "e6_interface_preflight",
                "e0_compatibility_decision",
            }:
                continue
            if route.physical_kind == "profiler":
                profiler_cells.append(cell)
                continue
            schedule_root = prepared_root / "schedules" / cell.cell_id
            schedule_root.mkdir(mode=0o700, parents=True)
            arrival_path = None
            if cell.stage == "E5" and cell.task == "production_slo_power_prefix":
                arrival_path = self.e5_arrival_resolver.arrival_plan_path(
                    node=node,
                    execution_source_path=source_path,
                    materialized_cell_id=cell.cell_id,
                )
                if arrival_path is None:
                    raise FormalExperimentDagBlocked(
                        f"{node}: E5 arrival plan is unavailable for {cell.cell_id}"
                    )
            materialize_prepared_request_schedule(
                draft_path=draft_path,
                materialized_cell_id=cell.cell_id,
                private_output_root=schedule_root,
                e5_arrival_plan_path=arrival_path,
            )
            schedule_paths.append(
                str(schedule_root / "formal-request-schedule-receipt.json")
            )
        profiler_requirement_path = None
        if profiler_cells:
            requirement = prepared_root / "profiler-subject-requirement.json"
            publish_formal_single_operator_profiler_subject_requirement(
                execution_source_path=source_path,
                repository_root=self.config.repository_root,
                output_path=requirement,
            )
            profiler_requirement_path = str(requirement)
        bundle_path = prepared_root / "prepared-launch-bundle.json"
        bundle = finalize_prepared_launch_bundle(
            draft_path=draft_path,
            request_schedule_receipt_paths=tuple(sorted(schedule_paths)),
            output_path=bundle_path,
            profiler_subject_requirement_path=profiler_requirement_path,
            current_ns=current_ns,
        )
        if draft.execution_source_sha256 != bundle.execution_source_sha256:
            raise RuntimeError("prepared draft and bundle execution identities differ")
        launches = []
        session_specs = []
        for cell in rebuilt.materialization.cells:
            route = route_formal_single_operator_materialized_cell(
                node=node,
                phase=rebuilt.artifact.phase,
                cell=cell,
            )
            if route.physical_kind in {
                "e6_interface_preflight",
                "e0_compatibility_decision",
            }:
                continue
            cell_root = work_root / "cells" / cell.cell_id
            cell_root.mkdir(mode=0o700, parents=True)
            if route.physical_kind == "profiler":
                tool = self._profiler_tool(cell)
                plan = materialize_formal_single_operator_profiler_plan(
                    execution_source_path=source_path,
                    materialized_cell_id=cell.cell_id,
                    prepared_launch_bundle_path=bundle_path,
                    repository_root=self.config.repository_root,
                    private_output_root=cell_root,
                    tool_path=tool,
                    current_ns=current_ns,
                )
                plan_path = cell_root / "formal-single-operator-profiler-plan.json"
            elif route.physical_kind == "e5_failure":
                inputs = (
                    materialize_formal_single_operator_e5_failure_execution_descriptor(
                        execution_source_path=source_path,
                        materialized_cell_id=cell.cell_id,
                        prepared_launch_bundle_path=bundle_path,
                        repository_root=self.config.repository_root,
                        private_output_root=cell_root,
                        current_ns=current_ns,
                    )
                )
                descriptor_path = (
                    cell_root / "formal-single-operator-e5-failure-execution.json"
                )
                plan = materialize_formal_single_operator_e5_failure_run_plan(
                    failure_execution_descriptor_path=descriptor_path,
                )
                plan_path = cell_root / "formal-serving-run-plan.json"
                publish_formal_single_operator_admission(
                    plan_path=plan_path,
                    inventory_path=inputs.inventory.absolute_path,
                )
            else:
                prepared_inputs = materialize_formal_single_operator_prepared_downstream_run_plan_inputs(
                    execution_source_path=source_path,
                    materialized_cell_id=cell.cell_id,
                    prepared_launch_bundle_path=bundle_path,
                    private_output_root=cell_root,
                    current_ns=current_ns,
                )
                plan = materialize_formal_single_operator_prepared_downstream_serving_run_plan(
                    prepared_downstream_run_plan_inputs_path=(
                        cell_root
                        / "formal-single-operator-prepared-downstream-inputs.json"
                    )
                )
                plan_path = cell_root / "formal-serving-run-plan.json"
            launch_row = self._cell_launch(
                node=node,
                cell=cell,
                attempt_number=1,
                node_materialization=node_materialization,
                execution_source_path=source_path,
                run_plan=plan,
                run_plan_path=plan_path,
                run_root=cell_root,
                physical_kind=route.physical_kind,
            )
            launches.append(launch_row)
            if route.physical_kind == "serving":
                from lightcone_spec.config import load_run_config
                from lightcone_spec.orchestration.formal_physical_dispatch import (
                    formal_serving_process_runtime_contract,
                )
                from lightcone_spec.orchestration.formal_serving_session_group import (
                    build_formal_serving_session_group_spec,
                )
                from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
                from lightcone_spec.runtime.proof_artifact import (
                    CanonicalJsonProofBinding,
                )

                compile_launch = CompileLaunchManifest.load(
                    plan.launch_manifest.absolute_path
                )
                run_config = load_run_config(compile_launch.run_config_path)
                formal_serving_process_runtime_contract(str(plan_path))
                execution_policy = plan.serving_execution_policy
                if execution_policy is None:
                    raise ValueError(
                        "prepared serving group lacks its source timing policy"
                    )
                session_specs.append(
                    build_formal_serving_session_group_spec(
                        node=node,
                        stage=cell.stage,
                        phase=rebuilt.artifact.phase,
                        materialized_cell_id=cell.cell_id,
                        attempt=1,
                        physical_kind="serving",
                        method_family=(
                            "lightcone"
                            if cell.method_role == "LightCone"
                            else run_config.method
                        ),
                        protocol_lock_sha256=launch_row.attempt.identity[
                            "protocol_lock_sha256"
                        ],
                        source_snapshot_sha256=launch_row.attempt.identity[
                            "source_sha256"
                        ],
                        inventory_sha256=compile_launch.inventory_sha256,
                        run_plan=CanonicalJsonProofBinding.bind(plan_path),
                        prepared_launch_entry_sha256=(
                            prepared_inputs.prepared_launch_entry_sha256
                        ),
                        compile_launch_manifest_sha256=compile_launch.sha256,
                        request_schedule_sha256=(
                            prepared_inputs.request_schedule_receipt.semantic_sha256
                        ),
                        launch=compile_launch,
                        config=run_config,
                        output_directory=str(cell_root),
                        # The outer watchdog timeout is a fail-safe bound that
                        # can reserve hours per request wave; using it as a
                        # launch-group estimate would make every compatible
                        # pair a singleton.  The source-owned serving policy
                        # instead binds the actual warmup/arrival/deadline/
                        # drain timing window used for deterministic grouping.
                        estimated_duration_seconds=max(
                            1.0,
                            (execution_policy.minimum_process_timeout_us / 1_000_000),
                        ),
                        dispatch_order_key=(
                            launch_row.command.launch_compatibility_key,
                            cell.method_role,
                            cell.cell_id,
                        ),
                    )
                )
        grouped_launches, physical_groups = self._materialize_serving_session_groups(
            node=node,
            work_root=work_root,
            node_materialization=node_materialization,
            launches=tuple(launches),
            specs=tuple(session_specs),
        )
        return DagExecutionPlan(
            execution_source=ControllerArtifactBinding.bind(source_path),
            prepared_launch=ControllerArtifactBinding.bind(bundle_path),
            launches=tuple(
                sorted(grouped_launches, key=lambda row: row.attempt.cell_id)
            ),
            physical_attempt_groups=tuple(
                sorted(physical_groups, key=lambda row: row.group_id)
            ),
        )

    def _profiler_tool(self, cell: Any) -> str:
        variant = dict(cell.dimensions).get("profiler")
        expected = "ncu" if variant == "nsight_compute" else "nsys"
        matches = tuple(
            row.absolute_path
            for row in self.config.profiler_tools
            if Path(row.absolute_path).name == expected
        )
        if len(matches) != 1:
            raise FormalExperimentDagBlocked(
                f"e4_profiler: exact {expected} tool binding is unavailable"
            )
        return matches[0]

    def retry_attempt(
        self,
        previous_command: QueuedCommandSpec,
        next_attempt: int,
    ) -> tuple[CellAttemptSpec, QueuedCommandSpec]:
        """Rebuild one infrastructure retry into fresh attempt-owned paths.

        The prior command supplies no scientific scalar.  The durable node,
        execution source, materialized cell, prepared bundle, and source-owned
        plan producers are reopened before the new command is accepted.  The
        serving runtime contract then proves that path churn did not alter the
        path-independent scientific command digest.
        """

        if type(previous_command) is not QueuedCommandSpec:
            raise TypeError("formal retry requires an exact queued command")
        if (
            type(next_attempt) is not int
            or next_attempt != previous_command.attempt + 1
        ):
            raise ValueError("formal retry attempt is not contiguous")
        previous = self.store.attempt(
            previous_command.cell_id,
            previous_command.attempt,
        )
        if (
            previous["status"] != "FAILED"
            or not str(previous["failure_code"] or "").startswith("INFRASTRUCTURE:")
            or previous["retry_decision"] != "RETRY_INFRASTRUCTURE_AUTOMATIC"
            or previous["scientific_command_sha256"] is None
        ):
            raise ExperimentOperatorError(
                "formal retry lacks one sealed infrastructure failure"
            )
        candidates = tuple(
            row
            for row in self.store.controller_nodes()
            if (row["stage"], row["phase"]) == (previous["stage"], previous["phase"])
        )
        if len(candidates) != 1:
            raise ExperimentOperatorError("formal retry node identity is ambiguous")
        controller = candidates[0]
        node = str(controller["node"])
        if node == "preflight":
            raise ExperimentOperatorError(
                "preflight physical-group retries require group authority"
            )
        execution_source_path = str(controller["execution_source_path"] or "")
        node_materialization_path = str(controller["node_materialization_path"] or "")
        if not execution_source_path or not node_materialization_path:
            raise ExperimentOperatorError("formal retry node is not fully planned")
        node_materialization = ControllerArtifactBinding.bind(node_materialization_path)
        from lightcone_spec.experiments.formal_single_operator_stages import (
            rebuild_formal_single_operator_node_materialization,
        )

        rebuilt = rebuild_formal_single_operator_node_materialization(
            node_materialization.absolute_path
        )
        cells = tuple(
            cell
            for cell in rebuilt.materialization.cells
            if cell.cell_id == previous_command.cell_id
        )
        if len(cells) != 1:
            raise ExperimentOperatorError(
                "formal retry cell is absent from durable materialization"
            )
        cell = cells[0]
        retry_root = (
            self._node_root(node)
            / "execution"
            / "work"
            / "retries"
            / cell.cell_id
            / f"attempt-{next_attempt:04d}"
        )
        if retry_root.exists():
            _preserve_partial_directory(
                retry_root,
                label=f"{node} incomplete retry attempt {next_attempt}",
            )
        retry_root.mkdir(mode=0o700, parents=True)

        physical_kind = "serving"
        if node in _EARLY_NODES or node in _E4_DIRECT_NODES:
            from lightcone_spec.experiments.formal_single_operator_early_execution import (
                materialize_formal_single_operator_early_run_plan_inputs,
            )
            from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
                materialize_formal_single_operator_e4_direct_run_plan_inputs,
            )
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                materialize_formal_single_operator_downstream_serving_run_plan,
                materialize_formal_single_operator_serving_run_plan,
            )

            preflight_inputs = self._preflight_inputs_path()
            if node in _EARLY_NODES:
                materialize_formal_single_operator_early_run_plan_inputs(
                    execution_source_path=execution_source_path,
                    materialized_cell_id=cell.cell_id,
                    preflight_inputs_path=preflight_inputs,
                    private_output_root=retry_root,
                )
                run_plan = materialize_formal_single_operator_serving_run_plan(
                    early_run_plan_inputs_path=(
                        retry_root / "formal-single-operator-early-run-plan-inputs.json"
                    )
                )
            else:
                materialize_formal_single_operator_e4_direct_run_plan_inputs(
                    execution_source_path=execution_source_path,
                    materialized_cell_id=cell.cell_id,
                    repository_root=self.config.repository_root,
                    preflight_inputs_path=preflight_inputs,
                    private_output_root=retry_root,
                )
                run_plan = (
                    materialize_formal_single_operator_downstream_serving_run_plan(
                        downstream_run_plan_inputs_path=(
                            retry_root
                            / "formal-single-operator-downstream-run-plan-inputs.json"
                        )
                    )
                )
            run_plan_path = retry_root / "formal-serving-run-plan.json"
        elif node in _PREPARED_NODES:
            from lightcone_spec.experiments.formal_failure_execution import (
                materialize_formal_single_operator_e5_failure_execution_descriptor,
            )
            from lightcone_spec.experiments.formal_single_operator_profiler import (
                materialize_formal_single_operator_profiler_plan,
            )
            from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
                materialize_formal_single_operator_prepared_downstream_run_plan_inputs,
                route_formal_single_operator_materialized_cell,
            )
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                materialize_formal_single_operator_e5_failure_run_plan,
                materialize_formal_single_operator_prepared_downstream_serving_run_plan,
            )
            from lightcone_spec.orchestration.formal_single_operator_admission import (
                publish_formal_single_operator_admission,
            )

            prepared_bundle_path = str(controller["prepared_launch_path"] or "")
            if not prepared_bundle_path:
                raise ExperimentOperatorError(
                    "prepared formal retry lacks the durable launch bundle"
                )
            route = route_formal_single_operator_materialized_cell(
                node=node,
                phase=rebuilt.artifact.phase,
                cell=cell,
            )
            physical_kind = route.physical_kind
            current_ns = self._clock(
                self._node_root(node) / "execution",
                "planning-clock.json",
            )
            if physical_kind == "profiler":
                run_plan = materialize_formal_single_operator_profiler_plan(
                    execution_source_path=execution_source_path,
                    materialized_cell_id=cell.cell_id,
                    prepared_launch_bundle_path=prepared_bundle_path,
                    repository_root=self.config.repository_root,
                    private_output_root=retry_root,
                    tool_path=self._profiler_tool(cell),
                    current_ns=current_ns,
                )
                run_plan_path = retry_root / "formal-single-operator-profiler-plan.json"
            elif physical_kind == "e5_failure":
                inputs = (
                    materialize_formal_single_operator_e5_failure_execution_descriptor(
                        execution_source_path=execution_source_path,
                        materialized_cell_id=cell.cell_id,
                        prepared_launch_bundle_path=prepared_bundle_path,
                        repository_root=self.config.repository_root,
                        private_output_root=retry_root,
                        current_ns=current_ns,
                    )
                )
                descriptor_path = (
                    retry_root / "formal-single-operator-e5-failure-execution.json"
                )
                run_plan = materialize_formal_single_operator_e5_failure_run_plan(
                    failure_execution_descriptor_path=descriptor_path,
                )
                run_plan_path = retry_root / "formal-serving-run-plan.json"
                publish_formal_single_operator_admission(
                    plan_path=run_plan_path,
                    inventory_path=inputs.inventory.absolute_path,
                )
            elif physical_kind in {
                "e6_interface_preflight",
                "e0_compatibility_decision",
            }:
                raise ExperimentOperatorError(
                    "adopted auxiliary cells cannot enter the serving retry builder"
                )
            else:
                materialize_formal_single_operator_prepared_downstream_run_plan_inputs(
                    execution_source_path=execution_source_path,
                    materialized_cell_id=cell.cell_id,
                    prepared_launch_bundle_path=prepared_bundle_path,
                    private_output_root=retry_root,
                    current_ns=current_ns,
                )
                run_plan = materialize_formal_single_operator_prepared_downstream_serving_run_plan(
                    prepared_downstream_run_plan_inputs_path=(
                        retry_root
                        / "formal-single-operator-prepared-downstream-inputs.json"
                    )
                )
                run_plan_path = retry_root / "formal-serving-run-plan.json"
        else:
            raise ExperimentOperatorError(
                "formal retry node has no source-owned serving producer"
            )

        launch = self._cell_launch(
            node=node,
            cell=cell,
            attempt_number=next_attempt,
            node_materialization=node_materialization,
            execution_source_path=execution_source_path,
            run_plan=run_plan,
            run_plan_path=run_plan_path,
            run_root=retry_root,
            physical_kind=physical_kind,
        )
        if (
            launch.attempt.scientific_command_sha256
            != previous["scientific_command_sha256"]
        ):
            raise ExperimentOperatorError(
                "formal retry changed its path-independent scientific command"
            )
        return launch.attempt, launch.command

    def _attempt_spec(
        self,
        *,
        node: str,
        cell: Any,
        command: QueuedCommandSpec,
        scientific_command_sha256: str | None,
        output_directory: Path,
        execution_source_path: str,
    ) -> CellAttemptSpec:
        from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
        from lightcone_spec.experiments.formal_single_operator_content import (
            load_trusted_single_operator_content_bundle,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            formal_single_operator_node_spec,
            load_formal_single_operator_execution_source,
        )

        protocol_lock = protocol_lock_from_dict(
            _read_canonical_json(
                self.config.protocol_lock.absolute_path,
                label="ProtocolLock",
            )
        )
        content = load_trusted_single_operator_content_bundle(
            self.config.content_source.absolute_path
        )
        source = load_formal_single_operator_execution_source(execution_source_path)
        spec = formal_single_operator_node_spec(node)
        dimensions = dict(cell.dimensions)
        scientific_axes = {
            "backend": cell.backend,
            "method_role": cell.method_role,
            "model": cell.model,
            "publication_policy": cell.publication_policy,
            "recipe_sha256": cell.recipe_sha256,
            "task": cell.task,
            **dimensions,
        }
        block_value = dimensions.get("block")
        seed_value = dimensions.get("seed")
        return CellAttemptSpec(
            cell_id=cell.cell_id,
            attempt=command.attempt,
            stage=spec.stage,
            phase=spec.phase,
            block=None if block_value is None else str(block_value),
            seed=seed_value if type(seed_value) is int else None,
            scientific_axes=scientific_axes,
            identity={
                "source_sha256": (content.source_snapshot.source_snapshot_sha256),
                "patch_sha256": protocol_lock.patch_manifest_sha256,
                "registry_sha256": protocol_lock.registry_sha256,
                "protocol_lock_sha256": protocol_lock.sha256,
                "content_source_sha256": content.semantic_sha256,
                "materialization_sha256": source.materialization_sha256,
                "execution_source_sha256": source.sha256,
            },
            command_sha256=command.command_sha256,
            scientific_command_sha256=scientific_command_sha256,
            output_directory=str(output_directory),
        )

    def _cell_launch(
        self,
        *,
        node: str,
        cell: Any,
        attempt_number: int,
        node_materialization: ControllerArtifactBinding,
        execution_source_path: str,
        run_plan: Any,
        run_plan_path: Path,
        run_root: Path,
        physical_kind: str,
    ) -> DagCellLaunch:
        from lightcone_spec.config import load_run_config
        from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
            formal_single_operator_launch_compatibility_key,
        )
        from lightcone_spec.orchestration.formal_cell_worker import (
            FormalCellWorkerSpec,
        )
        from lightcone_spec.orchestration.formal_physical_dispatch import (
            formal_serving_process_runtime_contract,
        )
        from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
        from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

        if physical_kind == "profiler":
            subject = CanonicalJsonProofBinding.bind(
                run_plan.subject_run_plan.absolute_path
            )
            from lightcone_spec.orchestration.formal_physical_dispatch import (
                FormalServingRunPlan,
            )

            subject_plan = FormalServingRunPlan.from_dict(subject.reopen())
            serving_plan = subject_plan
            scientific_plan_path = subject.absolute_path
            gpu_uuids = subject_plan.gpu_uuids
            actual_path = run_root / "capture" / "profiler-terminal.json"
            timing_class = "PROFILER"
        else:
            serving_plan = run_plan
            scientific_plan_path = str(run_plan_path)
            gpu_uuids = serving_plan.gpu_uuids
            actual_path = (
                Path(serving_plan.live_run_receipt_output_path)
                if physical_kind == "e5_failure"
                else run_root / "formal-single-operator-manifest.json"
            )
            timing_class = "FAILURE" if physical_kind == "e5_failure" else "HEADLINE"
        runtime_contract = formal_serving_process_runtime_contract(scientific_plan_path)
        if runtime_contract.plan_sha256 != serving_plan.sha256:
            raise ValueError("serving runtime contract plan identity differs")
        launch = CompileLaunchManifest.load(serving_plan.launch_manifest.absolute_path)
        config = load_run_config(launch.run_config_path)
        compatibility_key = formal_single_operator_launch_compatibility_key(
            launch=launch,
            config=config,
        )
        inventory_uuids = self._inventory_gpu_uuids()
        required_gpu_count = len(gpu_uuids)
        preferred_gpu_index = None
        paired_gpu_key = None
        if required_gpu_count == 1:
            try:
                preferred_gpu_index = inventory_uuids.index(gpu_uuids[0])
            except ValueError as error:
                raise ValueError("run plan GPU is outside driver inventory") from error
            paired_gpu_key = _semantic_sha256(
                {
                    "stage": cell.stage,
                    "block": dict(cell.dimensions).get("block"),
                    "cell_gpu_uuid": gpu_uuids[0],
                    "cell_id": cell.cell_id,
                }
            )
        elif required_gpu_count == 2:
            timing_class = "EXCLUSIVE" if timing_class == "HEADLINE" else timing_class
        else:
            raise ValueError("formal cell run plan GPU count differs")
        operator_root = run_root / "operator"
        operator_root.mkdir(mode=0o700)
        worker_spec_path = operator_root / "worker-spec.json"
        worker_spec = FormalCellWorkerSpec(
            schema_version=1,
            kind="formal_single_operator_cell_worker",
            cell_id=cell.cell_id,
            attempt=attempt_number,
            repository_root=self.config.repository_root,
            node_materialization_path=node_materialization.absolute_path,
            actual_result_path=str(actual_path),
            evidence_root=str(run_root),
            evidence_manifest_path=str(operator_root / "evidence-manifest.json"),
            job_argv=(
                self.python_executable,
                "-m",
                "lightcone_spec.cli.main",
                "formal-single-operator",
                "execute-run",
                "--repository-root",
                self.config.repository_root,
                "--run-plan",
                str(run_plan_path),
            ),
            failure_class_on_nonzero=(
                "FAILURE_DIAGNOSTIC" if physical_kind == "e5_failure" else "SCIENTIFIC"
            ),
            included_in_analysis_on_complete=physical_kind != "profiler",
            complete_exclusion_reason=(
                "profiler_only_non_headline" if physical_kind == "profiler" else None
            ),
        )
        spec_sha = self._publish_or_reopen_worker_spec(
            worker_spec,
            worker_spec_path,
        )
        command = QueuedCommandSpec(
            cell_id=cell.cell_id,
            attempt=attempt_number,
            argv=(
                self.python_executable,
                "-m",
                "lightcone_spec.orchestration.formal_cell_worker",
                "--spec",
                str(worker_spec_path),
            ),
            launch_compatibility_key=compatibility_key,
            required_gpu_count=required_gpu_count,
            timing_class=timing_class,  # type: ignore[arg-type]
            predicted_high_water_bytes=_CELL_SPOOL_HIGH_WATER_BYTES,
            monitored_path=self.config.run_root,
            log_path=str(operator_root / "command.log"),
            expected_terminal_path=str(operator_root / "terminal.json"),
            expected_junit_path=str(operator_root / "junit.xml"),
            expected_raw_log_path=str(operator_root / "raw.json"),
            atomic_pointer_path=str(operator_root / "result-pointer.json"),
            child_exit_receipt_path=str(operator_root / "child-exit.json"),
            environment=(
                ("LIGHTCONE_CELL_WORKER_SPEC_SHA256", spec_sha),
                (
                    "LIGHTCONE_OPERATOR_PROGRESS_LOG_PATHS_JSON",
                    json.dumps(
                        runtime_contract.progress_log_paths,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            ),
            paired_gpu_key=paired_gpu_key,
            preferred_gpu_index=preferred_gpu_index,
            priority=(50 if physical_kind in {"profiler", "e5_failure"} else 0),
            max_runtime_seconds=runtime_contract.outer_max_runtime_seconds,
        )
        attempt = self._attempt_spec(
            node=node,
            cell=cell,
            command=command,
            scientific_command_sha256=(runtime_contract.scientific_command_sha256),
            output_directory=run_root,
            execution_source_path=execution_source_path,
        )
        return DagCellLaunch(attempt=attempt, command=command)

    @staticmethod
    def _publish_or_reopen_worker_spec(spec: Any, path: Path) -> str:
        from lightcone_spec.orchestration.formal_cell_worker import (
            FormalCellWorkerSpec,
            load_formal_cell_worker_spec,
            publish_formal_cell_worker_spec,
        )

        if type(spec) is not FormalCellWorkerSpec:
            raise TypeError("worker-spec publication requires an exact spec")
        if path.exists():
            rebound, digest = load_formal_cell_worker_spec(path)
            if rebound != spec:
                raise FormalExperimentDagBlocked(
                    "worker-spec output is occupied by another command"
                )
            return digest
        return publish_formal_cell_worker_spec(spec, path)

    def _inventory_gpu_uuids(self) -> tuple[str, ...]:
        from lightcone_spec.experiments.gpu_pool import GpuInventory

        inventory = GpuInventory.from_dict(
            _read_canonical_json(
                self.config.inventory.absolute_path,
                label="GPU inventory",
            )
        )
        ready = tuple(sorted(row.uuid for row in inventory.devices if row.ready))
        if len(ready) != 2:
            raise ValueError("formal DAG driver requires exactly two ready GPUs")
        return ready

    def _publish_plan_journal(self, path: Path, plan: DagExecutionPlan) -> None:
        value = {
            "schema_version": _PLAN_JOURNAL_SCHEMA_VERSION,
            "kind": _PLAN_JOURNAL_KIND,
            "execution_source": asdict(plan.execution_source),
            "prepared_launch": (
                None if plan.prepared_launch is None else asdict(plan.prepared_launch)
            ),
            "launches": [
                {
                    "attempt": asdict(row.attempt),
                    "command": asdict(row.command),
                }
                for row in plan.launches
            ],
            "physical_attempt_groups": [
                {
                    "group_id": group.group_id,
                    "group_kind": group.group_kind,
                    "leader_cell_id": group.leader_cell_id,
                    "members": [
                        {
                            "attempt": asdict(member.attempt),
                            "command": asdict(member.command),
                            "logical_kind": member.logical_kind,
                        }
                        for member in group.members
                    ],
                }
                for group in plan.physical_attempt_groups
            ],
        }
        _publish_no_replace(path, value)

    def _load_plan_journal(
        self,
        path: Path,
        *,
        expected_node: str,
        expected_node_materialization: ControllerArtifactBinding,
    ) -> DagExecutionPlan:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            load_formal_single_operator_execution_source,
        )

        value = _read_canonical_json(path, label="execution-plan journal")
        if set(value) != {
            "schema_version",
            "kind",
            "execution_source",
            "prepared_launch",
            "launches",
            "physical_attempt_groups",
        } or (
            value["schema_version"] != _PLAN_JOURNAL_SCHEMA_VERSION
            or value["kind"] != _PLAN_JOURNAL_KIND
        ):
            raise ValueError("execution-plan journal schema differs")
        execution = ControllerArtifactBinding(**value["execution_source"])
        if ControllerArtifactBinding.bind(execution.absolute_path) != execution:
            raise ValueError("journal execution source changed")
        source = load_formal_single_operator_execution_source(execution.absolute_path)
        if (
            source.node != expected_node
            or source.materialization_source.absolute_path
            != self.store.controller_node(expected_node)["materialization_path"]
            or expected_node_materialization.absolute_path
            != self.store.controller_node(expected_node)["node_materialization_path"]
        ):
            raise ValueError("journal belongs to another controller node")
        prepared_raw = value["prepared_launch"]
        prepared = (
            None if prepared_raw is None else ControllerArtifactBinding(**prepared_raw)
        )
        if (
            prepared is not None
            and ControllerArtifactBinding.bind(prepared.absolute_path) != prepared
        ):
            raise ValueError("journal prepared launch changed")
        raw_launches = value["launches"]
        raw_groups = value["physical_attempt_groups"]
        if type(raw_launches) is not list or type(raw_groups) is not list:
            raise TypeError("execution-plan journal launch arrays differ")
        launches = tuple(self._decode_launch(row) for row in raw_launches)
        groups = []
        for raw in raw_groups:
            registered_fields = {
                "group_id",
                "leader_cell_id",
                "members",
            }
            if (
                type(raw) is not dict
                or set(raw)
                not in (
                    registered_fields,
                    {*registered_fields, "group_kind"},
                )
                or type(raw["members"]) is not list
            ):
                raise ValueError("execution-plan physical group fields differ")
            members = tuple(self._decode_group_member(row) for row in raw["members"])
            groups.append(
                DagPhysicalAttemptGroup(
                    group_id=raw["group_id"],
                    leader_cell_id=raw["leader_cell_id"],
                    members=members,
                    group_kind=raw.get("group_kind", "preflight_exact_ten"),
                )
            )
        return DagExecutionPlan(
            execution_source=execution,
            prepared_launch=prepared,
            launches=launches,
            physical_attempt_groups=tuple(groups),
        )

    @staticmethod
    def _decode_attempt(value: object) -> CellAttemptSpec:
        if type(value) is not dict:
            raise TypeError("journal attempt is not an object")
        return CellAttemptSpec(**value)

    @staticmethod
    def _decode_command(value: object) -> QueuedCommandSpec:
        if type(value) is not dict:
            raise TypeError("journal command is not an object")
        row = dict(value)
        row["argv"] = tuple(row["argv"])
        row["environment"] = tuple(tuple(item) for item in row["environment"])
        return QueuedCommandSpec(**row)

    @classmethod
    def _decode_launch(cls, value: object) -> DagCellLaunch:
        if type(value) is not dict or set(value) != {"attempt", "command"}:
            raise ValueError("journal launch fields differ")
        return DagCellLaunch(
            attempt=cls._decode_attempt(value["attempt"]),
            command=cls._decode_command(value["command"]),
        )

    @classmethod
    def _decode_group_member(cls, value: object) -> PhysicalAttemptGroupMemberSpec:
        if type(value) is not dict or set(value) != {
            "attempt",
            "command",
            "logical_kind",
        }:
            raise ValueError("journal group-member fields differ")
        return PhysicalAttemptGroupMemberSpec(
            attempt=cls._decode_attempt(value["attempt"]),
            command=cls._decode_command(value["command"]),
            logical_kind=value["logical_kind"],
        )

    def actual_results(
        self,
        node: str,
        attempts: tuple[dict[str, Any], ...],
    ) -> Mapping[str, str]:
        from lightcone_spec.orchestration.formal_cell_worker import (
            revalidate_formal_cell_worker_terminal,
        )
        from lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker import (
            formal_preflight_exact_ten_group_spec_path,
        )

        auxiliary: dict[str, str] = {}
        if self.auxiliary_runtime is not None:
            auxiliary = dict(self.auxiliary_runtime.actual_result_paths(node, attempts))
        results = dict(auxiliary)
        for attempt in attempts:
            cell_id = str(attempt["cell_id"])
            if cell_id in results:
                continue
            command = self.store.command_for_attempt(cell_id, int(attempt["attempt"]))
            if command is None:
                raise FormalExperimentDagBlocked(
                    f"{node}: adopted actual-result resolver is unavailable for {cell_id}"
                )
            if formal_preflight_exact_ten_group_spec_path(command) is not None:
                raw = _read_canonical_json(
                    command.expected_raw_log_path,
                    label="exact-ten logical raw evidence",
                )
                completion = raw.get("completion")
                if type(completion) is not dict or set(completion) != {
                    "absolute_path",
                    "raw_sha256",
                    "semantic_sha256",
                    "size_bytes",
                }:
                    raise ValueError("exact-ten raw evidence lacks completion binding")
                if (
                    attempt.get("raw_log_sha256")
                    != _file_sha256(command.expected_raw_log_path)
                    or _file_sha256(completion["absolute_path"])
                    != completion["raw_sha256"]
                ):
                    raise ValueError("exact-ten completion evidence changed")
                results[cell_id] = completion["absolute_path"]
                continue
            terminal = _read_canonical_json(
                command.expected_terminal_path,
                label="formal cell worker terminal",
            )
            if attempt.get("terminal_sha256") != _file_sha256(
                command.expected_terminal_path
            ):
                raise ValueError("formal cell terminal differs from ledger")
            evidence = revalidate_formal_cell_worker_terminal(
                terminal,
                command=command,
            )
            actual_path = terminal.get("actual_result_path")
            actual_sha = terminal.get("actual_result_raw_sha256")
            if (
                type(actual_path) is not str
                or type(actual_sha) is not str
                or evidence.get(actual_path) != actual_sha
                or _file_sha256(actual_path) != actual_sha
            ):
                raise ValueError("formal cell actual path is not terminal-bound")
            results[cell_id] = actual_path
        if tuple(sorted(results)) != tuple(
            sorted(str(row["cell_id"]) for row in attempts)
        ):
            raise ValueError("formal actual-result coverage differs")
        return results

    def _record_selection_once(self, rebuilt: Any, decision_path: Path) -> None:
        decision = rebuilt.decision
        source_sha = ControllerArtifactBinding.bind(decision_path).sha256
        value = decision.to_dict()
        try:
            self.store.record_selection_decision(
                decision_id=decision.sha256,
                stage=decision.stage,
                phase=decision.phase,
                decision_kind=decision.decision_kind,
                source_sha256=source_sha,
                decision=value,
                occurred_at_ns=rebuilt.artifact.completed_ns,
            )
        except ExperimentOperatorError:
            matches = tuple(
                row
                for row in self.store._selection_rows()
                if row["decision_id"] == decision.sha256
            )
            if len(matches) != 1 or {
                "stage": matches[0]["stage"],
                "phase": matches[0]["phase"],
                "decision_kind": matches[0]["decision_kind"],
                "source_sha256": matches[0]["source_sha256"],
                "decision": matches[0]["decision"],
            } != {
                "stage": decision.stage,
                "phase": decision.phase,
                "decision_kind": decision.decision_kind,
                "source_sha256": source_sha,
                "decision": value,
            }:
                raise

    def _completed_actual_attempt(self, cell_id: str) -> int:
        latest = self.store.latest_attempt(cell_id)
        if latest is None or latest["status"] != "COMPLETE":
            raise ValueError("validated actual metric lacks a COMPLETE attempt")
        return int(latest["attempt"])

    @staticmethod
    def _serving_descriptive_rows(
        observation: dict[str, object],
    ) -> tuple[dict[str, object], ...]:
        """Reduce one deep-validated terminal without inventing uncertainty."""

        from lightcone_spec.experiments import formal_single_operator_stages as stages
        from lightcone_spec.experiments.formal_slo_metrics import linear_p99_ns

        requests = stages._single_operator_request_rows(observation)  # type: ignore[attr-defined]
        slo = stages._single_operator_slo_goodput(observation)  # type: ignore[attr-defined]
        statuses = {
            "completed": 0,
            "rejected": 0,
            "timed_out": 0,
            "cancelled": 0,
            "unfinished": 0,
        }
        native_aborted = 0
        intervals: list[int] = []
        for request in requests:
            status = request.get("terminal_status")
            if status not in statuses:
                raise ValueError("serving metric terminal status is unregistered")
            statuses[str(status)] += 1
            if request.get("native_terminal_status") == "aborted":
                native_aborted += 1
            timestamps = request.get("token_observed_ns")
            if type(timestamps) is not list or any(
                type(value) is not int for value in timestamps
            ):
                raise TypeError("serving metric native timestamps differ")
            if status == "completed":
                intervals.extend(right - left for left, right in pairwise(timestamps))
        offered = len(requests)
        if offered < 1 or sum(statuses.values()) != offered:
            raise ValueError("serving metric offered-request denominator differs")
        if (
            statuses["completed"] != slo.completed_requests
            or statuses["unfinished"] != slo.error_requests
            or slo.eligible_requests != offered
        ):
            raise ValueError("serving metric SLO denominator differs")
        denominator = {
            "offered_request_count": offered,
            "completed_request_count": statuses["completed"],
            "rejected_request_count": statuses["rejected"],
            "timed_out_request_count": statuses["timed_out"],
            "cancelled_request_count": statuses["cancelled"],
            "unfinished_request_count": statuses["unfinished"],
            # Native aborted is supporting server evidence for client timeout
            # or cancellation, never a sixth client denominator status.
            "aborted_request_count": native_aborted,
            "terminal_status_counts": {
                name: statuses[name] for name in sorted(statuses)
            },
        }
        common = {
            **denominator,
            "request_count": offered,
            "independent_block_count": 1,
            "paired": False,
        }
        rows: list[dict[str, object]] = [
            {
                **common,
                "metric_name": "slo_goodput_tokens_per_second",
                "point_estimate": float(slo.goodput_tokens_per_second),
                "reducer_method": "formal_slo_goodput_v2",
                "attributes": {
                    **denominator,
                    "formal_slo_protocol_sha256": slo.protocol_sha256,
                    "formal_slo_observation_sha256": slo.sha256,
                    "formal_slo_status": slo.status,
                    "qualified_request_count": slo.qualified_requests,
                    "qualified_output_token_count": slo.qualified_output_tokens,
                    "scored_window_ns": slo.scored_window_ns,
                },
            }
        ]
        p99 = linear_p99_ns(tuple(intervals))
        if p99 is not None:
            rows.append(
                {
                    **common,
                    "metric_name": "native_p99_itl_ms",
                    "point_estimate": float(p99 / 1_000_000),
                    "reducer_method": "linear_native_p99_v1",
                    "attributes": {
                        **denominator,
                        "native_itl_sample_count": len(intervals),
                        "native_timestamp_source": (
                            "validated_unsigned_native_itl_pointer_bundle"
                        ),
                    },
                }
            )
        counters = observation.get("performance_counters")
        if type(counters) is not dict:
            raise TypeError("serving metric performance counters differ")
        for name in _SERVING_SCALAR_COUNTER_METRICS:
            value = counters.get(name)
            if value is None:
                continue
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise ValueError(f"serving metric counter {name} is invalid")
            rows.append(
                {
                    **common,
                    "metric_name": f"native_runtime_counter/{name}",
                    "point_estimate": float(value),
                    "reducer_method": "validated_native_terminal_counter_v1",
                    "attributes": {
                        **denominator,
                        "native_counter_name": name,
                    },
                }
            )
        return tuple(rows)

    def _actual_metrics(self, rebuilt: Any) -> tuple[MetricRecord, ...]:
        """Project every validated actual while keeping final CIs reducer-owned."""

        from lightcone_spec.experiments import formal_single_operator_stages as stages

        decision = rebuilt.decision
        cells = {cell.cell_id: cell for cell in rebuilt.materialization.cells}
        actual_results = rebuilt.artifact.actual_results
        actuals = {actual.cell_id: actual for actual in actual_results}
        if set(cells) != set(actuals) or len(actuals) != len(actual_results):
            raise ValueError("actual metric projection coverage differs")
        output: list[MetricRecord] = []

        def append(
            *,
            cell: Any,
            actual: Any,
            metric_name: str,
            point_estimate: float,
            reducer_method: str,
            attributes: Mapping[str, Any],
            independent_block_count: int | None = None,
            request_count: int | None = None,
            paired: bool | None = None,
        ) -> None:
            common = {
                "result_identity_sha256": actual.result_identity_sha256,
                "validator_kind": actual.validator_kind,
                "validator_protocol_sha256": actual.validator_protocol_sha256,
                "method_role": cell.method_role,
                "model": cell.model,
                "backend": cell.backend,
                "task": cell.task,
                "dimensions": [list(row) for row in cell.dimensions],
            }
            output.append(
                MetricRecord(
                    stage=decision.stage,
                    phase=decision.phase,
                    cell_id=cell.cell_id,
                    attempt=self._completed_actual_attempt(cell.cell_id),
                    metric_name=metric_name,
                    metric_kind="descriptive",
                    point_estimate=point_estimate,
                    ci_low=None,
                    ci_high=None,
                    independent_block_count=independent_block_count,
                    request_count=request_count,
                    paired=paired,
                    reducer_method=reducer_method,
                    attributes={**common, **dict(attributes)},
                )
            )

        for cell_id in sorted(cells):
            cell = cells[cell_id]
            actual = actuals[cell_id]
            duration_ns = actual.finished_ns - actual.started_ns
            if duration_ns < 1:
                raise ValueError("validated actual duration is not positive")
            append(
                cell=cell,
                actual=actual,
                metric_name="validated_actual_wall_seconds",
                point_estimate=duration_ns / 1_000_000_000,
                reducer_method="source_validated_actual_lifecycle_v1",
                attributes={
                    "started_ns": actual.started_ns,
                    "finished_ns": actual.finished_ns,
                },
            )
            payload = actual.reducer_payload
            serving = payload.get("serving_observation")
            if serving is not None:
                observation = stages._serving_observation(actual, cell)  # type: ignore[attr-defined]
                for row in self._serving_descriptive_rows(observation):
                    append(
                        cell=cell,
                        actual=actual,
                        metric_name=str(row["metric_name"]),
                        point_estimate=float(row["point_estimate"]),
                        reducer_method=str(row["reducer_method"]),
                        independent_block_count=int(row["independent_block_count"]),
                        request_count=int(row["request_count"]),
                        paired=bool(row["paired"]),
                        attributes=row["attributes"],  # type: ignore[arg-type]
                    )
                continue
            if "raw_profile_size_bytes" in payload:
                size = payload["raw_profile_size_bytes"]
                if type(size) is not int or size < 1:
                    raise ValueError("profiler descriptive size is invalid")
                append(
                    cell=cell,
                    actual=actual,
                    metric_name="profiler_raw_profile_size_bytes",
                    point_estimate=float(size),
                    reducer_method="profiler_terminal_descriptive_v1",
                    attributes={
                        "profiler_variant": payload.get("profiler_variant"),
                        "raw_profile_sha256": payload.get("raw_profile_sha256"),
                    },
                )
            if "diagnostic_status" in payload:
                status = payload["diagnostic_status"]
                if status not in {"PASS", "FAIL"}:
                    raise ValueError("failure diagnostic status differs")
                append(
                    cell=cell,
                    actual=actual,
                    metric_name="failure_diagnostic_pass_indicator",
                    point_estimate=1.0 if status == "PASS" else 0.0,
                    reducer_method="failure_lifecycle_terminal_v1",
                    attributes={
                        "diagnostic_status": status,
                        "failure": payload.get("failure"),
                        "topology": payload.get("topology"),
                        "cohort_count": payload.get("cohort_count"),
                        "process_exit_code": payload.get("process_exit_code"),
                    },
                )
            if cell.task == "immutable_metadata_interface_and_fit_preflight":
                append(
                    cell=cell,
                    actual=actual,
                    metric_name="e6_interface_fit_pass_indicator",
                    point_estimate=1.0,
                    reducer_method="e6_interface_fit_terminal_v1",
                    attributes={
                        "interface_sha256": payload.get("interface_sha256"),
                        "verified_authority_sha256": payload.get(
                            "verified_authority_sha256"
                        ),
                        "trust_mode": payload.get("trust_mode"),
                    },
                )
            if "disposition" in payload:
                disposition = payload["disposition"]
                if disposition not in {"VALID", "N/A"}:
                    raise ValueError("compatibility disposition differs")
                append(
                    cell=cell,
                    actual=actual,
                    metric_name="e0_compatibility_valid_indicator",
                    point_estimate=1.0 if disposition == "VALID" else 0.0,
                    reducer_method="e0_compatibility_decision_v1",
                    attributes={
                        "disposition": disposition,
                        "reason_code": payload.get("reason_code"),
                        "decision_id": payload.get("decision_id"),
                        "interface_sha256": payload.get("interface_sha256"),
                    },
                )
        unique: dict[tuple[str, int, str, str], MetricRecord] = {}
        for metric in output:
            key = (
                metric.cell_id,
                metric.attempt,
                metric.metric_name,
                _semantic_sha256(metric.attributes),
            )
            prior = unique.get(key)
            if prior is not None and prior != metric:
                raise ValueError("actual metric identity collision")
            unique[key] = metric
        return tuple(unique[key] for key in sorted(unique))

    def _interference_metrics(
        self,
        *,
        rebuilt: Any,
        diagnostic_binding: ControllerArtifactBinding,
    ) -> tuple[MetricRecord, ...]:
        """Project only defined paired-BCa summaries from the sealed diagnostic."""

        value = _read_canonical_json(
            diagnostic_binding.absolute_path,
            label="fresh interference diagnostic metric source",
        )
        if (
            ControllerArtifactBinding.bind(diagnostic_binding.absolute_path)
            != diagnostic_binding
        ):
            raise ValueError("fresh interference diagnostic changed")
        diagnostic = value["diagnostic"]
        assert isinstance(diagnostic, dict)
        anchor = str(value["metric_anchor_cell_id"])
        if anchor not in {cell.cell_id for cell in rebuilt.materialization.cells}:
            raise ValueError("fresh interference metric anchor is foreign")
        attempt = self._completed_actual_attempt(anchor)
        common = {
            "confidence_level": 0.95,
            "diagnostic_evidence_sha256": diagnostic_binding.sha256,
            "diagnostic_status": value["status"],
            "scheduler_mode": value["scheduler_mode"],
            "reason_codes": diagnostic.get("reason_codes"),
            "paired_unit": "repetition",
            "paired_trace_request_count": value["paired_trace_request_count"],
            "observation_request_count": value["observation_request_count"],
            "completed_request_count": value["completed_request_count"],
            "request_counts_by_observation": value["request_counts_by_observation"],
        }

        def metric(prefix: str) -> MetricRecord | None:
            point = diagnostic.get(f"{prefix}_mean_relative_difference")
            low = diagnostic.get(f"{prefix}_ci_lower_relative_difference")
            high = diagnostic.get(f"{prefix}_ci_upper_relative_difference")
            if point is None or low is None or high is None:
                if any(item is not None for item in (point, low, high)):
                    raise ValueError("fresh interference interval is partial")
                return None
            return MetricRecord(
                stage=rebuilt.decision.stage,
                phase=rebuilt.decision.phase,
                cell_id=anchor,
                attempt=attempt,
                metric_name=f"interference_{prefix}_relative_difference",
                metric_kind="headline",
                point_estimate=float(point),
                ci_low=float(low),
                ci_high=float(high),
                independent_block_count=int(value["independent_block_count"]),
                request_count=int(value["observation_request_count"]),
                paired=True,
                reducer_method=str(value["reducer_method"]),
                attributes={**common, "interference_metric": prefix},
            )

        goodput = metric("goodput")
        if goodput is None:
            raise ValueError("fresh interference goodput interval is unavailable")
        p99 = metric("p99_itl")
        return (goodput,) if p99 is None else (goodput, p99)

    def _metric_anchor(
        self, materialization: Any, context: Mapping[str, Any]
    ) -> tuple[str, int]:
        criteria: dict[str, object] = {}
        raw_dimensions = context.get("dimensions")
        if type(raw_dimensions) is list:
            if any(
                type(row) is not list or len(row) != 2 or type(row[0]) is not str
                for row in raw_dimensions
            ):
                raise ValueError("reducer metric family dimensions differ")
            criteria.update({str(row[0]): row[1] for row in raw_dimensions})
        for name in ("compatibility_decision_id", "load", "anchor_id"):
            if name in context:
                target = "p99_anchor_id" if name == "anchor_id" else name
                criteria[target] = context[name]
        candidates = []
        for cell in materialization.cells:
            dimensions = dict(cell.dimensions)
            if cell.method_role != "LightCone" or any(
                dimensions.get(name) != value for name, value in criteria.items()
            ):
                continue
            latest = self.store.latest_attempt(cell.cell_id)
            if latest is None or latest["status"] != "COMPLETE":
                continue
            block = dimensions.get("block")
            candidates.append(
                (
                    block if type(block) is int else 1 << 30,
                    cell.cell_id,
                    int(latest["attempt"]),
                )
            )
        if not candidates:
            # Some non-serving descriptive reducers have no LightCone role.
            for cell in materialization.cells:
                latest = self.store.latest_attempt(cell.cell_id)
                if latest is not None and latest["status"] == "COMPLETE":
                    candidates.append((1 << 30, cell.cell_id, int(latest["attempt"])))
        if not candidates:
            raise ValueError("reducer metric has no COMPLETE provenance anchor")
        _, cell_id, attempt = min(candidates)
        return cell_id, attempt

    def _reducer_metrics(self, rebuilt: Any) -> tuple[MetricRecord, ...]:
        """Project only explicit reducer intervals; never invent a confidence band."""

        decision = rebuilt.decision
        output: list[MetricRecord] = []

        def add(
            *,
            context: Mapping[str, Any],
            metric_name: str,
            point: object,
            ci_low: object,
            ci_high: object,
            block_count: object,
            request_count: object,
            paired: object,
            reducer: object,
            attributes: Mapping[str, Any],
        ) -> None:
            if (
                type(metric_name) is not str
                or type(point) not in {int, float}
                or type(ci_low) not in {int, float}
                or type(ci_high) not in {int, float}
                or type(block_count) is not int
                or block_count < 1
                or type(request_count) is not int
                or request_count < 1
                or type(paired) is not bool
                or type(reducer) is not str
                or not reducer
            ):
                raise ValueError("explicit reducer metric fields differ")
            normalized_attributes = dict(attributes)
            confidence = normalized_attributes.get(
                "confidence_level",
                normalized_attributes.get("confidence"),
            )
            if type(confidence) not in {int, float} or float(confidence) != 0.95:
                raise ValueError("headline reducer metric is not a 95% interval")
            normalized_attributes["confidence_level"] = 0.95
            cell_id, attempt = self._metric_anchor(rebuilt.materialization, context)
            output.append(
                MetricRecord(
                    stage=decision.stage,
                    phase=decision.phase,
                    cell_id=cell_id,
                    attempt=attempt,
                    metric_name=metric_name,
                    metric_kind="headline",
                    point_estimate=float(point),
                    ci_low=float(ci_low),
                    ci_high=float(ci_high),
                    independent_block_count=block_count,
                    request_count=request_count,
                    paired=paired,
                    reducer_method=reducer,
                    attributes={
                        "decision_sha256": decision.sha256,
                        **normalized_attributes,
                    },
                )
            )

        def walk(value: object, inherited: Mapping[str, Any]) -> None:
            if type(value) is list:
                for item in value:
                    walk(item, inherited)
                return
            if type(value) is not dict:
                return
            context = dict(inherited)
            for name in (
                "dimensions",
                "compatibility_decision_id",
                "load",
                "anchor_id",
                "block_count",
                "independent_block_count",
                "request_count",
                "paired",
                "confidence",
                "reducer",
                "registered_reducer_method",
                "family_sha256",
                "result_sha256",
            ):
                if name in value:
                    context[name] = value[name]
            direct = {
                "point_estimate",
                "ci_low",
                "ci_high",
                "metric_name",
                "reducer_method",
            }
            if direct.issubset(value):
                add(
                    context=context,
                    metric_name=str(value["metric_name"]),
                    point=value["point_estimate"],
                    ci_low=value["ci_low"],
                    ci_high=value["ci_high"],
                    block_count=context.get("independent_block_count"),
                    request_count=context.get("request_count"),
                    paired=context.get("paired"),
                    reducer=value["reducer_method"],
                    attributes={
                        "statistical_scale": "native_metric",
                        "confidence": context.get("confidence"),
                        "family": {
                            name: context[name]
                            for name in (
                                "dimensions",
                                "compatibility_decision_id",
                                "load",
                                "anchor_id",
                                "family_sha256",
                                "result_sha256",
                            )
                            if name in context
                        },
                    },
                )
                return
            contrast = {
                "name",
                "block_ids",
                "mean_log_ratio",
                "mean_relative_gain",
                "ci_lower_relative_gain",
                "ci_upper_relative_gain",
                "raw_p_value",
                "confidence",
                "independent_unit",
            }
            if contrast.issubset(value):
                block_ids = value["block_ids"]
                if type(block_ids) is not list:
                    raise TypeError("paired reducer block IDs must be an array")
                add(
                    context=context,
                    metric_name=f"goodput_relative_gain/{value['name']}",
                    point=value["mean_relative_gain"],
                    ci_low=value["ci_lower_relative_gain"],
                    ci_high=value["ci_upper_relative_gain"],
                    block_count=len(block_ids),
                    request_count=context.get("request_count"),
                    paired=True,
                    reducer=context.get("reducer", "paired_block_bca"),
                    attributes={
                        "statistical_scale": "paired_relative_gain",
                        "mean_log_ratio": value["mean_log_ratio"],
                        "raw_p_value": value["raw_p_value"],
                        "confidence": value["confidence"],
                        "independent_unit": value["independent_unit"],
                        "block_ids": block_ids,
                        "family": {
                            name: context[name]
                            for name in (
                                "dimensions",
                                "compatibility_decision_id",
                                "load",
                                "family_sha256",
                                "result_sha256",
                            )
                            if name in context
                        },
                    },
                )
                return
            hierarchical = {
                "name",
                "mean_log_ratio",
                "mean_relative_gain",
                "ci_lower_relative_gain",
                "ci_upper_relative_gain",
                "confidence",
                "repetitions",
                "independent_units",
            }
            if hierarchical.issubset(value):
                add(
                    context=context,
                    metric_name=(f"hierarchical_goodput_relative_gain/{value['name']}"),
                    point=value["mean_relative_gain"],
                    ci_low=value["ci_lower_relative_gain"],
                    ci_high=value["ci_upper_relative_gain"],
                    block_count=context.get("block_count"),
                    request_count=context.get("request_count"),
                    paired=True,
                    reducer="hierarchical_block_request_bootstrap",
                    attributes={
                        "statistical_scale": "hierarchical_relative_gain",
                        "mean_log_ratio": value["mean_log_ratio"],
                        "confidence": value["confidence"],
                        "bootstrap_repetitions": value["repetitions"],
                        "independent_units": value["independent_units"],
                        "family": {
                            name: context[name]
                            for name in (
                                "dimensions",
                                "family_sha256",
                                "result_sha256",
                            )
                            if name in context
                        },
                    },
                )
                return
            for item in value.values():
                walk(item, context)

        explicit = _explicit_headline_metric_payload_rows(
            rebuilt.artifact.node,
            decision.payload,
        )
        if explicit is None:
            walk(decision.payload, {})
        else:
            for row in explicit:
                add(
                    context=row["context"],
                    metric_name=row["metric_name"],
                    point=row["point_estimate"],
                    ci_low=row["ci_low"],
                    ci_high=row["ci_high"],
                    block_count=row["block_count"],
                    request_count=row["request_count"],
                    paired=row["paired"],
                    reducer=row["reducer_method"],
                    attributes={
                        "statistical_scale": "registered_final_reducer",
                        "family": row["context"],
                        **row["attributes"],
                    },
                )
        unique: dict[tuple[str, int, str, str], MetricRecord] = {}
        for metric in output:
            key = (
                metric.cell_id,
                metric.attempt,
                metric.metric_name,
                _semantic_sha256(metric.attributes),
            )
            previous = unique.get(key)
            if previous is not None and previous != metric:
                raise ValueError("reducer metric identity collision")
            unique[key] = metric
        return tuple(unique[key] for key in sorted(unique))

    def _record_metric_once(self, metric: MetricRecord, *, recorded_at_ns: int) -> None:
        self._record_metrics_once((metric,), recorded_at_ns=recorded_at_ns)

    def _record_metrics_once(
        self,
        metrics: Sequence[MetricRecord],
        *,
        recorded_at_ns: int,
    ) -> None:
        """Atomically insert a large stage projection or exactly replay it.

        Calling ``ExperimentOperatorStore.record_metric`` once per serving
        metric would force one FULL-synchronous SQLite commit per row.  Late
        stages can contain more than 10^5 cells, so the exact same invariants
        are checked here under one store transaction and one WAL commit.
        """

        rows = tuple(metrics)
        if any(type(metric) is not MetricRecord for metric in rows):
            raise TypeError("metric batch requires exact MetricRecord rows")
        recorded = self.store._validated_time(recorded_at_ns)
        identities = tuple(
            (
                metric.cell_id,
                metric.attempt,
                metric.metric_name,
                json.dumps(
                    dict(metric.attributes),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            )
            for metric in rows
        )
        if len(set(identities)) != len(identities):
            raise ValueError("metric batch contains duplicate identities")
        with self.store._transaction():
            for metric, identity in zip(rows, identities, strict=True):
                attempt = self.store._require_attempt(
                    metric.cell_id,
                    metric.attempt,
                )
                if attempt["status"] != "COMPLETE":
                    raise ValueError("metrics require a COMPLETE attempt")
                if (attempt["stage"], attempt["phase"]) != (
                    metric.stage,
                    metric.phase,
                ):
                    raise ValueError("metric stage/phase differs from its attempt")
                existing = self.store._connection.execute(
                    "SELECT * FROM metrics_long WHERE cell_id = ? AND attempt = ? "
                    "AND metric_name = ? AND attributes_json = ?",
                    identity,
                ).fetchone()
                expected = (
                    metric.stage,
                    metric.phase,
                    metric.metric_kind,
                    metric.point_estimate,
                    metric.ci_low,
                    metric.ci_high,
                    metric.independent_block_count,
                    metric.request_count,
                    None if metric.paired is None else int(metric.paired),
                    metric.reducer_method,
                    recorded,
                )
                if existing is not None:
                    observed = (
                        existing["stage"],
                        existing["phase"],
                        existing["metric_kind"],
                        existing["point_estimate"],
                        existing["ci_low"],
                        existing["ci_high"],
                        existing["independent_block_count"],
                        existing["request_count"],
                        existing["paired"],
                        existing["reducer_method"],
                        existing["recorded_at_ns"],
                    )
                    if observed != expected:
                        raise ExperimentOperatorError(
                            "retained metric differs from replay"
                        )
                    continue
                self.store._connection.execute(
                    """
                    INSERT INTO metrics_long (
                        stage, phase, cell_id, attempt, metric_name, metric_kind,
                        point_estimate, ci_low, ci_high, independent_block_count,
                        request_count, paired, reducer_method, attributes_json,
                        recorded_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metric.stage,
                        metric.phase,
                        metric.cell_id,
                        metric.attempt,
                        metric.metric_name,
                        metric.metric_kind,
                        metric.point_estimate,
                        metric.ci_low,
                        metric.ci_high,
                        metric.independent_block_count,
                        metric.request_count,
                        None if metric.paired is None else int(metric.paired),
                        metric.reducer_method,
                        identity[3],
                        recorded,
                    ),
                )

    def reduce(
        self,
        node: str,
        node_materialization: ControllerArtifactBinding,
        actual_result_paths: Mapping[str, str],
    ) -> DagReduction:
        from lightcone_spec.experiments.formal_single_operator_stages import (
            FormalSingleOperatorStageBlocked,
            rebuild_formal_single_operator_stage_completion,
            reduce_formal_single_operator_node,
        )

        root = self._node_root(node) / "reduction"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        decision_path = root / "decision.json"
        completion_path = root / "completion.json"
        if decision_path.exists() != completion_path.exists():
            _preserve_partial_directory(
                root,
                label=f"{node} partial reduction",
            )
            root.mkdir(mode=0o700)
            decision_path = root / "decision.json"
            completion_path = root / "completion.json"
        if not completion_path.exists():
            try:
                reduce_formal_single_operator_node(
                    node_materialization_path=node_materialization.absolute_path,
                    actual_result_paths=actual_result_paths,
                    repository_root=self.config.repository_root,
                    decision_output_path=decision_path,
                    completion_output_path=completion_path,
                    completed_ns=self._clock(root, "reduction-clock.json"),
                )
            except FormalSingleOperatorStageBlocked as error:
                raise FormalExperimentDagBlocked(
                    f"{node}: scientific reduction blocked: {error}"
                ) from error
        rebuilt = rebuild_formal_single_operator_stage_completion(completion_path)
        if (
            rebuilt.artifact.node != node
            or rebuilt.artifact.node_materialization_source.absolute_path
            != node_materialization.absolute_path
            or {
                row.cell_id: row.source.absolute_path
                for row in rebuilt.artifact.actual_results
            }
            != dict(actual_result_paths)
        ):
            raise ExperimentOperatorError("rebuilt reduction differs from actual paths")
        reduction = DagReduction(
            decision=ControllerArtifactBinding.bind(decision_path),
            completion=ControllerArtifactBinding.bind(completion_path),
        )
        self._record_selection_once(rebuilt, decision_path)
        projected_metrics = [
            *self._actual_metrics(rebuilt),
            *self._reducer_metrics(rebuilt),
        ]
        envelope = None
        if node == "preflight":
            envelope = self.interference_gate_resolver.resolve(
                completion=reduction.completion,
                actual_result_paths=actual_result_paths,
                gpu_uuids=self._inventory_gpu_uuids(),
            )
            fresh_diagnostic = None
            if type(self.interference_gate_resolver) is (
                FreshPreflightInterferenceGateResolver
            ):
                fresh_diagnostic = self.interference_gate_resolver.diagnostic_binding(
                    reduction.completion
                )
                projected_metrics.extend(
                    self._interference_metrics(
                        rebuilt=rebuilt,
                        diagnostic_binding=fresh_diagnostic,
                    )
                )
            if (
                type(envelope) is not InterferenceEnvelope
                or envelope.mode not in {"ISOLATED", "DUAL_SINGLE"}
                or envelope.gpu_uuids != self._inventory_gpu_uuids()
                or envelope.evidence_sha256
                not in {
                    reduction.completion.sha256,
                    *(_file_sha256(path) for path in actual_result_paths.values()),
                    *(() if fresh_diagnostic is None else (fresh_diagnostic.sha256,)),
                }
            ):
                raise ValueError(
                    "post-preflight interference resolver returned unsafe authority"
                )
        self._record_metrics_once(
            projected_metrics,
            recorded_at_ns=rebuilt.artifact.completed_ns,
        )
        if envelope is not None:
            self.store.configure_interference_envelope(envelope)
        self._publish_retained_dependency_manifest(
            node=node,
            reduction=reduction,
            actual_result_paths=actual_result_paths,
        )
        return reduction

    def _publish_retained_dependency_manifest(
        self,
        *,
        node: str,
        reduction: DagReduction,
        actual_result_paths: Mapping[str, str],
    ) -> RetainedFutureDependencyManifest:
        row = self.store.controller_node(node)
        retained_paths = {
            reduction.decision.absolute_path,
            reduction.completion.absolute_path,
            self.config.protocol_lock.absolute_path,
            self.config.content_source.absolute_path,
            self.config.runtime_authority_manifest.absolute_path,
            self.config.inventory.absolute_path,
            self.config.doctor_report.absolute_path,
            self.config.preflight_workload_authority.absolute_path,
            *(item.absolute_path for item in self.config.profiler_tools),
            *actual_result_paths.values(),
        }
        if self.config.session_reset_authority_directory is not None:
            retained_paths.update(
                str(path)
                for path in Path(self.config.session_reset_authority_directory).glob(
                    "*.json"
                )
                if path.is_file() and not path.is_symlink()
            )
        if node == "preflight":
            diagnostic_path = FreshPreflightInterferenceGateResolver.diagnostic_path(
                reduction.completion
            )
            if diagnostic_path.exists():
                retained_paths.add(str(diagnostic_path))
        for name in (
            "materialization_path",
            "node_materialization_path",
            "execution_source_path",
            "prepared_launch_path",
        ):
            path = row.get(name)
            if type(path) is str:
                retained_paths.add(path)
        latest_auxiliary = self.store.latest_controller_auxiliary_group(node)
        archive_roots = {str(self._node_root(node) / "execution" / "work")}
        retained_transitive_roots: set[str] = set()
        if latest_auxiliary is not None:
            publication = latest_auxiliary.get("publication_path")
            if type(publication) is str:
                retained_paths.add(publication)
            output = latest_auxiliary.get("output_directory")
            if type(output) is str:
                # E6/E0 auxiliary materialization deep-replays the campaign and
                # its raw proofs on every downstream predecessor rebuild.
                retained_transitive_roots.add(output)
        execution_path = row.get("execution_source_path")
        if type(execution_path) is str:
            from lightcone_spec.experiments.formal_single_operator_stages import (
                load_formal_single_operator_execution_source,
            )

            source = load_formal_single_operator_execution_source(execution_path)
            for catalog_path in sorted(
                Path(self.config.prerequisite_index_catalog_directory).glob("*.json")
            ):
                try:
                    catalog = _read_canonical_json(
                        catalog_path,
                        label="retained prerequisite catalog binding",
                    )
                except (OSError, ValueError):
                    continue
                if catalog.get("kind") != _PREREQUISITE_BINDING_KIND:
                    continue
                binding = PrerequisiteIndexCatalogBinding.from_dict(catalog)
                if (
                    binding.node == node
                    and binding.execution_source_sha256 == source.sha256
                ):
                    retained_paths.add(str(catalog_path))
                    retained_paths.add(binding.prerequisite_index_path)
        files = tuple(
            sorted(
                (DriverFileBinding.bind(path) for path in retained_paths),
                key=lambda item: item.absolute_path,
            )
        )
        manifest = RetainedFutureDependencyManifest(
            schema_version=1,
            kind=_RETAINED_DEPENDENCY_MANIFEST_KIND,
            run_id=Path(self.config.run_root).name,
            run_root=self.config.run_root,
            node=node,
            completion=reduction.completion,
            decision=reduction.decision,
            retained_files=files,
            retained_transitive_roots=tuple(sorted(retained_transitive_roots)),
            archive_candidate_roots=tuple(sorted(archive_roots)),
            archive_safe_after_reduction=True,
            remote_eviction_authorized_for_nonretained_files=True,
            remote_eviction_scope=(
                "archive_candidate_roots_excluding_retained_files_and_transitive_roots"
            ),
            eviction_preconditions=(
                "local_sha_manifest_verified",
                "local_rehydrate_test_passed",
            ),
            transitive_evidence_must_rehydrate_at_original_paths=True,
        )
        output = self._retained_manifest_path(node)
        if output.exists():
            rebound = load_retained_future_dependency_manifest(output)
            if rebound != manifest:
                raise FormalExperimentDagBlocked(
                    f"{node}: retained dependency manifest changed"
                )
        else:
            _publish_no_replace(output, manifest.to_dict())
        return manifest

    def auxiliary_plan(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
    ) -> AuxiliaryPhysicalGroupSpec | None:
        if self.auxiliary_runtime is None:
            raise FormalExperimentDagBlocked(
                f"{node}: pre-materialization GPU auxiliary executor is unavailable"
            )
        return self.auxiliary_runtime.plan(node, predecessor)

    def auxiliary_launch(self, spec: AuxiliaryPhysicalGroupSpec) -> SpawnedProcess:
        if self.auxiliary_runtime is None:
            raise FormalExperimentDagBlocked(
                f"{spec.node}: GPU auxiliary executor is unavailable"
            )
        return self.auxiliary_runtime.launch(spec)

    def auxiliary_terminal(
        self,
        spec: AuxiliaryPhysicalGroupSpec,
        durable_group: Mapping[str, Any],
    ) -> AuxiliaryGroupTerminal | None:
        if self.auxiliary_runtime is None:
            raise FormalExperimentDagBlocked(
                f"{spec.node}: GPU auxiliary terminal validator is unavailable"
            )
        return self.auxiliary_runtime.terminal(spec, durable_group)

    def auxiliary_adoptions(
        self,
        node: str,
        node_materialization: ControllerArtifactBinding,
        spec: AuxiliaryPhysicalGroupSpec,
    ) -> tuple[AuxiliaryCellAdoption, ...]:
        if self.auxiliary_runtime is None:
            raise FormalExperimentDagBlocked(
                f"{node}: GPU auxiliary adoption mapper is unavailable"
            )
        return self.auxiliary_runtime.adoptions(node, node_materialization, spec)


def build_production_formal_dag_driver(
    config_path: str | Path,
    *,
    prerequisite_resolver: PrerequisiteIndexResolver | None = None,
    e5_arrival_resolver: E5ArrivalPlanResolver | None = None,
    auxiliary_runtime: AuxiliaryPhysicalRuntime | None = None,
    interference_gate_resolver: InterferenceGateResolver | None = None,
) -> FormalSingleOperatorDagDriver:
    """Open schema-6 WAL state and compose the production callback/runtime set."""

    from lightcone_spec.experiments.gpu_pool import GpuInventory
    from lightcone_spec.orchestration.experiment_operator_production import (
        ProductionSchedulerRuntime,
    )

    config = load_path_bound_formal_dag_driver_config(config_path)
    database_path = Path(config.run_root) / "operator.sqlite3"
    store = ExperimentOperatorStore(
        database_path,
        run_id=Path(config.run_root).name,
    )
    store.initialize_stage_plan(default_formal_stage_plan())
    inventory = GpuInventory.from_dict(
        _read_canonical_json(config.inventory.absolute_path, label="GPU inventory")
    )
    gpu_uuids = tuple(sorted(row.uuid for row in inventory.devices if row.ready))
    if len(gpu_uuids) != 2:
        store.close()
        raise ValueError("formal DAG production driver requires exactly two ready GPUs")
    try:
        store.interference_envelope()
    except ExperimentOperatorError:
        store.configure_interference_envelope(
            InterferenceEnvelope(
                "UNRESOLVED",
                gpu_uuids,
                config.inventory.raw_sha256,
            )
        )
    builder = ProductionFormalDagCallbackBuilder(
        config=config,
        store=store,
        prerequisite_resolver=prerequisite_resolver,
        e5_arrival_resolver=e5_arrival_resolver,
        auxiliary_runtime=(
            auxiliary_runtime
            or DirectoryAuxiliaryPhysicalRuntime(config=config, store=store)
        ),
        interference_gate_resolver=interference_gate_resolver,
    )
    lock_path = Path(config.run_root) / "formal-dag-driver.lock"
    scheduler = FormalExperimentSchedulerDaemon(
        store,
        lock_path=lock_path,
        callbacks=ProductionSchedulerRuntime(
            retry_builder=builder.retry_attempt
        ).callbacks(),
    )
    return FormalSingleOperatorDagDriver(
        store=store,
        callbacks=builder.callbacks(),
        scheduler=scheduler,
        lock_path=lock_path,
        progress_root=Path(config.run_root) / "results" / "progress",
    )


def _publish_auxiliary_worker_terminal(
    *,
    descriptor: AuxiliaryWorkerDescriptor,
    status: Literal["COMPLETE", "FAILED"],
    exit_code: int,
    started_ns: int,
    finished_ns: int,
    publication: ControllerArtifactBinding,
    failure_code: str | None,
    failure_class: str | None,
    failure_detail: str | None,
) -> None:
    value = {
        "schema_version": 2,
        "kind": _AUXILIARY_WORKER_TERMINAL_KIND,
        "descriptor_sha256": descriptor.sha256,
        "node": descriptor.node,
        "attempt": descriptor.attempt,
        "status": status,
        "exit_code": exit_code,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "publication": asdict(publication),
        "failure_code": failure_code,
        "failure_class": failure_class,
        "failure_detail": failure_detail,
    }
    _publish_no_replace(descriptor.terminal_output_path, value)


def _resolve_completed_e0_onlinespec_authority(
    *,
    descriptor: AuxiliaryWorkerDescriptor,
) -> object | None:
    """Select the E0 OnlineSPEC authority only after all 108 real probes.

    The input catalog may bind the source authority before the compatibility
    campaign starts, because its eventual VALID count is not yet known.  The
    scientific bundle must nevertheless claim that authority iff at least one
    completed probe is VALID.  This helper deep-opens the complete terminal
    set first, fails closed when a VALID campaign lacks the authority, and
    publishes a small audit record when a pre-bound authority is deliberately
    unused by an all-N/A campaign.
    """

    if descriptor.node != "e0_tuning":
        raise ValueError("OnlineSPEC authority selection requires E0 tuning")
    from lightcone_spec.experiments.formal_registry import (
        e0_onlinespec_source_authority_from_dict,
    )
    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        load_e0_compatibility_probe_terminal,
    )
    from lightcone_spec.orchestration.formal_e0_compatibility_physical import (
        completed_e0_compatibility_terminal_paths,
    )

    terminal_paths = completed_e0_compatibility_terminal_paths(
        descriptor.campaign.absolute_path
    )
    terminals = tuple(
        load_e0_compatibility_probe_terminal(path) for path in terminal_paths
    )
    if len(terminals) != 108 or len({row.key for row in terminals}) != 108:
        raise ValueError("E0 authority selection lacks exact 108 terminals")
    valid_count = sum(row.disposition == "VALID" for row in terminals)
    bound = descriptor.onlinespec_source_authority
    if valid_count:
        if bound is None:
            raise ValueError(
                "VALID E0 compatibility decisions require bound OnlineSPEC authority"
            )
        authority = e0_onlinespec_source_authority_from_dict(
            _read_canonical_json(
                bound.absolute_path,
                label="OnlineSPEC source authority",
            )
        )
        authority.revalidate()
        disposition = "USED_VALID_PRESENT"
    else:
        authority = None
        disposition = "BOUND_UNUSED_ALL_NA" if bound is not None else "UNBOUND_ALL_NA"
    record = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_onlinespec_authority_disposition",
        "descriptor_sha256": descriptor.sha256,
        "campaign_sha256": descriptor.campaign.raw_sha256,
        "probe_terminal_sha256s": [row.sha256 for row in terminals],
        "valid_decision_count": valid_count,
        "bound_authority": None if bound is None else bound.to_dict(),
        "disposition": disposition,
        "claimed_by_compatibility_bundle": bool(valid_count),
    }
    record_path = Path(descriptor.publication_output_path).with_name(
        "onlinespec-source-authority-disposition.json"
    )
    if record_path.exists():
        if (
            _read_canonical_json(
                record_path,
                label="E0 OnlineSPEC authority disposition",
            )
            != record
        ):
            raise ValueError("E0 OnlineSPEC authority disposition changed")
    else:
        _publish_no_replace(record_path, record)
    return authority


def _auxiliary_failure_class(error: BaseException) -> str:
    """Classify auxiliary failures without turning scientific faults into retries."""

    name = type(error).__name__
    detail = str(error).lower()
    if isinstance(error, (OSError, TimeoutError, subprocess.TimeoutExpired)):
        return "INFRASTRUCTURE"
    if name == "FormalSingleOperatorE6InterfaceFitBlocked":
        if "timeout" in detail or "unavailable" in detail:
            return "INFRASTRUCTURE"
        if "exact" in detail or "junit" in detail or "pytest" in detail:
            return "EXACTNESS"
        return "SCIENTIFIC"
    if name == "FormalE0CompatibilityPhysicalBlocked":
        infrastructure_markers = (
            "runtime",
            "server",
            "transport",
            "timeout",
            "gpu",
            "port",
            "cleanup",
        )
        return (
            "INFRASTRUCTURE"
            if any(marker in detail for marker in infrastructure_markers)
            else "SCIENTIFIC"
        )
    return "SCIENTIFIC"


def _run_auxiliary_worker(descriptor_path: str | Path) -> int:
    """Execute one source-owned campaign; used only by the path-bound CLI."""

    descriptor = AuxiliaryWorkerDescriptor.from_dict(
        _read_canonical_json(descriptor_path, label="auxiliary worker descriptor")
    )
    from lightcone_spec.orchestration.experiment_operator_production import (
        OperatorTerminalContext,
    )
    from lightcone_spec.orchestration.formal_cell_worker import (
        ChildHeartbeatPublisher,
    )

    expected_environment = {
        "LIGHTCONE_AUXILIARY_GROUP_ID": descriptor.group_id,
        "LIGHTCONE_AUXILIARY_ATTEMPT": str(descriptor.attempt),
        "LIGHTCONE_AUXILIARY_COMMAND_SHA256": _semantic_sha256(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "lightcone_spec.orchestration.formal_single_operator_dag_driver",
                    "auxiliary-worker",
                    "--descriptor",
                    str(Path(descriptor_path).resolve(strict=False)),
                ]
            }
        ),
        "LIGHTCONE_AUXILIARY_HEARTBEAT_PATH": descriptor.heartbeat_output_path,
    }
    if any(
        os.environ.get(name) != value for name, value in expected_environment.items()
    ):
        raise ValueError("auxiliary worker control environment differs")
    root = Path(descriptor.terminal_output_path).parent
    heartbeat = ChildHeartbeatPublisher(
        path=Path(descriptor.heartbeat_output_path),
        context=OperatorTerminalContext(
            cell_id=descriptor.group_id,
            attempt=descriptor.attempt,
            command_sha256=expected_environment["LIGHTCONE_AUXILIARY_COMMAND_SHA256"],
            expected_terminal_path=descriptor.terminal_output_path,
            expected_junit_path=str(root / "auxiliary-heartbeat-control.junit.xml"),
            expected_raw_log_path=str(root / "auxiliary-heartbeat-control.raw.json"),
            atomic_pointer_path=str(root / "auxiliary-heartbeat-control.pointer.json"),
        ),
        clock_ns=time.time_ns,
    )
    heartbeat.start()
    try:
        return _run_auxiliary_worker_bound(descriptor)
    finally:
        heartbeat.stop()


def _run_auxiliary_worker_bound(descriptor: AuxiliaryWorkerDescriptor) -> int:
    """Run the already deep-opened auxiliary descriptor under a heartbeat."""

    existing = DirectoryAuxiliaryPhysicalRuntime._worker_terminal(descriptor)
    if existing is not None:
        return 0 if existing["status"] == "COMPLETE" else 42
    retained_failure = Path(descriptor.terminal_output_path).with_name(
        "auxiliary-worker-failure.json"
    )
    if retained_failure.exists():
        value = _read_canonical_json(
            retained_failure, label="retained auxiliary worker failure"
        )
        expected = {
            "schema_version",
            "kind",
            "descriptor_sha256",
            "node",
            "attempt",
            "started_ns",
            "finished_ns",
            "failure_code",
            "failure_class",
            "failure_detail",
        }
        if (
            set(value) != expected
            or value["schema_version"] != 2
            or value["kind"] != "formal_single_operator_auxiliary_worker_failure"
            or value["descriptor_sha256"] != descriptor.sha256
            or value["node"] != descriptor.node
            or value["attempt"] != descriptor.attempt
            or value["failure_class"]
            not in {"INFRASTRUCTURE", "SCIENTIFIC", "EXACTNESS", "UNSAFE"}
        ):
            raise ValueError("retained auxiliary worker failure identity differs")
        publication = ControllerArtifactBinding.bind(retained_failure)
        _publish_auxiliary_worker_terminal(
            descriptor=descriptor,
            status="FAILED",
            exit_code=42,
            started_ns=int(value["started_ns"]),
            finished_ns=int(value["finished_ns"]),
            publication=publication,
            failure_code=str(value["failure_code"]),
            failure_class=str(value["failure_class"]),
            failure_detail=str(value["failure_detail"]),
        )
        return 42
    started_ns = time.time_ns()
    try:
        if descriptor.node == "e6_pilot":
            from lightcone_spec.experiments.formal_single_operator_e6_interface import (
                _load_campaign,
                execute_formal_single_operator_e6_interface_fit_plan,
                finalize_formal_single_operator_e6_interface_fit_bundle,
                revalidate_formal_single_operator_e6_interface_fit_bundle,
                revalidate_formal_single_operator_e6_interface_fit_plan,
            )

            campaign = _load_campaign(descriptor.campaign.absolute_path)
            for binding in campaign.plans:
                plan = revalidate_formal_single_operator_e6_interface_fit_plan(
                    binding.absolute_path
                )
                terminal_path = (
                    Path(plan.evidence_directory) / "e6-interface-fit-terminal.json"
                )
                if not terminal_path.exists():
                    execute_formal_single_operator_e6_interface_fit_plan(
                        binding.absolute_path
                    )
            publication_path = Path(descriptor.publication_output_path)
            if publication_path.exists():
                revalidate_formal_single_operator_e6_interface_fit_bundle(
                    publication_path
                )
            else:
                finalize_formal_single_operator_e6_interface_fit_bundle(
                    campaign_path=descriptor.campaign.absolute_path,
                    output_path=publication_path,
                )
        else:
            from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
                revalidate_trusted_e0_compatibility_bundle,
            )
            from lightcone_spec.orchestration.formal_e0_compatibility_physical import (
                execute_formal_e0_compatibility_probe_group,
                publish_completed_e0_compatibility_physical_campaign,
                revalidate_formal_e0_compatibility_physical_campaign,
            )

            campaign = revalidate_formal_e0_compatibility_physical_campaign(
                descriptor.campaign.absolute_path
            )
            # Group plans already carry a deterministic GPU UUID.  Execute one
            # fixed nine-task group at a time so no GPU can receive two servers.
            for binding in campaign.groups:
                execute_formal_e0_compatibility_probe_group(binding.absolute_path)
            authority = _resolve_completed_e0_onlinespec_authority(
                descriptor=descriptor,
            )
            publication_path = Path(descriptor.publication_output_path)
            if publication_path.exists():
                revalidate_trusted_e0_compatibility_bundle(publication_path)
            else:
                assert descriptor.evidence_manifest_output_path is not None
                publish_completed_e0_compatibility_physical_campaign(
                    campaign_path=descriptor.campaign.absolute_path,
                    protocol_lock_path=descriptor.protocol_lock.absolute_path,
                    e6_completion_path=(
                        descriptor.predecessor_completion.absolute_path
                    ),
                    onlinespec_source_authority=authority,
                    bundle_output_path=publication_path,
                    evidence_manifest_output_path=(
                        descriptor.evidence_manifest_output_path
                    ),
                )
        publication = ControllerArtifactBinding.bind(descriptor.publication_output_path)
        finished_ns = max(time.time_ns(), started_ns + 1)
        _publish_auxiliary_worker_terminal(
            descriptor=descriptor,
            status="COMPLETE",
            exit_code=0,
            started_ns=started_ns,
            finished_ns=finished_ns,
            publication=publication,
            failure_code=None,
            failure_class=None,
            failure_detail=None,
        )
        return 0
    except Exception as error:
        finished_ns = max(time.time_ns(), started_ns + 1)
        failure_path = Path(descriptor.terminal_output_path).with_name(
            "auxiliary-worker-failure.json"
        )
        failure_value = {
            "schema_version": 2,
            "kind": "formal_single_operator_auxiliary_worker_failure",
            "descriptor_sha256": descriptor.sha256,
            "node": descriptor.node,
            "attempt": descriptor.attempt,
            "started_ns": started_ns,
            "finished_ns": finished_ns,
            "failure_code": f"{type(error).__module__}.{type(error).__name__}",
            "failure_class": _auxiliary_failure_class(error),
            "failure_detail": str(error)[:2048] or type(error).__name__,
        }
        if not failure_path.exists():
            _publish_no_replace(failure_path, failure_value)
        elif (
            _read_canonical_json(failure_path, label="auxiliary worker failure")
            != failure_value
        ):
            # A retained failure belongs to another process outcome.  Do not
            # replace it or claim a terminal for the ambiguous attempt.
            raise
        publication = ControllerArtifactBinding.bind(failure_path)
        _publish_auxiliary_worker_terminal(
            descriptor=descriptor,
            status="FAILED",
            exit_code=42,
            started_ns=started_ns,
            finished_ns=finished_ns,
            publication=publication,
            failure_code=str(failure_value["failure_code"]),
            failure_class=str(failure_value["failure_class"]),
            failure_detail=str(failure_value["failure_detail"]),
        )
        return 42


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    operations = parser.add_subparsers(dest="operation", required=True)
    write = operations.add_parser("write-config", allow_abbrev=False)
    write.add_argument("--repository-root", required=True)
    write.add_argument("--run-root", required=True)
    write.add_argument("--protocol-lock", required=True)
    write.add_argument("--content-source", required=True)
    write.add_argument("--runtime-authority-manifest", required=True)
    write.add_argument("--inventory", required=True)
    write.add_argument("--doctor-report", required=True)
    write.add_argument("--preflight-workload-authority", required=True)
    write.add_argument("--profiler-tool", action="append", default=[])
    write.add_argument("--prerequisite-index-catalog", required=True)
    write.add_argument("--session-reset-authority-directory")
    write.add_argument("--output", required=True)
    bind_auxiliary = operations.add_parser("bind-auxiliary-inputs", allow_abbrev=False)
    bind_auxiliary.add_argument(
        "--node", required=True, choices=("e6_pilot", "e0_tuning")
    )
    bind_auxiliary.add_argument("--predecessor-completion", required=True)
    bind_auxiliary.add_argument("--input", action="append", required=True)
    bind_auxiliary.add_argument("--onlinespec-source-authority")
    bind_auxiliary.add_argument("--output", required=True)
    for name in ("once", "run", "status"):
        operation = operations.add_parser(name, allow_abbrev=False)
        operation.add_argument("--config", required=True)
    resume_node = operations.add_parser("resume-node", allow_abbrev=False)
    resume_node.add_argument("--config", required=True)
    resume_node.add_argument("--node", required=True)
    resume_node.add_argument("--reason", required=True)
    resume_dispatch = operations.add_parser("resume-dispatch", allow_abbrev=False)
    resume_dispatch.add_argument("--config", required=True)
    resume_dispatch.add_argument("--reason", required=True)
    resume_dispatch.add_argument("--manual-evidence")
    auxiliary_worker = operations.add_parser("auxiliary-worker", allow_abbrev=False)
    auxiliary_worker.add_argument("--descriptor", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "auxiliary-worker":
        return _run_auxiliary_worker(args.descriptor)
    if args.operation == "bind-auxiliary-inputs":
        binding = publish_auxiliary_input_catalog_binding(
            node=args.node,
            predecessor_completion_path=args.predecessor_completion,
            input_paths=tuple(args.input),
            onlinespec_source_authority_path=(args.onlinespec_source_authority),
            output_path=args.output,
        )
        print(json.dumps({"binding_sha256": binding.sha256}, sort_keys=True))
        return 0
    if args.operation == "write-config":
        config = publish_path_bound_formal_dag_driver_config(
            repository_root=args.repository_root,
            run_root=args.run_root,
            protocol_lock_path=args.protocol_lock,
            content_source_path=args.content_source,
            runtime_authority_manifest_path=args.runtime_authority_manifest,
            inventory_path=args.inventory,
            doctor_report_path=args.doctor_report,
            preflight_workload_authority_path=args.preflight_workload_authority,
            profiler_tool_paths=args.profiler_tool,
            prerequisite_index_catalog_directory=(args.prerequisite_index_catalog),
            session_reset_authority_directory=(args.session_reset_authority_directory),
            output_path=args.output,
        )
        print(json.dumps({"config_sha256": config.sha256}, sort_keys=True))
        return 0
    driver = build_production_formal_dag_driver(args.config)
    try:
        if args.operation == "resume-node":
            driver.resume_node(node=args.node, reason=args.reason)
            print(
                json.dumps(
                    {
                        "node": args.node,
                        "state": driver.store.controller_node(args.node)["state"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.operation == "resume-dispatch":
            driver.resume_dispatch(
                reason=args.reason,
                manual_evidence_path=args.manual_evidence,
            )
            state, reason = driver.store.dispatch_control()
            print(
                json.dumps(
                    {"dispatch_state": state, "dispatch_stop_reason": reason},
                    sort_keys=True,
                )
            )
            return 0
        if args.operation == "once":
            cycle = driver.run_once()
            print(json.dumps(cycle.to_dict(), sort_keys=True))
            if cycle.controller.action == "BLOCKED":
                return 42
            return 43 if cycle.controller.action == "COMPLETE" else 0
        if args.operation == "run":
            driver.run_forever()
            active = next(
                (
                    row
                    for row in driver.store.controller_nodes()
                    if row["state"] != "REDUCED"
                ),
                None,
            )
            return 43 if active is None else 42
        snapshot = driver.store.snapshot()
        awaiting_audit = all(
            row["state"] == "REDUCED" for row in snapshot["controller_nodes"]
        )
        print(
            json.dumps(
                {
                    "run_id": snapshot["run_id"],
                    "run_state": (
                        "DAG_REDUCED_AWAITING_FINAL_AUDIT"
                        if awaiting_audit
                        else "STOPPED"
                        if snapshot["dispatch_state"] == "STOP"
                        else "DAG_ACTIVE"
                    ),
                    "controller_nodes": snapshot["controller_nodes"],
                    "dispatch_state": snapshot["dispatch_state"],
                    "dispatch_stop_reason": snapshot["dispatch_stop_reason"],
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        driver.store.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuxiliaryInputCatalogBinding",
    "AuxiliaryPhysicalRuntime",
    "AuxiliaryWorkerDescriptor",
    "DirectoryAuxiliaryPhysicalRuntime",
    "DirectoryPrerequisiteIndexResolver",
    "DriverFileBinding",
    "E5ArrivalPlanResolver",
    "FormalDagDriverCycle",
    "FormalDagNodeCodeCapability",
    "FormalSingleOperatorDagDriver",
    "FreshPreflightInterferenceGateResolver",
    "InterferenceGateResolver",
    "IsolatedInterferenceGateResolver",
    "PathBoundFormalDagDriverConfig",
    "PrerequisiteIndexCatalogBinding",
    "PrerequisiteIndexResolver",
    "ProductionFormalDagCallbackBuilder",
    "RetainedFutureDependencyManifest",
    "build_production_formal_dag_driver",
    "formal_single_operator_dag_code_capabilities",
    "load_path_bound_formal_dag_driver_config",
    "load_retained_future_dependency_manifest",
    "main",
    "publish_auxiliary_input_catalog_binding",
    "publish_path_bound_formal_dag_driver_config",
    "publish_prerequisite_index_catalog_binding",
]
