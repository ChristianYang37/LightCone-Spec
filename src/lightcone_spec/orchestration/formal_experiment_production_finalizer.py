"""Production finalizer for the trusted single-operator formal campaign.

The DAG driver intentionally stops at ``DAG_REDUCED_AWAITING_FINAL_AUDIT``.
This module owns the remaining fail-closed sequence:

1. stop dispatch and restore any rolling-archive members needed at their
   original paths;
2. replay the complete 21-node scientific state before doing a whole-run
   TRANSFER -> LOCAL_SHA_VERIFY -> full REHYDRATE_VERIFY archive;
3. export current projections, observe a stable zero-writer shutdown window,
   and publish the immutable pre-shutdown audit;
4. issue at most one AutoDL ``power_off`` mutation, then use only status/list
   observations until shutdown is dually confirmed and billing is closed;
5. publish the trusted empirical final receipt plus an accounting breakdown.

The public CLI accepts paths only.  The AutoDL token is read from the fixed
``AUTODL_DEVELOPER_TOKEN`` environment variable or an injected in-memory
mapping and is never placed in configuration, argv, logs, or evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.orchestration.autodl_provider_runtime import (
    AutoDlProApiClient,
    transition_autodl_instance_power,
)
from lightcone_spec.orchestration.experiment_operator import (
    ArchiveRequest,
    ExperimentOperatorStore,
    SingletonOperatorLock,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    ProductionArchiveRuntime,
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_experiment_final_audit import (
    FINAL_ARCHIVE_SAFE_BOUNDARY,
    FinalAuditArtifactBinding,
    FormalExperimentFinalCompletionReceipt,
    FormalExperimentFinalizationReadiness,
    FormalExperimentPreShutdownAuditReceipt,
    audit_finalization_readiness,
    load_final_completion,
    load_pre_shutdown_audit,
    publish_final_completion,
    publish_pre_shutdown_audit,
)
from lightcone_spec.orchestration.formal_rolling_archive import (
    load_archive_request,
    restore_evicted_files,
)
from lightcone_spec.orchestration.formal_shutdown_probe import (
    collect_formal_shutdown_probe,
    shutdown_probe_is_safe,
)
from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    DriverFileBinding,
    load_path_bound_formal_dag_driver_config,
)
from lightcone_spec.runtime.proof_artifact import (
    publish_canonical_json_no_replace,
)

_CONFIG_KIND = "formal_experiment_production_finalizer_config"
_INSTANCE_KIND = "formal_autodl_instance_identity"
_PORTS_KIND = "formal_measurement_port_registry"
_RESTORE_KIND = "formal_finalization_rehydration_catalog"
_ACCOUNTING_KIND = "formal_experiment_final_accounting_breakdown"
_COMPLETION_KIND = "formal_experiment_production_finalizer_completion"
_TOKEN_ENVIRONMENT_NAME = "AUTODL_DEVELOPER_TOKEN"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class FormalExperimentProductionFinalizerError(RuntimeError):
    """The campaign cannot safely advance through final shutdown."""


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise FormalExperimentProductionFinalizerError(
            f"{label} is not a lowercase SHA-256"
        )
    return value


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise FormalExperimentProductionFinalizerError(
            f"{label} must be absolute and normalized"
        )
    return path


def _existing_file(value: str | Path, label: str) -> Path:
    path = _absolute(value, label)
    if path.is_symlink() or not path.is_file():
        raise FormalExperimentProductionFinalizerError(
            f"{label} is not one regular file"
        )
    return path


def _load_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _existing_file(path, label)
    payload = source.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalExperimentProductionFinalizerError(
            f"{label} is not JSON"
        ) from error
    if type(value) is not dict or payload != canonical_json_bytes(value):
        raise FormalExperimentProductionFinalizerError(f"{label} is not canonical JSON")
    return value


def _publish_or_replay(path: Path, value: object, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        if _load_canonical(path, label) != value:
            raise FormalExperimentProductionFinalizerError(
                f"{label} is immutable and differs"
            )
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    publish_canonical_json_no_replace(path, value)


@dataclass(frozen=True)
class FinalizationRestoreEntry:
    plan_path: str
    remote_archive_result_path: str
    receipt_path: str
    lock_path: str

    def __post_init__(self) -> None:
        _existing_file(self.plan_path, "finalization restore plan")
        _existing_file(
            self.remote_archive_result_path,
            "finalization remote archive result",
        )
        receipt = _absolute(self.receipt_path, "finalization restore receipt")
        lock = _absolute(self.lock_path, "finalization restore lock")
        if receipt == lock:
            raise FormalExperimentProductionFinalizerError(
                "restore receipt and lock paths overlap"
            )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise FormalExperimentProductionFinalizerError(
                "finalization restore entry fields differ"
            )
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FinalizationRehydrationCatalog:
    schema_version: Literal[1]
    kind: Literal["formal_finalization_rehydration_catalog"]
    entries: tuple[FinalizationRestoreEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _RESTORE_KIND:
            raise FormalExperimentProductionFinalizerError(
                "finalization restore catalog identity differs"
            )
        identities = tuple(
            (entry.plan_path, entry.remote_archive_result_path)
            for entry in self.entries
        )
        if (
            type(self.entries) is not tuple
            or any(type(row) is not FinalizationRestoreEntry for row in self.entries)
            or identities != tuple(sorted(set(identities)))
        ):
            raise FormalExperimentProductionFinalizerError(
                "finalization restore entries are not unique and sorted"
            )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "entries",
        }:
            raise FormalExperimentProductionFinalizerError(
                "finalization restore catalog fields differ"
            )
        raw_entries = value["entries"]
        if type(raw_entries) is not list:
            raise FormalExperimentProductionFinalizerError(
                "finalization restore entries are not an array"
            )
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            kind=value["kind"],  # type: ignore[arg-type]
            entries=tuple(
                FinalizationRestoreEntry.from_dict(row) for row in raw_entries
            ),
        )


@dataclass(frozen=True)
class PathBoundFormalExperimentProductionFinalizerConfig:
    """Immutable path-only inputs for final archive and shutdown."""

    schema_version: Literal[1]
    kind: Literal["formal_experiment_production_finalizer_config"]
    dag_driver_config: DriverFileBinding
    instance_identity: DriverFileBinding
    measurement_port_registry: DriverFileBinding
    final_archive_request: DriverFileBinding
    rehydration_catalog: DriverFileBinding | None
    finalization_root: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _CONFIG_KIND:
            raise FormalExperimentProductionFinalizerError(
                "production-finalizer config identity differs"
            )
        required = (
            self.dag_driver_config,
            self.instance_identity,
            self.measurement_port_registry,
            self.final_archive_request,
        )
        if any(type(row) is not DriverFileBinding for row in required) or (
            self.rehydration_catalog is not None
            and type(self.rehydration_catalog) is not DriverFileBinding
        ):
            raise TypeError("production-finalizer inputs require exact bindings")
        driver = load_path_bound_formal_dag_driver_config(
            self.dag_driver_config.absolute_path
        )
        root = _absolute(self.finalization_root, "finalization root")
        run_root = Path(driver.run_root)
        if root == run_root or not root.is_relative_to(run_root):
            raise FormalExperimentProductionFinalizerError(
                "finalization root must be a child of the exact run root"
            )
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise FormalExperimentProductionFinalizerError(
                "finalization root is unsafe"
            )
        _load_instance_identity(self.instance_identity.absolute_path)
        _load_measurement_ports(self.measurement_port_registry.absolute_path)
        request = load_archive_request(self.final_archive_request.absolute_path)
        if (
            request.safe_boundary != FINAL_ARCHIVE_SAFE_BOUNDARY
            or request.cell_id is not None
            or request.attempt is not None
        ):
            raise FormalExperimentProductionFinalizerError(
                "final archive request is not the whole-run safe boundary"
            )
        remote_root = _absolute(request.remote_payload_root, "archive payload root")
        if remote_root == run_root or not remote_root.is_relative_to(run_root):
            raise FormalExperimentProductionFinalizerError(
                "final archive payload must be one sealed child of the run root"
            )
        local_root = _absolute(request.local_final_root, "final local archive root")
        if local_root == run_root or local_root.is_relative_to(run_root):
            raise FormalExperimentProductionFinalizerError(
                "final local archive must stay outside the remote run root"
            )
        if self.rehydration_catalog is not None:
            FinalizationRehydrationCatalog.from_dict(
                _load_canonical(
                    self.rehydration_catalog.absolute_path,
                    "finalization rehydration catalog",
                )
            )

    @property
    def run_root(self) -> Path:
        return Path(
            load_path_bound_formal_dag_driver_config(
                self.dag_driver_config.absolute_path
            ).run_root
        )

    @property
    def database_path(self) -> Path:
        return self.run_root / "operator.sqlite3"

    @property
    def progress_root(self) -> Path:
        return self.run_root / "results" / "progress"

    @property
    def lock_path(self) -> Path:
        return self.run_root / "formal-dag-driver.lock"

    @property
    def shutdown_probe_path(self) -> Path:
        return Path(self.finalization_root) / "shutdown-probe.json"

    @property
    def pre_shutdown_audit_path(self) -> Path:
        return Path(self.finalization_root) / "pre-shutdown-audit.json"

    @property
    def power_transition_path(self) -> Path:
        return Path(self.finalization_root) / "power-off-transition.json"

    @property
    def power_request_journal_path(self) -> Path:
        return self.power_transition_path.with_name(
            self.power_transition_path.name + ".request.json"
        )

    @property
    def final_completion_path(self) -> Path:
        return Path(self.finalization_root) / "final-completion.json"

    @property
    def accounting_path(self) -> Path:
        return Path(self.finalization_root) / "final-accounting.json"

    @property
    def supervisor_completion_path(self) -> Path:
        return Path(self.finalization_root) / "production-finalizer-completion.json"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "dag_driver_config": self.dag_driver_config.to_dict(),
            "instance_identity": self.instance_identity.to_dict(),
            "measurement_port_registry": self.measurement_port_registry.to_dict(),
            "final_archive_request": self.final_archive_request.to_dict(),
            "rehydration_catalog": (
                None
                if self.rehydration_catalog is None
                else self.rehydration_catalog.to_dict()
            ),
            "finalization_root": self.finalization_root,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise FormalExperimentProductionFinalizerError(
                "production-finalizer config fields differ"
            )
        row = dict(value)
        for name in (
            "dag_driver_config",
            "instance_identity",
            "measurement_port_registry",
            "final_archive_request",
        ):
            row[name] = DriverFileBinding.from_dict(row[name])
        if row["rehydration_catalog"] is not None:
            row["rehydration_catalog"] = DriverFileBinding.from_dict(
                row["rehydration_catalog"]
            )
        return cls(**row)  # type: ignore[arg-type]


def publish_path_bound_production_finalizer_config(
    *,
    dag_driver_config_path: str | Path,
    instance_identity_path: str | Path,
    measurement_port_registry_path: str | Path,
    final_archive_request_path: str | Path,
    rehydration_catalog_path: str | Path | None,
    finalization_root: str | Path,
    output_path: str | Path,
) -> PathBoundFormalExperimentProductionFinalizerConfig:
    """Publish one path-bound config; every public input is a filesystem path."""

    root = _absolute(finalization_root, "finalization root")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = PathBoundFormalExperimentProductionFinalizerConfig(
        schema_version=1,
        kind=_CONFIG_KIND,
        dag_driver_config=DriverFileBinding.bind(dag_driver_config_path),
        instance_identity=DriverFileBinding.bind(instance_identity_path),
        measurement_port_registry=DriverFileBinding.bind(
            measurement_port_registry_path
        ),
        final_archive_request=DriverFileBinding.bind(final_archive_request_path),
        rehydration_catalog=(
            None
            if rehydration_catalog_path is None
            else DriverFileBinding.bind(rehydration_catalog_path)
        ),
        finalization_root=str(root),
    )
    destination = _absolute(output_path, "production-finalizer config output")
    _publish_or_replay(destination, config.to_dict(), label="finalizer config")
    return load_path_bound_production_finalizer_config(destination)


def load_path_bound_production_finalizer_config(
    path: str | Path,
) -> PathBoundFormalExperimentProductionFinalizerConfig:
    return PathBoundFormalExperimentProductionFinalizerConfig.from_dict(
        _load_canonical(path, "production-finalizer config")
    )


def _load_instance_identity(path: str | Path) -> str:
    value = _load_canonical(path, "AutoDL instance identity")
    if set(value) != {"schema_version", "kind", "instance_uuid"} or (
        value.get("schema_version") != 1
        or value.get("kind") != _INSTANCE_KIND
        or type(value.get("instance_uuid")) is not str
        or not value["instance_uuid"].startswith("pro-")
    ):
        raise FormalExperimentProductionFinalizerError(
            "AutoDL instance identity differs"
        )
    return str(value["instance_uuid"])


def _load_measurement_ports(path: str | Path) -> tuple[int, ...]:
    value = _load_canonical(path, "measurement port registry")
    raw = value.get("ports")
    if (
        set(value) != {"schema_version", "kind", "ports"}
        or value.get("schema_version") != 1
        or value.get("kind") != _PORTS_KIND
        or type(raw) is not list
    ):
        raise FormalExperimentProductionFinalizerError(
            "measurement port registry differs"
        )
    ports = tuple(raw)
    if ports != tuple(sorted(set(ports))) or any(
        isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
        for port in ports
    ):
        raise FormalExperimentProductionFinalizerError(
            "measurement port registry is not canonical"
        )
    return ports


@dataclass(frozen=True)
class FormalExperimentProductionFinalizerCompletion:
    schema_version: Literal[1]
    kind: Literal["formal_experiment_production_finalizer_completion"]
    status: Literal["COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"]
    run_id: str
    instance_uuid: str
    final_archive_id: str
    completed_at_ns: int
    final_completion: FinalAuditArtifactBinding
    accounting_breakdown: FinalAuditArtifactBinding
    power_request_journal: FinalAuditArtifactBinding
    power_transition_evidence: FinalAuditArtifactBinding
    archive_request: FinalAuditArtifactBinding
    provider_request_id: str
    archive_manifest_sha256: str
    archive_content_tree_sha256: str
    compute_gpu_hours: float
    reserved_gpu_hours: float
    whole_instance_billed_gpu_hours: float
    wall_time_hours: float

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _COMPLETION_KIND
            or self.status != "COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"
            or not self.run_id
            or not self.instance_uuid.startswith("pro-")
            or not self.final_archive_id
            or type(self.completed_at_ns) is not int
            or self.completed_at_ns < 1
            or not self.provider_request_id
        ):
            raise FormalExperimentProductionFinalizerError(
                "production-finalizer completion identity differs"
            )
        bindings = (
            self.final_completion,
            self.accounting_breakdown,
            self.power_request_journal,
            self.power_transition_evidence,
            self.archive_request,
        )
        if any(type(row) is not FinalAuditArtifactBinding for row in bindings):
            raise TypeError("production-finalizer completion bindings differ")
        _require_sha256(self.archive_manifest_sha256, "final archive manifest")
        _require_sha256(self.archive_content_tree_sha256, "final archive tree")
        for label, value in (
            ("compute GPU-hours", self.compute_gpu_hours),
            ("reserved GPU-hours", self.reserved_gpu_hours),
            ("whole-instance billed GPU-hours", self.whole_instance_billed_gpu_hours),
            ("wall-time hours", self.wall_time_hours),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise FormalExperimentProductionFinalizerError(
                    f"production-finalizer {label} is invalid"
                )
        if self.reserved_gpu_hours < self.compute_gpu_hours:
            raise FormalExperimentProductionFinalizerError(
                "final reserved GPU-hours are below compute GPU-hours"
            )

    @property
    def receipt_sha256(self) -> str:
        return _semantic_sha256(self.to_dict(include_receipt_sha256=False))

    def to_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "final_completion": self.final_completion.to_dict(),
            "accounting_breakdown": self.accounting_breakdown.to_dict(),
            "power_request_journal": self.power_request_journal.to_dict(),
            "power_transition_evidence": self.power_transition_evidence.to_dict(),
            "archive_request": self.archive_request.to_dict(),
        }
        if include_receipt_sha256:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise FormalExperimentProductionFinalizerError(
                "production-finalizer completion is not one object"
            )
        row = dict(value)
        expected = _require_sha256(
            row.pop("receipt_sha256", None),
            "production-finalizer completion receipt",
        )
        if set(row) != set(cls.__dataclass_fields__):
            raise FormalExperimentProductionFinalizerError(
                "production-finalizer completion fields differ"
            )
        for name in (
            "final_completion",
            "accounting_breakdown",
            "power_request_journal",
            "power_transition_evidence",
            "archive_request",
        ):
            row[name] = FinalAuditArtifactBinding.from_dict(row[name])
        receipt = cls(**row)  # type: ignore[arg-type]
        if receipt.receipt_sha256 != expected:
            raise FormalExperimentProductionFinalizerError(
                "production-finalizer completion digest differs"
            )
        return receipt


def load_production_finalizer_completion(
    path: str | Path,
) -> FormalExperimentProductionFinalizerCompletion:
    receipt = FormalExperimentProductionFinalizerCompletion.from_dict(
        _load_canonical(path, "production-finalizer completion")
    )
    for label, binding in (
        ("final completion", receipt.final_completion),
        ("accounting", receipt.accounting_breakdown),
        ("power request", receipt.power_request_journal),
        ("power transition", receipt.power_transition_evidence),
        ("archive request", receipt.archive_request),
    ):
        binding.reopen(label=f"production-finalizer {label}")
    return receipt


class ProductionFinalizerRuntime:
    """Injectable OS/provider boundary; defaults are the production paths."""

    def __init__(
        self,
        *,
        archive_runtime: ProductionArchiveRuntime | None = None,
        readiness_auditor: Callable[
            [ExperimentOperatorStore], FormalExperimentFinalizationReadiness
        ] = audit_finalization_readiness,
        probe_collector: Callable[..., dict[str, object]] = (
            collect_formal_shutdown_probe
        ),
        pre_shutdown_publisher: Callable[
            ..., FormalExperimentPreShutdownAuditReceipt
        ] = publish_pre_shutdown_audit,
        final_publisher: Callable[..., FormalExperimentFinalCompletionReceipt] = (
            publish_final_completion
        ),
        provider_client_factory: Callable[[str], AutoDlProApiClient] | None = None,
        environment: Mapping[str, str] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        sleeper: Callable[[float], None] = time.sleep,
        maximum_confirmation_attempts: int = 24,
        confirmation_interval_seconds: float = 5.0,
    ) -> None:
        self.archive_runtime = archive_runtime or ProductionArchiveRuntime(
            full_rehydrate=True
        )
        if not self.archive_runtime.full_rehydrate:
            raise ValueError("final archive requires full rehydrate verification")
        self.readiness_auditor = readiness_auditor
        self.probe_collector = probe_collector
        self.pre_shutdown_publisher = pre_shutdown_publisher
        self.final_publisher = final_publisher
        self.provider_client_factory = provider_client_factory
        self.environment = environment
        self.clock_ns = clock_ns
        self.sleeper = sleeper
        self.maximum_confirmation_attempts = maximum_confirmation_attempts
        self.confirmation_interval_seconds = confirmation_interval_seconds

    def restore_original_paths(
        self,
        config: PathBoundFormalExperimentProductionFinalizerConfig,
    ) -> tuple[FinalAuditArtifactBinding, ...]:
        if config.rehydration_catalog is None:
            return ()
        catalog = FinalizationRehydrationCatalog.from_dict(
            _load_canonical(
                config.rehydration_catalog.absolute_path,
                "finalization rehydration catalog",
            )
        )
        bindings = []
        for entry in catalog.entries:
            restore_evicted_files(
                plan_path=entry.plan_path,
                remote_archive_result_path=entry.remote_archive_result_path,
                receipt_path=entry.receipt_path,
                lock_path=entry.lock_path,
                clock_ns=self.clock_ns,
            )
            bindings.append(
                FinalAuditArtifactBinding.bind(
                    entry.receipt_path,
                    label="finalization restore receipt",
                )
            )
        return tuple(bindings)

    def archive(
        self,
        store: ExperimentOperatorStore,
        request: ArchiveRequest,
    ) -> None:
        store.run_archive_callbacks(request, self.archive_runtime.callbacks())

    def collect_shutdown_probe(
        self,
        *,
        config: PathBoundFormalExperimentProductionFinalizerConfig,
        instance_uuid: str,
        measurement_ports: tuple[int, ...],
    ) -> dict[str, object]:
        return self.probe_collector(
            database_path=config.database_path,
            instance_uuid=instance_uuid,
            run_root=config.run_root,
            measurement_ports=measurement_ports,
            clock_ns=self.clock_ns,
            sleeper=self.sleeper,
        )

    def power_off(
        self,
        *,
        store: ExperimentOperatorStore,
        config: PathBoundFormalExperimentProductionFinalizerConfig,
        instance_uuid: str,
    ) -> None:
        transition_autodl_instance_power(
            store=store,
            operation="power_off",
            instance_uuid=instance_uuid,
            output_path=config.power_transition_path,
            safety_probe_path=config.shutdown_probe_path,
            token_environment_name=_TOKEN_ENVIRONMENT_NAME,
            environment=self.environment,
            client_factory=self.provider_client_factory,
            clock_ns=self.clock_ns,
            sleeper=self.sleeper,
            maximum_confirmation_attempts=self.maximum_confirmation_attempts,
            confirmation_interval_seconds=self.confirmation_interval_seconds,
        )


class FormalExperimentProductionFinalizer:
    """Crash-resumable supervisor for archive, shutdown, and completion."""

    def __init__(
        self,
        config: PathBoundFormalExperimentProductionFinalizerConfig,
        *,
        runtime: ProductionFinalizerRuntime | None = None,
    ) -> None:
        if type(config) is not PathBoundFormalExperimentProductionFinalizerConfig:
            raise TypeError("production finalizer requires an exact path config")
        self.config = config
        self.runtime = runtime or ProductionFinalizerRuntime()

    def run(self) -> FormalExperimentProductionFinalizerCompletion:
        config = self.config
        rebound = PathBoundFormalExperimentProductionFinalizerConfig.from_dict(
            config.to_dict()
        )
        if rebound != config:
            raise FormalExperimentProductionFinalizerError(
                "production-finalizer path bindings changed"
            )
        Path(config.finalization_root).mkdir(mode=0o700, parents=True, exist_ok=True)
        with (
            SingletonOperatorLock(config.lock_path),
            ExperimentOperatorStore(config.database_path) as store,
        ):
            try:
                return self._run_locked(store)
            except BaseException:
                # Before a mutation journal exists, every failure remains a
                # local STOP.  Once journaled, no further mutation is made:
                # a restart may only finish status/list confirmation.
                if not config.power_request_journal_path.exists():
                    self._ensure_stop(
                        store,
                        reason="production_finalizer_failed_closed",
                    )
                raise

    def _run_locked(
        self,
        store: ExperimentOperatorStore,
    ) -> FormalExperimentProductionFinalizerCompletion:
        config = self.config
        instance_uuid = _load_instance_identity(config.instance_identity.absolute_path)
        if config.supervisor_completion_path.exists():
            receipt = load_production_finalizer_completion(
                config.supervisor_completion_path
            )
            if receipt.run_id != store.run_id or receipt.instance_uuid != instance_uuid:
                raise FormalExperimentProductionFinalizerError(
                    "existing production completion has foreign lineage"
                )
            return receipt

        if not config.pre_shutdown_audit_path.exists():
            self._prepare_pre_shutdown(
                store=store,
                instance_uuid=instance_uuid,
            )
        else:
            pre = load_pre_shutdown_audit(config.pre_shutdown_audit_path)
            if pre.run_id != store.run_id or pre.instance_uuid != instance_uuid:
                raise FormalExperimentProductionFinalizerError(
                    "existing pre-shutdown audit has foreign lineage"
                )

        if not config.power_transition_path.exists():
            self.runtime.power_off(
                store=store,
                config=config,
                instance_uuid=instance_uuid,
            )
        final = self._publish_or_load_final(store)
        accounting = self._publish_accounting(
            store=store,
            final=final,
        )
        request = load_archive_request(config.final_archive_request.absolute_path)
        receipt = FormalExperimentProductionFinalizerCompletion(
            schema_version=1,
            kind=_COMPLETION_KIND,
            status="COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL",
            run_id=final.run_id,
            instance_uuid=final.instance_uuid,
            final_archive_id=request.archive_id,
            completed_at_ns=final.finalized_at_ns,
            final_completion=FinalAuditArtifactBinding.bind(
                config.final_completion_path,
                label="final completion",
            ),
            accounting_breakdown=accounting,
            power_request_journal=FinalAuditArtifactBinding.bind(
                config.power_request_journal_path,
                label="power request journal",
            ),
            power_transition_evidence=FinalAuditArtifactBinding.bind(
                config.power_transition_path,
                label="power transition evidence",
            ),
            archive_request=FinalAuditArtifactBinding.bind(
                config.final_archive_request.absolute_path,
                label="final archive request",
            ),
            provider_request_id=final.provider_request_id,
            archive_manifest_sha256=final.archive_manifest_sha256,
            archive_content_tree_sha256=final.archive_content_tree_sha256,
            compute_gpu_hours=final.compute_gpu_hours,
            reserved_gpu_hours=final.reserved_gpu_hours,
            whole_instance_billed_gpu_hours=final.billed_gpu_hours,
            wall_time_hours=final.wall_time_seconds / 3600.0,
        )
        _publish_or_replay(
            config.supervisor_completion_path,
            receipt.to_dict(),
            label="production-finalizer completion",
        )
        reopened = load_production_finalizer_completion(
            config.supervisor_completion_path
        )
        if reopened != receipt:
            raise AssertionError("production-finalizer completion changed")
        return receipt

    def _prepare_pre_shutdown(
        self,
        *,
        store: ExperimentOperatorStore,
        instance_uuid: str,
    ) -> None:
        config = self.config
        snapshot = store.snapshot()
        controllers = tuple(snapshot["controller_nodes"])
        if len(controllers) != 21 or any(
            row["state"] != "REDUCED" for row in controllers
        ):
            raise FormalExperimentProductionFinalizerError(
                "finalization requires all exact 21 controller nodes REDUCED"
            )
        self._ensure_stop(store, reason="DAG_REDUCED_AWAITING_FINAL_AUDIT")
        snapshot = store.snapshot()
        if any(row["status"] == "RUNNING" for row in snapshot["attempts"]) or any(
            row["status"] == "RUNNING"
            for row in snapshot["controller_auxiliary_groups"]
        ):
            raise FormalExperimentProductionFinalizerError(
                "a RUNNING attempt or writer blocks finalization"
            )
        self.runtime.restore_original_paths(config)
        readiness = self.runtime.readiness_auditor(store)
        if readiness.run_id != store.run_id or readiness.node_count != 21:
            raise FormalExperimentProductionFinalizerError(
                "finalization readiness has foreign lineage"
            )
        request = load_archive_request(config.final_archive_request.absolute_path)
        self.runtime.archive(store, request)
        checkpoint = store.archive_checkpoint(request.archive_id)
        if (
            checkpoint["state"] != "EVICTION_AUTHORIZED"
            or checkpoint["safe_boundary"] != FINAL_ARCHIVE_SAFE_BOUNDARY
            or type(checkpoint["transfer_receipt"]) is not dict
            or type(checkpoint["local_sha_receipt"]) is not dict
            or type(checkpoint["rehydrate_receipt"]) is not dict
        ):
            raise FormalExperimentProductionFinalizerError(
                "whole-run archive is not fully rehydrated and authorized"
            )
        store.export_progress(config.progress_root)
        ports = _load_measurement_ports(config.measurement_port_registry.absolute_path)
        if config.shutdown_probe_path.exists():
            probe = _load_canonical(config.shutdown_probe_path, "shutdown probe")
        else:
            probe = self.runtime.collect_shutdown_probe(
                config=config,
                instance_uuid=instance_uuid,
                measurement_ports=ports,
            )
            _publish_or_replay(
                config.shutdown_probe_path,
                probe,
                label="shutdown probe",
            )
        if not shutdown_probe_is_safe(probe):
            raise FormalExperimentProductionFinalizerError(
                "shutdown probe observed a writer, GPU process, port, or log growth"
            )
        self.runtime.pre_shutdown_publisher(
            store=store,
            instance_uuid=instance_uuid,
            progress_export_root=config.progress_root,
            final_archive_id=request.archive_id,
            shutdown_probe_path=config.shutdown_probe_path,
            output_path=config.pre_shutdown_audit_path,
        )

    @staticmethod
    def _ensure_stop(store: ExperimentOperatorStore, *, reason: str) -> None:
        state, _current_reason = store.dispatch_control()
        if state != "STOP":
            store.set_dispatch_stop(reason)

    def _publish_or_load_final(
        self,
        store: ExperimentOperatorStore,
    ) -> FormalExperimentFinalCompletionReceipt:
        config = self.config
        if config.final_completion_path.exists():
            receipt = load_final_completion(config.final_completion_path)
        else:
            receipt = self.runtime.final_publisher(
                store=store,
                pre_shutdown_audit_path=config.pre_shutdown_audit_path,
                power_transition_evidence_path=config.power_transition_path,
                output_path=config.final_completion_path,
            )
        instance_uuid = _load_instance_identity(config.instance_identity.absolute_path)
        if (
            receipt.run_id != store.run_id
            or receipt.instance_uuid != instance_uuid
            or Path(receipt.pre_shutdown_audit.absolute_path)
            != config.pre_shutdown_audit_path
            or Path(receipt.power_transition_evidence.absolute_path)
            != config.power_transition_path
        ):
            raise FormalExperimentProductionFinalizerError(
                "final completion has foreign run, provider, or artifact lineage"
            )
        receipt.pre_shutdown_audit.reopen(label="finalizer pre-shutdown audit")
        receipt.power_transition_evidence.reopen(
            label="finalizer power transition evidence"
        )
        return receipt

    def _publish_accounting(
        self,
        *,
        store: ExperimentOperatorStore,
        final: FormalExperimentFinalCompletionReceipt,
    ) -> FinalAuditArtifactBinding:
        config = self.config
        snapshot = store.snapshot()
        attempts = tuple(
            row for row in snapshot["attempts"] if not bool(row["is_legacy_import"])
        )
        latest: dict[str, int] = {}
        for row in attempts:
            latest[str(row["cell_id"])] = max(
                latest.get(str(row["cell_id"]), 0), int(row["attempt"])
            )
        buckets: dict[str, dict[str, float | int]] = {}
        for row in attempts:
            task = str(dict(row["scientific_axes"]).get("task", ""))
            if int(row["attempt"]) < latest[str(row["cell_id"])]:
                activity = "retained_retry"
            elif task in {"compile", "compile_environment_patch"}:
                activity = "compile"
            elif task == "mechanism_profile_only":
                activity = "profiler"
            elif task == "deterministic_failure_injection":
                activity = "failure_diagnostic"
            elif task == "compatibility_decision":
                activity = "compatibility"
            else:
                activity = "scientific_serving_or_tuning"
            bucket = buckets.setdefault(
                activity,
                {
                    "attempt_count": 0,
                    "compute_gpu_seconds": 0.0,
                    "reserved_gpu_seconds": 0.0,
                    "allocated_billed_gpu_seconds": 0.0,
                },
            )
            bucket["attempt_count"] = int(bucket["attempt_count"]) + 1
            for source, target in (
                ("compute_gpu_seconds", "compute_gpu_seconds"),
                ("reserved_gpu_seconds", "reserved_gpu_seconds"),
                ("billed_gpu_seconds", "allocated_billed_gpu_seconds"),
            ):
                bucket[target] = float(bucket[target]) + float(row[source])
        pre = load_pre_shutdown_audit(config.pre_shutdown_audit_path)
        attempt_compute = sum(
            float(row["compute_gpu_seconds"]) for row in buckets.values()
        )
        attempt_reserved = sum(
            float(row["reserved_gpu_seconds"]) for row in buckets.values()
        )
        # Unadopted failed auxiliary attempts are intentionally outside the
        # logical cell ledger but remain part of the registered compute total.
        auxiliary_compute = pre.compute_gpu_seconds - attempt_compute
        auxiliary_reserved = pre.reserved_gpu_seconds - attempt_reserved
        if auxiliary_compute < -1e-9 or auxiliary_reserved < -1e-9:
            raise FormalExperimentProductionFinalizerError(
                "final accounting is below logical attempt accounting"
            )
        if auxiliary_compute > 1e-9 or auxiliary_reserved > 1e-9:
            buckets["unadopted_auxiliary_retry"] = {
                "attempt_count": sum(
                    1
                    for group in snapshot["controller_auxiliary_groups"]
                    if group["adopted_at_ns"] is None
                ),
                "compute_gpu_seconds": max(0.0, auxiliary_compute),
                "reserved_gpu_seconds": max(0.0, auxiliary_reserved),
                "allocated_billed_gpu_seconds": max(
                    0.0,
                    pre.allocated_billed_gpu_seconds
                    - sum(
                        float(row["allocated_billed_gpu_seconds"])
                        for row in buckets.values()
                    ),
                ),
            }
        intervals = tuple(snapshot["provider_billing_intervals"])
        if not intervals or any(
            not bool(row["complete"]) or row["instance_uuid"] != final.instance_uuid
            for row in intervals
        ):
            raise FormalExperimentProductionFinalizerError(
                "final accounting requires closed provider billing intervals"
            )
        gpu_counts = {int(row["gpu_count"]) for row in intervals}
        if len(gpu_counts) != 1:
            raise FormalExperimentProductionFinalizerError(
                "provider GPU inventory changed across billing intervals"
            )
        request = load_archive_request(config.final_archive_request.absolute_path)
        archive = store.archive_checkpoint(request.archive_id)
        archive_wall_seconds = (
            int(archive["updated_at_ns"]) - int(archive["created_at_ns"])
        ) / 1e9
        if archive_wall_seconds < 0:
            raise FormalExperimentProductionFinalizerError(
                "final archive accounting time is negative"
            )
        archive_gpu_seconds = archive_wall_seconds * next(iter(gpu_counts))
        archive_windows = []
        for row in sorted(snapshot["archives"], key=lambda value: value["archive_id"]):
            wall_seconds = (int(row["updated_at_ns"]) - int(row["created_at_ns"])) / 1e9
            if wall_seconds < 0:
                raise FormalExperimentProductionFinalizerError(
                    "an archive checkpoint accounting window is negative"
                )
            archive_windows.append(
                {
                    "archive_id": row["archive_id"],
                    "safe_boundary": row["safe_boundary"],
                    "state": row["state"],
                    "wall_time_seconds": wall_seconds,
                    "observed_window_gpu_seconds": (
                        wall_seconds * next(iter(gpu_counts))
                    ),
                    "additive": row["archive_id"] == request.archive_id,
                }
            )
        billed_gpu_seconds = final.billed_gpu_hours * 3600.0
        residual = billed_gpu_seconds - pre.reserved_gpu_seconds - archive_gpu_seconds
        if residual < -1e-6:
            raise FormalExperimentProductionFinalizerError(
                "reserved plus archive GPU time exceeds whole-instance billing"
            )
        value = {
            "schema_version": 1,
            "kind": _ACCOUNTING_KIND,
            "run_id": final.run_id,
            "instance_uuid": final.instance_uuid,
            "attempt_activity": {name: buckets[name] for name in sorted(buckets)},
            "compute_gpu_seconds": pre.compute_gpu_seconds,
            "reserved_gpu_seconds": pre.reserved_gpu_seconds,
            "allocated_billed_gpu_seconds": pre.allocated_billed_gpu_seconds,
            "whole_instance_billed_gpu_seconds": billed_gpu_seconds,
            "provider_boot_intervals": [
                {
                    "provider_started_at_ns": row["provider_started_at_ns"],
                    "provider_stopped_at_ns": row["provider_stopped_or_observed_at_ns"],
                    "gpu_count": row["gpu_count"],
                    "duration_seconds": row["duration_seconds"],
                    "whole_instance_billed_gpu_seconds": row[
                        "whole_instance_billed_gpu_seconds"
                    ],
                    "sample_count": row["sample_count"],
                    "response_sha256s": list(row["response_sha256s"]),
                }
                for row in intervals
            ],
            "wall_time_seconds": final.wall_time_seconds,
            "powered_wall_time_seconds": final.powered_wall_time_seconds,
            "final_archive_wall_time_seconds": archive_wall_seconds,
            "final_archive_reserved_gpu_seconds": archive_gpu_seconds,
            "archive_checkpoint_windows": archive_windows,
            "idle_and_control_residual_gpu_seconds": max(0.0, residual),
            "semantics": {
                "attempt_unit": "ledger_attempt",
                "retry_class": "superseded_nonlegacy_attempt",
                "archive_class": "final_archive_checkpoint_window_x_provider_gpu_count",
                "archive_window_note": (
                    "rolling windows are descriptive and may overlap attempts; "
                    "only the terminal whole-run archive is subtracted"
                ),
                "idle_class": "whole_instance_minus_attempt_reserved_minus_final_archive",
                "whole_instance": "provider_boot_interval_wall_time_x_gpu_count",
            },
        }
        _publish_or_replay(
            config.accounting_path,
            value,
            label="final accounting breakdown",
        )
        return FinalAuditArtifactBinding.bind(
            config.accounting_path,
            label="final accounting breakdown",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    operations = parser.add_subparsers(dest="operation", required=True)
    write = operations.add_parser("write-config", allow_abbrev=False)
    write.add_argument("--dag-driver-config", required=True)
    write.add_argument("--instance-identity", required=True)
    write.add_argument("--measurement-port-registry", required=True)
    write.add_argument("--final-archive-request", required=True)
    write.add_argument("--rehydration-catalog")
    write.add_argument("--finalization-root", required=True)
    write.add_argument("--output", required=True)
    for name in ("run", "status"):
        command = operations.add_parser(name, allow_abbrev=False)
        command.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.operation == "write-config":
        config = publish_path_bound_production_finalizer_config(
            dag_driver_config_path=arguments.dag_driver_config,
            instance_identity_path=arguments.instance_identity,
            measurement_port_registry_path=arguments.measurement_port_registry,
            final_archive_request_path=arguments.final_archive_request,
            rehydration_catalog_path=arguments.rehydration_catalog,
            finalization_root=arguments.finalization_root,
            output_path=arguments.output,
        )
        print(json.dumps({"config_sha256": _semantic_sha256(config.to_dict())}))
        return 0
    config = load_path_bound_production_finalizer_config(arguments.config)
    if arguments.operation == "status":
        if config.supervisor_completion_path.exists():
            receipt = load_production_finalizer_completion(
                config.supervisor_completion_path
            )
            value = {
                "run_id": receipt.run_id,
                "status": receipt.status,
                "receipt_sha256": receipt.receipt_sha256,
            }
        else:
            with ExperimentOperatorStore(config.database_path) as store:
                controllers = tuple(store.snapshot()["controller_nodes"])
            reduced = len(controllers) == 21 and all(
                row["state"] == "REDUCED" for row in controllers
            )
            value = {
                "run_id": config.run_root.name,
                "status": (
                    "POWER_CONFIRMATION_PENDING"
                    if config.power_request_journal_path.exists()
                    else "DAG_REDUCED_AWAITING_FINAL_AUDIT"
                    if reduced
                    else "DAG_ACTIVE"
                ),
            }
        print(json.dumps(value, sort_keys=True))
        return 0
    try:
        receipt = FormalExperimentProductionFinalizer(config).run()
    except BaseException as error:  # noqa: BLE001
        print(
            f"formal production finalizer: {type(error).__name__}: {error}",
            file=os.sys.stderr,
        )
        return 42
    print(
        json.dumps(
            {
                "run_id": receipt.run_id,
                "status": receipt.status,
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FinalizationRehydrationCatalog",
    "FinalizationRestoreEntry",
    "FormalExperimentProductionFinalizer",
    "FormalExperimentProductionFinalizerCompletion",
    "FormalExperimentProductionFinalizerError",
    "PathBoundFormalExperimentProductionFinalizerConfig",
    "ProductionFinalizerRuntime",
    "load_path_bound_production_finalizer_config",
    "load_production_finalizer_completion",
    "main",
    "publish_path_bound_production_finalizer_config",
]
