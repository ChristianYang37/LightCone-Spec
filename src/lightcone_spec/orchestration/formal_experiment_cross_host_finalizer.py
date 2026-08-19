"""Three-part cross-host completion protocol for the formal AutoDL campaign.

The GPU host cannot safely power itself off and then finish provider polling.
The local archive host cannot open remote privileged-home paths after shutdown.
This module therefore separates the irreversible boundary into three receipts:

* remote scientific closure: deep-audits SQLite and source paths, exports the
  final progress snapshot, seals a digest-addressed whole-run payload, and
  promises that no later remote SQLite write is needed;
* local pre-power composite: pulls and fully rehydrates that payload, validates
  every remote path through the embedded archive mapping, obtains one fresh
  post-archive zero-writer probe, and authorizes remote eviction without
  deleting anything;
* stateless final completion: journals ``power_off`` once on the local host,
  proves status/list shutdown, closes provider billing from the remote receipt,
  and publishes completion without touching remote SQLite.

The production CLI accepts only path arguments/configuration.  The provider
token exists only in the local process environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, Self

from lightcone_spec.orchestration.autodl_provider_runtime import (
    AutoDlPowerOffSafetyProbe,
    AutoDlPowerTransitionReceipt,
    AutoDlProApiClient,
    transition_autodl_instance_power_stateless,
)
from lightcone_spec.orchestration.experiment_operator import (
    ArchiveRequest,
    ArchiveStepReceipt,
    ExperimentOperatorStore,
    ProviderRuntimeSample,
    SingletonOperatorLock,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    MINIMUM_LOCAL_ARCHIVE_FREE_BYTES,
    ProductionArchiveRuntime,
    canonical_json_bytes,
)
from lightcone_spec.orchestration.formal_experiment_final_audit import (
    TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
    FinalAuditArtifactBinding,
    FormalExperimentFinalizationReadiness,
    audit_finalization_readiness,
)
from lightcone_spec.orchestration.formal_experiment_production_finalizer import (
    FinalizationRehydrationCatalog,
)
from lightcone_spec.orchestration.formal_rolling_archive import restore_evicted_files
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

_PROTOCOL_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "schema_version": 1,
            "kind": "formal_cross_host_finalization_protocol",
            "remote": [
                "stop_and_zero_running",
                "deep_21_node_scientific_readiness",
                "final_progress_export",
                "digest_addressed_whole_run_payload",
                "zero_writer_probe",
                "no_later_remote_sqlite_writes",
            ],
            "local": [
                "transfer",
                "local_sha",
                "full_rehydrate",
                "remote_path_to_archive_member_mapping",
                "fresh_post_archive_zero_writer_probe",
                "eviction_authorization_without_deletion",
            ],
            "provider": [
                "single_power_off_request_journal",
                "status_list_dual_shutdown",
                "closed_boot_interval_billing",
                "stateless_final_receipt",
            ],
            "formal_measured": False,
        }
    )
).hexdigest()
_REMOTE_RECEIPT_KIND = "formal_remote_scientific_closure"
_POST_ARCHIVE_PROBE_KIND = "formal_cross_host_post_archive_shutdown_probe"
_COMPOSITE_KIND = "formal_local_pre_poweroff_composite"
_FINAL_KIND = "formal_cross_host_final_completion"
_ARCHIVE_MANIFEST_KIND = "formal_archive_sha256_manifest"
_REMOTE_CONFIG_KIND = "formal_cross_host_remote_closure_config"
_LOCAL_CONFIG_KIND = "formal_cross_host_local_finalizer_config"
_ENDPOINT_KIND = "formal_cross_host_ssh_endpoint"
_SHA256 = frozenset("0123456789abcdef")
_TOKEN_ENV = "AUTODL_DEVELOPER_TOKEN"
_INSTANCE_KIND = "formal_autodl_instance_identity"
_PORTS_KIND = "formal_measurement_port_registry"


class FormalCrossHostFinalizationError(RuntimeError):
    """A cross-host completion invariant failed closed."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise FormalCrossHostFinalizationError(f"{label} is not a lowercase SHA-256")
    return value


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise FormalCrossHostFinalizationError(
            f"{label} must be absolute and normalized"
        )
    return path


def _regular(value: str | Path, label: str) -> Path:
    path = _absolute(value, label)
    if path.is_symlink() or not path.is_file():
        raise FormalCrossHostFinalizationError(f"{label} is not a regular file")
    return path


def _canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _regular(path, label)
    payload = source.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalCrossHostFinalizationError(f"{label} is not JSON") from error
    if type(value) is not dict or payload != canonical_json_bytes(value):
        raise FormalCrossHostFinalizationError(f"{label} is not canonical JSON")
    return value


def _publish(path: str | Path, value: object, label: str) -> Path:
    destination = _absolute(path, label)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if _canonical(destination, label) != value:
            raise FormalCrossHostFinalizationError(f"{label} is immutable")
        return destination
    publish_canonical_json_no_replace(destination, value)
    return destination


@dataclass(frozen=True)
class RemoteArchivePathBinding:
    remote_absolute_path: str
    archive_relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _absolute(self.remote_absolute_path, "remote archive source")
        pure = PurePosixPath(self.archive_relative_path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != self.archive_relative_path
        ):
            raise FormalCrossHostFinalizationError(
                "archive relative path escapes its root"
            )
        _require_sha(self.sha256, "remote archive member")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise FormalCrossHostFinalizationError(
                "remote archive member size is invalid"
            )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise FormalCrossHostFinalizationError(
                "remote archive path binding fields differ"
            )
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalRemoteScientificClosureReceipt:
    schema_version: Literal[1]
    kind: Literal["formal_remote_scientific_closure"]
    protocol_sha256: str
    trust: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]
    run_id: str
    instance_uuid: str
    closed_at_ns: int
    readiness: FormalExperimentFinalizationReadiness
    operator_snapshot_sha256: str
    progress_export_manifest_sha256: str
    progress_export_files: tuple[tuple[str, str], ...]
    provider_billing_intervals: tuple[Mapping[str, Any], ...]
    remote_run_root: str
    remote_payload_root: str
    archive_manifest_sha256: str
    archive_file_count: int
    archive_payload_bytes: int
    archive_path_bindings: tuple[RemoteArchivePathBinding, ...]
    shutdown_probe: Mapping[str, Any]
    shutdown_probe_sha256: str
    no_later_remote_sqlite_writes: Literal[True]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _REMOTE_RECEIPT_KIND
            or self.protocol_sha256 != _PROTOCOL_SHA256
            or self.trust != TRUSTED_SINGLE_OPERATOR_EMPIRICAL
            or self.formal_measured is not False
            or self.no_later_remote_sqlite_writes is not True
            or not self.run_id
            or not self.instance_uuid.startswith("pro-")
            or type(self.closed_at_ns) is not int
            or self.closed_at_ns < 1
            or type(self.readiness) is not FormalExperimentFinalizationReadiness
            or self.readiness.run_id != self.run_id
        ):
            raise FormalCrossHostFinalizationError(
                "remote scientific closure identity differs"
            )
        for label, digest in (
            ("operator snapshot", self.operator_snapshot_sha256),
            ("progress manifest", self.progress_export_manifest_sha256),
            ("archive manifest", self.archive_manifest_sha256),
            ("shutdown probe", self.shutdown_probe_sha256),
        ):
            _require_sha(digest, f"remote closure {label}")
        run_root = _absolute(self.remote_run_root, "remote run root")
        payload = _absolute(self.remote_payload_root, "remote payload root")
        if payload == run_root or not payload.is_relative_to(run_root):
            raise FormalCrossHostFinalizationError(
                "remote payload is outside the exact run root"
            )
        if (
            isinstance(self.archive_file_count, bool)
            or not isinstance(self.archive_file_count, int)
            or self.archive_file_count < 1
            or isinstance(self.archive_payload_bytes, bool)
            or not isinstance(self.archive_payload_bytes, int)
            or self.archive_payload_bytes < 0
        ):
            raise FormalCrossHostFinalizationError("remote archive coverage is invalid")
        paths = tuple(row.remote_absolute_path for row in self.archive_path_bindings)
        if (
            not self.archive_path_bindings
            or paths != tuple(sorted(set(paths)))
            or any(
                type(row) is not RemoteArchivePathBinding
                for row in self.archive_path_bindings
            )
        ):
            raise FormalCrossHostFinalizationError(
                "remote archive mapping is not exact and sorted"
            )
        if self.progress_export_files != tuple(sorted(self.progress_export_files)):
            raise FormalCrossHostFinalizationError(
                "remote progress file bindings are not sorted"
            )
        for name, digest in self.progress_export_files:
            if not name or "/" in name:
                raise FormalCrossHostFinalizationError(
                    "remote progress export filename differs"
                )
            _require_sha(digest, f"remote progress export {name}")
        if type(self.shutdown_probe) is not dict:
            raise FormalCrossHostFinalizationError(
                "remote shutdown probe is not one object"
            )
        try:
            probe = AutoDlPowerOffSafetyProbe(**self.shutdown_probe)
        except (TypeError, ValueError) as error:
            raise FormalCrossHostFinalizationError(
                "remote shutdown probe is unsafe"
            ) from error
        if probe.run_id != self.run_id or probe.instance_uuid != self.instance_uuid:
            raise FormalCrossHostFinalizationError(
                "remote shutdown probe lineage differs"
            )
        if self.shutdown_probe_sha256 != _sha(dict(self.shutdown_probe)):
            raise FormalCrossHostFinalizationError(
                "remote shutdown probe digest differs"
            )
        _validate_open_billing(
            self.provider_billing_intervals,
            instance_uuid=self.instance_uuid,
        )

    @property
    def receipt_sha256(self) -> str:
        return _sha(self.to_dict(include_receipt_sha256=False))

    def to_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "readiness": {
                **asdict(self.readiness),
                "required_archive_sha256s": sorted(
                    self.readiness.required_archive_sha256s
                ),
            },
            "progress_export_files": [list(row) for row in self.progress_export_files],
            "provider_billing_intervals": [
                _jsonable_interval(row) for row in self.provider_billing_intervals
            ],
            "archive_path_bindings": [
                asdict(row) for row in self.archive_path_bindings
            ],
            "shutdown_probe": dict(self.shutdown_probe),
        }
        if include_receipt_sha256:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise FormalCrossHostFinalizationError(
                "remote scientific closure is not one object"
            )
        row = dict(value)
        expected = _require_sha(
            row.pop("receipt_sha256", None),
            "remote scientific closure receipt",
        )
        if set(row) != set(cls.__dataclass_fields__):
            raise FormalCrossHostFinalizationError(
                "remote scientific closure fields differ"
            )
        readiness = dict(row["readiness"])
        readiness["required_archive_sha256s"] = frozenset(
            readiness["required_archive_sha256s"]
        )
        row["readiness"] = FormalExperimentFinalizationReadiness(**readiness)
        row["progress_export_files"] = tuple(
            tuple(item) for item in row["progress_export_files"]
        )
        row["provider_billing_intervals"] = tuple(row["provider_billing_intervals"])
        row["archive_path_bindings"] = tuple(
            RemoteArchivePathBinding.from_dict(item)
            for item in row["archive_path_bindings"]
        )
        receipt = cls(**row)  # type: ignore[arg-type]
        if receipt.receipt_sha256 != expected:
            raise FormalCrossHostFinalizationError(
                "remote scientific closure digest differs"
            )
        return receipt


def load_remote_scientific_closure(
    path: str | Path,
) -> FormalRemoteScientificClosureReceipt:
    return FormalRemoteScientificClosureReceipt.from_dict(
        _canonical(path, "remote scientific closure")
    )


def _jsonable_interval(value: Mapping[str, Any]) -> dict[str, object]:
    row = dict(value)
    responses = row.get("response_sha256s")
    if type(responses) is tuple:
        row["response_sha256s"] = list(responses)
    return row


def _validate_open_billing(
    intervals: Sequence[Mapping[str, Any]], *, instance_uuid: str
) -> Mapping[str, Any]:
    if not intervals:
        raise FormalCrossHostFinalizationError(
            "remote scientific closure lacks provider billing evidence"
        )
    prior_end = 0
    open_rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(intervals):
        row = dict(raw)
        required = {
            "instance_uuid",
            "provider_started_at_ns",
            "provider_stopped_or_observed_at_ns",
            "complete",
            "gpu_count",
            "duration_seconds",
            "whole_instance_billed_gpu_seconds",
            "sample_count",
            "response_sha256s",
        }
        start = row.get("provider_started_at_ns")
        end = row.get("provider_stopped_or_observed_at_ns")
        gpu_count = row.get("gpu_count")
        responses = row.get("response_sha256s")
        if (
            set(row) != required
            or row.get("instance_uuid") != instance_uuid
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 1
            or end < start
            or start < prior_end
            or type(row.get("complete")) is not bool
            or isinstance(gpu_count, bool)
            or not isinstance(gpu_count, int)
            or gpu_count < 1
            or isinstance(row.get("sample_count"), bool)
            or not isinstance(row.get("sample_count"), int)
            or row["sample_count"] < 1
            or type(responses) not in {tuple, list}
            or not responses
            or any(
                _require_sha(item, "provider response") != item for item in responses
            )
        ):
            raise FormalCrossHostFinalizationError("provider billing interval differs")
        duration = (end - start) / 1e9
        billed = duration * gpu_count
        if not math.isclose(
            float(row["duration_seconds"]), duration, abs_tol=1e-9
        ) or not math.isclose(
            float(row["whole_instance_billed_gpu_seconds"]),
            billed,
            abs_tol=1e-8,
        ):
            raise FormalCrossHostFinalizationError(
                "provider billing arithmetic differs"
            )
        if not row["complete"]:
            open_rows.append(raw)
            if index != len(intervals) - 1:
                raise FormalCrossHostFinalizationError(
                    "open provider billing interval is not last"
                )
        prior_end = end
    if len(open_rows) != 1:
        raise FormalCrossHostFinalizationError(
            "scientific closure requires exactly one open provider interval"
        )
    return open_rows[0]


def _instance_identity(path: str | Path) -> str:
    value = _canonical(path, "AutoDL instance identity")
    if (
        set(value) != {"schema_version", "kind", "instance_uuid"}
        or value.get("schema_version") != 1
        or value.get("kind") != _INSTANCE_KIND
        or type(value.get("instance_uuid")) is not str
        or not value["instance_uuid"].startswith("pro-")
    ):
        raise FormalCrossHostFinalizationError("AutoDL instance identity differs")
    return str(value["instance_uuid"])


def _measurement_ports(path: str | Path) -> tuple[int, ...]:
    value = _canonical(path, "measurement port registry")
    raw = value.get("ports")
    if (
        set(value) != {"schema_version", "kind", "ports"}
        or value.get("schema_version") != 1
        or value.get("kind") != _PORTS_KIND
        or type(raw) is not list
    ):
        raise FormalCrossHostFinalizationError("measurement port registry differs")
    ports = tuple(raw)
    if ports != tuple(sorted(set(ports))) or any(
        isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
        for port in ports
    ):
        raise FormalCrossHostFinalizationError("measurement ports are not canonical")
    return ports


@dataclass(frozen=True)
class PathBoundRemoteClosureConfig:
    schema_version: Literal[1]
    kind: Literal["formal_cross_host_remote_closure_config"]
    dag_driver_config: DriverFileBinding
    instance_identity: DriverFileBinding
    measurement_port_registry: DriverFileBinding
    rehydration_catalog: DriverFileBinding | None
    closure_root: str
    payload_root: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _REMOTE_CONFIG_KIND:
            raise FormalCrossHostFinalizationError(
                "remote closure config identity differs"
            )
        if any(
            type(binding) is not DriverFileBinding
            for binding in (
                self.dag_driver_config,
                self.instance_identity,
                self.measurement_port_registry,
            )
        ):
            raise TypeError("remote closure sources require exact bindings")
        driver = load_path_bound_formal_dag_driver_config(
            self.dag_driver_config.absolute_path
        )
        run_root = Path(driver.run_root)
        closure = _absolute(self.closure_root, "remote closure root")
        payload = _absolute(self.payload_root, "remote payload root")
        if (
            closure == run_root
            or payload == run_root
            or not closure.is_relative_to(run_root)
            or not payload.is_relative_to(run_root)
            or closure == payload
            or closure.is_relative_to(payload)
            or payload.is_relative_to(closure)
        ):
            raise FormalCrossHostFinalizationError(
                "remote closure and payload must be distinct run-root children"
            )
        _instance_identity(self.instance_identity.absolute_path)
        _measurement_ports(self.measurement_port_registry.absolute_path)
        if self.rehydration_catalog is not None:
            if type(self.rehydration_catalog) is not DriverFileBinding:
                raise TypeError("remote rehydration catalog binding differs")
            FinalizationRehydrationCatalog.from_dict(
                _canonical(
                    self.rehydration_catalog.absolute_path,
                    "remote rehydration catalog",
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
    def lock_path(self) -> Path:
        return self.run_root / "formal-dag-driver.lock"

    @property
    def receipt_path(self) -> Path:
        return Path(self.closure_root) / "scientific-closure.json"

    @property
    def operator_snapshot_path(self) -> Path:
        return Path(self.closure_root) / "operator-snapshot.json"

    @property
    def database_snapshot_path(self) -> Path:
        return Path(self.closure_root) / "operator-snapshot.sqlite3"

    @property
    def progress_root(self) -> Path:
        return Path(self.closure_root) / "progress"

    @property
    def export_authority_path(self) -> Path:
        return Path(self.closure_root) / "progress-export-authority.json"

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "dag_driver_config": self.dag_driver_config.to_dict(),
            "instance_identity": self.instance_identity.to_dict(),
            "measurement_port_registry": self.measurement_port_registry.to_dict(),
            "rehydration_catalog": (
                None
                if self.rehydration_catalog is None
                else self.rehydration_catalog.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise FormalCrossHostFinalizationError(
                "remote closure config fields differ"
            )
        row = dict(value)
        for name in (
            "dag_driver_config",
            "instance_identity",
            "measurement_port_registry",
        ):
            row[name] = DriverFileBinding.from_dict(row[name])
        if row["rehydration_catalog"] is not None:
            row["rehydration_catalog"] = DriverFileBinding.from_dict(
                row["rehydration_catalog"]
            )
        return cls(**row)  # type: ignore[arg-type]


def publish_remote_closure_config(
    *,
    dag_driver_config_path: str | Path,
    instance_identity_path: str | Path,
    measurement_port_registry_path: str | Path,
    rehydration_catalog_path: str | Path | None = None,
    closure_root: str | Path,
    payload_root: str | Path,
    output_path: str | Path,
) -> PathBoundRemoteClosureConfig:
    config = PathBoundRemoteClosureConfig(
        schema_version=1,
        kind=_REMOTE_CONFIG_KIND,
        dag_driver_config=DriverFileBinding.bind(dag_driver_config_path),
        instance_identity=DriverFileBinding.bind(instance_identity_path),
        measurement_port_registry=DriverFileBinding.bind(
            measurement_port_registry_path
        ),
        rehydration_catalog=(
            None
            if rehydration_catalog_path is None
            else DriverFileBinding.bind(rehydration_catalog_path)
        ),
        closure_root=str(_absolute(closure_root, "remote closure root")),
        payload_root=str(_absolute(payload_root, "remote payload root")),
    )
    _publish(output_path, config.to_dict(), "remote closure config")
    return load_remote_closure_config(output_path)


def load_remote_closure_config(path: str | Path) -> PathBoundRemoteClosureConfig:
    return PathBoundRemoteClosureConfig.from_dict(
        _canonical(path, "remote closure config")
    )


def _readonly_database_semantic_sha256(database: Path) -> str:
    if database.is_symlink() or not database.is_file():
        raise FormalCrossHostFinalizationError(
            "operator database is not a regular file"
        )
    connection = sqlite3.connect(
        f"file:{database.as_posix()}?mode=ro",
        uri=True,
        isolation_level=None,
    )
    try:
        rows = tuple(connection.iterdump())
    except sqlite3.Error as error:
        raise FormalCrossHostFinalizationError(
            "operator database cannot be dumped read-only"
        ) from error
    finally:
        connection.close()
    return _sha({"sqlite_iterdump": rows})


def _backup_database_readonly(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    source_connection = sqlite3.connect(
        f"file:{source.as_posix()}?mode=ro",
        uri=True,
        isolation_level=None,
    )
    destination_connection = sqlite3.connect(temporary)
    try:
        source_connection.backup(destination_connection)
        destination_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        destination_connection.commit()
    except sqlite3.Error as error:
        raise FormalCrossHostFinalizationError(
            "read-only operator database backup failed"
        ) from error
    finally:
        destination_connection.close()
        source_connection.close()
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)


def _nested_existing_paths(value: object) -> set[Path]:
    paths: set[Path] = set()
    if type(value) is dict:
        for key, item in value.items():
            if (
                type(key) is str
                and key.endswith(("_path", "_directory"))
                and type(item) is str
                and item.startswith("/")
            ):
                candidate = Path(item)
                if candidate.exists() and not candidate.is_symlink():
                    if candidate.is_file():
                        paths.add(candidate.resolve(strict=True))
                    elif candidate.is_dir():
                        for child in candidate.rglob("*"):
                            if child.is_file() and not child.is_symlink():
                                paths.add(child.resolve(strict=True))
            paths.update(_nested_existing_paths(item))
    elif type(value) in {tuple, list}:
        for item in value:
            paths.update(_nested_existing_paths(item))
    return paths


def _whole_run_source_files(
    *,
    run_root: Path,
    payload_root: Path,
    snapshot: Mapping[str, Any],
    extra_sources: Sequence[Path] = (),
) -> tuple[Path, ...]:
    sources: set[Path] = set()
    excluded_names = {
        "operator.sqlite3",
        "operator.sqlite3-wal",
        "operator.sqlite3-shm",
        "formal-dag-driver.lock",
    }
    for directory, names, filenames in os.walk(run_root, followlinks=False):
        base = Path(directory)
        names[:] = sorted(
            name
            for name in names
            if not (base / name).is_symlink()
            and not (base / name).is_relative_to(payload_root)
        )
        for name in sorted(filenames):
            path = base / name
            if name in excluded_names or path.is_symlink() or not path.is_file():
                continue
            if path.is_relative_to(payload_root):
                continue
            sources.add(path.resolve(strict=True))
    for source in _nested_existing_paths(snapshot):
        if not source.is_relative_to(payload_root):
            sources.add(source)
    for source in extra_sources:
        sources.add(_regular(source, "whole-run authority source"))
    return tuple(sorted(sources, key=str))


def _closure_authority_sources(
    config: PathBoundRemoteClosureConfig,
) -> tuple[Path, ...]:
    driver = load_path_bound_formal_dag_driver_config(
        config.dag_driver_config.absolute_path
    )
    bindings = [
        config.dag_driver_config,
        config.instance_identity,
        config.measurement_port_registry,
        driver.protocol_lock,
        driver.content_source,
        driver.runtime_authority_manifest,
        driver.inventory,
        driver.doctor_report,
        driver.preflight_workload_authority,
        *driver.profiler_tools,
    ]
    if config.rehydration_catalog is not None:
        bindings.append(config.rehydration_catalog)
    return tuple(Path(binding.absolute_path) for binding in bindings)


def _controller_artifact_paths(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    names = (
        "materialization_path",
        "node_materialization_path",
        "execution_source_path",
        "prepared_launch_path",
        "decision_path",
        "completion_path",
    )
    paths = []
    for row in snapshot["controller_nodes"]:
        for name in names:
            value = row.get(name)
            if value is not None:
                if type(value) is not str or not value.startswith("/"):
                    raise FormalCrossHostFinalizationError(
                        "controller artifact path is not absolute"
                    )
                paths.append(value)
    return tuple(sorted(set(paths)))


def _publish_digest_archive(
    *,
    sources: Sequence[Path],
    payload_root: Path,
) -> tuple[str, tuple[RemoteArchivePathBinding, ...], int]:
    manifest_path = payload_root / "sha256_manifest.json"
    if manifest_path.exists():
        manifest = _canonical(manifest_path, "remote archive manifest")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != _ARCHIVE_MANIFEST_KIND
            or type(manifest.get("files")) is not list
        ):
            raise FormalCrossHostFinalizationError(
                "existing remote archive manifest differs"
            )
        mapping_path = payload_root / "remote_path_bindings.json"
        mapping_value = _canonical(mapping_path, "remote archive path mapping")
        raw = mapping_value.get("bindings")
        if (
            mapping_value.get("schema_version") != 1
            or mapping_value.get("kind") != "formal_remote_path_archive_mapping"
            or type(raw) is not list
        ):
            raise FormalCrossHostFinalizationError(
                "existing remote archive path mapping differs"
            )
        bindings = tuple(RemoteArchivePathBinding.from_dict(row) for row in raw)
        expected_paths = tuple(sorted(str(source) for source in sources))
        if tuple(row.remote_absolute_path for row in bindings) != expected_paths:
            raise FormalCrossHostFinalizationError(
                "existing remote archive does not cover current whole-run sources"
            )
        for row in bindings:
            source = _regular(row.remote_absolute_path, "existing archive source")
            if (
                source.stat().st_size != row.size_bytes
                or _file_sha(source) != row.sha256
            ):
                raise FormalCrossHostFinalizationError(
                    "existing remote archive source changed before closure"
                )
        checked = _verify_remote_archive(payload_root, manifest)
        return _file_sha(manifest_path), bindings, checked

    if payload_root.exists() and any(payload_root.iterdir()):
        raise FormalCrossHostFinalizationError(
            "unsealed remote payload root is not empty"
        )
    payload_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    by_digest: dict[str, tuple[str, int]] = {}
    bindings: list[RemoteArchivePathBinding] = []
    for source in sources:
        before = source.stat(follow_symlinks=False)
        digest = _file_sha(source)
        after = source.stat(follow_symlinks=False)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        if identity(before) != identity(after):
            raise FormalCrossHostFinalizationError(
                f"archive source changed while hashing: {source}"
            )
        size = after.st_size
        relative = f"objects/{digest[:2]}/{digest}"
        destination = payload_root / relative
        if digest not in by_digest:
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                os.link(source, destination, follow_symlinks=False)
            except OSError:
                shutil.copy2(source, destination, follow_symlinks=False)
            if _file_sha(destination) != digest:
                raise FormalCrossHostFinalizationError(
                    "digest archive object changed during publication"
                )
            by_digest[digest] = (relative, size)
        elif by_digest[digest][1] != size:
            raise AssertionError("equal SHA-256 objects have different sizes")
        bindings.append(
            RemoteArchivePathBinding(
                remote_absolute_path=str(source),
                archive_relative_path=relative,
                sha256=digest,
                size_bytes=size,
            )
        )
    bindings.sort(key=lambda row: row.remote_absolute_path)
    mapping = {
        "schema_version": 1,
        "kind": "formal_remote_path_archive_mapping",
        "bindings": [asdict(row) for row in bindings],
    }
    _publish(
        payload_root / "remote_path_bindings.json",
        mapping,
        "remote archive path mapping",
    )
    # The mapping is itself part of the verified payload, while it cannot map
    # itself recursively.  All source paths continue to resolve via bindings.
    mapping_sha = _file_sha(payload_root / "remote_path_bindings.json")
    mapping_size = (payload_root / "remote_path_bindings.json").stat().st_size
    rows = [
        {"path": relative, "sha256": digest, "size_bytes": size}
        for digest, (relative, size) in sorted(by_digest.items())
    ]
    rows.append(
        {
            "path": "remote_path_bindings.json",
            "sha256": mapping_sha,
            "size_bytes": mapping_size,
        }
    )
    rows.sort(key=lambda row: row["path"])
    manifest = {
        "schema_version": 1,
        "kind": _ARCHIVE_MANIFEST_KIND,
        "files": rows,
    }
    _publish(manifest_path, manifest, "remote archive manifest")
    checked = _verify_remote_archive(payload_root, manifest)
    return _file_sha(manifest_path), tuple(bindings), checked


def _verify_remote_archive(root: Path, manifest: Mapping[str, Any]) -> int:
    rows = manifest.get("files")
    if type(rows) is not list or not rows:
        raise FormalCrossHostFinalizationError("archive manifest is empty")
    total = 0
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256", "size_bytes"}:
            raise FormalCrossHostFinalizationError("archive manifest row differs")
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise FormalCrossHostFinalizationError("archive manifest path differs")
        seen.add(str(relative))
        source = root / str(relative)
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != row["size_bytes"]
            or _file_sha(source) != _require_sha(row["sha256"], "archive row")
        ):
            raise FormalCrossHostFinalizationError("archive manifest member differs")
        total += int(row["size_bytes"])
    return total


def _activity_accounting(snapshot: Mapping[str, Any]) -> dict[str, object]:
    attempts = tuple(
        row for row in snapshot["attempts"] if not bool(row["is_legacy_import"])
    )
    latest: dict[str, int] = {}
    for row in attempts:
        cell_id = str(row["cell_id"])
        latest[cell_id] = max(latest.get(cell_id, 0), int(row["attempt"]))
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
    unadopted = tuple(
        row
        for row in snapshot["controller_auxiliary_groups"]
        if row["adopted_at_ns"] is None
    )
    if unadopted:
        buckets["unadopted_auxiliary_retry"] = {
            "attempt_count": len(unadopted),
            "compute_gpu_seconds": sum(
                float(row["compute_gpu_seconds"]) for row in unadopted
            ),
            "reserved_gpu_seconds": sum(
                float(row["reserved_gpu_seconds"]) for row in unadopted
            ),
            "allocated_billed_gpu_seconds": sum(
                float(row["billed_gpu_seconds"]) for row in unadopted
            ),
        }
    return {name: buckets[name] for name in sorted(buckets)}


def _restore_original_paths_before_closure(
    config: PathBoundRemoteClosureConfig,
    *,
    clock_ns: Callable[[], int],
) -> None:
    if config.rehydration_catalog is None:
        return
    catalog = FinalizationRehydrationCatalog.from_dict(
        _canonical(
            config.rehydration_catalog.absolute_path,
            "remote rehydration catalog",
        )
    )
    for entry in catalog.entries:
        restore_evicted_files(
            plan_path=entry.plan_path,
            remote_archive_result_path=entry.remote_archive_result_path,
            receipt_path=entry.receipt_path,
            lock_path=entry.lock_path,
            clock_ns=clock_ns,
        )


def seal_remote_scientific_closure(
    config: PathBoundRemoteClosureConfig,
    *,
    readiness_auditor: Callable[
        [ExperimentOperatorStore], FormalExperimentFinalizationReadiness
    ] = audit_finalization_readiness,
    probe_collector: Callable[..., dict[str, object]] = collect_formal_shutdown_probe,
    clock_ns: Callable[[], int] = time.time_ns,
    sleeper: Callable[[float], None] = time.sleep,
) -> FormalRemoteScientificClosureReceipt:
    """Publish part A, after which the remote SQLite is read-only forever."""

    if type(config) is not PathBoundRemoteClosureConfig:
        raise TypeError("remote scientific closure requires an exact path config")
    if config.receipt_path.exists() or config.receipt_path.is_symlink():
        return load_remote_scientific_closure(config.receipt_path)
    Path(config.closure_root).mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot: dict[str, Any]
    readiness: FormalExperimentFinalizationReadiness
    provider_intervals: tuple[Mapping[str, Any], ...]
    with SingletonOperatorLock(config.lock_path):
        _restore_original_paths_before_closure(config, clock_ns=clock_ns)
        with ExperimentOperatorStore(config.database_path) as store:
            state, reason = store.dispatch_control()
            if state != "STOP":
                store.set_dispatch_stop("DAG_REDUCED_AWAITING_FINAL_AUDIT")
            elif reason != "DAG_REDUCED_AWAITING_FINAL_AUDIT":
                raise FormalCrossHostFinalizationError(
                    "scientific closure found STOP for another unresolved reason"
                )
            snapshot = store.snapshot()
            controllers = tuple(snapshot["controller_nodes"])
            if len(controllers) != 21 or any(
                row["state"] != "REDUCED" for row in controllers
            ):
                raise FormalCrossHostFinalizationError(
                    "scientific closure requires all exact 21 nodes REDUCED"
                )
            if any(row["status"] == "RUNNING" for row in snapshot["attempts"]) or any(
                row["status"] == "RUNNING"
                for row in snapshot["controller_auxiliary_groups"]
            ):
                raise FormalCrossHostFinalizationError(
                    "a RUNNING attempt or evidence writer blocks scientific closure"
                )
            readiness = readiness_auditor(store)
            if (
                readiness.run_id != store.run_id
                or readiness.node_count != 21
                or readiness.latest_complete_attempt_count
                != readiness.expected_cell_count
                or readiness.selection_decision_count < 1
                or readiness.metric_count < 1
                or readiness.headline_metric_count < 1
            ):
                raise FormalCrossHostFinalizationError(
                    "scientific readiness lacks exact coverage, selections, or metrics"
                )
            if config.export_authority_path.exists():
                export_authority = _canonical(
                    config.export_authority_path,
                    "progress export authority",
                )
            else:
                export_authority = {
                    "schema_version": 1,
                    "kind": "formal_closed_progress_export_authority",
                    "run_id": readiness.run_id,
                    "exported_at_ns": int(clock_ns()),
                }
                _publish(
                    config.export_authority_path,
                    export_authority,
                    "progress export authority",
                )
            if (
                export_authority.get("schema_version") != 1
                or export_authority.get("kind")
                != "formal_closed_progress_export_authority"
                or export_authority.get("run_id") != readiness.run_id
                or type(export_authority.get("exported_at_ns")) is not int
            ):
                raise FormalCrossHostFinalizationError(
                    "progress export authority differs"
                )
            store.export_progress(
                config.progress_root,
                exported_at_ns=export_authority["exported_at_ns"],
            )
            snapshot = store.snapshot()
            provider_intervals = tuple(snapshot["provider_billing_intervals"])
            _validate_open_billing(
                provider_intervals,
                instance_uuid=_instance_identity(
                    config.instance_identity.absolute_path
                ),
            )

        database_semantic = _readonly_database_semantic_sha256(config.database_path)
        _backup_database_readonly(
            config.database_path,
            config.database_snapshot_path,
        )
        database_snapshot_sha = _file_sha(config.database_snapshot_path)
        operator_snapshot = {
            "schema_version": 1,
            "kind": "formal_closed_operator_snapshot",
            "run_id": readiness.run_id,
            "sqlite_semantic_sha256": database_semantic,
            "sqlite_backup_sha256": database_snapshot_sha,
            "snapshot": snapshot,
            "activity_accounting": _activity_accounting(snapshot),
        }
        _publish(
            config.operator_snapshot_path,
            operator_snapshot,
            "closed operator snapshot",
        )
        sources = _whole_run_source_files(
            run_root=config.run_root,
            payload_root=Path(config.payload_root),
            snapshot=snapshot,
            extra_sources=_closure_authority_sources(config),
        )
        archive_manifest_sha, bindings, archive_bytes = _publish_digest_archive(
            sources=sources,
            payload_root=Path(config.payload_root),
        )
        archived_digests = {row.sha256 for row in bindings}
        archived_paths = {row.remote_absolute_path for row in bindings}
        if not set(_controller_artifact_paths(snapshot)).issubset(archived_paths):
            raise FormalCrossHostFinalizationError(
                "whole-run archive omits a controller artifact path"
            )
        if not readiness.required_archive_sha256s.issubset(archived_digests):
            missing = sorted(readiness.required_archive_sha256s - archived_digests)
            raise FormalCrossHostFinalizationError(
                "whole-run archive omits required evidence: " + ",".join(missing)
            )
        progress_manifest_path = config.progress_root / "export_manifest.json"
        progress_manifest = _canonical(
            progress_manifest_path,
            "final progress export manifest",
        )
        raw_progress_files = progress_manifest.get("files")
        if (
            progress_manifest.get("run_id") != readiness.run_id
            or type(raw_progress_files) is not dict
        ):
            raise FormalCrossHostFinalizationError(
                "final progress export manifest differs"
            )
        progress_files = tuple(sorted(raw_progress_files.items()))
        for name, digest in progress_files:
            source = config.progress_root / name
            if _file_sha(source) != _require_sha(digest, f"progress export {name}"):
                raise FormalCrossHostFinalizationError(
                    "final progress export changed before closure"
                )
        instance_uuid = _instance_identity(config.instance_identity.absolute_path)
        probe = probe_collector(
            database_path=config.database_path,
            instance_uuid=instance_uuid,
            run_root=config.run_root,
            measurement_ports=_measurement_ports(
                config.measurement_port_registry.absolute_path
            ),
            readonly_database=True,
            clock_ns=clock_ns,
            sleeper=sleeper,
        )
        if not shutdown_probe_is_safe(probe):
            raise FormalCrossHostFinalizationError(
                "scientific closure probe observed a writer, GPU process, or port"
            )
        if (
            _readonly_database_semantic_sha256(config.database_path)
            != database_semantic
        ):
            raise FormalCrossHostFinalizationError(
                "operator database changed while sealing scientific closure"
            )
        manifest = _canonical(
            Path(config.payload_root) / "sha256_manifest.json",
            "remote archive manifest",
        )
        receipt = FormalRemoteScientificClosureReceipt(
            schema_version=1,
            kind=_REMOTE_RECEIPT_KIND,
            protocol_sha256=_PROTOCOL_SHA256,
            trust=TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
            formal_measured=False,
            run_id=readiness.run_id,
            instance_uuid=instance_uuid,
            closed_at_ns=int(clock_ns()),
            readiness=readiness,
            operator_snapshot_sha256=_file_sha(config.operator_snapshot_path),
            progress_export_manifest_sha256=_file_sha(progress_manifest_path),
            progress_export_files=progress_files,
            provider_billing_intervals=provider_intervals,
            remote_run_root=str(config.run_root),
            remote_payload_root=config.payload_root,
            archive_manifest_sha256=archive_manifest_sha,
            archive_file_count=len(manifest["files"]),
            archive_payload_bytes=archive_bytes,
            archive_path_bindings=bindings,
            shutdown_probe=probe,
            shutdown_probe_sha256=_sha(probe),
            no_later_remote_sqlite_writes=True,
        )
        _publish(
            config.receipt_path,
            receipt.to_dict(),
            "remote scientific closure",
        )
    return load_remote_scientific_closure(config.receipt_path)


def publish_remote_post_archive_probe(
    config: PathBoundRemoteClosureConfig,
    *,
    output_path: str | Path,
    probe_collector: Callable[..., dict[str, object]] = collect_formal_shutdown_probe,
    clock_ns: Callable[[], int] = time.time_ns,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Publish a fresh probe without opening the write-capable SQLite store."""

    output = _absolute(output_path, "post-archive probe output")
    probe_root = Path(config.closure_root) / "post-archive-probes"
    if output.parent != probe_root:
        raise FormalCrossHostFinalizationError(
            "post-archive probe output is outside its immutable spool"
        )
    if output.exists() or output.is_symlink():
        return _canonical(output, "post-archive probe")
    closure = load_remote_scientific_closure(config.receipt_path)
    snapshot_binding = next(
        (
            row
            for row in closure.archive_path_bindings
            if row.remote_absolute_path == str(config.operator_snapshot_path)
        ),
        None,
    )
    if snapshot_binding is None:
        raise FormalCrossHostFinalizationError(
            "scientific closure did not bind the operator snapshot"
        )
    operator_snapshot = _canonical(
        config.operator_snapshot_path,
        "closed operator snapshot",
    )
    expected_semantic = _require_sha(
        operator_snapshot.get("sqlite_semantic_sha256"),
        "closed SQLite semantic snapshot",
    )
    if (
        _file_sha(config.operator_snapshot_path) != closure.operator_snapshot_sha256
        or _readonly_database_semantic_sha256(config.database_path) != expected_semantic
    ):
        raise FormalCrossHostFinalizationError(
            "remote SQLite changed after scientific closure"
        )
    probe = probe_collector(
        database_path=config.database_path,
        instance_uuid=closure.instance_uuid,
        run_root=config.run_root,
        measurement_ports=_measurement_ports(
            config.measurement_port_registry.absolute_path
        ),
        readonly_database=True,
        clock_ns=clock_ns,
        sleeper=sleeper,
    )
    if not shutdown_probe_is_safe(probe):
        raise FormalCrossHostFinalizationError(
            "post-archive probe observed a writer, RUNNING attempt, GPU, or port"
        )
    if _readonly_database_semantic_sha256(config.database_path) != expected_semantic:
        raise FormalCrossHostFinalizationError(
            "remote SQLite changed during post-archive probe"
        )
    wrapper = {
        "schema_version": 1,
        "kind": _POST_ARCHIVE_PROBE_KIND,
        "scientific_closure_sha256": closure.receipt_sha256,
        "raw_probe": probe,
        "raw_probe_sha256": _sha(probe),
        "database_semantic_sha256": expected_semantic,
        "no_remote_sqlite_write_performed": True,
    }
    _publish(output, wrapper, "post-archive probe")
    return wrapper


_SSH_TARGET = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$")
_REMOTE_SAFE_PATH = re.compile(r"^[A-Za-z0-9_./+:-]+$")


@dataclass(frozen=True)
class CrossHostSshEndpoint:
    schema_version: Literal[1]
    kind: Literal["formal_cross_host_ssh_endpoint"]
    ssh_target: str
    ssh_port: int
    remote_python: str
    remote_finalizer_script: str
    remote_config_path: str
    remote_closure_path: str
    remote_probe_root: str
    ssh_executable: str = "ssh"
    rsync_executable: str = "rsync"
    ssh_identity_file: str | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _ENDPOINT_KIND
            or _SSH_TARGET.fullmatch(self.ssh_target) is None
            or isinstance(self.ssh_port, bool)
            or not isinstance(self.ssh_port, int)
            or not 1 <= self.ssh_port <= 65535
        ):
            raise FormalCrossHostFinalizationError(
                "cross-host SSH endpoint identity differs"
            )
        for label, value in (
            ("remote Python", self.remote_python),
            ("remote finalizer", self.remote_finalizer_script),
            ("remote config", self.remote_config_path),
            ("remote closure", self.remote_closure_path),
            ("remote probe root", self.remote_probe_root),
        ):
            path = Path(value)
            if (
                not path.is_absolute()
                or path != path.resolve(strict=False)
                or _REMOTE_SAFE_PATH.fullmatch(value) is None
            ):
                raise FormalCrossHostFinalizationError(
                    f"{label} must be a safe normalized absolute path"
                )
        for value in (self.ssh_executable, self.rsync_executable):
            if type(value) is not str or not value or "\x00" in value:
                raise FormalCrossHostFinalizationError(
                    "SSH/rsync executable identity differs"
                )
        if self.ssh_identity_file is not None:
            identity = _regular(self.ssh_identity_file, "SSH private identity")
            if stat.S_IMODE(identity.stat().st_mode) & 0o077:
                raise FormalCrossHostFinalizationError(
                    "SSH private identity permissions exceed 0600"
                )

    @property
    def ssh_argv(self) -> tuple[str, ...]:
        argv = [
            self.ssh_executable,
            "-p",
            str(self.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=20",
        ]
        if self.ssh_identity_file is not None:
            argv.extend(("-i", self.ssh_identity_file))
        return tuple(argv)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise FormalCrossHostFinalizationError("SSH endpoint fields differ")
        return cls(**value)  # type: ignore[arg-type]


def publish_cross_host_ssh_endpoint(
    *, output_path: str | Path, **paths: object
) -> CrossHostSshEndpoint:
    endpoint = CrossHostSshEndpoint(
        schema_version=1,
        kind=_ENDPOINT_KIND,
        **paths,  # type: ignore[arg-type]
    )
    _publish(output_path, endpoint.to_dict(), "cross-host SSH endpoint")
    return load_cross_host_ssh_endpoint(output_path)


def load_cross_host_ssh_endpoint(path: str | Path) -> CrossHostSshEndpoint:
    return CrossHostSshEndpoint.from_dict(_canonical(path, "SSH endpoint"))


@dataclass(frozen=True)
class PathBoundCrossHostFinalizerConfig:
    schema_version: Literal[1]
    kind: Literal["formal_cross_host_local_finalizer_config"]
    endpoint: DriverFileBinding
    local_finalization_root: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _LOCAL_CONFIG_KIND:
            raise FormalCrossHostFinalizationError(
                "local cross-host finalizer config identity differs"
            )
        if type(self.endpoint) is not DriverFileBinding:
            raise TypeError("local cross-host endpoint requires an exact binding")
        load_cross_host_ssh_endpoint(self.endpoint.absolute_path)
        root = _absolute(self.local_finalization_root, "local finalization root")
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise FormalCrossHostFinalizationError("local finalization root is unsafe")

    @property
    def closure_path(self) -> Path:
        return Path(self.local_finalization_root) / "remote-scientific-closure.json"

    @property
    def archive_partial_root(self) -> Path:
        return Path(self.local_finalization_root) / "whole-run.partial"

    @property
    def archive_final_root(self) -> Path:
        return Path(self.local_finalization_root) / "whole-run.final"

    @property
    def transfer_receipt_path(self) -> Path:
        return Path(self.local_finalization_root) / "archive-transfer.json"

    @property
    def local_sha_receipt_path(self) -> Path:
        return Path(self.local_finalization_root) / "archive-local-sha.json"

    @property
    def rehydrate_receipt_path(self) -> Path:
        return Path(self.local_finalization_root) / "archive-rehydrate.json"

    @property
    def probe_root(self) -> Path:
        return Path(self.local_finalization_root) / "post-archive-probes"

    @property
    def composite_root(self) -> Path:
        return Path(self.local_finalization_root) / "pre-power-composites"

    @property
    def power_transition_path(self) -> Path:
        return Path(self.local_finalization_root) / "power-transition.json"

    @property
    def power_request_journal_path(self) -> Path:
        output = self.power_transition_path
        return output.with_name(output.name + ".request.json")

    @property
    def power_intent_journal_path(self) -> Path:
        output = self.power_transition_path
        return output.with_name(output.name + ".intent.json")

    @property
    def final_completion_path(self) -> Path:
        return Path(self.local_finalization_root) / "final-completion.json"

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "endpoint": self.endpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise FormalCrossHostFinalizationError(
                "local finalizer config fields differ"
            )
        row = dict(value)
        row["endpoint"] = DriverFileBinding.from_dict(row["endpoint"])
        return cls(**row)  # type: ignore[arg-type]


def publish_cross_host_finalizer_config(
    *,
    endpoint_path: str | Path,
    local_finalization_root: str | Path,
    output_path: str | Path,
) -> PathBoundCrossHostFinalizerConfig:
    config = PathBoundCrossHostFinalizerConfig(
        schema_version=1,
        kind=_LOCAL_CONFIG_KIND,
        endpoint=DriverFileBinding.bind(endpoint_path),
        local_finalization_root=str(
            _absolute(local_finalization_root, "local finalization root")
        ),
    )
    _publish(output_path, config.to_dict(), "local cross-host finalizer config")
    return load_cross_host_finalizer_config(output_path)


def load_cross_host_finalizer_config(
    path: str | Path,
) -> PathBoundCrossHostFinalizerConfig:
    return PathBoundCrossHostFinalizerConfig.from_dict(
        _canonical(path, "local cross-host finalizer config")
    )


class CrossHostFinalizationTransport(Protocol):
    def seal_remote(self) -> None: ...

    def fetch_file(self, remote_path: str, local_path: Path) -> None: ...

    def publish_post_archive_probe(self, remote_output_path: str) -> None: ...

    def archive_runtime(self) -> ProductionArchiveRuntime: ...


class SshCrossHostFinalizationTransport:
    """Passwordless SSH/rsync transport for the production split protocol."""

    def __init__(
        self,
        endpoint: CrossHostSshEndpoint,
        *,
        ssh_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
        rsync_runner: Callable[
            ..., subprocess.CompletedProcess[bytes]
        ] = subprocess.run,
    ) -> None:
        self.endpoint = endpoint
        self.ssh_runner = ssh_runner
        self.rsync_runner = rsync_runner

    def _remote_call(self, *arguments: str) -> None:
        if any(
            _REMOTE_SAFE_PATH.fullmatch(value) is None
            and value not in {"--config", "--output"}
            for value in arguments
        ):
            raise FormalCrossHostFinalizationError(
                "remote finalizer argument is not one safe path/operation"
            )
        remote = shlex.join(
            (
                self.endpoint.remote_python,
                self.endpoint.remote_finalizer_script,
                *arguments,
            )
        )
        completed = self.ssh_runner(
            [*self.endpoint.ssh_argv, self.endpoint.ssh_target, remote],
            check=False,
            capture_output=True,
            shell=False,
        )
        if completed.returncode != 0:
            raise FormalCrossHostFinalizationError(
                f"remote finalizer exited {completed.returncode}"
            )

    def seal_remote(self) -> None:
        self._remote_call(
            "remote-seal",
            "--config",
            self.endpoint.remote_config_path,
        )

    def fetch_file(self, remote_path: str, local_path: Path) -> None:
        if (
            _REMOTE_SAFE_PATH.fullmatch(remote_path) is None
            or not Path(remote_path).is_absolute()
        ):
            raise FormalCrossHostFinalizationError("remote fetch path is unsafe")
        local_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = local_path.with_name(
            f".{local_path.name}.{uuid.uuid4().hex}.partial"
        )
        self.rsync_runner(
            [
                self.endpoint.rsync_executable,
                "-a",
                "--checksum",
                "-e",
                shlex.join(self.endpoint.ssh_argv),
                "--",
                f"{self.endpoint.ssh_target}:{remote_path}",
                str(temporary),
            ],
            check=True,
            shell=False,
        )
        if temporary.is_symlink() or not temporary.is_file():
            raise FormalCrossHostFinalizationError(
                "remote file transfer did not publish one regular file"
            )
        os.replace(temporary, local_path)

    def publish_post_archive_probe(self, remote_output_path: str) -> None:
        self._remote_call(
            "remote-post-archive-probe",
            "--config",
            self.endpoint.remote_config_path,
            "--output",
            remote_output_path,
        )

    def archive_runtime(self) -> ProductionArchiveRuntime:
        return ProductionArchiveRuntime(
            rsync_executable=self.endpoint.rsync_executable,
            runner=self.rsync_runner,
            full_rehydrate=True,
            minimum_local_free_bytes=MINIMUM_LOCAL_ARCHIVE_FREE_BYTES,
            rsync_source=lambda request: (
                f"{self.endpoint.ssh_target}:{request.remote_payload_root}"
            ),
            rsync_remote_shell=shlex.join(self.endpoint.ssh_argv),
        )


def _archive_step_receipt(path: Path, expected_step: str) -> ArchiveStepReceipt:
    value = _canonical(path, f"archive {expected_step} receipt")
    try:
        receipt = ArchiveStepReceipt(**value)
    except (TypeError, ValueError) as error:
        raise FormalCrossHostFinalizationError(
            f"archive {expected_step} receipt differs"
        ) from error
    if receipt.step != expected_step:
        raise FormalCrossHostFinalizationError(
            f"archive {expected_step} receipt has another step"
        )
    return receipt


def _run_local_archive(
    *,
    config: PathBoundCrossHostFinalizerConfig,
    closure: FormalRemoteScientificClosureReceipt,
    runtime: ProductionArchiveRuntime,
) -> tuple[ArchiveRequest, ArchiveStepReceipt, ArchiveStepReceipt, ArchiveStepReceipt]:
    if not runtime.full_rehydrate:
        raise FormalCrossHostFinalizationError(
            "whole-run archive requires full rehydrate"
        )
    request = ArchiveRequest(
        archive_id=f"cross-host-whole-run-{closure.receipt_sha256[:24]}",
        safe_boundary="FORMAL_CROSS_HOST_WHOLE_RUN_SCIENTIFIC_CLOSURE_V1",
        remote_payload_root=closure.remote_payload_root,
        local_partial_root=str(config.archive_partial_root),
        local_final_root=str(config.archive_final_root),
        remote_manifest_sha256=closure.archive_manifest_sha256,
        predicted_payload_bytes=closure.archive_payload_bytes,
    )
    if config.transfer_receipt_path.exists():
        transfer = _archive_step_receipt(config.transfer_receipt_path, "TRANSFER")
    else:
        transfer = runtime.transfer(request, None)
        _publish(
            config.transfer_receipt_path,
            asdict(transfer),
            "archive transfer receipt",
        )
    if config.local_sha_receipt_path.exists():
        local_sha = _archive_step_receipt(
            config.local_sha_receipt_path,
            "LOCAL_SHA_VERIFY",
        )
    else:
        local_sha = runtime.verify_local_sha(request, transfer)
        _publish(
            config.local_sha_receipt_path,
            asdict(local_sha),
            "archive local SHA receipt",
        )
    if config.rehydrate_receipt_path.exists():
        rehydrate = _archive_step_receipt(
            config.rehydrate_receipt_path,
            "REHYDRATE_VERIFY",
        )
    else:
        rehydrate = runtime.rehydrate(request, local_sha)
        _publish(
            config.rehydrate_receipt_path,
            asdict(rehydrate),
            "archive rehydrate receipt",
        )
    expected = (
        closure.archive_manifest_sha256,
        closure.archive_file_count,
        closure.archive_payload_bytes,
    )
    for receipt in (transfer, local_sha, rehydrate):
        if (
            receipt.manifest_sha256,
            receipt.checked_file_count,
            receipt.checked_bytes,
        ) != expected:
            raise FormalCrossHostFinalizationError(
                "archive step does not cover the exact whole-run manifest"
            )
    if rehydrate.content_tree_sha256 is None:
        raise AssertionError("full rehydrate receipt lacks a content-tree digest")
    return request, transfer, local_sha, rehydrate


def _binding_by_remote_path(
    closure: FormalRemoteScientificClosureReceipt,
) -> dict[str, RemoteArchivePathBinding]:
    return {row.remote_absolute_path: row for row in closure.archive_path_bindings}


def _local_archive_member(
    archive_root: Path,
    binding: RemoteArchivePathBinding,
) -> Path:
    source = archive_root / binding.archive_relative_path
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size != binding.size_bytes
        or _file_sha(source) != binding.sha256
    ):
        raise FormalCrossHostFinalizationError(
            "local archive member differs from remote path binding"
        )
    return source


def _deep_validate_local_archive(
    *,
    config: PathBoundCrossHostFinalizerConfig,
    closure: FormalRemoteScientificClosureReceipt,
) -> dict[str, object]:
    root = config.archive_final_root
    manifest = _canonical(root / "sha256_manifest.json", "local archive manifest")
    if _file_sha(root / "sha256_manifest.json") != closure.archive_manifest_sha256:
        raise FormalCrossHostFinalizationError(
            "local archive manifest does not match remote closure"
        )
    checked_bytes = _verify_remote_archive(root, manifest)
    if (
        len(manifest["files"]) != closure.archive_file_count
        or checked_bytes != closure.archive_payload_bytes
    ):
        raise FormalCrossHostFinalizationError(
            "local archive coverage differs from remote closure"
        )
    mapping_value = _canonical(
        root / "remote_path_bindings.json",
        "local remote-path mapping",
    )
    mapped = tuple(
        RemoteArchivePathBinding.from_dict(row)
        for row in mapping_value.get("bindings", [])
    )
    if mapped != closure.archive_path_bindings:
        raise FormalCrossHostFinalizationError(
            "archive companion mapping differs from scientific closure"
        )
    for binding in mapped:
        _local_archive_member(root, binding)
    digests = {row.sha256 for row in mapped}
    if not closure.readiness.required_archive_sha256s.issubset(digests):
        raise FormalCrossHostFinalizationError(
            "local archive omits required scientific evidence"
        )
    by_path = _binding_by_remote_path(closure)
    operator_binding = next(
        (
            row
            for path, row in by_path.items()
            if path.endswith("/operator-snapshot.json")
        ),
        None,
    )
    if operator_binding is None:
        raise FormalCrossHostFinalizationError(
            "archive omits the closed operator snapshot"
        )
    operator_source = _local_archive_member(root, operator_binding)
    if _file_sha(operator_source) != closure.operator_snapshot_sha256:
        raise FormalCrossHostFinalizationError(
            "closed operator snapshot digest differs"
        )
    operator_snapshot = _canonical(operator_source, "archived operator snapshot")
    if (
        operator_snapshot.get("run_id") != closure.run_id
        or operator_snapshot.get("snapshot", {}).get("dispatch_state") != "STOP"
        or len(operator_snapshot.get("snapshot", {}).get("controller_nodes", [])) != 21
        or any(
            row.get("state") != "REDUCED"
            for row in operator_snapshot.get("snapshot", {}).get("controller_nodes", [])
        )
    ):
        raise FormalCrossHostFinalizationError(
            "archived operator snapshot is not the closed 21-node DAG"
        )
    if not set(_controller_artifact_paths(operator_snapshot["snapshot"])).issubset(
        by_path
    ):
        raise FormalCrossHostFinalizationError(
            "archive companion mapping omits a controller artifact path"
        )
    sqlite_backup_sha = _require_sha(
        operator_snapshot.get("sqlite_backup_sha256"),
        "operator SQLite backup",
    )
    if sqlite_backup_sha not in digests:
        raise FormalCrossHostFinalizationError(
            "archive omits the SQLite snapshot bound by the operator snapshot"
        )
    progress_binding = next(
        (
            row
            for path, row in by_path.items()
            if path.endswith("/progress/export_manifest.json")
        ),
        None,
    )
    if (
        progress_binding is None
        or progress_binding.sha256 != closure.progress_export_manifest_sha256
    ):
        raise FormalCrossHostFinalizationError(
            "archive omits the final progress manifest"
        )
    for name, digest in closure.progress_export_files:
        match = next(
            (
                row
                for path, row in by_path.items()
                if path.endswith(f"/progress/{name}")
            ),
            None,
        )
        if match is None or match.sha256 != digest:
            raise FormalCrossHostFinalizationError(
                f"archive omits final progress projection {name}"
            )
    return operator_snapshot


@dataclass(frozen=True)
class FormalLocalPrePoweroffComposite:
    schema_version: Literal[1]
    kind: Literal["formal_local_pre_poweroff_composite"]
    protocol_sha256: str
    trust: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]
    run_id: str
    instance_uuid: str
    composed_at_ns: int
    remote_scientific_closure: FinalAuditArtifactBinding
    archive_transfer: FinalAuditArtifactBinding
    archive_local_sha: FinalAuditArtifactBinding
    archive_full_rehydrate: FinalAuditArtifactBinding
    archive_local_root: str
    archive_manifest_sha256: str
    archive_content_tree_sha256: str
    archive_file_count: int
    archive_payload_bytes: int
    post_archive_probe_wrapper: FinalAuditArtifactBinding
    shutdown_probe: FinalAuditArtifactBinding
    remote_eviction_authorized: Literal[True]
    remote_deletion_performed: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _COMPOSITE_KIND
            or self.protocol_sha256 != _PROTOCOL_SHA256
            or self.trust != TRUSTED_SINGLE_OPERATOR_EMPIRICAL
            or self.formal_measured is not False
            or not self.run_id
            or not self.instance_uuid.startswith("pro-")
            or type(self.composed_at_ns) is not int
            or self.composed_at_ns < 1
            or self.remote_eviction_authorized is not True
            or self.remote_deletion_performed is not False
        ):
            raise FormalCrossHostFinalizationError(
                "local pre-power composite identity differs"
            )
        for binding in (
            self.remote_scientific_closure,
            self.archive_transfer,
            self.archive_local_sha,
            self.archive_full_rehydrate,
            self.post_archive_probe_wrapper,
            self.shutdown_probe,
        ):
            if type(binding) is not FinalAuditArtifactBinding:
                raise TypeError("local composite artifact binding differs")
        root = _absolute(self.archive_local_root, "local archive root")
        if root.is_symlink() or not root.is_dir():
            raise FormalCrossHostFinalizationError("local archive root is not present")
        _require_sha(self.archive_manifest_sha256, "local archive manifest")
        _require_sha(self.archive_content_tree_sha256, "local archive tree")
        if (
            isinstance(self.archive_file_count, bool)
            or not isinstance(self.archive_file_count, int)
            or self.archive_file_count < 1
            or isinstance(self.archive_payload_bytes, bool)
            or not isinstance(self.archive_payload_bytes, int)
            or self.archive_payload_bytes < 0
        ):
            raise FormalCrossHostFinalizationError("local archive coverage is invalid")

    @property
    def receipt_sha256(self) -> str:
        return _sha(self.to_dict(include_receipt_sha256=False))

    def to_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "remote_scientific_closure": self.remote_scientific_closure.to_dict(),
            "archive_transfer": self.archive_transfer.to_dict(),
            "archive_local_sha": self.archive_local_sha.to_dict(),
            "archive_full_rehydrate": self.archive_full_rehydrate.to_dict(),
            "post_archive_probe_wrapper": self.post_archive_probe_wrapper.to_dict(),
            "shutdown_probe": self.shutdown_probe.to_dict(),
        }
        if include_receipt_sha256:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise FormalCrossHostFinalizationError(
                "local pre-power composite is not one object"
            )
        row = dict(value)
        expected = _require_sha(
            row.pop("receipt_sha256", None),
            "local pre-power composite",
        )
        if set(row) != set(cls.__dataclass_fields__):
            raise FormalCrossHostFinalizationError(
                "local pre-power composite fields differ"
            )
        for name in (
            "remote_scientific_closure",
            "archive_transfer",
            "archive_local_sha",
            "archive_full_rehydrate",
            "post_archive_probe_wrapper",
            "shutdown_probe",
        ):
            row[name] = FinalAuditArtifactBinding.from_dict(row[name])
        receipt = cls(**row)  # type: ignore[arg-type]
        if receipt.receipt_sha256 != expected:
            raise FormalCrossHostFinalizationError(
                "local pre-power composite digest differs"
            )
        return receipt


def load_local_pre_poweroff_composite(
    path: str | Path,
) -> FormalLocalPrePoweroffComposite:
    receipt = FormalLocalPrePoweroffComposite.from_dict(
        _canonical(path, "local pre-power composite")
    )
    for label, binding in (
        ("remote closure", receipt.remote_scientific_closure),
        ("archive transfer", receipt.archive_transfer),
        ("archive local SHA", receipt.archive_local_sha),
        ("archive full rehydrate", receipt.archive_full_rehydrate),
        ("post-archive probe", receipt.post_archive_probe_wrapper),
        ("shutdown probe", receipt.shutdown_probe),
    ):
        binding.reopen(label=f"local composite {label}")
    return receipt


def _validate_post_archive_probe(
    *,
    closure: FormalRemoteScientificClosureReceipt,
    wrapper_path: Path,
    raw_probe_path: Path,
    now_ns: int,
) -> dict[str, object]:
    wrapper = _canonical(wrapper_path, "post-archive probe wrapper")
    probe = wrapper.get("raw_probe")
    if (
        set(wrapper)
        != {
            "schema_version",
            "kind",
            "scientific_closure_sha256",
            "raw_probe",
            "raw_probe_sha256",
            "database_semantic_sha256",
            "no_remote_sqlite_write_performed",
        }
        or wrapper.get("schema_version") != 1
        or wrapper.get("kind") != _POST_ARCHIVE_PROBE_KIND
        or wrapper.get("scientific_closure_sha256") != closure.receipt_sha256
        or type(probe) is not dict
        or wrapper.get("raw_probe_sha256") != _sha(probe)
        or wrapper.get("no_remote_sqlite_write_performed") is not True
    ):
        raise FormalCrossHostFinalizationError(
            "post-archive probe wrapper lineage differs"
        )
    try:
        typed = AutoDlPowerOffSafetyProbe(**probe)
    except (TypeError, ValueError) as error:
        raise FormalCrossHostFinalizationError(
            "post-archive shutdown probe is unsafe"
        ) from error
    if (
        typed.run_id != closure.run_id
        or typed.instance_uuid != closure.instance_uuid
        or typed.observed_at_ns > now_ns
        or now_ns - typed.observed_at_ns > 300 * 1_000_000_000
        or not shutdown_probe_is_safe(probe)
    ):
        raise FormalCrossHostFinalizationError(
            "post-archive shutdown probe is stale or unsafe"
        )
    _publish(raw_probe_path, probe, "local raw shutdown probe")
    return wrapper


@dataclass(frozen=True)
class FormalCrossHostFinalCompletion:
    schema_version: Literal[1]
    kind: Literal["formal_cross_host_final_completion"]
    status: Literal["COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"]
    protocol_sha256: str
    trust: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]
    run_id: str
    instance_uuid: str
    finalized_at_ns: int
    pre_poweroff_composite: FinalAuditArtifactBinding
    power_request_journal: FinalAuditArtifactBinding
    power_transition_evidence: FinalAuditArtifactBinding
    provider_request_id: str
    provider_sample_id: str
    provider_response_sha256: str
    archive_manifest_sha256: str
    archive_content_tree_sha256: str
    compute_gpu_seconds: float
    reserved_gpu_seconds: float
    allocated_billed_gpu_seconds: float
    whole_instance_billed_gpu_seconds: float
    compute_gpu_hours: float
    reserved_gpu_hours: float
    allocated_billed_gpu_hours: float
    whole_instance_billed_gpu_hours: float
    powered_wall_time_seconds: float
    wall_time_seconds: float
    wall_time_hours: float
    archive_and_rehydrate_wall_time_seconds: float
    archive_and_rehydrate_billed_gpu_seconds: float
    idle_and_control_gpu_seconds: float
    idle_archive_and_control_gpu_seconds: float
    provider_billing_intervals: tuple[Mapping[str, Any], ...]
    activity_accounting: Mapping[str, Any]
    remote_sqlite_writes_after_closure: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _FINAL_KIND
            or self.status != "COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"
            or self.protocol_sha256 != _PROTOCOL_SHA256
            or self.trust != TRUSTED_SINGLE_OPERATOR_EMPIRICAL
            or self.formal_measured is not False
            or not self.run_id
            or not self.instance_uuid.startswith("pro-")
            or type(self.finalized_at_ns) is not int
            or self.finalized_at_ns < 1
            or not self.provider_request_id
            or self.remote_sqlite_writes_after_closure is not False
        ):
            raise FormalCrossHostFinalizationError(
                "cross-host final completion identity differs"
            )
        if any(
            type(row) is not FinalAuditArtifactBinding
            for row in (
                self.pre_poweroff_composite,
                self.power_request_journal,
                self.power_transition_evidence,
            )
        ):
            raise TypeError("final completion artifact binding differs")
        for label, digest in (
            ("provider sample", self.provider_sample_id),
            ("provider response", self.provider_response_sha256),
            ("archive manifest", self.archive_manifest_sha256),
            ("archive content tree", self.archive_content_tree_sha256),
        ):
            _require_sha(digest, label)
        for label, value in (
            ("compute", self.compute_gpu_seconds),
            ("reserved", self.reserved_gpu_seconds),
            ("allocated billed", self.allocated_billed_gpu_seconds),
            ("whole-instance billed", self.whole_instance_billed_gpu_seconds),
            ("compute hours", self.compute_gpu_hours),
            ("reserved hours", self.reserved_gpu_hours),
            ("allocated billed hours", self.allocated_billed_gpu_hours),
            (
                "whole-instance billed hours",
                self.whole_instance_billed_gpu_hours,
            ),
            ("powered wall", self.powered_wall_time_seconds),
            ("wall", self.wall_time_seconds),
            ("wall hours", self.wall_time_hours),
            ("archive/rehydrate wall", self.archive_and_rehydrate_wall_time_seconds),
            (
                "archive/rehydrate billed",
                self.archive_and_rehydrate_billed_gpu_seconds,
            ),
            ("idle/control", self.idle_and_control_gpu_seconds),
            ("idle/archive/control", self.idle_archive_and_control_gpu_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise FormalCrossHostFinalizationError(
                    f"final {label} accounting is invalid"
                )
        if self.reserved_gpu_seconds + 1e-9 < self.compute_gpu_seconds:
            raise FormalCrossHostFinalizationError(
                "reserved GPU time is below compute GPU time"
            )
        for seconds, hours in (
            (self.compute_gpu_seconds, self.compute_gpu_hours),
            (self.reserved_gpu_seconds, self.reserved_gpu_hours),
            (self.allocated_billed_gpu_seconds, self.allocated_billed_gpu_hours),
            (
                self.whole_instance_billed_gpu_seconds,
                self.whole_instance_billed_gpu_hours,
            ),
            (self.wall_time_seconds, self.wall_time_hours),
        ):
            if not math.isclose(hours, seconds / 3600.0, abs_tol=1e-12):
                raise FormalCrossHostFinalizationError(
                    "final seconds/hour accounting differs"
                )
        if not self.provider_billing_intervals or any(
            not bool(row.get("complete")) for row in self.provider_billing_intervals
        ):
            raise FormalCrossHostFinalizationError(
                "final provider billing is not closed"
            )
        if type(self.activity_accounting) is not dict:
            raise FormalCrossHostFinalizationError(
                "final activity accounting is not one object"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha(self.to_dict(include_receipt_sha256=False))

    def to_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "pre_poweroff_composite": self.pre_poweroff_composite.to_dict(),
            "power_request_journal": self.power_request_journal.to_dict(),
            "power_transition_evidence": self.power_transition_evidence.to_dict(),
            "provider_billing_intervals": [
                _jsonable_interval(row) for row in self.provider_billing_intervals
            ],
            "activity_accounting": dict(self.activity_accounting),
        }
        if include_receipt_sha256:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise FormalCrossHostFinalizationError(
                "cross-host final completion is not one object"
            )
        row = dict(value)
        expected = _require_sha(
            row.pop("receipt_sha256", None),
            "cross-host final completion",
        )
        if set(row) != set(cls.__dataclass_fields__):
            raise FormalCrossHostFinalizationError(
                "cross-host final completion fields differ"
            )
        for name in (
            "pre_poweroff_composite",
            "power_request_journal",
            "power_transition_evidence",
        ):
            row[name] = FinalAuditArtifactBinding.from_dict(row[name])
        row["provider_billing_intervals"] = tuple(row["provider_billing_intervals"])
        receipt = cls(**row)  # type: ignore[arg-type]
        if receipt.receipt_sha256 != expected:
            raise FormalCrossHostFinalizationError(
                "cross-host final completion digest differs"
            )
        return receipt


def load_cross_host_final_completion(
    path: str | Path,
) -> FormalCrossHostFinalCompletion:
    receipt = FormalCrossHostFinalCompletion.from_dict(
        _canonical(path, "cross-host final completion")
    )
    receipt.pre_poweroff_composite.reopen(label="final pre-power composite")
    receipt.power_request_journal.reopen(label="final power request journal")
    receipt.power_transition_evidence.reopen(label="final power evidence")
    return receipt


def _load_provider_shutdown(
    path: Path,
    *,
    run_id: str,
    instance_uuid: str,
    authority_sha256: str,
) -> tuple[AutoDlPowerTransitionReceipt, ProviderRuntimeSample]:
    value = _canonical(path, "stateless power transition evidence")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "autodl_stateless_power_off_transition_evidence"
        or value.get("run_id") != run_id
        or value.get("shutdown_authority_sha256") != authority_sha256
        or type(value.get("receipt")) is not dict
        or type(value.get("final_provider_evidence")) is not dict
    ):
        raise FormalCrossHostFinalizationError(
            "stateless power transition lineage differs"
        )
    try:
        receipt = AutoDlPowerTransitionReceipt(**value["receipt"])
        sample = ProviderRuntimeSample(**value["final_provider_evidence"]["sample"])
    except (KeyError, TypeError, ValueError) as error:
        raise FormalCrossHostFinalizationError(
            "stateless provider shutdown evidence differs"
        ) from error
    if (
        receipt.operation != "power_off"
        or receipt.target_state != "shutdown"
        or receipt.instance_uuid != instance_uuid
        or sample.instance_uuid != instance_uuid
        or sample.state != "shutdown"
        or receipt.provider_sample_id != sample.sample_id
        or receipt.provider_response_sha256 != sample.response_sha256
    ):
        raise FormalCrossHostFinalizationError(
            "AutoDL status/list shutdown confirmation differs"
        )
    return receipt, sample


def _closed_provider_intervals(
    closure: FormalRemoteScientificClosureReceipt,
    sample: ProviderRuntimeSample,
) -> tuple[dict[str, object], ...]:
    open_interval = dict(
        _validate_open_billing(
            closure.provider_billing_intervals,
            instance_uuid=closure.instance_uuid,
        )
    )
    start = int(open_interval["provider_started_at_ns"])
    stop = sample.provider_stopped_at_ns
    if (
        stop is None
        or sample.provider_started_at_ns != start
        or sample.gpu_count != open_interval["gpu_count"]
        or stop < int(open_interval["provider_stopped_or_observed_at_ns"])
    ):
        raise FormalCrossHostFinalizationError(
            "shutdown sample does not close the exact open billing interval"
        )
    duration = (stop - start) / 1e9
    responses = sorted({*open_interval["response_sha256s"], sample.response_sha256})
    closed = {
        **open_interval,
        "provider_stopped_or_observed_at_ns": stop,
        "complete": True,
        "duration_seconds": duration,
        "whole_instance_billed_gpu_seconds": duration * sample.gpu_count,
        "sample_count": int(open_interval["sample_count"]) + 1,
        "response_sha256s": responses,
    }
    rows = [
        _jsonable_interval(row)
        for row in closure.provider_billing_intervals
        if bool(row["complete"])
    ]
    rows.append(closed)
    return tuple(rows)


def _journal_authority(path: Path) -> str:
    value = _canonical(path, "stateless power request journal")
    redacted = value.get("redacted_provider_response")
    if (
        value.get("kind") != "autodl_stateless_power_mutation_request_journal"
        or value.get("operation") != "power_off"
        or type(redacted) is not dict
        or redacted.get("code") != "Success"
        or not value.get("provider_request_id")
    ):
        raise FormalCrossHostFinalizationError(
            "power_off request journal lacks code Success"
        )
    return _require_sha(
        value.get("shutdown_authority_sha256"),
        "power shutdown authority",
    )


def _intent_authority(path: Path) -> str:
    value = _canonical(path, "stateless power mutation intent")
    unsigned = dict(value)
    digest = unsigned.pop("intent_sha256", None)
    if (
        value.get("kind") != "autodl_stateless_power_mutation_intent"
        or value.get("operation") != "power_off"
        or digest != _sha(unsigned)
    ):
        raise FormalCrossHostFinalizationError(
            "stateless power mutation intent differs"
        )
    return _require_sha(
        value.get("shutdown_authority_sha256"),
        "power intent shutdown authority",
    )


class FormalCrossHostProductionFinalizer:
    """Crash-resumable local supervisor for parts B and C."""

    def __init__(
        self,
        config: PathBoundCrossHostFinalizerConfig,
        *,
        transport: CrossHostFinalizationTransport | None = None,
        archive_runtime: ProductionArchiveRuntime | None = None,
        environment: Mapping[str, str] | None = None,
        provider_client_factory: Callable[[str], AutoDlProApiClient] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        sleeper: Callable[[float], None] = time.sleep,
        maximum_confirmation_attempts: int = 24,
        confirmation_interval_seconds: float = 5.0,
    ) -> None:
        if type(config) is not PathBoundCrossHostFinalizerConfig:
            raise TypeError("cross-host finalizer requires an exact path config")
        self.config = config
        endpoint = load_cross_host_ssh_endpoint(config.endpoint.absolute_path)
        self.transport = transport or SshCrossHostFinalizationTransport(endpoint)
        self.archive = archive_runtime
        self.environment = environment
        self.provider_client_factory = provider_client_factory
        self.clock_ns = clock_ns
        self.sleeper = sleeper
        self.maximum_confirmation_attempts = maximum_confirmation_attempts
        self.confirmation_interval_seconds = confirmation_interval_seconds

    def run(self) -> FormalCrossHostFinalCompletion:
        config = self.config
        root = Path(config.local_finalization_root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if config.final_completion_path.exists():
            return load_cross_host_final_completion(config.final_completion_path)
        endpoint = load_cross_host_ssh_endpoint(config.endpoint.absolute_path)
        if not config.closure_path.exists():
            self.transport.seal_remote()
            self.transport.fetch_file(
                endpoint.remote_closure_path,
                config.closure_path,
            )
        closure = load_remote_scientific_closure(config.closure_path)
        archive_runtime = self.archive or self.transport.archive_runtime()
        _request, transfer, local_sha, rehydrate = _run_local_archive(
            config=config,
            closure=closure,
            runtime=archive_runtime,
        )
        operator_snapshot = _deep_validate_local_archive(
            config=config,
            closure=closure,
        )
        composite_path, composite = self._power_authority(
            endpoint=endpoint,
            closure=closure,
            transfer=transfer,
            local_sha=local_sha,
            rehydrate=rehydrate,
        )
        if not config.power_transition_path.exists():
            transition_autodl_instance_power_stateless(
                operation="power_off",
                instance_uuid=closure.instance_uuid,
                run_id=closure.run_id,
                shutdown_authority_sha256=composite.receipt_sha256,
                output_path=config.power_transition_path,
                safety_probe_path=composite.shutdown_probe.absolute_path,
                token_environment_name=_TOKEN_ENV,
                environment=self.environment,
                client_factory=self.provider_client_factory,
                clock_ns=self.clock_ns,
                sleeper=self.sleeper,
                maximum_confirmation_attempts=self.maximum_confirmation_attempts,
                confirmation_interval_seconds=self.confirmation_interval_seconds,
            )
        journal_authority = _journal_authority(config.power_request_journal_path)
        if journal_authority != composite.receipt_sha256:
            raise FormalCrossHostFinalizationError(
                "power request journal points at another pre-power composite"
            )
        transition, final_sample = _load_provider_shutdown(
            config.power_transition_path,
            run_id=closure.run_id,
            instance_uuid=closure.instance_uuid,
            authority_sha256=composite.receipt_sha256,
        )
        intervals = _closed_provider_intervals(closure, final_sample)
        whole_billed = sum(
            float(row["whole_instance_billed_gpu_seconds"]) for row in intervals
        )
        powered_wall = sum(float(row["duration_seconds"]) for row in intervals)
        start = min(int(row["provider_started_at_ns"]) for row in intervals)
        stop = max(int(row["provider_stopped_or_observed_at_ns"]) for row in intervals)
        wall = (stop - start) / 1e9
        allocated = closure.readiness.allocated_billed_gpu_seconds
        residual = whole_billed - allocated
        if residual < -1e-8:
            raise FormalCrossHostFinalizationError(
                "whole-instance billing is below allocated attempt billing"
            )
        activity = operator_snapshot.get("activity_accounting")
        if type(activity) is not dict:
            raise FormalCrossHostFinalizationError(
                "archived operator snapshot lacks activity accounting"
            )
        archive_wall = max(
            0.0,
            (composite.composed_at_ns - closure.closed_at_ns) / 1e9,
        )
        open_interval = _validate_open_billing(
            closure.provider_billing_intervals,
            instance_uuid=closure.instance_uuid,
        )
        archive_billed = archive_wall * int(open_interval["gpu_count"])
        if archive_billed > residual + 1e-8:
            raise FormalCrossHostFinalizationError(
                "archive billing exceeds the unallocated provider interval"
            )
        idle_control = max(0.0, residual - archive_billed)
        activity = {
            **activity,
            "whole_run_archive_and_rehydrate": {
                "wall_time_seconds": archive_wall,
                "whole_instance_billed_gpu_seconds": archive_billed,
            },
            "provider_idle_and_control": {
                "whole_instance_billed_gpu_seconds": idle_control,
            },
        }
        receipt = FormalCrossHostFinalCompletion(
            schema_version=1,
            kind=_FINAL_KIND,
            status="COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL",
            protocol_sha256=_PROTOCOL_SHA256,
            trust=TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
            formal_measured=False,
            run_id=closure.run_id,
            instance_uuid=closure.instance_uuid,
            finalized_at_ns=int(self.clock_ns()),
            pre_poweroff_composite=FinalAuditArtifactBinding.bind(
                composite_path,
                label="pre-power composite",
            ),
            power_request_journal=FinalAuditArtifactBinding.bind(
                config.power_request_journal_path,
                label="power request journal",
            ),
            power_transition_evidence=FinalAuditArtifactBinding.bind(
                config.power_transition_path,
                label="power transition evidence",
            ),
            provider_request_id=transition.provider_request_id,
            provider_sample_id=transition.provider_sample_id,
            provider_response_sha256=transition.provider_response_sha256,
            archive_manifest_sha256=closure.archive_manifest_sha256,
            archive_content_tree_sha256=(rehydrate.content_tree_sha256 or ""),
            compute_gpu_seconds=closure.readiness.compute_gpu_seconds,
            reserved_gpu_seconds=closure.readiness.reserved_gpu_seconds,
            allocated_billed_gpu_seconds=allocated,
            whole_instance_billed_gpu_seconds=whole_billed,
            compute_gpu_hours=closure.readiness.compute_gpu_seconds / 3600.0,
            reserved_gpu_hours=closure.readiness.reserved_gpu_seconds / 3600.0,
            allocated_billed_gpu_hours=allocated / 3600.0,
            whole_instance_billed_gpu_hours=whole_billed / 3600.0,
            powered_wall_time_seconds=powered_wall,
            wall_time_seconds=wall,
            wall_time_hours=wall / 3600.0,
            archive_and_rehydrate_wall_time_seconds=archive_wall,
            archive_and_rehydrate_billed_gpu_seconds=archive_billed,
            idle_and_control_gpu_seconds=idle_control,
            idle_archive_and_control_gpu_seconds=max(0.0, residual),
            provider_billing_intervals=intervals,
            activity_accounting=activity,
            remote_sqlite_writes_after_closure=False,
        )
        _publish(
            config.final_completion_path,
            receipt.to_dict(),
            "cross-host final completion",
        )
        return load_cross_host_final_completion(config.final_completion_path)

    def _power_authority(
        self,
        *,
        endpoint: CrossHostSshEndpoint,
        closure: FormalRemoteScientificClosureReceipt,
        transfer: ArchiveStepReceipt,
        local_sha: ArchiveStepReceipt,
        rehydrate: ArchiveStepReceipt,
    ) -> tuple[Path, FormalLocalPrePoweroffComposite]:
        config = self.config
        config.composite_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if config.power_request_journal_path.exists():
            authority = _journal_authority(config.power_request_journal_path)
            path = config.composite_root / f"{authority}.json"
            return path, load_local_pre_poweroff_composite(path)
        if config.power_intent_journal_path.exists():
            authority = _intent_authority(config.power_intent_journal_path)
            path = config.composite_root / f"{authority}.json"
            return path, load_local_pre_poweroff_composite(path)

        probe_id = uuid.uuid4().hex
        remote_probe = str(Path(endpoint.remote_probe_root) / f"{probe_id}.json")
        local_wrapper = config.probe_root / f"{probe_id}.wrapper.json"
        local_raw = config.probe_root / f"{probe_id}.raw.json"
        self.transport.publish_post_archive_probe(remote_probe)
        self.transport.fetch_file(remote_probe, local_wrapper)
        _validate_post_archive_probe(
            closure=closure,
            wrapper_path=local_wrapper,
            raw_probe_path=local_raw,
            now_ns=int(self.clock_ns()),
        )
        content_tree = rehydrate.content_tree_sha256
        if content_tree is None:
            raise AssertionError("full rehydrate lacks a content-tree digest")
        composite = FormalLocalPrePoweroffComposite(
            schema_version=1,
            kind=_COMPOSITE_KIND,
            protocol_sha256=_PROTOCOL_SHA256,
            trust=TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
            formal_measured=False,
            run_id=closure.run_id,
            instance_uuid=closure.instance_uuid,
            composed_at_ns=int(self.clock_ns()),
            remote_scientific_closure=FinalAuditArtifactBinding.bind(
                config.closure_path,
                label="remote scientific closure",
            ),
            archive_transfer=FinalAuditArtifactBinding.bind(
                config.transfer_receipt_path,
                label="archive transfer",
            ),
            archive_local_sha=FinalAuditArtifactBinding.bind(
                config.local_sha_receipt_path,
                label="archive local SHA",
            ),
            archive_full_rehydrate=FinalAuditArtifactBinding.bind(
                config.rehydrate_receipt_path,
                label="archive full rehydrate",
            ),
            archive_local_root=str(config.archive_final_root),
            archive_manifest_sha256=closure.archive_manifest_sha256,
            archive_content_tree_sha256=content_tree,
            archive_file_count=closure.archive_file_count,
            archive_payload_bytes=closure.archive_payload_bytes,
            post_archive_probe_wrapper=FinalAuditArtifactBinding.bind(
                local_wrapper,
                label="post-archive probe wrapper",
            ),
            shutdown_probe=FinalAuditArtifactBinding.bind(
                local_raw,
                label="raw shutdown probe",
            ),
            remote_eviction_authorized=True,
            remote_deletion_performed=False,
        )
        path = config.composite_root / f"{composite.receipt_sha256}.json"
        _publish(path, composite.to_dict(), "local pre-power composite")
        return path, load_local_pre_poweroff_composite(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("remote-seal", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    probe = commands.add_parser("remote-post-archive-probe")
    probe.add_argument("--config", required=True)
    probe.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "remote-seal":
        result: object = seal_remote_scientific_closure(
            load_remote_closure_config(arguments.config)
        ).to_dict()
    elif arguments.command == "remote-post-archive-probe":
        result = publish_remote_post_archive_probe(
            load_remote_closure_config(arguments.config),
            output_path=arguments.output,
        )
    else:
        result = (
            FormalCrossHostProductionFinalizer(
                load_cross_host_finalizer_config(arguments.config)
            )
            .run()
            .to_dict()
        )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "CrossHostFinalizationTransport",
    "CrossHostSshEndpoint",
    "FormalCrossHostFinalCompletion",
    "FormalCrossHostFinalizationError",
    "FormalCrossHostProductionFinalizer",
    "FormalLocalPrePoweroffComposite",
    "FormalRemoteScientificClosureReceipt",
    "PathBoundCrossHostFinalizerConfig",
    "PathBoundRemoteClosureConfig",
    "RemoteArchivePathBinding",
    "SshCrossHostFinalizationTransport",
    "load_cross_host_final_completion",
    "load_cross_host_finalizer_config",
    "load_cross_host_ssh_endpoint",
    "load_local_pre_poweroff_composite",
    "load_remote_closure_config",
    "load_remote_scientific_closure",
    "main",
    "publish_cross_host_finalizer_config",
    "publish_cross_host_ssh_endpoint",
    "publish_remote_closure_config",
    "publish_remote_post_archive_probe",
    "seal_remote_scientific_closure",
]


if __name__ == "__main__":
    raise SystemExit(main())
