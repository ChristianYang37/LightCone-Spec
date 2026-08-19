"""Fail-closed final completion audit for the trusted single-operator run.

The DAG controller deliberately stops at ``REDUCED``.  Reduction says that a
node's registered reducer ran; it does not say that the whole experiment is
archived, statistically reportable, idle, provider-closed, or complete.  This
module supplies that missing boundary in two irreversible, independently
replayable steps:

* :func:`publish_pre_shutdown_audit` proves exact current coverage, durable
  statistics/exports, a fully rehydrated archive, accounting, and a safe idle
  shutdown boundary.
* :func:`publish_final_completion` consumes an *existing* AutoDL power-off
  transition receipt and proves dual endpoint shutdown plus closed provider
  billing before publishing the only final completion receipt.

Neither function accepts a provider token or a network callback.  The final
claim is intentionally trusted empirical evidence and never formal MEASURED
evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self
from urllib.parse import quote

from lightcone_spec.orchestration.autodl_provider_runtime import (
    AutoDlApiResponse,
    AutoDlPowerOffSafetyProbe,
    AutoDlPowerTransitionReceipt,
    AutoDlProviderRuntimeError,
    reduce_autodl_provider_responses,
)
from lightcone_spec.orchestration.experiment_operator import (
    ControllerArtifactBinding,
    ExperimentOperatorStore,
    ProviderRuntimeSample,
)
from lightcone_spec.runtime.proof_artifact import (
    publish_canonical_json_no_replace,
)

TRUSTED_SINGLE_OPERATOR_EMPIRICAL = "trusted_single_operator_empirical_no_signature"
FINAL_ARCHIVE_SAFE_BOUNDARY = "formal_experiment:final:all_evidence_sealed"
_SHA256_HEX = frozenset("0123456789abcdef")
_EXPECTED_EXPORT_FILES = frozenset(
    {
        "stage_plan.csv",
        "cell_ledger.csv",
        "stage_summary.csv",
        "selection_decisions.jsonl",
        "watchdog_events.jsonl",
        "dashboard.md",
        "metrics_long.parquet",
        "instance_billing.csv",
        "controller_state.csv",
    }
)
_NON_HEADLINE_TASKS = frozenset(
    {
        "compile_environment_patch",
        "compile",
        "exactness_memory_telemetry",
        "mechanism_profile_only",
        "deterministic_failure_injection",
        "immutable_metadata_interface_and_fit_preflight",
        "compatibility_decision",
    }
)


def _canonical_payload(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_file(value: object) -> bytes:
    return _canonical_payload(value) + b"\n"


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def _file_semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_file(value)).hexdigest()


FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256 = _semantic_sha256(
    {
        "schema_version": 1,
        "kind": "formal_experiment_final_audit_protocol",
        "controller_terminal_state": "DAG_REDUCED_AWAITING_FINAL_AUDIT",
        "pre_shutdown": [
            "exact_21_node_deep_rebuild",
            "materialization_ledger_bidirectional_exact_coverage",
            "latest_nonlegacy_attempts_complete",
            "selection_and_metric_projection_current",
            "headline_95pct_interval_and_count_semantics",
            "descriptive_rows_without_fake_interval",
            "archive_transfer_local_sha_full_rehydrate_content_tree",
            "progress_export_manifest_deep_replay",
            "compute_reserved_accounting_without_group_double_charge",
            "safe_idle_shutdown_probe",
        ],
        "post_shutdown": [
            "existing_power_off_transition_only",
            "provider_mutation_code_success",
            "status_list_same_uuid_dual_shutdown",
            "provider_billing_intervals_closed",
            "atomic_no_replace_final_receipt",
        ],
        "trust": TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
        "formal_measured": False,
    }
)


class FormalExperimentFinalAuditError(RuntimeError):
    """The experiment cannot truthfully be declared complete."""


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise FormalExperimentFinalAuditError(f"{label} is not a lowercase SHA-256")
    return value


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise FormalExperimentFinalAuditError(f"{label} is empty or malformed")
    return value


def _require_nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalExperimentFinalAuditError(f"{label} is not finite")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise FormalExperimentFinalAuditError(f"{label} is not non-negative")
    return result


def _absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise FormalExperimentFinalAuditError(
            f"{label} must be absolute and normalized"
        )
    return path


def _stable_file_sha256(path: Path, label: str) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise FormalExperimentFinalAuditError(f"{label} is not one regular file")
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
    if identity(before) != identity(after):
        raise FormalExperimentFinalAuditError(f"{label} changed while hashed")
    return digest.hexdigest(), before.st_size


def _load_canonical_object(path: str | Path, label: str) -> dict[str, Any]:
    source = _absolute_path(path, label)
    raw_sha256, size = _stable_file_sha256(source, label)
    if size < 3 or size > 64 * 1024 * 1024:
        raise FormalExperimentFinalAuditError(f"{label} has an invalid size")
    payload = source.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalExperimentFinalAuditError(f"{label} is not JSON") from error
    if type(value) is not dict or payload != _canonical_file(value):
        raise FormalExperimentFinalAuditError(f"{label} is not canonical JSON")
    if hashlib.sha256(payload).hexdigest() != raw_sha256:
        raise AssertionError("stable canonical JSON digest changed")
    return value


@dataclass(frozen=True)
class FinalAuditArtifactBinding:
    """Stable path/raw identity for a completion-audit input."""

    absolute_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _absolute_path(self.absolute_path, "final-audit artifact")
        _require_sha256(self.sha256, "final-audit artifact")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise FormalExperimentFinalAuditError(
                "final-audit artifact size is invalid"
            )

    @classmethod
    def bind(cls, path: str | Path, *, label: str) -> Self:
        source = _absolute_path(path, label)
        digest, size = _stable_file_sha256(source, label)
        return cls(str(source), digest, size)

    def reopen(self, *, label: str) -> Path:
        source = _absolute_path(self.absolute_path, label)
        digest, size = _stable_file_sha256(source, label)
        if digest != self.sha256 or size != self.size_bytes:
            raise FormalExperimentFinalAuditError(f"{label} changed")
        return source

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise FormalExperimentFinalAuditError(
                "final-audit artifact binding fields differ"
            )
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalExperimentPreShutdownAuditReceipt:
    schema_version: int
    kind: Literal["formal_experiment_pre_shutdown_audit"]
    protocol_sha256: str
    run_id: str
    instance_uuid: str
    trust: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]
    controller_state: Literal["DAG_REDUCED_AWAITING_FINAL_AUDIT"]
    audited_at_ns: int
    node_count: int
    expected_cell_count: int
    expected_cell_ids_sha256: str
    latest_complete_attempt_count: int
    retained_retry_attempt_count: int
    selection_decision_count: int
    metric_count: int
    headline_metric_count: int
    coverage_sha256: str
    selection_sha256: str
    metrics_sha256: str
    accounting_sha256: str
    compute_gpu_seconds: float
    reserved_gpu_seconds: float
    allocated_billed_gpu_seconds: float
    observed_whole_instance_billed_gpu_seconds: float
    observed_wall_time_seconds: float
    progress_export_manifest: FinalAuditArtifactBinding
    progress_export_files: tuple[tuple[str, str], ...]
    final_archive_id: str
    final_archive_manifest_sha256: str
    final_archive_content_tree_sha256: str
    final_archive_local_root: str
    shutdown_probe: FinalAuditArtifactBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_experiment_pre_shutdown_audit"
            or self.protocol_sha256 != FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256
            or self.trust != TRUSTED_SINGLE_OPERATOR_EMPIRICAL
            or self.formal_measured is not False
            or self.controller_state != "DAG_REDUCED_AWAITING_FINAL_AUDIT"
            or self.node_count != 21
        ):
            raise FormalExperimentFinalAuditError("pre-shutdown audit identity differs")
        _require_text(self.run_id, "pre-shutdown run ID")
        if not self.instance_uuid.startswith("pro-"):
            raise FormalExperimentFinalAuditError("pre-shutdown instance UUID differs")
        if type(self.audited_at_ns) is not int or self.audited_at_ns < 1:
            raise FormalExperimentFinalAuditError("pre-shutdown audit time is invalid")
        for label, count in (
            ("expected cells", self.expected_cell_count),
            ("latest complete attempts", self.latest_complete_attempt_count),
            ("retained retries", self.retained_retry_attempt_count),
            ("selection decisions", self.selection_decision_count),
            ("metrics", self.metric_count),
            ("headline metrics", self.headline_metric_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise FormalExperimentFinalAuditError(f"{label} count is invalid")
        if (
            self.expected_cell_count < 1
            or self.latest_complete_attempt_count != self.expected_cell_count
            or self.selection_decision_count < 1
            or self.metric_count < 1
            or self.headline_metric_count < 1
        ):
            raise FormalExperimentFinalAuditError(
                "pre-shutdown audit lacks required coverage or statistics"
            )
        for label, digest in (
            ("expected cell set", self.expected_cell_ids_sha256),
            ("coverage", self.coverage_sha256),
            ("selection", self.selection_sha256),
            ("metrics", self.metrics_sha256),
            ("accounting", self.accounting_sha256),
            ("archive manifest", self.final_archive_manifest_sha256),
            ("archive content tree", self.final_archive_content_tree_sha256),
        ):
            _require_sha256(digest, f"pre-shutdown {label}")
        for label, value in (
            ("compute GPU-seconds", self.compute_gpu_seconds),
            ("reserved GPU-seconds", self.reserved_gpu_seconds),
            ("allocated billed GPU-seconds", self.allocated_billed_gpu_seconds),
            (
                "observed whole-instance billed GPU-seconds",
                self.observed_whole_instance_billed_gpu_seconds,
            ),
            ("observed wall seconds", self.observed_wall_time_seconds),
        ):
            _require_nonnegative_finite(value, label)
        if self.reserved_gpu_seconds < self.compute_gpu_seconds:
            raise FormalExperimentFinalAuditError(
                "reserved GPU time is below compute GPU time"
            )
        if type(self.progress_export_manifest) is not FinalAuditArtifactBinding:
            raise TypeError("progress export manifest binding differs")
        if type(self.shutdown_probe) is not FinalAuditArtifactBinding:
            raise TypeError("shutdown probe binding differs")
        if (
            type(self.progress_export_files) is not tuple
            or self.progress_export_files != tuple(sorted(self.progress_export_files))
            or {name for name, _digest in self.progress_export_files}
            != _EXPECTED_EXPORT_FILES
        ):
            raise FormalExperimentFinalAuditError(
                "progress export file binding set differs"
            )
        for name, digest in self.progress_export_files:
            _require_text(name, "progress export filename")
            _require_sha256(digest, f"progress export {name}")
        _require_text(self.final_archive_id, "final archive ID")
        _absolute_path(self.final_archive_local_root, "final archive local root")

    @property
    def receipt_sha256(self) -> str:
        return _semantic_sha256(self.to_dict(include_receipt_sha256=False))

    def to_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "progress_export_manifest": self.progress_export_manifest.to_dict(),
            "progress_export_files": [
                [name, digest] for name, digest in self.progress_export_files
            ],
            "shutdown_probe": self.shutdown_probe.to_dict(),
        }
        if include_receipt_sha256:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise FormalExperimentFinalAuditError(
                "pre-shutdown audit must be one object"
            )
        row = dict(value)
        expected = _require_sha256(
            row.pop("receipt_sha256", None),
            "pre-shutdown audit receipt",
        )
        if set(row) != set(cls.__dataclass_fields__):
            raise FormalExperimentFinalAuditError("pre-shutdown audit fields differ")
        row["progress_export_manifest"] = FinalAuditArtifactBinding.from_dict(
            row["progress_export_manifest"]
        )
        raw_files = row["progress_export_files"]
        if type(raw_files) is not list or any(
            type(item) is not list or len(item) != 2 for item in raw_files
        ):
            raise FormalExperimentFinalAuditError(
                "pre-shutdown progress file bindings differ"
            )
        row["progress_export_files"] = tuple((item[0], item[1]) for item in raw_files)
        row["shutdown_probe"] = FinalAuditArtifactBinding.from_dict(
            row["shutdown_probe"]
        )
        receipt = cls(**row)  # type: ignore[arg-type]
        if receipt.receipt_sha256 != expected:
            raise FormalExperimentFinalAuditError("pre-shutdown audit digest differs")
        return receipt


@dataclass(frozen=True)
class FormalExperimentFinalCompletionReceipt:
    schema_version: int
    kind: Literal["formal_experiment_final_completion"]
    protocol_sha256: str
    status: Literal["COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"]
    trust: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]
    run_id: str
    instance_uuid: str
    finalized_at_ns: int
    pre_shutdown_audit: FinalAuditArtifactBinding
    power_transition_evidence: FinalAuditArtifactBinding
    power_transition_receipt_sha256: str
    provider_request_id: str
    provider_sample_id: str
    provider_response_sha256: str
    shutdown_probe_sha256: str
    coverage_sha256: str
    selection_sha256: str
    metrics_sha256: str
    archive_manifest_sha256: str
    archive_content_tree_sha256: str
    progress_export_manifest_sha256: str
    wall_time_seconds: float
    powered_wall_time_seconds: float
    compute_gpu_hours: float
    reserved_gpu_hours: float
    billed_gpu_hours: float

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_experiment_final_completion"
            or self.protocol_sha256 != FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256
            or self.status != "COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL"
            or self.trust != TRUSTED_SINGLE_OPERATOR_EMPIRICAL
            or self.formal_measured is not False
        ):
            raise FormalExperimentFinalAuditError("final completion identity differs")
        _require_text(self.run_id, "final completion run ID")
        if not self.instance_uuid.startswith("pro-"):
            raise FormalExperimentFinalAuditError(
                "final completion instance UUID differs"
            )
        if type(self.finalized_at_ns) is not int or self.finalized_at_ns < 1:
            raise FormalExperimentFinalAuditError("final completion time is invalid")
        if (
            type(self.pre_shutdown_audit) is not FinalAuditArtifactBinding
            or type(self.power_transition_evidence) is not FinalAuditArtifactBinding
        ):
            raise TypeError("final completion artifact binding differs")
        for label, digest in (
            ("power transition receipt", self.power_transition_receipt_sha256),
            ("provider sample", self.provider_sample_id),
            ("provider response", self.provider_response_sha256),
            ("shutdown probe", self.shutdown_probe_sha256),
            ("coverage", self.coverage_sha256),
            ("selection", self.selection_sha256),
            ("metrics", self.metrics_sha256),
            ("archive manifest", self.archive_manifest_sha256),
            ("archive content tree", self.archive_content_tree_sha256),
            ("progress export manifest", self.progress_export_manifest_sha256),
        ):
            _require_sha256(digest, f"final completion {label}")
        _require_text(self.provider_request_id, "provider request ID")
        for label, value in (
            ("wall time", self.wall_time_seconds),
            ("powered wall time", self.powered_wall_time_seconds),
            ("compute GPU-hours", self.compute_gpu_hours),
            ("reserved GPU-hours", self.reserved_gpu_hours),
            ("billed GPU-hours", self.billed_gpu_hours),
        ):
            _require_nonnegative_finite(value, label)
        if self.reserved_gpu_hours < self.compute_gpu_hours:
            raise FormalExperimentFinalAuditError(
                "final reserved GPU-hours are below compute GPU-hours"
            )

    @property
    def receipt_sha256(self) -> str:
        return _semantic_sha256(self.to_dict(include_receipt_sha256=False))

    def to_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "pre_shutdown_audit": self.pre_shutdown_audit.to_dict(),
            "power_transition_evidence": self.power_transition_evidence.to_dict(),
        }
        if include_receipt_sha256:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise FormalExperimentFinalAuditError(
                "final completion receipt must be one object"
            )
        row = dict(value)
        expected = _require_sha256(
            row.pop("receipt_sha256", None), "final completion receipt"
        )
        if set(row) != set(cls.__dataclass_fields__):
            raise FormalExperimentFinalAuditError("final completion fields differ")
        row["pre_shutdown_audit"] = FinalAuditArtifactBinding.from_dict(
            row["pre_shutdown_audit"]
        )
        row["power_transition_evidence"] = FinalAuditArtifactBinding.from_dict(
            row["power_transition_evidence"]
        )
        receipt = cls(**row)  # type: ignore[arg-type]
        if receipt.receipt_sha256 != expected:
            raise FormalExperimentFinalAuditError("final completion digest differs")
        return receipt


@dataclass(frozen=True)
class _CoverageAudit:
    expected_cell_ids: tuple[str, ...]
    latest_attempts: Mapping[str, Mapping[str, Any]]
    controller_stage_phases: frozenset[tuple[str, str]]
    nonempty_stage_phases: frozenset[tuple[str, str]]
    retained_retry_attempt_count: int
    required_archive_sha256s: frozenset[str]
    coverage_sha256: str
    compute_gpu_seconds: float
    reserved_gpu_seconds: float
    allocated_billed_gpu_seconds: float
    accounting_sha256: str


@dataclass(frozen=True)
class _ProgressAudit:
    manifest: FinalAuditArtifactBinding
    files: tuple[tuple[str, str], ...]
    selections: tuple[dict[str, Any], ...]
    metrics: tuple[dict[str, Any], ...]
    headline_metric_count: int


@dataclass(frozen=True)
class _ArchiveAudit:
    archive_id: str
    manifest_sha256: str
    content_tree_sha256: str
    local_root: str


@dataclass(frozen=True)
class FormalExperimentFinalizationReadiness:
    """Read-only proof that the reduced DAG is ready for final archiving.

    This is deliberately not a completion claim.  It runs the same deep DAG,
    bidirectional ledger, selection, and metric checks used by the final audit,
    but it neither requires nor creates an archive or shutdown probe.
    """

    run_id: str
    node_count: int
    expected_cell_count: int
    latest_complete_attempt_count: int
    retained_retry_attempt_count: int
    selection_decision_count: int
    metric_count: int
    headline_metric_count: int
    expected_cell_ids_sha256: str
    coverage_sha256: str
    accounting_sha256: str
    compute_gpu_seconds: float
    reserved_gpu_seconds: float
    allocated_billed_gpu_seconds: float
    required_archive_sha256s: frozenset[str]

    def __post_init__(self) -> None:
        _require_text(self.run_id, "finalization-readiness run ID")
        if self.node_count != 21:
            raise FormalExperimentFinalAuditError(
                "finalization readiness is not the exact 21-node DAG"
            )
        for label, value in (
            ("expected cells", self.expected_cell_count),
            ("latest attempts", self.latest_complete_attempt_count),
            ("retained retries", self.retained_retry_attempt_count),
            ("selections", self.selection_decision_count),
            ("metrics", self.metric_count),
            ("headline metrics", self.headline_metric_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FormalExperimentFinalAuditError(
                    f"finalization-readiness {label} count is invalid"
                )
        if (
            self.expected_cell_count < 1
            or self.latest_complete_attempt_count != self.expected_cell_count
            or self.selection_decision_count < 1
            or self.metric_count < 1
            or self.headline_metric_count < 1
        ):
            raise FormalExperimentFinalAuditError(
                "finalization readiness lacks required coverage or statistics"
            )
        for label, value in (
            ("expected cells", self.expected_cell_ids_sha256),
            ("coverage", self.coverage_sha256),
            ("accounting", self.accounting_sha256),
        ):
            _require_sha256(value, f"finalization-readiness {label}")
        for label, value in (
            ("compute GPU-seconds", self.compute_gpu_seconds),
            ("reserved GPU-seconds", self.reserved_gpu_seconds),
            ("allocated billed GPU-seconds", self.allocated_billed_gpu_seconds),
        ):
            _require_nonnegative_finite(
                value,
                f"finalization-readiness {label}",
            )
        if self.reserved_gpu_seconds < self.compute_gpu_seconds:
            raise FormalExperimentFinalAuditError(
                "finalization-readiness reserved time is below compute time"
            )
        if (
            type(self.required_archive_sha256s) is not frozenset
            or not self.required_archive_sha256s
        ):
            raise FormalExperimentFinalAuditError(
                "finalization readiness has no archive evidence"
            )
        for value in self.required_archive_sha256s:
            _require_sha256(value, "finalization-readiness archive evidence")


def _rebuild_final_stage_completion(path: str | Path) -> Any:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        rebuild_formal_single_operator_stage_completion,
    )

    return rebuild_formal_single_operator_stage_completion(path)


def _controller_binding(
    row: Mapping[str, Any], prefix: str, *, required: bool = True
) -> ControllerArtifactBinding | None:
    path = row.get(f"{prefix}_path")
    digest = row.get(f"{prefix}_sha256")
    if path is None and digest is None and not required:
        return None
    if type(path) is not str or type(digest) is not str:
        raise FormalExperimentFinalAuditError(
            f"controller {row.get('node')} lacks {prefix} binding"
        )
    try:
        binding = ControllerArtifactBinding(path, digest)
        reopened = ControllerArtifactBinding.bind(path)
    except (TypeError, ValueError, RuntimeError) as error:
        raise FormalExperimentFinalAuditError(
            f"controller {row.get('node')} {prefix} binding changed"
        ) from error
    if binding != reopened:
        raise FormalExperimentFinalAuditError(
            f"controller {row.get('node')} {prefix} binding changed"
        )
    return binding


def _audit_controller_chain(
    snapshot: Mapping[str, Any],
) -> tuple[tuple[str, ...], frozenset[str], str]:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FORMAL_SINGLE_OPERATOR_NODE_SPECS,
    )

    controllers = tuple(snapshot["controller_nodes"])
    stage_plan = tuple(snapshot["stage_plan"])
    expected_metadata = tuple(
        (spec.node, spec.ordinal, spec.stage, spec.phase)
        for spec in FORMAL_SINGLE_OPERATOR_NODE_SPECS
    )
    if len(controllers) != 21 or len(stage_plan) != 21:
        raise FormalExperimentFinalAuditError(
            "controller does not contain the exact 21-node DAG"
        )
    actual_metadata = tuple(
        (controller["node"], controller["ordinal"], stage["stage"], stage["phase"])
        for controller, stage in zip(controllers, stage_plan, strict=True)
    )
    if actual_metadata != expected_metadata:
        raise FormalExperimentFinalAuditError(
            "controller does not contain the exact 21-node DAG"
        )
    if any(row["state"] != "REDUCED" for row in controllers):
        raise FormalExperimentFinalAuditError(
            "DAG_REDUCED_AWAITING_FINAL_AUDIT requires every node REDUCED"
        )
    by_node = {str(row["node"]): row for row in controllers}
    final_binding = _controller_binding(controllers[-1], "completion")
    assert final_binding is not None
    try:
        rebuilt = _rebuild_final_stage_completion(final_binding.absolute_path)
    except BaseException as error:
        raise FormalExperimentFinalAuditError(
            "final completion chain did not deep-rebuild"
        ) from error

    seen_nodes: list[str] = []
    expected_cells: list[str] = []
    required_digests: set[str] = set()
    current_path = final_binding.absolute_path
    current = rebuilt
    while current is not None:
        artifact = current.artifact
        node = str(artifact.node)
        if node in seen_nodes or node not in by_node:
            raise FormalExperimentFinalAuditError(
                "rebuilt completion chain contains a duplicate or foreign node"
            )
        row = by_node[node]
        seen_nodes.append(node)
        completion = _controller_binding(row, "completion")
        materialization = _controller_binding(row, "materialization")
        node_materialization = _controller_binding(row, "node_materialization")
        execution = _controller_binding(row, "execution_source")
        prepared = _controller_binding(row, "prepared_launch", required=False)
        decision = _controller_binding(row, "decision")
        assert completion and materialization and node_materialization and execution
        assert decision
        if completion.absolute_path != current_path:
            raise FormalExperimentFinalAuditError(
                f"controller completion path for {node} differs from rebuilt chain"
            )
        if (
            artifact.node_materialization_source.absolute_path
            != node_materialization.absolute_path
            or artifact.node_materialization_source.raw_sha256
            != node_materialization.sha256
            or artifact.decision_source.absolute_path != decision.absolute_path
            or artifact.decision_source.raw_sha256 != decision.sha256
            or current.node_materialization.materialization_source.absolute_path
            != materialization.absolute_path
            or current.node_materialization.materialization_source.raw_sha256
            != materialization.sha256
            or current.node_materialization.sha256
            != artifact.node_materialization_sha256
            or current.decision.sha256 != artifact.decision_sha256
            or current.materialization.sha256 != artifact.materialization_sha256
        ):
            raise FormalExperimentFinalAuditError(
                f"controller artifacts for {node} differ from deep rebuild"
            )
        cell_ids = tuple(cell.cell_id for cell in current.materialization.cells)
        if cell_ids != tuple(sorted(set(cell_ids))):
            raise FormalExperimentFinalAuditError(
                f"materialization cell IDs for {node} are not canonical"
            )
        expected_digest = _semantic_sha256(cell_ids)
        if (
            int(row["expected_cell_count"]) != len(cell_ids)
            or row["expected_cell_ids_sha256"] != expected_digest
        ):
            raise FormalExperimentFinalAuditError(
                f"controller expected cells for {node} differ from materialization"
            )
        actual_ids = tuple(value.cell_id for value in artifact.actual_results)
        if actual_ids != cell_ids:
            raise FormalExperimentFinalAuditError(
                f"completion actual-result coverage for {node} differs"
            )
        expected_cells.extend(cell_ids)
        required_digests.update(
            binding.sha256
            for binding in (
                completion,
                materialization,
                node_materialization,
                execution,
                decision,
                prepared,
            )
            if binding is not None
        )
        current_path = (
            ""
            if artifact.predecessor_source is None
            else artifact.predecessor_source.absolute_path
        )
        current = current.predecessor

    expected_reverse = tuple(reversed([row[0] for row in expected_metadata]))
    if tuple(seen_nodes) != expected_reverse:
        raise FormalExperimentFinalAuditError(
            "rebuilt completion chain is not the exact fixed DAG"
        )
    ordered_cells = tuple(sorted(expected_cells))
    if len(set(ordered_cells)) != len(ordered_cells):
        raise FormalExperimentFinalAuditError(
            "materialized cell IDs overlap across controller nodes"
        )
    coverage_sha256 = _semantic_sha256(
        {
            "nodes": [
                {
                    "node": row["node"],
                    "expected_cell_count": row["expected_cell_count"],
                    "expected_cell_ids_sha256": row["expected_cell_ids_sha256"],
                    "completion_sha256": row["completion_sha256"],
                }
                for row in controllers
            ],
            "expected_cell_ids": ordered_cells,
        }
    )
    return ordered_cells, frozenset(required_digests), coverage_sha256


def _finite_accounting(
    row: Mapping[str, Any], label: str
) -> tuple[float, float, float]:
    compute = _require_nonnegative_finite(
        row.get("compute_gpu_seconds"), f"{label} compute GPU-seconds"
    )
    reserved = _require_nonnegative_finite(
        row.get("reserved_gpu_seconds"), f"{label} reserved GPU-seconds"
    )
    billed = _require_nonnegative_finite(
        row.get("billed_gpu_seconds"), f"{label} billed GPU-seconds"
    )
    if reserved < compute:
        raise FormalExperimentFinalAuditError(
            f"{label} reserved GPU time is below compute GPU time"
        )
    return compute, reserved, billed


def _audit_ledger(
    snapshot: Mapping[str, Any],
    *,
    expected_cell_ids: tuple[str, ...],
    controller_archive_digests: frozenset[str],
    coverage_sha256: str,
) -> _CoverageAudit:
    attempts = tuple(snapshot["attempts"])
    fresh = tuple(row for row in attempts if not bool(row["is_legacy_import"]))
    latest: dict[str, Mapping[str, Any]] = {}
    for row in fresh:
        cell_id = str(row["cell_id"])
        prior = latest.get(cell_id)
        if prior is None or int(row["attempt"]) > int(prior["attempt"]):
            latest[cell_id] = row
    expected_set = set(expected_cell_ids)
    missing = tuple(sorted(expected_set - set(latest)))
    extra = tuple(sorted(set(latest) - expected_set))
    if missing or extra:
        raise FormalExperimentFinalAuditError(
            f"ledger/materialization coverage differs: missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    nonterminal = tuple(
        sorted(
            (str(row["cell_id"]), int(row["attempt"]), str(row["status"]))
            for row in fresh
            if row["status"] in {"PENDING", "RUNNING", "BLOCKED"}
        )
    )
    if nonterminal:
        raise FormalExperimentFinalAuditError(
            "nonlegacy ledger retains active or blocked attempts"
        )
    bad_latest = tuple(
        sorted(
            (cell_id, int(row["attempt"]), str(row["status"]))
            for cell_id, row in latest.items()
            if row["status"] != "COMPLETE"
        )
    )
    if bad_latest:
        raise FormalExperimentFinalAuditError(
            "latest nonlegacy attempts are not exact COMPLETE coverage"
        )
    retry_rows = tuple(
        row
        for row in fresh
        if int(row["attempt"]) < int(latest[str(row["cell_id"])]["attempt"])
    )
    for row in retry_rows:
        if row["status"] not in {"FAILED", "STALE_IDENTITY"} or not all(
            row.get(field) for field in ("exclusion_reason", "retry_decision")
        ):
            raise FormalExperimentFinalAuditError(
                "retained retry attempt is not terminal and explained"
            )
    evidence_digests = set(controller_archive_digests)
    for cell_id, row in latest.items():
        if (
            row.get("exit_code") != 0
            or row.get("started_at_ns") is None
            or row.get("finished_at_ns") is None
            or int(row["finished_at_ns"]) <= int(row["started_at_ns"])
            or not row.get("assigned_gpu_uuids")
        ):
            raise FormalExperimentFinalAuditError(
                f"COMPLETE attempt {cell_id} lacks valid physical lifecycle"
            )
        for field in ("terminal_sha256", "junit_sha256", "raw_log_sha256"):
            evidence_digests.add(_require_sha256(row.get(field), f"{cell_id} {field}"))
        evidence_files = row.get("evidence_files")
        if type(evidence_files) is not dict or not evidence_files:
            raise FormalExperimentFinalAuditError(
                f"COMPLETE attempt {cell_id} lacks raw evidence bindings"
            )
        for path, digest in evidence_files.items():
            _require_text(path, f"{cell_id} evidence path")
            evidence_digests.add(_require_sha256(digest, f"{cell_id} evidence"))
    for row in retry_rows:
        for field in ("terminal_sha256", "junit_sha256", "raw_log_sha256"):
            if row.get(field) is not None:
                evidence_digests.add(
                    _require_sha256(row[field], f"retained retry {field}")
                )
        for digest in dict(row.get("evidence_files") or {}).values():
            evidence_digests.add(_require_sha256(digest, "retained retry evidence"))

    by_key = {(str(row["cell_id"]), int(row["attempt"])): row for row in fresh}
    for group in tuple(snapshot["physical_attempt_groups"]):
        members = tuple(group["members"])
        if group["status"] != "COMPLETE" or len(members) != 10:
            raise FormalExperimentFinalAuditError(
                "physical attempt group is not exact-ten COMPLETE"
            )
        rows = tuple(
            by_key[(str(member["cell_id"]), int(member["attempt"]))]
            for member in members
        )
        leader_key = (str(group["leader_cell_id"]), int(group["leader_attempt"]))
        if leader_key not in by_key:
            raise FormalExperimentFinalAuditError(
                "physical accounting leader is absent from the ledger"
            )
        leader = by_key[leader_key]
        for field in (
            "compute_gpu_seconds",
            "reserved_gpu_seconds",
            "billed_gpu_seconds",
        ):
            total = sum(float(row[field]) for row in rows)
            if not math.isclose(
                total,
                float(leader[field]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise FormalExperimentFinalAuditError(
                    "physical attempt group GPU time is double-accounted"
                )
        if any(
            row is not leader
            and any(
                float(row[field]) != 0.0
                for field in (
                    "compute_gpu_seconds",
                    "reserved_gpu_seconds",
                    "billed_gpu_seconds",
                )
            )
            for row in rows
        ):
            raise FormalExperimentFinalAuditError(
                "physical attempt group charges a nonleader member"
            )

    auxiliary_by_node: dict[str, list[Mapping[str, Any]]] = {}
    for group in tuple(snapshot["controller_auxiliary_groups"]):
        auxiliary_by_node.setdefault(str(group["node"]), []).append(group)
        if group["status"] in {"PENDING", "RUNNING"}:
            raise FormalExperimentFinalAuditError(
                "controller auxiliary group is still active"
            )
    for groups in auxiliary_by_node.values():
        latest_group = max(groups, key=lambda row: int(row["attempt"]))
        if (
            latest_group["status"] != "COMPLETE"
            or latest_group["adopted_at_ns"] is None
        ):
            raise FormalExperimentFinalAuditError(
                "latest controller auxiliary group is not COMPLETE and adopted"
            )
        adopted_rows = []
        for job in tuple(latest_group["jobs"]):
            key = (job.get("adopted_cell_id"), job.get("adopted_cell_attempt"))
            if job["status"] != "COMPLETE" or None in key or key not in by_key:
                raise FormalExperimentFinalAuditError(
                    "controller auxiliary job is not durably adopted"
                )
            adopted_rows.append(by_key[key])
        for field in (
            "compute_gpu_seconds",
            "reserved_gpu_seconds",
            "billed_gpu_seconds",
        ):
            if not math.isclose(
                sum(float(row[field]) for row in adopted_rows),
                float(latest_group[field]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise FormalExperimentFinalAuditError(
                    "controller auxiliary GPU time is double-accounted"
                )

    accounting_rows = []
    compute = reserved = allocated_billed = 0.0
    for row in fresh:
        row_compute, row_reserved, row_billed = _finite_accounting(
            row, f"attempt {row['cell_id']}/{row['attempt']}"
        )
        compute += row_compute
        reserved += row_reserved
        allocated_billed += row_billed
        accounting_rows.append(
            {
                "cell_id": row["cell_id"],
                "attempt": row["attempt"],
                "status": row["status"],
                "compute_gpu_seconds": row_compute,
                "reserved_gpu_seconds": row_reserved,
                "billed_gpu_seconds": row_billed,
            }
        )
    for groups in auxiliary_by_node.values():
        for group in groups:
            if group["adopted_at_ns"] is not None:
                continue
            if group["status"] != "FAILED" or not all(
                group.get(field) for field in ("failure_code", "exclusion_reason")
            ):
                raise FormalExperimentFinalAuditError(
                    "unadopted auxiliary accounting is not a retained failed attempt"
                )
            group_compute, group_reserved, group_billed = _finite_accounting(
                group,
                f"auxiliary group {group['group_id']}/{group['attempt']}",
            )
            compute += group_compute
            reserved += group_reserved
            allocated_billed += group_billed
            accounting_rows.append(
                {
                    "auxiliary_group_id": group["group_id"],
                    "attempt": group["attempt"],
                    "status": group["status"],
                    "compute_gpu_seconds": group_compute,
                    "reserved_gpu_seconds": group_reserved,
                    "billed_gpu_seconds": group_billed,
                }
            )
            for job in tuple(group["jobs"]):
                for field in ("terminal_sha256", "junit_sha256", "raw_log_sha256"):
                    if job.get(field) is not None:
                        evidence_digests.add(
                            _require_sha256(job[field], f"retained auxiliary {field}")
                        )
                for digest in dict(job.get("evidence_files") or {}).values():
                    evidence_digests.add(
                        _require_sha256(digest, "retained auxiliary evidence")
                    )
    accounting_sha256 = _semantic_sha256(
        {
            "rows": accounting_rows,
            "compute_gpu_seconds": compute,
            "reserved_gpu_seconds": reserved,
            "allocated_billed_gpu_seconds": allocated_billed,
        }
    )
    return _CoverageAudit(
        expected_cell_ids=expected_cell_ids,
        latest_attempts=latest,
        controller_stage_phases=frozenset(
            (str(row["stage"]), str(row["phase"])) for row in snapshot["stage_plan"]
        ),
        nonempty_stage_phases=frozenset(
            (str(row["stage"]), str(row["phase"])) for row in latest.values()
        ),
        retained_retry_attempt_count=len(retry_rows),
        required_archive_sha256s=frozenset(evidence_digests),
        coverage_sha256=coverage_sha256,
        compute_gpu_seconds=compute,
        reserved_gpu_seconds=reserved,
        allocated_billed_gpu_seconds=allocated_billed,
        accounting_sha256=accounting_sha256,
    )


def _read_database_projection(store: ExperimentOperatorStore) -> dict[str, Any]:
    database = _absolute_path(store.path.resolve(), "operator database")
    uri = f"file:{quote(str(database))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise FormalExperimentFinalAuditError(
                "operator database quick_check failed"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise FormalExperimentFinalAuditError(
                "operator database foreign-key check failed"
            )
        connection.execute("BEGIN")
        selections = tuple(
            {
                "decision_id": row["decision_id"],
                "occurred_at_ns": int(row["occurred_at_ns"]),
                "stage": row["stage"],
                "phase": row["phase"],
                "decision_kind": row["decision_kind"],
                "source_sha256": row["source_sha256"],
                "decision": json.loads(row["decision_json"]),
            }
            for row in connection.execute(
                "SELECT * FROM selection_decisions ORDER BY occurred_at_ns, decision_id"
            ).fetchall()
        )
        metrics = tuple(
            {
                "stage": row["stage"],
                "phase": row["phase"],
                "cell_id": row["cell_id"],
                "attempt": int(row["attempt"]),
                "metric_name": row["metric_name"],
                "metric_kind": row["metric_kind"],
                "point_estimate": float(row["point_estimate"]),
                "ci_low": None if row["ci_low"] is None else float(row["ci_low"]),
                "ci_high": (None if row["ci_high"] is None else float(row["ci_high"])),
                "independent_block_count": row["independent_block_count"],
                "request_count": row["request_count"],
                "paired": None if row["paired"] is None else bool(row["paired"]),
                "reducer_method": row["reducer_method"],
                "attributes_json": row["attributes_json"],
                "recorded_at_ns": int(row["recorded_at_ns"]),
            }
            for row in connection.execute(
                "SELECT * FROM metrics_long ORDER BY stage, phase, cell_id, "
                "attempt, metric_name, attributes_json"
            ).fetchall()
        )
        events = tuple(
            {
                "event_id": int(row["event_id"]),
                "occurred_at_ns": int(row["occurred_at_ns"]),
                "event_type": row["event_type"],
                "severity": row["severity"],
                "cell_id": row["cell_id"],
                "attempt": row["attempt"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in connection.execute(
                "SELECT * FROM watchdog_events ORDER BY event_id"
            ).fetchall()
        )
        provider_samples = tuple(
            dict(row)
            for row in connection.execute(
                "SELECT * FROM provider_runtime_samples "
                "ORDER BY instance_uuid, provider_started_at_ns, observed_at_ns"
            ).fetchall()
        )
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {
        "selections": selections,
        "metrics": metrics,
        "events": events,
        "provider_samples": provider_samples,
    }


def _validate_statistics(
    projection: Mapping[str, Any], coverage: _CoverageAudit
) -> int:
    selections = tuple(projection["selections"])
    metrics = tuple(projection["metrics"])
    if not selections:
        raise FormalExperimentFinalAuditError("selection decision projection is empty")
    if not metrics:
        raise FormalExperimentFinalAuditError("metric projection is empty")
    for row in selections:
        _require_text(row["decision_id"], "selection decision ID")
        _require_text(row["decision_kind"], "selection decision kind")
        _require_sha256(row["source_sha256"], "selection decision source")
        if type(row["decision"]) is not dict or not row["decision"]:
            raise FormalExperimentFinalAuditError("selection decision payload is empty")
    selection_stage_phases = {
        (str(row["stage"]), str(row["phase"])) for row in selections
    }
    if selection_stage_phases != coverage.controller_stage_phases:
        raise FormalExperimentFinalAuditError(
            "selection decisions do not cover every applicable controller node"
        )
    headline = 0
    for row in metrics:
        cell_id = str(row["cell_id"])
        attempt = int(row["attempt"])
        latest = coverage.latest_attempts.get(cell_id)
        if latest is None or int(latest["attempt"]) != attempt:
            raise FormalExperimentFinalAuditError(
                "metric references a missing or superseded attempt"
            )
        point = row["point_estimate"]
        if not math.isfinite(float(point)):
            raise FormalExperimentFinalAuditError("metric point estimate is non-finite")
        _require_text(row["metric_name"], "metric name")
        _require_text(row["reducer_method"], "metric reducer")
        try:
            attributes = json.loads(row["attributes_json"])
        except json.JSONDecodeError as error:
            raise FormalExperimentFinalAuditError(
                "metric attributes are not JSON"
            ) from error
        if type(attributes) is not dict:
            raise FormalExperimentFinalAuditError(
                "metric attributes are not one object"
            )
        axes = latest["scientific_axes"]
        task = axes.get("task") if type(axes) is dict else None
        if row["metric_kind"] == "headline":
            headline += 1
            if task in _NON_HEADLINE_TASKS:
                raise FormalExperimentFinalAuditError(
                    f"non-headline task {task!r} carries a headline metric"
                )
            ci_low = row["ci_low"]
            ci_high = row["ci_high"]
            blocks = row["independent_block_count"]
            requests = row["request_count"]
            if (
                ci_low is None
                or ci_high is None
                or not math.isfinite(float(ci_low))
                or not math.isfinite(float(ci_high))
                or float(ci_low) > float(ci_high)
                or isinstance(blocks, bool)
                or not isinstance(blocks, int)
                or blocks < 1
                or isinstance(requests, bool)
                or not isinstance(requests, int)
                or requests < 1
                or type(row["paired"]) is not bool
                or attributes.get("confidence_level") != 0.95
            ):
                raise FormalExperimentFinalAuditError(
                    "headline metric lacks a valid 95% interval/count/reducer identity"
                )
        elif row["metric_kind"] == "descriptive":
            if row["ci_low"] is not None or row["ci_high"] is not None:
                raise FormalExperimentFinalAuditError(
                    "descriptive metric carries a fabricated confidence interval"
                )
        else:
            raise FormalExperimentFinalAuditError("metric kind is unregistered")
    if headline < 1:
        raise FormalExperimentFinalAuditError(
            "quantitative headline metric projection is empty"
        )
    metric_stage_phases = {(str(row["stage"]), str(row["phase"])) for row in metrics}
    if metric_stage_phases != coverage.nonempty_stage_phases:
        raise FormalExperimentFinalAuditError(
            "metrics do not cover every applicable controller node"
        )
    return headline


def audit_finalization_readiness(
    store: ExperimentOperatorStore,
) -> FormalExperimentFinalizationReadiness:
    """Deep-audit the immutable scientific state before final archiving.

    The operator database is sampled before and after the replay.  A caller
    therefore cannot use this result while a controller, reducer, scheduler,
    or evidence writer is mutating durable scientific state.
    """

    if type(store) is not ExperimentOperatorStore:
        raise TypeError("finalization readiness requires an exact operator store")
    before = store.snapshot()
    expected_cells, controller_digests, coverage_sha = _audit_controller_chain(before)
    coverage = _audit_ledger(
        before,
        expected_cell_ids=expected_cells,
        controller_archive_digests=controller_digests,
        coverage_sha256=coverage_sha,
    )
    projection = _read_database_projection(store)
    headline_count = _validate_statistics(projection, coverage)
    after = store.snapshot()
    if before != after:
        raise FormalExperimentFinalAuditError(
            "operator state changed during finalization readiness audit"
        )
    return FormalExperimentFinalizationReadiness(
        run_id=store.run_id,
        node_count=len(before["controller_nodes"]),
        expected_cell_count=len(expected_cells),
        latest_complete_attempt_count=len(coverage.latest_attempts),
        retained_retry_attempt_count=coverage.retained_retry_attempt_count,
        selection_decision_count=len(projection["selections"]),
        metric_count=len(projection["metrics"]),
        headline_metric_count=headline_count,
        expected_cell_ids_sha256=_semantic_sha256(expected_cells),
        coverage_sha256=coverage.coverage_sha256,
        accounting_sha256=coverage.accounting_sha256,
        compute_gpu_seconds=coverage.compute_gpu_seconds,
        reserved_gpu_seconds=coverage.reserved_gpu_seconds,
        allocated_billed_gpu_seconds=coverage.allocated_billed_gpu_seconds,
        required_archive_sha256s=coverage.required_archive_sha256s,
    )


def _parse_csv(path: Path, label: str) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise FormalExperimentFinalAuditError(f"{label} is not valid CSV") from error


def _json_lines_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_file(dict(row)) for row in rows)


def _audit_progress_export(
    *,
    root: str | Path,
    store: ExperimentOperatorStore,
    snapshot: Mapping[str, Any],
    projection: Mapping[str, Any],
    coverage: _CoverageAudit,
) -> _ProgressAudit:
    export_root = _absolute_path(root, "progress export root")
    if export_root.is_symlink() or not export_root.is_dir():
        raise FormalExperimentFinalAuditError(
            "progress export root is not a safe directory"
        )
    manifest_path = export_root / "export_manifest.json"
    manifest = _load_canonical_object(manifest_path, "progress export manifest")
    if set(manifest) != {"run_id", "exported_at_ns", "files"}:
        raise FormalExperimentFinalAuditError("progress export manifest fields differ")
    if manifest["run_id"] != store.run_id:
        raise FormalExperimentFinalAuditError("progress export belongs to another run")
    files = manifest["files"]
    if type(files) is not dict or set(files) != _EXPECTED_EXPORT_FILES:
        raise FormalExperimentFinalAuditError("progress export file set differs")
    file_bindings = []
    for name in sorted(files):
        expected = _require_sha256(files[name], f"progress export {name}")
        path = export_root / name
        actual, _size = _stable_file_sha256(path, f"progress export {name}")
        if actual != expected:
            raise FormalExperimentFinalAuditError(
                f"progress export {name} digest differs"
            )
        file_bindings.append((name, actual))

    selections_path = export_root / "selection_decisions.jsonl"
    if selections_path.read_bytes() != _json_lines_bytes(projection["selections"]):
        raise FormalExperimentFinalAuditError(
            "selection decision export is stale or incomplete"
        )
    events_path = export_root / "watchdog_events.jsonl"
    if events_path.read_bytes() != _json_lines_bytes(projection["events"]):
        raise FormalExperimentFinalAuditError(
            "watchdog event export is stale or incomplete"
        )

    try:
        import pyarrow.parquet as pq

        metric_rows = tuple(
            dict(row)
            for row in pq.read_table(export_root / "metrics_long.parquet").to_pylist()
        )
    except BaseException as error:
        raise FormalExperimentFinalAuditError(
            "metrics Parquet could not be read"
        ) from error
    if metric_rows != tuple(projection["metrics"]):
        raise FormalExperimentFinalAuditError("metrics Parquet is stale or incomplete")

    ledger_rows = _parse_csv(export_root / "cell_ledger.csv", "cell ledger")
    attempts = tuple(snapshot["attempts"])
    if len(ledger_rows) != len(attempts):
        raise FormalExperimentFinalAuditError(
            "cell ledger export row count differs from SQLite"
        )
    actual_by_key = {
        (str(row["cell_id"]), int(row["attempt"])): row for row in attempts
    }
    for row in ledger_rows:
        try:
            key = (row["cell_id"], int(row["attempt"]))
            actual = actual_by_key[key]
        except (KeyError, ValueError) as error:
            raise FormalExperimentFinalAuditError(
                "cell ledger export contains a foreign attempt"
            ) from error
        if (
            row["stage"] != actual["stage"]
            or row["phase"] != actual["phase"]
            or row["status"] != actual["status"]
            or row["command_sha256"] != actual["command_sha256"]
            or row["terminal_sha256"] != (actual["terminal_sha256"] or "")
            or row["junit_sha256"] != (actual["junit_sha256"] or "")
            or row["raw_log_sha256"] != (actual["raw_log_sha256"] or "")
            or json.loads(row["scientific_axes"]) != actual["scientific_axes"]
            or json.loads(row["identity"]) != actual["identity"]
            or json.loads(row["evidence_files"]) != actual["evidence_files"]
            or row["included_in_analysis"] != str(bool(actual["included_in_analysis"]))
            or not math.isclose(
                float(row["compute_gpu_seconds"]),
                float(actual["compute_gpu_seconds"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row["reserved_gpu_seconds"]),
                float(actual["reserved_gpu_seconds"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise FormalExperimentFinalAuditError(
                "cell ledger export differs from SQLite"
            )
    if set(actual_by_key) != {
        (row["cell_id"], int(row["attempt"])) for row in ledger_rows
    }:
        raise FormalExperimentFinalAuditError(
            "cell ledger export coverage differs from SQLite"
        )

    controller_rows = _parse_csv(
        export_root / "controller_state.csv", "controller state"
    )
    controllers = tuple(snapshot["controller_nodes"])
    if len(controller_rows) != len(controllers):
        raise FormalExperimentFinalAuditError("controller export row count differs")
    for exported, current in zip(controller_rows, controllers, strict=True):
        if any(
            exported[field] != str(current[field] if current[field] is not None else "")
            for field in (
                "node",
                "ordinal",
                "state",
                "materialization_sha256",
                "node_materialization_sha256",
                "execution_source_sha256",
                "decision_sha256",
                "completion_sha256",
                "expected_cell_count",
                "expected_cell_ids_sha256",
            )
        ):
            raise FormalExperimentFinalAuditError(
                "controller export differs from SQLite"
            )

    stage_rows = _parse_csv(export_root / "stage_plan.csv", "stage plan")
    stage_summary = tuple(snapshot["stage_plan"])
    if len(stage_rows) != 21 or len(stage_summary) != 21:
        raise FormalExperimentFinalAuditError(
            "stage plan export is not the exact 21-node plan"
        )
    for exported, current in zip(stage_rows, stage_summary, strict=True):
        if (
            exported["node"] != current["node"]
            or exported["completed"] != str(current["completed"])
            or exported["materialized_cells"] != str(current["materialized_cells"])
            or exported["running"] != "0"
            or exported["failed"] != "0"
            or exported["blocked"] != "0"
        ):
            raise FormalExperimentFinalAuditError(
                "stage plan export differs from current exact coverage"
            )
    summary_rows = _parse_csv(export_root / "stage_summary.csv", "stage summary")
    if len(summary_rows) != 21:
        raise FormalExperimentFinalAuditError("stage summary export is incomplete")
    billing_rows = _parse_csv(export_root / "instance_billing.csv", "billing")
    if len(billing_rows) != len(snapshot["provider_billing_intervals"]):
        raise FormalExperimentFinalAuditError(
            "provider billing export is stale or incomplete"
        )
    dashboard = (export_root / "dashboard.md").read_text(encoding="utf-8")
    if f"`{store.run_id}`" not in dashboard:
        raise FormalExperimentFinalAuditError(
            "progress dashboard belongs to another run"
        )

    headline = _validate_statistics(projection, coverage)
    return _ProgressAudit(
        manifest=FinalAuditArtifactBinding.bind(
            manifest_path, label="progress export manifest"
        ),
        files=tuple(file_bindings),
        selections=tuple(projection["selections"]),
        metrics=tuple(projection["metrics"]),
        headline_metric_count=headline,
    )


def _safe_manifest_member(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
        raise FormalExperimentFinalAuditError("archive manifest path escapes its root")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise FormalExperimentFinalAuditError(
            "archive manifest member is not one regular file"
        )
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise FormalExperimentFinalAuditError(
            "archive manifest member resolves outside its root"
        )
    return path


def _replay_archive_content_tree(
    local_root: str | Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    root = _absolute_path(local_root, "final archive local root")
    if root.is_symlink() or not root.is_dir():
        raise FormalExperimentFinalAuditError(
            "final archive local root is not one safe directory"
        )
    manifest_path = root / "sha256_manifest.json"
    manifest = _load_canonical_object(manifest_path, "final archive manifest")
    actual_manifest_sha, _size = _stable_file_sha256(
        manifest_path, "final archive manifest"
    )
    if actual_manifest_sha != expected_manifest_sha256:
        raise FormalExperimentFinalAuditError("final archive manifest digest differs")
    if (
        set(manifest) != {"schema_version", "kind", "files"}
        or manifest["schema_version"] != 1
        or manifest["kind"] != "formal_archive_sha256_manifest"
        or type(manifest["files"]) is not list
        or not manifest["files"]
    ):
        raise FormalExperimentFinalAuditError("final archive manifest identity differs")
    rows = []
    prior = None
    for raw in manifest["files"]:
        if type(raw) is not dict or set(raw) != {"path", "sha256", "size_bytes"}:
            raise FormalExperimentFinalAuditError(
                "final archive manifest row fields differ"
            )
        relative = _require_text(raw["path"], "archive relative path")
        if prior is not None and relative <= prior:
            raise FormalExperimentFinalAuditError(
                "archive manifest paths are not unique and sorted"
            )
        prior = relative
        digest = _require_sha256(raw["sha256"], "archive payload")
        size_bytes = raw["size_bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise FormalExperimentFinalAuditError("archive payload size is invalid")
        member = _safe_manifest_member(root, relative)
        actual_sha, actual_size = _stable_file_sha256(member, "archive payload")
        if actual_sha != digest or actual_size != size_bytes:
            raise FormalExperimentFinalAuditError("archive payload identity differs")
        rows.append(dict(raw))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    if actual_files != {row["path"] for row in rows}:
        raise FormalExperimentFinalAuditError(
            "final archive has unregistered or missing files"
        )
    content_tree = _file_semantic_sha256(
        {"manifest_sha256": expected_manifest_sha256, "files": rows}
    )
    return content_tree, tuple(rows)


def _audit_final_archive(
    snapshot: Mapping[str, Any],
    *,
    final_archive_id: str,
    required_sha256s: frozenset[str],
) -> _ArchiveAudit:
    matches = tuple(
        row for row in snapshot["archives"] if row["archive_id"] == final_archive_id
    )
    if len(matches) != 1:
        raise FormalExperimentFinalAuditError(
            "final archive ID does not resolve to one checkpoint"
        )
    row = matches[0]
    if (
        row["state"] != "EVICTION_AUTHORIZED"
        or row["safe_boundary"] != FINAL_ARCHIVE_SAFE_BOUNDARY
        or row["cell_id"] is not None
        or row["attempt"] is not None
    ):
        raise FormalExperimentFinalAuditError(
            "final archive lacks the archive-safe whole-run rehydrated receipt"
        )
    receipts = tuple(
        row.get(name)
        for name in ("transfer_receipt", "local_sha_receipt", "rehydrate_receipt")
    )
    if any(type(receipt) is not dict for receipt in receipts):
        raise FormalExperimentFinalAuditError(
            "final archive lacks transfer/local-SHA/rehydrate receipts"
        )
    transfer, local_sha, rehydrate = receipts
    assert isinstance(transfer, dict)
    assert isinstance(local_sha, dict)
    assert isinstance(rehydrate, dict)
    manifest_sha = _require_sha256(
        row["remote_manifest_sha256"], "final archive manifest"
    )
    if (
        transfer.get("step") != "TRANSFER"
        or local_sha.get("step") != "LOCAL_SHA_VERIFY"
        or rehydrate.get("step") != "REHYDRATE_VERIFY"
        or any(receipt.get("manifest_sha256") != manifest_sha for receipt in receipts)
    ):
        raise FormalExperimentFinalAuditError(
            "final archive verification sequence differs"
        )
    content_tree, rows = _replay_archive_content_tree(
        row["local_final_root"],
        expected_manifest_sha256=manifest_sha,
    )
    checked_count = len(rows)
    checked_bytes = sum(int(item["size_bytes"]) for item in rows)
    if (
        any(
            receipt.get("checked_file_count") != checked_count
            or receipt.get("checked_bytes") != checked_bytes
            for receipt in receipts
        )
        or rehydrate.get("content_tree_sha256") != content_tree
        or int(row["predicted_payload_bytes"]) != checked_bytes
    ):
        raise FormalExperimentFinalAuditError(
            "final archive verification did not cover the complete content tree"
        )
    archived_digests = {str(item["sha256"]) for item in rows}
    missing = required_sha256s - archived_digests
    if missing:
        raise FormalExperimentFinalAuditError(
            f"final archive omits {len(missing)} bound evidence digest(s)"
        )
    return _ArchiveAudit(
        archive_id=final_archive_id,
        manifest_sha256=manifest_sha,
        content_tree_sha256=content_tree,
        local_root=str(_absolute_path(row["local_final_root"], "archive root")),
    )


def _audit_shutdown_probe(
    *,
    path: str | Path,
    store: ExperimentOperatorStore,
    snapshot: Mapping[str, Any],
    instance_uuid: str,
    audited_at_ns: int,
) -> FinalAuditArtifactBinding:
    value = _load_canonical_object(path, "shutdown probe")
    try:
        probe = AutoDlPowerOffSafetyProbe(**value)
    except (TypeError, ValueError, AutoDlProviderRuntimeError) as error:
        raise FormalExperimentFinalAuditError("shutdown probe is unsafe") from error
    if (
        probe.instance_uuid != instance_uuid
        or probe.run_id != store.run_id
        or probe.observed_at_ns > audited_at_ns
        or audited_at_ns - probe.observed_at_ns > 300 * 1_000_000_000
        or snapshot["dispatch_state"] != "STOP"
        or any(row["status"] == "RUNNING" for row in snapshot["attempts"])
        or any(
            row["status"] == "RUNNING"
            for row in snapshot["controller_auxiliary_groups"]
        )
    ):
        raise FormalExperimentFinalAuditError(
            "shutdown probe lineage or live operator state is unsafe"
        )
    return FinalAuditArtifactBinding.bind(path, label="shutdown probe")


def _audit_pre_shutdown_provider(
    snapshot: Mapping[str, Any], *, instance_uuid: str, audited_at_ns: int
) -> tuple[float, float]:
    intervals = tuple(snapshot["provider_billing_intervals"])
    if not intervals or any(row["instance_uuid"] != instance_uuid for row in intervals):
        raise FormalExperimentFinalAuditError(
            "pre-shutdown provider billing lineage differs"
        )
    open_rows = tuple(row for row in intervals if not bool(row["complete"]))
    if len(open_rows) != 1 or open_rows[0] is not intervals[-1]:
        raise FormalExperimentFinalAuditError(
            "pre-shutdown audit requires exactly one current open provider interval"
        )
    earliest = min(int(row["provider_started_at_ns"]) for row in intervals)
    if audited_at_ns < earliest:
        raise FormalExperimentFinalAuditError(
            "pre-shutdown audit time precedes provider start"
        )
    billed = sum(float(row["whole_instance_billed_gpu_seconds"]) for row in intervals)
    return billed, (audited_at_ns - earliest) / 1e9


def publish_pre_shutdown_audit(
    *,
    store: ExperimentOperatorStore,
    instance_uuid: str,
    progress_export_root: str | Path,
    final_archive_id: str,
    shutdown_probe_path: str | Path,
    output_path: str | Path,
    audited_at_ns: int | None = None,
) -> FormalExperimentPreShutdownAuditReceipt:
    """Publish the exact safe boundary that authorizes a provider power-off.

    This function is intentionally read-only with respect to SQLite.  It may
    publish one new no-replace receipt, but it cannot dispatch work, call the
    provider, or mutate accounting.
    """

    if type(store) is not ExperimentOperatorStore:
        raise TypeError("pre-shutdown audit requires an exact operator store")
    if type(instance_uuid) is not str or not instance_uuid.startswith("pro-"):
        raise FormalExperimentFinalAuditError("instance UUID is malformed")
    destination = _absolute_path(output_path, "pre-shutdown audit output")
    existing = None
    if destination.exists() or destination.is_symlink():
        existing = load_pre_shutdown_audit(destination)
        if existing.run_id != store.run_id or existing.instance_uuid != instance_uuid:
            raise FormalExperimentFinalAuditError(
                "existing pre-shutdown audit has foreign lineage"
            )
    now = int(
        existing.audited_at_ns
        if audited_at_ns is None and existing is not None
        else time.time_ns()
        if audited_at_ns is None
        else audited_at_ns
    )
    if now < 1:
        raise FormalExperimentFinalAuditError("audit time is invalid")
    before = store.snapshot()
    expected_cells, controller_digests, coverage_sha = _audit_controller_chain(before)
    coverage = _audit_ledger(
        before,
        expected_cell_ids=expected_cells,
        controller_archive_digests=controller_digests,
        coverage_sha256=coverage_sha,
    )
    projection = _read_database_projection(store)
    progress = _audit_progress_export(
        root=progress_export_root,
        store=store,
        snapshot=before,
        projection=projection,
        coverage=coverage,
    )
    archive = _audit_final_archive(
        before,
        final_archive_id=final_archive_id,
        required_sha256s=coverage.required_archive_sha256s,
    )
    probe = _audit_shutdown_probe(
        path=shutdown_probe_path,
        store=store,
        snapshot=before,
        instance_uuid=instance_uuid,
        audited_at_ns=now,
    )
    billed_observed, wall_observed = _audit_pre_shutdown_provider(
        before,
        instance_uuid=instance_uuid,
        audited_at_ns=now,
    )
    after = store.snapshot()
    if before != after:
        raise FormalExperimentFinalAuditError(
            "operator state changed during final pre-shutdown audit"
        )
    receipt = FormalExperimentPreShutdownAuditReceipt(
        schema_version=1,
        kind="formal_experiment_pre_shutdown_audit",
        protocol_sha256=FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256,
        run_id=store.run_id,
        instance_uuid=instance_uuid,
        trust=TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
        formal_measured=False,
        controller_state="DAG_REDUCED_AWAITING_FINAL_AUDIT",
        audited_at_ns=now,
        node_count=21,
        expected_cell_count=len(expected_cells),
        expected_cell_ids_sha256=_semantic_sha256(expected_cells),
        latest_complete_attempt_count=len(coverage.latest_attempts),
        retained_retry_attempt_count=coverage.retained_retry_attempt_count,
        selection_decision_count=len(progress.selections),
        metric_count=len(progress.metrics),
        headline_metric_count=progress.headline_metric_count,
        coverage_sha256=coverage.coverage_sha256,
        selection_sha256=dict(progress.files)["selection_decisions.jsonl"],
        metrics_sha256=dict(progress.files)["metrics_long.parquet"],
        accounting_sha256=coverage.accounting_sha256,
        compute_gpu_seconds=coverage.compute_gpu_seconds,
        reserved_gpu_seconds=coverage.reserved_gpu_seconds,
        allocated_billed_gpu_seconds=coverage.allocated_billed_gpu_seconds,
        observed_whole_instance_billed_gpu_seconds=billed_observed,
        observed_wall_time_seconds=wall_observed,
        progress_export_manifest=progress.manifest,
        progress_export_files=progress.files,
        final_archive_id=archive.archive_id,
        final_archive_manifest_sha256=archive.manifest_sha256,
        final_archive_content_tree_sha256=archive.content_tree_sha256,
        final_archive_local_root=archive.local_root,
        shutdown_probe=probe,
    )
    if existing is not None:
        if existing != receipt:
            raise FormalExperimentFinalAuditError(
                "existing pre-shutdown audit differs from replay"
            )
        return existing
    try:
        publish_canonical_json_no_replace(destination, receipt.to_dict())
    except RuntimeError:
        raced = load_pre_shutdown_audit(destination)
        if raced != receipt:
            raise FormalExperimentFinalAuditError(
                "pre-shutdown audit output became occupied by another receipt"
            )
    reopened = load_pre_shutdown_audit(destination)
    if reopened != receipt:
        raise AssertionError("published pre-shutdown audit changed")
    return receipt


def load_pre_shutdown_audit(
    path: str | Path,
) -> FormalExperimentPreShutdownAuditReceipt:
    return FormalExperimentPreShutdownAuditReceipt.from_dict(
        _load_canonical_object(path, "pre-shutdown audit")
    )


@dataclass(frozen=True)
class _PowerOffAudit:
    transition_binding: FinalAuditArtifactBinding
    receipt: AutoDlPowerTransitionReceipt
    receipt_sha256: str
    sample: ProviderRuntimeSample


def _audit_existing_power_off_transition(
    *,
    path: str | Path,
    pre: FormalExperimentPreShutdownAuditReceipt,
) -> _PowerOffAudit:
    value = _load_canonical_object(path, "power-off transition evidence")
    if (
        set(value)
        != {
            "schema_version",
            "kind",
            "request_journal_path",
            "request_journal_sha256",
            "receipt",
            "receipt_sha256",
            "confirmation_history",
            "final_provider_evidence",
            "safety_probe",
        }
        or value.get("schema_version") != 1
        or value.get("kind") != ("autodl_power_transition_evidence")
    ):
        raise FormalExperimentFinalAuditError(
            "power-off transition evidence fields differ"
        )
    try:
        receipt = AutoDlPowerTransitionReceipt(**value["receipt"])
    except (TypeError, ValueError, AutoDlProviderRuntimeError) as error:
        raise FormalExperimentFinalAuditError(
            "power-off transition receipt differs"
        ) from error
    receipt_sha = _file_semantic_sha256(asdict(receipt))
    if (
        receipt.operation != "power_off"
        or receipt.target_state != "shutdown"
        or receipt.instance_uuid != pre.instance_uuid
        or receipt.safety_probe_path != pre.shutdown_probe.absolute_path
        or receipt.safety_probe_sha256 != pre.shutdown_probe.sha256
        or value["receipt_sha256"] != receipt_sha
    ):
        raise FormalExperimentFinalAuditError(
            "power-off transition receipt has foreign lineage"
        )
    pre.shutdown_probe.reopen(label="power-off shutdown probe")
    if value["safety_probe"] != _load_canonical_object(
        pre.shutdown_probe.absolute_path, "power-off shutdown probe"
    ):
        raise FormalExperimentFinalAuditError(
            "power-off transition embeds another shutdown probe"
        )

    journal_path = _absolute_path(
        value["request_journal_path"], "power mutation request journal"
    )
    journal_binding = FinalAuditArtifactBinding.bind(
        journal_path, label="power mutation request journal"
    )
    if journal_binding.sha256 != value["request_journal_sha256"]:
        raise FormalExperimentFinalAuditError(
            "power mutation request journal digest differs"
        )
    journal = _load_canonical_object(journal_path, "power mutation request journal")
    journal_unsigned = dict(journal)
    journal_sha = journal_unsigned.pop("journal_sha256", None)
    response = journal.get("redacted_provider_response")
    if (
        set(journal)
        != {
            "schema_version",
            "kind",
            "operation",
            "instance_uuid",
            "provider_request_id",
            "requested_at_ns",
            "redacted_provider_response",
            "safety_probe_sha256",
            "journal_sha256",
        }
        or journal.get("schema_version") != 1
        or journal.get("kind") != "autodl_power_mutation_request_journal"
        or journal.get("operation") != "power_off"
        or journal.get("instance_uuid") != pre.instance_uuid
        or journal.get("provider_request_id") != receipt.provider_request_id
        or journal.get("requested_at_ns") != receipt.requested_at_ns
        or journal.get("safety_probe_sha256") != pre.shutdown_probe.sha256
        or journal_sha != _file_semantic_sha256(journal_unsigned)
        or type(response) is not dict
        or response.get("code") != "Success"
        or response.get("request_id") != receipt.provider_request_id
    ):
        raise FormalExperimentFinalAuditError(
            "power mutation journal does not prove code Success"
        )

    provider = value["final_provider_evidence"]
    if type(provider) is not dict or set(provider) != {
        "schema_version",
        "kind",
        "instance_uuid",
        "observed_at_ns",
        "redacted_raw_responses",
        "response_sha256",
        "sample",
        "sample_id",
    }:
        raise FormalExperimentFinalAuditError("final provider evidence fields differ")
    raw = provider["redacted_raw_responses"]
    if type(raw) is not dict or set(raw) != {"status_response", "list_responses"}:
        raise FormalExperimentFinalAuditError(
            "final provider status/list evidence differs"
        )
    if (
        type(raw["status_response"]) is not dict
        or type(raw["list_responses"]) is not list
    ):
        raise FormalExperimentFinalAuditError(
            "final provider status/list evidence is malformed"
        )
    try:
        sample, rebuilt_provider = reduce_autodl_provider_responses(
            instance_uuid=pre.instance_uuid,
            observed_at_ns=int(provider["observed_at_ns"]),
            status_response=AutoDlApiResponse(200, raw["status_response"]),
            list_response=tuple(
                AutoDlApiResponse(200, row) for row in raw["list_responses"]
            ),
        )
    except (TypeError, ValueError, AutoDlProviderRuntimeError) as error:
        raise FormalExperimentFinalAuditError(
            "AutoDL status/list did not independently prove shutdown"
        ) from error
    if (
        sample.state != "shutdown"
        or rebuilt_provider != provider
        or sample.sample_id != receipt.provider_sample_id
        or sample.response_sha256 != receipt.provider_response_sha256
        or sample.observed_at_ns != receipt.confirmed_at_ns
    ):
        raise FormalExperimentFinalAuditError(
            "power-off final provider sample differs from its receipt"
        )
    history = value["confirmation_history"]
    if (
        type(history) is not list
        or len(history) != receipt.confirmation_attempt_count
        or not history
        or history[-1].get("state") != "shutdown"
        or history[-1].get("response_sha256") != sample.response_sha256
        or tuple(row.get("attempt") for row in history)
        != tuple(range(1, len(history) + 1))
    ):
        raise FormalExperimentFinalAuditError("power-off confirmation history differs")
    return _PowerOffAudit(
        transition_binding=FinalAuditArtifactBinding.bind(
            path, label="power-off transition evidence"
        ),
        receipt=receipt,
        receipt_sha256=receipt_sha,
        sample=sample,
    )


def _reopen_pre_shutdown_components(
    receipt: FormalExperimentPreShutdownAuditReceipt,
) -> None:
    receipt.progress_export_manifest.reopen(label="progress export manifest")
    receipt.shutdown_probe.reopen(label="shutdown probe")
    manifest = _load_canonical_object(
        receipt.progress_export_manifest.absolute_path,
        "progress export manifest",
    )
    if (
        manifest.get("run_id") != receipt.run_id
        or tuple(sorted(dict(manifest.get("files", {})).items()))
        != receipt.progress_export_files
    ):
        raise FormalExperimentFinalAuditError(
            "pre-shutdown progress export binding changed"
        )
    for name, digest in receipt.progress_export_files:
        path = Path(receipt.progress_export_manifest.absolute_path).parent / name
        actual, _size = _stable_file_sha256(path, f"progress export {name}")
        if actual != digest:
            raise FormalExperimentFinalAuditError(
                "pre-shutdown progress export file changed"
            )
    tree, _rows = _replay_archive_content_tree(
        receipt.final_archive_local_root,
        expected_manifest_sha256=receipt.final_archive_manifest_sha256,
    )
    if tree != receipt.final_archive_content_tree_sha256:
        raise FormalExperimentFinalAuditError(
            "pre-shutdown final archive content tree changed"
        )


def publish_final_completion(
    *,
    store: ExperimentOperatorStore,
    pre_shutdown_audit_path: str | Path,
    power_transition_evidence_path: str | Path,
    output_path: str | Path,
    finalized_at_ns: int | None = None,
) -> FormalExperimentFinalCompletionReceipt:
    """Publish final completion from an existing provider power-off receipt.

    There is deliberately no token, client, network, or mutation callback in
    this API.  Call ``transition_autodl_instance_power`` first, then pass its
    immutable evidence path here.
    """

    if type(store) is not ExperimentOperatorStore:
        raise TypeError("final completion requires an exact operator store")
    destination = _absolute_path(output_path, "final completion output")
    existing = None
    if destination.exists() or destination.is_symlink():
        existing = load_final_completion(destination)
        if existing.run_id != store.run_id:
            raise FormalExperimentFinalAuditError(
                "existing final completion belongs to another run"
            )
    pre = load_pre_shutdown_audit(pre_shutdown_audit_path)
    if pre.run_id != store.run_id:
        raise FormalExperimentFinalAuditError(
            "pre-shutdown audit belongs to another run"
        )
    _reopen_pre_shutdown_components(pre)
    power = _audit_existing_power_off_transition(
        path=power_transition_evidence_path,
        pre=pre,
    )
    now = int(
        existing.finalized_at_ns
        if finalized_at_ns is None and existing is not None
        else time.time_ns()
        if finalized_at_ns is None
        else finalized_at_ns
    )
    if now < power.receipt.confirmed_at_ns:
        raise FormalExperimentFinalAuditError(
            "final completion time precedes provider shutdown"
        )
    snapshot = store.snapshot()
    if any(row["state"] != "REDUCED" for row in snapshot["controller_nodes"]):
        raise FormalExperimentFinalAuditError(
            "controller left REDUCED before final completion"
        )
    intervals = tuple(snapshot["provider_billing_intervals"])
    if (
        not intervals
        or any(row["instance_uuid"] != pre.instance_uuid for row in intervals)
        or any(not bool(row["complete"]) for row in intervals)
        or power.sample.response_sha256
        not in {digest for row in intervals for digest in row["response_sha256s"]}
    ):
        raise FormalExperimentFinalAuditError(
            "provider billing interval is open or has foreign shutdown evidence"
        )
    projection = _read_database_projection(store)
    matching_samples = tuple(
        row
        for row in projection["provider_samples"]
        if row["sample_id"] == power.sample.sample_id
        and row["state"] == "shutdown"
        and row["instance_uuid"] == pre.instance_uuid
    )
    if len(matching_samples) != 1:
        raise FormalExperimentFinalAuditError(
            "provider shutdown sample is not durably recorded once"
        )
    earliest = min(int(row["provider_started_at_ns"]) for row in intervals)
    latest = max(int(row["provider_stopped_or_observed_at_ns"]) for row in intervals)
    powered_seconds = sum(float(row["duration_seconds"]) for row in intervals)
    billed_seconds = store.whole_instance_billed_gpu_seconds(require_complete=True)
    pre_binding = FinalAuditArtifactBinding.bind(
        pre_shutdown_audit_path, label="pre-shutdown audit"
    )
    receipt = FormalExperimentFinalCompletionReceipt(
        schema_version=1,
        kind="formal_experiment_final_completion",
        protocol_sha256=FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256,
        status="COMPLETE_TRUSTED_SINGLE_OPERATOR_EMPIRICAL",
        trust=TRUSTED_SINGLE_OPERATOR_EMPIRICAL,
        formal_measured=False,
        run_id=pre.run_id,
        instance_uuid=pre.instance_uuid,
        finalized_at_ns=now,
        pre_shutdown_audit=pre_binding,
        power_transition_evidence=power.transition_binding,
        power_transition_receipt_sha256=power.receipt_sha256,
        provider_request_id=power.receipt.provider_request_id,
        provider_sample_id=power.sample.sample_id,
        provider_response_sha256=power.sample.response_sha256,
        shutdown_probe_sha256=pre.shutdown_probe.sha256,
        coverage_sha256=pre.coverage_sha256,
        selection_sha256=pre.selection_sha256,
        metrics_sha256=pre.metrics_sha256,
        archive_manifest_sha256=pre.final_archive_manifest_sha256,
        archive_content_tree_sha256=pre.final_archive_content_tree_sha256,
        progress_export_manifest_sha256=pre.progress_export_manifest.sha256,
        wall_time_seconds=(latest - earliest) / 1e9,
        powered_wall_time_seconds=powered_seconds,
        compute_gpu_hours=pre.compute_gpu_seconds / 3600.0,
        reserved_gpu_hours=pre.reserved_gpu_seconds / 3600.0,
        billed_gpu_hours=billed_seconds / 3600.0,
    )
    if existing is not None:
        if existing != receipt:
            raise FormalExperimentFinalAuditError(
                "existing final completion differs from replay"
            )
        return existing
    try:
        publish_canonical_json_no_replace(destination, receipt.to_dict())
    except RuntimeError:
        raced = load_final_completion(destination)
        if raced != receipt:
            raise FormalExperimentFinalAuditError(
                "final completion output became occupied by another receipt"
            )
    reopened = load_final_completion(destination)
    if reopened != receipt:
        raise AssertionError("published final completion changed")
    return receipt


def load_final_completion(
    path: str | Path,
) -> FormalExperimentFinalCompletionReceipt:
    return FormalExperimentFinalCompletionReceipt.from_dict(
        _load_canonical_object(path, "final completion receipt")
    )


__all__ = [
    "FINAL_ARCHIVE_SAFE_BOUNDARY",
    "FORMAL_EXPERIMENT_FINAL_AUDIT_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_EMPIRICAL",
    "FinalAuditArtifactBinding",
    "FormalExperimentFinalAuditError",
    "FormalExperimentFinalCompletionReceipt",
    "FormalExperimentFinalizationReadiness",
    "FormalExperimentPreShutdownAuditReceipt",
    "audit_finalization_readiness",
    "load_final_completion",
    "load_pre_shutdown_audit",
    "publish_final_completion",
    "publish_pre_shutdown_audit",
]
